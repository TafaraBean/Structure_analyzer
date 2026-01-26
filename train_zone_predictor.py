import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt

load_dotenv()

class ZonePredictor:
    """Neural network to predict next supply/demand zone location."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_zone_labels(self, filename='zone_labels_for_ml.csv'):
        """Load the labeled data from zone detection."""
        print(f"📊 Loading zone labels from {filename}...")
        
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   Near supply: {df['is_near_supply'].sum()} ({df['is_near_supply'].mean()*100:.1f}%)")
        print(f"   Near demand: {df['is_near_demand'].sum()} ({df['is_near_demand'].mean()*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """Create features for zone prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        
        # Moving averages and distance
        for period in [5, 10, 20]:
            ma = talib.SMA(df['close'], timeperiod=period)
            features[f'price_to_ma_{period}'] = (df['close'] - ma) / ma
            features[f'ma_{period}_slope'] = ma.pct_change()
        
        # Volatility
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['atr_normalized'] = features['atr_14'] / df['close']
        features['volatility_20'] = df['close'].rolling(20).std() / df['close']
        
        # Bollinger Bands position
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        features['bb_width'] = (upper - lower) / middle
        
        # RSI
        features['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        features['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        
        # ADX (trend strength)
        features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Price position in recent range
        for period in [10, 20, 50]:
            high_range = df['high'].rolling(period).max()
            low_range = df['low'].rolling(period).min()
            features[f'range_position_{period}'] = (df['close'] - low_range) / (high_range - low_range)
        
        # Volume
        features['volume_ma_ratio'] = df['tick_volume'] / talib.SMA(df['tick_volume'], timeperiod=10)
        
        # Statistical features
        for period in [10, 20]:
            features[f'skew_{period}'] = df['close'].rolling(period).skew()
            features[f'kurt_{period}'] = df['close'].rolling(period).kurt()
        
        # Lag features
        for lag in [1, 2, 3]:
            features[f'close_lag_{lag}'] = df['close'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        
        return features
    
    def prepare_data(self, df):
        """Prepare features and targets for training."""
        print(f"\n🔧 Preparing training data...")
        
        # Create features
        features = self.create_features(df)
        
        # Targets: predict if NEXT candle will be near a zone
        # Shift labels backward to predict future
        target_supply = df['is_near_supply'].shift(-1)
        target_demand = df['is_near_demand'].shift(-1)
        
        # Also predict distance to next zone (regression task)
        target_dist_supply = df['distance_to_nearest_supply'].shift(-1)
        target_dist_demand = df['distance_to_nearest_demand'].shift(-1)
        
        # Combine
        data = pd.concat([
            features,
            target_supply.rename('next_near_supply'),
            target_demand.rename('next_near_demand'),
            target_dist_supply.rename('next_dist_supply'),
            target_dist_demand.rename('next_dist_demand')
        ], axis=1)
        
        # Drop NaN
        data = data.dropna()
        
        # Replace inf with large number
        data = data.replace([np.inf, -np.inf], 10.0)
        
        print(f"✅ Data prepared: {len(data)} samples")
        print(f"   Next near supply: {data['next_near_supply'].sum()} ({data['next_near_supply'].mean()*100:.1f}%)")
        print(f"   Next near demand: {data['next_near_demand'].sum()} ({data['next_near_demand'].mean()*100:.1f}%)")
        
        return data
    
    def build_model(self, input_dim):
        """Build multi-output neural network."""
        # Input
        inputs = layers.Input(shape=(input_dim,))
        
        # Shared layers
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output branches
        # Classification: Is next candle near a zone?
        supply_class = layers.Dense(16, activation='relu')(x)
        supply_class = layers.Dense(1, activation='sigmoid', name='supply_class')(supply_class)
        
        demand_class = layers.Dense(16, activation='relu')(x)
        demand_class = layers.Dense(1, activation='sigmoid', name='demand_class')(demand_class)
        
        # Regression: Distance to next zone
        supply_dist = layers.Dense(16, activation='relu')(x)
        supply_dist = layers.Dense(1, activation='linear', name='supply_dist')(supply_dist)
        
        demand_dist = layers.Dense(16, activation='relu')(x)
        demand_dist = layers.Dense(1, activation='linear', name='demand_dist')(demand_dist)
        
        # Create model
        model = keras.Model(
            inputs=inputs,
            outputs=[supply_class, demand_class, supply_dist, demand_dist]
        )
        
        # Compile with multiple losses
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'supply_class': 'binary_crossentropy',
                'demand_class': 'binary_crossentropy',
                'supply_dist': 'mse',
                'demand_dist': 'mse'
            },
            loss_weights={
                'supply_class': 1.0,
                'demand_class': 1.0,
                'supply_dist': 0.5,
                'demand_dist': 0.5
            },
            metrics={
                'supply_class': ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
                'demand_class': ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
                'supply_dist': ['mae'],
                'demand_dist': ['mae']
            }
        )
        
        return model
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """Train the multi-output model."""
        print(f"\n🎯 Training zone predictor...")
        
        # Separate features and targets
        feature_cols = [c for c in data.columns if not c.startswith('next_')]
        X = data[feature_cols]
        
        y_supply_class = data['next_near_supply']
        y_demand_class = data['next_near_demand']
        y_supply_dist = data['next_dist_supply']
        y_demand_dist = data['next_dist_demand']
        
        # Time-series split
        split_idx = int(len(X) * (1 - validation_split))
        
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_sc_train, y_sc_val = y_supply_class.iloc[:split_idx], y_supply_class.iloc[split_idx:]
        y_dc_train, y_dc_val = y_demand_class.iloc[:split_idx], y_demand_class.iloc[split_idx:]
        y_sd_train, y_sd_val = y_supply_dist.iloc[:split_idx], y_supply_dist.iloc[split_idx:]
        y_dd_train, y_dd_val = y_demand_dist.iloc[:split_idx], y_demand_dist.iloc[split_idx:]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        print(f"\n   Model architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            X_train_scaled,
            {
                'supply_class': y_sc_train,
                'demand_class': y_dc_train,
                'supply_dist': y_sd_train,
                'demand_dist': y_dd_train
            },
            validation_data=(
                X_val_scaled,
                {
                    'supply_class': y_sc_val,
                    'demand_class': y_dc_val,
                    'supply_dist': y_sd_val,
                    'demand_dist': y_dd_val
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history, X_val_scaled, (y_sc_val, y_dc_val, y_sd_val, y_dd_val)
    
    def evaluate(self, X_val, y_vals):
        """Evaluate model predictions."""
        print(f"\n📊 Evaluating model...")
        
        y_sc_val, y_dc_val, y_sd_val, y_dd_val = y_vals
        
        # Predictions
        preds = self.model.predict(X_val)
        supply_class_pred, demand_class_pred, supply_dist_pred, demand_dist_pred = preds
        
        # Classification metrics
        supply_class_binary = (supply_class_pred > 0.5).astype(int).flatten()
        demand_class_binary = (demand_class_pred > 0.5).astype(int).flatten()
        
        print(f"\n{'='*60}")
        print("  SUPPLY ZONE PREDICTION")
        print(f"{'='*60}")
        print(f"Accuracy: {(supply_class_binary == y_sc_val).mean():.2%}")
        print(f"Predicted positives: {supply_class_binary.sum()} ({supply_class_binary.mean()*100:.1f}%)")
        print(f"Actual positives: {y_sc_val.sum()} ({y_sc_val.mean()*100:.1f}%)")
        
        print(f"\n{'='*60}")
        print("  DEMAND ZONE PREDICTION")
        print(f"{'='*60}")
        print(f"Accuracy: {(demand_class_binary == y_dc_val).mean():.2%}")
        print(f"Predicted positives: {demand_class_binary.sum()} ({demand_class_binary.mean()*100:.1f}%)")
        print(f"Actual positives: {y_dc_val.sum()} ({y_dc_val.mean()*100:.1f}%)")
        
        # Distance prediction metrics
        supply_dist_mae = mean_absolute_error(y_sd_val, supply_dist_pred)
        demand_dist_mae = mean_absolute_error(y_dd_val, demand_dist_pred)
        
        print(f"\n{'='*60}")
        print("  DISTANCE PREDICTION")
        print(f"{'='*60}")
        print(f"Supply distance MAE: {supply_dist_mae:.4f}")
        print(f"Demand distance MAE: {demand_dist_mae:.4f}")
        
        return preds
    
    def plot_results(self, history):
        """Plot training history."""
        print(f"\n📈 Creating visualizations...")
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Supply classification loss
        ax = axes[0, 0]
        ax.plot(history.history['supply_class_loss'], label='Train', color='cyan')
        ax.plot(history.history['val_supply_class_loss'], label='Val', color='orange')
        ax.set_title('Supply Zone Classification Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        # Demand classification loss
        ax = axes[0, 1]
        ax.plot(history.history['demand_class_loss'], label='Train', color='cyan')
        ax.plot(history.history['val_demand_class_loss'], label='Val', color='orange')
        ax.set_title('Demand Zone Classification Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        # Supply distance loss
        ax = axes[1, 0]
        ax.plot(history.history['supply_dist_loss'], label='Train', color='cyan')
        ax.plot(history.history['val_supply_dist_loss'], label='Val', color='orange')
        ax.set_title('Supply Distance Prediction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        # Demand distance loss
        ax = axes[1, 1]
        ax.plot(history.history['demand_dist_loss'], label='Train', color='cyan')
        ax.plot(history.history['val_demand_dist_loss'], label='Val', color='orange')
        ax.set_title('Demand Distance Prediction Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        filename = 'zone_predictor_training.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Training curves saved as {filename}")
        
        plt.show()
    
    def visualize_predictions(self, df, X_val, y_vals, preds, sample_bars=300):
        """Visualize predictions on price chart."""
        print(f"\n📊 Creating prediction visualization...")
        
        y_sc_val, y_dc_val, y_sd_val, y_dd_val = y_vals
        supply_class_pred, demand_class_pred, supply_dist_pred, demand_dist_pred = preds
        
        # Get validation data indices
        val_start_idx = len(df) - len(X_val)
        df_val = df.iloc[val_start_idx:]
        
        # Use last N bars
        plot_start = max(0, len(df_val) - sample_bars)
        df_plot = df_val.iloc[plot_start:]
        
        # Adjust predictions to match plot
        supply_pred_plot = supply_class_pred[plot_start:].flatten()
        demand_pred_plot = demand_class_pred[plot_start:].flatten()
        y_sc_plot = y_sc_val.iloc[plot_start:].values
        y_dc_plot = y_dc_val.iloc[plot_start:].values
        
        plt.style.use('dark_background')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1, 1]})
        
        fig.suptitle('Neural Network Zone Predictions vs Actual Zones', 
                     fontsize=14, color='white')
        
        # Panel 1: Price with actual zones
        ax1.plot(df_plot.index, df_plot['close'], color='white', linewidth=1, label='Close Price')
        
        # Highlight actual zones
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if i >= len(y_sc_plot):
                break
            
            # Actual supply zones (ground truth)
            if y_sc_plot[i] == 1:
                ax1.axvspan(idx, idx, color='red', alpha=0.3)
            
            # Actual demand zones (ground truth)
            if y_dc_plot[i] == 1:
                ax1.axvspan(idx, idx, color='green', alpha=0.3)
        
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.set_title('Price Chart with Actual Zones (Shaded)', fontsize=11, pad=10)
        
        # Panel 2: Supply zone predictions
        ax2.plot(df_plot.index, supply_pred_plot, color='red', linewidth=2, label='Predicted Probability')
        ax2.axhline(y=0.5, color='yellow', linestyle='--', linewidth=1, label='Threshold')
        ax2.fill_between(df_plot.index, 0, supply_pred_plot, 
                        where=(supply_pred_plot > 0.5), color='red', alpha=0.3)
        
        # Mark actual supply zones
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if i >= len(y_sc_plot):
                break
            if y_sc_plot[i] == 1:
                ax2.scatter(idx, 1.0, marker='v', color='yellow', s=50, zorder=5)
        
        ax2.set_ylabel('Probability', fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.set_title('Supply Zone Predictions (Yellow markers = Actual)', fontsize=11, pad=10)
        
        # Panel 3: Demand zone predictions
        ax3.plot(df_plot.index, demand_pred_plot, color='green', linewidth=2, label='Predicted Probability')
        ax3.axhline(y=0.5, color='yellow', linestyle='--', linewidth=1, label='Threshold')
        ax3.fill_between(df_plot.index, 0, demand_pred_plot,
                        where=(demand_pred_plot > 0.5), color='green', alpha=0.3)
        
        # Mark actual demand zones
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if i >= len(y_dc_plot):
                break
            if y_dc_plot[i] == 1:
                ax3.scatter(idx, 1.0, marker='^', color='yellow', s=50, zorder=5)
        
        ax3.set_ylabel('Probability', fontsize=10)
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.2)
        ax3.set_title('Demand Zone Predictions (Yellow markers = Actual)', fontsize=11, pad=10)
        
        plt.tight_layout()
        
        filename = 'zone_predictions_chart.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Prediction chart saved as {filename}")
        
        plt.show()
        print(f"📈 Visualization displayed")
    
    def save_model(self, filename='zone_predictor_model.h5'):
        """Save trained model."""
        self.model.save(filename)
        print(f"\n💾 Model saved as {filename}")

def main():
    print("="*60)
    print("  SUPPLY/DEMAND ZONE PREDICTOR - TRAINING")
    print("="*60)
    
    predictor = ZonePredictor()
    
    # Load labeled data
    df = predictor.load_zone_labels('zone_labels_for_ml.csv')
    
    # Prepare data
    data = predictor.prepare_data(df)
    
    # Train
    history, X_val, y_vals = predictor.train(data, epochs=100, batch_size=32)
    
    # Evaluate
    preds = predictor.evaluate(X_val, y_vals)
    
    # Plot training curves
    predictor.plot_results(history)
    
    # Visualize predictions on chart
    predictor.visualize_predictions(df, X_val, y_vals, preds, sample_bars=300)
    
    # Save
    predictor.save_model()
    
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\n✅ Model can now predict:")
    print(f"   1. If next candle will be near a supply zone")
    print(f"   2. If next candle will be near a demand zone")
    print(f"   3. Distance to nearest supply zone")
    print(f"   4. Distance to nearest demand zone")
    print(f"\n💡 Use this model during sideways regimes to find reversal entries!")
    
    print(f"\n👋 Complete")

if __name__ == "__main__":
    main()
