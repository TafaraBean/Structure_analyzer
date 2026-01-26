import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt

load_dotenv()

# Focal Loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.
    Focuses training on hard examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        cross_entropy = -y_true * keras.backend.log(y_pred)
        loss = alpha * keras.backend.pow(1 - y_pred, gamma) * cross_entropy
        
        return keras.backend.mean(loss)
    
    return focal_loss_fixed

class ZoneFormationPredictor:
    """Predict when NEW supply/demand zones will FORM (not just proximity)."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_zone_data(self, filename='supply_demand_zones.csv'):
        """Load detected zones."""
        print(f"📊 Loading zone data from {filename}...")
        
        zones_df = pd.read_csv(filename)
        
        print(f"✅ Loaded {len(zones_df)} zones")
        print(f"   Supply: {(zones_df['type'] == 'supply').sum()}")
        print(f"   Demand: {(zones_df['type'] == 'demand').sum()}")
        
        return zones_df
    
    def load_price_data(self, filename='zone_labels_for_ml.csv'):
        """Load price data."""
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"✅ Loaded {len(df)} price bars")
        return df
    
    def create_zone_formation_labels(self, df, zones_df, lookforward=10):
        """
        Create labels for ZONE FORMATION events.
        
        Label = 1 if a new zone will form within next N candles
        Label = 0 otherwise
        
        This is much more balanced and useful!
        """
        print(f"\n🏷️  Creating zone formation labels (lookforward={lookforward})...")
        
        # Convert zone times to datetime
        zones_df['time'] = pd.to_datetime(zones_df['time'])
        
        # Initialize labels
        df['will_form_supply'] = 0
        df['will_form_demand'] = 0
        df['zone_strength'] = 0.0
        
        # For each zone, mark the candles BEFORE it formed
        for _, zone in zones_df.iterrows():
            zone_time = zone['time']
            zone_type = zone['type']
            zone_strength = zone['strength']
            
            # Find index in df
            if zone_time not in df.index:
                continue
            
            zone_idx = df.index.get_loc(zone_time)
            
            # Mark previous N candles as "will form zone"
            start_idx = max(0, zone_idx - lookforward)
            
            if zone_type == 'supply':
                df.iloc[start_idx:zone_idx, df.columns.get_loc('will_form_supply')] = 1
                df.iloc[start_idx:zone_idx, df.columns.get_loc('zone_strength')] = zone_strength
            else:  # demand
                df.iloc[start_idx:zone_idx, df.columns.get_loc('will_form_demand')] = 1
                df.iloc[start_idx:zone_idx, df.columns.get_loc('zone_strength')] = zone_strength
        
        supply_pct = df['will_form_supply'].mean() * 100
        demand_pct = df['will_form_demand'].mean() * 100
        
        print(f"✅ Labels created:")
        print(f"   Will form supply: {df['will_form_supply'].sum()} ({supply_pct:.1f}%)")
        print(f"   Will form demand: {df['will_form_demand'].sum()} ({demand_pct:.1f}%)")
        print(f"   ✅ Much better balance!")
        
        return df
    
    def create_features(self, df):
        """Create features for prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Price momentum
        features['returns'] = df['close'].pct_change()
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        
        # Volatility
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['atr_normalized'] = features['atr_14'] / df['close']
        features['volatility_10'] = df['close'].rolling(10).std() / df['close']
        features['volatility_20'] = df['close'].rolling(20).std() / df['close']
        
        # Price position in range
        for period in [10, 20, 50]:
            high_range = df['high'].rolling(period).max()
            low_range = df['low'].rolling(period).min()
            features[f'range_position_{period}'] = (df['close'] - low_range) / (high_range - low_range)
            features[f'range_width_{period}'] = (high_range - low_range) / df['close']
        
        # RSI extremes (zones often form at extremes)
        features['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        features['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        features['rsi_extreme'] = ((features['rsi_14'] < 30) | (features['rsi_14'] > 70)).astype(int)
        
        # Trend strength
        features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        features['bb_width'] = (upper - lower) / middle
        features['touching_bb_upper'] = (df['high'] >= upper).astype(int)
        features['touching_bb_lower'] = (df['low'] <= lower).astype(int)
        
        # Volume
        features['volume_ma_ratio'] = df['tick_volume'] / talib.SMA(df['tick_volume'], timeperiod=10)
        
        # Candle patterns
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Swing detection
        features['is_swing_high'] = ((df['high'] > df['high'].shift(1)) & 
                                     (df['high'] > df['high'].shift(-1))).astype(int)
        features['is_swing_low'] = ((df['low'] < df['low'].shift(1)) & 
                                    (df['low'] < df['low'].shift(-1))).astype(int)
        
        return features
    
    def prepare_data(self, df):
        """Prepare features and targets."""
        print(f"\n🔧 Preparing training data...")
        
        features = self.create_features(df)
        
        # Targets: predict if zone will form in FUTURE
        target_supply = df['will_form_supply'].shift(-1)
        target_demand = df['will_form_demand'].shift(-1)
        
        # Combine
        data = pd.concat([
            features,
            target_supply.rename('target_supply'),
            target_demand.rename('target_demand')
        ], axis=1)
        
        data = data.dropna()
        
        print(f"✅ Data prepared: {len(data)} samples")
        print(f"   Future supply zones: {data['target_supply'].sum()} ({data['target_supply'].mean()*100:.1f}%)")
        print(f"   Future demand zones: {data['target_demand'].sum()} ({data['target_demand'].mean()*100:.1f}%)")
        
        return data
    
    def build_model(self, input_dim):
        """Build classification model."""
        inputs = layers.Input(shape=(input_dim,))
        
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Outputs
        supply_out = layers.Dense(1, activation='sigmoid', name='supply')(x)
        demand_out = layers.Dense(1, activation='sigmoid', name='demand')(x)
        
        model = keras.Model(inputs=inputs, outputs=[supply_out, demand_out])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={'supply': focal_loss(gamma=2.0, alpha=0.75), 
                  'demand': focal_loss(gamma=2.0, alpha=0.75)},
            metrics={
                'supply': ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
                'demand': ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
            }
        )
        
        return model
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """Train model."""
        print(f"\n🎯 Training zone formation predictor...")
        
        feature_cols = [c for c in data.columns if not c.startswith('target_')]
        X = data[feature_cols]
        y_supply = data['target_supply']
        y_demand = data['target_demand']
        
        # Time-series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_s_train, y_s_val = y_supply.iloc[:split_idx], y_supply.iloc[split_idx:]
        y_d_train, y_d_val = y_demand.iloc[:split_idx], y_demand.iloc[split_idx:]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Class weights
        supply_weight = len(y_s_train) / (2 * y_s_train.sum()) if y_s_train.sum() > 0 else 1
        demand_weight = len(y_d_train) / (2 * y_d_train.sum()) if y_d_train.sum() > 0 else 1
        
        class_weight_supply = {0: 1.0, 1: supply_weight}
        class_weight_demand = {0: 1.0, 1: demand_weight}
        
        print(f"\n   Class weights:")
        print(f"   Supply: {class_weight_supply}")
        print(f"   Demand: {class_weight_demand}")
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Train
        history = self.model.fit(
            X_train_scaled,
            {'supply': y_s_train, 'demand': y_d_train},
            validation_data=(X_val_scaled, {'supply': y_s_val, 'demand': y_d_val}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history, X_val_scaled, (y_s_val, y_d_val)
    
    def evaluate(self, X_val, y_vals):
        """Evaluate predictions."""
        print(f"\n📊 Evaluating model...")
        
        y_s_val, y_d_val = y_vals
        
        preds = self.model.predict(X_val)
        supply_pred, demand_pred = preds
        
        supply_binary = (supply_pred > 0.5).astype(int).flatten()
        demand_binary = (demand_pred > 0.5).astype(int).flatten()
        
        print(f"\n{'='*60}")
        print("  SUPPLY ZONE FORMATION PREDICTION")
        print(f"{'='*60}")
        print(classification_report(y_s_val, supply_binary, target_names=['No Zone', 'Zone Forming']))
        
        print(f"\n{'='*60}")
        print("  DEMAND ZONE FORMATION PREDICTION")
        print(f"{'='*60}")
        print(classification_report(y_d_val, demand_binary, target_names=['No Zone', 'Zone Forming']))
        
        return preds

def main():
    print("="*60)
    print("  ZONE FORMATION PREDICTOR (REFRAMED)")
    print("  Predicts: Will a NEW zone FORM soon?")
    print("="*60)
    
    predictor = ZoneFormationPredictor()
    
    # Load data
    zones_df = predictor.load_zone_data('supply_demand_zones.csv')
    df = predictor.load_price_data('zone_labels_for_ml.csv')
    
    # Create better labels
    df = predictor.create_zone_formation_labels(df, zones_df, lookforward=10)
    
    # Prepare data
    data = predictor.prepare_data(df)
    
    # Train
    history, X_val, y_vals = predictor.train(data, epochs=100, batch_size=32)
    
    # Evaluate
    predictor.evaluate(X_val, y_vals)
    
    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}")
    print(f"\n✅ This model now predicts ZONE FORMATION events")
    print(f"   Much more useful than proximity!")
    print(f"\n👋 Done")

if __name__ == "__main__":
    main()
