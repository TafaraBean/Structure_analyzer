import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import talib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ZoneEnsembleTrainer:
    """Train ensemble of specialized NNs for zone reversal prediction."""
    
    def __init__(self, labels_file='zone_labels_bayesian_for_ml.csv'):
        self.labels_file = labels_file
        self.df = None
        self.models = {}
        self.scalers = {}
        self.feature_groups = {}
        self.history = {}
        
    def load_data(self):
        """Load ML labels from CSV."""
        print(f"\n📂 Loading data from {self.labels_file}...")
        
        if not os.path.exists(self.labels_file):
            print(f"❌ File not found: {self.labels_file}")
            print(f"   Please run detect_supply_demand_zones.py first")
            return False
        
        self.df = pd.read_csv(self.labels_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(self.df)} rows")
        print(f"   Columns: {list(self.df.columns)}")
        
        return True
    
    def engineer_price_action_features(self):
        """Extract price action features."""
        print(f"\n🔧 Engineering Price Action features...")
        
        features = pd.DataFrame(index=self.df.index)
        
        # Candlestick characteristics
        features['body_size'] = abs(self.df['close'] - self.df['open']) / self.df['close']
        features['upper_wick'] = (self.df['high'] - self.df[['open', 'close']].max(axis=1)) / self.df['close']
        features['lower_wick'] = (self.df[['open', 'close']].min(axis=1) - self.df['low']) / self.df['close']
        features['body_to_range'] = features['body_size'] / ((self.df['high'] - self.df['low']) / self.df['close'] + 1e-10)
        
        # Price position
        features['close_position'] = (self.df['close'] - self.df['low']) / (self.df['high'] - self.df['low'] + 1e-10)
        
        # Recent highs/lows
        for period in [5, 10, 20]:
            features[f'dist_to_high_{period}'] = (self.df['high'].rolling(period).max() - self.df['close']) / self.df['close']
            features[f'dist_to_low_{period}'] = (self.df['close'] - self.df['low'].rolling(period).min()) / self.df['close']
        
        # Consecutive patterns
        features['consecutive_up'] = (self.df['close'] > self.df['open']).rolling(3).sum()
        features['consecutive_down'] = (self.df['close'] < self.df['open']).rolling(3).sum()
        
        # Candlestick patterns (simplified)
        features['is_doji'] = (features['body_size'] < 0.001).astype(int)
        features['is_hammer'] = ((features['lower_wick'] > 2 * features['body_size']) & 
                                (features['upper_wick'] < features['body_size'])).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['price_action'] = features
        print(f"   ✅ Created {len(features.columns)} price action features")
        
    def engineer_volume_features(self):
        """Extract volume features."""
        print(f"\n🔧 Engineering Volume features...")
        
        features = pd.DataFrame(index=self.df.index)
        
        # Volume metrics
        features['volume_ratio'] = self.df['tick_volume'] / (self.df['tick_volume'].rolling(20).mean() + 1)
        features['volume_spike'] = (self.df['tick_volume'] > 2 * self.df['tick_volume'].rolling(20).mean()).astype(int)
        
        # Volume trend
        features['volume_trend'] = self.df['tick_volume'].rolling(5).mean() / (self.df['tick_volume'].rolling(20).mean() + 1)
        
        # Volume change
        features['volume_change'] = self.df['tick_volume'].pct_change().fillna(0)
        
        # Volume momentum
        for period in [3, 5, 10]:
            features[f'volume_ma_{period}'] = self.df['tick_volume'].rolling(period).mean() / (self.df['tick_volume'].rolling(20).mean() + 1)
        
        features.fillna(0, inplace=True)
        self.feature_groups['volume'] = features
        print(f"   ✅ Created {len(features.columns)} volume features")
        
    def engineer_volatility_features(self):
        """Extract volatility features."""
        print(f"\n🔧 Engineering Volatility features...")
        
        features = pd.DataFrame(index=self.df.index)
        
        # ATR
        features['atr_14'] = talib.ATR(self.df['high'].values, self.df['low'].values, 
                                       self.df['close'].values, timeperiod=14) / self.df['close']
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.df['close'].values, timeperiod=20)
        features['bb_width'] = (upper - lower) / self.df['close']
        features['bb_position'] = (self.df['close'] - lower) / (upper - lower + 1e-10)
        
        # Historical volatility
        features['hist_vol_20'] = self.df['close'].pct_change().rolling(20).std()
        
        # Range metrics
        for period in [5, 10, 20]:
            features[f'range_{period}'] = (self.df['high'].rolling(period).max() - 
                                          self.df['low'].rolling(period).min()) / self.df['close']
        
        # Volatility regime
        features['vol_regime'] = (features['atr_14'] > features['atr_14'].rolling(50).mean()).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['volatility'] = features
        print(f"   ✅ Created {len(features.columns)} volatility features")
        
    def engineer_regime_features(self):
        """Extract regime features."""
        print(f"\n🔧 Engineering Regime features...")
        
        features = pd.DataFrame(index=self.df.index)
        
        # ADX
        features['adx'] = talib.ADX(self.df['high'].values, self.df['low'].values, 
                                    self.df['close'].values, timeperiod=14) / 100
        
        # Range consolidation
        for period in [10, 20, 50]:
            high_range = self.df['high'].rolling(period).max()
            low_range = self.df['low'].rolling(period).min()
            features[f'consolidation_{period}'] = (high_range - low_range) / self.df['close']
        
        # Trend strength
        features['ema_diff'] = (talib.EMA(self.df['close'].values, 12) - 
                               talib.EMA(self.df['close'].values, 26)) / self.df['close']
        
        # Price momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = self.df['close'].pct_change(period).fillna(0)
        
        features.fillna(0, inplace=True)
        self.feature_groups['regime'] = features
        print(f"   ✅ Created {len(features.columns)} regime features")
        
    def engineer_mtf_features(self):
        """Extract multi-timeframe features."""
        print(f"\n🔧 Engineering MTF features...")
        
        features = pd.DataFrame(index=self.df.index)
        
        # DO NOT use composite_zone_score or high_prob_zone - they're derived from the target!
        # This would be data leakage
        
        # Higher timeframe momentum (simulated with longer periods)
        features['htf_momentum_50'] = self.df['close'].pct_change(50).fillna(0)
        features['htf_momentum_100'] = self.df['close'].pct_change(100).fillna(0)
        
        # Longer-term moving averages
        features['ma_50'] = (self.df['close'] - talib.SMA(self.df['close'].values, 50)) / self.df['close']
        features['ma_100'] = (self.df['close'] - talib.SMA(self.df['close'].values, 100)) / self.df['close']
        features['ma_200'] = (self.df['close'] - talib.SMA(self.df['close'].values, 200)) / self.df['close']
        
        features.fillna(0, inplace=True)
        self.feature_groups['mtf'] = features
        print(f"   ✅ Created {len(features.columns)} MTF features")
        
    def prepare_targets(self):
        """Prepare target variables."""
        print(f"\n🎯 Preparing targets...")
        
        # Target: Is near supply or demand zone
        self.y_supply = self.df['is_near_supply'].values
        self.y_demand = self.df['is_near_demand'].values
        
        # Combined target (either supply or demand)
        self.y_combined = ((self.df['is_near_supply'] == 1) | (self.df['is_near_demand'] == 1)).astype(int).values
        
        print(f"   Supply zones: {self.y_supply.sum()} ({self.y_supply.sum()/len(self.y_supply)*100:.1f}%)")
        print(f"   Demand zones: {self.y_demand.sum()} ({self.y_demand.sum()/len(self.y_demand)*100:.1f}%)")
        print(f"   Combined: {self.y_combined.sum()} ({self.y_combined.sum()/len(self.y_combined)*100:.1f}%)")
        
    def build_specialist_model(self, input_dim, name):
        """Build a specialist neural network."""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim, name=f'{name}_dense1'),
            layers.Dropout(0.3, name=f'{name}_dropout1'),
            layers.Dense(32, activation='relu', name=f'{name}_dense2'),
            layers.Dropout(0.2, name=f'{name}_dropout2'),
            layers.Dense(1, activation='sigmoid', name=f'{name}_output')
        ], name=name)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        return model
    
    def train_specialist(self, feature_group_name, X_train, X_val, y_train, y_val):
        """Train a single specialist model."""
        print(f"\n🧠 Training {feature_group_name.upper()} Specialist...")
        
        # Check class distribution
        train_pos = y_train.sum()
        train_neg = len(y_train) - train_pos
        val_pos = y_val.sum()
        val_neg = len(y_val) - val_pos
        
        print(f"   📊 Class Distribution:")
        print(f"      Train: {train_pos} positive ({train_pos/len(y_train)*100:.2f}%), {train_neg} negative")
        print(f"      Val:   {val_pos} positive ({val_pos/len(y_val)*100:.2f}%), {val_neg} negative")
        
        # Check if validation set has any positive examples
        if val_pos == 0:
            print(f"   ⚠️  WARNING: Validation set has NO positive examples!")
            print(f"      Model will appear perfect but is useless.")
            return None
        
        # Calculate class weights to handle imbalance
        if train_pos > 0:
            class_weight = {
                0: 1.0,
                1: train_neg / train_pos  # Weight positive class higher
            }
            print(f"   ⚖️  Class weights: {{0: 1.0, 1: {class_weight[1]:.2f}}}")
        else:
            class_weight = None
            print(f"   ⚠️  WARNING: No positive examples in training set!")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers[feature_group_name] = scaler
        
        # Build model
        model = self.build_specialist_model(X_train.shape[1], feature_group_name)
        
        # Train
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=32,
            class_weight=class_weight,  # Handle class imbalance
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(X_val_scaled, y_val, verbose=0)
        
        # Get predictions to see what model is actually predicting
        val_preds = model.predict(X_val_scaled, verbose=0).flatten()
        pred_positive = (val_preds > 0.5).sum()
        
        print(f"   ✅ Validation Metrics:")
        print(f"      Accuracy:  {val_acc:.4f}")
        print(f"      Precision: {val_prec:.4f}")
        print(f"      Recall:    {val_rec:.4f}")
        print(f"      AUC:       {val_auc:.4f}")
        print(f"   🔍 Predictions: {pred_positive}/{len(val_preds)} predicted positive ({pred_positive/len(val_preds)*100:.2f}%)")
        print(f"      Mean prediction: {val_preds.mean():.4f}, Std: {val_preds.std():.4f}")
        
        # Save
        self.models[feature_group_name] = model
        self.history[feature_group_name] = history.history
        model.save(f'zone_nn_{feature_group_name}.keras')  # Use .keras format
        
        return model
    
    def train_ensemble(self):
        """Train all specialist models."""
        print(f"\n{'='*60}")
        print(f"  TRAINING ENSEMBLE OF SPECIALIST NEURAL NETWORKS")
        print(f"{'='*60}")
        
        # Engineer all features
        self.engineer_price_action_features()
        self.engineer_volume_features()
        self.engineer_volatility_features()
        self.engineer_regime_features()
        self.engineer_mtf_features()
        self.prepare_targets()
        
        # Split data
        train_size = int(0.7 * len(self.df))
        val_size = int(0.15 * len(self.df))
        
        # Train each specialist
        for group_name, features in self.feature_groups.items():
            X = features.values
            X_train = X[:train_size]
            X_val = X[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            
            y_train = self.y_combined[:train_size]
            y_val = self.y_combined[train_size:train_size+val_size]
            y_test = self.y_combined[train_size+val_size:]
            
            self.train_specialist(group_name, X_train, X_val, y_train, y_val)
        
        print(f"\n{'='*60}")
        print(f"  ✅ ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"\n📁 Saved Models:")
        for name in self.models.keys():
            print(f"   - zone_nn_{name}.h5")
        
    def evaluate_ensemble(self):
        """Evaluate ensemble performance."""
        print(f"\n📊 Evaluating Ensemble...")
        
        # TODO: Implement ensemble evaluation and meta-model
        pass


def main():
    print("="*60)
    print("  ZONE REVERSAL ENSEMBLE NEURAL NETWORK TRAINER")
    print("  Symbol: EURUSDm | Timeframe: M15")
    print("="*60)
    
    trainer = ZoneEnsembleTrainer()
    
    if not trainer.load_data():
        return
    
    trainer.train_ensemble()
    
    print(f"\n👋 Training Complete!")


if __name__ == "__main__":
    main()
