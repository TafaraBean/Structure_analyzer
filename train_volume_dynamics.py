import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VolumeDynamicsNN:
    """Neural network analyzing volume velocity and acceleration for reversal prediction."""
    
    def __init__(self, labels_file='zone_labels_bayesian_for_ml.csv'):
        self.labels_file = labels_file
        self.df = None
        self.model = None
        self.scaler = None
        self.features = None
        
    def load_data(self):
        """Load ML labels from CSV."""
        print(f"\n📂 Loading data from {self.labels_file}...")
        
        if not os.path.exists(self.labels_file):
            print(f"❌ File not found: {self.labels_file}")
            return False
        
        self.df = pd.read_csv(self.labels_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(self.df)} rows")
        return True
    
    def engineer_volume_dynamics_features(self):
        """Extract volume velocity and acceleration features."""
        print(f"\n🔧 Engineering Volume Dynamics features...")
        
        features = pd.DataFrame(index=self.df.index)
        volume = self.df['tick_volume']
        
        # === POSITION (Volume itself) ===
        # Rolling volume statistics
        for window in [3, 5, 10, 20]:
            features[f'volume_ma_{window}'] = volume.rolling(window).mean()
            features[f'volume_std_{window}'] = volume.rolling(window).std()
            features[f'volume_zscore_{window}'] = (volume - features[f'volume_ma_{window}']) / (features[f'volume_std_{window}'] + 1e-10)
        
        # Volume percentile (where is current volume relative to recent range?)
        for window in [20, 50, 100]:
            features[f'volume_percentile_{window}'] = volume.rolling(window).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5
            )
        
        # === VELOCITY (First Derivative - Rate of Change) ===
        # Volume velocity = change in volume
        for period in [1, 2, 3, 5]:
            features[f'volume_velocity_{period}'] = volume.diff(period)
            features[f'volume_velocity_pct_{period}'] = volume.pct_change(period)
        
        # Rolling velocity statistics
        for window in [5, 10, 20]:
            vel = volume.diff()
            features[f'velocity_ma_{window}'] = vel.rolling(window).mean()
            features[f'velocity_std_{window}'] = vel.rolling(window).std()
        
        # Velocity momentum (is velocity increasing or decreasing?)
        features['velocity_momentum_3'] = features['volume_velocity_1'].diff(3)
        features['velocity_momentum_5'] = features['volume_velocity_1'].diff(5)
        
        # === ACCELERATION (Second Derivative - Change in Velocity) ===
        # Volume acceleration = change in velocity
        vel_1 = volume.diff()
        for period in [1, 2, 3]:
            features[f'volume_accel_{period}'] = vel_1.diff(period)
        
        # Rolling acceleration statistics
        for window in [5, 10]:
            accel = vel_1.diff()
            features[f'accel_ma_{window}'] = accel.rolling(window).mean()
            features[f'accel_std_{window}'] = accel.rolling(window).std()
        
        # === SNAP/JERK (Third Derivative - Change in Acceleration) ===
        # Volume snap = change in acceleration
        accel_1 = vel_1.diff()
        features['volume_snap_1'] = accel_1.diff(1)
        features['volume_snap_2'] = accel_1.diff(2)
        
        # Rolling snap statistics
        snap = accel_1.diff()
        features['snap_ma_5'] = snap.rolling(5).mean()
        features['snap_std_5'] = snap.rolling(5).std()
        
        # === EXHAUSTION INDICATORS ===
        # High volume with decreasing velocity = exhaustion
        features['exhaustion_signal'] = (
            (features['volume_zscore_20'] > 1.5) &  # High volume
            (features['volume_velocity_1'] < 0)      # Decreasing
        ).astype(int)
        
        # Acceleration reversal (acceleration changes sign)
        features['accel_reversal'] = (features['volume_accel_1'] * features['volume_accel_1'].shift(1) < 0).astype(int)
        
        # Snap reversal (snap changes sign)
        features['snap_reversal'] = (features['volume_snap_1'] * features['volume_snap_1'].shift(1) < 0).astype(int)
        
        # === VOLUME REGIME ===
        # Classify volume regime
        vol_ma_20 = features['volume_ma_20']
        features['high_volume_regime'] = (volume > 1.5 * vol_ma_20).astype(int)
        features['low_volume_regime'] = (volume < 0.5 * vol_ma_20).astype(int)
        
        # Volume trend (increasing or decreasing over time)
        features['volume_trend_10'] = (features['volume_ma_5'] > features['volume_ma_10']).astype(int)
        features['volume_trend_20'] = (features['volume_ma_10'] > features['volume_ma_20']).astype(int)
        
        # === DIVERGENCE INDICATORS ===
        # Price vs Volume divergence
        price_change = self.df['close'].pct_change(5)
        volume_change = volume.pct_change(5)
        features['price_volume_divergence'] = (
            ((price_change > 0) & (volume_change < 0)) |  # Price up, volume down
            ((price_change < 0) & (volume_change > 0))     # Price down, volume up
        ).astype(int)
        
        features.fillna(0, inplace=True)
        self.features = features
        
        print(f"   ✅ Created {len(features.columns)} volume dynamics features")
        print(f"\n📊 Feature Categories:")
        print(f"   - Position (volume stats): 16 features")
        print(f"   - Velocity (1st derivative): 14 features")
        print(f"   - Acceleration (2nd derivative): 9 features")
        print(f"   - Snap/Jerk (3rd derivative): 4 features")
        print(f"   - Exhaustion signals: 3 features")
        print(f"   - Regime indicators: 4 features")
        print(f"   - Divergence: 1 feature")
        
        return features
    
    def build_model(self, input_dim):
        """Build volume dynamics neural network."""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu', name='dense1'),
            layers.Dropout(0.3, name='dropout1'),
            layers.Dense(64, activation='relu', name='dense2'),
            layers.Dropout(0.2, name='dropout2'),
            layers.Dense(32, activation='relu', name='dense3'),
            layers.Dropout(0.1, name='dropout3'),
            layers.Dense(1, activation='sigmoid', name='output')
        ], name='volume_dynamics_nn')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        return model
    
    def train(self):
        """Train the volume dynamics model."""
        print(f"\n{'='*60}")
        print(f"  TRAINING VOLUME DYNAMICS NEURAL NETWORK")
        print(f"{'='*60}")
        
        # Engineer features
        self.engineer_volume_dynamics_features()
        
        # Prepare target
        y = ((self.df['is_near_supply'] == 1) | (self.df['is_near_demand'] == 1)).astype(int).values
        
        print(f"\n🎯 Target Distribution:")
        print(f"   Positive: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"   Negative: {len(y) - y.sum()} ({(len(y) - y.sum())/len(y)*100:.1f}%)")
        
        # Split data
        train_size = int(0.7 * len(self.df))
        val_size = int(0.15 * len(self.df))
        
        X = self.features.values
        X_train = X[:train_size]
        X_val = X[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size+val_size]
        y_test = y[train_size+val_size:]
        
        # Check class distribution
        train_pos = y_train.sum()
        val_pos = y_val.sum()
        test_pos = y_test.sum()
        
        print(f"\n📊 Split Distribution:")
        print(f"   Train: {train_pos}/{len(y_train)} positive ({train_pos/len(y_train)*100:.1f}%)")
        print(f"   Val:   {val_pos}/{len(y_val)} positive ({val_pos/len(y_val)*100:.1f}%)")
        print(f"   Test:  {test_pos}/{len(y_test)} positive ({test_pos/len(y_test)*100:.1f}%)")
        
        # Calculate class weights
        class_weight = {
            0: 1.0,
            1: (len(y_train) - train_pos) / train_pos if train_pos > 0 else 1.0
        }
        print(f"\n⚖️  Class weights: {{0: 1.0, 1: {class_weight[1]:.2f}}}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        self.model = self.build_model(X_train.shape[1])
        
        print(f"\n🏗️  Model Architecture:")
        self.model.summary()
        
        # Train
        print(f"\n🚀 Training...")
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Evaluate on test set
        print(f"\n📊 Test Set Evaluation:")
        test_loss, test_acc, test_prec, test_rec, test_auc = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        test_preds = self.model.predict(X_test_scaled, verbose=0).flatten()
        pred_positive = (test_preds > 0.5).sum()
        
        print(f"   Accuracy:  {test_acc:.4f}")
        print(f"   Precision: {test_prec:.4f}")
        print(f"   Recall:    {test_rec:.4f}")
        print(f"   AUC:       {test_auc:.4f}")
        print(f"   Predictions: {pred_positive}/{len(test_preds)} positive ({pred_positive/len(test_preds)*100:.2f}%)")
        
        # Save model
        self.model.save('volume_dynamics_nn.keras')
        print(f"\n💾 Model saved to: volume_dynamics_nn.keras")
        
        return history


def main():
    print("="*60)
    print("  VOLUME DYNAMICS NEURAL NETWORK")
    print("  Analyzing Volume Velocity & Acceleration")
    print("="*60)
    
    trainer = VolumeDynamicsNN()
    
    if not trainer.load_data():
        return
    
    trainer.train()
    
    print(f"\n👋 Training Complete!")


if __name__ == "__main__":
    main()
