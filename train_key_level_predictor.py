import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt

class KeyLevelPredictor:
    """Predict when price is at a key support/resistance level."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, filename='key_level_labels.csv'):
        """Load labeled data."""
        print(f"📊 Loading data from {filename}...")
        
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"   At key level: {df['at_key_level'].sum()} ({df['at_key_level'].mean()*100:.1f}%)")
        print(f"   Near support: {df['near_support'].sum()} ({df['near_support'].mean()*100:.1f}%)")
        print(f"   Near resistance: {df['near_resistance'].sum()} ({df['near_resistance'].mean()*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """Create features for prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Price momentum
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_3'] = df['close'].pct_change(3)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        
        # Volatility
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['atr_normalized'] = features['atr_14'] / df['close']
        
        # Price position in range
        for period in [10, 20, 50]:
            high_range = df['high'].rolling(period).max()
            low_range = df['low'].rolling(period).min()
            features[f'range_position_{period}'] = (df['close'] - low_range) / (high_range - low_range)
        
        # RSI
        features['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        features['rsi_extreme'] = ((features['rsi_14'] < 30) | (features['rsi_14'] > 70)).astype(int)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        features['touching_bb'] = ((df['high'] >= upper) | (df['low'] <= lower)).astype(int)
        
        # Volume
        features['volume_ratio'] = df['tick_volume'] / talib.SMA(df['tick_volume'], timeperiod=10)
        
        # Candle patterns
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Distance from recent highs/lows
        features['dist_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        features['dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        return features
    
    def prepare_data(self, df):
        """Prepare features and targets."""
        print(f"\n🔧 Preparing training data...")
        
        features = self.create_features(df)
        
        # Target: Is CURRENT price at a key level?
        target = df['at_key_level']
        
        # Combine
        data = pd.concat([features, target.rename('target')], axis=1)
        data = data.dropna()
        
        print(f"✅ Data prepared: {len(data)} samples")
        print(f"   At key level: {data['target'].sum()} ({data['target'].mean()*100:.1f}%)")
        
        return data
    
    def build_model(self, input_dim):
        """Build neural network."""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()]
        )
        
        return model
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """Train model."""
        print(f"\n🎯 Training key level predictor...")
        
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Time-series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"   Train: {len(X_train)} samples ({y_train.mean()*100:.1f}% at level)")
        print(f"   Val: {len(X_val)} samples ({y_val.mean()*100:.1f}% at level)")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Class weights
        pos_weight = len(y_train) / (2 * y_train.sum()) if y_train.sum() > 0 else 1
        class_weight = {0: 1.0, 1: pos_weight}
        
        print(f"   Class weight for 'at level': {pos_weight:.2f}")
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Train
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history, X_val_scaled, y_val
    
    def evaluate(self, X_val, y_val):
        """Evaluate model."""
        print(f"\n📊 Evaluating model...")
        
        y_pred_proba = self.model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print(f"\n{'='*60}")
        print("  CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(classification_report(y_val, y_pred, target_names=['Not at Level', 'At Key Level']))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted")
        print(f"              Not at Level  At Key Level")
        print(f"Actual Not at Level   {cm[0,0]:4d}         {cm[0,1]:4d}")
        print(f"       At Key Level   {cm[1,0]:4d}         {cm[1,1]:4d}")
        
        # ROC AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"\nROC AUC Score: {auc:.4f}")
        
        # Check predictions
        unique_preds = np.unique(y_pred)
        print(f"\nUnique predictions: {unique_preds}")
        print(f"At level predictions: {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.1f}%)")
        print(f"Not at level predictions: {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.1f}%)")
        
        # Check if blind firing
        if len(unique_preds) == 1:
            print(f"\n⚠️  WARNING: Model is blind firing (predicting only one class)!")
        else:
            print(f"\n✅ Model is predicting both classes - good!")
        
        return y_pred, y_pred_proba
    
    def save_model(self, filename='key_level_predictor.h5'):
        """Save model."""
        self.model.save(filename)
        print(f"\n💾 Model saved as {filename}")

def main():
    print("="*60)
    print("  KEY LEVEL PREDICTOR - TRAINING")
    print("  Predicts: Is price at a key support/resistance level?")
    print("="*60)
    
    predictor = KeyLevelPredictor()
    
    # Load data
    df = predictor.load_data('key_level_labels.csv')
    
    # Prepare data
    data = predictor.prepare_data(df)
    
    # Train
    history, X_val, y_val = predictor.train(data, epochs=100, batch_size=32)
    
    # Evaluate
    predictor.evaluate(X_val, y_val)
    
    # Save
    predictor.save_model()
    
    print(f"\n{'='*60}")
    print("  COMPLETE")
    print(f"{'='*60}")
    print(f"\n✅ Model trained to predict key level proximity")
    print(f"   Use this during sideways regimes for reversal entries!")
    print(f"\n👋 Done")

if __name__ == "__main__":
    main()
