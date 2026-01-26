import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

# Import the RegimeLabelGenerator from regime_labeler
import sys
sys.path.append(os.path.dirname(__file__))
from regime_labeler import RegimeLabelGenerator

class RegimePredictor:
    """Neural network to predict next candle's market regime."""
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15, bars=5000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def init_mt5(self):
        """Initialize MT5."""
        path = os.getenv("MT5_PATH")
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        params = {}
        if path: params["path"] = path
        
        if not mt5.initialize(**params):
            print(f"❌ MT5 Init failed")
            return False
        if login and password and server:
            mt5.login(login=int(login), password=password, server=server)
        print(f"✅ Connected to MT5")
        return True
    
    def create_features(self, df):
        """Create technical indicator features for prediction."""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
        # Volatility indicators
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        features['atr_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
        features['natr_14'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # Momentum indicators
        features['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
        features['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        features['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
        
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(df['close'])
        
        features['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        features['cci_20'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Trend indicators
        features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume features
        features['volume_sma_10'] = talib.SMA(df['tick_volume'], timeperiod=10)
        features['volume_ratio'] = df['tick_volume'] / features['volume_sma_10']
        
        # Statistical features
        for period in [10, 20, 50]:
            features[f'std_{period}'] = df['close'].rolling(period).std()
            features[f'skew_{period}'] = df['close'].rolling(period).skew()
            features[f'kurt_{period}'] = df['close'].rolling(period).kurt()
        
        # Lag features (previous candles)
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
            features[f'adx_lag_{lag}'] = features['adx_14'].shift(lag)
        
        return features
    
    def prepare_data(self):
        """Fetch data, generate labels, create features."""
        print(f"\n📊 Preparing data...")
        
        # Generate labels using consensus method
        labeler = RegimeLabelGenerator(self.symbol, self.timeframe, self.bars)
        labeler.df = labeler.fetch_data()
        
        print(f"🏷️  Generating consensus labels...")
        result = labeler.method_consensus()
        
        # Create features
        print(f"🔧 Creating features...")
        features = self.create_features(labeler.df)
        
        # Target: Next candle's regime (shift labels backward by 1)
        target = (result['labels'].shift(-1) == 'sideways').astype(int)
        
        # Combine
        data = pd.concat([features, target.rename('target')], axis=1)
        data = data.dropna()
        
        print(f"✅ Data prepared: {len(data)} samples")
        print(f"   Sideways (1): {(data['target'] == 1).sum()} ({(data['target'] == 1).mean()*100:.1f}%)")
        print(f"   Trending (0): {(data['target'] == 0).sum()} ({(data['target'] == 0).mean()*100:.1f}%)")
        
        return data
    
    def build_model(self, input_dim, class_weight=None):
        """Build neural network with dropout and batch normalization."""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Use focal loss to handle class imbalance
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """Train the neural network."""
        print(f"\n🎯 Training neural network...")
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        self.feature_names = X.columns.tolist()
        
        # Split data (time-series aware - no shuffling)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Calculate class weights to handle imbalance
        class_counts = y_train.value_counts()
        total = len(y_train)
        class_weight = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        
        print(f"\n   Class weights: {class_weight}")
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1], class_weight)
        
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
        """Evaluate model and print metrics."""
        print(f"\n📊 Evaluating model...")
        
        # Predictions
        y_pred_proba = self.model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        print(f"\n{'='*60}")
        print("  CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(classification_report(y_val, y_pred, target_names=['Trending', 'Sideways']))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Trending  Sideways")
        print(f"Actual Trending    {cm[0,0]:4d}     {cm[0,1]:4d}")
        print(f"       Sideways    {cm[1,0]:4d}     {cm[1,1]:4d}")
        
        # ROC AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"\nROC AUC Score: {auc:.4f}")
        
        # Check if model is predicting both classes
        unique_preds = np.unique(y_pred)
        print(f"\nUnique predictions: {unique_preds}")
        print(f"Sideways predictions: {(y_pred == 1).sum()} ({(y_pred == 1).mean()*100:.1f}%)")
        print(f"Trending predictions: {(y_pred == 0).sum()} ({(y_pred == 0).mean()*100:.1f}%)")
        
        return y_pred, y_pred_proba
    
    def plot_results(self, history, y_val, y_pred, y_pred_proba):
        """Plot training history and results."""
        print(f"\n📈 Creating visualizations...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        plt.style.use('dark_background')
        
        # Training history - Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history.history['loss'], label='Train Loss', color='cyan')
        ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.2)
        
        # Training history - Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history.history['accuracy'], label='Train Accuracy', color='cyan')
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.2)
        
        # Confusion Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Trending', 'Sideways'],
                   yticklabels=['Trending', 'Sideways'])
        ax3.set_title('Confusion Matrix')
        ax3.set_ylabel('Actual')
        ax3.set_xlabel('Predicted')
        
        # ROC Curve
        ax4 = fig.add_subplot(gs[1, 1])
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        auc = roc_auc_score(y_val, y_pred_proba)
        ax4.plot(fpr, tpr, color='cyan', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.2)
        
        # Prediction distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(y_pred_proba[y_val == 0], bins=50, alpha=0.5, label='Actual Trending', color='red')
        ax5.hist(y_pred_proba[y_val == 1], bins=50, alpha=0.5, label='Actual Sideways', color='green')
        ax5.axvline(x=0.5, color='yellow', linestyle='--', linewidth=2, label='Threshold')
        ax5.set_xlabel('Predicted Probability (Sideways)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Prediction Probability Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.2)
        
        # Prediction over time
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(y_pred_proba, color='cyan', linewidth=1, alpha=0.7, label='Predicted Probability')
        ax6.fill_between(range(len(y_val)), 0, 1, where=(y_val == 1), 
                        color='green', alpha=0.2, label='Actual Sideways')
        ax6.axhline(y=0.5, color='yellow', linestyle='--', linewidth=1)
        ax6.set_xlabel('Sample')
        ax6.set_ylabel('Probability')
        ax6.set_title('Predictions Over Time (Validation Set)')
        ax6.legend()
        ax6.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        filename = 'regime_prediction_results.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Results saved as {filename}")
        
        plt.show()
        print(f"📊 Visualization displayed")
    
    def save_model(self, filename='regime_predictor_model.h5'):
        """Save trained model."""
        self.model.save(filename)
        print(f"\n💾 Model saved as {filename}")

def main():
    print("="*60)
    print("  REGIME PREDICTION - NEURAL NETWORK TRAINING")
    print("  Symbol: EURUSDm | Timeframe: M15")
    print("="*60)
    
    predictor = RegimePredictor('EURUSDm', mt5.TIMEFRAME_M15, bars=5000)
    
    if not predictor.init_mt5():
        return
    
    # Prepare data
    data = predictor.prepare_data()
    
    # Train model
    history, X_val, y_val = predictor.train(data, epochs=100, batch_size=32, validation_split=0.2)
    
    # Evaluate
    y_pred, y_pred_proba = predictor.evaluate(X_val, y_val)
    
    # Plot results
    predictor.plot_results(history, y_val, y_pred, y_pred_proba)
    
    # Save model
    predictor.save_model()
    
    mt5.shutdown()
    print(f"\n👋 Training complete")

if __name__ == "__main__":
    main()
