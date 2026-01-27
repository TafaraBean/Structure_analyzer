import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import talib
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class UltimateEnsembleStressTest:
    """Test 10+ specialized models with different indicator combinations."""
    
    def __init__(self, labels_file='zone_labels_bayesian_for_ml.csv'):
        self.labels_file = labels_file
        self.df = None
        self.models = {}
        self.scalers = {}
        self.feature_groups = {}
        self.results = []
        
    def load_data(self):
        """Load data."""
        print(f"\n📂 Loading {self.labels_file}...")
        self.df = pd.read_csv(self.labels_file)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df.set_index('time', inplace=True)
        print(f"✅ Loaded {len(self.df)} rows")
        return True
    
    def build_model(self, input_dim, name):
        """Build specialist model."""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        return model
    
    # === MODEL 1: ORDER FLOW ===
    def engineer_order_flow_features(self):
        """Volume + Price Action combined."""
        print(f"\n🔧 Model 1: Order Flow...")
        features = pd.DataFrame(index=self.df.index)
        
        # Volume pressure
        features['buy_pressure'] = (self.df['close'] - self.df['low']) / (self.df['high'] - self.df['low'] + 1e-10)
        features['sell_pressure'] = (self.df['high'] - self.df['close']) / (self.df['high'] - self.df['low'] + 1e-10)
        
        # Volume-weighted price action
        features['vwap_dist'] = (self.df['close'] - ((self.df['high'] + self.df['low'] + self.df['close']) / 3))
        
        # Accumulation/Distribution
        mfm = ((self.df['close'] - self.df['low']) - (self.df['high'] - self.df['close'])) / (self.df['high'] - self.df['low'] + 1e-10)
        features['ad_line'] = (mfm * self.df['tick_volume']).cumsum()
        
        # On-Balance Volume
        obv = (np.sign(self.df['close'].diff()) * self.df['tick_volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ma'] = obv.rolling(20).mean()
        
        features.fillna(0, inplace=True)
        self.feature_groups['order_flow'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    # === MODEL 2: MARKET STRUCTURE ===
    def engineer_market_structure_features(self):
        """Support/Resistance + Fractals."""
        print(f"\n🔧 Model 2: Market Structure...")
        features = pd.DataFrame(index=self.df.index)
        
        # Pivot points
        pivot = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        features['pivot_dist'] = (self.df['close'] - pivot) / self.df['close']
        
        # Higher highs / Lower lows
        features['higher_high'] = (self.df['high'] > self.df['high'].shift(1)).astype(int)
        features['lower_low'] = (self.df['low'] < self.df['low'].shift(1)).astype(int)
        
        # Swing detection
        for window in [5, 10, 20]:
            features[f'at_high_{window}'] = (self.df['high'] == self.df['high'].rolling(window).max()).astype(int)
            features[f'at_low_{window}'] = (self.df['low'] == self.df['low'].rolling(window).min()).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['market_structure'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    # === MODEL 3: MOMENTUM EXHAUSTION ===
    def engineer_momentum_exhaustion_features(self):
        """RSI + Stochastic + CCI."""
        print(f"\n🔧 Model 3: Momentum Exhaustion...")
        features = pd.DataFrame(index=self.df.index)
        
        # RSI
        for period in [7, 14, 21]:
            rsi = talib.RSI(self.df['close'].values, timeperiod=period)
            features[f'rsi_{period}'] = rsi / 100
            features[f'rsi_overbought_{period}'] = (rsi > 70).astype(int)
            features[f'rsi_oversold_{period}'] = (rsi < 30).astype(int)
        
        # Stochastic
        slowk, slowd = talib.STOCH(self.df['high'].values, self.df['low'].values, 
                                   self.df['close'].values, fastk_period=14, slowk_period=3, slowd_period=3)
        slowk_s = pd.Series(slowk, index=self.df.index)
        slowd_s = pd.Series(slowd, index=self.df.index)
        features['stoch_k'] = slowk / 100
        features['stoch_d'] = slowd / 100
        features['stoch_cross'] = ((slowk_s > slowd_s) & (slowk_s.shift(1) <= slowd_s.shift(1))).astype(int)
        
        # CCI
        cci = talib.CCI(self.df['high'].values, self.df['low'].values, self.df['close'].values, timeperiod=20)
        features['cci'] = cci / 200  # Normalize
        features['cci_extreme'] = (np.abs(cci) > 100).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['momentum_exhaustion'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    # === MODEL 4: TREND REVERSAL ===
    def engineer_trend_reversal_features(self):
        """MACD + ADX + Parabolic SAR."""
        print(f"\n🔧 Model 4: Trend Reversal...")
        features = pd.DataFrame(index=self.df.index)
        
        # MACD
        macd, signal, hist = talib.MACD(self.df['close'].values)
        macd_s = pd.Series(macd, index=self.df.index)
        signal_s = pd.Series(signal, index=self.df.index)
        features['macd'] = macd / self.df['close']
        features['macd_signal'] = signal / self.df['close']
        features['macd_hist'] = hist / self.df['close']
        features['macd_cross'] = ((macd_s > signal_s) & (macd_s.shift(1) <= signal_s.shift(1))).astype(int)
        
        # ADX
        adx = talib.ADX(self.df['high'].values, self.df['low'].values, self.df['close'].values, timeperiod=14)
        features['adx'] = adx / 100
        features['adx_strong'] = (adx > 25).astype(int)
        
        # Parabolic SAR
        sar = talib.SAR(self.df['high'].values, self.df['low'].values)
        sar_s = pd.Series(sar, index=self.df.index)
        features['sar_above'] = (self.df['close'] > sar_s).astype(int)
        features['sar_flip'] = (features['sar_above'] != features['sar_above'].shift(1)).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['trend_reversal'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    # === MODEL 5: VOLATILITY BREAKOUT ===
    def engineer_volatility_breakout_features(self):
        """ATR + Bollinger + Keltner."""
        print(f"\n🔧 Model 5: Volatility Breakout...")
        features = pd.DataFrame(index=self.df.index)
        
        # ATR
        atr = talib.ATR(self.df['high'].values, self.df['low'].values, self.df['close'].values, timeperiod=14)
        atr_s = pd.Series(atr, index=self.df.index)
        features['atr'] = atr / self.df['close']
        features['atr_expanding'] = (atr_s > atr_s.shift(5)).astype(int)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.df['close'].values, timeperiod=20)
        features['bb_width'] = (upper - lower) / self.df['close']
        features['bb_position'] = (self.df['close'] - lower) / (upper - lower + 1e-10)
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean()).astype(int)
        
        # Keltner Channels
        kelt_middle = talib.EMA(self.df['close'].values, 20)
        kelt_upper = kelt_middle + 2 * atr
        kelt_lower = kelt_middle - 2 * atr
        features['kelt_position'] = (self.df['close'] - kelt_lower) / (kelt_upper - kelt_lower + 1e-10)
        
        features.fillna(0, inplace=True)
        self.feature_groups['volatility_breakout'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    # === MODEL 6: DIVERGENCE HUNTER ===
    def engineer_divergence_features(self):
        """Price vs Indicator divergences."""
        print(f"\n🔧 Model 6: Divergence Hunter...")
        features = pd.DataFrame(index=self.df.index)
        
        # Price momentum
        price_mom = self.df['close'].pct_change(10)
        
        # RSI divergence
        rsi = talib.RSI(self.df['close'].values, timeperiod=14)
        rsi_mom = pd.Series(rsi).pct_change(10)
        features['rsi_divergence'] = ((price_mom > 0) & (rsi_mom < 0) | (price_mom < 0) & (rsi_mom > 0)).astype(int)
        
        # MACD divergence
        macd, _, _ = talib.MACD(self.df['close'].values)
        macd_mom = pd.Series(macd).pct_change(10)
        features['macd_divergence'] = ((price_mom > 0) & (macd_mom < 0) | (price_mom < 0) & (macd_mom > 0)).astype(int)
        
        # Volume divergence
        vol_mom = self.df['tick_volume'].pct_change(10)
        features['volume_divergence'] = ((price_mom > 0) & (vol_mom < 0) | (price_mom < 0) & (vol_mom > 0)).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['divergence'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    # === MODEL 7: FIBONACCI ===
    def engineer_fibonacci_features(self):
        """Fibonacci retracements."""
        print(f"\n🔧 Model 7: Fibonacci...")
        features = pd.DataFrame(index=self.df.index)
        
        # Calculate swing high/low over window
        for window in [20, 50]:
            swing_high = self.df['high'].rolling(window).max()
            swing_low = self.df['low'].rolling(window).min()
            range_size = swing_high - swing_low
            
            # Fib levels
            for level, ratio in [('236', 0.236), ('382', 0.382), ('500', 0.500), ('618', 0.618)]:
                fib_level = swing_low + range_size * ratio
                features[f'near_fib_{level}_{window}'] = (np.abs(self.df['close'] - fib_level) / self.df['close'] < 0.001).astype(int)
        
        features.fillna(0, inplace=True)
        self.feature_groups['fibonacci'] = features
        print(f"   ✅ {len(features.columns)} features")
    
    def train_all_models(self):
        """Train all specialist models."""
        print(f"\n{'='*60}")
        print(f"  ULTIMATE ENSEMBLE STRESS TEST")
        print(f"  Testing 7 Specialized Models")
        print(f"{'='*60}")
        
        # Engineer all features
        self.engineer_order_flow_features()
        self.engineer_market_structure_features()
        self.engineer_momentum_exhaustion_features()
        self.engineer_trend_reversal_features()
        self.engineer_volatility_breakout_features()
        self.engineer_divergence_features()
        self.engineer_fibonacci_features()
        
        # Prepare target
        y = ((self.df['is_near_supply'] == 1) | (self.df['is_near_demand'] == 1)).astype(int).values
        
        # Split data
        train_size = int(0.7 * len(self.df))
        val_size = int(0.15 * len(self.df))
        
        # Train each model
        for name, features in self.feature_groups.items():
            print(f"\n{'='*60}")
            print(f"Training: {name.upper()}")
            print(f"{'='*60}")
            
            X = features.values
            X_train = X[:train_size]
            X_val = X[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            
            y_train = y[:train_size]
            y_val = y[train_size:train_size+val_size]
            y_test = y[train_size+val_size:]
            
            # Class weights
            train_pos = y_train.sum()
            class_weight = {0: 1.0, 1: (len(y_train) - train_pos) / train_pos if train_pos > 0 else 1.0}
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[name] = scaler
            
            # Build and train
            model = self.build_model(X_train.shape[1], name)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=50,
                batch_size=32,
                class_weight=class_weight,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate on test set
            test_loss, test_acc, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)
            
            print(f"✅ Test Results:")
            print(f"   Accuracy: {test_acc:.4f}")
            print(f"   AUC:      {test_auc:.4f}")
            
            # Save results
            self.results.append({
                'name': name,
                'auc': test_auc,
                'accuracy': test_acc,
                'model': model,
                'scaler': scaler
            })
            
            model.save(f'model_{name}.keras')
        
        # Sort by AUC
        self.results.sort(key=lambda x: x['auc'], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"  FINAL RANKINGS")
        print(f"{'='*60}")
        for i, result in enumerate(self.results, 1):
            print(f"{i}. {result['name']:25s} AUC: {result['auc']:.4f}")
        
        # Build meta-ensemble with top 3
        self.build_meta_ensemble(y_test, X_test)
    
    def build_meta_ensemble(self, y_test, X_test_dict):
        """Combine top 3 models."""
        print(f"\n{'='*60}")
        print(f"  META-ENSEMBLE (Top 3 Models)")
        print(f"{'='*60}")
        
        top3 = self.results[:3]
        print(f"\nSelected models:")
        for i, model_info in enumerate(top3, 1):
            print(f"  {i}. {model_info['name']} (AUC: {model_info['auc']:.4f})")
        
        # Get predictions from top 3
        train_size = int(0.7 * len(self.df))
        val_size = int(0.15 * len(self.df))
        
        ensemble_preds = []
        for model_info in top3:
            features = self.feature_groups[model_info['name']].values
            X_test = features[train_size+val_size:]
            X_test_scaled = model_info['scaler'].transform(X_test)
            preds = model_info['model'].predict(X_test_scaled, verbose=0).flatten()
            ensemble_preds.append(preds)
        
        # Simple averaging
        avg_preds = np.mean(ensemble_preds, axis=0)
        avg_preds_binary = (avg_preds > 0.5).astype(int)
        
        # Calculate ensemble metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        ensemble_acc = accuracy_score(y_test, avg_preds_binary)
        ensemble_auc = roc_auc_score(y_test, avg_preds)
        
        print(f"\n✅ Ensemble Performance:")
        print(f"   Accuracy: {ensemble_acc:.4f}")
        print(f"   AUC:      {ensemble_auc:.4f}")
        print(f"\n🎯 Improvement over best individual:")
        print(f"   Best individual AUC: {top3[0]['auc']:.4f}")
        print(f"   Ensemble AUC:        {ensemble_auc:.4f}")
        print(f"   Gain:                {(ensemble_auc - top3[0]['auc']):.4f}")


def main():
    print("="*60)
    print("  ULTIMATE ENSEMBLE STRESS TEST")
    print("  EURUSDm | M15")
    print("="*60)
    
    tester = UltimateEnsembleStressTest()
    tester.load_data()
    tester.train_all_models()
    
    print(f"\n👋 Complete!")


if __name__ == "__main__":
    main()
