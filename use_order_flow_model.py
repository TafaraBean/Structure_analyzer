import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import os
from dotenv import load_dotenv

load_dotenv()


class OrderFlowReversalDetector:
    """Use trained Order Flow model to detect reversals in real-time."""
    
    def __init__(self, model_path='model_order_flow.keras', symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.scaler = StandardScaler()
        
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
    
    def load_model(self):
        """Load trained model."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print(f"   Run ultimate_ensemble_test.py first to train the model")
            return False
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Loaded model: {self.model_path}")
        return True
    
    def fetch_data(self, bars=100):
        """Fetch recent price data."""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            print(f"❌ Failed to fetch data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    def calculate_order_flow_features(self, df):
        """Calculate the 6 Order Flow features."""
        features = pd.DataFrame(index=df.index)
        
        # 1. Buy Pressure
        features['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # 2. Sell Pressure
        features['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        
        # 3. VWAP Distance
        features['vwap_dist'] = (df['close'] - ((df['high'] + df['low'] + df['close']) / 3))
        
        # 4. Accumulation/Distribution Line
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        features['ad_line'] = (mfm * df['tick_volume']).cumsum()
        
        # 5. On-Balance Volume
        obv = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
        features['obv'] = obv
        
        # 6. OBV Moving Average
        features['obv_ma'] = obv.rolling(20).mean()
        
        features.fillna(0, inplace=True)
        return features
    
    def predict_reversal(self, df):
        """Predict reversal probability for latest candle."""
        # Calculate features
        features = self.calculate_order_flow_features(df)
        
        # Get latest features
        latest_features = features.iloc[-1:].values
        
        # Scale (fit on historical data, transform latest)
        self.scaler.fit(features.values)
        latest_scaled = self.scaler.transform(latest_features)
        
        # Predict
        prediction = self.model.predict(latest_scaled, verbose=0)[0][0]
        
        return prediction
    
    def get_signal(self, threshold=0.6):
        """Get trading signal based on order flow analysis."""
        print(f"\n{'='*60}")
        print(f"  ORDER FLOW REVERSAL DETECTOR")
        print(f"  {self.symbol} | {self.timeframe_name()}")
        print(f"{'='*60}")
        
        # Fetch data
        df = self.fetch_data(bars=100)
        if df is None:
            return None
        
        # Get current price info
        current = df.iloc[-1]
        print(f"\n📊 Current Price: {current['close']:.5f}")
        print(f"   Time: {df.index[-1]}")
        
        # Calculate features for analysis
        features = self.calculate_order_flow_features(df)
        latest = features.iloc[-1]
        
        print(f"\n🔍 Order Flow Analysis:")
        print(f"   Buy Pressure:  {latest['buy_pressure']:.3f} {'🟢' if latest['buy_pressure'] > 0.6 else '🔴' if latest['buy_pressure'] < 0.4 else '⚪'}")
        print(f"   Sell Pressure: {latest['sell_pressure']:.3f} {'🔴' if latest['sell_pressure'] > 0.6 else '🟢' if latest['sell_pressure'] < 0.4 else '⚪'}")
        print(f"   VWAP Dist:     {latest['vwap_dist']:.5f}")
        print(f"   A/D Line:      {latest['ad_line']:.0f}")
        print(f"   OBV:           {latest['obv']:.0f}")
        
        # Get prediction
        reversal_prob = self.predict_reversal(df)
        
        print(f"\n🎯 Reversal Probability: {reversal_prob:.1%}")
        
        # Generate signal
        if reversal_prob >= threshold:
            # Determine direction based on pressure
            if latest['buy_pressure'] > latest['sell_pressure']:
                signal = "BULLISH REVERSAL"
                emoji = "🟢📈"
                direction = "LONG"
            else:
                signal = "BEARISH REVERSAL"
                emoji = "🔴📉"
                direction = "SHORT"
            
            print(f"\n{emoji} {signal} DETECTED!")
            print(f"   Direction: {direction}")
            print(f"   Confidence: {reversal_prob:.1%}")
            print(f"   Entry: {current['close']:.5f}")
            
            return {
                'signal': signal,
                'direction': direction,
                'probability': reversal_prob,
                'entry_price': current['close'],
                'time': df.index[-1],
                'buy_pressure': latest['buy_pressure'],
                'sell_pressure': latest['sell_pressure']
            }
        else:
            print(f"\n⚪ No reversal signal (probability too low)")
            return None
    
    def timeframe_name(self):
        """Get timeframe name."""
        tf_map = {
            mt5.TIMEFRAME_M1: 'M1',
            mt5.TIMEFRAME_M5: 'M5',
            mt5.TIMEFRAME_M15: 'M15',
            mt5.TIMEFRAME_M30: 'M30',
            mt5.TIMEFRAME_H1: 'H1',
            mt5.TIMEFRAME_H4: 'H4',
            mt5.TIMEFRAME_D1: 'D1'
        }
        return tf_map.get(self.timeframe, 'Unknown')
    
    def monitor_live(self, interval_seconds=60, threshold=0.6):
        """Monitor for reversal signals continuously."""
        import time
        
        print(f"\n🔄 Starting live monitoring...")
        print(f"   Checking every {interval_seconds} seconds")
        print(f"   Reversal threshold: {threshold:.1%}")
        print(f"   Press Ctrl+C to stop\n")
        
        try:
            while True:
                signal = self.get_signal(threshold=threshold)
                
                if signal:
                    print(f"\n🚨 ALERT: {signal['signal']}")
                    print(f"   Take {signal['direction']} position at {signal['entry_price']:.5f}")
                
                print(f"\n⏳ Next check in {interval_seconds}s...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Monitoring stopped")


def main():
    print("="*60)
    print("  ORDER FLOW REVERSAL DETECTOR")
    print("  Using trained neural network")
    print("="*60)
    
    # Initialize detector
    detector = OrderFlowReversalDetector(
        model_path='model_order_flow.keras',
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    # Connect to MT5
    if not detector.init_mt5():
        return
    
    # Load model
    if not detector.load_model():
        return
    
    # Get current signal
    signal = detector.get_signal(threshold=0.6)
    
    # Optional: Start live monitoring
    # detector.monitor_live(interval_seconds=60, threshold=0.6)
    
    mt5.shutdown()
    print(f"\n👋 Complete")


if __name__ == "__main__":
    main()
