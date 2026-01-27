import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class LowProbabilityAnalyzer:
    """Analyze low reversal probability zones and their correlation with trends."""
    
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
            return False
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Loaded model: {self.model_path}")
        return True
    
    def fetch_test_data(self):
        """Fetch the held-out test set (last 15% of 3000 bars)."""
        total_bars = 3000
        test_bars = int(0.15 * total_bars)
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, total_bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Take last 15% (test set)
        df_test = df.iloc[-test_bars:]
        
        print(f"✅ Loaded {len(df_test)} bars (test set)")
        return df_test
    
    def calculate_order_flow_features(self, df):
        """Calculate Order Flow features."""
        features = pd.DataFrame(index=df.index)
        
        features['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        features['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        features['vwap_dist'] = (df['close'] - ((df['high'] + df['low'] + df['close']) / 3))
        
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        features['ad_line'] = (mfm * df['tick_volume']).cumsum()
        
        obv = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ma'] = obv.rolling(20).mean()
        
        features.fillna(0, inplace=True)
        return features
    
    def calculate_trend_indicators(self, df):
        """Calculate trend strength indicators."""
        # ADX (trend strength)
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # EMA trend
        ema_20 = talib.EMA(df['close'].values, 20)
        ema_50 = talib.EMA(df['close'].values, 50)
        
        # Trend direction
        trend_up = ema_20 > ema_50
        trend_down = ema_20 < ema_50
        
        return pd.DataFrame({
            'adx': adx,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'trend_up': trend_up,
            'trend_down': trend_down,
            'strong_trend': adx > 25  # ADX > 25 indicates strong trend
        }, index=df.index)
    
    def analyze(self):
        """Analyze low probability zones and trend acceleration."""
        print(f"\n{'='*60}")
        print(f"  LOW PROBABILITY ZONE ANALYSIS")
        print(f"  Threshold: <30% reversal probability")
        print(f"{'='*60}")
        
        # Fetch test data
        df = self.fetch_test_data()
        if df is None:
            return
        
        # Calculate features
        features = self.calculate_order_flow_features(df)
        
        # Get predictions
        self.scaler.fit(features.values)
        scaled_features = self.scaler.transform(features.values)
        predictions = self.model.predict(scaled_features, verbose=0).flatten()
        
        # Calculate trend indicators
        trend_data = self.calculate_trend_indicators(df)
        
        # Calculate derivatives (rate of change)
        adx_change = pd.Series(trend_data['adx']).diff(5)  # ADX change over 5 periods
        prob_change = pd.Series(predictions).diff(5)  # Probability change over 5 periods
        
        # Identify low probability zones (<30%)
        low_prob_mask = predictions < 0.5
        low_prob_count = low_prob_mask.sum()
        
        # Identify TREND ACCELERATION zones (ADX rising + Prob decreasing)
        trend_accel_mask = (adx_change > 0) & (prob_change < 0) & (trend_data['adx'] > 20)
        trend_accel_count = trend_accel_mask.sum()
        
        print(f"\n📊 Low Probability Zones (<30%):")
        print(f"   Total bars: {len(df)}")
        print(f"   Low prob bars: {low_prob_count} ({low_prob_count/len(df)*100:.1f}%)")
        
        print(f"\n🚀 Trend Acceleration Zones:")
        print(f"   (ADX rising + Reversal prob decreasing)")
        print(f"   Acceleration bars: {trend_accel_count} ({trend_accel_count/len(df)*100:.1f}%)")
        
        # Analyze correlation with trends
        low_prob_indices = np.where(low_prob_mask)[0]
        
        if len(low_prob_indices) > 0:
            # Get trend data for low prob zones
            low_prob_adx = trend_data['adx'].iloc[low_prob_indices]
            low_prob_strong_trend = trend_data['strong_trend'].iloc[low_prob_indices]
            
            avg_adx_low_prob = low_prob_adx.mean()
            strong_trend_pct = low_prob_strong_trend.sum() / len(low_prob_indices) * 100
            
            print(f"\n🔍 Trend Analysis (Low Prob Zones):")
            print(f"   Average ADX: {avg_adx_low_prob:.1f}")
            print(f"   Strong trend (ADX>25): {strong_trend_pct:.1f}%")
            
            # Compare to overall dataset
            avg_adx_overall = trend_data['adx'].mean()
            strong_trend_overall = trend_data['strong_trend'].sum() / len(df) * 100
            
            print(f"\n📈 Comparison to Overall Dataset:")
            print(f"   Overall avg ADX: {avg_adx_overall:.1f}")
            print(f"   Overall strong trend: {strong_trend_overall:.1f}%")
            print(f"\n💡 Insight:")
            if strong_trend_pct > strong_trend_overall * 1.2:
                print(f"   ✅ Low prob zones are {strong_trend_pct/strong_trend_overall:.1f}x more likely to be trending!")
            else:
                print(f"   ⚠️  Low prob zones are NOT strongly correlated with trends")
        
        # Analyze trend acceleration zones
        if trend_accel_count > 0:
            accel_indices = np.where(trend_accel_mask)[0]
            accel_adx = trend_data['adx'].iloc[accel_indices]
            accel_prob = predictions[accel_indices]
            
            print(f"\n🔥 Trend Acceleration Analysis:")
            print(f"   Average ADX: {accel_adx.mean():.1f}")
            print(f"   Average reversal prob: {accel_prob.mean()*100:.1f}%")
            print(f"   → These zones show strengthening trends with low reversal risk!")
        
        # Create visualization
        self.plot_analysis(df, predictions, trend_data, adx_change, prob_change, trend_accel_mask)
    
    def plot_analysis(self, df, predictions, trend_data, adx_change, prob_change, trend_accel_mask):
        """Plot price chart with low probability and trend acceleration zones highlighted."""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        # Subplot 1: Price with zones highlighted
        ax1.plot(df.index, df['close'], linewidth=1, color='black', label='Price')
        ax1.plot(df.index, trend_data['ema_20'], linewidth=1, color='blue', alpha=0.5, label='EMA 20')
        ax1.plot(df.index, trend_data['ema_50'], linewidth=1, color='red', alpha=0.5, label='EMA 50')
        
        # Highlight low probability zones (yellow)
        low_prob_mask = predictions < 0.3
        for i in range(len(df)):
            if low_prob_mask[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.2, color='yellow', label='Low Prob (<30%)' if i == np.where(low_prob_mask)[0][0] else '')
        
        # Highlight trend acceleration zones (green) - OVERLAY on top
        for i in range(len(df)):
            if trend_accel_mask[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.4, color='green', label='Trend Acceleration' if i == np.where(trend_accel_mask)[0][0] else '')
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price Chart: Yellow=Low Prob, Green=Trend Acceleration (ADX↑ + Prob↓)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        # Subplot 2: Reversal Probability
        ax2.plot(df.index, predictions * 100, linewidth=1, color='purple', label='Reversal Probability')
        ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='30% Threshold')
        ax2.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='70% Threshold')
        ax2.fill_between(df.index, 0, predictions * 100, where=(predictions < 0.3), 
                         alpha=0.2, color='yellow', label='Low Prob Zone')
        
        # Highlight trend acceleration zones
        for i in range(len(df)):
            if trend_accel_mask[i]:
                ax2.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='green')
        
        ax2.set_ylabel('Probability (%)')
        ax2.set_title('Reversal Probability (Green = Decreasing)', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(alpha=0.3)
        
        # Subplot 3: ADX (Trend Strength)
        ax3.plot(df.index, trend_data['adx'], linewidth=1, color='orange', label='ADX')
        ax3.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Strong Trend (>25)')
        ax3.fill_between(df.index, 0, trend_data['adx'], where=(trend_data['adx'] > 25), 
                         alpha=0.2, color='red', label='Strong Trend')
        
        # Highlight trend acceleration zones
        for i in range(len(df)):
            if trend_accel_mask[i]:
                ax3.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='green')
        
        ax3.set_ylabel('ADX')
        ax3.set_title('Trend Strength - ADX (Green = Rising)', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)
        
        # Subplot 4: Rate of Change (NEW!)
        ax4_twin = ax4.twinx()
        
        # ADX change (left axis)
        ax4.plot(df.index, adx_change, linewidth=1, color='orange', alpha=0.7, label='ADX Change')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.fill_between(df.index, 0, adx_change, where=(adx_change > 0), 
                         alpha=0.2, color='orange', label='ADX Rising')
        ax4.set_ylabel('ADX Change', color='orange')
        ax4.tick_params(axis='y', labelcolor='orange')
        
        # Probability change (right axis)
        ax4_twin.plot(df.index, prob_change * 100, linewidth=1, color='purple', alpha=0.7, label='Prob Change')
        ax4_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4_twin.fill_between(df.index, 0, prob_change * 100, where=(prob_change < 0), 
                              alpha=0.2, color='purple', label='Prob Decreasing')
        ax4_twin.set_ylabel('Probability Change (%)', color='purple')
        ax4_twin.tick_params(axis='y', labelcolor='purple')
        
        # Highlight trend acceleration zones
        for i in range(len(df)):
            if trend_accel_mask[i]:
                ax4.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.4, color='green', label='Acceleration' if i == np.where(trend_accel_mask)[0][0] else '')
        
        ax4.set_xlabel('Time')
        ax4.set_title('Rate of Change: Green = ADX Rising + Probability Decreasing', fontweight='bold')
        ax4.legend(loc='upper left')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'low_probability_analysis.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Chart saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  LOW PROBABILITY ZONE ANALYSIS")
    print("  Analyzing correlation with trending markets")
    print("="*60)
    
    analyzer = LowProbabilityAnalyzer(
        model_path='model_order_flow.keras',
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not analyzer.init_mt5():
        return
    
    if not analyzer.load_model():
        mt5.shutdown()
        return
    
    analyzer.analyze()
    
    mt5.shutdown()
    print(f"\n👋 Analysis Complete!")


if __name__ == "__main__":
    main()
