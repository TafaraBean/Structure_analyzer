import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MomentumAnalyzer:
    """Analyze high momentum zones (high price change + high ADX)."""
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.symbol = symbol
        self.timeframe = timeframe
        
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
    
    def calculate_momentum_metrics(self, df):
        """Calculate price change and ADX."""
        metrics = pd.DataFrame(index=df.index)
        
        # Price percentage change over different periods
        for period in [5, 10, 20]:
            metrics[f'price_change_{period}'] = df['close'].pct_change(period) * 100  # Convert to %
        
        # Absolute price change (for magnitude, regardless of direction)
        for period in [5, 10, 20]:
            metrics[f'abs_price_change_{period}'] = np.abs(df['close'].pct_change(period)) * 100
        
        # ADX (trend strength)
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['adx'] = adx
        
        # ADX percentile (relative strength)
        metrics['adx_percentile'] = pd.Series(adx).rolling(100).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5
        )
        
        # Trend direction
        ema_20 = talib.EMA(df['close'].values, 20)
        ema_50 = talib.EMA(df['close'].values, 50)
        metrics['trend_up'] = ema_20 > ema_50
        metrics['trend_down'] = ema_20 < ema_50
        
        metrics.fillna(0, inplace=True)
        return metrics
    
    def analyze(self, price_threshold=1.0, adx_threshold=30):
        """
        Analyze high momentum zones.
        
        Args:
            price_threshold: Minimum absolute price change % (default 1.0%)
            adx_threshold: Minimum ADX value (default 30)
        """
        print(f"\n{'='*60}")
        print(f"  HIGH MOMENTUM ZONE ANALYSIS")
        print(f"  Price Change > {price_threshold}% + ADX > {adx_threshold}")
        print(f"{'='*60}")
        
        # Fetch test data
        df = self.fetch_test_data()
        if df is None:
            return
        
        # Calculate metrics
        metrics = self.calculate_momentum_metrics(df)
        
        # Identify high momentum zones (using 10-period price change)
        high_momentum_mask = (metrics['abs_price_change_10'] > price_threshold) & (metrics['adx'] > adx_threshold)
        high_momentum_count = high_momentum_mask.sum()
        
        # Separate bullish and bearish momentum
        bullish_momentum = high_momentum_mask & (metrics['price_change_10'] > 0)
        bearish_momentum = high_momentum_mask & (metrics['price_change_10'] < 0)
        
        print(f"\n📊 High Momentum Zones:")
        print(f"   Total bars: {len(df)}")
        print(f"   High momentum bars: {high_momentum_count} ({high_momentum_count/len(df)*100:.1f}%)")
        print(f"   Bullish momentum: {bullish_momentum.sum()} ({bullish_momentum.sum()/len(df)*100:.1f}%)")
        print(f"   Bearish momentum: {bearish_momentum.sum()} ({bearish_momentum.sum()/len(df)*100:.1f}%)")
        
        # Analyze characteristics of high momentum zones
        if high_momentum_count > 0:
            high_mom_indices = np.where(high_momentum_mask)[0]
            
            avg_price_change = metrics['abs_price_change_10'].iloc[high_mom_indices].mean()
            avg_adx = metrics['adx'].iloc[high_mom_indices].mean()
            max_price_change = metrics['abs_price_change_10'].iloc[high_mom_indices].max()
            max_adx = metrics['adx'].iloc[high_mom_indices].max()
            
            print(f"\n🔥 High Momentum Characteristics:")
            print(f"   Average price change: {avg_price_change:.2f}%")
            print(f"   Average ADX: {avg_adx:.1f}")
            print(f"   Max price change: {max_price_change:.2f}%")
            print(f"   Max ADX: {max_adx:.1f}")
            
            # Compare to overall dataset
            overall_avg_change = metrics['abs_price_change_10'].mean()
            overall_avg_adx = metrics['adx'].mean()
            
            print(f"\n📈 Comparison to Overall Dataset:")
            print(f"   Overall avg price change: {overall_avg_change:.2f}%")
            print(f"   Overall avg ADX: {overall_avg_adx:.1f}")
            print(f"\n💡 Insight:")
            print(f"   High momentum zones have {avg_price_change/overall_avg_change:.1f}x stronger price moves")
            print(f"   High momentum zones have {avg_adx/overall_avg_adx:.1f}x stronger trends")
        
        # Create visualization
        self.plot_analysis(df, metrics, high_momentum_mask, bullish_momentum, bearish_momentum, 
                          price_threshold, adx_threshold)
    
    def plot_analysis(self, df, metrics, high_momentum_mask, bullish_momentum, bearish_momentum,
                     price_threshold, adx_threshold):
        """Plot price chart with high momentum zones highlighted."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        # Subplot 1: Price with momentum zones highlighted
        ax1.plot(df.index, df['close'], linewidth=1, color='black', label='Price')
        
        # Highlight bullish momentum zones (green)
        for i in range(len(df)):
            if bullish_momentum[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='green', label='Bullish Momentum' if i == np.where(bullish_momentum)[0][0] else '')
        
        # Highlight bearish momentum zones (red)
        for i in range(len(df)):
            if bearish_momentum[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='red', label='Bearish Momentum' if i == np.where(bearish_momentum)[0][0] else '')
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'Price Chart: Green=Bullish Momentum, Red=Bearish Momentum (|ΔP|>{price_threshold}% + ADX>{adx_threshold})', 
                     fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        # Subplot 2: Price Change %
        ax2.plot(df.index, metrics['price_change_10'], linewidth=1, color='blue', label='10-Period Price Change %')
        ax2.axhline(y=price_threshold, color='green', linestyle='--', alpha=0.5, label=f'+{price_threshold}% Threshold')
        ax2.axhline(y=-price_threshold, color='red', linestyle='--', alpha=0.5, label=f'-{price_threshold}% Threshold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Fill areas
        ax2.fill_between(df.index, 0, metrics['price_change_10'], 
                        where=(metrics['price_change_10'] > price_threshold), 
                        alpha=0.2, color='green', label='Strong Bullish')
        ax2.fill_between(df.index, 0, metrics['price_change_10'], 
                        where=(metrics['price_change_10'] < -price_threshold), 
                        alpha=0.2, color='red', label='Strong Bearish')
        
        # Highlight high momentum zones
        for i in range(len(df)):
            if high_momentum_mask[i]:
                color = 'green' if metrics['price_change_10'].iloc[i] > 0 else 'red'
                ax2.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.2, color=color)
        
        ax2.set_ylabel('Price Change (%)')
        ax2.set_title('10-Period Price Change %', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(alpha=0.3)
        
        # Subplot 3: ADX
        ax3.plot(df.index, metrics['adx'], linewidth=1, color='orange', label='ADX')
        ax3.axhline(y=adx_threshold, color='red', linestyle='--', alpha=0.5, label=f'ADX > {adx_threshold}')
        ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.3, label='Strong Trend (25)')
        ax3.fill_between(df.index, 0, metrics['adx'], where=(metrics['adx'] > adx_threshold), 
                         alpha=0.2, color='red', label='High ADX')
        
        # Highlight high momentum zones
        for i in range(len(df)):
            if high_momentum_mask[i]:
                color = 'green' if metrics['price_change_10'].iloc[i] > 0 else 'red'
                ax3.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.2, color=color)
        
        ax3.set_ylabel('ADX')
        ax3.set_xlabel('Time')
        ax3.set_title('Trend Strength (ADX)', fontweight='bold')
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'high_momentum_analysis.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Chart saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  HIGH MOMENTUM ZONE ANALYSIS")
    print("  Analyzing Price Change + ADX Correlation")
    print("="*60)
    
    analyzer = MomentumAnalyzer(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not analyzer.init_mt5():
        return
    
    # Analyze with customizable thresholds
    analyzer.analyze(
        price_threshold=0.10,  # 1% price change
        adx_threshold=20      # ADX > 30
    )
    
    mt5.shutdown()
    print(f"\n👋 Analysis Complete!")


if __name__ == "__main__":
    main()
