import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class SidewaysDetector:
    """Detect sideways/ranging market conditions."""
    
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
    
    def fetch_data(self, bars=3000):
        """Fetch recent data."""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return df
    
    def detect_sideways_zones(self, df, lookback=20, price_range_threshold=0.3, adx_threshold=20):
        """
        Detect sideways market zones using multiple methods.
        
        Methods:
        1. Price oscillation around mean (tight range)
        2. Low ADX (weak trend)
        3. Price staying within narrow bands
        
        Args:
            lookback: Rolling window size
            price_range_threshold: Max % range for sideways (default 0.3%)
            adx_threshold: Max ADX for sideways (default 20)
        """
        metrics = pd.DataFrame(index=df.index)
        
        # Method 1: Rolling price range (High-Low range as % of price)
        metrics['rolling_high'] = df['high'].rolling(lookback).max()
        metrics['rolling_low'] = df['low'].rolling(lookback).min()
        metrics['rolling_range_pct'] = ((metrics['rolling_high'] - metrics['rolling_low']) / df['close']) * 100
        
        # Method 2: Price change volatility (std of returns)
        metrics['price_change'] = df['close'].pct_change() * 100
        metrics['rolling_std'] = metrics['price_change'].rolling(lookback).std()
        
        # Method 3: ADX (trend strength - low ADX = sideways)
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['adx'] = adx
        
        # Method 4: Bollinger Band width (narrow bands = sideways)
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
        metrics['bb_width'] = ((upper - lower) / middle) * 100
        
        # Method 5: Linear regression slope (flat slope = sideways)
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope
        
        metrics['price_slope'] = df['close'].rolling(lookback).apply(calculate_slope, raw=False)
        metrics['abs_slope'] = np.abs(metrics['price_slope'])
        
        # Combine methods to identify sideways zones
        sideways_mask = (
            (metrics['rolling_range_pct'] > price_range_threshold) 
        )
        
        metrics['is_sideways'] = sideways_mask
        
        return metrics
    
    def analyze(self, lookback=20, price_range_threshold=0.2, adx_threshold=20):
        """Analyze and visualize sideways zones."""
        print(f"\n{'='*60}")
        print(f"  SIDEWAYS MARKET DETECTOR")
        print(f"{'='*60}")
        print(f"\n⚙️  Parameters:")
        print(f"   Lookback window:     {lookback} bars")
        print(f"   Price range thresh:  {price_range_threshold}%")
        print(f"   ADX threshold:       {adx_threshold}")
        
        # Fetch data
        df = self.fetch_data()
        if df is None:
            return
        
        # Detect sideways zones
        metrics = self.detect_sideways_zones(df, lookback, price_range_threshold, adx_threshold)
        
        # Calculate statistics
        sideways_bars = metrics['is_sideways'].sum()
        sideways_pct = sideways_bars / len(df) * 100
        
        print(f"\n📊 Sideways Zone Statistics:")
        print(f"   Total bars:        {len(df)}")
        print(f"   Sideways bars:     {sideways_bars} ({sideways_pct:.1f}%)")
        print(f"   Trending bars:     {len(df) - sideways_bars} ({100-sideways_pct:.1f}%)")
        
        # Analyze sideways characteristics
        sideways_data = metrics[metrics['is_sideways']]
        if len(sideways_data) > 0:
            print(f"\n🔍 Sideways Zone Characteristics:")
            print(f"   Avg price range:   {sideways_data['rolling_range_pct'].mean():.3f}%")
            print(f"   Avg ADX:           {sideways_data['adx'].mean():.1f}")
            print(f"   Avg volatility:    {sideways_data['rolling_std'].mean():.3f}%")
            print(f"   Avg BB width:      {sideways_data['bb_width'].mean():.3f}%")
        
        # Find longest sideways periods
        sideways_periods = self.find_consecutive_periods(metrics['is_sideways'])
        if sideways_periods:
            longest = max(sideways_periods, key=lambda x: x['length'])
            print(f"\n📏 Longest Sideways Period:")
            print(f"   Duration:          {longest['length']} bars")
            print(f"   Start:             {df.index[longest['start']]}")
            print(f"   End:               {df.index[longest['end']]}")
        
        # Plot
        self.plot_sideways_zones(df, metrics, price_range_threshold, adx_threshold, 0.02)
    
    def find_consecutive_periods(self, mask):
        """Find consecutive True periods in a boolean mask."""
        periods = []
        in_period = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_period:
                start = i
                in_period = True
            elif not val and in_period:
                periods.append({
                    'start': start,
                    'end': i - 1,
                    'length': i - start
                })
                in_period = False
        
        # Handle case where period extends to end
        if in_period:
            periods.append({
                'start': start,
                'end': len(mask) - 1,
                'length': len(mask) - start
            })
        
        return periods
    
    def plot_sideways_zones(self, df, metrics, price_range_threshold=0.2, adx_threshold=20, vol_threshold=0.02):
        """Plot price with sideways zones highlighted."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Price with sideways zones
        ax1.plot(df.index, df['close'], linewidth=1, color='black', label='Price')
        
        # Highlight sideways zones
        for i in range(len(df)):
            if metrics['is_sideways'].iloc[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='yellow', label='Sideways' if i == np.where(metrics['is_sideways'])[0][0] else '')
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price Chart with Sideways Zones (Yellow)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Rolling price range
        ax2.plot(df.index, metrics['rolling_range_pct'], linewidth=1, color='blue', label='Rolling Range %')
        ax2.axhline(y=price_range_threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({price_range_threshold}%)')
        ax2.fill_between(df.index, 0, metrics['rolling_range_pct'], 
                         where=(metrics['rolling_range_pct'] < price_range_threshold), alpha=0.2, color='yellow')
        ax2.set_ylabel('Range (%)')
        ax2.set_title('Rolling Price Range (Tight = Sideways)', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: ADX
        ax3.plot(df.index, metrics['adx'], linewidth=1, color='orange', label='ADX')
        ax3.axhline(y=adx_threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({adx_threshold})')
        ax3.fill_between(df.index, 0, metrics['adx'], 
                         where=(metrics['adx'] < adx_threshold), alpha=0.2, color='yellow')
        ax3.set_ylabel('ADX')
        ax3.set_title('ADX - Trend Strength (Low = Sideways)', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Volatility (std of returns)
        ax4.plot(df.index, metrics['rolling_std'], linewidth=1, color='purple', label='Rolling Std')
        ax4.axhline(y=vol_threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({vol_threshold}%)')
        ax4.fill_between(df.index, 0, metrics['rolling_std'], 
                         where=(metrics['rolling_std'] < vol_threshold), alpha=0.2, color='yellow')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Volatility (%)')
        ax4.set_title('Price Volatility (Low = Sideways)', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'sideways_detection.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Chart saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  SIDEWAYS MARKET DETECTOR")
    print("  Identify ranging/consolidation zones")
    print("="*60)
    
    detector = SidewaysDetector(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not detector.init_mt5():
        return
    
    # Analyze with customizable parameters
    detector.analyze(
        lookback=4,                # 20-bar rolling window
        price_range_threshold=0.5,  # Max 0.2% range (tighter than before)
        adx_threshold=20            # ADX < 20
    )
    
    mt5.shutdown()
    print(f"\n👋 Analysis Complete!")


if __name__ == "__main__":
    main()
