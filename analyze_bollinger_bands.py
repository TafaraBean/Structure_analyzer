import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class BollingerBandAnalyzer:
    """Analyze Bollinger Bands and distance from bands."""
    
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
    
    def fetch_data(self, bars=1000):
        """Fetch recent data."""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return df
    
    def calculate_bb_distance(self, df, period=20, std_dev=2):
        """
        Calculate Bollinger Bands and distance indicators.
        
        Returns:
            DataFrame with BB bands and distance metrics
        """
        metrics = pd.DataFrame(index=df.index)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'].values, 
            timeperiod=period, 
            nbdevup=std_dev, 
            nbdevdn=std_dev
        )
        
        metrics['bb_upper'] = upper
        metrics['bb_middle'] = middle
        metrics['bb_lower'] = lower
        
        # Distance from upper band (positive = below band, negative = above band)
        metrics['dist_from_upper'] = (upper - df['close']) / df['close'] * 100
        
        # Distance from lower band (positive = above band, negative = below band)
        metrics['dist_from_lower'] = (df['close'] - lower) / df['close'] * 100
        
        # Double smooth the distance indicators (EMA of EMA for smoothness)
        smooth_period = 5
        
        # First smoothing
        metrics['dist_upper_smooth1'] = metrics['dist_from_upper'].ewm(span=smooth_period, adjust=False).mean()
        metrics['dist_lower_smooth1'] = metrics['dist_from_lower'].ewm(span=smooth_period, adjust=False).mean()
        
        # Second smoothing (double smooth)
        metrics['dist_upper_smooth'] = metrics['dist_upper_smooth1'].ewm(span=smooth_period, adjust=False).mean()
        metrics['dist_lower_smooth'] = metrics['dist_lower_smooth1'].ewm(span=smooth_period, adjust=False).mean()
        
        # First derivative (velocity/rate of change)
        metrics['dist_upper_velocity'] = metrics['dist_upper_smooth'].diff()
        metrics['dist_lower_velocity'] = metrics['dist_lower_smooth'].diff()
        
        # Second derivative (acceleration/rate of change of velocity)
        metrics['dist_upper_acceleration'] = metrics['dist_upper_velocity'].diff()
        metrics['dist_lower_acceleration'] = metrics['dist_lower_velocity'].diff()
        
        # Bollinger Band width (volatility indicator)
        metrics['bb_width'] = (upper - lower) / middle * 100
        
        # Position within bands (0 = at lower, 100 = at upper)
        metrics['bb_position'] = ((df['close'] - lower) / (upper - lower)) * 100
        
        # BULLISH BREAKOUT: Price above upper band + distance from lower increasing (positive velocity)
        # (Price breaking out upward and accelerating away from lower band)
        metrics['bullish_breakout'] = (
            (metrics['dist_from_upper'] < 0) &  # Price above upper band
            (metrics['dist_lower_velocity'] > 0)   # Distance from lower increasing
        )
        
        # BEARISH BREAKOUT: Price below lower band + distance from upper increasing (positive velocity)
        # (Price breaking out downward and accelerating away from upper band)
        metrics['bearish_breakout'] = (
            (metrics['dist_from_lower'] < 0) &   # Price below lower band
            (metrics['dist_upper_velocity'] > 0)   # Distance from upper increasing
        )
        
        return metrics
    
    def analyze(self, period=20, std_dev=2):
        """Analyze Bollinger Bands."""
        print(f"\n{'='*60}")
        print(f"  BOLLINGER BAND ANALYZER")
        print(f"{'='*60}")
        print(f"\n⚙️  Parameters:")
        print(f"   Period:     {period}")
        print(f"   Std Dev:    {std_dev}")
        
        # Fetch data
        df = self.fetch_data(15000)
        if df is None:
            return
        
        # Calculate BB metrics
        metrics = self.calculate_bb_distance(df, period, std_dev)
        
        # Statistics
        print(f"\n📊 Bollinger Band Statistics:")
        print(f"   Avg BB width:           {metrics['bb_width'].mean():.3f}%") 
        print(f"   Current BB width:       {metrics['bb_width'].iloc[-1]:.3f}%")
        print(f"   Avg dist from upper:    {metrics['dist_from_upper'].mean():.3f}%")
        print(f"   Avg dist from lower:    {metrics['dist_from_lower'].mean():.3f}%")
        print(f"   Current position:       {metrics['bb_position'].iloc[-1]:.1f}% (0=lower, 100=upper)")
        
        # Extreme readings
        upper_touches = (metrics['dist_from_upper'] < 0).sum()
        lower_touches = (metrics['dist_from_lower'] < 0).sum()
        
        print(f"\n🎯 Band Touches:")
        print(f"   Upper band touches:     {upper_touches} ({upper_touches/len(df)*100:.1f}%)")
        print(f"   Lower band touches:     {lower_touches} ({lower_touches/len(df)*100:.1f}%)")
        
        # Breakout zones
        bullish_breakouts = metrics['bullish_breakout'].sum()
        bearish_breakouts = metrics['bearish_breakout'].sum()
        
        print(f"\n🚀 Breakout Zones:")
        print(f"   Bullish breakouts:      {bullish_breakouts} ({bullish_breakouts/len(df)*100:.1f}%)")
        print(f"   Bearish breakouts:      {bearish_breakouts} ({bearish_breakouts/len(df)*100:.1f}%)")
        print(f"   Total breakouts:        {bullish_breakouts + bearish_breakouts} ({(bullish_breakouts + bearish_breakouts)/len(df)*100:.1f}%)")
        
        # Plot
        self.plot_bb_analysis(df, metrics)
    
    def plot_bb_analysis(self, df, metrics):
        """Plot Bollinger Bands and distance indicators with derivatives."""
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(5, 1, hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        
        # Plot 1: Price with Bollinger Bands
        ax1.plot(df.index, df['close'], linewidth=1.5, color='black', label='Close', zorder=3)
        ax1.plot(df.index, metrics['bb_upper'], linewidth=1, color='red', alpha=0.7, label='Upper Band', linestyle='--')
        ax1.plot(df.index, metrics['bb_middle'], linewidth=1, color='blue', alpha=0.7, label='Middle (SMA)')
        ax1.plot(df.index, metrics['bb_lower'], linewidth=1, color='green', alpha=0.7, label='Lower Band', linestyle='--')
        
        # Fill between bands
        ax1.fill_between(df.index, metrics['bb_upper'], metrics['bb_lower'], alpha=0.1, color='gray')
        
        # Highlight bullish breakout zones (green)
        for i in range(len(df)):
            if metrics['bullish_breakout'].iloc[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='green', label='Bullish Breakout' if i == np.where(metrics['bullish_breakout'])[0][0] else '')
        
        # Highlight bearish breakout zones (red)
        for i in range(len(df)):
            if metrics['bearish_breakout'].iloc[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color='red', label='Bearish Breakout' if i == np.where(metrics['bearish_breakout'])[0][0] else '')
        
        ax1.set_ylabel('Price')
        ax1.set_title('Price with Bollinger Bands (Green=Bullish, Red=Bearish)', fontweight='bold', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Smoothed Distance from Upper Band
        ax2.plot(df.index, metrics['dist_from_upper'], linewidth=0.5, color='red', alpha=0.3, label='Raw')
        ax2.plot(df.index, metrics['dist_upper_smooth'], linewidth=2, color='red', label='Double Smoothed')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax2.fill_between(df.index, 0, metrics['dist_upper_smooth'], 
                         where=(metrics['dist_upper_smooth'] < 0), alpha=0.3, color='red')
        
        ax2.set_ylabel('Distance (%)')
        ax2.set_title('Distance from Upper Band (Smoothed)', fontweight='bold', fontsize=11)
        ax2.legend(loc='upper left')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Smoothed Distance from Lower Band
        ax3.plot(df.index, metrics['dist_from_lower'], linewidth=0.5, color='green', alpha=0.3, label='Raw')
        ax3.plot(df.index, metrics['dist_lower_smooth'], linewidth=2, color='green', label='Double Smoothed')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax3.fill_between(df.index, 0, metrics['dist_lower_smooth'], 
                         where=(metrics['dist_lower_smooth'] < 0), alpha=0.3, color='red')
        
        ax3.set_ylabel('Distance (%)')
        ax3.set_title('Distance from Lower Band (Smoothed)', fontweight='bold', fontsize=11)
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)
        
        # Plot 4: First Derivative (Velocity)
        ax4.plot(df.index, metrics['dist_upper_velocity'], linewidth=1, color='red', alpha=0.7, label='Upper Velocity')
        ax4.plot(df.index, metrics['dist_lower_velocity'], linewidth=1, color='green', alpha=0.7, label='Lower Velocity')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax4.fill_between(df.index, 0, metrics['dist_upper_velocity'], 
                         where=(metrics['dist_upper_velocity'] > 0), alpha=0.2, color='red')
        ax4.fill_between(df.index, 0, metrics['dist_lower_velocity'], 
                         where=(metrics['dist_lower_velocity'] > 0), alpha=0.2, color='green')
        
        ax4.set_ylabel('Velocity')
        ax4.set_title('First Derivative - Velocity (Rate of Change)', fontweight='bold', fontsize=11)
        ax4.legend(loc='upper left')
        ax4.grid(alpha=0.3)
        
        # Plot 5: Second Derivative (Acceleration)
        ax5.plot(df.index, metrics['dist_upper_acceleration'], linewidth=1, color='red', alpha=0.7, label='Upper Acceleration')
        ax5.plot(df.index, metrics['dist_lower_acceleration'], linewidth=1, color='green', alpha=0.7, label='Lower Acceleration')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax5.fill_between(df.index, 0, metrics['dist_upper_acceleration'], 
                         where=(metrics['dist_upper_acceleration'] > 0), alpha=0.2, color='red')
        ax5.fill_between(df.index, 0, metrics['dist_lower_acceleration'], 
                         where=(metrics['dist_lower_acceleration'] > 0), alpha=0.2, color='green')
        
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Acceleration')
        ax5.set_title('Second Derivative - Acceleration (Rate of Change of Velocity)', fontweight='bold', fontsize=11)
        ax5.legend(loc='upper left')
        ax5.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'bollinger_band_analysis.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Chart saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  BOLLINGER BAND ANALYZER")
    print("  Distance from Upper/Lower Bands")
    print("="*60)
    
    analyzer = BollingerBandAnalyzer(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not analyzer.init_mt5():
        return
    
    # Analyze with customizable parameters
    analyzer.analyze(
        period=500,    # BB period
        std_dev=2     # Standard deviations
    )
    
    mt5.shutdown()
    print(f"\n👋 Analysis Complete!")


if __name__ == "__main__":
    main()
