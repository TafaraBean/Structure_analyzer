import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MultiTimeframeBBBreakout:
    """
    Multi-timeframe Bollinger Band breakout detector.
    Identifies breakouts that occur across multiple BB periods simultaneously.
    """
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bb_periods = [20, 50, 100, 300, 500]  # Multiple BB lengths
        
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
    
    def fetch_data(self, bars=2000):
        """Fetch recent data."""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return df
    
    def calculate_multi_bb(self, df, std_dev=2):
        """
        Calculate Bollinger Bands for multiple periods.
        """
        metrics = pd.DataFrame(index=df.index)
        
        # Calculate BB for each period
        for period in self.bb_periods:
            upper, middle, lower = talib.BBANDS(
                df['close'].values, 
                timeperiod=period, 
                nbdevup=std_dev, 
                nbdevdn=std_dev
            )
            
            metrics[f'bb_upper_{period}'] = upper
            metrics[f'bb_middle_{period}'] = middle
            metrics[f'bb_lower_{period}'] = lower
            
            # Distance from bands
            metrics[f'dist_upper_{period}'] = (upper - df['close']) / df['close'] * 100
            metrics[f'dist_lower_{period}'] = (df['close'] - lower) / df['close'] * 100
            
            # Breakout detection for this period
            metrics[f'above_upper_{period}'] = metrics[f'dist_upper_{period}'] < 0
            metrics[f'below_lower_{period}'] = metrics[f'dist_lower_{period}'] < 0
        
        # CONFLUENCE: Breakout across ALL periods simultaneously
        # Bullish confluence: Price above upper band for ALL periods
        bullish_confluence = True
        for period in self.bb_periods:
            bullish_confluence = bullish_confluence & metrics[f'above_upper_{period}']
        metrics['bullish_confluence'] = bullish_confluence
        
        # Bearish confluence: Price below lower band for ALL periods
        bearish_confluence = True
        for period in self.bb_periods:
            bearish_confluence = bearish_confluence & metrics[f'below_lower_{period}']
        metrics['bearish_confluence'] = bearish_confluence
        
        # Count how many periods have breakout (strength indicator)
        metrics['bullish_strength'] = sum(metrics[f'above_upper_{period}'] for period in self.bb_periods)
        metrics['bearish_strength'] = sum(metrics[f'below_lower_{period}'] for period in self.bb_periods)
        
        return metrics
    
    def analyze(self, std_dev=2):
        """Analyze multi-timeframe BB breakouts."""
        print(f"\n{'='*60}")
        print(f"  MULTI-TIMEFRAME BB BREAKOUT DETECTOR")
        print(f"{'='*60}")
        print(f"\n⚙️  Parameters:")
        print(f"   BB Periods:  {self.bb_periods}")
        print(f"   Std Dev:     {std_dev}")
        
        # Fetch data
        df = self.fetch_data()
        if df is None:
            return
        
        # Calculate multi-BB metrics
        metrics = self.calculate_multi_bb(df, std_dev)
        
        # Statistics
        bullish_conf = metrics['bullish_confluence'].sum()
        bearish_conf = metrics['bearish_confluence'].sum()
        
        print(f"\n🚀 Confluence Breakouts (ALL {len(self.bb_periods)} periods):")
        print(f"   Bullish confluence:  {bullish_conf} bars ({bullish_conf/len(df)*100:.2f}%)")
        print(f"   Bearish confluence:  {bearish_conf} bars ({bearish_conf/len(df)*100:.2f}%)")
        print(f"   Total confluence:    {bullish_conf + bearish_conf} bars ({(bullish_conf + bearish_conf)/len(df)*100:.2f}%)")
        
        # Strength distribution
        print(f"\n💪 Breakout Strength Distribution:")
        for strength in range(1, len(self.bb_periods) + 1):
            bull_count = (metrics['bullish_strength'] == strength).sum()
            bear_count = (metrics['bearish_strength'] == strength).sum()
            if bull_count > 0 or bear_count > 0:
                print(f"   {strength}/{len(self.bb_periods)} periods: Bullish={bull_count}, Bearish={bear_count}")
        
        # Individual period breakouts
        print(f"\n📊 Individual Period Breakouts:")
        for period in self.bb_periods:
            bull = metrics[f'above_upper_{period}'].sum()
            bear = metrics[f'below_lower_{period}'].sum()
            print(f"   BB({period}): Bullish={bull} ({bull/len(df)*100:.1f}%), Bearish={bear} ({bear/len(df)*100:.1f}%)")
        
        # Plot
        self.plot_confluence(df, metrics)
    
    def plot_confluence(self, df, metrics):
        """Plot multi-timeframe BB with confluence highlights."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Plot 1: Price with all BB bands
        ax1.plot(df.index, df['close'], linewidth=2, color='black', label='Close', zorder=10)
        
        # Plot BB bands for each period with different colors
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        for i, period in enumerate(self.bb_periods):
            color = colors[i % len(colors)]
            ax1.plot(df.index, metrics[f'bb_upper_{period}'], 
                    linewidth=0.8, color=color, alpha=0.5, linestyle='--', label=f'BB({period}) Upper')
            ax1.plot(df.index, metrics[f'bb_lower_{period}'], 
                    linewidth=0.8, color=color, alpha=0.5, linestyle='--')
        
        # Highlight confluence zones
        for i in range(len(df)):
            if metrics['bullish_confluence'].iloc[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.4, color='green', 
                           label='Bullish Confluence' if i == np.where(metrics['bullish_confluence'])[0][0] else '')
            if metrics['bearish_confluence'].iloc[i]:
                ax1.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.4, color='red', 
                           label='Bearish Confluence' if i == np.where(metrics['bearish_confluence'])[0][0] else '')
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'Price with Multi-Timeframe BB ({self.bb_periods}) - Green/Red = Full Confluence', 
                     fontweight='bold', fontsize=12)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Bullish Strength (how many periods have bullish breakout)
        ax2.bar(df.index, metrics['bullish_strength'], width=0.8, color='green', alpha=0.6, label='Bullish Strength')
        ax2.axhline(y=len(self.bb_periods), color='darkgreen', linestyle='--', linewidth=2, 
                   label=f'Full Confluence ({len(self.bb_periods)}/{len(self.bb_periods)})')
        ax2.set_ylabel('# Periods')
        ax2.set_title('Bullish Breakout Strength (# of BB periods with breakout)', fontweight='bold', fontsize=11)
        ax2.set_ylim(0, len(self.bb_periods) + 0.5)
        ax2.legend(loc='upper left')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Bearish Strength
        ax3.bar(df.index, metrics['bearish_strength'], width=0.8, color='red', alpha=0.6, label='Bearish Strength')
        ax3.axhline(y=len(self.bb_periods), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Full Confluence ({len(self.bb_periods)}/{len(self.bb_periods)})')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('# Periods')
        ax3.set_title('Bearish Breakout Strength (# of BB periods with breakout)', fontweight='bold', fontsize=11)
        ax3.set_ylim(0, len(self.bb_periods) + 0.5)
        ax3.legend(loc='upper left')
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'multi_bb_confluence.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Chart saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  MULTI-TIMEFRAME BB CONFLUENCE DETECTOR")
    print("  Capture the most powerful breakouts")
    print("="*60)
    
    detector = MultiTimeframeBBBreakout(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not detector.init_mt5():
        return
    
    # Analyze
    detector.analyze(std_dev=2)
    
    mt5.shutdown()
    print(f"\n👋 Analysis Complete!")


if __name__ == "__main__":
    main()
