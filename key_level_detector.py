import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches

load_dotenv()

class KeyLevelDetector:
    """
    Detect key support/resistance levels using clustering.
    More tractable than predicting zone formation.
    """
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15, bars=3000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.df = None
        self.key_levels = []
        
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
    
    def fetch_data(self):
        """Fetch historical data."""
        print(f"📊 Fetching {self.bars} bars for {self.symbol}...")
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.bars)
        
        if rates is None:
            print(f"❌ Failed to fetch data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars")
        self.df = df
        return df
    
    def find_swing_points(self, window=5):
        """Find swing highs and lows."""
        print(f"\n🔍 Finding swing points (window={window})...")
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(self.df) - window):
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            
            # Swing high
            left_highs = self.df['high'].iloc[i-window:i]
            right_highs = self.df['high'].iloc[i+1:i+window+1]
            
            if (current_high >= left_highs.max()) and (current_high >= right_highs.max()):
                swing_highs.append({
                    'price': current_high,
                    'time': self.df.index[i],
                    'idx': i,
                    'type': 'resistance'
                })
            
            # Swing low
            left_lows = self.df['low'].iloc[i-window:i]
            right_lows = self.df['low'].iloc[i+1:i+window+1]
            
            if (current_low <= left_lows.min()) and (current_low <= right_lows.min()):
                swing_lows.append({
                    'price': current_low,
                    'time': self.df.index[i],
                    'idx': i,
                    'type': 'support'
                })
        
        print(f"✅ Found {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
        
        return swing_highs + swing_lows
    
    def find_key_levels_percentile(self, swing_points, n_levels=10, tolerance=0.0005):
        """
        Find key levels using percentile approach instead of clustering.
        
        This works better when market has wide range but DBSCAN groups everything.
        
        Args:
            n_levels: Number of levels to find
            tolerance: Price tolerance for grouping nearby swings (0.05%)
        """
        print(f"\n📊 Finding key levels using percentile approach...")
        
        if len(swing_points) < 2:
            print(f"⚠️  Not enough swing points")
            return []
        
        # Separate highs and lows
        highs = [p for p in swing_points if p['type'] == 'resistance']
        lows = [p for p in swing_points if p['type'] == 'support']
        
        key_levels = []
        
        # Process resistance levels (from swing highs)
        if highs:
            high_prices = sorted([p['price'] for p in highs])
            # Find levels at percentiles
            percentiles = np.linspace(0, 100, min(n_levels//2 + 1, len(high_prices)))
            
            for pct in percentiles:
                level_price = np.percentile(high_prices, pct)
                
                # Count touches within tolerance
                touches = sum(1 for p in high_prices if abs(p - level_price) / level_price <= tolerance)
                
                if touches >= 2:  # At least 2 touches
                    key_levels.append({
                        'price': level_price,
                        'type': 'resistance',
                        'touches': touches,
                        'strength': touches,
                        'std': 0,
                        'points': [p for p in highs if abs(p['price'] - level_price) / level_price <= tolerance]
                    })
        
        # Process support levels (from swing lows)
        if lows:
            low_prices = sorted([p['price'] for p in lows])
            percentiles = np.linspace(0, 100, min(n_levels//2 + 1, len(low_prices)))
            
            for pct in percentiles:
                level_price = np.percentile(low_prices, pct)
                
                # Count touches within tolerance
                touches = sum(1 for p in low_prices if abs(p - level_price) / level_price <= tolerance)
                
                if touches >= 2:
                    key_levels.append({
                        'price': level_price,
                        'type': 'support',
                        'touches': touches,
                        'strength': touches,
                        'std': 0,
                        'points': [p for p in lows if abs(p['price'] - level_price) / level_price <= tolerance]
                    })
        
        # Remove duplicates (levels too close to each other)
        filtered_levels = []
        key_levels.sort(key=lambda x: x['price'])
        
        for level in key_levels:
            # Check if too close to existing level
            too_close = any(abs(level['price'] - existing['price']) / level['price'] < tolerance * 2 
                          for existing in filtered_levels)
            if not too_close:
                filtered_levels.append(level)
        
        # Sort by strength
        filtered_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        print(f"✅ Found {len(filtered_levels)} key levels")
        for i, level in enumerate(filtered_levels[:10], 1):
            print(f"   {i}. {level['type'].upper()} @ {level['price']:.5f} "
                  f"({level['touches']} touches, strength: {level['strength']:.2f})")
        
        self.key_levels = filtered_levels
        return filtered_levels
    
    def create_proximity_labels(self, proximity_threshold=0.001):
        """
        Create labels: Is price near a key level?
        
        This is much more tractable than predicting zone formation.
        """
        print(f"\n🏷️  Creating proximity labels (threshold={proximity_threshold*100:.1f}%)...")
        
        self.df['near_support'] = 0
        self.df['near_resistance'] = 0
        self.df['at_key_level'] = 0
        self.df['level_strength'] = 0.0
        
        for _, row in self.df.iterrows():
            current_price = row['close']
            
            # Check proximity to each key level
            for level in self.key_levels:
                distance = abs(current_price - level['price']) / level['price']
                
                if distance <= proximity_threshold:
                    self.df.loc[row.name, 'at_key_level'] = 1
                    self.df.loc[row.name, 'level_strength'] = level['strength']
                    
                    if level['type'] == 'support':
                        self.df.loc[row.name, 'near_support'] = 1
                    else:
                        self.df.loc[row.name, 'near_resistance'] = 1
        
        at_level_pct = self.df['at_key_level'].mean() * 100
        near_support_pct = self.df['near_support'].mean() * 100
        near_resistance_pct = self.df['near_resistance'].mean() * 100
        
        print(f"✅ Labels created:")
        print(f"   At key level: {self.df['at_key_level'].sum()} ({at_level_pct:.1f}%)")
        print(f"   Near support: {self.df['near_support'].sum()} ({near_support_pct:.1f}%)")
        print(f"   Near resistance: {self.df['near_resistance'].sum()} ({near_resistance_pct:.1f}%)")
        
        return self.df
    
    def visualize_levels(self, sample_bars=500):
        """Visualize key levels on price chart."""
        print(f"\n📈 Creating visualization...")
        
        # Use last N bars
        start_idx = max(0, len(self.df) - sample_bars)
        df_sample = self.df.iloc[start_idx:]
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        fig.suptitle(f'{self.symbol} - Key Support/Resistance Levels', 
                     fontsize=14, color='white')
        
        # Panel 1: Price with key levels
        ax1.plot(df_sample.index, df_sample['close'], color='white', linewidth=1, label='Close')
        
        # Plot key levels
        for level in self.key_levels:
            color = 'green' if level['type'] == 'support' else 'red'
            alpha = min(0.3 + (level['strength'] / 10), 0.8)
            
            ax1.axhline(y=level['price'], color=color, linestyle='--', 
                       linewidth=2, alpha=alpha, 
                       label=f"{level['type'].capitalize()} @ {level['price']:.5f}")
            
            # Add zone around level
            zone_height = level['price'] * 0.001
            ax1.axhspan(level['price'] - zone_height, level['price'] + zone_height,
                       color=color, alpha=0.1)
        
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='upper left', fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.2)
        ax1.set_title('Price Chart with Key Levels', fontsize=11, pad=10)
        
        # Panel 2: Proximity indicator
        at_level = df_sample['at_key_level'].values
        ax2.fill_between(df_sample.index, 0, at_level, color='yellow', alpha=0.5)
        ax2.set_ylabel('At Key Level', fontsize=10)
        ax2.set_xlabel('Time', fontsize=10)
        ax2.set_ylim(0, 1.2)
        ax2.grid(True, alpha=0.2)
        ax2.set_title('Key Level Proximity Indicator', fontsize=11, pad=10)
        
        plt.tight_layout()
        
        filename = 'key_levels_chart.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Chart saved as {filename}")
        
        # plt.show()  # Commented out to allow script to complete
        print(f"📊 Chart saved (not displayed to allow script completion)")
    
    def export_labels(self, filename='key_level_labels.csv'):
        """Export labels for ML training."""
        self.df.to_csv(filename)
        print(f"\n💾 Labels exported to {filename}")
        print(f"   Ready for neural network training!")

def main():
    print("="*60)
    print("  KEY LEVEL DETECTOR")
    print("  Identify support/resistance levels for reversal entries")
    print("="*60)
    
    detector = KeyLevelDetector('EURUSDm', mt5.TIMEFRAME_M15, bars=3000)
    
    if not detector.init_mt5():
        return
    
    # Fetch data
    detector.fetch_data()
    
    # Find swing points
    swing_points = detector.find_swing_points(window=5)
    
    # Find key levels using percentile approach (better for wide ranges)
    key_levels = detector.find_key_levels_percentile(swing_points, n_levels=10, tolerance=0.0005)
    
    if key_levels:
        # Create proximity labels (tighter threshold for better balance)
        detector.create_proximity_labels(proximity_threshold=0.0003)
        
        # Visualize
        detector.visualize_levels(sample_bars=500)
        
        # Export
        detector.export_labels()
        
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"\n✅ Key Level Detection Complete")
        print(f"   Found {len(key_levels)} key levels")
        print(f"   Labels created for ML training")
        print(f"\n📊 Next Step:")
        print(f"   Train NN to predict: 'Is price at a key level?'")
        print(f"   This is much more tractable than zone formation!")
    else:
        print(f"\n⚠️  No key levels found. Try adjusting parameters.")
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    main()
