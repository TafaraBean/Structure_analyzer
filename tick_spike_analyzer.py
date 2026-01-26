import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.signal import find_peaks

# Load environment variables
load_dotenv()

def init_mt5():
    """Initialize MT5 connection."""
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    params = {}
    if path: 
        params["path"] = path
    
    if not mt5.initialize(**params):
        print(f"❌ MT5 Init failed: {mt5.last_error()}")
        return False
        
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
            print(f"❌ Login failed: {mt5.last_error()}")
            return False
    
    print(f"✅ Connected to MT5: {mt5.account_info().server}")
    return True

def get_hourly_ticks(symbol, hours=1):
    """Get all ticks from the last N hours."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    print(f"\n🔍 Fetching ticks from {start_time} to {end_time}...")
    
    ticks = mt5.copy_ticks_range(
        symbol,
        start_time,
        end_time,
        mt5.COPY_TICKS_ALL
    )
    
    if ticks is None or len(ticks) == 0:
        print(f"❌ No ticks found: {mt5.last_error()}")
        return None
    
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['time_msc'] = pd.to_datetime(df['time_msc'], unit='ms')
    
    print(f"✅ Found {len(df)} ticks")
    return df

def detect_price_spikes(df, method='zscore', threshold=2.5):
    """
    Detect aggressive price spikes using multiple methods.
    
    Methods:
    - zscore: Statistical outlier detection
    - velocity: Rate of price change
    - volatility: Local volatility spikes
    - combined: Combination of all methods
    """
    # Calculate price changes
    df['price_change'] = df['bid'].diff().abs()
    df['price_velocity'] = df['price_change'] / df['time_msc'].diff().dt.total_seconds().fillna(1)
    
    # Method 1: Z-Score (Statistical Outliers)
    if method in ['zscore', 'combined']:
        z_scores = np.abs(stats.zscore(df['price_change'].fillna(0)))
        spike_zscore = z_scores > threshold
    else:
        spike_zscore = np.zeros(len(df), dtype=bool)
    
    # Method 2: Velocity Spikes (Rapid price changes)
    if method in ['velocity', 'combined']:
        velocity_threshold = df['price_velocity'].quantile(0.95)
        spike_velocity = df['price_velocity'] > velocity_threshold
    else:
        spike_velocity = np.zeros(len(df), dtype=bool)
    
    # Method 3: Local Volatility Spikes
    if method in ['volatility', 'combined']:
        # Rolling standard deviation
        rolling_std = df['price_change'].rolling(window=50, min_periods=10).std()
        spike_volatility = df['price_change'] > (rolling_std * 3)
    else:
        spike_volatility = np.zeros(len(df), dtype=bool)
    
    # Combine methods
    if method == 'combined':
        spikes = spike_zscore | spike_velocity | spike_volatility
    elif method == 'zscore':
        spikes = spike_zscore
    elif method == 'velocity':
        spikes = spike_velocity
    else:  # volatility
        spikes = spike_volatility
    
    return spikes

def find_spike_zones(df, spikes, min_zone_duration_sec=5):
    """Group consecutive spikes into zones."""
    zones = []
    in_zone = False
    zone_start = None
    
    for i, is_spike in enumerate(spikes):
        if is_spike and not in_zone:
            # Start new zone
            in_zone = True
            zone_start = i
        elif not is_spike and in_zone:
            # End zone
            zone_end = i - 1
            zone_duration = (df['time_msc'].iloc[zone_end] - df['time_msc'].iloc[zone_start]).total_seconds()
            
            if zone_duration >= min_zone_duration_sec:
                zones.append({
                    'start_idx': zone_start,
                    'end_idx': zone_end,
                    'start_time': df['time_msc'].iloc[zone_start],
                    'end_time': df['time_msc'].iloc[zone_end],
                    'duration': zone_duration,
                    'max_price': df['bid'].iloc[zone_start:zone_end+1].max(),
                    'min_price': df['bid'].iloc[zone_start:zone_end+1].min(),
                    'price_range': df['bid'].iloc[zone_start:zone_end+1].max() - df['bid'].iloc[zone_start:zone_end+1].min(),
                    'tick_count': zone_end - zone_start + 1
                })
            in_zone = False
    
    return zones

def visualize_tick_spikes(df, spikes, zones, symbol):
    """Create comprehensive visualization of tick data with spike zones."""
    print(f"\n📊 Creating visualization...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    
    fig.suptitle(f'{symbol} - Tick Spike Analysis (Last Hour)\n{len(zones)} Spike Zones Detected', 
                 fontsize=14, color='white')
    
    # Panel 1: Price with spike zones
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df['time_msc'], df['bid'], color='cyan', linewidth=0.5, alpha=0.7, label='Bid Price')
    
    # Highlight spike zones
    for zone in zones:
        ax1.axvspan(zone['start_time'], zone['end_time'], 
                   color='red', alpha=0.3, label='Spike Zone' if zone == zones[0] else '')
    
    # Mark individual spikes
    spike_times = df.loc[spikes, 'time_msc']
    spike_prices = df.loc[spikes, 'bid']
    ax1.scatter(spike_times, spike_prices, color='yellow', s=20, zorder=5, alpha=0.6, label='Spike Points')
    
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_title('Bid Price with Spike Zones', fontsize=11, pad=10)
    
    # Panel 2: Price Velocity
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df['time_msc'], df['price_velocity'], color='orange', linewidth=0.8, label='Price Velocity')
    ax2.axhline(y=df['price_velocity'].quantile(0.95), color='red', linestyle='--', 
               linewidth=1, alpha=0.5, label='95th Percentile')
    ax2.fill_between(df['time_msc'], 0, df['price_velocity'], color='orange', alpha=0.2)
    ax2.set_ylabel('Velocity (pips/sec)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_title('Price Change Velocity', fontsize=11, pad=10)
    
    # Panel 3: Rolling Volatility
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    rolling_std = df['price_change'].rolling(window=50, min_periods=10).std()
    ax3.plot(df['time_msc'], rolling_std * 10000, color='lime', linewidth=1, label='Rolling Volatility')
    ax3.fill_between(df['time_msc'], 0, rolling_std * 10000, color='lime', alpha=0.2)
    ax3.set_ylabel('Volatility (pips)', fontsize=10)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_title('Local Volatility (50-tick window)', fontsize=11, pad=10)
    
    # Panel 4: Tick Frequency
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    # Calculate ticks per second in bins
    df['time_sec'] = df['time_msc'].dt.floor('S')
    tick_freq = df.groupby('time_sec').size()
    ax4.bar(tick_freq.index, tick_freq.values, color='purple', alpha=0.6, width=0.0005)
    ax4.set_ylabel('Ticks/Second', fontsize=10)
    ax4.set_xlabel('Time', fontsize=10)
    ax4.grid(True, alpha=0.2)
    ax4.set_title('Tick Frequency', fontsize=11, pad=10)
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'tick_spikes_{symbol}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Chart saved as {filename}")
    
    plt.show()
    print(f"📈 Visualization displayed")

def main():
    """Main execution."""
    if not init_mt5():
        return
    
    # Configuration
    SYMBOL = "EURUSDm"
    HOURS = 1
    DETECTION_METHOD = 'combined'  # 'zscore', 'velocity', 'volatility', 'combined'
    
    print(f"\n{'='*60}")
    print(f"  TICK SPIKE ANALYZER")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Duration: {HOURS} hour(s)")
    print(f"  Detection Method: {DETECTION_METHOD}")
    print(f"{'='*60}")
    
    # Get tick data
    df = get_hourly_ticks(SYMBOL, HOURS)
    
    if df is None:
        mt5.shutdown()
        return
    
    # Detect spikes
    print(f"\n🔬 Analyzing tick patterns...")
    spikes = detect_price_spikes(df, method=DETECTION_METHOD, threshold=2.5)
    
    # Find spike zones
    zones = find_spike_zones(df, spikes, min_zone_duration_sec=3)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Total Ticks: {len(df)}")
    print(f"   Spike Points: {spikes.sum()}")
    print(f"   Spike Zones: {len(zones)}")
    
    if zones:
        print(f"\n🎯 Top 5 Most Aggressive Spike Zones:")
        # Sort by price range
        sorted_zones = sorted(zones, key=lambda x: x['price_range'], reverse=True)[:5]
        
        for i, zone in enumerate(sorted_zones, 1):
            print(f"\n   Zone {i}:")
            print(f"      Time: {zone['start_time'].strftime('%H:%M:%S')} - {zone['end_time'].strftime('%H:%M:%S')}")
            print(f"      Duration: {zone['duration']:.1f} seconds")
            print(f"      Price Range: {zone['price_range']*10000:.1f} pips")
            print(f"      Tick Count: {zone['tick_count']}")
    
    # Visualize
    visualize_tick_spikes(df, spikes, zones, SYMBOL)
    
    # Optional: Save to CSV
    save_csv = input("\n💾 Save tick data with spike markers to CSV? (y/n): ").lower().strip()
    if save_csv == 'y':
        df['is_spike'] = spikes
        filename = f"tick_spikes_{SYMBOL}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved to {filename}")
    
    mt5.shutdown()
    print(f"\n👋 Disconnected from MT5")

if __name__ == "__main__":
    main()
