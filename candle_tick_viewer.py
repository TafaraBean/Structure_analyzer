import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load environment variables
load_dotenv()

def permutation_entropy(series, order=3, delay=1):
    """Calculate permutation entropy of a time series."""
    import math
    
    vals = series.values if hasattr(series, 'values') else series
    n = len(vals)
    
    if n < order * delay:
        return np.nan
    
    # Create permutation patterns
    partitions = np.array([vals[j:j+order*delay:delay] for j in range(n - order * delay + 1)])
    ords = np.argsort(partitions, axis=1)
    
    # Count unique patterns
    _, counts = np.unique(ords, axis=0, return_counts=True)
    probs = counts / len(ords)
    
    # Calculate entropy
    pe = -np.sum(probs * np.log2(probs + 1e-9))
    
    # Normalize by maximum entropy
    max_entropy = np.log2(math.factorial(order))
    
    return pe / max_entropy if max_entropy > 0 else 0

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

def get_candle_ticks(symbol, timeframe, candle_index=0):
    """
    Get all ticks from a specific candle.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSDm')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M15)
        candle_index: Index of candle (0 = most recent, 1 = previous, etc.)
    
    Returns:
        DataFrame with tick data
    """
    # Get the candle
    rates = mt5.copy_rates_from_pos(symbol, timeframe, candle_index, 1)
    
    if rates is None or len(rates) == 0:
        print(f"❌ Failed to get candle data: {mt5.last_error()}")
        return None
    
    candle = rates[0]
    candle_time = pd.to_datetime(candle['time'], unit='s')
    
    # Determine candle duration based on timeframe
    timeframe_minutes = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H4: 240,
        mt5.TIMEFRAME_D1: 1440,
    }
    
    duration_minutes = timeframe_minutes.get(timeframe, 15)
    candle_end = candle_time + timedelta(minutes=duration_minutes)
    
    print(f"\n📊 Candle Info:")
    print(f"   Time: {candle_time}")
    print(f"   Open: {candle['open']:.5f}")
    print(f"   High: {candle['high']:.5f}")
    print(f"   Low: {candle['low']:.5f}")
    print(f"   Close: {candle['close']:.5f}")
    print(f"   Tick Volume: {candle['tick_volume']}")
    print(f"   Duration: {duration_minutes} minutes")
    
    # Get ticks for this candle period
    print(f"\n🔍 Fetching ticks from {candle_time} to {candle_end}...")
    
    ticks = mt5.copy_ticks_range(
        symbol,
        candle_time,
        candle_end,
        mt5.COPY_TICKS_ALL
    )
    
    if ticks is None or len(ticks) == 0:
        print(f"❌ No ticks found: {mt5.last_error()}")
        return None
    
    # Convert to DataFrame
    df_ticks = pd.DataFrame(ticks)
    df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
    df_ticks['time_msc'] = pd.to_datetime(df_ticks['time_msc'], unit='ms')
    
    print(f"✅ Found {len(df_ticks)} ticks")
    
    return df_ticks, candle

def analyze_ticks(df_ticks, candle):
    """Analyze tick data and print statistics."""
    if df_ticks is None or len(df_ticks) == 0:
        return
    
    print(f"\n📈 Tick Analysis:")
    print(f"   Total Ticks: {len(df_ticks)}")
    print(f"   Bid Ticks: {(df_ticks['flags'] & mt5.TICK_FLAG_BID).sum()}")
    print(f"   Ask Ticks: {(df_ticks['flags'] & mt5.TICK_FLAG_ASK).sum()}")
    print(f"   Buy Ticks: {(df_ticks['flags'] & mt5.TICK_FLAG_BUY).sum()}")
    print(f"   Sell Ticks: {(df_ticks['flags'] & mt5.TICK_FLAG_SELL).sum()}")
    
    print(f"\n💰 Price Range:")
    print(f"   Highest Bid: {df_ticks['bid'].max():.5f}")
    print(f"   Lowest Bid: {df_ticks['bid'].min():.5f}")
    print(f"   Highest Ask: {df_ticks['ask'].max():.5f}")
    print(f"   Lowest Ask: {df_ticks['ask'].min():.5f}")
    
    # Verify candle OHLC
    print(f"\n✔️  Candle Verification:")
    first_bid = df_ticks.iloc[0]['bid']
    last_bid = df_ticks.iloc[-1]['bid']
    high_bid = df_ticks['bid'].max()
    low_bid = df_ticks['bid'].min()
    
    print(f"   Candle Open: {candle['open']:.5f} | First Tick Bid: {first_bid:.5f}")
    print(f"   Candle High: {candle['high']:.5f} | Max Tick Bid: {high_bid:.5f}")
    print(f"   Candle Low: {candle['low']:.5f} | Min Tick Bid: {low_bid:.5f}")
    print(f"   Candle Close: {candle['close']:.5f} | Last Tick Bid: {last_bid:.5f}")
    
    # Show first and last few ticks
    print(f"\n🔝 First 5 Ticks:")
    print(df_ticks[['time_msc', 'bid', 'ask', 'last', 'volume']].head(5).to_string(index=False))
    
    print(f"\n🔚 Last 5 Ticks:")
    print(df_ticks[['time_msc', 'bid', 'ask', 'last', 'volume']].tail(5).to_string(index=False))
    
    # Calculate Permutation Entropy
    print(f"\n🧮 Permutation Entropy Analysis:")
    
    # Calculate PE for bid prices
    pe_bid_3 = permutation_entropy(df_ticks['bid'], order=3, delay=1)
    pe_bid_4 = permutation_entropy(df_ticks['bid'], order=4, delay=1)
    pe_bid_5 = permutation_entropy(df_ticks['bid'], order=5, delay=1)
    
    # Calculate PE for spread
    spread = df_ticks['ask'] - df_ticks['bid']
    pe_spread = permutation_entropy(spread, order=3, delay=1)
    
    print(f"   Bid Price PE (order=3): {pe_bid_3:.4f}")
    print(f"   Bid Price PE (order=4): {pe_bid_4:.4f}")
    print(f"   Bid Price PE (order=5): {pe_bid_5:.4f}")
    print(f"   Spread PE (order=3): {pe_spread:.4f}")
    
    # Interpretation
    if pe_bid_4 < 0.3:
        regime = "Highly Ordered (Low Complexity)"
    elif pe_bid_4 < 0.6:
        regime = "Moderate Complexity"
    else:
        regime = "Chaotic (High Complexity)"
    
    print(f"   Tick Regime: {regime}")
    
    return {'pe_bid_3': pe_bid_3, 'pe_bid_4': pe_bid_4, 'pe_bid_5': pe_bid_5, 'pe_spread': pe_spread}

def visualize_ticks(df_ticks, candle, symbol, pe_stats=None):
    """Create line graph visualization of tick data."""
    if df_ticks is None or len(df_ticks) == 0:
        return
    
    print(f"\n📊 Creating visualization...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
    
    # Title with PE info
    title = f'{symbol} - Tick Data Analysis\nCandle: {pd.to_datetime(candle["time"], unit="s")}'
    if pe_stats:
        title += f'\nPermutation Entropy (order=4): {pe_stats["pe_bid_4"]:.4f}'
    fig.suptitle(title, fontsize=14, color='white')
    
    # Panel 1: Bid and Ask Prices
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_ticks['time_msc'], df_ticks['bid'], color='cyan', linewidth=0.8, label='Bid', alpha=0.9)
    ax1.plot(df_ticks['time_msc'], df_ticks['ask'], color='orange', linewidth=0.8, label='Ask', alpha=0.9)
    
    # Add candle OHLC as horizontal lines
    ax1.axhline(y=candle['open'], color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='Open')
    ax1.axhline(y=candle['high'], color='green', linestyle='--', linewidth=1, alpha=0.5, label='High')
    ax1.axhline(y=candle['low'], color='red', linestyle='--', linewidth=1, alpha=0.5, label='Low')
    ax1.axhline(y=candle['close'], color='white', linestyle='--', linewidth=1, alpha=0.5, label='Close')
    
    ax1.set_ylabel('Price', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.2)
    ax1.set_title('Bid/Ask Price Movement', fontsize=11, pad=10)
    
    # Panel 2: Spread
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    spread = df_ticks['ask'] - df_ticks['bid']
    ax2.plot(df_ticks['time_msc'], spread * 10000, color='lime', linewidth=1, label='Spread (pips)')
    ax2.fill_between(df_ticks['time_msc'], 0, spread * 10000, color='lime', alpha=0.2)
    ax2.set_ylabel('Spread (pips)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_title('Bid-Ask Spread', fontsize=11, pad=10)
    
    # Panel 3: Rolling Permutation Entropy
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Calculate rolling PE
    window_size = max(50, len(df_ticks) // 10)
    pe_rolling = []
    times_rolling = []
    
    for i in range(window_size, len(df_ticks)):
        window_data = df_ticks['bid'].iloc[i-window_size:i]
        pe_val = permutation_entropy(window_data, order=3, delay=1)
        pe_rolling.append(pe_val)
        times_rolling.append(df_ticks['time_msc'].iloc[i])
    
    if pe_rolling:
        ax3.plot(times_rolling, pe_rolling, color='magenta', linewidth=1.5, label='Rolling PE (order=3)')
        ax3.axhline(y=0.5, color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='PE=0.5')
        ax3.fill_between(times_rolling, 0, pe_rolling, color='magenta', alpha=0.2)
        ax3.set_ylabel('Permutation Entropy', fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.2)
        ax3.set_title('Tick Complexity (Rolling PE)', fontsize=11, pad=10)
    
    # Panel 4: Volume/Distribution
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if 'volume' in df_ticks.columns and df_ticks['volume'].sum() > 0:
        ax4.bar(df_ticks['time_msc'], df_ticks['volume'], color='purple', alpha=0.6, width=0.0001)
        ax4.set_ylabel('Volume', fontsize=10)
        ax4.set_title('Tick Volume', fontsize=11, pad=10)
    else:
        # Show tick count over time (binned)
        ax4.hist(df_ticks['time_msc'], bins=50, color='purple', alpha=0.6, edgecolor='white')
        ax4.set_ylabel('Tick Count', fontsize=10)
        ax4.set_title('Tick Distribution', fontsize=11, pad=10)
    
    ax4.grid(True, alpha=0.2)
    ax4.set_xlabel('Time', fontsize=10)
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'tick_chart_{symbol}.png'
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
    TIMEFRAME = mt5.TIMEFRAME_M15
    CANDLE_INDEX = 1  # 0 = current (incomplete), 1 = last completed candle
    
    print(f"\n{'='*60}")
    print(f"  TICK DATA ANALYZER")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Timeframe: M15")
    print(f"  Candle: {CANDLE_INDEX} (0=current, 1=previous)")
    print(f"{'='*60}")
    
    # Get ticks
    result = get_candle_ticks(SYMBOL, TIMEFRAME, CANDLE_INDEX)
    
    if result:
        df_ticks, candle = result
        pe_stats = analyze_ticks(df_ticks, candle)
        
        # Visualize ticks with PE stats
        visualize_ticks(df_ticks, candle, SYMBOL, pe_stats)
        
        # Optional: Save to CSV
        save_csv = input("\n💾 Save ticks to CSV? (y/n): ").lower().strip()
        if save_csv == 'y':
            filename = f"ticks_{SYMBOL}_{CANDLE_INDEX}.csv"
            df_ticks.to_csv(filename, index=False)
            print(f"✅ Saved to {filename}")
    
    mt5.shutdown()
    print(f"\n👋 Disconnected from MT5")

if __name__ == "__main__":
    main()
