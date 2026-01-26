import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# --- Configuration ---
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 2000
ZOOM_LAST = 1000 

# Entropy Settings
PE_ORDER = 3        # Dimension of permutation (4 or 5 is standard)
PE_DELAY = 1        # Delay between points
PE_WINDOW = 30     # Rolling window size to compute entropy

def init_mt5():
    """Initialize connection to MetaTrader 5."""
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    params = {}
    if path: params["path"] = path

    if not mt5.initialize(**params):
        print(f"❌ MT5 Initialize failed, error code = {mt5.last_error()}")
        return False
    
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
             print(f"⚠️ MT5 Login failed, error code = {mt5.last_error()}")
    
    print(f"✅ Connected to MT5: {mt5.terminal_info().name}")
    return True

def fetch_data(symbol, timeframe, n_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print(f"❌ Failed to get rates for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def permutation_entropy(time_series, order=3, delay=1):
    """
    Calculate Permutation Entropy for a 1D array.
    This is for a single window.
    """
    n = len(time_series)
    permutations = np.array(list(np.argsort(time_series[i:i + order * delay:delay]) 
                                 for i in range(n - order * delay + 1)))
    
    # Count unique permutations
    _, counts = np.unique(permutations, axis=0, return_counts=True)
    
    # Probabilities
    probs = counts / len(permutations)
    
    # Shannon Entropy
    pe = -np.sum(probs * np.log2(probs))
    
    # Normalize by log2(n!)
    pe_norm = pe / np.log2(math.factorial(order))
    
    return pe_norm

def rolling_permutation_entropy(series, window, order=3, delay=1):
    """
    Apply Permutation Entropy over a rolling window.
    """
    # Note: efficient rolling apply in manual loop is acceptable for 1000 bars
    results = np.full(len(series), np.nan)
    
    values = series.values
    for i in range(window, len(series)):
        segment = values[i-window:i]
        results[i] = permutation_entropy(segment, order, delay)
        
    return pd.Series(results, index=series.index)

def main():
    if not init_mt5():
        sys.exit()

    print(f"📥 Fetching {BARS} bars for {SYMBOL}...")
    df = fetch_data(SYMBOL, TIMEFRAME, BARS)
    if df is None: return

    # --- Indicators ---
    print("🧮 Calculating Indicators...")
    
    # 1. Permutation Entropy (Measure of Disorder)
    # 0 = Predictable, 1 = Random Noise
    print(f"   - Permutation Entropy (Window: {PE_WINDOW}, Order: {PE_ORDER})")
    df['PE'] = rolling_permutation_entropy(df['close'], window=PE_WINDOW, order=PE_ORDER, delay=PE_DELAY)
    
    # 2. ADX (Trend Strength)
    print("   - ADX (Trend Strength)")
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 3. ATR (Volatility)
    print("   - ATR (Volatility)")
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # --- Filtering ---
    # Tradeable Regime: Low Entropy (Predictable)
    # Threshold is somewhat empirical, usually < 0.9 or < 0.8 depending on window/order
    # Let's visualize it to find the sweet spot
    entropy_threshold = 0.85
    
    # Slice for Zoom
    df_zoom = df.tail(ZOOM_LAST).copy() # Use copy to avoid setting on slice warnings

    print("🎨 Generating Dashboard...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.15)
    
    # --- Panel 1: Price Action ---
    ax1 = fig.add_subplot(gs[0])
    
    # Candlestick plotting
    width_body = 0.0025; width_wick = 0.0005 
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    col_up = '#089981'; col_down = '#f23645'
    
    ax1.bar(up.index, up.high-up.low, width_wick, bottom=up.low, color=col_up, alpha=0.9)
    ax1.bar(down.index, down.high-down.low, width_wick, bottom=down.low, color=col_down, alpha=0.9)
    ax1.bar(up.index, up.close-up.open, width_body, bottom=up.open, color=col_up, alpha=1.0)
    ax1.bar(down.index, down.open-down.close, width_body, bottom=down.close, color=col_down, alpha=1.0)
    
    # Highlight Low Entropy Zones
    # We define Low Entropy as "Predictable Structure"
    # Find spans where PE < threshold
    is_low_entropy = df_zoom['PE'] < entropy_threshold
    
    # Use fill_between to highlight regions
    # Create a boolean mask and fill
    ylim = ax1.get_ylim()
    # We need to map dates to numbers for fill_between if using index directly
    # But since we used bars with index dates, we can use fill_between with dates
    ax1.fill_between(df_zoom.index, df_zoom['low'].min(), df_zoom['high'].max(), 
                     where=is_low_entropy, color='green', alpha=0.15, label='Low Entropy (Tradeable)')

    ax1.set_title(f"{SYMBOL} Market Regime Analysis", fontsize=16, fontweight='bold', color='white')
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # --- Panel 2: Permutation Entropy ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df_zoom.index, df_zoom['PE'], color='yellow', linewidth=1.5)
    ax2.axhline(y=entropy_threshold, color='green', linestyle='--', alpha=0.7, label=f'Threshold ({entropy_threshold})')
    ax2.set_ylabel("Entropy (PE)")
    ax2.set_ylim(0, 1.0)
    ax2.fill_between(df_zoom.index, 0, entropy_threshold, color='green', alpha=0.1)
    ax2.text(df_zoom.index[0], 0.9, "1.0 = Chaos/Noise", color='gray', fontsize=8)
    ax2.text(df_zoom.index[0], 0.1, "0.0 = Perfect Order", color='gray', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.1)
    
    # --- Panel 3: ADX ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df_zoom.index, df_zoom['ADX'], color='cyan', linewidth=1.5)
    ax3.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel("ADX (Trend)")
    ax3.text(df_zoom.index[0], 26, "Trend > 25", color='gray', fontsize=8)
    ax3.grid(alpha=0.1)

    # --- Panel 4: ATR ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df_zoom.index, df_zoom['ATR'], color='orange', linewidth=1.5)
    ax4.set_ylabel("ATR (Vol)")
    ax4.grid(alpha=0.1)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
