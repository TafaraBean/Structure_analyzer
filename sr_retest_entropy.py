import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
PE_ORDER = 4
PE_DELAY = 1
PE_WINDOW = 30  # Window to measure entropy *during* the retest

# S/R Settings
FRACTAL_WINDOW = 5  # 5 bars on each side for a "Major" fractal
LEVEL_TOLERANCE = 0.005 # 0.05% tolerance for a "Touch"
RETEST_COOLDOWN = 50 # Bars to wait before registering another retest of same level

def init_mt5():
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    params = {}
    if path: params["path"] = path
    if not mt5.initialize(**params):
        print(f"❌ MT5 Init failed: {mt5.last_error()}")
        return False
    if login and password and server:
        mt5.login(login=int(login), password=password, server=server)
    print(f"✅ Connected: {mt5.terminal_info().name}")
    return True

def fetch_data(symbol, timeframe, n_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def permutation_entropy(time_series, order=3, delay=1):
    n = len(time_series)
    if n < order * delay: return 0.0
    
    # 1. Create partitions
    partitions = np.array([time_series[i:i+order*delay:delay] for i in range(n - order * delay + 1)])
    
    # 2. Get argsort (ranks) for each partition to get the "ordinal pattern"
    # Example: [10, 12, 11] -> [0, 2, 1]
    ords = np.argsort(partitions, axis=1)
    
    # 3. Count unique patterns
    # Turn rows into tuples to use unique
    unique_patterns, counts = np.unique(ords, axis=0, return_counts=True)
    
    # 4. Probabilities
    probs = counts / len(ords)
    
    # 5. Shannon Entropy
    pe = -np.sum(probs * np.log2(probs + 1e-9)) # Add epsilon to avoid log(0)
    
    # 6. Normalize
    pe_norm = pe / np.log2(math.factorial(order))
    return pe_norm

def find_fractals(df, window=5):
    """
    Find major Fractal Highs and Lows.
    Returns: List of {'price': float, 'type': 'support'/'resistance', 'index': int}
    """
    levels = []
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(window, len(df) - window):
        # Resistance (Fractal High)
        if highs[i] == max(highs[i-window:i+window+1]):
            levels.append({'price': highs[i], 'type': 'resistance', 'index': df.index[i], 'idx_num': i})
            
        # Support (Fractal Low)
        if lows[i] == min(lows[i-window:i+window+1]):
            levels.append({'price': lows[i], 'type': 'support', 'index': df.index[i], 'idx_num': i})
            
    return levels

def analyze_retests(df, levels):
    """
    Scan future price action to see when these levels are 'retested'.
    Calculate Entropy at that moment.
    """
    retests = []
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # We iterate through time. For each bar, check active levels.
    # To optimize: distinct levels list using clustering could be better, but let's stick to raw fractals first.
    # We only care about levels "created in the past".
    
    # Simplify: Take the last N levels found, and see if they get hit later.
    
    processed_levels = []
    
    # Filter levels to significant ones or just process all (might be many)
    # Strategy: Just iterate all found levels, look forward from their creation
    
    for lvl in levels:
        start_idx = lvl['idx_num'] + FRACTAL_WINDOW + 1 # Start looking after it's confirmed
        lvl_price = lvl['price']
        
        # Look forward
        cooldown = 0
        for i in range(start_idx, len(df)):
            if cooldown > 0: 
                cooldown -= 1
                continue
            
            # Check proximity
            # Tolerance band
            dist = abs(close[i] - lvl_price) / lvl_price
            
            is_touch = False
            if dist < LEVEL_TOLERANCE:
                is_touch = True
                
            # If Low touches Support or High touches Resistance (more precise)
            if lvl['type'] == 'support' and low[i] <= lvl_price * (1 + LEVEL_TOLERANCE) and low[i] >= lvl_price * (1 - LEVEL_TOLERANCE):
                 is_touch = True
            if lvl['type'] == 'resistance' and high[i] >= lvl_price * (1 - LEVEL_TOLERANCE) and high[i] <= lvl_price * (1 + LEVEL_TOLERANCE):
                 is_touch = True
                 
            if is_touch:
                # Calculate Entropy of the window surrounding this touch
                # e.g. i - 15 to i + 15 (if we are backtesting)
                # For real-time we'd look back. Let's look back 30 bars to assess the "Approach" quality
                
                lb = max(0, i - PE_WINDOW)
                segment = close[lb:i+1]
                
                pe = permutation_entropy(segment, order=PE_ORDER, delay=PE_DELAY)
                
                retests.append({
                    'time': df.index[i],
                    'price': lvl_price, # Plot at level price
                    'pe': pe,
                    'type': lvl['type']
                })
                
                cooldown = RETEST_COOLDOWN # Don't spam dots
                
    return retests

def main():
    if not init_mt5(): sys.exit()
    
    print(f"📥 Fetching {BARS} bars...")
    df = fetch_data(SYMBOL, TIMEFRAME, BARS)
    if df is None: return
    
    # Only analyze the last portion for the visual, but use full history for level finding
    # Actually, we need to find levels in the past to see retests in the NOW.
    
    print("🔍 Detecting Fractals & Retests...")
    levels = find_fractals(df, window=FRACTAL_WINDOW)
    retests = analyze_retests(df, levels)
    
    # Filter retests for the Zoom window
    start_zoom = df.index[-ZOOM_LAST]
    visible_retests = [r for r in retests if r['time'] >= start_zoom]
    
    print(f"   Found {len(levels)} Levels, {len(retests)} Retest Events.")
    
    # --- Plotting ---
    print("🎨 Generating Chart...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    df_zoom = df.tail(ZOOM_LAST)
    
    # 1. Candlesticks
    width = 0.002
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    col_up = '#089981'; col_down = '#f23645'
    
    ax.bar(up.index, up.close-up.open, width, bottom=up.open, color=col_up)
    ax.bar(up.index, up.high-up.low, 0.0005, bottom=up.low, color=col_up)
    ax.bar(down.index, down.open-down.close, width, bottom=down.close, color=col_down)
    ax.bar(down.index, down.high-down.low, 0.0005, bottom=down.low, color=col_down)
    
    # 2. Plot Retests
    # We plot the retest points. Color them based on Entropy.
    # Low Entropy (< 0.8) = Green (Safe/Clean Retest)
    # High Entropy (> 0.8) = Red (Chaos/Dangerous)
    
    times = [r['time'] for r in visible_retests]
    prices = [r['price'] for r in visible_retests]
    pes = [r['pe'] for r in visible_retests]
    
    # Scatter plot with colormap
    sc = ax.scatter(times, prices, c=pes, cmap='RdYlGn_r', s=100, edgecolors='white', zorder=10, label='Retest Quality')
    # RdYlGn_r: Red (High) to Green (Low). _r reverses it so Low PE is Green.
    
    # Add Colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Permutation Entropy (Green=Low/Ordered, Red=High/Chaos)')
    
    ax.set_title(f"{SYMBOL} Support/Resistance Retest Analysis", fontsize=16, color='white')
    ax.set_ylabel("Price")
    ax.grid(alpha=0.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
