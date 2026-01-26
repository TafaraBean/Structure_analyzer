import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib
import matplotlib.colors as mcolors
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# --- Configuration ---
SYMBOL = "EURUSDm"
MASTER_TIMEFRAME = mt5.TIMEFRAME_M15
BARS_MASTER = 5000 
ZOOM_LAST = 2000

# Entropy Settings
PE_ORDER = 4
PE_DELAY = 1
PE_WINDOW = 30
MA_PERIOD = 20
ENTROPY_THRESHOLD = 0.2 # Threshold for "Stable" (Green)
CONFLUENCE_MIN = 4      # Minimum TFs to trigger highlight
ADX_THRESHOLD = 30      # Threshold for "Strong" Trend

TIMEFRAMES = {
    'M15': {'tf': mt5.TIMEFRAME_M15, 'delta': pd.Timedelta(minutes=15)},
    'M30': {'tf': mt5.TIMEFRAME_M30, 'delta': pd.Timedelta(minutes=30)},
    'H1':  {'tf': mt5.TIMEFRAME_H1,  'delta': pd.Timedelta(hours=1)},
    'H4':  {'tf': mt5.TIMEFRAME_H4,  'delta': pd.Timedelta(hours=4)},
    'D1':  {'tf': mt5.TIMEFRAME_D1,  'delta': pd.Timedelta(days=1)},
    'W1':  {'tf': mt5.TIMEFRAME_W1,  'delta': pd.Timedelta(weeks=1)},
}

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
    return True

def clean_data(rates):
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def fetch_tf_data(symbol, tf_constant, bars):
    rates = mt5.copy_rates_from_pos(symbol, tf_constant, 0, bars)
    return clean_data(rates)

def permutation_entropy(series, order=3, delay=1):
    """Vectorized Permutation Entropy locally."""
    # We use a simplified sliding window approach for speed on small series
    # But for a full column, applying rolling is best.
    # Re-using the logic from market_regime but purely numpy for speed if possible
    
    # Just use the loop implementation for clarity as pandas rolling.apply is slow with custom complex funcs
    # We will compute PE on a rolling window of the MA.
    
    vals = series.values
    n = len(vals)
    result = np.full(n, np.nan)
    
    # Pre-compute factorial log
    denom = np.log2(math.factorial(order))
    
    for i in range(PE_WINDOW, n):
        window = vals[i-PE_WINDOW : i]
        
        # Compute PE for this window
        # 1. Embed
        n_w = len(window)
        if n_w < order*delay: continue
        
        partitions = np.array([window[j:j+order*delay:delay] for j in range(n_w - order * delay + 1)])
        ords = np.argsort(partitions, axis=1)
        _, counts = np.unique(ords, axis=0, return_counts=True)
        probs = counts / len(ords)
        pe = -np.sum(probs * np.log2(probs + 1e-9))
        
        result[i] = pe / denom
        
    return pd.Series(result, index=series.index)

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def main():
    if not init_mt5(): sys.exit()
    
    print(f"📥 Fetching Master Data ({SYMBOL}, M15)...")
    df_master = fetch_tf_data(SYMBOL, MASTER_TIMEFRAME, BARS_MASTER)
    if df_master is None: return
    
    # Initialize Heatmap Data
    # Index = Master Time, Columns = TFs
    heatmap_entropy = pd.DataFrame(index=df_master.index)
    heatmap_adx = pd.DataFrame(index=df_master.index) # Values will be ADX * Direction (+1/-1)
    heatmap_adx_entropy = pd.DataFrame(index=df_master.index) # Stability of ADX
    
    print("🔄 Processing Timeframes...")
    for name, tf_info in TIMEFRAMES.items():
        print(f"   - Processing {name}...")
        
        # 1. Fetch adequate history
        # We need enough bars to cover the Master duration.
        # Ratio of bars needed approx TimeDelta(Master) / TimeDelta(TF)
        # Just fetching 1000 bars of H4 is plenty for 2000 bars of M5 (M5=5min, H4=240min, ratio 48)
        # 2000 M5 bars ~ 166 hours. 1000 H4 bars ~ 4000 hours. Plenty.
        
        df_tf = fetch_tf_data(SYMBOL, tf_info['tf'], 1000)
        if df_tf is None: continue
        
        # 2. Compute Indicator (EMA 20)
        ema = calculate_ema(df_tf['close'], MA_PERIOD)
        
        # --- Entropy Calc ---
        pe = permutation_entropy(ema, order=PE_ORDER, delay=PE_DELAY)
        
        # --- ADX/DI Calc ---
        # We need High/Low/Close
        adx = talib.ADX(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        
        # Combine into a "Directional Strength" score
        # If +DI > -DI: Score = ADX
        # If -DI > +DI: Score = -ADX
        # Note: talib outputs numpy arrays
        direction = np.where(plus_di >= minus_di, 1, -1)
        adx_score = adx * direction
        adx_series = pd.Series(adx_score, index=df_tf.index)
        
        # --- Entropy of ADX ---
        # Stability of the "Strength"
        pe_adx = permutation_entropy(pd.Series(adx, index=df_tf.index), order=PE_ORDER, delay=PE_DELAY)

        # 4. SYNC TO MASTER (CRITICAL: PREVENT LOOKAHEAD)
        # Shift timestamps by Bar Duration (Availability Time)
        
        # Entropy Sync
        pe_available = pe.copy()
        pe_available.index = pe_available.index + tf_info['delta']
        aligned_pe = pe_available.reindex(df_master.index, method='ffill')
        
        # ADX Sync
        adx_available = adx_series.copy()
        adx_available.index = adx_available.index + tf_info['delta']
        aligned_adx = adx_available.reindex(df_master.index, method='ffill')
        
        # ADX Entropy Sync
        pe_adx_available = pe_adx.copy()
        pe_adx_available.index = pe_adx_available.index + tf_info['delta']
        aligned_pe_adx = pe_adx_available.reindex(df_master.index, method='ffill')
        
        heatmap_entropy[name] = aligned_pe
        heatmap_adx[name] = aligned_adx
        heatmap_adx_entropy[name] = aligned_pe_adx

    # Crop to Zoom Area
    df_zoom = df_master
    hm_ent_zoom = heatmap_entropy.loc[df_zoom.index].T 
    hm_adx_zoom = heatmap_adx.loc[df_zoom.index].T
    hm_adx_ent_zoom = heatmap_adx_entropy.loc[df_zoom.index].T
    
    # Ensure TFs are ordered H4 -> M1 (Top to Bottom)
    ordered_tfs = ['W1', 'D1', 'H4', 'H1', 'M30', 'M15']
    hm_ent_zoom = hm_ent_zoom.reindex(ordered_tfs)
    hm_adx_zoom = hm_adx_zoom.reindex(ordered_tfs)
    hm_adx_ent_zoom = hm_adx_ent_zoom.reindex(ordered_tfs)
    
    # --- Calculate Confluence (Entropy) ---
    stable_mask = hm_ent_zoom < ENTROPY_THRESHOLD
    confluence_score = stable_mask.sum(axis=0) 
    is_stable_confluence = confluence_score >= CONFLUENCE_MIN
    
    # --- Calculate Confluence (ADX/Direction) ---
    # Bullish: Score > 25
    bull_mask = hm_adx_zoom > ADX_THRESHOLD
    bull_score = bull_mask.sum(axis=0)
    is_bull_confluence = bull_score >= CONFLUENCE_MIN
    
    # Bearish: Score < -25
    bear_mask = hm_adx_zoom < -ADX_THRESHOLD
    bear_score = bear_mask.sum(axis=0)
    is_bear_confluence = bear_score >= CONFLUENCE_MIN
    
    # --- Visualization ---
    print("🎨 Generating Plot...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 16)) # Taller figure
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.1)
    
    # Panel 1: Price
    ax1 = fig.add_subplot(gs[0])
    
    # Width needs to be bigger for M15. 15 min = 15/(24*60) = 0.0104
    width = 0.008
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    col_up = '#089981'; col_down = '#f23645'
    
    ax1.bar(up.index, up.close-up.open, width, bottom=up.open, color=col_up)
    ax1.bar(up.index, up.high-up.low, 0.002, bottom=up.low, color=col_up)
    ax1.bar(down.index, down.open-down.close, width, bottom=down.close, color=col_down)
    ax1.bar(down.index, down.high-down.low, 0.002, bottom=down.low, color=col_down)
    
    # Highlight Confluence Zones
    # Use fill_between
    # We need to map boolean mask to dates. 
    # 'is_high_confluence' index is already dates (columns of heatmap_zoom) which match df_zoom index
    
    y_min, y_max = ax1.get_ylim() # Current limits (might need auto-scaling first or just set large)
    # Better: just use transform or min/max of data
    d_min = df_zoom['low'].min()
    d_max = df_zoom['high'].max()
    
    # 1. Stability Highlight (Blue/Cyan now to separate from Bull Green)
    ax1.fill_between(df_zoom.index, d_min, d_max, where=is_stable_confluence, 
                     color='cyan', alpha=0.1, label=f'Stability Confluence (>={CONFLUENCE_MIN} TFs)')
                     
    # 2. Bull Trend Highlight (Green)
    ax1.fill_between(df_zoom.index, d_min, d_max, where=is_bull_confluence, 
                     color='green', alpha=0.15, label=f'Bull Trend Confluence (>={CONFLUENCE_MIN} TFs)')

    # 3. Bear Trend Highlight (Red)
    ax1.fill_between(df_zoom.index, d_min, d_max, where=is_bear_confluence, 
                     color='red', alpha=0.15, label=f'Bear Trend Confluence (>={CONFLUENCE_MIN} TFs)')
    
    ax1.set_title(f"Multi-Timeframe Trend Stability & Direction", fontsize=16, color='white')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylabel("Price (M15)")
    ax1.grid(alpha=0.1)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # Panel 2: Entropy Heatmap
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Convert index to nums
    x_nums = mdates.date2num(hm_ent_zoom.columns)
    # Use centers for Y as well for simple 'nearest' shading (or auto)
    y_vals = np.arange(len(hm_ent_zoom.index))
    
    # Pcolormesh
    # shading='nearest' centers the color on the grid point X, Y
    c = ax2.pcolormesh(x_nums, y_vals, hm_ent_zoom.values, cmap='RdYlGn_r', shading='nearest', vmin=0, vmax=1)
    
    # Set Y-ticks to be center 
    ax2.set_yticks(y_vals)
    ax2.set_yticklabels(hm_ent_zoom.index)
    
    ax2.set_ylabel("Stability (Entropy)")
    
    # Colorbar for Entropy
    # cbar = plt.colorbar(c, ax=ax2, orientation='horizontal', pad=0.1) # Removed to save space
    # cbar.set_label("Entropy (Green=Stable, Red=Chaos)")
    
    # Panel 3: ADX Heatmap
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Normalize ADX 0-50 for colormap (-50 to 50 due to direction)
    norm = mcolors.Normalize(vmin=-50, vmax=50)
    
    # RdYlGn: 
    # -50 (Red) -> 0 (Yellow) -> +50 (Green)
    # This matches our intuition: Red = Bear, Green = Bull
    # But wait, usually High ADX (Trend) is good regardless of direction?
    # No, user wants Directional info usually. 
    # Green = Strong UP, Red = Strong DOWN. Yellow = No Trend. Perfect.
    
    c3 = ax3.pcolormesh(x_nums, y_vals, hm_adx_zoom.values, cmap='RdYlGn', shading='nearest', norm=norm)
    
    ax3.set_yticks(y_vals)
    ax3.set_yticklabels(hm_adx_zoom.index)
    ax3.set_ylabel("Direction (ADX)")
    
    # Colorbar for ADX
    # cbar3 = plt.colorbar(c3, ax=ax3, orientation='horizontal', pad=0.1)
    # cbar3.set_label("Trend Strength (Green=Strong Bull, Red=Strong Bear, Yellow=Weak)")
    
    # Panel 4: ADX Entropy (Stability of Strength)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    c4 = ax4.pcolormesh(x_nums, y_vals, hm_adx_ent_zoom.values, cmap='RdYlGn_r', shading='nearest', vmin=0, vmax=1)
    ax4.set_yticks(y_vals)
    ax4.set_yticklabels(hm_adx_ent_zoom.index)
    ax4.set_ylabel("ADX Stability")
    
    cbar4 = plt.colorbar(c4, ax=ax4, orientation='horizontal', pad=0.2)
    cbar4.set_label("ADX Entropy (Green=Smooth Strenghtening, Red=Erratic Strength)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
