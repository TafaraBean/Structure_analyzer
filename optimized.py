import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import webbrowser

# --- CONFIGURATION ---
# 1. Data Sources (Include 2026)
CSV_FILES = ["Exness_XAUUSDm_2024.csv", "Exness_XAUUSDm_2025.csv", "Exness_XAUUSDm_2026.csv"]
CACHE_FILE = "btc_5min_2024_2026.parquet"
CHUNK_SIZE = 5000000

# 2. Strategy Settings
TIMEFRAME = "5min"
INITIAL_CAPITAL = 10000

# 🏆 WINNING SETTINGS
WINNING_PARAMS = {
    'tp_mult': 1.5,
    'sl_mult': 2.0,
    'use_adx': False,
    'use_rsi': True,
    'use_vol': False,
    'use_mfi': False,
    'rsi_min': 30,
    'rsi_max': 70
}

TARGET_PORTFOLIO = [
    'CDLLONGLEGGEDDOJI', 
    'CDLRICKSHAWMAN', 
    'CDLHIGHWAVE', 
    'CDLENGULFING', 
    'CDLBELTHOLD'
]

# ==========================================
# PART 1: DATA FOUNDRY (Builder)
# ==========================================
def build_dataset():
    if os.path.exists(CACHE_FILE):
        print(f"✅ Found existing 5-min dataset: {CACHE_FILE}")
        return True

    print(f"[*] Building 5-Minute Dataset (2024-2026)...")
    all_candles = []
    
    for f in CSV_FILES:
        if not os.path.exists(f): 
            print(f"⚠️ Warning: {f} not found.")
            continue
            
        print(f"   -> Processing {f}...")
        try:
            # Read Raw Ticks/M1 Data
            chunk_iterator = pd.read_csv(
                f, 
                chunksize=CHUNK_SIZE, 
                header=0, 
                usecols=[2, 3], 
                names=['time', 'bid'], 
                quotechar='"'
            )
            
            for i, chunk in enumerate(chunk_iterator):
                chunk['time'] = pd.to_datetime(chunk['time'], format='ISO8601', errors='coerce')
                chunk.dropna(inplace=True)
                chunk.set_index('time', inplace=True)
                
                # RESAMPLE TO 5 MINUTES DIRECTLY
                resampled = chunk['bid'].resample('5min').agg({
                    'open': 'first', 
                    'high': 'max', 
                    'low': 'min', 
                    'close': 'last', 
                    'bid': 'count'
                }).rename(columns={'bid': 'volume'})
                
                resampled.dropna(inplace=True)
                all_candles.append(resampled)
                print(f"      Processed chunk {i+1}...", end='\r')
            print("")
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")

    if not all_candles:
        print("❌ No data processed.")
        return False

    print("[*] Merging and Saving...")
    full_df = pd.concat(all_candles)
    
    # Final Groupby to handle overlapping chunks and ensure 5min integrity
    full_df = full_df.groupby(full_df.index).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    
    full_df.sort_index(inplace=True)
    full_df.to_parquet(CACHE_FILE)
    print(f"✅ DATA READY: {len(full_df)} 5-min candles saved.")
    return True

def load_data():
    if not os.path.exists(CACHE_FILE):
        if not build_dataset():
            sys.exit()
    
    print("Loading data...")
    df = pd.read_parquet(CACHE_FILE)
    
    # Calculate Indicators needed for Simulation
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    
    return df.dropna()

# ==========================================
# PART 2: SIMULATION ENGINE
# ==========================================
def calculate_trades(df):
    strategies = {}
    closes = df['close'].values
    atrs = df['ATR'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    
    tp_mult = WINNING_PARAMS['tp_mult']
    sl_mult = WINNING_PARAMS['sl_mult']
    
    # Limit lookforward to 12 bars (1 hour) for scalping efficiency
    max_hold = 12 
    
    for pat in TARGET_PORTFOLIO:
        if not hasattr(talib, pat): continue
        
        func = getattr(talib, pat)
        sigs = func(opens, highs, lows, closes)
        
        indices = np.where(sigs != 0)[0]
        # Ensure enough data for lookahead
        indices = indices[(indices > 50) & (indices < len(df) - max_hold)]
        
        trade_list = []
        for idx in indices:
            direction = 1 if sigs[idx] == 100 else -1
            entry = closes[idx]
            sl_dist = atrs[idx] * sl_mult
            
            realized = 0
            
            if direction == 1:
                sl = entry - sl_dist
                tp = entry + (sl_dist * (tp_mult/sl_mult))
                for j in range(idx+1, idx+max_hold):
                    if lows[j] <= sl: realized = -1; break
                    if highs[j] >= tp: realized = (tp_mult/sl_mult); break
            else:
                sl = entry + sl_dist
                tp = entry - (sl_dist * (tp_mult/sl_mult))
                for j in range(idx+1, idx+max_hold):
                    if highs[j] >= sl: realized = -1; break
                    if lows[j] <= tp: realized = (tp_mult/sl_mult); break
            
            if realized != 0:
                trade_list.append((df.index[idx], realized * 100)) # $100 Risk
                
        strategies[pat] = trade_list
    return strategies

# ==========================================
# PART 3: MAIN & VISUALIZATION
# ==========================================
if __name__ == "__main__":
    df = load_data()
    
    print(f"Dataset Range: {df.index[0]} to {df.index[-1]}")
    
    # 1. Calculate Raw Trades
    print("Simulating trades...")
    raw_strategies = calculate_trades(df)
    
    # 2. Apply Regime Filters (No Lookahead)
    print("Applying filters...")
    mask = pd.Series(True, index=df.index)
    
    if WINNING_PARAMS['use_rsi']:
        mask &= (df['RSI'] >= WINNING_PARAMS['rsi_min']) & (df['RSI'] <= WINNING_PARAMS['rsi_max'])
    
    # Shift mask to ensure we filter based on PREVIOUS candle
    valid_entries = mask.shift(1).fillna(False)
    
    # 3. Build Equity Curve
    equity_series = pd.Series(0.0, index=df.index)
    total_trades = 0
    
    for pat, trades in raw_strategies.items():
        for entry_time, pnl in trades:
            if valid_entries.loc[entry_time]:
                equity_series.loc[entry_time] += pnl
                total_trades += 1
                
    # Cumulative Sum
    equity_curve = equity_series.cumsum() + INITIAL_CAPITAL
    
    # 4. Calculate Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak)
    
    # 5. Plotting
    print("Generating chart...")
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Total Equity (2024-2026)", "Drawdown ($)", "RSI Context")
    )
    
    # Top: Equity
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve, 
        mode='lines', name='Equity',
        line=dict(color='#00ff00', width=2),
        fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.05)'
    ), row=1, col=1)
    
    # Middle: Drawdown
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, 
        mode='lines', name='Drawdown',
        line=dict(color='#ff0000', width=1),
        fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'
    ), row=2, col=1)
    
    # Bottom: RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange', width=1), name='RSI'), row=3, col=1)
    if WINNING_PARAMS['use_rsi']:
        fig.add_hrect(
            y0=WINNING_PARAMS['rsi_min'], y1=WINNING_PARAMS['rsi_max'], 
            row=3, col=1, fillcolor="green", opacity=0.1, line_width=0,
            annotation_text="Trading Zone"
        )
    
    # Layout
    final_eq = equity_curve.iloc[-1]
    ret_pct = ((final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    fig.update_layout(
        title=f"🚀 Strategy Performance (2024-2026): +{ret_pct:.1f}% Return | {total_trades} Trades",
        template="plotly_dark",
        height=900,
        hovermode="x unified"
    )
    
    filename = "equity_report_2026.html"
    fig.write_html(filename)
    print(f"Done! Opening {filename}...")
    webbrowser.open(filename)