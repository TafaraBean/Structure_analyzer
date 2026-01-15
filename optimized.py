import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import webbrowser

# --- CONFIGURATION ---
CSV_FILES = ["data/Exness_XAUUSDm_2024.csv", "data/Exness_XAUUSDm_2025.csv", "data/Exness_XAUUSDm_2026.csv"]
CACHE_FILE = "data/XAU_5mins_resampled.parquet"

# 🏆 STRATEGY SETTINGS
INITIAL_CAPITAL = 10000
TIMEFRAME = "5min"

# 💰 RISK SETTINGS
LOT_SIZE = 0.01          
CONTRACT_SIZE = 100      

WINNING_PARAMS = {
    'use_rsi': True, 'rsi_min': 30, 'rsi_max': 70,
    'sl_mult': 2.0, 'tp_mult': 1.5,
    'be_trigger': 1.0, 'step_mult': 0.2, 'max_hold': 120,
    'daily_target_usd': 800.0 
}

# 🚫 FILTERS
AVOID_NY_SESSION = False
NY_START_HOUR = 13 
NY_END_HOUR = 23
ALLOW_STACKING = False 

TARGET_PORTFOLIO = [
    'CDLLONGLEGGEDDOJI', 'CDLRICKSHAWMAN', 'CDLHIGHWAVE', 
    'CDLENGULFING', 'CDLBELTHOLD'
]

# ==========================================
# PART 1: MEMORY-SAFE DATA BUILDER (CHUNKING)
# ==========================================
def build_data_cache():
    print(f"[*] Building Data Cache (Memory Safe Mode)...")
    if not os.path.exists("data"): os.makedirs("data")
    
    csvs_found = [f for f in CSV_FILES if os.path.exists(f)]
    if not csvs_found: sys.exit("❌ No CSV files found.")

    all_candles = []

    for f in csvs_found:
        print(f"    -> Processing {f} in chunks...")
        try:
            # 1. Read in Chunks (1 million rows at a time)
            # This prevents RAM explosion
            chunk_size = 1_000_000
            chunk_iterator = pd.read_csv(f, sep=None, engine='python', chunksize=chunk_size)
            
            for i, chunk in enumerate(chunk_iterator):
                # Clean headers
                chunk.columns = [c.lower().strip() for c in chunk.columns]
                
                # Parse Dates & Remove Timezone (Naive)
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp']).dt.tz_localize(None)
                chunk.set_index('timestamp', inplace=True)
                
                # Resample this small chunk
                # We use 'bid' for price
                resampled_chunk = chunk['bid'].resample('5min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                
                # Store the small result
                all_candles.append(resampled_chunk)
                print(f"       Processed Chunk {i+1}...", end='\r')
                
            print(f"\n       ✅ File {f} complete.")
            
        except Exception as e:
            print(f"\n    ⚠️ Error reading {f}: {e}")

    if not all_candles:
        sys.exit("❌ Failed to load any data.")

    print("    -> Merging and Finalizing...")
    # 2. Combine all small chunks
    df = pd.concat(all_candles).sort_index()
    
    # 3. Deduplicate (Handle edge cases where chunks split a candle)
    # We group by the index again to merge any split 5-min bars
    df = df.groupby(df.index).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    print("    -> Calculating Indicators...")
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    df['ATR'] = talib.ATR(h, l, c, timeperiod=14)
    df['RSI'] = talib.RSI(c, timeperiod=14)
    
    df.dropna(inplace=True)
    df.to_parquet(CACHE_FILE)
    print(f"✅ Cache Built: {len(df)} candles.")
    return df

def load_data():
    try:
        df = pd.read_parquet(CACHE_FILE)
        # Check for timezone issues
        if df.index.tz is not None:
            print("⚠️ Cache has timezones. Rebuilding...")
            return build_data_cache()
        return df
    except:
        return build_data_cache()

# ==========================================
# PART 2: INFINITY RUNNER
# ==========================================
def calculate_trades_real_money(df):
    all_potential_trades = []
    
    opens = df['open'].values; highs = df['high'].values
    lows = df['low'].values; closes = df['close'].values
    atrs = df['ATR'].values; times = df.index
    n_candles = len(closes)
    
    sl_mult = WINNING_PARAMS['sl_mult']
    tp_mult = WINNING_PARAMS['tp_mult']
    be_trigger = WINNING_PARAMS['be_trigger']
    step_mult = WINNING_PARAMS['step_mult']
    max_hold = WINNING_PARAMS['max_hold']
    daily_target = WINNING_PARAMS['daily_target_usd']

    print("[*] Simulating Trades...")
    for pat in TARGET_PORTFOLIO:
        if not hasattr(talib, pat): continue
        func = getattr(talib, pat)
        sigs = func(opens, highs, lows, closes)
        indices = np.where(sigs != 0)[0]
        indices = indices[(indices > 50) & (indices < n_candles - max_hold)]
        
        for idx in indices:
            if AVOID_NY_SESSION:
                if NY_START_HOUR <= times[idx].hour < NY_END_HOUR: continue

            entry_price = closes[idx]
            entry_time = times[idx]
            atr = atrs[idx]
            direction = 1 if sigs[idx] == 100 else -1
            
            sl_dist = atr * sl_mult; tp_dist = atr * tp_mult
            be_dist = atr * be_trigger; step_dist = atr * step_mult
            
            if direction == 1: current_sl = entry_price - sl_dist; current_tp = entry_price + tp_dist
            else: current_sl = entry_price + sl_dist; current_tp = entry_price - tp_dist
            
            is_runner = False; exit_pnl_price = 0.0; trade_finished = False
            exit_time = entry_time 
            
            for i in range(1, max_hold):
                curr = idx + i
                c_high = highs[curr]; c_low = lows[curr]; c_close = closes[curr]
                current_time = times[curr]
                
                if direction == 1:
                    if c_low <= current_sl: 
                        exit_pnl_price = current_sl - entry_price; trade_finished = True; exit_time = current_time; break
                    if not is_runner and c_high >= current_tp: 
                        exit_pnl_price = current_tp - entry_price; trade_finished = True; exit_time = current_time; break
                    if not is_runner and c_high >= (entry_price + be_dist): 
                        is_runner = True; current_sl = entry_price; current_tp = 99999999
                    if is_runner:
                        steps = int((c_high - entry_price) / step_dist)
                        if steps >= 1:
                            new_sl = entry_price + (steps * step_dist) - step_dist
                            if new_sl > current_sl: current_sl = new_sl
                else:
                    if c_high >= current_sl: 
                        exit_pnl_price = entry_price - current_sl; trade_finished = True; exit_time = current_time; break
                    if not is_runner and c_low <= current_tp: 
                        exit_pnl_price = entry_price - current_tp; trade_finished = True; exit_time = current_time; break
                    if not is_runner and c_low <= (entry_price - be_dist): 
                        is_runner = True; current_sl = entry_price; current_tp = -99999999
                    if is_runner:
                        steps = int((entry_price - c_low) / step_dist)
                        if steps >= 1:
                            new_sl = entry_price - (steps * step_dist) + step_dist
                            if new_sl < current_sl: current_sl = new_sl
                
                if i == (max_hold - 1): # Time Stop
                    exit_pnl_price = (c_close - entry_price) if direction == 1 else (entry_price - c_close)
                    trade_finished = True
                    exit_time = current_time

            if trade_finished:
                real_usd_pnl = exit_pnl_price * CONTRACT_SIZE * LOT_SIZE
                all_potential_trades.append({
                    'time': entry_time, 
                    'exit_time': exit_time,
                    'pnl': real_usd_pnl
                })

    # 2. Filtering
    if not all_potential_trades: return pd.Series(dtype=float)
    
    trades_df = pd.DataFrame(all_potential_trades).sort_values('time')
    
    mask = pd.Series(True, index=df.index)
    if WINNING_PARAMS['use_rsi']:
        mask &= (df['RSI'] >= WINNING_PARAMS['rsi_min']) & (df['RSI'] <= WINNING_PARAMS['rsi_max'])
    
    valid_entries = mask.shift(1).fillna(False).astype(bool)
    
    final_trades = []
    
    # Initialize as naive timestamp (matches the data now)
    last_trade_exit = pd.Timestamp.min 
    
    trades_df['date'] = trades_df['time'].dt.date
    
    for date, group in trades_df.groupby('date'):
        daily_pnl = 0.0
        
        for _, row in group.iterrows():
            if not valid_entries.asof(row['time']): continue
            if daily_pnl >= daily_target: continue
            
            if not ALLOW_STACKING:
                if row['time'] < last_trade_exit:
                    continue
            
            final_trades.append({'time': row['time'], 'pnl': row['pnl']})
            daily_pnl += row['pnl']
            
            if row['exit_time'] > last_trade_exit:
                last_trade_exit = row['exit_time']
            
    return pd.DataFrame(final_trades)

# ==========================================
# PART 3: VISUALIZATION
# ==========================================
if __name__ == "__main__":
    df = load_data()
    print("\n-------------------------------------------")
    print(f" ♾️ INFINITY RUNNER (No Stacking: {not ALLOW_STACKING})")
    print(f" Lot Size: {LOT_SIZE} | Daily Target: ${WINNING_PARAMS['daily_target_usd']}")
    print("-------------------------------------------\n")

    executed_trades = calculate_trades_real_money(df)
    
    if executed_trades.empty:
        print("No trades executed.")
    else:
        executed_trades.set_index('time', inplace=True)
        equity_curve = executed_trades['pnl'].cumsum() + INITIAL_CAPITAL
        
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak)
        final_eq = equity_curve.iloc[-1]
        ret_pct = ((final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        print(f"💰 Final Equity: ${final_eq:,.2f}")
        print(f"📈 Total Return: {ret_pct:.2f}%")
        print(f"🔢 Total Trades: {len(executed_trades)}")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines+markers', name='Equity', line=dict(color='#00ff00')), row=1, col=1)
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown', line=dict(color='#ff0000'), fill='tozeroy'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
        fig.add_hrect(y0=30, y1=70, row=3, col=1, fillcolor="green", opacity=0.1, line_width=0)
        
        mode_str = "STACKING" if ALLOW_STACKING else "NO_STACKING"
        fig.update_layout(title=f"Infinity Runner ({mode_str} | Target: ${WINNING_PARAMS['daily_target_usd']})", template="plotly_dark", height=900)
        
        filename = f"infinity_runner_{mode_str.lower()}.html"
        fig.write_html(filename)
        print(f"\n✅ Report generated: {filename}")
        webbrowser.open(filename)