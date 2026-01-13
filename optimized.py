import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import webbrowser

# --- CONFIGURATION ---
# Ensure these match your filenames exactly
CSV_FILES = ["data/Exness_XAUUSDm_2024.csv", "data/Exness_XAUUSDm_2025.csv", "data/Exness_XAUUSDm_2026.csv"]
CACHE_FILE = "data/XAU_5min_resampled.parquet"

# 🏆 STRATEGY SETTINGS
INITIAL_CAPITAL = 10000
TIMEFRAME = "5min"

WINNING_PARAMS = {
    'use_rsi': True, 'rsi_min': 30, 'rsi_max': 70,
    'sl_mult': 2.0, 'tp_mult': 1.5,
    'be_trigger': 1.0, 'step_mult': 0.2, 'max_hold': 80
}

TARGET_PORTFOLIO = [
    'CDLLONGLEGGEDDOJI', 'CDLRICKSHAWMAN', 'CDLHIGHWAVE', 
    'CDLENGULFING', 'CDLBELTHOLD'
]

# ==========================================
# PART 1: TICK-TO-CANDLE CONVERTER
# ==========================================
def build_data_cache():
    print(f"[*] Building Data Cache from TICK DATA...")
    
    if not os.path.exists("data"): os.makedirs("data")

    df_list = []
    csvs_found = [f for f in CSV_FILES if os.path.exists(f)]
    
    if not csvs_found:
        sys.exit("❌ No CSV files found. Please check paths in CSV_FILES.")

    for f in csvs_found:
        print(f"    -> Processing {f} (This may take a moment)...")
        try:
            # 1. Read Tick Data (Tab separated usually, or comma)
            # We explicitly name columns based on your provided format
            # Format: Exness | Symbol | Timestamp | Bid | Ask
            # We assume it has a header. If not, add 'header=None'. 
            # Based on your snippet, it seems to have a header.
            temp = pd.read_csv(f, sep=None, engine='python')
            
            # Clean column names
            temp.columns = [c.lower().strip() for c in temp.columns]
            
            # 2. Parse Timestamp
            # Your format: 2026-01-01 23:05:00.141Z
            temp['timestamp'] = pd.to_datetime(temp['timestamp'])
            temp.set_index('timestamp', inplace=True)
            
            # 3. Resample to M5 Candles
            # We use 'bid' as the price reference
            print(f"       Resampling {len(temp)} ticks to 5-min candles...")
            
            ohlc = temp['bid'].resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
            
            # Remove empty periods (weekends/holidays)
            ohlc.dropna(inplace=True)
            df_list.append(ohlc)
            
        except Exception as e:
            print(f"    ⚠️ Error reading {f}: {e}")

    if not df_list:
        sys.exit("❌ Failed to load any data.")

    # Merge all years
    df = pd.concat(df_list).sort_index()
    
    # --- CALCULATE INDICATORS ---
    print("    -> Calculating Indicators...")
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    
    df['ATR'] = talib.ATR(h, l, c, timeperiod=14)
    df['RSI'] = talib.RSI(c, timeperiod=14)
    
    df.dropna(inplace=True)
    df.to_parquet(CACHE_FILE)
    print(f"✅ Cache Built: {len(df)} candles saved to {CACHE_FILE}")
    return df

def load_data():
    if not os.path.exists(CACHE_FILE):
        return build_data_cache()
    
    print(f"[*] Loading cache: {CACHE_FILE}")
    try:
        return pd.read_parquet(CACHE_FILE)
    except:
        return build_data_cache()

# ==========================================
# PART 2: INFINITY RUNNER ENGINE (Unchanged)
# ==========================================
def calculate_trades(df):
    strategies = {}
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['ATR'].values
    times = df.index
    
    sl_mult = WINNING_PARAMS['sl_mult']
    tp_mult = WINNING_PARAMS['tp_mult']
    be_trigger = WINNING_PARAMS['be_trigger']
    step_mult = WINNING_PARAMS['step_mult']
    max_hold = WINNING_PARAMS['max_hold']
    n_candles = len(closes)

    for pat in TARGET_PORTFOLIO:
        if not hasattr(talib, pat): continue
        print(f"Simulating {pat}...")
        func = getattr(talib, pat)
        sigs = func(opens, highs, lows, closes)
        
        # Filter indices to ensure we have room for lookback and lookforward
        indices = np.where(sigs != 0)[0]
        indices = indices[(indices > 50) & (indices < n_candles - max_hold)]
        
        trade_log = [] 
        for idx in indices:
            entry_price = closes[idx]
            atr = atrs[idx]
            direction = 1 if sigs[idx] == 100 else -1
            
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
            be_dist = atr * be_trigger
            step_dist = atr * step_mult
            
            if direction == 1:
                current_sl = entry_price - sl_dist
                current_tp = entry_price + tp_dist
            else:
                current_sl = entry_price + sl_dist
                current_tp = entry_price - tp_dist
                
            is_runner = False
            exit_pnl = 0.0
            trade_finished = False # New flag to track completion
            
            for i in range(1, max_hold):
                curr = idx + i
                c_high = highs[curr]
                c_low = lows[curr]
                c_close = closes[curr] # We need close for the Time Stop
                
                if direction == 1:
                    # 1. Check Stop Loss
                    if c_low <= current_sl:
                        exit_pnl = current_sl - entry_price
                        trade_finished = True
                        break
                    
                    # 2. Check Take Profit (if not running yet)
                    if not is_runner and c_high >= current_tp:
                        exit_pnl = current_tp - entry_price
                        trade_finished = True
                        break
                    
                    # 3. Trailing Stop / Break Even Logic
                    if not is_runner and c_high >= (entry_price + be_dist):
                        is_runner = True; current_sl = entry_price; current_tp = 99999999
                    if is_runner:
                        steps = int((c_high - entry_price) / step_dist)
                        if steps >= 1:
                            new_sl = entry_price + (steps * step_dist) - step_dist
                            if new_sl > current_sl: current_sl = new_sl
                            
                else: # Short Direction
                    # 1. Check Stop Loss
                    if c_high >= current_sl:
                        exit_pnl = entry_price - current_sl
                        trade_finished = True
                        break
                    
                    # 2. Check Take Profit
                    if not is_runner and c_low <= current_tp:
                        exit_pnl = entry_price - current_tp
                        trade_finished = True
                        break
                    
                    # 3. Trailing Stop / Break Even Logic
                    if not is_runner and c_low <= (entry_price - be_dist):
                        is_runner = True; current_sl = entry_price; current_tp = -99999999
                    if is_runner:
                        steps = int((entry_price - c_low) / step_dist)
                        if steps >= 1:
                            new_sl = entry_price - (steps * step_dist) + step_dist
                            if new_sl < current_sl: current_sl = new_sl

                # 4. TIME STOP (Force Exit at end of Max Hold)
                # If we are at the last candle and haven't hit SL/TP yet
                if i == (max_hold - 1):
                    if direction == 1:
                        exit_pnl = c_close - entry_price
                    else:
                        exit_pnl = entry_price - c_close
                    trade_finished = True

            # Record trade if it finished (Even if PnL is 0.0, we record it now)
            if trade_finished:
                r_multiple = exit_pnl / sl_dist if sl_dist != 0 else 0
                trade_log.append((times[idx], r_multiple * 100)) 

        strategies[pat] = trade_log
    return strategies

# ==========================================
# PART 3: VISUALIZATION
# ==========================================
if __name__ == "__main__":
    df = load_data()
    print("\n-------------------------------------------")
    print(" ♾️ INFINITY RUNNER BACKTEST (From Tick Data)")
    print("-------------------------------------------\n")

    raw_strategies = calculate_trades(df)
    
    mask = pd.Series(True, index=df.index)
    if WINNING_PARAMS['use_rsi']:
        mask &= (df['RSI'] >= WINNING_PARAMS['rsi_min']) & (df['RSI'] <= WINNING_PARAMS['rsi_max'])
    valid_entries = mask.shift(1).fillna(False)
    
    equity_series = pd.Series(0.0, index=df.index)
    total_trades = 0
    for pat, trades in raw_strategies.items():
        for entry_time, pnl in trades:
            if valid_entries.loc[entry_time]:
                equity_series.loc[entry_time] += pnl 
                total_trades += 1
                
    equity_curve = equity_series.cumsum() + INITIAL_CAPITAL
    
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak)
    final_eq = equity_curve.iloc[-1]
    ret_pct = ((final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"💰 Final Equity: ${final_eq:,.2f}")
    print(f"📈 Total Return: {ret_pct:.2f}%")
    print(f"🔢 Total Trades: {total_trades}")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity', line=dict(color='#00ff00')), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode='lines', name='Drawdown', line=dict(color='#ff0000'), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='orange'), name='RSI'), row=3, col=1)
    fig.add_hrect(y0=30, y1=70, row=3, col=1, fillcolor="green", opacity=0.1, line_width=0)
    fig.update_layout(title="Strategy Performance: Tick-Resampled Data", template="plotly_dark", height=900)
    
    filename = "infinity_runner_ticks_report.html"
    fig.write_html(filename)
    print(f"\n✅ Report generated: {filename}")
    webbrowser.open(filename)