import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import math

# --- ⚙️ CONFIGURATION ---
SYMBOL = "XAUUSDm"       
TIMEFRAME = mt5.TIMEFRAME_M5
HOURS_TO_TEST = 500     
LOT_SIZE = 0.3

# --- 🏆 STRATEGY SETTINGS ---
PARAMS = {
    'rsi_min': 30,
    'rsi_max': 70,
    'sl_mult': 1.0,      
    'tp_mult': 1.5,      
    'be_trigger': 1.0,   
    'step_mult': 0.5    
}

TARGET_PATTERNS = [
    'CDLLONGLEGGEDDOJI', 'CDLRICKSHAWMAN', 'CDLHIGHWAVE', 
    'CDLENGULFING', 'CDLBELTHOLD'
]

# ==========================================
# 1. DATA ENGINE (Hybrid: Candles + Ticks)
# ==========================================
def get_data():
    if not mt5.initialize(): sys.exit("❌ MT5 Init Failed")
    
    # Time window
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=HOURS_TO_TEST + 5) # Buffer for indicators
    
    # 1. Fetch M5 Candles (For Signals)
    print(f"[*] Fetching Candles...")
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_time, end_time)
    df_m5 = pd.DataFrame(rates)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
    
    # Indicators
    df_m5['RSI'] = ta.rsi(df_m5['close'], length=14)
    df_m5['ATR'] = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=14)
    
    # Patterns
    op = df_m5['open'].values; hi = df_m5['high'].values
    lo = df_m5['low'].values; cl = df_m5['close'].values
    df_m5['Signal'] = 0
    for pat in TARGET_PATTERNS:
        func = getattr(talib, pat)
        res = func(op, hi, lo, cl)
        mask = res != 0
        df_m5.loc[mask, 'Signal'] = res[mask]

    # Filter to test window
    test_start = end_time - timedelta(hours=HOURS_TO_TEST)
    df_m5 = df_m5[df_m5['time'] >= test_start].reset_index(drop=True)

    # 2. Fetch Ticks (For Execution)
    print(f"[*] Fetching TICKS (Precision Mode)...")
    ticks = mt5.copy_ticks_range(SYMBOL, test_start, end_time, mt5.COPY_TICKS_ALL)
    
    if ticks is None: sys.exit("❌ No ticks found.")
    
    # Convert ticks to a structured array for fast lookups
    # Structure: [time (s), bid, ask]
    # We use 'last' price if available, else bid/ask depending on direction
    return df_m5, ticks

# ==========================================
# 2. TICK SIMULATOR
# ==========================================
def run_tick_backtest(df_m5, ticks):
    symbol_info = mt5.symbol_info(SYMBOL)
    contract_size = symbol_info.trade_contract_size
    print(f"[*] Contract Size: {contract_size} (1.00 move = ${contract_size * LOT_SIZE:.2f})")
    
    trades = []
    equity_curve = []
    
    # Index ticks by time for fast seeking
    # We use searchsorted on the timestamp array
    tick_times = ticks['time'] 
    
    i = 0
    while i < len(df_m5):
        candle = df_m5.iloc[i]
        
        # Check Signal on CLOSED candle (i-1 logic handled by iterating forward)
        # Actually, in df_m5 loop, 'candle' represents the signal candle. 
        # Trade starts at the OPEN of the NEXT candle (or immediately after signal close).
        
        signal = candle['Signal']
        rsi = candle['RSI']
        
        # Filters
        if signal != 0 and (PARAMS['rsi_min'] <= rsi <= PARAMS['rsi_max']):
            
            # --- ENTER TRADE ---
            entry_time = candle['time'] + timedelta(minutes=5) # Open of next candle
            
            # Find the tick closest to entry time
            # Using searchsorted to find index in tick array
            start_tick_idx = np.searchsorted(tick_times, entry_time.timestamp())
            
            if start_tick_idx >= len(ticks): break
            
            # Get Entry Price (Ask for Buy, Bid for Sell)
            entry_tick = ticks[start_tick_idx]
            
            atr = candle['ATR']
            sl_dist = atr * PARAMS['sl_mult']
            tp_dist = atr * PARAMS['tp_mult']
            
            be_trigger_dist = atr * PARAMS['be_trigger']
            step_dist = atr * PARAMS['step_mult']
            
            type_str = "BUY" if signal == 100 else "SELL"
            
            if signal == 100:
                entry_price = entry_tick['ask']
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                entry_price = entry_tick['bid']
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist
            
            # --- TICK LOOP (THE RUNNER) ---
            # Iterate ticks from entry until closed
            
            exit_price = 0.0
            exit_reason = ""
            tp_active = True # TP is active until BE hit
            
            for j in range(start_tick_idx, len(ticks)):
                curr_tick = ticks[j]
                
                # Check Time Limit (optional, prevents infinite loops)
                if curr_tick['time'] - entry_tick['time'] > 86400: # 24h max hold
                    exit_price = curr_tick['bid'] if signal==100 else curr_tick['ask']
                    exit_reason = "Timeout"
                    break
                
                # Update Prices
                curr_bid = curr_tick['bid']
                curr_ask = curr_tick['ask']
                
                # 1. CHECK EXIT (SL / TP)
                if signal == 100: # BUY
                    # Hit SL? (Sell at Bid)
                    if curr_bid <= sl:
                        exit_price = sl # Assume fill at SL (or curr_bid for slippage)
                        exit_reason = "SL Hit"
                        break
                    # Hit TP? (Sell at Bid)
                    if tp_active and curr_bid >= tp:
                        exit_price = tp
                        exit_reason = "Fixed TP"
                        break
                        
                else: # SELL
                    # Hit SL? (Buy at Ask)
                    if curr_ask >= sl:
                        exit_price = sl
                        exit_reason = "SL Hit"
                        break
                    # Hit TP? (Buy at Ask)
                    if tp_active and curr_ask <= tp:
                        exit_price = tp
                        exit_reason = "Fixed TP"
                        break
                
                # 2. RUNNER LOGIC (Manage SL/TP)
                if signal == 100: # BUY
                    dist = curr_bid - entry_price
                    
                    # BE Trigger
                    if tp_active and dist >= be_trigger_dist:
                        tp_active = False # Remove TP
                        sl = entry_price  # Move to BE
                        # print(f"    (Tick {j}) BE Triggered")
                        
                    # Infinity Step
                    if not tp_active:
                        steps = math.floor(dist / step_dist)
                        if steps >= 1:
                            new_sl = entry_price + (steps * step_dist) - step_dist
                            if new_sl > sl: sl = new_sl
                            
                else: # SELL
                    dist = entry_price - curr_ask
                    
                    if tp_active and dist >= be_trigger_dist:
                        tp_active = False
                        sl = entry_price
                        
                    if not tp_active:
                        steps = math.floor(dist / step_dist)
                        if steps >= 1:
                            new_sl = entry_price - (steps * step_dist) + step_dist
                            if new_sl < sl or sl == 0: sl = new_sl

            # --- TRADE RESULT ---
            if exit_price != 0:
                if signal == 100:
                    raw_profit = exit_price - entry_price
                else:
                    raw_profit = entry_price - exit_price
                    
                # Dollar Value Calculation
                # Standard Formula: Price_Diff * Contract_Size * Volume
                usd_profit = raw_profit * contract_size * LOT_SIZE
                
                trades.append({
                    'Time': datetime.fromtimestamp(entry_tick['time']),
                    'Type': type_str,
                    'Entry': entry_price,
                    'Exit': exit_price,
                    'SL': sl,
                    'Reason': exit_reason,
                    'Profit_USD': usd_profit
                })
                
                # Advance Main Loop: Skip candles covered by this trade
                # Find which candle index corresponds to the exit time
                exit_time_s = ticks[j]['time']
                
                # Fast forward 'i' to the candle after exit
                while i < len(df_m5) and df_m5.iloc[i]['time'].timestamp() < exit_time_s:
                    i += 1
                continue # Skip the i+=1 below to avoid double counting
                
        i += 1
        
    return pd.DataFrame(trades)

# ==========================================
# 3. REPORTING
# ==========================================
if __name__ == "__main__":
    df_m5, ticks = get_data()
    print(f"[*] Simulating Strategy on {len(ticks)} ticks...")
    
    df_res = run_tick_backtest(df_m5, ticks)
    
    print("\n" + "="*50)
    print(f"📊 PRECISION BACKTEST RESULTS (Last {HOURS_TO_TEST} Hours)")
    print("="*50)
    
    if df_res.empty:
        print("No trades found.")
    else:
        # Cumulative Equity
        df_res['Equity'] = df_res['Profit_USD'].cumsum()
        
        total_pnl = df_res['Profit_USD'].sum()
        win_rate = (len(df_res[df_res['Profit_USD'] > 0]) / len(df_res)) * 100
        
        print(f"Total Trades : {len(df_res)}")
        print(f"Win Rate     : {win_rate:.1f}%")
        print(f"Net Profit   : ${total_pnl:.2f}")
        print("-" * 50)
        print(df_res[['Time', 'Type', 'Entry', 'Exit', 'Reason', 'Profit_USD']].to_string(index=False))
        print("-" * 50)
        
        # Plot
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(
            x=df_res['Time'], y=df_res['Equity'],
            mode='lines+markers', name='Equity',
            line=dict(color='#00FF00')
        ))
        fig.update_layout(title=f"Precise PnL (0.01 Lots) - {SYMBOL}", template="plotly_dark")
        fig.write_html("tick_backtest.html")
        print("✅ Chart saved: tick_backtest.html")