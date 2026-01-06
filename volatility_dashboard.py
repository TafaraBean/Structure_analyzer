import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
import talib 
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="MT5 Risk Manager", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ MT5 Dashboard: Dynamic Risk & Daily Targets")
st.markdown("""
**Objective:** Backtest with Professional Risk Management.
**Logic:**
1.  **Entry:** Smart Filters (Trend + Volatility Decline).
2.  **Sizing:** Dynamic. Calculates Lot Size based on % Risk of *current* equity.
3.  **Daily Target:** Stops trading for the day once a specific profit target is hit to lock in gains.
""")

# --- SIDEBAR: DATA CONTROLS ---
st.sidebar.header("🔌 Connection & Settings")

with st.sidebar.expander("MetaTrader 5 Credentials", expanded=True):
    env_login = os.getenv("MT5_LOGIN")
    default_login = int(env_login) if env_login else 0
    default_pass = os.getenv("MT5_PASSWORD", "")
    default_server = os.getenv("MT5_SERVER", "HFMarketsSA-Live2")
    
    mt5_login = st.number_input("MT5 Login ID", value=default_login, step=1)
    mt5_pass = st.text_input("MT5 Password", value=default_pass, type="password") 
    mt5_server = st.text_input("MT5 Server", value=default_server)

st.sidebar.markdown("---")
mt5_symbol = st.sidebar.text_input("Asset Symbol", value="USA100", help="Exact symbol name from Market Watch")

timeframe_map = {
    "1 Minute (M1)": mt5.TIMEFRAME_M1,
    "5 Minutes (M5)": mt5.TIMEFRAME_M5,
    "15 Minutes (M15)": mt5.TIMEFRAME_M15,
    "1 Hour (H1)": mt5.TIMEFRAME_H1,
    "4 Hours (H4)": mt5.TIMEFRAME_H4,
    "Daily (D1)": mt5.TIMEFRAME_D1
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(timeframe_map.keys()), index=1)
selected_tf_mt5 = timeframe_map[selected_tf_label]

candles_to_fetch = st.sidebar.slider("Number of Candles", 5000, 50000, 25000)

st.sidebar.markdown("---")
st.sidebar.header("💰 Money Management")
initial_capital = st.sidebar.number_input("Start Balance ($)", value=100000.0, step=1000.0)

# RISK LOGIC SWITCH
sizing_mode = st.sidebar.radio("Position Sizing Mode", ["Risk % of Equity", "Fixed Lot Size"])

risk_pct = 3.0
fixed_lot = 1.0

if sizing_mode == "Risk % of Equity":
    risk_pct = st.sidebar.number_input("Risk per Trade (%)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    st.sidebar.caption(f"Calculates lots to lose {risk_pct}% if SL is hit.")
else:
    fixed_lot = st.sidebar.number_input("Fixed Lot Size", min_value=0.01, value=1.0, step=0.01)

st.sidebar.header("🎯 Targets")
sl_multiplier = st.sidebar.number_input("Stop Loss (x ATR)", value=1.5)
tp_multiplier = st.sidebar.number_input("Take Profit (x ATR)", value=2.0)
max_hold_bars = st.sidebar.number_input("Max Time Stop", value=50)

# --- NEW: DAILY TARGET ---
st.sidebar.subheader("Daily Limits")
use_daily_target = st.sidebar.checkbox("Use Daily Profit Target", value=True)
daily_target_amt = st.sidebar.number_input("Daily Target ($)", value=200.0, step=50.0)
# -------------------------

st.sidebar.header("🛡️ Risk Controls")
use_trailing = st.sidebar.checkbox("Trailing Stop", value=True)
trail_trigger_pct = st.sidebar.slider("Trail Trigger %", 0.1, 0.9, 0.5)
max_consecutive_losses = st.sidebar.number_input("Max Losses", value=3)
use_cooldown = st.sidebar.checkbox("Cooldown", value=True)
cooldown_bars = st.sidebar.number_input("Cooldown Bars", value=50)

st.sidebar.header("Signal Filters")
# The "Smart" Filters
use_trend_filter = st.sidebar.checkbox("Trend Filter (MA Velocity)", value=True)
use_vol_filter = st.sidebar.checkbox("Volatility Decline Filter", value=True)

atr_period = st.sidebar.number_input("ATR Period", value=14)
vol_threshold_ma = st.sidebar.slider("Vol Threshold", 20, 200, 150)
trend_ma_period = st.sidebar.number_input("Trend MA Period", 20, 200, 50)

# Patterns
st.sidebar.subheader("Patterns")
show_engulfing = st.sidebar.checkbox("Engulfing", value=True)
show_hammer = st.sidebar.checkbox("Hammer (Bull)", value=True)
show_star = st.sidebar.checkbox("Shooting Star (Bear)", value=True)
show_hanging = st.sidebar.checkbox("Hanging Man (Bear)", value=True)

live_mode = st.sidebar.checkbox("🔴 Live Refresh", value=False)

# --- HELPER FUNCTIONS ---

def get_mt5_data(login, password, server, symbol, timeframe, num_candles):
    # 1. Initialize
    if not mt5.initialize():
        return None, None, f"MT5 Init Failed: {mt5.last_error()}"
    
    # 2. Login (Only if credentials provided)
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
            err = mt5.last_error()
            mt5.shutdown()
            return None, None, f"Login Failed: {err}"
    
    # 3. Enable Symbol & Get Contract Size (MUST BE DONE BEFORE SHUTDOWN)
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        return None, None, f"Symbol '{symbol}' not found in Market Watch."
    
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        mt5.shutdown()
        return None, None, f"Could not get info for {symbol}"
        
    # SAVE THIS NOW because we can't get it after shutdown
    contract_size = sym_info.trade_contract_size 
    
    # 4. Retry Loop for Data Fetching
    attempts = 0
    max_retries = 5
    rates = None
    
    with st.spinner(f"Requesting {num_candles} candles for {symbol}..."):
        while attempts < max_retries:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
            
            if rates is not None and len(rates) > 0:
                break
            
            attempts += 1
            time.sleep(1) # Wait for download
            
    mt5.shutdown() # <--- Connection closes here
    
    # 5. Final Validation
    if rates is None:
        return None, None, "Timed out waiting for data. MT5 is downloading history in the background—try again in 10 seconds."
    
    if len(rates) == 0:
        return None, None, "Data received but was empty."

    # 6. Process
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Return the variable we saved earlier, NOT a new API call
    return df, contract_size, "Success"

def identify_signals(df, atr_period, vol_ma, trend_ma):
    hi, lo, cl, op = df['high'].values, df['low'].values, df['close'].values, df['open'].values
    
    # 1. Volatility & Trend
    df['ATR'] = talib.ATR(hi, lo, cl, timeperiod=atr_period)
    df['ATR_MA'] = df['ATR'].rolling(window=vol_ma).mean()
    
    # Logic: ATR Declining after being above threshold
    df['ATR_Declining'] = (df['ATR'] < df['ATR'].shift(1)) & (df['ATR'].shift(1) > df['ATR_MA'].shift(1))
    
    # Trend Velocity
    df['Trend_MA'] = talib.SMA(cl, timeperiod=trend_ma)
    df['MA_Velocity'] = df['Trend_MA'].diff() # Pos = Up, Neg = Down

    # 2. The 4 Patterns (TA-Lib)
    df['Engulf_Bull'] = talib.CDLENGULFING(op, hi, lo, cl) == 100
    df['Engulf_Bear'] = talib.CDLENGULFING(op, hi, lo, cl) == -100
    df['Hammer'] = talib.CDLHAMMER(op, hi, lo, cl) == 100
    df['ShootingStar'] = talib.CDLSHOOTINGSTAR(op, hi, lo, cl) == -100
    df['HangingMan'] = talib.CDLHANGINGMAN(op, hi, lo, cl) == -100

    return df

def run_backtest_dynamic(df, contract_size, start_bal, sizing_type, risk_val, sl_mult, tp_mult, max_bars, patterns_active, max_losses, do_cooldown, cooldown_len, do_trail, trail_pct, use_trend, use_vol, use_daily_target, daily_target):
    
    # 1. Base Signals
    buy_mask = pd.Series(False, index=df.index)
    sell_mask = pd.Series(False, index=df.index)
    
    if patterns_active['Engulfing']:
        buy_mask |= df['Engulf_Bull']
        sell_mask |= df['Engulf_Bear']
    if patterns_active['Hammer']: buy_mask |= df['Hammer']
    if patterns_active['ShootingStar']: sell_mask |= df['ShootingStar']
    if patterns_active['HangingMan']: sell_mask |= df['HangingMan']
    
    # 2. Apply Filters
    if use_trend:
        buy_mask &= (df['MA_Velocity'] > 0)
        sell_mask &= (df['MA_Velocity'] < 0)
        
    if use_vol:
        buy_mask &= df['ATR_Declining']
        sell_mask &= df['ATR_Declining']

    potential_signals = pd.DataFrame(index=df.index)
    potential_signals['Signal'] = 0
    potential_signals.loc[buy_mask, 'Signal'] = 1
    potential_signals.loc[sell_mask, 'Signal'] = -1
    potential_signals.loc[buy_mask & sell_mask, 'Signal'] = 0 
    
    # 3. Execution Loop
    trades_log = []
    consecutive_losses = 0
    cooldown_until = -1
    
    # Track Running Balance for compounding
    current_equity = start_bal
    
    # Track Daily PnL (Date -> Float)
    daily_pnl_map = {}
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['ATR'].values
    times = df.index
    signals = potential_signals['Signal'].values
    n_rows = len(df)
    
    for i in range(n_rows - max_bars - 1):
        if i < cooldown_until: continue
        
        # Stop if blown
        if current_equity <= 0: break
        
        # --- CHECK DAILY TARGET ---
        current_date = times[i].date()
        if use_daily_target:
            today_realized = daily_pnl_map.get(current_date, 0.0)
            if today_realized >= daily_target:
                continue # Skip trading for the rest of this day
        # --------------------------
        
        sig = signals[i]
        if sig == 0: continue
        if np.isnan(atrs[i]): continue
            
        # Entry
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        current_atr = atrs[i]
        
        # Targets
        sl_dist = sl_mult * current_atr
        tp_dist = tp_mult * current_atr
        
        # --- DYNAMIC SIZING LOGIC ---
        trade_lots = 0.0
        
        if sizing_type == "Risk % of Equity":
            # 1. Calc Risk Amount ($)
            risk_amt = current_equity * (risk_val / 100.0)
            
            # 2. Calc Lot Size
            if sl_dist > 0:
                calc_lots = risk_amt / (contract_size * sl_dist)
            else:
                calc_lots = 0.01
                
            # Rounding and Limits (Min 0.01)
            trade_lots = max(0.01, round(calc_lots, 2))
        else:
            trade_lots = risk_val # Fixed Lot
            
        current_sl = entry_price - sl_dist if sig == 1 else entry_price + sl_dist
        tp_price = entry_price + tp_dist if sig == 1 else entry_price - tp_dist
        trail_trigger = entry_price + (tp_dist * trail_pct) if sig == 1 else entry_price - (tp_dist * trail_pct)
        
        exit_price, exit_idx, exit_reason = 0.0, -1, "Time"
        
        for j in range(entry_idx, entry_idx + max_bars):
            if j >= n_rows: break
            c_high, c_low = highs[j], lows[j]
            
            # SL
            if (sig == 1 and c_low <= current_sl) or (sig == -1 and c_high >= current_sl):
                exit_price = current_sl
                exit_idx = j
                exit_reason = "SL"
                if do_trail and abs(current_sl - entry_price) < 0.0001: exit_reason = "BE (Trailing)"
                elif do_trail and ((sig==1 and current_sl > entry_price) or (sig==-1 and current_sl < entry_price)): exit_reason = "Profit (Trailing)"
                break
                
            # TP
            if (sig == 1 and c_high >= tp_price) or (sig == -1 and c_low <= tp_price):
                exit_price = tp_price
                exit_idx = j
                exit_reason = "TP"
                break
                
            # Trail
            if do_trail:
                if sig == 1:
                    if c_high >= trail_trigger:
                        new_sl = c_high - sl_dist
                        if new_sl > current_sl: current_sl = new_sl
                else:
                    if c_low <= trail_trigger:
                        new_sl = c_low + sl_dist
                        if new_sl < current_sl: current_sl = new_sl
                        
        if exit_idx == -1:
            exit_idx = min(entry_idx + max_bars - 1, n_rows - 1)
            exit_price = closes[exit_idx]
        
        # Calc PnL
        pnl = (exit_price - entry_price) * sig * trade_lots * contract_size
        
        # Update Balance (Compounding happens here)
        current_equity += pnl
        
        # --- UPDATE DAILY PNL ---
        exit_date = times[exit_idx].date()
        daily_pnl_map[exit_date] = daily_pnl_map.get(exit_date, 0.0) + pnl
        # ------------------------
        
        trades_log.append({
            'Entry_Time': times[entry_idx], 'Signal': "BUY" if sig==1 else "SELL",
            'Entry_Price': entry_price, 'Exit_Price': exit_price,
            'Lots': trade_lots, # Log the dynamic lot size
            'PnL': pnl, 
            'Equity': current_equity,
            'Reason': exit_reason,
            'Vol_ATR': current_atr
        })
        
        if pnl <= 0:
            consecutive_losses += 1
            if consecutive_losses >= max_losses and do_cooldown:
                cooldown_until = i + cooldown_len
                consecutive_losses = 0
        else:
            consecutive_losses = 0
            
        cooldown_until = max(cooldown_until, exit_idx)

    if not trades_log: return None, {}
    df_res = pd.DataFrame(trades_log)
    df_res.set_index('Entry_Time', inplace=True)
    
    metrics = {
        "Net PnL": df_res['PnL'].sum(),
        "Win Rate": (df_res['PnL'] > 0).mean() * 100,
        "Trades": len(df_res),
        "Final Equity": df_res['Equity'].iloc[-1],
        "Max Drawdown": (df_res['Equity'].cummax() - df_res['Equity']).max()
    }
    return df_res, metrics

# --- MAIN ---

if fetch_needed := st.sidebar.button("🚀 Run Risk Backtest") or live_mode:
    with st.spinner("Calculating Risk, Compounding & Targets..."):
        df, contract, msg = get_mt5_data(mt5_login, mt5_pass, mt5_server, mt5_symbol, selected_tf_mt5, candles_to_fetch)
        if df is not None:
            df = identify_signals(df, atr_period, vol_threshold_ma, trend_ma_period)
            
            pats = {
                'Engulfing': show_engulfing, 'Hammer': show_hammer,
                'ShootingStar': show_star, 'HangingMan': show_hanging
            }
            
            # Determine Risk Val
            risk_val = risk_pct if sizing_mode == "Risk % of Equity" else fixed_lot
            
            trades, metrics = run_backtest_dynamic(
                df, contract, initial_capital, sizing_mode, risk_val, 
                sl_multiplier, tp_multiplier, max_hold_bars, pats, 
                max_consecutive_losses, use_cooldown, cooldown_bars, 
                use_trailing, trail_trigger_pct, use_trend_filter, use_vol_filter,
                use_daily_target, daily_target_amt
            )
            
            if trades is not None:
                st.subheader(f"📊 Results ({sizing_mode}: {risk_val}{'%' if sizing_mode.startswith('Risk') else ' Lots'})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Net PnL", f"${metrics['Net PnL']:,.2f}")
                c2.metric("Final Equity", f"${metrics['Final Equity']:,.2f}")
                c3.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
                c4.metric("Max Drawdown", f"${metrics['Max Drawdown']:,.2f}")
                
                fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=trades.index, y=trades['Equity'], name='Equity', fill='tozeroy', line=dict(color='#00CC96')), row=1, col=1)
                
                # Bar chart for PnL
                fig.add_trace(go.Bar(x=trades.index, y=trades['PnL'], name='PnL', marker_color=np.where(trades['PnL']>0, 'green', 'red')), row=2, col=1)
                fig.update_layout(height=600, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(trades.style.format({'Entry_Price':'{:.2f}', 'Exit_Price':'{:.2f}', 'PnL':'${:.2f}', 'Equity':'${:.2f}', 'Lots':'{:.2f}'}))
            else:
                st.warning("No trades found.")
        else:
            st.error(msg)