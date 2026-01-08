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
st.set_page_config(page_title="MT5 Base Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ MT5 Base Dashboard: Fixed Lot & Trailing")
st.markdown("""
**Objective:** Backtest the core strategy (Patterns + Volatility + Trailing).
**Logic:**
1.  **Entry:** Pattern (Engulfing/Hammer) + High Volatility.
2.  **Sizing:** Fixed Lot Size (Standard).
3.  **Exit:** 1:2 Risk/Reward with Dynamic Trailing.
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

candles_to_fetch = st.sidebar.slider("Number of Candles", 1000, 10000, 5000)

st.sidebar.markdown("---")
st.sidebar.header("💰 Position Sizing")
trade_lot_size = st.sidebar.number_input("Lot Size (Standard)", min_value=0.01, value=1.0, step=0.01, help="Fixed volume per trade.")
initial_capital = st.sidebar.number_input("Start Balance ($)", value=100000.0, step=1000.0)

st.sidebar.subheader("🎯 Target & Trail")
sl_multiplier = st.sidebar.number_input("Stop Loss (x ATR)", min_value=0.1, value=1.0)
tp_multiplier = st.sidebar.number_input("Take Profit (x ATR)", min_value=0.1, value=2.0)
max_hold_bars = st.sidebar.number_input("Max Time Stop (Bars)", min_value=5, value=50)

st.sidebar.subheader("⛓️ Trailing Logic")
use_trailing = st.sidebar.checkbox("Enable Trailing Stop", value=True)
trail_trigger_pct = st.sidebar.slider("Trigger Trail at % to Target", 0.1, 0.9, 0.5)

st.sidebar.header("🛡️ Risk Management")
max_consecutive_losses = st.sidebar.number_input("Stop after X Consecutive Losses", min_value=1, value=3)
use_cooldown = st.sidebar.checkbox("Use Cooldown?", value=True)
cooldown_bars = st.sidebar.number_input("Cooldown Duration (Bars)", min_value=1, value=50)

st.sidebar.header("⚔️ Signal Filters")
atr_period = st.sidebar.number_input("ATR Period", value=14)
vol_threshold_ma = st.sidebar.slider("Volatility Threshold (ATR MA)", 20, 200, 50)

# Pattern Selectors
show_engulfing = st.sidebar.checkbox("Engulfing", value=True)
show_hammer = st.sidebar.checkbox("Hammer (Bull)", value=True)
show_star = st.sidebar.checkbox("Shooting Star (Bear)", value=True)
show_hanging = st.sidebar.checkbox("Hanging Man (Bear)", value=True)

live_mode = st.sidebar.checkbox("🔴 Enable Live Refresh (10s)", value=False)

# --- HELPER FUNCTIONS ---

def get_mt5_data(login, password, server, symbol, timeframe, num_candles):
    if not mt5.initialize():
        return None, None, f"MT5 Init Failed: {mt5.last_error()}"
    
    if not mt5.login(login=int(login), password=password, server=server):
        err = mt5.last_error()
        mt5.shutdown()
        return None, None, f"MT5 Login Failed: {err}"
    
    # FETCH SYMBOL INFO (Important for Contract Size)
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        mt5.shutdown()
        return None, None, f"Symbol {symbol} not found."

    contract_size = sym_info.trade_contract_size
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        return None, None, "No data received from MT5"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, contract_size, "Success"

def identify_patterns_talib(df, atr_period=14, vol_ma=50):
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    op = df['open'].values
    
    df['ATR'] = talib.ATR(hi, lo, cl, timeperiod=atr_period)
    df['ATR_MA'] = df['ATR'].rolling(window=vol_ma).mean()
    # Simple High Volatility Check (No "Smart" decline logic yet)
    df['High_Vol'] = df['ATR'] > df['ATR_MA']

    engulf = talib.CDLENGULFING(op, hi, lo, cl)
    df['Engulf_Bull'] = engulf == 100
    df['Engulf_Bear'] = engulf == -100
    df['Hammer'] = talib.CDLHAMMER(op, hi, lo, cl) == 100
    df['ShootingStar'] = talib.CDLSHOOTINGSTAR(op, hi, lo, cl) == -100
    df['HangingMan'] = talib.CDLHANGINGMAN(op, hi, lo, cl) == -100

    return df

def run_fixed_lot_backtest(df, contract_size, lot_size, start_bal, sl_mult, tp_mult, max_bars, patterns_active, max_losses, do_cooldown, cooldown_len, do_trail, trail_pct):
    """
    Backtest using Fixed Lot PnL Calculation.
    """
    # 1. Signals
    buy_mask = pd.Series(False, index=df.index)
    sell_mask = pd.Series(False, index=df.index)
    
    if patterns_active['Engulfing']:
        buy_mask |= df['Engulf_Bull']
        sell_mask |= df['Engulf_Bear']
    if patterns_active['Hammer']:
        buy_mask |= df['Hammer']
    if patterns_active['ShootingStar']:
        sell_mask |= df['ShootingStar']
    if patterns_active['HangingMan']:
        sell_mask |= df['HangingMan']
        
    buy_mask &= df['High_Vol']
    sell_mask &= df['High_Vol']
    
    potential_signals = pd.DataFrame(index=df.index)
    potential_signals['Signal'] = 0
    potential_signals.loc[buy_mask, 'Signal'] = 1
    potential_signals.loc[sell_mask, 'Signal'] = -1
    potential_signals.loc[buy_mask & sell_mask, 'Signal'] = 0 
    
    # 2. Iterate
    trades_log = []
    consecutive_loss_count = 0
    circuit_breaker_active_until = -1
    
    # Arrays
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['ATR'].values
    times = df.index
    signals = potential_signals['Signal'].values
    
    n_rows = len(df)
    
    for i in range(n_rows - max_bars - 1):
        
        if i < circuit_breaker_active_until: continue
        sig = signals[i]
        if sig == 0: continue
            
        # SETUP TRADE
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        current_atr = atrs[i]
        
        if np.isnan(current_atr): continue

        # Targets
        sl_dist = sl_mult * current_atr
        tp_dist = tp_mult * current_atr
        
        current_sl = 0.0
        tp_price = 0.0
        trail_trigger_price = 0.0
        
        if sig == 1: # BUY
            current_sl = entry_price - sl_dist
            tp_price = entry_price + tp_dist
            trail_trigger_price = entry_price + (tp_dist * trail_pct)
        else: # SELL
            current_sl = entry_price + sl_dist
            tp_price = entry_price - tp_dist
            trail_trigger_price = entry_price - (tp_dist * trail_pct)
            
        # SCAN
        exit_price = 0.0
        exit_idx = -1
        exit_reason = "Time"
        
        for j in range(entry_idx, entry_idx + max_bars):
            if j >= n_rows: break
            
            c_high = highs[j]
            c_low = lows[j]
            
            # 1. SL CHECK
            sl_hit = False
            if sig == 1:
                if c_low <= current_sl: sl_hit = True
            else:
                if c_high >= current_sl: sl_hit = True
            
            if sl_hit:
                exit_price = current_sl
                exit_idx = j
                exit_reason = "SL"
                if do_trail and abs(current_sl - entry_price) < 0.0001: exit_reason = "BE (Trailing)"
                elif do_trail and ((sig==1 and current_sl > entry_price) or (sig==-1 and current_sl < entry_price)): exit_reason = "Profit (Trailing)"
                break
                
            # 2. TP CHECK
            tp_hit = False
            if sig == 1:
                if c_high >= tp_price: tp_hit = True
            else:
                if c_low <= tp_price: tp_hit = True
            
            if tp_hit:
                exit_price = tp_price
                exit_idx = j
                exit_reason = "TP"
                break
                
            # 3. TRAIL UPDATE
            if do_trail:
                if sig == 1: 
                    if c_high >= trail_trigger_price:
                        potential_sl = c_high - sl_dist
                        if potential_sl > current_sl: current_sl = potential_sl
                else: 
                    if c_low <= trail_trigger_price:
                        potential_sl = c_low + sl_dist
                        if potential_sl < current_sl: current_sl = potential_sl
                            
        # Time Stop
        if exit_idx == -1:
            exit_idx = min(entry_idx + max_bars - 1, n_rows - 1)
            exit_price = closes[exit_idx]
            exit_reason = "Time"

        # Calc PnL (DOLLAR VALUE)
        price_diff = exit_price - entry_price
        # PnL = Diff * Signal * Lot * Contract
        realized_pnl = price_diff * sig * lot_size * contract_size
        
        is_win = realized_pnl > 0
        
        trades_log.append({
            'Entry_Time': times[entry_idx],
            'Signal': sig,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL': realized_pnl,
            'Reason': exit_reason,
            'Consecutive_Losses': consecutive_loss_count
        })
        
        # Risk Logic
        if not is_win:
            consecutive_loss_count += 1
            if consecutive_loss_count >= max_losses:
                if do_cooldown:
                    circuit_breaker_active_until = i + cooldown_len
                    consecutive_loss_count = 0
                else:
                    break
        else:
            consecutive_loss_count = 0
            
        circuit_breaker_active_until = max(circuit_breaker_active_until, exit_idx)

    # 3. Results
    if not trades_log:
        return None, {}
        
    trades_df = pd.DataFrame(trades_log)
    trades_df.set_index('Entry_Time', inplace=True)
    
    # EQUITY CURVE (Sum of PnL)
    trades_df['Equity'] = start_bal + trades_df['PnL'].cumsum()
    
    metrics = {
        "Win Rate": (trades_df['PnL'] > 0).mean() * 100,
        "Trades": len(trades_df),
        "Total PnL": trades_df['PnL'].sum(),
        "Return %": (trades_df['PnL'].sum() / start_bal) * 100,
        "Final Equity": trades_df['Equity'].iloc[-1],
        "TP Hits": len(trades_df[trades_df['Reason'] == 'TP']),
        "SL Hits": len(trades_df[trades_df['Reason'] == 'SL']),
        "Trail Saves": len(trades_df[trades_df['Reason'].str.contains('Trailing')])
    }
    
    return trades_df, metrics

# --- MAIN LOGIC ---

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

fetch_needed = st.sidebar.button("🚀 Run Base Strategy", type="primary") or live_mode

if fetch_needed:
    with st.spinner("Calculating PnL..."):
        df, contract_size, msg = get_mt5_data(mt5_login, mt5_pass, mt5_server, mt5_symbol, selected_tf_mt5, candles_to_fetch)
        
        if df is not None:
            df = identify_patterns_talib(df, atr_period, vol_threshold_ma)
            
            patterns_active = {
                'Engulfing': show_engulfing, 'Hammer': show_hammer,
                'ShootingStar': show_star, 'HangingMan': show_hanging
            }
            
            trades, metrics = run_fixed_lot_backtest(
                df, contract_size, trade_lot_size, initial_capital,
                sl_multiplier, tp_multiplier, max_hold_bars, 
                patterns_active, max_consecutive_losses, use_cooldown, cooldown_bars,
                use_trailing, trail_trigger_pct
            )
            
            if trades is not None:
                st.subheader(f"💰 PnL Report ({trade_lot_size} Lots on {mt5_symbol})")
                st.caption(f"Contract Size detected: {contract_size}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Net PnL", f"${metrics['Total PnL']:,.2f}", f"{metrics['Return %']:.2f}%")
                c2.metric("Final Balance", f"${metrics['Final Equity']:,.2f}")
                c3.metric("Win Rate", f"{metrics['Win Rate']:.1f}%", f"{metrics['Trades']} Trades")
                c4.metric("Trail Saves", f"{metrics['Trail Saves']}", "Breakeven/Locked")
                
                # Equity Curve
                fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=trades.index, y=trades['Equity'], mode='lines', name='Equity ($)', fill='tozeroy', line=dict(color='#00CC96')), row=1, col=1)
                
                # Color code
                colors = ['green' if r > 0 else 'red' for r in trades['PnL']]
                fig.add_trace(go.Bar(x=trades.index, y=trades['PnL'], name='PnL ($)', marker_color=colors), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", showlegend=False, margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Detailed Trade Ledger"):
                    st.dataframe(trades[['Signal', 'Entry_Price', 'Exit_Price', 'Reason', 'PnL', 'Equity']].style.format({
                        'Entry_Price': '{:.2f}', 'Exit_Price': '{:.2f}', 'PnL': '${:,.2f}', 'Equity': '${:,.2f}'
                    }))
            else:
                st.warning("No trades found.")
        else:
            st.error(msg)

    if live_mode:
        time.sleep(10)
        st.rerun()

else:
    st.info("👈 Click 'Run Base Strategy'.")