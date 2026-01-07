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
st.set_page_config(page_title="MT5 Hybrid Pattern Manager", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ MT5 Dashboard: Hybrid Pattern Filters")
st.markdown("""
**Objective:** High-Quality Signal Filtering.
**Logic:**
1.  **Detection:** TA-Lib (Standard Definitions).
2.  **Confirmation:** Custom Geometry (Min Wick Length & Min Body Size).
3.  **Context:** Trend + Volatility Filters.
4.  **Limits:** Daily Profit Target & **Max Daily Loss**.
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

candles_to_fetch = st.sidebar.slider("Number of Candles", 5000, 100000, 50000)

st.sidebar.markdown("---")
st.sidebar.header("💰 Money Management")
initial_capital = st.sidebar.number_input("Start Balance ($)", value=100000.0, step=1000.0)

# RISK LOGIC SWITCH
sizing_mode = st.sidebar.radio("Position Sizing Mode", ["Risk % of Equity", "Fixed Lot Size"])

risk_pct = 3.0
fixed_lot = 1.0

if sizing_mode == "Risk % of Equity":
    risk_pct = st.sidebar.number_input("Risk per Trade (%)", min_value=0.1, max_value=20.0, value=3.0, step=0.1)
else:
    fixed_lot = st.sidebar.number_input("Fixed Lot Size", min_value=0.01, value=1.0, step=0.01)

# --- TRADING COSTS ---
st.sidebar.header("💸 Trading Costs")
spread_pips = st.sidebar.number_input("Spread (Pips)", value=0.2, step=0.1)

st.sidebar.header("🎯 Targets")
sl_multiplier = st.sidebar.number_input("Stop Loss (x ATR)", value=1.5)
tp_multiplier = st.sidebar.number_input("Take Profit (x ATR)", value=2.0)
max_hold_bars = st.sidebar.number_input("Max Time Stop", value=50)

# --- DAILY LIMITS ---
st.sidebar.subheader("Daily Limits")
use_daily_target = st.sidebar.checkbox("Use Daily Profit Target", value=True)
daily_target_amt = st.sidebar.number_input("Daily Target ($)", value=200.0, step=50.0)

# NEW: Max Loss Input
use_daily_loss = st.sidebar.checkbox("Use Daily Max Loss", value=True)
daily_loss_amt = st.sidebar.number_input("Max Daily Loss ($)", value=100.0, step=50.0, help="Stop trading if daily loss exceeds this amount.")

st.sidebar.header("🛡️ Risk Controls")
use_trailing = st.sidebar.checkbox("Trailing Stop", value=True)
trail_trigger_pct = st.sidebar.slider("Trail Trigger %", 0.1, 0.9, 0.5)
max_consecutive_losses = st.sidebar.number_input("Max Losses", value=3)
use_cooldown = st.sidebar.checkbox("Cooldown", value=True)
cooldown_bars = st.sidebar.number_input("Cooldown Bars", value=50)

st.sidebar.header("Signal Filters")
use_trend_filter = st.sidebar.checkbox("Trend Filter (MA Velocity)", value=True)
use_vol_filter = st.sidebar.checkbox("Volatility Filter", value=True)

vol_filter_mode = st.sidebar.selectbox(
    "Volatility Regime", 
    ["Original (Declining)", "Squeeze (Below MA)", "Expansion (Above MA)"]
)

atr_period = st.sidebar.number_input("ATR Period", value=14)
vol_threshold_ma = st.sidebar.slider("Vol Threshold MA", 20, 200, 118)
trend_ma_period = st.sidebar.number_input("Trend MA Period", 20, 200, 20)

# --- PATTERN STRENGTH CONFIRMATION ---
st.sidebar.subheader("🕯️ Pattern Strength Filters")
min_wick_ratio = st.sidebar.slider("Min Wick Ratio", 1.5, 5.0, 2.33, help="For Hammer/Star: Wick must be X times larger than body.")
min_engulf_mult = st.sidebar.slider("Min Engulf Body Size", 0.5, 3.0, 1.8, help="Engulfing candle body must be X times the average body size.")

st.sidebar.text("Active Patterns:")
show_engulfing = st.sidebar.checkbox("Engulfing", value=True)
show_hammer = st.sidebar.checkbox("Hammer (Bull)", value=True)
show_star = st.sidebar.checkbox("Shooting Star (Bear)", value=True)
show_hanging = st.sidebar.checkbox("Hanging Man (Bear)", value=True)

live_mode = st.sidebar.checkbox("🔴 Live Refresh", value=False)

# --- HELPER FUNCTIONS ---

def get_mt5_data(login, password, server, symbol, timeframe, num_candles):
    if not mt5.initialize(): return None, None, None, None, f"MT5 Init Failed"
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
            mt5.shutdown(); return None, None, None, None, f"Login Failed"
    if not mt5.symbol_select(symbol, True): mt5.shutdown(); return None, None, None, None, f"Symbol not found"
    
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None: mt5.shutdown(); return None, None, None, None, f"Info Failed"
    
    contract_size = sym_info.trade_contract_size
    point = sym_info.point
    digits = sym_info.digits
    
    attempts = 0
    rates = None
    with st.spinner(f"Requesting data..."):
        while attempts < 3:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
            if rates is not None and len(rates) > 0: break
            attempts += 1; time.sleep(1)
            
    mt5.shutdown()
    if rates is None or len(rates) == 0: return None, None, None, None, "No Data"

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, contract_size, point, digits, "Success"

def identify_signals(df, atr_period, vol_ma, trend_ma, vol_mode, wick_ratio, engulf_mult):
    # Vectorized OHLC
    op = df['open']
    hi = df['high']
    lo = df['low']
    cl = df['close']
    
    # 1. Volatility & Trend Indicators
    df['ATR'] = talib.ATR(hi, lo, cl, timeperiod=atr_period)
    df['ATR_MA'] = df['ATR'].rolling(window=vol_ma).mean()
    
    if vol_mode == "Original (Declining)":
        df['Vol_Condition'] = (df['ATR'] < df['ATR'].shift(1)) & (df['ATR'].shift(1) > df['ATR_MA'].shift(1))
    elif vol_mode == "Squeeze (Below MA)":
        df['Vol_Condition'] = df['ATR'] < df['ATR_MA']
    else: 
        df['Vol_Condition'] = df['ATR'] > df['ATR_MA']
    
    df['Trend_MA'] = talib.SMA(cl, timeperiod=trend_ma)
    df['MA_Velocity'] = df['Trend_MA'].diff()

    # --- 2. CALCULATE GEOMETRY FOR FILTERS ---
    body_size = (cl - op).abs()
    upper_wick = hi - np.maximum(op, cl)
    lower_wick = np.minimum(op, cl) - lo
    
    # Calculate Average Body Size (Smoothed over 14 bars)
    avg_body_size = body_size.rolling(window=14).mean()
    
    # --- 3. TA-LIB DETECTION ---
    ta_engulf = talib.CDLENGULFING(op, hi, lo, cl)
    ta_hammer = talib.CDLHAMMER(op, hi, lo, cl)
    ta_star = talib.CDLSHOOTINGSTAR(op, hi, lo, cl)
    ta_hanging = talib.CDLHANGINGMAN(op, hi, lo, cl)
    
    # --- 4. HYBRID FILTERING ---
    is_significant_body = body_size > (avg_body_size * engulf_mult)
    
    df['Hybrid_Engulf_Bull'] = (ta_engulf == 100) & is_significant_body
    df['Hybrid_Engulf_Bear'] = (ta_engulf == -100) & is_significant_body
    
    # Reversal Filter: Must be TA-Lib Pattern AND Wick > (Multiplier * Body)
    df['Hybrid_Hammer'] = (ta_hammer == 100) & (lower_wick > (body_size * wick_ratio))
    df['Hybrid_Star'] = (ta_star == -100) & (upper_wick > (body_size * wick_ratio))
    df['Hybrid_Hanging'] = (ta_hanging == -100) & (lower_wick > (body_size * wick_ratio))

    return df

def plot_volatility_regime(df):
    """Creates a standalone panel to monitor Volatility Expansion/Contraction"""
    # Slice last 500 bars for better visualization performance
    disp_df = df.iloc[-500:].copy()
    
    last_atr = disp_df['ATR'].iloc[-1]
    last_ma = disp_df['ATR_MA'].iloc[-1]
    
    # Determine State
    state = "EXPANSION (High Vol)" if last_atr > last_ma else "SQUEEZE (Low Vol)"
    state_color = "green" if last_atr > last_ma else "red"
    
    st.markdown(f"### 📉 Volatility Regime Monitor")
    st.caption(f"Comparing current **ATR** vs **Moving Average ({int(st.session_state.get('vol_threshold_ma', 118))})**")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current ATR", f"{last_atr:.4f}")
    c2.metric("Threshold (MA)", f"{last_ma:.4f}")
    c3.markdown(f"**State:** :{state_color}[{state}]")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.6, 0.4],
                        subplot_titles=("Price Action", "ATR vs Threshold"))
    
    # Price
    fig.add_trace(go.Candlestick(x=disp_df.index, open=disp_df['open'], high=disp_df['high'],
                                 low=disp_df['low'], close=disp_df['close'], name="Price"), row=1, col=1)
    
    # Volatility
    # ATR Line (Green)
    fig.add_trace(go.Scatter(x=disp_df.index, y=disp_df['ATR'], 
                             line=dict(color='#00E396', width=2), name="ATR"), row=2, col=1)
    
    # Threshold Line (Dashed)
    fig.add_trace(go.Scatter(x=disp_df.index, y=disp_df['ATR_MA'], 
                             line=dict(color='gray', width=1, dash='dash'), name="Threshold"), row=2, col=1)
    
    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

def run_backtest_dynamic(df, contract_size, point, digits, spread_pips, start_bal, sizing_type, risk_val, sl_mult, tp_mult, max_bars, patterns_active, max_losses, do_cooldown, cooldown_len, do_trail, trail_pct, use_trend, use_vol, use_daily_target, daily_target, use_daily_loss, daily_loss):
    
    if digits == 3 or digits == 5: pip_size = point * 10
    else: pip_size = point
    spread_price_delta = spread_pips * pip_size

    buy_mask = pd.Series(False, index=df.index)
    sell_mask = pd.Series(False, index=df.index)
    
    # Map Hybrid Columns
    if patterns_active['Engulfing']:
        buy_mask |= df['Hybrid_Engulf_Bull']
        sell_mask |= df['Hybrid_Engulf_Bear']
        
    if patterns_active['Hammer']: 
        buy_mask |= df['Hybrid_Hammer']
        
    if patterns_active['HangingMan']: 
        sell_mask |= df['Hybrid_Hanging']
        
    if patterns_active['ShootingStar']: 
        sell_mask |= df['Hybrid_Star']
        
    if use_trend:
        buy_mask &= (df['MA_Velocity'] > 0)
        sell_mask &= (df['MA_Velocity'] < 0)
        
    if use_vol:
        buy_mask &= df['Vol_Condition']
        sell_mask &= df['Vol_Condition']

    potential_signals = pd.DataFrame(index=df.index)
    potential_signals['Signal'] = 0
    potential_signals.loc[buy_mask, 'Signal'] = 1
    potential_signals.loc[sell_mask, 'Signal'] = -1
    potential_signals.loc[buy_mask & sell_mask, 'Signal'] = 0 
    
    trades_log = []
    consecutive_losses = 0
    cooldown_until = -1
    current_equity = start_bal
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
        if current_equity <= 0: break
        
        current_date = times[i].date()
        current_day_pnl = daily_pnl_map.get(current_date, 0.0)
        
        # --- CHECK DAILY LIMITS ---
        if use_daily_target and current_day_pnl >= daily_target: continue 
        if use_daily_loss and current_day_pnl <= -daily_loss: continue
        # --------------------------
        
        sig = signals[i]
        if sig == 0: continue
        if np.isnan(atrs[i]): continue
            
        entry_idx = i + 1
        entry_price = opens[entry_idx]
        current_atr = atrs[i]
        
        sl_dist = sl_mult * current_atr
        tp_dist = tp_mult * current_atr
        
        trade_lots = 0.0
        if sizing_type == "Risk % of Equity":
            risk_amt = current_equity * (risk_val / 100.0)
            if sl_dist > 0: calc_lots = risk_amt / (contract_size * sl_dist)
            else: calc_lots = 0.01
            trade_lots = max(0.01, round(calc_lots, 2))
        else:
            trade_lots = risk_val
            
        current_sl = entry_price - sl_dist if sig == 1 else entry_price + sl_dist
        tp_price = entry_price + tp_dist if sig == 1 else entry_price - tp_dist
        trail_trigger = entry_price + (tp_dist * trail_pct) if sig == 1 else entry_price - (tp_dist * trail_pct)
        
        exit_price, exit_idx, exit_reason = 0.0, -1, "Time"
        
        for j in range(entry_idx, entry_idx + max_bars):
            if j >= n_rows: break
            c_high, c_low = highs[j], lows[j]
            
            # SL
            if (sig == 1 and c_low <= current_sl) or (sig == -1 and c_high >= current_sl):
                exit_price = current_sl; exit_idx = j; exit_reason = "SL"
                if do_trail and abs(current_sl - entry_price) < 0.0001: exit_reason = "BE (Trailing)"
                elif do_trail and ((sig==1 and current_sl > entry_price) or (sig==-1 and current_sl < entry_price)): exit_reason = "Profit (Trailing)"
                break
                
            # TP
            if (sig == 1 and c_high >= tp_price) or (sig == -1 and c_low <= tp_price):
                exit_price = tp_price; exit_idx = j; exit_reason = "TP"; break
                
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
        
        raw_pnl = (exit_price - entry_price) * sig * trade_lots * contract_size
        spread_cost = spread_price_delta * contract_size * trade_lots
        net_pnl = raw_pnl - spread_cost
        
        current_equity += net_pnl
        
        exit_date = times[exit_idx].date()
        daily_pnl_map[exit_date] = daily_pnl_map.get(exit_date, 0.0) + net_pnl
        
        trades_log.append({
            'Entry_Time': times[entry_idx], 'Signal': "BUY" if sig==1 else "SELL",
            'Entry_Price': entry_price, 'Exit_Price': exit_price,
            'Lots': trade_lots, 'PnL': net_pnl, 'Spread_Cost': spread_cost,
            'Equity': current_equity, 'Reason': exit_reason
        })
        
        if net_pnl <= 0:
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
        "Total Spread Paid": df_res['Spread_Cost'].sum(),
        "Win Rate": (df_res['PnL'] > 0).mean() * 100,
        "Trades": len(df_res),
        "Final Equity": df_res['Equity'].iloc[-1],
        "Max Drawdown": (df_res['Equity'].cummax() - df_res['Equity']).max()
    }
    return df_res, metrics

# --- MAIN ---

if fetch_needed := st.sidebar.button("🚀 Run Risk Backtest") or live_mode:
    with st.spinner("Applying Hybrid Filters..."):
        df, contract, point, digits, msg = get_mt5_data(mt5_login, mt5_pass, mt5_server, mt5_symbol, selected_tf_mt5, candles_to_fetch)
        
        if df is not None:
            # 1. Calculate Indicators & Filters
            df = identify_signals(df, atr_period, vol_threshold_ma, trend_ma_period, vol_filter_mode, min_wick_ratio, min_engulf_mult)
            
            # 2. NEW: Display Volatility Monitor 

            plot_volatility_regime(df)
            
            # 3. Prepare Backtest Params
            pats = {
                'Engulfing': show_engulfing, 'Hammer': show_hammer,
                'ShootingStar': show_star, 'HangingMan': show_hanging
            }
            
            risk_val = risk_pct if sizing_mode == "Risk % of Equity" else fixed_lot
            
            # 4. Run Strategy
            trades, metrics = run_backtest_dynamic(
                df, contract, point, digits, spread_pips, initial_capital, sizing_mode, risk_val, 
                sl_multiplier, tp_multiplier, max_hold_bars, pats, 
                max_consecutive_losses, use_cooldown, cooldown_bars, 
                use_trailing, trail_trigger_pct, use_trend_filter, use_vol_filter,
                use_daily_target, daily_target_amt,
                use_daily_loss, daily_loss_amt
            )
            
            if trades is not None:
                st.subheader(f"📊 Results ({sizing_mode}: {risk_val})")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Net PnL", f"${metrics['Net PnL']:,.2f}")
                c2.metric("Final Equity", f"${metrics['Final Equity']:,.2f}")
                c3.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
                c4.metric("Max Drawdown", f"${metrics['Max Drawdown']:,.2f}")
                c5.metric("Spread Paid", f"${metrics['Total Spread Paid']:,.2f}")
                
                fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4])
                fig.add_trace(go.Scatter(x=trades.index, y=trades['Equity'], name='Equity', fill='tozeroy', line=dict(color='#00CC96')), row=1, col=1)
                fig.add_trace(go.Bar(x=trades.index, y=trades['PnL'], name='PnL', marker_color=np.where(trades['PnL']>0, 'green', 'red')), row=2, col=1)
                fig.update_layout(height=600, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(trades)
            else:
                st.warning("No trades found.")
        else:
            st.error(msg)