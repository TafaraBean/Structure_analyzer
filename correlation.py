import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="StatArb Alpha Node (Hybrid)", layout="wide", page_icon="🛡️")
load_dotenv()

# Sidebar: Connection
st.sidebar.header("🔌 Connection")
LOGIN = st.sidebar.number_input("MT5 Login", value=int(os.getenv("MT5_LOGIN", 0)))
PASSWORD = st.sidebar.text_input("MT5 Password", value=os.getenv("MT5_PASSWORD", ""), type="password")
SERVER = st.sidebar.text_input("MT5 Server", value=os.getenv("MT5_SERVER", "HFMarketsSA-Live2"))

st.sidebar.markdown("---")
# Sidebar: Assets
st.sidebar.header("🔬 Statistical Params")
ASSET_Y = st.sidebar.text_input("Dependent Asset (Y)", value="BTCUSDz")
ASSET_X = st.sidebar.text_input("Predictor Asset (X)", value="ETHUSDz")

tf_map = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1
}
tf_label = st.sidebar.selectbox("Timeframe", options=list(tf_map.keys()), index=1)
TIMEFRAME = tf_map[tf_label]
LOOKBACK = st.sidebar.slider("Sample Size (Bars)", 1000, 10000, 2000)

st.sidebar.markdown("---")
# Sidebar: Model
st.sidebar.header("🧠 Predictive Model")
REGRESSION_WINDOW = st.sidebar.number_input("Training Window (Rolling)", value=120)
LAG_PERIODS = st.sidebar.number_input("Predictive Lag", value=1, min_value=1)

st.sidebar.markdown("---")
# Sidebar: Filters
st.sidebar.header("🛡️ Safety & Filters")

# 1. Profit Threshold
ENTRY_THRESH = st.sidebar.number_input("Min Predicted Return (%)", value=0.02, step=0.01, format="%.2f", 
                                       help="Only trade if expected profit > fees (e.g. 0.02%).")

# 2. Hysteresis
EXIT_THRESH = st.sidebar.number_input("Sticky Exit Threshold (%)", value=0.005, step=0.001, format="%.3f",
                                      help="Stay in trade until signal reverses by this amount.")

# 3. Z-Score Guard (NEW)
Z_SCORE_GUARD = st.sidebar.number_input("Max Z-Score (Guard)", value=2.0, step=0.1, 
                                        help="Don't BUY if Spread Z > 2.0 (Too expensive). Don't SELL if Z < -2.0 (Too cheap).")

# 4. Volatility Gate (NEW)
MIN_VOL_GATE = st.sidebar.number_input("Min Volatility (%)", value=0.01, step=0.01, format="%.2f",
                                       help="Don't trade if recent volatility is lower than this (Dead Market).")

MIN_R2_THRESH = st.sidebar.number_input("Min R-Squared", value=0.01, step=0.01, format="%.2f")

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
@st.cache_resource
def init_mt5(login, password, server):
    if not mt5.initialize():
        return False, f"Init failed: {mt5.last_error()}"
    if not mt5.login(login, password=password, server=server):
        return False, f"Login failed: {mt5.last_error()}"
    return True, "Connected"

def fetch_data(symbol, timeframe, n):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df['close']

def calculate_hybrid_model(df, window, lag):
    """
    Calculates both Velocity (Short Term) and Cointegration Z-Score (Long Term).
    """
    # 1. Log Returns & Prices
    df['log_Y'] = np.log(df['Y'])
    df['log_X'] = np.log(df['X'])
    
    df['ret_Y'] = df['log_Y'].diff()
    df['ret_X'] = df['log_X'].diff()
    
    # 2. Velocity Prediction (Rolling OLS)
    df['feat_X_lag'] = df['ret_X'].shift(lag)
    data_reg = df.dropna().copy()

    exog = sm.add_constant(data_reg['feat_X_lag'])
    endog = data_reg['ret_Y']
    rols = RollingOLS(endog, exog, window=window)
    rres = rols.fit()
    
    data_reg['alpha'] = rres.params['const']
    data_reg['beta_lag'] = rres.params['feat_X_lag']
    data_reg['r_squared'] = rres.rsquared
    data_reg['pred_ret_Y'] = data_reg['alpha'] + (data_reg['beta_lag'] * data_reg['feat_X_lag'])
    data_reg['residual'] = data_reg['ret_Y'] - data_reg['pred_ret_Y']

    # 3. Cointegration Z-Score (The Safety Guard)
    # We calculate a simple spread: Y - beta*X
    # Note: For strict cointegration we usually use Log prices. 
    # We use the current dynamic beta to normalize the spread.
    data_reg['spread'] = data_reg['log_Y'] - (data_reg['beta_lag'] * data_reg['log_X'])
    
    # Normalize Spread (Z-Score)
    # We use a longer window for the Z-Score mean to establish "Fair Value"
    z_window = window * 2
    spread_mean = data_reg['spread'].rolling(z_window).mean()
    spread_std = data_reg['spread'].rolling(z_window).std()
    data_reg['spread_z'] = (data_reg['spread'] - spread_mean) / spread_std
    
    # 4. Volatility (Realized Vol of Y)
    data_reg['volatility_Y'] = data_reg['ret_Y'].rolling(window=20).std() * 100 # In %

    # 5. Benchmarks
    data_reg['cum_bh'] = data_reg['ret_Y'].cumsum()

    return data_reg

def generate_trade_log(df, entry_thresh_pct, exit_thresh_pct, min_r2, max_z_guard, min_vol):
    """
    Generates trades with Hybrid Logic:
    - Velocity says GO
    - Z-Score says SAFE
    - Volatility says LIQUID
    """
    trades = []
    active_trade = None
    
    entry_dec = entry_thresh_pct / 100.0
    exit_dec = exit_thresh_pct / 100.0
    
    for i in range(1, len(df)):
        # Warmup Check
        if pd.isna(df['r_squared'].iloc[i]) or df['r_squared'].iloc[i] == 0 or pd.isna(df['spread_z'].iloc[i]):
            continue

        ts = df.index[i]
        price = df['Y'].iloc[i]
        
        # Signals
        r2_curr = df['r_squared'].iloc[i]
        pred_ret = df['pred_ret_Y'].iloc[i]
        z_curr = df['spread_z'].iloc[i]
        vol_curr = df['volatility_Y'].iloc[i]

        # --- LOGIC IF NO POSITION ---
        if active_trade is None:
            
            # GLOBAL FILTERS (Must pass these to even consider trading)
            is_liquid = vol_curr >= min_vol
            is_confident = r2_curr >= min_r2
            
            if is_liquid and is_confident:
                
                # LONG CHECK
                # 1. Velocity predicts UP (> Threshold)
                # 2. Price is NOT too expensive (Z < Guard)
                if (pred_ret > entry_dec) and (z_curr < max_z_guard):
                    active_trade = {
                        'entry_time': ts, 'side': 'LONG', 'entry_price': price, 
                        'model_r2': r2_curr, 'pred_return': pred_ret, 'z_score': z_curr
                    }
                
                # SHORT CHECK
                # 1. Velocity predicts DOWN (< -Threshold)
                # 2. Price is NOT too cheap (Z > -Guard)
                elif (pred_ret < -entry_dec) and (z_curr > -max_z_guard):
                    active_trade = {
                        'entry_time': ts, 'side': 'SHORT', 'entry_price': price, 
                        'model_r2': r2_curr, 'pred_return': pred_ret, 'z_score': z_curr
                    }

        # --- LOGIC IF IN POSITION (STICKY EXIT) ---
        else:
            should_close = False
            
            if active_trade['side'] == 'LONG':
                # Close if velocity turns bearish beyond sticky threshold
                if pred_ret < -exit_dec:
                    should_close = True
            
            elif active_trade['side'] == 'SHORT':
                # Close if velocity turns bullish beyond sticky threshold
                if pred_ret > exit_dec:
                    should_close = True
            
            if should_close:
                # Close Trade
                active_trade['exit_time'] = ts
                active_trade['exit_price'] = price
                pnl = (price - active_trade['entry_price']) / active_trade['entry_price'] if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - price) / active_trade['entry_price']
                active_trade['return_pct'] = pnl * 100
                trades.append(active_trade)
                
                # Try Immediate Flip (Reverse)
                # Must pass all filters again
                active_trade = None
                
                is_liquid = vol_curr >= min_vol
                is_confident = r2_curr >= min_r2
                
                if is_liquid and is_confident:
                    if (pred_ret > entry_dec) and (z_curr < max_z_guard):
                        active_trade = {'entry_time': ts, 'side': 'LONG', 'entry_price': price, 'model_r2': r2_curr, 'pred_return': pred_ret, 'z_score': z_curr}
                    elif (pred_ret < -entry_dec) and (z_curr > -max_z_guard):
                        active_trade = {'entry_time': ts, 'side': 'SHORT', 'entry_price': price, 'model_r2': r2_curr, 'pred_return': pred_ret, 'z_score': z_curr}

    # Force close final
    if active_trade is not None:
        active_trade['exit_time'] = df.index[-1]
        active_trade['exit_price'] = df['Y'].iloc[-1]
        pnl = (active_trade['exit_price'] - active_trade['entry_price']) / active_trade['entry_price'] if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - active_trade['exit_price']) / active_trade['entry_price']
        active_trade['return_pct'] = pnl * 100
        trades.append(active_trade)
        
    return pd.DataFrame(trades)

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================
st.title("🛡️ StatArb: Hybrid Safety Model")
st.markdown("Combines **Velocity Prediction** (Offense) with **Cointegration Z-Score** (Defense) to filter bad trades.")

status, msg = init_mt5(LOGIN, PASSWORD, SERVER)
if not status:
    st.error(msg)
    st.stop()
else:
    st.sidebar.success("🟢 MT5 Connected")

if st.sidebar.button("🔄 Refresh Analysis", type="primary"):
    st.rerun()

with st.spinner("Calculating Hybrid Metrics..."):
    py = fetch_data(ASSET_Y, TIMEFRAME, LOOKBACK)
    px = fetch_data(ASSET_X, TIMEFRAME, LOOKBACK)

if py is None or px is None:
    st.error("Could not fetch data.")
    st.stop()

df = pd.concat([py, px], axis=1).dropna()
df.columns = ['Y', 'X']

# Run Model
model_data = calculate_hybrid_model(df, REGRESSION_WINDOW, LAG_PERIODS)
last = model_data.iloc[-1]

# --- METRICS ROW ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Return", f"{last['pred_ret_Y']*100:.4f}%", 
          delta="Bullish" if last['pred_ret_Y'] > 0 else "Bearish")
c2.metric("Spread Z-Score", f"{last['spread_z']:.2f}", 
          delta="Expensive" if last['spread_z'] > 0 else "Cheap", delta_color="inverse")
c3.metric("Current Volatility", f"{last['volatility_Y']:.3f}%", 
          delta="Safe" if last['volatility_Y'] > MIN_VOL_GATE else "Dead Market")
c4.metric("Model Confidence (R²)", f"{last['r_squared']:.3f}")

# --- GENERATE TRADES ---
trade_df = generate_trade_log(model_data, ENTRY_THRESH, EXIT_THRESH, MIN_R2_THRESH, Z_SCORE_GUARD, MIN_VOL_GATE)

# --- VISUALIZATION ---
fig = make_subplots(
    rows=4, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03,
    row_heights=[0.3, 0.2, 0.2, 0.3], 
    subplot_titles=(
        "1. Performance: Hybrid Strategy vs Buy & Hold", 
        "2. Velocity Signal (Short Term Alpha)", 
        "3. Z-Score Guard (Long Term Safety)", 
        "4. Regime: Volatility & R-Squared"
    )
)

# Row 1: Equity
model_data['cum_strategy'] = 0.0
if not trade_df.empty:
    equity_curve = [0]
    trade_times = [model_data.index[0]]
    running_pnl = 0
    for idx, row in trade_df.iterrows():
        running_pnl += row['return_pct']
        equity_curve.append(running_pnl)
        trade_times.append(row['exit_time'])
    trade_equity_df = pd.DataFrame({'time': trade_times, 'equity': equity_curve}).set_index('time')
    fig.add_trace(go.Scatter(x=trade_equity_df.index, y=trade_equity_df['equity'], name='Hybrid Strategy (%)', line=dict(color='green', width=2, shape='hv')), row=1, col=1)

fig.add_trace(go.Scatter(x=model_data.index, y=model_data['cum_bh']*100, name='Buy & Hold (%)', line=dict(color='gray', width=1)), row=1, col=1)

# Row 2: Velocity
colors = np.where(model_data['pred_ret_Y'] > 0, 'green', 'red')
fig.add_trace(go.Bar(x=model_data.index, y=model_data['pred_ret_Y'], name='Pred Return', marker_color=colors), row=2, col=1)
entry_dec = ENTRY_THRESH / 100.0
fig.add_hline(y=entry_dec, line_dash="dot", line_color="black", row=2, col=1)
fig.add_hline(y=-entry_dec, line_dash="dot", line_color="black", row=2, col=1)

# Row 3: Z-Score
fig.add_trace(go.Scatter(x=model_data.index, y=model_data['spread_z'], name='Spread Z-Score', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=Z_SCORE_GUARD, line_color="red", row=3, col=1, annotation_text="Overbought Guard")
fig.add_hline(y=-Z_SCORE_GUARD, line_color="green", row=3, col=1, annotation_text="Oversold Guard")
fig.add_hrect(y0=Z_SCORE_GUARD, y1=5, fillcolor="red", opacity=0.1, layer="below", row=3, col=1)
fig.add_hrect(y0=-Z_SCORE_GUARD, y1=-5, fillcolor="green", opacity=0.1, layer="below", row=3, col=1)

# Row 4: Volatility
fig.add_trace(go.Scatter(x=model_data.index, y=model_data['volatility_Y'], name='Volatility', line=dict(color='orange')), row=4, col=1)
fig.add_hline(y=MIN_VOL_GATE, line_color="red", line_dash="dot", row=4, col=1, annotation_text="Dead Market Level")

fig.update_layout(height=1100, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- TRADE LOG ---
st.markdown("---")
st.subheader("📝 Trade Ledger (Safety Audited)")

if not trade_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    total_trades = len(trade_df)
    win_rate = len(trade_df[trade_df['return_pct'] > 0]) / total_trades * 100
    avg_pnl = trade_df['return_pct'].mean()
    cum_pnl = trade_df['return_pct'].sum()
    
    col1.metric("Trades Taken", total_trades)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Avg Trade", f"{avg_pnl:.4f}%")
    col4.metric("Net Return", f"{cum_pnl:.2f}%")

    display_cols = ['entry_time', 'side', 'entry_price', 'exit_time', 'return_pct', 'model_r2', 'z_score']
    
    def color_pnl(val):
        return f'color: {"green" if val > 0 else "red"}'

    st.dataframe(
        trade_df[display_cols].style.applymap(color_pnl, subset=['return_pct'])
        .format({'entry_price': "{:.4f}", 'return_pct': "{:.4f}%", 'model_r2': "{:.3f}", 'z_score': "{:.2f}"}),
        use_container_width=True,
        height=400
    )
    
    # CSV Download
    csv = trade_df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Log", csv, "hybrid_trades.csv", "text/csv")
else:
    st.warning("No trades found. The Z-Score Guard or Volatility Gate might be too strict.")