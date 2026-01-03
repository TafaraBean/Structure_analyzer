"""
HOW TO RUN THIS APP:
--------------------
1. Open your terminal or command prompt.
2. Navigate to the folder containing this file.
3. Run the following command:
   streamlit run nvidia_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import MetaTrader5 as mt5
import plotly.io as pio
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Attempt to import FPDF for PDF generation
try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="MT5 Structural Backtester", layout="wide", initial_sidebar_state="expanded")

st.title("🧱 MT5 Structural Backtester: Golden Zone Mode")
st.markdown("""
**Strategy:** Trend-Filtered Reversion on **Volume-Weighted** Levels.
**Logic:**
* **Volume Weighted:** Levels are pulled towards areas of high institutional activity (Log Volume * Volatility).
* **Snapbacks:** Uses Velocity Reversals (Physics) to find pivot points.
* **Golden Zone:** Validates levels against **Higher Timeframe** Structure & **Fibonacci**.
* **Adaptive Updates:** Levels update only when price drifts > X% from structure (No Daily Reset).
""")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🔌 Connection & Settings")

with st.sidebar.expander("MetaTrader 5 Credentials", expanded=True):
    # Fetch from environment variables, default to empty/zero if not set
    env_login = os.getenv("MT5_LOGIN")
    default_login = int(env_login) if env_login else 0
    default_pass = os.getenv("MT5_PASSWORD", "")
    default_server = os.getenv("MT5_SERVER", "HFMarketsSA-Live2")

    mt5_login = st.number_input("MT5 Login ID", value=default_login, step=1)
    mt5_pass = st.text_input("MT5 Password", value=default_pass, type="password") 
    mt5_server = st.text_input("MT5 Server", value=default_server)
    mt5_symbol = st.text_input("Asset Symbol", value="USA100", help="Exact symbol name from your Market Watch")

# Timeframe Selection
timeframe_map = {
    "1 Hour (H1)": mt5.TIMEFRAME_H1,
    "15 Minutes (M15)": mt5.TIMEFRAME_M15,
    "5 Minutes (M5)": mt5.TIMEFRAME_M5
}
selected_tf_label = st.sidebar.selectbox("Timeframe", list(timeframe_map.keys()), index=0)
selected_tf_mt5 = timeframe_map[selected_tf_label]

# HTF Mapping
htf_map = {
    mt5.TIMEFRAME_M5: mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_M15: mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H1: mt5.TIMEFRAME_D1
}

# GLOBAL CONFIG FOR COSTS
COSTS = {
    "spread": 1.2,  # Points per trade (Round turn cost approx)
}

# Data Depth
days_to_fetch = st.sidebar.slider("Days of History", min_value=30, max_value=730, value=90)

# Strategy Selector
st.sidebar.markdown("---")
st.sidebar.header("⚔️ Strategy Logic")

n_clusters = st.sidebar.slider("Number of Levels", 3, 10, 3)
ma_period = st.sidebar.number_input("Trend MA Period", min_value=10, max_value=200, value=40)
min_history_days = st.sidebar.number_input("Lookback Days (Structure)", value=7, min_value=1)
recalc_threshold = st.sidebar.slider("Update Threshold (% Distance)", 0.1, 5.0, 1.0, 0.1, help="Recalculate levels when price is this % away from nearest level")
hold_period = st.sidebar.number_input("Hold Period (Bars)", value=11, min_value=1)

c1, c2 = st.columns(2)
use_trend_filter = c1.checkbox("Use Trend Filter", value=True)
use_confluence = c2.checkbox("Use Golden Zone (HTF+Fib)", value=True, help="Filters levels against Higher Timeframe & Fibs")

strategies = st.sidebar.multiselect(
    "Active Orders",
    ["Support Bounce (Buy Limit)", "Resistance Reject (Sell Limit)"],
    default=["Support Bounce (Buy Limit)", "Resistance Reject (Sell Limit)"]
)

# --- HELPER FUNCTIONS ---

def df_to_markdown(df):
    """Converts a DataFrame to a Markdown table string (Removes tabulate dependency)."""
    if df.empty: return ""
    df_str = df.astype(str)
    headers = df_str.columns.tolist()
    md = "| " + " | ".join(headers) + " |\n"
    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for _, row in df_str.iterrows():
        md += "| " + " | ".join(row.tolist()) + " |\n"
    return md

def get_candles_count(timeframe, days):
    if timeframe == mt5.TIMEFRAME_H1: return days * 24
    elif timeframe == mt5.TIMEFRAME_M15: return days * 24 * 4
    elif timeframe == mt5.TIMEFRAME_M5: return days * 24 * 12
    return days * 24

def get_candles_per_day(timeframe):
    if timeframe == mt5.TIMEFRAME_H1: return 24
    elif timeframe == mt5.TIMEFRAME_M15: return 96
    elif timeframe == mt5.TIMEFRAME_M5: return 288
    return 24

def get_mt5_data(login, password, server, symbol, timeframe, days):
    if not mt5.initialize(): return None, f"MT5 Init Failed: {mt5.last_error()}"
    if not mt5.terminal_info(): 
        if not mt5.login(login=int(login), password=password, server=server):
            return None, f"MT5 Login Failed: {mt5.last_error()}"
    
    # Base Data
    num_candles = get_candles_count(timeframe, days)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    
    if rates is None or len(rates) == 0: 
        mt5.shutdown()
        return None, None, "No data received"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # HTF Data (Always fetch Daily/H1 depending on base)
    htf_tf = htf_map.get(timeframe, mt5.TIMEFRAME_D1)
    # Fetch enough HTF candles to cover the same period
    htf_candles = days * 24 if htf_tf == mt5.TIMEFRAME_H1 else days + 50
    
    rates_htf = mt5.copy_rates_from_pos(symbol, htf_tf, 0, htf_candles)
    if rates_htf is not None:
        df_htf = pd.DataFrame(rates_htf)
        df_htf['time'] = pd.to_datetime(df_htf['time'], unit='s')
        df_htf.set_index('time', inplace=True)
    else:
        df_htf = None
        
    mt5.shutdown()
    return df, df_htf, "Success"

def calculate_levels_weighted_kmeans(df_slice, n_clusters):
    """
    Finds S/R levels using Physics (Velocity Reversals) + Volume/Volatility Weighting + K-Means.
    """
    if len(df_slice) < 20: return []
    
    velocity = df_slice['close'].diff()
    reversal_mask = (velocity * velocity.shift(1)) < 0
    
    rev_highs = df_slice.loc[reversal_mask, 'high'].values.reshape(-1, 1)
    rev_lows = df_slice.loc[reversal_mask, 'low'].values.reshape(-1, 1)
    
    # Weighting Logic
    vol_col = 'real_volume' if 'real_volume' in df_slice.columns and df_slice['real_volume'].sum() > 0 else 'tick_volume'
    vol_series = df_slice[vol_col]
    range_series = df_slice['high'] - df_slice['low']
    
    def normalize(arr):
        if len(arr) == 0 or np.max(arr) == np.min(arr): return np.ones_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) + 0.1

    w_vol = normalize(vol_series.loc[reversal_mask].values)
    w_rng = normalize(range_series.loc[reversal_mask].values)
    combined_weights = w_vol * w_rng
    
    prices = np.concatenate([rev_highs, rev_lows]).reshape(-1, 1)
    weights = np.concatenate([combined_weights, combined_weights])
    
    if len(prices) < n_clusters: return []
        
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(prices, sample_weight=weights)
        return sorted(kmeans.cluster_centers_.flatten())
    except:
        return []

def get_fibs(df_slice):
    high = df_slice['high'].max()
    low = df_slice['low'].min()
    diff = high - low
    return [low, low + 0.236*diff, low + 0.382*diff, low + 0.5*diff, low + 0.618*diff, low + 0.786*diff, high]

def run_strategy_backtest(df, df_htf, min_history_days, n_clusters, active_strategies, use_trend_filter, use_confluence, ma_period, hold_period, selected_timeframe_const, recalc_threshold):
    """
    Simulates trading with Adaptive Level Updates (Drift-Based), Confluence, and Hold Period logic.
    """
    trades = []
    equity = [0]
    history_segments = [] 
    
    # DEFINE FLAGS EARLY (FIX: SCOPE ISSUE)
    use_sup_bounce = "Support Bounce (Buy Limit)" in active_strategies
    use_res_reject = "Resistance Reject (Sell Limit)" in active_strategies
    
    df_calc = df.copy()
    df_calc['MA'] = df_calc['close'].rolling(window=ma_period).mean()
    df_calc['MA_Slope'] = df_calc['MA'].diff()
    
    candles_per_day = get_candles_per_day(selected_timeframe_const)
    lookback_candles = min_history_days * candles_per_day
    warmup_idx = max(lookback_candles, ma_period + 1, 50)
    
    if len(df_calc) <= warmup_idx:
        return pd.DataFrame(), [], [], df_calc

    # Pre-calc arrays
    open_arr = df_calc['open'].values
    high_arr = df_calc['high'].values
    low_arr = df_calc['low'].values
    close_arr = df_calc['close'].values
    slope_arr = df_calc['MA_Slope'].fillna(0).values
    times = df_calc.index
    dates = df_calc.index.date
    
    len_df = len(df_calc)
    active_trade = None 
    
    # Force first calc
    current_levels = []
    seg_start_time = times[warmup_idx]
    last_recalc_time = times[warmup_idx]

    # --- LEVEL CALCULATION HELPER ---
    def calc_confluent_levels(base_data, current_ts):
        # K-MEANS CALCULATION
        raw_levels = calculate_levels_weighted_kmeans(
            base_data, n_clusters
        )
        if not use_confluence or not raw_levels:
            return raw_levels
            
        final_levels = []
        # HTF Logic
        htf_levels = []
        if df_htf is not None:
            # Slice HTF to prevent lookahead
            slice_htf = df_htf[df_htf.index < current_ts]
            if len(slice_htf) > 20:
                # Use fewer clusters for HTF overview
                htf_levels = calculate_levels_weighted_kmeans(
                    slice_htf.iloc[-60:], max(3, n_clusters // 2)
                )
        # Fibs
        fibs = get_fibs(base_data)
        
        tol = 0.003 # 0.3% Tolerance
        
        for bl in raw_levels:
            is_htf = any(abs(bl - hl)/bl < tol for hl in htf_levels) if htf_levels else False
            is_fib = any(abs(bl - fl)/bl < tol for fl in fibs) if fibs else False
            if is_htf or is_fib: final_levels.append(bl)
            
        return sorted(final_levels)
    
    # Initial Calculation using past data ONLY
    current_levels = calc_confluent_levels(df_calc.iloc[max(0, warmup_idx - lookback_candles):warmup_idx], times[warmup_idx])

    for i in range(warmup_idx, len_df):
        current_open = open_arr[i]
        current_high = high_arr[i]
        current_low = low_arr[i]
        current_close = close_arr[i]
        current_time = times[i]
        
        # --- ADAPTIVE RECALCULATION (Distance Based) ---
        needs_recalc = False
        if not current_levels:
             needs_recalc = True
        else:
            # Vectorized distance to nearest level
            dist_min = np.min(np.abs(current_open - np.array(current_levels)))
            dist_pct = (dist_min / current_open) * 100
            if dist_pct > recalc_threshold:
                needs_recalc = True
        
        if needs_recalc:
            history_segments.append({
                'start': last_recalc_time, 'end': times[i-1], 'levels': current_levels
            })
            # Expanding window: Use all data up to i
            current_levels = calc_confluent_levels(df_calc.iloc[:i], current_time)
            last_recalc_time = current_time
            
        # --- MANAGE TRADE ---
        if active_trade:
            bars_held = i - active_trade['open_idx']
            if bars_held >= hold_period:
                if active_trade['type'] == 'BUY':
                    pnl = (current_close - active_trade['entry']) - COSTS['spread']
                    trades.append({'Time': times[i], 'Type': 'BUY_EXIT', 'Entry': active_trade['entry'], 'Exit': current_close, 'PnL': pnl})
                else:
                    pnl = (active_trade['entry'] - current_close) - COSTS['spread']
                    trades.append({'Time': times[i], 'Type': 'SELL_EXIT', 'Entry': active_trade['entry'], 'Exit': current_close, 'PnL': pnl})
                active_trade = None
            continue

        # --- TREND FILTER ---
        ma_slope = slope_arr[i-1]
        is_uptrend = ma_slope > 0
        is_downtrend = ma_slope < 0
        
        # --- ENTRY ---
        if current_levels:
            nearest_sup = -1.0
            nearest_res = 99999999.0
            
            for lvl in current_levels:
                if lvl < current_open and lvl > nearest_sup: nearest_sup = lvl
                if lvl > current_open and lvl < nearest_res: nearest_res = lvl
            
            if nearest_sup == -1.0: nearest_sup = None
            if nearest_res == 99999999.0: nearest_res = None
            
            # Buy Limit
            if use_sup_bounce and nearest_sup and current_low <= nearest_sup:
                if not use_trend_filter or is_uptrend:
                    active_trade = {'type': 'BUY', 'entry': nearest_sup, 'open_idx': i}
                    continue
            
            # Sell Limit
            if use_res_reject and nearest_res and current_high >= nearest_res:
                if not use_trend_filter or is_downtrend:
                    active_trade = {'type': 'SELL', 'entry': nearest_res, 'open_idx': i}
    
    # Save final segment
    history_segments.append({
        'start': last_recalc_time, 'end': times[-1], 'levels': current_levels
    })
    
    daily_pnls = pd.Series(0.0, index=df_calc.index)
    if trades:
        trade_df_tmp = pd.DataFrame(trades)
        for _, t in trade_df_tmp.iterrows():
            daily_pnls[t['Time']] = t['PnL']
            
    equity_curve = daily_pnls.cumsum().values
    
    return pd.DataFrame(trades), equity_curve, history_segments, df_calc

# --- MAIN LOGIC ---

if 'data' not in st.session_state:
    st.session_state['data'] = None

if st.sidebar.button("🚀 Run Backtest", type="primary"):
    with st.spinner(f"Fetching data and running fixed strategy analysis..."):
        df, df_htf, msg = get_mt5_data(mt5_login, mt5_pass, mt5_server, mt5_symbol, selected_tf_mt5, days_to_fetch)
        
        if df is not None:
            st.session_state['data'] = df
            st.session_state['data_htf'] = df_htf
            st.success(f"Loaded {len(df)} candles.")
        else:
            st.error(msg)

# --- VISUALIZATION ---

if st.session_state['data'] is not None:
    df_raw = st.session_state['data']
    df_htf_raw = st.session_state.get('data_htf', None)
    
    # RUN
    trades_df, equity_curve, segments, df_processed = run_strategy_backtest(
        df_raw, df_htf_raw, min_history_days, n_clusters, strategies, use_trend_filter, use_confluence, ma_period, hold_period, selected_tf_mt5, recalc_threshold
    )
    
    # 1. METRICS
    st.subheader("Performance Metrics")
    
    total_pnl = 0.0
    win_rate = 0.0
    total_trades = 0
    pf = 0.0
    
    if not trades_df.empty:
        total_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['PnL'] > 0]) / total_trades * 100
        total_pnl = trades_df['PnL'].sum()
        
        wins = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
        losses = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
        pf = wins / losses if losses > 0 else float('inf')
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total PnL (Points)", f"{total_pnl:.2f}")
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Profit Factor", f"{pf:.2f}")
        c4.metric("Trades Executed", total_trades)
        
        if use_trend_filter:
            st.caption(f"✅ Filter Active: Trading in direction of {ma_period}-period Moving Average.")
    else:
        st.warning("No trades triggered. Try adjusting the number of levels or history.")

    # 2. CHART
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df_processed.index, open=df_processed['open'], high=df_processed['high'], 
        low=df_processed['low'], close=df_processed['close'], name=mt5_symbol
    ))
    
    fig.add_trace(go.Scatter(
        x=df_processed.index, y=df_processed['MA'], 
        mode='lines', name=f'MA ({ma_period})', 
        line=dict(color='yellow', width=1)
    ))
    
    for seg in segments:
        t_start, t_end = seg['start'], seg['end']
        for lvl in seg['levels']:
            fig.add_shape(type="line", x0=t_start, y0=lvl, x1=t_end, y1=lvl, line=dict(color='gold' if use_confluence else 'rgba(255,255,0,0.4)', width=2 if use_confluence else 1, dash='solid' if use_confluence else 'dot'))

    if not trades_df.empty:
        buys = trades_df[trades_df['Type'] == 'BUY_EXIT']
        sells = trades_df[trades_df['Type'] == 'SELL_EXIT']
        
        fig.add_trace(go.Scatter(x=buys['Time'], y=buys['Exit'], mode='markers', name='Buy Exit', 
                                 marker=dict(symbol='triangle-up', color='green', size=10),
                                 text=buys['PnL'].round(2)))
                                 
        fig.add_trace(go.Scatter(x=sells['Time'], y=sells['Exit'], mode='markers', name='Sell Exit', 
                                 marker=dict(symbol='triangle-down', color='red', size=10),
                                 text=sells['PnL'].round(2)))

    fig.update_layout(height=700, template="plotly_dark", title=f"Strategy Backtest: {mt5_symbol}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. EQUITY CURVE
    st.subheader("📈 Strategy Equity Curve")
    if len(equity_curve) > 0:
        st.area_chart(pd.DataFrame({'PnL': equity_curve}, index=df_processed.index))
    
    # 4. EXPORT
    st.markdown("---")
    with st.expander("📄 Export Results (Report)", expanded=True):
        report_md = f"""
# Backtest Report: {mt5_symbol}
**Strategy:** Fixed Structure Reversion
**Timeframe:** {selected_tf_label}
**History:** {days_to_fetch} Days

## Performance
- **Total PnL:** {total_pnl:.2f}
- **Win Rate:** {win_rate:.1f}%
- **Profit Factor:** {pf:.2f}
"""
        if not trades_df.empty:
            report_md += df_to_markdown(trades_df.tail(50))
        
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Download Report (MD)", report_md, file_name="report.md")
        with c2:
            if FPDF:
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Courier", size=10)
                    pdf.multi_cell(0, 5, report_md.encode('latin-1', 'ignore').decode('latin-1'))
                    st.download_button("📥 Download Report (PDF)", pdf.output(dest='S').encode('latin-1'), file_name="report.pdf")
                except: st.error("PDF generation failed.")
            else:
                st.warning("Install 'fpdf' for PDF export.")

    with st.expander("Detailed Trade Log"):
        st.dataframe(trades_df.sort_values('Time', ascending=False))

else:
    st.info("👈 Enter credentials and click 'Run Backtest' to begin.")