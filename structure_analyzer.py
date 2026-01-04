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
from sklearn.cluster import KMeans
from scipy.stats import linregress
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="MT5 Structural Scanner", layout="wide", initial_sidebar_state="expanded")

st.title("⚡ MT5 Structural Scanner: Golden Zone & Trend Health")
st.markdown("""
**Objective:** Visualize Algorithmic Structure and Trend Health dynamically.
**Logic:**
* **Structure:** Volume-Weighted K-Means on Velocity Reversals (Snapbacks).
* **Confluence:** Filtered by Higher Timeframe (HTF) & Fibonacci levels.
* **Trend Exhaustion:** Regression Channel analysis to detect when a trend is "leaking" (Overextended).
""")

# --- SIDEBAR: DATA CONTROLS ---
st.sidebar.header("🔌 Connection & Timeframe")

with st.sidebar.expander("MetaTrader 5 Credentials", expanded=True):
    # Fetch from environment variables
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

# HTF Mapping (Base -> Higher)
HTF_MAPPING = {
    mt5.TIMEFRAME_M5: mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_M15: mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H1: mt5.TIMEFRAME_D1
}

# --- LIVE MODE TOGGLE ---
st.sidebar.markdown("---")
live_mode = st.sidebar.checkbox("🔴 Enable Live Refresh (10s)", value=False)

# Data Depth
days_to_fetch = st.sidebar.slider("Days of History", min_value=30, max_value=730, value=90)

# Strategy Selector
st.sidebar.markdown("---")
st.sidebar.header("⚔️ Signal Logic")

n_clusters = st.sidebar.slider("Number of Levels", 3, 10, 3)
ma_period = st.sidebar.number_input("Trend MA Period", min_value=10, max_value=200, value=40)
min_history_days = st.sidebar.number_input("Lookback Days (Structure)", value=7, min_value=1)

c1, c2 = st.columns(2)
use_trend_filter = c1.checkbox("Use Trend Filter", value=True)
# This checkbox controls both HTF and Fib confluence
use_confluence = c2.checkbox("Use Golden Zone (HTF+Fib)", value=True, help="Filters levels against Higher Timeframe & Fibs")
use_fibs = use_confluence # Link fibs to confluence switch for simplicity

# Trend Exhaustion Settings
st.sidebar.markdown("---")
st.sidebar.header("📉 Trend Exhaustion")
show_exhaustion = st.sidebar.checkbox("Show Exhaustion Channels", value=False)
exhaustion_lookback = st.sidebar.slider("Regression Lookback", 50, 300, 100, help="Window to find Max/Min pivots for the channel.")

strategies = st.sidebar.multiselect(
    "Show Signals",
    ["Support Bounce (Buy)", "Resistance Reject (Sell)"],
    default=["Support Bounce (Buy)", "Resistance Reject (Sell)"]
)

# --- HELPER FUNCTIONS ---

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

def get_mt5_data(login, password, server, symbol, timeframe, num_candles):
    """Connects to MT5 and fetches data."""
    if not mt5.initialize():
        return None, f"MT5 Init Failed: {mt5.last_error()}"
    
    if not mt5.terminal_info(): 
        if not mt5.login(login=int(login), password=password, server=server):
            return None, f"MT5 Login Failed: {mt5.last_error()}"
    
    # Check symbol
    if mt5.symbol_info(symbol) is None:
        mt5.shutdown()
        return None, f"Symbol '{symbol}' not found in MT5."
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        return None, "No data received from MT5"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, "Success"

# CACHE WITH TTL (Time To Live) = 10 Seconds
@st.cache_data(ttl=10)
def get_market_data_visual(interval_label, days_back, use_mt5=False, mt5_creds=None):
    """Fetches Data for Visualization."""
    if use_mt5 and mt5_creds:
        tf_const = timeframe_map[interval_label]
        n_candles = get_candles_count(tf_const, days_back)
        
        df_base, msg = get_mt5_data(
            mt5_creds['login'], mt5_creds['pass'], mt5_creds['server'], 
            mt5_creds['symbol'], tf_const, n_candles
        )
        
        # Fetch HTF
        df_htf = None
        htf_const = HTF_MAPPING.get(tf_const, mt5.TIMEFRAME_D1)
        # Fetch fewer candles for HTF (same time duration)
        htf_candles = days_back * 24 if htf_const == mt5.TIMEFRAME_H1 else days_back + 50
        
        df_htf, msg_htf = get_mt5_data(
             mt5_creds['login'], mt5_creds['pass'], mt5_creds['server'], 
             mt5_creds['symbol'], htf_const, htf_candles
        )
        
        # FIX: Remove the last row from HTF data (Current unfinished candle)
        # This prevents repainting bias where the daily level shifts during the day.
        if df_htf is not None and not df_htf.empty:
            df_htf = df_htf.iloc[:-1]
        
        return df_base, df_htf
    return None, None

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

    # Extract weights at reversal points
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

def calculate_fib_levels(df_ohlc):
    """Calculates Fibonacci Retracement levels based on global High/Low."""
    max_price = df_ohlc['high'].max()
    min_price = df_ohlc['low'].min()
    diff = max_price - min_price
    
    levels = {
        0.0: min_price,
        0.236: min_price + diff * 0.236,
        0.382: min_price + diff * 0.382,
        0.5: min_price + diff * 0.5,
        0.618: min_price + diff * 0.618,
        0.786: min_price + diff * 0.786,
        1.0: max_price
    }
    return levels

def calculate_trend_exhaustion(df, lookback=100, std_dev_mult=2.0):
    """
    Calculates a regression channel between the High/Low of the period 
    and measures how much price 'leaks' outside this channel.
    Zero Lookahead: Uses ONLY the data provided in 'df'.
    """
    if len(df) < lookback: return None, None, None, None
    
    # 1. Identify Pivots in the lookback window
    window_df = df.iloc[-lookback:]
    id_max = window_df['high'].idxmax()
    id_min = window_df['low'].idxmin()
    
    # Get integer locations for regression
    t_start = df.index.get_loc(min(id_max, id_min))
    t_end = df.index.get_loc(max(id_max, id_min))
    
    # 2. Fit Regression on the Trend Segment (Pivot to Pivot)
    if t_end - t_start < 5: return None, None, None, None
    
    trend_segment = df.iloc[t_start : t_end+1]
    x = np.arange(t_start, t_end+1)
    y = trend_segment['close'].values
    
    slope, intercept, _, _, _ = linregress(x, y)
    
    # Calculate Standard Deviation of Residuals for Channel Width
    residuals = y - (slope * x + intercept)
    std_resid = np.std(residuals)
    channel_width = std_resid * std_dev_mult
    
    # 3. Project Channel Forward
    # Create arrays for the full lookback window
    full_window_indices = np.arange(len(df) - lookback, len(df))
    
    mid_line = slope * full_window_indices + intercept
    upper_line = mid_line + channel_width
    lower_line = mid_line - channel_width
    
    # 4. Measure Leakage (Density outside channel)
    current_closes = df['close'].iloc[-lookback:].values
    is_above = current_closes > upper_line
    is_below = current_closes < lower_line
    is_leaking = is_above | is_below
    
    recent_n = int(lookback * 0.2)
    leakage_score = np.mean(is_leaking[-recent_n:]) * 100
    
    channel_data = pd.DataFrame({
        'mid': mid_line,
        'upper': upper_line,
        'lower': lower_line,
        'is_leaking': is_leaking
    }, index=df.index[-lookback:])
    
    # Return regression params for future projection
    reg_params = (slope, intercept, channel_width)
    
    return channel_data, leakage_score, slope, reg_params

def get_confluent_levels(base_levels, htf_levels, fib_levels, tolerance=0.003):
    """
    Identifies levels that exist in Base AND (HTF OR Fibs) within tolerance.
    """
    confluent = []
    fib_vals = list(fib_levels.values()) if fib_levels else []
    
    for b_lvl in base_levels:
        htf_match = False
        if htf_levels:
            dist_pct_htf = min([abs(b_lvl - h_lvl) / b_lvl for h_lvl in htf_levels])
            if dist_pct_htf <= tolerance: htf_match = True
        
        fib_match = False
        if fib_vals:
            dist_pct_fib = min([abs(b_lvl - f_lvl) / b_lvl for f_lvl in fib_vals])
            if dist_pct_fib <= tolerance: fib_match = True

        if htf_match or fib_match:
            confluent.append({
                'level': b_lvl,
                'type': 'HTF+Fib' if (htf_match and fib_match) else ('HTF' if htf_match else 'Fib')
            })
            
    return confluent

def highlight_regimes(fig, index, mask, color, label):
    if not mask.any(): return
    blocks = mask.ne(mask.shift()).cumsum()
    true_blocks = blocks[mask]
    for _, block in true_blocks.groupby(true_blocks):
        fig.add_vrect(x0=block.index[0], x1=block.index[-1], fillcolor=color, opacity=0.15, layer="below", line_width=0)

# --- DASHBOARD LOGIC ---

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

# Button Logic: If Live Mode is ON, we don't need to click the button to refresh
fetch_needed = st.sidebar.button("🚀 Render Chart", type="primary") or live_mode

if fetch_needed:
    mt5_creds = {
        'login': mt5_login, 'pass': mt5_pass, 'server': mt5_server, 'symbol': mt5_symbol
    }
    
    # In live mode, we skip the spinner to make it less distracting
    if live_mode:
        df_base, df_htf = get_market_data_visual(selected_tf_label, days_to_fetch, use_mt5=True, mt5_creds=mt5_creds)
    else:
        with st.spinner("Calculating Structure..."):
            df_base, df_htf = get_market_data_visual(selected_tf_label, days_to_fetch, use_mt5=True, mt5_creds=mt5_creds)
    
    if df_base is not None:
        st.session_state['viz_data'] = df_base
        st.session_state['viz_htf'] = df_htf
        st.session_state['data_loaded'] = True
    else:
        st.error("Failed to fetch data from MT5.")

# --- MAIN DISPLAY ---

if st.session_state['data_loaded'] and 'viz_data' in st.session_state:
    df_full = st.session_state['viz_data']
    df_htf_full = st.session_state.get('viz_htf', None)
    
    # --- TIME TRAVEL SLIDER ---
    st.markdown("### ⏳ Time Travel (Verify History)")
    
    max_idx = len(df_full) - 1
    candles_per_day = get_candles_per_day(selected_tf_mt5)
    min_idx = min_history_days * candles_per_day
    
    if min_idx >= max_idx:
        st.error(f"Not enough data for Lookback of {min_history_days} days. Fetch more history.")
    else:
        # Default to "Now"
        sim_idx = st.slider("Playback Position", min_value=min_idx, max_value=max_idx, value=max_idx)
        
        sim_time = df_full.index[sim_idx]
        st.caption(f"Simulating Analysis at: **{sim_time}**")
        
        # 1. STRICT DATA PARTITIONING
        # "Memory" (Past): Data STRICTLY UP TO sim_idx
        df_slice = df_full.iloc[sim_idx - min_idx : sim_idx] 
        
        # 2. Calculate Base Structure on MEMORY only
        base_levels = calculate_levels_weighted_kmeans(df_slice, n_clusters=n_clusters)
        
        # 3. Calculate Confluence (STRICT Anti-Lookahead)
        htf_levels = []
        if use_confluence and df_htf_full is not None:
            # SAFETY FILTER: Only use HTF candles where the DATE is strictly BEFORE sim_time date
            sim_date = pd.Timestamp(sim_time.date())
            df_htf_safe = df_htf_full[df_htf_full.index < sim_date]
            
            if len(df_htf_safe) > 20:
                htf_levels = calculate_levels_weighted_kmeans(df_htf_safe.iloc[-60:], max(3, n_clusters // 2))
            
        fib_levels = calculate_fib_levels(df_slice) if use_fibs else {}
        
        confluent_data = get_confluent_levels(base_levels, htf_levels, fib_levels)
        confluent_map = {c['level']: c['type'] for c in confluent_data}
        
        # 4. PLOTTING
        fig = go.Figure()

        # Ghost Trace (Future)
        df_future = df_full.iloc[sim_idx:]
        if not df_future.empty:
            fig.add_trace(go.Candlestick(
                x=df_future.index,
                open=df_future['open'], high=df_future['high'],
                low=df_future['low'], close=df_future['close'],
                name="Future (Unknown)",
                increasing_line_color='gray', decreasing_line_color='gray',
                opacity=0.3
            ))
        
        # Active Trace (History)
        fig.add_trace(go.Candlestick(
            x=df_slice.index,
            open=df_slice['open'], high=df_slice['high'],
            low=df_slice['low'], close=df_slice['close'],
            name="Analyzed Data"
        ))
        
        # Simulation Marker
        fig.add_vline(x=sim_time.value/1e6, line_width=2, line_dash="dash", line_color="white", annotation_text="ANALYSIS TIME")

        # --- TREND EXHAUSTION OVERLAY ---
        if show_exhaustion:
            # Calculate strictly on the visible slice (Past)
            ch_data, leak_score, slope, reg_params = calculate_trend_exhaustion(df_slice, lookback=exhaustion_lookback)
            
            if ch_data is not None:
                # 1. Plot Channel Bounds (History)
                fig.add_trace(go.Scatter(
                    x=ch_data.index, y=ch_data['upper'], mode='lines', name='Upper Channel',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=ch_data.index, y=ch_data['lower'], mode='lines', name='Lower Channel',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # 2. Visualizing Leakage
                mask_up = ch_data['is_leaking'] & (df_slice['close'].iloc[-exhaustion_lookback:] > ch_data['upper'])
                mask_down = ch_data['is_leaking'] & (df_slice['close'].iloc[-exhaustion_lookback:] < ch_data['lower'])
                
                if mask_up.any():
                    fig.add_trace(go.Scatter(x=ch_data.index[mask_up], y=df_slice['close'].iloc[-exhaustion_lookback:][mask_up],
                        mode='markers', name='Upside Exhaustion', marker=dict(color='rgba(255, 0, 0, 0.6)', symbol='x', size=6)))
                
                if mask_down.any():
                    fig.add_trace(go.Scatter(x=ch_data.index[mask_down], y=df_slice['close'].iloc[-exhaustion_lookback:][mask_down],
                        mode='markers', name='Downside Exhaustion', marker=dict(color='rgba(255, 0, 0, 0.6)', symbol='x', size=6)))

                # 3. Future Projection (Visualization only)
                # Project the regression lines into the future (Gray zone) to check if trend holds
                if not df_future.empty:
                    f_slope, f_intercept, f_width = reg_params
                    # Generate indices for future (continuing from end of slice)
                    start_idx = len(df_full) - len(df_future) # Incorrect for index math, using timestamps better
                    # Simpler: Create X array extension
                    future_len = len(df_future)
                    # We need the X values relative to the regression calculation
                    # X for regression ended at len(df_slice) - 1
                    # So future X is range(len(df_slice), len(df_slice) + future_len)
                    
                    # NOTE: Regression was calc'd on a small window at the END of df_slice.
                    # We need to project that specific line forward.
                    # Regression X values were 0..lookback-1 corresponding to df_slice[-lookback:]
                    # So next candle is X = lookback
                    
                    x_fut = np.arange(exhaustion_lookback, exhaustion_lookback + future_len)
                    # To align with chart, we need to shift the intercept relative to the slice window?
                    # The regression was y = mx + c where x=0 is df_slice.index[-lookback]
                    
                    # Re-calculate mid-line using the stored params for the future segment
                    fut_mid = f_slope * x_fut + f_intercept
                    fut_upper = fut_mid + f_width
                    fut_lower = fut_mid - f_width
                    
                    fig.add_trace(go.Scatter(x=df_future.index, y=fut_upper, mode='lines', name='Channel Project', line=dict(color='gray', width=1, dash='dot'), opacity=0.3, showlegend=False))
                    fig.add_trace(go.Scatter(x=df_future.index, y=fut_lower, mode='lines', name='Channel Project', line=dict(color='gray', width=1, dash='dot'), opacity=0.3, showlegend=False))

                # 4. Status Metric
                trend_dir = "UP" if slope > 0 else "DOWN"
                health = "HEALTHY"
                health_color = "green"
                if leak_score > 20: 
                    health = "EXHAUSTED / LEAKING"
                    health_color = "red"
                elif leak_score > 5:
                    health = "WEAKENING"
                    health_color = "orange"
                st.info(f"**Trend Diagnostic:** {trend_dir} Trend | Structure: :{health_color}[{health}] (Leakage: {leak_score:.1f}%)")

        # Fib Overlay (Based on slice)
        if use_fibs:
            for ratio, val in fib_levels.items():
                f_color = "gold" if ratio == 0.618 else "cyan"
                f_width = 2 if ratio == 0.618 else 1
                fig.add_shape(
                    type="line", x0=df_slice.index[0], x1=df_full.index[-1], 
                    y0=val, y1=val,
                    line=dict(color=f_color, width=f_width, dash="dot"),
                    opacity=0.5
                )

        # Structural Lines (Projected from Slice into Future)
        for l in base_levels:
            is_confluent = l in confluent_map
            conf_type = confluent_map.get(l, "")
            
            if is_confluent:
                if conf_type == 'HTF+Fib':
                    color = "gold"
                    width = 4
                    label = f"🏆 GOLDEN ZONE: {l:.2f}"
                elif conf_type == 'Fib':
                    color = "cyan"
                    width = 2
                    label = f"Fib Confluence: {l:.2f}"
                else:
                    color = "mediumpurple"
                    width = 3
                    label = f"HTF Confluence: {l:.2f}"
                dash = "solid"
            else:
                color = "yellow"
                width = 1
                dash = "dot"
                label = f"Lvl: {l:.2f}"
            
            # Find Start Point (First Interaction WITHIN MEMORY)
            mask = (df_slice['low'] <= l) & (df_slice['high'] >= l)
            
            if mask.any():
                start_date = mask.idxmax()
                
                # Draw Segment
                fig.add_shape(
                    type="line",
                    x0=start_date, y0=l,
                    x1=df_full.index[-1], y1=l, # Project forward to show validity
                    line=dict(color=color, width=width, dash=dash),
                )
                
                # Label at Future edge
                fig.add_annotation(
                    x=df_full.index[-1], y=l,
                    text=label,
                    showarrow=False,
                    xanchor="left",
                    yshift=5,
                    font=dict(color=color)
                )

        current_price = df_slice['close'].iloc[-1]
        title_text = f"Time Travel: {mt5_symbol} @ {sim_time} | Price: {current_price:.2f}"

        fig.update_layout(
            title=title_text,
            yaxis_title="Price",
            template="plotly_dark",
            height=800,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            uirevision='constant' 
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Structural Snapshot:** Levels derived from data spanning {df_slice.index[0]} to {sim_time}. Future price action (Gray) validates if levels held.")
                
    # --- AUTO REFRESH LOGIC ---
    if live_mode:
        time.sleep(10)
        st.rerun()

else:
    st.info("👈 Enter settings and click 'Render Chart' to visualize structure.")