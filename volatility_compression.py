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

st.set_page_config(page_title="Volatility Compression Hunter", layout="wide")
st.title("🗜️ Volatility Compression Hunter")
st.markdown("""
**Theory:** Markets move from High Volatility to Low Volatility and back. 
**Goal:** Identify periods where price is "coiling" (Compression) to anticipate an explosive breakout.
""")

# --- SIDEBAR ---
with st.sidebar.expander("MT5 Settings", expanded=True):
    env_login = os.getenv("MT5_LOGIN")
    default_login = int(env_login) if env_login else 0
    default_pass = os.getenv("MT5_PASSWORD", "")
    default_server = os.getenv("MT5_SERVER", "HFMarketsSA-Live2")
    
    mt5_login = st.number_input("Login", value=default_login, step=1)
    mt5_pass = st.text_input("Password", value=default_pass, type="password") 
    mt5_server = st.text_input("Server", value=default_server)

st.sidebar.markdown("---")
symbol = st.sidebar.text_input("Symbol", "USA100")
timeframe_map = {
    "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, 
    "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}
tf_label = st.sidebar.selectbox("Timeframe", list(timeframe_map.keys()), index=1)
tf_mt5 = timeframe_map[tf_label]
lookback = st.sidebar.slider("Lookback Candles", 1000, 10000, 3000)

st.sidebar.subheader("Compression Settings")
bb_period = st.sidebar.number_input("Bollinger Period", value=20)
bb_std = st.sidebar.number_input("Bollinger Std Dev", value=2.0)
squeeze_threshold = st.sidebar.slider("Low Volatility Percentile (%)", 1, 20, 5, help="Highlight when volatility is in the bottom X% of history.")

# --- FUNCTIONS ---

def get_mt5_data(login, password, server, symbol, timeframe, num_candles):
    if not mt5.initialize(): return None, f"Init Failed: {mt5.last_error()}"
    
    # Login only if needed
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
            err = mt5.last_error()
            mt5.shutdown()
            return None, f"Login Failed: {err}"

    # Enable Symbol
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        return None, f"Symbol {symbol} not found."
    
    # Retry Loop
    rates = None
    attempts = 0
    while attempts < 3:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
        if rates is not None and len(rates) > 0: break
        attempts += 1
        time.sleep(1)
            
    mt5.shutdown()
    
    if rates is None or len(rates) == 0: return None, "No Data"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, "Success"

def calculate_compression(df, bb_period, bb_std, threshold_pct):
    # 1. Bollinger Bands
    df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
    
    # 2. Bandwidth (Raw Volatility Metric)
    # Formula: (Upper - Lower) / Middle
    df['Bandwidth'] = (df['upper'] - df['lower']) / df['middle']
    
    # 3. Percentile Rank (The "Squeeze" Detector)
    # "Is today's bandwidth lower than 95% of the last 120 bars?"
    rolling_window = 120 # Look back roughly 5 days on H1
    df['Bandwidth_Rank'] = df['Bandwidth'].rolling(window=rolling_window).rank(pct=True) * 100
    
    # 4. Identify Squeeze Zones
    df['Squeeze_Active'] = df['Bandwidth_Rank'] <= threshold_pct
    
    # 5. Momentum (for direction bias)
    df['Momentum'] = talib.MOM(df['close'], timeperiod=12)
    
    return df

# --- MAIN ---

if st.sidebar.button("🔍 Analyze Volatility"):
    with st.spinner("Crunching numbers..."):
        df, msg = get_mt5_data(mt5_login, mt5_pass, mt5_server, symbol, tf_mt5, lookback)
        
        if df is not None:
            df = calculate_compression(df, bb_period, bb_std, squeeze_threshold)
            
            # --- STATISTICS ---
            current_bw = df['Bandwidth'].iloc[-1]
            avg_bw = df['Bandwidth'].mean()
            is_squeezing = df['Squeeze_Active'].iloc[-1]
            
            st.subheader(f"Status: {'🔴 EXTREME COMPRESSION' if is_squeezing else '🟢 Normal / Expansion'}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Bandwidth", f"{current_bw:.4f}", delta_color="inverse", delta=f"{current_bw - avg_bw:.4f} vs Avg")
            c2.metric("Squeeze Threshold", f"{df['Bandwidth'].quantile(squeeze_threshold/100):.4f}", f"Bottom {squeeze_threshold}%")
            c3.metric("Last Price", f"{df['close'].iloc[-1]}")

            # --- PLOTTING ---
            # Create a 3-pane chart: Price, Bandwidth, and Histogram of Volatility
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05, 
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f"{symbol} Price & Bands", "Bollinger Bandwidth (Volatility)", "Momentum")
            )

            # Row 1: Price + Bands
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['upper'], line=dict(color='gray', width=1), name="Upper BB"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['lower'], line=dict(color='gray', width=1), name="Lower BB", fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

            # Highlight Squeezes on Price
            squeeze_dates = df[df['Squeeze_Active']].index
            # We add vertical shapes for squeeze zones
            shapes = []
            for date in squeeze_dates:
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=date, x1=date, y0=0, y1=1, fillcolor="orange", opacity=0.3, layer="below", line_width=0))
            # Note: Adding too many shapes slows Plotly. We'll visualize via the subplot instead.

            # Row 2: Bandwidth
            # Color code: Orange if squeezing, Blue if normal
            colors = np.where(df['Squeeze_Active'], 'orange', '#00CC96')
            fig.add_trace(go.Bar(x=df.index, y=df['Bandwidth'], name="Bandwidth", marker_color=colors), row=2, col=1)
            # Add a horizontal line for the historical low threshold
            threshold_val = df['Bandwidth'].quantile(squeeze_threshold/100)
            fig.add_hline(y=threshold_val, line_dash="dot", line_color="red", annotation_text="Squeeze Zone", row=2, col=1)

            # Row 3: Momentum (To show direction during breakout)
            fig.add_trace(go.Bar(x=df.index, y=df['Momentum'], name="Momentum", marker_color='cyan'), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # --- HISTOGRAM ANALYSIS ---
            st.markdown("### 📊 Distribution of Volatility")
            st.markdown("This histogram shows how rare the current volatility level is. If the current level (red line) is far to the left, we are in a rare compression event.")
            
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=df['Bandwidth'], nbinsx=100, name="Historical Bandwidth", marker_color='#333'))
            hist_fig.add_vline(x=current_bw, line_width=3, line_color="red", annotation_text="CURRENT")
            hist_fig.update_layout(height=300, template="plotly_dark", margin=dict(t=0, b=0))
            st.plotly_chart(hist_fig, use_container_width=True)

        else:
            st.error(msg)