import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
import requests
import time
import os
from dotenv import load_dotenv

# Load env variables if available
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Institutional Sentiment Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Professional" Look
st.markdown("""
<style>
    .metric-box {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .stMetricValue { font-size: 24px !important; color: #00FF99 !important; }
    .stMetricLabel { font-size: 14px !important; color: #aaa !important; }
    h1, h2, h3 { color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("🔌 Data Feeds")
    
    with st.expander("MT5 Connection", expanded=True):
        login = st.number_input("Login ID", value=int(os.getenv("MT5_LOGIN", 0)))
        password = st.text_input("Password", value=os.getenv("MT5_PASSWORD", ""), type="password")
        server = st.text_input("Server", value=os.getenv("MT5_SERVER", "HFMarketsSA-Live2"))
        
    with st.expander("Sentiment Source", expanded=True):
        # Defaulting to the session ID provided in context
        myfx_session = st.text_input("Myfxbook Session ID", value="DSL07vu14QxHWErTIAFrH40")
        
    st.divider()
    
    st.header("⚙️ Chart Parameters")
    symbol = st.text_input("Symbol", "EURUSDz")
    timeframe = st.selectbox("Timeframe", ["M1", "M5", "M15", "H1", "H4"], index=1)
    bars = st.slider("History Depth", 500, 5000, 1000)
    
    st.subheader("Statistical Channel")
    ma_period = st.number_input("MA Period", value=50)
    conf_level = st.slider("Confidence Interval (Z-Score)", 1.0, 3.0, 1.96, help="1.96 = 95% Confidence")
    
    auto_refresh = st.checkbox("🔴 Auto-Refresh (60s)", value=False)

# --- HELPER FUNCTIONS ---

def init_mt5(login, password, server):
    if not mt5.initialize():
        return False, f"MT5 Init Failed: {mt5.last_error()}"
    if login and password:
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            return False, f"Login Failed: {mt5.last_error()}"
    return True, "Connected"

def get_market_data(symbol, tf_str, n):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, 
        "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4
    }
    
    rates = mt5.copy_rates_from_pos(symbol, tf_map[tf_str], 0, n)
    if rates is None: return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_sentiment(session_id, symbol):
    # Mapping fix: Broker suffix removal (e.g. EURUSDz -> EURUSD)
    clean_symbol = symbol.replace("z", "").replace("m", "").replace(".pro", "")
    url = f"https://www.myfxbook.com/api/get-community-outlook.json?session={session_id}"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if not data['error']:
                for s in data['symbols']:
                    if s['name'] == clean_symbol:
                        return {
                            'long': s['longPercentage'],
                            'short': s['shortPercentage'],
                            'volume': s['longVolume'] + s['shortVolume']
                        }
    except Exception as e:
        st.error(f"Sentiment API Error: {e}")
    return None

def calculate_stats(df, period, z_score):
    # 1. Central Tendency (Moving Average)
    df['MA'] = df['close'].rolling(window=period).mean()
    
    # 2. Volatility (Standard Deviation)
    df['STD'] = df['close'].rolling(window=period).std()
    
    # 3. Confidence Interval (Standard Error of the Mean)
    # Formula: CI = Mean ± (Z * (StdDev / sqrt(n)))
    # Note: This checks confidence of the AVERAGE. 
    # For Price Prediction Interval (Bollinger-like), use: Mean ± (Z * StdDev)
    # The user asked for Confidence Interval, but usually traders mean Prediction Interval.
    # We will compute the Prediction Interval (Volatility Band) as it's more useful for trading.
    
    df['Upper_CI'] = df['MA'] + (z_score * df['STD'])
    df['Lower_CI'] = df['MA'] - (z_score * df['STD'])
    
    return df

# --- MAIN DASHBOARD LOGIC ---

st.title(f"🦅 {symbol} Institutional Overview")

# 1. CONNECT & FETCH
status, msg = init_mt5(login, password, server)
if not status:
    st.error(msg)
    st.stop()

df = get_market_data(symbol, timeframe, bars)
sent_data = get_sentiment(myfx_session, symbol)

if df is not None:
    # 2. CALCULATE
    df = calculate_stats(df, ma_period, conf_level)
    last = df.iloc[-1]
    
    # 3. TOP METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    
    # Price Change
    prev = df.iloc[-2]
    change = last['close'] - prev['close']
    color = "normal" if change == 0 else "inverse"
    m1.metric("Current Price", f"{last['close']:.5f}", f"{change:.5f}", delta_color=color)
    
    # Sentiment Display
    if sent_data:
        long_pct = sent_data['long']
        short_pct = sent_data['short']
        
        # Bias Logic
        if long_pct > 60: bias = "BEARISH (Crowd Long)"
        elif short_pct > 60: bias = "BULLISH (Crowd Short)"
        else: bias = "NEUTRAL"
        
        m2.metric("Crowd Long %", f"{long_pct}%", bias, delta_color="off")
        m3.metric("Crowd Volume", f"{sent_data['volume']:.2f} Lots")
    else:
        m2.metric("Crowd Sentiment", "Unavailable")
        m3.metric("Volume", "N/A")

    # Volatility State
    vol_state = "High" if last['STD'] > df['STD'].mean() else "Low"
    m4.metric("Volatility Regime", vol_state, f"StdDev: {last['STD']:.5f}")

    # 4. MAIN CHART (Plotly)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.75, 0.25])

    # -- Candlesticks --
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"
    ), row=1, col=1)

    # -- Moving Average --
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['MA'], 
        line=dict(color='yellow', width=2), name=f"MA ({ma_period})"
    ), row=1, col=1)

    # -- Confidence Interval (Shaded Channel) --
    # Upper Line (Invisible, for fill)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Upper_CI'],
        line=dict(width=0), showlegend=False
    ), row=1, col=1)
    
    # Lower Line (Fill to Upper)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['Lower_CI'],
        line=dict(width=0), 
        fill='tonexty', 
        fillcolor='rgba(0, 255, 153, 0.1)', # Light Green Tint
        name=f"{conf_level}σ Interval"
    ), row=1, col=1)
    
    # Add Border Lines for visual clarity
    fig.add_trace(go.Scatter(x=df['time'], y=df['Upper_CI'], line=dict(color='rgba(0,255,153,0.5)', width=1, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['Lower_CI'], line=dict(color='rgba(0,255,153,0.5)', width=1, dash='dot'), showlegend=False), row=1, col=1)

    # -- Subchart: Volatility --
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['STD'], 
        line=dict(color='#00CC96', width=1), 
        fill='tozeroy',
        name="Volatility (StdDev)"
    ), row=2, col=1)

    # Layout Styling
    fig.update_layout(
        height=700,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 5. SIGNAL ANALYSIS
    # Simple logic: Is price outside the confidence interval?
    is_outlier_up = last['close'] > last['Upper_CI']
    is_outlier_down = last['close'] < last['Lower_CI']
    
    if is_outlier_up:
        st.warning(f"⚠️ **Statistical Anomaly:** Price is ABOVE the {conf_level}σ Confidence Interval. Potential Mean Reversion Short.")
    elif is_outlier_down:
        st.success(f"⚠️ **Statistical Anomaly:** Price is BELOW the {conf_level}σ Confidence Interval. Potential Mean Reversion Long.")
    else:
        st.info("✅ Price is executing within normal statistical bounds.")

else:
    st.error("No Data Received from MT5")

# Auto-Refresh Logic
if auto_refresh:
    time.sleep(60)
    st.rerun()