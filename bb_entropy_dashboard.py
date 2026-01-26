import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import talib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(layout="wide", page_title="BB Width Entropy Dashboard")

# --- Functions ---

@st.cache_resource
def init_mt5():
    """Initialize MT5 connection."""
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    params = {}
    if path: params["path"] = path
    
    if not mt5.initialize(**params):
        st.error(f"❌ MT5 Init failed: {mt5.last_error()}")
        return False
        
    if login and password and server:
        mt5.login(login=int(login), password=password, server=server)
    return True

@st.cache_data(ttl=60)
def fetch_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def permutation_entropy(series, order=3, delay=1, window_size=30):
    """Vectorized Permutation Entropy."""
    vals = series.values
    n = len(vals)
    result = np.full(n, np.nan)
    denom = np.log2(math.factorial(order))
    
    for i in range(window_size, n):
        window = vals[i-window_size : i]
        n_w = len(window)
        if n_w < order*delay: continue
        
        partitions = np.array([window[j:j+order*delay:delay] for j in range(n_w - order * delay + 1)])
        ords = np.argsort(partitions, axis=1)
        _, counts = np.unique(ords, axis=0, return_counts=True)
        probs = counts / len(ords)
        pe = -np.sum(probs * np.log2(probs + 1e-9))
        result[i] = pe / denom
        
    return pd.Series(result, index=series.index)

# --- Sidebar ---
st.sidebar.title("Configuration")
SYMBOL = st.sidebar.text_input("Symbol", "EURUSDm")
TIMEFRAME_MAP = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}
TIMEFRAME_STR = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=2)
TIMEFRAME = TIMEFRAME_MAP[TIMEFRAME_STR]
BARS = st.sidebar.number_input("Bars", min_value=100, max_value=10000, value=2000)

st.sidebar.subheader("Bollinger Bands")
BB_PERIOD = st.sidebar.slider("BB Period", 5, 100, 20)
BB_STD = st.sidebar.slider("BB Std Dev", 1.0, 3.0, 2.0, 0.1)

st.sidebar.subheader("Permutation Entropy")
PE_ORDER = st.sidebar.slider("PE Order", 3, 7, 4)
PE_DELAY = st.sidebar.slider("PE Delay", 1, 5, 1)
PE_WINDOW = st.sidebar.slider("PE Window", 10, 100, 30)

if st.sidebar.button("Run Analysis") or True:
    if not init_mt5():
        st.stop()
        
    status_text = st.empty()
    status_text.text(f"Fetching {BARS} bars for {SYMBOL}...")
    
    df = fetch_data(SYMBOL, TIMEFRAME, BARS)
    
    if df is None:
        st.error("Failed to fetch data. Check Symbol/Connection.")
        st.stop()
        
    status_text.text("Calculating Bollinger Bands...")
    
    # Calculate Bollinger Bands
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=BB_PERIOD, nbdevup=BB_STD, nbdevdn=BB_STD)
    bb_width = upper - lower
    bb_width_pct = (bb_width / middle) * 100  # Width as percentage of middle band
    
    status_text.text("Calculating Permutation Entropy...")
    
    # Calculate Permutation Entropy of BB Width
    pe_bb_width = permutation_entropy(bb_width_pct, order=PE_ORDER, delay=PE_DELAY, window_size=PE_WINDOW)
    
    status_text.text("Rendering Visualization...")
    
    # --- Plotting ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.2)
    
    # Panel 1: Price with Bollinger Bands
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], color='white', linewidth=1, label='Close')
    ax1.plot(df.index, upper, color='cyan', linewidth=1, alpha=0.7, label='Upper BB')
    ax1.plot(df.index, middle, color='yellow', linewidth=1, alpha=0.7, label='Middle BB')
    ax1.plot(df.index, lower, color='cyan', linewidth=1, alpha=0.7, label='Lower BB')
    ax1.fill_between(df.index, upper, lower, color='cyan', alpha=0.1)
    ax1.set_title(f"{SYMBOL} ({TIMEFRAME_STR}) - Bollinger Bands & Width Entropy", fontsize=14, color='white')
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    
    # Panel 2: BB Width (%)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, bb_width_pct, color='orange', linewidth=1.5, label='BB Width %')
    ax2.set_ylabel("BB Width (%)")
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.2)
    
    # Panel 3: Permutation Entropy of BB Width
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, pe_bb_width, color='lime', linewidth=1.5, label='PE(BB Width)')
    ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='PE=0.5')
    ax3.set_ylabel("Permutation Entropy")
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.2)
    
    # Panel 4: Interpretation (High/Low Entropy Zones)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    # Color zones based on entropy level
    low_entropy = pe_bb_width < 0.3
    high_entropy = pe_bb_width > 0.7
    
    ax4.fill_between(df.index, 0, 1, where=low_entropy, color='green', alpha=0.3, label='Low Entropy (Stable)')
    ax4.fill_between(df.index, 0, 1, where=high_entropy, color='red', alpha=0.3, label='High Entropy (Chaotic)')
    ax4.set_ylabel("Regime")
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    status_text.empty()
    st.pyplot(fig)
    
    # --- Statistics ---
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current BB Width %", f"{bb_width_pct.iloc[-1]:.3f}%")
    with col2:
        st.metric("Current PE", f"{pe_bb_width.iloc[-1]:.3f}")
    with col3:
        st.metric("Avg PE", f"{pe_bb_width.mean():.3f}")
    with col4:
        regime = "Stable" if pe_bb_width.iloc[-1] < 0.5 else "Chaotic"
        st.metric("Current Regime", regime)
    
    # --- Interpretation Guide ---
    st.subheader("Interpretation")
    st.markdown("""
    **Permutation Entropy (PE) of Bollinger Band Width:**
    - **Low PE (< 0.3)**: BB Width is changing in a **predictable/stable** pattern → Market volatility is structured
    - **Medium PE (0.3-0.7)**: BB Width has **moderate complexity** → Transitional volatility regime
    - **High PE (> 0.7)**: BB Width is changing **chaotically** → Market volatility is unpredictable
    
    **Trading Implications:**
    - Low entropy zones may indicate consolidation or structured volatility expansion/contraction
    - High entropy zones suggest erratic volatility changes, potentially signaling regime shifts
    """)
