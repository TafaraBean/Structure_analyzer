import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Wick Context Analyzer", layout="wide")
st.title("🕯️ Wick Context: Finding The 'Lock'")
st.markdown("""
**Objective:** Identify zones where Market Structure aligns across both candle colors.
**Bullish Lock (Green Zone):** Bulls are Strong (Green candles close high) AND Bears are Weak (Red candles have long bottom wicks).
**Bearish Lock (Red Zone):** Bears are Strong (Red candles close low) AND Bulls are Weak (Green candles have long top wicks).
""")

# --- SIDEBAR ---
with st.sidebar.expander("MetaTrader 5 Credentials", expanded=True):
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
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, 
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1
}
tf_label = st.sidebar.selectbox("Timeframe", list(timeframe_map.keys()), index=1)
tf_mt5 = timeframe_map[tf_label]
lookback = st.sidebar.slider("Candles to Load", 1000, 50000, 5000)

st.sidebar.subheader("Rolling Settings")
rolling_window = st.sidebar.slider("Rolling Mean Window", 10, 200, 50)

# --- FUNCTIONS ---
def get_mt5_data(login, password, server, symbol, timeframe, n):
    if not mt5.initialize(): return None, f"Init Failed: {mt5.last_error()}"
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
            mt5.shutdown(); return None, f"Login Failed"
    if not mt5.symbol_select(symbol, True): return None, f"Symbol not found"
    
    rates = None
    attempts = 0
    while attempts < 3:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if rates is not None and len(rates) > 0: break
        attempts += 1; time.sleep(1)
            
    mt5.shutdown()
    if rates is None: return None, "No Data"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, "Success"

def calculate_context_stats(df, window):
    # 1. Geometry
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    
    df['Top_Wick'] = hi - np.maximum(op, cl)
    df['Bottom_Wick'] = np.minimum(op, cl) - lo
    
    # 2. Split Data by Color
    is_green = cl > op
    is_red = cl < op
    
    # 3. Rolling Calculations (Handle gaps with min_periods)
    # Green Candles: Bulls
    df['Green_Top_Avg'] = np.where(is_green, df['Top_Wick'], np.nan)
    df['Green_Top_Avg'] = df['Green_Top_Avg'].rolling(window, min_periods=1).mean()
    
    df['Green_Bot_Avg'] = np.where(is_green, df['Bottom_Wick'], np.nan)
    df['Green_Bot_Avg'] = df['Green_Bot_Avg'].rolling(window, min_periods=1).mean()
    
    # Red Candles: Bears
    df['Red_Top_Avg'] = np.where(is_red, df['Top_Wick'], np.nan)
    df['Red_Top_Avg'] = df['Red_Top_Avg'].rolling(window, min_periods=1).mean()
    
    df['Red_Bot_Avg'] = np.where(is_red, df['Bottom_Wick'], np.nan)
    df['Red_Bot_Avg'] = df['Red_Bot_Avg'].rolling(window, min_periods=1).mean()
    
    # --- 4. CONTEXT DETECTION (THE LOCK) ---
    
    # BULLISH LOCK:
    # 1. Green Line > Orange Line (Bulls closing high)
    # 2. Cyan Line > Red Line (Bears getting rejected)
    df['Bull_Lock'] = (df['Green_Bot_Avg'] > df['Green_Top_Avg']) & (df['Red_Bot_Avg'] > df['Red_Top_Avg'])
    
    # BEARISH LOCK:
    # 1. Orange Line > Green Line (Bulls failing to hold highs)
    # 2. Red Line > Cyan Line (Bears closing low)
    df['Bear_Lock'] = (df['Green_Top_Avg'] > df['Green_Bot_Avg']) & (df['Red_Top_Avg'] > df['Red_Bot_Avg'])
    
    return df

# --- MAIN ---
if st.sidebar.button("📊 Analyze Context"):
    with st.spinner("Finding market locks..."):
        df, msg = get_mt5_data(mt5_login, mt5_pass, mt5_server, symbol, tf_mt5, lookback)
        
        if df is not None:
            df = calculate_context_stats(df, rolling_window)
            
            # --- STATISTICS ---
            bull_lock_count = df['Bull_Lock'].sum()
            bear_lock_count = df['Bear_Lock'].sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Bullish Lock Zones", f"{bull_lock_count}", "Strong Buy Context")
            c2.metric("Bearish Lock Zones", f"{bear_lock_count}", "Strong Sell Context")
            
            current_state = "Neutral / Mixed"
            if df['Bull_Lock'].iloc[-1]: current_state = "🟢 BULLISH LOCK"
            elif df['Bear_Lock'].iloc[-1]: current_state = "🔴 BEARISH LOCK"
            c3.metric("Current State", current_state)
            
            # --- CHARTS ---
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(
                    "Price Action & Context Zones", 
                    "Green Candle Anatomy (Buying Behavior)", 
                    "Red Candle Anatomy (Selling Behavior)"
                )
            )
            
            # 1. Price Chart
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
            
            # Highlight Zones
            # We use markers at the high/low to indicate the zone cleanly
            bull_zone = df[df['Bull_Lock']]
            fig.add_trace(go.Scatter(
                x=bull_zone.index, y=bull_zone['low'], 
                mode='markers', marker=dict(symbol='square', color='rgba(0, 255, 0, 0.3)', size=5),
                name="Bullish Lock Zone"
            ), row=1, col=1)
            
            bear_zone = df[df['Bear_Lock']]
            fig.add_trace(go.Scatter(
                x=bear_zone.index, y=bear_zone['high'], 
                mode='markers', marker=dict(symbol='square', color='rgba(255, 0, 0, 0.3)', size=5),
                name="Bearish Lock Zone"
            ), row=1, col=1)
            
            # 2. Green Stats (Bulls)
            # Orange Line = Top Wick Avg (Resistance on Bulls)
            fig.add_trace(go.Scatter(x=df.index, y=df['Green_Top_Avg'], line=dict(color='orange', width=2), name="Avg Green Top Wick"), row=2, col=1)
            # Green Line = Bot Wick Avg (Support on Bulls)
            fig.add_trace(go.Scatter(x=df.index, y=df['Green_Bot_Avg'], line=dict(color='green', width=2), name="Avg Green Bot Wick"), row=2, col=1)
            
            # 3. Red Stats (Bears)
            # Red Line = Top Wick Avg (Resistance on Bears)
            fig.add_trace(go.Scatter(x=df.index, y=df['Red_Top_Avg'], line=dict(color='red', width=2), name="Avg Red Top Wick"), row=3, col=1)
            # Cyan Line = Bot Wick Avg (Support on Bears)
            fig.add_trace(go.Scatter(x=df.index, y=df['Red_Bot_Avg'], line=dict(color='cyan', width=2), name="Avg Red Bot Wick"), row=3, col=1)

            fig.update_layout(height=1000, template="plotly_dark", xaxis_rangeslider_visible=False)
            fig.update_xaxes(matches='x')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("""
            **Analysis Guide:**
            * **Bullish Lock (Green Squares):** Occurs when buyers are dominating their own candles (Green Line > Orange) AND buyers are ALSO dominating the sellers' candles (Cyan Line > Red). This is a "double confirmation" of demand.
            * **Bearish Lock (Red Squares):** Occurs when sellers are dominating their own candles (Red Line > Cyan) AND sellers are ALSO dominating the buyers' candles (Orange Line > Green).
            """)
            
        else:
            st.error(msg)