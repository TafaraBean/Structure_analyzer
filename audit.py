import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import MetaTrader5 as mt5
import talib 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Pattern Logic Verifier (M5)", layout="wide")
st.title("🔬 TA-Lib Pattern Logic Verifier (M5 Timeframe)")
st.markdown("""
**Goal:** Verify the Integer Codes (-100 vs 100) for candlestick patterns on the **5-Minute** chart.
* **100** usually means **Bullish**.
* **-100** usually means **Bearish**.
""")

# --- SETUP ---
with st.sidebar.expander("MT5 Login", expanded=True):
    login = int(os.getenv("MT5_LOGIN", 0))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "HFMarketsSA-Live2")

symbol = st.sidebar.text_input("Symbol", "USA100")
candles = st.sidebar.number_input("Candles to Analyze", value=1000)

# --- SET TIMEFRAME TO M5 ---
tf = mt5.TIMEFRAME_M5 
# ---------------------------

# --- FUNCTION ---
def get_data(login, password, server, symbol, n):
    if not mt5.initialize(): return None
    if login: mt5.login(login=login, password=password, server=server)
    if not mt5.symbol_select(symbol, True): return None
    
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    mt5.shutdown()
    
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

# --- MAIN ---
if st.button("🔍 Check Patterns on M5"):
    df = get_data(login, password, server, symbol, candles)
    
    if df is not None:
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        
        # 1. CALCULATE PATTERNS
        df['HangingMan'] = talib.CDLHANGINGMAN(o, h, l, c)
        df['ShootingStar'] = talib.CDLSHOOTINGSTAR(o, h, l, c)
        df['Hammer'] = talib.CDLHAMMER(o, h, l, c)
        df['Engulfing'] = talib.CDLENGULFING(o, h, l, c)
        
        # 2. FILTER FOR HITS
        # We only want to see rows where at least one pattern is NOT 0
        hits = df[(df['HangingMan'] != 0) | (df['ShootingStar'] != 0) | 
                  (df['Hammer'] != 0) | (df['Engulfing'] != 0)].copy()
        
        st.subheader(f"Found {len(hits)} Pattern Occurrences (M5)")
        
        # Show the raw integer codes
        st.write("### 🔢 Raw TA-Lib Integer Codes")
        st.dataframe(hits[['close', 'HangingMan', 'ShootingStar', 'Hammer', 'Engulfing']].style.background_gradient(cmap='RdYlGn', vmin=-100, vmax=100))
        
        # 3. PLOT
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
        
        # Add Markers for each pattern
        # Hanging Man (Expect -100)
        hm = df[df['HangingMan'] != 0]
        fig.add_trace(go.Scatter(
            x=hm.index, y=hm['high'], mode='markers+text', 
            marker=dict(symbol='arrow-down', size=10, color='red'),
            text=hm['HangingMan'], textposition="top center",
            name="Hanging Man"
        ))
        
        # Shooting Star (Expect -100)
        ss = df[df['ShootingStar'] != 0]
        fig.add_trace(go.Scatter(
            x=ss.index, y=ss['high'], mode='markers+text', 
            marker=dict(symbol='arrow-down', size=10, color='orange'),
            text=ss['ShootingStar'], textposition="top center",
            name="Shooting Star"
        ))

        # Hammer (Expect 100)
        ham = df[df['Hammer'] != 0]
        fig.add_trace(go.Scatter(
            x=ham.index, y=ham['low'], mode='markers+text', 
            marker=dict(symbol='arrow-up', size=10, color='green'),
            text=ham['Hammer'], textposition="bottom center",
            name="Hammer"
        ))

        # Engulfing (Expect 100 or -100)
        eng = df[df['Engulfing'] != 0]
        fig.add_trace(go.Scatter(
            x=eng.index, y=eng['high'], mode='markers+text', 
            marker=dict(symbol='circle', size=8, color='blue'),
            text=eng['Engulfing'], textposition="top center",
            name="Engulfing"
        ))

        fig.update_layout(height=800, template="plotly_dark", title=f"Pattern Visualization M5 ({symbol})")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Could not fetch data.")