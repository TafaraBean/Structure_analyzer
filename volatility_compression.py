import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
import yfinance as yf

# --- CONFIGURATION ---
st.set_page_config(page_title="Newtonian Zones & Breakouts", layout="wide")

# --- DATA ENGINE ---
def get_data(source, symbol, history_depth, start_time, end_time):
    if source == 'MetaTrader5':
        if not mt5.initialize(): return None, None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, history_depth)
        
        # Fetch Daily for Levels (Need enough history for ATR)
        rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, int(history_depth/40) + 20)
        mt5.shutdown()
        
        if rates is None or rates_d1 is None: return None, None

        df_m5 = pd.DataFrame(rates)
        df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
        df_m5.set_index('time', inplace=True)
        
        df_d1 = pd.DataFrame(rates_d1)
        df_d1['time'] = pd.to_datetime(df_d1['time'], unit='s')
        df_d1.set_index('time', inplace=True)
        
    else: # Yahoo
        yf_sym = "GC=F" if symbol == "XAUUSDm" else symbol
        df_m5 = yf.download(yf_sym, period="60d", interval="5m", progress=False)
        df_d1 = yf.download(yf_sym, period="1y", interval="1d", progress=False)
        
        if isinstance(df_m5.columns, pd.MultiIndex):
            df_m5 = df_m5.xs(df_m5.columns.get_level_values(1)[0], axis=1, level=1)
        if isinstance(df_d1.columns, pd.MultiIndex):
            df_d1 = df_d1.xs(df_d1.columns.get_level_values(1)[0], axis=1, level=1)
        df_m5.columns = [c.lower() for c in df_m5.columns]
        df_d1.columns = [c.lower() for c in df_d1.columns]

    df_m5 = df_m5.between_time(start_time, end_time)
    return df_m5, df_d1

# --- PHYSICS & LEVELS ENGINE ---
def process_data(df_m5, df_d1, atr_period, ci_mult, ma_period):
    # 1. PDH/PDL Levels
    d1 = df_d1.copy()
    d1['tr'] = np.maximum(d1['high'] - d1['low'], 
                          np.maximum(abs(d1['high'] - d1['close'].shift(1)), 
                                     abs(d1['low'] - d1['close'].shift(1))))
    d1['ATR'] = d1['tr'].rolling(atr_period).mean()
    
    # Shift to get Yesterday's data
    d1['Prev_High'] = d1['high'].shift(1)
    d1['Prev_Low'] = d1['low'].shift(1)
    d1['Prev_ATR'] = d1['ATR'].shift(1)
    
    # Merge
    d1['date_key'] = d1.index.date
    df_m5['date_key'] = df_m5.index.date
    merged = df_m5.merge(d1[['date_key', 'Prev_High', 'Prev_Low', 'Prev_ATR']], on='date_key', how='left')
    merged.index = df_m5.index
    merged.dropna(inplace=True)

    # 2. Zones Calculation (The "Bands")
    width = merged['Prev_ATR'] * ci_mult
    merged['PDH_Top'] = merged['Prev_High'] + width
    merged['PDH_Bot'] = merged['Prev_High'] - width
    merged['PDL_Top'] = merged['Prev_Low'] + width
    merged['PDL_Bot'] = merged['Prev_Low'] - width

    # 3. PHYSICS (Kinematics)
    merged['MA'] = merged['close'].ewm(span=ma_period).mean()
    merged['Velocity'] = merged['MA'].diff().rolling(3).mean()
    merged['Acceleration'] = merged['Velocity'].diff().rolling(3).mean()
    
    return merged.dropna()

# --- BACKTEST ENGINE ---
def run_backtest(df, start_bal, trailing_atr_mult, trade_breakouts, trade_reversals):
    balance = start_bal
    equity = [start_bal]
    trades = []
    
    pos = 0 # 1=Long, -1=Short
    entry_price = 0
    sl = 0
    highest_price = 0
    lowest_price = 0
    lot_value_per_point = 1 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        
        # --- EXIT LOGIC (Trailing) ---
        if pos == 1:
            if row['high'] > highest_price: highest_price = row['high']
            trail_sl = highest_price - (row['Prev_ATR'] * trailing_atr_mult)
            if trail_sl > sl: sl = trail_sl
            
            if row['low'] <= sl: 
                pnl = (sl - entry_price) * lot_value_per_point * 100 
                balance += pnl
                trades.append({'Exit Time': row.name, 'Type': 'Long', 'PnL': pnl, 'Reason': 'Trail SL'})
                pos = 0
                
        elif pos == -1:
            if row['low'] < lowest_price: lowest_price = row['low']
            trail_sl = lowest_price + (row['Prev_ATR'] * trailing_atr_mult)
            if trail_sl < sl: sl = trail_sl
            
            if row['high'] >= sl:
                pnl = (entry_price - sl) * lot_value_per_point * 100
                balance += pnl
                trades.append({'Exit Time': row.name, 'Type': 'Short', 'PnL': pnl, 'Reason': 'Trail SL'})
                pos = 0

        # --- ENTRY LOGIC (Physics + Zones) ---
        if pos == 0:
            if trade_breakouts:
                # Long Breakout: Close > Top of PDH Zone + Positive Velocity + Positive Acceleration
                if (row['close'] > row['PDH_Top']) and (row['Velocity'] > 0) and (row['Acceleration'] > 0):
                    pos = 1
                    entry_price = row['close']
                    sl = row['PDH_Bot'] 
                    highest_price = entry_price
                
                # Short Breakout: Close < Bottom of PDL Zone + Negative Velocity + Negative Acceleration
                elif (row['close'] < row['PDL_Bot']) and (row['Velocity'] < 0) and (row['Acceleration'] < 0):
                    pos = -1
                    entry_price = row['close']
                    sl = row['PDL_Top']
                    lowest_price = entry_price

            if trade_reversals and pos == 0:
                # Long Reversal: Inside PDL Zone + Decelerating Drop (Acc > 0)
                if (row['PDL_Bot'] < row['close'] < row['PDL_Top']) and (row['Velocity'] < 0) and (row['Acceleration'] > 0):
                    pos = 1
                    entry_price = row['close']
                    sl = row['PDL_Bot'] - (row['Prev_ATR'] * 0.15)
                    highest_price = entry_price

                # Short Reversal: Inside PDH Zone + Decelerating Rally (Acc < 0)
                elif (row['PDH_Bot'] < row['close'] < row['PDH_Top']) and (row['Velocity'] > 0) and (row['Acceleration'] < 0):
                    pos = -1
                    entry_price = row['close']
                    sl = row['PDH_Top'] + (row['Prev_ATR'] * 0.15)
                    lowest_price = entry_price
        
        equity.append(balance)
        
    return pd.DataFrame(trades), equity

# --- DASHBOARD ---
st.title("⚛️ Physics-Based Volatility Bands")

with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Source", ["MetaTrader5", "Yahoo Finance"])
    symbol = st.text_input("Symbol", "XAUUSDm")
    history_len = st.slider("History Depth", 1000, 10000, 5000)
    
    st.markdown("---")
    st.subheader("Physics & Bands")
    ci_mult = st.slider("Band Thickness (x ATR)", 0.05, 0.4, 0.15, help="Wider bands = Fewer, safer trades.")
    trail_mult = st.slider("Trailing Stop (x ATR)", 1.0, 5.0, 2.5, help="Hold trades longer.")
    
    st.subheader("Strategy Mode")
    do_break = st.checkbox("Trade Breakouts", True)
    do_rev = st.checkbox("Trade Reversals", True)

if st.button("Run Analysis"):
    with st.spinner("Calculating Physics..."):
        df_m5, df_d1 = get_data(source, symbol, history_len, "01:00", "23:00")
        
        if df_m5 is not None:
            data = process_data(df_m5, df_d1, 14, ci_mult, 20)
            trades, eq = run_backtest(data, 10000, trail_mult, do_break, do_rev)
            
            # --- GAPLESS VISUALIZATION ---
            data['date_str'] = data.index.strftime('%Y-%m-%d %H:%M')
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
            
            # 1. Price Candles
            fig.add_trace(go.Candlestick(x=data['date_str'], open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Price"), row=1, col=1)

            # 2. PDH VOLATILITY BAND (RED)
            # Top Line (Invisible)
            fig.add_trace(go.Scatter(
                x=data['date_str'], y=data['PDH_Top'],
                line=dict(color='rgba(255,0,0,0)'), showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            # Bottom Line (Filled to Top)
            fig.add_trace(go.Scatter(
                x=data['date_str'], y=data['PDH_Bot'],
                fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', # RED SHADE
                line=dict(color='rgba(255,0,0,0)'), name="PDH Volatility Band", hoverinfo='skip'
            ), row=1, col=1)
            # Center Line
            fig.add_trace(go.Scatter(x=data['date_str'], y=data['Prev_High'], line=dict(color='red', width=1, dash='dash'), name="Prev High"), row=1, col=1)

            # 3. PDL VOLATILITY BAND (GREEN)
            # Top Line (Invisible)
            fig.add_trace(go.Scatter(
                x=data['date_str'], y=data['PDL_Top'],
                line=dict(color='rgba(0,255,0,0)'), showlegend=False, hoverinfo='skip'
            ), row=1, col=1)
            # Bottom Line (Filled to Top)
            fig.add_trace(go.Scatter(
                x=data['date_str'], y=data['PDL_Bot'],
                fill='tonexty', fillcolor='rgba(0, 255, 0, 0.2)', # GREEN SHADE
                line=dict(color='rgba(0,255,0,0)'), name="PDL Volatility Band", hoverinfo='skip'
            ), row=1, col=1)
            # Center Line
            fig.add_trace(go.Scatter(x=data['date_str'], y=data['Prev_Low'], line=dict(color='green', width=1, dash='dash'), name="Prev Low"), row=1, col=1)

            # 4. Velocity Pane
            colors = ['#00ff00' if v > 0 else '#ff0000' for v in data['Velocity']]
            fig.add_trace(go.Bar(x=data['date_str'], y=data['Velocity'], marker_color=colors, name="Velocity"), row=2, col=1)
            
            # Layout
            fig.update_layout(
                title=f"{symbol} | Physics-Based Volatility Bands",
                height=800,
                xaxis_rangeslider_visible=False,
                xaxis_type='category', # GAPLESS
                template="plotly_dark",
                showlegend=True,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            fig.update_xaxes(tickmode='auto', nticks=15)
            
            # --- DISPLAY ---
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.metric("Total Return", f"${eq[-1]-10000:.2f}", f"{((eq[-1]-10000)/10000)*100:.1f}%")
                st.metric("Trades", len(trades))
                
                # Simple Win Rate
                if len(trades) > 0:
                    wr = (len(trades[trades['PnL']>0]) / len(trades)) * 100
                    st.metric("Win Rate", f"{wr:.1f}%")
                    
                    st.write("### Equity Curve")
                    st.line_chart(eq)
                    
                    with st.expander("Trade List"):
                        st.dataframe(trades)
        else:
            st.error("Data Fetch Failed")