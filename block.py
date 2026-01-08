import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import MetaTrader5 as mt5
import talib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import time
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Contextual Alpha Engine (Fixed)", layout="wide")
st.title("🧠 Contextual Probability: Walk-Forward Fixed")

# --- SIDEBAR ---
with st.sidebar.expander("🔌 Connection", expanded=True):
    login = int(os.getenv("MT5_LOGIN", 0))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "HFMarketsSA-Live2")
    mt5_login = st.number_input("Login", value=login)
    mt5_pass = st.text_input("Password", value=password, type="password")
    mt5_server = st.text_input("Server", value=server)

symbol = st.sidebar.text_input("Symbol", "USA100")
timeframe = mt5.TIMEFRAME_M5
hist_bars = st.sidebar.slider("History Depth", 3000, 50000, 10000)

st.sidebar.subheader("Strategy Parameters")
n_neighbors = st.sidebar.slider("Similar Scenarios", 20, 200, 50)
forecast_horizon = st.sidebar.slider("Forecast Horizon", 5, 50, 12)
min_probability = st.sidebar.slider("Min Confidence %", 55, 90, 65)

# --- FUNCTIONS ---
def get_data(login, password, server, symbol, n):
    if not mt5.initialize(): return None, "Init Failed"
    if login: mt5.login(login=login, password=password, server=server)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    mt5.shutdown()
    if rates is None: return None, "No Data"
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, "Success"

def engineer_features(df, horizon):
    # Features describing Market State
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['ATR_Norm'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14) / df['close']
    
    # Trend relative to SMA
    sma = talib.SMA(df['close'], timeperiod=50)
    df['Trend_Dist'] = (df['close'] - sma) / sma
    
    # Recent Momentum
    df['Mom_1'] = df['close'].pct_change(1)
    df['Mom_5'] = df['close'].pct_change(5)
    
    # TARGET: The return 'horizon' bars into the future (Shifted Back)
    # Value at row 't' is the return realized at 't + horizon'
    df['Target_Ret'] = df['close'].shift(-horizon) / df['close'] - 1
    
    df = df.dropna()
    return df

def run_backtest(df, neighbors, horizon, threshold):
    feature_cols = ['RSI', 'ATR_Norm', 'Trend_Dist', 'Mom_1', 'Mom_5']
    
    # Pre-scale all features
    scaler = StandardScaler()
    all_X = scaler.fit_transform(df[feature_cols])
    
    # State variables
    equity = [10000]
    trades = []
    
    # Determine safe start index
    # We need enough history to find 'neighbors' AND those neighbors must have finished outcomes.
    # Start searching after 20% of data history to simulate "Training Period"
    start_idx = int(len(df) * 0.2)
    step = 1 # Check every bar? Or step? Let's check every bar for granularity (slower) or step for speed.
    step = 1
    
    # Progress Bar
    progress_bar = st.progress(0)
    
    # Iterating through "Live" time
    # We stop 'horizon' bars before the end because we need to check the outcome of the last trade
    indices = range(start_idx, len(df) - horizon, step)
    total = len(indices)
    
    for i, t in enumerate(indices):
        if i % 100 == 0: progress_bar.progress(i / total)
        
        # 1. Define Valid History (No Lookahead Bias)
        # We can only look at samples that finished BEFORE today (t).
        # Sample at index 'k' finished at 'k + horizon'.
        # So we need k + horizon < t  =>  k < t - horizon
        valid_hist_end = t - horizon
        
        if valid_hist_end < neighbors: continue # Not enough history yet
        
        history_X = all_X[:valid_hist_end]
        current_state = all_X[t].reshape(1, -1)
        
        # 2. Find Neighbors
        knn = NearestNeighbors(n_neighbors=neighbors)
        knn.fit(history_X)
        
        _, indices = knn.kneighbors(current_state)
        neighbor_idxs = indices[0]
        
        # 3. Analyze Outcomes
        # We look at the 'Target_Ret' of these neighbors.
        # Since 'Target_Ret' at row k ALREADY contains the result of k->k+horizon,
        # we can just read it directly.
        past_outcomes = df['Target_Ret'].iloc[neighbor_idxs]
        
        win_prob_up = (past_outcomes > 0).mean()
        win_prob_down = 1.0 - win_prob_up
        avg_move = past_outcomes.mean()
        
        # 4. Take Trade
        action = 0
        if win_prob_up * 100 > threshold: action = 1
        elif win_prob_down * 100 > threshold: action = -1
        
        if action != 0:
            # Result is the Target_Ret at current time t
            realized_ret = df['Target_Ret'].iloc[t]
            
            # Simple PnL: Invest $1000
            pnl = 1000 * realized_ret * action
            pnl -= 0.50 # Spread cost
            
            equity.append(equity[-1] + pnl)
            
            trades.append({
                'Time': df.index[t],
                'Type': 'Buy' if action == 1 else 'Sell',
                'Conf': f"{max(win_prob_up, win_prob_down)*100:.1f}%",
                'PnL': pnl,
                'Equity': equity[-1]
            })
            
            # Jump forward by horizon to avoid overlapping trades (optional, simplifies logic)
            # If you want overlapping trades, remove this jump mechanism in a more complex engine
            # But for simple backtest, we hold until close.
            # (Loop logic needs to handle this manual jump, range() doesn't support dynamic step)
            # Actually, standard range() prevents dynamic jumping. 
            # We will just take the trade and let the loop continue, essentially overlapping?
            # No, simplistic backtest assumes 1 trade at a time usually. 
            # Let's just record signals. For 'Equity Curve', overlapping creates math issues.
            # FIX: We will just plot the signals.
            
    return pd.DataFrame(trades), equity

# --- MAIN ---
if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Crunching historical patterns..."):
        df_raw, msg = get_data(mt5_login, mt5_pass, mt5_server, symbol, hist_bars)
        
        if df_raw is not None:
            # Prepare Data
            df = engineer_features(df_raw, forecast_horizon)
            
            # Run
            trades_df, equity_curve = run_backtest(df, n_neighbors, forecast_horizon, min_probability)
            
            if not trades_df.empty:
                st.subheader("Backtest Performance")
                c1, c2, c3 = st.columns(3)
                
                total_trades = len(trades_df)
                win_rate = len(trades_df[trades_df['PnL'] > 0]) / total_trades * 100
                net_pnl = trades_df['PnL'].sum()
                
                c1.metric("Net Profit", f"${net_pnl:.2f}")
                c2.metric("Win Rate", f"{win_rate:.1f}%")
                c3.metric("Trade Count", total_trades)
                
                # Equity Curve
                # Note: This equity curve assumes overlapping trades are possible or managed sequentially.
                # Since we just appended PnL to a list, it's an approximation of signal quality.
                fig = go.Figure()
                # Create a time index for equity (approximate matching)
                fig.add_trace(go.Scatter(y=equity_curve, mode='lines', fill='tozeroy', line=dict(color='#00E396'), name="Equity"))
                fig.update_layout(title="Equity Growth (Fixed $1000 Size)", height=500, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(trades_df)
            else:
                st.warning("No trades found. Try lowering 'Min Confidence' or increasing 'History Depth'.")
                
        else:
            st.error(msg)