import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import MetaTrader5 as mt5
from scipy.optimize import minimize
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Live GARCH Trainer", layout="wide")
st.title("🧪 Live GARCH-X Volatility Lab")
st.markdown("""
**Objective:** Train and Visualize GARCH models on live data.
**Workflow:** 1. Fetch Data -> 2. Train Model -> 3. Visualize Predicted Volatility vs. Realized Squeeze.
""")

# --- SIDEBAR ---
with st.sidebar.expander("Credentials", expanded=True):
    login = int(os.getenv("MT5_LOGIN", 0))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "HFMarketsSA-Live2")
    
    mt5_login = st.number_input("Login", value=login)
    mt5_pass = st.text_input("Password", value=password, type="password")
    mt5_server = st.text_input("Server", value=server)

symbol = st.sidebar.text_input("Symbol", "USA100")
timeframe = mt5.TIMEFRAME_M5
lookback = st.sidebar.slider("Training Sample Size", 3000, 50000, 10000, help="More data = Slower training but more stable parameters.")

st.sidebar.markdown("---")
st.sidebar.subheader("Plotting Settings")
plot_window = st.sidebar.slider("Zoom Last N Bars", 100, 2000, 500)

# --- MATH FUNCTIONS (MLE) ---
def garch_likelihood(params, returns, exog):
    mu = params[0]
    omega = params[1]
    alpha = params[2]
    beta = params[3]
    gamma = params[4:]
    
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10
    
    T = len(returns)
    sigma2 = np.zeros(T)
    epsilon = returns - mu
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        exog_effect = np.dot(exog[t], gamma)
        val = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1] + exog_effect
        sigma2[t] = max(val, 1e-6)
        
    loglik = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + (epsilon**2) / sigma2)
    return -loglik

def get_data(login, password, server, symbol, n):
    if not mt5.initialize(): return None, f"Init Failed"
    if login: mt5.login(login=login, password=password, server=server)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    mt5.shutdown()
    
    if rates is None: return None, "No Data"
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df, "Success"

def train_and_predict(df):
    # 1. Feature Engineering
    df['return'] = 100 * np.log(df['close'] / df['close'].shift(1))
    df['lag5_return_abs'] = df['return'].shift(5).abs().fillna(0)
    df['time_feature'] = (df.index.hour + df.index.minute/60.0) / 24.0
    
    df_clean = df.dropna()
    returns = df_clean['return'].values
    exog = df_clean[['lag5_return_abs', 'time_feature']].values
    
    # 2. Optimization
    initial_var = np.var(returns)
    initial_guess = [0.0, initial_var * 0.05, 0.1, 0.85, 0.0, 0.0]
    
    bounds = [(None,None), (1e-6,None), (0.01,0.99), (0.01,0.99), (None,None), (None,None)]
    
    res = minimize(garch_likelihood, initial_guess, args=(returns, exog),
                   method='L-BFGS-B', bounds=bounds)
    
    params = res.x
    
    # 3. Generate Full Series (In-Sample Prediction)
    T = len(returns)
    sigma2 = np.zeros(T)
    epsilon = returns - params[0]
    sigma2[0] = np.var(returns)
    
    # Unpack for loop
    omega, alpha, beta = params[1], params[2], params[3]
    gamma = params[4:]
    
    for t in range(1, T):
        exog_effect = np.dot(exog[t], gamma)
        val = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1] + exog_effect
        sigma2[t] = max(val, 1e-6)
        
    df_clean['GARCH_Vol'] = np.sqrt(sigma2)
    df_clean['Realized_Vol'] = df_clean['return'].rolling(14).std()
    
    return df_clean, params

# --- MAIN ---
if st.sidebar.button("🚀 Train New Model"):
    with st.spinner("Fetching Data & Optimizing Parameters..."):
        df_raw, msg = get_data(mt5_login, mt5_pass, mt5_server, symbol, lookback)
        
        if df_raw is not None:
            # Store in session state to persist after interaction
            df_res, params = train_and_predict(df_raw)
            st.session_state['garch_data'] = df_res
            st.session_state['garch_params'] = params
            st.success("Model Trained Successfully!")
        else:
            st.error(msg)

# --- VISUALIZATION ---
if 'garch_data' in st.session_state:
    df = st.session_state['garch_data']
    params = st.session_state['garch_params']
    
    # Slice for plotting
    plot_df = df.iloc[-plot_window:].copy()
    
    # Logic: Divergence
    plot_df['Div'] = plot_df['GARCH_Vol'] - plot_df['Realized_Vol']
    
    # --- METRICS PANEL ---
    st.markdown("### 🎛️ Model Diagnostics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Alpha (Shock)", f"{params[2]:.4f}")
    c2.metric("Beta (Memory)", f"{params[3]:.4f}")
    c3.metric("Gamma (Shift)", f"{params[4]:.4f}", help="Impact of Lag-5 Return")
    c4.metric("Gamma (Time)", f"{params[5]:.4f}", help="Impact of Time of Day")
    c5.metric("Current Vol", f"{plot_df['GARCH_Vol'].iloc[-1]:.2f}")

    # --- CHARTS ---
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, 
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price Action & Divergence Signal", "Volatility Battle (Predicted vs Realized)", "Model Fit (Returns vs Envelope)")
    )
    
    # 1. Price
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], 
                                 low=plot_df['low'], close=plot_df['close'], name="Price"), row=1, col=1)
    
    # Signals (Where GARCH >> Realized)
    # We define a "Significant Divergence" as GARCH being 20% higher than Realized
    sig_mask = plot_df['GARCH_Vol'] > (plot_df['Realized_Vol'] * 1.2)
    sig_df = plot_df[sig_mask]
    
    fig.add_trace(go.Scatter(x=sig_df.index, y=sig_df['low'], mode='markers',
                             marker=dict(symbol='diamond', size=5, color='yellow'),
                             name="Vol Expansion Signal"), row=1, col=1)

    # 2. Volatility Comparison
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['GARCH_Vol'], 
                             line=dict(color='#FF0055', width=2), name="GARCH (Predicted)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Realized_Vol'], 
                             line=dict(color='#00B5F0', width=1.5), name="Realized (Actual)"), row=2, col=1)
    
    # Fill area where GARCH > Realized
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['GARCH_Vol'], fill='tonexty', 
                             fillcolor='rgba(255, 0, 85, 0.1)', line=dict(width=0), 
                             showlegend=False, hoverinfo='skip'), row=2, col=1)

    # 3. Model Fit (The "GARCH Envelope")
    # Plot Returns
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['return'], 
                             line=dict(color='gray', width=1), opacity=0.5, name="Returns"), row=3, col=1)
    # Plot +Sigma
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['GARCH_Vol'], 
                             line=dict(color='orange', width=1), name="+1 Sigma"), row=3, col=1)
    # Plot -Sigma
    fig.add_trace(go.Scatter(x=plot_df.index, y=-plot_df['GARCH_Vol'], 
                             line=dict(color='orange', width=1), name="-1 Sigma", showlegend=False), row=3, col=1)

    fig.update_layout(height=900, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    * **Middle Panel:** The **Red Line** (GARCH) is your "Leading Indicator". When it spikes *before* the Blue Line (Realized), it predicts turbulence.
    * **Bottom Panel:** This checks if the model is "sane". The returns (grey noise) should mostly stay inside the Orange Envelope. If returns constantly break the envelope, the model is underestimating risk.
    """)