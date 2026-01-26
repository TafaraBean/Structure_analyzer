import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import talib
import os
import matplotlib.ticker as mticker
import talib
import os
from dotenv import load_dotenv

# --- Backtest Class ---
class Backtester:
    def __init__(self, data, signals, lot_size, rr, atr_period, sl_atr_mult, tp_atr_mult, trail_atr_dist, be_atr_trigger):
        self.df = data.copy()
        self.signals = signals # Dict with 'bull' and 'bear' boolean series
        self.lot_size = lot_size
        self.rr = rr
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.trail_atr_dist = trail_atr_dist # Trailing Stop Distance in ATR
        self.be_atr_trigger = be_atr_trigger # Breakeven Trigger Distance in ATR
        
        # Prepare Data
        self.df['atr'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.atr_period)
        self.df['bull_signal'] = self.signals['bull']
        self.df['bear_signal'] = self.signals['bear']
        
    def run(self):
        balance = 10000 # Starting Balance
        equity = []
        trades = []
        active_trade = None # {'type': 'buy'/'sell', 'entry': price, 'sl': price, 'tp': price, 'size': lots}
        
        # Loop through bars
        # Note: Iterating rows is slow in pandas, but for <10k bars it's acceptable for Streamlit
        # We start after ATR period
        
        for i in range(self.atr_period, len(self.df)):
            curr_bar = self.df.iloc[i]
            prev_bar = self.df.iloc[i-1]
            
            # 1. Manage Active Trade
            if active_trade:
                # Check for Exit (Hit SL or TP)
                exit_price = None
                pnl = 0
                
                if active_trade['type'] == 'buy':
                    # Check Low for SL
                    if curr_bar['low'] <= active_trade['sl']:
                        exit_price = active_trade['sl']
                    # Check High for TP
                    elif curr_bar['high'] >= active_trade['tp']:
                        exit_price = active_trade['tp']
                    
                    if exit_price:
                        pnl = (exit_price - active_trade['entry']) * active_trade['size'] * 100000 # Standard Lot approx (simplify conversion?)
                        # Assuming Forex Standard Lot (100k units). For Gold it's different. 
                        # Simplifying: PnL = PriceDiff * Volume * ContractSize. 
                        # Let's assume generic PnL = PriceDiff * LotSize * 1. 
                        # Use a multiplier? 
                        # Let's just use raw price diff * lot size for now, user can adjust logic if needed.
                        # Wait, user usually thinks in dollars. 
                        # Let's stick to PriceDiff * LotSize.
                        pnl = (exit_price - active_trade['entry']) * self.lot_size 
                        active_trade = None
                        
                    else:
                        # Manage Trade (Trailing / BE)
                        # Breakeven
                        if self.be_atr_trigger > 0:
                            if curr_bar['high'] >= active_trade['entry'] + (curr_bar['atr'] * self.be_atr_trigger):
                                active_trade['sl'] = max(active_trade['sl'], active_trade['entry'])
                        
                        # Trailing
                        if self.trail_atr_dist > 0:
                            new_sl = curr_bar['close'] - (curr_bar['atr'] * self.trail_atr_dist)
                            active_trade['sl'] = max(active_trade['sl'], new_sl)

                elif active_trade['type'] == 'sell':
                    # Check High for SL
                    if curr_bar['high'] >= active_trade['sl']:
                        exit_price = active_trade['sl']
                    # Check Low for TP
                    elif curr_bar['low'] <= active_trade['tp']:
                        exit_price = active_trade['tp']
                        
                    if exit_price:
                        pnl = (active_trade['entry'] - exit_price) * self.lot_size
                        active_trade = None
                    else:
                        # Manage Trade
                        # Breakeven
                        if self.be_atr_trigger > 0:
                            if curr_bar['low'] <= active_trade['entry'] - (curr_bar['atr'] * self.be_atr_trigger):
                                active_trade['sl'] = min(active_trade['sl'], active_trade['entry'])
                                
                        # Trailing
                        if self.trail_atr_dist > 0:
                            new_sl = curr_bar['close'] + (curr_bar['atr'] * self.trail_atr_dist)
                            active_trade['sl'] = min(active_trade['sl'], new_sl)
            
                if pnl != 0:
                    balance += pnl
                    # Store trade result?
            
            # 2. Check for New Entry (if no active trade)
            if active_trade is None:
                # We use signals from Previous Closed Candle, executed at Current Open
                # The 'bull_signal' is aligned to the candle that generated it. 
                # So if df.iloc[i-1]['bull_signal'] is True, we enter at df.iloc[i]['open']
                
                if prev_bar['bull_signal']:
                    atr = prev_bar['atr']
                    sl_dist = atr * self.sl_atr_mult
                    risk = sl_dist
                    reward = risk * self.rr
                    
                    entry = curr_bar['open']
                    sl = entry - risk
                    tp = entry + reward
                    
                    active_trade = {
                        'type': 'buy',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'size': self.lot_size
                    }
                    
                elif prev_bar['bear_signal']:
                    atr = prev_bar['atr']
                    sl_dist = atr * self.sl_atr_mult
                    risk = sl_dist
                    reward = risk * self.rr
                    
                    entry = curr_bar['open']
                    sl = entry + risk
                    tp = entry - reward
                    
                    active_trade = {
                        'type': 'sell',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'size': self.lot_size
                    }
                    
            equity.append(balance)
            
        return pd.Series(equity, index=self.df.index[self.atr_period:])

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Market Regime Dashboard")

# --- Constants & Config ---
# Updated Timeframes (M15 Base)
TIMEFRAMES = {
    'M15': {'tf': mt5.TIMEFRAME_M15, 'delta': pd.Timedelta(minutes=15)},
    'M30': {'tf': mt5.TIMEFRAME_M30, 'delta': pd.Timedelta(minutes=30)},
    'H1':  {'tf': mt5.TIMEFRAME_H1,  'delta': pd.Timedelta(hours=1)},
    'H4':  {'tf': mt5.TIMEFRAME_H4,  'delta': pd.Timedelta(hours=4)},
    'D1':  {'tf': mt5.TIMEFRAME_D1,  'delta': pd.Timedelta(days=1)},
    'W1':  {'tf': mt5.TIMEFRAME_W1,  'delta': pd.Timedelta(weeks=1)},
}

# --- Functions ---

@st.cache_resource
def init_mt5():
    """Initialize MT5 connection (Cached Resource to avoid re-connecting)."""
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

@st.cache_data(ttl=60) # Cache data for 60 seconds
def fetch_tf_data(symbol, tf_constant, bars):
    rates = mt5.copy_rates_from_pos(symbol, tf_constant, 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_vwma(close, volume, period):
    return (close * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()

def detect_engulfing(df):
    """
    Returns boolean Series for Bullish and Bearish Engulfing.
    Bullish Engulfing: 
    1. Prev Candle Red (Close < Open)
    2. Curr Candle Green (Close > Open)
    3. Curr Body engulfs Prev Body (Open < Prev Close AND Close > Prev Open) - Strict body engulfing
    """
    o = df['open']
    c = df['close']
    h = df['high']
    l = df['low']
    
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    
    # Bullish Engulfing
    # Prev Red: prev_c < prev_o
    # Curr Green: c > o
    # Engulfs Body: c > prev_o AND o < prev_c
    bull_engulf = (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)
    
    # Bearish Engulfing
    # Prev Green: prev_c > prev_o
    # Curr Red: c < o
    # Engulfs Body: c < prev_o AND o > prev_c
    bear_engulf = (prev_c > prev_o) & (c < o) & (c < prev_o) & (o > prev_c)
    
    return bull_engulf, bear_engulf

def permutation_entropy(series, order=3, delay=1, window_size=30):
    """Vectorized Permutation Entropy."""
    vals = series.values
    n = len(vals)
    result = np.full(n, np.nan)
    denom = np.log2(math.factorial(order))
    
    # We use a loop for the rolling window as before
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

def calculate_metrics(df_master, symbol, bars_master, ma_period, pe_order, pe_delay, pe_window, selected_tfs):
    """Core Logic: Calculate Heatmaps efficiently based on selection."""
    
    heatmap_entropy = pd.DataFrame(index=df_master.index)
    heatmap_adx = pd.DataFrame(index=df_master.index) 
    heatmap_adx_entropy = pd.DataFrame(index=df_master.index)
    
    progress_bar = st.progress(0)
    
    # Filter config based on selection
    active_tfs = {k:v for k,v in TIMEFRAMES.items() if k in selected_tfs}
    tf_keys = list(active_tfs.keys())
    
    if not tf_keys:
        return heatmap_entropy, heatmap_adx, heatmap_adx_entropy

    for idx, (name, tf_info) in enumerate(active_tfs.items()):
        # Fetch appropriate amount of data
        # H4/D1 need fewer bars than M15, but fetching 1000 is safe buffer
        df_tf = fetch_tf_data(symbol, tf_info['tf'], 1000)
        
        if df_tf is None: continue
        
        # 1. EMA & Entropy
        ema = calculate_ema(df_tf['close'], ma_period)
        pe = permutation_entropy(ema, order=pe_order, delay=pe_delay, window_size=pe_window)
        
        # 2. ADX/DI
        adx = talib.ADX(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        
        direction = np.where(plus_di >= minus_di, 1, -1)
        adx_score = adx * direction
        adx_series = pd.Series(adx_score, index=df_tf.index)
        
        # 3. ADX Entropy
        pe_adx = permutation_entropy(pd.Series(adx, index=df_tf.index), order=pe_order, delay=pe_delay, window_size=pe_window)
        
        # 4. Sync Strategy (Strict Lookahead Prevention)
        def sync_to_master(series):
            s = series.copy()
            s.index = s.index + tf_info['delta'] # Shift to available time
            return s.reindex(df_master.index, method='ffill')
            
        heatmap_entropy[name] = sync_to_master(pe)
        heatmap_adx[name] = sync_to_master(adx_series)
        heatmap_adx_entropy[name] = sync_to_master(pe_adx)
        
        progress_bar.progress((idx + 1) / len(tf_keys))
        
    return heatmap_entropy, heatmap_adx, heatmap_adx_entropy

# --- Sidebar ---
st.sidebar.title("Configuration")
SYMBOL = st.sidebar.text_input("Symbol", "EURUSDm")
# Updated default Bars to reflect M15 context (2000 M15 bars is huge history, ~20 days)
BARS_MASTER = st.sidebar.number_input("Bars (Master)", min_value=500, max_value=20000, value=5000)
# Default Zoom set to full to satisfy "always display full history"
ZOOM_LAST = st.sidebar.number_input("Zoom Last N Bars", min_value=100, max_value=BARS_MASTER, value=BARS_MASTER)

st.sidebar.subheader("Timeframes")
all_tfs = list(TIMEFRAMES.keys())
# Default to all
SELECTED_TFS = st.sidebar.multiselect("Include Timeframes", all_tfs, default=all_tfs)

st.sidebar.subheader("Parameters")
MA_PERIOD = st.sidebar.slider("MA Period", 10, 200, 20)
PE_WINDOW = st.sidebar.slider("Entropy Window", 10, 100, 30)
PE_ORDER = st.sidebar.slider("Entropy Order", 3, 5, 4)

st.sidebar.subheader("VWMA Settings")
VWMA_SHORT = st.sidebar.number_input("VWMA Short Period", min_value=5, max_value=200, value=20)
VWMA_LONG = st.sidebar.number_input("VWMA Long Period", min_value=10, max_value=500, value=50)

st.sidebar.subheader("Thresholds")
ENTROPY_THRESHOLD = st.sidebar.slider("Entropy Threshold (Stable)", 0.1, 1.0, 0.2, 0.05)
ADX_THRESHOLD = st.sidebar.slider("ADX Threshold (Strong)", 10, 50, 30, 5)
# Adjust max confluence dynamically
max_conf = len(SELECTED_TFS) if SELECTED_TFS else 1
CONFLUENCE_MIN = st.sidebar.slider("Min Confluence TFs", 1, max_conf, min(4, max_conf))

st.sidebar.subheader("Backtest Settings")
LOT_SIZE = st.sidebar.number_input("Lot Size", 0.01, 100.0, 1.0, 0.01)
RISK_REWARD = st.sidebar.number_input("Risk:Reward Ratio", 0.1, 10.0, 2.0, 0.1)
ATR_PERIOD = st.sidebar.number_input("ATR Period", 1, 50, 14)
SL_ATR_MULT = st.sidebar.number_input("SL ATR Multiplier", 0.1, 10.0, 1.5, 0.1)
TRAIL_ATR = st.sidebar.number_input("Trailing ATR Dist (0=Off)", 0.0, 10.0, 0.0, 0.1)
BE_ATR = st.sidebar.number_input("Breakeven ATR Trigger (0=Off)", 0.0, 10.0, 0.0, 0.1)

st.sidebar.subheader("Strategy Filters")
USE_STABILITY_FILTER = st.sidebar.checkbox("Filter by Stability (Entropy)", value=False)
USE_TREND_FILTER = st.sidebar.checkbox("Filter by Trend (ADX)", value=False)

st.sidebar.subheader("Playback")
PLAYBACK_MODE = st.sidebar.checkbox("Enable Playback Mode")

if st.sidebar.button("Run Analysis"):
    st_run = True
else:
    st_run = True # Auto run on load usually preferred, but button is good for reset

# --- Main Page ---
if st_run:
    if not init_mt5():
        st.stop()
        
    status_text = st.empty()
    status_text.text(f"Fetching {BARS_MASTER} bars for {SYMBOL} (M15 Master)...")
    
    # Fetch Master as M15 now
    df_master = fetch_tf_data(SYMBOL, mt5.TIMEFRAME_M15, BARS_MASTER)
    
    if df_master is None:
        st.error("Failed to fetch Master Data. Check Symbol/Connection.")
        st.stop()
        
    if not SELECTED_TFS:
        st.warning("Please select at least one timeframe.")
        st.stop()
        
    # --- Playback Logic ---
    if PLAYBACK_MODE:
        total_bars = len(df_master)
        # Default to max (Latest)
        if ZOOM_LAST < total_bars:
            playback_idx = st.sidebar.slider("Playback Position (End Index)", 
                                             min_value=ZOOM_LAST, 
                                             max_value=total_bars, 
                                             value=total_bars)
        else:
            playback_idx = total_bars
            st.sidebar.info(f"Playback disabled: Zoom ({ZOOM_LAST}) >= Total Bars ({total_bars})")
    else:
        playback_idx = len(df_master)
        
    status_text.text("Calculating Multi-Timeframe Matrix...")
    
    hm_ent, hm_adx, hm_adx_ent = calculate_metrics(
        df_master, SYMBOL, BARS_MASTER, MA_PERIOD, PE_ORDER, 1, PE_WINDOW, SELECTED_TFS
    )
    
    status_text.text("Running Backtest on Full History...")
    
    # --- Metrics & Signals on FULL Data ---
    # We must calculate signals on the full range for the backtest to be accurate and consistent
    # regardless of the zoom level.
    
    # 1. Reindex full heatmaps to ordered TFs
    full_order = ['W1', 'D1', 'H4', 'H1', 'M30', 'M15']
    ordered_tfs = [tf for tf in full_order if tf in SELECTED_TFS]
    
    hm_ent_full = hm_ent.T.reindex(ordered_tfs)
    hm_adx_full = hm_adx.T.reindex(ordered_tfs)
    hm_adx_ent_full = hm_adx_ent.T.reindex(ordered_tfs)
    
    # 2. Confluence on Full Data
    stable_mask_full = hm_ent_full < ENTROPY_THRESHOLD
    stable_confluence_full = (stable_mask_full.sum(axis=0) >= CONFLUENCE_MIN)
    
    bull_mask_full = hm_adx_full > ADX_THRESHOLD
    bull_confluence_full = (bull_mask_full.sum(axis=0) >= CONFLUENCE_MIN)
    
    bear_mask_full = hm_adx_full < -ADX_THRESHOLD
    bear_confluence_full = (bear_mask_full.sum(axis=0) >= CONFLUENCE_MIN)
    
    # 3. VWMA Crossovers on Full Data
    # Calculate VWMAs
    vwma_short_full = calculate_vwma(df_master['close'], df_master['tick_volume'], VWMA_SHORT)
    vwma_long_full = calculate_vwma(df_master['close'], df_master['tick_volume'], VWMA_LONG)
    
    # Identify Crossovers
    # Bullish: Short crosses ABOVE Long
    crossover_bull = (vwma_short_full > vwma_long_full) & (vwma_short_full.shift(1) <= vwma_long_full.shift(1))
    
    # Bearish: Short crosses BELOW Long
    crossover_bear = (vwma_short_full < vwma_long_full) & (vwma_short_full.shift(1) >= vwma_long_full.shift(1))
    
    # Filter by Stable Confluence (Optional)
    final_bull = crossover_bull
    final_bear = crossover_bear
    
    if USE_STABILITY_FILTER:
        final_bull = final_bull & stable_confluence_full
        final_bear = final_bear & stable_confluence_full
        
    if USE_TREND_FILTER:
        # Bullish: ADX Bullish Confluence
        final_bull = final_bull & bull_confluence_full
        # Bearish: ADX Bearish Confluence
        final_bear = final_bear & bear_confluence_full
    
    # --- Backtest Run ---
    bt_engine = Backtester(
        df_master, 
        {'bull': final_bull, 'bear': final_bear},
        LOT_SIZE, RISK_REWARD, ATR_PERIOD, SL_ATR_MULT, 
        RISK_REWARD * SL_ATR_MULT, # TP mult
        TRAIL_ATR, BE_ATR
    )
    equity_curve = bt_engine.run()
    
    # Metrics
    if len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        if returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * (252**0.5)
        else:
            sharpe = 0
        final_eq = equity_curve.iloc[-1]
        total_ret = ((final_eq - 10000) / 10000) * 100
        dd_series = equity_curve / equity_curve.cummax() - 1
        max_dd = dd_series.min() * 100
    else:
        sharpe = 0; total_ret = 0; max_dd = 0

    status_text.text("Rendering Visualization...")
    
    # --- Slicing for Plotting ---
    # Crop based on Playback Index
    end_idx = playback_idx
    start_idx = max(0, end_idx - ZOOM_LAST)
    
    df_zoom = df_master.iloc[start_idx : end_idx]
    
    # Slice Maps
    hm_ent_zoom = hm_ent_full.iloc[:, start_idx : end_idx]
    hm_adx_zoom = hm_adx_full.iloc[:, start_idx : end_idx]
    hm_adx_ent_zoom = hm_adx_ent_full.iloc[:, start_idx : end_idx]
    
    # Slice Confluence
    stable_confluence = stable_confluence_full.iloc[start_idx : end_idx]
    bull_confluence = bull_confluence_full.iloc[start_idx : end_idx]
    bear_confluence = bear_confluence_full.iloc[start_idx : end_idx]
    
    # Dynamic Sort (Already done above, skipping redundant block but keeping variable names clean)
    # just re-using hm_ent_zoom which is already sorted
    
    # --- Matplotlib Chart ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.15)
    
    # Prepare data for gapless plotting (Integer X-Axis)
    df_plot = df_zoom.reset_index()
    df_plot['x_idx'] = df_plot.index
    
    # Panel 1: Price
    ax1 = fig.add_subplot(gs[0])
    
    width = 0.6
    width2 = 0.05
    
    up = df_plot[df_plot.close >= df_plot.open]
    down = df_plot[df_plot.close < df_plot.open]
    col_up = '#089981'; col_down = '#f23645'
    
    ax1.bar(up.x_idx, up.close-up.open, width, bottom=up.open, color=col_up)
    ax1.bar(up.x_idx, up.high-up.low, width2, bottom=up.low, color=col_up)
    ax1.bar(down.x_idx, down.open-down.close, width, bottom=down.close, color=col_down)
    ax1.bar(down.x_idx, down.high-down.low, width2, bottom=down.low, color=col_down)
    
    # Highlights
    d_min = df_plot['low'].min(); d_max = df_plot['high'].max()
    
    ax1.fill_between(df_plot.x_idx, d_min, d_max, where=stable_confluence.values, color='cyan', alpha=0.1, label='Stability Confluence')
    ax1.fill_between(df_plot.x_idx, d_min, d_max, where=bull_confluence.values, color='green', alpha=0.15, label='Bull Confluence')
    ax1.fill_between(df_plot.x_idx, d_min, d_max, where=bear_confluence.values, color='red', alpha=0.15, label='Bear Confluence')
    
    ax1.set_title(f"{SYMBOL} Dashboard | Stability & Direction | Sharpe: {sharpe:.2f} | Return: {total_ret:.2f}% | MaxDD: {max_dd:.2f}%", fontsize=16, color='white')
    ax1.set_ylabel("Price (M15)")
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.3)
    
    # --- VWMA Plotting ---
    # Retrieve pre-calculated VWMAs and slice
    vwma_short_plot = vwma_short_full.iloc[start_idx : end_idx]
    vwma_long_plot = vwma_long_full.iloc[start_idx : end_idx]
    
    # Plot using x_idx
    ax1.plot(df_plot.x_idx, vwma_short_plot.values, color='orange', linewidth=1.5, label=f'VWMA {VWMA_SHORT}')
    ax1.plot(df_plot.x_idx, vwma_long_plot.values, color='purple', linewidth=1.5, label=f'VWMA {VWMA_LONG}')
    
    # --- Crossover Highlights in Stable Zones ---
    # Retrieve pre-calculated signals and slice
    # Note: final_bull was calculated in the Full Data section
    plot_bull = final_bull.iloc[start_idx : end_idx]
    plot_bear = final_bear.iloc[start_idx : end_idx]
    
    # Get indices for scatter plot
    bull_indices = df_plot.loc[plot_bull.values, 'x_idx']
    bull_prices = df_plot.loc[plot_bull.values, 'low'] * 0.9995 
    
    bear_indices = df_plot.loc[plot_bear.values, 'x_idx']
    bear_prices = df_plot.loc[plot_bear.values, 'high'] * 1.0005 
    
    # Plot Markers
    ax1.scatter(bull_indices, bull_prices, marker='^', color='lime', s=100, zorder=5, label='Bull Signal')
    ax1.scatter(bear_indices, bear_prices, marker='v', color='magenta', s=100, zorder=5, label='Bear Signal')
    
    # Re-add legend
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.3)

    # Custom Formatter for X-axis    
    # Custom Formatter for X-axis
    def format_date(x, pos=None):
        idx = int(x + 0.5)
        if 0 <= idx < len(df_plot):
            return df_plot['time'].iloc[idx].strftime('%m-%d %H:%M')
        return ''
        
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(format_date))
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

    # Heatmap Helpers
    # Y-Ticks need to match the ordered_tfs
    y_vals = np.arange(len(ordered_tfs))
    x_nums = df_plot['x_idx'].values
    
    # Panel 2: Entropy
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    c2 = ax2.pcolormesh(x_nums, y_vals, hm_ent_zoom.values, cmap='RdYlGn_r', shading='nearest', vmin=0, vmax=1)
    ax2.set_yticks(y_vals); ax2.set_yticklabels(ordered_tfs)
    ax2.set_ylabel("Stability (PE)")
    
    # Panel 3: ADX Direction
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    norm = mcolors.Normalize(vmin=-50, vmax=50)
    c3 = ax3.pcolormesh(x_nums, y_vals, hm_adx_zoom.values, cmap='RdYlGn', shading='nearest', norm=norm)
    ax3.set_yticks(y_vals); ax3.set_yticklabels(ordered_tfs)
    ax3.set_ylabel("Direction (ADX)")
    
    # Panel 4: ADX Stability
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    c4 = ax4.pcolormesh(x_nums, y_vals, hm_adx_ent_zoom.values, cmap='RdYlGn_r', shading='nearest', vmin=0, vmax=1)
    ax4.set_yticks(y_vals); ax4.set_yticklabels(ordered_tfs)
    ax4.set_ylabel("ADX Stability")
    
    ax4.set_yticks(y_vals); ax4.set_yticklabels(ordered_tfs)
    ax4.set_ylabel("ADX Stability")
    
    # Panel 5: Equity Curve
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    # Align equity curve to plot indices
    # equity_curve index is timestamp. We need to map it to x_idx.
    
    # Filter equity to zoom window
    eq_zoom = equity_curve.reindex(df_zoom.index).fillna(method='ffill')
    
    ax5.plot(df_plot.x_idx, eq_zoom.values, color='cyan', linewidth=1.5)
    ax5.set_ylabel("Equity")
    ax5.grid(True, alpha=0.1)
    
    status_text.empty()
    st.pyplot(fig)
