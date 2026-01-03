"""
MT5 Magnet Strategy Backtester (Fixed Lookahead)
------------------------------------------------
Strategy: "Golden Zone Magnets + Price Action"
1. Identify Golden Zones (Confluence).
2. Bias: Trade towards the Golden Zone (Magnet).
3. Entry: Wait for price to touch a Minor Level AND form a Valid Candlestick Pattern.
4. Risk Management: 1:2 Risk/Reward.

FIXES:
- Lookahead Bias: Strictly filters HTF data to exclude the current day.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.io as pio
import talib 

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

MT5_CREDS = {
    "login": 54704353,          
    "password": "p@s5k3yHFM",   
    "server": "HFMarketsSA-Live2",
    "symbol": "#BTCUSD",         
    "timeframe": mt5.TIMEFRAME_M5,
    "days_history": 180 
}

HTF_MAPPING = {
    mt5.TIMEFRAME_M5: mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_M15: mt5.TIMEFRAME_H1,
    mt5.TIMEFRAME_H1: mt5.TIMEFRAME_D1
}

STRATEGY_PARAMS = {
    "n_clusters": 15,            
    "min_history_days": 7,      
    "ma_period": 50,            
    "atr_period": 14,           
    "sl_atr_mult": 1.5,         
    "rr_ratio": 2.0,            
    "spread": 1.2,              
    "confluence_tol": 0.003,
    "use_patterns": True        
}

# ==========================================
# 🛠️ DATA ENGINE
# ==========================================

def connect_mt5():
    if not mt5.initialize():
        print(f"❌ MT5 Init Failed: {mt5.last_error()}")
        return False
    if not mt5.terminal_info():
        if not mt5.login(login=MT5_CREDS['login'], password=MT5_CREDS['password'], server=MT5_CREDS['server']):
            print(f"❌ Login Failed: {mt5.last_error()}")
            return False
    return True

def fetch_data_multi_tf():
    base_tf = MT5_CREDS['timeframe']
    htf_tf = HTF_MAPPING.get(base_tf, mt5.TIMEFRAME_D1)
    
    if base_tf == mt5.TIMEFRAME_H1: base_per_day = 24
    elif base_tf == mt5.TIMEFRAME_M15: base_per_day = 96
    else: base_per_day = 288
    
    count_base = MT5_CREDS['days_history'] * base_per_day
    count_htf = MT5_CREDS['days_history'] * 24 
    
    print(f"📥 Fetching data for {MT5_CREDS['symbol']}...")
    
    rates_base = mt5.copy_rates_from_pos(MT5_CREDS['symbol'], base_tf, 0, count_base)
    if rates_base is None: return None, None
    df_base = pd.DataFrame(rates_base)
    df_base['time'] = pd.to_datetime(df_base['time'], unit='s')
    df_base.set_index('time', inplace=True)
    
    rates_htf = mt5.copy_rates_from_pos(MT5_CREDS['symbol'], htf_tf, 0, count_htf)
    if rates_htf is None: return df_base, None
    df_htf = pd.DataFrame(rates_htf)
    df_htf['time'] = pd.to_datetime(df_htf['time'], unit='s')
    df_htf.set_index('time', inplace=True)
    
    print(f"✅ Loaded: Base={len(df_base)} | HTF={len(df_htf)}")
    return df_base, df_htf

# ==========================================
# 🧠 STRUCTURAL LOGIC
# ==========================================

def calculate_levels_weighted_kmeans(df_slice, n_clusters):
    if len(df_slice) < 20: return []
    
    velocity = df_slice['close'].diff()
    reversal_mask = (velocity * velocity.shift(1)) < 0
    
    rev_highs = df_slice.loc[reversal_mask, 'high'].values.reshape(-1, 1)
    rev_lows = df_slice.loc[reversal_mask, 'low'].values.reshape(-1, 1)
    
    vol_col = 'real_volume' if 'real_volume' in df_slice.columns and df_slice['real_volume'].sum() > 0 else 'tick_volume'
    vol_series = df_slice[vol_col]
    range_series = df_slice['high'] - df_slice['low']
    
    def normalize(arr):
        if len(arr) == 0 or np.max(arr) == np.min(arr): return np.ones_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) + 0.1

    w_vol = normalize(np.log1p(vol_series.loc[reversal_mask].values))
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

def get_fibs(df_slice):
    high = df_slice['high'].max()
    low = df_slice['low'].min()
    diff = high - low
    return [low, low + 0.236*diff, low + 0.382*diff, low + 0.5*diff, low + 0.618*diff, low + 0.786*diff, high]

def find_golden_zones(base_levels, htf_levels, fib_levels, tol=0.003):
    golden = []
    fib_vals = list(fib_levels) if fib_levels else []
    for lvl in base_levels:
        is_htf = any(abs(lvl - h)/lvl < tol for h in htf_levels) if htf_levels else False
        is_fib = any(abs(lvl - f)/lvl < tol for f in fib_vals) if fib_vals else False
        if is_htf or is_fib: golden.append(lvl)
    return golden

# ==========================================
# ⚔️ STRATEGY ENGINE
# ==========================================

def run_magnet_backtest(df_base, df_htf):
    print("🏃 Starting Magnet Strategy Backtest (Fixed Logic)...")
    start_time = time.time()
    
    df = df_base.copy()
    
    # 1. Indicators
    df['MA'] = df['close'].rolling(STRATEGY_PARAMS['ma_period']).mean()
    df['MA_Slope'] = df['MA'].diff()
    
    # 2. Pattern Recognition (Vectorized with TA-Lib)
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    
    pat_engulfing = talib.CDLENGULFING(o, h, l, c)
    pat_hammer = talib.CDLHAMMER(o, h, l, c)       
    pat_shooting = talib.CDLSHOOTINGSTAR(o, h, l, c)
    
    bullish_signals = ((pat_engulfing == 100) | (pat_hammer == 100)).astype(int)
    bearish_signals = ((pat_engulfing == -100) | (pat_shooting == -100)).astype(int) * -1
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(STRATEGY_PARAMS['atr_period']).mean()

    # Window Setup
    tf = MT5_CREDS['timeframe']
    candles_per_day = 24 if tf == mt5.TIMEFRAME_H1 else 96
    lookback_candles = STRATEGY_PARAMS['min_history_days'] * candles_per_day
    warmup_idx = max(lookback_candles, STRATEGY_PARAMS['ma_period'] + 50)
    
    if len(df) <= warmup_idx: return pd.DataFrame(), []

    trades = []
    equity = [0]
    active_trade = None
    
    current_levels = []
    current_golden = []
    last_date = df.index[warmup_idx-1].date()
    
    times = df.index
    dates = df.index.date
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    slopes = df['MA_Slope'].fillna(0).values
    atrs = df['ATR'].fillna(0).values
    
    def update_structure(idx):
        start_slice = max(0, idx - lookback_candles)
        slice_base = df.iloc[start_slice:idx]
        minor_levels = calculate_levels_weighted_kmeans(slice_base, STRATEGY_PARAMS['n_clusters'] + 2)
        
        htf_levels = []
        if df_htf is not None:
            # FIX: STRICTLY FILTER HTF DATA TO PREVIOUS DAYS ONLY
            # df.index[idx] is the current bar time.
            # We want Daily bars where Date < Current Date.
            # This ensures we don't see the current incomplete daily candle OR the completed one if backtesting end of day.
            current_date_ts = pd.Timestamp(dates[idx])
            slice_htf = df_htf[df_htf.index < current_date_ts]
            
            if len(slice_htf) > 20:
                htf_levels = calculate_levels_weighted_kmeans(slice_htf.iloc[-60:], 5)
        
        fibs = get_fibs(slice_base)
        golden_zones = find_golden_zones(minor_levels, htf_levels, fibs, STRATEGY_PARAMS['confluence_tol'])
        return sorted(minor_levels), sorted(golden_zones)

    current_levels, current_golden = update_structure(warmup_idx)
    
    for i in range(warmup_idx, len(df)):
        if dates[i] != last_date:
            current_levels, current_golden = update_structure(i)
            last_date = dates[i]
            
        curr_open = opens[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_close = closes[i]
        
        # --- 1. MANAGE ACTIVE TRADE ---
        daily_pnl = 0
        if active_trade:
            sl_hit = False
            tp_hit = False
            
            if active_trade['type'] == 'BUY':
                if curr_low <= active_trade['sl']:
                    pnl = (active_trade['sl'] - active_trade['entry']) - STRATEGY_PARAMS['spread']
                    trades.append({'Time': times[i], 'Type': 'BUY_SL', 'Entry': active_trade['entry'], 'Exit': active_trade['sl'], 'PnL': pnl})
                    sl_hit = True
                elif curr_high >= active_trade['tp']:
                    pnl = (active_trade['tp'] - active_trade['entry']) - STRATEGY_PARAMS['spread']
                    trades.append({'Time': times[i], 'Type': 'BUY_TP', 'Entry': active_trade['entry'], 'Exit': active_trade['tp'], 'PnL': pnl})
                    tp_hit = True
            else: 
                if curr_high >= active_trade['sl']:
                    pnl = (active_trade['entry'] - active_trade['sl']) - STRATEGY_PARAMS['spread']
                    trades.append({'Time': times[i], 'Type': 'SELL_SL', 'Entry': active_trade['entry'], 'Exit': active_trade['sl'], 'PnL': pnl})
                    sl_hit = True
                elif curr_low <= active_trade['tp']:
                    pnl = (active_trade['entry'] - active_trade['tp']) - STRATEGY_PARAMS['spread']
                    trades.append({'Time': times[i], 'Type': 'SELL_TP', 'Entry': active_trade['entry'], 'Exit': active_trade['tp'], 'PnL': pnl})
                    tp_hit = True
            
            if sl_hit or tp_hit:
                daily_pnl = pnl
                active_trade = None
            
            equity.append(equity[-1] + daily_pnl)
            continue 

        # --- 2. HUNT FOR SIGNALS ---
        ma_slope = slopes[i-1]
        is_uptrend = ma_slope > 0
        is_downtrend = ma_slope < 0
        
        magnet_up = False
        magnet_down = False
        
        if current_golden:
            if any(g > curr_open for g in current_golden): magnet_up = True
            if any(g < curr_open for g in current_golden): magnet_down = True
            
        if current_levels:
            nearest_sup = max([l for l in current_levels if l < curr_open], default=None)
            nearest_res = min([l for l in current_levels if l > curr_open], default=None)
            
            atr = atrs[i-1]
            if atr == 0: continue
            
            is_bullish_pattern = bullish_signals[i] == 1
            is_bearish_pattern = bearish_signals[i] == -1
            
            # LONG SETUP
            if magnet_up and is_uptrend and nearest_sup:
                valid_entry = False
                if STRATEGY_PARAMS['use_patterns']:
                    if curr_low <= nearest_sup and is_bullish_pattern:
                        valid_entry = True
                        entry_price = curr_close 
                else:
                    if curr_low <= nearest_sup:
                        valid_entry = True
                        entry_price = nearest_sup
                
                if valid_entry:
                    sl_dist = atr * STRATEGY_PARAMS['sl_atr_mult']
                    tp_dist = sl_dist * STRATEGY_PARAMS['rr_ratio']
                    active_trade = {'type': 'BUY', 'entry': entry_price, 'sl': entry_price - sl_dist, 'tp': entry_price + tp_dist, 'open_idx': i}
                    continue

            # SHORT SETUP
            if magnet_down and is_downtrend and nearest_res:
                valid_entry = False
                if STRATEGY_PARAMS['use_patterns']:
                    if curr_high >= nearest_res and is_bearish_pattern:
                        valid_entry = True
                        entry_price = curr_close
                else:
                    if curr_high >= nearest_res:
                        valid_entry = True
                        entry_price = nearest_res
                
                if valid_entry:
                    sl_dist = atr * STRATEGY_PARAMS['sl_atr_mult']
                    tp_dist = sl_dist * STRATEGY_PARAMS['rr_ratio']
                    active_trade = {'type': 'SELL', 'entry': entry_price, 'sl': entry_price + sl_dist, 'tp': entry_price - tp_dist, 'open_idx': i}
                    continue
        
        equity.append(equity[-1])
        
    print(f"✅ Backtest finished in {time.time() - start_time:.2f}s")
    return pd.DataFrame(trades), equity

def plot_magnet_chart(df, trades):
    if trades.empty: return
    equity_curve = [0]
    pnl_series = pd.Series(0.0, index=df.index)
    for _, t in trades.iterrows(): pnl_series[t['Time']] = t['PnL']
    equity_df = pnl_series.cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_df.index, y=equity_df.values, mode='lines', name='Strategy Equity', line=dict(color='cyan')))
    fig.update_layout(title=f"Magnet Strategy (Patterns): {MT5_CREDS['symbol']}", template="plotly_dark", height=600)
    pio.write_html(fig, f"magnet_equity_{MT5_CREDS['symbol']}.html")

# ==========================================
# 🏁 MAIN
# ==========================================
if __name__ == "__main__":
    if connect_mt5():
        df_base, df_htf = fetch_data_multi_tf()
        if df_base is not None:
            trades, equity = run_magnet_backtest(df_base, df_htf)
            
            if not trades.empty:
                total = len(trades)
                wins = trades[trades['PnL'] > 0]['PnL'].sum()
                losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())
                pf = wins / losses if losses > 0 else 0
                wr = len(trades[trades['PnL'] > 0]) / total * 100
                
                print("\n" + "="*40)
                print(f"🧲 MAGNET PATTERN REPORT: {MT5_CREDS['symbol']}")
                print("-" * 40)
                print(f"Total Trades:   {total}")
                print(f"Net PnL:        {trades['PnL'].sum():.2f}")
                print(f"Win Rate:       {wr:.2f}%")
                print(f"Profit Factor:  {pf:.2f}")
                print("="*40)
                
                trades.to_csv(f"magnet_results_{MT5_CREDS['symbol']}.csv", index=False)
                plot_magnet_chart(df_base, trades)
            else:
                print("⚠️ No trades found.")
        mt5.shutdown()