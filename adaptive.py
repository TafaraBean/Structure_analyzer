import optuna
import pandas as pd
import numpy as np
import talib
import os
import sys
import plotly.graph_objects as go
import webbrowser

# --- CONFIGURATION ---
CACHE_FILE = "data/XAU_5mins_resampled.parquet"
TRAIN_SPLIT = 0.7 
INITIAL_CAPITAL = 10000

# 🕒 TIME SETTINGS
FORCE_CLOSE_HOUR = 23  # Close everything at 11 PM
NO_ENTRY_AFTER = 22    # Don't take new trades after 10 PM

TARGET_PORTFOLIO = [
    'CDLLONGLEGGEDDOJI', 'CDLRICKSHAWMAN', 'CDLHIGHWAVE', 
    'CDLENGULFING', 'CDLBELTHOLD'
]

LOT_SIZE = 0.01          
CONTRACT_SIZE = 100      

def load_and_prep_data():
    if not os.path.exists(CACHE_FILE):
        sys.exit("❌ Cache not found. Run your main data builder script first.")
    
    print("[*] Loading Data...")
    df = pd.read_parquet(CACHE_FILE)
    
    # INDICATORS
    # Volatility Baseline: 100-period Average of ATR
    df['ATR_MA'] = talib.SMA(df['ATR'].values, timeperiod=100)
    
    # Pattern Recognition
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    combined_sig = np.zeros(len(df))
    for pat in TARGET_PORTFOLIO:
        func = getattr(talib, pat)
        sig = func(opens, highs, lows, closes)
        combined_sig = np.where(sig != 0, sig, combined_sig)
    
    df['signal'] = combined_sig
    df.dropna(inplace=True)
    return df

# ==========================================
# ⚙️ SIMULATION ENGINE (Pure Gross Profit)
# ==========================================
def simulate_regime_trades(df_slice, params):
    opens = df_slice['open'].values
    highs = df_slice['high'].values
    lows = df_slice['low'].values
    closes = df_slice['close'].values
    atrs = df_slice['ATR'].values
    atr_mas = df_slice['ATR_MA'].values
    signals = df_slice['signal'].values
    times = df_slice.index
    
    n_candles = len(closes)
    
    indices = np.where(signals != 0)[0]
    indices = indices[indices < n_candles - 121] 
    
    trade_results = [] 
    max_hold = 120
    
    for idx in indices:
        # 🚫 FILTER: Don't enter late at night
        if times[idx].hour >= NO_ENTRY_AFTER:
            continue

        # --- 1. REGIME DETECTION ---
        current_atr = atrs[idx]
        baseline_atr = atr_mas[idx]
        
        # Is Volatility > Threshold * Average?
        is_high_vol = current_atr > (baseline_atr * params['vol_threshold'])
        
        if is_high_vol:
            sl_mult = params['h_sl']; tp_mult = params['h_tp']
            be_trigger = params['h_be']; step_mult = params['h_step']
        else:
            sl_mult = params['n_sl']; tp_mult = params['n_tp']
            be_trigger = params['n_be']; step_mult = params['n_step']

        # --- 2. SETUP (Next Open Entry) ---
        direction = 1 if signals[idx] == 100 else -1
        
        # Enter on the NEXT candle's Open (No Lookahead)
        entry_idx = idx + 1
        entry_price = opens[entry_idx] 
        entry_time = times[entry_idx]
        
        sl_dist = current_atr * sl_mult
        tp_dist = current_atr * tp_mult
        be_dist = current_atr * be_trigger
        step_dist = current_atr * step_mult
        
        if direction == 1:
            current_sl = entry_price - sl_dist; current_tp = entry_price + tp_dist
        else:
            current_sl = entry_price + sl_dist; current_tp = entry_price - tp_dist
            
        is_runner = False
        exit_pnl_price = 0.0
        exit_time = entry_time
        
        # --- 3. MANAGEMENT ---
        for i in range(0, max_hold): 
            curr = entry_idx + i
            if curr >= n_candles: break

            c_open = opens[curr] 
            c_high = highs[curr]
            c_low = lows[curr]
            c_close = closes[curr]
            current_time = times[curr]
            
            # ⏰ HARD EXIT: Force Close at 11 PM
            if current_time.hour == FORCE_CLOSE_HOUR:
                exit_pnl_price = (c_close - entry_price) if direction == 1 else (entry_price - c_close)
                exit_time = current_time
                break 

            # Standard Logic (Optimistic Fills - No Gap Check)
            if direction == 1: 
                if c_low <= current_sl: 
                    exit_pnl_price = current_sl - entry_price; exit_time = current_time; break 
                if not is_runner and c_high >= current_tp: 
                    exit_pnl_price = current_tp - entry_price; exit_time = current_time; break
                
                # Break Even & Trailing
                if not is_runner and c_high >= (entry_price + be_dist): 
                    is_runner = True; current_sl = entry_price; current_tp = 9999999 
                if is_runner: 
                    steps = int((c_high - entry_price) / step_dist)
                    if steps >= 1:
                        new_sl = entry_price + (steps * step_dist) - step_dist
                        if new_sl > current_sl: current_sl = new_sl
                            
            else: 
                if c_high >= current_sl: 
                    exit_pnl_price = entry_price - current_sl; exit_time = current_time; break
                if not is_runner and c_low <= current_tp: 
                    exit_pnl_price = entry_price - current_tp; exit_time = current_time; break
                
                if not is_runner and c_low <= (entry_price - be_dist): 
                    is_runner = True; current_sl = entry_price; current_tp = -9999999
                if is_runner: 
                    steps = int((entry_price - c_low) / step_dist)
                    if steps >= 1:
                        new_sl = entry_price - (steps * step_dist) + step_dist
                        if new_sl < current_sl: current_sl = new_sl
            
            # Time Stop (Max Hold)
            if i == (max_hold - 1):
                exit_pnl_price = (c_close - entry_price) if direction == 1 else (entry_price - c_close)
                exit_time = current_time

        # 💰 NO COSTS APPLIED (Pure Gross PnL)
        raw_pnl = exit_pnl_price * CONTRACT_SIZE * LOT_SIZE
        trade_results.append({'exit_time': exit_time, 'pnl': raw_pnl})

    return pd.DataFrame(trade_results)

# ==========================================
# 🎯 OPTUNA OBJECTIVE
# ==========================================
def objective(trial):
    # Volatility Threshold
    vol_threshold = trial.suggest_float("vol_threshold", 1.0, 1.8)
    
    # Normal Regime Params
    n_sl = trial.suggest_float("n_sl", 1.5, 3.5, step=0.1)
    n_tp = trial.suggest_float("n_tp", 1.0, 3.0, step=0.1)
    n_be = trial.suggest_float("n_be", 0.8, 2.0, step=0.1)
    n_step = trial.suggest_float("n_step", 0.2, 0.6, step=0.1)
    
    # High Vol Regime Params
    h_sl = trial.suggest_float("h_sl", 2.0, 5.0, step=0.1)
    h_tp = trial.suggest_float("h_tp", 2.0, 6.0, step=0.1)
    h_be = trial.suggest_float("h_be", 0.3, 1.2, step=0.1) 
    h_step = trial.suggest_float("h_step", 0.4, 1.0, step=0.1)
    
    params = {
        'vol_threshold': vol_threshold,
        'n_sl': n_sl, 'n_tp': n_tp, 'n_be': n_be, 'n_step': n_step,
        'h_sl': h_sl, 'h_tp': h_tp, 'h_be': h_be, 'h_step': h_step
    }
    
    trades_df = simulate_regime_trades(train_df, params)
    
    # 30 trades minimum to ensure statistical validity
    if trades_df.empty or len(trades_df) < 30: return -9999
    
    pnl_array = trades_df['pnl'].values
    avg = np.mean(pnl_array)
    std = np.std(pnl_array)
    
    # Simply maximize Sharpe (no minimum $ filter)
    if avg <= 0: return -9999
    return avg / (std + 1e-6)

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    full_df = load_and_prep_data()
    
    split_idx = int(len(full_df) * TRAIN_SPLIT)
    split_date = full_df.index[split_idx] 
    
    train_df = full_df.iloc[:split_idx].copy()
    test_df = full_df.iloc[split_idx:].copy()
    
    print(f"📊 Data Split: Train={len(train_df)} | Test={len(test_df)}")

    # 1. OPTIMIZATION
    print("\n🤖 Optimization Started (Train Set)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100) 

    best = study.best_params
    
    # 2. DETAILED REPORTING
    print("\n" + "="*60)
    print("🏆 BEST PARAMETERS (PURE GROSS PROFIT)")
    print("   (No Spread, No Commissions, No Quality Filter)")
    print("="*60)
    print(f"⚡ VOLATILITY TRIGGER: ATR > {best['vol_threshold']:.2f} x (100-SMA)")
    
    print("\n🔵 NORMAL REGIME (Low Volatility)")
    print(f"   • Stop Loss   : {best['n_sl']} ATR")
    print(f"   • Take Profit : {best['n_tp']} ATR")
    print(f"   • BE Trigger  : {best['n_be']} ATR")
    print(f"   • Trailing Step: {best['n_step']} ATR")

    print("\n🔴 HIGH VOLATILITY REGIME")
    print(f"   • Stop Loss   : {best['h_sl']} ATR")
    print(f"   • Take Profit : {best['h_tp']} ATR")
    print(f"   • BE Trigger  : {best['h_be']} ATR")
    print(f"   • Trailing Step: {best['h_step']} ATR")
    print("="*60 + "\n")
    
    # 3. FULL SIMULATION
    print("🎬 Simulating Full History...")
    full_trades_df = simulate_regime_trades(full_df, best)
    
    if full_trades_df.empty:
        print("⚠️ No trades generated.")
        sys.exit()

    full_trades_df.sort_values('exit_time', inplace=True)
    full_trades_df.set_index('exit_time', inplace=True)
    
    # Stats
    avg_trade_val = full_trades_df['pnl'].mean()
    equity_series = full_trades_df['pnl'].cumsum() + INITIAL_CAPITAL
    final_equity = equity_series.iloc[-1]
    total_ret = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"💰 Average Gross Profit per Trade: ${avg_trade_val:.2f}")
    
    # 4. PLOTTING
    print("[*] Generating Full Equity Chart...")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
        mode='lines', name='Total Equity', line=dict(color='#00ff00', width=2)))
    
    fig.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="white")
    
    fig.add_annotation(x=full_df.index[int(split_idx/2)], y=INITIAL_CAPITAL,
                       text="TRAIN (In-Sample)", showarrow=False, font=dict(size=16, color="gray"))
    
    fig.add_annotation(x=full_df.index[split_idx + int(len(test_df)/2)], y=INITIAL_CAPITAL,
                       text="TEST (Out-of-Sample)", showarrow=False, font=dict(size=16, color="yellow"))

    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="gray")

    fig.update_layout(
        title=f"Gross Equity Curve (Pure Alpha)<br><sup>Return: {total_ret:.2f}% | Avg Trade: ${avg_trade_val:.2f}</sup>",
        xaxis_title="Date", yaxis_title="Equity ($)", template="plotly_dark", height=700
    )

    filename = "full_equity_PURE.html"
    fig.write_html(filename)
    print(f"🚀 Opening plot: {filename}")
    webbrowser.open(filename)