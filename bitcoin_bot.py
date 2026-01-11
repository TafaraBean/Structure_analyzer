import pandas as pd
import pandas_ta as ta
import optuna
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import os
import sys
import time
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")

CSV_FILES = ["Exness_BTCUSD_2024.csv", "Exness_BTCUSD_2025.csv", "Exness_BTCUSD_2026.csv"]
PARQUET_FILE = "btc_learner_v3.parquet"
OPTUNA_DB = "sqlite:///btc_memory.db"  # <--- THE BRAIN (Persistent Database)

# Strategy Constants
INITIAL_CAPITAL = 10000
COST_PCT = 0.0006  
SPLIT_RATIO = 0.70  
N_TRIALS = 150      
CHUNK_SIZE = 5000000

# Live Settings
SYMBOL = "BTCUSDz"
TIMEFRAME = mt5.TIMEFRAME_M1
LOT_SIZE = 0.01
MAGIC = 555003

# ==========================================
# PART 1: DATA FOUNDRY
# ==========================================
def build_dataset():
    if os.path.exists(PARQUET_FILE): 
        print(f"✅ Found existing dataset: {PARQUET_FILE}")
        return True
    
    print(f"[*] Building Dataset...")
    all_candles = []
    for f in CSV_FILES:
        if not os.path.exists(f): continue
        try:
            chunk_iterator = pd.read_csv(f, chunksize=CHUNK_SIZE, header=0, usecols=[2, 3], names=['time', 'bid'], quotechar='"')
            for i, chunk in enumerate(chunk_iterator):
                chunk['time'] = pd.to_datetime(chunk['time'], format='ISO8601', errors='coerce')
                chunk.dropna(inplace=True)
                chunk.set_index('time', inplace=True)
                resampled = chunk['bid'].resample('1min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'bid': 'count'}).rename(columns={'bid': 'volume'})
                resampled.dropna(inplace=True)
                all_candles.append(resampled)
                print(f"   Processed chunk {i+1}...", end='\r')
        except: pass
    
    if not all_candles: return False
    full_df = pd.concat(all_candles)
    full_df = full_df.groupby(full_df.index).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    full_df.sort_index(inplace=True)
    full_df.to_parquet(PARQUET_FILE)
    print(f"\n✅ DATA READY: {len(full_df)} candles.")
    return True

# ==========================================
# PART 2: THE SCIENTIST (Feature Engineering)
# ==========================================
def engineer_features(df):
    df = df.copy()
    
    # 1. Bollinger Position (Location)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_Lower'] = bb.iloc[:, 0]
    df['BB_Upper'] = bb.iloc[:, 2]
    df['BB_Pos'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # 2. ADX (Trend Strength)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14']
    
    # 3. RSI (Momentum)
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    # 4. Slope (Velocity)
    df['Slope'] = ta.slope(df['close'], length=5)
    
    # 5. Wick Ratios (Rejection)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['Up_Wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['Low_Wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['Up_Wick_Ratio'] = df['Up_Wick'] / df['ATR']
    df['Low_Wick_Ratio'] = df['Low_Wick'] / df['ATR']
    
    # 6. Direction (Memory)
    df['Dir'] = np.where(df['close'] > df['open'], 1, -1)

    # --- HONESTY SHIFT ---
    feature_cols = ['BB_Pos', 'ADX', 'RSI', 'Slope', 'Up_Wick_Ratio', 'Low_Wick_Ratio', 'Dir']
    for col in feature_cols:
        df[f'Prev_{col}'] = df[col].shift(1)
        
    df['Prev_ATR'] = df['ATR'].shift(1) 
    
    return df.dropna(), [f'Prev_{c}' for c in feature_cols]

def select_top_features(df, feature_cols):
    print("\n[*] Training Decision Tree to find Top Features...")
    
    # Target: 0.1% Move
    df['Future_Ret'] = df['close'].shift(-5) / df['close'] - 1
    conditions = [(df['Future_Ret'] > 0.001), (df['Future_Ret'] < -0.001)]
    choices = [1, -1]
    df['Target'] = np.select(conditions, choices, default=0)
    
    mask = df['Target'] != 0
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, 'Target']
    
    if len(y) < 100: return []
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("-" * 40)
    print("🧠 MARKET DNA (Feature Importance)")
    print("-" * 40)
    
    ranked_features = []
    for f in range(len(feature_cols)):
        name = feature_cols[indices[f]]
        score = importances[indices[f]]
        print(f"{f+1}. {name:<20} : {score*100:.1f}%")
        if score > 0.05: 
            ranked_features.append(name)
            
    return ranked_features[:3]

# ==========================================
# PART 3: THE ENGINEER (Dynamic Logic)
# ==========================================
def run_dynamic_backtest(df, params, active_features):
    trail = params['trail_mult']
    
    trades = []
    equity = [INITIAL_CAPITAL]
    balance = INITIAL_CAPITAL
    
    opens = df['open'].values
    atrs = df['Prev_ATR'].values
    
    feats = {f: df[f].values for f in active_features}
    
    pos = 0 
    entry_px = 0.0
    sl_px = 0.0
    
    for i in range(len(df)):
        current_bal = balance
        
        # 1. Manage Exits
        if pos != 0:
            curr_high = df.iloc[i]['high']
            curr_low = df.iloc[i]['low']
            if (pos == 1 and curr_low <= sl_px) or (pos == -1 and curr_high >= sl_px):
                exit_px = sl_px
                net_ret = ((exit_px - entry_px)/entry_px if pos == 1 else (entry_px - exit_px)/entry_px) - COST_PCT
                trades.append(net_ret)
                balance *= (1 + net_ret)
                pos = 0 
                current_bal = balance
            else:
                if pos == 1:
                    new_sl = curr_high - (atrs[i] * trail)
                    if new_sl > sl_px: sl_px = new_sl
                elif pos == -1:
                    new_sl = curr_low + (atrs[i] * trail)
                    if new_sl < sl_px: sl_px = new_sl
                    
        # 2. Check Entries
        if pos == 0:
            buy_score = 0
            sell_score = 0
            
            # --- LOGIC GATES ---
            if 'Prev_BB_Pos' in active_features:
                val = feats['Prev_BB_Pos'][i]
                if val < params.get('bb_low', 0.2): buy_score += 1
                if val > params.get('bb_high', 0.8): sell_score += 1
                
            if 'Prev_ADX' in active_features:
                val = feats['Prev_ADX'][i]
                if val < params.get('adx_limit', 25): 
                    buy_score += 1; sell_score += 1
                else:
                    buy_score -= 10; sell_score -= 10
                    
            if 'Prev_RSI' in active_features:
                val = feats['Prev_RSI'][i]
                if val < params.get('rsi_low', 30): buy_score += 1
                if val > params.get('rsi_high', 70): sell_score += 1
            
            if 'Prev_Slope' in active_features:
                val = feats['Prev_Slope'][i]
                if val < params.get('slope_low', -10): buy_score += 1 
                if val > params.get('slope_high', 10): sell_score += 1
            
            if 'Prev_Up_Wick_Ratio' in active_features:
                val = feats['Prev_Up_Wick_Ratio'][i]
                if val > params.get('wick_ratio', 0.5): sell_score += 1
            
            if 'Prev_Low_Wick_Ratio' in active_features:
                val = feats['Prev_Low_Wick_Ratio'][i]
                if val > params.get('wick_ratio', 0.5): buy_score += 1

            thresh = len(active_features) 
            
            if buy_score >= thresh:
                pos = 1
                entry_px = opens[i]
                sl_px = entry_px - (atrs[i] * trail)
            elif sell_score >= thresh:
                pos = -1
                entry_px = opens[i]
                sl_px = entry_px + (atrs[i] * trail)
        
        # Only log equity periodically to save memory in massive loops
        if detailed: equity.append(current_bal)
        
    return (trades, equity) if detailed else trades

def objective(trial, train_df, active_features):
    params = {'trail_mult': trial.suggest_float('trail_mult', 1.0, 5.0)}
    
    if 'Prev_BB_Pos' in active_features:
        params['bb_low'] = trial.suggest_float('bb_low', 0.0, 0.4)
        params['bb_high'] = trial.suggest_float('bb_high', 0.6, 1.0)
    
    if 'Prev_ADX' in active_features:
        params['adx_limit'] = trial.suggest_int('adx_limit', 15, 40)
        
    if 'Prev_RSI' in active_features:
        params['rsi_low'] = trial.suggest_int('rsi_low', 20, 45)
        params['rsi_high'] = trial.suggest_int('rsi_high', 55, 80)
        
    if 'Prev_Slope' in active_features:
        params['slope_low'] = trial.suggest_float('slope_low', -50, -5)
        params['slope_high'] = trial.suggest_float('slope_high', 5, 50)
        
    if 'Prev_Up_Wick_Ratio' in active_features or 'Prev_Low_Wick_Ratio' in active_features:
        params['wick_ratio'] = trial.suggest_float('wick_ratio', 0.3, 1.0)
        
    try:
        trades = run_dynamic_backtest(train_df, params, active_features)
        count = len(trades)
        if count < 100: return -1
        trade_series = np.array(trades)
        if np.std(trade_series) == 0: return 0
        return (np.mean(trade_series) / np.std(trade_series)) * np.sqrt(count)
    except: return -1

# ==========================================
# PART 4: LIVE BOT
# ==========================================
def start_live_bot(best_params, active_features):
    print("\n" + "="*50)
    print("🚀 LIVE AUTONOMOUS BOT V3 (PERSISTENT MEMORY)")
    print(f"   Active Logic: {active_features}")
    print("="*50)
    
    if not mt5.initialize(): return

    def get_live_data():
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 300)
        if rates is None: return None
        df = pd.DataFrame(rates)
        
        # Calc Features
        bb = ta.bbands(df['close'], length=20, std=2)
        df['BB_Pos'] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['ADX'] = adx['ADX_14']
        
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['Slope'] = ta.slope(df['close'], length=5)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        df['Up_Wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['Low_Wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['Up_Wick_Ratio'] = df['Up_Wick'] / df['ATR']
        df['Low_Wick_Ratio'] = df['Low_Wick'] / df['ATR']
        
        # Return PREVIOUS candle
        prev = df.iloc[-2]
        return {
            'Prev_BB_Pos': prev['BB_Pos'],
            'Prev_ADX': prev['ADX'],
            'Prev_RSI': prev['RSI'],
            'Prev_Slope': prev['Slope'],
            'Prev_Up_Wick_Ratio': prev['Up_Wick_Ratio'],
            'Prev_Low_Wick_Ratio': prev['Low_Wick_Ratio'],
            'ATR': prev['ATR'],
            'close': df.iloc[-1]['close']
        }

    print("[*] Monitoring...")
    while True:
        try:
            data = get_live_data()
            if data is None: 
                time.sleep(1)
                continue
            
            buy_score = 0
            sell_score = 0
            
            if 'Prev_BB_Pos' in active_features:
                if data['Prev_BB_Pos'] < best_params.get('bb_low', 0): buy_score += 1
                if data['Prev_BB_Pos'] > best_params.get('bb_high', 1): sell_score += 1
            
            if 'Prev_ADX' in active_features:
                if data['Prev_ADX'] < best_params.get('adx_limit', 99):
                    buy_score += 1; sell_score += 1
                else:
                    buy_score -= 10; sell_score -= 10
            
            if 'Prev_RSI' in active_features:
                if data['Prev_RSI'] < best_params.get('rsi_low', 0): buy_score += 1
                if data['Prev_RSI'] > best_params.get('rsi_high', 100): sell_score += 1
            
            if 'Prev_Up_Wick_Ratio' in active_features:
                if data['Prev_Up_Wick_Ratio'] > best_params.get('wick_ratio', 0.5): sell_score += 1
                
            if 'Prev_Low_Wick_Ratio' in active_features:
                if data['Prev_Low_Wick_Ratio'] > best_params.get('wick_ratio', 0.5): buy_score += 1

            thresh = len(active_features)
            
            positions = mt5.positions_get(symbol=SYMBOL)
            is_flat = positions is None or len([p for p in positions if p.magic == MAGIC]) == 0
            
            if is_flat:
                if buy_score >= thresh:
                    print(f"🚀 BUY SIGNAL")
                    sl = data['close'] - (data['ATR'] * best_params['trail_mult'])
                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": LOT_SIZE, "type": mt5.ORDER_TYPE_BUY, "price": mt5.symbol_info_tick(SYMBOL).ask, "sl": sl, "magic": MAGIC})
                    time.sleep(60)
                elif sell_score >= thresh:
                    print(f"🚀 SELL SIGNAL")
                    sl = data['close'] + (data['ATR'] * best_params['trail_mult'])
                    mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": LOT_SIZE, "type": mt5.ORDER_TYPE_SELL, "price": mt5.symbol_info_tick(SYMBOL).bid, "sl": sl, "magic": MAGIC})
                    time.sleep(60)
            time.sleep(1)
        except KeyboardInterrupt: break
        except Exception as e: print(e); time.sleep(5)

if __name__ == "__main__":
    if not build_dataset(): sys.exit()
    df_raw = pd.read_parquet(PARQUET_FILE)
    
    df_eng, features = engineer_features(df_raw)
    split_idx = int(len(df_eng) * SPLIT_RATIO)
    train_df = df_eng.iloc[:split_idx].copy()
    test_df = df_eng.iloc[split_idx:].copy()
    
    active_features = select_top_features(train_df, features)
    if not active_features: sys.exit()
    print(f"✅ ACTIVE MODULES: {active_features}")
    
    print(f"[*] Optimizing (Memory Active: {OPTUNA_DB})...")
    # THE MEMORY UPGRADE:
    study = optuna.create_study(
        study_name="btc_learner_v3", 
        direction="maximize", 
        storage=OPTUNA_DB, 
        load_if_exists=True
    )
    study.optimize(lambda t: objective(t, train_df, active_features), n_trials=N_TRIALS)
    best_params = study.best_params
    
    print(f"[*] Best Params: {best_params}")
    t_te, eq_te = run_dynamic_backtest(test_df, best_params, active_features, detailed=True)
    
    if len(t_te) > 0:
        roi = (eq_te[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        print(f"\n📊 TEST RESULT: ROI {roi:.2f}% | Trades {len(t_te)}")
        plt.plot(eq_te); plt.show(block=False); plt.pause(5); plt.close()
        if roi > 0:
            if input(">>> Start Live? (y/n): ").lower() == 'y': start_live_bot(best_params, active_features)
    else: print("❌ No trades in Test set.")