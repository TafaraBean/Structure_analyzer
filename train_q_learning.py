import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import talib
import os
import joblib
import random
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load environment variables
load_dotenv()

# --- Configuration ---
SYMBOL = "XAUUSDm"
BARS_TOTAL = 30000 
TRAIN_WINDOW = 5000
TEST_WINDOW = 1000
STEP_SIZE = 1000 # Usually equal to Test Window

# Model Config
N_CLUSTERS = 50 
ACTIONS = [0, 1, 2] # 0=Flat, 1=Long, 2=Short
LEARNING_RATE = 0.01 # Tuned for WFA
DISCOUNT_FACTOR = 0.95
EPSILON = 0.2
EPISODES_PER_FOLD = 1000 # Faster training per fold

# Feature Settings
MA_PERIOD = 20
PE_ORDER = 4
PE_DELAY = 1
PE_WINDOW = 30

TIMEFRAMES = {
    'M1':  {'tf': mt5.TIMEFRAME_M1,  'delta': pd.Timedelta(minutes=1)},
    'M5':  {'tf': mt5.TIMEFRAME_M5,  'delta': pd.Timedelta(minutes=5)},
    'M15': {'tf': mt5.TIMEFRAME_M15, 'delta': pd.Timedelta(minutes=15)},
    'H1':  {'tf': mt5.TIMEFRAME_H1,  'delta': pd.Timedelta(hours=1)},
}

def init_mt5():
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    params = {}
    if path: params["path"] = path
    if not mt5.initialize(**params):
        print(f"❌ MT5 Init failed: {mt5.last_error()}")
        return False
    if login and password and server:
        mt5.login(login=int(login), password=password, server=server)
    return True

def fetch_data(symbol, tf, bars):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def permutation_entropy(series, order=3, delay=1, window_size=30):
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

def prepare_features():
    if not init_mt5(): return None
    print(f"📥 Fetching Master Data ({BARS_TOTAL} bars)...")
    df_master = fetch_data(SYMBOL, mt5.TIMEFRAME_M1, BARS_TOTAL)
    if df_master is None: return None
    
    X = pd.DataFrame(index=df_master.index)
    print("🛠️  Engineering Features (Regime Detection)...")
    
    for name, tf_info in TIMEFRAMES.items():
        df_tf = fetch_data(SYMBOL, tf_info['tf'], int(BARS_TOTAL/2)) # Approx
        if df_tf is None: continue
        
        ema = calculate_ema(df_tf['close'], MA_PERIOD)
        pe_ema = permutation_entropy(ema, order=PE_ORDER, delay=PE_DELAY, window_size=PE_WINDOW)
        adx = talib.ADX(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        pe_adx = permutation_entropy(pd.Series(adx, index=df_tf.index), order=PE_ORDER, delay=PE_DELAY, window_size=PE_WINDOW)
        
        def sync(series, col_name):
            s = pd.Series(series, index=df_tf.index)
            s.index = s.index + tf_info['delta']
            aligned = s.reindex(df_master.index, method='ffill')
            X[f"{name}_{col_name}"] = aligned
            
        sync(pe_ema, "PE_EMA")
        sync(adx, "ADX")
        sync(pe_adx, "PE_ADX")
    
    X.dropna(inplace=True)
    prices = df_master['close'].loc[X.index]
    return X, prices

def train_agent(X_train, prices_train):
    """
    Trains a Q-Learning agent on a specific slice of data.
    Returns: Scaler, KMeans, Q-Table
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    train_states = kmeans.fit_predict(X_scaled)
    
    q_table = np.zeros((N_CLUSTERS, len(ACTIONS)))
    
    # Training Loop
    # Using Diff Sharpe Reward
    A_t = 0
    B_t = 1e-6
    eta = 0.05
    
    for episode in range(EPISODES_PER_FOLD):
        state = train_states[0]
        
        for t in range(len(train_states) - 1):
            if random.uniform(0, 1) < EPSILON:
                action = random.choice(ACTIONS)
            else:
                action = np.argmax(q_table[state])
            
            curr_price = prices_train.iloc[t]
            next_price = prices_train.iloc[t+1]
            raw_ret = (next_price - curr_price) / curr_price
            
            search_reward = 0
            if action == 1: search_reward = raw_ret
            elif action == 2: search_reward = -raw_ret
            
            # Diff Sharpe
            delta_A = search_reward - A_t
            delta_B = (search_reward**2) - B_t
            variance = B_t - A_t**2
            if variance < 1e-9: variance = 1e-9
            std_dev = np.sqrt(variance)
            diff_sharpe = (B_t * delta_A - 0.5 * A_t * delta_B) / (std_dev**3)
            
            A_t += eta * delta_A
            B_t += eta * delta_B
            
            reward = diff_sharpe * 100
            
            next_state = train_states[t+1]
            old_val = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_val = old_val + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max - old_val)
            q_table[state, action] = new_val
            state = next_state
            
    return scaler, kmeans, q_table

def run_backtest(X_test, prices_test, scaler, kmeans, q_table):
    """
    Runs the trained agent on test data.
    Returns: Equity Curve (add value, not cumulative sum yet), Actions
    """
    X_test_scaled = scaler.transform(X_test)
    test_states = kmeans.predict(X_test_scaled)
    
    pnls = []
    bh_pnls = []
    actions = []
    
    for t in range(len(test_states) - 1):
        state = test_states[t]
        action = np.argmax(q_table[state])
        actions.append(action)
        
        curr_price = prices_test.iloc[t]
        next_price = prices_test.iloc[t+1]
        change = next_price - curr_price
        
        pnl = 0
        if action == 1: pnl = change
        elif action == 2: pnl = -change
        
        pnls.append(pnl)
        bh_pnls.append(change)
        
    return pnls, bh_pnls, actions

def walk_forward_optimization():
    X, prices = prepare_features()
    if X is None: return
    
    total_samples = len(X)
    print(f"📊 Dataset Size: {total_samples} samples")
    
    oos_equity = [1000] # Starting Capital
    oos_bh_equity = [1000]
    
    all_oos_indices = []
    all_oos_equity = [1000]
    all_oos_bh = [1000]
    
    # WFA Loop
    start_index = 0
    fold = 1
    
    while start_index + TRAIN_WINDOW + TEST_WINDOW <= total_samples:
        train_start = start_index
        train_end = start_index + TRAIN_WINDOW
        test_end = train_end + TEST_WINDOW
        
        print(f"\n🔄 Fold {fold}: Train[{train_start}:{train_end}] -> Test[{train_end}:{test_end}]")
        
        X_train = X.iloc[train_start:train_end]
        p_train = prices.iloc[train_start:train_end]
        
        X_test = X.iloc[train_end:test_end]
        p_test = prices.iloc[train_end:test_end]
        
        # 1. Train
        scaler, kmeans, q_table = train_agent(X_train, p_train)
        
        # 2. Test (Out-of-Sample)
        pnls, bh_pnls, actions = run_backtest(X_test, p_test, scaler, kmeans, q_table)
        
        # 3. Stitch Results
        for p, bh_p in zip(pnls, bh_pnls):
            all_oos_equity.append(all_oos_equity[-1] + p)
            all_oos_bh.append(all_oos_bh[-1] + bh_p)
            
        # Indices for plotting (minus last one due to lookahead in loop)
        all_oos_indices.extend(p_test.index[:-1])
        
        start_index += STEP_SIZE
        fold += 1
        
    # --- Visualization ---
    print("🎨 Generating Walk-Forward Chart...")
    plt.style.use('dark_background')
    plt.figure(figsize=(16, 8))
    
    # Adjust length (equity has initial 1000)
    plt.plot(all_oos_indices, all_oos_equity[1:], color='cyan', label='WFA Strategy Equity')
    plt.plot(all_oos_indices, all_oos_bh[1:], color='white', linestyle='--', alpha=0.5, label='Buy & Hold')
    
    # Metrics
    final_ret = (all_oos_equity[-1] - 1000) / 1000 * 100
    bh_ret = (all_oos_bh[-1] - 1000) / 1000 * 100
    
    plt.title(f"Walk Forward Efficiency (Stitched OOS) | Strategy: {final_ret:.2f}% vs B&H: {bh_ret:.2f}%", fontsize=16, color='white')
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(alpha=0.1)
    
    plt.savefig("walk_forward_equity.png")
    
    # Save the LATEST model (trained on the most recent window) for Live Trading
    print("💾 Saving Latest Model (from last fold)...")
    joblib.dump(kmeans, "regime_kmeans.pkl")
    joblib.dump(scaler, "regime_scaler.pkl")
    np.save("q_table.npy", q_table)
    
    print("✅ WFA Complete.")

if __name__ == "__main__":
    walk_forward_optimization()
