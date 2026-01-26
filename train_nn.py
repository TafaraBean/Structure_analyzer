import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import talib
import os
from dotenv import load_dotenv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Load environment variables
load_dotenv()

# --- Configuration ---
SYMBOL = "XAUUSDm"
BARS_TRAIN = 10000 # Fetch enough data for training
TARGET_TIMEFRAME = mt5.TIMEFRAME_M1

# Feature Settings
MA_PERIOD = 20
PE_ORDER = 4
PE_DELAY = 1
PE_WINDOW = 30

TIMEFRAMES = {
    'M1':  {'tf': mt5.TIMEFRAME_M1,  'delta': pd.Timedelta(minutes=1)},
    'M5':  {'tf': mt5.TIMEFRAME_M5,  'delta': pd.Timedelta(minutes=5)},
    'M15': {'tf': mt5.TIMEFRAME_M15, 'delta': pd.Timedelta(minutes=15)},
    'M30': {'tf': mt5.TIMEFRAME_M30, 'delta': pd.Timedelta(minutes=30)},
    'H1':  {'tf': mt5.TIMEFRAME_H1,  'delta': pd.Timedelta(hours=1)},
    'H4':  {'tf': mt5.TIMEFRAME_H4,  'delta': pd.Timedelta(hours=4)},
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

def prepare_dataset():
    if not init_mt5(): return None, None
    
    print(f"📥 Fetching Master Data ({BARS_TRAIN} bars)...")
    df_master = fetch_data(SYMBOL, mt5.TIMEFRAME_M1, BARS_TRAIN)
    if df_master is None: return None, None
    
    # Feature DataFrame
    X = pd.DataFrame(index=df_master.index)
    
    print("🛠️  Engineering Features (Multi-Timeframe)...")
    
    for name, tf_info in TIMEFRAMES.items():
        print(f"   - Processing {name}...")
        
        # Calculate roughly how many bars needed for this TF to cover Master range
        # duration = master_end - master_start
        # bars = duration / tf_delta
        # Just fetch max allowed (e.g. 5000 of high TF is massive)
        df_tf = fetch_data(SYMBOL, tf_info['tf'], 5000)
        if df_tf is None: continue
        
        # === FEATURES ===
        
        # 1. EMA Entropy (Stability)
        ema = calculate_ema(df_tf['close'], MA_PERIOD)
        pe_ema = permutation_entropy(ema, order=PE_ORDER, delay=PE_DELAY, window_size=PE_WINDOW)
        
        # 2. ADX/DI (Strength/Direction)
        adx = talib.ADX(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df_tf['high'], df_tf['low'], df_tf['close'], timeperiod=14)
        
        # Encode Direction: 1 if Bullish, -1 if Bearish
        # We can use +DI and -DI directly as features too
        direction_score = np.where(plus_di > minus_di, 1, -1) * adx
        
        # 3. ADX Entropy (Acceleration Stability)
        pe_adx = permutation_entropy(pd.Series(adx, index=df_tf.index), order=PE_ORDER, delay=PE_DELAY, window_size=PE_WINDOW)
        
        # === SYNC & LOOKAHEAD PREVENTION ===
        # Data at 'Time T' in higher TF is only available at 'Time T + Delta'
        # We shift the index FORWARD by delta.
        # e.g. H1 bar opened at 10:00. Close/High/Low/ADX is known at 11:00.
        # So we shift index to 11:00.
        # M1 bars at 10:01..10:59 will 'ffill' from the PREVIOUS known data (09:00 bar, avail at 10:00).
        # This is strictly correct.
        
        def sync(series, col_name):
            s = pd.Series(series, index=df_tf.index)
            s.index = s.index + tf_info['delta'] # SHIFT
            aligned = s.reindex(df_master.index, method='ffill')
            X[f"{name}_{col_name}"] = aligned
            
        sync(pe_ema, "PE_EMA")
        sync(adx, "ADX")
        sync(plus_di, "P_DI")
        sync(minus_di, "M_DI")
        sync(pe_adx, "PE_ADX")
        
    print("🎯 Generating Labels...")
    # Target: Next Candle Direction (1 if Close[t+1] > Close[t], else 0)
    # Shift -1 looks into future. This is allowed for TARGET generation (y), not Features (X).
    future_close = df_master['close'].shift(-1)
    y = (future_close > df_master['close']).astype(int)
    
    # Drop NaNs
    # M1 features usually ready instantly, but H4 features ready only after 4 hours.
    # The first few rows will be NaN.
    # The last row `y` will be NaN / Invalid (no future).
    
    data = X.copy()
    data['target'] = y
    
    # Remove last row (no target)
    data = data.iloc[:-1]
    
    # Drop rows with NaNs (warming up periods)
    data.dropna(inplace=True)
    
    print(f"✅ Final Dataset: {data.shape[0]} samples, {data.shape[1]-1} features.")
    
    return data.drop(columns=['target']), data['target']

def train_model():
    X, y = prepare_dataset()
    if X is None: return
    
    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split (Time Series Split - No Shuffle!)
    # Train on Past, Test on Future
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    print(f"🧠 Training MLPClassifier on {len(X_train)} samples...")
    # A simple but capable network
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
    clf.fit(X_train, y_train)
    
    print("📊 Evaluating on Test Set (Future Data)...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏆 TEST ACCURACY: {acc:.2%}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance (Proxy via weights? Hard for MLP. Just saving Model)
    joblib.dump(clf, "trend_mlp.pkl")
    joblib.dump(scaler, "trend_scaler.pkl")
    print("💾 Model saved to trend_mlp.pkl")

if __name__ == "__main__":
    train_model()
