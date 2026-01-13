import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
import optuna
import os
import sys

# --- CONFIGURATION ---
XAU_FILE = "data/Exness_XAUUSDm_2024.csv" 
DXY_FILE = "data/Exness_DXYm_2024.csv"
CACHE_FILE = "data/aligned_gold_dxy_cache.parquet" # <--- The Speed File
TIMEFRAME = "1H"
PREDICT_WINDOW = 12 

# ==========================================
# 1. DATA LOADING & CACHING
# ==========================================
# ==========================================
# 1. DATA LOADING & CACHING (TICK DATA FIX)
# ==========================================
def load_data(force_rebuild=False):
    # A. Check Cache
    if os.path.exists(CACHE_FILE) and not force_rebuild:
        print(f"⚡ CACHE FOUND: Loading data from {CACHE_FILE}...")
        try:
            return pd.read_parquet(CACHE_FILE)
        except Exception as e:
            print(f"⚠️ Cache corrupted, rebuilding... ({e})")

    # B. Process Tick Data
    print("🐢 No cache found. Processing TICK CSVs (First Run)...")
    
    if not os.path.exists(XAU_FILE):
        print(f"❌ Error: File {XAU_FILE} not found.")
        sys.exit()

    # 1. Read Tick Data
    # We use engine='python' and sep=None to auto-detect tabs/commas
    print(f"   -> Reading {XAU_FILE}...")
    try:
        df_ticks = pd.read_csv(XAU_FILE, sep=None, engine='python')
        
        # Normalize column names
        df_ticks.columns = [c.lower().strip() for c in df_ticks.columns]
        
        # Verify we have what we need
        if 'timestamp' not in df_ticks.columns or 'bid' not in df_ticks.columns:
            print(f"❌ Error: columns 'timestamp' and 'bid' required. Found: {list(df_ticks.columns)}")
            sys.exit()
            
        # 2. Parse Timestamp
        print("   -> Parsing Timestamps...")
        df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp'])
        df_ticks.set_index('timestamp', inplace=True)
        
        # 3. Resample Ticks to OHLC Candles
        print(f"   -> Resampling Ticks to {TIMEFRAME} Candles...")
        
        # We use the BID price to build the candle
        df_xau = df_ticks['bid'].resample(TIMEFRAME).agg('ohlc')
        
        # Rename columns to standard lowercase
        df_xau.columns = ['open', 'high', 'low', 'close']
        
        # Remove empty periods (weekends)
        df_xau.dropna(inplace=True)
        
        print(f"   ✅ Created {len(df_xau)} candles from tick data.")

    except Exception as e:
        print(f"❌ Error reading tick data: {e}")
        sys.exit()

    # 4. Load or Synthesize DXY
    print("   -> Aligning DXY...")
    df_dxy = df_xau.copy()
    
    # Simulate DXY Inverse (Replace with real DXY loading if you have it)
    noise = np.random.normal(0, 5, size=len(df_xau))
    df_dxy['close'] = (1 / df_xau['close']) * 200000 + noise 
    
    # 5. Merge
    df = pd.DataFrame(index=df_xau.index)
    df['xau_close'] = df_xau['close']
    df['dxy_close'] = df_dxy['close']
    
    # 6. Save Cache
    print(f"💾 Saving cache to {CACHE_FILE}...")
    if not os.path.exists("data"): os.makedirs("data")
    df.to_parquet(CACHE_FILE)
    
    return df

# ==========================================
# 2. FEATURE ENGINEERING (Re-runs every time)
# ==========================================
# We don't cache this part usually, so you can tweak parameters 
# (like period=14 to period=20) without deleting the cache file.
def engineer_features(df):
    print("[*] Calculating Statistics & Features...")
    
    # -- 1. Basic Returns --
    df['xau_ret'] = df['xau_close'].pct_change()
    df['dxy_ret'] = df['dxy_close'].pct_change()
    
    # -- 2. Rolling Correlation --
    windows = [10, 30, 60]
    for w in windows:
        df[f'corr_{w}'] = df['xau_ret'].rolling(w).corr(df['dxy_ret'])
        
    # -- 3. Z-Score Spread (Mean Reversion) --
    def zscore(series, window):
        return (series - series.rolling(window).mean()) / series.rolling(window).std()
    
    df['xau_z'] = zscore(df['xau_close'], 50)
    df['dxy_z'] = zscore(df['dxy_close'], 50)
    df['z_spread'] = df['xau_z'] - df['dxy_z'] 
    
    # -- 4. Volatility Ratio --
    df['xau_vol'] = df['xau_ret'].rolling(20).std()
    df['dxy_vol'] = df['dxy_ret'].rolling(20).std()
    df['vol_ratio'] = df['xau_vol'] / (df['dxy_vol'] + 1e-9)
    
    # -- 5. Technicals --
    df['rsi_xau'] = talib.RSI(df['xau_close'], timeperiod=14)
    
    # -- 6. TARGET: Future Returns (Forward Looking) --
    df['target_return'] = df['xau_close'].shift(-PREDICT_WINDOW) - df['xau_close']
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 3. FEATURE SELECTION (Decision Tree)
# ==========================================
def analyze_importance(df):
    print("\n[*] Training Decision Tree to find best features...")
    
    # Exclude raw price columns, only keep features
    feature_cols = [c for c in df.columns if c not in ['xau_close', 'dxy_close', 'target_return', 'xau_ret', 'dxy_ret', 'xau_z', 'dxy_z']]
    
    X = df[feature_cols]
    y = df['target_return']
    
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    top_5 = importances.head(5)
    print("--- 🏆 TOP PREDICTIVE FEATURES ---")
    print(top_5)
    
    # Return list of top feature names
    return top_5.index.tolist()

# ==========================================
# 4. OPTUNA OPTIMIZATION
# ==========================================
def optimize_strategy(df, top_features):
    print("\n[*] Starting Optuna Optimization...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Reduce clutter
    
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    
    def objective(trial):
        # Dynamic Thresholds for the top identified features
        f1 = top_features[0]
        
        # Optuna suggests a threshold for the top feature
        # We assume Mean Reversion: Buy if feature is Extreme (Low)
        thresh1 = trial.suggest_float("thresh1", train_df[f1].min(), train_df[f1].max())
        
        # Logic: Enter LONG if Feature 1 < Threshold
        # (This logic might need inversion based on the specific feature type, 
        # but the optimizer will find the valid range regardless)
        
        entry_mask = (train_df[f1] < thresh1)
        
        # Add a filter: Only trade if correlation is NOT perfectly inverse?
        # Or RSI filter
        use_rsi = trial.suggest_categorical("use_rsi", [True, False])
        if use_rsi:
            rsi_max = trial.suggest_int("rsi_max", 20, 50)
            entry_mask &= (train_df['rsi_xau'] < rsi_max)

        entries = train_df[entry_mask]
        
        if len(entries) < 20: return -100 # Penalize inactive strategies
        
        # Simple Sharpe approximation
        # cost per trade spread estimate (approx $0.50 per trade)
        net_returns = entries['target_return'] - 0.50 
        
        sharpe = net_returns.mean() / (net_returns.std() + 1e-6)
        return sharpe

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    
    print("--- 🎯 BEST PARAMETERS ---")
    print(study.best_params)
    return study.best_params, top_features[0]

# ==========================================
# 5. BACKTEST
# ==========================================
def run_final_backtest(df, best_params, primary_feature):
    print("\n[*] Running Final Backtest...")
    
    thresh = best_params['thresh1']
    
    # Reconstruct Signal
    df['signal'] = 0
    mask = (df[primary_feature] < thresh)
    
    if best_params.get('use_rsi'):
        mask &= (df['rsi_xau'] < best_params['rsi_max'])
        
    df.loc[mask, 'signal'] = 1
    
    # Calculate Equity
    df['pnl'] = df['signal'] * (df['target_return'] - 0.50) # spread cost
    df['equity'] = df['pnl'].cumsum()
    
    train_size = int(len(df) * 0.7)
    
    # Plot
    fig = go.Figure()
    # Train Data (Blue)
    fig.add_trace(go.Scatter(x=df.index[:train_size], y=df['equity'][:train_size], 
                             mode='lines', name='Train Equity', line=dict(color='#00BFFF')))
    # Test Data (Orange)
    fig.add_trace(go.Scatter(x=df.index[train_size:], y=df['equity'][train_size:], 
                             mode='lines', name='Test Equity', line=dict(color='#FFA500')))
    
    fig.update_layout(
        title=f"Correlation Strategy | Trigger: {primary_feature} < {thresh:.2f}",
        template="plotly_dark",
        height=600
    )
    fig.show()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # 1. Load (Fast Cache)
    df_raw = load_data()
    
    # 2. Engineer (Always fresh)
    df = engineer_features(df_raw)
    
    # 3. Analyze
    top_features = analyze_importance(df)
    
    # 4. Optimize
    best_params, prim_feat = optimize_strategy(df, top_features)
    
    # 5. Visualize
    run_final_backtest(df, best_params, prim_feat)