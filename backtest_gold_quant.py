import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import optuna
from backtesting import Backtest, Strategy
from numba import njit
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
DATA_LEN = 20000 
RISK_PCT = 0.01

# --- 1. NUMBA OPTIMIZED MATH KERNELS ---
@njit
def get_linreg_rolling(prices, window=7):
    """
    Calculates Rolling Slope and R-Squared for Linear Regression.
    Returns: (slopes, r_squareds)
    """
    n = len(prices)
    slopes = np.zeros(n)
    r_sqs = np.zeros(n)
    
    # X axis is just 0, 1, 2, ... window-1
    x = np.arange(window)
    x_mean = np.mean(x)
    x_diff_sq_sum = np.sum((x - x_mean)**2)
    
    for t in range(window, n):
        y = prices[t-window : t]
        y_mean = np.mean(y)
        
        # Slope
        numerator = np.sum((x - x_mean) * (y - y_mean))
        slope = numerator / x_diff_sq_sum
        slopes[t] = slope
        
        # R-Squared
        y_pred = slope * (x - x_mean) + y_mean
        ssr = np.sum((y_pred - y_mean)**2)
        sst = np.sum((y - y_mean)**2)
        
        if sst == 0:
            r2 = 0.0
        else:
            r2 = ssr / sst
        r_sqs[t] = r2
        
    return slopes, r_sqs

# --- 2. DATA LOADERS ---
def fetch_data():
    if not mt5.initialize():
        print("MT5 Start Failed")
        return None
        
    print(f"Fetching {DATA_LEN} M5 bars for {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, DATA_LEN)
    mt5.shutdown()
    
    if rates is None: return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    df.set_index('time', inplace=True)
    
    return df

# --- 3. STRATEGY CLASS ---
class MomentumSlopeV2(Strategy):
    # Optuna Parameters (Defaults)
    slope_thresh_bull = 0.5
    slope_thresh_bear = -0.5
    chandelier_mult = 3.0
    
    # Manually track SL to avoid library limitations
    current_sl = None 
    
    def init(self):
        # Register indicators for Plotting
        # These will appear in separate panels or overlay on chart
        self.slope = self.I(lambda: self.data.Slope, name="LinReg Slope", overlay=False)
        self.r2 = self.I(lambda: self.data.R2, name="R-Squared", overlay=False)
        self.rsi = self.I(lambda: self.data.RSI, name="RSI", overlay=False)
        self.atr = self.I(lambda: self.data.ATR, name="ATR", overlay=False)
        
        # We can also plot the Chandelier lines for visualization (optional but tricky with dynamic SL)
        
    def next(self):
        # 1. TIME FILTER
        current_time = self.data.index[-1]
        hour = current_time.hour
        
        # HARD EXIT at 21:00
        if hour >= 21:
            if self.position:
                self.position.close()
                self.current_sl = None
            return
            
        is_trading_time = (8 <= hour < 20)
        
        # 2. INDICATORS
        price = self.data.Close[-1]
        slope = self.slope[-1]
        r2 = self.r2[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        
        # 3. MANAGE EXITS (Chandelier Trailing)
        if self.position:
            # Check Manual Stop Hit
            if self.position.is_long:
                if self.current_sl and low <= self.current_sl:
                    self.position.close()
                    self.current_sl = None
                    return
            elif self.position.is_short:
                if self.current_sl and high >= self.current_sl:
                    self.position.close()
                    self.current_sl = None
                    return
            
            # Update Trail
            dist = atr * self.chandelier_mult
            if self.position.is_long:
                new_sl = high - dist
                if self.current_sl is None or new_sl > self.current_sl:
                    self.current_sl = new_sl
            elif self.position.is_short:
                new_sl = low + dist
                if self.current_sl is None or new_sl < self.current_sl:
                    self.current_sl = new_sl
                    
        else:
            # NO POSITION
            self.current_sl = None
            if not is_trading_time: return
            
            # 4. ENTRY LOGIC
            # Quality Filter: R2 must be high (Clean Trend)
            if r2 < 0.8: return
            
            # Long
            if slope > self.slope_thresh_bull and rsi > 50:
                self.buy()
                self.current_sl = price - (atr * self.chandelier_mult)
                
            # Short
            elif slope < self.slope_thresh_bear and rsi < 50:
                self.sell()
                self.current_sl = price + (atr * self.chandelier_mult)

# --- 4. OPTUNA OBJECTIVE ---
def objective(trial, raw_df):
    slope_intensity = trial.suggest_float("slope_intensity", 0.05, 3.0) 
    atr_mult = trial.suggest_float("atr_mult", 1.5, 6.0)
    
    MomentumSlopeV2.slope_thresh_bull = slope_intensity
    MomentumSlopeV2.slope_thresh_bear = -slope_intensity
    MomentumSlopeV2.chandelier_mult = atr_mult
    
    bt = Backtest(raw_df, MomentumSlopeV2, cash=10000, commission=.0002)
    stats = bt.run()
    
    sharpe = stats['Sharpe Ratio']
    if stats['# Trades'] < 30: sharpe = -1.0 # Penalty for inactivity
        
    return sharpe if not np.isnan(sharpe) else -1.0

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Fetch Data
    data = fetch_data()
    
    if data is not None:
        print("Calculating Momentum Indicators (Slope, R2)...")
        import pandas_ta as ta
        
        # Numba Calc
        close_arr = data['Close'].to_numpy()
        slopes, r2s = get_linreg_rolling(close_arr, window=7)
        data['Slope'] = slopes
        data['R2'] = r2s
        
        # Standard Indicators
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data.dropna(inplace=True)
        
        # B. Optimize
        print(f"\n🔎 Starting Optuna V2 (Momentum Slope) - 100 Trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, data), n_trials=100)
        
        print("\n" + "="*40)
        print("🏆 BEST V2 PARAMETERS")
        print("="*40)
        print(study.best_params)
        print(f"Best Sharpe: {study.best_value:.2f}")
        
        # C. Re-Run Best & Plot
        print("\n📉 Visualizing Best Strategy...")
        bp = study.best_params
        MomentumSlopeV2.slope_thresh_bull = bp['slope_intensity']
        MomentumSlopeV2.slope_thresh_bear = -bp['slope_intensity']
        MomentumSlopeV2.chandelier_mult = bp['atr_mult']
        
        bt = Backtest(data, MomentumSlopeV2, cash=10000, commission=.0002)
        stats = bt.run()
        print(stats)
        
        # The plot will show Price, Equity, and the Indicators we registered with self.I()
        bt.plot()
