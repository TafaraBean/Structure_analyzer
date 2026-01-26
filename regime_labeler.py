import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

load_dotenv()

class RegimeLabelGenerator:
    """Generate ground truth labels for market regimes (Sideways/Trending)."""
    
    def __init__(self, symbol, timeframe, bars=5000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.df = None
        
    def init_mt5(self):
        """Initialize MT5 connection."""
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
        
        print(f"✅ Connected to MT5")
        return True
    
    def fetch_data(self):
        """Fetch historical data."""
        print(f"📊 Fetching {self.bars} bars for {self.symbol}...")
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.bars)
        
        if rates is None:
            print(f"❌ Failed to fetch data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars")
        self.df = df
        return df
    
    def method_adx(self, period=14, threshold=25):
        """
        Method 1: ADX-based labeling.
        
        Sideways: ADX < threshold
        Trending: ADX >= threshold
        """
        adx = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
        
        labels = pd.Series('sideways', index=self.df.index)
        labels[adx >= threshold] = 'trending'
        
        sideways_pct = (labels == 'sideways').sum() / len(labels) * 100
        
        return {
            'labels': labels,
            'adx': adx,
            'sideways_pct': sideways_pct,
            'method': 'ADX',
            'params': {'period': period, 'threshold': threshold}
        }
    
    def method_linear_regression(self, window=20, r2_threshold=0.3, slope_threshold=0.0001):
        """
        Method 2: Linear Regression Slope + R².
        
        Most robust method - uses both trend strength and linearity.
        Sideways: Low R² (< r2_threshold) OR Low slope
        """
        slopes = []
        r2_scores = []
        
        for i in range(window, len(self.df)):
            y = self.df['close'].iloc[i-window:i].values
            X = np.arange(window).reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            slope = abs(model.coef_[0])
            
            slopes.append(slope)
            r2_scores.append(r2)
        
        # Pad with NaN
        slopes = [np.nan] * window + slopes
        r2_scores = [np.nan] * window + r2_scores
        
        slopes_series = pd.Series(slopes, index=self.df.index)
        r2_series = pd.Series(r2_scores, index=self.df.index)
        
        # Label as sideways if low R² OR low slope
        labels = pd.Series('trending', index=self.df.index)
        labels[(r2_series < r2_threshold) | (slopes_series < slope_threshold)] = 'sideways'
        
        sideways_pct = (labels == 'sideways').sum() / len(labels) * 100
        
        return {
            'labels': labels,
            'r2': r2_series,
            'slope': slopes_series,
            'sideways_pct': sideways_pct,
            'method': 'Linear Regression',
            'params': {'window': window, 'r2_threshold': r2_threshold, 'slope_threshold': slope_threshold}
        }
    
    def method_volatility_ratio(self, window=20, threshold=0.5):
        """
        Method 3: Price Range / ATR Ratio.
        
        Sideways: Price range is small relative to ATR
        """
        atr = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        
        # Calculate rolling range
        rolling_range = self.df['high'].rolling(window).max() - self.df['low'].rolling(window).min()
        
        # Ratio of range to expected range (ATR * window)
        expected_range = atr * window
        ratio = rolling_range / expected_range
        
        labels = pd.Series('trending', index=self.df.index)
        labels[ratio < threshold] = 'sideways'
        
        sideways_pct = (labels == 'sideways').sum() / len(labels) * 100
        
        return {
            'labels': labels,
            'ratio': ratio,
            'sideways_pct': sideways_pct,
            'method': 'Volatility Ratio',
            'params': {'window': window, 'threshold': threshold}
        }
    
    def method_hurst_exponent(self, window=100):
        """
        Method 4: Hurst Exponent (Fractal Dimension).
        
        Sideways: H ≈ 0.5 (random walk)
        Trending: H > 0.5 (persistent) or H < 0.5 (mean reverting)
        """
        def hurst(ts):
            """Calculate Hurst exponent."""
            lags = range(2, min(20, len(ts)//2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            
            # Filter out zeros
            valid_idx = [i for i, t in enumerate(tau) if t > 0]
            if len(valid_idx) < 2:
                return 0.5
            
            lags_valid = [lags[i] for i in valid_idx]
            tau_valid = [tau[i] for i in valid_idx]
            
            poly = np.polyfit(np.log(lags_valid), np.log(tau_valid), 1)
            return poly[0]
        
        hurst_values = []
        
        for i in range(window, len(self.df)):
            ts = self.df['close'].iloc[i-window:i].values
            h = hurst(ts)
            hurst_values.append(h)
        
        # Pad with NaN
        hurst_values = [np.nan] * window + hurst_values
        hurst_series = pd.Series(hurst_values, index=self.df.index)
        
        # Sideways if H close to 0.5 (random walk)
        labels = pd.Series('trending', index=self.df.index)
        labels[(hurst_series > 0.45) & (hurst_series < 0.55)] = 'sideways'
        
        sideways_pct = (labels == 'sideways').sum() / len(labels) * 100
        
        return {
            'labels': labels,
            'hurst': hurst_series,
            'sideways_pct': sideways_pct,
            'method': 'Hurst Exponent',
            'params': {'window': window}
        }
    
    def method_consensus(self, adx_threshold=25, r2_threshold=0.3, vol_threshold=0.5):
        """
        Method 5: Multi-Indicator Consensus (BEST FOR ML).
        
        Combines ADX, Linear Regression, and Volatility Ratio.
        Labels as sideways only when multiple indicators agree.
        """
        # Get individual methods
        adx_result = self.method_adx(threshold=adx_threshold)
        lr_result = self.method_linear_regression(r2_threshold=r2_threshold)
        vol_result = self.method_volatility_ratio(threshold=vol_threshold)
        
        # Count votes for sideways
        votes = pd.DataFrame({
            'adx': (adx_result['labels'] == 'sideways').astype(int),
            'lr': (lr_result['labels'] == 'sideways').astype(int),
            'vol': (vol_result['labels'] == 'sideways').astype(int)
        })
        
        vote_count = votes.sum(axis=1)
        
        # Require at least 2 out of 3 to agree
        labels = pd.Series('trending', index=self.df.index)
        labels[vote_count >= 2] = 'sideways'
        
        sideways_pct = (labels == 'sideways').sum() / len(labels) * 100
        
        return {
            'labels': labels,
            'votes': vote_count,
            'adx': adx_result['adx'],
            'r2': lr_result['r2'],
            'vol_ratio': vol_result['ratio'],
            'sideways_pct': sideways_pct,
            'method': 'Consensus',
            'params': {'adx_threshold': adx_threshold, 'r2_threshold': r2_threshold, 'vol_threshold': vol_threshold}
        }
    
    def visualize_comparison(self, results_dict, sample_bars=500):
        """Visualize all labeling methods for comparison."""
        print(f"\n📊 Creating comparison visualization...")
        
        # Use last N bars for visualization
        start_idx = max(0, len(self.df) - sample_bars)
        df_sample = self.df.iloc[start_idx:]
        
        n_methods = len(results_dict)
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(16, 4 * (n_methods + 1)), sharex=True)
        
        fig.suptitle(f'Market Regime Labeling Methods Comparison - {self.symbol}', 
                     fontsize=14, color='white')
        
        # Panel 0: Price
        ax0 = axes[0]
        ax0.plot(df_sample.index, df_sample['close'], color='white', linewidth=1, label='Close Price')
        ax0.set_ylabel('Price', fontsize=10)
        ax0.legend(loc='upper left', fontsize=8)
        ax0.grid(True, alpha=0.2)
        ax0.set_title('Price Chart', fontsize=11, pad=10)
        
        # Panels for each method
        for i, (method_name, result) in enumerate(results_dict.items(), 1):
            ax = axes[i]
            
            labels_sample = result['labels'].iloc[start_idx:]
            
            # Highlight sideways zones
            sideways_mask = labels_sample == 'sideways'
            
            # Plot price with regime highlighting
            ax.plot(df_sample.index, df_sample['close'], color='cyan', linewidth=1, alpha=0.7)
            
            # Shade sideways zones
            for idx in df_sample.index:
                if sideways_mask.loc[idx]:
                    ax.axvspan(idx, idx, color='yellow', alpha=0.2)
            
            ax.set_ylabel('Price', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_title(f'{method_name} | Sideways: {result["sideways_pct"]:.1f}%', 
                        fontsize=11, pad=10)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='yellow', alpha=0.3, label='Sideways'),
                Patch(facecolor='none', label='Trending')
            ]
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        axes[-1].set_xlabel('Time', fontsize=10)
        
        plt.tight_layout()
        
        filename = 'regime_labeling_comparison.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison chart saved as {filename}")
        
        plt.show()
        print(f"📈 Visualization displayed")
    
    def export_labels(self, result, filename='regime_labels.csv'):
        """Export labels to CSV for ML training."""
        export_df = self.df.copy()
        export_df['regime'] = result['labels']
        export_df['regime_binary'] = (result['labels'] == 'sideways').astype(int)
        
        export_df.to_csv(filename)
        print(f"\n💾 Labels exported to {filename}")
        print(f"   Sideways: {(export_df['regime_binary'] == 1).sum()} bars ({result['sideways_pct']:.1f}%)")
        print(f"   Trending: {(export_df['regime_binary'] == 0).sum()} bars ({100 - result['sideways_pct']:.1f}%)")

def main():
    print("="*60)
    print("  MARKET REGIME LABELING - GROUND TRUTH GENERATOR")
    print("  Symbol: EURUSDm | Timeframe: M15")
    print("="*60)
    
    labeler = RegimeLabelGenerator('EURUSDm', mt5.TIMEFRAME_M15, bars=2000)
    
    if not labeler.init_mt5():
        return
    
    labeler.fetch_data()
    
    if labeler.df is None:
        mt5.shutdown()
        return
    
    # Generate labels using all methods
    print(f"\n🔬 Generating labels using multiple methods...\n")
    
    results = {}
    
    print("1️⃣  ADX Method...")
    results['ADX'] = labeler.method_adx(period=14, threshold=25)
    print(f"   Sideways: {results['ADX']['sideways_pct']:.1f}%")
    
    print("\n2️⃣  Linear Regression Method...")
    results['Linear Regression'] = labeler.method_linear_regression(window=20, r2_threshold=0.3)
    print(f"   Sideways: {results['Linear Regression']['sideways_pct']:.1f}%")
    
    print("\n3️⃣  Volatility Ratio Method...")
    results['Volatility Ratio'] = labeler.method_volatility_ratio(window=20, threshold=0.5)
    print(f"   Sideways: {results['Volatility Ratio']['sideways_pct']:.1f}%")
    
    print("\n4️⃣  Hurst Exponent Method...")
    results['Hurst Exponent'] = labeler.method_hurst_exponent(window=100)
    print(f"   Sideways: {results['Hurst Exponent']['sideways_pct']:.1f}%")
    
    print("\n5️⃣  Consensus Method (RECOMMENDED)...")
    results['Consensus'] = labeler.method_consensus(adx_threshold=25, r2_threshold=0.3, vol_threshold=0.5)
    print(f"   Sideways: {results['Consensus']['sideways_pct']:.1f}%")
    
    # Visualize comparison
    labeler.visualize_comparison(results, sample_bars=500)
    
    # Export best method (Consensus)
    labeler.export_labels(results['Consensus'], filename='regime_labels_consensus.csv')
    
    # Summary
    print(f"\n{'='*60}")
    print("  RECOMMENDATION")
    print(f"{'='*60}")
    print(f"\n✅ Best Method: CONSENSUS")
    print(f"   - Combines ADX, Linear Regression, and Volatility Ratio")
    print(f"   - Requires 2/3 indicators to agree")
    print(f"   - Most robust for ML training")
    print(f"   - Sideways: {results['Consensus']['sideways_pct']:.1f}%")
    print(f"\n📊 Alternative: LINEAR REGRESSION")
    print(f"   - Uses R² and slope")
    print(f"   - Good balance of accuracy and simplicity")
    print(f"   - Sideways: {results['Linear Regression']['sideways_pct']:.1f}%")
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    main()
