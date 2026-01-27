import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import talib
import os
import optuna
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class WalkForwardOptimizer:
    """Walk-forward optimization for momentum scalper."""
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.symbol = symbol
        self.timeframe = timeframe
        self.all_data = None
        self.walk_forward_results = []
        
    def init_mt5(self):
        """Initialize MT5."""
        path = os.getenv("MT5_PATH")
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        params = {}
        if path: params["path"] = path
        
        if not mt5.initialize(**params):
            print(f"❌ MT5 Init failed")
            return False
        if login and password and server:
            mt5.login(login=int(login), password=password, server=server)
        print(f"✅ Connected to MT5")
        return True
    
    def fetch_data(self):
        """Fetch data."""
        total_bars = 5000
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, total_bars)
        if rates is None:
            return False
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        self.all_data = df
        
        print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return True
    
    def calculate_momentum_metrics(self, df):
        """Calculate price change and ADX."""
        metrics = pd.DataFrame(index=df.index)
        
        metrics['price_change_10'] = df['close'].pct_change(10) * 100
        metrics['abs_price_change_10'] = np.abs(df['close'].pct_change(10)) * 100
        
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['adx'] = adx
        
        metrics.fillna(0, inplace=True)
        return metrics
    
    def backtest_strategy(self, df, price_threshold, adx_threshold, stop_loss_pips, take_profit_pips):
        """Run backtest with given parameters."""
        metrics = self.calculate_momentum_metrics(df)
        
        bullish_momentum = (metrics['price_change_10'] > price_threshold) & (metrics['adx'] > adx_threshold)
        bearish_momentum = (metrics['price_change_10'] < -price_threshold) & (metrics['adx'] > adx_threshold)
        
        balance = 10000
        position = None
        trades = []
        
        for i in range(100, len(df)):
            current_price = df['close'].iloc[i]
            
            if position:
                if position['direction'] == 'LONG':
                    pips_profit = (current_price - position['entry']) / 0.0001
                else:
                    pips_profit = (position['entry'] - current_price) / 0.0001
                
                hit_tp = pips_profit >= take_profit_pips
                hit_sl = pips_profit <= -stop_loss_pips
                
                if hit_tp or hit_sl:
                    profit = pips_profit * 0.0001 * position['size']
                    balance += profit
                    
                    trades.append({
                        'pips': pips_profit,
                        'profit': profit
                    })
                    
                    position = None
            
            if position is None:
                if bullish_momentum.iloc[i]:
                    position = {
                        'entry': current_price,
                        'direction': 'LONG',
                        'size': 100000
                    }
                elif bearish_momentum.iloc[i]:
                    position = {
                        'entry': current_price,
                        'direction': 'SHORT',
                        'size': 100000
                    }
        
        if len(trades) == 0:
            return 0.0, 0, 0.0, 0.0
        
        trades_df = pd.DataFrame(trades)
        
        returns = trades_df['profit'] / 10000
        sharpe_per_trade = returns.mean() / (returns.std() + 1e-10)
        win_rate = len(trades_df[trades_df['pips'] > 0]) / len(trades_df)
        total_return = (balance - 10000) / 10000
        total_pips = trades_df['pips'].sum()
        
        return sharpe_per_trade, len(trades), total_return, total_pips
    
    def optimize_window(self, train_df, val_df, n_trials=50):
        """Optimize parameters for a single window."""
        
        def objective(trial):
            price_threshold = trial.suggest_float('price_threshold', 0.05, 0.5, step=0.05)
            adx_threshold = trial.suggest_int('adx_threshold', 15, 35, step=5)
            stop_loss_pips = trial.suggest_int('stop_loss_pips', 5, 20, step=5)
            take_profit_pips = trial.suggest_int('take_profit_pips', 3, 15, step=2)
            
            sharpe, num_trades, _, _ = self.backtest_strategy(
                val_df, price_threshold, adx_threshold, stop_loss_pips, take_profit_pips
            )
            
            if num_trades < 5:
                return -999.0
            
            return sharpe
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value
    
    def walk_forward_analysis(self, train_window=1000, test_window=500, step_size=250, n_trials=50):
        """
        Perform walk-forward optimization.
        
        Args:
            train_window: Size of training window (bars)
            test_window: Size of test window (bars)
            step_size: How many bars to step forward each iteration
            n_trials: Optuna trials per window
        """
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD OPTIMIZATION")
        print(f"{'='*60}")
        print(f"\n⚙️  Configuration:")
        print(f"   Train window: {train_window} bars")
        print(f"   Test window:  {test_window} bars")
        print(f"   Step size:    {step_size} bars")
        print(f"   Trials/window: {n_trials}")
        
        total_bars = len(self.all_data)
        current_start = 0
        window_num = 1
        
        all_test_results = []
        
        while current_start + train_window + test_window <= total_bars:
            train_end = current_start + train_window
            test_start = train_end
            test_end = test_start + test_window
            
            train_df = self.all_data.iloc[current_start:train_end]
            test_df = self.all_data.iloc[test_start:test_end]
            
            print(f"\n{'='*60}")
            print(f"  Window {window_num}")
            print(f"{'='*60}")
            print(f"Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} bars)")
            print(f"Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} bars)")
            
            # Optimize on train window
            print(f"🔍 Optimizing parameters...")
            best_params, train_sharpe = self.optimize_window(train_df, train_df, n_trials=n_trials)
            
            print(f"\n🏆 Best parameters:")
            print(f"   Price threshold:  {best_params['price_threshold']:.2f}%")
            print(f"   ADX threshold:    {best_params['adx_threshold']}")
            print(f"   Stop loss:        {best_params['stop_loss_pips']} pips")
            print(f"   Take profit:      {best_params['take_profit_pips']} pips")
            
            # Test on out-of-sample window
            test_sharpe, test_trades, test_return, test_pips = self.backtest_strategy(
                test_df,
                best_params['price_threshold'],
                best_params['adx_threshold'],
                best_params['stop_loss_pips'],
                best_params['take_profit_pips']
            )
            
            print(f"\n📊 Test Results:")
            print(f"   Sharpe per trade: {test_sharpe:.4f}")
            print(f"   Total trades:     {test_trades}")
            print(f"   Return:           {test_return*100:.2f}%")
            print(f"   Total pips:       {test_pips:.1f}")
            
            # Store results
            self.walk_forward_results.append({
                'window': window_num,
                'train_start': train_df.index[0],
                'train_end': train_df.index[-1],
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                'params': best_params,
                'train_sharpe': train_sharpe,
                'test_sharpe': test_sharpe,
                'test_trades': test_trades,
                'test_return': test_return,
                'test_pips': test_pips
            })
            
            all_test_results.append({
                'sharpe': test_sharpe,
                'return': test_return,
                'pips': test_pips
            })
            
            # Move forward
            current_start += step_size
            window_num += 1
        
        # Aggregate results
        self.print_summary(all_test_results)
        self.plot_walk_forward()
    
    def print_summary(self, all_test_results):
        """Print summary of walk-forward results."""
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD SUMMARY")
        print(f"{'='*60}")
        
        results_df = pd.DataFrame(all_test_results)
        
        print(f"\n📊 Aggregate Performance:")
        print(f"   Total windows:     {len(results_df)}")
        print(f"   Avg Sharpe:        {results_df['sharpe'].mean():.4f}")
        print(f"   Avg Return/window: {results_df['return'].mean()*100:.2f}%")
        print(f"   Total Return:      {results_df['return'].sum()*100:.2f}%")
        print(f"   Total Pips:        {results_df['pips'].sum():.1f}")
        print(f"   Positive windows:  {len(results_df[results_df['return'] > 0])} ({len(results_df[results_df['return'] > 0])/len(results_df)*100:.1f}%)")
        
        print(f"\n📈 Consistency:")
        print(f"   Best window:       {results_df['return'].max()*100:.2f}%")
        print(f"   Worst window:      {results_df['return'].min()*100:.2f}%")
        print(f"   Std dev:           {results_df['return'].std()*100:.2f}%")
    
    def plot_walk_forward(self):
        """Plot walk-forward results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        results_df = pd.DataFrame(self.walk_forward_results)
        
        # Returns per window
        ax1.bar(results_df['window'], results_df['test_return'] * 100, 
                color=['green' if x > 0 else 'red' for x in results_df['test_return']])
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Return per Window', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Sharpe per window
        ax2.plot(results_df['window'], results_df['test_sharpe'], marker='o', linewidth=2, color='blue')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Sharpe per Trade')
        ax2.set_title('Sharpe Ratio per Window', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Cumulative return
        cumulative_return = (1 + results_df['test_return']).cumprod() - 1
        ax3.plot(results_df['window'], cumulative_return * 100, linewidth=2, color='green')
        ax3.fill_between(results_df['window'], 0, cumulative_return * 100, alpha=0.3, color='green')
        ax3.set_xlabel('Window')
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_title('Cumulative Return Across Windows', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Parameter stability
        ax4.plot(results_df['window'], [p['price_threshold'] for p in results_df['params']], 
                marker='o', label='Price Threshold', linewidth=2)
        ax4.plot(results_df['window'], [p['adx_threshold']/10 for p in results_df['params']], 
                marker='s', label='ADX/10', linewidth=2)
        ax4.plot(results_df['window'], [p['stop_loss_pips'] for p in results_df['params']], 
                marker='^', label='Stop Loss', linewidth=2)
        ax4.plot(results_df['window'], [p['take_profit_pips'] for p in results_df['params']], 
                marker='v', label='Take Profit', linewidth=2)
        ax4.set_xlabel('Window')
        ax4.set_ylabel('Parameter Value')
        ax4.set_title('Parameter Evolution', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'walk_forward_analysis.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Walk-forward chart saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  WALK-FORWARD OPTIMIZATION")
    print("  Adaptive parameter optimization")
    print("="*60)
    
    optimizer = WalkForwardOptimizer(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not optimizer.init_mt5():
        return
    
    if not optimizer.fetch_data():
        mt5.shutdown()
        return
    
    # Run walk-forward analysis
    optimizer.walk_forward_analysis(
        train_window=1000,  # Train on 1000 bars
        test_window=500,    # Test on 500 bars
        step_size=250,      # Move forward 250 bars each time
        n_trials=50         # 50 Optuna trials per window
    )
    
    mt5.shutdown()
    print(f"\n👋 Walk-Forward Analysis Complete!")


if __name__ == "__main__":
    main()
