import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import talib
import os
import optuna
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MomentumScalperOptimizer:
    """Optimize momentum scalper parameters using Optuna."""
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_data = None
        self.test_data = None
        
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
        """Fetch recent data and split into train/test."""
        # Use recent 3000 bars only (focus on current regime)
        total_bars = 30000
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, total_bars)
        if rates is None:
            return False
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Simple 70/30 train/test split
        train_size = int(0.7 * len(df))
        
        self.train_data = df.iloc[:train_size]
        self.test_data = df.iloc[train_size:]
        
        print(f"✅ Data loaded (Recent 3000 bars):")
        print(f"   Train: {len(self.train_data)} bars ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        print(f"   Test:  {len(self.test_data)} bars ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        return True
    
    def calculate_momentum_metrics(self, df):
        """Calculate price change and ADX."""
        metrics = pd.DataFrame(index=df.index)
        
        metrics['price_change_10'] = df['close'].pct_change(3) * 100
        metrics['abs_price_change_10'] = np.abs(df['close'].pct_change(10)) * 100
        
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        vol = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['adx'] = adx
        metrics['atr'] = vol
        
        metrics.fillna(0, inplace=True)
        return metrics
    
    def backtest_strategy(self, df, price_threshold, adx_threshold, stop_loss_pips, take_profit_pips, track_equity=False):
        """Run backtest with given parameters."""
        metrics = self.calculate_momentum_metrics(df)
        
        bullish_momentum = (metrics['price_change_10'] > price_threshold) & (metrics['adx'] > adx_threshold)
        bearish_momentum = (metrics['price_change_10'] < -price_threshold) & (metrics['adx'] > adx_threshold)
        
        balance = 10000
        position = None
        trades = []
        equity_curve = [] if track_equity else None
        
        for i in range(100, len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
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
            
            # Track equity
            if track_equity:
                current_equity = balance
                if position:
                    if position['direction'] == 'LONG':
                        pips_profit = (current_price - position['entry']) / 0.0001
                    else:
                        pips_profit = (position['entry'] - current_price) / 0.0001
                    current_equity += pips_profit * 0.0001 * position['size']
                
                equity_curve.append({
                    'time': current_time,
                    'equity': current_equity
                })
        
        # Calculate metrics
        if len(trades) == 0:
            return 0.0, 0, 0.0, equity_curve  # No trades = bad
        
        trades_df = pd.DataFrame(trades)
        
        # Sharpe ratio per trade
        returns = trades_df['profit'] / 10000
        sharpe_per_trade = returns.mean() / (returns.std() + 1e-10)
        
        # Total return
        total_return = (balance - 10000) / 10000
        
        return sharpe_per_trade, len(trades), total_return, equity_curve
    
    def objective(self, trial):
        """Optuna objective function - maximize Sharpe per trade."""
        # Suggest parameters
        price_threshold = trial.suggest_float('price_threshold', 0.05, 0.5, step=0.05)
        adx_threshold = trial.suggest_int('adx_threshold', 0, 35, step=5)
        stop_loss_pips = trial.suggest_int('stop_loss_pips', 5, 20, step=5)
        take_profit_pips = trial.suggest_int('take_profit_pips', 3, 15, step=2)
        vol_threshold = trial.suggest_float('vol_threshold', 0.0005, 0.001, step=0.0005)
        
        # Backtest on TRAIN set (not validation) - don't track equity for speed
        sharpe, num_trades, total_return, _ = self.backtest_strategy(
            self.train_data,
            price_threshold,
            adx_threshold,
            stop_loss_pips,
            take_profit_pips,
            track_equity=False  # Don't track equity during optimization for speed
        )
        
        # Penalize if too few trades
        if num_trades < 20:
            return -999.0
        
        # Optimize for Sharpe per trade
        return sharpe
    
    def optimize(self, n_trials=100):
        """Run Optuna optimization."""
        print(f"\n{'='*60}")
        print(f"  OPTUNA PARAMETER OPTIMIZATION")
        print(f"  Objective: Maximize Sharpe Ratio per Trade")
        print(f"  Using Recent 3000 Bars (Current Market Regime)")
        print(f"{'='*60}")
        print(f"\n🔍 Running {n_trials} trials on TRAIN set...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Best parameters
        best_params = study.best_params
        best_sharpe = study.best_value
        
        print(f"\n{'='*60}")
        print(f"  OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"\n🏆 Best Parameters:")
        print(f"   Price threshold:  {best_params['price_threshold']:.2f}%")
        print(f"   ADX threshold:    {best_params['adx_threshold']}")
        print(f"   Stop loss:        {best_params['stop_loss_pips']} pips")
        print(f"   Take profit:      {best_params['take_profit_pips']} pips")
        print(f"   Train Sharpe:     {best_sharpe:.4f}")
        
        # Test on train set (in-sample) with equity tracking
        print(f"\n📊 Performance on TRAIN set (In-Sample):")
        train_sharpe, train_trades, train_return, train_equity = self.backtest_strategy(
            self.train_data,
            best_params['price_threshold'],
            best_params['adx_threshold'],
            best_params['stop_loss_pips'],
            best_params['take_profit_pips'],
            track_equity=True
        )
        print(f"   Sharpe per trade: {train_sharpe:.4f}")
        print(f"   Total trades:     {train_trades}")
        print(f"   Return:           {train_return*100:.2f}%")
        
        # Test on unseen test set with equity tracking
        print(f"\n📊 Performance on TEST set (Out-of-Sample):")
        test_sharpe, test_trades, test_return, test_equity = self.backtest_strategy(
            self.test_data,
            best_params['price_threshold'],
            best_params['adx_threshold'],
            best_params['stop_loss_pips'],
            best_params['take_profit_pips'],
            track_equity=True
        )
        print(f"   Sharpe per trade: {test_sharpe:.4f}")
        print(f"   Total trades:     {test_trades}")
        print(f"   Return:           {test_return*100:.2f}%")
        
        # Generalization check
        print(f"\n🔍 Generalization Check:")
        if test_sharpe > 0 and train_sharpe > 0:
            ratio = test_sharpe / train_sharpe
            if ratio > 0.8:
                print(f"   ✅ Excellent! Test is {ratio*100:.1f}% of train performance")
            elif ratio > 0.6:
                print(f"   ✅ Good! Test is {ratio*100:.1f}% of train performance")
            else:
                print(f"   ⚠️  Test is only {ratio*100:.1f}% of train - may be overfit")
        else:
            print(f"   ⚠️  Negative Sharpe detected - strategy not profitable")
        
        # Plot optimization history and equity curves
        self.plot_optimization(study)
        self.plot_equity_curves(train_equity, test_equity)
        
        return best_params
    
    def plot_optimization(self, study):
        """Plot optimization history."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimization history
        trials = study.trials
        values = [t.value for t in trials if t.value is not None and t.value > -999]
        
        ax1.plot(values, linewidth=1, alpha=0.7)
        ax1.axhline(y=study.best_value, color='red', linestyle='--', label=f'Best: {study.best_value:.4f}')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Sharpe per Trade')
        ax1.set_title('Optimization History', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            ax2.barh(params, importances, color='steelblue')
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance', fontweight='bold')
            ax2.grid(alpha=0.3, axis='x')
        except:
            ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        filename = 'optuna_optimization.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Optimization chart saved: {filename}")
        plt.show()
    
    def plot_equity_curves(self, train_equity, test_equity):
        """Plot equity curves for train and test sets."""
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Train equity curve
        train_df = pd.DataFrame(train_equity)
        ax1.plot(train_df['time'], train_df['equity'], linewidth=2, color='#2196F3')
        ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Balance')
        ax1.fill_between(train_df['time'], 10000, train_df['equity'], 
                         where=(train_df['equity'] >= 10000), alpha=0.3, color='green')
        ax1.fill_between(train_df['time'], 10000, train_df['equity'], 
                         where=(train_df['equity'] < 10000), alpha=0.3, color='red')
        ax1.set_title('Train Set - Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Test equity curve
        test_df = pd.DataFrame(test_equity)
        ax2.plot(test_df['time'], test_df['equity'], linewidth=2, color='#FF9800')
        ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Balance')
        ax2.fill_between(test_df['time'], 10000, test_df['equity'], 
                         where=(test_df['equity'] >= 10000), alpha=0.3, color='green')
        ax2.fill_between(test_df['time'], 10000, test_df['equity'], 
                         where=(test_df['equity'] < 10000), alpha=0.3, color='red')
        ax2.set_title('Test Set - Equity Curve (Out-of-Sample)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Equity ($)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Train drawdown
        train_df['peak'] = train_df['equity'].cummax()
        train_df['drawdown'] = (train_df['equity'] - train_df['peak']) / train_df['peak'] * 100
        
        ax3.fill_between(train_df['time'], 0, train_df['drawdown'], alpha=0.3, color='red')
        ax3.plot(train_df['time'], train_df['drawdown'], linewidth=1, color='darkred')
        ax3.set_title('Train Set - Drawdown', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(alpha=0.3)
        
        # Test drawdown
        test_df['peak'] = test_df['equity'].cummax()
        test_df['drawdown'] = (test_df['equity'] - test_df['peak']) / test_df['peak'] * 100
        
        ax4.fill_between(test_df['time'], 0, test_df['drawdown'], alpha=0.3, color='red')
        ax4.plot(test_df['time'], test_df['drawdown'], linewidth=1, color='darkred')
        ax4.set_title('Test Set - Drawdown', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'equity_curves.png'
        plt.savefig(filename, dpi=150)
        print(f"📊 Equity curves saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  MOMENTUM SCALPER PARAMETER OPTIMIZATION")
    print("  Using Optuna to maximize Sharpe per trade")
    print("="*60)
    
    optimizer = MomentumScalperOptimizer(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not optimizer.init_mt5():
        return
    
    if not optimizer.fetch_data():
        mt5.shutdown()
        return
    
    # Run optimization
    best_params = optimizer.optimize(n_trials=100)
    
    print(f"\n💡 Use these parameters in your live trading!")
    
    mt5.shutdown()
    print(f"\n👋 Optimization Complete!")


if __name__ == "__main__":
    main()
