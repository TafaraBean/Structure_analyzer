import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from itertools import product
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

load_dotenv()

class ScalpingOptimizer:
    def __init__(self, symbol, timeframe, bars=5000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.best_sharpe = -np.inf
        self.best_params = None
        self.results = []
        
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
        return df
    
    def strategy_mean_reversion(self, df, bb_period, bb_std, rsi_period, rsi_oversold, rsi_overbought):
        """Mean reversion scalping strategy."""
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
        
        # RSI
        rsi = talib.RSI(df['close'], timeperiod=rsi_period)
        
        # Signals
        buy_signal = (df['close'] <= lower) & (rsi < rsi_oversold)
        sell_signal = (df['close'] >= upper) & (rsi > rsi_overbought)
        
        return buy_signal, sell_signal
    
    def strategy_momentum(self, df, fast_ema, slow_ema, atr_period, atr_mult):
        """Momentum scalping strategy."""
        # EMAs
        ema_fast = talib.EMA(df['close'], timeperiod=fast_ema)
        ema_slow = talib.EMA(df['close'], timeperiod=slow_ema)
        
        # ATR for volatility filter
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        atr_threshold = atr.rolling(50).mean() * atr_mult
        
        # Signals
        buy_signal = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1)) & (atr > atr_threshold)
        sell_signal = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1)) & (atr > atr_threshold)
        
        return buy_signal, sell_signal
    
    def strategy_breakout(self, df, lookback, atr_period, breakout_mult):
        """Breakout scalping strategy."""
        # High/Low breakout levels
        high_level = df['high'].rolling(window=lookback).max()
        low_level = df['low'].rolling(window=lookback).min()
        
        # ATR
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        
        # Signals
        buy_signal = (df['close'] > high_level.shift(1)) & (atr > atr.rolling(50).mean() * breakout_mult)
        sell_signal = (df['close'] < low_level.shift(1)) & (atr > atr.rolling(50).mean() * breakout_mult)
        
        return buy_signal, sell_signal
    
    def backtest(self, df, buy_signals, sell_signals, tp_pips, sl_pips, lot_size=0.01):
        """Backtest strategy with fixed TP/SL."""
        balance = 10000
        trades = []
        position = None
        
        pip_value = 0.0001  # For EURUSD
        
        for i in range(len(df)):
            if position is not None:
                # Check exit conditions
                current_price = df['close'].iloc[i]
                
                if position['type'] == 'buy':
                    if current_price >= position['tp'] or current_price <= position['sl']:
                        exit_price = position['tp'] if current_price >= position['tp'] else position['sl']
                        pnl = (exit_price - position['entry']) / pip_value * lot_size
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': df.index[i],
                            'type': 'buy',
                            'entry': position['entry'],
                            'exit': exit_price,
                            'pnl': pnl
                        })
                        
                        balance += pnl
                        position = None
                
                elif position['type'] == 'sell':
                    if current_price <= position['tp'] or current_price >= position['sl']:
                        exit_price = position['tp'] if current_price <= position['tp'] else position['sl']
                        pnl = (position['entry'] - exit_price) / pip_value * lot_size
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': df.index[i],
                            'type': 'sell',
                            'entry': position['entry'],
                            'exit': exit_price,
                            'pnl': pnl
                        })
                        
                        balance += pnl
                        position = None
            
            # Check for new entries
            if position is None:
                if buy_signals.iloc[i]:
                    entry_price = df['close'].iloc[i]
                    position = {
                        'type': 'buy',
                        'entry': entry_price,
                        'entry_time': df.index[i],
                        'tp': entry_price + (tp_pips * pip_value),
                        'sl': entry_price - (sl_pips * pip_value)
                    }
                
                elif sell_signals.iloc[i]:
                    entry_price = df['close'].iloc[i]
                    position = {
                        'type': 'sell',
                        'entry': entry_price,
                        'entry_time': df.index[i],
                        'tp': entry_price - (tp_pips * pip_value),
                        'sl': entry_price + (sl_pips * pip_value)
                    }
        
        return trades, balance
    
    def calculate_metrics(self, trades):
        """Calculate performance metrics."""
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'sharpe_per_trade': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0
            }
        
        pnls = [t['pnl'] for t in trades]
        
        total_trades = len(trades)
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1
        
        # Sharpe per trade
        sharpe_per_trade = avg_pnl / std_pnl if std_pnl > 0 else 0
        
        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'sharpe_per_trade': sharpe_per_trade,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'std_pnl': std_pnl
        }
    
    def optimize(self, target_sharpe=0.2, max_iterations=1000):
        """Run optimization to find best parameters."""
        print(f"\n🎯 Target: Sharpe per trade >= {target_sharpe}")
        print(f"🔄 Max iterations: {max_iterations}\n")
        
        # Parameter grids for different strategies
        strategies = [
            {
                'name': 'mean_reversion',
                'func': self.strategy_mean_reversion,
                'params': {
                    'bb_period': [10, 15, 20],
                    'bb_std': [1.5, 2.0, 2.5],
                    'rsi_period': [7, 10, 14],
                    'rsi_oversold': [25, 30, 35],
                    'rsi_overbought': [65, 70, 75]
                }
            },
            {
                'name': 'momentum',
                'func': self.strategy_momentum,
                'params': {
                    'fast_ema': [3, 5, 8],
                    'slow_ema': [13, 21, 34],
                    'atr_period': [10, 14, 20],
                    'atr_mult': [0.5, 1.0, 1.5]
                }
            },
            {
                'name': 'breakout',
                'func': self.strategy_breakout,
                'params': {
                    'lookback': [5, 10, 15, 20],
                    'atr_period': [10, 14, 20],
                    'breakout_mult': [0.5, 1.0, 1.5]
                }
            }
        ]
        
        # TP/SL combinations
        tp_sl_combos = [
            (5, 3), (8, 5), (10, 5), (15, 8), (20, 10)
        ]
        
        iteration = 0
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Testing Strategy: {strategy['name'].upper()}")
            print(f"{'='*60}")
            
            # Generate parameter combinations
            param_names = list(strategy['params'].keys())
            param_values = [strategy['params'][k] for k in param_names]
            param_combos = list(product(*param_values))
            
            for params in param_combos:
                for tp_pips, sl_pips in tp_sl_combos:
                    iteration += 1
                    
                    if iteration > max_iterations:
                        print(f"\n⚠️  Reached max iterations ({max_iterations})")
                        return self.best_params, self.best_sharpe
                    
                    # Create parameter dict
                    param_dict = dict(zip(param_names, params))
                    
                    # Generate signals
                    buy_sig, sell_sig = strategy['func'](self.df, **param_dict)
                    
                    # Backtest
                    trades, final_balance = self.backtest(self.df, buy_sig, sell_sig, tp_pips, sl_pips)
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(trades)
                    
                    # Store result
                    result = {
                        'iteration': iteration,
                        'strategy': strategy['name'],
                        'params': param_dict,
                        'tp_pips': tp_pips,
                        'sl_pips': sl_pips,
                        **metrics
                    }
                    
                    self.results.append(result)
                    
                    # Check if target achieved
                    if metrics['sharpe_per_trade'] >= target_sharpe and metrics['total_trades'] >= 20:
                        print(f"\n🎉 TARGET ACHIEVED! Iteration {iteration}")
                        print(f"   Strategy: {strategy['name']}")
                        print(f"   Sharpe per trade: {metrics['sharpe_per_trade']:.4f}")
                        print(f"   Total trades: {metrics['total_trades']}")
                        print(f"   Win rate: {metrics['win_rate']:.2%}")
                        print(f"   Avg PnL: ${metrics['avg_pnl']:.2f}")
                        print(f"   Total PnL: ${metrics['total_pnl']:.2f}")
                        print(f"   Parameters: {param_dict}")
                        print(f"   TP/SL: {tp_pips}/{sl_pips} pips")
                        
                        self.best_sharpe = metrics['sharpe_per_trade']
                        self.best_params = result
                        
                        return result, metrics['sharpe_per_trade']
                    
                    # Update best
                    if metrics['sharpe_per_trade'] > self.best_sharpe and metrics['total_trades'] >= 20:
                        self.best_sharpe = metrics['sharpe_per_trade']
                        self.best_params = result
                        
                        print(f"✨ New best! Iter {iteration} | Sharpe: {metrics['sharpe_per_trade']:.4f} | "
                              f"Trades: {metrics['total_trades']} | Strategy: {strategy['name']}")
                    
                    # Progress update
                    if iteration % 50 == 0:
                        print(f"⏳ Iteration {iteration}/{max_iterations} | Best Sharpe: {self.best_sharpe:.4f}")
        
        print(f"\n⚠️  Target not reached. Best Sharpe: {self.best_sharpe:.4f}")
        return self.best_params, self.best_sharpe
    
    def save_results(self, filename='scalping_optimization_results.json'):
        """Save all results to JSON."""
        with open(filename, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_sharpe': self.best_sharpe,
                'all_results': self.results
            }, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filename}")
    
    def plot_equity_curve(self, trades, title="Equity Curve"):
        """Plot equity curve from trades."""
        if not trades:
            print("⚠️  No trades to plot")
            return
        
        # Build equity curve
        equity = [10000]  # Starting balance
        trade_numbers = [0]
        
        for i, trade in enumerate(trades, 1):
            equity.append(equity[-1] + trade['pnl'])
            trade_numbers.append(i)
        
        # Calculate drawdown
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        
        # Create plot
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        fig.suptitle(title, fontsize=14, color='white')
        
        # Panel 1: Equity Curve
        ax1.plot(trade_numbers, equity, color='cyan', linewidth=2, label='Equity')
        ax1.axhline(y=10000, color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='Starting Balance')
        
        # Fill areas
        ax1.fill_between(trade_numbers, 10000, equity, 
                        where=[e >= 10000 for e in equity], 
                        color='green', alpha=0.2, interpolate=True)
        ax1.fill_between(trade_numbers, 10000, equity, 
                        where=[e < 10000 for e in equity], 
                        color='red', alpha=0.2, interpolate=True)
        
        ax1.set_ylabel('Equity ($)', fontsize=10)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.2)
        ax1.set_title(f'Equity Curve | Final: ${equity[-1]:.2f} | Return: {((equity[-1]-10000)/10000*100):.2f}%', 
                     fontsize=11, pad=10)
        
        # Panel 2: Drawdown
        ax2.fill_between(trade_numbers, 0, drawdown, color='red', alpha=0.4, label='Drawdown')
        ax2.plot(trade_numbers, drawdown, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_xlabel('Trade Number', fontsize=10)
        ax2.legend(loc='lower left', fontsize=9)
        ax2.grid(True, alpha=0.2)
        ax2.set_title(f'Drawdown | Max: {drawdown.min():.2f}%', fontsize=11, pad=10)
        
        plt.tight_layout()
        
        # Save
        filename = 'equity_curve.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n📈 Equity curve saved as {filename}")
        
        plt.show()
        print(f"📊 Chart displayed")

def main():
    print("="*60)
    print("  BLACKBOX SCALPING STRATEGY OPTIMIZER")
    print("  Symbol: EURUSDm | Timeframe: M1")
    print("  Target: Sharpe per trade >= 0.2")
    print("="*60)
    
    optimizer = ScalpingOptimizer('EURUSDm', mt5.TIMEFRAME_M1, bars=5000)
    
    if not optimizer.init_mt5():
        return
    
    optimizer.df = optimizer.fetch_data()
    
    if optimizer.df is None:
        mt5.shutdown()
        return
    
    # Run optimization
    best_result, best_sharpe = optimizer.optimize(target_sharpe=0.2, max_iterations=1000)
    
    # Save results
    optimizer.save_results()
    
    # Final summary
    print(f"\n{'='*60}")
    print("  OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    
    if best_result:
        print(f"\nBest Configuration:")
        print(f"  Strategy: {best_result['strategy']}")
        print(f"  Sharpe per trade: {best_result['sharpe_per_trade']:.4f}")
        print(f"  Total trades: {best_result['total_trades']}")
        print(f"  Win rate: {best_result['win_rate']:.2%}")
        print(f"  Avg PnL per trade: ${best_result['avg_pnl']:.2f}")
        print(f"  Total PnL: ${best_result['total_pnl']:.2f}")
        print(f"  TP/SL: {best_result['tp_pips']}/{best_result['sl_pips']} pips")
        print(f"\n  Parameters:")
        for k, v in best_result['params'].items():
            print(f"    {k}: {v}")
        
        # Re-run best strategy to get trades for equity curve
        print(f"\n📊 Generating equity curve...")
        
        if best_result['strategy'] == 'mean_reversion':
            buy_sig, sell_sig = optimizer.strategy_mean_reversion(optimizer.df, **best_result['params'])
        elif best_result['strategy'] == 'momentum':
            buy_sig, sell_sig = optimizer.strategy_momentum(optimizer.df, **best_result['params'])
        else:  # breakout
            buy_sig, sell_sig = optimizer.strategy_breakout(optimizer.df, **best_result['params'])
        
        trades, _ = optimizer.backtest(optimizer.df, buy_sig, sell_sig, 
                                      best_result['tp_pips'], best_result['sl_pips'])
        
        # Plot equity curve
        title = f"Equity Curve - {best_result['strategy'].title()} Strategy\n" \
                f"Sharpe: {best_result['sharpe_per_trade']:.4f} | Win Rate: {best_result['win_rate']:.1%} | " \
                f"Trades: {best_result['total_trades']}"
        
        optimizer.plot_equity_curve(trades, title=title)
    
    mt5.shutdown()
    print(f"\n👋 Optimization complete")

if __name__ == "__main__":
    main()
