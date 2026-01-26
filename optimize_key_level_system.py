import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from itertools import product
from key_level_detector import KeyLevelDetector
from regime_labeler import RegimeLabelGenerator

load_dotenv()

class KeyLevelOptimizer:
    """
    Optimize key level trading system parameters.
    Target: Sharpe per trade >= 0.2, Sharpe ratio >= 1.8
    """
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15, bars=2000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.df = None
        self.key_levels = []
        self.best_config = None
        self.best_sharpe_per_trade = -999
        
    def init_mt5(self):
        """Initialize MT5."""
        path = os.getenv("MT5_PATH")
        login = os.getenv("MT5_LOGIN")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        params = {}
        if path: params["path"] = path
        
        if not mt5.initialize(**params):
            return False
        if login and password and server:
            mt5.login(login=int(login), password=password, server=server)
        return True
    
    def prepare_data(self):
        """Fetch data and detect regimes + key levels."""
        # Detect key levels
        level_detector = KeyLevelDetector(self.symbol, self.timeframe, self.bars)
        level_detector.df = level_detector.fetch_data()
        
        swing_points = level_detector.find_swing_points(window=5)
        self.key_levels = level_detector.find_key_levels_percentile(swing_points, n_levels=10, tolerance=0.0005)
        
        # Detect sideways regimes
        regime_labeler = RegimeLabelGenerator(self.symbol, self.timeframe, self.bars)
        regime_labeler.df = level_detector.df
        regime_result = regime_labeler.method_consensus()
        
        # Combine data
        self.df = level_detector.df.copy()
        self.df['regime'] = regime_result['labels']
        self.df['is_sideways'] = (regime_result['labels'] == 'sideways').astype(int)
        
        # Add technical indicators for filters
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=14)
        self.df['volume_ma'] = talib.SMA(self.df['tick_volume'], timeperiod=20)
        self.df['volume_ratio'] = self.df['tick_volume'] / self.df['volume_ma']
        
        return self.df
    
    def backtest_config(self, config):
        """
        Backtest a specific configuration.
        
        Config parameters:
        - tp_pips: Take profit
        - sl_pips: Stop loss
        - proximity_pips: Distance to level for entry
        - min_level_strength: Minimum touches for level
        - use_rsi_filter: Filter by RSI
        - rsi_buy_max: Max RSI for buy
        - rsi_sell_min: Min RSI for sell
        - use_volume_filter: Filter by volume
        - min_volume_ratio: Minimum volume ratio
        - use_strength_sizing: Use level strength for position sizing
        """
        pip_value = 0.0001
        tp_distance = config['tp_pips'] * pip_value
        sl_distance = config['sl_pips'] * pip_value
        
        trades = []
        signals = []
        
        for i in range(len(self.df)):
            # Only trade in sideways
            if not self.df['is_sideways'].iloc[i]:
                continue
            
            current_price = self.df['close'].iloc[i]
            current_low = self.df['low'].iloc[i]
            current_high = self.df['high'].iloc[i]
            current_rsi = self.df['rsi'].iloc[i]
            current_volume_ratio = self.df['volume_ratio'].iloc[i]
            
            # Check proximity to each key level
            for level in self.key_levels:
                if level['strength'] < config['min_level_strength']:
                    continue
                
                level_price = level['price']
                distance_pips = abs(current_price - level_price) / pip_value
                
                if distance_pips > config['proximity_pips']:
                    continue
                
                # Support level - buy signal
                if level['type'] == 'support' and current_low <= level_price * 1.0001:
                    # RSI filter
                    if config['use_rsi_filter'] and current_rsi > config['rsi_buy_max']:
                        continue
                    
                    # Volume filter
                    if config['use_volume_filter'] and current_volume_ratio < config['min_volume_ratio']:
                        continue
                    
                    # Position sizing based on level strength
                    if config['use_strength_sizing']:
                        lot_size = 0.01 * (level['strength'] / 10)  # Scale by strength
                    else:
                        lot_size = 0.01
                    
                    signals.append({
                        'idx': i,
                        'direction': 1,
                        'level_price': level_price,
                        'level_strength': level['strength'],
                        'lot_size': lot_size
                    })
                    break
                
                # Resistance level - sell signal
                elif level['type'] == 'resistance' and current_high >= level_price * 0.9999:
                    # RSI filter
                    if config['use_rsi_filter'] and current_rsi < config['rsi_sell_min']:
                        continue
                    
                    # Volume filter
                    if config['use_volume_filter'] and current_volume_ratio < config['min_volume_ratio']:
                        continue
                    
                    # Position sizing
                    if config['use_strength_sizing']:
                        lot_size = 0.01 * (level['strength'] / 10)
                    else:
                        lot_size = 0.01
                    
                    signals.append({
                        'idx': i,
                        'direction': -1,
                        'level_price': level_price,
                        'level_strength': level['strength'],
                        'lot_size': lot_size
                    })
                    break
        
        # Simulate trades
        for signal in signals:
            i = signal['idx']
            entry_price = self.df['close'].iloc[i]
            direction = signal['direction']
            lot_size = signal['lot_size']
            
            # Set TP and SL
            if direction == 1:
                tp_price = entry_price + tp_distance
                sl_price = entry_price - sl_distance
            else:
                tp_price = entry_price - tp_distance
                sl_price = entry_price + sl_distance
            
            # Simulate trade
            exit_price = None
            exit_reason = None
            
            for j in range(i + 1, min(i + 100, len(self.df))):
                bar_high = self.df['high'].iloc[j]
                bar_low = self.df['low'].iloc[j]
                
                if direction == 1:
                    if bar_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                        break
                    elif bar_low <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'
                        break
                else:
                    if bar_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                        break
                    elif bar_high >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'
                        break
            
            if exit_price is None:
                exit_price = self.df['close'].iloc[min(i + 100, len(self.df) - 1)]
                exit_reason = 'Timeout'
            
            # Calculate PnL
            if direction == 1:
                pnl = (exit_price - entry_price) / pip_value * 10 * lot_size
            else:
                pnl = (entry_price - exit_price) / pip_value * 10 * lot_size
            
            trades.append({
                'pnl': pnl,
                'exit_reason': exit_reason,
                'lot_size': lot_size
            })
        
        # Calculate metrics
        if len(trades) == 0:
            return None
        
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        std_pnl = trades_df['pnl'].std()
        
        if std_pnl == 0:
            return None
        
        sharpe_per_trade = avg_pnl / std_pnl if std_pnl > 0 else 0
        
        # Sharpe ratio (annualized)
        returns = trades_df['pnl'] / 10000  # As fraction of account
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'sharpe_per_trade': sharpe_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'config': config
        }
    
    def optimize(self, max_iterations=1000):
        """
        Optimize parameters until targets are met.
        Target: Sharpe per trade >= 0.2, Sharpe ratio >= 1.8
        """
        print(f"\n🎯 Target: Sharpe per trade >= 0.2, Sharpe ratio >= 1.8")
        print(f"🔄 Max iterations: {max_iterations}\n")
        
        # Parameter grid
        tp_sl_ratios = [(8, 5), (10, 5), (12, 6), (15, 7), (20, 10)]
        proximity_pips = [3, 5, 7, 10]
        min_level_strengths = [3, 5, 7, 10]
        rsi_configs = [
            {'use': False},
            {'use': True, 'buy_max': 50, 'sell_min': 50},
            {'use': True, 'buy_max': 40, 'sell_min': 60},
            {'use': True, 'buy_max': 35, 'sell_min': 65}
        ]
        volume_configs = [
            {'use': False},
            {'use': True, 'min_ratio': 1.0},
            {'use': True, 'min_ratio': 1.2}
        ]
        strength_sizing = [False, True]
        
        iteration = 0
        
        for tp, sl in tp_sl_ratios:
            for prox in proximity_pips:
                for min_str in min_level_strengths:
                    for rsi_cfg in rsi_configs:
                        for vol_cfg in volume_configs:
                            for use_sizing in strength_sizing:
                                iteration += 1
                                
                                if iteration > max_iterations:
                                    print(f"\n⚠️  Reached max iterations ({max_iterations})")
                                    return self.best_config
                                
                                config = {
                                    'tp_pips': tp,
                                    'sl_pips': sl,
                                    'proximity_pips': prox,
                                    'min_level_strength': min_str,
                                    'use_rsi_filter': rsi_cfg['use'],
                                    'rsi_buy_max': rsi_cfg.get('buy_max', 50),
                                    'rsi_sell_min': rsi_cfg.get('sell_min', 50),
                                    'use_volume_filter': vol_cfg['use'],
                                    'min_volume_ratio': vol_cfg.get('min_ratio', 1.0),
                                    'use_strength_sizing': use_sizing
                                }
                                
                                result = self.backtest_config(config)
                                
                                if result is None:
                                    continue
                                
                                # Check if new best
                                if result['sharpe_per_trade'] > self.best_sharpe_per_trade:
                                    self.best_sharpe_per_trade = result['sharpe_per_trade']
                                    self.best_config = result
                                    print(f"✨ New best! Iter {iteration} | Sharpe/trade: {result['sharpe_per_trade']:.4f} | "
                                          f"Sharpe ratio: {result['sharpe_ratio']:.2f} | Trades: {result['total_trades']} | "
                                          f"Win rate: {result['win_rate']:.1f}%")
                                
                                # Check if targets met
                                if result['sharpe_per_trade'] >= 0.2 and result['sharpe_ratio'] >= 1.8:
                                    print(f"\n🎉 TARGETS ACHIEVED! Iteration {iteration}")
                                    print(f"   Sharpe per trade: {result['sharpe_per_trade']:.4f}")
                                    print(f"   Sharpe ratio: {result['sharpe_ratio']:.2f}")
                                    print(f"   Total trades: {result['total_trades']}")
                                    print(f"   Win rate: {result['win_rate']:.1f}%")
                                    print(f"   Total PnL: ${result['total_pnl']:.2f}")
                                    return result
        
        print(f"\n⚠️  Targets not met after {iteration} iterations")
        print(f"   Best Sharpe per trade: {self.best_sharpe_per_trade:.4f}")
        return self.best_config
    
    def print_results(self, result):
        """Print optimization results."""
        if result is None:
            print("\n❌ No valid configuration found")
            return
        
        print(f"\n{'='*60}")
        print("  OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nBest Configuration:")
        print(f"  TP/SL: {result['config']['tp_pips']}/{result['config']['sl_pips']} pips")
        print(f"  Proximity: {result['config']['proximity_pips']} pips")
        print(f"  Min level strength: {result['config']['min_level_strength']}")
        print(f"  RSI filter: {result['config']['use_rsi_filter']}")
        if result['config']['use_rsi_filter']:
            print(f"    Buy max RSI: {result['config']['rsi_buy_max']}")
            print(f"    Sell min RSI: {result['config']['rsi_sell_min']}")
        print(f"  Volume filter: {result['config']['use_volume_filter']}")
        if result['config']['use_volume_filter']:
            print(f"    Min volume ratio: {result['config']['min_volume_ratio']}")
        print(f"  Strength-based sizing: {result['config']['use_strength_sizing']}")
        
        print(f"\nPerformance:")
        print(f"  Sharpe per trade: {result['sharpe_per_trade']:.4f}")
        print(f"  Sharpe ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Total trades: {result['total_trades']}")
        print(f"  Win rate: {result['win_rate']:.1f}%")
        print(f"  Total PnL: ${result['total_pnl']:.2f}")
        print(f"  Avg PnL per trade: ${result['avg_pnl']:.2f}")

def main():
    print("="*60)
    print("  KEY LEVEL SYSTEM OPTIMIZER")
    print("  Target: Sharpe/trade >= 0.2, Sharpe ratio >= 1.8")
    print("="*60)
    
    optimizer = KeyLevelOptimizer('EURUSDm', mt5.TIMEFRAME_M15, bars=2000)
    
    if not optimizer.init_mt5():
        print("❌ MT5 connection failed")
        return
    
    print(f"✅ Connected to MT5")
    
    # Prepare data
    print(f"\n📊 Preparing data...")
    optimizer.prepare_data()
    print(f"✅ Data prepared")
    print(f"   Bars: {len(optimizer.df)}")
    print(f"   Sideways: {optimizer.df['is_sideways'].sum()} ({optimizer.df['is_sideways'].mean()*100:.1f}%)")
    print(f"   Key levels: {len(optimizer.key_levels)}")
    
    # Optimize
    result = optimizer.optimize(max_iterations=1000)
    
    # Print results
    optimizer.print_results(result)
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    main()
