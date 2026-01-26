import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from key_level_detector import KeyLevelDetector
from regime_labeler import RegimeLabelGenerator

load_dotenv()

# Optimized configuration
OPTIMIZED_CONFIG = {
    'tp_pips': 8,
    'sl_pips': 5,
    'proximity_pips': 3,
    'min_level_strength': 3,
    'use_rsi_filter': True,
    'rsi_buy_max': 40,
    'rsi_sell_min': 60,
}

def extended_backtest():
    """Extended backtest with more data and detailed metrics."""
    
    print("="*60)
    print("  EXTENDED BACKTEST - OPTIMIZED SYSTEM")
    print("  Testing on 5000 bars for robustness")
    print("="*60)
    
    # Initialize MT5
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    params = {}
    if path: params["path"] = path
    
    mt5.initialize(**params)
    if login and password and server:
        mt5.login(login=int(login), password=password, server=server)
    
    print(f"✅ Connected to MT5")
    
    # Prepare data with MORE bars
    print(f"\n📊 Preparing data (5000 bars)...")
    level_detector = KeyLevelDetector('EURUSDm', mt5.TIMEFRAME_M15, 5000)
    level_detector.df = level_detector.fetch_data()
    
    swing_points = level_detector.find_swing_points(window=5)
    key_levels = level_detector.find_key_levels_percentile(swing_points, n_levels=10, tolerance=0.0005)
    
    regime_labeler = RegimeLabelGenerator('EURUSDm', mt5.TIMEFRAME_M15, 5000)
    regime_labeler.df = level_detector.df
    regime_result = regime_labeler.method_consensus()
    
    df = level_detector.df.copy()
    df['regime'] = regime_result['labels']
    df['is_sideways'] = (regime_result['labels'] == 'sideways').astype(int)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    print(f"✅ Data prepared: {len(df)} bars")
    print(f"   Sideways: {df['is_sideways'].sum()} bars ({df['is_sideways'].mean()*100:.1f}%)")
    print(f"   Key levels: {len(key_levels)}")
    
    # Generate signals
    print(f"\n🎯 Generating signals...")
    df['signal'] = 0
    
    pip_value = 0.0001
    
    for i in range(len(df)):
        if not df['is_sideways'].iloc[i]:
            continue
        
        current_price = df['close'].iloc[i]
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_rsi = df['rsi'].iloc[i]
        
        for level in key_levels:
            if level['strength'] < OPTIMIZED_CONFIG['min_level_strength']:
                continue
            
            level_price = level['price']
            distance_pips = abs(current_price - level_price) / pip_value
            
            if distance_pips > OPTIMIZED_CONFIG['proximity_pips']:
                continue
            
            if level['type'] == 'support' and current_low <= level_price * 1.0001:
                if current_rsi > OPTIMIZED_CONFIG['rsi_buy_max']:
                    continue
                df.iloc[i, df.columns.get_loc('signal')] = 1
                break
            
            elif level['type'] == 'resistance' and current_high >= level_price * 0.9999:
                if current_rsi < OPTIMIZED_CONFIG['rsi_sell_min']:
                    continue
                df.iloc[i, df.columns.get_loc('signal')] = -1
                break
    
    print(f"✅ Signals generated: {(df['signal'] != 0).sum()}")
    
    # Backtest
    print(f"\n💰 Running backtest...")
    trades = []
    equity = 10000
    equity_curve = [equity]
    peak_equity = equity
    max_drawdown = 0
    
    tp_distance = OPTIMIZED_CONFIG['tp_pips'] * pip_value
    sl_distance = OPTIMIZED_CONFIG['sl_pips'] * pip_value
    
    for i in range(len(df)):
        if df['signal'].iloc[i] == 0:
            equity_curve.append(equity)
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
            continue
        
        entry_price = df['close'].iloc[i]
        entry_time = df.index[i]
        direction = df['signal'].iloc[i]
        
        if direction == 1:
            tp_price = entry_price + tp_distance
            sl_price = entry_price - sl_distance
        else:
            tp_price = entry_price - tp_distance
            sl_price = entry_price + sl_distance
        
        exit_price = None
        exit_reason = None
        exit_time = None
        bars_in_trade = 0
        
        for j in range(i + 1, min(i + 100, len(df))):
            bar_high = df['high'].iloc[j]
            bar_low = df['low'].iloc[j]
            bars_in_trade = j - i
            
            if direction == 1:
                if bar_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    exit_time = df.index[j]
                    break
                elif bar_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_time = df.index[j]
                    break
            else:
                if bar_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    exit_time = df.index[j]
                    break
                elif bar_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    exit_time = df.index[j]
                    break
        
        if exit_price is None:
            exit_price = df['close'].iloc[min(i + 100, len(df) - 1)]
            exit_time = df.index[min(i + 100, len(df) - 1)]
            exit_reason = 'Timeout'
            bars_in_trade = 100
        
        if direction == 1:
            pnl = (exit_price - entry_price) / pip_value * 10 * 0.01
        else:
            pnl = (entry_price - exit_price) / pip_value * 10 * 0.01
        
        equity += pnl
        equity_curve.append(equity)
        
        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'bars_in_trade': bars_in_trade
        })
    
    while len(equity_curve) < len(df):
        equity_curve.append(equity)
    
    df['equity'] = equity_curve[:len(df)]
    
    # Calculate comprehensive metrics
    trades_df = pd.DataFrame(trades)
    
    print(f"\n{'='*60}")
    print("  EXTENDED BACKTEST RESULTS")
    print(f"{'='*60}")
    
    if len(trades_df) > 0:
        total_trades = len(trades_df)
        winning_trades = (trades_df['pnl'] > 0).sum()
        losing_trades = (trades_df['pnl'] < 0).sum()
        win_rate = winning_trades / total_trades * 100
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        std_pnl = trades_df['pnl'].std()
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 999
        
        sharpe_per_trade = avg_pnl / std_pnl if std_pnl > 0 else 0
        
        returns = trades_df['pnl'] / 10000
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        avg_bars_in_trade = trades_df['bars_in_trade'].mean()
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in trades_df['pnl']:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        print(f"\n📊 Trade Statistics:")
        print(f"   Total trades: {total_trades}")
        print(f"   Winning trades: {winning_trades} ({win_rate:.1f}%)")
        print(f"   Losing trades: {losing_trades} ({100-win_rate:.1f}%)")
        print(f"   Max consecutive wins: {max_consecutive_wins}")
        print(f"   Max consecutive losses: {max_consecutive_losses}")
        
        print(f"\n💰 P&L Metrics:")
        print(f"   Total PnL: ${total_pnl:.2f}")
        print(f"   Avg PnL per trade: ${avg_pnl:.2f}")
        print(f"   Avg win: ${avg_win:.2f}")
        print(f"   Avg loss: ${avg_loss:.2f}")
        print(f"   Profit factor: {profit_factor:.2f}")
        print(f"   Final equity: ${equity:.2f}")
        print(f"   Return: {((equity - 10000) / 10000 * 100):.2f}%")
        print(f"   Max drawdown: {max_drawdown:.2f}%")
        
        print(f"\n📈 Risk-Adjusted Metrics:")
        print(f"   Sharpe per trade: {sharpe_per_trade:.4f}")
        print(f"   Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"   Avg bars in trade: {avg_bars_in_trade:.1f}")
        
        print(f"\n🎯 Exit Reasons:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            print(f"   {reason}: {count} ({count/total_trades*100:.1f}%)")
        
        # Monthly breakdown
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        
        print(f"\n📅 Monthly Performance:")
        for month, pnl in monthly_pnl.items():
            print(f"   {month}: ${pnl:.2f}")
        
        print(f"\n{'='*60}")
        
        # Save detailed results
        trades_df.to_csv('extended_backtest_trades.csv', index=False)
        print(f"\n💾 Detailed trades saved to extended_backtest_trades.csv")
    
    else:
        print(f"\n⚠️  No trades executed")
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    extended_backtest()
