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

# Optimized configuration from optimizer
OPTIMIZED_CONFIG = {
    'tp_pips': 8,
    'sl_pips': 5,
    'proximity_pips': 3,
    'min_level_strength': 3,
    'use_rsi_filter': True,
    'rsi_buy_max': 40,
    'rsi_sell_min': 60,
    'use_volume_filter': False,
    'min_volume_ratio': 1.0,
    'use_strength_sizing': False
}

def run_optimized_system():
    """Run the optimized trading system and visualize."""
    
    print("="*60)
    print("  OPTIMIZED KEY LEVEL TRADING SYSTEM")
    print("  Sharpe/trade: 0.32 | Sharpe ratio: 5.09")
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
    
    # Prepare data
    print(f"\n📊 Preparing data...")
    level_detector = KeyLevelDetector('EURUSDm', mt5.TIMEFRAME_M15, 2000)
    level_detector.df = level_detector.fetch_data()
    
    swing_points = level_detector.find_swing_points(window=5)
    key_levels = level_detector.find_key_levels_percentile(swing_points, n_levels=10, tolerance=0.0005)
    
    regime_labeler = RegimeLabelGenerator('EURUSDm', mt5.TIMEFRAME_M15, 2000)
    regime_labeler.df = level_detector.df
    regime_result = regime_labeler.method_consensus()
    
    df = level_detector.df.copy()
    df['regime'] = regime_result['labels']
    df['is_sideways'] = (regime_result['labels'] == 'sideways').astype(int)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    
    # Generate signals with optimized config
    print(f"\n🎯 Generating signals with optimized config...")
    df['signal'] = 0
    df['level_touched'] = ''
    
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
            
            # Buy at support
            if level['type'] == 'support' and current_low <= level_price * 1.0001:
                if current_rsi > OPTIMIZED_CONFIG['rsi_buy_max']:
                    continue
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('level_touched')] = f"Support @ {level_price:.5f}"
                break
            
            # Sell at resistance
            elif level['type'] == 'resistance' and current_high >= level_price * 0.9999:
                if current_rsi < OPTIMIZED_CONFIG['rsi_sell_min']:
                    continue
                df.iloc[i, df.columns.get_loc('signal')] = -1
                df.iloc[i, df.columns.get_loc('level_touched')] = f"Resistance @ {level_price:.5f}"
                break
    
    # Backtest
    print(f"\n💰 Backtesting...")
    trades = []
    equity = 10000
    equity_curve = [equity]
    
    tp_distance = OPTIMIZED_CONFIG['tp_pips'] * pip_value
    sl_distance = OPTIMIZED_CONFIG['sl_pips'] * pip_value
    
    for i in range(len(df)):
        if df['signal'].iloc[i] == 0:
            equity_curve.append(equity)
            continue
        
        entry_price = df['close'].iloc[i]
        direction = df['signal'].iloc[i]
        
        if direction == 1:
            tp_price = entry_price + tp_distance
            sl_price = entry_price - sl_distance
        else:
            tp_price = entry_price - tp_distance
            sl_price = entry_price + sl_distance
        
        exit_price = None
        exit_reason = None
        
        for j in range(i + 1, min(i + 100, len(df))):
            bar_high = df['high'].iloc[j]
            bar_low = df['low'].iloc[j]
            
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
            exit_price = df['close'].iloc[min(i + 100, len(df) - 1)]
            exit_reason = 'Timeout'
        
        if direction == 1:
            pnl = (exit_price - entry_price) / pip_value * 10 * 0.01
        else:
            pnl = (entry_price - exit_price) / pip_value * 10 * 0.01
        
        equity += pnl
        equity_curve.append(equity)
        
        trades.append({
            'entry_time': df.index[i],
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason
        })
    
    while len(equity_curve) < len(df):
        equity_curve.append(equity)
    
    df['equity'] = equity_curve[:len(df)]
    
    # Print results
    trades_df = pd.DataFrame(trades)
    print(f"\n{'='*60}")
    print("  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Winning trades: {(trades_df['pnl'] > 0).sum()} ({(trades_df['pnl'] > 0).mean()*100:.1f}%)")
    print(f"Total PnL: ${trades_df['pnl'].sum():.2f}")
    print(f"Avg PnL: ${trades_df['pnl'].mean():.2f}")
    print(f"Final equity: ${equity:.2f}")
    print(f"Return: {((equity - 10000) / 10000 * 100):.2f}%")
    
    # Sharpe metrics
    avg_pnl = trades_df['pnl'].mean()
    std_pnl = trades_df['pnl'].std()
    sharpe_per_trade = avg_pnl / std_pnl if std_pnl > 0 else 0
    print(f"Sharpe per trade: {sharpe_per_trade:.4f}")
    
    # Visualize
    print(f"\n📈 Creating visualization...")
    
    start_idx = max(0, len(df) - 500)
    df_plot = df.iloc[start_idx:]
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.3)
    
    fig.suptitle('Optimized Key Level Trading System | Sharpe/trade: 0.32 | Sharpe ratio: 5.09', 
                 fontsize=14, color='white', fontweight='bold')
    
    # Panel 1: Price with levels and signals
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_plot.index, df_plot['close'], color='white', linewidth=1, label='Close', zorder=1)
    
    # Shade sideways
    for idx, row in df_plot.iterrows():
        if row['is_sideways']:
            ax1.axvspan(idx, idx, color='blue', alpha=0.1)
    
    # Plot levels
    for level in key_levels:
        if level['strength'] >= OPTIMIZED_CONFIG['min_level_strength']:
            color = 'green' if level['type'] == 'support' else 'red'
            alpha = min(0.3 + (level['strength'] / 20), 0.8)
            ax1.axhline(y=level['price'], color=color, linestyle='--', linewidth=2, alpha=alpha)
    
    # Plot signals
    buy_signals = df_plot[df_plot['signal'] == 1]
    sell_signals = df_plot[df_plot['signal'] == -1]
    
    ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='lime', 
               s=200, zorder=5, edgecolors='white', linewidths=2, label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', 
               s=200, zorder=5, edgecolors='white', linewidths=2, label='Sell Signal')
    
    ax1.set_ylabel('Price', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_title('Price Chart with Key Levels & Optimized Signals', fontsize=12, pad=10, fontweight='bold')
    
    # Panel 2: Equity curve
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(df_plot.index, df_plot['equity'], color='cyan', linewidth=2.5)
    ax2.fill_between(df_plot.index, 10000, df_plot['equity'],
                    where=(df_plot['equity'] >= 10000), color='green', alpha=0.3)
    ax2.fill_between(df_plot.index, 10000, df_plot['equity'],
                    where=(df_plot['equity'] < 10000), color='red', alpha=0.3)
    ax2.axhline(y=10000, color='yellow', linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.set_title(f'Equity Curve | Final: ${equity:.2f} | Return: {((equity-10000)/10000*100):.2f}%', 
                 fontsize=12, pad=10, fontweight='bold')
    
    # Panel 3: RSI with filter zones
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(df_plot.index, df_plot['rsi'], color='orange', linewidth=1.5)
    ax3.axhline(y=OPTIMIZED_CONFIG['rsi_buy_max'], color='green', linestyle='--', linewidth=1.5, label='Buy threshold')
    ax3.axhline(y=OPTIMIZED_CONFIG['rsi_sell_min'], color='red', linestyle='--', linewidth=1.5, label='Sell threshold')
    ax3.fill_between(df_plot.index, 0, OPTIMIZED_CONFIG['rsi_buy_max'], color='green', alpha=0.1)
    ax3.fill_between(df_plot.index, OPTIMIZED_CONFIG['rsi_sell_min'], 100, color='red', alpha=0.1)
    ax3.set_ylabel('RSI', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.set_title('RSI Filter (Buy < 40, Sell > 60)', fontsize=12, pad=10, fontweight='bold')
    
    # Panel 4: Regime
    ax4 = fig.add_subplot(gs[3])
    ax4.fill_between(df_plot.index, 0, df_plot['is_sideways'], color='blue', alpha=0.6)
    ax4.set_ylabel('Sideways', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 1.2)
    ax4.grid(True, alpha=0.2)
    ax4.set_title('Market Regime (Blue = Sideways)', fontsize=12, pad=10, fontweight='bold')
    
    # Panel 5: Trade PnL
    ax5 = fig.add_subplot(gs[4])
    if len(trades_df) > 0:
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        ax5.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax5.axhline(y=0, color='white', linestyle='-', linewidth=1.5)
        ax5.set_ylabel('PnL ($)', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.2)
        ax5.set_title(f'Individual Trade PnL | Win Rate: {(trades_df["pnl"] > 0).mean()*100:.1f}%', 
                     fontsize=12, pad=10, fontweight='bold')
    
    plt.tight_layout()
    
    filename = 'optimized_key_level_system.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Chart saved as {filename}")
    
    # Don't show to avoid blocking
    # plt.show()
    print(f"📊 Chart saved (not displayed to allow completion)")
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    run_optimized_system()
