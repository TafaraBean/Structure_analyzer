import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from key_level_detector import KeyLevelDetector
from regime_labeler import RegimeLabelGenerator

load_dotenv()

class KeyLevelTradingSystem:
    """
    Complete trading system:
    1. Detect sideways regimes
    2. Identify key support/resistance levels
    3. Enter reversals at key levels during sideways periods
    4. Backtest and visualize
    """
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15, bars=2000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.df = None
        self.key_levels = []
        self.trades = []
        
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
    
    def prepare_data(self):
        """Fetch data and detect regimes + key levels."""
        print(f"\n📊 Preparing data...")
        
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
        
        print(f"✅ Data prepared")
        print(f"   Sideways periods: {self.df['is_sideways'].sum()} bars ({self.df['is_sideways'].mean()*100:.1f}%)")
        print(f"   Key levels: {len(self.key_levels)}")
        
        return self.df
    
    def generate_signals(self, proximity_pips=5, min_level_strength=5):
        """
        Generate trading signals:
        - Only trade during sideways regimes
        - Enter long at support levels
        - Enter short at resistance levels
        """
        print(f"\n🎯 Generating trading signals...")
        
        self.df['signal'] = 0
        self.df['level_touched'] = ''
        self.df['level_strength'] = 0
        
        pip_value = 0.0001  # For EURUSD
        
        for i in range(len(self.df)):
            # Only trade in sideways
            if not self.df['is_sideways'].iloc[i]:
                continue
            
            current_price = self.df['close'].iloc[i]
            current_low = self.df['low'].iloc[i]
            current_high = self.df['high'].iloc[i]
            
            # Check proximity to each key level
            for level in self.key_levels:
                if level['strength'] < min_level_strength:
                    continue
                
                level_price = level['price']
                distance_pips = abs(current_price - level_price) / pip_value
                
                # Support level - buy signal
                if level['type'] == 'support' and distance_pips <= proximity_pips:
                    # Check if price touched the level (low reached it)
                    if current_low <= level_price * 1.0001:  # 0.01% tolerance
                        self.df.iloc[i, self.df.columns.get_loc('signal')] = 1  # Buy
                        self.df.iloc[i, self.df.columns.get_loc('level_touched')] = f"Support @ {level_price:.5f}"
                        self.df.iloc[i, self.df.columns.get_loc('level_strength')] = level['strength']
                        break
                
                # Resistance level - sell signal
                elif level['type'] == 'resistance' and distance_pips <= proximity_pips:
                    # Check if price touched the level (high reached it)
                    if current_high >= level_price * 0.9999:  # 0.01% tolerance
                        self.df.iloc[i, self.df.columns.get_loc('signal')] = -1  # Sell
                        self.df.iloc[i, self.df.columns.get_loc('level_touched')] = f"Resistance @ {level_price:.5f}"
                        self.df.iloc[i, self.df.columns.get_loc('level_strength')] = level['strength']
                        break
        
        buy_signals = (self.df['signal'] == 1).sum()
        sell_signals = (self.df['signal'] == -1).sum()
        
        print(f"✅ Signals generated:")
        print(f"   Buy signals (support): {buy_signals}")
        print(f"   Sell signals (resistance): {sell_signals}")
        print(f"   Total signals: {buy_signals + sell_signals}")
        
        return self.df
    
    def backtest(self, tp_pips=10, sl_pips=5, lot_size=0.01):
        """
        Backtest the strategy.
        
        Args:
            tp_pips: Take profit in pips
            sl_pips: Stop loss in pips
        """
        print(f"\n💰 Backtesting strategy (TP: {tp_pips} pips, SL: {sl_pips} pips)...")
        
        pip_value = 0.0001
        tp_distance = tp_pips * pip_value
        sl_distance = sl_pips * pip_value
        
        self.trades = []
        equity = 10000
        equity_curve = [equity]
        
        for i in range(len(self.df)):
            if self.df['signal'].iloc[i] == 0:
                equity_curve.append(equity)
                continue
            
            entry_price = self.df['close'].iloc[i]
            entry_time = self.df.index[i]
            direction = self.df['signal'].iloc[i]
            level_info = self.df['level_touched'].iloc[i]
            
            # Set TP and SL
            if direction == 1:  # Buy
                tp_price = entry_price + tp_distance
                sl_price = entry_price - sl_distance
            else:  # Sell
                tp_price = entry_price - tp_distance
                sl_price = entry_price + sl_distance
            
            # Simulate trade
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for j in range(i + 1, min(i + 100, len(self.df))):  # Max 100 bars
                bar_high = self.df['high'].iloc[j]
                bar_low = self.df['low'].iloc[j]
                
                if direction == 1:  # Buy
                    if bar_high >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                        exit_time = self.df.index[j]
                        break
                    elif bar_low <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'
                        exit_time = self.df.index[j]
                        break
                else:  # Sell
                    if bar_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                        exit_time = self.df.index[j]
                        break
                    elif bar_high >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'SL'
                        exit_time = self.df.index[j]
                        break
            
            # If no exit, close at market
            if exit_price is None:
                exit_price = self.df['close'].iloc[min(i + 100, len(self.df) - 1)]
                exit_time = self.df.index[min(i + 100, len(self.df) - 1)]
                exit_reason = 'Timeout'
            
            # Calculate PnL
            if direction == 1:
                pnl = (exit_price - entry_price) / pip_value * 10 * lot_size  # $10 per pip per lot
            else:
                pnl = (entry_price - exit_price) / pip_value * 10 * lot_size
            
            equity += pnl
            equity_curve.append(equity)
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': 'BUY' if direction == 1 else 'SELL',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'level_info': level_info
            })
        
        # Pad equity curve
        while len(equity_curve) < len(self.df):
            equity_curve.append(equity)
        
        self.df['equity'] = equity_curve[:len(self.df)]
        
        # Calculate metrics
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) > 0:
            total_trades = len(trades_df)
            winning_trades = (trades_df['pnl'] > 0).sum()
            losing_trades = (trades_df['pnl'] < 0).sum()
            win_rate = winning_trades / total_trades * 100
            total_pnl = trades_df['pnl'].sum()
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            print(f"\n{'='*60}")
            print("  BACKTEST RESULTS")
            print(f"{'='*60}")
            print(f"\nTotal trades: {total_trades}")
            print(f"Winning trades: {winning_trades} ({win_rate:.1f}%)")
            print(f"Losing trades: {losing_trades}")
            print(f"Total PnL: ${total_pnl:.2f}")
            print(f"Avg win: ${avg_win:.2f}")
            print(f"Avg loss: ${avg_loss:.2f}")
            print(f"Final equity: ${equity:.2f}")
            print(f"Return: {((equity - 10000) / 10000 * 100):.2f}%")
        else:
            print(f"\n⚠️  No trades executed")
        
        return self.trades
    
    def visualize(self, sample_bars=500):
        """Visualize the trading system."""
        print(f"\n📈 Creating visualization...")
        
        # Use last N bars
        start_idx = max(0, len(self.df) - sample_bars)
        df_plot = self.df.iloc[start_idx:]
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        fig.suptitle(f'{self.symbol} - Key Level Trading System', fontsize=14, color='white')
        
        # Panel 1: Price with key levels and trades
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df_plot.index, df_plot['close'], color='white', linewidth=1, label='Close', zorder=1)
        
        # Shade sideways periods
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            if row['is_sideways']:
                ax1.axvspan(idx, idx, color='blue', alpha=0.1)
        
        # Plot key levels
        for level in self.key_levels:
            color = 'green' if level['type'] == 'support' else 'red'
            alpha = min(0.3 + (level['strength'] / 30), 0.8)
            ax1.axhline(y=level['price'], color=color, linestyle='--', linewidth=1.5, alpha=alpha)
        
        # Plot trade entries
        buy_entries = df_plot[df_plot['signal'] == 1]
        sell_entries = df_plot[df_plot['signal'] == -1]
        
        ax1.scatter(buy_entries.index, buy_entries['close'], marker='^', color='lime', 
                   s=150, zorder=5, edgecolors='white', linewidths=1.5, label='Buy Signal')
        ax1.scatter(sell_entries.index, sell_entries['close'], marker='v', color='red', 
                   s=150, zorder=5, edgecolors='white', linewidths=1.5, label='Sell Signal')
        
        ax1.set_ylabel('Price', fontsize=10)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.2)
        ax1.set_title('Price Chart with Key Levels & Trade Signals', fontsize=11, pad=10)
        
        # Panel 2: Equity curve
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df_plot.index, df_plot['equity'], color='cyan', linewidth=2)
        ax2.fill_between(df_plot.index, 10000, df_plot['equity'], 
                        where=(df_plot['equity'] >= 10000), color='green', alpha=0.3)
        ax2.fill_between(df_plot.index, 10000, df_plot['equity'],
                        where=(df_plot['equity'] < 10000), color='red', alpha=0.3)
        ax2.axhline(y=10000, color='yellow', linestyle='--', linewidth=1)
        ax2.set_ylabel('Equity ($)', fontsize=10)
        ax2.grid(True, alpha=0.2)
        ax2.set_title('Equity Curve', fontsize=11, pad=10)
        
        # Panel 3: Regime indicator
        ax3 = fig.add_subplot(gs[2])
        ax3.fill_between(df_plot.index, 0, df_plot['is_sideways'], color='blue', alpha=0.5)
        ax3.set_ylabel('Sideways', fontsize=10)
        ax3.set_ylim(0, 1.2)
        ax3.grid(True, alpha=0.2)
        ax3.set_title('Market Regime (Blue = Sideways)', fontsize=11, pad=10)
        
        # Panel 4: Trade distribution
        ax4 = fig.add_subplot(gs[3])
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
            ax4.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7)
            ax4.axhline(y=0, color='white', linestyle='-', linewidth=1)
            ax4.set_ylabel('PnL ($)', fontsize=10)
            ax4.set_xlabel('Trade Number', fontsize=10)
            ax4.grid(True, alpha=0.2)
            ax4.set_title('Individual Trade PnL', fontsize=11, pad=10)
        
        plt.tight_layout()
        
        filename = 'key_level_trading_system.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Chart saved as {filename}")
        
        plt.show()
        print(f"📊 Visualization displayed")

def main():
    print("="*60)
    print("  KEY LEVEL TRADING SYSTEM")
    print("  Sideways Regime + Support/Resistance Reversals")
    print("="*60)
    
    system = KeyLevelTradingSystem('EURUSDm', mt5.TIMEFRAME_M15, bars=2000)
    
    if not system.init_mt5():
        return
    
    # Prepare data
    system.prepare_data()
    
    # Generate signals
    system.generate_signals(proximity_pips=5, min_level_strength=5)
    
    # Backtest
    system.backtest(tp_pips=10, sl_pips=5, lot_size=0.01)
    
    # Visualize
    system.visualize(sample_bars=500)
    
    mt5.shutdown()
    print(f"\n👋 Complete")

if __name__ == "__main__":
    main()
