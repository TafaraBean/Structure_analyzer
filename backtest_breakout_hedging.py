import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class BreakoutHedgingStrategy:
    """
    Breakout strategy after sideways consolidation.
    
    Strategy:
    1. Detect sideways/consolidation zones
    2. When price breaks out, open BOTH long and short positions (hedge)
    3. Close the losing position quickly
    4. Trail the winning position with trailing stop
    """
    
    def __init__(self, symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.symbol = symbol
        self.timeframe = timeframe
        self.trades = []
        self.equity_curve = []
        
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
    
    def fetch_test_data(self):
        """Fetch test data."""
        total_bars = 3000
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, total_bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        return df
    
    def detect_sideways_zones(self, df, lookback=20, price_range_threshold=0.1, adx_threshold=20):
        """Detect sideways zones."""
        metrics = pd.DataFrame(index=df.index)
        
        # Rolling price range
        metrics['rolling_high'] = df['high'].rolling(lookback).max()
        metrics['rolling_low'] = df['low'].rolling(lookback).min()
        metrics['rolling_range_pct'] = ((metrics['rolling_high'] - metrics['rolling_low']) / df['close']) * 100
        
        # Price volatility
        metrics['price_change'] = df['close'].pct_change() * 100
        metrics['rolling_std'] = metrics['price_change'].rolling(lookback).std()
        
        # ADX
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['adx'] = adx
        
        # Sideways detection
        sideways_mask = (
            (metrics['rolling_range_pct'] < price_range_threshold) &
            (metrics['adx'] < adx_threshold) &
            (metrics['rolling_std'] < 0.02)
        )
        
        metrics['is_sideways'] = sideways_mask
        
        # Detect breakout (was sideways, now not)
        metrics['was_sideways'] = metrics['is_sideways'].shift(1)
        metrics['breakout'] = (metrics['was_sideways'] == True) & (metrics['is_sideways'] == False)
        
        metrics.fillna(False, inplace=True)
        return metrics
    
    def backtest(self, initial_stop_pips=15, trailing_stop_pips=5, loser_stop_pips=10):
        """
        Backtest breakout hedging strategy.
        
        Args:
            initial_stop_pips: Initial stop loss for both positions
            trailing_stop_pips: Trailing stop distance for winner
            loser_stop_pips: Quick stop for losing position
        """
        print(f"\n{'='*60}")
        print(f"  BREAKOUT HEDGING STRATEGY BACKTEST")
        print(f"{'='*60}")
        print(f"\n⚙️  Strategy:")
        print(f"   1. Detect sideways consolidation")
        print(f"   2. On breakout: Open LONG + SHORT (hedge)")
        print(f"   3. Close loser at {loser_stop_pips} pips loss")
        print(f"   4. Trail winner with {trailing_stop_pips} pip stop")
        
        # Fetch data
        df = self.fetch_test_data()
        if df is None:
            return
        
        # Detect sideways zones
        metrics = self.detect_sideways_zones(df, lookback=20, price_range_threshold=0.09, adx_threshold=20)
        
        # Simulate trading
        balance = 10000
        long_position = None
        short_position = None
        
        print(f"\n🔄 Running backtest...")
        
        for i in range(100, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check long position
            if long_position:
                pips_profit = (current_price - long_position['entry']) / 0.0001
                
                # Update max profit and trailing stop
                if pips_profit > long_position['max_profit']:
                    long_position['max_profit'] = pips_profit
                    long_position['trailing_stop'] = current_price - trailing_stop_pips * 10 * 0.0001
                
                # Check exit
                hit_trailing = current_price <= long_position['trailing_stop']
                hit_initial_sl = pips_profit <= -initial_stop_pips
                
                if hit_trailing or hit_initial_sl:
                    profit = pips_profit * 0.0001 * 100000
                    balance += profit
                    
                    self.trades.append({
                        'time': current_time,
                        'direction': 'LONG',
                        'pips': pips_profit,
                        'profit': profit,
                        'exit_reason': 'TRAILING' if hit_trailing else 'STOP_LOSS'
                    })
                    
                    long_position = None
            
            # Check short position
            if short_position:
                pips_profit = (short_position['entry'] - current_price) / 0.0001
                
                # Update max profit and trailing stop
                if pips_profit > short_position['max_profit']:
                    short_position['max_profit'] = pips_profit
                    short_position['trailing_stop'] = current_price + trailing_stop_pips * 10 * 0.0001
                
                # Check exit
                hit_trailing = current_price >= short_position['trailing_stop']
                hit_initial_sl = pips_profit <= -initial_stop_pips
                
                if hit_trailing or hit_initial_sl:
                    profit = pips_profit * 0.0001 * 100000
                    balance += profit
                    
                    self.trades.append({
                        'time': current_time,
                        'direction': 'SHORT',
                        'pips': pips_profit,
                        'profit': profit,
                        'exit_reason': 'TRAILING' if hit_trailing else 'STOP_LOSS'
                    })
                    
                    short_position = None
            
            # Check for breakout signal
            if metrics['breakout'].iloc[i] and long_position is None and short_position is None:
                # Open BOTH positions (hedge)
                long_position = {
                    'entry': current_price,
                    'trailing_stop': current_price - initial_stop_pips * 10 * 0.0001,
                    'max_profit': 0
                }
                
                short_position = {
                    'entry': current_price,
                    'trailing_stop': current_price + initial_stop_pips * 10 * 0.0001,
                    'max_profit': 0
                }
                
                print(f"🔀 Breakout at {current_time}: Hedged LONG + SHORT at {current_price:.5f}")
            
            # Close losing position early
            if long_position and short_position:
                long_pips = (current_price - long_position['entry']) / 0.0001
                short_pips = (short_position['entry'] - current_price) / 0.0001
                
                # If one is losing by loser_stop_pips, close it
                if long_pips <= -loser_stop_pips:
                    profit = long_pips * 0.0001 * 100000
                    balance += profit
                    self.trades.append({
                        'time': current_time,
                        'direction': 'LONG',
                        'pips': long_pips,
                        'profit': profit,
                        'exit_reason': 'LOSER_CLOSED'
                    })
                    long_position = None
                    print(f"❌ Closed losing LONG at {current_time}")
                
                elif short_pips <= -loser_stop_pips:
                    profit = short_pips * 0.0001 * 100000
                    balance += profit
                    self.trades.append({
                        'time': current_time,
                        'direction': 'SHORT',
                        'pips': short_pips,
                        'profit': profit,
                        'exit_reason': 'LOSER_CLOSED'
                    })
                    short_position = None
                    print(f"❌ Closed losing SHORT at {current_time}")
            
            # Track equity
            current_equity = balance
            if long_position:
                long_pips = (current_price - long_position['entry']) / 0.0001
                current_equity += long_pips * 0.0001 * 100000
            if short_position:
                short_pips = (short_position['entry'] - current_price) / 0.0001
                current_equity += short_pips * 0.0001 * 100000
            
            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity
            })
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze backtest results."""
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*60}")
        
        if not self.trades:
            print(f"\n❌ No trades executed")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pips'] > 0])
        losing_trades = len(trades_df[trades_df['pips'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pips = trades_df['pips'].sum()
        avg_win = trades_df[trades_df['pips'] > 0]['pips'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pips'] < 0]['pips'].mean() if losing_trades > 0 else 0
        
        total_profit = trades_df['profit'].sum()
        final_balance = 10000 + total_profit
        
        print(f"\n📊 Trade Statistics:")
        print(f"   Total trades:    {total_trades}")
        print(f"   Winning trades:  {winning_trades} ({win_rate:.1%})")
        print(f"   Losing trades:   {losing_trades}")
        print(f"   Win rate:        {win_rate:.1%}")
        
        print(f"\n💰 Performance:")
        print(f"   Total pips:      {total_pips:.1f}")
        print(f"   Avg win:         {avg_win:.1f} pips")
        print(f"   Avg loss:        {avg_loss:.1f} pips")
        print(f"   Risk/Reward:     1:{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "   Risk/Reward:     N/A")
        print(f"   Total profit:    ${total_profit:.2f}")
        print(f"   Final balance:   ${final_balance:.2f}")
        print(f"   Return:          {(final_balance - 10000) / 10000 * 100:.2f}%")
        
        # Sharpe Ratio
        returns = trades_df['profit'] / 10000
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
        
        # Max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:    {sharpe:.2f}")
        print(f"   Max Drawdown:    {max_drawdown:.2f}%")
        
        # Exit reasons
        print(f"\n🚪 Exit Reasons:")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            print(f"   {reason}: {count}")
        
        # Direction breakdown
        print(f"\n📊 Direction Breakdown:")
        for direction in ['LONG', 'SHORT']:
            dir_trades = trades_df[trades_df['direction'] == direction]
            if len(dir_trades) > 0:
                dir_wins = len(dir_trades[dir_trades['pips'] > 0])
                dir_pips = dir_trades['pips'].sum()
                print(f"   {direction}: {len(dir_trades)} trades, {dir_wins} wins ({dir_wins/len(dir_trades)*100:.1f}%), {dir_pips:.1f} pips")
        
        # Plot
        self.plot_equity_curve()
    
    def plot_equity_curve(self):
        """Plot equity curve."""
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Equity curve
        ax1.plot(equity_df['time'], equity_df['equity'], linewidth=2, color='#2196F3')
        ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Balance')
        ax1.fill_between(equity_df['time'], 10000, equity_df['equity'], 
                         where=(equity_df['equity'] >= 10000), alpha=0.3, color='green', label='Profit')
        ax1.fill_between(equity_df['time'], 10000, equity_df['equity'], 
                         where=(equity_df['equity'] < 10000), alpha=0.3, color='red', label='Loss')
        
        ax1.set_title('Breakout Hedging Strategy - Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        
        ax2.fill_between(equity_df['time'], 0, equity_df['drawdown'], alpha=0.3, color='red')
        ax2.plot(equity_df['time'], equity_df['drawdown'], linewidth=1, color='darkred')
        ax2.set_title('Drawdown %', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'breakout_hedging_equity.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Equity curve saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  BREAKOUT HEDGING STRATEGY")
    print("  Hedge on breakout, trail the winner")
    print("="*60)
    
    strategy = BreakoutHedgingStrategy(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not strategy.init_mt5():
        return
    
    # Run backtest
    strategy.backtest(
        initial_stop_pips=15,   # Initial stop for both positions
        trailing_stop_pips=5,   # Trailing stop for winner
        loser_stop_pips=10      # Quick exit for loser
    )
    
    mt5.shutdown()
    print(f"\n👋 Backtest Complete!")


if __name__ == "__main__":
    main()
