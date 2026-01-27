import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import talib
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MomentumScalper:
    """Aggressive trend-following scalper using momentum zones."""
    
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
        """Fetch the held-out test set (last 15% of 3000 bars)."""
        total_bars = 15000
        test_bars = int(0.15 * total_bars)
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, total_bars)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Take last 15% (test set)
        df_test = df.iloc[-test_bars:]
        
        print(f"✅ Loaded {len(df_test)} bars (test set)")
        return df_test
    
    def calculate_momentum_metrics(self, df):
        """Calculate price change and ADX."""
        metrics = pd.DataFrame(index=df.index)
        
        # Price percentage change
        metrics['price_change_10'] = df['close'].pct_change(10) * 100
        metrics['abs_price_change_10'] = np.abs(df['close'].pct_change(10)) * 100
        
        # ADX (trend strength)
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['adx'] = adx
        
        # ATR for stop loss
        atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        metrics['atr'] = atr
        
        metrics.fillna(0, inplace=True)
        return metrics
    
    def backtest(self, price_threshold=0.10, adx_threshold=20, stop_loss_pips=10, take_profit_pips=5):
        """
        Backtest aggressive momentum scalping strategy.
        
        Strategy:
        - Enter LONG in green zones (bullish momentum)
        - Enter SHORT in red zones (bearish momentum)
        - Quick exits with tight stops
        
        Args:
            price_threshold: Minimum price change % to enter
            adx_threshold: Minimum ADX to enter
            stop_loss_pips: Stop loss in pips
            take_profit_pips: Take profit in pips
        """
        print(f"\n{'='*60}")
        print(f"  AGGRESSIVE MOMENTUM SCALPER BACKTEST")
        print(f"  Test Set (Last 15% of data)")
        print(f"{'='*60}")
        print(f"\n⚙️  Strategy:")
        print(f"   Entry: Price change > {price_threshold}% + ADX > {adx_threshold}")
        print(f"   GREEN zone → BUY")
        print(f"   RED zone → SELL")
        print(f"   Stop loss: {stop_loss_pips} pips")
        print(f"   Take profit: {take_profit_pips} pips")
        
        # Fetch test data
        df = self.fetch_test_data()
        if df is None:
            return
        
        # Calculate metrics
        metrics = self.calculate_momentum_metrics(df)
        
        # Identify momentum zones
        bullish_momentum = (metrics['price_change_10'] > price_threshold) & (metrics['adx'] > adx_threshold)
        bearish_momentum = (metrics['price_change_10'] < -price_threshold) & (metrics['adx'] > adx_threshold)
        
        # Simulate trading
        balance = 10000
        position = None
        
        print(f"\n🔄 Running backtest...")
        
        for i in range(100, len(df)):  # Start after 100 bars for indicators
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check if we have an open position
            if position:
                # Calculate P&L
                if position['direction'] == 'LONG':
                    pips_profit = (current_price - position['entry']) / 0.0001
                else:  # SHORT
                    pips_profit = (position['entry'] - current_price) / 0.0001
                
                # Check exit conditions
                hit_tp = pips_profit >= take_profit_pips
                hit_sl = pips_profit <= -stop_loss_pips
                
                if hit_tp or hit_sl:
                    profit = pips_profit * 0.0001 * position['size']
                    balance += profit
                    
                    exit_reason = 'TAKE_PROFIT' if hit_tp else 'STOP_LOSS'
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry'],
                        'exit_price': current_price,
                        'pips': pips_profit,
                        'profit': profit,
                        'balance': balance,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
            
            # Look for new signals if no position
            if position is None:
                # Enter LONG in bullish momentum (green zone)
                if bullish_momentum.iloc[i]:
                    position = {
                        'entry_time': current_time,
                        'entry': current_price,
                        'direction': 'LONG',
                        'size': 100000,  # 1 lot
                        'zone': 'GREEN'
                    }
                
                # Enter SHORT in bearish momentum (red zone)
                elif bearish_momentum.iloc[i]:
                    position = {
                        'entry_time': current_time,
                        'entry': current_price,
                        'direction': 'SHORT',
                        'size': 100000,  # 1 lot
                        'zone': 'RED'
                    }
            
            # Track equity
            current_equity = balance
            if position:
                if position['direction'] == 'LONG':
                    pips_profit = (current_price - position['entry']) / 0.0001
                else:
                    pips_profit = (position['entry'] - current_price) / 0.0001
                current_equity += pips_profit * 0.0001 * position['size']
            
            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity
            })
        
        # Close any remaining position
        if position:
            current_price = df['close'].iloc[-1]
            if position['direction'] == 'LONG':
                pips_profit = (current_price - position['entry']) / 0.0001
            else:
                pips_profit = (position['entry'] - current_price) / 0.0001
            
            profit = pips_profit * 0.0001 * position['size']
            balance += profit
            
            self.trades.append({
                'entry_time': position['entry_time'],
                'exit_time': df.index[-1],
                'direction': position['direction'],
                'entry_price': position['entry'],
                'exit_price': current_price,
                'pips': pips_profit,
                'profit': profit,
                'balance': balance,
                'exit_reason': 'END_OF_DATA'
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
        final_balance = trades_df['balance'].iloc[-1]
        
        # Risk/Reward
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        print(f"\n📊 Trade Statistics:")
        print(f"   Total trades:    {total_trades}")
        print(f"   Winning trades:  {winning_trades} ({win_rate:.1%})")
        print(f"   Losing trades:   {losing_trades}")
        print(f"   Win rate:        {win_rate:.1%}")
        
        print(f"\n💰 Performance:")
        print(f"   Total pips:      {total_pips:.1f}")
        print(f"   Avg win:         {avg_win:.1f} pips")
        print(f"   Avg loss:        {avg_loss:.1f} pips")
        print(f"   Risk/Reward:     1:{risk_reward:.2f}")
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
        
        # Plot equity curve
        self.plot_equity_curve()
        
        # Show sample trades
        print(f"\n📋 Sample Trades (last 10):")
        for _, trade in trades_df.tail(10).iterrows():
            emoji = "🟢" if trade['pips'] > 0 else "🔴"
            print(f"   {emoji} {trade['direction']:5s} | {trade['pips']:6.1f} pips | ${trade['profit']:7.2f} | {trade['exit_reason']}")
    
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
        
        ax1.set_title('Aggressive Momentum Scalper - Equity Curve', fontsize=14, fontweight='bold')
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
        
        filename = 'momentum_scalper_equity.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Equity curve saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  AGGRESSIVE MOMENTUM SCALPER BACKTEST")
    print("  Quick In & Out - Minimal Risk")
    print("="*60)
    
    scalper = MomentumScalper(
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not scalper.init_mt5():
        return
    
    # Run backtest
    scalper.backtest(
        price_threshold=0.10,   # 0.1% price change (very sensitive)
        adx_threshold=20,       # ADX > 20 (moderate trend)
        stop_loss_pips=10,      # Tight 10 pip stop
        take_profit_pips=5      # Quick 5 pip profit
    )
    
    mt5.shutdown()
    print(f"\n👋 Backtest Complete!")


if __name__ == "__main__":
    main()
