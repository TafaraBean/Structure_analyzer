import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class OrderFlowBacktest:
    """Backtest Order Flow model with quick profit-taking strategy."""
    
    def __init__(self, model_path='model_order_flow.keras', symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.scaler = StandardScaler()
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
    
    def load_model(self):
        """Load trained model."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Loaded model: {self.model_path}")
        return True
    
    def fetch_test_data(self):
        """Fetch ONLY the held-out test set (last 15% of original 3000 bars)."""
        # Original training used 3000 bars
        # Train: 70% = 2100 bars
        # Val: 15% = 450 bars  
        # Test: 15% = 450 bars (bars 2551-3000)
        
        # We need to fetch 3000 bars and take the last 450 (test set only)
        total_bars = 3000
        test_bars = int(0.15 * total_bars)  # 450 bars
        
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, total_bars)
        
        if rates is None:
            print(f"❌ Failed to fetch data")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Take ONLY the last 450 bars (test set that model never saw)
        df_test = df.iloc[-test_bars:]
        
        print(f"✅ Loaded {len(df_test)} bars (HELD-OUT TEST SET)")
        print(f"   From: {df_test.index[0]}")
        print(f"   To:   {df_test.index[-1]}")
        print(f"   ⚠️  This data was NOT used in training!")
        
        return df_test
    
    def calculate_order_flow_features(self, df):
        """Calculate Order Flow features."""
        features = pd.DataFrame(index=df.index)
        
        features['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        features['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        features['vwap_dist'] = (df['close'] - ((df['high'] + df['low'] + df['close']) / 3))
        
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        features['ad_line'] = (mfm * df['tick_volume']).cumsum()
        
        obv = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ma'] = obv.rolling(20).mean()
        
        features.fillna(0, inplace=True)
        return features
    
    def run_backtest(self, threshold=0.7, stop_loss_pips=20, trailing_stop_pips=5):
        """
        Run backtest with trailing stop strategy.
        
        Args:
            threshold: Reversal probability threshold (0-1) - HIGHER for quality
            stop_loss_pips: Initial stop loss in pips
            trailing_stop_pips: Trailing stop distance in pips
        """
        print(f"\n{'='*60}")
        print(f"  ORDER FLOW BACKTEST - HELD-OUT TEST SET")
        print(f"  {self.symbol} | M15")
        print(f"{'='*60}")
        print(f"\n⚙️  Strategy:")
        print(f"   Reversal threshold: {threshold:.1%} (HIGH QUALITY)")
        print(f"   Initial stop loss: {stop_loss_pips} pips")
        print(f"   Trailing stop: {trailing_stop_pips} pips (locks in profit)")
        
        # Fetch ONLY test data (unseen during training)
        df = self.fetch_test_data()
        if df is None:
            return
        
        # Calculate features
        features = self.calculate_order_flow_features(df)
        
        # Fit scaler on all data
        self.scaler.fit(features.values)
        
        # Simulate trading
        balance = 10000  # Starting balance
        position = None  # Current position
        
        print(f"\n🔄 Running backtest...")
        
        for i in range(100, len(df)):  # Start after 100 bars for indicators
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check if we have an open position
            if position:
                # Calculate current P&L
                if position['direction'] == 'LONG':
                    pips_profit = (current_price - position['entry']) / 0.0001
                else:  # SHORT
                    pips_profit = (position['entry'] - current_price) / 0.0001
                
                # Update trailing stop if in profit
                if pips_profit > 0:
                    # Calculate new trailing stop level
                    if position['direction'] == 'LONG':
                        new_stop = current_price - (trailing_stop_pips * 0.0001)
                        position['stop_loss'] = max(position['stop_loss'], new_stop)
                    else:  # SHORT
                        new_stop = current_price + (trailing_stop_pips * 0.0001)
                        position['stop_loss'] = min(position['stop_loss'], new_stop)
                
                # Check if stop loss hit
                stop_hit = False
                if position['direction'] == 'LONG':
                    stop_hit = current_price <= position['stop_loss']
                else:  # SHORT
                    stop_hit = current_price >= position['stop_loss']
                
                if stop_hit:
                    # Calculate final P&L
                    if position['direction'] == 'LONG':
                        exit_pips = (position['stop_loss'] - position['entry']) / 0.0001
                    else:
                        exit_pips = (position['entry'] - position['stop_loss']) / 0.0001
                    
                    profit = exit_pips * 0.0001 * position['size']
                    balance += profit
                    
                    # Determine exit reason
                    exit_reason = 'TRAILING_STOP' if exit_pips > 0 else 'STOP_LOSS'
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry'],
                        'exit_price': position['stop_loss'],
                        'pips': exit_pips,
                        'profit': profit,
                        'balance': balance,
                        'exit_reason': exit_reason,
                        'max_profit': position.get('max_profit', 0)
                    })
                    
                    position = None
                else:
                    # Track maximum profit reached
                    if 'max_profit' not in position or pips_profit > position['max_profit']:
                        position['max_profit'] = pips_profit
            
            # Look for new signals if no position
            if position is None:
                # Get features up to current bar
                current_features = features.iloc[:i+1]
                latest_features = current_features.iloc[-1:].values
                latest_scaled = self.scaler.transform(latest_features)
                
                # Predict
                prediction = self.model.predict(latest_scaled, verbose=0)[0][0]
                
                # Check for signal (HIGHER THRESHOLD for quality)
                if prediction >= threshold:
                    # Determine direction
                    buy_pressure = features['buy_pressure'].iloc[i]
                    sell_pressure = features['sell_pressure'].iloc[i]
                    
                    direction = 'LONG' if buy_pressure > sell_pressure else 'SHORT'
                    
                    # Set initial stop loss
                    if direction == 'LONG':
                        initial_stop = current_price - (stop_loss_pips * 0.0001)
                    else:
                        initial_stop = current_price + (stop_loss_pips * 0.0001)
                    
                    # Open position
                    position = {
                        'entry_time': current_time,
                        'entry': current_price,
                        'direction': direction,
                        'size': 100000,  # 1 lot
                        'probability': prediction,
                        'stop_loss': initial_stop,
                        'max_profit': 0
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
                'exit_reason': 'END_OF_DATA',
                'max_profit': position.get('max_profit', 0)
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
        
        print(f"\n📊 Trade Statistics:")
        print(f"   Total trades:    {total_trades}")
        print(f"   Winning trades:  {winning_trades} ({win_rate:.1%})")
        print(f"   Losing trades:   {losing_trades}")
        print(f"   Win rate:        {win_rate:.1%}")
        
        print(f"\n💰 Performance:")
        print(f"   Total pips:      {total_pips:.1f}")
        print(f"   Avg win:         {avg_win:.1f} pips")
        print(f"   Avg loss:        {avg_loss:.1f} pips")
        print(f"   Total profit:    ${total_profit:.2f}")
        print(f"   Final balance:   ${final_balance:.2f}")
        print(f"   Return:          {(final_balance - 10000) / 10000 * 100:.2f}%")
        
        # Sharpe Ratio
        returns = trades_df['profit'] / 10000  # Returns as percentage
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:    {sharpe:.2f}")
        
        # Exit reasons
        print(f"\n🚪 Exit Reasons:")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            print(f"   {reason}: {count}")
        
        # Plot equity curve
        self.plot_equity_curve()
        
        # Show sample trades
        print(f"\n📋 Sample Trades (last 5):")
        for _, trade in trades_df.tail(5).iterrows():
            emoji = "🟢" if trade['pips'] > 0 else "🔴"
            print(f"   {emoji} {trade['direction']:5s} | {trade['pips']:6.1f} pips | ${trade['profit']:7.2f} | {trade['exit_reason']}")
    
    def plot_equity_curve(self):
        """Plot equity curve."""
        equity_df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['time'], equity_df['equity'], linewidth=2, color='#2196F3')
        plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Balance')
        plt.fill_between(equity_df['time'], 10000, equity_df['equity'], 
                         where=(equity_df['equity'] >= 10000), alpha=0.3, color='green', label='Profit')
        plt.fill_between(equity_df['time'], 10000, equity_df['equity'], 
                         where=(equity_df['equity'] < 10000), alpha=0.3, color='red', label='Loss')
        
        plt.title('Order Flow Strategy - Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        filename = 'order_flow_equity_curve.png'
        plt.savefig(filename, dpi=150)
        print(f"\n📊 Equity curve saved: {filename}")
        plt.show()


def main():
    print("="*60)
    print("  ORDER FLOW BACKTEST - HELD-OUT TEST SET")
    print("  Testing on unseen data (last 15% of training data)")
    print("="*60)
    
    backtest = OrderFlowBacktest(
        model_path='model_order_flow.keras',
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not backtest.init_mt5():
        return
    
    if not backtest.load_model():
        return
    
    # Run backtest with trailing stop + high quality signals
    backtest.run_backtest(
        threshold=0.7,           # Higher threshold for quality (was 0.6)
        stop_loss_pips=20,       # Initial stop loss
        trailing_stop_pips=5     # Trail by 5 pips to lock in profits
    )
    
    mt5.shutdown()
    print(f"\n👋 Backtest Complete!")


if __name__ == "__main__":
    main()
