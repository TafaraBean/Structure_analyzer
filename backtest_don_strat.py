import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timezone, timedelta
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

# ==========================
# CONFIG (MATCH LIVE BOT)
# ==========================
SYMBOL = "XAUUSDm"
CORE_LOT = 0.01
STACK_LOT = 0.01

CORE_EARLY_KILL_LOSS = -40
CORE_EARLY_BE_PROFIT = 40
CORE_COOLDOWN_MINUTES = 5

STACK_TP_RAND = 20
STACK_MAX_LOSS_RAND = -25
STACK_TRAIL_KILL = 15

STACK_MIN_INTERVAL_SEC = 90
STACK_LOSS_PAUSE_SEC = 120

STARTUP_WARMUP_MINUTES = 1

# Spread simulation (typical XAU spread in points)
SPREAD = 0.3

# ==========================
# MT5 INIT
# ==========================
load_dotenv()
MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
MT5_PATH = r"C:\Program Files\MetaTrader 5-2\terminal64.exe"

if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print(f"❌ MT5 INIT FAILED: {mt5.last_error()}")
    raise RuntimeError("MT5 INIT FAILED")

# Check connection
account = mt5.account_info()
if account is None:
    print(f"❌ Cannot get account info: {mt5.last_error()}")
    mt5.shutdown()
    raise RuntimeError("MT5 not connected")

print(f"✅ MT5 Connected | Account: {account.login} | Balance: ${account.balance}")

# Select symbol
if not mt5.symbol_select(SYMBOL, True):
    print(f"❌ Failed to select symbol {SYMBOL}: {mt5.last_error()}")
    print("\nAvailable symbols containing 'XAU':")
    symbols = mt5.symbols_get()
    if symbols:
        for s in symbols:
            if 'XAU' in s.name:
                print(f"  - {s.name}")
    mt5.shutdown()
    raise RuntimeError(f"Symbol {SYMBOL} not available")

symbol_info = mt5.symbol_info(SYMBOL)
print(f"✅ Symbol {SYMBOL} selected | Spread: {symbol_info.spread}")

# ==========================
# FETCH HISTORICAL DATA
# ==========================
def fetch_data(start_date, end_date):
    """Fetch M1 data for the specified date range"""
    print(f"\nFetching M1 data from {start_date} to {end_date}...")
    
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"❌ Failed to fetch data: {error}")
        raise RuntimeError(f"Failed to fetch historical data: {error}")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    print(f"✅ Fetched {len(df)} candles")
    print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
    return df

# ==========================
# POSITION CLASS
# ==========================
class Position:
    def __init__(self, ticket, pos_type, entry_price, volume, comment, open_time):
        self.ticket = ticket
        self.type = pos_type  # 0=BUY, 1=SELL
        self.price_open = entry_price
        self.volume = volume
        self.comment = comment
        self.open_time = open_time
        self.sl = 0
        self.profit = 0
        self.peak_profit = 0
        
    def update_profit(self, current_price):
        """Calculate current P&L"""
        if self.type == 0:  # BUY
            points = current_price - self.price_open
        else:  # SELL
            points = self.price_open - current_price
        
        # Convert points to dollars (XAU: $1 per 0.01 lot per point)
        self.profit = points * (self.volume / 0.01)
        self.peak_profit = max(self.peak_profit, self.profit)
        
        return self.profit

# ==========================
# BACKTESTING ENGINE
# ==========================
class DonStratBacktester:
    def __init__(self, data):
        self.data = data
        self.positions = []
        self.closed_trades = []
        self.next_ticket = 1
        
        # State variables (match live bot)
        self.core_position = None
        self.stack_positions = []
        self.core_open_candle_time = None
        self.core_cooldown_until = None
        self.last_stack_time = None
        self.stack_pause_until = None
        self.startup_until = None
        
        # Performance tracking
        self.equity_curve = []
        self.balance = 0
        
    def get_entry_price(self, pos_type, candle):
        """Simulate realistic entry price with spread"""
        if pos_type == 0:  # BUY
            return candle['close'] + SPREAD/2
        else:  # SELL
            return candle['close'] - SPREAD/2
    
    def get_exit_price(self, pos_type, candle):
        """Simulate realistic exit price with spread"""
        if pos_type == 0:  # BUY (close with sell)
            return candle['close'] - SPREAD/2
        else:  # SELL (close with buy)
            return candle['close'] + SPREAD/2
    
    def open_position(self, pos_type, candle, comment):
        """Open a new position"""
        entry_price = self.get_entry_price(pos_type, candle)
        volume = CORE_LOT if comment == "CORE" else STACK_LOT
        
        pos = Position(
            ticket=self.next_ticket,
            pos_type=pos_type,
            entry_price=entry_price,
            volume=volume,
            comment=comment,
            open_time=candle['time']
        )
        
        self.next_ticket += 1
        
        if comment == "CORE":
            self.core_position = pos
            self.core_open_candle_time = candle['time']
        else:
            self.stack_positions.append(pos)
            self.last_stack_time = candle['time']
        
        return pos
    
    def close_position(self, pos, candle, reason=""):
        """Close a position and record the trade"""
        exit_price = self.get_exit_price(pos.type, candle)
        
        # Calculate final P&L
        if pos.type == 0:  # BUY
            points = exit_price - pos.price_open
        else:  # SELL
            points = pos.price_open - exit_price
        
        final_pnl = points * (pos.volume / 0.01)
        
        trade = {
            'ticket': pos.ticket,
            'type': 'BUY' if pos.type == 0 else 'SELL',
            'comment': pos.comment,
            'entry_time': pos.open_time,
            'exit_time': candle['time'],
            'entry_price': pos.price_open,
            'exit_price': exit_price,
            'pnl': final_pnl,
            'reason': reason
        }
        
        self.closed_trades.append(trade)
        self.balance += final_pnl
        
        # Remove from active positions
        if pos.comment == "CORE":
            self.core_position = None
        else:
            self.stack_positions = [p for p in self.stack_positions if p.ticket != pos.ticket]
        
        return final_pnl
    
    def maybe_open_core(self, idx, candle, current_time):
        """CORE entry logic (exact replica)"""
        # Warmup period
        if self.startup_until and current_time < self.startup_until:
            return
        
        # Already have CORE
        if self.core_position:
            return
        
        # Cooldown active
        if self.core_cooldown_until and current_time < self.core_cooldown_until:
            return
        
        # Need at least 2 candles
        if idx < 1:
            return
        
        # Analyze previous closed candle
        prev_candle = self.data.iloc[idx - 1]
        
        open_price = prev_candle['open']
        close_price = prev_candle['close']
        high = prev_candle['high']
        low = prev_candle['low']
        
        body = close_price - open_price
        range_ = high - low
        
        # Ignore doji
        if abs(body) < 0.6:
            return
        
        # Require commitment (body ≥ 50% of candle)
        if range_ == 0 or abs(body) / range_ < 0.5:
            return
        
        # Determine direction
        direction = 0 if body > 0 else 1  # 0=BUY, 1=SELL
        
        # Open CORE
        self.open_position(direction, candle, "CORE")
    
    def manage_core(self, candle, current_time):
        """CORE management logic (exact replica)"""
        if not self.core_position:
            return
        
        # Update P&L
        self.core_position.update_profit(candle['close'])
        pnl = self.core_position.profit
        
        # Early kill at -$40
        if pnl <= CORE_EARLY_KILL_LOSS:
            self.close_position(self.core_position, candle, "CORE_EARLY_KILL")
            self.core_cooldown_until = current_time + timedelta(minutes=CORE_COOLDOWN_MINUTES)
            return
        
        # Move to BE at +$40
        if pnl >= CORE_EARLY_BE_PROFIT and self.core_position.sl == 0:
            self.core_position.sl = self.core_position.price_open
    
    def maybe_open_stack(self, candle, current_time):
        """STACK entry logic (exact replica)"""
        # Need CORE at BE
        if not self.core_position or self.core_position.sl != self.core_position.price_open:
            return
        
        # Already have STACK
        if self.stack_positions:
            return
        
        # Pause cooldown
        if self.stack_pause_until and current_time < self.stack_pause_until:
            return
        
        # Spacing requirement
        if self.last_stack_time:
            elapsed = (current_time - self.last_stack_time).total_seconds()
            if elapsed < STACK_MIN_INTERVAL_SEC:
                return
        
        # Open STACK in same direction as CORE
        self.open_position(self.core_position.type, candle, "STACK")
    
    def manage_stacks(self, candle, current_time):
        """STACK management logic (exact replica)"""
        for stack in self.stack_positions[:]:  # Copy list to allow removal
            stack.update_profit(candle['close'])
            pnl = stack.profit
            peak = stack.peak_profit
            
            # Trailing kill: if peak ≥ +$10, kill if drops $15 from peak
            if peak >= 10 and pnl <= peak - STACK_TRAIL_KILL:
                self.close_position(stack, candle, "STACK_TRAIL_KILL")
                self.stack_pause_until = current_time + timedelta(seconds=STACK_LOSS_PAUSE_SEC)
                continue
            
            # Take profit at +$20
            if pnl >= STACK_TP_RAND:
                self.close_position(stack, candle, "STACK_TP")
                continue
            
            # Stop loss at -$25
            if pnl <= STACK_MAX_LOSS_RAND:
                self.close_position(stack, candle, "STACK_SL")
                self.stack_pause_until = current_time + timedelta(seconds=STACK_LOSS_PAUSE_SEC)
    
    def run(self):
        """Main backtest loop"""
        print("\n" + "="*60)
        print("STARTING BACKTEST")
        print("="*60)
        
        # Set startup warmup
        self.startup_until = self.data.iloc[0]['time'] + timedelta(minutes=STARTUP_WARMUP_MINUTES)
        
        for idx, row in self.data.iterrows():
            candle = row
            current_time = candle['time']
            
            # Update all active positions
            if self.core_position:
                self.core_position.update_profit(candle['close'])
            
            for stack in self.stack_positions:
                stack.update_profit(candle['close'])
            
            # Execute strategy logic (same order as live bot)
            self.maybe_open_core(idx, candle, current_time)
            self.manage_core(candle, current_time)
            self.manage_stacks(candle, current_time)
            self.maybe_open_stack(candle, current_time)
            
            # Track equity
            total_floating = sum(p.profit for p in [self.core_position] + self.stack_positions if p)
            equity = self.balance + total_floating
            self.equity_curve.append({
                'time': current_time,
                'balance': self.balance,
                'equity': equity
            })
        
        # Close any remaining positions at end
        if self.core_position:
            self.close_position(self.core_position, self.data.iloc[-1], "END_OF_BACKTEST")
        
        for stack in self.stack_positions[:]:
            self.close_position(stack, self.data.iloc[-1], "END_OF_BACKTEST")
        
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)

# ==========================
# PERFORMANCE ANALYSIS
# ==========================
def analyze_performance(backtester):
    """Generate comprehensive performance report"""
    trades_df = pd.DataFrame(backtester.closed_trades)
    equity_df = pd.DataFrame(backtester.equity_curve)
    
    if len(trades_df) == 0:
        print("\n⚠️ NO TRADES EXECUTED")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE REPORT")
    print("="*60)
    
    # Overall metrics
    total_trades = len(trades_df)
    core_trades = trades_df[trades_df['comment'] == 'CORE']
    stack_trades = trades_df[trades_df['comment'] == 'STACK']
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = trades_df['pnl'].sum()
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    max_win = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()
    
    # Drawdown calculation
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total Trades: {total_trades}")
    print(f"  CORE Trades: {len(core_trades)}")
    print(f"  STACK Trades: {len(stack_trades)}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Max Drawdown: ${max_drawdown:.2f}")
    
    print(f"\n💰 PROFIT/LOSS")
    print(f"  Winning Trades: {len(winning_trades)} (Avg: ${avg_win:.2f})")
    print(f"  Losing Trades: {len(losing_trades)} (Avg: ${avg_loss:.2f})")
    print(f"  Largest Win: ${max_win:.2f}")
    print(f"  Largest Loss: ${max_loss:.2f}")
    
    # CORE vs STACK breakdown
    print(f"\n🧱 CORE PERFORMANCE")
    core_pnl = core_trades['pnl'].sum()
    core_wins = len(core_trades[core_trades['pnl'] > 0])
    core_win_rate = core_wins / len(core_trades) * 100 if len(core_trades) > 0 else 0
    print(f"  Total P&L: ${core_pnl:.2f}")
    print(f"  Win Rate: {core_win_rate:.1f}%")
    
    print(f"\n📦 STACK PERFORMANCE")
    stack_pnl = stack_trades['pnl'].sum()
    stack_wins = len(stack_trades[stack_trades['pnl'] > 0])
    stack_win_rate = stack_wins / len(stack_trades) * 100 if len(stack_trades) > 0 else 0
    print(f"  Total P&L: ${stack_pnl:.2f}")
    print(f"  Win Rate: {stack_win_rate:.1f}%")
    
    # Exit reasons
    print(f"\n🚪 EXIT REASONS")
    for reason, count in trades_df['reason'].value_counts().items():
        print(f"  {reason}: {count}")
    
    return trades_df, equity_df

# ==========================
# VISUALIZATION
# ==========================
def create_visualizations(backtester, trades_df, equity_df, data):
    """Create comprehensive visualization charts"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Price chart with trade markers
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data['time'], data['close'], label='Price', linewidth=0.8, alpha=0.7)
    
    # Mark CORE trades
    core_entries = trades_df[trades_df['comment'] == 'CORE']
    for _, trade in core_entries.iterrows():
        color = 'green' if trade['pnl'] > 0 else 'red'
        ax1.scatter(trade['entry_time'], trade['entry_price'], marker='^' if trade['type'] == 'BUY' else 'v',
                   color=color, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    ax1.set_title('Price Chart with CORE Entries', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Equity curve
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(equity_df['time'], equity_df['equity'], label='Equity', linewidth=2, color='blue')
    ax2.plot(equity_df['time'], equity_df['balance'], label='Balance', linewidth=1, alpha=0.7, color='orange')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(equity_df['time'], 0, equity_df['equity'], alpha=0.2)
    ax2.set_title('Equity Curve', fontweight='bold')
    ax2.set_ylabel('P&L ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Drawdown chart
    ax3 = plt.subplot(3, 2, 3)
    ax3.fill_between(equity_df['time'], 0, equity_df['drawdown'], color='red', alpha=0.3)
    ax3.plot(equity_df['time'], equity_df['drawdown'], color='darkred', linewidth=1.5)
    ax3.set_title('Drawdown', fontweight='bold')
    ax3.set_ylabel('Drawdown ($)')
    ax3.grid(True, alpha=0.3)
    
    # 4. P&L distribution
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('P&L Distribution', fontweight='bold')
    ax4.set_xlabel('P&L ($)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    # 5. CORE vs STACK comparison
    ax5 = plt.subplot(3, 2, 5)
    core_pnl = trades_df[trades_df['comment'] == 'CORE']['pnl'].sum()
    stack_pnl = trades_df[trades_df['comment'] == 'STACK']['pnl'].sum()
    colors = ['green' if x > 0 else 'red' for x in [core_pnl, stack_pnl]]
    ax5.bar(['CORE', 'STACK'], [core_pnl, stack_pnl], color=colors, edgecolor='black', linewidth=2)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_title('CORE vs STACK P&L', fontweight='bold')
    ax5.set_ylabel('Total P&L ($)')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cumulative P&L over time
    ax6 = plt.subplot(3, 2, 6)
    trades_df_sorted = trades_df.sort_values('exit_time')
    trades_df_sorted['cumulative_pnl'] = trades_df_sorted['pnl'].cumsum()
    ax6.plot(trades_df_sorted['exit_time'], trades_df_sorted['cumulative_pnl'], 
            linewidth=2, marker='o', markersize=3)
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax6.set_title('Cumulative P&L', fontweight='bold')
    ax6.set_ylabel('Cumulative P&L ($)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = r'c:\Users\tafar\OneDrive\Desktop\Structure_bot\backtest_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    plt.show()

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    # Define backtest period - last 7 days from now
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)
    
    print(f"\n🔍 Attempting to fetch data from {start_date} to {end_date}")
    
    # Try to fetch data
    try:
        data = fetch_data(start_date, end_date)
    except RuntimeError as e:
        print(f"\n⚠️ Failed to fetch data for the specified range.")
        print(f"Trying alternative: last 5 days of available data...")
        
        # Try shorter period
        start_date = end_date - timedelta(days=5)
        try:
            data = fetch_data(start_date, end_date)
        except:
            print("\n❌ Could not fetch historical data. Possible reasons:")
            print("  1. MT5 terminal is not running")
            print("  2. Symbol 'XAUUSDz' not available")
            print("  3. No historical data for this period")
            print("\nTrying to fetch last 1000 bars instead...")
            
            # Last resort: use copy_rates_from_pos
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 10000)
            if rates is None or len(rates) == 0:
                print("\n❌ FATAL: Cannot fetch any historical data")
                mt5.shutdown()
                sys.exit(1)
            
            data = pd.DataFrame(rates)
            data['time'] = pd.to_datetime(data['time'], unit='s')
            print(f"✅ Fetched {len(data)} candles (most recent available)")
            print(f"Date range: {data['time'].min()} to {data['time'].max()}")
    
    # Run backtest
    backtester = DonStratBacktester(data)
    backtester.run()
    
    # Analyze performance
    trades_df, equity_df = analyze_performance(backtester)
    
    # Create visualizations
    if trades_df is not None:
        create_visualizations(backtester, trades_df, equity_df, data)
    
    # Save detailed trade log
    if trades_df is not None:
        output_csv = r'c:\Users\tafar\OneDrive\Desktop\Structure_bot\backtest_trades.csv'
        trades_df.to_csv(output_csv, index=False)
        print(f"\n💾 Trade log saved to: {output_csv}")
    
    mt5.shutdown()
    print("\n✅ Backtest complete!")
