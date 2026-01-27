import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import time
import os
from datetime import datetime


class LiveOrderFlowTrader:
    """Live trading with Order Flow model - trailing stop strategy."""
    
    def __init__(self, model_path='model_order_flow.keras', symbol='EURUSDm', timeframe=mt5.TIMEFRAME_M15):
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = None
        self.scaler = StandardScaler()
        self.position_ticket = None
        self.position_info = None
        
        # Strategy parameters
        self.threshold = 0.7  # High quality signals only
        self.stop_loss_pips = 20
        self.trailing_stop_pips = 5
        self.lot_size = 0.01  # Start small!
        
    def init_mt5(self):
        """Initialize MT5 with specific path."""
        mt5_path = r"C:\Program Files\MetaTrader 5-2\terminal64.exe"
        
        if not mt5.initialize(path=mt5_path):
            print(f"❌ MT5 Init failed: {mt5.last_error()}")
            return False
        
        print(f"✅ Connected to MT5")
        print(f"   Terminal: {mt5_path}")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"   Account: {account_info.login}")
            print(f"   Balance: ${account_info.balance:.2f}")
            print(f"   Server: {account_info.server}")
        
        return True
    
    def load_model(self):
        """Load trained model."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        self.model = keras.models.load_model(self.model_path)
        print(f"✅ Loaded model: {self.model_path}")
        return True
    
    def fetch_data(self, bars=100):
        """Fetch recent CLOSED candles only (exclude current forming candle)."""
        # Fetch bars+1 to get enough data, then exclude the current forming candle
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars + 1)
        if rates is None:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # CRITICAL: Remove the last candle (current forming candle at index 0)
        # This ensures we only analyze CLOSED candles, matching backtest behavior
        df = df.iloc[:-1]
        
        return df
    
    def is_new_candle(self):
        """Check if a new candle has formed since last check."""
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 1, 1)  # Get last CLOSED candle
        if rates is None:
            return False, None
        
        last_closed_time = pd.to_datetime(rates[0]['time'], unit='s')
        return True, last_closed_time
    
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
    
    def predict_reversal(self, df):
        """Predict reversal probability."""
        features = self.calculate_order_flow_features(df)
        self.scaler.fit(features.values)
        latest_features = features.iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(latest_scaled, verbose=0)[0][0]
        return prediction, features.iloc[-1]
    
    def open_position(self, direction, current_price):
        """Open a position."""
        point = mt5.symbol_info(self.symbol).point
        
        if direction == 'LONG':
            order_type = mt5.ORDER_TYPE_BUY
            sl = current_price - self.stop_loss_pips * 10 * point
        else:
            order_type = mt5.ORDER_TYPE_SELL
            sl = current_price + self.stop_loss_pips * 10 * point
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": current_price,
            "sl": sl,
            "deviation": 20,
            "magic": 234000,
            "comment": "OrderFlow_AI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {result.retcode}")
            return False
        
        self.position_ticket = result.order
        self.position_info = {
            'direction': direction,
            'entry': current_price,
            'sl': sl,
            'max_profit_pips': 0
        }
        
        print(f"✅ {direction} position opened at {current_price:.5f}")
        print(f"   Stop loss: {sl:.5f}")
        return True
    
    def update_trailing_stop(self):
        """Update trailing stop if in profit."""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            self.position_ticket = None
            return
        
        position = positions[0]
        current_price = position.price_current
        point = mt5.symbol_info(self.symbol).point
        
        # Calculate profit in pips
        if position.type == mt5.ORDER_TYPE_BUY:
            pips_profit = (current_price - position.price_open) / (10 * point)
        else:
            pips_profit = (position.price_open - current_price) / (10 * point)
        
        # Update max profit
        if pips_profit > self.position_info['max_profit_pips']:
            self.position_info['max_profit_pips'] = pips_profit
        
        # Update trailing stop if in profit
        if pips_profit > 0:
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = current_price - self.trailing_stop_pips * 10 * point
                if new_sl > position.sl:
                    self.modify_position(position.ticket, new_sl)
            else:
                new_sl = current_price + self.trailing_stop_pips * 10 * point
                if new_sl < position.sl:
                    self.modify_position(position.ticket, new_sl)
    
    def modify_position(self, ticket, new_sl):
        """Modify position stop loss."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🔄 Trailing stop updated to {new_sl:.5f}")
    
    def calculate_seconds_until_next_candle(self):
        """Calculate seconds until next M15 candle opens."""
        now = datetime.now()
        
        # M15 candles open at: 00, 15, 30, 45 minutes
        current_minute = now.minute
        current_second = now.second
        
        # Find next candle open time
        if current_minute < 15:
            next_candle_minute = 15
        elif current_minute < 30:
            next_candle_minute = 30
        elif current_minute < 45:
            next_candle_minute = 45
        else:
            next_candle_minute = 0  # Next hour
        
        # Calculate seconds until next candle
        if next_candle_minute == 0:
            # Next hour
            seconds_until = (60 - current_minute - 1) * 60 + (60 - current_second)
        else:
            # Same hour
            minutes_until = next_candle_minute - current_minute - 1
            seconds_until = minutes_until * 60 + (60 - current_second)
        
        # Add 5 seconds buffer to ensure candle has closed
        return seconds_until + 5
    
    def run_live(self, check_interval=60):
        """Run live trading loop - only trades on CLOSED candles."""
        print(f"\n{'='*60}")
        print(f"  LIVE ORDER FLOW TRADING")
        print(f"  {self.symbol} | M15")
        print(f"{'='*60}")
        print(f"\n⚙️  Strategy:")
        print(f"   Reversal threshold: {self.threshold:.1%}")
        print(f"   Stop loss: {self.stop_loss_pips} pips")
        print(f"   Trailing stop: {self.trailing_stop_pips} pips")
        print(f"   Lot size: {self.lot_size}")
        print(f"\n⚠️  IMPORTANT: Only trades on CLOSED candles (no lookahead bias)")
        print(f"\n🔄 Starting live trading...")
        print(f"   Timeframe: M15 (checks when new candle opens)")
        print(f"   Press Ctrl+C to stop\n")
        
        last_candle_time = None
        
        try:
            while True:
                # Fetch CLOSED candles only
                df = self.fetch_data(bars=100)
                if df is None:
                    print(f"⚠️  Failed to fetch data, retrying in 30s...")
                    time.sleep(30)
                    continue
                
                # Get last CLOSED candle
                last_closed_candle_time = df.index[-1]
                current_price = df['close'].iloc[-1]
                
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                print(f"Last closed candle: {last_closed_candle_time}")
                print(f"Price: {current_price:.5f}")
                
                # Check if we have a position
                positions = mt5.positions_get(symbol=self.symbol)
                
                if positions:
                    # Update trailing stop (can do this anytime)
                    self.update_trailing_stop()
                    position = positions[0]
                    point = mt5.symbol_info(self.symbol).point
                    
                    if position.type == mt5.ORDER_TYPE_BUY:
                        pips = (current_price - position.price_open) / (10 * point)
                        print(f"📊 LONG position: {pips:+.1f} pips (Max: {self.position_info['max_profit_pips']:.1f})")
                    else:
                        pips = (position.price_open - current_price) / (10 * point)
                        print(f"📊 SHORT position: {pips:+.1f} pips (Max: {self.position_info['max_profit_pips']:.1f})")
                else:
                    # Only check for NEW signals when a NEW candle has CLOSED
                    if last_candle_time is None or last_closed_candle_time > last_candle_time:
                        print(f"🆕 New candle closed! Checking for signal...")
                        
                        # Look for new signal
                        prediction, features = self.predict_reversal(df)
                        
                        print(f"🎯 Reversal probability: {prediction:.1%}")
                        
                        if prediction >= self.threshold:
                            # Determine direction
                            direction = 'LONG' if features['buy_pressure'] > features['sell_pressure'] else 'SHORT'
                            
                            print(f"\n🚨 SIGNAL DETECTED!")
                            print(f"   Direction: {direction}")
                            print(f"   Confidence: {prediction:.1%}")
                            print(f"   Buy pressure: {features['buy_pressure']:.3f}")
                            print(f"   Sell pressure: {features['sell_pressure']:.3f}")
                            
                            # Get current market price for entry
                            tick = mt5.symbol_info_tick(self.symbol)
                            entry_price = tick.ask if direction == 'LONG' else tick.bid
                            
                            # Open position
                            self.open_position(direction, entry_price)
                        else:
                            print(f"⚪ No signal (threshold: {self.threshold:.1%})")
                        
                        # Update last candle time
                        last_candle_time = last_closed_candle_time
                    else:
                        print(f"⏸️  Waiting for new candle to close...")
                
                # Calculate smart sleep time
                seconds_until_next = self.calculate_seconds_until_next_candle()
                minutes = seconds_until_next // 60
                seconds = seconds_until_next % 60
                
                print(f"⏳ Next candle in {minutes}m {seconds}s...")
                time.sleep(seconds_until_next)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Live trading stopped")
            print(f"   Closing any open positions...")
            
            # Close all positions
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                for position in positions:
                    tick = mt5.symbol_info_tick(self.symbol)
                    close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "position": position.ticket,
                        "symbol": self.symbol,
                        "volume": position.volume,
                        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "price": close_price,
                        "deviation": 20,
                        "magic": 234000,
                        "comment": "Close_on_exit",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(request)
            
            print(f"✅ All positions closed")


def main():
    print("="*60)
    print("  LIVE ORDER FLOW TRADING BOT")
    print("  Demo Account - Start Small!")
    print("="*60)
    
    trader = LiveOrderFlowTrader(
        model_path='model_order_flow.keras',
        symbol='EURUSDm',
        timeframe=mt5.TIMEFRAME_M15
    )
    
    if not trader.init_mt5():
        return
    
    if not trader.load_model():
        mt5.shutdown()
        return
    
    # Run live trading
    trader.run_live(check_interval=60)  # Check every 60 seconds
    
    mt5.shutdown()
    print(f"\n👋 Complete")


if __name__ == "__main__":
    main()
