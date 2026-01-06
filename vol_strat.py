import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import time
import os
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# CREDENTIALS
LOGIN = int(os.getenv("MT5_LOGIN"))
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")

# TRADING SETTINGS
SYMBOL = "XAUUSDz"       
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_MINUTES = 5
LOT_SIZE = 0.35          
MAGIC_NUMBER = 123456   
DEVIATION = 20          
LOG_FILE = "trade_log.csv"

# STRATEGY PARAMETERS
ATR_PERIOD = 14
VOL_THRESHOLD_MA = 105   # Moving Average of ATR to define "High Vol"
TREND_MA_PERIOD = 50    # To determine Trend Direction (Velocity)
SL_ATR_MULT = 1.00       
TP_ATR_MULT = 2.00       

# RISK MANAGEMENT
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_BARS = 60      
USE_TRAILING = True
TRAIL_TRIGGER_PCT = 0.5 
MAX_HOLD_BARS = 50     

# GLOBAL STATE
consecutive_losses = 0
cooldown_until = datetime.min
last_processed_candle_time = None

def initialize_mt5():
    if not mt5.initialize():
        print(f"Startup Failed: {mt5.last_error()}")
        return False
    if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
        print(f"Login Failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    print(f"✅ Connected to {SERVER} as {LOGIN}")
    return True

def log_trade_attempt(symbol, signal, price, sl, tp, volume, status, comment, ticket, atr_val=0, vel_val=0, retcode=0):
    file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Symbol", "Type", "Price", "SL", "TP", "Volume", "Status", "RetCode", "Comment", "Ticket", "ATR", "Trend_Vel"])
            
            type_str = "ACTION"
            if signal == 1: type_str = "BUY"
            elif signal == -1: type_str = "SELL"
            elif signal == 99: type_str = "CLOSE_BUY"
            elif signal == -99: type_str = "CLOSE_SELL"
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, type_str, f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}",
                volume, status, retcode, comment, ticket, f"{atr_val:.4f}", f"{vel_val:.5f}"
            ])
            print(f"📝 Logged to {LOG_FILE} (RetCode: {retcode})")
    except Exception as e:
        print(f"⚠️ Failed to write log: {e}")

def get_closed_candles(symbol, timeframe, n=200):
    # Fetch extra candles to ensure indicators (SMA 50) are stable
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n + 1)
    if rates is None or len(rates) == 0: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.iloc[:-1] 
    return df

def check_for_signal(df):
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    op = df['open'].values
    
    # 1. Volatility Calculations
    atr = talib.ATR(hi, lo, cl, timeperiod=ATR_PERIOD)
    atr_ma = talib.SMA(atr, timeperiod=VOL_THRESHOLD_MA)
    
    # "Smart" Volatility Logic:
    # Condition: Current ATR < Prev ATR (Declining)
    # AND Prev ATR > Prev Threshold (It was high just a moment ago)
    # Using [-1] for current closed candle and [-2] for the one before it
    prev_atr = atr[-2]
    curr_atr = atr[-1]
    prev_atr_ma = atr_ma[-2]
    
    # The Decline Trigger:
    is_declining = (curr_atr < prev_atr) and (prev_atr > prev_atr_ma)
    
    # 2. Trend Calculations (Velocity)
    trend_ma = talib.SMA(cl, timeperiod=TREND_MA_PERIOD)
    curr_vel = trend_ma[-1] - trend_ma[-2] # Positive = UpTrend
    
    if not is_declining:
        # We return 0 signal but pass data for logging/debugging
        return 0, curr_atr, curr_vel

    # 3. Patterns
    engulf = talib.CDLENGULFING(op, hi, lo, cl)[-1]
    hammer = talib.CDLHAMMER(op, hi, lo, cl)[-1]
    star = talib.CDLSHOOTINGSTAR(op, hi, lo, cl)[-1]
    hanging = talib.CDLHANGINGMAN(op, hi, lo, cl)[-1]
    
    signal = 0
    
    # BUY LOGIC (Includes Trend Filter)
    if (engulf == 100) or (hammer == 100):
        if curr_vel > 0: # Only Buy if Trend is Up
            signal = 1
        else:
            print(f"⚠️ Buy Signal Ignored: Against Trend (Vel: {curr_vel:.4f})")
            
    # SELL LOGIC (Counter-trend allowed based on previous analysis, or restrict if you prefer)
    if (engulf == -100) or (star == -100) or (hanging == -100):
        signal = -1 if signal != 1 else 0
            
    return signal, curr_atr, curr_vel

def execute_trade(symbol, signal, atr, lot, vel):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return False
    
    sl_dist = SL_ATR_MULT * atr
    tp_dist = TP_ATR_MULT * atr
    
    price = tick.ask if signal == 1 else tick.bid
    sl = price - sl_dist if signal == 1 else price + sl_dist
    tp = price + tp_dist if signal == 1 else price - tp_dist
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": "SmartBot_v6",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    status = "SUCCESS" if result.retcode == mt5.TRADE_RETCODE_DONE else "FAILED"
    
    log_trade_attempt(symbol, signal, price, sl, tp, lot, status, result.comment, result.order, atr, vel, result.retcode)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order Failed: {result.comment} (Code: {result.retcode})")
        return False
    return True

def close_position(position, reason="Time Stop"):
    tick = mt5.symbol_info_tick(position.symbol)
    order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": reason,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    # Log close
    s_code = 99 if order_type == mt5.ORDER_TYPE_BUY else -99
    status = "CLOSED" if result.retcode == mt5.TRADE_RETCODE_DONE else "CLOSE_FAIL"
    log_trade_attempt(position.symbol, s_code, price, 0, 0, position.volume, status, f"{reason}: {result.comment}", position.ticket, 0, 0, result.retcode)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"⏳ Trade Closed ({reason}) | Ticket: {position.ticket}")

def check_time_stops():
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if positions is None: return
    current_server_time = mt5.symbol_info_tick(SYMBOL).time
    max_duration_sec = MAX_HOLD_BARS * TIMEFRAME_MINUTES * 60

    for pos in positions:
        if (current_server_time - pos.time) > max_duration_sec:
            print(f"⚠️ Time Stop: {pos.ticket}")
            close_position(pos, reason="Time Stop")

def manage_trailing_stops():
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if positions is None: return

    for pos in positions:
        tick = mt5.symbol_info_tick(SYMBOL)
        current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        if abs(pos.tp - pos.price_open) < 0.00001: continue

        dist_covered = abs(current_price - pos.price_open)
        total_dist = abs(pos.tp - pos.price_open)
        
        if (dist_covered / total_dist) >= TRAIL_TRIGGER_PCT:
            initial_risk = abs(pos.price_open - pos.sl)
            new_sl = 0.0
            update = False
            
            if pos.type == mt5.ORDER_TYPE_BUY:
                proposed = current_price - initial_risk
                if proposed > pos.sl:
                    new_sl, update = proposed, True
            else:
                proposed = current_price + initial_risk
                if proposed < pos.sl:
                    new_sl, update = proposed, True
            
            if update:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                    "magic": MAGIC_NUMBER
                }
                mt5.order_send(req)
                print(f"⛓️ Trailing Updated: {pos.ticket}")

def check_history_for_losses():
    global consecutive_losses
    from_date = datetime.now() - timedelta(hours=24)
    history = mt5.history_deals_get(from_date, datetime.now(), group="*")
    if history is None: return
    
    my_trades = [h for h in history if h.magic == MAGIC_NUMBER and h.entry == mt5.DEAL_ENTRY_OUT]
    my_trades.sort(key=lambda x: x.time)
    
    streak = 0
    for trade in my_trades:
        if trade.profit < 0: streak += 1
        elif trade.profit > 0: streak = 0
    consecutive_losses = streak

# --- MAIN LOOP ---
if __name__ == "__main__":
    if initialize_mt5():
        print("🤖 Smart Bot V6 Started (Smart Filters Active)...")
        
        while True:
            try:
                # 1. Maintenance
                if USE_TRAILING: manage_trailing_stops()
                check_time_stops()
                
                # 2. Candle Scan
                check_df = get_closed_candles(SYMBOL, TIMEFRAME, n=2)
                if check_df is not None:
                    latest_close_time = check_df['time'].iloc[-1]
                    
                    if latest_close_time != last_processed_candle_time:
                        print(f"\n🔎 Scanning: {latest_close_time}")
                        
                        if datetime.now() < cooldown_until:
                            print(f"❄️ Cooldown Active. (Losses: {consecutive_losses})")
                        else:
                            # Fetch 200 candles for stable SMA 50 calculation
                            df = get_closed_candles(SYMBOL, TIMEFRAME, n=200)
                            sig, atr, vel = check_for_signal(df)
                            
                            if sig != 0:
                                print(f"🚀 Signal: {'BUY' if sig==1 else 'SELL'} | ATR: {atr:.2f} | Vel: {vel:.5f}")
                                check_history_for_losses()
                                
                                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                                    print("🛑 Max Losses Hit. Cooling Down.")
                                    cooldown_until = datetime.now() + timedelta(minutes=5 * COOLDOWN_BARS)
                                else:
                                    execute_trade(SYMBOL, sig, atr, LOT_SIZE, vel)
                            else:
                                print(f"💤 No Signal. (ATR: {atr:.2f} | Declining? {atr < df['high'].values[-1]*0.001})") 
                        
                        last_processed_candle_time = latest_close_time
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                mt5.shutdown()
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(5)