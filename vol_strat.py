import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import time
import os
import csv
import requests  # <--- NEW IMPORT
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# CREDENTIALS
LOGIN = int(os.getenv("MT5_LOGIN"))
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL") # <--- NEW CONFIG

# TRADING SETTINGS
SYMBOL = "XAUUSDz"       
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_MINUTES = 5
LOT_SIZE = 0.01          
MAGIC_NUMBER = 123456   
DEVIATION = 20          
LOG_FILE = "trade_log.csv"

# STRATEGY PARAMETERS
ATR_PERIOD = 14
VOL_THRESHOLD_MA = 30    
SL_ATR_MULT = 1.75       
TP_ATR_MULT = 3.42       

# RISK MANAGEMENT
MAX_CONSECUTIVE_LOSSES = 8
COOLDOWN_BARS = 60      
USE_TRAILING = True
TRAIL_TRIGGER_PCT = 0.33 
MAX_HOLD_BARS = 50 

# GLOBAL STATE
consecutive_losses = 0
cooldown_until = datetime.min
last_processed_candle_time = None

# --- DISCORD FUNCTION ---
def send_discord_alert(title, message, color=0x3498db):
    """
    Sends an embedded message to Discord.
    Colors: Green (0x2ecc71), Red (0xe74c3c), Blue (0x3498db), Orange (0xe67e22)
    """
    if not DISCORD_URL: return

    data = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "footer": {"text": f"BaseBot • {datetime.now().strftime('%H:%M:%S')}"}
        }]
    }
    try:
        requests.post(DISCORD_URL, json=data)
    except Exception as e:
        print(f"⚠️ Discord Error: {e}")

# --- MT5 FUNCTIONS ---
def initialize_mt5():
    if not mt5.initialize():
        print(f"Startup Failed: {mt5.last_error()}")
        return False
    if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
        print(f"Login Failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    print(f"✅ Connected to {SERVER} as {LOGIN}")
    send_discord_alert("🤖 Bot Started", f"Connected to {SERVER}\nSymbol: {SYMBOL}", 0x3498db) # <--- ALERT
    return True

def log_trade_attempt(symbol, signal, price, sl, tp, volume, status, comment, ticket, atr_val=0, retcode=0):
    # ... (Keep your existing CSV logging logic exactly as is) ...
    # I am omitting the full CSV code block here to save space, 
    # but keep your original `log_trade_attempt` function body here.
    file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Timestamp", "Symbol", "Type", "Price", "SL", "TP", "Volume", "Status", "RetCode", "Comment", "Ticket", "ATR"])
            
            type_str = "ACTION"
            if signal == 1: type_str = "BUY"
            elif signal == -1: type_str = "SELL"
            elif signal == 99: type_str = "CLOSE_BUY"
            elif signal == -99: type_str = "CLOSE_SELL"
            
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                symbol, type_str, f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}",
                volume, status, retcode, comment, ticket, f"{atr_val:.4f}"
            ])
            print(f"📝 Logged to {LOG_FILE} (RetCode: {retcode})")
    except Exception as e:
        print(f"⚠️ Failed to write log: {e}")

def get_closed_candles(symbol, timeframe, n=200):
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
    
    atr = talib.ATR(hi, lo, cl, timeperiod=ATR_PERIOD)
    atr_ma = talib.SMA(atr, timeperiod=VOL_THRESHOLD_MA)
    curr_atr = atr[-1]
    
    if curr_atr <= atr_ma[-1]: return 0, curr_atr 

    engulf = talib.CDLENGULFING(op, hi, lo, cl)[-1]
    hammer = talib.CDLHAMMER(op, hi, lo, cl)[-1]
    star = talib.CDLSHOOTINGSTAR(op, hi, lo, cl)[-1]
    hanging = talib.CDLHANGINGMAN(op, hi, lo, cl)[-1]
    
    signal = 0
    if (engulf == 100) or (hammer == 100): signal = 1
    if (engulf == -100) or (star == -100) or (hanging == -100): signal = -1 if signal != 1 else 0
            
    return signal, curr_atr

def execute_trade(symbol, signal, atr, lot):
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
        "comment": "BaseBot_v1",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    status = "SUCCESS" if result.retcode == mt5.TRADE_RETCODE_DONE else "FAILED"
    
    log_trade_attempt(symbol, signal, price, sl, tp, lot, status, result.comment, result.order, atr, result.retcode)
    
    # --- DISCORD ALERT BLOCK ---
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        side = "BUY 🟢" if signal == 1 else "SELL 🔴"
        color = 0x2ecc71 if signal == 1 else 0xe74c3c
        msg = (f"**Symbol:** {symbol}\n"
               f"**Side:** {side}\n"
               f"**Price:** {price:.2f}\n"
               f"**SL:** {sl:.2f} | **TP:** {tp:.2f}\n"
               f"**Vol:** {lot} | **ATR:** {atr:.2f}")
        send_discord_alert(f"🚀 New Trade Executed", msg, color)
    else:
        send_discord_alert("❌ Trade Failed", f"Code: {result.retcode}\n{result.comment}", 0xe67e22)
    # ---------------------------

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
    s_code = 99 if order_type == mt5.ORDER_TYPE_BUY else -99
    status = "CLOSED" if result.retcode == mt5.TRADE_RETCODE_DONE else "CLOSE_FAIL"
    log_trade_attempt(position.symbol, s_code, price, 0, 0, position.volume, status, f"{reason}: {result.comment}", position.ticket, 0, result.retcode)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"⏳ Trade Closed ({reason}) | Ticket: {position.ticket}")
        # --- DISCORD EXIT ALERT ---
        send_discord_alert("🏁 Position Closed", f"**Ticket:** {position.ticket}\n**Reason:** {reason}\n**Close Price:** {price}", 0x3498db)

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
                if proposed > pos.sl: new_sl, update = proposed, True
            else:
                proposed = current_price + initial_risk
                if proposed < pos.sl: new_sl, update = proposed, True
            
            if update:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": pos.ticket,
                    "sl": new_sl,
                    "tp": pos.tp,
                    "magic": MAGIC_NUMBER
                }
                res = mt5.order_send(req)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"⛓️ Trailing Updated: {pos.ticket}")
                    # Optional: Uncomment if you want spam for every trail update
                    # send_discord_alert("⛓️ Trailing Stop Moved", f"Ticket: {pos.ticket}\nNew SL: {new_sl}", 0x95a5a6)

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
        print("🤖 BASE Bot Started (No Smart Filters)...")
        
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
                            df = get_closed_candles(SYMBOL, TIMEFRAME, n=100)
                            sig, atr = check_for_signal(df)
                            
                            if sig != 0:
                                print(f"🚀 Signal: {'BUY' if sig==1 else 'SELL'} | ATR: {atr:.2f}")
                                check_history_for_losses()
                                
                                if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                                    print("🛑 Max Losses Hit. Cooling Down.")
                                    send_discord_alert("❄️ Risk Cooldown Activated", f"Max consecutive losses ({consecutive_losses}) reached.\nPausing for {COOLDOWN_BARS} bars.", 0xe74c3c) # <--- ALERT
                                    cooldown_until = datetime.now() + timedelta(minutes=5 * COOLDOWN_BARS)
                                else:
                                    execute_trade(SYMBOL, sig, atr, LOT_SIZE)
                            else:
                                print(f"💤 No Signal. (ATR: {atr:.2f})") 
                        
                        last_processed_candle_time = latest_close_time
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                send_discord_alert("🛑 Bot Stopped", "Manual shutdown detected.", 0x95a5a6) # <--- ALERT
                mt5.shutdown()
                break
            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(5)