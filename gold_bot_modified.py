import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import talib
import time
import sys
import math
import requests
import json
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# --- 🔐 SECRETS ---
load_dotenv()
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL3")

# --- ⚙️ LIVE CONFIGURATION ---
SYMBOL = "XAUUSDm"       
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.01          
MAGIC_NUMBER = 555999    
DEVIATION = 20           
MT5_PATH = None         

# --- CLI ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to terminal64.exe")
parser.add_argument("--magic", type=int, help="Magic Number")
parser.add_argument("--symbol", type=str, help="Symbol")
args = parser.parse_args()

if args.path: MT5_PATH = args.path
if args.magic: MAGIC_NUMBER = args.magic
if args.symbol: SYMBOL = args.symbol

# --- 🏆 STRATEGY SETTINGS ---
PARAMS = {
    # Entry Filters
    'rsi_min': 30,
    'rsi_max': 70,

    # Initial Risk
    'sl_mult': 2.0,      # Initial Stop Loss (2.0 ATR)
    
    # ⚡ Infinity Runner Logic
    # No TP. Trade runs until it hits the Trailing Stop.
    'be_trigger': 1.5,   # ATRs in profit to trigger "Safety Mode" (Move SL to Entry)
    'step_mult': 0.3     # Trailing Step Size
}

# ⚠️ WINNING PATTERNS ONLY
TARGET_PATTERNS = [
    'CDLHIGHWAVE', 
    'CDLENGULFING'
]

# ⛔ TIME FILTERS (KILL ZONES)
FORBIDDEN_HOURS = [9, 16, 20] 

# ==========================================
# 📢 DISCORD ENGINE
# ==========================================
def send_discord_alert(title, message, color_type="INFO"):
    if not DISCORD_URL: return
    colors = {"BUY": 5763719, "SELL": 15548997, "INFO": 3447003, "ERROR": 15158332}
    try:
        requests.post(DISCORD_URL, json={
            "username": "Gold Sniper (No Limits)",
            "embeds": [{
                "title": title, "description": message,
                "color": colors.get(color_type, 3447003),
                "timestamp": datetime.utcnow().isoformat()
            }]
        })
    except: pass

# ==========================================
# 🛠️ EXECUTION ENGINE
# ==========================================
def execute_trade_robust(action, sl_dist, comment):
    """
    Opens a trade with NO Take Profit (TP=0.0).
    """
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    
    if tick is None or info is None:
        print("❌ Tick Data Unavailable")
        return

    # 1. Get Live Entry Price
    price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid

    # 2. Calculate Stop Loss Only
    if action == mt5.ORDER_TYPE_BUY:
        sl = price - sl_dist
    else:
        sl = price + sl_dist

    # 3. Validate Minimum Stops
    min_dist = info.trade_stops_level * info.point
    if abs(price - sl) < min_dist:
        print(f"⚠️ SL too close (<{min_dist}). Adjusting.")
        correction = min_dist + (10 * info.point) 
        if action == mt5.ORDER_TYPE_BUY: sl = price - correction
        else: sl = price + correction

    # 4. Normalize
    price = round(price, info.digits)
    sl = round(sl, info.digits)

    # 5. Send Order (TP = 0.0)
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": 0.0, # <--- NO TARGET
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(req)
    dir_str = "BUY" if action == 0 else "SELL"
    
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"⚡ {dir_str} OPEN @ {price} | SL: {sl} | TP: 🔓 (Infinity)")
        msg = f"**{comment}**\nEntry: {price}\nSL: {sl}\nTP: Infinity"
        send_discord_alert(f"🚀 NEW {dir_str}", msg, dir_str)
        
        try:
            with open("trade_audit_log.csv", "a") as f:
                f.write(f"{datetime.now()},{res.order},{dir_str},{price},{comment}\n")
        except: pass
        
    else:
        print(f"❌ Entry Failed: {res.comment} ({res.retcode})")

def modify_position(ticket, new_sl):
    info = mt5.symbol_info(SYMBOL)
    sl_norm = round(new_sl, info.digits)
    
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": SYMBOL,
        "sl": sl_norm,
        "magic": MAGIC_NUMBER
    }
    
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"🛡️ Position Modified: SL->{sl_norm}")
        return True
    return False

# ==========================================
# 📊 MARKET ANALYZER
# ==========================================
def get_market_data():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 200)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def manage_positions(df):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return

    tick = mt5.symbol_info_tick(SYMBOL)
    current_atr = df.iloc[-1]['ATR']
    
    be_trigger_dist = current_atr * PARAMS['be_trigger']
    step_dist = current_atr * PARAMS['step_mult']

    for pos in positions:
        if pos.magic != MAGIC_NUMBER: continue
        
        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        current_price = tick.bid if is_buy else tick.ask
        entry = pos.price_open
        
        # Calculate current profit distance
        profit_dist = (current_price - entry) if is_buy else (entry - current_price)

        # STATE DETECTION based on Stop Loss position
        # If SL is worse than entry, we are in "Risk Phase"
        is_in_risk_phase = (pos.sl < entry) if is_buy else (pos.sl > entry)
        
        if is_in_risk_phase:
            # --- PHASE 1: WAITING FOR BREAK-EVEN ---
            if profit_dist >= be_trigger_dist:
                print(f"🔓 BE Trigger Hit (+{profit_dist:.2f}). Securing Trade...")
                # Move SL to Entry (or slightly better to cover swap/comm)
                modify_position(pos.ticket, entry)
                send_discord_alert("🛡️ SECURED", f"Trade #{pos.ticket} SL Moved to Break-Even.", "INFO")
                
        else:
            # --- PHASE 2: INFINITY TRAILING ---
            # SL is already at or better than Entry. Now we trail.
            
            if is_buy:
                steps_climbed = math.floor(profit_dist / step_dist)
                if steps_climbed >= 1:
                    # Trailing Formula
                    new_sl = entry + (steps_climbed * step_dist) - step_dist
                    # Only move SL UP
                    if new_sl > (pos.sl + 0.01):
                        modify_position(pos.ticket, new_sl)
                        
            else: # SELL
                steps_climbed = math.floor(profit_dist / step_dist)
                if steps_climbed >= 1:
                    new_sl = entry - (steps_climbed * step_dist) + step_dist
                    # Only move SL DOWN
                    if new_sl < (pos.sl - 0.01) or pos.sl == 0.0:
                        modify_position(pos.ticket, new_sl)

def scan_market(df):
    # 0. Time Filter (SERVER TIME FIX)
    last_candle_time = df.iloc[-1]['time']
    current_server_hour = last_candle_time.hour
    
    if current_server_hour in FORBIDDEN_HOURS:
        print(f"🛑 Kill Zone Active (Server: {current_server_hour}:00). Paused...", end='\r')
        return

    # 1. Single Trade Logic (NON-STACKING)
    positions = mt5.positions_get(symbol=SYMBOL)
    my_positions = [p for p in positions if p.magic == MAGIC_NUMBER] if positions else []
    
    if len(my_positions) > 0:
        print(f"🛡️ Trade Active (Monitoring #{my_positions[0].ticket})...", end='\r')
        return

    # 2. RSI Filter
    filter_candle = df.iloc[-2]
    if not (PARAMS['rsi_min'] <= filter_candle['RSI'] <= PARAMS['rsi_max']):
        print(f"⏳ Filter Wait: RSI {filter_candle['RSI']:.1f}   ", end='\r')
        return

    # 3. Pattern Recognition
    op = df['open'].values; hi = df['high'].values
    lo = df['low'].values; cl = df['close'].values
    
    signal = 0; detected_pat = ""
    for pat in TARGET_PATTERNS:
        func = getattr(talib, pat)
        score = func(op, hi, lo, cl)[-2]
        if score == 100: signal = 1; detected_pat = pat; break
        elif score == -100: signal = -1; detected_pat = pat; break
            
    # 4. Execution
    if signal != 0:
        atr = df.iloc[-2]['ATR']
        sl_dist = atr * PARAMS['sl_mult']
        
        print(f"\n🚀 SIGNAL FOUND: {detected_pat}")
        
        if signal == 1:
            execute_trade_robust(mt5.ORDER_TYPE_BUY, sl_dist, detected_pat)
        else:
            execute_trade_robust(mt5.ORDER_TYPE_SELL, sl_dist, detected_pat)
            
        time.sleep(300)

# ==========================================
# 🚀 MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    init_params = {}
    if MT5_PATH:
        init_params["path"] = MT5_PATH

    if not mt5.initialize(**init_params): 
        print(f"❌ MT5 Init Failed (Path: {MT5_PATH})")
        print(f"Error: {mt5.last_error()}")
        sys.exit()
    
    print("------------------------------------------")
    print(f"✅ Gold Sniper NO-LIMIT: {SYMBOL} [M5]")
    print(f"♾️  Mode: Single Shot | TP: Infinity")
    print(f"🛡️  Kill Zones (Server): {FORBIDDEN_HOURS}")
    print(f"🎯  Targets: {TARGET_PATTERNS}")
    print("------------------------------------------")
    
    send_discord_alert("🤖 Bot Started", f"Symbol: {SYMBOL}\nMode: No-Limit Runner", "INFO")
    
    last_heartbeat = time.time()
    
    try:
        while True:
            if time.time() - last_heartbeat >= 300: 
                print(f"💓 [{datetime.now().strftime('%H:%M:%S')}] Scan Active...")
                last_heartbeat = time.time()

            try:
                df = get_market_data()
                if df is not None:
                    manage_positions(df)
                    scan_market(df)
            except Exception as e:
                print(f"⚠️ Error in loop: {e}")
                
            time.sleep(2)
            
    except KeyboardInterrupt:
        mt5.shutdown()
        print("\n🛑 Stopped.")