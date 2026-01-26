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
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL2")

# --- ⚙️ LIVE CONFIGURATION ---
SYMBOL = "XAUUSDm"       
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.01          
MAGIC_NUMBER = 555999    
DEVIATION = 20           
MT5_PATH = None         
MAX_POSITIONS = 20 

# --- CLI ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to terminal64.exe")
parser.add_argument("--magic", type=int, help="Magic Number")
parser.add_argument("--symbol", type=str, help="Symbol")
args = parser.parse_args()

if args.path: MT5_PATH = args.path
if args.magic: MAGIC_NUMBER = args.magic
if args.symbol: SYMBOL = args.symbol

# --- 🏆 STRATEGY SETTINGS (OPTIMIZED) ---
PARAMS = {
    # Entry Filters
    'rsi_min': 30,
    'rsi_max': 70,

    # Initial Risk
    'sl_mult': 2.0,      # Initial Stop Loss (2.0 ATR)
    'tp_mult': 1.5,      # Initial Fixed TP (1.5 ATR) - Placeholder until Runner logic takes over
    
    # ⚡ Infinity Runner Logic (RELAXED)
    # Changed from 0.2 to 1.5 to stop choking trades too early
    'be_trigger': 1.5,   # ATRs in profit to trigger Break-Even & Remove TP
    'step_mult': 0.3     # Increased from 0.2 to 0.3 to let winners run further
}

# ⚠️ REMOVED LOSING PATTERNS (Doji, Rickshaw, Belthold)
TARGET_PATTERNS = [
    'CDLHIGHWAVE', 
    'CDLENGULFING'
]

# ⛔ TIME FILTERS (KILL ZONES)
# Hours to strictly avoid based on your data analysis
FORBIDDEN_HOURS = [9, 16, 20] 

# ==========================================
# 📢 DISCORD ENGINE
# ==========================================
def send_discord_alert(title, message, color_type="INFO"):
    if not DISCORD_URL: return
    colors = {"BUY": 5763719, "SELL": 15548997, "INFO": 3447003, "ERROR": 15158332}
    try:
        requests.post(DISCORD_URL, json={
            "username": "Gold Sniper (Optimized)",
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
def execute_trade_robust(action, sl_dist, tp_dist, comment):
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    
    if tick is None or info is None:
        print("❌ Tick Data Unavailable")
        return

    # 1. Get Live Entry Price
    price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid

    # 2. Calculate Targets
    if action == mt5.ORDER_TYPE_BUY:
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist

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
    tp = round(tp, info.digits)

    # 5. Send Order
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    res = mt5.order_send(req)
    dir_str = "BUY" if action == 0 else "SELL"
    
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"⚡ {dir_str} OPEN @ {price} | SL: {sl} | TP: {tp}")
        msg = f"**{comment}**\nEntry: {price}\nSL: {sl}\nTP: {tp}"
        send_discord_alert(f"🚀 NEW {dir_str}", msg, dir_str)
        
        # 🟢 CRITICAL LOGGING FIX: Save pattern to local file
        try:
            with open("trade_audit_log.csv", "a") as f:
                f.write(f"{datetime.now()},{res.order},{dir_str},{price},{comment}\n")
        except: pass
        
    else:
        print(f"❌ Entry Failed: {res.comment} ({res.retcode})")

def modify_position(ticket, new_sl, new_tp=None):
    info = mt5.symbol_info(SYMBOL)
    sl_norm = round(new_sl, info.digits)
    
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": SYMBOL,
        "sl": sl_norm,
        "magic": MAGIC_NUMBER
    }
    
    if new_tp is not None:
        req["tp"] = round(new_tp, info.digits)
        
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
        
        # --- BRANCH A: FIXED TP PHASE (Waiting for BE Trigger) ---
        if pos.tp != 0.0:
            profit_dist = (current_price - entry) if is_buy else (entry - current_price)
            
            if profit_dist >= be_trigger_dist:
                print(f"🔓 BE Triggered on #{pos.ticket}. Removing TP & Securing Entry...")
                if modify_position(pos.ticket, entry, new_tp=0.0):
                    send_discord_alert("🚀 RUNNER MODE ACTIVATED", 
                                     f"Trade #{pos.ticket} is now Risk-Free.\nTarget Removed (Infinity Mode).", "INFO")

        # --- BRANCH B: INFINITY RUNNER PHASE (Trailing Steps) ---
        else:
            if is_buy:
                profit_dist = current_price - entry
                steps_climbed = math.floor(profit_dist / step_dist)
                
                if steps_climbed >= 1:
                    new_sl = entry + (steps_climbed * step_dist) - step_dist
                    if new_sl > (pos.sl + 0.01):
                        modify_position(pos.ticket, new_sl)
                        
            else: # SELL
                profit_dist = entry - current_price
                steps_climbed = math.floor(profit_dist / step_dist)
                
                if steps_climbed >= 1:
                    new_sl = entry - (steps_climbed * step_dist) + step_dist
                    if new_sl < (pos.sl - 0.01) or pos.sl == 0.0:
                        modify_position(pos.ticket, new_sl)

def scan_market(df):
    # 0. Time Filter (Kill Zones)
    current_hour = datetime.now().hour
    if current_hour in FORBIDDEN_HOURS:
        print(f"🛑 Kill Zone Active ({current_hour}:00). Scanning Paused...", end='\r')
        return

    # 1. Stacking Logic
    positions = mt5.positions_get(symbol=SYMBOL)
    my_positions = [p for p in positions if p.magic == MAGIC_NUMBER] if positions else []
    
    if len(my_positions) >= MAX_POSITIONS:
        print(f"🛡️ Max Positions ({len(my_positions)}/{MAX_POSITIONS}). Waiting...", end='\r')
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
        tp_dist = atr * PARAMS['tp_mult']
        
        print(f"\n🚀 SIGNAL FOUND: {detected_pat}")
        
        if signal == 1:
            execute_trade_robust(mt5.ORDER_TYPE_BUY, sl_dist, tp_dist, detected_pat)
        else:
            execute_trade_robust(mt5.ORDER_TYPE_SELL, sl_dist, tp_dist, detected_pat)
            
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
    print(f"✅ Gold Sniper OPTIMIZED: {SYMBOL} [M5]")
    print(f"♾️  Mode: Infinity Stacking (Max: {MAX_POSITIONS})")
    print(f"🛡️  Kill Zones: {FORBIDDEN_HOURS}")
    print(f"🎯  Targets: {TARGET_PATTERNS}")
    print("------------------------------------------")
    
    send_discord_alert("🤖 Bot Started", f"Symbol: {SYMBOL}\nOptimized Mode Active", "INFO")
    
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