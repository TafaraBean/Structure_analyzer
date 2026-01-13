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
from datetime import datetime
from dotenv import load_dotenv

# --- 🔐 SECRETS ---
# Create a .env file with: DISCORD_WEBHOOK_URL=your_url_here
load_dotenv()
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- ⚙️ LIVE CONFIGURATION ---
SYMBOL = "XAUUSDm"       # Ensure this matches your broker's symbol exactly
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.01          
MAGIC_NUMBER = 555999    
DEVIATION = 20           

# --- 🏆 STRATEGY SETTINGS ---
# All distances are calculated using ATR
PARAMS = {
    # Entry Filters
    'rsi_min': 30,
    'rsi_max': 70,

    # Initial Risk
    'sl_mult': 2.0,      # Initial Stop Loss (2.0 ATR)
    'tp_mult': 1.5,      # Initial Fixed TP (1.5 ATR) - Only active until BE hit
    
    # ⚡ Infinity Runner Logic
    'be_trigger': 1.0,   # ATRs in profit to trigger Break-Even & Remove TP
    'step_mult': 0.2     # Size of the trailing step (0.2 ATR)
}

TARGET_PATTERNS = [
    'CDLLONGLEGGEDDOJI', 'CDLRICKSHAWMAN', 'CDLHIGHWAVE', 
    'CDLENGULFING', 'CDLBELTHOLD'
]

# ==========================================
# 📢 DISCORD ENGINE
# ==========================================
def send_discord_alert(title, message, color_type="INFO"):
    if not DISCORD_URL: return
    colors = {"BUY": 5763719, "SELL": 15548997, "INFO": 3447003, "ERROR": 15158332}
    try:
        requests.post(DISCORD_URL, json={
            "username": "Gold Sniper",
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
    """
    Opens a trade with initial fixed stops calculated from LIVE price.
    """
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

    # 3. Validate Minimum Stops (Broker Constraint)
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
    else:
        print(f"❌ Entry Failed: {res.comment} ({res.retcode})")

def modify_position(ticket, new_sl, new_tp=None):
    """
    Updates SL and optionally TP. Pass new_tp=0.0 to remove TP.
    """
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
    """
    The Core Engine: Handles Break-Even Trigger and Infinity Trailing.
    """
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return

    tick = mt5.symbol_info_tick(SYMBOL)
    current_atr = df.iloc[-1]['ATR']
    
    # Calculate Distances based on current Volatility
    be_trigger_dist = current_atr * PARAMS['be_trigger']
    step_dist = current_atr * PARAMS['step_mult']

    for pos in positions:
        if pos.magic != MAGIC_NUMBER: continue
        
        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        current_price = tick.bid if is_buy else tick.ask
        entry = pos.price_open
        
        # --- BRANCH A: FIXED TP PHASE (Waiting for BE Trigger) ---
        if pos.tp != 0.0:
            # Check how far we are in profit
            profit_dist = (current_price - entry) if is_buy else (entry - current_price)
            
            if profit_dist >= be_trigger_dist:
                print(f"🔓 BE Triggered on #{pos.ticket}. Removing TP & Securing Entry...")
                
                # Move SL to Entry, Remove TP (Set to 0.0)
                if modify_position(pos.ticket, entry, new_tp=0.0):
                    send_discord_alert("🚀 RUNNER MODE ACTIVATED", 
                                     f"Trade #{pos.ticket} is now Risk-Free.\nTarget Removed (Infinity Mode).", "INFO")

        # --- BRANCH B: INFINITY RUNNER PHASE (Trailing Steps) ---
        else:
            if is_buy:
                profit_dist = current_price - entry
                # Calculate Step Level: e.g., if profit is 3.2 ATRs, we are at Step 3
                steps_climbed = math.floor(profit_dist / step_dist)
                
                if steps_climbed >= 1:
                    # SL trails 1 step behind current step
                    # Formula: Entry + (Steps * StepSize) - StepSize
                    new_sl = entry + (steps_climbed * step_dist) - step_dist
                    
                    # Only move SL up
                    if new_sl > (pos.sl + 0.01):
                        modify_position(pos.ticket, new_sl)
                        
            else: # SELL
                profit_dist = entry - current_price
                steps_climbed = math.floor(profit_dist / step_dist)
                
                if steps_climbed >= 1:
                    # Formula: Entry - (Steps * StepSize) + StepSize
                    new_sl = entry - (steps_climbed * step_dist) + step_dist
                    
                    # Only move SL down
                    if new_sl < (pos.sl - 0.01) or pos.sl == 0.0:
                        modify_position(pos.ticket, new_sl)

def scan_market(df):
    # 1. Single Trade Rule
    positions = mt5.positions_get(symbol=SYMBOL)
    my_positions = [p for p in positions if p.magic == MAGIC_NUMBER] if positions else []
    if len(my_positions) > 0:
        print(f"🛡️ Active Position (Monitoring)...", end='\r')
        return

    # 2. RSI Filter (Index -2: Closed Candle)
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
        score = func(op, hi, lo, cl)[-2] # Closed candle
        if score == 100: signal = 1; detected_pat = pat; break
        elif score == -100: signal = -1; detected_pat = pat; break
            
    # 4. Execution
    if signal != 0:
        atr = df.iloc[-2]['ATR']
        
        # Calculate Initial Distances (Price Delta)
        sl_dist = atr * PARAMS['sl_mult']
        tp_dist = atr * PARAMS['tp_mult']
        
        print(f"\n🚀 SIGNAL FOUND: {detected_pat}")
        
        if signal == 1:
            execute_trade_robust(mt5.ORDER_TYPE_BUY, sl_dist, tp_dist, detected_pat)
        else:
            execute_trade_robust(mt5.ORDER_TYPE_SELL, sl_dist, tp_dist, detected_pat)
            
        time.sleep(300) # Wait 5 mins to prevent double entry on same candle

# ==========================================
# 🚀 MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize(): sys.exit("MT5 Init Failed")
    
    print("------------------------------------------")
    print(f"✅ Gold Sniper Live: {SYMBOL} [M5]")
    print(f"♾️  Mode: Infinity Runner (Step: {PARAMS['step_mult']} ATR)")
    print("------------------------------------------")
    
    send_discord_alert("🤖 Bot Started", f"Symbol: {SYMBOL}\nMode: Infinity Runner", "INFO")
    
    last_heartbeat = time.time()
    
    try:
        while True:
            # Heartbeat (Every 5 mins)
            if time.time() - last_heartbeat >= 300: 
                print(f"💓 [{datetime.now().strftime('%H:%M:%S')}] Scan Active...")
                last_heartbeat = time.time()

            # Logic
            try:
                df = get_market_data()
                if df is not None:
                    manage_positions(df) # Check stops/trails first
                    scan_market(df)      # Then look for new entries
            except Exception as e:
                print(f"⚠️ Error in loop: {e}")
                
            time.sleep(2) # Fast poll for trailing accuracy
            
    except KeyboardInterrupt:
        mt5.shutdown()
        print("\n🛑 Stopped.")