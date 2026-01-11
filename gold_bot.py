import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import talib
import time
import sys
import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# --- 🔐 SECRETS ---
load_dotenv()
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")

# --- ⚙️ LIVE CONFIGURATION ---
SYMBOL = "XAUUSDm"       
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.01          
MAGIC_NUMBER = 555999    
DEVIATION = 20           

# --- 🏆 STRATEGY SETTINGS ---
PARAMS = {
    'tp_mult': 1.5,
    'sl_mult': 2.0,      
    'trail_mult': 1.5,   
    'be_trigger': 0.30,  
    'rsi_min': 30,
    'rsi_max': 70
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

    # Colors: Green (BUY), Red (SELL), Blue (INFO/TRAIL), Orange (ERROR)
    colors = {
        "BUY": 5763719, 
        "SELL": 15548997, 
        "INFO": 3447003, 
        "ERROR": 15158332
    }
    
    payload = {
        "username": "Pattern Sniper",
        "embeds": [{
            "title": title,
            "description": message,
            "color": colors.get(color_type, 3447003),
            "timestamp": datetime.utcnow().isoformat()
        }]
    }
    
    try:
        requests.post(DISCORD_URL, json=payload)
    except Exception as e:
        print(f"⚠️ Discord Fail: {e}")

# ==========================================
# 🛠️ CORE FUNCTIONS
# ==========================================

def get_market_data():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 200)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def modify_position(ticket, new_sl, new_tp=None, reason="Trailing"):
    if new_tp is None:
        pos = mt5.positions_get(ticket=ticket)
        if pos: new_tp = pos[0].tp
        else: return

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": SYMBOL,
        "sl": new_sl,
        "tp": new_tp,
        "magic": MAGIC_NUMBER
    }
    res = mt5.order_send(request)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"⚠️ Modify Failed: {res.comment}")
    else:
        msg = f"Ticket: {ticket}\nNew SL: {new_sl:.2f}\nReason: {reason}"
        print(f"🔒 {reason} -> {new_sl:.2f}")
        send_discord_alert("🛡️ Stop Loss Updated", msg, "INFO")

def execute_trade(action, sl, tp, comment):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
    
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
        print(f"⚡ {dir_str} OPEN @ {price} | SL: {sl:.2f} | TP: {tp:.2f}")
        
        # DISCORD ALERT
        msg = f"**{comment}**\nPrice: {price}\nSL: {sl:.2f}\nTP: {tp:.2f}\nLot: {LOT_SIZE}"
        send_discord_alert(f"🚀 NEW {dir_str} TRADE", msg, dir_str)
    else:
        print(f"❌ Entry Failed: {res.comment}")
        send_discord_alert("❌ Execution Failed", f"{res.comment}", "ERROR")

# ==========================================
# 🎮 LOGIC MODULES
# ==========================================

def manage_positions(df):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return

    current_atr = df.iloc[-1]['ATR']
    tick = mt5.symbol_info_tick(SYMBOL)
    trail_dist = current_atr * PARAMS['trail_mult']
    
    for pos in positions:
        if pos.magic != MAGIC_NUMBER: continue
        
        is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
        current_price = tick.bid if is_buy else tick.ask
        entry_price = pos.price_open
        tp_price = pos.tp
        
        # 1. Calc Progress
        if tp_price != 0:
            total_dist = abs(tp_price - entry_price)
            curr_dist = abs(current_price - entry_price)
            in_profit = (current_price > entry_price) if is_buy else (current_price < entry_price)
            progress = (curr_dist / total_dist) if in_profit else 0
        else: progress = 0

        # 2. Logic
        be_sl = entry_price
        trail_sl = (current_price - trail_dist) if is_buy else (current_price + trail_dist)
        
        new_sl = pos.sl
        update_needed = False
        reason = ""
        
        if is_buy:
            # Breakeven Check
            if progress >= PARAMS['be_trigger'] and new_sl < be_sl:
                new_sl = be_sl
                update_needed = True
                reason = "Breakeven Lock (30%)"
            # Trailing Check
            elif trail_sl > new_sl:
                new_sl = trail_sl
                update_needed = True
                reason = "Dynamic Trail"
                
        else: # Sell
            # Breakeven Check
            if progress >= PARAMS['be_trigger'] and (new_sl > be_sl or new_sl == 0):
                new_sl = be_sl
                update_needed = True
                reason = "Breakeven Lock (30%)"
            # Trailing Check
            elif (trail_sl < new_sl) or (new_sl == 0):
                new_sl = trail_sl
                update_needed = True
                reason = "Dynamic Trail"

        # 3. Execute
        if update_needed:
            min_dist = tick.ask - tick.bid 
            if abs(current_price - new_sl) > min_dist:
                modify_position(pos.ticket, new_sl, reason=reason)

def scan_market(df):
    last_candle = df.iloc[-2]      
    filter_candle = df.iloc[-3]    
    
    rsi_val = filter_candle['RSI']
    if not (PARAMS['rsi_min'] <= rsi_val <= PARAMS['rsi_max']):
        print(f"⏳ Filter Wait: RSI {rsi_val:.1f}", end='\r')
        return

    positions = mt5.positions_get(symbol=SYMBOL)
    my_positions = [p for p in positions if p.magic == MAGIC_NUMBER] if positions else []
    if len(my_positions) > 0:
        print(f"🛡️ Managing Position ({len(my_positions)} active)...", end='\r')
        return

    op = df['open'].values; hi = df['high'].values
    lo = df['low'].values; cl = df['close'].values
    
    signal = 0; detected_pat = ""
    
    for pat in TARGET_PATTERNS:
        func = getattr(talib, pat)
        res = func(op, hi, lo, cl)
        score = res[-2] 
        
        if score == 100: signal = 1; detected_pat = pat; break
        elif score == -100: signal = -1; detected_pat = pat; break
            
    if signal != 0:
        atr = last_candle['ATR']
        close_price = last_candle['close']
        
        sl_dist = atr * PARAMS['sl_mult']
        tp_dist = sl_dist * (PARAMS['tp_mult'] / PARAMS['sl_mult']) 
        
        print(f"\n🚀 SIGNAL: {detected_pat} ({'Bull' if signal==1 else 'Bear'})")
        
        if signal == 1:
            sl = close_price - sl_dist
            tp = close_price + tp_dist
            execute_trade(mt5.ORDER_TYPE_BUY, sl, tp, detected_pat)
        else:
            sl = close_price + sl_dist
            tp = close_price - tp_dist
            execute_trade(mt5.ORDER_TYPE_SELL, sl, tp, detected_pat)
            
        print("💤 Signal cooldown (5m)...")
        time.sleep(300) 

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
if __name__ == "__main__":
    if not mt5.initialize(): sys.exit("MT5 Init Failed")
    
    print(f"✅ Bot Live: {SYMBOL} [M5]")
    print(f"   Discord: {'Active' if DISCORD_URL else 'Disabled'}")
    send_discord_alert("🤖 Bot Started", f"Symbol: {SYMBOL}\nStrategy: Pattern Sniper", "INFO")
    
    try:
        while True:
            df = get_market_data()
            if df is not None:
                manage_positions(df)
                scan_market(df)
            time.sleep(5) 
            
    except KeyboardInterrupt:
        mt5.shutdown()
        print("\n🛑 Bot Stopped.")