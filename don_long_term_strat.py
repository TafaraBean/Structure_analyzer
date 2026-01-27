import MetaTrader5 as mt5
import time
import os
import math
from datetime import datetime, timezone
from dotenv import load_dotenv

# ==========================
# ENV / CONNECTION
# ==========================
load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD")
MT5_SERVER = os.getenv("MT5_SERVER")
DISCORD_URL = os.getenv("DISCORD_WEBHOOK_URL")

MT5_PATH = r"C:\Program Files\MetaTrader 5-2\terminal64.exe"

# ==========================
# SYMBOL / RISK CONFIG
# ==========================
SYMBOL = "XAUUSDm"
LOT_SIZE = 0.01
MAGIC = 777001

CORE_MAX_LOSS_RAND = -100
CORE_BE_PROFIT_RAND = 100

STACK_TP_RAND = 20
STACK_MAX_LOSS_RAND = -25
MAX_STACKS = 1
STACK_MAX_BARS = 5

HEARTBEAT_SEC = 5

# ==========================
# HELPERS
# ==========================
def now():
    return datetime.now(timezone.utc)

def log(msg):
    print(f"[{now().strftime('%H:%M:%S')}] {msg}")

def send_discord(msg):
    if not DISCORD_URL:
        return
    import requests
    try:
        requests.post(DISCORD_URL, json={
            "username": "CORE BOT",
            "content": msg
        }, timeout=3)
    except:
        pass

# ==========================
# MT5 INIT
# ==========================
if not mt5.initialize(
    path=MT5_PATH,
    login=MT5_LOGIN,
    password=MT5_PASSWORD,
    server=MT5_SERVER
):
    raise RuntimeError("MT5 INIT FAILED")

mt5.symbol_select(SYMBOL, True)

log("CORE BOT STARTED")
send_discord("🟢 Core bot started and connected")

# ==========================
# STATE
# ==========================
last_bar_time = None
loss_streak = 0

# ==========================
# POSITION QUERIES
# ==========================
def positions():
    pos = mt5.positions_get(symbol=SYMBOL)
    return [p for p in pos if p.magic == MAGIC] if pos else []

def core_position():
    for p in positions():
        if p.comment == "CORE":
            return p
    return None

def stack_positions():
    return [p for p in positions() if p.comment == "STACK"]

# ==========================
# PNL IN RAND
# ==========================
def pnl_rand(pos):
    return pos.profit

# ==========================
# MARKET DATA (NO LOOKAHEAD)
# ==========================
def get_last_closed_bar():
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 3)
    if rates is None or len(rates) < 2:
        return None
    return rates[-2]

# ==========================
# CORE LOGIC
# ==========================
def maybe_open_core(bar):
    if core_position():
        return

    if stack_positions():
        return

    # simple directional logic (neutral but safe)
    body = bar['close'] - bar['open']

    if abs(body) < 0.5:
        return

    direction = mt5.ORDER_TYPE_BUY if body > 0 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(SYMBOL).ask if direction == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": direction,
        "price": price,
        "magic": MAGIC,
        "comment": "CORE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    res = mt5.order_send(request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        send_discord(f"🧱 CORE OPENED {'BUY' if direction==0 else 'SELL'}")
        log("CORE OPENED")

# ==========================
# STACK LOGIC
# ==========================
def maybe_open_stack(bar):
    core = core_position()
    if not core:
        return

    if len(stack_positions()) >= MAX_STACKS:
        return

    if loss_streak >= 2:
        return

    direction = core.type
    price = mt5.symbol_info_tick(SYMBOL).ask if direction == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": direction,
        "price": price,
        "magic": MAGIC,
        "comment": "STACK",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    res = mt5.order_send(request)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        log("STACK OPENED")

# ==========================
# MANAGEMENT
# ==========================
def manage_positions():
    global loss_streak

    core = core_position()
    stacks = stack_positions()

    # CORE protection
    if core:
        pnl = pnl_rand(core)

        if pnl <= CORE_MAX_LOSS_RAND:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "position": core.ticket,
                "symbol": SYMBOL,
                "volume": core.volume,
                "type": mt5.ORDER_TYPE_SELL if core.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(SYMBOL).bid if core.type == 0 else mt5.symbol_info_tick(SYMBOL).ask,
            })
            send_discord("❌ CORE KILLED (invalid thesis)")
            return

        if pnl >= CORE_BE_PROFIT_RAND and core.sl == 0:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": core.ticket,
                "symbol": SYMBOL,
                "sl": core.price_open,
            })
            send_discord("🔒 CORE AT BREAKEVEN")

    # STACK exits
    for s in stacks:
        pnl = pnl_rand(s)

        if pnl >= STACK_TP_RAND:
            close_position(s)
            loss_streak = 0
            continue

        if pnl <= STACK_MAX_LOSS_RAND:
            close_position(s)
            loss_streak += 1

# ==========================
# CLOSE POSITION
# ==========================
def close_position(pos):
    price = mt5.symbol_info_tick(SYMBOL).bid if pos.type == 0 else mt5.symbol_info_tick(SYMBOL).ask
    mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": SYMBOL,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "price": price,
    })

# ==========================
# MAIN LOOP
# ==========================
last_heartbeat = time.time()

while True:
    bar = get_last_closed_bar()
    if bar and bar['time'] != last_bar_time:
        last_bar_time = bar['time']
        manage_positions()
        maybe_open_core(bar)
        maybe_open_stack(bar)

    if time.time() - last_heartbeat >= HEARTBEAT_SEC:
        log("heartbeat alive")
        last_heartbeat = time.time()

    time.sleep(1)
