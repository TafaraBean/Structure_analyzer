import MetaTrader5 as mt5
import time
import os
import sys
from datetime import datetime, timezone, timedelta
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
# SYMBOL / CONFIG
# ==========================
SYMBOL = "XAUUSDm"
MAGIC = 777002

CORE_LOT = 0.01
STACK_LOT = 0.01

CORE_EARLY_KILL_LOSS = -40
CORE_EARLY_BE_PROFIT = 40
CORE_MAX_AGE_CANDLES = 5
CORE_COOLDOWN_MINUTES = 5

STACK_TP_RAND = 20
STACK_MAX_LOSS_RAND = -25
STACK_TRAIL_KILL = 15

STACK_MIN_INTERVAL_SEC = 90
STACK_LOSS_PAUSE_SEC = 120

STARTUP_WARMUP_MINUTES = 1
ACCOUNT_FLOOR = 1800

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
        requests.post(DISCORD_URL, json={"content": msg}, timeout=3)
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

account = mt5.account_info()
if not account:
    raise RuntimeError("ACCOUNT INFO FAILED")

log(f"INCOME BOT STARTED | Balance: {account.balance}")
send_discord(f"🟢 Income bot started | Balance: {account.balance}")

# ==========================
# STATE
# ==========================
stack_peaks = {}
core_open_candle_time = None
core_cooldown_until = None
startup_until = now() + timedelta(minutes=STARTUP_WARMUP_MINUTES)

last_stack_time = None
stack_pause_until = None

startup_logged = False
core_cooldown_logged = False
stack_cooldown_logged = False
stack_spacing_logged = False
stack_blocked_logged = False

# ==========================
# POSITIONS
# ==========================
def positions():
    p = mt5.positions_get(symbol=SYMBOL)
    return [x for x in p if x.magic == MAGIC] if p else []

def core_position():
    for p in positions():
        if p.comment == "CORE":
            return p
    return None

def stack_positions():
    return [p for p in positions() if p.comment == "STACK"]

# ==========================
# MARKET DATA
# ==========================
def get_candles(count=5):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, count)
    if rates is None or len(rates) < 3:
        return None
    return rates

# ==========================
# EMERGENCY SHUTDOWN
# ==========================
def emergency_shutdown(reason):
    log("🛑 EMERGENCY SHUTDOWN")
    send_discord(f"🛑 {reason}")
    for p in positions():
        close_position(p)
    mt5.shutdown()
    sys.exit(0)

# ==========================
# ACCOUNT SAFETY
# ==========================
def check_account_floor():
    acc = mt5.account_info()
    if acc and (acc.balance <= ACCOUNT_FLOOR or acc.equity <= ACCOUNT_FLOOR):
        emergency_shutdown(
            f"DAILY LOSS LIMIT REACHED\nBalance: {acc.balance}\nEquity: {acc.equity}"
        )

# ==========================
# 🧠 CORE ENTRY (PATIENT / ORIGINAL LOGIC)
# ==========================
def maybe_open_core():
    global core_open_candle_time, startup_logged, core_cooldown_logged

    if now() < startup_until:
        if not startup_logged:
            log("⏳ STARTUP WARMUP – OBSERVING MARKET")
            startup_logged = True
        return
    else:
        startup_logged = False

    if core_position():
        return

    if core_cooldown_until and now() < core_cooldown_until:
        if not core_cooldown_logged:
            log("🧊 CORE COOLDOWN – WAITING")
            core_cooldown_logged = True
        return
    else:
        core_cooldown_logged = False

    rates = get_candles()
    if rates is None:
        return

    candle = rates[-2]

    open_price = candle['open']
    close_price = candle['close']
    high = candle['high']
    low = candle['low']

    body = close_price - open_price
    range_ = high - low

    # Ignore doji / indecision
    if abs(body) < 0.6:
        return

    # Require commitment (body ≥ 50% of candle)
    if abs(body) / range_ < 0.5:
        return

    direction = mt5.ORDER_TYPE_BUY if body > 0 else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction == mt5.ORDER_TYPE_BUY else tick.bid

    res = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": CORE_LOT,
        "type": direction,
        "price": price,
        "magic": MAGIC,
        "comment": "CORE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    })

    if res.retcode == mt5.TRADE_RETCODE_DONE:
        core_open_candle_time = candle['time']
        send_discord(f"🧱 CORE OPENED ({'BUY' if direction == 0 else 'SELL'})")

# ==========================
# CORE MANAGEMENT (UNCHANGED)
# ==========================
def manage_core():
    global core_cooldown_until

    core = core_position()
    if not core:
        return

    pnl = core.profit

    if pnl <= CORE_EARLY_KILL_LOSS:
        close_position(core)
        core_cooldown_until = now() + timedelta(minutes=CORE_COOLDOWN_MINUTES)
        send_discord("❌ CORE KILLED EARLY (−40)")
        return

    if pnl >= CORE_EARLY_BE_PROFIT and core.sl == 0:
        mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "position": core.ticket,
            "symbol": SYMBOL,
            "sl": core.price_open
        })
        send_discord("🔒 CORE AT BE (+40)")

# ==========================
# STACK ENTRY / MANAGEMENT (UNCHANGED)
# ==========================
def maybe_open_stack():
    global last_stack_time, stack_cooldown_logged, stack_spacing_logged, stack_blocked_logged

    core = core_position()
    if not core:
        return

    if core.sl != core.price_open:
        if not stack_blocked_logged:
            log("⏳ CORE NOT AT BE – STACKS BLOCKED")
            stack_blocked_logged = True
        return
    else:
        stack_blocked_logged = False

    if stack_positions():
        return

    if stack_pause_until and now() < stack_pause_until:
        if not stack_cooldown_logged:
            log("⏸ STACK COOLDOWN ACTIVE")
            stack_cooldown_logged = True
        return
    else:
        stack_cooldown_logged = False

    if last_stack_time and (now() - last_stack_time).total_seconds() < STACK_MIN_INTERVAL_SEC:
        if not stack_spacing_logged:
            log("⏳ STACK SPACING ACTIVE")
            stack_spacing_logged = True
        return
    else:
        stack_spacing_logged = False

    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if core.type == mt5.ORDER_TYPE_BUY else tick.bid

    res = mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": STACK_LOT,
        "type": core.type,
        "price": price,
        "magic": MAGIC,
        "comment": "STACK",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    })

    if res.retcode == mt5.TRADE_RETCODE_DONE:
        last_stack_time = now()
        stack_peaks[res.order] = 0
        send_discord("📦 STACK OPENED")

def manage_stacks():
    global stack_pause_until

    for s in stack_positions():
        pnl = s.profit
        peak = stack_peaks.get(s.ticket, pnl)
        stack_peaks[s.ticket] = max(peak, pnl)

        if peak >= 10 and pnl <= peak - STACK_TRAIL_KILL:
            close_position(s)
            stack_pause_until = now() + timedelta(seconds=STACK_LOSS_PAUSE_SEC)
            send_discord("⚠️ STACK TRAIL KILLED")
            continue

        if pnl >= STACK_TP_RAND:
            close_position(s)
            send_discord("✅ STACK TP")

        if pnl <= STACK_MAX_LOSS_RAND:
            close_position(s)
            stack_pause_until = now() + timedelta(seconds=STACK_LOSS_PAUSE_SEC)
            send_discord("❌ STACK STOPPED")

# ==========================
# CLOSE POSITION
# ==========================
def close_position(pos):
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.bid if pos.type == 0 else tick.ask

    mt5.order_send({
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": SYMBOL,
        "volume": pos.volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
        "price": price
    })

# ==========================
# MAIN LOOP
# ==========================
try:
    while True:
        check_account_floor()
        maybe_open_core()
        manage_core()
        manage_stacks()
        maybe_open_stack()
        time.sleep(1)
except KeyboardInterrupt:
    log("🟡 MANUAL SHUTDOWN – POSITIONS LEFT OPEN")
    mt5.shutdown()
