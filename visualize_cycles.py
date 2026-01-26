import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import periodogram
import pywt
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# --- Configuration ---
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS_TO_ANALYZE = 1000   
ZOOM_BARS = 1000          # Reduced to 80 for even better visibility
MAX_CYCLE_LENGTH = 20    
WAVELET_NAME = 'cmor'

def init_mt5():
    """Initialize connection to MetaTrader 5."""
    path = os.getenv("MT5_PATH")
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    params = {}
    if path: params["path"] = path

    if not mt5.initialize(**params):
        print(f"❌ MT5 Initialize failed, error code = {mt5.last_error()}")
        return False
    
    if login and password and server:
        if not mt5.login(login=int(login), password=password, server=server):
             print(f"⚠️ MT5 Login failed, error code = {mt5.last_error()}")
    
    print(f"✅ Connected to MT5: {mt5.terminal_info().name}")
    return True

def fetch_data(symbol, timeframe, n_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None:
        print(f"❌ Failed to get rates for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calculate_fft(close_prices):
    detrended = close_prices - np.mean(close_prices)
    freqs, power = periodogram(detrended, detrend='linear')
    periods = 1 / freqs[1:]
    power = power[1:]
    return periods, power

def get_dominant_cycle_sine(prices, periods, powers, max_period=60):
    valid_indices = (periods <= max_period) & (periods >= 6)
    
    if not np.any(valid_indices):
        return np.full_like(prices, np.mean(prices)), 0
        
    filtered_powers = powers[valid_indices]
    filtered_periods = periods[valid_indices]
    
    max_idx = np.argmax(filtered_powers)
    dom_period = filtered_periods[max_idx]
    
    # Reconstruct Sine Wave
    t = np.arange(len(prices))
    omega = 2 * np.pi / dom_period
    
    sin_term = np.sin(omega * t)
    cos_term = np.cos(omega * t)
    
    data_detrend = prices - np.mean(prices)
    
    coeff_sin = np.dot(data_detrend, sin_term)
    coeff_cos = np.dot(data_detrend, cos_term)
    
    phase = np.arctan2(coeff_cos, coeff_sin)
    amplitude = np.sqrt(coeff_sin**2 + coeff_cos**2) / (len(t)/2)
    
    reconstructed = amplitude * np.sin(omega * t + phase) + np.mean(prices)
    
    return reconstructed, dom_period

def calculate_cwt(close_prices):
    data = close_prices - np.mean(close_prices)
    scales = np.arange(2, 100)
    coef, freqs = pywt.cwt(data, scales, WAVELET_NAME)
    power = (abs(coef)) ** 2
    periods = 1 / freqs
    return periods, power, scales

def main():
    if not init_mt5():
        sys.exit()
        
    print(f"📥 Fetching {BARS_TO_ANALYZE} bars (Visualizing last {ZOOM_BARS})...")
    df = fetch_data(SYMBOL, TIMEFRAME, BARS_TO_ANALYZE)
    if df is None: return

    close = df['close'].values
    
    # 1. Math Engines
    periods, fft_power = calculate_fft(close)
    sine_wave, dom_period = get_dominant_cycle_sine(close, periods, fft_power, max_period=MAX_CYCLE_LENGTH)
    cwt_periods, cwt_power, cwt_scales = calculate_cwt(close)
    
    print(f"🌊 Dominant Cycle: {dom_period:.2f} candles ({dom_period/12:.2f} hours)")

    # 2. Slicing for Zoom
    df_zoom = df.tail(ZOOM_BARS)
    sine_zoom = sine_wave[-ZOOM_BARS:]
    
    # --- FIX: Re-center the wave to the CURRENT price ---
    # We take the wave, strip its old average, and add the NEW average
    sine_zoom = sine_zoom - np.mean(sine_zoom) + np.mean(df_zoom['close'])
    
    cwt_power_zoom = cwt_power[:, -ZOOM_BARS:]

    # 3. Visualization
    print("🎨 Generating Zoomed Candlestick Chart...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 1], hspace=0.15)
    
    # --- PANEL 1: Crisp Candlesticks ---
    ax1 = fig.add_subplot(gs[0])
    
    # --- WIDTH CORRECTION ---
    # 5 minutes in "Matplotlib Days" = 5 / (24*60) = 0.00347
    # We set width to 0.0025 to leave a small gap between candles
    width_body = 0.0025 
    width_wick = 0.0005 
    
    up = df_zoom[df_zoom.close >= df_zoom.open]
    down = df_zoom[df_zoom.close < df_zoom.open]
    
    col_up = '#089981'   # Green
    col_down = '#f23645' # Red
    
    # Plot Wicks
    ax1.bar(up.index, up.high - up.low, width_wick, bottom=up.low, color=col_up, alpha=0.9)
    ax1.bar(down.index, down.high - down.low, width_wick, bottom=down.low, color=col_down, alpha=0.9)
    
    # Plot Bodies
    ax1.bar(up.index, up.close - up.open, width_body, bottom=up.open, color=col_up, alpha=1.0)
    ax1.bar(down.index, down.open - down.close, width_body, bottom=down.close, color=col_down, alpha=1.0)
    
    # Plot Cycle Overlay
    ax1.plot(df_zoom.index, sine_zoom, color='cyan', alpha=0.8, linewidth=2, label=f'Cycle ({dom_period:.1f} bars)')
    
    ax1.set_title(f"{SYMBOL} Zoomed Analysis (Last {ZOOM_BARS} Bars)", fontsize=16, fontweight='bold', color='white')
    ax1.legend(loc='upper left')
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.15)
    
    # Formatting X-axis dates properly
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # --- PANEL 2: Heatmap ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Note: pcolormesh needs numerical x-axis for proper alignment with sharex
    # We use mdates.date2num to ensure alignment
    x_nums = mdates.date2num(df_zoom.index)
    c = ax2.pcolormesh(x_nums, cwt_periods, cwt_power_zoom, cmap='inferno', shading='auto')
    
    ax2.set_ylabel("Period")
    ax2.set_ylim(0, 80) 
    ax2.axhline(y=dom_period, color='cyan', linestyle='--', alpha=0.6)
    
    # --- PANEL 3: Spectrum ---
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(periods, fft_power, color='magenta')
    ax3.set_ylabel("Power")
    ax3.set_xlabel("Cycle Length (Bars)")
    ax3.set_xlim(5, 80) 
    ax3.grid(alpha=0.2)
    ax3.fill_between(periods, fft_power, color='magenta', alpha=0.3)
    ax3.axvline(x=dom_period, color='cyan', linestyle='--', label="Dominant")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()