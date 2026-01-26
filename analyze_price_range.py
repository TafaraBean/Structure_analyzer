import MetaTrader5 as mt5
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize MT5
path = os.getenv("MT5_PATH")
login = os.getenv("MT5_LOGIN")
password = os.getenv("MT5_PASSWORD")
server = os.getenv("MT5_SERVER")
params = {}
if path: params["path"] = path

mt5.initialize(**params)
if login and password and server:
    mt5.login(login=int(login), password=password, server=server)

# Fetch data
rates = mt5.copy_rates_from_pos('EURUSDm', mt5.TIMEFRAME_M15, 0, 3000)
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Analyze price range
print("="*60)
print("  EURUSD PRICE RANGE ANALYSIS")
print("="*60)
print(f"\nData period: {df['time'].min()} to {df['time'].max()}")
print(f"Total bars: {len(df)}")
print(f"\nPrice Statistics:")
print(f"  Highest: {df['high'].max():.5f}")
print(f"  Lowest: {df['low'].min():.5f}")
print(f"  Range: {(df['high'].max() - df['low'].min()):.5f} ({((df['high'].max() - df['low'].min()) / df['close'].mean() * 100):.2f}%)")
print(f"  Mean: {df['close'].mean():.5f}")
print(f"  Std Dev: {df['close'].std():.5f}")

# Check if trending or ranging
price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
print(f"\nOverall price change: {price_change:.2f}%")

if abs(price_change) < 1:
    print("  → Market is RANGING (tight consolidation)")
else:
    print(f"  → Market is TRENDING")

# Quartiles
quartiles = df['close'].quantile([0.25, 0.5, 0.75])
print(f"\nPrice Quartiles:")
print(f"  25%: {quartiles[0.25]:.5f}")
print(f"  50%: {quartiles[0.50]:.5f}")
print(f"  75%: {quartiles[0.75]:.5f}")

mt5.shutdown()
