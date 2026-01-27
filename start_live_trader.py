import subprocess
import os
import sys

def start_live_trader():
    """Launch live trader in a new command prompt window."""
    print("="*60)
    print("  LIVE ORDER FLOW TRADING BOT LAUNCHER")
    print("="*60)
    print()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the live trader script
    trader_script = os.path.join(script_dir, "live_order_flow_trader.py")
    
    # Check if the trader script exists
    if not os.path.exists(trader_script):
        print(f"❌ Error: live_order_flow_trader.py not found!")
        print(f"   Expected at: {trader_script}")
        input("\nPress Enter to exit...")
        return
    
    # Check if model exists
    model_path = os.path.join(script_dir, "model_order_flow.keras")
    if not os.path.exists(model_path):
        print(f"⚠️  Warning: model_order_flow.keras not found!")
        print(f"   Expected at: {model_path}")
        print(f"   Make sure you've trained the model first.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print(f"✅ Found trader script: {trader_script}")
    print(f"✅ Launching in new CMD window...")
    print()
    print("📋 Instructions:")
    print("   - A new window will open with the live trader")
    print("   - Press Ctrl+C in that window to stop trading")
    print("   - The bot will auto-close positions on exit")
    print()
    
    # Launch in new CMD window
    # Use 'start' command to open new window, 'cmd /k' to keep window open
    cmd = f'start "Order Flow Live Trader" cmd /k "cd /d {script_dir} && python live_order_flow_trader.py"'
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("✅ Live trader launched successfully!")
        print()
        print("💡 Tip: Check the new window for trading activity")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching trader: {e}")
        input("\nPress Enter to exit...")
        return
    
    input("Press Enter to close this launcher...")


if __name__ == "__main__":
    start_live_trader()
