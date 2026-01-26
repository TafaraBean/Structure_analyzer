import subprocess
import time
import os

# ==========================================
# ⚙️ MULTI-ACCOUNT CONFIGURATION
# ==========================================
# Update these paths to match your actual installation folders!
BOTS = [
    # {
    #     "name": "Standard Account (Single Trade)",
    #     "script": "gold_bot.py",
    #     "path": r"C:\Program Files\terminal64.exe",
    #     "magic": 555999,
    #     "symbol": "XAUUSDm" 
    # },
    {
        "name": "slow account",
        "script": "gold_bot_modified.py",
        "path": r"C:\Program Files\MetaTrader 5-3\terminal64.exe", # <--- UPDATE THIS PATH for 3rd Account
        "magic": 777111,
        "symbol": "XAUUSDm"
    }
]

# ==========================================
# 🚀 LAUNCH ENGINE
# ==========================================
if __name__ == "__main__":
    processes = []
    
    print(f"🚀 Launching {len(BOTS)} Sniper Instances...")
    print("------------------------------------------------")

    for bot in BOTS:
        print(f"🔸 Starting: {bot['name']}...")
        
        # Verify path exists first
        if not os.path.exists(bot['path']):
            print(f"❌ ERROR: Path not found: {bot['path']}")
            print("   -> Please check the 'path' in BOTS config inside launcher.py")
            continue

        cmd = [
            "cmd", "/k", "python", bot['script'],
            "--path", bot['path'],
            "--magic", str(bot['magic']),
            "--symbol", bot['symbol']
        ]
        
        # 'creationflags=subprocess.CREATE_NEW_CONSOLE' opens a new window for each bot
        # This way, if one crashes, the others stay alive.
        p = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        processes.append(p)
        
        # Sleep 2s to stagger logins and CPU usage
        time.sleep(2) 

    print("------------------------------------------------")
    print("✅ All bots launched in separate windows.")
    print("⚠️  Keep this script open to track PIDs, or close it to detach.")