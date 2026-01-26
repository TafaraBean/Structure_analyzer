import subprocess
import time
import sys
import ctypes
import datetime
import os

# ================= CONFIGURATION =================
ADAPTER_NAME = "Ethernet" 
CHECK_HOST = "8.8.8.8"
CHECK_INTERVAL = 5 
# =================================================

def log_event(message):
    """Logs to console and file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    try:
        # Save log in the same folder as the script
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wifi_log.txt")
        with open(log_path, "a") as f:
            f.write(log_entry + "\n")
    except:
        pass # Ignore file errors to keep script running

def is_admin():
    """Checks for admin rights"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def check_connection():
    """Pings Google DNS. Returns True if connected."""
    try:
        # creationflags=0x08000000 hides the mini-popup of the PING command
        subprocess.run(
            ["ping", "-n", "1", CHECK_HOST], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            creationflags=0x08000000 
        )
        return True
    except:
        return False

def reset_adapter():
    log_event("!!! Connection lost. Resetting adapter...")
    
    # We use shell=True here so you can potentially see the output if needed
    subprocess.run(f'netsh interface set interface "{ADAPTER_NAME}" admin=disable', shell=True)
    time.sleep(3)
    subprocess.run(f'netsh interface set interface "{ADAPTER_NAME}" admin=enable', shell=True)
    
    time.sleep(8) # Wait for Windows to negotiate
    
    if check_connection():
        log_event("+++ Connection restored successfully.")
    else:
        log_event("--- Reset failed. Internet still down.")

def main():
    # --- SELF-ELEVATION BLOCK ---
    # If not admin, relaunch the script with "runas" (Admin) privileges
    if not is_admin():
        print("Requesting Administrator privileges...")
        # This command forces a new CMD window to open as Admin
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit() # Exit the non-admin instance
    # -----------------------------

    # If we are here, we are Admin. Set title of the CMD window
    os.system(f"title Auto-Wifi Fixer - Monitoring {ADAPTER_NAME}")
    
    print(f"--- WATCHDOG STARTED FOR: {ADAPTER_NAME} ---")
    print(f"--- Checking {CHECK_HOST} every {CHECK_INTERVAL} seconds ---")
    log_event("Script started monitoring.")

    while True:
        if not check_connection():
            reset_adapter()
        
        # Small delay to prevent high CPU usage
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping...")
        time.sleep(1)