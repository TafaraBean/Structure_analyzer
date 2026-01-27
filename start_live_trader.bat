@echo off
echo ============================================================
echo   LIVE ORDER FLOW TRADING BOT
echo   Starting in new window...
echo ============================================================
echo.

cd /d "C:\Users\tafar\OneDrive\Desktop\Structure_bot"

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting live trader...
echo Press Ctrl+C in the new window to stop trading
echo.

start "Order Flow Live Trader" cmd /k "python live_order_flow_trader.py"

echo.
echo ✅ Live trader launched in new window!
echo.
pause
