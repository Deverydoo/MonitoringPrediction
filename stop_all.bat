@echo off
REM TFT Monitoring System - Windows Stop Script
REM Gracefully stops inference daemon and dashboard

echo ============================================
echo TFT Monitoring System - Stopping...
echo ============================================
echo.

echo [INFO] Stopping Streamlit Dashboard...
taskkill /FI "WINDOWTITLE eq TFT Dashboard*" /T /F 2>nul
if errorlevel 1 (
    echo [WARNING] Dashboard window not found, trying by process name...
    taskkill /IM streamlit.exe /F 2>nul
)

echo [INFO] Stopping TFT Inference Daemon...
taskkill /FI "WINDOWTITLE eq TFT Inference Daemon*" /T /F 2>nul
if errorlevel 1 (
    echo [WARNING] Daemon window not found, stopping by port...
    REM Find and kill process using port 8000
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
)

echo.
echo ============================================
echo System Stopped!
echo ============================================
echo.
pause
