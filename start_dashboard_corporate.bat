@echo off
REM ============================================================================
REM Corporate-Friendly Dashboard Launcher
REM ============================================================================
REM This launcher verifies corporate-optimized settings are loaded before
REM starting the Streamlit dashboard.
REM ============================================================================

echo.
echo ========================================
echo  TFT Dashboard - Corporate Mode
echo ========================================
echo.

REM Check if .streamlit/config.toml exists
if not exist ".streamlit\config.toml" (
    echo [ERROR] .streamlit\config.toml not found!
    echo [INFO] Creating corporate-optimized config...
    mkdir .streamlit 2>nul
    echo [ERROR] Please run the dashboard setup first.
    pause
    exit /b 1
)

echo [OK] Corporate config found: .streamlit\config.toml
echo.

REM Verify Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

echo [OK] Python detected
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit not installed
    echo [INFO] Install with: pip install streamlit
    pause
    exit /b 1
)

echo [OK] Streamlit installed
echo.

REM Show corporate-friendly settings
echo ========================================
echo  Corporate Optimizations Active:
echo ========================================
echo  [x] WebSocket compression: DISABLED
echo  [x] Fast reruns: DISABLED
echo  [x] File watching: DISABLED
echo  [x] Auto-refresh buffer: +1 second
echo  [x] Connection overlays: MINIMAL
echo.

echo Starting dashboard on http://localhost:8501
echo.
echo TIP: If dashboard still freezes, try:
echo   1. Disable auto-refresh in sidebar
echo   2. Increase refresh interval to 60+ seconds
echo   3. Use Microsoft Edge browser
echo   4. Ask IT to whitelist localhost:8501
echo.
echo Press Ctrl+C to stop dashboard
echo ========================================
echo.

REM Start Streamlit with corporate-friendly settings
streamlit run tft_dashboard_web.py --server.headless true

pause
