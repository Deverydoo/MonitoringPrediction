@echo off
REM ============================================================================
REM TFT Monitoring System - Corporate Environment Launcher
REM ============================================================================
REM This script starts all three services in the correct order:
REM   1. Metrics Generator Daemon (port 8001) - Generates realistic server data
REM   2. TFT Inference Daemon (port 8000) - Runs predictions
REM   3. Streamlit Dashboard (port 8501) - Web interface
REM
REM Each service runs in a separate window so you can monitor logs independently.
REM Press Ctrl+C in any window to stop that service.
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo  TFT Monitoring System - Corporate Mode Launcher
echo ============================================================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM ============================================================================
REM PRE-FLIGHT CHECKS
REM ============================================================================

echo [1/5] Running pre-flight checks...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo [INFO] Install Python 3.8+ and add to PATH
    pause
    exit /b 1
)
echo [OK] Python detected:
python --version
echo.

REM Check required packages
echo [2/5] Checking required packages...
set MISSING_PACKAGES=0

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch not installed
    set MISSING_PACKAGES=1
)

python -c "import pytorch_forecasting" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch Forecasting not installed
    set MISSING_PACKAGES=1
)

python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FastAPI not installed
    set MISSING_PACKAGES=1
)

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit not installed
    set MISSING_PACKAGES=1
)

if !MISSING_PACKAGES! equ 1 (
    echo.
    echo [ERROR] Missing required packages. Install with:
    echo pip install torch pytorch-forecasting fastapi uvicorn streamlit pandas plotly
    pause
    exit /b 1
)

echo [OK] All required packages installed
echo.

REM Check for trained model
echo [3/5] Checking for trained model...
if not exist "models\" (
    echo [WARNING] models/ directory not found
    echo [INFO] You'll need to train a model first: python tft_trainer.py
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" exit /b 0
) else (
    REM Find the most recent model
    for /f "delims=" %%d in ('dir /b /ad /o-d models\tft_model_* 2^>nul') do (
        set LATEST_MODEL=%%d
        goto :model_found
    )
    echo [WARNING] No trained models found in models/
    echo [INFO] Train a model first: python tft_trainer.py
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" exit /b 0
    goto :skip_model

    :model_found
    echo [OK] Found model: !LATEST_MODEL!
    :skip_model
)
echo.

REM Check corporate config
echo [4/5] Checking corporate configuration...
if not exist ".streamlit\config.toml" (
    echo [WARNING] .streamlit\config.toml not found
    echo [INFO] Creating corporate-optimized configuration...
    mkdir .streamlit 2>nul

    REM Create minimal corporate config inline
    (
        echo [server]
        echo enableWebsocketCompression = false
        echo enableXsrfProtection = false
        echo fileWatcherType = "none"
        echo enableCORS = false
        echo port = 8501
        echo headless = true
        echo.
        echo [browser]
        echo gatherUsageStats = false
        echo.
        echo [runner]
        echo magicEnabled = false
        echo fastReruns = false
        echo.
        echo [client]
        echo toolbarMode = "minimal"
        echo.
        echo [logger]
        echo level = "error"
    ) > .streamlit\config.toml

    echo [OK] Created .streamlit\config.toml
) else (
    echo [OK] Corporate config exists: .streamlit\config.toml
)
echo.

REM Check ports
echo [5/5] Checking if ports are available...
netstat -ano | findstr ":8000 " >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8000 already in use (Inference Daemon)
    echo [INFO] Stop the existing process or it will conflict
)

netstat -ano | findstr ":8001 " >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8001 already in use (Metrics Generator)
    echo [INFO] Stop the existing process or it will conflict
)

netstat -ano | findstr ":8501 " >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Port 8501 already in use (Dashboard)
    echo [INFO] Stop the existing process or it will conflict
)
echo.

REM ============================================================================
REM SERVICE LAUNCH
REM ============================================================================

echo ============================================================================
echo  Starting Services...
echo ============================================================================
echo.

echo Corporate-Friendly Optimizations:
echo   [x] WebSocket compression: DISABLED
echo   [x] Fast reruns: DISABLED
echo   [x] File watching: DISABLED
echo   [x] Auto-refresh buffer: +1 second
echo   [x] Connection overlays: MINIMAL
echo.

echo Service Launch Order:
echo   1. Metrics Generator (port 8001) - Wait 3 seconds
echo   2. Inference Daemon (port 8000) - Wait 5 seconds
echo   3. Dashboard (port 8501) - Ready to use!
echo.

echo Each service opens in a separate window.
echo Press Ctrl+C in any window to stop that service.
echo.

pause

REM Start Metrics Generator Daemon (port 8001)
echo [1/3] Starting Metrics Generator Daemon on port 8001...
start "TFT Metrics Generator (Port 8001)" cmd /k "python metrics_generator_daemon.py"
echo [OK] Metrics Generator starting in new window
echo      Wait 3 seconds for initialization...
timeout /t 3 /nobreak >nul
echo.

REM Start Inference Daemon (port 8000)
echo [2/3] Starting TFT Inference Daemon on port 8000...
start "TFT Inference Daemon (Port 8000)" cmd /k "python tft_inference_daemon.py"
echo [OK] Inference Daemon starting in new window
echo      Wait 5 seconds for model loading...
timeout /t 5 /nobreak >nul
echo.

REM Start Dashboard (port 8501)
echo [3/3] Starting Streamlit Dashboard on port 8501...
start "TFT Dashboard (Port 8501)" cmd /k "streamlit run tft_dashboard_web.py --server.headless true"
echo [OK] Dashboard starting in new window
echo      Wait 3 seconds for Streamlit initialization...
timeout /t 3 /nobreak >nul
echo.

REM ============================================================================
REM SUCCESS
REM ============================================================================

echo ============================================================================
echo  SUCCESS! All Services Started
echo ============================================================================
echo.

echo Services running:
echo   [1] Metrics Generator:  http://localhost:8001
echo   [2] Inference Daemon:   http://localhost:8000
echo   [3] Dashboard:          http://localhost:8501
echo.

echo Opening dashboard in browser...
timeout /t 2 /nobreak >nul
start http://localhost:8501

echo.
echo ============================================================================
echo  TIPS
echo ============================================================================
echo.
echo * Dashboard loads in 2-3 seconds (corporate-optimized)
echo * Three scenario buttons in sidebar: Healthy, Degrading, Critical
echo * Auto-refresh interval configurable (default: 30s)
echo * Press Ctrl+C in any window to stop that service
echo * To stop all services, close all three windows
echo.

echo Health check URLs:
echo   http://localhost:8001/health  (Metrics Generator)
echo   http://localhost:8000/health  (Inference Daemon)
echo.

echo If dashboard freezes:
echo   1. Disable auto-refresh in sidebar
echo   2. Increase refresh interval to 60+ seconds
echo   3. Use Microsoft Edge browser
echo   4. See CORPORATE_BROWSER_FIX.md for details
echo.

echo ============================================================================
echo  Press any key to exit this launcher window
echo  (Services will keep running in their own windows)
echo ============================================================================

pause >nul
