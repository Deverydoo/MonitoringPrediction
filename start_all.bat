@echo off
REM TFT Monitoring System - Windows Startup Script
REM Starts inference daemon and dashboard in separate windows

echo ============================================
echo TFT Monitoring System - Starting...
echo ============================================
echo.

REM Check if conda environment exists
call conda activate py310 2>nul
if errorlevel 1 (
    echo [ERROR] Conda environment 'py310' not found
    echo.
    echo Please create it first:
    echo   conda create -n py310 python=3.10
    echo   conda activate py310
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [OK] Conda environment: py310
echo.

REM Generate/verify API key configuration
echo [INFO] Checking API key configuration...
python generate_api_key.py
if errorlevel 1 (
    echo [ERROR] Failed to generate API key
    pause
    exit /b 1
)
echo.

REM Load API key from .env file
if exist .env (
    for /f "usebackq tokens=1,2 delims==" %%a in (.env) do (
        if "%%a"=="TFT_API_KEY" set TFT_API_KEY=%%b
    )
    if defined TFT_API_KEY (
        echo [OK] API key loaded: %TFT_API_KEY:~0,20%...
    ) else (
        echo [WARNING] TFT_API_KEY not found in .env file
    )
) else (
    echo [WARNING] .env file not found, running without API key
)
echo.

REM Check if model exists
if not exist "models\" (
    echo [WARNING] No models found in models/ directory
    echo.
    echo To train a model:
    echo   1. Generate data: python metrics_generator.py --servers 20 --hours 720
    echo   2. Train model: python tft_trainer.py --epochs 20
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
)

echo [INFO] Starting TFT Inference Daemon (clean architecture)...
start "TFT Inference Daemon" cmd /k "run_daemon.bat"

echo [INFO] Waiting for daemon to initialize (5 seconds)...
timeout /t 5 /nobreak >nul

echo [INFO] Pre-compiling Python modules for faster startup...
python precompile.py >nul 2>&1
echo [OK] Bytecode compilation complete
echo.

echo [INFO] Starting Metrics Generator Daemon (stream mode with REST API)...
start "Metrics Generator" cmd /k "conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python metrics_generator_daemon.py --stream --servers 20"

echo [INFO] Waiting for data stream to start (3 seconds)...
timeout /t 3 /nobreak >nul

echo [INFO] Starting Streamlit Dashboard (production mode)...
start "TFT Dashboard" cmd /k "conda activate py310 && streamlit run tft_dashboard_web.py --server.fileWatcherType none --server.runOnSave false"

echo.
echo ============================================
echo System Started!
echo ============================================
echo.
echo Inference Daemon:   http://localhost:8000
echo Metrics Generator:  Streaming 20 servers (HEALTHY scenario)
echo Dashboard:          http://localhost:8501
echo.
echo [NETWORK ACCESS ENABLED]
echo Participants can access the dashboard at:
echo   http://YOUR_IP_ADDRESS:8501
echo.
echo To find your IP address, run: ipconfig
echo Look for "IPv4 Address" under your network adapter
echo.
echo Close this window or press any key to continue...
echo (Daemon and Dashboard will keep running)
pause >nul
