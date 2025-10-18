@echo off
REM NordIQ AI Systems - Windows Startup Script
REM Copyright (c) 2025 NordIQ AI, LLC.
REM Nordic precision, AI intelligence

echo ============================================
echo NordIQ AI Systems - Starting...
echo ============================================

cd /d "%~dp0"

call conda activate py310 2>nul
if errorlevel 1 (
    echo [ERROR] Conda environment 'py310' not found
    pause
    exit /b 1
)

echo [OK] Conda environment: py310
echo.

echo [INFO] Checking API key...
python bin\generate_api_key.py
echo.

REM Load API key from .env file (trim whitespace)
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="TFT_API_KEY" (
            set "TFT_API_KEY=%%b"
        )
    )
)

REM Debug: Show API key (first 8 chars only)
if defined TFT_API_KEY (
    echo [OK] API key loaded from .env
) else (
    echo [ERROR] Failed to load API key from .env
    pause
    exit /b 1
)

if not exist "models\" (
    echo [WARNING] No models found
    echo Run: python src\training\main.py train
)

echo [INFO] Starting Inference Daemon...
start "TFT Inference Daemon" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\tft_inference_daemon.py"

timeout /t 5 /nobreak >nul

echo [INFO] Starting Metrics Generator...
start "Metrics Generator" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\metrics_generator_daemon.py --stream --servers 20"

timeout /t 3 /nobreak >nul

echo [INFO] Starting Dashboard...
start "NordIQ Dashboard" cmd /k "cd /d "%~dp0" && conda activate py310 && streamlit run src\dashboard\tft_dashboard_web.py --server.fileWatcherType none"

echo.
echo ============================================
echo System Started!
echo ============================================
echo.
echo Inference Daemon:   http://localhost:8000
echo Metrics Generator:  Streaming
echo Dashboard:          http://localhost:8501
echo.
pause
