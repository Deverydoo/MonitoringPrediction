@echo off
REM Tachyon Argus - Dash Production Startup Script

REM ============================================
REM CONFIGURATION - Adjust these as needed
REM ============================================
set CONDA_ENV=py310
set INFERENCE_PORT=8000
set METRICS_PORT=8001
set DASHBOARD_PORT=8050

REM ============================================

echo ============================================
echo Tachyon Argus Dashboard - Starting...
echo ============================================

cd /d "%~dp0"

call conda activate %CONDA_ENV% 2>nul
if errorlevel 1 (
    echo [ERROR] Conda environment '%CONDA_ENV%' not found
    pause
    exit /b 1
)

echo [OK] Conda environment: %CONDA_ENV%
echo.

echo [INFO] Checking API key...
python bin\generate_api_key.py
echo.

REM Load API key from .env file (trim whitespace and newlines)
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="TFT_API_KEY" (
            set "TFT_API_KEY=%%b"
        )
    )
)

REM Strip trailing whitespace from API key
for /f "tokens=* delims= " %%a in ("%TFT_API_KEY%") do set "TFT_API_KEY=%%a"

REM Validate API key loaded
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

echo.
echo ============================================
echo Starting Backend Services...
echo ============================================
echo.

echo [INFO] Starting Inference Daemon (port %INFERENCE_PORT%)...
start "TFT Inference Daemon" cmd /k "cd /d "%~dp0" && conda activate %CONDA_ENV% && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\tft_inference_daemon.py --port %INFERENCE_PORT%"

timeout /t 5 /nobreak >nul

echo [INFO] Starting Metrics Generator (port %METRICS_PORT%)...
start "Metrics Generator" cmd /k "cd /d "%~dp0" && conda activate %CONDA_ENV% && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\metrics_generator_daemon.py --stream --servers 20 --port %METRICS_PORT%"

timeout /t 3 /nobreak >nul

echo.
echo ============================================
echo Starting Dash Production Dashboard...
echo ============================================
echo.

echo [INFO] Starting Dashboard (port %DASHBOARD_PORT%)...
start "Argus Dashboard - PRODUCTION" cmd /k "cd /d "%~dp0" && conda activate %CONDA_ENV% && set TFT_API_KEY=%TFT_API_KEY% && python dash_app.py --port %DASHBOARD_PORT%"

echo.
echo ============================================
echo Dash Dashboard Started!
echo ============================================
echo.
echo Backend Services:
echo   Inference Daemon:   http://localhost:%INFERENCE_PORT%
echo   Metrics Generator:  http://localhost:%METRICS_PORT%
echo.
echo Dashboard:
echo   Dashboard:          http://localhost:%DASHBOARD_PORT%
echo.
echo ============================================
echo Performance Expectations:
echo ============================================
echo.
echo   Target:    ^<500ms page loads
echo   Expected:  ~78ms render time
echo.
echo   All 10 tabs available at http://localhost:%DASHBOARD_PORT%
echo.
echo Press any key to exit (dashboard will keep running)...
pause
