@echo off
REM ArgusAI - PoC Comparison Startup Script
REM Starts both Streamlit and Dash dashboards for side-by-side comparison
REM Built by Craig Giannelli and Claude Code
REM Predictive System Monitoring

echo ============================================
echo ArgusAI PoC Comparison - Starting...
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

REM Load API key from .env file (trim whitespace and newlines)
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="TFT_API_KEY" (
            REM Strip any trailing whitespace/newlines by using delayed expansion
            set "TFT_API_KEY=%%b"
        )
    )
)

REM Strip trailing whitespace from API key (Windows batch bug workaround)
REM This removes any newline or carriage return characters
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

echo [INFO] Starting Inference Daemon...
start "TFT Inference Daemon" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\tft_inference_daemon.py"

timeout /t 5 /nobreak >nul

echo [INFO] Starting Metrics Generator...
start "Metrics Generator" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\metrics_generator_daemon.py --stream --servers 20"

timeout /t 3 /nobreak >nul

echo.
echo ============================================
echo Starting Dashboards for Comparison...
echo ============================================
echo.

echo [INFO] Starting Streamlit Dashboard (Port 8501)...
start "Streamlit Dashboard - LEGACY" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\dashboard\tft_dashboard_web.py"

timeout /t 2 /nobreak >nul

echo [INFO] Starting Dash PoC (Port 8050)...
start "Dash PoC - NEW" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python dash_poc.py"

echo.
echo ============================================
echo PoC Comparison Ready!
echo ============================================
echo.
echo Backend Services:
echo   Inference Daemon:   http://localhost:8000
echo   Metrics Generator:  Streaming (20 servers)
echo.
echo Dashboards (Compare Side-by-Side):
echo   Streamlit (OLD):    http://localhost:8501
echo   Dash PoC (NEW):     http://localhost:8050
echo.
echo ============================================
echo Performance Comparison Tips:
echo ============================================
echo.
echo 1. Open both URLs in separate browser windows
echo 2. Watch the render timers at the top of each dashboard
echo 3. Try switching tabs and refreshing
echo 4. Compare load times and responsiveness
echo.
echo Expected Performance:
echo   Streamlit: 2-4s page load (optimized)
echo   Dash PoC:  ~38ms render time (target: ^<500ms)
echo.
echo Press any key to exit (dashboards will keep running)...
pause
