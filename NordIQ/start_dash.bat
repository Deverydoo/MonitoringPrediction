@echo off
REM ArgusAI - Dash Production Startup Script
REM Built by Craig Giannelli and Claude Code
REM Predictive System Monitoring

echo ============================================
echo ArgusAI Dash Dashboard - Starting...
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

echo [INFO] Starting Inference Daemon...
start "TFT Inference Daemon" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\tft_inference_daemon.py"

timeout /t 5 /nobreak >nul

echo [INFO] Starting Metrics Generator...
start "Metrics Generator" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\metrics_generator_daemon.py --stream --servers 20"

timeout /t 3 /nobreak >nul

echo.
echo ============================================
echo Starting Dash Production Dashboard...
echo ============================================
echo.

echo [INFO] Starting Dash app (Port 8050)...
start "Dash Dashboard - PRODUCTION" cmd /k "cd /d "%~dp0" && conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python dash_app.py"

echo.
echo ============================================
echo Dash Dashboard Started!
echo ============================================
echo.
echo Backend Services:
echo   Inference Daemon:   http://localhost:8000
echo   Metrics Generator:  Streaming (20 servers)
echo.
echo Dashboard:
echo   Dash Dashboard:     http://localhost:8050
echo.
echo ============================================
echo Performance Expectations:
echo ============================================
echo.
echo   Target:    ^<500ms page loads
echo   Expected:  ~78ms render time (15× faster than Streamlit)
echo.
echo   Tabs Available: 3/11 (27%%)
echo   - Overview (KPIs, risk distribution)
echo   - Heatmap (risk visualization)
echo   - Top 5 Risks (gauge charts)
echo.
echo   Coming Soon: 8 more tabs (Week 2-4)
echo   - Historical, Insights, Cost, Auto-Remediation, etc.
echo.
echo Press any key to exit (dashboard will keep running)...
pause
