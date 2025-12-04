@echo off
REM ArgusAI - Daemon Management Script (Windows)
REM Built by Craig Giannelli and Claude Code
REM
REM Usage:
REM   daemon.bat start [inference|metrics|dashboard|all]
REM   daemon.bat stop [inference|metrics|dashboard|all]
REM   daemon.bat restart [inference|metrics|dashboard|all]
REM   daemon.bat status

REM ============================================
REM CONFIGURATION - Adjust these as needed
REM ============================================
set CONDA_ENV=py310

REM ============================================

cd /d "%~dp0"

if "%1"=="" goto show_usage
if "%1"=="start" goto start_service
if "%1"=="stop" goto stop_service
if "%1"=="restart" goto restart_service
if "%1"=="status" goto show_status
goto show_usage

:start_service
if "%2"=="" goto start_all
if "%2"=="all" goto start_all
if "%2"=="inference" goto start_inference
if "%2"=="metrics" goto start_metrics
if "%2"=="dashboard" goto start_dashboard
echo [ERROR] Unknown service: %2
goto show_usage

:stop_service
if "%2"=="" goto stop_all
if "%2"=="all" goto stop_all
if "%2"=="inference" goto stop_inference
if "%2"=="metrics" goto stop_metrics
if "%2"=="dashboard" goto stop_dashboard
echo [ERROR] Unknown service: %2
goto show_usage

:restart_service
if "%2"=="" set SERVICE=all
if not "%2"=="" set SERVICE=%2
echo [INFO] Restarting %SERVICE%...
call :stop_service_internal %SERVICE%
timeout /t 2 /nobreak >nul
call :start_service_internal %SERVICE%
goto end

:start_all
echo ============================================
echo Starting All NordIQ Services
echo ============================================
echo.
call :check_environment
call :load_api_key
call :start_inference
timeout /t 5 /nobreak >nul
call :start_metrics
timeout /t 3 /nobreak >nul
call :start_dashboard
echo.
echo ============================================
echo All Services Started!
echo ============================================
call :show_urls
goto end

:start_inference
echo [INFO] Starting Inference Daemon...
start "TFT Inference Daemon" cmd /k "cd /d "%~dp0" && conda activate %CONDA_ENV% && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\tft_inference_daemon.py"
echo [OK] Inference Daemon started (port 8000)
goto :eof

:start_metrics
echo [INFO] Starting Metrics Generator...
start "Metrics Generator" cmd /k "cd /d "%~dp0" && conda activate %CONDA_ENV% && set TFT_API_KEY=%TFT_API_KEY% && python src\daemons\metrics_generator_daemon.py --stream --servers 20"
echo [OK] Metrics Generator started
goto :eof

:start_dashboard
echo [INFO] Starting Dashboard...
start "ArgusAI Dashboard (Dash)" cmd /k "cd /d "%~dp0" && conda activate %CONDA_ENV% && python dash_app.py"
echo [OK] Dashboard started (port 8501)
goto :eof

:stop_all
echo ============================================
echo Stopping All NordIQ Services
echo ============================================
echo.
call :stop_inference
call :stop_metrics
call :stop_dashboard
echo.
echo [OK] All services stopped
goto end

:stop_inference
echo [INFO] Stopping Inference Daemon...
taskkill /FI "WINDOWTITLE eq TFT Inference Daemon*" /T /F 2>nul
if errorlevel 1 (
    echo [WARN] Inference Daemon not running
) else (
    echo [OK] Inference Daemon stopped
)
goto :eof

:stop_metrics
echo [INFO] Stopping Metrics Generator...
taskkill /FI "WINDOWTITLE eq Metrics Generator*" /T /F 2>nul
if errorlevel 1 (
    echo [WARN] Metrics Generator not running
) else (
    echo [OK] Metrics Generator stopped
)
goto :eof

:stop_dashboard
echo [INFO] Stopping Dashboard...
taskkill /FI "WINDOWTITLE eq ArgusAI Dashboard (Dash)*" /T /F 2>nul
if errorlevel 1 (
    echo [WARN] Dashboard not running
) else (
    echo [OK] Dashboard stopped
)
goto :eof

:show_status
echo ============================================
echo NordIQ Services Status
echo ============================================
echo.

REM Check inference daemon
echo Checking Inference Daemon (port 8000)...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo   [DOWN] Inference Daemon
) else (
    echo   [UP]   Inference Daemon - http://localhost:8000
)

REM Check dashboard
echo Checking Dashboard (port 8501)...
curl -s http://localhost:8501 >nul 2>&1
if errorlevel 1 (
    echo   [DOWN] Dashboard
) else (
    echo   [UP]   Dashboard - http://localhost:8501
)

REM Check metrics generator (by window title)
tasklist /FI "WINDOWTITLE eq Metrics Generator*" 2>nul | find "python" >nul
if errorlevel 1 (
    echo   [DOWN] Metrics Generator
) else (
    echo   [UP]   Metrics Generator
)

echo.
goto end

:check_environment
call conda activate %CONDA_ENV% 2>nul
if errorlevel 1 (
    echo [ERROR] Conda environment '%CONDA_ENV%' not found
    echo Run: conda create -n %CONDA_ENV% python=3.10
    exit /b 1
)
echo [OK] Conda environment: %CONDA_ENV%

if not exist "models\" (
    echo [WARN] No trained models found
    echo Run: python src\training\main.py train
)
goto :eof

:load_api_key
echo [INFO] Checking API key...
python bin\generate_api_key.py

REM Load API key from .env file
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="TFT_API_KEY" (
            set "TFT_API_KEY=%%b"
        )
    )
)

REM Strip trailing whitespace
for /f "tokens=* delims= " %%a in ("%TFT_API_KEY%") do set "TFT_API_KEY=%%a"

if defined TFT_API_KEY (
    echo [OK] API key loaded
) else (
    echo [ERROR] Failed to load API key
    exit /b 1
)
goto :eof

:show_urls
echo.
echo Access Points:
echo   Inference API:  http://localhost:8000
echo   API Docs:       http://localhost:8000/docs
echo   Dashboard:      http://localhost:8501
echo.
goto :eof

:show_usage
echo.
echo NordIQ Daemon Management
echo ========================
echo.
echo Usage:
echo   daemon.bat start [service]     - Start service(s)
echo   daemon.bat stop [service]      - Stop service(s)
echo   daemon.bat restart [service]   - Restart service(s)
echo   daemon.bat status              - Show service status
echo.
echo Services:
echo   all         - All services (default)
echo   inference   - Inference daemon only (port 8000)
echo   metrics     - Metrics generator only
echo   dashboard   - Dashboard only (port 8501)
echo.
echo Examples:
echo   daemon.bat start               - Start all services
echo   daemon.bat start inference     - Start inference daemon only
echo   daemon.bat stop dashboard      - Stop dashboard only
echo   daemon.bat restart metrics     - Restart metrics generator
echo   daemon.bat status              - Check what's running
echo.
goto end

:start_service_internal
if "%1"=="all" goto start_all
if "%1"=="inference" goto start_inference
if "%1"=="metrics" goto start_metrics
if "%1"=="dashboard" goto start_dashboard
goto :eof

:stop_service_internal
if "%1"=="all" goto stop_all
if "%1"=="inference" goto stop_inference
if "%1"=="metrics" goto stop_metrics
if "%1"=="dashboard" goto stop_dashboard
goto :eof

:end
