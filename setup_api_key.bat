@echo off
REM Setup API key for TFT Monitoring System
REM This script copies the example .env file and sets up the API key

echo.
echo ================================================
echo TFT Monitoring System - API Key Setup
echo ================================================
echo.

if exist .env (
    echo [WARNING] .env file already exists!
    echo.
    choice /C YN /M "Do you want to overwrite it"
    if errorlevel 2 goto :skip
)

echo Creating .env file from .env.example...
copy .env.example .env >nul

echo.
echo ================================================
echo API Key Setup Complete!
echo ================================================
echo.
echo The API key has been configured:
echo   - Dashboard: .streamlit/secrets.toml
echo   - Daemon:    .env (TFT_API_KEY)
echo.
echo Next steps:
echo   1. Start the daemon: python tft_inference_daemon.py
echo   2. Start the dashboard: streamlit run tft_dashboard_web.py
echo.
echo The dashboard and daemon will now authenticate with the API key.
echo.
goto :end

:skip
echo.
echo Skipping .env file creation (already exists)
echo.

:end
pause
