@echo off
REM Install security dependencies for TFT Inference Daemon
REM Run this before deploying to production

echo ========================================
echo Installing Security Dependencies
echo ========================================
echo.

echo [1/3] Installing slowapi (rate limiting)...
pip install slowapi
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install slowapi
    pause
    exit /b 1
)
echo OK: slowapi installed
echo.

echo [2/3] Installing pyarrow (secure Parquet persistence)...
pip install pyarrow
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install pyarrow
    pause
    exit /b 1
)
echo OK: pyarrow installed
echo.

echo [3/3] Verifying installation...
python -c "import slowapi, pyarrow; print('Security dependencies verified!')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Verification failed
    pause
    exit /b 1
)
echo.

echo ========================================
echo SUCCESS: All security dependencies installed
echo ========================================
echo.
echo Next steps:
echo 1. Set TFT_API_KEY environment variable
echo    Example: set TFT_API_KEY=your-secret-key-here
echo.
echo 2. Set CORS_ORIGINS environment variable
echo    Example: set CORS_ORIGINS=http://localhost:8501
echo.
echo 3. Start the daemon:
echo    python tft_inference_daemon.py
echo.
pause
