@echo off
REM ============================================================================
REM TFT Monitoring System - Stop All Services (Windows)
REM ============================================================================
REM This script stops all three services gracefully by terminating their
REM processes on the specified ports.
REM ============================================================================

echo.
echo ============================================================================
echo  Stopping TFT Monitoring System
echo ============================================================================
echo.

REM Kill processes by port (more reliable than window titles)
echo [1/3] Stopping Metrics Generator (port 8001)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001 "') do (
    taskkill /PID %%a /F >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Metrics Generator stopped
    )
)

echo [2/3] Stopping Inference Daemon (port 8000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 "') do (
    taskkill /PID %%a /F >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Inference Daemon stopped
    )
)

echo [3/3] Stopping Dashboard (port 8501)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501 "') do (
    taskkill /PID %%a /F >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Dashboard stopped
    )
)

echo.
echo ============================================================================
echo  All services stopped
echo ============================================================================
echo.

pause
