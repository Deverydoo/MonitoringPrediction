@echo off
REM ArgusAI - Windows Stop Script

echo Stopping ArgusAI...
echo.

taskkill /FI "WINDOWTITLE eq TFT Inference Daemon*" /T /F 2>nul
taskkill /FI "WINDOWTITLE eq Metrics Generator*" /T /F 2>nul
taskkill /FI "WINDOWTITLE eq NordIQ Dashboard*" /T /F 2>nul

echo.
echo All services stopped.
pause
