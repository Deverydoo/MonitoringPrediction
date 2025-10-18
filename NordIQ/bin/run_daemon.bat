@echo off
REM Helper script to run daemon with environment variables from .env

REM Load environment variables from .env
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (.env) do (
        set "%%a=%%b"
    )
)

REM Activate conda and run daemon
call conda activate py310
python tft_inference_daemon.py --port 8000
