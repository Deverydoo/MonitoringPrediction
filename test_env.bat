@echo off
echo Testing .env file loading...
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%a in (.env) do (
        if "%%a"=="TFT_API_KEY" (
            echo Found TFT_API_KEY: %%b
        )
    )
)
