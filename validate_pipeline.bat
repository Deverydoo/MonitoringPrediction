@echo off
echo ================================================================================
echo LINBORG PIPELINE VALIDATION - Code Inspection Only
echo ================================================================================
echo.

echo STEP 1: Check metrics_generator_daemon.py for LINBORG metrics
echo --------------------------------------------------------------------------------
findstr /C:"cpu_user_pct" metrics_generator_daemon.py >nul
if %ERRORLEVEL%==0 (
    echo [PASS] Found cpu_user_pct in daemon code
) else (
    echo [FAIL] cpu_user_pct NOT FOUND in daemon code
)

findstr /C:"cpu_iowait_pct" metrics_generator_daemon.py >nul
if %ERRORLEVEL%==0 (
    echo [PASS] Found cpu_iowait_pct in daemon code
) else (
    echo [FAIL] cpu_iowait_pct NOT FOUND in daemon code
)

findstr /C:"mem_used_pct" metrics_generator_daemon.py >nul
if %ERRORLEVEL%==0 (
    echo [PASS] Found mem_used_pct in daemon code
) else (
    echo [FAIL] mem_used_pct NOT FOUND in daemon code
)

findstr /C:"for metric in \['cpu', 'mem'," metrics_generator_daemon.py >nul
if %ERRORLEVEL%==0 (
    echo [FAIL] OLD metric loop still exists!
) else (
    echo [PASS] OLD metric loop removed
)

echo.
echo STEP 2: Check tft_inference_daemon.py for LINBORG metrics
echo --------------------------------------------------------------------------------
findstr /C:"cpu_idle_pct" tft_inference_daemon.py >nul
if %ERRORLEVEL%==0 (
    echo [PASS] Found cpu_idle_pct in inference daemon
) else (
    echo [FAIL] cpu_idle_pct NOT FOUND
)

findstr /C:"'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct'" tft_inference_daemon.py >nul
if %ERRORLEVEL%==0 (
    echo [PASS] Heuristic loop includes all CPU metrics
) else (
    echo [FAIL] Heuristic loop incomplete
)

echo.
echo STEP 3: Check training data exists
echo --------------------------------------------------------------------------------
if exist training\server_metrics.parquet (
    echo [PASS] Training data file exists
    dir training\server_metrics.parquet | findstr /C:".parquet"
) else (
    echo [FAIL] Training data NOT FOUND
)

echo.
echo STEP 4: Check model exists
echo --------------------------------------------------------------------------------
if exist models\tft_model_20251014_131232\ (
    echo [PASS] Today's model exists
    dir models\tft_model_20251014_131232\model.safetensors | findstr /C:"safetensors"
) else (
    echo [WARN] Expected model tft_model_20251014_131232 not found
)

echo.
echo ================================================================================
echo VALIDATION COMPLETE
echo ================================================================================
pause
