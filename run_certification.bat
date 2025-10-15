@echo off
echo.
echo ======================================================================
echo  RUNNING END-TO-END CERTIFICATION TEST
echo  This will verify all optimizations work correctly
echo ======================================================================
echo.

REM Activate conda environment
call conda activate py310

REM Run certification test
python end_to_end_certification.py

echo.
pause
