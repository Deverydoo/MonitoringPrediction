@echo off
call conda activate py310
python validate_linborg_schema.py
pause
