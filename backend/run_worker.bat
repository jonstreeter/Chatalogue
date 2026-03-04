@echo off
call .venv\Scripts\activate
set PythonUnbuffered=1
python debug_process.py
pause
