@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" scripts\backup_data.py
) else (
  py -3 scripts\backup_data.py
)
pause
