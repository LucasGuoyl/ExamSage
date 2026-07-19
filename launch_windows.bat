@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/
  pause
  exit /b 1
)

py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_CMD=py -3.12"
) else (
  py -3.11 -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo ExamSage requires Python 3.11 or 3.12.
    pause
    exit /b 1
  )
  set "PYTHON_CMD=py -3.11"
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating ExamSage environment...
  %PYTHON_CMD% -m venv .venv || goto :error
)

echo Installing or updating ExamSage...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error

echo Opening ExamSage in your browser...
".venv\Scripts\python.exe" scripts\launch_app.py
exit /b 0

:error
echo.
echo ExamSage could not start. Copy the error above when opening a GitHub issue.
pause
exit /b 1
