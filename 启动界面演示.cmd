@echo off
setlocal
set "python=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%python%" (
  echo Project Python environment was not found.
  pause
  exit /b 1
)
start "" "%python%" "%~dp0run_prototype.py"
