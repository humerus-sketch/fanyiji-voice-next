@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch_local.ps1" -Preview
if errorlevel 1 pause
