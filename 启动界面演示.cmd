@echo off
setlocal
set "app=%~dp0dist\0.2.0-prototype\FanyiJiVoiceNext\FanyiJiVoiceNext.exe"
if not exist "%app%" (
  echo Prototype executable was not found. Keep the complete project folder.
  pause
  exit /b 1
)
start "" "%app%"
