@echo off
setlocal
set "app=%~dp0dist\0.5.2-voice\FanyiJiVoiceNext\FanyiJiVoiceNext.exe"
set "profile=%~dp0settings\voice.settings.json"
if not exist "%app%" (
  echo FanyiJiVoiceNext executable was not found. Keep the complete project folder.
  pause
  exit /b 1
)
if not exist "%profile%" (
  echo Voice settings were not found: "%profile%"
  echo Restore settings\voice.settings.json from your backup before starting.
  pause
  exit /b 1
)
start "" "%app%" --model-root "%~dp0models" --settings-file "%profile%" --require-settings
