@echo off
setlocal
set "app=%~dp0dist\0.5.2-dev-voice-r2\FanyiJiVoiceNext\FanyiJiVoiceNext.exe"
set "profile=%~dp0settings\voice.preview.settings.json"
set "sourceProfile=%~dp0settings\voice.settings.json"
if not exist "%app%" (
  echo The local 0.5.2-dev preview has not been built yet.
  pause
  exit /b 1
)
if not exist "%profile%" (
  if not exist "%sourceProfile%" (
    echo Voice settings were not found. Restore settings\voice.settings.json first.
    pause
    exit /b 1
  )
  copy /b "%sourceProfile%" "%profile%" >nul
  if errorlevel 1 exit /b 1
)
start "" "%app%" --model-root "%~dp0models" --settings-file "%profile%" --require-settings
