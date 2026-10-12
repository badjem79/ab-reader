@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo Ambiente non installato. Esegui prima setup.bat
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" app.py %*
