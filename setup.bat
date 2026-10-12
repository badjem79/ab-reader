@echo off
setlocal
cd /d "%~dp0"

echo === AB Reader - installazione ===
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creo l'ambiente virtuale...
    py -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERRORE: Python non trovato. Installa Python 3.10 o superiore da python.org
        pause
        exit /b 1
    )
)

echo Installo le dipendenze...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERRORE durante l'installazione delle dipendenze.
    pause
    exit /b 1
)

echo.
echo Scarico i modelli vocali (circa 350 MB, solo la prima volta)...
".venv\Scripts\python.exe" -m abreader.download_models
if errorlevel 1 (
    echo.
    echo ERRORE durante il download dei modelli.
    pause
    exit /b 1
)

echo.
echo === Fatto. Avvia il lettore con run.bat ===
pause
