@echo off
cd /d "%~dp0"
set "KITCHEN_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%KITCHEN_PYTHON%" goto run
set "KITCHEN_PYTHON=%~dp0..\..\work\kitchen-venv\Scripts\python.exe"
if exist "%KITCHEN_PYTHON%" goto run
python -m venv .venv
if errorlevel 1 goto failed
set "KITCHEN_PYTHON=%~dp0.venv\Scripts\python.exe"
"%KITCHEN_PYTHON%" -m pip install -r requirements-ml.txt
if errorlevel 1 goto failed
:run
echo Open http://127.0.0.1:8765 in your browser after the server starts.
echo Close this window or press Ctrl+C to stop the app.
"%KITCHEN_PYTHON%" server.py
pause
exit /b
:failed
echo Setup failed. See README.md for manual setup instructions.
pause
