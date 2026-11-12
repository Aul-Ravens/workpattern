@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Install the dependencies listed in README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m workpattern.server
if errorlevel 1 pause
