@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHONUTF8=1"

if not exist "%SCRIPT_DIR%run.py" (
  echo Could not find run.py next to this file.
  exit /b 1
)

python "%SCRIPT_DIR%run.py" %*
exit /b %errorlevel%
