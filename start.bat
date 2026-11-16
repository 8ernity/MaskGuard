@echo off
setlocal
cd /d "%~dp0"

title MaskGuard - AI-Powered Face Mask Detection Web Application

:: Windows OpenBLAS and oneDNN stability settings
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set TF_ENABLE_ONEDNN_OPTS=0

:: 1. Verify or create virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Python virtual environment not found at .venv\
    echo [*] Setting up virtual environment...
    where uv >nul 2>&1
    if not errorlevel 1 (
        call uv venv .venv --python 3.11
        call uv pip install -r requirements.txt
    ) else (
        python -m venv .venv
        call .venv\Scripts\pip.exe install -r requirements.txt
    )
)

call .venv\Scripts\activate.bat

:: 2. Optional workflow CLI arguments
if /i "%~1"=="eval" (
    python training\evaluate.py
    if "%~2"=="" pause
    exit /b 0
)
if /i "%~1"=="train" (
    python training\train.py --epochs 30 --batch-size 32
    python training\evaluate.py
    if "%~2"=="" pause
    exit /b 0
)
if /i "%~1"=="test" (
    pytest tests/ -v
    if "%~2"=="" pause
    exit /b 0
)

:: 3. Launch Web Server in this console and open browser automatically
python run_server.py
