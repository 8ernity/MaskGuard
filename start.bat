@echo off
setlocal
cd /d "%~dp0"

title MaskGuard - AI-Powered Face Mask Detection Web Application

:: Isolate from caller Python environments and configure Windows OpenBLAS/oneDNN stability
set "PYTHONHOME="
set "PYTHONPATH="
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "TF_ENABLE_ONEDNN_OPTS=0"

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in "%cd%".
    pause
    exit /b 1
)

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "REQ_STAMP=%VENV_DIR%\.requirements.sha256"

:: 1. Verify or create virtual environment
if exist "%VENV_PY%" if exist "%VENV_DIR%\Scripts\activate.bat" goto :check_deps

echo [*] Python virtual environment not found at %VENV_DIR%\
echo [*] Setting up virtual environment...

where uv >nul 2>&1
if errorlevel 1 goto :create_venv_python

echo [*] Found uv. Creating Python 3.11 virtual environment...
uv venv "%VENV_DIR%" --python 3.11 --seed
if not errorlevel 1 goto :install_deps
echo [!] uv venv setup failed. Falling back to system Python...

:create_venv_python
set "SYS_PY="
where py >nul 2>&1
if errorlevel 1 goto :try_plain_python

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py -3.11"
    goto :run_venv_create
)
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py -3.12"
    goto :run_venv_create
)
py -3.10 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py -3.10"
    goto :run_venv_create
)
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "SYS_PY=py -3"
    goto :run_venv_create
)

:try_plain_python
where python >nul 2>&1
if errorlevel 1 goto :no_python
python -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)" >nul 2>&1
if errorlevel 1 goto :no_python
set "SYS_PY=python"

:run_venv_create
echo [*] Creating virtual environment with %SYS_PY%...
%SYS_PY% -m venv "%VENV_DIR%"
if not errorlevel 1 goto :install_deps
echo [ERROR] Failed to create Python virtual environment.
echo         Verify Python 3.11 ^(64-bit^) with venv support is installed.
pause
exit /b 1

:no_python
echo [ERROR] Python 3.10-3.12 ^(64-bit^) or 'uv' was not found on PATH.
echo         Install Python 3.11 from https://www.python.org/downloads/
echo         ^(enable "Add python.exe to PATH"^) or install uv, then re-run start.bat.
pause
exit /b 1

:: 2. Verify dependencies match requirements.txt and are all installed
:check_deps
if not exist "%REQ_STAMP%" goto :install_deps
"%VENV_PY%" -c "import hashlib, importlib.metadata as md, pathlib, re; req=pathlib.Path('requirements.txt').read_bytes(); stamp=pathlib.Path(r'%REQ_STAMP%').read_text(encoding='utf-8').strip(); assert hashlib.sha256(req).hexdigest()==stamp; norm=lambda s: re.sub(r'[-_.]+','-',s).lower(); installed={norm(d.metadata['Name']) for d in md.distributions() if d.metadata and d.metadata.get('Name')}; reqs=[norm(re.split(r'[<>=!~;\s\[]', l.strip())[0]) for l in req.decode('utf-8').splitlines() if l.strip() and not l.strip().startswith('#')]; raise SystemExit(0 if all(p in installed for p in reqs if p) else 1)" >nul 2>&1
if not errorlevel 1 goto :deps_ready

:install_deps
echo [*] Installing/syncing MaskGuard dependencies from requirements.txt...
where uv >nul 2>&1
if errorlevel 1 goto :pip_install

uv pip install --python "%VENV_PY%" -r requirements.txt
if not errorlevel 1 goto :write_stamp
echo [!] uv pip install failed. Falling back to pip...

:pip_install
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 "%VENV_PY%" -m ensurepip --upgrade
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if not errorlevel 1 goto :write_stamp
echo [ERROR] Failed to install dependencies from requirements.txt.
echo         Check your network connection or Python version ^(3.11 recommended^).
pause
exit /b 1

:write_stamp
"%VENV_PY%" -c "import hashlib, pathlib; pathlib.Path(r'%REQ_STAMP%').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(), encoding='utf-8')"
if errorlevel 1 (
    echo [ERROR] Failed to write dependency verification stamp.
    pause
    exit /b 1
)

:deps_ready
call "%VENV_DIR%\Scripts\activate.bat"

:: 3. Optional workflow CLI arguments
if /i "%~1"=="eval" (
    "%VENV_PY%" training\evaluate.py
    set "EXIT_CODE=%ERRORLEVEL%"
    if "%~2"=="" pause
    exit /b %EXIT_CODE%
)
if /i "%~1"=="train" (
    "%VENV_PY%" training\train.py --epochs 30 --batch-size 32
    if not errorlevel 1 "%VENV_PY%" training\evaluate.py
    set "EXIT_CODE=%ERRORLEVEL%"
    if "%~2"=="" pause
    exit /b %EXIT_CODE%
)
if /i "%~1"=="test" (
    "%VENV_PY%" -m pytest tests/ -v
    set "EXIT_CODE=%ERRORLEVEL%"
    if "%~2"=="" pause
    exit /b %EXIT_CODE%
)
if /i "%~1"=="desktop" (
    "%VENV_PY%" main.py
    set "EXIT_CODE=%ERRORLEVEL%"
    if errorlevel 1 pause
    exit /b %EXIT_CODE%
)

:: 4. Launch Web Server in this console and open browser automatically
"%VENV_PY%" run_server.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] MaskGuard server exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
