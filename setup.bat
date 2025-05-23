@echo off

REM OWL Control Setup Script for Windows
REM This script sets up both the Python and Electron/Next.js components

echo OWL Control Setup Script
echo ======================

REM Check for required tools
echo.
echo Checking required tools...

REM Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Node.js is not installed. Please install Node.js from https://nodejs.org/
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('node --version') do echo ✓ Node.js %%i found
)

REM Check npm
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: npm is not installed. Please install npm.
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('npm --version') do echo ✓ npm %%i found
)

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed. Please install Python 3.
    exit /b 1
) else (
    for /f "tokens=*" %%i in ('python --version') do echo ✓ Python %%i found
)

REM Setup Python environment
echo.
echo Setting up Python environment...

REM Check for uv or poetry
set PYTHON_TOOL=
where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ uv found
    set PYTHON_TOOL=uv
) else (
    where poetry >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ poetry found
        set PYTHON_TOOL=poetry
    ) else (
        echo Neither uv nor poetry found. Installing uv...
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        
        REM Add uv to PATH for current session
        set PATH=%USERPROFILE%\.cargo\bin;%PATH%
        
        where uv >nul 2>nul
        if %ERRORLEVEL% EQU 0 (
            echo ✓ uv installed successfully
            set PYTHON_TOOL=uv
        ) else (
            echo Failed to install uv. Please install uv or poetry manually.
            exit /b 1
        )
    )
)

REM Install Python dependencies based on tool choice
echo Installing Python dependencies with %PYTHON_TOOL%...
if "%PYTHON_TOOL%"=="uv" (
    REM Create and activate virtual environment with uv
    uv venv
    call .venv\Scripts\activate
    
    REM Install dependencies with uv
    REM Since we have a pyproject.toml, we can install as a package
    uv pip install -e .
) else (
    REM Use poetry
    poetry install
)

REM Setup Node.js/Electron environment
echo.
echo Setting up Node.js/Electron environment...

REM Install npm dependencies
echo Installing npm dependencies...
npm install

REM Build the Electron app
echo Building the Electron app...
npm run build

REM Create necessary directories
echo.
echo Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "dist" mkdir dist

REM Setup environment file
echo.
echo Setting up environment configuration...
if not exist ".env" (
    echo Creating .env file...
    (
        echo # Environment configuration for OWL Control
        echo NODE_ENV=development
        echo PYTHON_TOOL=%PYTHON_TOOL%
        echo.
        echo # Add your configuration here
        echo # OBS_WEBSOCKET_URL=ws://localhost:4444
        echo # OBS_WEBSOCKET_PASSWORD=your_password
    ) > .env
    echo Created .env file. Please update it with your configuration.
) else (
    echo .env file already exists
)

echo.
echo Setup complete!
echo.
echo To run the application:
echo 1. Activate the Python environment:
if "%PYTHON_TOOL%"=="uv" (
    echo    .venv\Scripts\activate
) else (
    echo    poetry shell
)
echo 2. Run the Python backend:
echo    python vg_control\main.py
echo 3. In a separate terminal, run the Electron app:
echo    npm start
echo.
echo For development mode:
echo    npm run dev
echo.
echo To package the application:
echo    npm run package

pause