@echo off
setlocal enabledelayedexpansion

echo 🧪 Quick Packaging Test for OWL Control
echo =======================================

REM Function to check if command exists
:check_command
where %1 >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ %1 is not installed
    exit /b 1
) else (
    echo ✅ %1 is available
    exit /b 0
)

echo.
echo Checking prerequisites...

call :check_command node
if errorlevel 1 goto :error

call :check_command npm  
if errorlevel 1 goto :error

call :check_command python
if errorlevel 1 (
    call :check_command python3
    if errorlevel 1 goto :error
)

call :check_command pip
if errorlevel 1 (
    call :check_command pip3
    if errorlevel 1 goto :error
)

echo.
echo Running quick Python bundle test...

REM Clean previous Python dist
if exist python-dist rmdir /s /q python-dist
if exist vg_control.spec del vg_control.spec

echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Trying with pip3...
    pip3 install -r requirements.txt
    if errorlevel 1 goto :error
)

echo Bundling Python code...
node scripts/bundle-python.js
if errorlevel 1 goto :error

REM Check results
if exist python-dist (
    echo.
    echo ✅ Python bundling successful!
    echo Contents of python-dist:
    dir python-dist
    
    REM Test if the bundled executable works
    if exist "python-dist\vg_control_backend\vg_control_backend.exe" (
        echo.
        echo ✅ Python executable found!
    ) else (
        echo.
        echo ❌ Python executable not found
        goto :error
    )
) else (
    echo.
    echo ❌ Python bundling failed!
    goto :error
)

echo.
echo Quick test completed!
echo To run full packaging test: npm run test:packaging
goto :end

:error
echo.
echo ❌ Test failed! Check the errors above.
exit /b 1

:end
echo.
echo Press any key to continue...
pause >nul