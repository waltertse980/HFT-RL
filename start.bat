@echo off
title HFT-RL Trader
color 0A

echo.
echo  ================================================
echo   HFT-RL Trader - Starting All Services
echo  ================================================
echo.

:: Check Node is installed
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install from https://nodejs.org
    pause
    exit /b 1
)

:: Check Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install from https://python.org
    pause
    exit /b 1
)

echo [1/2] Starting Python engine on port 8001...
cd python_engine
start "HFT Python Engine" cmd /k "python -m uvicorn api_server:app --port 8001 --reload"
cd ..

:: Wait for Python to start
timeout /t 3 /nobreak >nul

echo [2/2] Starting Dashboard on port 5000...
start "HFT Dashboard" cmd /k "node dist\index.cjs"

:: Wait for dashboard to start
timeout /t 2 /nobreak >nul

echo.
echo  ================================================
echo   All services started!
echo.
echo   Dashboard:     http://localhost:5000
echo   Python API:    http://localhost:8001
echo   API Docs:      http://localhost:8001/docs
echo  ================================================
echo.
echo  Opening dashboard in browser...
timeout /t 1 /nobreak >nul
start http://localhost:5000

echo.
echo  Press any key to stop all services...
pause >nul

echo Stopping services...
taskkill /FI "WindowTitle eq HFT Python Engine*" /F >nul 2>&1
taskkill /FI "WindowTitle eq HFT Dashboard*" /F >nul 2>&1
echo Done.
