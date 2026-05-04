@echo off
title HFT Python Engine (port 8001)
cd /d "%~dp0"

:: Activate venv if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] No .venv found - using system Python
)

:: Load .env
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" set "%%A=%%B"
    )
)

echo Starting HFT Python Engine on http://localhost:8001
echo API docs: http://localhost:8001/docs
echo.
python -m uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
