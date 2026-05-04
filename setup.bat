@echo off
title HFT-RL Trader - First-Time Setup
color 0B

echo.
echo  ================================================
echo   HFT-RL Trader - First-Time Setup
echo  ================================================
echo.

:: ── Node dependencies ──────────────────────────────
echo [1/3] Installing Node.js dependencies...
call npm install
if %errorlevel% neq 0 (
    echo [ERROR] npm install failed
    pause
    exit /b 1
)
echo  Node deps OK.
echo.

:: ── Python venv ────────────────────────────────────
echo [2/3] Setting up Python virtual environment...
cd python_engine

if not exist ".venv" (
    python -m venv .venv
    echo  Virtual environment created.
) else (
    echo  Virtual environment already exists.
)

call .venv\Scripts\activate.bat

echo  Installing Python requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Some packages may have failed. Check output above.
)

:: Try to install PyTorch with CUDA 12 (RTX 3070)
echo.
echo  Installing PyTorch with CUDA 12.1 support (for RTX 3070)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [WARN] CUDA PyTorch install failed - falling back to CPU version
    pip install torch torchvision
)

cd ..
echo  Python env OK.
echo.

:: ── .env file ──────────────────────────────────────
echo [3/3] Setting up configuration...
if not exist "python_engine\.env" (
    echo # HFT-RL Trader Configuration > python_engine\.env
    echo ALPACA_API_KEY=your_alpaca_key_here >> python_engine\.env
    echo ALPACA_API_SECRET=your_alpaca_secret_here >> python_engine\.env
    echo ALPACA_PAPER_URL=https://paper-api.alpaca.markets >> python_engine\.env
    echo PYTHON_API_URL=http://localhost:8001 >> python_engine\.env
    echo  .env file created at python_engine\.env
    echo  IMPORTANT: Edit python_engine\.env and add your Alpaca API keys!
) else (
    echo  .env already exists.
)

echo.
echo  ================================================
echo   Setup complete!
echo.
echo   Next steps:
echo   1. Edit python_engine\.env with your Alpaca keys
echo   2. Double-click start.bat to launch
echo  ================================================
echo.
pause
