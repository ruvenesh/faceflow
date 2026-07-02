@echo off
echo Checking environment...

:: Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing/verifying dependencies...
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

:: Launch the app
echo Launching FaceFlow...
python app.py
pause
