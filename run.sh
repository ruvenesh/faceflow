#!/bin/bash

# FaceFlow Launcher for Linux / WSL

# Cool ASCII Art Header with cyan colors
echo -e "\e[36m"
echo "  ___               ___ _          "
echo " | __|_ _ __ ___   | __| |_____ __ "
echo " | _/ _\` / _/ -_)  | _|| / _ \ V  V /"
echo " |_|\__,_\__\___|  |_| |_\___/\_/\_/ "
echo -e "\e[0m"
echo -e "\e[1m\e[32m>>> Deepfake Data Harvester & Processing Pipeline <<<\e[0m"
echo ""
echo -e "\e[34m[System]\e[0m Checking environment..."

# Check if python3-venv is installed (Debian/Ubuntu specific check)
if ! dpkg -l | grep -q python3-venv; then
    echo "python3-venv is not installed. Please install it with: sudo apt install python3-venv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\e[34m[System]\e[0m Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "\e[34m[System]\e[0m Installing/verifying dependencies (this might take a moment)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Launch the app
echo -e "\e[32m[Success]\e[0m Dependencies are ready!"
echo -e "\e[35m[App]\e[0m Launching FaceFlow..."
python app.py
