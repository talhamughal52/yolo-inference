#!/bin/bash

# Quick fix script for Python 3.13 compatibility issues
# This script helps set up the environment with Python 3.12

echo "=========================================="
echo "Python Version Fix Script"
echo "=========================================="
echo ""

# Check if Python 3.12 is available
if command -v python3.12 &> /dev/null; then
    echo "✓ Python 3.12 found: $(python3.12 --version)"
    PYTHON_CMD=python3.12
elif command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 found: $(python3.11 --version)"
    PYTHON_CMD=python3.11
else
    echo "❌ Python 3.11 or 3.12 not found."
    echo ""
    echo "To install Python 3.12 on macOS:"
    echo "  brew install python@3.12"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Remove old venv if it exists
if [ -d "venv" ]; then
    echo ""
    read -p "Remove existing venv and create new one with $PYTHON_CMD? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf venv
    else
        echo "Keeping existing venv. Exiting."
        exit 0
    fi
fi

# Create new venv
echo "Creating new virtual environment with $PYTHON_CMD..."
$PYTHON_CMD -m venv venv

# Activate and install
echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the inference script, run:"
echo "  python yolo_inference.py"
echo ""

