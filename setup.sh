#!/bin/bash

# Setup script for YOLO Inference and ONNX Conversion Project

echo "=========================================="
echo "YOLO Inference Project Setup"
echo "=========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "✓ Python 3 found: $(python3 --version)"

# Check Python version compatibility (PyTorch doesn't support Python 3.13 yet)
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo ""
    echo "⚠️  WARNING: Python 3.13+ detected. PyTorch may not have pre-built wheels for Python 3.13 yet."
    echo "   Recommended: Use Python 3.11 or 3.12 for best compatibility."
    echo ""
    echo "   To install Python 3.12 on macOS:"
    echo "     brew install python@3.12"
    echo "     python3.12 -m venv venv"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for python3.12 or python3.11 as alternatives
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    if command -v python3.12 &> /dev/null; then
        echo "Found python3.12, using it instead..."
        PYTHON_CMD=python3.12
    elif command -v python3.11 &> /dev/null; then
        echo "Found python3.11, using it instead..."
        PYTHON_CMD=python3.11
    else
        PYTHON_CMD=python3
        echo "⚠️  Using python3 (may have compatibility issues)"
    fi
else
    PYTHON_CMD=python3
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies from requirements.txt..."
echo "This may take several minutes, especially for PyTorch..."

# Try installing torch first to check compatibility
if pip install torch 2>&1 | grep -q "No matching distribution"; then
    echo ""
    echo "❌ ERROR: PyTorch installation failed. This is likely due to Python version incompatibility."
    echo ""
    echo "Solutions:"
    echo "1. Use Python 3.11 or 3.12:"
    echo "   brew install python@3.12"
    echo "   python3.12 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "2. Or try installing PyTorch from source (very slow):"
    echo "   pip install torch --no-binary torch"
    echo ""
    exit 1
fi

pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the inference script, run:"
echo "  python yolo_inference.py"
echo ""

