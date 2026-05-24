#!/bin/bash
# ==============================================================================
# Environment Setup and Provisioning Script
# AI-Assisted Detection of Stealth Network Reconnaissance
# ==============================================================================

set -e

PROJECT_ROOT="/home/yi/Stealth System"
VENV_DIR="$PROJECT_ROOT/venv"

echo "=========================================================="
echo "Initializing AI-Assisted Stealth IDS Setup"
echo "=========================================================="

# 1. Create directory structure
echo "Step 1: Creating project directory structure..."
mkdir -p "$PROJECT_ROOT/pcaps"
mkdir -p "$PROJECT_ROOT/dataset"
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/dashboard"
mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/configs"
mkdir -p "$PROJECT_ROOT/src"

echo "Directory structure created successfully."

# 2. Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed on this system." >&2
    exit 1
fi

# 3. Initialize virtual environment
echo "Step 2: Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# 4. Activate venv and install requirements
echo "Step 3: Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

echo "Python packages installed successfully."

# 5. Non-root packet capture capabilities configuration
echo "Step 4: Setting up capabilities for non-root live packet capture..."
PYTHON_BIN="$VENV_DIR/bin/python3"
if [ -f "$PYTHON_BIN" ]; then
    echo "Applying setcap cap_net_raw,cap_net_admin+eip to $PYTHON_BIN..."
    # Attempt to setcap; if sudo isn't available or fails, inform the user
    if sudo setcap cap_net_raw,cap_net_admin+eip "$PYTHON_BIN" 2>/dev/null; then
        echo "Capabilities set successfully! You can run sniffing scripts without sudo."
    else
        echo "--------------------------------------------------------"
        echo "[WARNING] Could not apply setcap automatically."
        echo "To capture packets without running as sudo, please run:"
        echo "  sudo setcap cap_net_raw,cap_net_admin+eip \"$PYTHON_BIN\""
        echo "--------------------------------------------------------"
    fi
fi

# Make src/ a python package
touch "$PROJECT_ROOT/src/__init__.py"

echo "=========================================================="
echo "Project setup completed successfully!"
echo "To activate the virtual environment, run:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "=========================================================="
