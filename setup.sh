#!/usr/bin/env bash

# ------------------------------------------------------------
# Quick setup script for mujoco‑scene‑editor
# ------------------------------------------------------------
# This script automates the creation of a compatible Python environment,
# installs the project and its dependencies (including the `robits`
# package from source), starts the local Ollama server and pulls the
# required model.
#
# Prerequisites:
#   * Python 3.12 must be installed and available as `python3.12`.
#   * `git` must be installed.
#   * Ollama must be installed (see https://ollama.com).
#   * (Optional) `conda`/`mamba` can be used to create the environment.
# ------------------------------------------------------------

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "=== Checking for Python 3.12 ==="
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Error: python3.12 not found. Install it first (e.g., sudo dnf install python3.12)."
    exit 1
fi

PYTHON=python3.12

# ------------------------------------------------------------
# Create virtual environment
# ------------------------------------------------------------
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment…"
    rm -rf "$VENV_DIR"
fi
echo "Creating virtual environment with $PYTHON…"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# Install the project (editable) and its dependencies
# ------------------------------------------------------------
echo "Installing mujoco‑scene‑editor (editable)…"
pip install -e .

# ------------------------------------------------------------
# Install robits from source (required for Python 3.12)
# ------------------------------------------------------------
if [ ! -d "robits" ]; then
    echo "Cloning robits repository…"
    git clone https://github.com/robits/robits.git
fi
cd robits
echo "Installing robits (editable)…"
pip install -e .
cd "$PROJECT_ROOT"

# ------------------------------------------------------------
# Verify installation of console script
# ------------------------------------------------------------
if ! command -v mjprompt >/dev/null 2>&1; then
    echo "Error: mjprompt command not found after installation."
    exit 1
fi

# ------------------------------------------------------------
# Start Ollama server (if not already running) and pull model
# ------------------------------------------------------------
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    echo "Starting Ollama server in background…"
    nohup ollama serve >/dev/null 2>&1 &
    # Give it a few seconds to start
    sleep 5
fi

echo "Pulling required model gpt-oss:120b-cloud…"
ollama pull gpt-oss:120b-cloud

echo "Setup complete!"
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo "Then run: mjprompt"
