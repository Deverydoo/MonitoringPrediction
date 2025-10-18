#!/bin/bash
# NordIQ AI Systems - Linux/Mac Startup Script
# Copyright (c) 2025 NordIQ AI, LLC.
# Nordic precision, AI intelligence

echo "============================================"
echo "NordIQ AI Systems - Starting..."
echo "============================================"
echo

cd "$(dirname "$0")"

# Activate conda environment
if ! conda activate py310 2>/dev/null; then
    echo "[ERROR] Conda environment 'py310' not found"
    exit 1
fi

echo "[OK] Conda environment: py310"
echo

# Generate API key
echo "[INFO] Checking API key..."
python bin/generate_api_key.py
echo

# Load API key
if [ -f .env ]; then
    export $(cat .env | grep TFT_API_KEY)
fi

# Check for models
if [ ! -d "models" ]; then
    echo "[WARNING] No models found"
    echo "Run: python src/training/main.py train"
fi

# Start Inference Daemon
echo "[INFO] Starting Inference Daemon..."
gnome-terminal -- bash -c "cd $(pwd) && conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python src/daemons/tft_inference_daemon.py; exec bash" 2>/dev/null || \
xterm -e "cd $(pwd) && conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python src/daemons/tft_inference_daemon.py" &

sleep 5

# Start Metrics Generator
echo "[INFO] Starting Metrics Generator..."
gnome-terminal -- bash -c "cd $(pwd) && conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python src/daemons/metrics_generator_daemon.py --stream --servers 20; exec bash" 2>/dev/null || \
xterm -e "cd $(pwd) && conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python src/daemons/metrics_generator_daemon.py --stream --servers 20" &

sleep 3

# Start Dashboard
echo "[INFO] Starting Dashboard..."
gnome-terminal -- bash -c "cd $(pwd) && conda activate py310 && streamlit run src/dashboard/tft_dashboard_web.py --server.fileWatcherType none; exec bash" 2>/dev/null || \
xterm -e "cd $(pwd) && conda activate py310 && streamlit run src/dashboard/tft_dashboard_web.py --server.fileWatcherType none" &

echo
echo "============================================"
echo "System Started!"
echo "============================================"
echo
echo "Inference Daemon:   http://localhost:8000"
echo "Metrics Generator:  Streaming"
echo "Dashboard:          http://localhost:8501"
echo
