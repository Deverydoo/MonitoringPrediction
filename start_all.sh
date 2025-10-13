#!/bin/bash
# TFT Monitoring System - Linux/Mac Startup Script
# Starts inference daemon and dashboard in separate terminals

echo "============================================"
echo "TFT Monitoring System - Starting..."
echo "============================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found in PATH"
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "py310"; then
    echo "[ERROR] Conda environment 'py310' not found"
    echo ""
    echo "Please create it first:"
    echo "  conda create -n py310 python=3.10"
    echo "  conda activate py310"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "[OK] Conda environment: py310"
echo ""

# Check if model exists
if [ ! -d "models" ]; then
    echo "[WARNING] No models found in models/ directory"
    echo ""
    echo "To train a model:"
    echo "  1. Generate data: python metrics_generator.py --servers 20 --hours 720"
    echo "  2. Train model: python tft_trainer.py --epochs 20"
    echo ""
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310

echo "[INFO] Starting TFT Inference Daemon (clean architecture)..."
# Detect terminal emulator and start daemon
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "conda activate py310 && python tft_inference_daemon.py --port 8000; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -e "conda activate py310 && python tft_inference_daemon.py --port 8000; exec bash" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && conda activate py310 && python tft_inference_daemon.py --port 8000"'
else
    echo "[WARNING] Could not detect terminal emulator. Starting in background..."
    nohup python tft_inference_daemon.py --port 8000 > daemon.log 2>&1 &
    echo "Daemon PID: $!"
fi

echo "[INFO] Waiting for daemon to initialize (5 seconds)..."
sleep 5

echo "[INFO] Starting Metrics Generator Daemon (stream mode with REST API)..."
# Start metrics generator daemon
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "conda activate py310 && python metrics_generator_daemon.py --stream --servers 20; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -e "conda activate py310 && python metrics_generator_daemon.py --stream --servers 20; exec bash" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && conda activate py310 && python metrics_generator_daemon.py --stream --servers 20"'
else
    echo "[WARNING] Could not detect terminal emulator. Starting in background..."
    nohup python metrics_generator_daemon.py --stream --servers 20 > generator.log 2>&1 &
    echo "Generator PID: $!"
fi

echo "[INFO] Waiting for data stream to start (3 seconds)..."
sleep 3

echo "[INFO] Starting Streamlit Dashboard..."
# Start dashboard
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "conda activate py310 && streamlit run tft_dashboard_web.py; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -e "conda activate py310 && streamlit run tft_dashboard_web.py; exec bash" &
elif [[ "$OSTYPE" == "darwin"* ]]; then
    osascript -e 'tell app "Terminal" to do script "cd '"$(pwd)"' && conda activate py310 && streamlit run tft_dashboard_web.py"'
else
    echo "[WARNING] Could not detect terminal emulator. Starting in background..."
    nohup streamlit run tft_dashboard_web.py > dashboard.log 2>&1 &
    echo "Dashboard PID: $!"
fi

echo ""
echo "============================================"
echo "System Started!"
echo "============================================"
echo ""
echo "Inference Daemon:   http://localhost:8000"
echo "Metrics Generator:  Streaming 20 servers (HEALTHY scenario)"
echo "Dashboard:          http://localhost:8501"
echo ""
echo "[NETWORK ACCESS ENABLED]"
echo "Participants can access the dashboard at:"
echo "  http://YOUR_IP_ADDRESS:8501"
echo ""
echo "To find your IP address:"
echo "  Linux: ip addr show | grep 'inet '"
echo "  Mac:   ifconfig | grep 'inet '"
echo ""
echo "Logs:"
echo "  Daemon:    tail -f daemon.log (if running in background)"
echo "  Dashboard: tail -f dashboard.log (if running in background)"
echo ""
