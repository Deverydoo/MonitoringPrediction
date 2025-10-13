#!/bin/bash
# TFT Monitoring System - Linux/Mac Stop Script
# Gracefully stops inference daemon and dashboard

echo "============================================"
echo "TFT Monitoring System - Stopping..."
echo "============================================"
echo ""

echo "[INFO] Stopping Streamlit Dashboard..."
pkill -f "streamlit run tft_dashboard_web.py" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[OK] Dashboard stopped"
else
    echo "[WARNING] Dashboard not running"
fi

echo "[INFO] Stopping TFT Inference Daemon..."
pkill -f "tft_inference.py --daemon" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "[OK] Daemon stopped"
else
    echo "[WARNING] Daemon not running"
fi

# Alternative: Kill by port
echo "[INFO] Checking for processes on port 8000..."
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:8000 2>/dev/null)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
        echo "[OK] Killed process on port 8000 (PID: $PID)"
    fi
elif command -v fuser &> /dev/null; then
    fuser -k 8000/tcp 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "[OK] Killed process on port 8000"
    fi
fi

echo ""
echo "============================================"
echo "System Stopped!"
echo "============================================"
echo ""
