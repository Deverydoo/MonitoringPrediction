#!/bin/bash
# NordIQ AI Systems - Production Startup Script
# Copyright (c) 2025 NordIQ AI, LLC.
# Nordic precision, AI intelligence
#
# PRODUCTION MODE: Runs all services as background daemons
# - No conda activation (assumes correct environment is active)
# - Single Putty session (all processes daemonized)
# - PID tracking and log management
# - Service health checks

set -e  # Exit on error

# ============================================
# CONFIGURATION
# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/logs/pids"
DATA_DIR="$SCRIPT_DIR/data"
MODELS_DIR="$SCRIPT_DIR/models"

# Ports
INFERENCE_PORT=8000
METRICS_PORT=8001
DASHBOARD_PORT=8050

# ============================================
# SETUP
# ============================================

# Create necessary directories
mkdir -p "$LOG_DIR" "$PID_DIR" "$DATA_DIR" "$MODELS_DIR"

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
fi

# ============================================
# HELPER FUNCTIONS
# ============================================

check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -i ":$port" -t >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln | grep ":$port " >/dev/null 2>&1
    else
        # Fallback: try to connect
        timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null
    fi
}

wait_for_port() {
    local port=$1
    local service=$2
    local max_wait=30
    local elapsed=0

    echo -n "[INFO] Waiting for $service (port $port)..."
    while [ $elapsed -lt $max_wait ]; do
        if check_port $port; then
            echo " OK (${elapsed}s)"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
    done
    echo " TIMEOUT"
    return 1
}

check_python() {
    if ! command -v python >/dev/null 2>&1; then
        echo "[ERROR] Python not found in PATH"
        echo "        Activate your Python environment first:"
        echo "        conda activate py310"
        exit 1
    fi

    # Check for required packages
    if ! python -c "import torch, pytorch_forecasting, streamlit" 2>/dev/null; then
        echo "[WARNING] Some required packages may be missing"
        echo "          Ensure you're in the correct Python environment"
    fi
}

# ============================================
# MAIN STARTUP
# ============================================

echo "============================================"
echo "NordIQ AI Systems - Production Startup"
echo "============================================"
echo

cd "$SCRIPT_DIR"

# Check Python environment
check_python
echo "[OK] Python: $(python --version 2>&1)"
echo

# Generate/check API key
if [ -f "$SCRIPT_DIR/bin/generate_api_key.py" ]; then
    echo "[INFO] Checking API key..."
    python "$SCRIPT_DIR/bin/generate_api_key.py"
    # Reload .env after generation
    if [ -f "$SCRIPT_DIR/.env" ]; then
        export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
    fi
    echo
fi

# Check for models
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
    echo "[WARNING] No models found in $MODELS_DIR"
    echo "          Run training first: python src/training/tft_trainer.py"
    echo
fi

# Stop any existing services
echo "[INFO] Checking for existing services..."
for port in $INFERENCE_PORT $METRICS_PORT $DASHBOARD_PORT; do
    if check_port $port; then
        echo "[WARNING] Port $port is already in use"
        echo "          Run: ./stop_all.sh to stop existing services"
        exit 1
    fi
done
echo "[OK] All ports available"
echo

# ============================================
# START SERVICES
# ============================================

echo "[INFO] Starting services in daemon mode..."
echo

# 1. Start Inference Daemon
echo "[1/3] Starting Inference Daemon (port $INFERENCE_PORT)..."
nohup python src/daemons/tft_inference_daemon.py \
    --port $INFERENCE_PORT \
    > "$LOG_DIR/inference_daemon.log" 2>&1 &
INFERENCE_PID=$!
echo $INFERENCE_PID > "$PID_DIR/inference.pid"
echo "      PID: $INFERENCE_PID"
echo "      Log: $LOG_DIR/inference_daemon.log"

# Wait for inference daemon to be ready
if ! wait_for_port $INFERENCE_PORT "Inference Daemon"; then
    echo "[ERROR] Inference daemon failed to start"
    echo "        Check logs: tail -f $LOG_DIR/inference_daemon.log"
    exit 1
fi
echo

# 2. Start Metrics Generator Daemon
echo "[2/3] Starting Metrics Generator (port $METRICS_PORT)..."
nohup python src/daemons/metrics_generator_daemon.py \
    --stream \
    --servers 20 \
    --port $METRICS_PORT \
    > "$LOG_DIR/metrics_generator.log" 2>&1 &
METRICS_PID=$!
echo $METRICS_PID > "$PID_DIR/metrics.pid"
echo "      PID: $METRICS_PID"
echo "      Log: $LOG_DIR/metrics_generator.log"

# Wait for metrics generator to be ready
if ! wait_for_port $METRICS_PORT "Metrics Generator"; then
    echo "[ERROR] Metrics generator failed to start"
    echo "        Check logs: tail -f $LOG_DIR/metrics_generator.log"
    exit 1
fi
echo

# 3. Start Dash Dashboard
echo "[3/3] Starting Dash Dashboard (port $DASHBOARD_PORT)..."
nohup python dash_app.py \
    > "$LOG_DIR/dashboard.log" 2>&1 &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > "$PID_DIR/dashboard.pid"
echo "      PID: $DASHBOARD_PID"
echo "      Log: $LOG_DIR/dashboard.log"

# Wait for dashboard to be ready
if ! wait_for_port $DASHBOARD_PORT "Dashboard"; then
    echo "[ERROR] Dashboard failed to start"
    echo "        Check logs: tail -f $LOG_DIR/dashboard.log"
    exit 1
fi
echo

# ============================================
# STARTUP COMPLETE
# ============================================

echo "============================================"
echo "✅ NordIQ AI Systems - All Services Running"
echo "============================================"
echo
echo "Services:"
echo "  • Inference Daemon:   http://localhost:$INFERENCE_PORT"
echo "  • Metrics Generator:  http://localhost:$METRICS_PORT"
echo "  • Dashboard (Dash):   http://localhost:$DASHBOARD_PORT"
echo
echo "Process IDs:"
echo "  • Inference:  $INFERENCE_PID"
echo "  • Metrics:    $METRICS_PID"
echo "  • Dashboard:  $DASHBOARD_PID"
echo
echo "Logs:"
echo "  • Inference:  tail -f $LOG_DIR/inference_daemon.log"
echo "  • Metrics:    tail -f $LOG_DIR/metrics_generator.log"
echo "  • Dashboard:  tail -f $LOG_DIR/dashboard.log"
echo
echo "Management:"
echo "  • Stop all:   ./stop_all.sh"
echo "  • Status:     ./status.sh (or check PIDs above)"
echo "  • Logs:       ls -lh $LOG_DIR/"
echo
echo "🎉 System ready for production use!"
echo
