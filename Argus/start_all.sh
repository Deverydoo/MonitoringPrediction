#!/bin/bash
# Tachyon Argus - Production Startup Script
#
# PRODUCTION MODE: Runs all services as background daemons
# - No conda activation (assumes correct environment is active)
# - Single Putty session (all processes daemonized)
# - PID tracking and log management
# - Service health checks

set -e  # Exit on error

# ============================================
# CONFIGURATION - Adjust these as needed
# ============================================

# Conda environment name (change this if using a different environment)
CONDA_ENV="py310"

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

# Load NordIQ API key from dedicated file
if [ -f "$SCRIPT_DIR/.nordiq_key" ]; then
    export NORDIQ_API_KEY=$(cat "$SCRIPT_DIR/.nordiq_key")
fi

# Also load other environment variables from .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    # Load .env but don't overwrite NORDIQ_API_KEY if already set
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z $key ]] && continue
        # Skip if key is NORDIQ_API_KEY or TFT_API_KEY (we manage these)
        [[ $key == "NORDIQ_API_KEY" ]] && continue
        [[ $key == "TFT_API_KEY" ]] && continue
        # Export other variables
        export "$key=$value"
    done < "$SCRIPT_DIR/.env"
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

check_dependencies() {
    # Check if dash is installed (implies correct Python environment)
    if ! python -c "import dash" 2>/dev/null; then
        echo "[ERROR] Dash not installed or Python environment not activated"
        echo "        Install: pip install dash dash-bootstrap-components"
        echo "        Or activate your environment: conda activate $CONDA_ENV"
        exit 1
    fi
    echo "[OK] Dash installed"
}

# ============================================
# MAIN STARTUP
# ============================================

echo "============================================"
echo "ArgusAI - Production Startup"
echo "============================================"
echo

cd "$SCRIPT_DIR"

# Check dependencies
check_dependencies
echo

# Generate/check API key
if [ -f "$SCRIPT_DIR/bin/generate_api_key.py" ]; then
    echo "[INFO] Checking API key..."
    python "$SCRIPT_DIR/bin/generate_api_key.py"
    # Reload .nordiq_key after generation
    if [ -f "$SCRIPT_DIR/.nordiq_key" ]; then
        export NORDIQ_API_KEY=$(cat "$SCRIPT_DIR/.nordiq_key")
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
echo "✅ ArgusAI - All Services Running"
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
