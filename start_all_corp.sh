#!/bin/bash
################################################################################
# TFT Monitoring System - Silent Daemon Launcher (init.d/cron compatible)
################################################################################
# Usage:
#   ./start_all_corp.sh          # Silent daemon mode (production)
#   ./start_all_corp.sh verbose  # Verbose mode (debugging)
#
# Exit codes:
#   0 = Success
#   1 = Pre-flight check failed
#   2 = Service start failed
################################################################################

# Determine if verbose mode
VERBOSE=0
if [ "$1" = "verbose" ] || [ "$1" = "-v" ] || [ "$1" = "--verbose" ]; then
    VERBOSE=1
fi

# Logging function
log() {
    if [ $VERBOSE -eq 1 ]; then
        echo "$1"
    fi
    # Always log to file
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> logs/startup.log 2>/dev/null
}

# Error logging function (always to stderr and file)
log_error() {
    echo "ERROR: $1" >&2
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $1" >> logs/startup.log 2>/dev/null
}

# Change to script directory
cd "$(dirname "$0")" || exit 1

# Create logs directory
mkdir -p logs 2>/dev/null

################################################################################
# PRE-FLIGHT CHECKS (Silent)
################################################################################

log "Starting TFT Monitoring System..."

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found in PATH"
    exit 1
fi
log "Python detected: $(python3 --version 2>&1)"

# Check required packages (silent)
MISSING_PACKAGES=0
python3 -c "import torch" 2>/dev/null || MISSING_PACKAGES=1
python3 -c "import pytorch_forecasting" 2>/dev/null || MISSING_PACKAGES=1
python3 -c "import fastapi" 2>/dev/null || MISSING_PACKAGES=1
python3 -c "import streamlit" 2>/dev/null || MISSING_PACKAGES=1

if [ $MISSING_PACKAGES -eq 1 ]; then
    log_error "Missing required packages (torch, pytorch-forecasting, fastapi, streamlit)"
    exit 1
fi
log "All required packages installed"

# Check for trained model (non-fatal warning)
if [ ! -d "models" ]; then
    log "WARNING: models/ directory not found"
else
    LATEST_MODEL=$(ls -t models/tft_model_* 2>/dev/null | head -1)
    if [ -z "$LATEST_MODEL" ]; then
        log "WARNING: No trained models found in models/"
    else
        log "Found model: $(basename "$LATEST_MODEL")"
    fi
fi

# Create corporate config if missing (silent)
if [ ! -f ".streamlit/config.toml" ]; then
    log "Creating corporate configuration..."
    mkdir -p .streamlit 2>/dev/null
    cat > .streamlit/config.toml << 'EOF'
[server]
enableWebsocketCompression = false
enableXsrfProtection = false
fileWatcherType = "none"
enableCORS = false
port = 8501
headless = true

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false
fastReruns = false

[client]
toolbarMode = "minimal"

[logger]
level = "error"
EOF
    log "Corporate configuration created"
else
    log "Corporate configuration exists"
fi

# Generate/verify API key configuration
log "Checking API key configuration..."
python3 generate_api_key.py >> logs/startup.log 2>&1
if [ $? -ne 0 ]; then
    log_error "Failed to generate API key"
    exit 1
fi

# Load API key from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    log "API key loaded from .env"
else
    log "WARNING: .env file not found, running without API key"
fi

# Check ports (silent check, continue anyway)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":8000.*LISTEN"; then
    log "WARNING: Port 8000 already in use"
fi
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":8001.*LISTEN"; then
    log "WARNING: Port 8001 already in use"
fi
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -an 2>/dev/null | grep -q ":8501.*LISTEN"; then
    log "WARNING: Port 8501 already in use"
fi

################################################################################
# SERVICE LAUNCH (Silent Daemon Mode)
################################################################################

# Start Metrics Generator Daemon (port 8001)
log "Starting Metrics Generator (port 8001)..."
TFT_API_KEY="$TFT_API_KEY" python3 metrics_generator_daemon.py --stream --servers 20 > logs/metrics_generator.log 2>&1 &
METRICS_PID=$!
echo $METRICS_PID > logs/metrics_generator.pid
log "Metrics Generator started (PID: $METRICS_PID)"

# Verify process started
sleep 1
if ! kill -0 $METRICS_PID 2>/dev/null; then
    log_error "Metrics Generator failed to start"
    exit 2
fi

# Wait for initialization
sleep 2

# Start Inference Daemon (port 8000)
log "Starting Inference Daemon (port 8000)..."
TFT_API_KEY="$TFT_API_KEY" python3 tft_inference_daemon.py --port 8000 > logs/inference_daemon.log 2>&1 &
INFERENCE_PID=$!
echo $INFERENCE_PID > logs/inference_daemon.pid
log "Inference Daemon started (PID: $INFERENCE_PID)"

# Verify process started
sleep 1
if ! kill -0 $INFERENCE_PID 2>/dev/null; then
    log_error "Inference Daemon failed to start"
    exit 2
fi

# Wait for model loading
sleep 4

# Start Dashboard (port 8501)
log "Starting Dashboard (port 8501)..."
streamlit run tft_dashboard_web.py --server.headless true > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo $DASHBOARD_PID > logs/dashboard.pid
log "Dashboard started (PID: $DASHBOARD_PID)"

# Verify process started
sleep 1
if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
    log_error "Dashboard failed to start"
    exit 2
fi

# Wait for Streamlit initialization
sleep 2

################################################################################
# VERIFICATION
################################################################################

# Verify all services are running
ALL_RUNNING=1

if ! kill -0 $METRICS_PID 2>/dev/null; then
    log_error "Metrics Generator died after startup (PID: $METRICS_PID)"
    ALL_RUNNING=0
fi

if ! kill -0 $INFERENCE_PID 2>/dev/null; then
    log_error "Inference Daemon died after startup (PID: $INFERENCE_PID)"
    ALL_RUNNING=0
fi

if ! kill -0 $DASHBOARD_PID 2>/dev/null; then
    log_error "Dashboard died after startup (PID: $DASHBOARD_PID)"
    ALL_RUNNING=0
fi

if [ $ALL_RUNNING -eq 0 ]; then
    log_error "One or more services failed to start. Check logs/"
    exit 2
fi

################################################################################
# SUCCESS
################################################################################

log "All services started successfully"
log "Metrics Generator: PID $METRICS_PID (port 8001)"
log "Inference Daemon:  PID $INFERENCE_PID (port 8000)"
log "Dashboard:         PID $DASHBOARD_PID (port 8501)"

# Only output to stdout in verbose mode
if [ $VERBOSE -eq 1 ]; then
    echo "TFT Monitoring System started"
    echo "PIDs: Metrics=$METRICS_PID Inference=$INFERENCE_PID Dashboard=$DASHBOARD_PID"
    echo "Logs: logs/*.log"
fi

exit 0
