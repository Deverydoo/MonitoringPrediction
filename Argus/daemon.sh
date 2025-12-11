#!/bin/bash
# Tachyon Argus - Daemon Management Script (Linux/Mac)
#
# Usage:
#   ./daemon.sh start [inference|metrics|dashboard|all]
#   ./daemon.sh stop [inference|metrics|dashboard|all]
#   ./daemon.sh restart [inference|metrics|dashboard|all]
#   ./daemon.sh status

# ============================================
# CONFIGURATION - Adjust these as needed
# ============================================
CONDA_ENV="py310"

# ============================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# PID file locations
PID_DIR="$SCRIPT_DIR/.pids"
mkdir -p "$PID_DIR"

INFERENCE_PID="$PID_DIR/inference.pid"
METRICS_PID="$PID_DIR/metrics.pid"
DASHBOARD_PID="$PID_DIR/dashboard.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_environment() {
    # Activate conda environment
    if ! conda activate "$CONDA_ENV" 2>/dev/null; then
        log_error "Conda environment '$CONDA_ENV' not found"
        echo "Run: conda create -n $CONDA_ENV python=3.10"
        exit 1
    fi
    log_ok "Conda environment: $CONDA_ENV"

    # Check for models
    if [ ! -d "models" ]; then
        log_warn "No trained models found"
        echo "Run: python src/training/main.py train"
    fi
}

load_api_key() {
    log_info "Checking API key..."
    python bin/generate_api_key.py

    # Load API key from .env
    if [ -f .env ]; then
        export $(cat .env | grep TFT_API_KEY | xargs)
    fi

    if [ -z "$TFT_API_KEY" ]; then
        log_error "Failed to load API key"
        exit 1
    fi
    log_ok "API key loaded"
}

start_inference() {
    log_info "Starting Inference Daemon..."

    # Check if already running
    if [ -f "$INFERENCE_PID" ] && kill -0 $(cat "$INFERENCE_PID") 2>/dev/null; then
        log_warn "Inference Daemon already running (PID: $(cat $INFERENCE_PID))"
        return 0
    fi

    # Start daemon
    nohup python src/daemons/tft_inference_daemon.py > logs/inference.log 2>&1 &
    echo $! > "$INFERENCE_PID"

    # Wait and check if started
    sleep 2
    if kill -0 $(cat "$INFERENCE_PID") 2>/dev/null; then
        log_ok "Inference Daemon started (PID: $(cat $INFERENCE_PID), port 8000)"
    else
        log_error "Inference Daemon failed to start"
        rm -f "$INFERENCE_PID"
        return 1
    fi
}

start_metrics() {
    log_info "Starting Metrics Generator..."

    # Check if already running
    if [ -f "$METRICS_PID" ] && kill -0 $(cat "$METRICS_PID") 2>/dev/null; then
        log_warn "Metrics Generator already running (PID: $(cat $METRICS_PID))"
        return 0
    fi

    # Start daemon
    nohup python src/daemons/metrics_generator_daemon.py --stream --servers 20 > logs/metrics.log 2>&1 &
    echo $! > "$METRICS_PID"

    # Wait and check if started
    sleep 2
    if kill -0 $(cat "$METRICS_PID") 2>/dev/null; then
        log_ok "Metrics Generator started (PID: $(cat $METRICS_PID))"
    else
        log_error "Metrics Generator failed to start"
        rm -f "$METRICS_PID"
        return 1
    fi
}

start_dashboard() {
    log_info "Starting Dashboard..."

    # Check if already running
    if [ -f "$DASHBOARD_PID" ] && kill -0 $(cat "$DASHBOARD_PID") 2>/dev/null; then
        log_warn "Dashboard already running (PID: $(cat $DASHBOARD_PID))"
        return 0
    fi

    # Start dashboard
    nohup python dash_app.py > logs/dashboard.log 2>&1 &
    echo $! > "$DASHBOARD_PID"

    # Wait and check if started
    sleep 2
    if kill -0 $(cat "$DASHBOARD_PID") 2>/dev/null; then
        log_ok "Dashboard started (PID: $(cat $DASHBOARD_PID), port 8501)"
    else
        log_error "Dashboard failed to start"
        rm -f "$DASHBOARD_PID"
        return 1
    fi
}

stop_inference() {
    log_info "Stopping Inference Daemon..."

    if [ -f "$INFERENCE_PID" ]; then
        PID=$(cat "$INFERENCE_PID")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            sleep 2
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
            fi
            log_ok "Inference Daemon stopped"
        else
            log_warn "Inference Daemon not running"
        fi
        rm -f "$INFERENCE_PID"
    else
        log_warn "Inference Daemon not running (no PID file)"
    fi
}

stop_metrics() {
    log_info "Stopping Metrics Generator..."

    if [ -f "$METRICS_PID" ]; then
        PID=$(cat "$METRICS_PID")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            sleep 2
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
            fi
            log_ok "Metrics Generator stopped"
        else
            log_warn "Metrics Generator not running"
        fi
        rm -f "$METRICS_PID"
    else
        log_warn "Metrics Generator not running (no PID file)"
    fi
}

stop_dashboard() {
    log_info "Stopping Dashboard..."

    if [ -f "$DASHBOARD_PID" ]; then
        PID=$(cat "$DASHBOARD_PID")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            sleep 2
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID"
            fi
            log_ok "Dashboard stopped"
        else
            log_warn "Dashboard not running"
        fi
        rm -f "$DASHBOARD_PID"
    else
        log_warn "Dashboard not running (no PID file)"
    fi
}

start_all() {
    echo "============================================"
    echo "Starting All NordIQ Services"
    echo "============================================"
    echo

    check_environment
    load_api_key

    start_inference
    sleep 5

    start_metrics
    sleep 3

    start_dashboard

    echo
    echo "============================================"
    echo "All Services Started!"
    echo "============================================"
    show_urls
}

stop_all() {
    echo "============================================"
    echo "Stopping All NordIQ Services"
    echo "============================================"
    echo

    stop_dashboard
    stop_metrics
    stop_inference

    echo
    log_ok "All services stopped"
}

show_status() {
    echo "============================================"
    echo "NordIQ Services Status"
    echo "============================================"
    echo

    # Check inference daemon
    echo "Checking Inference Daemon (port 8000)..."
    if [ -f "$INFERENCE_PID" ] && kill -0 $(cat "$INFERENCE_PID") 2>/dev/null; then
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}[UP]${NC}   Inference Daemon - http://localhost:8000 (PID: $(cat $INFERENCE_PID))"
        else
            echo -e "  ${YELLOW}[WARN]${NC} Inference Daemon - process running but not responding"
        fi
    else
        echo -e "  ${RED}[DOWN]${NC} Inference Daemon"
    fi

    # Check dashboard
    echo "Checking Dashboard (port 8501)..."
    if [ -f "$DASHBOARD_PID" ] && kill -0 $(cat "$DASHBOARD_PID") 2>/dev/null; then
        if curl -s http://localhost:8501 > /dev/null 2>&1; then
            echo -e "  ${GREEN}[UP]${NC}   Dashboard - http://localhost:8501 (PID: $(cat $DASHBOARD_PID))"
        else
            echo -e "  ${YELLOW}[WARN]${NC} Dashboard - process running but not responding"
        fi
    else
        echo -e "  ${RED}[DOWN]${NC} Dashboard"
    fi

    # Check metrics generator
    echo "Checking Metrics Generator..."
    if [ -f "$METRICS_PID" ] && kill -0 $(cat "$METRICS_PID") 2>/dev/null; then
        echo -e "  ${GREEN}[UP]${NC}   Metrics Generator (PID: $(cat $METRICS_PID))"
    else
        echo -e "  ${RED}[DOWN]${NC} Metrics Generator"
    fi

    echo
}

show_urls() {
    echo
    echo "Access Points:"
    echo "  Inference API:  http://localhost:8000"
    echo "  API Docs:       http://localhost:8000/docs"
    echo "  Dashboard:      http://localhost:8501"
    echo
    echo "Logs:"
    echo "  Inference:      logs/inference.log"
    echo "  Metrics:        logs/metrics.log"
    echo "  Dashboard:      logs/dashboard.log"
    echo
}

show_usage() {
    echo
    echo "NordIQ Daemon Management"
    echo "========================"
    echo
    echo "Usage:"
    echo "  ./daemon.sh start [service]     - Start service(s)"
    echo "  ./daemon.sh stop [service]      - Stop service(s)"
    echo "  ./daemon.sh restart [service]   - Restart service(s)"
    echo "  ./daemon.sh status              - Show service status"
    echo
    echo "Services:"
    echo "  all         - All services (default)"
    echo "  inference   - Inference daemon only (port 8000)"
    echo "  metrics     - Metrics generator only"
    echo "  dashboard   - Dashboard only (port 8501)"
    echo
    echo "Examples:"
    echo "  ./daemon.sh start               - Start all services"
    echo "  ./daemon.sh start inference     - Start inference daemon only"
    echo "  ./daemon.sh stop dashboard      - Stop dashboard only"
    echo "  ./daemon.sh restart metrics     - Restart metrics generator"
    echo "  ./daemon.sh status              - Check what's running"
    echo
}

# Main command handling
case "$1" in
    start)
        SERVICE="${2:-all}"
        case "$SERVICE" in
            all)
                start_all
                ;;
            inference)
                check_environment
                load_api_key
                start_inference
                ;;
            metrics)
                check_environment
                load_api_key
                start_metrics
                ;;
            dashboard)
                check_environment
                start_dashboard
                ;;
            *)
                log_error "Unknown service: $SERVICE"
                show_usage
                exit 1
                ;;
        esac
        ;;

    stop)
        SERVICE="${2:-all}"
        case "$SERVICE" in
            all)
                stop_all
                ;;
            inference)
                stop_inference
                ;;
            metrics)
                stop_metrics
                ;;
            dashboard)
                stop_dashboard
                ;;
            *)
                log_error "Unknown service: $SERVICE"
                show_usage
                exit 1
                ;;
        esac
        ;;

    restart)
        SERVICE="${2:-all}"
        log_info "Restarting $SERVICE..."
        case "$SERVICE" in
            all)
                stop_all
                sleep 2
                start_all
                ;;
            inference)
                stop_inference
                sleep 2
                check_environment
                load_api_key
                start_inference
                ;;
            metrics)
                stop_metrics
                sleep 2
                check_environment
                load_api_key
                start_metrics
                ;;
            dashboard)
                stop_dashboard
                sleep 2
                check_environment
                start_dashboard
                ;;
            *)
                log_error "Unknown service: $SERVICE"
                show_usage
                exit 1
                ;;
        esac
        ;;

    status)
        show_status
        ;;

    *)
        show_usage
        exit 1
        ;;
esac
