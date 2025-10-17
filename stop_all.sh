#!/bin/bash
################################################################################
# TFT Monitoring System - Silent Stop Script (init.d/cron compatible)
################################################################################
# Usage:
#   ./stop_all.sh          # Silent mode (production)
#   ./stop_all.sh verbose  # Verbose mode (debugging)
#
# Exit codes:
#   0 = All services stopped successfully
#   1 = Some services were not running
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
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> logs/shutdown.log 2>/dev/null
}

# Change to script directory
cd "$(dirname "$0")" || exit 1

# Create logs directory if needed
mkdir -p logs 2>/dev/null

log "Stopping TFT Monitoring System..."

################################################################################
# STOP SERVICES
################################################################################

EXIT_CODE=0

# Function to stop a service (silent)
stop_service() {
    local NAME=$1
    local PID_FILE=$2
    local PORT=$3
    local STOPPED=0

    # Try PID file first
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
            kill $PID 2>/dev/null
            sleep 1
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID 2>/dev/null
                sleep 1
            fi
            if ! kill -0 $PID 2>/dev/null; then
                log "$NAME stopped (PID: $PID)"
                STOPPED=1
            else
                log "WARNING: Failed to stop $NAME (PID: $PID)"
                EXIT_CODE=1
            fi
            rm -f "$PID_FILE"
        else
            log "$NAME not running (stale PID file)"
            rm -f "$PID_FILE"
            EXIT_CODE=1
        fi
    fi

    # Try to find by port if not stopped yet
    if [ $STOPPED -eq 0 ]; then
        if command -v lsof &> /dev/null; then
            PID=$(lsof -ti:$PORT 2>/dev/null)
            if [ -n "$PID" ]; then
                kill $PID 2>/dev/null
                sleep 1
                if kill -0 $PID 2>/dev/null; then
                    kill -9 $PID 2>/dev/null
                    sleep 1
                fi
                if ! kill -0 $PID 2>/dev/null; then
                    log "$NAME stopped by port (PID: $PID)"
                    STOPPED=1
                else
                    log "WARNING: Failed to stop $NAME by port (PID: $PID)"
                    EXIT_CODE=1
                fi
            else
                log "$NAME not running (port $PORT not in use)"
                EXIT_CODE=1
            fi
        else
            log "WARNING: Cannot verify $NAME (lsof not available)"
            EXIT_CODE=1
        fi
    fi
}

# Stop all services
stop_service "Metrics Generator" "logs/metrics_generator.pid" "8001"
stop_service "Inference Daemon" "logs/inference_daemon.pid" "8000"
stop_service "Dashboard" "logs/dashboard.pid" "8501"

################################################################################
# VERIFICATION
################################################################################

# Verify all ports are free
sleep 1

if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log "WARNING: Port 8001 still in use"
    EXIT_CODE=1
fi

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log "WARNING: Port 8000 still in use"
    EXIT_CODE=1
fi

if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    log "WARNING: Port 8501 still in use"
    EXIT_CODE=1
fi

################################################################################
# COMPLETE
################################################################################

if [ $EXIT_CODE -eq 0 ]; then
    log "All services stopped successfully"
    if [ $VERBOSE -eq 1 ]; then
        echo "TFT Monitoring System stopped"
    fi
else
    log "WARNING: Some services were not running or failed to stop"
    if [ $VERBOSE -eq 1 ]; then
        echo "WARNING: Some services were not running or failed to stop"
        echo "Check logs/shutdown.log for details"
    fi
fi

exit $EXIT_CODE
