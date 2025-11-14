#!/bin/bash
# ArgusAI - Production Stop Script
# Built by Craig Giannelli and Claude Code
# Predictive System Monitoring

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/logs/pids"

echo "============================================"
echo "ArgusAI - Stopping Services"
echo "============================================"
echo

cd "$SCRIPT_DIR"

# Function to stop service by PID file
stop_service() {
    local name=$1
    local pid_file="$PID_DIR/${name}.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "[INFO] Stopping $name (PID: $pid)..."
            kill $pid 2>/dev/null

            # Wait up to 10 seconds for graceful shutdown
            local elapsed=0
            while ps -p $pid > /dev/null 2>&1 && [ $elapsed -lt 10 ]; do
                sleep 1
                elapsed=$((elapsed + 1))
            done

            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "[WARN] Force stopping $name..."
                kill -9 $pid 2>/dev/null
            fi

            echo "[OK] $name stopped"
        else
            echo "[INFO] $name not running (stale PID)"
        fi
        rm -f "$pid_file"
    else
        echo "[INFO] $name: no PID file found"
    fi
}

# Stop services in reverse order
stop_service "dashboard"
stop_service "metrics"
stop_service "inference"

# Fallback: kill any remaining processes
echo
echo "[INFO] Checking for remaining processes..."
pkill -f "dash_app.py" 2>/dev/null && echo "[OK] Killed remaining dashboard processes"
pkill -f "tft_inference_daemon.py" 2>/dev/null && echo "[OK] Killed remaining inference processes"
pkill -f "metrics_generator_daemon.py" 2>/dev/null && echo "[OK] Killed remaining metrics processes"

echo
echo "============================================"
echo "✅ All NordIQ Services Stopped"
echo "============================================"
echo
echo "To restart: ./start_all.sh"
echo
