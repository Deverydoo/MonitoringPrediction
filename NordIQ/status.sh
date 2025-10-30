#!/bin/bash
# NordIQ AI Systems - Status Check Script
# Copyright (c) 2025 NordIQ AI, LLC.
# Nordic precision, AI intelligence

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/logs/pids"
LOG_DIR="$SCRIPT_DIR/logs"

# Ports
INFERENCE_PORT=8000
METRICS_PORT=8001
DASHBOARD_PORT=8050

echo "============================================"
echo "NordIQ AI Systems - Service Status"
echo "============================================"
echo

cd "$SCRIPT_DIR"

# Function to check if port is listening
check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -i ":$port" -t >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln | grep ":$port " >/dev/null 2>&1
    else
        timeout 1 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null
    fi
}

# Function to check service status
check_service() {
    local name=$1
    local port=$2
    local pid_file="$PID_DIR/${name}.pid"
    local log_file="$LOG_DIR/${name}_daemon.log"

    if [ "$name" == "dashboard" ]; then
        log_file="$LOG_DIR/dashboard.log"
    fi

    echo "[$name]"

    # Check PID file
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Status:   ✅ RUNNING"
            echo "  PID:      $pid"
            echo "  Port:     $port"

            # Check if port is actually listening
            if check_port $port; then
                echo "  Endpoint: http://localhost:$port"
            else
                echo "  Warning:  Port $port not listening"
            fi

            # Show process info
            if command -v ps >/dev/null 2>&1; then
                local cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
                local mem=$(ps -p $pid -o %mem= 2>/dev/null | tr -d ' ')
                local time=$(ps -p $pid -o etime= 2>/dev/null | tr -d ' ')
                [ -n "$cpu" ] && echo "  CPU:      ${cpu}%"
                [ -n "$mem" ] && echo "  Memory:   ${mem}%"
                [ -n "$time" ] && echo "  Uptime:   $time"
            fi
        else
            echo "  Status:   ❌ STOPPED (stale PID)"
            echo "  PID:      $pid (not running)"
        fi
    else
        echo "  Status:   ❌ STOPPED (no PID file)"

        # Check if process is running anyway
        if check_port $port; then
            echo "  Warning:  Port $port is in use by another process"
        fi
    fi

    # Show log file info
    if [ -f "$log_file" ]; then
        local log_size=$(du -h "$log_file" 2>/dev/null | cut -f1)
        local log_lines=$(wc -l < "$log_file" 2>/dev/null)
        echo "  Log:      $log_file"
        echo "  Log Size: $log_size ($log_lines lines)"

        # Show last error if any
        if grep -i "error\|exception\|failed" "$log_file" | tail -1 > /dev/null 2>&1; then
            local last_error=$(grep -i "error\|exception\|failed" "$log_file" | tail -1)
            echo "  Last Err: ${last_error:0:60}..."
        fi
    fi

    echo
}

# Check all services
check_service "inference" $INFERENCE_PORT
check_service "metrics" $METRICS_PORT
check_service "dashboard" $DASHBOARD_PORT

# Summary
echo "============================================"
echo "Summary"
echo "============================================"
echo

running=0
stopped=0

for service in inference metrics dashboard; do
    pid_file="$PID_DIR/${service}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            running=$((running + 1))
        else
            stopped=$((stopped + 1))
        fi
    else
        stopped=$((stopped + 1))
    fi
done

if [ $running -eq 3 ]; then
    echo "✅ All services running ($running/3)"
elif [ $running -eq 0 ]; then
    echo "❌ All services stopped (0/3)"
    echo "   Run: ./start_all.sh to start"
else
    echo "⚠️  Partial operation ($running/3 running)"
    echo "   Run: ./start_all.sh to restart all"
fi

echo
echo "Commands:"
echo "  Start:   ./start_all.sh"
echo "  Stop:    ./stop_all.sh"
echo "  Logs:    tail -f $LOG_DIR/<service>.log"
echo
