#!/bin/bash
# weekly_retrain.sh - Automated weekly model retraining
# Purpose: Trigger model retraining with accumulated production data
# Schedule: Run weekly via cron (recommended: Sunday 2 AM)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORDIQ_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$NORDIQ_ROOT/logs/retrain.log"
API_KEY_FILE="$NORDIQ_ROOT/.nordiq_key"
DAEMON_URL="http://localhost:8000"
EPOCHS=5  # Number of training epochs
INCREMENTAL=true  # Use incremental training

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if API key exists
if [ ! -f "$API_KEY_FILE" ]; then
    log "ERROR: API key file not found at $API_KEY_FILE"
    exit 1
fi

API_KEY=$(cat "$API_KEY_FILE")

log "=== Starting weekly retraining process ==="

# Step 1: Check if system is ready to train
log "Checking training readiness..."
STATS=$(curl -s -H "X-API-Key: $API_KEY" "$DAEMON_URL/admin/training-stats")

if [ $? -ne 0 ]; then
    log "ERROR: Failed to connect to inference daemon at $DAEMON_URL"
    exit 1
fi

READY=$(echo "$STATS" | grep -o '"ready_to_train":[^,]*' | cut -d':' -f2 | tr -d ' ')

if [ "$READY" != "true" ]; then
    REASON=$(echo "$STATS" | grep -o '"ready_reason":"[^"]*"' | cut -d'"' -f4)
    log "Not ready to train: $REASON"

    # Log data buffer stats
    TOTAL_RECORDS=$(echo "$STATS" | grep -o '"total_records":[^,]*' | cut -d':' -f2 | tr -d ' ')
    log "Current data buffer: $TOTAL_RECORDS records"

    exit 0  # Not an error, just not ready yet
fi

log "System ready to train!"

# Log data buffer statistics
TOTAL_RECORDS=$(echo "$STATS" | grep -o '"total_records":[^,]*' | cut -d':' -f2 | tr -d ' ')
FILE_COUNT=$(echo "$STATS" | grep -o '"file_count":[^,]*' | cut -d':' -f2 | tr -d ' ')
DISK_USAGE=$(echo "$STATS" | grep -o '"disk_usage_mb":[^,]*' | cut -d':' -f2 | tr -d ' ')

log "Data buffer stats:"
log "  - Total records: $TOTAL_RECORDS"
log "  - File count: $FILE_COUNT"
log "  - Disk usage: ${DISK_USAGE} MB"

# Step 2: Trigger training
log "Triggering training (epochs=$EPOCHS, incremental=$INCREMENTAL)..."
RESPONSE=$(curl -s -X POST -H "X-API-Key: $API_KEY" \
    "$DAEMON_URL/admin/trigger-training?epochs=$EPOCHS&incremental=$INCREMENTAL")

SUCCESS=$(echo "$RESPONSE" | grep -o '"success":[^,]*' | cut -d':' -f2 | tr -d ' ')

if [ "$SUCCESS" != "true" ]; then
    ERROR=$(echo "$RESPONSE" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
    log "ERROR: Training failed to start: $ERROR"
    exit 1
fi

JOB_ID=$(echo "$RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
log "Training started successfully! Job ID: $JOB_ID"

# Step 3: Monitor progress
log "Monitoring training progress..."
LAST_PROGRESS=0

while true; do
    sleep 30  # Check every 30 seconds

    STATUS_RESPONSE=$(curl -s -H "X-API-Key: $API_KEY" \
        "$DAEMON_URL/admin/training-status?job_id=$JOB_ID")

    STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    PROGRESS=$(echo "$STATUS_RESPONSE" | grep -o '"progress_pct":[^,]*' | cut -d':' -f2 | tr -d ' ')

    # Log progress updates (only when changed)
    if [ "$PROGRESS" != "$LAST_PROGRESS" ]; then
        log "Training progress: $PROGRESS% (status: $STATUS)"
        LAST_PROGRESS=$PROGRESS
    fi

    # Check if completed
    if [ "$STATUS" = "completed" ]; then
        MODEL_PATH=$(echo "$STATUS_RESPONSE" | grep -o '"model_path":"[^"]*"' | cut -d'"' -f4)
        DURATION=$(echo "$STATUS_RESPONSE" | grep -o '"duration_seconds":[^,]*' | cut -d':' -f2 | tr -d ' ')

        log "Training completed successfully!"
        log "  - Model path: $MODEL_PATH"
        log "  - Duration: ${DURATION}s ($(($DURATION / 60))m)"
        log "  - Model automatically reloaded"
        break
    fi

    # Check if failed
    if [ "$STATUS" = "failed" ]; then
        ERROR=$(echo "$STATUS_RESPONSE" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
        log "ERROR: Training failed: $ERROR"
        exit 1
    fi

    # Check if cancelled
    if [ "$STATUS" = "cancelled" ]; then
        log "Training was cancelled"
        exit 1
    fi
done

# Step 4: Verify model was reloaded
log "Verifying new model..."
MODEL_INFO=$(curl -s -H "X-API-Key: $API_KEY" "$DAEMON_URL/admin/model-info")
LOADED_MODEL=$(echo "$MODEL_INFO" | grep -o '"model_path":"[^"]*"' | cut -d'"' -f4)

log "Current loaded model: $LOADED_MODEL"

log "=== Weekly retraining completed successfully ==="
exit 0
