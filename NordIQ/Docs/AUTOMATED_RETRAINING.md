# Automated Retraining System

**NordIQ Inference Daemon - Version 2.3**

Complete automated retraining system with background training, hot model reload, and continuous learning.

---

## Overview

The inference daemon now features a **complete automated retraining pipeline**:

1. **Data Buffer** - Continuously accumulates incoming metrics
2. **Auto-Retrainer** - Triggers training jobs in background
3. **Background Training** - Non-blocking model training
4. **Hot Reload** - Automatic model reload after training
5. **Continuous Learning** - Incremental training on new data

**Use Case:** System runs 24/7, collecting real production data. Train new models weekly/monthly without downtime, automatically reload them, and continuously improve predictions.

---

## Architecture

```
Metrics → Inference Daemon → Data Buffer (60 days)
                ↓                    ↓
          Predictions         Auto-Retrainer
                                     ↓
                           Background Training Job
                                     ↓
                            Hot Reload New Model
                                     ↓
                           Predictions (improved!)
```

### Components:

**1. Data Buffer (`core/data_buffer.py`)**
- Accumulates all incoming metrics to daily parquet files
- Automatic file rotation at midnight
- Configurable retention (default: 60 days)
- Efficient parquet storage (~10 MB per 100K records)

**2. Auto-Retrainer (`core/auto_retrainer.py`)**
- Manages training jobs and status tracking
- Background thread execution (non-blocking)
- Automatic model reload after training
- Training history and statistics

**3. TFT Trainer (`training/tft_trainer.py`)**
- Incremental training support
- Profile-based transfer learning
- GPU acceleration
- Checkpoint resume

---

## API Endpoints

### 1. Trigger Training

**POST** `/admin/trigger-training`

Start a new training job in the background.

**Query Parameters:**
- `epochs` (int, default=5) - Number of epochs to train
- `incremental` (bool, default=true) - Resume from checkpoint
- `blocking` (bool, default=false) - Wait for completion

```bash
# Start 10-epoch training in background
curl -X POST -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/admin/trigger-training?epochs=10&incremental=true"
```

**Response (Success):**
```json
{
  "success": true,
  "job_id": "train_20250130_143022",
  "status": "queued",
  "message": "Training started in background",
  "epochs": 10,
  "incremental": true
}
```

**Response (Insufficient Data):**
```json
{
  "success": false,
  "error": "Insufficient data: 50000 < 100000 records",
  "status": "rejected"
}
```

---

### 2. Training Status

**GET** `/admin/training-status`

Get status of current or specific training job.

**Query Parameters:**
- `job_id` (string, optional) - Specific job ID. If omitted, returns current job.

```bash
# Get current job status
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/admin/training-status

# Get specific job status
curl -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/admin/training-status?job_id=train_20250130_143022"
```

**Response (Running):**
```json
{
  "job_id": "train_20250130_143022",
  "status": "running",
  "progress_pct": 65,
  "epochs": 10,
  "incremental": true,
  "started_at": "2025-01-30T14:30:22",
  "completed_at": null,
  "duration_seconds": 320,
  "model_path": null,
  "error": null
}
```

**Response (Completed):**
```json
{
  "job_id": "train_20250130_143022",
  "status": "completed",
  "progress_pct": 100,
  "epochs": 10,
  "incremental": true,
  "started_at": "2025-01-30T14:30:22",
  "completed_at": "2025-01-30T14:42:15",
  "duration_seconds": 713,
  "model_path": "models/tft_model_20250130_144215",
  "error": null
}
```

---

### 3. Training Statistics

**GET** `/admin/training-stats`

Get overall training statistics and system readiness.

```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/admin/training-stats
```

**Response:**
```json
{
  "current_job": {
    "job_id": "train_20250130_143022",
    "status": "running",
    "progress_pct": 45
  },
  "history": {
    "total_trainings": 12,
    "successful": 11,
    "failed": 1,
    "last_training": "2025-01-30T14:30:22"
  },
  "data_buffer": {
    "total_records": 2500000,
    "file_count": 45,
    "date_range": {
      "start": "2024-12-15",
      "end": "2025-01-30"
    },
    "disk_usage_mb": 245.3
  },
  "ready_to_train": true,
  "ready_reason": "Ready to train",
  "config": {
    "training_days": 30,
    "min_records_threshold": 100000
  }
}
```

---

### 4. Cancel Training

**POST** `/admin/cancel-training`

Cancel the currently running training job (soft cancel).

```bash
curl -X POST -H "X-API-Key": YOUR_KEY" \
  http://localhost:8000/admin/cancel-training
```

**Response:**
```json
{
  "success": true,
  "message": "Training job marked as cancelled",
  "job_id": "train_20250130_143022"
}
```

---

## Complete Workflow Example

### Manual Retraining

```bash
# 1. Check if system is ready to train
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats

# Response shows:
# - ready_to_train: true
# - total_records: 2500000 (enough data)
# - date_range: 45 days accumulated

# 2. Trigger training (10 epochs, incremental)
curl -X POST -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/admin/trigger-training?epochs=10&incremental=true"

# Response:
# {
#   "success": true,
#   "job_id": "train_20250130_143022",
#   "status": "queued"
# }

# 3. Monitor progress
while true; do
  STATUS=$(curl -s -H "X-API-Key: $API_KEY" \
    http://localhost:8000/admin/training-status)

  PROGRESS=$(echo $STATUS | jq -r '.progress_pct')
  STATUS_VAL=$(echo $STATUS | jq -r '.status')

  echo "Status: $STATUS_VAL | Progress: $PROGRESS%"

  if [ "$STATUS_VAL" != "running" ]; then
    break
  fi

  sleep 10
done

# 4. Check result
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-status

# If successful:
# - status: "completed"
# - model_path: "models/tft_model_20250130_144215"
# - Model automatically reloaded!

# 5. Verify new model is loaded
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/model-info
```

---

## Automated Scheduled Retraining

### Option 1: Cron Job (Linux)

```bash
#!/bin/bash
# weekly_retrain.sh - Run weekly retraining

API_KEY=$(cat /path/to/.nordiq_key)
DAEMON_URL="http://localhost:8000"

echo "[$(date)] Starting weekly retraining..."

# Check if ready
STATS=$(curl -s -H "X-API-Key: $API_KEY" $DAEMON_URL/admin/training-stats)
READY=$(echo $STATS | jq -r '.ready_to_train')

if [ "$READY" != "true" ]; then
    REASON=$(echo $STATS | jq -r '.ready_reason')
    echo "[$(date)] Not ready to train: $REASON"
    exit 1
fi

# Trigger training
RESPONSE=$(curl -s -X POST -H "X-API-Key: $API_KEY" \
  "$DAEMON_URL/admin/trigger-training?epochs=5&incremental=true")

SUCCESS=$(echo $RESPONSE | jq -r '.success')

if [ "$SUCCESS" == "true" ]; then
    JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
    echo "[$(date)] Training started: $JOB_ID"
else
    ERROR=$(echo $RESPONSE | jq -r '.error')
    echo "[$(date)] Training failed to start: $ERROR"
    exit 1
fi

echo "[$(date)] Weekly retraining complete!"
```

**Install cron:**
```cron
# Retrain every Sunday at 2 AM
0 2 * * 0 /path/to/weekly_retrain.sh >> /var/log/nordiq_retrain.log 2>&1
```

---

### Option 2: Python Scheduler

```python
# scheduled_retraining.py
import schedule
import time
import requests
from datetime import datetime

DAEMON_URL = "http://localhost:8000"
API_KEY = open('.nordiq_key').read().strip()

def weekly_retrain():
    """Run weekly retraining."""
    print(f"[{datetime.now()}] Starting weekly retraining...")

    headers = {"X-API-Key": API_KEY}

    # Check readiness
    response = requests.get(f"{DAEMON_URL}/admin/training-stats", headers=headers)
    stats = response.json()

    if not stats['ready_to_train']:
        print(f"Not ready: {stats['ready_reason']}")
        return

    # Trigger training
    response = requests.post(
        f"{DAEMON_URL}/admin/trigger-training",
        params={'epochs': 5, 'incremental': True},
        headers=headers
    )

    result = response.json()

    if result['success']:
        print(f"Training started: {result['job_id']}")
    else:
        print(f"Training failed: {result['error']}")

# Schedule every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(weekly_retrain)

print("Scheduler started. Press Ctrl+C to exit.")

while True:
    schedule.run_pending()
    time.sleep(60)
```

**Run as systemd service:**
```ini
[Unit]
Description=NordIQ Scheduled Retraining
After=network.target

[Service]
Type=simple
User=nordiq
WorkingDirectory=/opt/nordiq
ExecStart=/usr/bin/python3 scheduled_retraining.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Data Buffer Management

### Understanding the Data Buffer

The data buffer accumulates ALL incoming metrics into daily parquet files:

```
data_buffer/
├── metrics_2025-01-15.parquet  (45,000 records, 4.2 MB)
├── metrics_2025-01-16.parquet  (47,200 records, 4.5 MB)
├── metrics_2025-01-17.parquet  (46,800 records, 4.4 MB)
...
├── metrics_2025-01-29.parquet  (48,100 records, 4.6 MB)
└── metrics_2025-01-30.parquet  (12,300 records, 1.1 MB)  ← Today (incomplete)
```

**Default Configuration:**
- Retention: 60 days
- Auto-rotation: Midnight
- Flush interval: 10,000 records (~33 minutes at 5s polling)

**Storage Requirements:**
- ~50K records/day at 5s intervals
- ~5 MB/day compressed parquet
- ~300 MB for 60 days retention

---

### Training Window Selection

By default, training uses last 30 days:

```python
# In auto_retrainer.py
training_days = 30  # Use last 30 days of data
```

**Adjust for your needs:**
- **7 days**: Fast training, recent patterns only
- **30 days**: Balanced (default)
- **60 days**: Maximum historical context

---

## Training Parameters

### Epochs

Number of training iterations:

- **5 epochs**: Quick refresh (~10 minutes on GPU)
- **10 epochs**: Standard retraining (~20 minutes)
- **20 epochs**: Deep retraining (~40 minutes)

**Recommendation**: Use 5-10 epochs for weekly retraining.

### Incremental Training

**incremental=true** (default):
- Resumes from latest checkpoint
- Adds epochs to existing model
- Faster training (leverages previous learning)
- Continuous improvement

**incremental=false**:
- Trains from scratch
- Forgets previous training
- Useful if data distribution changed drastically

**Recommendation:** Always use `incremental=true` for continuous learning.

---

## Performance Impact

### During Training

**What happens:**
1. Data buffer exports training window to temp file (~5-10 seconds)
2. Training runs in background thread (~10-40 minutes)
3. Model automatically reloads (~5-10 seconds)

**System impact:**
- ✅ Predictions: Continue normally (uses current model)
- ✅ Data ingestion: Continue normally
- ✅ Dashboard: Continue normally
- ⚠️ CPU/GPU: Training uses resources (consider scheduling off-peak)

### After Training

**Immediate:**
- New model loaded automatically
- Predictions use new model instantly
- No warmup needed (rolling window preserved)

**Benefits:**
- Improved accuracy with recent data
- Adapts to changing patterns
- Continuous learning from production

---

## Monitoring Training

### Poll for Progress

```python
import requests
import time

API_KEY = "your-key"
headers = {"X-API-Key": API_KEY}

# Trigger training
response = requests.post(
    "http://localhost:8000/admin/trigger-training",
    params={'epochs': 10},
    headers=headers
)

job_id = response.json()['job_id']

# Poll until complete
while True:
    response = requests.get(
        f"http://localhost:8000/admin/training-status?job_id={job_id}",
        headers=headers
    )

    status = response.json()

    print(f"Status: {status['status']} | Progress: {status['progress_pct']}%")

    if status['status'] in ['completed', 'failed', 'cancelled']:
        break

    time.sleep(10)

# Check result
if status['status'] == 'completed':
    print(f"✅ Training successful!")
    print(f"   Model: {status['model_path']}")
    print(f"   Duration: {status['duration_seconds']}s")
else:
    print(f"❌ Training failed: {status['error']}")
```

---

## Troubleshooting

### Issue: "Insufficient data"

**Cause:** Not enough accumulated data.

**Solution:**
```bash
# Check data buffer status
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats

# Look at data_buffer section:
# - total_records: Must be >= 100,000
# - file_count: Must be >= 30 days

# Wait for more data to accumulate, or reduce threshold in daemon initialization
```

### Issue: Training fails immediately

**Cause:** Training data export or model training error.

**Solution:**
```bash
# Check daemon logs
tail -f logs/inference_daemon.log

# Look for errors in:
# - Data export step
# - Model training step
# - Model reload step
```

### Issue: Training hangs at 40%

**Cause:** Training is CPU/GPU intensive and takes time.

**Solution:** This is normal. Training 10 epochs on 30 days of data takes 20-40 minutes. Monitor logs to ensure progress.

### Issue: Model not reloaded after training

**Cause:** Reload callback failed.

**Solution:**
```bash
# Check training status for error
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-status

# If model trained but not reloaded, manually reload
curl -X POST -H "X-API-Key: $API_KEY" \
  http://localhost:8000/admin/reload-model
```

---

## Best Practices

### 1. Weekly Retraining Schedule

Train every week with 5-10 epochs:

```bash
# Every Sunday at 2 AM
0 2 * * 0 /path/to/weekly_retrain.sh
```

**Benefits:**
- Model stays current with recent patterns
- Adapts to seasonal changes
- Low computational overhead

### 2. Monitor Data Buffer

Check regularly that data is accumulating:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats
```

Watch for:
- `total_records` steadily increasing
- `file_count` growing daily
- `disk_usage_mb` within reasonable limits

### 3. Keep Training History

Training jobs are logged. Review periodically:

```json
{
  "history": {
    "total_trainings": 12,
    "successful": 11,
    "failed": 1,
    "last_training": "2025-01-30T14:30:22"
  }
}
```

If failures > 10%, investigate root cause.

### 4. Off-Peak Training

Schedule training during low-traffic periods:
- Weekends
- Early morning hours (2-4 AM)
- Maintenance windows

Reduces impact on production predictions.

---

## Summary: Complete Continuous Learning Pipeline

**What You Have Now:**

1. ✅ **Data Collection** - All metrics automatically buffered
2. ✅ **Automated Training** - Background training via API
3. ✅ **Hot Reload** - New models load without restart
4. ✅ **Continuous Learning** - Incremental training improves model
5. ✅ **Progress Monitoring** - Real-time status tracking
6. ✅ **Zero Downtime** - Predictions continue during training

**Complete Workflow:**

```
Day 1-30: Accumulate data → 1.5M records
Day 30: Trigger training → 20 minutes
Day 30: Auto-reload model → 5 seconds
Day 31-60: Better predictions with recent data
Day 60: Retrain again → Continuous improvement
```

**Production-Ready:**
- Schedule weekly/monthly retraining
- Monitor via API endpoints
- Automatic model updates
- No manual intervention needed

---

**Version:** 2.3.0
**Features:** Automated Retraining + Hot Reload + Continuous Learning
**Daemon Version Required:** >=2.3
**API Compatibility:** Fully backward compatible

© 2025 NordIQ AI, LLC. All rights reserved.
