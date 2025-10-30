# Getting Started with NordIQ - Quick Setup Guide

This guide walks you through setting up NordIQ from zero to production in 15 minutes.

---

## Prerequisites

- Linux server with SSH access (Putty/terminal)
- Python 3.8+ environment with conda or venv
- Required packages installed (see [REQUIREMENTS.md](REQUIREMENTS.md))

---

## Step 1: Initial Setup (5 minutes)

### Clone and Navigate
```bash
git clone https://github.com/yourusername/MonitoringPrediction.git
cd MonitoringPrediction/NordIQ
```

### Activate Your Python Environment
```bash
# Option A: Conda
conda activate nordiq

# Option B: venv
source venv/bin/activate
```

### Install Required Packages
```bash
# For full system (inference + dashboard)
pip install -r requirements.txt

# OR for minimal inference-only
pip install -r requirements_inference.txt
```

### Generate API Key
```bash
python bin/generate_api_key.py

# Verify key was created
cat .nordiq_key
# Should show: tft-xxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Step 2: Start All Services (2 minutes)

### Start Everything
```bash
./start_all.sh
```

You should see:
```
🚀 Starting NordIQ Services...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting Inference Daemon...
  ✓ Inference Daemon started (PID: 12345)
  ✓ Inference Daemon ready on http://localhost:8000

Starting Metrics Generator...
  ✓ Metrics Generator started (PID: 12346)
  ✓ Metrics Generator ready on http://localhost:8001

Starting Dashboard...
  ✓ Dashboard started (PID: 12347)
  ✓ Dashboard ready on http://localhost:8050

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All services running! Dashboard: http://localhost:8050
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Check Status
```bash
./status.sh
```

---

## Step 3: Verify Everything Works (3 minutes)

### Load Your API Key
```bash
API_KEY=$(cat .nordiq_key)
```

### Test Inference Daemon
```bash
# Health check
curl -H "X-API-Key: $API_KEY" http://localhost:8000/health

# Should return: {"status": "healthy"}

# Get current model info
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/model-info
```

### Test Dashboard
```bash
# Open browser to: http://localhost:8050
# Or use curl:
curl http://localhost:8050
```

### Check Data Buffer
```bash
# See how much data has been collected
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats
```

Expected response:
```json
{
  "data_buffer": {
    "total_records": 1523,
    "file_count": 1,
    "date_range": {
      "start": "2025-10-30",
      "end": "2025-10-30"
    },
    "disk_usage_mb": 0.3
  },
  "ready_to_train": false,
  "ready_reason": "Not enough records (need 100,000, have 1,523)"
}
```

---

## Step 4: Wait for Data Collection (Variable Time)

**The system is now running!** It will:
- ✅ Collect metrics every 5 seconds
- ✅ Make predictions on incoming data
- ✅ Store all data to buffer for retraining
- ✅ Serve dashboard at http://localhost:8050

**Wait time depends on your setup:**
- **Demo/test system**: ~1 hour to collect 100K records
- **Production (100 servers)**: ~3-7 days for quality training data
- **High-volume (1000+ servers)**: ~12-24 hours

You can check progress anytime:
```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats | grep total_records
```

---

## Step 5: First Manual Retraining (Test) (20 minutes)

Once you have enough data (`ready_to_train: true`):

### Trigger Training
```bash
# Start background training (5 epochs, ~10 minutes)
curl -X POST -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/admin/trigger-training?epochs=5&incremental=true"

# Response:
# {
#   "success": true,
#   "job_id": "train_20251030_143022",
#   "status": "queued",
#   "message": "Training started in background"
# }
```

### Monitor Progress
```bash
# Check status (repeat every minute)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-status

# You'll see progress_pct increase: 0% -> 20% -> 45% -> 90% -> 100%
```

### Verify New Model
```bash
# After training completes, verify model was reloaded
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/model-info

# Check loaded_at timestamp - should be recent
```

**Important**: The dashboard continues serving predictions during training! No downtime.

---

## Step 6: Setup Automated Weekly Retraining (5 minutes)

### Test the Script First
```bash
# Dry run (won't train if not ready)
./bin/weekly_retrain.sh

# Check the log
tail logs/retrain.log
```

### Install Cron Job (Linux)
```bash
# Edit crontab
crontab -e

# Add this line (runs every Sunday at 2 AM):
0 2 * * 0 /opt/nordiq/bin/weekly_retrain.sh >> /var/log/nordiq_retrain.log 2>&1

# Or if running from home directory:
0 2 * * 0 ~/MonitoringPrediction/NordIQ/bin/weekly_retrain.sh >> ~/nordiq_retrain.log 2>&1

# Save and exit

# Verify cron job was installed
crontab -l
```

### Monitor Automated Retraining
```bash
# Check cron logs
tail -f /var/log/nordiq_retrain.log

# Or check NordIQ's retrain log
tail -f logs/retrain.log
```

---

## You're Done! 🎉

Your system is now:
- ✅ Running 24/7 collecting data
- ✅ Making predictions in real-time
- ✅ Automatically retraining weekly
- ✅ Hot-reloading new models without downtime

---

## Daily Operations

### Check System Health
```bash
./status.sh
```

### View Logs
```bash
# Inference daemon
tail -f logs/inference_daemon.log

# Metrics generator
tail -f logs/metrics_daemon.log

# Dashboard
tail -f logs/dashboard.log

# Automated retraining
tail -f logs/retrain.log
```

### Stop Services
```bash
./stop_all.sh
```

### Restart Services
```bash
./stop_all.sh && ./start_all.sh
```

---

## Troubleshooting

### Services Won't Start

**Check if ports are in use:**
```bash
netstat -tuln | grep -E "8000|8001|8050"
```

**Check logs for errors:**
```bash
cat logs/inference_daemon.log
cat logs/metrics_daemon.log
cat logs/dashboard.log
```

**Verify API key exists:**
```bash
ls -la .nordiq_key
cat .nordiq_key  # Should show: tft-xxxxx...
```

### Training Fails

**Check data buffer has enough records:**
```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats
# Need at least 100,000 records
```

**Check training logs:**
```bash
tail -f logs/inference_daemon.log | grep -i training
```

**Check disk space:**
```bash
df -h .
```

### Dashboard Shows No Data

**Wait for warmup period:**
- Rolling window needs 20-30 minutes to fill
- Check metrics daemon is running: `./status.sh`
- Verify metrics are being generated: `curl http://localhost:8001/health`

**Check dashboard logs:**
```bash
tail -f logs/dashboard.log
```

### Hot Reload Fails

**Check available models:**
```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/models
```

**Verify model directory:**
```bash
ls -lh models/
```

**Check GPU memory (if using GPU):**
```bash
nvidia-smi
```

---

## Advanced Configuration

### Change Refresh Interval
Edit `tft_dashboard_web.py`:
```python
# Around line 25
DASHBOARD_REFRESH_SECONDS = 60  # Change to 30, 60, 120, etc.
```

### Change Training Schedule
Edit crontab:
```bash
crontab -e

# Daily at 2 AM:
0 2 * * * /path/to/weekly_retrain.sh

# Twice a week (Sunday and Wednesday):
0 2 * * 0,3 /path/to/weekly_retrain.sh

# Monthly (1st of month):
0 2 1 * * /path/to/weekly_retrain.sh
```

### Change Data Buffer Retention
Edit `tft_inference_daemon.py`:
```python
# Around line 50
self.data_buffer = DataBuffer(
    buffer_dir="data_buffer",
    retention_days=60  # Change to 30, 90, etc.
)
```

### Change Training Window
Edit `tft_inference_daemon.py`:
```python
# Around line 60
self.auto_retrainer = AutoRetrainer(
    data_buffer=self.data_buffer,
    reload_callback=self.reload_model,
    training_days=30,  # Change to 7, 14, 60, etc.
    min_records_threshold=100000
)
```

---

## Production Best Practices

### 1. Run on Dedicated Server
- Separate from application servers
- Dedicated CPU/GPU resources
- Adequate disk space for data buffer (10+ GB)

### 2. Monitor System Resources
```bash
# Check CPU/Memory usage
./status.sh

# Monitor disk usage
df -h data_buffer/
```

### 3. Backup Models
```bash
# Weekly backup of models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

### 4. Log Rotation
```bash
# Install logrotate config
sudo tee /etc/logrotate.d/nordiq << EOF
/opt/nordiq/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

### 5. Security
```bash
# Restrict API key permissions
chmod 600 .nordiq_key

# Firewall rules (adjust ports as needed)
sudo ufw allow 8050/tcp  # Dashboard (if remote access)
sudo ufw deny 8000/tcp   # Inference (internal only)
sudo ufw deny 8001/tcp   # Metrics (internal only)
```

---

## What Happens Automatically

### Inference Daemon
- ✅ Loads TFT model on startup
- ✅ Warms up rolling window (20-30 minutes)
- ✅ Receives metrics from generator
- ✅ Makes predictions
- ✅ Stores to data buffer
- ✅ Serves API endpoints

### Metrics Generator
- ✅ Generates synthetic metrics every 5 seconds
- ✅ Sends to inference daemon
- ✅ Rotates through server profiles

### Dashboard
- ✅ Fetches data from inference daemon
- ✅ Displays predictions, alerts, risks
- ✅ Auto-refreshes based on config

### Data Buffer
- ✅ Accumulates all metrics
- ✅ Stores in daily parquet files
- ✅ Manages retention (60 days default)
- ✅ Provides data for retraining

### Automated Retraining (Once Cron Job Installed)
- ✅ Checks readiness weekly
- ✅ Triggers training if enough data
- ✅ Monitors progress
- ✅ Hot-reloads new model
- ✅ Logs everything

---

## What Requires Manual Setup

1. **Initial API Key Generation** - Once: `python bin/generate_api_key.py`
2. **Starting Services** - Each deployment: `./start_all.sh`
3. **Installing Cron Job** - Once: Add to crontab
4. **First Manual Test Training** - Once: Test before automation

That's it! Everything else is automatic.

---

## Timeline Summary

| Time | What Happens |
|------|-------------|
| 0 min | Run start_all.sh - services start |
| 1-30 min | Rolling window warmup |
| 1-7 days | Data collection (depends on volume) |
| First training | Manual test training (~20 min) |
| Weekly | Automated retraining (cron job) |
| Forever | System runs continuously, improves weekly |

---

## Getting Help

### Documentation
- [README.md](README.md) - Project overview
- [REQUIREMENTS.md](REQUIREMENTS.md) - Installation details
- [Docs/HOT_MODEL_RELOAD.md](Docs/HOT_MODEL_RELOAD.md) - Model reload guide
- [Docs/AUTOMATED_RETRAINING.md](Docs/AUTOMATED_RETRAINING.md) - Training guide

### Logs
- `logs/inference_daemon.log` - Inference engine logs
- `logs/metrics_daemon.log` - Metrics generator logs
- `logs/dashboard.log` - Dashboard logs
- `logs/retrain.log` - Automated retraining logs

### API Endpoints
```bash
# Load API key first
API_KEY=$(cat .nordiq_key)

# Health checks
curl -H "X-API-Key: $API_KEY" http://localhost:8000/health
curl http://localhost:8001/health

# System status
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/model-info

# Training
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-status
```

---

© 2025 NordIQ AI, LLC. All rights reserved.
