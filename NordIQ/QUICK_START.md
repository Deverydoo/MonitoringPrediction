# NordIQ Quick Start - 5 Commands to Production

Get NordIQ running in production with just 5 commands.

---

## The 5 Commands

```bash
# 1. Clone and navigate
cd MonitoringPrediction/NordIQ

# 2. Generate API key
python bin/generate_api_key.py

# 3. Start all services
./start_all.sh

# 4. Test the system (after 1 hour)
curl -H "X-API-Key: $(cat .nordiq_key)" http://localhost:8000/admin/training-stats

# 5. Setup weekly retraining
crontab -e
# Add: 0 2 * * 0 ~/MonitoringPrediction/NordIQ/bin/weekly_retrain.sh >> ~/retrain.log 2>&1
```

**That's it!** System is now running 24/7.

---

## What Each Command Does

### 1. Navigate to NordIQ Directory
```bash
cd MonitoringPrediction/NordIQ
```
- Gets you into the main NordIQ folder
- All commands run from here

### 2. Generate API Key
```bash
python bin/generate_api_key.py
```
- Creates `.nordiq_key` file
- Generates secure authentication token
- Only needed once

**Output:**
```
Generated new API key: tft-a3b9c8d7e2f1g4h5i6j7k8l9
API key saved to: .nordiq_key
```

### 3. Start All Services
```bash
./start_all.sh
```
- Starts inference daemon (port 8000)
- Starts metrics generator (port 8001)
- Starts dashboard (port 8050)
- All run in background (daemon mode)

**Output:**
```
🚀 Starting NordIQ Services...
  ✓ Inference Daemon ready on http://localhost:8000
  ✓ Metrics Generator ready on http://localhost:8001
  ✓ Dashboard ready on http://localhost:8050

✅ All services running!
```

**Access Dashboard:** http://localhost:8050

### 4. Test System Status
```bash
curl -H "X-API-Key: $(cat .nordiq_key)" http://localhost:8000/admin/training-stats
```
- Checks how much data collected
- Shows if ready to train
- Confirms everything working

**Example Output:**
```json
{
  "data_buffer": {
    "total_records": 1523,
    "file_count": 1,
    "disk_usage_mb": 0.3
  },
  "ready_to_train": false,
  "ready_reason": "Not enough records (need 100,000, have 1,523)"
}
```

**Wait until `ready_to_train: true` before training!**

### 5. Setup Automated Retraining
```bash
crontab -e
```
Add this line:
```
0 2 * * 0 ~/MonitoringPrediction/NordIQ/bin/weekly_retrain.sh >> ~/retrain.log 2>&1
```

- Runs every Sunday at 2 AM
- Automatically retrains model
- Hot-reloads without downtime
- Logs to `~/retrain.log`

---

## Verification Checklist

After running the 5 commands, verify:

```bash
# ✅ Services running?
./status.sh

# ✅ Dashboard accessible?
curl http://localhost:8050

# ✅ API working?
curl -H "X-API-Key: $(cat .nordiq_key)" http://localhost:8000/health

# ✅ Data collecting?
curl -H "X-API-Key: $(cat .nordiq_key)" http://localhost:8000/admin/training-stats

# ✅ Cron job installed?
crontab -l | grep weekly_retrain
```

If all return successful responses, **you're done!**

---

## Timeline

```
Minute 0:   Run start_all.sh
            └─> Services start

Minute 1-30: Rolling window warmup
            └─> Dashboard shows predictions

Hour 1-168:  Data collection (1 week)
            └─> Buffer fills with quality data

Week 1:      First automated retraining
            └─> Sunday 2 AM, cron triggers
            └─> Model trains (~20 min)
            └─> Hot reload (~5 sec)
            └─> Improved predictions!

Week 2+:     Weekly retraining continues
            └─> Model improves continuously
```

---

## Daily Operations

### View System Status
```bash
./status.sh
```

### View Logs
```bash
tail -f logs/inference_daemon.log
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

## Manual Training (Optional)

Want to manually trigger training?

```bash
# Check if ready
API_KEY=$(cat .nordiq_key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-stats

# If ready_to_train is true, trigger training
curl -X POST -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/admin/trigger-training?epochs=5&incremental=true"

# Monitor progress
curl -H "X-API-Key: $API_KEY" http://localhost:8000/admin/training-status
```

---

## Troubleshooting

### Services won't start?
```bash
# Check if ports are in use
netstat -tuln | grep -E "8000|8001|8050"

# Check logs
cat logs/inference_daemon.log
```

### Dashboard shows no data?
- Wait 20-30 minutes for warmup
- Check services: `./status.sh`

### Training fails?
```bash
# Need at least 100,000 records
curl -H "X-API-Key: $(cat .nordiq_key)" \
  http://localhost:8000/admin/training-stats
```

---

## What Happens Automatically

Once you complete the 5 commands:

**Every 5 seconds:**
- ✅ Generate metrics
- ✅ Make predictions
- ✅ Store to buffer
- ✅ Update dashboard

**Every Sunday at 2 AM (after cron installed):**
- ✅ Check if enough data
- ✅ Train new model (if ready)
- ✅ Hot-reload model
- ✅ Log everything

**Zero manual intervention needed!**

---

## Advanced: Custom Schedule

Want different training schedule? Edit cron:

```bash
# Daily at 2 AM
0 2 * * * /path/to/weekly_retrain.sh

# Twice a week (Sunday and Wednesday)
0 2 * * 0,3 /path/to/weekly_retrain.sh

# Monthly (1st of month)
0 2 1 * * /path/to/weekly_retrain.sh

# Every 3 days at midnight
0 0 */3 * * /path/to/weekly_retrain.sh
```

---

## Full Documentation

For complete details, see:
- [GETTING_STARTED.md](GETTING_STARTED.md) - Detailed setup guide
- [README.md](README.md) - Project overview
- [Docs/AUTOMATED_RETRAINING.md](Docs/AUTOMATED_RETRAINING.md) - Training details
- [Docs/HOT_MODEL_RELOAD.md](Docs/HOT_MODEL_RELOAD.md) - Model reload guide

---

## Summary

```bash
# Setup (once)
python bin/generate_api_key.py
./start_all.sh
crontab -e  # Add weekly_retrain.sh

# Daily operations
./status.sh       # Check health
./stop_all.sh     # Stop services
./start_all.sh    # Start services

# That's it!
```

**Production-ready in 5 commands. Zero ongoing maintenance.**

---

Built by Craig Giannelli and Claude Code
