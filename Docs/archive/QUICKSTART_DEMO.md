# 🚀 Quick Start: TFT Monitoring Dashboard Demo

**Complete end-to-end demo** of the TFT monitoring system with live streaming data.

This guide shows how to:
1. Generate training data with LINBORG metrics
2. Train the TFT model
3. Run the inference daemon
4. Stream live data to the daemon
5. View predictions in the interactive dashboard

---

## Prerequisites

- Python 3.10+ with conda environment activated
- GPU (recommended): NVIDIA RTX 4090 or similar
- ~10GB disk space for training data
- ~4-6 hours for complete training (20 epochs)

---

## Step 1: Generate Training Data (5-10 minutes)

Generate 30 days of realistic server metrics with LINBORG-compatible format:

```bash
python metrics_generator.py \
    --hours 720 \
    --num_ml_compute 20 \
    --num_database 15 \
    --num_web_api 25 \
    --num_conductor_mgmt 5 \
    --num_data_ingest 10 \
    --num_risk_analytics 8 \
    --num_generic 7 \
    --out_dir ./training \
    --format parquet
```

**Expected Output**:
```
✅ Production dataset generated!
📊 Latest file: server_metrics.parquet (300.7 MB)
📈 Dataset Summary:
   Total records: 4,838,400
   Servers: 90
   Time span: 29 days
   LINBORG Metrics: 14/14 present
```

**Verify LINBORG Metrics**:
The dataset should include these 14 production metrics:
- cpu_user_pct, cpu_sys_pct, **cpu_iowait_pct** (CRITICAL), cpu_idle_pct, java_cpu_pct
- mem_used_pct, swap_used_pct, disk_usage_pct
- net_in_mb_s, net_out_mb_s
- back_close_wait, front_close_wait
- load_average, uptime_days

---

## Step 2: Train TFT Model (4-6 hours for 20 epochs)

Train the model with profile-based transfer learning:

```bash
python tft_trainer.py \
    --data_path ./training/server_metrics.parquet \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --out_dir ./models
```

**Shorter Training Options**:
```bash
# Quick validation (1 hour)
python tft_trainer.py --data_path ./training/server_metrics.parquet --epochs 5

# Overnight training (2-3 hours) - good for demos
python tft_trainer.py --data_path ./training/server_metrics.parquet --epochs 10
```

**Expected Output**:
```
[TRANSFER] Profile feature enabled - model will learn per-profile patterns
[TRANSFER] Profiles detected: ['ml_compute', 'database', 'web_api', ...]
[INFO] Epoch 20/20 completed
   Train Loss: 2.45 | Val Loss: 4.12 [BEST]

✅ TRAINING COMPLETED SUCCESSFULLY!
📁 Model saved: ./models/tft_model_20251013_100205
```

**Target Metrics** (20 epochs, 720 hours data):
- Train Loss: 2-4
- Val Loss: 4-6
- Expected Accuracy: 85-90% (see MODEL_TRAINING_GUIDELINES.md)

---

## Step 3: Start Inference Daemon (Terminal 1)

The daemon loads the trained model and exposes a REST API for predictions:

```bash
python tft_inference_daemon.py
```

**Expected Output**:
```
[OK] Found model: models/tft_model_20251013_100205
[INFO] Loading TFT model from: models/tft_model_20251013_100205
[OK] Server mapping loaded: 90 servers
[OK] Model loaded successfully

🚀 TFT Inference Daemon
==================================================
📊 Model: tft_model_20251013_100205
🔮 Predictions: 96 steps (8 hours ahead)
🧠 Transfer Learning: ENABLED (7 profiles)
🔥 Device: cuda
💾 Data Window: 288 steps (24 hours)
==================================================

🌐 Daemon starting on http://0.0.0.0:8000
   Endpoints:
     • POST /feed/data - Receive streaming metrics
     • GET  /predict - Get latest predictions
     • GET  /health - Health check
     • GET  /stats - System statistics
```

**Verify Daemon is Running**:
```bash
# In another terminal
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true, "data_points": 0}
```

---

## Step 4: Stream Live Data to Daemon (Terminal 2)

Send real-time server metrics to the inference daemon:

```bash
python metrics_generator.py --stream --daemon-url http://localhost:8000
```

**Stream Mode Options**:

### Healthy Scenario (Default)
```bash
python metrics_generator.py --stream --scenario healthy
```
- Normal operation, low risk scores
- Good for baseline testing

### Degrading Scenario
```bash
python metrics_generator.py --stream --scenario degrading
```
- Gradual performance degradation
- Risk scores increase over time
- Demonstrates early warning capability

### Critical Scenario
```bash
python metrics_generator.py --stream --scenario critical
```
- Multiple servers in distress
- High risk scores, alerts triggered
- Shows incident detection

**Expected Output**:
```
🌊 STREAM MODE: Continuous data feed to inference daemon
==================================================
Scenario: healthy
Daemon URL: http://localhost:8000
Interval: 5 seconds
Servers: 90

Press Ctrl+C to stop streaming
==================================================

[09:42:15] Tick    1 | 🟢 HEALTHY | 90 servers | Elapsed: 5s
[09:42:20] Tick    2 | 🟢 HEALTHY | 90 servers | Elapsed: 10s
[09:42:25] Tick    3 | 🟢 HEALTHY | 90 servers | Elapsed: 15s
...
```

**What's Happening**:
1. Generates realistic LINBORG metrics every 5 seconds
2. POSTs data to daemon at `/feed/data`
3. Daemon accumulates 24-hour rolling window (288 steps)
4. Once window fills, daemon generates predictions
5. Dashboard can now display real-time predictions

---

## Step 5: Launch Dashboard (Terminal 3)

Open the interactive Streamlit dashboard:

```bash
streamlit run tft_dashboard_web.py
```

**Expected Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

**Open browser**: Navigate to `http://localhost:8501`

---

## Dashboard Tour

### 📊 Overview Tab
**Fleet-wide health status**:
- Risk Score Gauge (0-100)
- Environment incident probability (30min, 8h)
- Alert table showing servers at risk
- Real-time updates every 5 seconds

**Key Metrics Displayed**:
- **CPU Used %** (100 - idle) - Total CPU utilization
- **I/O Wait %** (CRITICAL) - Disk/storage bottleneck indicator
- **Memory %** - RAM utilization
- **Load Average** - System queue depth

### 🔝 Top 5 Servers Tab
**Detailed server drill-down**:
- Risk gauge for each server
- Current vs Predicted comparison (30min ahead)
- 8-hour prediction timeline with uncertainty bands (p10/p50/p90)
- Profile-specific context

**What You'll See**:
```
Server: ppml0015 (Risk: 57.3)
Profile: ML Compute

Current State vs Predictions (30min ahead):
┌─────────────┬────────┬───────────┬────────┐
│ Metric      │ Current│ Predicted │ Δ      │
├─────────────┼────────┼───────────┼────────┤
│ CPU Used    │ 64.5%  │ 78.2%     │ +13.7% │
│ I/O Wait    │ 2.3%   │ 8.1%      │ +5.8%  │
│ Memory      │ 72.1%  │ 74.5%     │ +2.4%  │
│ Load Avg    │ 6.8    │ 9.2       │ +2.4   │
└─────────────┴────────┴───────────┴────────┘
```

### 🎛️ Interactive Scenarios Tab
**Test different failure modes**:
- Healthy Fleet (baseline)
- Degrading Server (gradual failure)
- Critical Server (imminent failure)
- Multiple Servers at Risk

**How It Works**:
1. Click "Run Scenario" button
2. Dashboard simulates the scenario data
3. Watch predictions update in real-time
4. Risk scores change based on scenario
5. Reset to return to live data

### 🚨 Alerting Strategy Tab
**Production alert configuration**:
- Graduated severity system (7 levels)
- Risk thresholds: Imminent Failure (90+), Critical (80-89), Danger (70-79)
- Escalation paths with timings
- Delivery methods (Phone, SMS, Slack, Email)

**Alert Table Columns**:
- Severity (color-coded)
- Server name
- Risk score
- Profile (ML Compute, Database, etc.)
- CPU/Memory/I/O Wait predictions
- Recommended action

---

## Monitoring I/O Wait (Critical Metric)

**Why I/O Wait Matters**: "System troubleshooting 101"

I/O wait indicates the percentage of CPU time spent waiting for disk/storage operations. High I/O wait = storage bottleneck.

**Risk Scoring**:
- **≥30%**: +50 risk (CRITICAL - severe bottleneck)
- **≥20%**: +30 risk (High I/O contention)
- **≥10%**: +15 risk (Elevated)
- **≥5%**: +5 risk (Noticeable)

**Profile-Specific Expectations**:
- **ML Compute**: <10% (compute-bound, not I/O-bound)
- **Database**: 15% average (I/O intensive, this is normal)
- **Web/API**: <5% (mostly CPU/network)

**In Dashboard**:
- Overview tab shows "I/O Wait Now" and "I/O Wait Predicted"
- Top 5 servers display I/O wait trends
- Risk calculation heavily weights I/O wait

---

## Profile-Based Transfer Learning Demo

**The Power of Profiles**: Add a brand new server, get instant predictions!

### Test Transfer Learning

1. **While streaming**, add a new ML server to the fleet:
```bash
# Edit the streaming command to add a new server
# The model will recognize the "ml_compute" profile
# Predictions will be strong from day 1 (no retraining needed)
```

2. **Dashboard will show**:
   - New server appears in Overview tab
   - Risk score calculated immediately
   - Predictions based on ML Compute profile patterns
   - No "cold start" period needed

3. **7 Profiles Available**:
   - ML_COMPUTE: High CPU/Memory, low I/O wait
   - DATABASE: High I/O wait (15% normal), high network
   - WEB_API: Network-heavy, mostly idle CPU
   - CONDUCTOR_MGMT: Orchestration, moderate load
   - DATA_INGEST: ETL workload, high CPU/disk
   - RISK_ANALYTICS: EOD spikes (4-7pm), compute-intensive
   - GENERIC: Light workload, flat baseline

---

## Scenarios for Tuesday Demo

### Scenario 1: Healthy Fleet (Baseline)
```bash
python metrics_generator.py --stream --scenario healthy
```
**Talking Points**:
- "This is our production fleet of 90 servers"
- "7 different server profiles, each with unique patterns"
- "ML training nodes are compute-heavy, databases show high I/O wait"
- "Dashboard updates every 5 seconds with 8-hour predictions"

### Scenario 2: I/O Wait Spike Detection
```bash
python metrics_generator.py --stream --scenario degrading
```
**Talking Points**:
- "Watch this database server - I/O wait is climbing"
- "Currently 8%, predicted to hit 22% in 30 minutes"
- "This is 'System Troubleshooting 101' - storage bottleneck forming"
- "Alert will trigger at 10%, escalate at 20%"
- "We have 30 minutes to investigate before users are impacted"

### Scenario 3: Multiple Server Failures
```bash
python metrics_generator.py --stream --scenario critical
```
**Talking Points**:
- "Multiple servers in distress - environment risk elevated"
- "3 ML servers showing high risk (80+)"
- "Root cause: shared storage array I/O wait spike across all 3"
- "Graduated alerts: Critical (80-89), Imminent Failure (90+)"
- "Predictions show degradation will continue for next 2 hours"

### Scenario 4: Transfer Learning - New Server
**Talking Points**:
- "Let's add a brand new ML training server - ppml0099"
- "Server has ZERO historical data, just joined the fleet"
- "Model recognizes 'ml_compute' profile"
- "Strong predictions immediately - no retraining required"
- "This is 80% reduction in cold start time vs old approach"

---

## Common Issues & Troubleshooting

### Issue 1: Daemon "No predictions yet - insufficient data"
**Cause**: Daemon needs 24 hours (288 steps) of data before generating predictions.

**Solution**:
```bash
# Wait ~24 minutes (288 steps × 5 seconds)
# Or use fast-forward mode:
python metrics_generator.py --stream --tick_seconds 1
# This generates 288 steps in ~5 minutes
```

### Issue 2: Dashboard shows "Cannot connect to daemon"
**Cause**: Inference daemon not running or wrong URL.

**Check**:
```bash
curl http://localhost:8000/health
```

**Solution**:
```bash
# Restart daemon (Terminal 1)
python tft_inference_daemon.py

# Verify dashboard URL in sidebar (default: http://localhost:8000)
```

### Issue 3: Missing LINBORG Metrics in Training Data
**Cause**: Training data was generated before LINBORG refactor.

**Solution**:
```bash
# Regenerate training data with new metrics
python metrics_generator.py --hours 168 --out_dir ./training --format parquet

# Retrain model
python tft_trainer.py --data_path ./training/server_metrics.parquet --epochs 10
```

**Verify**:
```python
import pandas as pd
df = pd.read_parquet('./training/server_metrics.parquet')
linborg_metrics = ['cpu_user_pct', 'cpu_iowait_pct', 'mem_used_pct', ...]
print(f"LINBORG Metrics: {len([m for m in linborg_metrics if m in df.columns])}/14")
```

### Issue 4: Risk Scores All Show "100"
**Cause**: Old baseline tuning (before granular scoring).

**Solution**: Already fixed in latest version. Risk calculation now uses:
- CPU: Graduated thresholds (98% = +60, 95% = +40, 90% = +20)
- I/O Wait: Graduated thresholds (30% = +50, 20% = +30, 10% = +15)
- Memory: Profile-specific (DB vs non-DB)
- Load Average: Graduated thresholds (>12 = +25, >8 = +15)

---

## Command Reference

### Generate Data
```bash
# Quick test (24 hours, 20 servers)
python metrics_generator.py --hours 24 --num_ml_compute 20

# Production (30 days, 90 servers)
python metrics_generator.py --hours 720 \
    --num_ml_compute 20 --num_database 15 --num_web_api 25 \
    --num_conductor_mgmt 5 --num_data_ingest 10 \
    --num_risk_analytics 8 --num_generic 7
```

### Train Model
```bash
# Quick (5 epochs, ~1 hour)
python tft_trainer.py --data_path ./training/server_metrics.parquet --epochs 5

# Recommended (10 epochs, ~3 hours)
python tft_trainer.py --data_path ./training/server_metrics.parquet --epochs 10

# Production (20 epochs, ~6 hours)
python tft_trainer.py --data_path ./training/server_metrics.parquet --epochs 20
```

### Run Inference
```bash
# Start daemon
python tft_inference_daemon.py

# Stream data (healthy)
python metrics_generator.py --stream --scenario healthy

# Stream data (degrading)
python metrics_generator.py --stream --scenario degrading

# Stream data (critical)
python metrics_generator.py --stream --scenario critical
```

### Launch Dashboard
```bash
# Default port 8501
streamlit run tft_dashboard_web.py

# Custom port
streamlit run tft_dashboard_web.py --server.port 8502
```

---

## Performance Benchmarks

### Training Time (RTX 4090, 90 servers, 720 hours data)
- 1 epoch: ~20-30 minutes
- 5 epochs: ~1.5-2.5 hours
- 10 epochs: ~3-5 hours
- 20 epochs: ~6-10 hours

### Inference Latency
- Single prediction: <50ms
- Batch (90 servers): <200ms
- Dashboard refresh: <100ms (optimized)

### Data Generation Speed
- 24 hours (20 servers): ~30-60 seconds
- 168 hours (20 servers): ~2-3 minutes
- 720 hours (90 servers): ~5-10 minutes

### Streaming Performance
- 90 servers at 5-second intervals: ~10-20ms per batch
- CPU usage: <5% (mostly I/O wait for network)
- Memory: ~500MB rolling window

---

## Next Steps

### For Development
1. **Tune Risk Thresholds**: Edit `calculate_server_risk_score()` in tft_dashboard_web.py
2. **Add New Profiles**: Edit `PROFILE_BASELINES` in metrics_generator.py
3. **Custom Alerting**: Modify alerting logic in Overview tab
4. **Export Predictions**: Use `/predict` endpoint to export JSON

### For Production
1. **Deploy Daemon**: Containerize with Docker
2. **HA Setup**: Load balance multiple daemon instances
3. **Monitoring**: Add Prometheus metrics
4. **Alerting**: Integrate with PagerDuty/Slack
5. **Data Pipeline**: Connect to real Linborg monitoring

### Documentation
- **SESSION_2025-10-13_LINBORG_METRICS_REFACTOR.md**: Complete refactor details
- **MODEL_TRAINING_GUIDELINES.md**: Training best practices
- **PRESENTATION_FINAL.md**: Demo script with talking points
- **CURRENT_STATE_RAG.md**: System architecture

---

## Summary

**Complete Demo Flow**:
1. ✅ Generate training data (5-10 min)
2. ✅ Train model (1-6 hours depending on epochs)
3. ✅ Start inference daemon (instant)
4. ✅ Stream live data (continuous)
5. ✅ Launch dashboard (instant)
6. ✅ Watch real-time predictions with 8-hour forecast

**Key Features Demonstrated**:
- 14 LINBORG production metrics
- I/O Wait monitoring (critical troubleshooting metric)
- Profile-based transfer learning (7 profiles)
- Graduated risk scoring (0-100, granular)
- Interactive scenarios (healthy/degrading/critical)
- Real-time updates (5-second refresh)

**Ready for Tuesday Demo!** 🎉
