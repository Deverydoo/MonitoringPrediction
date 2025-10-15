# TFT Monitoring System - Presentation Startup Guide

**Date**: October 14, 2025
**Status**: Production-Ready with LINBORG Metrics
**Purpose**: Flawless corporate presentation

---

## Pre-Flight Checklist

### ✅ Code Validation (COMPLETED)

- [x] `metrics_generator_daemon.py` - Lines 233-239: All 14 LINBORG metrics defined
- [x] `tft_inference_daemon.py` - Lines 653-657: All 14 LINBORG metrics in heuristic loop
- [x] Training data exists: `training/server_metrics.parquet`
- [x] Model exists: `models/tft_model_20251014_131232/`
- [x] Dashboard updated with color-coding and help bubbles

### ✅ LINBORG Metrics (14 Total)

**CPU (5 metrics)**:
- `cpu_user_pct` - User space CPU
- `cpu_sys_pct` - System/kernel CPU
- `cpu_iowait_pct` - I/O wait (CRITICAL troubleshooting metric)
- `cpu_idle_pct` - Idle CPU (used to calculate % CPU Used = 100 - idle)
- `java_cpu_pct` - Java/Spark CPU usage

**Memory (2 metrics)**:
- `mem_used_pct` - Memory utilization
- `swap_used_pct` - Swap usage (thrashing indicator)

**Disk (1 metric)**:
- `disk_usage_pct` - Disk space usage

**Network (2 metrics)**:
- `net_in_mb_s` - Network ingress (MB/s)
- `net_out_mb_s` - Network egress (MB/s)

**Connections (2 metrics)**:
- `back_close_wait` - TCP backend connections
- `front_close_wait` - TCP frontend connections

**System (2 metrics)**:
- `load_average` - System load average
- `uptime_days` - Days since reboot (maintenance tracking)

---

## Startup Sequence (Critical Path)

### Terminal 1: Metrics Generator Daemon

```bash
cd D:\machine_learning\MonitoringPrediction
python metrics_generator_daemon.py --stream --servers 20
```

**Expected Output**:
```
========================================
🚀 STREAMING STARTED
========================================
   Scenario: HEALTHY
   Fleet: 20 servers
   Target: http://localhost:8000
   Interval: 5 seconds
========================================
```

**Verification**:
```bash
curl http://localhost:8001/
```

Should return:
```json
{
  "service": "Metrics Generator Daemon",
  "status": "streaming",
  "scenario": "healthy",
  "fleet_size": 20
}
```

---

### Terminal 2: TFT Inference Daemon

```bash
cd D:\machine_learning\MonitoringPrediction
python tft_inference_daemon.py --port 8000
```

**Expected Output**:
```
🚀 TFT Inference Daemon Starting...
✅ Model loaded: models/tft_model_20251014_131232
✅ Server encoder loaded: 20 servers
🌐 Daemon started on port 8000
```

**Verification**:
```bash
curl http://localhost:8000/status
```

Should show:
```json
{
  "running": true,
  "warmup": {
    "is_warmed_up": true,
    "message": "Model ready - using TFT predictions"
  }
}
```

---

### Terminal 3: Streamlit Dashboard

```bash
cd D:\machine_learning\MonitoringPrediction
streamlit run tft_dashboard_web.py
```

**Expected Output**:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Browser**: Navigate to `http://localhost:8501`

---

## Verification Steps (Before Presentation)

### 1. Check Dashboard - Fleet Risk Distribution
- Should show bar chart with server names
- Should have help bubble: "No news is good news!"
- If all servers healthy (Risk=0), chart will be flat - THIS IS CORRECT

### 2. Check Dashboard - Active Alerts
- If NO alerts (healthy system): Shows **Top 5 Busiest Servers** table
- Table should have columns: Server, Profile, Status, CPU, Memory, I/O Wait, Swap, Load
- **ALL values should be NON-ZERO** (except Status which shows "Healthy")

### 3. Expected Values (Healthy System)

| Server | Profile | Status | CPU | Memory | I/O Wait | Swap | Load |
|--------|---------|--------|-----|--------|----------|------|------|
| pprisk001 | Risk Analytics | ✅ Healthy | **30-60%** | **50-70%** | **5-15%** | **1-5%** | **8-15** |
| ppdb001 | Database | ✅ Healthy | **40-65%** | **60-80%** | **10-20%** | **1-5%** | **4-8** |
| ppml0001 | ML Compute | ✅ Healthy | **45-75%** | **65-85%** | **1-3%** | **2-8%** | **6-10** |

### 4. Red Flags (MUST FIX)

❌ **All values showing 0.0%**
- **Cause**: Daemons not restarted after code changes
- **Fix**: Restart BOTH daemons (metrics + inference)

❌ **Only Load Average has values, rest are 0.0%**
- **Cause**: Metrics daemon sending incomplete data
- **Fix**: Verify daemon code has all 14 LINBORG metrics (see line 233-239)

❌ **Dashboard shows "No predictions available"**
- **Cause**: Inference daemon not warmed up yet
- **Fix**: Wait 30 seconds for warmup (150 records needed)

---

## Demo Scenario Controls

### Change Scenario via API

**Set to Degrading**:
```bash
curl -X POST http://localhost:8001/scenario/set -H "Content-Type: application/json" -d "{\"scenario\": \"degrading\"}"
```

**Set to Critical**:
```bash
curl -X POST http://localhost:8001/scenario/set -H "Content-Type: application/json" -d "{\"scenario\": \"critical\"}"
```

**Reset to Healthy**:
```bash
curl -X POST http://localhost:8001/scenario/set -H "Content-Type: application/json" -d "{\"scenario\": \"healthy\"}"
```

**Check Current Scenario**:
```bash
curl http://localhost:8001/scenario/status
```

---

## Presentation Flow

### 1. **Opening** (2 minutes)
- Show healthy dashboard
- Point out "Top 5 Busiest Servers" - all values populated
- Explain: "No news is good news - empty risk chart means everything healthy"

### 2. **LINBORG Metrics** (3 minutes)
- Explain 14 production metrics
- Highlight **I/O Wait** as "system troubleshooting 101"
- Show CPU calculation: **% Used = 100 - Idle**
- Point out color-coding: 🟡 🟠 🔴

### 3. **Live Degradation** (5 minutes)
- Switch to `degrading` scenario
- Watch servers turn yellow/orange in Fleet Risk Distribution
- Active Alerts table populates with warnings
- Show color-coded cells (I/O Wait goes 🟠, Memory goes 🟡)
- Point out predictive nature: "Predicted (30m)" column

### 4. **Critical Scenario** (3 minutes)
- Switch to `critical` scenario
- Multiple servers go red
- Show imminent failure warnings
- Demonstrate early warning value

### 5. **Recovery** (2 minutes)
- Switch back to `healthy`
- Watch servers recover
- Return to "Top 5 Busiest" display

---

## Troubleshooting

### Problem: Dashboard shows all zeros

**Quick Check**:
```bash
# Check metrics daemon output
curl http://localhost:8001/status

# Check inference daemon predictions
curl http://localhost:8000/predictions/current | python -c "import json, sys; data=json.load(sys.stdin); server=list(data['predictions'].keys())[0]; print(list(data['predictions'][server].keys()))"
```

**Expected**: Should list all 14 LINBORG metrics

**If only seeing 6 metrics**: Restart metrics daemon

---

### Problem: Daemon crashes or errors

**Check logs**:
- Metrics daemon: Check terminal where it's running
- Inference daemon: Check terminal where it's running

**Common fixes**:
1. Port already in use: Kill existing process
2. Model not found: Check `models/` directory
3. Training data missing: Re-run data generation

---

## Success Criteria

Before presenting, verify ALL of these:

- [ ] Metrics daemon running on port 8001
- [ ] Inference daemon running on port 8000
- [ ] Dashboard accessible at http://localhost:8501
- [ ] Fleet Risk Distribution shows servers (or flat if healthy)
- [ ] Top 5 Busiest Servers shows **NON-ZERO values** for ALL columns
- [ ] Can switch scenarios and see dashboard update
- [ ] Color-coding works (🟡 🟠 🔴 appear when degrading)
- [ ] No errors in any daemon terminals

---

## Backup Plan

If something breaks during presentation:

1. **Have screenshot/recording ready** of working system
2. **Restart sequence**: 30 seconds to recover
   - Ctrl+C on both daemons
   - Restart metrics daemon
   - Restart inference daemon
   - Refresh dashboard
3. **Worst case**: Use static training data visualization from `_StartHere.ipynb`

---

**Last Updated**: October 14, 2025
**Validated By**: Claude Code + Human Review
**Ready for**: Corporate Presentation
