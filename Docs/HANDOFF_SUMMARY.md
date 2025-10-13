# Team Handoff Summary

**Date:** October 13, 2025
**From:** ML Engineering Team
**To:** Production Integration Team

---

## What You're Getting

A production-ready TFT (Temporal Fusion Transformer) Inference Daemon that:
- Accepts server metrics via REST API
- Generates 8-hour predictions for 20 servers
- Provides incident probability forecasts
- Has been tested and validated

---

## Quick Start (5 Minutes)

### 1. Start the Inference Daemon

```bash
cd /path/to/MonitoringPrediction
python tft_inference_daemon.py
```

You should see:
```
[OK] Found model: models/tft_model_20251013_100205
[SUCCESS] TFT model loaded!
[READY] Daemon started - waiting for data feed
```

### 2. Test with Sample Data

```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "timestamp": "2025-10-13T16:00:00Z",
      "server_name": "ppdb001",
      "cpu_pct": 45.2,
      "mem_pct": 67.8,
      "disk_io_mb_s": 123.4,
      "latency_ms": 12.5,
      "state": "healthy"
    }]
  }'
```

### 3. Check Status

```bash
curl http://localhost:8000/status | jq
```

### 4. View Dashboard

```bash
streamlit run tft_dashboard_web.py
```

Open browser to: http://localhost:8501

---

## Documentation Files

### Essential Reading (Start Here)

1. **[PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md)** ⭐
   - Complete integration guide
   - Data format specification
   - Code examples in 6 languages
   - Troubleshooting guide
   - **Read this first!**

2. **[QUICK_REFERENCE_API.md](QUICK_REFERENCE_API.md)** ⭐
   - One-page API reference
   - Keep this handy while coding
   - Quick copy-paste examples

3. **[production_metrics_forwarder_TEMPLATE.py](../production_metrics_forwarder_TEMPLATE.py)** ⭐
   - Production-ready template script
   - Just implement metric collection
   - Includes retry logic, logging, alerts

### Technical Details

4. **[HOW_PREDICTIONS_WORK.md](HOW_PREDICTIONS_WORK.md)**
   - Architecture overview
   - How the model works
   - Quantile forecasts explained

5. **[MODEL_TRAINING_GUIDELINES.md](MODEL_TRAINING_GUIDELINES.md)**
   - How the model was trained
   - Retraining procedures
   - Performance benchmarks

6. **[DATA_CONTRACT.md](DATA_CONTRACT.md)**
   - Data format specification
   - Field definitions
   - Validation rules

### Bug Fixes and Sessions

7. **[BUGFIX_8_SERVER_LIMIT.md](BUGFIX_8_SERVER_LIMIT.md)**
   - Critical bug fix documentation
   - Shows obsessive debugging process

8. **[SESSION_2025-10-13_DASHBOARD_POLISH.md](SESSION_2025-10-13_DASHBOARD_POLISH.md)**
   - Recent improvements
   - Dashboard features
   - Risk scoring tuning

---

## Your Task: Implement Metric Collection

### Step 1: Copy Template

```bash
cp production_metrics_forwarder_TEMPLATE.py your_metrics_forwarder.py
```

### Step 2: Implement This Function

Open `your_metrics_forwarder.py` and replace the placeholder in:

```python
def collect_metrics_from_your_system() -> List[Dict]:
    """
    REPLACE THIS with your actual metrics collection.

    Query your monitoring system (Prometheus, InfluxDB, CloudWatch, etc.)
    and return metrics in this format:
    """
    records = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    # YOUR CODE HERE
    # Example: Query Prometheus, InfluxDB, CloudWatch, etc.

    for server in SERVER_LIST:
        records.append({
            "timestamp": timestamp,
            "server_name": server,  # MUST match training data
            "cpu_pct": 45.2,        # Get from your system
            "mem_pct": 67.8,        # Get from your system
            "disk_io_mb_s": 123.4,  # Get from your system
            "latency_ms": 12.5,     # Get from your system
            "state": "healthy"      # Derive from metrics
        })

    return records
```

### Step 3: Configure

Update these variables in the template:

```python
INFERENCE_URL = "http://localhost:8000/feed/data"  # Your inference daemon URL
SERVER_LIST = ['ppdb001', 'ppdb002', ...]          # Your actual servers
SEND_INTERVAL = 5                                   # Seconds between batches
```

### Step 4: Test

```bash
python your_metrics_forwarder.py
```

You should see:
```
[INFO] ✓ Sent 20 records | Tick: 1 | Warmup: 1/20 servers
[INFO] ✓ Sent 20 records | Tick: 2 | Warmup: 2/20 servers
...
[INFO] ✓ Sent 20 records | Tick: 150 | Warmup: ✓ READY
```

---

## Data Format Requirements

### Exact Schema

```json
{
  "records": [
    {
      "timestamp": "2025-10-13T16:00:00Z",  // ISO 8601 format
      "server_name": "ppdb001",              // Must match training
      "cpu_pct": 45.2,                       // 0.0 - 100.0
      "mem_pct": 67.8,                       // 0.0 - 100.0
      "disk_io_mb_s": 123.4,                 // 0.0+
      "latency_ms": 12.5,                    // 0.0+
      "state": "healthy"                     // See valid states
    }
  ]
}
```

### Valid States

```
healthy, heavy_load, critical_issue, idle,
maintenance, morning_spike, offline, recovery
```

### Trained Server Names

**IMPORTANT:** Use these exact names (case-sensitive):

```
ppml0001, ppml0002, ppml0003, ppml0004
ppdb001, ppdb002, ppdb003
ppweb001, ppweb002, ppweb003, ppweb004, ppweb005, ppweb006, ppweb007, ppweb008
ppcon01
ppetl001, ppetl002
pprisk001
ppgen001
```

---

## Integration Patterns

Choose the pattern that matches your infrastructure:

### Pattern 1: Prometheus/Grafana Stack
- Use `prometheus_api_client` library
- Query metrics every 5 seconds
- Transform to inference format
- See template for code example

### Pattern 2: InfluxDB/TimescaleDB
- Use `influxdb_client` library
- Fetch last 5 seconds of data
- Batch send all servers
- See template for code example

### Pattern 3: Log File Tailing
- Use `tail -F` or Python file watching
- Parse JSON logs
- POST to inference daemon
- See PRODUCTION_INTEGRATION_GUIDE.md

### Pattern 4: Direct Agent Integration
- Collectd, Telegraf, or custom agent
- Use `exec` plugin or custom output
- POST directly from agent
- See PRODUCTION_INTEGRATION_GUIDE.md

---

## Important Timing Requirements

| Aspect | Requirement | Notes |
|--------|-------------|-------|
| **Send Frequency** | Every 5 seconds | Matches training data |
| **Warmup Time** | ~12.5 minutes per server | 150 records × 5s |
| **Request Timeout** | 5 seconds | Use in HTTP client |
| **Batch Size** | All 20 servers per request | More efficient |
| **Prediction Latency** | 1-2 seconds | GPU mode |

---

## Expected Behavior

### During Warmup (First 12.5 Minutes)

```
[INFO] ✓ Sent 20 records | Tick: 1 | Warmup: 0/20 servers
[INFO] ✓ Sent 20 records | Tick: 2 | Warmup: 0/20 servers
...
[INFO] ✓ Sent 20 records | Tick: 30 | Warmup: 1/20 servers
...
[INFO] ✓ Sent 20 records | Tick: 150 | Warmup: 20/20 servers
```

Dashboard shows: "Model is warming up..."

### After Warmup

```
[INFO] ✓ Sent 20 records | Tick: 151 | Warmup: ✓ READY
[OK] TFT predictions generated for 20 servers
```

Dashboard shows:
- All 20 servers with predictions
- 8-hour forecast graphs
- Risk scores and alerts
- Incident probabilities

---

## Troubleshooting

### Issue: "insufficient_data" Error

**Symptom:**
```json
{"error": "insufficient_data", "message": "Need at least 100 records"}
```

**Solution:** Keep sending data. Need 100 total records minimum.

---

### Issue: Only 8 Servers Showing

**Symptom:** Dashboard shows 8 servers instead of 20

**Solution:** This was a bug that has been fixed. Update to latest code.

**Verification:**
```bash
curl http://localhost:8000/predictions/current | jq '.predictions | keys | length'
# Should return: 20
```

---

### Issue: Predictions Using Heuristics

**Symptom:** Logs show `[WARNING] TFT prediction failed, falling back to heuristic`

**Possible Causes:**
1. Server name doesn't match training data
2. Not enough history per server
3. Data quality issues (NaN, out of range)

**Solution:**
- Verify server names match EXACTLY (case-sensitive)
- Check warmup: `curl http://localhost:8000/status | jq '.warmup.is_warmed_up'`
- Validate data format

---

### Issue: High Latency or Timeouts

**Symptom:** POST requests taking >5 seconds

**Solution:**
- Check GPU: `nvidia-smi` (if using GPU)
- Reduce batch size (send 10 servers per request instead of 20)
- Check network latency
- Verify daemon isn't CPU-bound

---

## Monitoring Checklist

Set up monitoring for:

- [ ] Daemon uptime (>99%)
- [ ] Prediction latency (<5s)
- [ ] Request success rate (>95%)
- [ ] Warmup status (false for >15 min = problem)
- [ ] TFT fallback rate (<10%)
- [ ] Disk space for persistence file

---

## Production Deployment Checklist

### Before Deployment

- [ ] Tested with sample data
- [ ] Verified server names match training
- [ ] Confirmed data format matches contract
- [ ] Tested network connectivity to daemon
- [ ] Verified GPU available (if using)
- [ ] Implemented metric collection function
- [ ] Tested retry logic and error handling

### During Deployment

- [ ] Start inference daemon
- [ ] Monitor logs for "[SUCCESS] TFT model loaded!"
- [ ] Start metrics forwarder
- [ ] Monitor warmup progress
- [ ] Wait for "Warmup: ✓ READY"
- [ ] Verify predictions in dashboard

### After Deployment

- [ ] All 20 servers showing predictions
- [ ] No "falling back to heuristic" warnings
- [ ] Prediction latency <2 seconds
- [ ] Set up log rotation
- [ ] Configure process monitoring (systemd/supervisord)
- [ ] Document runbook for incidents

---

## Getting Help

### Check Logs First

```bash
# Inference daemon logs
tail -f inference.log | grep -i error

# Forwarder logs
tail -f metrics_forwarder.log | grep -i error
```

### Verify System Health

```bash
# Is daemon running?
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status | jq

# Are predictions working?
curl http://localhost:8000/predictions/current | jq '.metadata.model_type'
# Should return: "TFT"
```

### Common Debug Commands

```bash
# Check GPU utilization
nvidia-smi

# Check model loaded
curl http://localhost:8000/status | jq '.running'

# Check warmup progress
watch -n 2 'curl -s http://localhost:8000/status | jq .warmup'

# Count predictions
curl -s http://localhost:8000/predictions/current | jq '.predictions | keys | length'
```

---

## Contact Information

For issues or questions:

1. **Check Documentation:** See PRODUCTION_INTEGRATION_GUIDE.md first
2. **Check Logs:** Most issues show clear error messages
3. **Provide Context:** Include logs, sample data, and error messages

---

## Summary

You have everything needed to integrate production data:

✅ **Running inference daemon** (port 8000)
✅ **Data format specification** (see QUICK_REFERENCE_API.md)
✅ **Template script** (production_metrics_forwarder_TEMPLATE.py)
✅ **Complete documentation** (6 guides + code examples)
✅ **Tested and validated** (model predicting for 20 servers)

**Your only task:** Implement `collect_metrics_from_your_system()` function.

**Estimated time:** 2-4 hours (depending on your monitoring system)

Good luck! 🚀

---

**Last Updated:** October 13, 2025
**Maintained By:** ML Engineering Team
