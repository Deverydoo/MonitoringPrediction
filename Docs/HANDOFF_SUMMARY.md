# Team Handoff Summary

**Date:** October 17, 2025 (Updated)
**Version:** 1.0.0 - Production Ready
**From:** ML Engineering Team
**To:** Production Integration Team

---

## What You're Getting

A **production-ready** TFT (Temporal Fusion Transformer) Predictive Monitoring System that:
- ✅ Accepts server metrics via REST API with API key authentication
- ✅ Generates 8-hour predictions for 20 servers
- ✅ Provides incident probability forecasts
- ✅ **NEW:** Production data adapters for MongoDB/Elasticsearch
- ✅ **NEW:** Automatic API key management
- ✅ **NEW:** Performance-optimized dashboard (16x faster)
- ✅ **NEW:** Incremental training and adaptive retraining
- ✅ Has been tested, validated, and production-hardened

---

## 🎯 Quick Decision Guide

**Choose your integration path based on your data source:**

| Your Situation | Recommended Path | Setup Time |
|----------------|------------------|------------|
| ✅ "Our metrics are in **MongoDB**" | Use MongoDB Adapter | 10-15 min |
| ✅ "Our metrics are in **Elasticsearch**" | Use Elasticsearch Adapter | 10-15 min |
| ✅ "We use **Prometheus/Grafana**" | Use Custom Forwarder | 2-4 hours |
| ✅ "We use **InfluxDB/TimescaleDB**" | Use Custom Forwarder | 2-4 hours |
| ✅ "We have **multiple sources**" | Use Custom Forwarder | 2-4 hours |
| ✅ "We need **custom logic**" | Use Custom Forwarder | 2-4 hours |

**👉 If you have MongoDB/Elasticsearch:** Jump to [Production Adapters](#approach-1-mongodbělasticsearch-adapters-new---easiest)

**👉 If you need custom integration:** Jump to [Custom Forwarder](#approach-2-custom-metrics-forwarder-original-method)

---

## Quick Start (5 Minutes)

### Automated Startup (Recommended)

```bash
cd /path/to/MonitoringPrediction

# Windows
start_all.bat

# Linux/Mac
./start_all.sh
```

This automatically:
1. ✅ Generates/verifies API key (`.env` file)
2. ✅ Starts TFT Inference Daemon (port 8000)
3. ✅ Starts Metrics Generator (demo mode)
4. ✅ Starts Dashboard (port 8501)

### Manual Startup (Advanced)

#### 1. Generate API Key (First Time Only)

```bash
python generate_api_key.py
```

You should see:
```
[OK] Dashboard configuration: .streamlit/secrets.toml
[OK] Daemon configuration: .env
Generated API Key: abc123def456...
```

#### 2. Start the Inference Daemon

```bash
python tft_inference_daemon.py
```

You should see:
```
[OK] Found model: models/tft_model_20251017_122454
[SUCCESS] TFT model loaded!
[READY] Daemon started - waiting for data feed
```

#### 3. Test with Sample Data

```bash
# Get API key from .env
API_KEY=$(grep TFT_API_KEY .env | cut -d= -f2)

curl -X POST http://localhost:8000/feed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '[{
    "timestamp": "2025-10-17T16:00:00Z",
    "server_name": "ppdb001",
    "cpu_pct": 45.2,
    "mem_pct": 67.8,
    "disk_io_mb_s": 123.4,
    "latency_ms": 12.5,
    "state": "healthy"
  }]'
```

#### 4. Check Status

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/status | jq
```

#### 5. View Dashboard

```bash
streamlit run tft_dashboard_web.py
```

Open browser to: http://localhost:8501

---

## Documentation Files

### 🔥 NEW: Production Integration Options (v1.0.0)

**You now have TWO ways to integrate with production:**

#### Option 1: Production Data Adapters (MongoDB/Elasticsearch) 🆕 RECOMMENDED
**Perfect for:** Organizations with existing monitoring databases (Linborg, Prometheus, Grafana)

1. **[ADAPTER_ARCHITECTURE.md](ADAPTER_ARCHITECTURE.md)** ⭐⭐⭐ **CRITICAL**
   - **READ THIS FIRST if using production databases**
   - Explains microservices architecture
   - How adapters run as independent daemons
   - Data flow and communication protocols
   - Production deployment guides

2. **[PRODUCTION_DATA_ADAPTERS.md](PRODUCTION_DATA_ADAPTERS.md)** ⭐⭐
   - Quick reference for adapter setup
   - MongoDB vs Elasticsearch comparison
   - 3-step quick start guide
   - Production checklist

3. **[adapters/README.md](../adapters/README.md)** ⭐⭐
   - Complete adapter documentation (697 lines)
   - Installation and configuration
   - Field mapping reference
   - Security best practices
   - Troubleshooting guide

4. **[API_KEY_SETUP.md](API_KEY_SETUP.md)** ⭐
   - API key authentication explained
   - Automatic generation and management
   - Security considerations

#### Option 2: Custom Metrics Forwarder (Original Method)
**Perfect for:** Custom integrations, multiple data sources, or non-database sources

1. **[PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md)** ⭐
   - Complete integration guide
   - Data format specification
   - Code examples in 6 languages
   - Troubleshooting guide

2. **[QUICK_REFERENCE_API.md](QUICK_REFERENCE_API.md)** ⭐
   - One-page API reference
   - Keep this handy while coding
   - Quick copy-paste examples

3. **[production_metrics_forwarder_TEMPLATE.py](../production_metrics_forwarder_TEMPLATE.py)** ⭐
   - Production-ready template script
   - Just implement metric collection
   - Includes retry logic, logging, alerts

### Technical Details & Advanced Features

5. **[PERFORMANCE_OPTIMIZATION.md](PERFORMANCE_OPTIMIZATION.md)** 🆕
   - Dashboard caching strategies (16x faster)
   - Python bytecode compilation
   - Production mode configurations
   - Performance benchmarks

6. **[ADAPTIVE_RETRAINING_PLAN.md](ADAPTIVE_RETRAINING_PLAN.md)** 🆕
   - Automatic drift detection (4 metrics)
   - 88% SLA alignment triggers
   - Retraining safeguards
   - Implementation roadmap

7. **[CONTINUOUS_LEARNING_PLAN.md](CONTINUOUS_LEARNING_PLAN.md)** 🆕
   - Incremental training system
   - Profile-based transfer learning
   - Day-1 prediction capability
   - Model versioning strategy

8. **[HOW_PREDICTIONS_WORK.md](HOW_PREDICTIONS_WORK.md)**
   - Architecture overview
   - How the model works
   - Quantile forecasts explained

9. **[MODEL_TRAINING_GUIDELINES.md](MODEL_TRAINING_GUIDELINES.md)**
   - How the model was trained
   - Retraining procedures
   - Performance benchmarks

10. **[DATA_CONTRACT.md](DATA_CONTRACT.md)**
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

## 🚀 Production Integration: Two Approaches

### Approach 1: MongoDB/Elasticsearch Adapters (NEW - Easiest)

**Use this if:** Your monitoring data is already in MongoDB or Elasticsearch (Linborg, Prometheus, etc.)

#### Quick Start with Adapters

```bash
# Step 1: Generate API key (done automatically by start_all scripts)
python generate_api_key.py

# Step 2: Install adapter dependencies
pip install pymongo  # For MongoDB
# OR
pip install elasticsearch  # For Elasticsearch

# Step 3: Configure adapter
cd adapters/
cp mongodb_adapter_config.json.template mongodb_config.json
# Edit mongodb_config.json with your database credentials

# Step 4: Test adapter (one-time fetch)
python mongodb_adapter.py --config mongodb_config.json --once --verbose

# Step 5: Run adapter daemon (continuous streaming)
python mongodb_adapter.py --config mongodb_config.json --daemon --interval 5
```

#### What the Adapter Does

```
┌────────────────────────────────────────────────────────┐
│           ADAPTER ARCHITECTURE                         │
└────────────────────────────────────────────────────────┘

Your MongoDB/Elasticsearch
    ↓ (adapter queries every 5 seconds)
Adapter Process (independent daemon)
    ↓ (HTTP POST to /feed with API key)
TFT Inference Daemon (port 8000)
    ↓ (generates predictions)
Dashboard (port 8501)
    ↓ (displays to user)
```

**Key Concept:** The adapter is an **independent daemon** that continuously:
1. Fetches metrics from your database
2. Transforms to TFT format
3. POSTs to inference daemon via HTTP

**No code required!** Just configure database credentials.

#### Adapter Configuration Example (MongoDB)

```json
{
  "mongodb": {
    "uri": "mongodb://your-mongo-server:27017",
    "database": "linborg",
    "collection": "server_metrics",
    "username": "tft_readonly",
    "password": "your-password"
  },
  "tft_daemon": {
    "url": "http://localhost:8000"
    // API key automatically loaded from .env
  }
}
```

#### Field Mapping

The adapter automatically maps your database fields to TFT format:

| Your DB Field | TFT Field | Notes |
|---------------|-----------|-------|
| `timestamp` / `@timestamp` | `timestamp` | ISO 8601 format |
| `server_name` / `hostname` | `server_name` | Must match training |
| `cpu_usage_pct` / `cpu.pct` | `cpu_pct` | 0-100 |
| `memory_pct` / `mem.pct` | `mem_pct` | 0-100 |
| `disk_io_mb_s` | `disk_io_mb_s` | MB/s |
| `latency_ms` | `latency_ms` | Milliseconds |
| `state` / `status` | `state` | See valid states |

See [adapters/README.md](../adapters/README.md) for complete field mapping reference.

---

### Approach 2: Custom Metrics Forwarder (Original Method)

**Use this if:** You need custom logic, multiple data sources, or non-database integrations

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

## 🤔 Which Integration Approach Should You Use?

### Decision Matrix

| Factor | MongoDB/ES Adapters | Custom Forwarder |
|--------|---------------------|------------------|
| **Data Source** | MongoDB or Elasticsearch | Any (Prometheus, InfluxDB, logs, APIs) |
| **Coding Required** | ❌ No (just config) | ✅ Yes (implement `collect_metrics()`) |
| **Setup Time** | ⏱️ 10-15 minutes | ⏱️ 2-4 hours |
| **Flexibility** | ⚠️ Limited (DB schema must match) | ✅ Complete (any source) |
| **Authentication** | ✅ Automatic (from .env) | ✅ Automatic (from .env) |
| **Production Ready** | ✅ Yes (tested) | ✅ Yes (template provided) |
| **Maintenance** | ✅ Low (no code) | ⚠️ Medium (your code) |
| **Multiple Sources** | ❌ No (one DB per adapter) | ✅ Yes (aggregate in code) |
| **Custom Logic** | ❌ No | ✅ Yes (filtering, aggregation) |

### Recommendations

**✅ Use MongoDB/Elasticsearch Adapters if:**
- Your monitoring data is in MongoDB or Elasticsearch
- You want fastest time to production (10 minutes)
- You don't need custom data transformations
- You prefer zero-code configuration approach
- Example: "Our Linborg metrics are stored in MongoDB"

**✅ Use Custom Metrics Forwarder if:**
- You use Prometheus, InfluxDB, CloudWatch, or other sources
- You need to aggregate data from multiple systems
- You need custom business logic or filtering
- Your data format doesn't match LINBORG schema
- Example: "We pull metrics from Prometheus and Datadog APIs"

**✅ Use Both if:**
- You have multiple monitoring systems
- Some data in databases, some from APIs
- Run multiple adapters simultaneously to aggregate data

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

### v1.0.0 Production Features

✅ **Two Integration Approaches**
   - MongoDB/Elasticsearch adapters (zero code, 10 min setup)
   - Custom metrics forwarder (flexible, 2-4 hr setup)

✅ **Running Infrastructure**
   - Inference daemon (port 8000) with API key auth
   - Dashboard (port 8501) with 16x performance improvement
   - Automated startup scripts (start_all.bat/sh)

✅ **Production Hardening**
   - API key authentication (automatic from .env)
   - Incremental training support
   - Adaptive retraining plan (drift detection)
   - Performance optimization (caching, bytecode compilation)

✅ **Complete Documentation**
   - 10+ comprehensive guides (5,000+ lines)
   - Production adapter architecture (critical reading)
   - API reference and code examples
   - Troubleshooting guides

✅ **Tested and Validated**
   - Model predicting for 20 servers
   - 88% accuracy SLA
   - Production security hardening
   - Performance benchmarks

### Your Task (Choose One)

**Option A: MongoDB/Elasticsearch Adapter (Fastest)**
1. Configure adapter with DB credentials (5 min)
2. Test with `--once --verbose` (2 min)
3. Run adapter daemon (3 min)
4. **Total time: 10-15 minutes**

**Option B: Custom Metrics Forwarder**
1. Copy template script (1 min)
2. Implement `collect_metrics_from_your_system()` (1-3 hr)
3. Configure and test (30 min)
4. **Total time: 2-4 hours**

### Next Steps

1. **Read:** [ADAPTER_ARCHITECTURE.md](ADAPTER_ARCHITECTURE.md) (if using adapters) OR [PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md) (if custom)
2. **Choose:** MongoDB/Elasticsearch adapter OR custom forwarder
3. **Configure:** Database credentials OR implement metric collection
4. **Test:** Run with `--once --verbose` first
5. **Deploy:** Start adapter daemon or custom forwarder
6. **Monitor:** Check dashboard for predictions

Good luck! 🚀

---

**Last Updated:** October 17, 2025
**Version:** 1.0.0 - Production Ready
**Maintained By:** ML Engineering Team

## 📞 Quick Reference

- **Inference Daemon:** http://localhost:8000
- **Dashboard:** http://localhost:8501
- **API Key Location:** `.env` file (auto-generated)
- **Adapters:** `adapters/` directory
- **Documentation:** `Docs/` directory
- **Critical Doc:** [ADAPTER_ARCHITECTURE.md](ADAPTER_ARCHITECTURE.md)
