# Real Data Integration Guide
**Stop Using Demo Data - Connect Your Production Systems**

**Version:** 1.0
**Audience:** DevOps Engineers, SREs, System Administrators
**Purpose:** Complete guide to transition from demo data to real production monitoring data

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Step-by-Step Integration](#step-by-step-integration)
5. [Data Source Adapters](#data-source-adapters)
6. [Verification & Testing](#verification--testing)
7. [Troubleshooting](#troubleshooting)
8. [Production Checklist](#production-checklist)

---

## Overview

The NordIQ system ships with a **metrics generator daemon** that creates realistic demo data. This guide shows you how to:

1. **Stop the demo data generator**
2. **Connect your real monitoring systems** (Elasticsearch, Prometheus, MongoDB, etc.)
3. **Transform your logs** into the required format
4. **Feed the inference engine** with production data
5. **Verify predictions** are working correctly

**Time to Complete:** 1-2 hours for first integration

---

## Prerequisites

### 1. Model Training Required

⚠️ **CRITICAL:** Your TFT model must be trained on data matching your production server profiles.

**Check your model training:**
```bash
# Review what profiles your model knows
cat models/tft_model_*/training_info.json | grep -A 10 "unique_profiles"
```

**Required profiles in training data:**
- Match your actual server types (database, web, ML compute, etc.)
- Include the 14 LINBORG metrics your production systems generate
- Cover at least 7-30 days of historical data

**If your model was trained on demo data only**, you'll need to:
1. Export 30 days of production logs
2. Run `_Starthere.ipynb` with your real data
3. Train a new model (see [Training Guide](../getting-started/QUICK_START.md))

### 2. System Access Requirements

You'll need:
- ✅ Access to your monitoring system (read permissions)
- ✅ Ability to run Python scripts on your network
- ✅ Network connectivity to inference daemon (port 8000)
- ✅ Optional: Write access for custom adapters

### 3. Required Metrics

Your production systems must provide these **14 LINBORG metrics** (or equivalents):

| Metric | Description | Source Examples |
|--------|-------------|-----------------|
| `cpu_user_pct` | User CPU % | `top`, Prometheus `node_cpu_seconds_total{mode="user"}` |
| `cpu_sys_pct` | System CPU % | `top`, Prometheus `node_cpu_seconds_total{mode="system"}` |
| `cpu_iowait_pct` | I/O Wait % (critical) | `iostat`, Prometheus `node_cpu_seconds_total{mode="iowait"}` |
| `cpu_idle_pct` | Idle CPU % | Calculated: 100 - (user + sys + iowait) |
| `java_cpu_pct` | Java process CPU | `jstat`, JMX, Application metrics |
| `mem_used_pct` | Memory used % | `free -m`, Prometheus `node_memory_MemAvailable_bytes` |
| `swap_used_pct` | Swap used % | `free -m`, Prometheus `node_memory_SwapTotal_bytes` |
| `disk_usage_pct` | Disk usage % | `df -h`, Prometheus `node_filesystem_avail_bytes` |
| `net_in_mb_s` | Network in (MB/s) | `ifstat`, Prometheus `node_network_receive_bytes_total` |
| `net_out_mb_s` | Network out (MB/s) | `ifstat`, Prometheus `node_network_transmit_bytes_total` |
| `back_close_wait` | Backend TCP CLOSE_WAIT | `netstat`, Prometheus `node_netstat_Tcp_CurrEstab` |
| `front_close_wait` | Frontend TCP CLOSE_WAIT | `netstat`, Application-level metrics |
| `load_average` | System load (1min) | `uptime`, Prometheus `node_load1` |
| `uptime_days` | Days since last reboot | `uptime -s`, Prometheus `node_boot_time_seconds` |

**Don't have all 14?** See [Partial Metrics Support](#partial-metrics-support) below.

---

## Understanding the Architecture

### Demo Mode (Current State)
```
┌─────────────────────────┐
│ Metrics Generator       │ Port 8001
│ (Demo Data)             │ Generates fake metrics
└──────────┬──────────────┘
           │ POST /feed/data (every 5 sec)
           ↓
┌─────────────────────────┐
│ Inference Daemon        │ Port 8000
│ (TFT Model)             │ Makes predictions
└──────────┬──────────────┘
           │ GET /predictions/current
           ↓
┌─────────────────────────┐
│ Dashboard               │ Port 8501
│ (Streamlit or Dash)     │ Visualizes
└─────────────────────────┘
```

### Production Mode (Target State)
```
┌─────────────────────────┐
│ Your Monitoring System  │ Elasticsearch, Prometheus, etc.
│ (Real Logs)             │
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│ Data Adapter            │ Python script (you write)
│ (Transform & Send)      │ Converts logs → LINBORG format
└──────────┬──────────────┘
           │ POST /feed/data (every 5 sec)
           ↓
┌─────────────────────────┐
│ Inference Daemon        │ Port 8000
│ (TFT Model)             │ Makes predictions on REAL data
└──────────┬──────────────┘
           │ GET /predictions/current
           ↓
┌─────────────────────────┐
│ Dashboard               │ Port 8501
│ (Production Monitoring) │ Shows real predictions!
└─────────────────────────┘
```

---

## Step-by-Step Integration

### Step 1: Stop the Demo Metrics Generator

**Find the metrics generator process:**
```bash
# Windows
tasklist | findstr "metrics_generator"

# Linux/Mac
ps aux | grep metrics_generator
```

**Stop it:**
```bash
# If using start_all scripts, edit them first
# Windows: Edit start_all.bat, comment out metrics generator line
# Linux/Mac: Edit start_all.sh, comment out metrics generator line

# Or kill directly (get PID from above)
kill <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

**Verify it's stopped:**
```bash
curl http://localhost:8001/health
# Should fail with "connection refused"
```

### Step 2: Identify Your Data Source

Choose your monitoring system:

**Option A: Elasticsearch (Logs stored in ELK stack)**
- Best for: Centralized logging systems
- Adapter: `adapters/elasticsearch_adapter.py`
- Query: Elasticsearch DSL

**Option B: Prometheus (Metrics already collected)**
- Best for: Cloud-native, Kubernetes environments
- Adapter: `adapters/prometheus_adapter.py`
- Query: PromQL

**Option C: MongoDB (NoSQL metrics store)**
- Best for: Custom monitoring solutions
- Adapter: `adapters/mongodb_adapter.py`
- Query: MongoDB aggregation pipeline

**Option D: Direct REST API (Your custom metrics API)**
- Best for: Proprietary systems, custom tooling
- Adapter: `adapters/custom_rest_adapter.py`
- Query: HTTP GET/POST

**Option E: File Watcher (Parse log files directly)**
- Best for: Legacy systems, file-based logging
- Adapter: `adapters/file_watcher_adapter.py`
- Query: Tail files, parse with regex

### Step 3: Install and Configure Your Adapter

**Example: Elasticsearch Adapter**

```bash
# Install dependencies
pip install elasticsearch python-dateutil requests

# Copy adapter template
cp NordIQ/src/core/adapters/elasticsearch_adapter.py ./my_adapter.py

# Configure connection
nano my_adapter.py
```

**Edit connection settings:**
```python
# my_adapter.py
ELASTICSEARCH_HOST = "your-elk-server.company.com"
ELASTICSEARCH_PORT = 9200
ELASTICSEARCH_INDEX = "server-metrics-*"
ELASTICSEARCH_USER = "readonly_user"
ELASTICSEARCH_PASSWORD = "your_password"

# Inference daemon endpoint
INFERENCE_URL = "http://localhost:8000/feed/data"
INFERENCE_API_KEY = "your-api-key-here"  # From .env file

# Polling interval (match your log frequency)
POLL_INTERVAL_SECONDS = 5
```

### Step 4: Map Your Metrics to LINBORG Format

**Define field mappings in your adapter:**

```python
# Example: Your Elasticsearch field names → LINBORG names
FIELD_MAPPING = {
    # Source field           # LINBORG field
    "host.name":             "server_name",
    "system.cpu.user.pct":   "cpu_user_pct",
    "system.cpu.system.pct": "cpu_sys_pct",
    "system.cpu.iowait.pct": "cpu_iowait_pct",
    "system.cpu.idle.pct":   "cpu_idle_pct",
    "java.process.cpu.pct":  "java_cpu_pct",
    "system.memory.used.pct": "mem_used_pct",
    "system.swap.used.pct":  "swap_used_pct",
    "system.disk.used.pct":  "disk_usage_pct",
    "system.network.in.mbps": "net_in_mb_s",
    "system.network.out.mbps": "net_out_mb_s",
    "tcp.connections.close_wait_backend": "back_close_wait",
    "tcp.connections.close_wait_frontend": "front_close_wait",
    "system.load.1m":        "load_average",
    "@timestamp":            "timestamp",
}

# Profile detection (map server names to profiles)
def detect_profile(server_name):
    """Map server names to NordIQ profiles."""
    if server_name.startswith("db"):
        return "database"
    elif server_name.startswith("web"):
        return "web_api"
    elif server_name.startswith("ml"):
        return "ml_compute"
    # ... add your naming conventions
    else:
        return "generic"
```

### Step 5: Test Your Adapter

**Run in test mode first:**
```bash
python my_adapter.py --test --limit 10
```

**Expected output:**
```
✅ Connected to Elasticsearch: your-elk-server.company.com:9200
✅ Found 10 records in last 5 seconds
✅ Transformed to LINBORG format:
   Server: db01, CPU: 45.2%, Mem: 67.8%, Profile: database
   Server: web02, CPU: 23.1%, Mem: 45.3%, Profile: web_api
   ...
✅ Validation passed: All 14 metrics present
⚠️  DRY RUN - Not sending to inference daemon
```

### Step 6: Send First Batch to Inference Daemon

**Start adapter in live mode:**
```bash
python my_adapter.py --live
```

**Monitor the adapter logs:**
```
🚀 Adapter started - polling every 5 seconds
📡 Querying Elasticsearch: server-metrics-* (last 5 seconds)
📊 Retrieved 20 servers x 1 datapoint = 20 records
✅ POST /feed/data → 200 OK (147 records accepted, 0 rejected)
⏰ Warmup status: 12 servers warmed up, 8 still warming (need 150 records each)
```

**Warmup period:** The inference engine needs **150 records per server** before predictions start (about 12-15 minutes at 5-second intervals).

### Step 7: Verify Predictions in Dashboard

**Check inference daemon health:**
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "servers_tracked": 20,
  "servers_warmed_up": 20,
  "predictions_active": true,
  "last_update": "2025-10-30T15:30:00Z"
}
```

**Open dashboard:**
```
http://localhost:8501
```

You should now see **real server names** and **real risk scores** based on your production data!

---

## Data Source Adapters

### Elasticsearch Adapter Template

```python
#!/usr/bin/env python3
"""
Elasticsearch → NordIQ Adapter
Polls Elasticsearch for server metrics and feeds inference daemon
"""

import time
import requests
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta

# Configuration
ES_HOST = "localhost"
ES_PORT = 9200
ES_INDEX = "metricbeat-*"
INFERENCE_URL = "http://localhost:8000/feed/data"
API_KEY = "your-api-key"
POLL_INTERVAL = 5  # seconds

def query_elasticsearch(es_client, since_time):
    """Query Elasticsearch for metrics since given timestamp."""
    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": since_time.isoformat(),
                    "lt": datetime.now().isoformat()
                }
            }
        },
        "size": 10000,
        "sort": [{"@timestamp": "asc"}]
    }

    response = es_client.search(index=ES_INDEX, body=query)
    return response['hits']['hits']

def transform_to_linborg(es_record):
    """Transform Elasticsearch record to LINBORG format."""
    source = es_record['_source']

    return {
        "timestamp": source["@timestamp"],
        "server_name": source["host"]["name"],
        "profile": detect_profile(source["host"]["name"]),
        "cpu_user_pct": source.get("system", {}).get("cpu", {}).get("user", {}).get("pct", 0) * 100,
        "cpu_sys_pct": source.get("system", {}).get("cpu", {}).get("system", {}).get("pct", 0) * 100,
        "cpu_iowait_pct": source.get("system", {}).get("cpu", {}).get("iowait", {}).get("pct", 0) * 100,
        "cpu_idle_pct": source.get("system", {}).get("cpu", {}).get("idle", {}).get("pct", 0) * 100,
        "java_cpu_pct": source.get("java", {}).get("process", {}).get("cpu", {}).get("pct", 0) * 100,
        "mem_used_pct": source.get("system", {}).get("memory", {}).get("used", {}).get("pct", 0) * 100,
        "swap_used_pct": source.get("system", {}).get("swap", {}).get("used", {}).get("pct", 0) * 100,
        "disk_usage_pct": source.get("system", {}).get("filesystem", {}).get("used", {}).get("pct", 0) * 100,
        "net_in_mb_s": source.get("system", {}).get("network", {}).get("in", {}).get("bytes", 0) / 1024 / 1024,
        "net_out_mb_s": source.get("system", {}).get("network", {}).get("out", {}).get("bytes", 0) / 1024 / 1024,
        "back_close_wait": source.get("tcp", {}).get("close_wait", {}).get("backend", 0),
        "front_close_wait": source.get("tcp", {}).get("close_wait", {}).get("frontend", 0),
        "load_average": source.get("system", {}).get("load", {}).get("1", 0),
        "uptime_days": (datetime.now() - datetime.fromisoformat(source.get("system", {}).get("uptime", {}).get("boot_time", datetime.now().isoformat()))).days
    }

def detect_profile(server_name):
    """Detect server profile from hostname."""
    name_lower = server_name.lower()
    if "db" in name_lower or "database" in name_lower:
        return "database"
    elif "web" in name_lower or "api" in name_lower:
        return "web_api"
    elif "ml" in name_lower or "gpu" in name_lower:
        return "ml_compute"
    elif "etl" in name_lower or "kafka" in name_lower:
        return "data_ingest"
    else:
        return "generic"

def send_to_inference(records):
    """Send batch of records to inference daemon."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    payload = {"records": records}

    response = requests.post(INFERENCE_URL, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()

def main():
    """Main adapter loop."""
    es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])

    print(f"🚀 Elasticsearch adapter started")
    print(f"📡 Polling {ES_INDEX} every {POLL_INTERVAL} seconds")

    last_query_time = datetime.now() - timedelta(seconds=POLL_INTERVAL)

    while True:
        try:
            # Query Elasticsearch
            es_records = query_elasticsearch(es, last_query_time)
            last_query_time = datetime.now()

            if not es_records:
                print(f"⏰ No new records (last {POLL_INTERVAL}s)")
                time.sleep(POLL_INTERVAL)
                continue

            # Transform to LINBORG format
            linborg_records = [transform_to_linborg(rec) for rec in es_records]

            # Send to inference daemon
            result = send_to_inference(linborg_records)

            print(f"✅ Sent {len(linborg_records)} records → {result.get('accepted', 0)} accepted")

            time.sleep(POLL_INTERVAL)

        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
```

**Save as:** `my_elasticsearch_adapter.py`

**Run:**
```bash
python my_elasticsearch_adapter.py
```

---

## Verification & Testing

### Check Inference Daemon is Receiving Data

```bash
# Check warmup status
curl http://localhost:8000/status

# Expected:
{
  "servers_tracked": 25,
  "servers_warmed_up": 25,
  "total_records_received": 3750,
  "predictions_active": true
}
```

### Verify Predictions are Updating

```bash
# Get current predictions
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/current

# Check timestamps are recent (< 5 seconds old)
```

### Test Specific Server

```bash
# Get predictions for one server
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/db01

# Verify:
# - Risk score makes sense
# - Predictions array has 96 steps (8 hours)
# - Timestamps are in the future
```

---

## Troubleshooting

### Issue: "No records found in Elasticsearch"

**Cause:** Time range mismatch or incorrect index pattern

**Solution:**
```bash
# Check your index exists
curl -X GET "localhost:9200/_cat/indices?v" | grep metric

# Verify timestamp field
curl -X GET "localhost:9200/metricbeat-*/_mapping" | grep timestamp
```

### Issue: "Validation failed - missing metrics"

**Cause:** Your data doesn't have all 14 LINBORG metrics

**Solution:** See [Partial Metrics Support](#partial-metrics-support)

### Issue: "Predictions not starting after 30 minutes"

**Cause:** Warmup threshold not reached (need 150 records per server)

**Solution:**
```bash
# Check warmup status per server
curl http://localhost:8000/debug/warmup

# Increase poll frequency temporarily (catch up faster)
# In your adapter: POLL_INTERVAL = 1  # 1 second
```

### Issue: "Risk scores stuck at 50"

**Cause:** Model not trained on your server profiles or real data

**Solution:** Retrain model with production data (see Prerequisites)

---

## Partial Metrics Support

If you don't have all 14 LINBORG metrics, you have options:

### Option 1: Provide Default Values

```python
def transform_to_linborg(source):
    return {
        "timestamp": source["timestamp"],
        "server_name": source["hostname"],
        "cpu_user_pct": source.get("cpu_user", 0),  # Available
        "cpu_sys_pct": source.get("cpu_sys", 0),    # Available
        "cpu_iowait_pct": 0,  # Not available - default to 0
        "cpu_idle_pct": 100 - source.get("cpu_user", 0) - source.get("cpu_sys", 0),
        "java_cpu_pct": 0,  # Not tracked - default to 0
        "mem_used_pct": source.get("memory_used", 0),  # Available
        # ... etc
    }
```

### Option 2: Estimate Missing Metrics

```python
# Estimate java_cpu from total CPU if not available
java_cpu_pct = max(0, cpu_user_pct - 20)  # Rough estimate

# Estimate TCP connections from other metrics
back_close_wait = int(net_in_mb_s * 10)  # Rough correlation
```

### Option 3: Retrain Model Without Missing Metrics

If certain metrics are never available, retrain your model without them:

1. Edit `NordIQ/src/generators/metrics_generator.py`
2. Remove metrics you don't have from `NORDIQ_METRICS`
3. Retrain model with `_Starthere.ipynb`

---

## Production Checklist

Before going live with real data:

- [ ] ✅ Model trained on 30+ days of production data
- [ ] ✅ Model knows your server profiles (db, web, ml, etc.)
- [ ] ✅ All 14 LINBORG metrics available (or defaults configured)
- [ ] ✅ Adapter tested in dry-run mode (--test flag)
- [ ] ✅ First live batch sent successfully
- [ ] ✅ Warmup completed (150 records/server received)
- [ ] ✅ Predictions visible in dashboard
- [ ] ✅ Risk scores make sense (not stuck at 50)
- [ ] ✅ Timestamps updating every 5 seconds
- [ ] ✅ Error monitoring configured (adapter logs)
- [ ] ✅ Fallback plan if adapter fails (restart script)
- [ ] ✅ Demo metrics generator disabled

---

## Next Steps

Once real data is flowing:

1. **Monitor adapter logs** - Set up log aggregation for your adapter
2. **Set up alerting** - Configure alerts for adapter failures
3. **Optimize poll frequency** - Tune based on your data volume
4. **Scale horizontally** - Run multiple adapters for high availability
5. **Automate restarts** - Use systemd/supervisor for production resilience

**See Also:**
- [Data Ingestion Guide](DATA_INGESTION_GUIDE.md) - Complete /feed/data API specification
- [Scaling Guide](SCALING_GUIDE.md) - Multi-server, high availability
- [Monitoring Guide](MONITORING_METRICS.md) - Production observability

---

**Questions?** See [Troubleshooting Guide](../operations/TROUBLESHOOTING.md) or contact support.
