# Production Integration Guide
**TFT Inference Daemon - Production Data Integration**

**Version:** 2.0
**Date:** October 13, 2025
**Audience:** DevOps Engineers, SREs, Data Engineers

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Data Contract Specification](#data-contract-specification)
4. [API Endpoints](#api-endpoints)
5. [Data Format Examples](#data-format-examples)
6. [Integration Patterns](#integration-patterns)
7. [Language-Specific Examples](#language-specific-examples)
8. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
9. [Production Checklist](#production-checklist)

---

## Overview

The TFT Inference Daemon accepts server metrics via a REST API and generates 8-hour predictions for incident forecasting. This guide shows you how to send production log data to the daemon.

### Architecture

```
Your Production Logs → Data Transformation → HTTP POST → Inference Daemon → Predictions
                                                ↓
                                         Dashboard (Streamlit)
```

### Key Requirements

- **Data Format:** JSON array of metric records
- **Endpoint:** `POST http://localhost:8000/feed/data`
- **Frequency:** Every 5 seconds (recommended)
- **Warmup:** 150 records per server needed before predictions start

---

## Quick Start

### 1. Start the Inference Daemon

```bash
python tft_inference_daemon.py
```

The daemon will:
- Load the TFT model from `models/tft_model_*/`
- Listen on port 8000
- Wait for data feed

### 2. Send Test Data

```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "timestamp": "2025-10-13T16:00:00",
        "server_name": "ppdb001",
        "cpu_pct": 45.2,
        "mem_pct": 67.8,
        "disk_io_mb_s": 123.4,
        "latency_ms": 12.5,
        "state": "healthy"
      }
    ]
  }'
```

### 3. Check Status

```bash
curl http://localhost:8000/status
```

Response:
```json
{
  "running": true,
  "tick_count": 1,
  "window_size": 1,
  "warmup": {
    "is_warmed_up": false,
    "progress_percent": 0,
    "threshold": 150,
    "current_size": 1,
    "message": "Warming up: 0/1 servers ready"
  }
}
```

### 4. Get Predictions (After Warmup)

```bash
curl http://localhost:8000/predictions/current
```

---

## Data Contract Specification

### Required Fields

| Field | Type | Description | Valid Range | Example |
|-------|------|-------------|-------------|---------|
| `timestamp` | ISO 8601 string | Metric collection time | Any valid datetime | `"2025-10-13T16:00:00"` |
| `server_name` | string | Unique server identifier | Must match training data | `"ppdb001"` |
| `cpu_pct` | float | CPU utilization percentage | 0.0 - 100.0 | `45.2` |
| `mem_pct` | float | Memory utilization percentage | 0.0 - 100.0 | `67.8` |
| `disk_io_mb_s` | float | Disk I/O throughput (MB/s) | 0.0 - ∞ | `123.4` |
| `latency_ms` | float | Request latency (milliseconds) | 0.0 - ∞ | `12.5` |
| `state` | string | Current operational state | See valid states below | `"healthy"` |

### Valid States

```python
VALID_STATES = [
    'critical_issue',  # Active incident, requires immediate attention
    'healthy',         # Normal operations
    'heavy_load',      # High utilization but stable
    'idle',            # Low activity
    'maintenance',     # Planned maintenance window
    'morning_spike',   # Expected morning traffic surge
    'offline',         # Server unavailable
    'recovery'         # Recovering from incident
]
```

### Server Naming Convention

The model was trained with these server prefixes:
- `ppml####` - ML Compute (e.g., ppml0001, ppml0002)
- `ppdb###` - Database (e.g., ppdb001, ppdb002, ppdb003)
- `ppweb###` - Web API (e.g., ppweb001 through ppweb008)
- `ppcon##` - Conductor Management (e.g., ppcon01)
- `ppetl###` - Data Ingest/ETL (e.g., ppetl001, ppetl002)
- `pprisk###` - Risk Analytics (e.g., pprisk001)

**Important:** Use the exact server names from training, or predictions will fall back to heuristics.

---

## API Endpoints

### POST /feed/data

**Purpose:** Submit server metrics for prediction

**Request:**
```json
{
  "records": [
    {
      "timestamp": "2025-10-13T16:00:00",
      "server_name": "ppdb001",
      "cpu_pct": 45.2,
      "mem_pct": 67.8,
      "disk_io_mb_s": 123.4,
      "latency_ms": 12.5,
      "state": "healthy"
    },
    {
      "timestamp": "2025-10-13T16:00:00",
      "server_name": "ppdb002",
      "cpu_pct": 52.1,
      "mem_pct": 71.3,
      "disk_io_mb_s": 98.7,
      "latency_ms": 15.8,
      "state": "healthy"
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "status": "accepted",
  "tick": 42,
  "window_size": 840,
  "servers_tracked": 20,
  "servers_ready": 20,
  "warmup_complete": true
}
```

**Notes:**
- Send all servers in a single batch (recommended)
- Can send individual servers, but batching is more efficient
- Frequency: every 5 seconds (matches training data)

---

### GET /predictions/current

**Purpose:** Retrieve latest predictions for all servers

**Response (200 OK):**
```json
{
  "predictions": {
    "ppdb001": {
      "cpu_percent": {
        "p50": [45.3, 46.1, 47.2, ...],  // 96 values (8 hours)
        "p10": [42.1, 43.0, 44.2, ...],  // Optimistic forecast
        "p90": [48.5, 49.2, 50.3, ...],  // Pessimistic forecast
        "current": 45.2,
        "trend": 0.02
      },
      "memory_percent": { ... },
      "disk_percent": { ... },
      "load_average": { ... }
    },
    "ppdb002": { ... }
  },
  "alerts": [
    {
      "server": "ppdb001",
      "metric": "cpu_percent",
      "severity": "warning",
      "predicted_value": 92.3,
      "threshold": 90.0,
      "steps_ahead": 12,
      "minutes_ahead": 60,
      "message": "ppdb001: cpu_percent predicted to reach 92.3"
    }
  ],
  "environment": {
    "prob_30m": 0.15,  // 15% chance of incident in next 30 minutes
    "prob_8h": 0.42,   // 42% chance of incident in next 8 hours
    "high_risk_servers": 3,
    "total_servers": 20,
    "fleet_health": "warning"
  },
  "metadata": {
    "model_type": "TFT",
    "prediction_time": "2025-10-13T16:30:15.123456",
    "horizon_steps": 96,
    "device": "cuda:0"
  }
}
```

---

### GET /status

**Purpose:** Check daemon health and warmup status

**Response:**
```json
{
  "running": true,
  "tick_count": 305,
  "window_size": 6000,
  "warmup": {
    "is_warmed_up": true,
    "progress_percent": 100,
    "threshold": 150,
    "current_size": 289,
    "required_size": 150,
    "message": "Model ready - using TFT predictions"
  }
}
```

---

### GET /health

**Purpose:** Simple health check for load balancers

**Response:**
```json
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

---

## Data Format Examples

### Example 1: Single Server, Single Record

```json
{
  "records": [
    {
      "timestamp": "2025-10-13T16:00:00.000Z",
      "server_name": "ppweb001",
      "cpu_pct": 35.4,
      "mem_pct": 55.2,
      "disk_io_mb_s": 45.6,
      "latency_ms": 8.3,
      "state": "healthy"
    }
  ]
}
```

### Example 2: Multiple Servers (Batch - Recommended)

```json
{
  "records": [
    {
      "timestamp": "2025-10-13T16:00:00",
      "server_name": "ppml0001",
      "cpu_pct": 78.5,
      "mem_pct": 82.1,
      "disk_io_mb_s": 234.5,
      "latency_ms": 45.2,
      "state": "heavy_load"
    },
    {
      "timestamp": "2025-10-13T16:00:00",
      "server_name": "ppml0002",
      "cpu_pct": 81.2,
      "mem_pct": 79.8,
      "disk_io_mb_s": 198.7,
      "latency_ms": 52.1,
      "state": "heavy_load"
    },
    {
      "timestamp": "2025-10-13T16:00:00",
      "server_name": "ppdb001",
      "cpu_pct": 45.2,
      "mem_pct": 67.8,
      "disk_io_mb_s": 123.4,
      "latency_ms": 12.5,
      "state": "healthy"
    }
  ]
}
```

### Example 3: Critical Incident

```json
{
  "records": [
    {
      "timestamp": "2025-10-13T16:05:30",
      "server_name": "ppweb003",
      "cpu_pct": 98.7,
      "mem_pct": 94.2,
      "disk_io_mb_s": 567.8,
      "latency_ms": 234.5,
      "state": "critical_issue"
    }
  ]
}
```

### Example 4: Maintenance Window

```json
{
  "records": [
    {
      "timestamp": "2025-10-13T02:00:00",
      "server_name": "ppdb002",
      "cpu_pct": 5.2,
      "mem_pct": 45.1,
      "disk_io_mb_s": 12.3,
      "latency_ms": 0.0,
      "state": "maintenance"
    }
  ]
}
```

---

## Integration Patterns

### Pattern 1: Log File Tailing (Recommended)

Best for: Existing log infrastructure with file-based metrics

```bash
#!/bin/bash
# Tail server metrics log and POST to inference daemon

tail -F /var/log/server-metrics.log | while read line; do
  # Parse log line (assumes JSON format)
  timestamp=$(echo $line | jq -r '.timestamp')
  server=$(echo $line | jq -r '.hostname')
  cpu=$(echo $line | jq -r '.cpu')
  mem=$(echo $line | jq -r '.memory')

  # Transform to inference format
  payload=$(cat <<EOF
{
  "records": [{
    "timestamp": "$timestamp",
    "server_name": "$server",
    "cpu_pct": $cpu,
    "mem_pct": $mem,
    "disk_io_mb_s": 0.0,
    "latency_ms": 0.0,
    "state": "healthy"
  }]
}
EOF
)

  # POST to inference daemon
  curl -X POST http://localhost:8000/feed/data \
    -H "Content-Type: application/json" \
    -d "$payload"
done
```

---

### Pattern 2: Prometheus/Grafana Export

Best for: Prometheus-based monitoring stacks

**Step 1:** Export Prometheus metrics to JSON

```python
# prometheus_to_inference.py
import requests
import time
from datetime import datetime

PROMETHEUS_URL = "http://prometheus:9090"
INFERENCE_URL = "http://localhost:8000/feed/data"

def query_prometheus(metric, server):
    """Query Prometheus for a specific metric."""
    query = f'{metric}{{instance=~"{server}.*"}}'
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
    return resp.json()['data']['result'][0]['value'][1]

def collect_and_send():
    """Collect metrics and send to inference daemon."""
    servers = ['ppdb001', 'ppdb002', 'ppdb003']  # Your server list

    records = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    for server in servers:
        try:
            cpu = float(query_prometheus('node_cpu_usage', server))
            mem = float(query_prometheus('node_memory_usage', server))
            disk_io = float(query_prometheus('node_disk_io_bytes', server)) / 1024 / 1024
            latency = float(query_prometheus('http_request_duration_ms', server))

            records.append({
                "timestamp": timestamp,
                "server_name": server,
                "cpu_pct": cpu,
                "mem_pct": mem,
                "disk_io_mb_s": disk_io,
                "latency_ms": latency,
                "state": "healthy"  # Derive from metrics or external source
            })
        except Exception as e:
            print(f"Error collecting {server}: {e}")

    if records:
        resp = requests.post(INFERENCE_URL, json={"records": records})
        print(f"Sent {len(records)} records: {resp.json()}")

if __name__ == "__main__":
    while True:
        collect_and_send()
        time.sleep(5)  # Every 5 seconds
```

---

### Pattern 3: Direct Integration with Monitoring Agent

Best for: Custom monitoring agents or instrumented applications

```python
# Example: Collectd plugin or Telegraf exec plugin
import json
import sys
from datetime import datetime
import requests

def collect_metrics():
    """Collect metrics from local system."""
    import psutil

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "server_name": "ppweb001",  # From hostname or config
        "cpu_pct": psutil.cpu_percent(interval=1),
        "mem_pct": psutil.virtual_memory().percent,
        "disk_io_mb_s": psutil.disk_io_counters().read_bytes / 1024 / 1024,
        "latency_ms": 0.0,  # Measure from actual requests
        "state": "healthy"
    }

def send_to_inference(record):
    """Send metrics to inference daemon."""
    try:
        resp = requests.post(
            "http://localhost:8000/feed/data",
            json={"records": [record]},
            timeout=2
        )
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to send: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    while True:
        metrics = collect_metrics()
        result = send_to_inference(metrics)
        if result:
            print(f"Sent: {result['tick']}")
        time.sleep(5)
```

---

### Pattern 4: Batch Processing from Time-Series Database

Best for: InfluxDB, TimescaleDB, or other TSDB

```python
# influxdb_batch_export.py
from influxdb_client import InfluxDBClient
import requests
from datetime import datetime, timedelta

INFLUX_URL = "http://influxdb:8086"
INFLUX_TOKEN = "your-token"
INFLUX_ORG = "your-org"
INFLUX_BUCKET = "server-metrics"
INFERENCE_URL = "http://localhost:8000/feed/data"

def fetch_latest_metrics():
    """Fetch last 5 seconds of metrics from InfluxDB."""
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -5s)
      |> filter(fn: (r) => r["_measurement"] == "server_metrics")
      |> pivot(rowKey:["_time", "host"], columnKey: ["_field"], valueColumn: "_value")
    '''

    tables = query_api.query(query)

    records = []
    for table in tables:
        for record in table.records:
            records.append({
                "timestamp": record.get_time().isoformat(),
                "server_name": record.values.get("host"),
                "cpu_pct": record.values.get("cpu", 0.0),
                "mem_pct": record.values.get("memory", 0.0),
                "disk_io_mb_s": record.values.get("disk_io", 0.0),
                "latency_ms": record.values.get("latency", 0.0),
                "state": "healthy"
            })

    return records

def send_batch(records):
    """Send batch to inference daemon."""
    if not records:
        return

    resp = requests.post(INFERENCE_URL, json={"records": records})
    print(f"Sent {len(records)} records: {resp.json()}")

if __name__ == "__main__":
    import time
    while True:
        records = fetch_latest_metrics()
        send_batch(records)
        time.sleep(5)
```

---

## Language-Specific Examples

### Python

```python
import requests
from datetime import datetime

def send_metrics(server_name, cpu, memory, disk_io, latency, state="healthy"):
    """Send metrics to inference daemon."""
    payload = {
        "records": [{
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "server_name": server_name,
            "cpu_pct": cpu,
            "mem_pct": memory,
            "disk_io_mb_s": disk_io,
            "latency_ms": latency,
            "state": state
        }]
    }

    response = requests.post(
        "http://localhost:8000/feed/data",
        json=payload,
        timeout=2
    )

    return response.json()

# Usage
result = send_metrics("ppdb001", 45.2, 67.8, 123.4, 12.5)
print(f"Tick: {result['tick']}, Warmup: {result['warmup_complete']}")
```

---

### Bash/cURL

```bash
#!/bin/bash
# send_metrics.sh

SERVER="ppdb001"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -d "{
    \"records\": [{
      \"timestamp\": \"$TIMESTAMP\",
      \"server_name\": \"$SERVER\",
      \"cpu_pct\": 45.2,
      \"mem_pct\": 67.8,
      \"disk_io_mb_s\": 123.4,
      \"latency_ms\": 12.5,
      \"state\": \"healthy\"
    }]
  }"
```

---

### Node.js

```javascript
// send_metrics.js
const axios = require('axios');

async function sendMetrics(serverName, cpu, memory, diskIO, latency, state = 'healthy') {
  const payload = {
    records: [{
      timestamp: new Date().toISOString(),
      server_name: serverName,
      cpu_pct: cpu,
      mem_pct: memory,
      disk_io_mb_s: diskIO,
      latency_ms: latency,
      state: state
    }]
  };

  try {
    const response = await axios.post('http://localhost:8000/feed/data', payload, {
      timeout: 2000
    });
    console.log(`Sent: tick ${response.data.tick}`);
    return response.data;
  } catch (error) {
    console.error('Failed to send metrics:', error.message);
    return null;
  }
}

// Usage
sendMetrics('ppdb001', 45.2, 67.8, 123.4, 12.5);
```

---

### Go

```go
// send_metrics.go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

type MetricRecord struct {
	Timestamp   string  `json:"timestamp"`
	ServerName  string  `json:"server_name"`
	CPUPct      float64 `json:"cpu_pct"`
	MemPct      float64 `json:"mem_pct"`
	DiskIOMBs   float64 `json:"disk_io_mb_s"`
	LatencyMS   float64 `json:"latency_ms"`
	State       string  `json:"state"`
}

type FeedRequest struct {
	Records []MetricRecord `json:"records"`
}

func sendMetrics(serverName string, cpu, memory, diskIO, latency float64) error {
	record := MetricRecord{
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
		ServerName: serverName,
		CPUPct:     cpu,
		MemPct:     memory,
		DiskIOMBs:  diskIO,
		LatencyMS:  latency,
		State:      "healthy",
	}

	payload := FeedRequest{Records: []MetricRecord{record}}
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	resp, err := http.Post(
		"http://localhost:8000/feed/data",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	fmt.Printf("Sent metrics: %s\n", resp.Status)
	return nil
}

func main() {
	sendMetrics("ppdb001", 45.2, 67.8, 123.4, 12.5)
}
```

---

### PowerShell

```powershell
# send_metrics.ps1

function Send-Metrics {
    param(
        [string]$ServerName,
        [double]$CPU,
        [double]$Memory,
        [double]$DiskIO,
        [double]$Latency,
        [string]$State = "healthy"
    )

    $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    $payload = @{
        records = @(
            @{
                timestamp = $timestamp
                server_name = $ServerName
                cpu_pct = $CPU
                mem_pct = $Memory
                disk_io_mb_s = $DiskIO
                latency_ms = $Latency
                state = $State
            }
        )
    } | ConvertTo-Json -Depth 3

    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/feed/data" `
                                      -Method Post `
                                      -Body $payload `
                                      -ContentType "application/json" `
                                      -TimeoutSec 2

        Write-Host "Sent: tick $($response.tick)"
        return $response
    }
    catch {
        Write-Error "Failed to send metrics: $_"
        return $null
    }
}

# Usage
Send-Metrics -ServerName "ppdb001" -CPU 45.2 -Memory 67.8 -DiskIO 123.4 -Latency 12.5
```

---

## Monitoring & Troubleshooting

### Check Daemon Status

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status | jq

# Check if warmed up
curl http://localhost:8000/status | jq '.warmup.is_warmed_up'
```

### Monitor Warmup Progress

```bash
#!/bin/bash
# monitor_warmup.sh

while true; do
  STATUS=$(curl -s http://localhost:8000/status)
  WARMED_UP=$(echo $STATUS | jq -r '.warmup.is_warmed_up')
  PROGRESS=$(echo $STATUS | jq -r '.warmup.progress_percent')
  MESSAGE=$(echo $STATUS | jq -r '.warmup.message')

  clear
  echo "========================================="
  echo "TFT Inference Daemon - Warmup Monitor"
  echo "========================================="
  echo "Progress: $PROGRESS%"
  echo "Status: $MESSAGE"
  echo "Warmed Up: $WARMED_UP"
  echo "========================================="

  if [ "$WARMED_UP" = "true" ]; then
    echo "✅ Model is ready for predictions!"
    break
  fi

  sleep 2
done
```

### Common Issues

#### Issue 1: "insufficient_data" Error

**Symptom:**
```json
{
  "error": "insufficient_data",
  "message": "Need at least 100 records, have 42"
}
```

**Solution:** Wait for more data to accumulate. The daemon needs at least 100 total records before making predictions.

---

#### Issue 2: Warmup Not Progressing

**Symptom:**
```json
{
  "warmup": {
    "is_warmed_up": false,
    "progress_percent": 0,
    "current_size": 45,
    "required_size": 150
  }
}
```

**Solution:** Each server needs 150 individual records. Keep sending data every 5 seconds. Warmup takes ~12.5 minutes per server at 5-second intervals.

---

#### Issue 3: Predictions Falling Back to Heuristics

**Symptom:** Logs show `[WARNING] TFT prediction failed: ...` and falls back to heuristic predictions.

**Possible Causes:**
1. **Unknown server name** - Server not in training data
2. **Insufficient history** - Not enough consecutive records
3. **Data quality issues** - NaN values, out-of-range values

**Solution:**
- Verify server names match training data exactly (case-sensitive)
- Check `/status` endpoint to ensure `warmup_complete: true`
- Validate data contract compliance (see Data Contract Specification)

---

#### Issue 4: High Latency or Timeouts

**Symptom:** POST requests taking >2 seconds or timing out

**Possible Causes:**
1. GPU busy with prediction
2. Too many records in single batch
3. Network issues

**Solution:**
- Batch size: Keep batches under 50 servers per request
- Frequency: Send every 5 seconds (don't send faster)
- Timeout: Use 5-second timeout on HTTP client
- Check GPU utilization: `nvidia-smi` (if using GPU)

---

### Logging and Debugging

#### Enable Debug Logging

The daemon outputs structured logs to stdout:

```
[INFO] Loading persisted rolling window from inference_rolling_window.pkl...
[OK] Loaded 5700 records from disk
[OK] Warmup status: 20/20 servers ready
[SUCCESS] Model is WARMED UP - ready for predictions immediately!
[READY] Daemon started - waiting for data feed
[DEBUG] Input data: 5780 records, 20 unique servers
[DEBUG] Prepared data: 5780 records, 20 unique server_ids
[DEBUG] Prediction dataset created with 20 samples
[DEBUG] Running TFT model prediction...
[OK] TFT predictions generated for 20 servers
```

#### Capture Logs

```bash
# Redirect to file
python tft_inference_daemon.py > inference.log 2>&1

# Monitor in real-time
tail -f inference.log

# Search for errors
grep -i error inference.log
grep -i warning inference.log
```

---

## Production Checklist

### Pre-Deployment

- [ ] Verify data contract compliance (`python data_validator.py <your_data_file>`)
- [ ] Test with sample data using `curl` or Python script
- [ ] Confirm server names match training data exactly
- [ ] Test network connectivity to port 8000
- [ ] Verify TFT model loaded successfully (check startup logs)
- [ ] Confirm GPU available (if using GPU acceleration)

### Deployment

- [ ] Start daemon with persistence enabled (default)
- [ ] Monitor warmup progress until `warmup_complete: true`
- [ ] Configure data feed to send every 5 seconds
- [ ] Set up log rotation for daemon logs
- [ ] Configure monitoring/alerting for daemon health
- [ ] Test prediction retrieval from dashboard

### Post-Deployment

- [ ] Monitor prediction latency (<2 seconds)
- [ ] Verify all servers receiving predictions
- [ ] Check for "falling back to heuristic" warnings
- [ ] Monitor disk space for rolling window persistence file
- [ ] Set up automated restarts (systemd, supervisord, etc.)
- [ ] Document incident response procedures

### Monitoring Metrics

Track these metrics in your monitoring system:

| Metric | Threshold | Alert Level |
|--------|-----------|-------------|
| Daemon uptime | <99% | Warning |
| Prediction latency | >5s | Critical |
| Warmup status | `false` for >15 min | Warning |
| TFT fallback rate | >10% | Warning |
| POST request failures | >5% | Critical |
| Window size | <1000 records | Warning |

---

## Performance Considerations

### Throughput

- **Recommended:** 20 servers × 5-second intervals = 4 requests/second
- **Maximum:** ~100 requests/second (with batching)
- **Prediction latency:** 1-2 seconds (GPU), 3-5 seconds (CPU)

### Resource Usage

- **Memory:** ~2-4 GB (rolling window + model)
- **GPU VRAM:** ~1-2 GB (if using GPU)
- **Disk:** ~100 MB (rolling window persistence file)
- **CPU:** 10-20% (idle), 50-80% (during prediction)

### Scaling

For large deployments (>100 servers):

1. **Horizontal Scaling:** Run multiple inference daemons with load balancer
2. **Vertical Scaling:** Use larger GPU (V100, A100 for faster predictions)
3. **Batch Optimization:** Send all servers in single request
4. **Prediction Caching:** Cache predictions for 30 seconds, refresh in background

---

## Support and Resources

- **Model Training Guide:** See `MODEL_TRAINING_GUIDELINES.md`
- **Data Contract:** See `DATA_CONTRACT.md`
- **Architecture Overview:** See `HOW_PREDICTIONS_WORK.md`
- **Bug Reports:** File issues with sample data and logs
- **Performance Tuning:** See `gpu_profiles.py` for GPU optimization

---

## Example: Complete Integration Script

Here's a complete production-ready script:

```python
#!/usr/bin/env python3
"""
Production metrics forwarder for TFT Inference Daemon
Reads from your monitoring system and forwards to inference daemon.
"""

import requests
import time
import logging
from datetime import datetime
from typing import List, Dict
import sys

# Configuration
INFERENCE_URL = "http://localhost:8000/feed/data"
SEND_INTERVAL = 5  # seconds
TIMEOUT = 5  # seconds
MAX_RETRIES = 3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('metrics_forwarder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def collect_metrics_from_your_system() -> List[Dict]:
    """
    REPLACE THIS with your actual metrics collection logic.

    This should return a list of metric records in the inference format.
    """
    # Example: Query your monitoring system
    # This is a placeholder - integrate with your actual system

    servers = ['ppdb001', 'ppdb002', 'ppdb003']  # Your server list
    records = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    for server in servers:
        # Replace with actual metric collection
        # e.g., from Prometheus, InfluxDB, CloudWatch, etc.
        records.append({
            "timestamp": timestamp,
            "server_name": server,
            "cpu_pct": 45.0,  # Get from your system
            "mem_pct": 60.0,  # Get from your system
            "disk_io_mb_s": 100.0,  # Get from your system
            "latency_ms": 10.0,  # Get from your system
            "state": "healthy"  # Derive from your system
        })

    return records

def send_to_inference(records: List[Dict]) -> bool:
    """Send metrics to inference daemon with retry logic."""
    if not records:
        logger.warning("No records to send")
        return False

    payload = {"records": records}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                INFERENCE_URL,
                json=payload,
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"✓ Sent {len(records)} records | "
                    f"Tick: {result['tick']} | "
                    f"Warmup: {'✓' if result['warmup_complete'] else f\"{result['servers_ready']}/{result['servers_tracked']}\"}"
                )
                return True
            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            logger.error(f"Timeout on attempt {attempt}/{MAX_RETRIES}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed on attempt {attempt}/{MAX_RETRIES}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}/{MAX_RETRIES}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(1)  # Wait before retry

    logger.error(f"Failed to send after {MAX_RETRIES} attempts")
    return False

def check_daemon_health() -> bool:
    """Check if inference daemon is running."""
    try:
        response = requests.get(
            "http://localhost:8000/health",
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

def main():
    """Main loop."""
    logger.info("=" * 60)
    logger.info("TFT Inference Metrics Forwarder")
    logger.info("=" * 60)
    logger.info(f"Inference URL: {INFERENCE_URL}")
    logger.info(f"Send interval: {SEND_INTERVAL}s")
    logger.info("=" * 60)

    # Check daemon health before starting
    if not check_daemon_health():
        logger.error("Inference daemon is not responding!")
        logger.error("Start it with: python tft_inference_daemon.py")
        sys.exit(1)

    logger.info("✓ Inference daemon is healthy")

    consecutive_failures = 0
    max_consecutive_failures = 10

    try:
        while True:
            start_time = time.time()

            # Collect metrics from your system
            records = collect_metrics_from_your_system()

            # Send to inference daemon
            success = send_to_inference(records)

            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"Failed {consecutive_failures} times in a row. "
                        "Check daemon health and network connectivity."
                    )
                    # Optional: Exit or alert

            # Wait for next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, SEND_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Summary

This guide provides everything your team needs to integrate production log data with the TFT Inference Daemon:

1. **Data Contract:** Exact format requirements
2. **API Specification:** Complete endpoint documentation
3. **Integration Patterns:** Multiple approaches for different architectures
4. **Code Examples:** Ready-to-use snippets in 6 languages
5. **Troubleshooting:** Common issues and solutions
6. **Production Checklist:** Deployment and monitoring guidelines

For questions or issues, provide sample data and logs for faster troubleshooting.

---

**Version:** 2.0
**Last Updated:** October 13, 2025
**Maintained By:** ML Engineering Team
