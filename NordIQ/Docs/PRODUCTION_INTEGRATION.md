# Production Integration Guide
**Complete Guide to Real-World Data Ingestion and Integration**

**Version:** 2.0 (Merged Edition)
**Audience:** DevOps Engineers, SREs, Integration Developers, System Administrators
**Purpose:** Comprehensive technical specification for transitioning from demo data to production monitoring with complete integration patterns and deployment strategies

---

## Table of Contents

1. [Overview](#overview)
2. [Integration Options](#integration-options)
3. [Data Format Requirements](#data-format-requirements)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Data Source Adapters](#data-source-adapters)
6. [Testing and Validation](#testing-and-validation)
7. [Production Deployment](#production-deployment)
8. [Monitoring and Operations](#monitoring-and-operations)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The NordIQ system accepts server metrics via the **`POST /feed/data`** endpoint and uses a Temporal Fusion Transformer (TFT) model to generate risk predictions. This guide provides everything needed to:

1. **Understand the architecture** - From demo mode to production
2. **Choose your data source** - Elasticsearch, Prometheus, MongoDB, custom APIs, or file watchers
3. **Format your data** - Transform your metrics into the required LINBORG format
4. **Implement an adapter** - Connect your monitoring system to NordIQ
5. **Deploy and monitor** - Run in production with confidence

### Key Facts

- **Endpoint:** `POST http://localhost:8000/feed/data`
- **Content-Type:** `application/json`
- **Authentication:** Required (`X-API-Key` header)
- **Recommended Frequency:** Every 5 seconds
- **Batch Size:** 1-1000 records per request
- **Warmup Requirement:** 150 records per server before predictions start (~12-15 minutes)
- **Required Metrics:** 14 LINBORG metrics (CPU, Memory, Disk, Network, TCP, System)
- **Time to Integrate:** 1-2 hours for first implementation

### Architecture Overview

#### Demo Mode (Current State)
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
│ (Streamlit or Dash)     │ Visualizes predictions
└─────────────────────────┘
```

#### Production Mode (Target State)
```
┌─────────────────────────┐
│ Your Monitoring System  │ Elasticsearch, Prometheus, MongoDB, etc.
│ (Real Logs/Metrics)     │ Production data sources
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│ Data Adapter            │ Python script (you create)
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
│ (Production Monitoring) │ Shows real risk predictions
└─────────────────────────┘
```

---

## Integration Options

Choose the integration method that best fits your monitoring infrastructure:

### Option A: Elasticsearch / ELK Stack

**Best for:** Centralized logging systems, structured JSON logs

**Advantages:**
- Full-text search and aggregation capabilities
- Well-established indexing patterns
- Rich query language (Elasticsearch DSL)
- Supports complex filtering

**Requirements:**
- Elasticsearch 7.0+ or 8.0+
- Network access to ES cluster
- Index pattern with timestamp field

**Typical Servers:** Deployed in ELK stacks for log aggregation

**Setup Time:** 30-45 minutes

### Option B: Prometheus / Time-Series Data

**Best for:** Cloud-native, Kubernetes environments, time-series metrics

**Advantages:**
- Optimized for time-series data
- PromQL for flexible queries
- Native Kubernetes integration
- Extremely efficient storage

**Requirements:**
- Prometheus server running
- Metrics already collected
- PromQL familiarity

**Typical Servers:** Cloud platforms, microservices, container orchestration

**Setup Time:** 20-30 minutes

### Option C: MongoDB / NoSQL Store

**Best for:** Custom monitoring solutions, flexible schema

**Advantages:**
- Flexible document structure
- Aggregation pipeline capabilities
- Horizontal scalability
- No schema enforcement

**Requirements:**
- MongoDB 4.0+
- Write access to collection
- Proper indexing on timestamp

**Typical Servers:** Custom applications, multi-tenant systems

**Setup Time:** 30-40 minutes

### Option D: Custom REST API

**Best for:** Proprietary systems, existing monitoring APIs

**Advantages:**
- Works with any REST endpoint
- No additional infrastructure needed
- Language-agnostic

**Requirements:**
- HTTP endpoint for metrics retrieval
- API documentation
- Authentication mechanism (if any)

**Typical Servers:** Legacy systems, commercial monitoring tools

**Setup Time:** 40-60 minutes

### Option E: File Watcher / Log File Parser

**Best for:** Legacy systems, file-based logging

**Advantages:**
- No external dependencies
- Works with any text format
- Simple regex-based parsing

**Requirements:**
- Local or NFS access to log files
- Consistent log format
- Appropriate file permissions

**Typical Servers:** Older systems, custom file-based logging

**Setup Time:** 45-60 minutes

---

## Data Format Requirements

### The 14 LINBORG Metrics

Your production systems must provide these metrics (or equivalents):

#### CPU Metrics (5 fields)

| Field | Type | Range | Description | Examples |
|-------|------|-------|-------------|----------|
| `cpu_user_pct` | float | 0.0-100.0 | User CPU utilization percentage | `top`, Prometheus `node_cpu_seconds_total{mode="user"}` |
| `cpu_sys_pct` | float | 0.0-100.0 | System/kernel CPU percentage | `top`, Prometheus `node_cpu_seconds_total{mode="system"}` |
| `cpu_iowait_pct` | float | 0.0-100.0 | I/O Wait percentage (CRITICAL) | `iostat`, Prometheus `node_cpu_seconds_total{mode="iowait"}` |
| `cpu_idle_pct` | float | 0.0-100.0 | Idle CPU percentage | Calculated: 100 - (user+sys+iowait) |
| `java_cpu_pct` | float | 0.0-100.0 | Java process CPU percentage | `jstat`, JMX metrics, Application monitoring |

**Validation Rule:** `cpu_user_pct + cpu_sys_pct + cpu_iowait_pct + cpu_idle_pct ≈ 100` (±5% tolerance)

#### Memory Metrics (2 fields)

| Field | Type | Range | Description | Examples |
|-------|------|-------|-------------|----------|
| `mem_used_pct` | float | 0.0-100.0 | Memory utilization percentage | `free -m`, Prometheus `node_memory_MemAvailable_bytes` |
| `swap_used_pct` | float | 0.0-100.0 | Swap memory utilization | `free -m`, Prometheus `node_memory_SwapTotal_bytes` |

**Warnings:** Alert if `mem_used_pct > 95%` or `swap_used_pct > 10%`

#### Disk Metrics (1 field)

| Field | Type | Range | Description | Examples |
|-------|------|-------|-------------|----------|
| `disk_usage_pct` | float | 0.0-100.0 | Filesystem utilization percentage | `df -h`, Prometheus `node_filesystem_avail_bytes` |

#### Network Metrics (2 fields)

| Field | Type | Range | Description | Examples |
|-------|------|-------|-------------|----------|
| `net_in_mb_s` | float | 0.0-10000.0 | Inbound network throughput (MB/s) | `ifstat`, Prometheus `node_network_receive_bytes_total` |
| `net_out_mb_s` | float | 0.0-10000.0 | Outbound network throughput (MB/s) | `ifstat`, Prometheus `node_network_transmit_bytes_total` |

**Note:** Verify units are MB/s, not Mbps or other formats

#### TCP Connection Metrics (2 fields)

| Field | Type | Range | Description | Examples |
|-------|------|-------|-------------|----------|
| `back_close_wait` | integer | 0-100000 | Backend TCP CLOSE_WAIT count | `netstat`, Prometheus `node_netstat_Tcp_CurrEstab` |
| `front_close_wait` | integer | 0-100000 | Frontend TCP CLOSE_WAIT count | `netstat`, Application metrics |

**Warning:** Values > 1000 suggest connection leaks

#### System Metrics (2 fields)

| Field | Type | Range | Description | Examples |
|-------|------|-------|-------------|----------|
| `load_average` | float | 0.0-1000.0 | System load average (1-minute) | `uptime`, Prometheus `node_load1` |
| `uptime_days` | integer | 0-36500 | Days since last reboot | `uptime -s`, Prometheus `node_boot_time_seconds` |

### Data Contract v3.0

#### Required Metadata Fields

Every record MUST include these fields:

| Field | Type | Required | Description | Valid Format |
|-------|------|----------|-------------|-------------|
| `timestamp` | string | Yes | Measurement timestamp | ISO 8601 with timezone (e.g., `2025-10-30T15:30:00Z`) |
| `server_name` | string | Yes | Unique server identifier | 3-64 alphanumeric chars, `-`, `_`, `.` allowed |
| `profile` | string | Yes | Server type/profile | See Valid Profiles table below |

#### Valid Profiles

Your servers must be classified into these profiles:

| Profile | Description | Example Servers | Use Case |
|---------|-------------|-----------------|----------|
| `database` | Database servers (I/O intensive) | ppdb001, mongo-prod-3, postgres-1 | Primary data stores, heavily I/O bound |
| `web_api` | Web/API servers (network intensive) | ppweb001, api-gateway-2, lb-prod | HTTP services, REST APIs, load balancers |
| `ml_compute` | ML training nodes (CPU/GPU intensive) | ppml0001, gpu-node-5, tensor-1 | Model training, heavy computation |
| `conductor_mgmt` | Job scheduling, orchestration | ppcon01, k8s-master-1, airflow-1 | Job scheduling, container orchestration |
| `data_ingest` | ETL, streaming data (Kafka, Spark) | ppetl001, kafka-broker-3, spark-1 | Streaming, ETL, data pipelines |
| `risk_analytics` | Risk calculations, simulations | pprisk001, quant-server-2, calc-1 | Financial calculations, analytics |
| `generic` | Utility, monitoring, unknown | ppgen001, utility-box, monitor-1 | Any server not fitting above categories |

**Important:** Incorrect profile selection degrades prediction accuracy. Choose the profile matching the server's primary workload.

#### Optional Fields

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| `state` | string | Operational state hint | `healthy`, `degrading`, `stressed`, `critical`, `recovering`, `maintenance`, `offline`, `unknown` |
| `problem_child` | boolean | Problem server flag | `true` or `false` |
| `notes` | string | Free-form notes | Max 500 characters |

**Note:** If `state` is omitted, the inference engine auto-detects it from metrics.

### Timestamp Validation

**Valid formats:**
```
2025-10-30T15:30:00Z              # UTC
2025-10-30T15:30:00+00:00         # UTC with offset
2025-10-30T15:30:00.123Z          # With milliseconds
2025-10-30T10:30:00-05:00         # EST
```

**Invalid formats (will be rejected):**
```
2025-10-30 15:30:00               # Missing 'T'
10/30/2025 3:30 PM                # US format
1730300400                         # Unix timestamp
```

**Rules:**
- Must be ISO 8601 format
- Must include timezone (`Z` or `±HH:MM`)
- Can be past or present (future rejected)
- Max 1 hour old recommended (older data accepted for warmup only)

### Server Name Validation

**Valid server names:**
```
ppdb001              # Alphanumeric with numbers
web-server-01        # Hyphens allowed
ml_gpu_005           # Underscores allowed
k8s.prod.node3       # Dots allowed
```

**Invalid server names:**
```
server 01            # Spaces not allowed
db@prod              # Special characters not allowed
x                    # Too short (minimum 3 chars)
very-long-name-exceeding-64-characters-limit  # Too long
```

**Rules:**
- Length: 3-64 characters
- Allowed: `a-z`, `A-Z`, `0-9`, `-`, `_`, `.`
- Must be unique per server
- Case-sensitive (`Server01` ≠ `server01`)

### Request Body Structure

#### Minimal Valid Request

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "server01",
      "profile": "generic",
      "cpu_user_pct": 25.0,
      "cpu_sys_pct": 10.0,
      "cpu_iowait_pct": 2.0,
      "cpu_idle_pct": 63.0,
      "java_cpu_pct": 0.0,
      "mem_used_pct": 45.0,
      "swap_used_pct": 0.0,
      "disk_usage_pct": 60.0,
      "net_in_mb_s": 5.0,
      "net_out_mb_s": 3.0,
      "back_close_wait": 0,
      "front_close_wait": 0,
      "load_average": 1.5,
      "uptime_days": 10
    }
  ]
}
```

#### Typical Production Request

Multiple servers, single timestamp (5-second poll):

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppdb001",
      "profile": "database",
      "cpu_user_pct": 45.2,
      "cpu_sys_pct": 15.3,
      "cpu_iowait_pct": 18.5,
      "cpu_idle_pct": 21.0,
      "java_cpu_pct": 0.0,
      "mem_used_pct": 78.4,
      "swap_used_pct": 3.2,
      "disk_usage_pct": 82.1,
      "net_in_mb_s": 45.3,
      "net_out_mb_s": 38.7,
      "back_close_wait": 25,
      "front_close_wait": 12,
      "load_average": 8.5,
      "uptime_days": 45
    },
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppweb002",
      "profile": "web_api",
      "cpu_user_pct": 32.1,
      "cpu_sys_pct": 8.4,
      "cpu_iowait_pct": 1.2,
      "cpu_idle_pct": 58.3,
      "java_cpu_pct": 28.5,
      "mem_used_pct": 55.2,
      "swap_used_pct": 0.1,
      "disk_usage_pct": 45.8,
      "net_in_mb_s": 125.4,
      "net_out_mb_s": 98.3,
      "back_close_wait": 8,
      "front_close_wait": 15,
      "load_average": 3.2,
      "uptime_days": 28
    }
  ]
}
```

#### Batch Historical Request

For backfilling data with multiple timestamps:

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppdb001",
      "profile": "database",
      ...all 14 metrics...
    },
    {
      "timestamp": "2025-10-30T15:30:05Z",
      "server_name": "ppdb001",
      "profile": "database",
      ...all 14 metrics...
    },
    {
      "timestamp": "2025-10-30T15:30:10Z",
      "server_name": "ppdb001",
      "profile": "database",
      ...all 14 metrics...
    }
  ]
}
```

**Batching Rules:**
- Maximum 1000 records per request
- All timestamps should be within 1-hour window
- Older data (for warmup) is acceptable

---

## Step-by-Step Setup

### Prerequisites Checklist

Before starting integration:

- [ ] Model trained on data matching production server profiles
- [ ] All required metrics available from your monitoring system
- [ ] Network connectivity to inference daemon (port 8000)
- [ ] Python 3.7+ and required packages installed
- [ ] API key generated (see below)
- [ ] Access to your monitoring system (read permissions)

### Generating API Keys

```bash
# Show existing key
python NordIQ/bin/generate_api_key.py --show

# Generate new key
python NordIQ/bin/generate_api_key.py

# Revoke old key
python NordIQ/bin/generate_api_key.py --revoke <old-key>
```

Store the API key securely in your environment:
```bash
export NORDIQ_API_KEY="your-api-key-here"
```

### Step 1: Verify Model Training

**Critical:** Your TFT model must be trained on data matching your production environment.

```bash
# Review model training information
cat models/tft_model_*/training_info.json | grep -A 10 "unique_profiles"

# Expected output should show your server profiles:
# - database
# - web_api
# - ml_compute
# etc.
```

**If your model was trained on demo data only:**

1. Export 30 days of production logs
2. Run `NordIQ/_Starthere.ipynb` with your real data
3. Train a new model (see Training Guide in docs)
4. Deploy new model to inference daemon

### Step 2: Stop the Demo Metrics Generator

**Find the metrics generator:**
```bash
# Linux/Mac
ps aux | grep metrics_generator

# Windows
tasklist | findstr "metrics_generator"
```

**Stop it:**
```bash
# If using start_all scripts:
# Edit start_all.sh (Linux/Mac) or start_all.bat (Windows)
# Comment out the metrics generator line

# Or kill directly (replace <PID> with actual process ID):
kill <PID>           # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

**Verify it's stopped:**
```bash
curl http://localhost:8001/health
# Should fail with "connection refused"
```

### Step 3: Identify Your Data Source

Examine your monitoring infrastructure and choose one option:

**Questions to answer:**
- Where are your server metrics stored?
- What format are they in?
- How are they indexed/queried?
- What's the minimum query latency?

**Decision matrix:**

| Monitoring System | Best Option | Complexity |
|------------------|------------|-----------|
| ELK Stack (Elasticsearch) | Elasticsearch Adapter | Medium |
| Prometheus | Prometheus Adapter | Low |
| MongoDB | MongoDB Adapter | Medium |
| Custom REST API | Custom REST Adapter | Medium |
| Log files (syslog, etc.) | File Watcher Adapter | High |

### Step 4: Set Up Data Adapter

Choose your integration option and follow the appropriate section in [Data Source Adapters](#data-source-adapters) below.

### Step 5: Map Your Metrics to LINBORG Format

Define field mappings in your adapter:

```python
# Example: Elasticsearch field names → LINBORG names
FIELD_MAPPING = {
    "host.name":                    "server_name",
    "system.cpu.user.pct":          "cpu_user_pct",
    "system.cpu.system.pct":        "cpu_sys_pct",
    "system.cpu.iowait.pct":        "cpu_iowait_pct",
    "system.cpu.idle.pct":          "cpu_idle_pct",
    "java.process.cpu.pct":         "java_cpu_pct",
    "system.memory.used.pct":       "mem_used_pct",
    "system.swap.used.pct":         "swap_used_pct",
    "system.disk.used.pct":         "disk_usage_pct",
    "system.network.in.mbps":       "net_in_mb_s",
    "system.network.out.mbps":      "net_out_mb_s",
    "tcp.connections.close_wait.backend":  "back_close_wait",
    "tcp.connections.close_wait.frontend": "front_close_wait",
    "system.load.1m":               "load_average",
    "@timestamp":                   "timestamp",
}

# Define profile detection logic
def detect_profile(server_name):
    """Map server names to NordIQ profiles."""
    name_lower = server_name.lower()

    if "db" in name_lower or "database" in name_lower:
        return "database"
    elif "web" in name_lower or "api" in name_lower:
        return "web_api"
    elif "ml" in name_lower or "gpu" in name_lower:
        return "ml_compute"
    elif "etl" in name_lower or "kafka" in name_lower:
        return "data_ingest"
    elif "conductor" in name_lower or "scheduler" in name_lower:
        return "conductor_mgmt"
    elif "risk" in name_lower or "quant" in name_lower:
        return "risk_analytics"
    else:
        return "generic"
```

### Step 6: Test Your Adapter (Dry Run)

```bash
# Test mode (dry run - no data sent to inference)
python my_adapter.py --test --limit 10
```

**Expected output:**
```
Connected to Elasticsearch: your-elk-server.company.com:9200
Found 10 records in last 5 seconds
Transformed to LINBORG format:
   Server: db01, CPU: 45.2%, Mem: 67.8%, Profile: database
   Server: web02, CPU: 23.1%, Mem: 45.3%, Profile: web_api
   ...
Validation passed: All 14 metrics present
DRY RUN - Not sending to inference daemon
```

### Step 7: Deploy Adapter to Production

```bash
# Run in live mode (sends to inference)
python my_adapter.py --live

# Monitor logs
tail -f adapter.log
```

**Expected logs:**
```
2025-10-30 15:30:00 - Adapter started - polling every 5 seconds
2025-10-30 15:30:05 - Querying Elasticsearch: server-metrics-* (last 5 seconds)
2025-10-30 15:30:05 - Retrieved 20 servers x 1 datapoint = 20 records
2025-10-30 15:30:05 - POST /feed/data → 200 OK (147 records accepted, 0 rejected)
2025-10-30 15:30:05 - Warmup status: 12 servers warmed up, 8 still warming (need 150 records each)
```

**Warmup period:** The inference engine needs **150 records per server** before predictions start. At 5-second intervals, this takes approximately 12-15 minutes.

---

## Data Source Adapters

Choose and implement the adapter matching your monitoring system.

### Elasticsearch/ELK Stack Adapter

**Best for:** Centralized logging, structured JSON logs, log aggregation

**Installation:**
```bash
pip install elasticsearch>=7.0 python-dateutil requests
```

**Configuration:**
```python
# my_elasticsearch_adapter.py
ES_HOST = "your-elk-server.company.com"
ES_PORT = 9200
ES_INDEX = "metricbeat-*"  # or your index pattern
INFERENCE_URL = "http://localhost:8000/feed/data"
API_KEY = "your-api-key-here"
POLL_INTERVAL = 5  # seconds
```

**Full Implementation:**

```python
#!/usr/bin/env python3
"""
Elasticsearch → NordIQ Adapter
Polls Elasticsearch for server metrics and feeds inference daemon
"""

import time
import requests
import logging
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    try:
        response = es_client.search(index=ES_INDEX, body=query)
        return response['hits']['hits']
    except Exception as e:
        logger.error(f"Elasticsearch query failed: {e}")
        return []

def transform_to_linborg(es_record):
    """Transform Elasticsearch record to LINBORG format."""
    source = es_record['_source']

    # Helper function to safely extract nested values
    def get_nested(obj, path, default=0):
        """Safely get nested dictionary value."""
        keys = path.split('.')
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key, {})
            else:
                return default
        return obj if obj else default

    return {
        "timestamp": source.get("@timestamp", datetime.now().isoformat()),
        "server_name": source.get("host", {}).get("name", "unknown"),
        "profile": detect_profile(source.get("host", {}).get("name", "")),
        "cpu_user_pct": float(get_nested(source, "system.cpu.user.pct", 0)) * 100,
        "cpu_sys_pct": float(get_nested(source, "system.cpu.system.pct", 0)) * 100,
        "cpu_iowait_pct": float(get_nested(source, "system.cpu.iowait.pct", 0)) * 100,
        "cpu_idle_pct": float(get_nested(source, "system.cpu.idle.pct", 0)) * 100,
        "java_cpu_pct": float(get_nested(source, "java.process.cpu.pct", 0)) * 100,
        "mem_used_pct": float(get_nested(source, "system.memory.used.pct", 0)) * 100,
        "swap_used_pct": float(get_nested(source, "system.swap.used.pct", 0)) * 100,
        "disk_usage_pct": float(get_nested(source, "system.filesystem.used.pct", 0)) * 100,
        "net_in_mb_s": float(get_nested(source, "system.network.in.bytes", 0)) / 1024 / 1024,
        "net_out_mb_s": float(get_nested(source, "system.network.out.bytes", 0)) / 1024 / 1024,
        "back_close_wait": int(get_nested(source, "tcp.close_wait.backend", 0)),
        "front_close_wait": int(get_nested(source, "tcp.close_wait.frontend", 0)),
        "load_average": float(get_nested(source, "system.load.1", 0)),
        "uptime_days": int(get_nested(source, "system.uptime_days", 0))
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
    elif "conductor" in name_lower or "scheduler" in name_lower:
        return "conductor_mgmt"
    elif "risk" in name_lower or "quant" in name_lower:
        return "risk_analytics"
    else:
        return "generic"

def validate_record(record):
    """Validate and fix common metric issues."""
    # Clamp CPU percentages
    cpu_sum = (record.get("cpu_user_pct", 0) +
               record.get("cpu_sys_pct", 0) +
               record.get("cpu_iowait_pct", 0) +
               record.get("cpu_idle_pct", 0))

    if abs(cpu_sum - 100.0) > 5.0:  # Allow 5% tolerance
        # Normalize CPU metrics
        if cpu_sum > 0:
            scale = 100.0 / cpu_sum
            record["cpu_user_pct"] *= scale
            record["cpu_sys_pct"] *= scale
            record["cpu_iowait_pct"] *= scale
            record["cpu_idle_pct"] *= scale

    # Clamp percentage values
    for key in ["cpu_user_pct", "cpu_sys_pct", "cpu_iowait_pct", "cpu_idle_pct",
                "mem_used_pct", "swap_used_pct", "disk_usage_pct"]:
        record[key] = max(0.0, min(100.0, record.get(key, 0)))

    # Ensure non-negative
    for key in ["net_in_mb_s", "net_out_mb_s", "load_average"]:
        record[key] = max(0.0, record.get(key, 0))

    for key in ["back_close_wait", "front_close_wait", "uptime_days"]:
        record[key] = max(0, record.get(key, 0))

    return record

def send_to_inference(records):
    """Send batch of records to inference daemon."""
    if not records:
        return None

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    payload = {"records": records}

    try:
        response = requests.post(INFERENCE_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error("Request timeout - inference daemon not responding")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection failed - check inference daemon is running")
        return None
    except Exception as e:
        logger.error(f"Failed to send data: {e}")
        return None

def main():
    """Main adapter loop."""
    try:
        es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])
        logger.info(f"Starting Elasticsearch adapter")
        logger.info(f"Polling {ES_INDEX} every {POLL_INTERVAL} seconds")

        last_query_time = datetime.now() - timedelta(seconds=POLL_INTERVAL)

        while True:
            try:
                # Query Elasticsearch
                es_records = query_elasticsearch(es, last_query_time)
                last_query_time = datetime.now()

                if not es_records:
                    logger.debug(f"No new records (last {POLL_INTERVAL}s)")
                    time.sleep(POLL_INTERVAL)
                    continue

                # Transform to LINBORG format
                linborg_records = [transform_to_linborg(rec) for rec in es_records]

                # Validate records
                linborg_records = [validate_record(rec) for rec in linborg_records]

                # Send to inference daemon
                result = send_to_inference(linborg_records)

                if result:
                    logger.info(f"Sent {len(linborg_records)} records → "
                               f"{result.get('accepted', 0)} accepted, "
                               f"{result.get('rejected', 0)} rejected")

                time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Adapter stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(POLL_INTERVAL)

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")

if __name__ == "__main__":
    main()
```

**Run:**
```bash
python my_elasticsearch_adapter.py
```

### Prometheus Adapter

**Best for:** Kubernetes, cloud-native environments, already-collected metrics

**Installation:**
```bash
pip install prometheus-client requests
```

**Configuration:**
```python
PROMETHEUS_URL = "http://prometheus-server:9090"
INFERENCE_URL = "http://localhost:8000/feed/data"
API_KEY = "your-api-key-here"
POLL_INTERVAL = 5  # seconds
```

**PromQL Queries:**

```python
# CPU metrics
QUERIES = {
    "cpu_user_pct": 'rate(node_cpu_seconds_total{mode="user"}[1m])*100',
    "cpu_sys_pct": 'rate(node_cpu_seconds_total{mode="system"}[1m])*100',
    "cpu_iowait_pct": 'rate(node_cpu_seconds_total{mode="iowait"}[1m])*100',
    "cpu_idle_pct": 'rate(node_cpu_seconds_total{mode="idle"}[1m])*100',
    "mem_used_pct": '(1-node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes)*100',
    "disk_usage_pct": '(1-node_filesystem_avail_bytes/node_filesystem_size_bytes)*100',
    # ... etc
}
```

### MongoDB Adapter

**Best for:** Custom monitoring solutions, flexible schema storage

**Installation:**
```bash
pip install pymongo requests
```

**Configuration:**
```python
MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DB = "metrics"
MONGODB_COLLECTION = "server_metrics"
INFERENCE_URL = "http://localhost:8000/feed/data"
```

**MongoDB Aggregation Pipeline:**

```python
AGGREGATION_PIPELINE = [
    {
        "$match": {
            "timestamp": {
                "$gte": datetime.now() - timedelta(seconds=5)
            }
        }
    },
    {
        "$group": {
            "_id": "$server_name",
            "latest": {
                "$last": "$$ROOT"
            }
        }
    },
    {
        "$sort": {"_id": 1}
    }
]
```

### Custom REST API Adapter

**Best for:** Proprietary systems, existing monitoring APIs

**Template:**

```python
import requests
from datetime import datetime, timedelta

def query_custom_api(since_time):
    """Query your custom metrics API."""
    url = "https://your-monitoring-api.com/metrics"
    params = {
        "since": since_time.isoformat(),
        "limit": 10000
    }
    headers = {
        "Authorization": "Bearer your-token"
    }

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    return response.json()

def transform_to_linborg(api_record):
    """Transform your API format to LINBORG."""
    return {
        "timestamp": api_record["timestamp"],
        "server_name": api_record["hostname"],
        "profile": detect_profile(api_record["hostname"]),
        # ... map all 14 metrics
    }
```

### File Watcher Adapter

**Best for:** Legacy systems, file-based logging

**Template:**

```python
import re
from pathlib import Path
from datetime import datetime, timedelta

def tail_log_file(file_path, num_lines=100):
    """Read last N lines from log file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines[-num_lines:]

def parse_log_line(line):
    """Parse log line into metrics."""
    # Example: "2025-10-30T15:30:00Z server01 cpu=45.2 mem=67.8 ..."
    match = re.match(
        r'(\S+)\s+(\S+)\s+cpu=(\d+\.?\d*)\s+mem=(\d+\.?\d*)',
        line
    )

    if match:
        timestamp, server, cpu, mem = match.groups()
        return {
            "timestamp": timestamp,
            "server_name": server,
            "cpu_user_pct": float(cpu),
            # ... parse other fields
        }
    return None
```

---

## Testing and Validation

### API Endpoint Testing

#### Test with curl

```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "records": [
      {
        "timestamp": "2025-10-30T15:30:00Z",
        "server_name": "test-server",
        "profile": "generic",
        "cpu_user_pct": 25.0,
        "cpu_sys_pct": 10.0,
        "cpu_iowait_pct": 2.0,
        "cpu_idle_pct": 63.0,
        "java_cpu_pct": 0.0,
        "mem_used_pct": 45.0,
        "swap_used_pct": 0.0,
        "disk_usage_pct": 60.0,
        "net_in_mb_s": 5.0,
        "net_out_mb_s": 3.0,
        "back_close_wait": 0,
        "front_close_wait": 0,
        "load_average": 1.5,
        "uptime_days": 10
      }
    ]
  }'
```

#### Test with Python

```python
import requests
from datetime import datetime, timezone

def test_inference_endpoint():
    url = "http://localhost:8000/feed/data"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key"
    }

    payload = {
        "records": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server_name": "test-server",
                "profile": "generic",
                "cpu_user_pct": 25.0,
                "cpu_sys_pct": 10.0,
                "cpu_iowait_pct": 2.0,
                "cpu_idle_pct": 63.0,
                "java_cpu_pct": 0.0,
                "mem_used_pct": 45.0,
                "swap_used_pct": 0.0,
                "disk_usage_pct": 60.0,
                "net_in_mb_s": 5.0,
                "net_out_mb_s": 3.0,
                "back_close_wait": 0,
                "front_close_wait": 0,
                "load_average": 1.5,
                "uptime_days": 10
            }
        ]
    }

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    test_inference_endpoint()
```

### Response Validation

#### Success Response (200 OK)

```json
{
  "status": "success",
  "message": "Data ingestion successful",
  "accepted": 147,
  "rejected": 0,
  "servers_updated": 20,
  "timestamp": "2025-10-30T15:30:05Z",
  "warmup_status": {
    "total_servers": 20,
    "warmed_up": 18,
    "warming": 2,
    "warmup_threshold": 150
  }
}
```

#### Partial Success (200 OK with errors)

```json
{
  "status": "partial_success",
  "message": "Some records rejected due to validation errors",
  "accepted": 145,
  "rejected": 2,
  "servers_updated": 20,
  "timestamp": "2025-10-30T15:30:05Z",
  "errors": [
    {
      "record_index": 5,
      "server_name": "ppweb003",
      "error": "cpu_user_pct out of range: 105.5 (max 100.0)"
    },
    {
      "record_index": 12,
      "server_name": "ppdb007",
      "error": "Missing required field: mem_used_pct"
    }
  ]
}
```

#### Common Error Responses

**401 Unauthorized - Invalid/missing API key:**
```json
{
  "detail": "Invalid or missing API key"
}
```

**400 Bad Request - Invalid JSON:**
```json
{
  "status": "error",
  "error": "Validation failed",
  "details": "Request body must be valid JSON with 'records' array",
  "timestamp": "2025-10-30T15:30:05Z"
}
```

**413 Payload Too Large - Too many records:**
```json
{
  "status": "error",
  "error": "Payload too large",
  "details": "Maximum 1000 records per request (received 1500)",
  "timestamp": "2025-10-30T15:30:05Z"
}
```

### Health Check

```bash
# Check inference daemon health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "servers_tracked": 20,
  "servers_warmed_up": 20,
  "predictions_active": true,
  "last_update": "2025-10-30T15:30:00Z"
}
```

### Warmup Status Monitoring

```bash
# Check warmup progress
curl http://localhost:8000/status

# Response shows which servers are ready for predictions
{
  "servers_tracked": 25,
  "servers_warmed_up": 25,
  "total_records_received": 3750,
  "predictions_active": true,
  "warmup_details": {
    "ppdb001": {
      "records_received": 150,
      "status": "warmed_up"
    },
    "ppdb002": {
      "records_received": 75,
      "status": "warming"
    }
  }
}
```

### Prediction Verification

```bash
# Get current predictions for all servers
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/current

# Get predictions for specific server
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/ppdb001

# Expected response includes:
# - risk_score: Current risk (0-100)
# - predictions: Array of future risk scores (96 steps = 8 hours)
# - timestamp: When predictions were generated
```

---

## Production Deployment

### Environment Setup

**Create `.env` file:**
```bash
# Inference Daemon
INFERENCE_HOST=0.0.0.0
INFERENCE_PORT=8000
INFERENCE_API_KEY=your-secure-api-key

# Data Source Configuration (example for Elasticsearch)
ES_HOST=your-elk-server.company.com
ES_PORT=9200
ES_USER=readonly_user
ES_PASSWORD=secure_password
ES_INDEX=metricbeat-*

# Adapter Configuration
POLL_INTERVAL=5
MAX_BATCH_SIZE=100
ADAPTER_LOG_LEVEL=INFO

# Optional: Monitoring/Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
PAGERDUTY_KEY=your-pagerduty-key
```

### Systemd Service (Linux)

**Create `/etc/systemd/system/nordiq-adapter.service`:**

```ini
[Unit]
Description=NordIQ Metrics Adapter
After=network.target
Wants=nordiq-inference.service

[Service]
Type=simple
User=nordiq
WorkingDirectory=/opt/nordiq
Environment="PATH=/opt/nordiq/venv/bin"
EnvironmentFile=/etc/nordiq/.env
ExecStart=/opt/nordiq/venv/bin/python /opt/nordiq/adapters/elasticsearch_adapter.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable nordiq-adapter
sudo systemctl start nordiq-adapter

# Monitor logs
sudo journalctl -u nordiq-adapter -f
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy adapter
COPY adapters/elasticsearch_adapter.py .

# Run adapter
CMD ["python", "elasticsearch_adapter.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  inference:
    image: nordiq:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/tft_model
    volumes:
      - ./models:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  adapter:
    image: nordiq-adapter:latest
    depends_on:
      - inference
    environment:
      - ES_HOST=elasticsearch
      - ES_PORT=9200
      - INFERENCE_URL=http://inference:8000/feed/data
      - API_KEY=${NORDIQ_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nordiq-adapter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nordiq-adapter
  template:
    metadata:
      labels:
        app: nordiq-adapter
    spec:
      containers:
      - name: adapter
        image: nordiq-adapter:latest
        env:
        - name: INFERENCE_URL
          value: "http://nordiq-inference:8000/feed/data"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: nordiq-secrets
              key: api-key
        - name: ES_HOST
          value: "elasticsearch.default.svc.cluster.local"
        - name: ES_PORT
          value: "9200"
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## Monitoring and Operations

### Best Practices for Data Ingest

#### Polling Frequency

**Recommended: 5 seconds**

```python
POLL_INTERVAL = 5  # seconds

while True:
    records = fetch_metrics_from_source()
    send_to_inference(records)
    time.sleep(POLL_INTERVAL)
```

**Why 5 seconds?**
- Matches model training frequency
- Good balance between responsiveness and overhead
- TFT architecture optimized for this interval
- Standard in monitoring industry

**Alternatives:**
- **1 second:** High overhead, not necessary
- **10 seconds:** Acceptable, slightly delayed warmup
- **60 seconds:** Too slow, significant warmup delay

#### Batch Size

**Recommended: 20-100 servers per request**

```python
# Good - batch multiple servers
records = []
for server in get_active_servers():
    records.append(fetch_metrics(server))
send_to_inference(records)  # Single POST

# Bad - individual POSTs
for server in servers:
    record = fetch_metrics(server)
    send_to_inference([record])  # Don't do this!
```

**Benefits:**
- Reduces HTTP overhead
- Better throughput
- Atomic timestamp consistency
- Reduced load on inference daemon

#### Timestamp Consistency

```python
# Good - single query timestamp
current_time = datetime.now(timezone.utc)
records = []
for server in servers:
    records.append({
        "timestamp": current_time.isoformat(),
        "server_name": server,
        # ... metrics
    })

# Bad - different timestamps per server
for server in servers:
    record = {
        "timestamp": datetime.now().isoformat(),  # Don't do this!
        "server_name": server,
        # ... metrics
    }
```

#### Handling Offline Servers

```python
# Good - explicitly mark as offline
for server in all_servers:
    if is_online(server):
        record = fetch_metrics(server)
    else:
        # Create explicit offline record
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_name": server,
            "profile": get_profile(server),
            "state": "offline",
            # All metrics = 0
            "cpu_user_pct": 0.0,
            "cpu_sys_pct": 0.0,
            # ... etc
        }
    records.append(record)
```

### Adapter Monitoring

**Key metrics to track:**

```python
import time
import logging

logger = logging.getLogger(__name__)

class AdapterMetrics:
    def __init__(self):
        self.total_requests = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_errors = 0
        self.last_success_time = None
        self.start_time = time.time()

    def record_success(self, accepted, rejected):
        self.total_requests += 1
        self.total_accepted += accepted
        self.total_rejected += rejected
        self.last_success_time = datetime.now()

    def record_error(self):
        self.total_errors += 1

    def get_stats(self):
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_errors": self.total_errors,
            "acceptance_rate": (self.total_accepted / max(1, self.total_accepted + self.total_rejected)) * 100,
            "last_success": self.last_success_time.isoformat() if self.last_success_time else None
        }
```

### Alert Conditions

**Critical alerts (immediate action):**

1. **Adapter not running**
   - Check if process is alive
   - Restart service
   - Check logs for errors

2. **Data source unreachable**
   - Verify network connectivity
   - Check data source health
   - Verify authentication

3. **High rejection rate**
   - Rejection rate > 10%
   - Check validation errors in response
   - Review data quality from source

4. **Inference daemon not responding**
   - Check if daemon is running
   - Verify network connectivity
   - Check daemon logs

**Warning alerts (investigate within hour):**

1. **Warmup stuck**
   - Servers not reaching warmup threshold
   - Check poll frequency
   - Verify data source is returning records

2. **Slow response times**
   - Request latency > 5 seconds
   - Check network
   - Check inference daemon load

3. **Low data volume**
   - Fewer servers being tracked than expected
   - Check data source query
   - Verify field mappings

### Prometheus Metrics Export

```python
from prometheus_client import Counter, Gauge, Histogram

# Define metrics
records_sent = Counter('nordiq_adapter_records_sent_total', 'Total records sent')
records_accepted = Counter('nordiq_adapter_records_accepted_total', 'Total records accepted')
records_rejected = Counter('nordiq_adapter_records_rejected_total', 'Total records rejected')
request_latency = Histogram('nordiq_adapter_request_latency_seconds', 'Request latency')
servers_tracked = Gauge('nordiq_servers_tracked', 'Number of servers being tracked')
adapter_errors = Counter('nordiq_adapter_errors_total', 'Total adapter errors')

# Use in adapter
start_time = time.time()
try:
    result = send_to_inference(records)
    records_sent.inc(len(records))
    records_accepted.inc(result.get('accepted', 0))
    records_rejected.inc(result.get('rejected', 0))
finally:
    request_latency.observe(time.time() - start_time)
```

---

## Troubleshooting

### Issue: "No records found in data source"

**Symptoms:**
- Adapter logs show "No new records"
- Inference daemon shows 0 servers tracked
- No data flowing to inference engine

**Possible Causes:**
1. Query time window incorrect
2. Data source index/collection doesn't exist
3. Authentication failed silently
4. No matching data in source

**Solutions:**

```bash
# Test Elasticsearch connection
curl -X GET "localhost:9200/_cat/indices?v"

# Check index exists and has data
curl -X GET "localhost:9200/metricbeat-*/_count"

# Verify query returns data
curl -X POST "localhost:9200/metricbeat-*/_search" \
  -H "Content-Type: application/json" \
  -d '{"query": {"match_all": {}}, "size": 10}'

# Check timestamp format matches data
curl -X POST "localhost:9200/metricbeat-*/_search" \
  -H "Content-Type: application/json" \
  -d '{"_source": ["@timestamp"], "size": 1}'
```

**For Prometheus:**
```bash
# Test connection
curl http://localhost:9090/api/v1/query?query=up

# Test specific metric
curl http://localhost:9090/api/v1/query?query=node_cpu_seconds_total
```

### Issue: "Validation failed - missing metrics"

**Symptoms:**
- Response includes rejected records
- Error message mentions specific field
- Acceptance rate < 100%

**Possible Causes:**
1. Metric not available in data source
2. Field mapping incorrect
3. Metric has different name in source
4. Metric values are null/empty

**Solutions:**

```python
# Check what fields are in your data
def inspect_records(records, num_to_inspect=5):
    """Debug: Show all fields in first N records."""
    import json
    for i, record in enumerate(records[:num_to_inspect]):
        print(f"Record {i}: {json.dumps(record, indent=2)}")

# Verify all 14 metrics are present
def validate_linborg_record(record):
    """Check all required fields."""
    required = [
        "timestamp", "server_name", "profile",
        "cpu_user_pct", "cpu_sys_pct", "cpu_iowait_pct", "cpu_idle_pct",
        "java_cpu_pct", "mem_used_pct", "swap_used_pct",
        "disk_usage_pct", "net_in_mb_s", "net_out_mb_s",
        "back_close_wait", "front_close_wait", "load_average", "uptime_days"
    ]

    missing = [f for f in required if f not in record or record[f] is None]
    if missing:
        print(f"Missing fields in {record.get('server_name')}: {missing}")
        return False
    return True
```

**Handling partial metrics:**

```python
# Option 1: Provide sensible defaults
def transform_with_defaults(source):
    return {
        "timestamp": source.get("timestamp"),
        "server_name": source.get("hostname"),
        "cpu_user_pct": source.get("cpu_user", 0),      # Available
        "cpu_sys_pct": source.get("cpu_sys", 0),        # Available
        "cpu_iowait_pct": 0,                            # Not available
        "cpu_idle_pct": 100 - source.get("cpu_user", 0) - source.get("cpu_sys", 0),
        "java_cpu_pct": 0,                              # Not tracked
        # ... etc
    }

# Option 2: Skip servers missing critical metrics
if source.get("cpu_iowait_pct") is None:
    logger.warning(f"Skipping {source['hostname']} - missing cpu_iowait_pct")
    return None
```

### Issue: "Predictions not starting after 30+ minutes"

**Symptoms:**
- Warmup status stuck below threshold
- Predictions not generating
- Risk scores show "N/A" in dashboard

**Possible Causes:**
1. Poll frequency too slow
2. Data source queries return incomplete data
3. Server filtering removing most records
4. Model not trained on these server profiles

**Solutions:**

```bash
# Check warmup progress per server
curl http://localhost:8000/debug/warmup

# Expected: Shows record count per server
{
  "ppdb001": {"records": 150, "status": "warmed_up"},
  "ppdb002": {"records": 75, "status": "warming"},
  "ppweb001": {"records": 200, "status": "warmed_up"}
}
```

**Fixing warmup:**

```python
# Increase poll frequency temporarily
# In adapter: POLL_INTERVAL = 1  # Instead of 5
# This catches up faster: 150 records in 2.5 minutes instead of 12.5

# Or backfill historical data
def backfill_warmup_data():
    """Load historical data to speed up warmup."""
    # Query last 30 minutes of data
    since = datetime.now() - timedelta(minutes=30)
    historical = query_elasticsearch(since)

    # Send in large batches
    for i in range(0, len(historical), 1000):
        batch = historical[i:i+1000]
        send_to_inference(batch)
        time.sleep(0.5)
```

### Issue: "Risk scores stuck at 50"

**Symptoms:**
- All servers show risk = 50
- Predictions not varying
- Model not responding to metric changes

**Possible Causes:**
1. Model trained on demo data, not your profiles
2. Wrong model version deployed
3. Data not matching training data distribution
4. All metrics at same values

**Solutions:**

1. **Retrain model with production data:**
   ```bash
   # See documentation in _Starthere.ipynb
   # Export 30 days of production logs
   # Run training notebook with your data
   # Deploy new model
   ```

2. **Verify model knows your profiles:**
   ```bash
   cat models/tft_model_*/training_info.json | grep -A 20 "profiles"

   # Should show your server types:
   # - database
   # - web_api
   # etc.
   ```

3. **Check metric variance:**
   ```python
   # Verify metrics are actually changing
   predictions_response = requests.get(
       f"http://localhost:8000/predictions/{server_name}",
       headers={"X-API-Key": api_key}
   ).json()

   # Check if prediction scores vary
   scores = [p["risk"] for p in predictions_response["predictions"]]
   if len(set(scores)) == 1:
       print("WARNING: All predictions same value - model issue detected")
   ```

### Issue: "Connection refused / Inference daemon not running"

**Symptoms:**
- Adapter fails to send data
- `curl localhost:8000/health` fails
- Daemon logs not showing activity

**Possible Causes:**
1. Inference daemon process crashed
2. Port 8000 in use by another process
3. Network/firewall blocking
4. Model load failed during startup

**Solutions:**

```bash
# Check if daemon is running
ps aux | grep inference_daemon

# Check if port is in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Restart daemon
systemctl restart nordiq-inference

# Check daemon logs
tail -f /var/log/nordiq/inference_daemon.log

# Try different port
INFERENCE_URL = "http://localhost:8001/feed/data"
```

### Issue: "High rejection rate (>5%)"

**Symptoms:**
- Response shows many rejected records
- Error messages about validation
- Acceptance rate declining over time

**Solutions:**

```python
# Log and analyze rejected records
def send_with_logging(records):
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    response = requests.post(
        INFERENCE_URL,
        json={"records": records},
        headers=headers
    )

    result = response.json()

    if result.get("rejected", 0) > 0:
        logger.warning(f"Rejected: {result['rejected']}")
        for error in result.get("errors", []):
            logger.warning(f"  {error['server_name']}: {error['error']}")

            # Log the problematic record
            problem_record = records[error['record_index']]
            logger.debug(f"  Record: {problem_record}")

    return result
```

### Partial Metrics Support

**If you don't have all 14 LINBORG metrics:**

**Option 1: Provide default values for unavailable metrics**
```python
def transform_to_linborg(source):
    return {
        "timestamp": source["timestamp"],
        "server_name": source["hostname"],
        "cpu_user_pct": source.get("cpu_user", 0),      # Available
        "cpu_sys_pct": source.get("cpu_sys", 0),        # Available
        "cpu_iowait_pct": 0,                            # Not available
        "cpu_idle_pct": 100 - (source.get("cpu_user", 0) + source.get("cpu_sys", 0)),
        "java_cpu_pct": 0,                              # Not available
        "mem_used_pct": source.get("memory_used", 0),   # Available
        "swap_used_pct": 0,                             # Not available
        # ... complete all 14 fields
    }
```

**Option 2: Estimate missing metrics**
```python
# Estimate java_cpu from total if not available
if "java_cpu" not in source and "total_cpu" in source:
    java_cpu_pct = max(0, source["total_cpu"] - 20)  # Rough estimate

# Estimate TCP connections
if "close_wait" not in source:
    back_close_wait = int(source.get("net_in_mb_s", 0) * 10)
```

**Option 3: Retrain model without unavailable metrics**
```
1. Edit NordIQ/src/generators/metrics_generator.py
2. Remove metrics you don't have from NORDIQ_METRICS
3. Retrain model with _Starthere.ipynb
4. Deploy new model
```

---

## Appendix: Pre-Deployment Checklist

Before going live with production data:

### Model & Data
- [ ] Model trained on 30+ days of production data
- [ ] Model knows your server profiles (db, web, ml, etc.)
- [ ] All 14 LINBORG metrics available or defaults configured
- [ ] Data quality validated (no null/NaN values)

### Adapter Implementation
- [ ] Adapter tested in dry-run mode (`--test` flag)
- [ ] Field mappings verified for all 14 metrics
- [ ] Profile detection logic covers all server types
- [ ] Error handling and retry logic implemented
- [ ] Logging configured for troubleshooting

### Inference Daemon
- [ ] Daemon running and responding to health checks
- [ ] API key generated and secured
- [ ] Port 8000 accessible from adapter network
- [ ] Model successfully loaded at startup

### First Live Deployment
- [ ] First live batch sent successfully
- [ ] Warmup completed (150 records/server received)
- [ ] Predictions visible in dashboard
- [ ] Risk scores make sense (not all 0 or 100)
- [ ] Timestamps updating every 5 seconds

### Monitoring
- [ ] Adapter logs being captured
- [ ] Alert rules configured for failures
- [ ] Metrics exported to monitoring system
- [ ] Fallback plan if adapter fails
- [ ] Restart/auto-recovery configured

### Operational
- [ ] Demo metrics generator disabled
- [ ] Production data adapter in systemd/supervisor
- [ ] Documentation updated for your environment
- [ ] On-call runbook prepared
- [ ] Team trained on troubleshooting

---

**Built by Craig Giannelli and Claude Code**

---

**Additional Resources:**
- API Reference: See `/NordIQ/Docs/for-developers/API_REFERENCE.md`
- Data Format Specification: See `/NordIQ/Docs/for-developers/DATA_FORMAT_SPEC.md`
- Troubleshooting Guide: See `/NordIQ/Docs/operations/TROUBLESHOOTING.md`

**Questions or Issues?**
Refer to the troubleshooting section above or consult the full documentation at `/NordIQ/Docs/`.
