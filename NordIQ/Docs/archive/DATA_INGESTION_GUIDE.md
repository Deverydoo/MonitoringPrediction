# Data Ingestion Guide
**Complete Specification for Feeding the Inference Engine**

**Version:** 3.0
**Audience:** DevOps Engineers, Integration Developers
**Purpose:** Complete technical specification for sending metrics data to the TFT inference daemon

---

## Table of Contents

1. [Overview](#overview)
2. [Endpoint Specification](#endpoint-specification)
3. [Data Contract v3.0](#data-contract-v30)
4. [Request Format](#request-format)
5. [Response Format](#response-format)
6. [Validation Rules](#validation-rules)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)

---

## Overview

The inference daemon accepts server metrics via the **`POST /feed/data`** endpoint. This document provides the complete specification for formatting and sending your data.

###Key Facts

- **Endpoint:** `POST http://localhost:8000/feed/data`
- **Content-Type:** `application/json`
- **Authentication:** Required (X-API-Key header)
- **Frequency:** Every 5 seconds (recommended)
- **Batch Size:** 1-1000 records per request
- **Warmup:** 150 records per server before predictions start

---

## Endpoint Specification

### HTTP Method & URL

```
POST http://localhost:8000/feed/data
```

### Headers

```http
Content-Type: application/json
X-API-Key: your-api-key-here
```

**Get your API key:**
```bash
# Show existing key
python NordIQ/bin/generate_api_key.py --show

# Or generate new key
python NordIQ/bin/generate_api_key.py
```

### Request Body Structure

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppdb001",
      "profile": "database",
      "cpu_user_pct": 45.2,
      "cpu_sys_pct": 15.3,
      "cpu_iowait_pct": 8.5,
      "cpu_idle_pct": 31.0,
      "java_cpu_pct": 38.0,
      "mem_used_pct": 67.8,
      "swap_used_pct": 2.1,
      "disk_usage_pct": 78.4,
      "net_in_mb_s": 25.3,
      "net_out_mb_s": 18.7,
      "back_close_wait": 15,
      "front_close_wait": 8,
      "load_average": 4.25,
      "uptime_days": 45
    }
  ]
}
```

---

## Data Contract v3.0

### Required Fields

Every record MUST include these fields:

| Field | Type | Required | Description | Valid Range |
|-------|------|----------|-------------|-------------|
| `timestamp` | string (ISO 8601) | ✅ Yes | Measurement timestamp | ISO format with timezone |
| `server_name` | string | ✅ Yes | Unique server identifier | 3-64 alphanumeric chars |
| `profile` | string | ✅ Yes | Server profile type | See [Valid Profiles](#valid-profiles) |

### Metric Fields (14 LINBORG Metrics)

All metric fields are **required**. If a metric is unavailable, send `0` or a reasonable default.

#### CPU Metrics

| Field | Type | Required | Description | Valid Range | Notes |
|-------|------|----------|-------------|-------------|-------|
| `cpu_user_pct` | float | ✅ Yes | User CPU % | 0.0 - 100.0 | Application workload |
| `cpu_sys_pct` | float | ✅ Yes | System CPU % | 0.0 - 100.0 | Kernel overhead |
| `cpu_iowait_pct` | float | ✅ Yes | I/O Wait % | 0.0 - 100.0 | **CRITICAL** - Disk bottleneck indicator |
| `cpu_idle_pct` | float | ✅ Yes | Idle CPU % | 0.0 - 100.0 | Should equal 100 - (user + sys + iowait) |
| `java_cpu_pct` | float | ✅ Yes | Java process CPU % | 0.0 - 100.0 | JVM-specific, 0 if no Java |

**Validation:** `cpu_user_pct + cpu_sys_pct + cpu_iowait_pct + cpu_idle_pct` should ~= 100

#### Memory Metrics

| Field | Type | Required | Description | Valid Range | Notes |
|-------|------|----------|-------------|-------------|-------|
| `mem_used_pct` | float | ✅ Yes | Memory used % | 0.0 - 100.0 | RAM utilization |
| `swap_used_pct` | float | ✅ Yes | Swap used % | 0.0 - 100.0 | Thrashing indicator |

#### Disk Metrics

| Field | Type | Required | Description | Valid Range | Notes |
|-------|------|----------|-------------|-------------|-------|
| `disk_usage_pct` | float | ✅ Yes | Disk usage % | 0.0 - 100.0 | Filesystem capacity |

#### Network Metrics

| Field | Type | Required | Description | Valid Range | Notes |
|-------|------|----------|-------------|-------------|-------|
| `net_in_mb_s` | float | ✅ Yes | Network in (MB/s) | 0.0 - 10000.0 | Inbound throughput |
| `net_out_mb_s` | float | ✅ Yes | Network out (MB/s) | 0.0 - 10000.0 | Outbound throughput |

#### TCP Connection Metrics

| Field | Type | Required | Description | Valid Range | Notes |
|-------|------|----------|-------------|-------------|-------|
| `back_close_wait` | integer | ✅ Yes | Backend TCP CLOSE_WAIT count | 0 - 100000 | Connection leak indicator |
| `front_close_wait` | integer | ✅ Yes | Frontend TCP CLOSE_WAIT count | 0 - 100000 | Client connection issues |

#### System Metrics

| Field | Type | Required | Description | Valid Range | Notes |
|-------|------|----------|-------------|-------------|-------|
| `load_average` | float | ✅ Yes | System load (1 minute) | 0.0 - 1000.0 | Linux `uptime` output |
| `uptime_days` | integer | ✅ Yes | Days since last reboot | 0 - 36500 | Maintenance tracking |

### Optional Fields

| Field | Type | Required | Description | Valid Range |
|-------|------|----------|-------------|-------------|
| `state` | string | ❌ No | Operational state | See [Valid States](#valid-states) |
| `problem_child` | boolean | ❌ No | Problem server flag | `true` or `false` |
| `notes` | string | ❌ No | Freeform notes | Max 500 chars |

### Valid Profiles

Your `profile` field must be one of these values:

| Profile | Description | Example Servers |
|---------|-------------|-----------------|
| `ml_compute` | ML training nodes (CPU/GPU intensive) | ppml0001, gpu-node-5 |
| `database` | Database servers (I/O intensive) | ppdb001, mongo-prod-3 |
| `web_api` | Web/API servers (network intensive) | ppweb001, api-gateway-2 |
| `conductor_mgmt` | Job scheduling, orchestration | ppcon01, k8s-master-1 |
| `data_ingest` | ETL, streaming (Kafka, Spark) | ppetl001, kafka-broker-3 |
| `risk_analytics` | Risk calculations, simulations | pprisk001, quant-server-2 |
| `generic` | Utility, monitoring, unknown | ppgen001, utility-box |

**Profile Selection:**
- Choose the profile that **best matches** your server's primary workload
- Model predictions are optimized per profile
- Incorrect profile = degraded prediction accuracy

### Valid States

Optional `state` field values:

| State | Description | Typical Behavior |
|-------|-------------|------------------|
| `healthy` | Normal operation | Default state |
| `degrading` | Performance declining | Early warning |
| `stressed` | High load, near limits | Approaching critical |
| `critical` | Severe issues | Immediate attention needed |
| `recovering` | Coming back online | Post-incident |
| `maintenance` | Planned downtime | Ignore alerts |
| `offline` | Server unavailable | No predictions |
| `unknown` | State unclear | Use with caution |

**Note:** If omitted, inference engine will auto-detect state based on metrics.

---

## Request Format

### Minimal Valid Request

**Smallest possible request (1 server, 1 datapoint):**

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

### Typical Production Request

**Multiple servers, single timestamp (5-second poll):**

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

### Batch Historical Request

**Backfilling data (multiple timestamps):**

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppdb001",
      "profile": "database",
      ...
    },
    {
      "timestamp": "2025-10-30T15:30:05Z",
      "server_name": "ppdb001",
      "profile": "database",
      ...
    },
    {
      "timestamp": "2025-10-30T15:30:10Z",
      "server_name": "ppdb001",
      "profile": "database",
      ...
    }
  ]
}
```

**Batching rules:**
- Max 1000 records per request
- All timestamps should be within 1 hour window
- Older data is fine (backfilling warmup)

---

## Response Format

### Success Response (200 OK)

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

**Field descriptions:**
- `accepted` - Number of records accepted
- `rejected` - Number of records that failed validation
- `servers_updated` - Unique servers in this batch
- `warmup_status.warmed_up` - Servers ready for predictions
- `warmup_status.warming` - Servers still collecting baseline data

### Partial Success (200 OK with warnings)

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

### Error Response (400 Bad Request)

```json
{
  "status": "error",
  "error": "Validation failed",
  "details": "Request body must be valid JSON with 'records' array",
  "timestamp": "2025-10-30T15:30:05Z"
}
```

### Error Response (401 Unauthorized)

```json
{
  "detail": "Invalid or missing API key"
}
```

**Common causes:**
- Missing `X-API-Key` header
- Incorrect API key value
- Expired API key (if rotation enabled)

### Error Response (413 Payload Too Large)

```json
{
  "status": "error",
  "error": "Payload too large",
  "details": "Maximum 1000 records per request (received 1500)",
  "timestamp": "2025-10-30T15:30:05Z"
}
```

### Error Response (500 Internal Server Error)

```json
{
  "status": "error",
  "error": "Internal server error",
  "details": "Failed to process prediction for server ppml0001",
  "timestamp": "2025-10-30T15:30:05Z"
}
```

---

## Validation Rules

### Timestamp Validation

```python
# Valid formats
"2025-10-30T15:30:00Z"           # UTC
"2025-10-30T15:30:00+00:00"      # UTC with offset
"2025-10-30T15:30:00.123Z"       # With milliseconds
"2025-10-30T10:30:00-05:00"      # EST

# Invalid formats
"2025-10-30 15:30:00"            # Missing 'T'
"10/30/2025 3:30 PM"             # US format
"1730300400"                      # Unix timestamp (must convert first)
```

**Rules:**
- Must be ISO 8601 format
- Must include timezone (`Z` or `±HH:MM`)
- Can be past or present (future rejected)
- Max 1 hour old recommended (older = warmup only)

### Server Name Validation

```python
# Valid server names
"ppdb001"         # Alphanumeric with numbers
"web-server-01"   # Hyphens allowed
"ml_gpu_005"      # Underscores allowed
"k8s.prod.node3"  # Dots allowed

# Invalid server names
"server 01"       # Spaces not allowed
"db@prod"         # Special chars not allowed
"x"               # Too short (min 3 chars)
"very-long-name-that-exceeds-64-characters-and-goes-on-and-on"  # Too long
```

**Rules:**
- Length: 3-64 characters
- Allowed: `a-z`, `A-Z`, `0-9`, `-`, `_`, `.`
- Must be unique per server
- Case-sensitive (`Server01` ≠ `server01`)

### Metric Validation Rules

#### CPU Metrics
- All percentages: 0.0 ≤ value ≤ 100.0
- Sum constraint: `user + sys + iowait + idle` ≈ 100 (±5% tolerance)

#### Memory Metrics
- All percentages: 0.0 ≤ value ≤ 100.0
- Warning if `mem_used_pct > 95` or `swap_used_pct > 10`

#### Network Metrics
- `net_in_mb_s`: 0.0 ≤ value ≤ 10000.0
- `net_out_mb_s`: 0.0 ≤ value ≤ 10000.0
- Warning if > 1000 MB/s (check units)

#### Connection Metrics
- `back_close_wait`: 0 ≤ value ≤ 100000
- `front_close_wait`: 0 ≤ value ≤ 100000
- Warning if > 1000 (connection leak suspected)

---

## Error Handling

### Client-Side Error Handling

```python
import requests
import time

def send_metrics_with_retry(records, max_retries=3):
    """Send metrics with exponential backoff retry."""

    url = "http://localhost:8000/feed/data"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key"
    }
    payload = {"records": records}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result["rejected"] > 0:
                    print(f"⚠️  {result['rejected']} records rejected:")
                    for error in result.get("errors", []):
                        print(f"   - {error['server_name']}: {error['error']}")
                return result

            elif response.status_code == 401:
                print("❌ Authentication failed - check API key")
                return None  # Don't retry auth errors

            elif response.status_code == 413:
                print("❌ Payload too large - split into smaller batches")
                return None  # Don't retry, need to split batch

            elif response.status_code >= 500:
                print(f"⚠️  Server error (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue

            else:
                print(f"❌ Unexpected status: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print(f"⚠️  Request timeout (attempt {attempt+1}/{max_retries})")
            time.sleep(2 ** attempt)
            continue

        except requests.exceptions.ConnectionError:
            print(f"⚠️  Connection failed (attempt {attempt+1}/{max_retries})")
            time.sleep(2 ** attempt)
            continue

    print("❌ Max retries exceeded")
    return None
```

### Validation Error Recovery

```python
def validate_and_fix_record(record):
    """Validate record and apply fixes for common issues."""

    # Fix CPU sum if slightly off
    cpu_sum = (record["cpu_user_pct"] + record["cpu_sys_pct"] +
               record["cpu_iowait_pct"] + record["cpu_idle_pct"])

    if abs(cpu_sum - 100.0) > 0.1:  # Allow 0.1% tolerance
        # Normalize to 100%
        scale = 100.0 / cpu_sum
        record["cpu_user_pct"] *= scale
        record["cpu_sys_pct"] *= scale
        record["cpu_iowait_pct"] *= scale
        record["cpu_idle_pct"] *= scale

    # Clamp values to valid ranges
    for key in ["cpu_user_pct", "cpu_sys_pct", "cpu_iowait_pct", "cpu_idle_pct",
                "mem_used_pct", "swap_used_pct", "disk_usage_pct"]:
        record[key] = max(0.0, min(100.0, record[key]))

    # Ensure non-negative
    for key in ["net_in_mb_s", "net_out_mb_s", "load_average"]:
        record[key] = max(0.0, record[key])

    for key in ["back_close_wait", "front_close_wait", "uptime_days"]:
        record[key] = max(0, record[key])

    return record
```

---

## Best Practices

### Polling Frequency

**Recommended: 5 seconds**

```python
import time

POLL_INTERVAL = 5  # seconds

while True:
    records = fetch_metrics_from_source()
    send_to_inference(records)
    time.sleep(POLL_INTERVAL)
```

**Why 5 seconds?**
- Matches demo data frequency
- Good balance: responsiveness vs. overhead
- TFT model optimized for 5-second intervals

**Other intervals:**
- 1 second: High overhead, not necessary
- 10 seconds: Acceptable, slightly delayed warmup
- 60 seconds: Too slow, delays warmup significantly

### Batch Size

**Recommended: 20-100 servers per request**

```python
# Good - batch multiple servers
records = [
    fetch_server("ppdb001"),
    fetch_server("ppdb002"),
    # ... all servers at same timestamp
]
send_to_inference(records)  # Single POST

# Bad - individual POSTs
for server in servers:
    record = fetch_server(server)
    send_to_inference([record])  # Don't do this!
```

**Why batch?**
- Reduces HTTP overhead
- Better throughput
- Atomic timestamp consistency

### Data Quality

**Ensure consistent timestamps:**

```python
# Good - single query timestamp
current_time = datetime.now(timezone.utc)
records = [
    {
        "timestamp": current_time.isoformat(),
        "server_name": "ppdb001",
        ...
    },
    {
        "timestamp": current_time.isoformat(),  # Same timestamp
        "server_name": "ppdb002",
        ...
    }
]

# Bad - different timestamps per server
for server in servers:
    record = {
        "timestamp": datetime.now().isoformat(),  # Don't do this!
        "server_name": server,
        ...
    }
```

### Handling Missing Servers

If a server goes offline:

```python
# Option 1: Don't send (inference continues with last known data)
active_servers = [s for s in servers if is_online(s)]
records = [fetch_server(s) for s in active_servers]

# Option 2: Send with "offline" state (preferred)
for server in servers:
    if is_online(server):
        record = fetch_server(server)
    else:
        record = create_offline_record(server)  # All metrics = 0, state = "offline"
    records.append(record)
```

---

## Code Examples

### Python (requests)

```python
import requests
from datetime import datetime, timezone

def send_metrics(records):
    """Send metrics to inference daemon."""

    url = "http://localhost:8000/feed/data"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key-here"
    }

    payload = {"records": records}

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()

    return response.json()

# Example usage
records = [
    {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_name": "ppdb001",
        "profile": "database",
        "cpu_user_pct": 45.2,
        "cpu_sys_pct": 15.3,
        "cpu_iowait_pct": 8.5,
        "cpu_idle_pct": 31.0,
        "java_cpu_pct": 0.0,
        "mem_used_pct": 67.8,
        "swap_used_pct": 2.1,
        "disk_usage_pct": 78.4,
        "net_in_mb_s": 25.3,
        "net_out_mb_s": 18.7,
        "back_close_wait": 15,
        "front_close_wait": 8,
        "load_average": 4.25,
        "uptime_days": 45
    }
]

result = send_metrics(records)
print(f"✅ Accepted: {result['accepted']}, Rejected: {result['rejected']}")
```

### Node.js (axios)

```javascript
const axios = require('axios');

async function sendMetrics(records) {
  const url = 'http://localhost:8000/feed/data';
  const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key-here'
  };

  const payload = { records: records };

  try {
    const response = await axios.post(url, payload, { headers, timeout: 10000 });
    return response.data;
  } catch (error) {
    if (error.response) {
      console.error(`Error: ${error.response.status} - ${error.response.data}`);
    } else {
      console.error(`Error: ${error.message}`);
    }
    throw error;
  }
}

// Example usage
const records = [
  {
    timestamp: new Date().toISOString(),
    server_name: 'ppdb001',
    profile: 'database',
    cpu_user_pct: 45.2,
    cpu_sys_pct: 15.3,
    cpu_iowait_pct: 8.5,
    cpu_idle_pct: 31.0,
    java_cpu_pct: 0.0,
    mem_used_pct: 67.8,
    swap_used_pct: 2.1,
    disk_usage_pct: 78.4,
    net_in_mb_s: 25.3,
    net_out_mb_s: 18.7,
    back_close_wait: 15,
    front_close_wait: 8,
    load_average: 4.25,
    uptime_days: 45
  }
];

sendMetrics(records)
  .then(result => console.log(`✅ Accepted: ${result.accepted}`))
  .catch(error => console.error('Failed to send metrics'));
```

### curl

```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "records": [
      {
        "timestamp": "2025-10-30T15:30:00Z",
        "server_name": "ppdb001",
        "profile": "database",
        "cpu_user_pct": 45.2,
        "cpu_sys_pct": 15.3,
        "cpu_iowait_pct": 8.5,
        "cpu_idle_pct": 31.0,
        "java_cpu_pct": 0.0,
        "mem_used_pct": 67.8,
        "swap_used_pct": 2.1,
        "disk_usage_pct": 78.4,
        "net_in_mb_s": 25.3,
        "net_out_mb_s": 18.7,
        "back_close_wait": 15,
        "front_close_wait": 8,
        "load_average": 4.25,
        "uptime_days": 45
      }
    ]
  }'
```

---

## Next Steps

Once you've successfully ingested data:

1. **Monitor warmup progress** - Check `/status` endpoint
2. **Verify predictions** - Use `/predictions/current` endpoint
3. **Set up monitoring** - Track adapter health and latency
4. **Optimize batching** - Tune batch size for your fleet
5. **Implement fallbacks** - Handle inference daemon restarts gracefully

**See Also:**
- [Real Data Integration Guide](REAL_DATA_INTEGRATION.md) - Complete production setup
- [API Reference](../for-developers/API_REFERENCE.md) - All available endpoints
- [Data Format Specification](../for-developers/DATA_FORMAT_SPEC.md) - Detailed schema

---

**Questions?** See [Troubleshooting Guide](../operations/TROUBLESHOOTING.md) or contact support.
