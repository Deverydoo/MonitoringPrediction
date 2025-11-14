# API Contract - Exact Specification

**Read time**: 10 minutes

This document specifies the EXACT contract between your adapter and the NordIQ inference daemon.

---

## Endpoint

```
URL:     POST http://localhost:8000/feed/data
Method:  POST
Auth:    X-API-Key header
Rate:    60 requests/minute (1 per second)
Format:  application/json
```

---

## Authentication

**Header**: `X-API-Key`
**Value**: Load from `.nordiq_key` file

```python
# Load API key
with open('.nordiq_key', 'r') as f:
    api_key = f.read().strip()

# Use in request
headers = {'X-API-Key': api_key}
```

---

## Request Format

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T14:35:00",
      "server_name": "ppdb001",
      "cpu_pct": 45.2,
      "memory_pct": 78.5,
      "disk_pct": 62.3,
      "network_in_mbps": 125.4,
      "network_out_mbps": 89.2,
      "disk_read_mbps": 42.1,
      "disk_write_mbps": 18.3,
      "status": "healthy",
      "profile": "database"
    },
    {
      "timestamp": "2025-10-30T14:35:00",
      "server_name": "ppweb003",
      "cpu_pct": 32.1,
      "memory_pct": 45.6,
      "disk_pct": 38.9,
      "network_in_mbps": 89.3,
      "network_out_mbps": 156.7,
      "disk_read_mbps": 5.2,
      "disk_write_mbps": 2.1,
      "status": "healthy",
      "profile": "web_api"
    }
  ]
}
```

**Key points**:
- `records` is an **array** (can send multiple servers at once)
- All servers in same batch should have **same timestamp**
- Send all servers together for efficiency

---

## Response Format

### Success (200 OK)

```json
{
  "success": true,
  "records_received": 2,
  "servers_updated": ["ppdb001", "ppweb003"],
  "warmup_status": {
    "ready": 2,
    "total": 2,
    "is_ready": true
  }
}
```

### Error (400 Bad Request)

```json
{
  "detail": "Missing required field: cpu_pct"
}
```

### Error (401 Unauthorized)

```json
{
  "detail": "Invalid API key"
}
```

### Error (429 Too Many Requests)

```json
{
  "detail": "Rate limit exceeded: 60 requests/minute"
}
```

---

## Required Fields

These fields **MUST** be present in every record:

### 1. timestamp
- **Type**: String (ISO 8601 format)
- **Format**: `"YYYY-MM-DDTHH:MM:SS"` or `"YYYY-MM-DDTHH:MM:SS.ffffff"`
- **Examples**:
  - `"2025-10-30T14:35:00"`
  - `"2025-10-30T14:35:00.123456"`
  - `"2025-10-30T14:35:00Z"` (UTC)
  - `"2025-10-30T14:35:00-05:00"` (with timezone)

```python
# Generate timestamp
from datetime import datetime
timestamp = datetime.now().isoformat()  # "2025-10-30T14:35:00.123456"
```

### 2. server_name
- **Type**: String
- **Requirements**: Non-empty, unique identifier
- **Examples**: `"ppdb001"`, `"prodweb03"`, `"ml-gpu-01"`
- **Used for**: Profile auto-detection, tracking, dashboard display

### 3. cpu_pct
- **Type**: Float
- **Range**: 0.0 - 100.0
- **Unit**: Percentage (0-100, not 0-1)
- **Description**: CPU usage percentage across all cores

### 4. memory_pct
- **Type**: Float
- **Range**: 0.0 - 100.0
- **Unit**: Percentage (0-100, not 0-1)
- **Description**: Memory (RAM) usage percentage

### 5. disk_pct
- **Type**: Float
- **Range**: 0.0 - 100.0
- **Unit**: Percentage (0-100, not 0-1)
- **Description**: Disk usage percentage (primary/root filesystem)

### 6. network_in_mbps
- **Type**: Float
- **Range**: 0.0 - 10000.0
- **Unit**: Megabits per second (Mbps)
- **Description**: Network inbound throughput

**Unit conversions**:
```python
# From bytes/sec to Mbps
network_in_mbps = bytes_per_sec / 125_000  # or / (1_000_000 / 8)

# From MB/s to Mbps
network_in_mbps = mb_per_sec * 8

# From Kbps to Mbps
network_in_mbps = kbps / 1000
```

### 7. network_out_mbps
- **Type**: Float
- **Range**: 0.0 - 10000.0
- **Unit**: Megabits per second (Mbps)
- **Description**: Network outbound throughput

### 8. disk_read_mbps
- **Type**: Float
- **Range**: 0.0 - 10000.0
- **Unit**: Megabytes per second (MB/s) - **NOT Mbps!**
- **Description**: Disk read throughput

**Important**: This is **MB/s** (megabytes), not Mbps (megabits)!

### 9. disk_write_mbps
- **Type**: Float
- **Range**: 0.0 - 10000.0
- **Unit**: Megabytes per second (MB/s) - **NOT Mbps!**
- **Description**: Disk write throughput

**Important**: This is **MB/s** (megabytes), not Mbps (megabits)!

---

## Optional Fields

### status
- **Type**: String
- **Values**: `"healthy"`, `"degraded"`, `"warning"`, `"critical"`
- **Default**: `"healthy"` (if not provided)
- **Description**: Current server health status

```python
# If you have a status field
'status': raw.get('status', 'healthy')

# If you don't, omit it - inference daemon will default to 'healthy'
```

### profile
- **Type**: String
- **Values**: See Profile section below
- **Default**: Auto-detected from server name
- **Description**: Server workload profile

**Recommendation**: **Omit this field** and let inference daemon auto-detect from server name!

---

## Profiles

### Valid Profile Values

```python
PROFILES = [
    'ml_compute',       # ML/AI training, GPU workloads
    'database',         # Database servers (SQL, NoSQL)
    'web_api',          # Web servers, API gateways
    'conductor_mgmt',   # Orchestration, management
    'data_ingest',      # ETL, Kafka, data pipelines
    'risk_analytics',   # Risk calculation, analytics
    'generic'           # Unknown/other workloads
]
```

### Auto-Detection (Built-in)

**If you DON'T include `profile` field**, the inference daemon auto-detects based on server name prefix:

```python
# Inference daemon logic (tft_inference_daemon.py:579-589)
if server_name.startswith('ppml'):   profile = 'ml_compute'
if server_name.startswith('ppdb'):   profile = 'database'
if server_name.startswith('ppweb'):  profile = 'web_api'
if server_name.startswith('ppcon'):  profile = 'conductor_mgmt'
if server_name.startswith('ppetl'):  profile = 'data_ingest'
if server_name.startswith('pprisk'): profile = 'risk_analytics'
else:                                profile = 'generic'
```

**Examples**:
- `ppdb001` → `database`
- `ppweb03` → `web_api`
- `ppml-gpu-02` → `ml_compute`
- `unknown-server` → `generic`

**Why profiles matter**: Different profiles have different risk thresholds:
- Database servers: High memory OK (page cache)
- ML servers: High CPU OK (training)
- Web servers: High network OK (traffic)

---

## Complete Example

### Python Code

```python
import requests
from datetime import datetime

# Load API key
with open('.nordiq_key', 'r') as f:
    api_key = f.read().strip()

# Prepare data
records = [
    {
        'timestamp': datetime.now().isoformat(),
        'server_name': 'ppdb001',
        'cpu_pct': 45.2,
        'memory_pct': 78.5,
        'disk_pct': 62.3,
        'network_in_mbps': 125.4,
        'network_out_mbps': 89.2,
        'disk_read_mbps': 42.1,
        'disk_write_mbps': 18.3
        # Note: No 'profile' - will be auto-detected as 'database'
        # Note: No 'status' - will default to 'healthy'
    },
    {
        'timestamp': datetime.now().isoformat(),
        'server_name': 'ppweb003',
        'cpu_pct': 32.1,
        'memory_pct': 45.6,
        'disk_pct': 38.9,
        'network_in_mbps': 89.3,
        'network_out_mbps': 156.7,
        'disk_read_mbps': 5.2,
        'disk_write_mbps': 2.1
    }
]

# Send to inference daemon
response = requests.post(
    'http://localhost:8000/feed/data',
    json={'records': records},
    headers={'X-API-Key': api_key},
    timeout=30
)

# Check response
if response.status_code == 200:
    result = response.json()
    print(f"Success! Sent {result['records_received']} records")
else:
    print(f"Error {response.status_code}: {response.text}")
```

### curl Example

```bash
API_KEY=$(cat .nordiq_key)

curl -X POST http://localhost:8000/feed/data \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "timestamp": "2025-10-30T14:35:00",
        "server_name": "ppdb001",
        "cpu_pct": 45.2,
        "memory_pct": 78.5,
        "disk_pct": 62.3,
        "network_in_mbps": 125.4,
        "network_out_mbps": 89.2,
        "disk_read_mbps": 42.1,
        "disk_write_mbps": 18.3
      }
    ]
  }'
```

---

## Batching

### Why Batch?

**Don't do this** (inefficient):
```python
for server in servers:
    # One POST per server = 100 requests/minute for 100 servers
    requests.post('/feed/data', json={'records': [server]})
```

**Do this instead** (efficient):
```python
# One POST for all servers = 1 request/minute
requests.post('/feed/data', json={'records': servers})
```

### Batch Size Recommendations

| Server Count | Batch Strategy |
|--------------|----------------|
| 1-50 servers | Send all at once |
| 50-200 servers | Send all at once (still OK) |
| 200-1000 servers | Split into 2-4 batches |
| 1000+ servers | Split into 50-100 server batches |

**Rate limit**: 60 requests/minute = 1 per second max

```python
# Example: 500 servers, split into 5 batches of 100
batch_size = 100
for i in range(0, len(servers), batch_size):
    batch = servers[i:i+batch_size]
    requests.post('/feed/data', json={'records': batch})
    time.sleep(1)  # Rate limiting
```

---

## Data Validation

### What the Inference Daemon Validates

**Field presence** (required fields):
- ❌ Missing `timestamp` → 400 error
- ❌ Missing `server_name` → 400 error
- ❌ Missing any of 7 metric fields → 400 error

**Field types**:
- ❌ String instead of float → 400 error
- ❌ Non-ISO timestamp → 400 error

**Field ranges** (soft validation):
- ⚠️ Values outside 0-100 (percentages) → Logged as warning, clamped
- ⚠️ Negative values → Logged as warning, clamped to 0
- ⚠️ Very large values → Logged as warning, clamped to max

**Defaults applied**:
- ✅ Missing `status` → Defaults to `"healthy"`
- ✅ Missing `profile` → Auto-detected from `server_name`

### Your Validation (Recommended)

```python
def validate_record(record):
    """Validate record before sending."""
    # Required fields
    required = ['timestamp', 'server_name', 'cpu_pct', 'memory_pct',
                'disk_pct', 'network_in_mbps', 'network_out_mbps',
                'disk_read_mbps', 'disk_write_mbps']

    for field in required:
        if field not in record:
            raise ValueError(f"Missing required field: {field}")

    # Clamp percentages to 0-100
    for field in ['cpu_pct', 'memory_pct', 'disk_pct']:
        record[field] = max(0.0, min(100.0, record[field]))

    # Clamp throughput to 0-10000
    for field in ['network_in_mbps', 'network_out_mbps',
                  'disk_read_mbps', 'disk_write_mbps']:
        record[field] = max(0.0, min(10000.0, record[field]))

    return record
```

---

## Error Handling

### Connection Errors

```python
try:
    response = requests.post(
        'http://localhost:8000/feed/data',
        json={'records': records},
        headers={'X-API-Key': api_key},
        timeout=30
    )
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to inference daemon")
    print("Is it running? Try: ./start_all.sh")
except requests.exceptions.Timeout:
    print("ERROR: Request timed out (>30s)")
except requests.exceptions.HTTPError as e:
    print(f"ERROR: HTTP {e.response.status_code}: {e.response.text}")
```

### Common Errors

| Status | Error | Cause | Fix |
|--------|-------|-------|-----|
| 401 | Unauthorized | Invalid API key | Check `.nordiq_key` file |
| 400 | Bad Request | Missing/invalid field | Check required fields |
| 429 | Rate limit | >60 requests/min | Batch records, slow down |
| 500 | Server error | Inference daemon error | Check daemon logs |

---

## Testing

### Test Inference Daemon is Running

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "service": "tft_inference_daemon", "running": true}
```

### Test Authentication

```bash
API_KEY=$(cat .nordiq_key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/status
# Expected: JSON with system status
```

### Test Feed Data Endpoint

```bash
API_KEY=$(cat .nordiq_key)

curl -X POST http://localhost:8000/feed/data \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "timestamp": "2025-10-30T14:35:00",
      "server_name": "test001",
      "cpu_pct": 45.2,
      "memory_pct": 78.5,
      "disk_pct": 62.3,
      "network_in_mbps": 125.4,
      "network_out_mbps": 89.2,
      "disk_read_mbps": 42.1,
      "disk_write_mbps": 18.3
    }]
  }'

# Expected: {"success": true, "records_received": 1, ...}
```

### Verify Dashboard

```bash
# Wait 20-30 minutes for warmup, then check:
open http://localhost:8050

# Or check predictions via API:
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predictions/current
```

---

## Summary Checklist

When implementing your adapter, verify:

- [ ] Endpoint: `POST http://localhost:8000/feed/data`
- [ ] Header: `X-API-Key` from `.nordiq_key` file
- [ ] Body: `{"records": [...]}`
- [ ] All 9 required fields present
- [ ] timestamp in ISO 8601 format
- [ ] cpu/memory/disk as percentages (0-100)
- [ ] network/disk throughput in Mbps/MB/s
- [ ] Batch multiple servers in single POST
- [ ] Stay under 60 requests/minute
- [ ] Handle connection errors
- [ ] Optionally validate data before sending

---

Built by Craig Giannelli and Claude Code
