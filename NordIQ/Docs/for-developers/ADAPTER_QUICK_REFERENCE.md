# Data Adapter Quick Reference Card

**For**: Claude 3.7 or any developer building data adapters
**Use**: Quick lookup while coding - no fluff, just facts

---

## API Endpoint

```
POST http://localhost:8000/feed/data
Header: X-API-Key: {from .nordiq_key file}
Content-Type: application/json
Rate Limit: 60/minute
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
    }
  ]
}
```

---

## Required Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `timestamp` | ISO 8601 string | - | `"2025-10-30T14:35:00"` |
| `server_name` | string | - | Unique identifier |
| `cpu_pct` | float | 0-100 | CPU usage % |
| `memory_pct` | float | 0-100 | Memory usage % |
| `disk_pct` | float | 0-100 | Disk usage % |
| `network_in_mbps` | float | 0-10000 | Network in (Mbps) |
| `network_out_mbps` | float | 0-10000 | Network out (Mbps) |
| `disk_read_mbps` | float | 0-10000 | Disk read (MB/s) |
| `disk_write_mbps` | float | 0-10000 | Disk write (MB/s) |

---

## Optional Fields

| Field | Type | Values | Default |
|-------|------|--------|---------|
| `status` | string | "healthy", "degraded", "warning", "critical" | "healthy" |
| `profile` | string | See below | Auto-detected |

---

## Profiles

```python
PROFILES = [
    'ml_compute',       # ML/AI workloads
    'database',         # Database servers
    'web_api',          # Web/API servers
    'conductor_mgmt',   # Orchestration
    'data_ingest',      # ETL/pipelines
    'risk_analytics',   # Risk calculation
    'generic'           # Default/unknown
]
```

---

## Profile Auto-Detection (Built-in)

If you **don't include** `profile` field, inference daemon auto-detects:

```python
# Server name pattern matching (tft_inference_daemon.py:579)
if server_name.startswith('ppml'):   profile = 'ml_compute'
if server_name.startswith('ppdb'):   profile = 'database'
if server_name.startswith('ppweb'):  profile = 'web_api'
if server_name.startswith('ppcon'):  profile = 'conductor_mgmt'
if server_name.startswith('ppetl'):  profile = 'data_ingest'
if server_name.startswith('pprisk'): profile = 'risk_analytics'
else:                                profile = 'generic'
```

**Recommendation**: Let inference daemon handle it unless you need custom logic.

---

## Minimal Working Example

```python
import requests
import time
from datetime import datetime

API_KEY = open('.nordiq_key').read().strip()
DAEMON_URL = "http://localhost:8000"

while True:
    # Your data source here
    records = [
        {
            "timestamp": datetime.now().isoformat(),
            "server_name": "server001",
            "cpu_pct": 45.2,
            "memory_pct": 78.5,
            "disk_pct": 62.3,
            "network_in_mbps": 125.4,
            "network_out_mbps": 89.2,
            "disk_read_mbps": 42.1,
            "disk_write_mbps": 18.3
        }
    ]

    # Send to inference daemon
    response = requests.post(
        f"{DAEMON_URL}/feed/data",
        json={"records": records},
        headers={"X-API-Key": API_KEY}
    )

    print(f"Sent {len(records)} records: {response.status_code}")
    time.sleep(5)
```

---

## Common Transformations

### Elasticsearch → NordIQ
```python
def transform_elasticsearch(hit):
    src = hit['_source']
    return {
        'timestamp': src['@timestamp'],
        'server_name': src['host']['name'],
        'cpu_pct': src['system']['cpu']['total']['pct'] * 100,
        'memory_pct': src['system']['memory']['actual']['used']['pct'] * 100,
        'disk_pct': src['system']['filesystem']['used']['pct'] * 100,
        'network_in_mbps': src['system']['network']['in']['bytes'] / 1_000_000,
        'network_out_mbps': src['system']['network']['out']['bytes'] / 1_000_000,
        'disk_read_mbps': src['system']['diskio']['read']['bytes'] / 1_000_000,
        'disk_write_mbps': src['system']['diskio']['write']['bytes'] / 1_000_000
    }
```

### MongoDB → NordIQ
```python
def transform_mongodb(doc):
    return {
        'timestamp': doc['collected_at'].isoformat(),
        'server_name': doc['hostname'],
        'cpu_pct': doc['metrics']['cpu']['usage_pct'],
        'memory_pct': doc['metrics']['memory']['usage_pct'],
        'disk_pct': doc['metrics']['disk']['usage_pct'],
        'network_in_mbps': doc['metrics']['network']['in_mbps'],
        'network_out_mbps': doc['metrics']['network']['out_mbps'],
        'disk_read_mbps': doc['metrics']['disk']['read_mbps'],
        'disk_write_mbps': doc['metrics']['disk']['write_mbps']
    }
```

### Generic REST API → NordIQ
```python
def transform_generic(raw):
    return {
        'timestamp': raw.get('time', datetime.now().isoformat()),
        'server_name': raw.get('host', 'unknown'),
        'cpu_pct': raw.get('cpu', 0.0),
        'memory_pct': raw.get('mem', 0.0),
        'disk_pct': raw.get('disk', 0.0),
        'network_in_mbps': raw.get('net_in', 0.0),
        'network_out_mbps': raw.get('net_out', 0.0),
        'disk_read_mbps': raw.get('disk_r', 0.0),
        'disk_write_mbps': raw.get('disk_w', 0.0)
    }
```

---

## Batching Best Practice

```python
# Batch multiple servers in single POST
records = []
for server in servers:
    records.append(transform(server))

# Send all at once (stay under 60/minute rate limit)
response = requests.post(
    f"{DAEMON_URL}/feed/data",
    json={"records": records},
    headers={"X-API-Key": API_KEY}
)
```

**Why**: More efficient, inference daemon expects batches.

---

## Error Handling

```python
def send_with_retry(records, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                f"{DAEMON_URL}/feed/data",
                json={"records": records},
                headers={"X-API-Key": API_KEY},
                timeout=30
            )
            response.raise_for_status()
            return True
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"Failed after {max_attempts} attempts: {e}")
                return False
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

## Testing Commands

```bash
# Test inference daemon is running
curl http://localhost:8000/health

# Load API key
API_KEY=$(cat .nordiq_key)

# Send test record
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

# Check dashboard
open http://localhost:8050

# Check predictions
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predictions/current
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` | Invalid API key | Check `.nordiq_key` file |
| `429 Too Many Requests` | Rate limit exceeded | Batch records, reduce frequency |
| `400 Bad Request` | Invalid data format | Check required fields |
| `Connection refused` | Daemon not running | Run `./start_all.sh` |
| `No data on dashboard` | Warmup period | Wait 20-30 minutes |

---

## File Locations

```
NordIQ/
├── .nordiq_key                          # API key (load this)
├── src/daemons/
│   ├── tft_inference_daemon.py          # Target endpoint
│   ├── your_adapter_daemon.py           # Your code here
├── logs/
│   ├── inference_daemon.log             # Check for errors
│   ├── adapter_daemon.log               # Your logs
├── Docs/for-developers/
│   ├── DATA_ADAPTER_GUIDE.md            # Full guide
│   └── ADAPTER_QUICK_REFERENCE.md       # This file
```

---

## Questions to Ask User

When building adapter, ask:

1. **Data source?** (Linborg, Elasticsearch, custom API, etc.)
2. **Access method?** (REST, streaming, database query, etc.)
3. **Field names?** (What does your data look like?)
4. **Server count?** (Affects batching strategy)
5. **Custom profiles?** (Do you have CMDB/tags, or use auto-detection?)
6. **Authentication?** (How to auth to data source?)

---

## Development Checklist

- [ ] Connect to data source
- [ ] Transform to required 9 fields
- [ ] Optional: Add profile matching
- [ ] Send via POST to `/feed/data`
- [ ] Add error handling and retries
- [ ] Test with real data
- [ ] Monitor dashboard for predictions

---

## Performance Tips

1. **Batch records** - Send 50-100 servers per POST
2. **Stay under rate limit** - Max 60 POST/minute
3. **Handle missing data** - Fill with 0.0 or skip record
4. **Log everything** - Debug production issues
5. **Use async** - Don't block on HTTP POST
6. **Clear buffers** - Prevent memory leaks

---

## Full Documentation

- [DATA_ADAPTER_GUIDE.md](DATA_ADAPTER_GUIDE.md) - Complete guide with templates
- [ELASTICSEARCH_INTEGRATION.md](../for-production/ELASTICSEARCH_INTEGRATION.md) - Elasticsearch adapter
- [MONGODB_INTEGRATION.md](../for-production/MONGODB_INTEGRATION.md) - MongoDB adapter

---

© 2025 NordIQ AI, LLC. All rights reserved.
