# Quick Start - Data Adapter Overview

**Read time**: 5 minutes

---

## What You're Building

A **data adapter daemon** that:
1. Polls/subscribes to Wells Fargo monitoring data (Linborg or similar)
2. Transforms metrics to NordIQ format
3. POSTs to NordIQ inference daemon
4. Runs continuously in background

**That's it!** Everything else is already built.

---

## Why It's Easy

### You Only Need 3 Functions

```python
def poll_data_source():
    """Connect to their data source, return raw records."""
    # CUSTOMIZE THIS - query their API/database
    pass

def transform_record(raw):
    """Transform their format to NordIQ format."""
    # CUSTOMIZE THIS - map their field names to ours
    return {
        'timestamp': raw['their_timestamp_field'],
        'server_name': raw['their_hostname_field'],
        'cpu_pct': raw['their_cpu_field'],
        # ... 6 more fields
    }

def main_loop():
    """Poll, transform, send. Repeat."""
    while True:
        raw = poll_data_source()
        nordiq = [transform_record(r) for r in raw]
        send_to_inference_daemon(nordiq)
        time.sleep(5)
```

**The send_to_inference_daemon() function is already written in the template!**

---

## What NordIQ Expects

### Just 9 Fields

```json
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
```

**That's it!** Find these 9 metrics in their data source, map the names, done.

---

## Profile Matching - YOU DON'T NEED TO DO IT

**Good news**: The inference daemon automatically detects server profiles from names.

**Server name patterns** (auto-detected):
- `ppdb*` → database
- `ppweb*` → web_api
- `ppml*` → ml_compute
- `ppetl*` → data_ingest
- `pprisk*` → risk_analytics
- Everything else → generic

**You don't need to include a `profile` field unless you want custom matching!**

---

## Architecture

```
┌────────────────────────────┐
│ Wells Fargo Data Source    │
│ (Linborg, Elasticsearch,   │
│  or whatever they have)    │
└────────────────────────────┘
              ↓
    ┌─────────────────────┐
    │  YOUR ADAPTER       │  ← You build this (4-6 hours)
    │  (100 lines)        │
    └─────────────────────┘
              ↓ HTTP POST /feed/data
    ┌─────────────────────┐
    │  NORDIQ INFERENCE   │  ← Already exists
    │  (Predictions, ML)  │
    └─────────────────────┘
              ↓
    ┌─────────────────────┐
    │  DASHBOARD          │  ← Already exists
    │  (Visualization)    │
    └─────────────────────┘
```

---

## Example - Complete Transformation

**Their data** (example):
```json
{
  "collected_at": "2025-10-30T14:35:00Z",
  "hostname": "proddb01",
  "cpu_usage": 45.2,
  "mem_usage": 78.5,
  "disk_usage": 62.3,
  "net_in": 125.4,
  "net_out": 89.2,
  "disk_r": 42.1,
  "disk_w": 18.3
}
```

**Your transformation**:
```python
def transform_record(raw):
    return {
        'timestamp': raw['collected_at'],
        'server_name': raw['hostname'],
        'cpu_pct': raw['cpu_usage'],
        'memory_pct': raw['mem_usage'],
        'disk_pct': raw['disk_usage'],
        'network_in_mbps': raw['net_in'],
        'network_out_mbps': raw['net_out'],
        'disk_read_mbps': raw['disk_r'],
        'disk_write_mbps': raw['disk_w']
    }
```

**Done!** That's the entire transformation.

---

## Step-by-Step Process

### 1. Ask User for Sample Data (5 minutes)
```
"What does your monitoring data look like? Can you show me a sample JSON record?"
```

### 2. Identify Field Mappings (5 minutes)
Make a table:

| NordIQ Field | Their Field | Notes |
|--------------|-------------|-------|
| timestamp | collected_at | ISO 8601 format |
| server_name | hostname | Unique identifier |
| cpu_pct | cpu_usage | Already 0-100 |
| memory_pct | mem_usage | Already 0-100 |
| ... | ... | ... |

### 3. Copy Template (2 minutes)
Use `03_MINIMAL_TEMPLATE.py`

### 4. Customize 3 Functions (2-3 hours)
- `poll_data_source()` - Query their API
- `transform_record()` - Map field names (see above)
- Optionally: `match_profile()` - Or skip, use auto-detection

### 5. Test (1 hour)
```bash
# Test connection
python your_adapter.py  # Should print raw records

# Test transformation
python your_adapter.py  # Should show NordIQ format

# Test full integration
./start_all.sh  # Start NordIQ
python your_adapter.py  # Should send successfully

# Verify dashboard
open http://localhost:8050  # Wait 20-30 min for warmup
```

### 6. Deploy (30 minutes)
Add to `start_all.sh`, done!

---

## Common Data Sources

### REST API (Most Common)
```python
def poll_data_source():
    response = requests.get("https://linborg.wellsfargo.com/api/metrics")
    return response.json()['servers']
```

### Elasticsearch
```python
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://localhost:9200'])
result = es.search(index="metrics-*", body={...})
return [hit['_source'] for hit in result['hits']['hits']]
```

### MongoDB
```python
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')
return list(client.monitoring.metrics.find())
```

### Database (SQL)
```python
import psycopg2
conn = psycopg2.connect("dbname=monitoring user=...")
cursor = conn.cursor()
cursor.execute("SELECT * FROM metrics WHERE timestamp > %s", [last_poll])
return cursor.fetchall()
```

**See `05_COMMON_TRANSFORMATIONS.md` for complete examples!**

---

## What If They Don't Have These Metrics?

### Missing disk metrics?
Set to 0.0:
```python
'disk_read_mbps': raw.get('disk_r', 0.0),
'disk_write_mbps': raw.get('disk_w', 0.0),
```

### Missing network metrics?
Set to 0.0:
```python
'network_in_mbps': raw.get('net_in', 0.0),
'network_out_mbps': raw.get('net_out', 0.0),
```

### Different units?
Convert:
```python
# Their CPU is 0.0-1.0, we need 0-100
'cpu_pct': raw['cpu'] * 100,

# Their memory is in bytes, we need percentage
'memory_pct': (raw['mem_used'] / raw['mem_total']) * 100,

# Their network is in bytes/sec, we need Mbps
'network_in_mbps': raw['net_in_bytes'] / 1_000_000,
```

---

## Validation

The inference daemon will:
- ✅ Accept valid records
- ✅ Fill missing `status` with "healthy"
- ✅ Auto-detect `profile` if missing
- ❌ Reject records missing required fields
- ❌ Reject values outside valid ranges

**Don't worry about validation - inference daemon handles it!**

---

## Timeline

| Task | Time |
|------|------|
| Understanding (reading docs) | 30 min |
| Sample data from user | 15 min |
| Field mapping | 15 min |
| Customize template | 2 hours |
| Testing | 1 hour |
| Deployment | 30 min |
| **TOTAL** | **4-5 hours** |

---

## Success Looks Like

After your adapter runs for 20-30 minutes:

1. ✅ Dashboard shows your servers
2. ✅ Predictions appear (8-hour forecast)
3. ✅ Risk scores calculated
4. ✅ Alerts generated (if any issues)
5. ✅ No errors in logs

**Then you're done!**

---

## Next Steps

1. Read `02_API_CONTRACT.md` (10 min) - Exact field specifications
2. Ask user for sample data
3. Copy `03_MINIMAL_TEMPLATE.py`
4. Customize and test
5. Deploy

---

## Key Takeaway

**This is not a complex project.**

You're building a simple ETL pipeline:
- **Extract**: Query their API/database
- **Transform**: Map 9 field names
- **Load**: POST to our REST API

**4-6 hours of work. That's it!**

---

© 2025 NordIQ AI, LLC. All rights reserved.
