# Summary for Claude 3.7

**Dear Claude 3.7** (or any AI assistant helping with this project),

This folder contains everything you need to build a data adapter for the NordIQ inference engine.

---

## What You're Building

A **data adapter daemon** that:
1. Connects to Wells Fargo's monitoring data source (Linborg or similar)
2. Transforms metrics to NordIQ format (9 required fields)
3. POSTs to NordIQ inference daemon REST API
4. Runs continuously in background

**Estimated time**: 4-6 hours

---

## Files in This Folder

| File | Purpose | Read Time |
|------|---------|-----------|
| `00_READ_ME_FIRST.md` | Navigation and quick reference | 3 min |
| `01_QUICK_START.md` | Overview, why it's easy, examples | 5 min |
| `02_API_CONTRACT.md` | Exact field specs, API details | 10 min |
| `03_MINIMAL_TEMPLATE.py` | Ready-to-use Python template | - |
| `04_TESTING_GUIDE.md` | How to test and verify | 5 min |
| `05_SUMMARY_FOR_CLAUDE.md` | This file | 3 min |

---

## Read These First

1. **`00_READ_ME_FIRST.md`** - Start here for navigation
2. **`01_QUICK_START.md`** - Understand what you're building
3. **`02_API_CONTRACT.md`** - Learn the exact requirements

---

## Then Use This Template

**`03_MINIMAL_TEMPLATE.py`** - Copy and customize 3 functions:

```python
# Function 1: Connect to data source
def poll_data_source():
    # CUSTOMIZE: Query Linborg API, Elasticsearch, etc.
    pass

# Function 2: Transform field names
def transform_record(raw):
    # CUSTOMIZE: Map their field names to NordIQ's 9 fields
    pass

# Function 3: (OPTIONAL) Profile matching
def match_profile(server_name):
    # CUSTOMIZE: Or just return 'generic' for auto-detection
    pass
```

**Everything else is already implemented!**

---

## The 9 Required Fields

Your only job is to map these 9 fields from their data source:

| # | Field | Type | Range | Notes |
|---|-------|------|-------|-------|
| 1 | `timestamp` | string | ISO 8601 | `"2025-10-30T14:35:00"` |
| 2 | `server_name` | string | non-empty | `"ppdb001"` |
| 3 | `cpu_pct` | float | 0-100 | Percentage |
| 4 | `memory_pct` | float | 0-100 | Percentage |
| 5 | `disk_pct` | float | 0-100 | Percentage |
| 6 | `network_in_mbps` | float | 0-10000 | Megabits/sec |
| 7 | `network_out_mbps` | float | 0-10000 | Megabits/sec |
| 8 | `disk_read_mbps` | float | 0-10000 | Megabytes/sec |
| 9 | `disk_write_mbps` | float | 0-10000 | Megabytes/sec |

---

## Questions to Ask the User

Before you start, ask:

1. **"What is your data source?"**
   - Linborg API? Elasticsearch? MongoDB? Custom?

2. **"What does your data look like?"**
   - Ask for sample JSON/data structure
   - You need to know their field names!

3. **"How do I access it?"**
   - REST API endpoint? Database? Kafka topic?
   - Authentication method?

4. **"How many servers?"**
   - Affects batching strategy

---

## Implementation Steps

### Step 1: Get Sample Data (15 min)
Ask user for sample JSON from their monitoring system.

### Step 2: Map Fields (15 min)
Make a table mapping their fields to NordIQ fields:

| Their Field | NordIQ Field | Conversion Needed? |
|-------------|--------------|---------------------|
| `hostname` | `server_name` | No |
| `cpu_usage` | `cpu_pct` | No (already 0-100) |
| `cpu` | `cpu_pct` | Yes (multiply by 100 if 0-1) |
| ... | ... | ... |

### Step 3: Customize Template (2-3 hours)

Copy `03_MINIMAL_TEMPLATE.py` and customize:

```python
# 1. Configure data source
DATA_SOURCE_URL = "http://linborg.wellsfargo.com/api/metrics"
DATA_SOURCE_AUTH = {"username": "...", "password": "..."}

# 2. Implement poll_data_source()
def poll_data_source():
    response = requests.get(DATA_SOURCE_URL, auth=DATA_SOURCE_AUTH)
    return response.json()['servers']  # Adjust to their format

# 3. Implement transform_record()
def transform_record(raw):
    return {
        'timestamp': raw['their_time_field'],
        'server_name': raw['their_hostname_field'],
        'cpu_pct': raw['their_cpu_field'],
        # ... map other 6 fields
    }
```

### Step 4: Test (1 hour)

Follow `04_TESTING_GUIDE.md`:
1. Test data source connection
2. Test transformation
3. Test inference daemon integration
4. Test full flow
5. Verify dashboard

### Step 5: Deploy (30 min)

Add to `start_all.sh`, configure logging, done!

---

## Key Insights

### Profile Matching - Skip It!

The inference daemon **already does profile matching** based on server name prefixes:
- `ppdb*` → database
- `ppweb*` → web_api
- `ppml*` → ml_compute
- etc.

**Unless the user has special requirements, just omit the `profile` field!**

### Batching - Important!

**Don't do this** (inefficient):
```python
for server in servers:
    requests.post('/feed/data', json={'records': [server]})  # 100 requests
```

**Do this instead**:
```python
requests.post('/feed/data', json={'records': servers})  # 1 request
```

Rate limit is 60 requests/minute, so batch all servers together.

### Missing Data - OK!

If they don't have disk I/O metrics, that's fine:
```python
'disk_read_mbps': raw.get('disk_read', 0.0),  # Defaults to 0.0
'disk_write_mbps': raw.get('disk_write', 0.0)
```

---

## Common Data Sources

### If they use REST API

```python
def poll_data_source():
    response = requests.get(
        "http://their-api/metrics",
        auth=("user", "pass")
    )
    return response.json()['servers']
```

### If they use Elasticsearch

```python
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://localhost:9200'])

def poll_data_source():
    result = es.search(index="metrics-*", body={
        "query": {"range": {"@timestamp": {"gte": "now-5s"}}}
    })
    return [hit['_source'] for hit in result['hits']['hits']]
```

### If they use MongoDB

```python
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017')

def poll_data_source():
    return list(client.monitoring.metrics.find())
```

### If they use Kafka

Use `04_STREAMING_TEMPLATE.py` instead (more complex, has batching logic).

---

## What NOT to Do

❌ **Don't over-engineer**
- You're building a simple ETL pipeline
- 100-200 lines of code is sufficient
- Don't add unnecessary abstractions

❌ **Don't implement profile matching unless needed**
- Inference daemon already does it
- Only implement if they have special CMDB requirements

❌ **Don't worry about validation**
- Inference daemon validates data
- Just map the fields correctly

❌ **Don't forget to batch**
- Send all servers in one POST
- Don't send one POST per server

---

## What the Inference Daemon Does (Already Built)

You don't need to implement ANY of this:

- ✅ Profile auto-detection
- ✅ Data validation
- ✅ Rolling window management
- ✅ TFT predictions
- ✅ Risk scoring
- ✅ Alert generation
- ✅ Dashboard serving
- ✅ Data buffering for retraining
- ✅ Hot model reload
- ✅ Automated retraining

**Your ONLY job is to get data from Point A (their source) to Point B (inference daemon)!**

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

# Check dashboard (wait 30 min for warmup)
open http://localhost:8050
```

---

## Success Criteria

Your adapter is complete when:

1. ✅ Connects to their data source successfully
2. ✅ Transforms all 9 required fields correctly
3. ✅ POSTs to inference daemon without errors
4. ✅ Dashboard shows servers (after 30 min warmup)
5. ✅ Predictions appear on dashboard
6. ✅ Runs continuously without crashes

---

## Common Mistakes to Avoid

### Mistake 1: Wrong units

```python
# WRONG: CPU is 0-1, need 0-100
'cpu_pct': raw['cpu']

# RIGHT: Convert to 0-100
'cpu_pct': raw['cpu'] * 100
```

### Mistake 2: Wrong timestamp format

```python
# WRONG: Unix timestamp
'timestamp': raw['ts']  # 1698765432

# RIGHT: ISO 8601
'timestamp': datetime.fromtimestamp(raw['ts']).isoformat()
```

### Mistake 3: Not batching

```python
# WRONG: One POST per server
for server in servers:
    send_to_inference_daemon([server], api_key)

# RIGHT: One POST for all servers
send_to_inference_daemon(servers, api_key)
```

### Mistake 4: Over-engineering profile matching

```python
# WRONG: Complex CMDB lookups (usually not needed)
def match_profile(name):
    cmdb = query_cmdb(name)
    tags = get_service_tags(name)
    # 100 lines of logic...

# RIGHT: Let inference daemon handle it
def match_profile(name):
    return 'generic'  # Auto-detected!
```

---

## Debugging Tips

### If data source connection fails:

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test directly
response = requests.get(DATA_SOURCE_URL, auth=DATA_SOURCE_AUTH)
print(response.status_code)
print(response.text)
```

### If transformation fails:

```python
# Print raw record
def transform_record(raw):
    print(f"Raw record: {raw}")

    # Show available fields
    print(f"Available fields: {list(raw.keys())}")

    # Try transformation
    result = {...}
    print(f"Transformed: {result}")
    return result
```

### If inference daemon rejects data:

Check logs:
```bash
tail -f logs/inference_daemon.log
```

Look for validation errors.

---

## Timeline

| Task | Time |
|------|------|
| Read documentation | 30 min |
| Get sample data from user | 15 min |
| Map fields | 15 min |
| Customize template | 2 hours |
| Test connection | 30 min |
| Test transformation | 30 min |
| Test full integration | 30 min |
| Wait for dashboard warmup | 30 min |
| Deploy | 30 min |
| **TOTAL** | **~5-6 hours** |

---

## Final Checklist

Before saying you're done:

- [ ] Read `00_READ_ME_FIRST.md`
- [ ] Read `01_QUICK_START.md`
- [ ] Read `02_API_CONTRACT.md`
- [ ] Asked user for sample data
- [ ] Mapped all 9 fields
- [ ] Customized `03_MINIMAL_TEMPLATE.py`
- [ ] Tested data source connection
- [ ] Tested transformation
- [ ] Tested inference daemon integration
- [ ] Verified dashboard shows data (after warmup)
- [ ] Adapter runs continuously without errors
- [ ] Added to startup scripts
- [ ] Documented configuration

---

## If You Get Stuck

1. Check `02_API_CONTRACT.md` for exact field specs
2. Check `04_TESTING_GUIDE.md` for debugging steps
3. Check inference daemon logs: `logs/inference_daemon.log`
4. Verify inference daemon is running: `./status.sh`
5. Test API directly: `curl http://localhost:8000/health`

---

## Remember

**This is a simple ETL pipeline:**
- **Extract**: Query their data source
- **Transform**: Map 9 field names
- **Load**: POST to REST API

**Don't over-complicate it!**

The template does 90% of the work. You just fill in 3 functions based on their data format.

---

## Good Luck!

You have everything you need to succeed:
- ✅ Complete API specification
- ✅ Ready-to-use template
- ✅ Testing guide
- ✅ Common examples

**4-6 hours of focused work and you'll be done!**

---

© 2025 NordIQ AI, LLC. All rights reserved.
