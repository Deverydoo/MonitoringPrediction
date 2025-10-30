# Testing Guide - Verify Your Adapter Works

**Read time**: 5 minutes

---

## Testing Checklist

- [ ] Test 1: Data source connection
- [ ] Test 2: Transformation logic
- [ ] Test 3: Inference daemon connection
- [ ] Test 4: Full integration
- [ ] Test 5: Dashboard verification

---

## Test 1: Data Source Connection

**Purpose**: Verify you can connect to and query your data source.

**Code**: Add to bottom of your adapter:

```python
if __name__ == "__main__":
    print("Test 1: Data Source Connection")
    print("=" * 60)

    raw_records = poll_data_source()

    if not raw_records:
        print("❌ FAIL: No records returned")
        print("Check your DATA_SOURCE_URL and authentication")
    else:
        print(f"✅ PASS: Got {len(raw_records)} records")
        print("\nSample record:")
        print(json.dumps(raw_records[0], indent=2, default=str))
```

**Run**:
```bash
python your_adapter.py
```

**Expected Output**:
```
Test 1: Data Source Connection
============================================================
✅ PASS: Got 25 records

Sample record:
{
  "hostname": "ppdb001",
  "cpu_usage": 45.2,
  "mem_usage": 78.5,
  ...
}
```

**If it fails**:
- Check `DATA_SOURCE_URL` is correct
- Check authentication credentials
- Check network connectivity
- Check data source is running

---

## Test 2: Transformation Logic

**Purpose**: Verify field mapping works correctly.

**Code**: Add to bottom of your adapter:

```python
if __name__ == "__main__":
    print("Test 2: Transformation Logic")
    print("=" * 60)

    # Get sample raw record
    raw_records = poll_data_source()
    if not raw_records:
        print("❌ FAIL: No records to transform")
        exit(1)

    raw = raw_records[0]
    print("Raw record:")
    print(json.dumps(raw, indent=2, default=str))

    # Transform
    try:
        nordiq = transform_record(raw)
        print("\nTransformed record:")
        print(json.dumps(nordiq, indent=2, default=str))

        # Verify required fields
        required = ['timestamp', 'server_name', 'cpu_pct', 'memory_pct',
                    'disk_pct', 'network_in_mbps', 'network_out_mbps',
                    'disk_read_mbps', 'disk_write_mbps']

        missing = [f for f in required if f not in nordiq]
        if missing:
            print(f"\n❌ FAIL: Missing required fields: {missing}")
        else:
            print(f"\n✅ PASS: All required fields present!")

            # Verify types
            print("\nField validation:")
            print(f"  timestamp: {nordiq['timestamp']} (should be ISO 8601)")
            print(f"  server_name: {nordiq['server_name']} (should be non-empty)")
            print(f"  cpu_pct: {nordiq['cpu_pct']} (should be 0-100)")
            print(f"  memory_pct: {nordiq['memory_pct']} (should be 0-100)")
            print(f"  disk_pct: {nordiq['disk_pct']} (should be 0-100)")

    except Exception as e:
        print(f"\n❌ FAIL: Transformation error: {e}")
        import traceback
        traceback.print_exc()
```

**Run**:
```bash
python your_adapter.py
```

**Expected Output**:
```
Test 2: Transformation Logic
============================================================
Raw record:
{
  "hostname": "ppdb001",
  "cpu_usage": 45.2,
  ...
}

Transformed record:
{
  "timestamp": "2025-10-30T14:35:00",
  "server_name": "ppdb001",
  "cpu_pct": 45.2,
  ...
}

✅ PASS: All required fields present!

Field validation:
  timestamp: 2025-10-30T14:35:00 (should be ISO 8601)
  server_name: ppdb001 (should be non-empty)
  cpu_pct: 45.2 (should be 0-100)
  memory_pct: 78.5 (should be 0-100)
  disk_pct: 62.3 (should be 0-100)
```

**If it fails**:
- Check field names match your data source
- Check unit conversions (% vs 0-1, Mbps vs MB/s, etc.)
- Check for typos in field names

---

## Test 3: Inference Daemon Connection

**Purpose**: Verify NordIQ inference daemon is running and accessible.

**Test Commands**:

```bash
# Test 1: Health check (no auth)
curl http://localhost:8000/health

# Expected:
# {"status":"healthy","service":"tft_inference_daemon","running":true}

# Test 2: Status check (with auth)
API_KEY=$(cat .nordiq_key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/status

# Expected:
# {"status":"running","model_loaded":true,...}

# Test 3: Send test record
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

# Expected:
# {"success":true,"records_received":1,"servers_updated":["test001"],...}
```

**If it fails**:
- Check if NordIQ is running: `./status.sh`
- Start NordIQ if needed: `./start_all.sh`
- Check port 8000 is not blocked
- Check API key exists: `cat .nordiq_key`

---

## Test 4: Full Integration

**Purpose**: Run your adapter and verify it sends data successfully.

**Steps**:

1. **Start NordIQ** (if not running):
```bash
cd NordIQ
./start_all.sh
```

2. **Run your adapter**:
```bash
python your_adapter.py
```

3. **Expected Output**:
```
============================================================
NordIQ Data Adapter Starting
============================================================
✓ API key loaded successfully
✓ Connected to inference daemon at http://localhost:8000
✓ Polling every 5 seconds
Press Ctrl+C to stop
============================================================
[Iteration 1] Polling data source...
  → Polled 25 raw records
  → Transformed 25 records
✓ Sent 25 records, updated 25 servers
[Iteration 2] Polling data source...
  → Polled 25 raw records
  → Transformed 25 records
✓ Sent 25 records, updated 25 servers
...
```

4. **Check inference daemon logs**:
```bash
tail -f logs/inference_daemon.log
```

Look for:
```
[FEED] Received 25 records from API
[FEED] Updated servers: ppdb001, ppweb003, ...
```

5. **Let it run for 5-10 minutes** to verify stability.

6. **Stop with Ctrl+C**:
```
^C
============================================================
Shutting down gracefully...
============================================================
NordIQ Data Adapter Stopped
============================================================
```

**If it fails**:
- Check error messages in adapter output
- Check inference daemon logs: `tail -f logs/inference_daemon.log`
- Verify transformations are correct (Test 2)
- Check for network issues

---

## Test 5: Dashboard Verification

**Purpose**: Verify predictions appear on dashboard.

**Timeline**:
- **0-20 minutes**: Warmup period (rolling window filling)
- **20-30 minutes**: Predictions start appearing
- **30+ minutes**: Full predictions available

**Steps**:

1. **Run adapter for at least 30 minutes**:
```bash
python your_adapter.py
# Wait 30 minutes...
```

2. **Open dashboard**:
```bash
# Linux/Mac
open http://localhost:8050

# Windows
start http://localhost:8050

# Or manually: http://localhost:8050
```

3. **Verify dashboard shows**:
- [ ] Server list (your servers from data source)
- [ ] Risk scores
- [ ] Predictions (8-hour forecast)
- [ ] Alerts (if any issues detected)

4. **Check predictions via API**:
```bash
API_KEY=$(cat .nordiq_key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predictions/current | jq
```

Expected:
```json
{
  "timestamp": "2025-10-30T14:35:00",
  "servers": [
    {
      "server_name": "ppdb001",
      "current": {
        "cpu_pct": 45.2,
        "memory_pct": 78.5,
        ...
      },
      "predictions": [...],
      "risk_score": 35.2,
      "risk_level": "low",
      "alerts": []
    },
    ...
  ]
}
```

**If dashboard is empty**:
- **Wait longer**: Warmup takes 20-30 minutes
- **Check adapter is still running**: Should see continuous "Iteration N" messages
- **Check data is being received**: `tail -f logs/inference_daemon.log`
- **Check for errors**: In both adapter and daemon logs

---

## Debugging Common Issues

### Issue: "Cannot connect to inference daemon"

**Check**:
```bash
# Is daemon running?
./status.sh

# Is port 8000 open?
curl http://localhost:8000/health

# Start daemon if needed
./start_all.sh
```

### Issue: "401 Unauthorized"

**Check**:
```bash
# Does API key file exist?
ls -la .nordiq_key

# Is it valid?
cat .nordiq_key  # Should show: tft-xxxxx...

# Regenerate if needed
python bin/generate_api_key.py
```

### Issue: "No records from data source"

**Check**:
```bash
# Test data source directly
curl http://your-data-source-url/api/metrics

# Check authentication
# Check network connectivity
# Check data source is running
```

### Issue: "Transformation errors"

**Debug**:
```python
# Add detailed logging to transform_record()
def transform_record(raw):
    try:
        print(f"Raw record: {raw}")

        result = {
            'timestamp': raw['your_timestamp_field'],
            # ...
        }

        print(f"Transformed: {result}")
        return result
    except Exception as e:
        print(f"Error transforming: {e}")
        print(f"Raw record was: {raw}")
        raise
```

### Issue: "Dashboard shows no data after 30 minutes"

**Check**:
```bash
# Check warmup status
API_KEY=$(cat .nordiq_key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/status | jq '.warmup_status'

# Expected:
# {
#   "is_ready": true,
#   "servers_ready": 25,
#   "total_servers": 25
# }
```

If `is_ready` is false:
- Adapter hasn't been running long enough
- Not enough data points per server (need 30+ per server)
- Check adapter is sending data continuously

---

## Performance Testing

### Test Rate Limiting

**Purpose**: Verify you stay under 60 requests/minute.

```python
import time

start = time.time()
request_count = 0

for i in range(100):
    send_to_inference_daemon(records, api_key)
    request_count += 1

elapsed = time.time() - start
rate = request_count / (elapsed / 60)

print(f"Request rate: {rate:.1f} requests/minute")
if rate > 60:
    print("❌ FAIL: Exceeding rate limit!")
else:
    print("✅ PASS: Within rate limit")
```

### Test Batching

**Purpose**: Verify batching works for large server counts.

```python
# Test with 500 servers
test_records = [generate_test_record(f"server{i:04d}") for i in range(500)]

start = time.time()
send_to_inference_daemon(test_records, api_key)
elapsed = time.time() - start

print(f"Sent {len(test_records)} records in {elapsed:.2f}s")
if elapsed < 5:
    print("✅ PASS: Batch sending is efficient")
else:
    print("⚠ WARN: Batch sending is slow, consider splitting into smaller batches")
```

---

## Final Verification Checklist

Before deploying to production:

- [ ] Data source connection stable for 1+ hours
- [ ] Transformation produces valid records (all 9 fields)
- [ ] Adapter sends data without errors
- [ ] Inference daemon receives data successfully
- [ ] Dashboard shows servers after warmup
- [ ] Predictions appear on dashboard
- [ ] No memory leaks (check with `top` or `htop`)
- [ ] No error messages in logs
- [ ] Rate limiting respected (<60 req/min)
- [ ] Handles data source downtime gracefully

---

## Success Criteria

Your adapter is ready when:

1. ✅ Runs continuously for 1+ hours without errors
2. ✅ Dashboard shows your servers with predictions
3. ✅ Risk scores calculated correctly
4. ✅ Alerts generated when appropriate
5. ✅ Logs show regular successful sends
6. ✅ No memory or CPU issues

---

## Next Steps

Once testing is complete:
1. Add to startup scripts (`start_all.sh`, `stop_all.sh`)
2. Configure log rotation
3. Deploy to production
4. Monitor for 24 hours
5. Done!

---

© 2025 NordIQ AI, LLC. All rights reserved.
