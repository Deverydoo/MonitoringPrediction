# Quick Reference: TFT Inference API

**One-page reference for developers integrating with the TFT Inference Daemon**

---

## Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/feed/data` | Submit server metrics |
| `GET` | `/predictions/current` | Get predictions for all servers |
| `GET` | `/status` | Check daemon health and warmup |
| `GET` | `/health` | Simple health check |

**Base URL:** `http://localhost:8000`

---

## Data Format

### POST /feed/data

```json
{
  "records": [
    {
      "timestamp": "2025-10-13T16:00:00Z",
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

### Field Requirements

| Field | Type | Range | Example |
|-------|------|-------|---------|
| `timestamp` | ISO 8601 string | Any | `"2025-10-13T16:00:00Z"` |
| `server_name` | string | Must match training | `"ppdb001"` |
| `cpu_pct` | float | 0.0 - 100.0 | `45.2` |
| `mem_pct` | float | 0.0 - 100.0 | `67.8` |
| `disk_io_mb_s` | float | 0.0+ | `123.4` |
| `latency_ms` | float | 0.0+ | `12.5` |
| `state` | string | See below | `"healthy"` |

### Valid States

```
healthy, heavy_load, critical_issue, idle,
maintenance, morning_spike, offline, recovery
```

---

## Quick Examples

### Python

```python
import requests
from datetime import datetime

requests.post("http://localhost:8000/feed/data", json={
    "records": [{
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "server_name": "ppdb001",
        "cpu_pct": 45.2,
        "mem_pct": 67.8,
        "disk_io_mb_s": 123.4,
        "latency_ms": 12.5,
        "state": "healthy"
    }]
})
```

### cURL

```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -d '{"records":[{"timestamp":"2025-10-13T16:00:00Z","server_name":"ppdb001","cpu_pct":45.2,"mem_pct":67.8,"disk_io_mb_s":123.4,"latency_ms":12.5,"state":"healthy"}]}'
```

### Node.js

```javascript
await axios.post('http://localhost:8000/feed/data', {
  records: [{
    timestamp: new Date().toISOString(),
    server_name: 'ppdb001',
    cpu_pct: 45.2,
    mem_pct: 67.8,
    disk_io_mb_s: 123.4,
    latency_ms: 12.5,
    state: 'healthy'
  }]
});
```

---

## Response Formats

### /feed/data Response

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

### /predictions/current Response

```json
{
  "predictions": {
    "ppdb001": {
      "cpu_percent": {
        "p50": [45.3, 46.1, ...],  // 96 values
        "p10": [42.1, 43.0, ...],
        "p90": [48.5, 49.2, ...],
        "current": 45.2,
        "trend": 0.02
      }
    }
  },
  "environment": {
    "prob_30m": 0.15,
    "prob_8h": 0.42
  }
}
```

### /status Response

```json
{
  "running": true,
  "tick_count": 305,
  "warmup": {
    "is_warmed_up": true,
    "progress_percent": 100
  }
}
```

---

## Important Notes

- **Frequency:** Send every 5 seconds
- **Warmup:** Needs 150 records per server (~12.5 minutes)
- **Server Names:** Must exactly match training data (case-sensitive)
- **Timeout:** Use 5-second timeout on HTTP client
- **Batching:** Send all servers in one request (recommended)

---

## Trained Server Names

```
ppml0001, ppml0002, ppml0003, ppml0004
ppdb001, ppdb002, ppdb003
ppweb001, ppweb002, ppweb003, ppweb004,
ppweb005, ppweb006, ppweb007, ppweb008
ppcon01
ppetl001, ppetl002
pprisk001
ppgen001
```

---

## Error Handling

| Error | Meaning | Solution |
|-------|---------|----------|
| `insufficient_data` | <100 total records | Keep sending data |
| `warmup_complete: false` | Need more per-server data | Wait ~12 min per server |
| 500 Internal Error | Model prediction failed | Check logs, verify data format |
| Connection refused | Daemon not running | Start with `python tft_inference_daemon.py` |

---

## Health Check

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status | jq

# Check warmup
curl http://localhost:8000/status | jq '.warmup.is_warmed_up'
```

---

For full documentation, see **PRODUCTION_INTEGRATION_GUIDE.md**
