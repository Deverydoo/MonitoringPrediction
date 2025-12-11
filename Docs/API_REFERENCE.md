# Tachyon Argus - API Reference

Complete REST API documentation for the TFT Inference Daemon.

**Base URL:** `http://localhost:8000`

## Authentication

All endpoints except `/health` and `/status` require API key authentication.

**Header:**
```
X-API-Key: your-api-key
```

**Environment Variable:**
```bash
export TACHYON_API_KEY="your-api-key"
```

---

## Health & Status

### GET /health

Health check endpoint (no authentication required).

**Response:**
```json
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

### GET /status

Daemon operational status (no authentication required).

**Response:**
```json
{
  "model_loaded": true,
  "model_path": "models/tft_model_20251215",
  "rolling_window_size": 2880,
  "servers_tracked": 45,
  "last_prediction": "2025-01-15T10:05:00",
  "uptime_seconds": 3600,
  "warmup_complete": true
}
```

---

## Data Ingestion

### POST /feed/data

Feed metrics data to the inference engine.

**Rate Limit:** 60 requests/minute

**Request Body:**
```json
{
  "records": [
    {
      "timestamp": "2025-01-15T10:00:00",
      "server_name": "ppdb001",
      "status": "healthy",
      "cpu_user_pct": 25.3,
      "cpu_sys_pct": 8.2,
      "cpu_iowait_pct": 2.1,
      "cpu_idle_pct": 64.4,
      "java_cpu_pct": 15.5,
      "mem_used_pct": 72.1,
      "swap_used_pct": 5.2,
      "disk_usage_pct": 45.8,
      "net_in_mb_s": 12.5,
      "net_out_mb_s": 8.3,
      "back_close_wait": 5,
      "front_close_wait": 3,
      "load_average": 2.4,
      "uptime_days": 45
    }
  ]
}
```

**Required Fields:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `timestamp` | string | ISO 8601 | e.g., "2025-01-15T10:00:00" |
| `server_name` | string | - | Unique server identifier |
| `status` | string | enum | See status values below |
| `cpu_user_pct` | float | 0-100 | User CPU percentage |
| `cpu_sys_pct` | float | 0-100 | System CPU percentage |
| `cpu_iowait_pct` | float | 0-100 | I/O wait percentage |
| `cpu_idle_pct` | float | 0-100 | Idle CPU percentage |
| `java_cpu_pct` | float | 0-100 | Java process CPU |
| `mem_used_pct` | float | 0-100 | Memory usage percentage |
| `swap_used_pct` | float | 0-100 | Swap usage percentage |
| `disk_usage_pct` | float | 0-100 | Disk usage percentage |
| `net_in_mb_s` | float | 0+ | Network in (MB/s) |
| `net_out_mb_s` | float | 0+ | Network out (MB/s) |
| `back_close_wait` | int | 0+ | Backend CLOSE_WAIT |
| `front_close_wait` | int | 0+ | Frontend CLOSE_WAIT |
| `load_average` | float | 0+ | System load average |
| `uptime_days` | int | 0-365 | Days since reboot |

**Valid Status Values:**
- `healthy`
- `critical_issue`
- `heavy_load`
- `idle`
- `maintenance`
- `morning_spike`
- `offline`
- `recovery`

**Response (Success):**
```json
{
  "status": "accepted",
  "records_received": 45,
  "tick": 1234,
  "rolling_window_size": 2880,
  "warmup_complete": true
}
```

**Response (Warmup):**
```json
{
  "status": "warming_up",
  "records_received": 45,
  "tick": 50,
  "warmup_complete": false,
  "progress": "45% (need 100 records minimum)"
}
```

---

## Predictions

### GET /predictions/current

Get current predictions for all servers.

**Rate Limit:** 30 requests/minute

**Response:**
```json
{
  "predictions": {
    "ppdb001": {
      "risk_score": 72.5,
      "profile": "Database",
      "alert": {
        "level": "warning",
        "score": 72.5,
        "color": "#FFA500",
        "emoji": "🟠",
        "label": "🟠 Warning"
      },
      "display_metrics": {
        "cpu": "45.2%",
        "memory": "78.1%",
        "disk_io": "23.4 MB/s"
      },
      "forecast": {
        "cpu_pct": [45.2, 46.1, 47.3],
        "mem_pct": [78.1, 78.4, 79.2],
        "timestamps": ["2025-01-15T10:00:00", "2025-01-15T10:05:00", "2025-01-15T10:10:00"]
      }
    }
  },
  "summary": {
    "total_servers": 45,
    "critical": 2,
    "warning": 5,
    "degraded": 3,
    "healthy": 35,
    "fleet_health": 87.2,
    "prob_30m": 0.15,
    "prob_8h": 0.42
  },
  "alerts": [
    {
      "server_name": "ppdb003",
      "level": "critical",
      "risk_score": 92.1,
      "message": "Memory exhaustion predicted in 45 minutes"
    }
  ],
  "timestamp": "2025-01-15T10:05:00"
}
```

**Error (Warmup):**
```json
{
  "error": "insufficient_data",
  "message": "Need at least 100 records, have 45",
  "predictions": {}
}
```

---

## Alerts

### GET /alerts/active

Get currently active alerts.

**Rate Limit:** 30 requests/minute

**Response:**
```json
{
  "timestamp": "2025-01-15T10:05:00",
  "count": 3,
  "alerts": [
    {
      "server_name": "ppdb003",
      "level": "critical",
      "risk_score": 92.1,
      "profile": "Database",
      "message": "Memory exhaustion predicted",
      "time_to_issue": "45 minutes"
    }
  ]
}
```

---

## Explainability (XAI)

### GET /explain/{server_name}

Get XAI explanation for a server's prediction.

**Rate Limit:** 30 requests/minute

**Parameters:**
- `server_name` (path): Server identifier (e.g., "ppdb001")

**Response:**
```json
{
  "server_name": "ppdb001",
  "prediction": {
    "risk_score": 72.5,
    "alert_level": "warning"
  },
  "shap": {
    "top_features": [
      {"feature": "mem_pct", "importance": 0.85, "stars": "★★★★★"},
      {"feature": "cpu_pct", "importance": 0.62, "stars": "★★★☆☆"},
      {"feature": "disk_io", "importance": 0.34, "stars": "★★☆☆☆"}
    ]
  },
  "attention": {
    "focus_periods": ["Last 30 minutes", "2 hours ago"],
    "analysis": "Model focused on recent memory spike"
  },
  "counterfactuals": [
    {
      "scenario": "Reduce memory by 15%",
      "impact": "Risk drops from 72.5 to 45.2",
      "recommendation": "Consider restarting memory-heavy processes"
    }
  ]
}
```

---

## Historical Data

### GET /historical/summary

Get summary statistics for executive reporting.

**Parameters:**
- `time_range` (query): `30m`, `1h`, `8h`, `1d`, `1w`, `1M` (default: `1d`)

**Response:**
```json
{
  "success": true,
  "time_range": "1d",
  "total_alerts": 47,
  "alerts_by_level": {
    "critical": 5,
    "warning": 22,
    "degraded": 20
  },
  "avg_resolution_time_minutes": 23.4,
  "incidents_prevented": 3,
  "fleet_health_avg": 85.2
}
```

### GET /historical/alerts

Get alert events for a time range.

**Parameters:**
- `time_range` (query): `30m`, `1h`, `8h`, `1d`, `1w`, `1M` (default: `1h`)
- `server_name` (query, optional): Filter by server

**Response:**
```json
{
  "success": true,
  "time_range": "8h",
  "count": 15,
  "alerts": [
    {
      "timestamp": "2025-01-15T08:30:00",
      "server_name": "ppdb001",
      "event_type": "escalated",
      "previous_level": "warning",
      "new_level": "critical",
      "risk_score": 85.3,
      "resolved_at": "2025-01-15T09:15:00",
      "resolution_duration_minutes": 45
    }
  ]
}
```

### GET /historical/server/{server_name}

Get detailed history for a specific server.

**Parameters:**
- `server_name` (path): Server identifier
- `time_range` (query): `30m`, `1h`, `8h`, `1d`, `1w`, `1M` (default: `1d`)

**Response:**
```json
{
  "success": true,
  "server_name": "ppdb001",
  "time_range": "1d",
  "alert_count": 3,
  "avg_risk_score": 42.5,
  "max_risk_score": 85.3,
  "time_in_warning_pct": 15.2,
  "time_in_critical_pct": 2.1,
  "risk_trend": "improving"
}
```

### GET /historical/environment

Get environment health snapshots over time.

**Parameters:**
- `time_range` (query): `30m`, `1h`, `8h`, `1d`, `1w`, `1M` (default: `1h`)

**Response:**
```json
{
  "success": true,
  "time_range": "1h",
  "count": 12,
  "snapshots": [
    {
      "timestamp": "2025-01-15T09:00:00",
      "total_servers": 45,
      "critical_count": 1,
      "warning_count": 4,
      "healthy_count": 40,
      "fleet_health": 88.9
    }
  ]
}
```

### GET /historical/export/{table}

Export historical data as CSV.

**Parameters:**
- `table` (path): `alerts` or `environment`
- `time_range` (query): `30m`, `1h`, `8h`, `1d`, `1w`, `1M` (default: `1d`)

**Response:**
```json
{
  "success": true,
  "table": "alerts",
  "time_range": "1w",
  "csv_data": "timestamp,server_name,event_type,...\n2025-01-15T08:30:00,ppdb001,...",
  "filename": "argus_alerts_1w_20250115_103000.csv"
}
```

---

## Administration

### GET /admin/models

List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "tft_model_20251215_143022",
      "path": "models/tft_model_20251215_143022",
      "created": "2025-12-15T14:30:22",
      "size_mb": 2.4,
      "is_current": true
    }
  ],
  "current_model": "tft_model_20251215_143022"
}
```

### POST /admin/reload-model

Hot reload a model without daemon restart.

**Request Body (optional):**
```json
{
  "model_path": "models/tft_model_20251215_143022"
}
```

If no `model_path` provided, reloads the latest model.

**Response:**
```json
{
  "success": true,
  "model": "tft_model_20251215_143022",
  "message": "Model reloaded successfully"
}
```

### POST /admin/trigger-training

Trigger model retraining.

**Parameters:**
- `epochs` (query, optional): Number of epochs (default: 10)
- `incremental` (query, optional): Continue from existing model (default: true)

**Response:**
```json
{
  "success": true,
  "training_id": "train_20250115_103000",
  "message": "Training started",
  "epochs": 10,
  "incremental": true
}
```

### GET /admin/training-status

Get current training status.

**Response:**
```json
{
  "training_active": true,
  "training_id": "train_20250115_103000",
  "progress": {
    "epoch": 5,
    "total_epochs": 10,
    "percent": 50,
    "eta_minutes": 15
  }
}
```

### GET /admin/training-stats

Get training statistics.

**Response:**
```json
{
  "last_training": "2025-01-15T08:00:00",
  "total_trainings": 15,
  "avg_training_time_minutes": 45,
  "models_produced": 12
}
```

### POST /admin/cancel-training

Cancel running training job.

**Response:**
```json
{
  "success": true,
  "message": "Training cancelled"
}
```

---

## Error Responses

### 401 Unauthorized

```json
{
  "detail": "Invalid or missing API key"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "records", 0, "cpu_user_pct"],
      "msg": "value is not a valid float",
      "type": "type_error.float"
    }
  ]
}
```

### 429 Rate Limited

```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal server error",
  "error": "Model not loaded"
}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| POST /feed/data | 60/minute |
| GET /predictions/* | 30/minute |
| GET /alerts/* | 30/minute |
| GET /explain/* | 30/minute |
| GET /historical/* | 30/minute |
| POST /admin/* | 10/minute |

---

## Alert Levels

| Level | Score Range | Color | Hex |
|-------|-------------|-------|-----|
| Critical | 80-100 | Red | #FF0000 |
| Warning | 60-79 | Orange | #FFA500 |
| Degraded | 50-59 | Yellow | #FFD700 |
| Healthy | 0-49 | Green | #00FF00 |

---

## SDK Examples

### Python

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-api-key"
HEADERS = {"X-API-Key": API_KEY}

# Get predictions
response = requests.get(f"{API_URL}/predictions/current", headers=HEADERS)
predictions = response.json()

# Feed data
data = {"records": [...]}
response = requests.post(f"{API_URL}/feed/data", json=data, headers=HEADERS)
```

### JavaScript

```javascript
const API_URL = 'http://localhost:8000';
const API_KEY = 'your-api-key';
const headers = { 'X-API-Key': API_KEY };

// Get predictions
const response = await fetch(`${API_URL}/predictions/current`, { headers });
const predictions = await response.json();
```

### cURL

```bash
# Get predictions
curl -H "X-API-Key: your-key" http://localhost:8000/predictions/current

# Feed data
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"records": [...]}'
```
