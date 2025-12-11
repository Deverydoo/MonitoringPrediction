# Tachyon Argus Inference Engine - Dashboard Integration Guide

## Overview

The TFT Inference Daemon exposes a REST API for building monitoring dashboards. The daemon handles all prediction logic, risk scoring, and alert generation - your dashboard just needs to fetch and display.

**Base URL:** `http://localhost:8000` (configurable via `--port`)

## Authentication

All endpoints (except `/health` and `/status`) require an API key.

### Setting Up the API Key

**Option 1: Environment Variable (Recommended)**
```bash
export TACHYON_API_KEY="your-secure-key-here"
```

**Option 2: Key File**
Create `Argus/.tachyon_key` with your key:
```
your-secure-key-here
```

### Using the API Key

Include in request header:
```
X-API-Key: your-secure-key-here
```

---

## Core Endpoints

### 1. Health Check (No Auth Required)

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

Use this for dashboard connection status indicator.

---

### 2. Get Current Predictions

```
GET /predictions/current
```

This is your main data source. Returns predictions for all servers with pre-calculated risk scores.

**Response Structure:**
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
        "cpu_pct": [45.2, 46.1, 47.3, ...],  // 96 steps (8 hours)
        "mem_pct": [78.1, 78.4, 79.2, ...],
        "timestamps": ["2025-01-15T10:00:00", ...]
      }
    },
    "ppdb002": { ... },
    "ppml001": { ... }
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

**Key Fields for Dashboard:**
- `predictions[server].risk_score` - 0-100 risk score
- `predictions[server].alert.level` - "critical", "warning", "degraded", "healthy"
- `predictions[server].alert.color` - Hex color for UI
- `summary` - Fleet-wide statistics
- `alerts` - Active alerts to display

---

### 3. Get Active Alerts

```
GET /alerts/active
```

Returns currently active alerts:
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

### 4. Get XAI Explanation

```
GET /explain/{server_name}
```

Example: `GET /explain/ppdb001`

Returns explainability data for a specific server:
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

### 5. Daemon Status

```
GET /status
```

Returns operational status:
```json
{
  "model_loaded": true,
  "model_path": "models/tft_model_20250115",
  "rolling_window_size": 2880,
  "servers_tracked": 45,
  "last_prediction": "2025-01-15T10:05:00",
  "uptime_seconds": 3600
}
```

---

## Historical Data Endpoints

For executive dashboards and reporting.

### Summary Statistics

```
GET /historical/summary?time_range=1d
```

Time ranges: `30m`, `1h`, `8h`, `1d`, `1w`, `1M`

```json
{
  "success": true,
  "time_range": "1d",
  "total_alerts": 47,
  "alerts_by_level": {"critical": 5, "warning": 22, "degraded": 20},
  "avg_resolution_time_minutes": 23.4,
  "incidents_prevented": 3
}
```

### Alert History

```
GET /historical/alerts?time_range=8h&server_name=ppdb001
```

### Server History

```
GET /historical/server/{server_name}?time_range=1d
```

### Export CSV

```
GET /historical/export/alerts?time_range=1w
GET /historical/export/environment?time_range=1d
```

---

## Admin Endpoints

### List Available Models

```
GET /admin/models
```

### Reload Model (Hot Swap)

```
POST /admin/reload-model
POST /admin/reload-model?model_path=models/tft_model_20250115
```

### Trigger Retraining

```
POST /admin/trigger-training?epochs=10&incremental=true
```

### Training Status

```
GET /admin/training-status
GET /admin/training-stats
```

---

## Dashboard Implementation Examples

### React/JavaScript

```javascript
const API_BASE = 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_TACHYON_API_KEY;

const headers = {
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json'
};

// Health check
async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

// Get predictions (poll every 30 seconds)
async function getPredictions() {
  const res = await fetch(`${API_BASE}/predictions/current`, { headers });
  return res.json();
}

// Get explanation for a server
async function getExplanation(serverName) {
  const res = await fetch(`${API_BASE}/explain/${serverName}`, { headers });
  return res.json();
}

// Polling loop
setInterval(async () => {
  const data = await getPredictions();
  updateDashboard(data);
}, 30000);
```

### Python

```python
import requests

API_BASE = "http://localhost:8000"
API_KEY = "your-api-key"
HEADERS = {"X-API-Key": API_KEY}

def get_predictions():
    response = requests.get(f"{API_BASE}/predictions/current", headers=HEADERS)
    return response.json()

def get_alerts():
    response = requests.get(f"{API_BASE}/alerts/active", headers=HEADERS)
    return response.json()

def get_server_explanation(server_name):
    response = requests.get(f"{API_BASE}/explain/{server_name}", headers=HEADERS)
    return response.json()
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Get predictions
curl -H "X-API-Key: your-key" http://localhost:8000/predictions/current

# Get alerts
curl -H "X-API-Key: your-key" http://localhost:8000/alerts/active

# Get explanation
curl -H "X-API-Key: your-key" http://localhost:8000/explain/ppdb001
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/feed/data` | 60/minute |
| `/predictions/current` | 30/minute |
| `/alerts/active` | 30/minute |
| `/explain/{server}` | 30/minute |

---

## Recommended Polling Intervals

| Dashboard Component | Interval |
|---------------------|----------|
| Fleet overview | 30 seconds |
| Server detail view | 15 seconds |
| Alerts panel | 10 seconds |
| Historical charts | 5 minutes |

---

## Alert Levels Reference

| Level | Risk Score | Color | Use Case |
|-------|------------|-------|----------|
| Critical | 80-100 | Red (#FF0000) | Immediate attention required |
| Warning | 60-79 | Orange (#FFA500) | Investigate soon |
| Degraded | 50-59 | Yellow (#FFD700) | Monitor closely |
| Healthy | 0-49 | Green (#00FF00) | Normal operation |

---

## Starting the Inference Engine

```bash
cd Argus
python src/daemons/tft_inference_daemon.py --port 8000
```

With auto-retraining enabled:
```bash
python src/daemons/tft_inference_daemon.py --enable-retraining
```

---

## Troubleshooting

**"insufficient_data" error:**
- The daemon needs at least 100 data points before predictions work
- Wait for the metrics generator to feed enough data

**401 Unauthorized:**
- Check your API key is set correctly
- Verify X-API-Key header is included

**Connection refused:**
- Ensure the daemon is running on the expected port
- Check firewall rules

**Stale predictions:**
- Verify the metrics generator is running and feeding data
- Check `/status` endpoint for `last_prediction` timestamp
