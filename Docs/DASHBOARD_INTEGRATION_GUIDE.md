# Tachyon Argus Inference Engine - Dashboard Integration Guide

## Overview

The TFT Inference Daemon exposes a REST API for building monitoring dashboards. The daemon handles all prediction logic, risk scoring, and alert generation - your dashboard just needs to fetch and display.

**Base URL:** `http://localhost:8000` (configurable via `--port`)

### New in v2.1

- **Cascading Failure Detection**: Fleet-wide health monitoring and cross-server correlation analysis
- **Model Drift Monitoring**: Automatic drift detection with retraining triggers
- **Multi-Target Predictions**: CPU, memory, swap, I/O wait, and load predictions
- **Fleet Health Scoring**: Real-time fleet-wide health metrics

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

## Cascading Failure Detection Endpoints

These endpoints enable dashboards to display fleet-wide health and detect environment-wide issues.

### 6. Cascade Status

```
GET /cascade/status
```

Returns full cascade detection status including recent events:

```json
{
  "current_status": {
    "cascade_detected": false,
    "timestamp": "2025-01-15T10:05:00",
    "total_servers": 45,
    "servers_with_anomalies": 3,
    "anomaly_rate": 0.067,
    "correlation_score": 0.234,
    "cascades": []
  },
  "tracking": {
    "servers": 45,
    "metrics_tracked": ["cpu_user_pct", "mem_used_pct", "cpu_iowait_pct", "load_average", "swap_used_pct"],
    "window_size": 100
  },
  "recent_events": [
    {
      "timestamp": "2025-01-15T09:30:00",
      "cascades": [{"metric": "cpu_user_pct", "affected_servers": ["srv001", "srv002", "srv003"], "severity": "medium"}],
      "correlation_score": 0.782,
      "affected_servers": ["srv001", "srv002", "srv003", "srv004"]
    }
  ],
  "event_count": 5,
  "thresholds": {
    "correlation": 0.7,
    "cascade_servers": 3,
    "anomaly_z_score": 2.0
  }
}
```

**Dashboard Use Cases:**
- Display cascade event history timeline
- Show correlated servers in a network graph
- Alert when correlation score exceeds threshold

---

### 7. Fleet Health Score

```
GET /cascade/health
```

Returns a simple fleet-wide health score (ideal for dashboard header):

```json
{
  "health_score": 85.2,
  "status": "healthy",
  "correlation_score": 0.234,
  "anomaly_rate": 0.067,
  "anomalous_servers": 3,
  "total_servers": 45,
  "cascade_risk": "low"
}
```

**Health Status Levels:**

| Status | Health Score | Description |
|--------|--------------|-------------|
| `healthy` | 80-100 | Fleet operating normally |
| `degraded` | 60-79 | Some servers showing issues |
| `warning` | 40-59 | Multiple correlated issues |
| `critical` | 0-39 | Cascading failure in progress |

**Cascade Risk Levels:**

| Risk | Correlation Score | Meaning |
|------|-------------------|---------|
| `low` | 0 - 0.5 | Servers operating independently |
| `medium` | 0.5 - 0.7 | Some correlation detected |
| `high` | > 0.7 | Significant cross-server correlation |

---

## Model Drift Monitoring Endpoints

Monitor model performance and automatic retraining status.

### 8. Drift Status

```
GET /drift/status
```

Returns current drift detection status:

```json
{
  "drift_detected": false,
  "auto_retrain_enabled": true,
  "last_retrain": "2025-01-15T02:00:00",
  "next_check": "2025-01-15T11:00:00",
  "metrics": {
    "per": 0.05,
    "dss": 0.12,
    "fds": 0.08,
    "anomaly_rate": 0.03
  },
  "thresholds": {
    "per": 0.10,
    "dss": 0.20,
    "fds": 0.15,
    "anomaly_rate": 0.05
  }
}
```

**Drift Metrics Explained:**

| Metric | Name | Threshold | Description |
|--------|------|-----------|-------------|
| `per` | Prediction Error Rate | 10% | Rolling average prediction error |
| `dss` | Distribution Shift Score | 20% | Input feature distribution change |
| `fds` | Feature Drift Score | 15% | Individual feature drift detection |
| `anomaly_rate` | Anomaly Rate | 5% | Rate of anomalous predictions |

---

### 9. Drift Report

```
GET /drift/report
```

Returns detailed drift analysis report:

```json
{
  "report_timestamp": "2025-01-15T10:05:00",
  "overall_health": "good",
  "needs_retraining": false,
  "metrics": {
    "per": {"value": 0.05, "threshold": 0.10, "status": "ok"},
    "dss": {"value": 0.12, "threshold": 0.20, "status": "ok"},
    "fds": {"value": 0.08, "threshold": 0.15, "status": "ok"},
    "anomaly_rate": {"value": 0.03, "threshold": 0.05, "status": "ok"}
  },
  "feature_drift": {
    "cpu_user_pct": {"drift": 0.02, "status": "stable"},
    "mem_used_pct": {"drift": 0.08, "status": "stable"},
    "load_average": {"drift": 0.15, "status": "drifting"}
  },
  "recommendations": [],
  "auto_retrain": {
    "enabled": true,
    "last_triggered": null,
    "total_drift_trainings": 0
  }
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

# Fleet health (cascade detection)
curl -H "X-API-Key: your-key" http://localhost:8000/cascade/health

# Cascade status
curl -H "X-API-Key: your-key" http://localhost:8000/cascade/status

# Drift status
curl -H "X-API-Key: your-key" http://localhost:8000/drift/status

# Full drift report
curl -H "X-API-Key: your-key" http://localhost:8000/drift/report
```

---

## Dashboard Component Examples

### Fleet Health Header Widget

Display a prominent fleet health indicator at the top of your dashboard:

```javascript
// React component for fleet health header
function FleetHealthHeader() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    const fetchHealth = async () => {
      const res = await fetch(`${API_BASE}/cascade/health`, { headers });
      setHealth(await res.json());
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  if (!health) return <div>Loading...</div>;

  const statusColors = {
    healthy: '#00FF00',
    degraded: '#FFD700',
    warning: '#FFA500',
    critical: '#FF0000'
  };

  return (
    <div className="fleet-health-header">
      <div className="health-score" style={{color: statusColors[health.status]}}>
        {health.health_score.toFixed(1)}
      </div>
      <div className="health-status">{health.status.toUpperCase()}</div>
      <div className="cascade-risk">
        Cascade Risk: <span className={`risk-${health.cascade_risk}`}>
          {health.cascade_risk.toUpperCase()}
        </span>
      </div>
      <div className="stats">
        {health.anomalous_servers} / {health.total_servers} servers with anomalies
      </div>
    </div>
  );
}
```

### Cascade Event Timeline

Display recent cascade events:

```javascript
// React component for cascade timeline
function CascadeTimeline() {
  const [cascadeStatus, setCascadeStatus] = useState(null);

  useEffect(() => {
    const fetchStatus = async () => {
      const res = await fetch(`${API_BASE}/cascade/status`, { headers });
      setCascadeStatus(await res.json());
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 60000);
    return () => clearInterval(interval);
  }, []);

  if (!cascadeStatus) return <div>Loading...</div>;

  return (
    <div className="cascade-timeline">
      <h3>Recent Cascade Events ({cascadeStatus.event_count})</h3>
      {cascadeStatus.recent_events.map((event, idx) => (
        <div key={idx} className="cascade-event">
          <div className="timestamp">{event.timestamp}</div>
          <div className="correlation">
            Correlation: {(event.correlation_score * 100).toFixed(1)}%
          </div>
          <div className="affected">
            Affected: {event.affected_servers.join(', ')}
          </div>
          {event.cascades.map((cascade, cidx) => (
            <div key={cidx} className={`cascade-detail severity-${cascade.severity}`}>
              {cascade.metric}: {cascade.affected_servers.length} servers
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
```

### Model Drift Indicator

Show model health status:

```javascript
// React component for drift monitoring
function DriftIndicator() {
  const [drift, setDrift] = useState(null);

  useEffect(() => {
    const fetchDrift = async () => {
      const res = await fetch(`${API_BASE}/drift/status`, { headers });
      setDrift(await res.json());
    };
    fetchDrift();
    const interval = setInterval(fetchDrift, 300000); // Every 5 minutes
    return () => clearInterval(interval);
  }, []);

  if (!drift) return <div>Loading...</div>;

  return (
    <div className={`drift-indicator ${drift.drift_detected ? 'warning' : 'ok'}`}>
      <div className="drift-status">
        Model: {drift.drift_detected ? '⚠️ DRIFT DETECTED' : '✅ Healthy'}
      </div>
      <div className="drift-metrics">
        <span title="Prediction Error Rate">
          PER: {(drift.metrics.per * 100).toFixed(1)}%
        </span>
        <span title="Distribution Shift Score">
          DSS: {(drift.metrics.dss * 100).toFixed(1)}%
        </span>
        <span title="Feature Drift Score">
          FDS: {(drift.metrics.fds * 100).toFixed(1)}%
        </span>
      </div>
      {drift.auto_retrain_enabled && (
        <div className="auto-retrain">Auto-retrain: Enabled</div>
      )}
    </div>
  );
}
```

### Python Dashboard Example (Streamlit)

```python
import streamlit as st
import requests
import time

API_BASE = "http://localhost:8000"
HEADERS = {"X-API-Key": st.secrets["api_key"]}

# Fleet Health Header
st.header("Fleet Health")
health = requests.get(f"{API_BASE}/cascade/health", headers=HEADERS).json()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Health Score", f"{health['health_score']:.1f}",
            delta=None if health['status'] == 'healthy' else "⚠️")
col2.metric("Status", health['status'].upper())
col3.metric("Cascade Risk", health['cascade_risk'].upper())
col4.metric("Anomalous Servers", f"{health['anomalous_servers']}/{health['total_servers']}")

# Drift Status
st.subheader("Model Health")
drift = requests.get(f"{API_BASE}/drift/status", headers=HEADERS).json()

if drift['drift_detected']:
    st.error("⚠️ Model drift detected - retraining may be triggered")
else:
    st.success("✅ Model performing within acceptable range")

# Drift metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("PER", f"{drift['metrics']['per']*100:.1f}%",
            delta=f"/{drift['thresholds']['per']*100:.0f}%")
col2.metric("DSS", f"{drift['metrics']['dss']*100:.1f}%",
            delta=f"/{drift['thresholds']['dss']*100:.0f}%")
col3.metric("FDS", f"{drift['metrics']['fds']*100:.1f}%",
            delta=f"/{drift['thresholds']['fds']*100:.0f}%")
col4.metric("Anomaly Rate", f"{drift['metrics']['anomaly_rate']*100:.1f}%",
            delta=f"/{drift['thresholds']['anomaly_rate']*100:.0f}%")

# Cascade Events
st.subheader("Cascade Events")
cascade = requests.get(f"{API_BASE}/cascade/status", headers=HEADERS).json()

if cascade['current_status']['cascade_detected']:
    st.error(f"🔴 ACTIVE CASCADE: {cascade['current_status']['servers_with_anomalies']} servers affected")

for event in cascade['recent_events']:
    with st.expander(f"Event: {event['timestamp']}"):
        st.write(f"Correlation: {event['correlation_score']:.2%}")
        st.write(f"Affected servers: {', '.join(event['affected_servers'])}")
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/feed/data` | 60/minute |
| `/predictions/current` | 30/minute |
| `/alerts/active` | 30/minute |
| `/explain/{server}` | 30/minute |
| `/cascade/status` | 30/minute |
| `/cascade/health` | 60/minute |
| `/drift/status` | 30/minute |
| `/drift/report` | 10/minute |

---

## Recommended Polling Intervals

| Dashboard Component | Interval | Endpoint |
|---------------------|----------|----------|
| Fleet health header | 30 seconds | `/cascade/health` |
| Fleet overview | 30 seconds | `/predictions/current` |
| Server detail view | 15 seconds | `/predictions/current` |
| Alerts panel | 10 seconds | `/alerts/active` |
| Cascade timeline | 60 seconds | `/cascade/status` |
| Drift indicator | 5 minutes | `/drift/status` |
| Historical charts | 5 minutes | `/historical/*` |

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
