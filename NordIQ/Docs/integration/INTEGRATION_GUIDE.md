# NordIQ Dashboard Integration Guide

**Version:** 1.0.0
**Last Updated:** October 29, 2025
**Audience:** Integration developers, DevOps engineers, dashboard creators
**Purpose:** Connect to the NordIQ TFT Inference Daemon and integrate predictions into custom dashboards

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Authentication](#authentication)
5. [Integration Examples](#integration-examples)
6. [Data Format Reference](#data-format-reference)
7. [Grafana Integration](#grafana-integration)
8. [Custom Dashboard Integration](#custom-dashboard-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The NordIQ Dashboard is powered by the **TFT Inference Daemon**, a FastAPI-based REST API service that:

- Runs AI predictions every 5 seconds
- Provides real-time server health forecasts (up to 8 hours ahead)
- Calculates risk scores using contextual intelligence
- Supports multiple concurrent dashboard connections
- Includes explainable AI (XAI) endpoints for prediction transparency

**Use Cases:**
- Build custom dashboards with your preferred UI framework
- Integrate predictions into existing monitoring tools (Grafana, Datadog, etc.)
- Create specialized views for specific teams (SRE, DevOps, Management)
- Export data to business intelligence tools
- Feed predictions into incident management systems

---

## Quick Start

### 1. Start the Inference Daemon

```bash
# Navigate to NordIQ directory
cd NordIQ

# Start all services (includes inference daemon on port 8000)
start_all.bat  # Windows
./start_all.sh # Linux/Mac

# Or start inference daemon only
cd src/daemons
python tft_inference_daemon.py
```

### 2. Test the Connection

```bash
# Health check (no authentication required)
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

### 3. Get Predictions (with API key)

```bash
# Set your API key
export TFT_API_KEY="your-api-key-here"

# Get predictions
curl -H "X-API-Key: $TFT_API_KEY" http://localhost:8000/predictions/current
```

---

## API Reference

### Base URL
```
http://localhost:8000  # Default (development)
https://your-domain.com/api  # Production
```

### Endpoints

#### 1. Health Check
**No authentication required**

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

**Use Case:** Load balancer health checks, uptime monitoring

---

#### 2. Service Status
**No authentication required**

```http
GET /status
```

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "model_path": "/path/to/model",
  "uptime_seconds": 3600,
  "predictions_count": 720,
  "window_size": 200,
  "servers_tracked": 20
}
```

**Use Case:** Monitoring daemon health, debugging

---

#### 3. Current Predictions
**Authentication required**

```http
GET /predictions/current
Headers:
  X-API-Key: your-api-key-here
```

**Rate Limit:** 30 requests/minute (1 every 2 seconds)

**Response:**
```json
{
  "timestamp": "2025-10-29T14:30:00",
  "prediction_count": 20,
  "summary": {
    "critical": 2,
    "warning": 5,
    "degrading": 3,
    "healthy": 10,
    "avg_risk_score": 42.3,
    "max_risk_score": 87.5,
    "environment_status": "Warning"
  },
  "predictions": {
    "ppml0001": {
      "server_name": "ppml0001",
      "profile": "ML Compute",
      "risk_score": 87.5,
      "alert_level": "Critical",
      "alert_priority": "P1",
      "alert_color": "#DC2626",
      "alert_emoji": "🔴",
      "cpu_idle_pct": {
        "current": 15.2,
        "predicted_30min": 8.1,
        "predicted_1hr": 5.3,
        "predicted_8hr": 12.0,
        "trend": "degrading"
      },
      "mem_used_pct": {
        "current": 92.3,
        "predicted_30min": 96.1,
        "predicted_1hr": 98.2,
        "predicted_8hr": 95.0,
        "trend": "degrading"
      },
      "cpu_iowait_pct": {
        "current": 12.5,
        "predicted_30min": 18.3,
        "predicted_1hr": 22.1,
        "predicted_8hr": 15.2,
        "trend": "degrading"
      },
      "... (11 more metrics)"
    },
    "... (19 more servers)"
  },
  "alerts": [
    {
      "server_name": "ppml0001",
      "profile": "ML Compute",
      "risk_score": 87.5,
      "priority": "P1",
      "level": "Critical",
      "message": "High risk: CPU 85%, Memory 92%, I/O Wait 12%"
    }
  ]
}
```

**Use Case:** Real-time dashboard updates, alerting systems

---

#### 4. Active Alerts
**Authentication required**

```http
GET /alerts/active
Headers:
  X-API-Key: your-api-key-here
```

**Rate Limit:** 30 requests/minute

**Response:**
```json
{
  "timestamp": "2025-10-29T14:30:00",
  "count": 7,
  "alerts": [
    {
      "server_name": "ppml0001",
      "profile": "ML Compute",
      "risk_score": 87.5,
      "priority": "P1",
      "level": "Critical",
      "message": "High risk: CPU 85%, Memory 92%, I/O Wait 12%",
      "recommended_action": "Immediate investigation required"
    },
    {
      "server_name": "ppdb002",
      "profile": "Database",
      "risk_score": 72.3,
      "priority": "P2",
      "level": "Warning",
      "message": "Elevated I/O wait: 18%",
      "recommended_action": "Check disk subsystem"
    }
  ]
}
```

**Use Case:** Alert routing, incident management, on-call notifications

---

#### 5. Explain Prediction (XAI)
**Authentication required**

```http
GET /explain/{server_name}
Headers:
  X-API-Key: your-api-key-here
```

**Rate Limit:** 30 requests/minute

**Response:**
```json
{
  "server_name": "ppml0001",
  "timestamp": "2025-10-29T14:30:00",
  "prediction": {
    "risk_score": 87.5,
    "alert_level": "Critical"
  },
  "shap": {
    "feature_importance": [
      {
        "metric": "mem_used_pct",
        "importance": 0.45,
        "stars": "★★★★★",
        "description": "Memory utilization (92%) is the primary driver"
      },
      {
        "metric": "cpu_iowait_pct",
        "importance": 0.32,
        "stars": "★★★★",
        "description": "I/O wait (12%) is significantly elevated"
      },
      {
        "metric": "cpu_idle_pct",
        "importance": 0.23,
        "stars": "★★★",
        "description": "Low CPU idle (15%) indicates sustained load"
      }
    ]
  },
  "attention": {
    "temporal_focus": "last_30_minutes",
    "description": "Model is focusing on recent degradation pattern"
  },
  "counterfactuals": [
    {
      "scenario": "reduce_memory_10pct",
      "predicted_risk": 65.2,
      "impact": "Risk reduced by 22.3 points",
      "actionable": true,
      "recommendation": "Free up ~9 GB memory to reduce risk to Warning level"
    }
  ]
}
```

**Use Case:** Root cause analysis, capacity planning, what-if scenarios

---

#### 6. Feed Data (Demo Mode)
**Authentication required**

```http
POST /feed/data
Headers:
  X-API-Key: your-api-key-here
Content-Type: application/json

Body:
{
  "records": [
    {
      "timestamp": "2025-10-29T14:30:00",
      "server_name": "ppml0001",
      "profile": "ml_compute",
      "state": "heavy_load",
      "cpu_user_pct": 65.2,
      "cpu_sys_pct": 10.1,
      "cpu_iowait_pct": 12.5,
      "cpu_idle_pct": 12.2,
      "java_cpu_pct": 58.3,
      "mem_used_pct": 92.3,
      "swap_used_pct": 15.0,
      "disk_usage_pct": 68.5,
      "net_in_mb_s": 25.3,
      "net_out_mb_s": 18.7,
      "back_close_wait": 45,
      "front_close_wait": 12,
      "load_average": 8.5,
      "uptime_days": 15
    }
  ]
}
```

**Rate Limit:** 60 requests/minute

**Response:**
```json
{
  "status": "success",
  "records_received": 1,
  "window_size": 201
}
```

**Use Case:** Testing, demos, integration with production metrics collectors

---

## Authentication

### API Key Setup

The daemon uses **API Key authentication** via the `X-API-Key` header.

#### 1. Generate API Key

```bash
cd NordIQ/bin
python generate_api_key.py
```

This creates/updates `.env` file with:
```bash
TFT_API_KEY=your-generated-key-here
```

#### 2. Configure API Key

**Option A: Environment Variable (Recommended)**
```bash
# Linux/Mac
export TFT_API_KEY="your-api-key-here"

# Windows
set TFT_API_KEY=your-api-key-here
```

**Option B: .env File**
```bash
# Create/edit .env file in NordIQ directory
echo "TFT_API_KEY=your-api-key-here" > .env
```

**Option C: Streamlit Secrets (Dashboard)**
```toml
# NordIQ/.streamlit/secrets.toml
[api]
key = "your-api-key-here"
```

#### 3. Use API Key in Requests

**cURL:**
```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8000/predictions/current
```

**Python:**
```python
import requests

headers = {"X-API-Key": "your-api-key-here"}
response = requests.get("http://localhost:8000/predictions/current", headers=headers)
```

**JavaScript:**
```javascript
fetch('http://localhost:8000/predictions/current', {
  headers: {
    'X-API-Key': 'your-api-key-here'
  }
})
```

### Development Mode (No Authentication)

If `TFT_API_KEY` is not set, the daemon runs in **development mode**:
- `/health` and `/status` remain public
- Other endpoints work without authentication
- **NOT recommended for production**

---

## Integration Examples

### Python Client

```python
import requests
from typing import Optional, Dict

class NordIQClient:
    """Python client for NordIQ Inference Daemon."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        """Get headers with API key if configured."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def check_health(self) -> bool:
        """Check if daemon is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.ok
        except:
            return False

    def get_predictions(self) -> Optional[Dict]:
        """Get current predictions."""
        try:
            response = requests.get(
                f"{self.base_url}/predictions/current",
                headers=self._headers(),
                timeout=5
            )
            if response.ok:
                return response.json()
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_alerts(self) -> Optional[Dict]:
        """Get active alerts."""
        try:
            response = requests.get(
                f"{self.base_url}/alerts/active",
                headers=self._headers(),
                timeout=5
            )
            if response.ok:
                return response.json()
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_critical_servers(self) -> list:
        """Get list of servers in critical state."""
        predictions = self.get_predictions()
        if not predictions:
            return []

        critical = []
        for server_name, pred in predictions.get('predictions', {}).items():
            if pred.get('risk_score', 0) >= 80:
                critical.append({
                    'server': server_name,
                    'profile': pred.get('profile'),
                    'risk_score': pred.get('risk_score'),
                    'alert_level': pred.get('alert_level')
                })

        return sorted(critical, key=lambda x: x['risk_score'], reverse=True)

# Usage
client = NordIQClient(api_key="your-api-key-here")

# Check health
if client.check_health():
    print("✅ Daemon is healthy")

# Get predictions
predictions = client.get_predictions()
if predictions:
    print(f"📊 Monitoring {predictions['prediction_count']} servers")
    print(f"⚠️  {predictions['summary']['critical']} critical alerts")

# Get critical servers
critical = client.get_critical_servers()
for server in critical:
    print(f"🔴 {server['server']}: Risk {server['risk_score']}")
```

---

### JavaScript/Node.js Client

```javascript
class NordIQClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
  }

  _headers() {
    const headers = { 'Content-Type': 'application/json' };
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }
    return headers;
  }

  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  async getPredictions() {
    try {
      const response = await fetch(`${this.baseUrl}/predictions/current`, {
        headers: this._headers()
      });
      if (response.ok) {
        return await response.json();
      }
      return null;
    } catch (error) {
      console.error('Error fetching predictions:', error);
      return null;
    }
  }

  async getAlerts() {
    try {
      const response = await fetch(`${this.baseUrl}/alerts/active`, {
        headers: this._headers()
      });
      if (response.ok) {
        return await response.json();
      }
      return null;
    } catch (error) {
      console.error('Error fetching alerts:', error);
      return null;
    }
  }

  async getCriticalServers() {
    const predictions = await this.getPredictions();
    if (!predictions) return [];

    const critical = [];
    for (const [serverName, pred] of Object.entries(predictions.predictions || {})) {
      if ((pred.risk_score || 0) >= 80) {
        critical.push({
          server: serverName,
          profile: pred.profile,
          risk_score: pred.risk_score,
          alert_level: pred.alert_level
        });
      }
    }

    return critical.sort((a, b) => b.risk_score - a.risk_score);
  }
}

// Usage
const client = new NordIQClient('http://localhost:8000', 'your-api-key-here');

// Check health
const isHealthy = await client.checkHealth();
console.log(isHealthy ? '✅ Daemon is healthy' : '❌ Daemon is down');

// Get predictions
const predictions = await client.getPredictions();
if (predictions) {
  console.log(`📊 Monitoring ${predictions.prediction_count} servers`);
  console.log(`⚠️  ${predictions.summary.critical} critical alerts`);
}

// Get critical servers
const critical = await client.getCriticalServers();
critical.forEach(server => {
  console.log(`🔴 ${server.server}: Risk ${server.risk_score}`);
});
```

---

### React Dashboard Component

```jsx
import React, { useState, useEffect } from 'react';

function NordIQDashboard() {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await fetch('http://localhost:8000/predictions/current', {
          headers: {
            'X-API-Key': process.env.REACT_APP_NORDIQ_API_KEY
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        setPredictions(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    // Fetch immediately
    fetchPredictions();

    // Refresh every 10 seconds
    const interval = setInterval(fetchPredictions, 10000);

    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!predictions) return <div>No data</div>;

  const { summary, predictions: servers } = predictions;

  return (
    <div className="nordiq-dashboard">
      <h1>NordIQ Fleet Monitor</h1>

      {/* Summary Cards */}
      <div className="summary-grid">
        <div className="card critical">
          <h3>Critical</h3>
          <p className="count">{summary.critical}</p>
        </div>
        <div className="card warning">
          <h3>Warning</h3>
          <p className="count">{summary.warning}</p>
        </div>
        <div className="card healthy">
          <h3>Healthy</h3>
          <p className="count">{summary.healthy}</p>
        </div>
        <div className="card avg-risk">
          <h3>Avg Risk</h3>
          <p className="count">{summary.avg_risk_score.toFixed(1)}</p>
        </div>
      </div>

      {/* Server List */}
      <div className="server-list">
        <h2>Servers</h2>
        {Object.entries(servers).map(([name, server]) => (
          <div
            key={name}
            className={`server-card ${server.alert_level.toLowerCase()}`}
          >
            <div className="server-header">
              <h3>{name}</h3>
              <span className="profile">{server.profile}</span>
              <span className="risk-score">{server.risk_score}</span>
            </div>
            <div className="metrics">
              <div className="metric">
                <span>CPU:</span>
                <span>{(100 - server.cpu_idle_pct.current).toFixed(1)}%</span>
              </div>
              <div className="metric">
                <span>Memory:</span>
                <span>{server.mem_used_pct.current.toFixed(1)}%</span>
              </div>
              <div className="metric">
                <span>I/O Wait:</span>
                <span>{server.cpu_iowait_pct.current.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default NordIQDashboard;
```

---

## Data Format Reference

### 14 NordIQ Metrics

All predictions include these 14 metrics:

| Metric | Type | Range | Description |
|--------|------|-------|-------------|
| `cpu_user_pct` | float | 0-100 | User space CPU usage |
| `cpu_sys_pct` | float | 0-100 | System/kernel CPU usage |
| `cpu_iowait_pct` | float | 0-100 | I/O wait percentage (critical metric) |
| `cpu_idle_pct` | float | 0-100 | Idle CPU (display as: 100 - idle) |
| `java_cpu_pct` | float | 0-100 | Java/Spark CPU usage |
| `mem_used_pct` | float | 0-100 | Memory utilization |
| `swap_used_pct` | float | 0-100 | Swap usage |
| `disk_usage_pct` | float | 0-100 | Disk space usage |
| `net_in_mb_s` | float | 0+ | Network ingress (MB/s) |
| `net_out_mb_s` | float | 0+ | Network egress (MB/s) |
| `back_close_wait` | int | 0+ | TCP backend connections |
| `front_close_wait` | int | 0+ | TCP frontend connections |
| `load_average` | float | 0+ | System load average |
| `uptime_days` | int | 0-30 | Days since last reboot |

### Metric Object Format

Each metric in the prediction includes:

```json
{
  "current": 85.3,           // Current value (latest measurement)
  "predicted_30min": 92.1,   // Predicted value 30 minutes ahead
  "predicted_1hr": 95.8,     // Predicted value 1 hour ahead
  "predicted_8hr": 88.2,     // Predicted value 8 hours ahead
  "trend": "degrading"       // Trend: "improving", "stable", "degrading"
}
```

### Risk Score Calculation

Risk scores (0-100) are calculated using:
- **70% current state** - What's happening now
- **30% predictions** - Early warning signals
- **Profile awareness** - Different thresholds per server type
- **Multi-metric correlation** - Combined impact assessment

**Risk Score Ranges:**
- **90-100**: Imminent Failure (P1 - 5 min SLA)
- **80-89**: Critical (P1 - 15 min SLA)
- **70-79**: Danger (P2 - 30 min SLA)
- **60-69**: Warning (P2 - 1 hour SLA)
- **50-59**: Degrading (P3 - 2 hour SLA)
- **30-49**: Watch (monitoring only)
- **0-29**: Healthy (no action needed)

### Server Profiles

7 server profiles with different operational characteristics:

| Profile | Prefix | Normal CPU | Normal Memory | Normal I/O Wait |
|---------|--------|-----------|---------------|----------------|
| ML Compute | ppml#### | 40-80% | 60-90% | <5% |
| Database | ppdb### | 20-60% | 70-98% | 5-20% |
| Web API | ppweb### | 10-40% | 30-60% | <2% |
| Conductor Mgmt | ppcon## | 5-30% | 20-50% | <2% |
| Data Ingest | ppdi### | 30-70% | 50-80% | 5-15% |
| Risk Analytics | ppra### | 40-80% | 60-90% | <5% |
| Generic | ppsrv### | 20-50% | 40-70% | <5% |

---

## Grafana Integration

Grafana can visualize NordIQ predictions using the **JSON API** data source plugin.

### Setup Steps

#### 1. Install JSON API Plugin

```bash
grafana-cli plugins install marcusolsson-json-datasource
# Restart Grafana
```

#### 2. Configure Data Source

In Grafana UI:
1. Go to **Configuration → Data Sources → Add data source**
2. Select **JSON API**
3. Configure:
   - **Name:** NordIQ Predictions
   - **URL:** `http://localhost:8000`
   - **Custom HTTP Headers:**
     - Header: `X-API-Key`
     - Value: `your-api-key-here`
   - **Timeout:** 5s
4. Click **Save & Test**

#### 3. Create Dashboard

**Panel 1: Server Risk Scores (Time Series)**

Query configuration:
- **Endpoint:** `/predictions/current`
- **Fields:**
  - `$.predictions.*.server_name` → Label
  - `$.predictions.*.risk_score` → Value
- **Refresh:** 10s

JSONPath query:
```json
{
  "target": "risk_scores",
  "refId": "A",
  "type": "timeserie"
}
```

**Panel 2: Alert Summary (Stat)**

Query configuration:
- **Endpoint:** `/predictions/current`
- **Fields:**
  - `$.summary.critical` → Critical Count
  - `$.summary.warning` → Warning Count
  - `$.summary.healthy` → Healthy Count
- **Visualization:** Stat
- **Thresholds:**
  - Critical: > 5
  - Warning: > 2
  - Good: 0-2

**Panel 3: Critical Servers (Table)**

Query configuration:
- **Endpoint:** `/alerts/active`
- **Fields:**
  - `$.alerts[*].server_name` → Server
  - `$.alerts[*].profile` → Profile
  - `$.alerts[*].risk_score` → Risk
  - `$.alerts[*].level` → Level
  - `$.alerts[*].message` → Message
- **Visualization:** Table
- **Sort:** Risk Score (descending)

**Panel 4: Environment Status (Gauge)**

Query configuration:
- **Endpoint:** `/predictions/current`
- **Fields:**
  - `$.summary.avg_risk_score` → Value
- **Visualization:** Gauge
- **Thresholds:**
  - 0-30: Green (Healthy)
  - 30-60: Yellow (Caution)
  - 60-80: Orange (Warning)
  - 80-100: Red (Critical)

### Grafana Variables

Create variables for dynamic filtering:

**Variable: server_profile**
- Type: Custom
- Options: `ML Compute`, `Database`, `Web API`, `Conductor Mgmt`, `Data Ingest`, `Risk Analytics`, `Generic`
- Multi-value: Yes

**Variable: alert_level**
- Type: Custom
- Options: `Critical`, `Warning`, `Degrading`, `Healthy`
- Multi-value: Yes

Use in queries:
```json
{
  "target": "filtered_predictions",
  "profile": "${server_profile}",
  "level": "${alert_level}"
}
```

### Alerting in Grafana

Create alert rules based on NordIQ predictions:

**Alert 1: Critical Server Detected**
- **Condition:** `$.summary.critical > 0`
- **For:** 1 minute
- **Action:** Send notification to Slack/PagerDuty

**Alert 2: Fleet Average Risk High**
- **Condition:** `$.summary.avg_risk_score > 70`
- **For:** 5 minutes
- **Action:** Email SRE team

**Alert 3: Prediction Service Down**
- **Condition:** No data for 30 seconds
- **Action:** Page on-call engineer

---

## Custom Dashboard Integration

### Building Your Own Dashboard

#### Requirements
- HTTP client library (requests, axios, fetch, etc.)
- JSON parsing
- UI framework of your choice
- WebSocket support (optional, for real-time updates)

#### Architecture Pattern

```
┌─────────────────┐
│   Your UI/      │
│   Dashboard     │
└────────┬────────┘
         │ HTTP REST
         ↓
┌─────────────────┐
│  NordIQ TFT     │
│  Inference      │
│  Daemon         │
│  (Port 8000)    │
└────────┬────────┘
         │ WebSocket (optional)
         ↓
┌─────────────────┐
│  Metrics        │
│  Generator      │
│  (Port 8001)    │
└─────────────────┘
```

#### Minimal Dashboard (HTML + Vanilla JS)

```html
<!DOCTYPE html>
<html>
<head>
  <title>NordIQ Fleet Monitor</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    .summary { display: flex; gap: 20px; margin-bottom: 30px; }
    .card {
      padding: 20px;
      border-radius: 8px;
      background: #f5f5f5;
      text-align: center;
      min-width: 150px;
    }
    .card.critical { background: #fee2e2; color: #dc2626; }
    .card.warning { background: #fef3c7; color: #d97706; }
    .card.healthy { background: #d1fae5; color: #059669; }
    .servers { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
    .server { padding: 15px; border-radius: 8px; background: white; border-left: 4px solid #ccc; }
    .server.critical { border-left-color: #dc2626; }
    .server.warning { border-left-color: #d97706; }
    .server.healthy { border-left-color: #059669; }
    .metric { display: flex; justify-content: space-between; padding: 5px 0; }
  </style>
</head>
<body>
  <h1>NordIQ Fleet Monitor</h1>
  <div id="summary" class="summary"></div>
  <div id="servers" class="servers"></div>

  <script>
    const API_URL = 'http://localhost:8000';
    const API_KEY = 'your-api-key-here';

    async function fetchPredictions() {
      try {
        const response = await fetch(`${API_URL}/predictions/current`, {
          headers: { 'X-API-Key': API_KEY }
        });

        if (!response.ok) {
          console.error('Failed to fetch predictions:', response.status);
          return;
        }

        const data = await response.json();
        updateDashboard(data);
      } catch (error) {
        console.error('Error:', error);
      }
    }

    function updateDashboard(data) {
      // Update summary
      const summary = document.getElementById('summary');
      summary.innerHTML = `
        <div class="card critical">
          <h3>Critical</h3>
          <h2>${data.summary.critical}</h2>
        </div>
        <div class="card warning">
          <h3>Warning</h3>
          <h2>${data.summary.warning}</h2>
        </div>
        <div class="card healthy">
          <h3>Healthy</h3>
          <h2>${data.summary.healthy}</h2>
        </div>
        <div class="card">
          <h3>Avg Risk</h3>
          <h2>${data.summary.avg_risk_score.toFixed(1)}</h2>
        </div>
      `;

      // Update servers
      const servers = document.getElementById('servers');
      const serverHTML = Object.entries(data.predictions).map(([name, server]) => {
        const cpuUsed = (100 - server.cpu_idle_pct.current).toFixed(1);
        const alertClass = server.alert_level.toLowerCase();

        return `
          <div class="server ${alertClass}">
            <h3>${name} <small>${server.profile}</small></h3>
            <div style="font-size: 24px; font-weight: bold; color: ${server.alert_color}">
              ${server.risk_score}
            </div>
            <div class="metrics">
              <div class="metric">
                <span>CPU:</span>
                <span>${cpuUsed}%</span>
              </div>
              <div class="metric">
                <span>Memory:</span>
                <span>${server.mem_used_pct.current.toFixed(1)}%</span>
              </div>
              <div class="metric">
                <span>I/O Wait:</span>
                <span>${server.cpu_iowait_pct.current.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        `;
      }).join('');

      servers.innerHTML = serverHTML;
    }

    // Fetch immediately and then every 10 seconds
    fetchPredictions();
    setInterval(fetchPredictions, 10000);
  </script>
</body>
</html>
```

Save as `monitor.html` and open in browser!

---

## Best Practices

### 1. Polling vs WebSocket

**Polling (Current)**
- ✅ Simple to implement
- ✅ Works with all HTTP clients
- ✅ Easy to debug
- ⚠️ Rate limit: 30 requests/minute (1 every 2 seconds)

**Recommended Polling Interval:**
- Real-time dashboards: 10-15 seconds
- Management views: 30-60 seconds
- Historical analysis: 5+ minutes

**WebSocket (Future Enhancement)**
- ✅ Real-time push updates
- ✅ Lower latency
- ⚠️ More complex implementation
- Note: Not yet implemented in daemon

### 2. Error Handling

```python
import time
import requests

def fetch_with_retry(url, headers, max_retries=3, backoff=2):
    """Fetch with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.ok:
                return response.json()

            # If 403, don't retry (auth error)
            if response.status_code == 403:
                raise Exception("Authentication failed")

            # Rate limit hit, wait and retry
            if response.status_code == 429:
                time.sleep(backoff ** attempt)
                continue

        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            raise

    return None
```

### 3. Caching Strategies

**Client-Side Caching:**
```python
import time

class CachedClient:
    def __init__(self, client, cache_ttl=10):
        self.client = client
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_time = {}

    def get_predictions(self):
        now = time.time()
        if 'predictions' in self._cache:
            if now - self._cache_time['predictions'] < self.cache_ttl:
                return self._cache['predictions']

        data = self.client.get_predictions()
        if data:
            self._cache['predictions'] = data
            self._cache_time['predictions'] = now

        return data
```

**Use Case:** Reduce API calls when multiple dashboard components need same data

### 4. Rate Limit Management

The daemon enforces rate limits:
- `/predictions/current`: 30/minute
- `/alerts/active`: 30/minute
- `/explain/{server}`: 30/minute
- `/feed/data`: 60/minute

**Best Practices:**
- Cache predictions for 10+ seconds
- Batch requests when possible
- Use exponential backoff on 429 responses
- Monitor rate limit headers (if available)

### 5. Data Processing

**Extract Critical Servers:**
```python
def get_critical_servers(predictions, threshold=80):
    """Get servers above risk threshold."""
    critical = []
    for name, pred in predictions.get('predictions', {}).items():
        if pred.get('risk_score', 0) >= threshold:
            critical.append({
                'server': name,
                'risk': pred['risk_score'],
                'profile': pred['profile'],
                'cpu': 100 - pred['cpu_idle_pct']['current'],
                'memory': pred['mem_used_pct']['current'],
                'iowait': pred['cpu_iowait_pct']['current']
            })
    return sorted(critical, key=lambda x: x['risk'], reverse=True)
```

**Calculate Trends:**
```python
def calculate_trend(metric_obj):
    """Calculate if metric is improving or degrading."""
    current = metric_obj['current']
    predicted = metric_obj['predicted_1hr']

    diff = predicted - current
    if abs(diff) < 5:
        return 'stable'
    elif diff > 0:
        return 'degrading'
    else:
        return 'improving'
```

### 6. Security

**Production Checklist:**
- ✅ Always use HTTPS in production
- ✅ Store API keys in environment variables (never in code)
- ✅ Rotate API keys periodically
- ✅ Use network firewalls to restrict access
- ✅ Enable rate limiting
- ✅ Monitor for unauthorized access attempts
- ✅ Log all API requests

**Example: Secure Configuration**
```python
import os
from dotenv import load_load_dotenv()

# Load from .env file
NORDIQ_API_URL = os.getenv('NORDIQ_API_URL', 'http://localhost:8000')
NORDIQ_API_KEY = os.getenv('NORDIQ_API_KEY')

if not NORDIQ_API_KEY:
    raise ValueError("NORDIQ_API_KEY environment variable is required")
```

---

## Troubleshooting

### Issue: "Connection refused"

**Symptom:** `requests.exceptions.ConnectionError: Connection refused`

**Solution:**
```bash
# Check if daemon is running
curl http://localhost:8000/health

# If not running, start it
cd NordIQ
start_all.bat  # or ./start_all.sh
```

### Issue: "403 Forbidden"

**Symptom:** HTTP 403 response from API

**Solution:**
```bash
# Check API key is set
echo $TFT_API_KEY

# If empty, set it
export TFT_API_KEY="your-key-here"

# Verify key matches daemon
curl -H "X-API-Key: $TFT_API_KEY" http://localhost:8000/predictions/current
```

### Issue: "429 Too Many Requests"

**Symptom:** HTTP 429 response

**Solution:**
- Reduce polling frequency
- Implement client-side caching
- Add exponential backoff retry logic

### Issue: Empty predictions

**Symptom:** `{"predictions": {}, "error": "insufficient_data"}`

**Solution:**
```bash
# Daemon needs at least 100 data points
# Wait 8-10 minutes for rolling window to fill
# Or check metrics generator is running:
cd NordIQ/src/daemons
python metrics_generator_daemon.py --stream
```

### Issue: Stale data

**Symptom:** Timestamp in predictions is old

**Solution:**
```bash
# Restart daemon to reset rolling window
cd NordIQ
stop_all.bat
start_all.bat
```

### Issue: High latency

**Symptom:** API requests take >5 seconds

**Solution:**
- Check GPU availability (daemon uses GPU if available)
- Reduce number of servers monitored
- Increase daemon timeout settings
- Check system resources (CPU/Memory)

---

## Support & Resources

### Documentation
- [CURRENT_STATE.md](Docs/RAG/CURRENT_STATE.md) - System overview
- [PROJECT_CODEX.md](Docs/RAG/PROJECT_CODEX.md) - Development rules
- [API_KEY_SETUP.md](Docs/API_KEY_SETUP.md) - Authentication guide
- [WHY_TFT.md](Docs/WHY_TFT.md) - Technical background

### Example Code
- [api_client.py](NordIQ/src/dashboard/Dashboard/utils/api_client.py) - Reference Python client
- [tft_dashboard_web.py](NordIQ/src/dashboard/tft_dashboard_web.py) - Full dashboard implementation

### Contact
- **Company:** NordIQ AI, LLC
- **Website:** nordiqai.io
- **Developer:** Craig Giannelli

---

## Appendix: Complete API Examples

### Example 1: Monitor Critical Servers (Python)

```python
#!/usr/bin/env python3
"""Monitor critical servers and send alerts."""

import requests
import time
import smtplib
from email.mime.text import MIMEText

DAEMON_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"
ALERT_EMAIL = "oncall@example.com"
CHECK_INTERVAL = 30  # seconds

def get_critical_servers():
    """Get servers in critical state."""
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{DAEMON_URL}/predictions/current", headers=headers)

    if not response.ok:
        return []

    data = response.json()
    critical = []

    for name, pred in data.get('predictions', {}).items():
        if pred.get('risk_score', 0) >= 80:
            critical.append({
                'server': name,
                'risk': pred['risk_score'],
                'profile': pred['profile'],
                'level': pred['alert_level']
            })

    return critical

def send_alert(servers):
    """Send email alert for critical servers."""
    if not servers:
        return

    message = "Critical Servers Detected:\n\n"
    for server in servers:
        message += f"- {server['server']} ({server['profile']}): Risk {server['risk']}\n"

    # Send email (configure SMTP settings)
    msg = MIMEText(message)
    msg['Subject'] = f'Alert: {len(servers)} Critical Servers'
    msg['From'] = 'alerts@example.com'
    msg['To'] = ALERT_EMAIL

    # smtp = smtplib.SMTP('smtp.example.com')
    # smtp.send_message(msg)
    # smtp.quit()

    print(f"Alert sent: {len(servers)} critical servers")

def main():
    """Monitor loop."""
    print("Starting NordIQ monitor...")
    last_alert = {}

    while True:
        critical = get_critical_servers()

        # Only alert if new servers are critical
        current = {s['server']: s['risk'] for s in critical}
        if current != last_alert:
            send_alert(critical)
            last_alert = current

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()
```

### Example 2: Export to CSV (Python)

```python
#!/usr/bin/env python3
"""Export predictions to CSV for analysis."""

import requests
import pandas as pd
from datetime import datetime

DAEMON_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"

def export_predictions_to_csv(filename='predictions.csv'):
    """Export current predictions to CSV."""
    headers = {"X-API-Key": API_KEY}
    response = requests.get(f"{DAEMON_URL}/predictions/current", headers=headers)

    if not response.ok:
        print(f"Error: HTTP {response.status_code}")
        return

    data = response.json()
    timestamp = data.get('timestamp', datetime.now().isoformat())

    # Flatten predictions to rows
    rows = []
    for name, pred in data.get('predictions', {}).items():
        row = {
            'timestamp': timestamp,
            'server_name': name,
            'profile': pred.get('profile'),
            'risk_score': pred.get('risk_score'),
            'alert_level': pred.get('alert_level'),
            'cpu_used_pct': 100 - pred.get('cpu_idle_pct', {}).get('current', 0),
            'mem_used_pct': pred.get('mem_used_pct', {}).get('current', 0),
            'iowait_pct': pred.get('cpu_iowait_pct', {}).get('current', 0),
            'cpu_predicted_1hr': 100 - pred.get('cpu_idle_pct', {}).get('predicted_1hr', 0),
            'mem_predicted_1hr': pred.get('mem_used_pct', {}).get('predicted_1hr', 0)
        }
        rows.append(row)

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Exported {len(rows)} predictions to {filename}")

if __name__ == '__main__':
    export_predictions_to_csv()
```

### Example 3: Slack Integration (Python)

```python
#!/usr/bin/env python3
"""Send NordIQ alerts to Slack."""

import requests
import time

DAEMON_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"
SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
CHECK_INTERVAL = 60  # seconds

def send_slack_alert(servers):
    """Send critical server alert to Slack."""
    if not servers:
        return

    # Build Slack message
    text = f"🚨 *Alert: {len(servers)} Critical Servers*\n\n"
    for server in servers:
        emoji = "🔴" if server['risk'] >= 90 else "🟠"
        text += f"{emoji} *{server['server']}* ({server['profile']}): Risk {server['risk']}\n"

    payload = {
        "text": text,
        "username": "NordIQ Monitor",
        "icon_emoji": ":warning:"
    }

    response = requests.post(SLACK_WEBHOOK, json=payload)
    if response.ok:
        print(f"Slack alert sent: {len(servers)} servers")
    else:
        print(f"Slack error: {response.status_code}")

def monitor():
    """Monitor and alert."""
    headers = {"X-API-Key": API_KEY}
    last_critical = set()

    while True:
        try:
            response = requests.get(f"{DAEMON_URL}/alerts/active", headers=headers)
            if response.ok:
                data = response.json()
                alerts = data.get('alerts', [])

                # Filter critical only
                critical = [a for a in alerts if a['risk_score'] >= 80]
                current_critical = {a['server_name'] for a in critical}

                # Alert only on new critical servers
                if current_critical != last_critical:
                    send_slack_alert(critical)
                    last_critical = current_critical

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    monitor()
```

---

**Document Version:** 1.0.0
**Last Updated:** October 29, 2025
**Company:** NordIQ AI, LLC
**License:** Business Source License 1.1
