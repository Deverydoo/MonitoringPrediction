# NordIQ API Reference
**Complete REST API Documentation for the TFT Inference Daemon**

**Version:** 3.0
**Audience:** Software Developers, Integration Engineers
**Purpose:** Technical reference for all API endpoints and their parameters

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Endpoints](#endpoints)
5. [Response Codes](#response-codes)
6. [Rate Limiting](#rate-limiting)
7. [Versioning](#versioning)
8. [Examples](#examples)

---

## Overview

The NordIQ TFT Inference Daemon exposes a RESTful API for:

- **Data ingestion** - Send server metrics for prediction
- **Prediction retrieval** - Get current and historical predictions
- **Health monitoring** - Check system status
- **Explainability** - Understand why predictions were made (XAI)
- **Alert management** - Query active alerts

**API Style:** REST
**Data Format:** JSON
**Authentication:** API Key (X-API-Key header)
**Protocol:** HTTP/HTTPS

---

## Authentication

All endpoints (except `/health`) require API key authentication.

### API Key Header

```http
X-API-Key: your-api-key-here
```

### Getting Your API Key

```bash
cd NordIQ/bin
python generate_api_key.py --show
```

### Example Authenticated Request

```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/current
```

### Authentication Errors

**401 Unauthorized:**
```json
{
  "detail": "Invalid or missing API key"
}
```

**Troubleshooting:**
- Verify API key is correct
- Check `.env` file contains matching key
- Ensure header name is `X-API-Key` (case-sensitive)

---

## Base URL

### Local Development
```
http://localhost:8000
```

### Production (Example)
```
https://nordiq.company.com:8000
```

### Network Access
```
http://192.168.1.100:8000
```

**Note:** Replace with your actual hostname/IP.

---

## Endpoints

### 1. Health Check

**Check if the inference daemon is running**

```http
GET /health
```

**Authentication:** Not required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "version": "3.0.0",
  "running": true,
  "model_loaded": true,
  "uptime_seconds": 3600
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. System Status

**Get detailed system status including warmup progress**

```http
GET /status
```

**Authentication:** Required

**Response (200 OK):**
```json
{
  "status": "healthy",
  "servers_tracked": 25,
  "servers_warmed_up": 25,
  "servers_warming": 0,
  "total_records_received": 3750,
  "predictions_active": true,
  "last_update": "2025-10-30T15:30:00Z",
  "model_info": {
    "name": "tft_model_20251015_080653",
    "loaded": true,
    "profiles_supported": [
      "ml_compute",
      "database",
      "web_api",
      "conductor_mgmt",
      "data_ingest",
      "risk_analytics",
      "generic"
    ]
  },
  "warmup_threshold": 150
}
```

**Example:**
```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/status
```

---

### 3. Feed Data

**Send server metrics to the inference engine**

```http
POST /feed/data
```

**Authentication:** Required

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppdb001",
      "profile": "database",
      "cpu_user_pct": 45.2,
      "cpu_sys_pct": 15.3,
      "cpu_iowait_pct": 8.5,
      "cpu_idle_pct": 31.0,
      "java_cpu_pct": 0.0,
      "mem_used_pct": 67.8,
      "swap_used_pct": 2.1,
      "disk_usage_pct": 78.4,
      "net_in_mb_s": 25.3,
      "net_out_mb_s": 18.7,
      "back_close_wait": 15,
      "front_close_wait": 8,
      "load_average": 4.25,
      "uptime_days": 45
    }
  ]
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "accepted": 1,
  "rejected": 0,
  "servers_updated": 1,
  "timestamp": "2025-10-30T15:30:05Z",
  "warmup_status": {
    "total_servers": 1,
    "warmed_up": 0,
    "warming": 1,
    "warmup_threshold": 150
  }
}
```

**Response (400 Bad Request):**
```json
{
  "status": "error",
  "error": "Validation failed",
  "details": "Missing required field: mem_used_pct",
  "timestamp": "2025-10-30T15:30:05Z"
}
```

**See:** [Data Ingestion Guide](../for-production/DATA_INGESTION_GUIDE.md) for complete specification

**Example:**
```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"records": [...]}'
```

---

### 4. Get Current Predictions

**Retrieve latest predictions for all servers**

```http
GET /predictions/current
```

**Authentication:** Required

**Query Parameters:** None

**Response (200 OK):**
```json
{
  "timestamp": "2025-10-30T15:30:00Z",
  "summary": {
    "total_servers": 25,
    "critical": 2,
    "warning": 5,
    "degrading": 8,
    "healthy": 10,
    "avg_risk_score": 45.3
  },
  "predictions": {
    "ppdb001": {
      "server_name": "ppdb001",
      "profile": "database",
      "risk_score": 85.2,
      "risk_level": "Critical",
      "predicted_failures": [
        {
          "metric": "mem_used_pct",
          "current_value": 78.4,
          "predicted_value": 95.2,
          "time_to_failure_minutes": 45,
          "confidence": 0.89
        }
      ],
      "predictions": [
        {
          "timestamp": "2025-10-30T15:35:00Z",
          "cpu_user_pct": 46.1,
          "mem_used_pct": 79.2,
          "risk_score": 86.0
        },
        {
          "timestamp": "2025-10-30T15:40:00Z",
          "cpu_user_pct": 47.3,
          "mem_used_pct": 82.5,
          "risk_score": 88.5
        }
      ]
    },
    "ppweb002": {
      "server_name": "ppweb002",
      "profile": "web_api",
      "risk_score": 25.1,
      "risk_level": "Healthy",
      "predicted_failures": [],
      "predictions": [...]
    }
  }
}
```

**Example:**
```bash
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/current
```

---

### 5. Get Server Prediction

**Retrieve predictions for a specific server**

```http
GET /predictions/{server_name}
```

**Authentication:** Required

**Path Parameters:**
- `server_name` (string, required) - Server identifier (e.g., "ppdb001")

**Query Parameters:**
- `horizon` (integer, optional) - Number of prediction steps (default: 96, max: 96)
- `include_history` (boolean, optional) - Include past data (default: false)

**Response (200 OK):**
```json
{
  "server_name": "ppdb001",
  "profile": "database",
  "timestamp": "2025-10-30T15:30:00Z",
  "risk_score": 85.2,
  "risk_level": "Critical",
  "current_metrics": {
    "cpu_user_pct": 45.2,
    "mem_used_pct": 78.4,
    "disk_usage_pct": 82.1,
    "load_average": 8.5
  },
  "predicted_failures": [
    {
      "metric": "mem_used_pct",
      "current_value": 78.4,
      "predicted_value": 95.2,
      "threshold": 90.0,
      "time_to_failure_minutes": 45,
      "confidence": 0.89,
      "severity": "critical"
    }
  ],
  "predictions": [
    {
      "timestamp": "2025-10-30T15:35:00Z",
      "step": 1,
      "cpu_user_pct": 46.1,
      "mem_used_pct": 79.2,
      "disk_usage_pct": 82.3,
      "risk_score": 86.0
    },
    {
      "timestamp": "2025-10-30T15:40:00Z",
      "step": 2,
      "cpu_user_pct": 47.3,
      "mem_used_pct": 82.5,
      "disk_usage_pct": 82.5,
      "risk_score": 88.5
    }
  ]
}
```

**Response (404 Not Found):**
```json
{
  "detail": "Server 'unknown_server' not found"
}
```

**Example:**
```bash
# Basic request
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/predictions/ppdb001

# With parameters
curl -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/predictions/ppdb001?horizon=48&include_history=true"
```

---

### 6. Get Active Alerts

**Retrieve all active alerts (warnings and critical)**

```http
GET /alerts/active
```

**Authentication:** Required

**Query Parameters:**
- `level` (string, optional) - Filter by level: `Critical`, `Warning`, `Degrading`
- `profile` (string, optional) - Filter by profile: `database`, `web_api`, etc.
- `min_risk` (integer, optional) - Minimum risk score (0-100)

**Response (200 OK):**
```json
{
  "timestamp": "2025-10-30T15:30:00Z",
  "total_alerts": 7,
  "alerts": [
    {
      "server_name": "ppdb001",
      "profile": "database",
      "risk_score": 85.2,
      "level": "Critical",
      "message": "Memory exhaustion predicted in 45 minutes",
      "predicted_failure": {
        "metric": "mem_used_pct",
        "time_to_failure_minutes": 45,
        "confidence": 0.89
      },
      "timestamp": "2025-10-30T15:30:00Z"
    },
    {
      "server_name": "ppml0003",
      "profile": "ml_compute",
      "risk_score": 78.5,
      "level": "Warning",
      "message": "CPU trending high, degradation likely",
      "predicted_failure": {
        "metric": "cpu_user_pct",
        "time_to_failure_minutes": 120,
        "confidence": 0.76
      },
      "timestamp": "2025-10-30T15:30:00Z"
    }
  ]
}
```

**Example:**
```bash
# All alerts
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/alerts/active

# Critical only
curl -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/alerts/active?level=Critical"

# Database servers with risk > 80
curl -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/alerts/active?profile=database&min_risk=80"
```

---

### 7. Explainable AI (XAI)

**Get explanation for why a prediction was made**

```http
GET /xai/explain/{server_name}
```

**Authentication:** Required

**Path Parameters:**
- `server_name` (string, required) - Server identifier

**Query Parameters:**
- `step` (integer, optional) - Prediction step to explain (default: 0 = next step)

**Response (200 OK):**
```json
{
  "server_name": "ppdb001",
  "timestamp": "2025-10-30T15:30:00Z",
  "prediction_step": 1,
  "predicted_risk_score": 86.0,
  "explanation": {
    "top_contributing_factors": [
      {
        "feature": "mem_used_pct",
        "importance": 0.42,
        "current_value": 78.4,
        "direction": "increasing",
        "contribution_to_risk": "High memory usage is primary risk factor"
      },
      {
        "feature": "cpu_iowait_pct",
        "importance": 0.28,
        "current_value": 18.5,
        "direction": "stable",
        "contribution_to_risk": "High I/O wait indicates disk bottleneck"
      },
      {
        "feature": "swap_used_pct",
        "importance": 0.15,
        "current_value": 3.2,
        "direction": "increasing",
        "contribution_to_risk": "Memory thrashing detected"
      }
    ],
    "profile_baseline": {
      "profile": "database",
      "normal_mem_used_pct": 65.0,
      "current_deviation": "+13.4%"
    },
    "attention_weights": {
      "recent_history": 0.65,
      "temporal_patterns": 0.25,
      "profile_knowledge": 0.10
    }
  },
  "recommendation": "Memory exhaustion likely. Consider: 1) Restart memory-heavy processes, 2) Add RAM, 3) Review query performance"
}
```

**Example:**
```bash
# Explain next prediction
curl -H "X-API-Key: YOUR_KEY" http://localhost:8000/xai/explain/ppdb001

# Explain prediction at step 10 (50 minutes ahead)
curl -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/xai/explain/ppdb001?step=10"
```

---

### 8. Historical Predictions

**Query past predictions for analysis**

```http
GET /predictions/historical
```

**Authentication:** Required

**Query Parameters:**
- `server_name` (string, optional) - Filter by server
- `start_time` (ISO 8601, required) - Start of time range
- `end_time` (ISO 8601, required) - End of time range
- `profile` (string, optional) - Filter by profile

**Response (200 OK):**
```json
{
  "start_time": "2025-10-29T00:00:00Z",
  "end_time": "2025-10-30T00:00:00Z",
  "total_records": 17280,
  "servers": ["ppdb001", "ppweb002", "ppml0001"],
  "predictions": [
    {
      "timestamp": "2025-10-29T00:00:00Z",
      "server_name": "ppdb001",
      "risk_score": 45.2,
      "risk_level": "Healthy",
      "cpu_user_pct": 35.1,
      "mem_used_pct": 62.3
    },
    {
      "timestamp": "2025-10-29T00:00:05Z",
      "server_name": "ppdb001",
      "risk_score": 45.8,
      "risk_level": "Healthy",
      "cpu_user_pct": 36.2,
      "mem_used_pct": 62.5
    }
  ]
}
```

**Example:**
```bash
curl -H "X-API-Key: YOUR_KEY" \
  "http://localhost:8000/predictions/historical?server_name=ppdb001&start_time=2025-10-29T00:00:00Z&end_time=2025-10-30T00:00:00Z"
```

---

### 9. Scenario Control (Demo Mode Only)

**Control demo scenario for testing**

```http
POST /scenario/set
```

**Authentication:** Required

**Request Body:**
```json
{
  "scenario": "critical",
  "servers": ["ppdb001", "ppweb002"],
  "duration_minutes": 10
}
```

**Valid Scenarios:**
- `healthy` - Normal operation
- `degrading` - Gradual performance decline
- `critical` - Severe issues

**Response (200 OK):**
```json
{
  "status": "success",
  "scenario": "critical",
  "servers_affected": 2,
  "duration_minutes": 10,
  "message": "Scenario activated successfully"
}
```

**Note:** Only available when metrics generator daemon is running.

**Example:**
```bash
curl -X POST http://localhost:8000/scenario/set \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"scenario": "critical", "servers": ["ppdb001"], "duration_minutes": 10}'
```

---

## Response Codes

### Success Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful, data returned |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request successful, no data to return |

### Client Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Resource doesn't exist (e.g., server not found) |
| 413 | Payload Too Large | Request body exceeds size limit (max 1000 records) |
| 422 | Unprocessable Entity | Validation failed on request data |
| 429 | Too Many Requests | Rate limit exceeded |

### Server Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | Inference daemon is starting up or overloaded |

---

## Rate Limiting

**Current limits:**
- **Feed data:** 120 requests/minute (1 per 500ms)
- **Get predictions:** 60 requests/minute (1 per second)
- **Other endpoints:** 120 requests/minute

**Rate limit headers:**
```http
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 115
X-RateLimit-Reset: 1698765432
```

**Rate limit exceeded (429):**
```json
{
  "error": "Rate limit exceeded",
  "limit": 120,
  "retry_after_seconds": 15
}
```

**Best practices:**
- Batch multiple servers in single `/feed/data` request
- Cache `/predictions/current` responses for 5-10 seconds
- Use `/predictions/{server}` for single server queries

---

## Versioning

**Current version:** 3.0

**Version in response headers:**
```http
X-API-Version: 3.0
```

**Breaking changes:**
- Version changes indicated in release notes
- Old versions supported for 6 months after deprecation
- Use `Accept-Version` header to pin version (optional)

**Example:**
```bash
curl -H "Accept-Version: 3.0" http://localhost:8000/predictions/current
```

---

## Examples

### Python Client

```python
import requests
from datetime import datetime, timezone

class NordIQClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

    def health_check(self):
        """Check if service is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def get_status(self):
        """Get detailed system status."""
        response = requests.get(f"{self.base_url}/status", headers=self.headers)
        response.raise_for_status()
        return response.json()

    def send_metrics(self, records):
        """Send server metrics."""
        payload = {"records": records}
        response = requests.post(
            f"{self.base_url}/feed/data",
            json=payload,
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def get_predictions(self):
        """Get current predictions for all servers."""
        response = requests.get(
            f"{self.base_url}/predictions/current",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_server_prediction(self, server_name, horizon=96):
        """Get predictions for specific server."""
        response = requests.get(
            f"{self.base_url}/predictions/{server_name}",
            params={"horizon": horizon},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_active_alerts(self, level=None, min_risk=None):
        """Get active alerts."""
        params = {}
        if level:
            params["level"] = level
        if min_risk:
            params["min_risk"] = min_risk

        response = requests.get(
            f"{self.base_url}/alerts/active",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()

# Example usage
client = NordIQClient(api_key="your-api-key")

# Check health
health = client.health_check()
print(f"Service status: {health['status']}")

# Get predictions
predictions = client.get_predictions()
print(f"Total servers: {predictions['summary']['total_servers']}")
print(f"Critical servers: {predictions['summary']['critical']}")

# Get specific server
server_pred = client.get_server_prediction("ppdb001")
print(f"Risk score: {server_pred['risk_score']}")

# Get critical alerts only
alerts = client.get_active_alerts(level="Critical")
print(f"Critical alerts: {len(alerts['alerts'])}")
```

### JavaScript Client

```javascript
class NordIQClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey
    };
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }

  async getStatus() {
    const response = await fetch(`${this.baseUrl}/status`, {
      headers: this.headers
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async sendMetrics(records) {
    const response = await fetch(`${this.baseUrl}/feed/data`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ records })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async getPredictions() {
    const response = await fetch(`${this.baseUrl}/predictions/current`, {
      headers: this.headers
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async getServerPrediction(serverName, horizon = 96) {
    const url = `${this.baseUrl}/predictions/${serverName}?horizon=${horizon}`;
    const response = await fetch(url, { headers: this.headers });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async getActiveAlerts(level = null, minRisk = null) {
    const params = new URLSearchParams();
    if (level) params.append('level', level);
    if (minRisk) params.append('min_risk', minRisk);

    const url = `${this.baseUrl}/alerts/active?${params}`;
    const response = await fetch(url, { headers: this.headers });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
}

// Example usage
const client = new NordIQClient('http://localhost:8000', 'your-api-key');

// Check health
const health = await client.healthCheck();
console.log(`Service status: ${health.status}`);

// Get predictions
const predictions = await client.getPredictions();
console.log(`Critical servers: ${predictions.summary.critical}`);

// Get critical alerts
const alerts = await client.getActiveAlerts('Critical');
console.log(`Critical alerts: ${alerts.alerts.length}`);
```

---

## See Also

- [Data Ingestion Guide](../for-production/DATA_INGESTION_GUIDE.md) - Complete `/feed/data` specification
- [Data Format Specification](DATA_FORMAT_SPEC.md) - Detailed schema documentation
- [Custom Dashboard Guide](CUSTOM_DASHBOARD_GUIDE.md) - Build your own UI
- [Python Client Examples](PYTHON_CLIENT_EXAMPLES.md) - More code samples

---

**Questions?** See [Troubleshooting Guide](../operations/TROUBLESHOOTING.md) or contact support.
