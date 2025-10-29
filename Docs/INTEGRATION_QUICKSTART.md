# NordIQ Integration Quick Start

**5-Minute Guide to Connecting Your Dashboard**

---

## What You'll Learn

How to connect to the NordIQ TFT Inference Daemon and get AI predictions into your own dashboard in under 5 minutes.

**Full Documentation:** See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for complete API reference, examples, and troubleshooting.

---

## Step 1: Start the Daemon (30 seconds)

```bash
cd NordIQ
start_all.bat  # Windows
./start_all.sh # Linux/Mac
```

This starts the inference daemon on **http://localhost:8000**

---

## Step 2: Get Your API Key (30 seconds)

```bash
cd NordIQ/bin
python generate_api_key.py --show
```

Copy the key shown, you'll need it for API requests.

---

## Step 3: Test the Connection (30 seconds)

```bash
# Health check (no key needed)
curl http://localhost:8000/health

# Get predictions (with key)
curl -H "X-API-Key: YOUR-KEY-HERE" http://localhost:8000/predictions/current
```

---

## Step 4: Choose Your Integration (3 minutes)

### Option A: Python Dashboard

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your-key-here"

headers = {"X-API-Key": API_KEY}
response = requests.get(f"{API_URL}/predictions/current", headers=headers)
predictions = response.json()

# Show critical servers
for name, server in predictions['predictions'].items():
    if server['risk_score'] >= 80:
        print(f"🔴 {name}: Risk {server['risk_score']}")
```

### Option B: JavaScript Dashboard

```javascript
const API_URL = 'http://localhost:8000';
const API_KEY = 'your-key-here';

fetch(`${API_URL}/predictions/current`, {
  headers: { 'X-API-Key': API_KEY }
})
  .then(response => response.json())
  .then(data => {
    console.log(`Monitoring ${data.prediction_count} servers`);
    console.log(`Critical alerts: ${data.summary.critical}`);
  });
```

### Option C: Grafana

1. Install JSON API plugin: `grafana-cli plugins install marcusolsson-json-datasource`
2. Add data source: Configuration → Data Sources → JSON API
3. Configure:
   - URL: `http://localhost:8000`
   - Custom Header: `X-API-Key` = `your-key-here`
4. Create panels using `/predictions/current` endpoint

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#grafana-integration) for complete Grafana setup.

---

## Key API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `/health` | No | Check daemon is running |
| `/status` | No | Get daemon status |
| `/predictions/current` | Yes | Get all server predictions + risk scores |
| `/alerts/active` | Yes | Get only servers with alerts |
| `/explain/{server}` | Yes | Get AI explanation for prediction |

**Full API Reference:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#api-reference)

---

## Response Format (Simplified)

```json
{
  "timestamp": "2025-10-29T14:30:00",
  "prediction_count": 20,
  "summary": {
    "critical": 2,
    "warning": 5,
    "healthy": 13,
    "avg_risk_score": 42.3
  },
  "predictions": {
    "ppml0001": {
      "server_name": "ppml0001",
      "profile": "ML Compute",
      "risk_score": 87.5,
      "alert_level": "Critical",
      "cpu_idle_pct": {
        "current": 15.2,
        "predicted_1hr": 5.3
      },
      "mem_used_pct": {
        "current": 92.3,
        "predicted_1hr": 98.2
      },
      "... (12 more metrics)"
    }
  }
}
```

**Full Data Format:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#data-format-reference)

---

## Common Tasks

### Get Critical Servers Only

```python
predictions = requests.get(
    f"{API_URL}/alerts/active",
    headers={"X-API-Key": API_KEY}
).json()

for alert in predictions['alerts']:
    print(f"{alert['server_name']}: {alert['message']}")
```

### Export to CSV

```python
import pandas as pd

predictions = requests.get(
    f"{API_URL}/predictions/current",
    headers={"X-API-Key": API_KEY}
).json()

rows = []
for name, server in predictions['predictions'].items():
    rows.append({
        'server': name,
        'risk': server['risk_score'],
        'cpu': 100 - server['cpu_idle_pct']['current'],
        'memory': server['mem_used_pct']['current']
    })

df = pd.DataFrame(rows)
df.to_csv('predictions.csv', index=False)
```

### Send to Slack

```python
import requests

# Get critical alerts
alerts = requests.get(
    f"{API_URL}/alerts/active",
    headers={"X-API-Key": API_KEY}
).json()

# Format message
text = f"🚨 {alerts['count']} Critical Servers\n"
for alert in alerts['alerts'][:5]:  # Top 5
    text += f"- {alert['server_name']}: Risk {alert['risk_score']}\n"

# Send to Slack
requests.post("https://hooks.slack.com/YOUR/WEBHOOK", json={"text": text})
```

**More Examples:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#integration-examples)

---

## Rate Limits

- `/predictions/current`: **30/minute** (poll every 10-30 seconds)
- `/alerts/active`: **30/minute**
- `/explain/{server}`: **30/minute**

**Best Practice:** Cache predictions for 10+ seconds, don't poll faster than every 5 seconds.

---

## Troubleshooting

**"Connection refused"** → Daemon not running, run `start_all.bat`

**"403 Forbidden"** → Wrong API key, check with `python generate_api_key.py --show`

**"Empty predictions"** → Wait 8-10 minutes for rolling window to fill with data

**Full Troubleshooting:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md#troubleshooting)

---

## What's Next?

1. **Read Full Guide:** [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
   - Complete API reference
   - Grafana setup guide
   - React/Vue/Angular examples
   - Security best practices

2. **Explore Code:**
   - [api_client.py](../NordIQ/src/dashboard/Dashboard/utils/api_client.py) - Reference Python client
   - [tft_dashboard_web.py](../NordIQ/src/dashboard/tft_dashboard_web.py) - Full dashboard example

3. **Customize:**
   - Build your own dashboard UI
   - Integrate with existing monitoring tools
   - Create specialized views for your team

---

## Support

- **Documentation:** [Docs/](.)
- **Company:** NordIQ AI, LLC
- **Website:** nordiqai.io

---

**Happy Integrating!** 🚀
