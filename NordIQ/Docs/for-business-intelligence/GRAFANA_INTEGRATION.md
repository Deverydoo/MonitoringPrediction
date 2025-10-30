# Grafana Integration Guide
**Visualize NordIQ Predictions in Grafana Dashboards**

**Version:** 1.0
**Audience:** BI Analysts, Grafana Administrators, DevOps Teams
**Purpose:** Complete guide to integrate NordIQ predictions into Grafana for business intelligence and monitoring

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation & Setup](#installation--setup)
4. [Dashboard Configuration](#dashboard-configuration)
5. [Panel Examples](#panel-examples)
6. [Variables & Filtering](#variables--filtering)
7. [Alerting Configuration](#alerting-configuration)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Grafana can visualize Nord IQ AI predictions using the **JSON API data source plugin**. This integration allows you to:

- **Monitor risk scores** across your entire fleet in real-time
- **Create alerts** based on predicted incidents before they happen
- **Build custom dashboards** for different teams (SRE, DevOps, Management)
- **Correlate predictions** with other monitoring data (Prometheus, Elasticsearch)
- **Export reports** for capacity planning and incident analysis

**Time to Complete:** 30-45 minutes

---

## Prerequisites

### 1. NordIQ System Running

Ensure the inference daemon is running and accessible:

```bash
# Test inference daemon
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

**Not running?** See [Quick Start Guide](../getting-started/QUICK_START.md)

### 2. Grafana Installed

**Version Required:** Grafana 8.0 or higher

**Installation:**
```bash
# Ubuntu/Debian
sudo apt-get install -y grafana

# CentOS/RHEL
sudo yum install grafana

# macOS
brew install grafana

# Docker
docker run -d -p 3000:3000 --name=grafana grafana/grafana
```

**Access Grafana:** http://localhost:3000
- Default username: `admin`
- Default password: `admin` (change on first login)

### 3. Network Access

Ensure Grafana can reach the inference daemon:

```bash
# From Grafana server, test connectivity
curl http://localhost:8000/health

# If remote, use full hostname
curl http://nordiq-server.company.com:8000/health
```

### 4. API Key

Get your NordIQ API key:

```bash
cd NordIQ/bin
python generate_api_key.py --show
```

**Copy the key** - you'll need it for Grafana configuration.

---

## Installation & Setup

### Step 1: Install JSON API Plugin

The **marcusolsson-json-datasource** plugin allows Grafana to query REST APIs.

**Install via Grafana CLI:**
```bash
# Stop Grafana first
sudo systemctl stop grafana-server

# Install plugin
grafana-cli plugins install marcusolsson-json-datasource

# Start Grafana
sudo systemctl start grafana-server
```

**Install via Docker:**
```bash
docker run -d \
  -p 3000:3000 \
  -e "GF_INSTALL_PLUGINS=marcusolsson-json-datasource" \
  --name=grafana \
  grafana/grafana
```

**Verify installation:**
1. Open Grafana: http://localhost:3000
2. Go to **Configuration → Plugins**
3. Search for "JSON API"
4. Should show as **Installed**

### Step 2: Configure Data Source

**Add NordIQ as a data source:**

1. In Grafana, go to **Configuration → Data Sources**
2. Click **Add data source**
3. Search for and select **JSON API**
4. Configure the settings:

```
Name: NordIQ Predictions
URL: http://localhost:8000
```

**If NordIQ is on a different server:**
```
URL: http://nordiq-server.company.com:8000
```

**Add authentication:**
1. Scroll to **Custom HTTP Headers**
2. Click **+ Add header**
3. Configure:
   - **Header:** `X-API-Key`
   - **Value:** `your-api-key-here` (from prerequisites)

**Advanced settings:**
- **Timeout:** `5` seconds
- **TLS Skip Verify:** Only if using self-signed certificates (not recommended)

**Test the connection:**
1. Click **Save & Test** at the bottom
2. Should see: ✅ **Data source is working**

**Troubleshooting:**
- ❌ "Failed to fetch" → Check URL and network connectivity
- ❌ "401 Unauthorized" → Check API key is correct
- ❌ "Timeout" → Increase timeout or check inference daemon health

---

## Dashboard Configuration

### Quick Start: Import Template

**Option A: Import Pre-Built Dashboard**

1. Download dashboard JSON: [NordIQ_Dashboard_Template.json](#)
2. In Grafana, go to **Dashboards → Import**
3. Upload JSON file or paste JSON content
4. Select **NordIQ Predictions** as data source
5. Click **Import**

**Option B: Build From Scratch**

Follow the panel examples below to create custom dashboards.

---

## Panel Examples

### Panel 1: Fleet Risk Overview (Time Series)

**Shows risk scores for all servers over time**

**Create new panel:**
1. Click **Add panel** → **Add new panel**
2. Select **NordIQ Predictions** data source
3. Configure query:

```json
{
  "url": "/predictions/current",
  "method": "GET"
}
```

4. **Transform data:**
   - Click **Transform** tab
   - Add transformation: **Extract fields**
   - JSONPath: `$.predictions.*.risk_score`
   - Labels from: `$.predictions.*.server_name`

5. **Visualization settings:**
   - Type: **Time series**
   - Title: "Fleet Risk Scores"
   - Legend: Show (right side)
   - Y-axis: 0-100 (Risk Score)
   - Thresholds:
     - 0-30: Green (Healthy)
     - 30-60: Yellow (Caution)
     - 60-80: Orange (Warning)
     - 80-100: Red (Critical)

6. **Refresh interval:** 10 seconds

### Panel 2: Critical Servers Count (Stat)

**Shows how many servers are in critical state**

**Create new panel:**
1. Click **Add panel** → **Add new panel**
2. Select **NordIQ Predictions** data source
3. Configure query:

```json
{
  "url": "/predictions/current",
  "method": "GET"
}
```

4. **Transform data:**
   - Add transformation: **Extract fields**
   - JSONPath: `$.summary.critical`

5. **Visualization settings:**
   - Type: **Stat**
   - Title: "Critical Servers"
   - Unit: none
   - Color scheme: Thresholds
   - Thresholds:
     - 0: Green
     - 1: Orange
     - 5: Red

6. **Refresh interval:** 5 seconds

### Panel 3: Server Status Table

**Shows detailed list of all servers with risk scores**

**Create new panel:**
1. Click **Add panel** → **Add new panel**
2. Select **NordIQ Predictions** data source
3. Configure query:

```json
{
  "url": "/predictions/current",
  "method": "GET"
}
```

4. **Transform data:**
   - Add transformation: **Extract fields**
   - JSONPath: `$.predictions.*`
   - Fields to extract:
     - `server_name` → Server
     - `profile` → Profile
     - `risk_score` → Risk Score
     - `risk_level` → Risk Level
     - `predicted_failures` → Predicted Issues

5. **Visualization settings:**
   - Type: **Table**
   - Title: "Server Status Overview"
   - Column settings:
     - Risk Score: Bar gauge, thresholds (0-30 green, 30-60 yellow, 60-80 orange, 80-100 red)
     - Risk Level: Color by value (Healthy=green, Warning=orange, Critical=red)
   - Sort: Risk Score (descending)

6. **Refresh interval:** 10 seconds

### Panel 4: Environment Health Gauge

**Shows average fleet risk score as a gauge**

**Create new panel:**
1. Click **Add panel** → **Add new panel**
2. Select **NordIQ Predictions** data source
3. Configure query:

```json
{
  "url": "/predictions/current",
  "method": "GET"
}
```

4. **Transform data:**
   - Add transformation: **Extract fields**
   - JSONPath: `$.summary.avg_risk_score`

5. **Visualization settings:**
   - Type: **Gauge**
   - Title: "Average Fleet Risk"
   - Min: 0, Max: 100
   - Unit: percent (0-100)
   - Thresholds:
     - 0-30: Green (Healthy Environment)
     - 30-60: Yellow (Monitor Closely)
     - 60-80: Orange (Action Needed)
     - 80-100: Red (Critical - Immediate Action)

6. **Refresh interval:** 10 seconds

### Panel 5: Alerts Timeline (Bar Chart)

**Shows when critical events are predicted to occur**

**Create new panel:**
1. Click **Add panel** → **Add new panel**
2. Select **NordIQ Predictions** data source
3. Configure query:

```json
{
  "url": "/alerts/active",
  "method": "GET"
}
```

4. **Transform data:**
   - Add transformation: **Extract fields**
   - JSONPath: `$.alerts.*`
   - Extract: `server_name`, `timestamp`, `level`, `risk_score`

5. **Visualization settings:**
   - Type: **Bar chart**
   - Title: "Active Alerts"
   - X-axis: Time
   - Y-axis: Server Name
   - Color: By alert level

6. **Refresh interval:** 30 seconds

### Panel 6: Profile Distribution (Pie Chart)

**Shows breakdown of servers by profile**

**Create new panel:**
1. Click **Add panel** → **Add new panel**
2. Select **NordIQ Predictions** data source
3. Configure query:

```json
{
  "url": "/predictions/current",
  "method": "GET"
}
```

4. **Transform data:**
   - Add transformation: **Group by**
   - Field: `profile`
   - Calculation: Count

5. **Visualization settings:**
   - Type: **Pie chart**
   - Title: "Server Profile Distribution"
   - Legend: Bottom
   - Show percentages: Yes

---

## Variables & Filtering

### Create Dashboard Variables

Variables allow dynamic filtering of data across all panels.

### Variable 1: Server Profile

**Purpose:** Filter by server type (database, web, ML, etc.)

**Configuration:**
1. Click **Dashboard settings** (gear icon) → **Variables**
2. Click **Add variable**
3. Configure:
   - **Name:** `server_profile`
   - **Label:** Server Profile
   - **Type:** Custom
   - **Custom options:**
     ```
     ml_compute,database,web_api,conductor_mgmt,data_ingest,risk_analytics,generic
     ```
   - **Multi-value:** Yes
   - **Include All option:** Yes

**Use in query:**
```json
{
  "url": "/predictions/current?profile=${server_profile}",
  "method": "GET"
}
```

### Variable 2: Risk Level

**Purpose:** Filter by risk severity

**Configuration:**
1. Click **Add variable**
2. Configure:
   - **Name:** `risk_level`
   - **Label:** Risk Level
   - **Type:** Custom
   - **Custom options:**
     ```
     Healthy,Degrading,Warning,Critical
     ```
   - **Multi-value:** Yes
   - **Include All option:** Yes

**Use in query:**
```json
{
  "url": "/predictions/current?level=${risk_level}",
  "method": "GET"
}
```

### Variable 3: Time Range

**Purpose:** Query historical predictions

**Configuration:**
1. Click **Add variable**
2. Configure:
   - **Name:** `time_range`
   - **Label:** Time Range
   - **Type:** Interval
   - **Auto Option:** Yes
   - **Intervals:** `1m,5m,15m,1h,6h,24h`

**Use in query:**
```json
{
  "url": "/predictions/historical?range=${time_range}",
  "method": "GET"
}
```

---

## Alerting Configuration

### Alert 1: Critical Server Detected

**Trigger when any server enters critical state**

**Configuration:**
1. Open a panel showing critical count (Panel 2)
2. Click **Alert** tab
3. Click **Create alert rule from this panel**
4. Configure:
   - **Name:** Critical Server Detected
   - **Evaluate every:** `1m` for `2m`
   - **Condition:**
     - WHEN: `last()`
     - OF: Critical Count
     - IS ABOVE: `0`
   - **Alert state if no data:** `Alerting`
   - **Alert state if execution error:** `Alerting`

5. **Notifications:**
   - Add notification channel: Slack, PagerDuty, Email
   - Message:
     ```
     🚨 Critical Server Alert
     ${server_name} has entered critical state!
     Risk Score: ${risk_score}
     Predicted Issue: ${predicted_failure}

     Dashboard: ${dashboard_link}
     ```

### Alert 2: Fleet Average Risk High

**Trigger when overall environment health deteriorates**

**Configuration:**
1. Open gauge panel (Panel 4)
2. Click **Alert** tab
3. Configure:
   - **Name:** Fleet Risk Elevated
   - **Evaluate every:** `5m` for `10m`
   - **Condition:**
     - WHEN: `avg()`
     - OF: Average Risk Score
     - IS ABOVE: `70`
   - **Alert state if no data:** `No Data`

4. **Notifications:**
   - Notification channel: Email (SRE team)
   - Message:
     ```
     ⚠️ Environment Health Warning
     Fleet average risk score has exceeded 70% for 10 minutes.
     Current average: ${avg_risk}%

     Critical servers: ${critical_count}
     Warning servers: ${warning_count}

     Review dashboard: ${dashboard_link}
     ```

### Alert 3: Prediction Service Down

**Trigger when inference daemon stops responding**

**Configuration:**
1. Open any panel
2. Click **Alert** tab
3. Configure:
   - **Name:** NordIQ Service Down
   - **Evaluate every:** `30s` for `1m`
   - **Condition:**
     - WHEN: `last()`
     - HAS NO VALUE
   - **Alert state if no data:** `Alerting`
   - **Alert state if execution error:** `Alerting`

4. **Notifications:**
   - Notification channel: PagerDuty (on-call)
   - Severity: Critical
   - Message:
     ```
     🔴 URGENT: NordIQ Prediction Service Down
     The TFT inference daemon has stopped responding.
     No prediction data received for 1 minute.

     Action Required: Check daemon status immediately
     ```

### Alert Best Practices

1. **Start conservative** - Begin with high thresholds, tune down based on false positives
2. **Use evaluation delays** - Prevent alert flapping with `for Xm` conditions
3. **Escalation chains** - Email → Slack → PagerDuty based on duration
4. **Silence during maintenance** - Use Grafana alert silencing for planned work
5. **Test alerts** - Use "Test rule" button to verify notification delivery

---

## Best Practices

### Performance Optimization

1. **Refresh intervals:**
   - Real-time panels (risk scores): 5-10 seconds
   - Historical panels: 1-5 minutes
   - Reports/analytics: 15 minutes

2. **Query caching:**
   - Enable Grafana query caching for panels that don't need real-time data
   - Configuration → Data Sources → NordIQ → Cache TTL: `60s`

3. **Dashboard organization:**
   - **Overview dashboard**: High-level metrics, 10-second refresh
   - **Detailed dashboard**: Per-server analysis, 30-second refresh
   - **Historical dashboard**: Trends and analytics, 5-minute refresh

### Security

1. **API key management:**
   - Store API keys in Grafana environment variables (not dashboard JSON)
   - Use Grafana secrets management for production

2. **Access control:**
   - Create Grafana teams (SRE, DevOps, Management)
   - Assign dashboard permissions by team
   - Use viewer roles for read-only access

3. **Network security:**
   - Run Grafana and NordIQ on same private network
   - Use HTTPS for Grafana (reverse proxy with nginx/Apache)
   - Firewall rules: Only allow Grafana server to access port 8000

### Dashboard Design

1. **Layout:**
   - Top row: Key metrics (critical count, average risk, health gauge)
   - Middle: Time series charts (trends over time)
   - Bottom: Detailed tables and lists

2. **Color consistency:**
   - Green: 0-30 (Healthy)
   - Yellow: 30-60 (Caution)
   - Orange: 60-80 (Warning)
   - Red: 80-100 (Critical)

3. **Annotations:**
   - Add deployment markers
   - Mark incident times
   - Note configuration changes

---

## Troubleshooting

### Issue: "Data source is not working"

**Cause:** Cannot connect to inference daemon

**Solutions:**
```bash
# Check daemon is running
curl http://localhost:8000/health

# Check firewall
sudo ufw status  # Linux
netsh advfirewall show allprofiles  # Windows

# Test from Grafana server
curl -H "X-API-Key: YOUR_KEY" http://nordiq-host:8000/predictions/current
```

### Issue: "401 Unauthorized"

**Cause:** API key missing or incorrect

**Solutions:**
```bash
# Verify API key
cd NordIQ/bin
python generate_api_key.py --show

# Update in Grafana data source
# Configuration → Data Sources → NordIQ → Custom Headers → X-API-Key
```

### Issue: "No data in panels"

**Cause:** JSONPath extraction incorrect or data format changed

**Solutions:**
1. Test query in browser:
   ```
   http://localhost:8000/predictions/current
   ```

2. Copy JSON response

3. In Grafana panel, click **Query inspector** → **JSON**

4. Verify JSONPath matches actual structure:
   - Use: `$.predictions.*.risk_score`
   - Not: `$.predictions[0].risk_score`

5. Test JSONPath: https://jsonpath.com

### Issue: "Panels not refreshing"

**Cause:** Query caching or stale data

**Solutions:**
1. Check panel refresh interval: **Panel → Edit → Query options → Refresh**
2. Disable browser cache: **F12 → Network → Disable cache**
3. Force refresh: **Ctrl+Shift+R** (hard reload)

### Issue: "Alerts not firing"

**Cause:** Condition not met or notification channel misconfigured

**Solutions:**
1. Test alert rule: **Alert tab → Test rule**
2. Check notification channel: **Alerting → Notification channels → Send test**
3. Review alert history: **Alerting → Alert rules → [rule name] → State history**
4. Check evaluation: Ensure condition matches actual data

---

## Next Steps

Once Grafana is configured:

1. **Create team dashboards** - Customize views for different teams
2. **Set up alerting** - Configure notification channels (Slack, PagerDuty, email)
3. **Export dashboards** - Save JSON for backup and version control
4. **Schedule reports** - Use Grafana Enterprise for PDF reports
5. **Integrate with other tools** - Correlate with Prometheus, Elasticsearch data

**See Also:**
- [API Reference](../for-developers/API_REFERENCE.md) - All available endpoints
- [Data Format Specification](../for-developers/DATA_FORMAT_SPEC.md) - Prediction JSON schema
- [Real Data Integration](../for-production/REAL_DATA_INTEGRATION.md) - Connect production systems

---

**Questions?** See [Troubleshooting Guide](../operations/TROUBLESHOOTING.md) or contact support.
