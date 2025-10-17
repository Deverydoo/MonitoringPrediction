# Adapter Architecture & Data Flow

**Version:** 1.0.0
**Created:** 2025-10-17
**Status:** Critical Documentation - System Architecture

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [How Adapters Work](#how-adapters-work)
4. [Data Flow (Step-by-Step)](#data-flow-step-by-step)
5. [Process Management](#process-management)
6. [Communication Protocols](#communication-protocols)
7. [Startup Procedures](#startup-procedures)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## 🎯 Overview

### **Critical Concept: Adapters Run as Independent Daemons**

**Adapters are NOT called by the inference daemon.**
**Adapters actively PUSH data to the inference daemon.**

This is a **microservices architecture** where each component runs independently and communicates via HTTP APIs.

---

## 🏗️ Architecture Diagram

### **Three Independent Processes**

```
┌─────────────────────────────────────────────────────────────────┐
│                  PRODUCTION ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ Process 1: Data Source Adapter (MongoDB/Elasticsearch)      │
│ Port: None (HTTP client only)                               │
│ Role: Active Data Fetcher & Pusher                          │
└──────────────────────────────────────────────────────────────┘
        │
        │ Every 5 seconds:
        │ 1. Fetch from database
        │ 2. Transform to TFT format
        │ 3. HTTP POST to /feed
        │
        ↓ HTTP POST /feed

┌──────────────────────────────────────────────────────────────┐
│ Process 2: TFT Inference Daemon                             │
│ Port: 8000 (HTTP server)                                    │
│ Role: Data Receiver, Prediction Generator                   │
└──────────────────────────────────────────────────────────────┘
        │
        │ On demand:
        │ Dashboard requests predictions
        │
        ↓ HTTP GET /predict

┌──────────────────────────────────────────────────────────────┐
│ Process 3: Streamlit Dashboard                              │
│ Port: 8501 (web server)                                     │
│ Role: Visualization & User Interface                        │
└──────────────────────────────────────────────────────────────┘
```

### **Communication Flow**

```
MongoDB/Elasticsearch
    ↓ (query)
MongoDB Adapter (Process 1)
    ↓ (HTTP POST to /feed)
TFT Inference Daemon (Process 2)
    ↓ (HTTP GET from /predict)
Dashboard (Process 3)
    ↓ (display)
User Browser
```

---

## 🔄 How Adapters Work

### **Adapter Daemon Loop (Continuous)**

```python
# adapters/mongodb_adapter.py - Main Loop

def run_daemon(self, interval: int = 5):
    """Run continuous streaming daemon."""

    logger.info("🚀 Starting MongoDB adapter daemon")
    logger.info(f"   Fetch interval: {interval} seconds")

    self.last_fetch_time = datetime.utcnow() - timedelta(seconds=interval)

    while True:  # ← Infinite loop
        # Step 1: Fetch new metrics from MongoDB
        metrics = self.fetch_recent_metrics(since=self.last_fetch_time)
        # Query: db.server_metrics.find({timestamp: {$gte: last_fetch_time}})

        if metrics:
            # Step 2: Transform MongoDB docs → TFT format
            records = self.transform_to_tft_format(metrics)
            # Converts field names, handles nested structures

            # Step 3: Forward to TFT daemon
            self.forward_to_tft_daemon(records)
            # HTTP POST http://localhost:8000/feed

            # Step 4: Update state
            self.last_fetch_time = latest_metric_timestamp

        # Step 5: Sleep until next interval
        time.sleep(interval)  # Default: 5 seconds
```

### **Key Characteristics**

| Characteristic | Description |
|----------------|-------------|
| **Active Fetcher** | Adapter actively queries database every N seconds |
| **Push-Based** | Adapter pushes data to inference daemon (not pulled) |
| **Stateful** | Tracks last fetch time to avoid duplicate data |
| **Independent Process** | Runs separately, can restart without affecting inference |
| **HTTP Client** | Makes HTTP POST requests to daemon's `/feed` endpoint |
| **No Port** | Doesn't listen on any port (client-only) |

---

## 📊 Data Flow (Step-by-Step)

### **Minute-by-Minute Timeline Example**

```
Time    | Adapter Action                      | Inference Daemon              | Dashboard
--------|-------------------------------------|-------------------------------|------------------
12:00:00| ─ Fetch metrics (12:00:00-11:59:55)| ─ Waiting for data           | ─ Shows old data
12:00:01| ─ Transform 20 records             | ─ Waiting                     | ─ Shows old data
12:00:02| ─ POST /feed (20 records)          | ✅ Received 20 records        | ─ Shows old data
        |                                     | ─ Add to warmup buffer       |
        |                                     | ─ Check if warmed up         |
12:00:03| ─ Sleep 5 seconds                  | ─ Ready for predictions      | ─ Shows old data
12:00:04| ─ Sleeping...                      | ─ Idle                        | ─ Shows old data
12:00:05| ─ Sleeping...                      | ─ Idle                        | ─ Sleeping...
12:00:06| ─ Sleeping...                      | ─ Idle                        | 🔄 Refresh (30s)
12:00:07| ─ Sleeping...                      | ─ Idle                        | ← GET /predict
        |                                     | ✅ Generate predictions      |
        |                                     | → Return predictions         |
12:00:08| ─ Wake up, start next fetch        | ─ Idle                        | ✅ Display new data
12:00:09| ─ Fetch metrics (12:00:08-12:00:03)| ─ Waiting for data           | ─ Shows new data
...     | (Repeat every 5 seconds)           | (Serves predictions on req)   | (Refreshes 30s)
```

---

## 🔌 Communication Protocols

### **🔐 Authentication**

All HTTP requests between components use API key authentication via the `X-API-Key` header.

**How API Keys Work:**
1. Run `python generate_api_key.py` (done automatically by startup scripts)
2. API key is stored in `.env` file: `TFT_API_KEY=abc123...`
3. Adapters automatically load key from `.env` when they start
4. Every HTTP request includes: `X-API-Key: abc123...`
5. Daemon validates key before processing request

**Priority Order (Adapters):**
1. `api_key` field in adapter config file (explicit override)
2. `.env` file in project root (recommended - automatic)
3. `TFT_API_KEY` environment variable (fallback)

**Security:** API keys provide authentication to prevent unauthorized data injection or prediction access.

---

### **1. Adapter → Inference Daemon (Push)**

**Endpoint:** `POST /feed`

**Request:**
```http
POST http://localhost:8000/feed HTTP/1.1
Content-Type: application/json
X-API-Key: abc123def456...  ← Automatically loaded from .env

[
  {
    "timestamp": "2025-10-17T12:00:00Z",
    "server_name": "ppml0001",
    "profile": "ml_compute",
    "cpu_user_pct": 65.4,
    "cpu_sys_pct": 12.3,
    "cpu_iowait_pct": 2.1,
    "cpu_idle_pct": 20.2,
    "java_cpu_pct": 45.0,
    "mem_used_pct": 85.2,
    "swap_used_pct": 0.0,
    "disk_usage_pct": 45.0,
    "net_in_mb_s": 25.3,
    "net_out_mb_s": 18.7,
    "back_close_wait": 0,
    "front_close_wait": 0,
    "load_average": 8.5,
    "uptime_days": 42.0
  },
  {
    "timestamp": "2025-10-17T12:00:00Z",
    "server_name": "ppdb001",
    ...
  }
]
```

**Response:**
```json
{
  "status": "ok",
  "received": 20,
  "warmup_progress": 65.2
}
```

**Frequency:** Every 5 seconds (configurable via `--interval`)

---

### **2. Dashboard → Inference Daemon (Pull)**

**Endpoint:** `GET /predict`

**Request:**
```http
GET http://localhost:8000/predict HTTP/1.1
X-API-Key: abc123def456...  ← Automatically loaded from .streamlit/secrets.toml
```

**Response:**
```json
{
  "predictions": {
    "ppml0001": {
      "current": {...},
      "predictions_30m": {...},
      "predictions_8h": {...},
      "risk_score": 58
    },
    ...
  },
  "timestamp": "2025-10-17T12:00:05Z"
}
```

**Frequency:** Every 30 seconds (configurable in dashboard)

---

## 🚀 Process Management

### **Process Lifecycle**

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS STATES                           │
└─────────────────────────────────────────────────────────────┘

1. Inference Daemon (Must start FIRST)
   ─────────────────────────────────────
   Start: python tft_inference_daemon.py --port 8000
   State: LISTENING on port 8000
   Waits: For /feed POST requests
   Critical: Must be running before adapter starts

2. Adapter Daemon (Start SECOND)
   ─────────────────────────────────────
   Start: python adapters/mongodb_adapter.py --daemon
   State: RUNNING continuous loop
   Action: Fetching → Transforming → POSTing
   Depends: Requires inference daemon at localhost:8000

3. Dashboard (Start THIRD)
   ─────────────────────────────────────
   Start: streamlit run tft_dashboard_web.py
   State: WEB SERVER on port 8501
   Action: Fetching predictions every 30s
   Depends: Requires inference daemon at localhost:8000
```

### **Checking Running Processes**

```bash
# Linux
ps aux | grep -E "(tft_inference|mongodb_adapter|elasticsearch_adapter|streamlit)"

# Windows
tasklist | findstr /I "python streamlit"

# Expected output:
python tft_inference_daemon.py --port 8000
python mongodb_adapter.py --daemon --interval 5
streamlit run tft_dashboard_web.py
```

### **Process Dependencies**

```
Inference Daemon (port 8000)
    ↑
    ├── MongoDB Adapter (depends on daemon)
    └── Dashboard (depends on daemon)
```

**Critical:** Inference daemon MUST be running before starting adapter or dashboard.

---

## 🔧 Startup Procedures

### **Development Mode (Manual Start)**

```bash
# Terminal 1: Inference Daemon (FIRST)
conda activate py310
python tft_inference_daemon.py --port 8000
# Wait for: ✅ Inference daemon started on port 8000

# Terminal 2: Adapter (SECOND)
conda activate py310
python adapters/mongodb_adapter.py --daemon --interval 5
# Wait for: ✅ Connected to MongoDB
# Wait for: ✅ Forwarded X records to TFT daemon

# Terminal 3: Dashboard (THIRD)
conda activate py310
streamlit run tft_dashboard_web.py
# Open browser: http://localhost:8501
```

### **Production Mode (Automated Script)**

Create `start_all_production.bat`:

```batch
@echo off
echo ============================================
echo TFT Production System - Starting
echo ============================================
echo.

REM Load API key
for /f "usebackq tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="TFT_API_KEY" set TFT_API_KEY=%%b
)

REM Step 1: Start Inference Daemon (CRITICAL FIRST)
echo [1/3] Starting Inference Daemon...
start "TFT Inference Daemon" cmd /k "conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python tft_inference_daemon.py --port 8000"

REM Wait for daemon to initialize
echo [INFO] Waiting for daemon to initialize (8 seconds)...
timeout /t 8 /nobreak >nul

REM Step 2: Start Adapter (SECOND)
echo [2/3] Starting MongoDB Adapter...
start "MongoDB Adapter" cmd /k "conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python adapters\mongodb_adapter.py --daemon --config adapters\mongodb_adapter_config.json"

REM Wait for adapter to connect
echo [INFO] Waiting for adapter to connect (3 seconds)...
timeout /t 3 /nobreak >nul

REM Step 3: Start Dashboard (THIRD)
echo [3/3] Starting Dashboard...
start "TFT Dashboard" cmd /k "conda activate py310 && streamlit run tft_dashboard_web.py --server.fileWatcherType none --server.runOnSave false"

echo.
echo ============================================
echo System Started!
echo ============================================
echo Inference Daemon:   http://localhost:8000
echo MongoDB Adapter:    Streaming from production
echo Dashboard:          http://localhost:8501
echo.
echo Press any key to continue...
pause >nul
```

**Linux version (`start_all_production.sh`):**

```bash
#!/bin/bash
echo "============================================"
echo "TFT Production System - Starting"
echo "============================================"
echo ""

# Load API key from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Step 1: Start Inference Daemon (CRITICAL FIRST)
echo "[1/3] Starting Inference Daemon..."
gnome-terminal -- bash -c "conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python tft_inference_daemon.py --port 8000; exec bash"

# Wait for daemon to initialize
echo "[INFO] Waiting for daemon to initialize (8 seconds)..."
sleep 8

# Step 2: Start Adapter (SECOND)
echo "[2/3] Starting MongoDB Adapter..."
gnome-terminal -- bash -c "conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python adapters/mongodb_adapter.py --daemon --config adapters/mongodb_adapter_config.json; exec bash"

# Wait for adapter to connect
echo "[INFO] Waiting for adapter to connect (3 seconds)..."
sleep 3

# Step 3: Start Dashboard (THIRD)
echo "[3/3] Starting Dashboard..."
gnome-terminal -- bash -c "conda activate py310 && streamlit run tft_dashboard_web.py --server.fileWatcherType none --server.runOnSave false; exec bash"

echo ""
echo "============================================"
echo "System Started!"
echo "============================================"
echo "Inference Daemon:   http://localhost:8000"
echo "MongoDB Adapter:    Streaming from production"
echo "Dashboard:          http://localhost:8501"
```

---

## 🏭 Production Deployment

### **Systemd Services (Linux)**

#### **1. Inference Daemon Service**

`/etc/systemd/system/tft-inference-daemon.service`:

```ini
[Unit]
Description=TFT Inference Daemon
After=network.target
Documentation=https://github.com/yourorg/tft-monitoring

[Service]
Type=simple
User=tft
Group=tft
WorkingDirectory=/opt/tft-monitoring
EnvironmentFile=/etc/tft/.env
ExecStart=/opt/tft-monitoring/venv/bin/python tft_inference_daemon.py --port 8000
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### **2. MongoDB Adapter Service**

`/etc/systemd/system/tft-mongodb-adapter.service`:

```ini
[Unit]
Description=TFT MongoDB Adapter
After=network.target tft-inference-daemon.service mongodb.service
Requires=tft-inference-daemon.service
Documentation=https://github.com/yourorg/tft-monitoring

[Service]
Type=simple
User=tft
Group=tft
WorkingDirectory=/opt/tft-monitoring/adapters
EnvironmentFile=/etc/tft/.env
ExecStart=/opt/tft-monitoring/venv/bin/python mongodb_adapter.py --daemon --config /etc/tft/mongodb_adapter_config.json
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### **3. Dashboard Service**

`/etc/systemd/system/tft-dashboard.service`:

```ini
[Unit]
Description=TFT Streamlit Dashboard
After=network.target tft-inference-daemon.service
Requires=tft-inference-daemon.service
Documentation=https://github.com/yourorg/tft-monitoring

[Service]
Type=simple
User=tft
Group=tft
WorkingDirectory=/opt/tft-monitoring
ExecStart=/opt/tft-monitoring/venv/bin/streamlit run tft_dashboard_web.py --server.fileWatcherType none --server.runOnSave false --server.port 8501
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### **Enable and Start Services**

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable tft-inference-daemon
sudo systemctl enable tft-mongodb-adapter
sudo systemctl enable tft-dashboard

# Start services (in order)
sudo systemctl start tft-inference-daemon
sleep 5
sudo systemctl start tft-mongodb-adapter
sleep 3
sudo systemctl start tft-dashboard

# Check status
sudo systemctl status tft-inference-daemon
sudo systemctl status tft-mongodb-adapter
sudo systemctl status tft-dashboard

# View logs
journalctl -u tft-inference-daemon -f
journalctl -u tft-mongodb-adapter -f
journalctl -u tft-dashboard -f
```

---

### **Docker Compose (Container Deployment)**

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Service 1: Inference Daemon (must start first)
  tft-inference-daemon:
    image: tft-monitoring:latest
    container_name: tft-inference-daemon
    command: python tft_inference_daemon.py --port 8000
    ports:
      - "8000:8000"
    environment:
      - TFT_API_KEY=${TFT_API_KEY}
    volumes:
      - ./models:/app/models
      - ./training:/app/training
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Service 2: MongoDB Adapter (depends on daemon)
  tft-mongodb-adapter:
    image: tft-monitoring:latest
    container_name: tft-mongodb-adapter
    command: python adapters/mongodb_adapter.py --daemon --config /config/mongodb_adapter_config.json
    depends_on:
      tft-inference-daemon:
        condition: service_healthy
    environment:
      - TFT_API_KEY=${TFT_API_KEY}
    volumes:
      - ./adapters/mongodb_adapter_config.json:/config/mongodb_adapter_config.json:ro
    restart: unless-stopped

  # Service 3: Dashboard (depends on daemon)
  tft-dashboard:
    image: tft-monitoring:latest
    container_name: tft-dashboard
    command: streamlit run tft_dashboard_web.py --server.fileWatcherType none --server.runOnSave false --server.port 8501
    ports:
      - "8501:8501"
    depends_on:
      - tft-inference-daemon
    environment:
      - TFT_API_KEY=${TFT_API_KEY}
    restart: unless-stopped

networks:
  default:
    name: tft-network
```

**Start with Docker Compose:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop all services
docker-compose down
```

---

### **Windows Service (NSSM)**

```batch
REM Download NSSM from https://nssm.cc/

REM Service 1: Inference Daemon
nssm install TFTInferenceDaemon ^
  "C:\Python310\python.exe" ^
  "D:\tft-monitoring\tft_inference_daemon.py" ^
  --port 8000

nssm set TFTInferenceDaemon AppDirectory D:\tft-monitoring
nssm set TFTInferenceDaemon AppEnvironmentExtra TFT_API_KEY=your-api-key

REM Service 2: MongoDB Adapter (depends on daemon)
nssm install TFTMongoDBAdapter ^
  "C:\Python310\python.exe" ^
  "D:\tft-monitoring\adapters\mongodb_adapter.py" ^
  --daemon --config D:\tft-monitoring\adapters\mongodb_adapter_config.json

nssm set TFTMongoDBAdapter AppDirectory D:\tft-monitoring\adapters
nssm set TFTMongoDBAdapter DependOnService TFTInferenceDaemon

REM Service 3: Dashboard (depends on daemon)
nssm install TFTDashboard ^
  "C:\Python310\Scripts\streamlit.exe" ^
  run tft_dashboard_web.py

nssm set TFTDashboard AppDirectory D:\tft-monitoring
nssm set TFTDashboard DependOnService TFTInferenceDaemon

REM Start services (in order)
nssm start TFTInferenceDaemon
timeout /t 5 /nobreak
nssm start TFTMongoDBAdapter
timeout /t 3 /nobreak
nssm start TFTDashboard
```

---

## 🐛 Troubleshooting

### **Problem 1: "Adapter can't connect to inference daemon"**

**Symptoms:**
```
❌ Error forwarding to TFT daemon: Connection refused
```

**Diagnosis:**
```bash
# Check if inference daemon is running
curl http://localhost:8000/health

# Check if port 8000 is listening
netstat -an | grep 8000        # Linux
netstat -an | findstr 8000     # Windows
```

**Solution:**
```bash
# Start inference daemon FIRST
python tft_inference_daemon.py --port 8000

# Wait 5 seconds, then start adapter
python adapters/mongodb_adapter.py --daemon
```

---

### **Problem 2: "Dashboard shows 0/0 servers"**

**Symptoms:**
- Dashboard shows "Environment Status: Unknown"
- Fleet Status: 0/0 servers

**Diagnosis:**
```bash
# Check adapter logs
# Should show: ✅ Forwarded X records to TFT daemon

# Check inference daemon logs
# Should show: [FEED] Received X metrics

# Check dashboard connection
curl http://localhost:8000/predict -H "X-API-Key: your-key"
```

**Solution:**
1. Verify adapter is running and forwarding data
2. Check API key matches in adapter config and .env
3. Wait for warmup (288 data points per server needed)

---

### **Problem 3: "Adapter fetches duplicate data"**

**Symptoms:**
```
📊 Fetched 1000 metrics (same timestamps repeated)
```

**Diagnosis:**
```bash
# Check adapter state tracking
# Adapter should track last_fetch_time

# Check database has new data
mongo linborg --eval "db.server_metrics.find().sort({timestamp: -1}).limit(1)"
```

**Solution:**
- Adapter tracks state internally (last_fetch_time)
- Only fetches data newer than last fetch
- If duplicate data appears, check database clock sync

---

### **Problem 4: "Authentication failed" or "403 Forbidden"**

**Symptoms:**
```
❌ TFT daemon error: 403 Forbidden
⚠️ No API key configured - daemon may reject request
```

**Diagnosis:**
```bash
# Step 1: Check if .env file exists
cat .env | grep TFT_API_KEY
# Should show: TFT_API_KEY=abc123...

# Step 2: Check if adapter is loading the key
python adapters/mongodb_adapter.py --once --verbose
# Should see: ✅ Loaded API key from .env

# Step 3: Test daemon authentication
curl -X POST http://localhost:8000/health \
  -H "X-API-Key: $(grep TFT_API_KEY .env | cut -d= -f2)"
# Should return: {"status": "ok"}
```

**Solution:**
```bash
# Generate API key if missing
python generate_api_key.py

# Run adapter from project root directory
cd /path/to/MonitoringPrediction
python adapters/mongodb_adapter.py --daemon

# Or set environment variable explicitly
export TFT_API_KEY=$(grep TFT_API_KEY .env | cut -d= -f2)
python adapters/mongodb_adapter.py --daemon
```

**Note:** Adapters must be run from project root directory OR have access to `.env` file in parent directory.

---

### **Problem 5: "Process crashes after system restart"**

**Symptoms:**
- One or more processes not running after reboot

**Diagnosis:**
```bash
# Check systemd service status
sudo systemctl status tft-inference-daemon
sudo systemctl status tft-mongodb-adapter

# Check logs
journalctl -u tft-inference-daemon -n 50
```

**Solution:**
```bash
# Ensure services are enabled
sudo systemctl enable tft-inference-daemon
sudo systemctl enable tft-mongodb-adapter
sudo systemctl enable tft-dashboard

# Check dependencies are correct
# Adapter should have: After=tft-inference-daemon.service
```

---

## ❓ FAQ

### **Q: Can I run multiple adapters at the same time?**

**A:** Yes! You can run multiple adapters simultaneously:

```bash
# Terminal 1: MongoDB adapter
python adapters/mongodb_adapter.py --daemon

# Terminal 2: Elasticsearch adapter
python adapters/elasticsearch_adapter.py --daemon
```

Both will POST to the same `/feed` endpoint. This is useful if you have metrics in multiple data sources.

---

### **Q: What happens if the adapter crashes?**

**A:** Inference daemon continues working:

- ✅ Inference daemon keeps serving predictions (using cached data)
- ✅ Dashboard continues displaying last known predictions
- ⚠️ No new data arrives until adapter restarts
- ✅ When adapter restarts, it fetches missed data (state tracked)

**Use systemd/Docker restart policies to auto-restart crashed adapters.**

---

### **Q: What happens if inference daemon crashes?**

**A:** Adapter and dashboard lose connectivity:

- ❌ Adapter gets "Connection refused" errors (retries automatically)
- ❌ Dashboard shows "Daemon not connected"
- ✅ When daemon restarts, both reconnect automatically
- ⚠️ Warmup period required (288 data points per server)

**Critical: Inference daemon is the core component.**

---

### **Q: Can I change the fetch interval while running?**

**A:** No, restart required:

```bash
# Stop adapter
Ctrl+C  # or: sudo systemctl stop tft-mongodb-adapter

# Restart with new interval
python adapters/mongodb_adapter.py --daemon --interval 10
# or: sudo systemctl start tft-mongodb-adapter
```

---

### **Q: How do I know which data source is feeding the daemon?**

**A:** Check inference daemon logs:

```bash
# With metrics generator (simulated)
[FEED] Received 20 metrics from 127.0.0.1
[FEED] Server names: ppml0001, ppml0002...

# With MongoDB adapter (production)
[FEED] Received 47 metrics from 127.0.0.1
[FEED] Server names: prod-ml-01, prod-db-03...
```

Or check adapter logs:
```bash
✅ Forwarded 47 records to TFT daemon
```

---

### **Q: Can adapter and inference daemon run on different machines?**

**A:** Yes! Configure adapter to point to remote daemon:

```json
{
  "tft_daemon": {
    "url": "http://inference-server.example.com:8000",
    "api_key": "your-api-key"
  }
}
```

**Requirements:**
- Network connectivity between machines
- Firewall allows port 8000
- API key authentication configured

---

### **Q: Do I need both adapter AND metrics generator?**

**A:** No, choose one:

| Scenario | Use This |
|----------|----------|
| **Development/Testing** | `metrics_generator_daemon.py` (simulated data) |
| **Production** | `mongodb_adapter.py` or `elasticsearch_adapter.py` (real data) |
| **Hybrid** | Run both (mix simulated + real data) |

**Same `/feed` endpoint - daemon doesn't care about data source.**

---

### **Q: How much data does the adapter send per interval?**

**A:** Depends on your fleet size:

```
Servers × Metrics × Interval = Data Rate

Example:
- 50 servers
- 14 LINBORG metrics each
- 5 second interval
- ~700 metrics every 5 seconds
- ~140 metrics/second
- ~350 KB/request (JSON)
```

**Performance:** Adapters easily handle 100+ servers at 5-second intervals.

---

## 📚 Related Documentation

- **[PRODUCTION_DATA_ADAPTERS.md](PRODUCTION_DATA_ADAPTERS.md)** - Quick reference
- **[adapters/README.md](../adapters/README.md)** - Comprehensive adapter guide
- **[API_KEY_SETUP.md](API_KEY_SETUP.md)** - Security configuration
- **[LINBORG_METRICS.md](LINBORG_METRICS.md)** - Metric definitions

---

## ✅ Summary

### **Key Takeaways**

1. ✅ **Adapters run as separate daemon processes**
2. ✅ **Adapters actively PUSH data to inference daemon**
3. ✅ **Inference daemon does NOT call adapters**
4. ✅ **Three independent processes communicate via HTTP**
5. ✅ **Start order matters: Daemon FIRST, then adapter, then dashboard**
6. ✅ **Each component can restart independently**
7. ✅ **This is a microservices architecture**

### **Architecture Pattern**

```
Data Source → Adapter (fetcher) → Inference Daemon (processor) → Dashboard (viewer)
  (passive)     (active)              (server)                      (client)
```

**Remember:** Adapters are the "active" component that drives the data flow!

---

**Document Version:** 1.0.0
**Last Updated:** 2025-10-17
**Status:** ✅ Production Critical Documentation
