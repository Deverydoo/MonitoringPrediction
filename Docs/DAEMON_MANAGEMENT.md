# NordIQ Daemon Management Guide

**Version:** 1.0.0
**Last Updated:** October 29, 2025
**Purpose:** Complete guide to managing NordIQ services (inference, metrics, dashboard)

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Command Reference](#command-reference)
4. [Service Details](#service-details)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)
7. [Production Deployment](#production-deployment)
8. [Advanced Configuration](#advanced-configuration)

---

## Overview

NordIQ consists of 3 independent services:

| Service | Port | Purpose | Required |
|---------|------|---------|----------|
| **Inference Daemon** | 8000 | AI predictions, REST API | ✅ Yes |
| **Metrics Generator** | 8001 | Demo data generation | For demos only |
| **Dashboard** | 8501 | Web UI (Streamlit) | Optional |

**Two Management Scripts:**
- `daemon.bat` / `daemon.sh` - Individual service control (NEW!)
- `start_all.bat` / `start_all.sh` - Start all services at once

---

## Quick Start

### Start All Services

**Windows:**
```bash
cd NordIQ
daemon.bat start
```

**Linux/Mac:**
```bash
cd NordIQ
./daemon.sh start
```

### Check Status

```bash
# Windows
daemon.bat status

# Linux/Mac
./daemon.sh status
```

### Stop All Services

```bash
# Windows
daemon.bat stop

# Linux/Mac
./daemon.sh stop
```

---

## Command Reference

### Daemon Management Script

**Syntax:**
```bash
daemon.bat <command> [service]     # Windows
./daemon.sh <command> [service]    # Linux/Mac
```

### Commands

#### start
Start one or more services

```bash
# Start all services
daemon.bat start
daemon.bat start all

# Start individual services
daemon.bat start inference    # Inference daemon only
daemon.bat start metrics      # Metrics generator only
daemon.bat start dashboard    # Dashboard only
```

**Example Output:**
```
============================================
Starting All NordIQ Services
============================================

[OK] Conda environment: py310
[INFO] Checking API key...
[OK] API key loaded
[INFO] Starting Inference Daemon...
[OK] Inference Daemon started (port 8000)
[INFO] Starting Metrics Generator...
[OK] Metrics Generator started
[INFO] Starting Dashboard...
[OK] Dashboard started (port 8501)

============================================
All Services Started!
============================================

Access Points:
  Inference API:  http://localhost:8000
  API Docs:       http://localhost:8000/docs
  Dashboard:      http://localhost:8501
```

---

#### stop
Stop one or more services

```bash
# Stop all services
daemon.bat stop
daemon.bat stop all

# Stop individual services
daemon.bat stop inference
daemon.bat stop metrics
daemon.bat stop dashboard
```

**Example Output:**
```
============================================
Stopping All NordIQ Services
============================================

[INFO] Stopping Dashboard...
[OK] Dashboard stopped
[INFO] Stopping Metrics Generator...
[OK] Metrics Generator stopped
[INFO] Stopping Inference Daemon...
[OK] Inference Daemon stopped

[OK] All services stopped
```

---

#### restart
Restart one or more services

```bash
# Restart all services
daemon.bat restart

# Restart individual services
daemon.bat restart inference
daemon.bat restart metrics
daemon.bat restart dashboard
```

**Use Cases:**
- After code changes
- After configuration updates
- If service becomes unresponsive

---

#### status
Check service status

```bash
daemon.bat status
./daemon.sh status
```

**Example Output:**
```
============================================
NordIQ Services Status
============================================

Checking Inference Daemon (port 8000)...
  [UP]   Inference Daemon - http://localhost:8000 (PID: 12345)
Checking Dashboard (port 8501)...
  [UP]   Dashboard - http://localhost:8501 (PID: 12346)
Checking Metrics Generator...
  [UP]   Metrics Generator (PID: 12347)
```

**Status Indicators:**
- `[UP]` - Service running and responding
- `[DOWN]` - Service not running
- `[WARN]` - Process exists but not responding (Linux/Mac only)

---

## Service Details

### 1. Inference Daemon

**Purpose:** AI prediction engine with REST API

**Port:** 8000

**Endpoints:**
- `/health` - Health check
- `/status` - Detailed status
- `/predictions/current` - Get predictions
- `/alerts/active` - Get alerts
- `/explain/{server}` - XAI explanations
- `/docs` - Interactive API documentation

**Logs:**
- Windows: Console window titled "TFT Inference Daemon"
- Linux/Mac: `logs/inference.log`

**Requirements:**
- Trained TFT model in `models/` directory
- API key in `.env` file
- Port 8000 available

**Start Individually:**
```bash
# Windows
daemon.bat start inference

# Linux/Mac
./daemon.sh start inference
```

**Direct Start (for debugging):**
```bash
cd NordIQ
conda activate py310
python src/daemons/tft_inference_daemon.py
```

---

### 2. Metrics Generator

**Purpose:** Generate realistic demo metrics for testing

**Port:** None (streams to inference daemon)

**Modes:**
- `--stream` - Continuous streaming (demo mode)
- `--hours N` - Generate training data

**Scenarios:**
- `healthy` - Normal operations (0 P1 alerts)
- `degrading` - Some issues (~5 P2 alerts)
- `critical` - Major problems (8-10 P1 alerts)

**Logs:**
- Windows: Console window titled "Metrics Generator"
- Linux/Mac: `logs/metrics.log`

**Start Individually:**
```bash
# Windows
daemon.bat start metrics

# Linux/Mac
./daemon.sh start metrics
```

**Direct Start (for debugging):**
```bash
cd NordIQ
conda activate py310
python src/daemons/metrics_generator_daemon.py --stream --servers 20
```

**Change Scenario (while running):**
Type in console window:
- `healthy` + Enter
- `degrading` + Enter
- `critical` + Enter

---

### 3. Dashboard

**Purpose:** Web-based monitoring interface

**Port:** 8501

**Features:**
- Real-time server monitoring
- Risk score visualization
- Historical trends
- Alert management
- XAI explanations

**Logs:**
- Windows: Console window titled "NordIQ Dashboard"
- Linux/Mac: `logs/dashboard.log`

**Requirements:**
- Inference daemon running (port 8000)
- Port 8501 available

**Start Individually:**
```bash
# Windows
daemon.bat start dashboard

# Linux/Mac
./daemon.sh start dashboard
```

**Direct Start (for debugging):**
```bash
cd NordIQ
conda activate py310
streamlit run src/dashboard/tft_dashboard_web.py
```

---

## Common Workflows

### Demo Workflow

**Scenario:** Present system to stakeholders

```bash
# 1. Start all services
daemon.bat start

# 2. Wait 30 seconds for services to initialize

# 3. Open dashboard
# Browser: http://localhost:8501

# 4. Switch scenarios
# In Metrics Generator window, type:
healthy     # Show normal operations
degrading   # Show some issues
critical    # Show major problems

# 5. Stop when done
daemon.bat stop
```

---

### Development Workflow

**Scenario:** Develop and test code changes

```bash
# 1. Start only what you need
daemon.bat start inference    # If testing API
daemon.bat start dashboard    # If testing UI

# 2. Make code changes

# 3. Restart affected service
daemon.bat restart inference  # After daemon changes
daemon.bat restart dashboard  # After UI changes

# 4. Test changes

# 5. Stop when done
daemon.bat stop
```

---

### Production Workflow

**Scenario:** Run in production (without metrics generator)

```bash
# 1. Start inference daemon only
daemon.bat start inference

# 2. Connect production metrics forwarder
# (See INTEGRATION_GUIDE.md)

# 3. Optionally start dashboard
daemon.bat start dashboard

# 4. Monitor with status checks
daemon.bat status

# 5. Logs are in logs/ directory (Linux/Mac)
tail -f logs/inference.log
tail -f logs/dashboard.log
```

---

### Troubleshooting Workflow

**Scenario:** Service not responding

```bash
# 1. Check status
daemon.bat status

# 2. If service is UP but not responding
daemon.bat restart inference    # Restart problematic service

# 3. If service won't start
daemon.bat stop                 # Stop everything
# Check logs in console windows (Windows)
# Check logs/*.log files (Linux/Mac)
# Fix issue
daemon.bat start               # Start again

# 4. If ports are in use
# Windows: taskkill /F /IM python.exe
# Linux/Mac: killall python
daemon.bat start
```

---

## Troubleshooting

### Issue: "Conda environment 'py310' not found"

**Symptom:**
```
[ERROR] Conda environment 'py310' not found
```

**Solution:**
```bash
# Create environment
conda create -n py310 python=3.10

# Install dependencies
conda activate py310
pip install -r requirements.txt
```

---

### Issue: "No trained models found"

**Symptom:**
```
[WARN] No trained models found
```

**Solution:**
```bash
# Train a model
cd NordIQ
python src/training/main.py train --epochs 5

# Or copy existing model to models/ directory
```

---

### Issue: "Port already in use"

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**

**Windows:**
```bash
# Find process using port
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# Kill process (replace PID)
taskkill /F /PID <PID>
```

**Linux/Mac:**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9
```

---

### Issue: Service started but not responding

**Symptom:**
```
[WARN] Inference Daemon - process running but not responding
```

**Solution:**
```bash
# 1. Check logs
# Windows: Look at console window
# Linux/Mac: tail -f logs/inference.log

# 2. Look for errors in logs

# 3. Restart service
daemon.bat restart inference

# 4. If still not working, check API key
python bin/generate_api_key.py --show
```

---

### Issue: "Failed to load API key"

**Symptom:**
```
[ERROR] Failed to load API key
```

**Solution:**
```bash
# Generate new API key
cd NordIQ
python bin/generate_api_key.py

# Verify .env file exists
cat .env  # Linux/Mac
type .env # Windows

# Should contain: TFT_API_KEY=...
```

---

### Issue: Dashboard shows "Connection refused"

**Symptom:** Dashboard displays connection error

**Solution:**
```bash
# 1. Check inference daemon is running
daemon.bat status

# 2. If down, start it
daemon.bat start inference

# 3. Wait 10 seconds for startup

# 4. Refresh dashboard
```

---

## Production Deployment

### Deployment Checklist

**Before deploying to production:**

- [ ] Trained model exists in `models/` directory
- [ ] API key configured in `.env` file
- [ ] Ports 8000 and 8501 are available
- [ ] Firewall rules configured (if needed)
- [ ] SSL/TLS configured for external access
- [ ] Log rotation configured (Linux/Mac)
- [ ] Monitoring/alerting configured
- [ ] Backup strategy defined

---

### systemd Service (Linux)

Create `/etc/systemd/system/nordiq-inference.service`:

```ini
[Unit]
Description=NordIQ Inference Daemon
After=network.target

[Service]
Type=simple
User=nordiq
WorkingDirectory=/opt/nordiq
Environment="TFT_API_KEY=your-key-here"
ExecStart=/opt/conda/envs/py310/bin/python src/daemons/tft_inference_daemon.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable nordiq-inference
sudo systemctl start nordiq-inference
sudo systemctl status nordiq-inference
```

---

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY NordIQ/ .

# Set API key via environment
ENV TFT_API_KEY=""

# Expose ports
EXPOSE 8000 8501

# Start services
CMD ["python", "src/daemons/tft_inference_daemon.py"]
```

**Build and run:**
```bash
docker build -t nordiq-inference .
docker run -d -p 8000:8000 -e TFT_API_KEY="your-key" nordiq-inference
```

---

### Nginx Reverse Proxy

**For external access with SSL:**

```nginx
server {
    listen 443 ssl;
    server_name nordiq.example.com;

    ssl_certificate /etc/ssl/certs/nordiq.crt;
    ssl_certificate_key /etc/ssl/private/nordiq.key;

    # Inference API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Dashboard
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

### Log Rotation (Linux)

Create `/etc/logrotate.d/nordiq`:

```
/opt/nordiq/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 nordiq nordiq
}
```

---

## Advanced Configuration

### Environment Variables

Configure services via environment variables:

```bash
# API Configuration
export TFT_API_KEY="your-api-key-here"
export INFERENCE_PORT=8000
export DASHBOARD_PORT=8501

# Model Configuration
export MODEL_PATH="/path/to/model"
export PREDICTION_HORIZON=96

# Performance Tuning
export BATCH_SIZE=32
export NUM_WORKERS=4
```

---

### Custom Startup Options

**Inference Daemon:**
```bash
# Custom port
python src/daemons/tft_inference_daemon.py --port 9000

# Custom model
python src/daemons/tft_inference_daemon.py --model-dir ./models/custom_model

# Debug mode
python src/daemons/tft_inference_daemon.py --debug
```

**Metrics Generator:**
```bash
# More servers
python src/daemons/metrics_generator_daemon.py --stream --servers 50

# Faster updates (every 2 seconds)
python src/daemons/metrics_generator_daemon.py --stream --interval 2

# Start in specific scenario
python src/daemons/metrics_generator_daemon.py --stream --scenario critical
```

**Dashboard:**
```bash
# Custom port
streamlit run src/dashboard/tft_dashboard_web.py --server.port 9501

# Disable auto-reload (production)
streamlit run src/dashboard/tft_dashboard_web.py --server.fileWatcherType none

# Custom theme
streamlit run src/dashboard/tft_dashboard_web.py --theme.base dark
```

---

### Process Management (Linux/Mac)

**Using screen (for persistent sessions):**
```bash
# Start inference in screen
screen -S nordiq-inference
python src/daemons/tft_inference_daemon.py
# Ctrl+A, D to detach

# Reattach
screen -r nordiq-inference

# List screens
screen -ls
```

**Using tmux:**
```bash
# Start session
tmux new -s nordiq

# Split panes
Ctrl+B %  # Vertical split
Ctrl+B "  # Horizontal split

# Start services in different panes
# Pane 1: python src/daemons/tft_inference_daemon.py
# Pane 2: python src/daemons/metrics_generator_daemon.py
# Pane 3: streamlit run src/dashboard/tft_dashboard_web.py

# Detach: Ctrl+B D
# Reattach: tmux attach -t nordiq
```

---

## Monitoring & Health Checks

### Health Check Script

Create `check_health.sh`:

```bash
#!/bin/bash

# Check inference daemon
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Inference Daemon: OK"
else
    echo "✗ Inference Daemon: DOWN"
    exit 1
fi

# Check dashboard
if curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "✓ Dashboard: OK"
else
    echo "✗ Dashboard: DOWN"
    exit 1
fi

echo "All services healthy"
```

**Use with cron (every 5 minutes):**
```bash
*/5 * * * * /opt/nordiq/check_health.sh || /opt/nordiq/daemon.sh restart
```

---

### Performance Monitoring

**CPU/Memory usage:**
```bash
# Linux
top -p $(cat .pids/*.pid)

# Windows
tasklist | findstr python
```

**Request latency:**
```bash
# Test API response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/predictions/current
```

**curl-format.txt:**
```
time_namelookup:  %{time_namelookup}\n
time_connect:     %{time_connect}\n
time_starttransfer: %{time_starttransfer}\n
time_total:       %{time_total}\n
```

---

## Best Practices

### Development
- ✅ Use `daemon.sh start inference` for API development
- ✅ Use `daemon.sh restart` after code changes
- ✅ Check logs immediately after restart
- ✅ Use `--debug` flag for verbose logging

### Production
- ✅ Use systemd/Docker for automatic restart
- ✅ Configure log rotation
- ✅ Set up health check monitoring
- ✅ Use reverse proxy (nginx) for SSL
- ✅ Run services as non-root user
- ✅ Backup models and configuration regularly

### Security
- ✅ Keep API keys in `.env` (never commit)
- ✅ Use SSL/TLS for external access
- ✅ Restrict port access via firewall
- ✅ Rotate API keys periodically
- ✅ Monitor logs for unauthorized access

---

## Support & Resources

### Documentation
- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - API integration
- [INTEGRATION_QUICKSTART.md](INTEGRATION_QUICKSTART.md) - 5-minute guide
- [API_KEY_SETUP.md](API_KEY_SETUP.md) - Authentication

### Command Summary

| Command | Purpose |
|---------|---------|
| `daemon.bat start` | Start all services |
| `daemon.bat start inference` | Start inference only |
| `daemon.bat stop` | Stop all services |
| `daemon.bat restart` | Restart all services |
| `daemon.bat status` | Check service status |

### Quick Links
- **API Docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501
- **Health Check:** http://localhost:8000/health

---

## Appendix: Service Architecture

```
┌─────────────────────────────────────────────────────┐
│                  NordIQ System                      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────┐   ┌──────────────────┐      │
│  │   Inference      │◄──│   Metrics        │      │
│  │   Daemon         │   │   Generator      │      │
│  │   (Port 8000)    │   │   (Demo)         │      │
│  └────────┬─────────┘   └──────────────────┘      │
│           │                                         │
│           │ REST API                                │
│           │                                         │
│  ┌────────▼─────────┐                              │
│  │   Dashboard      │                              │
│  │   (Port 8501)    │                              │
│  └──────────────────┘                              │
│                                                     │
└─────────────────────────────────────────────────────┘

Production Deployment (no metrics generator):

┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌──────────────────┐                              │
│  │   Production     │──►  ┌──────────────────┐    │
│  │   Metrics        │     │   Inference      │    │
│  │   Forwarder      │     │   Daemon         │    │
│  │                  │     │   (Port 8000)    │    │
│  └──────────────────┘     └────────┬─────────┘    │
│                                     │              │
│                            REST API │              │
│                                     │              │
│                            ┌────────▼─────────┐   │
│                            │   Dashboard      │   │
│                            │   or Custom UI   │   │
│                            └──────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0.0
**Last Updated:** October 29, 2025
**Company:** NordIQ AI, LLC
**License:** Business Source License 1.1
