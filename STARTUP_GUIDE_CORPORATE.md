# Corporate Environment Startup Guide

## Quick Start

### Windows
```bash
start_all_corp.bat
```

### Linux/Mac
```bash
chmod +x start_all_corp.sh stop_all.sh
./start_all_corp.sh
```

---

## What Gets Started

The all-in-one launcher starts **three services** in the correct order:

1. **Metrics Generator Daemon** (port 8001)
   - Generates realistic LINBORG server metrics
   - Provides three scenarios: healthy, degrading, critical
   - REST API for scenario control

2. **TFT Inference Daemon** (port 8000)
   - Loads trained TFT model
   - Generates predictions every 5 seconds
   - REST API + WebSocket streaming

3. **Streamlit Dashboard** (port 8501)
   - Web interface for monitoring and visualization
   - Corporate-optimized for strict browser security policies
   - Auto-opens in default browser

---

## Corporate Optimizations

The startup scripts include **corporate-friendly settings** to prevent browser freezing:

### Configuration Applied
- ✅ **WebSocket compression**: DISABLED (proxy-friendly)
- ✅ **Fast reruns**: DISABLED (reduces JavaScript overhead)
- ✅ **File watching**: DISABLED (production stability)
- ✅ **Auto-refresh buffer**: +1 second (prevents security timeout triggers)
- ✅ **Connection overlays**: MINIMAL (no gray "Connecting..." freeze)

### Pre-Flight Checks
The startup scripts verify before launching:
1. Python 3.8+ installed
2. Required packages (torch, pytorch-forecasting, fastapi, streamlit)
3. Trained model exists (warns if missing)
4. Corporate config exists (creates if missing)
5. Ports available (warns if in use)

---

## Service Ports

| Service | Port | Health Check URL |
|---------|------|------------------|
| Metrics Generator | 8001 | http://localhost:8001/health |
| Inference Daemon | 8000 | http://localhost:8000/health |
| Dashboard | 8501 | http://localhost:8501 |

---

## Windows-Specific Details

### How It Works
- Each service launches in a **separate CMD window**
- Window titles show service name and port
- Logs visible in real-time in each window
- Press **Ctrl+C** in any window to stop that service

### Stopping Services
```bash
stop_all.bat
```

Gracefully terminates all three services by port.

### Troubleshooting

**Port Already in Use?**
```bash
# Find what's using port 8000
netstat -ano | findstr ":8000"

# Kill process by PID (replace 1234 with actual PID)
taskkill /PID 1234 /F
```

**Python Not Found?**
- Add Python to PATH: `System Properties → Environment Variables → Path`
- Or use full path in script: `C:\Python38\python.exe`

---

## Linux/Mac-Specific Details

### How It Works
- Each service launches in **background** with output redirected
- Logs written to `logs/` directory
- PID files created: `logs/metrics_generator.pid`, `logs/inference_daemon.pid`, `logs/dashboard.pid`

### Viewing Logs
```bash
# Watch all logs live
tail -f logs/*.log

# Watch specific service
tail -f logs/dashboard.log
```

### Stopping Services
```bash
./stop_all.sh
```

Uses PID files for graceful shutdown, falls back to port-based killing.

### Troubleshooting

**Port Already in Use?**
```bash
# Find what's using port 8000
lsof -i :8000

# Kill process by PID
kill -9 <PID>
```

**Permission Denied?**
```bash
# Make scripts executable
chmod +x start_all_corp.sh stop_all.sh

# Then run
./start_all_corp.sh
```

---

## Expected Startup Times

| Service | Initialization Time |
|---------|---------------------|
| Metrics Generator | 3 seconds |
| Inference Daemon | 5 seconds (model loading) |
| Dashboard | 3 seconds (Streamlit startup) |
| **Total** | **~11 seconds** |

---

## Dashboard Usage

Once started, the dashboard opens at **http://localhost:8501**

### Scenario Control

In the sidebar, use **Interactive Demo Control** buttons:

- **🟢 Healthy**: All servers operating normally (0 P1 alerts)
- **🟡 Degrading**: 20% of fleet showing performance issues (2-5 P2 alerts)
- **🔴 Critical**: 50% of fleet in crisis (15-20 P1 alerts)

### Auto-Refresh

Default: **30 seconds** (configurable in sidebar)

**If dashboard freezes:**
1. **Disable auto-refresh** (uncheck in sidebar)
2. **Increase interval** to 60+ seconds
3. **Use manual refresh** button: 🔄 Refresh Now
4. See [CORPORATE_BROWSER_FIX.md](CORPORATE_BROWSER_FIX.md) for details

---

## Health Checks

Verify all services are running:

```bash
# Windows (PowerShell)
curl http://localhost:8001/health
curl http://localhost:8000/health

# Linux/Mac
curl http://localhost:8001/health
curl http://localhost:8000/health
```

Expected responses:
```json
{"status": "healthy"}
```

---

## Common Issues

### Issue 1: "Port already in use"
**Symptoms**: Services fail to start, port conflict errors

**Solution**:
```bash
# Windows
stop_all.bat

# Linux/Mac
./stop_all.sh
```

Then restart with `start_all_corp.bat` or `./start_all_corp.sh`

---

### Issue 2: "No trained model found"
**Symptoms**: Warning during startup, inference daemon may fail

**Solution**:
```bash
python tft_trainer.py
```

Wait for training to complete (1-5 hours depending on GPU), then restart services.

---

### Issue 3: Dashboard freezes for 30 seconds
**Symptoms**: Gray "Connecting..." banner, unresponsive UI

**Solution**: Already applied in corporate config! If still freezing:
1. Disable auto-refresh in sidebar
2. Use Microsoft Edge browser (best corporate compatibility)
3. Ask IT to whitelist `localhost:8501` in security policies
4. See [CORPORATE_BROWSER_FIX.md](CORPORATE_BROWSER_FIX.md)

---

### Issue 4: Missing packages
**Symptoms**: `ModuleNotFoundError` during startup

**Solution**:
```bash
pip install torch pytorch-forecasting fastapi uvicorn streamlit pandas plotly requests
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

---

### Issue 5: Services not stopping
**Symptoms**: `stop_all` script doesn't work, ports still in use

**Manual kill (Windows)**:
```bash
# Find PIDs
netstat -ano | findstr ":8000 :8001 :8501"

# Kill each PID
taskkill /PID <PID> /F
```

**Manual kill (Linux/Mac)**:
```bash
# Find PIDs
lsof -ti :8000 :8001 :8501

# Kill all
kill -9 $(lsof -ti :8000 :8001 :8501)
```

---

## Production Deployment Notes

For production use in corporate environment:

### 1. Disable Demo Mode
In sidebar, **uncheck** "Enable Demo Mode" - use real metrics instead.

### 2. Configure Real Metrics Source
Update `config/api_config.py` with your actual metrics endpoint:
```python
API_CONFIG = {
    'metrics_generator_url': 'http://your-real-metrics-server:8001',
    # ...
}
```

### 3. Deploy Behind Reverse Proxy
Corporate environments often require:
- **Nginx**: Acts as SSL terminator and corporate-friendly HTTP proxy
- **Apache**: Similar reverse proxy capability
- **Corporate app gateway**: Often whitelisted by security policies

Example Nginx config:
```nginx
server {
    listen 80;
    server_name tft-dashboard.corp.internal;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### 4. Run as System Service

**Windows (NSSM)**:
```bash
nssm install TFTMetricsGenerator python metrics_generator_daemon.py
nssm install TFTInferenceDaemon python tft_inference_daemon.py
nssm install TFTDashboard streamlit run tft_dashboard_web.py
```

**Linux (systemd)**:
```bash
# Create service files in /etc/systemd/system/
sudo systemctl enable tft-metrics tft-inference tft-dashboard
sudo systemctl start tft-metrics tft-inference tft-dashboard
```

---

## Related Documentation

- [CORPORATE_BROWSER_FIX.md](CORPORATE_BROWSER_FIX.md) - Browser compatibility troubleshooting
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration system reference
- [.streamlit/config.toml](.streamlit/config.toml) - Streamlit corporate settings

---

## Summary

### Windows Quick Reference
```bash
# Start all services
start_all_corp.bat

# Stop all services
stop_all.bat

# View service windows
# Check CMD windows with titles: "TFT Metrics Generator", "TFT Inference Daemon", "TFT Dashboard"
```

### Linux/Mac Quick Reference
```bash
# Start all services
./start_all_corp.sh

# View logs
tail -f logs/*.log

# Stop all services
./stop_all.sh

# Check status
lsof -i :8000 :8001 :8501
```

### Service URLs
- **Dashboard**: http://localhost:8501 (main interface)
- **Inference API**: http://localhost:8000 (predictions)
- **Metrics API**: http://localhost:8001 (data generation)

**Ready to launch!** 🚀
