# Production Deployment Guide - Silent Daemon Mode

## Overview

The TFT Monitoring System includes **silent daemon mode** for production deployment via:
- **init.d** scripts (SysV init systems)
- **systemd** service files (modern Linux distributions)
- **cron** jobs (scheduled starts)

All scripts are **completely silent** by default - no output to stdout, all logs written to `logs/` directory.

---

## Silent Mode Usage

### Basic Commands

```bash
# Start all services (silent, daemon mode)
./start_all_corp.sh

# Stop all services (silent)
./stop_all.sh

# Check exit code
./start_all_corp.sh
echo $?  # 0 = success, 1 = pre-flight failed, 2 = service start failed

# View logs
tail -f logs/startup.log
tail -f logs/shutdown.log
tail -f logs/*.log
```

### Verbose Mode (Debugging)

```bash
# Start with output to stdout (for debugging)
./start_all_corp.sh verbose

# Stop with output to stdout
./stop_all.sh verbose

# Alternative syntax
./start_all_corp.sh -v
./stop_all.sh --verbose
```

### Exit Codes

**start_all_corp.sh**:
- `0` = Success (all services started)
- `1` = Pre-flight check failed (Python, packages, etc.)
- `2` = Service start failed (process died)

**stop_all.sh**:
- `0` = All services stopped successfully
- `1` = Some services were not running or failed to stop

---

## Log Files

All operations logged to `logs/` directory:

| Log File | Purpose |
|----------|---------|
| `startup.log` | Startup script activity |
| `shutdown.log` | Shutdown script activity |
| `metrics_generator.log` | Metrics Generator daemon output |
| `inference_daemon.log` | Inference Daemon output |
| `dashboard.log` | Streamlit Dashboard output |

**PID Files**:
- `logs/metrics_generator.pid`
- `logs/inference_daemon.pid`
- `logs/dashboard.pid`

---

## Systemd Deployment (Recommended)

### Installation

1. **Copy service files**:
   ```bash
   sudo cp systemd/*.service /etc/systemd/system/
   ```

2. **Update paths** in service files:
   ```bash
   # Edit each service file
   sudo nano /etc/systemd/system/tft-monitoring.service

   # Change WorkingDirectory to your installation path
   WorkingDirectory=/opt/tft-monitoring
   ```

3. **Create service user**:
   ```bash
   sudo useradd -r -s /bin/bash -d /opt/tft-monitoring tftuser
   sudo chown -R tftuser:tftuser /opt/tft-monitoring
   ```

4. **Create log directory**:
   ```bash
   sudo mkdir -p /var/log/tft-monitoring
   sudo chown tftuser:tftuser /var/log/tft-monitoring
   ```

5. **Reload systemd**:
   ```bash
   sudo systemctl daemon-reload
   ```

### Service Files Included

**Option 1: All-in-One Service**
```bash
# Single service that manages all three components
sudo systemctl enable tft-monitoring.service
sudo systemctl start tft-monitoring.service
sudo systemctl status tft-monitoring.service
```

**Option 2: Individual Services** (better control)
```bash
# Enable all services
sudo systemctl enable tft-metrics-generator.service
sudo systemctl enable tft-inference-daemon.service
sudo systemctl enable tft-dashboard.service

# Start all services (automatic ordering via dependencies)
sudo systemctl start tft-dashboard.service

# Status check
sudo systemctl status tft-*.service
```

### Service Management

```bash
# Start/Stop/Restart
sudo systemctl start tft-monitoring.service
sudo systemctl stop tft-monitoring.service
sudo systemctl restart tft-monitoring.service

# View logs (journalctl)
sudo journalctl -u tft-monitoring.service -f
sudo journalctl -u tft-monitoring.service --since "1 hour ago"

# Check if running
sudo systemctl is-active tft-monitoring.service
```

### Service Dependencies

The individual service files have proper dependencies:

```
tft-metrics-generator.service
    ↓ (Requires)
tft-inference-daemon.service
    ↓ (Requires)
tft-dashboard.service
```

Starting `tft-dashboard.service` automatically starts all dependencies.

---

## init.d Deployment (Legacy Systems)

### Installation

1. **Copy init.d script**:
   ```bash
   sudo cp init.d/tft-monitoring /etc/init.d/
   sudo chmod +x /etc/init.d/tft-monitoring
   ```

2. **Update paths** in script:
   ```bash
   sudo nano /etc/init.d/tft-monitoring

   # Change these variables:
   TFT_HOME="/opt/tft-monitoring"
   TFT_USER="tftuser"
   ```

3. **Register service**:
   ```bash
   # Debian/Ubuntu
   sudo update-rc.d tft-monitoring defaults

   # RedHat/CentOS
   sudo chkconfig --add tft-monitoring
   sudo chkconfig tft-monitoring on
   ```

### Service Management

```bash
# Start/Stop/Restart
sudo service tft-monitoring start
sudo service tft-monitoring stop
sudo service tft-monitoring restart
sudo service tft-monitoring status
```

---

## Cron Deployment

### Automatic Startup on Reboot

Add to root's crontab:

```bash
sudo crontab -e
```

Add this line:
```cron
@reboot cd /opt/tft-monitoring && ./start_all_corp.sh
```

### Scheduled Restart (e.g., Daily at 3 AM)

```cron
0 3 * * * cd /opt/tft-monitoring && ./stop_all.sh && sleep 5 && ./start_all_corp.sh
```

### Health Check Monitor (Every 5 Minutes)

Create monitoring script `cron_health_check.sh`:
```bash
#!/bin/bash
cd /opt/tft-monitoring

# Check if services are running
if ! kill -0 $(cat logs/metrics_generator.pid 2>/dev/null) 2>/dev/null; then
    echo "$(date) - Metrics Generator died, restarting..." >> logs/health_check.log
    ./start_all_corp.sh
fi

if ! kill -0 $(cat logs/inference_daemon.pid 2>/dev/null) 2>/dev/null; then
    echo "$(date) - Inference Daemon died, restarting..." >> logs/health_check.log
    ./start_all_corp.sh
fi

if ! kill -0 $(cat logs/dashboard.pid 2>/dev/null) 2>/dev/null; then
    echo "$(date) - Dashboard died, restarting..." >> logs/health_check.log
    ./start_all_corp.sh
fi
```

Add to crontab:
```cron
*/5 * * * * /opt/tft-monitoring/cron_health_check.sh
```

---

## Production Configuration Checklist

### Security

- [ ] Create dedicated service user (`tftuser`)
- [ ] Set proper file permissions:
  ```bash
  sudo chown -R tftuser:tftuser /opt/tft-monitoring
  sudo chmod 750 /opt/tft-monitoring
  sudo chmod 640 /opt/tft-monitoring/*.sh
  sudo chmod +x /opt/tft-monitoring/*.sh
  ```
- [ ] Enable systemd security features (already in service files):
  - `PrivateTmp=yes`
  - `NoNewPrivileges=yes`
  - `ProtectSystem=strict`
  - `ProtectHome=yes`

### Networking

- [ ] Configure firewall:
  ```bash
  # Open ports (adjust for your firewall)
  sudo ufw allow 8000/tcp  # Inference Daemon
  sudo ufw allow 8001/tcp  # Metrics Generator
  sudo ufw allow 8501/tcp  # Dashboard
  ```

- [ ] Set up reverse proxy (nginx/apache) for HTTPS:
  ```bash
  # See STARTUP_GUIDE_CORPORATE.md for nginx config
  ```

### Monitoring

- [ ] Set up log rotation:
  ```bash
  sudo nano /etc/logrotate.d/tft-monitoring
  ```

  Add:
  ```
  /opt/tft-monitoring/logs/*.log {
      daily
      rotate 30
      compress
      delaycompress
      notifempty
      missingok
      create 0640 tftuser tftuser
  }
  ```

- [ ] Configure monitoring alerts:
  ```bash
  # Add to monitoring system (Nagios/Zabbix/etc.)
  check_http -H localhost -p 8000 -u /health
  check_http -H localhost -p 8001 -u /health
  ```

### Backups

- [ ] Back up configuration:
  ```bash
  tar -czf tft-config-backup.tar.gz config/ .streamlit/
  ```

- [ ] Back up trained models:
  ```bash
  tar -czf tft-models-backup.tar.gz models/
  ```

### Resource Limits

- [ ] Verify systemd resource limits (already set):
  - Metrics Generator: 1GB memory, 4096 file descriptors
  - Inference Daemon: 4GB memory, 4096 file descriptors
  - Dashboard: 2GB memory, 8192 file descriptors

- [ ] Adjust if needed:
  ```bash
  sudo systemctl edit tft-inference-daemon.service
  ```

  Add:
  ```ini
  [Service]
  MemoryLimit=8G
  ```

---

## Verification

### After Installation

```bash
# Check all services started
ps aux | grep -E "(metrics_generator|tft_inference|streamlit)"

# Check PIDs
cat logs/*.pid

# Check logs
tail -n 50 logs/startup.log

# Verify ports
netstat -tlnp | grep -E ":(8000|8001|8501)"

# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health

# Dashboard access
curl -I http://localhost:8501
```

### Expected Output

**Successful startup**:
```
$ ./start_all_corp.sh
$ echo $?
0

$ tail -5 logs/startup.log
2025-10-16 14:30:05 - All services started successfully
2025-10-16 14:30:05 - Metrics Generator: PID 12345 (port 8001)
2025-10-16 14:30:05 - Inference Daemon:  PID 12346 (port 8000)
2025-10-16 14:30:05 - Dashboard:         PID 12347 (port 8501)
```

**Health checks**:
```
$ curl http://localhost:8000/health
{"status":"healthy"}

$ curl http://localhost:8001/health
{"status":"healthy"}
```

---

## Troubleshooting

### Services Won't Start

**Check logs**:
```bash
tail -100 logs/startup.log
tail -100 logs/metrics_generator.log
tail -100 logs/inference_daemon.log
tail -100 logs/dashboard.log
```

**Run in verbose mode**:
```bash
./start_all_corp.sh verbose
```

**Check pre-flight failures**:
```bash
# Verify Python
python3 --version

# Verify packages
python3 -c "import torch, pytorch_forecasting, fastapi, streamlit"

# Check ports
netstat -tlnp | grep -E ":(8000|8001|8501)"
```

### Services Die After Start

**Check service logs**:
```bash
# Look for errors after startup
tail -100 logs/inference_daemon.log
```

**Check systemd journal**:
```bash
sudo journalctl -u tft-inference-daemon.service --since "1 hour ago"
```

**Common issues**:
- **Out of memory**: Increase `MemoryLimit` in service file
- **Model not found**: Train model with `python3 tft_trainer.py`
- **Port in use**: Stop conflicting process
- **Permission denied**: Check file ownership (`chown tftuser:tftuser`)

### Silent Mode Issues

**No output visible**:
- ✅ This is expected! Check `logs/startup.log` instead
- Use verbose mode for debugging: `./start_all_corp.sh verbose`

**Logs not created**:
- Check directory permissions: `ls -la logs/`
- Create manually: `mkdir -p logs && chmod 755 logs`

---

## Performance Tuning

### Startup Times

Current timings (can be adjusted in `start_all_corp.sh`):

```bash
# Line 141: Wait after metrics generator
sleep 2  # Reduce to 1 for faster systems

# Line 158: Wait after inference daemon
sleep 4  # Reduce to 2 if model loads quickly

# Line 175: Wait after dashboard
sleep 2  # Reduce to 1 for faster systems
```

### Resource Allocation

**GPU Systems**:
```bash
# Add to systemd service file
Environment="CUDA_VISIBLE_DEVICES=0"

# Or set in /etc/environment
CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

**Multi-Core Systems**:
```bash
# Inference daemon uses all cores by default
# To limit, add to systemd service:
CPUQuota=50%  # Use only 50% of CPU
```

---

## Migration from Windows

If you developed on Windows and are deploying to Linux:

1. **Convert line endings**:
   ```bash
   dos2unix start_all_corp.sh stop_all.sh
   ```

2. **Make executable**:
   ```bash
   chmod +x start_all_corp.sh stop_all.sh
   ```

3. **Test startup**:
   ```bash
   ./start_all_corp.sh verbose
   ```

4. **Deploy to systemd** (recommended for production)

---

## Summary

### Silent Mode (Production)

```bash
# Start (no output)
./start_all_corp.sh

# Check status
echo $?  # 0 = success

# View logs
tail -f logs/startup.log
tail -f logs/*.log

# Stop (no output)
./stop_all.sh
```

### Verbose Mode (Debugging)

```bash
# Start with output
./start_all_corp.sh verbose

# Stop with output
./stop_all.sh verbose
```

### Systemd (Recommended)

```bash
# One-time setup
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tft-monitoring.service

# Daily usage
sudo systemctl start tft-monitoring.service
sudo systemctl status tft-monitoring.service
sudo journalctl -u tft-monitoring.service -f
```

### init.d (Legacy)

```bash
# One-time setup
sudo cp init.d/tft-monitoring /etc/init.d/
sudo chmod +x /etc/init.d/tft-monitoring
sudo update-rc.d tft-monitoring defaults

# Daily usage
sudo service tft-monitoring start
sudo service tft-monitoring status
```

---

## Related Documentation

- [STARTUP_GUIDE_CORPORATE.md](STARTUP_GUIDE_CORPORATE.md) - General startup guide
- [CORPORATE_BROWSER_FIX.md](CORPORATE_BROWSER_FIX.md) - Browser compatibility
- [CONFIG_GUIDE.md](CONFIG_GUIDE.md) - Configuration reference

---

**Production-ready!** Your TFT Monitoring System can now run as a silent daemon via init.d, systemd, or cron. 🚀
