# NordIQ Daemon Quick Reference

**One-page cheat sheet for managing NordIQ services**

---

## Quick Commands

### Start Services

```bash
# Windows
daemon.bat start              # All services
daemon.bat start inference    # Inference only
daemon.bat start metrics      # Metrics only
daemon.bat start dashboard    # Dashboard only

# Linux/Mac
./daemon.sh start              # All services
./daemon.sh start inference    # Inference only
./daemon.sh start metrics      # Metrics only
./daemon.sh start dashboard    # Dashboard only
```

### Stop Services

```bash
# Windows
daemon.bat stop              # All services
daemon.bat stop inference    # Inference only
daemon.bat stop metrics      # Metrics only
daemon.bat stop dashboard    # Dashboard only

# Linux/Mac
./daemon.sh stop              # All services
./daemon.sh stop inference    # Inference only
./daemon.sh stop metrics      # Metrics only
./daemon.sh stop dashboard    # Dashboard only
```

### Restart Services

```bash
# Windows
daemon.bat restart              # All services
daemon.bat restart inference    # Inference only
daemon.bat restart metrics      # Metrics only
daemon.bat restart dashboard    # Dashboard only

# Linux/Mac
./daemon.sh restart              # All services
./daemon.sh restart inference    # Inference only
./daemon.sh restart metrics      # Metrics only
./daemon.sh restart dashboard    # Dashboard only
```

### Check Status

```bash
# Windows
daemon.bat status

# Linux/Mac
./daemon.sh status
```

---

## Service Details

| Service | Port | Purpose | Required |
|---------|------|---------|----------|
| **Inference** | 8000 | AI predictions, REST API | ✅ Yes |
| **Metrics** | - | Demo data generation | Demo only |
| **Dashboard** | 8501 | Web UI | Optional |

---

## Access Points

- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

---

## Common Workflows

### Demo (All Services)
```bash
daemon.bat start       # Start everything
# Open: http://localhost:8501
daemon.bat stop        # Stop when done
```

### Development (API Only)
```bash
daemon.bat start inference    # Just the API
# Test API changes
daemon.bat restart inference  # After code changes
daemon.bat stop inference     # Stop when done
```

### Production (No Metrics)
```bash
daemon.bat start inference    # API only
daemon.bat start dashboard    # Optional UI
# Production metrics feed via API
daemon.bat stop               # Stop when done
```

---

## Troubleshooting

### Service Won't Start
```bash
daemon.bat status    # Check what's running
daemon.bat stop      # Stop everything
# Fix issue
daemon.bat start     # Try again
```

### Port In Use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Check Logs
```bash
# Windows: Check console windows
# Linux/Mac:
tail -f logs/inference.log
tail -f logs/dashboard.log
```

---

## Quick Tests

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Predictions
```bash
curl -H "X-API-Key: YOUR-KEY" http://localhost:8000/predictions/current
```

### Check API Key
```bash
python bin/generate_api_key.py --show
```

---

**Full Documentation:** [DAEMON_MANAGEMENT.md](../Docs/DAEMON_MANAGEMENT.md)
