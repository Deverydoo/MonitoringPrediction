# Secure Deployment Guide - TFT Monitoring System

**Version**: 2.0 (Post-Security Hardening)
**Date**: 2025-10-16
**Status**: ✅ Production Ready

---

## Overview

This guide covers deploying the TFT Monitoring System with all security improvements enabled:

1. **Inference Daemon** - API key authentication, CORS whitelist, Parquet persistence, rate limiting
2. **Dashboard** - API key client authentication
3. **Metrics Generator** - Data source (no authentication required for localhost)

---

## 🔐 Security Features

### Inference Daemon Security
- ✅ API Key authentication (`X-API-Key` header)
- ✅ CORS whitelist (configurable origins)
- ✅ Parquet persistence (no RCE vulnerability)
- ✅ Rate limiting (30-60 req/min)
- ✅ Resource limits (systemd hardening)

### Dashboard Security
- ✅ API key client support
- ✅ Streamlit secrets management
- ✅ Environment variable fallback
- ✅ No XSS vulnerabilities
- ✅ No hardcoded credentials

---

## 📋 Prerequisites

### 1. Install Security Dependencies

**On inference daemon server**:
```bash
# Install security libraries
./install_security_deps.sh  # Linux
# or
install_security_deps.bat   # Windows

# Verify installation
python3 -c "import slowapi, pyarrow; print('✅ Security dependencies OK')"
```

### 2. Generate API Key

**Generate a strong random key**:
```bash
# Linux/Mac
openssl rand -hex 32

# Windows (PowerShell)
[System.Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))

# Example output: a1b2c3d4e5f6...
```

**IMPORTANT**: Save this key securely - you'll need it for both daemon and dashboard!

---

## 🚀 Deployment Options

### Option 1: Development Mode (No Security)

**Use case**: Local testing, development

**Setup**:
```bash
# 1. Start daemon (no API key = development mode)
python tft_inference_daemon.py

# 2. Start dashboard (no API key = development mode)
streamlit run tft_dashboard_web.py

# 3. Start metrics generator
python metrics_generator_daemon.py
```

**Security**: ⚠️ None - suitable only for localhost testing

---

### Option 2: Environment Variables (Docker/systemd)

**Use case**: Production deployment, containerized environments

#### Inference Daemon Setup

```bash
# Set environment variables
export TFT_API_KEY="a1b2c3d4e5f6..."  # Your generated key
export CORS_ORIGINS="http://localhost:8501,https://dashboard.company.com"

# Start daemon
python tft_inference_daemon.py
```

#### Dashboard Setup

```bash
# Set same API key
export TFT_API_KEY="a1b2c3d4e5f6..."  # MUST match daemon key

# Start dashboard
streamlit run tft_dashboard_web.py
```

#### Metrics Generator Setup

```bash
# No API key needed (feeds data to daemon)
python metrics_generator_daemon.py
```

**Verification**:
```bash
# Test authentication works
curl -H "X-API-Key: a1b2c3d4e5f6..." http://localhost:8000/predictions/current

# Test authentication required (should fail with 403)
curl http://localhost:8000/predictions/current
```

---

### Option 3: Streamlit Secrets (Recommended for Production)

**Use case**: Production dashboards, secure credential management

#### Inference Daemon Setup

```bash
# Set API key via environment variable
export TFT_API_KEY="a1b2c3d4e5f6..."
export CORS_ORIGINS="http://localhost:8501"

# Start daemon
python tft_inference_daemon.py
```

#### Dashboard Setup

1. **Create secrets file**:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. **Edit `.streamlit/secrets.toml`**:
```toml
[daemon]
api_key = "a1b2c3d4e5f6..."  # MUST match daemon's TFT_API_KEY
url = "http://localhost:8000"
```

3. **Secure the secrets file**:
```bash
chmod 600 .streamlit/secrets.toml  # Only owner can read/write
echo ".streamlit/secrets.toml" >> .gitignore  # Never commit to git
```

4. **Start dashboard**:
```bash
streamlit run tft_dashboard_web.py
```

**Verification**:
```bash
# Dashboard should connect successfully and show predictions
# Check dashboard logs for:
# [INFO] No API key configured  ← BAD (means secrets not loaded)
# OR silence (good - means API key loaded from secrets)
```

---

### Option 4: Systemd Services (Linux Production)

**Use case**: Linux servers, production deployment with auto-start

#### 1. Install Services

```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/

# Edit inference daemon service to set API key
sudo nano /etc/systemd/system/tft-inference-daemon.service
```

#### 2. Configure API Key in Service File

Edit `/etc/systemd/system/tft-inference-daemon.service`:
```ini
[Service]
# Security environment variables (REQUIRED for production)
Environment="TFT_API_KEY=a1b2c3d4e5f6..."  # ← CHANGE THIS!
Environment="CORS_ORIGINS=http://localhost:8501"
```

#### 3. Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Start services
sudo systemctl start tft-metrics-generator
sudo systemctl start tft-inference-daemon
sudo systemctl start tft-dashboard

# Enable auto-start on boot
sudo systemctl enable tft-metrics-generator
sudo systemctl enable tft-inference-daemon
sudo systemctl enable tft-dashboard

# Check status
sudo systemctl status tft-inference-daemon
sudo journalctl -u tft-inference-daemon -f
```

---

## 🔍 Testing Security

### 1. Test API Key Required

```bash
# Should FAIL with HTTP 403 Forbidden
curl http://localhost:8000/predictions/current

# Should SUCCEED with HTTP 200 OK
curl -H "X-API-Key: a1b2c3d4e5f6..." http://localhost:8000/predictions/current
```

### 2. Test CORS Whitelist

```bash
# Should FAIL with CORS error
curl -H "Origin: https://evil.com" \
     -H "X-API-Key: a1b2c3d4e5f6..." \
     http://localhost:8000/predictions/current

# Should SUCCEED
curl -H "Origin: http://localhost:8501" \
     -H "X-API-Key: a1b2c3d4e5f6..." \
     http://localhost:8000/predictions/current
```

### 3. Test Rate Limiting

```bash
# Send 100 requests rapidly (should get HTTP 429 after 30 requests)
for i in {1..100}; do
  curl -H "X-API-Key: a1b2c3d4e5f6..." http://localhost:8000/predictions/current
  echo "Request $i"
done
```

### 4. Test Dashboard Authentication

```bash
# Start dashboard and check logs
streamlit run tft_dashboard_web.py

# Look for error messages in dashboard:
# "❌ Authentication failed - check API key configuration" ← BAD
# No error message and predictions visible ← GOOD
```

### 5. Test Parquet Migration

```bash
# Check persistence file format
ls -lh inference_rolling_window.*

# Should see:
# inference_rolling_window.parquet  (1-2 MB) ← GOOD (secure)
# inference_rolling_window.pkl      (2-3 MB) ← LEGACY (will be migrated)

# After first daemon startup, check logs:
# [MIGRATION] Found legacy pickle file - migrating to Parquet...
# [MIGRATION] Successfully migrated to Parquet format!
```

---

## 🐳 Docker Deployment

### Docker Compose Configuration

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  metrics-generator:
    build: .
    command: python metrics_generator_daemon.py
    ports:
      - "8001:8001"
    restart: always

  inference-daemon:
    build: .
    command: python tft_inference_daemon.py
    ports:
      - "8000:8000"
    environment:
      - TFT_API_KEY=${TFT_API_KEY}  # From .env file
      - CORS_ORIGINS=http://dashboard:8501
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: always
    depends_on:
      - metrics-generator

  dashboard:
    build: .
    command: streamlit run tft_dashboard_web.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - TFT_API_KEY=${TFT_API_KEY}  # From .env file
    restart: always
    depends_on:
      - inference-daemon
```

### Environment File

Create `.env`:
```bash
# Generate key with: openssl rand -hex 32
TFT_API_KEY=a1b2c3d4e5f6...
```

### Start Services

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f inference-daemon

# Stop all services
docker-compose down
```

---

## 🔧 Troubleshooting

### Dashboard: "Authentication failed - check API key"

**Cause**: API key mismatch between dashboard and daemon

**Fix**:
```bash
# Check daemon API key
echo $TFT_API_KEY

# Check dashboard secrets
cat .streamlit/secrets.toml

# Ensure they MATCH exactly (no extra spaces, quotes, etc.)
```

---

### Dashboard: "Not connected: Connection refused"

**Cause**: Inference daemon not running or wrong URL

**Fix**:
```bash
# Check daemon is running
curl http://localhost:8000/health

# Check daemon URL in dashboard
# Sidebar → Daemon URL should be "http://localhost:8000"
```

---

### Daemon: Pickle Migration Failed

**Cause**: Old pickle file corrupted or incompatible

**Fix**:
```bash
# Backup old pickle
mv inference_rolling_window.pkl inference_rolling_window.pkl.backup

# Start fresh (daemon will create new parquet file)
python tft_inference_daemon.py
```

---

### Rate Limiting: HTTP 429 Too Many Requests

**Cause**: Dashboard auto-refresh too aggressive or multiple clients

**Fix**:
```bash
# Dashboard: Increase refresh interval
# Sidebar → Refresh Settings → Set to 10+ seconds

# OR increase rate limits in tft_inference_daemon.py:
@limiter.limit("60/minute")  # Change to "120/minute" if needed
```

---

### CORS Error in Browser Console

**Cause**: Dashboard origin not in CORS whitelist

**Fix**:
```bash
# Add dashboard origin to CORS_ORIGINS
export CORS_ORIGINS="http://localhost:8501,http://192.168.1.100:8501"

# Restart daemon
```

---

## 📊 Monitoring Security

### Check Authentication Failures

```bash
# Monitor 403 errors (auth failures)
journalctl -u tft-inference-daemon -f | grep "403"

# Count auth failures in last hour
journalctl -u tft-inference-daemon --since "1 hour ago" | grep -c "403"
```

### Check Rate Limit Hits

```bash
# Monitor 429 errors (rate limit exceeded)
journalctl -u tft-inference-daemon -f | grep "429"
```

### Check CORS Violations

```bash
# Monitor CORS errors
journalctl -u tft-inference-daemon -f | grep "CORS"
```

---

## 🔐 Security Best Practices

### 1. API Key Rotation

**Rotate API keys every 90 days**:

```bash
# 1. Generate new key
NEW_KEY=$(openssl rand -hex 32)

# 2. Update daemon
export TFT_API_KEY=$NEW_KEY
sudo systemctl restart tft-inference-daemon

# 3. Update dashboard secrets
echo "[daemon]" > .streamlit/secrets.toml
echo "api_key = \"$NEW_KEY\"" >> .streamlit/secrets.toml

# 4. Restart dashboard
sudo systemctl restart tft-dashboard

# 5. Verify connectivity
curl -H "X-API-Key: $NEW_KEY" http://localhost:8000/predictions/current
```

---

### 2. Secure File Permissions

```bash
# Secrets file
chmod 600 .streamlit/secrets.toml

# Service files (contain API keys)
sudo chmod 600 /etc/systemd/system/tft-inference-daemon.service

# Parquet persistence files
chmod 640 inference_rolling_window.parquet
```

---

### 3. Firewall Configuration

```bash
# Allow only internal network to access daemon
sudo ufw allow from 10.0.0.0/8 to any port 8000
sudo ufw deny 8000

# Allow dashboard from internal network only
sudo ufw allow from 10.0.0.0/8 to any port 8501
sudo ufw deny 8501
```

---

### 4. Log Rotation

Create `/etc/logrotate.d/tft-monitoring`:
```
/var/log/tft-monitoring/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 tftuser tftuser
}
```

---

### 5. Backup API Keys

```bash
# Backup API key to secure location (NOT in git repo!)
echo "TFT_API_KEY=$(grep TFT_API_KEY /etc/systemd/system/tft-inference-daemon.service | cut -d'=' -f3 | tr -d '\"')" > /root/.tft_api_key
chmod 400 /root/.tft_api_key

# Restore from backup
source /root/.tft_api_key
export TFT_API_KEY
```

---

## 📚 Additional Resources

- [Security Improvements Complete](SECURITY_IMPROVEMENTS_COMPLETE.md) - Detailed security fixes
- [Dashboard Security Audit](DASHBOARD_SECURITY_AUDIT.md) - Dashboard vulnerability analysis
- [Parquet vs Pickle Comparison](PARQUET_VS_PICKLE_VS_JSON.md) - Persistence format analysis
- [Production Deployment](PRODUCTION_DEPLOYMENT.md) - Systemd and init.d setup

---

## ✅ Deployment Checklist

**Pre-Production**:
- [ ] Generate strong API key (`openssl rand -hex 32`)
- [ ] Install security dependencies (`./install_security_deps.sh`)
- [ ] Set `TFT_API_KEY` on daemon server
- [ ] Set `TFT_API_KEY` on dashboard server (same value!)
- [ ] Set `CORS_ORIGINS` to whitelist trusted origins
- [ ] Configure `.streamlit/secrets.toml` or environment variables
- [ ] Test API key authentication works
- [ ] Test CORS whitelist blocks unauthorized origins
- [ ] Test rate limiting activates (send 100+ requests)
- [ ] Verify Parquet persistence (check file size ~1-2 MB)
- [ ] Check old pickle file migrated successfully
- [ ] Secure file permissions (`chmod 600 secrets.toml`)
- [ ] Configure firewall rules (restrict ports to internal network)
- [ ] Set up log rotation (`/etc/logrotate.d/tft-monitoring`)
- [ ] Document API key location for operations team
- [ ] Create incident response plan for security events

**Post-Production**:
- [ ] Monitor authentication failures (`journalctl | grep 403`)
- [ ] Monitor rate limit hits (`journalctl | grep 429`)
- [ ] Schedule API key rotation (every 90 days)
- [ ] Review security logs weekly
- [ ] Backup API keys securely (not in git!)

---

**Last Updated**: 2025-10-16
**Version**: 2.0
**Status**: ✅ Production Ready
