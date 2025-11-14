# API Key Setup Guide

**Version**: 1.0.0
**Last Updated**: October 17, 2025

---

## Overview

The TFT Monitoring System uses API key authentication to secure communication between the dashboard and the inference daemon. This guide explains how to set up and manage API keys.

---

## Quick Setup (Recommended)

### Windows
```cmd
setup_api_key.bat
```

### Linux/Mac
```bash
./setup_api_key.sh
```

This will automatically configure the API key for both the dashboard and daemon.

---

## Manual Setup

### Step 1: Configure Dashboard API Key

The dashboard API key is stored in `.streamlit/secrets.toml` (already created):

```toml
[daemon]
api_key = "bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO"
```

### Step 2: Configure Daemon API Key

**Option A: Environment Variable (Recommended for Production)**

```bash
# Linux/Mac
export TFT_API_KEY="bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO"

# Windows
set TFT_API_KEY=bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO
```

**Option B: .env File (Recommended for Development)**

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

The `.env` file contains:
```bash
TFT_API_KEY=bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO
```

Then load it before starting the daemon:
```bash
# Linux/Mac (using python-dotenv or export)
export $(cat .env | xargs)

# Or install python-dotenv and the daemon will auto-load it
pip install python-dotenv
```

### Step 3: Start Services

```bash
# Terminal 1: Start daemon (will read TFT_API_KEY from environment)
python tft_inference_daemon.py

# Terminal 2: Start dashboard (reads from .streamlit/secrets.toml)
python dash_app.py
```

---

## How It Works

### Authentication Flow

1. **Dashboard** reads API key from `.streamlit/secrets.toml`
2. **Dashboard** sends requests to daemon with `X-API-Key` header
3. **Daemon** reads `TFT_API_KEY` from environment variable
4. **Daemon** validates the API key matches
5. **Daemon** returns data if valid, 401/403 if invalid

### Key Locations

```
MonitoringPrediction/
├── .streamlit/
│   └── secrets.toml          # Dashboard API key (DO NOT COMMIT)
├── .env                       # Daemon API key (DO NOT COMMIT)
├── .env.example              # Template (safe to commit)
└── .gitignore                # Protects secrets.toml and .env
```

---

## Production Deployment

### Systemd Service (Linux)

Add the API key to your systemd service file:

```ini
[Service]
Environment="TFT_API_KEY=bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO"
ExecStart=/path/to/venv/bin/python tft_inference_daemon.py
```

### Docker

Pass as environment variable:

```bash
docker run -e TFT_API_KEY="bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO" ...
```

Or use docker-compose.yml:

```yaml
services:
  tft-daemon:
    environment:
      - TFT_API_KEY=bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO
```

### Cloud Deployment

**AWS/Azure/GCP:**
- Use secret management service (AWS Secrets Manager, Azure Key Vault, GCP Secret Manager)
- Inject at runtime via environment variable

**Kubernetes:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: tft-api-key
type: Opaque
stringData:
  api_key: bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO
```

---

## Generating a New API Key

If you need to regenerate the API key for security reasons:

```bash
python -c "
import secrets
import string
alphabet = string.ascii_letters + string.digits
api_key = ''.join(secrets.choice(alphabet) for _ in range(64))
print(f'New API Key: {api_key}')
"
```

**Then update both**:
1. `.streamlit/secrets.toml` (dashboard)
2. `.env` or environment variable (daemon)

---

## Development Mode (No API Key)

For local development without API key:

1. **Do NOT set** `TFT_API_KEY` environment variable
2. **Do NOT configure** `.streamlit/secrets.toml`
3. Both daemon and dashboard will run without authentication

**Warning**: Development mode is insecure. Only use on localhost!

---

## Troubleshooting

### Issue: "No API key configured" message

**Cause**: Dashboard cannot find API key in secrets.toml or environment

**Fix**:
```bash
# Run setup script
./setup_api_key.bat    # Windows
./setup_api_key.sh     # Linux/Mac
```

### Issue: Dashboard shows connection errors (401/403)

**Cause**: API key mismatch between dashboard and daemon

**Fix**:
1. Check `.streamlit/secrets.toml` has correct key
2. Check daemon environment has `TFT_API_KEY` set
3. Ensure both keys match exactly
4. Restart both services

### Issue: Daemon not reading .env file

**Fix**:
```bash
# Option 1: Manually export
export TFT_API_KEY="your-key-here"

# Option 2: Install python-dotenv
pip install python-dotenv

# Option 3: Source .env before running
export $(cat .env | xargs)
python tft_inference_daemon.py
```

---

## Security Best Practices

### ✅ DO:
- Use different API keys for dev/staging/production
- Rotate API keys periodically (quarterly)
- Store API keys in secret management systems (production)
- Use `.gitignore` to prevent committing secrets
- Use HTTPS in production to protect API key in transit

### ❌ DON'T:
- Commit `.streamlit/secrets.toml` or `.env` to git
- Share API keys via email or chat
- Use development mode in production
- Reuse API keys across different environments
- Log API keys in application logs

---

## API Key Security

**Current Key Entropy**: 64 characters from [a-zA-Z0-9] = ~380 bits
**Strength**: Cryptographically secure (generated with `secrets` module)
**Lifetime**: No expiration (rotate manually as needed)

---

## Quick Reference

| Component | API Key Location | How to Set |
|-----------|-----------------|------------|
| **Dashboard** | `.streamlit/secrets.toml` | Pre-configured |
| **Daemon** | Environment variable `TFT_API_KEY` | Export or .env file |

**Setup Command**: `./setup_api_key.bat` (Windows) or `./setup_api_key.sh` (Linux/Mac)

---

## Related Documentation

- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Production deployment guide
- [AUTHENTICATION_IMPLEMENTATION_GUIDE.md](AUTHENTICATION_IMPLEMENTATION_GUIDE.md) - Authentication options
- [OKTA_SSO_INTEGRATION.md](OKTA_SSO_INTEGRATION.md) - SSO integration

---

**Maintained By**: Project Team
**Last Updated**: October 17, 2025
