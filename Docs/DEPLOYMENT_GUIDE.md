# Tachyon Argus - Deployment Guide

Production deployment and operational guide for Tachyon Argus.

## Deployment Options

| Option | Use Case | Complexity |
|--------|----------|------------|
| Single Server | Dev/Test, Small teams | Low |
| Docker | Containerized deployment | Medium |
| Kubernetes | Enterprise, Auto-scaling | High |

---

## Single Server Deployment

### Prerequisites

- Ubuntu 20.04+ or RHEL 8+
- Python 3.10+
- CUDA 11.8+ (if using GPU)
- 8GB+ RAM
- 50GB disk space

### Step 1: Install Dependencies

```bash
# System packages
sudo apt update
sudo apt install -y python3.10 python3.10-venv git

# CUDA (optional, for GPU support)
# Follow NVIDIA instructions for your distribution
```

### Step 2: Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-org/MonitoringPrediction.git
cd MonitoringPrediction

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r Argus/requirements_inference.txt
```

### Step 3: Configure

```bash
cd Argus

# Generate API key
python bin/generate_api_key.py
# Save the generated key

# Create environment file
cat > .env << EOF
TACHYON_API_KEY=your-generated-key
TACHYON_PORT=8000
TACHYON_HOST=0.0.0.0
EOF
```

### Step 4: Deploy Model

Copy your trained model to the `models/` directory:

```bash
# From your training environment
scp -r models/tft_model_YYYYMMDD_HHMMSS user@production:/path/to/Argus/models/
```

### Step 5: Create Systemd Service

```bash
sudo cat > /etc/systemd/system/tachyon-argus.service << EOF
[Unit]
Description=Tachyon Argus Inference Daemon
After=network.target

[Service]
Type=simple
User=tachyon
Group=tachyon
WorkingDirectory=/opt/tachyon/Argus
Environment=PATH=/opt/tachyon/venv/bin
ExecStart=/opt/tachyon/venv/bin/python src/daemons/tft_inference_daemon.py --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable tachyon-argus
sudo systemctl start tachyon-argus
```

### Step 6: Configure Firewall

```bash
# Allow API port
sudo ufw allow 8000/tcp

# Or with firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### Step 7: Verify Deployment

```bash
# Check service status
sudo systemctl status tachyon-argus

# Test health endpoint
curl http://localhost:8000/health

# Test with API key
curl -H "X-API-Key: your-key" http://localhost:8000/status
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY Argus/requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

# Copy application
COPY Argus/ ./Argus/

# Copy model (or mount as volume)
# COPY models/ ./Argus/models/

WORKDIR /app/Argus

EXPOSE 8000

CMD ["python", "src/daemons/tft_inference_daemon.py", "--port", "8000", "--host", "0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  tachyon-argus:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/Argus/models:ro
      - ./data_buffer:/app/Argus/data_buffer
    environment:
      - TACHYON_API_KEY=${TACHYON_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run

```bash
# Build image
docker build -t tachyon-argus:latest .

# Run container
docker run -d \
  --name tachyon-argus \
  -p 8000:8000 \
  -v $(pwd)/models:/app/Argus/models:ro \
  -e TACHYON_API_KEY=your-key \
  tachyon-argus:latest

# Or with docker-compose
docker-compose up -d
```

---

## GPU Support (NVIDIA)

### Docker with GPU

```yaml
# docker-compose.yml
version: '3.8'

services:
  tachyon-argus:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/Argus/models:ro
    environment:
      - TACHYON_API_KEY=${TACHYON_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### CUDA Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# ... rest of Dockerfile
```

---

## Reverse Proxy (Nginx)

### Nginx Configuration

```nginx
upstream tachyon {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name argus.yourdomain.com;

    ssl_certificate /etc/ssl/certs/argus.crt;
    ssl_certificate_key /etc/ssl/private/argus.key;

    location / {
        proxy_pass http://tachyon;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TACHYON_API_KEY` | - | API key (required) |
| `TACHYON_PORT` | 8000 | API port |
| `TACHYON_HOST` | 0.0.0.0 | Bind address |
| `TACHYON_MODEL_DIR` | ./models | Model directory |
| `TACHYON_LOG_LEVEL` | INFO | Log level |

### Command Line Arguments

```bash
python src/daemons/tft_inference_daemon.py \
  --port 8000 \
  --host 0.0.0.0 \
  --model-path models/tft_model_latest \
  --enable-retraining
```

---

## Monitoring & Logging

### Log Configuration

Logs are written to stdout/stderr by default. For file logging:

```bash
python src/daemons/tft_inference_daemon.py 2>&1 | tee -a /var/log/tachyon/argus.log
```

### Health Monitoring

Set up monitoring with your preferred tool:

**Prometheus Endpoint:**
```bash
curl http://localhost:8000/health
```

**Uptime Check Script:**
```bash
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$response" != "200" ]; then
    echo "Tachyon Argus is DOWN"
    # Send alert
fi
```

### Metrics to Monitor

| Metric | Source | Alert Threshold |
|--------|--------|-----------------|
| API Response Time | `/health` | > 500ms |
| Rolling Window Size | `/status` | < 100 |
| Memory Usage | System | > 90% |
| GPU Memory | nvidia-smi | > 90% |
| Error Rate | Logs | > 1% |

---

## Backup & Recovery

### What to Backup

1. **Models** - `Argus/models/`
2. **Configuration** - `.env`, API keys
3. **Historical Data** - `Argus/data_buffer/` (optional)

### Backup Script

```bash
#!/bin/bash
BACKUP_DIR=/backup/tachyon/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Backup models
cp -r /opt/tachyon/Argus/models $BACKUP_DIR/

# Backup config
cp /opt/tachyon/Argus/.env $BACKUP_DIR/

# Compress
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR
```

### Recovery

```bash
# Stop service
sudo systemctl stop tachyon-argus

# Restore from backup
tar -xzf /backup/tachyon/20250115.tar.gz -C /opt/tachyon/

# Start service
sudo systemctl start tachyon-argus
```

---

## Scaling

### Horizontal Scaling

For high availability, run multiple instances behind a load balancer:

```
                    ┌─────────────────┐
                    │  Load Balancer  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Instance 1  │   │   Instance 2  │   │   Instance 3  │
│   (GPU Node)  │   │   (GPU Node)  │   │   (CPU Node)  │
└───────────────┘   └───────────────┘   └───────────────┘
```

**Note:** Each instance maintains its own rolling window. For shared state, implement a Redis backend (future feature).

### Resource Sizing

| Workload | Servers | CPU | RAM | GPU |
|----------|---------|-----|-----|-----|
| Small | <50 | 4 cores | 8GB | Optional |
| Medium | 50-200 | 8 cores | 16GB | Recommended |
| Large | 200+ | 16 cores | 32GB | Required |

---

## Security Best Practices

### API Key Management

- Rotate API keys regularly (monthly)
- Use environment variables, not config files
- Different keys for different environments

### Network Security

- Use HTTPS in production (via reverse proxy)
- Restrict API access to known IPs
- Use VPN for internal access

### System Hardening

```bash
# Create dedicated user
sudo useradd -r -s /bin/false tachyon

# Set permissions
sudo chown -R tachyon:tachyon /opt/tachyon
sudo chmod 700 /opt/tachyon/Argus/models
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u tachyon-argus -n 100

# Check permissions
ls -la /opt/tachyon/Argus/models/

# Check Python environment
source /opt/tachyon/venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

### High Memory Usage

- Enable streaming mode for large datasets
- Check for memory leaks in historical store
- Reduce rolling window size if needed

### Slow Predictions

- Verify GPU is being used: check `/status` endpoint
- Check system load: `htop`, `nvidia-smi`
- Consider upgrading hardware

### Model Not Loading

- Verify model files exist: `ls -la models/`
- Check `dataset_parameters.pkl` is present
- Check model format (SafeTensors vs PyTorch)

---

## Maintenance

### Scheduled Tasks

| Task | Frequency | Command |
|------|-----------|---------|
| Log rotation | Daily | logrotate |
| Model backup | Daily | backup script |
| Health check | Every 5 min | monitoring |
| Retraining | Weekly | cron job |

### Weekly Retraining Cron

```bash
# /etc/cron.d/tachyon-retrain
0 2 * * 0 tachyon /opt/tachyon/Argus/bin/weekly_retrain.sh
```

### Model Hot Reload

Update models without restart:

```bash
# Copy new model
cp -r new_model/ /opt/tachyon/Argus/models/tft_model_new/

# Trigger reload via API
curl -X POST -H "X-API-Key: your-key" \
  "http://localhost:8000/admin/reload-model?model_path=models/tft_model_new"
```

---

## Support

For issues:
1. Check logs: `journalctl -u tachyon-argus`
2. Verify health: `curl http://localhost:8000/health`
3. Check status: `curl http://localhost:8000/status`
4. Review this guide's troubleshooting section
