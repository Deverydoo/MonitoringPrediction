# Tachyon Argus - Quick Start Guide

Get the predictive monitoring system running in 5 minutes.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB RAM minimum

## Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/MonitoringPrediction.git
cd MonitoringPrediction

# Create conda environment
conda env create -f environment.yml
conda activate tachyon
```

## Step 2: Generate API Key

```bash
cd Argus

# Windows
bin\setup_api_key.bat

# Linux/Mac
./bin/setup_api_key.sh
```

This creates `.tachyon_key` with your API key.

## Step 3: Start the System

### Option A: Full System (Inference + Demo Data + Dashboard)

```bash
# Windows
start_all.bat

# Linux/Mac
./start_all.sh
```

### Option B: Inference Daemon Only

```bash
python src/daemons/tft_inference_daemon.py --port 8000
```

## Step 4: Verify It's Running

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "service": "tft_inference_daemon", "running": true}
```

## Step 5: Feed Some Data

```bash
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "records": [{
      "timestamp": "2025-01-15T10:00:00",
      "server_name": "testserver001",
      "status": "healthy",
      "cpu_user_pct": 25.0,
      "cpu_sys_pct": 5.0,
      "cpu_iowait_pct": 2.0,
      "cpu_idle_pct": 68.0,
      "java_cpu_pct": 10.0,
      "mem_used_pct": 60.0,
      "swap_used_pct": 0.0,
      "disk_usage_pct": 40.0,
      "net_in_mb_s": 5.0,
      "net_out_mb_s": 3.0,
      "back_close_wait": 0,
      "front_close_wait": 0,
      "load_average": 1.5,
      "uptime_days": 30
    }]
  }'
```

## Step 6: Get Predictions

After feeding ~100 records (warmup period):

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/predictions/current
```

## Step 7: Check Fleet Health (Cascading Failure Detection)

The system automatically detects cascading failures across your server fleet:

```bash
# Get fleet health score
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/cascade/health

# Response shows fleet-wide status:
# {
#   "health_score": 85.2,
#   "status": "healthy",
#   "correlation_score": 0.234,
#   "anomaly_rate": 0.05,
#   "cascade_risk": "low"
# }
```

## Step 8: Monitor Model Drift

Check if the model needs retraining:

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/drift/status

# Response:
# {
#   "drift_detected": false,
#   "auto_retrain_enabled": true,
#   "last_retrain": "2025-01-15T10:30:00"
# }
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | REST API for predictions |
| Dashboard | http://localhost:8501 | Web dashboard (if started) |
| Health | http://localhost:8000/health | Health check endpoint |
| Cascade Status | http://localhost:8000/cascade/status | Cascading failure detection |
| Fleet Health | http://localhost:8000/cascade/health | Fleet-wide health score |
| Drift Status | http://localhost:8000/drift/status | Model drift monitoring |

## Key Features

- **Multi-Target Prediction**: Predicts CPU, memory, swap, I/O wait, and load simultaneously
- **Cascading Failure Detection**: Identifies fleet-wide issues even when individual servers appear healthy
- **Automatic Model Drift Detection**: Monitors prediction accuracy and triggers retraining when needed
- **Continuous Learning**: Model improves automatically as drift is detected

## What's Next?

- **Feed real data**: See [METRICS_FEED_GUIDE.md](METRICS_FEED_GUIDE.md)
- **Build a dashboard**: See [DASHBOARD_INTEGRATION_GUIDE.md](DASHBOARD_INTEGRATION_GUIDE.md)
- **Train your own model**: See [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Deploy to production**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## Troubleshooting

**"Model not found" error**
- You need a trained model in `Argus/models/`
- Either download a pre-trained model or train one (see Training Guide)

**"insufficient_data" error**
- The daemon needs ~100 data points before predictions work
- Keep feeding data or use the demo generator

**Connection refused**
- Check the daemon is running: `curl http://localhost:8000/health`
- Check the port isn't blocked by firewall

**GPU not detected**
- Verify CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`
- The system will fall back to CPU if no GPU is available
