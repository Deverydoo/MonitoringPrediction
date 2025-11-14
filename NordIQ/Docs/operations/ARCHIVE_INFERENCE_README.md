# TFT Inference Engine - Clean Architecture

**Status**: Production-ready architecture (October 2025)

This document describes the **clean, standalone inference architecture** that emerged from refactoring the monolithic inference system. The new design enables seamless production deployment.

---

## 🎯 Architecture Overview

The inference system uses a **clean separation of concerns** with three independent services:

```
┌─────────────────────┐
│ Metrics Generator   │  Port 8001 (REST API)
│ Daemon              │  - Generates realistic metrics
└──────────┬──────────┘  - Scenario control via REST
           │
           │ POST /feed/data
           ↓
┌─────────────────────┐
│ Inference Daemon    │  Port 8000 (REST API)
│ (Standalone)        │  - Loads TFT model
└──────────┬──────────┘  - Processes predictions
           │
           │ GET /predictions/current
           ↓
┌─────────────────────┐
│ Streamlit Dashboard │  Port 8501
│                     │  - Visualizations
└─────────────────────┘  - Alerts & monitoring
```

### Key Benefits

✅ **Production-Ready**: Services can be deployed independently
✅ **Zero Duplication**: One metrics generator for training AND demo data
✅ **Simple Integration**: REST API accepts standard log formats
✅ **Clean Code**: 900 lines vs. 1600-line monolith
✅ **No External Dependencies**: Standalone daemon includes all prediction logic

---

## 📁 File Structure

### Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `tft_inference_daemon.py` | 900 | **Standalone inference service** - includes embedded TFTInference class |
| `metrics_generator_daemon.py` | 400 | **Data generator service** - streams realistic metrics with scenario control |
| `tft_dashboard_web.py` | 1900 | **Dash dashboard** - visualization and monitoring |

### Supporting Files

| File | Purpose |
|------|---------|
| `server_encoder.py` | Hash-based server name encoding |
| `data_validator.py` | Data contract validation (v3.0) |
| `gpu_profiles.py` | GPU optimization profiles |
| `config.py` | Alert thresholds and configuration |

---

## 🚀 Quick Start

### 1. Start All Services (Demo Mode)

**Windows:**
```bash
start_all.bat
```

**Linux/Mac:**
```bash
./start_all.sh
```

This launches:
- Inference daemon on `http://localhost:8000` (or `http://192.168.x.x:8000`)
- Metrics generator on port 8001 (background)
- Dashboard on `http://localhost:8501` (or `http://192.168.x.x:8501`)

### 2. Start Services Individually

**Inference Daemon:**
```bash
python tft_inference_daemon.py --port 8000
```

**Metrics Generator:**
```bash
python metrics_generator_daemon.py --stream --servers 20
```

**Dashboard:**
```bash
python dash_app.py
```

---

## 🏗️ Inference Daemon Architecture

### Embedded TFTInference Class

The daemon includes a **complete, standalone copy** of the TFTInference class (lines 53-707):

```python
class TFTInference:
    """
    TFT inference engine that loads and uses the trained model.

    This is a standalone copy - no dependencies on old tft_inference.py file.
    """

    def __init__(self, model_path: Optional[str] = None, use_real_model: bool = True):
        # Auto-detect GPU and load model
        self.model_dir = self._find_model(model_path)
        self._load_model()

    def predict(self, data: DataFrame, horizon: int = 96) -> Dict:
        """
        Make predictions on input data.

        Returns:
            {
                'predictions': {...},  # Per-server forecasts
                'alerts': [...],       # Active alerts
                'environment': {...},  # Fleet-wide metrics
                'metadata': {...}      # Model info
            }
        """
```

### Key Features

1. **Model Loading**: Loads TFT model from safetensors with trained encoders
2. **GPU Optimization**: Auto-configures batch size and workers based on GPU
3. **Warmup Period**: Needs 150 timesteps per server before TFT predictions (uses heuristic fallback)
4. **Rolling Window**: Maintains last 6000 records for inference
5. **Dual Prediction Modes**:
   - TFT model predictions (after warmup)
   - Heuristic predictions (during warmup or fallback)

---

## 🔌 REST API Reference

### Inference Daemon (Port 8000)

#### 1. Feed Data to Daemon

**POST** `/feed/data`

Accepts metrics data in batches. This is how you integrate with production systems.

**Request:**
```json
{
  "records": [
    {
      "timestamp": "2025-10-12T21:40:00",
      "server_name": "ppml0001",
      "cpu_percent": 45.2,
      "memory_percent": 62.1,
      "disk_percent": 35.8,
      "load_average": 2.3,
      "status": "healthy",
      "profile": "ml_compute"
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "records_added": 20,
  "window_size": 500,
  "tick": 25
}
```

#### 2. Get Predictions

**GET** `/predictions/current`

Returns current predictions for all servers with 8-hour forecast.

**Response:**
```json
{
  "predictions": {
    "ppml0001": {
      "cpu_percent": {
        "p50": [45.2, 46.1, 47.3, ...],  // 96 steps (8 hours)
        "p10": [40.1, 41.2, ...],         // 10th percentile
        "p90": [50.3, 51.5, ...],         // 90th percentile
        "current": 45.2,
        "trend": 0.12
      },
      "memory_percent": {...},
      "disk_percent": {...},
      "load_average": {...}
    }
  },
  "alerts": [
    {
      "server": "ppml0001",
      "metric": "cpu_percent",
      "severity": "warning",
      "predicted_value": 85.3,
      "threshold": 80.0,
      "minutes_ahead": 25
    }
  ],
  "environment": {
    "incident_probability_30m": 0.15,
    "incident_probability_8h": 0.35,
    "high_risk_servers": 2,
    "total_servers": 20,
    "fleet_health": "healthy"
  },
  "metadata": {
    "model_type": "TFT",
    "model_dir": "models/tft_model_20251012_165527",
    "prediction_time": "2025-10-12T21:45:00",
    "horizon_steps": 96,
    "device": "cuda:0"
  }
}
```

#### 3. Get Active Alerts

**GET** `/alerts/active`

Returns only the active alerts from latest predictions.

**Response:**
```json
{
  "timestamp": "2025-10-12T21:45:00",
  "count": 3,
  "alerts": [...]
}
```

#### 4. Health Check

**GET** `/health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "tft_inference_daemon",
  "running": true
}
```

#### 5. Daemon Status

**GET** `/status`

Get detailed daemon status including warmup progress.

**Response:**
```json
{
  "running": true,
  "tick_count": 150,
  "window_size": 3000,
  "warmup": {
    "is_warmed_up": true,
    "progress_percent": 100,
    "threshold": 150,
    "message": "Model ready - using TFT predictions"
  },
  "fleet": {
    "total_servers": 20,
    "healthy_servers": 20,
    "fleet_status": "20/20 servers ready"
  }
}
```

---

## 🎮 Metrics Generator Daemon (Port 8001)

### Scenario Control API

The metrics generator has a REST API for on-the-fly scenario changes (no restart needed).

#### Change Scenario

**POST** `/scenario/set`

**Request:**
```json
{
  "scenario": "degrading"
}
```

**Valid scenarios:**
- `healthy` - Normal operations (1.0x multiplier)
- `degrading` - Early warning signs (1.15x multiplier, 15% increase)
- `critical` - Clear problems (1.6x multiplier, 60% increase)

**Response:**
```json
{
  "status": "success",
  "old_scenario": "healthy",
  "new_scenario": "degrading",
  "affected_servers": 3,
  "message": "Scenario changed to degrading. 3 servers affected."
}
```

#### Get Scenario Status

**GET** `/scenario/status`

**Response:**
```json
{
  "scenario": "degrading",
  "multiplier": 1.15,
  "affected_servers": ["ppml0003", "ppdb005", "ppweb012"],
  "uptime_seconds": 450
}
```

---

## 🏭 Production Deployment

### Integration Strategy

The clean architecture makes production deployment **incredibly simple**:

```
┌──────────────────────┐
│ Your Production Logs │
│ (syslog, splunk, etc)│
└──────────┬───────────┘
           │
           │ Parse & format logs
           ↓
┌──────────────────────┐
│ Log Forwarder        │  Transform to JSON format
│ (logstash, fluentd)  │  POST to inference daemon
└──────────┬───────────┘
           │
           │ POST http://inference-daemon:8000/feed/data
           ↓
┌──────────────────────┐
│ TFT Inference Daemon │  Analyze & predict
│                      │  Generate alerts
└──────────┬───────────┘
           │
           │ GET /predictions/current
           ↓
┌──────────────────────┐
│ Monitoring Dashboard │  Display predictions
│ (streamlit/grafana)  │  Route alerts
└──────────────────────┘
```

### Data Format Requirements

Your log forwarder needs to transform logs into this JSON format:

```json
{
  "records": [
    {
      "timestamp": "2025-10-12T21:40:00",      // ISO 8601 format
      "server_name": "production-web-01",      // Unique server identifier
      "cpu_percent": 45.2,                     // 0-100
      "memory_percent": 62.1,                  // 0-100
      "disk_percent": 35.8,                    // 0-100
      "load_average": 2.3,                     // System load
      "status": "healthy",                     // Optional: healthy/degrading/critical
      "profile": "web_api"                     // Optional: server type for transfer learning
    }
  ]
}
```

**Required fields:**
- `timestamp` - ISO 8601 format
- `server_name` - Unique identifier
- `cpu_percent`, `memory_percent`, `disk_percent`, `load_average` - Numeric metrics

**Optional fields:**
- `status` - Current server state (healthy, heavy_load, critical_issue, etc.)
- `profile` - Server type (ml_compute, database, web_api, etc.) - enables transfer learning

### Example: Logstash Integration

```ruby
# logstash.conf
output {
  http {
    url => "http://inference-daemon:8000/feed/data"
    http_method => "post"
    format => "json"
    mapping => {
      "records" => [
        {
          "timestamp" => "%{@timestamp}"
          "server_name" => "%{host}"
          "cpu_percent" => "%{cpu_pct}"
          "memory_percent" => "%{mem_pct}"
          "disk_percent" => "%{disk_pct}"
          "load_average" => "%{load_avg}"
        }
      ]
    }
  }
}
```

### Deployment Checklist

- [ ] Deploy inference daemon with trained model
- [ ] Configure log forwarder to POST to `/feed/data` every 5 seconds
- [ ] Wait 12.5 minutes for warmup (150 timesteps × 5 seconds)
- [ ] Verify predictions at `/predictions/current`
- [ ] Deploy dashboard pointing to inference daemon
- [ ] Configure alert routing from `/alerts/active`

### Performance Considerations

- **Throughput**: Handles 20-100 servers easily (tested with RTX 4090)
- **Latency**: <1 second (with GPU), 2-3 seconds (CPU)
- **Warmup Time**: 12.5 minutes per server (150 timesteps × 5 seconds)
- **Rolling Window**: Keeps last 6000 records (~8 hours of data)
- **Memory Usage**: ~500MB for model + data window

### CPU vs GPU for Inference

**Good news**: Inference runs fine on CPU! The model is tiny (87K parameters).

| Hardware | Latency | Throughput | Use Case |
|----------|---------|------------|----------|
| **CPU (modern)** | 2-3 seconds | 20-50 servers | ✅ Production (cost-effective) |
| **GPU (RTX 3060+)** | <1 second | 100+ servers | High-frequency predictions |
| **GPU (RTX 4090)** | <0.5 seconds | 500+ servers | Large-scale deployments |

**Recommendation**: Start with CPU for production. The 2-3 second latency is totally acceptable for 5-minute prediction intervals. Only use GPU if you need <1 second response times or have 100+ servers.

**Note**: Training DOES require GPU (30 min on RTX 4090 vs 4+ hours on CPU), but inference is very lightweight.

---

## 🔧 Configuration

### Alert Thresholds (config.py)

```python
CONFIG = {
    'alert_thresholds': {
        'cpu_percent': {'warning': 80, 'critical': 95},
        'memory_percent': {'warning': 85, 'critical': 95},
        'disk_percent': {'warning': 85, 'critical': 95},
        'load_average': {'warning': 8.0, 'critical': 12.0}
    }
}
```

### Scenario Multipliers (metrics_generator_daemon.py)

```python
self.scenario_multipliers = {
    'healthy': 1.0,      # Normal operations
    'degrading': 1.15,   # Subtle increase (15%) - early warning signs
    'critical': 1.6      # Significant issues (60%) - clear problems
}
```

### Warmup Settings (tft_inference_daemon.py)

```python
WARMUP_THRESHOLD = 150  # Timesteps needed per server before TFT predictions work
WINDOW_SIZE = 6000      # Keep last 6000 records (~8 hours at 5-second intervals)
PORT = 8000
```

---

## 🎓 Understanding Warmup

The inference daemon needs **150 timesteps per server** before the TFT model can make predictions.

### Why Warmup?

TFT (Temporal Fusion Transformer) needs historical context to make accurate predictions. The model uses the last 24 timesteps as encoder input to predict the next 96 timesteps.

### Warmup Timeline

| Time | Timesteps | Status | Prediction Mode |
|------|-----------|--------|-----------------|
| 0 min | 0 | Starting | Heuristic fallback |
| 2.5 min | 30 | Warming up (20%) | Heuristic fallback |
| 5 min | 60 | Warming up (40%) | Heuristic fallback |
| 7.5 min | 90 | Warming up (60%) | Heuristic fallback |
| 10 min | 120 | Warming up (80%) | Heuristic fallback |
| **12.5 min** | **150** | **Ready** | **TFT model** |

### Monitoring Warmup

Check warmup progress:
```bash
curl http://localhost:8000/status
```

Dashboard shows warmup status automatically in the sidebar.

### Heuristic Fallback

During warmup, the system uses **trend-based heuristic predictions**:
- Analyzes recent trends
- Applies statistical forecasting
- Still generates useful predictions
- Seamlessly switches to TFT when ready

---

## 🐛 Troubleshooting

### Dashboard Shows "0/0 servers"

**Symptom**: Fleet status shows `0/0` instead of `20/20`

**Causes:**
1. Inference daemon not connected
2. Metrics generator not streaming data
3. Wrong data format

**Solutions:**
```bash
# Check inference daemon status
curl http://localhost:8000/status

# Check if data is being received
curl http://localhost:8000/status | grep window_size

# Restart metrics generator
python metrics_generator_daemon.py --stream --servers 20
```

### Predictions Show All 100% CPU

**Symptom**: All predictions show maxed-out metrics

**Cause**: Heuristic fallback during warmup or data quality issues

**Solutions:**
1. Wait for warmup to complete (check `/status`)
2. Click "Healthy" scenario button to reset
3. Check training data quality

### "Connection Refused" Error

**Symptom**: Dashboard can't connect to daemon

**Cause**: Inference daemon not running or wrong port

**Solutions:**
```bash
# Check if daemon is running
netstat -ano | findstr :8000

# Restart daemon
python tft_inference_daemon.py --port 8000

# Check dashboard URL in tft_dashboard_web.py:44
# Should be: daemon_url = "http://localhost:8000"
```

### Slow Predictions

**Symptom**: Predictions take >5 seconds

**Causes:**
1. Running on CPU instead of GPU
2. Too many servers (>100)
3. Large rolling window

**Solutions:**
1. Verify GPU is detected: Check daemon startup logs for "GPU: NVIDIA..."
2. Reduce window size in config if needed
3. Consider batching requests

---

## 📊 Performance Metrics

Tested on **RTX 4090** with 20 servers:

| Metric | Value |
|--------|-------|
| Prediction Latency | <1 second |
| Throughput | 20 servers @ 5-second intervals |
| Memory Usage | ~500MB |
| GPU Utilization | ~15% (inference) |
| Warmup Time | 12.5 minutes |
| Model Parameters | 87,080 |
| Training Time | ~30 minutes (1 epoch) |

---

## 🎯 Next Steps

### Phase 1: Production Readiness ✅ COMPLETE
- [x] Clean architecture with separated services
- [x] REST API for data ingestion
- [x] Standalone inference daemon
- [x] Scenario control via REST
- [x] Documentation

### Phase 2: Production Integration (Next)
- [ ] Implement log forwarder (Logstash/Fluentd)
- [ ] Add authentication/authorization to APIs
- [ ] Set up alert routing (PagerDuty/Slack)
- [ ] Add monitoring/observability (Prometheus/Grafana)
- [ ] Load testing and optimization

### Phase 3: Advanced Features
- [ ] Multi-metric predictions (CPU + Memory simultaneously)
- [ ] Confidence scores for predictions
- [ ] Root cause analysis
- [ ] Auto-remediation triggers
- [ ] Transfer learning for new servers

---

## 📚 Related Documentation

- [PROJECT_CODEX.md](PROJECT_CODEX.md) - Overall project architecture
- [DATA_CONTRACT.md](DATA_CONTRACT.md) - Data format specifications
- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md) - Dashboard features and usage

---

## 🤝 Contributing

When modifying the inference system:

1. **Keep it standalone** - No external dependencies on old code
2. **Test the REST API** - Ensure all endpoints work
3. **Update this doc** - Document any API changes
4. **Performance test** - Verify <1 second prediction latency
5. **Check warmup** - Ensure graceful heuristic fallback

---

## 📝 Change Log

### October 2025 - Clean Architecture Refactor

**Major Changes:**
- Extracted TFTInference class into standalone daemon (900 lines, no dependencies)
- Created metrics_generator_daemon with REST API for scenario control
- Fixed response format to match dashboard expectations
- Added `/health` and `/alerts/active` endpoints
- Improved scenario multipliers (1.4→1.15 for degrading, 2.2→1.6 for critical)
- Added proper network address display (192.x.x.x instead of 0.0.0.0)

**Benefits:**
- Production deployment simplified - just stream logs to REST API
- No code duplication - one generator for training AND demo
- Clean separation - inference, generation, visualization are independent
- Easy integration - standard REST API, no custom protocols

**Files Created:**
- `tft_inference_daemon.py` - Standalone inference service
- `metrics_generator_daemon.py` - Data generator with scenario API
- `INFERENCE_README.md` - This documentation

**Files Deprecated:**
- `tft_inference.py` - Old monolithic file (1600 lines, keep for reference)

---

**Last Updated**: October 12, 2025
**Architecture Version**: 3.0 (Clean Architecture)
**API Version**: 2.0
