# Tachyon Argus - Architecture Overview

Technical architecture and design of the predictive infrastructure monitoring system.

## System Overview

Tachyon Argus is a machine learning system that predicts server failures 8 hours in advance using a Temporal Fusion Transformer (TFT) model with **cascading failure detection** and **continuous learning**.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tachyon Argus System                        │
│               Predictive Infrastructure Monitoring               │
│                    with Cascading Failure Detection              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Training    │    │   Inference   │    │   Dashboard   │
│   Pipeline    │    │    Daemon     │    │   (Optional)  │
│               │    │               │    │               │
│ - Fleet feats │    │ - REST API    │    │ - Plotly Dash │
│ - TFT model   │    │ - Predictions │    │ - Real-time   │
│ - Multi-tgt   │    │ - Cascade Det │    │ - Alerts      │
│ - Streaming   │    │ - Drift Mon   │    │ - XAI         │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Core Components

### 1. Inference Daemon (`tft_inference_daemon.py`)

The production service that makes predictions and detects cascading failures.

**Responsibilities:**
- Accept metrics via REST API
- Maintain rolling window of recent data
- Run TFT model predictions (multi-target)
- Calculate risk scores and alerts
- **Detect cascading failures** via correlation analysis
- **Monitor model drift** and trigger retraining
- Serve predictions via API

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    TFT Inference Daemon                          │
│                       (Port 8000)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Rolling    │    │     TFT      │    │    Risk      │       │
│  │   Window     │ -> │    Model     │ -> │   Scoring    │       │
│  │  (6000 pts)  │    │ (Multi-Tgt)  │    │  + Alerts    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ↑                   │                    │               │
│         │                   ▼                    ▼               │
│  POST /feed/data    ┌──────────────┐    GET /predictions        │
│         │           │  Correlation │                             │
│         │           │  Detector    │─────► GET /cascade/status   │
│         │           │  (Cascade)   │                             │
│         │           └──────────────┘                             │
│         │                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Drift     │    │    Auto      │    │     XAI      │       │
│  │   Monitor    │───>│  Retrainer   │    │  Explainers  │       │
│  │  (PER/DSS)   │    │  (Trigger)   │    │ SHAP/Attn/CF │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  GET /drift/status           GET /admin/training-status         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Rolling Window**: Maintains last 6000 data points (~8 hours for 45 servers)
- **Multi-Target Prediction**: Predicts CPU, memory, I/O wait, swap, load simultaneously
- **Risk Scoring**: Converts predictions to 0-100 risk scores
- **Cascading Failure Detection**: Cross-server correlation analysis
- **Drift Monitoring**: Automatic detection of model degradation
- **Auto-Retraining**: Drift-triggered model updates
- **XAI Explainability**: SHAP, Attention visualization, Counterfactuals
- **Hot Reload**: Swap models without restart

### 2. Training Pipeline (`tft_trainer.py`)

Trains the TFT prediction model with fleet-level features.

**Training Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| Standard | Load all data into memory | Small datasets |
| Streaming | Process 2-hour chunks | Large datasets |
| Incremental | Continue from existing model | Add more epochs |

**Fleet-Level Feature Engineering:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Fleet Feature Engineering                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each timestamp, compute:                                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Fleet Averages (per metric):                         │       │
│  │  - fleet_cpu_user_pct_mean    (fleet avg CPU)        │       │
│  │  - fleet_cpu_user_pct_std     (fleet variability)    │       │
│  │  - fleet_cpu_user_pct_max     (fleet peak)           │       │
│  │  - fleet_mem_used_pct_mean    (fleet avg memory)     │       │
│  │  - fleet_cpu_iowait_pct_mean  (fleet avg I/O wait)   │       │
│  │  - fleet_load_average_mean    (fleet avg load)       │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Fleet Stress Indicators:                             │       │
│  │  - fleet_pct_high_cpu     (% servers with CPU > 80%) │       │
│  │  - fleet_pct_high_mem     (% servers with mem > 85%) │       │
│  │  - fleet_pct_high_iowait  (% servers with IO > 15%)  │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Per-Server Anomaly Signals:                          │       │
│  │  - cpu_vs_fleet    (server CPU - fleet avg CPU)      │       │
│  │  - mem_vs_fleet    (server mem - fleet avg mem)      │       │
│  │  - iowait_vs_fleet (server I/O - fleet avg I/O)      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  Total: 18 fleet-level features for cascading detection         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Streaming Training Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Streaming Training                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  chunk_01.parquet -> [Fleet Features] -> [Train] -> [Free Mem]  │
│                                             ↓                    │
│  chunk_02.parquet -> [Fleet Features] -> [Train] -> [Free Mem]  │
│                                             ↓                    │
│  chunk_03.parquet -> [Fleet Features] -> [Train] -> [Free Mem]  │
│                                             ↓                    │
│        ...                                 ...                   │
│                                             ↓                    │
│  chunk_N.parquet  -> [Fleet Features] -> [Train] -> [Save]      │
│                                                                  │
│  Checkpoint saved every 5 chunks for resume capability          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Correlation Detector (`correlation_detector.py`)

**NEW** - Detects cascading failures by monitoring cross-server correlations.

**How It Works:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Cascading Failure Detection                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Track metrics per server over sliding window                │
│                                                                  │
│  2. Detect anomalies per server (z-score > 2.0)                 │
│                                                                  │
│  3. Calculate cross-server correlation:                         │
│     - If servers show similar patterns → high correlation       │
│     - High correlation + anomalies → CASCADE DETECTED           │
│                                                                  │
│  4. Generate cascade alert with:                                │
│     - Affected servers list                                     │
│     - Severity (critical/high/medium/low)                       │
│     - Correlation score                                         │
│     - Actionable recommendations                                │
│                                                                  │
│  Example Cascade Detection:                                     │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Time 10:00 - All servers normal                      │      │
│  │  Time 10:15 - Server A: CPU spike (anomaly)           │      │
│  │  Time 10:20 - Server B,C,D: CPU spikes (correlated)   │      │
│  │  Time 10:25 - CASCADE DETECTED: 4 servers, 87% corr   │      │
│  │             → Alert: "Check shared infrastructure"     │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Drift Monitor (`drift_monitor.py`)

Tracks model performance and triggers retraining when drift is detected.

**Drift Metrics:**

| Metric | Description | Threshold |
|--------|-------------|-----------|
| PER | Prediction Error Rate | 10% |
| DSS | Distribution Shift Score | 20% |
| FDS | Feature Drift Score | 15% |
| Anomaly Rate | Unusual pattern rate | 5% |

**Drift Detection Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Drift Detection & Auto-Retraining             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Prediction Made                                                 │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────────┐                                               │
│  │    Drift     │  Track: prediction errors, feature shifts     │
│  │   Monitor    │         distribution changes, anomalies       │
│  └──────────────┘                                               │
│        │                                                         │
│        ▼ (Every hour)                                           │
│  ┌──────────────┐                                               │
│  │   Calculate  │  Combined Score = PER×0.4 + DSS×0.3 +         │
│  │    Metrics   │                   FDS×0.2 + Anomaly×0.1       │
│  └──────────────┘                                               │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────────┐                                               │
│  │ Threshold    │  If any metric > threshold:                   │
│  │   Check      │     → needs_retraining = True                 │
│  └──────────────┘                                               │
│        │                                                         │
│        ▼ (If drift detected)                                    │
│  ┌──────────────┐                                               │
│  │    Auto      │  - 5 epoch incremental training               │
│  │  Retrainer   │  - Non-blocking (background thread)           │
│  │   Trigger    │  - 24h cooldown between triggers              │
│  └──────────────┘                                               │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────────┐                                               │
│  │    Model     │  Hot-reload new model without restart         │
│  │   Reload     │                                               │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Data Generator (`metrics_generator.py`)

Generates realistic synthetic server metrics.

**Features:**
- 7 server profiles (Database, ML, API, ETL, etc.)
- Realistic workload patterns (daily cycles, spikes)
- Failure scenarios for training
- Time-partitioned Parquet output

### 6. Core Libraries (`src/core/`)

Shared utilities used across components:

| Module | Purpose |
|--------|---------|
| `alert_levels.py` | Alert thresholds and colors |
| `server_encoder.py` | Hash-based server ID encoding |
| `data_validator.py` | Schema validation |
| `gpu_profiles.py` | GPU detection and configuration |
| `historical_store.py` | Parquet-based data storage |
| `nordiq_metrics.py` | Metrics definitions |
| `correlation_detector.py` | **NEW** Cross-server cascade detection |
| `drift_monitor.py` | Model drift tracking |
| `auto_retrainer.py` | Drift-triggered retraining |
| `data_buffer.py` | Data accumulation for training |

---

## Data Flow

### Training Data Flow

```
metrics_generator.py
        │
        ▼
┌─────────────────┐
│ training/       │
│ ├── partitioned/│  (2-hour Parquet chunks)
│ └── manifest    │
└─────────────────┘
        │
        ▼
  tft_trainer.py
        │ (Fleet feature engineering)
        ▼
┌─────────────────┐
│ models/         │
│ ├── model.safetensors │
│ ├── config.json       │
│ └── dataset_params.pkl│  (Includes fleet feature encoders)
└─────────────────┘
```

### Inference Data Flow

```
Your Monitoring System
        │
        ▼ POST /feed/data
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Daemon                              │
│                                                                  │
│  1. Validate schema (16 required fields)                        │
│  2. Add to rolling window (6000 points max)                     │
│  3. Update correlation detector (cascade check)                 │
│  4. Run TFT model prediction (96 steps = 8 hours, multi-target) │
│  5. Calculate risk scores (0-100)                               │
│  6. Determine alert levels (critical/warning/degraded/healthy)  │
│  7. Update drift monitor (track prediction accuracy)            │
│  8. Store in historical data (Parquet)                          │
│  9. Check drift threshold (hourly) → trigger retraining if needed│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
GET /predictions/current              GET /cascade/status
┌────────────────────────┐           ┌────────────────────────┐
│  Response:              │           │  Response:              │
│  {                      │           │  {                      │
│    "predictions": {...},│           │    "cascade_detected":  │
│    "summary": {...},    │           │      true,              │
│    "cascade_alert": {   │           │    "correlation_score": │
│      "severity": "high",│           │      0.87,              │
│      "affected": 5      │           │    "affected_servers":  │
│    }                    │           │      ["srv1", "srv2"]   │
│  }                      │           │  }                      │
└────────────────────────┘           └────────────────────────┘
```

---

## TFT Model Architecture

The Temporal Fusion Transformer (TFT) is designed for multi-horizon forecasting with interpretability.

### Key Components

1. **Variable Selection Networks**: Learn which input features are important (including fleet features)
2. **LSTM Encoder/Decoder**: Capture temporal patterns
3. **Multi-Head Attention**: Focus on relevant time periods
4. **Quantile Outputs**: Predict confidence intervals

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Hidden Size | 128 |
| Attention Heads | 8 |
| LSTM Layers | 2 |
| Dropout | 0.1 |
| Input Window | 96 steps (8 hours) |
| Prediction Horizon | 96 steps (8 hours) |
| Input Features | 14 server + 18 fleet = 32 |
| Target Metrics | 5 (multi-target) |
| Total Parameters | ~120K |

### Multi-Target Prediction

The model now predicts multiple metrics simultaneously:

| Target | Description |
|--------|-------------|
| `cpu_user_pct` | User CPU utilization |
| `cpu_iowait_pct` | I/O wait percentage |
| `mem_used_pct` | Memory utilization |
| `swap_used_pct` | Swap utilization |
| `load_average` | System load |

### Fleet-Level Features

**18 new features for cascading failure detection:**

| Feature Category | Features | Purpose |
|-----------------|----------|---------|
| Fleet Averages | `fleet_*_mean` (4) | Cross-server baseline |
| Fleet Variability | `fleet_*_std` (4) | Detect synchronized changes |
| Fleet Peaks | `fleet_*_max` (4) | Track fleet-wide extremes |
| Stress Indicators | `fleet_pct_high_*` (3) | % of fleet in stress |
| Server Deviation | `*_vs_fleet` (3) | Individual vs fleet |

### Server Profiles (Transfer Learning)

The model uses 7 server profiles for transfer learning:

| Profile | Prefix | Characteristics |
|---------|--------|-----------------|
| Database | `ppdb` | High memory, I/O heavy |
| ML Compute | `ppml` | GPU utilization, batch patterns |
| Web API | `ppapi` | Request patterns, connection pools |
| Conductor | `ppcond` | Orchestration, periodic jobs |
| ETL/Ingest | `ppetl` | Batch processing, overnight runs |
| Risk Analytics | `pprisk` | Compute-heavy, market hours |
| Generic | other | General workload |

---

## Risk Scoring

Predictions are converted to actionable risk scores:

```
TFT Predictions (96 timesteps × 5 targets)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Risk Score Calculation                                          │
│                                                                  │
│  1. Forecast analysis:                                          │
│     - Peak values across forecast horizon                        │
│     - Rate of change (acceleration)                              │
│     - Time to threshold breach                                   │
│                                                                  │
│  2. Multi-metric aggregation:                                   │
│     - CPU (weight: 0.25)                                        │
│     - Memory (weight: 0.30)                                     │
│     - Disk (weight: 0.20)                                       │
│     - Network (weight: 0.15)                                    │
│     - Connections (weight: 0.10)                                │
│                                                                  │
│  3. Profile adjustment:                                         │
│     - Database servers: higher memory weight                     │
│     - ML servers: higher CPU/GPU weight                         │
│                                                                  │
│  4. CASCADE BONUS (NEW):                                        │
│     - If cascade detected: +20 to all affected servers          │
│     - Elevates "green" servers in cascade to "warning"          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Alert Levels                                                    │
│                                                                  │
│  Score 80-100: CRITICAL (Red)    - Immediate action required     │
│  Score 60-79:  WARNING (Orange)  - Investigate soon              │
│  Score 50-59:  DEGRADED (Yellow) - Monitor closely               │
│  Score 0-49:   HEALTHY (Green)   - Normal operation              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Architecture

### REST Endpoints

```
┌─────────────────────────────────────────────────────────────────┐
│  Tachyon Argus API (FastAPI)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Core Endpoints:                                                 │
│    GET  /health              - Health check (no auth)            │
│    GET  /status              - Daemon status (no auth)           │
│    POST /feed/data           - Feed metrics data                 │
│    GET  /predictions/current - Get all predictions               │
│    GET  /alerts/active       - Get active alerts                 │
│    GET  /explain/{server}    - XAI explanation                   │
│                                                                  │
│  Cascade & Drift Endpoints (NEW):                                │
│    GET  /cascade/status      - Cascade detection status          │
│    GET  /cascade/health      - Fleet health (correlation-based)  │
│    GET  /drift/status        - Model drift metrics               │
│    GET  /drift/report        - Human-readable drift report       │
│                                                                  │
│  Historical Endpoints:                                           │
│    GET  /historical/summary  - Summary statistics                │
│    GET  /historical/alerts   - Alert history                     │
│    GET  /historical/server/{name} - Server history               │
│    GET  /historical/export/{table} - CSV export                  │
│                                                                  │
│  Admin Endpoints:                                                │
│    GET  /admin/models        - List available models             │
│    POST /admin/reload-model  - Hot reload model                  │
│    POST /admin/trigger-training - Start training                 │
│    GET  /admin/training-status - Training progress               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Authentication

API key authentication via header:
```
X-API-Key: your-api-key
```

Rate limits:
- `/feed/data`: 60/minute
- `/predictions/*`: 30/minute
- `/cascade/*`: 30/minute
- `/drift/*`: 30/minute
- `/explain/*`: 30/minute

---

## Continuous Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Continuous Learning Loop                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Metrics    │  Your monitoring system sends metrics          │
│  │   Ingestion  │                                               │
│  └──────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │    Data      │  Accumulates metrics for retraining           │
│  │   Buffer     │  60-day retention, auto-rotation              │
│  └──────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │    Drift     │───>│    Auto      │  Trigger if:              │
│  │   Monitor    │    │  Retrainer   │  - PER > 10%              │
│  └──────────────┘    └──────────────┘  - DSS > 20%              │
│                             │           - FDS > 15%              │
│                             │           - Anomaly > 5%           │
│                             ▼                                    │
│                      ┌──────────────┐                           │
│                      │   Training   │  5 epochs, incremental    │
│                      │   Pipeline   │  Non-blocking             │
│                      └──────────────┘                           │
│                             │                                    │
│                             ▼                                    │
│                      ┌──────────────┐                           │
│                      │    Model     │  Hot reload, no restart   │
│                      │   Reload     │                           │
│                      └──────────────┘                           │
│                             │                                    │
│                             ▼                                    │
│                      Model adapts to new patterns               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## XAI (Explainable AI) Components

### SHAP Explainer

Shows which features drove a prediction:
```json
{
  "top_features": [
    {"feature": "mem_pct", "importance": 0.85, "stars": "★★★★★"},
    {"feature": "fleet_pct_high_cpu", "importance": 0.72, "stars": "★★★★☆"},
    {"feature": "cpu_pct", "importance": 0.62, "stars": "★★★☆☆"}
  ]
}
```

### Attention Visualizer

Shows which time periods the model focused on:
```json
{
  "focus_periods": ["Last 30 minutes", "2 hours ago"],
  "analysis": "Model focused on recent memory spike and fleet-wide CPU increase"
}
```

### Counterfactual Generator

Shows what-if scenarios:
```json
{
  "scenarios": [
    {
      "action": "Reduce memory by 15%",
      "impact": "Risk drops from 72.5 to 45.2",
      "recommendation": "Consider restarting memory-heavy processes"
    }
  ]
}
```

---

## Storage Architecture

### Training Data

Time-partitioned Parquet files:
```
training/server_metrics_partitioned/
├── chunk_20251201_00.parquet  # 2-hour chunk
├── chunk_20251201_02.parquet
├── ...
└── chunk_manifest.json
```

### Model Artifacts

```
models/tft_model_YYYYMMDD_HHMMSS/
├── model.safetensors          # Model weights (SafeTensors format)
├── config.json                # Model configuration
├── dataset_parameters.pkl     # Data encoders (includes fleet features)
├── server_mapping.json        # Server hash mapping
└── training_info.json         # Training metadata
```

### Historical Data

Parquet-based historical store:
```
data_buffer/
├── alerts_YYYYMMDD.parquet
├── environment_YYYYMMDD.parquet
└── drift_metrics.json         # Current drift status
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| ML Framework | PyTorch 2.0+, PyTorch Forecasting |
| Model Format | SafeTensors |
| API Framework | FastAPI |
| Data Format | Parquet (Apache Arrow) |
| Dashboard | Plotly Dash |
| GPU Support | CUDA, cuDNN |
| Training | PyTorch Lightning |

---

## Security

- **API Authentication**: Key-based auth via X-API-Key header
- **Rate Limiting**: Prevents abuse (slowapi)
- **Input Validation**: Schema validation on all inputs
- **No PII**: Server names can be hashed for privacy

---

## Scalability Considerations

### Single Instance
- Handles 100+ servers on single GPU
- ~85ms inference latency
- 6000-point rolling window

### Multi-Instance (Future)
- Horizontal scaling via load balancer
- Shared model storage
- Redis for state coordination

---

## File Structure

```
Argus/
├── src/
│   ├── daemons/
│   │   ├── tft_inference_daemon.py    # Main inference service
│   │   ├── metrics_generator_daemon.py # Demo data generator
│   │   └── adaptive_retraining_daemon.py
│   ├── training/
│   │   ├── main.py                    # Training CLI
│   │   └── tft_trainer.py             # Training engine (fleet features)
│   ├── generators/
│   │   └── metrics_generator.py       # Data generation
│   └── core/
│       ├── alert_levels.py
│       ├── server_encoder.py
│       ├── data_validator.py
│       ├── gpu_profiles.py
│       ├── historical_store.py
│       ├── nordiq_metrics.py
│       ├── correlation_detector.py    # NEW: Cascade detection
│       ├── drift_monitor.py           # Drift tracking
│       ├── auto_retrainer.py          # Auto-retraining
│       ├── data_buffer.py             # Data accumulation
│       ├── config/
│       │   └── model_config.py        # Model hyperparameters
│       └── explainers/
│           ├── shap_explainer.py
│           ├── attention_visualizer.py
│           └── counterfactual_generator.py
├── models/                            # Trained models (gitignored)
├── training/                          # Training data (gitignored)
├── checkpoints/                       # Training checkpoints (gitignored)
└── data_buffer/                       # Historical data (gitignored)
```
