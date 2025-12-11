# Tachyon Argus - Architecture Overview

Technical architecture and design of the predictive infrastructure monitoring system.

## System Overview

Tachyon Argus is a machine learning system that predicts server failures 8 hours in advance using a Temporal Fusion Transformer (TFT) model.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tachyon Argus System                        │
│               Predictive Infrastructure Monitoring               │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Training    │    │   Inference   │    │   Dashboard   │
│   Pipeline    │    │    Daemon     │    │   (Optional)  │
│               │    │               │    │               │
│ - Data gen    │    │ - REST API    │    │ - Plotly Dash │
│ - TFT model   │    │ - Predictions │    │ - Real-time   │
│ - Streaming   │    │ - XAI/SHAP    │    │ - Alerts      │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Core Components

### 1. Inference Daemon (`tft_inference_daemon.py`)

The production service that makes predictions.

**Responsibilities:**
- Accept metrics via REST API
- Maintain rolling window of recent data
- Run TFT model predictions
- Calculate risk scores and alerts
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
│  │  (2880 pts)  │    │  (88K params)│    │  + Alerts    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ↑                                       │                │
│         │                                       ▼                │
│  POST /feed/data                      GET /predictions/current   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │     XAI      │    │  Historical  │    │    Auto      │       │
│  │  Explainers  │    │    Store     │    │  Retraining  │       │
│  │ SHAP/Attn/CF │    │  (Parquet)   │    │  (Optional)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Rolling Window**: Maintains last 2880 data points (~24 hours for 45 servers)
- **Risk Scoring**: Converts predictions to 0-100 risk scores
- **XAI Explainability**: SHAP, Attention visualization, Counterfactuals
- **Hot Reload**: Swap models without restart

### 2. Training Pipeline (`tft_trainer.py`)

Trains the TFT prediction model.

**Training Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| Standard | Load all data into memory | Small datasets |
| Streaming | Process 2-hour chunks | Large datasets |
| Incremental | Continue from existing model | Add more epochs |

**Streaming Training Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Streaming Training                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  chunk_01.parquet -> [Train] -> [Free Memory]                   │
│                          ↓                                       │
│  chunk_02.parquet -> [Train] -> [Free Memory]                   │
│                          ↓                                       │
│  chunk_03.parquet -> [Train] -> [Free Memory]                   │
│                          ↓                                       │
│        ...              ...          ...                         │
│                          ↓                                       │
│  chunk_N.parquet  -> [Train] -> [Save Model]                    │
│                                                                  │
│  Checkpoint saved every 5 chunks for resume capability          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Data Generator (`metrics_generator.py`)

Generates realistic synthetic server metrics.

**Features:**
- 7 server profiles (Database, ML, API, ETL, etc.)
- Realistic workload patterns (daily cycles, spikes)
- Failure scenarios for training
- Time-partitioned Parquet output

### 4. Core Libraries (`src/core/`)

Shared utilities used across components:

| Module | Purpose |
|--------|---------|
| `alert_levels.py` | Alert thresholds and colors |
| `server_encoder.py` | Hash-based server ID encoding |
| `data_validator.py` | Schema validation |
| `gpu_profiles.py` | GPU detection and configuration |
| `historical_store.py` | Parquet-based data storage |
| `nordiq_metrics.py` | Metrics definitions |

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
        │
        ▼
┌─────────────────┐
│ models/         │
│ ├── model.safetensors │
│ ├── config.json       │
│ └── dataset_params.pkl│
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
│  2. Add to rolling window (2880 points max)                     │
│  3. Run TFT model prediction (96 steps = 8 hours)               │
│  4. Calculate risk scores (0-100)                               │
│  5. Determine alert levels (critical/warning/degraded/healthy)  │
│  6. Store in historical data (Parquet)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼ GET /predictions/current
┌─────────────────────────────────────────────────────────────────┐
│  Response:                                                       │
│  {                                                               │
│    "predictions": {                                              │
│      "server001": {                                              │
│        "risk_score": 72.5,                                       │
│        "alert": {"level": "warning", "color": "#FFA500"},        │
│        "forecast": {"cpu_pct": [...], "mem_pct": [...]}          │
│      }                                                           │
│    },                                                            │
│    "summary": {"critical": 2, "warning": 5, "healthy": 38}       │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## TFT Model Architecture

The Temporal Fusion Transformer (TFT) is designed for multi-horizon forecasting with interpretability.

### Key Components

1. **Variable Selection Networks**: Learn which input features are important
2. **LSTM Encoder/Decoder**: Capture temporal patterns
3. **Multi-Head Attention**: Focus on relevant time periods
4. **Quantile Outputs**: Predict confidence intervals

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Hidden Size | 64 |
| Attention Heads | 4 |
| LSTM Layers | 2 |
| Dropout | 0.1 |
| Input Window | 96 steps (8 hours) |
| Prediction Horizon | 96 steps (8 hours) |
| Total Parameters | ~88K |

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
TFT Predictions (96 timesteps)
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
- `/explain/*`: 30/minute

---

## XAI (Explainable AI) Components

### SHAP Explainer

Shows which features drove a prediction:
```json
{
  "top_features": [
    {"feature": "mem_pct", "importance": 0.85, "stars": "★★★★★"},
    {"feature": "cpu_pct", "importance": 0.62, "stars": "★★★☆☆"}
  ]
}
```

### Attention Visualizer

Shows which time periods the model focused on:
```json
{
  "focus_periods": ["Last 30 minutes", "2 hours ago"],
  "analysis": "Model focused on recent memory spike"
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
├── dataset_parameters.pkl     # Data encoders (CRITICAL)
├── server_mapping.json        # Server hash mapping
└── training_info.json         # Training metadata
```

### Historical Data

Parquet-based historical store:
```
data_buffer/
├── alerts_YYYYMMDD.parquet
└── environment_YYYYMMDD.parquet
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
- 2880-point rolling window

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
│   │   └── tft_trainer.py             # Training engine
│   ├── generators/
│   │   └── metrics_generator.py       # Data generation
│   └── core/
│       ├── alert_levels.py
│       ├── server_encoder.py
│       ├── data_validator.py
│       ├── gpu_profiles.py
│       ├── historical_store.py
│       └── explainers/
│           ├── shap_explainer.py
│           ├── attention_visualizer.py
│           └── counterfactual_generator.py
├── models/                            # Trained models (gitignored)
├── training/                          # Training data (gitignored)
├── checkpoints/                       # Training checkpoints (gitignored)
└── data_buffer/                       # Historical data (gitignored)
```
