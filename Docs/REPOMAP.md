# Tachyon Argus - Repository Map

Complete directory structure and file descriptions for the predictive monitoring system.

## Project Root

```
MonitoringPrediction/
├── Argus/                    # Core application code
├── Docs/                     # Documentation
├── NordIQ/                   # NordIQ-specific configurations
├── environment.yml           # Conda environment specification
├── _StartHere.ipynb          # Interactive getting started notebook
└── README.md                 # Project overview
```

---

## Argus/ - Core Application

### Source Code (`src/`)

```
Argus/src/
├── core/                     # Core libraries and utilities
│   ├── config/               # Configuration modules
│   │   ├── api_config.py     # API settings and auth configuration
│   │   ├── metrics_config.py # Metric definitions and thresholds
│   │   └── model_config.py   # TFT model hyperparameters
│   │
│   ├── adapters/             # External system adapters
│   │   ├── elasticsearch_adapter.py  # Elasticsearch integration
│   │   └── mongodb_adapter.py        # MongoDB integration
│   │
│   ├── explainers/           # XAI explainability modules
│   │   ├── attention_visualizer.py   # Attention weight visualization
│   │   ├── counterfactual_generator.py # "What-if" scenario generation
│   │   └── shap_explainer.py         # SHAP value computation
│   │
│   ├── alert_levels.py       # Alert severity definitions
│   ├── auto_retrainer.py     # Automatic model retraining on drift
│   ├── constants.py          # Application constants
│   ├── correlation_detector.py # Cross-server cascading failure detection
│   ├── data_buffer.py        # Sliding window data buffer
│   ├── data_validator.py     # Input validation and sanitization
│   ├── drift_monitor.py      # Model drift detection (PER, DSS, FDS)
│   ├── gpu_profiles.py       # GPU workload profiles
│   ├── historical_store.py   # Time-series data persistence
│   ├── nordiq_metrics.py     # NordIQ metric mappings
│   ├── server_encoder.py     # Server ID encoding/decoding
│   └── server_profiles.py    # Server profile definitions
│
├── daemons/                  # Long-running services
│   ├── tft_inference_daemon.py       # Main prediction API server
│   ├── metrics_generator_daemon.py   # Demo metrics generator
│   └── adaptive_retraining_daemon.py # Standalone retraining daemon
│
├── generators/               # Data generation utilities
│   ├── demo_data_generator.py    # Training data generation
│   ├── demo_stream_generator.py  # Real-time data simulation
│   ├── metrics_generator.py      # Generic metrics generator
│   └── scenario_demo_generator.py # Scenario-based demo data
│
├── training/                 # Model training pipeline
│   ├── main.py               # CLI entry point for training
│   ├── tft_trainer.py        # TFT model trainer with fleet features
│   └── precompile.py         # Model precompilation utilities
│
└── dashboard/                # Dashboard backend (deprecated)
```

### Dashboard (`dash_*`)

```
Argus/
├── dash_app.py               # Streamlit main application
├── dash_config.py            # Dashboard configuration
├── dash_components/          # Reusable UI components
├── dash_tabs/                # Dashboard tab modules
│   ├── overview.py           # Main overview tab
│   ├── top_risks.py          # Server risk ranking
│   ├── heatmap.py            # Fleet heatmap visualization
│   ├── historical.py         # Historical trends
│   ├── insights.py           # AI-generated insights
│   ├── alerting.py           # Alert configuration
│   ├── cost_avoidance.py     # Cost impact analysis
│   ├── auto_remediation.py   # Automated remediation UI
│   ├── roadmap.py            # Feature roadmap
│   └── documentation.py      # In-app documentation
└── dash_utils/               # Dashboard utilities
    ├── api_client.py         # API communication
    ├── data_processing.py    # Data transformation
    └── performance.py        # Performance utilities
```

### Scripts and Utilities (`bin/`)

```
Argus/bin/
├── setup_api_key.bat         # Windows API key generator
├── setup_api_key.sh          # Unix API key generator
├── generate_api_key.py       # Python API key generator
├── start_all.bat             # Windows: Start all services
├── start_all.sh              # Unix: Start all services
└── weekly_retrain.sh         # Scheduled retraining script
```

### Models and Data

```
Argus/
├── models/                   # Trained model storage
│   ├── tft_model_YYYYMMDD_HHMMSS/  # Model directory
│   │   ├── model.safetensors       # Model weights
│   │   ├── config.json             # Model configuration
│   │   ├── dataset_parameters.pkl  # Data encoders
│   │   ├── server_mapping.json     # Server ID mapping
│   │   └── training_info.json      # Training metadata
│   └── streaming_checkpoint.pt     # Streaming training checkpoint
│
├── training/                 # Training data (gitignored)
│   └── server_metrics_partitioned/
│       ├── chunk_*.parquet         # Time-partitioned data
│       └── chunk_manifest.json     # Chunk metadata
│
└── checkpoints/              # Training checkpoints (gitignored)
```

---

## Docs/ - Documentation

```
Docs/
├── QUICK_START.md            # 5-minute getting started guide
├── ARCHITECTURE.md           # System architecture and design
├── TRAINING_GUIDE.md         # Model training instructions
├── DEPLOYMENT_GUIDE.md       # Production deployment guide
├── API_REFERENCE.md          # REST API documentation
├── METRICS_FEED_GUIDE.md     # Data ingestion guide
├── DASHBOARD_INTEGRATION_GUIDE.md  # Dashboard integration
├── DATA_PREPARATION_GUIDE.md # Data preparation instructions
└── REPOMAP.md                # This file
```

---

## Key Files by Function

### Prediction Pipeline

| File | Description |
|------|-------------|
| `src/daemons/tft_inference_daemon.py` | Main inference server with REST API |
| `src/core/data_buffer.py` | Sliding window buffer for time-series |
| `src/core/drift_monitor.py` | Prediction accuracy tracking |
| `src/core/correlation_detector.py` | Cascading failure detection |

### Training Pipeline

| File | Description |
|------|-------------|
| `src/training/main.py` | Training CLI (`python main.py train`) |
| `src/training/tft_trainer.py` | TFT model trainer with fleet features |
| `src/core/auto_retrainer.py` | Drift-triggered automatic retraining |
| `src/core/config/model_config.py` | Model hyperparameters |

### Configuration

| File | Description |
|------|-------------|
| `src/core/config/model_config.py` | TFT model settings (128 hidden, 8 heads) |
| `src/core/config/metrics_config.py` | Metric thresholds and names |
| `src/core/config/api_config.py` | API authentication settings |

### Explainability (XAI)

| File | Description |
|------|-------------|
| `src/core/explainers/shap_explainer.py` | SHAP value computation |
| `src/core/explainers/attention_visualizer.py` | Attention weight analysis |
| `src/core/explainers/counterfactual_generator.py` | What-if scenarios |

---

## New in v2.1

The following files were added or significantly updated for cascading failure detection:

| File | Change |
|------|--------|
| `src/core/correlation_detector.py` | **NEW** - Cross-server correlation analysis |
| `src/training/tft_trainer.py` | Added 18 fleet-level features |
| `src/core/auto_retrainer.py` | Added drift-triggered retraining |
| `src/daemons/tft_inference_daemon.py` | Added cascade/drift endpoints |
| `src/core/config/model_config.py` | Enabled multi-target prediction |

---

## Data Flow

```
                                    ┌─────────────────────┐
                                    │   Training Data     │
                                    │   (Parquet files)   │
                                    └──────────┬──────────┘
                                               │
                                               ▼
┌─────────────────┐              ┌─────────────────────────┐
│  Metrics Feed   │──────────────▶│    tft_trainer.py      │
│  (Real or Demo) │              │  + Fleet Features       │
└─────────────────┘              └──────────┬──────────────┘
        │                                   │
        │                                   ▼
        │                        ┌─────────────────────────┐
        │                        │   Trained TFT Model     │
        │                        │   (models/ directory)   │
        │                        └──────────┬──────────────┘
        │                                   │
        ▼                                   ▼
┌─────────────────────────────────────────────────────────┐
│              tft_inference_daemon.py                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Data Buffer  │  │ TFT Model    │  │ Correlation  │   │
│  │              │──▶│ Inference    │──▶│ Detector     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│          │                │                  │           │
│          ▼                ▼                  ▼           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Drift Monitor│  │ Predictions  │  │ Cascade      │   │
│  │              │  │ + Risk Scores│  │ Alerts       │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│          │                                               │
│          ▼                                               │
│  ┌──────────────┐                                        │
│  │Auto Retrainer│◀── Drift Triggered                     │
│  └──────────────┘                                        │
└─────────────────────────────────────────────────────────┘
```

---

## Getting Started

1. **New users**: Start with [QUICK_START.md](QUICK_START.md)
2. **Architecture overview**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
3. **API integration**: See [API_REFERENCE.md](API_REFERENCE.md)
4. **Training models**: Follow [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
