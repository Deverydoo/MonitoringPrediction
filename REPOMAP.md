# Tachyon Argus Repository Map

**Version 2.0.0** | Built by Craig Giannelli and Claude Code

> Complete folder structure and navigation guide for the Tachyon Argus predictive monitoring system.

Last Updated: December 11, 2025 | Repository Size: ~1.5 MB (compressed)

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Python Files** | ~60 active files |
| **Documentation Files** | ~30 active docs |
| **Repository Size** | ~1.5 MB (zipped) |
| **Project Version** | 2.0.0 (Tachyon Argus) |
| **License** | Business Source License 1.1 |

---

## Complete Directory Tree

```
MonitoringPrediction/
│
├── Core Project Files
│   ├── README.md                          # Main project documentation
│   ├── LICENSE                            # BSL 1.1 license
│   ├── VERSION                            # Current version
│   ├── CHANGELOG.md                       # Version history
│   ├── REPOMAP.md                         # This file
│   ├── .gitignore                         # Git exclusions
│   ├── environment.yml                    # Conda environment spec
│   ├── humanizer.py                       # AI text humanization utility
│   └── _StartHere.ipynb                   # Interactive workflow notebook
│
├── Argus/ (MAIN APPLICATION)
│   │
│   ├── Startup Scripts
│   │   ├── start_all.bat / start_all.sh  # Start all services
│   │   ├── stop_all.bat / stop_all.sh    # Stop all services
│   │   ├── daemon.bat / daemon.sh        # Run inference daemon only
│   │   ├── status.sh                     # Check service status
│   │   ├── README.md                     # Deployment guide
│   │   ├── GETTING_STARTED.md            # Quick start guide
│   │   ├── QUICK_START.md                # 5-minute setup
│   │   └── REQUIREMENTS.md               # Dependencies
│   │
│   ├── bin/ (Utilities)
│   │   ├── generate_api_key.py           # API key generation
│   │   ├── setup_api_key.bat / .sh       # API key setup scripts
│   │   ├── run_daemon.bat                # Windows daemon launcher
│   │   └── weekly_retrain.sh             # Scheduled retraining
│   │
│   ├── src/ (Source Code)
│   │   │
│   │   ├── daemons/ (Background Services)
│   │   │   ├── tft_inference_daemon.py   # Main inference server (REST API)
│   │   │   ├── metrics_generator_daemon.py # Demo data generator
│   │   │   └── adaptive_retraining_daemon.py # Auto-retraining service
│   │   │
│   │   ├── training/ (Model Training)
│   │   │   ├── main.py                   # Training CLI interface
│   │   │   ├── tft_trainer.py            # Training engine + streaming + checkpoints
│   │   │   └── precompile.py             # Bytecode optimization
│   │   │
│   │   ├── generators/ (Data Generation)
│   │   │   ├── metrics_generator.py      # Realistic metrics generator
│   │   │   ├── demo_data_generator.py    # Demo data for testing
│   │   │   ├── demo_stream_generator.py  # Streaming demo data
│   │   │   └── scenario_demo_generator.py # Scenario-based demos
│   │   │
│   │   └── core/ (Shared Libraries)
│   │       ├── alert_levels.py           # Alert thresholds and colors
│   │       ├── auto_retrainer.py         # Automated retraining logic
│   │       ├── constants.py              # Global constants
│   │       ├── data_buffer.py            # Data accumulation for training
│   │       ├── data_validator.py         # Schema validation (v2.0.0)
│   │       ├── drift_monitor.py          # Model drift detection
│   │       ├── gpu_profiles.py           # GPU configuration
│   │       ├── historical_store.py       # Historical data storage
│   │       ├── nordiq_metrics.py         # Metrics definitions
│   │       ├── server_encoder.py         # Server ID hashing
│   │       ├── server_profiles.py        # Server type profiles
│   │       │
│   │       ├── config/                   # Configuration modules
│   │       │   ├── api_config.py         # API settings
│   │       │   ├── metrics_config.py     # Metrics configuration
│   │       │   └── model_config.py       # Model hyperparameters
│   │       │
│   │       ├── adapters/                 # Production data adapters
│   │       │   ├── mongodb_adapter.py    # MongoDB integration
│   │       │   ├── elasticsearch_adapter.py # Elasticsearch integration
│   │       │   └── README.md             # Adapter documentation
│   │       │
│   │       └── explainers/               # XAI components
│   │           ├── shap_explainer.py     # SHAP feature importance
│   │           ├── attention_visualizer.py # Attention analysis
│   │           └── counterfactual_generator.py # What-if scenarios
│   │
│   ├── Dashboard (Plotly Dash)
│   │   ├── dash_app.py                   # Main Dash application
│   │   ├── dash_config.py                # Dashboard configuration
│   │   │
│   │   ├── dash_tabs/                    # Dashboard tab modules
│   │   │   ├── overview.py               # Fleet overview
│   │   │   ├── heatmap.py                # Server heatmap
│   │   │   ├── top_risks.py              # Top risk servers
│   │   │   ├── historical.py             # Historical trends
│   │   │   ├── insights.py               # XAI insights
│   │   │   ├── alerting.py               # Alert configuration
│   │   │   ├── auto_remediation.py       # Auto-remediation
│   │   │   ├── cost_avoidance.py         # Cost analysis
│   │   │   ├── roadmap.py                # Product roadmap
│   │   │   └── documentation.py          # In-app docs
│   │   │
│   │   ├── dash_utils/                   # Dashboard utilities
│   │   │   ├── api_client.py             # API integration
│   │   │   ├── data_processing.py        # Data transformation
│   │   │   └── performance.py            # Caching & performance
│   │   │
│   │   └── dash_components/              # Reusable components
│   │
│   ├── training/ (GITIGNORED - Generated)
│   │   └── server_metrics_partitioned/   # Time-chunked parquet data
│   │
│   ├── models/ (GITIGNORED - Generated)
│   │   └── tft_model_YYYYMMDD_HHMMSS/    # Trained model artifacts
│   │       ├── model.safetensors         # Model weights
│   │       ├── config.json               # Model config
│   │       ├── dataset_parameters.pkl    # Encoders
│   │       └── server_mapping.json       # Server hash mapping
│   │
│   ├── checkpoints/ (GITIGNORED - Generated)
│   │   └── streaming_checkpoint.pt       # Training checkpoint for resume
│   │
│   └── data_buffer/ (GITIGNORED - Generated)
│       └── *.parquet                     # Accumulated metrics for retraining
│
├── Docs/ (Documentation)
│   ├── CONTRIBUTING.md                   # Contribution guidelines
│   ├── DASHBOARD_INTEGRATION_GUIDE.md    # Dashboard API guide
│   ├── METRICS_FEED_GUIDE.md             # Metrics ingestion guide
│   │
│   └── archive/ (Historical docs - gitignored)
│       └── *.md                          # Archived session notes
│
├── scripts/ (Development Scripts)
│   ├── install_security_deps.bat / .sh   # Security setup
│   └── deprecated/                       # Deprecated scripts
│       ├── README.md
│       ├── validation/                   # Old validation scripts
│       └── security/                     # Old security scripts
│
└── BusinessPlanning/ (GITIGNORED - Confidential)
    └── *.md                              # Business strategy docs
```

---

## Key Entry Points

### For End Users
1. **Argus/QUICK_START.md** - 5-minute setup
2. **Argus/start_all.bat/sh** - One-command startup
3. **Dashboard:** http://localhost:8501 (after startup)
4. **API:** http://localhost:8000 (after startup)

### For Developers
1. **Docs/DASHBOARD_INTEGRATION_GUIDE.md** - Build dashboards
2. **Docs/METRICS_FEED_GUIDE.md** - Feed data to the engine
3. **Argus/src/daemons/tft_inference_daemon.py** - Main inference code
4. **Argus/src/training/tft_trainer.py** - Training engine

### For DevOps
1. **Argus/README.md** - Deployment guide
2. **Argus/src/core/adapters/** - Production adapters
3. **Argus/bin/weekly_retrain.sh** - Scheduled retraining

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Tachyon Argus System                   │
│            Predictive Infrastructure Monitoring          │
└─────────────────────────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Metrics    │   │   Training   │   │  Inference   │
│  Generator   │   │   Pipeline   │   │   Daemon     │
│              │   │              │   │              │
│ POST /feed   │   │ Streaming    │   │ REST API     │
│ → daemon     │   │ + Checkpoint │   │ Port 8000    │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │  Dashboard   │
                                      │  (Optional)  │
                                      │              │
                                      │ Plotly Dash  │
                                      │ Port 8501    │
                                      └──────────────┘
```

### Data Flow

```
Your Monitoring System
        │
        ▼ POST /feed/data
┌──────────────────────────────────────────────────────────┐
│              TFT Inference Daemon (Port 8000)            │
│                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│  │  Rolling   │ →  │    TFT     │ →  │   Risk     │     │
│  │  Window    │    │   Model    │    │  Scoring   │     │
│  │ (2880 pts) │    │ Prediction │    │  + Alerts  │     │
│  └────────────┘    └────────────┘    └────────────┘     │
│                                                          │
│  Endpoints:                                              │
│  - GET /predictions/current  → Server predictions       │
│  - GET /alerts/active        → Active alerts            │
│  - GET /explain/{server}     → XAI explanations         │
│  - GET /historical/*         → Historical data          │
│  - POST /admin/trigger-training → Manual retraining     │
└──────────────────────────────────────────────────────────┘
        │
        ▼ Your Dashboard
┌──────────────────────────────────────────────────────────┐
│  Any Dashboard Framework (React, Vue, Angular, etc.)     │
│  or the built-in Plotly Dash dashboard                   │
└──────────────────────────────────────────────────────────┘
```

---

## API Quick Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (no auth) |
| GET | `/status` | Daemon status (no auth) |
| POST | `/feed/data` | Feed metrics data |
| GET | `/predictions/current` | Get all predictions |
| GET | `/alerts/active` | Get active alerts |
| GET | `/explain/{server}` | XAI explanation |

### Historical Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/historical/summary` | Summary stats |
| GET | `/historical/alerts` | Alert history |
| GET | `/historical/server/{name}` | Server history |
| GET | `/historical/export/{table}` | CSV export |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/models` | List models |
| POST | `/admin/reload-model` | Hot reload model |
| POST | `/admin/trigger-training` | Start training |
| GET | `/admin/training-status` | Training progress |

See **Docs/DASHBOARD_INTEGRATION_GUIDE.md** for full API documentation.

---

## Technology Stack

### Core
- **Python 3.10+** - Primary language
- **PyTorch 2.0+** - Deep learning
- **PyTorch Forecasting** - TFT model
- **FastAPI** - REST API
- **Plotly Dash** - Dashboard

### Data
- **Parquet** - Training data (time-partitioned)
- **SafeTensors** - Model weights
- **Pickle** - Dataset parameters

### ML Features
- **Temporal Fusion Transformer (TFT)** - 8-hour forecasting
- **Transfer Learning** - 7 server profiles
- **Streaming Training** - Memory-efficient (2-hour chunks)
- **Checkpoint Resume** - Training resiliency

---

## Training Features

### Streaming Training
- Processes large datasets in 2-hour chunks
- Memory-efficient (~4 min per chunk)
- Suitable for weeks/months of data

### Checkpoint Support
- Saves every 5 chunks (~20 min intervals)
- Auto-resumes on process restart
- Stores: model weights, epoch, chunk index, loss

### Data Schema (v2.0.0)
16 required fields per record:
- `timestamp`, `server_name`, `status`
- CPU: `cpu_user_pct`, `cpu_sys_pct`, `cpu_iowait_pct`, `cpu_idle_pct`, `java_cpu_pct`
- Memory: `mem_used_pct`, `swap_used_pct`
- Disk: `disk_usage_pct`
- Network: `net_in_mb_s`, `net_out_mb_s`
- Connections: `back_close_wait`, `front_close_wait`
- System: `load_average`, `uptime_days`

See **Docs/METRICS_FEED_GUIDE.md** for complete schema.

---

## Quick Commands

### Start System
```bash
cd Argus
./start_all.sh        # Linux/Mac
start_all.bat         # Windows
```

### Start Inference Only
```bash
cd Argus
python src/daemons/tft_inference_daemon.py --port 8000
```

### Training
```bash
cd Argus
# Generate training data
python src/training/main.py generate --servers 45 --hours 336

# Train model (streaming for large datasets)
python src/training/main.py train --streaming --epochs 5

# Check training status
python src/training/main.py status
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Get predictions (requires API key)
curl -H "X-API-Key: your-key" http://localhost:8000/predictions/current

# Feed data
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"records": [...]}'
```

### Stop System
```bash
cd Argus
./stop_all.sh         # Linux/Mac
stop_all.bat          # Windows
```

---

## Gitignored Directories

These directories contain generated/large files and are not tracked:

| Directory | Content | Regenerate With |
|-----------|---------|-----------------|
| `Argus/training/` | Partitioned parquet data | `main.py generate` |
| `Argus/models/` | Trained model weights | `main.py train` |
| `Argus/checkpoints/` | Training checkpoints | Auto-created |
| `Argus/data_buffer/` | Accumulated metrics | Auto-created |
| `Docs/archive/` | Historical docs | N/A |
| `BusinessPlanning/` | Confidential docs | N/A |

---

## Server Profiles

| Prefix | Profile | Typical Workload |
|--------|---------|------------------|
| `ppdb` | Database | PostgreSQL, MySQL |
| `ppml` | ML Compute | Model training/inference |
| `ppapi` | Web API | REST/GraphQL servers |
| `ppcond` | Conductor | Orchestration services |
| `ppetl` | ETL/Ingest | Data pipelines |
| `pprisk` | Risk Analytics | Financial compute |
| Other | Generic | Unspecified |

---

## Alert Levels

| Level | Risk Score | Color | Action |
|-------|------------|-------|--------|
| Critical | 80-100 | Red | Immediate |
| Warning | 60-79 | Orange | Investigate |
| Degraded | 50-59 | Yellow | Monitor |
| Healthy | 0-49 | Green | Normal |

---

## Version History

### v2.0.0 (December 2025)
- Streaming training with 2-hour chunks
- Checkpoint support for training resiliency
- Repository cleanup (1.2GB → 1.5MB)
- New documentation guides

### v1.1.0 (November 2025)
- Rebranded to Tachyon Argus
- Dashboard migrated to Plotly Dash
- Automated retraining pipeline

### v1.0.0 (October 2025)
- Initial production release
- TFT model with 7 server profiles
- REST API + Dashboard

---

## License

Business Source License 1.1 (BSL 1.1)

- Free for non-production use
- Free for internal production use
- Commercial license required for SaaS
- Converts to Apache 2.0 after 2 years

---

## Credits

**Built by:**
- **Craig Giannelli** - System architect, product vision
- **Claude Code** - AI-assisted development

---

**Last Updated:** December 11, 2025
