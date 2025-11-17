# ArgusAI Repository Map

**Version 1.1.0** | Built by Craig Giannelli and Claude Code

> Complete folder structure and navigation guide for the ArgusAI predictive monitoring system.

Last Updated: November 17, 2025 | Repository Size: 677 MB

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Total Files** | 1,200+ files |
| **Python Files** | 113 files |
| **Documentation Files** | 227 markdown files |
| **Repository Size** | 677 MB |
| **Project Version** | 1.1.0 (ArgusAI Branding) |
| **License** | Business Source License 1.1 |
| **Development Time** | 67.5 hours (AI-assisted) |

---

## Complete Directory Tree

```
MonitoringPrediction/
│
├── 📋 Core Project Files
│   ├── README.md                          # Main project documentation
│   ├── LICENSE                            # BSL 1.1 license
│   ├── VERSION                            # Current version (1.1.0)
│   ├── CHANGELOG.md                       # Version history and changes
│   ├── REPOMAP.md                         # This file
│   ├── .gitignore                         # Git exclusions
│   ├── .gitattributes                     # Git attributes
│   ├── .env                               # Environment configuration
│   ├── .env.example                       # Environment template
│   │
│   ├── environment.yml                     # Conda environment spec
│   ├── humanizer.py                       # AI text humanization utility
│   ├── _StartHere.ipynb                   # Interactive workflow notebook
│   ├── TFT_Presentation.pptx              # Presentation materials
│   └── DOCUMENTATION_CONSOLIDATION_PLAN.md # Documentation restructuring plan
│
├── 🎯 NordIQ/ (MAIN APPLICATION)
│   │
│   ├── 🚀 Startup Scripts
│   │   ├── start_all.bat                  # Windows: Start all services
│   │   ├── start_all.sh                   # Linux/Mac: Start all services
│   │   ├── stop_all.bat                   # Windows: Stop all services
│   │   ├── stop_all.sh                    # Linux/Mac: Stop all services
│   │   ├── README.md                      # Deployment guide
│   │   ├── GETTING_STARTED.md             # Quick start guide
│   │   └── COMMIT_SUMMARY.md              # Recent changes
│   │
│   ├── 📦 bin/ (Utilities)
│   │   ├── generate_api_key.py            # API key generation
│   │   ├── setup_api_key.bat              # Windows setup
│   │   └── setup_api_key.sh               # Linux/Mac setup
│   │
│   ├── 💻 src/ (Source Code)
│   │   │
│   │   ├── 🤖 daemons/ (Background Services)
│   │   │   ├── tft_inference_daemon.py    # Production inference server
│   │   │   ├── metrics_generator_daemon.py # Demo data generator
│   │   │   ├── adaptive_retraining_daemon.py # Auto-retraining service
│   │   │   ├── checkpoints/               # Training checkpoints
│   │   │   ├── logs/                      # Service logs
│   │   │   ├── models/                    # Model symlinks
│   │   │   ├── plots/                     # Generated plots
│   │   │   └── training/                  # Training data
│   │   │
│   │   ├── 🎨 dashboard/ (Web Interface)
│   │   │   ├── tft_dashboard_web.py       # Main Dash application
│   │   │   ├── __init__.py
│   │   │   └── Dashboard/                 # Modular components
│   │   │       ├── __init__.py
│   │   │       ├── config/                # Dashboard configuration
│   │   │       │   ├── __init__.py
│   │   │       │   └── dashboard_config.py
│   │   │       ├── tabs/                  # Dashboard tabs
│   │   │       │   ├── __init__.py
│   │   │       │   ├── overview.py        # Fleet overview
│   │   │       │   ├── heatmap.py         # Server heatmap
│   │   │       │   ├── top_risks.py       # Top problem servers
│   │   │       │   ├── historical.py      # Historical trends
│   │   │       │   ├── advanced.py        # Advanced features
│   │   │       │   ├── alerting.py        # Alert configuration
│   │   │       │   ├── auto_remediation.py # Auto-remediation
│   │   │       │   ├── cost_avoidance.py  # Cost analysis
│   │   │       │   ├── roadmap.py         # Product roadmap
│   │   │       │   └── documentation.py   # In-app docs
│   │   │       └── utils/                 # Dashboard utilities
│   │   │           ├── __init__.py
│   │   │           ├── api_client.py      # API integration
│   │   │           ├── metrics.py         # Metrics helpers
│   │   │           ├── profiles.py        # Profile utilities
│   │   │           └── risk_scoring.py    # Risk calculations
│   │   │
│   │   ├── 🧠 training/ (Model Training)
│   │   │   ├── main.py                    # Training CLI interface
│   │   │   ├── tft_trainer.py             # Training engine
│   │   │   ├── precompile.py              # Bytecode optimization
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📊 generators/ (Data Generation)
│   │   │   ├── metrics_generator.py       # Realistic metrics generator
│   │   │   └── __init__.py
│   │   │
│   │   └── 🔧 core/ (Shared Libraries)
│   │       ├── __init__.py
│   │       ├── linborg_schema.py          # Data schema
│   │       │
│   │       ├── config/                    # Configuration
│   │       │   ├── __init__.py
│   │       │   ├── model_config.py        # Model hyperparameters
│   │       │   ├── metrics_config.py      # Server profiles
│   │       │   └── api_config.py          # API settings
│   │       │
│   │       ├── adapters/                  # Production adapters
│   │       │   ├── mongodb_adapter.py     # MongoDB integration
│   │       │   ├── elasticsearch_adapter.py # Elasticsearch integration
│   │       │   ├── mongodb_adapter_config.json.template
│   │       │   ├── elasticsearch_adapter_config.json.template
│   │       │   └── requirements.txt
│   │       │
│   │       ├── utils/                     # Core utilities
│   │       │   └── (various utility files)
│   │       │
│   │       └── explainers/                # XAI components
│   │           └── (explainability modules)
│   │
│   ├── 📁 data/ (Runtime Data)
│   │   ├── training/                      # Training datasets
│   │   │   └── *.parquet                  # Parquet data files
│   │   └── data_buffer/                   # Temporary buffers
│   │
│   ├── 🧪 models/ (Trained Models)
│   │   ├── tft_model_20251013_100205/     # Training session 1
│   │   │   ├── model.safetensors          # Model weights
│   │   │   ├── config.json                # Model config
│   │   │   ├── dataset_parameters.pkl     # Encoders (CRITICAL!)
│   │   │   ├── server_mapping.json        # Server hash mapping
│   │   │   └── training_info.json         # Training metadata
│   │   ├── tft_model_20251014_131232/     # Training session 2
│   │   ├── tft_model_20251015_080653/     # Training session 3
│   │   └── tft_model_20251017_122454/     # Latest model
│   │
│   ├── ⚡ lightning_logs/ (Training Logs)
│   │   ├── version_0/                     # Training run 0
│   │   │   ├── events.out.tfevents.*      # TensorBoard events
│   │   │   └── hparams.yaml               # Hyperparameters
│   │   ├── version_1/
│   │   ├── ... (version_2 through version_733)
│   │   └── version_733/                   # Latest training run
│   │
│   ├── 📝 logs/ (Application Logs)
│   │   └── *.log                          # Service logs
│   │
│   ├── 📊 plots/ (Generated Visualizations)
│   │   └── *.png                          # Training plots
│   │
│   ├── 💾 checkpoints/ (Training Checkpoints)
│   │   └── *.ckpt                         # Model checkpoints
│   │
│   ├── 🎭 Dash POC Files (Experimental)
│   │   ├── dash_app.py                    # POC Dash app
│   │   ├── dash_config.py                 # POC configuration
│   │   ├── dash_poc.py                    # POC prototype
│   │   ├── dash_poc_requirements.txt      # POC dependencies
│   │   ├── dash_components/               # POC components
│   │   ├── dash_tabs/                     # POC tabs
│   │   └── dash_utils/                    # POC utilities
│   │
│   ├── 📚 Docs/ (NordIQ Documentation)
│   │   ├── README.md                      # Documentation index
│   │   ├── GETTING_STARTED.md             # Quick start
│   │   ├── AUTOMATED_RETRAINING.md        # ⭐ Retraining system
│   │   ├── CONFIGURABLE_REFRESH_INTERVAL.md
│   │   ├── DEMO_CONTROLS_ADDED.md
│   │   ├── HOT_MODEL_RELOAD.md
│   │   ├── INSIGHTS_TAB_OPTIMIZATION.md
│   │   ├── PRODUCTION_INTEGRATION.md
│   │   ├── WELLS_FARGO_BRANDING.md
│   │   ├── WHAT_IF_SCENARIOS_IMPROVEMENTS.md
│   │   ├── XAI_TAB_LOADING_FIX.md
│   │   │
│   │   ├── understanding/                 # Conceptual guides
│   │   │   ├── ALERT_LEVELS.md
│   │   │   ├── CONTEXTUAL_RISK_INTELLIGENCE.md
│   │   │   ├── HOW_PREDICTIONS_WORK.md
│   │   │   ├── SERVER_PROFILES.md
│   │   │   └── WHY_TFT.md
│   │   │
│   │   ├── for-developers/                # Developer docs
│   │   │   ├── ADAPTER_QUICK_REFERENCE.md
│   │   │   ├── API_REFERENCE.md
│   │   │   ├── DATA_ADAPTER_GUIDE.md
│   │   │   └── DATA_FORMAT_SPEC.md
│   │   │
│   │   ├── for-production/                # Production guides
│   │   │   ├── ELASTICSEARCH_INTEGRATION.md
│   │   │   └── MONGODB_INTEGRATION.md
│   │   │
│   │   ├── for-business-intelligence/     # BI integration
│   │   │   └── GRAFANA_INTEGRATION.md
│   │   │
│   │   ├── authentication/                # Security docs
│   │   │   ├── AUTHENTICATION_IMPLEMENTATION_GUIDE.md
│   │   │   └── OKTA_SSO_INTEGRATION.md
│   │   │
│   │   ├── operations/                    # Operations guides
│   │   │   ├── DAEMON_MANAGEMENT.md
│   │   │   └── ARCHIVE_INFERENCE_README.md
│   │   │
│   │   ├── marketing/                     # Business docs
│   │   │   ├── CUSTOMER_BRANDING_GUIDE.md
│   │   │   ├── FUTURE_ROADMAP.md
│   │   │   ├── MANAGED_HOSTING_ECONOMICS.md
│   │   │   └── PROJECT_SUMMARY.md
│   │   │
│   │   ├── RAG/                           # AI assistant context
│   │   │   ├── SESSION_2025-10-29_DASH_MIGRATION_COMPLETE.md
│   │   │   ├── SESSION_2025-10-29_DASH_MIGRATION_WEEK2.md
│   │   │   ├── SESSION_2025-10-30_AUTOMATED_RETRAINING.md
│   │   │   └── SESSION_2025-10-30_FORCLAUDE_PACKAGE.md
│   │   │
│   │   └── archive/                       # Historical docs
│   │       ├── DATA_INGESTION_GUIDE.md
│   │       ├── REAL_DATA_INTEGRATION.md
│   │       └── getting-started/
│   │           ├── API_KEY_SETUP.md
│   │           ├── PYTHON_ENV.md
│   │           └── QUICK_START.md
│   │
│   ├── 📦 forclaude/ (Wells Fargo Integration Package) ⭐
│   │   ├── 00_READ_ME_FIRST.md            # Start here
│   │   ├── 01_QUICK_START.md              # 5-minute setup
│   │   ├── 02_API_CONTRACT.md             # API specification
│   │   ├── 03_MINIMAL_TEMPLATE.py         # Code template
│   │   ├── 04_TESTING_GUIDE.md            # Testing guide
│   │   ├── 05_SUMMARY_FOR_CLAUDE.md       # AI assistant summary
│   │   ├── FOR_WELLS_FARGO_AI_ENGINEERS.md # Wells Fargo guide
│   │   ├── README.md                      # Package overview
│   │   └── UPLOAD_THESE_FILES.txt         # File checklist
│   │
│   ├── .streamlit/                        # Streamlit config
│   └── .nordiq_key                        # API key (gitignored)
│
├── 📚 Docs/ (ROOT DOCUMENTATION) ⭐ NEW CONSOLIDATED STRUCTURE
│   │
│   ├── 🎯 Essential Guides (READ THESE FIRST)
│   │   ├── INDEX.md                       # Documentation index
│   │   ├── README.md                      # Documentation overview
│   │   ├── QUICKSTART.md                  # 30-second setup
│   │   ├── ARCHITECTURE_GUIDE.md          # ⭐ System architecture
│   │   ├── TRAINING_GUIDE.md              # ⭐ Training & retraining
│   │   ├── PERFORMANCE_COMPLETE.md        # ⭐ Performance optimization
│   │   ├── HANDOFF_SUMMARY.md             # Team handoff
│   │   └── CONTRIBUTING.md                # Contribution guide
│   │
│   ├── 📖 Core Documentation
│   │   ├── AUTOMATED_RETRAINING.md        # Auto-retraining system
│   │   ├── RETRAINING_PIPELINE.md         # Operational procedures
│   │   ├── STREAMLIT_ARCHITECTURE_AND_DATA_FLOW.md
│   │   ├── STREAMLIT_PERFORMANCE_OPTIMIZATION.md
│   │   ├── UNKNOWN_SERVER_HANDLING.md     # Hash encoding
│   │   ├── SPARSE_DATA_HANDLING.md        # Offline servers
│   │   ├── SMART_CACHE_STRATEGY.md        # Caching design
│   │   ├── SCRIPT_DEPRECATION_ANALYSIS.md
│   │   └── VERSION_HISTORY.md             # Version changelog
│   │
│   ├── 🎨 Performance & Optimization
│   │   ├── PERFORMANCE_COMPLETE.md        # ⭐ Complete guide
│   │   ├── PERFORMANCE_OPTIMIZATIONS_APPLIED.md
│   │   ├── PHASE_3_OPTIMIZATIONS_APPLIED.md
│   │   ├── PHASE_4_OPTIMIZATIONS_COMPLETE.md
│   │   └── COLOR_AUDIT_2025-10-18.md
│   │
│   ├── 🔍 Development References
│   │   ├── HUMAN_TODO_CHECKLIST.md        # Development tasks
│   │   ├── HUMAN_VS_AI_TIMELINE.md        # Development velocity
│   │   └── XAI_POLISH_CHECKLIST.md        # XAI improvements
│   │
│   ├── 🤖 RAG/ (For AI Assistants)
│   │   ├── README.md                      # RAG overview
│   │   ├── COMPLETE_HISTORY.md            # Full project history
│   │   ├── CURRENT_STATE.md               # ⭐ Current status
│   │   ├── PROJECT_CODEX.md               # ⭐ Development rules
│   │   ├── QUICK_START_NEXT_SESSION.md    # Session startup
│   │   ├── CLAUDE_SESSION_GUIDELINES.md   # AI guidelines
│   │   └── TIME_TRACKING.md               # Development hours
│   │
│   └── 📦 archive/ (Historical Documentation)
│       ├── REPOMAP.md                     # Previous repo map
│       ├── README.md                      # Archive index
│       ├── (70+ archived session docs)
│       ├── merged/                        # Merged into guides
│       │   ├── ADAPTER_ARCHITECTURE.md
│       │   ├── ADAPTIVE_RETRAINING_PLAN.md
│       │   ├── CONTINUOUS_LEARNING_PLAN.md
│       │   ├── DASHBOARD_PERFORMANCE_OPTIMIZATIONS.md
│       │   ├── DATA_CONTRACT.md
│       │   ├── FRAMEWORK_MIGRATION_ANALYSIS.md
│       │   ├── GPU_AUTO_CONFIGURATION.md
│       │   ├── MODEL_TRAINING_GUIDELINES.md
│       │   └── PERFORMANCE_OPTIMIZATION.md
│       └── sessions/                      # Session summaries
│           ├── SESSION_2025-10-17_*.md
│           ├── SESSION_2025-10-18_*.md
│           ├── SESSION_2025-10-19_*.md
│           ├── SESSION_2025-10-24_*.md
│           ├── SESSION_2025-10-29_*.md
│           └── SESSION_2025-10-30_*.md
│
├── 🏢 BusinessPlanning/ (CONFIDENTIAL - gitignored)
│   ├── README.md                          # Business overview
│   ├── BANK_PARTNERSHIP_PROPOSAL.md
│   ├── BUSINESS_NAME_IDEAS.md
│   ├── BUSINESS_STRATEGY.md
│   ├── CONFIDENTIAL_README.md
│   ├── CONSULTING_SERVICES_TEMPLATE.md
│   ├── DEVELOPMENT_TIMELINE_ANALYSIS.md
│   ├── DUAL_ROLE_STRATEGY.md
│   ├── FINAL_NAME_RECOMMENDATIONS.md
│   ├── IP_OWNERSHIP_EVIDENCE.md
│   ├── NEXT_STEPS_ACTION_PLAN.md
│   ├── NORDIQ_BRANDING_ANALYSIS.md
│   ├── NORDIQ_LAUNCH_CHECKLIST.md
│   ├── NORDIQ_WEBSITE_STRATEGY.md
│   └── TRADEMARK_ANALYSIS.md
│
├── 📦 Archive/ (Legacy Code)
│   └── Streamlit_Dashboard_Original/      # Original Streamlit implementation
│       ├── README.md
│       ├── tft_dashboard_web.py
│       └── Dashboard/
│           ├── __init__.py
│           ├── config/                    # Configuration
│           ├── tabs/                      # Dashboard tabs
│           └── utils/                     # Utilities
│
├── 🛠️ scripts/ (Development Scripts)
│   ├── install_security_deps.bat          # Windows security setup
│   ├── install_security_deps.sh           # Linux/Mac security setup
│   └── deprecated/                        # Deprecated scripts
│       ├── README.md
│       ├── validation/                    # Validation scripts
│       └── security/                      # Security scripts
│
└── 🔧 .claude/ (Claude Code Configuration)
    └── settings.local.json                # Local settings
```

---

## Key Entry Points

### For End Users
1. **NordIQ/README.md** - Deployment guide
2. **NordIQ/start_all.bat/sh** - One-command startup
3. **Dashboard:** http://localhost:8501 (after startup)
4. **API:** http://localhost:8000 (after startup)

### For Developers
1. **Docs/ARCHITECTURE_GUIDE.md** - ⭐ System architecture
2. **Docs/TRAINING_GUIDE.md** - ⭐ Training workflows
3. **Docs/RAG/PROJECT_CODEX.md** - Development rules
4. **NordIQ/src/** - Source code
5. **Docs/RAG/CURRENT_STATE.md** - Current project state

### For DevOps
1. **Docs/PERFORMANCE_COMPLETE.md** - ⭐ Performance guide
2. **NordIQ/Docs/operations/DAEMON_MANAGEMENT.md** - Service management
3. **NordIQ/src/core/adapters/** - Production adapters
4. **NordIQ/Docs/for-production/** - Production integration

### For AI Assistants
1. **Docs/RAG/PROJECT_CODEX.md** - ⭐ Development rules
2. **Docs/RAG/CURRENT_STATE.md** - ⭐ Current status
3. **Docs/RAG/QUICK_START_NEXT_SESSION.md** - Session startup
4. **Docs/RAG/CLAUDE_SESSION_GUIDELINES.md** - AI guidelines

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     ArgusAI System                      │
│             Predictive Infrastructure Monitoring         │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Metrics     │   │   Training   │   │  Inference   │
│  Generator   │   │   Pipeline   │   │   Daemon     │
│              │   │              │   │              │
│ Generate     │   │ TFT Model    │   │ REST API     │
│ synthetic    │   │ Training     │   │ WebSocket    │
│ data         │   │              │   │ Predictions  │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │  Dashboard   │
                                      │              │
                                      │ Plotly Dash  │
                                      │ Web UI       │
                                      └──────────────┘
```

### Data Flow

```
Production Logs → Adapters → Inference Daemon → API → Dashboard
                     ↓             ↓
              Buffer Queue    Predictions
                               Risk Scores
                                Alerts
```

---

## Technology Stack

### Core Technologies
- **Python 3.10+** - Primary language
- **PyTorch 2.0+** - Deep learning framework
- **PyTorch Forecasting** - TFT implementation
- **Plotly Dash** - Web dashboard framework
- **FastAPI** - REST API framework (in inference daemon)

### Data & Storage
- **Parquet** - Training data format (38x faster than JSON)
- **SafeTensors** - Model weight storage
- **MongoDB** - Production metrics (optional adapter)
- **Elasticsearch** - Production metrics (optional adapter)

### Machine Learning
- **Temporal Fusion Transformer (TFT)** - Prediction model
- **Transfer Learning** - 7 server profiles
- **Attention Mechanism** - Time-series analysis
- **CUDA/cuDNN** - GPU acceleration

### Development Tools
- **Conda** - Environment management
- **Git** - Version control
- **Claude Code** - AI-assisted development
- **Jupyter** - Interactive notebooks

---

## Performance Highlights

### Training Performance
- **30 days of data:** ~30 minutes on RTX 4090
- **Model size:** 88K parameters
- **Training runs:** 733+ versions tracked
- **Optimization:** Bytecode precompilation for 2-5x speedup

### Inference Performance
- **Latency:** <100ms per server prediction
- **Throughput:** 90 servers in ~85ms
- **Load time:** Dashboard loads in 2-3 seconds
- **Cache hit rate:** 98%+ on repeated queries

### Data Loading
- **Parquet vs JSON:** 38x faster (1.8s vs 68.7s for 30 days)
- **Memory efficiency:** 70% reduction with Parquet
- **Streaming:** Real-time WebSocket updates

---

## Key Features

### 1. Predictive Monitoring
- 8-hour advance warning of incidents
- 88% accuracy on critical failures
- Multi-factor risk scoring
- Contextual pattern recognition

### 2. Transfer Learning
- 7 server profiles (ML, DB, Web, etc.)
- Instant predictions for new servers
- Zero-shot learning capability
- 80% reduction in retraining frequency

### 3. Production Ready
- REST API + WebSocket support
- MongoDB/Elasticsearch adapters
- API key authentication
- Hot model reloading

### 4. Interactive Dashboard
- Real-time fleet monitoring
- Server heatmap visualization
- Risk trending and alerts
- Demo scenario controls

### 5. Auto-Retraining
- Drift detection
- Scheduled retraining
- Model A/B testing
- Performance monitoring

---

## Recent Changes (v1.1.0)

### Branding
- ✅ Rebranded to **ArgusAI** from TFT Monitoring
- ✅ New logo and color scheme
- ✅ Updated all documentation
- ✅ Professional marketing materials

### Framework Migration
- ✅ Dashboard migrated from Streamlit to Plotly Dash
- ✅ 2-5x performance improvement
- ✅ Better component architecture
- ✅ Enhanced caching strategy

### Documentation Consolidation
- ✅ Created **ARCHITECTURE_GUIDE.md** ⭐
- ✅ Created **TRAINING_GUIDE.md** ⭐
- ✅ Created **PERFORMANCE_COMPLETE.md** ⭐
- ✅ Archived 70+ historical documents
- ✅ Organized into clear categories

### Automated Retraining
- ✅ Drift detection system
- ✅ Scheduled retraining pipeline
- ✅ Performance monitoring
- ✅ Model versioning

### Wells Fargo Integration
- ✅ Created **forclaude/** package
- ✅ 5-minute integration guide
- ✅ API contract documentation
- ✅ Minimal code templates

---

## Consolidated Documentation Map

### Before (70+ scattered docs)
```
Docs/
├── SESSION_2025-10-10_SUMMARY.md
├── SESSION_2025-10-11_SUMMARY.md
├── ALL_PHASES_COMPLETE.md
├── DATA_LOADING_IMPROVEMENTS.md
├── BUGFIX_8_SERVER_LIMIT.md
├── CLEANUP_COMPLETE.md
├── ... (60+ more files)
```

### After (3 comprehensive guides) ⭐
```
Docs/
├── ARCHITECTURE_GUIDE.md          # Complete system design
├── TRAINING_GUIDE.md              # Complete training workflows
├── PERFORMANCE_COMPLETE.md        # Complete optimization guide
└── archive/                       # Historical docs
    ├── merged/                    # Source material
    └── sessions/                  # Session notes
```

### What Was Merged

**Into ARCHITECTURE_GUIDE.md:**
- ADAPTER_ARCHITECTURE.md
- DATA_CONTRACT.md
- GPU_AUTO_CONFIGURATION.md
- Deployment guides
- Microservices design

**Into TRAINING_GUIDE.md:**
- MODEL_TRAINING_GUIDELINES.md
- ADAPTIVE_RETRAINING_PLAN.md
- CONTINUOUS_LEARNING_PLAN.md
- Training best practices

**Into PERFORMANCE_COMPLETE.md:**
- PERFORMANCE_OPTIMIZATION.md
- DASHBOARD_PERFORMANCE_OPTIMIZATIONS.md
- FRAMEWORK_MIGRATION_ANALYSIS.md
- All optimization sessions

---

## Archive Structure

### What's in Archive/
1. **Streamlit_Dashboard_Original/** - Original Streamlit implementation before Dash migration
2. **scripts/deprecated/** - Deprecated validation and security scripts

### What's in Docs/archive/
1. **merged/** - Source docs that were consolidated into guides
2. **sessions/** - Historical session summaries (70+ files)
3. Individual archived docs (certification reports, completion docs, etc.)

**Archive Policy:**
- ✅ Keep if: Still referenced, contains unique info, operational value
- ❌ Archive if: >1 week old session notes, completed phases, superseded by newer docs

---

## Project Statistics

### Code Metrics
- **Total Python Files:** 113
- **Lines of Code:** ~15,000+ (estimated)
- **Core Modules:** 20+ modules
- **Dashboard Tabs:** 11 tabs
- **API Endpoints:** 15+ endpoints

### Documentation Metrics
- **Total Markdown Files:** 227
- **Active Documentation:** 30+ files
- **Archived Documentation:** 70+ files
- **Consolidated Guides:** 3 comprehensive guides
- **Total Documentation Pages:** 200+ pages (estimated)

### Training Metrics
- **Training Runs:** 733+ versions
- **Model Checkpoints:** 4 major versions
- **Lightning Logs:** 733 training sessions
- **Total Training Time:** 20-30 hours (cumulative)

### Development Metrics
- **Total Development Time:** 67.5 hours
- **AI-Assisted Ratio:** ~80%
- **Productivity Multiplier:** 10-20x vs traditional development
- **Git Commits:** 100+ commits

---

## File Size Breakdown

```
Repository Total: 677 MB

├── lightning_logs/          ~200 MB (733 training versions)
├── models/                  ~150 MB (4 trained models)
├── NordIQ/data/             ~100 MB (training datasets)
├── .git/                    ~50 MB (version history)
├── Archive/                 ~30 MB (legacy code)
├── Docs/                    ~20 MB (documentation)
├── BusinessPlanning/        ~5 MB (business docs)
├── scripts/                 ~2 MB (utility scripts)
└── Other files              ~120 MB (misc)
```

---

## Branding Information

### ArgusAI Identity
- **Full Name:** ArgusAI
- **Tagline:** "Predictive Infrastructure Monitoring"
- **Logo:** Argus (many-eyed giant from Greek mythology)
- **Theme:** Vigilance, foresight, comprehensive monitoring

### Brand Colors
- **Primary:** Deep blue (#1f3a93)
- **Secondary:** Electric blue (#00d4ff)
- **Accent:** Orange/Gold (#ff9500)
- **Background:** Dark theme (#0e1117)

### Visual Identity
- Multiple "eyes" representing comprehensive monitoring
- Future-focused, AI-powered aesthetic
- Professional enterprise branding
- Greek mythology connection (Argus Panoptes)

---

## Quick Commands Reference

### Start System
```bash
cd NordIQ
./start_all.sh        # Linux/Mac
start_all.bat         # Windows
```

### Training
```bash
cd NordIQ
python src/training/main.py generate --servers 20 --hours 720
python src/training/main.py train --epochs 20
python src/training/main.py status
```

### API Testing
```bash
curl http://localhost:8000/health
curl http://localhost:8000/predictions/current
curl http://localhost:8000/status
```

### Stop System
```bash
cd NordIQ
./stop_all.sh         # Linux/Mac
stop_all.bat          # Windows
```

---

## Navigation Tips

### Finding Specific Information

**Need to understand the system?**
→ Read `Docs/ARCHITECTURE_GUIDE.md`

**Need to train a model?**
→ Read `Docs/TRAINING_GUIDE.md`

**Need to optimize performance?**
→ Read `Docs/PERFORMANCE_COMPLETE.md`

**Need to deploy?**
→ Read `NordIQ/README.md`

**Need API docs?**
→ Read `NordIQ/Docs/for-developers/API_REFERENCE.md`

**Need production integration?**
→ Read `NordIQ/Docs/for-production/` guides

**Need business context?**
→ Read `BusinessPlanning/` (if you have access)

### Documentation Hierarchy

```
1. README.md (this file)           # Project overview
2. NordIQ/README.md                # Deployment guide
3. Docs/ARCHITECTURE_GUIDE.md      # Technical deep dive
4. Docs/TRAINING_GUIDE.md          # Training workflows
5. Docs/PERFORMANCE_COMPLETE.md    # Optimization guide
6. Docs/RAG/PROJECT_CODEX.md       # Development rules
7. NordIQ/Docs/*/                  # Specific topics
```

---

## Version History

### v1.1.0 (November 2025) - ArgusAI Branding
- Rebranded from TFT Monitoring to ArgusAI
- Massive documentation consolidation
- Framework migration (Streamlit → Plotly Dash)
- Business Source License 1.1 adoption
- Wells Fargo integration package

### v1.0.0 (October 2025) - Production Release
- Complete TFT model implementation
- 7 server profile system
- Dashboard with 11 tabs
- REST API + WebSocket
- MongoDB/Elasticsearch adapters
- Automated retraining pipeline

### Pre-v1.0 (September-October 2025) - Development
- Initial prototype
- Data contract system
- Hash-based server encoding
- Demo scenarios
- 67.5 hours of AI-assisted development

---

## Contributing

See `Docs/CONTRIBUTING.md` for contribution guidelines.

**Key areas for contribution:**
- Additional server profiles
- New dashboard visualizations
- Performance optimizations
- Documentation improvements
- Integration adapters
- Testing coverage

---

## License

Business Source License 1.1 (BSL 1.1)

- Free for non-production use
- Free for internal production use
- Requires commercial license for hosted/SaaS offerings
- Converts to Apache 2.0 license after 2 years

See `LICENSE` file for full details.

---

## Credits

**Built by:**
- **Craig Giannelli** - System architect, domain expert, product vision
- **Claude Code** - AI-assisted development, documentation, optimization

**Special Thanks:**
- PyTorch Forecasting team
- Plotly Dash team
- Research community (TFT paper authors)

---

## Support & Contact

**Documentation Issues:**
- Check `Docs/INDEX.md` for navigation
- Review `Docs/RAG/PROJECT_CODEX.md` for development rules

**Technical Issues:**
- Check `NordIQ/README.md` troubleshooting section
- Review `Docs/ARCHITECTURE_GUIDE.md` for design details

**Business Inquiries:**
- See `BusinessPlanning/` (confidential access required)

---

**Last Updated:** November 17, 2025
**Maintainer:** Craig Giannelli
**Status:** Production Ready

---

🔮 **Predict the Future. Prevent the Outage. Protect the Business.**

Built with AI + Coffee + Vibe Coding ⚡
