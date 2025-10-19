# NordIQ Repository Map

**Version:** 1.0.0
**Date:** 2025-10-19
**Purpose:** Comprehensive catalog of all repository files, their status, and cleanup recommendations
**Total Files:** ~295 (145 code files + 150 documentation)

---

## 🎯 Executive Summary

### Critical Finding: Severe Duplication Between Root and NordIQ/

**The repository has TWO complete copies of the application:**

1. **Root Directory** (Legacy) - Development/prototyping version
2. **NordIQ/** (Production) - Clean, organized, deployable version

**Duplication Issues:**
- 23+ Python files duplicated (500+ KB of duplicate code)
- 20+ scripts duplicated (.bat/.sh files)
- 4 trained models duplicated (500+ MB each = 2+ GB waste)
- Different versions (NordIQ is newer and has bug fixes from Oct 18)
- High risk of editing wrong version

**Recommended Action:** Archive root files, use NordIQ/ as primary.

---

## 📊 Repository Statistics

### File Counts by Type
```
Python Files:        86 files
Batch/Shell Scripts: 38 files
Markdown Docs:       95 files
JSON Config:         21 files
HTML/CSS/JS:         8 files (website)
Other:              47 files
------------------------
TOTAL:              295 files
```

### Size Breakdown
```
Source Code:         ~2.5 MB
Documentation:       ~1.8 MB
Trained Models:      ~4.2 GB (duplicated!)
Total Repository:    ~4.2 GB
```

### Code Distribution
```
NordIQ/ (Production):    ~1.2 MB (organized)
Root (Legacy):          ~1.3 MB (scattered)
Duplication Waste:      ~500 KB
```

---

## 🗂️ Directory Structure Overview

```
MonitoringPrediction/
├── NordIQ/                    # ✅ PRODUCTION - Self-contained application
│   ├── bin/                   # Utilities (API keys, helpers)
│   ├── src/                   # Application source code
│   │   ├── core/              # Shared libraries
│   │   ├── daemons/           # Services (inference, metrics)
│   │   ├── dashboard/         # Streamlit web UI
│   │   ├── generators/        # Data/demo generators
│   │   └── training/          # Model training
│   ├── models/                # Trained TFT models (DUPLICATE)
│   ├── data/                  # Runtime data
│   ├── logs/                  # Log files
│   ├── .streamlit/            # Streamlit config
│   └── start_all.bat/sh       # One-command startup
│
├── NordIQ-Website/            # ✅ ACTIVE - Business website (6 pages)
│   ├── index.html             # Landing page
│   ├── product.html           # Product showcase
│   ├── about.html             # Company info
│   ├── how-it-works.html      # Technical overview
│   ├── pricing.html           # Pricing tiers
│   ├── contact.html           # Contact form
│   └── css/main.css           # Styles
│
├── Docs/                      # ✅ ACTIVE - Technical documentation
│   ├── RAG/                   # AI assistant context (5 files)
│   ├── archive/               # Historical docs (89 files)
│   └── *.md                   # Active guides (40+ files)
│
├── BusinessPlanning/          # ✅ ACTIVE - Confidential business docs
│   └── *.md                   # 14 business strategy files
│
├── [ROOT FILES]               # ⚠️ LEGACY - Scattered duplicate files
│   ├── *.py                   # 23 Python files (DUPLICATES)
│   ├── *.bat/*.sh             # 20 scripts (DUPLICATES)
│   ├── Dashboard/             # Old dashboard structure
│   ├── config/                # Old config structure
│   ├── adapters/              # Old adapters
│   ├── explainers/            # Old explainers
│   └── models/                # Trained models (DUPLICATE)
│
├── scripts/deprecated/        # ⚠️ DEPRECATED - Old validation scripts
└── config_archive/            # ⚠️ DEPRECATED - Old config system
```

---

## 🔴 CRITICAL: Duplicate Files Analysis

### Core Application Files (Root → NordIQ/)

| File | Root Size | NordIQ Size | Status | Action |
|------|-----------|-------------|--------|--------|
| `tft_inference_daemon.py` | 62 KB | 82 KB | NordIQ newer (Oct 18) | DELETE root |
| `metrics_generator_daemon.py` | 26 KB | 26 KB | Same | DELETE root |
| `tft_dashboard_web.py` | 21 KB | 23 KB | NordIQ newer | DELETE root |
| `main.py` | 12 KB | 13 KB | NordIQ newer | DELETE root |
| `metrics_generator.py` | 47 KB | → generators/ | Moved | DELETE root |
| `server_encoder.py` | 10 KB | → core/ | Moved | DELETE root |
| `data_validator.py` | 15 KB | → core/ | Moved | DELETE root |
| `constants.py` | 9.6 KB | → core/ | Moved | DELETE root |
| `drift_monitor.py` | 15 KB | → core/ | Moved | DELETE root |
| `gpu_profiles.py` | 11 KB | → core/ | Moved | DELETE root |
| `server_profiles.py` | 7.6 KB | → core/ | Moved | DELETE root |
| `data_buffer.py` | 12 KB | → core/ | Moved | DELETE root |

**Total Duplicate Code:** ~500 KB
**Risk:** Extremely high - editing wrong version causes bugs

### Duplicate Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `start_all.bat/sh` | Start all services | Both exist |
| `stop_all.bat/sh` | Stop all services | Both exist |
| `setup_api_key.bat/sh` | API key setup | Both exist |
| `run_daemon.bat` | Run daemon | Root only |
| `start_all_corp.bat/sh` | Corporate launcher | Root only (obsolete?) |

**Action:** Keep NordIQ/ versions, delete root versions

### Duplicate Trained Models (2.1 GB Each!)

```
models/tft_model_20251013_100205/  (Root)
models/tft_model_20251014_131232/  (Root)
models/tft_model_20251015_080653/  (Root)
models/tft_model_20251017_122454/  (Root)

NordIQ/models/tft_model_20251013_100205/  (DUPLICATE)
NordIQ/models/tft_model_20251014_131232/  (DUPLICATE)
NordIQ/models/tft_model_20251015_080653/  (DUPLICATE)
NordIQ/models/tft_model_20251017_122454/  (DUPLICATE)
```

**Total Wasted Space:** ~8.4 GB (4 models × 2 copies)
**Action:** Delete root `models/` directory, keep only NordIQ/models/

### Duplicate Directories

| Directory | Root | NordIQ | Status |
|-----------|------|--------|--------|
| `Dashboard/` | Old structure | Moved to src/dashboard/ | DELETE root |
| `config/` | Old structure | Moved to src/core/config/ | ARCHIVE root |
| `adapters/` | Old structure | Moved to src/core/adapters/ | DELETE root |
| `explainers/` | Old structure | Moved to src/core/explainers/ | DELETE root |

---

## 📁 Detailed File Inventory

### NordIQ/ (Production Application) - ✅ KEEP

#### NordIQ/bin/ (Utilities)
```
generate_api_key.py         6.4 KB   ✅ API key management
run_daemon.bat              345 B    ✅ Daemon helper
setup_api_key.bat/sh        1.2 KB   ✅ API key setup
```

#### NordIQ/src/core/ (Shared Libraries)
```
__init__.py                 -        ✅ Package init
_path_setup.py             -        ✅ Path configuration
alert_levels.py            -        ✅ Alert severity definitions
constants.py               9.6 KB   ✅ System constants
data_buffer.py             12 KB    ✅ Data buffering
data_validator.py          15 KB    ✅ Schema validation
drift_monitor.py           15 KB    ✅ Model drift detection
gpu_profiles.py            11 KB    ✅ GPU configuration
nordiq_metrics.py          -        ✅ Metrics definitions
server_encoder.py          10 KB    ✅ Server name encoding
server_profiles.py         7.6 KB   ✅ Server profile definitions
```

#### NordIQ/src/core/config/ (Configuration)
```
__init__.py                -        ✅ Config package
api_config.py              -        ✅ API settings
metrics_config.py          -        ✅ Metrics definitions
model_config.py            -        ✅ Model parameters
```

#### NordIQ/src/core/adapters/ (Production Data Integration)
```
__init__.py                -        ✅ Adapters package
elasticsearch_adapter.py   -        ✅ Elasticsearch connector
mongodb_adapter.py         -        ✅ MongoDB connector
README.md                  -        ✅ Adapter documentation
requirements.txt           -        ✅ Adapter dependencies
```

#### NordIQ/src/core/explainers/ (XAI - Future)
```
__init__.py                -        🔮 Package init
attention_visualizer.py    -        🔮 Attention weights visualization
counterfactual_generator.py -       🔮 What-if analysis
shap_explainer.py          -        🔮 SHAP explanations
```

#### NordIQ/src/daemons/ (Services)
```
__init__.py                -        ✅ Daemons package
tft_inference_daemon.py    82 KB   ✅ Inference server (REST/WS)
metrics_generator_daemon.py 26 KB  ✅ Metrics simulation
adaptive_retraining_daemon.py 16 KB 🔮 Auto-retraining (future)
```

#### NordIQ/src/dashboard/ (Web UI)
```
tft_dashboard_web.py       23 KB   ✅ Main dashboard orchestration
Dashboard/
├── config/
│   └── dashboard_config.py 217 KB  ✅ Dashboard configuration
├── utils/
│   ├── api_client.py      64 B    ✅ Daemon API client
│   ├── metrics.py         185 B   ✅ Metrics extraction
│   ├── profiles.py        27 B    ✅ Profile utilities
│   └── risk_scoring.py    169 B   ✅ Risk calculation
└── tabs/
    ├── overview.py        577 B   ✅ Main dashboard tab
    ├── heatmap.py         155 B   ✅ Fleet heatmap
    ├── top_risks.py       218 B   ✅ Top 5 servers
    ├── historical.py      134 B   ✅ Trend analysis
    ├── cost_avoidance.py  192 B   ✅ ROI calculations
    ├── auto_remediation.py 192 B  ✅ Remediation suggestions
    ├── alerting.py        236 B   ✅ Alert routing
    ├── insights.py        -       ✅ XAI insights (new)
    ├── advanced.py        89 B    ✅ Diagnostics
    ├── documentation.py   542 B   ✅ User guide
    └── roadmap.py         278 B   ✅ Future vision
```

#### NordIQ/src/generators/ (Data Generation)
```
__init__.py                -        ✅ Generators package
metrics_generator.py       47 KB   ✅ Metrics generation library
demo_data_generator.py     23 KB   ✅ Training data generator
demo_stream_generator.py   19 KB   ✅ Live stream simulator
scenario_demo_generator.py 15 KB   ✅ Scenario-based demos
```

#### NordIQ/src/training/ (Model Training)
```
__init__.py                -        ✅ Training package
main.py                    13 KB   ✅ Training CLI
tft_trainer.py             40 KB   ✅ TFT model trainer
precompile.py              1.6 KB  ✅ Torch compilation
```

#### NordIQ/models/ (Trained Models)
```
tft_model_20251013_100205/  ~500 MB  ✅ 3-epoch demo model
tft_model_20251014_131232/  ~500 MB  ⚠️ Old version
tft_model_20251015_080653/  ~600 MB  ⚠️ Old version
tft_model_20251017_122454/  ~600 MB  ✅ Latest model
```

**Recommendation:** Keep only the latest 2 models, delete older versions

---

### Root Directory Files - ⚠️ LEGACY

#### Root Python Files (23 files) - DELETE ALL
```
adaptive_retraining_daemon.py   16 KB   ❌ DUPLICATE (→ NordIQ/src/daemons/)
constants.py                    9.6 KB  ❌ DUPLICATE (→ NordIQ/src/core/)
data_buffer.py                  12 KB   ❌ DUPLICATE (→ NordIQ/src/core/)
data_validator.py               15 KB   ❌ DUPLICATE (→ NordIQ/src/core/)
demo_data_generator.py          23 KB   ❌ DUPLICATE (→ NordIQ/src/generators/)
demo_stream_generator.py        19 KB   ❌ DUPLICATE (→ NordIQ/src/generators/)
drift_monitor.py                15 KB   ❌ DUPLICATE (→ NordIQ/src/core/)
generate_api_key.py             6.4 KB  ❌ DUPLICATE (→ NordIQ/bin/)
gpu_profiles.py                 11 KB   ❌ DUPLICATE (→ NordIQ/src/core/)
linborg_schema.py               6.9 KB  ⚠️ OLD SCHEMA (deprecated)
main.py                         12 KB   ❌ DUPLICATE (→ NordIQ/src/training/)
metrics_generator.py            47 KB   ❌ DUPLICATE (→ NordIQ/src/generators/)
metrics_generator_daemon.py     26 KB   ❌ DUPLICATE (→ NordIQ/src/daemons/)
precompile.py                   1.6 KB  ❌ DUPLICATE (→ NordIQ/src/training/)
production_metrics_forwarder_TEMPLATE.py 17 KB ⚠️ TEMPLATE (keep?)
run_demo.py                     4.5 KB  ⚠️ ONE-OFF SCRIPT
scenario_demo_generator.py      15 KB   ❌ DUPLICATE (→ NordIQ/src/generators/)
server_encoder.py               10 KB   ❌ DUPLICATE (→ NordIQ/src/core/)
server_profiles.py              7.6 KB  ❌ DUPLICATE (→ NordIQ/src/core/)
tft_dashboard.py                34 KB   ❌ OLD DASHBOARD (replaced)
tft_dashboard_web.py            21 KB   ❌ DUPLICATE (→ NordIQ/src/dashboard/)
tft_inference.py                59 KB   ❌ OLD VERSION (replaced with daemon)
tft_inference_daemon.py         62 KB   ❌ DUPLICATE (→ NordIQ/src/daemons/)
tft_trainer.py                  40 KB   ❌ DUPLICATE (→ NordIQ/src/training/)
```

#### Root Scripts (20 files) - DELETE MOST
```
install_security_deps.bat/sh    1.3 KB  ⚠️ KEEP (setup script)
run_certification.bat           424 B   ❌ ONE-OFF
run_daemon.bat                  345 B   ❌ DUPLICATE
setup_api_key.bat/sh            1.2 KB  ❌ DUPLICATE
start_all.bat/sh                3-6 KB  ❌ DUPLICATE
start_all_corp.bat/sh           7-9 KB  ⚠️ CORPORATE VERSION (obsolete?)
start_dashboard_corporate.bat   2.3 KB  ⚠️ CORPORATE VERSION (obsolete?)
stop_all.bat/sh                 1-5 KB  ❌ DUPLICATE
test_env.bat                    213 B   ⚠️ KEEP (testing)
validate_pipeline.bat           2.7 KB  ❌ ONE-OFF
validate_schema.bat             80 B    ❌ ONE-OFF
```

#### Root Directories - CLEAN UP
```
Dashboard/                  ❌ DELETE (moved to NordIQ/src/dashboard/)
config/                     ⚠️ ARCHIVE (old config system)
config_archive/             ⚠️ ALREADY ARCHIVED
adapters/                   ❌ DELETE (moved to NordIQ/src/core/adapters/)
explainers/                 ❌ DELETE (moved to NordIQ/src/core/explainers/)
models/                     ❌ DELETE (duplicate, 4.2 GB waste)
tabs/                       ❌ DELETE (old tab structure)
utils/                      ❌ DELETE (old utils)
checkpoints/                ⚠️ KEEP (training checkpoints)
lightning_logs/             ⚠️ KEEP (training logs)
training/                   ⚠️ KEEP (training data)
logs/                       ⚠️ KEEP (log files)
data_buffer/                ⚠️ KEEP (buffer data)
plots/                      ⚠️ KEEP (analysis plots)
scripts/deprecated/         ⚠️ ALREADY DEPRECATED
init.d/                     ⚠️ SYSTEMD (production deployment)
systemd/                    ⚠️ SYSTEMD (production deployment)
```

#### Root Data Files
```
inference_rolling_window.parquet  235 KB   ⚠️ KEEP (rolling data)
inference_rolling_window.pkl      1.7 MB   ❌ DELETE (old format)
TFT_Presentation.pptx             654 KB   ⚠️ KEEP (presentation)
environment.yml                   1.2 KB   ⚠️ KEEP (conda env)
kill_daemon.ps1                   377 B    ⚠️ KEEP (utility)
nul                               844 B    ❌ DELETE (error artifact)
training.gitkeep                  0 B      ⚠️ KEEP (git placeholder)
```

---

### Docs/ (Documentation) - ✅ ACTIVE

#### Docs/RAG/ (AI Context - 7 files) - ✅ KEEP ALL
```
CURRENT_STATE.md                  570 lines  ✅ Single source of truth
PROJECT_CODEX.md                  844 lines  ✅ Development rules
CLAUDE_SESSION_GUIDELINES.md      430 lines  ✅ Session protocol
MODULAR_REFACTOR_COMPLETE.md      262 lines  ✅ Architecture details
QUICK_START_NEXT_SESSION.md       256 lines  ✅ Quick start
TIME_TRACKING.md                  201 lines  ✅ Development timeline
README.md                         119 lines  ✅ RAG folder guide
SESSION_2025-10-17_SUMMARY.md     -          ✅ Recent session
SESSION_2025-10-18_DEBUGGING.md   -          ✅ Debugging session
SESSION_2025-10-18_PERFORMANCE_OPTIMIZATION.md ~7000 lines ✅ Performance work
SESSION_2025-10-18_PICKUP.md      -          ✅ Session recovery
SESSION_2025-10-18_SUMMARY.md     -          ✅ Session summary
SESSION_2025-10-18_WEBSITE.md     -          ✅ Website build
```

#### Docs/ (Technical Docs - 40+ files) - ✅ KEEP MOST
```
ADAPTER_ARCHITECTURE.md           ✅ Production integration
ADAPTIVE_RETRAINING_PLAN.md       ✅ Auto-retraining design
ALERT_LEVELS.md                   ✅ Alert severity definitions
API_KEY_SETUP.md                  ✅ Authentication guide
AUTHENTICATION_IMPLEMENTATION_GUIDE.md ✅ Auth options
AUTOMATED_RETRAINING.md           ✅ Retraining pipeline
COLOR_AUDIT_2025-10-18.md         ✅ Color consistency
COMPLETE_OPTIMIZATION_SUMMARY.md  ✅ Performance summary
CONTEXTUAL_RISK_INTELLIGENCE.md   ✅ Risk scoring philosophy
CONTINUOUS_LEARNING_PLAN.md       ✅ Online learning design
CONTRIBUTING.md                   ✅ Contribution guide
DAEMON_SHOULD_DO_HEAVY_LIFTING.md ✅ Architectural analysis
DASHBOARD_PERFORMANCE_OPTIMIZATIONS.md ✅ Performance guide
DATA_CONTRACT.md                  ✅ Schema specification
FUTURE_DASHBOARD_MIGRATION.md     ⚠️ Future plans
FUTURE_ROADMAP.md                 ✅ Product roadmap
GPU_AUTO_CONFIGURATION.md         ✅ GPU setup
HANDOFF_SUMMARY.md                ✅ Team handoff
HOW_PREDICTIONS_WORK.md           ✅ Prediction explanation
HUMAN_TODO_CHECKLIST.md           ⚠️ Task list
HUMAN_VS_AI_TIMELINE.md           ✅ Development comparison
INDEX.md                          ✅ Documentation index
INFERENCE_README.md               ✅ Inference guide
MANAGED_HOSTING_ECONOMICS.md      ✅ Hosting analysis
MODEL_TRAINING_GUIDELINES.md      ✅ Training guide
OKTA_SSO_INTEGRATION.md           ✅ SSO setup
PERFORMANCE_OPTIMIZATION.md       ✅ Performance guide
PRODUCTION_DATA_ADAPTERS.md       ✅ Adapter guide
PRODUCTION_INTEGRATION_GUIDE.md   ✅ Production integration
PROJECT_SUMMARY.md                ✅ Project overview
PYTHON_ENV.md                     ✅ Environment setup
QUICK_REFERENCE_API.md            ✅ API reference
QUICK_START.md                    ✅ Quick start
QUICKSTART.md                     ⚠️ DUPLICATE of QUICK_START.md?
README.md                         ✅ Main documentation
RETRAINING_PIPELINE.md            ✅ Retraining design
SCRIPT_DEPRECATION_ANALYSIS.md    ✅ Script cleanup analysis
SERVER_PROFILES.md                ✅ Profile system
SESSION_2025-10-17_FINAL_SUMMARY.md ✅ Session summary
SMART_CACHE_STRATEGY.md           ✅ Caching design
SPARSE_DATA_HANDLING.md           ✅ Data handling
STREAMLIT_ARCHITECTURE_AND_DATA_FLOW.md ✅ Dashboard architecture
UNKNOWN_SERVER_HANDLING.md        ✅ Unknown server logic
VERSION_HISTORY.md                ✅ Version changelog
WHY_TFT.md                        ✅ Model selection
XAI_POLISH_CHECKLIST.md           ✅ XAI implementation
```

#### Docs/archive/ (89 files) - ⚠️ ARCHIVE COMPLETE
```
SESSION_*.md                      ~50 files  ⚠️ Historical sessions
*_COMPLETE.md                     ~15 files  ⚠️ Completed milestones
*_SUMMARY.md                      ~10 files  ⚠️ Summaries
REPOMAP.md                        1 file     ⚠️ Old repomap
Various other archived docs       ~13 files  ⚠️ Historical
```

---

### BusinessPlanning/ (Confidential) - ✅ KEEP ALL

```
BANK_PARTNERSHIP_PROPOSAL.md      ✅ Partnership proposal
BUSINESS_NAME_IDEAS.md            ✅ Naming brainstorm
BUSINESS_STRATEGY.md              ✅ Go-to-market strategy
CONFIDENTIAL_README.md            ✅ Folder overview
CONSULTING_SERVICES_TEMPLATE.md   ✅ Services template
DEVELOPMENT_TIMELINE_ANALYSIS.md  ✅ Timeline analysis
DUAL_ROLE_STRATEGY.md             ✅ Employee/founder strategy
FINAL_NAME_RECOMMENDATIONS.md     ✅ Name selection
IP_OWNERSHIP_EVIDENCE.md          ✅ Intellectual property
NEXT_STEPS_ACTION_PLAN.md         ✅ Action plan
NORDIQ_BRANDING_ANALYSIS.md       ✅ Brand identity
NORDIQ_LAUNCH_CHECKLIST.md        ✅ 4-week launch plan
NORDIQ_WEBSITE_STRATEGY.md        ✅ Website strategy
README.md                         ✅ Business docs overview
TRADEMARK_ANALYSIS.md             ✅ Trademark research
```

---

### NordIQ-Website/ (Business Website) - ✅ KEEP ALL

```
index.html                        ✅ Landing page
product.html                      ✅ Product showcase
about.html                        ✅ Company info
how-it-works.html                 ✅ Technical overview
pricing.html                      ✅ Pricing tiers
contact.html                      ✅ Contact form
css/main.css                      ✅ Stylesheets
js/main.js                        ✅ JavaScript
images/README.md                  ✅ Image placeholder
DEPLOYMENT_CHECKLIST.md           ✅ Launch checklist
README.md                         ✅ Website overview
about.txt                         ⚠️ Notes/draft?
```

---

### Root-Level Important Files - ✅ KEEP

```
README.md                         22 KB   ✅ Repository overview
LICENSE                           1.8 KB  ✅ Business Source License 1.1
VERSION                           6 B     ✅ Version number (1.1.0)
CHANGELOG.md                      5.5 KB  ✅ Version history
_StartHere.ipynb                  254 KB  ⚠️ UPDATE (references old paths)
environment.yml                   1.2 KB  ✅ Conda environment
.gitignore                        -       ✅ Git configuration
```

---

### Root-Level Documentation - ⚠️ CONSOLIDATE

Many of these should be moved to Docs/ or archived:

```
CLEANUP_COMPLETE.md               7.7 KB  ⚠️ Move to Docs/archive/
CONFIG_GUIDE.md                   16 KB   ⚠️ Move to Docs/
CONFIGURATION_MIGRATION_COMPLETE.md 11 KB ⚠️ Move to Docs/archive/
CORPORATE_BROWSER_FIX.md          5.2 KB  ⚠️ Move to Docs/
CORPORATE_LAUNCHER_COMPLETE.md    11 KB   ⚠️ Move to Docs/archive/
DASHBOARD_SECURITY_AUDIT.md       12 KB   ⚠️ Move to Docs/
GPU_PROFILER_INTEGRATION.md       9.4 KB  ⚠️ Move to Docs/
PARQUET_VS_PICKLE_VS_JSON.md      18 KB   ⚠️ Move to Docs/
PRODUCTION_DEPLOYMENT.md          13 KB   ⚠️ Move to Docs/
REFACTORING_SUMMARY.md            9.7 KB  ⚠️ Move to Docs/archive/
SECURE_DEPLOYMENT_GUIDE.md        14 KB   ⚠️ Move to Docs/
SECURITY_ANALYSIS.md              18 KB   ⚠️ Move to Docs/
SECURITY_IMPROVEMENTS_COMPLETE.md 15 KB   ⚠️ Move to Docs/archive/
SILENT_DAEMON_MODE_COMPLETE.md    16 KB   ⚠️ Move to Docs/archive/
STARTUP_GUIDE_CORPORATE.md        8.7 KB  ⚠️ Move to Docs/
```

---

## 🧹 Cleanup Recommendations

### Priority 1: Critical Space Savings (2+ GB)

**Action: Delete duplicate models directory**
```bash
# BACKUP FIRST!
rm -rf models/  # Saves ~2.1 GB

# Keep only NordIQ/models/
# Optionally delete old model versions:
rm -rf NordIQ/models/tft_model_20251014_131232/  # 500 MB
rm -rf NordIQ/models/tft_model_20251015_080653/  # 600 MB
# Keep only: 20251013_100205 (demo) + 20251017_122454 (latest)
```

**Savings:** 2.1 - 3.2 GB

### Priority 2: Delete Duplicate Python Files

**Action: Delete all root .py files that exist in NordIQ/**
```bash
# VERIFY NORDIQ VERSIONS ARE NEWER FIRST!
# Then delete root versions:
rm -f adaptive_retraining_daemon.py
rm -f constants.py
rm -f data_buffer.py
rm -f data_validator.py
rm -f demo_data_generator.py
rm -f demo_stream_generator.py
rm -f drift_monitor.py
rm -f generate_api_key.py
rm -f gpu_profiles.py
rm -f main.py
rm -f metrics_generator.py
rm -f metrics_generator_daemon.py
rm -f precompile.py
rm -f scenario_demo_generator.py
rm -f server_encoder.py
rm -f server_profiles.py
rm -f tft_dashboard.py
rm -f tft_dashboard_web.py
rm -f tft_inference.py
rm -f tft_inference_daemon.py
rm -f tft_trainer.py
```

**Savings:** ~500 KB, eliminates confusion

### Priority 3: Delete Duplicate Directories

**Action: Remove old directory structures**
```bash
# VERIFY files are in NordIQ/ first!
rm -rf Dashboard/      # Moved to NordIQ/src/dashboard/
rm -rf adapters/       # Moved to NordIQ/src/core/adapters/
rm -rf explainers/     # Moved to NordIQ/src/core/explainers/
rm -rf tabs/           # Moved to NordIQ/src/dashboard/Dashboard/tabs/
rm -rf utils/          # Moved to NordIQ/src/dashboard/Dashboard/utils/

# Archive old config (don't delete, may need for reference)
# config/ and config_archive/ - keep for now
```

**Savings:** ~100 KB, cleaner structure

### Priority 4: Delete Duplicate Scripts

**Action: Keep NordIQ/ versions only**
```bash
# Delete root versions (NordIQ/ has newer versions)
rm -f run_daemon.bat
rm -f setup_api_key.bat
rm -f setup_api_key.sh
rm -f start_all.bat
rm -f start_all.sh
rm -f stop_all.bat
rm -f stop_all.sh

# Evaluate corporate versions (likely obsolete after NordIQ/ reorganization)
# rm -f start_all_corp.bat
# rm -f start_all_corp.sh
# rm -f start_dashboard_corporate.bat
```

**Savings:** ~50 KB, reduces confusion

### Priority 5: Consolidate Root Documentation

**Action: Move scattered docs to Docs/**
```bash
mkdir -p Docs/configuration/
mkdir -p Docs/security/
mkdir -p Docs/deployment/

# Move configuration docs
mv CONFIG_GUIDE.md Docs/configuration/
mv CONFIGURATION_MIGRATION_COMPLETE.md Docs/archive/

# Move security docs
mv DASHBOARD_SECURITY_AUDIT.md Docs/security/
mv SECURITY_ANALYSIS.md Docs/security/
mv SECURE_DEPLOYMENT_GUIDE.md Docs/deployment/

# Move deployment docs
mv PRODUCTION_DEPLOYMENT.md Docs/deployment/
mv STARTUP_GUIDE_CORPORATE.md Docs/deployment/

# Move completed work to archive
mv CLEANUP_COMPLETE.md Docs/archive/
mv CORPORATE_LAUNCHER_COMPLETE.md Docs/archive/
mv REFACTORING_SUMMARY.md Docs/archive/
mv SECURITY_IMPROVEMENTS_COMPLETE.md Docs/archive/
mv SILENT_DAEMON_MODE_COMPLETE.md Docs/archive/

# Move technical analysis
mv PARQUET_VS_PICKLE_VS_JSON.md Docs/
mv GPU_PROFILER_INTEGRATION.md Docs/
mv CORPORATE_BROWSER_FIX.md Docs/
```

**Benefits:** Organized documentation structure

### Priority 6: Clean Up Artifacts

**Action: Delete build artifacts and errors**
```bash
rm -f nul  # Error artifact
rm -f inference_rolling_window.pkl  # Old format (have .parquet version)
rm -f tft_dashboard_web.py.backup  # Backup file
```

**Savings:** ~1.9 MB

### Priority 7: Update References

**Action: Update _StartHere.ipynb to reference NordIQ/**
- Update all paths from root to `NordIQ/src/`
- Update imports to use new structure
- Test all cells

### Priority 8: Create Deprecation Notice

**Action: Add README.DEPRECATED.md to root**
```markdown
# DEPRECATED ROOT FILES

**Status:** This directory structure is deprecated as of Oct 18, 2025.

**Use NordIQ/ instead:** All development should use the `NordIQ/` directory.

Files remaining in root are:
- Development artifacts (logs, checkpoints)
- Git/repo configuration
- Business planning (BusinessPlanning/)
- Website (NordIQ-Website/)
- Documentation (Docs/)

**Do not edit root .py files** - they are duplicates and will be deleted.
```

---

## 📊 Cleanup Impact Summary

| Action | Space Saved | Files Removed | Risk | Priority |
|--------|-------------|---------------|------|----------|
| Delete duplicate models/ | 2.1 GB | 4 dirs | LOW | 1 |
| Delete duplicate .py files | 500 KB | 21 files | LOW | 2 |
| Delete duplicate directories | 100 KB | 5 dirs | LOW | 3 |
| Delete duplicate scripts | 50 KB | 10 files | LOW | 4 |
| Delete old model versions | 1.1 GB | 2 dirs | MEDIUM | 1 |
| Clean up artifacts | 1.9 MB | 3 files | LOW | 6 |
| **TOTAL** | **~3.7 GB** | **~50 items** | **LOW** | - |

---

## 🎯 Post-Cleanup Repository Structure

After cleanup, the repository should look like:

```
MonitoringPrediction/
├── NordIQ/                    # ✅ PRIMARY - All application code here
├── NordIQ-Website/            # ✅ Business website
├── Docs/                      # ✅ All documentation (organized)
├── BusinessPlanning/          # ✅ Business docs (confidential)
├── checkpoints/               # Development artifacts
├── lightning_logs/            # Training logs
├── training/                  # Training data
├── logs/                      # Runtime logs
├── data_buffer/               # Buffer data
├── plots/                     # Analysis plots
├── README.md                  # Main README
├── LICENSE                    # Business Source License
├── VERSION                    # Version number
├── CHANGELOG.md               # Version history
├── _StartHere.ipynb           # Getting started (updated)
├── environment.yml            # Conda environment
├── .gitignore                 # Git config
├── test_env.bat               # Environment test
├── install_security_deps.*    # Setup scripts
└── README.DEPRECATED.md       # Deprecation notice
```

**Result:** Clean, organized, 3.7 GB smaller, no confusion about which files to edit.

---

## 🔍 One-Off Scripts Identified

### Scripts That Can Be Deleted

1. **run_certification.bat** - One-time validation, no longer needed
2. **validate_pipeline.bat** - One-time validation, no longer needed
3. **validate_schema.bat** - One-time validation, no longer needed
4. **run_demo.py** - Replaced by scenario_demo_generator.py

### Scripts That Should Be Kept

1. **test_env.bat** - Useful for environment validation
2. **install_security_deps.bat/sh** - Setup script
3. **kill_daemon.ps1** - Utility for killing stuck daemons

### Scripts That Need Evaluation

1. **start_all_corp.bat/sh** - Corporate-specific launcher (still needed?)
2. **start_dashboard_corporate.bat** - Corporate browser workaround (still needed?)
3. **production_metrics_forwarder_TEMPLATE.py** - Template for production (keep as example)

---

## 🚨 Critical Warnings

### Do NOT Delete (Yet)

1. **config/** - Old config system, may need for reference during migration
2. **config_archive/** - Already archived, low priority
3. **checkpoints/** - Training checkpoints (may be needed)
4. **lightning_logs/** - Training logs (useful for debugging)
5. **training/** - Training data directory (active)
6. **init.d/** - Production deployment scripts
7. **systemd/** - Production deployment scripts

### Version Conflicts

**CRITICAL:** The NordIQ/src/daemons/tft_inference_daemon.py is **newer** (82 KB, Oct 18) than root version (62 KB, Oct 17).

**Action Required:** Ensure all changes from Oct 18 debugging session are in NordIQ/ before deleting root files.

**Verification:**
```bash
# Check modification dates
ls -lh tft_inference_daemon.py
ls -lh NordIQ/src/daemons/tft_inference_daemon.py

# Compare file sizes
du -h tft_inference_daemon.py
du -h NordIQ/src/daemons/tft_inference_daemon.py

# If NordIQ is newer and larger, safe to delete root
```

---

## 📋 Cleanup Checklist

### Pre-Cleanup (MUST DO FIRST)

- [ ] Create full backup of repository
- [ ] Verify NordIQ/ versions are newer than root
- [ ] Test NordIQ/start_all.bat to ensure it works
- [ ] Commit all current work to git
- [ ] Create git tag for pre-cleanup state: `git tag v1.1.0-pre-cleanup`

### Cleanup Execution

- [ ] Delete duplicate models/ directory (2.1 GB)
- [ ] Delete duplicate .py files (21 files)
- [ ] Delete duplicate directories (5 dirs)
- [ ] Delete duplicate scripts (10 files)
- [ ] Clean up artifacts (nul, .pkl, .backup)
- [ ] Move scattered docs to Docs/
- [ ] Create README.DEPRECATED.md
- [ ] Update _StartHere.ipynb paths
- [ ] Update .gitignore if needed

### Post-Cleanup

- [ ] Test NordIQ/start_all.bat
- [ ] Test dashboard loads correctly
- [ ] Test all 10 dashboard tabs
- [ ] Verify no import errors
- [ ] Commit cleanup changes
- [ ] Create git tag: `git tag v1.1.1-post-cleanup`
- [ ] Update Docs/RAG/CURRENT_STATE.md
- [ ] Create session summary

---

## 📊 File Purpose Matrix

### By Function

| Function | Root Location | NordIQ Location | Keep |
|----------|---------------|-----------------|------|
| Inference daemon | tft_inference_daemon.py | src/daemons/ | NordIQ |
| Metrics generation | metrics_generator_daemon.py | src/daemons/ | NordIQ |
| Dashboard | tft_dashboard_web.py | src/dashboard/ | NordIQ |
| Training | tft_trainer.py | src/training/ | NordIQ |
| Config | config/ | src/core/config/ | NordIQ |
| Utilities | *.py scattered | src/core/ | NordIQ |
| Scripts | *.bat/*.sh scattered | NordIQ/ root | NordIQ |

---

## 🔮 Future Maintenance

### Regular Cleanup Tasks

**Monthly:**
- Review and archive old session notes
- Delete old model versions (keep latest 2)
- Clean up log files (keep last 30 days)

**Quarterly:**
- Review deprecated scripts
- Update documentation structure
- Consolidate archived docs

**Before Major Releases:**
- Full repository cleanup
- Update REPOMAP.md
- Verify no duplicate files
- Test all scripts and tools

---

## 📞 Questions for User

Before executing cleanup, please confirm:

1. **Corporate Scripts:** Are `start_all_corp.*` and `start_dashboard_corporate.bat` still needed, or have they been replaced by NordIQ/start_all.*?

2. **Production Forwarder:** Should `production_metrics_forwarder_TEMPLATE.py` be kept as a template, or moved to NordIQ/src/core/?

3. **Old Models:** Can we delete models before Oct 17, or are they needed for comparison?

4. **Config Directory:** Can we delete the old `config/` directory, or should we keep it archived?

5. **_StartHere.ipynb:** Should this be updated to use NordIQ/ paths, or should we create a new NordIQ/_StartHere.ipynb?

---

## 📝 Next Steps

1. **Review this REPOMAP** - Verify accuracy
2. **Answer questions above** - Confirm cleanup scope
3. **Execute Priority 1 cleanup** - Delete duplicate models (2.1 GB)
4. **Execute Priority 2-4** - Delete duplicate code/scripts
5. **Update documentation** - Reflect new structure
6. **Create cleanup session summary** - Document changes

---

**Version:** 1.0.0
**Created:** 2025-10-19
**Author:** Claude (with human oversight)
**Purpose:** Repository organization and cleanup planning
**Status:** DRAFT - Awaiting approval before cleanup execution

---

**SAFETY NOTE:** All cleanup actions are reversible via git. A full backup and git tag should be created before any deletions.
