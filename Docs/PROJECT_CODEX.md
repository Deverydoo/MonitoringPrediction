# PROJECT CODEX - Rules & Conventions

**Version:** 1.0.0
**Status:** ⚠️ AUTHORITATIVE - All development must follow these rules
**Last Updated:** 2025-10-11

---

## 🎯 Purpose

This document defines the immutable rules, conventions, and standards for the TFT Monitoring Prediction System. All contributors and AI assistants must follow these guidelines.

---

## 📜 Core Principles

### 1. Model-First Development
> **"If it doesn't use the model, it is rejected."**

- ✅ All predictions MUST come from the TFT model
- ❌ No heuristics, no fallback logic, no "smart guesses"
- ✅ Heuristics ONLY acceptable for graceful degradation when daemon unavailable
- ✅ Always prefer model predictions over any other approach

### 2. Contract-Driven Development
> **DATA_CONTRACT.md is law.**

- ✅ All schema changes must update DATA_CONTRACT.md FIRST
- ✅ All code must validate against contract before processing
- ✅ Breaking changes require version bump and migration guide
- ❌ Never modify schemas without updating contract

### 3. Profile-Based Architecture
> **Servers are grouped by role, not treated individually.**

- ✅ All training data must include profile column
- ✅ Model must use profile as static_categorical
- ✅ New servers inherit profile patterns automatically
- ❌ Never train without profile feature enabled

### 4. Parquet-First Data
> **JSON is legacy, Parquet is standard.**

- ✅ Always use Parquet for training data (10-100x faster)
- ✅ CSV acceptable for small datasets (<10K rows)
- ❌ JSON only for configuration files, not data
- ✅ All generators must output Parquet by default

### 5. Hash-Based Stability
> **Server IDs must be deterministic and stable.**

- ✅ Use SHA256 hash-based encoding for all server names
- ❌ Never use sequential encoding (0,1,2,3...)
- ✅ Save server_mapping.json with every model
- ✅ Always decode server IDs back to names in output

---

## 🏗️ Architecture Rules

### File Structure Standards

**Core Modules:**
```
├── _StartHere.ipynb          # PRIMARY INTERFACE - Use this first
├── main.py                   # CLI interface
├── config.py                 # SINGLE SOURCE for all config
├── metrics_generator.py      # Data generation
├── tft_trainer.py            # Model training
├── tft_inference.py          # Inference daemon
├── tft_dashboard_web.py      # Web dashboard (ONLY dashboard)
├── server_encoder.py         # Server encoding utility
├── data_validator.py         # Contract validation
└── common_utils.py           # Shared utilities
```

**Data Structure:**
```
├── training/
│   ├── server_metrics.parquet      # Training data
│   ├── metrics_metadata.json       # Metadata
│   └── server_mapping.json         # Name↔ID mapping
├── models/
│   └── tft_model_YYYYMMDD_HHMMSS/
│       ├── model.safetensors       # Weights
│       ├── config.json             # Architecture
│       ├── training_info.json      # Metadata
│       └── server_mapping.json     # CRITICAL for inference
└── Docs/
    ├── ESSENTIAL_RAG.md            # Quick reference
    ├── PROJECT_CODEX.md            # This file
    ├── DATA_CONTRACT.md            # Schema spec
    ├── SERVER_PROFILES.md          # Profile system
    └── INDEX.md                    # Navigation
```

### Naming Conventions

**Server Names:**
- ML Compute: `ppml####` (e.g., ppml0001, ppml0099)
- Database: `ppdb###` (e.g., ppdb001, ppdb150)
- Web/API: `ppweb###` (e.g., ppweb001, ppweb500)
- Conductor: `ppcon##` (e.g., ppcon01, ppcon99)
- Data Ingest: `ppetl###` (e.g., ppetl001, ppetl100)
- Risk Analytics: `pprisk###` (e.g., pprisk001, pprisk050)
- Generic: `ppgen###` (e.g., ppgen001, ppgen100)

**File Naming:**
- Models: `tft_model_YYYYMMDD_HHMMSS/`
- Training data: `server_metrics.parquet` (standardized)
- Metadata: `metrics_metadata.json`, `training_info.json`
- Mappings: `server_mapping.json` (never rename)

**Variable Naming:**
- Column names: snake_case (`cpu_pct`, `mem_pct`, `disk_io_mb_s`)
- Python variables: snake_case (`server_name`, `time_idx`)
- Classes: PascalCase (`ServerEncoder`, `DataValidator`)
- Constants: UPPER_SNAKE (`VALID_STATES`, `CONTRACT_VERSION`)

---

## 🔒 Schema Rules

### Immutable Columns (REQUIRED)
```python
REQUIRED_COLUMNS = [
    'timestamp',      # datetime - ISO8601 format
    'server_name',    # string - Original hostname
    'cpu_pct',        # float - 0.0 to 100.0
    'mem_pct',        # float - 0.0 to 100.0
    'disk_io_mb_s',   # float - 0.0+
    'latency_ms',     # float - 0.0+
    'state',          # string - One of VALID_STATES
    'profile'         # string - One of 7 profiles (NEW)
]
```

### Immutable States (EXACTLY 8)
```python
VALID_STATES = [
    'critical_issue',   # Severe problems
    'healthy',          # Normal operations
    'heavy_load',       # High utilization
    'idle',            # Low activity
    'maintenance',      # Scheduled maintenance
    'morning_spike',    # Peak usage periods
    'offline',         # Server unavailable
    'recovery'          # Post-incident recovery
]
```

### Immutable Profiles (EXACTLY 7)
```python
VALID_PROFILES = [
    'ml_compute',       # ML training nodes
    'database',         # Oracle, Postgres, MongoDB
    'web_api',          # REST endpoints, gateways
    'conductor_mgmt',   # Job scheduling
    'data_ingest',      # Kafka, Spark, ETL
    'risk_analytics',   # VaR, Monte Carlo
    'generic'           # Fallback
]
```

**Rule:** Adding/removing states or profiles requires:
1. Update DATA_CONTRACT.md
2. Version bump (e.g., 1.0.0 → 2.0.0)
3. Regenerate ALL training data
4. Retrain ALL models
5. Update ALL documentation

---

## 🧪 Validation Rules

### Pre-Training Validation
```python
# REQUIRED before training
validator = DataValidator(strict=True)
is_valid, errors, warnings = validator.validate_schema(df)

if not is_valid:
    print(f"[ERROR] Validation failed: {errors}")
    sys.exit(1)

if warnings:
    print(f"[WARNING] {warnings}")
    # Continue but log warnings
```

### Pre-Inference Validation
```python
# REQUIRED before inference
is_compatible, errors = validator.validate_model_compatibility(
    model_dir, input_df
)

if not is_compatible:
    print(f"[ERROR] Model incompatible: {errors}")
    sys.exit(1)
```

### Model Loading Validation
```python
# REQUIRED when loading model
mapping_file = model_dir / "server_mapping.json"
if not mapping_file.exists():
    raise FileNotFoundError(
        "server_mapping.json not found. "
        "Retrain model with updated tft_trainer.py"
    )
```

---

## 📊 Training Rules

### Data Generation Standards
```python
# REQUIRED parameters
--hours >= 24          # Minimum 1 day
--hours = 720          # RECOMMENDED for production (30 days)
--offline_mode dense   # REQUIRED for realistic data
--out_dir ./training/  # Standardized output location
```

### Fleet Composition Standards
**Default (90 servers):**
- WEB_API: 25 (28%)
- ML_COMPUTE: 20 (22%)
- DATABASE: 15 (17%)
- DATA_INGEST: 10 (11%)
- RISK_ANALYTICS: 8 (9%)
- GENERIC: 7 (8%)
- CONDUCTOR_MGMT: 5 (5%)

**Rule:** Production fleets should maintain realistic proportions

### Training Requirements
```python
# MINIMUM for viable model
--epochs >= 5

# RECOMMENDED for production
--epochs >= 20

# REQUIRED features
--static_categoricals = ['profile']  # Transfer learning
--categorical_encoders = {
    'server_id': NaNLabelEncoder(add_nan=True),
    'state': NaNLabelEncoder(add_nan=True),
    'profile': NaNLabelEncoder(add_nan=True)
}
```

### Training Output Requirements
**MUST save:**
1. `model.safetensors` - Model weights
2. `config.json` - Architecture config
3. `training_info.json` - Metadata with contract version
4. `server_mapping.json` - CRITICAL for inference

**Rule:** Training fails if any of these files are not saved

---

## 🔄 Inference Rules

### Daemon Requirements
```python
# REQUIRED for production
--daemon              # Run as background service
--port 8000          # Standard port
--model-dir ./models/latest/  # Must have server_mapping.json
```

### API Endpoint Standards
```
GET  /health          # Must return {"status": "ok", "model_loaded": bool}
GET  /status          # Must return model info, uptime
GET  /predictions/current  # Must decode server names
GET  /alerts/active   # Must include server names (not IDs)
WS   /ws/predictions  # Future: streaming predictions
```

### Prediction Output Format
```python
{
    'server_name': 'ppml0015',  # DECODED name, not ID
    'profile': 'ml_compute',     # Profile for context
    'prediction_time': 'ISO8601',
    'predictions': {
        '30min': {'cpu_pct': 65.2, 'mem_pct': 72.1, ...},
        '1hr': {...},
        '8hr': {...}
    },
    'quantiles': {
        'p10': {...},  # Lower bound
        'p50': {...},  # Median
        'p90': {...}   # Upper bound
    }
}
```

**Rule:** Never return server IDs in final output, always decode

---

## 🎨 Dashboard Rules

### Web Dashboard Standards
- ✅ Use `tft_dashboard_web.py` (Streamlit)
- ❌ Terminal dashboards deprecated
- ✅ Connect to daemon via REST API
- ✅ Include demo modes (Healthy/Degrading/Critical)
- ✅ Display server names (not IDs)
- ✅ Show profile information
- ✅ Real-time updates (configurable refresh rate)

### Visualization Standards
**Required Panels:**
1. Overview - Fleet health summary
2. Heatmap - Visual risk grid
3. Top Servers - Problem servers with predictions
4. Historical - Trend analysis
5. Advanced - Settings, debug, model info

**Color Coding:**
- 🟢 Green: Healthy (risk < 0.3)
- 🟡 Yellow: Warning (risk 0.3-0.7)
- 🔴 Red: Critical (risk > 0.7)

---

## 🧩 Integration Rules

### Python Environment
```bash
# REQUIRED
conda activate py310

# REQUIRED packages
torch>=2.0
lightning>=2.0
pytorch-forecasting
safetensors
pandas
pyarrow          # For Parquet
streamlit        # For dashboard
fastapi          # For daemon
uvicorn          # For daemon
```

### Import Standards
```python
# Standard library first
import json
import sys
from pathlib import Path

# Third-party second
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

# Local modules third
from config import *
from server_encoder import ServerEncoder
from data_validator import DataValidator
```

---

## 📝 Documentation Rules

### Required Documentation
**Every new feature requires:**
1. Update ESSENTIAL_RAG.md
2. Update DATA_CONTRACT.md (if schema changes)
3. Update PROJECT_CODEX.md (if new rules)
4. Create/update specific guide (e.g., SERVER_PROFILES.md)
5. Update INDEX.md navigation

### Session Summary Requirements
**At end of every session, create:**
- `SESSION_YYYY-MM-DD_SUMMARY.md`
- Include: start/end time, duration, accomplishments
- Update: PROJECT_SUMMARY.md change notes
- List: modified files, next steps

### Documentation Style
- Use emoji sparingly (✅ ❌ ⚠️ 🔄 only)
- Include code examples for complex concepts
- Provide "Before/After" comparisons for changes
- Include quick reference sections
- Mark deprecated features clearly

---

## 🚨 Error Handling Rules

### Validation Errors
```python
# ALWAYS provide context
raise ValueError(
    f"Invalid state '{state}'. "
    f"Must be one of: {VALID_STATES}. "
    f"See DATA_CONTRACT.md for details."
)
```

### File Not Found Errors
```python
# ALWAYS provide fix instructions
raise FileNotFoundError(
    f"server_mapping.json not found in {model_dir}. "
    f"This file is required for decoding predictions. "
    f"Fix: Retrain model with updated tft_trainer.py"
)
```

### Dimension Mismatch Errors
```python
# ALWAYS include contract info
raise RuntimeError(
    f"State count mismatch: model has {model_states}, "
    f"contract requires {len(VALID_STATES)}. "
    f"Contract version: {CONTRACT_VERSION}. "
    f"Fix: Retrain model with current contract."
)
```

**Rule:** All errors must include:
1. What went wrong
2. Why it matters
3. How to fix it
4. Relevant documentation reference

---

## 🔐 Security Rules

### Data Protection
- ❌ Never commit training data to git
- ❌ Never commit model weights to git (use Git LFS if needed)
- ✅ Use `.gitignore` for `training/`, `models/`, `demo_data/`
- ✅ Sanitize server names in public documentation

### API Security
- ✅ Validate all inputs before processing
- ✅ Rate limit API endpoints
- ✅ Log all prediction requests
- ❌ Never expose raw model tensors via API

---

## 🧪 Testing Rules

### Unit Testing Requirements
**Every utility module must have:**
```python
if __name__ == '__main__':
    # Test basic functionality
    # Test edge cases
    # Test error conditions
    print("✅ All tests passed")
```

### Integration Testing Requirements
**Before every release:**
1. Generate fresh training data
2. Train model for 1 epoch (smoke test)
3. Start daemon, verify /health endpoint
4. Launch dashboard, verify connection
5. Test unknown server handling
6. Verify server name decoding

### Manual Testing Checklist
- [ ] Data generation produces correct schema
- [ ] Training saves all required files
- [ ] Inference daemon starts without errors
- [ ] Dashboard connects to daemon
- [ ] Predictions decode server names correctly
- [ ] Unknown servers handled gracefully
- [ ] Demo modes work correctly

---

## 🎯 Performance Rules

### Data Loading
- ✅ Always use Parquet for datasets >10K rows
- ✅ Use chunking for datasets >1M rows
- ❌ Never load entire dataset into memory if >10GB
- ✅ Profile memory usage for large datasets

### Model Training
- ✅ Use GPU if available (50-100x faster)
- ✅ Use mixed precision (FP16/BF16) for large models
- ✅ Save checkpoints every 5 epochs
- ✅ Implement early stopping (patience=3)

### Inference
- ✅ Batch predictions when possible
- ✅ Cache common predictions (optional)
- ✅ Target <100ms latency per server
- ❌ Never block on I/O in prediction loop

---

## 🔄 Version Control Rules

### Commit Messages
```
Format: <type>: <short description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- refactor: Code restructuring
- perf: Performance improvement
- test: Adding tests
- chore: Maintenance

Example:
feat: Add profile-based transfer learning to TFT trainer
```

### Branch Strategy
- `main` - Production ready code
- `develop` - Integration branch
- `feature/profile-system` - Feature branches
- `hotfix/dimension-mismatch` - Urgent fixes

### What NOT to Commit
- `training/` - Training data
- `models/` - Model weights
- `demo_data/` - Demo datasets
- `.ipynb_checkpoints/` - Notebook checkpoints
- `__pycache__/` - Python cache

---

## 🎓 Development Workflow

### Standard Workflow
```bash
# 1. Activate environment
conda activate py310

# 2. Generate data
jupyter notebook _StartHere.ipynb  # Cell 4

# 3. Train model
# In notebook, Cell 6 (10 epochs minimum)

# 4. Validate
python data_validator.py training/server_metrics.parquet

# 5. Test inference
python tft_inference.py --daemon --port 8000

# 6. Launch dashboard
streamlit run tft_dashboard_web.py

# 7. Test end-to-end
# Use dashboard demo modes
```

### Making Schema Changes
```bash
# 1. Update contract FIRST
vim Docs/DATA_CONTRACT.md
# Bump version, document changes

# 2. Update code
# - metrics_generator.py
# - tft_trainer.py
# - tft_inference.py

# 3. Regenerate everything
rm -rf training/* models/*
jupyter notebook _StartHere.ipynb

# 4. Test thoroughly
python data_validator.py training/server_metrics.parquet
# Train, test inference, test dashboard

# 5. Document
# Update ESSENTIAL_RAG.md, create session summary
```

---

## ⚠️ Breaking Change Protocol

### When Breaking Changes Are Needed

**Step 1: Documentation**
1. Update DATA_CONTRACT.md
2. Bump version (1.0.0 → 2.0.0)
3. Create MIGRATIONS.md with upgrade guide

**Step 2: Code Updates**
1. Update metrics_generator.py
2. Update tft_trainer.py
3. Update tft_inference.py
4. Update data_validator.py

**Step 3: Data Regeneration**
1. Delete old training data
2. Delete old models
3. Generate fresh data
4. Retrain models

**Step 4: Validation**
1. Run full test suite
2. Test end-to-end workflow
3. Verify dashboard integration
4. Test unknown server handling

**Step 5: Communication**
1. Update all documentation
2. Create session summary
3. Notify team of changes
4. Update README with migration guide

---

## 📊 Metrics & Monitoring

### Required Metrics
**During Training:**
- Training loss per epoch
- Validation loss per epoch
- Learning rate progression
- Estimated time remaining

**During Inference:**
- Predictions per second
- Average latency
- Error rate
- Unknown server count

**Dashboard:**
- Refresh rate
- Connection status
- Model version
- Last prediction time

---

## 🎯 Quality Standards

### Code Quality
- ✅ Type hints for function signatures
- ✅ Docstrings for all classes/functions
- ✅ Error handling with context
- ✅ Logging for important events
- ✅ Configuration via config.py, not hardcoded

### Documentation Quality
- ✅ Clear purpose statement
- ✅ Prerequisites listed
- ✅ Code examples included
- ✅ Troubleshooting section
- ✅ Last updated timestamp

### Model Quality
- ✅ Training loss converges
- ✅ Validation loss < 1.5x training loss
- ✅ Unknown server handling tested
- ✅ Profile feature enabled
- ✅ All required files saved

---

## 🚀 Production Readiness Checklist

**Before Deployment:**
- [ ] Data contract validated
- [ ] Model trained with profiles
- [ ] Inference daemon tested
- [ ] Dashboard tested
- [ ] Unknown servers tested
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Documentation complete
- [ ] Session summary created
- [ ] Backup strategy defined

---

## 📞 Quick Reference

### Contract Version
**Current:** 1.0.0
**Location:** `Docs/DATA_CONTRACT.md`

### Model Version
**Current:** 3.0.0 (Profile-Based Transfer Learning)
**Location:** `models/tft_model_*/training_info.json`

### Python Environment
**Required:** py310 (Python 3.10)
**Activation:** `conda activate py310`

### Critical Files
- `DATA_CONTRACT.md` - Schema spec
- `SERVER_PROFILES.md` - Profile definitions
- `ESSENTIAL_RAG.md` - Quick reference
- `PROJECT_CODEX.md` - This file

---

## 🎉 Success Criteria

**A feature is complete when:**
- ✅ Code follows all codex rules
- ✅ Data contract validated
- ✅ Tests pass
- ✅ Documentation updated
- ✅ End-to-end workflow tested
- ✅ Session summary created
- ✅ No regressions introduced

---

**Version:** 1.0.0
**Status:** AUTHORITATIVE
**Last Updated:** 2025-10-11
**Maintained By:** Project Team
**Review Frequency:** Before any major changes

⚠️ **THIS CODEX IS LAW - ALL DEVELOPMENT MUST FOLLOW THESE RULES**
