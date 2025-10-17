# PROJECT CODEX - Rules & Conventions

**Version:** 2.1.0 (Post-Presentation Development)
**Status:** ✅ ACTIVE DEVELOPMENT - Balanced rules for quality and progress
**Last Updated:** 2025-10-17

---

## 🎯 Purpose

This document defines the development rules, conventions, and standards for the TFT Monitoring Prediction System. These guidelines ensure quality while allowing iterative improvement.

**Development Phase**: Post-demo active development
**Philosophy**: Quality over speed, but allow for experimentation and enhancement

---

## 🚦 Development Approach (Post-Presentation)

### Feature Development Policy

**✅ ALLOWED - Incremental Enhancements:**
- New dashboard tabs or visualizations
- Performance optimizations
- Additional metrics or data sources
- UI/UX improvements
- Bug fixes and refinements
- Security enhancements
- Documentation improvements

**⚠️ CAUTION - Requires Planning:**
- Schema changes (update DATA_CONTRACT.md first)
- Breaking API changes (version bump required)
- Major architectural changes (document trade-offs)
- New external dependencies (justify necessity)

**❌ AVOID - High Risk:**
- Rushing features without testing
- Breaking changes without migration path
- Removing existing functionality
- Undocumented "clever" solutions
- Feature creep (discuss necessity first)

### Development Pace

**Recommended Approach:**
1. **Propose** - Describe what you want to add and why
2. **Plan** - Break into small, testable increments
3. **Implement** - One feature at a time, test as you go
4. **Validate** - Ensure no regressions, document changes
5. **Review** - Check that it aligns with system goals

**Key Principle**: "Make it work, make it right, make it fast" - in that order.

### Quality Gates

**Before committing new features:**
- ✅ Feature works as intended
- ✅ No breaking changes to existing functionality
- ✅ Documentation updated
- ✅ Tested with realistic scenarios
- ✅ Code follows existing patterns

**Not required (but encouraged):**
- Full test suite (manual testing is acceptable for now)
- Perfect optimization (working > perfect)
- Complete feature set (incremental is fine)

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

### Immutable Columns (REQUIRED) - LINBORG Metrics v2.0

```python
REQUIRED_COLUMNS = [
    # Core identification
    'timestamp',         # datetime - ISO8601 format
    'server_name',       # string - Original hostname
    'profile',           # string - One of 7 profiles
    'state',             # string - One of VALID_STATES

    # LINBORG CPU Metrics (5 metrics)
    'cpu_user_pct',      # float - 0.0 to 100.0 - User space CPU
    'cpu_sys_pct',       # float - 0.0 to 100.0 - System/kernel CPU
    'cpu_iowait_pct',    # float - 0.0 to 100.0 - I/O wait (CRITICAL)
    'cpu_idle_pct',      # float - 0.0 to 100.0 - Idle CPU
    'java_cpu_pct',      # float - 0.0 to 100.0 - Java/Spark CPU

    # LINBORG Memory Metrics (2 metrics)
    'mem_used_pct',      # float - 0.0 to 100.0 - Memory utilization
    'swap_used_pct',     # float - 0.0 to 100.0 - Swap usage

    # LINBORG Disk Metrics (1 metric)
    'disk_usage_pct',    # float - 0.0 to 100.0 - Disk space usage

    # LINBORG Network Metrics (2 metrics)
    'net_in_mb_s',       # float - 0.0+ - Network ingress MB/s
    'net_out_mb_s',      # float - 0.0+ - Network egress MB/s

    # LINBORG Connection Metrics (2 metrics)
    'back_close_wait',   # int - TCP backend connections
    'front_close_wait',  # int - TCP frontend connections

    # LINBORG System Metrics (2 metrics)
    'load_average',      # float - 0.0+ - System load average
    'uptime_days'        # int - 0-30 - Days since reboot
]

# Total: 4 core + 14 LINBORG metrics = 18 required columns
```

### DEPRECATED Columns (DO NOT USE):
```python
# OLD SYSTEM - WILL CAUSE ERRORS
DEPRECATED_COLUMNS = ['cpu_pct', 'mem_pct', 'disk_io_mb_s', 'latency_ms']  # ❌
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

## 🚨 LINBORG Metrics Rules

### 1. LINBORG Metric Naming (Immutable)
> **All 14 LINBORG metrics must use exact names. No variations allowed.**

```python
# ✅ CORRECT
time_varying_unknown_reals = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]

# ❌ WRONG - Will cause errors
time_varying_unknown_reals = ['cpu_pct', 'mem_pct', 'disk_io_mb_s', 'latency_ms']
time_varying_unknown_reals = ['cpu_percent', 'memory_percent', ...]  # Wrong names
```

### 2. CPU Display Rule
> **Always display "% CPU Used = 100 - cpu_idle_pct", never show raw idle**

```python
# ✅ CORRECT - User-facing display
cpu_used = 100 - cpu_idle_pct
print(f"CPU: {cpu_used:.1f}%")

# ❌ WRONG - Backwards for humans
print(f"CPU Idle: {cpu_idle_pct:.1f}%")  # Confusing
```

### 3. I/O Wait Priority Rule
> **I/O Wait is CRITICAL metric. Always include in risk scoring and alerts.**

```python
# ✅ REQUIRED - I/O wait must have high weight in risk calculation
if cpu_iowait >= 30:
    current_risk += 50  # CRITICAL - severe I/O bottleneck
elif cpu_iowait >= 20:
    current_risk += 30  # High I/O contention
elif cpu_iowait >= 10:
    current_risk += 15  # Elevated I/O wait

# ❌ WRONG - Omitting I/O wait from risk calculation
# (This misses critical bottleneck indicator)
```

### 4. Profile-Specific I/O Wait Thresholds
> **Database servers have different I/O wait "normal" than ML servers**

```python
# ✅ CORRECT - Profile-aware thresholds
if profile == 'database':
    iowait_threshold = 20  # Databases are I/O intensive
elif profile == 'ml_compute':
    iowait_threshold = 5   # ML should be compute-bound

# ❌ WRONG - One-size-fits-all threshold
iowait_threshold = 10  # Misses DB normal behavior
```

### 5. Integer vs Float Types
> **Connection counts are integers, percentages are floats**

```python
# ✅ CORRECT - Proper types
back_close_wait = int(value)      # TCP connections
cpu_iowait_pct = float(value)     # Percentage

# ❌ WRONG - Type mismatch
back_close_wait = 5.7  # Connections must be whole numbers
```

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
1. Update CURRENT_STATE.md (in Docs/RAG/)
2. Update DATA_CONTRACT.md (if schema changes)
3. Update PROJECT_CODEX.md (if new rules)
4. Create/update specific guide (e.g., SERVER_PROFILES.md)
5. Update INDEX.md navigation (in Docs/)

### Session Summary (Recommended)
**For significant work sessions, consider creating:**
- Session summary with key accomplishments
- Update RAG/CURRENT_STATE.md if major changes
- List modified files and next steps

**Not required for minor changes or bug fixes.**

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

## 🧪 Testing Guidelines

### Testing Philosophy (Post-Presentation)

**Pragmatic approach**: Test what matters, when it matters.

**Required Testing:**
- ✅ Manual testing for new features (does it work?)
- ✅ Regression testing (did we break anything?)
- ✅ End-to-end smoke test before commits

**Encouraged (but not blocking):**
- Unit tests for complex logic
- Integration tests for critical paths
- Performance benchmarks for optimizations

### Unit Testing (Encouraged)
**For complex utility modules, consider adding:**
```python
if __name__ == '__main__':
    # Test basic functionality
    # Test edge cases
    # Test error conditions
    print("✅ All tests passed")
```

### Manual Testing Checklist (New Features)
**Before committing significant changes:**
- [ ] Feature works as intended
- [ ] No obvious bugs or errors
- [ ] Dashboard still loads and functions
- [ ] Daemon still connects properly
- [ ] No regressions in existing features

### Integration Testing (Major Changes)
**Before releases or major updates:**
1. Start all three services (generator, daemon, dashboard)
2. Verify healthy scenario works
3. Test scenario switching
4. Check error handling
5. Verify documentation is updated

**Note**: Perfect test coverage is a goal, not a requirement. Focus on functional, working code first.

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

## 🔢 Versioning

### Semantic Versioning

This project uses [Semantic Versioning 2.0.0](https://semver.org/):

**Format**: `MAJOR.MINOR.PATCH`

**Current Version**: See [VERSION](../../../VERSION) file

### Version Increments

**MAJOR version** (e.g., 1.0.0 → 2.0.0) - Breaking changes:
- Schema changes requiring data regeneration
- API changes that break existing clients
- LINBORG metrics structure changes
- Profile system changes requiring retraining
- Incompatible configuration changes

**MINOR version** (e.g., 1.0.0 → 1.1.0) - New features:
- New dashboard tabs or visualizations
- New metrics (without breaking existing)
- New server profiles
- Performance improvements
- New API endpoints (backward compatible)
- Security enhancements

**PATCH version** (e.g., 1.0.0 → 1.0.1) - Bug fixes and docs:
- Bug fixes
- Documentation updates
- UI improvements (no functionality change)
- Code refactoring (no behavior change)
- Dependency updates (compatible)

### Version Update Process

**When making changes:**

1. **Determine version bump**:
   - Breaking change? → MAJOR
   - New feature? → MINOR
   - Bug fix/docs? → PATCH

2. **Update VERSION file**:
   ```bash
   echo "1.1.0" > VERSION
   ```

3. **Update CHANGELOG.md**:
   - Add new version section
   - List all changes under Added/Changed/Fixed
   - Include breaking changes at top if MAJOR

4. **Update key docs**:
   - Docs/RAG/CURRENT_STATE.md (version at top)
   - README.md badge (if automated, otherwise manual)

5. **Commit with version tag**:
   ```bash
   git add VERSION CHANGELOG.md
   git commit -m "chore: bump version to 1.1.0"
   git tag v1.1.0
   git push origin main --tags
   ```

### Examples

**Bug fix (1.0.0 → 1.0.1)**:
```bash
# Fixed dashboard refresh bug
echo "1.0.1" > VERSION
# Update CHANGELOG.md under [1.0.1] - Fixed
git commit -m "fix: dashboard refresh bug"
git tag v1.0.1
```

**New feature (1.0.1 → 1.1.0)**:
```bash
# Added new anomaly detection tab
echo "1.1.0" > VERSION
# Update CHANGELOG.md under [1.1.0] - Added
git commit -m "feat: add anomaly detection tab"
git tag v1.1.0
```

**Breaking change (1.1.0 → 2.0.0)**:
```bash
# Changed metric schema (breaks existing models)
echo "2.0.0" > VERSION
# Update CHANGELOG.md with migration guide
git commit -m "feat!: migrate to LINBORG v2 schema

BREAKING CHANGE: All models must be retrained"
git tag v2.0.0
```

### Version Checking

**In code, always check version compatibility**:

```python
# Load version
with open('VERSION', 'r') as f:
    SYSTEM_VERSION = f.read().strip()

# Check compatibility
def check_version_compatibility(required_version: str):
    """Check if system version meets requirements."""
    current = tuple(map(int, SYSTEM_VERSION.split('.')))
    required = tuple(map(int, required_version.split('.')))

    # MAJOR must match for compatibility
    if current[0] != required[0]:
        raise ValueError(
            f"Version mismatch: system {SYSTEM_VERSION}, "
            f"required {required_version}"
        )
```

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
- `Docs/RAG/CURRENT_STATE.md` - Single source of truth
- `Docs/RAG/PROJECT_CODEX.md` - This file (development rules)
- `Docs/DATA_CONTRACT.md` - Schema spec
- `Docs/SERVER_PROFILES.md` - Profile definitions

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

**Version:** 2.1.0 (Post-Presentation Development)
**Status:** ACTIVE - Balanced development guidelines
**Last Updated:** 2025-10-17
**Maintained By:** Project Team
**Review Frequency:** Quarterly or before major changes

💡 **THESE GUIDELINES ENSURE QUALITY WHILE ALLOWING PROGRESS**

**Remember**: The goal is sustainable, quality development - not perfection paralysis. When in doubt, discuss trade-offs and make informed decisions.
