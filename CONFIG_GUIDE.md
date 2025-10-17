# Configuration Guide - TFT Monitoring System

**SINGLE SOURCE OF TRUTH** for all configuration values.

Last Updated: October 16, 2025
Config Version: 2.0.0

---

## 🎯 Philosophy

**Problem**: Configuration sprawl - settings scattered across multiple files, JSON configs, hardcoded values
**Solution**: Centralized `config/` package - ONE place to change ANY setting

**Rule**: If you want to change **batch size**, **learning rate**, **CPU thresholds**, **API URLs**, or **ANYTHING** else, you change it in `config/` and **ONLY** in `config/`.

---

## 📁 Configuration Structure

```
config/
├── __init__.py              # Main entry point - imports all configs
├── model_config.py          # TFT model hyperparameters & training
├── metrics_config.py        # LINBORG baselines, state multipliers
└── api_config.py            # API URLs, ports, timeouts

Dashboard/config/            # Dashboard-specific (kept modular)
└── dashboard_config.py      # UI settings, risk thresholds
```

---

## 🔧 How to Use

### Import Everything

```python
# Import all configs at once
from config import MODEL_CONFIG, METRICS_CONFIG, API_CONFIG

# Access values
batch_size = MODEL_CONFIG['batch_size']
daemon_url = API_CONFIG['daemon_url']
profile_baselines = METRICS_CONFIG['profile_baselines']
```

### Import Specific Config

```python
# Import just what you need
from config.model_config import MODEL_CONFIG
from config.api_config import API_CONFIG

# Or import helper functions
from config.api_config import get_full_url, get_timeout
```

### Dashboard Config

```python
# Dashboard config stays in Dashboard/ for modularity
from Dashboard.config.dashboard_config import RISK_THRESHOLDS, CPU_THRESHOLDS
```

---

## 📋 Configuration Files Reference

### 1. `config/model_config.py` - Model & Training

**What's in here**: Everything related to TFT model architecture, training hyperparameters, data processing

**Key Settings**:

| Setting | Default | Description |
|---------|---------|-------------|
| `batch_size` | 32 | **Baseline batch size** (GPU profiler may override with optimal value) |
| `learning_rate` | 0.01 | Initial learning rate |
| `epochs` | 20 | Training epochs (3 for fast testing, 20 for production) |
| `hidden_size` | 32 | Model architecture size (32=production, 16=faster) |
| `attention_heads` | 8 | Attention mechanism heads |
| `context_length` | 288 | Lookback window (288 × 5min = 24 hours) |
| `prediction_horizon` | 96 | Forecast horizon (96 × 5min = 8 hours) |
| `num_workers` | 4 | Data loader workers (GPU profiler overrides with optimal value) |
| `validation_split` | 0.2 | 20% of data for validation |

**GPU Auto-Optimization**: The training system automatically detects your GPU and applies optimal settings:
- **RTX 4090**: batch_size=32, num_workers=8
- **Tesla A100**: batch_size=128, num_workers=32
- **H100**: batch_size=256, num_workers=32
- **CPU**: Uses config defaults

The GPU profiler (`gpu_profiles.py`) intelligently overrides config values when better hardware-specific settings are available.

**Example Changes**:

```python
# Want faster training? Change these:
MODEL_CONFIG['epochs'] = 3              # Fast test
MODEL_CONFIG['batch_size'] = 16         # Lower memory
MODEL_CONFIG['num_workers'] = 0         # Single-threaded

# Want better accuracy? Change these:
MODEL_CONFIG['epochs'] = 20             # Full training
MODEL_CONFIG['hidden_size'] = 32        # Larger model
MODEL_CONFIG['context_length'] = 576    # 48 hours lookback
```

**100% Confidence**: If you change `MODEL_CONFIG['batch_size']` to 64, ALL training scripts will use batch size 64 (unless GPU profiler detects better hardware-specific value).

---

### 1.5 GPU Profiler Integration (Smart Override System)

**How It Works**: The system uses a **two-tier approach**:
1. **Centralized config** provides baseline/fallback values
2. **GPU profiler** intelligently overrides when better hardware-specific settings are available

**When GPU Profiler Overrides**:
```python
# tft_trainer.py automatically detects GPU
if torch.cuda.is_available():
    gpu = setup_gpu()  # Detects RTX 4090, A100, H100, etc.

    # Smart override: Only if batch_size is at default (32)
    if config['batch_size'] == 32:
        config['batch_size'] = gpu.get_batch_size('train')
        # RTX 4090: 32 (keeps default)
        # A100: 128 (GPU-optimized!)
        # H100: 256 (beast mode!)
```

**GPU-Specific Profiles** (`gpu_profiles.py`):
| GPU | Batch Size (Train) | Batch Size (Inference) | Workers | Memory Reserved |
|-----|-------------------|----------------------|---------|-----------------|
| RTX 4090 | 32 | 128 | 8 | 85% (24GB) |
| RTX 3090 | 32 | 128 | 8 | 85% (24GB) |
| Tesla V100 | 64 | 256 | 16 | 90% (16-32GB) |
| Tesla A100 | 128 | 512 | 32 | 90% (40-80GB) |
| H100 | 256 | 1024 | 32 | 90% (80GB) |
| H200 | 512 | 2048 | 32 | 90% (141GB) |
| CPU/Generic | 16 | 64 | 4 | N/A |

**Manual Override**: If you want to force a specific batch size regardless of GPU:
```python
# In config/model_config.py
MODEL_CONFIG['batch_size'] = 64  # Will be used if not 32

# Or pass to trainer
trainer = TFTTrainer(config={'batch_size': 64})  # GPU profiler won't override
```

**Benefits**:
- ✅ Automatic optimization for different hardware
- ✅ No manual tuning needed
- ✅ Prevents OOM errors on smaller GPUs
- ✅ Maximizes throughput on high-end GPUs
- ✅ Fallback to safe defaults on CPU

---

### 2. `config/metrics_config.py` - LINBORG Metrics

**What's in here**: All the hardcoded baselines that were previously buried in `metrics_generator.py` (2000+ lines extracted!)

**Key Components**:

#### Profile Baselines (7 profiles × 14 metrics)

```python
from config.metrics_config import PROFILE_BASELINES, ServerProfile

# Get ML Compute baselines
ml_baselines = PROFILE_BASELINES[ServerProfile.ML_COMPUTE]
cpu_user_mean = ml_baselines['cpu_user'][0]  # 0.45 (45%)
cpu_user_std = ml_baselines['cpu_user'][1]   # 0.12

# Modify a baseline
PROFILE_BASELINES[ServerProfile.DATABASE]['cpu_iowait'] = (0.20, 0.06)  # Higher I/O wait
```

#### State Multipliers (8 states × 14 metrics)

```python
from config.metrics_config import STATE_MULTIPLIERS, ServerState

# Get CRITICAL_ISSUE multipliers
critical_mults = STATE_MULTIPLIERS[ServerState.CRITICAL_ISSUE]
cpu_mult = critical_mults['cpu_user']  # 2.5x normal CPU

# Make critical issues more severe
STATE_MULTIPLIERS[ServerState.CRITICAL_ISSUE]['cpu_user'] = 3.0  # Now 3x!
```

#### Fleet Distribution

```python
from config.metrics_config import METRICS_CONFIG

# Change fleet composition (percentages must sum to 1.0)
METRICS_CONFIG['fleet_distribution']['ml_compute'] = 0.30  # 30% ML servers
METRICS_CONFIG['fleet_distribution']['database'] = 0.20    # 20% databases
```

**100% Confidence**: Change `PROFILE_BASELINES` and `metrics_generator.py` will use the new values. No hardcoded overrides.

---

### 3. `config/api_config.py` - API Endpoints

**What's in here**: All URLs, ports, endpoints, timeouts

**Key Settings**:

| Setting | Default | Description |
|---------|---------|-------------|
| `daemon_url` | http://localhost:8000 | Inference daemon URL |
| `daemon_port` | 8000 | Inference daemon port |
| `metrics_generator_url` | http://localhost:8001 | Metrics generator URL |
| `dashboard_port` | 8501 | Streamlit dashboard port |

**Timeouts**:
- `health_check`: 2 seconds
- `prediction`: 5 seconds
- `feed_data`: 3 seconds

**Helper Functions**:

```python
from config.api_config import get_full_url, get_timeout, get_websocket_url

# Get full URL for an endpoint
url = get_full_url('daemon', 'predictions')  # http://localhost:8000/predictions
url = get_full_url('generator', 'scenario_status')  # http://localhost:8001/scenario/status

# Get timeout for operation
timeout = get_timeout('prediction')  # 5 seconds

# Get WebSocket URL
ws_url = get_websocket_url()  # ws://localhost:8000/ws
```

**100% Confidence**: Change `API_CONFIG['daemon_port']` to 9000, and ALL scripts will connect to port 9000.

---

### 4. `Dashboard/config/dashboard_config.py` - Dashboard UI

**What's in here**: Dashboard-specific settings, risk thresholds, display configuration

**Key Settings**:

| Setting | Value | Description |
|---------|-------|-------------|
| `DAEMON_URL` | http://localhost:8000 | Daemon URL (imports from api_config) |
| `REFRESH_INTERVAL` | 5 | Dashboard refresh seconds |

**Risk Thresholds**:
```python
RISK_THRESHOLDS = {
    'imminent_failure': 90,
    'critical': 80,
    'danger': 70,
    'warning': 60,
    'degrading': 50,
    'healthy': 0
}
```

**CPU Thresholds**:
```python
CPU_THRESHOLDS = {
    'critical': 98,   # 🔴
    'danger': 95,     # 🟠
    'warning': 90,    # 🟡
    'healthy': 0
}
```

---

## 🎯 Common Configuration Tasks

### Change Batch Size

```python
# In config/model_config.py (line 39)
MODEL_CONFIG['batch_size'] = 64  # Changed from 32 to 64
```

**Result**: ALL training scripts will now use batch size 64.

### Change Learning Rate

```python
# In config/model_config.py (line 40)
MODEL_CONFIG['learning_rate'] = 0.001  # Changed from 0.01 to 0.001
```

### Change Number of Epochs

```python
# In config/model_config.py (line 38)
MODEL_CONFIG['epochs'] = 10  # Changed from 20 to 10 for faster training
```

### Change Daemon Port

```python
# In config/api_config.py (line 12)
API_CONFIG['daemon_port'] = 9000  # Changed from 8000 to 9000
```

**Result**: Dashboard, metrics generator, ALL clients will connect to port 9000.

### Change ML Compute CPU Baseline

```python
# In config/metrics_config.py (line 77)
PROFILE_BASELINES[ServerProfile.ML_COMPUTE]['cpu_user'] = (0.55, 0.15)
# Changed from (0.45, 0.12) - now ML servers run hotter by default
```

### Change Fleet Size Distribution

```python
# In config/metrics_config.py (line 460)
METRICS_CONFIG['fleet_distribution']['ml_compute'] = 0.40  # 40% ML servers
METRICS_CONFIG['fleet_distribution']['database'] = 0.10    # 10% databases
```

---

## ✅ Configuration Migration Guide

### Old Way (DEPRECATED)

```python
# ❌ OLD: Hardcoded in metrics_generator.py
PROFILE_BASELINES = {
    ServerProfile.ML_COMPUTE: {
        "cpu_user": (0.45, 0.12),
        # ... 2000 more lines ...
    }
}

# ❌ OLD: JSON config file
with open('tft_config_adjusted.json') as f:
    config = json.load(f)
    batch_size = config['batch_size']  # Which JSON file? Who knows!
```

### New Way (CORRECT)

```python
# ✅ NEW: Import from config
from config import MODEL_CONFIG, METRICS_CONFIG

batch_size = MODEL_CONFIG['batch_size']  # ONE source
baselines = METRICS_CONFIG['profile_baselines']  # ONE source
```

---

## 🔍 Finding Configuration Values

**Question**: "Where do I change batch size?"
**Answer**: `config/model_config.py` line 39

**Question**: "Where do I change the daemon port?"
**Answer**: `config/api_config.py` line 12

**Question**: "Where are LINBORG metric baselines?"
**Answer**: `config/metrics_config.py` lines 77-230

**Question**: "Where are risk thresholds?"
**Answer**: `Dashboard/config/dashboard_config.py` lines 19-26

**Search Method**:
```bash
# Search all config files for a setting
grep -r "batch_size" config/
# Result: config/model_config.py:39:    'batch_size': 32,
```

---

## 🚫 What NOT to Do

### ❌ Don't Hardcode Values

```python
# ❌ WRONG
batch_size = 32  # Hardcoded!

# ✅ CORRECT
from config import MODEL_CONFIG
batch_size = MODEL_CONFIG['batch_size']
```

### ❌ Don't Create New Config Files

```python
# ❌ WRONG - Don't do this!
with open('my_config.json', 'w') as f:
    json.dump({'batch_size': 64}, f)

# ✅ CORRECT - Edit config/model_config.py
MODEL_CONFIG['batch_size'] = 64
```

### ❌ Don't Override in Application Code

```python
# ❌ WRONG - Overriding imported config
from config import MODEL_CONFIG
MODEL_CONFIG['batch_size'] = 64  # This works BUT defeats single-source-of-truth!

# ✅ CORRECT - Edit config/model_config.py directly
# Go to file, change line 39, done.
```

---

## 🧪 Testing Configuration Changes

### Validate Model Config

```python
from config.model_config import validate_config

is_valid, errors = validate_config()
if not is_valid:
    for error in errors:
        print(f"Config error: {error}")
```

### Test API Config

```python
from config.api_config import get_full_url, get_timeout

# Test URL construction
daemon_url = get_full_url('daemon', 'predictions')
print(f"Will connect to: {daemon_url}")

# Test timeout retrieval
timeout = get_timeout('prediction')
print(f"Prediction timeout: {timeout}s")
```

### Verify Metrics Config

```python
from config.metrics_config import PROFILE_BASELINES, ServerProfile

# Verify baselines exist for all profiles
for profile in ServerProfile:
    assert profile in PROFILE_BASELINES, f"Missing baselines for {profile}"
    baselines = PROFILE_BASELINES[profile]
    assert 'cpu_user' in baselines, f"Missing cpu_user for {profile}"
    print(f"✅ {profile.value}: {len(baselines)} metrics configured")
```

---

## 📚 Configuration Best Practices

1. **ONE PLACE RULE**: If a value appears in multiple places, it belongs in config/
2. **DOCUMENT CHANGES**: Add comments explaining WHY you changed a value
3. **TEST AFTER CHANGES**: Run validation functions after config edits
4. **VERSION CONTROL**: Commit config changes with descriptive messages
5. **PRODUCTION VALUES**: Use `API_CONFIG['production']` for prod settings

---

## 🔄 Migration Status

### ✅ Completed

- [x] Created `config/` package with `__init__.py`
- [x] Created `config/model_config.py` with all TFT hyperparameters
- [x] Created `config/metrics_config.py` with LINBORG baselines (2000+ lines extracted!)
- [x] Created `config/api_config.py` with all URLs and endpoints
- [x] Kept `Dashboard/config/dashboard_config.py` for modularity

### 🔧 In Progress

- [ ] Update `metrics_generator.py` to import from `config.metrics_config`
- [ ] Update `tft_trainer.py` to import from `config.model_config`
- [ ] Update dashboard to use `config.api_config`
- [ ] Archive obsolete files (`tft_config_adjusted.json`, old `config.py`)

### ⏳ Next Steps

- [ ] Add unit tests for configuration validation
- [ ] Create config diff tool to compare local vs production
- [ ] Document per-environment config overrides

---

## 🆘 Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'config'`

**Solution**: Make sure you're in the project root directory and `config/__init__.py` exists.

```bash
cd /d/machine_learning/MonitoringPrediction
python -c "from config import MODEL_CONFIG; print('Success!')"
```

### Value Not Changing

**Problem**: Changed config but code still uses old value
**Solution**: Check for these issues:

1. **Cached import**: Restart Python kernel / terminal
2. **Hardcoded override**: Search code for the variable name
3. **Wrong file**: Make sure you edited `config/` not old `config.py`

### Can't Find Setting

**Search all config files**:
```bash
grep -r "setting_name" config/
grep -r "setting_name" Dashboard/config/
```

---

## 📞 Support

**Questions**: Check this guide first, then search config files with `grep`
**Bugs**: If a setting doesn't take effect, there's likely a hardcoded override in application code
**Feature Requests**: Want a new config option? Add it to the appropriate `config/*.py` file

---

## 📝 Version History

- **v2.0.0** (Oct 16, 2025): Centralized config system, extracted 2000+ lines from metrics_generator
- **v1.0.0** (Oct 10, 2025): Initial config.py and tft_config_adjusted.json (deprecated)

---

**Remember**: If you have to ask "where do I change X?", the answer is **always** in `config/`!
