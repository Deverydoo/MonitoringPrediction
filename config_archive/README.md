# Configuration Archive

**Date Archived**: October 16, 2025
**Reason**: Migrated to centralized configuration system

---

## 📁 Archived Files

### 1. `config.py` (DEPRECATED)
**Original Purpose**: TFT model configuration and training hyperparameters

**Why Archived**:
- Configuration values scattered and duplicated
- Mixed old 4-metric system with newer settings
- No clear separation between model, metrics, and API config
- Alert thresholds outdated and inconsistent

**Replaced By**: `config/model_config.py`

**Key Values Migrated**:
- `batch_size`: 32 → `MODEL_CONFIG['batch_size']`
- `learning_rate`: 0.01 → `MODEL_CONFIG['learning_rate']`
- `epochs`: 20 → `MODEL_CONFIG['epochs']`
- `hidden_size`: 32 → `MODEL_CONFIG['hidden_size']`
- All training hyperparameters now in `config/model_config.py`

---

### 2. `tft_config_adjusted.json` (DEPRECATED)
**Original Purpose**: JSON version of config.py for external tools

**Why Archived**:
- Duplicate of config.py with slightly different values (confusion!)
- Referenced old 4-metric system (cpu_percent, memory_percent, disk_percent, load_average)
- Not compatible with new LINBORG 14-metric system
- JSON format less maintainable than Python config

**Problems**:
```json
{
  "target_metrics": [
    "cpu_percent",      // OLD - not LINBORG compatible
    "memory_percent",   // OLD - not LINBORG compatible
    "disk_percent",     // OLD - not LINBORG compatible
    "load_average"      // OLD - only 4 metrics!
  ]
}
```

**Replaced By**: `config/model_config.py` with LINBORG metrics

**New System** (14 LINBORG metrics):
```python
MODEL_CONFIG['target_metrics'] = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',  # CPU (5)
    'mem_used_pct', 'swap_used_pct',                                                     # Memory (2)
    'disk_usage_pct',                                                                     # Disk (1)
    'net_in_mb_s', 'net_out_mb_s',                                                       # Network (2)
    'back_close_wait', 'front_close_wait',                                               # Connections (2)
    'load_average', 'uptime_days'                                                        # System (2)
]
```

---

## 🔄 Migration Summary

### What Changed

| Old System | New System | Benefit |
|------------|------------|---------|
| `config.py` + `tft_config_adjusted.json` (2 sources) | `config/` package (organized modules) | Single source of truth |
| Mixed model, metrics, API config | Separated into model_config, metrics_config, api_config | Clear organization |
| Hardcoded values in modules | Centralized in config/ | Easy to find and change |
| JSON + Python mix | Pure Python (with validation) | Type safety, comments, functions |
| 4 metrics (outdated) | 14 LINBORG metrics | Production-compatible |

### Import Changes

**Old Way** (DEPRECATED):
```python
# ❌ DON'T DO THIS ANYMORE
from config import CONFIG
batch_size = CONFIG['batch_size']

# ❌ OR THIS
with open('tft_config_adjusted.json') as f:
    config = json.load(f)
    batch_size = config['batch_size']
```

**New Way** (CORRECT):
```python
# ✅ DO THIS
from config import MODEL_CONFIG
batch_size = MODEL_CONFIG['batch_size']
```

---

## 🚫 Do NOT Use These Files

These files are kept for **historical reference only**. They are:
- ❌ Not maintained
- ❌ Not compatible with current system
- ❌ Contain outdated metric definitions
- ❌ Have inconsistent values vs current config

**If you need a config value**, use the new system:
1. Check `config/model_config.py` for training/model settings
2. Check `config/metrics_config.py` for LINBORG baselines
3. Check `config/api_config.py` for URLs and ports
4. Check `Dashboard/config/dashboard_config.py` for UI settings

---

## 📚 Documentation

For the new configuration system, see:
- **CONFIG_GUIDE.md** - Complete usage guide
- **CONFIGURATION_MIGRATION_COMPLETE.md** - Migration summary

---

## 🔍 Historical Reference

These files are preserved for:
1. **Comparison**: See how configuration evolved
2. **Debugging**: Reference old values if needed
3. **Documentation**: Understand what changed and why
4. **Audit Trail**: Track configuration history

**Last Known Working Commit**: Before October 16, 2025 migration

---

## ⚠️ Warning

**DO NOT** restore these files to the project root. They will conflict with the new centralized configuration system and cause:
- Import errors
- Inconsistent behavior
- Configuration value conflicts
- System malfunction

If you need to reference old values, copy them to the new system in `config/` modules.

---

## 📅 Archive History

- **October 16, 2025**: Initial archive - migrated to centralized config system
  - Archived: `config.py`, `tft_config_adjusted.json`
  - Migration completed successfully
  - All imports verified working with new system

---

**Archive Status**: ✅ Safely archived, system fully migrated to new configuration
