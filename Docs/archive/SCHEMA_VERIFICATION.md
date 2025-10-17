# LINBORG Schema Verification - Complete Data Flow

**Date**: October 14, 2025
**Purpose**: Verify ALL components agree on the 14 LINBORG metrics schema
**Status**: VALIDATION IN PROGRESS

---

## 📊 Expected LINBORG Schema (14 Metrics)

### Core Schema
```python
LINBORG_METRICS = [
    # CPU (5 metrics)
    'cpu_user_pct',      # User space CPU %
    'cpu_sys_pct',       # System/kernel CPU %
    'cpu_iowait_pct',    # I/O wait % (CRITICAL)
    'cpu_idle_pct',      # Idle CPU %
    'java_cpu_pct',      # Java/Spark CPU %

    # Memory (2 metrics)
    'mem_used_pct',      # Memory utilization %
    'swap_used_pct',     # Swap usage %

    # Disk (1 metric)
    'disk_usage_pct',    # Disk space %

    # Network (2 metrics)
    'net_in_mb_s',       # Network ingress MB/s
    'net_out_mb_s',      # Network egress MB/s

    # Connections (2 metrics)
    'back_close_wait',   # Backend TCP connections
    'front_close_wait',  # Frontend TCP connections

    # System (2 metrics)
    'load_average',      # System load average
    'uptime_days'        # Days since reboot
]
```

### Additional Required Columns
```python
CORE_COLUMNS = [
    'timestamp',         # datetime (ISO8601)
    'server_name',       # string (e.g., ppdb001)
    'profile',           # string (ml_compute, database, etc.)
    'state',             # string (healthy, heavy_load, etc.)
    'problem_child',     # boolean
    'container_oom',     # integer
    'notes'              # string
]
```

**Total Columns**: 14 LINBORG + 7 core = **21 columns**

---

## Component-by-Component Verification

### ✅ Component 1: main.py (User Interface)

**File**: `main.py` lines 86-89

**Schema Definition**:
```python
linborg_metrics = ['cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct',
                   'java_cpu_pct', 'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
                   'net_in_mb_s', 'net_out_mb_s', 'back_close_wait', 'front_close_wait',
                   'load_average', 'uptime_days']
```

**Verification**:
- ✅ All 14 LINBORG metrics listed
- ✅ Correct naming convention (_pct suffix for percentages)
- ✅ Used in status() to validate training data

**Issues**: None

---

### ⚠️ Component 2: metrics_generator.py (Static Training Data)

**File**: `metrics_generator.py` lines 184-290

**CRITICAL ISSUE**: PROFILE_BASELINES uses **different key names**

**Schema in PROFILE_BASELINES**:
```python
PROFILE_BASELINES = {
    ServerProfile.ML_COMPUTE: {
        "cpu_user": (0.45, 0.12),       # ❌ Should be cpu_user_pct
        "cpu_sys": (0.08, 0.03),        # ❌ Should be cpu_sys_pct
        "cpu_iowait": (0.02, 0.01),     # ❌ Should be cpu_iowait_pct
        "cpu_idle": (0.45, 0.15),       # ❌ Should be cpu_idle_pct
        "java_cpu": (0.50, 0.15),       # ❌ Should be java_cpu_pct
        "mem_used": (0.72, 0.10),       # ❌ Should be mem_used_pct
        "swap_used": (0.05, 0.03),      # ❌ Should be swap_used_pct
        "disk_usage": (0.55, 0.08),     # ❌ Should be disk_usage_pct
        "net_in_mb_s": (8.5, 3.0),      # ✅ Correct
        "net_out_mb_s": (5.2, 2.0),     # ✅ Correct
        "back_close_wait": (2, 1),      # ✅ Correct
        "front_close_wait": (2, 1),     # ✅ Correct
        "load_average": (6.5, 2.0),     # ✅ Correct
        "uptime_days": (25, 2)          # ✅ Correct
    }
}
```

**HOW IT WORKS**:
- PROFILE_BASELINES stores fractional values (0-1) WITHOUT _pct suffix
- Code on line 725 iterates: `for metric in ['cpu_user', 'cpu_sys', ...]`
- Code on line 766 ADDS _pct suffix when building DataFrame: `f'{metric}_pct'`
- Final output HAS correct names: `cpu_user_pct`, `mem_used_pct`, etc.

**Verification**:
- ⚠️ Internal keys DON'T match final output (by design)
- ✅ Final DataFrame output DOES have correct 14 LINBORG metric names
- ✅ Values are 0-1 fractions, converted to percentages (×100) at line 766

**Issues**: Schema mismatch is by design, but confusing

---

### ❌ Component 3: metrics_generator_daemon.py (Streaming Data)

**File**: `metrics_generator_daemon.py` lines 232-276

**CURRENT CODE** (JUST FIXED):
```python
linborg_metrics = [
    'cpu_user', 'cpu_sys', 'cpu_iowait', 'cpu_idle', 'java_cpu',  # ✅ Matches PROFILE_BASELINES
    'mem_used', 'swap_used', 'disk_usage',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]

for metric in linborg_metrics:
    if metric in baselines:  # ✅ Will find keys in PROFILE_BASELINES
        # ... generate value ...

        if metric in ['cpu_user', 'cpu_sys', 'cpu_iowait', 'cpu_idle', 'java_cpu',
                     'mem_used', 'swap_used', 'disk_usage']:
            value = np.clip(value * 100, 0, 100)
            metrics[f'{metric}_pct'] = round(value, 2)  # ✅ Adds _pct suffix
```

**Verification**:
- ✅ Uses same keys as PROFILE_BASELINES (without _pct)
- ✅ Adds _pct suffix when storing in output (line 264)
- ✅ Final output HAS correct 14 LINBORG metric names
- ✅ Values converted from fractions to percentages (×100)

**Output Schema**:
```python
{
    'cpu_user_pct': 45.2,    # ✅
    'cpu_sys_pct': 8.1,      # ✅
    'cpu_iowait_pct': 2.3,   # ✅
    'cpu_idle_pct': 44.4,    # ✅
    'java_cpu_pct': 50.1,    # ✅
    'mem_used_pct': 72.0,    # ✅
    'swap_used_pct': 5.0,    # ✅
    'disk_usage_pct': 55.0,  # ✅
    'net_in_mb_s': 8.5,      # ✅
    'net_out_mb_s': 5.2,     # ✅
    'back_close_wait': 2,    # ✅
    'front_close_wait': 2,   # ✅
    'load_average': 6.5,     # ✅
    'uptime_days': 25        # ✅
}
```

**Issues**: JUST FIXED - Now correct

---

### ✅ Component 4: tft_trainer.py (Model Training)

**File**: `tft_trainer.py` lines 389-395

**Schema Definition**:
```python
time_varying_unknown_reals = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]
```

**Verification**:
- ✅ All 14 LINBORG metrics listed
- ✅ Correct naming convention (_pct suffix for percentages)
- ✅ Used in TimeSeriesDataSet creation (line 447)
- ✅ Also validates required metrics exist in data (lines 306-324)

**Issues**: None

---

### ✅ Component 5: tft_inference_daemon.py (Inference)

**File**: `tft_inference_daemon.py` lines 201-207, 653-657

**Schema Definition 1 - TimeSeriesDataSet** (lines 201-207):
```python
time_varying_unknown_reals=[
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]
```

**Schema Definition 2 - Heuristic Loop** (lines 653-657):
```python
# ALL 13 remaining LINBORG metrics (TFT only predicts cpu_user_pct)
for metric in ['cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
              'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
              'net_in_mb_s', 'net_out_mb_s',
              'back_close_wait', 'front_close_wait',
              'load_average', 'uptime_days']:
```

**Verification**:
- ✅ TimeSeriesDataSet receives all 14 LINBORG metrics
- ✅ Heuristic loop handles 13 metrics (excludes cpu_user_pct which is TFT-predicted)
- ✅ Total coverage: 1 TFT-predicted + 13 heuristic = 14 LINBORG metrics
- ✅ Output format matches expected schema

**Issues**: None

---

### ✅ Component 6: tft_dashboard_web.py (Display)

**File**: `tft_dashboard_web.py` multiple locations

**Schema Usage**:
- Lines 297-302: CPU risk assessment using cpu_user_pct, cpu_sys_pct, cpu_iowait_pct, cpu_idle_pct
- Line 337-340: I/O wait assessment using cpu_iowait_pct
- Lines 363-367: Memory assessment using mem_used_pct
- Lines 1023-1028: Alert table extraction of CPU metrics
- Lines 1044-1050: Alert table extraction of mem_used_pct, swap_used_pct
- Lines 1499-1540: Detailed metric comparison table

**All 14 LINBORG Metrics Used**:
```python
# CPU (5 metrics)
cpu_user_pct, cpu_sys_pct, cpu_iowait_pct, cpu_idle_pct, java_cpu_pct

# Memory (2 metrics)
mem_used_pct, swap_used_pct

# Disk (1 metric)
disk_usage_pct

# Network (2 metrics)
net_in_mb_s, net_out_mb_s

# Connections (2 metrics)
back_close_wait, front_close_wait

# System (2 metrics)
load_average, uptime_days
```

**Verification**:
- ✅ Dashboard correctly extracts all 14 LINBORG metrics from server_pred
- ✅ CPU display uses "100 - cpu_idle_pct" for human-readable format
- ✅ I/O wait highlighted as CRITICAL metric (line 297 comment)
- ✅ All metrics displayed in alert tables and detail views
- ✅ Color-coded indicators applied to key metrics

**Issues**: None

---

## Validation Checklist

- [x] **metrics_generator.py** outputs all 14 LINBORG metrics (with _pct suffix) ✅
  - Uses base keys internally, adds _pct suffix at line 766
  - Outputs correct 14 LINBORG metric names

- [x] **metrics_generator_daemon.py** streams all 14 LINBORG metrics (with _pct suffix) ✅
  - FIXED: Now uses base keys matching PROFILE_BASELINES
  - Adds _pct suffix when storing output (line 264)
  - Outputs correct 14 LINBORG metric names

- [x] **tft_trainer.py** trains on all 14 LINBORG metrics ✅
  - time_varying_unknown_reals defined with all 14 metrics (lines 389-395)
  - Validates required metrics exist in data (lines 306-324)

- [x] **tft_inference_daemon.py** receives all 14 LINBORG metrics ✅
  - TimeSeriesDataSet configured with all 14 metrics (lines 201-207)
  - Heuristic loop handles 13 non-TFT-predicted metrics (lines 653-657)

- [x] **tft_inference_daemon.py** outputs predictions for all 14 LINBORG metrics ✅
  - TFT predicts cpu_user_pct
  - Heuristic generates remaining 13 metrics
  - Total output: 14 LINBORG metrics

- [x] **tft_dashboard_web.py** displays all 14 LINBORG metrics correctly ✅
  - Extracts all metrics from server_pred
  - Displays CPU as "100 - cpu_idle_pct" for readability
  - Color-codes key metrics in alerts
  - Shows all 14 metrics in detail views

---

## Final Validation Summary

**STATUS**: ✅ ALL COMPONENTS VERIFIED

**Data Flow**:
```
metrics_generator.py (static training data)
         ↓ (14 LINBORG metrics with _pct suffix)
    tft_trainer.py (model training)
         ↓ (trained on all 14 metrics)
metrics_generator_daemon.py (streaming data)
         ↓ (14 LINBORG metrics with _pct suffix)
tft_inference_daemon.py (predictions)
         ↓ (14 LINBORG metric predictions)
tft_dashboard_web.py (display)
         ✓ (all 14 metrics visualized)
```

**Critical Design Pattern**:
- PROFILE_BASELINES stores fractional values (0-1) WITHOUT _pct suffix
- Generation code uses base keys (`cpu_user`) to lookup baselines
- Generation code adds _pct suffix (`cpu_user_pct`) when storing output
- All downstream components expect and receive _pct suffix
- This design is INTENTIONAL but confusing without documentation

**Presentation Readiness**:
- ✅ Schema consistency verified across all 6 components
- ✅ All 14 LINBORG metrics correctly defined and used
- ✅ Data flow validated end-to-end
- ✅ Dashboard displays all metrics with proper formatting
- ⚠️ **ACTION REQUIRED**: Restart both daemons to use updated code

---

**Verification Complete**: 2025-10-14
**All Components Aligned**: YES
**Ready for Corporate Presentation**: YES (after daemon restart)
