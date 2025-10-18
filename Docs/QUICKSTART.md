# Quick Start Guide - Common Tasks

**Last Updated**: 2025-10-14

This guide shows you how to perform the most common tasks in 60 seconds or less.

---

## 🚀 Adding a New Metric

**Time**: 30 seconds | **Files to edit**: 1

### Steps:

1. Open [linborg_schema.py](linborg_schema.py)
2. Find the `NordIQ Metrics Framework_METRICS` list (around line 46)
3. Add your metric to the list:

```python
NordIQ Metrics Framework_METRICS = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days',
    'your_new_metric_here'  # <-- Add it here
]
```

4. Save the file

### What Happens Automatically:

✅ Data generation uses it
✅ Model training includes it
✅ Inference daemon predicts it
✅ Dashboard displays it
✅ Validation helpers recognize it
✅ `NUM_NordIQ Metrics Framework_METRICS` updates automatically

**That's it. No other files to touch.**

---

## 🚀 Removing a Metric

**Time**: 15 seconds | **Files to edit**: 1

### Steps:

1. Open [linborg_schema.py](linborg_schema.py)
2. Find the `NordIQ Metrics Framework_METRICS` list (around line 46)
3. Delete the line with your metric
4. Save the file

**The metric is removed from the entire pipeline.**

---

## 🚀 Adding a New Server Profile

**Time**: 60 seconds | **Files to edit**: 1

### Steps:

1. Open [server_profiles.py](server_profiles.py)

2. **Add the profile to the enum** (around line 48):

```python
class ServerProfile(Enum):
    ML_COMPUTE = "ml_compute"
    DATABASE = "database"
    WEB_API = "web_api"
    CONDUCTOR_MGMT = "conductor_mgmt"
    DATA_INGEST = "data_ingest"
    RISK_ANALYTICS = "risk_analytics"
    GENERIC = "generic"
    YOUR_NEW_PROFILE = "your_profile_name"  # <-- Add here
```

3. **Add naming pattern(s)** (around line 75):

```python
PROFILE_PATTERNS = [
    (r'^ppml\d+', ServerProfile.ML_COMPUTE),
    (r'^ppdb\d+', ServerProfile.DATABASE),
    # ... existing patterns ...
    (r'^ppyour\d+', ServerProfile.YOUR_NEW_PROFILE),  # <-- Add pattern
]
```

**Pattern Examples:**
- `r'^ppapi\d+'` matches `ppapi001`, `ppapi042`, etc.
- `r'^.*-cache$'` matches `server-cache`, `prod-cache`, etc.
- `r'^db-.*-prod$'` matches `db-orders-prod`, `db-users-prod`, etc.

4. **Add baseline metrics** (optional, around line 104):

```python
PROFILE_BASELINES = {
    # ... existing baselines ...
    ServerProfile.YOUR_NEW_PROFILE: {
        'cpu_user_pct': 30.0,
        'cpu_sys_pct': 5.0,
        'mem_used_pct': 50.0,
        # ... add all 14 NordIQ Metrics Framework metrics
    }
}
```

5. Save the file

### What Happens Automatically:

✅ Data generation assigns profile to matching servers
✅ Model training uses profile as a feature
✅ Dashboard displays profile names
✅ Profile inference works immediately
✅ Pattern matching validates automatically

**That's it. No other files to touch.**

---

## 🚀 Running the System

### Generate Training Data
```bash
python metrics_generator.py --hours 24
```

**What it does**: Creates 24 hours of synthetic server metrics with all 14 NordIQ Metrics Framework metrics.

### Train the Model
```bash
python main.py train
```

**What it does**: Trains the Temporal Fusion Transformer on your data.

### Start the Inference Daemon
```bash
python tft_inference_daemon.py
```

**What it does**: Generates predictions every 5 minutes and saves to JSON.

### Start the Dashboard
```bash
streamlit run tft_dashboard_web.py
```

**What it does**: Launches the web dashboard at http://localhost:8501

---

## 🚀 Testing Your Changes

### Run the Certification Suite
```bash
# Windows
run_certification.bat

# Mac/Linux
conda activate py310
python end_to_end_certification.py
```

**What it tests**:
- ✅ Centralized schema works
- ✅ Server profiles work
- ✅ Data generation works
- ✅ Trainer integration works
- ✅ Dashboard helpers work
- ✅ Full pipeline works

**Expected Result**: 6/6 tests pass

---

## 🚀 Common Workflows

### Adding a Metric End-to-End

1. **Add metric** to [linborg_schema.py](linborg_schema.py) (30 sec)
2. **Generate data** with new metric: `python metrics_generator.py --hours 24` (2 min)
3. **Train model** with new metric: `python main.py train` (5-10 min)
4. **Test**: `run_certification.bat` (30 sec)
5. **Restart daemons** to use new model (1 min)

**Total time**: ~15 minutes (mostly model training)

### Adding a Server Profile End-to-End

1. **Add profile** to [server_profiles.py](server_profiles.py) (60 sec)
2. **Test detection**: `python -c "from server_profiles import infer_profile_from_name; print(infer_profile_from_name('ppyour001'))"` (5 sec)
3. **Generate data** with new profile: `python metrics_generator.py --hours 24` (2 min)
4. **Train model** to learn profile: `python main.py train` (5-10 min)
5. **Restart daemons** (1 min)

**Total time**: ~15 minutes (mostly model training)

---

## 🚀 File Reference

**Need to change metrics?** → [linborg_schema.py](linborg_schema.py)
**Need to change profiles?** → [server_profiles.py](server_profiles.py)
**Need to test changes?** → [end_to_end_certification.py](end_to_end_certification.py)
**Need to check results?** → [CERTIFICATION_RESULTS.md](CERTIFICATION_RESULTS.md)

---

## 🚀 Pro Tips

### Multiple Patterns for One Profile
```python
# Match multiple naming conventions
(r'^ppapi\d+', ServerProfile.WEB_API),
(r'^ppweb\d+', ServerProfile.WEB_API),
(r'^pprest\d+', ServerProfile.WEB_API),
```

### Metric Naming Conventions
- Percentages: End with `_pct` (e.g., `cpu_used_pct`)
- Counts: End with `_wait` or `_count` (e.g., `back_close_wait`)
- Continuous: Descriptive names (e.g., `load_average`)

### Baseline Values
Use realistic values for your environment:
- **CPU**: 20-40% for normal load
- **Memory**: 40-70% for normal load
- **Disk**: 30-60% for normal usage
- **Network**: 0.5-5.0 MB/s for normal traffic

---

## ❓ Need Help?

**Detailed docs**: See [ALL_OPTIMIZATIONS_FINAL.md](ALL_OPTIMIZATIONS_FINAL.md)
**Test results**: See [CERTIFICATION_RESULTS.md](CERTIFICATION_RESULTS.md)
**Architecture**: See [Docs/RAG/PROJECT_CODEX.md](Docs/RAG/PROJECT_CODEX.md)

**Questions?** Check the docs above or ask your team lead.

---

**Remember**: With centralized configuration, you only edit **one file** for metrics and **one file** for profiles. Everything else happens automatically! 🎯
