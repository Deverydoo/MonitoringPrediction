# Validation Report - October 10, 2025

**Time:** Morning session
**Purpose:** Validate TFT model integration and daemon functionality

---

## ✅ What Works

### 1. Model Files Exist
- ✅ Model directory: `models/tft_model_20251008_174422/`
- ✅ Files present:
  - `model.safetensors` (398 KB)
  - `config.json`
  - `training_metadata.json`
- ✅ Training data: `training/server_metrics.parquet` (28 MB, 432,000 samples)

### 2. Dependencies Available
- ✅ Python 3.10.16 (py310 environment)
- ✅ pytorch_forecasting
- ✅ fastapi
- ✅ uvicorn
- ✅ safetensors
- ✅ pandas, torch

### 3. Code Improvements
- ✅ Fixed Unicode emoji issues in `tft_inference.py`
- ✅ All emojis replaced with ASCII symbols `[OK]`, `[ERROR]`, etc.
- ✅ No more Windows console encoding errors

---

## ❌ Critical Issue Found: Schema Mismatch

### The Problem

The system has a **three-way schema mismatch** between:
1. Generated training data
2. Trainer expectations
3. Inference model loading

### Training Data Schema (Actual)
```python
# From training/server_metrics.parquet
Columns: [
    'timestamp',
    'server_name',        # <-- String name
    'profile',            # <-- compute, production, etc.
    'state',              # <-- healthy, degraded, etc.
    'problem_child',
    'cpu_pct',            # <-- Percent metrics
    'mem_pct',
    'disk_io_mb_s',
    'net_in_mb_s',
    'net_out_mb_s',
    'latency_ms',
    'error_rate',
    'gc_pause_ms',
    'container_oom',
    'notes'
]

# Missing: time_idx, server_id (numeric), hour, day_of_week, month, is_weekend
```

### Trainer Expected Schema
```python
# From tft_trainer.py:363-365
time_varying_unknown_reals = [
    'cpu_percent',        # <-- NOT cpu_pct
    'memory_percent',     # <-- NOT mem_pct
    'disk_percent',
    'load_average'
]
time_varying_known_reals = [
    'hour', 'day_of_week', 'month', 'is_weekend'  # <-- Missing in data
]
time_varying_unknown_categoricals = ['status']  # <-- Actually 'state' in data
group_ids = ['server_id']  # <-- Actually 'server_name' in data
```

### Model Loading Error
```
RuntimeError: Error(s) in loading state_dict for TemporalFusionTransformer:
    size mismatch for input_embeddings.embeddings.status.weight:
        copying a param with shape torch.Size([8, 5]) from checkpoint,
        the shape in current model is torch.Size([1, 1]).
    size mismatch for encoder_variable_selection.flattened_grn.fc1.weight:
        copying a param with shape torch.Size([10, 149]) from checkpoint,
        the shape in current model is torch.Size([10, 145]).
```

**Interpretation:**
- Model was trained with 8 different `status` categories, but inference dummy data only has 1
- Input feature size is 149 in trained model, but 145 in dummy dataset (4 features missing)

---

## 🔍 Root Cause Analysis

### How Did This Happen?

1. **metrics_generator.py** was updated to generate data with schema version "3.0_fleet"
2. **tft_trainer.py** expects an older schema with different column names
3. **tft_inference.py** creates dummy data matching tft_trainer schema
4. **Model file** from Oct 8 was trained on the "3.0_fleet" schema
5. **Result:** Inference can't load the trained model weights

### The Disconnect

```
metrics_generator.py (Oct 8)
    ↓
training/server_metrics.parquet (3.0_fleet schema)
    ↓
tft_trainer.py (expects OLD schema)
    ↓
ERROR: Can't train because columns don't match
```

**Hypothesis:** The model from Oct 8 was either:
- A) Trained on old data with the old schema
- B) Trained after manual column renaming
- C) Never actually trained successfully with the current data

---

## 🎯 Impact

### What Doesn't Work
- ❌ Cannot load trained model (schema mismatch)
- ❌ Cannot run daemon with real TFT predictions
- ❌ System falls back to heuristics
- ❌ Violates core principle: "If it doesn't use the model, it is rejected"

### What Still Works
- ✅ Demo data generation
- ✅ Dashboard with heuristic fallback
- ✅ Data generation
- ✅ File structure

---

## 🔧 Solutions

### Option 1: Fix Schema in metrics_generator.py (Recommended)
Update `metrics_generator.py` to generate data matching what trainer expects:
- Rename `cpu_pct` → `cpu_percent`
- Rename `mem_pct` → `memory_percent`
- Rename `state` → `status`
- Rename `server_name` → `server_id` (or add numeric mapping)
- Add time features: `hour`, `day_of_week`, `month`, `is_weekend`
- Add `time_idx` (sequential integer)

**Pros:**
- Existing trainer code doesn't need changes
- Inference code doesn't need changes
- Clean solution

**Cons:**
- Need to regenerate training data
- Need to retrain model

### Option 2: Fix Trainer and Inference to Match Current Data
Update `tft_trainer.py` and `tft_inference.py` to use current schema:
- Accept `cpu_pct`, `mem_pct`, etc.
- Use `state` instead of `status`
- Use `server_name` with proper encoding
- Add time feature extraction from `timestamp`

**Pros:**
- Can use existing training data
- More flexible for future changes

**Cons:**
- More code changes
- Still need to retrain model

### Option 3: Check if Old Training Data Exists
Look for old training data that matches what the model was actually trained on.

**Pros:**
- Might work immediately if data exists

**Cons:**
- Unlikely to exist
- Would use outdated data

---

## 📋 Recommended Action Plan

### Immediate (Today)
1. ✅ Document the issue (this report)
2. ⏭️ **Decision point:** Choose Option 1 or Option 2
3. ⏭️ Implement the fix
4. ⏭️ Regenerate training data
5. ⏭️ Retrain model
6. ⏭️ Test model loading
7. ⏭️ Test end-to-end system

### Quick Win (2-3 hours)
**Go with Option 1** - Fix metrics_generator.py:
```bash
# 1. Update metrics_generator.py to match trainer schema
# 2. Generate fresh data
python metrics_generator.py --servers 15 --hours 24 --output ./training/

# 3. Train model
python tft_trainer.py --training-dir ./training/ --epochs 20

# 4. Test model loading
python test_model_loading.py

# 5. Start daemon
python tft_inference.py --daemon --port 8000

# 6. Test dashboard
python tft_dashboard_refactored.py training/server_metrics.parquet
```

---

## 📝 Files Affected

### Need Updates
- `metrics_generator.py` - Fix output schema
- `tft_inference.py` - Update `_create_dummy_dataset()` to match actual data

### Already Fixed
- `tft_inference.py` - Emoji encoding issues resolved

### Working Fine
- `tft_trainer.py` - Schema expectations are consistent
- `tft_dashboard_refactored.py` - Works with current data
- `demo_data_generator.py` - Independent, works fine

---

## 💡 Lessons Learned

1. **Schema Versioning Needed:**
   - Document schema versions clearly
   - Add schema validation in trainer
   - Store schema with model metadata

2. **Integration Testing:**
   - Need end-to-end test that validates:
     - Data generation → Training → Model loading → Inference
   - Catch mismatches early

3. **Model Metadata:**
   - Store full schema used for training with model
   - Validate incoming data matches trained schema

---

## 🎯 Current Status

**System State:**
- Model files: ✅ Present but incompatible
- Dependencies: ✅ All installed
- Schema alignment: ❌ Broken
- TFT model loading: ❌ Fails
- Daemon operation: ⚠️ Runs but uses heuristics

**Next Step:**
Choose and implement Option 1 or Option 2 to fix the schema mismatch.

---

**Report Generated:** 2025-10-10 Morning
**Validation Status:** ❌ Blocked by schema mismatch
**Recommended Action:** Fix schema, regenerate data, retrain model
