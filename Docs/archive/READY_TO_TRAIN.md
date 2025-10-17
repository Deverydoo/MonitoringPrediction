# Ready to Train - 90 Server Fleet

**Status:** Code fixes complete - READY FOR USER TO EXECUTE
**Fleet Size:** 90 servers (matches presentation)
**Date:** 2025-10-12

---

## ✅ What's Been Fixed

1. **Checkpoint resume disabled** - `tft_trainer.py` now explicitly passes `ckpt_path=None` to prevent auto-resume
2. **Demo generator updated** - All 3 demo buttons now use 90 servers (was 25)
3. **Dashboard aligned** - Demo mode will simulate 90 servers to match training data
4. **Old data cleaned** - Training directory is empty, ready for fresh dataset

---

## 📋 Step-by-Step Training Process

### Step 1: Generate Training Dataset (90 servers, 720 hours)

```bash
# Activate conda environment
conda activate py310

# Generate dataset - takes ~2-3 minutes
python metrics_generator.py \
  --hours 720 \
  --num_ml_compute 20 \
  --num_database 15 \
  --num_web_api 25 \
  --num_conductor_mgmt 5 \
  --num_data_ingest 10 \
  --num_risk_analytics 8 \
  --num_generic 7 \
  --out_dir training/
```

**Expected output:**
- `training/server_metrics.parquet` - Main dataset
- `training/metrics_metadata.json` - Metadata
- Total: 90 servers, 720 hours, ~1.5M records

**Verify:**
```bash
# Should show 7 profiles in metadata
cat training/metrics_metadata.json
```

---

### Step 2: Train Model (10-20 epochs recommended)

**Option A: Via Notebook (Recommended for monitoring)**
```bash
jupyter notebook _StartHere.ipynb
# Run Cell 6 - Training cell
```

**Option B: Via Command Line**
```bash
python tft_trainer.py --epochs 10
```

**Expected duration:**
- 10 epochs: ~1-2 hours
- 20 epochs: ~2-4 hours

**Watch for:**
```
[TRANSFER] Profile feature enabled
[TRANSFER] Profiles detected: ['conductor_mgmt', 'data_ingest', 'database', 'generic', 'ml_compute', 'risk_analytics', 'web_api']
[TRANSFER] Model configured with profile-based transfer learning
[INFO] Training from scratch (checkpoints disabled for resume)
```

**Expected output:**
- `models/tft_model_YYYYMMDD_HHMMSS/model.safetensors`
- `models/tft_model_YYYYMMDD_HHMMSS/server_mapping.json`
- `models/tft_model_YYYYMMDD_HHMMSS/training_info.json`

---

### Step 3: Start Inference Daemon

```bash
python tft_inference.py --daemon --port 8000 --fleet-size 90
```

**Expected startup:**
```
[OK] Contract validation passed (v1.0.0)
[OK] Model loaded: models/tft_model_YYYYMMDD_HHMMSS
[PROFILE] Profile feature enabled - transfer learning active
[START] TFT Inference Daemon Starting...
   Model: tft_model_YYYYMMDD_HHMMSS
   Fleet size: 90 servers
INFO: Uvicorn running on http://0.0.0.0:8000
```

**NO dimension mismatch errors!** ✅

---

### Step 4: Launch Dashboard

```bash
# New terminal
streamlit run tft_dashboard_web.py
```

**Test demo modes:**
- Click "🟢 Stable" - Should see "All 90 servers streaming"
- Click "🟡 Degrading" - Should see servers gradually degrade
- Click "🔴 Critical" - Should see rapid degradation

All demo modes now use 90 servers (matches training).

---

## 🔍 Verification Checklist

After training completes:

- [ ] No dimension mismatch errors during model load
- [ ] `training_info.json` shows 8 states (critical_issue, healthy, heavy_load, idle, maintenance, morning_spike, offline, recovery)
- [ ] Model loads in inference daemon without errors
- [ ] Dashboard connects to daemon
- [ ] Demo modes show 90 servers
- [ ] Predictions are being generated

---

## 🎯 What Changed This Session

### tft_trainer.py (Line 686-691)
```python
# Explicitly pass ckpt_path=None to prevent auto-resume from any checkpoint
trainer.fit(
    self.model,
    train_dataloader,
    val_dataloader,
    ckpt_path=None  # Force training from scratch
)
```

### tft_dashboard_web.py (Lines 144, 155, 327, 332, 337)
```python
# Changed from 25 to 90 servers
def initialize_demo_generator(scenario: str, fleet_size: int = 90):
    generator = DemoStreamGenerator(num_servers=min(fleet_size, 90), seed=None)

# All 3 demo buttons now use 90
initialize_demo_generator('stable', fleet_size=90)
initialize_demo_generator('degrading', fleet_size=90)
initialize_demo_generator('critical', fleet_size=90)
```

### Deleted
- All old checkpoints (`checkpoints/*.ckpt`)
- Old mismatched model (`models/tft_model_20251012_083743`)
- Old 35-server training data

---

## 📊 Fleet Composition (90 Total)

| Profile | Count | % | Example Servers |
|---------|-------|---|----------------|
| web_api | 25 | 28% | ppweb001-ppweb025 |
| ml_compute | 20 | 22% | ppml0001-ppml0020 |
| database | 15 | 17% | ppdb001-ppdb015 |
| data_ingest | 10 | 11% | ppetl001-ppetl010 |
| risk_analytics | 8 | 9% | pprisk001-pprisk008 |
| generic | 7 | 8% | ppgen001-ppgen007 |
| conductor_mgmt | 5 | 5% | ppcon01-ppcon05 |

**This matches:**
- Presentation claims (90 servers)
- Demo generator hardcoded list (90 servers)
- Documentation (ESSENTIAL_RAG.md, PRESENTATION_MASTER.md)

---

## 🚨 Critical Reminders

1. **DO NOT generate data or train until you're ready** - Training takes 1-2+ hours
2. **Ensure clean environment** - No old checkpoints, models, or data
3. **Watch for "[TRANSFER] Profile feature enabled"** - Confirms correct architecture
4. **Verify no dimension mismatch** - Should load cleanly in inference daemon
5. **All 90 servers** - Training data, demo mode, everything aligned

---

## 🎤 For Presentation

When you demo, the numbers will match:
- "90 servers monitored in real-time" ✅
- Demo buttons: "All 90 servers streaming" ✅
- Fleet heatmap: 90 server tiles ✅
- Training stats: "90 servers, 7 profiles" ✅

---

**Version:** 1.0
**Created:** 2025-10-12 09:10 AM
**Status:** READY - Waiting for user to execute training when ready
**Estimated Total Time:** 2-5 minutes (data) + 1-4 hours (training) = ~1.5-4 hours total

**Next:** Run Step 1 when you're ready to commit 1-4 hours to training.
