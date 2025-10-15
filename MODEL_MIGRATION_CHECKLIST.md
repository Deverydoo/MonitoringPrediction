# Model Migration Checklist - Home to Work Laptop

**Date**: October 15, 2025
**Source**: Home computer (training)
**Destination**: Work laptop (presentation)
**Model**: 3-epoch, 2-week training on LINBORG metrics

---

## 📦 Files to Transfer

### Critical Model Files (MUST TRANSFER)

From home computer `models/tft_model_YYYYMMDD_HHMMSS/`:

- [ ] `model.safetensors` - Model weights (~88K parameters)
- [ ] `config.json` - Model architecture configuration
- [ ] `training_info.json` - Training metadata (epochs, samples, contract version)
- [ ] `server_mapping.json` - **CRITICAL** for decoding server names
- [ ] `checkpoint.pth` (if exists) - PyTorch checkpoint

**All 5 files MUST be in the same directory structure on work laptop**

### Supporting Files (Optional but Recommended)

- [ ] `training/server_metrics.parquet` - Training data (for reference)
- [ ] `training/metrics_metadata.json` - Generation metadata
- [ ] `training/server_mapping.json` - Name↔ID mapping backup

---

## 🚀 Transfer Methods

### Option 1: USB Drive (Recommended for Speed)
```bash
# On home computer
cd D:\machine_learning\MonitoringPrediction
mkdir transfer_package
cp -r models/tft_model_YYYYMMDD_HHMMSS transfer_package/
cp training/server_metrics.parquet transfer_package/
cp training/*.json transfer_package/

# Copy transfer_package to USB drive
# On work laptop, reverse the process
```

### Option 2: Network Share
```bash
# Compress first (faster transfer)
tar -czf model_package.tar.gz models/tft_model_YYYYMMDD_HHMMSS training/*.json

# Transfer via network share or cloud
# On work laptop:
tar -xzf model_package.tar.gz
```

### Option 3: Git LFS (if repo configured)
```bash
git lfs track "*.safetensors"
git lfs track "*.parquet"
git add models/tft_model_YYYYMMDD_HHMMSS/
git commit -m "feat: add 3-epoch 2-week LINBORG model"
git push

# On work laptop:
git pull
```

---

## ✅ Verification Steps (Work Laptop)

### Step 1: Verify File Integrity

```bash
cd D:\machine_learning\MonitoringPrediction

# Check all critical files exist
ls models/tft_model_YYYYMMDD_HHMMSS/model.safetensors
ls models/tft_model_YYYYMMDD_HHMMSS/config.json
ls models/tft_model_YYYYMMDD_HHMMSS/training_info.json
ls models/tft_model_YYYYMMDD_HHMMSS/server_mapping.json
```

Expected output:
```
✓ model.safetensors (size: ~350KB)
✓ config.json (size: ~2KB)
✓ training_info.json (size: ~1KB)
✓ server_mapping.json (size: ~1KB)
```

### Step 2: Update Latest Model Symlink

```bash
# Windows (run as admin or use mklink):
cd models
del tft_model_latest
mklink /D tft_model_latest tft_model_YYYYMMDD_HHMMSS

# Or manually update paths in code to point to new model
```

### Step 3: Test Model Loading

```python
# Quick test in Python console
from pathlib import Path
import json

model_dir = Path("models/tft_model_YYYYMMDD_HHMMSS")

# Check server_mapping.json
with open(model_dir / "server_mapping.json") as f:
    mapping = json.load(f)
    print(f"✓ Server mapping loaded: {len(mapping['name_to_id'])} servers")

# Check training_info.json
with open(model_dir / "training_info.json") as f:
    info = json.load(f)
    print(f"✓ Training info:")
    print(f"  - Epochs: {info['epochs']}")
    print(f"  - Servers: {info['num_servers']}")
    print(f"  - Contract version: {info['data_contract_version']}")
```

Expected output:
```
✓ Server mapping loaded: 20 servers
✓ Training info:
  - Epochs: 3
  - Servers: 20
  - Contract version: 2.0.0
```

### Step 4: Test Inference Daemon

```bash
# Terminal 1: Start inference daemon
python tft_inference_daemon.py --port 8000

# Expected output:
# 🚀 TFT Inference Daemon Starting...
# ✅ Model loaded: models/tft_model_YYYYMMDD_HHMMSS
# ✅ Server encoder loaded: 20 servers
# 🌐 Daemon started on port 8000
```

```bash
# Terminal 2: Test health endpoint
curl http://localhost:8000/status

# Expected JSON response:
# {
#   "running": true,
#   "model_loaded": true,
#   "warmup": {
#     "is_warmed_up": false,
#     "records_needed": 150
#   }
# }
```

### Step 5: Test Full Stack

```bash
# Terminal 1: Inference daemon (already running)
python tft_inference_daemon.py --port 8000

# Terminal 2: Metrics generator
python metrics_generator_daemon.py --stream --servers 20 --scenario healthy

# Terminal 3: Dashboard
streamlit run tft_dashboard_web.py

# Browser: http://localhost:8501
```

**Verify Dashboard Shows:**
- [ ] Fleet Risk Distribution chart displays
- [ ] Top 5 Busiest Servers table has NON-ZERO values
- [ ] All 14 LINBORG metrics visible (CPU, I/O Wait, Memory, etc.)
- [ ] Color-coding works (🟡 🟠 🔴)
- [ ] No error messages in any terminal

---

## 🚨 Troubleshooting

### Problem: "Model not found"
**Fix**: Update model path in `tft_inference_daemon.py` line ~50:
```python
model_dir = Path("models/tft_model_YYYYMMDD_HHMMSS")  # Use exact timestamp
```

### Problem: "server_mapping.json not found"
**Cause**: File didn't transfer completely
**Fix**: Re-copy the entire model directory from home computer

### Problem: "Contract version mismatch"
**Cause**: Training used different contract version
**Fix**: Check `training_info.json` - should show `"data_contract_version": "2.0.0"`

### Problem: Dashboard shows all zeros
**Cause**: Model warming up (needs 150 records)
**Fix**: Wait 30 seconds for warmup to complete

### Problem: "Dimension mismatch" error
**Cause**: Model trained with different number of metrics
**Fix**: Ensure home computer used LINBORG metrics (14 total), not old 4-metric system

---

## 📋 Pre-Presentation Checklist

**1 Hour Before Presentation:**

- [ ] Model directory copied to work laptop
- [ ] All 5 critical files verified
- [ ] Inference daemon starts without errors
- [ ] Metrics generator streams data successfully
- [ ] Dashboard loads and displays real values
- [ ] Can switch scenarios (healthy → degrading → critical)
- [ ] Color-coding works correctly
- [ ] No Python errors in any terminal

**30 Minutes Before:**

- [ ] Run full demo sequence once
- [ ] Take screenshots as backup
- [ ] Close unnecessary applications
- [ ] Test presentation mode transitions
- [ ] Have backup screen recording ready

**5 Minutes Before:**

- [ ] All 3 daemons running
- [ ] Dashboard showing healthy state
- [ ] Browser in fullscreen mode
- [ ] Terminal windows organized for quick switching

---

## 📝 Model Training Summary (for Reference)

**Home Computer Training:**
- Duration: ~X hours (record actual time)
- Epochs: 3
- Training data: 2 weeks (336 hours)
- Servers: 20 across 7 profiles
- Metrics: 14 LINBORG production metrics
- Expected accuracy: 75-80% (3 epochs is proof-of-concept quality)
- Target for production: 20 epochs → 85-90% accuracy

**Training Metrics to Note:**
- Train Loss: ~X.XX (record from training output)
- Val Loss: ~X.XX (record from training output)
- Training time per epoch: ~X minutes

**For Presentation:**
- "This model trained for 3 epochs on 2 weeks of data"
- "Proof-of-concept quality: 75-80% accuracy expected"
- "Production model (20 epochs) would achieve 85-90%"
- "Even at 3 epochs, demonstrates real predictive capability"

---

## 🎯 Success Criteria

Before presenting, confirm:

✅ **Model loads**: No errors in inference daemon startup
✅ **Predictions work**: Dashboard shows real metric values (not zeros)
✅ **Server names decode**: Shows "ppml0001", not "UNKNOWN_12345"
✅ **Risk scores realistic**: Values between 0-100, not all 100
✅ **Scenarios work**: Can switch between healthy/degrading/critical
✅ **No crashes**: System runs stable for 5+ minutes

---

**Migration Date**: _______________
**Verified By**: _______________
**Model Timestamp**: tft_model_YYYYMMDD_HHMMSS (fill in after training)
**Status**: Ready for presentation ✅

---

Good luck with the presentation! 🚀
