# Quick Start Guide

**TFT Monitoring Prediction System**
**Version:** 2.0 (Contract-Based)
**Last Updated:** 2025-10-11

---

## 🚀 30-Second Start

```bash
# Activate environment
conda activate py310

# Start daemon (uses latest trained model)
python tft_inference.py --daemon --port 8000

# Launch web dashboard (new terminal)
python dash_app.py

# Open browser: http://localhost:8501
```

That's it! ✅

---

## 📋 Full Setup (First Time)

### 1. Generate Training Data

```bash
# Generate 24 hours of metrics for 25 servers
python metrics_generator.py --servers 25 --hours 24 --output ./training/
```

**What it creates:**
- ✅ `training/server_metrics.parquet` - Training data
- ✅ `training/server_mapping.json` - Server name encoder
- ✅ `training/metrics_metadata.json` - Dataset info

**Time:** ~30-60 seconds

### 2. Train Model

```bash
# Train for 20 epochs (recommended)
python tft_trainer.py --dataset ./training/ --epochs 20
```

**What it creates:**
- ✅ `models/tft_model_YYYYMMDD_HHMMSS/model.safetensors` - TFT weights
- ✅ `models/tft_model_YYYYMMDD_HHMMSS/server_mapping.json` - Encoder (copied from training)
- ✅ `models/tft_model_YYYYMMDD_HHMMSS/training_info.json` - Contract version, states
- ✅ `models/tft_model_YYYYMMDD_HHMMSS/config.json` - Model architecture

**Time:** ~30-40 minutes on RTX 4090

### 3. Start Inference Daemon

```bash
python tft_inference.py --daemon --port 8000
```

**What it does:**
- ✅ Loads latest trained model
- ✅ Validates contract compatibility
- ✅ Starts REST API server on port 8000
- ✅ Provides real-time predictions

**Runs:** Until you stop it (Ctrl+C)

### 4. Launch Web Dashboard

```bash
# In a new terminal
python dash_app.py
```

**What it shows:**
- 📊 Fleet health overview
- 🔥 Server risk heatmap
- ⚠️ Top problem servers with predictions
- 📈 Historical trends
- ⚙️ Model information

**Access:** http://localhost:8501

---

## 🎓 Alternative: Use Notebook

```bash
# Launch Jupyter
jupyter notebook _StartHere.ipynb

# Run cells in order:
# Cell 6: Generate data
# Cell 7: Train model
# Cell 9: Dashboard (terminal-based)
```

---

## 🎨 Web Dashboard Features

### Main Tabs

1. **Overview** - Fleet status, incident probabilities
2. **Heatmap** - Visual risk grid for all servers
3. **Top Servers** - Problem servers with TFT predictions
4. **Historical** - Trend charts and analysis
5. **Advanced** - Settings, debug, model info

### Demo Mode

1. Click "Enable Demo Mode" in sidebar
2. Choose scenario:
   - **Stable** - Healthy baseline
   - **Degrading** - Gradual resource exhaustion
   - **Critical** - Acute failures
3. Click "Start Demo"
4. Watch predictions evolve in real-time!

---

## 🛠️ Common Tasks

### Check System Status

```bash
python main.py status
```

Shows:
- Available models
- Training data
- GPU status
- Dependencies

### Generate More Training Data

```bash
# 1 week of data
python metrics_generator.py --servers 25 --hours 168 --output ./training/

# 30 days of data (best performance)
python metrics_generator.py --servers 25 --hours 720 --output ./training/
```

### Retrain Model

```bash
# After generating new data
python tft_trainer.py --dataset ./training/ --epochs 20

# Restart daemon to use new model
# (Ctrl+C to stop, then restart)
python tft_inference.py --daemon --port 8000
```

### Test Prediction API

```bash
# Check health
curl http://localhost:8000/health

# Get predictions
curl http://localhost:8000/predictions/current

# Get alerts
curl http://localhost:8000/alerts/active
```

---

## 🔍 Troubleshooting

### "Cannot connect to daemon"

```bash
# Check if daemon is running
curl http://localhost:8000/health

# If not, start it:
python tft_inference.py --daemon --port 8000
```

### "Model not found"

```bash
# Check for models
ls models/

# If no models, train one:
python tft_trainer.py --dataset ./training/ --epochs 20
```

### "server_mapping.json not found"

Your model was trained before contract implementation.

**Fix:**
```bash
# Regenerate training data
python metrics_generator.py --servers 25 --hours 24 --output ./training/

# Retrain model
python tft_trainer.py --dataset ./training/ --epochs 20
```

### "Contract version mismatch"

Model trained with different contract version.

**Fix:** Retrain with current code

---

## 📚 Documentation

- **[DATA_CONTRACT.md](Docs/DATA_CONTRACT.md)** - Schema specification (MUST READ)
- **[DASHBOARD_GUIDE.md](Docs/DASHBOARD_GUIDE.md)** - Dashboard features
- **[UNKNOWN_SERVER_HANDLING.md](Docs/UNKNOWN_SERVER_HANDLING.md)** - How unknown servers work
- **[PROJECT_SUMMARY.md](Docs/PROJECT_SUMMARY.md)** - Complete system overview
- **[CONTRACT_IMPLEMENTATION_PLAN.md](Docs/CONTRACT_IMPLEMENTATION_PLAN.md)** - Implementation details

---

## 🎯 What's Different (v2.0)

### Hash-Based Server Encoding

**Before:**
```python
server_id = 0, 1, 2, 3...  # Sequential - breaks when fleet changes
```

**After:**
```python
server_id = hash('ppvra00a01') → '957601'  # Stable, deterministic
```

### Contract Validation

All components now validate against `DATA_CONTRACT.md`:
- ✅ 8 valid states (no more mismatches!)
- ✅ Server mapping saved with model
- ✅ Contract version tracking
- ✅ Automatic validation on load

### Better Unknown Server Support

- New servers automatically encoded via hash
- TFT handles unknowns via `add_nan=True`
- Predictions decoded back to server names
- No retraining needed for individual server additions

---

## 🎓 Workflow Summary

```
┌─────────────────────────────────────────────┐
│  1. Generate Data                           │
│  python metrics_generator.py                │
│  → training/server_metrics.parquet          │
│  → training/server_mapping.json             │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  2. Train Model                             │
│  python tft_trainer.py                      │
│  → models/tft_model_*/model.safetensors     │
│  → models/tft_model_*/server_mapping.json   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  3. Start Daemon                            │
│  python tft_inference.py --daemon           │
│  → REST API: http://localhost:8000          │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  4. Launch Dashboard                        │
│  python dash_app.py         │
│  → Web UI: http://localhost:8501            │
└─────────────────────────────────────────────┘
```

---

## ⚡ Pro Tips

1. **Use Web Dashboard** - Much better than terminal version
2. **Train with 720 hours** - Best model performance
3. **Enable Demo Mode** - Great for presentations
4. **Check Contract** - Always validate before training
5. **Save Models** - Keep old models as backups

---

**Ready to go!** 🚀

Start with the 30-second start at the top, or follow the full setup for first-time use.
