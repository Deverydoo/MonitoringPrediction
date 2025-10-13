# Quick Reference Card

**One-page reference for daily use**

---

## 🚀 Quick Commands

### Activate Environment (ALWAYS FIRST!)
```bash
conda activate py310
```

### Demo (One Command)
```bash
python run_demo.py
```

### Production System
```bash
# Terminal 1: Start daemon
python tft_inference.py --daemon --port 8000

# Terminal 2: Start dashboard
python tft_dashboard_refactored.py training/server_metrics.parquet
```

### Training
```bash
# Generate data
python metrics_generator.py --servers 15 --hours 720 --output ./training/

# Train model
python tft_trainer.py --training-dir ./training/ --epochs 20
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `tft_inference.py` | ✅ Real TFT model (DAEMON MODE) |
| `tft_trainer.py` | ✅ Model training |
| `tft_dashboard_refactored.py` | ✅ Dashboard (uses daemon) |
| `metrics_generator.py` | ✅ Training data generation |
| `demo_data_generator.py` | ✅ Demo scenarios |
| `config.py` | ✅ Configuration |
| `main.py` | ✅ CLI interface |

---

## 🗑️ DO NOT USE

- ❌ `inference.py` - Heuristics only
- ❌ `enhanced_inference.py` - Not integrated
- ❌ `Imferenceloading.py` - Legacy
- ❌ `training_core.py` - Old approach

---

## 📊 Demo Scenarios

```bash
# Healthy (no incidents)
python demo_data_generator.py --scenario healthy

# Degrading (default, gradual incident)
python demo_data_generator.py --scenario degrading

# Critical (severe spikes)
python demo_data_generator.py --scenario critical
```

---

## 🔍 Troubleshooting

**"Module not found"**
```bash
conda activate py310  # Did you activate?
```

**"Model not loaded"**
- Check `models/` directory exists
- Verify `model.safetensors` file present
- Run training if needed

**"Daemon not responding"**
```bash
# Check if running
curl http://localhost:8000/health

# Restart daemon
python tft_inference.py --daemon --port 8000
```

**Dashboard shows heuristics**
- Daemon not running → Start it first
- Wrong URL → Check `--daemon-url` parameter
- Use `--no-daemon` to explicitly use fallback

---

## 📚 Documentation Shortcuts

| Need | Read |
|------|------|
| **Getting started** | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) |
| **Quick demo** | [SETUP_DEMO.md](SETUP_DEMO.md) |
| **Environment** | [PYTHON_ENV.md](PYTHON_ENV.md) |
| **Model details** | [TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md) |
| **Operations** | [OPERATIONAL_MAINTENANCE_GUIDE.md](OPERATIONAL_MAINTENANCE_GUIDE.md) |
| **All docs** | [INDEX.md](INDEX.md) |

---

## 🎯 System Status Checklist

Before starting work:
- [ ] `conda activate py310` - Environment activated
- [ ] `ls models/` - Models exist
- [ ] `ls training/` - Training data exists
- [ ] Daemon running (if using predictions)

---

## 🔄 Typical Day

### Morning Check
```bash
conda activate py310
python main.py status
```

### Run Demo for Meeting
```bash
conda activate py310
python run_demo.py
```

### Production Monitoring
```bash
conda activate py310

# Terminal 1
python tft_inference.py --daemon --port 8000

# Terminal 2
python tft_dashboard_refactored.py training/server_metrics.parquet
```

### Model Retraining
```bash
conda activate py310
python metrics_generator.py --servers 20 --hours 720 --output ./training/
python tft_trainer.py --training-dir ./training/ --epochs 20
```

---

## 🚨 Remember

> **"If it doesn't use the model, it is rejected."**

All predictions MUST use the real TFT model via the daemon.

---

## 📞 Need Help?

1. Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. Check [INDEX.md](INDEX.md)
3. Check archived docs in `archive/`

---

**Last Updated:** 2025-10-10
**Version:** 2.0.0
