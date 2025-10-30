# NordIQ Python Requirements

This document explains the different requirement files and what you need to install depending on your use case.

---

## Quick Install Guide

### Production Deployment (Dashboard + Inference Only)

If you're deploying NordIQ in production and **NOT** training models:

```bash
# Install dashboard requirements
pip install -r requirements_dashboard.txt

# Install inference daemon requirements
pip install -r requirements_inference.txt
```

**Total packages:** ~10-12 packages

---

### Full Development Environment (Including Training)

If you need to **train models** or do full development:

```bash
# Install all requirements
pip install -r requirements.txt
```

**Total packages:** ~30+ packages (includes PyTorch Lightning, TensorBoard, etc.)

---

## Requirement Files Explained

### 1. `requirements_dashboard.txt` (Dash Dashboard Only)

**Use case:** Running the Dash web dashboard (port 8050)

**Packages:**
- `dash` - Dashboard framework (16x faster than Streamlit)
- `dash-bootstrap-components` - UI components
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `plotly` - Visualization (included with Dash)
- `requests` - HTTP client for API calls

**Install:**
```bash
pip install -r requirements_dashboard.txt
```

---

### 2. `requirements_inference.txt` (Inference Daemon)

**Use case:** Running the TFT inference daemon (port 8000) and metrics generator (port 8001)

**Packages:**
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `torch` - PyTorch (inference only)
- `safetensors` - Model loading
- `pandas` - Data processing
- `numpy` - Numerical operations
- `requests` - HTTP client
- `slowapi` - Rate limiting (optional)

**Install:**
```bash
pip install -r requirements_inference.txt
```

**Note:** For GPU acceleration, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. `requirements.txt` (Complete System)

**Use case:** Full development, including model training

**Additional packages beyond inference:**
- `pytorch-forecasting` - TFT model implementation
- `pytorch-lightning` - Training framework
- `tensorboard` - Training visualization
- `scikit-learn` - ML utilities
- `streamlit` - Alternative dashboard (slower but feature-rich)
- Plus training-specific dependencies

**Install:**
```bash
pip install -r requirements.txt
```

---

## Minimal Production Setup

For the absolute minimum production deployment:

```bash
# Dashboard only
pip install dash dash-bootstrap-components pandas numpy requests

# Inference daemon only
pip install fastapi uvicorn pydantic torch safetensors pandas numpy requests
```

---

## Installation by Environment

### Conda (Recommended)

```bash
# Create environment
conda create -n py310 python=3.10

# Activate
conda activate py310

# Install based on use case
pip install -r requirements_dashboard.txt  # Dashboard only
pip install -r requirements_inference.txt  # Inference only
pip install -r requirements.txt            # Full system
```

### venv (Alternative)

```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install based on use case
pip install -r requirements_dashboard.txt  # Dashboard only
pip install -r requirements_inference.txt  # Inference only
pip install -r requirements.txt            # Full system
```

---

## Production Recommendations

### For Dashboard Server (No ML)
```bash
pip install -r requirements_dashboard.txt
```
**Size:** ~200 MB
**Time:** 2-3 minutes

### For Inference Server (ML Predictions)
```bash
pip install -r requirements_inference.txt
```
**Size:** ~2-3 GB (includes PyTorch)
**Time:** 5-10 minutes

### For Training Server (Full ML Pipeline)
```bash
pip install -r requirements.txt
```
**Size:** ~4-5 GB (includes all training tools)
**Time:** 10-15 minutes

---

## Verification

After installation, verify your setup:

```bash
# Test dashboard dependencies
python -c "import dash, dash_bootstrap_components, pandas, numpy, requests; print('Dashboard OK')"

# Test inference dependencies
python -c "import fastapi, torch, pandas, numpy; print('Inference OK')"

# Test training dependencies (if installed)
python -c "import pytorch_forecasting, pytorch_lightning; print('Training OK')"
```

---

## Troubleshooting

### PyTorch CPU vs GPU

By default, pip installs CPU-only PyTorch. For GPU support:

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Package Conflicts

If you encounter conflicts:

```bash
# Start fresh
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

---

## Summary Table

| Use Case | File | Packages | Size | Time |
|----------|------|----------|------|------|
| Dashboard Only | `requirements_dashboard.txt` | ~6 | 200 MB | 2-3 min |
| Inference Only | `requirements_inference.txt` | ~10 | 2-3 GB | 5-10 min |
| Full System | `requirements.txt` | ~30+ | 4-5 GB | 10-15 min |

---

**Recommendation:** For production deployments, use the minimal requirements files to reduce installation time and disk space.
