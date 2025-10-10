# Python Environment Activation - py310

**Last Updated:** 2025-10-10
**Environment:** py310 (Miniconda3)
**Python Version:** 3.10.16

---

## 🎯 The Challenge

Windows command line doesn't always have `conda` in PATH, and even when activated, UTF-8 encoding can cause issues with emojis and special characters.

---

## ✅ Working Solution (Verified Oct 10, 2025)

### Method 1: Direct Python Executable (Most Reliable)

**Used by Claude Code Assistant - Works Every Time:**

```bash
# Full path to py310 Python executable
"C:\Users\craig\miniconda3\envs\py310\python.exe" script.py

# Example: Run training
cd D:\machine_learning\MonitoringPrediction
"C:\Users\craig\miniconda3\envs\py310\python.exe" tft_trainer.py --dataset ./training/ --epochs 20

# Example: Check dependencies
"C:\Users\craig\miniconda3\envs\py310\python.exe" -c "import pytorch_forecasting; print('OK')"
```

**Advantages:**
- ✅ Works even when conda not in PATH
- ✅ No environment activation needed
- ✅ Explicit and unambiguous
- ✅ Works in automation/scripts
- ✅ Works from any working directory

**Disadvantages:**
- ❌ Long path to type
- ❌ Need to know exact conda installation location

---

### Method 2: Conda Activate (User's Typical Method)

**Requires conda in PATH:**

```bash
# Standard conda activation
conda activate py310

# Then run scripts normally
python tft_trainer.py --dataset ./training/ --epochs 20

# Deactivate when done
conda deactivate
```

**Advantages:**
- ✅ Shorter commands after activation
- ✅ Standard conda workflow
- ✅ Works for multiple commands in sequence

**Disadvantages:**
- ❌ Requires conda in PATH
- ❌ May not work in all shells
- ❌ Need to activate in each new terminal

---

## 🔍 Environment Discovery

### Finding Miniconda Installation

```bash
# Check common locations
ls "C:\Users\craig\miniconda3\python.exe"
ls "C:\Users\craig\anaconda3\python.exe"
ls "C:\ProgramData\miniconda3\python.exe"
```

**Found on this system:**
```
C:\Users\craig\miniconda3\python.exe  ✅ EXISTS
```

### Finding py310 Environment

```bash
# List all conda environments
ls "C:\Users\craig\miniconda3\envs"

# Output:
# drwxr-xr-x automatic1111
# drwxr-xr-x llms
# drwxr-xr-x py310  ✅ FOUND
```

**Full path to py310 Python:**
```
C:\Users\craig\miniconda3\envs\py310\python.exe
```

---

## 🧪 Verification Commands

### Check Python Version
```bash
"C:\Users\craig\miniconda3\envs\py310\python.exe" --version
# Output: Python 3.10.16
```

### Check Executable Location
```bash
"C:\Users\craig\miniconda3\envs\py310\python.exe" -c "import sys; print(sys.executable)"
# Output: C:\Users\craig\miniconda3\envs\py310\python.exe
```

### Check Installed Packages
```bash
"C:\Users\craig\miniconda3\envs\py310\python.exe" -c "
import pytorch_forecasting
import fastapi
import uvicorn
import pandas
import torch
print('All dependencies available')
"
# Output: All dependencies available
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Unicode Encoding Errors

**Problem:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'
```

**Cause:** Windows console uses CP1252 encoding, can't handle UTF-8 emojis

**Solution:**
- ✅ Replace all emojis with ASCII: `[OK]`, `[ERROR]`, `[INFO]`
- ❌ Don't use: ✅ ❌ 🚀 📊 etc.

**Applied to:**
- `tft_trainer.py` - All emojis replaced
- `tft_inference.py` - All emojis replaced

### Issue 2: Conda Not Found

**Problem:**
```bash
conda activate py310
# Output: conda: command not found
```

**Solution:** Use direct Python executable path instead (Method 1)

### Issue 3: Wrong Python Version

**Problem:** Script runs but uses wrong Python (e.g., Python 3.13 instead of 3.10)

**Check:**
```bash
python --version  # Might show 3.13
"C:\Users\craig\miniconda3\envs\py310\python.exe" --version  # Shows 3.10.16
```

**Solution:** Always use full path to ensure correct environment

---

## 📋 Quick Reference

### Running Scripts

**Training:**
```bash
cd D:\machine_learning\MonitoringPrediction
"C:\Users\craig\miniconda3\envs\py310\python.exe" tft_trainer.py --dataset ./training/ --epochs 20
```

**Inference Daemon:**
```bash
cd D:\machine_learning\MonitoringPrediction
"C:\Users\craig\miniconda3\envs\py310\python.exe" tft_inference.py --daemon --port 8000
```

**Dashboard:**
```bash
cd D:\machine_learning\MonitoringPrediction
"C:\Users\craig\miniconda3\envs\py310\python.exe" tft_dashboard_refactored.py training/server_metrics.parquet
```

**Data Generation:**
```bash
cd D:\machine_learning\MonitoringPrediction
"C:\Users\craig\miniconda3\envs\py310\python.exe" metrics_generator.py --servers 25 --hours 72 --output ./training/
```

---

## 🔧 Environment Setup (For Reference)

### If py310 Doesn't Exist

**Create from scratch:**
```bash
# With conda in PATH
conda create -n py310 python=3.10

# Activate
conda activate py310

# Install dependencies
pip install torch lightning pytorch-forecasting safetensors
pip install fastapi uvicorn[standard] websockets
pip install pandas numpy matplotlib pyarrow
```

### Verify Installation

```bash
"C:\Users\craig\miniconda3\envs\py310\python.exe" -c "
import sys
print('Python:', sys.version)

# Check all required packages
packages = [
    'torch', 'lightning', 'pytorch_forecasting', 'safetensors',
    'fastapi', 'uvicorn', 'pandas', 'numpy', 'matplotlib', 'pyarrow'
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f'  {pkg:20s} OK')
    except ImportError as e:
        print(f'  {pkg:20s} MISSING')
"
```

---

## 💡 Best Practices

### For Interactive Development
```bash
# Use conda activate for multiple commands
conda activate py310
python script1.py
python script2.py
python script3.py
conda deactivate
```

### For Automation/Scripts
```bash
# Use direct path for reliability
"C:\Users\craig\miniconda3\envs\py310\python.exe" script.py
```

### For Claude Code Assistant
```bash
# Always use direct path (no conda in PATH)
cd "D:\machine_learning\MonitoringPrediction"
"C:\Users\craig\miniconda3\envs\py310\python.exe" <script> <args>
```

---

## 🗺️ Path Variables

**Environment-specific paths on this system:**

| Variable | Path |
|----------|------|
| Miniconda Root | `C:\Users\craig\miniconda3\` |
| py310 Environment | `C:\Users\craig\miniconda3\envs\py310\` |
| py310 Python | `C:\Users\craig\miniconda3\envs\py310\python.exe` |
| py310 Scripts | `C:\Users\craig\miniconda3\envs\py310\Scripts\` |
| py310 Packages | `C:\Users\craig\miniconda3\envs\py310\Lib\site-packages\` |
| Project Root | `D:\machine_learning\MonitoringPrediction\` |

---

## 📊 Verified Working Examples

### October 10, 2025 Session

**All these commands executed successfully:**

```bash
# 1. Check dependencies
"C:\Users\craig\miniconda3\envs\py310\python.exe" -c "import pytorch_forecasting; import fastapi; import uvicorn; print('All dependencies available')"
# ✅ SUCCESS

# 2. Test model loading
"C:\Users\craig\miniconda3\envs\py310\python.exe" test_model_loading.py
# ✅ SUCCESS (found schema mismatch - as expected)

# 3. Start training (validation)
"C:\Users\craig\miniconda3\envs\py310\python.exe" tft_trainer.py --dataset ./training/ --epochs 2
# ✅ SUCCESS - Training started, loaded 432,000 records, GPU detected

# 4. Data inspection
"C:\Users\craig\miniconda3\envs\py310\python.exe" -c "import pandas as pd; df = pd.read_parquet('training/server_metrics.parquet'); print(df.shape)"
# ✅ SUCCESS - (432000, 15)
```

---

## 🎯 TL;DR

**For Claude Code Assistant (Automated):**
```bash
"C:\Users\craig\miniconda3\envs\py310\python.exe" script.py
```

**For Interactive Use (Human):**
```bash
conda activate py310
python script.py
```

**Both work, but direct path is more reliable for automation!**

---

**Document Created:** 2025-10-10
**Last Verified:** 2025-10-10 Morning Session
**Status:** ✅ Working and tested
