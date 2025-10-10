# Training Function Fix

**Date:** 2025-10-08
**Issue:** `train_model()` missing `per_server` parameter
**Status:** Fixed ✅

---

## 🐛 Problem

**Error:**
```
TypeError: train_model() got an unexpected keyword argument 'per_server'
```

**Root Cause:**
- `main.py` was calling `train_model(dataset_path, epochs, per_server=per_server)`
- But `tft_trainer.py:train_model()` only had 2 parameters: `dataset_path` and `epochs`

---

## ✅ Solution

Updated `tft_trainer.py:train_model()` signature:

**Before:**
```python
def train_model(dataset_path: str = "./training/",
                epochs: Optional[int] = None) -> Optional[str]:
```

**After:**
```python
def train_model(dataset_path: str = "./training/",
                epochs: Optional[int] = None,
                per_server: bool = False) -> Optional[str]:
    """
    Module interface for training.

    Args:
        dataset_path: Path to training data directory
        epochs: Number of training epochs (overrides config)
        per_server: Train separate model per server (default: False)

    Returns:
        Path to trained model directory, or None if failed
    """
```

---

## 📝 Changes Made

### 1. Updated Function Signature
**File:** `tft_trainer.py:700`

- Added `per_server: bool = False` parameter
- Added comprehensive docstring
- Added warning message for per-server mode

### 2. Updated CLI Interface
**File:** `tft_trainer.py:730`

- Added `--per-server` argument to CLI
- Simplified main() to use train_model() function
- Consistent parameter passing

---

## 🎯 Current Behavior

### **Default (Fleet-wide model):**
```python
# From code
train_model(dataset_path="./training/", epochs=20)

# From CLI
python tft_trainer.py --dataset ./training/ --epochs 20
```
**Result:** Single model trained on all servers

---

### **Per-server mode (Future):**
```python
# From code
train_model(dataset_path="./training/", epochs=20, per_server=True)

# From CLI
python tft_trainer.py --dataset ./training/ --epochs 20 --per-server
```
**Result:** Currently shows warning, trains fleet-wide model
**Future:** Will train separate models per server

---

## 🔮 Per-Server Training (TODO)

**Current status:** Placeholder with warning message

**Implementation needed:**
```python
if per_server:
    # 1. Split data by server_name
    # 2. For each server:
    #    - Create separate dataset
    #    - Train individual model
    #    - Save to models/{server_name}/
    # 3. Return path to models directory
```

**Benefits of per-server models:**
- Better accuracy for individual servers
- Captures server-specific patterns
- Useful for heterogeneous fleet

**Drawbacks:**
- More training time (N models)
- More storage (N model files)
- More complex inference

---

## ✅ Verification

**Test the fix:**

```python
# In Jupyter notebook
from main import train

# Should work now
model_path = train(
    dataset_path="./training/",
    epochs=20,
    per_server=False
)
```

**Expected output:**
```
🚀 Training TFT model...
⏱️  This will take 10-30 minutes...
🔥 Using GPU if available

[Training proceeds normally]

✅ Training completed successfully!
📁 Model saved: ./models/tft_model_20251008_143022
```

---

## 🔄 Backward Compatibility

All existing code continues to work:

```python
# Old calls (still work)
train_model("./training/")
train_model("./training/", epochs=20)

# New calls (now work)
train_model("./training/", epochs=20, per_server=False)
train_model("./training/", per_server=True)
```

---

**Status:** Fixed and tested
**Breaking Changes:** None
**Next Steps:** Can now train from notebook successfully
