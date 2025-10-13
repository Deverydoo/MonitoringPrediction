# Critical Bug Fix: 8-Server Prediction Limit

**Date:** October 13, 2025
**Severity:** Critical
**Status:** ✅ FIXED

---

## Problem Description

The TFT inference daemon was only generating predictions for 8 out of 20 servers, despite:
- Successfully creating prediction dataset with 20 samples
- Model successfully predicting for all 20 servers
- All 20 servers being warmed up and ready

### Symptoms

```
[DEBUG] Prediction dataset created with 20 samples
[DEBUG] Running TFT model prediction...
[DEBUG] TFT model prediction complete
[WARNING] Pred tensor too small: idx=8, tensor_len=8, stopping early
[OK] TFT predictions generated for 8 servers  ❌ Should be 20!
```

The dashboard would show timeout errors or only display 8 servers instead of the full fleet of 20.

---

## Root Cause Analysis

### The Investigation

After extensive debugging with obsessive attention to detail, we discovered the issue was **NOT** a hardcoded value in our code, but rather a misunderstanding of PyTorch Forecasting's output structure.

### Key Discovery

The `model.predict()` method returns a `pytorch_forecasting.models.base_model.Prediction` object, which has an `.output` attribute that is an `Output` namedtuple with **8 fields**:

```python
('prediction', 'encoder_attention', 'decoder_attention', 'static_variables',
 'encoder_variables', 'decoder_variables', 'decoder_lengths', 'encoder_lengths')
```

### The Bug

The code was checking `len(pred_tensor)` which returned **8** (the number of namedtuple fields), NOT the number of server predictions:

```python
# BROKEN CODE
pred_tensor = raw_predictions.output  # This is the Output namedtuple

for idx, server_id in enumerate(servers):  # servers has 20 items
    if idx >= len(pred_tensor):  # len(pred_tensor) = 8 (number of fields!)
        print(f"[WARNING] Pred tensor too small: idx={idx}, tensor_len={len(pred_tensor)}")
        break  # ❌ Stops at index 8!
```

### The Actual Data Structure

The **actual prediction tensor** with all 20 servers was nested inside the namedtuple:

```python
pred_tensor.prediction  # torch.Size([20, 96, 7])
                        # 20 servers × 96 timesteps × 7 quantiles
```

---

## The Fix

### Code Changes

**File:** `tft_inference_daemon.py` (lines 536-559)

```python
# CRITICAL FIX: If pred_tensor is an Output namedtuple, extract the actual prediction tensor
if hasattr(pred_tensor, 'prediction'):
    actual_predictions = pred_tensor.prediction
    print(f"[FIX] Using pred_tensor.prediction (shape: {actual_predictions.shape})")
else:
    actual_predictions = pred_tensor
    print(f"[FIX] Using pred_tensor directly")

for idx, server_id in enumerate(servers):
    # Check against the actual prediction tensor, not the namedtuple length
    if idx >= len(actual_predictions):  # ✅ Now checks against 20, not 8!
        print(f"[WARNING] Pred tensor too small: idx={idx}, tensor_len={len(actual_predictions)}")
        break

    # ... rest of prediction processing
    if hasattr(actual_predictions, 'dim') and actual_predictions.dim() >= 2:
        pred_values = actual_predictions[idx].cpu().numpy()  # ✅ Uses actual_predictions
```

### What Changed

1. **Extract nested tensor:** Access `pred_tensor.prediction` to get the actual prediction data
2. **Use correct length check:** Compare against `len(actual_predictions)` (20) instead of `len(pred_tensor)` (8)
3. **Index correct tensor:** Use `actual_predictions[idx]` instead of `pred_tensor[idx]`

---

## Verification

### Before Fix

```
[DEBUG] Formatting predictions: 20 servers
[DEBUG] pred_tensor type: <class 'pytorch_forecasting.utils.TupleOutputMixIn.to_network_output.<locals>.Output'>
[DEBUG] len(pred_tensor): 8  ❌
[WARNING] Pred tensor too small: idx=8, tensor_len=8, stopping early
[OK] TFT predictions generated for 8 servers  ❌
```

### After Fix

```
[DEBUG] Formatting predictions: 20 servers
[DEBUG] pred_tensor.prediction.shape: torch.Size([20, 96, 7])
[FIX] Using pred_tensor.prediction (shape: torch.Size([20, 96, 7]))
[OK] TFT predictions generated for 20 servers  ✅
```

---

## Technical Details

### PyTorch Forecasting Output Structure

```python
# model.predict() returns:
Prediction(
    output=Output(
        prediction=torch.Tensor([20, 96, 7]),      # ← THE ACTUAL DATA WE NEED!
        encoder_attention=torch.Tensor([...]),     # Attention weights
        decoder_attention=torch.Tensor([...]),     # Attention weights
        static_variables=torch.Tensor([...]),      # Static features
        encoder_variables=torch.Tensor([...]),     # Encoder hidden states
        decoder_variables=torch.Tensor([...]),     # Decoder hidden states
        decoder_lengths=torch.Tensor([...]),       # Sequence lengths
        encoder_lengths=torch.Tensor([...])        # Sequence lengths
    ),
    x=Dict[...],  # Input data
    index=pd.DataFrame(...)  # Indices
)
```

### Tensor Shape Breakdown

```python
pred_tensor.prediction.shape = torch.Size([20, 96, 7])

# Dimension 0: 20 servers (ppml0001, ppml0002, ..., ppweb008)
# Dimension 1: 96 timesteps (8-hour prediction horizon at 5-minute intervals)
# Dimension 2: 7 quantiles (p0.02, p0.1, p0.25, p0.5, p0.75, p0.9, p0.98)
```

Our code extracts:
- `pred_values[:, 0, 0]` = p10 (10th percentile, optimistic forecast)
- `pred_values[:, 0, 1]` = p50 (50th percentile, median forecast)
- `pred_values[:, 0, 2]` = p90 (90th percentile, pessimistic forecast)

---

## Lessons Learned

### 1. Namedtuple Gotcha
When using `len()` on a namedtuple, you get the **number of fields**, not the length of data inside those fields.

### 2. Introspection is Key
The fix was found by adding extensive debugging:
```python
print(f"[DEBUG] pred_tensor has .shape: {hasattr(pred_tensor, 'shape')}")
print(f"[DEBUG] len(pred_tensor): {len(pred_tensor)}")
print(f"[DEBUG] pred_tensor.prediction exists: {hasattr(pred_tensor, 'prediction')}")
if hasattr(pred_tensor.prediction, 'shape'):
    print(f"[DEBUG] pred_tensor.prediction.shape: {pred_tensor.prediction.shape}")
```

### 3. Don't Assume Library Behavior
We assumed `model.predict()` would return a simple tensor. In reality, PyTorch Forecasting returns a complex nested structure with attention weights, hidden states, and other internal data.

### 4. Obsessive Attention to Detail Pays Off
The user insisted: "Give a detailed scan of the code, we need to be obsessive in the attention to detail."

This led to discovering that the "8" wasn't in our code at all - it was the number of fields in PyTorch Forecasting's internal data structure.

---

## Impact

### Before Fix
- ❌ Dashboard showed only 8 out of 20 servers
- ❌ Missing predictions for 12 servers (60% of fleet invisible!)
- ❌ Incomplete risk assessment
- ❌ Users saw timeouts and inconsistent data

### After Fix
- ✅ Dashboard shows all 20 servers
- ✅ Complete fleet visibility
- ✅ Accurate risk scores across entire environment
- ✅ No timeouts or missing data

---

## Related Files

- **Fixed:** `tft_inference_daemon.py` (lines 536-559)
- **Session:** `SESSION_2025-10-13_DASHBOARD_POLISH.md`

---

## Prevention

To prevent similar issues in the future:

1. **Always introspect library return types** - Don't assume structure
2. **Check `.shape` on tensors** - Verify dimensions match expectations
3. **Add assertions** - `assert len(predictions) == len(servers), "Prediction count mismatch!"`
4. **Log shapes and lengths** - Make mismatches obvious in logs

---

## Status

✅ **FIXED** - All 20 servers now receive predictions successfully.

**Verified:** October 13, 2025
**Tested:** Inference daemon restart showed `[OK] TFT predictions generated for 20 servers`

---

**Credit:** User's insistence on "obsessive attention to detail" led to finding this subtle but critical bug.
