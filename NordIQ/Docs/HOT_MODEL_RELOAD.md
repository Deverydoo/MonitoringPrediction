# Hot Model Reload Feature

**NordIQ Inference Daemon - Version 2.2**

Reload trained TFT models without restarting the inference daemon. Perfect for continuous training workflows.

---

## Overview

The inference daemon now supports **hot model reloading**, allowing you to:

- Train new models while the system is running
- Load new models without downtime
- Switch between model versions on-the-fly
- Automatic rollback if model loading fails

**Use Case:** You train models continuously with new data. Instead of restarting the daemon (which loses the rolling window state), you can hot-reload the new model seamlessly.

---

## New API Endpoints

### 1. List Available Models

**GET** `/admin/models`

List all trained models in the `models/` directory with metadata.

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/admin/models
```

**Response:**
```json
{
  "models": [
    {
      "path": "models/tft_model_20250130_143000",
      "name": "tft_model_20250130_143000",
      "timestamp": "20250130_143000",
      "is_current": true,
      "size_mb": 245.3,
      "modified": "2025-01-30T14:35:22",
      "training": {
        "epochs": 20,
        "training_time": 7200,
        "servers_trained": 50
      }
    },
    {
      "path": "models/tft_model_20250130_120000",
      "name": "tft_model_20250130_120000",
      "timestamp": "20250130_120000",
      "is_current": false,
      "size_mb": 243.1,
      "modified": "2025-01-30T12:05:15"
    }
  ],
  "count": 2,
  "models_dir": "models",
  "current_model": "models/tft_model_20250130_143000"
}
```

---

### 2. Reload Model (Latest)

**POST** `/admin/reload-model`

Reload the latest model from the `models/` directory.

```bash
curl -X POST -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/admin/reload-model
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Model reloaded successfully",
  "previous_model": "models/tft_model_20250130_120000",
  "new_model": "models/tft_model_20250130_143000",
  "model_timestamp": "20250130_143000"
}
```

**Response (Already Latest):**
```json
{
  "success": true,
  "message": "Model already loaded (same version)",
  "model_path": "models/tft_model_20250130_143000"
}
```

**Response (Failure with Rollback):**
```json
{
  "success": false,
  "error": "Failed to load new model, rolled back to previous",
  "current_model": "models/tft_model_20250130_120000"
}
```

---

### 3. Reload Specific Model

**POST** `/admin/reload-model?model_path=models/tft_model_20250130_120000`

Reload a specific model version.

```bash
curl -X POST -H "X-API-Key: YOUR_API_KEY" \
  "http://localhost:8000/admin/reload-model?model_path=models/tft_model_20250130_120000"
```

---

### 4. Get Current Model Info

**GET** `/admin/model-info`

Get detailed information about the currently loaded model.

```bash
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/admin/model-info
```

**Response:**
```json
{
  "loaded": true,
  "mode": "tft",
  "model_path": "models/tft_model_20250130_143000",
  "model_name": "tft_model_20250130_143000",
  "model_timestamp": "20250130_143000",
  "config": {
    "max_prediction_length": 96,
    "max_encoder_length": 288
  },
  "servers": {
    "total": 50,
    "known": 50
  }
}
```

---

## Workflow: Continuous Training

### Scenario: Train Models Daily

```bash
# 1. Train new model with today's data
python NordIQ/src/training/tft_trainer.py --dataset ./training

# 2. New model is saved to: models/tft_model_20250130_143000

# 3. Hot reload the new model (no daemon restart needed)
curl -X POST -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/admin/reload-model

# 4. Verify new model is loaded
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/admin/model-info
```

**Result:** New model is active, predictions now use latest training. Rolling window state is preserved (no warmup needed).

---

## Safety Features

### 1. Automatic Rollback

If the new model fails to load, the daemon automatically rolls back to the previous model:

```json
{
  "success": false,
  "error": "Failed to load new model, rolled back to previous",
  "current_model": "models/tft_model_20250130_120000"
}
```

Your system stays operational even if the new model is corrupted.

### 2. GPU Memory Management

On GPU systems, old model is cleared from VRAM before loading new model:

```python
if self.model:
    del self.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

Prevents OOM errors when loading large models.

### 3. State Preservation

Rolling window state is **NOT** cleared during reload:
- Warmup status: Preserved
- Server tracking: Preserved
- Historical data: Preserved

Only the model weights change. Predictions continue immediately.

---

## Python Client Example

```python
import requests

DAEMON_URL = "http://localhost:8000"
API_KEY = "your-api-key-here"

headers = {"X-API-Key": API_KEY}

# List available models
response = requests.get(f"{DAEMON_URL}/admin/models", headers=headers)
models = response.json()

print(f"Found {models['count']} models:")
for model in models['models']:
    current = " (CURRENT)" if model['is_current'] else ""
    print(f"  - {model['name']}{current}")
    print(f"    Size: {model['size_mb']} MB")
    print(f"    Modified: {model['modified']}")

# Reload latest model
print("\nReloading latest model...")
response = requests.post(f"{DAEMON_URL}/admin/reload-model", headers=headers)
result = response.json()

if result['success']:
    print(f"✅ Model reloaded: {result['new_model']}")
else:
    print(f"❌ Reload failed: {result['error']}")
```

---

## Integration with Training Pipeline

### Automated Retraining Script

```bash
#!/bin/bash
# retrain_and_reload.sh - Automated model refresh

echo "Starting automated retraining..."

# 1. Train new model
echo "[1/3] Training new model..."
python NordIQ/src/training/tft_trainer.py \
  --dataset ./training \
  --epochs 20 \
  --per-server false

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "❌ Training failed, aborting"
    exit 1
fi

echo "✅ Training complete"

# 2. Hot reload the new model
echo "[2/3] Reloading model in inference daemon..."
RESPONSE=$(curl -s -X POST \
  -H "X-API-Key: ${NORDIQ_API_KEY}" \
  http://localhost:8000/admin/reload-model)

SUCCESS=$(echo $RESPONSE | jq -r '.success')

if [ "$SUCCESS" == "true" ]; then
    NEW_MODEL=$(echo $RESPONSE | jq -r '.new_model')
    echo "✅ Model reloaded: $NEW_MODEL"
else:
    ERROR=$(echo $RESPONSE | jq -r '.error')
    echo "❌ Reload failed: $ERROR"
    exit 1
fi

# 3. Verify model is working
echo "[3/3] Verifying predictions..."
curl -s -H "X-API-Key: ${NORDIQ_API_KEY}" \
  http://localhost:8000/predictions/current | jq '.metadata.model_type'

echo "🎉 Retraining complete and active!"
```

**Schedule with cron:**
```cron
# Retrain daily at 2 AM
0 2 * * * /path/to/retrain_and_reload.sh >> /var/log/nordiq_retrain.log 2>&1
```

---

## Dashboard Integration

### Add Reload Button to Dashboard

```python
import streamlit as st
import requests

if st.button("🔄 Reload Latest Model"):
    with st.spinner("Reloading model..."):
        response = requests.post(
            f"{DAEMON_URL}/admin/reload-model",
            headers={"X-API-Key": API_KEY}
        )
        result = response.json()

        if result['success']:
            st.success(f"✅ Model reloaded: {result['new_model']}")
        else:
            st.error(f"❌ Reload failed: {result['error']}")

# Show current model info
response = requests.get(
    f"{DAEMON_URL}/admin/model-info",
    headers={"X-API-Key": API_KEY}
)
model_info = response.json()

st.info(f"Current Model: {model_info['model_name']}")
st.info(f"Timestamp: {model_info['model_timestamp']}")
```

---

## Performance Impact

**Hot Reload Time:**
- Small model (<100MB): ~2-5 seconds
- Medium model (100-300MB): ~5-10 seconds
- Large model (>300MB): ~10-20 seconds

**During reload:**
- ❌ New predictions: Blocked (returns error during load)
- ✅ Data ingestion: Continues normally
- ✅ Dashboard access: Continues (may show stale predictions)

**After reload:**
- ✅ Predictions use new model immediately
- ✅ No warmup needed (rolling window preserved)
- ✅ XAI components automatically updated

---

## Troubleshooting

### Issue: Reload Fails - "No model found"

**Cause:** Training didn't save model properly.

**Solution:**
```bash
# Check models directory
ls -lh models/

# Verify latest model has required files
ls -lh models/tft_model_*/
# Should see: model.safetensors, config.json, server_mapping.json
```

### Issue: Reload Fails - "Failed to load new model"

**Cause:** Model file is corrupted or incompatible.

**Solution:** Daemon automatically rolls back to previous model. Check logs:
```bash
tail -f logs/inference_daemon.log
```

Retrain the model with correct parameters.

### Issue: Out of Memory (GPU)

**Cause:** GPU doesn't have enough VRAM for two models simultaneously.

**Solution:** Daemon already handles this with `torch.cuda.empty_cache()`. If still failing, reduce model size or use CPU.

---

## API Authentication

All admin endpoints require API key authentication:

```bash
# Set API key
export NORDIQ_API_KEY=$(cat .nordiq_key)

# Use in requests
curl -H "X-API-Key: $NORDIQ_API_KEY" http://localhost:8000/admin/models
```

**Security Note:** Admin endpoints are protected. Only authenticated clients can reload models.

---

## Future Enhancements

Planned for future versions:

1. **Automatic Training Triggers** - Daemon automatically retrains when data buffer reaches threshold
2. **A/B Testing** - Run multiple models simultaneously and compare
3. **Gradual Rollout** - Load new model for subset of servers first
4. **Model Performance Tracking** - Track prediction accuracy per model version

---

## Summary

**Before (v2.1):**
- Train model → Stop daemon → Restart daemon → Lose warmup state → Wait 20+ minutes

**Now (v2.2):**
- Train model → POST /admin/reload-model → New model active in ~5 seconds

**Benefits:**
- ✅ Zero downtime
- ✅ Preserved warmup state
- ✅ Continuous training workflows
- ✅ Automatic rollback on failure
- ✅ Easy dashboard integration

---

**Version:** 2.2.0
**Feature:** Hot Model Reload
**Daemon Version Required:** >=2.2
**API Compatibility:** Backward compatible

Built by Craig Giannelli and Claude Code
