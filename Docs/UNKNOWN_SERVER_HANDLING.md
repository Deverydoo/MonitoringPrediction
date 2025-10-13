# Unknown Server Handling - Production Strategy

**Version:** 2.0 (Contract-Based)
**Last Updated:** 2025-10-11
**Status:** ✅ Fully Implemented with Hash-Based Encoding

## Overview

The TFT Monitoring System supports **unknown servers** natively using:
1. **Hash-based server encoding** - Deterministic, stable IDs for server names
2. **`add_nan=True`** in model training - Handles unknown categories gracefully
3. **Server mapping persistence** - Bidirectional encoding/decoding

**ALL servers use the TFT model** - no heuristic fallbacks needed!

## ✅ Current Implementation (Hash-Based + add_nan)

### How It Works

```
┌──────────────────────────────────────────┐
│   Incoming Server Metrics                │
│   server_name: "ppvra00a0099" (NEW!)     │
└──────────────┬───────────────────────────┘
               │
               ▼
       ┌───────────────────┐
       │  ServerEncoder    │
       │  (SHA256 hash)    │
       └───┬───────────┬───┘
           │           │
   Known   │           │   Unknown
server_name│           │   server_name
           ▼           ▼
   ┌─────────────┐  ┌──────────────────┐
   │ "ppvra00a01"│  │ "ppvra00a0099"   │
   │ → "957601"  │  │ → "123456" (new) │
   └─────┬───────┘  └─────┬────────────┘
         │                │
         │   Both go to TFT Model
         └────────┬───────┘
                  ▼
          ┌───────────────────┐
          │   TFT Model       │
          │  (add_nan=True)   │
          │                   │
          │  - Known: Uses    │
          │    learned pattern│
          │  - Unknown: Uses  │
          │    NaN category   │
          └────────┬──────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  Predictions       │
          │  (server_id)       │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  ServerEncoder     │
          │  .decode()         │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  "ppvra00a0099":   │
          │  {predictions...}  │
          └────────────────────┘
```

### Training Configuration

**File:** `tft_trainer.py`

```python
from server_encoder import ServerEncoder
from data_validator import DataValidator, CONTRACT_VERSION, VALID_STATES

# Step 1: Hash-based server encoding (lines ~286-292)
encoder = ServerEncoder()
encoder.create_mapping(df['server_name'].unique().tolist())
df['server_id'] = df['server_name'].apply(encoder.encode)

# Step 2: Save mapping with model (lines ~732-735)
mapping_path = model_dir / "server_mapping.json"
encoder.save_mapping(mapping_path)

# Step 3: TFT encoder config (lines ~386-389)
categorical_encoders = {
    'server_id': NaNLabelEncoder(add_nan=True),  # Allow unknown servers
    'status': NaNLabelEncoder(add_nan=True)       # Allow unknown statuses
}
```

### What Happens

**Known Servers** (trained on)
- Production: `pprva00a01-10` (10 servers)
- Staging: `psrva00a01-03` (3 servers)
- Compute: `cppr01-05` (5 servers)
- Service: `csrva01-05` (5 servers)
- Container: `crva01-02` (2 servers)
- Model learns **individual behavior patterns** for each

**Unknown Servers** (never seen)
- Any new server (e.g., `pprva00a99`, `cppr50`)
- Routed to special **NaN category** in model
- Uses **aggregate learned patterns** from all training servers
- Still gets full TFT predictions (attention, quantiles, etc.)

## Inference Code Location

**File:** `tft_inference.py`

### Key Components

#### 1. Load Server Mapping (lines ~321-330)
```python
# Load mapping from model directory
mapping_file = self.model_dir / "server_mapping.json"
self.server_encoder = ServerEncoder(mapping_file)
print(f"[OK] Server mapping loaded: {encoder.get_stats()['total_servers']} servers")
```

#### 2. Encode Server Names (lines ~569-578)
```python
# Convert server_name to server_id using hash-based encoder
if 'server_name' in df.columns and self.server_encoder:
    df['server_id'] = df['server_name'].apply(self.server_encoder.encode)
```

#### 3. TFT Prediction (ALL servers, known and unknown)
```python
# Step: Run TFT predictions
predictions = self._predict_with_tft(df, horizon=96)
# TFT handles unknown servers via add_nan=True
# No heuristic fallback needed!
```

#### 4. Decode Server IDs Back to Names (lines ~619-623)
```python
# Decode server_id back to server_name for output
if self.server_encoder:
    server_name = self.server_encoder.decode(server_id)

predictions[server_name] = server_preds  # Human-readable output
```

## Production Benefits

### ✅ Advantages

1. **Deterministic Encoding**: Same server name → same ID every time (SHA256 hash)
2. **Stable Under Fleet Changes**: Adding/removing servers doesn't break existing IDs
3. **Reversible**: Can decode predictions back to original server names
4. **No Training Required**: Unknown servers work immediately via `add_nan=True`
5. **Graceful Degradation**: System never fails on unknown servers
6. **Full TFT for All**: Both known and unknown servers get TFT predictions
7. **Scalability**: Add new servers without retraining
8. **Contract-Compliant**: Follows DATA_CONTRACT.md v1.0.0

### ⚠️ Limitations

1. **Accuracy Gap**: Unknown servers rely on NaN category (aggregate patterns, not personalized)
2. **Hash Collisions**: Rare but possible (< 0.0001% with 1M hash space for typical fleet sizes)
3. **One-Way Migration**: Changing hash algorithm requires retraining all models

## Future Enhancements

### Option 1: Online Learning ⏭️
- Retrain model periodically with new server data
- Automatically update server_mapping.json with new servers
- Incremental learning to avoid full retraining

### Option 2: Transfer Learning 🔮
- Use profile-based patterns (production servers learn from other production servers)
- Bootstrap new servers with similar profiles
- Multi-task learning across server types

### Option 3: Enhanced Hash Algorithm 🔧
- Use more sophisticated encoding (e.g., learned embeddings)
- Contextual encoding based on server metadata
- Semantic similarity for server grouping

## Testing Unknown Server Handling

### Quick Test

```bash
# 1. Train model with current servers (25 servers)
python tft_trainer.py --dataset ./training/ --epochs 1

# 2. Check server_mapping.json was created
cat models/tft_model_*/server_mapping.json

# 3. Start inference daemon
python tft_inference.py --daemon --port 8000

# 4. Feed data with NEW server name
curl -X POST http://localhost:8000/feed/data \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "timestamp": "2025-10-11T08:00:00",
      "server_name": "pprva00a99",  # Unknown server!
      "cpu_percent": 45.2,
      "memory_percent": 62.1,
      "disk_percent": 55.0,
      "load_average": 2.5
    }]
  }'

# 5. Get predictions (should include pprva00a99)
curl http://localhost:8000/predictions/current
```

### Expected Behavior

1. **New server**: `pprva00a99` gets hash-encoded (e.g., → `"456789"`)
2. **TFT model**: Routes to NaN category (unknown server handling)
3. **Predictions**: Return predictions for `pprva00a99` using aggregate patterns
4. **No errors**: System handles gracefully, no fallback to heuristics needed

## Model Files Structure

After training, the model directory contains:

```
models/tft_model_20251011_HHMMSS/
├── model.safetensors              # TFT weights
├── config.json                    # Model architecture
├── training_info.json             # Contract version, states, etc.
└── server_mapping.json            # ⭐ Server name ↔ ID mapping
```

**Critical:** `server_mapping.json` MUST be present for inference to work!

## Conclusion

The current implementation provides **production-ready unknown server support** using:

1. **Hash-Based Encoding** - Stable, deterministic server IDs
2. **add_nan=True** - TFT natively handles unknown categories
3. **Contract Compliance** - DATA_CONTRACT.md enforces consistency
4. **Full Reversibility** - Decode predictions back to server names

### Migration from Old System

If upgrading from pre-contract models:

1. **Retrain Required**: Old models don't have `server_mapping.json`
2. **One-Time Cost**: ~30-40 minutes on RTX 4090 for 20 epochs
3. **Benefits**: Stable encoding, contract validation, production-ready

### When to Retrain

- Adding >10% new servers to fleet
- Major infrastructure changes
- Model performance degradation
- Contract version upgrade

**No need to retrain** for:
- Individual server additions (handled by add_nan)
- Temporary server changes
- Server removals

---

**Version:** 2.0 (Hash-Based)
**Conforms To:** DATA_CONTRACT.md v1.0.0
**Status:** ✅ Production Ready
**Last Updated:** 2025-10-11
