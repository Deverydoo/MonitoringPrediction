# Sparse Data Handling - Offline Servers

**Version:** 1.0.0
**Created:** 2025-10-11
**Contract:** DATA_CONTRACT.md v1.0.0

---

## 🎯 Problem Statement

### Real-World Behavior
In production monitoring systems, when a server goes offline:
- **It stops sending telemetry completely**
- **No rows appear in logs** during offline period
- **Data is sparse** - not a complete grid of servers × timestamps
- Server reappears when it comes back online

### Example Real Logs:
```
timestamp            server_name    cpu_pct  state
2025-10-11 09:00:00  server-01      45.2     healthy
2025-10-11 09:00:00  server-02      32.1     healthy
2025-10-11 09:00:05  server-01      46.1     healthy
2025-10-11 09:00:05  server-02      0.0      offline    ❌ WRONG - server shouldn't appear
2025-10-11 09:00:10  server-01      47.3     healthy
# server-02 is offline - NO ROW appears here              ✅ CORRECT
2025-10-11 09:00:15  server-01      48.1     healthy
2025-10-11 09:00:15  server-02      12.3     recovery   ✅ server comes back
```

---

## 🏗️ Current Implementation vs. Real World

### Current (Dense Grid):
```python
# metrics_generator.py creates ALL combinations
for timestamp in schedule:
    for server in fleet:
        # Creates row even if server is offline
        row = {
            'timestamp': timestamp,
            'server_name': server,
            'state': state,
            'cpu_pct': 0.0 if state == 'offline' else value  # ❌ Row exists with zeros
        }
```

**Result:** Complete grid - every server has entry for every timestamp
- Offline servers: rows exist with `state='offline'` and metrics = 0
- Total rows: `num_timestamps × num_servers` (always)

### Real World (Sparse):
```python
# Reality: Only online servers send data
for timestamp in schedule:
    for server in fleet:
        if server_is_online(server, timestamp):  # ✅ Check if online
            row = {
                'timestamp': timestamp,
                'server_name': server,
                'state': get_state(server),
                'cpu_pct': get_cpu(server)
            }
            # No row created if offline
```

**Result:** Sparse data - servers only appear when online
- Offline servers: NO rows (gaps in time series)
- Total rows: `< num_timestamps × num_servers`

---

## 🤔 Why This Matters for TFT Training

### TimeSeriesDataSet Requirements

The TFT model in `pytorch_forecasting` has built-in support for sparse data:

```python
# From tft_trainer.py line 430
TimeSeriesDataSet(
    df,
    time_idx='time_idx',           # Sequential index
    group_ids=['server_id'],        # Group by server
    allow_missing_timesteps=True,   # ✅ KEY: Handles gaps in time series
    ...
)
```

**With `allow_missing_timesteps=True`:**
- ✅ Model handles servers disappearing from data
- ✅ Can train on sparse data (realistic)
- ✅ Learns that offline state = no data
- ✅ Production-ready behavior

**Current dense data with zeros:**
- ❌ Model learns: offline = all zeros (artificial pattern)
- ❌ Not realistic for production deployment
- ❌ Model expects complete grid (unrealistic)

---

## 🔧 Solution: Two Modes

### Mode 1: Dense (Current - Training Stability)
**Use Case:** Initial training, debugging, consistent baseline
**Implementation:** Keep all rows, use `offline_fill="zeros"`

```python
Config(offline_fill="zeros")  # All servers × all timestamps
```

**Pros:**
- ✅ Stable training (no missing data)
- ✅ Easy to debug
- ✅ Consistent row count
- ✅ Model sees explicit offline state

**Cons:**
- ❌ Unrealistic (servers don't send zeros when offline)
- ❌ Model learns artificial patterns
- ❌ Larger dataset size

### Mode 2: Sparse (Production-Ready)
**Use Case:** Production training, realistic scenarios
**Implementation:** Drop offline rows entirely

```python
Config(offline_mode="sparse")  # Only online servers
```

**Pros:**
- ✅ Realistic production behavior
- ✅ Smaller dataset (faster training)
- ✅ Model learns true patterns
- ✅ Better generalization

**Cons:**
- ⚠️ Requires `allow_missing_timesteps=True` (already enabled)
- ⚠️ Variable row counts per timestamp
- ⚠️ Need to handle gaps in inference

---

## 🚀 Implementation Plan

### Step 1: Add Sparse Mode to Config

```python
# metrics_generator.py
@dataclass
class Config:
    # ... existing fields ...
    offline_mode: str = "dense"  # "dense" or "sparse"

    def __post_init__(self):
        if self.offline_mode not in ["dense", "sparse"]:
            raise ValueError("offline_mode must be 'dense' or 'sparse'")
```

### Step 2: Modify Data Generation

```python
def simulate_states(fleet: pd.DataFrame, schedule: pd.DatetimeIndex, config: Config) -> pd.DataFrame:
    """Simulate server states using Markov chain."""
    results = []
    server_states = {name: ServerState.HEALTHY for name in fleet['server_name']}

    for timestamp in schedule:
        hour = timestamp.hour

        for _, server in fleet.iterrows():
            server_name = server['server_name']
            # ... state transition logic ...
            next_state = # ... sample state ...
            server_states[server_name] = next_state

            # NEW: Sparse mode - skip offline servers
            if config.offline_mode == "sparse" and next_state == ServerState.OFFLINE:
                continue  # ✅ Don't create row for offline server

            results.append({
                'timestamp': timestamp,
                'server_name': server_name,
                'profile': profile,
                'state': next_state.value,
                'problem_child': is_problem_child
            })

    return pd.DataFrame(results)
```

### Step 3: Update Metrics Generation

```python
def simulate_metrics(state_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add realistic metrics based on states and profiles."""
    # ... existing logic ...

    # In sparse mode, offline servers already removed from state_df
    # No need for special handling

    if config.offline_mode == "dense":
        # Handle offline state (current behavior)
        offline_mask = server_data['state'] == 'offline'
        if config.offline_fill == "zeros":
            smoothed_values[offline_mask] = 0.0
        # ...

    return df
```

---

## 📊 Expected Results

### Dense Mode (Current):
```python
# 24 hours, 25 servers, 5-second intervals
timestamps = 24 * 60 * 12 = 17,280
servers = 25
total_rows = 17,280 × 25 = 432,000 rows (always)
```

### Sparse Mode (Realistic):
```python
# Same setup, but ~5% offline at any time
average_online = 25 × 0.95 = ~23.75 servers
total_rows ≈ 17,280 × 23.75 = ~410,400 rows
reduction = ~5% fewer rows (faster training, smaller files)
```

---

## 🧪 Testing Strategy

### Test 1: Data Generation
```bash
# Dense mode (current)
python metrics_generator.py --hours 1 --offline_mode dense --out_dir ./test/dense/

# Sparse mode (new)
python metrics_generator.py --hours 1 --offline_mode sparse --out_dir ./test/sparse/

# Compare row counts
```

**Expected:**
- Dense: Exactly `(hours × 720) × num_servers` rows
- Sparse: ~5-10% fewer rows (depends on offline rate)

### Test 2: Model Training
```bash
# Train on sparse data
python tft_trainer.py --training-dir ./test/sparse/ --epochs 5

# Should work due to allow_missing_timesteps=True
```

**Expected:**
- ✅ Training succeeds
- ✅ Model handles missing timesteps
- ✅ Validation loss comparable to dense mode

### Test 3: Inference with Gaps
```python
# Create inference data with server offline periods
df = pd.DataFrame({
    'timestamp': [...],
    'server_name': ['server-01', 'server-01', 'server-03'],  # server-02 missing
    'cpu_pct': [45.2, 46.1, 32.1],
    # ...
})

# Model should predict despite gaps
predictions = model.predict(df)
```

**Expected:**
- ✅ Predictions for present servers
- ✅ Handles missing server-02 gracefully

---

## 🎯 Recommendations

### For Current Development:
**Use Dense Mode** - Proven stable, easier to debug

```python
Config(
    offline_mode="dense",
    offline_fill="zeros"  # Changed from "nan"
)
```

### For Production Deployment:
**Switch to Sparse Mode** - Realistic behavior

```python
Config(
    offline_mode="sparse"
    # offline_fill not needed in sparse mode
)
```

### Migration Path:
1. **Phase 1** (Current): Dense mode with zeros ✅ **DONE**
2. **Phase 2** (Next): Implement sparse mode (this document)
3. **Phase 3**: Train models on both modes, compare performance
4. **Phase 4**: Production deployment with sparse mode

---

## 🔍 Edge Cases

### Case 1: Server Offline Entire Training Period
**Sparse Mode:** No rows for that server
**TFT Behavior:** Server not in training data, treated as unknown in inference

**Solution:** Filter servers with < 80% uptime from training

### Case 2: Server Offline During Inference
**Sparse Mode:** No recent data for predictions
**TFT Behavior:** Can't predict (no context window)

**Solution:** Use last-seen state or mark as "data stale"

### Case 3: All Servers Offline at Same Timestamp
**Sparse Mode:** Zero rows for that timestamp
**TFT Behavior:** Timestamp not in dataset

**Solution:** Highly unlikely (<0.001% probability), acceptable gap

---

## 📚 Related Documentation

- [DATA_CONTRACT.md](DATA_CONTRACT.md) - Schema requirements
- [UNKNOWN_SERVER_HANDLING.md](UNKNOWN_SERVER_HANDLING.md) - Hash-based encoding
- [TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md) - Model capabilities

---

**Status:** 📋 Design Complete - Ready for Implementation
**Priority:** Medium (Phase 2 enhancement)
**Estimated Effort:** 2-3 hours
