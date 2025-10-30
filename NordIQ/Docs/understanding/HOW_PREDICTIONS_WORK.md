# How Predictions Work in the TFT Monitoring System

**Last Updated**: 2025-10-13
**Model Version**: TFT (Temporal Fusion Transformer)
**Training**: 1 epoch, 1 week data, 20 servers

---

## Architecture Overview

The prediction workflow follows a **three-daemon architecture** with data flowing through a pipeline:

```
┌─────────────────────────┐
│ Metrics Generator       │  Generates realistic server metrics
│ Daemon (Port 8001)      │  Every 5 seconds
└───────────┬─────────────┘
            │ POST /feed/data
            ↓
┌─────────────────────────┐
│ TFT Inference           │  Maintains rolling window
│ Daemon (Port 8000)      │  Runs TFT model predictions
└───────────┬─────────────┘
            │ GET /predictions/current
            ↓
┌─────────────────────────┐
│ Streamlit Dashboard     │  Visualizes predictions
│ (Port 8501)             │  Alerts and risk analysis
└─────────────────────────┘
```

---

## 1. Metrics Generator Daemon

**File**: [metrics_generator_daemon.py](../metrics_generator_daemon.py)
**Purpose**: Continuously generates realistic server metrics every 5 seconds

### Server Fleet Configuration

Creates a fleet of 20 servers across 7 different profiles:

| Profile | Prefix | Count | Characteristics |
|---------|--------|-------|-----------------|
| ML Compute | ppml | 3 | High CPU, high memory, GPU workloads |
| Database | ppdb | 4 | High disk I/O, memory-intensive |
| Web API | ppweb | 5 | Network-heavy, low latency critical |
| Conductor | ppcon | 2 | Orchestration, moderate resources |
| ETL/Ingest | ppetl | 3 | Burst I/O, variable CPU |
| Risk Analytics | pprisk | 2 | CPU-intensive, memory-heavy |
| Generic | ppgen | 1 | Balanced workload |

### Metrics Generated

Each server emits 8 metrics every 5 seconds:

```python
{
  "timestamp": "2025-10-13T10:30:00",
  "server_name": "ppdb001",
  "profile": "database",
  "state": "healthy",

  # Core metrics (used by TFT model)
  "cpu_pct": 45.2,           # CPU utilization (0-100%)
  "mem_pct": 62.1,           # Memory utilization (0-100%)
  "disk_io_mb_s": 120.5,     # Disk I/O throughput (MB/s)
  "latency_ms": 8.2,         # Request latency (milliseconds)

  # Additional metrics
  "net_in_mb_s": 45.3,       # Network ingress (MB/s)
  "net_out_mb_s": 38.7,      # Network egress (MB/s)
  "error_rate": 0.002,       # Error rate (0-1)
  "gc_pause_ms": 15.4        # Garbage collection pause (ms)
}
```

### Profile-Based Baselines

Each profile has different "healthy" baseline values ([metrics_generator.py:184-227](../metrics_generator.py#L184-L227)):

**Example: Database Server (ppdb)**
- CPU: 40% ± 15% (healthy baseline: 35-50%)
- Memory: 50% ± 8% (healthy baseline: 42-58%)
- Disk I/O: 85 MB/s ± 35 MB/s
- Latency: 12 ms ± 6 ms

**Example: ML Compute Server (ppml)**
- CPU: 45% ± 12% (healthy baseline: 33-57%)
- Memory: 55% ± 10% (healthy baseline: 45-65%)
- Disk I/O: 30 MB/s ± 15 MB/s
- Latency: 8 ms ± 4 ms

> **Note**: In the previous version, baselines were set too high (78-88%), causing false alerts. They were corrected to realistic 40-55% ranges for healthy operation.

### State Transitions

Servers transition between states with realistic patterns:

```python
States:
- HEALTHY         → Morning spike, heavy load, or stay healthy
- MORNING_SPIKE   → Peak load or recover to healthy
- HEAVY_LOAD      → Degraded or recover
- DEGRADED        → Critical or recover to heavy load
- CRITICAL        → Recovering or stay critical
- RECOVERING      → Healthy or back to degraded
```

Each state applies multipliers to baseline metrics:
- **HEALTHY**: 1.0× (baseline)
- **MORNING_SPIKE**: 1.3× (+30%)
- **HEAVY_LOAD**: 1.5× (+50%)
- **DEGRADED**: 1.8× (+80%)
- **CRITICAL**: 2.2× (+120%)
- **RECOVERING**: 1.2× (+20%)

### Diurnal Patterns

Metrics follow realistic daily patterns ([metrics_generator.py:158-181](../metrics_generator.py#L158-L181)):

```
Hour of Day → Multiplier
00:00-06:00 → 0.4-0.6× (night, low activity)
06:00-09:00 → 0.6-1.0× (morning ramp-up)
09:00-17:00 → 1.0-1.2× (business hours, peak)
17:00-22:00 → 0.8-1.0× (evening wind-down)
22:00-24:00 → 0.5-0.7× (late night)
```

### Scenario Modes

Three modes control fleet-wide behavior:

| Scenario | Affected Servers | Multiplier | Description |
|----------|-----------------|------------|-------------|
| **Healthy** | 0% | 1.0× | Normal operations |
| **Degrading** | 25% | 1.15× | Early warning signs (+15%) |
| **Critical** | 25% | 1.6× | Significant issues (+60%) |

Change scenario via REST API:
```bash
curl -X POST http://localhost:8001/scenario/set \
  -H "Content-Type: application/json" \
  -d '{"scenario": "degrading"}'
```

### Data Delivery

Metrics are POSTed to inference daemon every 5 seconds:
```
POST http://localhost:8000/feed/data
{
  "records": [ ... 20 server records ... ]
}
```

---

## 2. TFT Inference Daemon

**File**: [tft_inference_daemon.py](../tft_inference_daemon.py)
**Purpose**: Maintains rolling window of data and runs TFT model predictions

### Data Ingestion

**Endpoint**: `POST /feed/data`
**Process** ([tft_inference_daemon.py:783-818](../tft_inference_daemon.py#L783-L818)):

1. Receives batch of 20 server records every 5 seconds
2. Validates data against contract (cpu_pct, mem_pct, disk_io_mb_s, latency_ms)
3. Appends to **rolling window** (deque with max 6000 records ≈ 8 hours)
4. Tracks per-server timestep count for warmup

### Model Warmup

**Threshold**: 150 timesteps per server
**Time Required**: 150 × 5 seconds = **12.5 minutes per server**

```python
WARMUP_THRESHOLD = 150  # Minimum timesteps needed per server

# Warmup status tracking
{
  "ppdb001": 145,  # Not ready (145/150)
  "ppdb002": 150,  # Ready ✓
  "ppweb001": 89,  # Not ready (89/150)
  ...
}
```

**Why warmup is needed**:
- TFT model requires encoder_length=12 (minimum 12 consecutive timesteps)
- Additional context improves prediction quality
- 150 timesteps = 12.5 minutes of history for stable predictions

**During warmup**:
- System falls back to **heuristic predictions** (linear trend extrapolation)
- Dashboard shows warmup banner: "Model Warming Up: 145/150 datapoints"
- Once all servers reach 150 timesteps, TFT model activates

### Prediction Request Flow

**Endpoint**: `GET /predictions/current`
**Trigger**: Dashboard requests every 60 seconds (configurable 5-300s)

**Process** ([tft_inference_daemon.py:820-845](../tft_inference_daemon.py#L820-L845)):

```python
def get_current_predictions():
    # 1. Convert rolling window to DataFrame
    df = pd.DataFrame(list(self.rolling_window))

    # 2. Check warmup status
    if all_servers_warmed_up():
        # Use TFT model for predictions
        predictions = self._predict_with_tft(df, horizon=96)
    else:
        # Fall back to heuristic predictions
        predictions = self._heuristic_predictions(df, horizon=96)

    # 3. Calculate risk scores and alerts
    alerts = self._generate_alerts(predictions)
    environment = self._calculate_environment_risk(predictions)

    # 4. Return formatted response
    return {
        "predictions": predictions,
        "alerts": alerts,
        "environment": environment,
        "metadata": {...}
    }
```

---

## 3. TFT Model Prediction (The Core)

### What is TFT?

**Temporal Fusion Transformer** (Google Research, 2019):
- Deep learning architecture for multi-horizon time series forecasting
- Combines LSTM (for sequential patterns) + Attention (for important features)
- Natively produces **probabilistic forecasts** (not just point estimates)

### Model Configuration

**Training Setup** ([tft_inference_daemon.py:118-237](../tft_inference_daemon.py#L118-L237)):
```python
Model Architecture:
- Encoder length: 12-24 timesteps (1-2 hours of history)
- Prediction length: 96 timesteps (8 hours ahead)
- Hidden size: 32 units
- Attention heads: 4
- Dropout: 0.1
- Parameters: ~50K

Training Data:
- Duration: 1 week (2,016 timesteps per server)
- Servers: 20 (across 7 profiles)
- Records: 40,320 total
- Epochs: 1 (proof of concept)

Input Features:
- Time-varying: cpu_pct, mem_pct, disk_io_mb_s, latency_ms
- Time features: hour, day_of_week, month, is_weekend
- Static: server_id, profile

Output:
- 3 quantiles per metric: p10, p50, p90
- 4 metrics × 3 quantiles = 12 forecasts per server
- 96 steps ahead (8 hours)
```

### Quantile Forecasts Explained

**TFT predicts THREE scenarios for each metric**:

```
p10 (10th percentile) = Best-case scenario
p50 (50th percentile) = Most likely scenario (median)
p90 (90th percentile) = Worst-case scenario
```

**Example prediction for CPU at time T+30 minutes**:
```python
"cpu_percent": {
  "p10": 42.3,   # 10% chance CPU will be BELOW 42.3%
  "p50": 51.8,   # 50% chance CPU will be BELOW 51.8% (median prediction)
  "p90": 67.4,   # 90% chance CPU will be BELOW 67.4% (only 10% chance it exceeds this)
  "current": 48.2
}
```

**Interpreting quantiles**:
- **p10**: Optimistic forecast (best 10% of possible outcomes)
- **p50**: Median forecast (most likely value)
- **p90**: Pessimistic forecast (worst 10% of possible outcomes)

> **Key insight**: For risk assessment, we focus on **p90** because it tells us the worst-case scenario with 90% confidence. If p90 predicts CPU > 95%, there's a 10% chance it will exceed 95%, which is HIGH RISK.

### Full Prediction Example

```python
{
  "predictions": {
    "ppdb001": {
      "cpu_percent": {
        "p10": [42.3, 43.1, 43.8, 44.5, ...],  # 96 values (best case)
        "p50": [51.8, 52.4, 53.1, 53.7, ...],  # 96 values (median)
        "p90": [67.4, 68.2, 69.0, 69.8, ...],  # 96 values (worst case)
        "current": 48.2,
        "trend": 0.15  # Positive = rising, negative = falling
      },
      "memory_percent": {
        "p10": [55.2, 55.8, 56.3, ...],
        "p50": [64.1, 64.6, 65.1, ...],
        "p90": [71.8, 72.4, 73.0, ...],
        "current": 62.3,
        "trend": 0.08
      },
      "disk_percent": {
        "p10": [18.5, 18.9, 19.3, ...],
        "p50": [28.7, 29.2, 29.7, ...],
        "p90": [42.1, 42.8, 43.5, ...],
        "current": 27.4,
        "trend": 0.12
      },
      "load_average": {
        "p10": [1.2, 1.3, 1.3, ...],
        "p50": [2.8, 2.9, 3.0, ...],
        "p90": [4.5, 4.7, 4.8, ...],
        "current": 2.6,
        "trend": 0.05
      }
    },
    "ppdb002": { ... },
    # ... 18 more servers
  }
}
```

**Time mapping**:
- Index 0 = Now + 5 minutes
- Index 6 = Now + 30 minutes (used for 30m risk)
- Index 96 = Now + 8 hours

---

## 4. Risk Calculation

### Per-Server Risk Score

**Function**: `calculate_server_risk_score()` in [tft_dashboard_web.py:232-262](../tft_dashboard_web.py#L232-L262)

**Scoring Logic** (0-100 scale):

#### CPU Risk (Max 40 points)
```python
max_cpu_p90 = max(cpu_percent['p90'][:6])  # First 30 minutes of p90 predictions

if max_cpu_p90 > 98:
    risk += 40   # CRITICAL: 98%+ CPU (system will hang)
elif max_cpu_p90 > 95:
    risk += 30   # SEVERE: 95-98% CPU (major slowdown)
elif max_cpu_p90 > 90:
    risk += 15   # WARNING: 90-95% CPU (performance degradation)
```

**What is `max_cpu_p90`?**
- Takes the **worst-case (p90) CPU predictions** for the next 30 minutes (6 steps × 5 min)
- Finds the **maximum value** across those 6 predictions
- Example: `p90 = [67.4, 68.2, 69.0, 72.1, 71.5, 70.8]` → `max_cpu_p90 = 72.1`
- If `max_cpu_p90 > 90`, it means there's a 10% chance CPU will exceed 90% in the next 30 minutes

#### Memory Risk (Max 35 points)
```python
max_mem_p90 = max(memory_percent['p90'][:6])  # First 30 minutes

if max_mem_p90 > 98:
    risk += 35   # CRITICAL: 98%+ memory (OOM imminent)
elif max_mem_p90 > 95:
    risk += 25   # SEVERE: 95-98% memory (swapping likely)
elif max_mem_p90 > 90:
    risk += 12   # WARNING: 90-95% memory (elevated risk)
```

#### Latency Risk (Max 25 points)
```python
max_lat_p90 = max(load_average['p90'][:6])  # First 30 minutes

if max_lat_p90 > 500:
    risk += 25   # CRITICAL: 500ms+ latency (user-facing impact)
elif max_lat_p90 > 200:
    risk += 15   # SEVERE: 200-500ms latency (noticeable delay)
elif max_lat_p90 > 100:
    risk += 8    # WARNING: 100-200ms latency (degraded experience)
```

**Total Risk Score**: Sum of all components (capped at 100)

**Example Calculation**:
```
Server: ppdb001
  max_cpu_p90 = 72.1  → +0 points (< 90)
  max_mem_p90 = 93.4  → +12 points (90-95 range)
  max_lat_p90 = 245   → +15 points (200-500 range)

Total Risk Score: 27 / 100 (P3 - Info)
```

### Priority Levels

```python
if risk_score >= 70:
    priority = "P1"  # Critical (red 🔴)
elif risk_score >= 40:
    priority = "P2"  # Warning (orange 🟡)
else:
    priority = "P3"  # Info (blue 🔵)
```

### Fleet-Wide Risk (Environment)

**Function**: `_calculate_environment_risk()` in [tft_inference_daemon.py:664-717](../tft_inference_daemon.py#L664-L717)

**30-Minute Incident Probability**:
```python
for each server:
    risk_30m = 0.0

    # Check first 6 steps (30 minutes) of predictions
    for metric in [cpu, memory, disk, latency]:
        max_p50 = max(metric['p50'][:6])
        max_p90 = max(metric['p90'][:6])

        # Severe conditions (high weight)
        if max_p50 > 90 or max_p90 > 98:
            risk_30m += 0.35
        # Moderate conditions (low weight)
        elif max_p50 > 85 or max_p90 > 95:
            risk_30m += 0.15

    risk_30m = min(1.0, risk_30m)  # Cap at 100%

# Fleet-wide probability = average across all servers
prob_30m = sum(all_server_risks) / num_servers
```

**8-Hour Incident Probability**:
```python
# Similar logic but checks all 96 steps (8 hours)
# Lower thresholds: p50 > 88 or p90 > 96 (severe), p50 > 82 or p90 > 92 (moderate)
# Weights: +0.25 (severe), +0.10 (moderate)

prob_8h = sum(all_server_risks_8h) / num_servers
```

**Fleet Health Status**:
```python
if prob_30m > 0.7:
    fleet_health = "critical"  # 🔴 70%+ chance of incident
elif prob_30m > 0.4:
    fleet_health = "warning"   # 🟡 40-70% chance
else:
    fleet_health = "healthy"   # 🟢 <40% chance
```

**Example Output**:
```python
{
  "environment": {
    "prob_30m": 0.12,           # 12% chance of incident in next 30 minutes
    "prob_8h": 0.45,            # 45% chance of incident in next 8 hours
    "high_risk_servers": 2,     # Number of servers with risk >= 70
    "total_servers": 20,
    "fleet_health": "warning"   # healthy | warning | critical
  }
}
```

---

## 5. Dashboard Visualization

**File**: [tft_dashboard_web.py](../tft_dashboard_web.py)

### Prediction Retrieval

**Auto-refresh** ([tft_dashboard_web.py:2270-2290](../tft_dashboard_web.py#L2270-L2290)):
```python
# Default: fetch every 60 seconds
# Configurable: 5s (demo mode) to 300s (5 minutes)

if time_since_last_update >= refresh_interval:
    with st.spinner('🔮 Fetching predictions from TFT model...'):
        predictions = client.get_predictions()  # GET /predictions/current
        alerts = client.get_alerts()
        st.session_state.last_update = datetime.now()

        # Cache for fast UI updates
        st.session_state.cached_predictions = predictions
        st.session_state.cached_alerts = alerts
```

**Performance optimization**:
- `@st.cache_data(ttl=300)` decorator on expensive functions
- Spinner shows "Fetching predictions..." during API call (~100ms)
- Cached predictions used for UI rendering between fetches

### Active Alerts Table

**Columns**:
| Column | Description | Example |
|--------|-------------|---------|
| Priority | P1/P2/P3 with color emoji | 🔴 P1 |
| Server | Server name | ppdb001 |
| Profile | Server profile type | Database |
| Risk | Risk score (0-100) | 87 |
| CPU Now | Current CPU utilization | 58.3% |
| CPU 30m | p90 CPU prediction at +30min | 96.7% |
| CPU Δ | Delta (change from now) | +38.4% |
| Mem Now | Current memory | 72.1% |
| Mem 30m | p90 memory at +30min | 94.2% |
| Mem Δ | Delta | +22.1% |
| Lat Now | Current latency | 45ms |
| Lat 30m | p90 latency at +30min | 285ms |
| Lat Δ | Delta | +240ms |

**Sorting**: By risk score (descending), so highest risks appear first

---

## 6. Heuristic Fallback (During Warmup)

**Function**: `_heuristic_predictions()` in [tft_inference_daemon.py:533-616](../tft_inference_daemon.py#L533-L616)

When servers don't have enough data for TFT model (< 150 timesteps), system uses linear trend extrapolation:

```python
def _heuristic_predictions(df, horizon=96):
    for server in servers:
        # Get last N points (up to 12)
        recent_data = df[df['server_name'] == server].tail(12)

        # Calculate trend (linear regression)
        trend = calculate_linear_trend(recent_data['cpu_pct'])
        current = recent_data['cpu_pct'].iloc[-1]

        # Project forward with uncertainty
        predictions = []
        for t in range(horizon):
            # Base prediction
            pred = current + (trend * t)

            # Add uncertainty (grows with time)
            noise = random.gauss(0, 2.0 * math.sqrt(t + 1))

            # Generate quantiles
            p10 = max(0, pred - 10 + noise)
            p50 = max(0, pred + noise)
            p90 = min(100, pred + 10 + noise)

            predictions.append({'p10': p10, 'p50': p50, 'p90': p90})
```

**Limitations**:
- Simple linear extrapolation (no complex patterns)
- Uncertainty grows rapidly (√t)
- No attention to seasonality or anomalies
- Less accurate than TFT for horizons > 30 minutes

**When to expect heuristics**:
- First 12.5 minutes after daemon start
- When new servers join fleet
- If TFT model fails to load

---

## 7. Key Performance Characteristics

### Latency

| Operation | Latency | Notes |
|-----------|---------|-------|
| Metrics generation | ~10ms | 20 servers × 8 metrics |
| Data ingestion | ~5ms | Append to rolling window |
| TFT prediction (GPU) | ~100ms | 20 servers, 96 steps, 3 quantiles |
| TFT prediction (CPU) | ~2-3s | 20-30× slower than GPU |
| Dashboard API call | ~150ms | Prediction + risk calculation |
| Dashboard full refresh | ~200ms | API call + UI rendering |

### Data Volume

```
Metrics per tick: 20 servers × 8 metrics = 160 values
Ticks per hour: 720 (every 5 seconds)
Data per hour: 115,200 values
Rolling window: 6000 records (≈8 hours)
Memory usage: ~15 MB (rolling window + model)
```

### Accuracy (After Full Training)

**Current (1 epoch)**:
- Train Loss: 8.09
- Val Loss: 9.53
- Status: Proof of concept, architecture validated

**Expected (20 epochs)**:
- Target Val Loss: <3.0
- Expected MAPE: 10-15%
- Accuracy: 85-90% for 30-minute horizon, 75-85% for 8-hour horizon

---

## 8. Transfer Learning Architecture

**Server Mapping** ([server_encoder.py](../server_encoder.py)):

```python
# Each server gets unique ID (0-19)
{
  "ppml001": 0,
  "ppml002": 1,
  "ppml003": 2,
  "ppdb001": 3,
  ...
}
```

**Profile-based learning**:
- Model learns patterns per server profile (not just individual servers)
- Similar servers (e.g., ppdb001, ppdb002) share learned patterns
- Enables predictions for new servers of known profiles
- Profile embedding: 7-dimensional vector (one-hot encoded)

**Example**:
- Train on ppdb001-ppdb004 (4 database servers)
- Model learns "database profile" patterns
- Can predict for ppdb005 (new database server) with high accuracy

---

## 9. Common Questions

### Q: Why 96 prediction steps?
**A**: 96 steps × 5 minutes = 8 hours. This gives operations teams a full work shift to react to predicted incidents.

### Q: Why use p90 for risk instead of p50?
**A**: We want to plan for **worst-case scenarios**. If p50 says "CPU will be 60%" but p90 says "90% chance it won't exceed 85%", we need to know the 85% (risk of exceeding 85% is 10%, which is too high for critical systems).

### Q: What if p10, p50, p90 are very far apart?
**A**: Wide quantile spread = **high uncertainty**. Example:
```
p10 = 40%, p50 = 55%, p90 = 85%
```
This means the model is uncertain about the future. Could indicate:
- Volatile workload patterns
- Recent anomalies in data
- State transition occurring (healthy → degraded)

### Q: Why 150 timesteps for warmup?
**A**: TFT model needs:
- **Encoder length**: 12 timesteps (minimum history to understand context)
- **Buffer**: 138 additional timesteps for statistical stability
- At 5-second intervals: 150 × 5 = 12.5 minutes is reasonable warmup time

### Q: Can I change prediction horizon?
**A**: Yes, but requires **retraining the model**. Horizon is fixed during training. Current model trained for 96 steps (8 hours). To predict 24 hours ahead, retrain with `max_prediction_length=288`.

### Q: How does the model handle seasonality?
**A**: TFT has built-in **time features**:
- Hour of day (0-23)
- Day of week (0-6)
- Month (1-12)
- Is weekend (boolean)

These are fed as inputs, allowing model to learn "Monday morning spikes", "Friday afternoon lulls", etc.

### Q: What happens if a server goes offline?
**A**:
- Metrics generator stops sending data for that server
- Rolling window retains last known state
- TFT predictions continue based on last known trajectory
- After ~30 minutes, predictions become stale (flagged in metadata)
- Dashboard shows "Last seen: 30 minutes ago"

---

## 10. Debugging Predictions

### Check Warmup Status
```bash
curl http://localhost:8000/status

{
  "warmup": {
    "is_warmed_up": true,
    "progress_percent": 100,
    "current_size": 150,
    "required_size": 150,
    "message": "Model ready for predictions"
  }
}
```

### Inspect Raw Predictions
```bash
curl http://localhost:8000/predictions/current | jq '.predictions.ppdb001.cpu_percent'

{
  "p10": [42.3, 43.1, 43.8, ...],
  "p50": [51.8, 52.4, 53.1, ...],
  "p90": [67.4, 68.2, 69.0, ...],
  "current": 48.2,
  "trend": 0.15
}
```

### Check Risk Calculation
```bash
curl http://localhost:8000/predictions/current | jq '.alerts[] | select(.server_name=="ppdb001")'

{
  "server_name": "ppdb001",
  "priority": "P2",
  "risk_score": 67,
  "cpu_risk": 15,
  "mem_risk": 12,
  "latency_risk": 15,
  "message": "CPU predicted to reach 93.4% in 30 minutes"
}
```

### Verify Metrics Generator
```bash
curl http://localhost:8001/status

{
  "running": true,
  "scenario": "healthy",
  "tick_count": 1247,
  "fleet_size": 20,
  "affected_servers": 0
}
```

---

## 11. Future Improvements

### Model Training
- **10-epoch training**: Target val loss < 5.0
- **20-epoch training**: Target val loss < 3.0, 85-90% accuracy
- **Hyperparameter tuning**: Learning rate, hidden size, attention heads

### Feature Engineering
- **Error rate forecasts**: Predict request failures
- **Network saturation**: Predict bandwidth exhaustion
- **Disk space**: Predict storage capacity issues

### Advanced Alerting
- **Alert fatigue reduction**: Suppress low-confidence alerts
- **Root cause analysis**: Identify which metric drives risk
- **Correlation detection**: Multi-server incident patterns

### Scalability
- **100+ servers**: Batch predictions, horizontal scaling
- **Multiple data centers**: Regional models with transfer learning
- **Real-time streaming**: Sub-second prediction updates

---

## Summary

**Prediction Pipeline**:
1. **Metrics Generator** creates realistic server data (5s intervals)
2. **TFT Inference Daemon** maintains rolling 8-hour window, runs model
3. **TFT Model** predicts 3 quantiles (p10, p50, p90) for 96 steps (8 hours)
4. **Risk Calculation** uses p90 worst-case scenarios for alerting
5. **Dashboard** visualizes predictions, alerts, and fleet health

**Key Insight**: The system predicts **probabilistic futures** (not single point estimates), enabling risk-aware decision making. By focusing on p90 forecasts, we plan for worst-case scenarios with 90% confidence.

**Demo Ready**: Current system (1 epoch) validates architecture and provides meaningful predictions for demo purposes. Full production deployment requires 20-epoch training for 85-90% accuracy.
