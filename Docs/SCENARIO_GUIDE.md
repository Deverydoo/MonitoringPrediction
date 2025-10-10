# Demo Data Scenarios Guide

## Overview

The demo data generator now supports three configurable health scenarios to simulate different system behaviors. This allows you to test your monitoring system, dashboards, and models against various real-world conditions.

> **⚠️ Important**: This project uses the **`py310`** Python environment. Always activate it before running any scripts:
> ```bash
> conda activate py310
> ```
> See [PYTHON_ENV.md](PYTHON_ENV.md) for details.

---

## 🎯 Scenarios

### 1. **HEALTHY** - Stable System
- **Description**: 100% healthy system with stable baselines and no incidents
- **Use Case**: Testing baseline behavior, dashboard layout, healthy system patterns
- **Behavior**:
  - All metrics stay within normal ranges
  - No warnings or critical alerts
  - Minimal natural variation
  - No affected servers
  - All servers remain in 'online' state

**Example Metrics**:
```
CPU: 20-40% (stable)
Memory: 40-65% (stable)
Latency: 5-50ms (stable)
Error Rate: ~0.001 (minimal)
State: online
```

---

### 2. **DEGRADING** - Gradual Resource Exhaustion (Default)
- **Description**: System starts healthy and gradually degrades over time
- **Use Case**: Training predictive models, testing early warning systems
- **Behavior**:
  - **Phase 1 (0:00-1:30)**: Stable baseline (healthy)
  - **Phase 2 (1:30-2:30)**: Gradual escalation (CPU, RAM, IOWait increase)
  - **Phase 3 (2:30-3:30)**: Incident peak (critical metrics)
  - **Phase 4 (3:30-5:00)**: Recovery to normal
  - Affected servers show progressive resource exhaustion
  - Metrics increase smoothly (not spiky)

**Example Metrics (Peak)**:
```
CPU: 60-80% (gradual increase)
Memory: 70-85% (gradual increase)
Latency: 150-250ms (gradual increase)
Error Rate: 0.3-0.5 (gradual increase)
Disk I/O: Elevated (IOWait simulation)
State: warning → critical_issue → online
```

---

### 3. **CRITICAL** - Acute Failures with Spikes
- **Description**: System starts healthy then shows acute failure signs with severe random spikes
- **Use Case**: Testing alerting thresholds, failure detection, incident response
- **Behavior**:
  - **Phase 1 (0:00-1:30)**: Mostly stable with occasional micro-spikes
  - **Phase 2 (1:30-2:30)**: Rapid escalation with random spikes
  - **Phase 3 (2:30-3:30)**: Critical failures with severe spikes
  - **Phase 4 (3:30-5:00)**: Slower recovery with residual spikes
  - Affected servers show sudden, dramatic metric spikes
  - Simulates memory leaks, OOM conditions, IOWait storms

**Example Metrics (Peak)**:
```
CPU: 85-100% (severe spikes)
Memory: 80-100% (OOM pressure)
Latency: 300-500ms+ (severe spikes)
Error Rate: 0.7-1.0 (high failure rate)
Disk I/O: 100-200 MB/s spikes (IOWait storms)
State: critical_issue (frequent)
```

---

## 🚀 Usage

### Python API

```python
from demo_data_generator import generate_demo_dataset

# Generate HEALTHY scenario
generate_demo_dataset(
    output_dir="./demo_data/",
    duration_minutes=5,
    fleet_size=10,
    seed=42,
    scenario='healthy'  # or 'degrading' or 'critical'
)
```

### Command Line

```bash
# HEALTHY scenario
python demo_data_generator.py --scenario healthy --duration 5 --fleet-size 10

# DEGRADING scenario (default)
python demo_data_generator.py --scenario degrading --duration 5 --fleet-size 10

# CRITICAL scenario
python demo_data_generator.py --scenario critical --duration 5 --fleet-size 10
```

### Run Complete Demo with Dashboard

```bash
# With scenario selection
python run_demo.py --scenario healthy
python run_demo.py --scenario degrading
python run_demo.py --scenario critical
```

### Jupyter Notebook

In `_StartHere.ipynb`, Cell 4:

```python
# Configuration
DEMO_SCENARIO = 'degrading'  # Options: 'healthy', 'degrading', 'critical'

success = generate_demo_dataset(
    output_dir=DEMO_OUTPUT_DIR,
    duration_minutes=DEMO_DURATION_MIN,
    fleet_size=DEMO_FLEET_SIZE,
    seed=DEMO_SEED,
    scenario=DEMO_SCENARIO
)
```

---

## 📊 Scenario Comparison

| Metric | Healthy | Degrading | Critical |
|--------|---------|-----------|----------|
| CPU Peak | 25-45% | 60-80% | 85-100% |
| Memory Peak | 45-70% | 70-85% | 80-100% |
| Latency Peak | 10-60ms | 150-250ms | 300-500ms+ |
| Error Rate | ~0.001 | 0.3-0.5 | 0.7-1.0 |
| Disk I/O | Normal | Elevated | Severe spikes |
| Pattern | Stable | Gradual | Spiky |
| Critical States | None | Some | Frequent |
| Affected Servers | None | 4 | 4 |

---

## 🧪 Testing All Scenarios

Run the test script to validate all scenarios:

```bash
python test_scenarios.py
```

This generates 1-minute samples of each scenario for quick validation.

---

## 💡 Use Cases by Scenario

### HEALTHY
- ✅ Dashboard layout testing
- ✅ Baseline metric visualization
- ✅ Normal operations monitoring
- ✅ Control group for A/B testing

### DEGRADING
- ✅ Training TFT models for prediction
- ✅ Early warning system testing
- ✅ Capacity planning simulations
- ✅ Gradual degradation detection
- ✅ **Best for ML training** (smooth patterns)

### CRITICAL
- ✅ Alert threshold tuning
- ✅ Incident response testing
- ✅ Anomaly detection validation
- ✅ Spike detection algorithms
- ✅ Worst-case scenario planning

---

## 📝 Metadata

Each generated dataset includes a metadata JSON file with:
- Scenario type and description
- Phase boundaries (for degrading/critical)
- Affected servers list
- Phase distribution
- State distribution
- Generation timestamp

Example: `demo_dataset_metadata.json`

```json
{
  "scenario": "degrading",
  "scenario_description": "System starts healthy and gradually degrades...",
  "affected_servers": ["web-001", "api-005", "db-008", "cache-010"],
  "phase_distribution": {
    "stable": 180,
    "escalation": 120,
    "peak": 120,
    "recovery": 180
  }
}
```

---

## 🔧 Infrastructure & Customization

All scenarios simulate **IBM Spectrum Conductor** infrastructure with realistic node profiles:

**Fleet Profiles** (matching production dataset):
- **Production nodes** (`pprva00a##`): ~33% of fleet - Main production workloads
- **Compute nodes** (`cppr##`): ~33% of fleet - Worker/compute nodes
- **Service nodes** (`csrva##`): ~20% of fleet - Master/service nodes
- **Container nodes** (`crva##`): ~10% of fleet - Notebook/container nodes

**Example Fleet** (10 servers):
- 4x Production: pprva00a01, pprva00a02, pprva00a03, pprva00a04
- 3x Compute: cppr01, cppr02, cppr03
- 2x Service: csrva01, csrva02
- 1x Container: crva01

**Affected Servers**: First server of each profile type

**Server States**: `healthy`, `heavy_load`, `critical_issue`, `recovery` (matching production)

**Reproducibility**: Use the same seed for consistent results across scenarios

---

## 🎓 Training Recommendations

- **Model Training**: Use DEGRADING scenario (smooth, predictable patterns)
- **Alert Testing**: Use CRITICAL scenario (validates spike detection)
- **Baseline Metrics**: Use HEALTHY scenario (establishes normal ranges)
- **Mixed Dataset**: Combine all three for robust training

---

## 🐛 Troubleshooting

**Issue**: All scenarios look the same
- **Solution**: Check that you're using different `scenario` parameter values
- **Verify**: Check metadata file for `scenario` field

**Issue**: No critical states in HEALTHY scenario
- **Expected**: This is correct behavior

**Issue**: CRITICAL scenario too noisy
- **Solution**: This is intentional - use DEGRADING for smoother patterns

---

Generated by MonitoringPrediction v2.0
