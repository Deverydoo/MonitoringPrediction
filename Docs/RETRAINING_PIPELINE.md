# TFT Model Retraining Pipeline Design

**Version**: 1.0.0
**Date**: 2025-10-12
**Status**: Design Document

## Overview

Automated pipeline for detecting when the TFT model needs retraining due to fleet composition changes, and orchestrating the complete retraining workflow.

## Problem Statement

The TFT model uses categorical encoders (NaNLabelEncoder) that learn vocabulary during training. When new servers are added to the fleet or servers are decommissioned, the model's encoder vocabulary becomes stale, resulting in:

- New servers being classified as "unknown" category
- Degraded prediction accuracy for unknown servers
- Increasing percentage of predictions using fallback logic

**Critical Requirement**: "We intended X number servers, the model should use X number of models. If we train 400 in production and only 170 are ever predicted on the dashboard, that is not working. It's broken."

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Fleet Monitor Service                        │
│  - Tracks active servers in environment                          │
│  - Detects new servers and sunset servers                        │
│  - Calculates fleet drift metrics                                │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ├─> Fleet Drift > Threshold?
               │
               v
┌─────────────────────────────────────────────────────────────────┐
│                  Retraining Trigger Service                      │
│  - Monitors fleet drift percentage                               │
│  - Checks prediction coverage metrics                            │
│  - Evaluates model staleness                                     │
│  - Triggers retraining workflow                                  │
└──────────────┬──────────────────────────────────────────────────┘
               │
               v
┌─────────────────────────────────────────────────────────────────┐
│                  Retraining Orchestrator                         │
│  1. Data Collection  → Gather historical metrics                │
│  2. Dataset Generation → Create updated training data            │
│  3. Model Training    → Train with new server vocabulary         │
│  4. Validation       → Verify all servers predict correctly      │
│  5. Deployment       → Hot-swap model in inference service       │
│  6. Verification     → Confirm 100% server coverage              │
└─────────────────────────────────────────────────────────────────┘
```

## Detection Criteria

### 1. Fleet Drift Threshold

**Metric**: `(new_servers + sunset_servers) / total_known_servers`

**Thresholds**:
- **Low Priority** (10-20%): Schedule retraining during maintenance window
- **Medium Priority** (20-35%): Trigger retraining within 24 hours
- **High Priority** (>35%): Immediate retraining required

**Example**:
```
Known servers: 20
New servers: 5
Sunset servers: 2
Drift: (5 + 2) / 20 = 35% → Medium Priority
```

### 2. Unknown Server Detection

**Metric**: `unknown_predictions / total_predictions` over rolling window

**Thresholds**:
- **Acceptable** (<5%): Minor fleet additions, model handles gracefully
- **Warning** (5-15%): Noticeable degradation, schedule retraining
- **Critical** (>15%): Significant model staleness, immediate action

**Implementation**:
```python
# In tft_inference.py
def track_unknown_predictions(self, server_id):
    """Track when predictions are made for unknown servers"""
    if server_id not in self.training_data.categorical_encoders['server_id'].classes_:
        self.unknown_prediction_count += 1
        self.unknown_servers.add(server_id)
```

### 3. Server Sunset Detection

**Metric**: Days since last metrics received

**Policy**:
- **Offline**: No metrics for 24 hours → Mark as offline
- **Sunset Candidate**: No metrics for 7 days → Flag for review
- **Sunset Confirmed**: No metrics for 30 days → Remove from active fleet

**Implementation**:
```python
def detect_sunset_servers(self, metrics_df, days_threshold=7):
    """Identify servers that haven't reported metrics"""
    current_time = pd.Timestamp.now()
    last_seen = metrics_df.groupby('server_name')['timestamp'].max()

    sunset_candidates = []
    for server, last_timestamp in last_seen.items():
        days_offline = (current_time - last_timestamp).days
        if days_offline >= days_threshold:
            sunset_candidates.append({
                'server': server,
                'days_offline': days_offline,
                'last_seen': last_timestamp
            })

    return sunset_candidates
```

### 4. New Server Detection

**Trigger**: Server appears in production metrics but not in model vocabulary

**Implementation**:
```python
def detect_new_servers(self, current_servers, model_servers):
    """Identify servers not in trained model"""
    new_servers = set(current_servers) - set(model_servers)
    return list(new_servers)
```

## Retraining Workflow

### Phase 1: Data Collection

**Objective**: Gather fresh historical data for all active servers

**Steps**:
1. Query production metrics for last N days (default: 30 days)
2. Filter out sunset servers (no metrics for 30+ days)
3. Include all new servers with available metrics
4. Validate data quality using `data_validator.py`

**Requirements**:
- Minimum 7 days of history per server
- At least 1,000 samples per server
- Contract compliance (schema validation)

### Phase 2: Dataset Generation

**Objective**: Create training dataset with updated server fleet

**Command**:
```bash
python metrics_generator.py --days 30 --output training/server_metrics.parquet
```

**Validation**:
- Verify all active servers present
- Check for data quality issues
- Confirm temporal coverage (no large gaps)

### Phase 3: Model Training

**Objective**: Train TFT model with new server vocabulary

**Command**:
```bash
python tft_trainer.py --epochs 20
```

**Critical Files Created**:
- `models/tft_model_<timestamp>/model.safetensors` - Model weights
- `models/tft_model_<timestamp>/dataset_parameters.pkl` - **ENCODERS (CRITICAL)**
- `models/tft_model_<timestamp>/server_mapping.json` - Server hash mappings
- `models/tft_model_<timestamp>/training_info.json` - Metadata

**Success Criteria**:
- `dataset_parameters.pkl` file created ✓
- All active servers in encoder vocabulary ✓
- Training loss converged ✓
- Validation metrics acceptable ✓

### Phase 4: Validation

**Objective**: Verify model predicts all servers correctly

**Test Script**:
```python
def validate_server_coverage(model_dir, expected_servers):
    """Verify all expected servers can be predicted"""

    # Load dataset parameters
    params_file = Path(model_dir) / "dataset_parameters.pkl"
    with open(params_file, 'rb') as f:
        dataset_params = pickle.load(f)

    # Get server encoder vocabulary
    server_encoder = dataset_params['categorical_encoders']['server_id']
    known_servers = set(server_encoder.classes_)

    # Check coverage
    expected = set(expected_servers)
    missing = expected - known_servers

    if missing:
        raise ValueError(f"Model missing {len(missing)} servers: {missing}")

    coverage = len(known_servers & expected) / len(expected) * 100
    print(f"[OK] Server coverage: {coverage:.1f}% ({len(known_servers)}/{len(expected)})")

    return coverage == 100.0
```

### Phase 5: Deployment

**Objective**: Hot-swap new model into inference service

**Methods**:

#### Option A: Rolling Restart (Zero Downtime)
```bash
# Update model symlink
ln -sf models/tft_model_20251012_150000 models/latest

# Signal inference service to reload
curl -X POST http://localhost:8000/admin/reload-model

# Verify reload
curl http://localhost:8000/health
```

#### Option B: Blue-Green Deployment
```bash
# Start new inference instance with new model
python tft_inference.py --daemon --port 8001 --model models/tft_model_20251012_150000

# Verify new instance healthy
curl http://localhost:8001/health

# Switch load balancer to new instance
# ... load balancer config ...

# Graceful shutdown of old instance
curl -X POST http://localhost:8000/admin/shutdown
```

**Validation**:
- Health check passes ✓
- All servers returning predictions ✓
- No "unknown server" warnings ✓
- Response times acceptable ✓

### Phase 6: Verification

**Objective**: Confirm production predictions working correctly

**Checks**:
1. Dashboard shows all expected servers ✓
2. No "unknown class" warnings in logs ✓
3. Prediction latency within SLA ✓
4. Model version updated in metadata ✓

**Monitoring Window**: 24 hours

## Implementation

### File: `retrain_monitor.py`

```python
#!/usr/bin/env python3
"""
TFT Model Retraining Monitor

Tracks fleet composition and triggers retraining when thresholds exceeded.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pickle

class RetrainingMonitor:
    """Monitors fleet changes and triggers retraining workflow"""

    def __init__(self, config_file='config/retrain_config.json'):
        self.config = self.load_config(config_file)
        self.metrics_history = []

    def load_config(self, config_file):
        """Load retraining configuration"""
        with open(config_file) as f:
            return json.load(f)

    def get_current_fleet(self):
        """Get list of servers currently reporting metrics"""
        # Query production metrics DB
        # Return list of active server names
        pass

    def get_model_fleet(self, model_dir):
        """Get list of servers in trained model"""
        params_file = Path(model_dir) / "dataset_parameters.pkl"

        if not params_file.exists():
            raise FileNotFoundError(f"Model encoders not found: {params_file}")

        with open(params_file, 'rb') as f:
            dataset_params = pickle.load(f)

        server_encoder = dataset_params['categorical_encoders']['server_id']
        return set(server_encoder.classes_)

    def calculate_fleet_drift(self, current_servers, model_servers):
        """Calculate percentage drift in fleet composition"""
        current = set(current_servers)
        model = set(model_servers)

        new_servers = current - model
        sunset_servers = model - current

        total_drift = len(new_servers) + len(sunset_servers)
        drift_pct = (total_drift / len(model)) * 100 if model else 0

        return {
            'drift_pct': drift_pct,
            'new_servers': list(new_servers),
            'sunset_servers': list(sunset_servers),
            'total_current': len(current),
            'total_model': len(model)
        }

    def check_retraining_needed(self):
        """Evaluate if retraining should be triggered"""
        current_fleet = self.get_current_fleet()
        model_fleet = self.get_model_fleet(self.config['current_model_dir'])

        drift = self.calculate_fleet_drift(current_fleet, model_fleet)

        # Check thresholds
        if drift['drift_pct'] >= self.config['critical_drift_threshold']:
            return 'CRITICAL', drift
        elif drift['drift_pct'] >= self.config['warning_drift_threshold']:
            return 'WARNING', drift
        else:
            return 'OK', drift

    def trigger_retraining(self, priority='NORMAL'):
        """Initiate retraining workflow"""
        print(f"[RETRAIN] Triggering retraining workflow (Priority: {priority})")

        # Log trigger event
        self.log_event('retrain_triggered', {'priority': priority})

        # Execute retraining pipeline
        # 1. Data collection
        # 2. Dataset generation
        # 3. Model training
        # 4. Validation
        # 5. Deployment
        # 6. Verification

    def log_event(self, event_type, data):
        """Log retraining event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'data': data
        }

        log_file = Path('logs/retraining_events.json')
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def run(self, interval_seconds=3600):
        """Run monitoring loop"""
        print(f"[MONITOR] Starting retraining monitor (interval: {interval_seconds}s)")

        while True:
            try:
                status, drift = self.check_retraining_needed()

                print(f"[STATUS] Fleet drift: {drift['drift_pct']:.1f}% - {status}")
                print(f"  New servers: {len(drift['new_servers'])}")
                print(f"  Sunset servers: {len(drift['sunset_servers'])}")

                if status == 'CRITICAL':
                    self.trigger_retraining(priority='HIGH')
                elif status == 'WARNING':
                    self.trigger_retraining(priority='NORMAL')

                time.sleep(interval_seconds)

            except Exception as e:
                print(f"[ERROR] Monitor error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    monitor = RetrainingMonitor()
    monitor.run()
```

### File: `config/retrain_config.json`

```json
{
  "current_model_dir": "models/latest",
  "warning_drift_threshold": 20.0,
  "critical_drift_threshold": 35.0,
  "sunset_days_threshold": 30,
  "minimum_history_days": 7,
  "retraining_epochs": 20,
  "validation_required": true,
  "hot_swap_enabled": true,
  "notification_webhook": "https://slack.webhook.url/retraining"
}
```

## Operational Procedures

### Manual Retraining

When you need to manually trigger retraining:

```bash
# 1. Generate fresh dataset
python metrics_generator.py --days 30

# 2. Train model
python tft_trainer.py --epochs 20

# 3. Validate server coverage
python validate_model.py --model models/tft_model_<timestamp> --expected-servers server_list.txt

# 4. Deploy to inference service
./deploy_model.sh models/tft_model_<timestamp>

# 5. Verify dashboard
python verify_dashboard.py --expected-count 30
```

### Automated Retraining

Set up monitoring service:

```bash
# Start retraining monitor as service
python retrain_monitor.py --daemon

# Monitor logs
tail -f logs/retraining_events.json

# Check status
curl http://localhost:9000/status
```

## Metrics & Monitoring

### Key Metrics to Track

1. **Fleet Drift**: Percentage change in server composition
2. **Unknown Prediction Rate**: % of predictions for unknown servers
3. **Model Age**: Days since last training
4. **Prediction Coverage**: % of active servers with predictions
5. **Retraining Duration**: Time to complete full pipeline
6. **Retraining Success Rate**: % of successful retrainings

### Dashboard Alerts

- **Yellow Alert**: Fleet drift 20-35%, schedule retraining
- **Red Alert**: Fleet drift >35% or unknown rate >15%, immediate action
- **Model Staleness**: No retraining in >30 days with active fleet changes

## Testing Strategy

### Unit Tests

```python
def test_fleet_drift_calculation():
    """Test drift calculation logic"""
    monitor = RetrainingMonitor()

    current = ['s1', 's2', 's3', 's4', 's5']
    model = ['s1', 's2', 's3']

    drift = monitor.calculate_fleet_drift(current, model)

    assert drift['drift_pct'] == 66.67  # 2 new servers out of 3 = 66%
    assert set(drift['new_servers']) == {'s4', 's5'}
    assert drift['sunset_servers'] == []

def test_encoder_persistence():
    """Verify encoders saved and loaded correctly"""
    # Train model
    # Load dataset_parameters.pkl
    # Verify all servers in vocabulary
    pass

def test_hot_swap_deployment():
    """Test model reload without downtime"""
    # Start inference service
    # Trigger reload with new model
    # Verify predictions continue during reload
    pass
```

### Integration Tests

```python
def test_end_to_end_retraining():
    """Test complete retraining pipeline"""
    # 1. Add new servers to test environment
    # 2. Trigger retraining
    # 3. Verify new model includes new servers
    # 4. Validate all servers predict
    # 5. Check dashboard shows all servers
    pass
```

## Rollback Procedure

If new model fails validation:

```bash
# 1. Keep old model as backup
cp -r models/latest models/backup_<timestamp>

# 2. Revert symlink
ln -sf models/backup_<timestamp> models/latest

# 3. Reload inference service
curl -X POST http://localhost:8000/admin/reload-model

# 4. Verify rollback successful
curl http://localhost:8000/health
```

## Future Enhancements

### Phase 2 Features

1. **Incremental Learning**: Update encoders without full retraining
2. **A/B Testing**: Compare new model vs old model performance
3. **Automated Rollback**: Auto-revert if validation fails
4. **Slack Notifications**: Alert team when retraining triggered
5. **Model Performance Tracking**: Log accuracy metrics over time

### Phase 3 Features

1. **Kubernetes Integration**: Deploy as K8s CronJob
2. **Multi-Region Support**: Coordinate retraining across regions
3. **Cost Optimization**: Schedule retraining during low-cost periods
4. **Explainability**: Log why retraining was triggered

## References

- [DATA_CONTRACT.md](DATA_CONTRACT.md) - Data validation schema
- [tft_trainer.py:783-792](../tft_trainer.py#L783-L792) - Encoder persistence code
- [tft_inference.py:397-453](../tft_inference.py#L397-L453) - Encoder loading code
- [server_encoder.py](../server_encoder.py) - Hash-based server encoding

## Changelog

**v1.0.0** (2025-10-12)
- Initial design document
- Fleet drift detection
- Automated retraining workflow
- Hot-swap deployment strategy
