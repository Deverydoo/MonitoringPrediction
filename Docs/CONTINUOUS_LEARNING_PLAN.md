# Continuous Learning System - Design Document

**Project**: TFT Monitoring System v1.0.0
**Feature**: Daily Automatic Retraining
**Status**: Planning Phase
**Date**: 2025-10-17

---

## Executive Summary

Implement an automated continuous learning system where the TFT model retrains daily on new production data, keeping predictions accurate as infrastructure patterns evolve.

**Key Benefits:**
- Model stays current with changing workload patterns
- Catches new failure modes automatically
- No manual retraining required
- Improves prediction accuracy over time

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS LEARNING PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Live Data   │────▶│ Data Buffer  │────▶│  Training    │
│  (Inference) │     │  (Parquet)   │     │  Scheduler   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                      │
                            │                      ▼
                            │              ┌──────────────┐
                            │              │  Incremental │
                            │              │   Training   │
                            │              └──────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ Data Window  │     │  New Model   │
                     │  Management  │     │  Checkpoint  │
                     └──────────────┘     └──────────────┘
                            │                      │
                            └──────────────────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │   Inference  │
                               │    Daemon    │
                               └──────────────┘
```

---

## 2. Data Collection Strategy

### 2.1 Inference Daemon Data Buffering

**Current State:**
- Inference daemon receives metrics every 5 seconds
- Data is used for predictions then discarded
- No historical accumulation

**Proposed:**
```python
# In tft_inference_daemon.py
class DataBuffer:
    """
    Accumulates incoming metrics for daily retraining.

    Features:
    - Append-only parquet files (1 file per day)
    - Automatic rotation at midnight
    - Retention policy (keep last 60 days)
    """

    def __init__(self, buffer_dir='./data_buffer'):
        self.buffer_dir = Path(buffer_dir)
        self.current_date = datetime.now().date()
        self.current_file = None

    def append(self, records: List[Dict]):
        """Append incoming metrics to daily buffer file."""

    def rotate_if_needed(self):
        """Check if new day, rotate to new file."""

    def get_training_window(self, days=30) -> Path:
        """Return parquet file with last N days for training."""
```

**Storage Pattern:**
```
data_buffer/
├── metrics_2025-10-17.parquet  (today - being written)
├── metrics_2025-10-16.parquet  (yesterday)
├── metrics_2025-10-15.parquet
...
└── metrics_2025-09-17.parquet  (30 days ago)
```

### 2.2 Data Window Management

**Sliding Window Approach:**
- Keep last 30 days of data (configurable)
- Older data archived or deleted
- Training uses full 30-day window each night

**Why 30 Days?**
- Captures monthly patterns (EOD, weekly cycles)
- Includes edge cases and incidents
- Balances freshness vs. training stability
- ~25-50 MB per day = 750 MB - 1.5 GB total

---

## 3. Training Scheduler

### 3.1 Schedule Design

**Daily Training Window:**
- **Time**: 2:00 AM - 4:00 AM (low traffic)
- **Frequency**: Once per day
- **Duration**: 1-2 hours (1 epoch on 30 days of data)

**Why 2 AM?**
- Minimal production load
- Overnight batch jobs finished
- Before pre-market activity (7 AM)

### 3.2 Scheduler Implementation Options

#### Option A: Cron Job (Linux/Mac)
```bash
# /etc/cron.d/tft_training
0 2 * * * cd /path/to/MonitoringPrediction && python continuous_learning.py
```

#### Option B: Task Scheduler (Windows)
```xml
<!-- Windows Task Scheduler XML -->
<Task>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2025-10-18T02:00:00</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>python</Command>
      <Arguments>continuous_learning.py</Arguments>
    </Exec>
  </Actions>
</Task>
```

#### Option C: Python Scheduler (Cross-Platform) ⭐ **RECOMMENDED**
```python
# continuous_learning_daemon.py
import schedule
import time

def daily_training_job():
    """Runs at 2 AM daily."""
    print(f"[{datetime.now()}] Starting daily training...")

    # 1. Consolidate last 30 days of data
    # 2. Run 1 epoch of incremental training
    # 3. Deploy new model checkpoint
    # 4. Send success/failure notification

schedule.every().day.at("02:00").do(daily_training_job)

while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

---

## 4. Training Process

### 4.1 Daily Training Flow

```python
# continuous_learning.py
def daily_training_workflow():
    """
    Daily training workflow.

    Steps:
    1. Data validation
    2. Incremental training (1 epoch)
    3. Model validation
    4. Checkpoint promotion
    5. Notification
    """

    # Step 1: Prepare data
    buffer = DataBuffer('./data_buffer')
    training_data = buffer.get_training_window(days=30)

    if not validate_data(training_data):
        send_alert("Training skipped - insufficient data")
        return False

    # Step 2: Train incrementally (1 epoch)
    print(f"[TRAIN] Starting incremental training on 30-day window...")
    model_path = train_model(
        dataset_path=training_data,
        epochs=1,           # Just 1 epoch per day
        incremental=True    # Resume from checkpoint
    )

    if not model_path:
        send_alert("Training failed - check logs")
        return False

    # Step 3: Validate new model
    if not validate_model(model_path):
        send_alert("Model validation failed - rolling back")
        return False

    # Step 4: Promote checkpoint
    promote_checkpoint(model_path)

    # Step 5: Success notification
    total_epochs = get_total_epochs(model_path)
    send_notification(f"✅ Training complete: {total_epochs} total epochs")

    return True
```

### 4.2 Training Parameters

**Daily Training:**
- **Epochs**: 1 (incremental)
- **Batch Size**: 64 (from config)
- **Data Window**: 30 days (~720 hours)
- **Duration**: ~1-2 hours on GPU

**Why 1 Epoch/Day?**
- Continuous gradual learning
- Lower risk of overfitting
- Faster training (production-friendly)
- After 30 days = 30 total epochs (well-trained)

---

## 5. Model Validation & Safety

### 5.1 Validation Checks

Before deploying new checkpoint:

```python
def validate_model(model_path: Path) -> bool:
    """
    Validate new model before deployment.

    Checks:
    1. Model loads successfully
    2. Predictions are reasonable (no NaN, within bounds)
    3. Validation loss improved or stable
    4. Test predictions match expected ranges
    """

    # Load model
    model = load_checkpoint(model_path)

    # Generate test predictions
    test_data = get_validation_data()
    predictions = model.predict(test_data)

    # Check for NaN or invalid values
    if has_invalid_predictions(predictions):
        logger.error("Model produced invalid predictions")
        return False

    # Check prediction ranges
    if not predictions_in_valid_range(predictions):
        logger.error("Predictions outside valid range")
        return False

    # Compare validation loss
    current_loss = get_current_model_loss()
    new_loss = get_validation_loss(model)

    if new_loss > current_loss * 1.2:  # 20% worse
        logger.warning(f"New model worse: {new_loss} vs {current_loss}")
        return False

    return True
```

### 5.2 Rollback Strategy

If validation fails:
1. Keep current checkpoint active
2. Log failure reason
3. Alert operations team
4. Retry tomorrow with more data

---

## 6. Monitoring & Alerting

### 6.1 Training Metrics to Track

```python
training_metrics = {
    'timestamp': datetime.now(),
    'total_epochs': 31,
    'training_duration_mins': 87,
    'validation_loss': 0.045,
    'training_loss': 0.038,
    'data_samples': 518400,  # 30 days × 20 servers × 5s intervals
    'success': True
}
```

### 6.2 Alert Conditions

**Critical Alerts (page on-call):**
- Training failed 2 days in a row
- Model validation failed
- Data buffer corruption
- Prediction accuracy dropped >10%

**Warning Alerts (email/Slack):**
- Training took >3 hours
- Validation loss increased
- Disk space low (<20%)
- Missing data for some servers

### 6.3 Monitoring Dashboard

Track over time:
- Total epochs trained
- Daily training duration
- Validation loss trend
- Model performance metrics
- Data quality scores

---

## 7. Implementation Phases

### Phase 1: Data Collection (Week 1)
- [ ] Implement DataBuffer in inference daemon
- [ ] Add parquet append logic
- [ ] Implement daily rotation
- [ ] Test data accumulation for 7 days

### Phase 2: Training Automation (Week 2)
- [ ] Create continuous_learning.py script
- [ ] Implement data consolidation
- [ ] Add model validation checks
- [ ] Test manual daily training

### Phase 3: Scheduling (Week 3)
- [ ] Implement Python scheduler daemon
- [ ] Add systemd/Windows Service integration
- [ ] Configure 2 AM daily run
- [ ] Test automated execution

### Phase 4: Monitoring (Week 4)
- [ ] Add training metrics logging
- [ ] Implement alert system
- [ ] Create monitoring dashboard
- [ ] Document operations runbook

---

## 8. Configuration

### 8.1 New Config Section

```python
# config/continuous_learning_config.py
CONTINUOUS_LEARNING_CONFIG = {
    # Data Buffer
    'data_buffer_dir': './data_buffer',
    'data_retention_days': 60,
    'training_window_days': 30,

    # Training Schedule
    'training_time': '02:00',  # 2 AM
    'training_timezone': 'America/New_York',
    'epochs_per_day': 1,

    # Validation
    'validation_loss_threshold': 1.2,  # Max 20% worse
    'min_training_samples': 100000,

    # Monitoring
    'alert_email': 'ml-ops@company.com',
    'slack_webhook': 'https://hooks.slack.com/...',
    'log_dir': './logs/continuous_learning',

    # Safety
    'enable_auto_deploy': True,
    'require_validation': True,
    'max_training_duration_hours': 4
}
```

---

## 9. Benefits & ROI

### 9.1 Business Value

**Before (Manual Retraining):**
- Retrain monthly: 30-day lag in learning new patterns
- Manual process: 4-6 hours engineer time
- Risk of staleness: Model misses recent incidents
- Cost: ~$500/month in engineer time

**After (Continuous Learning):**
- Retrain daily: 1-day lag maximum
- Automated: Zero engineer time
- Always current: Learns new patterns immediately
- Cost: ~$50/month in compute (2 hours/day GPU)

**ROI: 10x cost reduction + better accuracy**

### 9.2 Technical Benefits

✅ Model adapts to seasonal patterns (holidays, quarter-end)
✅ Learns from recent incidents automatically
✅ Catches infrastructure changes (new servers, config updates)
✅ Reduces false positives as system evolves
✅ Improves prediction accuracy over time

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training fails | Predictions stale | Validation checks, rollback, alerts |
| Bad data in buffer | Model degrades | Data validation, anomaly detection |
| Disk full | Training stops | Retention policy, monitoring |
| Model overfits | Poor predictions | 1 epoch/day, validation loss check |
| Compute cost | Budget overrun | GPU scheduling, cost monitoring |

---

## 11. Success Metrics

Track these KPIs:

1. **Training Success Rate**: >95% daily
2. **Model Freshness**: <24 hours lag
3. **Prediction Accuracy**: Improve 5% month-over-month
4. **Automation Rate**: 100% (zero manual intervention)
5. **Training Duration**: <2 hours/day

---

## 12. Next Steps

1. **Immediate**: Review and approve this plan
2. **Week 1**: Implement data buffering in inference daemon
3. **Week 2**: Create continuous_learning.py script
4. **Week 3**: Deploy scheduler daemon
5. **Week 4**: Monitor and iterate

**Target Go-Live**: 4 weeks from approval

---

## Appendix A: File Structure

```
MonitoringPrediction/
├── data_buffer/                    # NEW - Daily metrics
│   ├── metrics_2025-10-17.parquet
│   └── metrics_2025-10-16.parquet
├── continuous_learning.py          # NEW - Training workflow
├── continuous_learning_daemon.py   # NEW - Scheduler
├── checkpoints/                    # Existing
│   └── last.ckpt                   # Updated daily
├── models/                         # Existing
│   └── tft_model_YYYYMMDD/         # New model each day
└── logs/
    └── continuous_learning/        # NEW - Training logs
        ├── training_2025-10-17.log
        └── metrics.json
```

---

**Document Status**: Draft for Review
**Approvers**: Engineering Lead, ML Team, Operations
**Next Review**: After Phase 1 completion
