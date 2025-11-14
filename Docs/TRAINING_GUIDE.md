# TFT Model Training Guide

**Version:** 2.0.0
**Last Updated:** 2025-11-14
**Status:** Production Reference

---

## Table of Contents

1. [Overview](#1-overview)
2. [Training Guidelines & Best Practices](#2-training-guidelines--best-practices)
   - 2.1 [Training Configurations](#21-training-configurations)
   - 2.2 [Accuracy Claims & Validation](#22-accuracy-claims--validation)
   - 2.3 [Key Metrics to Track](#23-key-metrics-to-track)
   - 2.4 [Demo Strategy](#24-demo-strategy)
3. [Adaptive Retraining Strategy](#3-adaptive-retraining-strategy)
   - 3.1 [System Architecture](#31-system-architecture)
   - 3.2 [Drift Detection System](#32-drift-detection-system)
   - 3.3 [Quiet Time Detection](#33-quiet-time-detection)
   - 3.4 [Training Decision Engine](#34-training-decision-engine)
   - 3.5 [Safeguards & Policies](#35-safeguards--policies)
4. [Continuous Learning System](#4-continuous-learning-system)
   - 4.1 [Data Collection Strategy](#41-data-collection-strategy)
   - 4.2 [Daily Training Workflow](#42-daily-training-workflow)
   - 4.3 [Model Validation & Safety](#43-model-validation--safety)
   - 4.4 [Scheduling Options](#44-scheduling-options)
5. [Implementation Details](#5-implementation-details)
   - 5.1 [Adaptive Retraining Implementation](#51-adaptive-retraining-implementation)
   - 5.2 [Continuous Learning Implementation](#52-continuous-learning-implementation)
   - 5.3 [Configuration Reference](#53-configuration-reference)
   - 5.4 [File Structure](#54-file-structure)
6. [Monitoring & Operations](#6-monitoring--operations)
   - 6.1 [Monitoring Dashboard](#61-monitoring-dashboard)
   - 6.2 [Alert Conditions](#62-alert-conditions)
   - 6.3 [Success Metrics](#63-success-metrics)
7. [Troubleshooting](#7-troubleshooting)
   - 7.1 [Common Issues](#71-common-issues)
   - 7.2 [FAQ](#72-faq)
   - 7.3 [Rollback Procedures](#73-rollback-procedures)
8. [Appendix](#8-appendix)
   - 8.1 [Example Scenarios](#81-example-scenarios)
   - 8.2 [Production Deployment Checklist](#82-production-deployment-checklist)
   - 8.3 [Benefits & ROI Analysis](#83-benefits--roi-analysis)

---

## 1. Overview

### Purpose

This comprehensive guide covers all aspects of training the TFT (Temporal Fusion Transformer) model for the NordIQ monitoring prediction system. It provides guidance for:

- **Initial Training**: Getting started with demos, validation, and production deployment
- **Adaptive Retraining**: Intelligent drift-based retraining during quiet periods
- **Continuous Learning**: Automated daily retraining for evolving infrastructure patterns

### Key Principles

**Training Philosophy:**
- Only claim what you can measure with validated metrics
- Train incrementally to avoid overfitting and excessive compute costs
- Adapt to data drift proactively while respecting production systems
- Automate where possible to reduce manual intervention

**Performance Targets:**
- Target accuracy: 85-90% based on TFT benchmarks and profile-based transfer learning
- SLA threshold: 88% accuracy minimum for production deployment
- Early warning buffer: Maintain 92%+ accuracy in normal operations

### System Architecture Context

The TFT model is part of a larger monitoring prediction system:
- **Inference Daemon**: Processes live metrics every 5 seconds, generates 8-hour predictions
- **REST API**: Serves predictions and supports interactive scenario testing
- **Dashboard**: Visualizes predictions and system health metrics
- **Training Pipeline**: Retrains model based on drift detection or scheduled intervals

---

## 2. Training Guidelines & Best Practices

### 2.1 Training Configurations

#### 2.1.1 Quick Demo (30 minutes)

**Use Case:** Proof of concept, architecture demonstration, time-constrained demos

```bash
# Generate data
python metrics_generator.py --hours 168 --out_dir ./training/

# Train model
python tft_trainer.py --epochs 1
```

**Specifications:**
- **Data:** 1 week (168 hours)
- **Epochs:** 1
- **Training Time:** ~30 minutes on RTX 4090
- **Records:** ~33,600 (20 servers × 168 hours × 12 samples/hour)

**What You Get:**
- Working end-to-end pipeline
- Model loads and generates predictions
- Demo-ready system for architecture validation
- Initial loss metrics for baseline

**Limitations:**
- Model has NOT converged
- No validated accuracy metrics available
- NOT production-ready
- Limited long-term pattern learning

**Appropriate Claims:**
> "This demonstrates the TFT architecture and prediction capability. The system is functional and ready for production training with extended data and epochs."

**Inappropriate Claims:**
- Any specific accuracy percentages
- "Production-ready predictions"
- "Fully trained model"
- Performance comparisons to other systems

---

#### 2.1.2 Validation Training (2-6 hours)

**Use Case:** Realistic validation, initial performance metrics, pilot deployments

```bash
# Generate data
python metrics_generator.py --hours 720 --out_dir ./training/

# Train model
python tft_trainer.py --epochs 10
```

**Specifications:**
- **Data:** 30 days (720 hours)
- **Epochs:** 10
- **Training Time:** ~4-6 hours on RTX 4090
- **Records:** ~172,800 (20 servers × 720 hours × 12 samples/hour)

**What You Get:**
- Partial convergence toward optimal performance
- Weekly pattern learning (weekday vs. weekend cycles)
- Measurable validation metrics for honest baselines
- Reasonable predictions for pilot testing

**Limitations:**
- Full convergence requires 20 epochs
- Monthly/seasonal patterns not fully captured
- Performance below optimal level

**Appropriate Claims:**
> "Model trained on 30 days of data with 10 epochs. Initial validation shows [X]% quantile loss. Performance will improve with additional training epochs."

**What to Report:**
- Train loss (final value from epoch 10)
- Validation loss (final value from epoch 10)
- Loss convergence trend across epochs
- Time per epoch
- Sample prediction examples vs. actuals

---

#### 2.1.3 Production Training (30-40 hours)

**Use Case:** Production deployment, validated performance claims, maximum accuracy

```bash
# Generate data
python metrics_generator.py --hours 720 --out_dir ./training/

# Train model
python tft_trainer.py --epochs 20
```

**Specifications:**
- **Data:** 30 days (720 hours)
- **Epochs:** 20
- **Training Time:** ~30-40 hours on RTX 4090
- **Records:** ~172,800 (20 servers × 720 hours × 12 samples/hour)

**What You Get:**
- Full convergence to optimal performance
- Production-quality predictions with validated accuracy
- Comprehensive weekly and monthly pattern learning
- Robust performance across diverse scenarios

**Appropriate Claims:**
> "Model trained on 30 days of historical data with 20 epochs to full convergence. Validation metrics show [X]% accuracy on held-out data."

**What to Report:**
- Final train/validation loss values
- Convergence curves showing loss reduction over epochs
- Per-metric prediction quality (CPU, memory, disk I/O)
- Quantile loss values for confidence intervals
- Production readiness assessment

---

#### 2.1.4 Large-Scale Production (8+ hours per epoch)

**Use Case:** Enterprise deployment with 100+ servers

```bash
# Generate data for large infrastructure
python metrics_generator.py \
    --hours 720 \
    --num_ml_compute 40 \
    --num_database 25 \
    --num_web_api 50 \
    --num_conductor_mgmt 8 \
    --num_data_ingest 15 \
    --num_risk_analytics 12 \
    --num_generic 10 \
    --out_dir ./training/

# Train model
python tft_trainer.py --epochs 20
```

**Specifications:**
- **Data:** 30 days, 160 servers
- **Epochs:** 20
- **Training Time:** ~8-12 hours per epoch (160-240 hours total)
- **Records:** ~1.4M records

**Optimization Recommendations:**
- Enable gradient accumulation to manage memory
- Enable checkpointing every 5 epochs for failure recovery
- Monitor GPU memory usage continuously
- Use mixed precision training (bf16-mixed) for speed
- Consider distributed training across multiple GPUs

---

### 2.2 Accuracy Claims & Validation

#### 2.2.1 Understanding the "88%" Target

The **88% accuracy** referenced in presentation materials represents:
- A **target** based on TFT and transfer learning benchmarks from academic literature
- Theoretical improvement expected from profile-based architecture (10-15% over baseline)
- **NOT** empirically validated from actual training runs (yet)
- **NOT** a guarantee without proper validation

**Source:** Published research on profile-based transfer learning showing 10-15% improvement over baseline prediction models.

---

#### 2.2.2 Honest Accuracy Communication

**Without Full Training (1 epoch):**
> "The TFT architecture with profile-based transfer learning is designed to achieve 85-90% accuracy based on published benchmarks. Full validation will come from production pilot data."

**With Validation Training (10 epochs):**
> "Model achieves [measured validation loss] on held-out data. Preliminary results show [describe prediction quality]. Full convergence expected with 20 epochs."

**With Production Training (20 epochs):**
> "Model achieves [X]% accuracy on validation set with [Y] quantile loss. Predictions show strong alignment with historical patterns."

---

#### 2.2.3 What NOT to Say

**Inappropriate Claims (Avoid These):**
- "88% accuracy" without empirical measurement
- "Industry-leading accuracy" without comparison data
- "Better than existing solutions" without head-to-head testing
- "Highly accurate" (too vague and unmeasurable)

---

#### 2.2.4 Focus on Proven Differentiators

When accuracy numbers are uncertain, pivot to features that ARE demonstrated:

**Proven Technical Features:**
- 8-hour prediction horizon (demonstrated and working)
- Profile-based transfer learning (implemented and tested)
- Production-ready architecture (REST API, daemon, dashboard)
- Sub-3-second inference latency for real-time predictions
- Interactive scenario testing capability
- Unknown server handling via hash-based encoding

**Validated Value Propositions:**
- Early warning time: Predict 30 minutes to 8 hours ahead
- Proactive operations: Act before incidents occur
- Reduced MTTR (Mean Time To Recovery): Faster incident response
- Capacity planning: Predict resource needs proactively
- Cost optimization: Right-size infrastructure based on predictions

---

### 2.3 Key Metrics to Track

#### 2.3.1 During Training

Monitor and record these metrics per epoch:

```python
{
  "epoch": 15,
  "train_loss": 0.089,
  "val_loss": 0.102,
  "learning_rate": 0.001,
  "time_seconds": 5420,
  "gpu_memory_mb": 8500,
  "samples_processed": 172800
}
```

**Loss Metrics:**
- Training loss per epoch (should decrease steadily)
- Validation loss per epoch (should track training loss)
- Quantile loss (final value at convergence)

**Convergence Indicators:**
- Loss curve trends (looking for plateau)
- Early stopping triggers (validation loss stops improving)
- Learning rate schedule adjustments

**Prediction Quality:**
- Sample predictions vs. actual values
- Confidence interval calibration (P10, P50, P90)
- Per-metric performance breakdown (CPU, memory, disk)

**System Performance:**
- Time per epoch (should remain consistent)
- GPU memory usage (watch for OOM errors)
- Training throughput (samples/second)

---

#### 2.3.2 After Training

Save comprehensive metadata to `training_info.json`:

```python
{
  "trained_at": "2025-11-14T02:30:00",
  "epochs": 20,
  "data_hours": 720,
  "num_servers": 20,
  "final_train_loss": 0.089,
  "final_val_loss": 0.102,
  "quantile_loss": 0.095,
  "training_time_hours": 38.5,
  "model_parameters": 87080,
  "convergence": "achieved",
  "early_stopping": false,
  "validation_samples": 34560,
  "mean_absolute_error": 4.2,
  "mean_squared_error": 23.7
}
```

---

### 2.4 Demo Strategy

#### 2.4.1 Timeline-Based Recommendations

**3 Days to Demo:**
```bash
# Run immediately (30 min)
python metrics_generator.py --hours 168 --out_dir ./training/
python tft_trainer.py --epochs 1

# OPTIONAL: Start longer training in background if time allows
python tft_trainer.py --epochs 10  # Using same data
```

**Strategy:** Use 1-epoch model for demo, mention that 10-epoch validation is running for future accuracy validation.

---

**1 Week to Demo:**
```bash
# Best balance of quality and time
python metrics_generator.py --hours 720 --out_dir ./training/
python tft_trainer.py --epochs 10
```

**Strategy:** Show actual validation metrics from 10-epoch training, commit to 20-epoch production training.

---

**1 Month to Demo:**
```bash
# Full production training
python metrics_generator.py --hours 720 --out_dir ./training/
python tft_trainer.py --epochs 20
```

**Strategy:** Make full accuracy claims with confidence backed by convergence metrics.

---

#### 2.4.2 Answering Accuracy Questions

**The Question:**
> "What's the model's accuracy?"

**Good Answers (Based on Training Level):**

**Quick Demo (1 epoch):**
> "This is a proof-of-concept model demonstrating the architecture. The TFT model with profile-based transfer learning is expected to achieve 85-90% accuracy based on benchmarks. Full validation requires completing the 20-epoch training cycle."

**Validation Training (10 epochs):**
> "The model is partially trained with 10 epochs on 30 days of data. Current validation loss is [X], showing good convergence. We expect final accuracy in the 85-90% range after completing the full 20-epoch training."

**Production Training (20 epochs):**
> "The model achieved [measured accuracy]% on held-out validation data after 20 epochs of training on 30 days of historical metrics. Quantile loss is [Y], indicating strong prediction quality."

---

## 3. Adaptive Retraining Strategy

### 3.1 System Architecture

The adaptive retraining system monitors data drift and prediction accuracy, automatically triggering retraining during quiet periods when the model shows signs of staleness.

```
┌─────────────────────────────────────────────────────────────────┐
│              ADAPTIVE RETRAINING SYSTEM                          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Live Data   │────▶│ Drift Monitor│────▶│ Retraining   │
│  (Inference) │     │  (Real-time) │     │  Decision    │
└──────────────┘     └──────────────┘     │   Engine     │
                            │              └──────────────┘
                            │                      │
                            ▼                      │
                     ┌──────────────┐             │
                     │ Prediction   │             │
                     │   Accuracy   │             │
                     │   Tracker    │             │
                     └──────────────┘             │
                            │                      │
                            └──────────────────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │ Quiet Time   │
                               │  Detector    │
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │  Training    │
                               │  Throttle    │
                               └──────────────┘
                                       │
                                       ▼
                               ┌──────────────┐
                               │ Incremental  │
                               │  Training    │
                               └──────────────┘
```

**Key Principles:**
- **Event-Driven**: Train when drift detected, not on fixed schedule
- **Context-Aware**: Only train during server quiet times
- **Safeguarded**: Minimum 6 hours between trainings, maximum 30 days without training
- **Intelligent**: Learns optimal training windows from historical load patterns

---

### 3.2 Drift Detection System

#### 3.2.1 Types of Drift

**Data Drift:** Input data distribution changes
- New server profiles added to infrastructure
- Infrastructure changes (new hardware, configuration updates)
- Workload pattern shifts (new applications, seasonal changes)

**Prediction Drift:** Model predictions become less accurate
- Actual metrics deviate from predictions
- Increasing prediction errors over time
- Model "forgetting" recent patterns due to staleness

---

#### 3.2.2 Drift Detection Metrics

**Four Key Metrics:**

1. **Prediction Error Rate (PER)**: Most important metric
   - Compares predictions vs. actual values over last 1000 samples
   - PER = mean(|predicted - actual| / actual) over window
   - Thresholds aligned with 88% accuracy SLA

2. **Distribution Shift Score (DSS)**: Statistical distribution comparison
   - Uses Kolmogorov-Smirnov test for distribution similarity
   - Compares current data distribution to baseline
   - Returns score 0.0-1.0 (0=identical, 1=completely different)

3. **Feature Drift Score (FDS)**: Sudden changes in feature statistics
   - FDS = Σ |current_mean - baseline_mean| / baseline_std
   - High FDS indicates data characteristics changed
   - Detects shifts in metric ranges or patterns

4. **Anomaly Rate**: Track rate of anomalous predictions
   - Anomaly = prediction outside expected range (e.g., CPU > 100%)
   - High anomaly rate suggests model confusion

---

#### 3.2.3 Drift Thresholds (Aligned with 88% SLA)

| Drift Score | Accuracy | Status | Action |
|------------|----------|--------|--------|
| < 0.08 | > 92% | Healthy | Monitor normally |
| 0.08-0.10 | 88-92% | Warning | Increase monitoring frequency |
| 0.10-0.12 | 86-88% | Urgent | Schedule retraining |
| > 0.12 | < 86% | Critical | Immediate retraining (SLA breach) |

**Retraining Trigger:** Drift score > 0.10 (88% accuracy threshold)

---

#### 3.2.4 Weighted Drift Calculation

```python
# Aggregate drift metrics into single signal
drift_score = (
    per * 0.40 +           # Prediction error most important
    dss * 0.30 +           # Distribution shift
    fds * 0.20 +           # Feature drift
    anomaly_rate * 0.10    # Anomaly rate
)
```

**Decision Logic:**
- `drift_score > 0.10`: Critical - trigger retraining (at/below SLA)
- `drift_score > 0.08`: Warning - monitor closely (approaching SLA)
- `drift_score < 0.08`: Healthy - continue normal operations (above SLA)

---

### 3.3 Quiet Time Detection

#### 3.3.1 What is "Quiet Time"?

Periods when infrastructure load is low enough to safely run training without impacting production:
- Server load is low (average CPU < 30%)
- Prediction frequency is low (< 50% of peak rate)
- No active incidents or alerts
- Minimal user activity

**Typical Quiet Windows:**
- Weekends (lower business activity)
- Nights (2-5 AM in local timezone)
- Post-market hours (6 PM - 8 PM for financial systems)

---

#### 3.3.2 Load Calculation

```python
# Calculate current infrastructure load (0.0-1.0 scale)
current_metrics = get_current_metrics()
avg_cpu = mean([server['cpu_user_pct'] for server in current_metrics])

pred_rate = get_prediction_rate_last_5min()
baseline_rate = historical_load['prediction_rate']['p95']
rate_ratio = pred_rate / baseline_rate

# Weighted load score
load_score = (avg_cpu / 100 * 0.6) + (rate_ratio * 0.4)
```

**Quiet Threshold:** `load_score < 0.3` (< 30% load)

---

#### 3.3.3 Quiet Window Prediction

The system predicts future quiet windows based on historical patterns:

```python
# Check next 24 hours in 15-minute intervals
for hours_ahead in range(0, 24, 0.25):
    check_time = now + timedelta(hours=hours_ahead)
    predicted_load = predict_load_at_time(check_time)

    # Find windows where load < 0.3 for at least 1.5 hours
    if predicted_load < 0.3:
        # Potential quiet window
```

**Learning Patterns:**
- Hour of day (diurnal cycles)
- Day of week (weekly patterns)
- Special events (holidays, quarter-end processing)

**Minimum Quiet Duration:** 1.5 hours (ensures 1 epoch training completes)

---

### 3.4 Training Decision Engine

#### 3.4.1 Decision Algorithm

The decision engine considers multiple inputs:
- Drift signal from DriftMonitor
- Quiet time status from QuietTimeDetector
- Training history (last training time, frequency)
- System safeguards and policies

**Main Decision Function:**

```python
def should_trigger_training() -> Dict:
    """
    Decides whether to trigger retraining now.

    Returns:
    {
        'trigger': True/False,
        'reason': 'Explanation of decision',
        'confidence': 0.0-1.0,
        'wait_until': datetime (if delaying),
        'priority': 'normal'|'high'|'critical'
    }
    """
```

---

#### 3.4.2 Decision Flow

**Step 1: Safeguard Checks**
- Check minimum time between trainings (6 hours)
- Check maximum time without training (30 days)
- Check weekly training limit (3 trainings max)

**Step 2: Drift Check**
- Get current drift signal from DriftMonitor
- If no drift detected, return False

**Step 3: Quiet Time Check**
- Check if current time is quiet enough
- If quiet now: trigger training immediately
- If not quiet: predict next quiet window

**Step 4: Priority Override**
- If drift_score > 0.12 (SLA breach) and no quiet window in 24h
- Train anyway despite load (critical priority)

---

### 3.5 Safeguards & Policies

| Safeguard | Value | Purpose |
|-----------|-------|---------|
| **Min Time Between Training** | 6 hours | Prevent training thrashing and model instability |
| **Max Time Without Training** | 30 days | Force periodic refresh to avoid staleness |
| **Max Trainings Per Week** | 3 times | Cost control and system stability |
| **Drift Threshold** | 0.10 | Maintain 88% SLA accuracy |
| **Warning Threshold** | 0.08 | Early warning buffer (92% accuracy) |
| **Quiet Load Threshold** | 0.3 | Avoid impacting production (< 30% load) |
| **Min Quiet Duration** | 1.5 hours | Ensure training completes successfully |

---

## 4. Continuous Learning System

### 4.1 Data Collection Strategy

#### 4.1.1 Data Buffer Implementation

The inference daemon continuously buffers incoming metrics for daily retraining.

**Current State:**
- Inference daemon receives metrics every 5 seconds
- Data used for predictions then discarded
- No historical accumulation

**Enhanced Implementation:**
```python
class DataBuffer:
    """
    Accumulates incoming metrics for daily retraining.

    Features:
    - Append-only parquet files (1 file per day)
    - Automatic rotation at midnight
    - Retention policy (keep last 60 days)
    - Efficient columnar storage
    """

    def append(self, records: List[Dict]):
        """Append incoming metrics to daily buffer file."""

    def rotate_if_needed(self):
        """Check if new day started, rotate to new file."""

    def get_training_window(self, days=30) -> Path:
        """Return combined parquet with last N days for training."""
```

**Storage Pattern:**
```
data_buffer/
├── metrics_2025-11-14.parquet  (today - actively written)
├── metrics_2025-11-13.parquet  (yesterday)
├── metrics_2025-11-12.parquet
...
└── metrics_2025-10-15.parquet  (30 days ago)
```

---

#### 4.1.2 Sliding Window Management

**Window Size:** 30 days (configurable)

**Why 30 Days?**
- Captures monthly patterns (end-of-month processing, weekly cycles)
- Includes edge cases and incidents for learning
- Balances freshness vs. training stability
- Reasonable storage footprint (~750 MB - 1.5 GB total)

**Retention Policy:**
- Keep last 60 days for retraining flexibility
- Archive older data for analysis
- Delete data older than retention period

---

### 4.2 Daily Training Workflow

#### 4.2.1 Training Schedule

**Daily Training Window:**
- **Time:** 2:00 AM - 4:00 AM (low traffic period)
- **Frequency:** Once per day
- **Duration:** 1-2 hours (1 epoch on 30 days of data)

**Why 2 AM?**
- Minimal production load
- Overnight batch jobs typically finished
- Before pre-market activity (financial systems)
- Consistent quiet window across weekdays

---

#### 4.2.2 Training Workflow

```python
def daily_training_workflow():
    """
    Daily training workflow executed at 2 AM.

    Steps:
    1. Data validation and preparation
    2. Incremental training (1 epoch)
    3. Model validation checks
    4. Checkpoint promotion
    5. Success/failure notification
    """

    # Step 1: Prepare training data
    buffer = DataBuffer('./data_buffer')
    training_data = buffer.get_training_window(days=30)

    if not validate_data(training_data):
        send_alert("Training skipped - insufficient data")
        return False

    # Step 2: Train incrementally (1 epoch)
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

    # Step 4: Promote checkpoint to production
    promote_checkpoint(model_path)

    # Step 5: Success notification
    total_epochs = get_total_epochs(model_path)
    send_notification(f"✅ Training complete: {total_epochs} total epochs")

    return True
```

---

#### 4.2.3 Why 1 Epoch Per Day?

**Benefits:**
- Continuous gradual learning without overfitting
- Lower risk of catastrophic forgetting
- Faster training time (production-friendly)
- After 30 days = 30 total epochs (well-trained model)
- Reduces compute costs vs. full retraining

**Incremental Training:**
- Resumes from last checkpoint
- Accumulates epochs over time
- Model evolves with infrastructure changes

---

### 4.3 Model Validation & Safety

#### 4.3.1 Validation Checks

Before deploying a new checkpoint to production:

```python
def validate_model(model_path: Path) -> bool:
    """
    Validate new model before deployment.

    Checks:
    1. Model loads successfully without errors
    2. Predictions are reasonable (no NaN, within bounds)
    3. Validation loss improved or remained stable
    4. Test predictions match expected ranges
    5. No regressions in per-metric performance
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

    # Check prediction ranges (e.g., CPU 0-100%)
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

**Validation Threshold:** New model must be within 20% of current model's validation loss.

---

#### 4.3.2 Rollback Strategy

If validation fails:
1. Keep current checkpoint active (no deployment)
2. Log failure reason with diagnostics
3. Alert operations team via configured channels
4. Retry tomorrow with additional day of data
5. Investigate root cause if failures persist

**Rollback Triggers:**
- Invalid predictions (NaN, infinite values)
- Predictions outside valid ranges
- Validation loss degraded by >20%
- Model loading errors
- Performance regressions

---

### 4.4 Scheduling Options

#### 4.4.1 Option A: Cron Job (Linux/Mac)

```bash
# /etc/cron.d/tft_training
0 2 * * * cd /path/to/MonitoringPrediction && python continuous_learning.py
```

**Pros:** Simple, reliable, built into OS
**Cons:** Linux/Mac only, less flexible

---

#### 4.4.2 Option B: Windows Task Scheduler

```xml
<!-- Windows Task Scheduler XML -->
<Task>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2025-11-14T02:00:00</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>python</Command>
      <Arguments>continuous_learning.py</Arguments>
      <WorkingDirectory>D:\Vibe_Projects\MonitoringPrediction</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
```

**Pros:** Native Windows integration
**Cons:** Windows only, GUI configuration

---

#### 4.4.3 Option C: Python Scheduler (RECOMMENDED)

```python
# continuous_learning_daemon.py
import schedule
import time
from datetime import datetime

def daily_training_job():
    """Runs at 2 AM daily."""
    print(f"[{datetime.now()}] Starting daily training...")

    # Execute training workflow
    success = daily_training_workflow()

    if success:
        print("Training completed successfully")
    else:
        print("Training failed - check logs")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(daily_training_job)

# Daemon loop
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

**Pros:** Cross-platform, flexible, easy configuration
**Cons:** Requires running daemon process

---

## 5. Implementation Details

### 5.1 Adaptive Retraining Implementation

#### 5.1.1 Drift Monitor Class

```python
class DriftMonitor:
    """
    Monitors data and prediction drift in real-time.

    Metrics tracked:
    1. Prediction Error Rate (PER) - 40% weight
    2. Distribution Shift Score (DSS) - 30% weight
    3. Feature Drift Score (FDS) - 20% weight
    4. Anomaly Rate - 10% weight
    """

    def __init__(self):
        self.baseline_stats = self.load_baseline()
        self.window_size = 1000  # Last 1000 predictions

    def calculate_prediction_error_rate(self, predictions, actuals):
        """
        Calculate MAPE (Mean Absolute Percentage Error).

        Returns: PER value (0.0-1.0)
        """
        errors = []
        for pred, actual in zip(predictions, actuals):
            if actual > 0:
                error = abs(pred - actual) / actual
                errors.append(error)

        return np.mean(errors)

    def calculate_distribution_shift(self, current_data, baseline_data):
        """
        Compare current data distribution to baseline.
        Uses Kolmogorov-Smirnov test.

        Returns: (dss_score, per_metric_details)
        """
        from scipy.stats import ks_2samp

        scores = {}
        for metric in ALL_METRICS:
            stat, p_value = ks_2samp(
                current_data[metric],
                baseline_data[metric]
            )
            scores[metric] = {
                'ks_stat': stat,
                'p_value': p_value,
                'drifted': p_value < 0.05
            }

        # Overall drift score (average KS statistic)
        dss = np.mean([s['ks_stat'] for s in scores.values()])
        return dss, scores

    def calculate_feature_drift(self, recent_window):
        """
        Detect sudden changes in feature statistics.

        Returns: (fds_score, per_metric_details)
        """
        drift_scores = {}

        for metric in ALL_METRICS:
            current_mean = recent_window[metric].mean()
            baseline_mean = self.baseline_stats[metric]['mean']
            baseline_std = self.baseline_stats[metric]['std']

            # Z-score of mean shift
            z_score = abs(current_mean - baseline_mean) / baseline_std
            drift_scores[metric] = z_score

        # Overall FDS (max z-score across metrics)
        fds = max(drift_scores.values())
        return fds, drift_scores

    def calculate_anomaly_rate(self, recent_predictions):
        """
        Track rate of anomalous predictions.

        Returns: Anomaly rate (0.0-1.0)
        """
        anomalies = sum(1 for pred in recent_predictions
                       if self.is_anomalous(pred))
        return anomalies / len(recent_predictions)

    def get_drift_signal(self) -> Dict:
        """
        Aggregate all drift metrics into single signal.

        Returns:
        {
            'drift_detected': True/False,
            'drift_score': 0.0-1.0,
            'confidence': 0.0-1.0,
            'recommendation': 'retrain'|'monitor'|'healthy',
            'metrics': {...}
        }
        """
        per = self.calculate_prediction_error_rate()
        dss, dist_scores = self.calculate_distribution_shift()
        fds, feat_scores = self.calculate_feature_drift()
        anomaly_rate = self.calculate_anomaly_rate()

        # Weighted drift score
        drift_score = (
            per * 0.40 +
            dss * 0.30 +
            fds * 0.20 +
            anomaly_rate * 0.10
        )

        # Decision thresholds
        if drift_score > 0.10:
            recommendation = 'retrain'
            drift_detected = True
            confidence = min(1.0, drift_score / 0.15)
        elif drift_score > 0.08:
            recommendation = 'monitor'
            drift_detected = False
            confidence = drift_score / 0.10
        else:
            recommendation = 'healthy'
            drift_detected = False
            confidence = 1.0 - (drift_score / 0.08)

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'confidence': confidence,
            'recommendation': recommendation,
            'metrics': {
                'prediction_error_rate': per,
                'distribution_shift_score': dss,
                'feature_drift_score': fds,
                'anomaly_rate': anomaly_rate
            },
            'timestamp': datetime.now().isoformat()
        }
```

---

#### 5.1.2 Quiet Time Detector Class

```python
class QuietTimeDetector:
    """
    Identifies optimal windows for retraining.

    Learns patterns from historical load data.
    """

    def __init__(self):
        self.historical_load = self.load_load_history()
        self.min_quiet_duration = 1.5  # Hours

    def calculate_current_load(self) -> float:
        """
        Calculate current infrastructure load.

        Returns: 0.0-1.0 (0=idle, 1=maxed out)
        """
        current_metrics = self.get_current_metrics()

        # Average CPU across all servers
        avg_cpu = np.mean([s['cpu_user_pct'] for s in current_metrics])

        # Average prediction request rate
        pred_rate = self.get_prediction_rate_last_5min()
        baseline_rate = self.historical_load['prediction_rate']['p95']
        rate_ratio = pred_rate / baseline_rate

        # Weighted load score
        load_score = (avg_cpu / 100 * 0.6) + (rate_ratio * 0.4)

        return load_score

    def predict_quiet_window(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Predict next quiet window based on historical patterns.

        Returns: (start_time, end_time) if found, else None
        """
        now = datetime.now()
        predictions = []

        # Check next 24 hours in 15-minute intervals
        for hours_ahead in np.arange(0, 24, 0.25):
            check_time = now + timedelta(hours=hours_ahead)
            predicted_load = self.predict_load_at_time(check_time)
            predictions.append((check_time, predicted_load))

        # Find windows where load < 0.3 for at least 1.5 hours
        quiet_windows = []
        window_start = None

        for i, (check_time, load) in enumerate(predictions):
            if load < 0.3:
                if window_start is None:
                    window_start = check_time
            else:
                if window_start is not None:
                    window_end = check_time
                    duration = (window_end - window_start).total_seconds() / 3600

                    if duration >= self.min_quiet_duration:
                        quiet_windows.append((window_start, window_end))

                    window_start = None

        return quiet_windows[0] if quiet_windows else None

    def is_quiet_now(self) -> bool:
        """Check if current time is quiet enough for training."""
        load = self.calculate_current_load()
        return load < 0.3

    def predict_load_at_time(self, target_time: datetime) -> float:
        """
        Predict infrastructure load at future time.

        Uses historical patterns by hour and weekday.
        """
        hour = target_time.hour
        weekday = target_time.weekday()

        # Get historical load for this hour/weekday
        key = f'{weekday}_{hour}'
        historical_avg = self.historical_load[key]['mean']
        historical_std = self.historical_load[key]['std']

        # Add uncertainty
        predicted_load = np.random.normal(historical_avg, historical_std * 0.5)

        return np.clip(predicted_load, 0, 1)
```

---

#### 5.1.3 Retraining Decision Engine

```python
class RetrainingDecisionEngine:
    """
    Decides when to trigger retraining.

    Inputs:
    - Drift signal (from DriftMonitor)
    - Quiet time status (from QuietTimeDetector)
    - Training history (last training time, frequency)

    Safeguards:
    - Min 6 hours between trainings
    - Max 30 days without training
    - Max 3 trainings per week
    """

    def __init__(self):
        self.min_hours_between_training = 6
        self.max_days_without_training = 30
        self.max_trainings_per_week = 3

        self.last_training_time = self.load_last_training_time()
        self.training_history = self.load_training_history()

    def should_trigger_training(self) -> Dict:
        """
        Main decision function.

        Returns decision with reason, confidence, and optional wait time.
        """
        now = datetime.now()

        # === SAFEGUARD CHECKS ===

        # Check 1: Minimum time between trainings
        hours_since_last = (now - self.last_training_time).total_seconds() / 3600
        if hours_since_last < self.min_hours_between_training:
            return {
                'trigger': False,
                'reason': f'Too soon (last training {hours_since_last:.1f}h ago)',
                'confidence': 0.0,
                'wait_until': self.last_training_time +
                             timedelta(hours=self.min_hours_between_training)
            }

        # Check 2: Maximum time without training (force refresh)
        days_since_last = hours_since_last / 24
        if days_since_last >= self.max_days_without_training:
            return {
                'trigger': True,
                'reason': f'Forced refresh ({days_since_last:.0f} days since last training)',
                'confidence': 1.0,
                'priority': 'high'
            }

        # Check 3: Weekly training limit
        trainings_this_week = self.count_trainings_last_7_days()
        if trainings_this_week >= self.max_trainings_per_week:
            return {
                'trigger': False,
                'reason': f'Weekly limit reached ({trainings_this_week}/{self.max_trainings_per_week})',
                'confidence': 0.0,
                'wait_until': self.get_next_week_start()
            }

        # === DRIFT CHECK ===

        drift_monitor = DriftMonitor()
        drift_signal = drift_monitor.get_drift_signal()

        if not drift_signal['drift_detected']:
            return {
                'trigger': False,
                'reason': 'No drift detected',
                'confidence': 1.0 - drift_signal['drift_score'],
                'drift_metrics': drift_signal['metrics']
            }

        # === QUIET TIME CHECK ===

        quiet_detector = QuietTimeDetector()

        if quiet_detector.is_quiet_now():
            return {
                'trigger': True,
                'reason': f'Drift detected + quiet time (load={quiet_detector.calculate_current_load():.2f})',
                'confidence': drift_signal['confidence'],
                'drift_metrics': drift_signal['metrics']
            }
        else:
            next_quiet = quiet_detector.predict_quiet_window()

            if next_quiet:
                start_time, end_time = next_quiet
                return {
                    'trigger': False,
                    'reason': 'Drift detected, waiting for quiet window',
                    'confidence': drift_signal['confidence'],
                    'wait_until': start_time,
                    'drift_metrics': drift_signal['metrics']
                }
            else:
                # No quiet window in next 24h - train anyway if SLA breach
                if drift_signal['drift_score'] > 0.12:
                    return {
                        'trigger': True,
                        'reason': 'SLA breach (< 88% accuracy), training despite load',
                        'confidence': drift_signal['confidence'],
                        'priority': 'critical'
                    }
                else:
                    return {
                        'trigger': False,
                        'reason': 'Moderate drift, waiting for better window',
                        'confidence': drift_signal['confidence']
                    }
```

---

#### 5.1.4 Adaptive Retraining Daemon

```python
class AdaptiveRetrainingDaemon:
    """
    Continuously monitors and triggers retraining when appropriate.

    Runs as background service.
    """

    def __init__(self):
        self.drift_monitor = DriftMonitor()
        self.quiet_detector = QuietTimeDetector()
        self.decision_engine = RetrainingDecisionEngine()
        self.running = True

    def run(self):
        """Main daemon loop - checks every 5 minutes."""
        print("[START] Adaptive Retraining Daemon started")

        while self.running:
            try:
                decision = self.decision_engine.should_trigger_training()
                self.log_decision(decision)

                if decision['trigger']:
                    print(f"\n[TRIGGER] Training triggered: {decision['reason']}")
                    self.execute_training(decision)
                else:
                    print(f"[SKIP] {decision['reason']}")

                time.sleep(300)  # Sleep 5 minutes

            except Exception as e:
                print(f"[ERROR] Daemon error: {e}")
                time.sleep(60)

    def execute_training(self, decision: Dict):
        """Execute training workflow."""
        try:
            print("[TRAIN] Starting incremental training...")

            buffer = DataBuffer('./data_buffer')
            training_data = buffer.get_training_window(days=30)

            model_path = train_model(
                dataset_path=training_data,
                epochs=1,
                incremental=True
            )

            if model_path:
                print(f"[SUCCESS] Training complete: {model_path}")

                self.decision_engine.last_training_time = datetime.now()
                self.decision_engine.save_training_history()
                self.drift_monitor.update_baseline()

                self.send_notification(
                    f"✅ Adaptive retraining completed\n"
                    f"Reason: {decision['reason']}\n"
                    f"Confidence: {decision['confidence']:.2f}"
                )
            else:
                print("[ERROR] Training failed")
                self.send_alert("❌ Adaptive retraining failed - check logs")

        except Exception as e:
            print(f"[ERROR] Training execution failed: {e}")
            self.send_alert(f"❌ Adaptive retraining error: {e}")

    def log_decision(self, decision: Dict):
        """Log decision for monitoring and analysis."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'drift_metrics': self.drift_monitor.get_drift_signal(),
            'current_load': self.quiet_detector.calculate_current_load()
        }

        with open('./logs/retraining_decisions.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

---

### 5.2 Continuous Learning Implementation

#### 5.2.1 Data Buffer Class

```python
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
        self.buffer_dir.mkdir(exist_ok=True)
        self.current_date = datetime.now().date()
        self.current_file = None
        self.retention_days = 60

    def append(self, records: List[Dict]):
        """Append incoming metrics to daily buffer file."""
        self.rotate_if_needed()

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Append to current file
        file_path = self.get_file_path(self.current_date)

        if file_path.exists():
            # Append to existing file
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_parquet(file_path, index=False)

    def rotate_if_needed(self):
        """Check if new day, rotate to new file."""
        today = datetime.now().date()

        if today != self.current_date:
            print(f"[ROTATE] New day detected: {today}")
            self.current_date = today
            self.cleanup_old_files()

    def cleanup_old_files(self):
        """Delete files older than retention period."""
        cutoff_date = datetime.now().date() - timedelta(days=self.retention_days)

        for file_path in self.buffer_dir.glob('metrics_*.parquet'):
            file_date_str = file_path.stem.replace('metrics_', '')
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()

            if file_date < cutoff_date:
                print(f"[CLEANUP] Deleting old file: {file_path}")
                file_path.unlink()

    def get_training_window(self, days=30) -> Path:
        """Return combined parquet with last N days for training."""
        start_date = datetime.now().date() - timedelta(days=days)
        dfs = []

        for file_path in sorted(self.buffer_dir.glob('metrics_*.parquet')):
            file_date_str = file_path.stem.replace('metrics_', '')
            file_date = datetime.strptime(file_date_str, '%Y-%m-%d').date()

            if file_date >= start_date:
                df = pd.read_parquet(file_path)
                dfs.append(df)

        if not dfs:
            raise ValueError(f"No data found for last {days} days")

        # Combine and save to temp file
        combined_df = pd.concat(dfs, ignore_index=True)
        temp_path = self.buffer_dir / f'training_window_{days}days.parquet'
        combined_df.to_parquet(temp_path, index=False)

        return temp_path

    def get_file_path(self, date) -> Path:
        """Get file path for specific date."""
        return self.buffer_dir / f'metrics_{date.strftime("%Y-%m-%d")}.parquet'
```

---

#### 5.2.2 Daily Training Script

```python
# continuous_learning.py

def daily_training_workflow():
    """
    Daily training workflow executed at 2 AM.

    Returns: True if successful, False otherwise
    """

    try:
        # Step 1: Prepare data
        print(f"[{datetime.now()}] Starting daily training workflow...")

        buffer = DataBuffer('./data_buffer')
        training_data = buffer.get_training_window(days=30)

        # Validate data quality
        if not validate_data(training_data):
            send_alert("Training skipped - insufficient or invalid data")
            return False

        print(f"[DATA] Training window prepared: {training_data}")

        # Step 2: Train incrementally (1 epoch)
        print("[TRAIN] Starting incremental training (1 epoch)...")

        start_time = time.time()

        model_path = train_model(
            dataset_path=training_data,
            epochs=1,
            incremental=True,
            checkpoint_path='./checkpoints/last.ckpt'
        )

        training_duration = time.time() - start_time

        if not model_path:
            send_alert("Training failed - check logs for details")
            return False

        print(f"[TRAIN] Training completed in {training_duration/60:.1f} minutes")

        # Step 3: Validate new model
        print("[VALIDATE] Validating new model...")

        if not validate_model(model_path):
            send_alert("Model validation failed - rolling back to previous checkpoint")
            return False

        print("[VALIDATE] Model validation passed")

        # Step 4: Promote checkpoint to production
        print("[DEPLOY] Promoting checkpoint to production...")
        promote_checkpoint(model_path)

        # Step 5: Success notification
        total_epochs = get_total_epochs(model_path)

        send_notification(
            f"✅ Daily training completed successfully\n"
            f"Total epochs: {total_epochs}\n"
            f"Training duration: {training_duration/60:.1f} minutes\n"
            f"Model path: {model_path}"
        )

        print(f"[SUCCESS] Daily training workflow completed")
        return True

    except Exception as e:
        error_msg = f"Daily training workflow failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        send_alert(f"❌ {error_msg}")
        return False


def validate_data(training_data_path: Path) -> bool:
    """Validate training data quality."""
    try:
        df = pd.read_parquet(training_data_path)

        # Check minimum samples
        if len(df) < 100000:
            print(f"[WARNING] Insufficient samples: {len(df)}")
            return False

        # Check for NaN values
        if df.isnull().any().any():
            print("[WARNING] Data contains NaN values")
            return False

        # Check date range
        date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        if date_range < 28:
            print(f"[WARNING] Date range too short: {date_range} days")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Data validation failed: {e}")
        return False


def promote_checkpoint(model_path: Path):
    """Promote new checkpoint to production."""
    production_checkpoint = Path('./checkpoints/last.ckpt')
    backup_checkpoint = Path('./checkpoints/last_backup.ckpt')

    # Backup current checkpoint
    if production_checkpoint.exists():
        shutil.copy(production_checkpoint, backup_checkpoint)

    # Copy new checkpoint to production
    shutil.copy(model_path, production_checkpoint)

    print(f"[DEPLOY] Checkpoint promoted to production")


def get_total_epochs(model_path: Path) -> int:
    """Get total epochs from checkpoint."""
    checkpoint = torch.load(model_path)
    return checkpoint.get('epoch', 0)


if __name__ == '__main__':
    daily_training_workflow()
```

---

#### 5.2.3 Scheduler Daemon

```python
# continuous_learning_daemon.py

import schedule
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/continuous_learning_daemon.log'),
        logging.StreamHandler()
    ]
)

def daily_training_job():
    """Job that runs at 2 AM daily."""
    logging.info("="*60)
    logging.info("DAILY TRAINING JOB STARTED")
    logging.info("="*60)

    try:
        success = daily_training_workflow()

        if success:
            logging.info("Daily training job completed successfully")
        else:
            logging.error("Daily training job failed")

    except Exception as e:
        logging.error(f"Daily training job crashed: {e}", exc_info=True)

    logging.info("="*60)


# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(daily_training_job)

logging.info("Continuous Learning Daemon started")
logging.info("Scheduled: Daily at 02:00 AM")

# Daemon loop
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
```

---

### 5.3 Configuration Reference

#### 5.3.1 Adaptive Retraining Config

```python
# config/adaptive_retraining_config.py

ADAPTIVE_RETRAINING_CONFIG = {
    # Drift Detection (Aligned with 88% Accuracy SLA)
    'drift_check_interval_mins': 5,
    'drift_threshold_critical': 0.10,    # 88% accuracy - trigger retraining
    'drift_threshold_warning': 0.08,     # 92% accuracy - start monitoring
    'prediction_window_size': 1000,      # Samples to track for drift

    # Quiet Time Detection
    'quiet_load_threshold': 0.3,         # < 30% load considered quiet
    'min_quiet_duration_hours': 1.5,     # Minimum quiet window duration
    'load_prediction_lookback_days': 30,

    # Training Safeguards
    'min_hours_between_training': 6,     # Prevent training thrashing
    'max_days_without_training': 30,     # Force periodic refresh
    'max_trainings_per_week': 3,         # Cost control

    # Training Parameters
    'training_window_days': 30,
    'epochs_per_training': 1,            # Incremental training

    # Monitoring
    'log_dir': './logs/adaptive_retraining',
    'alert_email': 'ml-ops@company.com',
    'drift_metrics_retention_days': 90
}
```

---

#### 5.3.2 Continuous Learning Config

```python
# config/continuous_learning_config.py

CONTINUOUS_LEARNING_CONFIG = {
    # Data Buffer
    'data_buffer_dir': './data_buffer',
    'data_retention_days': 60,           # Keep 60 days of data
    'training_window_days': 30,          # Train on 30 days

    # Training Schedule
    'training_time': '02:00',            # 2 AM daily
    'training_timezone': 'America/New_York',
    'epochs_per_day': 1,                 # Incremental: 1 epoch/day

    # Validation
    'validation_loss_threshold': 1.2,    # Max 20% worse than current
    'min_training_samples': 100000,      # Minimum samples required

    # Monitoring
    'alert_email': 'ml-ops@company.com',
    'slack_webhook': 'https://hooks.slack.com/...',
    'log_dir': './logs/continuous_learning',

    # Safety
    'enable_auto_deploy': True,          # Auto-promote validated checkpoints
    'require_validation': True,          # Always validate before deploy
    'max_training_duration_hours': 4     # Alert if training takes too long
}
```

---

### 5.4 File Structure

```
MonitoringPrediction/
├── config/
│   ├── adaptive_retraining_config.py
│   └── continuous_learning_config.py
│
├── data_buffer/                    # NEW - Daily metrics for retraining
│   ├── metrics_2025-11-14.parquet
│   ├── metrics_2025-11-13.parquet
│   └── ...
│
├── checkpoints/                    # Model checkpoints
│   ├── last.ckpt                   # Current production checkpoint
│   └── last_backup.ckpt            # Backup checkpoint
│
├── models/                         # Trained models
│   ├── tft_model_20251114/
│   └── ...
│
├── logs/
│   ├── adaptive_retraining/
│   │   ├── retraining_decisions.jsonl
│   │   └── training_2025-11-14.log
│   ├── continuous_learning/
│   │   ├── training_2025-11-14.log
│   │   └── metrics.json
│   └── continuous_learning_daemon.log
│
├── Docs/
│   └── TRAINING_GUIDE.md           # This document
│
├── adaptive_retraining_daemon.py   # NEW - Adaptive retraining daemon
├── continuous_learning.py          # NEW - Daily training workflow
├── continuous_learning_daemon.py   # NEW - Scheduler daemon
├── tft_trainer.py                  # Existing trainer
├── tft_inference_daemon.py         # Existing inference daemon
└── metrics_generator.py            # Existing data generator
```

---

## 6. Monitoring & Operations

### 6.1 Monitoring Dashboard

#### 6.1.1 Real-Time Metrics

```python
dashboard_metrics = {
    # Current Status
    'current_drift_score': 0.09,
    'current_accuracy': '91%',
    'sla_status': 'Above SLA (target: 88%)',
    'buffer_above_sla': '+3%',

    # Training Status
    'days_since_last_training': 5,
    'trainings_this_week': 1,
    'next_scheduled_training': '2025-11-15 02:00:00',

    # System Status
    'current_load': 0.28,
    'is_quiet': True,
    'data_buffer_size_gb': 1.2,
    'checkpoint_size_mb': 340,

    # Last 7 Days
    'last_7_days': {
        'trainings_triggered': 2,
        'trainings_skipped': 15,
        'avg_drift_score': 0.08,
        'avg_accuracy': '92%',
        'sla_breaches': 0
    },

    # Drift Breakdown
    'drift_breakdown': {
        'prediction_error_rate': 0.09,
        'distribution_shift': 0.07,
        'feature_drift': 0.08,
        'anomaly_rate': 0.02
    }
}
```

---

#### 6.1.2 Alert Levels

| Alert Level | Condition | Action |
|------------|-----------|--------|
| **Healthy** | drift < 0.08 (> 92% accuracy) | Monitor normally |
| **Warning** | drift 0.08-0.10 (88-92% accuracy) | Increase monitoring, prepare for retraining |
| **Critical** | drift > 0.10 (< 88% accuracy) | Schedule immediate retraining |
| **Emergency** | drift > 0.12 + no quiet window | Train despite load (SLA breach) |

---

### 6.2 Alert Conditions

#### 6.2.1 Critical Alerts (Page On-Call)

**Conditions:**
- Training failed 2 days in a row
- Model validation failed after training
- Data buffer corruption detected
- Prediction accuracy dropped >10% suddenly
- SLA breach (< 88% accuracy) persists for 6+ hours

**Notification Channels:**
- PagerDuty/on-call system
- SMS to ops team
- Slack critical channel

---

#### 6.2.2 Warning Alerts (Email/Slack)

**Conditions:**
- Training took >3 hours (expected: 1-2 hours)
- Validation loss increased compared to previous checkpoint
- Disk space low (<20% remaining)
- Missing data for some servers (>10% gap)
- Drift score in warning range (0.08-0.10)

**Notification Channels:**
- Email to ML ops team
- Slack monitoring channel

---

#### 6.2.3 Info Alerts (Slack Only)

**Conditions:**
- Training completed successfully
- New checkpoint promoted to production
- Weekly training summary
- Data buffer rotation occurred

---

### 6.3 Success Metrics

#### 6.3.1 Adaptive Retraining KPIs

1. **Training Efficiency**: 60% fewer trainings than fixed schedule
2. **Drift Response Time**: Training triggered within 4 hours of drift detection
3. **Production Impact**: Zero trainings during high-load periods
4. **Model Freshness**: Prediction error stays <10% (>90% accuracy)
5. **Cost Savings**: 60% reduction in training compute costs

**Target Values:**
- Trainings per month: 8-12 (vs. 30 with daily schedule)
- Average time to retraining after drift: < 4 hours
- Trainings during load >50%: 0
- Uptime of adaptive daemon: >99.9%

---

#### 6.3.2 Continuous Learning KPIs

1. **Training Success Rate**: >95% daily completion
2. **Model Freshness**: <24 hours lag maximum
3. **Prediction Accuracy**: Improve 5% month-over-month
4. **Automation Rate**: 100% (zero manual intervention)
5. **Training Duration**: <2 hours per day average

**Target Values:**
- Daily training completion: >28/30 days per month
- Training failures: <2 per month
- Validation pass rate: >98%
- Average training time: 1-2 hours
- Total epochs after 30 days: 30

---

## 7. Troubleshooting

### 7.1 Common Issues

#### 7.1.1 Training Failure Issues

**Issue:** Training fails with OOM (Out of Memory) error

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. Reduce batch size in config
2. Enable gradient accumulation
3. Use mixed precision training (bf16-mixed)
4. Clear GPU cache before training
5. Check for memory leaks in data loading

**Prevention:**
- Monitor GPU memory usage during training
- Set appropriate batch size for your hardware
- Enable automatic batch size finder

---

**Issue:** Training hangs or freezes

**Symptoms:**
- No progress for >30 minutes
- GPU utilization drops to 0%
- Process still running but no output

**Solutions:**
1. Check data loading pipeline (likely bottleneck)
2. Verify data buffer files are not corrupted
3. Kill process and restart with verbose logging
4. Check disk I/O (slow disk can cause hangs)

**Prevention:**
- Add timeout to training script (max 4 hours)
- Monitor training progress with callbacks
- Use profiler to identify bottlenecks

---

**Issue:** Validation loss suddenly increases

**Symptoms:**
- New model validation loss >20% worse
- Predictions become erratic
- Model validation fails

**Solutions:**
1. Check training data for corruption or anomalies
2. Verify learning rate schedule (may be too high)
3. Roll back to previous checkpoint
4. Investigate data distribution shift
5. Check for overfitting (train loss vs. val loss)

**Prevention:**
- Validate data quality before training
- Monitor train/val loss gap during training
- Use early stopping to prevent overfitting
- Keep backup of last known good checkpoint

---

#### 7.1.2 Drift Detection Issues

**Issue:** False positive drift alerts

**Symptoms:**
- Drift detected but predictions look fine
- Drift score fluctuates wildly
- Excessive retraining triggers

**Solutions:**
1. Adjust drift thresholds (increase from 0.10 to 0.12)
2. Increase prediction window size (1000 → 2000)
3. Add smoothing to drift score calculation
4. Check for anomalies in recent data

**Prevention:**
- Tune thresholds based on historical patterns
- Use longer windows for more stability
- Add drift score moving average

---

**Issue:** Drift not detected when it should be

**Symptoms:**
- Prediction accuracy degrading but no alert
- Model clearly stale but drift score low
- Manual inspection shows distribution shift

**Solutions:**
1. Decrease drift thresholds (lower from 0.10 to 0.08)
2. Check if baseline statistics are outdated
3. Verify all drift metrics are being calculated
4. Update baseline after successful training

**Prevention:**
- Regularly update baseline statistics
- Monitor all four drift metrics separately
- Set up secondary alert based on user feedback

---

#### 7.1.3 Quiet Time Detection Issues

**Issue:** No quiet windows detected

**Symptoms:**
- System never finds quiet time to train
- All predicted windows show high load
- Training delayed indefinitely

**Solutions:**
1. Lower quiet threshold (0.3 → 0.4)
2. Reduce minimum quiet duration (1.5h → 1.0h)
3. Override and train during best available window
4. Schedule training at fixed low-traffic time

**Prevention:**
- Collect longer historical load data (60 days)
- Consider weekend training windows
- Add manual override option for urgent retraining

---

**Issue:** Training starts during busy period

**Symptoms:**
- Training triggered when load >50%
- Production impact reported
- Prediction latency increased during training

**Solutions:**
1. Increase quiet threshold (0.3 → 0.25)
2. Add real-time load monitoring during training
3. Implement training pause if load exceeds threshold
4. Review load prediction accuracy

**Prevention:**
- Validate quiet time prediction accuracy
- Add safety margin to load threshold
- Monitor training impact on production metrics

---

#### 7.1.4 Data Buffer Issues

**Issue:** Data buffer files corrupted

**Symptoms:**
```
pyarrow.lib.ArrowInvalid: Parquet file is invalid
```

**Solutions:**
1. Delete corrupted file and regenerate from backup
2. Check disk space (corruption often due to disk full)
3. Verify write permissions
4. Recreate buffer directory

**Prevention:**
- Monitor disk space continuously
- Implement data buffer checksums
- Regular backup of buffer files
- Add data validation on write

---

**Issue:** Missing data in buffer

**Symptoms:**
- Data gaps in training window
- Fewer samples than expected
- Some servers missing from recent data

**Solutions:**
1. Check inference daemon status (may have crashed)
2. Verify data append is working correctly
3. Check for date/time synchronization issues
4. Investigate network issues if remote data source

**Prevention:**
- Monitor data ingestion rate
- Alert on data gaps >5 minutes
- Add redundancy to data collection
- Implement data quality checks

---

### 7.2 FAQ

#### Q: Why not train for 50 or 100 epochs?

**A:** Diminishing returns and overfitting risk. TFT models typically converge by epoch 20. Beyond that, you risk memorizing training data rather than learning generalizable patterns. The validation loss will plateau or start increasing (overfitting).

---

#### Q: Can I use less than 1 week of data?

**A:** Not recommended. TFT needs at least 7 days to learn weekly patterns (weekday vs. weekend behavior). 30 days is ideal for capturing monthly cycles and having sufficient samples for robust training.

---

#### Q: How do I know if my model converged?

**A:** Check if validation loss plateaus. If val_loss stops improving for 3+ consecutive epochs, you've likely converged. Early stopping will trigger automatically. A converged model shows stable loss and consistent predictions.

---

#### Q: What if val_loss is much higher than train_loss?

**A:** This indicates overfitting - the model is memorizing training data rather than learning patterns.

**Solutions:**
- Increase dropout rate (default: 0.15 → 0.25)
- Reduce model complexity (fewer hidden units)
- Add more training data (increase from 30 to 60 days)
- Use regularization (L1/L2)
- Reduce training epochs

---

#### Q: Can I claim accuracy from 1 epoch training?

**A:** No. 1 epoch is nowhere near convergence. You can only claim "proof of concept" or "initial validation." Any accuracy claims require at least 10 epochs for validation or 20 epochs for production.

---

#### Q: What's a good quantile loss value?

**A:** Depends on your data scale, but general guidelines:

- **< 0.10**: Excellent - production ready
- **0.10-0.20**: Good - acceptable for most use cases
- **0.20-0.30**: Acceptable - room for improvement
- **> 0.30**: Needs improvement - investigate data quality

---

#### Q: How often should I retrain the model?

**A:** It depends on your deployment strategy:

- **Adaptive Retraining**: Train when drift detected (8-12 times/month)
- **Continuous Learning**: Train daily at 2 AM (30 times/month)
- **Manual**: Train monthly or when performance degrades

**Recommendation:** Start with adaptive retraining for cost efficiency.

---

#### Q: What if I have more than 160 servers?

**A:** For large-scale deployments:

1. Use gradient accumulation to manage memory
2. Enable distributed training across multiple GPUs
3. Increase batch size proportionally
4. Consider model sharding for very large scales (1000+ servers)
5. Expect longer training times (scale linearly with data size)

---

#### Q: Can I train on CPU instead of GPU?

**A:** Technically yes, but **not recommended**.

- GPU training: 1-2 hours per epoch
- CPU training: 20-30 hours per epoch (10-15x slower)

For production use, GPU is essential. Minimum recommended: RTX 3060 or better.

---

#### Q: How do I handle new server types?

**A:** The model uses profile-based transfer learning:

1. New servers get hash-based encoding initially
2. Predictions based on similar existing profiles
3. After 7 days of data, create dedicated profile
4. Retrain to incorporate new profile
5. Model adapts automatically via continuous learning

---

### 7.3 Rollback Procedures

#### 7.3.1 Emergency Rollback

**When to rollback:**
- New model producing invalid predictions
- Validation loss increased significantly
- Production incidents correlated with deployment
- Critical bugs discovered in new checkpoint

**Immediate Rollback Steps:**

```bash
# 1. Stop inference daemon
systemctl stop tft_inference_daemon

# 2. Restore backup checkpoint
cp checkpoints/last_backup.ckpt checkpoints/last.ckpt

# 3. Restart inference daemon
systemctl start tft_inference_daemon

# 4. Verify predictions
python test_predictions.py

# 5. Alert team
python send_alert.py "Emergency rollback completed - investigating root cause"
```

**Rollback Verification:**
1. Check inference daemon logs for errors
2. Verify predictions are being generated
3. Compare prediction quality to baseline
4. Monitor for 30 minutes before declaring success

---

#### 7.3.2 Scheduled Rollback

**When to use:**
- Planned rollback for testing
- Reverting to specific older checkpoint
- Investigation of performance regression

**Scheduled Rollback Steps:**

```bash
# 1. Identify target checkpoint
ls -lh checkpoints/

# 2. Stop training and inference daemons
systemctl stop adaptive_retraining_daemon
systemctl stop continuous_learning_daemon
systemctl stop tft_inference_daemon

# 3. Backup current checkpoint
cp checkpoints/last.ckpt checkpoints/last_prerollback_$(date +%Y%m%d).ckpt

# 4. Restore target checkpoint
cp checkpoints/tft_model_20251110/checkpoint.ckpt checkpoints/last.ckpt

# 5. Update training history
python update_training_history.py --rollback-to 20251110

# 6. Restart daemons
systemctl start tft_inference_daemon
systemctl start continuous_learning_daemon
systemctl start adaptive_retraining_daemon

# 7. Monitor and verify
python verify_rollback.py
```

---

#### 7.3.3 Rollback Testing

Before performing emergency rollback in production, test in staging:

```python
# test_rollback.py

def test_rollback_procedure():
    """Test rollback without affecting production."""

    # 1. Create test environment
    test_env = setup_test_environment()

    # 2. Deploy "bad" checkpoint
    deploy_checkpoint(test_env, 'checkpoints/test_bad.ckpt')

    # 3. Verify degradation
    metrics_before = measure_prediction_quality(test_env)

    # 4. Execute rollback
    rollback(test_env, 'checkpoints/last_backup.ckpt')

    # 5. Verify recovery
    metrics_after = measure_prediction_quality(test_env)

    # 6. Validate rollback success
    assert metrics_after['accuracy'] > metrics_before['accuracy']
    assert metrics_after['latency'] < metrics_before['latency'] * 1.1

    print("Rollback procedure validated successfully")
```

---

## 8. Appendix

### 8.1 Example Scenarios

#### 8.1.1 Scenario 1: Gradual Drift (Ideal Case)

```
Day 1: Drift=0.05, Load=0.25 → No training (healthy, 95% accuracy)
       Model performing well, no action needed

Day 3: Drift=0.07, Load=0.28 → No training (healthy, 93% accuracy)
       Still above warning threshold, continue monitoring

Day 5: Drift=0.09, Load=0.27 → No training (warning, 91% accuracy)
       Approaching SLA threshold, monitoring increased

Day 7: Drift=0.11, Load=0.22 → TRAINING TRIGGERED ✅
       Drift crossed threshold (89% accuracy, below 88% SLA)
       Quiet time detected, training initiated

Result: Training triggered proactively before SLA breach becomes critical
```

**Lessons:**
- System detected drift before major problems
- Training triggered during quiet period
- SLA maintained throughout process

---

#### 8.1.2 Scenario 2: Sudden Infrastructure Change

```
12:00 PM: New servers added to infrastructure
          Drift=0.13, Load=0.75 → No training (too busy)
          87% accuracy (below SLA) but infrastructure busy
          Decision: Wait for quiet window

2:00 PM:  Drift=0.14, Load=0.70 → No training (still busy)
          86% accuracy (critical) but load too high
          Decision: Continue waiting, monitoring closely

6:00 PM:  Drift=0.14, Load=0.42 → No training (moderate load)
          Still waiting for optimal window

10:00 PM: Drift=0.14, Load=0.28 → TRAINING TRIGGERED ✅
          Drift critical + quiet window finally found
          Training initiated to learn new server profiles

Result: Training delayed until quiet period, but triggered same day
```

**Lessons:**
- System prioritized production stability
- Training occurred during first available quiet window
- Model adapted to infrastructure changes within 10 hours

---

#### 8.1.3 Scenario 3: Training Throttle Protection

```
Monday 2 AM:    Drift detected → Training #1 ✅
                (Weekly drift from weekend pattern shift)

Wednesday 3 AM: Drift detected → Training #2 ✅
                (New application deployment changed patterns)

Friday 1 AM:    Drift detected → Training #3 ✅
                (End-of-week processing spike)

Saturday 2 AM:  Drift detected → BLOCKED ❌
                Reason: 3 trainings this week already
                Decision: Wait until Monday (weekly limit)

Result: Weekly limit prevented excessive training costs
```

**Lessons:**
- Safeguards prevent training thrashing
- Cost control maintained even with multiple drift events
- System balances accuracy needs with operational costs

---

#### 8.1.4 Scenario 4: Forced Refresh

```
Day 1:  Model trained successfully
        Drift=0.05, Accuracy=95%

Day 10: Drift=0.06, Load=0.22 → No training (healthy, 94% accuracy)
        Model still performing well

Day 20: Drift=0.07, Load=0.25 → No training (healthy, 93% accuracy)
        Slight drift but still above SLA

Day 30: Drift=0.07, Load=0.28 → TRAINING TRIGGERED ✅
        Reason: 30 days elapsed, forced refresh
        Even though accuracy above SLA, periodic refresh executed

Result: Periodic refresh ensures model doesn't become stale
```

**Lessons:**
- Max 30-day rule enforces periodic updates
- Prevents gradual staleness accumulation
- Proactive refresh before problems emerge

---

### 8.2 Production Deployment Checklist

Before making production accuracy claims:

**Data Preparation:**
- [ ] Collected at least 30 days of historical metrics
- [ ] Data validated for completeness (no major gaps)
- [ ] All server types represented in training data
- [ ] Data distribution checked for anomalies

**Model Training:**
- [ ] Trained model with 20 epochs minimum
- [ ] Monitored convergence (validation loss plateaued)
- [ ] Recorded final train/validation loss values
- [ ] Saved training metrics for documentation
- [ ] Training completed without errors or interruptions

**Model Validation:**
- [ ] Generated prediction samples vs. actuals
- [ ] Calculated quantile loss metrics (P10, P50, P90)
- [ ] Documented convergence behavior
- [ ] Tested on held-out validation set
- [ ] Validated profile-based predictions
- [ ] Tested unknown server handling
- [ ] No NaN or invalid predictions detected

**Performance Testing:**
- [ ] Measured inference latency (<3 seconds target)
- [ ] Tested concurrent prediction load
- [ ] Verified prediction quality across all metrics (CPU, memory, disk)
- [ ] Tested 8-hour prediction horizon
- [ ] Validated confidence intervals calibration

**Production Readiness:**
- [ ] Checkpoint saved and backed up
- [ ] Model metadata documented (training_info.json)
- [ ] Inference daemon tested with new model
- [ ] Rollback procedure tested
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds configured
- [ ] Documentation updated with actual metrics

**Operations:**
- [ ] Runbook created for common issues
- [ ] On-call team trained on system
- [ ] Escalation procedures documented
- [ ] Backup and disaster recovery tested

---

### 8.3 Benefits & ROI Analysis

#### 8.3.1 Adaptive Retraining vs. Fixed Schedule

| Metric | Fixed Schedule (2 AM Daily) | Adaptive (Drift-Based) | Improvement |
|--------|----------------------------|------------------------|-------------|
| **Trainings per Month** | 30 | 8-12 | 60% reduction |
| **GPU Cost per Month** | $1,500 | $600 | $900 savings |
| **Training During Incidents** | Possible | Never | 100% safer |
| **Unnecessary Trainings** | Many | None | Efficient |
| **Drift Response Time** | Up to 24h | <4h | 83% faster |

**Cost Savings:** $900/month × 12 months = $10,800/year

**Additional Benefits:**
- Lower infrastructure impact (trains only during quiet times)
- Faster response to actual model degradation
- Better SLA maintenance (proactive drift detection)

---

#### 8.3.2 Continuous Learning ROI

**Before (Manual Retraining):**
- Retrain frequency: Monthly (30-day lag in learning new patterns)
- Engineer time: 4-6 hours per retraining
- Monthly cost: ~$500 in engineer time
- Risk: Model staleness between manual retrains

**After (Continuous Learning):**
- Retrain frequency: Daily (1-day maximum lag)
- Engineer time: Zero (fully automated)
- Monthly cost: ~$50 in compute (2 hours/day GPU at $0.80/hour)
- Risk: Minimal (continuous adaptation)

**ROI Calculation:**
- Cost reduction: $500 - $50 = $450/month saved
- Annual savings: $5,400/year
- Engineering time freed: 48-72 hours/year
- Model freshness: 30x improvement (1 day vs 30 day lag)

**Intangible Benefits:**
- Model adapts to seasonal patterns automatically
- Learns from recent incidents immediately
- Catches infrastructure changes quickly
- Reduces false positives as system evolves
- Improves prediction accuracy continuously

---

#### 8.3.3 Combined Strategy ROI

**Recommendation:** Use both systems together

- **Continuous Learning**: Baseline daily incremental training
- **Adaptive Retraining**: Emergency response to sudden drift

**Benefits:**
- Daily learning keeps model current (1-day lag)
- Adaptive system handles emergencies (sudden infrastructure changes)
- Safeguards prevent excessive training
- Optimal balance of cost and freshness

**Total Annual Savings:** $10,800 + $5,400 = $16,200/year

**Engineering Time Freed:** ~150 hours/year

---

## Conclusion

This comprehensive training guide provides everything needed to successfully train, deploy, and maintain the TFT monitoring prediction model. Key takeaways:

**Training Philosophy:**
- Start with appropriate training level for your use case (demo, validation, production)
- Only make accuracy claims backed by measured validation metrics
- Focus on proven differentiators when accuracy data is incomplete

**Retraining Strategy:**
- Use adaptive retraining for cost-efficient drift response
- Implement continuous learning for always-current models
- Leverage safeguards to prevent training thrashing
- Monitor drift proactively to maintain SLA

**Operations:**
- Validate models thoroughly before production deployment
- Implement comprehensive monitoring and alerting
- Test rollback procedures before emergencies
- Document all decisions and configurations

**Success Metrics:**
- Target 85-90% accuracy based on benchmarks
- Maintain 88% SLA threshold minimum
- Keep drift score <0.10 for healthy operations
- Respond to drift within 4 hours

By following these guidelines, you'll achieve production-ready model performance while maintaining operational efficiency and cost control.

---

**Built by Craig Giannelli and Claude Code**
