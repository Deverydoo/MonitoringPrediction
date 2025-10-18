# Adaptive Retraining System - Design Document

**Project**: TFT Monitoring System v1.0.0
**Feature**: Intelligent Drift-Based Retraining
**Status**: Planning Phase (Revised)
**Date**: 2025-10-17

---

## Executive Summary

Implement an **adaptive retraining system** that monitors data drift and prediction accuracy, automatically triggering retraining during quiet periods when the model shows signs of staleness.

**Key Principles:**
- **Event-Driven**: Train when drift detected, not on fixed schedule
- **Context-Aware**: Only train during server quiet times
- **Safeguarded**: Min 6 hours between trainings, max 30 days without training
- **Intelligent**: Learns optimal training windows from historical load

---

## 1. Architecture Overview

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

---

## 2. Drift Detection System

### 2.1 What is "Drift"?

**Data Drift**: Input data distribution changes
- New server profiles added
- Infrastructure changes (new hardware, configs)
- Workload pattern shifts (new applications)

**Prediction Drift**: Model predictions become less accurate
- Actual metrics deviate from predictions
- Increasing prediction errors over time
- Model "forgetting" recent patterns

### 2.2 Drift Detection Metrics

Track these metrics in real-time:

```python
class DriftMonitor:
    """
    Monitors data and prediction drift in real-time.

    Metrics tracked:
    1. Prediction Error Rate (PER)
    2. Distribution Shift Score (DSS)
    3. Feature Drift Score (FDS)
    4. Anomaly Rate
    """

    def __init__(self):
        self.baseline_stats = self.load_baseline()
        self.window_size = 1000  # Last 1000 predictions

    def calculate_prediction_error_rate(self, predictions, actuals):
        """
        Compare predictions vs actual values over last N samples.

        PER = mean(|predicted - actual| / actual) over window

        Thresholds (based on 88% accuracy SLA):
        - PER < 0.08: Healthy (>92% accuracy - buffer above SLA)
        - PER 0.08-0.10: Warning (88-92% accuracy - approaching SLA)
        - PER 0.10-0.12: Urgent (86-88% accuracy - at/below SLA)
        - PER > 0.12: Critical (< 86% accuracy - SLA breach)

        We trigger retraining at 10% error (88% accuracy) to maintain SLA.
        """
        errors = []
        for pred, actual in zip(predictions, actuals):
            # Calculate MAPE (Mean Absolute Percentage Error)
            if actual > 0:
                error = abs(pred - actual) / actual
                errors.append(error)

        per = np.mean(errors)
        return per

    def calculate_distribution_shift(self, current_data, baseline_data):
        """
        Compare current data distribution to baseline.

        Uses Kolmogorov-Smirnov test for distribution similarity.

        Returns:
        - score: 0.0-1.0 (0=identical, 1=completely different)
        - p_value: Statistical significance
        """
        from scipy.stats import ks_2samp

        scores = {}
        for metric in NordIQ Metrics Framework_METRICS:
            stat, p_value = ks_2samp(
                current_data[metric],
                baseline_data[metric]
            )
            scores[metric] = {
                'ks_stat': stat,
                'p_value': p_value,
                'drifted': p_value < 0.05  # Significant drift
            }

        # Overall drift score (average KS statistic)
        dss = np.mean([s['ks_stat'] for s in scores.values()])

        return dss, scores

    def calculate_feature_drift(self, recent_window):
        """
        Detect sudden changes in feature statistics.

        FDS = Σ |current_mean - baseline_mean| / baseline_std

        High FDS indicates data characteristics changed.
        """
        drift_scores = {}

        for metric in NordIQ Metrics Framework_METRICS:
            current_mean = recent_window[metric].mean()
            current_std = recent_window[metric].std()

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

        Anomaly = prediction outside expected range
        (e.g., CPU > 100%, memory < 0%)

        High anomaly rate suggests model confusion.
        """
        anomalies = 0
        for pred in recent_predictions:
            if self.is_anomalous(pred):
                anomalies += 1

        anomaly_rate = anomalies / len(recent_predictions)
        return anomaly_rate

    def get_drift_signal(self) -> Dict:
        """
        Aggregate all drift metrics into single signal.

        Returns:
        {
            'drift_detected': True/False,
            'confidence': 0.0-1.0,
            'metrics': {...},
            'recommendation': 'retrain' | 'monitor' | 'healthy'
        }
        """
        per = self.calculate_prediction_error_rate()
        dss, dist_scores = self.calculate_distribution_shift()
        fds, feat_scores = self.calculate_feature_drift()
        anomaly_rate = self.calculate_anomaly_rate()

        # Weighted drift score
        drift_score = (
            per * 0.40 +           # Prediction error most important
            dss * 0.30 +           # Distribution shift
            fds * 0.20 +           # Feature drift
            anomaly_rate * 0.10    # Anomaly rate
        )

        # Decision thresholds (aligned with 88% accuracy SLA)
        if drift_score > 0.10:
            # Critical: At or below SLA (88% accuracy)
            recommendation = 'retrain'
            drift_detected = True
            confidence = min(1.0, drift_score / 0.15)
        elif drift_score > 0.08:
            # Warning: Approaching SLA (88-92% accuracy)
            recommendation = 'monitor'
            drift_detected = False
            confidence = drift_score / 0.10
        else:
            # Healthy: Above SLA (>92% accuracy)
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

## 3. Quiet Time Detection

### 3.1 What is "Quiet Time"?

Periods when:
- Server load is low (avg CPU < 30%)
- Prediction frequency is low (< 50% of peak)
- No active incidents or alerts
- Minimal user activity

### 3.2 Quiet Time Detection Algorithm

```python
class QuietTimeDetector:
    """
    Identifies optimal windows for retraining.

    Learns patterns:
    - Weekends are quieter than weekdays
    - Nights (2-5 AM) are quieter than business hours
    - Post-market (6 PM - 8 PM) quieter than market hours
    """

    def __init__(self):
        self.historical_load = self.load_load_history()
        self.min_quiet_duration = 1.5  # Hours (1.5h = 1 epoch training time)

    def calculate_current_load(self) -> float:
        """
        Calculate current infrastructure load.

        Returns: 0.0-1.0 (0=idle, 1=maxed out)
        """
        # Query inference daemon for current metrics
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

        Returns:
        - (start_time, end_time) if quiet window found
        - None if no quiet window in next 24 hours
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
            if load < 0.3:  # Quiet threshold
                if window_start is None:
                    window_start = check_time
            else:
                if window_start is not None:
                    window_end = check_time
                    duration = (window_end - window_start).total_seconds() / 3600

                    if duration >= self.min_quiet_duration:
                        quiet_windows.append((window_start, window_end))

                    window_start = None

        # Return soonest quiet window
        if quiet_windows:
            return quiet_windows[0]
        else:
            return None

    def is_quiet_now(self) -> bool:
        """Check if current time is quiet enough for training."""
        load = self.calculate_current_load()
        return load < 0.3  # < 30% load

    def predict_load_at_time(self, target_time: datetime) -> float:
        """
        Predict infrastructure load at future time.

        Uses historical patterns:
        - Hour of day (diurnal)
        - Day of week (weekly)
        - Special events (holidays, quarter-end)
        """
        hour = target_time.hour
        weekday = target_time.weekday()

        # Get historical load for this hour/weekday
        historical_avg = self.historical_load[f'{weekday}_{hour}']['mean']
        historical_std = self.historical_load[f'{weekday}_{hour}']['std']

        # Add some uncertainty
        predicted_load = np.random.normal(historical_avg, historical_std * 0.5)

        return np.clip(predicted_load, 0, 1)
```

---

## 4. Training Decision Engine

### 4.1 Decision Algorithm

```python
class RetrainingDecisionEngine:
    """
    Decides when to trigger retraining.

    Inputs:
    - Drift signal (from DriftMonitor)
    - Quiet time status (from QuietTimeDetector)
    - Training history (last training time, frequency)

    Safeguards:
    - Min 6 hours between trainings (prevent thrashing)
    - Max 30 days without training (force refresh)
    - Max 3 trainings per week (cost control)
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

        Returns:
        {
            'trigger': True/False,
            'reason': str,
            'confidence': float,
            'wait_until': datetime (if not triggering now)
        }
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
                'wait_until': self.last_training_time + timedelta(hours=self.min_hours_between_training)
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
            # Perfect conditions: drift detected + quiet now
            return {
                'trigger': True,
                'reason': f'Drift detected + quiet time (load={quiet_detector.calculate_current_load():.2f})',
                'confidence': drift_signal['confidence'],
                'drift_metrics': drift_signal['metrics']
            }
        else:
            # Drift detected but not quiet - schedule for later
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
                if drift_signal['drift_score'] > 0.12:  # < 88% accuracy
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

## 5. Adaptive Retraining Daemon

### 5.1 Main Loop

```python
# adaptive_retraining_daemon.py

class AdaptiveRetrainingDaemon:
    """
    Continuously monitors and triggers retraining when appropriate.

    Runs as background service:
    - Checks drift every 5 minutes
    - Triggers training when conditions met
    - Logs all decisions for analysis
    """

    def __init__(self):
        self.drift_monitor = DriftMonitor()
        self.quiet_detector = QuietTimeDetector()
        self.decision_engine = RetrainingDecisionEngine()
        self.running = True

    def run(self):
        """Main daemon loop."""
        print("[START] Adaptive Retraining Daemon started")

        while self.running:
            try:
                # Check if training should be triggered
                decision = self.decision_engine.should_trigger_training()

                self.log_decision(decision)

                if decision['trigger']:
                    print(f"\n[TRIGGER] Training triggered: {decision['reason']}")
                    self.execute_training(decision)
                else:
                    print(f"[SKIP] {decision['reason']}")

                # Sleep 5 minutes before next check
                time.sleep(300)

            except Exception as e:
                print(f"[ERROR] Daemon error: {e}")
                time.sleep(60)  # Retry in 1 minute

    def execute_training(self, decision: Dict):
        """Execute training workflow."""
        try:
            print("[TRAIN] Starting incremental training...")

            # Prepare data
            buffer = DataBuffer('./data_buffer')
            training_data = buffer.get_training_window(days=30)

            # Train incrementally (1 epoch)
            model_path = train_model(
                dataset_path=training_data,
                epochs=1,
                incremental=True
            )

            if model_path:
                print(f"[SUCCESS] Training complete: {model_path}")

                # Update last training time
                self.decision_engine.last_training_time = datetime.now()
                self.decision_engine.save_training_history()

                # Reset drift baseline
                self.drift_monitor.update_baseline()

                # Send notification
                self.send_notification(
                    f"✅ Adaptive retraining completed\n"
                    f"Reason: {decision['reason']}\n"
                    f"Confidence: {decision['confidence']:.2f}\n"
                    f"Drift metrics: {decision.get('drift_metrics', {})}"
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

        # Append to JSON log
        with open('./logs/retraining_decisions.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

---

## 6. Safeguards Summary

| Safeguard | Value | Purpose |
|-----------|-------|---------|
| **Min Time Between Training** | 6 hours | Prevent training thrashing |
| **Max Time Without Training** | 30 days | Force periodic refresh |
| **Max Trainings Per Week** | 3 times | Cost control |
| **Drift Threshold** | 0.10 (10% error = 88% accuracy) | Maintain SLA |
| **Warning Threshold** | 0.08 (8% error = 92% accuracy) | Early warning buffer |
| **Quiet Load Threshold** | 0.3 (30% load) | Avoid impacting production |
| **Min Quiet Duration** | 1.5 hours | Ensure training completes |

---

## 7. Example Scenarios

### Scenario 1: Gradual Drift (Ideal Case)

```
Day 1: Drift=0.05, Load=0.25 → No training (healthy, 95% accuracy)
Day 3: Drift=0.07, Load=0.28 → No training (healthy, 93% accuracy)
Day 5: Drift=0.09, Load=0.27 → No training (warning, 91% accuracy - monitoring)
Day 7: Drift=0.11, Load=0.22 → TRAINING TRIGGERED ✅
        (Drift at 11% = 89% accuracy, below 88% SLA threshold)

Result: Training triggered before SLA breach becomes critical
```

### Scenario 2: Sudden Infrastructure Change

```
12:00 PM: New servers added, Drift=0.13, Load=0.75 → No training (too busy)
          (87% accuracy - below SLA but infrastructure busy)
2:00 PM:  Drift=0.14, Load=0.70 → No training (still busy)
          (86% accuracy - critical but waiting for quiet)
6:00 PM:  Drift=0.14, Load=0.42 → No training (moderate load)
10:00 PM: Drift=0.14, Load=0.28 → TRAINING TRIGGERED ✅
          (Drift critical + quiet window found)

Result: Training delayed until quiet period, but triggered same day
```

### Scenario 3: Training Throttle

```
Monday 2 AM: Drift detected → Training #1 ✅
Wednesday 3 AM: Drift detected → Training #2 ✅
Friday 1 AM: Drift detected → Training #3 ✅
Saturday 2 AM: Drift detected → BLOCKED ❌
               (3 trainings this week already)

Result: Weekly limit prevents excessive training
```

### Scenario 4: Forced Refresh

```
Day 1: Model trained
Day 10: Drift=0.06 → No training (healthy, 94% accuracy)
Day 20: Drift=0.07 → No training (healthy, 93% accuracy)
Day 30: Drift=0.07 → TRAINING TRIGGERED ✅
        (30 days elapsed, forced refresh even though above SLA)

Result: Periodic refresh ensures model doesn't become stale
```

---

## 8. Monitoring Dashboard

Track these metrics:

```python
retraining_dashboard = {
    'current_drift_score': 0.09,
    'current_accuracy': '91%',  # 100% - drift_score
    'sla_status': 'Above SLA (target: 88%)',
    'buffer_above_sla': '+3%',
    'days_since_last_training': 5,
    'trainings_this_week': 1,
    'next_scheduled_training': '2025-10-18 02:15:00',
    'current_load': 0.28,
    'is_quiet': True,

    'last_7_days': {
        'trainings_triggered': 2,
        'trainings_skipped': 15,
        'avg_drift_score': 0.08,
        'avg_accuracy': '92%',
        'sla_breaches': 0
    },

    'drift_breakdown': {
        'prediction_error_rate': 0.09,  # 91% accuracy
        'distribution_shift': 0.07,
        'feature_drift': 0.08,
        'anomaly_rate': 0.02
    },

    'alert_levels': {
        'healthy': 'drift < 0.08 (> 92% accuracy)',
        'warning': 'drift 0.08-0.10 (88-92% accuracy)',
        'critical': 'drift > 0.10 (< 88% accuracy)'
    }
}
```

---

## 9. Configuration

```python
# config/adaptive_retraining_config.py
ADAPTIVE_RETRAINING_CONFIG = {
    # Drift Detection (Aligned with 88% Accuracy SLA)
    'drift_check_interval_mins': 5,
    'drift_threshold_critical': 0.10,  # 88% accuracy - trigger retraining
    'drift_threshold_warning': 0.08,   # 92% accuracy - start monitoring
    'prediction_window_size': 1000,  # Samples to track

    # Quiet Time Detection
    'quiet_load_threshold': 0.3,  # <30% load
    'min_quiet_duration_hours': 1.5,
    'load_prediction_lookback_days': 30,

    # Training Safeguards
    'min_hours_between_training': 6,
    'max_days_without_training': 30,
    'max_trainings_per_week': 3,

    # Training Parameters
    'training_window_days': 30,
    'epochs_per_training': 1,

    # Monitoring
    'log_dir': './logs/adaptive_retraining',
    'alert_email': 'ml-ops@company.com',
    'drift_metrics_retention_days': 90
}
```

---

## 10. Benefits Over Fixed Schedule

| Fixed Schedule (2 AM Daily) | Adaptive (Drift-Based) |
|------------------------------|------------------------|
| Trains even if model is good | Only trains when needed |
| Fixed cost (daily GPU hours) | Variable cost (only when needed) |
| May train during incidents | Waits for quiet periods |
| 30 trainings/month | ~8-12 trainings/month (60% reduction) |
| No drift awareness | Responds to actual model staleness |

**Cost Savings**: 60% reduction in training costs (from $1500/mo → $600/mo GPU)

---

## 11. Implementation Timeline

### Phase 1: Drift Detection (Week 1-2)
- [ ] Implement DriftMonitor class
- [ ] Add prediction error tracking to inference daemon
- [ ] Calculate distribution shift metrics
- [ ] Test drift detection on historical data

### Phase 2: Quiet Time Detection (Week 2-3)
- [ ] Implement QuietTimeDetector class
- [ ] Collect historical load patterns (7 days)
- [ ] Build load prediction model
- [ ] Test quiet window detection

### Phase 3: Decision Engine (Week 3-4)
- [ ] Implement RetrainingDecisionEngine
- [ ] Add all safeguard checks
- [ ] Test decision logic with various scenarios
- [ ] Tune thresholds based on testing

### Phase 4: Daemon & Integration (Week 4-5)
- [ ] Implement AdaptiveRetrainingDaemon
- [ ] Integrate with existing training pipeline
- [ ] Add monitoring and alerting
- [ ] Deploy as system service

### Phase 5: Monitoring & Tuning (Week 5-6)
- [ ] Create drift monitoring dashboard
- [ ] Monitor for 2 weeks
- [ ] Tune thresholds based on real data
- [ ] Document runbook

**Target Go-Live**: 6 weeks from approval

---

## 12. Success Metrics

1. **Training Efficiency**: 60% fewer trainings than fixed schedule
2. **Drift Response Time**: Training triggered within 4 hours of drift detection
3. **Production Impact**: Zero trainings during high-load periods
4. **Model Freshness**: Prediction error stays <10%
5. **Cost Savings**: 60% reduction in training compute costs

---

## 13. Next Steps

1. **Review & Approve**: This adaptive approach vs fixed schedule
2. **Tune Thresholds**: Adjust drift threshold (0.15) based on your tolerance
3. **Define Quiet Times**: Review quiet load threshold (30%)
4. **Start Phase 1**: Implement drift detection first

**Ready to proceed?** This intelligent system will train when needed, not on blind schedules! 🎯

---

**Document Status**: Draft for Review (Revised Approach)
**Replaces**: CONTINUOUS_LEARNING_PLAN.md
**Approvers**: Engineering Lead, ML Team, Operations
