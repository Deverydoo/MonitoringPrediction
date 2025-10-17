#!/usr/bin/env python3
"""
Drift Monitor - Real-time model drift detection

Tracks 4 key metrics:
1. Prediction Error Rate (PER) - How accurate are predictions vs actuals
2. Distribution Shift Score (DSS) - Statistical distance from baseline
3. Feature Drift Score (FDS) - Individual feature distributions
4. Anomaly Rate - Unusual patterns in predictions

When drift exceeds thresholds, triggers retraining recommendation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from scipy import stats
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Real-time drift detection for TFT model.

    Monitors incoming predictions and actual values to detect when
    model performance degrades, triggering retraining when needed.
    """

    def __init__(
        self,
        baseline_stats_path: Optional[str] = None,
        window_size: int = 1000,
        save_path: str = "drift_metrics.json"
    ):
        """
        Initialize drift monitor.

        Args:
            baseline_stats_path: Path to baseline statistics JSON
            window_size: Number of recent predictions to track
            save_path: Where to save drift metrics
        """
        self.window_size = window_size
        self.save_path = save_path

        # Load or initialize baseline statistics
        if baseline_stats_path and Path(baseline_stats_path).exists():
            with open(baseline_stats_path, 'r') as f:
                self.baseline_stats = json.load(f)
            logger.info(f"✅ Loaded baseline stats from {baseline_stats_path}")
        else:
            self.baseline_stats = self._initialize_baseline()
            logger.warning("⚠️ No baseline stats found, using defaults")

        # Rolling windows for real-time tracking
        self.prediction_errors = deque(maxlen=window_size)
        self.feature_values = {
            'cpu_pct': deque(maxlen=window_size),
            'mem_pct': deque(maxlen=window_size),
            'disk_io_mb_s': deque(maxlen=window_size),
            'latency_ms': deque(maxlen=window_size)
        }
        self.prediction_confidences = deque(maxlen=window_size)
        self.anomaly_flags = deque(maxlen=window_size)

        # Drift metric history
        self.drift_history = []

        # Thresholds (configurable)
        self.thresholds = {
            'per_threshold': 0.10,      # 10% prediction error rate
            'dss_threshold': 0.20,      # 20% distribution shift
            'fds_threshold': 0.15,      # 15% feature drift
            'anomaly_threshold': 0.05   # 5% anomaly rate
        }

        # Weights for combined drift score
        self.weights = {
            'per_weight': 0.40,  # Prediction accuracy most important
            'dss_weight': 0.30,  # Distribution shift second
            'fds_weight': 0.20,  # Feature drift third
            'anomaly_weight': 0.10  # Anomalies least critical
        }

        logger.info(f"🔍 DriftMonitor initialized (window={window_size})")

    def _initialize_baseline(self) -> Dict:
        """Initialize baseline statistics with reasonable defaults."""
        return {
            'cpu_pct': {'mean': 50.0, 'std': 20.0},
            'mem_pct': {'mean': 60.0, 'std': 15.0},
            'disk_io_mb_s': {'mean': 100.0, 'std': 50.0},
            'latency_ms': {'mean': 50.0, 'std': 25.0},
            'prediction_error': {'mean': 0.05, 'std': 0.03}
        }

    def update(
        self,
        predictions: Dict[str, float],
        actuals: Optional[Dict[str, float]] = None,
        features: Optional[Dict[str, float]] = None
    ):
        """
        Update drift metrics with new data point.

        Args:
            predictions: Model predictions {'cpu_pct': 75.2, ...}
            actuals: Actual observed values (if available)
            features: Current feature values
        """
        # 1. Track prediction error (if actuals available)
        if actuals is not None:
            error = self._calculate_prediction_error(predictions, actuals)
            self.prediction_errors.append(error)

        # 2. Track feature values
        if features is not None:
            for feature, value in features.items():
                if feature in self.feature_values:
                    self.feature_values[feature].append(value)

        # 3. Track anomalies (simple z-score check)
        is_anomaly = self._detect_anomaly(features or predictions)
        self.anomaly_flags.append(is_anomaly)

    def _calculate_prediction_error(
        self,
        predictions: Dict[str, float],
        actuals: Dict[str, float]
    ) -> float:
        """
        Calculate normalized prediction error.

        Uses Mean Absolute Percentage Error (MAPE).
        """
        errors = []
        for metric in ['cpu_pct', 'mem_pct', 'disk_io_mb_s', 'latency_ms']:
            if metric in predictions and metric in actuals:
                pred = predictions[metric]
                actual = actuals[metric]
                if actual != 0:
                    mape = abs(pred - actual) / abs(actual)
                    errors.append(mape)

        return np.mean(errors) if errors else 0.0

    def _detect_anomaly(self, values: Dict[str, float]) -> bool:
        """
        Detect if current values are anomalous using z-score.

        Anomaly if any metric > 3 std devs from baseline.
        """
        for metric, value in values.items():
            if metric in self.baseline_stats:
                baseline = self.baseline_stats[metric]
                z_score = abs(value - baseline['mean']) / baseline['std']
                if z_score > 3.0:
                    return True
        return False

    def calculate_drift_metrics(self) -> Dict[str, float]:
        """
        Calculate all 4 drift metrics.

        Returns:
            Dict with per, dss, fds, anomaly_rate, combined_score
        """
        metrics = {}

        # 1. Prediction Error Rate (PER)
        if len(self.prediction_errors) > 0:
            metrics['per'] = np.mean(list(self.prediction_errors))
        else:
            metrics['per'] = 0.0

        # 2. Distribution Shift Score (DSS) - Kolmogorov-Smirnov test
        metrics['dss'] = self._calculate_distribution_shift()

        # 3. Feature Drift Score (FDS) - Per-feature drift
        metrics['fds'] = self._calculate_feature_drift()

        # 4. Anomaly Rate
        if len(self.anomaly_flags) > 0:
            metrics['anomaly_rate'] = sum(self.anomaly_flags) / len(self.anomaly_flags)
        else:
            metrics['anomaly_rate'] = 0.0

        # 5. Combined weighted drift score (0-1, higher = more drift)
        metrics['combined_score'] = (
            metrics['per'] * self.weights['per_weight'] +
            metrics['dss'] * self.weights['dss_weight'] +
            metrics['fds'] * self.weights['fds_weight'] +
            metrics['anomaly_rate'] * self.weights['anomaly_weight']
        )

        # 6. Recommendation
        metrics['needs_retraining'] = self._evaluate_retraining_need(metrics)

        # Save to history
        metrics['timestamp'] = datetime.now().isoformat()
        self.drift_history.append(metrics)

        return metrics

    def _calculate_distribution_shift(self) -> float:
        """
        Calculate distribution shift using KS test.

        Compares current window to baseline distribution.
        Returns: 0.0 (no shift) to 1.0 (complete shift)
        """
        shifts = []

        for feature, values in self.feature_values.items():
            if len(values) < 30:  # Need minimum samples
                continue

            # Create baseline distribution
            baseline = self.baseline_stats.get(feature, {})
            if not baseline:
                continue

            baseline_samples = np.random.normal(
                baseline['mean'],
                baseline['std'],
                size=len(values)
            )

            # KS test
            ks_stat, p_value = stats.ks_2samp(list(values), baseline_samples)
            shifts.append(ks_stat)

        return np.mean(shifts) if shifts else 0.0

    def _calculate_feature_drift(self) -> float:
        """
        Calculate per-feature drift scores.

        Uses z-score distance: |current_mean - baseline_mean| / baseline_std
        """
        drifts = []

        for feature, values in self.feature_values.items():
            if len(values) < 10:
                continue

            baseline = self.baseline_stats.get(feature, {})
            if not baseline:
                continue

            current_mean = np.mean(list(values))
            drift_score = abs(current_mean - baseline['mean']) / baseline['std']

            # Normalize to 0-1 range (cap at 3 std devs)
            normalized_drift = min(drift_score / 3.0, 1.0)
            drifts.append(normalized_drift)

        return np.mean(drifts) if drifts else 0.0

    def _evaluate_retraining_need(self, metrics: Dict[str, float]) -> bool:
        """
        Evaluate if retraining is needed based on thresholds.

        Retraining recommended if ANY metric exceeds threshold.
        """
        conditions = [
            metrics['per'] > self.thresholds['per_threshold'],
            metrics['dss'] > self.thresholds['dss_threshold'],
            metrics['fds'] > self.thresholds['fds_threshold'],
            metrics['anomaly_rate'] > self.thresholds['anomaly_threshold']
        ]

        return any(conditions)

    def get_drift_status(self) -> Dict:
        """
        Get current drift status summary.

        Returns:
            Dict with current metrics, trends, and recommendations
        """
        if not self.drift_history:
            metrics = self.calculate_drift_metrics()
        else:
            metrics = self.drift_history[-1]

        # Calculate trends (last 10 measurements)
        recent = self.drift_history[-10:] if len(self.drift_history) >= 10 else self.drift_history

        trends = {}
        if len(recent) >= 2:
            for key in ['per', 'dss', 'fds', 'anomaly_rate']:
                values = [m[key] for m in recent]
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                trends[key] = trend

        return {
            'current_metrics': metrics,
            'trends': trends,
            'window_size': len(self.prediction_errors),
            'data_points_tracked': len(self.drift_history),
            'recommendation': 'RETRAIN NOW' if metrics['needs_retraining'] else 'OK',
            'thresholds': self.thresholds
        }

    def save_metrics(self):
        """Save current drift metrics to disk."""
        status = self.get_drift_status()

        with open(self.save_path, 'w') as f:
            json.dump(status, f, indent=2)

        logger.info(f"💾 Saved drift metrics to {self.save_path}")

    def generate_report(self) -> str:
        """
        Generate human-readable drift report.

        Returns:
            Formatted report string
        """
        status = self.get_drift_status()
        metrics = status['current_metrics']

        report = []
        report.append("=" * 60)
        report.append("DRIFT DETECTION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {metrics['timestamp']}")
        report.append(f"Window Size: {self.window_size} predictions")
        report.append(f"Data Points: {status['data_points_tracked']}")
        report.append("")

        report.append("DRIFT METRICS:")
        report.append(f"  1. Prediction Error Rate (PER):  {metrics['per']:.2%} (threshold: {self.thresholds['per_threshold']:.2%})")
        report.append(f"  2. Distribution Shift Score (DSS): {metrics['dss']:.2%} (threshold: {self.thresholds['dss_threshold']:.2%})")
        report.append(f"  3. Feature Drift Score (FDS):     {metrics['fds']:.2%} (threshold: {self.thresholds['fds_threshold']:.2%})")
        report.append(f"  4. Anomaly Rate:                  {metrics['anomaly_rate']:.2%} (threshold: {self.thresholds['anomaly_threshold']:.2%})")
        report.append("")

        report.append(f"COMBINED DRIFT SCORE: {metrics['combined_score']:.2%}")
        report.append("")

        # Status
        if metrics['needs_retraining']:
            report.append("⚠️  STATUS: RETRAINING RECOMMENDED")
            report.append("   One or more metrics exceed thresholds")
        else:
            report.append("✅ STATUS: MODEL HEALTHY")
            report.append("   All metrics within acceptable ranges")

        report.append("=" * 60)

        return "\n".join(report)


if __name__ == '__main__':
    # Example usage
    print("🔍 Drift Monitor - Example Usage\n")

    # Initialize monitor
    monitor = DriftMonitor(window_size=100)

    # Simulate 150 data points
    np.random.seed(42)
    for i in range(150):
        # Simulate predictions
        predictions = {
            'cpu_pct': 50 + np.random.normal(0, 20),
            'mem_pct': 60 + np.random.normal(0, 15)
        }

        # Simulate actuals (with increasing error over time to simulate drift)
        drift_factor = i / 150  # Gradual drift
        actuals = {
            'cpu_pct': predictions['cpu_pct'] + np.random.normal(drift_factor * 10, 5),
            'mem_pct': predictions['mem_pct'] + np.random.normal(drift_factor * 8, 4)
        }

        # Update monitor
        monitor.update(predictions, actuals, predictions)

        # Check drift every 50 points
        if (i + 1) % 50 == 0:
            print(f"\n--- After {i + 1} predictions ---")
            metrics = monitor.calculate_drift_metrics()
            print(f"PER: {metrics['per']:.2%}")
            print(f"DSS: {metrics['dss']:.2%}")
            print(f"FDS: {metrics['fds']:.2%}")
            print(f"Anomaly Rate: {metrics['anomaly_rate']:.2%}")
            print(f"Combined Score: {metrics['combined_score']:.2%}")
            print(f"Needs Retraining: {metrics['needs_retraining']}")

    # Final report
    print("\n")
    print(monitor.generate_report())

    # Save metrics
    monitor.save_metrics()
    print(f"\n💾 Drift metrics saved to {monitor.save_path}")
