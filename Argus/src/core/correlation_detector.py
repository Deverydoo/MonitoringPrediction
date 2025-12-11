#!/usr/bin/env python3
"""
Correlation Detector - Cross-Server Anomaly Detection for Cascading Failures

Detects environment-wide issues by analyzing correlations between servers.
If multiple servers show correlated anomalies simultaneously, this indicates
a cascading failure or infrastructure-wide problem.

Key Features:
1. Cross-server correlation tracking
2. Simultaneous anomaly detection
3. Cascading failure probability scoring
4. Environment health degradation alerts
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationDetector:
    """
    Detects cascading failures by monitoring cross-server correlations.

    When multiple servers exhibit correlated behavior (e.g., all showing
    increased CPU at the same time), this indicates an environment-wide
    issue rather than isolated server problems.
    """

    def __init__(
        self,
        window_size: int = 100,
        correlation_threshold: float = 0.7,
        cascade_server_threshold: int = 3,
        anomaly_z_threshold: float = 2.0
    ):
        """
        Initialize correlation detector.

        Args:
            window_size: Number of timesteps to track per metric
            correlation_threshold: Min correlation to consider servers "correlated"
            cascade_server_threshold: Number of servers needed to trigger cascade alert
            anomaly_z_threshold: Z-score threshold for anomaly detection
        """
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold
        self.cascade_server_threshold = cascade_server_threshold
        self.anomaly_z_threshold = anomaly_z_threshold

        # Per-server metric history
        # Structure: {server_name: {metric: deque([values])}}
        self.server_metrics: Dict[str, Dict[str, deque]] = {}

        # Track metrics we care about for correlation
        self.tracked_metrics = [
            'cpu_user_pct', 'mem_used_pct', 'cpu_iowait_pct',
            'load_average', 'swap_used_pct'
        ]

        # Cascade event history
        self.cascade_events: List[Dict] = []

        # Baseline statistics per server
        self.server_baselines: Dict[str, Dict[str, Dict]] = {}

        logger.info(f"🔗 CorrelationDetector initialized (window={window_size})")

    def update(self, records: List[Dict]) -> Dict:
        """
        Update with new batch of server metrics.

        Args:
            records: List of server metric records

        Returns:
            Analysis result including any cascade alerts
        """
        timestamp = datetime.now()

        # Process each record
        for record in records:
            server_name = record.get('server_name')
            if not server_name:
                continue

            # Initialize server tracking if needed
            if server_name not in self.server_metrics:
                self.server_metrics[server_name] = {
                    metric: deque(maxlen=self.window_size)
                    for metric in self.tracked_metrics
                }
                self.server_baselines[server_name] = {}

            # Update metric values
            for metric in self.tracked_metrics:
                if metric in record:
                    value = float(record[metric])
                    self.server_metrics[server_name][metric].append(value)

                    # Update baseline statistics
                    self._update_baseline(server_name, metric, value)

        # Check for cascading failures
        cascade_result = self._detect_cascade(timestamp)

        return cascade_result

    def _update_baseline(self, server_name: str, metric: str, value: float):
        """Update running baseline statistics for a server metric."""
        if metric not in self.server_baselines[server_name]:
            self.server_baselines[server_name][metric] = {
                'mean': value,
                'std': 1.0,  # Start with reasonable default
                'count': 1
            }
        else:
            baseline = self.server_baselines[server_name][metric]
            count = baseline['count']

            # Running mean and std update (Welford's algorithm)
            new_count = count + 1
            delta = value - baseline['mean']
            new_mean = baseline['mean'] + delta / new_count
            delta2 = value - new_mean
            new_m2 = baseline.get('m2', 0) + delta * delta2

            baseline['mean'] = new_mean
            baseline['m2'] = new_m2
            baseline['std'] = max(1.0, np.sqrt(new_m2 / new_count))  # Floor at 1.0
            baseline['count'] = new_count

    def _detect_cascade(self, timestamp: datetime) -> Dict:
        """
        Detect cascading failures across servers.

        Returns:
            Dict with cascade analysis results
        """
        if len(self.server_metrics) < 2:
            return {
                'cascade_detected': False,
                'reason': 'Not enough servers to detect cascade'
            }

        # Collect current anomalies per server
        server_anomalies: Dict[str, List[str]] = {}  # server -> list of anomalous metrics

        for server_name, metrics in self.server_metrics.items():
            anomalous_metrics = []

            for metric, values in metrics.items():
                if len(values) < 10:
                    continue

                # Calculate current z-score
                baseline = self.server_baselines[server_name].get(metric)
                if not baseline:
                    continue

                current_value = values[-1]
                z_score = (current_value - baseline['mean']) / baseline['std']

                if abs(z_score) > self.anomaly_z_threshold:
                    anomalous_metrics.append(metric)

            if anomalous_metrics:
                server_anomalies[server_name] = anomalous_metrics

        # Check if multiple servers have simultaneous anomalies
        cascade_metrics: Dict[str, List[str]] = {}  # metric -> servers with that anomaly

        for server, metrics in server_anomalies.items():
            for metric in metrics:
                if metric not in cascade_metrics:
                    cascade_metrics[metric] = []
                cascade_metrics[metric].append(server)

        # Find cascade candidates (same metric anomaly on multiple servers)
        cascades = []
        for metric, servers in cascade_metrics.items():
            if len(servers) >= self.cascade_server_threshold:
                cascades.append({
                    'metric': metric,
                    'affected_servers': servers,
                    'server_count': len(servers),
                    'severity': self._calculate_cascade_severity(servers, metric)
                })

        # Calculate correlation scores for cascade confirmation
        correlation_score = self._calculate_cross_server_correlation()

        # Determine overall cascade status
        cascade_detected = len(cascades) > 0 or correlation_score > self.correlation_threshold

        result = {
            'cascade_detected': cascade_detected,
            'timestamp': timestamp.isoformat(),
            'total_servers': len(self.server_metrics),
            'servers_with_anomalies': len(server_anomalies),
            'anomaly_rate': len(server_anomalies) / max(1, len(self.server_metrics)),
            'correlation_score': correlation_score,
            'cascades': cascades
        }

        # Add alert if cascade detected
        if cascade_detected:
            result['alert'] = self._generate_cascade_alert(cascades, correlation_score)

            # Record event
            event = {
                'timestamp': timestamp.isoformat(),
                'cascades': cascades,
                'correlation_score': correlation_score,
                'affected_servers': list(server_anomalies.keys())
            }
            self.cascade_events.append(event)
            if len(self.cascade_events) > 100:  # Keep last 100 events
                self.cascade_events.pop(0)

            logger.warning(f"🔴 CASCADE DETECTED: {len(server_anomalies)} servers affected")
            for cascade in cascades:
                logger.warning(f"   {cascade['metric']}: {cascade['server_count']} servers")

        return result

    def _calculate_cascade_severity(self, servers: List[str], metric: str) -> str:
        """Calculate severity of a cascade event."""
        server_count = len(servers)
        total_servers = len(self.server_metrics)
        pct_affected = server_count / max(1, total_servers)

        if pct_affected >= 0.5:
            return 'critical'
        elif pct_affected >= 0.25:
            return 'high'
        elif pct_affected >= 0.1:
            return 'medium'
        else:
            return 'low'

    def _calculate_cross_server_correlation(self) -> float:
        """
        Calculate average correlation across servers for key metrics.

        Returns correlation score 0-1 (higher = more correlated = more likely cascade).
        """
        if len(self.server_metrics) < 2:
            return 0.0

        correlations = []

        # For each metric, calculate pairwise correlations between servers
        for metric in self.tracked_metrics:
            server_series = []

            for server_name, metrics in self.server_metrics.items():
                if metric in metrics and len(metrics[metric]) >= 20:
                    server_series.append(list(metrics[metric])[-20:])  # Last 20 values

            if len(server_series) < 2:
                continue

            # Calculate pairwise correlations
            for i in range(len(server_series)):
                for j in range(i + 1, len(server_series)):
                    try:
                        corr = np.corrcoef(server_series[i], server_series[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        pass

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def _generate_cascade_alert(
        self,
        cascades: List[Dict],
        correlation_score: float
    ) -> Dict:
        """Generate a cascade alert with recommendations."""
        # Determine overall severity
        severities = [c['severity'] for c in cascades]
        if 'critical' in severities:
            overall_severity = 'critical'
        elif 'high' in severities:
            overall_severity = 'high'
        elif 'medium' in severities:
            overall_severity = 'medium'
        else:
            overall_severity = 'low'

        # Get all affected servers
        affected_servers = set()
        for cascade in cascades:
            affected_servers.update(cascade['affected_servers'])

        # Generate recommendations
        recommendations = []
        for cascade in cascades:
            metric = cascade['metric']
            if metric == 'cpu_user_pct':
                recommendations.append("Check for runaway processes or batch jobs")
            elif metric == 'mem_used_pct':
                recommendations.append("Check for memory leaks or cache pressure")
            elif metric == 'cpu_iowait_pct':
                recommendations.append("Check storage system or network mounts")
            elif metric == 'load_average':
                recommendations.append("Check for process pile-up or resource contention")
            elif metric == 'swap_used_pct':
                recommendations.append("Check memory pressure - may need to restart services")

        return {
            'severity': overall_severity,
            'affected_server_count': len(affected_servers),
            'affected_servers': list(affected_servers),
            'correlation_score': f"{correlation_score:.2%}",
            'message': f"Cascading failure detected: {len(affected_servers)} servers affected",
            'recommendations': list(set(recommendations)),
            'cascade_details': cascades
        }

    def get_cascade_status(self) -> Dict:
        """Get current cascade detection status."""
        # Check current state
        current = self._detect_cascade(datetime.now())

        return {
            'current_status': current,
            'tracking': {
                'servers': len(self.server_metrics),
                'metrics_tracked': self.tracked_metrics,
                'window_size': self.window_size
            },
            'recent_events': self.cascade_events[-5:] if self.cascade_events else [],
            'event_count': len(self.cascade_events),
            'thresholds': {
                'correlation': self.correlation_threshold,
                'cascade_servers': self.cascade_server_threshold,
                'anomaly_z_score': self.anomaly_z_threshold
            }
        }

    def get_fleet_health_score(self) -> Dict:
        """
        Calculate overall fleet health based on correlation patterns.

        Returns:
            Dict with fleet health metrics
        """
        correlation_score = self._calculate_cross_server_correlation()

        # Count servers with recent anomalies
        anomalous_servers = 0
        for server_name, metrics in self.server_metrics.items():
            for metric, values in metrics.items():
                if len(values) < 10:
                    continue

                baseline = self.server_baselines[server_name].get(metric)
                if not baseline:
                    continue

                current_value = values[-1]
                z_score = abs((current_value - baseline['mean']) / baseline['std'])

                if z_score > self.anomaly_z_threshold:
                    anomalous_servers += 1
                    break  # Count server once even if multiple metrics anomalous

        total_servers = len(self.server_metrics)
        anomaly_rate = anomalous_servers / max(1, total_servers)

        # Calculate health score (100 = healthy, 0 = critical)
        # Penalize for high correlation (indicates systemic issues)
        # Penalize for high anomaly rate
        health_score = 100 * (1 - correlation_score * 0.5) * (1 - anomaly_rate)
        health_score = max(0, min(100, health_score))

        # Determine health status
        if health_score >= 80:
            status = 'healthy'
        elif health_score >= 60:
            status = 'degraded'
        elif health_score >= 40:
            status = 'warning'
        else:
            status = 'critical'

        return {
            'health_score': round(health_score, 1),
            'status': status,
            'correlation_score': round(correlation_score, 3),
            'anomaly_rate': round(anomaly_rate, 3),
            'anomalous_servers': anomalous_servers,
            'total_servers': total_servers,
            'cascade_risk': 'high' if correlation_score > 0.7 else 'medium' if correlation_score > 0.5 else 'low'
        }


if __name__ == '__main__':
    # Example usage
    print("🔗 Correlation Detector - Example Usage\n")

    detector = CorrelationDetector(
        window_size=50,
        cascade_server_threshold=2
    )

    # Simulate correlated server behavior (cascading failure)
    np.random.seed(42)

    for tick in range(100):
        # Create base signal (simulating shared infrastructure issue)
        base_cpu = 40 + 20 * np.sin(tick / 10) + np.random.normal(0, 5)

        records = []
        for server_idx in range(5):
            # Each server follows base signal + noise (correlated behavior)
            record = {
                'server_name': f'server{server_idx:03d}',
                'timestamp': datetime.now().isoformat(),
                'cpu_user_pct': base_cpu + np.random.normal(0, 5),
                'mem_used_pct': 60 + np.random.normal(0, 10),
                'cpu_iowait_pct': 5 + np.random.normal(0, 2),
                'load_average': 2 + np.random.normal(0, 0.5),
                'swap_used_pct': 2 + np.random.normal(0, 1)
            }
            records.append(record)

        result = detector.update(records)

        if tick % 20 == 0 or result.get('cascade_detected'):
            print(f"\nTick {tick}:")
            print(f"  Correlation: {result.get('correlation_score', 0):.2%}")
            print(f"  Anomaly Rate: {result.get('anomaly_rate', 0):.2%}")
            if result.get('cascade_detected'):
                print(f"  🔴 CASCADE DETECTED!")
                for cascade in result.get('cascades', []):
                    print(f"    {cascade['metric']}: {cascade['server_count']} servers")

    # Final status
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)

    status = detector.get_cascade_status()
    health = detector.get_fleet_health_score()

    print(f"\nFleet Health: {health['status'].upper()} ({health['health_score']})")
    print(f"Correlation Score: {health['correlation_score']:.2%}")
    print(f"Anomaly Rate: {health['anomaly_rate']:.2%}")
    print(f"Cascade Risk: {health['cascade_risk'].upper()}")
    print(f"\nRecent Events: {status['event_count']}")
