"""
Metrics Extraction Utilities

Helper functions for extracting and computing metrics from NordIQ Metrics Framework data.
"""

import numpy as np
import streamlit as st
from typing import Dict, Tuple


@st.cache_data(ttl=10)  # Cache for 10 seconds - frequently called
def extract_cpu_used(server_pred: Dict, metric_type: str = 'current') -> float:
    """
    Extract CPU Used % from NordIQ Metrics Framework metrics.

    NordIQ Metrics Framework stores cpu_idle_pct, but we display as "CPU Used = 100 - idle" for human readability.

    Args:
        server_pred: Server prediction dictionary containing NordIQ Metrics Framework metrics
        metric_type: Type of value to extract:
            - 'current': Current actual value (float)
            - 'p50': P50 prediction list (list of floats)
            - 'p90': P90 prediction list (list of floats)
            - 'p10': P10 prediction list (list of floats)

    Returns:
        CPU used percentage (0-100)

    Examples:
        >>> extract_cpu_used(server_pred, 'current')
        45.2  # Current CPU usage

        >>> extract_cpu_used(server_pred, 'p90')
        78.5  # P90 worst-case CPU (next 30min)
    """
    # Method 1 (Preferred): Calculate from cpu_idle_pct (100 - idle)
    cpu_idle_data = server_pred.get('cpu_idle_pct', {})

    if metric_type == 'current':
        cpu_idle = cpu_idle_data.get('current', 0)
        if isinstance(cpu_idle, (int, float)) and cpu_idle > 0:
            return 100 - cpu_idle

    elif metric_type in ['p50', 'p90', 'p10']:
        prediction_list = cpu_idle_data.get(metric_type, [])
        if isinstance(prediction_list, list) and len(prediction_list) > 0:
            # For predictions, use first 6 values (next 30 minutes at 5-min intervals)
            values = prediction_list[:6] if len(prediction_list) >= 6 else prediction_list

            if metric_type == 'p90':
                # P90 CPU used = P10 idle (lower idle = higher CPU)
                return 100 - min(values)
            elif metric_type == 'p10':
                # P10 CPU used = P90 idle (higher idle = lower CPU)
                return 100 - max(values)
            else:  # p50
                return 100 - np.mean(values)

    # Method 2 (Fallback): Sum components (user + sys + iowait)
    cpu_user = server_pred.get('cpu_user_pct', {}).get(metric_type, 0)
    cpu_sys = server_pred.get('cpu_sys_pct', {}).get(metric_type, 0)
    cpu_iowait = server_pred.get('cpu_iowait_pct', {}).get(metric_type, 0)

    # Handle prediction lists
    if isinstance(cpu_user, list):
        cpu_user = np.mean(cpu_user[:6]) if len(cpu_user) >= 6 else (np.mean(cpu_user) if cpu_user else 0)
    if isinstance(cpu_sys, list):
        cpu_sys = np.mean(cpu_sys[:6]) if len(cpu_sys) >= 6 else (np.mean(cpu_sys) if cpu_sys else 0)
    if isinstance(cpu_iowait, list):
        cpu_iowait = np.mean(cpu_iowait[:6]) if len(cpu_iowait) >= 6 else (np.mean(cpu_iowait) if cpu_iowait else 0)

    return float(cpu_user + cpu_sys + cpu_iowait)


@st.cache_data(ttl=60)  # Cache for 60 seconds - static threshold logic
def get_metric_color_indicator(value: float, metric_type: str, profile: str = 'Generic') -> str:
    """
    Return color indicator for a metric value based on thresholds.

    Args:
        value: Metric value to evaluate
        metric_type: Type of metric ('cpu', 'iowait', 'memory', 'swap', 'load')
        profile: Server profile for profile-specific thresholds

    Returns:
        Color indicator: '' (healthy), '🟡' (warning), '🟠' (danger), '🔴' (critical)
    """
    if metric_type == 'cpu':
        if value >= 98:
            return '🔴'
        elif value >= 95:
            return '🟠'
        elif value >= 90:
            return '🟡'
    elif metric_type == 'iowait':
        if value >= 30:
            return '🔴'
        elif value >= 20:
            return '🟠'
        elif value >= 10:
            return '🟡'
    elif metric_type == 'memory':
        if profile == 'Database':
            # Database: Higher thresholds (page cache is normal)
            if value >= 98:
                return '🟠'
            elif value >= 95:
                return '🟡'
        else:
            if value >= 98:
                return '🔴'
            elif value >= 95:
                return '🟠'
            elif value >= 90:
                return '🟡'
    elif metric_type == 'swap':
        if value >= 50:
            return '🔴'
        elif value >= 30:
            return '🟠'
        elif value >= 10:
            return '🟡'
    elif metric_type == 'load':
        if value > 12:
            return '🔴'
        elif value > 8:
            return '🟠'
        elif value > 6:
            return '🟡'
    return ''


@st.cache_data(ttl=5)  # Cache for 5 seconds - real-time data
def get_health_status(predictions: Dict, _calculate_risk_func=None, _risk_scores: Dict[str, float] = None) -> Tuple[str, str, str]:
    """
    Determine overall environment health status based on ACTUAL server risk scores.

    Args:
        predictions: Full predictions dictionary from daemon
        _calculate_risk_func: Function to calculate server risk score (DEPRECATED - use _risk_scores)
        _risk_scores: Pre-calculated risk scores dict (PERFORMANCE: 50-100x faster)

    Returns:
        Tuple of (status_text, status_color, status_emoji)
    """
    if not predictions or 'predictions' not in predictions:
        return "Unknown", "gray", "❓"

    # Calculate actual fleet health based on server risk scores
    server_preds = predictions.get('predictions', {})
    if not server_preds:
        return "Unknown", "gray", "❓"

    # Count servers by risk level
    critical_count = 0
    warning_count = 0
    caution_count = 0
    healthy_count = 0

    for server_name, server_pred in server_preds.items():
        # PERFORMANCE: Use pre-calculated risk scores if provided (50-100x faster!)
        if _risk_scores is not None:
            risk = _risk_scores.get(server_name, 0)
        else:
            # Fallback to function call (slow path for backward compatibility)
            risk = _calculate_risk_func(server_pred) if _calculate_risk_func else 0

        if risk >= 80:  # Critical/Imminent Failure
            critical_count += 1
        elif risk >= 60:  # Danger/Warning
            warning_count += 1
        elif risk >= 50:  # Degrading
            caution_count += 1
        else:  # Healthy/Watch
            healthy_count += 1

    total = len(server_preds)

    # Determine status based on percentage of unhealthy servers
    critical_pct = critical_count / total if total > 0 else 0
    warning_pct = warning_count / total if total > 0 else 0
    unhealthy_pct = (critical_count + warning_count) / total if total > 0 else 0

    if critical_pct > 0.3 or unhealthy_pct > 0.5:  # >30% critical OR >50% unhealthy
        return "Critical", "red", "🔴"
    elif critical_pct > 0.1 or unhealthy_pct > 0.3:  # >10% critical OR >30% unhealthy
        return "Warning", "orange", "🟠"
    elif unhealthy_pct > 0.1:  # >10% unhealthy (degrading servers)
        return "Caution", "yellow", "🟡"
    else:
        return "Healthy", "green", "🟢"
