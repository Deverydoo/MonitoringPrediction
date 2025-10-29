"""
Data Processing Utilities
==========================

Functions for extracting and processing prediction data.
"""

import numpy as np
from typing import Dict, Optional
from functools import lru_cache


def extract_cpu_used(server_pred: Dict, metric_type: str = 'current') -> float:
    """
    Extract CPU used % (100 - idle).

    Args:
        server_pred: Server prediction dict
        metric_type: 'current', 'p50', or 'p90'

    Returns:
        float: CPU used percentage (0-100)
    """
    if 'cpu_idle_pct' in server_pred:
        idle_metric = server_pred['cpu_idle_pct']
        if metric_type == 'current':
            idle = idle_metric.get('current', 0)
            return 100 - idle
        elif metric_type in ['p50', 'p90']:
            percentile = idle_metric.get(metric_type, [])
            if percentile and len(percentile) >= 6:
                avg_idle = np.mean(percentile[:6])
                return 100 - avg_idle
    return 0


def calculate_server_risk_score(server_pred: Dict) -> float:
    """
    Calculate risk score for a server (0-100).

    NOTE: In production, this should EXTRACT pre-calculated risk_score
    from daemon (proper architecture). This function is a fallback for
    backward compatibility.

    Args:
        server_pred: Server prediction dict

    Returns:
        float: Risk score (0-100)
    """
    # OPTIMIZATION: Extract pre-calculated risk score from daemon (Phase 3)
    if 'risk_score' in server_pred:
        return server_pred['risk_score']

    # FALLBACK: Calculate client-side (backward compatible)
    # This shouldn't happen with Phase 3 daemon
    current_risk = 0.0
    predicted_risk = 0.0

    # CPU Risk
    current_cpu = extract_cpu_used(server_pred, 'current')
    max_cpu_p90 = extract_cpu_used(server_pred, 'p90')

    if current_cpu >= 98:
        current_risk += 60
    elif current_cpu >= 95:
        current_risk += 40
    elif current_cpu >= 90:
        current_risk += 20

    if max_cpu_p90 >= 98:
        predicted_risk += 30
    elif max_cpu_p90 >= 95:
        predicted_risk += 20

    # I/O Wait Risk
    if 'cpu_iowait_pct' in server_pred:
        iowait = server_pred['cpu_iowait_pct']
        current_iowait = iowait.get('current', 0)
        p90 = iowait.get('p90', [])
        max_iowait_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_iowait

        if current_iowait >= 30:
            current_risk += 50
        elif current_iowait >= 20:
            current_risk += 30
        elif current_iowait >= 10:
            current_risk += 15

        if max_iowait_p90 >= 30:
            predicted_risk += 25
        elif max_iowait_p90 >= 20:
            predicted_risk += 15

    # Memory Risk
    if 'mem_used_pct' in server_pred:
        mem = server_pred['mem_used_pct']
        current_mem = mem.get('current', 0)
        p90 = mem.get('p90', [])
        max_mem_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_mem

        if current_mem >= 98:
            current_risk += 60
        elif current_mem >= 95:
            current_risk += 40
        elif current_mem >= 90:
            current_risk += 20

        if max_mem_p90 >= 98:
            predicted_risk += 30
        elif max_mem_p90 >= 95:
            predicted_risk += 20

    # Weighted final score (70% current, 30% predicted)
    final_risk = (current_risk * 0.7) + (predicted_risk * 0.3)
    return min(final_risk, 100)


def get_risk_color(risk_score: float) -> str:
    """
    Get color hex code for risk score.

    Args:
        risk_score: Risk score (0-100)

    Returns:
        str: Hex color code
    """
    if risk_score >= 80:
        return '#ff4444'  # Red
    elif risk_score >= 60:
        return '#ff9900'  # Orange
    elif risk_score >= 50:
        return '#ffcc00'  # Yellow
    else:
        return '#44ff44'  # Green


def extract_risk_scores(server_preds: Dict) -> Dict[str, float]:
    """
    Extract pre-calculated risk scores from daemon for all servers.

    ARCHITECTURAL NOTE: Daemon should pre-calculate all risk scores (Phase 3).
    Dashboard's job is to EXTRACT, not CALCULATE.

    Args:
        server_preds: Dictionary of server predictions

    Returns:
        dict: Mapping of server_name -> risk_score
    """
    risk_scores = {}
    for server_name, server_pred in server_preds.items():
        # Extract pre-calculated risk_score from daemon
        if 'risk_score' in server_pred:
            risk_scores[server_name] = server_pred['risk_score']
        else:
            # Fallback: calculate client-side (shouldn't happen with Phase 3 daemon)
            print(f"[WARN] Server {server_name} missing risk_score - calculating client-side")
            risk_scores[server_name] = calculate_server_risk_score(server_pred)
    return risk_scores
