"""
Risk Scoring Engine

Executive-friendly contextual intelligence for server health assessment.
"""

import streamlit as st
from typing import Dict
from .profiles import get_server_profile
from .metrics import extract_cpu_used
from core.alert_levels import (
    get_alert_color,
    get_alert_emoji,
    get_alert_label,
    format_risk_display,
    AlertLevel
)


@st.cache_data(ttl=10)  # Cache for 10 seconds - expensive calculation
def calculate_server_risk_score(server_pred: Dict) -> float:
    """
    Calculate aggregated risk score for a server (0-100).

    EXECUTIVE-FRIENDLY SCORING:
    - Prioritizes CURRENT state (70% weight) - "What's on fire NOW?"
    - Considers PREDICTIONS (30% weight) - "What will be on fire soon?"
    - Only flags truly critical situations

    Args:
        server_pred: Server prediction dictionary with NordIQ Metrics Framework metrics

    Returns:
        Risk score 0-100 (higher = more urgent)
    """
    current_risk = 0.0  # Based on current metrics
    predicted_risk = 0.0  # Based on 30-min predictions

    profile = get_server_profile(server_pred.get('server_name', ''))

    # =========================================================================
    # CPU RISK ASSESSMENT (NordIQ Metrics Framework: using centralized helper)
    # =========================================================================
    current_cpu = extract_cpu_used(server_pred, 'current')
    max_cpu_p90 = extract_cpu_used(server_pred, 'p90')

    # CURRENT CPU RISK (what matters most to executives)
    if current_cpu >= 98:
        current_risk += 60  # CRITICAL - System will hang
    elif current_cpu >= 95:
        current_risk += 40  # Severe degradation
    elif current_cpu >= 90:
        current_risk += 20  # High load

    # PREDICTED CPU RISK (early warning)
    if max_cpu_p90 >= 98:
        predicted_risk += 30  # Will become critical
    elif max_cpu_p90 >= 95:
        predicted_risk += 20  # Will degrade
    elif max_cpu_p90 >= 90:
        predicted_risk += 10  # Will be high

    # =========================================================================
    # I/O WAIT RISK ASSESSMENT (CRITICAL - "System troubleshooting 101")
    # =========================================================================
    if 'cpu_iowait_pct' in server_pred:
        iowait = server_pred['cpu_iowait_pct']
        current_iowait = iowait.get('current', 0)
        p90 = iowait.get('p90', [])
        max_iowait_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_iowait

        # I/O wait is a critical indicator - high iowait = disk/storage bottleneck
        # CURRENT
        if current_iowait >= 30:
            current_risk += 50  # CRITICAL - severe I/O bottleneck
        elif current_iowait >= 20:
            current_risk += 30  # High I/O contention
        elif current_iowait >= 10:
            current_risk += 15  # Elevated I/O wait
        elif current_iowait >= 5:
            current_risk += 5   # Noticeable

        # PREDICTED
        if max_iowait_p90 >= 30:
            predicted_risk += 25  # Will have severe bottleneck
        elif max_iowait_p90 >= 20:
            predicted_risk += 15  # Will have high contention
        elif max_iowait_p90 >= 10:
            predicted_risk += 8   # Will be elevated

    # =========================================================================
    # MEMORY RISK ASSESSMENT (NordIQ Metrics Framework: mem_used_pct)
    # =========================================================================
    if 'mem_used_pct' in server_pred:
        mem = server_pred['mem_used_pct']
        current_mem = mem.get('current', 0)
        p90 = mem.get('p90', [])
        max_mem_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_mem

        # Profile-specific thresholds
        if profile == 'Database':
            # Databases: high memory is normal (page cache), only >95% concerning
            # CURRENT
            if current_mem >= 98:
                current_risk += 40  # Very high
            elif current_mem >= 95:
                current_risk += 15  # Elevated
            # PREDICTED
            if max_mem_p90 >= 98:
                predicted_risk += 20  # Will be very high
        else:
            # Non-DB servers: high memory = OOM risk
            # CURRENT
            if current_mem >= 98:
                current_risk += 60  # CRITICAL - OOM imminent
            elif current_mem >= 95:
                current_risk += 40  # High memory pressure
            elif current_mem >= 90:
                current_risk += 20  # Elevated
            # PREDICTED
            if max_mem_p90 >= 98:
                predicted_risk += 30  # Will OOM
            elif max_mem_p90 >= 95:
                predicted_risk += 20  # Will have pressure

    # =========================================================================
    # LOAD AVERAGE RISK ASSESSMENT (System load indicator)
    # =========================================================================
    if 'load_average' in server_pred:
        load = server_pred['load_average']
        current_load = load.get('current', 0)
        p90 = load.get('p90', [])
        max_load_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_load

        # Load average > 8 on typical servers indicates heavy queuing
        # CURRENT
        if current_load > 12:
            current_risk += 25  # Severe load
        elif current_load > 8:
            current_risk += 15  # High load
        elif current_load > 6:
            current_risk += 5   # Elevated
        # PREDICTED
        if max_load_p90 > 12:
            predicted_risk += 12  # Will be severely loaded
        elif max_load_p90 > 8:
            predicted_risk += 8   # Will be high

    # =========================================================================
    # WEIGHTED FINAL SCORE
    # =========================================================================
    # 70% current state (executives care about NOW)
    # 30% predictions (early warning value)
    final_risk = (current_risk * 0.7) + (predicted_risk * 0.3)

    return min(final_risk, 100)


def get_risk_color(risk_score: float) -> str:
    """
    Get color for risk score visualization.
    DEPRECATED: Use core.alert_levels.get_alert_color() instead.

    This function remains for backward compatibility but delegates to
    the centralized alert_levels module.

    Args:
        risk_score: Risk score 0-100

    Returns:
        Hex color code

    Alert Levels:
        >= 70: Critical (Red #ff4444)
        >= 40: Warning (Orange #ff9900)
        >= 20: Watch (Yellow #ffcc00)
        <  20: Healthy (Green #44ff44)
    """
    # Delegate to centralized alert levels system
    return get_alert_color(risk_score, format="hex")
