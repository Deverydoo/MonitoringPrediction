"""
API Client for Daemon Communication
====================================

Handles all communication with the TFT Inference Daemon.
"""

import requests
from typing import Optional, Dict
from dash_config import DAEMON_URL, get_auth_headers


def fetch_predictions() -> Optional[Dict]:
    """
    Fetch current predictions from daemon.

    Returns:
        dict: Predictions data if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{DAEMON_URL}/predictions/current",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        elif response.status_code == 403:
            print("[ERROR] Authentication failed - check TFT_API_KEY environment variable")
            return None
        else:
            print(f"[ERROR] Daemon returned status {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Daemon request timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching predictions: {e}")
        return None


def check_daemon_health() -> bool:
    """
    Check if daemon is connected and healthy.

    Returns:
        bool: True if daemon is healthy, False otherwise
    """
    try:
        response = requests.get(f"{DAEMON_URL}/health", timeout=2)
        return response.ok
    except:
        return False


def fetch_alerts() -> Optional[Dict]:
    """
    Fetch current alerts from daemon.

    Returns:
        dict: Alerts data if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{DAEMON_URL}/alerts",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching alerts: {e}")
        return None


# =============================================================================
# CASCADE DETECTION ENDPOINTS
# =============================================================================

def fetch_cascade_status() -> Optional[Dict]:
    """
    Fetch cascading failure detection status.

    Returns full cascade detection status including:
    - Current cascade state
    - Recent cascade events
    - Correlation scores
    - Affected servers

    Returns:
        dict: Cascade status if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{DAEMON_URL}/cascade/status",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        elif response.status_code == 404:
            # Endpoint not available (older daemon version)
            return None
        else:
            print(f"[ERROR] Cascade status returned {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Cascade status request timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching cascade status: {e}")
        return None


def fetch_cascade_health() -> Optional[Dict]:
    """
    Fetch fleet health score based on correlation analysis.

    Returns simple fleet-wide health metrics:
    - health_score: 0-100 (higher = healthier)
    - status: healthy/degraded/warning/critical
    - correlation_score: 0-1 (higher = more correlated issues)
    - anomaly_rate: % of servers with anomalies
    - cascade_risk: low/medium/high

    Returns:
        dict: Fleet health if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{DAEMON_URL}/cascade/health",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            print(f"[ERROR] Cascade health returned {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Cascade health request timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching cascade health: {e}")
        return None


# =============================================================================
# DRIFT MONITORING ENDPOINTS
# =============================================================================

def fetch_drift_status() -> Optional[Dict]:
    """
    Fetch model drift detection status.

    Returns current drift metrics:
    - drift_detected: bool
    - auto_retrain_enabled: bool
    - metrics: {per, dss, fds, anomaly_rate}
    - thresholds: configured thresholds for each metric

    Returns:
        dict: Drift status if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{DAEMON_URL}/drift/status",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            print(f"[ERROR] Drift status returned {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Drift status request timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching drift status: {e}")
        return None


def fetch_drift_report() -> Optional[Dict]:
    """
    Fetch detailed drift analysis report.

    Returns comprehensive drift analysis:
    - overall_health: good/warning/critical
    - needs_retraining: bool
    - metrics: detailed per-metric breakdown
    - feature_drift: per-feature drift scores
    - recommendations: list of suggested actions
    - auto_retrain: retraining status info

    Returns:
        dict: Drift report if successful, None otherwise
    """
    try:
        response = requests.get(
            f"{DAEMON_URL}/drift/report",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            print(f"[ERROR] Drift report returned {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Drift report request timed out")
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching drift report: {e}")
        return None
