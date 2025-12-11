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
