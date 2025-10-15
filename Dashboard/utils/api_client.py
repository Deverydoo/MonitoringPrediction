"""
Daemon API Client

Client for communicating with TFT Inference Daemon API.
"""

import requests
import streamlit as st
from typing import Dict, List, Optional


class DaemonClient:
    """Client for TFT Inference Daemon API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    def check_health(self) -> Dict:
        """Check daemon health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.ok:
                return {'connected': True, 'data': response.json()}
            return {'connected': False, 'error': f"HTTP {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {'connected': False, 'error': 'Connection refused'}
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    def get_predictions(self) -> Optional[Dict]:
        """Get current predictions from daemon."""
        try:
            response = requests.get(f"{self.base_url}/predictions/current", timeout=5)
            if response.ok:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
            return None

    def get_alerts(self) -> Optional[Dict]:
        """Get active alerts from daemon."""
        try:
            response = requests.get(f"{self.base_url}/alerts/active", timeout=5)
            if response.ok:
                return response.json()
            return None
        except Exception as e:
            return None

    def feed_data(self, records: List[Dict]) -> bool:
        """Feed data to daemon for demo mode."""
        try:
            response = requests.post(
                f"{self.base_url}/feed/data",
                json={"records": records},
                timeout=5
            )
            if not response.ok:
                st.error(f"Daemon returned {response.status_code}: {response.text[:200]}")
            return response.ok
        except Exception as e:
            st.error(f"Error feeding data: {e}")
            return False
