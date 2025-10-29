"""
Dashboard utilities - shared helper functions.

Includes risk scoring, metric extraction, profile detection, and API client.
"""

from .risk_scoring import calculate_server_risk_score, get_risk_color
from .metrics import extract_cpu_used, get_health_status, get_metric_color_indicator
from .profiles import get_server_profile
from .api_client import DaemonClient

__all__ = [
    'calculate_server_risk_score',
    'get_risk_color',
    'extract_cpu_used',
    'get_health_status',
    'get_metric_color_indicator',
    'get_server_profile',
    'DaemonClient'
]
