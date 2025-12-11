"""
Centralized Configuration Package for TFT Monitoring System

SINGLE SOURCE OF TRUTH for all configuration values.

Usage:
    from config import MODEL_CONFIG, METRICS_CONFIG, API_CONFIG, DASHBOARD_CONFIG

    # Access values
    batch_size = MODEL_CONFIG['batch_size']
    daemon_url = API_CONFIG['daemon_url']
    profile_baselines = METRICS_CONFIG['profile_baselines']

Module Organization:
    - model_config.py: TFT model hyperparameters, training settings
    - metrics_config.py: NordIQ Metrics Framework metrics, profile baselines, state multipliers
    - api_config.py: API URLs, ports, endpoints, timeouts
    - dashboard_config.py: Dashboard UI settings, thresholds, display config (kept in Dashboard/ for modularity)

Rules:
    1. ALL configuration values MUST be defined in these modules
    2. NO hardcoded values in application code
    3. Configuration changes happen ONLY in config/ modules
    4. Use ALL_CAPS for constants to indicate immutability
    5. Document every config value with inline comments
"""

from core.config.model_config import MODEL_CONFIG
from core.config.metrics_config import METRICS_CONFIG
from core.config.api_config import API_CONFIG

# Dashboard config stays in Dashboard/config for modular structure
# Import it via: from Dashboard.config.dashboard_config import *

__all__ = [
    'MODEL_CONFIG',
    'METRICS_CONFIG',
    'API_CONFIG',
]

__version__ = '2.0.0'  # Centralized config system v2.0
