"""
Dashboard Configuration

All constants, thresholds, and configuration values for the TFT Monitoring Dashboard.

NOTE: API URLs and ports are imported from centralized config.api_config
"""

import os
import sys
from pathlib import Path

# Add src/ to path to find core.config module
NORDIQ_SRC = Path(__file__).resolve().parents[3]  # Go up to NordIQ/src/
if str(NORDIQ_SRC) not in sys.path:
    sys.path.insert(0, str(NORDIQ_SRC))

# Import API configuration from centralized config (SINGLE SOURCE OF TRUTH)
from core.config.api_config import API_CONFIG

# =============================================================================
# API Configuration
# =============================================================================

DAEMON_URL = API_CONFIG['daemon_url']
METRICS_GENERATOR_URL = API_CONFIG['metrics_generator_url']
REFRESH_INTERVAL = API_CONFIG['dashboard']['default_refresh_interval']  # 5 seconds

# =============================================================================
# Security Configuration
# =============================================================================

# API Key for daemon authentication (X-API-Key header)
# Priority: Streamlit secrets > Environment variable > Empty (dev mode)
try:
    import streamlit as st
    # Streamlit secrets uses dict-like access, not .get()
    if "daemon" in st.secrets and "api_key" in st.secrets["daemon"]:
        DAEMON_API_KEY = st.secrets["daemon"]["api_key"]
    else:
        DAEMON_API_KEY = ""
except Exception as e:
    print(f"[DEBUG] Could not load from st.secrets: {e}")
    DAEMON_API_KEY = ""

# Fallback to environment variable if not in Streamlit secrets
if not DAEMON_API_KEY:
    DAEMON_API_KEY = os.getenv("TFT_API_KEY", "")

# Warn if no API key configured (development mode)
if not DAEMON_API_KEY:
    print("[INFO] No API key configured - running in development mode")
    print("[INFO] Dashboard will only work if daemon has no TFT_API_KEY set")
    print("[INFO] For production: set TFT_API_KEY env var or add to .streamlit/secrets.toml")

# =============================================================================
# Risk Thresholds
# =============================================================================

RISK_THRESHOLDS = {
    'imminent_failure': 90,
    'critical': 80,
    'danger': 70,
    'warning': 60,
    'degrading': 50,
    'healthy': 0
}

# =============================================================================
# CPU Thresholds (% Used)
# =============================================================================

CPU_THRESHOLDS = {
    'critical': 98,   # 🔴
    'danger': 95,     # 🟠
    'warning': 90,    # 🟡
    'healthy': 0
}

# =============================================================================
# Memory Thresholds
# =============================================================================

# Standard profiles (ML Compute, Web API, etc.)
MEMORY_THRESHOLDS_STANDARD = {
    'critical': 98,   # 🔴
    'danger': 95,     # 🟠
    'warning': 90,    # 🟡
    'healthy': 0
}

# Database profile (higher thresholds - page cache is normal)
MEMORY_THRESHOLDS_DATABASE = {
    'critical': 99,   # 🔴 (databases cache aggressively)
    'danger': 98,     # 🟠
    'warning': 95,    # 🟡
    'healthy': 0
}

# =============================================================================
# I/O Wait Thresholds (CRITICAL - "System troubleshooting 101")
# =============================================================================

IOWAIT_THRESHOLDS = {
    'critical': 30,   # 🔴 Severe I/O bottleneck
    'danger': 20,     # 🟠 High I/O contention
    'warning': 10,    # 🟡 Elevated I/O wait
    'noticeable': 5,  # 🟡 Noticeable
    'healthy': 0
}

# =============================================================================
# Swap Thresholds (Thrashing Indicator)
# =============================================================================

SWAP_THRESHOLDS = {
    'critical': 50,   # 🔴 Heavy swap thrashing
    'danger': 25,     # 🟠 Significant swap usage
    'warning': 10,    # 🟡 Swap in use
    'healthy': 0
}

# =============================================================================
# Load Average Thresholds
# =============================================================================

LOAD_THRESHOLDS = {
    'severe': 12,     # 🔴 Severe overload
    'high': 8,        # 🟠 High load
    'elevated': 6,    # 🟡 Elevated
    'healthy': 0
}

# =============================================================================
# Server Profile Patterns
# =============================================================================

SERVER_PROFILES = {
    'ppml': 'ML Compute',
    'ppdb': 'Database',
    'ppapi': 'Web API',
    'ppcond': 'Conductor Mgmt',
    'ppetl': 'ETL/Ingest',
    'pprisk': 'Risk Analytics',
    'default': 'Generic'
}

# =============================================================================
# Risk Scoring Weights (Executive-Friendly Contextual Intelligence)
# =============================================================================

RISK_WEIGHTS = {
    'current_state': 0.70,      # Current metrics weighted 70%
    'predictions': 0.30,        # Future predictions weighted 30%
    'cpu_weight': 0.25,         # CPU contribution to risk
    'memory_weight': 0.25,      # Memory contribution to risk
    'iowait_weight': 0.20,      # I/O wait contribution to risk
    'swap_weight': 0.15,        # Swap contribution to risk
    'load_weight': 0.15         # Load average contribution to risk
}

# =============================================================================
# Cache TTL Values (seconds)
# =============================================================================

CACHE_TTL = {
    'real_time_data': 5,        # Health status, current predictions
    'computed_metrics': 10,     # Risk scores, CPU extraction
    'static_logic': 60,         # Color indicators, thresholds
    'profile_detection': 300    # Server profiles (rarely change)
}

# =============================================================================
# Dashboard Display Configuration
# =============================================================================

PAGE_CONFIG = {
    'page_title': '🔮 TFT Monitoring Dashboard',
    'page_icon': '🔮',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Number of servers to show in "Top N" displays
TOP_N_SERVERS = 5

# Color indicators
COLOR_INDICATORS = {
    'critical': '🔴',
    'danger': '🟠',
    'warning': '🟡',
    'healthy': ''
}

# Status emoji
STATUS_EMOJI = {
    'Critical': '🔴',
    'Degraded': '🟡',
    'Healthy': '🟢',
    'Unknown': '❓'
}

# =============================================================================
# Demo Mode Configuration
# =============================================================================

DEMO_SCENARIOS = ['healthy', 'degrading', 'critical']
DEFAULT_FLEET_SIZE = 90
DEMO_TICK_INTERVAL = 5  # seconds

# =============================================================================
# Profile-Specific Baselines (for context-aware alerts)
# =============================================================================

PROFILE_BASELINES = {
    'ML Compute': {
        'cpu_normal': (45, 75),      # Expected range
        'memory_normal': (65, 85),
        'iowait_normal': (0, 5),     # Should be compute-bound
        'load_normal': (4, 10)
    },
    'Database': {
        'cpu_normal': (40, 65),
        'memory_normal': (80, 98),   # High memory is normal (page cache)
        'iowait_normal': (5, 20),    # I/O intensive workload
        'load_normal': (2, 8)
    },
    'Web API': {
        'cpu_normal': (20, 40),
        'memory_normal': (30, 60),
        'iowait_normal': (0, 5),
        'load_normal': (1, 6)
    },
    'Conductor Mgmt': {
        'cpu_normal': (15, 35),
        'memory_normal': (40, 65),
        'iowait_normal': (0, 5),
        'load_normal': (1, 4)
    },
    'ETL/Ingest': {
        'cpu_normal': (50, 80),
        'memory_normal': (60, 85),
        'iowait_normal': (5, 15),    # I/O intensive
        'load_normal': (4, 10)
    },
    'Risk Analytics': {
        'cpu_normal': (55, 85),
        'memory_normal': (70, 90),
        'iowait_normal': (0, 8),
        'load_normal': (6, 14)
    },
    'Generic': {
        'cpu_normal': (30, 60),
        'memory_normal': (40, 70),
        'iowait_normal': (0, 10),
        'load_normal': (2, 8)
    }
}
