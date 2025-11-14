"""
Dash Production Configuration
=============================

Central configuration for the Dash dashboard application.
"""

import os
from typing import Optional

# =============================================================================
# DAEMON CONFIGURATION
# =============================================================================

DAEMON_URL = os.getenv("DAEMON_URL", "http://localhost:8000")
METRICS_GENERATOR_URL = os.getenv("METRICS_GENERATOR_URL", "http://localhost:8001")

# API Key for daemon authentication (X-API-Key header)
DAEMON_API_KEY = os.getenv("TFT_API_KEY", "")

if not DAEMON_API_KEY:
    print("[WARN] No API key configured - running in development mode")
    print("[WARN] Dashboard will only work if daemon has no TFT_API_KEY set")
    print("[INFO] For production: set TFT_API_KEY environment variable")

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

# Refresh interval for auto-refresh (milliseconds)
REFRESH_INTERVAL_DEFAULT = 30000  # 30 seconds (default)
REFRESH_INTERVAL_MIN = 5000  # 5 seconds (minimum)
REFRESH_INTERVAL_MAX = 300000  # 5 minutes (maximum)

# Performance targets
RENDER_TIME_TARGET_MS = 500  # Target render time
RENDER_TIME_EXCELLENT_MS = 100  # Excellent performance threshold

# =============================================================================
# BRANDING - WELLS FARGO
# =============================================================================

BRAND_NAME = "Wells Fargo"
BRAND_COLOR_PRIMARY = "#D71E28"  # Wells Fargo Red
BRAND_COLOR_SECONDARY = "#FFCD41"  # Wells Fargo Gold
BRAND_COLOR_BACKGROUND = "#FFFFFF"
BRAND_COLOR_TEXT = "#333333"

# Logo URLs (if available)
BRAND_LOGO_URL = None  # Add logo URL if available
BRAND_FAVICON_URL = None  # Add favicon URL if available

# =============================================================================
# APP METADATA
# =============================================================================

APP_TITLE = "ArgusAI - Predictive Infrastructure Monitoring"
APP_DESCRIPTION = "Predictive System Monitoring"
APP_VERSION = "2.0.0-dash"  # Version 2.0 = Dash migration
APP_COPYRIGHT = "Built by Craig Giannelli and Claude Code"

# =============================================================================
# FEATURE FLAGS
# =============================================================================

ENABLE_AUTO_REFRESH = True
ENABLE_PERFORMANCE_TIMER = True
ENABLE_DEBUG_MODE = os.getenv("DEBUG", "").lower() == "true"

# =============================================================================
# DEVELOPMENT vs PRODUCTION
# =============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"
IS_DEVELOPMENT = ENVIRONMENT == "development"

# Development mode settings
if IS_DEVELOPMENT:
    DEBUG = True
    DEV_TOOLS_HOT_RELOAD = True
else:
    DEBUG = False
    DEV_TOOLS_HOT_RELOAD = False

# =============================================================================
# PATHS
# =============================================================================

import sys
from pathlib import Path

# Add src/ to path for imports
NORDIQ_ROOT = Path(__file__).resolve().parent
NORDIQ_SRC = NORDIQ_ROOT / "src"
sys.path.insert(0, str(NORDIQ_SRC))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_auth_headers() -> dict:
    """Get authentication headers with API key if configured."""
    headers = {"Content-Type": "application/json"}
    if DAEMON_API_KEY:
        headers["X-API-Key"] = DAEMON_API_KEY
    return headers


def format_render_time(elapsed_ms: float) -> tuple[str, str]:
    """
    Format render time with color coding.

    Returns:
        tuple: (message, color)
            - message: Formatted message with emoji
            - color: "success", "warning", or "danger"
    """
    if elapsed_ms < RENDER_TIME_EXCELLENT_MS:
        return (f"⚡ Render time: {elapsed_ms:.0f}ms (Excellent!)", "success")
    elif elapsed_ms < RENDER_TIME_TARGET_MS:
        return (f"⚡ Render time: {elapsed_ms:.0f}ms (Target: <{RENDER_TIME_TARGET_MS}ms)", "success")
    elif elapsed_ms < RENDER_TIME_TARGET_MS * 2:
        return (f"⚡ Render time: {elapsed_ms:.0f}ms (Target: <{RENDER_TIME_TARGET_MS}ms)", "warning")
    else:
        return (f"⚡ Render time: {elapsed_ms:.0f}ms (Target: <{RENDER_TIME_TARGET_MS}ms)", "danger")


# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = f"""
/* Wells Fargo Branding */
.navbar {{
    background-color: {BRAND_COLOR_PRIMARY} !important;
}}

.brand-header {{
    color: {BRAND_COLOR_PRIMARY};
    font-weight: bold;
}}

/* Performance badge */
.performance-badge {{
    font-size: 14px;
    font-weight: bold;
    padding: 8px 16px;
    border-radius: 4px;
    margin-bottom: 16px;
}}

/* Card styling */
.card {{
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 16px;
}}

/* Tab styling */
.nav-tabs .nav-link.active {{
    background-color: {BRAND_COLOR_PRIMARY} !important;
    color: white !important;
    border-color: {BRAND_COLOR_PRIMARY} !important;
}}

.nav-tabs .nav-link {{
    color: {BRAND_COLOR_PRIMARY};
}}

.nav-tabs .nav-link:hover {{
    border-color: {BRAND_COLOR_SECONDARY};
}}
"""

# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    'DAEMON_URL',
    'DAEMON_API_KEY',
    'REFRESH_INTERVAL',
    'BRAND_NAME',
    'BRAND_COLOR_PRIMARY',
    'APP_TITLE',
    'APP_VERSION',
    'get_auth_headers',
    'format_render_time',
    'CUSTOM_CSS',
]
