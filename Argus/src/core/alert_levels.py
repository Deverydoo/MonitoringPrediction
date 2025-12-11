#!/usr/bin/env python3
"""
Alert Levels - ArgusAI Centralized Alert Level System
Single source of truth for risk scores, colors, labels, and emojis

Version: 1.2.1
Built by Craig Giannelli and Claude Code
"""

from enum import Enum
from typing import Dict, Tuple


class AlertLevel(Enum):
    """
    Standardized alert levels for server health monitoring.

    These levels represent the severity of server issues based on risk scores (0-100).
    Risk scores combine current state (70%) + predictions (30%).
    """
    HEALTHY = "healthy"          # 0-19: Normal operations
    WATCH = "watch"              # 20-39: Minor concerns, trending
    WARNING = "warning"          # 40-69: Significant issues, needs attention
    CRITICAL = "critical"        # 70-100: Severe issues, immediate action required


# =============================================================================
# ALERT LEVEL THRESHOLDS
# =============================================================================

ALERT_THRESHOLDS = {
    AlertLevel.CRITICAL: 70,   # >= 70 = Critical
    AlertLevel.WARNING: 40,    # >= 40 = Warning
    AlertLevel.WATCH: 20,      # >= 20 = Watch
    AlertLevel.HEALTHY: 0,     # >= 0 = Healthy
}


# =============================================================================
# COLOR PALETTE (HEX)
# =============================================================================

ALERT_COLORS_HEX = {
    AlertLevel.CRITICAL: "#ff4444",   # Bright Red
    AlertLevel.WARNING: "#ff9900",    # Orange
    AlertLevel.WATCH: "#ffcc00",      # Yellow
    AlertLevel.HEALTHY: "#44ff44",    # Green
}


# =============================================================================
# COLOR PALETTE (RGB)
# =============================================================================

ALERT_COLORS_RGB = {
    AlertLevel.CRITICAL: (255, 68, 68),      # Bright Red
    AlertLevel.WARNING: (255, 153, 0),       # Orange
    AlertLevel.WATCH: (255, 204, 0),         # Yellow
    AlertLevel.HEALTHY: (68, 255, 68),       # Green
}


# =============================================================================
# EMOJIS
# =============================================================================

ALERT_EMOJIS = {
    AlertLevel.CRITICAL: "🔴",    # Red circle
    AlertLevel.WARNING: "🟠",     # Orange circle
    AlertLevel.WATCH: "🟡",       # Yellow circle
    AlertLevel.HEALTHY: "🟢",     # Green circle
}


# =============================================================================
# DISPLAY LABELS
# =============================================================================

ALERT_LABELS = {
    AlertLevel.CRITICAL: "Critical",
    AlertLevel.WARNING: "Warning",
    AlertLevel.WATCH: "Watch",
    AlertLevel.HEALTHY: "Healthy",
}


# Alternative labels for specific contexts
ALERT_LABELS_SHORT = {
    AlertLevel.CRITICAL: "CRIT",
    AlertLevel.WARNING: "WARN",
    AlertLevel.WATCH: "WATCH",
    AlertLevel.HEALTHY: "OK",
}


# Action-oriented labels (for executive dashboards)
ALERT_LABELS_ACTION = {
    AlertLevel.CRITICAL: "Act Now",
    AlertLevel.WARNING: "Investigate",
    AlertLevel.WATCH: "Monitor",
    AlertLevel.HEALTHY: "Normal",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_alert_level(risk_score: float) -> AlertLevel:
    """
    Determine alert level from risk score (0-100).

    Args:
        risk_score: Risk score 0-100 (higher = more urgent)

    Returns:
        AlertLevel enum value

    Examples:
        >>> get_alert_level(58)
        AlertLevel.WARNING
        >>> get_alert_level(15)
        AlertLevel.HEALTHY
        >>> get_alert_level(75)
        AlertLevel.CRITICAL
    """
    if risk_score >= ALERT_THRESHOLDS[AlertLevel.CRITICAL]:
        return AlertLevel.CRITICAL
    elif risk_score >= ALERT_THRESHOLDS[AlertLevel.WARNING]:
        return AlertLevel.WARNING
    elif risk_score >= ALERT_THRESHOLDS[AlertLevel.WATCH]:
        return AlertLevel.WATCH
    else:
        return AlertLevel.HEALTHY


def get_alert_color(risk_score: float, format: str = "hex") -> str:
    """
    Get color for risk score visualization.

    Args:
        risk_score: Risk score 0-100
        format: "hex" or "rgb"

    Returns:
        Color code in requested format

    Examples:
        >>> get_alert_color(58)
        '#ff9900'
        >>> get_alert_color(58, format="rgb")
        (255, 153, 0)
    """
    level = get_alert_level(risk_score)

    if format == "rgb":
        return ALERT_COLORS_RGB[level]
    else:  # default to hex
        return ALERT_COLORS_HEX[level]


def get_alert_emoji(risk_score: float) -> str:
    """
    Get emoji for risk score visualization.

    Args:
        risk_score: Risk score 0-100

    Returns:
        Emoji character

    Examples:
        >>> get_alert_emoji(58)
        '🟠'
    """
    level = get_alert_level(risk_score)
    return ALERT_EMOJIS[level]


def get_alert_label(risk_score: float, style: str = "default") -> str:
    """
    Get label for risk score visualization.

    Args:
        risk_score: Risk score 0-100
        style: "default", "short", or "action"

    Returns:
        Label string

    Examples:
        >>> get_alert_label(58)
        'Warning'
        >>> get_alert_label(58, style="short")
        'WARN'
        >>> get_alert_label(58, style="action")
        'Investigate'
    """
    level = get_alert_level(risk_score)

    if style == "short":
        return ALERT_LABELS_SHORT[level]
    elif style == "action":
        return ALERT_LABELS_ACTION[level]
    else:
        return ALERT_LABELS[level]


def format_risk_display(risk_score: float, include_emoji: bool = True,
                        label_style: str = "default") -> str:
    """
    Format risk score for display with emoji and label.

    Args:
        risk_score: Risk score 0-100
        include_emoji: Whether to include emoji
        label_style: "default", "short", or "action"

    Returns:
        Formatted string

    Examples:
        >>> format_risk_display(58)
        '🟠 Warning'
        >>> format_risk_display(58, include_emoji=False)
        'Warning'
        >>> format_risk_display(58, label_style="action")
        '🟠 Investigate'
    """
    label = get_alert_label(risk_score, style=label_style)

    if include_emoji:
        emoji = get_alert_emoji(risk_score)
        return f"{emoji} {label}"
    else:
        return label


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_alert_system():
    """Validate alert level system integrity."""
    # Test all thresholds
    assert get_alert_level(0) == AlertLevel.HEALTHY
    assert get_alert_level(15) == AlertLevel.HEALTHY
    assert get_alert_level(19) == AlertLevel.HEALTHY
    assert get_alert_level(20) == AlertLevel.WATCH
    assert get_alert_level(35) == AlertLevel.WATCH
    assert get_alert_level(39) == AlertLevel.WATCH
    assert get_alert_level(40) == AlertLevel.WARNING
    assert get_alert_level(58) == AlertLevel.WARNING  # User's example!
    assert get_alert_level(69) == AlertLevel.WARNING
    assert get_alert_level(70) == AlertLevel.CRITICAL
    assert get_alert_level(100) == AlertLevel.CRITICAL

    # Test all levels have definitions
    for level in AlertLevel:
        assert level in ALERT_COLORS_HEX
        assert level in ALERT_COLORS_RGB
        assert level in ALERT_EMOJIS
        assert level in ALERT_LABELS
        assert level in ALERT_LABELS_SHORT
        assert level in ALERT_LABELS_ACTION
        assert level in ALERT_THRESHOLDS

    print("[OK] Alert system validation passed")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enum
    'AlertLevel',
    # Thresholds
    'ALERT_THRESHOLDS',
    # Colors
    'ALERT_COLORS_HEX',
    'ALERT_COLORS_RGB',
    # Emojis
    'ALERT_EMOJIS',
    # Labels
    'ALERT_LABELS',
    'ALERT_LABELS_SHORT',
    'ALERT_LABELS_ACTION',
    # Helper functions
    'get_alert_level',
    'get_alert_color',
    'get_alert_emoji',
    'get_alert_label',
    'format_risk_display',
]


if __name__ == "__main__":
    print("=" * 70)
    print("NordIQ Alert Levels System v1.2.1")
    print("=" * 70)

    # Run validation
    _validate_alert_system()

    print("\n" + "=" * 70)
    print("ALERT LEVELS:")
    print("=" * 70)

    for level in [AlertLevel.HEALTHY, AlertLevel.WATCH, AlertLevel.WARNING, AlertLevel.CRITICAL]:
        threshold = ALERT_THRESHOLDS[level]
        color = ALERT_COLORS_HEX[level]
        emoji = ALERT_EMOJIS[level]
        label = ALERT_LABELS[level]

        print(f"{emoji} {label:10} >= {threshold:3} | Color: {color}")

    print("\n" + "=" * 70)
    print("EXAMPLE: Risk Score 58 (User's Question)")
    print("=" * 70)
    print(f"Level: {get_alert_level(58)}")
    print(f"Color: {get_alert_color(58)}")
    print(f"Emoji: {get_alert_emoji(58)}")
    print(f"Label: {get_alert_label(58)}")
    print(f"Display: {format_risk_display(58)}")
    print(f"Display (action): {format_risk_display(58, label_style='action')}")
