"""
Performance Monitoring Utilities
=================================

Tools for measuring and displaying dashboard performance.
"""

import time
from datetime import datetime
from typing import Tuple
import dash_bootstrap_components as dbc
from dash import html


class PerformanceTimer:
    """Context manager for measuring render performance."""

    def __init__(self, label: str = "Operation"):
        self.label = label
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        print(f"[PERF] {self.label}: {self.elapsed_ms:.0f}ms")

    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.elapsed_ms is None:
            return (time.time() - self.start_time) * 1000
        return self.elapsed_ms


def format_performance_badge(elapsed_ms: float, target_ms: float = 500) -> dbc.Badge:
    """
    Create a Bootstrap badge showing render performance.

    Args:
        elapsed_ms: Elapsed time in milliseconds
        target_ms: Target time in milliseconds (default 500)

    Returns:
        dbc.Badge: Colored badge component
    """
    if elapsed_ms < target_ms / 5:  # Excellent (<100ms if target is 500ms)
        color = "success"
        emoji = "⚡"
        label = f"{emoji} Render time: {elapsed_ms:.0f}ms (Excellent!)"
    elif elapsed_ms < target_ms:
        color = "success"
        emoji = "✅"
        label = f"{emoji} Render time: {elapsed_ms:.0f}ms (Target: <{target_ms:.0f}ms)"
    elif elapsed_ms < target_ms * 2:
        color = "warning"
        emoji = "⚠️"
        label = f"{emoji} Render time: {elapsed_ms:.0f}ms (Target: <{target_ms:.0f}ms)"
    else:
        color = "danger"
        emoji = "❌"
        label = f"{emoji} Render time: {elapsed_ms:.0f}ms (Target: <{target_ms:.0f}ms)"

    return dbc.Badge(
        label,
        color=color,
        className="mb-2"
    )


def log_performance(operation: str, elapsed_ms: float, details: str = ""):
    """
    Log performance metrics to console.

    Args:
        operation: Name of the operation
        elapsed_ms: Elapsed time in milliseconds
        details: Optional details to include
    """
    detail_str = f" ({details})" if details else ""
    print(f"[PERF] {operation}: {elapsed_ms:.0f}ms{detail_str}")
