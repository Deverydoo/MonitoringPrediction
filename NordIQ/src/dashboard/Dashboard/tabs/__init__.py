"""
Dashboard Tabs - Modular tab components for TFT Monitoring Dashboard

Each tab module provides a render() function that takes predictions and renders the tab content.
"""

from . import overview
from . import heatmap
from . import top_risks
from . import historical
from . import cost_avoidance
from . import auto_remediation
from . import alerting
from . import advanced
from . import documentation
from . import roadmap

__all__ = [
    'overview',
    'heatmap',
    'top_risks',
    'historical',
    'cost_avoidance',
    'auto_remediation',
    'alerting',
    'advanced',
    'documentation',
    'roadmap'
]
