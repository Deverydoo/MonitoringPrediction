"""
ArgusAI - Predictive Infrastructure Monitoring

Built by Craig Giannelli and Claude Code
"""

import sys
from pathlib import Path

# Add src directory to Python path
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

__version__ = "1.1.0"
__company__ = "ArgusAI, LLC"
