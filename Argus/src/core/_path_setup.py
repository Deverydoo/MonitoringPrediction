"""
Path setup for application.
This ensures all modules can find each other regardless of where they're run from.
"""

import sys
from pathlib import Path

# Get the src directory (parent of core/)
SRC_DIR = Path(__file__).parent.parent.resolve()

# Add src to path so we can import from core, daemons, etc.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Add src/core to path for backward compatibility with existing imports
CORE_DIR = SRC_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
