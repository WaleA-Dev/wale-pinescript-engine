#!/usr/bin/env python3
"""
PineScript Backtest Engine - Desktop Application

A professional desktop application for backtesting TradingView
PineScript strategies locally with Databento data integration.

Usage:
    python app.py

Or run as executable after building with PyInstaller.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Pre-import problematic modules BEFORE PySide6 to avoid shiboken/dateutil conflict
# in PyInstaller builds. PySide6's shibokensupport hooks into imports and tries to
# inspect source code, which fails in frozen apps.
# Order matters: dateutil first, then pandas, then matplotlib with Agg backend.
try:
    import dateutil.tz  # noqa: F401
    import pandas  # noqa: F401
    import matplotlib  # noqa: F401
    matplotlib.use('Agg')
    import matplotlib.pyplot  # noqa: F401
except Exception:
    pass

def main():
    """Launch the desktop application."""
    from gui.main_window import main as gui_main
    gui_main()


if __name__ == '__main__':
    main()
