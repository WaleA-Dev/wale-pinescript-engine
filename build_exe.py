#!/usr/bin/env python3
"""
Build script for the Wale Backtest Engine standalone EXE (launcher + Flask).

Usage:
    python build_exe.py

Or use the spec directly:
    pyinstaller WaleBacktest.spec

Creates dist/WaleBacktest.exe — a launcher that starts the Flask server and opens the web dashboard.
"""

import subprocess
import sys
from pathlib import Path


def build():
    """Build the WaleBacktest launcher EXE using the spec file."""
    project_root = Path(__file__).resolve().parent
    spec = project_root / "WaleBacktest.spec"
    if not spec.exists():
        print("WaleBacktest.spec not found.")
        return 1
    print("=" * 60)
    print("Building Wale Backtest Engine EXE (launcher)")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", str(spec)],
        cwd=project_root,
        text=True,
    )
    if result.returncode == 0:
        print()
        print("=" * 60)
        print("BUILD SUCCESSFUL!")
        print(f"Executable: {project_root / 'dist' / 'WaleBacktest.exe'}")
        print("=" * 60)
        return 0
    print()
    print("=" * 60)
    print(f"BUILD FAILED (exit={result.returncode})")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(build())
