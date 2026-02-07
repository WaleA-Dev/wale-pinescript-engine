#!/usr/bin/env python3
"""
Build script for creating standalone EXE using PyInstaller.

Usage:
    python build_exe.py

This creates a standalone executable in the dist/ folder.
"""

import subprocess
import sys
import os
from pathlib import Path


def build():
    """Build the executable."""
    print("=" * 60)
    print("Building PineScript Backtest Engine EXE")
    print("=" * 60)

    # Get project root
    project_root = Path(__file__).parent

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=PineScriptBacktester",
        "--onefile",
        "--windowed",
        "--icon=NONE",  # Add icon path here if you have one
        f"--add-data={project_root / 'src'};src",
        f"--add-data={project_root / 'data_providers'};data_providers",
        f"--add-data={project_root / 'gui'};gui",
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=numpy",
        "--hidden-import=pandas",
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--collect-all=databento",
        "--clean",
        str(project_root / "app.py"),
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        print()
        print("=" * 60)
        print("BUILD SUCCESSFUL!")
        print(f"Executable: {project_root / 'dist' / 'PineScriptBacktester.exe'}")
        print("=" * 60)
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"BUILD FAILED: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(build())
