# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for PineScript Backtest Engine.

Build with:
    pyinstaller PineScriptBacktester.spec
"""

import sys
from pathlib import Path

# Project root
project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / 'app.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'src'), 'src'),
        (str(project_root / 'data_providers'), 'data_providers'),
        (str(project_root / 'gui'), 'gui'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'numpy',
        'pandas',
        'scipy',
        'scipy.special._cdflib',
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'openpyxl',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PineScriptBacktester',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file if desired
)
