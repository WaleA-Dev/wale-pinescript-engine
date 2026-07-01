# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the WaleBacktest launcher EXE.

Builds launcher.py (Tk launcher + embedded Flask web backtester) into a
single windowed executable. Paths are relative, so this works from any
checkout location:

    pyinstaller WaleBacktest.spec
    ./dist/WaleBacktest.exe
"""

import os

ROOT = os.path.abspath(os.getcwd())

datas = [
    (os.path.join(ROOT, "src"), "src"),
    (os.path.join(ROOT, "templates"), "templates"),
]

hiddenimports = [
    "flask",
    "numpy",
    "pandas",
    "pandas._libs.tslibs.timezones",
    "dateutil.tz",
    "scipy",
    "matplotlib",
    "matplotlib.backends.backend_agg",
    "yfinance",
]

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WaleBacktest",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
