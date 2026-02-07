# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('C:\\Users\\wale\\wale-pinescript-engine\\src', 'src'), ('C:\\Users\\wale\\wale-pinescript-engine\\data_providers', 'data_providers'), ('C:\\Users\\wale\\wale-pinescript-engine\\gui', 'gui')]
binaries = []
hiddenimports = ['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'numpy', 'pandas', 'pandas._libs.tslibs.timezones', 'dateutil.tz', 'scipy', 'matplotlib', 'matplotlib.backends.backend_agg']
tmp_ret = collect_all('databento')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['C:\\Users\\wale\\wale-pinescript-engine\\app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['C:\\Users\\wale\\wale-pinescript-engine\\hooks\\pyi_rth_pyside6_patch.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)
