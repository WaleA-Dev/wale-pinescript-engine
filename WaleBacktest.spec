# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    'flask', 'flask.json', 'jinja2', 'markupsafe', 'werkzeug',
    'yfinance', 'requests', 'scipy', 'scipy.stats', 'matplotlib', 'pandas', 'numpy',
    'webview', 'webview.platforms.edgechromium', 'webview.platforms.winforms',
    'clr', 'clr_loader', 'pythonnet', 'proxy_tools', 'bottle',
    'src.strategies.donchian', 'src.strategies.ema_crossover', 'src.strategies.ndx_trader',
    'src.strategies.pine_base',
    'src.bar_returns', 'src.optimization', 'src.permutation', 'src.data_loader',
    'src.alpaca_data', 'src.pine_runtime', 'src.pine_ta',
    'src.plotting', 'src.validation', 'src.validation.full_validation',
    'src.validation.in_sample_permutation', 'src.validation.walk_forward',
    'src.pine_translator', 'src.pine_translator.pipeline',
]
hiddenimports += collect_submodules('src')
hiddenimports += collect_submodules('flask')

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('src', 'src'),
        ('web_app.py', '.'),
        ('gui/assets/wale.ico', 'gui/assets'),
    ],
    hiddenimports=hiddenimports + ['web_app'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    name='WaleBacktest',
    icon='gui/assets/wale.ico',
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
