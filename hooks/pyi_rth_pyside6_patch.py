"""
PyInstaller runtime hook to fix PySide6/shiboken/dateutil/matplotlib conflict.

The issue: PySide6's shibokensupport hooks into the import system and tries
to inspect source code. In frozen PyInstaller apps, source isn't available,
causing '_SixMetaPathImporter' has no attribute '_path' errors.

Fix: Import dateutil.tz and matplotlib before PySide6 loads, so shibokensupport
doesn't intercept them. Also patch the shiboken feature import if needed.
"""
import sys

# Pre-import dateutil before PySide6's shiboken hooks can intercept it
try:
    import dateutil.tz
except Exception:
    pass

# Pre-import matplotlib with Agg backend before PySide6 can interfere
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
    import matplotlib.backends.backend_agg
except Exception:
    pass

# Patch: if shibokensupport is already loaded, disable its feature_imported hook
# so it won't interfere with other imports
try:
    import shibokensupport.signature.loader as _loader
    _orig = _loader.feature_imported

    def _safe_feature_imported(feature, *args, **kwargs):
        try:
            return _orig(feature, *args, **kwargs)
        except (AttributeError, OSError, TypeError):
            pass

    _loader.feature_imported = _safe_feature_imported
except Exception:
    pass
