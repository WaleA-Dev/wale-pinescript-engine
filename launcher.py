"""
Wale Backtest Engine — desktop application entry point.

Runs the Flask backend in a background thread and renders the dashboard in a
native window (Edge WebView2 via pywebview) so the app behaves like a normal
Windows program: its own window, taskbar icon, no browser. Falls back to the
default browser only if no WebView2 runtime is available.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

APP_NAME = "Wale Backtest Engine"
APP_ID = "WaleADev.WaleBacktest"  # taskbar grouping / icon identity

# ── Path setup (must happen before importing web_app) ────────────────────────
if getattr(sys, "frozen", False):
    _BUNDLE = Path(sys._MEIPASS)
    _ROOT = Path(sys.executable).resolve().parent
else:
    _BUNDLE = Path(__file__).resolve().parent
    _ROOT = _BUNDLE
sys.path.insert(0, str(_BUNDLE))


def _pick_port(preferred: int = 5000) -> int:
    """Use the preferred port if free, otherwise let the OS pick one."""
    for candidate in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", candidate))
                return s.getsockname()[1]
        except OSError:
            continue
    return preferred


PORT = int(os.environ.get("PORT", 0)) or _pick_port(5000)
URL = f"http://127.0.0.1:{PORT}"


def start_server():
    """Import and run Flask in this thread (daemon)."""
    from web_app import app, _preload
    _preload()
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


def wait_for_server(timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urlopen(URL, timeout=2)
            return True
        except (URLError, OSError):
            time.sleep(0.3)
    return False


def _error_box(message: str):
    try:
        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)  # MB_ICONERROR
    except Exception:
        print(message, file=sys.stderr)


def main() -> int:
    # Proper taskbar identity (icon + grouping) on Windows
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass

    server = threading.Thread(target=start_server, daemon=True)
    server.start()

    if not wait_for_server():
        _error_box("The backtest server failed to start.\n\n"
                   "If another copy of Wale Backtest is running, close it and try again.")
        return 1

    # ── Native window (Edge WebView2) ────────────────────────────────────────
    try:
        import webview

        webview.create_window(
            APP_NAME,
            URL,
            width=1480,
            height=920,
            min_size=(1100, 700),
            background_color="#09090b",
            text_select=True,
            zoomable=True,
        )
        # gui='edgechromium' -> WebView2 (ships with Windows 10/11)
        webview.start(gui="edgechromium")
        return 0
    except Exception:
        # No WebView2 runtime — fall back to the default browser. The dialog
        # keeps the server alive; dismissing it quits the app.
        import webbrowser
        webbrowser.open(URL)
        try:
            ctypes.windll.user32.MessageBoxW(
                0,
                "Microsoft Edge WebView2 is not available, so the dashboard "
                f"opened in your browser instead.\n\n{URL}\n\n"
                "Click OK when you are done to stop the server.",
                APP_NAME, 0x40,  # MB_ICONINFORMATION
            )
        except Exception:
            print(f"Dashboard: {URL} — press Ctrl+C to quit")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
        return 0
    finally:
        os._exit(0)  # take the Flask daemon thread down with the window


if __name__ == "__main__":
    sys.exit(main())
