#!/usr/bin/env python3
"""
Wale Backtest Engine - EXE Launcher

Tiny Tk window that starts the Flask web backtester in the background and
gives you an "Open Dashboard" button. This is the entry point for the
standalone WaleBacktest executable (see WaleBacktest.spec).

Usage:
    python launcher.py          # same launcher, run from source
    pyinstaller WaleBacktest.spec && ./dist/WaleBacktest.exe
"""

import os
import sys
import threading
import webbrowser
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PORT = int(os.environ.get("PORT", 5000))
URL = f"http://127.0.0.1:{PORT}"


def start_server() -> None:
    """Run the Flask app in this (daemon) thread."""
    from web_app import app, _preload

    try:
        _preload()
    except Exception:
        pass

    # Never use the reloader here: it forks the process and would leave
    # orphan servers running after the launcher quits.
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


def main() -> None:
    server = threading.Thread(target=start_server, daemon=True)
    server.start()

    root = tk.Tk()
    root.title("Wale Backtest Engine")
    root.geometry("360x180")
    root.resizable(False, False)

    tk.Label(root, text="Wale Backtest Engine", font=("Segoe UI", 14, "bold")).pack(pady=(18, 4))
    tk.Label(root, text=f"Server running at {URL}", font=("Segoe UI", 10)).pack(pady=(0, 12))

    tk.Button(
        root, text="Open Dashboard", width=24, height=2,
        command=lambda: webbrowser.open(URL),
    ).pack(pady=(0, 8))

    def stop_and_quit() -> None:
        root.destroy()
        # The Flask thread is a daemon; exiting the process stops it.
        os._exit(0)

    tk.Button(root, text="Stop && Quit", width=24, command=stop_and_quit).pack()
    root.protocol("WM_DELETE_WINDOW", stop_and_quit)

    root.mainloop()


if __name__ == "__main__":
    main()
