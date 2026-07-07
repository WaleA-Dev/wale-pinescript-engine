"""
Wale Backtest Engine — Web Application & EXE Entry Point
Flask backend for Pine translation, backtesting, and 4-step validation.
Data sources: Yahoo Finance, Dukascopy (duka_dl), CSV upload.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import io
import json
import os
import re
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

# Ensure project root on path for frozen exe
if getattr(sys, "frozen", False):
    # Single-file EXE: bundled data is in sys._MEIPASS, user data next to the EXE
    _BUNDLE = Path(sys._MEIPASS)
    _ROOT = Path(sys.executable).resolve().parent
else:
    _BUNDLE = Path(__file__).resolve().parent
    _ROOT = _BUNDLE
sys.path.insert(0, str(_BUNDLE))

from src.strategies import STRATEGY_REGISTRY, get_strategy_class
from src.data_loader import load_csv, download_yahoo, lookup_yahoo, CACHE_DIR
from src import alpaca_data

# In a frozen EXE, user-translated strategies live next to the EXE, not in the
# bundle. Extend the package search path so they can be imported.
if getattr(sys, "frozen", False):
    import src.strategies as _strategies_pkg
    _user_strat_dir = _ROOT / "src" / "strategies"
    _user_strat_dir.mkdir(parents=True, exist_ok=True)
    if str(_user_strat_dir) not in _strategies_pkg.__path__:
        _strategies_pkg.__path__.append(str(_user_strat_dir))
from src.bar_returns import compute_bar_returns, compute_metrics
from src.optimization import grid_search
from src.validation.full_validation import run_full_validation
from src.pine_translator.pipeline import TranslationPipeline

UPLOAD_DIR = _ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(_BUNDLE / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# ── State ────────────────────────────────────────────────────────────────────
_state = {"df": None, "data_info": None, "last_trades": None}
_validation_jobs: dict = {}


def _set_data(df: pd.DataFrame, source: str):
    _state["df"] = df
    _state["data_info"] = {
        "source": source,
        "bars": len(df),
        "start": str(df.index[0]) if len(df) > 0 else "",
        "end": str(df.index[-1]) if len(df) > 0 else "",
        "columns": list(df.columns),
    }


# ── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("converge.html")


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(str(_ROOT / "reports" / "plots"), filename)


@app.route("/favicon.ico")
def favicon():
    icon_dir = _BUNDLE / "gui" / "assets"
    if (icon_dir / "wale.ico").exists():
        return send_from_directory(str(icon_dir), "wale.ico")
    return ("", 404)


# ── API: Strategies ──────────────────────────────────────────────────────────

# Classes shipped inside the app — everything else is a user strategy.
_BUILTIN_CLASSES = {"DonchianBreakoutStrategy", "EMACrossoverStrategy", "NdxTraderStrategy"}


def _strategy_display_name(cls) -> str:
    """Human name: the Pine strategy title for transpiled ones, else the class."""
    doc = (cls.__doc__ or "").strip().splitlines()
    if doc:
        first = doc[0].replace("(auto-transpiled)", "").strip()
        first = re.sub(r"^translated from:\s*", "", first, flags=re.I).rstrip(":").strip()
        if first and not first.lower().startswith(("custom python", "base class")):
            return first if len(first) <= 60 else first[:57] + "..."
    name = cls.__name__
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name.removesuffix("Strategy")).strip()


def _strategy_files(module_name: str):
    """(py_path, pine_path) for a user strategy module on disk, or Nones."""
    import src.strategies as _sp
    for pkg_dir in _sp.__path__:
        py = Path(pkg_dir) / f"{module_name}.py"
        if py.exists():
            pine = py.with_suffix(".pine")
            return py, (pine if pine.exists() else None)
    return None, None


@app.route("/api/strategies", methods=["GET"])
def api_strategies():
    # Discover saved strategy modules on disk (incl. previously translated ones)
    import src.strategies as _sp
    load_errors = []
    for pkg_dir in _sp.__path__:
        for f in Path(pkg_dir).glob("*.py"):
            stem = f.stem
            if stem in ("__init__", "base", "pine_base"):
                continue
            if stem not in STRATEGY_REGISTRY:
                try:
                    get_strategy_class(stem)
                except Exception as e:
                    load_errors.append({"module": stem, "error": str(e)})

    from src.strategies.pine_base import PineStrategy
    seen = {}
    for name, cls in sorted(STRATEGY_REGISTRY.items()):
        if cls in seen:
            # keep the longest registry alias as the canonical module name
            if len(name) > len(seen[cls]["name"]):
                seen[cls]["name"] = name
            continue
        try:
            grid = cls().param_grid()
        except Exception:
            grid = {}
        is_builtin = cls.__name__ in _BUILTIN_CLASSES
        is_pine = isinstance(cls, type) and issubclass(cls, PineStrategy)
        seen[cls] = {
            "name": name,
            "class": cls.__name__,
            "display": _strategy_display_name(cls),
            "params": grid,
            "kind": "pine" if is_pine else "python",
            "builtin": is_builtin,
            "deletable": not is_builtin,
        }
    strategies = list(seen.values())
    for s in strategies:
        py, pine = _strategy_files(s["name"])
        s["has_source"] = py is not None or s["builtin"]
        s["has_pine"] = pine is not None
    return jsonify({"strategies": strategies, "load_errors": load_errors})


@app.route("/api/strategies/<name>/source", methods=["GET"])
def api_strategy_source(name):
    """Python (and original Pine, when saved) source for the library viewer."""
    import inspect
    key = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    py_path, pine_path = _strategy_files(key)
    python_src = None
    if py_path:
        python_src = py_path.read_text(encoding="utf-8")
    else:
        try:
            python_src = inspect.getsource(get_strategy_class(key))
        except Exception:
            pass
    if python_src is None:
        return jsonify({"error": f"No source found for '{name}'"}), 404
    return jsonify({
        "python": python_src,
        "pine": pine_path.read_text(encoding="utf-8") if pine_path else None,
    })


@app.route("/api/strategies/<name>", methods=["DELETE"])
def api_strategy_delete(name):
    """Delete a user strategy (its .py/.pine files) and unregister it."""
    key = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    cls = STRATEGY_REGISTRY.get(key)
    if cls is not None and cls.__name__ in _BUILTIN_CLASSES:
        return jsonify({"error": "Built-in strategies can't be deleted."}), 400
    py_path, pine_path = _strategy_files(key)
    if py_path is None:
        return jsonify({"error": f"No strategy file found for '{name}'"}), 404
    py_path.unlink()
    if pine_path:
        pine_path.unlink()
    # drop every registry alias resolving to this class + the cached module
    if cls is not None:
        for alias in [a for a, c in STRATEGY_REGISTRY.items() if c is cls]:
            STRATEGY_REGISTRY.pop(alias, None)
    sys.modules.pop(f"src.strategies.{key}", None)
    return jsonify({"success": True, "deleted": key})


@app.route("/api/translate", methods=["POST"])
def api_translate():
    pine_code = request.json.get("pine_code", "") if request.is_json else request.form.get("pine_code", "")
    if not pine_code.strip():
        return jsonify({"success": False, "issues": ["No Pine code provided"]}), 400
    # Use writable strategy dir (next to EXE or project root, not inside frozen bundle)
    strategy_dir = _ROOT / "src" / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = TranslationPipeline(strategy_dir=str(strategy_dir)).translate(pine_code, auto_save=True)
    except Exception as e:
        return jsonify({"success": False, "issues": [str(e)]}), 500
    # Auto-register the translated strategy so it appears in the dropdown
    if result.get("success") and result.get("strategy_name"):
        try:
            module_name = TranslationPipeline._safe_module_name(result["strategy_name"])
            result["module_name"] = module_name
            import importlib
            full = f"src.strategies.{module_name}"
            if full in sys.modules:
                importlib.reload(sys.modules[full])
            get_strategy_class(module_name)
            result["registered"] = True
        except Exception as e:
            # Registration failure is a hard failure: if the strategy can't be
            # loaded, a backtest would silently run something else entirely.
            result["success"] = False
            result["registered"] = False
            msg = str(e).split(". Available:")[0]
            result["issues"] = result.get("issues", []) + [
                f"Translated but couldn't be loaded as a runnable strategy: {msg}"]
    return jsonify(result)


@app.route("/api/save-strategy", methods=["POST"])
def api_save_strategy():
    """Save user-written Python strategy code to disk and auto-register it."""
    data = request.get_json(force=True)
    code = data.get("code", "")
    name = data.get("name", "custom").strip()
    if not code.strip():
        return jsonify({"success": False, "error": "No code provided"}), 400

    # Sanitize name to valid Python module name
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()
    if not name:
        name = "custom_strategy"

    # 1. Validate syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return jsonify({"success": False, "error": f"Syntax error: {e}"}), 400

    # 2. Check it contains a BaseStrategy subclass
    has_subclass = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if "BaseStrategy" in base_name:
                    has_subclass = True
                    break
    if not has_subclass:
        return jsonify({"success": False, "error": "Code must contain a class that inherits from BaseStrategy"}), 400

    # 3. Write to strategies directory
    strategy_dir = _ROOT / "src" / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    filepath = strategy_dir / f"{name}.py"
    filepath.write_text(code, encoding="utf-8")

    # 4. Import to register
    try:
        module_path = f"src.strategies.{name}"
        if module_path in sys.modules:
            importlib.reload(sys.modules[module_path])
        else:
            importlib.import_module(module_path)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        return jsonify({"success": False, "error": f"Import error: {e}"}), 400

    # 5. Test-run on synthetic data to verify generate_signals() works
    try:
        strategy_cls = get_strategy_class(name)
        test_df = pd.DataFrame({
            'open': np.tile([100.0, 101.0, 102.0, 103.0, 104.0], 20),
            'high': np.tile([102.0, 103.0, 104.0, 105.0, 106.0], 20),
            'low':  np.tile([99.0, 100.0, 101.0, 102.0, 103.0], 20),
            'close': np.tile([101.0, 102.0, 103.0, 104.0, 105.0], 20),
            'volume': np.full(100, 1000.0),
        }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
        signals = strategy_cls().generate_signals(test_df)
        if signals is None or (hasattr(signals, '__len__') and len(signals) == 0):
            filepath.unlink(missing_ok=True)
            return jsonify({"success": False, "error": "generate_signals() returned None or empty"}), 400
    except Exception as e:
        filepath.unlink(missing_ok=True)
        # Remove from registry so broken strategy doesn't linger
        module_path = f"src.strategies.{name}"
        sys.modules.pop(module_path, None)
        return jsonify({"success": False, "error": f"Runtime error in generate_signals(): {e}"}), 400

    return jsonify({"success": True, "name": name, "path": str(filepath)})


# ── API: Alpaca keys & trial ─────────────────────────────────────────────────

@app.route("/api/alpaca/status", methods=["GET"])
def api_alpaca_status():
    return jsonify({
        "has_keys": alpaca_data.has_keys(),
        "trial_remaining": alpaca_data.trial_remaining(),
        "trial_limit": alpaca_data.TRIAL_LIMIT,
        # path only — key contents are never sent to the UI
        "config_path": str(alpaca_data.CONFIG_FILE),
    })


@app.route("/api/alpaca/reveal", methods=["POST"])
def api_alpaca_reveal():
    """Open the key file's folder in Explorer with the file selected."""
    import subprocess
    path = alpaca_data.CONFIG_FILE
    if not path.exists():
        return jsonify({"error": "No key file yet — save your Alpaca keys first."}), 404
    subprocess.Popen(["explorer", "/select,", str(path)])
    return jsonify({"success": True})


@app.route("/api/alpaca/keys", methods=["POST"])
def api_alpaca_keys():
    data = request.get_json(force=True)
    key_id = (data.get("key_id") or "").strip()
    secret = (data.get("secret") or "").strip()
    if not key_id or not secret:
        return jsonify({"success": False, "error": "Both Key ID and Secret are required"}), 400
    check = alpaca_data.validate_keys(key_id, secret)
    if not check.get("valid"):
        return jsonify({"success": False, "error": check.get("error", "Invalid credentials")}), 400
    alpaca_data.save_keys(key_id, secret)
    return jsonify({"success": True, "mode": check.get("mode")})


@app.route("/api/alpaca/keys", methods=["DELETE"])
def api_alpaca_keys_clear():
    alpaca_data.clear_keys()
    return jsonify({"success": True})


def _trial_gate():
    """Return an error response if the no-key free trial is exhausted, else None."""
    if alpaca_data.has_keys():
        return None
    if alpaca_data.trial_remaining() <= 0:
        return jsonify({
            "error": "Free trial used up (5 downloads). Add your free Alpaca API key "
                     "(alpaca.markets) in the Data Source panel for unlimited data.",
            "trial_exhausted": True,
        }), 402
    return None


# ── API: Data ────────────────────────────────────────────────────────────────

@app.route("/api/data/alpaca", methods=["POST"])
def api_data_alpaca():
    """Fetch live OHLCV bars from Alpaca (free IEX feed, user's own key)."""
    data = request.get_json(force=True)
    symbol = (data.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    timeframe = data.get("timeframe", "1d")
    start = (data.get("start") or "").strip() or None
    end = (data.get("end") or "").strip() or None
    adjustment = data.get("adjustment", "split")
    if adjustment not in ("split", "all"):
        adjustment = "split"
    try:
        df = alpaca_data.fetch_bars(symbol, timeframe=timeframe, start=start, end=end,
                                    adjustment=adjustment)
        cache_file = CACHE_DIR / f"{symbol.upper()}_{timeframe}_{adjustment}_alpaca.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        save_df = df.copy()
        save_df.index.name = "timestamp"
        save_df.to_csv(str(cache_file))
        adj_label = "splits+div adj" if adjustment == "all" else "split adj"
        _set_data(df, f"{symbol.upper()} ({timeframe}, {adj_label}) via Alpaca IEX")
        return jsonify({"success": True, **_state["data_info"]})
    except alpaca_data.AlpacaAuthError as e:
        return jsonify({"error": str(e), "needs_keys": True}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/data/upload", methods=["POST"])
def api_data_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    save_path = UPLOAD_DIR / f.filename
    f.save(str(save_path))
    try:
        df = load_csv(save_path)
        _set_data(df, f.filename)
        return jsonify({"success": True, **_state["data_info"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/data/lookup", methods=["POST"])
def api_data_lookup():
    """Validate ticker and return max available date range."""
    data = request.get_json(force=True)
    symbol = data.get("symbol", "").strip()
    if not symbol:
        return jsonify({"valid": False, "error": "No symbol provided"}), 400
    try:
        result = lookup_yahoo(symbol)
        return jsonify(result)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400


@app.route("/api/data/download", methods=["POST"])
def api_data_download():
    data = request.get_json(force=True)
    symbol = data.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    interval = data.get("interval", "1d")
    start = data.get("start", "").strip() or None  # None = fetch max
    gate = _trial_gate()
    if gate is not None:
        return gate
    try:
        df = download_yahoo(symbol, interval=interval, start=start)
        _set_data(df, f"{symbol.upper()} ({interval}) via Yahoo Finance")
        info = dict(_state["data_info"])
        if not alpaca_data.has_keys():
            info["trial_remaining"] = alpaca_data.consume_trial()
        return jsonify({"success": True, **info})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/data/dukascopy", methods=["POST"])
def api_data_dukascopy():
    """Download tick/minute data from Dukascopy via duka_dl, resample to OHLCV."""
    data = request.get_json(force=True)
    symbol = data.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400
    start = data.get("start", "").strip()
    if not start:
        return jsonify({"error": "Start date required for Dukascopy"}), 400
    end = data.get("end", "")
    resample = data.get("resample", "1h")  # 1min, 5min, 15min, 1h, 4h, 1d

    gate = _trial_gate()
    if gate is not None:
        return gate

    if not end:
        end = (datetime.now() - timedelta(days=1)).strftime("%d-%m-%Y")

    try:
        import duka_dl
        import httpx
        import polars as pl

        dates = duka_dl.process_date(start, end)
        if not dates:
            return jsonify({"error": "Invalid date range"}), 400

        async def _fetch():
            sem = asyncio.Semaphore(10)
            frames = []
            async with httpx.AsyncClient() as client:
                tasks = [duka_dl.fetch_one_day(client, symbol, d, sem, "BID") for d in dates]
                results = await asyncio.gather(*tasks)
                for r in results:
                    if r is not None:
                        frames.append(r)
            if not frames:
                return None
            return pl.concat(frames)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_pl = loop.run_until_complete(_fetch())
        finally:
            loop.close()

        if result_pl is None or len(result_pl) == 0:
            return jsonify({"error": f"No data returned for {symbol}"}), 400

        # Convert polars to pandas
        pdf = result_pl.to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"])
        pdf = pdf.set_index("timestamp").sort_index()

        # Resample to requested timeframe
        resample_map = {
            "1min": "1min", "5min": "5min", "15min": "15min",
            "30min": "30min", "1h": "1h", "4h": "4h", "1d": "1D",
        }
        rs = resample_map.get(resample, "1h")
        ohlcv = pdf.resample(rs).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["open"])

        # Cache it
        cache_file = CACHE_DIR / f"{symbol.replace('-','_')}_{resample}_duka.csv"
        ohlcv.to_csv(str(cache_file))

        _set_data(ohlcv, f"{symbol} ({resample}) via Dukascopy")
        info = dict(_state["data_info"])
        if not alpaca_data.has_keys():
            info["trial_remaining"] = alpaca_data.consume_trial()
        return jsonify({"success": True, **info})

    except ImportError:
        return jsonify({"error": "duka_dl not installed. Run: pip install duka-dl"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/data/status", methods=["GET"])
def api_data_status():
    if _state["data_info"] is None:
        return jsonify({"loaded": False})
    return jsonify({"loaded": True, **_state["data_info"]})


# ── API: Backtest ────────────────────────────────────────────────────────────

def _format_dt(dt) -> str:
    """Format datetime: include time for intraday data, date-only for daily."""
    s = str(dt)
    if hasattr(dt, 'hour') and (dt.hour or dt.minute):
        return s[:16]  # "2024-01-15 09:30"
    return s[:10]      # "2024-01-15"


def _build_trade_list(df_result: pd.DataFrame) -> list:
    """Build trade list using next-bar-open execution prices (matches TradingView).

    Signal generated on bar N -> entry at open of bar N+1.
    Signal changes on bar M -> exit at open of bar M+1.
    """
    signal = df_result["signal"].values
    close = df_result["close"].values
    has_open = "open" in df_result.columns
    opn = df_result["open"].values if has_open else close
    dates = df_result.index
    trades = []
    entry_idx = entry_price = direction = None

    for i in range(1, len(signal)):
        if signal[i - 1] != signal[i]:
            # Position changed: signal[i-1] decided on bar i-1, fill at bar i open
            if entry_idx is not None:
                exit_price = opn[i] if has_open else close[i]
                pnl_pct = ((exit_price / entry_price) - 1) * (1 if direction == "LONG" else -1)
                trades.append({
                    "id": len(trades) + 1, "direction": direction,
                    "entry_date": _format_dt(dates[entry_idx]),
                    "entry_bar": int(entry_idx), "entry_price": round(float(entry_price), 2),
                    "exit_date": _format_dt(dates[i]),
                    "exit_bar": int(i), "exit_price": round(float(exit_price), 2),
                    "pnl_pct": round(float(pnl_pct * 100), 2),
                    "bars_held": int(i - entry_idx),
                })
            if signal[i] != 0:
                # New position starts: signal changed on bar i-1, fill at bar i open
                entry_idx = i
                entry_price = opn[i] if has_open else close[i]
                direction = "LONG" if signal[i] > 0 else "SHORT"
            else:
                entry_idx = entry_price = direction = None
    return trades


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    if _state["df"] is None:
        return jsonify({"error": "No data loaded. Fetch or upload data first."}), 400

    data = request.get_json(force=True)
    strategy_name = data.get("strategy")
    if not strategy_name:
        # Never silently fall back to a default strategy — the user must know
        # exactly which strategy is being backtested.
        return jsonify({"error": "No strategy selected."}), 400
    commission = float(data.get("commission", 0.001))

    try:
        strategy_cls = get_strategy_class(strategy_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    df = _state["df"].copy()
    params = data.get("params") or {}
    try:
        strategy = strategy_cls(**params)
    except TypeError as e:
        return jsonify({"error": f"Bad strategy parameters: {e}"}), 400

    bar_errors = 0
    first_error = None
    from src.strategies.pine_base import PineStrategy
    if isinstance(strategy, PineStrategy):
        # Event-driven TradingView-semantics engine: real stop/limit fills
        run = strategy.run(df)
        metrics = run["metrics"]
        eq = np.asarray(run["equity_curve"], dtype=float)
        dd = np.asarray(run["drawdown_curve"], dtype=float)
        trades = run["trades"]
        bar_errors = run.get("bar_errors", 0)
        first_error = run.get("first_error")
    else:
        # Legacy vectorized path
        signals = strategy.generate_signals(df)
        result_df = compute_bar_returns(df, signals=signals, commission=commission)
        metrics = compute_metrics(result_df["strategy_return"])
        eq = result_df["equity_curve"].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        trades = _build_trade_list(result_df)

    step = max(1, len(eq) // 5000) if len(eq) > 5000 else 1
    dates = [str(d)[:10] for d in df.index[::step]]

    _state["last_trades"] = trades

    return jsonify({
        "success": True, "strategy": strategy_name,
        "strategy_class": strategy_cls.__name__,
        "bar_errors": int(bar_errors), "first_error": first_error,
        "bars": len(df),
        "metrics": {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                        else round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in metrics.items()},
        "equity_curve": [round(float(x), 4) for x in eq[::step].tolist()],
        "drawdown_curve": [round(float(x), 2) for x in dd[::step].tolist()],
        "dates": dates,
        "trades": trades,
        "total_trades": len(trades),
    })


# ── API: Export & TradingView comparison ────────────────────────────────────

@app.route("/api/export/save", methods=["POST"])
def api_export_save():
    """Save an export to the user's Downloads folder (blob downloads don't
    work inside the native WebView2 window)."""
    data = request.get_json(force=True)
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", data.get("filename") or "export.csv")
    content = data.get("content") or ""
    if not content:
        return jsonify({"error": "Nothing to export yet."}), 400
    out_dir = Path.home() / "Downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(content, encoding="utf-8-sig")
    return jsonify({"success": True, "path": str(path)})


def _parse_tv_trades(text: str):
    """Parse TradingView's 'List of trades' CSV export into
    [{no, entry_date, entry_price, exit_date, exit_price}] sorted by entry."""
    import csv as _csv
    rows = list(_csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    trades: dict = {}
    for row in rows:
        no = (row.get("Trade #") or "").strip()
        typ = (row.get("Type") or "").lower()
        dt = (row.get("Date and time") or row.get("Date/Time") or "").strip()
        price_key = next((k for k in row if k and k.startswith("Price")), None)
        try:
            price = float((row.get(price_key) or "").replace(",", ""))
        except (TypeError, ValueError):
            continue
        if not no or not dt:
            continue
        t = trades.setdefault(no, {"no": no})
        if typ.startswith("entry"):
            t["entry_date"], t["entry_price"] = dt, price
        elif typ.startswith("exit"):
            t["exit_date"], t["exit_price"] = dt, price
    out = [t for t in trades.values() if "entry_date" in t]
    out.sort(key=lambda t: t["entry_date"])
    return out


def _compare_trades(tv, mine):
    """Align TV trades with engine trades by entry-date proximity and report
    per-trade date/price agreement."""
    from datetime import datetime

    def d10(x):
        return str(x)[:10]

    def _dt(x):
        return datetime.strptime(d10(x), "%Y-%m-%d")

    mine = sorted(mine, key=lambda t: str(t.get("entry_date") or ""))
    used = set()
    rows = []
    for t in tv:
        best, best_gap = None, 10**9
        for j, m in enumerate(mine):
            if j in used or not m.get("entry_date"):
                continue
            gap = abs((_dt(m["entry_date"]) - _dt(t["entry_date"])).days)
            if gap < best_gap:
                best, best_gap = j, gap
        if best is not None and best_gap <= 45:
            used.add(best)
            m = mine[best]
            row = {
                "tv_entry": d10(t["entry_date"]), "tv_exit": d10(t.get("exit_date", "")),
                "tv_entry_price": t.get("entry_price"), "tv_exit_price": t.get("exit_price"),
                "my_entry": d10(m["entry_date"]), "my_exit": d10(m.get("exit_date") or ""),
                "my_entry_price": m.get("entry_price"), "my_exit_price": m.get("exit_price"),
                "entry_match": d10(t["entry_date"]) == d10(m["entry_date"]),
                "exit_match": (d10(t.get("exit_date", "")) == d10(m.get("exit_date") or "")
                               and bool(t.get("exit_date"))),
            }
            ep, mp = t.get("entry_price"), m.get("entry_price")
            row["entry_px_diff_pct"] = (round((mp - ep) / ep * 100, 3)
                                        if ep and mp else None)
            rows.append(row)
        else:
            rows.append({"tv_entry": d10(t["entry_date"]),
                         "tv_exit": d10(t.get("exit_date", "")),
                         "tv_entry_price": t.get("entry_price"),
                         "tv_exit_price": t.get("exit_price"),
                         "missing": "engine"})
    for j, m in enumerate(mine):
        if j not in used:
            rows.append({"my_entry": d10(m["entry_date"]),
                         "my_exit": d10(m.get("exit_date") or ""),
                         "my_entry_price": m.get("entry_price"),
                         "my_exit_price": m.get("exit_price"),
                         "missing": "tv"})
    rows.sort(key=lambda r: r.get("tv_entry") or r.get("my_entry") or "")
    legs = sum(1 for r in rows if "missing" not in r) * 2
    exact = sum(1 for r in rows if r.get("entry_match")) +         sum(1 for r in rows if r.get("exit_match"))
    return {"rows": rows, "legs": legs, "legs_exact": exact,
            "tv_trades": len(tv), "engine_trades": len(mine)}


@app.route("/api/compare-tv", methods=["POST"])
def api_compare_tv():
    """Compare the last backtest against a TradingView 'List of trades' CSV."""
    if not _state.get("last_trades"):
        return jsonify({"error": "Run a backtest first, then compare."}), 400
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "No file uploaded."}), 400
    try:
        tv = _parse_tv_trades(f.read().decode("utf-8-sig", errors="replace"))
    except Exception as e:
        return jsonify({"error": f"Couldn't parse that CSV as a TradingView trade list: {e}"}), 400
    if not tv:
        return jsonify({"error": "No trades found — export TradingView's 'List of trades' "
                                 "(download icon above the trade table) and upload that file."}), 400
    return jsonify({"success": True, **_compare_trades(tv, _state["last_trades"])})


# ── API: Validation ──────────────────────────────────────────────────────────

def _run_validation_thread(job_id, strategy_name, data_path, n_perms_is, n_perms_wf, train_years):
    job = _validation_jobs[job_id]
    try:
        job.update(status="running", step=1, step_name="In-Sample Optimization", progress=10)
        result = run_full_validation(
            strategy_name=strategy_name, data_path=data_path,
            n_perms_is=n_perms_is, n_perms_wf=n_perms_wf,
            train_years=train_years, output_root=str(_ROOT / "reports"), quick=False,
        )
        job.update(status="complete", step=4, step_name="Complete", progress=100, result=result)
    except Exception as e:
        job.update(status="error", error=str(e), progress=100)


@app.route("/api/validate", methods=["POST"])
def api_validate():
    if _state["df"] is None:
        return jsonify({"error": "No data loaded"}), 400

    data = request.get_json(force=True)
    strategy_name = data.get("strategy", "donchian")
    n_perms_is = int(data.get("n_perms_is", 200))
    n_perms_wf = int(data.get("n_perms_wf", 100))
    train_years = float(data.get("train_years", 4.0))
    quick = data.get("quick", False)

    try:
        get_strategy_class(strategy_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    temp_csv = _ROOT / "data" / "cache" / f"_web_validation_{strategy_name}.csv"
    temp_csv.parent.mkdir(parents=True, exist_ok=True)
    save_df = _state["df"].copy()
    save_df.index.name = "timestamp"
    save_df.to_csv(str(temp_csv))

    if quick:
        try:
            result = run_full_validation(
                strategy_name=strategy_name, data_path=str(temp_csv),
                n_perms_is=n_perms_is, n_perms_wf=n_perms_wf,
                train_years=train_years, output_root=str(_ROOT / "reports"), quick=True,
            )
            return jsonify({"success": True, "job_id": None, "result": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    job_id = str(uuid.uuid4())[:8]
    _validation_jobs[job_id] = {
        "status": "starting", "step": 0, "step_name": "Initializing",
        "progress": 0, "result": None, "error": None, "strategy": strategy_name,
    }
    t = threading.Thread(
        target=_run_validation_thread,
        args=(job_id, strategy_name, str(temp_csv), n_perms_is, n_perms_wf, train_years),
        daemon=True,
    )
    t.start()
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/validate/<job_id>", methods=["GET"])
def api_validate_status(job_id):
    if job_id not in _validation_jobs:
        return jsonify({"error": "Job not found"}), 404
    job = _validation_jobs[job_id]
    resp = {"status": job["status"], "step": job["step"],
            "step_name": job["step_name"], "progress": job["progress"],
            "strategy": job.get("strategy")}
    if job["status"] == "complete":
        resp["result"] = job["result"]
    if job["status"] == "error":
        resp["error"] = job["error"]
    return jsonify(resp)


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    if _state["df"] is None:
        return jsonify({"error": "No data loaded"}), 400
    data = request.get_json(force=True)
    strategy_name = data.get("strategy")
    if not strategy_name:
        return jsonify({"error": "No strategy selected."}), 400
    try:
        strategy_cls = get_strategy_class(strategy_name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    grid = strategy_cls().param_grid()
    if not grid:
        return jsonify({"error": "Strategy has no param grid"}), 400
    try:
        best_params, best_score = grid_search(_state["df"], strategy_cls, grid)
        return jsonify({"success": True, "best_params": best_params,
                        "best_score": round(float(best_score), 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ─────────────────────────────────────────────────────────────────────

def _preload():
    qqq = CACHE_DIR / "QQQ_1d.csv"
    if qqq.exists():
        try:
            df = load_csv(qqq)
            _set_data(df, "QQQ (1d) from cache")
            print(f"  Pre-loaded QQQ daily: {len(df)} bars")
        except Exception:
            pass


if __name__ == "__main__":
    _preload()
    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Wale Backtest Engine")
    print(f"  {url}\n")

    # Auto-open browser (especially useful when running as .exe)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # use_reloader=False: translating a strategy writes a .py file at runtime,
    # which would otherwise restart the server and drop loaded data.
    app.run(host="127.0.0.1", port=port,
            debug=not getattr(sys, "frozen", False), use_reloader=False)
