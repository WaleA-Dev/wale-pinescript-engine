#!/usr/bin/env python3
"""
Autonomous overnight strategy migration + validation runner.

Implements the workflow in AUTONOMOUS_STRATEGY_MIGRATION_OVERNIGHT directive:
- discover strategies
- inventory data
- migrate/translate where possible
- run full validation and produce reports
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pine_translator.pipeline import TranslationPipeline
from src.strategies import STRATEGY_REGISTRY, update_validation_status
from src.validation.full_validation import run_full_validation


SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    "reports",
}


def _is_skipped_path(path: Path) -> bool:
    return any(part.lower() in SKIP_DIR_NAMES for part in path.parts)


@dataclass
class StrategyFile:
    name: str
    path: Path
    priority: int
    description: str


def _classify_priority(code: str) -> int:
    low = code.lower()
    complex_markers = ["request.security", "array.", "for ", "while ", "matrix", "varip", "mtf"]
    medium_markers = ["ta.macd", "ta.rsi", "ta.atr", "strategy.exit", "switch"]
    if any(m in low for m in complex_markers):
        return 3
    if any(m in low for m in medium_markers):
        return 2
    return 1


def _describe_logic(code: str) -> str:
    lines = []
    low = code.lower()
    if "crossover" in low:
        lines.append("Uses crossover entries")
    if "crossunder" in low:
        lines.append("Uses crossunder entries/exits")
    if "strategy.exit" in low:
        lines.append("Contains explicit stop/target exits")
    if "ta.ema" in low:
        lines.append("EMA-based indicators")
    if "ta.rsi" in low:
        lines.append("RSI filter")
    if "ta.atr" in low:
        lines.append("ATR filter/stops")
    if not lines:
        lines.append("Custom logic")
    return "; ".join(lines)


def discover_strategies(base: Path) -> List[StrategyFile]:
    candidates = []
    scan_dirs = [base / "archive" / "legacy_pine", base / "examples", base]
    seen_keys = set()
    for d in scan_dirs:
        if not d.exists():
            continue
        for pf in d.rglob("*.pine"):
            if _is_skipped_path(pf):
                continue
            try:
                code = pf.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            name_match = re.search(r"strategy\s*\(\s*['\"]([^'\"]+)['\"]", code)
            strategy_name = name_match.group(1) if name_match else pf.stem
            normalized_name = re.sub(r"\s+", " ", strategy_name.strip().lower())
            code_digest = hashlib.sha1(code.encode("utf-8", errors="ignore")).hexdigest()
            key = (normalized_name, code_digest)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                StrategyFile(
                    name=strategy_name,
                    path=pf,
                    priority=_classify_priority(code),
                    description=_describe_logic(code),
                )
            )
    # Deduplicate by absolute path as final guard.
    uniq = {c.path.resolve(): c for c in candidates}
    return sorted(uniq.values(), key=lambda x: (x.priority, x.name.lower(), str(x.path)))


def write_strategy_inventory(path: Path, discovered: List[StrategyFile]) -> None:
    lines = ["# Legacy Strategy Inventory", ""]
    for pri in [1, 2, 3]:
        lines.append(f"## Priority {pri}")
        group = [s for s in discovered if s.priority == pri]
        if not group:
            lines.append("- None")
            lines.append("")
            continue
        for i, s in enumerate(group, 1):
            lines.extend(
                [
                    f"{i}. **{s.name}**",
                    f"   - File: `{s.path}`",
                    f"   - Logic: {s.description}",
                    f"   - Status: PENDING",
                ]
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _infer_frequency_label(dt_series: pd.Series) -> str:
    dt = pd.to_datetime(dt_series, errors="coerce").dropna().sort_values().drop_duplicates()
    if len(dt) < 3:
        return "unknown"
    deltas = dt.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return "unknown"
    med = float(deltas.median())
    if med <= 0:
        return "unknown"
    minute = 60.0
    hour = 3600.0
    day = 86400.0
    if abs(med - minute) < 1:
        return "1m"
    if abs(med - 5 * minute) < 1:
        return "5m"
    if abs(med - 15 * minute) < 1:
        return "15m"
    if abs(med - 30 * minute) < 1:
        return "30m"
    if abs(med - hour) < 1:
        return "1H"
    if abs(med - 4 * hour) < 1:
        return "4H"
    if abs(med - day) < 1:
        return "1D"
    return f"{int(med)}s"


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def data_inventory(base: Path) -> List[dict]:
    out = []
    for pf in list(base.rglob("*.csv")) + list(base.rglob("*.parquet")):
        if _is_skipped_path(pf):
            continue
        try:
            df = _load_table(pf)
        except Exception:
            continue

        cols_norm = {c.lower().strip(): c for c in df.columns}
        has_ohlc = all(c in cols_norm for c in ("open", "high", "low", "close"))

        time_col = next((cols_norm[c] for c in ("time", "date", "datetime", "timestamp") if c in cols_norm), None)
        date_min = date_max = None
        freq = "unknown"
        if time_col:
            dt = pd.to_datetime(df[time_col], errors="coerce")
            if dt.notna().any():
                date_min, date_max = str(dt.min()), str(dt.max())
                freq = _infer_frequency_label(dt)

        nan_critical = None
        if has_ohlc:
            ohlc_df = pd.DataFrame(
                {c: pd.to_numeric(df[cols_norm[c]], errors="coerce") for c in ("open", "high", "low", "close")}
            )
            nan_critical = int(ohlc_df.isna().sum().sum())

        quality = "VALID" if has_ohlc and (nan_critical == 0) else "ISSUES"

        out.append(
            {
                "file": str(pf),
                "rows": int(len(df)),
                "format": pf.suffix.lower().lstrip("."),
                "has_ohlc": bool(has_ohlc),
                "nan_critical": nan_critical,
                "date_min": date_min,
                "date_max": date_max,
                "frequency": freq,
                "quality": quality,
            }
        )
    return sorted(out, key=lambda x: x["file"])


def write_data_inventory(path: Path, items: List[dict]) -> None:
    lines = ["# Available Data Inventory", ""]
    valid = [x for x in items if x["has_ohlc"]]
    if not valid:
        lines.append("No valid OHLC datasets found.")
    else:
        for i, it in enumerate(valid, 1):
            quality = it.get("quality", "UNKNOWN")
            if it.get("nan_critical") is not None and it.get("nan_critical") > 0:
                quality = f"{quality} (critical NaNs={it['nan_critical']})"
            lines.extend(
                [
                    f"{i}. **{Path(it['file']).name}**",
                    f"   - File: `{it['file']}`",
                    f"   - Rows: {it['rows']}",
                    f"   - Format: {it.get('format', 'csv')}",
                    f"   - Date Range: {it['date_min']} to {it['date_max']}",
                    f"   - Frequency: {it.get('frequency', 'unknown')}",
                    f"   - Quality: {quality}",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _best_data(items: List[dict]) -> str | None:
    valid = [x for x in items if x["has_ohlc"] and x.get("quality") == "VALID"]
    if not valid:
        return None
    # Prefer largest clean dataset, then deterministic name ordering.
    valid = sorted(valid, key=lambda x: (-x["rows"], x["file"]))
    return valid[0]["file"]


def attempt_with_retry(func, max_attempts: int = 3, label: str = "task"):
    """Retry wrapper for transient failures."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                print(f"[retry] {label} failed ({attempt}/{max_attempts}): {exc}")
            else:
                print(f"[retry] {label} failed ({attempt}/{max_attempts}) - giving up: {exc}")
    raise last_exc


def write_progress(path: Path, rows: List[dict]) -> None:
    total = len(rows)
    validated = sum(1 for r in rows if r.get("status") == "VALIDATED")
    overfit = sum(1 for r in rows if r.get("status") == "OVERFIT")
    poor = sum(1 for r in rows if r.get("status") == "POOR")
    blocked = sum(1 for r in rows if r.get("status") in {"BLOCKED", "TRANSLATION_FAILED", "ERROR"})
    pending = sum(1 for r in rows if r.get("status") == "PENDING")

    lines = [
        "# Migration Progress",
        "",
        "## Summary",
        f"- Total strategies discovered: {total}",
        f"- Validated: {validated}",
        f"- Overfit: {overfit}",
        f"- Poor: {poor}",
        f"- Migration blocked: {blocked}",
        f"- Pending: {pending}",
        "",
        "## Detailed Status",
        "",
        "| Strategy | Source | Status | Report |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('strategy','?')} | `{r.get('source','?')}` | {r.get('status','PENDING')} | {r.get('report','-')} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_summary(path: Path, rows: List[dict]) -> None:
    validated = [r for r in rows if r.get("status") == "VALIDATED"]
    overfit = [r for r in rows if r.get("status") == "OVERFIT"]
    poor = [r for r in rows if r.get("status") == "POOR"]
    blocked = [r for r in rows if r.get("status") in {"BLOCKED", "TRANSLATION_FAILED", "ERROR"}]

    lines = [
        "# Final Migration Summary",
        f"Date: {datetime.now().isoformat(sep=' ', timespec='seconds')}",
        "",
        "## Executive Summary",
        f"- Total processed: {len(rows)}",
        f"- Validated: {len(validated)}",
        f"- Overfit: {len(overfit)}",
        f"- Poor: {len(poor)}",
        f"- Blocked: {len(blocked)}",
        "",
        "## Validated Strategies",
    ]
    if validated:
        for r in validated:
            lines.append(
                f"- **{r['strategy']}**: IS PF={r.get('is_pf', float('nan')):.2f}, OOS PF={r.get('oos_pf', float('nan')):.2f}, report `{r.get('report','-')}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Failed/Weak Strategies"])
    for grp_name, grp in [("Overfit", overfit), ("Poor", poor), ("Blocked", blocked)]:
        lines.append(f"### {grp_name}")
        if grp:
            for r in grp:
                lines.append(f"- {r['strategy']}: {r.get('status')} ({r.get('reason','n/a')})")
        else:
            lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Run autonomous migration + validation")
    p.add_argument("--base", default=".", help="Project root")
    p.add_argument("--n-perms-is", type=int, default=1000)
    p.add_argument("--n-perms-wf", type=int, default=200)
    p.add_argument("--train-years", type=float, default=4.0)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--bars-per-day", type=int, default=24)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    reports = base / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Starting migration run...")

    discovered = discover_strategies(base)
    write_strategy_inventory(reports / "STRATEGY_INVENTORY.md", discovered)

    data_items = data_inventory(base)
    write_data_inventory(reports / "DATA_INVENTORY.md", data_items)
    data_path = _best_data(data_items)
    if data_path is None:
        print("No clean OHLC dataset (CSV/Parquet) found. Stopping.")
        return 1

    results: List[dict] = []

    # Seed with native strategies first.
    for native in sorted(set(["donchian", "ema_crossover"] + list(STRATEGY_REGISTRY.keys()))):
        if native in {"ema", "donchian_breakout"}:
            continue
        try:
            val = attempt_with_retry(
                lambda: run_full_validation(
                    strategy_name=native,
                    data_path=data_path,
                    n_perms_is=args.n_perms_is,
                    n_perms_wf=args.n_perms_wf,
                    train_years=args.train_years,
                    retrain_days=args.retrain_days,
                    bars_per_day=args.bars_per_day,
                    output_root=str(reports),
                ),
                max_attempts=3,
                label=f"validate_native:{native}",
            )
            update_validation_status(native, val["verdict"])
            results.append(
                {
                    "strategy": native,
                    "source": "native",
                    "status": val["verdict"],
                    "is_pf": val["is_pf"],
                    "oos_pf": val["oos_pf"],
                    "is_pvalue": val["is_pvalue"],
                    "wf_pvalue": val["wf_pvalue"],
                    "report": val["report_path"],
                    "reason": val.get("failed_step"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "strategy": native,
                    "source": "native",
                    "status": "BLOCKED",
                    "reason": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
        write_progress(reports / "MIGRATION_PROGRESS.md", results)

    # Translate and validate discovered pine files.
    pipeline = TranslationPipeline(strategy_dir=base / "src" / "strategies")
    for sf in discovered:
        try:
            code = sf.path.read_text(encoding="utf-8", errors="ignore")
            trans = attempt_with_retry(
                lambda: pipeline.translate(code, auto_save=True),
                max_attempts=3,
                label=f"translate:{sf.name}",
            )
            if not trans["success"] or trans.get("manual_review_needed", False):
                results.append(
                    {
                        "strategy": sf.name,
                        "source": str(sf.path),
                        "status": "TRANSLATION_FAILED",
                        "reason": "; ".join(list(trans.get("issues", [])) + list(trans.get("warnings", []))) or "unknown",
                    }
                )
                write_progress(reports / "MIGRATION_PROGRESS.md", results)
                continue

            val = attempt_with_retry(
                lambda: run_full_validation(
                    strategy_name=trans["strategy_name"],
                    data_path=data_path,
                    n_perms_is=args.n_perms_is,
                    n_perms_wf=args.n_perms_wf,
                    train_years=args.train_years,
                    retrain_days=args.retrain_days,
                    bars_per_day=args.bars_per_day,
                    output_root=str(reports),
                ),
                max_attempts=3,
                label=f"validate_translated:{trans['strategy_name']}",
            )
            update_validation_status(trans["strategy_name"], val["verdict"])
            results.append(
                {
                    "strategy": trans["strategy_name"],
                    "source": str(sf.path),
                    "status": val["verdict"],
                    "is_pf": val["is_pf"],
                    "oos_pf": val["oos_pf"],
                    "is_pvalue": val["is_pvalue"],
                    "wf_pvalue": val["wf_pvalue"],
                    "report": val["report_path"],
                    "reason": val.get("failed_step"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "strategy": sf.name,
                    "source": str(sf.path),
                    "status": "ERROR",
                    "reason": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
        write_progress(reports / "MIGRATION_PROGRESS.md", results)

    write_final_summary(reports / "FINAL_MIGRATION_SUMMARY.md", results)
    (reports / "migration_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] Migration run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
