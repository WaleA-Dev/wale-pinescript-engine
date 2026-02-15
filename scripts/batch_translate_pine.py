#!/usr/bin/env python3
"""
Batch translate and validate Pine files.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pine_translator.pipeline import TranslationPipeline
from src.validation.full_validation import run_full_validation


def translate_and_validate_one(
    pine_path: str,
    data_path: str,
    n_perms_is: int,
    n_perms_wf: int,
    train_years: float,
    retrain_days: int,
    bars_per_day: int,
    output_root: str,
) -> dict:
    pine_file = Path(pine_path)
    try:
        code = pine_file.read_text(encoding="utf-8")
        trans = TranslationPipeline().translate(code, auto_save=True)
        if not trans["success"] or trans.get("manual_review_needed", False):
            issues = list(trans.get("issues", [])) + list(trans.get("warnings", []))
            return {"file": pine_file.name, "status": "TRANSLATION_FAILED", "issues": issues}

        result = run_full_validation(
            strategy_name=trans["strategy_name"],
            data_path=data_path,
            n_perms_is=n_perms_is,
            n_perms_wf=n_perms_wf,
            train_years=train_years,
            retrain_days=retrain_days,
            bars_per_day=bars_per_day,
            output_root=output_root,
        )
        return {
            "file": pine_file.name,
            "strategy": trans["strategy_name"],
            "status": result["verdict"],
            "is_pf": result["is_pf"],
            "oos_pf": result["oos_pf"],
            "is_pvalue": result["is_pvalue"],
            "wf_pvalue": result["wf_pvalue"],
            "report": result["report_path"],
        }
    except Exception as exc:
        return {"file": pine_file.name, "status": "ERROR", "error": str(exc)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory containing .pine files")
    p.add_argument("--data", required=True, help="OHLC CSV file")
    p.add_argument("--n-perms-is", type=int, default=1000)
    p.add_argument("--n-perms-wf", type=int, default=200)
    p.add_argument("--train-years", type=float, default=4.0)
    p.add_argument("--retrain-days", type=int, default=30)
    p.add_argument("--bars-per-day", type=int, default=24)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--output-root", default="reports")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pine_files = sorted(Path(args.input_dir).glob("*.pine"))
    if not pine_files:
        print(f"No .pine files found in {args.input_dir}")
        return 1

    print(f"Found {len(pine_files)} Pine files")
    results = []

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(
                    translate_and_validate_one,
                    str(pf),
                    args.data,
                    args.n_perms_is,
                    args.n_perms_wf,
                    args.train_years,
                    args.retrain_days,
                    args.bars_per_day,
                    args.output_root,
                ): pf
                for pf in pine_files
            }
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                print(f"[{res.get('file', '?')}] {res.get('status', 'UNKNOWN')}")
    else:
        for pf in pine_files:
            res = translate_and_validate_one(
                str(pf),
                args.data,
                args.n_perms_is,
                args.n_perms_wf,
                args.train_years,
                args.retrain_days,
                args.bars_per_day,
                args.output_root,
            )
            results.append(res)
            print(f"[{res.get('file', '?')}] {res.get('status', 'UNKNOWN')}")

    summary = {
        "total": len(results),
        "validated": sum(1 for r in results if r.get("status") == "VALIDATED"),
        "overfit": sum(1 for r in results if r.get("status") == "OVERFIT"),
        "poor": sum(1 for r in results if r.get("status") == "POOR"),
        "failed": sum(1 for r in results if r.get("status") in {"ERROR", "TRANSLATION_FAILED"}),
        "results": results,
    }

    out = Path(args.output_root) / "BATCH_TRANSLATION_SUMMARY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
