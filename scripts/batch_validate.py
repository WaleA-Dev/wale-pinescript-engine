#!/usr/bin/env python3
"""Batch-validate all registered strategies sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.strategies import STRATEGY_REGISTRY
from src.validation.full_validation import run_full_validation


def parse_args():
    p = argparse.ArgumentParser(description="Batch-validate all strategies")
    p.add_argument("--data", required=True, help="CSV data path")
    p.add_argument("--quick", action="store_true", help="Quick mode")
    p.add_argument("--n-perms-is", type=int, default=200)
    p.add_argument("--n-perms-wf", type=int, default=100)
    p.add_argument("--train-years", type=float, default=4.0)
    p.add_argument("--output-root", default="reports")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seen = set()
    results = []

    for name in sorted(STRATEGY_REGISTRY.keys()):
        cls = STRATEGY_REGISTRY[name]
        if cls in seen:
            continue
        seen.add(cls)

        print(f"\n{'='*60}")
        print(f"VALIDATING: {name}")
        print(f"{'='*60}")

        try:
            result = run_full_validation(
                strategy_name=name,
                data_path=args.data,
                n_perms_is=args.n_perms_is,
                n_perms_wf=args.n_perms_wf,
                train_years=args.train_years,
                output_root=args.output_root,
                quick=args.quick,
            )
            results.append(result)
            print(f"  -> {result['verdict']}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({"strategy_name": name, "verdict": "ERROR", "error": str(e)})

    print(f"\n{'='*60}")
    print("BATCH RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['strategy_name']:20s} -> {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
