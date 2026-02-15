"""End-to-end 4-step validation workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..bar_returns import compute_bar_returns, compute_metrics
from ..optimization import grid_search
from ..plotting import plot_equity_curve, plot_permutation_distribution
from ..strategies import get_strategy_class
from .in_sample_permutation import in_sample_permutation_test
from .walk_forward import walk_forward_backtest, walk_forward_permutation_test

OHLC = ["open", "high", "low", "close"]


def _load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    missing = [c for c in OHLC if c not in cols]
    if missing:
        raise ValueError(f"Missing OHLC columns in data: {missing}")

    out = pd.DataFrame({c: pd.to_numeric(df[cols[c]], errors="coerce") for c in OHLC})
    if any(c in cols for c in ("time", "date", "datetime", "timestamp")):
        tcol = next(cols[c] for c in ("time", "date", "datetime", "timestamp") if c in cols)
        out["timestamp"] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    return out.dropna(subset=OHLC).reset_index(drop=True)


def _get_param_grid(strategy_cls):
    # Try instance method first
    try:
        instance = strategy_cls()
        grid = instance.param_grid()
        if grid:
            return grid
    except Exception:
        pass

    # Fallback to module-level
    if hasattr(strategy_cls, "PARAM_GRID_DEFAULT"):
        return getattr(strategy_cls, "PARAM_GRID_DEFAULT")
    mod = __import__(strategy_cls.__module__, fromlist=["dummy"])
    if hasattr(mod, "PARAM_GRID_DEFAULT"):
        return getattr(mod, "PARAM_GRID_DEFAULT")
    return [{}]


def _write_report(report_path: Path, lines: list[str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_full_validation(
    strategy_name: str,
    data_path: str,
    n_perms_is: int = 1000,
    n_perms_wf: int = 200,
    train_years: float = 4.0,
    retrain_days: int = 30,
    bars_per_day: int = 1,
    output_root: str = "reports",
    quick: bool = False,
) -> Dict:
    """Execute all 4 validation steps and generate report + plots."""
    strategy_cls = get_strategy_class(strategy_name)
    param_grid = _get_param_grid(strategy_cls)

    df = _load_data(data_path)
    train_bars = int(train_years * 252 * bars_per_day)
    if train_bars >= len(df):
        train_bars = max(20, int(len(df) * 0.7))
        train_years = train_bars / float(252 * bars_per_day)

    df_train = df.iloc[:train_bars].copy()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_key = strategy_name.lower().strip()

    plots_dir = Path(output_root) / "plots"
    reports_dir = Path(output_root) / "strategy_validations"
    report_path = reports_dir / f"{strategy_key}_validation_{stamp}.md"

    print(f"\n--- Step 1: In-Sample Optimization ({strategy_name}) ---")
    best_params, is_pf = grid_search(df_train, strategy_cls, param_grid)
    print(f"  Best params: {best_params}")
    print(f"  In-sample PF: {is_pf:.4f}")

    if quick:
        # Quick mode: skip permutation tests
        strategy = strategy_cls(**best_params)
        signals = strategy.generate_signals(df)
        run_df = compute_bar_returns(df, signals=signals)
        metrics = compute_metrics(run_df["strategy_return"])

        lines = [
            f"# Quick Validation Report: {strategy_name}",
            f"Date: {datetime.now().isoformat(sep=' ', timespec='seconds')}",
            f"Data: {data_path}",
            f"Bars: {len(df)}",
            "",
            "## Step 1: In-Sample Optimization",
            f"- Best params: {best_params}",
            f"- In-sample PF: {is_pf:.4f}",
            "",
            "## Metrics (full data, best params)",
            f"- Profit Factor: {metrics['profit_factor']:.4f}",
            f"- Sharpe Ratio: {metrics['sharpe_ratio']:.4f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.2%}",
            f"- Total Return: {metrics['total_return']:.2%}",
            f"- Win Rate: {metrics['win_rate']:.2%}",
            f"- Num Trades: {metrics['num_trades']}",
            "",
            "## Verdict",
            "- Mode: QUICK (permutation tests skipped)",
        ]
        _write_report(report_path, lines)
        print(f"  Quick metrics: PF={metrics['profit_factor']:.4f}, Sharpe={metrics['sharpe_ratio']:.4f}")
        return {
            "strategy_name": strategy_name,
            "best_params": best_params,
            "is_pf": float(is_pf),
            "metrics": metrics,
            "verdict": "QUICK",
            "report_path": str(report_path),
        }

    print(f"\n--- Step 2: In-Sample Permutation Test ({n_perms_is} perms) ---")
    np.random.seed(7)
    is_res = in_sample_permutation_test(df_train, strategy_cls, param_grid, n_perms=n_perms_is)
    print(f"  Real PF: {is_res['real_pf']:.4f}")
    print(f"  p-value: {is_res['p_value']:.6f}")
    print(f"  Passed (<0.01): {is_res['passed']}")

    print(f"\n--- Step 3: Walk-Forward Test ---")
    oos_returns = walk_forward_backtest(
        df=df, strategy_class=strategy_cls, best_params=best_params,
        train_years=train_years, retrain_days=retrain_days,
        param_grid=param_grid, bars_per_day=bars_per_day,
    )
    oos_metrics = compute_metrics(oos_returns)
    print(f"  OOS PF: {oos_metrics['profit_factor']:.4f}")
    print(f"  OOS Sharpe: {oos_metrics['sharpe_ratio']:.4f}")

    print(f"\n--- Step 4: Walk-Forward Permutation Test ({n_perms_wf} perms) ---")
    np.random.seed(9)
    wf_res = walk_forward_permutation_test(
        df=df, strategy_class=strategy_cls, best_params=best_params,
        n_perms=n_perms_wf, train_years=train_years, retrain_days=retrain_days,
        param_grid=param_grid, bars_per_day=bars_per_day,
    )
    print(f"  Real OOS PF: {wf_res['real_oos_pf']:.4f}")
    print(f"  p-value: {wf_res['p_value']:.6f}")
    print(f"  Passed: {wf_res['passed']}")

    if is_res["p_value"] >= 0.05:
        verdict = "OVERFIT"
        failed_step = "STEP_2_IS_PERMUTATION"
        failed_pvalue = is_res["p_value"]
    elif wf_res["p_value"] >= wf_res["threshold"]:
        verdict = "OVERFIT"
        failed_step = "STEP_4_WF_PERMUTATION"
        failed_pvalue = wf_res["p_value"]
    elif oos_metrics["profit_factor"] < 1.1:
        verdict = "POOR"
        failed_step = "STEP_3_OOS_WEAK"
        failed_pvalue = None
    else:
        verdict = "VALIDATED"
        failed_step = None
        failed_pvalue = None

    is_perm_plot = plot_permutation_distribution(
        real_value=is_res["real_pf"], perm_values=is_res["perm_pfs"],
        metric_name="In-Sample Profit Factor",
        output_path=plots_dir / f"{strategy_key}_is_perm_hist_{stamp}.png",
    )
    wf_perm_plot = plot_permutation_distribution(
        real_value=wf_res["real_oos_pf"], perm_values=wf_res["perm_oos_pfs"],
        metric_name="Walk-Forward Profit Factor",
        output_path=plots_dir / f"{strategy_key}_wf_perm_hist_{stamp}.png",
    )
    wf_equity_plot = plot_equity_curve(
        returns=oos_returns, title=f"{strategy_name} Walk-Forward Equity Curve",
        output_path=plots_dir / f"{strategy_key}_wf_equity_{stamp}.png",
    )

    lines = [
        f"# Validation Report: {strategy_name}",
        f"Date: {datetime.now().isoformat(sep=' ', timespec='seconds')}",
        f"Data: {data_path}",
        f"Bars: {len(df)}",
        "",
        "## Step 1: In-Sample Optimization",
        f"- Best params: {best_params}",
        f"- In-sample PF: {is_pf:.4f}",
        "",
        "## Step 2: In-Sample Permutation Test",
        f"- Permutations: {n_perms_is}",
        f"- Real PF: {is_res['real_pf']:.4f}",
        f"- p-value: {is_res['p_value']:.6f}",
        f"- Passed (<0.01): {is_res['passed']}",
        f"- Plot: {is_perm_plot}",
        "",
        "## Step 3: Walk-Forward Test",
        f"- OOS PF: {oos_metrics['profit_factor']:.4f}",
        f"- OOS Sharpe: {oos_metrics['sharpe_ratio']:.4f}",
        f"- OOS Max DD: {oos_metrics['max_drawdown']:.2%}",
        f"- OOS Total Return: {oos_metrics['total_return']:.2%}",
        f"- Plot: {wf_equity_plot}",
        "",
        "## Step 4: Walk-Forward Permutation Test",
        f"- Permutations: {n_perms_wf}",
        f"- Real OOS PF: {wf_res['real_oos_pf']:.4f}",
        f"- p-value: {wf_res['p_value']:.6f}",
        f"- Threshold: {wf_res['threshold']:.4f}",
        f"- Passed: {wf_res['passed']}",
        f"- Plot: {wf_perm_plot}",
        "",
        "## Final Verdict",
        f"- Verdict: **{verdict}**",
    ]
    if failed_step:
        lines.append(f"- Failed step: {failed_step}")
    if failed_pvalue is not None:
        lines.append(f"- Failed p-value: {failed_pvalue:.6f}")

    _write_report(report_path, lines)

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    return {
        "strategy_name": strategy_name,
        "best_params": best_params,
        "is_pf": float(is_pf),
        "is_pvalue": float(is_res["p_value"]),
        "oos_pf": float(oos_metrics["profit_factor"]),
        "oos_sharpe": float(oos_metrics["sharpe_ratio"]),
        "oos_total_return": float(oos_metrics["total_return"]),
        "oos_max_drawdown": float(oos_metrics["max_drawdown"]),
        "wf_pvalue": float(wf_res["p_value"]),
        "wf_threshold": float(wf_res["threshold"]),
        "failed_step": failed_step,
        "failed_pvalue": failed_pvalue,
        "verdict": verdict,
        "report_path": str(report_path),
        "plots": {
            "is_perm_hist": is_perm_plot,
            "wf_perm_hist": wf_perm_plot,
            "wf_equity": wf_equity_plot,
        },
    }
