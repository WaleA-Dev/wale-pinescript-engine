"""Plot helpers for validation reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_permutation_distribution(
    real_value: float,
    perm_values: list[float] | np.ndarray,
    metric_name: str,
    output_path: str | Path,
) -> str:
    """Create permutation histogram and return saved path."""
    vals = np.asarray(perm_values, dtype=float)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=35, alpha=0.75, color="#457B9D", edgecolor="#1D3557")
    ax.axvline(real_value, color="#E63946", linewidth=2.0, label=f"Real {metric_name}")
    ax.set_title(f"Permutation Distribution: {metric_name}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def plot_equity_curve(returns: pd.Series | np.ndarray, title: str, output_path: str | Path) -> str:
    """Create equity curve from log-return series and return saved path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    r = pd.Series(returns, dtype=float).fillna(0.0)
    eq = np.exp(r.cumsum())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(eq.to_numpy(), color="#264653", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity (normalized)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)
