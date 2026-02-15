"""
Bar-return computation and objective metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

ANNUALIZATION_DAILY = np.sqrt(252)
ANNUALIZATION_HOURLY = np.sqrt(252 * 24)


def compute_bar_returns(
    df: pd.DataFrame,
    signals=None,
    signal_col: str = "signal",
    commission: float = 0.0,
) -> pd.DataFrame:
    """
    Compute bar-level strategy returns with next-bar-open execution.

    Matches TradingView behaviour: signal on bar N close -> fill at bar N+1 open.
    On bars where position changes, the return is from open to close of that bar.
    On bars where position is held, the return is close-to-close.
    Accepts either a signals array or a signal column name in df.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    if (close <= 0).any():
        raise ValueError("close prices must be strictly positive")

    has_open = "open" in df.columns
    if has_open:
        opn = pd.to_numeric(out["open"], errors="coerce")

    if signals is not None:
        position = pd.Series(signals, index=df.index, dtype="float64").fillna(0.0)
    elif signal_col in df.columns:
        position = pd.to_numeric(out[signal_col], errors="coerce").fillna(0.0)
    else:
        raise ValueError(f"No signals provided and '{signal_col}' column not found")

    # Shift position by 1: signal on bar N -> position active from bar N+1
    active_pos = position.shift(1).fillna(0.0)

    if has_open:
        # Detect bars where position changes (entry/exit/reversal)
        pos_changed = active_pos != active_pos.shift(1)

        # Default: close-to-close return
        bar_ret = close.pct_change()

        # On position-change bars: use open-to-close return (filled at open)
        open_to_close_ret = (close - opn) / opn
        bar_ret = bar_ret.where(~pos_changed, open_to_close_ret)
    else:
        bar_ret = close.pct_change()

    strategy_ret = active_pos * bar_ret

    # Subtract commission on position changes
    if commission > 0:
        trades = active_pos.diff().abs()
        strategy_ret = strategy_ret - trades * commission

    equity = (1 + strategy_ret.fillna(0)).cumprod()

    out["return"] = bar_ret
    out["signal"] = position
    out["strategy_return"] = strategy_ret
    out["equity_curve"] = equity
    return out


def compute_metrics(strategy_returns: pd.Series | np.ndarray) -> Dict[str, float]:
    """
    Compute all performance metrics from return series.
    """
    r = pd.Series(strategy_returns, dtype="float64").dropna()
    if r.empty:
        return {
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "num_trades": 0,
            "expectancy": 0.0,
            "calmar_ratio": 0.0,
        }

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    std = float(r.std(ddof=1))
    sharpe = float((r.mean() / std) * ANNUALIZATION_DAILY) if std > 0 else 0.0

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min().item()) if len(dd) else 0.0
    max_dd = abs(max_dd)

    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0

    win_rate = float(len(wins) / len(r)) if len(r) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    num_trades = int((r != 0).sum())
    expectancy = float(r.mean()) if len(r) else 0.0

    # Annualize return: CAGR = (final_equity)^(252/n_bars) - 1
    n_bars = len(r)
    final_equity = equity.iloc[-1] if len(equity) else 1.0
    if n_bars > 0 and final_equity > 0:
        annual_return = float(final_equity ** (252.0 / n_bars) - 1.0)
    else:
        annual_return = 0.0
    calmar = float(annual_return / max_dd) if max_dd > 0 else 0.0

    return {
        "profit_factor": float(pf),
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "num_trades": num_trades,
        "expectancy": expectancy,
        "calmar_ratio": calmar,
    }
