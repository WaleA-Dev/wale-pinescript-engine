"""
TradingView-style broker emulator for transpiled Pine strategies.

Execution model (matches TradingView's broker emulator):
- strategy.entry / strategy.close called during bar N fill at bar N+1 open.
- strategy.exit stop/limit are standing orders checked intrabar from bar N+1.
- Intrabar path assumption (per TV docs): if the bar's open is closer to the
  high, price travels open->high->low->close; if closer to the low,
  open->low->high->close. Gaps through a level fill at open.
- Percent-of-equity sizing, whole-share quantities, percent commission per fill.
- Reversal entries (long entry while short) close and flip in one fill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BrokerConfig:
    initial_capital: float = 100000.0
    qty_type: str = "percent_of_equity"  # or "fixed", "cash_per_order"
    qty_value: float = 100.0
    commission_pct: float = 0.0          # percent per fill, e.g. 0.05 = 0.05%
    pyramiding: int = 1
    allow_fractional: bool = False


@dataclass
class TradeRecord:
    direction: str                # "LONG" | "SHORT"
    entry_bar: int
    entry_price: float
    qty: float
    entry_id: str = "Long"
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0


class Broker:
    """Per-run broker. Generated strategies call the strategy.* mapped methods."""

    def __init__(self, config: BrokerConfig, n_bars: int):
        self.cfg = config
        self.n = n_bars
        self.cash = config.initial_capital
        self.position_size = 0.0          # signed share qty
        self.position_avg_price = float("nan")
        self.entry_bar_index: int = -1
        self.current_entry_id = ""

        self._pending_entry: Optional[Dict[str, Any]] = None
        self._pending_close: bool = False
        self._pending_close_reason: str = ""
        # standing exit orders: active from the bar AFTER they are placed
        self._exit_order: Optional[Dict[str, Any]] = None
        self._exit_order_active: Optional[Dict[str, Any]] = None

        self.trades: List[TradeRecord] = []
        self.equity_curve = np.full(n_bars, np.nan)
        self.position_hist = np.zeros(n_bars)  # signed qty at each bar close
        self._i = -1
        self._open = self._high = self._low = self._close = float("nan")

    # ── engine side ─────────────────────────────────────────────────────────

    @property
    def closedtrades(self) -> int:
        return sum(1 for t in self.trades if t.exit_bar is not None)

    @property
    def wintrades(self) -> int:
        return sum(1 for t in self.trades if t.exit_bar is not None and t.pnl > 0)

    @property
    def losstrades(self) -> int:
        return sum(1 for t in self.trades if t.exit_bar is not None and t.pnl < 0)

    @property
    def eventrades(self) -> int:
        return sum(1 for t in self.trades if t.exit_bar is not None and t.pnl == 0)

    def begin_bar(self, i: int, o: float, h: float, l: float, c: float):
        self._i = i
        self._open, self._high, self._low, self._close = o, h, l, c

        # 1. Market fills queued on the previous bar happen at this open.
        if self._pending_close and self.position_size != 0:
            self._fill_exit(o, self._pending_close_reason or "Close")
        self._pending_close = False
        self._pending_close_reason = ""

        if self._pending_entry is not None:
            pe = self._pending_entry
            self._pending_entry = None
            self._fill_entry(pe["id"], pe["direction"], o)

        # 2. Exit orders placed on the previous bar become active now.
        if self._exit_order is not None:
            self._exit_order_active = self._exit_order
            self._exit_order = None

        # 3. Check standing stop/limit intrabar.
        self._check_exit_orders(o, h, l, c)

    def end_bar(self, i: int, c: float):
        self.position_hist[i] = self.position_size
        self.equity_curve[i] = self.equity(c)

    def equity(self, price: Optional[float] = None) -> float:
        p = self._close if price is None else price
        pos_val = self.position_size * p if self.position_size != 0 and not math.isnan(p) else 0.0
        return self.cash + pos_val

    # ── strategy.* API (called from on_bar) ─────────────────────────────────

    def entry(self, entry_id: str, direction: int, qty: Optional[float] = None):
        """strategy.entry — queue a market entry (fills next bar open)."""
        want = 1 if direction >= 0 else -1
        have = 0 if self.position_size == 0 else (1 if self.position_size > 0 else -1)
        if have == want:
            return  # pyramiding=1: ignore add-on entries in same direction
        self._pending_entry = {"id": entry_id, "direction": want, "qty": qty}

    def close(self, entry_id: Optional[str] = None, comment: str = ""):
        """strategy.close — queue market exit at next bar open."""
        if self.position_size != 0 or self._pending_entry is not None:
            self._pending_close = True
            self._pending_close_reason = comment or "Close"
            self._pending_entry = None  # close cancels a queued entry

    def close_all(self, comment: str = ""):
        self.close(None, comment or "Close All")

    def exit(self, exit_id: str, from_entry: Optional[str] = None,
             stop: Optional[float] = None, limit: Optional[float] = None,
             trail_points: Optional[float] = None, trail_offset: Optional[float] = None,
             loss: Optional[float] = None, profit: Optional[float] = None,
             comment: str = "", **_ignored):
        """strategy.exit — create/refresh standing stop/limit/trailing orders."""
        order = {
            "id": exit_id,
            "stop": _clean(stop), "limit": _clean(limit),
            "trail_points": _clean(trail_points), "trail_offset": _clean(trail_offset),
            "loss_ticks": _clean(loss), "profit_ticks": _clean(profit),
            "trail_armed": False, "trail_stop": float("nan"),
            "comment": comment,
        }
        if self._exit_order_active is not None and self.position_size != 0:
            # Refreshing an active order keeps trailing state
            prev = self._exit_order_active
            order["trail_armed"] = prev.get("trail_armed", False)
            order["trail_stop"] = prev.get("trail_stop", float("nan"))
            self._exit_order_active = order
        else:
            self._exit_order = order

    def cancel(self, *_a, **_k):
        self._exit_order = None
        self._exit_order_active = None

    cancel_all = cancel

    def position_size_at(self, bars_back: int) -> float:
        """strategy.position_size[k]"""
        idx = self._i - bars_back
        if idx < 0:
            return 0.0
        if bars_back <= 0:
            return self.position_size
        return float(self.position_hist[idx])

    # ── fills ────────────────────────────────────────────────────────────────

    def _order_qty(self, price: float, qty_override: Optional[float]) -> float:
        if qty_override is not None:
            return float(qty_override)
        cfg = self.cfg
        if cfg.qty_type == "percent_of_equity":
            budget = self.equity(price) * cfg.qty_value / 100.0
        elif cfg.qty_type == "cash_per_order":
            budget = cfg.qty_value
        else:  # fixed
            return float(cfg.qty_value)
        if price <= 0 or math.isnan(price):
            return 0.0
        qty = budget / price
        return qty if self.cfg.allow_fractional else float(math.floor(qty))

    def _fill_entry(self, entry_id: str, direction: int, price: float):
        if self.position_size != 0:
            if (self.position_size > 0) == (direction > 0):
                return
            self._fill_exit(price, "Reverse")
        qty = self._order_qty(price, None)
        if qty <= 0:
            return
        signed = qty * (1 if direction > 0 else -1)
        cost = qty * price
        commission = cost * self.cfg.commission_pct / 100.0
        self.cash -= signed * price  # short sale adds cash
        self.cash -= commission
        self.position_size = signed
        self.position_avg_price = price
        self.entry_bar_index = self._i
        self.current_entry_id = entry_id
        self.trades.append(TradeRecord(
            direction="LONG" if direction > 0 else "SHORT",
            entry_bar=self._i, entry_price=price, qty=qty, entry_id=entry_id,
        ))

    def _fill_exit(self, price: float, reason: str):
        if self.position_size == 0:
            return
        qty = abs(self.position_size)
        sign = 1 if self.position_size > 0 else -1
        proceeds = self.position_size * price
        commission = qty * price * self.cfg.commission_pct / 100.0
        self.cash += proceeds
        self.cash -= commission
        tr = self.trades[-1] if self.trades and self.trades[-1].exit_bar is None else None
        if tr is not None:
            tr.exit_bar = self._i
            tr.exit_price = price
            tr.exit_reason = reason
            entry_comm = tr.qty * tr.entry_price * self.cfg.commission_pct / 100.0
            tr.pnl = sign * (price - tr.entry_price) * qty - commission - entry_comm
            denom = tr.entry_price * qty
            tr.pnl_pct = 100.0 * tr.pnl / denom if denom else 0.0
        self.position_size = 0.0
        self.position_avg_price = float("nan")
        self.current_entry_id = ""
        self._exit_order_active = None
        self._exit_order = None

    # ── standing order evaluation ────────────────────────────────────────────

    def _resolve_levels(self, order: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        """Compute effective stop/limit for the current position."""
        long_pos = self.position_size > 0
        stop = order.get("stop")
        limit = order.get("limit")
        avg = self.position_avg_price

        # loss/profit in ticks ~ treated as price offset (mintick unknown; assume 0.01)
        tick = 0.01
        if stop is None and order.get("loss_ticks") is not None:
            stop = avg - order["loss_ticks"] * tick if long_pos else avg + order["loss_ticks"] * tick
        if limit is None and order.get("profit_ticks") is not None:
            limit = avg + order["profit_ticks"] * tick if long_pos else avg - order["profit_ticks"] * tick

        # trailing: arm at trail_points profit (in price units here), trail by trail_offset
        tp, toff = order.get("trail_points"), order.get("trail_offset")
        if tp is not None or toff is not None:
            extreme = self._high if long_pos else self._low
            arm_level = avg + tp * tick if long_pos else avg - tp * tick if tp is not None else avg
            if not order["trail_armed"]:
                if (long_pos and self._high >= arm_level) or (not long_pos and self._low <= arm_level):
                    order["trail_armed"] = True
            if order["trail_armed"]:
                off = (toff if toff is not None else tp or 0.0) * tick
                new_stop = extreme - off if long_pos else extreme + off
                prev = order["trail_stop"]
                if math.isnan(prev):
                    order["trail_stop"] = new_stop
                else:
                    order["trail_stop"] = max(prev, new_stop) if long_pos else min(prev, new_stop)
                ts = order["trail_stop"]
                stop = ts if stop is None else (max(stop, ts) if long_pos else min(stop, ts))
        return stop, limit

    def _check_exit_orders(self, o: float, h: float, l: float, c: float):
        order = self._exit_order_active
        if order is None or self.position_size == 0:
            return
        stop, limit = self._resolve_levels(order)
        if stop is None and limit is None:
            return
        long_pos = self.position_size > 0
        reason = order.get("comment") or order.get("id") or "Exit"

        def hit_stop():
            self._fill_exit(min(o, stop) if long_pos else max(o, stop), f"{reason} (stop)")

        def hit_limit():
            self._fill_exit(max(o, limit) if long_pos else min(o, limit), f"{reason} (limit)")

        stop_hit = stop is not None and ((l <= stop) if long_pos else (h >= stop))
        limit_hit = limit is not None and ((h >= limit) if long_pos else (l <= limit))

        if stop_hit and limit_hit:
            # TV broker emulator: if the open is closer to the high, price is
            # assumed to travel open->high->low->close; if closer to the low,
            # open->low->high->close.
            high_first = (h - o) < (o - l)
            if long_pos:
                hit_limit() if high_first else hit_stop()
            else:
                hit_stop() if high_first else hit_limit()
        elif stop_hit:
            hit_stop()
        elif limit_hit:
            hit_limit()


def _clean(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# ── Result assembly ──────────────────────────────────────────────────────────

def build_result(broker: Broker, df) -> Dict[str, Any]:
    """Convert broker state into the web-app result payload."""
    eq = np.nan_to_num(broker.equity_curve, nan=broker.cfg.initial_capital)
    eq_norm = eq / broker.cfg.initial_capital
    prev = np.concatenate(([1.0], eq_norm[:-1]))
    bar_ret = eq_norm / np.maximum(1e-12, prev) - 1.0

    closed = [t for t in broker.trades if t.exit_bar is not None]
    open_trades = [t for t in broker.trades if t.exit_bar is None]

    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

    peak = np.maximum.accumulate(eq_norm)
    dd = (eq_norm - peak) / peak
    max_dd = float(abs(dd.min())) if len(dd) else 0.0

    r = bar_ret[~np.isnan(bar_ret)]
    std = float(np.std(r, ddof=1)) if len(r) > 2 else 0.0
    sharpe = float(np.mean(r) / std * np.sqrt(252)) if std > 0 else 0.0

    total_return = float(eq_norm[-1] - 1.0) if len(eq_norm) else 0.0
    n_bars = max(1, len(eq_norm))
    annual = float(eq_norm[-1] ** (252.0 / n_bars) - 1.0) if eq_norm[-1] > 0 else 0.0

    metrics = {
        "profit_factor": float(pf) if not math.isinf(pf) else 999.0,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "avg_win": float(np.mean([t.pnl_pct for t in wins]) / 100.0) if wins else 0.0,
        "avg_loss": float(np.mean([t.pnl_pct for t in losses]) / 100.0) if losses else 0.0,
        "num_trades": len(closed),
        "expectancy": float(np.mean([t.pnl_pct for t in closed]) / 100.0) if closed else 0.0,
        "calmar_ratio": (annual / max_dd) if max_dd > 0 else 0.0,
        "net_profit": float(eq[-1] - broker.cfg.initial_capital) if len(eq) else 0.0,
    }

    dates = df.index

    def fmt(idx):
        d = dates[idx]
        s = str(d)
        if getattr(d, "hour", 0) or getattr(d, "minute", 0):
            return s[:16]
        return s[:10]

    trade_rows = []
    for k, t in enumerate(closed + open_trades):
        row = {
            "id": k + 1,
            "direction": t.direction,
            "entry_date": fmt(t.entry_bar),
            "entry_bar": int(t.entry_bar),
            "entry_price": round(float(t.entry_price), 4),
            "exit_date": fmt(t.exit_bar) if t.exit_bar is not None else "OPEN",
            "exit_bar": int(t.exit_bar) if t.exit_bar is not None else None,
            "exit_price": round(float(t.exit_price), 4) if t.exit_price is not None else None,
            "pnl_pct": round(float(t.pnl_pct), 2),
            "bars_held": int((t.exit_bar if t.exit_bar is not None else len(dates) - 1) - t.entry_bar),
            "reason": t.exit_reason,
        }
        trade_rows.append(row)
    trade_rows.sort(key=lambda x: x["entry_bar"])
    for k, row in enumerate(trade_rows):
        row["id"] = k + 1

    return {
        "metrics": metrics,
        "equity_curve": eq_norm,
        "drawdown_curve": dd * 100.0,
        "trades": trade_rows,
        "position_hist": broker.position_hist,
    }
