"""Regression guards for the single-file frontend (templates/converge.html).

An open position serializes with exit_price=null / exit_date="OPEN"
(see src/pine_runtime.py). The trades table and CSV export must never call
`.toFixed()` on a value that can be null, or the whole results view throws
"Cannot read properties of null (reading 'toFixed')" and renders nothing.
These tests assert every nullable-field `.toFixed(...)` sits behind a null
guard on the same line.
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "converge.html"

# Fields the backend can emit as null: exit_price for an open trade
# (src/pine_runtime.py), and the permutation p-values when validation runs in
# QUICK mode or returns partial results. pnl_pct is always a float from the
# backend (0.0 while open) so it can't throw, but its cell is still gated on
# exit_price for display.
NULLABLE = ("exit_price", "is_pvalue", "wf_pvalue")


def _lines():
    return TEMPLATE.read_text(encoding="utf-8").splitlines()


def test_template_exists():
    assert TEMPLATE.exists(), TEMPLATE


def test_no_unguarded_toFixed_on_nullable_fields():
    """Every line that calls .toFixed on a nullable field must also test that
    field for null (`!= null` / `== null`) on the same line."""
    offenders = []
    for n, line in enumerate(_lines(), 1):
        for field in NULLABLE:
            if re.search(rf"\b\w+\.{field}\.toFixed", line):
                guarded = (f"{field} != null" in line) or (f"{field} == null" in line)
                if not guarded:
                    offenders.append(f"L{n}: {line.strip()}")
    assert not offenders, "unguarded .toFixed on nullable field:\n" + "\n".join(offenders)


def test_open_trade_exit_price_is_guarded_in_trade_row():
    """The exact crash site: the exit-price cell in the trades table."""
    text = TEMPLATE.read_text(encoding="utf-8")
    # the pre-fix code was: '<td>$' + t.exit_price.toFixed(2) + '</td>'
    assert "'<td>$' + t.exit_price.toFixed(2)" not in text
    assert "t.exit_price != null ? '$' + t.exit_price.toFixed(2)" in text
