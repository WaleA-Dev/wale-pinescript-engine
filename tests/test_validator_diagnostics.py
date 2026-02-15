from datetime import datetime, timedelta

import pandas as pd

from src.backtest import ExitSignal, Trade
from src.validator import TradingViewValidator


def _closed_trade(
    trade_id: int,
    entry_time: datetime,
    exit_time: datetime,
    entry_price: float,
    exit_price: float,
    exit_signal: ExitSignal,
    pnl: float,
    entry_bar: int,
    exit_bar: int,
    direction: str = "long",
) -> Trade:
    t = Trade(
        trade_id=trade_id,
        entry_time=entry_time,
        entry_price=entry_price,
        entry_bar=entry_bar,
        direction=direction,
        qty=1,
        entry_value=entry_price,
    )
    t.exit_time = exit_time
    t.exit_price = exit_price
    t.exit_bar = exit_bar
    t.exit_signal = exit_signal
    t.pnl = pnl
    return t


def test_validator_emits_first_divergence_and_stop_target_mismatch():
    our = [
        _closed_trade(
            trade_id=1,
            entry_time=datetime(2025, 1, 2, 10, 0),
            exit_time=datetime(2025, 1, 2, 12, 0),
            entry_price=100.0,
            exit_price=95.0,
            exit_signal=ExitSignal.STOP_LOSS,
            pnl=-5.0,
            entry_bar=10,
            exit_bar=12,
            direction="long",
        )
    ]

    tv_df = pd.DataFrame(
        {
            "entry time": [datetime(2025, 1, 2, 10, 0)],
            "exit time": [datetime(2025, 1, 2, 12, 0)],
            "entry price": [100.0],
            "exit price": [105.0],
            "exit signal": ["PT"],
            "profit": [5.0],
            "type": ["Entry Long"],
        }
    )

    validator = TradingViewValidator(excel_df=tv_df, time_alignment="off")
    result = validator.validate(our)

    assert result.passed is False
    assert result.first_divergence is not None
    assert result.first_divergence["trade_number"] == 1
    assert result.first_divergence["our_entry_bar"] == 10
    assert "exit_signal_mismatch" in result.first_divergence["categories"]
    assert "stop_target_fill_mismatch" in result.first_divergence["categories"]
    assert result.mismatch_breakdown.get("stop_target_fill_mismatch", 0) == 1


def test_validator_pass_case_has_no_divergence():
    our = [
        _closed_trade(
            trade_id=1,
            entry_time=datetime(2025, 1, 2, 10, 0),
            exit_time=datetime(2025, 1, 2, 12, 0),
            entry_price=100.0,
            exit_price=105.0,
            exit_signal=ExitSignal.PROFIT_TARGET,
            pnl=5.0,
            entry_bar=3,
            exit_bar=4,
            direction="long",
        )
    ]

    tv_df = pd.DataFrame(
        {
            "entry time": [datetime(2025, 1, 2, 10, 0)],
            "exit time": [datetime(2025, 1, 2, 12, 0)],
            "entry price": [100.0],
            "exit price": [105.0],
            "exit signal": ["PT"],
            "profit": [5.0],
            "type": ["Entry Long"],
        }
    )

    validator = TradingViewValidator(excel_df=tv_df, time_alignment="off")
    result = validator.validate(our)

    assert result.passed is True
    assert result.first_divergence is None
    assert result.mismatch_breakdown == {}


def test_validator_flags_session_hour_blocker_for_parity_feasibility():
    our = [
        _closed_trade(
            trade_id=1,
            entry_time=datetime(2025, 1, 2, 14, 0),
            exit_time=datetime(2025, 1, 2, 15, 0),
            entry_price=100.0,
            exit_price=101.0,
            exit_signal=ExitSignal.PROFIT_TARGET,
            pnl=1.0,
            entry_bar=1,
            exit_bar=2,
            direction="long",
        )
    ]

    tv_rows = []
    for i in range(24):
        entry = datetime(2025, 1, 2, 1, 0) + timedelta(days=i)
        tv_rows.append(
            {
                "entry time": entry,
                "exit time": entry + timedelta(hours=1),
                "entry price": 100.0,
                "exit price": 101.0,
                "exit signal": "PT",
                "profit": 1.0,
                "type": "Entry Long",
            }
        )
    tv_df = pd.DataFrame(tv_rows)

    data_times = []
    for i in range(24):
        day = datetime(2025, 1, 2) + timedelta(days=i)
        for hour in range(13, 21):
            data_times.append(day.replace(hour=hour, minute=0))

    validator = TradingViewValidator(excel_df=tv_df, time_alignment="off")
    result = validator.validate(our, data_times=data_times)

    assert any("hours absent from CSV bars" in msg for msg in result.feasibility_blockers)
    assert result.session_diagnostics.get("tv_entries_outside_data_hours_pct") == 100.0
