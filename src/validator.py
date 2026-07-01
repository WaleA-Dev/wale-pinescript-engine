"""
TradingView Validator - Trade-by-Trade Comparison

This module validates backtest results against TradingView's Excel export.
It performs detailed comparison of:
1. Entry/exit times (exact match)
2. Entry/exit prices (within tolerance)
3. Exit signals (exact match)
4. P&L (within percentage tolerance)

Special handling for open trades at dataset end.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .backtest import Trade, BacktestResult


@dataclass
class ValidationResult:
    """Result of trade-by-trade validation."""
    passed: bool
    message: str
    total_trades_compared: int
    matched_trades: int
    mismatched_trades: int
    details: List[Dict[str, Any]]
    
    # Aggregate comparisons
    our_total_pnl: float = 0.0
    tv_total_pnl: float = 0.0
    pnl_difference: float = 0.0
    pnl_difference_pct: float = 0.0

    # Diagnostics
    # First mismatched trade with categorized causes (None when passed)
    first_divergence: Optional[Dict[str, Any]] = None
    # Counts of mismatch categories across all compared trades
    mismatch_breakdown: Dict[str, int] = field(default_factory=dict)
    # Structural reasons parity may be impossible (e.g. session coverage)
    feasibility_blockers: List[str] = field(default_factory=list)
    # Session/timezone coverage diagnostics
    session_diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeComparison:
    """Detailed comparison of a single trade."""
    trade_idx: int
    matched: bool
    differences: List[str]
    
    our_entry_time: datetime
    tv_entry_time: datetime
    our_exit_time: Optional[datetime]
    tv_exit_time: Optional[datetime]
    
    our_entry_price: float
    tv_entry_price: float
    our_exit_price: Optional[float]
    tv_exit_price: Optional[float]
    
    our_exit_signal: Optional[str]
    tv_exit_signal: Optional[str]
    
    our_pnl: float
    tv_pnl: float

    # Mismatch category tags (e.g. "exit_signal_mismatch")
    categories: List[str] = field(default_factory=list)


class TradingViewValidator:
    """
    Validates backtest results against TradingView Excel exports.
    
    Tolerances:
    - Entry/Exit Time: Exact bar match
    - Entry/Exit Price: 0.01 absolute difference
    - P&L: 2% relative difference (accounts for commission variations)
    """
    
    # Validation tolerances
    PRICE_TOLERANCE = 0.01
    PNL_TOLERANCE_PCT = 2.0
    TIME_TOLERANCE_SECONDS = 60  # Allow 1 minute variance for timezone issues
    
    def __init__(self, excel_path: str = None, excel_df: pd.DataFrame = None,
                 time_alignment: str = "auto"):
        """
        Initialize validator with TradingView export.
        
        Args:
            excel_path: Path to TradingView Excel export
            excel_df: Pre-loaded DataFrame
            time_alignment: "auto" detects and corrects a constant whole-hour
                offset between TV times and ours (timezone drift in exports);
                "off" compares timestamps as-is
        """
        if time_alignment not in ("auto", "off"):
            raise ValueError(f"time_alignment must be 'auto' or 'off', got {time_alignment!r}")
        self.time_alignment = time_alignment

        if excel_path:
            self.tv_trades = self._load_excel(excel_path)
        elif excel_df is not None:
            self.tv_trades = self._parse_dataframe(excel_df)
        else:
            self.tv_trades = []
    
    def _load_excel(self, path: str) -> List[Dict[str, Any]]:
        """Load and parse TradingView Excel export."""
        path = Path(path)
        
        if path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return self._parse_dataframe(df)
    
    def _parse_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse TradingView export DataFrame into trade list."""
        trades = []
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Common column name mappings
        col_mappings = {
            'entry_time': ['entry time', 'entry_time', 'entrytime', 'entry date', 'entry_date'],
            'exit_time': ['exit time', 'exit_time', 'exittime', 'exit date', 'exit_date'],
            'entry_price': ['entry price', 'entry_price', 'entryprice', 'entry'],
            'exit_price': ['exit price', 'exit_price', 'exitprice', 'exit'],
            'exit_signal': ['exit signal', 'exit_signal', 'exitsignal', 'signal', 'type'],
            'pnl': ['profit', 'pnl', 'p&l', 'profit/loss', 'net profit'],
            'pnl_pct': ['profit %', 'pnl %', 'profit_pct', 'return', 'return %'],
        }
        
        # Find actual column names
        actual_cols = {}
        for target, candidates in col_mappings.items():
            for candidate in candidates:
                if candidate in df.columns:
                    actual_cols[target] = candidate
                    break
        
        # Parse each row as a trade
        for idx, row in df.iterrows():
            trade = {}
            
            # Entry time
            if 'entry_time' in actual_cols:
                trade['entry_time'] = self._parse_datetime(row[actual_cols['entry_time']])
            
            # Exit time
            if 'exit_time' in actual_cols:
                trade['exit_time'] = self._parse_datetime(row[actual_cols['exit_time']])
            
            # Entry price
            if 'entry_price' in actual_cols:
                trade['entry_price'] = float(row[actual_cols['entry_price']])
            
            # Exit price
            if 'exit_price' in actual_cols:
                val = row[actual_cols['exit_price']]
                trade['exit_price'] = float(val) if pd.notna(val) else None
            
            # Exit signal
            if 'exit_signal' in actual_cols:
                val = row[actual_cols['exit_signal']]
                trade['exit_signal'] = str(val) if pd.notna(val) else None
            
            # P&L
            if 'pnl' in actual_cols:
                val = row[actual_cols['pnl']]
                trade['pnl'] = float(val) if pd.notna(val) else 0.0
            
            # P&L %
            if 'pnl_pct' in actual_cols:
                val = row[actual_cols['pnl_pct']]
                trade['pnl_pct'] = float(val) if pd.notna(val) else 0.0
            
            trades.append(trade)
        
        return trades
    
    def _parse_datetime(self, value) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if pd.isna(value):
            return None
        
        if isinstance(value, datetime):
            return value
        
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        
        if isinstance(value, str):
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y-%m-%d',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y %H:%M',
                '%m/%d/%Y',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y %H:%M',
                '%d/%m/%Y',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        
        return None
    
    def validate(self, our_trades: List[Trade], last_csv_time: datetime = None,
                 data_times: Optional[List[datetime]] = None) -> ValidationResult:
        """
        Validate our trades against TradingView export.
        
        Args:
            our_trades: List of Trade objects from our backtest
            last_csv_time: Last timestamp in the OHLC data (for open trade handling)
            data_times: Bar timestamps of the OHLC data used for the backtest;
                enables session-coverage feasibility diagnostics
            
        Returns:
            ValidationResult with detailed comparison
        """
        comparisons = []
        matched = 0
        mismatched = 0
        
        our_total_pnl = sum(t.pnl for t in our_trades if not t.is_open())
        tv_total_pnl = sum(t.get('pnl', 0) for t in self.tv_trades)

        # Session-coverage diagnostics are computed up front so they are
        # available even when validation bails out early (count mismatch).
        session_diagnostics, feasibility_blockers = self._session_feasibility(data_times)

        # Optional whole-hour timezone correction between TV export and us
        time_offset = self._detect_time_offset(our_trades)
        if time_offset:
            session_diagnostics["applied_time_offset_hours"] = time_offset.total_seconds() / 3600.0

        # Check trade count
        if len(our_trades) != len(self.tv_trades):
            # Allow for one extra trade if it's open
            if abs(len(our_trades) - len(self.tv_trades)) > 1:
                return ValidationResult(
                    passed=False,
                    message=f"Trade count mismatch: Ours={len(our_trades)}, TV={len(self.tv_trades)}",
                    total_trades_compared=0,
                    matched_trades=0,
                    mismatched_trades=0,
                    details=[],
                    our_total_pnl=our_total_pnl,
                    tv_total_pnl=tv_total_pnl,
                    pnl_difference=our_total_pnl - tv_total_pnl,
                    pnl_difference_pct=abs(our_total_pnl - tv_total_pnl) / abs(tv_total_pnl) * 100 if tv_total_pnl != 0 else 0,
                    mismatch_breakdown={"trade_count_mismatch": abs(len(our_trades) - len(self.tv_trades))},
                    feasibility_blockers=feasibility_blockers,
                    session_diagnostics=session_diagnostics,
                )
        
        # Compare trade by trade
        min_trades = min(len(our_trades), len(self.tv_trades))
        
        for i in range(min_trades):
            our_trade = our_trades[i]
            tv_trade = self.tv_trades[i]
            
            comparison = self._compare_trade(i, our_trade, tv_trade, last_csv_time, time_offset)
            comparisons.append(comparison)
            
            if comparison.matched:
                matched += 1
            else:
                mismatched += 1
        
        # Determine pass/fail
        passed = mismatched == 0
        
        first_mismatch = next((c for c in comparisons if not c.matched), None)
        if passed:
            message = f"All {matched} trades matched successfully."
        elif first_mismatch:
            message = f"Trade {first_mismatch.trade_idx + 1} mismatch: {', '.join(first_mismatch.differences)}"
        else:
            message = f"{mismatched} trades did not match."

        # Aggregate mismatch categories
        mismatch_breakdown: Dict[str, int] = {}
        for c in comparisons:
            for cat in c.categories:
                mismatch_breakdown[cat] = mismatch_breakdown.get(cat, 0) + 1

        # First divergence: where and why we first diverge from TV
        first_divergence = None
        if first_mismatch is not None:
            our_trade = our_trades[first_mismatch.trade_idx]
            first_divergence = {
                "trade_number": first_mismatch.trade_idx + 1,
                "our_entry_bar": our_trade.entry_bar,
                "our_exit_bar": our_trade.exit_bar,
                "our_entry_time": str(first_mismatch.our_entry_time) if first_mismatch.our_entry_time else None,
                "tv_entry_time": str(first_mismatch.tv_entry_time) if first_mismatch.tv_entry_time else None,
                "categories": list(first_mismatch.categories),
                "differences": list(first_mismatch.differences),
            }

        return ValidationResult(
            passed=passed,
            message=message,
            total_trades_compared=min_trades,
            matched_trades=matched,
            mismatched_trades=mismatched,
            details=[self._comparison_to_dict(c) for c in comparisons],
            our_total_pnl=our_total_pnl,
            tv_total_pnl=tv_total_pnl,
            pnl_difference=our_total_pnl - tv_total_pnl,
            pnl_difference_pct=abs(our_total_pnl - tv_total_pnl) / abs(tv_total_pnl) * 100 if tv_total_pnl != 0 else 0,
            first_divergence=first_divergence,
            mismatch_breakdown=mismatch_breakdown,
            feasibility_blockers=feasibility_blockers,
            session_diagnostics=session_diagnostics,
        )

    def _session_feasibility(self, data_times: Optional[List[datetime]]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Check whether TV trades are even representable with the given bars.

        If the TV export enters trades at hours that never appear in the CSV
        data (typically a session/timezone mismatch between the TV chart and
        the data feed), bar-parity validation cannot succeed no matter what
        the strategy logic does.
        """
        diagnostics: Dict[str, Any] = {}
        blockers: List[str] = []

        if not data_times or not self.tv_trades:
            return diagnostics, blockers

        data_hours = sorted({t.hour for t in data_times if t is not None})
        tv_entry_hours = [
            t['entry_time'].hour for t in self.tv_trades
            if t.get('entry_time') is not None
        ]
        if not tv_entry_hours:
            return diagnostics, blockers

        outside = [h for h in tv_entry_hours if h not in set(data_hours)]
        pct = round(100.0 * len(outside) / len(tv_entry_hours), 1)

        diagnostics["data_hours"] = data_hours
        diagnostics["tv_entry_hours"] = sorted(set(tv_entry_hours))
        diagnostics["tv_entries_outside_data_hours_pct"] = pct

        if pct > 0:
            missing = sorted(set(outside))
            blockers.append(
                f"{pct:.1f}% of TV entries occur at hours absent from CSV bars "
                f"(missing hours: {missing}, data hours: {data_hours}). "
                f"Check session/timezone alignment between the TV chart and the data feed."
            )

        return diagnostics, blockers

    def _detect_time_offset(self, our_trades: List[Trade]) -> Optional[timedelta]:
        """
        Detect a constant whole-hour offset between TV and our timestamps.

        Returns the timedelta to subtract from TV times, or None.
        """
        if self.time_alignment == "off":
            return None

        diffs = []
        for our_trade, tv_trade in zip(our_trades, self.tv_trades):
            tv_time = tv_trade.get('entry_time')
            if our_trade.entry_time and tv_time:
                diffs.append((tv_time - our_trade.entry_time).total_seconds())
        if not diffs:
            return None

        median = sorted(diffs)[len(diffs) // 2]
        hours = round(median / 3600)
        if hours != 0 and abs(median - hours * 3600) <= self.TIME_TOLERANCE_SECONDS:
            return timedelta(hours=hours)
        return None
    
    def _compare_trade(self, idx: int, our_trade: Trade, tv_trade: Dict[str, Any], 
                       last_csv_time: datetime = None,
                       time_offset: Optional[timedelta] = None) -> TradeComparison:
        """Compare a single trade."""
        differences = []
        categories: List[str] = []
        
        our_entry_time = our_trade.entry_time
        tv_entry_time = tv_trade.get('entry_time')
        
        our_exit_time = our_trade.exit_time
        tv_exit_time = tv_trade.get('exit_time')

        # Apply detected timezone offset to TV times before comparing
        if time_offset:
            if tv_entry_time:
                tv_entry_time = tv_entry_time - time_offset
            if tv_exit_time:
                tv_exit_time = tv_exit_time - time_offset
        
        our_entry_price = our_trade.entry_price
        tv_entry_price = tv_trade.get('entry_price', 0)
        
        our_exit_price = our_trade.exit_price
        tv_exit_price = tv_trade.get('exit_price')
        
        our_exit_signal = our_trade.exit_signal.value if our_trade.exit_signal else None
        tv_exit_signal = tv_trade.get('exit_signal')
        
        our_pnl = our_trade.pnl
        tv_pnl = tv_trade.get('pnl', 0)
        
        # Compare entry time
        if our_entry_time and tv_entry_time:
            time_diff = abs((our_entry_time - tv_entry_time).total_seconds())
            if time_diff > self.TIME_TOLERANCE_SECONDS:
                differences.append(f"entry_time ({our_entry_time} vs {tv_entry_time})")
                categories.append("entry_time_mismatch")
        
        # Compare exit time
        if our_exit_time and tv_exit_time:
            time_diff = abs((our_exit_time - tv_exit_time).total_seconds())
            if time_diff > self.TIME_TOLERANCE_SECONDS:
                differences.append(f"exit_time ({our_exit_time} vs {tv_exit_time})")
                categories.append("exit_time_mismatch")
        
        # Compare entry price
        if abs(our_entry_price - tv_entry_price) > self.PRICE_TOLERANCE:
            differences.append(f"entry_price ({our_entry_price:.2f} vs {tv_entry_price:.2f})")
            categories.append("entry_price_mismatch")
        
        # Compare exit price
        if our_exit_price is not None and tv_exit_price is not None:
            if abs(our_exit_price - tv_exit_price) > self.PRICE_TOLERANCE:
                differences.append(f"exit_price ({our_exit_price:.2f} vs {tv_exit_price:.2f})")
                categories.append("exit_price_mismatch")
        
        # Compare exit signal (normalize for comparison)
        if our_exit_signal and tv_exit_signal:
            our_signal_norm = self._normalize_signal(our_exit_signal)
            tv_signal_norm = self._normalize_signal(tv_exit_signal)
            if our_signal_norm != tv_signal_norm:
                differences.append(f"exit_signal ({our_exit_signal} vs {tv_exit_signal})")
                categories.append("exit_signal_mismatch")

                # One side exited via a stop, the other via a target: the
                # classic same-bar stop/target fill-order ambiguity
                stop_kinds = {"stop_loss", "trailing"}
                target_kinds = {"profit_target"}
                pair = {our_signal_norm, tv_signal_norm}
                if pair & stop_kinds and pair & target_kinds:
                    categories.append("stop_target_fill_mismatch")
        
        # Compare P&L
        if tv_pnl != 0:
            pnl_diff_pct = abs(our_pnl - tv_pnl) / abs(tv_pnl) * 100
            if pnl_diff_pct > self.PNL_TOLERANCE_PCT:
                differences.append(f"pnl ({our_pnl:.2f} vs {tv_pnl:.2f}, {pnl_diff_pct:.1f}% diff)")
                categories.append("pnl_mismatch")
        elif our_pnl != 0 and abs(our_pnl) > 1.0:
            differences.append(f"pnl ({our_pnl:.2f} vs {tv_pnl:.2f})")
            categories.append("pnl_mismatch")
        
        # Special handling for open trades
        is_last_trade = (idx == len(self.tv_trades) - 1)
        is_open_trade = (tv_exit_signal and 'open' in tv_exit_signal.lower()) or our_trade.is_open()
        
        if is_last_trade and is_open_trade:
            if last_csv_time and tv_exit_time and tv_exit_time > last_csv_time:
                # Trade is still open because dataset ended - ignore differences
                differences = []
                categories = []
        
        return TradeComparison(
            trade_idx=idx,
            matched=len(differences) == 0,
            differences=differences,
            our_entry_time=our_entry_time,
            tv_entry_time=tv_entry_time,
            our_exit_time=our_exit_time,
            tv_exit_time=tv_exit_time,
            our_entry_price=our_entry_price,
            tv_entry_price=tv_entry_price,
            our_exit_price=our_exit_price,
            tv_exit_price=tv_exit_price,
            our_exit_signal=our_exit_signal,
            tv_exit_signal=tv_exit_signal,
            our_pnl=our_pnl,
            tv_pnl=tv_pnl,
            categories=categories,
        )
    
    def _normalize_signal(self, signal: str) -> str:
        """Normalize exit signal names for comparison."""
        signal = signal.lower().strip()
        
        # Map common variations
        mappings = {
            'sl': 'stop_loss',
            'stop loss': 'stop_loss',
            'stoploss': 'stop_loss',
            'stop': 'stop_loss',
            'trail': 'trailing',
            'trailing stop': 'trailing',
            'trailingstop': 'trailing',
            'pt': 'profit_target',
            'profit target': 'profit_target',
            'profittarget': 'profit_target',
            'take profit': 'profit_target',
            'takeprofit': 'profit_target',
            'tp': 'profit_target',
            'ob': 'overbought',
            'overbought': 'overbought',
            'os': 'oversold',
            'oversold': 'oversold',
            'signal': 'signal',
            'open': 'open',
        }
        
        return mappings.get(signal, signal)
    
    def _comparison_to_dict(self, comparison: TradeComparison) -> Dict[str, Any]:
        """Convert TradeComparison to dictionary."""
        return {
            'trade_idx': comparison.trade_idx,
            'matched': comparison.matched,
            'differences': comparison.differences,
            'categories': comparison.categories,
            'our_entry_time': str(comparison.our_entry_time) if comparison.our_entry_time else None,
            'tv_entry_time': str(comparison.tv_entry_time) if comparison.tv_entry_time else None,
            'our_exit_time': str(comparison.our_exit_time) if comparison.our_exit_time else None,
            'tv_exit_time': str(comparison.tv_exit_time) if comparison.tv_exit_time else None,
            'our_entry_price': comparison.our_entry_price,
            'tv_entry_price': comparison.tv_entry_price,
            'our_exit_price': comparison.our_exit_price,
            'tv_exit_price': comparison.tv_exit_price,
            'our_exit_signal': comparison.our_exit_signal,
            'tv_exit_signal': comparison.tv_exit_signal,
            'our_pnl': comparison.our_pnl,
            'tv_pnl': comparison.tv_pnl,
        }
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TRADINGVIEW VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        lines.append(f"Status: {status}")
        lines.append(f"Message: {result.message}")
        lines.append("")
        
        lines.append(f"Trades Compared: {result.total_trades_compared}")
        lines.append(f"Matched: {result.matched_trades}")
        lines.append(f"Mismatched: {result.mismatched_trades}")
        lines.append("")
        
        lines.append(f"Our Total P&L: ${result.our_total_pnl:,.2f}")
        lines.append(f"TV Total P&L: ${result.tv_total_pnl:,.2f}")
        lines.append(f"Difference: ${result.pnl_difference:,.2f} ({result.pnl_difference_pct:.2f}%)")
        lines.append("")
        
        if result.mismatched_trades > 0:
            lines.append("-" * 60)
            lines.append("MISMATCHED TRADES:")
            lines.append("-" * 60)
            
            for detail in result.details:
                if not detail['matched']:
                    lines.append(f"\nTrade #{detail['trade_idx'] + 1}:")
                    for diff in detail['differences']:
                        lines.append(f"  - {diff}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def validate_against_tradingview(our_trades: List[Trade], excel_path: str,
                                  last_csv_time: datetime = None) -> ValidationResult:
    """
    Convenience function to validate trades against TradingView export.
    
    Args:
        our_trades: List of Trade objects from backtest
        excel_path: Path to TradingView Excel export
        last_csv_time: Last timestamp in OHLC data
        
    Returns:
        ValidationResult
    """
    validator = TradingViewValidator(excel_path=excel_path)
    return validator.validate(our_trades, last_csv_time)
