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
from datetime import datetime, timedelta, timezone
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
    mismatch_breakdown: Dict[str, int] = field(default_factory=dict)
    first_divergence: Optional[Dict[str, Any]] = None
    time_alignment_mode: str = "dynamic"
    inferred_time_offset_minutes: int = 0
    feasibility_warnings: List[str] = field(default_factory=list)
    feasibility_blockers: List[str] = field(default_factory=list)
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
    our_direction: Optional[str]
    tv_direction: Optional[str]
    
    our_pnl: float
    tv_pnl: float
    our_entry_bar: Optional[int] = None
    our_exit_bar: Optional[int] = None


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
    
    def __init__(
        self,
        excel_path: str = None,
        excel_df: pd.DataFrame = None,
        time_alignment: str = "dynamic",
    ):
        """
        Initialize validator with TradingView export.
        
        Args:
            excel_path: Path to TradingView Excel export
            excel_df: Pre-loaded DataFrame
        """
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
            excel_file = pd.ExcelFile(path)
            sheet_names_lower = [s.lower().strip() for s in excel_file.sheet_names]
            if 'list of trades' in sheet_names_lower:
                idx = sheet_names_lower.index('list of trades')
                df = excel_file.parse(excel_file.sheet_names[idx])
            else:
                # Fallback to first sheet for non-standard exports.
                df = excel_file.parse(excel_file.sheet_names[0])
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return self._parse_dataframe(df)
    
    def _parse_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse TradingView export DataFrame into trade list."""
        # TradingView XLSX "List of trades" sheet format:
        # rows are events (Entry long / Exit long) with a shared "Trade #".
        normalized_cols = [str(c).lower().strip() for c in df.columns]
        if 'trade #' in normalized_cols and 'type' in normalized_cols and 'date and time' in normalized_cols:
            return self._parse_tv_list_of_trades(df)

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
            if 'type' in df.columns:
                trade['direction'] = self._infer_direction_from_text(row.get('type'))
            
            trades.append(trade)
        
        return trades

    def _parse_tv_list_of_trades(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Parse TradingView XLSX 'List of trades' event rows into trade records."""
        parsed = df.copy()
        parsed.columns = parsed.columns.str.lower().str.strip()

        # Coerce key columns if present
        if 'date and time' in parsed.columns:
            parsed['date and time'] = pd.to_datetime(parsed['date and time'], errors='coerce')
        if 'price usd' in parsed.columns:
            parsed['price usd'] = pd.to_numeric(parsed['price usd'], errors='coerce')
        if 'net p&l usd' in parsed.columns:
            parsed['net p&l usd'] = pd.to_numeric(parsed['net p&l usd'], errors='coerce')
        if 'net p&l %' in parsed.columns:
            parsed['net p&l %'] = pd.to_numeric(parsed['net p&l %'], errors='coerce')

        trades: List[Dict[str, Any]] = []

        # Keep stable order by trade number and timestamp.
        if 'trade #' in parsed.columns:
            parsed = parsed.sort_values(['trade #', 'date and time'], kind='stable')

        for trade_id, g in parsed.groupby('trade #', sort=True):
            if pd.isna(trade_id):
                continue

            g = g.copy()
            g_type = g['type'].astype(str).str.lower().str.strip()

            entry_rows = g[g_type.str.contains('entry', na=False)]
            exit_rows = g[g_type.str.contains('exit', na=False)]

            if entry_rows.empty:
                continue

            entry_row = entry_rows.iloc[0]
            exit_row = exit_rows.iloc[0] if not exit_rows.empty else None

            trade: Dict[str, Any] = {
                'entry_time': self._parse_datetime(entry_row.get('date and time')),
                'entry_price': float(entry_row.get('price usd')) if pd.notna(entry_row.get('price usd')) else 0.0,
                'exit_time': None,
                'exit_price': None,
                'exit_signal': 'Open',
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'direction': self._infer_direction_from_text(entry_row.get('type')),
            }

            if exit_row is not None:
                trade['exit_time'] = self._parse_datetime(exit_row.get('date and time'))
                trade['exit_price'] = float(exit_row.get('price usd')) if pd.notna(exit_row.get('price usd')) else None
                trade['exit_signal'] = str(exit_row.get('signal')) if pd.notna(exit_row.get('signal')) else None
                if pd.notna(exit_row.get('net p&l usd')):
                    trade['pnl'] = float(exit_row.get('net p&l usd'))
                if pd.notna(exit_row.get('net p&l %')):
                    trade['pnl_pct'] = float(exit_row.get('net p&l %'))
            else:
                # Open trade: pull latest PnL if provided.
                if pd.notna(entry_row.get('net p&l usd')):
                    trade['pnl'] = float(entry_row.get('net p&l usd'))
                if pd.notna(entry_row.get('net p&l %')):
                    trade['pnl_pct'] = float(entry_row.get('net p&l %'))

            trades.append(trade)

        # Trade # is ascending already; keep explicit sort by entry time as fallback.
        trades.sort(key=lambda t: t.get('entry_time') or datetime.min)
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

    def _infer_direction_from_text(self, value: Any) -> Optional[str]:
        """Infer long/short direction labels from TradingView text fields."""
        if value is None or pd.isna(value):
            return None
        txt = str(value).lower()
        if "short" in txt:
            return "short"
        if "long" in txt:
            return "long"
        return None
    
    def validate(
        self,
        our_trades: List[Trade],
        last_csv_time: datetime = None,
        data_times: Optional[List[datetime]] = None,
    ) -> ValidationResult:
        """
        Validate our trades against TradingView export.
        
        Args:
            our_trades: List of Trade objects from our backtest
            last_csv_time: Last timestamp in the OHLC data (for open trade handling)
            data_times: Optional bar timestamps from the OHLC dataset for
                feasibility/session diagnostics.
            
        Returns:
            ValidationResult with detailed comparison
        """
        comparisons = []
        matched = 0
        mismatched = 0

        tv_trades = list(self.tv_trades)
        tv_trades_raw = list(tv_trades)
        inferred_offset = timedelta(0)

        # If our dataset is a shorter window than TV export, compare on overlap window.
        if our_trades:
            our_entry_times = [t.entry_time for t in our_trades if t.entry_time is not None]
            if our_entry_times:
                our_start = min(our_entry_times)
                tv_trades = [
                    t for t in tv_trades
                    if (t.get('entry_time') is None) or (t.get('entry_time') >= our_start)
                ]

        if last_csv_time is not None:
            tv_trades = [
                t for t in tv_trades
                if (t.get('entry_time') is None) or (t.get('entry_time') <= last_csv_time)
            ]

        # Align TV timestamps to engine timestamps.
        if self.time_alignment == "fixed":
            inferred_offset = self._infer_time_offset(our_trades, tv_trades)
            if inferred_offset != timedelta(0):
                tv_trades = [self._shift_trade_times(t, inferred_offset) for t in tv_trades]
        elif self.time_alignment == "dynamic":
            global_offset = self._infer_time_offset(our_trades, tv_trades)
            inferred_offset = global_offset
            month_offsets = self._infer_monthly_offsets(our_trades, tv_trades, global_offset)
            tv_trades = [
                self._shift_trade_times_dynamic(t, month_offsets, global_offset)
                for t in tv_trades
            ]

        # Include open-trade marked PnL so aggregate aligns with TV export semantics.
        our_total_pnl = sum((t.pnl or 0.0) for t in our_trades)
        tv_total_pnl = sum(t.get('pnl', 0) for t in tv_trades)

        # Compare trade by trade
        min_trades = min(len(our_trades), len(tv_trades))
        
        for i in range(min_trades):
            our_trade = our_trades[i]
            tv_trade = tv_trades[i]
            
            comparison = self._compare_trade(
                i, our_trade, tv_trade, last_csv_time, is_last_trade=(i == min_trades - 1)
            )
            comparisons.append(comparison)
            
            if comparison.matched:
                matched += 1
            else:
                mismatched += 1
        
        # Determine pass/fail
        count_gap = len(our_trades) - len(tv_trades)
        passed = mismatched == 0 and abs(count_gap) <= 1
        
        if passed:
            message = f"All {matched} trades matched successfully."
        else:
            if abs(count_gap) > 1:
                message = (
                    f"Trade count mismatch after overlap filter: "
                    f"Ours={len(our_trades)}, TV={len(tv_trades)}"
                )
            else:
                first_mismatch = next((c for c in comparisons if not c.matched), None)
                if first_mismatch:
                    message = f"Trade {first_mismatch.trade_idx + 1} mismatch: {', '.join(first_mismatch.differences)}"
                else:
                    message = f"{mismatched} trades did not match."

        mismatch_breakdown = self._build_mismatch_breakdown(comparisons)
        first_divergence = self._first_divergence(comparisons)
        feasibility_warnings, feasibility_blockers, session_diagnostics = self._assess_parity_feasibility(
            our_trades=our_trades,
            tv_trades=tv_trades,
            raw_tv_trades=tv_trades_raw,
            data_times=data_times,
            inferred_offset=inferred_offset,
        )
        
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
            mismatch_breakdown=mismatch_breakdown,
            first_divergence=first_divergence,
            time_alignment_mode=self.time_alignment,
            inferred_time_offset_minutes=int(inferred_offset.total_seconds() / 60),
            feasibility_warnings=feasibility_warnings,
            feasibility_blockers=feasibility_blockers,
            session_diagnostics=session_diagnostics,
        )

    def _to_naive_utc_datetime(self, value: Any) -> Optional[datetime]:
        """Parse timestamp-like input into a timezone-naive UTC datetime."""
        if value is None:
            return None
        try:
            ts = pd.to_datetime(value, errors='coerce')
        except Exception:
            return None
        if pd.isna(ts):
            return None
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                ts = ts.tz_convert('UTC').tz_localize(None)
            return ts.to_pydatetime()
        if isinstance(ts, datetime):
            if ts.tzinfo is not None:
                return ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts
        return None

    def _normalize_datetime_list(self, values: Any) -> List[datetime]:
        """Coerce a timestamp iterable into clean datetime objects."""
        if values is None:
            return []
        out: List[datetime] = []
        for raw in values:
            dt = self._to_naive_utc_datetime(raw)
            if dt is not None:
                out.append(dt)
        return out

    def _assess_parity_feasibility(
        self,
        our_trades: List[Trade],
        tv_trades: List[Dict[str, Any]],
        raw_tv_trades: Optional[List[Dict[str, Any]]],
        data_times: Optional[List[datetime]],
        inferred_offset: timedelta,
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        Determine whether strict trade-by-trade parity is feasible with the current
        data/session context, and provide actionable diagnostics.
        """
        warnings: List[str] = []
        blockers: List[str] = []

        tv_entry_times = self._normalize_datetime_list(
            [t.get('entry_time') for t in tv_trades if t.get('entry_time') is not None]
        )
        raw_tv_entry_times = self._normalize_datetime_list(
            [t.get('entry_time') for t in (raw_tv_trades or tv_trades) if t.get('entry_time') is not None]
        )
        data_ts = self._normalize_datetime_list(data_times) if data_times is not None else []

        diagnostics: Dict[str, Any] = {
            "our_trade_count": len(our_trades),
            "tv_trade_count": len(tv_trades),
            "tv_entry_count": len(tv_entry_times),
            "tv_entry_count_raw": len(raw_tv_entry_times),
            "data_bar_count": len(data_ts),
            "inferred_time_offset_minutes": int(inferred_offset.total_seconds() / 60),
        }

        if len(tv_trades) == 0:
            blockers.append("TradingView export produced zero trades after parsing/filtering.")
            return warnings, blockers, diagnostics

        if not tv_entry_times:
            blockers.append("TradingView export has no parseable entry timestamps.")
            return warnings, blockers, diagnostics

        offset_minutes = diagnostics["inferred_time_offset_minutes"]
        if abs(offset_minutes) >= 180:
            warnings.append(
                f"Large inferred time offset ({offset_minutes} minutes). "
                "Check chart timezone/session settings."
            )

        trade_count_gap = len(our_trades) - len(tv_trades)
        diagnostics["trade_count_gap"] = trade_count_gap
        if len(tv_trades) > 0:
            gap_pct = abs(trade_count_gap) / len(tv_trades) * 100.0
            diagnostics["trade_count_gap_pct"] = round(gap_pct, 2)
            if gap_pct >= 40.0:
                blockers.append(
                    f"Very large trade-count gap ({len(our_trades)} vs {len(tv_trades)}). "
                    "Current data/logic setup is unlikely to achieve strict parity."
                )
            elif gap_pct >= 20.0:
                warnings.append(
                    f"Large trade-count gap ({len(our_trades)} vs {len(tv_trades)}). "
                    "Parity may be blocked by data/session mismatch."
                )

        if not data_ts:
            warnings.append(
                "CSV bar timestamps were not supplied to validator; "
                "session-range feasibility checks are limited."
            )
            return warnings, blockers, diagnostics

        data_start = min(data_ts)
        data_end = max(data_ts)
        tv_start = min(tv_entry_times)
        tv_end = max(tv_entry_times)
        diagnostics["data_start"] = data_start.isoformat(sep=' ')
        diagnostics["data_end"] = data_end.isoformat(sep=' ')
        diagnostics["tv_start"] = tv_start.isoformat(sep=' ')
        diagnostics["tv_end"] = tv_end.isoformat(sep=' ')

        has_overlap = max(data_start, tv_start) <= min(data_end, tv_end)
        diagnostics["has_time_overlap"] = has_overlap
        if not has_overlap:
            blockers.append(
                "No overlap between CSV time range and TradingView trade timestamps."
            )

        in_data_range = sum(1 for t in tv_entry_times if data_start <= t <= data_end)
        in_data_range_pct = in_data_range / len(tv_entry_times) * 100.0
        diagnostics["tv_entries_within_data_range_pct"] = round(in_data_range_pct, 2)
        if in_data_range_pct < 60.0:
            blockers.append(
                f"Only {in_data_range_pct:.1f}% of TradingView entries fall inside CSV date range."
            )
        elif in_data_range_pct < 95.0:
            warnings.append(
                f"Only {in_data_range_pct:.1f}% of TradingView entries fall inside CSV date range."
            )

        data_hours = sorted({t.hour for t in data_ts})
        tv_hours = sorted({t.hour for t in tv_entry_times})
        missing_hours = sorted(set(tv_hours) - set(data_hours))
        outside_hour_count = sum(1 for t in tv_entry_times if t.hour in missing_hours)
        outside_hour_pct = outside_hour_count / len(tv_entry_times) * 100.0
        diagnostics["data_hours_utc"] = data_hours
        diagnostics["tv_entry_hours_utc"] = tv_hours
        diagnostics["tv_hours_missing_from_data"] = missing_hours
        diagnostics["tv_entries_outside_data_hours_pct"] = round(outside_hour_pct, 2)

        if outside_hour_pct >= 25.0:
            blockers.append(
                f"{outside_hour_pct:.1f}% of TradingView entries occur in hours absent from CSV bars."
            )
        elif outside_hour_pct > 0:
            warnings.append(
                f"{outside_hour_pct:.1f}% of TradingView entries occur in hours absent from CSV bars."
            )

        # Also record diagnostics for raw (pre-alignment) TV timestamps.
        raw_tv_hours = sorted({t.hour for t in raw_tv_entry_times}) if raw_tv_entry_times else []
        raw_missing_hours = sorted(set(raw_tv_hours) - set(data_hours))
        raw_outside_hour_count = (
            sum(1 for t in raw_tv_entry_times if t.hour in raw_missing_hours)
            if raw_tv_entry_times
            else 0
        )
        raw_outside_hour_pct = (
            raw_outside_hour_count / len(raw_tv_entry_times) * 100.0
            if raw_tv_entry_times
            else 0.0
        )
        diagnostics["raw_tv_entry_hours_utc"] = raw_tv_hours
        diagnostics["raw_tv_hours_missing_from_data"] = raw_missing_hours
        diagnostics["raw_tv_entries_outside_data_hours_pct"] = round(raw_outside_hour_pct, 2)

        if (
            abs(offset_minutes) >= 180
            and raw_outside_hour_pct >= 25.0
            and outside_hour_pct <= 5.0
        ):
            warnings.append(
                "Raw TradingView timestamps differ materially from CSV session hours, "
                "and only align after large timezone offset adjustment."
            )

        data_minutes = sorted({t.minute for t in data_ts})
        tv_minutes = sorted({t.minute for t in tv_entry_times})
        missing_minutes = sorted(set(tv_minutes) - set(data_minutes))
        outside_min_count = sum(1 for t in tv_entry_times if t.minute in missing_minutes)
        outside_min_pct = outside_min_count / len(tv_entry_times) * 100.0
        diagnostics["data_bar_minutes"] = data_minutes
        diagnostics["tv_entry_minutes"] = tv_minutes
        diagnostics["tv_minutes_missing_from_data"] = missing_minutes
        diagnostics["tv_entries_outside_data_minutes_pct"] = round(outside_min_pct, 2)

        if outside_min_pct >= 50.0 and len(tv_entry_times) >= 20:
            blockers.append(
                f"{outside_min_pct:.1f}% of TradingView entries use minute buckets absent from CSV bars."
            )
        elif outside_min_pct >= 15.0:
            warnings.append(
                f"{outside_min_pct:.1f}% of TradingView entries use minute buckets absent from CSV bars."
            )

        return warnings, blockers, diagnostics

    def _first_divergence(self, comparisons: List[TradeComparison]) -> Optional[Dict[str, Any]]:
        """Return the first mismatched trade and category metadata."""
        first = next((c for c in comparisons if not c.matched), None)
        if first is None:
            return None
        categories = self._categorize_mismatch(first)
        return {
            'trade_number': first.trade_idx + 1,
            'our_entry_bar': first.our_entry_bar,
            'our_exit_bar': first.our_exit_bar,
            'our_entry_time': str(first.our_entry_time) if first.our_entry_time else None,
            'tv_entry_time': str(first.tv_entry_time) if first.tv_entry_time else None,
            'our_exit_time': str(first.our_exit_time) if first.our_exit_time else None,
            'tv_exit_time': str(first.tv_exit_time) if first.tv_exit_time else None,
            'categories': categories,
            'differences': list(first.differences),
        }

    def _categorize_mismatch(self, comparison: TradeComparison) -> List[str]:
        """Classify mismatch details into coarse categories."""
        cats: List[str] = []
        for diff in comparison.differences:
            d = diff.lower()
            if d.startswith('entry_time'):
                cats.append('entry_time_mismatch')
            elif d.startswith('exit_time'):
                cats.append('exit_time_mismatch')
            elif d.startswith('entry_price'):
                cats.append('entry_price_mismatch')
            elif d.startswith('exit_price'):
                cats.append('exit_price_mismatch')
            elif d.startswith('direction'):
                cats.append('direction_mismatch')
            elif d.startswith('exit_signal'):
                cats.append('exit_signal_mismatch')
                if self._is_stop_target_conflict(comparison.our_exit_signal, comparison.tv_exit_signal):
                    cats.append('stop_target_fill_mismatch')
            elif d.startswith('pnl'):
                cats.append('pnl_mismatch')
            else:
                cats.append('other_mismatch')
        return list(dict.fromkeys(cats))

    def _is_stop_target_conflict(self, our_signal: Optional[str], tv_signal: Optional[str]) -> bool:
        """Detect stop-vs-target exit disagreements."""
        if not our_signal or not tv_signal:
            return False
        o = our_signal.lower()
        t = tv_signal.lower()

        our_stop = ('sl' in o) or ('stop' in o)
        tv_stop = ('sl' in t) or ('stop' in t)
        our_target = ('pt' in o) or ('profit' in o) or ('take' in o)
        tv_target = ('pt' in t) or ('profit' in t) or ('take' in t)
        return (our_stop and tv_target) or (our_target and tv_stop)

    def _build_mismatch_breakdown(self, comparisons: List[TradeComparison]) -> Dict[str, int]:
        """Count mismatch categories across all non-matching trades."""
        out: Dict[str, int] = {}
        for cmp in comparisons:
            if cmp.matched:
                continue
            cats = self._categorize_mismatch(cmp)
            if not cats:
                cats = ['other_mismatch']
            for cat in cats:
                out[cat] = out.get(cat, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def _shift_trade_times(self, trade: Dict[str, Any], offset: timedelta) -> Dict[str, Any]:
        """Return a shallow copy of trade with shifted entry/exit timestamps."""
        out = dict(trade)
        et = out.get('entry_time')
        xt = out.get('exit_time')
        if et is not None:
            out['entry_time'] = et + offset
        if xt is not None:
            out['exit_time'] = xt + offset
        return out

    def _shift_trade_times_dynamic(
        self,
        trade: Dict[str, Any],
        month_offsets: Dict[Tuple[int, int], timedelta],
        default_offset: timedelta,
    ) -> Dict[str, Any]:
        """Shift trade times using month-specific offsets with global fallback."""
        out = dict(trade)
        et = out.get('entry_time')
        xt = out.get('exit_time')

        month_key = None
        if et is not None:
            month_key = (et.year, et.month)
        elif xt is not None:
            month_key = (xt.year, xt.month)

        offset = month_offsets.get(month_key, default_offset) if month_key is not None else default_offset

        if et is not None:
            out['entry_time'] = et + offset
        if xt is not None:
            out['exit_time'] = xt + offset
        return out

    def _score_offset(
        self,
        our_times: List[datetime],
        tv_times: List[datetime],
        offset: timedelta,
        tolerance: timedelta,
    ) -> int:
        """Count entry-time matches for a candidate offset."""
        shifted_tv = [t + offset for t in tv_times]
        score = 0
        for ot in our_times:
            if any(abs(ot - tt) <= tolerance for tt in shifted_tv):
                score += 1
        return score

    def _infer_monthly_offsets(
        self,
        our_trades: List[Trade],
        tv_trades: List[Dict[str, Any]],
        global_offset: timedelta,
    ) -> Dict[Tuple[int, int], timedelta]:
        """
        Infer month-by-month offsets (DST-aware) and fall back to global offset
        when month samples are sparse.
        """
        our_times = [t.entry_time for t in our_trades if t.entry_time is not None]
        tv_times = [t.get('entry_time') for t in tv_trades if t.get('entry_time') is not None]
        if not our_times or not tv_times:
            return {}

        tolerance = timedelta(seconds=self.TIME_TOLERANCE_SECONDS)
        month_offsets: Dict[Tuple[int, int], timedelta] = {}

        month_keys = sorted({(t.year, t.month) for t in tv_times})
        for key in month_keys:
            tv_month = [t for t in tv_times if (t.year, t.month) == key]
            if len(tv_month) < 2:
                month_offsets[key] = global_offset
                continue

            # Include nearby our times to account for session/day offsets near boundaries.
            y, m = key
            our_month = [t for t in our_times if abs((t.year - y) * 12 + (t.month - m)) <= 1]
            if not our_month:
                month_offsets[key] = global_offset
                continue

            best_offset = global_offset
            best_score = self._score_offset(our_month, tv_month, global_offset, tolerance)

            for minutes in range(-14 * 60, 14 * 60 + 1, 30):
                candidate = timedelta(minutes=minutes)
                score = self._score_offset(our_month, tv_month, candidate, tolerance)
                if score > best_score:
                    best_score = score
                    best_offset = candidate

            # Guard against unstable monthly overfitting on tiny sample counts.
            min_required = max(1, int(len(tv_month) * 0.5))
            if best_score < min_required:
                month_offsets[key] = global_offset
            else:
                month_offsets[key] = best_offset

        return month_offsets

    def _infer_time_offset(self, our_trades: List[Trade], tv_trades: List[Dict[str, Any]]) -> timedelta:
        """
        Infer constant time offset between TV export times and engine times.

        Tries offsets in 30-minute steps over +/-14 hours and chooses the one
        with the highest number of entry-time matches within tolerance.
        """
        if not our_trades or not tv_trades:
            return timedelta(0)

        our_times = [t.entry_time for t in our_trades if t.entry_time is not None]
        tv_times = [t.get('entry_time') for t in tv_trades if t.get('entry_time') is not None]
        if not our_times or not tv_times:
            return timedelta(0)

        tolerance = timedelta(seconds=self.TIME_TOLERANCE_SECONDS)
        best_offset = timedelta(0)
        best_score = -1

        # 30-minute grid captures common exchange/session offsets.
        for minutes in range(-14 * 60, 14 * 60 + 1, 30):
            offset = timedelta(minutes=minutes)
            shifted_tv = [t + offset for t in tv_times]

            score = 0
            for ot in our_times:
                # Count a match if any shifted TV entry lands on our bar timestamp.
                if any(abs((ot - tt)) <= tolerance for tt in shifted_tv):
                    score += 1

            if score > best_score:
                best_score = score
                best_offset = offset

        return best_offset
    
    def _compare_trade(
        self,
        idx: int,
        our_trade: Trade,
        tv_trade: Dict[str, Any],
        last_csv_time: datetime = None,
        is_last_trade: bool = False,
    ) -> TradeComparison:
        """Compare a single trade."""
        differences = []
        
        our_entry_time = our_trade.entry_time
        tv_entry_time = tv_trade.get('entry_time')
        
        our_exit_time = our_trade.exit_time
        tv_exit_time = tv_trade.get('exit_time')
        
        our_entry_price = our_trade.entry_price
        tv_entry_price = tv_trade.get('entry_price', 0)
        
        our_exit_price = our_trade.exit_price
        tv_exit_price = tv_trade.get('exit_price')
        
        our_exit_signal = our_trade.exit_signal.value if our_trade.exit_signal else None
        tv_exit_signal = tv_trade.get('exit_signal')
        our_direction = our_trade.direction if getattr(our_trade, 'direction', None) else None
        tv_direction = tv_trade.get('direction')
        
        our_pnl = our_trade.pnl
        tv_pnl = tv_trade.get('pnl', 0)
        
        # Compare entry time
        if our_entry_time and tv_entry_time:
            time_diff = abs((our_entry_time - tv_entry_time).total_seconds())
            if time_diff > self.TIME_TOLERANCE_SECONDS:
                differences.append(f"entry_time ({our_entry_time} vs {tv_entry_time})")
        
        # Compare exit time
        if our_exit_time and tv_exit_time:
            time_diff = abs((our_exit_time - tv_exit_time).total_seconds())
            if time_diff > self.TIME_TOLERANCE_SECONDS:
                differences.append(f"exit_time ({our_exit_time} vs {tv_exit_time})")
        
        # Compare entry price
        if abs(our_entry_price - tv_entry_price) > self.PRICE_TOLERANCE:
            differences.append(f"entry_price ({our_entry_price:.2f} vs {tv_entry_price:.2f})")

        # Compare direction (long/short) when available.
        if our_direction and tv_direction and our_direction != tv_direction:
            differences.append(f"direction ({our_direction} vs {tv_direction})")
        
        # Compare exit price
        if our_exit_price is not None and tv_exit_price is not None:
            if abs(our_exit_price - tv_exit_price) > self.PRICE_TOLERANCE:
                differences.append(f"exit_price ({our_exit_price:.2f} vs {tv_exit_price:.2f})")
        
        # Compare exit signal (normalize for comparison)
        if our_exit_signal and tv_exit_signal:
            our_signal_norm = self._normalize_signal(our_exit_signal)
            tv_signal_norm = self._normalize_signal(tv_exit_signal)
            if our_signal_norm != tv_signal_norm:
                differences.append(f"exit_signal ({our_exit_signal} vs {tv_exit_signal})")
        
        # Compare P&L
        if tv_pnl != 0:
            pnl_diff_pct = abs(our_pnl - tv_pnl) / abs(tv_pnl) * 100
            if pnl_diff_pct > self.PNL_TOLERANCE_PCT:
                differences.append(f"pnl ({our_pnl:.2f} vs {tv_pnl:.2f}, {pnl_diff_pct:.1f}% diff)")
        elif our_pnl != 0 and abs(our_pnl) > 1.0:
            differences.append(f"pnl ({our_pnl:.2f} vs {tv_pnl:.2f})")
        
        # Special handling for open trades
        is_open_trade = (tv_exit_signal and 'open' in tv_exit_signal.lower()) or our_trade.is_open()
        
        if is_last_trade and is_open_trade:
            if last_csv_time and tv_exit_time and tv_exit_time > last_csv_time:
                # Trade is still open because dataset ended - ignore differences
                differences = []
        
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
            our_direction=our_direction,
            tv_direction=tv_direction,
            our_pnl=our_pnl,
            tv_pnl=tv_pnl,
            our_entry_bar=our_trade.entry_bar,
            our_exit_bar=our_trade.exit_bar,
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
            'our_direction': comparison.our_direction,
            'tv_direction': comparison.tv_direction,
            'our_pnl': comparison.our_pnl,
            'tv_pnl': comparison.tv_pnl,
            'our_entry_bar': comparison.our_entry_bar,
            'our_exit_bar': comparison.our_exit_bar,
        }
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        lines = []
        lines.append("=" * 60)
        lines.append("TRADINGVIEW VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        status = "[PASS]" if result.passed else "[FAIL]"
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

        if result.first_divergence:
            fd = result.first_divergence
            lines.append("FIRST DIVERGENCE")
            lines.append("-" * 60)
            lines.append(f"Trade #: {fd.get('trade_number')}")
            lines.append(f"Our Entry Bar: {fd.get('our_entry_bar')}")
            lines.append(f"Our Exit Bar: {fd.get('our_exit_bar')}")
            lines.append(f"Our Entry Time: {fd.get('our_entry_time')}")
            lines.append(f"TV Entry Time: {fd.get('tv_entry_time')}")
            lines.append(f"Our Exit Time: {fd.get('our_exit_time')}")
            lines.append(f"TV Exit Time: {fd.get('tv_exit_time')}")
            cats = fd.get('categories') or []
            lines.append(f"Categories: {', '.join(cats) if cats else 'n/a'}")
            lines.append("")

        if result.mismatch_breakdown:
            lines.append("MISMATCH BREAKDOWN")
            lines.append("-" * 60)
            for cat, count in result.mismatch_breakdown.items():
                lines.append(f"- {cat}: {count}")
            lines.append("")

        if result.feasibility_blockers or result.feasibility_warnings:
            lines.append("PARITY FEASIBILITY")
            lines.append("-" * 60)
            lines.append(f"Time alignment mode: {result.time_alignment_mode}")
            lines.append(f"Inferred offset (minutes): {result.inferred_time_offset_minutes}")
            if result.feasibility_blockers:
                lines.append("Blockers:")
                for msg in result.feasibility_blockers:
                    lines.append(f"  - {msg}")
            if result.feasibility_warnings:
                lines.append("Warnings:")
                for msg in result.feasibility_warnings:
                    lines.append(f"  - {msg}")
            session_diag = result.session_diagnostics or {}
            if session_diag:
                lines.append("Session diagnostics:")
                lines.append(
                    "  - TV entries within CSV range: "
                    f"{session_diag.get('tv_entries_within_data_range_pct', 'n/a')}%"
                )
                lines.append(
                    "  - TV entries outside CSV hours: "
                    f"{session_diag.get('tv_entries_outside_data_hours_pct', 'n/a')}%"
                )
                lines.append(
                    "  - Raw TV entries outside CSV hours: "
                    f"{session_diag.get('raw_tv_entries_outside_data_hours_pct', 'n/a')}%"
                )
                lines.append(
                    "  - TV entries outside CSV minute buckets: "
                    f"{session_diag.get('tv_entries_outside_data_minutes_pct', 'n/a')}%"
                )
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


def validate_against_tradingview(
    our_trades: List[Trade],
    excel_path: str,
    last_csv_time: datetime = None,
    time_alignment: str = "dynamic",
    data_times: Optional[List[datetime]] = None,
) -> ValidationResult:
    """
    Convenience function to validate trades against TradingView export.
    
    Args:
        our_trades: List of Trade objects from backtest
        excel_path: Path to TradingView Excel export
        last_csv_time: Last timestamp in OHLC data
        
    Returns:
        ValidationResult
    """
    validator = TradingViewValidator(excel_path=excel_path, time_alignment=time_alignment)
    return validator.validate(our_trades, last_csv_time, data_times=data_times)
