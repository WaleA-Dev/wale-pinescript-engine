"""
Backtest Engine - Position Management and Trade Execution

This module implements the core backtesting loop with:
1. Proper fill timing (signal on bar N close, fill on bar N+1 open)
2. Stop loss and take profit execution at intrabar prices
3. Dynamic ATR-based stop/TP recalculation per bar (matching TradingView)
4. Position sizing with percent_of_equity support
5. Trade tracking with detailed statistics
6. Custom oscillator entry (Saty Phase pattern)
7. Consolidation filter with EMA slope, range compression, momentum, ADX
8. Trailing stop with separate activation threshold
9. strategy.close() exit semantics (market close at next bar open)

The execution model matches TradingView's default behavior:
- process_orders_on_close = false
- Stops/limits trigger intrabar at the specified price (or gap open if beyond)
- strategy.exit() recalculates stop/limit levels every bar
- strategy.close() queues market close for next bar open
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum

from .indicators import ema, atr, adx, rsi, crossover, crossunder, highest, lowest


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class ExitSignal(Enum):
    STOP_LOSS = "SL"
    TRAILING_STOP = "Trail"
    PROFIT_TARGET = "PT"
    OB_EXIT = "OB"
    OS_EXIT = "OS"
    STALE_EXIT = "Stale"
    SIGNAL_EXIT = "Signal"
    TIME_EXIT = "Time"
    OPEN = "Open"
    MARGIN_CALL = "Margin call"


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    initial_capital: float = 100000.0
    commission_pct: float = 0.1
    order_size_pct: float = 100.0  # percent_of_equity value
    qty_type: str = "percent_of_equity"  # or "fixed", "percent_of_cash"
    slippage_pct: float = 0.0
    pyramiding: int = 0  # 0 = no pyramiding
    margin_pct: float = 100.0  # 100 = no margin
    parity_mode: str = "standard"  # "standard" or "strict"
    process_orders_on_close: bool = False
    calc_on_order_fills: bool = False


@dataclass
class Trade:
    """Represents a single trade with full tracking."""
    trade_id: int
    entry_time: datetime
    entry_price: float
    entry_bar: int
    direction: str = "long"  # "long" or "short"
    qty: int = 0
    entry_value: float = 0.0

    # Exit details (filled when trade closes)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_bar: Optional[int] = None
    exit_signal: Optional[ExitSignal] = None

    # P&L
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0

    # Intrabar tracking
    max_high: float = 0.0
    min_low: float = float('inf')
    highest_since_entry: float = 0.0  # Highest close since entry (for trailing)
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Stop management
    stop_loss_price: Optional[float] = None
    trailing_active: bool = False
    trail_stop: Optional[float] = None
    profit_target_price: Optional[float] = None

    # Whether exit orders have been placed (1-bar delay for strategy.exit)
    exit_orders_active: bool = False

    # Duration
    bars_in_trade: int = 0

    def is_open(self) -> bool:
        return self.exit_time is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'entry_bar': self.entry_bar,
            'direction': self.direction,
            'qty': self.qty,
            'entry_value': self.entry_value,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_bar': self.exit_bar,
            'exit_signal': self.exit_signal.value if self.exit_signal else None,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'bars_in_trade': self.bars_in_trade,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
        }


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray

    # Summary statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0

    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl': self.avg_pnl,
            'avg_winner': self.avg_winner,
            'avg_loser': self.avg_loser,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
        }


class BacktestEngine:
    """
    Core backtesting engine with TradingView-equivalent execution.

    Execution model per bar (process_orders_on_close = false):

    1. Execute pending entry order at bar open
    2. If in a position with ACTIVE exit orders:
       a. Recalculate stop/TP levels based on current ATR (dynamic exits)
       b. Check stop loss against bar low (long) or bar high (short)
       c. Check profit target against bar high (long) or bar low (short)
       d. If both could trigger on same bar, use bar direction to determine priority
    3. At bar close: evaluate entry signals, place strategy.exit() orders
       - Entry signals queue for next bar open
       - Exit orders become active on the NEXT bar
    """

    def __init__(self, config: BacktestConfig = None, params=None):
        self.config = config or BacktestConfig()
        # Accept StrategyParams from the new parser
        from .parser import StrategyParams
        self.params = params or StrategyParams()

        # State
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.position = 0
        self.current_trade: Optional[Trade] = None
        self.trades: List[Trade] = []
        self.trade_counter = 0

        # Pending orders
        self.pending_entry = False
        self.pending_entry_direction = "long"
        self.pending_exit = False
        self.pending_exit_signal: Optional[ExitSignal] = None

        # Tracking
        self.equity_curve: List[float] = []
        self.peak_equity = self.config.initial_capital
        self.drawdown_curve: List[float] = []

    def reset(self):
        """Reset engine state for a new backtest."""
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.position = 0
        self.current_trade = None
        self.trades = []
        self.trade_counter = 0
        self.pending_entry = False
        self.pending_entry_direction = "long"
        self.pending_exit = False
        self.pending_exit_signal = None
        self.equity_curve = []
        self.peak_equity = self.config.initial_capital
        self.drawdown_curve = []

    def run(self, df: pd.DataFrame,
            entry_signal_func: Optional[Callable] = None,
            exit_signal_func: Optional[Callable] = None) -> BacktestResult:
        """
        Run backtest on OHLC data.

        Args:
            df: DataFrame with columns: time, open, high, low, close
            entry_signal_func: Custom entry signal function(df, bar_idx, params) -> bool
            exit_signal_func: Custom exit signal function(df, bar_idx, params, trade) -> bool

        Returns:
            BacktestResult with trades and statistics
        """
        self.reset()

        # Validate data
        required_cols = ['time', 'open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Precompute indicators
        df = self._compute_indicators(df)

        n = len(df)

        # Main backtest loop
        for i in range(n):
            bar_time = df.iloc[i]['time']
            bar_open = df.iloc[i]['open']
            bar_high = df.iloc[i]['high']
            bar_low = df.iloc[i]['low']
            bar_close = df.iloc[i]['close']

            # Skip bars with invalid data
            if np.isnan(bar_open) or np.isnan(bar_high) or np.isnan(bar_low) or np.isnan(bar_close):
                self._record_equity(bar_close)
                continue

            def _queue_or_execute_close_exit(signal: ExitSignal) -> None:
                """Handle strategy.close-like exits with process_orders_on_close semantics."""
                if self.config.process_orders_on_close and self.current_trade:
                    self._execute_exit(bar_time, bar_close, i, signal)
                else:
                    self.pending_exit = True
                    self.pending_exit_signal = signal

            # 1. Execute pending exit at open (from signal exits like strategy.close)
            if self.pending_exit and self.current_trade:
                self._execute_exit(bar_time, bar_open, i, self.pending_exit_signal)
                self.pending_exit = False
                self.pending_exit_signal = None

            # 2. Execute pending entry at open
            if self.pending_entry and not self.current_trade:
                self._execute_entry(bar_time, bar_open, i, df, self.pending_entry_direction)
                self.pending_entry = False

            # 3. If in a trade with active exit orders, process exits.
            if self.current_trade and self.current_trade.exit_orders_active:
                trade = self.current_trade
                trade.bars_in_trade += 1

                # Update intrabar extremes.
                trade.max_high = max(trade.max_high, bar_high)
                trade.min_low = min(trade.min_low, bar_low)
                if trade.direction == "long":
                    trade.highest_since_entry = max(trade.highest_since_entry, bar_high)
                else:
                    trade.highest_since_entry = min(trade.highest_since_entry, bar_low)

                # Calculate excursions.
                if trade.direction == "long":
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100,
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100,
                    )
                else:
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100,
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100,
                    )

                # --- Intrabar strategy.exit checks using order levels from PREVIOUS close ---
                # If an intrabar exit fills, TradingView can still evaluate close logic
                # on the same bar and queue a fresh entry for next bar.
                trade_exited_intrabar = False
                if self.params.exit_type in ("strategy_exit", "mixed"):
                    stop_price, stop_signal = self._get_effective_stop_order(trade)
                    stop_hit = False
                    target_hit = False

                    if stop_price is not None:
                        stop_hit = self._check_stop_price(
                            bar_open,
                            bar_low,
                            bar_high,
                            stop_price,
                            trade.direction,
                        )
                    if trade.profit_target_price is not None:
                        target_hit = self._check_profit_target(
                            bar_open,
                            bar_high,
                            bar_low,
                            trade,
                        )

                    if stop_hit and target_hit:
                        # Ambiguous bar. Use directional heuristic consistently.
                        if bar_close < bar_open:
                            exit_price = self._get_stop_fill_price(
                                bar_open,
                                bar_low,
                                bar_high,
                                stop_price,
                                trade.direction,
                            )
                            self._execute_exit(bar_time, exit_price, i, stop_signal)
                        else:
                            exit_price = self._get_target_fill_price(
                                bar_open,
                                bar_high,
                                bar_low,
                                trade.profit_target_price,
                                trade.direction,
                            )
                            self._execute_exit(bar_time, exit_price, i, ExitSignal.PROFIT_TARGET)
                        trade_exited_intrabar = True
                    if stop_hit:
                        if not trade_exited_intrabar:
                            exit_price = self._get_stop_fill_price(
                                bar_open,
                                bar_low,
                                bar_high,
                                stop_price,
                                trade.direction,
                            )
                            self._execute_exit(bar_time, exit_price, i, stop_signal)
                            trade_exited_intrabar = True
                    if target_hit:
                        if not trade_exited_intrabar:
                            exit_price = self._get_target_fill_price(
                                bar_open,
                                bar_high,
                                bar_low,
                                trade.profit_target_price,
                                trade.direction,
                            )
                            self._execute_exit(bar_time, exit_price, i, ExitSignal.PROFIT_TARGET)
                            trade_exited_intrabar = True

                # --- Close-based exits (strategy.close semantics) ---
                exit_queued = False

                if (not trade_exited_intrabar) and self.current_trade and self.params.exit_type == "strategy_close":
                    if trade.stop_loss_price is not None:
                        if trade.direction == "long" and bar_close <= trade.stop_loss_price:
                            _queue_or_execute_close_exit(ExitSignal.STOP_LOSS)
                            exit_queued = True

                    if not exit_queued and self.params.use_trailing_stop and self.params.trailing_pct > 0:
                        if trade.trailing_active and trade.trail_stop is not None:
                            if trade.direction == "long" and bar_close <= trade.trail_stop:
                                _queue_or_execute_close_exit(ExitSignal.TRAILING_STOP)
                                exit_queued = True

                # Signal exits that coexist with strategy.exit orders.
                if (
                    (not trade_exited_intrabar)
                    and self.current_trade
                    and not exit_queued
                    and self.params.exit_type in ("strategy_close", "mixed")
                ):
                    if self.params.use_ob_exit and 'oscillator' in df.columns and i > 0:
                        osc_curr = df.iloc[i]['oscillator']
                        osc_prev = df.iloc[i - 1]['oscillator']
                        if not np.isnan(osc_curr) and not np.isnan(osc_prev):
                            if osc_prev >= self.params.ob_threshold and osc_curr < self.params.ob_threshold:
                                _queue_or_execute_close_exit(ExitSignal.OB_EXIT)
                                exit_queued = True

                    if (
                        not exit_queued
                        and self.params.use_os_exit
                        and 'oscillator' in df.columns
                        and i > 0
                    ):
                        osc_curr = df.iloc[i]['oscillator']
                        osc_prev = df.iloc[i - 1]['oscillator']
                        if not np.isnan(osc_curr) and not np.isnan(osc_prev):
                            if osc_prev <= self.params.os_threshold and osc_curr > self.params.os_threshold:
                                _queue_or_execute_close_exit(ExitSignal.OS_EXIT)
                                exit_queued = True

                    if (
                        not exit_queued
                        and getattr(self.params, "use_stale_recycle", False)
                        and not trade.trailing_active
                        and trade.bars_in_trade >= getattr(self.params, "stale_recycle_bars", 0)
                    ):
                        current_pnl_pct = self._current_pnl_pct(bar_close, trade)
                        if (
                            current_pnl_pct >= getattr(self.params, "stale_recycle_min_pnl_pct", 0.0)
                            and current_pnl_pct <= getattr(self.params, "stale_recycle_max_pnl_pct", 0.0)
                        ):
                            _queue_or_execute_close_exit(ExitSignal.STALE_EXIT)
                            exit_queued = True

                    if (
                        not exit_queued
                        and getattr(self.params, "use_emergency_exit", False)
                        and trade.bars_in_trade >= getattr(self.params, "max_hold_bars", 0)
                    ):
                        _queue_or_execute_close_exit(ExitSignal.TIME_EXIT)
                        exit_queued = True

                # Custom exit callback.
                if (not trade_exited_intrabar) and not exit_queued and exit_signal_func and self.current_trade:
                    if exit_signal_func(df, i, self.params, self.current_trade):
                        _queue_or_execute_close_exit(ExitSignal.SIGNAL_EXIT)

                # Refresh stop/trail levels at close; they become active next bar.
                if (not trade_exited_intrabar) and self.current_trade:
                    self._refresh_exit_orders_for_next_bar(
                        df=df,
                        bar_idx=i,
                        bar_close=bar_close,
                        bar_high=bar_high,
                        bar_low=bar_low,
                        trade=trade,
                    )

            elif self.current_trade and not self.current_trade.exit_orders_active:
                # Trade just entered this bar. Exit orders become active next bar.
                trade = self.current_trade
                trade.bars_in_trade += 1
                trade.max_high = max(trade.max_high, bar_high)
                trade.min_low = min(trade.min_low, bar_low)
                if trade.direction == "long":
                    trade.highest_since_entry = max(trade.highest_since_entry, bar_high)
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100,
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100,
                    )
                else:
                    trade.highest_since_entry = min(trade.highest_since_entry, bar_low)
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100,
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100,
                    )

                self._refresh_exit_orders_for_next_bar(
                    df=df,
                    bar_idx=i,
                    bar_close=bar_close,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    trade=trade,
                )
                trade.exit_orders_active = True

            # 4. Optional reversal-on-signal (pyramiding=0 style).
            if (
                self.current_trade
                and not self.pending_exit
                and not self.pending_entry
                and getattr(self.params, "allow_reversal", False)
            ):
                reverse_direction = self._default_entry_signal(df, i, allow_in_position=True)
                if reverse_direction and reverse_direction != self.current_trade.direction:
                    if self.config.process_orders_on_close:
                        self._execute_exit(bar_time, bar_close, i, ExitSignal.SIGNAL_EXIT)
                        if not self.current_trade:
                            self._execute_entry(bar_time, bar_close, i, df, reverse_direction)
                    else:
                        self.pending_exit = True
                        self.pending_exit_signal = ExitSignal.SIGNAL_EXIT
                        self.pending_entry = True
                        self.pending_entry_direction = reverse_direction

            # 5. At bar close: check entry signals (only if not in a trade)
            if not self.current_trade and not self.pending_entry:
                if entry_signal_func:
                    if entry_signal_func(df, i, self.params):
                        if self.config.process_orders_on_close:
                            self._execute_entry(bar_time, bar_close, i, df, "long")
                        else:
                            self.pending_entry = True
                            self.pending_entry_direction = "long"
                else:
                    # Default entry logic using parsed strategy indicators
                    direction = self._default_entry_signal(df, i)
                    if direction:
                        if self.config.process_orders_on_close:
                            self._execute_entry(bar_time, bar_close, i, df, direction)
                        else:
                            self.pending_entry = True
                            self.pending_entry_direction = direction

            # Record equity at bar close
            self._record_equity(bar_close)

        # Close any open trade at the end
        if self.current_trade:
            trade = self.current_trade
            mark_price = float(df.iloc[-1]['close']) if len(df) > 0 else trade.entry_price
            if trade.direction == "long":
                gross_pnl = (mark_price - trade.entry_price) * trade.qty
            else:
                gross_pnl = (trade.entry_price - mark_price) * trade.qty
            # TradingView open PnL includes entry commission plus a hypothetical
            # close commission at the marked price.
            mark_exit_commission = mark_price * trade.qty * (self.config.commission_pct / 100.0)
            net_pnl = gross_pnl - trade.commission - mark_exit_commission
            trade.pnl = net_pnl
            trade.pnl_pct = (net_pnl / trade.entry_value) * 100.0 if trade.entry_value > 0 else 0.0
            trade.exit_signal = ExitSignal.OPEN
            self.trades.append(trade)
            self.current_trade = None
            self.position = 0

        # Calculate statistics
        return self._calculate_results(df)

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute all indicators needed for the strategy."""
        df = df.copy()

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # ATR for stops (always compute)
        df['atr'] = atr(high, low, close, self.params.atr_length)

        # EMA200 (for filter and consolidation)
        if (self.params.use_ema_filter or self.params.use_consolidation_filter) and 'ema200' not in df.columns:
            df['ema200'] = ema(close, self.params.ema_length)

        # Two-EMA crossover system
        if self.params.use_ema_crossover or getattr(self.params, "use_rsi_pullback_entry", False):
            df['ema_fast'] = ema(close, self.params.ema_fast_length)
            df['ema_slow'] = ema(close, self.params.ema_slow_length)

        # Single EMA filter
        if self.params.use_ema_filter and 'ema200' not in df.columns:
            df['ema200'] = ema(close, self.params.ema_length)

        # ADX
        if self.params.use_adx_filter or self.params.use_consolidation_filter:
            df['adx'] = adx(high, low, close, self.params.adx_length)

        # RSI
        if self.params.use_rsi_filter or getattr(self.params, "use_rsi_pullback_entry", False):
            df['rsi'] = rsi(close, self.params.rsi_length)

        # Momentum EMAs
        if self.params.use_momentum_confirm or self.params.use_consolidation_filter:
            df['momentum_fast'] = ema(close, self.params.momentum_ema_fast)
            df['momentum_slow'] = ema(close, self.params.momentum_ema_slow)
            # Pine-style momentum confirmation used by anti-consolidation logic:
            # momentum_bullish = ema_fast > ema_slow
            # momentum_expanding = ema_fast > ema_fast[5] and ema_slow > ema_slow[5]
            momentum_expanding = (
                (df['momentum_fast'] > df['momentum_fast'].shift(5))
                & (df['momentum_slow'] > df['momentum_slow'].shift(5))
            )
            df['momentum_confirmed'] = (
                (df['momentum_fast'] > df['momentum_slow']) & momentum_expanding
            ).fillna(False)

        # Custom oscillator (Saty Phase pattern)
        if self.params.use_oscillator_entry:
            # Prefer pre-exported oscillator series when available. This improves
            # signal parity for TradingView-exported CSVs that already contain the
            # exact oscillator values used by the script.
            if 'oscillator' in df.columns and not df['oscillator'].isna().all():
                df['oscillator'] = pd.to_numeric(df['oscillator'], errors='coerce')
            else:
                osc_ema = ema(close, self.params.oscillator_ema_len)
                osc_atr = atr(high, low, close, self.params.oscillator_atr_len)

                n = len(close)
                raw_osc = np.full(n, np.nan)
                for j in range(n):
                    if not np.isnan(osc_ema[j]) and not np.isnan(osc_atr[j]) and osc_atr[j] != 0:
                        raw_osc[j] = ((close[j] - osc_ema[j]) / (self.params.oscillator_atr_mult * osc_atr[j])) * self.params.oscillator_scale

                # Smooth with EMA
                df['oscillator'] = ema(raw_osc, self.params.oscillator_smooth_len)

        # Consolidation filter
        if self.params.use_consolidation_filter:
            df['is_consolidating'] = self._compute_consolidation(df)

        return df

    def _compute_consolidation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute consolidation filter matching Saty Phase logic.

        Consolidation is based on trend/range compression:
        1. EMA200 slope is not trending
        2. Price range is not expanding

        Additional momentum/adx checks are handled separately in entry logic.
        """
        n = len(df)
        is_consolidating = np.zeros(n, dtype=bool)

        ema200 = df['ema200'].values if 'ema200' in df.columns else np.full(n, np.nan)
        high = df['high'].values
        low = df['low'].values

        slope_lookback = self.params.ema_slope_check_lookback
        slope_threshold = self.params.ema_slope_check_threshold
        range_lookback = self.params.consolidation_lookback
        range_threshold = self.params.consolidation_threshold

        for i in range(n):
            ema_is_trending = True
            price_is_expanding = True

            # 1) EMA slope trend check (absolute % change over lookback)
            if i >= slope_lookback and not np.isnan(ema200[i]) and not np.isnan(ema200[i - slope_lookback]):
                base = ema200[i - slope_lookback]
                if base != 0:
                    slope_pct = abs((ema200[i] - base) / base) * 100.0
                    ema_is_trending = slope_pct >= slope_threshold

            # 2) Price range expansion check over lookback window
            if i >= range_lookback:
                h_window = high[i - range_lookback + 1: i + 1]
                l_window = low[i - range_lookback + 1: i + 1]
                if len(h_window) > 0 and not np.isnan(h_window).any() and not np.isnan(l_window).any():
                    h_max = np.max(h_window)
                    l_min = np.min(l_window)
                    if l_min > 0:
                        price_range_pct = ((h_max - l_min) / l_min) * 100
                        price_is_expanding = price_range_pct >= range_threshold

            # Pine-style: consolidating if NOT trending or NOT expanding.
            is_consolidating[i] = (not ema_is_trending) or (not price_is_expanding)

        return is_consolidating

    def _default_entry_signal(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        allow_in_position: bool = False,
    ) -> Optional[str]:
        """
        Default entry signal logic matching TradingView PineScript patterns.

        Supports:
        - Oscillator crossover entry (Saty Phase pattern)
        - Two-EMA crossover entry
        - RSI/ADX/Momentum filters
        - Consolidation filter

        Returns "long", "short", or None.
        """
        if bar_idx < 1:
            return None

        # Already in a position - no entry (unless caller is explicitly checking reversals)
        if self.current_trade and not allow_in_position:
            return None

        # In strict parity mode only, honor explicit Buy markers from CSV.
        # Standard mode should always use parsed strategy logic.
        if self.config.parity_mode == "strict" and 'buy' in df.columns:
            buy_val = df.iloc[bar_idx]['buy']
            if not np.isnan(buy_val) and float(buy_val) > 0:
                if not self._buy_marker_state_consistent(df, bar_idx):
                    return None
                return "long"
            return None

        # --- Consolidation filter: block entry during consolidation ---
        if self.params.use_consolidation_filter:
            if 'is_consolidating' in df.columns:
                if df.iloc[bar_idx]['is_consolidating']:
                    # Pine-style override: when consolidating, allow entries only if
                    # momentum confirmation is enabled and currently true.
                    if not self.params.use_momentum_confirm:
                        return None
                    if 'momentum_confirmed' not in df.columns:
                        return None
                    if not bool(df.iloc[bar_idx]['momentum_confirmed']):
                        return None

        # --- Oscillator crossover entry ---
        if self.params.use_oscillator_entry:
            if 'oscillator' not in df.columns:
                return None

            osc_curr = df.iloc[bar_idx]['oscillator']
            osc_prev = df.iloc[bar_idx - 1]['oscillator']

            if np.isnan(osc_curr) or np.isnan(osc_prev):
                return None

            threshold = self.params.entry_threshold

            # Normal entry: oscillator crosses above threshold
            normal_entry = (osc_prev <= threshold and osc_curr > threshold)

            # Extreme entry: oscillator crosses from extreme level through threshold
            extreme_entry = False
            if self.params.use_extreme_entry:
                extreme_level = self.params.extreme_threshold
                # Check if oscillator was below extreme level recently and now crossing above threshold
                if osc_prev <= threshold and osc_curr > threshold:
                    # Look back to see if it was at extreme recently
                    lookback = min(bar_idx, 10)
                    for k in range(1, lookback + 1):
                        if bar_idx - k >= 0:
                            prev_osc = df.iloc[bar_idx - k]['oscillator']
                            if not np.isnan(prev_osc) and prev_osc <= extreme_level:
                                extreme_entry = True
                                break

            # Secondary oscillator entry rule for scripts that use asymmetric
            # thresholds, e.g. osc[1] <= -50 and osc > -110.
            secondary_entry = False
            if self.params.use_secondary_osc_entry:
                secondary_entry = (
                    osc_prev <= self.params.secondary_prev_threshold
                    and osc_curr > self.params.secondary_curr_threshold
                )

            if normal_entry or extreme_entry or secondary_entry:
                # EMA filter (optional)
                if self.params.use_ema_filter:
                    ema_col = 'ema200' if 'ema200' in df.columns else 'ema'
                    if ema_col in df.columns:
                        ema_val = df.iloc[bar_idx][ema_col]
                        if not np.isnan(ema_val) and df.iloc[bar_idx]['close'] < ema_val:
                            return None

                return "long"

            return None

        # --- RSI pullback crossover entry ---
        if getattr(self.params, "use_rsi_pullback_entry", False):
            if (
                'rsi' not in df.columns
                or 'ema_fast' not in df.columns
                or 'ema_slow' not in df.columns
            ):
                return None

            rsi_prev = df.iloc[bar_idx - 1]['rsi']
            rsi_curr = df.iloc[bar_idx]['rsi']
            ema_fast = df.iloc[bar_idx]['ema_fast']
            ema_slow = df.iloc[bar_idx]['ema_slow']

            if any(np.isnan(v) for v in (rsi_prev, rsi_curr, ema_fast, ema_slow)):
                return None

            trend_up = ema_fast > ema_slow
            trend_dn = ema_fast < ema_slow
            long_level = float(getattr(self.params, "rsi_pullback_long", 45.0))
            short_level = float(getattr(self.params, "rsi_pullback_short", 55.0))

            long_signal = trend_up and (rsi_prev <= long_level and rsi_curr > long_level)
            short_signal = trend_dn and (rsi_prev >= short_level and rsi_curr < short_level)

            if long_signal:
                return "long"
            if short_signal and self.params.enable_shorts:
                return "short"
            return None

        # --- Two-EMA crossover entry (original pattern) ---
        long_ok = True
        short_ok = self.params.enable_shorts

        if self.params.use_ema_crossover:
            if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
                return None
            ema_f = df.iloc[bar_idx]['ema_fast']
            ema_s = df.iloc[bar_idx]['ema_slow']
            if np.isnan(ema_f) or np.isnan(ema_s):
                return None
            if ema_f <= ema_s:
                long_ok = False
            if ema_f >= ema_s:
                short_ok = False

        # --- Single EMA filter ---
        if self.params.use_ema_filter:
            ema_col = 'ema200' if 'ema200' in df.columns else 'ema'
            if ema_col in df.columns and not np.isnan(df.iloc[bar_idx][ema_col]):
                if df.iloc[bar_idx]['close'] < df.iloc[bar_idx][ema_col]:
                    long_ok = False

        # --- ADX filter ---
        if self.params.use_adx_filter:
            if 'adx' not in df.columns or np.isnan(df.iloc[bar_idx]['adx']):
                return None
            if df.iloc[bar_idx]['adx'] < self.params.adx_threshold:
                long_ok = False
                short_ok = False

        # --- RSI filter ---
        if self.params.use_rsi_filter:
            if 'rsi' not in df.columns or np.isnan(df.iloc[bar_idx]['rsi']):
                return None
            rsi_val = df.iloc[bar_idx]['rsi']

            if self.params.rsi_min > 0 or self.params.rsi_max < 100:
                if rsi_val <= self.params.rsi_min or rsi_val >= self.params.rsi_max:
                    long_ok = False
                if rsi_val >= (100 - self.params.rsi_min):
                    short_ok = False
            else:
                if rsi_val > self.params.rsi_overbought or rsi_val < self.params.rsi_oversold:
                    long_ok = False

        # --- Momentum confirmation ---
        if self.params.use_momentum_confirm and not self.params.use_consolidation_filter:
            if 'momentum_fast' not in df.columns or 'momentum_slow' not in df.columns:
                return None
            mf = df.iloc[bar_idx]['momentum_fast']
            ms = df.iloc[bar_idx]['momentum_slow']
            if np.isnan(mf) or np.isnan(ms):
                return None
            if mf <= ms:
                long_ok = False

        if long_ok:
            return "long"
        elif short_ok:
            return "short"
        return None

    def _execute_entry(self, time: datetime, price: float, bar_idx: int,
                       df: pd.DataFrame, direction: str = "long"):
        """Execute an entry order with proper position sizing."""
        self.trade_counter += 1

        # Calculate position size based on qty_type
        if self.config.qty_type == "percent_of_equity":
            available_capital = self.equity * (self.config.order_size_pct / 100.0)
        else:
            available_capital = self.cash * (self.config.order_size_pct / 100.0)

        # Strict parity mode uses TradingView-like qty sizing off price, then applies
        # commission separately. Standard mode reserves commission in qty sizing.
        if self.config.parity_mode == "strict":
            qty = int(available_capital / price)
        else:
            price_with_commission = price * (1.0 + self.config.commission_pct / 100.0)
            qty = int(available_capital / price_with_commission)

        if qty <= 0:
            return

        entry_value = qty * price
        commission = entry_value * (self.config.commission_pct / 100.0)

        # Create trade
        trade = Trade(
            trade_id=self.trade_counter,
            entry_time=time,
            entry_price=price,
            entry_bar=bar_idx,
            direction=direction,
            qty=qty,
            entry_value=entry_value,
            commission=commission,
            max_high=price,
            min_low=price,
            highest_since_entry=price,
        )

        # Set initial stop/TP levels
        if self.params.sl_atr_mult > 0:
            # ATR-based stop loss
            atr_val = df.iloc[bar_idx]['atr'] if 'atr' in df.columns and not np.isnan(df.iloc[bar_idx]['atr']) else 0
            if atr_val > 0:
                if direction == "long":
                    trade.stop_loss_price = price - (atr_val * self.params.sl_atr_mult)
                else:
                    trade.stop_loss_price = price + (atr_val * self.params.sl_atr_mult)

            if self.params.tp_atr_mult > 0 and atr_val > 0:
                if direction == "long":
                    trade.profit_target_price = price + (atr_val * self.params.tp_atr_mult)
                else:
                    trade.profit_target_price = price - (atr_val * self.params.tp_atr_mult)
        elif self.params.stop_loss_pct > 0:
            # Percentage-based stop
            if direction == "long":
                trade.stop_loss_price = price * (1 - self.params.stop_loss_pct / 100.0)
            else:
                trade.stop_loss_price = price * (1 + self.params.stop_loss_pct / 100.0)

        if self.params.use_profit_target and self.params.profit_target_pct > 0 and self.params.tp_atr_mult <= 0:
            # For oscillator strategies with trailing, profit_target_pct is trail ACTIVATION threshold
            # Don't set a hard profit target price if we have trailing
            if not (self.params.use_trailing_stop and self.params.trailing_pct > 0):
                if direction == "long":
                    trade.profit_target_price = price * (1 + self.params.profit_target_pct / 100.0)
                else:
                    trade.profit_target_price = price * (1 - self.params.profit_target_pct / 100.0)

        # By default, exit orders become active on the next bar.
        trade.exit_orders_active = False

        # Update state
        self.current_trade = trade
        self.position = qty
        self.cash -= (entry_value + commission)

        # TradingView's calc_on_order_fills=true recalculates immediately after fill.
        # Enable exit orders right away so they can participate in the current bar's
        # intrabar path (for open fills) or be ready for next bar (for close fills).
        if self.config.calc_on_order_fills and self.current_trade is not None:
            self._refresh_exit_orders_for_next_bar(
                df=df,
                bar_idx=bar_idx,
                bar_close=float(df.iloc[bar_idx]['close']),
                bar_high=float(df.iloc[bar_idx]['high']),
                bar_low=float(df.iloc[bar_idx]['low']),
                trade=self.current_trade,
            )
            self.current_trade.exit_orders_active = True

    def _execute_exit(self, time: datetime, price: float, bar_idx: int, signal: ExitSignal):
        """Execute an exit order."""
        if not self.current_trade:
            return

        trade = self.current_trade
        exit_value = trade.qty * price
        commission = exit_value * (self.config.commission_pct / 100.0)

        # Calculate P&L
        if trade.direction == "long":
            gross_pnl = exit_value - trade.entry_value
        else:
            gross_pnl = trade.entry_value - exit_value

        net_pnl = gross_pnl - trade.commission - commission
        pnl_pct = (net_pnl / trade.entry_value) * 100.0

        # Update trade
        trade.exit_time = time
        trade.exit_price = price
        trade.exit_bar = bar_idx
        trade.exit_signal = signal
        trade.pnl = net_pnl
        trade.pnl_pct = pnl_pct
        trade.commission += commission

        # Update state
        self.trades.append(trade)
        self.cash += exit_value - commission
        self.equity = self.cash
        self.position = 0
        self.current_trade = None

    def _current_pnl_pct(self, bar_close: float, trade: Trade) -> float:
        """Current unrealized PnL percent for the active bar close."""
        if trade.direction == "long":
            return ((bar_close - trade.entry_price) / trade.entry_price) * 100.0
        return ((trade.entry_price - bar_close) / trade.entry_price) * 100.0

    def _refresh_exit_orders_for_next_bar(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        bar_close: float,
        bar_high: float,
        bar_low: float,
        trade: Trade,
    ) -> None:
        """
        Refresh stop/target/trailing order levels at bar close.

        These levels become active on the next bar, matching TradingView's
        default order-processing behavior.
        """
        # Dynamic ATR-based exits.
        if self.params.dynamic_exits and self.params.sl_atr_mult > 0:
            current_atr = df.iloc[bar_idx]['atr'] if 'atr' in df.columns and not np.isnan(df.iloc[bar_idx]['atr']) else 0
            if current_atr > 0:
                if trade.direction == "long":
                    trade.stop_loss_price = trade.entry_price - (current_atr * self.params.sl_atr_mult)
                    if self.params.tp_atr_mult > 0:
                        trade.profit_target_price = trade.entry_price + (current_atr * self.params.tp_atr_mult)
                else:
                    trade.stop_loss_price = trade.entry_price + (current_atr * self.params.sl_atr_mult)
                    if self.params.tp_atr_mult > 0:
                        trade.profit_target_price = trade.entry_price - (current_atr * self.params.tp_atr_mult)

        # Optional external stop columns from enriched CSV exports.
        # In strict parity mode, only trust them on bars with explicit position state.
        allow_external_levels = True
        if self.config.parity_mode == "strict":
            allow_external_levels = self._row_has_debug_position_state(df, bar_idx)

        if allow_external_levels and 'sl' in df.columns:
            ext_sl = df.iloc[bar_idx]['sl']
            if not np.isnan(ext_sl):
                trade.stop_loss_price = float(ext_sl)

        if allow_external_levels and 'trail' in df.columns:
            ext_trail = df.iloc[bar_idx]['trail']
            if not np.isnan(ext_trail):
                trade.trailing_active = True
                trade.trail_stop = float(ext_trail)

        # Update trailing stop state.
        if self.params.use_trailing_stop and self.params.trailing_pct > 0:
            if (
                getattr(self.params, "trailing_activation_on_close", False)
                or self.params.exit_type == "strategy_close"
            ):
                self._update_trailing_stop_close(bar_close, trade)
            else:
                self._update_trailing_stop(bar_high, bar_low, trade)

    def _get_effective_stop_order(self, trade: Trade) -> tuple[Optional[float], ExitSignal]:
        """
        Return the active stop price and its logical signal for this bar.

        For mixed strategy.exit logic, trailing and SL are often combined as a
        single stop order (e.g., max(stop, trail) for long).
        """
        stop_price = trade.stop_loss_price
        signal = ExitSignal.STOP_LOSS

        if (
            self.params.use_trailing_stop
            and trade.trailing_active
            and trade.trail_stop is not None
        ):
            if stop_price is None:
                stop_price = trade.trail_stop
            else:
                if trade.direction == "long":
                    stop_price = max(stop_price, trade.trail_stop)
                else:
                    stop_price = min(stop_price, trade.trail_stop)
            signal = ExitSignal.TRAILING_STOP

        return stop_price, signal

    def _row_has_debug_position_state(self, df: pd.DataFrame, bar_idx: int) -> bool:
        """
        True when debug/export columns indicate an active position on this bar.
        """
        row = df.iloc[bar_idx]

        for col in ("sl", "trail", "target", "dbg entryref", "dbg stop", "dbg trail"):
            if col in df.columns and not pd.isna(row[col]):
                return True

        if "dbg trailingactive" in df.columns and not pd.isna(row["dbg trailingactive"]):
            try:
                return float(row["dbg trailingactive"]) > 0
            except (TypeError, ValueError):
                return False

        return False

    def _buy_marker_state_consistent(self, df: pd.DataFrame, bar_idx: int) -> bool:
        """
        In strict parity mode, a buy marker is valid only if next bar shows
        position-state columns (SL/Trail/Target or related debug state).
        """
        if bar_idx + 1 >= len(df):
            return False
        return self._row_has_debug_position_state(df, bar_idx + 1)

    def _check_stop_price(
        self,
        bar_open: float,
        bar_low: float,
        bar_high: float,
        stop_price: float,
        direction: str,
    ) -> bool:
        """Check if an arbitrary stop price was touched (intrabar)."""
        if direction == "long":
            return bar_open <= stop_price or bar_low <= stop_price
        return bar_open >= stop_price or bar_high >= stop_price

    def _check_stop_loss(self, bar_open: float, bar_low: float, bar_high: float, trade: Trade) -> bool:
        """Check if stop loss was hit (intrabar)."""
        if trade.stop_loss_price is None:
            return False
        return self._check_stop_price(
            bar_open,
            bar_low,
            bar_high,
            trade.stop_loss_price,
            trade.direction,
        )

    def _check_trailing_stop(self, bar_open: float, bar_low: float, bar_high: float, trade: Trade) -> bool:
        """Check if trailing stop was hit (intrabar)."""
        if trade.trail_stop is None:
            return False
        return self._check_stop_price(
            bar_open,
            bar_low,
            bar_high,
            trade.trail_stop,
            trade.direction,
        )

    def _check_profit_target(self, bar_open: float, bar_high: float, bar_low: float, trade: Trade) -> bool:
        """Check if profit target was hit (intrabar)."""
        if trade.profit_target_price is None:
            return False
        if trade.direction == "long":
            return bar_open >= trade.profit_target_price or bar_high >= trade.profit_target_price
        else:
            return bar_open <= trade.profit_target_price or bar_low <= trade.profit_target_price

    def _update_trailing_stop(self, bar_high: float, bar_low: float, trade: Trade):
        """Update trailing stop based on intrabar prices (for strategy.exit)."""
        if not self.params.use_trailing_stop or self.params.trailing_pct <= 0:
            return

        # Determine activation threshold
        activation_pct = self.params.trail_activation_pct if self.params.trail_activation_pct > 0 else self.params.profit_target_pct

        if trade.direction == "long":
            activation_price = trade.entry_price * (1 + activation_pct / 100.0)
            if bar_high >= activation_price:
                trade.trailing_active = True
            if trade.trailing_active:
                new_trail = trade.max_high * (1 - self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail > trade.trail_stop:
                    trade.trail_stop = new_trail
        else:
            activation_price = trade.entry_price * (1 - activation_pct / 100.0)
            if bar_low <= activation_price:
                trade.trailing_active = True
            if trade.trailing_active:
                new_trail = trade.min_low * (1 + self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail < trade.trail_stop:
                    trade.trail_stop = new_trail

    def _update_trailing_stop_close(self, bar_close: float, trade: Trade):
        """
        Update trailing stop based on bar close prices (for strategy.close patterns).

        The Saty Phase strategy checks at bar close:
        - Activate trailing when profit >= profit_target_pct
        - Trail at trailing_pct below highest close since entry
        """
        if not self.params.use_trailing_stop or self.params.trailing_pct <= 0:
            return

        # Determine activation threshold
        activation_pct = self.params.trail_activation_pct if self.params.trail_activation_pct > 0 else self.params.profit_target_pct

        if trade.direction == "long":
            # Current profit percentage
            current_profit_pct = ((bar_close - trade.entry_price) / trade.entry_price) * 100

            # Activate trailing when profit exceeds threshold
            if current_profit_pct >= activation_pct:
                trade.trailing_active = True

            if trade.trailing_active:
                # Trail below highest tracked price since entry.
                new_trail = trade.highest_since_entry * (1 - self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail > trade.trail_stop:
                    trade.trail_stop = new_trail
        else:
            current_profit_pct = ((trade.entry_price - bar_close) / trade.entry_price) * 100
            if current_profit_pct >= activation_pct:
                trade.trailing_active = True
            if trade.trailing_active:
                new_trail = trade.highest_since_entry * (1 + self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail < trade.trail_stop:
                    trade.trail_stop = new_trail

    def _get_stop_fill_price(self, bar_open: float, bar_low: float, bar_high: float,
                              stop_price: float, direction: str) -> float:
        """Get fill price for stop order (handles gaps)."""
        if direction == "long":
            return bar_open if bar_open <= stop_price else stop_price
        else:
            return bar_open if bar_open >= stop_price else stop_price

    def _get_target_fill_price(self, bar_open: float, bar_high: float, bar_low: float,
                                target_price: float, direction: str) -> float:
        """Get fill price for profit target (handles gaps)."""
        if direction == "long":
            return bar_open if bar_open >= target_price else target_price
        else:
            return bar_open if bar_open <= target_price else target_price

    def _record_equity(self, current_price: float):
        """Record equity and drawdown at current bar."""
        if self.current_trade:
            position_value = self.current_trade.qty * current_price
            self.equity = self.cash + position_value
        else:
            self.equity = self.cash

        self.equity_curve.append(self.equity)

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        drawdown = self.peak_equity - self.equity
        self.drawdown_curve.append(drawdown)

    def _calculate_results(self, df: pd.DataFrame) -> BacktestResult:
        """Calculate final statistics."""
        result = BacktestResult(
            trades=self.trades,
            equity_curve=np.array(self.equity_curve),
            drawdown_curve=np.array(self.drawdown_curve),
        )

        closed_trades = [t for t in self.trades if not t.is_open()]

        if not closed_trades:
            return result

        # Basic counts
        result.total_trades = len(closed_trades)
        result.winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in closed_trades if t.pnl <= 0)
        result.win_rate = result.winning_trades / result.total_trades * 100 if result.total_trades > 0 else 0

        # P&L statistics
        pnls = [t.pnl for t in closed_trades]
        result.total_pnl = sum(pnls)
        result.avg_pnl = np.mean(pnls)

        winners = [t.pnl for t in closed_trades if t.pnl > 0]
        losers = [t.pnl for t in closed_trades if t.pnl <= 0]

        result.avg_winner = np.mean(winners) if winners else 0
        result.avg_loser = np.mean(losers) if losers else 0

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        result.max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0
        result.max_drawdown_pct = (result.max_drawdown / self.config.initial_capital) * 100

        # Risk-adjusted returns
        if len(self.equity_curve) > 1:
            eq = np.array(self.equity_curve)
            returns = np.diff(eq) / eq[:-1]

            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)

            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                result.sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)

            annual_return = (eq[-1] / eq[0] - 1) * (252 / len(eq))
            if result.max_drawdown_pct > 0:
                result.calmar_ratio = annual_return * 100 / result.max_drawdown_pct

        return result


def run_backtest(df: pd.DataFrame, params=None,
                 config: BacktestConfig = None,
                 entry_func: Callable = None,
                 exit_func: Callable = None) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        df: OHLC DataFrame
        params: Strategy parameters
        config: Backtest configuration
        entry_func: Custom entry signal function
        exit_func: Custom exit signal function

    Returns:
        BacktestResult
    """
    engine = BacktestEngine(config=config, params=params)
    return engine.run(df, entry_signal_func=entry_func, exit_signal_func=exit_func)
