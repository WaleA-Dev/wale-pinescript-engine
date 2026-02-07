"""
Backtest Engine - Position Management and Trade Execution

This module implements the core backtesting loop with:
1. Proper fill timing (signal on bar N close, fill on bar N+1 open)
2. Stop loss and take profit execution at intrabar prices
3. Dynamic ATR-based stop/TP recalculation per bar (matching TradingView)
4. Position sizing with percent_of_equity support
5. Trade tracking with detailed statistics

The execution model matches TradingView's default behavior:
- process_orders_on_close = false
- Stops/limits trigger intrabar at the specified price (or gap open if beyond)
- strategy.exit() recalculates stop/limit levels every bar
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum

from .indicators import ema, atr, adx, rsi, crossover, crossunder
from .parser import StrategyParams


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
    SIGNAL_EXIT = "Signal"
    TIME_EXIT = "Time"
    OPEN = "Open"


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


@dataclass
class Trade:
    """
    Represents a single trade with full tracking.

    Tracks entry/exit details, P&L, and intrabar extremes for
    stop loss and take profit calculations.
    """
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
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    # Stop management - these are recalculated per bar for dynamic exits
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

    def __init__(self, config: BacktestConfig = None, params: StrategyParams = None):
        self.config = config or BacktestConfig()
        self.params = params or StrategyParams()

        # State
        self.equity = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.position = 0  # Current position size
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

            # 1. Execute pending exit at open (from signal exits)
            if self.pending_exit and self.current_trade:
                self._execute_exit(bar_time, bar_open, i, self.pending_exit_signal)
                self.pending_exit = False
                self.pending_exit_signal = None

            # 2. Execute pending entry at open
            if self.pending_entry and not self.current_trade:
                self._execute_entry(bar_time, bar_open, i, df, self.pending_entry_direction)
                self.pending_entry = False

            # 3. If in a trade with active exit orders, check stops and targets
            if self.current_trade and self.current_trade.exit_orders_active:
                trade = self.current_trade
                trade.bars_in_trade += 1

                # Update intrabar extremes
                trade.max_high = max(trade.max_high, bar_high)
                trade.min_low = min(trade.min_low, bar_low)

                # Calculate excursions
                if trade.direction == "long":
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100
                    )
                else:
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100
                    )

                # Recalculate dynamic stop/TP levels based on current ATR
                if self.params.dynamic_exits and self.params.sl_atr_mult > 0:
                    current_atr = df.iloc[i]['atr'] if 'atr' in df.columns and not np.isnan(df.iloc[i]['atr']) else 0
                    if current_atr > 0:
                        if trade.direction == "long":
                            trade.stop_loss_price = trade.entry_price - (current_atr * self.params.sl_atr_mult)
                            if self.params.tp_atr_mult > 0:
                                trade.profit_target_price = trade.entry_price + (current_atr * self.params.tp_atr_mult)
                        else:
                            trade.stop_loss_price = trade.entry_price + (current_atr * self.params.sl_atr_mult)
                            if self.params.tp_atr_mult > 0:
                                trade.profit_target_price = trade.entry_price - (current_atr * self.params.tp_atr_mult)

                # Determine exit priority when both stop and TP could trigger
                stop_hit = False
                target_hit = False

                if trade.stop_loss_price:
                    stop_hit = self._check_stop_loss(bar_open, bar_low, bar_high, trade)
                if trade.profit_target_price:
                    target_hit = self._check_profit_target(bar_open, bar_high, bar_low, trade)

                if stop_hit and target_hit:
                    # Both could trigger - use bar direction to determine priority
                    # Bearish bar (close < open) → stop likely hit first
                    # Bullish bar (close >= open) → target likely hit first
                    if bar_close < bar_open:
                        # Bearish: stop first
                        exit_price = self._get_stop_fill_price(bar_open, bar_low, bar_high, trade.stop_loss_price, trade.direction)
                        self._execute_exit(bar_time, exit_price, i, ExitSignal.STOP_LOSS)
                    else:
                        # Bullish: target first
                        exit_price = self._get_target_fill_price(bar_open, bar_high, bar_low, trade.profit_target_price, trade.direction)
                        self._execute_exit(bar_time, exit_price, i, ExitSignal.PROFIT_TARGET)
                    self._record_equity(bar_close)
                    continue
                elif stop_hit:
                    exit_price = self._get_stop_fill_price(bar_open, bar_low, bar_high, trade.stop_loss_price, trade.direction)
                    self._execute_exit(bar_time, exit_price, i, ExitSignal.STOP_LOSS)
                    self._record_equity(bar_close)
                    continue
                elif target_hit:
                    exit_price = self._get_target_fill_price(bar_open, bar_high, bar_low, trade.profit_target_price, trade.direction)
                    self._execute_exit(bar_time, exit_price, i, ExitSignal.PROFIT_TARGET)
                    self._record_equity(bar_close)
                    continue

                # Check trailing stop (only if explicitly enabled)
                if self.params.use_trailing_stop and self.params.trailing_pct > 0:
                    self._update_trailing_stop(bar_high, bar_low, trade)
                    if trade.trailing_active and trade.trail_stop:
                        if self._check_trailing_stop(bar_open, bar_low, bar_high, trade):
                            exit_price = self._get_stop_fill_price(bar_open, bar_low, bar_high, trade.trail_stop, trade.direction)
                            self._execute_exit(bar_time, exit_price, i, ExitSignal.TRAILING_STOP)
                            self._record_equity(bar_close)
                            continue

                # Check custom exit signals
                if exit_signal_func and exit_signal_func(df, i, self.params, trade):
                    self.pending_exit = True
                    self.pending_exit_signal = ExitSignal.SIGNAL_EXIT

            elif self.current_trade and not self.current_trade.exit_orders_active:
                # Trade just entered this bar - count it but exit orders not yet active
                # At bar close, strategy.exit() will be called, making orders active next bar
                trade = self.current_trade
                trade.bars_in_trade += 1
                trade.max_high = max(trade.max_high, bar_high)
                trade.min_low = min(trade.min_low, bar_low)

                # Calculate excursions
                if trade.direction == "long":
                    trade.max_favorable_excursion = max(
                        trade.max_favorable_excursion,
                        (bar_high - trade.entry_price) / trade.entry_price * 100
                    )
                    trade.max_adverse_excursion = max(
                        trade.max_adverse_excursion,
                        (trade.entry_price - bar_low) / trade.entry_price * 100
                    )

                # At bar close: strategy.exit() runs, making exit orders active for next bar
                trade.exit_orders_active = True

            # 4. At bar close: check entry signals (only if not in a trade)
            if not self.current_trade and not self.pending_entry:
                if entry_signal_func:
                    if entry_signal_func(df, i, self.params):
                        self.pending_entry = True
                        self.pending_entry_direction = "long"
                else:
                    # Default entry logic using parsed strategy indicators
                    direction = self._default_entry_signal(df, i)
                    if direction:
                        self.pending_entry = True
                        self.pending_entry_direction = direction

            # Record equity at bar close
            self._record_equity(bar_close)

        # Close any open trade at the end
        if self.current_trade:
            self.current_trade.exit_signal = ExitSignal.OPEN

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

        # Two-EMA crossover system
        if self.params.use_ema_crossover:
            df['ema_fast'] = ema(close, self.params.ema_fast_length)
            df['ema_slow'] = ema(close, self.params.ema_slow_length)

        # Single EMA filter
        if self.params.use_ema_filter:
            df['ema'] = ema(close, self.params.ema_length)

        # ADX
        if self.params.use_adx_filter:
            df['adx'] = adx(high, low, close, self.params.adx_length)

        # RSI
        if self.params.use_rsi_filter:
            df['rsi'] = rsi(close, self.params.rsi_length)

        # Momentum EMAs (separate from crossover EMAs)
        if self.params.use_momentum_confirm:
            df['momentum_fast'] = ema(close, self.params.momentum_ema_fast)
            df['momentum_slow'] = ema(close, self.params.momentum_ema_slow)

        return df

    def _default_entry_signal(self, df: pd.DataFrame, bar_idx: int) -> Optional[str]:
        """
        Default entry signal logic matching TradingView PineScript patterns.

        Returns "long", "short", or None.
        """
        if bar_idx < 1:
            return None

        # Already in a position - no entry
        if self.current_trade:
            return None

        long_ok = True
        short_ok = self.params.enable_shorts

        # --- Two-EMA crossover trend filter ---
        if self.params.use_ema_crossover:
            if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
                return None
            ema_f = df.iloc[bar_idx]['ema_fast']
            ema_s = df.iloc[bar_idx]['ema_slow']
            if np.isnan(ema_f) or np.isnan(ema_s):
                return None

            # Long requires uptrend (emaFast > emaSlow)
            if ema_f <= ema_s:
                long_ok = False

            # Short requires downtrend (emaFast < emaSlow)
            if ema_f >= ema_s:
                short_ok = False

        # --- Single EMA filter ---
        if self.params.use_ema_filter:
            if 'ema' not in df.columns or np.isnan(df.iloc[bar_idx]['ema']):
                return None
            if df.iloc[bar_idx]['close'] < df.iloc[bar_idx]['ema']:
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

            # Use rsi_min/rsi_max if set (from parsed rsiMin/rsiMax)
            if self.params.rsi_min > 0 or self.params.rsi_max < 100:
                # Long: rsi > rsiMin AND rsi < rsiMax
                if rsi_val <= self.params.rsi_min or rsi_val >= self.params.rsi_max:
                    long_ok = False
                # Short: rsi < (100 - rsiMin)
                if rsi_val >= (100 - self.params.rsi_min):
                    short_ok = False
            else:
                # Fallback to overbought/oversold
                if rsi_val > self.params.rsi_overbought or rsi_val < self.params.rsi_oversold:
                    long_ok = False

        # --- Momentum confirmation ---
        if self.params.use_momentum_confirm:
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
            # TradingView: qty = floor(equity * pct / 100 / price)
            available_capital = self.equity * (self.config.order_size_pct / 100.0)
        else:
            # Default: percent of cash
            available_capital = self.cash * (self.config.order_size_pct / 100.0)

        qty = int(available_capital / price)

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

            # ATR-based profit target
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
            # Percentage-based profit target
            if direction == "long":
                trade.profit_target_price = price * (1 + self.params.profit_target_pct / 100.0)
            else:
                trade.profit_target_price = price * (1 - self.params.profit_target_pct / 100.0)

        # Exit orders are NOT active on entry bar
        # They become active after strategy.exit() runs at bar close
        trade.exit_orders_active = False

        # Update state
        self.current_trade = trade
        self.position = qty
        self.cash -= (entry_value + commission)

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

    def _check_stop_loss(self, bar_open: float, bar_low: float, bar_high: float, trade: Trade) -> bool:
        """Check if stop loss was hit."""
        if not trade.stop_loss_price:
            return False
        if trade.direction == "long":
            if bar_open <= trade.stop_loss_price:
                return True
            if bar_low <= trade.stop_loss_price:
                return True
        else:
            if bar_open >= trade.stop_loss_price:
                return True
            if bar_high >= trade.stop_loss_price:
                return True
        return False

    def _check_trailing_stop(self, bar_open: float, bar_low: float, bar_high: float, trade: Trade) -> bool:
        """Check if trailing stop was hit."""
        if not trade.trail_stop:
            return False
        if trade.direction == "long":
            if bar_open <= trade.trail_stop:
                return True
            if bar_low <= trade.trail_stop:
                return True
        else:
            if bar_open >= trade.trail_stop:
                return True
            if bar_high >= trade.trail_stop:
                return True
        return False

    def _check_profit_target(self, bar_open: float, bar_high: float, bar_low: float, trade: Trade) -> bool:
        """Check if profit target was hit."""
        if not trade.profit_target_price:
            return False
        if trade.direction == "long":
            if bar_open >= trade.profit_target_price:
                return True
            if bar_high >= trade.profit_target_price:
                return True
        else:
            if bar_open <= trade.profit_target_price:
                return True
            if bar_low <= trade.profit_target_price:
                return True
        return False

    def _update_trailing_stop(self, bar_high: float, bar_low: float, trade: Trade):
        """Update trailing stop based on price movement."""
        if not self.params.use_trailing_stop or self.params.trailing_pct <= 0:
            return

        if trade.direction == "long":
            activation_price = trade.entry_price * (1 + self.params.trailing_pct / 100.0)
            if bar_high >= activation_price:
                trade.trailing_active = True
            if trade.trailing_active:
                new_trail = trade.max_high * (1 - self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail > trade.trail_stop:
                    trade.trail_stop = new_trail
        else:
            activation_price = trade.entry_price * (1 - self.params.trailing_pct / 100.0)
            if bar_low <= activation_price:
                trade.trailing_active = True
            if trade.trailing_active:
                new_trail = trade.min_low * (1 + self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail < trade.trail_stop:
                    trade.trail_stop = new_trail

    def _get_stop_fill_price(self, bar_open: float, bar_low: float, bar_high: float,
                              stop_price: float, direction: str) -> float:
        """Get fill price for stop order (handles gaps)."""
        if direction == "long":
            if bar_open <= stop_price:
                return bar_open
            return stop_price
        else:
            if bar_open >= stop_price:
                return bar_open
            return stop_price

    def _get_target_fill_price(self, bar_open: float, bar_high: float, bar_low: float,
                                target_price: float, direction: str) -> float:
        """Get fill price for profit target (handles gaps)."""
        if direction == "long":
            if bar_open >= target_price:
                return bar_open
            return target_price
        else:
            if bar_open <= target_price:
                return bar_open
            return target_price

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


def run_backtest(df: pd.DataFrame, params: StrategyParams = None,
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
