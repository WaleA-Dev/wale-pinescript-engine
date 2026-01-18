"""
Backtest Engine - Position Management and Trade Execution

This module implements the core backtesting loop with:
1. Proper fill timing (signal on bar N close, fill on bar N+1 open)
2. Stop loss and trailing stop execution at intrabar prices
3. Position sizing with commission handling
4. Trade tracking with detailed statistics

The execution model matches TradingView's default behavior:
- process_orders_on_close = false
- Stops trigger intrabar at the stop price (or gap open if beyond)
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
    order_size_pct: float = 100.0
    slippage_pct: float = 0.0
    pyramiding: int = 0  # 0 = no pyramiding
    margin_pct: float = 100.0  # 100 = no margin


@dataclass
class Trade:
    """
    Represents a single trade with full tracking.
    
    Tracks entry/exit details, P&L, and intrabar extremes for
    stop loss and trailing stop calculations.
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
    
    # Stop management
    stop_loss_price: Optional[float] = None
    trailing_active: bool = False
    trail_stop: Optional[float] = None
    profit_target_price: Optional[float] = None
    
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
    
    Order Flow (per bar):
    1. Execute pending entry at open price
    2. Execute pending exit at open price  
    3. Check stop loss against bar low (long) or high (short)
    4. Check trailing stop against bar low (long) or high (short)
    5. Check profit target against bar high (long) or low (short)
    6. Evaluate exit signals
    7. Evaluate entry signals
    8. Queue new orders for next bar
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
            
            # 1. Execute pending exit at open
            if self.pending_exit and self.current_trade:
                self._execute_exit(bar_time, bar_open, i, self.pending_exit_signal)
                self.pending_exit = False
                self.pending_exit_signal = None
            
            # 2. Execute pending entry at open
            if self.pending_entry and not self.current_trade:
                self._execute_entry(bar_time, bar_open, i, df)
                self.pending_entry = False
            
            # If in a trade, check stops and update tracking
            if self.current_trade:
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
                
                # 3. Check stop loss (intrabar)
                if trade.stop_loss_price and self._check_stop_loss(bar_open, bar_low, bar_high, trade):
                    exit_price = self._get_stop_fill_price(bar_open, bar_low, bar_high, trade.stop_loss_price, trade.direction)
                    self._execute_exit(bar_time, exit_price, i, ExitSignal.STOP_LOSS)
                    self._record_equity(bar_close)
                    continue
                
                # 4. Check trailing stop (intrabar)
                self._update_trailing_stop(bar_high, bar_low, trade)
                if trade.trailing_active and trade.trail_stop:
                    if self._check_trailing_stop(bar_open, bar_low, bar_high, trade):
                        exit_price = self._get_stop_fill_price(bar_open, bar_low, bar_high, trade.trail_stop, trade.direction)
                        self._execute_exit(bar_time, exit_price, i, ExitSignal.TRAILING_STOP)
                        self._record_equity(bar_close)
                        continue
                
                # 5. Check profit target (intrabar)
                if trade.profit_target_price:
                    if self._check_profit_target(bar_open, bar_high, bar_low, trade):
                        exit_price = self._get_target_fill_price(bar_open, bar_high, bar_low, trade.profit_target_price, trade.direction)
                        self._execute_exit(bar_time, exit_price, i, ExitSignal.PROFIT_TARGET)
                        self._record_equity(bar_close)
                        continue
                
                # 6. Check custom exit signals
                if exit_signal_func and exit_signal_func(df, i, self.params, trade):
                    self.pending_exit = True
                    self.pending_exit_signal = ExitSignal.SIGNAL_EXIT
            
            # 7. Check entry signals (only if not in a trade)
            if not self.current_trade and not self.pending_entry:
                if entry_signal_func:
                    if entry_signal_func(df, i, self.params):
                        self.pending_entry = True
                else:
                    # Default entry logic using indicators
                    if self._default_entry_signal(df, i):
                        self.pending_entry = True
            
            # Record equity at bar close
            self._record_equity(bar_close)
        
        # Close any open trade at the end
        if self.current_trade:
            last_bar = df.iloc[-1]
            self.current_trade.exit_signal = ExitSignal.OPEN
        
        # Calculate statistics
        return self._calculate_results(df)
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute all indicators needed for the strategy."""
        df = df.copy()
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # EMA
        if self.params.use_ema_filter:
            df['ema'] = ema(close, self.params.ema_length)
        
        # ATR for stops
        df['atr'] = atr(high, low, close, self.params.atr_length)
        
        # ADX
        if self.params.use_adx_filter:
            df['adx'] = adx(high, low, close, self.params.adx_length)
        
        # RSI
        if self.params.use_rsi_filter:
            df['rsi'] = rsi(close, self.params.rsi_length)
        
        # Momentum EMAs
        if self.params.use_momentum_confirm:
            df['ema_fast'] = ema(close, self.params.momentum_ema_fast)
            df['ema_slow'] = ema(close, self.params.momentum_ema_slow)
        
        return df
    
    def _default_entry_signal(self, df: pd.DataFrame, bar_idx: int) -> bool:
        """
        Default entry signal logic.
        
        Override this or pass entry_signal_func for custom strategies.
        """
        if bar_idx < 1:
            return False
        
        # Check EMA filter
        if self.params.use_ema_filter:
            if 'ema' not in df.columns or np.isnan(df.iloc[bar_idx]['ema']):
                return False
            if df.iloc[bar_idx]['close'] < df.iloc[bar_idx]['ema']:
                return False
        
        # Check ADX filter
        if self.params.use_adx_filter:
            if 'adx' not in df.columns or np.isnan(df.iloc[bar_idx]['adx']):
                return False
            if df.iloc[bar_idx]['adx'] < self.params.adx_threshold:
                return False
        
        # Check RSI filter
        if self.params.use_rsi_filter:
            if 'rsi' not in df.columns or np.isnan(df.iloc[bar_idx]['rsi']):
                return False
            rsi_val = df.iloc[bar_idx]['rsi']
            if rsi_val > self.params.rsi_overbought or rsi_val < self.params.rsi_oversold:
                return False
        
        # Check momentum
        if self.params.use_momentum_confirm:
            if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
                return False
            if np.isnan(df.iloc[bar_idx]['ema_fast']) or np.isnan(df.iloc[bar_idx]['ema_slow']):
                return False
            if df.iloc[bar_idx]['ema_fast'] <= df.iloc[bar_idx]['ema_slow']:
                return False
        
        return True
    
    def _execute_entry(self, time: datetime, price: float, bar_idx: int, df: pd.DataFrame):
        """Execute an entry order."""
        self.trade_counter += 1
        
        # Calculate position size
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
            direction="long" if self.params.long_only else "short",
            qty=qty,
            entry_value=entry_value,
            commission=commission,
            max_high=price,
            min_low=price,
        )
        
        # Set stop loss
        if self.params.stop_loss_pct > 0:
            atr_val = df.iloc[bar_idx]['atr'] if 'atr' in df.columns and not np.isnan(df.iloc[bar_idx]['atr']) else 0
            if atr_val > 0:
                # ATR-based stop
                trade.stop_loss_price = price - (atr_val * self.params.atr_multiplier)
            else:
                # Percentage-based stop
                trade.stop_loss_price = price * (1 - self.params.stop_loss_pct / 100.0)
        
        # Set profit target
        if self.params.use_profit_target and self.params.profit_target_pct > 0:
            trade.profit_target_price = price * (1 + self.params.profit_target_pct / 100.0)
        
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
        if trade.direction == "long":
            # Gap down beyond stop
            if bar_open <= trade.stop_loss_price:
                return True
            # Intrabar stop hit
            if bar_low <= trade.stop_loss_price:
                return True
        else:
            # Gap up beyond stop
            if bar_open >= trade.stop_loss_price:
                return True
            # Intrabar stop hit
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
            # Activate trailing when price moves up enough
            activation_price = trade.entry_price * (1 + self.params.trailing_pct / 100.0)
            if bar_high >= activation_price:
                trade.trailing_active = True
            
            if trade.trailing_active:
                # Trail from highest high
                new_trail = trade.max_high * (1 - self.params.trailing_pct / 100.0)
                if trade.trail_stop is None or new_trail > trade.trail_stop:
                    trade.trail_stop = new_trail
        else:
            # Short position trailing
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
            # If gapped below stop, fill at open
            if bar_open <= stop_price:
                return bar_open
            return stop_price
        else:
            # If gapped above stop, fill at open
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
            # Mark-to-market
            position_value = self.current_trade.qty * current_price
            self.equity = self.cash + position_value
        else:
            self.equity = self.cash
        
        self.equity_curve.append(self.equity)
        
        # Update peak and drawdown
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
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            
            # Sharpe (assuming 0 risk-free rate, annualized for daily data)
            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Sortino (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                result.sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
            
            # Calmar
            annual_return = (self.equity_curve[-1] / self.equity_curve[0] - 1) * (252 / len(self.equity_curve))
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
