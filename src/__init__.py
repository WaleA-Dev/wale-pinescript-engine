# Wale PineScript Engine
# A production-grade PineScript to Python backtesting engine

from .indicators import ema, rma, atr, adx, sma, tr
from .parser import PineScriptParser, StrategyParams
from .backtest import BacktestEngine, BacktestConfig, Trade
from .validator import TradingViewValidator
from .bar_returns import compute_bar_returns, compute_metrics
from .optimization import grid_search
from .permutation import permute_bars
from .validation import (
    in_sample_permutation_test,
    run_full_validation,
    walk_forward_backtest,
    walk_forward_permutation_test,
)
from .pine_translator import PineParser, PineTranslator, TranslationPipeline, TranslationValidator

__version__ = "0.2.0"
__all__ = [
    "ema", "rma", "atr", "adx", "sma", "tr",
    "PineScriptParser", "StrategyParams",
    "BacktestEngine", "BacktestConfig", "Trade",
    "TradingViewValidator",
    "compute_bar_returns", "compute_metrics",
    "permute_bars", "grid_search",
    "in_sample_permutation_test", "walk_forward_backtest", "walk_forward_permutation_test",
    "run_full_validation",
    "PineParser", "PineTranslator", "TranslationPipeline", "TranslationValidator",
]
