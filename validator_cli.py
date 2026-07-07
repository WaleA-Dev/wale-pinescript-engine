"""
CLI entry point for the Strategy Validator .exe

Usage:
    StrategyValidator.exe --list-strategies
    StrategyValidator.exe --strategy donchian --data myfile.csv
    StrategyValidator.exe --strategy donchian --data myfile.csv --full-validation
    StrategyValidator.exe --download QQQ --interval 1d
    StrategyValidator.exe --validate-all --data myfile.csv
"""

import argparse
import sys
import os

# Ensure matplotlib uses non-interactive backend for CLI/frozen
import matplotlib
matplotlib.use("Agg")

# Add project root to path for frozen exe
if getattr(sys, 'frozen', False):
    _root = os.path.dirname(sys.executable)
else:
    _root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from src.strategies import STRATEGY_REGISTRY, get_strategy_class
from src.data_loader import load_data, download_yahoo
from src.bar_returns import compute_bar_returns, compute_metrics
from src.optimization import grid_search
from src.validation.full_validation import run_full_validation


def parse_args():
    p = argparse.ArgumentParser(
        description="Strategy Validator - Statistical validation for trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list-strategies", action="store_true", help="List available strategies")
    p.add_argument("--strategy", type=str, help="Strategy name to validate")
    p.add_argument("--data", type=str, help="Path to CSV data file with OHLC columns")
    p.add_argument("--download", type=str, help="Download data for symbol (e.g., QQQ, SPY)")
    p.add_argument("--interval", type=str, default="1d", help="Data interval (default: 1d)")
    p.add_argument("--quick", action="store_true", help="Quick mode: optimization only, skip permutation tests")
    p.add_argument("--full-validation", action="store_true", help="Full 4-step validation with permutation tests")
    p.add_argument("--validate-all", action="store_true", help="Validate all registered strategies")
    p.add_argument("--train-years", type=float, default=4.0, help="Training window in years")
    p.add_argument("--n-perms-is", type=int, default=200, help="In-sample permutations")
    p.add_argument("--n-perms-wf", type=int, default=100, help="Walk-forward permutations")
    p.add_argument("--output-root", default="reports", help="Output directory for reports")
    return p.parse_args()


def cmd_list_strategies():
    print("\nAvailable Strategies:")
    print("=" * 40)
    for name in sorted(set(STRATEGY_REGISTRY.keys())):
        cls = STRATEGY_REGISTRY[name]
        print(f"  {name:20s} -> {cls.__name__}")
    print()


def cmd_download(symbol, interval):
    print(f"\nDownloading {symbol} ({interval})...")
    df = download_yahoo(symbol, interval=interval)
    print(f"Done. {len(df)} bars downloaded.\n")


def cmd_quick_validate(strategy_name, data_path):
    print(f"\n--- Quick Validation: {strategy_name} ---")
    strategy_cls = get_strategy_class(strategy_name)
    df = load_data(data_path)
    print(f"Loaded {len(df)} bars from {data_path}")

    # Get param grid
    try:
        grid = strategy_cls().param_grid()
        if not grid:
            grid = {"__dummy": [None]}
    except Exception:
        grid = {"__dummy": [None]}

    if "__dummy" in grid:
        strategy = strategy_cls()
        signals = strategy.generate_signals(df)
        result = compute_bar_returns(df, signals=signals)
        metrics = compute_metrics(result["strategy_return"])
        best_params = {}
    else:
        best_params, _ = grid_search(df, strategy_cls, grid)
        strategy = strategy_cls(**best_params)
        signals = strategy.generate_signals(df)
        result = compute_bar_returns(df, signals=signals)
        metrics = compute_metrics(result["strategy_return"])

    print(f"\nResults for {strategy_name}:")
    print(f"  Best Params: {best_params}")
    print(f"  Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return:  {metrics['total_return']:.2%}")
    print(f"  Max Drawdown:  {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate:      {metrics['win_rate']:.2%}")
    print(f"  Num Trades:    {metrics['num_trades']}")
    print()


def cmd_full_validation(strategy_name, data_path, args):
    result = run_full_validation(
        strategy_name=strategy_name,
        data_path=data_path,
        n_perms_is=args.n_perms_is,
        n_perms_wf=args.n_perms_wf,
        train_years=args.train_years,
        output_root=args.output_root,
        quick=args.quick,
    )
    return result


def main():
    args = parse_args()

    if args.list_strategies:
        cmd_list_strategies()
        return 0

    if args.download:
        cmd_download(args.download, args.interval)
        return 0

    if args.validate_all:
        if not args.data:
            print("Error: --data is required with --validate-all")
            return 1
        seen = set()
        for name in sorted(STRATEGY_REGISTRY.keys()):
            cls = STRATEGY_REGISTRY[name]
            if cls in seen:
                continue
            seen.add(cls)
            try:
                if args.full_validation:
                    cmd_full_validation(name, args.data, args)
                else:
                    cmd_quick_validate(name, args.data)
            except Exception as e:
                print(f"  ERROR validating {name}: {e}")
        return 0

    if args.strategy:
        if not args.data:
            print("Error: --data is required with --strategy")
            return 1
        if args.full_validation:
            cmd_full_validation(args.strategy, args.data, args)
        else:
            cmd_quick_validate(args.strategy, args.data)
        return 0

    print("No action specified. Use --help for usage information.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
