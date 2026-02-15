#!/usr/bin/env python3
"""
Wale PineScript to Python Backtest Engine

A production-grade engine that converts TradingView PineScript strategies to Python,
enabling local backtesting with exact trade-by-trade validation.

Usage:
    python backtest_engine.py --csv data.csv --pine strategy.pine
    python backtest_engine.py --csv data.csv --pine strategy.pine --excel tv_export.xlsx --run_step1 true

See docs/03-cli-reference.md for full documentation.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.parser import PineScriptParser, StrategyParams
from src.backtest import BacktestEngine, BacktestConfig, BacktestResult, run_backtest
from src.validator import TradingViewValidator, validate_against_tradingview
from src.indicators import ema, rma, atr, adx, rsi


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PineScript to Python Backtest Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic backtest:
    python backtest_engine.py --csv data.csv --pine strategy.pine

  With TradingView validation:
    python backtest_engine.py --csv data.csv --pine strategy.pine --excel export.xlsx

  Full research run:
    python backtest_engine.py --csv data.csv --pine strategy.pine --excel export.xlsx --run_step1 true --run_step2 true
        """
    )
    
    # Required arguments
    parser.add_argument('--csv', required=True, help='Path to OHLC CSV data file')
    parser.add_argument('--pine', required=True, help='Path to PineScript strategy file')
    
    # Optional arguments
    parser.add_argument('--excel', default=None, help='TradingView Excel export for validation')
    parser.add_argument('--run_step1', type=str, default='false', help='Generate Step 1 research outputs (true/false)')
    parser.add_argument('--run_step2', type=str, default='false', help='Run robustness tests (true/false)')
    parser.add_argument('--holdout_months', type=int, default=6, help='Out-of-sample holdout period')
    parser.add_argument('--initial_capital', type=float, default=100000.0, help='Starting capital')
    parser.add_argument('--commission_pct', type=float, default=0.1, help='Commission as percent of trade value')
    parser.add_argument('--output_dir', default='backtest/out', help='Output directory for results')
    
    return parser.parse_args()


def str_to_bool(s: str) -> bool:
    """Convert string to boolean."""
    return s.lower() in ('true', '1', 'yes', 'y')


def load_ohlc_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate OHLC data from CSV.
    
    Expected columns: time/date, open, high, low, close
    Optional: volume
    """
    print(f"Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Find time column
    time_cols = ['time', 'date', 'datetime', 'timestamp']
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No time column found. Expected one of: {time_cols}")
    
    # Rename to standard names
    df = df.rename(columns={time_col: 'time'})
    
    # Parse datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Validate required columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert to float
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Data quality checks
    print(f"  Loaded {len(df)} bars")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Check for NaN
    nan_counts = df[required].isna().sum()
    if nan_counts.any():
        print(f"  Warning: NaN values found: {nan_counts.to_dict()}")
    
    # Check OHLC validity
    invalid_bars = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    if invalid_bars.any():
        print(f"  Warning: {invalid_bars.sum()} bars with invalid OHLC relationships")
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    return df


def create_output_dirs(base_dir: str):
    """Create output directory structure."""
    dirs = [
        base_dir,
        f"{base_dir}/research_outputs/step1",
        f"{base_dir}/research_outputs/step2",
        f"{base_dir}/plots",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def export_trades(trades, output_path: str):
    """Export trades to CSV."""
    if not trades:
        print("  No trades to export")
        return
    
    data = [t.to_dict() for t in trades]
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"  Exported {len(trades)} trades to {output_path}")


def export_equity_curve(equity_curve: np.ndarray, times: pd.Series, output_path: str):
    """Export equity curve to CSV."""
    df = pd.DataFrame({
        'time': times,
        'equity': equity_curve,
    })
    df.to_csv(output_path, index=False)
    print(f"  Exported equity curve to {output_path}")


def generate_step1_report(result: BacktestResult, output_path: str):
    """Generate Step 1 performance report."""
    lines = []
    lines.append("=" * 60)
    lines.append("STEP 1: BACKTEST PERFORMANCE REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    stats = result.to_dict()
    
    lines.append("TRADE STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total Trades: {stats['total_trades']}")
    lines.append(f"Winning Trades: {stats['winning_trades']}")
    lines.append(f"Losing Trades: {stats['losing_trades']}")
    lines.append(f"Win Rate: {stats['win_rate']:.2f}%")
    lines.append("")
    
    lines.append("PROFIT & LOSS")
    lines.append("-" * 40)
    lines.append(f"Total P&L: ${stats['total_pnl']:,.2f}")
    lines.append(f"Average P&L: ${stats['avg_pnl']:,.2f}")
    lines.append(f"Average Winner: ${stats['avg_winner']:,.2f}")
    lines.append(f"Average Loser: ${stats['avg_loser']:,.2f}")
    lines.append(f"Profit Factor: {stats['profit_factor']:.2f}")
    lines.append("")
    
    lines.append("RISK METRICS")
    lines.append("-" * 40)
    lines.append(f"Max Drawdown: ${stats['max_drawdown']:,.2f}")
    lines.append(f"Max Drawdown %: {stats['max_drawdown_pct']:.2f}%")
    lines.append(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    lines.append(f"Sortino Ratio: {stats['sortino_ratio']:.2f}")
    lines.append(f"Calmar Ratio: {stats['calmar_ratio']:.2f}")
    lines.append("")
    
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"  Generated report: {output_path}")
    return report


def generate_plots(result: BacktestResult, times: pd.Series, output_dir: str):
    """Generate performance plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  Warning: matplotlib not available, skipping plots")
        return
    
    # Equity and Drawdown plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Equity curve
    ax1.plot(times, result.equity_curve, 'b-', linewidth=1)
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curve')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Drawdown
    ax2.fill_between(times, result.drawdown_curve, 0, color='red', alpha=0.3)
    ax2.plot(times, result.drawdown_curve, 'r-', linewidth=1)
    ax2.set_ylabel('Drawdown ($)')
    ax2.set_xlabel('Date')
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/step1_equity_drawdown.png", dpi=150)
    plt.close()
    
    print(f"  Generated plot: {output_dir}/step1_equity_drawdown.png")
    
    # Trade P&L histogram
    if result.trades:
        pnls = [t.pnl for t in result.trades if not t.is_open()]
        if pnls:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Trade #')
            ax.set_ylabel('P&L ($)')
            ax.set_title('Trade P&L Distribution')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/step1_trade_pnl_hist.png", dpi=150)
            plt.close()
            
            print(f"  Generated plot: {output_dir}/step1_trade_pnl_hist.png")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("WALE PINESCRIPT TO PYTHON BACKTEST ENGINE")
    print("=" * 60 + "\n")
    
    # Create output directories
    create_output_dirs(args.output_dir)
    
    # Load OHLC data
    try:
        df = load_ohlc_data(args.csv)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Parse PineScript
    print(f"\nParsing PineScript from {args.pine}...")
    try:
        parser = PineScriptParser(pine_path=args.pine)
        params = parser.parse_params()
        settings = parser.parse_strategy_settings()
        print(f"  Strategy: {settings.title}")
        print(f"  Parsed {len(params.custom_params)} custom parameters")
    except Exception as e:
        print(f"Error parsing PineScript: {e}")
        return 1
    
    # Configure backtest using parsed strategy settings
    qty_type = "percent_of_equity"
    if settings.default_qty_type == "percent_of_equity":
        qty_type = "percent_of_equity"
    elif settings.default_qty_type == "fixed":
        qty_type = "fixed"

    # Use PineScript settings if they differ from defaults, otherwise use CLI args
    initial_capital = settings.initial_capital if settings.initial_capital != 100000.0 else args.initial_capital
    commission_pct = settings.commission_value if settings.commission_value != 0.1 else args.commission_pct

    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        order_size_pct=settings.default_qty_value,
        qty_type=qty_type,
        pyramiding=settings.pyramiding,
    )
    print(f"  Capital: ${initial_capital:,.0f}")
    print(f"  Commission: {commission_pct}%")
    print(f"  Position Size: {settings.default_qty_value}% of equity")
    print(f"  Pyramiding: {settings.pyramiding}")
    
    # Run backtest
    print("\nRunning backtest...")
    try:
        engine = BacktestEngine(config=config, params=params)
        result = engine.run(df)
        print(f"  Completed: {result.total_trades} trades")
        print(f"  Total P&L: ${result.total_pnl:,.2f}")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.1f}%")
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Validate against TradingView
    if args.excel:
        print(f"\nValidating against TradingView export: {args.excel}")
        try:
            last_csv_time = df['time'].max()
            validator = TradingViewValidator(excel_path=args.excel)
            validation = validator.validate(result.trades, last_csv_time)
            
            report = validator.generate_report(validation)
            print(report)
            
            # Save validation report
            with open(f"{args.output_dir}/validation_report.txt", 'w') as f:
                f.write(report)
            
        except Exception as e:
            print(f"Error validating: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate Step 1 outputs
    if str_to_bool(args.run_step1):
        print("\nGenerating Step 1 research outputs...")
        
        step1_dir = f"{args.output_dir}/research_outputs/step1"
        
        # Export trades
        export_trades(result.trades, f"{step1_dir}/trade_list.csv")
        
        # Export equity curve
        export_equity_curve(result.equity_curve, df['time'], f"{step1_dir}/equity_curve.csv")
        
        # Generate report
        report = generate_step1_report(result, f"{step1_dir}/step1_report.txt")
        print(report)
        
        # Generate plots
        generate_plots(result, df['time'], f"{args.output_dir}/plots")
    
    # Step 2: Robustness tests (placeholder)
    if str_to_bool(args.run_step2):
        print("\nStep 2 robustness tests not yet implemented.")
        print("See: https://github.com/WaleA-Dev/wale-montecarlo-engine")
    
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
