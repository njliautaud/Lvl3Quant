#!/usr/bin/env python3
"""
Momentum Strategy — Runner Script
Executes the backtest and saves results.

Usage:
    python run_backtest.py                    # default config
    python run_backtest.py --top_n 30         # override parameters
    python run_backtest.py --sweep            # parameter sweep
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from strategy.momentum.data import fetch_prices, fetch_spy, LIQUID_100
from strategy.momentum.engine import (
    MomentumConfig,
    run_backtest,
    compute_metrics,
    print_report,
)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/momentum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single(config: MomentumConfig, start: str = "2012-01-01", end: str | None = None, tag: str = "default"):
    """Run a single backtest with given config."""
    print(f"\n{'='*60}")
    print(f"  Running momentum backtest: {tag}")
    print(f"{'='*60}")

    # Fetch data
    prices = fetch_prices(tickers=LIQUID_100, start=start, end=end)
    spy = fetch_spy(start=start, end=end)

    # Run backtest
    result = run_backtest(prices, config)
    equity = result["equity_curve"]

    # Compute metrics
    metrics = compute_metrics(equity, spy=spy, config=config)

    # Print report
    print_report(metrics, config=result["config"])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{tag}_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Save equity curve
    equity.to_csv(run_dir / "equity_curve.csv")

    # Save metrics + config
    output = {
        "tag": tag,
        "timestamp": timestamp,
        "config": result["config"],
        "metrics": metrics,
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "n_rebalances": len(result["rebal_log"]),
    }
    (run_dir / "results.json").write_text(json.dumps(output, indent=2, default=str))

    # Save rebal log
    (run_dir / "rebal_log.json").write_text(json.dumps(result["rebal_log"], indent=2))

    print(f"\n  Results saved to: {run_dir}")

    return output


def run_sweep(start: str = "2012-01-01"):
    """Run a parameter sweep across key dimensions."""
    configs = [
        ("top10_monthly_eq", MomentumConfig(top_n=10, rebal_freq="monthly", weighting="equal")),
        ("top20_monthly_eq", MomentumConfig(top_n=20, rebal_freq="monthly", weighting="equal")),
        ("top30_monthly_eq", MomentumConfig(top_n=30, rebal_freq="monthly", weighting="equal")),
        ("top20_monthly_ivol", MomentumConfig(top_n=20, rebal_freq="monthly", weighting="inv_vol")),
        ("top20_weekly_eq", MomentumConfig(top_n=20, rebal_freq="weekly", weighting="equal")),
        ("top10_weekly_eq", MomentumConfig(top_n=10, rebal_freq="weekly", weighting="equal")),
        # Vary momentum weights
        ("top20_pure12m", MomentumConfig(top_n=20, w_12m=1.0, w_6m=0.0)),
        ("top20_pure6m", MomentumConfig(top_n=20, w_12m=0.0, w_6m=1.0)),
    ]

    all_results = []
    for tag, cfg in configs:
        try:
            result = run_single(cfg, start=start, tag=tag)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR in {tag}: {e}")
            continue

    # Summary table
    if all_results:
        print("\n\n" + "=" * 90)
        print("  PARAMETER SWEEP SUMMARY")
        print("=" * 90)
        print(f"  {'Tag':<25} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'MoWR%':>7} {'RegGap':>7} {'Pass':>5}")
        print("-" * 90)
        for r in all_results:
            m = r["metrics"]
            rg = m.get("regime", {})
            print(f"  {r['tag']:<25} {m['cagr']:>7.2f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
                  f"{m['max_drawdown_pct']:>8.2f} {m['monthly_win_rate']:>7.2f} "
                  f"{rg.get('regime_gap', 'N/A'):>7} {'Y' if rg.get('regime_pass', False) else 'N':>5}")

        # Save sweep summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = OUTPUT_DIR / f"sweep_summary_{timestamp}.json"
        summary_file.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"\n  Sweep summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Momentum Strategy Backtester")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--rebal", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--weighting", choices=["equal", "inv_vol"], default="equal")
    parser.add_argument("--cost", type=float, default=0.001, help="Cost per trade (0.001 = 0.1%%)")
    parser.add_argument("--w_12m", type=float, default=0.6)
    parser.add_argument("--w_6m", type=float, default=0.4)
    parser.add_argument("--start", type=str, default="2012-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--tag", type=str, default="default")

    args = parser.parse_args()

    if args.sweep:
        run_sweep(start=args.start)
    else:
        config = MomentumConfig(
            top_n=args.top_n,
            rebal_freq=args.rebal,
            weighting=args.weighting,
            cost_per_trade=args.cost,
            w_12m=args.w_12m,
            w_6m=args.w_6m,
        )
        run_single(config, start=args.start, end=args.end, tag=args.tag)


if __name__ == "__main__":
    main()
