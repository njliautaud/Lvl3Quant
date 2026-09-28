#!/usr/bin/env python3
"""
Monte Carlo Position Sizing Simulation
========================================

Uses the 314 actual paper trades to simulate equity curves at different
position sizes (1-10 contracts). Answers:
  1. What's the max drawdown distribution at each size?
  2. What's the probability of hitting a drawdown that triggers a stop?
  3. What's the optimal number of contracts for a $50K account?
  4. How long until the account doubles?

Method: Bootstrap resample daily trade sequences 10,000 times per position size.
Preserves intra-day trade clustering (samples whole days, not individual trades).

Author: Claude (autonomous build, 2026-06-29)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADES_PATH = ROOT / "paper_engines" / "logs" / "integrated_pipeline_paper" / "trades.csv"
DAILY_PATH = ROOT / "paper_engines" / "logs" / "integrated_pipeline_paper" / "daily_pnl.csv"
OUTPUT_DIR = ROOT / "output" / "monte_carlo_position_sizing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
ES_TICK_VALUE = 12.50
ACCOUNT_SIZE = 50_000
ES_MARGIN_PER_CONTRACT = 500  # Day trading margin (AMP)
MAX_DRAWDOWN_STOP = 0.20      # Kill switch at 20% account drawdown
N_SIMULATIONS = 10_000
TRADING_DAYS_PER_YEAR = 252
N_DAYS_FORWARD = 120          # ~6 months of trading days
CONTRACT_SIZES = [1, 2, 3, 4, 5, 7, 10]

np.random.seed(42)


def load_data():
    """Load daily PnL and individual trades."""
    daily = pd.read_csv(DAILY_PATH)
    trades = pd.read_csv(TRADES_PATH)

    # Filter to trading days only (days with trades)
    active_daily = daily[daily['n_trades'] > 0].copy()

    print(f"Total trades: {len(trades)}")
    print(f"Active trading days: {len(active_daily)}")
    print(f"Avg trades/active day: {len(trades) / len(active_daily):.1f}")
    print(f"Total net PnL (1 contract): {trades['net_pnl_ticks'].sum():.1f} ticks = ${trades['net_pnl_dollars'].sum():.0f}")

    return daily, trades, active_daily


def simulate_equity_curves(active_daily, n_contracts):
    """
    Bootstrap resample active trading days to simulate equity curves.
    Returns array of shape (N_SIMULATIONS, N_DAYS_FORWARD+1) with equity values.
    """
    daily_pnl_ticks = active_daily['net_pnl_ticks'].values
    n_active_days = len(daily_pnl_ticks)

    # Also need to account for non-trading days (56 active out of ~135 total = ~41% hit rate)
    # But for conservative analysis, simulate as if every day is a potential trading day
    # with ~41% probability of having trades
    pct_active = n_active_days / len(pd.read_csv(DAILY_PATH))

    equity_curves = np.zeros((N_SIMULATIONS, N_DAYS_FORWARD + 1))
    equity_curves[:, 0] = ACCOUNT_SIZE

    max_drawdowns = np.zeros(N_SIMULATIONS)
    stopped_out = np.zeros(N_SIMULATIONS, dtype=bool)
    days_to_double = np.full(N_SIMULATIONS, np.inf)
    final_equity = np.zeros(N_SIMULATIONS)

    for sim in range(N_SIMULATIONS):
        equity = ACCOUNT_SIZE
        peak = ACCOUNT_SIZE
        max_dd = 0
        doubled = False

        for day in range(1, N_DAYS_FORWARD + 1):
            # ~41% chance this day has trades (bootstrap from active days)
            if np.random.random() < pct_active:
                day_pnl = np.random.choice(daily_pnl_ticks)
                pnl_dollars = day_pnl * ES_TICK_VALUE * n_contracts
                equity += pnl_dollars

            equity_curves[sim, day] = equity

            # Track peak and drawdown
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

            # Check stop
            if dd >= MAX_DRAWDOWN_STOP:
                stopped_out[sim] = True
                equity_curves[sim, day:] = equity
                break

            # Check doubling
            if not doubled and equity >= ACCOUNT_SIZE * 2:
                days_to_double[sim] = day
                doubled = True

        max_drawdowns[sim] = max_dd
        final_equity[sim] = equity_curves[sim, -1]

    return {
        'equity_curves': equity_curves,
        'max_drawdowns': max_drawdowns,
        'stopped_out': stopped_out,
        'days_to_double': days_to_double,
        'final_equity': final_equity,
    }


def analyze_results(results_by_size):
    """Produce summary statistics for each position size."""
    summary = []

    for n_contracts, res in results_by_size.items():
        dd = res['max_drawdowns']
        fe = res['final_equity']
        dtd = res['days_to_double']
        so = res['stopped_out']

        row = {
            'contracts': n_contracts,
            'margin_required': n_contracts * ES_MARGIN_PER_CONTRACT,
            'margin_pct': n_contracts * ES_MARGIN_PER_CONTRACT / ACCOUNT_SIZE * 100,
            # Returns
            'median_final_equity': np.median(fe),
            'median_return_pct': (np.median(fe) - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100,
            'p5_return_pct': (np.percentile(fe, 5) - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100,
            'p95_return_pct': (np.percentile(fe, 95) - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100,
            # Drawdowns
            'median_max_dd_pct': np.median(dd) * 100,
            'p95_max_dd_pct': np.percentile(dd, 95) * 100,
            'p99_max_dd_pct': np.percentile(dd, 99) * 100,
            'worst_dd_pct': np.max(dd) * 100,
            # Risk
            'stopped_out_pct': so.mean() * 100,
            'prob_loss': (fe < ACCOUNT_SIZE).mean() * 100,
            # Doubling
            'median_days_to_double': np.median(dtd[dtd < np.inf]) if (dtd < np.inf).any() else np.inf,
            'pct_doubled': (dtd < np.inf).mean() * 100,
            # Max dollar drawdown
            'p95_dd_dollars': np.percentile(dd, 95) * ACCOUNT_SIZE,
            'worst_dd_dollars': np.max(dd) * ACCOUNT_SIZE,
        }
        summary.append(row)

    return pd.DataFrame(summary)


def main():
    print("=" * 70)
    print("MONTE CARLO POSITION SIZING SIMULATION")
    print(f"Account: ${ACCOUNT_SIZE:,} | Margin/contract: ${ES_MARGIN_PER_CONTRACT}")
    print(f"Simulations: {N_SIMULATIONS:,} | Forward days: {N_DAYS_FORWARD}")
    print(f"Max DD stop: {MAX_DRAWDOWN_STOP:.0%}")
    print("=" * 70)

    daily, trades, active_daily = load_data()

    # Per-trade stats
    wins = trades[trades['net_pnl_ticks'] > 0]
    losses = trades[trades['net_pnl_ticks'] <= 0]
    print(f"\nPer-trade stats:")
    print(f"  Win: avg +{wins['net_pnl_ticks'].mean():.1f}t (${wins['net_pnl_dollars'].mean():.0f})")
    print(f"  Loss: avg {losses['net_pnl_ticks'].mean():.1f}t (${losses['net_pnl_dollars'].mean():.0f})")
    print(f"  Win rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"  Reward/risk: {abs(wins['net_pnl_ticks'].mean() / losses['net_pnl_ticks'].mean()):.2f}")

    # Run simulations for each position size
    results_by_size = {}
    for n in CONTRACT_SIZES:
        print(f"\nSimulating {n} contract(s)...", end="", flush=True)
        results_by_size[n] = simulate_equity_curves(active_daily, n)
        print(f" done (stopped out: {results_by_size[n]['stopped_out'].mean()*100:.1f}%)")

    # Analyze
    summary = analyze_results(results_by_size)

    print("\n" + "=" * 70)
    print("POSITION SIZING RESULTS (120 trading days forward = ~6 months)")
    print("=" * 70)

    for _, row in summary.iterrows():
        n = int(row['contracts'])
        print(f"\n{'='*50}")
        print(f"  {n} CONTRACT{'S' if n > 1 else ''} (margin: ${row['margin_required']:,.0f} = {row['margin_pct']:.0f}% of account)")
        print(f"{'='*50}")
        print(f"  Median return:     +{row['median_return_pct']:.1f}%  (${row['median_final_equity']-ACCOUNT_SIZE:,.0f})")
        print(f"  5th-95th pctile:   {row['p5_return_pct']:+.1f}% to {row['p95_return_pct']:+.1f}%")
        print(f"  Prob of any loss:  {row['prob_loss']:.1f}%")
        print(f"  Median max DD:     {row['median_max_dd_pct']:.1f}% (${row['median_max_dd_pct']/100*ACCOUNT_SIZE:,.0f})")
        print(f"  95th pctile DD:    {row['p95_max_dd_pct']:.1f}% (${row['p95_dd_dollars']:,.0f})")
        print(f"  Worst-case DD:     {row['worst_dd_pct']:.1f}% (${row['worst_dd_dollars']:,.0f})")
        print(f"  Stopped out (≥20%): {row['stopped_out_pct']:.1f}%")
        if row['pct_doubled'] > 0:
            print(f"  Doubled account:   {row['pct_doubled']:.0f}% of sims (median {row['median_days_to_double']:.0f} days)")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find largest size where P(stop) < 1%
    safe_sizes = summary[summary['stopped_out_pct'] < 1.0]
    if len(safe_sizes) > 0:
        recommended = int(safe_sizes.iloc[-1]['contracts'])
        rec_row = safe_sizes.iloc[-1]
        print(f"\n  RECOMMENDED: {recommended} contract{'s' if recommended > 1 else ''}")
        print(f"  Rationale: Largest size with <1% chance of hitting 20% drawdown stop")
        print(f"  Expected 6-month return: +{rec_row['median_return_pct']:.1f}% (${rec_row['median_final_equity']-ACCOUNT_SIZE:,.0f})")
        print(f"  95th percentile max DD: {rec_row['p95_max_dd_pct']:.1f}% (${rec_row['p95_dd_dollars']:,.0f})")
    else:
        print("\n  WARNING: All sizes have >1% stop-out risk. Start with 1 contract.")

    # Conservative start
    one_ct = summary[summary['contracts'] == 1].iloc[0]
    print(f"\n  CONSERVATIVE START (1 contract):")
    print(f"    6-month median: +{one_ct['median_return_pct']:.1f}%")
    print(f"    Worst max DD: {one_ct['worst_dd_pct']:.1f}% (${one_ct['worst_dd_dollars']:,.0f})")
    print(f"    Prob of loss: {one_ct['prob_loss']:.1f}%")

    # Scale-up schedule
    print(f"\n  SCALE-UP SCHEDULE (add 1 contract when):")
    print(f"    1→2: After 20+ live trades with WR ≥ 45% and PF ≥ 1.3")
    print(f"    2→3: After 50+ live trades with WR ≥ 45% and PF ≥ 1.3")
    print(f"    3→5: After 100+ live trades, account up ≥ 15%")
    print(f"    5→7: After 200+ live trades, account up ≥ 30%")

    # Save results
    summary.to_csv(OUTPUT_DIR / "position_sizing_summary.csv", index=False)

    # Save full results as JSON
    results_json = {
        'metadata': {
            'account_size': ACCOUNT_SIZE,
            'margin_per_contract': ES_MARGIN_PER_CONTRACT,
            'n_simulations': N_SIMULATIONS,
            'n_forward_days': N_DAYS_FORWARD,
            'max_dd_stop': MAX_DRAWDOWN_STOP,
            'n_trades_input': len(trades),
            'n_active_days_input': len(active_daily),
        },
        'summary': summary.to_dict(orient='records'),
    }
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}")

    return summary


if __name__ == "__main__":
    main()
