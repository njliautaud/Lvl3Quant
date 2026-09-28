#!/usr/bin/env python3
"""
wheel_v3_monte_carlo.py — Monte Carlo simulation for forward-looking estimates.

Uses the validated best config's daily returns to bootstrap confidence intervals
for 1-year and 3-year forward performance. Answers the question:
"What's the realistic range of outcomes if we deploy this strategy?"

Also tests: drawdown probability, time-to-recovery estimates, and
probability of a losing year.
"""
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_v3_monte_carlo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from wheel_universe_v3_expand import load_all_data, compute_metrics
from wheel_earnings_filter import download_earnings_dates, build_earnings_lookup
from wheel_v3_dd_protection import run_portfolio_v3_with_overlay

logging.basicConfig(
    format='%(asctime)s [MC] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('MC')

N_SIMS = 10_000
TRADING_DAYS_PER_YEAR = 252


def bootstrap_returns(daily_returns, n_days, n_sims=N_SIMS, block_size=21):
    """
    Block bootstrap of daily returns to preserve autocorrelation.
    Uses overlapping blocks of 'block_size' trading days (~1 month).
    """
    n = len(daily_returns)
    all_paths = np.zeros((n_sims, n_days))

    for sim in range(n_sims):
        path = []
        while len(path) < n_days:
            start = np.random.randint(0, n - block_size)
            block = daily_returns[start:start + block_size]
            path.extend(block)
        all_paths[sim] = path[:n_days]

    return all_paths


def simulate_equity_paths(daily_returns_matrix, starting_equity=100_000):
    """Convert daily return paths to equity paths."""
    cumulative = np.cumprod(1 + daily_returns_matrix, axis=1)
    return starting_equity * cumulative


def compute_path_metrics(equity_paths, starting_equity=100_000):
    """Compute metrics across all simulated paths."""
    n_sims, n_days = equity_paths.shape
    years = n_days / TRADING_DAYS_PER_YEAR

    # Final equity
    final_equity = equity_paths[:, -1]
    total_returns = (final_equity / starting_equity - 1) * 100

    # CAGR per path
    cagr = ((final_equity / starting_equity) ** (1 / years) - 1) * 100

    # Max drawdown per path
    max_dd = np.zeros(n_sims)
    for i in range(n_sims):
        peak = np.maximum.accumulate(equity_paths[i])
        dd = (equity_paths[i] - peak) / peak
        max_dd[i] = dd.min() * 100

    # Sharpe per path (annualized)
    daily_rets = np.diff(equity_paths, axis=1) / equity_paths[:, :-1]
    sharpe = np.mean(daily_rets, axis=1) / np.std(daily_rets, axis=1) * np.sqrt(252)

    return {
        "final_equity": final_equity,
        "total_return_pct": total_returns,
        "cagr_pct": cagr,
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
    }


def percentile_table(values, name, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """Log a percentile distribution table."""
    pcts = np.percentile(values, percentiles)
    log.info(f"\n  {name} distribution:")
    for p, v in zip(percentiles, pcts):
        log.info(f"    p{p:02d}: {v:>10.1f}")
    log.info(f"    mean: {np.mean(values):>10.1f}")
    return dict(zip([f"p{p}" for p in percentiles], [round(v, 2) for v in pcts]))


def main():
    log.info("=" * 60)
    log.info("WHEEL V3 — MONTE CARLO FORWARD PROJECTIONS")
    log.info(f"  Simulations: {N_SIMS:,}")
    log.info(f"  Block size: 21 days (preserves monthly autocorrelation)")
    log.info("=" * 60)

    # Load data and run best config to get daily returns
    log.info("Loading data and running best config...")
    prices, spy_regime, sector_map = load_all_data()
    earnings_raw = download_earnings_dates(prices["ticker"].unique())
    earnings_lookup = build_earnings_lookup(earnings_raw)

    base_params = dict(
        starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        bear_mode="liq_csp_only", max_assignments_5d=3,
        max_share_positions=5, loss_cut_pct=-0.15,
        min_price=10.0, max_price=500.0,
        earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    equity, trades, _ = run_portfolio_v3_with_overlay(
        prices, spy_regime, sector_map, **base_params,
        overlay_type="eq_brake",
        eq_brake_lookback=60, eq_brake_threshold=0.03, eq_brake_scale=0.25,
    )

    eq_df = pd.DataFrame(equity)
    eq_df["return"] = eq_df["equity"].pct_change()
    daily_returns = eq_df["return"].dropna().values

    actual_metrics = compute_metrics(equity, 100_000)
    log.info(f"Actual backtest: CAGR {actual_metrics['cagr_pct']}%, "
             f"Sharpe {actual_metrics['sharpe']}, MaxDD {actual_metrics['max_dd_pct']}%")
    log.info(f"  {len(daily_returns)} daily returns available for bootstrapping")

    # Return characteristics
    log.info(f"\n  Daily return stats:")
    log.info(f"    Mean: {daily_returns.mean()*100:.4f}%")
    log.info(f"    Std:  {daily_returns.std()*100:.4f}%")
    log.info(f"    Skew: {pd.Series(daily_returns).skew():.3f}")
    log.info(f"    Kurt: {pd.Series(daily_returns).kurtosis():.3f}")
    log.info(f"    % positive: {(daily_returns > 0).mean()*100:.1f}%")
    log.info(f"    Worst day: {daily_returns.min()*100:.2f}%")
    log.info(f"    Best day:  {daily_returns.max()*100:.2f}%")

    results = {}

    # === 1-YEAR FORWARD PROJECTION ===
    log.info("\n" + "=" * 60)
    log.info("1-YEAR FORWARD PROJECTION (252 trading days)")
    log.info("=" * 60)

    paths_1y = bootstrap_returns(daily_returns, 252, N_SIMS)
    equity_1y = simulate_equity_paths(paths_1y)
    metrics_1y = compute_path_metrics(equity_1y)

    results["1yr"] = {}
    results["1yr"]["total_return"] = percentile_table(metrics_1y["total_return_pct"], "Total Return %")
    results["1yr"]["cagr"] = percentile_table(metrics_1y["cagr_pct"], "CAGR %")
    results["1yr"]["max_dd"] = percentile_table(metrics_1y["max_dd_pct"], "Max Drawdown %")
    results["1yr"]["sharpe"] = percentile_table(metrics_1y["sharpe"], "Sharpe Ratio")

    # Key probabilities
    prob_positive = (metrics_1y["total_return_pct"] > 0).mean() * 100
    prob_beat_spy = (metrics_1y["cagr_pct"] > 10).mean() * 100  # Rough SPY avg
    prob_lose_10 = (metrics_1y["total_return_pct"] < -10).mean() * 100
    prob_lose_20 = (metrics_1y["total_return_pct"] < -20).mean() * 100
    prob_dd_20 = (metrics_1y["max_dd_pct"] < -20).mean() * 100
    prob_dd_30 = (metrics_1y["max_dd_pct"] < -30).mean() * 100

    log.info(f"\n  Key probabilities (1-year):")
    log.info(f"    P(positive year):    {prob_positive:.1f}%")
    log.info(f"    P(beat 10% SPY):     {prob_beat_spy:.1f}%")
    log.info(f"    P(lose >10%):        {prob_lose_10:.1f}%")
    log.info(f"    P(lose >20%):        {prob_lose_20:.1f}%")
    log.info(f"    P(drawdown >20%):    {prob_dd_20:.1f}%")
    log.info(f"    P(drawdown >30%):    {prob_dd_30:.1f}%")

    results["1yr"]["probabilities"] = {
        "positive_year": round(prob_positive, 1),
        "beat_spy_10pct": round(prob_beat_spy, 1),
        "lose_10pct": round(prob_lose_10, 1),
        "lose_20pct": round(prob_lose_20, 1),
        "drawdown_20pct": round(prob_dd_20, 1),
        "drawdown_30pct": round(prob_dd_30, 1),
    }

    # === 3-YEAR FORWARD PROJECTION ===
    log.info("\n" + "=" * 60)
    log.info("3-YEAR FORWARD PROJECTION (756 trading days)")
    log.info("=" * 60)

    paths_3y = bootstrap_returns(daily_returns, 756, N_SIMS)
    equity_3y = simulate_equity_paths(paths_3y)
    metrics_3y = compute_path_metrics(equity_3y)

    results["3yr"] = {}
    results["3yr"]["total_return"] = percentile_table(metrics_3y["total_return_pct"], "Total Return %")
    results["3yr"]["cagr"] = percentile_table(metrics_3y["cagr_pct"], "CAGR %")
    results["3yr"]["max_dd"] = percentile_table(metrics_3y["max_dd_pct"], "Max Drawdown %")
    results["3yr"]["sharpe"] = percentile_table(metrics_3y["sharpe"], "Sharpe Ratio")

    # Final equity ranges
    final_eq = metrics_3y["final_equity"]
    log.info(f"\n  $100K grows to (3-year):")
    for p in [5, 25, 50, 75, 95]:
        v = np.percentile(final_eq, p)
        log.info(f"    p{p:02d}: ${v:,.0f}")

    results["3yr"]["final_equity"] = {
        f"p{p}": round(float(np.percentile(final_eq, p)), 0)
        for p in [5, 10, 25, 50, 75, 90, 95]
    }

    # === STRESS TEST: What if returns are 30% worse? ===
    log.info("\n" + "=" * 60)
    log.info("STRESS TEST: Returns 30% worse (conservative estimate)")
    log.info("=" * 60)

    stressed_returns = daily_returns * 0.70  # 30% haircut
    paths_stress = bootstrap_returns(stressed_returns, 252, N_SIMS)
    equity_stress = simulate_equity_paths(paths_stress)
    metrics_stress = compute_path_metrics(equity_stress)

    results["stress_1yr"] = {}
    results["stress_1yr"]["total_return"] = percentile_table(metrics_stress["total_return_pct"], "Total Return % (stressed)")
    results["stress_1yr"]["max_dd"] = percentile_table(metrics_stress["max_dd_pct"], "Max Drawdown % (stressed)")

    prob_positive_stress = (metrics_stress["total_return_pct"] > 0).mean() * 100
    prob_lose_10_stress = (metrics_stress["total_return_pct"] < -10).mean() * 100
    log.info(f"\n  Stressed probabilities (1-year, 30% haircut):")
    log.info(f"    P(positive year):    {prob_positive_stress:.1f}%")
    log.info(f"    P(lose >10%):        {prob_lose_10_stress:.1f}%")

    results["stress_1yr"]["probabilities"] = {
        "positive_year": round(prob_positive_stress, 1),
        "lose_10pct": round(prob_lose_10_stress, 1),
    }

    # === DRAWDOWN RECOVERY TIME ===
    log.info("\n" + "=" * 60)
    log.info("DRAWDOWN RECOVERY ANALYSIS")
    log.info("=" * 60)

    # How long does it take to recover from various drawdown levels?
    recovery_times = {5: [], 10: [], 15: [], 20: []}

    for sim in range(min(N_SIMS, 5000)):  # Use subset for speed
        eq = equity_1y[sim]
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak

        for dd_level in recovery_times.keys():
            threshold = -dd_level / 100
            in_dd = False
            dd_start = 0
            for day in range(len(dd)):
                if not in_dd and dd[day] <= threshold:
                    in_dd = True
                    dd_start = day
                elif in_dd and dd[day] >= 0:
                    recovery_times[dd_level].append(day - dd_start)
                    in_dd = False

    log.info(f"  Recovery times (trading days):")
    results["recovery"] = {}
    for dd_level, times in recovery_times.items():
        if times:
            med = np.median(times)
            p75 = np.percentile(times, 75)
            p95 = np.percentile(times, 95)
            log.info(f"    From {dd_level}% DD: median {med:.0f}d, p75 {p75:.0f}d, p95 {p95:.0f}d ({len(times)} events)")
            results["recovery"][f"dd_{dd_level}pct"] = {
                "median_days": round(float(med)),
                "p75_days": round(float(p75)),
                "p95_days": round(float(p95)),
                "n_events": len(times),
            }
        else:
            log.info(f"    From {dd_level}% DD: no events in simulations")

    # === SUMMARY ===
    log.info("\n" + "=" * 80)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 80)

    med_1y = np.median(metrics_1y["total_return_pct"])
    med_3y_eq = np.median(metrics_3y["final_equity"])

    log.info(f"  Strategy: Wheel (230 tickers, 30-delta, earnings filter, equity brake)")
    log.info(f"  Backtest: {actual_metrics['cagr_pct']}% CAGR, Sharpe {actual_metrics['sharpe']}, "
             f"MaxDD {actual_metrics['max_dd_pct']}%")
    log.info(f"")
    log.info(f"  1-Year Forward (median): +{med_1y:.1f}% return")
    log.info(f"    Realistic range (p10-p90): {np.percentile(metrics_1y['total_return_pct'], 10):.1f}% "
             f"to {np.percentile(metrics_1y['total_return_pct'], 90):.1f}%")
    log.info(f"    Probability of positive year: {prob_positive:.1f}%")
    log.info(f"    Probability of >20% drawdown: {prob_dd_20:.1f}%")
    log.info(f"")
    log.info(f"  3-Year Forward: $100K → ${med_3y_eq:,.0f} (median)")
    log.info(f"    Realistic range (p10-p90): ${np.percentile(final_eq, 10):,.0f} "
             f"to ${np.percentile(final_eq, 90):,.0f}")
    log.info(f"")
    log.info(f"  Even with 30% return haircut: {prob_positive_stress:.1f}% chance of positive year")

    # Save
    with open(OUT_DIR / "monte_carlo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary for SESSION_STATE
    summary = {
        "actual_cagr": actual_metrics["cagr_pct"],
        "actual_sharpe": actual_metrics["sharpe"],
        "actual_max_dd": actual_metrics["max_dd_pct"],
        "mc_1yr_median_return": round(med_1y, 1),
        "mc_1yr_p10_return": round(float(np.percentile(metrics_1y["total_return_pct"], 10)), 1),
        "mc_1yr_p90_return": round(float(np.percentile(metrics_1y["total_return_pct"], 90)), 1),
        "mc_1yr_prob_positive": round(prob_positive, 1),
        "mc_1yr_prob_dd_20": round(prob_dd_20, 1),
        "mc_3yr_median_equity": round(float(med_3y_eq), 0),
        "mc_3yr_p10_equity": round(float(np.percentile(final_eq, 10)), 0),
        "mc_3yr_p90_equity": round(float(np.percentile(final_eq, 90)), 0),
        "stress_prob_positive": round(prob_positive_stress, 1),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
