#!/usr/bin/env python3
"""
champion_montecarlo_risk.py — Monte Carlo Risk Analysis for Champion Config
==========================================================================
Uses the 2022-trade dataset from champion_extended_validation to compute:
1. Bootstrap confidence intervals for Sharpe, PF, WR
2. Monte Carlo drawdown distribution (10,000 simulations)
3. Probability of ruin at various account sizes
4. Kelly fraction and optimal position sizing
5. Expected annual P&L at different contract sizes

Usage:
    python alpha_discovery/champion_montecarlo_risk.py
"""

import json, logging, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
ROOT = Path("/home/nick/Lvl3Quant")
INPUT_DIR = ROOT / "output" / "champion_extended_validation"
OUTPUT_DIR = ROOT / "output" / "champion_montecarlo_risk"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s [MC-RISK] %(message)s", level=logging.INFO,
                    handlers=[logging.StreamHandler(), 
                              logging.FileHandler(str(ROOT / "logs" / "champion_montecarlo_risk.log"))])
log = logging.getLogger()

ES_TICK_VALUE = 12.50
ES_POINT_VALUE = 50.0
COMMISSION_RT = 4.70  # dollars per round trip

N_BOOTSTRAP = 10000
N_MC_PATHS = 10000
RNG = np.random.default_rng(42)


def bootstrap_metric(pnls: np.ndarray, metric_fn, n_boot=N_BOOTSTRAP):
    """Bootstrap confidence interval for a metric."""
    estimates = np.empty(n_boot)
    n = len(pnls)
    for i in range(n_boot):
        sample = RNG.choice(pnls, size=n, replace=True)
        estimates[i] = metric_fn(sample)
    return {
        "mean": float(np.mean(estimates)),
        "median": float(np.median(estimates)),
        "std": float(np.std(estimates)),
        "ci_2_5": float(np.percentile(estimates, 2.5)),
        "ci_97_5": float(np.percentile(estimates, 97.5)),
        "ci_5": float(np.percentile(estimates, 5)),
        "ci_95": float(np.percentile(estimates, 95)),
        "p_positive": float((estimates > 0).mean()),
    }


def compute_sharpe(pnls):
    return pnls.mean() / (pnls.std() + 1e-8)

def compute_sortino(pnls):
    down = pnls[pnls < 0]
    down_std = np.sqrt((down**2).mean()) if len(down) > 0 else 1e-8
    return pnls.mean() / (down_std + 1e-8)

def compute_pf(pnls):
    wins = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    return wins / (losses + 1e-8)

def compute_wr(pnls):
    return (pnls > 0).mean()

def compute_max_dd(pnls):
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return dd.max()


def monte_carlo_paths(trade_pnls: np.ndarray, n_paths=N_MC_PATHS):
    """Generate MC paths by shuffling trade order."""
    n_trades = len(trade_pnls)
    max_dds = np.empty(n_paths)
    final_pnls = np.empty(n_paths)
    max_losing_streaks = np.empty(n_paths)
    
    for i in range(n_paths):
        path = RNG.choice(trade_pnls, size=n_trades, replace=True)
        cum = np.cumsum(path)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dds[i] = dd.max()
        final_pnls[i] = cum[-1]
        
        # Max losing streak
        streak = 0
        max_streak = 0
        for p in path:
            if p < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        max_losing_streaks[i] = max_streak
    
    return max_dds, final_pnls, max_losing_streaks


def probability_of_ruin(trade_pnls: np.ndarray, account_sizes_ticks: list, n_paths=N_MC_PATHS):
    """Compute probability of hitting various ruin levels."""
    n_trades = len(trade_pnls)
    results = {}
    
    for account_size in account_sizes_ticks:
        ruin_count = 0
        for i in range(n_paths):
            path = RNG.choice(trade_pnls, size=n_trades, replace=True)
            cum = np.cumsum(path)
            if cum.min() < -account_size:
                ruin_count += 1
        results[account_size] = {
            "account_ticks": account_size,
            "account_dollars": account_size * ES_TICK_VALUE,
            "prob_ruin": ruin_count / n_paths,
            "prob_survive": 1 - ruin_count / n_paths,
        }
    
    return results


def kelly_fraction(trade_pnls: np.ndarray):
    """Compute Kelly criterion."""
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return {"kelly": 0, "half_kelly": 0}
    
    wr = len(wins) / len(trade_pnls)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    # Kelly: f = (bp - q) / b where b = avg_win/avg_loss, p = WR, q = 1-WR
    b = avg_win / avg_loss
    kelly = (b * wr - (1 - wr)) / b
    
    return {
        "win_rate": float(wr),
        "avg_win_ticks": float(avg_win),
        "avg_loss_ticks": float(avg_loss),
        "rr_ratio": float(b),
        "kelly_fraction": float(kelly),
        "half_kelly": float(kelly / 2),
        "quarter_kelly": float(kelly / 4),
    }


def annual_projections(trade_pnls: np.ndarray, n_days=137, contracts_list=[1, 2, 3, 5, 10]):
    """Project annual P&L at various contract sizes."""
    daily_pnl = {}
    
    trades_per_day = len(trade_pnls) / n_days
    avg_daily_pnl_ticks = trade_pnls.sum() / n_days
    
    projections = {}
    for n_contracts in contracts_list:
        annual_ticks = avg_daily_pnl_ticks * 252
        annual_dollars = annual_ticks * ES_TICK_VALUE * n_contracts
        annual_commission = trades_per_day * 252 * COMMISSION_RT * n_contracts
        net_annual = annual_dollars - annual_commission
        
        projections[n_contracts] = {
            "contracts": n_contracts,
            "avg_daily_ticks": float(avg_daily_pnl_ticks),
            "avg_daily_dollars": float(avg_daily_pnl_ticks * ES_TICK_VALUE * n_contracts),
            "annual_gross_dollars": float(annual_dollars),
            "annual_commission": float(annual_commission),
            "annual_net_dollars": float(net_annual),
            "daily_commission": float(trades_per_day * COMMISSION_RT * n_contracts),
        }
    
    return projections


def main():
    log.info("=" * 70)
    log.info("CHAMPION CONFIG — MONTE CARLO RISK ANALYSIS")
    log.info("=" * 70)
    
    # Load trades
    trades_file = INPUT_DIR / "trades_champion.csv"
    if not trades_file.exists():
        log.error(f"No trades file at {trades_file}")
        sys.exit(1)
    
    trades = pd.read_csv(trades_file)
    pnls = trades["net_pnl_ticks"].values
    log.info(f"Loaded {len(pnls)} trades from {trades['date'].nunique()} days")
    log.info(f"Total P&L: {pnls.sum():+.1f} ticks (${pnls.sum()*ES_TICK_VALUE:+,.0f})")
    
    results = {"timestamp": pd.Timestamp.now().isoformat(), "n_trades": len(pnls)}
    
    # 1. Bootstrap confidence intervals
    log.info("\n1. BOOTSTRAP CONFIDENCE INTERVALS (10,000 iterations)")
    metrics = {
        "per_trade_sharpe": bootstrap_metric(pnls, compute_sharpe),
        "profit_factor": bootstrap_metric(pnls, compute_pf),
        "win_rate": bootstrap_metric(pnls, compute_wr),
        "max_drawdown_ticks": bootstrap_metric(pnls, compute_max_dd),
    }
    
    for name, m in metrics.items():
        log.info(f"  {name:25s}: {m['mean']:8.3f}  95% CI [{m['ci_2_5']:8.3f}, {m['ci_97_5']:8.3f}]")
    
    results["bootstrap"] = metrics
    
    # 2. Monte Carlo drawdown distribution
    log.info("\n2. MONTE CARLO DRAWDOWN DISTRIBUTION (10,000 paths)")
    max_dds, final_pnls, max_streaks = monte_carlo_paths(pnls)
    
    dd_stats = {
        "median": float(np.median(max_dds)),
        "p75": float(np.percentile(max_dds, 75)),
        "p90": float(np.percentile(max_dds, 90)),
        "p95": float(np.percentile(max_dds, 95)),
        "p99": float(np.percentile(max_dds, 99)),
        "max": float(max_dds.max()),
        "median_dollars": float(np.median(max_dds) * ES_TICK_VALUE),
        "p95_dollars": float(np.percentile(max_dds, 95) * ES_TICK_VALUE),
        "p99_dollars": float(np.percentile(max_dds, 99) * ES_TICK_VALUE),
    }
    
    log.info(f"  Median max DD: {dd_stats['median']:.1f} ticks (${dd_stats['median_dollars']:,.0f})")
    log.info(f"  95th percentile: {dd_stats['p95']:.1f} ticks (${dd_stats['p95_dollars']:,.0f})")
    log.info(f"  99th percentile: {dd_stats['p99']:.1f} ticks (${dd_stats['p99_dollars']:,.0f})")
    log.info(f"  Worst case: {dd_stats['max']:.1f} ticks (${dd_stats['max']*ES_TICK_VALUE:,.0f})")
    
    streak_stats = {
        "median": float(np.median(max_streaks)),
        "p90": float(np.percentile(max_streaks, 90)),
        "p95": float(np.percentile(max_streaks, 95)),
        "p99": float(np.percentile(max_streaks, 99)),
    }
    log.info(f"\n  Max losing streak:")
    log.info(f"    Median: {streak_stats['median']:.0f}, P90: {streak_stats['p90']:.0f}, "
             f"P95: {streak_stats['p95']:.0f}, P99: {streak_stats['p99']:.0f}")
    
    results["monte_carlo"] = {"drawdown": dd_stats, "losing_streak": streak_stats}
    
    # 3. Probability of ruin
    log.info("\n3. PROBABILITY OF RUIN")
    account_sizes = [50, 100, 200, 500, 1000, 2000, 5000]  # in ticks
    ruin = probability_of_ruin(pnls, account_sizes)
    
    for sz, r in ruin.items():
        log.info(f"  Account ${r['account_dollars']:>8,.0f} ({sz:5d} ticks): "
                 f"P(ruin)={r['prob_ruin']:.4f} ({100*r['prob_ruin']:.2f}%)")
    
    results["ruin_probability"] = ruin
    
    # 4. Kelly fraction
    log.info("\n4. KELLY CRITERION")
    kelly = kelly_fraction(pnls)
    log.info(f"  Win rate: {kelly['win_rate']:.1%}")
    log.info(f"  Avg win: {kelly['avg_win_ticks']:.2f} ticks, Avg loss: {kelly['avg_loss_ticks']:.2f} ticks")
    log.info(f"  R:R ratio: {kelly['rr_ratio']:.2f}")
    log.info(f"  Full Kelly: {kelly['kelly_fraction']:.1%}")
    log.info(f"  Half Kelly: {kelly['half_kelly']:.1%}")
    log.info(f"  Quarter Kelly: {kelly['quarter_kelly']:.1%}")
    
    results["kelly"] = kelly
    
    # 5. Annual projections
    log.info("\n5. ANNUAL PROJECTIONS (based on 137 OOT days → 252 trading days)")
    projections = annual_projections(pnls, n_days=trades['date'].nunique())
    
    for n, p in projections.items():
        log.info(f"  {n:2d} contract(s): Net ${p['annual_net_dollars']:>12,.0f}/yr "
                 f"(gross ${p['annual_gross_dollars']:>12,.0f} - commission ${p['annual_commission']:>8,.0f})")
    
    results["annual_projections"] = projections
    
    # 6. Edge decay analysis
    log.info("\n6. EDGE DECAY (rolling 30-trade Sharpe)")
    window = 30
    rolling_sharpe = []
    for i in range(0, len(pnls) - window, window // 2):
        chunk = pnls[i:i+window]
        s = chunk.mean() / (chunk.std() + 1e-8)
        rolling_sharpe.append({"start_trade": i, "sharpe": float(s)})
    
    if rolling_sharpe:
        sharpes = [x["sharpe"] for x in rolling_sharpe]
        log.info(f"  Rolling Sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]")
        log.info(f"  First 5 windows avg: {np.mean(sharpes[:5]):.3f}")
        log.info(f"  Last 5 windows avg: {np.mean(sharpes[-5:]):.3f}")
        from scipy import stats as scipy_stats
        slope, _, r_value, p_value, _ = scipy_stats.linregress(range(len(sharpes)), sharpes)
        log.info(f"  Trend: slope={slope:.5f}, R²={r_value**2:.3f}, p={p_value:.4f}")
        decay_detected = p_value < 0.05 and slope < 0
        log.info(f"  Decay detected: {'YES ⚠️' if decay_detected else 'NO ✓'}")
    
    results["edge_decay"] = rolling_sharpe
    
    # Save results
    with open(OUTPUT_DIR / "risk_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"\nResults saved to {OUTPUT_DIR / 'risk_analysis.json'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
