#!/usr/bin/env python3
"""
Two-Book Portfolio Optimizer: Income + Growth
==============================================
Monte Carlo simulation + analytical optimization for combining:
  - Income Book (options selling): CSP + IC Condors + ETF Rotation
  - Growth Book (momentum + trend): Risk Parity combo

Outputs allocation table with risk/return tradeoffs and optimal allocations.
"""

import numpy as np
import json
import os
from datetime import datetime

# ─── Configuration ───────────────────────────────────────────────────────────

TRADING_DAYS = 252
N_SIMULATIONS = 10_000
N_YEARS = 1  # simulate 1 year at a time, repeat for multi-year stats
RISK_FREE_RATE = 0.043  # ~4.3% risk-free (T-bills mid-2026)

# Income Book (combined 68/14/18 allocation)
INCOME = {
    "name": "Income Book",
    "cagr": 0.142,          # 14.2% CAGR
    "sharpe": 3.18,
    "max_dd": -0.017,       # -1.7%
}

# Growth Book (risk parity combo)
GROWTH = {
    "name": "Growth Book",
    "cagr": 0.16,           # 16% CAGR
    "sharpe": 1.52,
    "max_dd": -0.192,       # -19.2%
}

# Correlation between books
# Income = mostly market-neutral CSPs + condors; Growth = long equity momentum
# Low but not zero — both have some equity beta exposure
CORRELATION = 0.15

# Allocation splits to test
SPLITS = [
    (1.0, 0.0),   # 100/0
    (0.8, 0.2),   # 80/20
    (0.7, 0.3),   # 70/30
    (0.6, 0.4),   # 60/40
    (0.5, 0.5),   # 50/50
    (0.4, 0.6),   # 40/60
    (0.3, 0.7),   # 30/70
    (0.2, 0.8),   # 20/80
    (0.0, 1.0),   # 0/100
]


# ─── Derive daily parameters from backtest stats ────────────────────────────

def derive_daily_params(book: dict) -> tuple:
    """
    From CAGR and Sharpe, derive daily mean return and daily volatility.

    daily_mean ≈ (1 + CAGR)^(1/252) - 1
    daily_vol = (daily_mean - rf_daily) / Sharpe_daily
    where Sharpe_daily = Sharpe_annual / sqrt(252)
    """
    daily_mean = (1 + book["cagr"]) ** (1 / TRADING_DAYS) - 1
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    sharpe_daily = book["sharpe"] / np.sqrt(TRADING_DAYS)

    excess_daily = daily_mean - rf_daily
    if sharpe_daily > 0:
        daily_vol = excess_daily / sharpe_daily
    else:
        daily_vol = 0.01  # fallback

    return daily_mean, daily_vol


def build_correlation_matrix():
    """2x2 correlation matrix for Income and Growth books."""
    return np.array([
        [1.0,         CORRELATION],
        [CORRELATION,  1.0       ],
    ])


# ─── Monte Carlo Simulation ─────────────────────────────────────────────────

def run_monte_carlo(w_income: float, w_growth: float, n_sims: int = N_SIMULATIONS) -> dict:
    """
    Run Monte Carlo simulation for a given Income/Growth allocation.
    Returns statistics dictionary.
    """
    mu_i, vol_i = derive_daily_params(INCOME)
    mu_g, vol_g = derive_daily_params(GROWTH)

    # Portfolio daily mean and vol (2-asset analytical)
    mu_p = w_income * mu_i + w_growth * mu_g
    var_p = (w_income * vol_i) ** 2 + (w_growth * vol_g) ** 2 + \
            2 * w_income * w_growth * vol_i * vol_g * CORRELATION
    vol_p = np.sqrt(var_p)

    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1

    # Generate correlated daily returns using Cholesky
    # But since we already have portfolio-level params, simulate directly
    rng = np.random.default_rng(42)

    # Simulate daily log returns for the combined portfolio
    # Use normal distribution with portfolio mean and vol
    daily_returns = rng.normal(mu_p, vol_p, size=(n_sims, TRADING_DAYS))

    # Compute equity curves
    cum_returns = np.cumprod(1 + daily_returns, axis=1)

    # Final wealth (starting from $1)
    final_wealth = cum_returns[:, -1]

    # CAGR for each sim (1-year sim, so CAGR = total return)
    cagrs = final_wealth - 1.0

    # Annualized Sharpe for each sim
    annual_returns = np.mean(daily_returns, axis=1) * TRADING_DAYS
    annual_vols = np.std(daily_returns, axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    sharpes = (annual_returns - RISK_FREE_RATE) / np.where(annual_vols > 0, annual_vols, 1e-10)

    # Sortino for each sim
    downside_returns = np.where(daily_returns < 0, daily_returns, 0)
    downside_vol = np.sqrt(np.mean(downside_returns ** 2, axis=1)) * np.sqrt(TRADING_DAYS)
    sortinos = (annual_returns - RISK_FREE_RATE) / np.where(downside_vol > 0, downside_vol, 1e-10)

    # Max drawdown for each sim
    max_dds = np.zeros(n_sims)
    for i in range(n_sims):
        curve = cum_returns[i]
        running_max = np.maximum.accumulate(curve)
        drawdowns = (curve - running_max) / running_max
        max_dds[i] = np.min(drawdowns)

    # Calmar = CAGR / |MaxDD|
    calmars = cagrs / np.where(np.abs(max_dds) > 1e-10, np.abs(max_dds), 1e-10)

    def pct(arr, p):
        return float(np.percentile(arr, p))

    return {
        "allocation": f"{int(w_income*100)}/{int(w_growth*100)}",
        "w_income": w_income,
        "w_growth": w_growth,
        "analytical_daily_mean": float(mu_p),
        "analytical_daily_vol": float(vol_p),
        "analytical_annual_vol": float(vol_p * np.sqrt(TRADING_DAYS)),
        "cagr": {
            "mean": float(np.mean(cagrs)),
            "median": float(np.median(cagrs)),
            "p5": pct(cagrs, 5),
            "p95": pct(cagrs, 95),
        },
        "sharpe": {
            "mean": float(np.mean(sharpes)),
            "median": float(np.median(sharpes)),
            "p5": pct(sharpes, 5),
            "p95": pct(sharpes, 95),
        },
        "sortino": {
            "mean": float(np.mean(sortinos)),
            "median": float(np.median(sortinos)),
            "p5": pct(sortinos, 5),
            "p95": pct(sortinos, 95),
        },
        "max_dd": {
            "mean": float(np.mean(max_dds)),
            "median": float(np.median(max_dds)),
            "p5": pct(max_dds, 5),  # worst 5% of drawdowns
            "p95": pct(max_dds, 95),
        },
        "calmar": {
            "mean": float(np.mean(calmars)),
            "median": float(np.median(calmars)),
        },
        "prob_positive": float(np.mean(cagrs > 0)),
        "prob_beat_rf": float(np.mean(cagrs > RISK_FREE_RATE)),
    }


# ─── Analytical Optimizations ───────────────────────────────────────────────

def compute_optimal_allocations() -> dict:
    """
    Compute Kelly, mean-variance, min-variance, and equal risk contribution allocations.
    """
    mu_i, vol_i = derive_daily_params(INCOME)
    mu_g, vol_g = derive_daily_params(GROWTH)
    rf_daily = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1

    # Annualize
    mu_i_ann = mu_i * TRADING_DAYS
    mu_g_ann = mu_g * TRADING_DAYS
    vol_i_ann = vol_i * np.sqrt(TRADING_DAYS)
    vol_g_ann = vol_g * np.sqrt(TRADING_DAYS)

    cov = CORRELATION * vol_i_ann * vol_g_ann

    # ── Mean-Variance Optimal (max Sharpe) ──
    # For 2 assets: w* = Σ^{-1} (μ - rf) / 1'Σ^{-1}(μ - rf)
    cov_matrix = np.array([
        [vol_i_ann**2, cov],
        [cov, vol_g_ann**2],
    ])
    excess = np.array([mu_i_ann - RISK_FREE_RATE, mu_g_ann - RISK_FREE_RATE])

    inv_cov = np.linalg.inv(cov_matrix)
    raw_weights = inv_cov @ excess

    # Normalize to sum to 1 (long-only constraint: clip negatives)
    if np.any(raw_weights < 0):
        raw_weights = np.clip(raw_weights, 0, None)

    w_sum = np.sum(raw_weights)
    if w_sum > 0:
        mv_weights = raw_weights / w_sum
    else:
        mv_weights = np.array([0.5, 0.5])

    # ── Minimum Variance ──
    # w_minvar = Σ^{-1} 1 / (1' Σ^{-1} 1)
    ones = np.ones(2)
    raw_mv = inv_cov @ ones
    if np.any(raw_mv < 0):
        raw_mv = np.clip(raw_mv, 0, None)
    w_mv_sum = np.sum(raw_mv)
    if w_mv_sum > 0:
        minvar_weights = raw_mv / w_mv_sum
    else:
        minvar_weights = np.array([0.5, 0.5])

    # ── Kelly Criterion (2 assets) ──
    # f* = Σ^{-1} (μ - rf) — full Kelly, then we normalize for fractional
    kelly_raw = inv_cov @ excess
    kelly_full = kelly_raw  # These are leverage ratios
    # For our purposes, normalize to sum=1 (no leverage)
    if np.any(kelly_raw < 0):
        kelly_raw_clipped = np.clip(kelly_raw, 0, None)
    else:
        kelly_raw_clipped = kelly_raw
    k_sum = np.sum(kelly_raw_clipped)
    if k_sum > 0:
        kelly_weights = kelly_raw_clipped / k_sum
    else:
        kelly_weights = np.array([0.5, 0.5])

    # ── Equal Risk Contribution (Risk Parity) ──
    # Each asset contributes equally to portfolio variance
    # Iterative solution for 2 assets
    # RC_i = w_i * (Σw)_i / (w'Σw)
    # For 2 assets, solve: w1*sigma1^2 + w1*w2*cov*sigma2/sigma1 = w2*sigma2^2 + w1*w2*cov*sigma1/sigma2
    # Simpler: w_i ∝ 1/vol_i (approximate for low correlation)
    # More precise: Newton's method

    def risk_contributions(w):
        port_var = w[0]**2 * vol_i_ann**2 + w[1]**2 * vol_g_ann**2 + \
                   2 * w[0] * w[1] * cov
        port_vol = np.sqrt(port_var)
        marginal = np.array([
            w[0] * vol_i_ann**2 + w[1] * cov,
            w[1] * vol_g_ann**2 + w[0] * cov,
        ]) / port_vol
        rc = w * marginal
        return rc, port_vol

    # Iterative: start from inverse-vol
    w_erc = np.array([1/vol_i_ann, 1/vol_g_ann])
    w_erc = w_erc / np.sum(w_erc)

    for _ in range(1000):
        rc, _ = risk_contributions(w_erc)
        target = np.mean(rc)
        # Adjust weights: increase weight of under-contributing asset
        adjustment = target / rc
        w_erc = w_erc * adjustment
        w_erc = np.clip(w_erc, 0.01, 0.99)
        w_erc = w_erc / np.sum(w_erc)

    return {
        "mean_variance_optimal": {
            "income": float(mv_weights[0]),
            "growth": float(mv_weights[1]),
            "label": f"{int(mv_weights[0]*100)}/{int(mv_weights[1]*100)}",
        },
        "minimum_variance": {
            "income": float(minvar_weights[0]),
            "growth": float(minvar_weights[1]),
            "label": f"{int(minvar_weights[0]*100)}/{int(minvar_weights[1]*100)}",
        },
        "kelly_normalized": {
            "income": float(kelly_weights[0]),
            "growth": float(kelly_weights[1]),
            "label": f"{int(kelly_weights[0]*100)}/{int(kelly_weights[1]*100)}",
            "kelly_full_leverage": {
                "income": float(kelly_full[0]),
                "growth": float(kelly_full[1]),
                "total_leverage": float(np.sum(kelly_full)),
            },
        },
        "equal_risk_contribution": {
            "income": float(w_erc[0]),
            "growth": float(w_erc[1]),
            "label": f"{int(w_erc[0]*100)}/{int(w_erc[1]*100)}",
        },
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  TWO-BOOK PORTFOLIO OPTIMIZER: Income + Growth")
    print("=" * 80)
    print()

    # Show inputs
    mu_i, vol_i = derive_daily_params(INCOME)
    mu_g, vol_g = derive_daily_params(GROWTH)

    print("INPUT PARAMETERS:")
    print(f"  Income Book:  CAGR={INCOME['cagr']*100:.1f}%, Sharpe={INCOME['sharpe']:.2f}, "
          f"MaxDD={INCOME['max_dd']*100:.1f}%")
    print(f"    -> Daily mean={mu_i*10000:.2f}bps, Daily vol={vol_i*10000:.2f}bps, "
          f"Annual vol={vol_i*np.sqrt(252)*100:.2f}%")
    print(f"  Growth Book:  CAGR={GROWTH['cagr']*100:.1f}%, Sharpe={GROWTH['sharpe']:.2f}, "
          f"MaxDD={GROWTH['max_dd']*100:.1f}%")
    print(f"    -> Daily mean={mu_g*10000:.2f}bps, Daily vol={vol_g*10000:.2f}bps, "
          f"Annual vol={vol_g*np.sqrt(252)*100:.2f}%")
    print(f"  Cross-book correlation: {CORRELATION:.2f}")
    print(f"  Risk-free rate: {RISK_FREE_RATE*100:.1f}%")
    print(f"  Simulations: {N_SIMULATIONS:,} x {TRADING_DAYS} days")
    print()

    # Run Monte Carlo for each allocation
    print("RUNNING MONTE CARLO SIMULATIONS...")
    results = []
    for w_i, w_g in SPLITS:
        r = run_monte_carlo(w_i, w_g)
        results.append(r)
        print(f"  {r['allocation']:>6s} Income/Growth  |  "
              f"CAGR: {r['cagr']['mean']*100:+6.1f}%  |  "
              f"Sharpe: {r['sharpe']['mean']:5.2f}  |  "
              f"Sortino: {r['sortino']['mean']:5.2f}  |  "
              f"MaxDD: {r['max_dd']['mean']*100:6.1f}%  |  "
              f"P(>0): {r['prob_positive']*100:5.1f}%")

    print()

    # Optimal allocations
    print("OPTIMAL ALLOCATIONS:")
    optima = compute_optimal_allocations()

    for method, data in optima.items():
        label = method.replace("_", " ").title()
        print(f"  {label:30s}:  Income={data['income']*100:5.1f}%  Growth={data['growth']*100:5.1f}%")

    # Full Kelly leverage info
    kelly_lev = optima["kelly_normalized"]["kelly_full_leverage"]
    print(f"\n  Full Kelly leverage (unconstrained):")
    print(f"    Income: {kelly_lev['income']:.1f}x, Growth: {kelly_lev['growth']:.1f}x, "
          f"Total: {kelly_lev['total_leverage']:.1f}x")
    print(f"    (Use half-Kelly or less in practice)")

    print()

    # Detailed table
    print("=" * 120)
    print(f"{'Alloc':>8s} | {'CAGR':>8s} {'(p5)':>7s} {'(p95)':>7s} | "
          f"{'Sharpe':>7s} {'(p5)':>7s} | {'Sortino':>8s} {'(p5)':>7s} | "
          f"{'MaxDD':>7s} {'(p5)':>7s} | {'Calmar':>7s} | {'P(>0)':>6s} {'P(>RF)':>6s}")
    print("-" * 120)

    for r in results:
        print(f"{r['allocation']:>8s} | "
              f"{r['cagr']['mean']*100:+7.1f}% {r['cagr']['p5']*100:+6.1f}% {r['cagr']['p95']*100:+6.1f}% | "
              f"{r['sharpe']['mean']:7.2f} {r['sharpe']['p5']:6.2f} | "
              f"{r['sortino']['mean']:8.2f} {r['sortino']['p5']:6.2f} | "
              f"{r['max_dd']['mean']*100:6.1f}% {r['max_dd']['p5']*100:6.1f}% | "
              f"{r['calmar']['mean']:7.1f} | "
              f"{r['prob_positive']*100:5.1f}% {r['prob_beat_rf']*100:5.1f}%")

    print("=" * 120)
    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 60)

    # Find best Sharpe allocation
    best_sharpe = max(results, key=lambda r: r["sharpe"]["mean"])
    best_sortino = max(results, key=lambda r: r["sortino"]["mean"])
    best_calmar = max(results, key=lambda r: r["calmar"]["mean"])
    best_cagr = max(results, key=lambda r: r["cagr"]["mean"])

    print(f"  Best Sharpe:  {best_sharpe['allocation']} (Sharpe={best_sharpe['sharpe']['mean']:.2f})")
    print(f"  Best Sortino: {best_sortino['allocation']} (Sortino={best_sortino['sortino']['mean']:.2f})")
    print(f"  Best Calmar:  {best_calmar['allocation']} (Calmar={best_calmar['calmar']['mean']:.1f})")
    print(f"  Best CAGR:    {best_cagr['allocation']} (CAGR={best_cagr['cagr']['mean']*100:+.1f}%)")
    print()

    print("  SITUATION-SPECIFIC:")
    print(f"    Small account ($441 Agentic):  0/100 Growth — need maximum growth,")
    print(f"      drawdown tolerance is high (small absolute $). CAGR={results[-1]['cagr']['mean']*100:+.1f}%")
    print()
    print(f"    Medium account ($5K-$50K):  {best_sharpe['allocation']} — maximize risk-adjusted")
    print(f"      returns. Sharpe={best_sharpe['sharpe']['mean']:.2f}, MaxDD={best_sharpe['max_dd']['mean']*100:.1f}%")
    print()

    # For large account, find allocation with best Sharpe that keeps MaxDD < -5%
    safe = [r for r in results if r["max_dd"]["p5"] > -0.10]  # 5th percentile DD > -10%
    if safe:
        best_safe = max(safe, key=lambda r: r["sharpe"]["mean"])
        print(f"    Large account ($100K+):  {best_safe['allocation']} — prioritize capital")
        print(f"      preservation. Sharpe={best_safe['sharpe']['mean']:.2f}, "
              f"worst-case MaxDD={best_safe['max_dd']['p5']*100:.1f}%")
    else:
        print(f"    Large account ($100K+):  100/0 Income — maximum capital preservation")

    print()
    print(f"    MV-Optimal:  {optima['mean_variance_optimal']['label']} Income/Growth")
    print(f"    Risk Parity: {optima['equal_risk_contribution']['label']} Income/Growth")
    print()

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "income_book": INCOME,
            "growth_book": GROWTH,
            "correlation": CORRELATION,
            "risk_free_rate": RISK_FREE_RATE,
            "n_simulations": N_SIMULATIONS,
            "trading_days": TRADING_DAYS,
        },
        "derived_params": {
            "income_daily_mean_bps": float(mu_i * 10000),
            "income_daily_vol_bps": float(vol_i * 10000),
            "income_annual_vol_pct": float(vol_i * np.sqrt(252) * 100),
            "growth_daily_mean_bps": float(mu_g * 10000),
            "growth_daily_vol_bps": float(vol_g * 10000),
            "growth_annual_vol_pct": float(vol_g * np.sqrt(252) * 100),
        },
        "monte_carlo_results": results,
        "optimal_allocations": optima,
        "recommendations": {
            "small_account": "0/100 Growth — maximize growth, tolerate drawdowns",
            "medium_account": f"{best_sharpe['allocation']} — best risk-adjusted returns",
            "large_account": f"{best_safe['allocation'] if safe else '100/0'} — capital preservation priority",
            "mv_optimal": optima["mean_variance_optimal"]["label"],
            "risk_parity": optima["equal_risk_contribution"]["label"],
        },
    }

    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"two_book_optimizer_{datetime.now().strftime('%Y%m%d')}.json")

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {outpath}")
    print()


if __name__ == "__main__":
    main()
