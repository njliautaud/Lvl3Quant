#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Optimizer
===================================
Combines all validated growth strategies into optimal portfolio allocations.

Strategies:
  1. ML CTA Trend Following   — Sharpe 2.97, CAGR 20.2%, MaxDD -4.9%
  2. ML Sector Rotation        — Sharpe 2.47, CAGR 20.7%, MaxDD -7.4%
  3. ML Vol Breakout (Straddles)— Sharpe 1.11, CAGR ~15%, MaxDD -10.2%
  4. ML Credit Spread Timing   — Sharpe 1.19, CAGR  8.0%, MaxDD -16.8%
  5. Stat Arb Pairs (z-score)  — Sharpe 0.81, CAGR  8.1%, MaxDD -11.5%

Capital: $100K fixed (no DCA).
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = BASE / "output" / "multi_strategy_portfolio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPITAL = 100_000
TRADING_DAYS = 252
N_YEARS = 16.9  # common backtest span

# ── Strategy metadata (from validated results.json files) ──────────────
STRATEGIES = {
    "ML_CTA_Trend": {
        "sharpe": 2.968, "cagr": 0.202, "max_dd": -0.049,
        "annual_vol": 0.063, "sortino": 4.336, "win_rate": 0.561,
        "profit_factor": 1.674, "spy_corr": 0.15,  # momentum-based, some SPY beta
    },
    "ML_Sector_Rotation": {
        "sharpe": 2.472, "cagr": 0.207, "max_dd": -0.074,
        "annual_vol": 0.077, "sortino": 3.368, "win_rate": 0.542,
        "profit_factor": 1.55, "spy_corr": 0.35,  # sector ETFs have equity beta
    },
    "ML_Vol_Breakout": {
        "sharpe": 1.111, "cagr": 0.15, "max_dd": -0.102,
        "annual_vol": 0.135, "sortino": 2.453, "win_rate": 0.490,
        "profit_factor": 2.095, "spy_corr": 0.10,  # options straddles, low equity beta
    },
    "ML_Credit_Timing": {
        "sharpe": 1.188, "cagr": 0.08, "max_dd": -0.168,
        "annual_vol": 0.067, "sortino": 1.319, "win_rate": 0.732,
        "profit_factor": 2.645, "spy_corr": 0.25,  # credit spreads have some equity beta
    },
    "Stat_Arb_Pairs": {
        "sharpe": 0.81, "cagr": 0.081, "max_dd": -0.115,
        "annual_vol": 0.10, "sortino": 1.15, "win_rate": 0.55,
        "profit_factor": 1.3, "spy_corr": 0.043,  # market-neutral by construction
    },
}

NAMES = list(STRATEGIES.keys())
N = len(NAMES)


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: Try to load real daily returns; fall back to synthetic
# ══════════════════════════════════════════════════════════════════════

def load_vol_breakout_daily_returns() -> pd.Series | None:
    """Load daily returns from vol breakout equity curve."""
    path = BASE / "output" / "ml_vol_breakout" / "equity_curve.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    # equity curve has multiple entries per date (per-trade); take last per day
    daily = df.groupby("date")["capital"].last()
    daily.index = pd.to_datetime(daily.index)
    rets = daily.pct_change().dropna()
    rets.name = "ML_Vol_Breakout"
    return rets


def build_correlation_matrix():
    """
    Build realistic cross-strategy correlation matrix.
    Based on strategy types and SPY correlation estimates.
    """
    # Pairwise correlations estimated from strategy characteristics:
    #   - CTA & Sector share some trend signals but different universes
    #   - Vol Breakout is options-based, low correlation to trend
    #   - Credit Timing captures spread moves, moderate equity correlation
    #   - Stat Arb is market-neutral, lowest correlation to everything
    corr = np.array([
        # CTA    Sec    Vol    Cred   StatArb
        [1.00,  0.51,  0.08,  0.12,  0.05],  # CTA (0.51 from results.json)
        [0.51,  1.00,  0.10,  0.20,  0.08],  # Sector
        [0.08,  0.10,  1.00,  0.05,  0.03],  # Vol Breakout
        [0.12,  0.20,  0.05,  1.00,  0.10],  # Credit
        [0.05,  0.08,  0.03,  0.10,  1.00],  # Stat Arb
    ])
    return pd.DataFrame(corr, index=NAMES, columns=NAMES)


def generate_synthetic_returns(n_days: int = None) -> pd.DataFrame:
    """
    Generate correlated daily return series matching each strategy's
    known Sharpe / vol / correlation properties using multivariate normal.
    """
    if n_days is None:
        n_days = int(N_YEARS * TRADING_DAYS)

    corr = build_correlation_matrix().values
    vols = np.array([STRATEGIES[n]["annual_vol"] for n in NAMES])
    means = np.array([STRATEGIES[n]["cagr"] for n in NAMES])

    # Daily parameters
    daily_vols = vols / np.sqrt(TRADING_DAYS)
    daily_means = means / TRADING_DAYS

    # Covariance from correlation + vols
    cov = np.outer(daily_vols, daily_vols) * corr

    np.random.seed(42)
    raw = np.random.multivariate_normal(daily_means, cov, size=n_days)

    dates = pd.bdate_range(start="2009-06-01", periods=n_days)
    df = pd.DataFrame(raw, index=dates, columns=NAMES)
    return df


def load_or_generate_returns() -> pd.DataFrame:
    """Load real returns where available, fill rest with synthetic."""
    n_days = int(N_YEARS * TRADING_DAYS)
    synthetic = generate_synthetic_returns(n_days)

    # Try loading vol breakout real returns
    vb_rets = load_vol_breakout_daily_returns()
    if vb_rets is not None and len(vb_rets) > 100:
        print(f"  Loaded {len(vb_rets)} real daily returns for ML_Vol_Breakout")
        # Align to synthetic dates for consistency
        common = synthetic.index.intersection(vb_rets.index)
        if len(common) > 100:
            synthetic.loc[common, "ML_Vol_Breakout"] = vb_rets.loc[common].values

    return synthetic


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: Portfolio optimization methods
# ══════════════════════════════════════════════════════════════════════

def portfolio_stats(w, mu, cov):
    """Annualized return, vol, sharpe for weight vector w."""
    ret = np.dot(w, mu) * TRADING_DAYS
    vol = np.sqrt(np.dot(w, cov @ w) * TRADING_DAYS)
    sharpe = ret / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def max_sharpe_weights(mu, cov):
    """Mean-variance max Sharpe (long-only, fully invested)."""
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, mu, cov)
        return -s

    w0 = np.ones(n) / n
    res = optimize.minimize(neg_sharpe, w0, method="SLSQP",
                            bounds=bounds, constraints=constraints,
                            options={"maxiter": 1000})
    return res.x if res.success else w0


def min_variance_weights(cov):
    """Global minimum variance (long-only, fully invested)."""
    n = cov.shape[0]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    def port_var(w):
        return np.dot(w, cov @ w)

    w0 = np.ones(n) / n
    res = optimize.minimize(port_var, w0, method="SLSQP",
                            bounds=bounds, constraints=constraints,
                            options={"maxiter": 1000})
    return res.x if res.success else w0


def risk_parity_weights(cov):
    """
    Risk parity: each strategy contributes equally to portfolio risk.
    Uses inverse-vol as starting point, then optimizes for equal risk contribution.
    """
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    # Inverse vol as initial guess
    w0 = (1.0 / vols)
    w0 /= w0.sum()

    def risk_budget_obj(w):
        w = np.abs(w)
        port_vol = np.sqrt(np.dot(w, cov @ w))
        if port_vol < 1e-12:
            return 1e6
        # Marginal risk contribution
        mrc = (cov @ w) / port_vol
        rc = w * mrc
        # Target: all risk contributions equal
        target_rc = port_vol / n
        return np.sum((rc - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.01, 1.0)] * n
    res = optimize.minimize(risk_budget_obj, w0, method="SLSQP",
                            bounds=bounds, constraints=constraints,
                            options={"maxiter": 2000})
    w = np.abs(res.x)
    w /= w.sum()
    return w


def hrp_weights(returns: pd.DataFrame):
    """
    Hierarchical Risk Parity (Lopez de Prado).
    Simplified implementation using correlation-based clustering.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    corr = returns.corr()
    cov = returns.cov()
    n = len(corr)

    # Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    link = linkage(condensed, method="single")
    order = leaves_list(link)

    # Recursive bisection
    def _get_cluster_var(cov_mat, cluster_items):
        cov_slice = cov_mat.iloc[cluster_items, cluster_items]
        ivp = 1.0 / np.diag(cov_slice)
        ivp /= ivp.sum()
        return np.dot(ivp, np.dot(cov_slice.values, ivp))

    def _recursive_bisection(cov_mat, sorted_idx):
        w = pd.Series(1.0, index=sorted_idx)
        cluster_items = [sorted_idx]

        while len(cluster_items) > 0:
            # bisect each cluster
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = _get_cluster_var(cov_mat, left)
                var_right = _get_cluster_var(cov_mat, right)
                alpha = 1 - var_left / (var_left + var_right)

                w[left] *= alpha
                w[right] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            cluster_items = new_clusters

        return w

    sorted_idx = list(order)
    w = _recursive_bisection(cov, sorted_idx)
    # Map back to original column order
    result = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        result[idx] = w.iloc[i]
    result /= result.sum()
    return result


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: Portfolio analytics
# ══════════════════════════════════════════════════════════════════════

def compute_portfolio_metrics(daily_rets: pd.Series, name: str) -> dict:
    """Compute full performance metrics for a daily return series."""
    cum = (1 + daily_rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(daily_rets) / TRADING_DAYS

    cagr = (cum.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = daily_rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = cagr / annual_vol if annual_vol > 1e-10 else 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 1e-10
    sortino = cagr / downside_vol if downside_vol > 1e-10 else 0

    # Max drawdown
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # Win rate
    win_rate = (daily_rets > 0).mean()

    # Profit factor
    gross_profit = daily_rets[daily_rets > 0].sum()
    gross_loss = abs(daily_rets[daily_rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")

    # Skewness & kurtosis
    skew = daily_rets.skew()
    kurt = daily_rets.kurtosis()

    return {
        "name": name,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "annual_vol_pct": round(annual_vol * 100, 2),
        "total_return_pct": round(total_ret * 100, 1),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "final_capital": round(CAPITAL * cum.iloc[-1], 0),
    }


def marginal_risk_contribution(w, cov):
    """Compute each strategy's % contribution to total portfolio risk."""
    port_vol = np.sqrt(np.dot(w, cov @ w))
    if port_vol < 1e-12:
        return np.zeros(len(w))
    mrc = (cov @ w) / port_vol
    rc = w * mrc
    rc_pct = rc / port_vol * 100
    return rc_pct


def leave_one_out_analysis(returns: pd.DataFrame, weights: np.ndarray) -> dict:
    """Test what happens if we remove each strategy one at a time."""
    base_rets = returns @ weights
    base_metrics = compute_portfolio_metrics(base_rets, "Full Portfolio")

    results = {}
    for i, name in enumerate(NAMES):
        # Remove strategy i, renormalize remaining weights
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        w_reduced = weights[mask].copy()
        w_reduced /= w_reduced.sum()

        reduced_rets = returns.iloc[:, mask] @ w_reduced
        m = compute_portfolio_metrics(reduced_rets, f"Without {name}")
        m["sharpe_change"] = round(m["sharpe"] - base_metrics["sharpe"], 3)
        m["dd_change_pct"] = round(m["max_dd_pct"] - base_metrics["max_dd_pct"], 2)
        results[name] = m

    return results


# ══════════════════════════════════════════════════════════════════════
#  STEP 4: Main execution
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  MULTI-STRATEGY PORTFOLIO OPTIMIZER")
    print(f"  Capital: ${CAPITAL:,.0f} | Strategies: {N}")
    print("=" * 70)

    # ── Load / generate returns ────────────────────────────────────
    print("\n[1] Loading daily returns...")
    returns = load_or_generate_returns()
    print(f"  Generated {len(returns)} trading days of returns ({len(returns)/TRADING_DAYS:.1f} years)")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # ── Correlation matrix ─────────────────────────────────────────
    print("\n[2] Correlation matrix (realized from return series):")
    corr = returns.corr()
    print(corr.round(3).to_string())

    # Individual strategy stats
    print("\n[3] Individual strategy performance (from simulated returns):")
    print(f"  {'Strategy':<22} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'Sortino':>8} {'Vol%':>7}")
    print("  " + "-" * 60)
    indiv_metrics = {}
    for name in NAMES:
        m = compute_portfolio_metrics(returns[name], name)
        indiv_metrics[name] = m
        print(f"  {name:<22} {m['sharpe']:>7.2f} {m['cagr_pct']:>7.1f} {m['max_dd_pct']:>8.1f} "
              f"{m['sortino']:>8.2f} {m['annual_vol_pct']:>7.1f}")

    # ── Covariance ─────────────────────────────────────────────────
    mu = returns.mean().values
    cov = returns.cov().values

    # ── Optimization methods ───────────────────────────────────────
    print("\n[4] Portfolio optimization results:")
    print("=" * 70)

    methods = {}

    # Equal weight
    w_eq = np.ones(N) / N
    methods["Equal_Weight"] = w_eq

    # Min variance
    w_mv = min_variance_weights(cov)
    methods["Min_Variance"] = w_mv

    # Max Sharpe
    w_ms = max_sharpe_weights(mu, cov)
    methods["Max_Sharpe"] = w_ms

    # Risk parity
    w_rp = risk_parity_weights(cov)
    methods["Risk_Parity"] = w_rp

    # HRP
    w_hrp = hrp_weights(returns)
    methods["HRP"] = w_hrp

    # Display weights
    print(f"\n  {'Method':<16}", end="")
    for name in NAMES:
        short = name.replace("ML_", "").replace("Stat_Arb_", "SA_")[:10]
        print(f" {short:>10}", end="")
    print()
    print("  " + "-" * (16 + 11 * N))

    for method, w in methods.items():
        print(f"  {method:<16}", end="")
        for wi in w:
            print(f" {wi:>10.1%}", end="")
        print()

    # ── Portfolio metrics for each method ──────────────────────────
    print(f"\n  {'Method':<16} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'Sortino':>8} {'Calmar':>7} {'Vol%':>7} {'Final$':>12}")
    print("  " + "-" * 80)

    portfolio_results = {}
    for method, w in methods.items():
        port_rets = returns @ w
        m = compute_portfolio_metrics(port_rets, method)
        portfolio_results[method] = {
            "weights": {name: round(float(w[i]), 4) for i, name in enumerate(NAMES)},
            "metrics": m,
        }
        print(f"  {method:<16} {m['sharpe']:>7.2f} {m['cagr_pct']:>7.1f} {m['max_dd_pct']:>8.1f} "
              f"{m['sortino']:>8.2f} {m['calmar']:>7.2f} {m['annual_vol_pct']:>7.1f} "
              f"${m['final_capital']:>11,.0f}")

    # Benchmark: SPY
    spy_m = {"sharpe": 0.887, "cagr_pct": 14.7, "max_dd_pct": -33.7,
             "sortino": 1.096, "calmar": 0.435, "annual_vol_pct": 17.1,
             "final_capital": CAPITAL * (1 + 9.065)}
    print(f"  {'SPY B&H':<16} {spy_m['sharpe']:>7.2f} {spy_m['cagr_pct']:>7.1f} {spy_m['max_dd_pct']:>8.1f} "
          f"{spy_m['sortino']:>8.2f} {spy_m['calmar']:>7.2f} {spy_m['annual_vol_pct']:>7.1f} "
          f"${spy_m['final_capital']:>11,.0f}")

    # ── Marginal risk contribution (for Max Sharpe) ────────────────
    print("\n[5] Marginal risk contribution (Max Sharpe portfolio):")
    mrc = marginal_risk_contribution(w_ms, cov)
    for i, name in enumerate(NAMES):
        print(f"  {name:<22}: {mrc[i]:>6.1f}% of portfolio risk (weight: {w_ms[i]:.1%})")

    # ── Leave-one-out analysis ─────────────────────────────────────
    print("\n[6] Leave-one-out analysis (removing each strategy from Max Sharpe portfolio):")
    loo = leave_one_out_analysis(returns, w_ms)
    print(f"  {'Removed':<22} {'New Sharpe':>10} {'Change':>8} {'New MaxDD%':>10} {'DD Change':>10}")
    print("  " + "-" * 65)
    for name, m in loo.items():
        print(f"  {name:<22} {m['sharpe']:>10.2f} {m['sharpe_change']:>+8.3f} "
              f"{m['max_dd_pct']:>10.1f} {m['dd_change_pct']:>+10.1f}")

    # ── Diversification ratio ─────────────────────────────────────
    print("\n[7] Diversification metrics:")
    for method, w in methods.items():
        indiv_vols = np.sqrt(np.diag(cov)) * np.sqrt(TRADING_DAYS)
        weighted_avg_vol = np.dot(w, indiv_vols)
        port_vol = np.sqrt(np.dot(w, cov @ w) * TRADING_DAYS)
        div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 1
        print(f"  {method:<16}: Diversification ratio = {div_ratio:.3f} "
              f"(1.0 = no benefit, higher = better)")

    # ── Best portfolio recommendation ─────────────────────────────
    best_method = max(portfolio_results, key=lambda k: portfolio_results[k]["metrics"]["sharpe"])
    best = portfolio_results[best_method]
    print("\n" + "=" * 70)
    print(f"  RECOMMENDED PORTFOLIO: {best_method}")
    print("=" * 70)
    print(f"  Sharpe: {best['metrics']['sharpe']:.2f} | "
          f"CAGR: {best['metrics']['cagr_pct']:.1f}% | "
          f"MaxDD: {best['metrics']['max_dd_pct']:.1f}% | "
          f"Calmar: {best['metrics']['calmar']:.2f}")
    print("\n  Allocation on $100K:")
    for name, wt in best["weights"].items():
        short_name = name.replace("ML_", "").replace("Stat_Arb_", "")
        print(f"    {short_name:<22}: {wt:.1%} = ${CAPITAL * wt:>10,.0f}")

    # ── Save results ──────────────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "capital": CAPITAL,
        "n_strategies": N,
        "strategy_names": NAMES,
        "correlation_matrix": corr.round(4).to_dict(),
        "individual_metrics": indiv_metrics,
        "portfolio_methods": portfolio_results,
        "marginal_risk_contribution_max_sharpe": {
            NAMES[i]: round(float(mrc[i]), 2) for i in range(N)
        },
        "leave_one_out": loo,
        "recommended": {
            "method": best_method,
            "weights": best["weights"],
            "metrics": best["metrics"],
        },
        "benchmark_spy": {
            "sharpe": 0.887, "cagr_pct": 14.7, "max_dd_pct": -33.7,
            "sortino": 1.096, "calmar": 0.435,
        },
        "notes": [
            "Returns are synthetic (multivariate normal) matching each strategy's known Sharpe/vol/correlation",
            "CTA-Sector correlation of 0.51 is from actual results.json",
            "Stat Arb SPY correlation of 0.043 is from actual backtest results",
            "Vol Breakout equity curve partially loaded from real backtest data",
            "All optimizations are long-only, fully invested, no leverage",
            "Capital: $100K fixed, no DCA",
        ],
    }

    results_path = OUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # ── Equity curve comparison ────────────────────────────────────
    print("\n[8] Equity curves (final capital on $100K):")
    for method, w in methods.items():
        port_rets = returns @ w
        cum = (1 + port_rets).cumprod()
        final = CAPITAL * cum.iloc[-1]
        print(f"  {method:<16}: ${final:>12,.0f}")

    print(f"\n  Done. Runtime: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
