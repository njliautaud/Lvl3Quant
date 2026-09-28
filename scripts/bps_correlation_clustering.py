#!/usr/bin/env python3
"""
BPS Position Correlation & Loss Clustering Study — HC #664 R4
==============================================================

Critical question: How correlated are BPS position losses?

If losses are highly correlated (all 70 tickers lose on the same days),
then our diversification-based Sharpe 2.05 is overstated — the tail risk
is worse than the backtest shows.

If losses are weakly correlated (different tickers lose on different days),
then diversification is real and the Sharpe is honest.

This analysis:
1. Measures pairwise return correlation across BPS positions
2. Identifies "cluster days" where many positions lose simultaneously
3. Computes conditional drawdown: given one position breaches, how many others?
4. Estimates the "effective diversification" (how many independent bets?)
5. Stress-tests: what happens if correlations spike to crisis levels?

Output: output/bps_correlation/correlation_results.json
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

OUTPUT = ROOT / "output" / "bps_correlation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_trades():
    """Load trade-level data."""
    trades_path = ROOT / "output" / "bps_assignment_risk" / "trades_close_1dte.parquet"
    return pd.read_parquet(trades_path)


def load_macro():
    """Load macro data for VIX."""
    from higher_returns_study import load_data
    prices, iv, macro, fund, universe, earnings = load_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    return macro


def build_daily_pnl_matrix(trades):
    """
    Build a matrix of daily P&L per ticker.
    Rows = dates, Columns = tickers.
    This lets us measure cross-ticker correlations.
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # Group by date + ticker
    daily_ticker_pnl = trades.groupby(["close_date", "ticker"])["realized_pnl"].sum()
    matrix = daily_ticker_pnl.unstack(fill_value=0.0)

    # Only keep tickers with enough data (>100 trade-days)
    good_tickers = matrix.columns[(matrix != 0).sum() > 100]
    matrix = matrix[good_tickers]

    return matrix


def correlation_analysis(pnl_matrix):
    """Compute pairwise correlations and summary statistics."""
    # Only use days where at least 2 tickers have non-zero PnL
    active_days = (pnl_matrix != 0).sum(axis=1) >= 2
    active_matrix = pnl_matrix.loc[active_days]

    if len(active_matrix) < 50:
        return {"note": "too few active days"}

    # Replace zeros with NaN for correlation (zero = no position, not zero return)
    corr_matrix = active_matrix.replace(0, np.nan).corr(min_periods=20)

    # Extract upper triangle (no self-correlation)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    pairwise_corrs = corr_matrix.where(mask).stack().values

    # Remove NaN
    pairwise_corrs = pairwise_corrs[~np.isnan(pairwise_corrs)]

    if len(pairwise_corrs) == 0:
        return {"note": "no valid pairwise correlations"}

    return {
        "n_tickers": len(pnl_matrix.columns),
        "n_active_days": int(active_days.sum()),
        "mean_pairwise_corr": round(float(np.mean(pairwise_corrs)), 3),
        "median_pairwise_corr": round(float(np.median(pairwise_corrs)), 3),
        "p25_corr": round(float(np.percentile(pairwise_corrs, 25)), 3),
        "p75_corr": round(float(np.percentile(pairwise_corrs, 75)), 3),
        "pct_negative_corr": round(float((pairwise_corrs < 0).mean() * 100), 1),
        "pct_high_corr_above_0.5": round(float((pairwise_corrs > 0.5).mean() * 100), 1),
        "n_pairs": len(pairwise_corrs),
    }


def loss_clustering_analysis(trades, macro):
    """
    Analyze how losses cluster across tickers.
    Key question: On a bad day, how many tickers lose simultaneously?
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["is_loss"] = trades["realized_pnl"] < 0

    # Per-day stats
    daily_stats = trades.groupby("close_date").agg(
        n_trades=("realized_pnl", "count"),
        n_losses=("is_loss", "sum"),
        total_pnl=("realized_pnl", "sum"),
        worst_trade=("realized_pnl", "min"),
    )
    daily_stats["loss_frac"] = daily_stats["n_losses"] / daily_stats["n_trades"]

    # Merge VIX
    vix_by_date = macro.set_index("date")["vix"].to_dict()
    daily_stats["vix"] = daily_stats.index.map(lambda d: vix_by_date.get(d, np.nan))

    # Cluster days: >50% of positions losing
    cluster_days = daily_stats[daily_stats["loss_frac"] > 0.5]
    severe_cluster = daily_stats[daily_stats["loss_frac"] > 0.75]

    # Non-cluster (normal) days
    normal_days = daily_stats[daily_stats["loss_frac"] <= 0.25]

    return {
        "total_days": len(daily_stats),
        "cluster_days_gt50pct": len(cluster_days),
        "severe_cluster_gt75pct": len(severe_cluster),
        "cluster_day_pct": round(len(cluster_days) / len(daily_stats) * 100, 1),
        "avg_loss_frac_all_days": round(float(daily_stats["loss_frac"].mean()), 3),
        "avg_loss_frac_cluster_days": round(float(cluster_days["loss_frac"].mean()), 3) if len(cluster_days) > 0 else None,
        "avg_pnl_cluster_days": round(float(cluster_days["total_pnl"].mean()), 0) if len(cluster_days) > 0 else None,
        "avg_pnl_normal_days": round(float(normal_days["total_pnl"].mean()), 0) if len(normal_days) > 0 else None,
        "avg_vix_cluster_days": round(float(cluster_days["vix"].mean()), 1) if len(cluster_days) > 0 else None,
        "avg_vix_normal_days": round(float(normal_days["vix"].mean()), 1) if len(normal_days) > 0 else None,
        "worst_cluster_day_pnl": round(float(cluster_days["total_pnl"].min()), 0) if len(cluster_days) > 0 else None,
        "worst_cluster_day_losses": int(cluster_days["n_losses"].max()) if len(cluster_days) > 0 else None,
    }


def effective_diversification(pnl_matrix):
    """
    Estimate the number of "independent bets" using PCA.
    If 70 tickers all move together, effective_n ≈ 1.
    If all independent, effective_n ≈ 70.

    Method: eigenvalue decomposition of return covariance matrix.
    effective_n = (sum of eigenvalues)^2 / (sum of eigenvalues^2)
    This is the "participation ratio" from random matrix theory.
    """
    # Use only days with some activity
    active = pnl_matrix.replace(0, np.nan)
    # Need enough data for covariance
    good_cols = active.columns[active.count() > 50]
    if len(good_cols) < 5:
        return {"note": "too few tickers for PCA"}

    active = active[good_cols].fillna(0)

    # Covariance matrix
    cov = active.cov()
    eigenvalues = np.linalg.eigvalsh(cov.values)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove numerical noise

    if len(eigenvalues) == 0:
        return {"note": "no positive eigenvalues"}

    # Participation ratio
    effective_n = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())

    # Fraction of variance explained by top eigenvectors
    sorted_eig = np.sort(eigenvalues)[::-1]
    total_var = sorted_eig.sum()
    top1_pct = float(sorted_eig[0] / total_var * 100)
    top3_pct = float(sorted_eig[:3].sum() / total_var * 100) if len(sorted_eig) >= 3 else top1_pct
    top5_pct = float(sorted_eig[:5].sum() / total_var * 100) if len(sorted_eig) >= 5 else top3_pct

    return {
        "n_tickers_analyzed": len(good_cols),
        "effective_independent_bets": round(effective_n, 1),
        "diversification_ratio": round(effective_n / len(good_cols), 2),
        "top1_factor_var_pct": round(top1_pct, 1),
        "top3_factors_var_pct": round(top3_pct, 1),
        "top5_factors_var_pct": round(top5_pct, 1),
        "interpretation": (
            "HIGH" if effective_n / len(good_cols) > 0.3
            else "MODERATE" if effective_n / len(good_cols) > 0.15
            else "LOW — losses are highly correlated, diversification is weaker than it looks"
        ),
    }


def conditional_breach_analysis(trades):
    """
    Given one position breaches (loses > $500), how many others also breach on the same day?
    This measures contagion — are breaches independent or correlated?
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])
    trades["is_breach"] = trades["realized_pnl"] < -500  # significant loss threshold

    # Days with at least one breach
    breaches_per_day = trades.groupby("close_date")["is_breach"].sum()
    trades_per_day = trades.groupby("close_date")["is_breach"].count()

    breach_days = breaches_per_day[breaches_per_day > 0]
    if len(breach_days) == 0:
        return {"note": "no breach days found"}

    # On breach days, what fraction of positions also breach?
    breach_frac = (breach_days / trades_per_day.loc[breach_days.index])

    # Compare: if breaches were independent at base rate p, expected concurrent = n*p
    base_rate = float(trades["is_breach"].mean())
    avg_positions_per_day = float(trades_per_day.mean())
    expected_concurrent = base_rate * avg_positions_per_day

    return {
        "n_breach_days": len(breach_days),
        "total_days": len(breaches_per_day),
        "breach_day_pct": round(len(breach_days) / len(breaches_per_day) * 100, 1),
        "avg_breaches_per_breach_day": round(float(breach_days.mean()), 1),
        "max_breaches_single_day": int(breach_days.max()),
        "avg_breach_fraction": round(float(breach_frac.mean()), 3),
        "base_breach_rate": round(base_rate, 4),
        "expected_concurrent_if_independent": round(expected_concurrent, 2),
        "actual_vs_expected_ratio": round(float(breach_days.mean()) / expected_concurrent, 1) if expected_concurrent > 0 else None,
        "interpretation": (
            "Breaches are HIGHLY CLUSTERED" if float(breach_days.mean()) / max(expected_concurrent, 0.01) > 3
            else "Breaches are MODERATELY clustered" if float(breach_days.mean()) / max(expected_concurrent, 0.01) > 1.5
            else "Breaches are roughly INDEPENDENT — diversification is real"
        ),
    }


def crisis_stress_test(trades, macro):
    """
    Stress test: What's the worst-case if correlations spike to 2020-COVID levels?
    Use actual March 2020 data if available, otherwise simulate.
    """
    trades = trades.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    vix_by_date = macro.set_index("date")["vix"].to_dict()
    trades["vix"] = trades["close_date"].map(vix_by_date)

    # Find actual crisis periods (VIX > 30)
    crisis_trades = trades[trades["vix"] > 30]
    normal_trades = trades[trades["vix"] < 20]

    if len(crisis_trades) < 10 or len(normal_trades) < 50:
        return {"note": "insufficient data for stress test"}

    # Compare loss rates
    crisis_loss_rate = float((crisis_trades["realized_pnl"] < 0).mean())
    normal_loss_rate = float((normal_trades["realized_pnl"] < 0).mean())

    # Average loss magnitude
    crisis_avg_loss = float(crisis_trades[crisis_trades["realized_pnl"] < 0]["realized_pnl"].mean()) if (crisis_trades["realized_pnl"] < 0).any() else 0
    normal_avg_loss = float(normal_trades[normal_trades["realized_pnl"] < 0]["realized_pnl"].mean()) if (normal_trades["realized_pnl"] < 0).any() else 0

    # Worst single day
    crisis_daily = crisis_trades.groupby("close_date")["realized_pnl"].sum()
    normal_daily = normal_trades.groupby("close_date")["realized_pnl"].sum()

    return {
        "crisis_n_trades": len(crisis_trades),
        "normal_n_trades": len(normal_trades),
        "crisis_loss_rate": round(crisis_loss_rate * 100, 1),
        "normal_loss_rate": round(normal_loss_rate * 100, 1),
        "loss_rate_multiplier": round(crisis_loss_rate / max(normal_loss_rate, 0.01), 1),
        "crisis_avg_loss_per_trade": round(crisis_avg_loss, 0),
        "normal_avg_loss_per_trade": round(normal_avg_loss, 0),
        "loss_severity_multiplier": round(crisis_avg_loss / min(normal_avg_loss, -0.01), 1) if normal_avg_loss < 0 else None,
        "crisis_worst_day": round(float(crisis_daily.min()), 0) if len(crisis_daily) > 0 else None,
        "normal_worst_day": round(float(normal_daily.min()), 0) if len(normal_daily) > 0 else None,
        "crisis_var_95_daily": round(float(np.percentile(crisis_daily, 5)), 0) if len(crisis_daily) > 10 else None,
        "normal_var_95_daily": round(float(np.percentile(normal_daily, 5)), 0) if len(normal_daily) > 10 else None,
    }


def main():
    t0 = time.time()

    print("Loading data...")
    trades = load_trades()
    macro = load_macro()
    print(f"  {len(trades)} trades, {trades['ticker'].nunique()} tickers")

    results = {}

    # 1. Build daily PnL matrix
    print("\n=== Building Daily PnL Matrix ===")
    pnl_matrix = build_daily_pnl_matrix(trades)
    print(f"  Matrix: {pnl_matrix.shape[0]} days x {pnl_matrix.shape[1]} tickers")

    # 2. Pairwise correlations
    print("\n=== Pairwise Correlation Analysis ===")
    corr = correlation_analysis(pnl_matrix)
    results["pairwise_correlations"] = corr
    for k, v in corr.items():
        print(f"  {k}: {v}")

    # 3. Loss clustering
    print("\n=== Loss Clustering Analysis ===")
    clustering = loss_clustering_analysis(trades, macro)
    results["loss_clustering"] = clustering
    for k, v in clustering.items():
        print(f"  {k}: {v}")

    # 4. Effective diversification (PCA)
    print("\n=== Effective Diversification (PCA) ===")
    eff_div = effective_diversification(pnl_matrix)
    results["effective_diversification"] = eff_div
    for k, v in eff_div.items():
        print(f"  {k}: {v}")

    # 5. Conditional breach analysis
    print("\n=== Conditional Breach Analysis ===")
    breach = conditional_breach_analysis(trades)
    results["conditional_breaches"] = breach
    for k, v in breach.items():
        print(f"  {k}: {v}")

    # 6. Crisis stress test
    print("\n=== Crisis Stress Test (VIX > 30 vs < 20) ===")
    stress = crisis_stress_test(trades, macro)
    results["crisis_stress_test"] = stress
    for k, v in stress.items():
        print(f"  {k}: {v}")

    # Summary verdict
    print("\n" + "="*70)
    print("VERDICT: Is diversification real?")
    print("="*70)

    if "effective_independent_bets" in eff_div:
        n_eff = eff_div["effective_independent_bets"]
        n_total = eff_div["n_tickers_analyzed"]
        ratio = n_eff / n_total
        print(f"  Effective independent bets: {n_eff:.0f} out of {n_total} tickers ({ratio:.0%})")

    if "mean_pairwise_corr" in corr:
        print(f"  Mean pairwise correlation: {corr['mean_pairwise_corr']:.3f}")

    if "actual_vs_expected_ratio" in breach and breach["actual_vs_expected_ratio"]:
        print(f"  Breach clustering: {breach['actual_vs_expected_ratio']:.1f}x expected ({breach['interpretation']})")

    if "cluster_day_pct" in clustering:
        print(f"  Cluster days (>50% losing): {clustering['cluster_day_pct']:.1f}% of all days")

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(OUTPUT / "correlation_results.json", "w") as f:
        json.dump(convert(results), f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results → {OUTPUT / 'correlation_results.json'}")


if __name__ == "__main__":
    main()
