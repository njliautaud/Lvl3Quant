#!/usr/bin/env python3
"""
Portfolio-Level Monte Carlo Simulator
=====================================
Tests how all strategies work TOGETHER across different market scenarios.

Loads real backtest equity curves, runs 1000-trial Monte Carlo with multiple
allocation schemes, stress-tests against historical crashes, and analyzes
cross-strategy correlations.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
GR_OUT = ROOT / "output" / "growth_research"
SIM_OUT = GR_OUT / "portfolio_simulator"
SIM_OUT.mkdir(parents=True, exist_ok=True)

N_TRIALS = 1000
TRADING_DAYS = 252
RISK_FREE = 0.045  # current ~4.5% risk-free rate

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

def load_backtest_returns() -> pd.DataFrame:
    """Load aligned daily returns from the combined optimizer + supplement with
    proxy returns for strategies not yet in the parquet."""

    # Primary source: aligned parquet with 8 real strategy return series
    pq_path = GR_OUT / "combined_optimizer" / "aligned_daily_returns.parquet"
    df = pd.read_parquet(pq_path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Rename for clarity
    rename_map = {
        "V5_CSP": "Wheel_CSP",
        "IC_Condors": "Iron_Condor",
        "ETF_Rotation_v3": "ETF_Rotation",
        "Vol_Harvest_SVXY": "Vol_Harvest",
        "TQQQ_Trend": "Growth_TQQQ",
        "BTC_Trend": "BTC_Trend",
        "QQQ_Collar": "Megacap_Momentum",
        "MultiAsset_Trend": "Multi_Trend",
    }
    df = df.rename(columns=rename_map)

    # Try to load growth_v1 returns and align
    g1_path = GR_OUT / "growth_v1_returns.csv"
    if g1_path.exists():
        g1 = pd.read_csv(g1_path, index_col=0, parse_dates=True)
        g1.columns = ["Growth_QQQ"]
        g1 = g1.reindex(df.index)
        # Fill NaN with 0 for dates outside g1 range
        g1 = g1.fillna(0)
        df["Growth_QQQ"] = g1["Growth_QQQ"]

    # Try to load trend following
    tf_path = GR_OUT / "trend_following_returns.csv"
    if tf_path.exists():
        tf = pd.read_csv(tf_path, index_col=0, parse_dates=True)
        tf.columns = ["Trend_Following"]
        tf = tf.reindex(df.index).fillna(0)
        df["Trend_Following"] = tf["Trend_Following"]

    # Proxy: VIX-Threshold Leveraged (UPRO strategy)
    # Modeled as: long UPRO when VIX < 20, cash when VIX >= 20
    # Proxy using 3x SPY returns with vol filter
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=df.index[0], end=df.index[-1], progress=False)["Close"]
        spy_ret = spy.pct_change().dropna()
        spy_ret = spy_ret.reindex(df.index).fillna(0)
        # Simulate UPRO-like 3x leveraged with simple vol filter
        rolling_vol = spy_ret.rolling(20).std() * np.sqrt(252)
        upro_ret = spy_ret * 3.0
        # When vol > 25% annualized, go to cash
        upro_ret[rolling_vol > 0.25] = 0.0
        df["VIX_Leveraged"] = upro_ret.values
    except Exception:
        # Fallback: synthetic UPRO-like returns
        spy_proxy = df["Megacap_Momentum"] * 2.5
        df["VIX_Leveraged"] = spy_proxy

    # Proxy: Momentum Crash Hedge
    # Modeled as: small negative carry in calm markets, large positive in crashes
    try:
        spy_ret_series = spy_ret.reindex(df.index).fillna(0)
        # Tail hedge: costs ~0.02% daily in calm, gains big in drawdowns
        rng_ch = np.random.RandomState(99)
        crash_hedge = -0.0002 + rng_ch.normal(0, 0.001, len(df))  # daily cost + noise
        # When SPY drops > 2% in a day, hedge pays ~3x the drop
        big_drops = spy_ret_series < -0.02
        crash_hedge[big_drops.values] = -spy_ret_series[big_drops].values * 3.0
        df["Crash_Hedge"] = crash_hedge
    except Exception:
        rng_ch = np.random.RandomState(99)
        df["Crash_Hedge"] = -0.0002 + rng_ch.normal(0, 0.001, len(df))

    # Proxy: BPS (Bull Put Spread) - similar to CSP but capped risk
    # Slightly lower return, lower vol than wheel
    df["BPS"] = df["Wheel_CSP"] * 0.8 + np.random.RandomState(42).normal(0, 0.0005, len(df))

    # Proxy: Strangle - sells both sides, higher vol
    df["Strangle"] = (df["Wheel_CSP"] + df["Iron_Condor"]) * 0.6

    print(f"Loaded {len(df.columns)} strategies, {len(df)} trading days "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    print(f"Strategies: {', '.join(df.columns)}")

    return df


def fetch_stress_data():
    """Fetch historical market data for stress test periods using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    periods = {
        "COVID_2020": ("2020-02-19", "2020-03-23"),
        "Rate_Hike_2022": ("2022-01-03", "2022-10-12"),
        "GFC_2008": ("2008-09-15", "2008-12-31"),
        "VIX_Spike": ("2020-02-24", "2020-03-18"),  # VIX > 40 period
        "Gradual_Bear": ("2022-01-03", "2022-06-17"),  # SPY -24% over ~6 months
    }

    tickers = ["SPY", "QQQ", "IWM", "^VIX"]
    stress_data = {}

    for name, (start, end) in periods.items():
        try:
            data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame()
            rets = data.pct_change().dropna()
            stress_data[name] = {
                "returns": rets,
                "spy_total_return": (data["SPY"].iloc[-1] / data["SPY"].iloc[0] - 1) if "SPY" in data.columns else None,
                "days": len(rets),
            }
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    return stress_data


# ---------------------------------------------------------------------------
# 2. ALLOCATION SCHEMES
# ---------------------------------------------------------------------------

def get_allocations(df: pd.DataFrame) -> dict:
    """Define allocation mixes."""
    cols = df.columns.tolist()
    n = len(cols)

    # Strategy categories
    income = ["Wheel_CSP", "Iron_Condor", "BPS", "Strangle"]
    growth = ["Growth_TQQQ", "Megacap_Momentum", "Growth_QQQ", "VIX_Leveraged"]
    hedge = ["Vol_Harvest", "BTC_Trend", "Crash_Hedge", "Multi_Trend", "Trend_Following"]

    # Filter to available columns
    income = [c for c in income if c in cols]
    growth = [c for c in growth if c in cols]
    hedge = [c for c in hedge if c in cols]
    other = [c for c in cols if c not in income + growth + hedge]

    allocations = {}

    # A) Equal weight
    eq_w = {c: 1.0 / n for c in cols}
    allocations["A_Equal_Weight"] = eq_w

    # B) Risk parity (inverse-vol)
    vols = df.std() * np.sqrt(TRADING_DAYS)
    inv_vol = 1.0 / vols.clip(lower=0.001)
    inv_vol_w = inv_vol / inv_vol.sum()
    allocations["B_Risk_Parity"] = inv_vol_w.to_dict()

    # C) 50/30/20 split
    split_w = {}
    for c in cols:
        if c in income:
            split_w[c] = 0.50 / max(len(income), 1)
        elif c in growth:
            split_w[c] = 0.30 / max(len(growth), 1)
        elif c in hedge:
            split_w[c] = 0.20 / max(len(hedge), 1)
        else:
            split_w[c] = 0.0
    # Normalize to ensure sum = 1
    total = sum(split_w.values())
    if total > 0:
        split_w = {k: v / total for k, v in split_w.items()}
    allocations["C_Income_Growth_Hedge"] = split_w

    # D) Max-Sharpe (mean-variance optimization)
    try:
        from scipy.optimize import minimize

        mu = df.mean().values * TRADING_DAYS
        cov = df.cov().values * TRADING_DAYS

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - RISK_FREE) / max(vol, 1e-8)

        n_assets = len(cols)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 0.5) for _ in range(n_assets)]  # max 50% per strategy
        x0 = np.ones(n_assets) / n_assets

        result = minimize(neg_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000})
        if result.success:
            opt_w = {cols[i]: float(result.x[i]) for i in range(n_assets)}
        else:
            opt_w = eq_w.copy()
            print("  Warning: MVO optimization failed, using equal weight")
    except Exception as e:
        opt_w = eq_w.copy()
        print(f"  Warning: MVO failed ({e}), using equal weight")

    allocations["D_Max_Sharpe"] = opt_w

    return allocations


# ---------------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------

def compute_metrics(daily_returns: np.ndarray) -> dict:
    """Compute portfolio metrics from a daily return series."""
    cumret = np.cumprod(1 + daily_returns)
    total_ret = cumret[-1] - 1
    n_years = len(daily_returns) / TRADING_DAYS

    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = np.std(daily_returns) * np.sqrt(TRADING_DAYS)
    sharpe = (cagr - RISK_FREE) / max(ann_vol, 1e-8)

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 1e-8
    sortino = (cagr - RISK_FREE) / max(downside_vol, 1e-8)

    # Max drawdown
    running_max = np.maximum.accumulate(cumret)
    drawdowns = cumret / running_max - 1
    max_dd = float(np.min(drawdowns))

    # Monthly returns for best/worst month
    n_months = max(len(daily_returns) // 21, 1)
    monthly_chunks = np.array_split(daily_returns, n_months)
    monthly_rets = [np.prod(1 + chunk) - 1 for chunk in monthly_chunks if len(chunk) > 0]

    return {
        "CAGR": round(float(cagr) * 100, 2),
        "AnnVol": round(float(ann_vol) * 100, 2),
        "Sharpe": round(float(sharpe), 2),
        "Sortino": round(float(sortino), 2),
        "MaxDD": round(float(max_dd) * 100, 2),
        "Calmar": round(float(cagr) / max(abs(max_dd), 1e-8), 2),
        "Best_Month": round(float(max(monthly_rets)) * 100, 2) if monthly_rets else 0,
        "Worst_Month": round(float(min(monthly_rets)) * 100, 2) if monthly_rets else 0,
        "WR_Daily": round(float(np.mean(daily_returns > 0)) * 100, 1),
    }


def run_monte_carlo(df: pd.DataFrame, allocations: dict, n_trials: int = N_TRIALS) -> dict:
    """Run Monte Carlo bootstrap simulation."""
    print(f"\nRunning {n_trials}-trial Monte Carlo simulation...")
    results = {}
    rng = np.random.RandomState(42)

    n_days = len(df)
    sim_horizon = TRADING_DAYS * 3  # 3-year simulation per trial

    for alloc_name, weights in allocations.items():
        w = np.array([weights.get(c, 0) for c in df.columns])
        # Portfolio daily returns (historical)
        port_daily = (df.values * w).sum(axis=1)

        trial_metrics = []
        for trial in range(n_trials):
            # Bootstrap: resample blocks of 5 days to preserve autocorrelation
            block_size = 5
            n_blocks = sim_horizon // block_size + 1
            block_starts = rng.randint(0, n_days - block_size, size=n_blocks)
            sim_returns = np.concatenate([port_daily[s:s+block_size] for s in block_starts])[:sim_horizon]
            trial_metrics.append(compute_metrics(sim_returns))

        # Aggregate across trials
        agg = {}
        for key in trial_metrics[0]:
            vals = [t[key] for t in trial_metrics]
            agg[key] = {
                "median": round(float(np.median(vals)), 2),
                "mean": round(float(np.mean(vals)), 2),
                "p5": round(float(np.percentile(vals, 5)), 2),
                "p25": round(float(np.percentile(vals, 25)), 2),
                "p75": round(float(np.percentile(vals, 75)), 2),
                "p95": round(float(np.percentile(vals, 95)), 2),
            }

        # Also compute actual (non-bootstrap) metrics
        actual = compute_metrics(port_daily)

        results[alloc_name] = {
            "weights": {k: round(v, 4) for k, v in weights.items() if v > 0.001},
            "actual_metrics": actual,
            "monte_carlo": agg,
        }

        print(f"  {alloc_name}: Sharpe={actual['Sharpe']:.2f}, "
              f"CAGR={actual['CAGR']:.1f}%, MaxDD={actual['MaxDD']:.1f}%, "
              f"MC_Sharpe_p5={agg['Sharpe']['p5']:.2f}")

    return results


# ---------------------------------------------------------------------------
# 4. STRESS TESTS
# ---------------------------------------------------------------------------

def run_stress_tests(df: pd.DataFrame, allocations: dict, stress_data: dict) -> dict:
    """Apply stress scenarios to each allocation mix."""
    print("\nRunning stress tests...")

    if not stress_data:
        print("  No stress data available (yfinance issue), using synthetic scenarios")
        return _synthetic_stress_tests(df, allocations)

    results = {}

    for scenario_name, sdata in stress_data.items():
        scenario_results = {}
        spy_drop = sdata.get("spy_total_return")
        days = sdata["days"]

        for alloc_name, weights in allocations.items():
            w = np.array([weights.get(c, 0) for c in df.columns])

            # Find overlapping dates
            stress_dates = sdata["returns"].index
            overlap = df.index.intersection(stress_dates)

            if len(overlap) > 10:
                # Use actual strategy returns during the stress period
                port_ret = (df.loc[overlap].values * w).sum(axis=1)
            else:
                # Estimate: use SPY beta of each strategy to project returns
                spy_rets = sdata["returns"]["SPY"] if "SPY" in sdata["returns"].columns else None
                if spy_rets is not None:
                    # Use average daily SPY return during stress, apply beta
                    avg_spy = spy_rets.mean()
                    # Estimate strategy betas from historical data
                    spy_hist = df.get("Megacap_Momentum", df.iloc[:, 0])
                    betas = []
                    for col in df.columns:
                        cov_val = np.cov(df[col].values, spy_hist.values)[0, 1]
                        var_val = np.var(spy_hist.values)
                        betas.append(cov_val / max(var_val, 1e-10))
                    port_beta = np.dot(w, betas)
                    port_ret = np.full(days, avg_spy * port_beta)
                else:
                    port_ret = np.zeros(days)

            total_ret = float(np.prod(1 + port_ret) - 1)
            max_dd_stress = float(np.min(np.cumprod(1 + port_ret) / np.maximum.accumulate(np.cumprod(1 + port_ret)) - 1))

            scenario_results[alloc_name] = {
                "total_return_pct": round(total_ret * 100, 2),
                "max_drawdown_pct": round(max_dd_stress * 100, 2),
                "days": len(port_ret),
                "survived": total_ret > -0.30,  # survived = didn't lose > 30%
            }

        spy_label = f"{spy_drop*100:.1f}%" if spy_drop else "N/A"
        results[scenario_name] = {
            "spy_total_return": spy_label,
            "days": days,
            "allocations": scenario_results,
        }
        best = min(scenario_results.items(), key=lambda x: abs(x[1]["total_return_pct"]))
        print(f"  {scenario_name} (SPY: {spy_label}): "
              f"Best alloc = {best[0]} ({best[1]['total_return_pct']:+.1f}%)")

    return results


def _synthetic_stress_tests(df: pd.DataFrame, allocations: dict) -> dict:
    """Fallback: use worst historical periods from the data itself."""
    results = {}
    for alloc_name, weights in allocations.items():
        w = np.array([weights.get(c, 0) for c in df.columns])
        port_daily = (df.values * w).sum(axis=1)

        # Find worst 22-day rolling return
        rolling_22 = pd.Series(port_daily).rolling(22).apply(lambda x: np.prod(1+x)-1, raw=True)
        worst_idx = rolling_22.idxmin()
        worst_period = port_daily[max(0, worst_idx-21):worst_idx+1]
        results[alloc_name] = {
            "worst_month_return": round(float(np.prod(1+worst_period)-1)*100, 2),
            "worst_month_end": str(df.index[worst_idx].date()) if worst_idx < len(df) else "N/A",
        }
    return {"Worst_Month_Historical": {"allocations": results}}


# ---------------------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ---------------------------------------------------------------------------

def analyze_correlations(df: pd.DataFrame) -> dict:
    """Analyze cross-strategy and SPY correlations."""
    print("\nAnalyzing correlations...")
    corr = df.corr()

    # Find most/least correlated pairs
    pairs = []
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], round(float(corr.iloc[i, j]), 3)))

    pairs.sort(key=lambda x: x[2])
    least_corr = pairs[:5]
    most_corr = pairs[-5:]

    # Portfolio-level diversification ratio
    avg_corr = float(corr.values[np.triu_indices_from(corr.values, k=1)].mean())

    # Effective number of independent bets (ENB)
    try:
        eigenvalues = np.linalg.eigvalsh(corr.values)
    except np.linalg.LinAlgError:
        # Fallback: use SVD which is more numerically stable
        eigenvalues = np.linalg.svd(corr.values, compute_uv=False)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    enb = float(np.exp(-np.sum(p * np.log(p + 1e-10))))

    result = {
        "avg_pairwise_correlation": round(avg_corr, 3),
        "effective_independent_bets": round(enb, 1),
        "n_strategies": len(cols),
        "most_correlated_pairs": [
            {"pair": f"{a} / {b}", "corr": c} for a, b, c in most_corr
        ],
        "least_correlated_pairs": [
            {"pair": f"{a} / {b}", "corr": c} for a, b, c in least_corr
        ],
        "correlation_matrix": {
            str(cols[i]): {str(cols[j]): round(float(corr.iloc[i, j]), 3)
                           for j in range(len(cols))}
            for i in range(len(cols))
        },
        "best_diversifiers": [],
    }

    # Strategies with lowest average correlation to others
    avg_corr_per = corr.mean().sort_values()
    result["best_diversifiers"] = [
        {"strategy": str(k), "avg_corr": round(float(v), 3)}
        for k, v in avg_corr_per.head(5).items()
    ]

    print(f"  Avg pairwise correlation: {avg_corr:.3f}")
    print(f"  Effective independent bets: {enb:.1f} (out of {len(cols)} strategies)")
    print(f"  Best diversifier: {avg_corr_per.index[0]} (avg corr = {avg_corr_per.iloc[0]:.3f})")

    return result


# ---------------------------------------------------------------------------
# 6. SUMMARY REPORT
# ---------------------------------------------------------------------------

def write_summary(mc_results: dict, stress_results: dict, corr_results: dict):
    """Write human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("PORTFOLIO SIMULATOR — SUMMARY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # --- Monte Carlo Results ---
    lines.append("\n--- ALLOCATION COMPARISON (Actual + Monte Carlo) ---\n")
    lines.append(f"{'Allocation':<25} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>7} {'Sortino':>8} {'Calmar':>7} {'WR%':>6}")
    lines.append("-" * 70)

    for name, data in mc_results.items():
        a = data["actual_metrics"]
        lines.append(f"{name:<25} {a['Sharpe']:>7.2f} {a['CAGR']:>6.1f}% {a['MaxDD']:>6.1f}% {a['Sortino']:>7.2f} {a['Calmar']:>7.2f} {a['WR_Daily']:>5.1f}")

    lines.append("\n--- MONTE CARLO CONFIDENCE (1000 trials, 3yr horizon) ---\n")
    lines.append(f"{'Allocation':<25} {'Sharpe p5':>10} {'Sharpe med':>11} {'Sharpe p95':>11} {'CAGR p5':>8} {'MaxDD p5':>9}")
    lines.append("-" * 75)
    for name, data in mc_results.items():
        mc = data["monte_carlo"]
        lines.append(f"{name:<25} {mc['Sharpe']['p5']:>10.2f} {mc['Sharpe']['median']:>11.2f} "
                     f"{mc['Sharpe']['p95']:>11.2f} {mc['CAGR']['p5']:>7.1f}% {mc['MaxDD']['p5']:>8.1f}%")

    # --- Top weights ---
    lines.append("\n--- ALLOCATION WEIGHTS (non-zero) ---\n")
    for name, data in mc_results.items():
        wts = data["weights"]
        sorted_wts = sorted(wts.items(), key=lambda x: -x[1])
        wt_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted_wts[:6])
        lines.append(f"{name}: {wt_str}")

    # --- Stress Tests ---
    lines.append("\n--- STRESS TEST RESULTS ---\n")
    for scenario, sdata in stress_results.items():
        spy_label = sdata.get("spy_total_return", "N/A")
        lines.append(f"\n  {scenario} (SPY: {spy_label}):")
        allocs = sdata.get("allocations", {})
        for aname, adata in allocs.items():
            ret = adata.get("total_return_pct", adata.get("worst_month_return", "?"))
            dd = adata.get("max_drawdown_pct", "?")
            survived = adata.get("survived", "?")
            lines.append(f"    {aname:<25} Return: {ret:>+7.1f}%  MaxDD: {dd:>7.1f}%  Survived: {survived}")

    # --- Correlations ---
    lines.append("\n--- CORRELATION INSIGHTS ---\n")
    lines.append(f"  Average pairwise correlation: {corr_results['avg_pairwise_correlation']:.3f}")
    lines.append(f"  Effective independent bets:   {corr_results['effective_independent_bets']:.1f} / {corr_results['n_strategies']}")

    lines.append("\n  Best diversifiers (lowest avg correlation to portfolio):")
    for d in corr_results["best_diversifiers"]:
        lines.append(f"    {d['strategy']:<25} avg_corr = {d['avg_corr']:.3f}")

    lines.append("\n  Most correlated pairs:")
    for p in corr_results["most_correlated_pairs"]:
        lines.append(f"    {p['pair']:<40} r = {p['corr']:.3f}")

    lines.append("\n  Least correlated pairs:")
    for p in corr_results["least_correlated_pairs"]:
        lines.append(f"    {p['pair']:<40} r = {p['corr']:.3f}")

    # --- Best combo ---
    lines.append("\n--- BEST PORTFOLIO ---\n")
    best_name = max(mc_results.keys(), key=lambda k: mc_results[k]["actual_metrics"]["Sharpe"])
    best = mc_results[best_name]
    lines.append(f"  Winner: {best_name}")
    lines.append(f"  Sharpe: {best['actual_metrics']['Sharpe']:.2f}  |  "
                 f"CAGR: {best['actual_metrics']['CAGR']:.1f}%  |  "
                 f"MaxDD: {best['actual_metrics']['MaxDD']:.1f}%  |  "
                 f"Sortino: {best['actual_metrics']['Sortino']:.2f}")
    wts = sorted(best["weights"].items(), key=lambda x: -x[1])
    lines.append(f"  Top weights: {', '.join(f'{k}={v:.1%}' for k, v in wts[:5])}")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)

    # Write report
    report_path = SIM_OUT / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport written to {report_path}")
    print("\n" + report)

    return report


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("PORTFOLIO SIMULATOR v1.0")
    print("=" * 60)

    # 1. Load data
    df = load_backtest_returns()

    # 2. Define allocations
    allocations = get_allocations(df)

    # 3. Monte Carlo simulation
    mc_results = run_monte_carlo(df, allocations)

    # 4. Stress tests
    stress_data = fetch_stress_data()
    stress_results = run_stress_tests(df, allocations, stress_data)

    # 5. Correlation analysis
    corr_results = analyze_correlations(df)

    # 6. Save all results to JSON
    full_results = {
        "generated": datetime.now().isoformat(),
        "n_strategies": len(df.columns),
        "strategies": df.columns.tolist(),
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_days": len(df),
        "n_monte_carlo_trials": N_TRIALS,
        "monte_carlo": mc_results,
        "stress_tests": stress_results,
        "correlations": corr_results,
    }

    json_path = SIM_OUT / "simulation_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\nFull results saved to {json_path}")

    # 7. Summary report
    write_summary(mc_results, stress_results, corr_results)


if __name__ == "__main__":
    main()
