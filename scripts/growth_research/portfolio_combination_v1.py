#!/usr/bin/env python3
"""
Portfolio Combination Study V1
Tests cross-strategy portfolio combinations using validated strategy daily returns.
Finds optimal blending weights via equal-weight, risk-parity, max-Sharpe, and regime-switching.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from datetime import datetime

warnings.filterwarnings('ignore')

# Paths
BASE = "/home/jupiter/Lvl3Quant"
OUT_DIR = f"{BASE}/output/growth_research/portfolio_combination_v1"
os.makedirs(OUT_DIR, exist_ok=True)

ANNUALIZE = 252

# ── Strategy data sources ──────────────────────────────────────────────
STRATEGY_SOURCES = {
    "CTA_Trend": {
        "path": f"{BASE}/output/ml_portfolio_combo/cta_daily_returns.csv",
        "date_col": "date", "ret_col": "return",
    },
    "Sector_ML_Rotation": {
        "path": f"{BASE}/output/ml_portfolio_combo/sector_daily_returns.csv",
        "date_col": "date", "ret_col": "return",
    },
    "CTA_Sector_Combo": {
        "path": f"{BASE}/output/ml_portfolio_combo/combo_daily_returns.csv",
        "date_col": "date", "ret_col": "return",
    },
    "ETF_Rotation_V3": {
        "path": f"{BASE}/output/etf_rotation_v2_lag_fix/book.parquet",
        "date_col": "date", "ret_col": "daily_ret", "format": "parquet",
    },
    "Sector_Enhanced": {
        "path": f"{BASE}/output/growth_research/sector_enhanced_allocator/daily_returns.csv",
        "date_col": "Unnamed: 0", "ret_col": "tilted_return",
    },
}


def load_strategy_returns():
    """Load all available daily return series, align on overlapping dates."""
    series = {}
    for name, cfg in STRATEGY_SOURCES.items():
        path = cfg["path"]
        if not os.path.exists(path):
            print(f"  SKIP {name}: file not found")
            continue
        try:
            if cfg.get("format") == "parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)

            date_col = cfg["date_col"]
            ret_col = cfg["ret_col"]

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)[[ret_col]].rename(columns={ret_col: name})
            df = df.dropna()
            df = df[~df.index.duplicated(keep='first')]
            series[name] = df
            print(f"  LOADED {name}: {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
    return series


def compute_metrics(returns_series, name=""):
    """Compute standard risk-adjusted metrics for a daily return series."""
    r = returns_series.dropna()
    if len(r) < 30:
        return None

    n = len(r)
    ann_ret = r.mean() * ANNUALIZE
    ann_vol = r.std() * np.sqrt(ANNUALIZE)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0].std() * np.sqrt(ANNUALIZE)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + r).cumprod()
    n_years = n / ANNUALIZE
    cagr = (cum.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0

    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    wr = (r > 0).mean()

    # Skewness and kurtosis
    skew = r.skew()
    kurt = r.kurtosis()

    return {
        "name": name,
        "n_days": n,
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "profit_factor": round(pf, 4),
        "win_rate": round(wr, 4),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
    }


def get_vix_data(start_date, end_date):
    """Download VIX daily close for regime classification."""
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix['Close'].rename('VIX')
    except Exception as e:
        print(f"  WARNING: Could not download VIX data: {e}")
        return None


def risk_parity_weights(cov_matrix):
    """Inverse-volatility weighting (simplified risk parity)."""
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


def max_sharpe_weights(returns_df, risk_free=0.0):
    """Mean-variance max Sharpe optimization with long-only constraints."""
    from scipy.optimize import minimize

    n = returns_df.shape[1]
    mu = returns_df.mean().values * ANNUALIZE
    cov = returns_df.cov().values * ANNUALIZE

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret - risk_free) / port_vol if port_vol > 1e-10 else 0

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.05, 0.60)] * n  # min 5%, max 60% per strategy
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        return np.ones(n) / n  # fallback to equal weight


def regime_switched_weights(returns_df, vix_series):
    """Different weights for low-vol (VIX<20) vs high-vol (VIX>=20) regimes."""
    aligned = returns_df.copy()
    aligned['VIX'] = vix_series
    aligned = aligned.dropna()

    low_vol = aligned[aligned['VIX'] < 20].drop(columns=['VIX'])
    high_vol = aligned[aligned['VIX'] >= 20].drop(columns=['VIX'])

    strat_names = returns_df.columns.tolist()
    n = len(strat_names)

    # Optimize separately for each regime
    if len(low_vol) > 60:
        w_low = max_sharpe_weights(low_vol)
    else:
        w_low = np.ones(n) / n

    if len(high_vol) > 60:
        w_high = max_sharpe_weights(high_vol)
    else:
        w_high = np.ones(n) / n

    # Build portfolio returns using regime-appropriate weights
    port_returns = pd.Series(index=aligned.index, dtype=float)
    for idx in aligned.index:
        row = aligned.loc[idx]
        if row['VIX'] < 20:
            port_returns[idx] = sum(w_low[i] * row[strat_names[i]] for i in range(n))
        else:
            port_returns[idx] = sum(w_high[i] * row[strat_names[i]] for i in range(n))

    return port_returns, w_low, w_high, len(low_vol), len(high_vol)


def regime_stratification(returns_series, vix_series):
    """Compute Sharpe in VIX<20 vs VIX>=20 regimes."""
    aligned = pd.DataFrame({'ret': returns_series, 'VIX': vix_series}).dropna()
    if len(aligned) < 60:
        return None

    low = aligned[aligned['VIX'] < 20]['ret']
    high = aligned[aligned['VIX'] >= 20]['ret']

    def sharpe(s):
        if len(s) < 20:
            return None
        return round((s.mean() * ANNUALIZE) / (s.std() * np.sqrt(ANNUALIZE)), 4) if s.std() > 0 else 0

    return {
        "low_vol_sharpe": sharpe(low),
        "high_vol_sharpe": sharpe(high),
        "low_vol_days": len(low),
        "high_vol_days": len(high),
        "regime_gap": round(abs((sharpe(low) or 0) - (sharpe(high) or 0)), 4),
    }


def permutation_test(returns_series, n_perms=1000):
    """Permutation test: shuffle daily returns, compare Sharpe distribution."""
    real_sharpe = returns_series.mean() / returns_series.std() * np.sqrt(ANNUALIZE)
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(returns_series.values)
        s = shuffled.mean() / shuffled.std() * np.sqrt(ANNUALIZE)
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    return {
        "real_sharpe": round(real_sharpe, 4),
        "perm_mean": round(perm_sharpes.mean(), 4),
        "perm_std": round(perm_sharpes.std(), 4),
        "p_value": round(p_value, 4),
        "pass": p_value < 0.05,
    }


def subperiod_stability(returns_series, n_splits=4):
    """Check if Sharpe is stable across sub-periods."""
    r = returns_series.dropna().values
    chunks = np.array_split(r, n_splits)
    sharpes = []
    for c in chunks:
        if len(c) > 20 and c.std() > 0:
            sharpes.append(c.mean() / c.std() * np.sqrt(ANNUALIZE))

    if len(sharpes) < 2:
        return None

    cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else float('inf')
    all_positive = all(s > 0 for s in sharpes)

    return {
        "quarter_sharpes": [round(s, 4) for s in sharpes],
        "cv": round(cv, 4),
        "all_positive": all_positive,
        "pass": cv < 1.0 and all_positive,
    }


def best_n_combinations(returns_df, n_select, vix_series=None):
    """Find the best n-strategy combination from all possible."""
    strat_names = returns_df.columns.tolist()
    if len(strat_names) < n_select:
        return None

    best = None
    best_sharpe = -999

    for combo in combinations(strat_names, n_select):
        sub = returns_df[list(combo)]
        # Equal weight
        port = sub.mean(axis=1)
        sharpe = port.mean() / port.std() * np.sqrt(ANNUALIZE) if port.std() > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best = combo

    if best is None:
        return None

    # Now compute full metrics for the best combo
    sub = returns_df[list(best)]
    port_ew = sub.mean(axis=1)

    # Also try risk-parity for best combo
    cov = sub.cov().values * ANNUALIZE
    rp_w = risk_parity_weights(cov)
    port_rp = (sub.values * rp_w).sum(axis=1)
    port_rp = pd.Series(port_rp, index=sub.index)

    ew_metrics = compute_metrics(port_ew, f"Best-{n_select} EW")
    rp_metrics = compute_metrics(port_rp, f"Best-{n_select} RP")

    result = {
        "strategies": list(best),
        "equal_weight": ew_metrics,
        "risk_parity": rp_metrics,
        "rp_weights": {best[i]: round(rp_w[i], 4) for i in range(len(best))},
    }

    if vix_series is not None:
        regime = regime_stratification(port_ew, vix_series)
        if regime:
            result["regime_stratification"] = regime

    return result


def main():
    print("=" * 70)
    print("PORTFOLIO COMBINATION STUDY V1")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Load strategies ──
    print("\n[1] Loading strategy daily returns...")
    strategy_data = load_strategy_returns()

    if len(strategy_data) < 2:
        print("ERROR: Need at least 2 strategies with daily returns. Aborting.")
        sys.exit(1)

    # Align on overlapping dates
    print(f"\n[2] Aligning {len(strategy_data)} strategies on overlapping dates...")
    returns_df = pd.concat(strategy_data.values(), axis=1).dropna()
    print(f"  Overlapping period: {returns_df.index.min().date()} to {returns_df.index.max().date()}")
    print(f"  Trading days: {len(returns_df)}")
    print(f"  Strategies: {returns_df.columns.tolist()}")

    # ── 2. Individual strategy metrics ──
    print("\n[3] Individual strategy metrics...")
    individual_metrics = {}
    for col in returns_df.columns:
        m = compute_metrics(returns_df[col], col)
        individual_metrics[col] = m
        print(f"  {col:25s} | Sharpe={m['sharpe']:6.3f} | Sortino={m['sortino']:6.3f} | "
              f"CAGR={m['cagr']*100:5.1f}% | MaxDD={m['max_dd']*100:6.1f}% | Calmar={m['calmar']:5.2f}")

    # ── 3. Correlation matrix ──
    print("\n[4] Correlation matrix...")
    corr = returns_df.corr()
    print(corr.round(3).to_string())

    # ── 4. Download VIX for regime analysis ──
    print("\n[5] Getting VIX data for regime analysis...")
    vix = get_vix_data(returns_df.index.min(), returns_df.index.max())

    # ── 5. Portfolio combinations ──
    print("\n[6] Testing portfolio combinations...")
    results = {
        "run_date": datetime.now().isoformat(),
        "overlapping_period": {
            "start": str(returns_df.index.min().date()),
            "end": str(returns_df.index.max().date()),
            "n_days": len(returns_df),
        },
        "strategies": returns_df.columns.tolist(),
        "individual_metrics": individual_metrics,
        "correlation_matrix": corr.round(4).to_dict(),
        "combinations": {},
    }

    n_strats = returns_df.shape[1]
    strat_names = returns_df.columns.tolist()

    # ── 5a. Equal Weight ──
    print("\n  [6a] Equal Weight...")
    port_ew = returns_df.mean(axis=1)
    ew_metrics = compute_metrics(port_ew, "Equal Weight All")
    ew_regime = regime_stratification(port_ew, vix) if vix is not None else None
    results["combinations"]["equal_weight"] = {
        "weights": {s: round(1.0/n_strats, 4) for s in strat_names},
        "metrics": ew_metrics,
        "regime": ew_regime,
    }
    print(f"    Sharpe={ew_metrics['sharpe']:.3f} | Sortino={ew_metrics['sortino']:.3f} | "
          f"CAGR={ew_metrics['cagr']*100:.1f}% | MaxDD={ew_metrics['max_dd']*100:.1f}%")

    # ── 5b. Risk Parity (Inverse Vol) ──
    print("\n  [6b] Risk Parity (Inverse Vol)...")
    cov = returns_df.cov().values * ANNUALIZE
    rp_w = risk_parity_weights(cov)
    port_rp = (returns_df.values * rp_w).sum(axis=1)
    port_rp = pd.Series(port_rp, index=returns_df.index)
    rp_metrics = compute_metrics(port_rp, "Risk Parity")
    rp_regime = regime_stratification(port_rp, vix) if vix is not None else None
    results["combinations"]["risk_parity"] = {
        "weights": {strat_names[i]: round(rp_w[i], 4) for i in range(n_strats)},
        "metrics": rp_metrics,
        "regime": rp_regime,
    }
    print(f"    Weights: {', '.join(f'{s}={rp_w[i]:.1%}' for i, s in enumerate(strat_names))}")
    print(f"    Sharpe={rp_metrics['sharpe']:.3f} | Sortino={rp_metrics['sortino']:.3f} | "
          f"CAGR={rp_metrics['cagr']*100:.1f}% | MaxDD={rp_metrics['max_dd']*100:.1f}%")

    # ── 5c. Max Sharpe Optimization ──
    print("\n  [6c] Max Sharpe Optimization...")
    ms_w = max_sharpe_weights(returns_df)
    port_ms = (returns_df.values * ms_w).sum(axis=1)
    port_ms = pd.Series(port_ms, index=returns_df.index)
    ms_metrics = compute_metrics(port_ms, "Max Sharpe")
    ms_regime = regime_stratification(port_ms, vix) if vix is not None else None
    results["combinations"]["max_sharpe"] = {
        "weights": {strat_names[i]: round(ms_w[i], 4) for i in range(n_strats)},
        "metrics": ms_metrics,
        "regime": ms_regime,
    }
    print(f"    Weights: {', '.join(f'{s}={ms_w[i]:.1%}' for i, s in enumerate(strat_names))}")
    print(f"    Sharpe={ms_metrics['sharpe']:.3f} | Sortino={ms_metrics['sortino']:.3f} | "
          f"CAGR={ms_metrics['cagr']*100:.1f}% | MaxDD={ms_metrics['max_dd']*100:.1f}%")

    # ── 5d. Regime-Switched Weights ──
    if vix is not None:
        print("\n  [6d] Regime-Switched (VIX<20 vs VIX>=20)...")
        port_regime, w_low, w_high, n_low, n_high = regime_switched_weights(returns_df, vix)
        rs_metrics = compute_metrics(port_regime, "Regime-Switched")
        rs_regime = regime_stratification(port_regime, vix)
        results["combinations"]["regime_switched"] = {
            "low_vol_weights": {strat_names[i]: round(w_low[i], 4) for i in range(n_strats)},
            "high_vol_weights": {strat_names[i]: round(w_high[i], 4) for i in range(n_strats)},
            "low_vol_days": n_low,
            "high_vol_days": n_high,
            "metrics": rs_metrics,
            "regime": rs_regime,
        }
        print(f"    Low-vol weights: {', '.join(f'{s}={w_low[i]:.1%}' for i, s in enumerate(strat_names))}")
        print(f"    High-vol weights: {', '.join(f'{s}={w_high[i]:.1%}' for i, s in enumerate(strat_names))}")
        print(f"    Sharpe={rs_metrics['sharpe']:.3f} | Sortino={rs_metrics['sortino']:.3f} | "
              f"CAGR={rs_metrics['cagr']*100:.1f}% | MaxDD={rs_metrics['max_dd']*100:.1f}%")

    # ── 5e. Best-2 and Best-3 combinations ──
    if n_strats >= 3:
        print("\n  [6e] Best-2 combination search...")
        best2 = best_n_combinations(returns_df, 2, vix)
        if best2:
            results["combinations"]["best_2"] = best2
            print(f"    Best pair: {best2['strategies']}")
            print(f"    EW Sharpe={best2['equal_weight']['sharpe']:.3f} | RP Sharpe={best2['risk_parity']['sharpe']:.3f}")

        print("\n  [6f] Best-3 combination search...")
        best3 = best_n_combinations(returns_df, 3, vix)
        if best3:
            results["combinations"]["best_3"] = best3
            print(f"    Best triple: {best3['strategies']}")
            print(f"    EW Sharpe={best3['equal_weight']['sharpe']:.3f} | RP Sharpe={best3['risk_parity']['sharpe']:.3f}")

    # ── 6. Adversarial Validation ──
    print("\n[7] Adversarial validation...")
    adversarial = {}

    # Test the best combination (max sharpe)
    for combo_name, combo_port in [
        ("equal_weight", port_ew),
        ("risk_parity", port_rp),
        ("max_sharpe", port_ms),
    ]:
        print(f"\n  Testing {combo_name}...")

        # Permutation test
        perm = permutation_test(combo_port)
        print(f"    Permutation: p={perm['p_value']:.4f} {'PASS' if perm['pass'] else 'FAIL'}")

        # Sub-period stability
        sub = subperiod_stability(combo_port)
        if sub:
            print(f"    Sub-period: CV={sub['cv']:.3f}, all_pos={sub['all_positive']} "
                  f"{'PASS' if sub['pass'] else 'FAIL'}")

        # Regime balance
        regime = regime_stratification(combo_port, vix) if vix is not None else None
        if regime:
            gap = regime['regime_gap']
            max_s = max(abs(regime['low_vol_sharpe'] or 0), abs(regime['high_vol_sharpe'] or 0))
            regime_ratio = gap / max_s if max_s > 0 else 0
            regime_pass = regime_ratio < 0.50
            print(f"    Regime balance: low_vol={regime['low_vol_sharpe']:.3f}, "
                  f"high_vol={regime['high_vol_sharpe']:.3f}, ratio={regime_ratio:.3f} "
                  f"{'PASS' if regime_pass else 'FAIL'}")

        adversarial[combo_name] = {
            "permutation": perm,
            "subperiod": sub,
            "regime_balance": regime,
        }

    # Also test regime-switched if available
    if vix is not None and 'regime_switched' in results["combinations"]:
        print(f"\n  Testing regime_switched...")
        perm = permutation_test(port_regime)
        sub = subperiod_stability(port_regime)
        print(f"    Permutation: p={perm['p_value']:.4f} {'PASS' if perm['pass'] else 'FAIL'}")
        if sub:
            print(f"    Sub-period: CV={sub['cv']:.3f} {'PASS' if sub['pass'] else 'FAIL'}")
        adversarial["regime_switched"] = {
            "permutation": perm,
            "subperiod": sub,
        }

    results["adversarial"] = adversarial

    # ── 7. Year-by-year breakdown for top methods ──
    print("\n[8] Year-by-year breakdown...")
    yearly = {}
    for combo_name, combo_port in [
        ("equal_weight", port_ew),
        ("risk_parity", port_rp),
        ("max_sharpe", port_ms),
    ]:
        yr_metrics = {}
        for year in sorted(combo_port.index.year.unique()):
            yr_data = combo_port[combo_port.index.year == year]
            if len(yr_data) > 20:
                m = compute_metrics(yr_data, f"{year}")
                yr_metrics[str(year)] = {
                    "sharpe": m["sharpe"],
                    "return": round(yr_data.sum(), 4),
                    "max_dd": m["max_dd"],
                    "n_days": m["n_days"],
                }
        yearly[combo_name] = yr_metrics
    results["yearly"] = yearly

    # Print year-by-year for max_sharpe
    print(f"\n  Max Sharpe year-by-year:")
    for yr, m in yearly.get("max_sharpe", {}).items():
        print(f"    {yr}: Sharpe={m['sharpe']:6.3f} | Return={m['return']*100:6.1f}% | MaxDD={m['max_dd']*100:6.1f}%")

    # ── 8. Summary ranking ──
    print("\n" + "=" * 70)
    print("SUMMARY: All Combinations Ranked by Sharpe")
    print("=" * 70)

    ranking = []
    for combo_name, combo_data in results["combinations"].items():
        m = combo_data.get("metrics")
        if m:
            ranking.append((combo_name, m["sharpe"], m["sortino"], m["cagr"], m["max_dd"], m["calmar"]))

    ranking.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Combination':30s} {'Sharpe':>8s} {'Sortino':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s}")
    print("-" * 72)
    for name, sharpe, sortino, cagr, maxdd, calmar in ranking:
        print(f"{name:30s} {sharpe:8.3f} {sortino:8.3f} {cagr*100:7.1f}% {maxdd*100:7.1f}% {calmar:8.2f}")

    # ── 9. Save results ──
    print(f"\n[9] Saving results to {OUT_DIR}/...")

    # Save JSON results
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save correlation matrix
    corr.to_csv(f"{OUT_DIR}/correlation_matrix.csv")

    # Save portfolio daily returns
    portfolio_returns = pd.DataFrame({
        "equal_weight": port_ew,
        "risk_parity": port_rp,
        "max_sharpe": port_ms,
    })
    if vix is not None and 'regime_switched' in results["combinations"]:
        portfolio_returns["regime_switched"] = port_regime
    portfolio_returns.to_csv(f"{OUT_DIR}/portfolio_daily_returns.csv")

    # Save individual aligned returns
    returns_df.to_csv(f"{OUT_DIR}/aligned_strategy_returns.csv")

    # ── 10. MLflow logging ──
    print("\n[10] Logging to MLflow...")
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("portfolio_combination_v1")

        with mlflow.start_run(run_name=f"combo_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log best combination metrics
            best_combo = ranking[0]
            mlflow.log_param("best_method", best_combo[0])
            mlflow.log_param("n_strategies", n_strats)
            mlflow.log_param("strategies", ",".join(strat_names))
            mlflow.log_param("overlap_days", len(returns_df))
            mlflow.log_param("overlap_start", str(returns_df.index.min().date()))
            mlflow.log_param("overlap_end", str(returns_df.index.max().date()))

            for combo_name, combo_data in results["combinations"].items():
                m = combo_data.get("metrics")
                if m:
                    mlflow.log_metric(f"{combo_name}_sharpe", m["sharpe"])
                    mlflow.log_metric(f"{combo_name}_sortino", m["sortino"])
                    mlflow.log_metric(f"{combo_name}_cagr", m["cagr"])
                    mlflow.log_metric(f"{combo_name}_maxdd", m["max_dd"])
                    mlflow.log_metric(f"{combo_name}_calmar", m["calmar"])

            # Log artifacts
            mlflow.log_artifact(f"{OUT_DIR}/results.json")
            mlflow.log_artifact(f"{OUT_DIR}/correlation_matrix.csv")
            mlflow.log_artifact(f"{OUT_DIR}/portfolio_daily_returns.csv")

            print("  MLflow logged successfully")
    except Exception as e:
        print(f"  MLflow logging failed: {e}")

    print("\nDONE.")
    return results


if __name__ == "__main__":
    main()
