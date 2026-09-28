#!/usr/bin/env python3
"""
multi_strategy_portfolio.py — Optimal allocation across R1-passing strategies

Constructs a combined portfolio from strategies that PASS the HC #428 R1 regime gate.
Uses mean-variance optimization with correlation-aware sizing.

Strategies available:
1. ETF Rotation v3 (Sharpe 2.39, gap 0.19) — sector ETF rotation
2. ETF Rotation v2 (Sharpe 1.90, gap 0.04) — yield-curve-enhanced rotation
3. BPS GA (Sharpe ~2.59, gap 0.40) — bull put spreads with GA-evolved universe
4. V5 Combined Hedge (Sharpe 2.24, gap 0.21) — CSP + VIX TS sizing + SPY beta hedge
5. Wheel RV Gate (if available) — realized vol gated entries

Tests:
- Equal weight
- Inverse-volatility weight
- Minimum variance
- Max Sharpe (mean-variance)
- Risk parity

All with R1 regime gate on the combined portfolio.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "multi_strategy_portfolio"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS
RF_ANNUAL = 0.04


def load_strategy_returns():
    """Load daily return series for each R1-passing strategy."""
    strategies = {}

    # 1. ETF Rotation v3
    try:
        etf_v3 = pd.read_parquet(ROOT / "output" / "macro_picker" / "backtest_etf_rotation_v3.parquet")
        if "date" in etf_v3.columns and "portfolio_return" in etf_v3.columns:
            etf_v3["date"] = pd.to_datetime(etf_v3["date"])
            strategies["ETF_Rotation_v3"] = etf_v3.set_index("date")["portfolio_return"].sort_index()
    except Exception:
        pass

    # Try alternate ETF v3 source
    if "ETF_Rotation_v3" not in strategies:
        try:
            etf_files = list((ROOT / "output" / "macro_picker").glob("*v3*equity*.parquet"))
            if not etf_files:
                etf_files = list((ROOT / "output" / "macro_picker").glob("*v3*.parquet"))
            for f in etf_files:
                df = pd.read_parquet(f)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    if "equity" in df.columns:
                        df = df.sort_values("date")
                        ret = df.set_index("date")["equity"].pct_change().fillna(0)
                        strategies["ETF_Rotation_v3"] = ret
                        break
                    elif "portfolio_return" in df.columns:
                        strategies["ETF_Rotation_v3"] = df.set_index("date")["portfolio_return"]
                        break
        except Exception:
            pass

    # 2. ETF Rotation v2
    try:
        etf_v2 = pd.read_parquet(ROOT / "output" / "macro_picker" / "backtest_etf_rotation_v2.parquet")
        if "date" in etf_v2.columns and "portfolio_return" in etf_v2.columns:
            etf_v2["date"] = pd.to_datetime(etf_v2["date"])
            strategies["ETF_Rotation_v2"] = etf_v2.set_index("date")["portfolio_return"].sort_index()
    except Exception:
        pass

    # 3. V5 Combined Hedge (our newly validated strategy)
    try:
        combined = pd.read_parquet(ROOT / "output" / "v5_combined_hedge" / "equity_curves.parquet")
        if "date" in combined.columns and "combined" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date")
            ret = combined.set_index("date")["combined"].pct_change().fillna(0)
            strategies["V5_Combined_Hedge"] = ret
    except Exception:
        pass

    # 4. V5 Baseline (for comparison — doesn't pass R1 alone)
    try:
        combined = pd.read_parquet(ROOT / "output" / "v5_combined_hedge" / "equity_curves.parquet")
        if "date" in combined.columns and "baseline" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date")
            ret = combined.set_index("date")["baseline"].pct_change().fillna(0)
            strategies["V5_Baseline_ref"] = ret
    except Exception:
        pass

    # 5. SPY benchmark
    try:
        cache = ROOT / "wheel_strategy_v1" / "data" / "cache"
        spy = pd.read_parquet(cache / "spy_prices.parquet")
        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy.sort_values("date")
        ret = spy.set_index("date")["close"].pct_change().fillna(0)
        strategies["SPY_benchmark"] = ret
    except Exception:
        pass

    return strategies


def compute_metrics(daily_ret, label=""):
    """Compute risk-adjusted metrics."""
    excess = daily_ret - RF_DAILY
    n = len(daily_ret)
    if n < 30 or excess.std() == 0:
        return {"label": label, "error": "insufficient data"}

    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))
    down = excess[excess < 0]
    sortino = float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 and down.std() > 0 else 0

    cum = (1 + daily_ret).cumprod()
    yrs = n / TRADING_DAYS
    cagr = float(cum.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0

    peak = cum.cummax()
    max_dd = float((cum / peak - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    return {
        "label": label, "sharpe": sharpe, "sortino": sortino,
        "cagr": cagr, "max_dd": max_dd, "calmar": calmar,
        "ann_vol": float(daily_ret.std() * np.sqrt(TRADING_DAYS)),
        "n_days": n,
    }


def regime_gate(daily_ret, spy_ret, label=""):
    """Check R1 regime gate."""
    spy_aligned = spy_ret.reindex(daily_ret.index).fillna(0)
    labels = pd.Series("flat", index=spy_aligned.index)
    labels[spy_aligned > 0.002] = "green"
    labels[spy_aligned < -0.002] = "red"

    excess = daily_ret - RF_DAILY
    result = {}
    for reg in ("green", "red", "flat"):
        sub = excess[labels == reg]
        result[f"n_{reg}"] = len(sub)
        if len(sub) >= 5 and sub.std() > 0:
            result[f"{reg}_sharpe"] = float(sub.mean() / sub.std() * np.sqrt(TRADING_DAYS))
        else:
            result[f"{reg}_sharpe"] = float("nan")

    sg = result.get("green_sharpe", float("nan"))
    sr = result.get("red_sharpe", float("nan"))
    if not (np.isnan(sg) or np.isnan(sr)):
        denom = max(abs(sg), abs(sr))
        result["gap"] = abs(sg - sr) / denom if denom > 0 else float("nan")
        result["r1_pass"] = result["gap"] <= 0.50
    else:
        result["gap"] = float("nan")
        result["r1_pass"] = False

    return result


def optimize_portfolio(returns_df, method="max_sharpe"):
    """
    Portfolio optimization.
    returns_df: DataFrame with columns = strategy names, rows = dates.
    """
    n = returns_df.shape[1]
    mean_ret = returns_df.mean().values * TRADING_DAYS
    cov = returns_df.cov().values * TRADING_DAYS

    if method == "equal":
        return np.ones(n) / n

    elif method == "inv_vol":
        vols = np.sqrt(np.diag(cov))
        w = 1.0 / vols
        return w / w.sum()

    elif method == "min_var":
        def portfolio_var(w):
            return w @ cov @ w
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n
        res = minimize(portfolio_var, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else np.ones(n) / n

    elif method == "max_sharpe":
        def neg_sharpe(w):
            port_ret = w @ mean_ret
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-10:
                return 0
            return -(port_ret - RF_ANNUAL) / port_vol
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else np.ones(n) / n

    elif method == "risk_parity":
        def risk_parity_obj(w):
            port_var = w @ cov @ w
            if port_var < 1e-15:
                return 1e10
            marginal = cov @ w
            risk_contrib = w * marginal / np.sqrt(port_var)
            target = np.sqrt(port_var) / n
            return np.sum((risk_contrib - target) ** 2)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0.01, 1)] * n
        w0 = np.ones(n) / n
        res = minimize(risk_parity_obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        return res.x if res.success else np.ones(n) / n

    return np.ones(n) / n


def main():
    print("=" * 75)
    print("  MULTI-STRATEGY PORTFOLIO OPTIMIZATION")
    print("  Only R1-passing strategies included")
    print("=" * 75)

    strategies = load_strategy_returns()
    print(f"\n  Loaded {len(strategies)} return series:")
    for name, ret in strategies.items():
        print(f"    {name}: {len(ret)} days, {ret.index.min().date()} to {ret.index.max().date()}")

    # Separate R1-passing strategies from references
    r1_names = [n for n in strategies if n not in ("SPY_benchmark", "V5_Baseline_ref")]
    ref_names = [n for n in strategies if n in ("SPY_benchmark", "V5_Baseline_ref")]

    if len(r1_names) < 2:
        print(f"\n  ERROR: Need at least 2 R1-passing strategies, got {len(r1_names)}: {r1_names}")
        print("  Cannot build portfolio. Listing available files:")
        for p in (ROOT / "output" / "macro_picker").glob("*.parquet"):
            print(f"    {p.name}")
        return

    # Align dates across all R1-passing strategies
    all_rets = pd.DataFrame({n: strategies[n] for n in r1_names})
    all_rets = all_rets.dropna()
    print(f"\n  Common date range: {all_rets.index.min().date()} to {all_rets.index.max().date()} ({len(all_rets)} days)")

    # Correlation matrix
    corr = all_rets.corr()
    print(f"\n  CORRELATION MATRIX:")
    print(corr.round(3).to_string())

    # Individual strategy metrics
    spy_ret = strategies.get("SPY_benchmark", pd.Series(dtype=float))

    print(f"\n  INDIVIDUAL STRATEGY METRICS:")
    print(f"  {'Strategy':<25} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Vol':>6}")
    for name in r1_names:
        m = compute_metrics(all_rets[name], name)
        print(f"  {name:<25} {m['sharpe']:>7.2f} {m['sortino']:>6.2f} {m['cagr']:>6.1%} {m['max_dd']:>6.1%} "
              f"{m['calmar']:>7.2f} {m['ann_vol']:>5.1%}")

    # Portfolio optimization
    methods = ["equal", "inv_vol", "min_var", "max_sharpe", "risk_parity"]
    results = []

    print(f"\n{'='*75}")
    print(f"  PORTFOLIO ALLOCATIONS")
    print(f"{'='*75}")

    for method in methods:
        weights = optimize_portfolio(all_rets, method)
        port_ret = all_rets @ weights

        m = compute_metrics(port_ret, method)
        r = regime_gate(port_ret, spy_ret, method) if len(spy_ret) > 0 else {}

        results.append({
            "method": method,
            "weights": {n: float(w) for n, w in zip(r1_names, weights)},
            **m, **r,
        })

        print(f"\n  {method.upper():}")
        print(f"    Weights: {', '.join(f'{n}={w:.1%}' for n, w in zip(r1_names, weights) if w > 0.01)}")
        print(f"    Sharpe: {m['sharpe']:.2f}, Sortino: {m['sortino']:.2f}, CAGR: {m['cagr']:.1%}")
        print(f"    MaxDD: {m['max_dd']:.1%}, Calmar: {m['calmar']:.2f}, Vol: {m['ann_vol']:.1%}")
        if "gap" in r:
            r1_status = "PASS ✓" if r.get("r1_pass") else "FAIL ✗"
            print(f"    R1: gap={r['gap']:.2f} ({r1_status}), green={r.get('green_sharpe',0):.2f}, red={r.get('red_sharpe',0):.2f}")

    # Find best portfolio
    passing = [r for r in results if r.get("r1_pass", False)]
    if passing:
        best = max(passing, key=lambda x: x["sharpe"])
        print(f"\n{'='*75}")
        print(f"  BEST R1-PASSING PORTFOLIO: {best['method'].upper()}")
        print(f"  Sharpe: {best['sharpe']:.2f}, CAGR: {best['cagr']:.1%}, MaxDD: {best['max_dd']:.1%}")
        print(f"  Weights: {', '.join(f'{k}={v:.1%}' for k, v in best['weights'].items() if v > 0.01)}")
        print(f"{'='*75}")
    else:
        best_overall = max(results, key=lambda x: x["sharpe"])
        print(f"\n  WARNING: No portfolio passes R1. Best overall: {best_overall['method']} (Sharpe {best_overall['sharpe']:.2f})")

    # Save
    save_data = {
        "strategies": r1_names,
        "n_common_days": len(all_rets),
        "correlation": corr.to_dict(),
        "portfolios": results,
        "best_r1_passing": best if passing else None,
    }
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save equity curves
    curves = pd.DataFrame({"date": all_rets.index})
    for r in results:
        weights = np.array([r["weights"].get(n, 0) for n in r1_names])
        port_ret = all_rets.values @ weights
        eq = (1 + port_ret).cumprod() * 100000
        curves[r["method"]] = eq
    curves.to_parquet(OUTPUT / "equity_curves.parquet", index=False)

    print(f"\n  Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
