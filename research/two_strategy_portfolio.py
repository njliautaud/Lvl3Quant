#!/usr/bin/env python3
"""
two_strategy_portfolio.py — 2-Strategy Portfolio Optimization

Combines V5 Combined Hedge (CSP wheel + VIX TS sizing + beta hedge)
and ETF v3 Beta Hedge (sector rotation + rolling 60d beta-scaled SPY hedge).

Weight grid: 0-100% in 5% steps. Reports Sharpe, Sortino, CAGR, MaxDD,
Calmar, R1 gap for each allocation. Finds optimal portfolios under
multiple criteria.

Usage:
    python3 /home/jupiter/Lvl3Quant/research/two_strategy_portfolio.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "portfolio_optimization"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE = 0.04
RF_DAILY = RISK_FREE / TRADING_DAYS


# ── Load Data ──────────────────────────────────────────────────────────────
def load_daily_returns():
    """Load daily return series for both strategies, aligned on common dates."""
    # V5 Combined Hedge
    v5 = pd.read_parquet(ROOT / "output" / "v5_combined_hedge" / "equity_curves.parquet")
    v5["date"] = pd.to_datetime(v5["date"])
    v5 = v5.sort_values("date").set_index("date")
    v5_ret = v5["combined"].pct_change()

    # ETF v3 Beta Hedge
    etf = pd.read_parquet(ROOT / "output" / "etf_v3_hedge" / "equity_curves.parquet")
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf.sort_values("date").set_index("date")
    etf_ret = etf["equity_hedged"].pct_change()

    # Align on common dates
    common = v5_ret.index.intersection(etf_ret.index)
    v5_ret = v5_ret.loc[common].fillna(0.0)
    etf_ret = etf_ret.loc[common].fillna(0.0)

    print(f"V5 Combined Hedge: {len(v5_ret)} days, {common[0].date()} to {common[-1].date()}")
    print(f"ETF v3 Beta Hedge: {len(etf_ret)} days")

    return v5_ret, etf_ret, common


def load_spy_close():
    """Load SPY daily close for regime classification."""
    spy = pd.read_parquet(ROOT / "wheel_strategy_v1" / "data" / "cache" / "spy_prices.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date")["close"].sort_index().astype(float)
    return spy


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(daily_ret: pd.Series, spy_close: pd.Series, label: str = "") -> dict:
    """Compute full metrics including R1 regime gap."""
    n = len(daily_ret)
    if n < 20:
        return {"label": label, "error": "too few days"}

    # Build equity curve from returns
    eq = (1 + daily_ret).cumprod()

    # Sharpe
    exc = daily_ret - RF_DAILY
    sharpe = float(exc.mean() / exc.std() * np.sqrt(TRADING_DAYS)) if exc.std() > 0 else 0.0

    # Sortino
    downside = exc[exc < 0]
    ds_std = np.sqrt((downside**2).mean()) if len(downside) > 0 else 1e-9
    sortino = float(exc.mean() / ds_std * np.sqrt(TRADING_DAYS))

    # CAGR
    years = n / TRADING_DAYS
    total_ret = float(eq.iloc[-1] / eq.iloc[0])
    cagr = float(total_ret ** (1.0 / years) - 1.0) if years > 0 else 0.0

    # MaxDD
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    # Win rate / Profit factor
    pnl = daily_ret[daily_ret != 0]
    wr = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")

    # Day concentration
    daily_pnl = eq.diff().dropna()
    total_pnl = daily_pnl.sum()
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else float("nan")

    # ── R1 Regime Split ──
    spy_ret = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret.index)
    labels[spy_ret > 0.002] = "green"
    labels[spy_ret < -0.002] = "red"
    aligned = labels.reindex(daily_ret.index)

    regime = {}
    for reg in ("green", "red", "flat"):
        sub = daily_ret[aligned == reg]
        regime[f"n_{reg}"] = int(len(sub))
        if len(sub) >= 5 and sub.std() > 0:
            exc_r = sub - RF_DAILY
            regime[f"{reg}_sharpe"] = float(exc_r.mean() / exc_r.std() * np.sqrt(TRADING_DAYS))
        else:
            regime[f"{reg}_sharpe"] = float("nan")

    sg = regime.get("green_sharpe", float("nan"))
    sr = regime.get("red_sharpe", float("nan"))
    if not (np.isnan(sg) or np.isnan(sr)):
        denom = max(abs(sg), abs(sr))
        regime_gap = abs(sg - sr) / denom if denom > 0 else float("nan")
        r1_pass = regime_gap <= 0.50
    else:
        regime_gap = float("nan")
        r1_pass = False

    return {
        "label": label,
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "pf": round(pf, 4),
        "wr": round(wr, 4),
        "day_conc": round(day_conc, 4),
        "n_days": n,
        "green_sharpe": round(sg, 4) if not np.isnan(sg) else None,
        "red_sharpe": round(sr, 4) if not np.isnan(sr) else None,
        "flat_sharpe": round(regime.get("flat_sharpe", float("nan")), 4),
        "regime_gap": round(regime_gap, 4) if not np.isnan(regime_gap) else None,
        "r1_pass": r1_pass,
        "n_green": regime.get("n_green", 0),
        "n_red": regime.get("n_red", 0),
        "n_flat": regime.get("n_flat", 0),
    }


# ── Portfolio Sweep ────────────────────────────────────────────────────────
def run_sweep(v5_ret: pd.Series, etf_ret: pd.Series, spy_close: pd.Series):
    """Run weight grid sweep: ETF 0-100% in 5% steps."""
    results = []
    weights = np.arange(0, 105, 5) / 100.0  # 0.00 to 1.00

    for w_etf in weights:
        w_v5 = 1.0 - w_etf
        port_ret = w_v5 * v5_ret + w_etf * etf_ret
        label = f"V5={w_v5:.0%}_ETF={w_etf:.0%}"
        m = compute_metrics(port_ret, spy_close, label)
        m["w_v5"] = round(w_v5, 2)
        m["w_etf"] = round(w_etf, 2)
        results.append(m)

    return results


def find_optima(results: list[dict]) -> dict:
    """Find optimal portfolios under various criteria."""
    passing = [r for r in results if r.get("r1_pass", False)]
    all_sorted = sorted(results, key=lambda x: x.get("sharpe", 0), reverse=True)

    optima = {}

    # Max Sharpe that passes R1
    if passing:
        best_sharpe_r1 = max(passing, key=lambda x: x.get("sharpe", 0))
        optima["max_sharpe_r1_pass"] = best_sharpe_r1
    else:
        optima["max_sharpe_r1_pass"] = None
        optima["note_no_r1"] = "No allocations pass R1"

    # Min gap with Sharpe > 1.5
    sharpe_ok = [r for r in results if r.get("sharpe", 0) > 1.5 and r.get("regime_gap") is not None]
    if sharpe_ok:
        min_gap = min(sharpe_ok, key=lambda x: x["regime_gap"])
        optima["min_gap_sharpe_gt_1_5"] = min_gap

    # Max Calmar that passes R1
    if passing:
        best_calmar_r1 = max(passing, key=lambda x: x.get("calmar", 0))
        optima["max_calmar_r1_pass"] = best_calmar_r1

    # Max Sortino that passes R1
    if passing:
        best_sortino_r1 = max(passing, key=lambda x: x.get("sortino", 0))
        optima["max_sortino_r1_pass"] = best_sortino_r1

    # Overall max Sharpe (regardless of R1)
    optima["max_sharpe_any"] = all_sorted[0]

    return optima


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("2-STRATEGY PORTFOLIO OPTIMIZATION")
    print("V5 Combined Hedge + ETF v3 Beta Hedge")
    print("=" * 70)

    v5_ret, etf_ret, common = load_daily_returns()
    spy_close = load_spy_close()

    # ── Individual Strategy Metrics ──
    print("\n── Individual Strategy Metrics (overlapping period) ──")
    m_v5 = compute_metrics(v5_ret, spy_close, "V5 Combined Hedge")
    m_etf = compute_metrics(etf_ret, spy_close, "ETF v3 Beta Hedge")
    for m in [m_v5, m_etf]:
        print(f"\n  {m['label']}:")
        print(f"    Sharpe={m['sharpe']:.2f}  Sortino={m['sortino']:.2f}  CAGR={m['cagr']:.1%}  MaxDD={m['max_dd']:.1%}")
        print(f"    Calmar={m['calmar']:.2f}  PF={m['pf']:.2f}  WR={m['wr']:.1%}")
        print(f"    Green Sharpe={m['green_sharpe']}  Red Sharpe={m['red_sharpe']}  Gap={m['regime_gap']}  R1={'PASS' if m['r1_pass'] else 'FAIL'}")

    # ── Correlation ──
    corr = float(v5_ret.corr(etf_ret))
    print(f"\n── Hedged Return Correlation: {corr:.4f} ──")

    # Also compute rolling correlation
    roll_corr = v5_ret.rolling(60).corr(etf_ret).dropna()
    print(f"   Rolling 60d correlation: mean={roll_corr.mean():.3f}, "
          f"min={roll_corr.min():.3f}, max={roll_corr.max():.3f}, "
          f"current={roll_corr.iloc[-1]:.3f}")

    # ── Weight Grid Sweep ──
    print("\n── Weight Grid Sweep ──")
    results = run_sweep(v5_ret, etf_ret, spy_close)

    print(f"\n  {'V5':>5} {'ETF':>5} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MaxDD':>7} "
          f"{'Calmar':>7} {'Gap':>6} {'R1':>4}")
    print("  " + "-" * 65)
    for r in results:
        gap_str = f"{r['regime_gap']:.3f}" if r['regime_gap'] is not None else "  N/A"
        r1_str = "PASS" if r['r1_pass'] else "FAIL"
        print(f"  {r['w_v5']:>5.0%} {r['w_etf']:>5.0%} {r['sharpe']:>7.2f} {r['sortino']:>6.2f} "
              f"{r['cagr']:>6.1%} {r['max_dd']:>7.1%} {r['calmar']:>7.2f} {gap_str:>6} {r1_str:>4}")

    # ── Optima ──
    optima = find_optima(results)

    print("\n── OPTIMAL ALLOCATIONS ──")
    for key, val in optima.items():
        if val is None or isinstance(val, str):
            print(f"\n  {key}: {val}")
            continue
        if not isinstance(val, dict):
            continue
        print(f"\n  {key}:")
        print(f"    V5={val['w_v5']:.0%}  ETF={val['w_etf']:.0%}")
        print(f"    Sharpe={val['sharpe']:.2f}  Sortino={val['sortino']:.2f}  CAGR={val['cagr']:.1%}")
        print(f"    MaxDD={val['max_dd']:.1%}  Calmar={val['calmar']:.2f}")
        gap_str = f"{val['regime_gap']:.3f}" if val['regime_gap'] is not None else "N/A"
        print(f"    Green={val['green_sharpe']}  Red={val['red_sharpe']}  Gap={gap_str}  R1={'PASS' if val['r1_pass'] else 'FAIL'}")

    # ── Save Results ──
    output = {
        "meta": {
            "date_range": f"{common[0].date()} to {common[-1].date()}",
            "n_days": len(common),
            "hedged_correlation": round(corr, 4),
            "rolling_60d_corr_mean": round(float(roll_corr.mean()), 4),
            "rolling_60d_corr_min": round(float(roll_corr.min()), 4),
            "rolling_60d_corr_max": round(float(roll_corr.max()), 4),
            "risk_free": RISK_FREE,
        },
        "individual": {
            "v5_combined_hedge": m_v5,
            "etf_v3_beta_hedge": m_etf,
        },
        "sweep": results,
        "optima": {k: v for k, v in optima.items() if isinstance(v, (dict, type(None)))},
    }

    out_path = OUTPUT / "two_strategy.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
