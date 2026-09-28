#!/usr/bin/env python3
"""
Optimal Portfolio v3 — Combined portfolio optimization across all 11 validated strategies.
Uses analytical covariance from stated metrics + correlation assumptions.
Finds optimal weights using 7 allocation methods.
Updated: added VIX Call Spread Income + Quality-Momentum Ranker.
"""

import numpy as np
from scipy.optimize import minimize
import json
from datetime import datetime
from pathlib import Path

np.random.seed(42)

# ─── Strategy definitions ───────────────────────────────────────────────────────
strategies = [
    # Growth
    {"name": "DL Stock Ranker",       "cat": "growth",  "sharpe": 2.37, "cagr": 0.62, "maxdd": -0.138, "wr": 0.84},
    {"name": "ETF Rotation v3",       "cat": "growth",  "sharpe": 2.50, "cagr": 0.15, "maxdd": -0.08,  "wr": 0.70},
    {"name": "Post-Earnings Bounce",  "cat": "growth",  "sharpe": 1.50, "cagr": 0.08, "maxdd": -0.10,  "wr": 0.627},
    # Income
    {"name": "Earnings Jade Lizard",  "cat": "income",  "sharpe": 2.30, "cagr": 0.172, "maxdd": -0.082, "wr": 0.789},
    {"name": "Pre-Earnings Vol Crush","cat": "income",  "sharpe": 1.76, "cagr": 0.548, "maxdd": -0.16,  "wr": 0.836},
    {"name": "Plain Jade Lizard",     "cat": "income",  "sharpe": 1.97, "cagr": 0.121, "maxdd": -0.05,  "wr": 0.75},
    {"name": "Stat Arb",              "cat": "income",  "sharpe": 0.81, "cagr": 0.081, "maxdd": -0.115, "wr": 0.55},
    {"name": "Commodity Trend",       "cat": "income",  "sharpe": 1.20, "cagr": 0.10,  "maxdd": -0.15,  "wr": 0.55},
    {"name": "Carry + Momentum",      "cat": "income",  "sharpe": 0.80, "cagr": 0.06,  "maxdd": -0.20,  "wr": 0.52},
    # New validated strategies (2026-07-24)
    {"name": "VIX Call Spread",       "cat": "income",  "sharpe": 1.75, "cagr": 0.164, "maxdd": -0.096, "wr": 0.835},
    {"name": "Quality-Momentum",      "cat": "growth",  "sharpe": 2.00, "cagr": 0.40,  "maxdd": -0.132, "wr": 0.885},  # Conservative estimate (survivorship-adjusted)
]

N = len(strategies)
names = [s["name"] for s in strategies]

# ─── Derive annualized vol from Sharpe = (CAGR - Rf) / vol ──────────────────────
RF = 0.05
vols = np.array([max((s["cagr"] - RF) / s["sharpe"], 0.02) for s in strategies])
mu = np.array([s["cagr"] for s in strategies])

print("Derived annualized volatilities:")
for i in range(N):
    print(f"  {names[i]:<28} vol={vols[i]:.3f}  mu={mu[i]:.3f}")
print()

# ─── Correlation matrix ─────────────────────────────────────────────────────────
corr = np.full((N, N), 0.10)  # baseline cross-category
np.fill_diagonal(corr, 1.0)

growth_idx = [0, 1, 2, 10]  # DL Ranker, ETF Rot, PEB, QM Ranker
income_idx = [3, 4, 5, 6, 7, 8, 9]  # EJL, PEVC, PJL, StatArb, Commodity, Carry, VIX
earnings_idx = [2, 3, 4]  # PEB, EJL, PEVC
vix_idx = 9  # VIX Call Spread

# Same category: 0.35
for grp in [growth_idx, income_idx]:
    for i in grp:
        for j in grp:
            if i != j:
                corr[i, j] = 0.35

# Cross category: 0.15
for i in growth_idx:
    for j in income_idx:
        corr[i, j] = 0.15
        corr[j, i] = 0.15

# Earnings-timing strategies: 0.40
for i in earnings_idx:
    for j in earnings_idx:
        if i != j:
            corr[i, j] = 0.40

# DL Ranker vs ETF Rotation: 0.30
corr[0, 1] = 0.30
corr[1, 0] = 0.30

# DL Ranker vs QM Ranker: 0.50 (similar stock-picking approach)
corr[0, 10] = 0.50
corr[10, 0] = 0.50

# VIX strategies are negatively correlated with equity growth
for i in growth_idx:
    corr[vix_idx, i] = -0.15
    corr[i, vix_idx] = -0.15

# ─── Covariance matrix ──────────────────────────────────────────────────────────
cov = np.outer(vols, vols) * corr

# Verify PSD
eigvals = np.linalg.eigvalsh(cov)
if np.min(eigvals) < -1e-10:
    print("WARNING: covariance not PSD, applying fix")
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    cov = vecs @ np.diag(vals) @ vecs.T

# ─── Portfolio metric functions ──────────────────────────────────────────────────
def port_return(w):
    return float(w @ mu)

def port_vol(w):
    return float(np.sqrt(w @ cov @ w))

def port_sharpe(w):
    v = port_vol(w)
    return (port_return(w) - RF) / v if v > 1e-10 else 0.0

def port_sortino(w):
    """Approximate: downside vol ~ 0.7 * total vol for diversified portfolio"""
    v = port_vol(w)
    return (port_return(w) - RF) / (v * 0.7) if v > 1e-10 else 0.0

def port_maxdd(w):
    """Approximate MaxDD from components with diversification benefit"""
    component_dd = np.array([abs(s["maxdd"]) for s in strategies])
    # Diversified: sqrt of sum of squared weighted DDs
    diversified = np.sqrt(np.sum((w * component_dd) ** 2))
    linear = float(w @ component_dd)
    return -(0.6 * diversified + 0.4 * linear)

def port_calmar(w):
    dd = abs(port_maxdd(w))
    return port_return(w) / dd if dd > 1e-10 else 0.0

def metrics(w, label):
    ret = port_return(w)
    vol = port_vol(w)
    sharpe = port_sharpe(w)
    sortino = port_sortino(w)
    maxdd = port_maxdd(w)
    calmar = port_calmar(w)
    weights = {names[i]: round(float(w[i]) * 100, 1) for i in range(N) if w[i] > 0.005}
    return {
        "portfolio": label,
        "weights_pct": weights,
        "expected_cagr_pct": round(ret * 100, 1),
        "annual_vol_pct": round(vol * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(maxdd * 100, 1),
        "calmar": round(calmar, 2),
    }

# ─── Optimization helpers ────────────────────────────────────────────────────────
def neg_sharpe(w):
    return -port_sharpe(w)

def port_var(w):
    return float(w @ cov @ w)

cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, 0.40)] * N  # max 40% any single strategy
w0 = np.ones(N) / N

# ─── 7 Portfolio allocations ─────────────────────────────────────────────────────
results = []

# 1. Equal Weight
results.append(metrics(np.ones(N) / N, "Equal Weight"))

# 2. Risk Parity (inverse vol)
inv_vol = 1.0 / vols
w_rp = inv_vol / inv_vol.sum()
results.append(metrics(w_rp, "Risk Parity"))

# 3. Max Sharpe
res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
               options={"maxiter": 2000, "ftol": 1e-14})
w_ms = np.maximum(res.x, 0); w_ms /= w_ms.sum()
results.append(metrics(w_ms, "Max Sharpe"))

# 4. Min Variance
res_mv = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=cons,
                  options={"maxiter": 2000, "ftol": 1e-14})
w_mv = np.maximum(res_mv.x, 0); w_mv /= w_mv.sum()
results.append(metrics(w_mv, "Min Variance"))

# 5. Growth 70 / Income 30
w_gf = np.zeros(N)
for i in growth_idx: w_gf[i] = 0.70 / len(growth_idx)
for i in income_idx: w_gf[i] = 0.30 / len(income_idx)
results.append(metrics(w_gf, "Growth 70 / Income 30"))

# 6. Income 70 / Growth 30
w_if = np.zeros(N)
for i in growth_idx: w_if[i] = 0.30 / len(growth_idx)
for i in income_idx: w_if[i] = 0.70 / len(income_idx)
results.append(metrics(w_if, "Income 70 / Growth 30"))

# 7. Agentic Options-Only (EJL, PEVC, PJL, VIX = indices 3,4,5,9)
options_idx = [3, 4, 5, 9]
def neg_sharpe_opts(w_sub):
    w_full = np.zeros(N)
    for k, idx in enumerate(options_idx):
        w_full[idx] = w_sub[k]
    return -port_sharpe(w_full)

w0_ao = np.ones(len(options_idx)) / len(options_idx)
res_ao = minimize(neg_sharpe_opts, w0_ao, method="SLSQP",
                  bounds=[(0.05, 0.60)] * len(options_idx),
                  constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                  options={"maxiter": 2000})
w_ao = np.zeros(N)
for k, idx in enumerate(options_idx):
    w_ao[idx] = res_ao.x[k]
results.append(metrics(w_ao, "Agentic Options-Only"))

# ─── SPY Benchmark ──────────────────────────────────────────────────────────────
spy_vol = (0.10 - RF) / 0.89  # back out vol from Sharpe
spy = {
    "portfolio": "SPY Benchmark",
    "weights_pct": {"SPY": 100.0},
    "expected_cagr_pct": 10.0,
    "annual_vol_pct": round(spy_vol * 100, 1),
    "sharpe": 0.89,
    "sortino": 1.27,
    "max_drawdown_pct": -33.7,
    "calmar": round(0.10 / 0.337, 2),
}

# ─── Print results ───────────────────────────────────────────────────────────────
print("=" * 110)
print(f"{'OPTIMAL PORTFOLIO v3 — 11 VALIDATED STRATEGIES':^110}")
print("=" * 110)
print()

hdr = f"{'Portfolio':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'Vol':>7} {'MaxDD':>8} {'Calmar':>7}"
print(hdr)
print("-" * len(hdr))

for r in results:
    print(f"{r['portfolio']:<30} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['expected_cagr_pct']:>6.1f}% {r['annual_vol_pct']:>6.1f}% {r['max_drawdown_pct']:>7.1f}% {r['calmar']:>7.2f}")

print("-" * len(hdr))
print(f"{'SPY Benchmark':<30} {spy['sharpe']:>7.2f} {spy['sortino']:>8.2f} {spy['expected_cagr_pct']:>6.1f}% {spy['annual_vol_pct']:>6.1f}% {spy['max_drawdown_pct']:>7.1f}% {spy['calmar']:>7.2f}")
print()

# Best by Sharpe
best = max(results, key=lambda r: r["sharpe"])
print(f"BEST BY SHARPE: {best['portfolio']}")
print(f"  Sharpe {best['sharpe']:.2f} | Sortino {best['sortino']:.2f} | CAGR {best['expected_cagr_pct']:.1f}% | MaxDD {best['max_drawdown_pct']:.1f}% | Calmar {best['calmar']:.2f}")
print()

# Top 3 weights breakdown
top3 = sorted(results, key=lambda r: r["sharpe"], reverse=True)[:3]
for r in top3:
    print(f"--- {r['portfolio']} (Sharpe {r['sharpe']:.2f}) ---")
    for strat, wt in sorted(r["weights_pct"].items(), key=lambda x: -x[1]):
        print(f"  {strat:<28} {wt:>5.1f}%")
    print()

# Sharpe multiples vs SPY
print("ALL PORTFOLIOS vs SPY (Sharpe 0.89):")
for r in sorted(results, key=lambda r: r["sharpe"], reverse=True):
    mult = r["sharpe"] / spy["sharpe"]
    print(f"  {r['portfolio']:<30} Sharpe {r['sharpe']:.2f}  ({mult:.1f}x SPY)")

# ─── Save JSON ───────────────────────────────────────────────────────────────────
output = {
    "generated": datetime.now().isoformat(),
    "risk_free_rate": RF,
    "strategies_input": strategies,
    "derived_vols": {names[i]: round(float(vols[i]), 4) for i in range(N)},
    "correlation_assumptions": {
        "same_category": 0.35,
        "cross_category": 0.15,
        "earnings_timing_mutual": 0.40,
        "dl_ranker_vs_etf_rotation": 0.30,
        "baseline": 0.10,
    },
    "portfolios": results,
    "spy_benchmark": spy,
    "best_portfolio_by_sharpe": best["portfolio"],
}

out_path = Path(__file__).resolve().parent / "research" / "findings" / "optimal_portfolio_v3_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")
