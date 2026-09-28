#!/usr/bin/env python3
"""
Sector Seasonality Backtest — Calendar-Based Rotation
=====================================================
Universe: 11 sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
OOT: Jan 2022 – Jul 2026
Starting capital: $645, max $300/sector, max 2-3 positions
6 variants (A-F) with permutation testing and 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── Config ───────────────────────────────────────────────────────────────────
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
START = "2016-01-01"
OOT_START = "2022-01-03"
OOT_END = "2026-07-28"
INITIAL_CAPITAL = 645.0
MAX_PER_SECTOR = 300.0
SLIPPAGE = 0.0002  # 0.02%
N_PERMS = 1000
RISK_FREE = 0.04
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/sector_seasonality_results.json")

# ─── Data download ───────────────────────────────────────────────────────────
print("Downloading sector ETF data (2016-2026)...")
tickers = SECTORS + ["SPY"]
raw = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

close = raw["Close"].copy()
close = close.ffill()

valid_sectors = [s for s in SECTORS if s in close.columns and close[s].notna().sum() > 252]
print(f"Valid sectors: {len(valid_sectors)}: {valid_sectors}")

spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)

# OOT mask
oot_mask = close.index >= OOT_START
dates_oot = close.index[oot_mask].tolist()
n_oot = len(dates_oot)
print(f"OOT period: {dates_oot[0].date()} to {dates_oot[-1].date()}, {n_oot} days")
print(f"Bull days: {regime.loc[oot_mask].sum()}, Bear days: {(~regime.loc[oot_mask].astype(bool)).sum()}")

# ─── Pre-compute monthly returns for seasonal averages (pre-OOT only) ───────
pre_oot = close.loc[close.index < OOT_START, valid_sectors]
monthly_close = pre_oot.resample("ME").last()
monthly_ret = monthly_close.pct_change().dropna()
monthly_ret["month"] = monthly_ret.index.month

# Average return per sector per month (2016-2021)
seasonal_avg = monthly_ret.groupby("month")[valid_sectors].mean()
print("\nHistorical monthly sector averages (2016-2021):")
print(seasonal_avg.round(4).to_string())


# ─── Backtest engine ─────────────────────────────────────────────────────────
def run_backtest(variant_name, get_holdings_fn, oot_close, regime_series):
    """
    Run a monthly-rebalancing backtest.
    get_holdings_fn(year, month) -> dict of {sector: weight} (weights sum to 1)
    Returns equity curve (Series), list of trades.
    """
    # Resample to monthly for rebalancing
    monthly = oot_close.resample("ME").last()
    # Add the first day
    first_day = oot_close.iloc[0:1]
    monthly = pd.concat([first_day, monthly]).drop_duplicates()

    capital = INITIAL_CAPITAL
    holdings = {}  # {sector: n_shares}
    trades = []
    equity = []
    equity_dates = []

    for i in range(len(monthly) - 1):
        date = monthly.index[i]
        next_date = monthly.index[i + 1]
        year, month = date.year, date.month

        # Get target allocation
        target = get_holdings_fn(year, month)

        # Calculate current portfolio value
        port_value = capital
        for sec, shares in holdings.items():
            if sec in monthly.columns:
                port_value += shares * monthly.loc[date, sec]

        # Sell everything
        for sec, shares in holdings.items():
            if sec in monthly.columns and shares > 0:
                sell_price = monthly.loc[date, sec] * (1 - SLIPPAGE)
                capital += shares * sell_price
        holdings = {}

        # Buy new positions
        for sec, weight in target.items():
            if sec not in monthly.columns:
                continue
            alloc = min(port_value * weight, MAX_PER_SECTOR)
            buy_price = monthly.loc[date, sec] * (1 + SLIPPAGE)
            shares = alloc / buy_price
            if shares > 0:
                capital -= shares * buy_price
                holdings[sec] = shares
                trades.append({
                    "date": str(date.date()),
                    "sector": sec,
                    "action": "BUY",
                    "shares": round(shares, 4),
                    "price": round(buy_price, 2)
                })

        # Record daily equity through this month
        daily_slice = oot_close.loc[(oot_close.index >= date) & (oot_close.index < next_date)]
        for d in daily_slice.index:
            val = capital
            for sec, shares in holdings.items():
                if sec in daily_slice.columns:
                    val += shares * daily_slice.loc[d, sec]
            equity.append(val)
            equity_dates.append(d)

    # Handle last period
    if len(monthly) > 0:
        last_date = monthly.index[-1]
        remaining = oot_close.loc[oot_close.index >= last_date]
        for d in remaining.index:
            val = capital
            for sec, shares in holdings.items():
                if sec in remaining.columns:
                    val += shares * remaining.loc[d, sec]
            equity.append(val)
            equity_dates.append(d)

    eq = pd.Series(equity, index=equity_dates)
    eq = eq[~eq.index.duplicated(keep='last')]
    return eq, trades


def compute_metrics(equity_curve, regime_series, trades):
    """Compute performance metrics from equity curve."""
    daily_ret = equity_curve.pct_change().dropna()
    if len(daily_ret) < 10:
        return None

    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = (ann_ret - RISK_FREE) / downside if downside > 0 else 0

    # Max drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    mdd = drawdown.min()

    # Profit factor
    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Win rate (monthly)
    monthly_ret = equity_curve.resample("ME").last().pct_change().dropna()
    wr = (monthly_ret > 0).mean()

    # Regime analysis
    aligned_regime = regime_series.reindex(daily_ret.index).ffill()
    bull_ret = daily_ret[aligned_regime == 1]
    bear_ret = daily_ret[aligned_regime == 0]

    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_ret) > 20:
        bull_ann = bull_ret.mean() * 252
        bull_vol = bull_ret.std() * np.sqrt(252)
        bull_sharpe = (bull_ann - RISK_FREE) / bull_vol if bull_vol > 0 else 0
    if len(bear_ret) > 20:
        bear_ann = bear_ret.mean() * 252
        bear_vol = bear_ret.std() * np.sqrt(252)
        bear_sharpe = (bear_ann - RISK_FREE) / bear_vol if bear_vol > 0 else 0

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

    n_trades = len(trades)

    return {
        "total_return": round(total_ret * 100, 2),
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr * 100, 1),
        "max_drawdown": round(mdd * 100, 2),
        "n_trades": n_trades,
        "final_equity": round(equity_curve.iloc[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
    }


def permutation_test(equity_curve, get_holdings_fn, oot_close, regime_series, n_perms=N_PERMS):
    """
    Permutation test: shuffle month->sector assignments to test if seasonal
    mapping matters. Returns p-value.
    """
    actual_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    count_better = 0
    months = list(range(1, 13))

    for _ in range(n_perms):
        # Shuffle the month mapping (permute which month maps to which)
        shuffled = list(np.random.permutation(months))
        month_map = {m: shuffled[i] for i, m in enumerate(months)}

        def shuffled_fn(year, month, _map=month_map, _orig=get_holdings_fn):
            return _orig(year, _map.get(month, month))

        eq, _ = run_backtest("perm", shuffled_fn, oot_close, regime_series)
        if len(eq) > 0:
            perm_ret = eq.iloc[-1] / eq.iloc[0] - 1
            if perm_ret >= actual_ret:
                count_better += 1

    return count_better / n_perms


def validate_5gates(metrics, perm_p):
    """Apply 5 validation gates."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    return gates


# ─── Define the 6 variants ──────────────────────────────────────────────────

# A) Energy Seasonality: XLE Sep-Feb, XLU Mar-Aug
def variant_a(year, month):
    if month in [9, 10, 11, 12, 1, 2]:
        return {"XLE": 1.0}
    else:
        return {"XLU": 1.0}

# B) Consumer Holiday: XLY Oct-Jan, XLP Feb-Sep
def variant_b(year, month):
    if month in [10, 11, 12, 1]:
        return {"XLY": 1.0}
    else:
        return {"XLP": 1.0}

# C) Tech Cycle: XLK Jan-Apr, XLV May-Aug, XLF Sep-Dec
def variant_c(year, month):
    if month in [1, 2, 3, 4]:
        return {"XLK": 1.0}
    elif month in [5, 6, 7, 8]:
        return {"XLV": 1.0}
    else:
        return {"XLF": 1.0}

# D) Best-Month-Per-Sector: each month buy the sector with best historical avg
def variant_d(year, month):
    best_sector = seasonal_avg.loc[month].idxmax()
    return {best_sector: 1.0}

# E) Combined Top-2 Seasonal: each month buy top 2 sectors by historical avg
def variant_e(year, month):
    top2 = seasonal_avg.loc[month].nlargest(2).index.tolist()
    return {top2[0]: 0.5, top2[1]: 0.5}

# F) Adversarial: Random sector assignments per quarter
rng_adv = np.random.RandomState(123)
random_quarter_map = {}
for q in range(1, 5):
    random_quarter_map[q] = rng_adv.choice(valid_sectors)

def variant_f(year, month):
    quarter = (month - 1) // 3 + 1
    sector = random_quarter_map[quarter]
    return {sector: 1.0}


# ─── Run all variants ────────────────────────────────────────────────────────
oot_close = close.loc[oot_mask, valid_sectors]

variants = {
    "A_Energy_Seasonality": variant_a,
    "B_Consumer_Holiday": variant_b,
    "C_Tech_Cycle": variant_c,
    "D_Best_Month_Sector": variant_d,
    "E_Top2_Seasonal": variant_e,
    "F_Adversarial_Random": variant_f,
}

# SPY benchmark
spy_oot = spy_close.loc[oot_mask]
spy_ret = spy_oot.pct_change().dropna()
spy_total = (spy_oot.iloc[-1] / spy_oot.iloc[0] - 1) * 100
spy_ann = ((1 + spy_total/100) ** (252/len(spy_ret)) - 1) * 100
spy_vol = spy_ret.std() * np.sqrt(252) * 100
spy_sharpe = (spy_ann/100 - RISK_FREE) / (spy_vol/100) if spy_vol > 0 else 0
print(f"\nSPY Benchmark: Total {spy_total:.1f}%, Ann {spy_ann:.1f}%, Sharpe {spy_sharpe:.2f}")

results = {"strategy": "Sector Seasonality (Calendar-Based Rotation)", "timestamp": datetime.now().isoformat()}
results["benchmark"] = {
    "total_return": round(spy_total, 2),
    "ann_return": round(spy_ann, 2),
    "sharpe": round(spy_sharpe, 3)
}
results["seasonal_averages"] = {str(m): {s: round(v, 5) for s, v in seasonal_avg.loc[m].items()} for m in range(1, 13)}
results["variants"] = {}

for name, fn in variants.items():
    print(f"\n{'='*60}")
    print(f"Running {name}...")

    eq, trades = run_backtest(name, fn, oot_close, regime)
    metrics = compute_metrics(eq, regime, trades)

    if metrics is None:
        print(f"  SKIP — insufficient data")
        results["variants"][name] = {"status": "insufficient_data"}
        continue

    print(f"  Return: {metrics['total_return']:.1f}%, Sharpe: {metrics['sharpe']:.3f}, "
          f"Sortino: {metrics['sortino']:.3f}, PF: {metrics['profit_factor']:.3f}, "
          f"WR: {metrics['win_rate']:.1f}%, MDD: {metrics['max_drawdown']:.1f}%")
    print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}, Bear Sharpe: {metrics['bear_sharpe']:.3f}, "
          f"Regime Gap: {metrics['regime_gap']:.3f}")
    print(f"  Trades: {metrics['n_trades']}, Final equity: ${metrics['final_equity']:.2f}")

    # Permutation test
    print(f"  Running permutation test ({N_PERMS} perms)...", end=" ", flush=True)
    perm_p = permutation_test(eq, fn, oot_close, regime)
    print(f"p={perm_p:.4f}")

    # 5-gate validation
    gates = validate_5gates(metrics, perm_p)
    passed = sum(gates.values())
    total_gates = len(gates)
    verdict = "PASS" if passed == total_gates else "FAIL"

    print(f"  Gates: {passed}/{total_gates} — {verdict}")
    for g, v in gates.items():
        status = "PASS" if v else "FAIL"
        print(f"    {g}: {status}")

    results["variants"][name] = {
        "metrics": metrics,
        "perm_p_value": round(perm_p, 4),
        "gates": {k: bool(v) for k, v in gates.items()},
        "gates_passed": f"{passed}/{total_gates}",
        "verdict": verdict,
    }

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Variant':<25} {'Sharpe':>7} {'Return':>8} {'MDD':>7} {'PF':>6} {'WR':>6} {'Perm-p':>7} {'Gates':>6} {'Verdict':>8}")
print("-" * 85)
for name, data in results["variants"].items():
    if "metrics" not in data:
        continue
    m = data["metrics"]
    print(f"{name:<25} {m['sharpe']:>7.3f} {m['total_return']:>7.1f}% {m['max_drawdown']:>6.1f}% "
          f"{m['profit_factor']:>6.2f} {m['win_rate']:>5.1f}% {data['perm_p_value']:>7.4f} "
          f"{data['gates_passed']:>6} {data['verdict']:>8}")

print(f"\nSPY: Sharpe {spy_sharpe:.3f}, Return {spy_total:.1f}%")

# Overall conclusion
any_pass = any(d.get("verdict") == "PASS" for d in results["variants"].values())
results["conclusion"] = "SOME VARIANTS PASSED" if any_pass else "ALL VARIANTS FAILED — no robust seasonal edge"
print(f"\nConclusion: {results['conclusion']}")

# Save
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {RESULTS_PATH}")
