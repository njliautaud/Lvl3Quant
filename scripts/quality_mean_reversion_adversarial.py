#!/usr/bin/env python3
"""
Adversarial Validation: Quality Mean Reversion Strategy
========================================================
6 adversarial tests × 3 variants (A, D, F) = 18 total checks.

Variants:
  A: Buy when stock drops >5% from 20-day high AND RSI(14) < 35. Hold 10 days.
  D: Buy when stock's 5-day return is in bottom 10% of 252-day distribution. Hold 10 days.
  F: Buy when BOTH drop >5% from 20d high AND price > 1 std dev below 20d mean. Hold 15 days.

Tests:
  1) Inverse Signal — buy expensive/overbought instead of cheap/oversold
  2) Random Entry Timing — 1000 random entry sets, percentile rank
  3) Sub-Period Stability — 4 sub-periods, all must have positive Sharpe
  4) Remove Top 3 Tickers — re-run without 3 best performers
  5) Parameter Sensitivity — % of parameter grid with Sharpe > 0.3
  6) Cost Sensitivity — at what slippage does Sharpe < 0.3?

OOT: Jan 2022 – Jul 2026. $645 capital. 0.02% slippage baseline.
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

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
DATA_START = "2020-01-01"  # need lookback for indicators
DATA_END = "2026-07-31"
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP",
    "HD", "COST", "UNH", "LLY", "V", "MA", "ABBV", "MRK",
    "WMT", "AMZN", "GOOGL", "META",
]

SUB_PERIODS = [
    ("P1: Jan2022-Jul2023", "2022-01-01", "2023-07-01"),
    ("P2: Jul2023-Jan2024", "2023-07-01", "2024-01-01"),
    ("P3: Jan2024-Jul2025", "2024-01-01", "2025-07-01"),
    ("P4: Jul2025-Jul2026", "2025-07-01", "2026-07-31"),
]

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(TICKERS, start=DATA_START, end=DATA_END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in TICKERS}
loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def calc_drawdown_from_high(series, window=20):
    """Percentage drop from rolling high."""
    rolling_high = series.rolling(window).max()
    return (series - rolling_high) / rolling_high


def calc_percentile_rank(series, ret_window=5, lookback=252):
    """Where is the current N-day return in its historical distribution?"""
    ret = series.pct_change(ret_window)
    rank = ret.rolling(lookback).apply(
        lambda x: (x[-1] <= x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True
    )
    return ret, rank


def calc_std_below_mean(series, window=20):
    """How many std devs below the rolling mean."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


# ── Precompute Indicators ────────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes.get(t)
    if c is None or len(c) < 300:
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["rsi14"] = calc_rsi(c, 14)
    ind["dd_20"] = calc_drawdown_from_high(c, 20)
    ret5, pctile = calc_percentile_rank(c, 5, 252)
    ind["ret5"] = ret5
    ind["pctile_rank"] = pctile
    ind["z_score_20"] = calc_std_below_mean(c, 20)
    # Forward returns for different hold periods
    for h in [10, 15, 20, 25]:
        ind[f"fwd_ret_{h}"] = c.pct_change(h).shift(-h)
    indicators[t] = ind.dropna(subset=["rsi14", "dd_20"])

valid_tickers = sorted(indicators.keys())
print(f"  Valid tickers for analysis: {len(valid_tickers)}")


# ── Signal Generation Functions ───────────────────────────────────────────
def gen_signals_A(tickers_list, oot_start=OOT_START, oot_end=OOT_END,
                  rsi_thresh=35, dd_thresh=-0.05, hold=10, inverse=False):
    """Variant A: drawdown > dd_thresh from 20d high AND RSI < rsi_thresh."""
    trades = []
    for t in tickers_list:
        if t not in indicators:
            continue
        df = indicators[t]
        mask = (df.index >= oot_start) & (df.index <= oot_end)
        df_oot = df[mask]
        for idx, row in df_oot.iterrows():
            if inverse:
                # Buy when EXPENSIVE: near highs and overbought
                cond = (row["dd_20"] > -0.01) and (row["rsi14"] > (100 - rsi_thresh))
            else:
                cond = (row["dd_20"] <= dd_thresh) and (row["rsi14"] < rsi_thresh)
            if cond and not np.isnan(row.get(f"fwd_ret_{hold}", np.nan)):
                trades.append({
                    "ticker": t, "date": idx, "hold": hold,
                    "fwd_ret": row[f"fwd_ret_{hold}"],
                })
    return trades


def gen_signals_D(tickers_list, oot_start=OOT_START, oot_end=OOT_END,
                  pctile_thresh=0.10, hold=10, inverse=False):
    """Variant D: 5-day return in bottom pctile_thresh of 252-day distribution."""
    trades = []
    for t in tickers_list:
        if t not in indicators:
            continue
        df = indicators[t]
        mask = (df.index >= oot_start) & (df.index <= oot_end)
        df_oot = df[mask]
        for idx, row in df_oot.iterrows():
            pr = row.get("pctile_rank", np.nan)
            if np.isnan(pr):
                continue
            if inverse:
                cond = pr >= (1.0 - pctile_thresh)  # Top percentile = expensive
            else:
                cond = pr <= pctile_thresh
            if cond and not np.isnan(row.get(f"fwd_ret_{hold}", np.nan)):
                trades.append({
                    "ticker": t, "date": idx, "hold": hold,
                    "fwd_ret": row[f"fwd_ret_{hold}"],
                })
    return trades


def gen_signals_F(tickers_list, oot_start=OOT_START, oot_end=OOT_END,
                  dd_thresh=-0.05, z_thresh=-1.0, hold=15, inverse=False):
    """Variant F: drawdown AND z-score below mean."""
    trades = []
    for t in tickers_list:
        if t not in indicators:
            continue
        df = indicators[t]
        mask = (df.index >= oot_start) & (df.index <= oot_end)
        df_oot = df[mask]
        for idx, row in df_oot.iterrows():
            z = row.get("z_score_20", np.nan)
            if np.isnan(z):
                continue
            if inverse:
                cond = (row["dd_20"] > -0.01) and (z > abs(z_thresh))
            else:
                cond = (row["dd_20"] <= dd_thresh) and (z <= z_thresh)
            if cond and not np.isnan(row.get(f"fwd_ret_{hold}", np.nan)):
                trades.append({
                    "ticker": t, "date": idx, "hold": hold,
                    "fwd_ret": row[f"fwd_ret_{hold}"],
                })
    return trades


# ── Portfolio Metrics ─────────────────────────────────────────────────────
def calc_sharpe(trades, slippage=SLIPPAGE_PCT):
    """Calculate annualized Sharpe from trade list."""
    if not trades or len(trades) < 5:
        return -99.0
    rets = np.array([t["fwd_ret"] - 2 * slippage for t in trades])
    if rets.std() == 0:
        return 0.0
    avg_hold = np.mean([t["hold"] for t in trades])
    trades_per_year = 252.0 / max(avg_hold, 1)
    return float((rets.mean() / rets.std()) * np.sqrt(trades_per_year))


def calc_metrics(trades, slippage=SLIPPAGE_PCT):
    """Full metrics suite."""
    if not trades or len(trades) < 5:
        return {"sharpe": -99.0, "n_trades": len(trades) if trades else 0,
                "win_rate": 0, "avg_ret": 0, "pf": 0}
    rets = np.array([t["fwd_ret"] - 2 * slippage for t in trades])
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 99.0
    avg_hold = np.mean([t["hold"] for t in trades])
    trades_per_year = 252.0 / max(avg_hold, 1)
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(trades_per_year)) if rets.std() > 0 else 0.0
    return {
        "sharpe": round(sharpe, 3),
        "n_trades": len(trades),
        "win_rate": round(float(len(wins) / len(rets)), 3),
        "avg_ret_pct": round(float(rets.mean() * 100), 3),
        "profit_factor": round(pf, 3),
    }


def ticker_pnl(trades, slippage=SLIPPAGE_PCT):
    """Return dict of ticker -> total P&L contribution."""
    pnl = {}
    for t in trades:
        r = t["fwd_ret"] - 2 * slippage
        pnl[t["ticker"]] = pnl.get(t["ticker"], 0) + r
    return pnl


# ── Signal generators by variant name ─────────────────────────────────────
SIGNAL_FUNCS = {"A": gen_signals_A, "D": gen_signals_D, "F": gen_signals_F}
DEFAULT_PARAMS = {
    "A": {"rsi_thresh": 35, "dd_thresh": -0.05, "hold": 10},
    "D": {"pctile_thresh": 0.10, "hold": 10},
    "F": {"dd_thresh": -0.05, "z_thresh": -1.0, "hold": 15},
}


def gen_signals(variant, tickers_list=None, oot_start=OOT_START, oot_end=OOT_END,
                inverse=False, **overrides):
    if tickers_list is None:
        tickers_list = valid_tickers
    params = {**DEFAULT_PARAMS[variant]}
    params.update(overrides)
    params["inverse"] = inverse
    params["oot_start"] = oot_start
    params["oot_end"] = oot_end
    return SIGNAL_FUNCS[variant](tickers_list, **params)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE SIGNAL
# ══════════════════════════════════════════════════════════════════════════
def test_inverse(variant):
    print(f"  [{variant}] Test 1: Inverse Signal ...")
    normal = gen_signals(variant)
    inverse = gen_signals(variant, inverse=True)
    normal_m = calc_metrics(normal)
    inverse_m = calc_metrics(inverse)
    # PASS if inverse Sharpe is meaningfully worse than normal
    passed = inverse_m["sharpe"] < normal_m["sharpe"] * 0.5
    return {
        "test": "inverse_signal",
        "passed": passed,
        "normal_sharpe": normal_m["sharpe"],
        "inverse_sharpe": inverse_m["sharpe"],
        "normal_trades": normal_m["n_trades"],
        "inverse_trades": inverse_m["n_trades"],
        "verdict": "PASS — inverse much worse" if passed else "FAIL — inverse also works, edge may be in universe not timing",
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM ENTRY TIMING (1000 permutations)
# ══════════════════════════════════════════════════════════════════════════
def test_random_timing(variant):
    print(f"  [{variant}] Test 2: Random Entry Timing ({N_PERM} iterations) ...")
    actual_trades = gen_signals(variant)
    actual_sharpe = calc_sharpe(actual_trades)
    n_trades = len(actual_trades)

    if n_trades < 10:
        return {
            "test": "random_timing",
            "passed": False,
            "actual_sharpe": actual_sharpe,
            "percentile": 0,
            "n_trades": n_trades,
            "verdict": "SKIP — too few trades",
        }

    # Build pool of all possible (ticker, date, hold, fwd_ret) from OOT
    hold = actual_trades[0]["hold"]
    pool = []
    for t in valid_tickers:
        df = indicators[t]
        mask = (df.index >= OOT_START) & (df.index <= OOT_END)
        df_oot = df[mask]
        for idx, row in df_oot.iterrows():
            fr = row.get(f"fwd_ret_{hold}", np.nan)
            if not np.isnan(fr):
                pool.append({"ticker": t, "date": idx, "hold": hold, "fwd_ret": fr})

    if len(pool) < n_trades:
        return {
            "test": "random_timing",
            "passed": False,
            "verdict": "SKIP — pool too small",
        }

    random_sharpes = []
    for _ in range(N_PERM):
        sample = np.random.choice(len(pool), size=n_trades, replace=False)
        random_trades = [pool[i] for i in sample]
        random_sharpes.append(calc_sharpe(random_trades))

    random_sharpes = np.array(random_sharpes)
    percentile = float((random_sharpes < actual_sharpe).mean() * 100)
    passed = percentile >= 80  # actual must beat 80%+ of random

    return {
        "test": "random_timing",
        "passed": passed,
        "actual_sharpe": round(actual_sharpe, 3),
        "percentile": round(percentile, 1),
        "random_mean_sharpe": round(float(random_sharpes.mean()), 3),
        "random_p95_sharpe": round(float(np.percentile(random_sharpes, 95)), 3),
        "n_trades": n_trades,
        "verdict": f"{'PASS' if passed else 'FAIL'} — actual at {percentile:.0f}th percentile of random",
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
def test_sub_period(variant):
    print(f"  [{variant}] Test 3: Sub-Period Stability ...")
    results = {}
    all_positive = True
    for label, start, end in SUB_PERIODS:
        trades = gen_signals(variant, oot_start=start, oot_end=end)
        m = calc_metrics(trades)
        results[label] = {
            "sharpe": m["sharpe"],
            "n_trades": m["n_trades"],
            "win_rate": m.get("win_rate", 0),
        }
        if m["sharpe"] <= 0 or m["n_trades"] < 3:
            all_positive = False

    return {
        "test": "sub_period_stability",
        "passed": all_positive,
        "periods": results,
        "verdict": "PASS — all sub-periods positive" if all_positive else "FAIL — at least one sub-period negative or insufficient trades",
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: REMOVE TOP 3 TICKERS
# ══════════════════════════════════════════════════════════════════════════
def test_remove_top3(variant):
    print(f"  [{variant}] Test 4: Remove Top 3 Tickers ...")
    full_trades = gen_signals(variant)
    full_sharpe = calc_sharpe(full_trades)
    full_metrics = calc_metrics(full_trades)

    # Find top 3 tickers by total P&L contribution
    pnl = ticker_pnl(full_trades)
    top3 = sorted(pnl, key=pnl.get, reverse=True)[:3]

    reduced_tickers = [t for t in valid_tickers if t not in top3]
    reduced_trades = gen_signals(variant, tickers_list=reduced_tickers)
    reduced_sharpe = calc_sharpe(reduced_trades)
    reduced_metrics = calc_metrics(reduced_trades)

    if full_sharpe > 0:
        drop_pct = (full_sharpe - reduced_sharpe) / full_sharpe * 100
    else:
        drop_pct = 0

    passed = drop_pct < 50  # Sharpe should not drop >50%

    return {
        "test": "remove_top3_tickers",
        "passed": passed,
        "removed_tickers": top3,
        "full_sharpe": round(full_sharpe, 3),
        "reduced_sharpe": round(reduced_sharpe, 3),
        "sharpe_drop_pct": round(drop_pct, 1),
        "full_trades": full_metrics["n_trades"],
        "reduced_trades": reduced_metrics["n_trades"],
        "verdict": f"{'PASS' if passed else 'FAIL'} — Sharpe drop {drop_pct:.1f}% after removing {top3}",
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
def test_param_sensitivity(variant):
    print(f"  [{variant}] Test 5: Parameter Sensitivity ...")

    if variant == "A":
        rsi_vals = [25, 30, 35, 40, 45]
        dd_vals = [-0.03, -0.05, -0.07, -0.10]
        grid = []
        for rsi in rsi_vals:
            for dd in dd_vals:
                trades = gen_signals(variant, rsi_thresh=rsi, dd_thresh=dd)
                s = calc_sharpe(trades)
                grid.append({
                    "rsi_thresh": rsi, "dd_thresh": dd,
                    "sharpe": round(s, 3), "n_trades": len(trades),
                })
        total = len(grid)
        passing = sum(1 for g in grid if g["sharpe"] > 0.3)
        param_label = "RSI × drawdown threshold"

    elif variant == "D":
        pctile_vals = [0.05, 0.10, 0.15, 0.20]
        grid = []
        for p in pctile_vals:
            trades = gen_signals(variant, pctile_thresh=p)
            s = calc_sharpe(trades)
            grid.append({
                "pctile_thresh": p,
                "sharpe": round(s, 3), "n_trades": len(trades),
            })
        total = len(grid)
        passing = sum(1 for g in grid if g["sharpe"] > 0.3)
        param_label = "percentile threshold"

    elif variant == "F":
        hold_vals = [10, 15, 20, 25]
        grid = []
        for h in hold_vals:
            trades = gen_signals(variant, hold=h)
            s = calc_sharpe(trades)
            grid.append({
                "hold_days": h,
                "sharpe": round(s, 3), "n_trades": len(trades),
            })
        total = len(grid)
        passing = sum(1 for g in grid if g["sharpe"] > 0.3)
        param_label = "hold period"

    pct_passing = passing / total * 100 if total > 0 else 0
    passed = pct_passing >= 50  # At least 50% of grid should pass

    return {
        "test": "parameter_sensitivity",
        "passed": passed,
        "param_type": param_label,
        "grid": grid,
        "total_combos": total,
        "passing_combos": passing,
        "pct_passing": round(pct_passing, 1),
        "verdict": f"{'PASS' if passed else 'FAIL'} — {pct_passing:.0f}% of {param_label} grid has Sharpe > 0.3",
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
def test_cost_sensitivity(variant):
    print(f"  [{variant}] Test 6: Cost Sensitivity ...")
    trades = gen_signals(variant)
    cost_levels = [0.0005, 0.0010, 0.0020, 0.0050]  # 5bps, 10bps, 20bps, 50bps
    results = {}
    break_cost = None

    for cost in cost_levels:
        s = calc_sharpe(trades, slippage=cost)
        label = f"{cost*10000:.0f}bps"
        results[label] = round(s, 3)
        if s < 0.3 and break_cost is None:
            break_cost = label

    # Also find exact breakeven by binary search
    lo, hi = 0.0001, 0.01
    for _ in range(20):
        mid = (lo + hi) / 2
        if calc_sharpe(trades, slippage=mid) > 0.3:
            lo = mid
        else:
            hi = mid
    breakeven_bps = round(lo * 10000, 1)

    passed = breakeven_bps >= 10  # Should survive at least 10bps

    return {
        "test": "cost_sensitivity",
        "passed": passed,
        "sharpe_by_cost": results,
        "breakeven_bps": breakeven_bps,
        "verdict": f"{'PASS' if passed else 'FAIL'} — strategy breaks (Sharpe < 0.3) at ~{breakeven_bps}bps slippage",
    }


# ══════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION: Quality Mean Reversion")
print("=" * 70)

all_results = {}
for variant in ["A", "D", "F"]:
    print(f"\n--- Variant {variant} ---")
    # Baseline metrics
    baseline = gen_signals(variant)
    baseline_m = calc_metrics(baseline)
    print(f"  Baseline: {baseline_m}")

    results = {
        "baseline": baseline_m,
        "tests": {},
    }

    results["tests"]["1_inverse_signal"] = test_inverse(variant)
    results["tests"]["2_random_timing"] = test_random_timing(variant)
    results["tests"]["3_sub_period_stability"] = test_sub_period(variant)
    results["tests"]["4_remove_top3"] = test_remove_top3(variant)
    results["tests"]["5_param_sensitivity"] = test_param_sensitivity(variant)
    results["tests"]["6_cost_sensitivity"] = test_cost_sensitivity(variant)

    all_results[f"variant_{variant}"] = results

# ── Save Results ──────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/quality_mean_reversion_adversarial.json")

# JSON-serialize dates
def json_serial(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

output_path.write_text(json.dumps(all_results, indent=2, default=json_serial))
print(f"\nResults saved to {output_path}")

# ── Summary Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION SUMMARY")
print("=" * 70)

test_names = [
    ("1_inverse_signal", "1. Inverse Signal"),
    ("2_random_timing", "2. Random Timing"),
    ("3_sub_period_stability", "3. Sub-Period Stability"),
    ("4_remove_top3", "4. Remove Top 3"),
    ("5_param_sensitivity", "5. Param Sensitivity"),
    ("6_cost_sensitivity", "6. Cost Sensitivity"),
]

header = f"{'Test':<25} {'Variant A':>12} {'Variant D':>12} {'Variant F':>12}"
print(header)
print("-" * len(header))

for key, label in test_names:
    row = f"{label:<25}"
    for v in ["A", "D", "F"]:
        t = all_results[f"variant_{v}"]["tests"][key]
        status = "PASS" if t["passed"] else "FAIL"
        # Add key metric
        if key == "1_inverse_signal":
            detail = f"inv={t['inverse_sharpe']:.2f}"
        elif key == "2_random_timing":
            detail = f"p{t.get('percentile', 0):.0f}"
        elif key == "3_sub_period_stability":
            n_pos = sum(1 for p in t["periods"].values() if p["sharpe"] > 0)
            detail = f"{n_pos}/4"
        elif key == "4_remove_top3":
            detail = f"-{t['sharpe_drop_pct']:.0f}%"
        elif key == "5_param_sensitivity":
            detail = f"{t['pct_passing']:.0f}%"
        elif key == "6_cost_sensitivity":
            detail = f"{t['breakeven_bps']}bp"
        else:
            detail = ""
        row += f" {status:>4} {detail:>6}"
    print(row)

# Overall verdict
print("\n--- Overall ---")
for v in ["A", "D", "F"]:
    tests = all_results[f"variant_{v}"]["tests"]
    passed = sum(1 for t in tests.values() if t["passed"])
    total = len(tests)
    baseline = all_results[f"variant_{v}"]["baseline"]
    print(f"Variant {v}: {passed}/{total} tests passed | "
          f"Baseline Sharpe={baseline['sharpe']:.3f}, WR={baseline['win_rate']:.1%}, "
          f"PF={baseline['profit_factor']:.2f}, N={baseline['n_trades']}")

print("\nDone.")
