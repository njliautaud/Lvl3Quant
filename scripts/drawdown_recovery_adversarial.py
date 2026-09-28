#!/usr/bin/env python3
"""
Adversarial Validation: Drawdown Recovery Pattern
===================================================
6 adversarial checks on two variants:

Variant A (50% Retracement):
  Stock drops >5% from 20-day high, then enters when it recovers 50% of drawdown.
  Hold 10 days.

Variant E (First Green After 3+ Red):
  Stock drops >5% from 20-day high AND 3+ consecutive red days,
  enter on first green day. Hold 5 days.

Tests:
1) Inverse Signal — buy on continued decline (no recovery confirmation)
2) Random Entry Timing — 1000 random iterations, report percentile
3) Sub-Period Stability — 4 sub-periods, all must have positive Sharpe
4) Remove Top 3 Tickers — Sharpe drop < 50%
5) Parameter Sensitivity — sweep params, report % with Sharpe > 0.3
6) Cost Sensitivity — test at 5/10/20/50 bps, report breakeven

OOT: Jan 2022 – Jul 2026. $645 capital, $200/trade max, 3 concurrent max, 0.02% slippage.
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
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% baseline
START = "2020-01-01"
END = "2026-07-31"
OOT_START = "2022-01-01"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP",
    "HD", "COST", "UNH", "LLY", "V", "MA", "ABBV", "MRK",
    "WMT", "AMZN", "GOOGL", "META",
]

ALL_TICKERS = sorted(set(TICKERS))

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def rolling_high(series, window=20):
    return series.rolling(window).max()


def drawdown_from_high(series, window=20):
    rh = rolling_high(series, window)
    return (series - rh) / rh


def consecutive_red_days(series):
    """Count consecutive red (close < previous close) days ending at each point."""
    red = (series < series.shift(1)).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(red)):
        if red.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


# ── Pre-compute indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 100:
        continue
    rh = rolling_high(c, 20)
    dd = (c - rh) / rh  # drawdown pct (negative when below high)
    # Recovery ratio: how much of drawdown has been recovered
    # dd_trough = rolling min of dd over last 20 days
    dd_trough = dd.rolling(20).min()
    # retracement = (dd - dd_trough) / (-dd_trough) when dd_trough < 0
    retracement = pd.Series(np.nan, index=c.index)
    mask = dd_trough < -0.001
    retracement[mask] = (dd[mask] - dd_trough[mask]) / (-dd_trough[mask])
    retracement = retracement.clip(0, 1)

    # Has been in >5% drawdown recently (within last 20 days)
    in_dd = dd_trough < -0.05

    # Consecutive red days (for variant E)
    crd = consecutive_red_days(c)
    # Was there a streak of 3+ red days recently? Track the max streak in last 10 days
    prev_crd = crd.shift(1)  # yesterday's streak count
    green_today = c > c.shift(1)

    indicators[t] = {
        "close": c,
        "dd": dd,
        "dd_trough": dd_trough,
        "retracement": retracement,
        "in_dd": in_dd,
        "crd": crd,
        "prev_crd": prev_crd,
        "green_today": green_today,
        "rh": rh,
    }


# ── Backtest Engine ──────────────────────────────────────────────────────
def generate_signals_A(tickers, retracement_level=0.50, hold_days=10,
                       dd_thresh=-0.05):
    """Variant A: Enter when retracement reaches level after >5% drawdown."""
    trades = []
    for t in tickers:
        if t not in indicators:
            continue
        ind = indicators[t]
        c = ind["close"]
        retr = ind["retracement"]
        in_dd = ind["dd_trough"] < dd_thresh

        for i in range(1, len(c)):
            date = c.index[i]
            if str(date.date()) < OOT_START:
                continue
            # Entry: in drawdown zone AND retracement crosses level
            if (in_dd.iloc[i] and
                retr.iloc[i] >= retracement_level and
                (i < 1 or retr.iloc[i-1] < retracement_level)):
                entry_price = c.iloc[i]
                exit_idx = min(i + hold_days, len(c) - 1)
                exit_price = c.iloc[exit_idx]
                trades.append({
                    "ticker": t,
                    "entry_date": date,
                    "exit_date": c.index[exit_idx],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "hold_days": hold_days,
                })
    return trades


def generate_signals_E(tickers, min_red_days=3, hold_days=5,
                       dd_thresh=-0.05):
    """Variant E: First green day after 3+ consecutive red days, in drawdown."""
    trades = []
    for t in tickers:
        if t not in indicators:
            continue
        ind = indicators[t]
        c = ind["close"]
        in_dd = ind["dd_trough"] < dd_thresh
        prev_crd = ind["prev_crd"]
        green = ind["green_today"]

        for i in range(1, len(c)):
            date = c.index[i]
            if str(date.date()) < OOT_START:
                continue
            # Entry: in drawdown, yesterday had 3+ red streak, today is green
            if (in_dd.iloc[i] and
                prev_crd.iloc[i] >= min_red_days and
                green.iloc[i]):
                entry_price = c.iloc[i]
                exit_idx = min(i + hold_days, len(c) - 1)
                exit_price = c.iloc[exit_idx]
                trades.append({
                    "ticker": t,
                    "entry_date": date,
                    "exit_date": c.index[exit_idx],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "hold_days": hold_days,
                })
    return trades


def generate_inverse_A(tickers, retracement_level=0.50, hold_days=10,
                       dd_thresh=-0.05):
    """Inverse of A: Enter when STILL declining (retracement < 0.2) in drawdown."""
    trades = []
    for t in tickers:
        if t not in indicators:
            continue
        ind = indicators[t]
        c = ind["close"]
        retr = ind["retracement"]
        dd = ind["dd"]
        in_dd = ind["dd_trough"] < dd_thresh

        for i in range(1, len(c)):
            date = c.index[i]
            if str(date.date()) < OOT_START:
                continue
            # Inverse: in drawdown but NO recovery — still declining
            if (in_dd.iloc[i] and
                retr.iloc[i] < 0.20 and
                dd.iloc[i] < dd_thresh):
                entry_price = c.iloc[i]
                exit_idx = min(i + hold_days, len(c) - 1)
                exit_price = c.iloc[exit_idx]
                trades.append({
                    "ticker": t,
                    "entry_date": date,
                    "exit_date": c.index[exit_idx],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "hold_days": hold_days,
                })
    return trades


def generate_inverse_E(tickers, min_red_days=3, hold_days=5,
                       dd_thresh=-0.05):
    """Inverse of E: Enter on ANOTHER red day (continued decline) instead of green."""
    trades = []
    for t in tickers:
        if t not in indicators:
            continue
        ind = indicators[t]
        c = ind["close"]
        in_dd = ind["dd_trough"] < dd_thresh
        crd = ind["crd"]
        green = ind["green_today"]

        for i in range(1, len(c)):
            date = c.index[i]
            if str(date.date()) < OOT_START:
                continue
            # Inverse: in drawdown, 3+ red streak, AND today is ALSO red
            if (in_dd.iloc[i] and
                crd.iloc[i] >= min_red_days and
                not green.iloc[i]):
                entry_price = c.iloc[i]
                exit_idx = min(i + hold_days, len(c) - 1)
                exit_price = c.iloc[exit_idx]
                trades.append({
                    "ticker": t,
                    "entry_date": date,
                    "exit_date": c.index[exit_idx],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "hold_days": hold_days,
                })
    return trades


def apply_portfolio_constraints(trades, max_concurrent=MAX_CONCURRENT):
    """Apply max concurrent positions constraint, sorted by date."""
    if not trades:
        return []
    trades_sorted = sorted(trades, key=lambda x: x["entry_date"])
    selected = []
    active_exits = []
    for t in trades_sorted:
        # Remove expired positions
        active_exits = [e for e in active_exits if e > t["entry_date"]]
        if len(active_exits) < max_concurrent:
            selected.append(t)
            active_exits.append(t["exit_date"])
    return selected


def compute_metrics(trades, slippage_pct=SLIPPAGE_PCT, capital=CAPITAL,
                    max_per_trade=MAX_PER_TRADE):
    """Compute Sharpe, PF, WR from trade list."""
    if not trades:
        return {"sharpe": 0, "pf": 0, "wr": 0, "n_trades": 0,
                "total_return": 0, "avg_return": 0}

    returns = []
    for t in trades:
        position_size = min(max_per_trade, capital)
        shares = position_size / t["entry_price"]
        entry_cost = t["entry_price"] * (1 + slippage_pct)
        exit_cost = t["exit_price"] * (1 - slippage_pct)
        pnl = shares * (exit_cost - entry_cost)
        ret = pnl / position_size
        returns.append(ret)

    returns = np.array(returns)
    n = len(returns)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    wr = len(wins) / n if n > 0 else 0
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else (99.0 if len(wins) > 0 else 0)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252 / 10)) if std_ret > 0 else 0  # annualized approx
    total_ret = returns.sum()

    return {
        "sharpe": round(float(sharpe), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 3),
        "n_trades": n,
        "total_return": round(float(total_ret), 4),
        "avg_return": round(float(mean_ret), 5),
    }


# ── Run baseline strategies ─────────────────────────────────────────────
print("\n" + "="*70)
print("DRAWDOWN RECOVERY — ADVERSARIAL VALIDATION")
print("="*70)

results = {"variant_A": {}, "variant_E": {}}

# Baseline runs
trades_A = apply_portfolio_constraints(generate_signals_A(TICKERS))
trades_E = apply_portfolio_constraints(generate_signals_E(TICKERS))
baseline_A = compute_metrics(trades_A)
baseline_E = compute_metrics(trades_E)

print(f"\n── Baseline Variant A (50% Retracement, hold 10d) ──")
print(f"   Sharpe={baseline_A['sharpe']}, PF={baseline_A['pf']}, "
      f"WR={baseline_A['wr']}, N={baseline_A['n_trades']}")

print(f"\n── Baseline Variant E (First Green After 3+ Red, hold 5d) ──")
print(f"   Sharpe={baseline_E['sharpe']}, PF={baseline_E['pf']}, "
      f"WR={baseline_E['wr']}, N={baseline_E['n_trades']}")

results["variant_A"]["baseline"] = baseline_A
results["variant_E"]["baseline"] = baseline_E

# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE SIGNAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("TEST 1: INVERSE SIGNAL")
print("─"*70)

inv_trades_A = apply_portfolio_constraints(generate_inverse_A(TICKERS))
inv_A = compute_metrics(inv_trades_A)
inv_trades_E = apply_portfolio_constraints(generate_inverse_E(TICKERS))
inv_E = compute_metrics(inv_trades_E)

# PASS if inverse is materially worse
pass_1A = inv_A["sharpe"] < baseline_A["sharpe"] * 0.5
pass_1E = inv_E["sharpe"] < baseline_E["sharpe"] * 0.5

print(f"  Variant A: Original Sharpe={baseline_A['sharpe']}, "
      f"Inverse Sharpe={inv_A['sharpe']} → {'PASS' if pass_1A else 'FAIL'}")
print(f"  Variant E: Original Sharpe={baseline_E['sharpe']}, "
      f"Inverse Sharpe={inv_E['sharpe']} → {'PASS' if pass_1E else 'FAIL'}")

results["variant_A"]["test1_inverse"] = {
    "inverse_metrics": inv_A, "pass": pass_1A,
    "logic": "Inverse Sharpe < 50% of original"
}
results["variant_E"]["test1_inverse"] = {
    "inverse_metrics": inv_E, "pass": pass_1E,
    "logic": "Inverse Sharpe < 50% of original"
}

# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM ENTRY TIMING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("TEST 2: RANDOM ENTRY TIMING (1000 iterations)")
print("─"*70)


def random_entry_test(gen_func, gen_kwargs, n_perm=N_PERM):
    """Generate random entry dates preserving trade count and ticker mix."""
    base_trades = gen_func(**gen_kwargs)
    base_trades = apply_portfolio_constraints(base_trades)
    if not base_trades:
        return 0, 0, []

    base_sharpe = compute_metrics(base_trades)["sharpe"]

    # Get valid OOT trading dates per ticker
    oot_dates = {}
    for t in TICKERS:
        if t not in indicators:
            continue
        c = indicators[t]["close"]
        valid = c.index[c.index >= OOT_START]
        if len(valid) > 20:
            oot_dates[t] = valid

    # Count trades per ticker in original
    ticker_counts = {}
    for tr in base_trades:
        ticker_counts[tr["ticker"]] = ticker_counts.get(tr["ticker"], 0) + 1

    random_sharpes = []
    hold = base_trades[0]["hold_days"] if base_trades else 10

    for _ in range(n_perm):
        rand_trades = []
        for t, count in ticker_counts.items():
            if t not in oot_dates or len(oot_dates[t]) < hold + 1:
                continue
            c = indicators[t]["close"]
            valid_idx = [i for i, d in enumerate(c.index) if d >= pd.Timestamp(OOT_START)]
            valid_idx = [i for i in valid_idx if i + hold < len(c)]
            if not valid_idx:
                continue
            chosen = np.random.choice(valid_idx, size=min(count, len(valid_idx)),
                                      replace=False)
            for idx in chosen:
                rand_trades.append({
                    "ticker": t,
                    "entry_date": c.index[idx],
                    "exit_date": c.index[min(idx + hold, len(c) - 1)],
                    "entry_price": float(c.iloc[idx]),
                    "exit_price": float(c.iloc[min(idx + hold, len(c) - 1)]),
                    "hold_days": hold,
                })
        rand_trades = apply_portfolio_constraints(rand_trades)
        m = compute_metrics(rand_trades)
        random_sharpes.append(m["sharpe"])

    percentile = np.mean([1 for s in random_sharpes if s < base_sharpe]) * 100
    return base_sharpe, percentile, random_sharpes


sharpe_A, pct_A, rand_A = random_entry_test(
    generate_signals_A, {"tickers": TICKERS})
sharpe_E, pct_E, rand_E = random_entry_test(
    generate_signals_E, {"tickers": TICKERS})

pass_2A = pct_A >= 90
pass_2E = pct_E >= 90

print(f"  Variant A: Sharpe={sharpe_A}, Percentile={pct_A:.1f}% "
      f"→ {'PASS' if pass_2A else 'FAIL'}")
print(f"  Variant E: Sharpe={sharpe_E}, Percentile={pct_E:.1f}% "
      f"→ {'PASS' if pass_2E else 'FAIL'}")

results["variant_A"]["test2_random_timing"] = {
    "percentile": round(pct_A, 1), "pass": pass_2A,
    "random_sharpe_mean": round(float(np.mean(rand_A)), 3) if rand_A else 0,
    "random_sharpe_std": round(float(np.std(rand_A)), 3) if rand_A else 0,
}
results["variant_E"]["test2_random_timing"] = {
    "percentile": round(pct_E, 1), "pass": pass_2E,
    "random_sharpe_mean": round(float(np.mean(rand_E)), 3) if rand_E else 0,
    "random_sharpe_std": round(float(np.std(rand_E)), 3) if rand_E else 0,
}

# ══════════════════════════════════════════════════════════════════════════
# TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("TEST 3: SUB-PERIOD STABILITY (4 periods)")
print("─"*70)


def sub_period_test(trades, n_periods=4):
    if not trades:
        return [], False
    dates = sorted(set(t["entry_date"] for t in trades))
    if len(dates) < n_periods:
        return [], False
    boundaries = np.array_split(dates, n_periods)
    period_sharpes = []
    for period_dates in boundaries:
        start_d, end_d = period_dates[0], period_dates[-1]
        period_trades = [t for t in trades
                         if start_d <= t["entry_date"] <= end_d]
        m = compute_metrics(period_trades)
        period_sharpes.append({
            "start": str(start_d.date()),
            "end": str(end_d.date()),
            "sharpe": m["sharpe"],
            "n_trades": m["n_trades"],
        })
    all_positive = all(p["sharpe"] > 0 for p in period_sharpes)
    return period_sharpes, all_positive


periods_A, pass_3A = sub_period_test(trades_A)
periods_E, pass_3E = sub_period_test(trades_E)

print("  Variant A sub-periods:")
for p in periods_A:
    print(f"    {p['start']} to {p['end']}: Sharpe={p['sharpe']}, N={p['n_trades']}")
print(f"    → {'PASS' if pass_3A else 'FAIL'}")

print("  Variant E sub-periods:")
for p in periods_E:
    print(f"    {p['start']} to {p['end']}: Sharpe={p['sharpe']}, N={p['n_trades']}")
print(f"    → {'PASS' if pass_3E else 'FAIL'}")

results["variant_A"]["test3_subperiod"] = {
    "periods": periods_A, "all_positive": pass_3A, "pass": pass_3A
}
results["variant_E"]["test3_subperiod"] = {
    "periods": periods_E, "all_positive": pass_3E, "pass": pass_3E
}

# ══════════════════════════════════════════════════════════════════════════
# TEST 4: REMOVE TOP 3 TICKERS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("TEST 4: REMOVE TOP 3 TICKERS")
print("─"*70)


def find_top_tickers(trades, n=3):
    """Find the N tickers with highest total return."""
    ticker_returns = {}
    for t in trades:
        ret = (t["exit_price"] - t["entry_price"]) / t["entry_price"]
        ticker_returns[t["ticker"]] = ticker_returns.get(t["ticker"], 0) + ret
    sorted_tickers = sorted(ticker_returns.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tickers[:n]]


def remove_top_test(gen_func, gen_kwargs, baseline_sharpe):
    all_trades = apply_portfolio_constraints(gen_func(**gen_kwargs))
    top3 = find_top_tickers(all_trades, 3)
    reduced_tickers = [t for t in gen_kwargs["tickers"] if t not in top3]
    new_kwargs = {**gen_kwargs, "tickers": reduced_tickers}
    reduced_trades = apply_portfolio_constraints(gen_func(**new_kwargs))
    reduced_metrics = compute_metrics(reduced_trades)
    drop = 1 - (reduced_metrics["sharpe"] / baseline_sharpe) if baseline_sharpe != 0 else 1
    return top3, reduced_metrics, drop


top3_A, reduced_A, drop_A = remove_top_test(
    generate_signals_A, {"tickers": TICKERS}, baseline_A["sharpe"])
top3_E, reduced_E, drop_E = remove_top_test(
    generate_signals_E, {"tickers": TICKERS}, baseline_E["sharpe"])

pass_4A = drop_A < 0.50
pass_4E = drop_E < 0.50

print(f"  Variant A: Top 3 = {top3_A}")
print(f"    Reduced Sharpe={reduced_A['sharpe']}, Drop={drop_A:.1%} "
      f"→ {'PASS' if pass_4A else 'FAIL'}")
print(f"  Variant E: Top 3 = {top3_E}")
print(f"    Reduced Sharpe={reduced_E['sharpe']}, Drop={drop_E:.1%} "
      f"→ {'PASS' if pass_4E else 'FAIL'}")

results["variant_A"]["test4_remove_top3"] = {
    "top3_removed": top3_A,
    "reduced_metrics": reduced_A,
    "sharpe_drop_pct": round(drop_A * 100, 1),
    "pass": pass_4A,
}
results["variant_E"]["test4_remove_top3"] = {
    "top3_removed": top3_E,
    "reduced_metrics": reduced_E,
    "sharpe_drop_pct": round(drop_E * 100, 1),
    "pass": pass_4E,
}

# ══════════════════════════════════════════════════════════════════════════
# TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("TEST 5: PARAMETER SENSITIVITY")
print("─"*70)

# Variant A: retracement 30/40/50/60/70% × hold 5/10/15/20d
retr_levels = [0.30, 0.40, 0.50, 0.60, 0.70]
hold_days_A = [5, 10, 15, 20]
param_results_A = []
for retr in retr_levels:
    for hd in hold_days_A:
        tr = apply_portfolio_constraints(
            generate_signals_A(TICKERS, retracement_level=retr, hold_days=hd))
        m = compute_metrics(tr)
        param_results_A.append({
            "retracement": retr, "hold_days": hd,
            "sharpe": m["sharpe"], "n_trades": m["n_trades"],
        })

n_above_A = sum(1 for p in param_results_A if p["sharpe"] > 0.3)
pct_above_A = n_above_A / len(param_results_A) * 100
pass_5A = pct_above_A >= 40  # at least 40% of param combos work

print(f"  Variant A: {n_above_A}/{len(param_results_A)} combos with Sharpe > 0.3 "
      f"({pct_above_A:.0f}%) → {'PASS' if pass_5A else 'FAIL'}")
for p in param_results_A:
    status = "✓" if p["sharpe"] > 0.3 else "✗"
    print(f"    retr={p['retracement']:.0%} hold={p['hold_days']}d: "
          f"Sharpe={p['sharpe']}, N={p['n_trades']} {status}")

# Variant E: consecutive red 2/3/4/5 × hold 3/5/7/10d
red_days_sweep = [2, 3, 4, 5]
hold_days_E = [3, 5, 7, 10]
param_results_E = []
for rd in red_days_sweep:
    for hd in hold_days_E:
        tr = apply_portfolio_constraints(
            generate_signals_E(TICKERS, min_red_days=rd, hold_days=hd))
        m = compute_metrics(tr)
        param_results_E.append({
            "min_red_days": rd, "hold_days": hd,
            "sharpe": m["sharpe"], "n_trades": m["n_trades"],
        })

n_above_E = sum(1 for p in param_results_E if p["sharpe"] > 0.3)
pct_above_E = n_above_E / len(param_results_E) * 100
pass_5E = pct_above_E >= 40

print(f"\n  Variant E: {n_above_E}/{len(param_results_E)} combos with Sharpe > 0.3 "
      f"({pct_above_E:.0f}%) → {'PASS' if pass_5E else 'FAIL'}")
for p in param_results_E:
    status = "✓" if p["sharpe"] > 0.3 else "✗"
    print(f"    red_days={p['min_red_days']} hold={p['hold_days']}d: "
          f"Sharpe={p['sharpe']}, N={p['n_trades']} {status}")

results["variant_A"]["test5_param_sensitivity"] = {
    "param_grid": param_results_A,
    "pct_above_0.3": round(pct_above_A, 1),
    "pass": pass_5A,
}
results["variant_E"]["test5_param_sensitivity"] = {
    "param_grid": param_results_E,
    "pct_above_0.3": round(pct_above_E, 1),
    "pass": pass_5E,
}

# ══════════════════════════════════════════════════════════════════════════
# TEST 6: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "─"*70)
print("TEST 6: COST SENSITIVITY")
print("─"*70)

cost_levels = [0.0005, 0.0010, 0.0020, 0.0050]  # 5/10/20/50 bps
cost_labels = ["5bps", "10bps", "20bps", "50bps"]


def cost_sensitivity_test(trades, baseline_sharpe):
    cost_results = []
    breakeven = None
    for cost, label in zip(cost_levels, cost_labels):
        m = compute_metrics(trades, slippage_pct=cost)
        cost_results.append({
            "cost_bps": label,
            "cost_pct": cost,
            "sharpe": m["sharpe"],
            "pf": m["pf"],
        })
        if m["sharpe"] <= 0 and breakeven is None:
            # Interpolate breakeven
            prev = cost_results[-2] if len(cost_results) > 1 else None
            if prev and prev["sharpe"] > 0:
                frac = prev["sharpe"] / (prev["sharpe"] - m["sharpe"])
                breakeven = prev["cost_pct"] + frac * (cost - prev["cost_pct"])
            else:
                breakeven = cost

    # If still profitable at 50bps, breakeven is above that
    if breakeven is None:
        # Try to find via binary search
        lo, hi = 0.005, 0.02
        for _ in range(20):
            mid = (lo + hi) / 2
            m = compute_metrics(trades, slippage_pct=mid)
            if m["sharpe"] > 0:
                lo = mid
            else:
                hi = mid
        breakeven = (lo + hi) / 2

    return cost_results, breakeven


cost_A, breakeven_A = cost_sensitivity_test(trades_A, baseline_A["sharpe"])
cost_E, breakeven_E = cost_sensitivity_test(trades_E, baseline_E["sharpe"])

pass_6A = breakeven_A > 0.0020  # survives at least 20bps
pass_6E = breakeven_E > 0.0020

print(f"  Variant A:")
for c in cost_A:
    print(f"    {c['cost_bps']}: Sharpe={c['sharpe']}, PF={c['pf']}")
print(f"    Breakeven ≈ {breakeven_A*10000:.0f} bps → {'PASS' if pass_6A else 'FAIL'}")

print(f"  Variant E:")
for c in cost_E:
    print(f"    {c['cost_bps']}: Sharpe={c['sharpe']}, PF={c['pf']}")
print(f"    Breakeven ≈ {breakeven_E*10000:.0f} bps → {'PASS' if pass_6E else 'FAIL'}")

results["variant_A"]["test6_cost_sensitivity"] = {
    "cost_levels": cost_A,
    "breakeven_bps": round(breakeven_A * 10000, 1),
    "pass": pass_6A,
}
results["variant_E"]["test6_cost_sensitivity"] = {
    "cost_levels": cost_E,
    "breakeven_bps": round(breakeven_E * 10000, 1),
    "pass": pass_6E,
}

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADVERSARIAL VALIDATION SUMMARY")
print("="*70)

test_names = [
    ("test1_inverse", "Inverse Signal"),
    ("test2_random_timing", "Random Entry Timing"),
    ("test3_subperiod", "Sub-Period Stability"),
    ("test4_remove_top3", "Remove Top 3 Tickers"),
    ("test5_param_sensitivity", "Parameter Sensitivity"),
    ("test6_cost_sensitivity", "Cost Sensitivity"),
]

for variant, label in [("variant_A", "Variant A (50% Retracement)"),
                        ("variant_E", "Variant E (First Green)")]:
    passes = sum(1 for k, _ in test_names if results[variant].get(k, {}).get("pass", False))
    print(f"\n  {label}: {passes}/6 PASS")
    for key, name in test_names:
        p = results[variant].get(key, {}).get("pass", False)
        print(f"    {'PASS' if p else 'FAIL'} — {name}")

    overall = "PASS" if passes >= 5 else ("MARGINAL" if passes >= 4 else "FAIL")
    results[variant]["overall"] = overall
    results[variant]["passes"] = passes
    print(f"    OVERALL: {overall}")

# ── Save results ──────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/drawdown_recovery_adversarial.json")
output_path.parent.mkdir(parents=True, exist_ok=True)


def default_serializer(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=default_serializer)

print(f"\nResults saved to {output_path}")
