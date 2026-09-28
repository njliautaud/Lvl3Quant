#!/usr/bin/env python3
"""
Adversarial Validation: Earnings Surprise Momentum
====================================================
6 adversarial checks to stress-test whether the edge is real.

1) Inverse Direction — buy after earnings MISSES instead of beats
2) Random Timing — 1000 random entry sets, percentile rank
3) Look-Ahead Removal — drop last 20% of data, check Sharpe drop
4) Cost Sensitivity — slippage 0.05% to 0.20%
5) Sub-Period Stability — 4 equal sub-periods, all must have positive Sharpe
6) Parameter Sensitivity — 125 param combos, % with Sharpe > 0.5

Strategy under test:
  After quarterly earnings, if EPS actual/estimate > 1.10 (10%+ beat)
  AND stock gaps up > 3% on earnings day, buy at close.
  Hold 40 trading days, sell at close.
  Kill switch: skip if VIX > 20 AND SPY < 50-SMA.
  Equal-weight, 1 position at a time, $669 capital, fractional shares.
  $0 commission, 0.02% slippage baseline.

OOT: Jan 2022 – Jul 2026.
"""

import json
import warnings
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02% baseline
START = "2021-06-01"   # extra buffer for 50-SMA on SPY
END = "2026-07-30"
OOT_START = "2022-01-01"
N_PERM = 1000

BEAT_THRESHOLD = 0.10   # 10% EPS beat
GAP_THRESHOLD = 0.03    # 3% gap up
HOLD_DAYS = 40           # trading days

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "CRM", "NFLX", "SHOP", "SQ", "SNOW", "PLTR", "COIN",
    "MELI", "MDB", "DDOG", "TTD",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY", "^VIX"]))

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))
vix_close = closes.get("^VIX", pd.Series(dtype=float))

# SPY 50-SMA
spy_sma50 = spy_close.rolling(50).mean() if len(spy_close) > 50 else pd.Series(dtype=float)

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 100)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")


# ── Earnings Data Download ───────────────────────────────────────────────
print("Downloading earnings data ...")
earnings_data = {}  # ticker -> list of (date, surprise_pct, gap_pct)

for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=30)
        if ed is None or len(ed) == 0:
            continue

        tc = closes.get(ticker, pd.Series(dtype=float))
        if len(tc) < 10:
            continue

        events = []
        for idx, row in ed.iterrows():
            eps_est = row.get("EPS Estimate", None)
            eps_act = row.get("Reported EPS", None)

            if pd.isna(eps_est) or pd.isna(eps_act) or eps_est is None or eps_act is None:
                continue
            if eps_est == 0:
                continue

            surprise_ratio = eps_act / eps_est  # > 1.10 means 10%+ beat

            # Earnings date (normalize to date only)
            earn_date = idx.date() if hasattr(idx, 'date') else pd.Timestamp(idx).date()
            earn_date = pd.Timestamp(earn_date)

            # Find the earnings date in price data (or next trading day)
            matching = tc.index[tc.index >= earn_date]
            if len(matching) == 0:
                continue
            trade_date = matching[0]

            # Find previous trading day for gap calculation
            prev_idx = tc.index.get_loc(trade_date)
            if prev_idx < 1:
                continue
            prev_date = tc.index[prev_idx - 1]

            gap_pct = (tc[trade_date] - tc[prev_date]) / tc[prev_date]

            events.append({
                "date": trade_date,
                "surprise_ratio": surprise_ratio,
                "gap_pct": gap_pct,
                "eps_actual": eps_act,
                "eps_estimate": eps_est,
            })

        earnings_data[ticker] = events
        time.sleep(0.2)  # rate limit
    except Exception as e:
        print(f"  Warning: {ticker} earnings fetch failed: {e}")

total_events = sum(len(v) for v in earnings_data.values())
print(f"  Total earnings events collected: {total_events}")


# ── Kill Switch ──────────────────────────────────────────────────────────
def kill_switch_active(date):
    """Returns True if VIX > 20 AND SPY < 50-SMA on given date."""
    try:
        vix_val = vix_close.asof(date)
        spy_val = spy_close.asof(date)
        sma_val = spy_sma50.asof(date)
        if pd.isna(vix_val) or pd.isna(spy_val) or pd.isna(sma_val):
            return False
        return vix_val > 20 and spy_val < sma_val
    except Exception:
        return False


# ── Core Backtest Engine ─────────────────────────────────────────────────
def run_backtest(beat_thresh=BEAT_THRESHOLD, gap_thresh=GAP_THRESHOLD,
                 hold_days=HOLD_DAYS, slippage=SLIPPAGE_PCT,
                 start_date=None, end_date=None, inverse=False,
                 custom_entries=None):
    """
    Run the earnings surprise momentum backtest.

    If inverse=True, buy after misses (surprise_ratio < 1.0) instead of beats.
    If custom_entries is provided, use those (ticker, date) pairs directly.

    Returns dict with trades list, equity curve, and metrics.
    """
    if start_date is None:
        start_date = pd.Timestamp(OOT_START)
    else:
        start_date = pd.Timestamp(start_date)
    if end_date is None:
        end_date = pd.Timestamp(END)
    else:
        end_date = pd.Timestamp(end_date)

    trades = []

    if custom_entries is not None:
        # Use provided entries directly
        for ticker, entry_date in custom_entries:
            entry_date = pd.Timestamp(entry_date)
            if entry_date < start_date or entry_date > end_date:
                continue
            if kill_switch_active(entry_date):
                continue

            tc = closes.get(ticker, pd.Series(dtype=float))
            if len(tc) == 0:
                continue

            # Find entry date in price data
            matching = tc.index[tc.index >= entry_date]
            if len(matching) == 0:
                continue
            actual_entry = matching[0]
            entry_loc = tc.index.get_loc(actual_entry)

            # Exit after hold_days trading days
            exit_loc = min(entry_loc + hold_days, len(tc) - 1)
            exit_date = tc.index[exit_loc]

            entry_price = tc.iloc[entry_loc] * (1 + slippage)
            exit_price = tc.iloc[exit_loc] * (1 - slippage)

            shares = CAPITAL / entry_price
            pnl = shares * (exit_price - entry_price)
            ret = (exit_price - entry_price) / entry_price

            trades.append({
                "ticker": ticker,
                "entry_date": str(actual_entry.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return": float(ret),
                "pnl": float(pnl),
            })
    else:
        # Standard earnings-based entries
        for ticker, events in earnings_data.items():
            tc = closes.get(ticker, pd.Series(dtype=float))
            if len(tc) == 0:
                continue

            for ev in events:
                entry_date = ev["date"]
                if entry_date < start_date or entry_date > end_date:
                    continue

                # Signal filter
                if inverse:
                    # Buy after misses
                    if ev["surprise_ratio"] >= 1.0:
                        continue
                else:
                    # Buy after beats
                    if ev["surprise_ratio"] < (1.0 + beat_thresh):
                        continue
                    if ev["gap_pct"] < gap_thresh:
                        continue

                # Kill switch
                if kill_switch_active(entry_date):
                    continue

                # Entry at close of earnings day
                if entry_date not in tc.index:
                    matching = tc.index[tc.index >= entry_date]
                    if len(matching) == 0:
                        continue
                    entry_date = matching[0]

                entry_loc = tc.index.get_loc(entry_date)

                # Exit after hold_days trading days
                exit_loc = min(entry_loc + hold_days, len(tc) - 1)
                exit_date = tc.index[exit_loc]

                entry_price = tc.iloc[entry_loc] * (1 + slippage)
                exit_price = tc.iloc[exit_loc] * (1 - slippage)

                shares = CAPITAL / entry_price
                pnl = shares * (exit_price - entry_price)
                ret = (exit_price - entry_price) / entry_price

                trades.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(exit_date.date()),
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "return": float(ret),
                    "pnl": float(pnl),
                })

    # Sort trades by entry date (1 position at a time)
    trades.sort(key=lambda x: x["entry_date"])

    # Enforce 1-position-at-a-time: skip overlapping trades
    filtered_trades = []
    last_exit = None
    for t in trades:
        if last_exit is not None and t["entry_date"] <= last_exit:
            continue
        filtered_trades.append(t)
        last_exit = t["exit_date"]

    trades = filtered_trades

    # Compute metrics
    if len(trades) == 0:
        return {"trades": [], "sharpe": 0.0, "sortino": 0.0, "pf": 0.0,
                "wr": 0.0, "n_trades": 0, "total_return": 0.0}

    returns = np.array([t["return"] for t in trades])
    total_pnl = sum(t["pnl"] for t in trades)
    total_return = total_pnl / CAPITAL

    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    sharpe = (avg_ret / std_ret) * np.sqrt(252 / max(hold_days, 1)) if std_ret > 1e-9 else 0.0

    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / down_std) * np.sqrt(252 / max(hold_days, 1)) if down_std > 1e-9 else 0.0

    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    winners = sum(1 for t in trades if t["return"] > 0)
    wr = winners / len(trades)

    return {
        "trades": trades,
        "sharpe": float(round(sharpe, 4)),
        "sortino": float(round(sortino, 4)),
        "pf": float(round(pf, 4)),
        "wr": float(round(wr, 4)),
        "n_trades": len(trades),
        "total_return": float(round(total_return, 4)),
    }


# ── Run baseline ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BASELINE BACKTEST")
print("=" * 70)

baseline = run_backtest()
print(f"  Trades: {baseline['n_trades']}")
print(f"  Sharpe: {baseline['sharpe']:.4f}")
print(f"  Sortino: {baseline['sortino']:.4f}")
print(f"  PF: {baseline['pf']:.4f}")
print(f"  WR: {baseline['wr']:.2%}")
print(f"  Total Return: {baseline['total_return']:.2%}")

results = {
    "strategy": "Earnings Surprise Momentum",
    "baseline": {
        "sharpe": baseline["sharpe"],
        "sortino": baseline["sortino"],
        "pf": baseline["pf"],
        "wr": baseline["wr"],
        "n_trades": baseline["n_trades"],
        "total_return": baseline["total_return"],
    },
    "tests": {},
}

# ── TEST 1: Inverse Direction ────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 1: INVERSE DIRECTION (buy after misses)")
print("=" * 70)

inverse_result = run_backtest(inverse=True)
inverse_sharpe = inverse_result["sharpe"]
test1_pass = inverse_sharpe <= 0.5

print(f"  Inverse trades: {inverse_result['n_trades']}")
print(f"  Inverse Sharpe: {inverse_sharpe:.4f}")
print(f"  Threshold: Inverse Sharpe <= 0.5")
print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

results["tests"]["1_inverse_direction"] = {
    "inverse_sharpe": inverse_sharpe,
    "inverse_n_trades": inverse_result["n_trades"],
    "inverse_total_return": inverse_result["total_return"],
    "threshold": "inverse Sharpe <= 0.5",
    "pass": test1_pass,
}

# ── TEST 2: Random Timing ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2: RANDOM TIMING (1000 permutations)")
print("=" * 70)

# Collect the real entry points
real_entries = []
for t in baseline.get("trades", []):
    real_entries.append((t["ticker"], t["entry_date"]))

# For random timing: keep same tickers and number of trades per ticker,
# but randomize entry dates within the OOT period
trade_dates_by_ticker = {}
for t in baseline.get("trades", []):
    ticker = t["ticker"]
    if ticker not in trade_dates_by_ticker:
        trade_dates_by_ticker[ticker] = []
    trade_dates_by_ticker[ticker].append(t["entry_date"])

# Get all available trading dates per ticker in OOT period
available_dates = {}
for ticker in TICKERS:
    tc = closes.get(ticker, pd.Series(dtype=float))
    if len(tc) == 0:
        continue
    mask = (tc.index >= pd.Timestamp(OOT_START)) & (tc.index <= pd.Timestamp(END))
    dates = tc.index[mask]
    # Leave buffer at end for hold period
    if len(dates) > HOLD_DAYS:
        dates = dates[:-HOLD_DAYS]
    available_dates[ticker] = dates

random_sharpes = []
for i in range(N_PERM):
    random_entries = []
    for ticker, count_list in trade_dates_by_ticker.items():
        n_trades = len(count_list)
        if ticker not in available_dates or len(available_dates[ticker]) < n_trades:
            continue
        chosen_idx = np.random.choice(len(available_dates[ticker]), size=n_trades, replace=False)
        for idx in chosen_idx:
            random_entries.append((ticker, available_dates[ticker][idx]))

    if len(random_entries) == 0:
        random_sharpes.append(0.0)
        continue

    r = run_backtest(custom_entries=random_entries)
    random_sharpes.append(r["sharpe"])

    if (i + 1) % 200 == 0:
        print(f"  ... {i+1}/{N_PERM} permutations done")

real_sharpe = baseline["sharpe"]
percentile = np.mean([1 for s in random_sharpes if real_sharpe > s]) / len(random_sharpes) * 100
test2_pass = percentile >= 90.0

print(f"  Real Sharpe: {real_sharpe:.4f}")
print(f"  Random Sharpe mean: {np.mean(random_sharpes):.4f}")
print(f"  Random Sharpe std: {np.std(random_sharpes):.4f}")
print(f"  Percentile: {percentile:.1f}th")
print(f"  Threshold: >= 90th percentile")
print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

results["tests"]["2_random_timing"] = {
    "real_sharpe": real_sharpe,
    "random_sharpe_mean": float(round(np.mean(random_sharpes), 4)),
    "random_sharpe_std": float(round(np.std(random_sharpes), 4)),
    "percentile": float(round(percentile, 1)),
    "threshold": ">= 90th percentile",
    "pass": test2_pass,
}

# ── TEST 3: Look-Ahead Bias ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3: LOOK-AHEAD BIAS (drop last 20% of data)")
print("=" * 70)

# Calculate 80% cutoff date
oot_start = pd.Timestamp(OOT_START)
oot_end = pd.Timestamp(END)
total_days = (oot_end - oot_start).days
cutoff_date = oot_start + timedelta(days=int(total_days * 0.8))

truncated = run_backtest(end_date=cutoff_date)
trunc_sharpe = truncated["sharpe"]

if real_sharpe != 0:
    pct_drop = (real_sharpe - trunc_sharpe) / abs(real_sharpe) * 100
else:
    pct_drop = 0.0

test3_pass = pct_drop <= 30.0

print(f"  Full period Sharpe: {real_sharpe:.4f}")
print(f"  First 80% Sharpe: {trunc_sharpe:.4f}")
print(f"  Cutoff date: {cutoff_date.date()}")
print(f"  Sharpe drop: {pct_drop:.1f}%")
print(f"  Threshold: drop <= 30%")
print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

results["tests"]["3_look_ahead_bias"] = {
    "full_sharpe": real_sharpe,
    "truncated_sharpe": trunc_sharpe,
    "cutoff_date": str(cutoff_date.date()),
    "pct_drop": float(round(pct_drop, 1)),
    "threshold": "drop <= 30%",
    "pass": test3_pass,
}

# ── TEST 4: Cost Sensitivity ────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 4: COST SENSITIVITY")
print("=" * 70)

cost_levels = [0.0005, 0.001, 0.0015, 0.002]  # 0.05%, 0.10%, 0.15%, 0.20%
cost_sharpes = {}
for slip in cost_levels:
    r = run_backtest(slippage=slip)
    label = f"{slip*100:.2f}%"
    cost_sharpes[label] = r["sharpe"]
    print(f"  Slippage {label}: Sharpe = {r['sharpe']:.4f}")

worst_sharpe = cost_sharpes.get("0.20%", 0.0)
test4_pass = worst_sharpe >= 0.5

print(f"  Threshold: Sharpe >= 0.5 at 0.20% slippage")
print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

results["tests"]["4_cost_sensitivity"] = {
    "sharpe_by_slippage": cost_sharpes,
    "sharpe_at_020pct": worst_sharpe,
    "threshold": "Sharpe >= 0.5 at 0.20%",
    "pass": test4_pass,
}

# ── TEST 5: Sub-Period Stability ─────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 5: SUB-PERIOD STABILITY (4 equal sub-periods)")
print("=" * 70)

period_length = total_days // 4
sub_periods = []
for i in range(4):
    sp_start = oot_start + timedelta(days=i * period_length)
    sp_end = oot_start + timedelta(days=(i + 1) * period_length) if i < 3 else oot_end
    sub_periods.append((sp_start, sp_end))

sub_sharpes = {}
all_positive = True
for i, (sp_start, sp_end) in enumerate(sub_periods):
    r = run_backtest(start_date=sp_start, end_date=sp_end)
    label = f"P{i+1} ({sp_start.date()} to {sp_end.date()})"
    sub_sharpes[label] = {"sharpe": r["sharpe"], "n_trades": r["n_trades"],
                          "total_return": r["total_return"]}
    if r["sharpe"] <= 0:
        all_positive = False
    print(f"  {label}: Sharpe={r['sharpe']:.4f}, trades={r['n_trades']}, ret={r['total_return']:.2%}")

test5_pass = all_positive

print(f"  Threshold: ALL sub-periods must have positive Sharpe")
print(f"  Result: {'PASS' if test5_pass else 'FAIL'}")

results["tests"]["5_sub_period_stability"] = {
    "sub_period_sharpes": sub_sharpes,
    "all_positive": all_positive,
    "threshold": "all sub-periods Sharpe > 0",
    "pass": test5_pass,
}

# ── TEST 6: Parameter Sensitivity ────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 6: PARAMETER SENSITIVITY (125 combinations)")
print("=" * 70)

beat_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
gap_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07]
hold_periods = [20, 30, 40, 50, 60]

param_results = []
total_combos = len(beat_thresholds) * len(gap_thresholds) * len(hold_periods)
count = 0
above_05 = 0

for bt in beat_thresholds:
    for gt in gap_thresholds:
        for hp in hold_periods:
            r = run_backtest(beat_thresh=bt, gap_thresh=gt, hold_days=hp)
            param_results.append({
                "beat_thresh": bt,
                "gap_thresh": gt,
                "hold_days": hp,
                "sharpe": r["sharpe"],
                "n_trades": r["n_trades"],
            })
            if r["sharpe"] > 0.5:
                above_05 += 1
            count += 1
            if count % 25 == 0:
                print(f"  ... {count}/{total_combos} combinations done")

pct_above = above_05 / total_combos * 100
test6_pass = pct_above >= 40.0

print(f"  Combinations with Sharpe > 0.5: {above_05}/{total_combos} ({pct_above:.1f}%)")
print(f"  Threshold: >= 40%")
print(f"  Result: {'PASS' if test6_pass else 'FAIL'}")

# Find best and worst combos
param_results.sort(key=lambda x: x["sharpe"], reverse=True)
best = param_results[0]
worst = param_results[-1]

results["tests"]["6_parameter_sensitivity"] = {
    "pct_above_05": float(round(pct_above, 1)),
    "combos_above_05": above_05,
    "total_combos": total_combos,
    "best_combo": best,
    "worst_combo": worst,
    "threshold": ">= 40% with Sharpe > 0.5",
    "pass": test6_pass,
}

# ── Final Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION SUMMARY: Earnings Surprise Momentum")
print("=" * 70)

test_names = {
    "1_inverse_direction": "Inverse Direction",
    "2_random_timing": "Random Timing",
    "3_look_ahead_bias": "Look-Ahead Bias",
    "4_cost_sensitivity": "Cost Sensitivity",
    "5_sub_period_stability": "Sub-Period Stability",
    "6_parameter_sensitivity": "Parameter Sensitivity",
}

passes = 0
for key, name in test_names.items():
    passed = results["tests"][key]["pass"]
    if passed:
        passes += 1
    status = "PASS" if passed else "FAIL"
    print(f"  {name:30s} [{status}]")

print(f"\n  Overall: {passes}/6 tests passed")
overall = passes >= 5
results["overall_pass"] = overall
results["tests_passed"] = passes
results["tests_total"] = 6
print(f"  Verdict: {'PASS (robust edge)' if overall else 'FAIL (edge not robust)'}")

# ── Save Results ─────────────────────────────────────────────────────────
out_path = Path("/home/jupiter/Lvl3Quant/data/earnings_surprise_adversarial_results.json")
# Convert any non-serializable types
def sanitize(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, bool):
        return bool(obj)
    return obj

def deep_sanitize(d):
    if isinstance(d, dict):
        return {k: deep_sanitize(v) for k, v in d.items()}
    if isinstance(d, list):
        return [deep_sanitize(v) for v in d]
    return sanitize(d)

results_clean = deep_sanitize(results)
# Remove full trade lists to keep JSON manageable
if "trades" in results_clean.get("baseline", {}):
    del results_clean["baseline"]["trades"]

with open(out_path, "w") as f:
    json.dump(results_clean, f, indent=2)

print(f"\nResults saved to {out_path}")
