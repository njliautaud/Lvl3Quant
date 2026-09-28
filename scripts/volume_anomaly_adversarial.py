#!/usr/bin/env python3
"""
Volume Anomaly — Adversarial Validation Suite
----------------------------------------------
6 adversarial tests on both Variant A and Variant E.

Test 1: Inverse Signal — does the opposite also work?
Test 2: Random Instruments — 13 large-cap single stocks
Test 3: Sub-Period Stability — 2022, 2023-2024, 2025-2026
Test 4: Top Trade Removal — remove best 3 trades
Test 5: Parameter Sensitivity — nearby params
Test 6: Sharpe Inflation Check — active-days Sharpe, Calmar, per-trade Sharpe

Scoring: Pass/Fail per test, X/6 per variant.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
RANDOM_STOCKS = ["JPM", "JNJ", "PG", "KO", "WMT", "HD", "MCD", "UNH", "V", "MA", "COST", "PEP", "ABT"]
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2021-01-01"
STARTING_CAPITAL = 645.0
MAX_POSITIONS = 3
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/volume_anomaly_adversarial_results.json")

SUB_PERIODS = [
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023-2024", "2023-01-01", "2025-01-01"),
    ("2025-2026", "2025-01-01", "2026-08-01"),
]

# ── Data Download ───────────────────────────────────────────────────────────
ALL_TICKERS = sorted(set(UNIVERSE + RANDOM_STOCKS))
print(f"Downloading data for {len(ALL_TICKERS)} tickers...")
raw = yf.download(ALL_TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    close_all = raw["Close"]
    open_all = raw["Open"]
    volume_all = raw["Volume"]
else:
    close_all = raw[["Close"]].copy()
    close_all.columns = [ALL_TICKERS[0]]
    open_all = raw[["Open"]].copy()
    open_all.columns = [ALL_TICKERS[0]]
    volume_all = raw[["Volume"]].copy()
    volume_all.columns = [ALL_TICKERS[0]]

print(f"  Data shape: {close_all.shape[0]} days x {close_all.shape[1]} tickers")
print(f"  Date range: {close_all.index[0].date()} to {close_all.index[-1].date()}")

# ── Precompute ──────────────────────────────────────────────────────────────
vol_avg_20_all = volume_all.rolling(20).mean()
oot_mask = close_all.index >= OOT_START

# SPY regime
spy_sma_200 = close_all["SPY"].rolling(200).mean()
regime_bull = close_all["SPY"] > spy_sma_200


# ── Signal generators ──────────────────────────────────────────────────────
def gen_signals_A(universe, vol_mult=2.0, hold=5, bullish=True):
    """
    Variant A: Buy when volume > vol_mult * 20d avg AND close > open (bullish).
    Inverse: close < open (bearish high-vol day).
    """
    signals = []
    for ticker in universe:
        if ticker not in close_all.columns:
            continue
        v = volume_all[ticker]
        va = vol_avg_20_all[ticker]
        c = close_all[ticker]
        o = open_all[ticker]
        if bullish:
            cond = oot_mask & (v > vol_mult * va) & (c > o) & va.notna()
        else:
            cond = oot_mask & (v > vol_mult * va) & (c < o) & va.notna()
        for d in close_all.index[cond]:
            signals.append({"ticker": ticker, "entry_date": d, "hold": hold})
    return signals


def gen_signals_E(universe, consec_days=3, vol_mult=1.5, hold=10):
    """Variant E: consec_days consecutive days of volume > vol_mult * 20d avg."""
    signals = []
    for ticker in universe:
        if ticker not in close_all.columns:
            continue
        v = volume_all[ticker]
        va = vol_avg_20_all[ticker]
        high_vol = (v > vol_mult * va).astype(int)
        consec = high_vol.rolling(consec_days).sum()
        cond = oot_mask & (consec >= consec_days) & va.notna()
        for d in close_all.index[cond]:
            signals.append({"ticker": ticker, "entry_date": d, "hold": hold})
    return signals


def gen_signals_E_inverse(universe, consec_days=3, vol_mult_low=0.5, hold=10):
    """Inverse of E: consec_days consecutive days of LOW volume (< vol_mult_low * avg)."""
    signals = []
    for ticker in universe:
        if ticker not in close_all.columns:
            continue
        v = volume_all[ticker]
        va = vol_avg_20_all[ticker]
        low_vol = (v < vol_mult_low * va).astype(int)
        consec = low_vol.rolling(consec_days).sum()
        cond = oot_mask & (consec >= consec_days) & va.notna()
        for d in close_all.index[cond]:
            signals.append({"ticker": ticker, "entry_date": d, "hold": hold})
    return signals


# ── Backtest engine ─────────────────────────────────────────────────────────
def backtest(signals, start_date=None, end_date=None):
    """
    Portfolio backtest with max 3 simultaneous positions, equal allocation.
    Returns dict of metrics including raw trade returns list.
    """
    if not signals:
        return _empty_result()

    trades = []
    for sig in signals:
        ticker = sig["ticker"]
        entry_date = sig["entry_date"]
        hold = sig["hold"]
        if entry_date not in close_all.index or ticker not in close_all.columns:
            continue
        if start_date and entry_date < pd.Timestamp(start_date):
            continue
        if end_date and entry_date >= pd.Timestamp(end_date):
            continue

        loc = close_all.index.get_loc(entry_date)
        entry_loc = loc + 1
        exit_loc = entry_loc + hold
        if exit_loc >= len(close_all):
            continue
        entry_price = close_all[ticker].iloc[entry_loc]
        exit_price = close_all[ticker].iloc[exit_loc]
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue
        ret = (exit_price - entry_price) / entry_price
        entry_actual = close_all.index[entry_loc]
        is_bull = regime_bull.loc[entry_actual] if entry_actual in regime_bull.index else True
        trades.append({
            "ticker": ticker, "entry_date": entry_actual,
            "exit_date": close_all.index[exit_loc],
            "ret": ret, "bull": is_bull, "hold": hold,
        })

    if not trades:
        return _empty_result()

    trades_df = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

    # Max positions constraint
    active_exits = []
    accepted = []
    for _, t in trades_df.iterrows():
        active_exits = [ex for ex in active_exits if ex > t["entry_date"]]
        if len(active_exits) < MAX_POSITIONS:
            active_exits.append(t["exit_date"])
            accepted.append(t)

    if not accepted:
        return _empty_result()

    at_df = pd.DataFrame(accepted)
    rets = at_df["ret"].values
    n_trades = len(rets)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    wr = len(wins) / n_trades
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss

    # Equity curve
    capital = STARTING_CAPITAL
    eq = [capital]
    for r in rets:
        alloc = capital / MAX_POSITIONS
        capital += alloc * r
        eq.append(capital)

    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = dd.min()

    # Annualize
    avg_hold = at_df["hold"].mean()
    trades_per_year = 252 / avg_hold if avg_hold > 0 else 50
    mean_ret = rets.mean()
    std_ret = rets.std() if rets.std() > 0 else 1e-9
    downside = rets[rets < 0]
    downside_std = downside.std() if len(downside) > 1 and downside.std() > 0 else 1e-9

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)
    total_return_pct = (eq[-1] - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    # Regime split
    bull_rets = at_df[at_df["bull"] == True]["ret"].values
    bear_rets = at_df[at_df["bull"] == False]["ret"].values

    def _sharpe(r):
        if len(r) < 2 or r.std() == 0:
            return 0.0
        return (r.mean() / r.std()) * np.sqrt(trades_per_year)

    sharpe_bull = _sharpe(bull_rets)
    sharpe_bear = _sharpe(bear_rets)
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "maxdd": round(maxdd * 100, 2),
        "n_trades": n_trades,
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "total_return_pct": round(total_return_pct, 2),
        "avg_hold": round(avg_hold, 1),
        "trade_returns": rets.tolist(),
    }


def _empty_result():
    return {
        "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "maxdd": 0,
        "n_trades": 0, "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 0,
        "total_return_pct": 0, "avg_hold": 0, "trade_returns": [],
    }


def clean(res):
    """Remove trade_returns for JSON storage."""
    return {k: v for k, v in res.items() if k != "trade_returns"}


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ADVERSARIAL VALIDATION — VOLUME ANOMALY")
print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL}")
print("=" * 70)

print("\n--- Computing baselines ---")
sig_A = gen_signals_A(UNIVERSE, vol_mult=2.0, hold=5, bullish=True)
res_A = backtest(sig_A)
print(f"[A] 2x Vol Bull 5d: Sharpe={res_A['sharpe']:.3f}, WR={res_A['wr']:.1%}, "
      f"Trades={res_A['n_trades']}, Ret={res_A['total_return_pct']:.1f}%")

sig_E = gen_signals_E(UNIVERSE, consec_days=3, vol_mult=1.5, hold=10)
res_E = backtest(sig_E)
print(f"[E] Multi-day Surge 10d: Sharpe={res_E['sharpe']:.3f}, WR={res_E['wr']:.1%}, "
      f"Trades={res_E['n_trades']}, Ret={res_E['total_return_pct']:.1f}%")

all_tests = {"A": {}, "E": {}}

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1: INVERSE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: INVERSE SIGNAL")
print("  A-inverse: Buy on high-vol BEARISH day (close < open), same 5d hold")
print("  E-inverse: Buy on LOW volume (< 0.5x avg) for 3 consecutive days, same 10d hold")
print("  FAIL if inverse Sharpe > 0.5")
print("=" * 70)

# A inverse: high volume bearish day
sig_A_inv = gen_signals_A(UNIVERSE, vol_mult=2.0, hold=5, bullish=False)
res_A_inv = backtest(sig_A_inv)
a1_pass = res_A_inv["sharpe"] <= 0.5
print(f"[A] Inverse: Sharpe={res_A_inv['sharpe']:.3f}, Trades={res_A_inv['n_trades']}, "
      f"WR={res_A_inv['wr']:.1%} -> {'PASS' if a1_pass else 'FAIL'}")

# E inverse: low volume for 3 consecutive days
sig_E_inv = gen_signals_E_inverse(UNIVERSE, consec_days=3, vol_mult_low=0.5, hold=10)
res_E_inv = backtest(sig_E_inv)
e1_pass = res_E_inv["sharpe"] <= 0.5
print(f"[E] Inverse: Sharpe={res_E_inv['sharpe']:.3f}, Trades={res_E_inv['n_trades']}, "
      f"WR={res_E_inv['wr']:.1%} -> {'PASS' if e1_pass else 'FAIL'}")

all_tests["A"]["test1_inverse"] = {
    "pass": bool(a1_pass), "inverse_sharpe": res_A_inv["sharpe"],
    "threshold": 0.5, "detail": clean(res_A_inv),
}
all_tests["E"]["test1_inverse"] = {
    "pass": bool(e1_pass), "inverse_sharpe": res_E_inv["sharpe"],
    "threshold": 0.5, "detail": clean(res_E_inv),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2: RANDOM INSTRUMENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: RANDOM INSTRUMENTS")
print(f"  Using 13 large-cap stocks: {', '.join(RANDOM_STOCKS)}")
print("  FAIL if random Sharpe > 80% of original")
print("=" * 70)

sig_A_rand = gen_signals_A(RANDOM_STOCKS, vol_mult=2.0, hold=5, bullish=True)
res_A_rand = backtest(sig_A_rand)
a2_thresh = 0.8 * res_A["sharpe"]
a2_pass = res_A_rand["sharpe"] <= a2_thresh
print(f"[A] Random stocks: Sharpe={res_A_rand['sharpe']:.3f} (threshold={a2_thresh:.3f}), "
      f"Trades={res_A_rand['n_trades']} -> {'PASS' if a2_pass else 'FAIL'}")

sig_E_rand = gen_signals_E(RANDOM_STOCKS, consec_days=3, vol_mult=1.5, hold=10)
res_E_rand = backtest(sig_E_rand)
e2_thresh = 0.8 * res_E["sharpe"]
e2_pass = res_E_rand["sharpe"] <= e2_thresh
print(f"[E] Random stocks: Sharpe={res_E_rand['sharpe']:.3f} (threshold={e2_thresh:.3f}), "
      f"Trades={res_E_rand['n_trades']} -> {'PASS' if e2_pass else 'FAIL'}")

all_tests["A"]["test2_random_instruments"] = {
    "pass": bool(a2_pass), "random_sharpe": res_A_rand["sharpe"],
    "original_sharpe": res_A["sharpe"], "threshold_pct": 0.8,
    "stocks_used": RANDOM_STOCKS, "detail": clean(res_A_rand),
}
all_tests["E"]["test2_random_instruments"] = {
    "pass": bool(e2_pass), "random_sharpe": res_E_rand["sharpe"],
    "original_sharpe": res_E["sharpe"], "threshold_pct": 0.8,
    "stocks_used": RANDOM_STOCKS, "detail": clean(res_E_rand),
}

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: SUB-PERIOD STABILITY")
print("  Periods: 2022, 2023-2024, 2025-2026")
print("  FAIL if any sub-period has negative Sharpe")
print("=" * 70)

for vname, sigs, res_base in [("A", sig_A, res_A), ("E", sig_E, res_E)]:
    sub_results = {}
    any_neg = False
    for pname, sd, ed in SUB_PERIODS:
        sr = backtest(sigs, start_date=sd, end_date=ed)
        sub_results[pname] = clean(sr)
        if sr["sharpe"] < 0:
            any_neg = True
        print(f"[{vname}] {pname}: Sharpe={sr['sharpe']:.3f}, Trades={sr['n_trades']}, "
              f"WR={sr['wr']:.1%}, Ret={sr['total_return_pct']:.1f}%")

    t3_pass = not any_neg
    print(f"    -> {'PASS' if t3_pass else 'FAIL'}")
    all_tests[vname]["test3_subperiod"] = {
        "pass": bool(t3_pass), "sub_periods": sub_results,
        "criterion": "all sub-period Sharpe >= 0",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4: TOP TRADE REMOVAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: TOP TRADE REMOVAL (remove best 3 trades)")
print("  FAIL if trimmed Sharpe drops below 0.5")
print("=" * 70)

for vname, res_base in [("A", res_A), ("E", res_E)]:
    rets = np.array(res_base["trade_returns"])
    avg_hold = res_base["avg_hold"]
    tpy = 252 / avg_hold if avg_hold > 0 else 50

    if len(rets) <= 3:
        trimmed_sharpe = 0.0
        top3 = rets.tolist()
    else:
        sorted_idx = np.argsort(rets)[::-1]
        top3 = rets[sorted_idx[:3]]
        kept = rets[np.sort(sorted_idx[3:])]
        std_k = kept.std() if kept.std() > 0 else 1e-9
        trimmed_sharpe = (kept.mean() / std_k) * np.sqrt(tpy)

    t4_pass = trimmed_sharpe >= 0.5
    print(f"[{vname}] Original Sharpe={res_base['sharpe']:.3f}, "
          f"Trimmed={trimmed_sharpe:.3f} (removed: {[f'{r*100:.2f}%' for r in top3]})")
    print(f"    -> {'PASS' if t4_pass else 'FAIL'}")

    all_tests[vname]["test4_top_trade_removal"] = {
        "pass": bool(t4_pass), "original_sharpe": res_base["sharpe"],
        "trimmed_sharpe": round(trimmed_sharpe, 3),
        "top_3_removed_pct": [round(r * 100, 2) for r in top3],
        "criterion": "trimmed Sharpe >= 0.5",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: PARAMETER SENSITIVITY")
print("  A: vol thresholds {1.5x, 2.5x} x holds {3d, 7d} = 4 combos")
print("  E: consec days {2, 4} x holds {7d, 14d} = 4 combos")
print("  FAIL if >50% of variations have Sharpe < 0.5")
print("=" * 70)

# Variant A: vol_mult in {1.5, 2.5}, hold in {3, 7}
print("[A] Parameter sweep:")
a_param_results = {}
a_below = 0
a_total = 0
for vm in [1.5, 2.5]:
    for hd in [3, 7]:
        sig = gen_signals_A(UNIVERSE, vol_mult=vm, hold=hd, bullish=True)
        res = backtest(sig)
        key = f"vol{vm}x_hold{hd}d"
        a_param_results[key] = {"sharpe": res["sharpe"], "n_trades": res["n_trades"],
                                "wr": round(res["wr"], 3)}
        a_total += 1
        if res["sharpe"] < 0.5:
            a_below += 1
        print(f"    vol={vm}x hold={hd}d: Sharpe={res['sharpe']:.3f}, "
              f"Trades={res['n_trades']}, WR={res['wr']:.1%}")

a5_pass = (a_below / a_total) <= 0.5
print(f"    {a_below}/{a_total} below 0.5 -> {'PASS' if a5_pass else 'FAIL'}")

all_tests["A"]["test5_param_sensitivity"] = {
    "pass": bool(a5_pass), "below_threshold": a_below, "total_variations": a_total,
    "pct_below": round(a_below / a_total * 100, 1), "variations": a_param_results,
    "criterion": "<=50% of variations have Sharpe < 0.5",
}

# Variant E: consec_days in {2, 4}, hold in {7, 14}
print("[E] Parameter sweep:")
e_param_results = {}
e_below = 0
e_total = 0
for cd in [2, 4]:
    for hd in [7, 14]:
        sig = gen_signals_E(UNIVERSE, consec_days=cd, vol_mult=1.5, hold=hd)
        res = backtest(sig)
        key = f"consec{cd}d_hold{hd}d"
        e_param_results[key] = {"sharpe": res["sharpe"], "n_trades": res["n_trades"],
                                "wr": round(res["wr"], 3)}
        e_total += 1
        if res["sharpe"] < 0.5:
            e_below += 1
        print(f"    consec={cd}d hold={hd}d: Sharpe={res['sharpe']:.3f}, "
              f"Trades={res['n_trades']}, WR={res['wr']:.1%}")

e5_pass = (e_below / e_total) <= 0.5
print(f"    {e_below}/{e_total} below 0.5 -> {'PASS' if e5_pass else 'FAIL'}")

all_tests["E"]["test5_param_sensitivity"] = {
    "pass": bool(e5_pass), "below_threshold": e_below, "total_variations": e_total,
    "pct_below": round(e_below / e_total * 100, 1), "variations": e_param_results,
    "criterion": "<=50% of variations have Sharpe < 0.5",
}

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6: SHARPE INFLATION CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: SHARPE INFLATION CHECK")
print("  Sparse trading can inflate Sharpe via deflated denominator")
print("  Reports: active-days-only Sharpe, Calmar, per-trade Sharpe")
print("  FAIL if activity-adjusted Sharpe < 0.5 while standard > 1.0")
print("=" * 70)

oot_trading_days = len(close_all.index[oot_mask])

for vname, res_base in [("A", res_A), ("E", res_E)]:
    rets = np.array(res_base["trade_returns"])
    n = len(rets)
    avg_hold = res_base["avg_hold"]

    # Active days estimate
    active_days = n * avg_hold
    active_frac = active_days / oot_trading_days if oot_trading_days > 0 else 0

    # Per-trade Sharpe (un-annualized)
    mean_r = rets.mean()
    std_r = rets.std() if rets.std() > 0 else 1e-9
    per_trade_sharpe = mean_r / std_r

    # Calmar: total return / max drawdown
    total_ret = res_base["total_return_pct"] / 100
    maxdd_abs = abs(res_base["maxdd"]) / 100
    calmar = total_ret / maxdd_abs if maxdd_abs > 1e-9 else 999.0

    # Activity-adjusted Sharpe: penalize for sparse trading
    adjusted_sharpe = res_base["sharpe"] * np.sqrt(active_frac) if active_frac > 0 else 0

    # Average return per trade
    avg_ret_per_trade = mean_r * 100

    print(f"[{vname}] Standard Sharpe: {res_base['sharpe']:.3f}")
    print(f"    Active fraction: {active_frac:.1%} ({int(active_days)}/{oot_trading_days} days)")
    print(f"    Activity-adjusted Sharpe: {adjusted_sharpe:.3f}")
    print(f"    Per-trade Sharpe (no annualization): {per_trade_sharpe:.4f}")
    print(f"    Average return per trade: {avg_ret_per_trade:.3f}%")
    print(f"    Calmar ratio (return/maxDD): {calmar:.3f}")
    print(f"    Total return: {res_base['total_return_pct']:.1f}%, MaxDD: {res_base['maxdd']:.1f}%")

    inflation_flag = (res_base["sharpe"] > 1.0 and adjusted_sharpe < 0.5)
    t6_pass = not inflation_flag
    print(f"    -> {'PASS' if t6_pass else 'FAIL'} "
          f"({'inflation detected' if inflation_flag else 'no inflation'})")

    all_tests[vname]["test6_sharpe_inflation"] = {
        "pass": bool(t6_pass),
        "standard_sharpe": res_base["sharpe"],
        "activity_adjusted_sharpe": round(adjusted_sharpe, 3),
        "per_trade_sharpe": round(per_trade_sharpe, 4),
        "avg_return_per_trade_pct": round(avg_ret_per_trade, 3),
        "calmar": round(calmar, 3),
        "active_fraction": round(active_frac, 3),
        "active_days": int(active_days),
        "oot_trading_days": oot_trading_days,
        "inflation_flagged": bool(inflation_flag),
        "criterion": "no inflation (adjusted Sharpe >= 0.5 if standard > 1.0)",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL ADVERSARIAL SCORECARD")
print("=" * 70)

TEST_KEYS = [
    ("test1_inverse", "Inverse Signal"),
    ("test2_random_instruments", "Random Instruments"),
    ("test3_subperiod", "Sub-Period Stability"),
    ("test4_top_trade_removal", "Top Trade Removal"),
    ("test5_param_sensitivity", "Parameter Sensitivity"),
    ("test6_sharpe_inflation", "Sharpe Inflation Check"),
]

for vname in ["A", "E"]:
    passed = sum(1 for tk, _ in TEST_KEYS if all_tests[vname][tk]["pass"])
    total = len(TEST_KEYS)
    all_tests[vname]["score"] = f"{passed}/{total}"
    all_tests[vname]["passed"] = passed
    all_tests[vname]["total"] = total

    print(f"\nVariant {vname}: {passed}/{total}")
    for tk, label in TEST_KEYS:
        status = "PASS" if all_tests[vname][tk]["pass"] else "FAIL"
        print(f"  {status}  {label}")

# ── Save results ────────────────────────────────────────────────────────────
output = {
    "test": "volume_anomaly_adversarial_validation",
    "run_timestamp": datetime.now().isoformat(),
    "oot_period": f"{OOT_START} to {OOT_END}",
    "universe": UNIVERSE,
    "random_stocks": RANDOM_STOCKS,
    "starting_capital": STARTING_CAPITAL,
    "max_positions": MAX_POSITIONS,
    "baselines": {
        "A": clean(res_A),
        "E": clean(res_E),
    },
    "adversarial_tests": all_tests,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_PATH}")
print("Done.")
