#!/usr/bin/env python3
"""
Adversarial Validation: Volatility Clustering Variant E
========================================================
6 adversarial checks to stress-test whether the BB-width contraction edge is real.

1) Inverse Signal — buy when BB width is HIGHEST 10% + price touches UPPER band
2) Random Entry Timing — 1000 random entry sets, percentile rank
3) Sub-Period Stability — 4 sub-periods, all must have positive Sharpe
4) Remove Top 3 Tickers — Sharpe drop must be < 50%
5) Parameter Sensitivity — BB width percentile x hold period grid
6) Cost Sensitivity — slippage sweep to find Sharpe < 0.3 breakpoint

Strategy under test:
  BB width (20,2) contracts to lowest 10% of 252-day range AND price touches lower BB.
  Hold 10 days. Quality stock universe.

OOT: Jan 2022 - Jul 2026. $645 account, max $200/trade, max 3 concurrent positions.
Slippage: 0.02% each way (baseline).
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
SLIPPAGE_PCT = 0.0002  # 0.02% each way
HOLD_DAYS = 10
BB_PERIOD = 20
BB_STD = 2
BB_WIDTH_PCTILE = 10  # lowest 10% of 252-day range
LOOKBACK_RANGE = 252

START = "2020-01-01"  # extra data for indicator warmup
END = "2026-07-31"
OOT_START = "2022-01-01"
N_PERM = 1000

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

ALL_TICKERS = sorted(set(TICKERS + ["SPY"]))

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
spy_close = closes.get("SPY", pd.Series(dtype=float))

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_bb(series, period=20, num_std=2):
    """Return upper_band, lower_band, bb_width."""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma  # normalized BB width
    return upper, lower, width


def get_regime(date, spy_prices):
    """Bull if SPY > 200-SMA, else Bear."""
    if date not in spy_prices.index:
        idx = spy_prices.index.get_indexer([date], method="ffill")[0]
        if idx < 0:
            return "bull"
    else:
        idx = spy_prices.index.get_loc(date)
    if idx < 200:
        return "bull"
    sma200 = spy_prices.iloc[max(0, idx - 199):idx + 1].mean()
    return "bull" if spy_prices.iloc[idx] > sma200 else "bear"


# ── Core Signal Generator ────────────────────────────────────────────────
def generate_signals(tickers, bb_width_pctile=BB_WIDTH_PCTILE, hold_days=HOLD_DAYS,
                     inverse=False):
    """
    Generate trade signals.
    Normal: BB width in lowest `bb_width_pctile`% of 252-day range AND price <= lower BB.
    Inverse: BB width in highest `bb_width_pctile`% AND price >= upper BB.
    Returns list of dicts with entry_date, exit_date, ticker, entry_price, exit_price.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)

    for ticker in tickers:
        px = closes.get(ticker)
        if px is None or len(px) < LOOKBACK_RANGE + BB_PERIOD + hold_days:
            continue

        upper, lower, width = calc_bb(px, BB_PERIOD, BB_STD)

        # Rolling percentile of BB width over 252 days
        width_pctile = width.rolling(LOOKBACK_RANGE).rank(pct=True) * 100

        for i in range(LOOKBACK_RANGE + BB_PERIOD, len(px) - hold_days):
            date = px.index[i]
            if date < oot_start:
                continue

            wp = width_pctile.iloc[i]
            price = px.iloc[i]
            ub = upper.iloc[i]
            lb = lower.iloc[i]

            if pd.isna(wp) or pd.isna(price) or pd.isna(ub) or pd.isna(lb):
                continue

            if inverse:
                # Buy when width is HIGHEST and price touches UPPER band
                triggered = (wp >= (100 - bb_width_pctile)) and (price >= ub)
            else:
                # Buy when width is LOWEST and price touches LOWER band
                triggered = (wp <= bb_width_pctile) and (price <= lb)

            if triggered:
                exit_idx = min(i + hold_days, len(px) - 1)
                entry_price = price
                exit_price = px.iloc[exit_idx]
                trades.append({
                    "ticker": ticker,
                    "entry_date": date,
                    "exit_date": px.index[exit_idx],
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                })

    return trades


# ── Portfolio Simulator ───────────────────────────────────────────────────
def simulate_portfolio(trades, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                       max_concurrent=MAX_CONCURRENT, slippage_pct=SLIPPAGE_PCT):
    """
    Simulate with position sizing, concurrency limits, and slippage.
    Returns equity curve (Series) and trade-level results.
    """
    if not trades:
        return pd.Series(dtype=float), []

    # Sort by entry date
    trades_sorted = sorted(trades, key=lambda t: t["entry_date"])

    # Remove overlapping trades (max concurrent)
    active_exits = []
    selected = []
    for t in trades_sorted:
        # Remove expired positions
        active_exits = [e for e in active_exits if e > t["entry_date"]]
        if len(active_exits) < max_concurrent:
            selected.append(t)
            active_exits.append(t["exit_date"])

    # Simulate P&L
    results = []
    equity = capital
    equity_curve = {}

    for t in selected:
        position_size = min(max_per_trade, equity * 0.95)  # don't go all-in
        if position_size < 10:
            continue

        entry_cost = t["entry_price"] * (1 + slippage_pct)
        exit_proceeds = t["exit_price"] * (1 - slippage_pct)
        shares = int(position_size / entry_cost)
        if shares < 1:
            continue

        pnl = shares * (exit_proceeds - entry_cost)
        ret = (exit_proceeds - entry_cost) / entry_cost
        equity += pnl

        results.append({
            "ticker": t["ticker"],
            "entry_date": t["entry_date"],
            "exit_date": t["exit_date"],
            "pnl": pnl,
            "return": ret,
            "equity": equity,
        })
        equity_curve[t["exit_date"]] = equity

    return pd.Series(equity_curve).sort_index(), results


def calc_metrics(results, equity_curve, capital=CAPITAL):
    """Calculate Sharpe, Sortino, WR, PF, MDD from trade results."""
    if not results:
        return {"sharpe": 0, "sortino": 0, "wr": 0, "pf": 0, "mdd": 0,
                "n_trades": 0, "total_return": 0}

    returns = [r["return"] for r in results]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    wr = len(wins) / len(returns) * 100 if returns else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1e-9
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 99.9

    # Annualize: assume ~25 trades/year for this strategy
    mean_ret = np.mean(returns)
    std_ret = np.std(returns) if len(returns) > 1 else 1e-9
    downside = np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 1e-9

    trades_per_year = max(len(returns) / 4.5, 1)  # ~4.5 years of OOT
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-9 else 0
    sortino = (mean_ret / downside) * np.sqrt(trades_per_year) if downside > 1e-9 else 0

    # MDD from equity curve
    if len(equity_curve) > 0:
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak
        mdd = dd.min() * 100
    else:
        mdd = 0

    total_pnl = sum(r["pnl"] for r in results)
    total_return = total_pnl / capital * 100

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "wr": round(float(wr), 1),
        "pf": round(float(pf), 2),
        "mdd": round(float(mdd), 1),
        "n_trades": len(results),
        "total_return": round(float(total_return), 1),
    }


def calc_regime_sharpes(results):
    """Split results by bull/bear regime and compute per-regime Sharpe."""
    bull_rets, bear_rets = [], []
    for r in results:
        regime = get_regime(r["entry_date"], spy_close)
        if regime == "bull":
            bull_rets.append(r["return"])
        else:
            bear_rets.append(r["return"])

    def sharpe_from_rets(rets):
        if len(rets) < 3:
            return 0.0
        m = np.mean(rets)
        s = np.std(rets)
        n_per_yr = max(len(rets) / 4.5, 1)
        return float((m / s) * np.sqrt(n_per_yr)) if s > 1e-9 else 0.0

    return {
        "bull_sharpe": round(sharpe_from_rets(bull_rets), 3),
        "bear_sharpe": round(sharpe_from_rets(bear_rets), 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


# ── Run Baseline ──────────────────────────────────────────────────────────
print("\n═══ BASELINE ═══")
base_trades = generate_signals(TICKERS)
base_eq, base_results = simulate_portfolio(base_trades)
base_metrics = calc_metrics(base_results, base_eq)
base_regimes = calc_regime_sharpes(base_results)
print(f"  Trades: {base_metrics['n_trades']}, Sharpe: {base_metrics['sharpe']}, "
      f"WR: {base_metrics['wr']}%, PF: {base_metrics['pf']}, MDD: {base_metrics['mdd']}%")
print(f"  Bull Sharpe: {base_regimes['bull_sharpe']}, Bear Sharpe: {base_regimes['bear_sharpe']}")

adversarial_results = {
    "strategy": "Volatility Clustering Variant E",
    "baseline": {**base_metrics, **base_regimes},
    "tests": {},
    "summary": {},
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE SIGNAL
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 1: INVERSE SIGNAL ═══")
inv_trades = generate_signals(TICKERS, inverse=True)
inv_eq, inv_results = simulate_portfolio(inv_trades)
inv_metrics = calc_metrics(inv_results, inv_eq)

inv_pass = inv_metrics["sharpe"] < base_metrics["sharpe"] * 0.5
print(f"  Inverse trades: {inv_metrics['n_trades']}, Sharpe: {inv_metrics['sharpe']}")
print(f"  Baseline Sharpe: {base_metrics['sharpe']}")
print(f"  {'PASS' if inv_pass else 'FAIL'}: Inverse Sharpe {'<' if inv_pass else '>='} 50% of baseline")

adversarial_results["tests"]["1_inverse_signal"] = {
    "inverse_sharpe": inv_metrics["sharpe"],
    "inverse_trades": inv_metrics["n_trades"],
    "inverse_wr": inv_metrics["wr"],
    "baseline_sharpe": base_metrics["sharpe"],
    "ratio": round(inv_metrics["sharpe"] / base_metrics["sharpe"], 3) if base_metrics["sharpe"] != 0 else 999,
    "pass": inv_pass,
    "detail": inv_metrics,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM ENTRY TIMING (1000 permutations)
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 2: RANDOM ENTRY TIMING ═══")
n_base_trades = len(base_trades)

# Build pool of all valid trading dates in OOT
oot_start = pd.Timestamp(OOT_START)
all_dates_tickers = []
for ticker in TICKERS:
    px = closes.get(ticker)
    if px is None or len(px) < LOOKBACK_RANGE + BB_PERIOD + HOLD_DAYS:
        continue
    valid_dates = px.index[px.index >= oot_start]
    valid_dates = valid_dates[:-HOLD_DAYS]  # need room for hold period
    for d in valid_dates:
        all_dates_tickers.append((d, ticker))

random_sharpes = []
for i in range(N_PERM):
    if not all_dates_tickers or n_base_trades == 0:
        break
    # Random sample of (date, ticker) pairs
    indices = np.random.choice(len(all_dates_tickers), size=min(n_base_trades, len(all_dates_tickers)),
                               replace=False)
    rand_trades = []
    for idx in indices:
        d, ticker = all_dates_tickers[idx]
        px = closes[ticker]
        loc = px.index.get_loc(d)
        exit_loc = min(loc + HOLD_DAYS, len(px) - 1)
        rand_trades.append({
            "ticker": ticker,
            "entry_date": d,
            "exit_date": px.index[exit_loc],
            "entry_price": float(px.iloc[loc]),
            "exit_price": float(px.iloc[exit_loc]),
        })

    _, rand_results = simulate_portfolio(rand_trades)
    rand_eq_series = pd.Series({r["exit_date"]: r["equity"] for r in rand_results}).sort_index() if rand_results else pd.Series(dtype=float)
    rand_m = calc_metrics(rand_results, rand_eq_series)
    random_sharpes.append(rand_m["sharpe"])

    if (i + 1) % 200 == 0:
        print(f"  Permutation {i + 1}/{N_PERM} done")

if random_sharpes:
    pctile = np.mean([1 for s in random_sharpes if s < base_metrics["sharpe"]]) * 100
    perm_pvalue = 1 - pctile / 100
else:
    pctile = 0
    perm_pvalue = 1.0

rand_pass = pctile >= 95
print(f"  Baseline Sharpe percentile: {pctile:.1f}%")
print(f"  Permutation p-value: {perm_pvalue:.4f}")
print(f"  Random Sharpe mean: {np.mean(random_sharpes):.3f}, std: {np.std(random_sharpes):.3f}")
print(f"  {'PASS' if rand_pass else 'FAIL'}: Percentile {'≥' if rand_pass else '<'} 95th")

adversarial_results["tests"]["2_random_timing"] = {
    "percentile": round(float(pctile), 1),
    "perm_pvalue": round(float(perm_pvalue), 4),
    "random_sharpe_mean": round(float(np.mean(random_sharpes)), 3),
    "random_sharpe_std": round(float(np.std(random_sharpes)), 3),
    "baseline_sharpe": base_metrics["sharpe"],
    "n_permutations": N_PERM,
    "pass": rand_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 3: SUB-PERIOD STABILITY ═══")
sub_periods = [
    ("2022-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2025-07-01"),
    ("2025-07-01", "2026-07-31"),
]

sub_results_list = []
all_positive = True

for sp_start, sp_end in sub_periods:
    sp_start_ts = pd.Timestamp(sp_start)
    sp_end_ts = pd.Timestamp(sp_end)

    sp_trades = [t for t in base_trades
                 if sp_start_ts <= t["entry_date"] < sp_end_ts]

    sp_eq, sp_res = simulate_portfolio(sp_trades)
    sp_m = calc_metrics(sp_res, sp_eq)

    label = f"{sp_start} to {sp_end}"
    sub_results_list.append({
        "period": label,
        "sharpe": sp_m["sharpe"],
        "n_trades": sp_m["n_trades"],
        "wr": sp_m["wr"],
        "pf": sp_m["pf"],
    })

    if sp_m["sharpe"] <= 0:
        all_positive = False

    print(f"  {label}: Sharpe={sp_m['sharpe']}, Trades={sp_m['n_trades']}, WR={sp_m['wr']}%")

sub_pass = all_positive
print(f"  {'PASS' if sub_pass else 'FAIL'}: All sub-periods {'have' if sub_pass else 'do NOT all have'} positive Sharpe")

adversarial_results["tests"]["3_sub_period_stability"] = {
    "periods": sub_results_list,
    "all_positive_sharpe": all_positive,
    "pass": sub_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: REMOVE TOP 3 TICKERS
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 4: REMOVE TOP 3 TICKERS ═══")

# Find per-ticker P&L
ticker_pnl = {}
for r in base_results:
    ticker_pnl[r["ticker"]] = ticker_pnl.get(r["ticker"], 0) + r["pnl"]

sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
top3 = [t[0] for t in sorted_tickers[:3]]
print(f"  Top 3 by P&L: {top3}")
print(f"  Their P&L: {[round(t[1], 2) for t in sorted_tickers[:3]]}")

reduced_tickers = [t for t in TICKERS if t not in top3]
red_trades = generate_signals(reduced_tickers)
red_eq, red_results = simulate_portfolio(red_trades)
red_metrics = calc_metrics(red_results, red_eq)

sharpe_drop = (1 - red_metrics["sharpe"] / base_metrics["sharpe"]) * 100 if base_metrics["sharpe"] != 0 else 100
top3_pass = sharpe_drop < 50

print(f"  Reduced Sharpe: {red_metrics['sharpe']} (drop: {sharpe_drop:.1f}%)")
print(f"  Reduced trades: {red_metrics['n_trades']}, WR: {red_metrics['wr']}%")
print(f"  {'PASS' if top3_pass else 'FAIL'}: Sharpe drop {sharpe_drop:.1f}% {'<' if top3_pass else '>='} 50%")

adversarial_results["tests"]["4_remove_top3"] = {
    "top3_tickers": top3,
    "top3_pnl": [round(t[1], 2) for t in sorted_tickers[:3]],
    "reduced_sharpe": red_metrics["sharpe"],
    "baseline_sharpe": base_metrics["sharpe"],
    "sharpe_drop_pct": round(float(sharpe_drop), 1),
    "reduced_detail": red_metrics,
    "pass": top3_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 5: PARAMETER SENSITIVITY ═══")
pctile_values = [5, 10, 15, 20, 25]
hold_values = [5, 10, 15, 20]

grid_results = []
sharpe_above_03 = 0
total_combos = len(pctile_values) * len(hold_values)

for pct in pctile_values:
    for hold in hold_values:
        g_trades = generate_signals(TICKERS, bb_width_pctile=pct, hold_days=hold)
        g_eq, g_res = simulate_portfolio(g_trades)
        g_m = calc_metrics(g_res, g_eq)

        grid_results.append({
            "bb_pctile": pct,
            "hold_days": hold,
            "sharpe": g_m["sharpe"],
            "n_trades": g_m["n_trades"],
            "wr": g_m["wr"],
        })

        if g_m["sharpe"] > 0.3:
            sharpe_above_03 += 1

        marker = " *" if pct == BB_WIDTH_PCTILE and hold == HOLD_DAYS else ""
        print(f"  Pctile={pct:2d}%, Hold={hold:2d}d: Sharpe={g_m['sharpe']:6.3f}, "
              f"Trades={g_m['n_trades']:3d}, WR={g_m['wr']:5.1f}%{marker}")

pct_above = sharpe_above_03 / total_combos * 100
param_pass = pct_above >= 50

print(f"\n  {sharpe_above_03}/{total_combos} ({pct_above:.0f}%) combos have Sharpe > 0.3")
print(f"  {'PASS' if param_pass else 'FAIL'}: {pct_above:.0f}% {'≥' if param_pass else '<'} 50%")

adversarial_results["tests"]["5_parameter_sensitivity"] = {
    "grid": grid_results,
    "combos_above_0_3": sharpe_above_03,
    "total_combos": total_combos,
    "pct_above_0_3": round(float(pct_above), 1),
    "pass": param_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ TEST 6: COST SENSITIVITY ═══")
slippage_levels = [0.0005, 0.001, 0.002, 0.005]  # 5bps, 10bps, 20bps, 50bps
cost_results = []
breakpoint_bps = None

for slip in slippage_levels:
    c_eq, c_res = simulate_portfolio(base_trades, slippage_pct=slip)
    c_m = calc_metrics(c_res, c_eq)
    bps = slip * 10000

    cost_results.append({
        "slippage_bps": int(bps),
        "sharpe": c_m["sharpe"],
        "n_trades": c_m["n_trades"],
        "wr": c_m["wr"],
        "pf": c_m["pf"],
        "total_return": c_m["total_return"],
    })

    if c_m["sharpe"] < 0.3 and breakpoint_bps is None:
        breakpoint_bps = int(bps)

    print(f"  {int(bps):2d}bps slippage: Sharpe={c_m['sharpe']:6.3f}, WR={c_m['wr']:.1f}%, "
          f"PF={c_m['pf']:.2f}, Return={c_m['total_return']:.1f}%")

# Strategy should survive at least 10bps
cost_pass = all(cr["sharpe"] > 0.3 for cr in cost_results if cr["slippage_bps"] <= 10)
if breakpoint_bps:
    print(f"  Sharpe drops below 0.3 at {breakpoint_bps}bps")
else:
    print(f"  Sharpe remains above 0.3 at all tested levels")
print(f"  {'PASS' if cost_pass else 'FAIL'}: Survives at ≤10bps")

adversarial_results["tests"]["6_cost_sensitivity"] = {
    "levels": cost_results,
    "breakpoint_bps": breakpoint_bps,
    "survives_10bps": cost_pass,
    "pass": cost_pass,
}


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
tests = adversarial_results["tests"]
pass_count = sum(1 for t in tests.values() if t["pass"])
total_tests = len(tests)

adversarial_results["summary"] = {
    "passed": pass_count,
    "total": total_tests,
    "all_passed": pass_count == total_tests,
    "verdict": "PASS" if pass_count >= 5 else "FAIL",
    "timestamp": datetime.now().isoformat(),
}

print("\n" + "═" * 60)
print("  ADVERSARIAL VALIDATION SUMMARY")
print("═" * 60)
for name, result in tests.items():
    status = "PASS ✓" if result["pass"] else "FAIL ✗"
    print(f"  {name}: {status}")
print(f"\n  Overall: {pass_count}/{total_tests} passed — "
      f"{'PASS' if pass_count >= 5 else 'FAIL'}")
print("═" * 60)

# Save results
out_path = Path("/home/jupiter/Lvl3Quant/data/vol_clustering_adversarial.json")
with open(out_path, "w") as f:
    json.dump(adversarial_results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
