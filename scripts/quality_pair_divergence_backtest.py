#!/usr/bin/env python3
"""
Quality Pair Divergence Backtest
================================
When two highly-correlated quality stocks diverge (one drops while the other
holds up or rises), buy the laggard. Quality stocks tend to converge because
they share similar earnings characteristics.

Universe: 10 highly-correlated quality pairs
OOT: Jan 2022 – Jul 2026
Starting capital: $645, max $200/trade, max 3 concurrent
6 variants with permutation testing and 5-gate validation.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ─── Config ───────────────────────────────────────────────────────────────────
PAIRS = [
    ("AAPL", "MSFT"),    # tech leaders
    ("JPM", "V"),        # financials
    ("JNJ", "PG"),       # consumer defensive
    ("KO", "PEP"),       # beverages
    ("UNH", "LLY"),      # healthcare
    ("HD", "COST"),       # consumer discretionary/retail
    ("AVGO", "MA"),       # tech/fintech
    ("ABBV", "MRK"),     # pharma
    ("AMZN", "GOOGL"),   # tech growth
    ("META", "GOOGL"),   # ad tech
]

ALL_TICKERS = sorted(set(t for p in PAIRS for t in p))
START = "2021-01-01"  # need lookback before OOT
OOT_START = "2022-01-03"
OOT_END = "2026-07-31"
INITIAL_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE = 0.0002  # 0.02% each way
LOOKBACK = 20
N_PERMS = 1000
RISK_FREE = 0.04

# ─── Data download ───────────────────────────────────────────────────────────
flush_print("Downloading price data...")
dl_tickers = ALL_TICKERS + ["SPY", "^VIX"]
raw = yf.download(dl_tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

close = raw["Close"].copy()
close = close.ffill()

# VIX
if "^VIX" in close.columns:
    vix_series = close["^VIX"].copy()
else:
    vix_series = pd.Series(20.0, index=close.index)

spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

# Validate all tickers present
for t in ALL_TICKERS:
    if t not in close.columns or close[t].isna().sum() > close.shape[0] * 0.5:
        flush_print(f"WARNING: {t} has insufficient data")

# OOT indices
oot_start_idx = close.index.searchsorted(pd.Timestamp(OOT_START))
dates_all = close.index
n_total = len(dates_all)
n_oot = n_total - oot_start_idx
flush_print(f"OOT period: {dates_all[oot_start_idx].date()} to {dates_all[-1].date()}, {n_oot} days")
oot_regime = regime.values[oot_start_idx:]
flush_print(f"Bull days: {oot_regime.sum()}, Bear days: {(oot_regime == 0).sum()}")

# ─── Pre-compute pair metrics as numpy arrays ────────────────────────────────
flush_print("Computing pair metrics...")

# Map pair keys to indices
pair_keys_list = [f"{a}/{b}" for a, b in PAIRS]
pair_key_to_idx = {k: i for i, k in enumerate(pair_keys_list)}
n_pairs = len(PAIRS)

# Pre-compute numpy arrays for speed: shape (n_total,) per pair
z_scores_all = np.full((n_pairs, n_total), np.nan)
roll_corrs_all = np.full((n_pairs, n_total), np.nan)
# For each pair, store ticker indices into close columns
pair_a_tickers = []
pair_b_tickers = []

# Close prices as dict of numpy arrays
close_np = {t: close[t].values for t in ALL_TICKERS}
vix_np = vix_series.values
regime_np = regime.values

avg_corrs = {}

for pi, (a, b) in enumerate(PAIRS):
    key = pair_keys_list[pi]
    pair_a_tickers.append(a)
    pair_b_tickers.append(b)

    ret_a = pd.Series(close_np[a]).pct_change()
    ret_b = pd.Series(close_np[b]).pct_change()

    # 20-day rolling correlation
    roll_corr = ret_a.rolling(LOOKBACK).corr(ret_b).values
    roll_corrs_all[pi] = roll_corr

    # 20-day relative performance
    cum_ret_a = pd.Series(close_np[a]).pct_change(LOOKBACK).values
    cum_ret_b = pd.Series(close_np[b]).pct_change(LOOKBACK).values
    rel_perf = cum_ret_a - cum_ret_b

    # Rolling z-score (60-day lookback)
    rp = pd.Series(rel_perf)
    rel_mean = rp.rolling(60).mean().values
    rel_std = rp.rolling(60).std().values
    rel_std[rel_std == 0] = np.nan
    z_score = (rel_perf - rel_mean) / rel_std

    z_scores_all[pi] = z_score

    # Average OOT correlation
    oot_corr = roll_corr[oot_start_idx:]
    avg_corrs[key] = float(np.nanmean(oot_corr))

# Top 5 most correlated pairs
top5_pairs = sorted(avg_corrs, key=lambda x: avg_corrs[x], reverse=True)[:5]
top5_indices = set(pair_key_to_idx[k] for k in top5_pairs)

flush_print(f"\nAverage OOT correlations:")
for k in sorted(avg_corrs, key=lambda x: avg_corrs[x], reverse=True):
    flush_print(f"  {k}: {avg_corrs[k]:.3f}")
flush_print(f"Top 5 most correlated: {top5_pairs}")


# ─── Optimized backtest engine using numpy arrays ────────────────────────────
def run_backtest_fast(pair_indices, z_threshold, hold_days, mean_revert_exit=False,
                      vix_filter=False, pair_trade=False, z_override=None):
    """
    Fast backtest using pre-computed numpy arrays.
    z_override: if provided, (n_pairs, n_oot) array of shuffled z-scores for OOT period
    """
    capital = INITIAL_CAPITAL
    equity_curve = np.empty(n_oot)
    trades = []
    # positions: list of (pair_idx, entry_oot_idx, long_ticker, long_entry_price, long_dollars,
    #                      short_ticker, short_entry_price, short_dollars)
    positions = []

    for i in range(n_oot):
        gi = oot_start_idx + i  # global index

        # Check exits
        new_positions = []
        for pos in positions:
            pi_pos, entry_i, long_t, long_ep, long_d, short_t, short_ep, short_d = pos
            days_held = i - entry_i

            should_exit = False
            if mean_revert_exit:
                if z_override is not None:
                    z_now = z_override[pi_pos, i]
                else:
                    z_now = z_scores_all[pi_pos, gi]
                if not np.isnan(z_now) and abs(z_now) < 0.5:
                    should_exit = True
                if days_held >= 40:
                    should_exit = True
            else:
                if days_held >= hold_days:
                    should_exit = True

            if should_exit:
                exit_price_long = close_np[long_t][gi]
                long_ret = (exit_price_long / long_ep) - 1 - 2 * SLIPPAGE
                pnl = long_d * long_ret

                if pair_trade and short_t is not None:
                    exit_price_short = close_np[short_t][gi]
                    short_ret = (short_ep / exit_price_short) - 1 - 2 * SLIPPAGE
                    pnl += short_d * short_ret

                capital += pnl
                entry_gi = oot_start_idx + entry_i
                trades.append({
                    "pair": pair_keys_list[pi_pos],
                    "entry_date": str(dates_all[entry_gi].date()),
                    "exit_date": str(dates_all[gi].date()),
                    "long": long_t,
                    "pnl": round(pnl, 2),
                    "days_held": days_held,
                    "regime": "bull" if regime_np[entry_gi] == 1 else "bear",
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Check entries
        if len(positions) < MAX_CONCURRENT:
            signals = []
            for pi in pair_indices:
                if z_override is not None:
                    z = z_override[pi, i]
                else:
                    z = z_scores_all[pi, gi]
                corr = roll_corrs_all[pi, gi]

                if np.isnan(z) or np.isnan(corr):
                    continue
                if corr < 0.3:
                    continue
                if vix_filter and not np.isnan(vix_np[gi]) and vix_np[gi] >= 25:
                    continue
                if any(p[0] == pi for p in positions):
                    continue
                if abs(z) > z_threshold:
                    a_t = pair_a_tickers[pi]
                    b_t = pair_b_tickers[pi]
                    if z > 0:
                        long_t, short_t = b_t, a_t
                    else:
                        long_t, short_t = a_t, b_t
                    signals.append((abs(z), pi, long_t, short_t))

            signals.sort(key=lambda x: -x[0])

            for _, pi_sig, long_t, short_t in signals:
                if len(positions) >= MAX_CONCURRENT:
                    break
                entry_price_long = close_np[long_t][gi]
                trade_size = min(MAX_PER_TRADE, capital * 0.4)
                if trade_size < 10:
                    continue

                if pair_trade:
                    entry_price_short = close_np[short_t][gi]
                    positions.append((pi_sig, i, long_t, entry_price_long, trade_size,
                                      short_t, entry_price_short, trade_size))
                else:
                    positions.append((pi_sig, i, long_t, entry_price_long, trade_size,
                                      None, 0.0, 0.0))

        equity_curve[i] = capital

    # Force close remaining positions
    gi_last = n_total - 1
    for pos in positions:
        pi_pos, entry_i, long_t, long_ep, long_d, short_t, short_ep, short_d = pos
        exit_price_long = close_np[long_t][gi_last]
        long_ret = (exit_price_long / long_ep) - 1 - 2 * SLIPPAGE
        pnl = long_d * long_ret
        if pair_trade and short_t is not None:
            exit_price_short = close_np[short_t][gi_last]
            short_ret = (short_ep / exit_price_short) - 1 - 2 * SLIPPAGE
            pnl += short_d * short_ret
        capital += pnl
        entry_gi = oot_start_idx + entry_i
        trades.append({
            "pair": pair_keys_list[pi_pos],
            "entry_date": str(dates_all[entry_gi].date()),
            "exit_date": str(dates_all[gi_last].date()),
            "long": long_t,
            "pnl": round(pnl, 2),
            "days_held": n_oot - 1 - entry_i,
            "regime": "bull" if regime_np[entry_gi] == 1 else "bear",
        })
        equity_curve[-1] = capital

    return capital, equity_curve, trades


def calc_metrics(equity_curve, trades, label):
    """Calculate performance metrics."""
    eq = np.array(equity_curve)
    if len(eq) < 2:
        return None

    daily_ret = np.diff(eq) / eq[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]

    total_return = (eq[-1] / INITIAL_CAPITAL - 1) * 100
    n_trades = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 1
    profit_factor = (sum(t["pnl"] for t in wins) / sum(abs(t["pnl"]) for t in losses)) if losses and sum(abs(t["pnl"]) for t in losses) > 0 else 99.9

    n_years = len(eq) / 252
    ann_ret = total_return / n_years if n_years > 0 else 0
    vol = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 1 else 1
    sharpe = (np.mean(daily_ret) * 252 - RISK_FREE) / vol if vol > 0 else 0

    downside = daily_ret[daily_ret < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 1 else 1
    sortino = (np.mean(daily_ret) * 252 - RISK_FREE) / downside_vol if downside_vol > 0 else 0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = abs(dd.min()) * 100

    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]
    bull_wr = len([t for t in bull_trades if t["pnl"] > 0]) / len(bull_trades) * 100 if bull_trades else 0
    bear_wr = len([t for t in bear_trades if t["pnl"] > 0]) / len(bear_trades) * 100 if bear_trades else 0

    pair_stats = {}
    for t in trades:
        p = t["pair"]
        if p not in pair_stats:
            pair_stats[p] = {"n": 0, "wins": 0, "pnl": 0}
        pair_stats[p]["n"] += 1
        if t["pnl"] > 0:
            pair_stats[p]["wins"] += 1
        pair_stats[p]["pnl"] += t["pnl"]
    for p in pair_stats:
        pair_stats[p]["wr"] = round(pair_stats[p]["wins"] / pair_stats[p]["n"] * 100, 1) if pair_stats[p]["n"] > 0 else 0
        pair_stats[p]["pnl"] = round(pair_stats[p]["pnl"], 2)

    return {
        "variant": label,
        "total_return_pct": round(total_return, 2),
        "final_equity": round(eq[-1], 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(min(profit_factor, 99.9), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "ann_return_pct": round(ann_ret, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "bull_trades": len(bull_trades),
        "bull_wr": round(bull_wr, 1),
        "bear_trades": len(bear_trades),
        "bear_wr": round(bear_wr, 1),
        "pair_breakdown": pair_stats,
    }


# ─── Define variants ─────────────────────────────────────────────────────────
all_pair_indices = list(range(n_pairs))
top5_pair_indices = [pair_key_to_idx[k] for k in top5_pairs]

variants = {
    "A": {"label": "A: All pairs, z>1.5, hold 10d", "pair_indices": all_pair_indices,
           "z_threshold": 1.5, "hold_days": 10, "mean_revert_exit": False,
           "vix_filter": False, "pair_trade": False},
    "B": {"label": "B: All pairs, z>2.0, hold 10d", "pair_indices": all_pair_indices,
           "z_threshold": 2.0, "hold_days": 10, "mean_revert_exit": False,
           "vix_filter": False, "pair_trade": False},
    "C": {"label": "C: Top 5 correlated, z>1.5, hold 10d", "pair_indices": top5_pair_indices,
           "z_threshold": 1.5, "hold_days": 10, "mean_revert_exit": False,
           "vix_filter": False, "pair_trade": False},
    "D": {"label": "D: All pairs, z>1.5, mean-revert exit", "pair_indices": all_pair_indices,
           "z_threshold": 1.5, "hold_days": 40, "mean_revert_exit": True,
           "vix_filter": False, "pair_trade": False},
    "E": {"label": "E: All pairs, z>1.5, hold 10d, VIX<25", "pair_indices": all_pair_indices,
           "z_threshold": 1.5, "hold_days": 10, "mean_revert_exit": False,
           "vix_filter": True, "pair_trade": False},
    "F": {"label": "F: All pairs, z>1.5, pair trade, hold 10d", "pair_indices": all_pair_indices,
           "z_threshold": 1.5, "hold_days": 10, "mean_revert_exit": False,
           "vix_filter": False, "pair_trade": True},
}

# ─── Run all variants ────────────────────────────────────────────────────────
results = {}
for vkey, vconf in variants.items():
    flush_print(f"\n{'='*60}")
    flush_print(f"Running variant {vconf['label']}...")

    capital, eq, trades = run_backtest_fast(
        pair_indices=vconf["pair_indices"],
        z_threshold=vconf["z_threshold"],
        hold_days=vconf["hold_days"],
        mean_revert_exit=vconf["mean_revert_exit"],
        vix_filter=vconf["vix_filter"],
        pair_trade=vconf["pair_trade"],
    )

    metrics = calc_metrics(eq, trades, vconf["label"])
    if metrics is None:
        flush_print(f"  No trades generated for variant {vkey}")
        results[vkey] = {"variant": vconf["label"], "error": "no trades"}
        continue

    flush_print(f"  Final equity: ${metrics['final_equity']:.2f} ({metrics['total_return_pct']:+.1f}%)")
    flush_print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1f}%, PF: {metrics['profit_factor']:.2f}")
    flush_print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, MaxDD: {metrics['max_drawdown_pct']:.1f}%")
    flush_print(f"  Bull WR: {metrics['bull_wr']:.1f}% ({metrics['bull_trades']}), Bear WR: {metrics['bear_wr']:.1f}% ({metrics['bear_trades']})")

    # ─── Permutation test (1000 shuffles) ─────────────────────────────────
    flush_print(f"  Running {N_PERMS} permutations...")

    # Pre-generate all shuffled z-score arrays for speed
    oot_z = z_scores_all[:, oot_start_idx:]  # (n_pairs, n_oot)
    perm_returns = np.empty(N_PERMS)

    for s in range(N_PERMS):
        rng = np.random.RandomState(s + 1)
        z_shuf = oot_z.copy()
        for pi in vconf["pair_indices"]:
            rng.shuffle(z_shuf[pi])

        p_cap, _, _ = run_backtest_fast(
            pair_indices=vconf["pair_indices"],
            z_threshold=vconf["z_threshold"],
            hold_days=vconf["hold_days"],
            mean_revert_exit=vconf["mean_revert_exit"],
            vix_filter=vconf["vix_filter"],
            pair_trade=vconf["pair_trade"],
            z_override=z_shuf,
        )
        perm_returns[s] = (p_cap / INITIAL_CAPITAL - 1) * 100

        if (s + 1) % 200 == 0:
            flush_print(f"    Permutation {s+1}/{N_PERMS}...")

    p_value = np.mean(perm_returns >= metrics["total_return_pct"])
    perm_mean = np.mean(perm_returns)
    edge_vs_random = metrics["total_return_pct"] - perm_mean

    flush_print(f"  Permutation p-value: {p_value:.4f}")
    flush_print(f"  Edge vs random: {edge_vs_random:+.2f}%")

    metrics["permutation_p_value"] = round(float(p_value), 4)
    metrics["perm_mean_return"] = round(float(perm_mean), 2)
    metrics["edge_vs_random"] = round(float(edge_vs_random), 2)

    # ─── 5-Gate Validation ────────────────────────────────────────────────
    gates = {}
    gates["G1_positive_return"] = bool(metrics["total_return_pct"] > 0)
    gates["G2_sharpe_above_0.3"] = bool(metrics["sharpe"] > 0.3)
    gates["G3_permutation_p_below_0.05"] = bool(p_value < 0.05)
    gates["G4_win_rate_above_45"] = bool(metrics["win_rate"] > 45)
    gates["G5_profit_factor_above_1.0"] = bool(metrics["profit_factor"] > 1.0)
    gates_passed = sum(gates.values())

    metrics["gates"] = gates
    metrics["gates_passed"] = f"{gates_passed}/5"
    metrics["PASS"] = gates_passed >= 4

    flush_print(f"  5-Gate: {gates_passed}/5 {'PASS' if gates_passed >= 4 else 'FAIL'}")
    for g, v in gates.items():
        flush_print(f"    {g}: {'PASS' if v else 'FAIL'}")

    results[vkey] = metrics

# ─── Summary ─────────────────────────────────────────────────────────────────
flush_print(f"\n{'='*60}")
flush_print("SUMMARY")
flush_print(f"{'='*60}")
flush_print(f"{'Variant':<45} {'Return':>8} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'p-val':>7} {'Gates':>6}")
flush_print("-" * 90)
for vkey in ["A", "B", "C", "D", "E", "F"]:
    r = results[vkey]
    if "error" in r:
        flush_print(f"{r['variant']:<45} {'NO TRADES':>8}")
        continue
    flush_print(f"{r['variant']:<45} {r['total_return_pct']:>7.1f}% {r['sharpe']:>8.3f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['permutation_p_value']:>7.4f} {r['gates_passed']:>6}")

# ─── Save results ────────────────────────────────────────────────────────────
output = {
    "strategy": "Quality Pair Divergence",
    "description": "Buy laggard when highly-correlated quality stock pairs diverge",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_each_way": SLIPPAGE,
    "n_permutations": N_PERMS,
    "pair_correlations": {k: round(v, 3) for k, v in avg_corrs.items()},
    "top5_most_correlated": top5_pairs,
    "variants": results,
    "timestamp": datetime.now().isoformat(),
}

# Clean non-serializable items
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj

output = clean_for_json(output)

out_path = Path("/home/jupiter/Lvl3Quant/data/quality_pair_divergence_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

flush_print(f"\nResults saved to {out_path}")
flush_print("Done.")
