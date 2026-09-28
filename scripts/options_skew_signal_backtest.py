#!/usr/bin/env python3
"""
Options Skew Signal Backtest — VIX-based proxy for put-call skew dynamics.

Uses VIX, VIX term structure proxies, and VIX regime transitions as signals
for SPY timing. Walk-forward OOT: Jan 2022 – Jul 2026. 6 variants.

5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────
ACCOUNT = 645.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra history for moving averages
PERM_ITERS = 1000
SEED = 42
np.random.seed(SEED)

# Cost assumptions
SHARE_SLIPPAGE = 0.0002        # 0.02%
OPTION_COMM_PER_CONTRACT = 0.65
OPTION_BID_ASK_FRAC = 0.03    # 3% of premium
OPTION_PREMIUM_FRAC = 0.02    # ATM weekly ~ 2% of SPY
SHARE_STOP_LOSS = -0.05
OPTION_STOP_LOSS = -0.50

# ── Data Download ──────────────────────────────────────────────────────
print("Downloading SPY and VIX data...")
spy = yf.download("SPY", start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
vix = yf.download("^VIX", start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)

# Handle multi-level columns from yfinance
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Align dates
common_dates = spy.index.intersection(vix.index)
spy = spy.loc[common_dates].copy()
vix = vix.loc[common_dates].copy()

# Precompute features
spy["ret"] = spy["Close"].pct_change()
spy["sma200"] = spy["Close"].rolling(200).mean()
spy["regime"] = np.where(spy["Close"] > spy["sma200"], "bull", "bear")

vix["close"] = vix["Close"]
vix["pct_5d"] = vix["close"].pct_change(5)
vix["range_5d"] = (vix["High"].rolling(5).max() - vix["Low"].rolling(5).min()) / vix["close"]
vix["sma5"] = vix["close"].rolling(5).mean()
vix["sma20"] = vix["close"].rolling(20).mean()
vix["above25_count"] = (vix["close"] > 25).rolling(6).sum()  # count of days >25 in last 6

# Merge
df = spy[["Close", "ret", "sma200", "regime"]].copy()
df.columns = ["spy_close", "spy_ret", "spy_sma200", "regime"]
df["vix"] = vix["close"]
df["vix_pct_5d"] = vix["pct_5d"]
df["vix_range_5d"] = vix["range_5d"]
df["vix_sma5"] = vix["sma5"]
df["vix_sma20"] = vix["sma20"]
df["vix_above25_count"] = vix["above25_count"]
df = df.dropna()

# OOT filter
oot_mask = df.index >= OOT_START
df_oot = df[oot_mask].copy()

print(f"Data: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"OOT:  {df_oot.index[0].strftime('%Y-%m-%d')} to {df_oot.index[-1].strftime('%Y-%m-%d')} ({len(df_oot)} days)")


# ── Backtesting Engine ────────────────────────────────────────────────
def backtest_shares(signals, hold_days, stop_loss=SHARE_STOP_LOSS):
    """Backtest share-based trades. One trade at a time, full allocation."""
    trades = []
    occupied_until = pd.Timestamp.min

    signal_dates = sorted(signals)
    for entry_date in signal_dates:
        if entry_date <= occupied_until:
            continue
        if entry_date not in df_oot.index:
            continue
        idx = df_oot.index.get_loc(entry_date)
        exit_idx = min(idx + hold_days, len(df_oot) - 1)

        entry_price = float(df_oot.iloc[idx]["spy_close"]) * (1 + SHARE_SLIPPAGE)
        regime = df_oot.iloc[idx]["regime"]

        actual_exit_idx = exit_idx
        for d in range(idx + 1, exit_idx + 1):
            day_ret = (float(df_oot.iloc[d]["spy_close"]) / entry_price) - 1
            if day_ret <= stop_loss:
                actual_exit_idx = d
                break

        exit_price = float(df_oot.iloc[actual_exit_idx]["spy_close"]) * (1 - SHARE_SLIPPAGE)
        ret = (exit_price / entry_price) - 1
        pnl = ACCOUNT * ret

        trades.append({
            "entry": df_oot.index[idx].strftime("%Y-%m-%d"),
            "exit": df_oot.index[actual_exit_idx].strftime("%Y-%m-%d"),
            "hold_days": actual_exit_idx - idx,
            "ret": float(ret),
            "pnl": float(pnl),
            "regime": str(regime),
            "stopped": actual_exit_idx != exit_idx,
        })
        occupied_until = df_oot.index[actual_exit_idx]

    return trades


def backtest_options_long(signals, hold_days, direction="put"):
    """Backtest buying ATM options. direction='put' or 'call'."""
    trades = []
    occupied_until = pd.Timestamp.min

    for entry_date in sorted(signals):
        if entry_date <= occupied_until:
            continue
        if entry_date not in df_oot.index:
            continue
        idx = df_oot.index.get_loc(entry_date)
        exit_idx = min(idx + hold_days, len(df_oot) - 1)

        spy_entry = float(df_oot.iloc[idx]["spy_close"])
        regime = df_oot.iloc[idx]["regime"]

        premium = spy_entry * OPTION_PREMIUM_FRAC
        entry_cost = premium * (1 + OPTION_BID_ASK_FRAC / 2)
        exit_cost_factor = (1 - OPTION_BID_ASK_FRAC / 2)

        cost_per_contract = entry_cost * 100 + OPTION_COMM_PER_CONTRACT
        n_contracts = max(1, int(ACCOUNT / cost_per_contract))
        total_cost = n_contracts * cost_per_contract

        actual_exit_idx = exit_idx
        for d in range(idx + 1, exit_idx + 1):
            spy_d = float(df_oot.iloc[d]["spy_close"])
            spy_move = (spy_d / spy_entry) - 1
            days_elapsed = d - idx
            time_decay = (days_elapsed / max(hold_days, 1)) * 0.6

            if direction == "put":
                directional_delta = -spy_move * 0.5 / (premium / spy_entry)
            else:
                directional_delta = spy_move * 0.5 / (premium / spy_entry)

            option_ret = directional_delta - time_decay
            option_ret = max(option_ret, -1.0)

            if option_ret <= OPTION_STOP_LOSS:
                actual_exit_idx = d
                break

        spy_exit = float(df_oot.iloc[actual_exit_idx]["spy_close"])
        spy_move = (spy_exit / spy_entry) - 1
        days_elapsed = actual_exit_idx - idx
        time_decay = (days_elapsed / max(hold_days, 1)) * 0.6

        if direction == "put":
            directional_delta = -spy_move * 0.5 / (premium / spy_entry)
        else:
            directional_delta = spy_move * 0.5 / (premium / spy_entry)

        option_ret = directional_delta - time_decay
        option_ret = max(option_ret, -1.0)

        exit_value = n_contracts * (premium * (1 + option_ret)) * 100 * exit_cost_factor
        exit_comm = n_contracts * OPTION_COMM_PER_CONTRACT
        net_pnl = exit_value - exit_comm - total_cost
        net_ret = net_pnl / total_cost

        trades.append({
            "entry": df_oot.index[idx].strftime("%Y-%m-%d"),
            "exit": df_oot.index[actual_exit_idx].strftime("%Y-%m-%d"),
            "hold_days": actual_exit_idx - idx,
            "ret": float(net_ret),
            "pnl": float(net_pnl),
            "regime": str(regime),
            "stopped": actual_exit_idx != exit_idx,
            "n_contracts": n_contracts,
        })
        occupied_until = df_oot.index[actual_exit_idx]

    return trades


def backtest_conditional_exit(signals, max_hold, exit_fn=None, stop_loss=SHARE_STOP_LOSS):
    """Backtest with conditional exit (not fixed hold)."""
    trades = []
    occupied_until = pd.Timestamp.min

    for entry_date in sorted(signals):
        if entry_date <= occupied_until:
            continue
        if entry_date not in df_oot.index:
            continue
        idx = df_oot.index.get_loc(entry_date)
        entry_price = float(df_oot.iloc[idx]["spy_close"]) * (1 + SHARE_SLIPPAGE)
        regime = df_oot.iloc[idx]["regime"]

        actual_exit_idx = min(idx + max_hold, len(df_oot) - 1)
        stopped = False

        for d in range(idx + 1, min(idx + max_hold + 1, len(df_oot))):
            day_ret = (float(df_oot.iloc[d]["spy_close"]) / entry_price) - 1
            if day_ret <= stop_loss:
                actual_exit_idx = d
                stopped = True
                break
            if exit_fn and exit_fn(df_oot.iloc[d]):
                actual_exit_idx = d
                break

        exit_price = float(df_oot.iloc[actual_exit_idx]["spy_close"]) * (1 - SHARE_SLIPPAGE)
        ret = (exit_price / entry_price) - 1
        pnl = ACCOUNT * ret

        trades.append({
            "entry": df_oot.index[idx].strftime("%Y-%m-%d"),
            "exit": df_oot.index[actual_exit_idx].strftime("%Y-%m-%d"),
            "hold_days": actual_exit_idx - idx,
            "ret": float(ret),
            "pnl": float(pnl),
            "regime": str(regime),
            "stopped": stopped,
        })
        occupied_until = df_oot.index[actual_exit_idx]

    return trades


# ── Signal Generators ─────────────────────────────────────────────────
def signal_A():
    """VIX Spike Reversal: VIX up >25% in 5d -> buy SPY 3 days later, hold 10d."""
    spike_dates = df_oot.index[df_oot["vix_pct_5d"] > 0.25]
    delayed = []
    for d in spike_dates:
        loc = df_oot.index.get_loc(d)
        if loc + 3 < len(df_oot):
            delayed.append(df_oot.index[loc + 3])
    return delayed

def signal_B():
    """VIX Compression Breakout: VIX 5d range < 10% of VIX -> straddle, hold 15d."""
    compressed = df_oot.index[df_oot["vix_range_5d"] < 0.10]
    return list(compressed)

def signal_C():
    """Term Structure Inversion: 5d VIX SMA > 20d VIX SMA by >15% -> buy SPY."""
    ratio = df_oot["vix_sma5"] / df_oot["vix_sma20"]
    inverted = df_oot.index[ratio > 1.15]
    return list(inverted)

def signal_D():
    """Fear Exhaustion: VIX >25 for 5+ days, then closes below 5d SMA -> buy SPY, hold 20d."""
    entries = []
    for i in range(1, len(df_oot)):
        row = df_oot.iloc[i]
        prev = df_oot.iloc[i - 1]
        if (float(row["vix_above25_count"]) >= 5 and
            float(prev["vix"]) > float(prev["vix_sma5"]) and
            float(row["vix"]) < float(row["vix_sma5"])):
            entries.append(df_oot.index[i])
    return entries

def signal_E():
    """Calm-to-Storm: VIX goes from <15 to >20 within 5 days -> buy SPY puts, hold 5d."""
    entries = []
    for i in range(5, len(df_oot)):
        vix_now = float(df_oot.iloc[i]["vix"])
        vix_5ago = float(df_oot.iloc[i - 5]["vix"])
        if vix_5ago < 15 and vix_now > 20:
            entries.append(df_oot.index[i])
    return entries

def signal_F():
    """Multi-Signal: Combine A+C+D. Trade when >=2 of 3 agree on same day."""
    a_dates = set(pd.DatetimeIndex(signal_A()))
    c_dates = set(pd.DatetimeIndex(signal_C()))
    d_dates = set(pd.DatetimeIndex(signal_D()))

    combined = []
    for d in df_oot.index:
        votes = sum([d in a_dates, d in c_dates, d in d_dates])
        if votes >= 2:
            combined.append(d)
    return combined


# ── Metrics ───────────────────────────────────────────────────────────
def compute_metrics(trades):
    if not trades or len(trades) < 2:
        return None

    rets = np.array([t["ret"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    n = len(rets)
    total_ret = np.sum(pnls) / ACCOUNT
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    avg_hold = np.mean([t["hold_days"] for t in trades])
    trades_per_year = 252 / max(avg_hold, 1)
    ann_factor = np.sqrt(trades_per_year)

    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0
    neg_rets = rets[rets < 0]
    downside = np.std(neg_rets, ddof=1) if len(neg_rets) > 1 else std_ret
    sortino = (mean_ret / downside * ann_factor) if downside > 0 else 0

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    wr = len(wins) / n
    pf = (np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else float("inf")

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum + ACCOUNT)
    dd = (cum + ACCOUNT - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    bull_rets = [t["ret"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["ret"] for t in trades if t["regime"] == "bear"]

    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_rets) > 1:
        s = np.std(bull_rets, ddof=1)
        bull_sharpe = float(np.mean(bull_rets) / s * ann_factor) if s > 0 else 0
    if len(bear_rets) > 1:
        s = np.std(bear_rets, ddof=1)
        bear_sharpe = float(np.mean(bear_rets) / s * ann_factor) if s > 0 else 0

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 0.01)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    return {
        "n_trades": int(n),
        "total_ret_pct": round(float(total_ret * 100), 2),
        "total_pnl": round(float(np.sum(pnls)), 2),
        "mean_ret_pct": round(float(mean_ret * 100), 3),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "win_rate": round(float(wr), 3),
        "profit_factor": round(float(min(pf, 99.0)), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
        "regime_gap": round(float(regime_gap), 3),
    }


def permutation_test(trades, n_iter=PERM_ITERS):
    """Shuffle entry dates, compute mean return each time."""
    if not trades or len(trades) < 5:
        return 1.0

    actual_mean = np.mean([t["ret"] for t in trades])
    n = len(trades)
    hold_days = [t["hold_days"] for t in trades]
    max_hd = max(hold_days)

    all_indices = list(range(len(df_oot) - max_hd - 1))
    if len(all_indices) < n:
        return 1.0

    count_better = 0
    for _ in range(n_iter):
        random_entries = np.random.choice(all_indices, size=n, replace=True)
        random_rets = []
        for j, ei in enumerate(random_entries):
            hd = hold_days[j % len(hold_days)]
            exit_i = min(ei + hd, len(df_oot) - 1)
            entry_p = float(df_oot.iloc[ei]["spy_close"])
            exit_p = float(df_oot.iloc[exit_i]["spy_close"])
            r = (exit_p / entry_p) - 1 - 2 * SHARE_SLIPPAGE
            random_rets.append(r)
        if np.mean(random_rets) >= actual_mean:
            count_better += 1

    return count_better / n_iter


def random_timing_benchmark(hold_days, n_samples=500):
    """Average return from buying SPY on a random day and holding."""
    valid_indices = list(range(len(df_oot) - hold_days - 1))
    if not valid_indices:
        return 0.0

    sample_size = min(n_samples, len(valid_indices))
    chosen = np.random.choice(valid_indices, size=sample_size, replace=False)
    rets = []
    for idx in chosen:
        entry_p = float(df_oot.iloc[idx]["spy_close"])
        exit_p = float(df_oot.iloc[idx + hold_days]["spy_close"])
        rets.append((exit_p / entry_p) - 1)

    return float(np.mean(rets))


# ── Run All Variants ──────────────────────────────────────────────────
results = {}

# Variant A: VIX Spike Reversal
print("\n=== Variant A: VIX Spike Reversal ===")
sig_a = signal_A()
print(f"  Signal fires: {len(sig_a)} raw dates")
trades_a = backtest_shares(pd.DatetimeIndex(sig_a), hold_days=10)
metrics_a = compute_metrics(trades_a)
if metrics_a:
    perm_a = permutation_test(trades_a)
    bench_a = random_timing_benchmark(10)
    metrics_a["perm_p"] = round(perm_a, 4)
    metrics_a["random_timing_mean_ret"] = round(bench_a * 100, 3)
    edge = ((metrics_a["mean_ret_pct"] / 100) / max(abs(bench_a), 0.0001) - 1) if bench_a != 0 else 999
    metrics_a["edge_vs_random"] = round(float(edge), 3)
    print(f"  Trades: {metrics_a['n_trades']}, Sharpe: {metrics_a['sharpe']}, "
          f"WR: {metrics_a['win_rate']}, MaxDD: {metrics_a['max_dd_pct']}%, "
          f"Perm-p: {perm_a:.4f}")
else:
    print("  Insufficient trades")
results["A_vix_spike_reversal"] = {"trades": trades_a, "metrics": metrics_a, "type": "shares"}

# Variant B: VIX Compression Breakout (straddle)
print("\n=== Variant B: VIX Compression Breakout ===")
sig_b = signal_B()
print(f"  Signal fires: {len(sig_b)} raw dates")
trades_b_call = backtest_options_long(pd.DatetimeIndex(sig_b), hold_days=15, direction="call")
trades_b_put = backtest_options_long(pd.DatetimeIndex(sig_b), hold_days=15, direction="put")
trades_b = []
for tc, tp in zip(trades_b_call, trades_b_put):
    trades_b.append({
        "entry": tc["entry"], "exit": tc["exit"],
        "hold_days": tc["hold_days"],
        "ret": float((tc["ret"] + tp["ret"]) / 2),
        "pnl": float((tc["pnl"] + tp["pnl"]) / 2),
        "regime": tc["regime"], "stopped": False,
    })
metrics_b = compute_metrics(trades_b)
if metrics_b:
    perm_b = permutation_test(trades_b)
    bench_b = random_timing_benchmark(15)
    metrics_b["perm_p"] = round(perm_b, 4)
    metrics_b["random_timing_mean_ret"] = round(bench_b * 100, 3)
    metrics_b["edge_vs_random"] = round(float(abs(metrics_b["mean_ret_pct"] / 100) / max(abs(bench_b), 0.0001) - 1), 3)
    print(f"  Trades: {metrics_b['n_trades']}, Sharpe: {metrics_b['sharpe']}, "
          f"WR: {metrics_b['win_rate']}, MaxDD: {metrics_b['max_dd_pct']}%, "
          f"Perm-p: {perm_b:.4f}")
else:
    print("  Insufficient trades")
results["B_vix_compression_breakout"] = {"trades": trades_b, "metrics": metrics_b, "type": "options_straddle"}

# Variant C: Term Structure Inversion
print("\n=== Variant C: Term Structure Inversion ===")
sig_c = signal_C()
print(f"  Signal fires: {len(sig_c)} raw dates")
def exit_c(row):
    return float(row["vix_sma5"]) < float(row["vix_sma20"])
trades_c = backtest_conditional_exit(pd.DatetimeIndex(sig_c), max_hold=30, exit_fn=exit_c)
metrics_c = compute_metrics(trades_c)
if metrics_c:
    perm_c = permutation_test(trades_c)
    avg_hold_c = max(int(metrics_c["avg_hold_days"]), 1)
    bench_c = random_timing_benchmark(avg_hold_c)
    metrics_c["perm_p"] = round(perm_c, 4)
    metrics_c["random_timing_mean_ret"] = round(bench_c * 100, 3)
    edge = ((metrics_c["mean_ret_pct"] / 100) / max(abs(bench_c), 0.0001) - 1) if bench_c != 0 else 999
    metrics_c["edge_vs_random"] = round(float(edge), 3)
    print(f"  Trades: {metrics_c['n_trades']}, Sharpe: {metrics_c['sharpe']}, "
          f"WR: {metrics_c['win_rate']}, MaxDD: {metrics_c['max_dd_pct']}%, "
          f"Perm-p: {perm_c:.4f}")
else:
    print("  Insufficient trades")
results["C_term_structure_inversion"] = {"trades": trades_c, "metrics": metrics_c, "type": "shares"}

# Variant D: Fear Exhaustion
print("\n=== Variant D: Fear Exhaustion ===")
sig_d = signal_D()
print(f"  Signal fires: {len(sig_d)} raw dates")
trades_d = backtest_shares(pd.DatetimeIndex(sig_d), hold_days=20)
metrics_d = compute_metrics(trades_d)
if metrics_d:
    perm_d = permutation_test(trades_d)
    bench_d = random_timing_benchmark(20)
    metrics_d["perm_p"] = round(perm_d, 4)
    metrics_d["random_timing_mean_ret"] = round(bench_d * 100, 3)
    edge = ((metrics_d["mean_ret_pct"] / 100) / max(abs(bench_d), 0.0001) - 1) if bench_d != 0 else 999
    metrics_d["edge_vs_random"] = round(float(edge), 3)
    print(f"  Trades: {metrics_d['n_trades']}, Sharpe: {metrics_d['sharpe']}, "
          f"WR: {metrics_d['win_rate']}, MaxDD: {metrics_d['max_dd_pct']}%, "
          f"Perm-p: {perm_d:.4f}")
else:
    print("  Insufficient trades")
results["D_fear_exhaustion"] = {"trades": trades_d, "metrics": metrics_d, "type": "shares"}

# Variant E: Calm-to-Storm (buy puts)
print("\n=== Variant E: Calm-to-Storm ===")
sig_e = signal_E()
print(f"  Signal fires: {len(sig_e)} raw dates")
trades_e = backtest_options_long(pd.DatetimeIndex(sig_e), hold_days=5, direction="put")
metrics_e = compute_metrics(trades_e)
if metrics_e:
    perm_e = permutation_test(trades_e)
    bench_e = random_timing_benchmark(5)
    metrics_e["perm_p"] = round(perm_e, 4)
    metrics_e["random_timing_mean_ret"] = round(bench_e * 100, 3)
    metrics_e["edge_vs_random"] = round(float(abs(metrics_e["mean_ret_pct"] / 100) / max(abs(bench_e), 0.0001) - 1), 3)
    print(f"  Trades: {metrics_e['n_trades']}, Sharpe: {metrics_e['sharpe']}, "
          f"WR: {metrics_e['win_rate']}, MaxDD: {metrics_e['max_dd_pct']}%, "
          f"Perm-p: {perm_e:.4f}")
else:
    print("  Insufficient trades")
results["E_calm_to_storm"] = {"trades": trades_e, "metrics": metrics_e, "type": "options_put"}

# Variant F: Multi-Signal Combo
print("\n=== Variant F: Multi-Signal Combo (A+C+D, 2/3 agree) ===")
sig_f = signal_F()
print(f"  Signal fires: {len(sig_f)} raw dates")
trades_f = backtest_shares(pd.DatetimeIndex(sig_f), hold_days=15)
metrics_f = compute_metrics(trades_f)
if metrics_f:
    perm_f = permutation_test(trades_f)
    bench_f = random_timing_benchmark(15)
    metrics_f["perm_p"] = round(perm_f, 4)
    metrics_f["random_timing_mean_ret"] = round(bench_f * 100, 3)
    edge = ((metrics_f["mean_ret_pct"] / 100) / max(abs(bench_f), 0.0001) - 1) if bench_f != 0 else 999
    metrics_f["edge_vs_random"] = round(float(edge), 3)
    print(f"  Trades: {metrics_f['n_trades']}, Sharpe: {metrics_f['sharpe']}, "
          f"WR: {metrics_f['win_rate']}, MaxDD: {metrics_f['max_dd_pct']}%, "
          f"Perm-p: {perm_f:.4f}")
else:
    print("  Insufficient trades")
results["F_multi_signal_combo"] = {"trades": trades_f, "metrics": metrics_f, "type": "shares"}


# ── 5-Gate Validation ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5-GATE VALIDATION SUMMARY")
print("=" * 70)

final_output = {
    "strategy": "Options Skew Signal (VIX Proxy)",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "account_size": ACCOUNT,
    "variants": {},
    "summary": {},
}

any_passed = False
for name, data in results.items():
    m = data["metrics"]
    if m is None:
        print(f"\n{name}: NO TRADES / INSUFFICIENT DATA")
        final_output["variants"][name] = {"status": "NO DATA", "type": data["type"]}
        continue

    g1 = m["sharpe"] > 0.5
    g2 = m.get("perm_p", 1.0) < 0.05
    g3 = m["regime_gap"] < 0.5
    g4 = m["max_dd_pct"] > -50
    g5 = m["n_trades"] >= 20

    passed = sum([g1, g2, g3, g4, g5])
    all_pass = passed == 5

    edge = m.get("edge_vs_random", 0)
    beats_random = edge > 0.5

    status = "PASS ALL GATES" if all_pass else f"FAIL ({5 - passed}/5 gates failed)"
    if all_pass and not beats_random:
        status += " [BUT FAILS RANDOM TIMING CHECK - likely beta]"
    elif all_pass and beats_random:
        status += " [PASSES RANDOM TIMING CHECK]"
        any_passed = True

    gate_marks = lambda x: "PASS" if x else "FAIL"
    print(f"\n{name} ({data['type']}):")
    print(f"  Sharpe={m['sharpe']:.3f} [{gate_marks(g1)}] | "
          f"Perm-p={m.get('perm_p', 'N/A')} [{gate_marks(g2)}] | "
          f"RegimeGap={m['regime_gap']:.3f} [{gate_marks(g3)}] | "
          f"MaxDD={m['max_dd_pct']:.1f}% [{gate_marks(g4)}] | "
          f"Trades={m['n_trades']} [{gate_marks(g5)}]")
    print(f"  WR={m['win_rate']:.1%}  Sortino={m['sortino']:.3f}  PF={m['profit_factor']:.2f}  "
          f"TotalPnL=${m['total_pnl']:.0f}  AvgHold={m['avg_hold_days']:.0f}d")
    print(f"  Bull Sharpe={m['bull_sharpe']:.3f} ({m['bull_trades']}t) | "
          f"Bear Sharpe={m['bear_sharpe']:.3f} ({m['bear_trades']}t)")
    print(f"  Sig Avg Ret: {m['mean_ret_pct']:.3f}% | "
          f"Random Timing: {m.get('random_timing_mean_ret', 'N/A')}% | "
          f"Edge vs Random: {edge:.1%}")
    print(f"  --> {status}")

    variant_out = {k: v for k, v in m.items()}
    variant_out["type"] = data["type"]
    variant_out["gate_results"] = {
        "sharpe_gt_0.5": g1, "perm_p_lt_0.05": g2,
        "regime_gap_lt_0.5": g3, "max_dd_gt_neg50": g4,
        "trades_gte_20": g5,
    }
    variant_out["passes_all_gates"] = all_pass
    variant_out["beats_random_timing"] = beats_random
    variant_out["status"] = status
    variant_out["sample_trades"] = data["trades"][:5] if data["trades"] else []
    final_output["variants"][name] = variant_out

print("\n" + "=" * 70)
if any_passed:
    print("VERDICT: At least one variant passed all 5 gates AND beats random timing.")
else:
    print("VERDICT: NO variant passes all 5 gates + random timing check.")
    print("This strategy is likely just market beta with extra steps.")
print("=" * 70)

final_output["summary"]["any_variant_passed_all"] = any_passed
final_output["summary"]["total_variants"] = len(results)
final_output["summary"]["verdict"] = (
    "At least one variant shows genuine edge" if any_passed
    else "No variant shows genuine edge beyond market beta"
)
final_output["summary"]["timestamp"] = datetime.now().isoformat()

output_path = Path("/home/jupiter/Lvl3Quant/data/options_skew_signal_results.json")
with open(output_path, "w") as f:
    json.dump(final_output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
