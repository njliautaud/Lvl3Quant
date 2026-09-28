#!/usr/bin/env python3
"""
Analyst Revision Backtest — Price-Based Proxies for Institutional Sentiment
===========================================================================
Universe: 20 growth stocks (minus SQ which is delisted), OOT Jan 2022 - Jul 2026
Starting capital: $645
6 variants with permutation testing and 5-gate validation.
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
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "CRM", "AMD", "NFLX", "UBER", "SHOP", "XYZ", "COIN", "DDOG",  # XYZ = Block (was SQ)
    "NET", "CRWD", "PANW", "ABNB"
]
START = "2021-01-01"
OOT_START = "2022-01-03"
OOT_END = "2026-07-25"
INITIAL_CAPITAL = 645.0
N_PERMS = 1000
RISK_FREE = 0.04

# ─── Data download ───────────────────────────────────────────────────────────
print("Downloading price data...")
tickers = UNIVERSE + ["SPY"]
raw = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

close = raw["Close"].copy()
volume = raw["Volume"].copy()
close = close.ffill()
volume = volume.ffill().fillna(0)

# Drop tickers with insufficient data
valid_stocks = [s for s in UNIVERSE if s in close.columns and close[s].notna().sum() > 252]
print(f"Valid tickers: {len(valid_stocks)}: {valid_stocks}")

spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)

# OOT indices
oot_mask = close.index >= OOT_START
dates_oot = close.index[oot_mask].tolist()
n_oot = len(dates_oot)
print(f"OOT period: {dates_oot[0].date()} to {dates_oot[-1].date()}, {n_oot} days")
print(f"Bull days: {regime.loc[oot_mask].sum()}, Bear days: {(~regime.loc[oot_mask].astype(bool)).sum()}")

# ─── Pre-compute ALL indicators (vectorized) ─────────────────────────────────
stocks = valid_stocks
close_s = close[stocks]
vol_s = volume[stocks]

high_252 = close_s.rolling(252).max()
low_252 = close_s.rolling(252).min()
sma20 = close_s.rolling(20).mean()
sma50 = close_s.rolling(50).mean()
sma200 = close_s.rolling(200).mean()
ret_60 = close_s.pct_change(60)
spy_ret_60 = spy_close.pct_change(60)
rel_strength = ret_60.subtract(spy_ret_60, axis=0)
price_high_60 = close_s.rolling(60).max()

# OBV
obv = pd.DataFrame(0.0, index=close.index, columns=stocks)
for s in stocks:
    pd_diff = close_s[s].diff()
    signed = vol_s[s].where(pd_diff > 0, -vol_s[s]).where(pd_diff != 0, 0)
    obv[s] = signed.cumsum()
obv_max_60 = obv.rolling(60).max()

# Forward returns for each holding period
fwd_ret = {}
for h in [20, 30]:
    fr = close_s.shift(-h) / close_s - 1
    fwd_ret[h] = fr

print("Indicators computed.")


# ─── Pre-compute signal matrices (boolean DataFrames) ────────────────────────
# Each signal: DataFrame[dates x stocks] = True/False

# A) New 52-week high (within 1%)
sig_new_high = (close_s >= high_252 * 0.99)

# B) Triple SMA: price > SMA20 > SMA50 > SMA200
sig_triple = (close_s > sma20) & (sma20 > sma50) & (sma50 > sma200)

# C) Relative strength > 0 (rank later)
sig_rs_positive = rel_strength > 0

# D) Accumulation: OBV near 60d high AND price near 60d high
sig_accum = (obv >= obv_max_60 * 0.98) & (close_s >= price_high_60 * 0.95)

# F) Contrarian: new 52-week low (within 1%)
sig_low = (close_s <= low_252 * 1.01)


# ─── Fast backtest engine ────────────────────────────────────────────────────
def fast_backtest(signal_dates_stocks, hold_days, top_n=3):
    """
    signal_dates_stocks: list of (date_idx_in_oot, [tickers])
    Returns equity curve and trade list.
    Non-overlapping: wait until current hold expires.
    """
    equity = INITIAL_CAPITAL
    eq_arr = np.full(n_oot, np.nan)
    trades = []
    next_free = 0  # next day idx we can open

    for day_i, tickers_eligible in signal_dates_stocks:
        if day_i < next_free or day_i >= n_oot:
            continue
        if len(tickers_eligible) == 0:
            continue

        sel = tickers_eligible[:top_n]
        date = dates_oot[day_i]
        exit_i = min(day_i + hold_days, n_oot - 1)
        exit_date = dates_oot[exit_i]

        alloc = equity / len(sel)
        trade_equity_start = equity

        for tick in sel:
            entry_p = close.loc[date, tick]
            exit_p = close.loc[exit_date, tick]
            if pd.isna(entry_p) or pd.isna(exit_p) or entry_p <= 0:
                continue
            shares = alloc / entry_p
            pnl = shares * (exit_p - entry_p)
            ret = exit_p / entry_p - 1
            equity += pnl
            r = int(regime.loc[date]) if date in regime.index else -1
            trades.append((str(date.date()), str(exit_date.date()), tick, entry_p, exit_p, pnl, ret, r))

        next_free = exit_i + 1

    # Build equity curve from trades
    # Simpler: reconstruct from trade P&Ls
    eq = INITIAL_CAPITAL
    eq_list = []
    trade_idx = 0
    active_trades = []

    for i in range(n_oot):
        date = dates_oot[i]
        # Mark to market: simple approach - between trade windows equity is flat
        eq_list.append(eq)

        # Check if any trades close on this day
        # We already computed final equity in the loop above

    # Actually just compute from cumulative PnL
    final_eq = INITIAL_CAPITAL + sum(t[5] for t in trades)

    # Build a simple daily equity curve based on trade timing
    eq_curve_simple = np.full(n_oot, INITIAL_CAPITAL, dtype=float)
    cumulative = INITIAL_CAPITAL
    for t in trades:
        entry_date = pd.Timestamp(t[0])
        exit_date = pd.Timestamp(t[1])
        entry_i = dates_oot.index(entry_date) if entry_date in dates_oot else None
        exit_i = dates_oot.index(exit_date) if exit_date in dates_oot else None
        if exit_i is not None:
            cumulative += t[5]  # pnl
            for j in range(exit_i, n_oot):
                eq_curve_simple[j] = max(eq_curve_simple[j], cumulative)

    # More accurate: compute daily interpolated equity
    eq_daily = np.full(n_oot, INITIAL_CAPITAL, dtype=float)
    base = INITIAL_CAPITAL
    sorted_trades = sorted(trades, key=lambda x: x[0])

    # Group trades by entry date
    trade_groups = {}
    for t in sorted_trades:
        ed = t[0]
        if ed not in trade_groups:
            trade_groups[ed] = []
        trade_groups[ed].append(t)

    cash = INITIAL_CAPITAL
    positions = []  # (tick, shares, entry_price)

    for i in range(n_oot):
        date = dates_oot[i]
        ds = str(date.date())

        # Close positions that expire
        new_positions = []
        for tick, shares, ep, exit_ds in positions:
            if ds >= exit_ds:
                exit_p = close.loc[date, tick] if date in close.index else ep
                cash += shares * exit_p
            else:
                new_positions.append((tick, shares, ep, exit_ds))
        positions = new_positions

        # Open new positions
        if ds in trade_groups:
            for t in trade_groups[ds]:
                tick = t[2]
                entry_p = t[3]
                exit_ds = t[1]
                alloc_per = cash / max(len(trade_groups[ds]), 1)
                shares = alloc_per / entry_p if entry_p > 0 else 0
                positions.append((tick, shares, entry_p, exit_ds))
                cash -= shares * entry_p

        # MTM
        mtm = cash
        for tick, shares, ep, exit_ds in positions:
            cp = close.loc[date, tick] if date in close.index else ep
            mtm += shares * cp
        eq_daily[i] = mtm

    return eq_daily, trades


def generate_signal_list(sig_matrix, rank_series=None, top_n=3):
    """Convert a boolean signal matrix to list of (day_idx, [tickers])."""
    result = []
    for i, date in enumerate(dates_oot):
        if date not in sig_matrix.index:
            continue
        row = sig_matrix.loc[date]
        eligible = [s for s in stocks if row.get(s, False)]
        if rank_series is not None and date in rank_series.index:
            # Sort by rank_series descending
            rank_vals = rank_series.loc[date]
            eligible = sorted(eligible, key=lambda s: -rank_vals.get(s, -999))
        result.append((i, eligible[:top_n]))
    return result


# ─── Build signal lists for each variant ──────────────────────────────────────
print("\nBuilding signal lists...")

# A: New high momentum - rank by closeness to high
sig_a = generate_signal_list(sig_new_high.loc[dates_oot[0]:dates_oot[-1]],
                              rank_series=(close_s / high_252).loc[dates_oot[0]:dates_oot[-1]], top_n=3)

# B: Triple SMA - take up to 3
sig_b = generate_signal_list(sig_triple.loc[dates_oot[0]:dates_oot[-1]], top_n=3)

# C: Relative strength - rank by rel_strength value
sig_c = []
for i, date in enumerate(dates_oot):
    if date not in rel_strength.index:
        sig_c.append((i, []))
        continue
    rs = rel_strength.loc[date].dropna()
    rs = rs[rs > 0].sort_values(ascending=False)
    sig_c.append((i, rs.index.tolist()[:3]))

# D: Accumulation
sig_d = generate_signal_list(sig_accum.loc[dates_oot[0]:dates_oot[-1]], top_n=3)

# E: Confluence (3+ of A,B,C,D)
sig_e = []
for i, date in enumerate(dates_oot):
    if date not in sig_new_high.index:
        sig_e.append((i, []))
        continue
    counts = {}
    for s in stocks:
        c = 0
        if sig_new_high.loc[date, s]: c += 1
        if sig_triple.loc[date, s]: c += 1
        if s in [x for x in (rel_strength.loc[date].dropna().sort_values(ascending=False).head(5).index if date in rel_strength.index else [])]: c += 1
        if sig_accum.loc[date, s]: c += 1
        if c >= 3:
            counts[s] = c
    ranked = sorted(counts.keys(), key=lambda s: -counts[s])
    sig_e.append((i, ranked[:3]))

# F: Contrarian (52-week lows)
sig_f = generate_signal_list(sig_low.loc[dates_oot[0]:dates_oot[-1]],
                              rank_series=(low_252 / close_s).loc[dates_oot[0]:dates_oot[-1]], top_n=3)

print("Signal lists built.")


# ─── Compute metrics ─────────────────────────────────────────────────────────
def compute_metrics(eq_daily, trades, label):
    eq = pd.Series(eq_daily, index=dates_oot)
    daily_ret = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    excess = daily_ret - RISK_FREE / 252
    sharpe = float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0

    downside = excess[excess < 0]
    ds_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 1e-9
    sortino = float(np.sqrt(252) * excess.mean() / ds_std)

    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = float(dd.min())

    n_trades = len(trades)
    if n_trades > 0:
        rets = [t[6] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        win_rate = len(wins) / n_trades
        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 1e-9
        profit_factor = gp / gl
    else:
        win_rate = 0.0
        profit_factor = 0.0

    # Regime stratified
    regime_oot_aligned = regime.reindex(eq.index).ffill()
    bull_ret = daily_ret[regime_oot_aligned == 1]
    bear_ret = daily_ret[regime_oot_aligned == 0]
    bull_ex = bull_ret - RISK_FREE / 252
    bear_ex = bear_ret - RISK_FREE / 252
    sharpe_bull = float(np.sqrt(252) * bull_ex.mean() / bull_ex.std()) if len(bull_ex) > 5 and bull_ex.std() > 0 else 0.0
    sharpe_bear = float(np.sqrt(252) * bear_ex.mean() / bear_ex.std()) if len(bear_ex) > 5 and bear_ex.std() > 0 else 0.0
    max_s = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_s if max_s > 0 else 0.0

    return {
        "label": label,
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_trades": n_trades,
        "sharpe_bull": round(sharpe_bull, 4),
        "sharpe_bear": round(sharpe_bear, 4),
        "regime_gap": round(regime_gap, 4),
        "final_equity": round(float(eq.iloc[-1]), 2),
    }


def sharpe_from_eq(eq_arr):
    eq = pd.Series(eq_arr)
    dr = eq.pct_change().dropna()
    ex = dr - RISK_FREE / 252
    return float(np.sqrt(252) * ex.mean() / ex.std()) if ex.std() > 0 else 0.0


# ─── Permutation test (fast) ─────────────────────────────────────────────────
def perm_test(signal_list, hold_days, actual_sharpe, label):
    """Shuffle stock selections, keep timing. Much faster than full re-run."""
    print(f"  Permutation test ({N_PERMS} shuffles) for {label}...")
    beat_count = 0

    for _ in range(N_PERMS):
        shuffled = []
        for day_i, ticks in signal_list:
            if len(ticks) == 0:
                shuffled.append((day_i, []))
            else:
                n = len(ticks)
                rand_ticks = list(np.random.choice(stocks, size=min(n, len(stocks)), replace=False))
                shuffled.append((day_i, rand_ticks))
        eq, _ = fast_backtest(shuffled, hold_days)
        s = sharpe_from_eq(eq)
        if s >= actual_sharpe:
            beat_count += 1

    return round(beat_count / N_PERMS, 4)


# ─── 5-Gate ──────────────────────────────────────────────────────────────────
def validate_5gate(m, perm_p):
    gates = {
        "sharpe_gt_0.5": m["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": m["regime_gap"] < 0.5,
        "mdd_gt_neg50": m["max_drawdown"] > -0.50,
        "trades_gte_20": m["total_trades"] >= 20,
    }
    gates["passed"] = int(sum(v for k, v in gates.items() if isinstance(v, bool)))
    gates["all_passed"] = all(v for k, v in gates.items() if isinstance(v, bool))
    return gates


# ─── Run all variants ────────────────────────────────────────────────────────
VARIANTS = [
    ("A_new_high_momentum_30d", sig_a, 30),
    ("B_triple_sma_bullish_20d", sig_b, 20),
    ("C_relative_strength_20d", sig_c, 20),
    ("D_accumulation_breakout_20d", sig_d, 20),
    ("E_confluence_3plus_20d", sig_e, 20),
    ("F_contrarian_52w_low_30d", sig_f, 30),
]

results = {}
for label, sig_list, hold in VARIANTS:
    print(f"\n{'='*60}")
    print(f"Running: {label}")

    eq, trades = fast_backtest(sig_list, hold)
    metrics = compute_metrics(eq, trades, label)

    print(f"  Return: {metrics['total_return']:.1%} | Sharpe: {metrics['sharpe']:.2f} | Trades: {metrics['total_trades']}")

    perm_p = perm_test(sig_list, hold, metrics["sharpe"], label)
    metrics["perm_p"] = perm_p
    print(f"  Perm p: {perm_p}")

    gates = validate_5gate(metrics, perm_p)
    metrics["five_gate"] = {k: bool(v) if isinstance(v, (bool, np.bool_)) else int(v) for k, v in gates.items()}
    print(f"  5-Gate: {gates['passed']}/5 | All: {gates['all_passed']}")

    # Sample trades
    metrics["sample_trades"] = [
        {"entry": t[0], "exit": t[1], "ticker": t[2],
         "entry_p": round(float(t[3]), 2), "exit_p": round(float(t[4]), 2),
         "pnl": round(float(t[5]), 2), "return": round(float(t[6]), 4), "regime": int(t[7])}
        for t in trades[:10]
    ]

    results[label] = metrics

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
hdr = f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'WR':>6} {'PF':>6} {'#Tr':>5} {'MDD':>8} {'PermP':>7} {'Gate':>5}"
print(hdr)
print("-" * len(hdr))
for label, m in results.items():
    print(f"{m['label']:<35} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>6.1%} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['total_trades']:>5d} {m['max_drawdown']:>7.1%} {m['perm_p']:>7.3f} {m['five_gate']['passed']:>2d}/5")

# ─── Save ─────────────────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/analyst_revision_results.json")
output = {
    "metadata": {
        "strategy": "Analyst Revision Proxies (Price-Based Institutional Sentiment)",
        "universe": UNIVERSE,
        "valid_tickers": valid_stocks,
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "initial_capital": INITIAL_CAPITAL,
        "n_permutations": N_PERMS,
        "risk_free_rate": RISK_FREE,
        "generated_at": datetime.now().isoformat(),
    },
    "variants": results,
}

def numpy_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=numpy_convert)

print(f"\nResults saved to {output_path}")
print("Done.")
