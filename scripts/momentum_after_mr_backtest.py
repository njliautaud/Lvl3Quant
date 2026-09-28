#!/usr/bin/env python3
"""
Momentum After Mean Reversion Backtest
=======================================
Two-phase strategy:
  Phase 1: Wait for a quality stock to complete a mean reversion recovery
           (close above 20-day SMA after being below it).
  Phase 2: Ride the momentum continuation (the "second leg").

Universe: 20 quality stocks
OOT: Jan 2022 – Jul 2026
Starting capital: $645, max $200/trade, max 3 concurrent
Slippage: 0.02% each way
6 variants with 1000-permutation testing and 5-gate validation.
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
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

# Sector ETF mapping for variant E
SECTOR_ETFS = {
    "AAPL": "XLK", "MSFT": "XLK", "AVGO": "XLK", "GOOGL": "XLK", "META": "XLK",
    "AMZN": "XLY",
    "JPM": "XLF", "V": "XLF", "MA": "XLF",
    "JNJ": "XLV", "UNH": "XLV", "LLY": "XLV", "ABBV": "XLV", "MRK": "XLV",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "HD": "XLY",
}
SECTOR_ETF_LIST = sorted(set(SECTOR_ETFS.values()))

START = "2021-01-01"  # need lookback before OOT
OOT_START = "2022-01-03"
OOT_END = "2026-07-31"
INITIAL_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE = 0.0002  # 0.02% each way
SMA_PERIOD = 20
RSI_PERIOD = 14
N_PERMS = 1000
RISK_FREE = 0.04

# ─── Data download ───────────────────────────────────────────────────────────
flush_print("Downloading price data...")
all_dl = TICKERS + SECTOR_ETF_LIST + ["SPY"]
raw = yf.download(all_dl, start=START, end=OOT_END, auto_adjust=True, progress=False)

close = raw["Close"].copy().ffill()
volume = raw["Volume"].copy().ffill().fillna(0)

spy_close = close["SPY"].copy()
spy_sma200 = spy_close.rolling(200).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

# Validate tickers
for t in TICKERS:
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

# ─── Pre-compute indicators as numpy arrays ─────────────────────────────────
flush_print("Computing indicators...")

close_np = {t: close[t].values for t in TICKERS}
vol_np = {t: volume[t].values for t in TICKERS}
regime_np = regime.values

# SMA20
sma20 = {}
for t in TICKERS:
    sma20[t] = pd.Series(close_np[t]).rolling(SMA_PERIOD).mean().values

# RSI
rsi = {}
for t in TICKERS:
    delta = pd.Series(close_np[t]).diff()
    gain = delta.where(delta > 0, 0.0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi[t] = (100 - 100 / (1 + rs)).values

# 20-day rolling high
high20 = {}
for t in TICKERS:
    high20[t] = pd.Series(close_np[t]).rolling(SMA_PERIOD).max().values

# 5-day rolling high
high5 = {}
for t in TICKERS:
    high5[t] = pd.Series(close_np[t]).rolling(5).max().values

# 20-day average volume
avg_vol20 = {}
for t in TICKERS:
    avg_vol20[t] = pd.Series(vol_np[t]).rolling(SMA_PERIOD).mean().values

# Sector ETF SMA20
sector_sma20 = {}
for etf in SECTOR_ETF_LIST:
    if etf in close.columns:
        sector_sma20[etf] = pd.Series(close[etf].values).rolling(SMA_PERIOD).mean().values
    else:
        sector_sma20[etf] = np.full(n_total, np.nan)

sector_close_np = {}
for etf in SECTOR_ETF_LIST:
    if etf in close.columns:
        sector_close_np[etf] = close[etf].values
    else:
        sector_close_np[etf] = np.full(n_total, np.nan)

# ─── Signal detection ────────────────────────────────────────────────────────
# For each ticker and each OOT day, pre-compute:
#   - recently_dipped: was >5% below 20-day high within last 20 days
#   - sma_cross_today: close crossed above SMA20 today (was below yesterday)
#   - rsi_above_50: RSI > 50
#   - new_5d_high_today: close == 5-day high AND close > yesterday's 5-day high
#   - volume_surge: today's volume > 1.5x 20-day average volume
#   - sector_above_sma: sector ETF close > its 20-day SMA

flush_print("Pre-computing signals...")

# recently_dipped[t][gi] = True if within last 20 days, close was >5% below 20d high
recently_dipped = {}
for t in TICKERS:
    rd = np.zeros(n_total, dtype=bool)
    for gi in range(SMA_PERIOD, n_total):
        lookback_start = max(0, gi - SMA_PERIOD)
        for j in range(lookback_start, gi + 1):
            h = high20[t][j]
            if not np.isnan(h) and h > 0 and close_np[t][j] < h * 0.95:
                rd[gi] = True
                break
    recently_dipped[t] = rd

# sma_cross_today[t][gi] = close[gi] > sma20[gi] AND close[gi-1] <= sma20[gi-1]
sma_cross = {}
for t in TICKERS:
    sc = np.zeros(n_total, dtype=bool)
    for gi in range(1, n_total):
        s_today = sma20[t][gi]
        s_yest = sma20[t][gi - 1]
        if (not np.isnan(s_today) and not np.isnan(s_yest) and
                close_np[t][gi] > s_today and close_np[t][gi - 1] <= s_yest):
            sc[gi] = True
    sma_cross[t] = sc

# Track days since last SMA cross (for variant C - new 5d high AFTER cross)
days_since_cross = {}
for t in TICKERS:
    dsc = np.full(n_total, 999, dtype=int)
    last_cross = -999
    for gi in range(n_total):
        if sma_cross[t][gi]:
            last_cross = gi
        dsc[gi] = gi - last_cross
    days_since_cross[t] = dsc

flush_print("Signal pre-computation done.")

# ─── Backtest engine ─────────────────────────────────────────────────────────
def run_backtest(variant, shuffled_entries=None):
    """
    Run backtest for a given variant.
    shuffled_entries: if not None, dict of {ticker: shuffled boolean array for OOT period}
                      used for permutation testing.
    """
    capital = INITIAL_CAPITAL
    equity_curve = np.empty(n_oot)
    trades = []
    # positions: list of (ticker, entry_oot_idx, entry_price, dollars, entry_rsi)
    positions = []

    for i in range(n_oot):
        gi = oot_start_idx + i  # global index

        # ─── Check exits ─────────────────────────────────────────────
        new_positions = []
        for pos in positions:
            ticker, entry_i, entry_price, dollars, entry_rsi = pos
            days_held = i - entry_i

            should_exit = False

            if variant == "A":
                if days_held >= 10:
                    should_exit = True
            elif variant == "B":
                if days_held >= 20:
                    should_exit = True
            elif variant == "C":
                if days_held >= 10:
                    should_exit = True
            elif variant == "D":
                # Exit when RSI > 70 or after 15 days
                r = rsi[ticker][gi]
                if not np.isnan(r) and r > 70:
                    should_exit = True
                if days_held >= 15:
                    should_exit = True
            elif variant == "E":
                if days_held >= 10:
                    should_exit = True
            elif variant == "F":
                if days_held >= 10:
                    should_exit = True

            if should_exit:
                exit_price = close_np[ticker][gi]
                ret = (exit_price / entry_price) - 1 - 2 * SLIPPAGE
                pnl = dollars * ret
                capital += pnl
                entry_gi = oot_start_idx + entry_i
                trades.append({
                    "ticker": ticker,
                    "entry_date": str(dates_all[entry_gi].date()),
                    "exit_date": str(dates_all[gi].date()),
                    "pnl": round(pnl, 2),
                    "days_held": days_held,
                    "regime": "bull" if regime_np[entry_gi] == 1 else "bear",
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # ─── Check entries ────────────────────────────────────────────
        if len(positions) < MAX_CONCURRENT:
            signals = []
            for t in TICKERS:
                # Skip if already holding this ticker
                if any(p[0] == t for p in positions):
                    continue

                if shuffled_entries is not None:
                    # Use shuffled entry signal
                    if not shuffled_entries[t][i]:
                        continue
                else:
                    # Base conditions for all variants:
                    # 1. Recently dipped (>5% below 20d high within last 20 days)
                    if not recently_dipped[t][gi]:
                        continue
                    # 2. RSI above 50
                    r = rsi[t][gi]
                    if np.isnan(r) or r <= 50:
                        continue

                    if variant in ("A", "B", "D", "E", "F"):
                        # Enter on SMA cross day
                        if not sma_cross[t][gi]:
                            continue
                    elif variant == "C":
                        # Enter on new 5-day high AFTER a recent SMA cross
                        dsc = days_since_cross[t][gi]
                        if dsc > 10 or dsc < 1:  # cross must be recent (within 10 days) but not today
                            continue
                        # Must be making a new 5-day high
                        h5_today = high5[t][gi]
                        h5_yest = high5[t][gi - 1] if gi > 0 else np.nan
                        if (np.isnan(h5_today) or np.isnan(h5_yest) or
                                close_np[t][gi] < h5_today or h5_today <= h5_yest):
                            continue

                    # Variant E: sector ETF must be above its 20-day SMA
                    if variant == "E":
                        etf = SECTOR_ETFS.get(t)
                        if etf and etf in sector_close_np and etf in sector_sma20:
                            etf_c = sector_close_np[etf][gi]
                            etf_s = sector_sma20[etf][gi]
                            if np.isnan(etf_c) or np.isnan(etf_s) or etf_c <= etf_s:
                                continue

                    # Variant F: volume > 1.5x average
                    if variant == "F":
                        v_today = vol_np[t][gi]
                        v_avg = avg_vol20[t][gi]
                        if np.isnan(v_avg) or v_avg <= 0 or v_today < 1.5 * v_avg:
                            continue

                # Signal strength: RSI distance from 50 (higher = stronger momentum)
                r_val = rsi[t][gi] if not np.isnan(rsi[t][gi]) else 50
                signals.append((r_val, t))

            # Sort by RSI (strongest momentum first)
            signals.sort(key=lambda x: -x[0])

            for _, t in signals:
                if len(positions) >= MAX_CONCURRENT:
                    break
                entry_price = close_np[t][gi]
                if np.isnan(entry_price) or entry_price <= 0:
                    continue
                trade_size = min(MAX_PER_TRADE, capital * 0.4)
                if trade_size < 10:
                    continue
                positions.append((t, i, entry_price, trade_size, rsi[t][gi]))

        equity_curve[i] = capital

    # Force close remaining positions at end
    gi_last = n_total - 1
    for pos in positions:
        ticker, entry_i, entry_price, dollars, entry_rsi = pos
        exit_price = close_np[ticker][gi_last]
        ret = (exit_price / entry_price) - 1 - 2 * SLIPPAGE
        pnl = dollars * ret
        capital += pnl
        entry_gi = oot_start_idx + entry_i
        trades.append({
            "ticker": ticker,
            "entry_date": str(dates_all[entry_gi].date()),
            "exit_date": str(dates_all[gi_last].date()),
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

    ticker_stats = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in ticker_stats:
            ticker_stats[tk] = {"n": 0, "wins": 0, "pnl": 0}
        ticker_stats[tk]["n"] += 1
        if t["pnl"] > 0:
            ticker_stats[tk]["wins"] += 1
        ticker_stats[tk]["pnl"] += t["pnl"]
    for tk in ticker_stats:
        ticker_stats[tk]["wr"] = round(ticker_stats[tk]["wins"] / ticker_stats[tk]["n"] * 100, 1) if ticker_stats[tk]["n"] > 0 else 0
        ticker_stats[tk]["pnl"] = round(ticker_stats[tk]["pnl"], 2)

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
        "ticker_breakdown": ticker_stats,
    }


# ─── Define variants ─────────────────────────────────────────────────────────
variant_labels = {
    "A": "A: SMA cross entry, hold 10d",
    "B": "B: SMA cross entry, hold 20d",
    "C": "C: New 5d high after SMA cross, hold 10d",
    "D": "D: SMA cross entry, exit RSI>70 or 15d max",
    "E": "E: SMA cross + sector ETF above SMA, hold 10d",
    "F": "F: SMA cross + volume surge (1.5x avg), hold 10d",
}

# ─── Pre-compute entry signals for permutation testing ───────────────────────
flush_print("Pre-computing entry signals for permutation base...")
# For each variant, capture which (ticker, oot_day) would trigger an entry
# (ignoring position limits — just the raw signal)
entry_signals = {}
for vkey in variant_labels:
    sig = {}
    for t in TICKERS:
        s = np.zeros(n_oot, dtype=bool)
        for i in range(n_oot):
            gi = oot_start_idx + i
            if not recently_dipped[t][gi]:
                continue
            r = rsi[t][gi]
            if np.isnan(r) or r <= 50:
                continue

            if vkey in ("A", "B", "D", "E", "F"):
                if not sma_cross[t][gi]:
                    continue
            elif vkey == "C":
                dsc = days_since_cross[t][gi]
                if dsc > 10 or dsc < 1:
                    continue
                h5_today = high5[t][gi]
                h5_yest = high5[t][gi - 1] if gi > 0 else np.nan
                if (np.isnan(h5_today) or np.isnan(h5_yest) or
                        close_np[t][gi] < h5_today or h5_today <= h5_yest):
                    continue

            if vkey == "E":
                etf = SECTOR_ETFS.get(t)
                if etf and etf in sector_close_np and etf in sector_sma20:
                    etf_c = sector_close_np[etf][gi]
                    etf_s = sector_sma20[etf][gi]
                    if np.isnan(etf_c) or np.isnan(etf_s) or etf_c <= etf_s:
                        continue

            if vkey == "F":
                v_today = vol_np[t][gi]
                v_avg = avg_vol20[t][gi]
                if np.isnan(v_avg) or v_avg <= 0 or v_today < 1.5 * v_avg:
                    continue

            s[i] = True
        sig[t] = s
    entry_signals[vkey] = sig

flush_print("Entry signal pre-computation done.")

# ─── Run all variants ────────────────────────────────────────────────────────
results = {}
for vkey in ["A", "B", "C", "D", "E", "F"]:
    label = variant_labels[vkey]
    flush_print(f"\n{'='*60}")
    flush_print(f"Running variant {label}...")

    capital, eq, trades = run_backtest(vkey)

    metrics = calc_metrics(eq, trades, label)
    if metrics is None or metrics["n_trades"] == 0:
        flush_print(f"  No trades generated for variant {vkey}")
        results[vkey] = {"variant": label, "error": "no trades", "n_trades": 0}
        continue

    flush_print(f"  Final equity: ${metrics['final_equity']:.2f} ({metrics['total_return_pct']:+.1f}%)")
    flush_print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1f}%, PF: {metrics['profit_factor']:.2f}")
    flush_print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, MaxDD: {metrics['max_drawdown_pct']:.1f}%")
    flush_print(f"  Bull WR: {metrics['bull_wr']:.1f}% ({metrics['bull_trades']}), Bear WR: {metrics['bear_wr']:.1f}% ({metrics['bear_trades']})")

    # ─── Permutation test (1000 shuffles) ─────────────────────────────────
    flush_print(f"  Running {N_PERMS} permutations...")
    perm_returns = np.empty(N_PERMS)
    base_sig = entry_signals[vkey]

    for s in range(N_PERMS):
        rng = np.random.RandomState(s + 1)
        shuffled = {}
        for t in TICKERS:
            arr = base_sig[t].copy()
            rng.shuffle(arr)
            shuffled[t] = arr

        p_cap, _, _ = run_backtest(vkey, shuffled_entries=shuffled)
        perm_returns[s] = (p_cap / INITIAL_CAPITAL - 1) * 100

        if (s + 1) % 200 == 0:
            flush_print(f"    Permutation {s+1}/{N_PERMS}...")

    p_value = float(np.mean(perm_returns >= metrics["total_return_pct"]))
    perm_mean = float(np.mean(perm_returns))
    edge_vs_random = metrics["total_return_pct"] - perm_mean

    flush_print(f"  Permutation p-value: {p_value:.4f}")
    flush_print(f"  Edge vs random: {edge_vs_random:+.2f}%")

    metrics["permutation_p_value"] = round(p_value, 4)
    metrics["perm_mean_return"] = round(perm_mean, 2)
    metrics["edge_vs_random"] = round(edge_vs_random, 2)

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
flush_print("SUMMARY — Momentum After Mean Reversion")
flush_print(f"{'='*60}")
flush_print(f"{'Variant':<50} {'Return':>8} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'p-val':>7} {'Gates':>6}")
flush_print("-" * 95)
for vkey in ["A", "B", "C", "D", "E", "F"]:
    r = results[vkey]
    if "error" in r:
        flush_print(f"{r['variant']:<50} {'NO TRADES':>8}")
        continue
    flush_print(f"{r['variant']:<50} {r['total_return_pct']:>7.1f}% {r['sharpe']:>8.3f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} {r['permutation_p_value']:>7.4f} {r['gates_passed']:>6}")

# ─── Save results ────────────────────────────────────────────────────────────
output = {
    "strategy": "Momentum After Mean Reversion",
    "description": "Two-phase: wait for QMR-like recovery (close above 20d SMA after dip), then ride momentum continuation",
    "oot_period": f"{OOT_START} to {OOT_END}",
    "initial_capital": INITIAL_CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_each_way": SLIPPAGE,
    "n_permutations": N_PERMS,
    "universe": TICKERS,
    "setup_conditions": {
        "recent_dip": ">5% below 20-day high within last 20 days",
        "sma_cross": "Close crossed above 20-day SMA",
        "rsi_recovery": "RSI > 50",
    },
    "variants": results,
    "timestamp": datetime.now().isoformat(),
}

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

output = clean_for_json(output)

out_path = Path("/home/jupiter/Lvl3Quant/data/momentum_after_mr_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
flush_print(f"\nResults saved to {out_path}")
flush_print("Done.")
