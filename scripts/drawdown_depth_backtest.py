#!/usr/bin/env python3
"""
Drawdown Depth Tiers on Quality Stocks Backtest
Tests whether deeper dips produce better risk-adjusted returns.

Variants:
  A: Shallow dip (3-5% from 20d high), RSI<45, hold 10d
  B: Medium dip (5-10%), RSI<40, hold 10d
  C: Deep dip (10-15%), RSI<35, hold 10d
  D: Crash dip (>15%), RSI<30, hold 10d
  E: Progressive sizing - any dip>3%, size proportional to depth, RSI<45, hold 10d
  F: Deep-only concentrated - dip>10%, RSI<35, $300 size, max 2 concurrent, hold 10d
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Parameters ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 669.0
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
START = "2022-01-01"
END = "2026-07-31"
N_PERMUTATIONS = 1000
RSI_PERIOD = 14

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
raw = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
    high = raw["High"]
    low = raw["Low"]
else:
    close = raw[["Close"]].copy()
    high = raw[["High"]].copy()
    low = raw[["Low"]].copy()

# Drop any ticker that has no data
close = close.dropna(axis=1, how="all")
high = high[close.columns]
low = low[close.columns]

print(f"Got data for {len(close.columns)} tickers, {len(close)} trading days")

# ── Indicators ──────────────────────────────────────────────────────────────
# 20-day rolling high
high_20 = close.rolling(20).max()

# Drawdown from 20-day high (negative %)
drawdown_pct = (close - high_20) / high_20 * 100  # e.g. -7.3 means 7.3% below high

# RSI(14)
def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

rsi = pd.DataFrame({t: compute_rsi(close[t]) for t in close.columns}, index=close.index)

# SPY for regime classification
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)["Close"]
if isinstance(spy, pd.DataFrame):
    spy = spy.iloc[:, 0]
spy_sma200 = spy.rolling(200).mean()

# ── Trade Generation ────────────────────────────────────────────────────────

def get_dip_depth(dd_val):
    """Return dip depth bucket from drawdown % (negative number)."""
    d = abs(dd_val)
    if d < 3:
        return None
    elif d < 5:
        return "3-5%"
    elif d < 10:
        return "5-10%"
    elif d < 15:
        return "10-15%"
    else:
        return "15%+"

def generate_signals(variant):
    """
    Generate trade signals for a given variant.
    Returns list of dicts: {ticker, entry_date, exit_date, entry_price, exit_price,
                            return_pct, size_usd, dip_depth_bucket}
    """
    trades = []
    dates = close.index.tolist()

    # Track concurrent positions for variant F
    active_positions = []  # list of exit_date indices

    for i in range(20, len(dates) - HOLD_DAYS):
        date = dates[i]

        # Clean up expired positions for variant F
        if variant == "F":
            active_positions = [x for x in active_positions if x > i]

        for ticker in close.columns:
            dd = drawdown_pct.loc[date, ticker]
            r = rsi.loc[date, ticker]

            if pd.isna(dd) or pd.isna(r):
                continue

            dip = abs(dd)
            bucket = get_dip_depth(dd)

            # Check variant-specific conditions
            signal = False
            size_usd = 100.0  # default

            if variant == "A":
                signal = (3 <= dip < 5) and (r < 45)
                size_usd = 100.0
            elif variant == "B":
                signal = (5 <= dip < 10) and (r < 40)
                size_usd = 100.0
            elif variant == "C":
                signal = (10 <= dip < 15) and (r < 35)
                size_usd = 100.0
            elif variant == "D":
                signal = (dip >= 15) and (r < 30)
                size_usd = 100.0
            elif variant == "E":
                if dip >= 3 and r < 45:
                    signal = True
                    if dip < 5:
                        size_usd = 100.0
                    elif dip < 10:
                        size_usd = 150.0
                    elif dip < 15:
                        size_usd = 200.0
                    else:
                        size_usd = 250.0
            elif variant == "F":
                if dip >= 10 and r < 35:
                    if len(active_positions) < 2:
                        signal = True
                        size_usd = 300.0

            if not signal:
                continue

            entry_price = close.loc[date, ticker]
            exit_idx = min(i + HOLD_DAYS, len(dates) - 1)
            exit_date = dates[exit_idx]
            exit_price = close.loc[exit_date, ticker]

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            # Apply slippage
            entry_adj = entry_price * (1 + SLIPPAGE_BPS / 10000)
            exit_adj = exit_price * (1 - SLIPPAGE_BPS / 10000)

            ret = (exit_adj - entry_adj) / entry_adj

            trades.append({
                "ticker": ticker,
                "entry_date": str(date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "return_pct": float(ret),
                "size_usd": size_usd,
                "dip_depth_bucket": bucket,
            })

            if variant == "F":
                active_positions.append(exit_idx)

    return trades

# ── Backtest Engine ─────────────────────────────────────────────────────────

def backtest_trades(trades, capital=CAPITAL):
    """
    Compute equity curve and metrics from trade list.
    Each trade uses its size_usd (or $100 default).
    Returns dict of metrics + daily returns series.
    """
    if not trades:
        return None, pd.Series(dtype=float)

    # Build daily P&L
    all_dates = close.index
    daily_pnl = pd.Series(0.0, index=all_dates)

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_ = pd.Timestamp(t["exit_date"])
        size = t["size_usd"]

        # Distribute return evenly across holding period
        mask = (all_dates >= entry) & (all_dates <= exit_)
        n_days = mask.sum()
        if n_days > 0:
            daily_pnl[mask] += (t["return_pct"] * size) / n_days

    # Equity curve
    equity = capital + daily_pnl.cumsum()
    daily_ret = daily_pnl / capital

    # Metrics
    total_ret = float((equity.iloc[-1] - capital) / capital)

    # Annualized
    n_years = len(all_dates) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    daily_std = daily_ret.std()
    ann_std = daily_std * np.sqrt(252) if daily_std > 0 else 0

    sharpe = ann_ret / ann_std if ann_std > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252) if (daily_ret < 0).sum() > 0 else 0.001
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Win rate (per trade)
    wins = sum(1 for t in trades if t["return_pct"] > 0)
    wr = wins / len(trades) if trades else 0

    # Profit factor
    gross_profit = sum(t["return_pct"] * t["size_usd"] for t in trades if t["return_pct"] > 0)
    gross_loss = abs(sum(t["return_pct"] * t["size_usd"] for t in trades if t["return_pct"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Avg return per trade
    avg_ret = np.mean([t["return_pct"] for t in trades])

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr * 100, 1),
        "n_trades": len(trades),
        "avg_return_per_trade_pct": round(avg_ret * 100, 3),
        "total_pnl_usd": round(float(equity.iloc[-1] - capital), 2),
    }, daily_ret

# ── 5-Gate Validation ───────────────────────────────────────────────────────

def permutation_test(trades, actual_sharpe, n_perms=N_PERMUTATIONS):
    """
    Randomly sample entry dates from the full date/ticker universe,
    compute 10-day forward returns, and compare Sharpe to actual.
    Tests whether our TIMING of entries adds value vs random entry.
    """
    if not trades or len(trades) < 5:
        return 1.0

    n_trades = len(trades)
    dates = close.index.tolist()
    tickers = close.columns.tolist()
    valid_start = 20
    valid_end = len(dates) - HOLD_DAYS

    # Precompute all possible 10-day forward returns for speed
    fwd_ret = {}
    for t in tickers:
        for i in range(valid_start, valid_end):
            entry_p = close.iloc[i][t]
            exit_p = close.iloc[min(i + HOLD_DAYS, len(dates) - 1)][t]
            if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                entry_adj = entry_p * (1 + SLIPPAGE_BPS / 10000)
                exit_adj = exit_p * (1 - SLIPPAGE_BPS / 10000)
                fwd_ret[(t, i)] = (exit_adj - entry_adj) / entry_adj

    all_keys = list(fwd_ret.keys())
    if len(all_keys) < n_trades:
        return 1.0

    rng = np.random.RandomState(42)
    count_better = 0

    for _ in range(n_perms):
        # Random sample of same number of trades
        idxs = rng.choice(len(all_keys), size=n_trades, replace=True)
        perm_rets = np.array([fwd_ret[all_keys[j]] for j in idxs])
        if perm_rets.std() > 0:
            perm_sharpe = perm_rets.mean() / perm_rets.std() * np.sqrt(252 / HOLD_DAYS)
        else:
            perm_sharpe = 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms

def regime_test(trades):
    """
    Check if strategy works in both bull and bear regimes.
    Returns regime gap ratio.
    """
    if not trades:
        return 1.0, {}, {}

    bull_rets = []
    bear_rets = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        # Find nearest spy date
        idx = spy.index.get_indexer([entry], method="nearest")[0]
        if idx < 0 or idx >= len(spy):
            continue
        spy_date = spy.index[idx]

        if pd.isna(spy_sma200.loc[spy_date]) if spy_date in spy_sma200.index else True:
            continue

        if spy.loc[spy_date] > spy_sma200.loc[spy_date]:
            bull_rets.append(t["return_pct"])
        else:
            bear_rets.append(t["return_pct"])

    bull_sharpe = 0
    bear_sharpe = 0

    if bull_rets and np.std(bull_rets) > 0:
        bull_sharpe = np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252 / HOLD_DAYS)
    if bear_rets and np.std(bear_rets) > 0:
        bear_sharpe = np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252 / HOLD_DAYS)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    gap = abs(bull_sharpe - bear_sharpe) / max_abs

    bull_info = {"n": len(bull_rets), "sharpe": round(bull_sharpe, 3),
                 "avg_ret": round(np.mean(bull_rets) * 100, 3) if bull_rets else 0}
    bear_info = {"n": len(bear_rets), "sharpe": round(bear_sharpe, 3),
                 "avg_ret": round(np.mean(bear_rets) * 100, 3) if bear_rets else 0}

    return gap, bull_info, bear_info

def validate_5gate(trades, metrics):
    """Run 5-gate validation. Returns dict of gate results."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = {
        "pass": metrics["sharpe"] > 0.5,
        "value": metrics["sharpe"],
        "threshold": 0.5,
    }

    # Gate 2: Permutation test p < 0.05
    # Compute trade-level Sharpe for permutation comparison
    rets = np.array([t["return_pct"] for t in trades])
    if rets.std() > 0:
        trade_sharpe = rets.mean() / rets.std() * np.sqrt(252 / HOLD_DAYS)
    else:
        trade_sharpe = 0
    p_val = permutation_test(trades, trade_sharpe)
    gates["permutation_p_lt_0.05"] = {
        "pass": p_val < 0.05,
        "value": round(p_val, 4),
        "threshold": 0.05,
    }

    # Gate 3: Regime gap < 0.5
    gap, bull_info, bear_info = regime_test(trades)
    gates["regime_gap_lt_0.5"] = {
        "pass": gap < 0.5,
        "value": round(gap, 3),
        "threshold": 0.5,
        "bull": bull_info,
        "bear": bear_info,
    }

    # Gate 4: Max drawdown > -50%
    gates["max_dd_gt_neg50"] = {
        "pass": metrics["max_drawdown_pct"] > -50,
        "value": metrics["max_drawdown_pct"],
        "threshold": -50,
    }

    # Gate 5: At least 20 trades
    gates["min_20_trades"] = {
        "pass": metrics["n_trades"] >= 20,
        "value": metrics["n_trades"],
        "threshold": 20,
    }

    gates["passed_all"] = all(g["pass"] for g in gates.values() if isinstance(g, dict) and "pass" in g)
    gates["gates_passed"] = sum(1 for g in gates.values() if isinstance(g, dict) and g.get("pass", False))

    return gates

# ── Depth-Bucket Analysis ──────────────────────────────────────────────────

def depth_bucket_analysis(trades):
    """Average return per trade by dip depth bucket."""
    buckets = {"3-5%": [], "5-10%": [], "10-15%": [], "15%+": []}
    for t in trades:
        b = t.get("dip_depth_bucket")
        if b and b in buckets:
            buckets[b].append(t["return_pct"])

    result = {}
    for b, rets in buckets.items():
        if rets:
            result[b] = {
                "n_trades": len(rets),
                "avg_return_pct": round(np.mean(rets) * 100, 3),
                "median_return_pct": round(np.median(rets) * 100, 3),
                "win_rate": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                "std_pct": round(np.std(rets) * 100, 3),
            }
    return result

# ── Run All Variants ────────────────────────────────────────────────────────

VARIANTS = {
    "A": "Shallow dip (3-5%, RSI<45)",
    "B": "Medium dip (5-10%, RSI<40)",
    "C": "Deep dip (10-15%, RSI<35)",
    "D": "Crash dip (>15%, RSI<30)",
    "E": "Progressive sizing (>3%, size by depth)",
    "F": "Deep-only concentrated (>10%, $300, max 2)",
}

results = {
    "strategy": "Drawdown Depth Tiers on Quality Stocks",
    "universe": UNIVERSE,
    "period": f"{START} to {END}",
    "capital": CAPITAL,
    "hold_days": HOLD_DAYS,
    "slippage_bps": SLIPPAGE_BPS,
    "run_timestamp": datetime.datetime.now().isoformat(),
    "variants": {},
}

print("\n" + "=" * 70)
print("DRAWDOWN DEPTH TIERS BACKTEST")
print("=" * 70)

for var_key, var_desc in VARIANTS.items():
    print(f"\n{'─' * 60}")
    print(f"Variant {var_key}: {var_desc}")
    print(f"{'─' * 60}")

    trades = generate_signals(var_key)
    print(f"  Trades generated: {len(trades)}")

    if not trades:
        print("  ⚠ No trades generated. Skipping.")
        results["variants"][var_key] = {
            "description": var_desc,
            "n_trades": 0,
            "status": "NO_TRADES",
        }
        continue

    metrics, daily_ret = backtest_trades(trades)
    if metrics is None:
        results["variants"][var_key] = {
            "description": var_desc,
            "n_trades": 0,
            "status": "NO_TRADES",
        }
        continue

    print(f"  Sharpe:    {metrics['sharpe']:.3f}")
    print(f"  Sortino:   {metrics['sortino']:.3f}")
    print(f"  PF:        {metrics['profit_factor']:.3f}")
    print(f"  WR:        {metrics['win_rate']:.1f}%")
    print(f"  MaxDD:     {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Total Ret: {metrics['total_return_pct']:.2f}%")
    print(f"  Total PnL: ${metrics['total_pnl_usd']:.2f}")
    print(f"  Avg Ret/Trade: {metrics['avg_return_per_trade_pct']:.3f}%")

    # 5-Gate Validation
    print(f"\n  5-Gate Validation:")
    gates = validate_5gate(trades, metrics)
    for gname, gval in gates.items():
        if isinstance(gval, dict) and "pass" in gval:
            status = "PASS" if gval["pass"] else "FAIL"
            print(f"    {gname}: {status} (value={gval['value']}, threshold={gval['threshold']})")
    print(f"    => {gates['gates_passed']}/5 gates passed. Overall: {'PASS' if gates['passed_all'] else 'FAIL'}")

    # Depth bucket analysis
    depth_analysis = depth_bucket_analysis(trades)
    if depth_analysis:
        print(f"\n  Return by Dip Depth:")
        for bucket, info in sorted(depth_analysis.items()):
            print(f"    {bucket}: avg={info['avg_return_pct']:.3f}%, "
                  f"WR={info['win_rate']:.1f}%, n={info['n_trades']}")

    results["variants"][var_key] = {
        "description": var_desc,
        "metrics": metrics,
        "gates": gates,
        "depth_bucket_analysis": depth_analysis,
        "passed_5gate": gates["passed_all"],
    }

# ── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Var':<4} {'Description':<40} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Trades':>7} {'Gates':>6}")
print("-" * 76)
for var_key, var_desc in VARIANTS.items():
    v = results["variants"][var_key]
    if v.get("status") == "NO_TRADES":
        print(f"{var_key:<4} {var_desc:<40} {'N/A':>7} {'N/A':>6} {'N/A':>6} {0:>7} {'0/5':>6}")
    else:
        m = v["metrics"]
        g = v["gates"]
        print(f"{var_key:<4} {var_desc:<40} {m['sharpe']:>7.3f} {m['win_rate']:>5.1f}% {m['profit_factor']:>6.3f} {m['n_trades']:>7} {g['gates_passed']}/5{'*' if g['passed_all'] else ''}")

# Cross-variant depth analysis
print("\n" + "=" * 70)
print("DEEPER DIPS = BETTER RETURNS?")
print("=" * 70)
# Collect from variant E (has all buckets)
if "E" in results["variants"] and results["variants"]["E"].get("depth_bucket_analysis"):
    da = results["variants"]["E"]["depth_bucket_analysis"]
    print("From Variant E (Progressive Sizing — all depth buckets):")
    print(f"{'Bucket':<10} {'Avg Ret%':>10} {'Med Ret%':>10} {'WR':>8} {'N':>6}")
    print("-" * 48)
    for bucket in ["3-5%", "5-10%", "10-15%", "15%+"]:
        if bucket in da:
            info = da[bucket]
            print(f"{bucket:<10} {info['avg_return_pct']:>10.3f} {info['median_return_pct']:>10.3f} {info['win_rate']:>7.1f}% {info['n_trades']:>6}")
        else:
            print(f"{bucket:<10} {'—':>10} {'—':>10} {'—':>8} {'—':>6}")

# Also compare across isolated variants A-D
print("\nCross-variant comparison (isolated tiers A-D):")
print(f"{'Var':<4} {'Tier':<15} {'Avg Ret%':>10} {'Sharpe':>8} {'WR':>8} {'N':>6}")
print("-" * 55)
for vk in ["A", "B", "C", "D"]:
    v = results["variants"][vk]
    if v.get("status") == "NO_TRADES":
        print(f"{vk:<4} {VARIANTS[vk][:15]:<15} {'—':>10} {'—':>8} {'—':>8} {'—':>6}")
    else:
        m = v["metrics"]
        print(f"{vk:<4} {VARIANTS[vk][:15]:<15} {m['avg_return_per_trade_pct']:>10.3f} {m['sharpe']:>8.3f} {m['win_rate']:>7.1f}% {m['n_trades']:>6}")

# ── Save Results ────────────────────────────────────────────────────────────

output_path = "/home/jupiter/Lvl3Quant/data/drawdown_depth_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")
print("Done.")
