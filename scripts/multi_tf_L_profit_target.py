#!/usr/bin/env python3
"""
Multi-TF Dual Signal L with Profit Target Exits
Tests whether adding profit targets improves over fixed 10-day hold.

Variants:
  A: Baseline -- fixed 10-day hold (control)
  B: 3% profit target, max 15-day hold
  C: 5% profit target, max 15-day hold
  D: 2% profit target, max 10-day hold
  E: Trailing stop 2% from peak, max 15-day hold
  F: Dynamic target = 1.5x ATR(14) at entry, max 15-day hold
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Parameters ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
START = "2022-01-01"
END = "2026-07-31"
N_PERMUTATIONS = 1000

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
all_tickers = TICKERS + ["SPY"]
data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yf.download
close = data["Close"].copy()
high = data["High"].copy()
low = data["Low"].copy()
open_ = data["Open"].copy()

spy_close = close["SPY"].dropna()
spy_sma200 = spy_close.rolling(200).mean()

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")


# ── Indicator Helpers ───────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(high_s, low_s, close_s, period=14):
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _count_trailing(arr):
    """Count trailing 1s in array."""
    count = 0
    for v in reversed(arr):
        if v == 1:
            count += 1
        else:
            break
    return count


# ── Precompute signals per ticker ───────────────────────────────────────────
print("Computing signals...")

signals = {}  # ticker -> list of entry dates

for ticker in TICKERS:
    c = close[ticker].dropna()
    h = high[ticker].dropna()
    l = low[ticker].dropna()
    o = open_[ticker].dropna()

    if len(c) < 100:
        continue

    # Daily RSI
    rsi = compute_rsi(c, 14)

    # 20-day high and drawdown
    high20 = c.rolling(20).max()
    drawdown_from_high = (c - high20) / high20

    # Consecutive red days: close < open
    red_day = (c < o).astype(int)
    # Count consecutive red days ending yesterday
    consec_red = red_day.rolling(10, min_periods=1).apply(
        lambda x: _count_trailing(x), raw=True
    )

    # Green day today
    green_today = c > o

    # Weekly RSI declining for 2+ weeks
    weekly_close = c.resample("W-FRI").last().dropna()
    weekly_rsi = compute_rsi(weekly_close, 14)
    weekly_rsi_declining = (weekly_rsi < weekly_rsi.shift(1)) & (weekly_rsi.shift(1) < weekly_rsi.shift(2))
    # Map weekly signal to daily dates (forward-fill to next week)
    weekly_rsi_declining = weekly_rsi_declining.reindex(c.index, method="ffill").fillna(False)

    # ATR for dynamic target
    atr = compute_atr(h, l, c, 14)

    ticker_signals = []
    for i in range(20, len(c)):
        dt = c.index[i]
        if (drawdown_from_high.iloc[i] <= -0.05 and
            rsi.iloc[i] < 35 and
            green_today.iloc[i] and
            consec_red.iloc[i - 1] >= 3 and  # yesterday had 3+ consecutive red
            weekly_rsi_declining.loc[dt]):
            ticker_signals.append({
                "date": dt,
                "price": c.iloc[i],
                "atr": atr.iloc[i] if not np.isnan(atr.iloc[i]) else c.iloc[i] * 0.02,
            })

    signals[ticker] = ticker_signals


# Flatten all signals into a sorted list
all_entries = []
for ticker, sigs in signals.items():
    for s in sigs:
        all_entries.append({
            "ticker": ticker,
            "date": s["date"],
            "price": s["price"],
            "atr": s["atr"],
        })

all_entries.sort(key=lambda x: x["date"])
print(f"Total entry signals found: {len(all_entries)}")


# ── Exit Strategy Functions ─────────────────────────────────────────────────
def simulate_variant(entries, variant, close_df):
    """
    Simulate a variant and return list of trades with entry/exit info.
    Each trade: {ticker, entry_date, entry_price, exit_date, exit_price, hold_days, return_pct}
    """
    trades = []
    open_positions = []  # list of dicts with exit tracking info

    all_dates = close_df.index

    for entry in entries:
        ticker = entry["ticker"]
        entry_date = entry["date"]
        entry_price = entry["price"]
        atr_at_entry = entry["atr"]

        # Check concurrent position limit
        # Close any positions that should have been closed by entry_date
        open_positions = [p for p in open_positions if p["max_exit_date"] >= entry_date]

        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Find entry index in all_dates
        try:
            entry_idx = all_dates.get_loc(entry_date)
        except KeyError:
            continue

        # Apply slippage to entry
        slipped_entry = entry_price * (1 + SLIPPAGE_BPS / 10000)

        # Position sizing
        shares = int(MAX_PER_TRADE / slipped_entry)
        if shares < 1:
            continue

        # Determine max hold and exit logic based on variant
        if variant == "A":
            max_hold = 10
        elif variant in ("B", "C", "E", "F"):
            max_hold = 15
        elif variant == "D":
            max_hold = 10
        else:
            max_hold = 10

        # Simulate day by day
        exit_price = None
        exit_date = None
        peak_price = slipped_entry

        for d in range(1, max_hold + 1):
            idx = entry_idx + d
            if idx >= len(all_dates):
                # Exit at last available date
                idx = len(all_dates) - 1
                day_date = all_dates[idx]
                exit_price = close_df[ticker].iloc[idx]
                exit_date = day_date
                break

            day_date = all_dates[idx]
            day_close = close_df[ticker].iloc[idx]

            if pd.isna(day_close):
                continue

            peak_price = max(peak_price, day_close)

            if variant == "A":
                # Fixed 10-day hold
                if d == max_hold:
                    exit_price = day_close
                    exit_date = day_date

            elif variant == "B":
                # 3% profit target
                ret = (day_close - slipped_entry) / slipped_entry
                if ret >= 0.03 or d == max_hold:
                    exit_price = day_close
                    exit_date = day_date

            elif variant == "C":
                # 5% profit target
                ret = (day_close - slipped_entry) / slipped_entry
                if ret >= 0.05 or d == max_hold:
                    exit_price = day_close
                    exit_date = day_date

            elif variant == "D":
                # 2% profit target
                ret = (day_close - slipped_entry) / slipped_entry
                if ret >= 0.02 or d == max_hold:
                    exit_price = day_close
                    exit_date = day_date

            elif variant == "E":
                # Trailing stop 2% from peak
                drawdown = (day_close - peak_price) / peak_price
                if drawdown <= -0.02 or d == max_hold:
                    exit_price = day_close
                    exit_date = day_date

            elif variant == "F":
                # Dynamic target = 1.5x ATR(14) at entry
                target_pct = (1.5 * atr_at_entry) / slipped_entry
                ret = (day_close - slipped_entry) / slipped_entry
                if ret >= target_pct or d == max_hold:
                    exit_price = day_close
                    exit_date = day_date

            if exit_price is not None:
                break

        if exit_price is None:
            # Fallback: exit at max hold
            idx = min(entry_idx + max_hold, len(all_dates) - 1)
            exit_price = close_df[ticker].iloc[idx]
            exit_date = all_dates[idx]

        if pd.isna(exit_price):
            continue

        # Apply exit slippage
        slipped_exit = exit_price * (1 - SLIPPAGE_BPS / 10000)

        trade_return = (slipped_exit - slipped_entry) / slipped_entry
        hold_days = (exit_date - entry_date).days

        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "entry_price": slipped_entry,
            "exit_date": exit_date,
            "exit_price": slipped_exit,
            "hold_days": hold_days,
            "return_pct": trade_return,
            "shares": shares,
            "pnl": shares * (slipped_exit - slipped_entry),
        })

        open_positions.append({
            "ticker": ticker,
            "max_exit_date": exit_date,
        })

    return trades


# ── Performance Metrics ─────────────────────────────────────────────────────
def compute_metrics(trades, spy_close_s, spy_sma200_s):
    if len(trades) == 0:
        return None

    returns = np.array([t["return_pct"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    hold_days = np.array([t["hold_days"] for t in trades])

    # Build daily equity curve from trades
    total_pnl = sum(pnls)
    total_return = total_pnl / CAPITAL

    # Approximate daily returns by spreading trade returns over hold days
    # For Sharpe/Sortino, use per-trade returns annualized
    n_trades = len(trades)
    win_rate = np.mean(returns > 0)

    # Annualize: assume ~252 trading days, avg trades per year
    first_date = min(t["entry_date"] for t in trades)
    last_date = max(t["exit_date"] for t in trades)
    years = max((last_date - first_date).days / 365.25, 0.5)
    trades_per_year = n_trades / years

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n_trades > 1 else 1e-6

    # Sharpe (annualized from per-trade)
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-6
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    equity = CAPITAL
    peak_equity = CAPITAL
    max_dd = 0
    equity_curve = []
    sorted_trades = sorted(trades, key=lambda t: t["exit_date"])
    for t in sorted_trades:
        equity += t["pnl"]
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity
        max_dd = min(max_dd, dd)

    # Regime analysis
    bull_returns = []
    bear_returns = []
    for t in trades:
        entry_d = t["entry_date"]
        if entry_d in spy_sma200_s.index and entry_d in spy_close_s.index:
            if spy_close_s.loc[entry_d] > spy_sma200_s.loc[entry_d]:
                bull_returns.append(t["return_pct"])
            else:
                bear_returns.append(t["return_pct"])
        else:
            # Find nearest
            nearest = spy_sma200_s.index[spy_sma200_s.index.get_indexer([entry_d], method="nearest")]
            if len(nearest) > 0:
                nd = nearest[0]
                if spy_close_s.loc[nd] > spy_sma200_s.loc[nd]:
                    bull_returns.append(t["return_pct"])
                else:
                    bear_returns.append(t["return_pct"])

    bull_returns = np.array(bull_returns) if bull_returns else np.array([0.0])
    bear_returns = np.array(bear_returns) if bear_returns else np.array([0.0])

    bull_mean = np.mean(bull_returns)
    bear_mean = np.mean(bear_returns)
    bull_std = np.std(bull_returns, ddof=1) if len(bull_returns) > 1 else 1e-6
    bear_std = np.std(bear_returns, ddof=1) if len(bear_returns) > 1 else 1e-6

    bull_sharpe = (bull_mean / bull_std) * np.sqrt(trades_per_year) if bull_std > 0 else 0
    bear_sharpe = (bear_mean / bear_std) * np.sqrt(trades_per_year) if bear_std > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(min(profit_factor, 99.99), 3),
        "max_drawdown": round(max_dd, 4),
        "total_return": round(total_return, 4),
        "num_trades": n_trades,
        "avg_hold_days": round(np.mean(hold_days), 1),
        "total_pnl": round(total_pnl, 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 4),
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
    }


# ── Permutation Test ────────────────────────────────────────────────────────
def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Shuffle entry dates randomly among the universe, measure p-value of Sharpe."""
    if len(trades) < 5:
        return 1.0

    actual_returns = np.array([t["return_pct"] for t in trades])
    actual_sharpe = np.mean(actual_returns) / (np.std(actual_returns, ddof=1) + 1e-9)

    # For permutation: randomly pick dates and tickers, compute returns
    all_dates_list = close.index.tolist()
    n = len(trades)
    count_better = 0

    rng = np.random.default_rng(42)

    for _ in range(n_perms):
        perm_returns = []
        for t in trades:
            # Random date, same ticker
            rand_idx = rng.integers(20, len(all_dates_list) - 20)
            rand_ticker = rng.choice(TICKERS)
            entry_p = close[rand_ticker].iloc[rand_idx]
            hold = t["hold_days"]
            exit_idx = min(rand_idx + max(hold, 1), len(all_dates_list) - 1)
            exit_p = close[rand_ticker].iloc[exit_idx]
            if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                perm_returns.append((exit_p - entry_p) / entry_p)
            else:
                perm_returns.append(0.0)

        perm_returns = np.array(perm_returns)
        perm_sharpe = np.mean(perm_returns) / (np.std(perm_returns, ddof=1) + 1e-9)
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round(count_better / n_perms, 4)


# ── 5-Gate Validation ───────────────────────────────────────────────────────
def five_gate(metrics, p_value):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50pct": metrics["max_drawdown"] > -0.50,
        "min_20_trades": metrics["num_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ── Run All Variants ────────────────────────────────────────────────────────
VARIANTS = {
    "A": "Baseline: fixed 10-day hold",
    "B": "3% profit target, max 15-day hold",
    "C": "5% profit target, max 15-day hold",
    "D": "2% profit target, max 10-day hold",
    "E": "Trailing stop 2% from peak, max 15-day hold",
    "F": "Dynamic target 1.5x ATR(14), max 15-day hold",
}

results = {}

for var_id, var_desc in VARIANTS.items():
    print(f"\n{'='*60}")
    print(f"Variant {var_id}: {var_desc}")
    print(f"{'='*60}")

    trades = simulate_variant(all_entries, var_id, close)
    print(f"  Trades: {len(trades)}")

    if len(trades) == 0:
        print("  No trades generated, skipping.")
        results[var_id] = {"description": var_desc, "error": "no trades"}
        continue

    metrics = compute_metrics(trades, spy_close, spy_sma200)

    print(f"  Sharpe: {metrics['sharpe']}")
    print(f"  Sortino: {metrics['sortino']}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {metrics['profit_factor']}")
    print(f"  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Avg Hold: {metrics['avg_hold_days']} days")
    print(f"  Bull Sharpe: {metrics['bull_sharpe']}, Bear Sharpe: {metrics['bear_sharpe']}")
    print(f"  Regime Gap: {metrics['regime_gap']:.4f}")

    print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...")
    p_value = permutation_test(trades)
    print(f"  Permutation p-value: {p_value}")

    gates = five_gate(metrics, p_value)
    print(f"  5-Gate Results:")
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {gate_name}: {status}")

    results[var_id] = {
        "description": var_desc,
        **metrics,
        "permutation_p": p_value,
        "five_gate": gates,
    }

# ── Summary Table ───────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("SUMMARY: Multi-TF L Profit Target Variants")
print(f"{'='*80}")
print(f"{'Var':<4} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Return':>8} {'#Tr':>4} {'AvgHold':>8} {'RegGap':>7} {'Perm_p':>7} {'Pass':>5}")
print("-" * 80)

for var_id in VARIANTS:
    r = results[var_id]
    if "error" in r:
        print(f"{var_id:<4} {'N/A':>7} -- no trades --")
        continue

    pass_str = "YES" if r["five_gate"]["all_pass"] else "NO"
    print(f"{var_id:<4} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} {r['max_drawdown']:>6.2%} {r['total_return']:>7.2%} {r['num_trades']:>4} {r['avg_hold_days']:>7.1f}d {r['regime_gap']:>7.4f} {r['permutation_p']:>7.4f} {pass_str:>5}")

# ── Save Results ────────────────────────────────────────────────────────────
output_path = "/home/jupiter/Lvl3Quant/data/multi_tf_L_profit_target_results.json"

# Convert dates to strings for JSON serialization
serializable = {}
for k, v in results.items():
    serializable[k] = v

with open(output_path, "w") as f:
    json.dump(serializable, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
