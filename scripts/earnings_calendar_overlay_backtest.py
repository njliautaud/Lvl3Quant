#!/usr/bin/env python3
"""
Earnings Calendar Overlay on Dual Signal D Backtest
====================================================
Tests whether avoiding trades near earnings dates improves/hurts Dual Signal D.

Earnings proxy: days where |daily return| > 5% (large moves ≈ earnings for quality stocks).

6 Variants:
  A: Baseline (no earnings filter)
  B: Skip pre-earnings (10 days before)
  C: Skip post-earnings (5 days after)
  D: Skip both (10 before, 5 after)
  E: Earnings-only (within 15 days before earnings)
  F: Post-earnings dip buy (within 10 days after earnings drop >5%)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META"
]
START_DATE = "2021-06-01"  # buffer for 20-day lookback
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
HOLD_DAYS = 10
DROP_THRESHOLD = 0.05   # 5% drop from 20d high
RSI_THRESHOLD = 35
CONSEC_RED_DAYS = 3
EARNINGS_MOVE_THRESHOLD = 0.05  # 5% absolute daily return = earnings proxy
N_PERMUTATIONS = 1000
SEED = 42

np.random.seed(SEED)

# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    """Download price data for all tickers + SPY."""
    all_tickers = TICKERS + ["SPY"]
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
            else:
                print(f"  {ticker}: SKIPPED (insufficient data)")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
    return data


def detect_earnings_dates(df):
    """Detect earnings dates as days where |daily return| > threshold."""
    returns = df["Close"].pct_change()
    earnings_mask = returns.abs() > EARNINGS_MOVE_THRESHOLD
    return df.index[earnings_mask].tolist()


def find_dual_signal_d_entries(df):
    """
    Dual Signal D: stock drops >5% from 20-day high + RSI<35 + first green after 3+ consecutive red days.
    Returns list of entry dates.
    """
    close = df["Close"].copy()
    high_20d = close.rolling(20).max()
    drop_pct = (close - high_20d) / high_20d
    rsi = compute_rsi(close)
    daily_ret = close.pct_change()
    is_green = daily_ret > 0
    is_red = daily_ret < 0

    # Count consecutive red days ending yesterday
    consec_red = pd.Series(0, index=close.index, dtype=int)
    for i in range(1, len(close)):
        if is_red.iloc[i-1]:
            consec_red.iloc[i] = consec_red.iloc[i-1] + 1
        else:
            consec_red.iloc[i] = 0

    entries = []
    for i in range(20, len(close)):
        dt = close.index[i]
        if (drop_pct.iloc[i] <= -DROP_THRESHOLD and
            rsi.iloc[i] < RSI_THRESHOLD and
            is_green.iloc[i] and
            consec_red.iloc[i] >= CONSEC_RED_DAYS):
            entries.append(dt)
    return entries


def is_near_earnings(entry_date, earnings_dates, days_before=None, days_after=None):
    """Check if entry_date is within days_before/days_after of any earnings date (trading days approx)."""
    for ed in earnings_dates:
        diff = (entry_date - ed).days
        if days_before is not None and -days_before * 1.5 <= diff < 0:
            # entry is before earnings (diff is negative): within days_before trading days
            # Use 1.5x multiplier to approximate trading days from calendar days
            return True
        if days_after is not None and 0 < diff <= days_after * 1.5:
            # entry is after earnings (diff is positive)
            return True
        if days_before is not None and days_after is not None:
            if -days_before * 1.5 <= diff <= days_after * 1.5:
                return True
    return False


def is_within_window(entry_date, earnings_dates, days_window, direction="before"):
    """Check if entry is within N trading days before/after any earnings date."""
    for ed in earnings_dates:
        diff_cal = (entry_date - ed).days
        diff_td = abs(diff_cal) / 1.4  # approx trading days
        if direction == "before" and diff_cal < 0 and diff_td <= days_window:
            return True
        if direction == "after" and diff_cal > 0 and diff_td <= days_window:
            return True
    return False


def is_post_earnings_drop(entry_date, df, earnings_dates, days_window=10):
    """Check if entry is within days_window after an earnings DROP (>5%)."""
    returns = df["Close"].pct_change()
    for ed in earnings_dates:
        diff_cal = (entry_date - ed).days
        diff_td = diff_cal / 1.4
        if 0 < diff_td <= days_window:
            # Check if the earnings move was a DROP
            if ed in returns.index:
                ret_val = returns.loc[ed]
                if ret_val < -EARNINGS_MOVE_THRESHOLD:
                    return True
    return False


def filter_entries(entries, earnings_dates, df, variant):
    """Filter entries based on variant."""
    if variant == "A":
        return entries  # baseline

    filtered = []
    for entry in entries:
        if variant == "B":
            # Skip pre-earnings: skip if within 10 trading days BEFORE earnings
            if not is_within_window(entry, earnings_dates, 10, "before"):
                filtered.append(entry)
        elif variant == "C":
            # Skip post-earnings: skip if within 5 trading days AFTER earnings
            if not is_within_window(entry, earnings_dates, 5, "after"):
                filtered.append(entry)
        elif variant == "D":
            # Skip both
            if (not is_within_window(entry, earnings_dates, 10, "before") and
                not is_within_window(entry, earnings_dates, 5, "after")):
                filtered.append(entry)
        elif variant == "E":
            # Earnings-only: ONLY take if within 15 days BEFORE earnings
            if is_within_window(entry, earnings_dates, 15, "before"):
                filtered.append(entry)
        elif variant == "F":
            # Post-earnings dip buy: ONLY if within 10 days AFTER earnings DROP
            if is_post_earnings_drop(entry, df, earnings_dates, 10):
                filtered.append(entry)
    return filtered


def run_backtest(all_data, spy_data, variant):
    """Run backtest for a given variant. Returns trade list and equity curve."""
    trades = []

    for ticker in TICKERS:
        if ticker not in all_data:
            continue
        df = all_data[ticker]
        earnings_dates = detect_earnings_dates(df)
        entries = find_dual_signal_d_entries(df)

        # Filter to OOT period
        oot_start = pd.Timestamp(OOT_START)
        entries = [e for e in entries if e >= oot_start]

        # Apply variant filter
        entries = filter_entries(entries, earnings_dates, df, variant)

        for entry_date in entries:
            entry_idx = df.index.get_loc(entry_date)
            exit_idx = min(entry_idx + HOLD_DAYS, len(df) - 1)
            exit_date = df.index[exit_idx]

            entry_price = float(df["Close"].iloc[entry_idx])
            exit_price = float(df["Close"].iloc[exit_idx])

            # Apply slippage
            entry_cost = entry_price * (1 + SLIPPAGE_PCT)
            exit_revenue = exit_price * (1 - SLIPPAGE_PCT)

            ret_pct = (exit_revenue - entry_cost) / entry_cost

            trades.append({
                "ticker": ticker,
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "return_pct": round(ret_pct, 6),
                "pnl_dollars": round(ret_pct * MAX_PER_TRADE, 2),
            })

    # Sort by entry date
    trades.sort(key=lambda t: t["entry_date"])

    # Simulate with capital constraints
    capital = STARTING_CAPITAL
    active_trades = []
    executed_trades = []
    equity_curve = [{"date": OOT_START, "equity": capital}]

    all_dates = sorted(set([t["entry_date"] for t in trades] + [t["exit_date"] for t in trades]))

    for date_str in all_dates:
        # Close trades that exit today
        still_active = []
        for at in active_trades:
            if at["exit_date"] == date_str:
                capital += at["allocation"] * (1 + at["return_pct"])
            else:
                still_active.append(at)
        active_trades = still_active

        # Open new trades
        new_entries = [t for t in trades if t["entry_date"] == date_str]
        for t in new_entries:
            if len(active_trades) >= MAX_CONCURRENT:
                continue
            alloc = min(MAX_PER_TRADE, capital * 0.95)  # keep 5% reserve
            if alloc < 10:
                continue
            capital -= alloc
            active_trade = dict(t)
            active_trade["allocation"] = alloc
            active_trades.append(active_trade)
            executed_trades.append(active_trade)

        equity_curve.append({"date": date_str, "equity": round(capital + sum(at["allocation"] for at in active_trades), 2)})

    # Close remaining
    for at in active_trades:
        capital += at["allocation"] * (1 + at["return_pct"])

    return executed_trades, equity_curve, capital


def compute_metrics(trades, equity_curve, final_capital):
    """Compute strategy metrics."""
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
            "max_drawdown_pct": 0, "total_return_pct": 0, "num_trades": 0,
            "avg_return_pct": 0, "total_pnl": 0,
        }

    returns = [t["return_pct"] for t in trades]
    returns_arr = np.array(returns)

    wins = returns_arr[returns_arr > 0]
    losses = returns_arr[returns_arr <= 0]

    win_rate = len(wins) / len(returns_arr) if len(returns_arr) > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    mean_ret = returns_arr.mean()
    std_ret = returns_arr.std() if len(returns_arr) > 1 else 0.0001

    # Annualize: assume ~25 trades/year avg
    trades_per_year = max(len(returns_arr) / 4.5, 1)  # ~4.5 years of OOT
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside_ret = returns_arr[returns_arr < 0]
    downside_std = downside_ret.std() if len(downside_ret) > 1 else 0.0001
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd

    total_return = (final_capital - STARTING_CAPITAL) / STARTING_CAPITAL
    total_pnl = sum(t.get("pnl_dollars", t["return_pct"] * MAX_PER_TRADE) for t in trades)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return * 100, 2),
        "num_trades": len(trades),
        "avg_return_pct": round(mean_ret * 100, 3),
        "total_pnl": round(total_pnl, 2),
    }


def regime_analysis(trades, spy_data):
    """Split trades into bull/bear regimes based on SPY > 200-SMA."""
    spy_close = spy_data["Close"]
    spy_sma200 = spy_close.rolling(200).mean()

    bull_returns = []
    bear_returns = []

    for t in trades:
        entry_dt = pd.Timestamp(t["entry_date"])
        # Find nearest SPY date
        nearest_idx = spy_data.index.searchsorted(entry_dt)
        nearest_idx = min(nearest_idx, len(spy_data) - 1)

        spy_val = float(spy_close.iloc[nearest_idx])
        sma_val = float(spy_sma200.iloc[nearest_idx]) if not pd.isna(spy_sma200.iloc[nearest_idx]) else spy_val

        if spy_val > sma_val:
            bull_returns.append(t["return_pct"])
        else:
            bear_returns.append(t["return_pct"])

    def calc_sharpe(rets):
        if len(rets) < 2:
            return 0
        arr = np.array(rets)
        tpy = max(len(arr) / 4.5, 1)
        std = arr.std()
        if std == 0:
            return 0
        return float((arr.mean() / std) * np.sqrt(tpy))

    bull_sharpe = calc_sharpe(bull_returns)
    bear_sharpe = calc_sharpe(bear_returns)
    gap = abs(bull_sharpe - bear_sharpe)

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(gap, 3),
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
    }


def permutation_test(trades, n_perm=N_PERMUTATIONS):
    """Permutation test: shuffle returns, compute fraction with higher mean."""
    if len(trades) < 5:
        return {"p_value": 1.0, "observed_mean": 0}

    returns = np.array([t["return_pct"] for t in trades])
    observed_mean = returns.mean()

    count_higher = 0
    for _ in range(n_perm):
        shuffled = returns.copy()
        np.random.shuffle(shuffled)
        # Randomly flip signs to test against null of zero mean
        signs = np.random.choice([-1, 1], size=len(shuffled))
        null_mean = (shuffled * signs).mean()
        if null_mean >= observed_mean:
            count_higher += 1

    p_value = count_higher / n_perm
    return {
        "p_value": round(p_value, 4),
        "observed_mean_pct": round(observed_mean * 100, 4),
    }


def five_gate_validation(metrics, regime, perm):
    """5-Gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm["p_value"] < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["num_trades"] >= 20,
    }
    gates["passed"] = sum(gates.values())
    gates["total"] = 5
    gates["all_passed"] = all(v for k, v in gates.items() if k not in ("passed", "total", "all_passed"))
    return gates


def main():
    print("=" * 70)
    print("EARNINGS CALENDAR OVERLAY ON DUAL SIGNAL D BACKTEST")
    print("=" * 70)

    # Download data
    all_data = download_data()
    spy_data = all_data.pop("SPY", None)
    if spy_data is None:
        print("ERROR: Could not download SPY data")
        return

    print(f"\nLoaded {len(all_data)} tickers + SPY")

    # Detect earnings events summary
    print("\n── Earnings Event Detection (|return| > 5%) ──")
    for ticker in TICKERS:
        if ticker in all_data:
            edates = detect_earnings_dates(all_data[ticker])
            print(f"  {ticker}: {len(edates)} events detected")

    # Run all variants
    variants = {
        "A": "Baseline (no earnings filter)",
        "B": "Skip pre-earnings (10d before)",
        "C": "Skip post-earnings (5d after)",
        "D": "Skip both (10d before + 5d after)",
        "E": "Earnings-only (within 15d before)",
        "F": "Post-earnings dip buy (within 10d after drop)",
    }

    results = {}
    baseline_metrics = None

    print("\n── Running Variants ──")
    for var_key, var_desc in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {var_key}: {var_desc}")
        print(f"{'─' * 60}")

        trades, equity_curve, final_capital = run_backtest(all_data, spy_data, var_key)
        metrics = compute_metrics(trades, equity_curve, final_capital)
        regime = regime_analysis(trades, spy_data)
        perm = permutation_test(trades)
        gates = five_gate_validation(metrics, regime, perm)

        if var_key == "A":
            baseline_metrics = metrics

        # Compare to baseline
        comparison = {}
        if baseline_metrics and var_key != "A":
            comparison = {
                "sharpe_delta": round(metrics["sharpe"] - baseline_metrics["sharpe"], 3),
                "sortino_delta": round(metrics["sortino"] - baseline_metrics["sortino"], 3),
                "wr_delta": round((metrics["win_rate"] - baseline_metrics["win_rate"]) * 100, 2),
                "pf_delta": round(metrics["profit_factor"] - baseline_metrics["profit_factor"], 3),
                "trades_delta": metrics["num_trades"] - baseline_metrics["num_trades"],
            }

        results[var_key] = {
            "variant": var_key,
            "description": var_desc,
            "metrics": metrics,
            "regime": regime,
            "permutation_test": perm,
            "five_gate": gates,
            "vs_baseline": comparison,
        }

        # Print summary
        m = metrics
        print(f"  Trades: {m['num_trades']}  |  WR: {m['win_rate']*100:.1f}%  |  PF: {m['profit_factor']:.2f}")
        print(f"  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}  |  MDD: {m['max_drawdown_pct']:.1f}%")
        print(f"  Total Return: {m['total_return_pct']:.1f}%  |  Total PnL: ${m['total_pnl']:.2f}")
        print(f"  Avg Return/Trade: {m['avg_return_pct']:.3f}%")
        print(f"  Regime — Bull Sharpe: {regime['bull_sharpe']:.3f} ({regime['bull_trades']} trades) | Bear: {regime['bear_sharpe']:.3f} ({regime['bear_trades']} trades) | Gap: {regime['regime_gap']:.3f}")
        print(f"  Perm test p-value: {perm['p_value']:.4f}")
        print(f"  5-Gate: {gates['passed']}/{gates['total']} {'PASS' if gates['all_passed'] else 'FAIL'}")
        if comparison:
            print(f"  vs Baseline — Sharpe Δ: {comparison['sharpe_delta']:+.3f}  WR Δ: {comparison['wr_delta']:+.1f}pp  Trades Δ: {comparison['trades_delta']:+d}")

    # Summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Var':<4} {'Description':<42} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MDD%':>6} {'Ret%':>7} {'Gate':>5}")
    print("-" * 100)
    for var_key in variants:
        r = results[var_key]
        m = r["metrics"]
        g = r["five_gate"]
        print(f"{var_key:<4} {r['description']:<42} {m['num_trades']:>6} {m['win_rate']*100:>5.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>5.1f}% {m['total_return_pct']:>6.1f}% {g['passed']}/{g['total']}")

    # Best variant
    valid_variants = {k: v for k, v in results.items() if v["metrics"]["num_trades"] >= 5}
    if valid_variants:
        best_key = max(valid_variants, key=lambda k: valid_variants[k]["metrics"]["sharpe"])
        print(f"\nBest by Sharpe: Variant {best_key} — {variants[best_key]} (Sharpe={results[best_key]['metrics']['sharpe']:.3f})")

    # Save results
    output = {
        "backtest_name": "Earnings Calendar Overlay on Dual Signal D",
        "run_date": datetime.now().isoformat(),
        "config": {
            "tickers": TICKERS,
            "oot_period": f"{OOT_START} to {OOT_END}",
            "starting_capital": STARTING_CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_pct": SLIPPAGE_PCT,
            "hold_days": HOLD_DAYS,
            "drop_threshold": DROP_THRESHOLD,
            "rsi_threshold": RSI_THRESHOLD,
            "consec_red_days": CONSEC_RED_DAYS,
            "earnings_move_threshold": EARNINGS_MOVE_THRESHOLD,
        },
        "variants": results,
    }

    output_path = Path("/home/jupiter/Lvl3Quant/data/earnings_calendar_overlay_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
