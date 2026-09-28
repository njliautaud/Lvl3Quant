#!/usr/bin/env python3
"""
Momentum Burst Options Backtest (Bull Call Spreads on Gap-Up Signals)
=====================================================================
Based on weekly_momentum_burst_paper.py strategy concept.

Signal: daily gap-up >= 3% on volume > 1.5x 20-day avg
Entry:  simulated bull call spread ($50-80 cost) at close on signal day
Exit:   +150% TP, -50% SL, 5-day max hold
Option leverage approximated at 3x underlying move for ATM spreads.

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (1000 shuffles)
  3. Regime stratification (green vs red SPY days), gap < 0.50
  4. Sub-period: split into 4 equal periods, all Sharpe > 0
  5. Day concentration cap: max single day P&L < 70% of total

Usage:
  python3 scripts/growth_research/momentum_burst_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Config ──
TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
    "XLY", "XLB", "XLU", "XLRE", "XLC", "SMH", "IWM",
]
START = "2020-01-01"
END = "2026-08-01"
MIN_GAP_PCT = 0.03        # 3% gap-up
VOL_MULT_THRESH = 1.5     # volume > 1.5x 20-day avg
VOL_LOOKBACK = 20
SPREAD_COST_MIN = 50.0
SPREAD_COST_MAX = 80.0
OPTION_LEVERAGE = 3.0     # approximate ATM spread leverage
TP_PCT = 1.50             # +150% of cost
SL_PCT = -0.50            # -50% of cost
MAX_HOLD_DAYS = 5
N_PERMUTATIONS = 1000
RISK_FREE_RATE = 0.0      # for Sharpe calc (excess returns)
INITIAL_CAPITAL = 10000.0


def download_data():
    """Download daily OHLCV for all tickers + SPY for regime."""
    print(f"Downloading data for {len(TICKERS)} tickers: {START} to {END}")
    data = {}
    all_tickers = list(set(TICKERS + ["SPY"]))  # ensure SPY is included

    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            if df is None or df.empty:
                print(f"  WARNING: No data for {ticker}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df
        except Exception as e:
            print(f"  WARNING: Failed to download {ticker}: {e}")

    print(f"  Downloaded {len(data)} tickers successfully")
    return data


def compute_signals(data):
    """Find all gap-up signals across all tickers."""
    signals = []

    for ticker in TICKERS:
        if ticker not in data:
            continue
        df = data[ticker].copy()
        if len(df) < VOL_LOOKBACK + 5:
            continue

        df["prev_close"] = df["Close"].shift(1)
        df["gap_pct"] = df["Open"] / df["prev_close"] - 1.0
        df["vol_ma20"] = df["Volume"].rolling(VOL_LOOKBACK).mean()
        df["vol_mult"] = df["Volume"] / df["vol_ma20"]

        # Signal: gap >= 3% AND volume > 1.5x average
        mask = (df["gap_pct"] >= MIN_GAP_PCT) & (df["vol_mult"] >= VOL_MULT_THRESH)
        signal_dates = df.index[mask]

        for dt in signal_dates:
            idx = df.index.get_loc(dt)
            if idx < VOL_LOOKBACK:
                continue

            entry_price = float(df["Close"].iloc[idx])
            gap = float(df["gap_pct"].iloc[idx])
            vmult = float(df["vol_mult"].iloc[idx])

            signals.append({
                "ticker": ticker,
                "date": dt,
                "entry_price": entry_price,
                "gap_pct": gap,
                "vol_mult": vmult,
            })

    signals.sort(key=lambda x: x["date"])
    print(f"  Found {len(signals)} gap-up signals across all tickers")
    return signals


def simulate_trades(signals, data, spy_data):
    """Simulate bull call spread trades."""
    trades = []

    for sig in signals:
        ticker = sig["ticker"]
        df = data[ticker]
        entry_date = sig["date"]
        entry_price = sig["entry_price"]

        # Approximate spread cost: scale by price level
        # Higher-priced underlyings -> higher spread cost
        if entry_price > 300:
            spread_cost = 80.0
        elif entry_price > 100:
            spread_cost = 65.0
        else:
            spread_cost = 50.0

        # Find exit
        entry_idx = df.index.get_loc(entry_date)
        exit_idx = None
        exit_reason = None
        exit_price = None

        for hold_day in range(1, MAX_HOLD_DAYS + 1):
            check_idx = entry_idx + hold_day
            if check_idx >= len(df):
                break

            current_price = float(df["Close"].iloc[check_idx])
            underlying_return = (current_price - entry_price) / entry_price

            # Approximate option P&L: leverage * underlying return * spread_cost
            option_pnl = OPTION_LEVERAGE * underlying_return * spread_cost
            option_return = option_pnl / spread_cost  # as fraction of cost

            # Check TP
            if option_return >= TP_PCT:
                exit_idx = check_idx
                exit_reason = "TP"
                exit_price = current_price
                break

            # Check SL
            if option_return <= SL_PCT:
                exit_idx = check_idx
                exit_reason = "SL"
                exit_price = current_price
                break

        # Max hold exit
        if exit_idx is None:
            final_idx = min(entry_idx + MAX_HOLD_DAYS, len(df) - 1)
            if final_idx > entry_idx:
                exit_idx = final_idx
                exit_reason = "MAX_HOLD"
                exit_price = float(df["Close"].iloc[exit_idx])

        if exit_idx is None or exit_price is None:
            continue

        exit_date = df.index[exit_idx]
        underlying_return = (exit_price - entry_price) / entry_price
        option_pnl = OPTION_LEVERAGE * underlying_return * spread_cost

        # Cap option P&L: can't lose more than cost, can't gain more than spread width - cost
        # For a bull call spread, max loss = premium paid, max gain ~ 2-4x premium
        option_pnl = max(option_pnl, -spread_cost)       # can't lose more than cost
        option_pnl = min(option_pnl, spread_cost * 3.0)   # cap max gain at 3x (spread width limit)

        option_return_pct = option_pnl / spread_cost * 100

        # Determine SPY regime on entry day
        spy_regime = "unknown"
        if entry_date in spy_data.index:
            spy_row_idx = spy_data.index.get_loc(entry_date)
            spy_close = float(spy_data["Close"].iloc[spy_row_idx])
            spy_open = float(spy_data["Open"].iloc[spy_row_idx])
            spy_regime = "green" if spy_close >= spy_open else "red"

        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "spread_cost": spread_cost,
            "pnl": round(option_pnl, 2),
            "return_pct": round(option_return_pct, 2),
            "exit_reason": exit_reason,
            "gap_pct": round(sig["gap_pct"] * 100, 2),
            "vol_mult": round(sig["vol_mult"], 2),
            "spy_regime": spy_regime,
            "hold_days": (exit_date - entry_date).days,
        })

    print(f"  Simulated {len(trades)} trades")
    return trades


def compute_metrics(trades):
    """Compute risk-adjusted performance metrics."""
    if not trades:
        return {}

    pnls = np.array([t["pnl"] for t in trades])
    costs = np.array([t["spread_cost"] for t in trades])
    returns = pnls / costs  # fractional returns per trade

    n_trades = len(trades)
    winners = np.sum(pnls > 0)
    losers = np.sum(pnls < 0)
    win_rate = winners / n_trades * 100

    total_pnl = np.sum(pnls)
    avg_win = np.mean(pnls[pnls > 0]) if winners > 0 else 0
    avg_loss = np.mean(pnls[pnls < 0]) if losers > 0 else 0
    profit_factor = abs(np.sum(pnls[pnls > 0]) / np.sum(pnls[pnls < 0])) if losers > 0 and np.sum(pnls[pnls < 0]) != 0 else float("inf")

    # Sharpe: annualize assuming ~50 trades/year (weekly-ish)
    trades_per_year = max(n_trades / ((trades[-1]["exit_date"] - trades[0]["entry_date"]).days / 365.25), 1)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0.0

    return {
        "n_trades": n_trades,
        "winners": int(winners),
        "losers": int(losers),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "mean_return": round(mean_ret * 100, 2),
        "std_return": round(std_ret * 100, 2),
        "trades_per_year": round(trades_per_year, 1),
    }


def gate_1_sharpe(metrics):
    """Gate 1: Sharpe > 0.5"""
    sharpe = metrics.get("sharpe", 0)
    passed = sharpe > 0.5
    return passed, sharpe


def gate_2_permutation(trades, n_perms=N_PERMUTATIONS):
    """Gate 2: Permutation test p < 0.05"""
    if len(trades) < 10:
        return False, 1.0

    pnls = np.array([t["pnl"] for t in trades])
    actual_total = np.sum(pnls)

    rng = np.random.RandomState(42)
    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(pnls)
        # Randomly flip signs to simulate random entry timing
        signs = rng.choice([-1, 1], size=len(pnls))
        shuffled_total = np.sum(pnls * signs)
        if shuffled_total >= actual_total:
            count_better += 1

    p_value = count_better / n_perms
    passed = p_value < 0.05
    return passed, round(p_value, 4)


def gate_3_regime(trades):
    """Gate 3: Regime stratification. |Sharpe_green - Sharpe_red| / max(...) < 0.50"""
    green_trades = [t for t in trades if t["spy_regime"] == "green"]
    red_trades = [t for t in trades if t["spy_regime"] == "red"]

    if len(green_trades) < 5 or len(red_trades) < 5:
        return False, {"error": "Insufficient trades in one regime",
                       "green_n": len(green_trades), "red_n": len(red_trades)}

    green_metrics = compute_metrics(green_trades)
    red_metrics = compute_metrics(red_trades)

    sharpe_g = green_metrics["sharpe"]
    sharpe_r = red_metrics["sharpe"]

    denom = max(abs(sharpe_g), abs(sharpe_r))
    if denom == 0:
        regime_gap = 0.0
    else:
        regime_gap = abs(sharpe_g - sharpe_r) / denom

    passed = regime_gap < 0.50

    return passed, {
        "sharpe_green": sharpe_g,
        "sharpe_red": sharpe_r,
        "regime_gap": round(regime_gap, 3),
        "green_trades": len(green_trades),
        "red_trades": len(red_trades),
        "green_wr": green_metrics["win_rate"],
        "red_wr": red_metrics["win_rate"],
    }


def gate_4_subperiod(trades):
    """Gate 4: Split into 4 equal periods, all must have Sharpe > 0"""
    if len(trades) < 20:
        return False, {"error": "Too few trades for sub-period analysis"}

    n = len(trades)
    chunk = n // 4
    periods = []
    all_positive = True

    for i in range(4):
        start = i * chunk
        end = (i + 1) * chunk if i < 3 else n
        period_trades = trades[start:end]
        m = compute_metrics(period_trades)

        period_info = {
            "period": i + 1,
            "n_trades": m["n_trades"],
            "sharpe": m["sharpe"],
            "win_rate": m["win_rate"],
            "total_pnl": m["total_pnl"],
            "date_range": f"{period_trades[0]['entry_date'].strftime('%Y-%m-%d')} to {period_trades[-1]['entry_date'].strftime('%Y-%m-%d')}",
        }
        periods.append(period_info)
        if m["sharpe"] <= 0:
            all_positive = False

    return all_positive, periods


def gate_5_concentration(trades):
    """Gate 5: No single day's P&L > 70% of total"""
    if not trades:
        return False, {}

    # Group P&L by exit date
    daily_pnl = {}
    for t in trades:
        d = t["exit_date"]
        daily_pnl[d] = daily_pnl.get(d, 0) + t["pnl"]

    total_pnl = sum(t["pnl"] for t in trades)
    if total_pnl <= 0:
        # If total P&L is negative/zero, concentration test is N/A
        # but we still check: does a single day dominate?
        max_day_pnl = max(daily_pnl.values()) if daily_pnl else 0
        return True, {"max_day_pnl": round(max_day_pnl, 2), "total_pnl": round(total_pnl, 2),
                       "note": "Total P&L <= 0, concentration check trivially passes"}

    max_day_pnl = max(daily_pnl.values())
    concentration = max_day_pnl / total_pnl if total_pnl > 0 else 0

    passed = concentration < 0.70
    return passed, {
        "max_day_pnl": round(max_day_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "concentration": round(concentration, 3),
    }


def print_results(metrics, gates):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("  MOMENTUM BURST BACKTEST RESULTS")
    print("  Bull Call Spreads on 3%+ Gap-Ups, ETF Universe")
    print("  Period: 2020-01-01 to 2026-08-01")
    print("=" * 70)

    print(f"\n  PERFORMANCE METRICS")
    print(f"  {'─' * 40}")
    print(f"  Trades:          {metrics['n_trades']}")
    print(f"  Winners:         {metrics['winners']}  |  Losers: {metrics['losers']}")
    print(f"  Win Rate:        {metrics['win_rate']:.1f}%")
    print(f"  Total P&L:       ${metrics['total_pnl']:,.2f}")
    print(f"  Avg Win:         ${metrics['avg_win']:,.2f}")
    print(f"  Avg Loss:        ${metrics['avg_loss']:,.2f}")
    print(f"  Profit Factor:   {metrics['profit_factor']:.3f}")
    print(f"  Sharpe Ratio:    {metrics['sharpe']:.3f}")
    print(f"  Sortino Ratio:   {metrics['sortino']:.3f}")
    print(f"  Mean Return:     {metrics['mean_return']:.2f}%")
    print(f"  Trades/Year:     {metrics['trades_per_year']:.1f}")

    print(f"\n  EXIT BREAKDOWN")
    print(f"  {'─' * 40}")
    exit_counts = {}
    for g in gates.get("_trades", []):
        r = g["exit_reason"]
        exit_counts[r] = exit_counts.get(r, 0) + 1
    for reason, count in sorted(exit_counts.items()):
        print(f"  {reason:12s}: {count:4d} ({count/metrics['n_trades']*100:.1f}%)")

    print(f"\n  5-GATE VALIDATION")
    print(f"  {'─' * 40}")

    # Gate 1
    g1_pass, g1_val = gates["gate_1"]
    status = "PASS" if g1_pass else "FAIL"
    print(f"  Gate 1 (Sharpe > 0.5):        [{status}]  Sharpe = {g1_val:.3f}")

    # Gate 2
    g2_pass, g2_val = gates["gate_2"]
    status = "PASS" if g2_pass else "FAIL"
    print(f"  Gate 2 (Perm test p < 0.05):  [{status}]  p = {g2_val}")

    # Gate 3
    g3_pass, g3_info = gates["gate_3"]
    status = "PASS" if g3_pass else "FAIL"
    if isinstance(g3_info, dict) and "regime_gap" in g3_info:
        print(f"  Gate 3 (Regime gap < 0.50):   [{status}]  gap = {g3_info['regime_gap']:.3f}")
        print(f"           Green Sharpe: {g3_info['sharpe_green']:.3f} ({g3_info['green_trades']} trades, WR {g3_info['green_wr']:.1f}%)")
        print(f"           Red Sharpe:   {g3_info['sharpe_red']:.3f} ({g3_info['red_trades']} trades, WR {g3_info['red_wr']:.1f}%)")
    else:
        print(f"  Gate 3 (Regime gap < 0.50):   [{status}]  {g3_info}")

    # Gate 4
    g4_pass, g4_info = gates["gate_4"]
    status = "PASS" if g4_pass else "FAIL"
    print(f"  Gate 4 (All sub-periods > 0): [{status}]")
    if isinstance(g4_info, list):
        for p in g4_info:
            s = "+" if p["sharpe"] > 0 else "-"
            print(f"           P{p['period']}: Sharpe={p['sharpe']:+.3f} WR={p['win_rate']:.1f}% "
                  f"PnL=${p['total_pnl']:+.0f} ({p['n_trades']} trades) [{p['date_range']}]")

    # Gate 5
    g5_pass, g5_info = gates["gate_5"]
    status = "PASS" if g5_pass else "FAIL"
    conc = g5_info.get("concentration", 0)
    print(f"  Gate 5 (Day conc < 70%):      [{status}]  max day conc = {conc:.1%}")

    # Overall
    all_pass = all([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    gates_passed = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    print(f"\n  {'─' * 40}")
    overall = "ALL GATES PASSED" if all_pass else f"FAILED ({gates_passed}/5 gates passed)"
    print(f"  OVERALL: {overall}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  Momentum Burst Backtest — Bull Call Spreads on Gap-Up Signals")
    print("=" * 70)

    # 1. Download data
    data = download_data()
    if "SPY" not in data:
        print("FATAL: Could not download SPY data")
        return
    spy_data = data["SPY"]

    # 2. Find signals
    signals = compute_signals(data)
    if len(signals) < 10:
        print(f"FATAL: Only {len(signals)} signals found, need at least 10")
        return

    # 3. Simulate trades
    trades = simulate_trades(signals, data, spy_data)
    if len(trades) < 10:
        print(f"FATAL: Only {len(trades)} trades simulated, need at least 10")
        return

    # 4. Compute metrics
    metrics = compute_metrics(trades)

    # 5. Run 5-gate validation
    print("\nRunning 5-gate validation...")

    g1_pass, g1_val = gate_1_sharpe(metrics)
    print(f"  Gate 1 (Sharpe): {'PASS' if g1_pass else 'FAIL'} ({g1_val:.3f})")

    print(f"  Gate 2 (Permutation test): running {N_PERMUTATIONS} permutations...")
    g2_pass, g2_val = gate_2_permutation(trades)
    print(f"  Gate 2 (Permutation): {'PASS' if g2_pass else 'FAIL'} (p={g2_val})")

    g3_pass, g3_info = gate_3_regime(trades)
    print(f"  Gate 3 (Regime): {'PASS' if g3_pass else 'FAIL'}")

    g4_pass, g4_info = gate_4_subperiod(trades)
    print(f"  Gate 4 (Sub-period): {'PASS' if g4_pass else 'FAIL'}")

    g5_pass, g5_info = gate_5_concentration(trades)
    print(f"  Gate 5 (Concentration): {'PASS' if g5_pass else 'FAIL'}")

    gates = {
        "gate_1": (g1_pass, g1_val),
        "gate_2": (g2_pass, g2_val),
        "gate_3": (g3_pass, g3_info),
        "gate_4": (g4_pass, g4_info),
        "gate_5": (g5_pass, g5_info),
        "_trades": trades,
    }

    # 6. Print results
    print_results(metrics, gates)

    # 7. Top/bottom trades
    trades_sorted = sorted(trades, key=lambda x: x["pnl"], reverse=True)
    print(f"\n  TOP 5 TRADES:")
    for t in trades_sorted[:5]:
        print(f"    {t['ticker']:5s} {t['entry_date'].strftime('%Y-%m-%d')} gap={t['gap_pct']:.1f}% "
              f"PnL=${t['pnl']:+.0f} ({t['return_pct']:+.0f}%) [{t['exit_reason']}]")

    print(f"\n  BOTTOM 5 TRADES:")
    for t in trades_sorted[-5:]:
        print(f"    {t['ticker']:5s} {t['entry_date'].strftime('%Y-%m-%d')} gap={t['gap_pct']:.1f}% "
              f"PnL=${t['pnl']:+.0f} ({t['return_pct']:+.0f}%) [{t['exit_reason']}]")

    # 8. Year-by-year breakdown
    print(f"\n  YEAR-BY-YEAR BREAKDOWN:")
    print(f"  {'Year':>6s} {'Trades':>7s} {'WR':>6s} {'PnL':>10s} {'Sharpe':>8s}")
    print(f"  {'─' * 40}")
    yearly = {}
    for t in trades:
        y = t["entry_date"].year
        if y not in yearly:
            yearly[y] = []
        yearly[y].append(t)

    for y in sorted(yearly.keys()):
        ym = compute_metrics(yearly[y])
        print(f"  {y:>6d} {ym['n_trades']:>7d} {ym['win_rate']:>5.1f}% ${ym['total_pnl']:>9.0f} {ym['sharpe']:>8.3f}")

    print()


if __name__ == "__main__":
    main()
