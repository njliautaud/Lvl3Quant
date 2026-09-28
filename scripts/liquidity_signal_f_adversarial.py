#!/usr/bin/env python3
"""
Adversarial Validation: Liquidity Signal F — Bid-Ask Spread Proxy
6 adversarial tests against the baseline strategy.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START_LOOKBACK = "2021-06-01"
START_DATE = "2022-01-01"
END_DATE = "2026-07-31"
CAPITAL = 669.0
MAX_POS_SIZE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
DIP_THRESHOLD = 5  # % below 20-day high
RSI_PERIOD = 14
RSI_THRESHOLD = 40
SPREAD_LOOKBACK = 60

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/liquidity_signal_f_adversarial.json"


# ── HELPERS ─────────────────────────────────────────────────────────────────
def fetch_data(tickers, start, end):
    """Download OHLCV data for all tickers."""
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                data[t] = df
        except Exception:
            pass
    return data


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_signals(data, spread_lookback=SPREAD_LOOKBACK, dip_threshold=DIP_THRESHOLD,
                    rsi_threshold=RSI_THRESHOLD, invert_spread=False):
    """Compute entry signals for each ticker. Returns dict of {ticker: Series of signal dates}."""
    signals = {}
    for ticker, df in data.items():
        h, l, c = df["High"], df["Low"], df["Close"]
        spread_proxy = (h - l) / ((h + l) / 2)
        spread_avg = spread_proxy.rolling(spread_lookback).mean()

        if invert_spread:
            spread_cond = spread_proxy > spread_avg  # INVERSE
        else:
            spread_cond = spread_proxy < spread_avg  # NORMAL

        high_20 = c.rolling(20).max()
        dip_cond = c < high_20 * (1 - dip_threshold / 100)

        rsi = compute_rsi(c, RSI_PERIOD)
        rsi_cond = rsi < rsi_threshold

        mask = spread_cond & dip_cond & rsi_cond
        # Only consider dates in the trading period
        mask = mask.loc[START_DATE:]
        sig_dates = mask[mask].index
        signals[ticker] = sig_dates
    return signals


def run_backtest(data, signals, capital=CAPITAL, max_pos=MAX_POS_SIZE,
                 max_concurrent=MAX_CONCURRENT, hold_days=HOLD_DAYS,
                 slippage_bps=SLIPPAGE_BPS, tickers_to_exclude=None):
    """Simple backtest: fixed hold period, max concurrent positions."""
    if tickers_to_exclude is None:
        tickers_to_exclude = set()

    trades = []
    # Collect all (date, ticker) entry candidates
    candidates = []
    for ticker, dates in signals.items():
        if ticker in tickers_to_exclude:
            continue
        for d in dates:
            candidates.append((d, ticker))
    candidates.sort(key=lambda x: x[0])

    active_positions = []  # list of (exit_date, ticker)
    equity = capital
    equity_curve = []

    # Build a date index from data
    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp(START_DATE)]

    # Track daily equity
    daily_pnl = {}
    for d in all_dates:
        daily_pnl[d] = 0.0

    for entry_date, ticker in candidates:
        if ticker in tickers_to_exclude:
            continue
        # Clean expired positions
        active_positions = [(ed, t) for ed, t in active_positions if ed > entry_date]
        if len(active_positions) >= max_concurrent:
            continue

        df = data[ticker]
        if entry_date not in df.index:
            continue

        entry_idx = df.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_date = df.index[exit_idx]

        entry_price = df["Close"].iloc[entry_idx]
        exit_price = df["Close"].iloc[exit_idx]

        shares = int(max_pos / entry_price) if entry_price > 0 else 0
        if shares == 0:
            continue

        cost_entry = entry_price * shares * (slippage_bps / 10000)
        cost_exit = exit_price * shares * (slippage_bps / 10000)
        pnl = (exit_price - entry_price) * shares - cost_entry - cost_exit

        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "shares": shares,
            "pnl": round(pnl, 4),
        })

        # Distribute PnL to exit date for daily tracking
        if exit_date in daily_pnl:
            daily_pnl[exit_date] += pnl

        active_positions.append((exit_date, ticker))

    # Compute metrics
    if not trades:
        return {"sharpe": 0, "sortino": 0, "wr": 0, "pf": 0, "mdd": 0,
                "n_trades": 0, "total_pnl": 0, "trades": [], "daily_pnl": {},
                "ticker_pnl": {}}

    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) if pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Daily returns for Sharpe/Sortino
    daily_rets = pd.Series(daily_pnl).sort_index()
    daily_rets = daily_rets / capital  # as fraction of capital

    mean_r = daily_rets.mean()
    std_r = daily_rets.std()
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0

    downside = daily_rets[daily_rets < 0].std()
    sortino = (mean_r / downside * np.sqrt(252)) if downside > 0 else 0

    # MDD
    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min() * 100  # as percentage

    # PnL by ticker
    ticker_pnl = {}
    for t in trades:
        ticker_pnl[t["ticker"]] = ticker_pnl.get(t["ticker"], 0) + t["pnl"]

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "wr": round(wr * 100, 2),
        "pf": round(pf, 4),
        "mdd": round(mdd, 2),
        "n_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "trades": trades,
        "daily_pnl": {str(k.date()): round(v, 4) for k, v in daily_pnl.items()},
        "ticker_pnl": {k: round(v, 2) for k, v in sorted(ticker_pnl.items(), key=lambda x: -x[1])},
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("Fetching data...")
    data = fetch_data(TICKERS, START_LOOKBACK, END_DATE)
    print(f"  Got data for {len(data)} tickers")

    # ── BASELINE ────────────────────────────────────────────────────────────
    print("\n=== BASELINE ===")
    signals = compute_signals(data)
    baseline = run_backtest(data, signals)
    print(f"  Sharpe={baseline['sharpe']}, Sortino={baseline['sortino']}, "
          f"WR={baseline['wr']}%, PF={baseline['pf']}, MDD={baseline['mdd']}%, "
          f"Trades={baseline['n_trades']}, PnL=${baseline['total_pnl']}")

    results = {
        "strategy": "Liquidity Signal F — Bid-Ask Spread Proxy",
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {
            "sharpe": baseline["sharpe"],
            "sortino": baseline["sortino"],
            "win_rate": baseline["wr"],
            "profit_factor": baseline["pf"],
            "max_drawdown_pct": baseline["mdd"],
            "n_trades": baseline["n_trades"],
            "total_pnl": baseline["total_pnl"],
        },
        "tests": {},
    }

    # ── TEST 1: INVERSE SIGNAL ──────────────────────────────────────────────
    print("\n=== TEST 1: Inverse Signal ===")
    inv_signals = compute_signals(data, invert_spread=True)
    inv_result = run_backtest(data, inv_signals)
    inv_ratio = inv_result["sharpe"] / baseline["sharpe"] if baseline["sharpe"] != 0 else float("inf")
    t1_pass = inv_result["sharpe"] < 0.5 * baseline["sharpe"]
    print(f"  Inverse Sharpe={inv_result['sharpe']}, Ratio={inv_ratio:.3f}, Pass={t1_pass}")
    results["tests"]["1_inverse_signal"] = {
        "description": "Buy when spread ABOVE avg (inverse). Pass if inverse Sharpe < 50% of baseline.",
        "baseline_sharpe": baseline["sharpe"],
        "inverse_sharpe": inv_result["sharpe"],
        "inverse_sortino": inv_result["sortino"],
        "inverse_wr": inv_result["wr"],
        "inverse_n_trades": inv_result["n_trades"],
        "ratio": round(inv_ratio, 4),
        "pass": t1_pass,
    }

    # ── TEST 2: RANDOM TIMING PERCENTILE ────────────────────────────────────
    print("\n=== TEST 2: Random Timing Percentile (1000 perms) ===")
    n_perms = 1000
    # For each perm, shuffle entry dates across all tickers
    all_entry_dates = []
    for ticker, dates in signals.items():
        for d in dates:
            all_entry_dates.append((d, ticker))

    # Get all trading dates in period
    trading_dates = sorted(set().union(*[set(df.loc[START_DATE:].index) for df in data.values()]))

    perm_sharpes = []
    for i in range(n_perms):
        # Random signals: same number of entries per ticker, random dates
        rand_signals = {}
        for ticker, dates in signals.items():
            n_entries = len(dates)
            if n_entries == 0:
                rand_signals[ticker] = pd.DatetimeIndex([])
                continue
            # Pick random dates from this ticker's available dates
            ticker_dates = data[ticker].loc[START_DATE:].index
            if len(ticker_dates) == 0:
                rand_signals[ticker] = pd.DatetimeIndex([])
                continue
            rand_idx = np.random.choice(len(ticker_dates), size=min(n_entries, len(ticker_dates)), replace=False)
            rand_signals[ticker] = ticker_dates[rand_idx]
        perm_result = run_backtest(data, rand_signals)
        perm_sharpes.append(perm_result["sharpe"])
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{n_perms} permutations done")

    perm_sharpes = np.array(perm_sharpes)
    percentile = np.mean(perm_sharpes < baseline["sharpe"]) * 100
    p_value = 1 - percentile / 100
    t2_pass = p_value < 0.05
    print(f"  Percentile={percentile:.1f}%, p-value={p_value:.4f}, Pass={t2_pass}")
    results["tests"]["2_random_timing"] = {
        "description": "1000 random-date permutations. Pass if real Sharpe p < 0.05.",
        "baseline_sharpe": baseline["sharpe"],
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 4),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 4),
        "perm_median_sharpe": round(float(np.median(perm_sharpes)), 4),
        "percentile": round(percentile, 2),
        "p_value": round(p_value, 4),
        "pass": t2_pass,
    }

    # ── TEST 3: SUB-PERIOD STABILITY ────────────────────────────────────────
    print("\n=== TEST 3: Sub-Period Stability (4 periods) ===")
    period_start = pd.Timestamp(START_DATE)
    period_end = pd.Timestamp(END_DATE)
    total_days = (period_end - period_start).days
    quarter = total_days // 4
    sub_periods = []
    for i in range(4):
        sp_start = period_start + timedelta(days=i * quarter)
        sp_end = period_start + timedelta(days=(i + 1) * quarter) if i < 3 else period_end
        sub_periods.append((sp_start, sp_end))

    sub_results = []
    all_positive = True
    for i, (sp_start, sp_end) in enumerate(sub_periods):
        # Filter signals to sub-period
        sub_signals = {}
        for ticker, dates in signals.items():
            mask = (dates >= sp_start) & (dates <= sp_end)
            sub_signals[ticker] = dates[mask]
        # Need to temporarily change START_DATE context — just filter trades after
        sr = run_backtest(data, sub_signals)
        sub_results.append({
            "period": f"{sp_start.date()} to {sp_end.date()}",
            "sharpe": sr["sharpe"],
            "sortino": sr["sortino"],
            "wr": sr["wr"],
            "n_trades": sr["n_trades"],
            "total_pnl": sr["total_pnl"],
        })
        if sr["sharpe"] <= 0:
            all_positive = False
        print(f"  Period {i+1} ({sp_start.date()} to {sp_end.date()}): Sharpe={sr['sharpe']}, Trades={sr['n_trades']}")

    t3_pass = all_positive
    print(f"  All positive Sharpe: {t3_pass}")
    results["tests"]["3_sub_period_stability"] = {
        "description": "Split into 4 equal sub-periods. Pass if ALL have positive Sharpe.",
        "sub_periods": sub_results,
        "all_positive_sharpe": all_positive,
        "pass": t3_pass,
    }

    # ── TEST 4: REMOVE TOP-3 TICKERS ───────────────────────────────────────
    print("\n=== TEST 4: Remove Top-3 Tickers ===")
    ticker_pnl = baseline["ticker_pnl"]
    top3 = list(ticker_pnl.keys())[:3]
    print(f"  Top 3 by PnL: {top3}")
    reduced = run_backtest(data, signals, tickers_to_exclude=set(top3))
    drop_pct = (1 - reduced["sharpe"] / baseline["sharpe"]) * 100 if baseline["sharpe"] != 0 else 100
    t4_pass = drop_pct < 50
    print(f"  Reduced Sharpe={reduced['sharpe']}, Drop={drop_pct:.1f}%, Pass={t4_pass}")
    results["tests"]["4_remove_top3_tickers"] = {
        "description": "Remove top 3 tickers by PnL. Pass if Sharpe drop < 50%.",
        "top3_tickers": top3,
        "top3_pnl": {t: ticker_pnl[t] for t in top3},
        "baseline_sharpe": baseline["sharpe"],
        "reduced_sharpe": reduced["sharpe"],
        "reduced_n_trades": reduced["n_trades"],
        "sharpe_drop_pct": round(drop_pct, 2),
        "pass": t4_pass,
    }

    # ── TEST 5: PARAMETER SENSITIVITY GRID ──────────────────────────────────
    print("\n=== TEST 5: Parameter Sensitivity Grid ===")
    dip_vals = [3, 5, 7, 10]
    rsi_vals = [30, 35, 40, 45, 50]
    spread_vals = [30, 45, 60, 90]
    hold_vals = [5, 7, 10, 15]

    total_combos = len(dip_vals) * len(rsi_vals) * len(spread_vals) * len(hold_vals)
    print(f"  Running {total_combos} parameter combinations...")

    grid_results = []
    count = 0
    above_threshold = 0
    for dip in dip_vals:
        for rsi_t in rsi_vals:
            for sp_lb in spread_vals:
                sig = compute_signals(data, spread_lookback=sp_lb, dip_threshold=dip, rsi_threshold=rsi_t)
                for hd in hold_vals:
                    res = run_backtest(data, sig, hold_days=hd)
                    if res["sharpe"] > 0.3:
                        above_threshold += 1
                    grid_results.append({
                        "dip": dip, "rsi": rsi_t, "spread_lb": sp_lb, "hold": hd,
                        "sharpe": res["sharpe"], "n_trades": res["n_trades"],
                    })
                    count += 1
                    if count % 80 == 0:
                        print(f"    ... {count}/{total_combos} done")

    pct_above = above_threshold / total_combos * 100
    t5_pass = pct_above > 50
    # Find best/worst
    grid_results.sort(key=lambda x: -x["sharpe"])
    print(f"  {above_threshold}/{total_combos} ({pct_above:.1f}%) have Sharpe > 0.3, Pass={t5_pass}")
    print(f"  Best: {grid_results[0]}")
    print(f"  Worst: {grid_results[-1]}")
    results["tests"]["5_parameter_sensitivity"] = {
        "description": "Grid over dip/rsi/spread_lb/hold. Pass if >50% combos have Sharpe > 0.3.",
        "total_combinations": total_combos,
        "above_threshold": above_threshold,
        "pct_above_0_3": round(pct_above, 2),
        "best_combo": grid_results[0],
        "worst_combo": grid_results[-1],
        "top5": grid_results[:5],
        "median_sharpe": round(float(np.median([g["sharpe"] for g in grid_results])), 4),
        "mean_sharpe": round(float(np.mean([g["sharpe"] for g in grid_results])), 4),
        "pass": t5_pass,
    }

    # ── TEST 6: COST SENSITIVITY ───────────────────────────────────────────
    print("\n=== TEST 6: Cost Sensitivity ===")
    slippage_levels = [0, 2, 5, 10, 20, 50]
    cost_results = []
    for slip in slippage_levels:
        res = run_backtest(data, signals, slippage_bps=slip)
        cost_results.append({
            "slippage_bps": slip,
            "sharpe": res["sharpe"],
            "total_pnl": res["total_pnl"],
            "n_trades": res["n_trades"],
        })
        print(f"  {slip}bps: Sharpe={res['sharpe']}, PnL=${res['total_pnl']}")

    # Estimate breakeven slippage (linear interpolation where Sharpe crosses 0)
    breakeven_bps = None
    for i in range(len(cost_results) - 1):
        s1 = cost_results[i]["sharpe"]
        s2 = cost_results[i + 1]["sharpe"]
        b1 = cost_results[i]["slippage_bps"]
        b2 = cost_results[i + 1]["slippage_bps"]
        if s1 > 0 >= s2:
            # Linear interpolation
            breakeven_bps = b1 + (b2 - b1) * s1 / (s1 - s2)
            break
    if breakeven_bps is None and cost_results[-1]["sharpe"] > 0:
        breakeven_bps = float("inf")  # profitable even at highest slippage
    elif breakeven_bps is None:
        breakeven_bps = 0

    t6_pass = breakeven_bps > 20
    breakeven_display = round(breakeven_bps, 1) if breakeven_bps != float("inf") else "inf"
    print(f"  Breakeven slippage: {breakeven_display} bps, Pass={t6_pass}")
    results["tests"]["6_cost_sensitivity"] = {
        "description": "Run at various slippage levels. Pass if breakeven > 20 bps.",
        "cost_curve": cost_results,
        "breakeven_bps": breakeven_display,
        "pass": t6_pass,
    }

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    test_passes = [v["pass"] for v in results["tests"].values()]
    tests_passed = sum(test_passes)
    tests_total = len(test_passes)
    overall = "PASS" if tests_passed >= 5 else ("MARGINAL" if tests_passed >= 4 else "FAIL")

    results["summary"] = {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "overall_verdict": overall,
        "test_results": {k: v["pass"] for k, v in results["tests"].items()},
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY: {tests_passed}/{tests_total} tests passed — {overall}")
    for k, v in results["tests"].items():
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  {k}: {status}")
    print(f"{'='*60}")

    # Remove trade-level detail to keep JSON manageable
    if "trades" in baseline:
        del baseline["trades"]
    if "daily_pnl" in baseline:
        del baseline["daily_pnl"]

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
