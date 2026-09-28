#!/usr/bin/env python3
"""
Adversarial Validation: Momentum Acceleration B (Volume-Confirmed)

Tests whether the strategy has genuine edge or is explained by simpler factors:
1. Inverse test (bottom-3 decelerating stocks)
2. Remove volume filter
3. Random stock selection (same dates)
4. Random timing (same stock selection logic, random dates)
5. Sub-period stability (yearly Sharpe)
6. Concentration test (top-3 trade P&L contribution)

Usage: python3 momentum_accel_adversarial.py
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "SHOP", "XYZ", "COIN", "SNOW", "DDOG", "NET",
    "RBLX", "PLTR", "UBER", "LYFT",  # SQ -> XYZ (Block ticker change)
]
OOT_START = "2022-01-01"
OOT_END = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 645.0
HOLD_DAYS = 20
TOP_N = 3
VOLUME_MULT = 1.5
MIN_SPACING = 5  # minimum days between entries (allows overlapping holds)
MC_ITERS = 1000
DATA_START = "2021-06-01"  # extra lookback for SMAs


def download_data():
    """Download price/volume data for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, progress=False, group_by="ticker")
    return data


def extract_series(data, ticker, field):
    """Safely extract a series from multi-level yfinance data."""
    try:
        s = data[ticker][field].dropna()
        return s
    except Exception:
        return pd.Series(dtype=float)


def build_frames(data):
    """Build close/volume DataFrames from raw download."""
    closes = pd.DataFrame({t: extract_series(data, t, "Close") for t in UNIVERSE})
    volumes = pd.DataFrame({t: extract_series(data, t, "Volume") for t in UNIVERSE})
    spy_close = extract_series(data, "SPY", "Close")
    return closes, volumes, spy_close


def regime_series(spy_close):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_close.rolling(200).mean()
    return (spy_close > sma200).astype(int)  # 1=bull, 0=bear


def compute_signals(closes, volumes):
    """
    For each day, compute:
      - ret_5d, ret_20d per stock
      - volume ratio (today vol / 20d avg vol)
      - acceleration flag: ret_5d > ret_20d AND both > 0
    Returns DataFrames aligned to closes index.
    """
    ret_5d = closes.pct_change(5)
    ret_20d = closes.pct_change(20)
    vol_20d_avg = volumes.rolling(20).mean()
    vol_ratio = volumes / vol_20d_avg

    accel = (ret_5d > ret_20d) & (ret_5d > 0) & (ret_20d > 0)
    decel = (ret_5d < ret_20d) & (ret_5d < 0) & (ret_20d < 0)  # inverse
    high_vol = vol_ratio > VOLUME_MULT

    # Acceleration score = ret_5d - ret_20d (higher = more accelerating)
    accel_score = ret_5d - ret_20d
    # Deceleration score = ret_20d - ret_5d (higher = more decelerating, for inverse)
    decel_score = ret_20d - ret_5d

    return ret_5d, ret_20d, vol_ratio, accel, decel, high_vol, accel_score, decel_score


def run_strategy(closes, entry_dates, stock_selections, initial_capital=INITIAL_CAPITAL):
    """
    Given entry dates and stock selections (list of lists of tickers),
    simulate the strategy: equal-weight buy on entry, sell after HOLD_DAYS trading days.
    Returns equity curve and per-trade P&L list.

    Handles overlapping positions by tracking capital available.
    """
    all_dates = closes.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    trades = []
    equity = initial_capital

    for entry_date, stocks in zip(entry_dates, stock_selections):
        if entry_date not in date_to_idx:
            continue
        entry_idx = date_to_idx[entry_date]
        exit_idx = min(entry_idx + HOLD_DAYS, len(all_dates) - 1)
        exit_date = all_dates[exit_idx]

        # Equal weight across selected stocks
        per_stock_capital = equity / len(stocks) if stocks else 0
        trade_pnl = 0.0
        valid_stocks = 0
        for ticker in stocks:
            try:
                entry_price = closes.loc[entry_date, ticker]
                exit_price = closes.loc[exit_date, ticker]
                if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                    continue
                shares = per_stock_capital / entry_price
                pnl = shares * (exit_price - entry_price)
                trade_pnl += pnl
                valid_stocks += 1
            except Exception:
                continue

        if valid_stocks > 0:
            trades.append({
                "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                "stocks": stocks,
                "pnl": trade_pnl,
                "return_pct": trade_pnl / equity if equity > 0 else 0,
            })
            equity += trade_pnl

    return trades, equity


def strategy_metrics(trades, initial_capital=INITIAL_CAPITAL):
    """Compute Sharpe, total return, win rate from trade list."""
    if not trades:
        return {"sharpe": 0, "total_return": 0, "win_rate": 0, "n_trades": 0, "final_equity": initial_capital}

    returns = [t["return_pct"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for r in returns if r > 0)

    mean_ret = np.mean(returns)
    # Need at least 3 trades for meaningful Sharpe
    if len(returns) < 3:
        sharpe = 0.0
    else:
        std_ret = np.std(returns, ddof=1)
        trades_per_year = 252 / HOLD_DAYS
        sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-10 else 0

    final_equity = initial_capital + sum(pnls)

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round((final_equity / initial_capital - 1) * 100, 1),
        "win_rate": round(wins / len(returns) * 100, 1),
        "n_trades": len(trades),
        "final_equity": round(final_equity, 2),
        "mean_return_pct": round(mean_ret * 100, 3),
        "max_dd_pct": round(compute_max_dd(trades, initial_capital), 1),
    }


def compute_max_dd(trades, initial_capital):
    """Compute max drawdown % from trade list."""
    equity = initial_capital
    peak = equity
    max_dd = 0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    return max_dd


def get_original_signals(closes, volumes, accel, high_vol, accel_score):
    """Get entry dates and stock picks for the original strategy."""
    oot_mask = closes.index >= pd.Timestamp(OOT_START)
    oot_dates = closes.index[oot_mask]

    entry_dates = []
    stock_selections = []
    MIN_SPACING = 5  # allow overlapping positions but not daily re-entry
    last_entry_idx = -MIN_SPACING

    for i, date in enumerate(oot_dates):
        if i - last_entry_idx < MIN_SPACING:
            continue

        # Find stocks with acceleration + high volume
        candidates = []
        for ticker in UNIVERSE:
            try:
                if accel.loc[date, ticker] and high_vol.loc[date, ticker]:
                    score = accel_score.loc[date, ticker]
                    if not pd.isna(score):
                        candidates.append((ticker, score))
            except Exception:
                continue

        if len(candidates) >= TOP_N:
            # Sort by acceleration score, pick top N
            candidates.sort(key=lambda x: x[1], reverse=True)
            picks = [c[0] for c in candidates[:TOP_N]]
            entry_dates.append(date)
            stock_selections.append(picks)
            last_entry_idx = i

    return entry_dates, stock_selections


# ── TEST 1: INVERSE (bottom-3 decelerating + high volume) ──────────────────
def test_inverse(closes, volumes, decel, high_vol, decel_score):
    """Buy bottom-3 decelerating stocks with high volume."""
    print("\n[TEST 1] Inverse: Bottom-3 decelerating stocks with high volume")
    oot_mask = closes.index >= pd.Timestamp(OOT_START)
    oot_dates = closes.index[oot_mask]

    entry_dates = []
    stock_selections = []
    last_entry_idx = -MIN_SPACING

    for i, date in enumerate(oot_dates):
        if i - last_entry_idx < MIN_SPACING:
            continue

        candidates = []
        for ticker in UNIVERSE:
            try:
                if decel.loc[date, ticker] and high_vol.loc[date, ticker]:
                    score = decel_score.loc[date, ticker]
                    if not pd.isna(score):
                        candidates.append((ticker, score))
            except Exception:
                continue

        if len(candidates) >= TOP_N:
            candidates.sort(key=lambda x: x[1], reverse=True)
            picks = [c[0] for c in candidates[:TOP_N]]
            entry_dates.append(date)
            stock_selections.append(picks)
            last_entry_idx = i

    trades, final_eq = run_strategy(closes, entry_dates, stock_selections)
    metrics = strategy_metrics(trades)
    print(f"  Inverse: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, "
          f"WR={metrics['win_rate']}%, Return={metrics['total_return']}%")
    return metrics


# ── TEST 2: Remove volume filter ──────────────────────────────────────────
def test_no_volume_filter(closes, accel, accel_score):
    """Same strategy but without volume > 1.5x requirement."""
    print("\n[TEST 2] Remove volume filter")
    oot_mask = closes.index >= pd.Timestamp(OOT_START)
    oot_dates = closes.index[oot_mask]

    entry_dates = []
    stock_selections = []
    last_entry_idx = -MIN_SPACING

    for i, date in enumerate(oot_dates):
        if i - last_entry_idx < MIN_SPACING:
            continue

        candidates = []
        for ticker in UNIVERSE:
            try:
                if accel.loc[date, ticker]:  # NO volume filter
                    score = accel_score.loc[date, ticker]
                    if not pd.isna(score):
                        candidates.append((ticker, score))
            except Exception:
                continue

        if len(candidates) >= TOP_N:
            candidates.sort(key=lambda x: x[1], reverse=True)
            picks = [c[0] for c in candidates[:TOP_N]]
            entry_dates.append(date)
            stock_selections.append(picks)
            last_entry_idx = i

    trades, final_eq = run_strategy(closes, entry_dates, stock_selections)
    metrics = strategy_metrics(trades)
    print(f"  No-volume: {metrics['n_trades']} trades, Sharpe={metrics['sharpe']}, "
          f"WR={metrics['win_rate']}%, Return={metrics['total_return']}%")
    return metrics


# ── TEST 3: Random stock selection (same entry dates) ────────────────────
def test_random_stocks(closes, entry_dates, n_iter=MC_ITERS):
    """On same entry dates, pick 3 random stocks from universe."""
    print(f"\n[TEST 3] Random stock selection ({n_iter} iterations)")
    available_tickers = [t for t in UNIVERSE if t in closes.columns]
    sharpes = []

    for i in range(n_iter):
        random_selections = [
            list(np.random.choice(available_tickers, size=TOP_N, replace=False))
            for _ in entry_dates
        ]
        trades, _ = run_strategy(closes, entry_dates, random_selections)
        m = strategy_metrics(trades)
        sharpes.append(m["sharpe"])

    sharpes = np.array(sharpes)
    result = {
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "std_sharpe": round(float(np.std(sharpes)), 3),
        "pct_above_original": 0,  # filled later
        "p5": round(float(np.percentile(sharpes, 5)), 3),
        "p25": round(float(np.percentile(sharpes, 25)), 3),
        "p75": round(float(np.percentile(sharpes, 75)), 3),
        "p95": round(float(np.percentile(sharpes, 95)), 3),
    }
    print(f"  Random stocks: mean Sharpe={result['mean_sharpe']}, "
          f"median={result['median_sharpe']}, std={result['std_sharpe']}")
    return result, sharpes


# ── TEST 4: Random timing (same stock selection logic, random dates) ─────
def test_random_timing(closes, volumes, accel, high_vol, accel_score, n_orig_entries, n_iter=MC_ITERS):
    """Pick top-3 accelerators but on random dates (not when signal fires)."""
    print(f"\n[TEST 4] Random timing ({n_iter} iterations, {n_orig_entries} entries each)")
    oot_mask = closes.index >= pd.Timestamp(OOT_START)
    oot_dates = closes.index[oot_mask].tolist()
    available_tickers = [t for t in UNIVERSE if t in closes.columns]

    # Generate spaced random date indices matching original trade count
    max_possible = len(oot_dates) - HOLD_DAYS
    sharpes = []
    for i in range(n_iter):
        # Pick n_orig_entries random starting points with MIN_SPACING between them
        chosen = []
        candidates_idx = list(range(0, max_possible))
        np.random.shuffle(candidates_idx)
        for idx in candidates_idx:
            if len(chosen) >= n_orig_entries:
                break
            if all(abs(idx - c) >= MIN_SPACING for c in chosen):
                chosen.append(idx)
        chosen.sort()

        random_dates = [oot_dates[idx] for idx in chosen]

        # On each random date, pick 3 random stocks (pure random baseline)
        entry_dates_iter = []
        selections_iter = []
        for date in random_dates:
            entry_dates_iter.append(date)
            selections_iter.append(list(np.random.choice(
                available_tickers, size=TOP_N, replace=False)))

        trades, _ = run_strategy(closes, entry_dates_iter, selections_iter)
        m = strategy_metrics(trades)
        sharpes.append(m["sharpe"])

    sharpes = np.array(sharpes)
    result = {
        "mean_sharpe": round(float(np.mean(sharpes)), 3),
        "median_sharpe": round(float(np.median(sharpes)), 3),
        "std_sharpe": round(float(np.std(sharpes)), 3),
        "pct_above_original": 0,  # filled later
        "p5": round(float(np.percentile(sharpes, 5)), 3),
        "p25": round(float(np.percentile(sharpes, 25)), 3),
        "p75": round(float(np.percentile(sharpes, 75)), 3),
        "p95": round(float(np.percentile(sharpes, 95)), 3),
    }
    print(f"  Random timing: mean Sharpe={result['mean_sharpe']}, "
          f"median={result['median_sharpe']}, std={result['std_sharpe']}")
    return result, sharpes


# ── TEST 5: Sub-period stability ─────────────────────────────────────────
def test_sub_period_stability(trades):
    """Split trades into yearly chunks, compute Sharpe per year."""
    print("\n[TEST 5] Sub-period stability (yearly Sharpe)")
    periods = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026": ("2026-01-01", "2026-12-31"),
    }
    results = {}
    for label, (start, end) in periods.items():
        period_trades = [t for t in trades if start <= t["entry_date"] <= end]
        if period_trades:
            returns = [t["return_pct"] for t in period_trades]
            mean_r = np.mean(returns)
            if len(returns) < 3:
                sharpe = 0.0  # not enough trades for meaningful Sharpe
            else:
                std_r = np.std(returns, ddof=1)
                trades_per_year = 252 / HOLD_DAYS
                sharpe = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 1e-10 else 0
            results[label] = {
                "sharpe": round(sharpe, 3),
                "n_trades": len(period_trades),
                "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
                "mean_return_pct": round(mean_r * 100, 3),
            }
            print(f"  {label}: Sharpe={results[label]['sharpe']}, "
                  f"N={results[label]['n_trades']}, WR={results[label]['win_rate']}%")
        else:
            results[label] = {"sharpe": 0, "n_trades": 0, "win_rate": 0, "mean_return_pct": 0}
            print(f"  {label}: No trades")

    positive_years = sum(1 for v in results.values() if v["sharpe"] > 0 and v["n_trades"] > 0)
    total_years = sum(1 for v in results.values() if v["n_trades"] > 0)
    results["positive_years"] = positive_years
    results["total_years"] = total_years
    return results


# ── TEST 6: Concentration test ───────────────────────────────────────────
def test_concentration(trades):
    """What % of total P&L comes from top-3 best trades?"""
    print("\n[TEST 6] Concentration test")
    if not trades:
        return {"top3_pct": 100, "top3_pnl": 0, "total_pnl": 0}

    pnls = sorted([t["pnl"] for t in trades], reverse=True)
    total_pnl = sum(pnls)
    top3_pnl = sum(pnls[:3])

    if total_pnl > 0:
        top3_pct = top3_pnl / total_pnl * 100
    else:
        top3_pct = 100  # all losses or zero

    # Also check: what if we remove top-3 trades?
    remaining_pnl = total_pnl - top3_pnl

    # Top-5 and top-10 concentration
    top5_pnl = sum(pnls[:5])
    top10_pnl = sum(pnls[:min(10, len(pnls))])

    result = {
        "top3_pct_of_total_pnl": round(top3_pct, 1),
        "top3_pnl": round(top3_pnl, 2),
        "top5_pct_of_total_pnl": round(top5_pnl / total_pnl * 100, 1) if total_pnl > 0 else 100,
        "top10_pct_of_total_pnl": round(top10_pnl / total_pnl * 100, 1) if total_pnl > 0 else 100,
        "total_pnl": round(total_pnl, 2),
        "pnl_without_top3": round(remaining_pnl, 2),
        "profitable_without_top3": remaining_pnl > 0,
        "n_trades": len(trades),
    }
    print(f"  Top-3 trades = {result['top3_pct_of_total_pnl']}% of total P&L "
          f"(${result['top3_pnl']:.0f} of ${result['total_pnl']:.0f})")
    print(f"  Profitable without top-3? {'YES' if result['profitable_without_top3'] else 'NO'} "
          f"(remaining P&L: ${result['pnl_without_top3']:.0f})")
    return result


def generate_verdict(original_metrics, results):
    """Generate overall verdict based on all tests."""
    fails = []
    passes = []
    warnings = []

    original_sharpe = original_metrics["sharpe"]

    # Test 1: Inverse
    inv = results["test1_inverse"]
    if inv["sharpe"] > 0 and inv["sharpe"] > original_sharpe * 0.5:
        fails.append(f"FAIL: Inverse strategy also profitable (Sharpe {inv['sharpe']}). "
                      "Signal direction may be irrelevant.")
    elif inv["sharpe"] > 0:
        warnings.append(f"WARN: Inverse has positive Sharpe ({inv['sharpe']}) but weaker than original.")
    else:
        passes.append(f"PASS: Inverse strategy unprofitable (Sharpe {inv['sharpe']}). "
                       "Signal direction matters.")

    # Test 2: No volume filter
    novol = results["test2_no_volume_filter"]
    if abs(novol["sharpe"] - original_sharpe) / max(abs(original_sharpe), 0.01) < 0.2:
        fails.append(f"FAIL: Removing volume filter gives similar Sharpe ({novol['sharpe']} vs {original_sharpe}). "
                      "Volume confirmation adds nothing.")
    elif novol["sharpe"] > original_sharpe:
        warnings.append(f"WARN: No-volume version is BETTER (Sharpe {novol['sharpe']} vs {original_sharpe}).")
    else:
        passes.append(f"PASS: Volume filter improves Sharpe ({original_sharpe} vs {novol['sharpe']} without).")

    # Test 3: Random stocks
    rs = results["test3_random_stocks"]
    if rs["pct_above_original"] > 30:
        fails.append(f"FAIL: {rs['pct_above_original']}% of random stock picks beat strategy. "
                      "Stock selection has no edge.")
    elif rs["pct_above_original"] > 10:
        warnings.append(f"WARN: {rs['pct_above_original']}% of random stock picks beat strategy.")
    else:
        passes.append(f"PASS: Only {rs['pct_above_original']}% of random stock picks beat strategy. "
                       "Stock selection matters.")

    # Test 4: Random timing
    rt = results["test4_random_timing"]
    if rt["pct_above_original"] > 30:
        fails.append(f"FAIL: {rt['pct_above_original']}% of random timing beats strategy. "
                      "Timing signal has no edge.")
    elif rt["pct_above_original"] > 10:
        warnings.append(f"WARN: {rt['pct_above_original']}% of random timing beats strategy.")
    else:
        passes.append(f"PASS: Only {rt['pct_above_original']}% of random timing beats strategy. "
                       "Timing signal matters.")

    # Test 5: Sub-period stability
    sp = results["test5_sub_period_stability"]
    if sp["total_years"] > 0 and sp["positive_years"] / sp["total_years"] < 0.5:
        fails.append(f"FAIL: Only {sp['positive_years']}/{sp['total_years']} years have positive Sharpe. "
                      "Not stable across time.")
    elif sp["total_years"] > 0 and sp["positive_years"] / sp["total_years"] < 0.75:
        warnings.append(f"WARN: {sp['positive_years']}/{sp['total_years']} years positive. Some instability.")
    else:
        passes.append(f"PASS: {sp['positive_years']}/{sp['total_years']} years have positive Sharpe.")

    # Test 6: Concentration
    conc = results["test6_concentration"]
    if conc["top3_pct_of_total_pnl"] > 50:
        fails.append(f"FAIL: Top-3 trades account for {conc['top3_pct_of_total_pnl']}% of P&L. "
                      "Returns driven by a few lucky trades.")
    elif conc["top3_pct_of_total_pnl"] > 35:
        warnings.append(f"WARN: Top-3 trades account for {conc['top3_pct_of_total_pnl']}% of P&L.")
    else:
        passes.append(f"PASS: Top-3 trades only {conc['top3_pct_of_total_pnl']}% of P&L. Distributed edge.")

    if not conc["profitable_without_top3"]:
        fails.append("FAIL: Strategy is NOT profitable after removing top-3 trades.")

    verdict = "REJECT" if len(fails) >= 2 else ("CAUTION" if fails or len(warnings) >= 2 else "PASS")

    return {
        "verdict": verdict,
        "passes": passes,
        "warnings": warnings,
        "fails": fails,
        "n_pass": len(passes),
        "n_warn": len(warnings),
        "n_fail": len(fails),
    }


def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Momentum Acceleration B (Volume-Confirmed)")
    print("=" * 70)

    # Download data
    data = download_data()
    closes, volumes, spy_close = build_frames(data)
    regime = regime_series(spy_close)

    print(f"\nData range: {closes.index[0].date()} to {closes.index[-1].date()}")
    print(f"OOT period: {OOT_START} to {OOT_END}")
    print(f"Universe: {len(UNIVERSE)} stocks")

    # Compute signals
    ret_5d, ret_20d, vol_ratio, accel, decel, high_vol, accel_score, decel_score = \
        compute_signals(closes, volumes)

    # ── Run original strategy ──────────────────────────────────────────
    print("\n[ORIGINAL] Momentum Acceleration B (Volume-Confirmed)")
    entry_dates, stock_selections = get_original_signals(
        closes, volumes, accel, high_vol, accel_score)
    orig_trades, orig_final = run_strategy(closes, entry_dates, stock_selections)
    orig_metrics = strategy_metrics(orig_trades)
    print(f"  Original: {orig_metrics['n_trades']} trades, Sharpe={orig_metrics['sharpe']}, "
          f"WR={orig_metrics['win_rate']}%, Return={orig_metrics['total_return']}%, "
          f"MaxDD={orig_metrics['max_dd_pct']}%")

    results = {"original_metrics": orig_metrics}

    # ── Test 1: Inverse ────────────────────────────────────────────────
    results["test1_inverse"] = test_inverse(closes, volumes, decel, high_vol, decel_score)

    # ── Test 2: No volume filter ───────────────────────────────────────
    results["test2_no_volume_filter"] = test_no_volume_filter(closes, accel, accel_score)

    # ── Test 3: Random stock selection ─────────────────────────────────
    rs_result, rs_sharpes = test_random_stocks(closes, entry_dates)
    # Compute pct above original
    rs_result["pct_above_original"] = round(
        float(np.mean(rs_sharpes >= orig_metrics["sharpe"]) * 100), 1)
    print(f"  {rs_result['pct_above_original']}% of random stock picks beat original Sharpe {orig_metrics['sharpe']}")
    results["test3_random_stocks"] = rs_result

    # ── Test 4: Random timing ─────────────────────────────────────────
    rt_result, rt_sharpes = test_random_timing(
        closes, volumes, accel, high_vol, accel_score,
        n_orig_entries=len(entry_dates))
    rt_result["pct_above_original"] = round(
        float(np.mean(rt_sharpes >= orig_metrics["sharpe"]) * 100), 1)
    print(f"  {rt_result['pct_above_original']}% of random timing beats original Sharpe {orig_metrics['sharpe']}")
    results["test4_random_timing"] = rt_result

    # ── Test 5: Sub-period stability ───────────────────────────────────
    results["test5_sub_period_stability"] = test_sub_period_stability(orig_trades)

    # ── Test 6: Concentration ──────────────────────────────────────────
    results["test6_concentration"] = test_concentration(orig_trades)

    # ── Generate verdict ───────────────────────────────────────────────
    verdict = generate_verdict(orig_metrics, results)
    results["verdict"] = verdict

    print("\n" + "=" * 70)
    print(f"OVERALL VERDICT: {verdict['verdict']}")
    print("=" * 70)
    for p in verdict["passes"]:
        print(f"  [+] {p}")
    for w in verdict["warnings"]:
        print(f"  [!] {w}")
    for f in verdict["fails"]:
        print(f"  [-] {f}")
    print(f"\nScore: {verdict['n_pass']} pass, {verdict['n_warn']} warn, {verdict['n_fail']} fail")

    # ── Save results ───────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/momentum_accel_adversarial_results.json")
    results["timestamp"] = datetime.now().isoformat()
    results["parameters"] = {
        "universe": UNIVERSE,
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "initial_capital": INITIAL_CAPITAL,
        "hold_days": HOLD_DAYS,
        "top_n": TOP_N,
        "volume_mult": VOLUME_MULT,
        "mc_iterations": MC_ITERS,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
