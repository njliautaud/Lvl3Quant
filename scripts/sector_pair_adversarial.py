#!/usr/bin/env python3
"""
Sector Pair Mean Reversion — ADVERSARIAL VALIDATION
=====================================================
6 adversarial tests on variants A, C, F to check if alpha is real or artifact.

Tests:
  1. Inverse Direction — buy leader instead of laggard
  2. Random Pair Entry — shuffle which ETF in the pair to buy (1000 iters)
  3. Random Timing — randomize entry dates (1000 iters)
  4. Sub-Period Stability — split OOT into 2 halves
  5. Top-Trade Removal — remove top 3 most profitable trades
  6. Alternative Pair Test — use 4 different sector pairs
"""

import json
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Constants (match original backtest) ──────────────────────────────────
ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002
LOOKBACK = 20
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2021-06-01"
VIX_THRESHOLD = 25.0
N_RANDOM_ITERS = 1000

PAIRS = [
    ("XLK", "XLC", "Tech vs Communications"),
    ("XLF", "XLV", "Financials vs Healthcare"),
    ("XLY", "XLP", "Consumer Disc vs Staples"),
    ("XLE", "XLK", "Energy vs Tech"),
]

ALT_PAIRS = [
    ("XLI", "XLB", "Industrials vs Materials"),
    ("XLU", "XLRE", "Utilities vs Real Estate"),
    ("IYT", "XLI", "Transportation vs Industrials"),
    ("XLC", "XLK", "Communications vs Tech"),
]


def download_data(extra_tickers=None):
    """Download all needed tickers."""
    tickers = set()
    for a, b, _ in PAIRS:
        tickers.add(a)
        tickers.add(b)
    for a, b, _ in ALT_PAIRS:
        tickers.add(a)
        tickers.add(b)
    tickers.add("SPY")
    tickers.add("^VIX")
    if extra_tickers:
        tickers.update(extra_tickers)

    print(f"Downloading {len(tickers)} tickers from {DATA_START} to {OOT_END}...")
    data = yf.download(list(tickers), start=DATA_START, end=OOT_END,
                       auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data

    if "^VIX" in closes.columns:
        closes = closes.rename(columns={"^VIX": "VIX"})

    closes = closes.ffill().dropna()
    print(f"  Got {len(closes)} trading days")
    return closes


def compute_zscore(closes, etf1, etf2, lookback=LOOKBACK):
    ratio = closes[etf1] / closes[etf2]
    roll_mean = ratio.rolling(lookback).mean()
    roll_std = ratio.rolling(lookback).std()
    return (ratio - roll_mean) / roll_std


def backtest_pair_shares(closes, etf1, etf2, z_entry=2.0, z_exit=0.0,
                         lookback=LOOKBACK, capital=ACCOUNT,
                         vix_gate=None, inverse=False,
                         oot_start=None, oot_end=None):
    """
    Backtest a single pair with shares.
    inverse=True: buy the LEADER instead of laggard.
    """
    z = compute_zscore(closes, etf1, etf2, lookback)
    start = pd.Timestamp(oot_start or OOT_START)
    end = pd.Timestamp(oot_end or OOT_END)
    mask = (z.index >= start) & (z.index <= end)
    z_oot = z[mask]

    trades = []
    position = None
    equity_curve = [capital]
    equity_dates = [z_oot.index[0]]
    current_capital = capital

    for date, zscore in z_oot.items():
        if np.isnan(zscore):
            equity_curve.append(current_capital)
            equity_dates.append(date)
            continue

        # Mark-to-market / exit
        if position is not None:
            ticker = position["ticker"]
            current_price = closes.loc[date, ticker]

            should_exit = False
            if position["side"] == "buy_etf2" and zscore <= z_exit:
                should_exit = True
            elif position["side"] == "buy_etf1" and zscore >= -z_exit:
                should_exit = True

            if should_exit:
                exit_price = current_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                current_capital += pnl + position["cost_basis"]
                trades.append({
                    "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "pair": f"{etf1}/{etf2}",
                    "side": position["side"],
                    "entry_price": round(position["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": position["shares"],
                    "pnl": round(pnl, 2),
                    "hold_days": (date - position["entry_date"]).days,
                })
                position = None

        # Entry
        if position is None:
            if vix_gate is not None:
                if "VIX" in closes.columns and date in closes.index:
                    if closes.loc[date, "VIX"] >= vix_gate:
                        equity_curve.append(current_capital)
                        equity_dates.append(date)
                        continue

            buy_ticker = None
            buy_side = None
            if zscore > z_entry:
                if inverse:
                    # Buy leader (etf1) instead of laggard (etf2)
                    buy_ticker = etf1
                    buy_side = "buy_etf2"  # keep side label for exit logic
                else:
                    buy_ticker = etf2
                    buy_side = "buy_etf2"
            elif zscore < -z_entry:
                if inverse:
                    buy_ticker = etf2
                    buy_side = "buy_etf1"
                else:
                    buy_ticker = etf1
                    buy_side = "buy_etf1"

            if buy_ticker is not None:
                entry_price = closes.loc[date, buy_ticker] * (1 + SLIPPAGE_PCT)
                invest = current_capital * 0.95
                shares = int(invest / entry_price)
                if shares > 0:
                    cost_basis = shares * entry_price
                    current_capital -= cost_basis
                    position = {
                        "ticker": buy_ticker,
                        "side": buy_side,
                        "entry_price": entry_price,
                        "entry_date": date,
                        "shares": shares,
                        "cost_basis": cost_basis,
                    }

        if position is not None:
            mtm = position["shares"] * closes.loc[date, position["ticker"]]
            equity_curve.append(current_capital + mtm)
        else:
            equity_curve.append(current_capital)
        equity_dates.append(date)

    # Force-close
    if position is not None:
        last_date = z_oot.index[-1]
        exit_price = closes.loc[last_date, position["ticker"]] * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - position["entry_price"]) * position["shares"]
        current_capital += pnl + position["cost_basis"]
        trades.append({
            "entry_date": position["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": last_date.strftime("%Y-%m-%d"),
            "ticker": position["ticker"],
            "pair": f"{etf1}/{etf2}",
            "side": position["side"],
            "entry_price": round(position["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "shares": position["shares"],
            "pnl": round(pnl, 2),
            "hold_days": (last_date - position["entry_date"]).days,
            "forced_close": True,
        })
        equity_curve[-1] = current_capital

    eq = pd.Series(equity_curve, index=equity_dates)
    eq = eq[~eq.index.duplicated(keep="last")]
    return trades, eq


def backtest_multi_pair(closes, pairs, z_entry=2.0, z_exit=0.0,
                        lookback=LOOKBACK, capital=ACCOUNT,
                        vix_gate=None, inverse=False,
                        oot_start=None, oot_end=None):
    """Run all pairs simultaneously with equal allocation."""
    n = len(pairs)
    alloc = capital / n
    all_trades = []
    equities = []

    for etf1, etf2, desc in pairs:
        trades, eq = backtest_pair_shares(
            closes, etf1, etf2, z_entry, z_exit, lookback, alloc,
            vix_gate=vix_gate, inverse=inverse,
            oot_start=oot_start, oot_end=oot_end,
        )
        all_trades.extend(trades)
        equities.append(eq)

    eq_df = pd.DataFrame({f"{p[0]}/{p[1]}": eq for p, eq in zip(pairs, equities)})
    eq_df = eq_df.ffill().bfill()
    combined = eq_df.sum(axis=1)
    return all_trades, combined


def compute_sharpe(equity_series, capital=None):
    """Compute annualized Sharpe from equity series."""
    daily_ret = equity_series.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        return (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    return 0.0


def compute_metrics_from_trades(trades, capital=ACCOUNT):
    """Lightweight metrics from trade list only."""
    if not trades:
        return {"sharpe_approx": 0, "total_pnl": 0, "num_trades": 0, "win_rate": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    arr = np.array(pnls)
    return {
        "sharpe_approx": float((arr.mean() / arr.std()) * np.sqrt(len(arr))) if arr.std() > 0 else 0,
        "total_pnl": sum(pnls),
        "num_trades": len(pnls),
        "win_rate": len(wins) / len(pnls) * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VARIANT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(closes, variant, inverse=False, pairs_override=None,
                oot_start=None, oot_end=None):
    """Run a specific variant and return (trades, equity_series, sharpe)."""
    pairs = pairs_override or PAIRS

    if variant == "A":
        trades, eq = backtest_pair_shares(
            closes, "XLK", "XLC", z_entry=2.0, z_exit=0.0,
            inverse=inverse, oot_start=oot_start, oot_end=oot_end)
    elif variant == "C":
        trades, eq = backtest_multi_pair(
            closes, pairs, z_entry=2.0, z_exit=0.0,
            inverse=inverse, oot_start=oot_start, oot_end=oot_end)
    elif variant == "F":
        trades, eq = backtest_pair_shares(
            closes, "XLK", "XLC", z_entry=2.0, z_exit=0.0,
            vix_gate=VIX_THRESHOLD, inverse=inverse,
            oot_start=oot_start, oot_end=oot_end)
    else:
        raise ValueError(f"Unknown variant {variant}")

    sharpe = compute_sharpe(eq)
    return trades, eq, sharpe


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE DIRECTION
# ═══════════════════════════════════════════════════════════════════════════

def test_inverse(closes, variant):
    """Buy leader instead of laggard. PASS if inverse Sharpe < 0.3."""
    trades, eq, sharpe = run_variant(closes, variant, inverse=True)
    passed = sharpe < 0.3
    return {
        "test": "1_inverse_direction",
        "inverse_sharpe": round(sharpe, 3),
        "inverse_trades": len(trades),
        "inverse_pnl": round(sum(t["pnl"] for t in trades), 2) if trades else 0,
        "threshold": "< 0.3",
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM PAIR ENTRY
# ═══════════════════════════════════════════════════════════════════════════

def test_random_pair(closes, variant):
    """
    On each trade signal, randomly pick which ETF in the pair to buy
    (ignore z-score direction). Run 1000 iterations.
    PASS if strategy beats random at p < 0.05.
    """
    # Get actual strategy result
    actual_trades, actual_eq, actual_sharpe = run_variant(closes, variant)
    actual_total_pnl = sum(t["pnl"] for t in actual_trades)

    # For random test, we need the z-score signal dates and pairs.
    # We'll re-run the backtest but randomly flip which ETF to buy.
    rng = np.random.default_rng(42)
    random_pnls = []

    if variant in ("A", "F"):
        etf1, etf2 = "XLK", "XLC"
        vix_gate = VIX_THRESHOLD if variant == "F" else None

        z = compute_zscore(closes, etf1, etf2)
        mask = z.index >= pd.Timestamp(OOT_START)
        z_oot = z[mask]

        for i in range(N_RANDOM_ITERS):
            # Run backtest but randomly flip direction on each entry
            trades = []
            position = None
            current_capital = ACCOUNT

            for date, zscore in z_oot.items():
                if np.isnan(zscore):
                    continue

                # Exit logic (same as normal)
                if position is not None:
                    ticker = position["ticker"]
                    current_price = closes.loc[date, ticker]
                    should_exit = False
                    if position["side"] == "buy_etf2" and zscore <= 0:
                        should_exit = True
                    elif position["side"] == "buy_etf1" and zscore >= 0:
                        should_exit = True

                    if should_exit:
                        exit_price = current_price * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price - position["entry_price"]) * position["shares"]
                        current_capital += pnl + position["cost_basis"]
                        trades.append({"pnl": pnl})
                        position = None

                # Entry: trigger on same z-score threshold, but random direction
                if position is None:
                    if vix_gate is not None:
                        if "VIX" in closes.columns and date in closes.index:
                            if closes.loc[date, "VIX"] >= vix_gate:
                                continue

                    if abs(zscore) > 2.0:
                        # Random pick which ETF to buy
                        if rng.random() < 0.5:
                            buy_ticker = etf1
                            buy_side = "buy_etf1"
                        else:
                            buy_ticker = etf2
                            buy_side = "buy_etf2"

                        entry_price = closes.loc[date, buy_ticker] * (1 + SLIPPAGE_PCT)
                        invest = current_capital * 0.95
                        shares = int(invest / entry_price)
                        if shares > 0:
                            cost_basis = shares * entry_price
                            current_capital -= cost_basis
                            position = {
                                "ticker": buy_ticker,
                                "side": buy_side,
                                "entry_price": entry_price,
                                "entry_date": date,
                                "shares": shares,
                                "cost_basis": cost_basis,
                            }

            # Force close
            if position is not None:
                last_date = z_oot.index[-1]
                exit_price = closes.loc[last_date, position["ticker"]] * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - position["entry_price"]) * position["shares"]
                current_capital += pnl + position["cost_basis"]
                trades.append({"pnl": pnl})

            random_pnls.append(sum(t["pnl"] for t in trades))

    elif variant == "C":
        # Multi-pair: for each iteration, randomly flip direction on each pair
        for i in range(N_RANDOM_ITERS):
            total_pnl = 0
            for etf1, etf2, _ in PAIRS:
                alloc = ACCOUNT / len(PAIRS)
                z = compute_zscore(closes, etf1, etf2)
                mask = z.index >= pd.Timestamp(OOT_START)
                z_oot = z[mask]

                trades = []
                position = None
                current_capital = alloc

                for date, zscore in z_oot.items():
                    if np.isnan(zscore):
                        continue

                    if position is not None:
                        ticker = position["ticker"]
                        current_price = closes.loc[date, ticker]
                        should_exit = False
                        if position["side"] == "buy_etf2" and zscore <= 0:
                            should_exit = True
                        elif position["side"] == "buy_etf1" and zscore >= 0:
                            should_exit = True
                        if should_exit:
                            exit_price = current_price * (1 - SLIPPAGE_PCT)
                            pnl = (exit_price - position["entry_price"]) * position["shares"]
                            current_capital += pnl + position["cost_basis"]
                            trades.append({"pnl": pnl})
                            position = None

                    if position is None and abs(zscore) > 2.0:
                        if rng.random() < 0.5:
                            buy_ticker = etf1
                            buy_side = "buy_etf1"
                        else:
                            buy_ticker = etf2
                            buy_side = "buy_etf2"

                        entry_price = closes.loc[date, buy_ticker] * (1 + SLIPPAGE_PCT)
                        invest = current_capital * 0.95
                        shares = int(invest / entry_price)
                        if shares > 0:
                            cost_basis = shares * entry_price
                            current_capital -= cost_basis
                            position = {
                                "ticker": buy_ticker,
                                "side": buy_side,
                                "entry_price": entry_price,
                                "entry_date": date,
                                "shares": shares,
                                "cost_basis": cost_basis,
                            }

                if position is not None:
                    last_date = z_oot.index[-1]
                    exit_price = closes.loc[last_date, position["ticker"]] * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price - position["entry_price"]) * position["shares"]
                    current_capital += pnl + position["cost_basis"]
                    trades.append({"pnl": pnl})

                total_pnl += sum(t["pnl"] for t in trades)
            random_pnls.append(total_pnl)

    random_pnls = np.array(random_pnls)
    p_value = np.mean(random_pnls >= actual_total_pnl)
    passed = p_value < 0.05

    return {
        "test": "2_random_pair_entry",
        "actual_pnl": round(actual_total_pnl, 2),
        "random_mean_pnl": round(float(random_pnls.mean()), 2),
        "random_std_pnl": round(float(random_pnls.std()), 2),
        "random_median_pnl": round(float(np.median(random_pnls)), 2),
        "p_value": round(float(p_value), 4),
        "threshold": "p < 0.05",
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: RANDOM TIMING
# ═══════════════════════════════════════════════════════════════════════════

def test_random_timing(closes, variant):
    """
    Keep same pairs and buy-laggard direction, but randomize WHEN we enter.
    Randomly select entry dates from the OOT period (same number of trades).
    PASS if strategy beats random at p < 0.05.
    """
    actual_trades, actual_eq, actual_sharpe = run_variant(closes, variant)
    actual_total_pnl = sum(t["pnl"] for t in actual_trades)
    n_trades = len(actual_trades)

    if n_trades == 0:
        return {"test": "3_random_timing", "passed": False, "reason": "no trades"}

    # Get all available OOT dates
    oot_mask = closes.index >= pd.Timestamp(OOT_START)
    oot_dates = closes.index[oot_mask]

    # Average hold days from actual trades
    avg_hold = int(np.mean([t["hold_days"] for t in actual_trades]))
    avg_hold = max(avg_hold, 1)

    rng = np.random.default_rng(42)
    random_pnls = []

    if variant in ("A", "F"):
        etf1, etf2 = "XLK", "XLC"

        for _ in range(N_RANDOM_ITERS):
            # Pick n_trades random entry dates
            entry_indices = rng.choice(len(oot_dates) - avg_hold - 1, size=n_trades, replace=True)
            total_pnl = 0
            capital = ACCOUNT

            for idx in sorted(entry_indices):
                entry_date = oot_dates[idx]
                exit_idx = min(idx + avg_hold, len(oot_dates) - 1)
                exit_date = oot_dates[exit_idx]

                # Randomly pick which ETF (but still buy laggard logic based on z-score)
                z_val = compute_zscore(closes, etf1, etf2).get(entry_date, 0)
                if np.isnan(z_val) or abs(z_val) < 0.01:
                    buy_ticker = etf2 if rng.random() < 0.5 else etf1
                elif z_val > 0:
                    buy_ticker = etf2  # laggard
                else:
                    buy_ticker = etf1

                entry_price = closes.loc[entry_date, buy_ticker] * (1 + SLIPPAGE_PCT)
                exit_price = closes.loc[exit_date, buy_ticker] * (1 - SLIPPAGE_PCT)
                invest = capital * 0.95
                shares = int(invest / entry_price)
                if shares > 0:
                    pnl = (exit_price - entry_price) * shares
                    total_pnl += pnl

            random_pnls.append(total_pnl)

    elif variant == "C":
        for _ in range(N_RANDOM_ITERS):
            total_pnl = 0
            for etf1, etf2, _ in PAIRS:
                # Count trades from this pair in actual
                pair_trades = [t for t in actual_trades if t["pair"] == f"{etf1}/{etf2}"]
                n_pair = len(pair_trades)
                if n_pair == 0:
                    continue
                alloc = ACCOUNT / len(PAIRS)

                entry_indices = rng.choice(len(oot_dates) - avg_hold - 1, size=n_pair, replace=True)
                for idx in sorted(entry_indices):
                    entry_date = oot_dates[idx]
                    exit_idx = min(idx + avg_hold, len(oot_dates) - 1)
                    exit_date = oot_dates[exit_idx]

                    z_val = compute_zscore(closes, etf1, etf2).get(entry_date, 0)
                    if np.isnan(z_val) or abs(z_val) < 0.01:
                        buy_ticker = etf2 if rng.random() < 0.5 else etf1
                    elif z_val > 0:
                        buy_ticker = etf2
                    else:
                        buy_ticker = etf1

                    entry_price = closes.loc[entry_date, buy_ticker] * (1 + SLIPPAGE_PCT)
                    exit_price = closes.loc[exit_date, buy_ticker] * (1 - SLIPPAGE_PCT)
                    invest = alloc * 0.95
                    shares = int(invest / entry_price)
                    if shares > 0:
                        pnl = (exit_price - entry_price) * shares
                        total_pnl += pnl

            random_pnls.append(total_pnl)

    random_pnls = np.array(random_pnls)
    p_value = np.mean(random_pnls >= actual_total_pnl)
    passed = p_value < 0.05

    return {
        "test": "3_random_timing",
        "actual_pnl": round(actual_total_pnl, 2),
        "random_mean_pnl": round(float(random_pnls.mean()), 2),
        "random_std_pnl": round(float(random_pnls.std()), 2),
        "p_value": round(float(p_value), 4),
        "threshold": "p < 0.05",
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: SUB-PERIOD STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def test_subperiod(closes, variant):
    """Split OOT into 2 halves. PASS if both halves have positive Sharpe."""
    half1_start = OOT_START
    half1_end = "2024-07-31"
    half2_start = "2024-08-01"
    half2_end = OOT_END

    trades1, eq1, sharpe1 = run_variant(closes, variant,
                                        oot_start=half1_start, oot_end=half1_end)
    trades2, eq2, sharpe2 = run_variant(closes, variant,
                                        oot_start=half2_start, oot_end=half2_end)

    pnl1 = sum(t["pnl"] for t in trades1) if trades1 else 0
    pnl2 = sum(t["pnl"] for t in trades2) if trades2 else 0

    passed = sharpe1 > 0 and sharpe2 > 0

    return {
        "test": "4_subperiod_stability",
        "half1_period": f"{half1_start} to {half1_end}",
        "half1_sharpe": round(sharpe1, 3),
        "half1_trades": len(trades1),
        "half1_pnl": round(pnl1, 2),
        "half2_period": f"{half2_start} to {half2_end}",
        "half2_sharpe": round(sharpe2, 3),
        "half2_trades": len(trades2),
        "half2_pnl": round(pnl2, 2),
        "threshold": "both halves Sharpe > 0",
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: TOP-TRADE REMOVAL
# ═══════════════════════════════════════════════════════════════════════════

def test_top_trade_removal(closes, variant):
    """Remove top 3 most profitable trades. PASS if Sharpe still > 0.3."""
    trades, eq, full_sharpe = run_variant(closes, variant)

    if len(trades) <= 3:
        return {
            "test": "5_top_trade_removal",
            "passed": False,
            "reason": f"Only {len(trades)} trades, can't remove 3",
        }

    # Sort by PnL descending, identify top 3
    sorted_trades = sorted(trades, key=lambda t: t["pnl"], reverse=True)
    top3 = sorted_trades[:3]
    top3_pnl = sum(t["pnl"] for t in top3)
    total_pnl = sum(t["pnl"] for t in trades)
    top3_pct = (top3_pnl / total_pnl * 100) if total_pnl != 0 else 0

    # Remove top 3 and reconstruct equity curve
    remaining_trades = sorted_trades[3:]
    remaining_pnls = [t["pnl"] for t in remaining_trades]
    remaining_total = sum(remaining_pnls)

    # Approximate remaining Sharpe from trade PnLs
    arr = np.array(remaining_pnls)
    if len(arr) > 1 and arr.std() > 0:
        remaining_sharpe_approx = (arr.mean() / arr.std()) * np.sqrt(len(arr))
    else:
        remaining_sharpe_approx = 0.0

    # Also reconstruct equity curve without top 3 trades for proper Sharpe
    # Re-run but skip the top 3 trade dates
    top3_entries = set(t["entry_date"] for t in top3)

    if variant in ("A", "F"):
        etf1, etf2 = "XLK", "XLC"
        vix_gate = VIX_THRESHOLD if variant == "F" else None
        z = compute_zscore(closes, etf1, etf2)
        mask = z.index >= pd.Timestamp(OOT_START)
        z_oot = z[mask]

        new_trades = []
        position = None
        current_capital = ACCOUNT
        equity_curve = [ACCOUNT]
        equity_dates = [z_oot.index[0]]

        for date, zscore in z_oot.items():
            if np.isnan(zscore):
                equity_curve.append(current_capital)
                equity_dates.append(date)
                continue

            if position is not None:
                ticker = position["ticker"]
                current_price = closes.loc[date, ticker]
                should_exit = False
                if position["side"] == "buy_etf2" and zscore <= 0:
                    should_exit = True
                elif position["side"] == "buy_etf1" and zscore >= 0:
                    should_exit = True
                if should_exit:
                    exit_price = current_price * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price - position["entry_price"]) * position["shares"]
                    current_capital += pnl + position["cost_basis"]
                    new_trades.append({"pnl": pnl})
                    position = None

            if position is None:
                if vix_gate is not None and "VIX" in closes.columns and date in closes.index:
                    if closes.loc[date, "VIX"] >= vix_gate:
                        equity_curve.append(current_capital)
                        equity_dates.append(date)
                        continue

                # Skip if this would be a top-3 trade
                date_str = date.strftime("%Y-%m-%d")
                if date_str in top3_entries:
                    equity_curve.append(current_capital)
                    equity_dates.append(date)
                    continue

                buy_ticker = None
                buy_side = None
                if zscore > 2.0:
                    buy_ticker = etf2
                    buy_side = "buy_etf2"
                elif zscore < -2.0:
                    buy_ticker = etf1
                    buy_side = "buy_etf1"

                if buy_ticker is not None:
                    entry_price = closes.loc[date, buy_ticker] * (1 + SLIPPAGE_PCT)
                    invest = current_capital * 0.95
                    shares = int(invest / entry_price)
                    if shares > 0:
                        cost_basis = shares * entry_price
                        current_capital -= cost_basis
                        position = {
                            "ticker": buy_ticker,
                            "side": buy_side,
                            "entry_price": entry_price,
                            "entry_date": date,
                            "shares": shares,
                            "cost_basis": cost_basis,
                        }

            if position is not None:
                mtm = position["shares"] * closes.loc[date, position["ticker"]]
                equity_curve.append(current_capital + mtm)
            else:
                equity_curve.append(current_capital)
            equity_dates.append(date)

        if position is not None:
            last_date = z_oot.index[-1]
            exit_price = closes.loc[last_date, position["ticker"]] * (1 - SLIPPAGE_PCT)
            pnl = (exit_price - position["entry_price"]) * position["shares"]
            current_capital += pnl + position["cost_basis"]
            new_trades.append({"pnl": pnl})
            equity_curve[-1] = current_capital

        eq_reduced = pd.Series(equity_curve, index=equity_dates)
        eq_reduced = eq_reduced[~eq_reduced.index.duplicated(keep="last")]
        remaining_sharpe = compute_sharpe(eq_reduced)
    else:
        # For variant C, use the approximate Sharpe from trade PnLs
        remaining_sharpe = remaining_sharpe_approx

    passed = remaining_sharpe > 0.3

    return {
        "test": "5_top_trade_removal",
        "full_sharpe": round(full_sharpe, 3),
        "full_trades": len(trades),
        "full_pnl": round(total_pnl, 2),
        "top3_pnl": round(top3_pnl, 2),
        "top3_pct_of_total": round(top3_pct, 1),
        "remaining_sharpe": round(remaining_sharpe, 3),
        "remaining_trades": len(remaining_trades),
        "remaining_pnl": round(remaining_total, 2),
        "top3_trades": [
            {"entry": t["entry_date"], "ticker": t["ticker"], "pnl": t["pnl"]}
            for t in top3
        ],
        "threshold": "remaining Sharpe > 0.3",
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: ALTERNATIVE PAIR TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_alt_pairs(closes, variant):
    """
    Replace original pairs with 4 different sector pairs.
    PASS if alt pairs Sharpe < strategy Sharpe * 0.5.
    """
    actual_trades, actual_eq, actual_sharpe = run_variant(closes, variant)

    if variant == "C":
        # Multi-pair: run with alt pairs
        alt_trades, alt_eq, alt_sharpe = run_variant(
            closes, variant, pairs_override=ALT_PAIRS)
    else:
        # For A and F (single pair XLK/XLC), test each alt pair individually
        # and take the best one as the comparison
        alt_sharpes = []
        for etf1, etf2, desc in ALT_PAIRS:
            try:
                vix_gate = VIX_THRESHOLD if variant == "F" else None
                t, e = backtest_pair_shares(
                    closes, etf1, etf2, z_entry=2.0, z_exit=0.0,
                    vix_gate=vix_gate)
                s = compute_sharpe(e)
                alt_sharpes.append({
                    "pair": f"{etf1}/{etf2} ({desc})",
                    "sharpe": round(s, 3),
                    "trades": len(t),
                    "pnl": round(sum(x["pnl"] for x in t), 2),
                })
            except Exception as ex:
                alt_sharpes.append({
                    "pair": f"{etf1}/{etf2} ({desc})",
                    "sharpe": 0,
                    "trades": 0,
                    "pnl": 0,
                    "error": str(ex),
                })

        best_alt = max(alt_sharpes, key=lambda x: x["sharpe"])
        alt_sharpe = best_alt["sharpe"]

    threshold = actual_sharpe * 0.5
    if variant == "C":
        passed = alt_sharpe < threshold
        alt_detail = {
            "alt_sharpe": round(alt_sharpe, 3),
            "alt_trades": len(alt_trades),
            "alt_pnl": round(sum(t["pnl"] for t in alt_trades), 2),
        }
    else:
        passed = alt_sharpe < threshold
        alt_detail = {
            "best_alt_sharpe": round(alt_sharpe, 3),
            "best_alt_pair": best_alt["pair"],
            "all_alt_results": alt_sharpes,
        }

    return {
        "test": "6_alternative_pairs",
        "actual_sharpe": round(actual_sharpe, 3),
        "threshold": f"alt Sharpe < {round(threshold, 3)} (50% of actual)",
        **alt_detail,
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SECTOR PAIR MEAN REVERSION — ADVERSARIAL VALIDATION")
    print(f"Testing variants A, C, F | {N_RANDOM_ITERS} random iterations")
    print(f"OOT: {OOT_START} to {OOT_END} | Account: ${ACCOUNT}")
    print("=" * 70)

    closes = download_data()

    variants = ["A", "C", "F"]
    all_results = {}

    for v in variants:
        print(f"\n{'='*70}")
        print(f"VARIANT {v} — ADVERSARIAL TESTS")
        print(f"{'='*70}")

        # Get baseline
        base_trades, base_eq, base_sharpe = run_variant(closes, v)
        base_pnl = sum(t["pnl"] for t in base_trades)
        print(f"  Baseline: Sharpe={base_sharpe:.3f}, Trades={len(base_trades)}, PnL=${base_pnl:.2f}")

        v_results = {
            "baseline_sharpe": round(base_sharpe, 3),
            "baseline_trades": len(base_trades),
            "baseline_pnl": round(base_pnl, 2),
            "tests": {},
        }

        # Test 1: Inverse Direction
        print(f"\n  Test 1: Inverse Direction...", end=" ", flush=True)
        r1 = test_inverse(closes, v)
        print(f"{'PASS' if r1['passed'] else 'FAIL'} (inverse Sharpe={r1['inverse_sharpe']:.3f})")
        v_results["tests"]["1_inverse"] = r1

        # Test 2: Random Pair Entry
        print(f"  Test 2: Random Pair Entry ({N_RANDOM_ITERS} iters)...", end=" ", flush=True)
        r2 = test_random_pair(closes, v)
        print(f"{'PASS' if r2['passed'] else 'FAIL'} (p={r2['p_value']:.4f})")
        v_results["tests"]["2_random_pair"] = r2

        # Test 3: Random Timing
        print(f"  Test 3: Random Timing ({N_RANDOM_ITERS} iters)...", end=" ", flush=True)
        r3 = test_random_timing(closes, v)
        print(f"{'PASS' if r3['passed'] else 'FAIL'} (p={r3['p_value']:.4f})")
        v_results["tests"]["3_random_timing"] = r3

        # Test 4: Sub-Period Stability
        print(f"  Test 4: Sub-Period Stability...", end=" ", flush=True)
        r4 = test_subperiod(closes, v)
        print(f"{'PASS' if r4['passed'] else 'FAIL'} "
              f"(H1 Sharpe={r4['half1_sharpe']:.3f}, H2 Sharpe={r4['half2_sharpe']:.3f})")
        v_results["tests"]["4_subperiod"] = r4

        # Test 5: Top-Trade Removal
        print(f"  Test 5: Top-Trade Removal...", end=" ", flush=True)
        r5 = test_top_trade_removal(closes, v)
        if "reason" in r5:
            print(f"FAIL ({r5['reason']})")
        else:
            print(f"{'PASS' if r5['passed'] else 'FAIL'} "
                  f"(remaining Sharpe={r5['remaining_sharpe']:.3f}, "
                  f"top3={r5['top3_pct_of_total']:.1f}% of PnL)")
        v_results["tests"]["5_top_trade_removal"] = r5

        # Test 6: Alternative Pairs
        print(f"  Test 6: Alternative Pairs...", end=" ", flush=True)
        r6 = test_alt_pairs(closes, v)
        if "best_alt_sharpe" in r6:
            print(f"{'PASS' if r6['passed'] else 'FAIL'} "
                  f"(best alt Sharpe={r6['best_alt_sharpe']:.3f} vs threshold={base_sharpe*0.5:.3f})")
        else:
            print(f"{'PASS' if r6['passed'] else 'FAIL'} "
                  f"(alt Sharpe={r6['alt_sharpe']:.3f} vs threshold={base_sharpe*0.5:.3f})")
        v_results["tests"]["6_alt_pairs"] = r6

        # Summary
        tests_passed = sum(1 for t in v_results["tests"].values() if t.get("passed", False))
        v_results["tests_passed"] = tests_passed
        v_results["tests_total"] = 6
        v_results["adversarial_verdict"] = "REAL ALPHA" if tests_passed >= 5 else (
            "SUSPICIOUS" if tests_passed >= 3 else "LIKELY ARTIFACT")

        all_results[v] = v_results

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)

    header = f"{'Var':<4} {'Base Sharpe':>11} {'Inverse':>8} {'RndPair':>8} {'RndTime':>8} {'SubPrd':>8} {'Top3Rm':>8} {'AltPair':>8} {'Score':>6} {'Verdict':<16}"
    print(header)
    print("-" * len(header))

    for v in variants:
        r = all_results[v]
        tests = r["tests"]
        inv = "PASS" if tests["1_inverse"].get("passed") else "FAIL"
        rp = "PASS" if tests["2_random_pair"].get("passed") else "FAIL"
        rt = "PASS" if tests["3_random_timing"].get("passed") else "FAIL"
        sp = "PASS" if tests["4_subperiod"].get("passed") else "FAIL"
        tr = "PASS" if tests["5_top_trade_removal"].get("passed") else "FAIL"
        ap = "PASS" if tests["6_alt_pairs"].get("passed") else "FAIL"
        score = f"{r['tests_passed']}/6"
        verdict = r["adversarial_verdict"]
        print(f"{v:<4} {r['baseline_sharpe']:>11.3f} {inv:>8} {rp:>8} {rt:>8} {sp:>8} {tr:>8} {ap:>8} {score:>6} {verdict:<16}")

    # ── Detailed pass/fail criteria ──
    print("\n  PASSING CRITERIA:")
    print("    Test 1 (Inverse): Inverse Sharpe < 0.3")
    print("    Test 2 (Random Pair): p < 0.05")
    print("    Test 3 (Random Timing): p < 0.05")
    print("    Test 4 (Sub-Period): Both halves Sharpe > 0")
    print("    Test 5 (Top-Trade): Remaining Sharpe > 0.3")
    print("    Test 6 (Alt Pairs): Alt Sharpe < 50% of actual")
    print("    VERDICT: 5-6 pass = REAL ALPHA | 3-4 = SUSPICIOUS | 0-2 = LIKELY ARTIFACT")

    # Save
    output = {
        "strategy": "Sector Pair Mean Reversion",
        "test_type": "Adversarial Validation",
        "run_timestamp": datetime.now().isoformat(),
        "random_iterations": N_RANDOM_ITERS,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "variants_tested": variants,
        "results": all_results,
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/sector_pair_adversarial_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
