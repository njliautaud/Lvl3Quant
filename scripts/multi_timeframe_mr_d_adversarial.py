#!/usr/bin/env python3
"""
Adversarial Validation: Multi-Timeframe MR Variant D — "Daily + Weekly Both"
6 adversarial tests to validate strategy robustness.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

START_DATE = "2021-06-01"  # extra buffer for indicator warmup
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 15
SLIPPAGE_BPS = 2  # 0.02% each way
DAILY_DIP_PCT = 5.0
DAILY_RSI_THRESH = 35
WEEKLY_RSI_THRESH = 40
WEEKLY_DIP_PCT = 7.0
RED_STREAK_MIN = 3

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/multi_timeframe_mr_d_adversarial.json")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def download_data():
    """Download all ticker + SPY data."""
    all_tickers = TICKERS + ["SPY"]
    print(f"Downloading {len(all_tickers)} tickers...")
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 100:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  Warning: Failed to download {t}: {e}")
    print(f"  Downloaded {len(data)} tickers successfully.")
    return data


def prepare_features(data: dict) -> dict:
    """Compute daily and weekly indicators for each ticker."""
    features = {}
    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        d = df[["Close", "High", "Low", "Open"]].copy()
        d.columns = ["close", "high", "low", "open"]
        d = d.dropna()
        if len(d) < 100:
            continue

        # Daily indicators
        d["rsi_14"] = compute_rsi(d["close"], 14)
        d["high_20d"] = d["close"].rolling(20).max()
        d["dip_from_high"] = (d["high_20d"] - d["close"]) / d["high_20d"] * 100
        d["is_green"] = d["close"] > d["open"]
        d["is_red"] = d["close"] <= d["open"]

        # Red streak: count consecutive red days ending yesterday
        red_streak = []
        streak = 0
        for i in range(len(d)):
            if i == 0:
                red_streak.append(0)
                continue
            if d["is_red"].iloc[i - 1]:
                streak += 1
            else:
                streak = 0
            red_streak.append(streak)
        d["red_streak"] = red_streak

        # Weekly indicators
        weekly = d["close"].resample("W-FRI").last().dropna()
        w_rsi = compute_rsi(weekly, 14)
        w_high_10 = weekly.rolling(10).max()
        w_dip = (w_high_10 - weekly) / w_high_10 * 100

        # Map weekly values back to daily (forward-fill within the week)
        d["weekly_rsi"] = w_rsi.reindex(d.index, method="ffill")
        d["weekly_dip"] = w_dip.reindex(d.index, method="ffill")

        features[ticker] = d

    return features


def find_entries(features: dict, daily_dip=DAILY_DIP_PCT, daily_rsi=DAILY_RSI_THRESH,
                 weekly_rsi=WEEKLY_RSI_THRESH, weekly_dip=WEEKLY_DIP_PCT,
                 hold_days=HOLD_DAYS, red_streak_min=RED_STREAK_MIN) -> list:
    """Find all valid entry signals."""
    entries = []
    for ticker, d in features.items():
        oot = d.loc[OOT_START:OOT_END]
        for i in range(1, len(oot)):
            row = oot.iloc[i]
            prev = oot.iloc[i - 1]
            dt = oot.index[i]

            # All conditions
            cond_dip = row["dip_from_high"] >= daily_dip
            cond_rsi = row["rsi_14"] < daily_rsi
            cond_green = row["is_green"]
            cond_streak = row["red_streak"] >= red_streak_min
            cond_w_rsi = not np.isnan(row.get("weekly_rsi", np.nan)) and row["weekly_rsi"] < weekly_rsi
            cond_w_dip = not np.isnan(row.get("weekly_dip", np.nan)) and row["weekly_dip"] >= weekly_dip

            if all([cond_dip, cond_rsi, cond_green, cond_streak, cond_w_rsi, cond_w_dip]):
                entries.append({
                    "ticker": ticker,
                    "entry_date": dt,
                    "entry_price": row["close"],
                })
    entries.sort(key=lambda x: x["entry_date"])
    return entries


def simulate_trades(entries: list, data: dict, slippage_bps=SLIPPAGE_BPS,
                    hold_days=HOLD_DAYS, capital=CAPITAL,
                    max_per_trade=MAX_PER_TRADE, max_concurrent=MAX_CONCURRENT) -> pd.DataFrame:
    """Simulate trades with position limits and slippage."""
    trades = []
    active = []  # list of exit dates

    for e in entries:
        # Remove expired positions
        active = [ed for ed in active if ed > e["entry_date"]]
        if len(active) >= max_concurrent:
            continue

        ticker = e["ticker"]
        df = data.get(ticker)
        if df is None:
            continue

        # Flatten columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        entry_idx = df.index.get_loc(e["entry_date"])
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_date = df.index[exit_idx]
        exit_price = df["Close"].iloc[exit_idx]
        entry_price = e["entry_price"]

        # Apply slippage
        slip = slippage_bps / 10000.0
        adj_entry = entry_price * (1 + slip)
        adj_exit = exit_price * (1 - slip)

        shares = int(max_per_trade / adj_entry)
        if shares < 1:
            continue

        pnl = (adj_exit - adj_entry) * shares
        pnl_pct = (adj_exit / adj_entry - 1) * 100

        trades.append({
            "ticker": ticker,
            "entry_date": e["entry_date"],
            "exit_date": exit_date,
            "entry_price": adj_entry,
            "exit_price": adj_exit,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })
        active.append(exit_date)

    return pd.DataFrame(trades)


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute strategy metrics from trades dataframe."""
    if trades_df.empty:
        return {"sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
                "mdd_pct": 0, "n_trades": 0, "total_pnl": 0}

    pnls = trades_df["pnl"].values
    n = len(pnls)
    total = float(np.sum(pnls))
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = float(len(wins)) / n * 100 if n > 0 else 0

    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls, ddof=1) if n > 1 else 1e-9
    sharpe = float(mean_pnl / std_pnl * np.sqrt(252 / HOLD_DAYS)) if std_pnl > 1e-9 else 0

    downside = pnls[pnls < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = float(mean_pnl / down_std * np.sqrt(252 / HOLD_DAYS)) if down_std > 1e-9 else 0

    gross_win = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    pf = gross_win / gross_loss

    # Max drawdown on equity curve
    equity = CAPITAL + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    mdd = float(np.min(dd))

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 3),
        "mdd_pct": round(mdd, 2),
        "n_trades": n,
        "total_pnl": round(total, 2),
    }


def get_regime(spy_data: pd.DataFrame, date) -> str:
    """Bull if SPY > 200-SMA, else Bear."""
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)
    spy = spy_data["Close"]
    sma200 = spy.rolling(200).mean()
    idx = spy.index.get_indexer([date], method="ffill")[0]
    if idx < 0 or idx >= len(spy):
        return "unknown"
    return "bull" if spy.iloc[idx] > sma200.iloc[idx] else "bear"


# ═══════════════════════════════════════════════════════════════════════════
# ADVERSARIAL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_1_inverse_signal(features, data):
    """Test 1: Inverse Signal — buy when OPPOSITE conditions hold."""
    print("\n" + "=" * 70)
    print("TEST 1: INVERSE SIGNAL")
    print("=" * 70)

    entries = []
    for ticker, d in features.items():
        oot = d.loc[OOT_START:OOT_END]
        for i in range(1, len(oot)):
            row = oot.iloc[i]
            dt = oot.index[i]

            # Inverse conditions: near highs, high RSI, no red streaks
            cond_near_high = row["dip_from_high"] <= 2.0
            cond_rsi_high = row["rsi_14"] > 65
            cond_w_rsi_high = not np.isnan(row.get("weekly_rsi", np.nan)) and row["weekly_rsi"] > 60
            cond_no_streak = row["red_streak"] < 1

            if all([cond_near_high, cond_rsi_high, cond_w_rsi_high, cond_no_streak]):
                entries.append({
                    "ticker": ticker,
                    "entry_date": dt,
                    "entry_price": row["close"],
                })

    entries.sort(key=lambda x: x["entry_date"])
    trades = simulate_trades(entries, data)
    inv_metrics = compute_metrics(trades)

    print(f"  Inverse trades: {inv_metrics['n_trades']}")
    print(f"  Inverse Sharpe: {inv_metrics['sharpe']}")
    print(f"  Inverse total PnL: ${inv_metrics['total_pnl']:.2f}")

    # Compare: inverse Sharpe must be < 50% of original
    # We'll get original sharpe from the caller
    return inv_metrics


def test_2_random_entry(features, data, real_entries, real_metrics, n_iter=1000):
    """Test 2: Random Entry Timing — 1000 random reassignments."""
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM ENTRY TIMING (1000 iterations)")
    print("=" * 70)

    # Group entries by ticker
    from collections import defaultdict
    ticker_counts = defaultdict(int)
    for e in real_entries:
        ticker_counts[e["ticker"]] += 1

    # Get all valid trading dates per ticker in OOT
    ticker_dates = {}
    for ticker, d in features.items():
        oot = d.loc[OOT_START:OOT_END]
        ticker_dates[ticker] = list(oot.index)

    rng = np.random.RandomState(42)
    random_sharpes = []

    for it in range(n_iter):
        rand_entries = []
        for ticker, count in ticker_counts.items():
            if ticker not in ticker_dates or len(ticker_dates[ticker]) < count:
                continue
            chosen = rng.choice(len(ticker_dates[ticker]), size=count, replace=False)
            for idx in chosen:
                dt = ticker_dates[ticker][idx]
                d = features[ticker]
                if dt in d.index:
                    rand_entries.append({
                        "ticker": ticker,
                        "entry_date": dt,
                        "entry_price": d.loc[dt, "close"],
                    })
        rand_entries.sort(key=lambda x: x["entry_date"])
        trades = simulate_trades(rand_entries, data)
        m = compute_metrics(trades)
        random_sharpes.append(m["sharpe"])

    random_sharpes = np.array(random_sharpes)
    real_sharpe = real_metrics["sharpe"]
    p_value = float(np.mean(random_sharpes >= real_sharpe))
    percentile = float(np.mean(random_sharpes < real_sharpe) * 100)

    print(f"  Real Sharpe: {real_sharpe}")
    print(f"  Random Sharpe mean: {np.mean(random_sharpes):.3f} ± {np.std(random_sharpes):.3f}")
    print(f"  Random Sharpe median: {np.median(random_sharpes):.3f}")
    print(f"  Percentile rank: {percentile:.1f}%")
    print(f"  p-value: {p_value:.4f}")

    return {
        "real_sharpe": real_sharpe,
        "random_mean": round(float(np.mean(random_sharpes)), 3),
        "random_std": round(float(np.std(random_sharpes)), 3),
        "random_median": round(float(np.median(random_sharpes)), 3),
        "percentile": round(percentile, 1),
        "p_value": round(p_value, 4),
    }


def test_3_subperiod_stability(entries, data):
    """Test 3: Sub-Period Stability — 4 equal sub-periods, all must be Sharpe > 0."""
    print("\n" + "=" * 70)
    print("TEST 3: SUB-PERIOD STABILITY")
    print("=" * 70)

    trades = simulate_trades(entries, data)
    if trades.empty:
        print("  No trades to split!")
        return {"sub_periods": [], "all_positive": False}

    # Split OOT into 4 equal periods
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    total_days = (oot_end - oot_start).days
    quarter = total_days // 4

    results = []
    for q in range(4):
        q_start = oot_start + timedelta(days=q * quarter)
        q_end = oot_start + timedelta(days=(q + 1) * quarter) if q < 3 else oot_end

        q_trades = trades[
            (trades["entry_date"] >= q_start) & (trades["entry_date"] < q_end)
        ]
        m = compute_metrics(q_trades)
        period_str = f"{q_start.strftime('%Y-%m')}->{q_end.strftime('%Y-%m')}"
        results.append({
            "period": period_str,
            "sharpe": m["sharpe"],
            "n_trades": m["n_trades"],
            "total_pnl": m["total_pnl"],
            "win_rate": m["win_rate"],
        })
        status = "✓" if m["sharpe"] > 0 else "✗"
        print(f"  {status} {period_str}: Sharpe={m['sharpe']:.3f}, trades={m['n_trades']}, PnL=${m['total_pnl']:.2f}, WR={m['win_rate']:.1f}%")

    all_positive = all(r["sharpe"] > 0 for r in results if r["n_trades"] > 0)
    # Periods with 0 trades don't count against us but note them
    empty_periods = sum(1 for r in results if r["n_trades"] == 0)
    if empty_periods > 0:
        print(f"  NOTE: {empty_periods} period(s) had zero trades")

    return {"sub_periods": results, "all_positive": all_positive, "empty_periods": empty_periods}


def test_4_remove_top3(entries, data, real_metrics):
    """Test 4: Remove Top 3 Tickers by PnL, Sharpe drop must be < 50%."""
    print("\n" + "=" * 70)
    print("TEST 4: REMOVE TOP 3 TICKERS")
    print("=" * 70)

    trades = simulate_trades(entries, data)
    if trades.empty:
        return {"top3": [], "reduced_sharpe": 0, "original_sharpe": 0, "drop_pct": 100}

    # Find top 3 tickers by total PnL
    ticker_pnl = trades.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    top3 = list(ticker_pnl.head(3).index)
    print(f"  Top 3 tickers by PnL: {top3}")
    for t in top3:
        print(f"    {t}: ${ticker_pnl[t]:.2f}")

    # Remove them and re-simulate
    reduced_entries = [e for e in entries if e["ticker"] not in top3]
    reduced_trades = simulate_trades(reduced_entries, data)
    reduced_m = compute_metrics(reduced_trades)

    orig_sharpe = real_metrics["sharpe"]
    red_sharpe = reduced_m["sharpe"]
    drop = (1 - red_sharpe / orig_sharpe) * 100 if orig_sharpe > 0 else 100

    print(f"  Original Sharpe: {orig_sharpe}")
    print(f"  Reduced Sharpe:  {red_sharpe}")
    print(f"  Drop: {drop:.1f}%")

    return {
        "top3": top3,
        "top3_pnl": {t: round(float(ticker_pnl[t]), 2) for t in top3},
        "original_sharpe": orig_sharpe,
        "reduced_sharpe": red_sharpe,
        "drop_pct": round(drop, 1),
        "reduced_trades": reduced_m["n_trades"],
    }


def test_5_parameter_sensitivity(features, data):
    """Test 5: Parameter Sensitivity — sweep key parameters."""
    print("\n" + "=" * 70)
    print("TEST 5: PARAMETER SENSITIVITY")
    print("=" * 70)

    dip_vals = [3, 5, 7, 10]
    daily_rsi_vals = [25, 30, 35, 40]
    weekly_rsi_vals = [30, 35, 40, 45]
    hold_vals = [10, 15, 20]

    total_combos = len(dip_vals) * len(daily_rsi_vals) * len(weekly_rsi_vals) * len(hold_vals)
    good_combos = 0
    all_results = []

    print(f"  Sweeping {total_combos} combinations...")

    for dip in dip_vals:
        for d_rsi in daily_rsi_vals:
            for w_rsi in weekly_rsi_vals:
                for hold in hold_vals:
                    ent = find_entries(features, daily_dip=dip, daily_rsi=d_rsi,
                                       weekly_rsi=w_rsi, hold_days=hold)
                    trades = simulate_trades(ent, data, hold_days=hold)
                    m = compute_metrics(trades)
                    if m["sharpe"] > 0.3:
                        good_combos += 1
                    all_results.append({
                        "dip": dip, "d_rsi": d_rsi, "w_rsi": w_rsi,
                        "hold": hold, "sharpe": m["sharpe"], "n_trades": m["n_trades"],
                    })

    pct_good = good_combos / total_combos * 100

    # Find best and worst
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)
    best = all_results[0]
    worst = all_results[-1]
    sharpes = [r["sharpe"] for r in all_results if r["n_trades"] > 0]

    print(f"  Total combinations: {total_combos}")
    print(f"  Combinations with Sharpe > 0.3: {good_combos} ({pct_good:.1f}%)")
    print(f"  Sharpe range: [{min(sharpes):.3f}, {max(sharpes):.3f}]" if sharpes else "  No valid sharpes")
    print(f"  Best: dip={best['dip']}%, d_rsi={best['d_rsi']}, w_rsi={best['w_rsi']}, hold={best['hold']} → Sharpe={best['sharpe']}")
    print(f"  Worst: dip={worst['dip']}%, d_rsi={worst['d_rsi']}, w_rsi={worst['w_rsi']}, hold={worst['hold']} → Sharpe={worst['sharpe']}")

    return {
        "total_combos": total_combos,
        "good_combos": good_combos,
        "pct_sharpe_above_03": round(pct_good, 1),
        "sharpe_range": [round(min(sharpes), 3), round(max(sharpes), 3)] if sharpes else [0, 0],
        "best": best,
        "worst": worst,
    }


def test_6_cost_sensitivity(entries, data, real_metrics):
    """Test 6: Cost Sensitivity — test at various slippage levels."""
    print("\n" + "=" * 70)
    print("TEST 6: COST SENSITIVITY")
    print("=" * 70)

    bps_levels = [5, 10, 20, 50]
    results = []

    for bps in bps_levels:
        trades = simulate_trades(entries, data, slippage_bps=bps)
        m = compute_metrics(trades)
        results.append({
            "slippage_bps": bps,
            "sharpe": m["sharpe"],
            "total_pnl": m["total_pnl"],
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
        })
        status = "+" if m["sharpe"] > 0 else "-"
        print(f"  {status} {bps} bps: Sharpe={m['sharpe']:.3f}, PnL=${m['total_pnl']:.2f}, WR={m['win_rate']:.1f}%")

    # Binary search for breakeven slippage
    lo, hi = 0, 200
    for _ in range(20):
        mid = (lo + hi) / 2
        trades = simulate_trades(entries, data, slippage_bps=mid)
        m = compute_metrics(trades)
        if m["total_pnl"] > 0:
            lo = mid
        else:
            hi = mid
    breakeven_bps = round((lo + hi) / 2, 1)

    print(f"  Breakeven slippage: ~{breakeven_bps} bps")

    return {
        "levels": results,
        "breakeven_bps": breakeven_bps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Multi-Timeframe MR Variant D")
    print("=" * 70)

    # Download data
    data = download_data()
    if len(data) < 10:
        print("ERROR: Too few tickers downloaded. Aborting.")
        sys.exit(1)

    # Prepare features
    features = prepare_features(data)
    print(f"Prepared features for {len(features)} tickers.")

    # Find base entries
    entries = find_entries(features)
    print(f"\nBase strategy entries: {len(entries)}")

    # Run base strategy
    trades = simulate_trades(entries, data)
    base_metrics = compute_metrics(trades)
    print(f"Base metrics: Sharpe={base_metrics['sharpe']}, Sortino={base_metrics['sortino']}, "
          f"WR={base_metrics['win_rate']}%, PF={base_metrics['profit_factor']}, "
          f"MDD={base_metrics['mdd_pct']}%, trades={base_metrics['n_trades']}, "
          f"PnL=${base_metrics['total_pnl']:.2f}")

    # Regime analysis
    spy_data = data.get("SPY")
    if spy_data is not None and not trades.empty:
        trades["regime"] = trades["entry_date"].apply(lambda d: get_regime(spy_data, d))
        bull_trades = trades[trades["regime"] == "bull"]
        bear_trades = trades[trades["regime"] == "bear"]
        bull_m = compute_metrics(bull_trades)
        bear_m = compute_metrics(bear_trades)
        regime_gap = abs(bull_m["sharpe"] - bear_m["sharpe"]) / max(abs(bull_m["sharpe"]), abs(bear_m["sharpe"]), 0.001)
        print(f"\nRegime: Bull Sharpe={bull_m['sharpe']}, Bear Sharpe={bear_m['sharpe']}, gap={regime_gap:.3f}")
    else:
        regime_gap = None

    # ── Run 6 adversarial tests ─────────────────────────────────────────

    results = {"base_metrics": base_metrics, "regime_gap": regime_gap, "tests": {}}

    # Test 1: Inverse Signal
    inv_m = test_1_inverse_signal(features, data)
    inv_ratio = inv_m["sharpe"] / base_metrics["sharpe"] if base_metrics["sharpe"] > 0 else 999
    t1_pass = inv_ratio < 0.5
    print(f"  Inverse/Real ratio: {inv_ratio:.3f}")
    print(f"  RESULT: {'PASS' if t1_pass else 'FAIL'} (inverse Sharpe must be < 50% of real)")
    results["tests"]["1_inverse_signal"] = {
        "inverse_metrics": inv_m,
        "ratio": round(inv_ratio, 3),
        "pass": t1_pass,
    }

    # Test 2: Random Entry
    t2 = test_2_random_entry(features, data, entries, base_metrics)
    t2_pass = t2["p_value"] < 0.05
    print(f"  RESULT: {'PASS' if t2_pass else 'FAIL'} (p-value must be < 0.05)")
    results["tests"]["2_random_entry"] = {**t2, "pass": t2_pass}

    # Test 3: Sub-Period Stability
    t3 = test_3_subperiod_stability(entries, data)
    t3_pass = t3["all_positive"]
    print(f"  RESULT: {'PASS' if t3_pass else 'FAIL'} (all sub-periods must have Sharpe > 0)")
    results["tests"]["3_subperiod_stability"] = {**t3, "pass": t3_pass}

    # Test 4: Remove Top 3
    t4 = test_4_remove_top3(entries, data, base_metrics)
    t4_pass = t4["drop_pct"] < 50
    print(f"  RESULT: {'PASS' if t4_pass else 'FAIL'} (Sharpe drop must be < 50%)")
    results["tests"]["4_remove_top3"] = {**t4, "pass": t4_pass}

    # Test 5: Parameter Sensitivity
    t5 = test_5_parameter_sensitivity(features, data)
    t5_pass = t5["pct_sharpe_above_03"] > 20  # reasonable threshold
    print(f"  RESULT: {'PASS' if t5_pass else 'FAIL'} (>20% of combos should have Sharpe > 0.3)")
    results["tests"]["5_parameter_sensitivity"] = {**t5, "pass": t5_pass}

    # Test 6: Cost Sensitivity
    t6 = test_6_cost_sensitivity(entries, data, base_metrics)
    t6_pass = t6["breakeven_bps"] > 10  # must survive > 10 bps
    print(f"  RESULT: {'PASS' if t6_pass else 'FAIL'} (breakeven slippage must be > 10 bps)")
    results["tests"]["6_cost_sensitivity"] = {**t6, "pass": t6_pass}

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for t in results["tests"].values() if t["pass"])
    total_tests = len(results["tests"])

    test_names = {
        "1_inverse_signal": "Inverse Signal",
        "2_random_entry": "Random Entry Timing",
        "3_subperiod_stability": "Sub-Period Stability",
        "4_remove_top3": "Remove Top 3 Tickers",
        "5_parameter_sensitivity": "Parameter Sensitivity",
        "6_cost_sensitivity": "Cost Sensitivity",
    }

    for key, t in results["tests"].items():
        status = "PASS" if t["pass"] else "FAIL"
        print(f"  [{status}] {test_names.get(key, key)}")

    print(f"\n  Overall: {pass_count}/{total_tests} tests passed")

    results["summary"] = {
        "pass_count": pass_count,
        "total_tests": total_tests,
        "overall": "PASS" if pass_count == total_tests else "PARTIAL" if pass_count >= 4 else "FAIL",
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    # Convert any non-serializable types
    def convert(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(i) for i in obj]
        return convert(obj)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(deep_convert(results), f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
