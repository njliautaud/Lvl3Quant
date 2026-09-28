#!/usr/bin/env python3
"""
AnalystC_H1 Adversarial Validation — 5-Check Framework
=======================================================
Strategy: Buy stocks gapping >3% on earnings beat, hold 40 days,
          half-size position when SPY < 200-SMA.

Tests:
  1. Inverse Direction — buy gap-down >3% on earnings miss
  2. Random Timing — buy same stocks on random dates (100 iterations)
  3. Top-Ticker Removal — remove top 5 most profitable tickers
  4. Survivorship Bias — remove delisted/acquired tickers
  5. Regime-Split Stability — sub-period analysis (2022, 2023, 2024, 2025, 2026-H1)
"""

import json
import sys
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuration ──────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%, $0 commission (RH shares)
HOLD_DAYS = 40

EARNINGS_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER",
    "LYFT", "COIN", "RBLX", "DDOG", "TTD", "SHOP", "NET", "ROKU"
]

# Tickers known to have been delisted/acquired during 2022-2026 period
# (None in this universe were delisted — all still trading as of Jul 2026)
# But we check programmatically below
KNOWN_DELISTED = []

EARNINGS_CACHE = Path("/home/jupiter/Lvl3Quant/data/earnings_dates_cache.json")
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/analystc_h1_adversarial_results.json")

RANDOM_TIMING_ITERATIONS = 100
RANDOM_SEED = 42


# ─── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all needed price data."""
    print("Downloading price data...")
    fetch_start = "2021-01-01"  # Extra for 200-SMA warmup

    all_tickers = list(set(EARNINGS_UNIVERSE + ["SPY"]))
    raw = yf.download(all_tickers, start=fetch_start, end=OOT_END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        opn = raw["Open"].copy()
    else:
        close = raw[["Close"]].copy()
        opn = raw[["Open"]].copy()

    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(close.columns)} tickers")
    return close, opn


def compute_regime(spy_close):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("bull", index=spy_close.index)
    regime[spy_close < sma200] = "bear"
    return regime


# ─── Core Strategy Logic ────────────────────────────────────────────────────
def run_strategy(close, opn, regime_s, gap_direction="up", gap_threshold=0.03,
                 ticker_subset=None, date_overrides=None, hold_days=HOLD_DAYS):
    """
    Run the AnalystC_H1 strategy.

    Args:
        gap_direction: "up" for gap >threshold, "down" for gap <-threshold
        gap_threshold: absolute gap threshold (default 0.03 = 3%)
        ticker_subset: list of tickers to use (default: EARNINGS_UNIVERSE)
        date_overrides: dict {ticker: [dates]} to use instead of gap detection
        hold_days: how many trading days to hold

    Returns:
        list of trade dicts
    """
    trades = []
    tickers = ticker_subset or EARNINGS_UNIVERSE

    for ticker in tickers:
        if ticker not in close.columns or ticker not in opn.columns:
            continue

        c = close[ticker].dropna()
        o = opn[ticker].dropna()

        if date_overrides and ticker in date_overrides:
            # Use provided dates instead of detecting gaps
            signal_dates = [pd.Timestamp(d) for d in date_overrides[ticker]
                           if pd.Timestamp(d) in c.index]
        else:
            # Find gap signals
            common_idx = c.index.intersection(o.index)
            prev_close = c.reindex(common_idx).shift(1)
            gap_pct = (o.reindex(common_idx) - prev_close) / prev_close

            # Filter to OOT period
            gap_pct = gap_pct[gap_pct.index >= pd.Timestamp(OOT_START)]

            if gap_direction == "up":
                signal_dates = gap_pct[gap_pct > gap_threshold].index
            else:  # "down"
                signal_dates = gap_pct[gap_pct < -gap_threshold].index

        for entry_date in signal_dates:
            if entry_date not in c.index:
                continue
            idx_pos = c.index.get_loc(entry_date)

            # Get regime on entry date
            r = regime_s.get(entry_date, "bull")

            # Half-size in bear (Hedge A)
            size_mult = 0.5 if r == "bear" else 1.0

            # Entry at open + slippage
            if entry_date in o.index:
                entry_price = o.loc[entry_date] * (1 + SLIPPAGE_PCT)
            else:
                continue

            # Exit hold_days trading days later
            exit_idx = min(idx_pos + hold_days, len(c) - 1)
            exit_date = c.index[exit_idx]
            exit_price = c.iloc[exit_idx] * (1 - SLIPPAGE_PCT)

            ret = (exit_price / entry_price - 1) * size_mult

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ticker": ticker,
                "ret": ret,
                "regime": r,
                "size_mult": size_mult,
            })

    return trades


# ─── Performance Metrics ────────────────────────────────────────────────────
def compute_metrics(trades, starting_capital=STARTING_CAPITAL):
    """Compute Sharpe, Sortino, PF, WR, MaxDD from trade list."""
    if not trades or len(trades) == 0:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "max_dd": -1.0,
                "n_trades": 0, "total_return": 0, "sharpe_bull": 0, "sharpe_bear": 0}

    df = pd.DataFrame(trades)
    rets = df["ret"].values
    n = len(rets)

    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0

    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 1e-9 else 0

    gross_profit = np.sum(rets[rets > 0])
    gross_loss = abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else 999.0

    wr = np.sum(rets > 0) / n

    equity = starting_capital * np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)

    total_return = equity[-1] / starting_capital - 1

    # Regime-specific
    bull_rets = df[df["regime"] == "bull"]["ret"].values
    bear_rets = df[df["regime"] == "bear"]["ret"].values

    def regime_sharpe(r):
        if len(r) < 2:
            return 0
        m, s = np.mean(r), np.std(r, ddof=1)
        return (m / s) * np.sqrt(252) if s > 1e-9 else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 4),
        "max_dd": round(float(max_dd), 4),
        "n_trades": int(n),
        "total_return": round(float(total_return), 4),
        "sharpe_bull": round(float(regime_sharpe(bull_rets)), 3),
        "sharpe_bear": round(float(regime_sharpe(bear_rets)), 3),
        "n_bull": int(len(bull_rets)),
        "n_bear": int(len(bear_rets)),
    }


# ─── Test 1: Inverse Direction ──────────────────────────────────────────────
def test_inverse_direction(close, opn, regime_s):
    """
    MOST IMPORTANT TEST: Buy stocks that gap DOWN >3% (earnings miss proxy)
    instead of gap UP. If inverse also makes money, the beat signal is irrelevant.
    """
    print("\n" + "=" * 70)
    print("TEST 1: INVERSE DIRECTION (Gap Down >3% instead of Gap Up)")
    print("=" * 70)

    # Run original
    original_trades = run_strategy(close, opn, regime_s, gap_direction="up")
    orig_metrics = compute_metrics(original_trades)
    print(f"  Original (gap up >3%): Sharpe={orig_metrics['sharpe']:.3f}, "
          f"N={orig_metrics['n_trades']}, WR={orig_metrics['wr']:.1%}")

    # Run inverse
    inverse_trades = run_strategy(close, opn, regime_s, gap_direction="down")
    inv_metrics = compute_metrics(inverse_trades)
    print(f"  Inverse  (gap dn >3%): Sharpe={inv_metrics['sharpe']:.3f}, "
          f"N={inv_metrics['n_trades']}, WR={inv_metrics['wr']:.1%}")

    # Pass criteria: inverse Sharpe must be meaningfully WORSE than original
    # If inverse Sharpe > 0.5 * original Sharpe, the direction doesn't matter much
    inverse_sharpe = inv_metrics['sharpe']
    orig_sharpe = orig_metrics['sharpe']

    if orig_sharpe <= 0:
        passed = False
        detail = "Original strategy has non-positive Sharpe, cannot validate"
    elif inverse_sharpe > 0.5 * orig_sharpe:
        passed = False
        detail = (f"FAIL: Inverse Sharpe ({inverse_sharpe:.3f}) is >{50}% of original "
                  f"({orig_sharpe:.3f}). The gap-up signal may be irrelevant — "
                  f"both directions make money (likely beta exposure).")
    elif inverse_sharpe > 0:
        passed = False
        detail = (f"WARNING: Inverse also profitable (Sharpe={inverse_sharpe:.3f}) but "
                  f"meaningfully lower than original ({orig_sharpe:.3f}). Some signal "
                  f"value exists but beta exposure is a concern.")
        # Be strict: if inverse is profitable at all, it's a concern
        # But pass if ratio < 0.3
        if inverse_sharpe < 0.3 * orig_sharpe:
            passed = True
            detail = (f"PASS (marginal): Inverse Sharpe ({inverse_sharpe:.3f}) is <30% "
                      f"of original ({orig_sharpe:.3f}). Direction has meaningful signal, "
                      f"but inverse still profitable — monitor beta exposure.")
    else:
        passed = True
        detail = (f"PASS: Inverse Sharpe ({inverse_sharpe:.3f}) is negative or near-zero. "
                  f"The gap-up (earnings beat) signal provides genuine directional edge.")

    print(f"  Result: {'PASS' if passed else 'FAIL'} — {detail}")

    return {
        "passed": passed,
        "details": detail,
        "inverse_sharpe": inverse_sharpe,
        "original_sharpe": orig_sharpe,
        "inverse_metrics": inv_metrics,
        "original_metrics": orig_metrics,
    }


# ─── Test 2: Random Timing ─────────────────────────────────────────────────
def test_random_timing(close, opn, regime_s):
    """
    Buy same stocks on random dates (offset -30 to +30 days from actual gap dates).
    100 iterations. If random timing produces similar Sharpe, edge isn't from earnings.
    """
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM TIMING (100 iterations, ±30 day offset)")
    print("=" * 70)

    # First get original gap-up signal dates per ticker
    original_trades = run_strategy(close, opn, regime_s, gap_direction="up")
    orig_metrics = compute_metrics(original_trades)
    orig_sharpe = orig_metrics['sharpe']
    print(f"  Original Sharpe: {orig_sharpe:.3f} ({orig_metrics['n_trades']} trades)")

    # Collect signal dates per ticker
    signal_dates_by_ticker = defaultdict(list)
    for t in original_trades:
        signal_dates_by_ticker[t['ticker']].append(t['entry_date'])

    rng = np.random.default_rng(RANDOM_SEED)
    random_sharpes = []
    trading_days = close.index[close.index >= pd.Timestamp(OOT_START)]

    for iteration in range(RANDOM_TIMING_ITERATIONS):
        # For each ticker's signal dates, offset by random -30 to +30 trading days
        date_overrides = {}
        for ticker, dates in signal_dates_by_ticker.items():
            offset_dates = []
            for d in dates:
                if d not in close.index:
                    continue
                d_idx = close.index.get_loc(d)
                offset = rng.integers(-30, 31)  # -30 to +30
                new_idx = max(0, min(len(close.index) - 1, d_idx + offset))
                new_date = close.index[new_idx]
                # Make sure it's in OOT period
                if new_date >= pd.Timestamp(OOT_START):
                    offset_dates.append(new_date)
            if offset_dates:
                date_overrides[ticker] = offset_dates

        random_trades = run_strategy(close, opn, regime_s, gap_direction="up",
                                     date_overrides=date_overrides)
        rm = compute_metrics(random_trades)
        random_sharpes.append(rm['sharpe'])

        if (iteration + 1) % 25 == 0:
            print(f"  Iteration {iteration+1}/{RANDOM_TIMING_ITERATIONS}...")

    random_sharpes = np.array(random_sharpes)
    mean_random = float(np.mean(random_sharpes))
    std_random = float(np.std(random_sharpes))
    p_value = float(np.mean(random_sharpes >= orig_sharpe))

    print(f"  Random Sharpe: mean={mean_random:.3f}, std={std_random:.3f}")
    print(f"  p-value (random >= original): {p_value:.4f}")

    passed = p_value < 0.05
    if passed:
        detail = (f"PASS: Original Sharpe ({orig_sharpe:.3f}) significantly exceeds "
                  f"random timing (mean={mean_random:.3f}, p={p_value:.4f}). "
                  f"Edge is timing-specific, not just stock selection.")
    else:
        detail = (f"FAIL: Original Sharpe ({orig_sharpe:.3f}) NOT significantly better "
                  f"than random timing (mean={mean_random:.3f}, p={p_value:.4f}). "
                  f"Edge may be from stock selection bias, not earnings timing.")

    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "details": detail,
        "mean_random_sharpe": mean_random,
        "std_random_sharpe": std_random,
        "p_value": p_value,
        "original_sharpe": orig_sharpe,
        "percentile_95": float(np.percentile(random_sharpes, 95)),
        "percentile_99": float(np.percentile(random_sharpes, 99)),
    }


# ─── Test 3: Top-Ticker Removal ────────────────────────────────────────────
def test_top_ticker_removal(close, opn, regime_s):
    """
    Remove the top 5 most profitable tickers and re-run.
    If Sharpe drops below 0.5, strategy is concentrated in a few names.
    """
    print("\n" + "=" * 70)
    print("TEST 3: TOP-TICKER REMOVAL (Remove top 5 by total P&L)")
    print("=" * 70)

    original_trades = run_strategy(close, opn, regime_s, gap_direction="up")
    orig_metrics = compute_metrics(original_trades)

    # Compute P&L by ticker
    ticker_pnl = defaultdict(float)
    ticker_count = defaultdict(int)
    for t in original_trades:
        ticker_pnl[t['ticker']] += t['ret']
        ticker_count[t['ticker']] += 1

    # Sort by total P&L descending
    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 10 tickers by total return:")
    for ticker, pnl in sorted_tickers[:10]:
        print(f"    {ticker}: {pnl:.4f} total ret ({ticker_count[ticker]} trades)")

    # Remove top 5
    top5 = [t[0] for t in sorted_tickers[:5]]
    print(f"\n  Removing top 5: {top5}")

    remaining = [t for t in EARNINGS_UNIVERSE if t not in top5]
    reduced_trades = run_strategy(close, opn, regime_s, gap_direction="up",
                                  ticker_subset=remaining)
    red_metrics = compute_metrics(reduced_trades)

    print(f"  Original:     Sharpe={orig_metrics['sharpe']:.3f}, N={orig_metrics['n_trades']}")
    print(f"  After removal: Sharpe={red_metrics['sharpe']:.3f}, N={red_metrics['n_trades']}")

    reduced_sharpe = red_metrics['sharpe']
    passed = reduced_sharpe >= 0.5
    sharpe_retention = reduced_sharpe / orig_metrics['sharpe'] if orig_metrics['sharpe'] > 0 else 0

    if passed:
        detail = (f"PASS: After removing top 5 tickers ({', '.join(top5)}), "
                  f"Sharpe={reduced_sharpe:.3f} (>{0.5} threshold). "
                  f"Retains {sharpe_retention:.0%} of original Sharpe. Not concentrated.")
    else:
        detail = (f"FAIL: After removing top 5 tickers ({', '.join(top5)}), "
                  f"Sharpe drops to {reduced_sharpe:.3f} (<{0.5} threshold). "
                  f"Strategy is concentrated in a few names.")

    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "details": detail,
        "reduced_sharpe": reduced_sharpe,
        "original_sharpe": orig_metrics['sharpe'],
        "sharpe_retention": round(sharpe_retention, 3),
        "tickers_removed": top5,
        "ticker_pnl_ranking": {t: round(float(p), 4) for t, p in sorted_tickers},
        "reduced_metrics": red_metrics,
    }


# ─── Test 4: Survivorship Bias ─────────────────────────────────────────────
def test_survivorship_bias(close, opn, regime_s):
    """
    Check for tickers that were delisted/acquired during backtest period.
    Remove them and re-run. Also check for data gaps that suggest survivorship issues.
    """
    print("\n" + "=" * 70)
    print("TEST 4: SURVIVORSHIP BIAS")
    print("=" * 70)

    original_trades = run_strategy(close, opn, regime_s, gap_direction="up")
    orig_metrics = compute_metrics(original_trades)

    # Check each ticker for data availability issues
    problematic_tickers = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    for ticker in EARNINGS_UNIVERSE:
        if ticker not in close.columns:
            problematic_tickers.append((ticker, "missing entirely"))
            continue

        c = close[ticker].dropna()
        if len(c) == 0:
            problematic_tickers.append((ticker, "no data"))
            continue

        # Check if data ends significantly before OOT_END (possible delisting)
        last_date = c.index[-1]
        if last_date < oot_end - pd.Timedelta(days=30):
            problematic_tickers.append((ticker, f"data ends {last_date.date()} (possible delisting)"))
            continue

        # Check for large gaps (>30 trading days) in the OOT period
        oot_data = c[c.index >= oot_start]
        if len(oot_data) > 1:
            gaps = oot_data.index.to_series().diff().dt.days
            max_gap = gaps.max()
            if max_gap > 45:  # 30 trading days ~ 45 calendar days
                problematic_tickers.append((ticker, f"large gap of {max_gap} cal days"))

    if problematic_tickers:
        print(f"  Problematic tickers found:")
        for ticker, reason in problematic_tickers:
            print(f"    {ticker}: {reason}")

        # Remove problematic tickers and re-run
        problem_names = [t[0] for t in problematic_tickers]
        clean_tickers = [t for t in EARNINGS_UNIVERSE if t not in problem_names]
        adjusted_trades = run_strategy(close, opn, regime_s, gap_direction="up",
                                       ticker_subset=clean_tickers)
        adj_metrics = compute_metrics(adjusted_trades)
    else:
        print("  No problematic tickers found — universe appears survivorship-free.")
        adj_metrics = orig_metrics
        adjusted_trades = original_trades

    adjusted_sharpe = adj_metrics['sharpe']
    sharpe_change = abs(adjusted_sharpe - orig_metrics['sharpe']) / orig_metrics['sharpe'] \
        if orig_metrics['sharpe'] > 0 else 0

    # Pass if results don't materially change (within 15%)
    passed = sharpe_change < 0.15 or len(problematic_tickers) == 0

    if len(problematic_tickers) == 0:
        detail = ("PASS: No survivorship bias detected. All 24 tickers have continuous "
                  "data throughout the OOT period (Jan 2022 – Jul 2026).")
    elif passed:
        detail = (f"PASS: After removing {len(problematic_tickers)} problematic ticker(s), "
                  f"Sharpe changes from {orig_metrics['sharpe']:.3f} to {adjusted_sharpe:.3f} "
                  f"({sharpe_change:.0%} change, <15% threshold).")
    else:
        detail = (f"FAIL: After removing {len(problematic_tickers)} problematic ticker(s), "
                  f"Sharpe changes from {orig_metrics['sharpe']:.3f} to {adjusted_sharpe:.3f} "
                  f"({sharpe_change:.0%} change, >15% threshold). Survivorship bias likely.")

    print(f"  Original Sharpe: {orig_metrics['sharpe']:.3f}")
    print(f"  Adjusted Sharpe: {adjusted_sharpe:.3f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "details": detail,
        "adjusted_sharpe": adjusted_sharpe,
        "original_sharpe": orig_metrics['sharpe'],
        "sharpe_change_pct": round(sharpe_change, 4),
        "problematic_tickers": [(t, r) for t, r in problematic_tickers],
        "n_problematic": len(problematic_tickers),
    }


# ─── Test 5: Regime-Split Stability ────────────────────────────────────────
def test_regime_split(close, opn, regime_s):
    """
    Sub-period analysis: 2022, 2023, 2024, 2025, 2026-H1.
    Check if strategy works in 2022 specifically (hardest year).
    """
    print("\n" + "=" * 70)
    print("TEST 5: REGIME-SPLIT / SUB-PERIOD STABILITY")
    print("=" * 70)

    # Run full strategy
    all_trades = run_strategy(close, opn, regime_s, gap_direction="up")
    full_metrics = compute_metrics(all_trades)

    # Define sub-periods
    sub_periods = {
        "2022": ("2022-01-01", "2022-12-31"),
        "2023": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026_H1": ("2026-01-01", "2026-07-28"),
    }

    sub_results = {}
    negative_periods = 0
    total_tested = 0

    for period_name, (start, end) in sub_periods.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        period_trades = [t for t in all_trades
                        if start_ts <= t['entry_date'] <= end_ts]

        if len(period_trades) < 3:
            sub_results[period_name] = {
                "sharpe": None, "n_trades": len(period_trades),
                "note": "Insufficient trades for analysis"
            }
            continue

        pm = compute_metrics(period_trades)
        sub_results[period_name] = {
            "sharpe": pm['sharpe'],
            "sortino": pm['sortino'],
            "pf": pm['pf'],
            "wr": pm['wr'],
            "n_trades": pm['n_trades'],
            "total_return": pm['total_return'],
        }

        total_tested += 1
        if pm['sharpe'] < 0:
            negative_periods += 1

        bull_count = sum(1 for t in period_trades if t['regime'] == 'bull')
        bear_count = sum(1 for t in period_trades if t['regime'] == 'bear')

        print(f"  {period_name}: Sharpe={pm['sharpe']:>7.3f}  PF={pm['pf']:>6.3f}  "
              f"WR={pm['wr']:>5.1%}  N={pm['n_trades']:>3d}  "
              f"(bull={bull_count}, bear={bear_count})")

    # 2022 specifically
    sharpe_2022 = sub_results.get("2022", {}).get("sharpe", None)
    print(f"\n  2022 (hardest year) Sharpe: {sharpe_2022}")

    # Pass criteria:
    # - No more than 1 sub-period with negative Sharpe
    # - 2022 must not be catastrophic (Sharpe > -0.5)
    # - Overall regime balance already verified (Bull 0.921, Bear 1.091)
    catastrophic_2022 = sharpe_2022 is not None and sharpe_2022 < -0.5
    too_many_negative = negative_periods > 1

    passed = not catastrophic_2022 and not too_many_negative

    if passed:
        detail = (f"PASS: Strategy is stable across sub-periods. "
                  f"{negative_periods}/{total_tested} periods with negative Sharpe. "
                  f"2022 Sharpe: {sharpe_2022}. "
                  f"Bull Sharpe: {full_metrics['sharpe_bull']:.3f}, "
                  f"Bear Sharpe: {full_metrics['sharpe_bear']:.3f}.")
    else:
        reasons = []
        if catastrophic_2022:
            reasons.append(f"2022 Sharpe={sharpe_2022:.3f} is catastrophic (<-0.5)")
        if too_many_negative:
            reasons.append(f"{negative_periods}/{total_tested} periods have negative Sharpe")
        detail = f"FAIL: {'; '.join(reasons)}"

    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        "passed": passed,
        "details": detail,
        "sub_periods": sub_results,
        "n_negative_periods": negative_periods,
        "total_periods_tested": total_tested,
        "sharpe_2022": sharpe_2022,
        "full_sharpe_bull": full_metrics['sharpe_bull'],
        "full_sharpe_bear": full_metrics['sharpe_bear'],
    }


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ANALYSTC_H1 ADVERSARIAL VALIDATION — 5-CHECK FRAMEWORK")
    print(f"Run date: {dt.datetime.now().isoformat()}")
    print("=" * 70)

    # Download data
    close, opn = download_data()

    # Compute regime
    spy_close = close["SPY"].dropna()
    regime_s = compute_regime(spy_close).to_dict()

    # Baseline
    print("\n--- BASELINE ---")
    baseline_trades = run_strategy(close, opn, regime_s, gap_direction="up")
    baseline = compute_metrics(baseline_trades)
    print(f"  Baseline: Sharpe={baseline['sharpe']:.3f}, Sortino={baseline['sortino']:.3f}, "
          f"PF={baseline['pf']:.3f}, WR={baseline['wr']:.1%}, N={baseline['n_trades']}, "
          f"MaxDD={baseline['max_dd']:.2%}")
    print(f"  Bull Sharpe={baseline['sharpe_bull']:.3f}, Bear Sharpe={baseline['sharpe_bear']:.3f}")

    # Run all 5 tests
    results = {}

    results["inverse_direction"] = test_inverse_direction(close, opn, regime_s)
    results["random_timing"] = test_random_timing(close, opn, regime_s)
    results["top_ticker_removal"] = test_top_ticker_removal(close, opn, regime_s)
    results["survivorship_bias"] = test_survivorship_bias(close, opn, regime_s)
    results["regime_split"] = test_regime_split(close, opn, regime_s)

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)

    pass_count = 0
    for test_name, result in results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {test_name}")
        if result["passed"]:
            pass_count += 1

    overall_pass = pass_count >= 4  # Require 4/5 to consider valid
    print(f"\n  Overall: {pass_count}/5 tests passed — {'VALID' if overall_pass else 'CONCERNS REMAIN'}")

    if not results["inverse_direction"]["passed"]:
        print("  *** CRITICAL: Inverse direction test FAILED — this is the most important test ***")
        print("  *** The gap-up signal may be irrelevant if gap-down also works ***")

    # Save results
    output = {
        "strategy": "AnalystC_H1",
        "run_date": dt.datetime.now().isoformat(),
        "baseline_metrics": baseline,
        "tests": {},
        "overall_pass": overall_pass,
        "pass_count": f"{pass_count}/5",
    }

    # Clean results for JSON serialization
    for test_name, result in results.items():
        clean = {}
        for k, v in result.items():
            if isinstance(v, (np.floating, np.float64)):
                clean[k] = float(v)
            elif isinstance(v, (np.integer, np.int64)):
                clean[k] = int(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, dict):
                clean[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, (np.floating, np.float64)):
                        clean[k][kk] = float(vv)
                    elif isinstance(vv, (np.integer, np.int64)):
                        clean[k][kk] = int(vv)
                    elif isinstance(vv, (np.bool_,)):
                        clean[k][kk] = bool(vv)
                    elif isinstance(vv, pd.Timestamp):
                        clean[k][kk] = str(vv)
                    else:
                        clean[k][kk] = vv
            elif isinstance(v, pd.Timestamp):
                clean[k] = str(v)
            else:
                clean[k] = v
        output["tests"][test_name] = clean

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")
    print("Done.")

    return pass_count, overall_pass


if __name__ == "__main__":
    pass_count, overall = main()
    sys.exit(0 if overall else 1)
