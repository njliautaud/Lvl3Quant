#!/usr/bin/env python3
"""
Optimal Trade Timing Study
==========================
Studies validated strategies to find optimal entry timing — day-of-week,
time-of-month, market condition filters that maximize edge.

Strategies tested:
  1. ETF Short-Term Reversal (buy bottom-5 ETFs by 5-day return, hold 5 days)
  2. VIX Panic Buying (buy SPY when VIX > 30)
  3. UPRO Leveraged Growth (VIX-threshold gating)

Validation (mandatory):
  - Permutation test (100 shuffles) for every enhancement claim
  - R1 regime gap < 0.50
  - Sub-period consistency (first half vs second half)
  - Enhanced vs baseline comparison

Rules: sliding window only, Sharpe/Sortino/PF/WR metrics, honest verdicts.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

np.random.seed(42)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/optimal_timing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_RATE = 0.05
N_PERM = 100

ETF_UNIVERSE = [
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "SPY", "QQQ", "IWM", "MDY", "EFA", "EEM",
    "TLT", "IEF", "HYG", "LQD", "GLD",
]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(returns, name=""):
    """Compute Sharpe, Sortino, PF, WR from a return series."""
    returns = returns.dropna()
    if len(returns) < 20:
        return None

    cum = (1 + returns).cumprod()
    years = max(len(returns) / 252, 0.1)
    cagr = cum.iloc[-1] ** (1 / years) - 1 if cum.iloc[-1] > 0 else -1.0

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (cagr - RISK_FREE_RATE) / downside if downside > 0 else 0

    gross_gains = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    pf = gross_gains / gross_losses if gross_losses > 0 else float('inf')

    wr = (returns > 0).mean()

    rolling_max = cum.cummax()
    max_dd = (cum / rolling_max - 1).min()

    return {
        "name": name,
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "profit_factor": round(pf, 4),
        "win_rate": round(wr, 4),
        "max_dd": round(max_dd, 4),
        "n_days": len(returns),
        "total_return": round(cum.iloc[-1] - 1, 4),
    }


def permutation_test(strategy_returns, baseline_returns, n_perm=N_PERM, metric="sharpe"):
    """Test if strategy significantly beats baseline via permutation."""
    strat_m = compute_metrics(strategy_returns)
    base_m = compute_metrics(baseline_returns)
    if strat_m is None or base_m is None:
        return 1.0, 0.0

    observed_diff = strat_m[metric] - base_m[metric]

    combined = pd.concat([strategy_returns, baseline_returns]).values
    n_strat = len(strategy_returns)

    count_ge = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_strat = pd.Series(combined[:n_strat])
        perm_base = pd.Series(combined[n_strat:])
        pm_s = compute_metrics(perm_strat)
        pm_b = compute_metrics(perm_base)
        if pm_s is not None and pm_b is not None:
            perm_diff = pm_s[metric] - pm_b[metric]
            if perm_diff >= observed_diff:
                count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)
    return p_value, observed_diff


def regime_test(returns, spy_returns):
    """R1 regime-agnostic test: |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    # Classify days by SPY close-to-close
    bull_mask = spy_returns > 0
    bear_mask = spy_returns <= 0

    # Align
    common = returns.index.intersection(spy_returns.index)
    ret_c = returns.loc[common]
    bull_c = bull_mask.loc[common]
    bear_c = bear_mask.loc[common]

    bull_ret = ret_c[bull_c]
    bear_ret = ret_c[bear_c]

    m_bull = compute_metrics(bull_ret, "bull")
    m_bear = compute_metrics(bear_ret, "bear")

    if m_bull is None or m_bear is None:
        return None, None, None

    s_bull = m_bull["sharpe"]
    s_bear = m_bear["sharpe"]
    denom = max(abs(s_bull), abs(s_bear), 0.001)
    gap = abs(s_bull - s_bear) / denom

    return gap, s_bull, s_bear


def sub_period_test(returns):
    """Split returns in half, check consistency."""
    n = len(returns)
    half = n // 2
    m1 = compute_metrics(returns.iloc[:half], "first_half")
    m2 = compute_metrics(returns.iloc[half:], "second_half")
    if m1 is None or m2 is None:
        return None, None
    return m1, m2


def full_validation(enhanced_returns, baseline_returns, spy_returns, label=""):
    """Run all validation checks. Return dict with pass/fail."""
    result = {"label": label}

    # 1. Metrics comparison
    m_enh = compute_metrics(enhanced_returns, f"{label}_enhanced")
    m_base = compute_metrics(baseline_returns, f"{label}_baseline")
    result["enhanced_metrics"] = m_enh
    result["baseline_metrics"] = m_base

    if m_enh is None or m_base is None:
        result["verdict"] = "FAIL"
        result["reason"] = "Insufficient data"
        return result

    # 2. Does enhanced beat baseline?
    sharpe_improvement = m_enh["sharpe"] - m_base["sharpe"]
    result["sharpe_improvement"] = round(sharpe_improvement, 4)

    # 3. Permutation test
    p_val, obs_diff = permutation_test(enhanced_returns, baseline_returns)
    result["permutation_p"] = round(p_val, 4)
    result["permutation_pass"] = p_val < 0.05

    # 4. Regime test
    gap, s_bull, s_bear = regime_test(enhanced_returns, spy_returns)
    result["regime_gap"] = round(gap, 4) if gap is not None else None
    result["regime_sharpe_bull"] = round(s_bull, 4) if s_bull is not None else None
    result["regime_sharpe_bear"] = round(s_bear, 4) if s_bear is not None else None
    result["regime_pass"] = gap is not None and gap < 0.50

    # 5. Sub-period consistency
    m1, m2 = sub_period_test(enhanced_returns)
    if m1 and m2:
        result["subperiod_first_sharpe"] = m1["sharpe"]
        result["subperiod_second_sharpe"] = m2["sharpe"]
        # Both halves should be positive
        result["subperiod_pass"] = m1["sharpe"] > 0 and m2["sharpe"] > 0
    else:
        result["subperiod_pass"] = False

    # Overall verdict
    beats_baseline = sharpe_improvement > 0
    perm_pass = result["permutation_pass"]
    regime_pass = result["regime_pass"]
    subperiod_pass = result["subperiod_pass"]

    if beats_baseline and perm_pass and regime_pass and subperiod_pass:
        result["verdict"] = "PASS"
    elif beats_baseline and (perm_pass or regime_pass):
        result["verdict"] = "MARGINAL"
        reasons = []
        if not perm_pass:
            reasons.append(f"perm p={result['permutation_p']}")
        if not regime_pass:
            reasons.append(f"regime gap={result['regime_gap']}")
        if not subperiod_pass:
            reasons.append("subperiod inconsistent")
        result["reason"] = "; ".join(reasons)
    else:
        result["verdict"] = "FAIL"
        reasons = []
        if not beats_baseline:
            reasons.append("doesn't beat baseline")
        if not perm_pass:
            reasons.append(f"perm p={result['permutation_p']}")
        if not regime_pass:
            reasons.append(f"regime gap={result['regime_gap']}")
        result["reason"] = "; ".join(reasons)

    return result


def print_validation(v):
    """Print validation result clearly."""
    verdict_emoji = {"PASS": "[PASS]", "MARGINAL": "[MARGINAL]", "FAIL": "[FAIL]"}
    print(f"\n  {verdict_emoji.get(v['verdict'], '???')} {v['label']}")

    if v.get("enhanced_metrics") and v.get("baseline_metrics"):
        em = v["enhanced_metrics"]
        bm = v["baseline_metrics"]
        print(f"    Enhanced:  Sharpe={em['sharpe']:.3f}  Sortino={em['sortino']:.3f}  PF={em['profit_factor']:.3f}  WR={em['win_rate']:.1%}")
        print(f"    Baseline:  Sharpe={bm['sharpe']:.3f}  Sortino={bm['sortino']:.3f}  PF={bm['profit_factor']:.3f}  WR={bm['win_rate']:.1%}")
        print(f"    Sharpe improvement: {v.get('sharpe_improvement', 0):+.4f}")

    print(f"    Permutation p={v.get('permutation_p', 'N/A')} ({'PASS' if v.get('permutation_pass') else 'FAIL'})")
    print(f"    Regime gap={v.get('regime_gap', 'N/A')} ({'PASS' if v.get('regime_pass') else 'FAIL'})")

    if v.get("subperiod_first_sharpe") is not None:
        print(f"    Sub-period: H1 Sharpe={v['subperiod_first_sharpe']:.3f}, H2 Sharpe={v['subperiod_second_sharpe']:.3f} ({'PASS' if v.get('subperiod_pass') else 'FAIL'})")

    if v.get("reason"):
        print(f"    Reason: {v['reason']}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_all_data():
    """Download all required data."""
    print("=" * 80)
    print("DOWNLOADING DATA")
    print("=" * 80)

    # ETFs for reversal strategy
    all_tickers = ETF_UNIVERSE + ["^VIX", "^VIX3M", "UPRO"]
    all_tickers = list(set(all_tickers))  # dedupe

    print(f"Downloading {len(all_tickers)} tickers...")

    prices = {}
    batch_size = 5
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start="2003-01-01", end="2026-07-17",
                               auto_adjust=True, progress=False, threads=False, timeout=30)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    col = t
                    if col in data["Close"].columns:
                        s = data["Close"][col].dropna()
                        if len(s) > 100:
                            prices[t] = s
            else:
                t = batch[0]
                s = data["Close"].dropna()
                if len(s) > 100:
                    prices[t] = s
        except Exception as e:
            print(f"    Error: {e}")

    df = pd.DataFrame(prices).sort_index()
    df = df.ffill()

    print(f"\nData: {len(df)} days, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Tickers with data: {sorted(df.columns.tolist())}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: ETF SHORT-TERM REVERSAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_reversal_baseline(prices, lookback=5, n_buy=5, hold=5):
    """Baseline reversal: buy bottom N ETFs by lookback-day return, hold for hold days."""
    etf_cols = [c for c in ETF_UNIVERSE if c in prices.columns]
    etf_prices = prices[etf_cols].dropna(how="all")

    returns_lookback = etf_prices.pct_change(lookback)
    daily_returns = etf_prices.pct_change(1)

    portfolio_returns = []

    # Trade every `hold` days (non-overlapping)
    trade_dates = etf_prices.index[lookback::hold]

    for entry_date in trade_dates:
        if entry_date not in returns_lookback.index:
            continue

        # Get lookback returns, pick bottom N
        lb_ret = returns_lookback.loc[entry_date].dropna()
        if len(lb_ret) < n_buy:
            continue

        losers = lb_ret.nsmallest(n_buy).index.tolist()

        # Hold for `hold` days
        entry_idx = etf_prices.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold, len(etf_prices) - 1)

        if exit_idx <= entry_idx:
            continue

        # Equal-weight portfolio return over holding period
        hold_slice = daily_returns.iloc[entry_idx + 1: exit_idx + 1]
        valid_losers = [l for l in losers if l in hold_slice.columns]
        if not valid_losers:
            continue

        port_daily = hold_slice[valid_losers].mean(axis=1)
        for dt, r in port_daily.items():
            portfolio_returns.append((dt, r))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_series = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_series.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def run_reversal_dow_filter(prices, target_dow, lookback=5, n_buy=5, hold=5):
    """Only enter reversal trades on a specific day of week."""
    etf_cols = [c for c in ETF_UNIVERSE if c in prices.columns]
    etf_prices = prices[etf_cols].dropna(how="all")

    returns_lookback = etf_prices.pct_change(lookback)
    daily_returns = etf_prices.pct_change(1)

    portfolio_returns = []
    trade_dates = etf_prices.index[lookback::hold]

    for entry_date in trade_dates:
        if entry_date not in returns_lookback.index:
            continue

        # Filter by day of week
        if entry_date.dayofweek != target_dow:
            continue

        lb_ret = returns_lookback.loc[entry_date].dropna()
        if len(lb_ret) < n_buy:
            continue

        losers = lb_ret.nsmallest(n_buy).index.tolist()

        entry_idx = etf_prices.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold, len(etf_prices) - 1)
        if exit_idx <= entry_idx:
            continue

        hold_slice = daily_returns.iloc[entry_idx + 1: exit_idx + 1]
        valid_losers = [l for l in losers if l in hold_slice.columns]
        if not valid_losers:
            continue

        port_daily = hold_slice[valid_losers].mean(axis=1)
        for dt, r in port_daily.items():
            portfolio_returns.append((dt, r))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_series = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_series.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def run_reversal_month_week_filter(prices, target_week, lookback=5, n_buy=5, hold=5):
    """Only enter reversal trades during a specific week of the month (1-5)."""
    etf_cols = [c for c in ETF_UNIVERSE if c in prices.columns]
    etf_prices = prices[etf_cols].dropna(how="all")

    returns_lookback = etf_prices.pct_change(lookback)
    daily_returns = etf_prices.pct_change(1)

    portfolio_returns = []
    trade_dates = etf_prices.index[lookback::hold]

    for entry_date in trade_dates:
        if entry_date not in returns_lookback.index:
            continue

        # Week of month: day 1-7 = week 1, 8-14 = week 2, etc.
        week_of_month = (entry_date.day - 1) // 7 + 1
        if week_of_month != target_week:
            continue

        lb_ret = returns_lookback.loc[entry_date].dropna()
        if len(lb_ret) < n_buy:
            continue

        losers = lb_ret.nsmallest(n_buy).index.tolist()

        entry_idx = etf_prices.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold, len(etf_prices) - 1)
        if exit_idx <= entry_idx:
            continue

        hold_slice = daily_returns.iloc[entry_idx + 1: exit_idx + 1]
        valid_losers = [l for l in losers if l in hold_slice.columns]
        if not valid_losers:
            continue

        port_daily = hold_slice[valid_losers].mean(axis=1)
        for dt, r in port_daily.items():
            portfolio_returns.append((dt, r))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_series = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_series.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def run_reversal_vix_filter(prices, vix_threshold, direction="below", lookback=5, n_buy=5, hold=5):
    """Only enter reversal trades when VIX is above/below threshold."""
    etf_cols = [c for c in ETF_UNIVERSE if c in prices.columns]
    etf_prices = prices[etf_cols].dropna(how="all")
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None

    if vix is None:
        return pd.Series(dtype=float)

    returns_lookback = etf_prices.pct_change(lookback)
    daily_returns = etf_prices.pct_change(1)

    portfolio_returns = []
    trade_dates = etf_prices.index[lookback::hold]

    for entry_date in trade_dates:
        if entry_date not in returns_lookback.index:
            continue
        if entry_date not in vix.index:
            continue

        current_vix = vix.loc[entry_date]
        if direction == "below" and current_vix >= vix_threshold:
            continue
        elif direction == "above" and current_vix < vix_threshold:
            continue

        lb_ret = returns_lookback.loc[entry_date].dropna()
        if len(lb_ret) < n_buy:
            continue

        losers = lb_ret.nsmallest(n_buy).index.tolist()

        entry_idx = etf_prices.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold, len(etf_prices) - 1)
        if exit_idx <= entry_idx:
            continue

        hold_slice = daily_returns.iloc[entry_idx + 1: exit_idx + 1]
        valid_losers = [l for l in losers if l in hold_slice.columns]
        if not valid_losers:
            continue

        port_daily = hold_slice[valid_losers].mean(axis=1)
        for dt, r in port_daily.items():
            portfolio_returns.append((dt, r))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_series = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_series.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def run_reversal_trend_filter(prices, lookback=5, n_buy=5, hold=5, sma_period=50):
    """Only enter reversal trades when SPY > its SMA (uptrend)."""
    etf_cols = [c for c in ETF_UNIVERSE if c in prices.columns]
    etf_prices = prices[etf_cols].dropna(how="all")
    spy = prices["SPY"].dropna() if "SPY" in prices.columns else None

    if spy is None:
        return pd.Series(dtype=float)

    spy_sma = spy.rolling(sma_period).mean()

    returns_lookback = etf_prices.pct_change(lookback)
    daily_returns = etf_prices.pct_change(1)

    portfolio_returns = []
    trade_dates = etf_prices.index[lookback::hold]

    for entry_date in trade_dates:
        if entry_date not in returns_lookback.index:
            continue
        if entry_date not in spy.index or entry_date not in spy_sma.index:
            continue

        # Only enter if SPY above 50 SMA (uptrend)
        if spy.loc[entry_date] <= spy_sma.loc[entry_date]:
            continue

        lb_ret = returns_lookback.loc[entry_date].dropna()
        if len(lb_ret) < n_buy:
            continue

        losers = lb_ret.nsmallest(n_buy).index.tolist()

        entry_idx = etf_prices.index.get_loc(entry_date)
        exit_idx = min(entry_idx + hold, len(etf_prices) - 1)
        if exit_idx <= entry_idx:
            continue

        hold_slice = daily_returns.iloc[entry_idx + 1: exit_idx + 1]
        valid_losers = [l for l in losers if l in hold_slice.columns]
        if not valid_losers:
            continue

        port_daily = hold_slice[valid_losers].mean(axis=1)
        for dt, r in port_daily.items():
            portfolio_returns.append((dt, r))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_series = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_series.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def study_reversal_timing(prices):
    """Run all reversal timing tests."""
    print("\n" + "=" * 80)
    print("STRATEGY 1: ETF SHORT-TERM REVERSAL — TIMING STUDY")
    print("=" * 80)

    spy_returns = prices["SPY"].pct_change(1).dropna() if "SPY" in prices.columns else None

    # Baseline
    print("\n--- Baseline (no filters) ---")
    baseline = run_reversal_baseline(prices)
    m_base = compute_metrics(baseline, "baseline")
    if m_base:
        print(f"  Sharpe={m_base['sharpe']:.3f}  Sortino={m_base['sortino']:.3f}  PF={m_base['profit_factor']:.3f}  WR={m_base['win_rate']:.1%}  CAGR={m_base['cagr']:.1%}  MaxDD={m_base['max_dd']:.1%}")
    else:
        print("  ERROR: Baseline produced insufficient data")
        return []

    results = []

    # Test 1: Day of Week
    print("\n--- Test 1: Day-of-Week Entry Filter ---")
    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    for dow in range(5):
        enhanced = run_reversal_dow_filter(prices, dow)
        if len(enhanced) < 50:
            print(f"  {dow_names[dow]}: insufficient data ({len(enhanced)} days)")
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"reversal_dow_{dow_names[dow]}")
        print_validation(v)
        results.append(v)

    # Test 2: Week of Month
    print("\n--- Test 2: Week-of-Month Entry Filter ---")
    for week in [1, 2, 3, 4]:
        enhanced = run_reversal_month_week_filter(prices, week)
        if len(enhanced) < 50:
            print(f"  Week {week}: insufficient data ({len(enhanced)} days)")
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"reversal_week{week}_of_month")
        print_validation(v)
        results.append(v)

    # Test 3: VIX Level Filter
    print("\n--- Test 3: VIX Level Entry Filter ---")
    for vix_thresh in [15, 20, 25, 30]:
        for direction in ["below", "above"]:
            label = f"reversal_vix_{direction}_{vix_thresh}"
            enhanced = run_reversal_vix_filter(prices, vix_thresh, direction)
            if len(enhanced) < 50:
                print(f"  VIX {direction} {vix_thresh}: insufficient data ({len(enhanced)} days)")
                continue

            v = full_validation(enhanced, baseline, spy_returns, label)
            print_validation(v)
            results.append(v)

    # Test 4: Trend Filter (SPY > 50 SMA)
    print("\n--- Test 4: Trend Filter (SPY > 50 SMA) ---")
    enhanced = run_reversal_trend_filter(prices)
    if len(enhanced) >= 50:
        v = full_validation(enhanced, baseline, spy_returns, "reversal_spy_above_50sma")
        print_validation(v)
        results.append(v)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: VIX PANIC BUYING
# ═══════════════════════════════════════════════════════════════════════════════

def run_panic_baseline(prices, vix_threshold=30, hold_days=20):
    """Baseline: buy SPY on close of day VIX > threshold, hold for N days."""
    spy = prices["SPY"].dropna()
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None

    if vix is None:
        return pd.Series(dtype=float)

    common = spy.index.intersection(vix.index)
    spy = spy.loc[common]
    vix = vix.loc[common]
    spy_ret = spy.pct_change(1)

    portfolio_returns = []
    in_trade = False
    trade_exit_idx = 0

    for i in range(1, len(common)):
        dt = common[i]
        idx = i

        if in_trade and idx >= trade_exit_idx:
            in_trade = False

        if not in_trade and vix.iloc[i] > vix_threshold:
            # Enter trade
            in_trade = True
            trade_exit_idx = min(idx + hold_days, len(common) - 1)

        if in_trade:
            portfolio_returns.append((dt, spy_ret.iloc[i]))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_df = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_df.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def run_panic_entry_timing(prices, entry_mode="cross", hold_days=20):
    """
    Test different entry timing within VIX panic episodes.
    entry_mode:
      - "cross": enter day VIX crosses above 30
      - "peak": enter day VIX peaks (starts declining within episode)
      - "decline": enter when VIX starts declining from peak (VIX < yesterday's VIX)
    """
    spy = prices["SPY"].dropna()
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None

    if vix is None:
        return pd.Series(dtype=float)

    common = spy.index.intersection(vix.index)
    spy = spy.loc[common]
    vix = vix.loc[common]
    spy_ret = spy.pct_change(1)

    # Identify panic episodes (VIX > 30)
    panic_mask = vix > 30

    portfolio_returns = []
    in_trade = False
    trade_exit_idx = 0
    triggered = False

    for i in range(2, len(common)):
        dt = common[i]
        idx = i

        if in_trade and idx >= trade_exit_idx:
            in_trade = False
            triggered = False

        if not in_trade and not triggered:
            if entry_mode == "cross":
                # Enter the day VIX crosses above 30
                if panic_mask.iloc[i] and not panic_mask.iloc[i - 1]:
                    in_trade = True
                    trade_exit_idx = min(idx + hold_days, len(common) - 1)
                    triggered = True

            elif entry_mode == "peak":
                # Enter when VIX has been > 30 and starts declining
                if panic_mask.iloc[i] and vix.iloc[i] < vix.iloc[i - 1] and vix.iloc[i - 1] >= vix.iloc[i - 2]:
                    in_trade = True
                    trade_exit_idx = min(idx + hold_days, len(common) - 1)
                    triggered = True

            elif entry_mode == "decline":
                # Enter after VIX has been declining for 2 consecutive days from > 30
                if panic_mask.iloc[i] and vix.iloc[i] < vix.iloc[i - 1] < vix.iloc[i - 2]:
                    in_trade = True
                    trade_exit_idx = min(idx + hold_days, len(common) - 1)
                    triggered = True

        if in_trade:
            portfolio_returns.append((dt, spy_ret.iloc[i]))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_df = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_df.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def run_panic_with_term_structure(prices, hold_days=20, ratio_threshold=1.0):
    """
    Entry filter: only buy panic when VIX/VIX3M ratio > threshold
    (backwardation = extreme fear = better entry).
    """
    spy = prices["SPY"].dropna()
    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None
    vix3m = prices["^VIX3M"].dropna() if "^VIX3M" in prices.columns else None

    if vix is None:
        return pd.Series(dtype=float)

    if vix3m is None:
        # Fall back: can't compute term structure
        print("    (VIX3M not available, skipping term structure test)")
        return pd.Series(dtype=float)

    common = spy.index.intersection(vix.index).intersection(vix3m.index)
    spy = spy.loc[common]
    vix = vix.loc[common]
    vix3m_c = vix3m.loc[common]
    spy_ret = spy.pct_change(1)

    ratio = vix / vix3m_c

    portfolio_returns = []
    in_trade = False
    trade_exit_idx = 0

    for i in range(1, len(common)):
        dt = common[i]
        idx = i

        if in_trade and idx >= trade_exit_idx:
            in_trade = False

        if not in_trade and vix.iloc[i] > 30 and ratio.iloc[i] > ratio_threshold:
            in_trade = True
            trade_exit_idx = min(idx + hold_days, len(common) - 1)

        if in_trade:
            portfolio_returns.append((dt, spy_ret.iloc[i]))

    if not portfolio_returns:
        return pd.Series(dtype=float)

    ret_df = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_df.groupby("date")["return"].mean()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series.sort_index()


def study_panic_timing(prices):
    """Run all VIX panic timing tests."""
    print("\n" + "=" * 80)
    print("STRATEGY 2: VIX PANIC BUYING — TIMING STUDY")
    print("=" * 80)

    spy_returns = prices["SPY"].pct_change(1).dropna()

    # Baseline
    print("\n--- Baseline (buy SPY when VIX > 30, hold 20 days) ---")
    baseline = run_panic_baseline(prices, vix_threshold=30, hold_days=20)
    m_base = compute_metrics(baseline, "baseline")
    if m_base:
        print(f"  Sharpe={m_base['sharpe']:.3f}  Sortino={m_base['sortino']:.3f}  PF={m_base['profit_factor']:.3f}  WR={m_base['win_rate']:.1%}  N_days={m_base['n_days']}")
    else:
        print("  ERROR: Baseline produced insufficient data")
        return []

    results = []

    # Test 1: Holding Period Optimization
    print("\n--- Test 1: Holding Period Optimization ---")
    for hold in [3, 5, 10, 15, 20, 30]:
        enhanced = run_panic_baseline(prices, vix_threshold=30, hold_days=hold)
        if len(enhanced) < 20:
            print(f"  Hold {hold}d: insufficient data")
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"panic_hold_{hold}d")
        print_validation(v)
        results.append(v)

    # Test 2: Entry Timing within Panic
    print("\n--- Test 2: Entry Timing within Panic ---")
    for mode in ["cross", "peak", "decline"]:
        enhanced = run_panic_entry_timing(prices, entry_mode=mode, hold_days=20)
        if len(enhanced) < 20:
            print(f"  Entry mode '{mode}': insufficient data ({len(enhanced)} days)")
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"panic_entry_{mode}")
        print_validation(v)
        results.append(v)

    # Test 3: VIX Term Structure Filter
    print("\n--- Test 3: VIX Term Structure (VIX/VIX3M ratio) ---")
    for ratio_thresh in [0.9, 1.0, 1.1, 1.2]:
        enhanced = run_panic_with_term_structure(prices, hold_days=20, ratio_threshold=ratio_thresh)
        if len(enhanced) < 20:
            print(f"  VIX/VIX3M > {ratio_thresh}: insufficient data")
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"panic_termstructure_gt_{ratio_thresh}")
        print_validation(v)
        results.append(v)

    # Test 4: VIX Threshold Levels
    print("\n--- Test 4: VIX Threshold Levels ---")
    for vix_thresh in [25, 30, 35, 40]:
        enhanced = run_panic_baseline(prices, vix_threshold=vix_thresh, hold_days=20)
        if len(enhanced) < 20:
            print(f"  VIX > {vix_thresh}: insufficient data")
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"panic_vix_gt_{vix_thresh}")
        print_validation(v)
        results.append(v)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: UPRO LEVERAGED GROWTH
# ═══════════════════════════════════════════════════════════════════════════════

def run_upro_baseline(prices, vix_low=17, vix_high=25, rebalance_freq="daily"):
    """
    Baseline UPRO strategy:
      - VIX < vix_low: 100% UPRO
      - vix_low <= VIX < vix_high: 50% UPRO, 50% cash (0 return)
      - VIX >= vix_high: 100% cash
    Rebalance frequency: daily, weekly, or monthly.
    """
    # Simulate UPRO as 3x SPY daily returns if UPRO not available
    if "UPRO" in prices.columns:
        upro = prices["UPRO"].dropna()
        upro_ret = upro.pct_change(1)
    else:
        spy = prices["SPY"].dropna()
        upro_ret = spy.pct_change(1) * 3  # Simulated 3x

    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None
    if vix is None:
        return pd.Series(dtype=float)

    common = upro_ret.index.intersection(vix.index)
    upro_ret = upro_ret.loc[common]
    vix = vix.loc[common]

    # Determine rebalance dates
    if rebalance_freq == "daily":
        rebal_mask = pd.Series(True, index=common)
    elif rebalance_freq == "weekly":
        rebal_mask = pd.Series(common.dayofweek == 0, index=common)  # Monday
        rebal_mask.iloc[0] = True
    elif rebalance_freq == "monthly":
        month_change = pd.Series(common.month, index=common).diff() != 0
        month_change.iloc[0] = True
        rebal_mask = month_change
    else:
        rebal_mask = pd.Series(True, index=common)

    portfolio_returns = []
    current_alloc = 0.0  # fraction in UPRO

    for i in range(1, len(common)):
        dt = common[i]

        if rebal_mask.iloc[i]:
            v = vix.iloc[i]
            if v < vix_low:
                current_alloc = 1.0
            elif v < vix_high:
                current_alloc = 0.5
            else:
                current_alloc = 0.0

        port_ret = current_alloc * upro_ret.iloc[i]
        portfolio_returns.append((dt, port_ret))

    ret_df = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_df.set_index("date")["return"]
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series


def run_upro_dow_rebalance(prices, target_dow, vix_low=17, vix_high=25):
    """Rebalance only on specific day of week."""
    if "UPRO" in prices.columns:
        upro = prices["UPRO"].dropna()
        upro_ret = upro.pct_change(1)
    else:
        spy = prices["SPY"].dropna()
        upro_ret = spy.pct_change(1) * 3

    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None
    if vix is None:
        return pd.Series(dtype=float)

    common = upro_ret.index.intersection(vix.index)
    upro_ret = upro_ret.loc[common]
    vix = vix.loc[common]

    portfolio_returns = []
    current_alloc = 0.0

    for i in range(1, len(common)):
        dt = common[i]

        if dt.dayofweek == target_dow:
            v = vix.iloc[i]
            if v < vix_low:
                current_alloc = 1.0
            elif v < vix_high:
                current_alloc = 0.5
            else:
                current_alloc = 0.0

        port_ret = current_alloc * upro_ret.iloc[i]
        portfolio_returns.append((dt, port_ret))

    ret_df = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_df.set_index("date")["return"]
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series


def run_upro_with_trend(prices, vix_low=17, vix_high=25, sma_period=50):
    """
    Enhanced UPRO: add trend filter.
    Only go full UPRO if VIX < 17 AND SPY > 50 SMA.
    If SPY < 50 SMA, reduce allocation by half.
    """
    if "UPRO" in prices.columns:
        upro = prices["UPRO"].dropna()
        upro_ret = upro.pct_change(1)
    else:
        spy = prices["SPY"].dropna()
        upro_ret = spy.pct_change(1) * 3

    vix = prices["^VIX"].dropna() if "^VIX" in prices.columns else None
    spy = prices["SPY"].dropna()
    spy_sma = spy.rolling(sma_period).mean()

    if vix is None:
        return pd.Series(dtype=float)

    common = upro_ret.index.intersection(vix.index).intersection(spy_sma.dropna().index)
    upro_ret = upro_ret.loc[common]
    vix_c = vix.loc[common]
    spy_c = spy.loc[common]
    sma_c = spy_sma.loc[common]

    portfolio_returns = []

    for i in range(1, len(common)):
        dt = common[i]
        v = vix_c.iloc[i]
        trend_up = spy_c.iloc[i] > sma_c.iloc[i]

        if v >= vix_high:
            alloc = 0.0
        elif v >= vix_low:
            alloc = 0.5 if trend_up else 0.25
        else:
            alloc = 1.0 if trend_up else 0.5

        port_ret = alloc * upro_ret.iloc[i]
        portfolio_returns.append((dt, port_ret))

    ret_df = pd.DataFrame(portfolio_returns, columns=["date", "return"])
    ret_series = ret_df.set_index("date")["return"]
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    return ret_series


def study_upro_timing(prices):
    """Run all UPRO timing tests."""
    print("\n" + "=" * 80)
    print("STRATEGY 3: UPRO LEVERAGED GROWTH — TIMING STUDY")
    print("=" * 80)

    # Filter to 2010+ for UPRO relevance
    prices_upro = prices.loc["2010-01-01":]

    spy_returns = prices_upro["SPY"].pct_change(1).dropna()

    # Baseline
    print("\n--- Baseline (daily rebalance, VIX thresholds 17/25) ---")
    baseline = run_upro_baseline(prices_upro, vix_low=17, vix_high=25, rebalance_freq="daily")
    m_base = compute_metrics(baseline, "baseline")
    if m_base:
        print(f"  Sharpe={m_base['sharpe']:.3f}  Sortino={m_base['sortino']:.3f}  PF={m_base['profit_factor']:.3f}  WR={m_base['win_rate']:.1%}  CAGR={m_base['cagr']:.1%}  MaxDD={m_base['max_dd']:.1%}")
    else:
        print("  ERROR: Baseline produced insufficient data")
        return []

    results = []

    # Test 1: Day-of-Week Rebalancing
    print("\n--- Test 1: Day-of-Week for Rebalancing ---")
    dow_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    for dow in range(5):
        enhanced = run_upro_dow_rebalance(prices_upro, dow)
        if len(enhanced) < 200:
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"upro_rebal_dow_{dow_names[dow]}")
        print_validation(v)
        results.append(v)

    # Test 2: Rebalance Frequency
    print("\n--- Test 2: Rebalance Frequency ---")
    for freq in ["daily", "weekly", "monthly"]:
        enhanced = run_upro_baseline(prices_upro, vix_low=17, vix_high=25, rebalance_freq=freq)
        if len(enhanced) < 200:
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"upro_rebal_freq_{freq}")
        print_validation(v)
        results.append(v)

    # Test 3: Trend Filter (SPY > SMA)
    print("\n--- Test 3: Trend Filter (SPY > SMA) ---")
    for sma in [20, 50, 100, 200]:
        enhanced = run_upro_with_trend(prices_upro, sma_period=sma)
        if len(enhanced) < 200:
            continue

        v = full_validation(enhanced, baseline, spy_returns, f"upro_trend_sma{sma}")
        print_validation(v)
        results.append(v)

    # Test 4: VIX Threshold Variations
    print("\n--- Test 4: VIX Threshold Variations ---")
    for vix_low in [15, 17, 20]:
        for vix_high in [22, 25, 30]:
            if vix_low >= vix_high:
                continue
            enhanced = run_upro_baseline(prices_upro, vix_low=vix_low, vix_high=vix_high, rebalance_freq="daily")
            if len(enhanced) < 200:
                continue

            v = full_validation(enhanced, baseline, spy_returns, f"upro_vix_{vix_low}_{vix_high}")
            print_validation(v)
            results.append(v)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("OPTIMAL TRADE TIMING STUDY")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Download data
    prices = download_all_data()

    # Run all three strategy studies
    all_results = {}

    reversal_results = study_reversal_timing(prices)
    all_results["reversal"] = reversal_results

    panic_results = study_panic_timing(prices)
    all_results["panic"] = panic_results

    upro_results = study_upro_timing(prices)
    all_results["upro"] = upro_results

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — OPTIMAL TIMING STUDY")
    print("=" * 80)

    for strategy_name, strategy_results in all_results.items():
        print(f"\n{'─' * 40}")
        print(f"  {strategy_name.upper()}")
        print(f"{'─' * 40}")

        passes = [r for r in strategy_results if r.get("verdict") == "PASS"]
        marginals = [r for r in strategy_results if r.get("verdict") == "MARGINAL"]
        fails = [r for r in strategy_results if r.get("verdict") == "FAIL"]

        print(f"  Total tests: {len(strategy_results)}")
        print(f"  PASS: {len(passes)}  |  MARGINAL: {len(marginals)}  |  FAIL: {len(fails)}")

        if passes:
            print(f"\n  Genuine improvements (PASS):")
            for p in passes:
                em = p.get("enhanced_metrics", {})
                print(f"    {p['label']}: Sharpe={em.get('sharpe', 'N/A'):.3f}, improvement={p.get('sharpe_improvement', 0):+.4f}")

        if marginals:
            print(f"\n  Marginal improvements (some validation failed):")
            for m in marginals:
                em = m.get("enhanced_metrics", {})
                print(f"    {m['label']}: Sharpe={em.get('sharpe', 'N/A'):.3f}, reason={m.get('reason', '')}")

        if not passes and not marginals:
            print(f"\n  NO timing filters beat the baseline with statistical significance.")
            print(f"  Verdict: The baseline strategy timing is already near-optimal.")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────

    # Flatten for JSON
    def safe_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    summary = {
        "study": "optimal_timing",
        "run_date": datetime.now().isoformat(),
        "strategies": {}
    }

    for strategy_name, strategy_results in all_results.items():
        summary["strategies"][strategy_name] = {
            "n_tests": len(strategy_results),
            "n_pass": len([r for r in strategy_results if r.get("verdict") == "PASS"]),
            "n_marginal": len([r for r in strategy_results if r.get("verdict") == "MARGINAL"]),
            "n_fail": len([r for r in strategy_results if r.get("verdict") == "FAIL"]),
            "tests": strategy_results,
        }

    json_path = OUTPUT_DIR / "optimal_timing_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=safe_json)
    print(f"\nSaved summary to {json_path}")

    # CSV with all test results
    rows = []
    for strategy_name, strategy_results in all_results.items():
        for r in strategy_results:
            row = {"strategy": strategy_name, "label": r.get("label", "")}
            row["verdict"] = r.get("verdict", "")
            row["reason"] = r.get("reason", "")
            em = r.get("enhanced_metrics", {})
            bm = r.get("baseline_metrics", {})
            for k in ["sharpe", "sortino", "profit_factor", "win_rate", "cagr", "max_dd", "n_days"]:
                row[f"enhanced_{k}"] = em.get(k, "")
                row[f"baseline_{k}"] = bm.get(k, "")
            row["sharpe_improvement"] = r.get("sharpe_improvement", "")
            row["permutation_p"] = r.get("permutation_p", "")
            row["regime_gap"] = r.get("regime_gap", "")
            row["subperiod_first_sharpe"] = r.get("subperiod_first_sharpe", "")
            row["subperiod_second_sharpe"] = r.get("subperiod_second_sharpe", "")
            rows.append(row)

    csv_path = OUTPUT_DIR / "optimal_timing_details.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved details to {csv_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
