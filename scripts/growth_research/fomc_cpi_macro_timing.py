#!/usr/bin/env python3
"""
FOMC Drift + CPI Reaction Macro Timing Strategy
=================================================
Exploits two well-documented academic anomalies:

1. PRE-FOMC DRIFT: SPY tends to drift up 3-5 trading days before FOMC
   announcements. Academic ref: Lucca & Moench (2015, Fed NY Staff Report).
   Buy at close T-5, sell at close T+1 (day after announcement).

2. CPI SURPRISE REACTION: Position based on whether CPI comes in above/below
   consensus. Strong CPI → sell (rate hike fear). Weak CPI → buy (dovish hope).
   We use a simpler version: buy SPY on day of CPI release (market tends to
   overreact to CPI, then mean-revert) OR position based on trailing CPI trend.

3. COMBINED: Enter only when both FOMC and CPI windows overlap or when
   the macro calendar is dense (FOMC week + recent CPI release).

Vehicles: SPY (unleveraged) or UPRO (3x leveraged, optional).
Period: 2004-2026 (CPI surprise data reliable from ~2004).

Walk-forward: 36-month train, 6-month test, sliding.
Adversarial: permutation test, sub-period consistency, outlier removal, regime test.
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/fomc_cpi_macro_timing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# FOMC ANNOUNCEMENT DATES (last day of each meeting, when statement released)
# These are the ANNOUNCEMENT dates, not multi-day meeting dates.
# Source: Federal Reserve Board calendar
# ──────────────────────────────────────────────────────────────────────
FOMC_ANNOUNCEMENT_DATES = [
    # 2004
    "2004-01-28","2004-03-16","2004-05-04","2004-06-30","2004-08-10",
    "2004-09-21","2004-11-10","2004-12-14",
    # 2005
    "2005-02-02","2005-03-22","2005-05-03","2005-06-30","2005-08-09",
    "2005-09-20","2005-11-01","2005-12-13",
    # 2006
    "2006-01-31","2006-03-28","2006-05-10","2006-06-29","2006-08-08",
    "2006-09-20","2006-10-25","2006-12-12",
    # 2007
    "2007-01-31","2007-03-21","2007-05-09","2007-06-28","2007-08-07",
    "2007-09-18","2007-10-31","2007-12-11",
    # 2008
    "2008-01-22","2008-01-30","2008-03-18","2008-04-30","2008-06-25",
    "2008-08-05","2008-09-16","2008-10-08","2008-10-29","2008-12-16",
    # 2009
    "2009-01-28","2009-03-18","2009-04-29","2009-06-24","2009-08-12",
    "2009-09-23","2009-11-04","2009-12-16",
    # 2010
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10",
    "2010-09-21","2010-11-03","2010-12-14",
    # 2011
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09",
    "2011-09-21","2011-11-02","2011-12-13",
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01",
    "2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31",
    "2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30",
    "2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29",
    "2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-17",
    # 2026
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29",
]

# ──────────────────────────────────────────────────────────────────────
# CPI RELEASE DATES (BLS publishes ~13th of month for prior month)
# Source: BLS schedule. These are the actual release dates.
# ──────────────────────────────────────────────────────────────────────
CPI_RELEASE_DATES = [
    # 2004
    "2004-01-16","2004-02-20","2004-03-17","2004-04-14","2004-05-14",
    "2004-06-15","2004-07-14","2004-08-19","2004-09-14","2004-10-19",
    "2004-11-17","2004-12-15",
    # 2005
    "2005-01-14","2005-02-23","2005-03-23","2005-04-20","2005-05-18",
    "2005-06-14","2005-07-15","2005-08-16","2005-09-15","2005-10-14",
    "2005-11-16","2005-12-14",
    # 2006
    "2006-01-18","2006-02-22","2006-03-15","2006-04-19","2006-05-17",
    "2006-06-14","2006-07-19","2006-08-16","2006-09-15","2006-10-18",
    "2006-11-15","2006-12-15",
    # 2007
    "2007-01-18","2007-02-21","2007-03-15","2007-04-17","2007-05-15",
    "2007-06-15","2007-07-18","2007-08-14","2007-09-19","2007-10-17",
    "2007-11-15","2007-12-14",
    # 2008
    "2008-01-16","2008-02-20","2008-03-14","2008-04-16","2008-05-14",
    "2008-06-13","2008-07-16","2008-08-14","2008-09-16","2008-10-16",
    "2008-11-19","2008-12-16",
    # 2009
    "2009-01-16","2009-02-20","2009-03-18","2009-04-15","2009-05-15",
    "2009-06-17","2009-07-15","2009-08-14","2009-09-16","2009-10-15",
    "2009-11-18","2009-12-16",
    # 2010
    "2010-01-15","2010-02-19","2010-03-18","2010-04-14","2010-05-19",
    "2010-06-17","2010-07-16","2010-08-13","2010-09-17","2010-10-15",
    "2010-11-17","2010-12-15",
    # 2011
    "2011-01-14","2011-02-17","2011-03-17","2011-04-15","2011-05-13",
    "2011-06-15","2011-07-15","2011-08-18","2011-09-15","2011-10-19",
    "2011-11-16","2011-12-16",
    # 2012
    "2012-01-19","2012-02-17","2012-03-16","2012-04-13","2012-05-15",
    "2012-06-14","2012-07-17","2012-08-15","2012-09-14","2012-10-16",
    "2012-11-15","2012-12-14",
    # 2013
    "2013-01-16","2013-02-21","2013-03-15","2013-04-16","2013-05-16",
    "2013-06-18","2013-07-16","2013-08-15","2013-09-17","2013-10-30",
    "2013-11-20","2013-12-17",
    # 2014
    "2014-01-16","2014-02-20","2014-03-18","2014-04-15","2014-05-15",
    "2014-06-17","2014-07-22","2014-08-19","2014-09-17","2014-10-22",
    "2014-11-20","2014-12-17",
    # 2015
    "2015-01-16","2015-02-26","2015-03-24","2015-04-17","2015-05-22",
    "2015-06-18","2015-07-17","2015-08-19","2015-09-16","2015-10-15",
    "2015-11-17","2015-12-15",
    # 2016
    "2016-01-20","2016-02-19","2016-03-16","2016-04-14","2016-05-17",
    "2016-06-16","2016-07-15","2016-08-16","2016-09-16","2016-10-18",
    "2016-11-17","2016-12-15",
    # 2017
    "2017-01-18","2017-02-15","2017-03-15","2017-04-14","2017-05-12",
    "2017-06-14","2017-07-14","2017-08-11","2017-09-14","2017-10-13",
    "2017-11-15","2017-12-13",
    # 2018
    "2018-01-12","2018-02-14","2018-03-13","2018-04-11","2018-05-10",
    "2018-06-12","2018-07-12","2018-08-10","2018-09-13","2018-10-11",
    "2018-11-14","2018-12-12",
    # 2019
    "2019-01-11","2019-02-13","2019-03-12","2019-04-10","2019-05-10",
    "2019-06-12","2019-07-11","2019-08-13","2019-09-12","2019-10-10",
    "2019-11-13","2019-12-11",
    # 2020
    "2020-01-14","2020-02-13","2020-03-11","2020-04-10","2020-05-12",
    "2020-06-10","2020-07-14","2020-08-12","2020-09-11","2020-10-13",
    "2020-11-12","2020-12-10",
    # 2021
    "2021-01-13","2021-02-10","2021-03-10","2021-04-13","2021-05-12",
    "2021-06-10","2021-07-13","2021-08-11","2021-09-14","2021-10-13",
    "2021-11-10","2021-12-10",
    # 2022
    "2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11",
    "2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13",
    "2022-11-10","2022-12-13",
    # 2023
    "2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10",
    "2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12",
    "2023-11-14","2023-12-12",
    # 2024
    "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15",
    "2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10",
    "2024-11-13","2024-12-11",
    # 2025
    "2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13",
    "2025-06-11","2025-07-15",
    # 2026
    "2026-01-14","2026-02-11","2026-03-11","2026-04-14","2026-05-12",
    "2026-06-10","2026-07-14",
]


# ──────────────────────────────────────────────────────────────────────
# Data download
# ──────────────────────────────────────────────────────────────────────
def download_data(ticker="SPY", start="2003-01-01", end="2026-07-17"):
    """Download price data via yfinance."""
    print(f"Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    # Remove timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    print(f"  Got {len(df)} trading days: {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Signal generators
# ──────────────────────────────────────────────────────────────────────

def find_trading_days_before(target_date, trading_days, n_days):
    """Find the n trading days before target_date."""
    td_before = [td for td in trading_days if td < pd.Timestamp(target_date)]
    return td_before[-n_days:] if len(td_before) >= n_days else td_before


def find_trading_days_after(target_date, trading_days, n_days):
    """Find the n trading days on or after target_date."""
    td_after = [td for td in trading_days if td >= pd.Timestamp(target_date)]
    return td_after[:n_days] if len(td_after) >= n_days else td_after


def signal_pre_fomc_drift(dates, trading_days, entry_days_before=5, hold_days_after=1):
    """
    Pre-FOMC Drift: Long SPY from T-entry_days_before through T+hold_days_after.
    Academic anomaly: equities drift up before FOMC announcements.
    """
    signals = pd.Series(0, index=dates)
    td_list = sorted(trading_days)

    fomc_dates_ts = [pd.Timestamp(d) for d in FOMC_ANNOUNCEMENT_DATES]

    for fomc_d in fomc_dates_ts:
        # Entry window: T-N to T-1
        pre_days = find_trading_days_before(fomc_d, td_list, entry_days_before)
        for d in pre_days:
            if d in signals.index:
                signals[d] = 1

        # Hold through announcement day and T+1
        post_days = find_trading_days_after(fomc_d, td_list, hold_days_after + 1)
        for d in post_days:
            if d in signals.index:
                signals[d] = 1

    return signals


def signal_fomc_announcement_day(dates, trading_days):
    """
    FOMC Announcement Day Only: Long on the day of the FOMC announcement.
    Tests if the announcement day itself has excess returns.
    """
    signals = pd.Series(0, index=dates)
    fomc_dates_ts = [pd.Timestamp(d) for d in FOMC_ANNOUNCEMENT_DATES]
    td_set = set(trading_days)

    for fomc_d in fomc_dates_ts:
        # Map to nearest trading day
        if fomc_d in td_set:
            signals[fomc_d] = 1
        else:
            # Find next trading day
            for td in sorted(trading_days):
                if td >= fomc_d:
                    signals[td] = 1
                    break

    return signals


def signal_cpi_release_day(dates, trading_days):
    """
    CPI Release Day: Long on CPI release day.
    Tests if CPI release days have systematic returns.
    """
    signals = pd.Series(0, index=dates)
    cpi_dates_ts = [pd.Timestamp(d) for d in CPI_RELEASE_DATES]
    td_set = set(trading_days)

    for cpi_d in cpi_dates_ts:
        if cpi_d in td_set:
            signals[cpi_d] = 1
        else:
            for td in sorted(trading_days):
                if td >= cpi_d:
                    signals[td] = 1
                    break

    return signals


def signal_cpi_post_reaction(dates, trading_days, hold_days=3):
    """
    CPI Post-Release: Long for hold_days after CPI release.
    Tests mean-reversion after CPI overreaction.
    """
    signals = pd.Series(0, index=dates)
    td_list = sorted(trading_days)
    cpi_dates_ts = [pd.Timestamp(d) for d in CPI_RELEASE_DATES]

    for cpi_d in cpi_dates_ts:
        post_days = find_trading_days_after(cpi_d, td_list, hold_days + 1)
        # Skip the release day itself, take next hold_days
        if len(post_days) > 1:
            for d in post_days[1:]:  # Skip day 0 (release day)
                if d in signals.index:
                    signals[d] = 1

    return signals


def signal_fomc_cpi_cluster(dates, trading_days, window_days=5):
    """
    FOMC+CPI Cluster: Long when both FOMC and CPI fall within window_days of each other.
    These "dense macro weeks" tend to have higher vol and drift.
    Signal fires for the entire window around the cluster.
    """
    signals = pd.Series(0, index=dates)
    td_list = sorted(trading_days)
    fomc_dates_ts = sorted([pd.Timestamp(d) for d in FOMC_ANNOUNCEMENT_DATES])
    cpi_dates_ts = sorted([pd.Timestamp(d) for d in CPI_RELEASE_DATES])

    # Find FOMC-CPI pairs within window_days of each other
    for fomc_d in fomc_dates_ts:
        for cpi_d in cpi_dates_ts:
            gap = abs((fomc_d - cpi_d).days)
            if gap <= window_days:
                # Mark days from min(fomc, cpi) - 2 to max(fomc, cpi) + 1
                start_d = min(fomc_d, cpi_d)
                end_d = max(fomc_d, cpi_d)
                pre = find_trading_days_before(start_d, td_list, 2)
                post = find_trading_days_after(end_d, td_list, 2)
                for d in pre + post:
                    if d in signals.index:
                        signals[d] = 1
                # Also mark days between
                for td in td_list:
                    if start_d <= td <= end_d and td in signals.index:
                        signals[td] = 1

    return signals


def signal_pre_fomc_tight(dates, trading_days):
    """
    Tight Pre-FOMC: Long only T-3 to T-1 before FOMC. More selective version.
    """
    return signal_pre_fomc_drift(dates, trading_days, entry_days_before=3, hold_days_after=0)


def signal_macro_quiet_week(dates, trading_days, buffer_days=7):
    """
    Macro Quiet Week: Long when NO FOMC or CPI within buffer_days.
    Inverse hypothesis: maybe quiet weeks are better for equities.
    """
    signals = pd.Series(1, index=dates)  # Default long
    td_list = sorted(trading_days)
    all_macro = sorted(
        [pd.Timestamp(d) for d in FOMC_ANNOUNCEMENT_DATES] +
        [pd.Timestamp(d) for d in CPI_RELEASE_DATES]
    )

    for macro_d in all_macro:
        # Kill signal around macro events
        for td in td_list:
            if abs((td - macro_d).days) <= buffer_days and td in signals.index:
                signals[td] = 0

    return signals


# ──────────────────────────────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(returns, ann_factor=252):
    """Compute standard performance metrics from a return series."""
    if len(returns) < 10 or returns.std() == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0,
            "maxdd": 0.0, "wr": 0.5, "pf": 1.0,
            "n_trades": len(returns), "avg_ret": 0.0
        }

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret > 0 else 0

    downside = returns[returns < 0].std()
    sortino = (mean_ret / downside) * np.sqrt(ann_factor) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    n_years = len(returns) / ann_factor
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) * 100 if n_years > 0 and cum.iloc[-1] > 0 else 0

    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    maxdd = drawdown.min() * 100

    wr = (returns > 0).mean()
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 2),
        "maxdd": round(maxdd, 2),
        "wr": round(wr, 4),
        "pf": round(pf, 3),
        "n_trades": len(returns),
        "avg_ret": round(mean_ret * 10000, 2),  # bps
    }


# ──────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ──────────────────────────────────────────────────────────────────────
def walk_forward_test(df, signal_series, train_months=36, test_months=6):
    """
    Walk-forward validation (sliding window, per HC #0).
    Training: check if signal days have positive mean return.
    If yes: go long on signal days in test window.
    Returns concatenated OOT returns.
    """
    all_oot_returns = []

    start_date = df.index[0]
    end_date = df.index[-1]
    current_start = start_date

    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if train_end > end_date:
            break

        train_mask = (df.index >= current_start) & (df.index < train_end)
        test_mask = (df.index >= train_end) & (df.index < test_end)

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(train_data) < 100 or len(test_data) < 20:
            current_start += pd.DateOffset(months=test_months)
            continue

        # Training: check if signal days have positive returns
        train_signal = signal_series.reindex(train_data.index, fill_value=0)
        signal_returns = train_data["Return"][train_signal == 1]

        if len(signal_returns) > 5 and signal_returns.mean() > 0:
            # Deploy in test window
            test_signal = signal_series.reindex(test_data.index, fill_value=0)
            oot_returns = test_data["Return"][test_signal == 1]
            all_oot_returns.append(oot_returns)

        current_start += pd.DateOffset(months=test_months)

    if all_oot_returns:
        return pd.concat(all_oot_returns)
    return pd.Series(dtype=float)


# ──────────────────────────────────────────────────────────────────────
# Adversarial validation gates (HC #705)
# ──────────────────────────────────────────────────────────────────────

def permutation_test(df, signal_series, n_perms=200, train_months=36, test_months=6):
    """Shuffle signal dates, keep returns. Compute Sharpe p-value."""
    real_returns = walk_forward_test(df, signal_series, train_months, test_months)
    if len(real_returns) < 10:
        return 1.0, 0.0

    real_sharpe = compute_metrics(real_returns)["sharpe"]

    better_count = 0
    for i in range(n_perms):
        shuffled = signal_series.copy()
        shuffled_values = shuffled.values.copy()
        np.random.shuffle(shuffled_values)
        shuffled = pd.Series(shuffled_values, index=signal_series.index)

        perm_returns = walk_forward_test(df, shuffled, train_months, test_months)
        if len(perm_returns) >= 10:
            perm_sharpe = compute_metrics(perm_returns)["sharpe"]
            if perm_sharpe >= real_sharpe:
                better_count += 1

    p_value = better_count / n_perms
    return p_value, real_sharpe


def sub_period_consistency(returns):
    """Split into 2+ non-overlapping blocks, check consistency."""
    if len(returns) < 40:
        return {"half1_sharpe": 0, "half2_sharpe": 0, "consistent": False, "both_positive": False}

    # Split into 3 blocks for robustness
    n = len(returns)
    b1 = returns.iloc[:n//3]
    b2 = returns.iloc[n//3:2*n//3]
    b3 = returns.iloc[2*n//3:]

    m1 = compute_metrics(b1)
    m2 = compute_metrics(b2)
    m3 = compute_metrics(b3)

    # Require at least 2 of 3 blocks positive
    positives = sum(1 for m in [m1, m2, m3] if m["sharpe"] > 0)
    both_positive = positives >= 2

    # Also check standard 2-block
    mid = n // 2
    h1_m = compute_metrics(returns.iloc[:mid])
    h2_m = compute_metrics(returns.iloc[mid:])

    return {
        "half1_sharpe": h1_m["sharpe"],
        "half2_sharpe": h2_m["sharpe"],
        "block1_sharpe": m1["sharpe"],
        "block2_sharpe": m2["sharpe"],
        "block3_sharpe": m3["sharpe"],
        "consistent": h1_m["sharpe"] > 0 and h2_m["sharpe"] > 0,
        "both_positive": both_positive,
        "positive_blocks": f"{positives}/3",
    }


def outlier_removal_test(returns, n_remove=5):
    """Remove top N return days and re-check metrics."""
    if len(returns) < 20:
        return compute_metrics(returns)

    sorted_abs = returns.abs().sort_values(ascending=False)
    outlier_idx = sorted_abs.index[:n_remove]
    cleaned = returns.drop(outlier_idx)
    return compute_metrics(cleaned)


def regime_test(df, signal_series, train_months=36, test_months=6):
    """
    R1 regime test: classify days by TRAILING 20-day SPY regime.
    Check gap < 0.50 per HC #428.
    """
    oot_returns = walk_forward_test(df, signal_series, train_months, test_months)
    if len(oot_returns) < 30:
        return {"regime_gap": 1.0, "pass": False, "green_sharpe": 0, "red_sharpe": 0, "flat_sharpe": 0}

    # TRAILING 20-day return (known at open, no lookahead)
    trailing_ret = df["Close"].pct_change(20).reindex(oot_returns.index)
    green = oot_returns[trailing_ret > 0.02]
    red = oot_returns[trailing_ret < -0.02]
    flat = oot_returns[(trailing_ret >= -0.02) & (trailing_ret <= 0.02)]

    green_sharpe = compute_metrics(green)["sharpe"] if len(green) > 10 else 0
    red_sharpe = compute_metrics(red)["sharpe"] if len(red) > 10 else 0
    flat_sharpe = compute_metrics(flat)["sharpe"] if len(flat) > 10 else 0

    max_abs = max(abs(green_sharpe), abs(red_sharpe), 0.001)
    regime_gap = abs(green_sharpe - red_sharpe) / max_abs

    return {
        "regime_gap": round(regime_gap, 3),
        "pass": regime_gap <= 0.50,
        "green_sharpe": green_sharpe,
        "red_sharpe": red_sharpe,
        "flat_sharpe": flat_sharpe,
        "n_green": len(green),
        "n_red": len(red),
        "n_flat": len(flat),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    print("=" * 90)
    print("FOMC DRIFT + CPI REACTION MACRO TIMING STRATEGY")
    print("Walk-Forward Validation with Adversarial Checks (HC #705)")
    print("=" * 90)
    print()

    # Download data
    df = download_data("SPY", start="2003-06-01", end="2026-07-17")
    trading_days = df.index.tolist()

    print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} trading days)")
    bh_metrics = compute_metrics(df["Return"])
    print(f"Buy & Hold: Sharpe={bh_metrics['sharpe']}, Sortino={bh_metrics['sortino']}, "
          f"CAGR={bh_metrics['cagr']:.1f}%, MaxDD={bh_metrics['maxdd']:.1f}%")
    print()

    # ──────────────────────────────────────────────────────────────
    # Generate all signals
    # ──────────────────────────────────────────────────────────────
    print("Generating signals...")
    strategies = {}

    strategies["Pre-FOMC 5d"] = signal_pre_fomc_drift(df.index, trading_days, 5, 1)
    strategies["Pre-FOMC 3d (tight)"] = signal_pre_fomc_tight(df.index, trading_days)
    strategies["FOMC Day Only"] = signal_fomc_announcement_day(df.index, trading_days)
    strategies["CPI Release Day"] = signal_cpi_release_day(df.index, trading_days)
    strategies["CPI Post-Reaction 3d"] = signal_cpi_post_reaction(df.index, trading_days, 3)
    strategies["FOMC+CPI Cluster"] = signal_fomc_cpi_cluster(df.index, trading_days, 5)
    strategies["Macro Quiet Week"] = signal_macro_quiet_week(df.index, trading_days, 7)

    for name, sig in strategies.items():
        n_days = int(sig.sum())
        pct = n_days / len(sig) * 100
        print(f"  {name:25s}: {n_days:5d} signal days ({pct:.1f}%)")

    # ──────────────────────────────────────────────────────────────
    # Walk-forward + adversarial testing
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("WALK-FORWARD RESULTS (36-month train, 6-month test, sliding)")
    print("=" * 90)

    results = {}

    for name, sig in strategies.items():
        print(f"\n{'─' * 70}")
        print(f"  STRATEGY: {name}")
        print(f"{'─' * 70}")

        oot_returns = walk_forward_test(df, sig)
        if len(oot_returns) < 10:
            print(f"  Insufficient OOT data ({len(oot_returns)} days). SKIP.")
            results[name] = {"status": "SKIP", "reason": "insufficient data"}
            continue

        metrics = compute_metrics(oot_returns)
        print(f"  OOT Days:  {metrics['n_trades']}")
        print(f"  Sharpe:    {metrics['sharpe']:+.3f}")
        print(f"  Sortino:   {metrics['sortino']:+.3f}")
        print(f"  CAGR:      {metrics['cagr']:+.2f}%")
        print(f"  MaxDD:     {metrics['maxdd']:.2f}%")
        print(f"  Win Rate:  {metrics['wr']:.1%}")
        print(f"  Profit F.: {metrics['pf']:.3f}")
        print(f"  Avg Ret:   {metrics['avg_ret']:+.2f} bps/day")

        # ── Adversarial checks ──
        print(f"\n  ADVERSARIAL CHECKS (HC #705):")

        # 1. Permutation test (200 shuffles)
        print(f"    [1] Permutation test (200 shuffles)...", end=" ", flush=True)
        p_value, real_sharpe = permutation_test(df, sig, n_perms=200)
        perm_pass = p_value < 0.05
        print(f"p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

        # 2. Sub-period consistency
        sub = sub_period_consistency(oot_returns)
        sub_pass = sub["both_positive"]
        print(f"    [2] Sub-period: H1={sub['half1_sharpe']:+.3f}, H2={sub['half2_sharpe']:+.3f}, "
              f"Blocks={sub['positive_blocks']} {'PASS' if sub_pass else 'FAIL'}")

        # 3. Outlier removal
        cleaned = outlier_removal_test(oot_returns, n_remove=5)
        outlier_pass = cleaned["sharpe"] > 0
        print(f"    [3] Outlier removal (top 5): Sharpe={cleaned['sharpe']:+.3f} "
              f"{'PASS' if outlier_pass else 'FAIL'}")

        # 4. Regime test (R1)
        regime = regime_test(df, sig)
        regime_pass = regime["pass"]
        print(f"    [4] Regime (R1): gap={regime['regime_gap']:.3f} "
              f"(G={regime['green_sharpe']:+.3f} n={regime.get('n_green',0)}, "
              f"R={regime['red_sharpe']:+.3f} n={regime.get('n_red',0)}, "
              f"F={regime['flat_sharpe']:+.3f} n={regime.get('n_flat',0)}) "
              f"{'PASS' if regime_pass else 'FAIL'}")

        gates = sum([perm_pass, sub_pass, outlier_pass, regime_pass])
        all_pass = gates == 4

        if all_pass:
            banner = "*** ALL 4 GATES PASSED ***"
        else:
            failed = []
            if not perm_pass: failed.append("Perm")
            if not sub_pass: failed.append("SubPeriod")
            if not outlier_pass: failed.append("Outlier")
            if not regime_pass: failed.append("Regime")
            banner = f"FAILED {4-gates}/4 gates ({', '.join(failed)})"

        print(f"\n  VERDICT: {banner}")

        results[name] = {
            "status": "PASS" if all_pass else "FAIL",
            "gates_passed": f"{gates}/4",
            "metrics": metrics,
            "perm_p": round(p_value, 4),
            "perm_pass": perm_pass,
            "sub_period": sub,
            "sub_pass": sub_pass,
            "cleaned_sharpe": cleaned["sharpe"],
            "outlier_pass": outlier_pass,
            "regime": regime,
            "regime_pass": regime_pass,
        }

    # ──────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 130)
    print("SUMMARY TABLE — FOMC + CPI MACRO TIMING")
    print("=" * 130)
    header = (f"{'Strategy':<26} {'Status':<7} {'Gates':<7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'CAGR%':>7} {'MaxDD%':>7} {'WR':>6} {'PF':>6} {'Perm-p':>7} {'SubPrd':>7} {'Regime':>7}")
    print(header)
    print("─" * 130)

    for name, res in results.items():
        if res["status"] == "SKIP":
            print(f"{name:<26} {'SKIP':<7}")
            continue
        m = res["metrics"]
        sp = res["sub_period"]
        rg = res["regime"]
        print(f"{name:<26} {res['status']:<7} {res['gates_passed']:<7} "
              f"{m['sharpe']:>+7.3f} {m['sortino']:>+8.3f} "
              f"{m['cagr']:>+7.2f} {m['maxdd']:>7.2f} "
              f"{m['wr']:>6.1%} {m['pf']:>6.3f} "
              f"{res['perm_p']:>7.3f} "
              f"{'P' if res['sub_pass'] else 'F':>7} "
              f"{'P' if res['regime_pass'] else 'F':>7}")

    # ──────────────────────────────────────────────────────────────
    # PASS / FAIL banner
    # ──────────────────────────────────────────────────────────────
    passed = [n for n, r in results.items() if r.get("status") == "PASS"]
    failed = [n for n, r in results.items() if r.get("status") == "FAIL"]

    print("\n" + "=" * 90)
    if passed:
        print("STRATEGIES THAT PASSED ALL 4 ADVERSARIAL GATES:")
        for name in passed:
            m = results[name]["metrics"]
            print(f"  {name}: Sharpe={m['sharpe']:+.3f}, Sortino={m['sortino']:+.3f}, "
                  f"CAGR={m['cagr']:+.2f}%, WR={m['wr']:.1%}")
    else:
        print("NO STRATEGIES PASSED ALL 4 GATES.")
        print("(This is informative — academic pre-FOMC drift may not survive adversarial testing.)")

    if failed:
        print(f"\nFailed strategies ({len(failed)}):")
        for name in failed:
            print(f"  {name}: gates={results[name]['gates_passed']}")

    print("=" * 90)

    # ──────────────────────────────────────────────────────────────
    # B&H comparison (exposure-adjusted)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("EXPOSURE-ADJUSTED COMPARISON vs BUY & HOLD")
    print("=" * 90)
    for name, res in results.items():
        if res.get("status") == "SKIP":
            continue
        m = res["metrics"]
        n_signal = m["n_trades"]
        total_days = len(df)
        exposure = n_signal / total_days * 100
        # Sharpe already accounts for vol, so compare directly
        sharpe_premium = m["sharpe"] - bh_metrics["sharpe"]
        print(f"  {name:<26}: Exposure={exposure:5.1f}%, "
              f"Sharpe={m['sharpe']:+.3f} (vs B&H {bh_metrics['sharpe']:+.3f}, "
              f"premium={sharpe_premium:+.3f})")

    print()

    # Save results
    save_path = OUTPUT_DIR / "results.json"
    # Convert for JSON
    json_results = {}
    for name, res in results.items():
        jr = {}
        for k, v in res.items():
            if isinstance(v, dict):
                jr[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                         for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                jr[k] = float(v)
            else:
                jr[k] = v
        json_results[name] = jr

    with open(save_path, "w") as f:
        json.dump({
            "strategy": "FOMC Drift + CPI Reaction Macro Timing",
            "date_run": str(dt.datetime.now()),
            "data_range": f"{df.index[0].date()} to {df.index[-1].date()}",
            "buy_hold": bh_metrics,
            "results": json_results,
        }, f, indent=2, default=str)

    print(f"Results saved to {save_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
