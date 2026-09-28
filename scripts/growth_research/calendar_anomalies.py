#!/usr/bin/env python3
"""
Calendar Anomaly Strategy — Walk-Forward Validation
=====================================================
Tests 7 calendar anomalies on SPY (2000–2026) with adversarial checks.

Anomalies:
  1. Turn-of-Month (TOM): days -1 to +3
  2. Pre-Holiday drift
  3. FOMC pre-meeting drift (1-3 days before)
  4. Monthly Seasonality (Nov-Apr vs May-Oct)
  5. Day-of-Week (Monday/Friday effects)
  6. Triple Witching week
  7. Combined signal (3+ anomalies firing)

Walk-forward: 36-month train, 6-month test, sliding.
Adversarial: permutation test, sub-period consistency, outlier removal, regime test.
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/calendar_anomalies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# FOMC meeting dates (2000-2026) — actual dates from Federal Reserve
# ──────────────────────────────────────────────────────────────────────
FOMC_DATES = [
    # 2000
    "2000-02-01","2000-02-02","2000-03-21","2000-05-16","2000-06-27","2000-06-28",
    "2000-08-22","2000-10-03","2000-11-15","2000-12-19",
    # 2001
    "2001-01-03","2001-01-30","2001-01-31","2001-03-20","2001-04-18",
    "2001-05-15","2001-06-26","2001-06-27","2001-08-21","2001-09-17",
    "2001-10-02","2001-11-06","2001-12-11",
    # 2002
    "2002-01-29","2002-01-30","2002-03-19","2002-05-07","2002-06-25","2002-06-26",
    "2002-08-13","2002-09-24","2002-11-06","2002-12-10",
    # 2003
    "2003-01-28","2003-01-29","2003-03-18","2003-05-06","2003-06-24","2003-06-25",
    "2003-08-12","2003-09-16","2003-10-28","2003-12-09",
    # 2004
    "2004-01-27","2004-01-28","2004-03-16","2004-05-04","2004-06-29","2004-06-30",
    "2004-08-10","2004-09-21","2004-11-10","2004-12-14",
    # 2005
    "2005-02-01","2005-02-02","2005-03-22","2005-05-03","2005-06-29","2005-06-30",
    "2005-08-09","2005-09-20","2005-11-01","2005-12-13",
    # 2006
    "2006-01-31","2006-03-27","2006-03-28","2006-05-10","2006-06-28","2006-06-29",
    "2006-08-08","2006-09-20","2006-10-24","2006-10-25","2006-12-12",
    # 2007
    "2007-01-30","2007-01-31","2007-03-20","2007-03-21","2007-05-09","2007-06-27",
    "2007-06-28","2007-08-07","2007-09-18","2007-10-30","2007-10-31","2007-12-11",
    # 2008
    "2008-01-09","2008-01-21","2008-01-22","2008-01-29","2008-01-30","2008-03-11",
    "2008-03-18","2008-04-29","2008-04-30","2008-06-24","2008-06-25","2008-08-05",
    "2008-09-16","2008-10-07","2008-10-08","2008-10-28","2008-10-29","2008-12-15","2008-12-16",
    # 2009
    "2009-01-27","2009-01-28","2009-03-17","2009-03-18","2009-04-28","2009-04-29",
    "2009-06-23","2009-06-24","2009-08-11","2009-08-12","2009-09-22","2009-09-23",
    "2009-11-03","2009-11-04","2009-12-15","2009-12-16",
    # 2010
    "2010-01-26","2010-01-27","2010-03-16","2010-04-27","2010-04-28","2010-06-22",
    "2010-06-23","2010-08-10","2010-09-21","2010-11-02","2010-11-03","2010-12-14",
    # 2011
    "2011-01-25","2011-01-26","2011-03-15","2011-04-26","2011-04-27","2011-06-21",
    "2011-06-22","2011-08-09","2011-09-20","2011-09-21","2011-11-01","2011-11-02","2011-12-13",
    # 2012
    "2012-01-24","2012-01-25","2012-03-13","2012-04-24","2012-04-25","2012-06-19",
    "2012-06-20","2012-07-31","2012-08-01","2012-09-12","2012-09-13","2012-10-23",
    "2012-10-24","2012-12-11","2012-12-12",
    # 2013
    "2013-01-29","2013-01-30","2013-03-19","2013-03-20","2013-04-30","2013-05-01",
    "2013-06-18","2013-06-19","2013-07-30","2013-07-31","2013-09-17","2013-09-18",
    "2013-10-29","2013-10-30","2013-12-17","2013-12-18",
    # 2014
    "2014-01-28","2014-01-29","2014-03-18","2014-03-19","2014-04-29","2014-04-30",
    "2014-06-17","2014-06-18","2014-07-29","2014-07-30","2014-09-16","2014-09-17",
    "2014-10-28","2014-10-29","2014-12-16","2014-12-17",
    # 2015
    "2015-01-27","2015-01-28","2015-03-17","2015-03-18","2015-04-28","2015-04-29",
    "2015-06-16","2015-06-17","2015-07-28","2015-07-29","2015-09-16","2015-09-17",
    "2015-10-27","2015-10-28","2015-12-15","2015-12-16",
    # 2016
    "2016-01-26","2016-01-27","2016-03-15","2016-03-16","2016-04-26","2016-04-27",
    "2016-06-14","2016-06-15","2016-07-26","2016-07-27","2016-09-20","2016-09-21",
    "2016-11-01","2016-11-02","2016-12-13","2016-12-14",
    # 2017
    "2017-01-31","2017-02-01","2017-03-14","2017-03-15","2017-05-02","2017-05-03",
    "2017-06-13","2017-06-14","2017-07-25","2017-07-26","2017-09-19","2017-09-20",
    "2017-10-31","2017-11-01","2017-12-12","2017-12-13",
    # 2018
    "2018-01-30","2018-01-31","2018-03-20","2018-03-21","2018-05-01","2018-05-02",
    "2018-06-12","2018-06-13","2018-07-31","2018-08-01","2018-09-25","2018-09-26",
    "2018-11-07","2018-11-08","2018-12-18","2018-12-19",
    # 2019
    "2019-01-29","2019-01-30","2019-03-19","2019-03-20","2019-04-30","2019-05-01",
    "2019-06-18","2019-06-19","2019-07-30","2019-07-31","2019-09-17","2019-09-18",
    "2019-10-29","2019-10-30","2019-12-10","2019-12-11",
    # 2020
    "2020-01-28","2020-01-29","2020-03-03","2020-03-15","2020-03-23","2020-04-28",
    "2020-04-29","2020-06-09","2020-06-10","2020-07-28","2020-07-29","2020-09-15",
    "2020-09-16","2020-11-04","2020-11-05","2020-12-15","2020-12-16",
    # 2021
    "2021-01-26","2021-01-27","2021-03-16","2021-03-17","2021-04-27","2021-04-28",
    "2021-06-15","2021-06-16","2021-07-27","2021-07-28","2021-09-21","2021-09-22",
    "2021-11-02","2021-11-03","2021-12-14","2021-12-15",
    # 2022
    "2022-01-25","2022-01-26","2022-03-15","2022-03-16","2022-05-03","2022-05-04",
    "2022-06-14","2022-06-15","2022-07-26","2022-07-27","2022-09-20","2022-09-21",
    "2022-11-01","2022-11-02","2022-12-13","2022-12-14",
    # 2023
    "2023-01-31","2023-02-01","2023-03-21","2023-03-22","2023-05-02","2023-05-03",
    "2023-06-13","2023-06-14","2023-07-25","2023-07-26","2023-09-19","2023-09-20",
    "2023-10-31","2023-11-01","2023-12-12","2023-12-13",
    # 2024
    "2024-01-30","2024-01-31","2024-03-19","2024-03-20","2024-04-30","2024-05-01",
    "2024-06-11","2024-06-12","2024-07-30","2024-07-31","2024-09-17","2024-09-18",
    "2024-11-06","2024-11-07","2024-12-17","2024-12-18",
    # 2025
    "2025-01-28","2025-01-29","2025-03-18","2025-03-19","2025-05-06","2025-05-07",
    "2025-06-17","2025-06-18","2025-07-29","2025-07-30","2025-09-16","2025-09-17",
    "2025-10-28","2025-10-29","2025-12-16","2025-12-17",
    # 2026
    "2026-01-27","2026-01-28","2026-03-17","2026-03-18","2026-04-28","2026-04-29",
    "2026-06-16","2026-06-17","2026-07-28","2026-07-29","2026-09-15","2026-09-16",
    "2026-11-03","2026-11-04","2026-12-15","2026-12-16",
]
FOMC_DATES = pd.to_datetime(FOMC_DATES)

# US market holidays (major ones)
US_HOLIDAYS = [
    # New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day,
    # Independence Day, Labor Day, Thanksgiving, Christmas
    # We'll generate them programmatically for 2000-2026
]

def get_us_market_holidays(start_year=2000, end_year=2026):
    """Generate approximate US market holiday dates."""
    holidays = []
    for year in range(start_year, end_year + 1):
        # New Year's Day (Jan 1, observed)
        d = dt.date(year, 1, 1)
        if d.weekday() == 5: d = dt.date(year, 1, 3)  # Sat -> Mon
        elif d.weekday() == 6: d = dt.date(year, 1, 2) # Sun -> Mon
        holidays.append(d)

        # MLK Day (3rd Monday of January)
        d = dt.date(year, 1, 1)
        mondays = 0
        while mondays < 3:
            d += dt.timedelta(days=1)
            if d.weekday() == 0: mondays += 1
        holidays.append(d)

        # Presidents' Day (3rd Monday of February)
        d = dt.date(year, 2, 1)
        mondays = 0
        while mondays < 3:
            if d.weekday() == 0: mondays += 1
            if mondays < 3: d += dt.timedelta(days=1)
        holidays.append(d)

        # Good Friday (approximate — Friday before Easter)
        # Use anonymous algorithm for Easter
        a = year % 19
        b = year // 100
        c = year % 100
        d_val = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d_val - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter = dt.date(year, month, day)
        good_friday = easter - dt.timedelta(days=2)
        holidays.append(good_friday)

        # Memorial Day (last Monday of May)
        d = dt.date(year, 5, 31)
        while d.weekday() != 0:
            d -= dt.timedelta(days=1)
        holidays.append(d)

        # Juneteenth (June 19, from 2022+)
        if year >= 2022:
            d = dt.date(year, 6, 19)
            if d.weekday() == 5: d = dt.date(year, 6, 18)
            elif d.weekday() == 6: d = dt.date(year, 6, 20)
            holidays.append(d)

        # Independence Day (July 4)
        d = dt.date(year, 7, 4)
        if d.weekday() == 5: d = dt.date(year, 7, 3)
        elif d.weekday() == 6: d = dt.date(year, 7, 5)
        holidays.append(d)

        # Labor Day (1st Monday of September)
        d = dt.date(year, 9, 1)
        while d.weekday() != 0:
            d += dt.timedelta(days=1)
        holidays.append(d)

        # Thanksgiving (4th Thursday of November)
        d = dt.date(year, 11, 1)
        thursdays = 0
        while thursdays < 4:
            if d.weekday() == 3: thursdays += 1
            if thursdays < 4: d += dt.timedelta(days=1)
        holidays.append(d)

        # Christmas (Dec 25)
        d = dt.date(year, 12, 25)
        if d.weekday() == 5: d = dt.date(year, 12, 24)
        elif d.weekday() == 6: d = dt.date(year, 12, 26)
        holidays.append(d)

    return pd.to_datetime(holidays)


def download_spy_data():
    """Download SPY data from yfinance."""
    cache_file = OUTPUT_DIR / "spy_data.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 5000:
            print(f"Loaded cached SPY data: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
            return df

    print("Downloading SPY data from yfinance...")
    spy = yf.download("SPY", start="2000-01-01", end="2026-07-15", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].copy()
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    spy["Return"] = spy["Close"].pct_change()
    spy = spy.dropna()
    spy.to_parquet(cache_file)
    print(f"Downloaded SPY data: {len(spy)} days ({spy.index[0].date()} to {spy.index[-1].date()})")
    return spy


# ──────────────────────────────────────────────────────────────────────
# Signal generators
# ──────────────────────────────────────────────────────────────────────

def signal_tom(dates, trading_days):
    """Turn-of-Month: days -1 to +3 around month boundary."""
    signals = pd.Series(0, index=dates)
    td_list = sorted(trading_days)
    td_set = set(td_list)

    for i, d in enumerate(td_list):
        month = d.month
        year = d.year
        # Find next trading day
        if i + 1 < len(td_list):
            next_td = td_list[i + 1]
            # If next trading day is in a different month, this is day -1
            if next_td.month != month or next_td.year != year:
                if d in signals.index:
                    signals[d] = 1  # Day -1

        # Check if this is day +1, +2, or +3 of the month
        # Find the first trading day of this month
        first_td_of_month = None
        for td in td_list:
            if td.month == month and td.year == year:
                first_td_of_month = td
                break
        if first_td_of_month is not None:
            # Count trading days from first of month
            count = 0
            for td in td_list:
                if td.month == month and td.year == year:
                    count += 1
                    if td == d:
                        break
            if 1 <= count <= 3 and d in signals.index:
                signals[d] = 1

    return signals


def signal_pre_holiday(dates, holidays, trading_days):
    """Pre-Holiday: 1-2 trading days before market holidays."""
    signals = pd.Series(0, index=dates)
    td_list = sorted(trading_days)

    for hol in holidays:
        # Find the 1-2 trading days before this holiday
        pre_days = []
        for td in reversed(td_list):
            if td < pd.Timestamp(hol):
                pre_days.append(td)
                if len(pre_days) >= 2:
                    break
        for d in pre_days:
            if d in signals.index:
                signals[d] = 1

    return signals


def signal_fomc(dates, fomc_dates, trading_days):
    """FOMC Drift: 1-3 trading days before FOMC meetings."""
    signals = pd.Series(0, index=dates)
    td_list = sorted(trading_days)

    # Get unique FOMC meeting start dates (first day of each 2-day meeting)
    fomc_starts = set()
    fomc_sorted = sorted(fomc_dates)
    i = 0
    while i < len(fomc_sorted):
        fomc_starts.add(fomc_sorted[i])
        # Skip consecutive days (same meeting)
        while i + 1 < len(fomc_sorted) and (fomc_sorted[i+1] - fomc_sorted[i]).days <= 1:
            i += 1
        i += 1

    for fomc_d in fomc_starts:
        pre_days = []
        for td in reversed(td_list):
            if td < pd.Timestamp(fomc_d):
                pre_days.append(td)
                if len(pre_days) >= 3:
                    break
        for d in pre_days:
            if d in signals.index:
                signals[d] = 1

    return signals


def signal_seasonality(dates):
    """Monthly Seasonality: Nov-Apr = 1, May-Oct = 0."""
    signals = pd.Series(0, index=dates)
    for d in dates:
        if d.month in [11, 12, 1, 2, 3, 4]:
            signals[d] = 1
    return signals


def signal_day_of_week(dates, day=0):
    """Day-of-Week effect. day=0 is Monday, day=4 is Friday."""
    signals = pd.Series(0, index=dates)
    for d in dates:
        if d.weekday() == day:
            signals[d] = 1
    return signals


def signal_triple_witching(dates):
    """Triple Witching week: week of 3rd Friday of Mar/Jun/Sep/Dec."""
    signals = pd.Series(0, index=dates)
    tw_months = [3, 6, 9, 12]

    for d in dates:
        if d.month in tw_months:
            # Find 3rd Friday of this month
            first_day = dt.date(d.year, d.month, 1)
            first_friday = first_day
            while first_friday.weekday() != 4:
                first_friday += dt.timedelta(days=1)
            third_friday = first_friday + dt.timedelta(weeks=2)

            # Check if d is in the same week (Mon-Fri containing 3rd Friday)
            week_start = third_friday - dt.timedelta(days=third_friday.weekday())  # Monday
            week_end = week_start + dt.timedelta(days=4)  # Friday
            if week_start <= d.date() <= week_end:
                signals[d] = 1

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
        "avg_ret": round(mean_ret * 10000, 2),  # in bps
    }


# ──────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ──────────────────────────────────────────────────────────────────────

def walk_forward_test(df, signal_series, train_months=36, test_months=6):
    """
    Walk-forward validation.
    In training window: check if signal has positive mean return.
    If yes, go long on signal days in test window.
    If negative, skip (signal=0) in test window.
    Returns concatenated OOT returns.
    """
    all_oot_returns = []
    all_oot_signals = []

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
            all_oot_signals.append(test_signal[test_signal == 1])

        current_start += pd.DateOffset(months=test_months)

    if all_oot_returns:
        return pd.concat(all_oot_returns)
    else:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────────────────────────────
# Adversarial checks (HC #705)
# ──────────────────────────────────────────────────────────────────────

def permutation_test(df, signal_series, n_perms=200, train_months=36, test_months=6):
    """Shuffle signal dates, keep returns. Compute p-value."""
    real_returns = walk_forward_test(df, signal_series, train_months, test_months)
    if len(real_returns) < 10:
        return 1.0, 0.0  # No signal

    real_sharpe = compute_metrics(real_returns)["sharpe"]

    better_count = 0
    for _ in range(n_perms):
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
    """Split returns into 2 halves, check consistency."""
    if len(returns) < 20:
        return {"half1_sharpe": 0, "half2_sharpe": 0, "consistent": False}

    mid = len(returns) // 2
    h1 = returns.iloc[:mid]
    h2 = returns.iloc[mid:]

    m1 = compute_metrics(h1)
    m2 = compute_metrics(h2)

    consistent = (m1["sharpe"] > 0 and m2["sharpe"] > 0) or (m1["sharpe"] < 0 and m2["sharpe"] < 0)
    # Both positive = good. Both negative = consistently bad. Mixed = unstable.
    return {
        "half1_sharpe": m1["sharpe"],
        "half2_sharpe": m2["sharpe"],
        "consistent": consistent,
        "both_positive": m1["sharpe"] > 0 and m2["sharpe"] > 0
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
    R1 regime test: classify days by TRAILING 20-day SPY regime (bull/bear/flat).
    This avoids the tautological problem of classifying by same-day return.
    Check if signal works across regimes.
    """
    oot_returns = walk_forward_test(df, signal_series, train_months, test_months)
    if len(oot_returns) < 30:
        return {"regime_gap": 1.0, "pass": False, "green_sharpe": 0, "red_sharpe": 0, "flat_sharpe": 0}

    # Classify regime by TRAILING 20-day SPY return (known at open, no lookahead)
    trailing_ret = df["Close"].pct_change(20).reindex(oot_returns.index)
    green = oot_returns[trailing_ret > 0.02]   # Bull: trailing 20d > +2%
    red = oot_returns[trailing_ret < -0.02]    # Bear: trailing 20d < -2%
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
# Combined signal
# ──────────────────────────────────────────────────────────────────────

def build_combined_signal(signals_dict, threshold=3):
    """Score each day 0-N based on how many calendar effects fire. Trade when >= threshold."""
    if not signals_dict:
        return pd.Series(dtype=float)

    score = None
    for name, sig in signals_dict.items():
        if score is None:
            score = sig.copy().astype(float)
        else:
            score = score.add(sig.reindex(score.index, fill_value=0), fill_value=0)

    combined = (score >= threshold).astype(int)
    return combined


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("CALENDAR ANOMALY STRATEGY — WALK-FORWARD VALIDATION")
    print("=" * 80)
    print()

    # Download data
    df = download_spy_data()
    trading_days = df.index.tolist()
    holidays = get_us_market_holidays(2000, 2026)

    print(f"\nData range: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} trading days)")
    print(f"Buy & Hold CAGR: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1)*100:.2f}%")
    bh_metrics = compute_metrics(df["Return"])
    print(f"Buy & Hold Sharpe: {bh_metrics['sharpe']}, Sortino: {bh_metrics['sortino']}")
    print()

    # Generate all signals
    print("Generating signals...")
    signals = {}
    signals["TOM"] = signal_tom(df.index, trading_days)
    signals["Pre-Holiday"] = signal_pre_holiday(df.index, holidays, trading_days)
    signals["FOMC Drift"] = signal_fomc(df.index, FOMC_DATES, trading_days)
    signals["Seasonality"] = signal_seasonality(df.index)
    signals["Monday"] = signal_day_of_week(df.index, day=0)
    signals["Friday"] = signal_day_of_week(df.index, day=4)
    signals["Triple Witch"] = signal_triple_witching(df.index)

    for name, sig in signals.items():
        n_days = sig.sum()
        pct = n_days / len(sig) * 100
        print(f"  {name:15s}: {int(n_days):5d} signal days ({pct:.1f}%)")

    # Build combined signal
    # Use all individual signals for scoring (excluding Monday/Friday separately, use best DOW)
    combo_signals = {k: v for k, v in signals.items() if k not in ["Monday", "Friday"]}
    # Add a generic "positive DOW" — we'll pick the best in walk-forward
    signals["Combined (3+)"] = build_combined_signal(combo_signals, threshold=3)
    print(f"  {'Combined (3+)':15s}: {int(signals['Combined (3+)'].sum()):5d} signal days ({signals['Combined (3+)'].sum()/len(df)*100:.1f}%)")

    # ──────────────────────────────────────────────────────────────────
    # Walk-forward test for each anomaly
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS (36-month train, 6-month test, sliding)")
    print("=" * 80)

    results = {}
    for name, sig in signals.items():
        print(f"\n{'─' * 60}")
        print(f"Testing: {name}")
        print(f"{'─' * 60}")

        oot_returns = walk_forward_test(df, sig)
        if len(oot_returns) < 10:
            print(f"  ⚠ Insufficient OOT data ({len(oot_returns)} days). Skipping.")
            results[name] = {"status": "SKIP", "reason": "insufficient data"}
            continue

        metrics = compute_metrics(oot_returns)
        print(f"  OOT Days: {metrics['n_trades']}")
        print(f"  Sharpe:   {metrics['sharpe']:+.3f}")
        print(f"  Sortino:  {metrics['sortino']:+.3f}")
        print(f"  CAGR:     {metrics['cagr']:+.2f}%")
        print(f"  MaxDD:    {metrics['maxdd']:.2f}%")
        print(f"  WR:       {metrics['wr']:.1%}")
        print(f"  PF:       {metrics['pf']:.3f}")
        print(f"  Avg Ret:  {metrics['avg_ret']:+.2f} bps/day")

        # Adversarial checks
        print(f"\n  Adversarial Checks:")

        # 1. Permutation test
        print(f"    Running permutation test (200 shuffles)...", end=" ", flush=True)
        p_value, real_sharpe = permutation_test(df, sig)
        perm_pass = p_value < 0.05
        print(f"p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

        # 2. Sub-period consistency
        sub_period = sub_period_consistency(oot_returns)
        sub_pass = sub_period["both_positive"]
        print(f"    Sub-period: H1 Sharpe={sub_period['half1_sharpe']:+.3f}, "
              f"H2 Sharpe={sub_period['half2_sharpe']:+.3f} "
              f"{'PASS' if sub_pass else 'FAIL'}")

        # 3. Outlier removal
        cleaned_metrics = outlier_removal_test(oot_returns)
        outlier_pass = cleaned_metrics["sharpe"] > 0
        print(f"    Outlier removal (top 5): Sharpe={cleaned_metrics['sharpe']:+.3f} "
              f"{'PASS' if outlier_pass else 'FAIL'}")

        # 4. Regime test
        regime = regime_test(df, sig)
        regime_pass = regime["pass"]
        print(f"    Regime test: gap={regime['regime_gap']:.3f} "
              f"(G={regime['green_sharpe']:+.3f}, R={regime['red_sharpe']:+.3f}, "
              f"F={regime['flat_sharpe']:+.3f}) "
              f"{'PASS' if regime_pass else 'FAIL'}")

        gates_passed = sum([perm_pass, sub_pass, outlier_pass, regime_pass])
        all_pass = gates_passed == 4

        results[name] = {
            "status": "PASS" if all_pass else "FAIL",
            "gates_passed": f"{gates_passed}/4",
            "metrics": metrics,
            "perm_p": p_value,
            "sub_period": sub_period,
            "cleaned_sharpe": cleaned_metrics["sharpe"],
            "regime": regime,
            "perm_pass": perm_pass,
            "sub_pass": sub_pass,
            "outlier_pass": outlier_pass,
            "regime_pass": regime_pass,
        }

        verdict = "*** ALL GATES PASSED ***" if all_pass else f"FAILED ({4-gates_passed} gate(s))"
        print(f"\n  Verdict: {verdict}")

    # ──────────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    header = f"{'Anomaly':<18} {'Status':<8} {'Gates':<7} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR':>7} {'PF':>7} {'Perm-p':>7} {'SubPrd':>7} {'Regime':>7}"
    print(header)
    print("─" * 120)

    for name, res in results.items():
        if res["status"] == "SKIP":
            print(f"{name:<18} {'SKIP':<8} {'--':<7} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'--':>7} {'--':>7} {'--':>7} {'--':>7} {'--':>7}")
            continue

        m = res["metrics"]
        print(f"{name:<18} {res['status']:<8} {res['gates_passed']:<7} "
              f"{m['sharpe']:>+8.3f} {m['sortino']:>+8.3f} {m['cagr']:>+8.2f} {m['maxdd']:>8.2f} "
              f"{m['wr']:>7.1%} {m['pf']:>7.3f} "
              f"{res['perm_p']:>7.3f} "
              f"{'Y' if res['sub_pass'] else 'N':>7} "
              f"{'Y' if res['regime_pass'] else 'N':>7}")

    print("─" * 120)

    # Buy & Hold benchmark
    print(f"\n{'Benchmark (B&H)':<18} {'--':<8} {'--':<7} "
          f"{bh_metrics['sharpe']:>+8.3f} {bh_metrics['sortino']:>+8.3f} "
          f"{((df['Close'].iloc[-1]/df['Close'].iloc[0])**(252/len(df))-1)*100:>+8.2f} "
          f"{'--':>8} {bh_metrics['wr']:>7.1%} {bh_metrics['pf']:>7.3f} "
          f"{'--':>7} {'--':>7} {'--':>7}")

    # ──────────────────────────────────────────────────────────────────
    # Conclusions
    # ──────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    passing = [name for name, res in results.items() if res.get("status") == "PASS"]
    failing = [name for name, res in results.items() if res.get("status") == "FAIL"]
    skipped = [name for name, res in results.items() if res.get("status") == "SKIP"]

    if passing:
        print(f"\nALL GATES PASSED ({len(passing)}):")
        for name in passing:
            m = results[name]["metrics"]
            print(f"  - {name}: Sharpe={m['sharpe']:+.3f}, CAGR={m['cagr']:+.2f}%, "
                  f"WR={m['wr']:.1%}, Perm-p={results[name]['perm_p']:.3f}")
    else:
        print("\nNo anomalies passed ALL 4 adversarial gates.")

    if failing:
        print(f"\nFAILED ({len(failing)}):")
        for name in failing:
            m = results[name]["metrics"]
            fails = []
            if not results[name]["perm_pass"]: fails.append("permutation")
            if not results[name]["sub_pass"]: fails.append("sub-period")
            if not results[name]["outlier_pass"]: fails.append("outlier")
            if not results[name]["regime_pass"]: fails.append("regime")
            print(f"  - {name}: Sharpe={m['sharpe']:+.3f}, failed: {', '.join(fails)}")

    # Partial passes (3/4 gates)
    partial = [name for name, res in results.items()
               if res.get("gates_passed") == "3/4"]
    if partial:
        print(f"\nPARTIAL PASS (3/4 gates) — worth investigating:")
        for name in partial:
            m = results[name]["metrics"]
            print(f"  - {name}: Sharpe={m['sharpe']:+.3f}, CAGR={m['cagr']:+.2f}%")

    # Save results
    save_results = {}
    for name, res in results.items():
        save_results[name] = {
            k: v for k, v in res.items()
            if k not in ["metrics", "sub_period", "regime"]
        }
        if "metrics" in res:
            save_results[name]["metrics"] = res["metrics"]
        if "sub_period" in res:
            save_results[name]["sub_period"] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in res["sub_period"].items()
            }
        if "regime" in res:
            save_results[name]["regime"] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in res["regime"].items()
            }

    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    np.random.seed(42)
    main()
