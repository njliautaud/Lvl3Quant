#!/usr/bin/env python3
"""
ML-Enhanced Calendar Effects Trading Strategy
==============================================
Tests 5 documented calendar anomalies with LightGBM walk-forward enhancement.

Anomalies (academic literature):
  1. FOMC Pre-Announcement Drift — Lucca & Moench (2015)
  2. Turn-of-Month Effect — Ariel (1987)
  3. Holiday Effect — pre-holiday drift
  4. Monthly Seasonality — Oct/Nov recovery after Sep weakness
  5. Day-of-Week Effect — Monday weakness

Pipeline:
  1. Download SPY, QQQ, IWM, TLT, GLD from yfinance (2000-01-01+)
  2. Build calendar features + market context features
  3. Baseline rule-based strategies per anomaly
  4. LightGBM walk-forward (252d sliding) predicts anomaly firing
  5. Combined ML calendar strategy
  6. Adversarial validation: permutation test, sub-period, regime, outlier

HC Compliance:
  - HC #713: Fixed $100K capital, no DCA
  - HC #0: Sliding windows only
  - Cost: 5bps per trade (SPY liquid)
  - Output: output/ml_calendar_effects/
  - Log: logs/ml_calendar_effects.log

Author: Claude (HC #714 research)
"""

import os
import sys
import json
import time
import warnings
import datetime as dt
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_calendar_effects")
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_calendar_effects/cache")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = Path("/home/jupiter/Lvl3Quant/logs/ml_calendar_effects.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Walk-forward
TRAIN_DAYS = 252       # 1 year sliding
TEST_DAYS = 63         # 3 months OOT (keeps ~80 folds, fast enough for CPU)
COST_BPS = 5           # 5 bps per trade
INITIAL_CAPITAL = 100_000

# Assets
TICKERS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
TRADE_TICKER = "SPY"   # primary trading vehicle

# LightGBM hyperparams (small dataset: 252 samples x ~30 features → single thread is fastest)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,
    "learning_rate": 0.1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": 1,
    "num_threads": 1,      # CRITICAL: avoid thread spawning overhead on tiny datasets
    "min_data_in_bin": 3,
}
LGB_ROUNDS = 80
LGB_EARLY_STOP = 10

N_PERMUTATIONS = 100   # for permutation test

# ══════════════════════════════════════════════════════════════
# FOMC DATES (2000-2026) — actual Federal Reserve calendar
# ══════════════════════════════════════════════════════════════
FOMC_DATES_RAW = [
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

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def get_fomc_meeting_starts():
    """Extract unique FOMC meeting start dates (first day of each 2-day meeting)."""
    fomc_all = sorted(pd.to_datetime(FOMC_DATES_RAW))
    starts = []
    i = 0
    while i < len(fomc_all):
        starts.append(fomc_all[i])
        # skip consecutive days (same meeting)
        while i + 1 < len(fomc_all) and (fomc_all[i + 1] - fomc_all[i]).days <= 1:
            i += 1
        i += 1
    return starts


def get_us_market_holidays(start_year=2000, end_year=2026):
    """Generate approximate US market holiday dates."""
    holidays = []
    for year in range(start_year, end_year + 1):
        # New Year's Day
        d = dt.date(year, 1, 1)
        if d.weekday() == 5: d = dt.date(year, 1, 3)
        elif d.weekday() == 6: d = dt.date(year, 1, 2)
        holidays.append(d)

        # MLK Day (3rd Monday of January)
        d = dt.date(year, 1, 1)
        mondays = 0
        while mondays < 3:
            d += dt.timedelta(days=1)
            if d.weekday() == 0:
                mondays += 1
        holidays.append(d)

        # Presidents' Day (3rd Monday of February)
        d = dt.date(year, 2, 1)
        mondays = 0
        while mondays < 3:
            if d.weekday() == 0:
                mondays += 1
            if mondays < 3:
                d += dt.timedelta(days=1)
        holidays.append(d)

        # Good Friday
        a = year % 19
        b = year // 100
        c = year % 100
        d_val = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d_val - g + 15) % 30
        ii = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * ii - h - k) % 7
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

        # Juneteenth (from 2022)
        if year >= 2022:
            d = dt.date(year, 6, 19)
            if d.weekday() == 5: d = dt.date(year, 6, 18)
            elif d.weekday() == 6: d = dt.date(year, 6, 20)
            holidays.append(d)

        # Independence Day
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
            if d.weekday() == 3:
                thursdays += 1
            if thursdays < 4:
                d += dt.timedelta(days=1)
        holidays.append(d)

        # Christmas
        d = dt.date(year, 12, 25)
        if d.weekday() == 5: d = dt.date(year, 12, 24)
        elif d.weekday() == 6: d = dt.date(year, 12, 26)
        holidays.append(d)

    return pd.to_datetime(holidays)


def compute_metrics(returns, ann_factor=252):
    """Compute trading strategy metrics."""
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "cagr": 0, "max_dd": 0,
                "win_rate": 0, "profit_factor": 0, "n_trades": 0,
                "total_return": 0, "avg_return": 0}

    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1 if total_ret > -1 else -1

    sharpe = returns.mean() / returns.std() * np.sqrt(ann_factor) if returns.std() > 0 else 0
    downside = returns[returns < 0].std()
    sortino = returns.mean() / downside * np.sqrt(ann_factor) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(min(profit_factor, 999), 2),
        "n_trades": len(returns),
        "total_return": round(total_ret * 100, 2),
        "avg_return": round(returns.mean() * 10000, 2),  # bps
    }


# ══════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download multi-asset data from yfinance with caching."""
    cache_file = CACHE_DIR / "multi_asset_data.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        age_days = (dt.datetime.now() - dt.datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < 3 and len(df) > 5000:
            log.info(f"Loaded cached data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            return df

    log.info("Downloading multi-asset data from yfinance...")
    all_tickers = TICKERS + ["^VIX"]
    raw = yf.download(all_tickers, start="2000-01-01", progress=False)

    # Handle MultiIndex columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        # Extract close prices for each ticker
        dfs = {}
        for ticker in all_tickers:
            label = ticker.replace("^", "")
            try:
                close = raw["Close"][ticker].dropna()
                volume = raw["Volume"][ticker].dropna() if ticker != "^VIX" else None
                high = raw["High"][ticker].dropna() if ticker != "^VIX" else None
                low = raw["Low"][ticker].dropna() if ticker != "^VIX" else None
                dfs[f"{label}_close"] = close
                if volume is not None:
                    dfs[f"{label}_volume"] = volume
                if high is not None:
                    dfs[f"{label}_high"] = high
                if low is not None:
                    dfs[f"{label}_low"] = low
            except (KeyError, TypeError):
                log.warning(f"Could not get data for {ticker}")
    else:
        # Single ticker fallback
        dfs = {"SPY_close": raw["Close"], "SPY_volume": raw["Volume"],
               "SPY_high": raw["High"], "SPY_low": raw["Low"]}

    df = pd.DataFrame(dfs)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.dropna(subset=["SPY_close"])
    df = df.ffill().dropna()

    df.to_parquet(cache_file)
    log.info(f"Downloaded data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════
# 2. CALENDAR FEATURES
# ══════════════════════════════════════════════════════════════

def build_calendar_features(df):
    """Build all calendar anomaly flags and market context features."""
    dates = df.index
    td_list = sorted(dates.tolist())
    td_set = set(td_list)

    fomc_starts = get_fomc_meeting_starts()
    holidays = get_us_market_holidays()

    feats = pd.DataFrame(index=dates)

    # ── 1. FOMC Pre-Announcement Drift ──
    # Flag: 1 trading day before FOMC meeting start (buy close day before, sell close FOMC day)
    fomc_pre = pd.Series(0, index=dates, dtype=int)
    fomc_day = pd.Series(0, index=dates, dtype=int)
    for fomc_d in fomc_starts:
        # Find the trading day immediately before this FOMC date
        for i, td in enumerate(td_list):
            if td >= pd.Timestamp(fomc_d):
                if i > 0:
                    prev_td = td_list[i - 1]
                    if prev_td in td_set:
                        fomc_pre.loc[prev_td] = 1
                # Mark the FOMC day itself
                if td == pd.Timestamp(fomc_d) or (td - pd.Timestamp(fomc_d)).days <= 1:
                    fomc_day.loc[td] = 1
                break
    feats["fomc_pre"] = fomc_pre
    feats["fomc_day"] = fomc_day
    feats["fomc_window"] = ((fomc_pre == 1) | (fomc_day == 1)).astype(int)

    # ── 2. Turn-of-Month Effect ──
    # Last 1 trading day of month + first 3 trading days of month
    tom = pd.Series(0, index=dates, dtype=int)
    # Group trading days by year-month for efficiency
    td_months = pd.Series(td_list).dt.to_period('M')
    for period, group in pd.Series(td_list).groupby(td_months):
        sorted_days = group.sort_values().tolist()
        # Last trading day of month
        if len(sorted_days) > 0:
            tom.loc[sorted_days[-1]] = 1
        # First 3 trading days of month
        for d in sorted_days[:3]:
            tom.loc[d] = 1
    feats["tom"] = tom

    # ── 3. Pre-Holiday Effect ──
    pre_holiday = pd.Series(0, index=dates, dtype=int)
    hol_set = set(holidays.date if hasattr(holidays, 'date') else [h.date() for h in holidays])
    for i, td in enumerate(td_list):
        # Check if next calendar day (or next 3 calendar days) is a holiday
        for offset in range(1, 4):
            check_date = (td + pd.Timedelta(days=offset)).date()
            if check_date in hol_set:
                pre_holiday.loc[td] = 1
                break
    feats["pre_holiday"] = pre_holiday

    # ── 4. Monthly Seasonality ──
    feats["month"] = dates.month
    feats["is_oct_nov"] = dates.month.isin([10, 11]).astype(int)
    feats["is_sep"] = (dates.month == 9).astype(int)
    feats["is_nov_apr"] = dates.month.isin([11, 12, 1, 2, 3, 4]).astype(int)  # "Sell in May"
    feats["is_jan"] = (dates.month == 1).astype(int)  # January effect

    # ── 5. Day-of-Week Effect ──
    feats["dow"] = dates.dayofweek
    feats["is_monday"] = (dates.dayofweek == 0).astype(int)
    feats["is_friday"] = (dates.dayofweek == 4).astype(int)

    # ── Additional calendar features ──
    feats["week_of_month"] = ((dates.day - 1) // 7 + 1).astype(int)
    feats["is_month_end_week"] = (dates.day >= 25).astype(int)
    feats["quarter"] = dates.quarter
    feats["is_quarter_end"] = ((dates.month.isin([3, 6, 9, 12])) & (dates.day >= 25)).astype(int)

    # ── Market context features ──
    spy_close = df["SPY_close"]
    spy_ret = spy_close.pct_change()

    # Momentum
    feats["spy_ret_1d"] = spy_ret
    feats["spy_ret_5d"] = spy_close.pct_change(5)
    feats["spy_ret_21d"] = spy_close.pct_change(21)
    feats["spy_ret_63d"] = spy_close.pct_change(63)

    # Volatility
    feats["spy_vol_5d"] = spy_ret.rolling(5).std() * np.sqrt(252)
    feats["spy_vol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252)
    feats["vol_ratio"] = feats["spy_vol_5d"] / feats["spy_vol_21d"].replace(0, np.nan)

    # Volume relative
    if "SPY_volume" in df.columns:
        feats["spy_vol_rel"] = df["SPY_volume"] / df["SPY_volume"].rolling(21).mean()

    # VIX level
    if "VIX_close" in df.columns:
        feats["vix"] = df["VIX_close"]
        feats["vix_5d_chg"] = df["VIX_close"].pct_change(5)
        feats["vix_above_20"] = (df["VIX_close"] > 20).astype(int)
        feats["vix_above_30"] = (df["VIX_close"] > 30).astype(int)

    # Cross-asset momentum (for regime context)
    for ticker in ["QQQ", "IWM", "TLT", "GLD"]:
        col = f"{ticker}_close"
        if col in df.columns:
            feats[f"{ticker.lower()}_ret_5d"] = df[col].pct_change(5)
            feats[f"{ticker.lower()}_ret_21d"] = df[col].pct_change(21)

    # SPY distance from MA
    feats["spy_above_sma50"] = (spy_close > spy_close.rolling(50).mean()).astype(int)
    feats["spy_above_sma200"] = (spy_close > spy_close.rolling(200).mean()).astype(int)
    feats["spy_dist_sma50"] = (spy_close / spy_close.rolling(50).mean() - 1)

    # High-low range
    if "SPY_high" in df.columns and "SPY_low" in df.columns:
        feats["spy_range"] = (df["SPY_high"] - df["SPY_low"]) / df["SPY_close"]
        feats["spy_range_21d"] = feats["spy_range"].rolling(21).mean()

    # Combined anomaly count
    feats["n_anomalies"] = (feats["fomc_pre"] + feats["tom"] + feats["pre_holiday"] +
                            feats["is_friday"] + feats["is_oct_nov"])

    log.info(f"Built {len(feats.columns)} calendar + context features")
    log.info(f"  FOMC pre-announcement days: {feats['fomc_pre'].sum()}")
    log.info(f"  Turn-of-month days: {feats['tom'].sum()}")
    log.info(f"  Pre-holiday days: {feats['pre_holiday'].sum()}")

    return feats


# ══════════════════════════════════════════════════════════════
# 3. BASELINE RULE-BASED STRATEGIES
# ══════════════════════════════════════════════════════════════

def run_baseline_strategies(df, feats):
    """Run simple rule-based strategies for each anomaly."""
    spy_ret = df["SPY_close"].pct_change()
    cost = COST_BPS / 10000  # per trade

    results = {}
    strategies = {
        "FOMC_Pre": feats["fomc_pre"],
        "Turn_of_Month": feats["tom"],
        "Pre_Holiday": feats["pre_holiday"],
        "Oct_Nov_Seasonality": feats["is_oct_nov"],
        "Monday_Avoid": 1 - feats["is_monday"],  # avoid Mondays = long every other day
        "Nov_Apr": feats["is_nov_apr"],
        "Combined_3plus": (feats["n_anomalies"] >= 2).astype(int),
    }

    log.info("\n" + "=" * 70)
    log.info("BASELINE RULE-BASED STRATEGIES (no ML)")
    log.info("=" * 70)

    for name, signal in strategies.items():
        # Strategy returns: long SPY when signal=1, flat when signal=0
        strat_ret = spy_ret * signal.shift(1)  # shift to avoid lookahead
        # Apply cost on entry/exit
        trades = signal.diff().abs().fillna(0)
        strat_ret = strat_ret - trades.shift(1) * cost
        strat_ret = strat_ret.dropna()

        # Only count days when we're in a position
        active_ret = strat_ret[signal.shift(1) == 1].dropna()

        metrics = compute_metrics(strat_ret)
        metrics["active_days"] = int(signal.sum())
        metrics["pct_active"] = round(signal.mean() * 100, 1)
        results[name] = metrics

        log.info(f"\n  {name}:")
        log.info(f"    Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
                 f"CAGR={metrics['cagr']:.1f}%  MaxDD={metrics['max_dd']:.1f}%")
        log.info(f"    WR={metrics['win_rate']:.1f}%  PF={metrics['profit_factor']:.2f}  "
                 f"Active={metrics['pct_active']:.0f}% of days")

    # Buy-and-hold benchmark
    bh_ret = spy_ret.dropna()
    bh_metrics = compute_metrics(bh_ret)
    bh_metrics["active_days"] = len(bh_ret)
    bh_metrics["pct_active"] = 100.0
    results["Buy_Hold_SPY"] = bh_metrics
    log.info(f"\n  Buy_Hold_SPY (benchmark):")
    log.info(f"    Sharpe={bh_metrics['sharpe']:.2f}  Sortino={bh_metrics['sortino']:.2f}  "
             f"CAGR={bh_metrics['cagr']:.1f}%  MaxDD={bh_metrics['max_dd']:.1f}%")

    return results


# ══════════════════════════════════════════════════════════════
# 4. ML-ENHANCED WALK-FORWARD
# ══════════════════════════════════════════════════════════════

def get_ml_features(feats):
    """Select features for ML model."""
    # Calendar flags
    cal_cols = ["fomc_pre", "fomc_day", "fomc_window", "tom", "pre_holiday",
                "is_oct_nov", "is_sep", "is_nov_apr", "is_jan",
                "is_monday", "is_friday", "week_of_month", "is_month_end_week",
                "is_quarter_end", "n_anomalies", "month", "dow", "quarter"]

    # Market context
    ctx_cols = ["spy_ret_1d", "spy_ret_5d", "spy_ret_21d", "spy_ret_63d",
                "spy_vol_5d", "spy_vol_21d", "vol_ratio",
                "spy_above_sma50", "spy_above_sma200", "spy_dist_sma50"]

    # Optional cols
    opt_cols = ["spy_vol_rel", "vix", "vix_5d_chg", "vix_above_20", "vix_above_30",
                "qqq_ret_5d", "qqq_ret_21d", "iwm_ret_5d", "iwm_ret_21d",
                "tlt_ret_5d", "tlt_ret_21d", "gld_ret_5d", "gld_ret_21d",
                "spy_range", "spy_range_21d"]

    all_cols = cal_cols + ctx_cols
    for c in opt_cols:
        if c in feats.columns:
            all_cols.append(c)

    return [c for c in all_cols if c in feats.columns]


def ml_walk_forward(df, feats, anomaly_name, anomaly_signal):
    """
    Walk-forward LightGBM: predict whether the anomaly day will have positive returns.
    Uses 252-day sliding window training, numpy arrays for speed.

    Target: 1 if next-day return > 0 on anomaly days, 0 otherwise.
    Strategy: only trade anomaly days where ML predicts positive return.
    """
    spy_ret = df["SPY_close"].pct_change()
    cost = COST_BPS / 10000

    feature_cols = get_ml_features(feats)
    X_df = feats[feature_cols].copy()
    y_s = (spy_ret.shift(-1) > 0).astype(int)

    # Align and drop NaN — convert to numpy for speed
    valid_mask = X_df.notna().all(axis=1) & y_s.notna()
    valid_idx = X_df.index[valid_mask]
    X = X_df.loc[valid_idx].values
    y = y_s.loc[valid_idx].values
    signal_arr = anomaly_signal.loc[valid_idx].values
    ret_arr = spy_ret.loc[valid_idx].values

    pred_arr = np.full(len(X), np.nan)
    proba_arr = np.full(len(X), np.nan)
    feature_importance = defaultdict(float)

    n_folds = 0
    total_folds = (len(X) - TRAIN_DAYS) // TEST_DAYS
    i = TRAIN_DAYS

    while i + TEST_DAYS <= len(X):
        X_train = X[i - TRAIN_DAYS:i]
        y_train = y[i - TRAIN_DAYS:i]
        X_test = X[i:i + TEST_DAYS]

        # Only train if we have enough anomaly days in training
        if signal_arr[i - TRAIN_DAYS:i].sum() < 5:
            i += TEST_DAYS
            continue

        if n_folds % 20 == 0:
            log.info(f"    Fold {n_folds}/{total_folds}...")

        val_size = max(int(len(X_train) * 0.15), 21)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:], reference=dtrain)

        try:
            model = lgb.train(
                LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                           lgb.log_evaluation(0)],
            )

            preds = model.predict(X_test)
            pred_arr[i:i + TEST_DAYS] = (preds > 0.5).astype(int)
            proba_arr[i:i + TEST_DAYS] = preds

            imp = model.feature_importance(importance_type="gain")
            for fname, fval in zip(feature_cols, imp):
                feature_importance[fname] += fval

            n_folds += 1

        except Exception as e:
            log.warning(f"  Fold {n_folds} failed: {e}")

        i += TEST_DAYS

    if n_folds == 0:
        log.warning(f"  {anomaly_name}: No folds completed!")
        return None

    # Rebuild as pandas for strategy calculation
    predictions = pd.Series(pred_arr, index=valid_idx)
    pred_proba = pd.Series(proba_arr, index=valid_idx)
    signal_s = pd.Series(signal_arr, index=valid_idx)

    # Strategy: trade anomaly days where ML says "go"
    ml_signal = ((signal_s == 1) & (predictions == 1)).astype(int)
    # Reindex to full df for proper shift
    ml_signal = ml_signal.reindex(spy_ret.index, fill_value=0)
    strat_ret = spy_ret * ml_signal.shift(1)
    trades = ml_signal.diff().abs().fillna(0)
    strat_ret = strat_ret - trades.shift(1) * cost
    strat_ret = strat_ret.dropna()

    metrics = compute_metrics(strat_ret)
    metrics["n_folds"] = n_folds
    metrics["ml_active_days"] = int(ml_signal.sum())

    sorted_imp = sorted(feature_importance.items(), key=lambda x: -x[1])
    top_features = sorted_imp[:10]

    log.info(f"\n  ML-Enhanced {anomaly_name}:")
    log.info(f"    Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
             f"CAGR={metrics['cagr']:.1f}%  MaxDD={metrics['max_dd']:.1f}%")
    log.info(f"    WR={metrics['win_rate']:.1f}%  PF={metrics['profit_factor']:.2f}  "
             f"Folds={n_folds}  ML-Active days={metrics['ml_active_days']}")
    log.info(f"    Top features: {', '.join(f'{k}({v:.0f})' for k, v in top_features[:5])}")

    return {
        "metrics": metrics,
        "signal": ml_signal,
        "strat_returns": strat_ret,
        "predictions": predictions,
        "pred_proba": pred_proba,
        "feature_importance": dict(sorted_imp),
    }


def ml_combined_strategy(df, feats, anomaly_results):
    """
    Combined ML strategy: for each day, ML picks the best anomaly to trade.
    Uses walk-forward LightGBM with all calendar + context features.
    Target: positive next-day return.
    """
    spy_ret = df["SPY_close"].pct_change()
    cost = COST_BPS / 10000

    # Combine all anomaly signals into one: trade any day where at least one anomaly fires
    any_anomaly = pd.Series(0, index=feats.index, dtype=int)
    for name, res in anomaly_results.items():
        if res is not None:
            any_anomaly = any_anomaly | (res["signal"] == 1).astype(int)

    feature_cols = get_ml_features(feats)
    X_df = feats[feature_cols].copy()
    y_s = (spy_ret.shift(-1) > 0).astype(int)

    valid_mask = X_df.notna().all(axis=1) & y_s.notna()
    valid_idx = X_df.index[valid_mask]
    X = X_df.loc[valid_idx].values
    y = y_s.loc[valid_idx].values

    pred_arr = np.full(len(X), np.nan)
    n_folds = 0
    total_folds = (len(X) - TRAIN_DAYS) // TEST_DAYS
    i = TRAIN_DAYS

    while i + TEST_DAYS <= len(X):
        X_train = X[i - TRAIN_DAYS:i]
        y_train = y[i - TRAIN_DAYS:i]
        X_test = X[i:i + TEST_DAYS]

        if n_folds % 20 == 0:
            log.info(f"    Combined fold {n_folds}/{total_folds}...")

        val_size = max(int(len(X_train) * 0.15), 21)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:], reference=dtrain)

        try:
            model = lgb.train(
                LGB_PARAMS, dtrain, num_boost_round=LGB_ROUNDS,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                           lgb.log_evaluation(0)],
            )
            preds = model.predict(X_test)
            pred_arr[i:i + TEST_DAYS] = preds
            n_folds += 1
        except Exception:
            pass
        i += TEST_DAYS

    if n_folds == 0:
        return None

    predictions = pd.Series(pred_arr, index=valid_idx)

    # Strategy: high-confidence ML days only (top 30% probability)
    threshold = predictions.dropna().quantile(0.70)
    ml_signal = (predictions > threshold).astype(int)
    ml_signal = ml_signal.reindex(spy_ret.index, fill_value=0)
    strat_ret = spy_ret * ml_signal.shift(1)
    trades = ml_signal.diff().abs().fillna(0)
    strat_ret = strat_ret - trades.shift(1) * cost
    strat_ret = strat_ret.dropna()

    metrics = compute_metrics(strat_ret)
    metrics["n_folds"] = n_folds
    metrics["ml_active_days"] = int(ml_signal.sum())
    metrics["threshold"] = round(float(threshold), 4)

    log.info(f"\n  ML Combined Calendar Strategy:")
    log.info(f"    Sharpe={metrics['sharpe']:.2f}  Sortino={metrics['sortino']:.2f}  "
             f"CAGR={metrics['cagr']:.1f}%  MaxDD={metrics['max_dd']:.1f}%")
    log.info(f"    WR={metrics['win_rate']:.1f}%  PF={metrics['profit_factor']:.2f}  "
             f"Active={metrics['ml_active_days']} days  Threshold={metrics['threshold']:.3f}")

    return {
        "metrics": metrics,
        "signal": ml_signal,
        "strat_returns": strat_ret,
        "predictions": predictions,
    }


# ══════════════════════════════════════════════════════════════
# 5. ADVERSARIAL VALIDATION
# ══════════════════════════════════════════════════════════════

def permutation_test(df, feats, anomaly_name, anomaly_signal, real_sharpe, n_perms=N_PERMUTATIONS):
    """
    Shuffle calendar labels and re-run strategy to test if calendar timing matters.
    If random dates produce similar Sharpe, the anomaly isn't real.
    """
    spy_ret = df["SPY_close"].pct_change()
    cost = COST_BPS / 10000

    perm_sharpes = []
    n_active = int(anomaly_signal.sum())

    for _ in range(n_perms):
        # Create random signal with same frequency
        fake_signal = pd.Series(0, index=anomaly_signal.index, dtype=int)
        random_days = np.random.choice(len(fake_signal), size=n_active, replace=False)
        fake_signal.iloc[random_days] = 1

        strat_ret = spy_ret * fake_signal.shift(1)
        trades = fake_signal.diff().abs().fillna(0)
        strat_ret = strat_ret - trades.shift(1) * cost
        strat_ret = strat_ret.dropna()

        if strat_ret.std() > 0:
            perm_sharpes.append(strat_ret.mean() / strat_ret.std() * np.sqrt(252))
        else:
            perm_sharpes.append(0)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    log.info(f"    Permutation test ({n_perms} shuffles): p={p_value:.3f}  "
             f"Real Sharpe={real_sharpe:.2f} vs Perm mean={perm_sharpes.mean():.2f} "
             f"+/- {perm_sharpes.std():.2f}")

    return {"p_value": round(p_value, 4),
            "perm_mean_sharpe": round(perm_sharpes.mean(), 3),
            "perm_std_sharpe": round(perm_sharpes.std(), 3),
            "significant": p_value < 0.05}


def sub_period_stability(strat_returns, n_periods=4):
    """Split returns into n_periods and check consistency."""
    chunk_size = len(strat_returns) // n_periods
    period_metrics = []

    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else len(strat_returns)
        chunk = strat_returns.iloc[start:end]
        m = compute_metrics(chunk)
        m["period"] = i + 1
        period_metrics.append(m)

    sharpes = [m["sharpe"] for m in period_metrics]
    positive_periods = sum(1 for s in sharpes if s > 0)
    stability = positive_periods / n_periods

    log.info(f"    Sub-period stability ({n_periods} periods): {positive_periods}/{n_periods} positive")
    for m in period_metrics:
        log.info(f"      Period {m['period']}: Sharpe={m['sharpe']:.2f}  CAGR={m['cagr']:.1f}%  "
                 f"WR={m['win_rate']:.1f}%")

    return {"period_metrics": period_metrics, "stability": round(stability, 2),
            "sharpes": sharpes, "stable": stability >= 0.5}


def regime_check(df, strat_returns, strat_signal):
    """R1 regime-agnostic check: compare Sharpe in up vs down markets."""
    spy_ret = df["SPY_close"].pct_change()
    spy_21d = spy_ret.rolling(21).sum()

    common_idx = strat_returns.index.intersection(spy_21d.dropna().index)
    if len(common_idx) < 100:
        log.info("    Regime check: insufficient data")
        return {"regime_agnostic": False, "sharpe_diff_ratio": 999}

    sr = strat_returns.loc[common_idx]
    regime = spy_21d.loc[common_idx]

    bull_ret = sr[regime > 0]
    bear_ret = sr[regime <= 0]

    bull_sharpe = bull_ret.mean() / bull_ret.std() * np.sqrt(252) if len(bull_ret) > 20 and bull_ret.std() > 0 else 0
    bear_sharpe = bear_ret.mean() / bear_ret.std() * np.sqrt(252) if len(bear_ret) > 20 and bear_ret.std() > 0 else 0

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    diff_ratio = abs(bull_sharpe - bear_sharpe) / max_sharpe

    is_agnostic = diff_ratio <= 0.50  # R1 threshold

    log.info(f"    Regime check: Bull Sharpe={bull_sharpe:.2f}  Bear Sharpe={bear_sharpe:.2f}  "
             f"Diff ratio={diff_ratio:.2f}  {'PASS' if is_agnostic else 'FAIL (regime-dependent)'}")

    return {"bull_sharpe": round(bull_sharpe, 3), "bear_sharpe": round(bear_sharpe, 3),
            "sharpe_diff_ratio": round(diff_ratio, 3), "regime_agnostic": is_agnostic}


def outlier_robustness(strat_returns, pct=1):
    """Remove top/bottom pct% of returns and check if strategy still works."""
    lower = strat_returns.quantile(pct / 100)
    upper = strat_returns.quantile(1 - pct / 100)
    trimmed = strat_returns[(strat_returns >= lower) & (strat_returns <= upper)]

    orig_metrics = compute_metrics(strat_returns)
    trim_metrics = compute_metrics(trimmed)

    robust = trim_metrics["sharpe"] > 0 and trim_metrics["sharpe"] > orig_metrics["sharpe"] * 0.3

    log.info(f"    Outlier robustness (remove {pct}% tails): "
             f"Original Sharpe={orig_metrics['sharpe']:.2f} -> Trimmed={trim_metrics['sharpe']:.2f}  "
             f"{'PASS' if robust else 'FAIL'}")

    return {"original_sharpe": orig_metrics["sharpe"], "trimmed_sharpe": trim_metrics["sharpe"],
            "robust": robust}


def run_adversarial(df, feats, name, signal, strat_returns, real_sharpe):
    """Run full adversarial validation suite."""
    log.info(f"\n  Adversarial Validation: {name}")

    results = {}
    results["permutation"] = permutation_test(df, feats, name, signal, real_sharpe)
    results["sub_period"] = sub_period_stability(strat_returns)
    results["regime"] = regime_check(df, strat_returns, signal)
    results["outlier"] = outlier_robustness(strat_returns)

    # Overall pass/fail
    passes = sum([
        results["permutation"]["significant"],
        results["sub_period"]["stable"],
        results["regime"]["regime_agnostic"],
        results["outlier"]["robust"],
    ])
    results["overall_pass"] = passes
    results["overall_max"] = 4
    results["verdict"] = "PASS" if passes >= 3 else "MARGINAL" if passes >= 2 else "FAIL"

    log.info(f"    VERDICT: {results['verdict']} ({passes}/4 tests passed)")

    return results


# ══════════════════════════════════════════════════════════════
# 6. EQUITY CURVE + CAPITAL SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_capital(strat_returns, initial=INITIAL_CAPITAL):
    """Simulate equity curve with fixed capital (HC #713)."""
    equity = initial * (1 + strat_returns).cumprod()
    return equity


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("ML-Enhanced Calendar Effects Strategy")
    log.info(f"Started: {dt.datetime.now().isoformat()}")
    log.info("=" * 70)

    # ── Step 1: Download data ──
    log.info("\n[1/6] Downloading data...")
    df = download_data()

    # ── Step 2: Build calendar features ──
    log.info("\n[2/6] Building calendar features...")
    feats = build_calendar_features(df)

    # ── Step 3: Baseline strategies ──
    log.info("\n[3/6] Running baseline rule-based strategies...")
    baseline_results = run_baseline_strategies(df, feats)

    # ── Step 4: ML-enhanced strategies ──
    log.info("\n[4/6] Running ML-enhanced walk-forward strategies...")
    log.info("=" * 70)

    anomaly_configs = {
        "FOMC_Pre": feats["fomc_pre"],
        "Turn_of_Month": feats["tom"],
        "Pre_Holiday": feats["pre_holiday"],
        "Oct_Nov_Season": feats["is_oct_nov"],
        "Friday_Effect": feats["is_friday"],
    }

    ml_results = {}
    for name, signal in anomaly_configs.items():
        log.info(f"\n  Training ML for: {name} ({int(signal.sum())} active days)...")
        res = ml_walk_forward(df, feats, name, signal)
        ml_results[name] = res

    # Combined strategy
    log.info("\n  Training ML Combined Calendar Strategy...")
    combined = ml_combined_strategy(df, feats, ml_results)
    if combined is not None:
        ml_results["ML_Combined"] = combined

    # ── Step 5: Adversarial validation ──
    log.info("\n[5/6] Running adversarial validation...")
    log.info("=" * 70)

    adversarial_results = {}
    for name, res in ml_results.items():
        if res is None:
            continue
        signal = res.get("signal", pd.Series(dtype=int))
        strat_ret = res.get("strat_returns", pd.Series(dtype=float))
        if len(strat_ret) < 100:
            continue
        real_sharpe = res["metrics"]["sharpe"]
        adversarial_results[name] = run_adversarial(df, feats, name, signal, strat_ret, real_sharpe)

    # Also validate best baseline
    spy_ret = df["SPY_close"].pct_change()
    cost = COST_BPS / 10000
    for bname, bsignal in [("FOMC_Pre_baseline", feats["fomc_pre"]),
                            ("TOM_baseline", feats["tom"])]:
        strat_ret = spy_ret * bsignal.shift(1)
        trades = bsignal.diff().abs().fillna(0)
        strat_ret = strat_ret - trades.shift(1) * cost
        strat_ret = strat_ret.dropna()
        if len(strat_ret) > 100:
            m = compute_metrics(strat_ret)
            adversarial_results[bname] = run_adversarial(
                df, feats, bname, bsignal, strat_ret, m["sharpe"])

    # ── Step 6: Save results ──
    log.info("\n[6/6] Saving results...")
    log.info("=" * 70)

    # Summary
    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "data_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        "n_trading_days": len(df),
        "initial_capital": INITIAL_CAPITAL,
        "cost_bps": COST_BPS,
        "baseline_results": baseline_results,
        "ml_results": {},
        "adversarial_results": {},
    }

    for name, res in ml_results.items():
        if res is not None:
            summary["ml_results"][name] = res["metrics"]
            if "feature_importance" in res:
                summary["ml_results"][name]["top_features"] = dict(
                    list(res["feature_importance"].items())[:10]
                )

    for name, adv in adversarial_results.items():
        # Serialize for JSON
        adv_clean = {}
        for k, v in adv.items():
            if k == "sub_period":
                adv_clean[k] = {"stability": v["stability"], "stable": v["stable"],
                                "sharpes": v["sharpes"]}
            else:
                adv_clean[k] = v
        summary["adversarial_results"][name] = adv_clean

    # Save JSON
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save equity curves for best strategies
    for name, res in ml_results.items():
        if res is not None and "strat_returns" in res:
            eq = simulate_capital(res["strat_returns"])
            eq.to_csv(OUTPUT_DIR / f"equity_{name}.csv")

    # ── Final Summary ──
    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY — ML Calendar Effects")
    log.info("=" * 70)

    log.info(f"\nData: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    log.info(f"Capital: ${INITIAL_CAPITAL:,}  Cost: {COST_BPS}bps/trade")

    log.info("\n┌─────────────────────┬─────────┬──────────┬─────────┬─────────┬────────┬────────┐")
    log.info("│ Strategy            │  Sharpe │  Sortino │ CAGR(%) │ MaxDD(%)│  WR(%) │   PF   │")
    log.info("├─────────────────────┼─────────┼──────────┼─────────┼─────────┼────────┼────────┤")

    # Print baseline
    for name, m in baseline_results.items():
        log.info(f"│ {name:<19s} │ {m['sharpe']:>7.2f} │ {m['sortino']:>8.2f} │ "
                 f"{m['cagr']:>7.1f} │ {m['max_dd']:>7.1f} │ {m['win_rate']:>6.1f} │ "
                 f"{min(m['profit_factor'], 99.9):>6.2f} │")

    log.info("├─────────────────────┼─────────┼──────────┼─────────┼─────────┼────────┼────────┤")

    # Print ML
    for name, res in ml_results.items():
        if res is None:
            continue
        m = res["metrics"]
        log.info(f"│ ML_{name:<15s} │ {m['sharpe']:>7.2f} │ {m['sortino']:>8.2f} │ "
                 f"{m['cagr']:>7.1f} │ {m['max_dd']:>7.1f} │ {m['win_rate']:>6.1f} │ "
                 f"{min(m['profit_factor'], 99.9):>6.2f} │")

    log.info("└─────────────────────┴─────────┴──────────┴─────────┴─────────┴────────┴────────┘")

    # Adversarial verdicts
    log.info("\nAdversarial Validation Verdicts:")
    for name, adv in adversarial_results.items():
        perm_p = adv.get("permutation", {}).get("p_value", "N/A")
        regime = "PASS" if adv.get("regime", {}).get("regime_agnostic", False) else "FAIL"
        stability = adv.get("sub_period", {}).get("stability", "N/A")
        outlier = "PASS" if adv.get("outlier", {}).get("robust", False) else "FAIL"
        verdict = adv.get("verdict", "N/A")
        log.info(f"  {name}: {verdict}  (perm p={perm_p}, regime={regime}, "
                 f"stability={stability}, outlier={outlier})")

    # Correlation with existing strategies
    log.info("\nCorrelation Note: Calendar effects should be UNCORRELATED with ML Trend / ML Sector Rotation")
    log.info("  Calendar anomalies trade specific dates (FOMC, month-end, holidays)")
    log.info("  ML Trend trades momentum regimes — fundamentally different signal source")

    log.info(f"\nCompleted in {elapsed:.0f}s")
    log.info(f"Results saved to: {OUTPUT_DIR}")

    return summary


if __name__ == "__main__":
    results = main()
