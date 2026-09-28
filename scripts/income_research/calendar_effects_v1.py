#!/usr/bin/env python3
"""
Calendar Effects Backtest v1 — Turn-of-Month, Pre-Holiday, January, Santa Rally
================================================================================
Tests well-documented calendar anomalies driven by institutional fund flows:
  1. TOM_last2_first2: Buy close T-2 before month-end, sell close T+2 of new month
  2. TOM_last1_first1: Buy close last day of month, sell close 1st day of next month
  3. Pre_Holiday: Buy close day before holiday, sell close day after
  4. Last_Day_of_Month: Buy close T-1, sell close on last trading day
  5. January_First5: Buy close last trading day Dec, sell close 5th trading day Jan
  6. Santa_Rally: Buy close ~Dec 20, sell close ~Jan 3

Adversarial checks built IN (HC #705):
  - Permutation test: random DATE entry (not return shuffling), p < 0.05
  - Regime stratification: SPY green/red/flat months
  - Sub-period stability: 2010-2015 vs 2016-2020 vs 2021-2026
  - Outlier removal: winsorize at 1st/99th percentile
  - Transaction costs: 0.1% equity slippage, options commissions
  - BS-priced call debit spread overlay for surviving signals

Tickers: SPY (primary), QQQ, IWM for confirmation.
Data: 2010-2026 from yfinance.
"""

import sys, json, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "calendar_effects_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 200       # $200 per trade
EQUITY_SLIPPAGE_PCT    = 0.001     # 0.1% round-trip
OPTIONS_COMMISSION_LEG = 0.65      # $0.65 per leg
RISK_FREE_RATE         = 0.045
N_PERMUTATIONS         = 500       # More permutations for robustness
CONFIDENCE_LEVEL       = 0.05

TICKERS = ['SPY', 'QQQ', 'IWM']

# US Market Holidays (month, day) for fixed holidays
# Variable holidays computed programmatically
FIXED_HOLIDAYS = {
    (1, 1):   "New Year's Day",
    (6, 19):  "Juneteenth",
    (7, 4):   "Independence Day",
    (12, 25): "Christmas Day",
}

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def fetch_prices(tickers, cache_path):
    """Fetch daily OHLCV from yfinance with caching."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Check if we have the tickers we need and sufficient date range
        existing_tickers = df.columns.get_level_values(1).unique() if isinstance(df.columns, pd.MultiIndex) else []
        need_fetch = False
        for t in tickers:
            if t not in existing_tickers:
                need_fetch = True
                break
        if not need_fetch and len(df) > 3000:  # ~12 years of data
            print(f"  Loaded {len(df)} rows from cache")
            return df

    import yfinance as yf
    print(f"  Downloading {tickers} from yfinance (2010-2026)...")
    df = yf.download(tickers, start="2010-01-01", end="2026-07-15",
                     auto_adjust=True, threads=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"  Saved {len(df)} rows to cache")
    return df


def get_close(df, ticker):
    """Extract close prices for a single ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(1):
            return df['Close'][ticker].dropna()
        elif ticker in df.columns.get_level_values(0):
            return df[ticker]['Close'].dropna()
    # Single ticker
    if 'Close' in df.columns:
        return df['Close'].dropna()
    raise ValueError(f"Cannot extract close for {ticker}")


# ═════════════════════════════════════════════════════════════════════════════
# HOLIDAY CALENDAR
# ═════════════════════════════════════════════════════════════════════════════

def nth_weekday_of_month(year, month, weekday, n):
    """Get the nth occurrence of weekday in month. weekday: 0=Mon, 6=Sun."""
    from calendar import monthcalendar
    cal = monthcalendar(year, month)
    days = [week[weekday] for week in cal if week[weekday] != 0]
    if n <= len(days):
        return datetime(year, month, days[n-1])
    return None


def last_weekday_of_month(year, month, weekday):
    """Get last occurrence of weekday in month."""
    from calendar import monthcalendar
    cal = monthcalendar(year, month)
    days = [week[weekday] for week in cal if week[weekday] != 0]
    return datetime(year, month, days[-1])


def compute_easter(year):
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)


def get_market_holidays(year):
    """Return list of market holiday dates for a given year."""
    holidays = []

    # New Year's Day
    d = datetime(year, 1, 1)
    if d.weekday() == 5: d = datetime(year-1, 12, 31)  # Sat -> Fri
    elif d.weekday() == 6: d = datetime(year, 1, 2)      # Sun -> Mon
    holidays.append(("New Year", d))

    # MLK Day: 3rd Monday of January
    mlk = nth_weekday_of_month(year, 1, 0, 3)
    if mlk: holidays.append(("MLK Day", mlk))

    # Presidents' Day: 3rd Monday of February
    pres = nth_weekday_of_month(year, 2, 0, 3)
    if pres: holidays.append(("Presidents Day", pres))

    # Good Friday: 2 days before Easter
    easter = compute_easter(year)
    good_friday = easter - timedelta(days=2)
    holidays.append(("Good Friday", good_friday))

    # Memorial Day: last Monday of May
    mem = last_weekday_of_month(year, 5, 0)
    holidays.append(("Memorial Day", mem))

    # Juneteenth
    d = datetime(year, 6, 19)
    if d.weekday() == 5: d -= timedelta(days=1)
    elif d.weekday() == 6: d += timedelta(days=1)
    if year >= 2021:  # Federal holiday since 2021
        holidays.append(("Juneteenth", d))

    # Independence Day
    d = datetime(year, 7, 4)
    if d.weekday() == 5: d -= timedelta(days=1)
    elif d.weekday() == 6: d += timedelta(days=1)
    holidays.append(("July 4th", d))

    # Labor Day: 1st Monday of September
    labor = nth_weekday_of_month(year, 9, 0, 1)
    if labor: holidays.append(("Labor Day", labor))

    # Thanksgiving: 4th Thursday of November
    thanks = nth_weekday_of_month(year, 11, 3, 4)
    if thanks: holidays.append(("Thanksgiving", thanks))

    # Christmas
    d = datetime(year, 12, 25)
    if d.weekday() == 5: d -= timedelta(days=1)
    elif d.weekday() == 6: d += timedelta(days=1)
    holidays.append(("Christmas", d))

    return holidays


def get_all_holiday_dates(start_year=2010, end_year=2026):
    """Get all market holiday dates across years."""
    all_holidays = []
    for yr in range(start_year, end_year + 1):
        for name, dt in get_market_holidays(yr):
            all_holidays.append((name, dt.date() if isinstance(dt, datetime) else dt))
    return all_holidays


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def identify_calendar_entries(close_prices):
    """
    Identify entry/exit dates for each calendar signal.
    Returns dict: signal_name -> list of (entry_date, exit_date) tuples.
    """
    dates = close_prices.index
    # Convert to python dates for easier comparison
    date_list = [d.date() if hasattr(d, 'date') else d for d in dates]
    date_set = set(date_list)

    def nearest_trading_day(target, direction='before'):
        """Find nearest trading day to target date."""
        td = target if isinstance(target, type(date_list[0])) else target.date()
        for offset in range(10):
            if direction == 'before':
                candidate = td - timedelta(days=offset)
            else:
                candidate = td + timedelta(days=offset)
            if candidate in date_set:
                return candidate
        return None

    def trading_days_before_month_end(dt, n):
        """Get the nth-to-last trading day of the month."""
        yr, mo = dt.year, dt.month
        month_dates = [d for d in date_list if d.year == yr and d.month == mo]
        if len(month_dates) >= n:
            return month_dates[-n]
        return None

    def trading_days_after_month_start(yr, mo, n):
        """Get the nth trading day of the month."""
        month_dates = [d for d in date_list if d.year == yr and d.month == mo]
        if len(month_dates) >= n:
            return month_dates[n-1]
        return None

    signals = defaultdict(list)

    # Group dates by (year, month)
    from itertools import groupby
    months = defaultdict(list)
    for d in date_list:
        months[(d.year, d.month)].append(d)

    sorted_months = sorted(months.keys())

    # ── TOM_last2_first2 ──
    # Buy close 2 trading days before month-end, sell close 2nd trading day of new month
    for i, (yr, mo) in enumerate(sorted_months[:-1]):
        next_yr, next_mo = sorted_months[i+1]
        entry = trading_days_before_month_end(months[(yr, mo)][0], 2)
        exit_d = trading_days_after_month_start(next_yr, next_mo, 2)
        if entry and exit_d and entry in date_set and exit_d in date_set:
            signals['TOM_last2_first2'].append((entry, exit_d))

    # ── TOM_last1_first1 ──
    # Buy close last trading day of month, sell close 1st trading day of next month
    for i, (yr, mo) in enumerate(sorted_months[:-1]):
        next_yr, next_mo = sorted_months[i+1]
        entry = months[(yr, mo)][-1]  # last trading day
        exit_d = months[(next_yr, next_mo)][0]  # first trading day
        if entry in date_set and exit_d in date_set:
            signals['TOM_last1_first1'].append((entry, exit_d))

    # ── Last_Day_of_Month ──
    # Buy close 2nd-to-last trading day, sell close last trading day
    for (yr, mo), dts in months.items():
        if len(dts) >= 2:
            entry = dts[-2]
            exit_d = dts[-1]
            signals['Last_Day_of_Month'].append((entry, exit_d))

    # ── Pre_Holiday ──
    # Buy close day before market holiday, sell close day after
    all_holidays = get_all_holiday_dates()
    for name, hol_date in all_holidays:
        # Find last trading day before holiday
        entry = None
        for offset in range(1, 10):
            candidate = hol_date - timedelta(days=offset)
            if candidate in date_set:
                entry = candidate
                break
        # Find first trading day after holiday
        exit_d = None
        for offset in range(1, 10):
            candidate = hol_date + timedelta(days=offset)
            if candidate in date_set:
                exit_d = candidate
                break
        if entry and exit_d:
            signals['Pre_Holiday'].append((entry, exit_d))

    # ── January_First5 ──
    # Buy close last trading day of December, sell close 5th trading day of January
    for yr in range(2010, 2027):
        if (yr, 12) in months and (yr+1, 1) in months:
            entry = months[(yr, 12)][-1]
            jan_days = months[(yr+1, 1)]
            if len(jan_days) >= 5:
                exit_d = jan_days[4]
                signals['January_First5'].append((entry, exit_d))

    # ── Santa_Rally ──
    # Buy close ~Dec 20, sell close ~Jan 3
    for yr in range(2010, 2027):
        if (yr, 12) in months and (yr+1, 1) in months:
            dec_days = months[(yr, 12)]
            # Find trading day closest to Dec 20
            entry = None
            for d in dec_days:
                if d.day >= 18 and d.day <= 22:
                    entry = d
            if not entry:
                # Fallback: 8th-to-last trading day of Dec
                if len(dec_days) >= 8:
                    entry = dec_days[-8]
            jan_days = months[(yr+1, 1)]
            # Find trading day closest to Jan 3
            exit_d = None
            for d in jan_days:
                if d.day >= 2 and d.day <= 4:
                    exit_d = d
            if not exit_d and len(jan_days) >= 2:
                exit_d = jan_days[1]
            if entry and exit_d:
                signals['Santa_Rally'].append((entry, exit_d))

    return signals


# ═════════════════════════════════════════════════════════════════════════════
# EQUITY BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def run_equity_backtest(close_prices, trades, slippage_pct=EQUITY_SLIPPAGE_PCT):
    """
    Run equity sim: buy at entry close, sell at exit close.
    Returns per-trade results.
    """
    results = []
    for entry_date, exit_date in trades:
        try:
            entry_ts = pd.Timestamp(entry_date)
            exit_ts = pd.Timestamp(exit_date)
            if entry_ts not in close_prices.index or exit_ts not in close_prices.index:
                continue
            entry_price = close_prices.loc[entry_ts]
            exit_price = close_prices.loc[exit_ts]
            if pd.isna(entry_price) or pd.isna(exit_price):
                continue
            # Apply slippage
            entry_cost = entry_price * (1 + slippage_pct / 2)
            exit_proceeds = exit_price * (1 - slippage_pct / 2)
            ret = (exit_proceeds - entry_cost) / entry_cost
            hold_days = (exit_ts - entry_ts).days
            results.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': ret,
                'hold_days': hold_days,
                'pnl_dollars': ret * RISK_PER_TRADE,
            })
        except Exception:
            continue
    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONS OVERLAY — BS-priced call debit spread
# ═════════════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_debit_spread(S, hold_days, ret_pct, sigma=0.18, r=RISK_FREE_RATE):
    """
    Price ATM / ATM+1% call debit spread.
    Returns (entry_debit, exit_value, net_pnl_per_spread).
    """
    K_low = S                        # ATM
    K_high = S * 1.01                # ATM + 1%
    T_entry = max(hold_days + 7, 14) / 252   # Buy with ~1-2 week buffer
    T_exit = T_entry - hold_days / 252

    # Entry: buy K_low call, sell K_high call
    entry_debit = bs_call_price(S, K_low, T_entry, r, sigma) - \
                  bs_call_price(S, K_high, T_entry, r, sigma)

    # Exit: stock moved by ret_pct
    S_exit = S * (1 + ret_pct)
    exit_value = bs_call_price(S_exit, K_low, max(T_exit, 1/252), r, sigma) - \
                 bs_call_price(S_exit, K_high, max(T_exit, 1/252), r, sigma)

    # Commissions: 4 legs total (open + close)
    commission = 4 * OPTIONS_COMMISSION_LEG

    # How many spreads can we buy with RISK_PER_TRADE?
    if entry_debit <= 0:
        return 0, 0, 0
    n_contracts = max(1, int(RISK_PER_TRADE / (entry_debit * 100)))
    net_pnl = n_contracts * (exit_value - entry_debit) * 100 - commission

    return entry_debit, exit_value, net_pnl


def run_options_overlay(equity_results, close_prices):
    """Apply BS-priced call debit spread to equity trades."""
    if equity_results.empty:
        return pd.DataFrame()

    # Estimate historical vol
    log_ret = np.log(close_prices / close_prices.shift(1)).dropna()
    hist_vol = log_ret.rolling(21).std() * np.sqrt(252)

    opts_results = []
    for _, row in equity_results.iterrows():
        entry_ts = pd.Timestamp(row['entry_date'])
        if entry_ts in hist_vol.index:
            sigma = hist_vol.loc[entry_ts]
            if pd.isna(sigma) or sigma <= 0:
                sigma = 0.18
        else:
            sigma = 0.18

        entry_debit, exit_value, net_pnl = price_debit_spread(
            row['entry_price'], max(row['hold_days'], 1), row['return'], sigma
        )
        opts_results.append({
            'entry_date': row['entry_date'],
            'exit_date': row['exit_date'],
            'entry_debit': entry_debit,
            'exit_value': exit_value,
            'net_pnl': net_pnl,
            'equity_return': row['return'],
        })
    return pd.DataFrame(opts_results)


# ═════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL CHECKS (ALL BUILT IN)
# ═════════════════════════════════════════════════════════════════════════════

def permutation_test_date_entry(close_prices, observed_trades, observed_mean_ret, n_perms=N_PERMUTATIONS):
    """
    CRITICAL: Random DATE entry permutation (not return shuffling).
    For each permutation, randomly select N entry dates from all available trading days,
    compute forward returns with same hold periods, compare mean to observed.
    """
    n_trades = len(observed_trades)
    if n_trades < 5:
        return 1.0, 0  # Not enough trades

    # Compute hold periods for each observed trade
    hold_periods = []
    for entry_d, exit_d in observed_trades:
        entry_ts = pd.Timestamp(entry_d)
        exit_ts = pd.Timestamp(exit_d)
        if entry_ts in close_prices.index and exit_ts in close_prices.index:
            entry_idx = close_prices.index.get_loc(entry_ts)
            exit_idx = close_prices.index.get_loc(exit_ts)
            hold_periods.append(exit_idx - entry_idx)
    if not hold_periods:
        return 1.0, 0

    median_hold = int(np.median(hold_periods))
    if median_hold < 1:
        median_hold = 1

    # All possible entry indices (leave room for hold period)
    all_indices = np.arange(0, len(close_prices) - median_hold)
    prices = close_prices.values

    count_ge = 0
    perm_means = []
    for _ in range(n_perms):
        # Random entry dates
        chosen = np.random.choice(all_indices, size=n_trades, replace=True)
        rets = (prices[chosen + median_hold] - prices[chosen]) / prices[chosen]
        # Apply slippage
        rets -= EQUITY_SLIPPAGE_PCT
        pm = np.mean(rets)
        perm_means.append(pm)
        if pm >= observed_mean_ret:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perms + 1)
    return p_value, np.mean(perm_means)


def regime_stratification(equity_results, spy_close):
    """
    Stratify trades by SPY regime (green/red/flat months).
    Reject if |Sharpe_green - Sharpe_red| / max > 0.50.
    """
    if equity_results.empty or len(equity_results) < 10:
        return None, True  # Not enough data, pass vacuously

    # Monthly SPY returns for regime classification
    spy_monthly = spy_close.resample('ME').last().pct_change().dropna()

    results_with_regime = []
    for _, row in equity_results.iterrows():
        entry_ts = pd.Timestamp(row['entry_date'])
        # Find the month
        month_key = entry_ts.to_period('M').to_timestamp()
        # Classify regime
        closest = spy_monthly.index[spy_monthly.index <= entry_ts]
        if len(closest) > 0:
            mo_ret = spy_monthly.loc[closest[-1]]
            if mo_ret > 0.02:
                regime = 'green'
            elif mo_ret < -0.02:
                regime = 'red'
            else:
                regime = 'flat'
        else:
            regime = 'flat'
        row_dict = row.to_dict()
        row_dict['regime'] = regime
        results_with_regime.append(row_dict)

    regime_df = pd.DataFrame(results_with_regime)
    regime_stats = {}
    for regime in ['green', 'red', 'flat']:
        subset = regime_df[regime_df['regime'] == regime]
        if len(subset) >= 3:
            rets = subset['return'].values
            sr = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252 / max(np.mean(subset['hold_days']), 1))
            regime_stats[regime] = {
                'n_trades': len(subset),
                'mean_ret': float(np.mean(rets)),
                'win_rate': float(np.mean(rets > 0)),
                'sharpe': float(sr),
            }

    # Check regime gap
    passed = True
    if 'green' in regime_stats and 'red' in regime_stats:
        sg = regime_stats['green']['sharpe']
        sr = regime_stats['red']['sharpe']
        gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.01)
        if gap > 0.50:
            passed = False

    return regime_stats, passed


def sub_period_stability(equity_results):
    """Check if edge is stable across sub-periods."""
    if equity_results.empty or len(equity_results) < 10:
        return None, True

    periods = {
        '2010-2015': (datetime(2010,1,1), datetime(2015,12,31)),
        '2016-2020': (datetime(2016,1,1), datetime(2020,12,31)),
        '2021-2026': (datetime(2021,1,1), datetime(2026,12,31)),
    }

    period_stats = {}
    for name, (start, end) in periods.items():
        mask = equity_results['entry_date'].apply(
            lambda d: start.date() <= (d if not hasattr(d, 'date') else d) <= end.date()
        )
        subset = equity_results[mask]
        if len(subset) >= 3:
            rets = subset['return'].values
            period_stats[name] = {
                'n_trades': len(subset),
                'mean_ret': float(np.mean(rets)),
                'win_rate': float(np.mean(rets > 0)),
                'total_pnl': float(np.sum(subset['pnl_dollars'])),
            }

    # Check: edge should be positive in at least 2 of 3 periods
    positive_periods = sum(1 for s in period_stats.values() if s['mean_ret'] > 0)
    passed = positive_periods >= 2

    return period_stats, passed


def outlier_robustness(equity_results):
    """Winsorize at 1st/99th percentile and recheck edge."""
    if equity_results.empty or len(equity_results) < 10:
        return None, True

    rets = equity_results['return'].values
    p1, p99 = np.percentile(rets, [1, 99])
    winsorized = np.clip(rets, p1, p99)

    original_mean = np.mean(rets)
    winsorized_mean = np.mean(winsorized)

    # Edge should survive winsorization (at least 50% of original)
    if original_mean > 0:
        passed = winsorized_mean > 0 and winsorized_mean >= original_mean * 0.5
    else:
        passed = True  # Already negative, not an edge

    return {
        'original_mean': float(original_mean),
        'winsorized_mean': float(winsorized_mean),
        'retention_pct': float(winsorized_mean / original_mean * 100) if original_mean != 0 else 0,
    }, passed


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity_results):
    """Compute strategy performance metrics."""
    if equity_results.empty:
        return {}

    rets = equity_results['return'].values
    pnls = equity_results['pnl_dollars'].values
    hold = equity_results['hold_days'].values

    n = len(rets)
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    win_rate = np.mean(rets > 0)

    # Annualized Sharpe (approximate based on avg hold)
    avg_hold = max(np.mean(hold), 1)
    trades_per_year = 252 / avg_hold
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(trades_per_year)

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_ret
    sortino = (mean_ret / (downside_std + 1e-10)) * np.sqrt(trades_per_year)

    # Profit Factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown on cumulative equity
    cum_pnl = np.cumsum(pnls) + STARTING_CAPITAL
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = (cum_pnl - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # Equity curve
    final_equity = cum_pnl[-1]
    total_return_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    return {
        'n_trades': n,
        'mean_return_pct': float(mean_ret * 100),
        'std_return_pct': float(std_ret * 100),
        'win_rate': float(win_rate),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(profit_factor),
        'max_drawdown_pct': float(max_dd * 100),
        'total_pnl': float(np.sum(pnls)),
        'final_equity': float(final_equity),
        'total_return_pct': float(total_return_pct),
        'avg_hold_days': float(avg_hold),
        'trades_per_year': float(trades_per_year),
        'avg_pnl_per_trade': float(np.mean(pnls)),
        'median_return_pct': float(np.median(rets) * 100),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("CALENDAR EFFECTS BACKTEST v1")
    print("=" * 80)
    print()

    # ── Load Data ──
    print("[1/5] Loading price data...")
    cache_path = OUTPUT / "price_cache.parquet"
    existing_cache = ROOT / "output" / "oversold_bounce_v1" / "spy_cache.parquet"

    # Try to use existing SPY cache as starting point
    raw = fetch_prices(TICKERS, cache_path)

    # Extract SPY close for regime analysis
    spy_close = get_close(raw, 'SPY')
    print(f"  SPY: {len(spy_close)} trading days, {spy_close.index[0].date()} to {spy_close.index[-1].date()}")

    all_results = {}

    for ticker in TICKERS:
        print(f"\n{'─' * 70}")
        print(f"  TICKER: {ticker}")
        print(f"{'─' * 70}")

        close = get_close(raw, ticker)
        if len(close) < 500:
            print(f"  SKIP: only {len(close)} data points")
            continue

        # ── Generate signals ──
        print(f"\n[2/5] Identifying calendar entries for {ticker}...")
        signals = identify_calendar_entries(close)

        ticker_results = {}

        for sig_name, trades in sorted(signals.items()):
            print(f"\n  ── {sig_name} ({len(trades)} trades) ──")
            if len(trades) < 10:
                print(f"    SKIP: only {len(trades)} events (need >= 10)")
                continue

            # ── Run equity backtest ──
            eq_results = run_equity_backtest(close, trades)
            if eq_results.empty or len(eq_results) < 10:
                print(f"    SKIP: only {len(eq_results)} valid trades after filtering")
                continue

            metrics = compute_metrics(eq_results)
            print(f"    Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1%}, "
                  f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}, "
                  f"PF: {metrics['profit_factor']:.2f}")
            print(f"    Mean ret: {metrics['mean_return_pct']:.3f}%, "
                  f"Total P&L: ${metrics['total_pnl']:.0f}, Max DD: {metrics['max_drawdown_pct']:.1f}%")

            # ── Adversarial Check 1: Permutation Test (Random Date Entry) ──
            print(f"    [Perm] Running {N_PERMUTATIONS} random-date permutations...")
            p_value, perm_mean = permutation_test_date_entry(
                close, trades, np.mean(eq_results['return'].values)
            )
            perm_pass = p_value < CONFIDENCE_LEVEL
            print(f"    [Perm] p={p_value:.4f} (need <{CONFIDENCE_LEVEL}) → {'PASS' if perm_pass else 'FAIL'}")
            if not perm_pass:
                print(f"    [Perm] Observed mean: {np.mean(eq_results['return'].values)*100:.3f}%, "
                      f"Random date mean: {perm_mean*100:.3f}%")

            # ── Adversarial Check 2: Regime Stratification ──
            regime_stats, regime_pass = regime_stratification(eq_results, spy_close)
            if regime_stats:
                for regime, stats in regime_stats.items():
                    print(f"    [Regime] {regime:5s}: n={stats['n_trades']:3d}, "
                          f"WR={stats['win_rate']:.1%}, Sharpe={stats['sharpe']:.2f}")
            print(f"    [Regime] → {'PASS' if regime_pass else 'FAIL (regime-dependent)' }")

            # ── Adversarial Check 3: Sub-period Stability ──
            period_stats, period_pass = sub_period_stability(eq_results)
            if period_stats:
                for period, stats in period_stats.items():
                    print(f"    [Period] {period}: n={stats['n_trades']:3d}, "
                          f"mean={stats['mean_ret']*100:.3f}%, WR={stats['win_rate']:.1%}")
            print(f"    [Period] → {'PASS' if period_pass else 'FAIL (not stable across periods)'}")

            # ── Adversarial Check 4: Outlier Robustness ──
            outlier_stats, outlier_pass = outlier_robustness(eq_results)
            if outlier_stats:
                print(f"    [Outlier] Original: {outlier_stats['original_mean']*100:.3f}%, "
                      f"Winsorized: {outlier_stats['winsorized_mean']*100:.3f}%, "
                      f"Retention: {outlier_stats['retention_pct']:.0f}%")
            print(f"    [Outlier] → {'PASS' if outlier_pass else 'FAIL'}")

            # ── Summary ──
            all_pass = perm_pass and regime_pass and period_pass and outlier_pass
            n_pass = sum([perm_pass, regime_pass, period_pass, outlier_pass])
            verdict = "PASS ALL" if all_pass else f"FAIL ({4 - n_pass}/4 checks failed)"
            print(f"\n    ★ VERDICT: {verdict}")

            # ── Options overlay for surviving signals ──
            opts_results = None
            if metrics['mean_return_pct'] > 0:
                opts_df = run_options_overlay(eq_results, close)
                if not opts_df.empty:
                    opts_pnl = opts_df['net_pnl'].sum()
                    opts_wr = (opts_df['net_pnl'] > 0).mean()
                    print(f"    [Options] Debit spread overlay: ${opts_pnl:.0f} total, "
                          f"WR={opts_wr:.1%}, avg=${opts_df['net_pnl'].mean():.1f}/trade")
                    opts_results = {
                        'total_pnl': float(opts_pnl),
                        'win_rate': float(opts_wr),
                        'avg_pnl': float(opts_df['net_pnl'].mean()),
                        'n_trades': len(opts_df),
                    }

            ticker_results[sig_name] = {
                'metrics': metrics,
                'permutation': {'p_value': float(p_value), 'passed': perm_pass, 'perm_mean': float(perm_mean)},
                'regime': {'stats': regime_stats, 'passed': regime_pass},
                'sub_period': {'stats': period_stats, 'passed': period_pass},
                'outlier': {'stats': outlier_stats, 'passed': outlier_pass},
                'all_checks_passed': all_pass,
                'n_checks_passed': n_pass,
                'verdict': verdict,
                'options_overlay': opts_results,
            }

        all_results[ticker] = ticker_results

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("FINAL SUMMARY — CALENDAR EFFECTS BACKTEST")
    print("=" * 80)

    summary_rows = []
    for ticker, signals in all_results.items():
        for sig, data in signals.items():
            m = data['metrics']
            summary_rows.append({
                'Ticker': ticker,
                'Signal': sig,
                'N': m['n_trades'],
                'WR': f"{m['win_rate']:.1%}",
                'Sharpe': f"{m['sharpe']:.2f}",
                'Sortino': f"{m['sortino']:.2f}",
                'PF': f"{m['profit_factor']:.2f}",
                'Mean%': f"{m['mean_return_pct']:.3f}",
                'TotalPnL': f"${m['total_pnl']:.0f}",
                'Perm_p': f"{data['permutation']['p_value']:.3f}",
                'Checks': f"{data['n_checks_passed']}/4",
                'Verdict': 'PASS' if data['all_checks_passed'] else 'FAIL',
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print()
        print(summary_df.to_string(index=False))

    # ── Surviving strategies ──
    print("\n" + "─" * 70)
    print("SURVIVING STRATEGIES (passed ALL 4 checks on SPY):")
    print("─" * 70)
    survivors = []
    for sig, data in all_results.get('SPY', {}).items():
        if data['all_checks_passed']:
            m = data['metrics']
            survivors.append(sig)
            print(f"  {sig}: Sharpe={m['sharpe']:.2f}, Sortino={m['sortino']:.2f}, "
                  f"WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}")
            if data['options_overlay']:
                o = data['options_overlay']
                print(f"    Options: ${o['total_pnl']:.0f} total, WR={o['win_rate']:.1%}")
            # Cross-confirmation
            confirmed = []
            for t in ['QQQ', 'IWM']:
                if t in all_results and sig in all_results[t]:
                    td = all_results[t][sig]
                    if td['metrics']['mean_return_pct'] > 0:
                        confirmed.append(t)
            if confirmed:
                print(f"    Cross-confirmed on: {', '.join(confirmed)}")

    if not survivors:
        print("  None passed all 4 adversarial checks.")
        # Show best partial passes
        print("\n  Best partial passes:")
        best = sorted(all_results.get('SPY', {}).items(),
                      key=lambda x: x[1]['n_checks_passed'], reverse=True)
        for sig, data in best[:3]:
            m = data['metrics']
            print(f"  {sig}: {data['n_checks_passed']}/4 checks, "
                  f"Sharpe={m['sharpe']:.2f}, WR={m['win_rate']:.1%}, "
                  f"perm_p={data['permutation']['p_value']:.3f}")

    # ── Robinhood Implementation Notes ──
    print("\n" + "─" * 70)
    print("ROBINHOOD IMPLEMENTATION ($440 account, Level 2 options):")
    print("─" * 70)
    if survivors:
        print("  For surviving signals:")
        print("  - Use SPY call debit spreads (ATM / ATM+1%)")
        print("  - ~1 contract per trade (~$50-150 risk)")
        print("  - Max 2 concurrent positions")
        print("  - Set calendar alerts for entry dates")
        print("  - Entry: buy at 3:55pm on signal day")
        print("  - Exit: sell at 3:55pm on exit day")
    else:
        print("  No strategies survived all adversarial checks.")
        print("  Calendar effects may be too weak for small-account options trading")
        print("  after transaction costs.")

    # ── Save results ──
    results_file = OUTPUT / "results.json"
    # Convert any non-serializable types
    def clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        if isinstance(obj, type(pd.NaT)):
            return None
        if hasattr(obj, 'date'):
            return str(obj)
        return obj

    import json

    def json_serialize(obj):
        if isinstance(obj, dict):
            return {k: json_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_serialize(v) for v in obj]
        return clean_for_json(obj)

    with open(results_file, 'w') as f:
        json.dump(json_serialize(all_results), f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    # Save summary CSV
    if summary_rows:
        summary_df.to_csv(OUTPUT / "summary.csv", index=False)
        print(f"  Summary saved to {OUTPUT / 'summary.csv'}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    np.random.seed(42)
    results = main()
