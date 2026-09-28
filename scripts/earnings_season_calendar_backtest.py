#!/usr/bin/env python3
"""
Earnings Season Calendar Pattern Backtest
==========================================
Trades the predictable 4x/year earnings season calendar pattern.
6 variants (A-F) tested with permutation validation.

OOT: Jan 2022 - Jul 2026 | Capital: $645 | Slippage: 0.02% | Commission: $0
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
START_DATE = "2022-01-01"
END_DATE = "2026-07-29"
N_PERMUTATIONS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/earnings_season_calendar_results.json")

# Earnings season windows (month, day_start, month_end, day_end)
EARNINGS_WINDOWS = [
    (1, 15, 2, 5),   # Q4 reports
    (4, 15, 5, 5),   # Q1 reports
    (7, 15, 8, 5),   # Q2 reports
    (10, 15, 11, 5), # Q3 reports
]


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download required price data."""
    tickers = ["SPY", "QQQ", "XLK", "XLC", "^VIX"]
    data = {}
    for t in tickers:
        key = t.replace("^", "")
        df = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[key] = df
    return data


# ── Earnings Season Helpers ────────────────────────────────────────────────
def get_earnings_windows(years, offset_days=0):
    """Return list of (start_date, end_date) for earnings season windows.
    offset_days shifts the windows for permutation testing."""
    windows = []
    for year in years:
        for m_start, d_start, m_end, d_end in EARNINGS_WINDOWS:
            try:
                s = dt.date(year, m_start, d_start) + dt.timedelta(days=offset_days)
                e = dt.date(year, m_end, d_end) + dt.timedelta(days=offset_days)
                windows.append((s, e))
            except ValueError:
                continue
    return windows


def is_in_earnings_season(date, windows):
    """Check if a date falls within any earnings window."""
    d = date.date() if hasattr(date, 'date') else date
    for s, e in windows:
        if s <= d <= e:
            return True
    return False


def get_peak_week_start_dates(windows, trading_dates):
    """Get the start of 'peak week' (first full trading week) in each window."""
    peak_starts = []
    td_set = set(trading_dates)
    for s, e in windows:
        # Find first Monday >= s that has trading days
        d = s
        while d <= e:
            if d.weekday() == 0 and d in td_set:
                peak_starts.append(d)
                break
            d += dt.timedelta(days=1)
        else:
            # No Monday found, use first trading day
            for d2 in sorted(td_set):
                if s <= d2 <= e:
                    peak_starts.append(d2)
                    break
    return peak_starts


def get_peak_week_end_dates(windows, trading_dates):
    """Get the Friday ending peak week in each window."""
    td_sorted = sorted(trading_dates)
    peak_ends = []
    for s, e in windows:
        # Peak week = first full week. Find end of that week (Friday)
        d = s
        while d <= e:
            if d.weekday() == 0 and d in set(td_sorted):
                # Found Monday, find Friday
                fri = d + dt.timedelta(days=4)
                # Find nearest trading day <= fri
                best = None
                for td in td_sorted:
                    if d <= td <= fri:
                        best = td
                if best:
                    peak_ends.append(best)
                break
            d += dt.timedelta(days=1)
    return peak_ends


def find_next_trading_day(target, trading_dates, max_search=10):
    """Find nearest trading day >= target."""
    for i in range(max_search):
        candidate = target + dt.timedelta(days=i)
        if candidate in trading_dates:
            return candidate
    return None


def find_trading_day_offset(target, trading_dates_sorted, offset):
    """Find the trading day that is 'offset' trading days from target."""
    try:
        idx = trading_dates_sorted.index(target)
        new_idx = idx + offset
        if 0 <= new_idx < len(trading_dates_sorted):
            return trading_dates_sorted[new_idx]
    except ValueError:
        pass
    return None


# ── Backtest Engine ────────────────────────────────────────────────────────
def apply_slippage(price, direction="buy"):
    """Apply slippage to price."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def run_trades(trades, initial_capital=INITIAL_CAPITAL):
    """Run a list of trades and compute equity curve + metrics.
    Each trade: dict with 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'ticker'
    """
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "total_return_pct": 0,
            "mdd_pct": 0, "n_trades": 0, "win_rate": 0,
            "profit_factor": 0, "trades": [],
            "daily_returns": [],
        }

    trades = sorted(trades, key=lambda t: t["entry_date"])
    capital = initial_capital
    equity_curve = [capital]
    trade_results = []
    daily_returns = []

    for t in trades:
        entry_p = apply_slippage(t["entry_price"], "buy")
        exit_p = apply_slippage(t["exit_price"], "sell")

        shares = int(capital / entry_p)
        if shares <= 0:
            continue

        pnl = (exit_p - entry_p) * shares
        ret = (exit_p - entry_p) / entry_p
        capital += pnl
        equity_curve.append(capital)
        daily_returns.append(ret)

        trade_results.append({
            "entry_date": str(t["entry_date"]),
            "exit_date": str(t["exit_date"]),
            "ticker": t["ticker"],
            "entry_price": round(entry_p, 2),
            "exit_price": round(exit_p, 2),
            "shares": shares,
            "pnl": round(pnl, 2),
            "return_pct": round(ret * 100, 3),
        })

    # Metrics
    rets = np.array(daily_returns)
    n_trades = len(trade_results)

    if n_trades == 0:
        return {
            "sharpe": 0, "sortino": 0, "total_return_pct": 0,
            "mdd_pct": 0, "n_trades": 0, "win_rate": 0,
            "profit_factor": 0, "trades": [], "daily_returns": [],
        }

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0

    # Annualize: ~4 trades/year, so multiply by sqrt(4)
    trades_per_year = max(n_trades / 4.5, 1)  # ~4.5 years of data
    mean_ret = np.mean(rets)
    std_ret = np.std(rets) if np.std(rets) > 0 else 1e-10
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year)

    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 and np.std(downside) > 0 else 1e-10
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year)

    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    profit_factor = gross_wins / gross_losses

    # MDD from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(np.min(dd)) * 100

    total_return = (capital - initial_capital) / initial_capital * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_return, 2),
        "mdd_pct": round(mdd, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "final_capital": round(capital, 2),
        "trades": trade_results,
        "daily_returns": [round(r, 6) for r in daily_returns],
    }


# ── Strategy Implementations ──────────────────────────────────────────────
def strategy_a_pre_earnings_rally(data, windows=None):
    """Buy SPY 5 trading days before peak earnings week, sell at end of peak week."""
    spy = data["SPY"]
    trading_dates = sorted([d.date() for d in spy.index])
    td_set = set(trading_dates)
    if windows is None:
        years = list(range(2022, 2027))
        windows = get_earnings_windows(years)

    trades = []
    for s, e in windows:
        # Find first trading day in window
        peak_start = find_next_trading_day(s, td_set)
        if peak_start is None:
            continue
        # Entry = 5 trading days before peak_start
        entry_date = find_trading_day_offset(peak_start, trading_dates, -5)
        if entry_date is None:
            continue
        # Exit = end of peak week (~5 trading days into window)
        exit_date = find_trading_day_offset(peak_start, trading_dates, 4)
        if exit_date is None:
            continue

        entry_ts = pd.Timestamp(entry_date)
        exit_ts = pd.Timestamp(exit_date)
        if entry_ts not in spy.index or exit_ts not in spy.index:
            continue

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": float(spy.loc[entry_ts, "Close"]),
            "exit_price": float(spy.loc[exit_ts, "Close"]),
            "ticker": "SPY",
        })
    return run_trades(trades)


def strategy_b_post_earnings_drift(data, windows=None):
    """Buy QQQ at end of peak earnings week, hold 15 trading days."""
    qqq = data["QQQ"]
    trading_dates = sorted([d.date() for d in qqq.index])
    td_set = set(trading_dates)
    if windows is None:
        years = list(range(2022, 2027))
        windows = get_earnings_windows(years)

    trades = []
    for s, e in windows:
        peak_start = find_next_trading_day(s, td_set)
        if peak_start is None:
            continue
        # End of peak week
        peak_end = find_trading_day_offset(peak_start, trading_dates, 4)
        if peak_end is None:
            continue
        # Hold 15 trading days
        exit_date = find_trading_day_offset(peak_end, trading_dates, 15)
        if exit_date is None:
            continue

        entry_ts = pd.Timestamp(peak_end)
        exit_ts = pd.Timestamp(exit_date)
        if entry_ts not in qqq.index or exit_ts not in qqq.index:
            continue

        trades.append({
            "entry_date": peak_end,
            "exit_date": exit_date,
            "entry_price": float(qqq.loc[entry_ts, "Close"]),
            "exit_price": float(qqq.loc[exit_ts, "Close"]),
            "ticker": "QQQ",
        })
    return run_trades(trades)


def strategy_c_sector_rotation(data, windows=None):
    """During earnings season hold XLK+XLC, between seasons hold SPY."""
    spy = data["SPY"]
    xlk = data["XLK"]
    xlc = data["XLC"]
    trading_dates = sorted([d.date() for d in spy.index])
    if windows is None:
        years = list(range(2022, 2027))
        windows = get_earnings_windows(years)

    # Build rotation schedule
    trades = []
    # During earnings season: XLK + XLC (split capital)
    for s, e in windows:
        entry_date = find_next_trading_day(s, set(trading_dates))
        exit_date = find_next_trading_day(e, set(trading_dates))
        if entry_date is None or exit_date is None:
            continue
        if entry_date >= exit_date:
            continue

        entry_ts = pd.Timestamp(entry_date)
        exit_ts = pd.Timestamp(exit_date)

        for ticker, df in [("XLK", xlk), ("XLC", xlc)]:
            if entry_ts in df.index and exit_ts in df.index:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(df.loc[entry_ts, "Close"]),
                    "exit_price": float(df.loc[exit_ts, "Close"]),
                    "ticker": ticker,
                })

    # Between seasons: hold SPY
    sorted_windows = sorted(windows, key=lambda w: w[0])
    for i in range(len(sorted_windows) - 1):
        gap_start = sorted_windows[i][1] + dt.timedelta(days=1)
        gap_end = sorted_windows[i + 1][0] - dt.timedelta(days=1)
        entry_date = find_next_trading_day(gap_start, set(trading_dates))
        exit_date = find_next_trading_day(gap_end, set(trading_dates))
        if entry_date is None or exit_date is None:
            continue
        if entry_date >= exit_date:
            continue

        entry_ts = pd.Timestamp(entry_date)
        exit_ts = pd.Timestamp(exit_date)
        if entry_ts in spy.index and exit_ts in spy.index:
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": float(spy.loc[entry_ts, "Close"]),
                "exit_price": float(spy.loc[exit_ts, "Close"]),
                "ticker": "SPY",
            })

    # For sector rotation, run trades sequentially managing capital manually
    # since we have overlapping XLK/XLC trades during earnings season
    if not trades:
        return run_trades([])

    # Handle the split: during earnings, split capital 50/50 for XLK/XLC
    trades_sorted = sorted(trades, key=lambda t: (t["entry_date"], t["ticker"]))
    capital = INITIAL_CAPITAL
    results_list = []
    daily_rets = []
    i = 0
    while i < len(trades_sorted):
        t = trades_sorted[i]
        # Check if next trade has same entry date (XLK+XLC pair)
        if i + 1 < len(trades_sorted) and trades_sorted[i + 1]["entry_date"] == t["entry_date"]:
            # Split capital
            t2 = trades_sorted[i + 1]
            half_cap = capital / 2
            for sub_t, sub_cap in [(t, half_cap), (t2, half_cap)]:
                ep = apply_slippage(sub_t["entry_price"], "buy")
                xp = apply_slippage(sub_t["exit_price"], "sell")
                shares = int(sub_cap / ep)
                if shares > 0:
                    pnl = (xp - ep) * shares
                    ret = (xp - ep) / ep
                    capital += pnl
                    daily_rets.append(ret)
                    results_list.append({
                        "entry_date": str(sub_t["entry_date"]),
                        "exit_date": str(sub_t["exit_date"]),
                        "ticker": sub_t["ticker"],
                        "pnl": round(pnl, 2),
                        "return_pct": round(ret * 100, 3),
                    })
            i += 2
        else:
            ep = apply_slippage(t["entry_price"], "buy")
            xp = apply_slippage(t["exit_price"], "sell")
            shares = int(capital / ep)
            if shares > 0:
                pnl = (xp - ep) * shares
                ret = (xp - ep) / ep
                capital += pnl
                daily_rets.append(ret)
                results_list.append({
                    "entry_date": str(t["entry_date"]),
                    "exit_date": str(t["exit_date"]),
                    "ticker": t["ticker"],
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret * 100, 3),
                })
            i += 1

    rets = np.array(daily_rets)
    n = len(rets)
    if n == 0:
        return run_trades([])

    wins = rets[rets > 0]
    losses = rets[rets < 0]
    tpy = max(n / 4.5, 1)
    mean_r = np.mean(rets)
    std_r = np.std(rets) if np.std(rets) > 0 else 1e-10
    sharpe = (mean_r / std_r) * np.sqrt(tpy)
    ds = rets[rets < 0]
    ds_std = np.std(ds) if len(ds) > 0 and np.std(ds) > 0 else 1e-10
    sortino = (mean_r / ds_std) * np.sqrt(tpy)
    gw = float(np.sum(wins)) if len(wins) > 0 else 0
    gl = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-10
    pf = gw / gl
    eq = [INITIAL_CAPITAL]
    c = INITIAL_CAPITAL
    for r in rets:
        c *= (1 + r)
        eq.append(c)
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(np.min(dd)) * 100
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_ret, 2),
        "mdd_pct": round(mdd, 2),
        "n_trades": n,
        "win_rate": round((len(wins) / n) * 100, 1),
        "profit_factor": round(pf, 3),
        "final_capital": round(capital, 2),
        "trades": results_list,
        "daily_returns": [round(r, 6) for r in daily_rets],
    }


def strategy_d_vix_collapse(data, windows=None):
    """Buy SPY when VIX rises >15% during earnings season then drops 5%+. Hold 10 days."""
    spy = data["SPY"]
    vix = data["VIX"]
    trading_dates = sorted([d.date() for d in spy.index])
    if windows is None:
        years = list(range(2022, 2027))
        windows = get_earnings_windows(years)

    trades = []
    for s, e in windows:
        # Get VIX data during this window
        window_dates = [d for d in trading_dates if s <= d <= e]
        if len(window_dates) < 5:
            continue

        # Find VIX rise >15% from window start
        start_ts = pd.Timestamp(window_dates[0])
        if start_ts not in vix.index:
            continue
        vix_start = float(vix.loc[start_ts, "Close"])

        # Track VIX peak during window
        vix_peak = vix_start
        vix_peak_date = None
        for wd in window_dates:
            ts = pd.Timestamp(wd)
            if ts not in vix.index:
                continue
            v = float(vix.loc[ts, "Close"])
            if v > vix_peak:
                vix_peak = v
                vix_peak_date = wd

        # Check if VIX rose >15%
        if vix_peak_date is None or (vix_peak - vix_start) / vix_start < 0.15:
            continue

        # Now look for 5% drop from peak
        entry_date = None
        for wd in window_dates:
            if wd <= vix_peak_date:
                continue
            ts = pd.Timestamp(wd)
            if ts not in vix.index:
                continue
            v = float(vix.loc[ts, "Close"])
            if (vix_peak - v) / vix_peak >= 0.05:
                entry_date = wd
                break

        if entry_date is None:
            # Also check days after window end
            post_dates = [d for d in trading_dates if e < d <= e + dt.timedelta(days=10)]
            for wd in post_dates:
                ts = pd.Timestamp(wd)
                if ts not in vix.index:
                    continue
                v = float(vix.loc[ts, "Close"])
                if (vix_peak - v) / vix_peak >= 0.05:
                    entry_date = wd
                    break

        if entry_date is None:
            continue

        exit_date = find_trading_day_offset(entry_date, trading_dates, 10)
        if exit_date is None:
            continue

        entry_ts = pd.Timestamp(entry_date)
        exit_ts = pd.Timestamp(exit_date)
        if entry_ts not in spy.index or exit_ts not in spy.index:
            continue

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": float(spy.loc[entry_ts, "Close"]),
            "exit_price": float(spy.loc[exit_ts, "Close"]),
            "ticker": "SPY",
        })
    return run_trades(trades)


def strategy_e_avoidance(data, windows=None):
    """Stay in cash during earnings season. Hold SPY between seasons."""
    spy = data["SPY"]
    trading_dates = sorted([d.date() for d in spy.index])
    if windows is None:
        years = list(range(2022, 2027))
        windows = get_earnings_windows(years)

    sorted_windows = sorted(windows, key=lambda w: w[0])
    trades = []

    # Add gap before first window
    first_td = trading_dates[0]
    if first_td < sorted_windows[0][0]:
        entry_date = find_next_trading_day(first_td, set(trading_dates))
        exit_date_target = sorted_windows[0][0] - dt.timedelta(days=1)
        exit_date = None
        # Find last trading day before exit_date_target
        for d in reversed(trading_dates):
            if d <= exit_date_target:
                exit_date = d
                break
        if entry_date and exit_date and entry_date < exit_date:
            entry_ts = pd.Timestamp(entry_date)
            exit_ts = pd.Timestamp(exit_date)
            if entry_ts in spy.index and exit_ts in spy.index:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(spy.loc[entry_ts, "Close"]),
                    "exit_price": float(spy.loc[exit_ts, "Close"]),
                    "ticker": "SPY",
                })

    # Gaps between windows
    for i in range(len(sorted_windows) - 1):
        gap_start = sorted_windows[i][1] + dt.timedelta(days=1)
        gap_end = sorted_windows[i + 1][0] - dt.timedelta(days=1)
        entry_date = find_next_trading_day(gap_start, set(trading_dates))
        exit_date = None
        for d in reversed(trading_dates):
            if d <= gap_end:
                exit_date = d
                break
        if entry_date and exit_date and entry_date < exit_date:
            entry_ts = pd.Timestamp(entry_date)
            exit_ts = pd.Timestamp(exit_date)
            if entry_ts in spy.index and exit_ts in spy.index:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(spy.loc[entry_ts, "Close"]),
                    "exit_price": float(spy.loc[exit_ts, "Close"]),
                    "ticker": "SPY",
                })

    # Gap after last window
    last_end = sorted_windows[-1][1] + dt.timedelta(days=1)
    last_td = trading_dates[-1]
    if last_end <= last_td:
        entry_date = find_next_trading_day(last_end, set(trading_dates))
        exit_date = last_td
        if entry_date and entry_date < exit_date:
            entry_ts = pd.Timestamp(entry_date)
            exit_ts = pd.Timestamp(exit_date)
            if entry_ts in spy.index and exit_ts in spy.index:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": float(spy.loc[entry_ts, "Close"]),
                    "exit_price": float(spy.loc[exit_ts, "Close"]),
                    "ticker": "SPY",
                })

    return run_trades(trades)


def strategy_f_adversarial(data, windows=None):
    """Buy SPY at random 3-week periods 4x/year, NOT aligned with earnings."""
    spy = data["SPY"]
    trading_dates = sorted([d.date() for d in spy.index])
    if windows is None:
        years = list(range(2022, 2027))
        windows = get_earnings_windows(years)

    # Generate non-earnings 3-week blocks: offset earnings windows by ~6 weeks
    np.random.seed(42)
    fake_windows = []
    for s, e in windows:
        offset = 45 + np.random.randint(0, 15)  # 45-60 day offset
        fs = s + dt.timedelta(days=offset)
        fe = fs + dt.timedelta(days=21)
        # Make sure it doesn't overlap with real earnings
        overlaps = any(ws <= fs <= we or ws <= fe <= we for ws, we in windows)
        if not overlaps:
            fake_windows.append((fs, fe))

    trades = []
    for s, e in fake_windows:
        entry_date = find_next_trading_day(s, set(trading_dates))
        exit_target = find_next_trading_day(e, set(trading_dates))
        if entry_date is None or exit_target is None:
            continue
        if entry_date >= exit_target:
            continue

        entry_ts = pd.Timestamp(entry_date)
        exit_ts = pd.Timestamp(exit_target)
        if entry_ts not in spy.index or exit_ts not in spy.index:
            continue

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_target,
            "entry_price": float(spy.loc[entry_ts, "Close"]),
            "exit_price": float(spy.loc[exit_ts, "Close"]),
            "ticker": "SPY",
        })
    return run_trades(trades)


# ── Permutation Test ───────────────────────────────────────────────────────
def permutation_test(strategy_func, data, observed_sharpe, n_perms=N_PERMUTATIONS):
    """Shift earnings dates randomly and compute p-value."""
    np.random.seed(123)
    years = list(range(2022, 2027))
    count_better = 0

    for _ in range(n_perms):
        offset = np.random.randint(-90, 91)  # Random shift -90 to +90 days
        shifted_windows = get_earnings_windows(years, offset_days=offset)
        try:
            result = strategy_func(data, windows=shifted_windows)
            if result["sharpe"] >= observed_sharpe:
                count_better += 1
        except Exception:
            continue

    p_value = (count_better + 1) / (n_perms + 1)
    return p_value


# ── Regime Analysis ────────────────────────────────────────────────────────
def regime_analysis(trades_list, spy_data):
    """Classify trades by market regime (green/red/flat days based on SPY)."""
    if not trades_list:
        return {"regime_gap": 0, "green_sharpe": 0, "red_sharpe": 0}

    green_rets = []
    red_rets = []

    for t in trades_list:
        entry = pd.Timestamp(t["entry_date"])
        exit_d = pd.Timestamp(t["exit_date"])
        if entry not in spy_data.index or exit_d not in spy_data.index:
            continue

        # Regime = SPY direction during the trade period
        spy_ret = (float(spy_data.loc[exit_d, "Close"]) - float(spy_data.loc[entry, "Close"])) / float(spy_data.loc[entry, "Close"])
        trade_ret = t.get("return_pct", 0)
        if isinstance(trade_ret, str):
            trade_ret = float(trade_ret)
        trade_ret = trade_ret / 100  # Convert from pct

        if spy_ret > 0.002:
            green_rets.append(trade_ret)
        elif spy_ret < -0.002:
            red_rets.append(trade_ret)
        # Flat: ignored for regime analysis

    green_sharpe = np.mean(green_rets) / np.std(green_rets) if len(green_rets) > 1 and np.std(green_rets) > 0 else 0
    red_sharpe = np.mean(red_rets) / np.std(red_rets) if len(red_rets) > 1 and np.std(red_rets) > 0 else 0

    max_abs = max(abs(green_sharpe), abs(red_sharpe), 1e-10)
    regime_gap = abs(green_sharpe - red_sharpe) / max_abs

    return {
        "regime_gap": round(regime_gap, 3),
        "green_sharpe": round(green_sharpe, 3),
        "red_sharpe": round(red_sharpe, 3),
        "green_trades": len(green_rets),
        "red_trades": len(red_rets),
    }


# ── Validation Gates ──────────────────────────────────────────────────────
def validate(result, p_value, regime_info):
    """Apply 5 validation gates."""
    gates = {
        "sharpe_gt_0.5": result["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": regime_info["regime_gap"] < 0.5,
        "mdd_gt_neg50": result["mdd_pct"] > -50,
        "trades_gte_20": result["n_trades"] >= 20,
    }
    return gates, all(gates.values())


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EARNINGS SEASON CALENDAR PATTERN BACKTEST")
    print("=" * 70)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Initial Capital: ${INITIAL_CAPITAL}")
    print(f"Slippage: {SLIPPAGE_PCT*100}% | Commission: $0")
    print()

    print("Downloading market data...")
    data = download_data()
    for k, v in data.items():
        print(f"  {k}: {len(v)} bars ({v.index[0].date()} to {v.index[-1].date()})")
    print()

    strategies = {
        "A_Pre_Earnings_Rally": strategy_a_pre_earnings_rally,
        "B_Post_Earnings_Drift": strategy_b_post_earnings_drift,
        "C_Sector_Rotation": strategy_c_sector_rotation,
        "D_VIX_Collapse": strategy_d_vix_collapse,
        "E_Earnings_Avoidance": strategy_e_avoidance,
        "F_Adversarial_Random": strategy_f_adversarial,
    }

    all_results = {}

    for name, func in strategies.items():
        print(f"{'─' * 60}")
        print(f"Strategy {name}")
        print(f"{'─' * 60}")

        result = func(data)
        print(f"  Trades: {result['n_trades']} | Win Rate: {result['win_rate']}%")
        print(f"  Sharpe: {result['sharpe']} | Sortino: {result['sortino']}")
        print(f"  Total Return: {result['total_return_pct']}% | MDD: {result['mdd_pct']}%")
        print(f"  Profit Factor: {result['profit_factor']} | Final Capital: ${result.get('final_capital', 'N/A')}")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...", end=" ", flush=True)
        p_value = permutation_test(func, data, result["sharpe"])
        print(f"p={p_value:.4f}")

        # Regime analysis
        regime_info = regime_analysis(result["trades"], data["SPY"])
        print(f"  Regime: green_sharpe={regime_info['green_sharpe']}, red_sharpe={regime_info['red_sharpe']}, gap={regime_info['regime_gap']}")

        # Validation
        gates, passed = validate(result, p_value, regime_info)
        gate_str = " | ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items())
        print(f"  Gates: {gate_str}")
        print(f"  >>> {'PASS' if passed else 'FAIL'} <<<")
        print()

        all_results[name] = {
            "metrics": {k: v for k, v in result.items() if k not in ("trades", "daily_returns")},
            "p_value": round(p_value, 4),
            "regime": regime_info,
            "gates": gates,
            "passed_all_gates": passed,
            "trades": result["trades"],
        }

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<30} {'Sharpe':>7} {'Return%':>8} {'MDD%':>7} {'WR%':>6} {'PF':>6} {'p-val':>7} {'Pass':>5}")
    print("-" * 82)
    for name, r in all_results.items():
        m = r["metrics"]
        print(f"{name:<30} {m['sharpe']:>7.3f} {m['total_return_pct']:>7.1f}% {m['mdd_pct']:>6.1f}% {m['win_rate']:>5.1f} {m['profit_factor']:>6.3f} {r['p_value']:>7.4f} {'YES' if r['passed_all_gates'] else 'NO':>5}")

    # Save results
    output = {
        "metadata": {
            "strategy": "Earnings Season Calendar Pattern",
            "period": f"{START_DATE} to {END_DATE}",
            "initial_capital": INITIAL_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": 0,
            "n_permutations": N_PERMUTATIONS,
            "run_date": str(dt.datetime.now()),
        },
        "variants": all_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
