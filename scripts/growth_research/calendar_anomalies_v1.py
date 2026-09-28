#!/usr/bin/env python3
"""
Calendar & Flow Anomalies Research v1
=====================================
Tests structurally-driven effects from institutional FLOWS (not price patterns).

Strategies:
  A. TOM Long — buy T-3, sell T+3 around month-end
  B. TOM Enhanced — TOM but only when VIX > 20 or VIX backwardation
  C. TOQ Long — TOM at quarter-ends only
  D. Pre-Holiday — buy before major US holidays, sell after
  E. OpEx Week Mean Reversion — contrarian entry into OpEx Friday
  F. Anti-TOM — flat/short during mid-month

Universe: SPY (primary), QQQ, IWM for robustness. 2004-2026.
Validation: permutation test, regime gap, decade consistency.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/calendar_anomalies_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(ticker: str, start: str = "2004-01-01", end: str = "2026-07-22") -> pd.DataFrame:
    """Download OHLCV from yfinance."""
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    print(f"  {ticker}: {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def download_vix(start: str = "2004-01-01", end: str = "2026-07-22") -> pd.DataFrame:
    """Download VIX index."""
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[["Close"]].dropna()
    vix.columns = ["VIX"]
    vix.index = pd.to_datetime(vix.index)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    return vix


def download_vix_futures_proxy(start: str = "2004-01-01", end: str = "2026-07-22") -> pd.DataFrame:
    """Download VIX 3-month futures ETF as backwardation proxy (VXZ or VIXY)."""
    # Use VIX9D vs VIX as rough backwardation proxy
    vix9d = yf.download("^VIX9D", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(vix9d.columns, pd.MultiIndex):
        vix9d.columns = vix9d.columns.get_level_values(0)
    if len(vix9d) > 0:
        vix9d = vix9d[["Close"]].dropna()
        vix9d.columns = ["VIX9D"]
        vix9d.index = pd.to_datetime(vix9d.index)
        if vix9d.index.tz is not None:
            vix9d.index = vix9d.index.tz_localize(None)
        return vix9d
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def get_month_end_trading_days(dates: pd.DatetimeIndex) -> dict:
    """For each month, find the actual trading day indices near month boundaries."""
    dates_series = pd.Series(range(len(dates)), index=dates)
    result = {}
    
    # Group by year-month
    for ym, group in dates_series.groupby(dates_series.index.to_period("M")):
        month_indices = group.values
        result[ym] = {
            "all_indices": month_indices,
            "last_3": month_indices[-3:] if len(month_indices) >= 3 else month_indices,
            "first_3": None  # filled in next pass
        }
    
    # Fill first_3 for each month
    periods = sorted(result.keys())
    for i, p in enumerate(periods):
        result[p]["first_3"] = result[p]["all_indices"][:3] if len(result[p]["all_indices"]) >= 3 else result[p]["all_indices"]
    
    return result


def get_tom_signals(dates: pd.DatetimeIndex) -> pd.Series:
    """Return +1 for TOM days (last 3 + first 3 of each month), 0 otherwise."""
    signal = pd.Series(0, index=dates)
    month_data = get_month_end_trading_days(dates)
    
    for ym, info in month_data.items():
        # Last 3 trading days of month
        for idx in info["last_3"]:
            signal.iloc[idx] = 1
        # First 3 trading days of month
        for idx in info["first_3"]:
            signal.iloc[idx] = 1
    
    return signal


def get_toq_signals(dates: pd.DatetimeIndex) -> pd.Series:
    """Return +1 for TOQ days (last 3 + first 3 of quarter-end months only)."""
    signal = pd.Series(0, index=dates)
    month_data = get_month_end_trading_days(dates)
    quarter_end_months = {3, 6, 9, 12}
    
    periods = sorted(month_data.keys())
    for i, p in enumerate(periods):
        if p.month in quarter_end_months:
            # Last 3 of quarter-end month
            for idx in month_data[p]["last_3"]:
                signal.iloc[idx] = 1
            # First 3 of the NEXT month (start of new quarter)
            if i + 1 < len(periods):
                next_p = periods[i + 1]
                for idx in month_data[next_p]["first_3"]:
                    signal.iloc[idx] = 1
    
    return signal


def get_us_holidays(years: list) -> list:
    """Generate major US market holidays. Returns list of dates."""
    holidays = []
    for year in years:
        # Fixed holidays (approximate — actual market closures vary)
        candidates = [
            # New Year's Day
            datetime(year, 1, 1),
            # MLK — 3rd Monday of January
            _nth_weekday(year, 1, 0, 3),  # Monday=0
            # Presidents Day — 3rd Monday of February
            _nth_weekday(year, 2, 0, 3),
            # Good Friday — approximate (2 days before Easter Sunday)
            _easter(year) - timedelta(days=2),
            # Memorial Day — last Monday of May
            _last_weekday(year, 5, 0),
            # Juneteenth
            datetime(year, 6, 19),
            # July 4th
            datetime(year, 7, 4),
            # Labor Day — 1st Monday of September
            _nth_weekday(year, 9, 0, 1),
            # Thanksgiving — 4th Thursday of November
            _nth_weekday(year, 11, 3, 4),  # Thursday=3
            # Christmas
            datetime(year, 12, 25),
        ]
        holidays.extend(candidates)
    return [h.date() if isinstance(h, datetime) else h for h in holidays]


def _nth_weekday(year, month, weekday, n):
    """Get the nth occurrence of a weekday in a month."""
    first = datetime(year, month, 1)
    # Days until first occurrence
    days_ahead = weekday - first.weekday()
    if days_ahead < 0:
        days_ahead += 7
    first_occ = first + timedelta(days=days_ahead)
    return first_occ + timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    """Get the last occurrence of a weekday in a month."""
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    days_back = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_back)


def _easter(year):
    """Compute Easter Sunday (Anonymous Gregorian algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime(year, month, day)


def get_pre_holiday_signals(dates: pd.DatetimeIndex) -> pd.Series:
    """Return +1 on the trading day BEFORE a market holiday."""
    signal = pd.Series(0, index=dates)
    years = sorted(dates.year.unique())
    holidays = get_us_holidays(list(years))
    holiday_set = set(holidays)
    
    dates_list = dates.tolist()
    for i in range(len(dates_list) - 1):
        # Check if the next calendar day or next few days include a holiday
        current = dates_list[i].date() if hasattr(dates_list[i], 'date') else dates_list[i]
        next_trading = dates_list[i + 1].date() if hasattr(dates_list[i + 1], 'date') else dates_list[i + 1]
        
        # Check all calendar days between current and next trading day
        check = current + timedelta(days=1)
        while check < next_trading:
            if check in holiday_set:
                signal.iloc[i] = 1
                break
            check += timedelta(days=1)
        
        # Also check if next trading day is right after a holiday
        # (handles case where holiday is on weekend — market closed Mon)
        if (next_trading - current).days > 3:  # gap > weekend
            signal.iloc[i] = 1
    
    return signal


def get_opex_signals(dates: pd.DatetimeIndex, returns: pd.Series) -> pd.Series:
    """
    OpEx week mean reversion: 
    OpEx = 3rd Friday of each month.
    Signal: if 5-day return going into OpEx week is positive -> short (sell), negative -> long (buy).
    """
    signal = pd.Series(0, index=dates)
    ret_5d = returns.rolling(5).sum()
    
    for year in dates.year.unique():
        for month in range(1, 13):
            # Find 3rd Friday
            opex = _nth_weekday(int(year), month, 4, 3)  # Friday=4
            opex_date = opex.date()
            
            # Find the Monday of OpEx week (5 trading days before Friday roughly)
            opex_week_start = opex - timedelta(days=4)
            
            # Find trading days in OpEx week
            mask = (dates.date >= opex_week_start.date()) & (dates.date <= opex_date)
            opex_days = dates[mask]
            
            if len(opex_days) > 0:
                # Get 5-day return as of the day before OpEx week
                first_opex_day = opex_days[0]
                idx = dates.get_loc(first_opex_day)
                if idx > 0 and idx - 1 < len(ret_5d):
                    prev_ret = ret_5d.iloc[idx - 1]
                    if not np.isnan(prev_ret):
                        # Contrarian: positive prior return -> -1 (sell), negative -> +1 (buy)
                        trade_signal = -1 if prev_ret > 0 else 1
                        for d in opex_days:
                            signal.loc[d] = trade_signal
    
    return signal


def get_anti_tom_signals(dates: pd.DatetimeIndex) -> pd.Series:
    """Return -1 for mid-month days (not in TOM window), 0 otherwise."""
    tom = get_tom_signals(dates)
    anti = pd.Series(0, index=dates)
    anti[tom == 0] = -1  # short during mid-month
    return anti


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(returns: pd.Series, signal: pd.Series, name: str,
                      regime_returns: pd.Series = None) -> dict:
    """
    Compute performance metrics for a calendar strategy.
    
    returns: daily log returns of the asset
    signal: +1 (long), -1 (short), 0 (flat)
    """
    strat_returns = returns * signal
    strat_returns = strat_returns[signal != 0]  # only count active days
    
    if len(strat_returns) < 10:
        return {"name": name, "n_trades": 0, "error": "Too few trading days"}
    
    # Identify distinct trade periods (contiguous blocks of non-zero signal)
    active = (signal != 0).astype(int)
    trade_blocks = (active.diff().fillna(active) != 0).cumsum()
    trade_blocks = trade_blocks[signal != 0]
    n_trades = trade_blocks.nunique()
    
    # Per-trade returns
    trade_returns = strat_returns.groupby(trade_blocks).sum()
    
    # Metrics
    total_return = strat_returns.sum()
    mean_trade = trade_returns.mean()
    std_trade = trade_returns.std()
    
    # Annualize: assume ~252 trading days
    active_days = len(strat_returns)
    ann_factor = 252 / max(active_days, 1) * (active_days / max(1, (strat_returns.index[-1] - strat_returns.index[0]).days / 365.25))
    
    ann_return = strat_returns.mean() * 252
    ann_std = strat_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_std if ann_std > 0 else 0
    
    # Sortino
    downside = strat_returns[strat_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = ann_return / downside_std if downside_std > 0 else 0
    
    # Win rate & profit factor
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    wr = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Max drawdown
    cum = strat_returns.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()
    
    result = {
        "name": name,
        "n_trades": int(n_trades),
        "n_active_days": int(active_days),
        "total_return_pct": round(float(total_return * 100), 2),
        "mean_trade_return_pct": round(float(mean_trade * 100), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "win_rate": round(float(wr), 3),
        "profit_factor": round(float(min(pf, 99.9)), 3),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "ann_return_pct": round(float(ann_return * 100), 2),
        "ann_vol_pct": round(float(ann_std * 100), 2),
    }
    
    # ----- Validation Gate 1: Permutation test -----
    observed_sharpe = sharpe
    n_perms = 200
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = signal.copy()
        shuffled_vals = shuffled.values.copy()
        np.random.shuffle(shuffled_vals)
        shuffled = pd.Series(shuffled_vals, index=signal.index)
        perm_ret = returns * shuffled
        perm_ret = perm_ret[shuffled != 0]
        if len(perm_ret) > 1:
            p_ann_ret = perm_ret.mean() * 252
            p_ann_std = perm_ret.std() * np.sqrt(252)
            p_sharpe = p_ann_ret / p_ann_std if p_ann_std > 0 else 0
            perm_sharpes.append(p_sharpe)
    
    if perm_sharpes:
        perm_p = np.mean([1 if ps >= observed_sharpe else 0 for ps in perm_sharpes])
        result["permutation_p"] = round(float(perm_p), 4)
        result["permutation_pass"] = bool(perm_p < 0.05)
    
    # ----- Validation Gate 2: Regime gap -----
    if regime_returns is not None:
        # Classify months as green/red based on monthly return
        monthly_ret = regime_returns.resample("M").sum()
        green_months = set(monthly_ret[monthly_ret > 0].index.to_period("M"))
        red_months = set(monthly_ret[monthly_ret <= 0].index.to_period("M"))
        
        strat_monthly = strat_returns.copy()
        strat_monthly_period = strat_monthly.index.to_period("M")
        
        green_mask = strat_monthly_period.isin(green_months)
        red_mask = strat_monthly_period.isin(red_months)
        
        green_rets = strat_returns[green_mask]
        red_rets = strat_returns[red_mask]
        
        if len(green_rets) > 5 and len(red_rets) > 5:
            sharpe_green = (green_rets.mean() * 252) / (green_rets.std() * np.sqrt(252)) if green_rets.std() > 0 else 0
            sharpe_red = (red_rets.mean() * 252) / (red_rets.std() * np.sqrt(252)) if red_rets.std() > 0 else 0
            
            max_abs = max(abs(sharpe_green), abs(sharpe_red))
            regime_gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0
            
            result["sharpe_green"] = round(float(sharpe_green), 3)
            result["sharpe_red"] = round(float(sharpe_red), 3)
            result["regime_gap"] = round(float(regime_gap), 3)
            result["regime_pass"] = bool(regime_gap < 0.50)
    
    # ----- Validation Gate 3: Decade consistency -----
    decades = {
        "2004-2010": (pd.Timestamp("2004-01-01"), pd.Timestamp("2010-12-31")),
        "2011-2016": (pd.Timestamp("2011-01-01"), pd.Timestamp("2016-12-31")),
        "2017-2026": (pd.Timestamp("2017-01-01"), pd.Timestamp("2026-12-31")),
    }
    decade_results = {}
    all_positive = True
    for label, (s, e) in decades.items():
        mask = (strat_returns.index >= s) & (strat_returns.index <= e)
        d_rets = strat_returns[mask]
        if len(d_rets) > 5:
            d_sharpe = (d_rets.mean() * 252) / (d_rets.std() * np.sqrt(252)) if d_rets.std() > 0 else 0
            decade_results[label] = round(float(d_sharpe), 3)
            if d_sharpe <= 0:
                all_positive = False
        else:
            decade_results[label] = None
            all_positive = False
    
    result["decade_sharpes"] = decade_results
    result["decade_pass"] = all_positive
    
    # Overall pass
    result["all_gates_pass"] = (
        result.get("permutation_pass", False) and
        result.get("regime_pass", False) and
        result.get("decade_pass", False)
    )
    
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    print("=" * 70)
    print("CALENDAR & FLOW ANOMALIES RESEARCH v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Download data
    tickers = {"SPY": None, "QQQ": None, "IWM": None}
    for t in tickers:
        tickers[t] = download_data(t)
    
    vix = download_vix()
    vix9d = download_vix_futures_proxy()
    
    all_results = {}
    
    for ticker_name, df in tickers.items():
        print(f"\n{'='*60}")
        print(f"TESTING: {ticker_name}")
        print(f"{'='*60}")
        
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        dates = returns.index
        
        # Merge VIX
        vix_aligned = vix.reindex(dates).ffill()
        
        # VIX backwardation proxy
        if len(vix9d) > 0:
            vix9d_aligned = vix9d.reindex(dates).ffill()
            # Backwardation = short-term VIX > VIX (inverted term structure)
            backwardation = (vix9d_aligned["VIX9D"] > vix_aligned["VIX"]).fillna(False)
        else:
            backwardation = pd.Series(False, index=dates)
        
        strategies = {}
        
        # A. TOM Long
        tom_signal = get_tom_signals(dates)
        strategies["A_TOM_Long"] = tom_signal
        
        # B. TOM Enhanced (VIX > 20 or backwardation)
        vix_stress = (vix_aligned["VIX"] > 20).fillna(False) | backwardation
        tom_enhanced = tom_signal.copy()
        tom_enhanced[~vix_stress] = 0
        strategies["B_TOM_Enhanced"] = tom_enhanced
        
        # C. TOQ Long
        toq_signal = get_toq_signals(dates)
        strategies["C_TOQ_Long"] = toq_signal
        
        # D. Pre-Holiday
        preholiday_signal = get_pre_holiday_signals(dates)
        strategies["D_Pre_Holiday"] = preholiday_signal
        
        # E. OpEx Week Mean Reversion
        opex_signal = get_opex_signals(dates, returns)
        strategies["E_OpEx_MeanRev"] = opex_signal
        
        # F. Anti-TOM (short mid-month)
        anti_tom = get_anti_tom_signals(dates)
        strategies["F_Anti_TOM"] = anti_tom
        
        ticker_results = []
        for strat_name, signal in strategies.items():
            print(f"\n  Evaluating {strat_name}...")
            active_days = (signal != 0).sum()
            print(f"    Active days: {active_days} / {len(signal)}")
            
            result = evaluate_strategy(returns, signal, f"{ticker_name}_{strat_name}",
                                       regime_returns=returns)
            ticker_results.append(result)
            
            # Print summary
            print(f"    Sharpe: {result.get('sharpe', 'N/A')}")
            print(f"    Sortino: {result.get('sortino', 'N/A')}")
            print(f"    WR: {result.get('win_rate', 'N/A')}")
            print(f"    PF: {result.get('profit_factor', 'N/A')}")
            print(f"    Perm p: {result.get('permutation_p', 'N/A')} (pass={result.get('permutation_pass', 'N/A')})")
            print(f"    Regime gap: {result.get('regime_gap', 'N/A')} (pass={result.get('regime_pass', 'N/A')})")
            print(f"    Decade: {result.get('decade_sharpes', 'N/A')} (pass={result.get('decade_pass', 'N/A')})")
            print(f"    ALL GATES: {'PASS' if result.get('all_gates_pass') else 'FAIL'}")
        
        all_results[ticker_name] = ticker_results
    
    # ----- Summary table -----
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'Perm':>6} {'Regime':>7} {'Decade':>7} {'PASS':>5}")
    print("-" * 80)
    
    for ticker_name, results in all_results.items():
        for r in results:
            name = r["name"]
            sharpe = r.get("sharpe", 0)
            sortino = r.get("sortino", 0)
            wr = r.get("win_rate", 0)
            pf = r.get("profit_factor", 0)
            perm = "Y" if r.get("permutation_pass") else "N"
            regime = "Y" if r.get("regime_pass") else "N"
            decade = "Y" if r.get("decade_pass") else "N"
            allp = "YES" if r.get("all_gates_pass") else "NO"
            print(f"{name:<30} {sharpe:>7.3f} {sortino:>8.3f} {wr:>6.3f} {pf:>6.2f} {perm:>6} {regime:>7} {decade:>7} {allp:>5}")
    
    # ----- Cross-ticker robustness -----
    print("\n" + "=" * 60)
    print("CROSS-TICKER ROBUSTNESS CHECK")
    print("=" * 60)
    
    strat_names = [r["name"].split("_", 1)[1] for r in all_results.get("SPY", [])]
    for sname in strat_names:
        passes = []
        sharpes = []
        for ticker_name in ["SPY", "QQQ", "IWM"]:
            for r in all_results.get(ticker_name, []):
                if r["name"].endswith(sname):
                    passes.append(r.get("all_gates_pass", False))
                    sharpes.append(r.get("sharpe", 0))
        
        n_pass = sum(passes)
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        robust = "ROBUST" if n_pass >= 2 else "WEAK"
        print(f"  {sname:<25} passes: {n_pass}/3  avg_sharpe: {avg_sharpe:.3f}  -> {robust}")
    
    # ----- Save results -----
    output = {
        "timestamp": datetime.now().isoformat(),
        "description": "Calendar & Flow Anomalies Research v1",
        "tickers_tested": list(tickers.keys()),
        "strategies": {
            "A_TOM_Long": "Buy T-3 before month-end, sell T+3 into new month",
            "B_TOM_Enhanced": "TOM but only when VIX>20 or backwardation",
            "C_TOQ_Long": "TOM at quarter-ends only (Mar/Jun/Sep/Dec)",
            "D_Pre_Holiday": "Long the day before major US holidays",
            "E_OpEx_MeanRev": "Contrarian entry during OpEx week based on 5-day prior return",
            "F_Anti_TOM": "Short during mid-month (avoid TOM window)",
        },
        "validation_gates": {
            "permutation": "200 shuffles, p < 0.05",
            "regime_gap": "|Sharpe_green - Sharpe_red| / max < 0.50",
            "decade_consistency": "Positive Sharpe in all 3 sub-periods",
        },
        "results": all_results,
    }
    
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    np.random.seed(42)
    run_all()
