#!/usr/bin/env python3
"""
Calendar Effects Sector ETF Dip-Buying Backtest
Tests whether calendar/seasonal patterns improve sector ETF dip-buying.
8 variants + baseline (RSI<35 with no calendar filter).
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE', 'XLV']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]
START = '2020-01-01'
END = datetime.now().strftime('%Y-%m-%d')
COST_RT = 0.0010  # 0.10% round-trip
RSI_THRESHOLD = 35
RSI_PERIOD = 14
HOLD_DAYS = 5
TP_PCT = 0.03
SL_PCT = -0.05
N_PERMUTATIONS = 1000
DAY_CONC_CAP = 0.70
REGIME_GAP_CAP = 0.50

RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/scripts/growth_research/results/calendar_effects_results.json')


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    data = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')
    # Forward-fill small gaps
    close = close.ffill(limit=3)
    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")
    return close


# ── Indicators ──────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_all_rsi(close):
    rsi_df = pd.DataFrame(index=close.index)
    for col in SECTOR_ETFS:
        if col in close.columns:
            rsi_df[col] = compute_rsi(close[col], RSI_PERIOD)
    return rsi_df


# ── Calendar Filters ────────────────────────────────────────────────────────
def get_trading_day_of_month(dates):
    """Return trading day of month (1-indexed) for each date."""
    df = pd.DataFrame({'date': dates, 'month': dates.month, 'year': dates.year})
    df['td_of_month'] = df.groupby(['year', 'month']).cumcount() + 1
    return df['td_of_month'].values


def get_trading_days_left_in_month(dates):
    """Return trading days remaining in the month for each date."""
    df = pd.DataFrame({'date': dates, 'month': dates.month, 'year': dates.year})
    df['total_td'] = df.groupby(['year', 'month'])['date'].transform('count')
    df['td_of_month'] = df.groupby(['year', 'month']).cumcount() + 1
    df['td_left'] = df['total_td'] - df['td_of_month']
    return df['td_left'].values


def filter_turn_of_month(dates):
    """Last 2 TDs of month OR first 3 TDs of next month."""
    td_of_month = get_trading_day_of_month(dates)
    td_left = get_trading_days_left_in_month(dates)
    return (td_left <= 1) | (td_of_month <= 3)  # last 2 or first 3


def filter_mid_month_avoid(dates):
    """AVOID days 10-20 of the calendar month. Only buy outside that window."""
    days = dates.day
    return (days < 10) | (days > 20)


def filter_opex_week(dates):
    """Week containing 3rd Friday of the month."""
    mask = np.zeros(len(dates), dtype=bool)
    for i, d in enumerate(dates):
        # Find 3rd Friday of this month
        first_day = d.replace(day=1)
        # Days until Friday (4)
        days_to_fri = (4 - first_day.weekday()) % 7
        first_fri = first_day + timedelta(days=days_to_fri)
        third_fri = first_fri + timedelta(days=14)
        # Week containing 3rd Friday: Mon-Fri of that week
        week_start = third_fri - timedelta(days=third_fri.weekday())
        week_end = week_start + timedelta(days=4)
        mask[i] = week_start <= d <= week_end
    return mask


def filter_post_opex(dates):
    """3 trading days AFTER monthly opex (3rd Friday)."""
    # Find all 3rd Fridays
    opex_dates = set()
    seen_months = set()
    for d in dates:
        key = (d.year, d.month)
        if key not in seen_months:
            seen_months.add(key)
            first_day = d.replace(day=1)
            days_to_fri = (4 - first_day.weekday()) % 7
            first_fri = first_day + timedelta(days=days_to_fri)
            third_fri = first_fri + timedelta(days=14)
            opex_dates.add(third_fri)

    date_list = list(dates)
    mask = np.zeros(len(dates), dtype=bool)
    for opex in opex_dates:
        # Find the index of opex or first day after
        for offset in range(1, 8):
            check = opex + timedelta(days=offset)
            if check in date_list:
                idx = date_list.index(check)
                # Mark 3 TDs after opex
                count = 0
                for j in range(idx, min(idx + 5, len(dates))):
                    if count < 3:
                        mask[j] = True
                        count += 1
                break
    return mask


def filter_quarter_end(dates):
    """Last 5 TDs of each quarter (Mar, Jun, Sep, Dec)."""
    quarter_end_months = {3, 6, 9, 12}
    td_left = get_trading_days_left_in_month(dates)
    months = dates.month
    return np.array([(m in quarter_end_months) and (tl <= 4) for m, tl in zip(months, td_left)])


def filter_january_effect(dates):
    """First 10 TDs of January."""
    td_of_month = get_trading_day_of_month(dates)
    return (dates.month == 1) & (td_of_month <= 10)


def filter_holiday_drift(dates):
    """2 TDs before major holidays. Approximate by calendar proximity."""
    # Major US holidays (approximate dates each year)
    holidays_by_year = {}
    for d in dates:
        y = d.year
        if y not in holidays_by_year:
            holidays_by_year[y] = []
            # Memorial Day: last Monday of May
            may31 = pd.Timestamp(y, 5, 31)
            mem_day = may31 - timedelta(days=(may31.weekday()))  # Monday
            if mem_day.month != 5:
                mem_day = may31 - timedelta(days=may31.weekday() + 7)
            # July 4th
            jul4 = pd.Timestamp(y, 7, 4)
            # Labor Day: first Monday of Sep
            sep1 = pd.Timestamp(y, 9, 1)
            labor_day = sep1 + timedelta(days=(7 - sep1.weekday()) % 7)
            if labor_day.day > 7:
                labor_day = sep1  # Sep 1 is Monday
            # Thanksgiving: 4th Thursday of Nov
            nov1 = pd.Timestamp(y, 11, 1)
            days_to_thu = (3 - nov1.weekday()) % 7
            first_thu = nov1 + timedelta(days=days_to_thu)
            thanksgiving = first_thu + timedelta(days=21)
            # Christmas
            christmas = pd.Timestamp(y, 12, 25)

            holidays_by_year[y] = [mem_day, jul4, labor_day, thanksgiving, christmas]

    date_list = list(dates)
    mask = np.zeros(len(dates), dtype=bool)
    for y, hols in holidays_by_year.items():
        for hol in hols:
            # Find 2 TDs before the holiday
            for i in range(len(date_list) - 1, -1, -1):
                if date_list[i] < hol:
                    mask[i] = True
                    if i > 0:
                        mask[i - 1] = True
                    break
    return mask


def filter_earnings_season(dates):
    """Weeks 3-5 after quarter end (approx Jan 15-Feb 15, Apr 15-May 15, Jul 15-Aug 15, Oct 15-Nov 15)."""
    month = dates.month
    day = dates.day
    mask = (
        ((month == 1) & (day >= 15)) | ((month == 2) & (day <= 15)) |
        ((month == 4) & (day >= 15)) | ((month == 5) & (day <= 15)) |
        ((month == 7) & (day >= 15)) | ((month == 8) & (day <= 15)) |
        ((month == 10) & (day >= 15)) | ((month == 11) & (day <= 15))
    )
    return mask


# ── Backtest Engine ─────────────────────────────────────────────────────────
def run_backtest(close, rsi_df, spy_close, calendar_mask, label):
    """Run dip-buying backtest with calendar filter.

    Entry: RSI < 35 AND calendar_mask is True. Buy lowest-RSI qualifying sector.
    Exit: 5-day hold, +3% TP, -5% SL (whichever first).
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    entry_ticker = None
    entry_idx = None

    dates = close.index

    for i in range(RSI_PERIOD + 1, len(dates)):
        dt = dates[i]

        # Check if we need to exit
        if in_trade:
            days_held = i - entry_idx
            current_price = close.loc[dt, entry_ticker] if entry_ticker in close.columns else np.nan
            if pd.isna(current_price):
                continue
            ret = (current_price / entry_price) - 1

            exit_reason = None
            if ret >= TP_PCT:
                exit_reason = 'TP'
            elif ret <= SL_PCT:
                exit_reason = 'SL'
            elif days_held >= HOLD_DAYS:
                exit_reason = 'TIME'

            if exit_reason:
                net_ret = ret - COST_RT
                # Regime: SPY close vs open approximated by close-to-close
                spy_ret = (spy_close.iloc[i] / spy_close.iloc[entry_idx]) - 1 if entry_idx < len(spy_close) else 0
                regime = 'green' if spy_ret >= 0 else 'red'
                trades.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': dt.strftime('%Y-%m-%d'),
                    'ticker': entry_ticker,
                    'entry_price': float(entry_price),
                    'exit_price': float(current_price),
                    'gross_ret': float(ret),
                    'net_ret': float(net_ret),
                    'exit_reason': exit_reason,
                    'days_held': days_held,
                    'regime': regime,
                    'day_of_week': entry_date.weekday(),
                })
                in_trade = False
                continue

        # Check for entry
        if not in_trade and calendar_mask[i]:
            # Find sectors with RSI < threshold
            candidates = {}
            for etf in SECTOR_ETFS:
                if etf in rsi_df.columns and not pd.isna(rsi_df.loc[dt, etf]):
                    if rsi_df.loc[dt, etf] < RSI_THRESHOLD:
                        candidates[etf] = rsi_df.loc[dt, etf]

            if candidates:
                # Buy lowest RSI
                best = min(candidates, key=candidates.get)
                entry_price = close.loc[dt, best]
                if not pd.isna(entry_price):
                    entry_date = dt
                    entry_ticker = best
                    entry_idx = i
                    in_trade = True

    return trades


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(trades, label):
    if not trades:
        return {
            'label': label, 'trades': 0, 'wr': 0, 'avg_ret': 0,
            'sharpe': 0, 'sortino': 0, 'pf': 0, 'max_dd': 0,
            'regime_sharpe_green': 0, 'regime_sharpe_red': 0,
            'regime_gap': 999, 'regime_pass': False,
            'perm_p': 1.0, 'perm_pass': False,
            'day_conc': 1.0, 'day_conc_pass': False,
            'overall_pass': False,
        }

    rets = np.array([t['net_ret'] for t in trades])
    n = len(rets)
    wins = np.sum(rets > 0)
    wr = wins / n
    avg_ret = np.mean(rets)

    # Sharpe (annualized, ~252/hold_days trades per year potential)
    trades_per_year = 252 / HOLD_DAYS
    if np.std(rets) > 0:
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe if avg_ret > 0 else 0

    # Profit Factor
    gross_profit = np.sum(rets[rets > 0]) if np.any(rets > 0) else 0
    gross_loss = abs(np.sum(rets[rets < 0])) if np.any(rets < 0) else 0.0001
    pf = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max Drawdown (cumulative)
    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    # Regime stratified Sharpe
    green_rets = np.array([t['net_ret'] for t in trades if t['regime'] == 'green'])
    red_rets = np.array([t['net_ret'] for t in trades if t['regime'] == 'red'])

    def _sharpe(r):
        if len(r) < 3 or np.std(r) == 0:
            return 0
        return (np.mean(r) / np.std(r)) * np.sqrt(trades_per_year)

    sharpe_green = _sharpe(green_rets)
    sharpe_red = _sharpe(red_rets)
    max_abs = max(abs(sharpe_green), abs(sharpe_red), 0.001)
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs
    regime_pass = regime_gap < REGIME_GAP_CAP

    # Permutation test
    observed_mean = np.mean(rets)
    count_better = 0
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(rets)
        # Shuffle the sign of returns (null: no edge)
        rand_signs = np.random.choice([-1, 1], size=n)
        shuffled_rets = np.abs(rets) * rand_signs
        if np.mean(shuffled_rets) >= observed_mean:
            count_better += 1
    perm_p = count_better / N_PERMUTATIONS
    perm_pass = perm_p < 0.05

    # Day concentration
    day_counts = np.zeros(5)
    for t in trades:
        day_counts[t['day_of_week']] += 1
    day_conc = float(np.max(day_counts) / n) if n > 0 else 1
    day_conc_pass = day_conc < DAY_CONC_CAP

    overall_pass = regime_pass and perm_pass and day_conc_pass and sharpe > 0.3

    return {
        'label': label,
        'trades': n,
        'wr': round(wr * 100, 1),
        'avg_ret': round(avg_ret * 100, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'max_dd': round(max_dd * 100, 2),
        'regime_sharpe_green': round(sharpe_green, 3),
        'regime_sharpe_red': round(sharpe_red, 3),
        'regime_gap': round(regime_gap, 3),
        'regime_pass': regime_pass,
        'perm_p': round(perm_p, 4),
        'perm_pass': perm_pass,
        'day_conc': round(day_conc, 3),
        'day_conc_pass': day_conc_pass,
        'overall_pass': overall_pass,
        'green_trades': len(green_rets),
        'red_trades': len(red_rets),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    close = download_data()
    rsi_df = compute_all_rsi(close)
    spy_close = close[BENCHMARK]
    dates = close.index

    # Pre-compute calendar masks
    print("\nComputing calendar masks...")
    masks = {
        'Baseline (No Filter)': np.ones(len(dates), dtype=bool),
        'A: Turn-of-Month': filter_turn_of_month(dates),
        'B: Mid-Month Avoid': filter_mid_month_avoid(dates),
        'C: OpEx Week': filter_opex_week(dates),
        'D: Post-OpEx Bounce': filter_post_opex(dates),
        'E: Quarter-End': filter_quarter_end(dates),
        'F: January Effect': filter_january_effect(dates),
        'G: Holiday Drift': filter_holiday_drift(dates),
        'H: Earnings Season': filter_earnings_season(dates),
    }

    for name, mask in masks.items():
        pct = mask.sum() / len(mask) * 100
        print(f"  {name}: {mask.sum()} eligible days ({pct:.1f}%)")

    # Run backtests
    print("\nRunning backtests...")
    all_results = []
    baseline_sharpe = None

    for name, mask in masks.items():
        print(f"  {name}...")
        trades = run_backtest(close, rsi_df, spy_close, mask, name)
        metrics = compute_metrics(trades, name)

        if name == 'Baseline (No Filter)':
            baseline_sharpe = metrics['sharpe']
            metrics['sharpe_improvement'] = 0
        else:
            metrics['sharpe_improvement'] = round(metrics['sharpe'] - baseline_sharpe, 3) if baseline_sharpe else 0

        all_results.append(metrics)

    # Print results table
    print("\n" + "=" * 130)
    print("CALENDAR EFFECTS SECTOR ETF DIP-BUYING BACKTEST RESULTS")
    print(f"Period: {close.index[0].date()} to {close.index[-1].date()} | Cost: {COST_RT*100:.2f}% RT | RSI < {RSI_THRESHOLD}")
    print("=" * 130)

    header = f"{'Variant':<28} {'Trades':>6} {'WR%':>6} {'AvgRet%':>8} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD%':>7} {'ShpGrn':>7} {'ShpRed':>7} {'RGap':>6} {'PermP':>6} {'DayC':>6} {'Δ Shp':>6} {'PASS':>5}"
    print(header)
    print("-" * 130)

    for r in all_results:
        pass_str = "YES" if r['overall_pass'] else "NO"
        delta = r.get('sharpe_improvement', 0)
        delta_str = f"{delta:+.3f}" if r['label'] != 'Baseline (No Filter)' else "  ---"
        regime_flag = "✓" if r['regime_pass'] else "✗"
        perm_flag = "✓" if r['perm_pass'] else "✗"
        day_flag = "✓" if r['day_conc_pass'] else "✗"

        line = (
            f"{r['label']:<28} "
            f"{r['trades']:>6} "
            f"{r['wr']:>5.1f}% "
            f"{r['avg_ret']:>7.3f}% "
            f"{r['sharpe']:>7.3f} "
            f"{r['sortino']:>8.3f} "
            f"{r['pf']:>6.3f} "
            f"{r['max_dd']:>6.2f}% "
            f"{r['regime_sharpe_green']:>7.3f} "
            f"{r['regime_sharpe_red']:>7.3f} "
            f"{r['regime_gap']:>5.3f}{regime_flag} "
            f"{r['perm_p']:>5.4f}{perm_flag} "
            f"{r['day_conc']:>5.3f}{day_flag} "
            f"{delta_str:>6} "
            f"{pass_str:>5}"
        )
        print(line)
        if r['label'] == 'Baseline (No Filter)':
            print("-" * 130)

    print("=" * 130)
    print(f"\nGates: Regime Gap < {REGIME_GAP_CAP} | Perm p < 0.05 | Day Conc < {DAY_CONC_CAP} | Sharpe > 0.3")
    print(f"Permutation test: {N_PERMUTATIONS} sign-shuffles")

    # Summary
    passing = [r for r in all_results if r['overall_pass'] and r['label'] != 'Baseline (No Filter)']
    improving = [r for r in all_results if r.get('sharpe_improvement', 0) > 0 and r['label'] != 'Baseline (No Filter)']

    print(f"\n--- SUMMARY ---")
    print(f"Baseline Sharpe: {baseline_sharpe:.3f}")
    print(f"Variants improving over baseline: {len(improving)}/8")
    print(f"Variants passing ALL gates: {len(passing)}/8")

    if passing:
        best = max(passing, key=lambda x: x['sharpe'])
        print(f"Best passing variant: {best['label']} (Sharpe={best['sharpe']:.3f}, +{best['sharpe_improvement']:.3f} vs baseline)")
    else:
        print("No variant passes all gates.")

    # Save JSON
    output = {
        'metadata': {
            'backtest': 'Calendar Effects Sector ETF Dip-Buying',
            'period': f"{close.index[0].date()} to {close.index[-1].date()}",
            'cost_rt_pct': COST_RT * 100,
            'rsi_threshold': RSI_THRESHOLD,
            'hold_days': HOLD_DAYS,
            'tp_pct': TP_PCT * 100,
            'sl_pct': SL_PCT * 100,
            'n_permutations': N_PERMUTATIONS,
            'sectors': SECTOR_ETFS,
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': all_results,
        'summary': {
            'baseline_sharpe': baseline_sharpe,
            'variants_improving': len(improving),
            'variants_passing': len(passing),
            'best_variant': passing[0]['label'] if passing else None,
            'best_sharpe': passing[0]['sharpe'] if passing else None,
        }
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    output = convert(output)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
