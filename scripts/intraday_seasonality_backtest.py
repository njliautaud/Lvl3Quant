#!/usr/bin/env python3
"""
Intraday Seasonality / Time-of-Day Effects Backtest
=====================================================
6 variants exploiting well-documented intraday calendar anomalies.
These patterns are driven by different return drivers than daily momentum/mean-reversion,
so should have LOW correlation to QQQ buy-and-hold.

Variants:
  A) Overnight Premium Capture — Buy close, sell next open
  B) Day Session Short — Short open, cover close (inverse of overnight premium)
  C) Monday Effect — Two sub-variants: close-Fri→close-Mon, open-Mon→close-Mon
  D) End-of-Month Window Dressing — Last 3 trading days before month-end → first day of new month
  E) Turn-of-Month — Last 2 + first 3 trading days of each month, cash otherwise
  F) Holiday Effect — Buy 2 days before major holidays, sell day after reopening

Period: Jan 2022 – Jul 2026, 0.02% slippage, 4.5% risk-free rate.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, ≥20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
SMA_WINDOW = 200
PERM_ITERATIONS = 1000
RF_RATE = 0.045  # 4.5% annual risk-free rate
RF_DAILY = RF_RATE / 252

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/intraday_seasonality_results.json")

# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("Downloading data...")
data_start = '2020-06-01'  # Extra history for 200-SMA warmup

spy_raw = yf.download('SPY', start=data_start, end=OOT_END, progress=False, auto_adjust=False)
qqq_raw = yf.download('QQQ', start=data_start, end=OOT_END, progress=False, auto_adjust=False)

# Flatten MultiIndex columns if present
for df in [spy_raw, qqq_raw]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

print(f"  SPY: {len(spy_raw)} days, QQQ: {len(qqq_raw)} days")

# Ensure we have Open and Close
assert 'Open' in qqq_raw.columns and 'Close' in qqq_raw.columns, "Need Open/Close columns"

spy = spy_raw.copy()
qqq = qqq_raw.copy()

# Compute regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
spy['SMA200'] = spy['Close'].rolling(SMA_WINDOW).mean()
spy['regime'] = np.where(spy['Close'] > spy['SMA200'], 'bull', 'bear')

# QQQ buy-and-hold daily returns for correlation calculation
qqq['daily_ret'] = qqq['Close'].pct_change()

# Filter to OOT period
spy_oot = spy.loc[OOT_START:]
qqq_oot = qqq.loc[OOT_START:]

print(f"\nOOT period: {spy_oot.index[0].date()} to {spy_oot.index[-1].date()}")
print(f"  QQQ days: {len(qqq_oot)}, SPY Bull: {(spy_oot['regime']=='bull').sum()}, Bear: {(spy_oot['regime']=='bear').sum()}")


# ── HELPERS ─────────────────────────────────────────────────────────────────
def apply_slippage(price, direction='buy'):
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def get_regime(date):
    if date in spy.index:
        return spy.loc[date, 'regime']
    prior = spy.index[spy.index <= date]
    if len(prior) > 0:
        return spy.loc[prior[-1], 'regime']
    return 'unknown'


def compute_correlation_with_qqq(trades_df):
    """Compute correlation between strategy daily returns and QQQ buy-and-hold daily returns."""
    if len(trades_df) < 5:
        return 0.0

    # Build a daily return series for the strategy
    # For each trade, distribute the return across holding days
    all_dates = qqq_oot.index
    strat_daily = pd.Series(0.0, index=all_dates)

    for _, trade in trades_df.iterrows():
        entry_d = trade['entry_date']
        exit_d = trade['exit_date']
        ret = trade['return_pct']
        # Find trading days in this window
        mask = (all_dates >= entry_d) & (all_dates <= exit_d)
        n_days = mask.sum()
        if n_days > 0:
            # Distribute return evenly across holding days
            daily_contribution = ret / n_days
            strat_daily[mask] += daily_contribution

    # Compute correlation on days where strategy is active
    active_mask = strat_daily != 0
    if active_mask.sum() < 10:
        return 0.0

    qqq_daily_oot = qqq_oot['daily_ret'].reindex(all_dates).fillna(0)
    corr = np.corrcoef(strat_daily[active_mask].values, qqq_daily_oot[active_mask].values)[0, 1]
    return round(float(corr), 4) if not np.isnan(corr) else 0.0


def compute_metrics(trades_df, label):
    if len(trades_df) < 2:
        return {
            'variant': label, 'n_trades': len(trades_df),
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'cagr_pct': 0, 'total_return_pct': 0, 'max_dd_pct': 0,
            'perm_p': 1.0, 'regime_gap': 1.0, 'gates_passed': 0,
            'gate_details': {}, 'qqq_correlation': 0.0,
        }

    returns = trades_df['return_pct'].values
    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualization factor based on trade frequency
    total_days = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days
    years = max(total_days / 365.25, 0.1)
    trades_per_year = n / years
    ann_factor = np.sqrt(trades_per_year)

    # Sharpe (excess over risk-free, scaled per trade)
    rf_per_trade = RF_RATE / max(trades_per_year, 1)
    excess_mean = mean_ret - rf_per_trade
    sharpe = (excess_mean / max(std_ret, 1e-9)) * ann_factor if std_ret > 1e-9 else 0

    # Sortino
    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (excess_mean / max(down_std, 1e-9)) * ann_factor if down_std > 1e-9 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / max(gross_loss, 1e-9)

    # Win rate
    wr = (returns > 0).sum() / n

    # Total return & CAGR
    cumulative = (1 + returns).cumprod()
    total_ret = (cumulative[-1] - 1) * 100
    cagr = ((cumulative[-1]) ** (1 / years) - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    max_dd = dd.min() * 100

    # Permutation test
    observed_mean = mean_ret
    count_ge = 0
    all_qqq_dates = qqq_oot.index.tolist()
    n_qqq = len(all_qqq_dates)
    hold_lengths = (trades_df['exit_date'] - trades_df['entry_date']).dt.days.values
    median_hold = max(int(np.median(hold_lengths)), 1)

    for _ in range(PERM_ITERATIONS):
        rand_indices = np.random.randint(0, max(n_qqq - median_hold - 5, 1), size=n)
        perm_rets = []
        for idx in rand_indices:
            entry_d = all_qqq_dates[idx]
            exit_idx = min(idx + median_hold, n_qqq - 1)
            exit_d = all_qqq_dates[exit_idx]
            if entry_d in qqq.index and exit_d in qqq.index:
                e_price = qqq.loc[entry_d, 'Close']
                x_price = qqq.loc[exit_d, 'Close']
                perm_rets.append((x_price / e_price) - 1)
        if len(perm_rets) > 0 and np.mean(perm_rets) >= observed_mean:
            count_ge += 1
    perm_p = count_ge / PERM_ITERATIONS

    # Regime analysis
    bull_trades = trades_df[trades_df['regime'] == 'bull']['return_pct']
    bear_trades = trades_df[trades_df['regime'] == 'bear']['return_pct']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe = np.mean(bull_trades) / max(np.std(bull_trades, ddof=1), 1e-9) * ann_factor
        bear_sharpe = np.mean(bear_trades) / max(np.std(bear_trades, ddof=1), 1e-9) * ann_factor
        denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
        regime_gap = abs(bull_sharpe - bear_sharpe) / denom
    else:
        regime_gap = 0.0

    # QQQ correlation
    qqq_corr = compute_correlation_with_qqq(trades_df)

    # 5-gate check
    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = regime_gap < 0.5
    g4 = max_dd > -50
    g5 = n >= 20
    gates_passed = sum([g1, g2, g3, g4, g5])

    return {
        'variant': label,
        'n_trades': int(n),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'win_rate': round(float(wr), 3),
        'cagr_pct': round(float(cagr), 2),
        'total_return_pct': round(float(total_ret), 2),
        'max_dd_pct': round(float(max_dd), 2),
        'perm_p': round(float(perm_p), 4),
        'regime_gap': round(float(regime_gap), 3),
        'gates_passed': gates_passed,
        'gate_details': {
            'sharpe_gt_0.5': bool(g1),
            'perm_p_lt_0.05': bool(g2),
            'regime_gap_lt_0.5': bool(g3),
            'max_dd_gt_neg50': bool(g4),
            'trades_gte_20': bool(g5),
        },
        'qqq_correlation': qqq_corr,
        'bull_trades': int(len(bull_trades)),
        'bear_trades': int(len(bear_trades)),
        'mean_return_pct': round(float(mean_ret * 100), 4),
        'bull_sharpe': round(float(bull_sharpe), 3) if len(bull_trades) >= 3 else None,
        'bear_sharpe': round(float(bear_sharpe), 3) if len(bear_trades) >= 3 else None,
    }


# ── VARIANT A: Overnight Premium Capture ────────────────────────────────────
print("\n─── Variant A: Overnight Premium Capture ───")
print("  Buy QQQ at close, sell at next day's open")

def run_overnight_premium():
    trades = []
    dates = qqq_oot.index.tolist()

    for i in range(len(dates) - 1):
        entry_date = dates[i]
        exit_date = dates[i + 1]

        # Buy at today's close, sell at tomorrow's open
        entry_price = apply_slippage(float(qqq.loc[entry_date, 'Close']), 'buy')
        exit_price = apply_slippage(float(qqq.loc[exit_date, 'Open']), 'sell')

        ret = (exit_price / entry_price) - 1
        regime = get_regime(entry_date)

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'return_pct': ret,
            'regime': regime,
        })

    df = pd.DataFrame(trades)
    print(f"  {len(df)} trades generated")
    return compute_metrics(df, 'A) Overnight Premium Capture')


# ── VARIANT B: Day Session Short ────────────────────────────────────────────
print("\n─── Variant B: Day Session Short ───")
print("  Short QQQ at open, cover at close (same day)")

def run_day_session_short():
    trades = []
    dates = qqq_oot.index.tolist()

    for date in dates:
        open_price = float(qqq.loc[date, 'Open'])
        close_price = float(qqq.loc[date, 'Close'])

        # Short at open (borrow and sell), cover at close (buy back)
        entry_price = apply_slippage(open_price, 'sell')  # sell at open
        exit_price = apply_slippage(close_price, 'buy')   # buy at close

        # Short P&L: entry_price - exit_price (profit when price falls)
        ret = (entry_price - exit_price) / entry_price
        regime = get_regime(date)

        trades.append({
            'entry_date': date,
            'exit_date': date,
            'return_pct': ret,
            'regime': regime,
        })

    df = pd.DataFrame(trades)
    print(f"  {len(df)} trades generated")
    return compute_metrics(df, 'B) Day Session Short')


# ── VARIANT C: Monday Effect ────────────────────────────────────────────────
print("\n─── Variant C: Monday Effect ───")

def run_monday_effect():
    """C1: Buy Friday close, sell Monday close. C2: Buy Monday open, sell Monday close."""
    trades_c1 = []
    trades_c2 = []
    dates = qqq_oot.index.tolist()

    for i, date in enumerate(dates):
        dow = date.dayofweek  # 0=Mon, 4=Fri

        # C1: Buy Friday close → sell Monday close
        if dow == 4:  # Friday
            # Find next Monday
            for j in range(i + 1, min(i + 5, len(dates))):
                if dates[j].dayofweek == 0:
                    monday = dates[j]
                    entry_price = apply_slippage(float(qqq.loc[date, 'Close']), 'buy')
                    exit_price = apply_slippage(float(qqq.loc[monday, 'Close']), 'sell')
                    ret = (exit_price / entry_price) - 1
                    trades_c1.append({
                        'entry_date': date,
                        'exit_date': monday,
                        'return_pct': ret,
                        'regime': get_regime(date),
                    })
                    break

        # C2: Buy Monday open → sell Monday close
        if dow == 0:  # Monday
            entry_price = apply_slippage(float(qqq.loc[date, 'Open']), 'buy')
            exit_price = apply_slippage(float(qqq.loc[date, 'Close']), 'sell')
            ret = (exit_price / entry_price) - 1
            trades_c2.append({
                'entry_date': date,
                'exit_date': date,
                'return_pct': ret,
                'regime': get_regime(date),
            })

    df_c1 = pd.DataFrame(trades_c1)
    df_c2 = pd.DataFrame(trades_c2)
    print(f"  C1 (Fri close→Mon close): {len(df_c1)} trades")
    print(f"  C2 (Mon open→Mon close): {len(df_c2)} trades")

    metrics_c1 = compute_metrics(df_c1, 'C1) Monday Effect (Fri→Mon)')
    metrics_c2 = compute_metrics(df_c2, 'C2) Monday Effect (Mon intraday)')
    return metrics_c1, metrics_c2


# ── VARIANT D: End-of-Month Window Dressing ─────────────────────────────────
print("\n─── Variant D: End-of-Month Window Dressing ───")
print("  Buy 3 trading days before month-end, sell first trading day of new month")

def run_eom_window_dressing():
    trades = []
    dates = qqq_oot.index.tolist()
    months = qqq_oot.index.to_period('M').unique()

    for i, month in enumerate(months):
        if i >= len(months) - 1:
            continue  # Need next month

        next_month = months[i + 1]

        # Trading days in this month
        month_days = [d for d in dates if d.to_period('M') == month]
        if len(month_days) < 4:
            continue

        # Entry: 3 trading days before month end (so the 4th-to-last day)
        entry_date = month_days[-3]

        # Exit: first trading day of next month
        next_month_days = [d for d in dates if d.to_period('M') == next_month]
        if len(next_month_days) == 0:
            continue
        exit_date = next_month_days[0]

        entry_price = apply_slippage(float(qqq.loc[entry_date, 'Close']), 'buy')
        exit_price = apply_slippage(float(qqq.loc[exit_date, 'Close']), 'sell')
        ret = (exit_price / entry_price) - 1

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'return_pct': ret,
            'regime': get_regime(entry_date),
        })

    df = pd.DataFrame(trades)
    print(f"  {len(df)} trades generated")
    return compute_metrics(df, 'D) EOM Window Dressing')


# ── VARIANT E: Turn-of-Month ────────────────────────────────────────────────
print("\n─── Variant E: Turn-of-Month ───")
print("  Long QQQ last 2 + first 3 trading days of each month, cash otherwise")

def run_turn_of_month():
    trades = []
    dates = qqq_oot.index.tolist()
    months = qqq_oot.index.to_period('M').unique()

    for i, month in enumerate(months):
        month_days = [d for d in dates if d.to_period('M') == month]
        if len(month_days) < 5:
            continue

        # Last 2 trading days of this month
        tom_entry = month_days[-2]

        # First 3 trading days of next month (or this month's first 3 for first iteration)
        if i < len(months) - 1:
            next_month = months[i + 1]
            next_month_days = [d for d in dates if d.to_period('M') == next_month]
            if len(next_month_days) < 3:
                continue
            tom_exit = next_month_days[2]  # 3rd trading day of next month
        else:
            continue

        entry_price = apply_slippage(float(qqq.loc[tom_entry, 'Close']), 'buy')
        exit_price = apply_slippage(float(qqq.loc[tom_exit, 'Close']), 'sell')
        ret = (exit_price / entry_price) - 1

        trades.append({
            'entry_date': tom_entry,
            'exit_date': tom_exit,
            'return_pct': ret,
            'regime': get_regime(tom_entry),
        })

    df = pd.DataFrame(trades)
    print(f"  {len(df)} trades generated")
    return compute_metrics(df, 'E) Turn-of-Month')


# ── VARIANT F: Holiday Effect ───────────────────────────────────────────────
print("\n─── Variant F: Holiday Effect ───")
print("  Buy 2 days before major holidays, sell day after market reopens")

def run_holiday_effect():
    """
    Major US market holidays:
    - New Year's Day (Jan 1)
    - MLK Day (3rd Mon Jan)
    - Presidents' Day (3rd Mon Feb)
    - Memorial Day (last Mon May)
    - Independence Day (Jul 4)
    - Labor Day (1st Mon Sep)
    - Thanksgiving (4th Thu Nov)
    - Christmas (Dec 25)
    """
    # Define approximate holiday dates for 2022-2026
    holidays = []
    for year in range(2022, 2027):
        # New Year's
        holidays.append(pd.Timestamp(f'{year}-01-01'))
        # MLK Day: 3rd Monday of January
        jan1 = pd.Timestamp(f'{year}-01-01')
        first_mon = jan1 + timedelta(days=(7 - jan1.dayofweek) % 7)
        if first_mon.day > 1:
            mlk = first_mon + timedelta(weeks=2)
        else:
            mlk = first_mon + timedelta(weeks=2)
        holidays.append(mlk)
        # Presidents' Day: 3rd Monday of February
        feb1 = pd.Timestamp(f'{year}-02-01')
        first_mon_feb = feb1 + timedelta(days=(7 - feb1.dayofweek) % 7)
        if first_mon_feb.month != 2:
            first_mon_feb = feb1 + timedelta(days=(0 - feb1.dayofweek) % 7)
        pres = first_mon_feb + timedelta(weeks=2)
        holidays.append(pres)
        # Memorial Day: last Monday of May
        may31 = pd.Timestamp(f'{year}-05-31')
        mem = may31 - timedelta(days=(may31.dayofweek - 0) % 7)
        holidays.append(mem)
        # Independence Day
        holidays.append(pd.Timestamp(f'{year}-07-04'))
        # Labor Day: 1st Monday of September
        sep1 = pd.Timestamp(f'{year}-09-01')
        labor = sep1 + timedelta(days=(7 - sep1.dayofweek) % 7)
        if labor.month != 9:
            labor = sep1
        holidays.append(labor)
        # Thanksgiving: 4th Thursday of November
        nov1 = pd.Timestamp(f'{year}-11-01')
        first_thu = nov1 + timedelta(days=(3 - nov1.dayofweek) % 7)
        thanks = first_thu + timedelta(weeks=3)
        holidays.append(thanks)
        # Christmas
        holidays.append(pd.Timestamp(f'{year}-12-25'))

    trades = []
    dates = qqq_oot.index.tolist()
    date_set = set(dates)

    for holiday in holidays:
        if holiday < pd.Timestamp(OOT_START) or holiday > pd.Timestamp(OOT_END):
            continue

        # Find the last trading day before the holiday
        # The holiday itself may not be a trading day; find the gap
        pre_holiday_dates = [d for d in dates if d < holiday]
        if len(pre_holiday_dates) < 3:
            continue

        # Entry: 2 trading days before the last pre-holiday trading day
        entry_date = pre_holiday_dates[-2]

        # Exit: first trading day after the holiday
        post_holiday_dates = [d for d in dates if d > holiday]
        if len(post_holiday_dates) == 0:
            continue
        exit_date = post_holiday_dates[0]

        # Skip if entry == exit (no actual holiday gap)
        if entry_date >= exit_date:
            continue

        entry_price = apply_slippage(float(qqq.loc[entry_date, 'Close']), 'buy')
        exit_price = apply_slippage(float(qqq.loc[exit_date, 'Close']), 'sell')
        ret = (exit_price / entry_price) - 1

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'return_pct': ret,
            'regime': get_regime(entry_date),
        })

    df = pd.DataFrame(trades)
    # Deduplicate overlapping trades (some holidays are close together)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['entry_date', 'exit_date'])
    print(f"  {len(df)} trades generated")
    return compute_metrics(df, 'F) Holiday Effect')


# ── RUN ALL VARIANTS ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING ALL VARIANTS")
print("=" * 70)

results = []

result_a = run_overnight_premium()
results.append(result_a)

result_b = run_day_session_short()
results.append(result_b)

result_c1, result_c2 = run_monday_effect()
results.append(result_c1)
results.append(result_c2)

result_d = run_eom_window_dressing()
results.append(result_d)

result_e = run_turn_of_month()
results.append(result_e)

result_f = run_holiday_effect()
results.append(result_f)


# ── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INTRADAY SEASONALITY BACKTEST RESULTS")
print(f"Period: {OOT_START} to {OOT_END} | Slippage: {SLIPPAGE_PCT*100:.2f}% | RF: {RF_RATE*100:.1f}%")
print("=" * 70)

header = f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'CAGR%':>7} {'MaxDD%':>7} {'N':>5} {'QQQ_r':>7} {'Perm_p':>7} {'Gates':>5}"
print(header)
print("-" * len(header))

for r in results:
    print(f"{r['variant']:<35} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['profit_factor']:>6.2f} {r['win_rate']:>6.1%} {r['cagr_pct']:>7.2f} {r['max_dd_pct']:>7.2f} {r['n_trades']:>5d} {r['qqq_correlation']:>7.3f} {r['perm_p']:>7.4f} {r['gates_passed']:>3d}/5")

# Gate details
print("\n─── Gate Details ───")
for r in results:
    gates = r['gate_details']
    status = "PASS" if r['gates_passed'] >= 4 else ("MARGINAL" if r['gates_passed'] >= 3 else "FAIL")
    flags = " | ".join([f"{k}={'Y' if v else 'N'}" for k, v in gates.items()])
    print(f"  {r['variant']:<35} [{status}] {flags}")

# Regime breakdown
print("\n─── Regime Analysis ───")
for r in results:
    bull_s = f"{r.get('bull_sharpe', 'N/A')}" if r.get('bull_sharpe') is not None else "N/A"
    bear_s = f"{r.get('bear_sharpe', 'N/A')}" if r.get('bear_sharpe') is not None else "N/A"
    print(f"  {r['variant']:<35} Bull Sharpe: {bull_s:>8} | Bear Sharpe: {bear_s:>8} | Gap: {r['regime_gap']:.3f}")

# QQQ correlation analysis
print("\n─── QQQ Buy-and-Hold Correlation ───")
for r in results:
    corr = r['qqq_correlation']
    tag = "UNCORRELATED" if abs(corr) < 0.2 else ("LOW" if abs(corr) < 0.4 else "HIGH")
    print(f"  {r['variant']:<35} Correlation: {corr:>7.4f} [{tag}]")

# Identify winners
print("\n─── WINNERS (≥4 gates passed) ───")
winners = [r for r in results if r['gates_passed'] >= 4]
if winners:
    for w in sorted(winners, key=lambda x: x['sharpe'], reverse=True):
        print(f"  ★ {w['variant']} — Sharpe {w['sharpe']:.3f}, QQQ corr {w['qqq_correlation']:.3f}")
else:
    print("  No variants passed ≥4 gates.")

# Marginal
marginal = [r for r in results if r['gates_passed'] == 3]
if marginal:
    print("\n─── MARGINAL (3 gates) ───")
    for m in marginal:
        print(f"  ● {m['variant']} — Sharpe {m['sharpe']:.3f}, QQQ corr {m['qqq_correlation']:.3f}")

# ── SAVE RESULTS ────────────────────────────────────────────────────────────
output = {
    'metadata': {
        'script': 'intraday_seasonality_backtest.py',
        'run_date': datetime.now().isoformat(),
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'slippage_pct': SLIPPAGE_PCT,
        'risk_free_rate': RF_RATE,
        'perm_iterations': PERM_ITERATIONS,
        'n_variants': len(results),
    },
    'results': results,
    'summary': {
        'winners_4plus_gates': [r['variant'] for r in results if r['gates_passed'] >= 4],
        'best_sharpe': max(results, key=lambda x: x['sharpe'])['variant'],
        'lowest_qqq_correlation': min(results, key=lambda x: abs(x['qqq_correlation']))['variant'],
        'best_combined': sorted(results, key=lambda x: x['sharpe'] * (1 - abs(x['qqq_correlation'])), reverse=True)[0]['variant'] if results else None,
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_PATH}")
print("Done.")
