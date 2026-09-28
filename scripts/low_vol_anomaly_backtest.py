#!/usr/bin/env python3
"""
Low Volatility Anomaly (Defensive Equity) Backtest
===================================================
Academic basis: Baker, Bradley & Wurgler (2011) "Benchmarks as Limits to Arbitrage."
Low-volatility stocks consistently outperform high-volatility stocks on a
risk-adjusted basis due to investors' lottery-ticket bias overpaying for volatile stocks.

6 variants, OOT Jan 2022 - Jul 2026, 5-gate validation.
Universe: 50 top S&P 500 stocks, $645 starting capital.
Cost: $0 commission, 0.02% slippage per side.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per side

STOCKS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'XOM', 'UNH', 'V', 'MA', 'COST', 'PG', 'JNJ', 'HD', 'WMT',
    'NFLX', 'CRM', 'ABBV', 'BAC', 'ORCL', 'CVX', 'MRK', 'KO', 'PEP', 'AMD',
    'ACN', 'ADBE', 'TMO', 'CSCO', 'LIN', 'MCD', 'ABT', 'WFC', 'GE', 'DHR',
    'PM', 'QCOM', 'TXN', 'ISRG', 'INTU', 'AMGN', 'CAT', 'AMAT', 'BX', 'NOW',
]

ALL_TICKERS = list(set(STOCKS + ['SPY']))

DATA_START = '2021-06-01'  # Extra lookback for 60d vol + 200-SMA
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'

VOL_LOOKBACK = 60     # 60-day realized volatility
SMA_LOOKBACK = 200    # 200-day SMA for regime/quality filter
N_LONG = 10           # Bottom 10 by vol (variants A/C/D/E/F)
N_LONG_B = 5          # Bottom 5 (variant B)
N_SHORT = 10          # Top 10 by vol (variant E)
N_PERMUTATIONS = 1000

np.random.seed(42)

# ─── DATA DOWNLOAD ────────────────────────────────────────────────────────────
print("Downloading price data...")
data = yf.download(ALL_TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    close = data['Close'].copy()
else:
    close = data.copy()

if isinstance(close.columns, pd.MultiIndex):
    close.columns = close.columns.get_level_values(-1)

close = close.ffill().dropna(how='all')
close = close.ffill().bfill()

# Filter to stocks that actually have data
available_stocks = [s for s in STOCKS if s in close.columns]
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
print(f"Stocks available: {len(available_stocks)}/{len(STOCKS)}")

# ─── PRECOMPUTE ───────────────────────────────────────────────────────────────
returns = close.pct_change()

# 60-day rolling realized volatility (annualized std of daily returns)
rolling_vol = returns[available_stocks].rolling(VOL_LOOKBACK).std() * np.sqrt(252)

# SPY 200-SMA for regime detection
spy_close = close['SPY']
spy_sma200 = spy_close.rolling(SMA_LOOKBACK).mean()
regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

# Per-stock 200-SMA (for variant F)
stock_sma200 = close[available_stocks].rolling(SMA_LOOKBACK).mean()

# OOT filter
oot_mask = close.index >= OOT_START
oot_regime = regime[oot_mask]
bull_days = int((oot_regime == 1).sum())
bear_days = int((oot_regime == 0).sum())
print(f"OOT: {OOT_START} to {OOT_END}, {bull_days} bull days, {bear_days} bear days")


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_monthly_rebal_dates(idx, start, end):
    """First trading day of each month in range."""
    mask = (idx >= start) & (idx <= end)
    sub = idx[mask]
    dates = []
    seen = set()
    for d in sub:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            dates.append(d)
    return dates


def get_weekly_rebal_dates(idx, start, end):
    """First trading day of each week in range."""
    mask = (idx >= start) & (idx <= end)
    sub = idx[mask]
    dates = []
    seen = set()
    for d in sub:
        key = (d.isocalendar()[0], d.isocalendar()[1])
        if key not in seen:
            seen.add(key)
            dates.append(d)
    return dates


def rank_by_vol(date, n_bottom=10, n_top=0, above_sma=False):
    """
    Rank available stocks by trailing 60d vol at given date.
    Returns (long_picks, short_picks).
    """
    if date not in rolling_vol.index:
        return [], []

    vols = rolling_vol.loc[date].dropna()
    if above_sma:
        # Filter: stock must be above its own 200-SMA
        prices_at_date = close.loc[date, available_stocks]
        sma_at_date = stock_sma200.loc[date]
        above_mask = prices_at_date > sma_at_date
        eligible = above_mask[above_mask].index.tolist()
        vols = vols[[s for s in vols.index if s in eligible]]

    if len(vols) < n_bottom:
        return list(vols.nsmallest(len(vols)).index), []

    long_picks = list(vols.nsmallest(n_bottom).index)
    short_picks = list(vols.nlargest(n_top).index) if n_top > 0 else []
    return long_picks, short_picks


def simulate_long_only(select_fn, rebal_dates, name):
    """
    Equal-weight long-only monthly/weekly rebalance simulation.
    Returns daily return series.
    """
    all_daily_rets = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        long_picks, _ = select_fn(date)
        if not long_picks:
            # No picks: stay in cash (0% return)
            period = returns.loc[date:next_date].iloc[1:]  # exclude rebal day
            if len(period) > 0:
                cash_rets = pd.Series(0.0, index=period.index)
                all_daily_rets.append(cash_rets)
            continue

        # Equal weight, with slippage on entry
        period_rets = returns.loc[date:next_date, long_picks].iloc[1:]
        if len(period_rets) > 0:
            # Slippage: reduce first day's return by slippage (entry cost)
            adjusted = period_rets.copy()
            adjusted.iloc[0] = adjusted.iloc[0] - SLIPPAGE_PCT
            # Slippage on exit (last day)
            adjusted.iloc[-1] = adjusted.iloc[-1] - SLIPPAGE_PCT
            port_ret = adjusted.mean(axis=1)
            all_daily_rets.append(port_ret)

    # Handle last period to end of OOT
    if rebal_dates:
        last_date = rebal_dates[-1]
        long_picks, _ = select_fn(last_date)
        if long_picks:
            period_rets = returns.loc[last_date:OOT_END, long_picks].iloc[1:]
            if len(period_rets) > 0:
                adjusted = period_rets.copy()
                adjusted.iloc[0] = adjusted.iloc[0] - SLIPPAGE_PCT
                port_ret = adjusted.mean(axis=1)
                all_daily_rets.append(port_ret)

    if all_daily_rets:
        return pd.concat(all_daily_rets).sort_index()
    return pd.Series(dtype=float)


def simulate_long_short(select_fn, rebal_dates, name):
    """
    Long bottom-N, short top-N, equal weight, dollar-neutral.
    Returns daily return series.
    """
    all_daily_rets = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        long_picks, short_picks = select_fn(date)
        if not long_picks or not short_picks:
            period = returns.loc[date:next_date].iloc[1:]
            if len(period) > 0:
                all_daily_rets.append(pd.Series(0.0, index=period.index))
            continue

        period_rets = returns.loc[date:next_date].iloc[1:]
        if len(period_rets) == 0:
            continue

        # Long side (equal weight)
        long_rets = period_rets[[s for s in long_picks if s in period_rets.columns]]
        if len(long_rets.columns) == 0:
            continue
        long_avg = long_rets.mean(axis=1)

        # Short side (equal weight, negative returns = profit when stock falls)
        short_rets = period_rets[[s for s in short_picks if s in period_rets.columns]]
        if len(short_rets.columns) == 0:
            continue
        short_avg = -short_rets.mean(axis=1)  # short = negative exposure

        # Dollar-neutral: 50% long, 50% short
        port_ret = 0.5 * long_avg + 0.5 * short_avg

        # Slippage on both legs
        adjusted = port_ret.copy()
        adjusted.iloc[0] = adjusted.iloc[0] - 2 * SLIPPAGE_PCT  # entry both legs
        adjusted.iloc[-1] = adjusted.iloc[-1] - 2 * SLIPPAGE_PCT  # exit both legs

        all_daily_rets.append(adjusted)

    # Last period
    if rebal_dates:
        last_date = rebal_dates[-1]
        long_picks, short_picks = select_fn(last_date)
        if long_picks and short_picks:
            period_rets = returns.loc[last_date:OOT_END].iloc[1:]
            if len(period_rets) > 0:
                long_rets = period_rets[[s for s in long_picks if s in period_rets.columns]]
                short_rets = period_rets[[s for s in short_picks if s in period_rets.columns]]
                if len(long_rets.columns) > 0 and len(short_rets.columns) > 0:
                    port_ret = 0.5 * long_rets.mean(axis=1) - 0.5 * short_rets.mean(axis=1)
                    adjusted = port_ret.copy()
                    adjusted.iloc[0] = adjusted.iloc[0] - 2 * SLIPPAGE_PCT
                    all_daily_rets.append(adjusted)

    if all_daily_rets:
        return pd.concat(all_daily_rets).sort_index()
    return pd.Series(dtype=float)


def compute_metrics(daily_rets, name):
    """Compute all required metrics from a daily return series."""
    daily_rets = daily_rets.dropna()
    if len(daily_rets) < 20:
        return None

    total_ret = (1 + daily_rets).prod() - 1
    n_years = len(daily_rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / max(ann_vol, 1e-8)

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = (daily_rets.mean() * 252) / max(downside, 1e-8)

    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Monthly returns for win rate, PF, trade count
    monthly = daily_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly = monthly.dropna()
    win_rate = (monthly > 0).sum() / max(len(monthly), 1)
    gross_profit = monthly[monthly > 0].sum()
    gross_loss = abs(monthly[monthly < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 1e-8)
    n_trades = len(monthly)  # each monthly rebalance = a trade

    # Regime-stratified Sharpe
    aligned_regime = oot_regime.reindex(daily_rets.index).ffill()
    bull_rets = daily_rets[aligned_regime == 1]
    bear_rets = daily_rets[aligned_regime == 0]

    bull_sharpe = (bull_rets.mean() * 252) / max(bull_rets.std() * np.sqrt(252), 1e-8) if len(bull_rets) > 20 else 0
    bear_sharpe = (bear_rets.mean() * 252) / max(bear_rets.std() * np.sqrt(252), 1e-8) if len(bear_rets) > 20 else 0

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    final_equity = round(INITIAL_CAPITAL * (1 + total_ret), 2)

    return {
        'name': name,
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'win_rate_monthly_pct': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'total_trades': n_trades,
        'final_equity': final_equity,
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'daily_returns': daily_rets,  # kept for permutation test, removed before save
    }


def permutation_test(actual_sharpe, daily_rets, n_perms=N_PERMUTATIONS):
    """
    Shuffle vol rankings each month to test if the specific ranking matters.
    For each permutation: randomly pick N stocks (ignoring vol rank) and compute Sharpe.
    If random picks beat actual Sharpe often, the vol ranking has no edge.
    """
    monthly_dates = get_monthly_rebal_dates(close.index, OOT_START, OOT_END)
    count_better = 0

    for _ in range(n_perms):
        sim_daily_rets = []
        for i in range(len(monthly_dates) - 1):
            date = monthly_dates[i]
            next_date = monthly_dates[i + 1]

            # Random 10 stocks instead of vol-ranked
            eligible = [s for s in available_stocks if s in returns.columns]
            if len(eligible) < N_LONG:
                continue
            chosen = list(np.random.choice(eligible, size=N_LONG, replace=False))

            period_rets = returns.loc[date:next_date, chosen].iloc[1:]
            if len(period_rets) > 0:
                sim_daily_rets.append(period_rets.mean(axis=1))

        if sim_daily_rets:
            combined = pd.concat(sim_daily_rets)
            sim_sharpe = combined.mean() * 252 / max(combined.std() * np.sqrt(252), 1e-8)
            if sim_sharpe >= actual_sharpe:
                count_better += 1

    return count_better / n_perms


# ─── STRATEGY A: Bottom 10 by 60d vol, monthly rebalance, equal weight ────────
print("\n[A] Bottom 10 by 60d vol, monthly rebalance...")
monthly_dates = get_monthly_rebal_dates(close.index, OOT_START, OOT_END)

def select_a(date):
    return rank_by_vol(date, n_bottom=N_LONG)

strat_a_rets = simulate_long_only(select_a, monthly_dates, "A")
result_a = compute_metrics(strat_a_rets, "A_Bottom10_Monthly")


# ─── STRATEGY B: Bottom 5 by 60d vol (more concentrated) ─────────────────────
print("[B] Bottom 5 by 60d vol, monthly rebalance...")

def select_b(date):
    return rank_by_vol(date, n_bottom=N_LONG_B)

strat_b_rets = simulate_long_only(select_b, monthly_dates, "B")
result_b = compute_metrics(strat_b_rets, "B_Bottom5_Monthly")


# ─── STRATEGY C: Bottom 10, half-size in bear (SPY < 200-SMA) ────────────────
print("[C] Bottom 10, half-size in bear...")

def simulate_half_bear(select_fn, rebal_dates, name):
    """Like long-only but 50% cash allocation in bear regime."""
    all_daily_rets = []

    for i in range(len(rebal_dates) - 1):
        date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        long_picks, _ = select_fn(date)
        if not long_picks:
            period = returns.loc[date:next_date].iloc[1:]
            if len(period) > 0:
                all_daily_rets.append(pd.Series(0.0, index=period.index))
            continue

        # Check regime at rebalance date
        is_bear = regime.loc[:date].iloc[-1] == 0 if len(regime.loc[:date]) > 0 else False
        weight = 0.5 if is_bear else 1.0

        period_rets = returns.loc[date:next_date, long_picks].iloc[1:]
        if len(period_rets) > 0:
            adjusted = period_rets.copy()
            adjusted.iloc[0] = adjusted.iloc[0] - SLIPPAGE_PCT
            adjusted.iloc[-1] = adjusted.iloc[-1] - SLIPPAGE_PCT
            port_ret = adjusted.mean(axis=1) * weight
            all_daily_rets.append(port_ret)

    # Last period
    if rebal_dates:
        last_date = rebal_dates[-1]
        long_picks, _ = select_fn(last_date)
        if long_picks:
            is_bear = regime.loc[:last_date].iloc[-1] == 0
            weight = 0.5 if is_bear else 1.0
            period_rets = returns.loc[last_date:OOT_END, long_picks].iloc[1:]
            if len(period_rets) > 0:
                adjusted = period_rets.copy()
                adjusted.iloc[0] = adjusted.iloc[0] - SLIPPAGE_PCT
                port_ret = adjusted.mean(axis=1) * weight
                all_daily_rets.append(port_ret)

    if all_daily_rets:
        return pd.concat(all_daily_rets).sort_index()
    return pd.Series(dtype=float)

strat_c_rets = simulate_half_bear(select_a, monthly_dates, "C")
result_c = compute_metrics(strat_c_rets, "C_Bottom10_HalfBear")


# ─── STRATEGY D: Bottom 10, WEEKLY rebalance ─────────────────────────────────
print("[D] Bottom 10, weekly rebalance...")
weekly_dates = get_weekly_rebal_dates(close.index, OOT_START, OOT_END)

strat_d_rets = simulate_long_only(select_a, weekly_dates, "D")
result_d = compute_metrics(strat_d_rets, "D_Bottom10_Weekly")


# ─── STRATEGY E: Long bottom 10, short top 10 (market neutral) ───────────────
print("[E] Long bottom 10, short top 10 (market neutral)...")

def select_e(date):
    return rank_by_vol(date, n_bottom=N_LONG, n_top=N_SHORT)

strat_e_rets = simulate_long_short(select_e, monthly_dates, "E")
result_e = compute_metrics(strat_e_rets, "E_LongShort_Neutral")


# ─── STRATEGY F: Bottom 10 + above own 200-SMA (quality + low vol) ───────────
print("[F] Bottom 10 + above own 200-SMA (quality + low vol)...")

def select_f(date):
    return rank_by_vol(date, n_bottom=N_LONG, above_sma=True)

strat_f_rets = simulate_long_only(select_f, monthly_dates, "F")
result_f = compute_metrics(strat_f_rets, "F_Bottom10_AboveSMA")


# ─── PERMUTATION TESTS & 5-GATE VALIDATION ───────────────────────────────────
print("\nRunning permutation tests (1000 shuffles each)...")
results_all = []

for label, result in [('A', result_a), ('B', result_b), ('C', result_c),
                       ('D', result_d), ('E', result_e), ('F', result_f)]:
    if result is None:
        print(f"  [{label}] Skipped (insufficient data)")
        continue

    daily_rets = result.pop('daily_returns')
    perm_p = permutation_test(result['sharpe'], daily_rets)
    result['perm_p'] = round(perm_p, 4)

    # 5-gate validation
    gates = {
        'sharpe_gt_0.5': result['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': result['regime_gap'] < 0.5,
        'mdd_gt_neg50': result['max_drawdown_pct'] > -50,
        'trades_gte_20': result['total_trades'] >= 20,
    }
    result['gates'] = gates
    result['gates_passed'] = sum(gates.values())
    result['all_gates_pass'] = all(gates.values())

    results_all.append(result)
    status = "PASS" if result['all_gates_pass'] else "FAIL"
    print(f"  [{label}] {result['name']}: Sharpe={result['sharpe']:.3f}, "
          f"RegimeGap={result['regime_gap']:.3f}, perm_p={perm_p:.4f} -> {status} ({result['gates_passed']}/5)")


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("LOW VOLATILITY ANOMALY BACKTEST — OOT Jan 2022 - Jul 2026")
print("=" * 90)
print(f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MDD%':>7} "
      f"{'WR%':>5} {'PF':>6} {'RGap':>6} {'perm_p':>7} {'Gates':>6}")
print("-" * 90)
for r in sorted(results_all, key=lambda x: -x['sharpe']):
    status = "PASS" if r['all_gates_pass'] else f"{r['gates_passed']}/5"
    print(f"{r['name']:<30} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr_pct']:>6.1f}% "
          f"{r['max_drawdown_pct']:>6.1f}% {r['win_rate_monthly_pct']:>4.1f}% {r['profit_factor']:>6.3f} "
          f"{r['regime_gap']:>6.3f} {r['perm_p']:>7.4f} {status:>6}")

print("\nRegime-Stratified Detail:")
print(f"{'Strategy':<30} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Regime Gap':>11} {'Final $':>9}")
print("-" * 80)
for r in sorted(results_all, key=lambda x: x['regime_gap']):
    print(f"{r['name']:<30} {r['bull_sharpe']:>12.3f} {r['bear_sharpe']:>12.3f} "
          f"{r['regime_gap']:>11.3f} {r['final_equity']:>8.2f}")

print("\n--- Variant E (Long/Short) Analysis ---")
if result_e is not None:
    e = next((r for r in results_all if r['name'] == 'E_LongShort_Neutral'), None)
    if e:
        print(f"  This is the key test: if low-vol beats high-vol in BOTH regimes,")
        print(f"  the anomaly is real (not just beta exposure).")
        print(f"  Bull Sharpe: {e['bull_sharpe']:.3f}, Bear Sharpe: {e['bear_sharpe']:.3f}")
        print(f"  Regime Gap: {e['regime_gap']:.3f} ({'LOW - genuine anomaly' if e['regime_gap'] < 0.5 else 'HIGH - may be beta'})")
        if e['bull_sharpe'] > 0 and e['bear_sharpe'] > 0:
            print(f"  POSITIVE in both regimes -> strong evidence of low-vol anomaly")
        elif e['bear_sharpe'] > e['bull_sharpe']:
            print(f"  Stronger in bear -> defensive nature confirmed")


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
output = {
    'run_timestamp': datetime.now().isoformat(),
    'strategy': 'Low Volatility Anomaly (Defensive Equity)',
    'academic_basis': 'Baker, Bradley & Wurgler (2011) - Benchmarks as Limits to Arbitrage',
    'config': {
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'starting_capital': INITIAL_CAPITAL,
        'universe_size': len(available_stocks),
        'vol_lookback_days': VOL_LOOKBACK,
        'slippage_per_side': SLIPPAGE_PCT,
        'commission': 0,
        'n_permutations': N_PERMUTATIONS,
        'regime_method': 'SPY_200SMA',
        'bull_days': bull_days,
        'bear_days': bear_days,
    },
    'variants': {
        'A': 'Bottom 10 by 60d vol, monthly rebalance, equal weight',
        'B': 'Bottom 5 by 60d vol (more concentrated)',
        'C': 'Bottom 10, half-size in bear (SPY < 200-SMA)',
        'D': 'Bottom 10, weekly rebalance',
        'E': 'Long bottom 10, short top 10 (market neutral)',
        'F': 'Bottom 10 + above own 200-SMA (quality + low vol)',
    },
    'strategies': results_all,
    'summary': {
        'total_strategies': len(results_all),
        'passing_all_gates': sum(1 for r in results_all if r['all_gates_pass']),
        'best_sharpe': max((r['sharpe'] for r in results_all), default=None),
        'best_regime_gap': min((r['regime_gap'] for r in results_all), default=None),
        'strategies_passing': [r['name'] for r in results_all if r['all_gates_pass']],
        'strategies_with_low_regime_gap': [r['name'] for r in results_all if r['regime_gap'] < 0.5],
    }
}

output_path = '/home/jupiter/Lvl3Quant/data/low_vol_anomaly_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print(f"Strategies passing ALL 5 gates: {output['summary']['passing_all_gates']}/{len(results_all)}")
if output['summary']['best_regime_gap'] is not None:
    print(f"Lowest regime gap: {output['summary']['best_regime_gap']:.3f}")
