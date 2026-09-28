#!/usr/bin/env python3
"""
Sector Rotation with Timing Overlay Backtest
=============================================
Tests 6 variants of sector rotation combined with timing filters.
Validates with permutation tests, regime analysis, and standard metrics.

Optimized: permutation test uses precomputed monthly returns (vectorized).
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
BENCHMARK = 'SPY'
START = '2021-06-01'
OOT_START = '2022-01-01'
OOT_END = '2026-07-28'
INITIAL_CAPITAL = 645.0
MAX_PER_SECTOR = 200.0
SLIPPAGE_PCT = 0.0002
N_PERMS = 1000
SEED = 42

# ── Data ────────────────────────────────────────────────────────────────────
print("Downloading data...")
tickers = SECTORS + [BENCHMARK, '^VIX']
data = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)
close = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
if '^VIX' in close.columns:
    close.rename(columns={'^VIX': 'VIX'}, inplace=True)
close = close.dropna(subset=[BENCHMARK]).ffill()
print(f"Data: {close.shape}, {close.index[0].date()} to {close.index[-1].date()}")

# ── Indicators ──────────────────────────────────────────────────────────────
ret_5d = close[SECTORS].pct_change(5)
ret_10d = close[SECTORS].pct_change(10)
ret_21d = close[SECTORS].pct_change(21)
spy_sma50 = close[BENCHMARK].rolling(50).mean()
spy_sma200 = close[BENCHMARK].rolling(200).mean()
sector_sma50 = close[SECTORS].rolling(50).mean()
vix = close.get('VIX', pd.Series(20.0, index=close.index))
vix_sma5 = vix.rolling(5).mean()

# OOT dates and rebalance dates
oot_dates = close.index[close.index >= OOT_START]
rebal_dates = oot_dates.to_series().groupby([oot_dates.year, oot_dates.month]).first().values
rebal_dates = pd.DatetimeIndex(rebal_dates)
print(f"OOT days: {len(oot_dates)}, Rebalance dates: {len(rebal_dates)}")

# ── Precompute monthly sector returns (rebal-to-rebal) ──────────────────────
# For each rebal period, compute the return of each sector from rebal_date[i] to rebal_date[i+1]
monthly_sector_returns = {}  # rebal_date -> {sector: return}
rebal_list = list(rebal_dates)

for i in range(len(rebal_list) - 1):
    d_start = rebal_list[i]
    d_end = rebal_list[i + 1]
    rets = {}
    for s in SECTORS:
        p0 = close[s].loc[d_start]
        p1 = close[s].loc[:d_end].iloc[-1]
        if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
            rets[s] = (p1 / p0) - 1
        else:
            rets[s] = 0.0
    monthly_sector_returns[d_start] = rets

# Last period: from last rebal to end
if len(rebal_list) > 0:
    d_start = rebal_list[-1]
    rets = {}
    for s in SECTORS:
        p0 = close[s].loc[d_start]
        p1 = close[s].iloc[-1]
        if p0 > 0 and not np.isnan(p0) and not np.isnan(p1):
            rets[s] = (p1 / p0) - 1
        else:
            rets[s] = 0.0
    monthly_sector_returns[d_start] = rets


# ── Sector Selection Functions ──────────────────────────────────────────────

def select_A(date):
    v = vix.loc[:date].iloc[-1]
    v_ma = vix_sma5.loc[:date].iloc[-1]
    if v > 22 or v > v_ma:
        return []
    mom = ret_21d.loc[date].dropna()
    return mom.nlargest(min(3, len(mom))).index.tolist() if len(mom) else []

def select_B(date):
    if close[BENCHMARK].loc[:date].iloc[-1] < spy_sma50.loc[:date].iloc[-1]:
        return []
    eligible = [s for s in SECTORS
                if not np.isnan(sector_sma50[s].loc[:date].iloc[-1])
                and close[s].loc[:date].iloc[-1] > sector_sma50[s].loc[:date].iloc[-1]]
    if not eligible:
        return []
    mom = ret_21d.loc[date][eligible].dropna()
    return mom.nlargest(min(3, len(mom))).index.tolist() if len(mom) else []

def select_C(date):
    mom = ret_21d.loc[date].dropna()
    if len(mom) < 3:
        return mom.index.tolist()
    return mom.nlargest(2).index.tolist() + mom.nsmallest(1).index.tolist()

def select_D(date):
    mom = ret_21d.loc[date].dropna()
    if len(mom) < 3:
        return mom.index.tolist()
    spy_p = close[BENCHMARK].loc[:date].iloc[-1]
    spy_ma = spy_sma200.loc[:date].iloc[-1]
    v = vix.loc[:date].iloc[-1]
    if spy_p > spy_ma and v < 22:
        return mom.nlargest(3).index.tolist()
    else:
        return mom.nsmallest(3).index.tolist()

def select_E(date):
    r5, r10, r21_ = ret_5d.loc[date], ret_10d.loc[date], ret_21d.loc[date]
    acc = [s for s in SECTORS
           if pd.notna(r5[s]) and pd.notna(r10[s]) and pd.notna(r21_[s])
           and r5[s] > r10[s] > r21_[s] and r5[s] > 0]
    if not acc:
        return []
    return r5[acc].sort_values(ascending=False).head(3).index.tolist()

def select_F(date, rng=None):
    if rng is None:
        rng = np.random.RandomState(hash(str(date)) % 2**31)
    return list(rng.choice(SECTORS, size=3, replace=False))


# ── Full Day-by-Day Backtest (for accurate metrics) ─────────────────────────

def run_backtest(selector_fn, exit_fn=None):
    capital = INITIAL_CAPITAL
    holdings = {}
    portfolio_values = []
    trades = 0

    for date in oot_dates:
        # Exit check
        if exit_fn and holdings:
            kept = exit_fn(date, list(holdings.keys()))
            for t in list(holdings.keys()):
                if t not in kept:
                    p = close[t].loc[:date].iloc[-1]
                    capital += holdings[t] * p * (1 - SLIPPAGE_PCT)
                    del holdings[t]
                    trades += 1

        if date in rebal_dates:
            for t, sh in holdings.items():
                p = close[t].loc[:date].iloc[-1]
                capital += sh * p * (1 - SLIPPAGE_PCT)
                if sh > 0:
                    trades += 1
            holdings = {}
            selected = selector_fn(date)
            if selected:
                n = len(selected)
                alloc = min(MAX_PER_SECTOR, capital / n)
                for t in selected:
                    p = close[t].loc[:date].iloc[-1]
                    if p > 0 and not np.isnan(p):
                        bp = p * (1 + SLIPPAGE_PCT)
                        sh = alloc / bp
                        holdings[t] = sh
                        capital -= sh * bp
                        trades += 1

        mtm = capital
        for t, sh in holdings.items():
            p = close[t].loc[:date].iloc[-1]
            if not np.isnan(p):
                mtm += sh * p
        portfolio_values.append(mtm)

    pv = pd.DataFrame({'value': portfolio_values}, index=oot_dates)
    pv['return'] = pv['value'].pct_change()
    return pv, trades


# ── FAST Permutation Test (vectorized monthly returns) ──────────────────────

def fast_permutation_test(selector_fn, actual_sharpe, n_perms=N_PERMS):
    """
    Permutation test using precomputed monthly returns.
    Instead of running full day-by-day backtest for each perm,
    compute portfolio return as weighted average of selected sector returns.
    """
    # Get actual selections and their counts per rebalance date
    selections = {}
    counts = {}
    for d in rebal_list:
        sel = selector_fn(d)
        selections[d] = sel
        counts[d] = len(sel)

    # Compute actual monthly portfolio returns (equal-weight, approximate)
    # This is an approximation that ignores compounding within month and position sizing caps
    actual_monthly_rets = []
    for d in rebal_list:
        if d in monthly_sector_returns:
            sel = selections[d]
            if sel:
                avg_ret = np.mean([monthly_sector_returns[d].get(s, 0.0) for s in sel])
                actual_monthly_rets.append(avg_ret - SLIPPAGE_PCT * 2)  # approx slippage
            else:
                actual_monthly_rets.append(0.0)

    actual_monthly_rets = np.array(actual_monthly_rets)

    # Build matrix of all sector returns per period: shape (n_periods, n_sectors)
    n_periods = len(rebal_list)
    sector_ret_matrix = np.zeros((n_periods, len(SECTORS)))
    for i, d in enumerate(rebal_list):
        if d in monthly_sector_returns:
            for j, s in enumerate(SECTORS):
                sector_ret_matrix[i, j] = monthly_sector_returns[d].get(s, 0.0)

    rng = np.random.RandomState(SEED)
    perm_sharpes = np.zeros(n_perms)

    for p in range(n_perms):
        perm_rets = np.zeros(n_periods)
        for i, d in enumerate(rebal_list):
            n = counts[d]
            if n > 0:
                idx = rng.choice(len(SECTORS), size=min(n, len(SECTORS)), replace=False)
                perm_rets[i] = np.mean(sector_ret_matrix[i, idx]) - SLIPPAGE_PCT * 2
            else:
                perm_rets[i] = 0.0

        if len(perm_rets) > 2 and perm_rets.std() > 0:
            perm_sharpes[p] = perm_rets.mean() / perm_rets.std() * np.sqrt(12)  # annualized monthly

    # Recompute actual sharpe on monthly basis for apples-to-apples comparison
    if len(actual_monthly_rets) > 2 and actual_monthly_rets.std() > 0:
        actual_monthly_sharpe = actual_monthly_rets.mean() / actual_monthly_rets.std() * np.sqrt(12)
    else:
        actual_monthly_sharpe = 0.0

    p_value = float(np.mean(perm_sharpes >= actual_monthly_sharpe))

    return {
        'p_value': round(p_value, 4),
        'actual_sharpe': round(actual_sharpe, 3),
        'actual_monthly_sharpe': round(float(actual_monthly_sharpe), 3),
        'perm_mean_sharpe': round(float(np.mean(perm_sharpes)), 3),
        'perm_std_sharpe': round(float(np.std(perm_sharpes)), 3),
        'perm_95th': round(float(np.percentile(perm_sharpes, 95)), 3)
    }


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pv, trades):
    rets = pv['return'].dropna()
    if len(rets) < 20 or rets.std() == 0:
        return {'sharpe': 0, 'sortino': 0, 'total_return': 0, 'max_drawdown': 0,
                'trades': int(trades), 'profit_factor': 0, 'win_rate': 0, 'cagr': 0,
                'final_value': round(float(pv['value'].iloc[-1]), 2),
                'start_value': round(float(pv['value'].iloc[0]), 2)}

    ann = np.sqrt(252)
    sharpe = float(rets.mean() / rets.std() * ann)
    ds = rets[rets < 0].std()
    sortino = float(rets.mean() / ds * ann) if ds > 0 else 0.0
    total_ret = float(pv['value'].iloc[-1] / pv['value'].iloc[0] - 1)
    cummax = pv['value'].cummax()
    max_dd = float(((pv['value'] - cummax) / cummax).min())
    years = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr = float((pv['value'].iloc[-1] / pv['value'].iloc[0]) ** (1/years) - 1) if years > 0 else 0.0

    monthly = rets.resample('ME').sum()
    wins = monthly[monthly > 0]
    losses = monthly[monthly < 0]
    wr = float(len(wins) / len(monthly)) if len(monthly) > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 999.0

    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'total_return': round(total_ret * 100, 2), 'max_drawdown': round(max_dd * 100, 2),
        'trades': int(trades), 'profit_factor': round(pf, 3),
        'win_rate': round(wr * 100, 1), 'cagr': round(cagr * 100, 2),
        'final_value': round(float(pv['value'].iloc[-1]), 2),
        'start_value': round(float(pv['value'].iloc[0]), 2)
    }


def regime_analysis(pv):
    rets = pv['return'].dropna()
    spy_regime = (close[BENCHMARK] > spy_sma200).reindex(rets.index).ffill()
    bull_rets = rets[spy_regime == True]
    bear_rets = rets[spy_regime == False]
    ann = np.sqrt(252)
    bs = float(bull_rets.mean() / bull_rets.std() * ann) if len(bull_rets) > 20 and bull_rets.std() > 0 else 0.0
    brs = float(bear_rets.mean() / bear_rets.std() * ann) if len(bear_rets) > 20 and bear_rets.std() > 0 else 0.0
    gap = abs(bs - brs) / max(abs(bs), abs(brs)) if max(abs(bs), abs(brs)) > 0 else 0.0
    return {'bull_sharpe': round(bs, 3), 'bear_sharpe': round(brs, 3),
            'regime_gap': round(gap, 3), 'bull_days': int(len(bull_rets)), 'bear_days': int(len(bear_rets))}


# ── Run All Variants ────────────────────────────────────────────────────────

variants = {
    'A_VIX_Gate': {'selector': select_A, 'exit': lambda d, h: [] if vix.loc[:d].iloc[-1] > 25 else h,
                   'desc': 'Sector Mom + VIX Gate (VIX<22 & declining)'},
    'B_Trend_Filter': {'selector': select_B, 'exit': None,
                       'desc': 'Sector Mom + Trend Filter (SPY>50SMA, sector>50SMA)'},
    'C_MomRevert_Combo': {'selector': select_C, 'exit': None,
                          'desc': 'Top-2 Momentum + Worst-1 (natural hedge)'},
    'D_RiskOnOff': {'selector': select_D, 'exit': None,
                    'desc': 'Risk-On/Off Switch (SPY>200SMA & VIX<22)'},
    'E_MomAccel': {'selector': select_E, 'exit': None,
                   'desc': 'Momentum Acceleration (5d>10d>21d)'},
    'F_Random': {'selector': select_F, 'exit': None,
                 'desc': 'ADVERSARIAL: Random 3 sectors'}
}

results = {}

for vname, vconfig in variants.items():
    print(f"\n{'='*60}")
    print(f"Variant {vname}: {vconfig['desc']}")
    print(f"{'='*60}")

    pv, trades = run_backtest(vconfig['selector'], vconfig.get('exit'))
    metrics = compute_metrics(pv, trades)
    regime = regime_analysis(pv)

    print(f"  Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, Return={metrics['total_return']}%")
    print(f"  MaxDD={metrics['max_drawdown']}%, WR={metrics['win_rate']}%, PF={metrics['profit_factor']}")
    print(f"  Final=${metrics['final_value']} from ${metrics['start_value']}, Trades={metrics['trades']}")
    print(f"  Bull Sharpe={regime['bull_sharpe']}, Bear Sharpe={regime['bear_sharpe']}, Gap={regime['regime_gap']}")

    print(f"  Permutation test ({N_PERMS} shuffles, vectorized)...")
    perm = fast_permutation_test(vconfig['selector'], metrics['sharpe'])
    print(f"  Perm p={perm['p_value']}, Monthly Sharpe={perm['actual_monthly_sharpe']}, Perm mean={perm['perm_mean_sharpe']}")

    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm['p_value'] < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown'] > -50,
        'trades_gte_20': metrics['trades'] >= 20
    }
    passed = sum(gates.values())
    print(f"  Gates: {passed}/5 — " + ", ".join(f"{'OK' if v else 'FAIL'}:{k}" for k, v in gates.items()))

    results[vname] = {
        'description': vconfig['desc'],
        'metrics': metrics, 'regime': regime, 'permutation': perm,
        'validation_gates': {k: bool(v) for k, v in gates.items()},
        'gates_passed': f"{passed}/5",
        'verdict': 'VIABLE' if passed >= 4 else ('MARGINAL' if passed >= 3 else 'REJECT')
    }

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SECTOR ROTATION TIMING OVERLAY — SUMMARY")
print(f"{'='*70}")
header = f"{'Variant':<25} {'Sharpe':>7} {'Sort':>7} {'Ret%':>7} {'MaxDD':>7} {'PF':>6} {'WR':>6} {'Gap':>6} {'Pp':>7} {'Gates':>6} {'Verdict':<10}"
print(header)
print("-" * len(header))

for vname, r in results.items():
    m, rg, p = r['metrics'], r['regime'], r['permutation']
    print(f"{vname:<25} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['total_return']:>6.1f}% {m['max_drawdown']:>6.1f}% {m['profit_factor']:>6.2f} {m['win_rate']:>5.1f}% {rg['regime_gap']:>6.3f} {p['p_value']:>7.4f} {r['gates_passed']:>6} {r['verdict']:<10}")

best = max(results.items(), key=lambda x: sum(x[1]['validation_gates'].values()) * 10 + x[1]['metrics']['sharpe'])
print(f"\nBest: {best[0]} — {best[1]['verdict']} ({best[1]['description']})")

# ── Save ────────────────────────────────────────────────────────────────────
output_path = Path('/home/jupiter/Lvl3Quant/data/sector_rotation_timing_results.json')

def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

results['_meta'] = {
    'run_date': datetime.now().isoformat(),
    'oot_period': f'{OOT_START} to {OOT_END}',
    'initial_capital': INITIAL_CAPITAL,
    'sectors': SECTORS,
    'n_permutations': N_PERMS,
    'slippage_pct': SLIPPAGE_PCT,
    'best_variant': best[0]
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=convert)

print(f"\nResults saved. Done.")
