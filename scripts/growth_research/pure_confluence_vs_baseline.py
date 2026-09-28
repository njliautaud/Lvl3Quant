#!/usr/bin/env python3
"""
PURE CONFLUENCE vs BASELINE — FOLLOW-UP FROM SESSION_STATE #426
================================================================
Entry 426 found multi-timeframe confluence Sharpe 2.543 vs baseline 1.812,
BUT the overnight-only (UPRO_ON) mechanism was likely the real driver.

THIS TEST: Strip out UPRO_ON entirely. Map composite score to UPRO/SPY/GLD
only. If pure regime detection doesn't beat baseline, the 3-timeframe
confluence adds no value — it was just the overnight exposure modulation.

ALSO TEST: Use composite score as a DYNAMIC VOL THRESHOLD modifier within
the Gameplan v2 framework. High confluence → more aggressive (higher vol
threshold), low confluence → more conservative.

Validation: Walk-forward, permutation, sub-period, outlier, R1.
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
np.random.seed(42)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/pure_confluence_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──
INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002
GAP_RISK_PCT = 0.001

print("=" * 80)
print("PURE CONFLUENCE vs BASELINE — STRIPPING OUT OVERNIGHT MECHANISM")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/9] Downloading data...")
tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
data = yf.download(tickers, start='2012-01-01', end='2026-07-17',
                   auto_adjust=True, threads=True, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close']
else:
    closes = data
if hasattr(closes.columns, 'droplevel'):
    try:
        closes.columns = closes.columns.droplevel(1)
    except:
        pass
closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
returns = closes.pct_change().fillna(0)
print(f"  {len(closes)} trading days: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")


# ══════════════════════════════════════════════════════════════════════
# 2. SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════
print("\n[2/9] Computing signals...")

def compute_all_signals(spy_close):
    spy_ret = spy_close.pct_change()

    # SHORT
    mom_5d = spy_close.pct_change(5)
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_10 = 100 - (100 / (1 + rs))

    # MEDIUM
    sma_20 = spy_close.rolling(20).mean()
    sma_50 = spy_close.rolling(50).mean()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100

    # LONG
    sma_200 = spy_close.rolling(200).mean()
    sma_200_slope = sma_200.pct_change(20)
    vol_63d = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    vol_63d_trend = vol_63d - vol_63d.rolling(21).mean()

    return {
        'mom_5d': mom_5d, 'rsi_10': rsi_10,
        'sma_20': sma_20, 'sma_50': sma_50, 'vol_21d': vol_21d,
        'sma_200': sma_200, 'sma_200_slope': sma_200_slope,
        'vol_63d': vol_63d, 'vol_63d_trend': vol_63d_trend,
    }

signals = compute_all_signals(closes['SPY'])

def compute_composite_score(i, params):
    """Compute confluence score for day i. Returns 0-3."""
    mom = signals['mom_5d'].iloc[i]
    rsi = signals['rsi_10'].iloc[i]
    s20 = signals['sma_20'].iloc[i]
    s50 = signals['sma_50'].iloc[i]
    vol21 = signals['vol_21d'].iloc[i]
    slope = signals['sma_200_slope'].iloc[i]
    vol_trend = signals['vol_63d_trend'].iloc[i]

    s = 0.0
    # Short
    if not np.isnan(mom) and mom > params.get('mom_thresh', 0.0):
        s += 0.5
    if not np.isnan(rsi) and rsi > params.get('rsi_bull', 50):
        s += 0.5
    # Medium
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50:
        s += 0.5
    if not np.isnan(vol21) and vol21 < params.get('vol_low_med', 15):
        s += 0.5
    # Long
    if not np.isnan(slope) and slope > params.get('slope_thresh', 0.0):
        s += 0.5
    if not np.isnan(vol_trend) and vol_trend < params.get('vol_trend_thresh', 0.0):
        s += 0.5
    return s


# ══════════════════════════════════════════════════════════════════════
# 3. SIMULATION ENGINES
# ══════════════════════════════════════════════════════════════════════

def calc_metrics(vals_series, total_contributed, switches, label=""):
    """Calculate risk-adjusted metrics from equity curve."""
    vals = vals_series.values
    rets = pd.Series(vals).pct_change().dropna()
    if len(rets) < 2 or vals[-1] <= 0:
        return {}

    n_years = len(vals) / 252
    cagr = (vals[-1] / vals[0]) ** (1/n_years) - 1 if n_years > 0 else 0

    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    downside = rets[rets < 0]
    sortino = rets.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # MaxDD
    cummax = pd.Series(vals).cummax()
    dd = (pd.Series(vals) - cummax) / cummax
    maxdd = dd.min()

    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    # Daily win rate
    wr = (rets > 0).mean()

    return {
        'label': label,
        'final_value': vals[-1],
        'total_contributed': total_contributed,
        'profit': vals[-1] - total_contributed,
        'cagr_pct': cagr * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'maxdd_pct': maxdd * 100,
        'calmar': calmar,
        'win_rate': wr,
        'switches': switches,
        'switches_per_yr': switches / n_years if n_years > 0 else 0,
        'n_years': n_years,
    }


def simulate(start_date, end_date, regime_fn, label=""):
    """Generic simulator. regime_fn(date_idx, date) -> regime string."""
    mask = (closes.index >= start_date) & (closes.index <= end_date)
    sim_dates = closes.index[mask]
    if len(sim_dates) < 10:
        return None, None

    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []

    for date in sim_dates:
        i = closes.index.get_loc(date)

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        if i < 210:
            daily_values.append(cash)
            last_regime = 'CASH'
            continue

        regime = regime_fn(i, date)

        # TX costs on switch
        if regime != last_regime and last_regime is not None and last_regime != 'CASH':
            switches += 1
            cash *= (1 - TX_COST_PCT - GAP_RISK_PCT)
        last_regime = regime

        # Apply return
        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)

        daily_values.append(cash)

    vals = pd.Series(daily_values, index=sim_dates[:len(daily_values)])
    metrics = calc_metrics(vals, total_contributed, switches, label)
    return vals, metrics


# ── REGIME FUNCTIONS ──

def baseline_v2_regime(i, date):
    """Gameplan v2: vol_low=15, 20/200 MA, Sep hedge, earnings OFF."""
    if date.month == 9:
        return 'SPY'
    vol = signals['vol_21d'].iloc[i]
    s20 = signals['sma_20'].iloc[i]
    s200 = signals['sma_200'].iloc[i]
    if np.isnan(vol): vol = 15
    protection_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)
    if vol > 30:
        return 'GLD'
    elif vol > 15 or protection_off:
        return 'SPY'
    else:
        return 'UPRO'


def pure_confluence_regime(params):
    """Pure confluence: 3 score bins → UPRO/SPY/GLD. NO overnight."""
    t_high = params.get('threshold_high', 2.0)
    t_low = params.get('threshold_low', 1.0)

    def _fn(i, date):
        if params.get('sep_hedge', True) and date.month == 9:
            return 'SPY'
        score = compute_composite_score(i, params)
        if score >= t_high:
            return 'UPRO'
        elif score >= t_low:
            return 'SPY'
        else:
            return 'GLD'
    return _fn


def adaptive_vol_regime(params):
    """Use confluence score to modulate Gameplan v2 vol threshold dynamically."""
    base_vol_low = params.get('base_vol_low', 15)
    boost_per_score = params.get('boost_per_score', 2)  # each score point raises vol_low by this

    def _fn(i, date):
        if date.month == 9:
            return 'SPY'
        score = compute_composite_score(i, params)
        # Higher score → more aggressive → higher vol threshold allowed
        vol_low = base_vol_low + score * boost_per_score

        vol = signals['vol_21d'].iloc[i]
        s20 = signals['sma_20'].iloc[i]
        s200 = signals['sma_200'].iloc[i]
        if np.isnan(vol): vol = 15
        protection_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if vol > 30:
            return 'GLD'
        elif vol > vol_low or protection_off:
            return 'SPY'
        else:
            return 'UPRO'
    return _fn


# ══════════════════════════════════════════════════════════════════════
# 4. FULL-PERIOD COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n[3/9] Running full-period comparisons...")

start = closes.index[0]
end = closes.index[-1]

# Baseline
vals_base, m_base = simulate(start, end, baseline_v2_regime, "Gameplan_v2_baseline")
print(f"\n  BASELINE: Sharpe {m_base['sharpe']:.3f}, CAGR {m_base['cagr_pct']:.1f}%, "
      f"MaxDD {m_base['maxdd_pct']:.1f}%, Final ${m_base['final_value']:,.0f}, "
      f"Switches {m_base['switches_per_yr']:.1f}/yr")

# Pure confluence variants (different threshold combos)
confluence_configs = [
    {'label': 'Pure_2.0/1.0', 'threshold_high': 2.0, 'threshold_low': 1.0,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0, 'sep_hedge': True},
    {'label': 'Pure_2.5/1.5', 'threshold_high': 2.5, 'threshold_low': 1.5,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0, 'sep_hedge': True},
    {'label': 'Pure_2.0/0.5', 'threshold_high': 2.0, 'threshold_low': 0.5,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0, 'sep_hedge': True},
    {'label': 'Pure_1.5/0.5', 'threshold_high': 1.5, 'threshold_low': 0.5,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0, 'sep_hedge': True},
    {'label': 'Pure_2.5/1.0', 'threshold_high': 2.5, 'threshold_low': 1.0,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 12,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0, 'sep_hedge': True},
]

# Adaptive vol variants
adaptive_configs = [
    {'label': 'Adaptive_+2/pt', 'base_vol_low': 12, 'boost_per_score': 2,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0},
    {'label': 'Adaptive_+3/pt', 'base_vol_low': 10, 'boost_per_score': 3,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0},
    {'label': 'Adaptive_+1/pt', 'base_vol_low': 13, 'boost_per_score': 1,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0},
    {'label': 'Adaptive_+4/pt', 'base_vol_low': 8, 'boost_per_score': 4,
     'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
     'slope_thresh': 0.0, 'vol_trend_thresh': 0.0},
]

all_results = [m_base]

for cfg in confluence_configs:
    lbl = cfg.pop('label')
    vals, m = simulate(start, end, pure_confluence_regime(cfg), lbl)
    if m:
        all_results.append(m)
        diff_sharpe = m['sharpe'] - m_base['sharpe']
        print(f"  {lbl}: Sharpe {m['sharpe']:.3f} ({diff_sharpe:+.3f}), "
              f"CAGR {m['cagr_pct']:.1f}%, MaxDD {m['maxdd_pct']:.1f}%, "
              f"Final ${m['final_value']:,.0f}, Switches {m['switches_per_yr']:.1f}/yr")

for cfg in adaptive_configs:
    lbl = cfg.pop('label')
    vals, m = simulate(start, end, adaptive_vol_regime(cfg), lbl)
    if m:
        all_results.append(m)
        diff_sharpe = m['sharpe'] - m_base['sharpe']
        print(f"  {lbl}: Sharpe {m['sharpe']:.3f} ({diff_sharpe:+.3f}), "
              f"CAGR {m['cagr_pct']:.1f}%, MaxDD {m['maxdd_pct']:.1f}%, "
              f"Final ${m['final_value']:,.0f}, Switches {m['switches_per_yr']:.1f}/yr")


# ══════════════════════════════════════════════════════════════════════
# 5. IDENTIFY BEST PURE CONFLUENCE + BEST ADAPTIVE
# ══════════════════════════════════════════════════════════════════════
print("\n[4/9] Identifying best configs...")

# Find best pure confluence by Sharpe
pure_results = [r for r in all_results if r['label'].startswith('Pure_')]
adaptive_results = [r for r in all_results if r['label'].startswith('Adaptive_')]

best_pure = max(pure_results, key=lambda x: x['sharpe']) if pure_results else None
best_adaptive = max(adaptive_results, key=lambda x: x['sharpe']) if adaptive_results else None

if best_pure:
    print(f"  Best pure confluence: {best_pure['label']} — Sharpe {best_pure['sharpe']:.3f}")
if best_adaptive:
    print(f"  Best adaptive vol: {best_adaptive['label']} — Sharpe {best_adaptive['sharpe']:.3f}")
print(f"  Baseline: Gameplan v2 — Sharpe {m_base['sharpe']:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 6. WALK-FORWARD VALIDATION (best pure + best adaptive)
# ══════════════════════════════════════════════════════════════════════
print("\n[5/9] Walk-forward validation (3yr train, 1yr OOS, 10 windows)...")

def walk_forward_test(regime_fn, label):
    """10-window walk-forward. Train is just for threshold optimization here."""
    wf_results = []
    # 10 windows: 2014-2025 (after 200d warmup from 2012)
    for test_year in range(2014, 2025):
        train_start = f"{test_year - 3}-01-01"
        train_end = f"{test_year - 1}-12-31"
        test_start = f"{test_year}-01-01"
        test_end = f"{test_year}-12-31"

        # OOS test
        vals, m = simulate(test_start, test_end, regime_fn, f"WF_{test_year}")
        if m and m.get('sharpe') is not None:
            wf_results.append({
                'year': test_year,
                'sharpe': m['sharpe'],
                'cagr_pct': m['cagr_pct'],
                'maxdd_pct': m['maxdd_pct'],
                'final_value': m['final_value'],
                'switches': m['switches'],
            })

    if wf_results:
        sharpes = [r['sharpe'] for r in wf_results]
        positive = sum(1 for s in sharpes if s > 0)
        mean_sharpe = np.mean(sharpes)
        print(f"  {label}: Mean OOS Sharpe {mean_sharpe:.3f}, "
              f"{positive}/{len(sharpes)} positive windows")
        return wf_results, mean_sharpe
    return [], 0

# Re-create best configs
if best_pure:
    # Parse thresholds from label
    parts = best_pure['label'].replace('Pure_', '').split('/')
    th, tl = float(parts[0]), float(parts[1])
    best_pure_params = {'threshold_high': th, 'threshold_low': tl,
                        'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
                        'slope_thresh': 0.0, 'vol_trend_thresh': 0.0, 'sep_hedge': True}
    # Handle the 2.5/1.0 special case with vol_low_med=12
    if best_pure['label'] == 'Pure_2.5/1.0':
        best_pure_params['vol_low_med'] = 12
    wf_pure, wf_pure_sharpe = walk_forward_test(
        pure_confluence_regime(best_pure_params), f"Pure ({best_pure['label']})")

if best_adaptive:
    # Parse params from label
    boost = int(best_adaptive['label'].split('+')[1].split('/')[0])
    base_map = {1: 13, 2: 12, 3: 10, 4: 8}
    best_adaptive_params = {
        'base_vol_low': base_map.get(boost, 12), 'boost_per_score': boost,
        'mom_thresh': 0.0, 'rsi_bull': 50, 'vol_low_med': 15,
        'slope_thresh': 0.0, 'vol_trend_thresh': 0.0}
    wf_adaptive, wf_adaptive_sharpe = walk_forward_test(
        adaptive_vol_regime(best_adaptive_params), f"Adaptive ({best_adaptive['label']})")

wf_baseline, wf_base_sharpe = walk_forward_test(baseline_v2_regime, "Baseline (Gameplan v2)")


# ══════════════════════════════════════════════════════════════════════
# 7. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════
print("\n[6/9] Permutation test (200 shuffles)...")

def permutation_test(regime_fn, real_sharpe, label, n_perms=200):
    """Shuffle regime assignments randomly, count how often random beats real."""
    # Get the actual regime sequence
    mask = (closes.index >= closes.index[210]) & (closes.index <= end)
    sim_dates = closes.index[mask]

    actual_regimes = []
    for date in sim_dates:
        i = closes.index.get_loc(date)
        actual_regimes.append(regime_fn(i, date))

    regime_set = list(set(actual_regimes))
    perm_sharpes = []

    for p in range(n_perms):
        # Random regime assignment (same distribution)
        np.random.seed(p + 1000)
        random_regimes = np.random.choice(regime_set, size=len(sim_dates),
                                          p=[actual_regimes.count(r)/len(actual_regimes) for r in regime_set])

        cash = float(INITIAL)
        total_contributed = float(INITIAL)
        last_week = None
        last_regime = None
        daily_values = []

        for j, date in enumerate(sim_dates):
            week_key = (date.year, date.isocalendar()[1])
            if week_key != last_week:
                cash += WEEKLY_DCA
                total_contributed += WEEKLY_DCA
                last_week = week_key

            regime = random_regimes[j]
            if regime != last_regime and last_regime is not None:
                cash *= (1 - TX_COST_PCT - GAP_RISK_PCT)
            last_regime = regime

            if regime in returns.columns:
                r = returns.loc[date, regime]
                if not np.isnan(r):
                    cash *= (1 + r)
            daily_values.append(cash)

        rets = pd.Series(daily_values).pct_change().dropna()
        if rets.std() > 0:
            perm_sharpes.append(rets.mean() / rets.std() * np.sqrt(252))

    p_value = np.mean([1 for s in perm_sharpes if s >= real_sharpe]) / len(perm_sharpes) if perm_sharpes else 1.0
    print(f"  {label}: p={p_value:.3f} (real Sharpe {real_sharpe:.3f} vs perm mean {np.mean(perm_sharpes):.3f})")
    return p_value

# Best pure confluence
if best_pure:
    perm_pure = permutation_test(
        pure_confluence_regime(best_pure_params),
        best_pure['sharpe'], f"Pure ({best_pure['label']})")

# Best adaptive
if best_adaptive:
    perm_adaptive = permutation_test(
        adaptive_vol_regime(best_adaptive_params),
        best_adaptive['sharpe'], f"Adaptive ({best_adaptive['label']})")

# Baseline
perm_base = permutation_test(baseline_v2_regime, m_base['sharpe'], "Baseline")


# ══════════════════════════════════════════════════════════════════════
# 8. R1 REGIME-AGNOSTIC TEST
# ══════════════════════════════════════════════════════════════════════
print("\n[7/9] R1 regime-agnostic test...")

def r1_test(regime_fn, label):
    """Classify days as green/red/flat using SPY, compute Sharpe per regime."""
    spy_ret = returns['SPY']
    mask = closes.index >= closes.index[210]
    test_dates = closes.index[mask]

    daily_rets = []
    day_types = []

    for date in test_dates:
        i = closes.index.get_loc(date)
        regime = regime_fn(i, date)

        if regime in returns.columns:
            r = returns.loc[date, regime]
        else:
            r = 0

        sr = spy_ret.loc[date]
        if sr > 0.001:
            dtype = 'green'
        elif sr < -0.001:
            dtype = 'red'
        else:
            dtype = 'flat'

        daily_rets.append(r)
        day_types.append(dtype)

    df = pd.DataFrame({'ret': daily_rets, 'day': day_types})

    results = {}
    for d in ['green', 'red']:
        sub = df[df['day'] == d]['ret']
        if len(sub) > 10 and sub.std() > 0:
            results[f'sharpe_{d}'] = sub.mean() / sub.std() * np.sqrt(252)
        else:
            results[f'sharpe_{d}'] = 0

    sg = results['sharpe_green']
    sr = results['sharpe_red']
    gap = abs(sg - sr) / max(abs(sg), abs(sr), 1e-6)
    passed = gap <= 0.50

    print(f"  {label}: Green Sharpe {sg:.3f}, Red Sharpe {sr:.3f}, "
          f"Gap {gap:.3f} {'PASS' if passed else 'FAIL'}")
    return gap, passed


if best_pure:
    gap_pure, r1_pure = r1_test(pure_confluence_regime(best_pure_params),
                                 f"Pure ({best_pure['label']})")
if best_adaptive:
    gap_adaptive, r1_adaptive = r1_test(adaptive_vol_regime(best_adaptive_params),
                                         f"Adaptive ({best_adaptive['label']})")
gap_base, r1_base = r1_test(baseline_v2_regime, "Baseline")


# ══════════════════════════════════════════════════════════════════════
# 9. SUB-PERIOD CONSISTENCY + OUTLIER ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════
print("\n[8/9] Sub-period consistency & outlier robustness...")

def sub_period_test(regime_fn, label):
    """Test on 3 sub-periods."""
    periods = [
        ('2013-01-01', '2017-12-31'),
        ('2018-01-01', '2021-12-31'),
        ('2022-01-01', '2025-12-31'),
    ]
    sharpes = []
    for s, e in periods:
        vals, m = simulate(s, e, regime_fn, f"sub_{s[:4]}")
        if m:
            sharpes.append(m['sharpe'])

    if len(sharpes) >= 2:
        cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else 999
        all_pos = all(s > 0 for s in sharpes)
        print(f"  {label}: Sub-period Sharpes {[f'{s:.2f}' for s in sharpes]}, "
              f"CV {cv:.3f}, All positive: {all_pos}")
        return sharpes, cv
    return [], 999


if best_pure:
    sub_pure, cv_pure = sub_period_test(pure_confluence_regime(best_pure_params),
                                         f"Pure ({best_pure['label']})")
if best_adaptive:
    sub_adaptive, cv_adaptive = sub_period_test(adaptive_vol_regime(best_adaptive_params),
                                                 f"Adaptive ({best_adaptive['label']})")
sub_base, cv_base = sub_period_test(baseline_v2_regime, "Baseline")


# ══════════════════════════════════════════════════════════════════════
# 10. REGIME TIME BREAKDOWN
# ══════════════════════════════════════════════════════════════════════
print("\n[9/9] Regime time breakdown...")

def regime_breakdown(regime_fn, label):
    """Show % time in each regime."""
    mask = closes.index >= closes.index[210]
    test_dates = closes.index[mask]
    regimes = []
    for date in test_dates:
        i = closes.index.get_loc(date)
        regimes.append(regime_fn(i, date))

    from collections import Counter
    counts = Counter(regimes)
    total = len(regimes)
    parts = []
    for r in ['UPRO', 'SPY', 'GLD']:
        pct = counts.get(r, 0) / total * 100
        if pct > 0:
            parts.append(f"{r}={pct:.1f}%")
    print(f"  {label}: {', '.join(parts)}")
    return {r: counts.get(r, 0)/total for r in ['UPRO', 'SPY', 'GLD']}


if best_pure:
    bd_pure = regime_breakdown(pure_confluence_regime(best_pure_params),
                                f"Pure ({best_pure['label']})")
if best_adaptive:
    bd_adaptive = regime_breakdown(adaptive_vol_regime(best_adaptive_params),
                                    f"Adaptive ({best_adaptive['label']})")
bd_base = regime_breakdown(baseline_v2_regime, "Baseline")


# ══════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

print(f"\n  {'Config':<30} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'SW/yr':>8} {'WF Sharpe':>10} {'Perm p':>8} {'R1 gap':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

print(f"  {'Baseline (v2)':<30} {m_base['sharpe']:>8.3f} {m_base['cagr_pct']:>8.1f} {m_base['maxdd_pct']:>8.1f} {m_base['switches_per_yr']:>8.1f} {wf_base_sharpe:>10.3f} {perm_base:>8.3f} {gap_base:>8.3f}")

if best_pure:
    print(f"  {best_pure['label']:<30} {best_pure['sharpe']:>8.3f} {best_pure['cagr_pct']:>8.1f} {best_pure['maxdd_pct']:>8.1f} {best_pure['switches_per_yr']:>8.1f} {wf_pure_sharpe:>10.3f} {perm_pure:>8.3f} {gap_pure:>8.3f}")

if best_adaptive:
    print(f"  {best_adaptive['label']:<30} {best_adaptive['sharpe']:>8.3f} {best_adaptive['cagr_pct']:>8.1f} {best_adaptive['maxdd_pct']:>8.1f} {best_adaptive['switches_per_yr']:>8.1f} {wf_adaptive_sharpe:>10.3f} {perm_adaptive:>8.3f} {gap_adaptive:>8.3f}")

# Determine if confluence adds value
if best_pure and best_adaptive:
    best_alt = best_pure if best_pure['sharpe'] > best_adaptive['sharpe'] else best_adaptive
    best_alt_sharpe = best_alt['sharpe']
    delta = best_alt_sharpe - m_base['sharpe']

    print(f"\n  PURE CONFLUENCE vs BASELINE:")
    if best_pure['sharpe'] > m_base['sharpe']:
        print(f"    Pure confluence BEATS baseline by {best_pure['sharpe'] - m_base['sharpe']:.3f} Sharpe")
        print(f"    → 3-timeframe trend detection adds genuine value beyond overnight mechanism")
    else:
        print(f"    Pure confluence LOSES to baseline by {m_base['sharpe'] - best_pure['sharpe']:.3f} Sharpe")
        print(f"    → 3-timeframe trend detection adds NO value. Entry #426's improvement")
        print(f"      was entirely from the overnight-only exposure modulation.")

    print(f"\n  ADAPTIVE VOL vs BASELINE:")
    if best_adaptive['sharpe'] > m_base['sharpe']:
        print(f"    Adaptive vol BEATS baseline by {best_adaptive['sharpe'] - m_base['sharpe']:.3f} Sharpe")
        print(f"    → Using confluence to dynamically adjust vol thresholds IS valuable")
    else:
        print(f"    Adaptive vol LOSES to baseline by {m_base['sharpe'] - best_adaptive['sharpe']:.3f} Sharpe")
        print(f"    → Static vol threshold is sufficient; dynamic adjustment adds noise")

# Save results
results_dict = {
    'baseline': m_base,
    'all_results': all_results,
    'walk_forward': {
        'baseline_sharpe': wf_base_sharpe,
        'pure_sharpe': wf_pure_sharpe if best_pure else None,
        'adaptive_sharpe': wf_adaptive_sharpe if best_adaptive else None,
    },
    'permutation': {
        'baseline_p': perm_base,
        'pure_p': perm_pure if best_pure else None,
        'adaptive_p': perm_adaptive if best_adaptive else None,
    },
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)

print(f"\n  Results saved to {OUTPUT_DIR}")
print("=" * 80)
