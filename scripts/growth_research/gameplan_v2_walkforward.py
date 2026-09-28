#!/usr/bin/env python3
"""
Gameplan v2 Walk-Forward Validation with Realistic Costs — HC #708 Entry 424
=============================================================================
Gold-standard overfitting detection for the vol-adjusted UPRO system.

Method:
  - 3-year rolling training window, 1-year out-of-sample test, slide 1 year
  - In each training window: grid-search vol thresholds + MA periods
  - Apply best params to OOS year (NO peeking)
  - Concatenate all OOS returns for aggregate metrics
  - Compare WF metrics vs fixed-param backtest (entry 418)
  - Permutation test on WF results
  - Realistic costs: 0.02% per switch (UPRO bid-ask spread)
  - Model overnight gap risk on regime switches

Walk-forward windows (with data from 2012):
  Train 2013-2015 → Test 2016
  Train 2014-2016 → Test 2017
  ...
  Train 2022-2024 → Test 2025
  (2012 used for warmup/indicators only)

Checks:
  1. WF vs fixed-param comparison (overfitting ratio)
  2. Parameter stability across windows
  3. Permutation test (1000 shuffles)
  4. Sub-period consistency
  5. Overnight gap risk on switches
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
    "C:/Users/claude/Lvl3Quant/output/growth_research/gameplan_v2_walkforward"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──
INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002   # 0.02% UPRO bid-ask spread per switch
GAP_RISK_PCT = 0.001   # 0.1% overnight gap risk modeled per switch

# ── PARAMETER GRID ──
VOL_LOW_GRID = [15, 17, 20, 22, 25]           # Low vol threshold (%)
VOL_HIGH_GRID = [25, 28, 30, 33, 35, 40]      # High vol threshold (%)
MA_SHORT_GRID = [10, 15, 20, 30]              # Short MA period
MA_LONG_GRID = [100, 150, 200, 250]           # Long MA period
MA_TYPE_GRID = ['SMA', 'EMA']                 # MA type
SEP_HEDGE_GRID = [True, False]                # September hedge
EARNINGS_AGGR_GRID = [True, False]            # Earnings aggression

print("=" * 80)
print("GAMEPLAN v2 WALK-FORWARD VALIDATION — HC #708 Entry 424")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/7] Downloading data...")

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
# 2. CORE SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════

def compute_ma(series, period, ma_type='SMA'):
    """Compute moving average (SMA or EMA)."""
    if ma_type == 'EMA':
        return series.ewm(span=period, adjust=False).mean()
    else:
        return series.rolling(period).mean()


def get_regime(vol_pct, ma_short_val, ma_long_val, date,
               vol_low=20, vol_high=30, sep_hedge=True, earnings_aggr=True):
    """Determine allocation regime given parameters."""
    # September hedge
    if sep_hedge and date.month == 9:
        return 'SPY'

    # Earnings aggression: raise vol_low by 5
    is_earnings = False
    if earnings_aggr:
        m, d = date.month, date.day
        is_earnings = ((m == 1 and d >= 15) or (m == 2 and d <= 15) or
                      (m == 4 and d >= 15) or (m == 5 and d <= 15) or
                      (m == 7 and d >= 15) or (m == 8 and d <= 15) or
                      (m == 10 and d >= 15) or (m == 11 and d <= 15))

    effective_vol_low = vol_low + 5 if is_earnings else vol_low

    if np.isnan(vol_pct):
        vol_pct = 15

    # Protection overlay: short MA < long MA
    protection_off = (not np.isnan(ma_short_val) and not np.isnan(ma_long_val)
                     and ma_short_val < ma_long_val)

    if vol_pct > vol_high:
        return 'GLD'
    elif vol_pct > effective_vol_low or protection_off:
        return 'SPY'
    else:
        return 'UPRO'


def simulate(start_date, end_date, params, initial=None, starting_cash=None):
    """
    Simulate the Gameplan v2 system over a date range with given parameters.
    Returns daily portfolio values and metadata.
    """
    vol_low = params['vol_low']
    vol_high = params['vol_high']
    ma_short = params['ma_short']
    ma_long = params['ma_long']
    ma_type = params['ma_type']
    sep_hedge = params['sep_hedge']
    earnings_aggr = params['earnings_aggr']
    tx_cost = params.get('tx_cost', TX_COST_PCT)
    gap_risk = params.get('gap_risk', GAP_RISK_PCT)

    spy = closes['SPY']

    # Compute indicators using ALL available history up to end_date (no look-ahead)
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100  # as percentage
    ma_short_vals = compute_ma(spy, ma_short, ma_type)
    ma_long_vals = compute_ma(spy, ma_long, ma_type)

    # Filter to simulation period
    mask = (closes.index >= start_date) & (closes.index <= end_date)
    sim_dates = closes.index[mask]

    if len(sim_dates) == 0:
        return None, None, None

    cash = starting_cash if starting_cash is not None else (initial if initial is not None else INITIAL)
    total_contributed = cash
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []
    daily_regimes = []
    daily_dates = []

    for date in sim_dates:
        i = closes.index.get_loc(date)

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Skip if not enough warmup for longest MA
        if i < ma_long + 5:
            daily_values.append(cash)
            daily_regimes.append('CASH')
            daily_dates.append(date)
            continue

        # Get regime
        regime = get_regime(
            vol_21d.iloc[i],
            ma_short_vals.iloc[i],
            ma_long_vals.iloc[i],
            date,
            vol_low=vol_low, vol_high=vol_high,
            sep_hedge=sep_hedge, earnings_aggr=earnings_aggr
        )

        # Transaction costs on switches
        if regime != last_regime and last_regime is not None and last_regime != 'CASH':
            switches += 1
            cash *= (1 - tx_cost)     # bid-ask spread
            cash *= (1 - gap_risk)    # overnight gap risk

        last_regime = regime

        # Apply return
        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)
        daily_dates.append(date)

    vals = pd.Series(daily_values, index=daily_dates)
    regs = pd.Series(daily_regimes, index=daily_dates)
    return vals, total_contributed, switches


def compute_metrics(values, total_contributed=None):
    """Compute risk-adjusted metrics."""
    if values is None or len(values) < 10:
        return {'sharpe': -999, 'cagr': -999, 'max_dd': -1, 'sortino': -999,
                'calmar': -999, 'final_value': 0}

    daily_ret = values.pct_change().dropna()
    if len(daily_ret) == 0:
        return {'sharpe': -999, 'cagr': -999, 'max_dd': -1, 'sortino': -999,
                'calmar': -999, 'final_value': 0}

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 1
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min()

    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    result = {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'calmar': round(calmar, 4),
        'final_value': round(values.iloc[-1], 2),
        'ann_vol': round(ann_vol, 4)
    }
    if total_contributed:
        result['total_contributed'] = round(total_contributed, 2)
        result['profit'] = round(values.iloc[-1] - total_contributed, 2)
    return result


# ══════════════════════════════════════════════════════════════════════
# 3. WALK-FORWARD ENGINE
# ══════════════════════════════════════════════════════════════════════
print("\n[2/7] Running walk-forward optimization...")

def generate_param_combos():
    """Generate parameter grid (pruned for speed)."""
    combos = []
    for vl, vh, ms, ml, mt, sh, ea in product(
        VOL_LOW_GRID, VOL_HIGH_GRID, MA_SHORT_GRID, MA_LONG_GRID,
        MA_TYPE_GRID, SEP_HEDGE_GRID, EARNINGS_AGGR_GRID
    ):
        # Prune invalid: vol_low must be < vol_high
        if vl >= vh:
            continue
        # Prune: short MA must be < long MA
        if ms >= ml:
            continue
        combos.append({
            'vol_low': vl, 'vol_high': vh,
            'ma_short': ms, 'ma_long': ml,
            'ma_type': mt, 'sep_hedge': sh, 'earnings_aggr': ea
        })
    return combos

param_combos = generate_param_combos()
print(f"  Parameter grid: {len(param_combos)} combinations")

# Define walk-forward windows
# Need warmup (250 days for SMA200), so first usable year is ~2013
wf_windows = []
for test_start_year in range(2016, 2026):  # Test years 2016-2025
    train_start = pd.Timestamp(f'{test_start_year - 3}-01-01')
    train_end = pd.Timestamp(f'{test_start_year - 1}-12-31')
    test_start = pd.Timestamp(f'{test_start_year}-01-01')
    test_end = pd.Timestamp(f'{test_start_year}-12-31')
    # Clip to available data
    if test_end > closes.index[-1]:
        test_end = closes.index[-1]
    if test_start > closes.index[-1]:
        continue
    wf_windows.append({
        'train_start': train_start, 'train_end': train_end,
        'test_start': test_start, 'test_end': test_end,
        'label': f'Train {test_start_year-3}-{test_start_year-1} → Test {test_start_year}'
    })

print(f"  Walk-forward windows: {len(wf_windows)}")
for w in wf_windows:
    print(f"    {w['label']}")

# Run walk-forward
wf_results = []
best_params_per_window = []
oos_daily_returns = []

for wi, window in enumerate(wf_windows):
    print(f"\n  Window {wi+1}/{len(wf_windows)}: {window['label']}")

    # ── TRAINING PHASE: grid search on train period ──
    best_sharpe = -999
    best_params = None
    train_results = []

    # Subsample if grid is too large (>1000)
    combos_to_test = param_combos
    if len(combos_to_test) > 800:
        # Random subsample + always include default params
        np.random.seed(42 + wi)
        indices = np.random.choice(len(combos_to_test),
                                   min(800, len(combos_to_test)), replace=False)
        combos_to_test = [param_combos[i] for i in indices]
        # Always include the "default" Gameplan v2 params
        default_params = {'vol_low': 20, 'vol_high': 30, 'ma_short': 20,
                         'ma_long': 200, 'ma_type': 'SMA', 'sep_hedge': True,
                         'earnings_aggr': True}
        if default_params not in combos_to_test:
            combos_to_test.append(default_params)

    for params in combos_to_test:
        vals, contrib, switches = simulate(
            window['train_start'], window['train_end'], params)
        if vals is None:
            continue
        metrics = compute_metrics(vals, contrib)
        s = metrics['sharpe']
        train_results.append((params, s, metrics))
        if s > best_sharpe:
            best_sharpe = s
            best_params = params.copy()

    if best_params is None:
        print(f"    WARNING: No valid params found, using defaults")
        best_params = {'vol_low': 20, 'vol_high': 30, 'ma_short': 20,
                      'ma_long': 200, 'ma_type': 'SMA', 'sep_hedge': True,
                      'earnings_aggr': True}

    # Record param stats
    all_sharpes = sorted([s for _, s, _ in train_results if s > -900], reverse=True)
    print(f"    Best train Sharpe: {best_sharpe:.3f} | "
          f"Params: vol={best_params['vol_low']}/{best_params['vol_high']}, "
          f"MA={best_params['ma_type']}{best_params['ma_short']}/{best_params['ma_long']}, "
          f"Sep={best_params['sep_hedge']}, Earn={best_params['earnings_aggr']}")
    if len(all_sharpes) >= 10:
        print(f"    Train Sharpe range: [{all_sharpes[-1]:.3f}, {all_sharpes[0]:.3f}], "
              f"median={all_sharpes[len(all_sharpes)//2]:.3f}")

    # ── OOS PHASE: apply best params to test period ──
    oos_vals, oos_contrib, oos_switches = simulate(
        window['test_start'], window['test_end'], best_params)

    if oos_vals is not None and len(oos_vals) > 5:
        oos_metrics = compute_metrics(oos_vals, oos_contrib)
        oos_ret = oos_vals.pct_change().dropna()
        oos_daily_returns.append(oos_ret)

        print(f"    OOS Sharpe: {oos_metrics['sharpe']:.3f} | "
              f"CAGR: {oos_metrics['cagr']*100:.1f}% | "
              f"MaxDD: {oos_metrics['max_dd']*100:.1f}% | "
              f"Switches: {oos_switches}")
    else:
        oos_metrics = {'sharpe': 0, 'cagr': 0, 'max_dd': 0}
        print(f"    OOS: insufficient data")

    wf_results.append({
        'window': window['label'],
        'train_sharpe': round(best_sharpe, 4),
        'oos_sharpe': oos_metrics['sharpe'],
        'oos_cagr': oos_metrics.get('cagr', 0),
        'oos_max_dd': oos_metrics.get('max_dd', 0),
        'oos_switches': oos_switches if oos_vals is not None else 0,
        'best_params': best_params
    })
    best_params_per_window.append(best_params.copy())

# ══════════════════════════════════════════════════════════════════════
# 4. AGGREGATE WF OOS METRICS
# ══════════════════════════════════════════════════════════════════════
print("\n[3/7] Computing aggregate walk-forward metrics...")

if oos_daily_returns:
    all_oos_rets = pd.concat(oos_daily_returns).sort_index()
    # Remove any duplicates (overlapping days shouldn't happen but safety)
    all_oos_rets = all_oos_rets[~all_oos_rets.index.duplicated(keep='first')]

    wf_sharpe = all_oos_rets.mean() / all_oos_rets.std() * np.sqrt(252) if all_oos_rets.std() > 0 else 0
    wf_cagr_ann = all_oos_rets.mean() * 252
    neg = all_oos_rets[all_oos_rets < 0]
    wf_sortino = all_oos_rets.mean() * 252 / (neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0

    # Reconstruct equity curve
    wf_equity = (1 + all_oos_rets).cumprod()
    wf_dd = (wf_equity - wf_equity.cummax()) / wf_equity.cummax()
    wf_max_dd = wf_dd.min()

    years = (all_oos_rets.index[-1] - all_oos_rets.index[0]).days / 365.25
    wf_cagr = wf_equity.iloc[-1] ** (1/years) - 1 if years > 0 else 0

    print(f"\n  WALK-FORWARD AGGREGATE OOS METRICS:")
    print(f"    Sharpe:  {wf_sharpe:.3f}")
    print(f"    Sortino: {wf_sortino:.3f}")
    print(f"    CAGR:    {wf_cagr*100:.1f}%")
    print(f"    MaxDD:   {wf_max_dd*100:.1f}%")
    print(f"    OOS days: {len(all_oos_rets)}")
    print(f"    OOS period: {all_oos_rets.index[0].strftime('%Y-%m-%d')} to {all_oos_rets.index[-1].strftime('%Y-%m-%d')}")
else:
    wf_sharpe = wf_sortino = wf_cagr = 0
    wf_max_dd = -1
    all_oos_rets = pd.Series()

# ══════════════════════════════════════════════════════════════════════
# 5. FIXED-PARAM BASELINE (for comparison)
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] Running fixed-param baseline (Gameplan v2 defaults)...")

fixed_params = {
    'vol_low': 20, 'vol_high': 30,
    'ma_short': 20, 'ma_long': 200,
    'ma_type': 'SMA',
    'sep_hedge': True, 'earnings_aggr': True,
    'tx_cost': TX_COST_PCT, 'gap_risk': GAP_RISK_PCT
}

# Run fixed params over full period
fixed_vals, fixed_contrib, fixed_switches = simulate(
    closes.index[0], closes.index[-1], fixed_params)
fixed_metrics = compute_metrics(fixed_vals, fixed_contrib) if fixed_vals is not None else {}

# Also run fixed params over JUST the OOS periods for fair comparison
if len(all_oos_rets) > 0:
    oos_start = all_oos_rets.index[0]
    oos_end = all_oos_rets.index[-1]
    fixed_oos_vals, fixed_oos_contrib, fixed_oos_switches = simulate(
        oos_start, oos_end, fixed_params)
    if fixed_oos_vals is not None:
        fixed_oos_ret = fixed_oos_vals.pct_change().dropna()
        fixed_oos_sharpe = fixed_oos_ret.mean() / fixed_oos_ret.std() * np.sqrt(252) if fixed_oos_ret.std() > 0 else 0
        fixed_oos_equity = (1 + fixed_oos_ret).cumprod()
        fixed_oos_dd = (fixed_oos_equity - fixed_oos_equity.cummax()) / fixed_oos_equity.cummax()
        fixed_oos_max_dd = fixed_oos_dd.min()
        years_oos = (fixed_oos_ret.index[-1] - fixed_oos_ret.index[0]).days / 365.25
        fixed_oos_cagr = fixed_oos_equity.iloc[-1] ** (1/years_oos) - 1 if years_oos > 0 else 0
    else:
        fixed_oos_sharpe = fixed_oos_cagr = 0
        fixed_oos_max_dd = -1

print(f"\n  FIXED-PARAM BASELINE (full period):")
print(f"    Sharpe:  {fixed_metrics.get('sharpe', 0):.3f}")
print(f"    CAGR:    {fixed_metrics.get('cagr', 0)*100:.1f}%")
print(f"    MaxDD:   {fixed_metrics.get('max_dd', 0)*100:.1f}%")
print(f"    Switches: {fixed_switches}")
print(f"    Final:   ${fixed_metrics.get('final_value', 0):,.0f}")

if len(all_oos_rets) > 0:
    print(f"\n  FIXED-PARAM BASELINE (OOS period only, for fair comparison):")
    print(f"    Sharpe:  {fixed_oos_sharpe:.3f}")
    print(f"    CAGR:    {fixed_oos_cagr*100:.1f}%")
    print(f"    MaxDD:   {fixed_oos_max_dd*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# 6. OVERFITTING ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Overfitting analysis...")

# Overfitting ratio: WF OOS Sharpe / Fixed OOS Sharpe
# >1.0 = WF is better (good, params adapted well)
# ~1.0 = no overfitting (params are stable/robust)
# <0.7 = overfitting concern (optimized params degrade OOS)

if len(all_oos_rets) > 0 and fixed_oos_sharpe != 0:
    overfit_ratio = wf_sharpe / fixed_oos_sharpe
else:
    overfit_ratio = 1.0

print(f"\n  OVERFITTING DIAGNOSTICS:")
print(f"    WF OOS Sharpe:     {wf_sharpe:.3f}")
print(f"    Fixed OOS Sharpe:  {fixed_oos_sharpe:.3f}")
print(f"    Overfitting ratio: {overfit_ratio:.3f}")

if overfit_ratio >= 0.9:
    overfit_verdict = "NO OVERFITTING — walk-forward performs comparably to fixed params"
elif overfit_ratio >= 0.7:
    overfit_verdict = "MILD OVERFITTING — some degradation but still profitable"
elif overfit_ratio >= 0.5:
    overfit_verdict = "MODERATE OVERFITTING — significant OOS degradation, caution"
else:
    overfit_verdict = "SEVERE OVERFITTING — optimized params fail out-of-sample"

print(f"    Verdict: {overfit_verdict}")

# Train-to-test Sharpe degradation per window
print(f"\n  PER-WINDOW TRAIN→OOS DEGRADATION:")
degradations = []
for wr in wf_results:
    if wr['train_sharpe'] > 0:
        deg = wr['oos_sharpe'] / wr['train_sharpe']
        degradations.append(deg)
    else:
        deg = 1.0
        degradations.append(deg)
    print(f"    {wr['window']}: Train {wr['train_sharpe']:.3f} → OOS {wr['oos_sharpe']:.3f} "
          f"(ratio {deg:.2f})")

avg_degradation = np.mean(degradations) if degradations else 1.0
print(f"\n  Mean train→OOS ratio: {avg_degradation:.3f} "
      f"({'healthy' if avg_degradation > 0.5 else 'OVERFITTING CONCERN'})")

# ══════════════════════════════════════════════════════════════════════
# 7. PARAMETER STABILITY
# ══════════════════════════════════════════════════════════════════════
print("\n[6/7] Parameter stability across windows...")

param_names = ['vol_low', 'vol_high', 'ma_short', 'ma_long', 'ma_type',
               'sep_hedge', 'earnings_aggr']

print(f"\n  BEST PARAMS PER WINDOW:")
print(f"  {'Window':<35} {'VolLow':>6} {'VolHigh':>7} {'MAshort':>7} {'MAlong':>7} {'Type':>4} {'Sep':>4} {'Earn':>5}")
for wi, bp in enumerate(best_params_per_window):
    lbl = wf_results[wi]['window']
    print(f"  {lbl:<35} {bp['vol_low']:>6} {bp['vol_high']:>7} {bp['ma_short']:>7} "
          f"{bp['ma_long']:>7} {bp['ma_type']:>4} {str(bp['sep_hedge']):>4} {str(bp['earnings_aggr']):>5}")

# Stability metrics for numeric params
for pname in ['vol_low', 'vol_high', 'ma_short', 'ma_long']:
    vals_p = [bp[pname] for bp in best_params_per_window]
    cv = np.std(vals_p) / np.mean(vals_p) if np.mean(vals_p) > 0 else 0
    print(f"\n  {pname}: mean={np.mean(vals_p):.1f}, std={np.std(vals_p):.1f}, "
          f"CV={cv:.2f}, range=[{min(vals_p)}, {max(vals_p)}]")

# Boolean params: count how often each choice is selected
for pname in ['sep_hedge', 'earnings_aggr', 'ma_type']:
    vals_p = [bp[pname] for bp in best_params_per_window]
    from collections import Counter
    cnt = Counter(vals_p)
    print(f"  {pname}: {dict(cnt)}")

# ══════════════════════════════════════════════════════════════════════
# 8. PERMUTATION TEST ON WF OOS
# ══════════════════════════════════════════════════════════════════════
print("\n[7/7] Permutation test on walk-forward OOS returns...")

N_PERMS = 1000
if len(all_oos_rets) > 10:
    real_sharpe = wf_sharpe
    perm_sharpes = []

    for p in range(N_PERMS):
        shuffled = all_oos_rets.values.copy()
        # Shuffle the SIGN of returns (keeps magnitude distribution)
        signs = np.random.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    perm_p = np.mean(perm_sharpes >= real_sharpe)

    print(f"\n  PERMUTATION TEST (sign-shuffle, {N_PERMS} permutations):")
    print(f"    Real WF OOS Sharpe:  {real_sharpe:.3f}")
    print(f"    Permutation mean:    {perm_sharpes.mean():.3f}")
    print(f"    Permutation p95:     {np.percentile(perm_sharpes, 95):.3f}")
    print(f"    p-value:             {perm_p:.4f}")
    print(f"    Verdict:             {'SIGNIFICANT (p < 0.05)' if perm_p < 0.05 else 'NOT SIGNIFICANT'}")
else:
    perm_p = 1.0
    print("  Insufficient OOS data for permutation test")

# ══════════════════════════════════════════════════════════════════════
# 9. OVERNIGHT GAP RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OVERNIGHT GAP RISK ON REGIME SWITCHES")
print("=" * 60)

# Measure actual overnight gaps in UPRO on switch days
spy = closes['SPY']
upro = closes['UPRO']
spy_ret = spy.pct_change()
vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100
sma20 = compute_ma(spy, 20, 'SMA')
sma200 = compute_ma(spy, 200, 'SMA')

# Find switch days using fixed params
regimes_fixed = []
for i in range(len(closes)):
    if i < 250:
        regimes_fixed.append('WARMUP')
        continue
    date = closes.index[i]
    r = get_regime(vol_21d.iloc[i], sma20.iloc[i], sma200.iloc[i], date)
    regimes_fixed.append(r)

switch_days = []
for i in range(1, len(regimes_fixed)):
    if regimes_fixed[i] != regimes_fixed[i-1] and regimes_fixed[i-1] != 'WARMUP':
        switch_days.append(i)

# Overnight gaps on switch days vs normal days
if len(switch_days) > 5:
    upro_overnight = (closes['UPRO'].shift(-1).values / closes['UPRO'].values) - 1  # rough proxy
    # Actually compute open gaps: open[t+1] / close[t] - 1
    # Since we only have close data, approximate with next-day return contribution
    normal_daily_ret = returns['UPRO'].abs()
    switch_abs_rets = [normal_daily_ret.iloc[i] for i in switch_days if i < len(normal_daily_ret)]
    normal_abs_rets = [normal_daily_ret.iloc[i] for i in range(250, len(normal_daily_ret)) if i not in switch_days]

    print(f"\n  Switch days: {len(switch_days)} over {len(closes)/252:.1f} years ({len(switch_days)/(len(closes)/252):.1f}/yr)")
    if switch_abs_rets and normal_abs_rets:
        print(f"  Avg |return| on switch days: {np.mean(switch_abs_rets)*100:.3f}%")
        print(f"  Avg |return| on normal days: {np.mean(normal_abs_rets)*100:.3f}%")
        print(f"  Ratio: {np.mean(switch_abs_rets)/np.mean(normal_abs_rets):.2f}x")
        print(f"  P95 |return| on switch days: {np.percentile(switch_abs_rets, 95)*100:.3f}%")
        print(f"  Our gap_risk model ({GAP_RISK_PCT*100:.2f}%) vs actual switch-day cost")

# ══════════════════════════════════════════════════════════════════════
# 10. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY — GAMEPLAN v2 WALK-FORWARD VALIDATION")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ WALK-FORWARD OOS (optimized per window)                        │
│   Sharpe:  {wf_sharpe:>7.3f}                                          │
│   CAGR:    {wf_cagr*100:>6.1f}%                                          │
│   MaxDD:   {wf_max_dd*100:>6.1f}%                                          │
│   Sortino: {wf_sortino:>7.3f}                                          │
│                                                                │
│ FIXED-PARAM BASELINE (same OOS period)                         │
│   Sharpe:  {fixed_oos_sharpe:>7.3f}                                          │
│   CAGR:    {fixed_oos_cagr*100:>6.1f}%                                          │
│   MaxDD:   {fixed_oos_max_dd*100:>6.1f}%                                          │
│                                                                │
│ OVERFITTING ANALYSIS                                           │
│   WF/Fixed Sharpe ratio: {overfit_ratio:>5.3f}                            │
│   Train→OOS degradation: {avg_degradation:>5.3f}                            │
│   Permutation p-value:   {perm_p:>5.4f}                            │
│                                                                │
│ VERDICT: {overfit_verdict:<53s} │
│ Permutation: {'REAL EDGE (p<0.05)' if perm_p < 0.05 else 'NOT SIGNIFICANT':>20s}                             │
└─────────────────────────────────────────────────────────────────┘
""")

# Determine if system is trustworthy
trust_score = 0
trust_reasons = []
if overfit_ratio >= 0.8:
    trust_score += 1
    trust_reasons.append("No significant overfitting (ratio >= 0.8)")
elif overfit_ratio >= 0.5:
    trust_reasons.append(f"Some overfitting (ratio {overfit_ratio:.2f})")
else:
    trust_reasons.append(f"OVERFITTING CONCERN (ratio {overfit_ratio:.2f})")

if perm_p < 0.05:
    trust_score += 1
    trust_reasons.append(f"Statistically significant edge (p={perm_p:.4f})")
else:
    trust_reasons.append(f"Edge NOT significant (p={perm_p:.4f})")

if avg_degradation > 0.4:
    trust_score += 1
    trust_reasons.append(f"Healthy train→OOS retention ({avg_degradation:.2f})")
else:
    trust_reasons.append(f"Poor train→OOS retention ({avg_degradation:.2f})")

# Check if WF OOS Sharpe is positive
if wf_sharpe > 0.5:
    trust_score += 1
    trust_reasons.append(f"Strong OOS Sharpe ({wf_sharpe:.3f})")
elif wf_sharpe > 0:
    trust_reasons.append(f"Weak OOS Sharpe ({wf_sharpe:.3f})")
else:
    trust_reasons.append(f"NEGATIVE OOS Sharpe ({wf_sharpe:.3f})")

# Parameter stability
vol_low_cv = np.std([bp['vol_low'] for bp in best_params_per_window]) / np.mean([bp['vol_low'] for bp in best_params_per_window]) if best_params_per_window else 1
if vol_low_cv < 0.3:
    trust_score += 1
    trust_reasons.append(f"Stable parameters across windows (vol_low CV={vol_low_cv:.2f})")
else:
    trust_reasons.append(f"Unstable parameters (vol_low CV={vol_low_cv:.2f})")

print(f"  TRUST SCORE: {trust_score}/5")
for r in trust_reasons:
    print(f"    {'✓' if 'NOT' not in r and 'CONCERN' not in r and 'Poor' not in r and 'Weak' not in r and 'Unstable' not in r and 'NEGATIVE' not in r else '✗'} {r}")

# Overall recommendation
if trust_score >= 4:
    recommendation = "HIGH CONFIDENCE — System is robust, safe to deploy with real money"
elif trust_score >= 3:
    recommendation = "MODERATE CONFIDENCE — System works but has some fragility"
elif trust_score >= 2:
    recommendation = "LOW CONFIDENCE — Significant overfitting or instability detected"
else:
    recommendation = "DO NOT DEPLOY — System fails walk-forward validation"

print(f"\n  RECOMMENDATION: {recommendation}")

# ══════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
results_out = {
    'study': 'Gameplan v2 Walk-Forward Validation',
    'hc_entry': 424,
    'timestamp': dt.datetime.now().isoformat(),
    'walk_forward': {
        'windows': len(wf_windows),
        'train_years': 3,
        'test_years': 1,
        'param_grid_size': len(param_combos),
        'tx_cost_pct': TX_COST_PCT,
        'gap_risk_pct': GAP_RISK_PCT
    },
    'wf_oos_metrics': {
        'sharpe': round(wf_sharpe, 4),
        'sortino': round(wf_sortino, 4),
        'cagr': round(wf_cagr, 4),
        'max_dd': round(wf_max_dd, 4),
        'oos_days': len(all_oos_rets)
    },
    'fixed_baseline_metrics': {
        'sharpe_full': fixed_metrics.get('sharpe', 0),
        'sharpe_oos_period': round(fixed_oos_sharpe, 4) if len(all_oos_rets) > 0 else 0,
        'cagr_oos_period': round(fixed_oos_cagr, 4) if len(all_oos_rets) > 0 else 0,
        'max_dd_oos_period': round(fixed_oos_max_dd, 4) if len(all_oos_rets) > 0 else 0
    },
    'overfitting': {
        'wf_vs_fixed_ratio': round(overfit_ratio, 4),
        'train_to_oos_degradation': round(avg_degradation, 4),
        'verdict': overfit_verdict
    },
    'permutation': {
        'n_perms': N_PERMS,
        'p_value': round(perm_p, 4),
        'significant': perm_p < 0.05
    },
    'per_window_results': wf_results,
    'best_params_per_window': [{k: str(v) if isinstance(v, bool) else v
                                for k, v in bp.items()}
                               for bp in best_params_per_window],
    'trust_score': trust_score,
    'trust_max': 5,
    'trust_reasons': trust_reasons,
    'recommendation': recommendation
}

# Serialize params that might have numpy types
def serialize(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")

results_path = OUTPUT_DIR / 'walkforward_results.json'
with open(results_path, 'w') as f:
    json.dump(results_out, f, indent=2, default=serialize)

print(f"\nResults saved to {results_path}")
print("DONE.")
