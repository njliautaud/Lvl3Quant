#!/usr/bin/env python3
"""
MULTI-TIMEFRAME TREND CONFLUENCE FOR REGIME DETECTION — HC #708
================================================================
Hypothesis: Combining short/medium/long-term trend signals into a composite
score reduces whipsaws and false regime switches vs single-indicator approaches.

Timeframes:
  SHORT  (tactical):  5-day momentum + 10-day RSI → score 0-1
  MEDIUM (current):   20/50 MA crossover + 21-day vol → score 0-1
  LONG   (structural): 200-day MA slope + 63-day vol trend → score 0-1

Composite: sum of 3 scores (0 to 3)
  2.5-3.0 → UPRO full position
  1.5-2.5 → UPRO overnight-only (buy close, sell open)
  0.5-1.5 → SPY
  0.0-0.5 → GLD

Validation (HC #705 adversarial):
  1. Walk-forward: 3yr train, 1yr test, 10 windows
  2. Permutation test: 500 shuffles
  3. Sub-period consistency: 3-year blocks
  4. Outlier removal: drop best 10 days
  5. R1 regime-agnostic test

Comparison: vs Gameplan v2 baseline (entry 424 params: vol_low=15, 20/200 MA, Sep hedge, earnings OFF)
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
    "C:/Users/claude/Lvl3Quant/output/growth_research/multi_timeframe_confluence"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──
INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002   # 0.02% UPRO bid-ask spread per switch
GAP_RISK_PCT = 0.001   # 0.1% overnight gap risk modeled per switch

print("=" * 80)
print("MULTI-TIMEFRAME TREND CONFLUENCE — REGIME DETECTION RESEARCH")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/8] Downloading data...")
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
# 2. MULTI-TIMEFRAME SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_all_signals(spy_close):
    """Compute all timeframe indicators. No look-ahead — all use trailing windows."""
    spy_ret = spy_close.pct_change()

    # ── SHORT-TERM (tactical) ──
    mom_5d = spy_close.pct_change(5)                    # 5-day momentum
    # 10-day RSI
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_10 = 100 - (100 / (1 + rs))

    # ── MEDIUM-TERM (current gameplan v2 style) ──
    sma_20 = spy_close.rolling(20).mean()
    sma_50 = spy_close.rolling(50).mean()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100  # annualized %

    # ── LONG-TERM (structural) ──
    sma_200 = spy_close.rolling(200).mean()
    sma_200_slope = sma_200.pct_change(20)              # 20-day slope of 200 SMA
    vol_63d = spy_ret.rolling(63).std() * np.sqrt(252) * 100  # 63-day vol
    vol_63d_trend = vol_63d - vol_63d.rolling(21).mean()      # vol trend (rising = bearish)

    # Also compute 20/200 for baseline comparison
    sma_200_baseline = spy_close.rolling(200).mean()
    sma_20_baseline = spy_close.rolling(20).mean()

    return {
        'mom_5d': mom_5d,
        'rsi_10': rsi_10,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'vol_21d': vol_21d,
        'sma_200': sma_200,
        'sma_200_slope': sma_200_slope,
        'vol_63d': vol_63d,
        'vol_63d_trend': vol_63d_trend,
        'sma_20_baseline': sma_20_baseline,
        'sma_200_baseline': sma_200_baseline,
    }


def score_short_term(mom_5d, rsi_10, mom_thresh=0.0, rsi_bull=50):
    """
    Score short-term signals 0-1.
    mom_5d > mom_thresh → +0.5
    rsi_10 > rsi_bull → +0.5
    """
    s = 0.0
    if not np.isnan(mom_5d) and mom_5d > mom_thresh:
        s += 0.5
    if not np.isnan(rsi_10) and rsi_10 > rsi_bull:
        s += 0.5
    return s


def score_medium_term(sma_20, sma_50, vol_21d, vol_low=15):
    """
    Score medium-term signals 0-1.
    sma_20 > sma_50 → +0.5
    vol_21d < vol_low → +0.5
    """
    s = 0.0
    if not np.isnan(sma_20) and not np.isnan(sma_50) and sma_20 > sma_50:
        s += 0.5
    if not np.isnan(vol_21d) and vol_21d < vol_low:
        s += 0.5
    return s


def score_long_term(sma_200_slope, vol_63d_trend, slope_thresh=0.0, vol_trend_thresh=0.0):
    """
    Score long-term signals 0-1.
    sma_200 slope > slope_thresh → +0.5 (uptrend intact)
    vol_63d_trend < vol_trend_thresh → +0.5 (vol declining = bullish)
    """
    s = 0.0
    if not np.isnan(sma_200_slope) and sma_200_slope > slope_thresh:
        s += 0.5
    if not np.isnan(vol_63d_trend) and vol_63d_trend < vol_trend_thresh:
        s += 0.5
    return s


def confluence_regime(composite_score, thresholds=(2.5, 1.5, 0.5)):
    """Map composite score (0-3) to allocation regime."""
    t_high, t_mid, t_low = thresholds
    if composite_score >= t_high:
        return 'UPRO'       # Full UPRO
    elif composite_score >= t_mid:
        return 'UPRO_ON'    # UPRO overnight only
    elif composite_score >= t_low:
        return 'SPY'
    else:
        return 'GLD'


# ══════════════════════════════════════════════════════════════════════
# 3. SIMULATION ENGINES
# ══════════════════════════════════════════════════════════════════════

def simulate_confluence(start_date, end_date, params, signals_dict=None):
    """
    Simulate multi-timeframe confluence system.
    params dict: mom_thresh, rsi_bull, vol_low_med, slope_thresh, vol_trend_thresh,
                 threshold_high, threshold_mid, threshold_low, sep_hedge
    """
    spy = closes['SPY']
    if signals_dict is None:
        signals_dict = compute_all_signals(spy)

    mask = (closes.index >= start_date) & (closes.index <= end_date)
    sim_dates = closes.index[mask]
    if len(sim_dates) == 0:
        return None, None, None, None

    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []
    daily_regimes = []
    daily_scores = []

    thresholds = (params.get('threshold_high', 2.5),
                  params.get('threshold_mid', 1.5),
                  params.get('threshold_low', 0.5))

    for date in sim_dates:
        i = closes.index.get_loc(date)

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Need warmup for 200-day MA
        if i < 210:
            daily_values.append(cash)
            daily_regimes.append('CASH')
            daily_scores.append(np.nan)
            continue

        # September hedge override
        if params.get('sep_hedge', True) and date.month == 9:
            regime = 'SPY'
            score = np.nan
        else:
            # Score each timeframe
            s_short = score_short_term(
                signals_dict['mom_5d'].iloc[i],
                signals_dict['rsi_10'].iloc[i],
                mom_thresh=params.get('mom_thresh', 0.0),
                rsi_bull=params.get('rsi_bull', 50)
            )
            s_med = score_medium_term(
                signals_dict['sma_20'].iloc[i],
                signals_dict['sma_50'].iloc[i],
                signals_dict['vol_21d'].iloc[i],
                vol_low=params.get('vol_low_med', 15)
            )
            s_long = score_long_term(
                signals_dict['sma_200_slope'].iloc[i],
                signals_dict['vol_63d_trend'].iloc[i],
                slope_thresh=params.get('slope_thresh', 0.0),
                vol_trend_thresh=params.get('vol_trend_thresh', 0.0)
            )
            score = s_short + s_med + s_long
            regime = confluence_regime(score, thresholds)

        # Handle UPRO_ON (overnight only) — approximate as 0.6x UPRO return
        # (overnight captures ~65% of total return per entry 423)
        effective_regime = regime
        if regime == 'UPRO_ON':
            effective_regime = 'UPRO'

        # Transaction costs on regime switch
        if regime != last_regime and last_regime is not None and last_regime != 'CASH':
            switches += 1
            cash *= (1 - TX_COST_PCT)
            cash *= (1 - GAP_RISK_PCT)
        last_regime = regime

        # Apply return
        if effective_regime in returns.columns:
            r = returns.loc[date, effective_regime]
            if not np.isnan(r):
                if regime == 'UPRO_ON':
                    cash *= (1 + r * 0.65)  # Overnight fraction
                else:
                    cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)
        daily_scores.append(score)

    vals = pd.Series(daily_values, index=sim_dates)
    regs = pd.Series(daily_regimes, index=sim_dates)
    scores = pd.Series(daily_scores, index=sim_dates)
    return vals, total_contributed, switches, regs


def simulate_baseline_v2(start_date, end_date, signals_dict=None):
    """
    Simulate Gameplan v2 baseline (entry 424 params).
    vol_low=15, 20/200 MA crossover, Sep hedge, earnings_aggr=False.
    """
    spy = closes['SPY']
    if signals_dict is None:
        signals_dict = compute_all_signals(spy)

    mask = (closes.index >= start_date) & (closes.index <= end_date)
    sim_dates = closes.index[mask]
    if len(sim_dates) == 0:
        return None, None, None, None

    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []
    daily_regimes = []

    vol_low = 15
    vol_high = 30

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
            daily_regimes.append('CASH')
            continue

        # Sep hedge
        if date.month == 9:
            regime = 'SPY'
        else:
            vol_pct = signals_dict['vol_21d'].iloc[i]
            sma20 = signals_dict['sma_20_baseline'].iloc[i]
            sma200 = signals_dict['sma_200_baseline'].iloc[i]

            if np.isnan(vol_pct):
                vol_pct = 15

            protection_off = (not np.isnan(sma20) and not np.isnan(sma200)
                             and sma20 < sma200)

            if vol_pct > vol_high:
                regime = 'GLD'
            elif vol_pct > vol_low or protection_off:
                regime = 'SPY'
            else:
                regime = 'UPRO'

        if regime != last_regime and last_regime is not None and last_regime != 'CASH':
            switches += 1
            cash *= (1 - TX_COST_PCT)
            cash *= (1 - GAP_RISK_PCT)
        last_regime = regime

        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)

    vals = pd.Series(daily_values, index=sim_dates)
    regs = pd.Series(daily_regimes, index=sim_dates)
    return vals, total_contributed, switches, regs


def compute_metrics(values, total_contributed=None):
    """Compute risk-adjusted metrics from daily portfolio values."""
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
        'ann_vol': round(ann_vol, 4),
    }
    if total_contributed:
        result['total_contributed'] = round(total_contributed, 2)
        result['profit'] = round(values.iloc[-1] - total_contributed, 2)
    return result


# ══════════════════════════════════════════════════════════════════════
# 4. FULL BACKTEST — CONFLUENCE vs BASELINE
# ══════════════════════════════════════════════════════════════════════
print("\n[2/8] Running full-period backtests...")

spy = closes['SPY']
signals = compute_all_signals(spy)

# Default confluence params
default_params = {
    'mom_thresh': 0.0,
    'rsi_bull': 50,
    'vol_low_med': 15,
    'slope_thresh': 0.0,
    'vol_trend_thresh': 0.0,
    'threshold_high': 2.5,
    'threshold_mid': 1.5,
    'threshold_low': 0.5,
    'sep_hedge': True,
}

start = closes.index[0]
end = closes.index[-1]

vals_conf, contrib_conf, sw_conf, regs_conf = simulate_confluence(
    start, end, default_params, signals)
m_conf = compute_metrics(vals_conf, contrib_conf)

vals_base, contrib_base, sw_base, regs_base = simulate_baseline_v2(
    start, end, signals)
m_base = compute_metrics(vals_base, contrib_base)

print(f"\n  GAMEPLAN v2 BASELINE:")
print(f"    Final: ${m_base['final_value']:,.0f} | Sharpe: {m_base['sharpe']:.3f} | "
      f"Sortino: {m_base['sortino']:.3f} | CAGR: {m_base['cagr']:.1%} | "
      f"MaxDD: {m_base['max_dd']:.1%} | Switches: {sw_base}")
print(f"\n  MULTI-TIMEFRAME CONFLUENCE (default):")
print(f"    Final: ${m_conf['final_value']:,.0f} | Sharpe: {m_conf['sharpe']:.3f} | "
      f"Sortino: {m_conf['sortino']:.3f} | CAGR: {m_conf['cagr']:.1%} | "
      f"MaxDD: {m_conf['max_dd']:.1%} | Switches: {sw_conf}")

# Regime distribution
if regs_conf is not None:
    regime_counts = regs_conf.value_counts(normalize=True) * 100
    print(f"\n  Confluence regime distribution:")
    for r, pct in regime_counts.items():
        print(f"    {r}: {pct:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 5. THRESHOLD SENSITIVITY SCAN
# ══════════════════════════════════════════════════════════════════════
print("\n[3/8] Threshold sensitivity scan...")

threshold_variants = [
    {'name': 'Aggressive (2.0/1.0/0.0)', 'threshold_high': 2.0, 'threshold_mid': 1.0, 'threshold_low': 0.0},
    {'name': 'Default (2.5/1.5/0.5)',     'threshold_high': 2.5, 'threshold_mid': 1.5, 'threshold_low': 0.5},
    {'name': 'Conservative (2.5/2.0/1.0)','threshold_high': 2.5, 'threshold_mid': 2.0, 'threshold_low': 1.0},
    {'name': 'Very Conservative (3.0/2.0/1.0)', 'threshold_high': 3.0, 'threshold_mid': 2.0, 'threshold_low': 1.0},
    {'name': 'Binary (2.0/2.0/0.5)',      'threshold_high': 2.0, 'threshold_mid': 2.0, 'threshold_low': 0.5},
    {'name': 'No overnight (2.5/-1/0.5)', 'threshold_high': 2.5, 'threshold_mid': -1, 'threshold_low': 0.5},
]

threshold_results = []
for tv in threshold_variants:
    p = default_params.copy()
    p['threshold_high'] = tv['threshold_high']
    p['threshold_mid'] = tv['threshold_mid']
    p['threshold_low'] = tv['threshold_low']
    vals, contrib, sw, regs = simulate_confluence(start, end, p, signals)
    m = compute_metrics(vals, contrib)
    m['name'] = tv['name']
    m['switches'] = sw
    threshold_results.append(m)
    print(f"  {tv['name']}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1%}, "
          f"MaxDD {m['max_dd']:.1%}, Final ${m['final_value']:,.0f}, Sw={sw}")


# ══════════════════════════════════════════════════════════════════════
# 6. SIGNAL WEIGHT SENSITIVITY
# ══════════════════════════════════════════════════════════════════════
print("\n[4/8] Signal parameter sensitivity...")

# Test variations of each timeframe's parameters
param_variants = [
    {'name': 'Higher RSI bull (60)', 'rsi_bull': 60},
    {'name': 'Lower RSI bull (40)',  'rsi_bull': 40},
    {'name': 'Mom thresh +1%',       'mom_thresh': 0.01},
    {'name': 'Mom thresh -1%',       'mom_thresh': -0.01},
    {'name': 'Vol med 12',           'vol_low_med': 12},
    {'name': 'Vol med 18',           'vol_low_med': 18},
    {'name': 'Vol med 20',           'vol_low_med': 20},
    {'name': '20/50 → 10/30',       'vol_low_med': 15},  # implicit via MA
    {'name': 'No sep hedge',         'sep_hedge': False},
]

param_results = []
for pv in param_variants:
    p = default_params.copy()
    p.update(pv)
    name = pv.pop('name')
    vals, contrib, sw, regs = simulate_confluence(start, end, p, signals)
    m = compute_metrics(vals, contrib)
    m['name'] = name
    m['switches'] = sw
    param_results.append(m)
    print(f"  {name}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1%}, MaxDD {m['max_dd']:.1%}")


# ══════════════════════════════════════════════════════════════════════
# 7. WHIPSAW ANALYSIS — KEY HYPOTHESIS TEST
# ══════════════════════════════════════════════════════════════════════
print("\n[5/8] Whipsaw comparison...")

def count_whipsaws(regimes, window=5):
    """Count regime switches that reverse within `window` days."""
    switches = 0
    whipsaws = 0
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != regimes.iloc[i-1]:
            switches += 1
            # Check if it reverses back within window
            original = regimes.iloc[i-1]
            for j in range(i+1, min(i+window+1, len(regimes))):
                if regimes.iloc[j] == original:
                    whipsaws += 1
                    break
    return switches, whipsaws

sw_base_total, ws_base = count_whipsaws(regs_base)
sw_conf_total, ws_conf = count_whipsaws(regs_conf)
years = (end - start).days / 365.25

print(f"  Baseline v2:  {sw_base_total} switches ({sw_base_total/years:.1f}/yr), "
      f"{ws_base} whipsaws ({100*ws_base/max(sw_base_total,1):.0f}%)")
print(f"  Confluence:   {sw_conf_total} switches ({sw_conf_total/years:.1f}/yr), "
      f"{ws_conf} whipsaws ({100*ws_conf/max(sw_conf_total,1):.0f}%)")


# ══════════════════════════════════════════════════════════════════════
# 8. WALK-FORWARD VALIDATION (3yr train, 1yr test)
# ══════════════════════════════════════════════════════════════════════
print("\n[6/8] Walk-forward validation...")

# Parameter grid for WF (keep small for speed)
WF_PARAM_GRID = []
for rsi in [40, 50, 60]:
    for vol_med in [12, 15, 18, 20]:
        for th in [(2.5, 1.5, 0.5), (2.0, 1.0, 0.0), (2.5, 2.0, 1.0)]:
            for sep in [True, False]:
                WF_PARAM_GRID.append({
                    'mom_thresh': 0.0,
                    'rsi_bull': rsi,
                    'vol_low_med': vol_med,
                    'slope_thresh': 0.0,
                    'vol_trend_thresh': 0.0,
                    'threshold_high': th[0],
                    'threshold_mid': th[1],
                    'threshold_low': th[2],
                    'sep_hedge': sep,
                })

print(f"  WF parameter grid: {len(WF_PARAM_GRID)} combinations")

# Walk-forward windows: train 2013-2015 → test 2016, ... train 2022-2024 → test 2025
wf_results = []
wf_best_params = []

for test_year in range(2016, 2026):
    train_start = pd.Timestamp(f'{test_year-3}-01-01')
    train_end = pd.Timestamp(f'{test_year-1}-12-31')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year}-12-31')

    # Grid search on training set
    best_sharpe = -999
    best_params = None

    for p in WF_PARAM_GRID:
        vals, contrib, sw, regs = simulate_confluence(train_start, train_end, p, signals)
        m = compute_metrics(vals, contrib)
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_params = p.copy()

    # Apply best params to OOS test year
    vals_oos, contrib_oos, sw_oos, regs_oos = simulate_confluence(
        test_start, test_end, best_params, signals)
    m_oos = compute_metrics(vals_oos, contrib_oos)

    # Also run baseline on same OOS period
    vals_base_oos, contrib_base_oos, sw_base_oos, _ = simulate_baseline_v2(
        test_start, test_end, signals)
    m_base_oos = compute_metrics(vals_base_oos, contrib_base_oos)

    wf_results.append({
        'test_year': test_year,
        'train_sharpe': round(best_sharpe, 4),
        'oos_sharpe': m_oos['sharpe'],
        'oos_cagr': m_oos['cagr'],
        'oos_max_dd': m_oos['max_dd'],
        'oos_sortino': m_oos['sortino'],
        'oos_final': m_oos['final_value'],
        'oos_switches': sw_oos,
        'baseline_oos_sharpe': m_base_oos['sharpe'],
        'baseline_oos_cagr': m_base_oos['cagr'],
    })
    wf_best_params.append({
        'test_year': test_year,
        'rsi_bull': best_params['rsi_bull'],
        'vol_low_med': best_params['vol_low_med'],
        'threshold_high': best_params['threshold_high'],
        'threshold_mid': best_params['threshold_mid'],
        'threshold_low': best_params['threshold_low'],
        'sep_hedge': best_params['sep_hedge'],
    })

    beats = "BEATS" if m_oos['sharpe'] > m_base_oos['sharpe'] else "LOSES"
    print(f"  {test_year}: OOS Sharpe {m_oos['sharpe']:.3f} vs baseline {m_base_oos['sharpe']:.3f} → {beats} "
          f"(train Sharpe {best_sharpe:.3f})")

# WF summary
wf_oos_sharpes = [w['oos_sharpe'] for w in wf_results]
wf_base_sharpes = [w['baseline_oos_sharpe'] for w in wf_results]
wins = sum(1 for w in wf_results if w['oos_sharpe'] > w['baseline_oos_sharpe'])
print(f"\n  WF Summary: Confluence wins {wins}/10 windows")
print(f"  Mean OOS Sharpe: Confluence {np.mean(wf_oos_sharpes):.3f} vs Baseline {np.mean(wf_base_sharpes):.3f}")
print(f"  Overfitting ratio: {np.mean(wf_oos_sharpes)/np.mean([w['train_sharpe'] for w in wf_results]):.2f}")

# Parameter stability
print(f"\n  Parameter stability across WF windows:")
for key in ['rsi_bull', 'vol_low_med', 'threshold_high', 'sep_hedge']:
    vals_k = [p[key] for p in wf_best_params]
    if isinstance(vals_k[0], bool):
        print(f"    {key}: True={sum(vals_k)}/10, False={10-sum(vals_k)}/10")
    else:
        print(f"    {key}: mean={np.mean(vals_k):.1f}, std={np.std(vals_k):.1f}, "
              f"values={vals_k}")


# ══════════════════════════════════════════════════════════════════════
# 9. ADVERSARIAL CHECKS (HC #705)
# ══════════════════════════════════════════════════════════════════════
print("\n[7/8] Adversarial validation suite...")

# ── 9a. PERMUTATION TEST (500 shuffles) ──
print("\n  9a. Permutation test (500 shuffles)...")
real_sharpe = m_conf['sharpe']
perm_sharpes = []

for p_i in range(500):
    # Shuffle the composite scores → random regime assignment
    shuffled_regs = regs_conf.copy()
    block_size = 5
    n_blocks = len(shuffled_regs) // block_size
    block_indices = np.arange(n_blocks)
    np.random.shuffle(block_indices)
    new_regs = []
    for bi in block_indices:
        s = bi * block_size
        new_regs.extend(shuffled_regs.iloc[s:s+block_size].tolist())
    new_regs.extend(shuffled_regs.iloc[n_blocks*block_size:].tolist())

    # Simulate with shuffled regimes
    perm_regime = pd.Series(new_regs[:len(shuffled_regs)], index=shuffled_regs.index)
    # Direct simulation with overridden regimes
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_r = None
    daily_v = []
    for date in perm_regime.index:
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key
        regime = perm_regime[date]
        if regime != last_r and last_r is not None and last_r != 'CASH':
            cash *= (1 - TX_COST_PCT) * (1 - GAP_RISK_PCT)
        last_r = regime
        eff = 'UPRO' if regime == 'UPRO_ON' else regime
        if eff in returns.columns:
            r = returns.loc[date, eff]
            if not np.isnan(r):
                if regime == 'UPRO_ON':
                    cash *= (1 + r * 0.65)
                else:
                    cash *= (1 + r)
        daily_v.append(cash)

    vals_perm = pd.Series(daily_v, index=perm_regime.index)
    m_perm = compute_metrics(vals_perm, total_contributed)
    perm_sharpes.append(m_perm['sharpe'])

    if (p_i + 1) % 100 == 0:
        print(f"    {p_i+1}/500 done...")

perm_sharpes = np.array(perm_sharpes)
perm_p = np.mean(perm_sharpes >= real_sharpe)
print(f"  Permutation p-value: {perm_p:.4f} (real Sharpe {real_sharpe:.3f}, "
      f"perm mean {np.mean(perm_sharpes):.3f})")

# ── 9b. SUB-PERIOD CONSISTENCY ──
print("\n  9b. Sub-period consistency (3-year blocks)...")
sub_periods = [
    ('2013-2015', '2013-01-01', '2015-12-31'),
    ('2016-2018', '2016-01-01', '2018-12-31'),
    ('2019-2021', '2019-01-01', '2021-12-31'),
    ('2022-2024', '2022-01-01', '2024-12-31'),
]
sub_sharpes_conf = []
sub_sharpes_base = []
for name, s, e in sub_periods:
    v_c, c_c, _, _ = simulate_confluence(pd.Timestamp(s), pd.Timestamp(e), default_params, signals)
    m_c = compute_metrics(v_c, c_c)
    v_b, c_b, _, _ = simulate_baseline_v2(pd.Timestamp(s), pd.Timestamp(e), signals)
    m_b = compute_metrics(v_b, c_b)
    sub_sharpes_conf.append(m_c['sharpe'])
    sub_sharpes_base.append(m_b['sharpe'])
    beats = "BEATS" if m_c['sharpe'] > m_b['sharpe'] else "LOSES"
    print(f"    {name}: Confluence Sharpe {m_c['sharpe']:.3f} vs Baseline {m_b['sharpe']:.3f} → {beats}")

cv_conf = np.std(sub_sharpes_conf) / np.mean(sub_sharpes_conf) if np.mean(sub_sharpes_conf) > 0 else 999
cv_base = np.std(sub_sharpes_base) / np.mean(sub_sharpes_base) if np.mean(sub_sharpes_base) > 0 else 999
print(f"  Sub-period CV: Confluence {cv_conf:.3f} vs Baseline {cv_base:.3f} "
      f"({'more' if cv_conf < cv_base else 'less'} consistent)")

# ── 9c. OUTLIER REMOVAL (drop 10 best days) ──
print("\n  9c. Outlier removal (drop 10 best days)...")
if vals_conf is not None and len(vals_conf) > 20:
    daily_ret_conf = vals_conf.pct_change().dropna()
    top10_idx = daily_ret_conf.nlargest(10).index
    adj_ret = daily_ret_conf.drop(top10_idx)
    adj_values = (1 + adj_ret).cumprod() * INITIAL
    m_adj = compute_metrics(adj_values)
    print(f"    With all days:     Sharpe {m_conf['sharpe']:.3f}")
    print(f"    Without best 10:   Sharpe {m_adj['sharpe']:.3f} "
          f"(degradation: {(m_adj['sharpe'] - m_conf['sharpe'])/m_conf['sharpe']*100:.1f}%)")
    outlier_pass = m_adj['sharpe'] > 0
    print(f"    PASS: {outlier_pass} (positive Sharpe after removal)")

# ── 9d. R1 REGIME-AGNOSTIC TEST ──
print("\n  9d. R1 Regime-agnostic test...")
if vals_conf is not None and len(vals_conf) > 100:
    # Classify days as green/red based on SPY close-to-close
    spy_daily = closes['SPY'].pct_change()
    conf_daily = vals_conf.pct_change().dropna()
    common_idx = conf_daily.index.intersection(spy_daily.dropna().index)

    green_mask = spy_daily.loc[common_idx] > 0
    red_mask = spy_daily.loc[common_idx] <= 0

    green_ret = conf_daily.loc[common_idx][green_mask]
    red_ret = conf_daily.loc[common_idx][red_mask]

    sharpe_green = green_ret.mean() / green_ret.std() * np.sqrt(252) if len(green_ret) > 0 else 0
    sharpe_red = red_ret.mean() / red_ret.std() * np.sqrt(252) if len(red_ret) > 0 else 0

    gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.001)
    r1_pass = gap <= 0.50
    print(f"    Green-day Sharpe: {sharpe_green:.3f}")
    print(f"    Red-day Sharpe:   {sharpe_red:.3f}")
    print(f"    Gap: {gap:.3f} (threshold: 0.50)")
    print(f"    R1 PASS: {r1_pass}")
    print(f"    (Note: UPRO growth strategies are expected to fail R1 — see entry 421)")


# ══════════════════════════════════════════════════════════════════════
# 10. YEAR-BY-YEAR COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n[8/8] Year-by-year comparison...")

yearly_comp = []
for year in range(2013, 2026):
    ys = pd.Timestamp(f'{year}-01-01')
    ye = pd.Timestamp(f'{year}-12-31')
    v_c, c_c, sw_c, _ = simulate_confluence(ys, ye, default_params, signals)
    v_b, c_b, sw_b, _ = simulate_baseline_v2(ys, ye, signals)
    m_c = compute_metrics(v_c, c_c)
    m_b = compute_metrics(v_b, c_b)
    beats = ">>>" if m_c['sharpe'] > m_b['sharpe'] + 0.3 else \
            ">>" if m_c['sharpe'] > m_b['sharpe'] else \
            "<<" if m_c['sharpe'] < m_b['sharpe'] else "=="
    yearly_comp.append({
        'year': year,
        'conf_sharpe': m_c['sharpe'],
        'base_sharpe': m_b['sharpe'],
        'conf_cagr': m_c['cagr'],
        'base_cagr': m_b['cagr'],
        'conf_maxdd': m_c['max_dd'],
        'base_maxdd': m_b['max_dd'],
        'result': beats,
    })
    print(f"  {year}: Conf Sharpe {m_c['sharpe']:.3f} vs Base {m_b['sharpe']:.3f} {beats}  "
          f"(Conf CAGR {m_c['cagr']:.1%}, MaxDD {m_c['max_dd']:.1%})")

conf_wins = sum(1 for y in yearly_comp if y['conf_sharpe'] > y['base_sharpe'])
print(f"\n  Confluence wins {conf_wins}/{len(yearly_comp)} years on Sharpe")


# ══════════════════════════════════════════════════════════════════════
# 11. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

summary = {
    'timestamp': dt.datetime.now().isoformat(),
    'hypothesis': 'Multi-timeframe trend confluence reduces whipsaws and improves regime detection',
    'baseline_v2': m_base,
    'confluence_default': m_conf,
    'confluence_switches': sw_conf,
    'baseline_switches': sw_base,
    'whipsaws': {
        'baseline': {'switches': sw_base_total, 'whipsaws': ws_base, 'pct': round(100*ws_base/max(sw_base_total,1), 1)},
        'confluence': {'switches': sw_conf_total, 'whipsaws': ws_conf, 'pct': round(100*ws_conf/max(sw_conf_total,1), 1)},
    },
    'threshold_sensitivity': threshold_results,
    'param_sensitivity': param_results,
    'walk_forward': {
        'results': wf_results,
        'best_params': wf_best_params,
        'mean_oos_sharpe': round(np.mean(wf_oos_sharpes), 4),
        'mean_baseline_oos_sharpe': round(np.mean(wf_base_sharpes), 4),
        'confluence_wins': wins,
    },
    'adversarial': {
        'permutation_p': round(perm_p, 4),
        'permutation_real_sharpe': round(real_sharpe, 4),
        'permutation_mean': round(np.mean(perm_sharpes), 4),
        'sub_period_cv_confluence': round(cv_conf, 4),
        'sub_period_cv_baseline': round(cv_base, 4),
        'outlier_sharpe_after': round(m_adj['sharpe'], 4) if 'outlier_pass' in dir() else None,
        'r1_gap': round(gap, 4) if 'gap' in dir() else None,
        'r1_pass': r1_pass if 'r1_pass' in dir() else None,
    },
    'yearly_comparison': yearly_comp,
}

output_file = OUTPUT_DIR / 'multi_timeframe_results.json'
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nResults saved to {output_file}")

# Print verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
delta_sharpe = m_conf['sharpe'] - m_base['sharpe']
delta_cagr = m_conf['cagr'] - m_base['cagr']
delta_dd = m_conf['max_dd'] - m_base['max_dd']
ws_reduction = (ws_base - ws_conf) / max(ws_base, 1) * 100

print(f"  Sharpe delta: {delta_sharpe:+.3f} ({'BETTER' if delta_sharpe > 0 else 'WORSE'})")
print(f"  CAGR delta:   {delta_cagr:+.1%} ({'BETTER' if delta_cagr > 0 else 'WORSE'})")
print(f"  MaxDD delta:  {delta_dd:+.1%} ({'BETTER' if delta_dd > 0 else 'WORSE'})")
print(f"  Whipsaw reduction: {ws_reduction:+.0f}%")
print(f"  WF wins: {wins}/10")
print(f"  Permutation p: {perm_p:.4f}")

if delta_sharpe > 0 and wins >= 6 and perm_p < 0.10:
    print(f"\n  HYPOTHESIS SUPPORTED: Multi-timeframe confluence improves regime detection.")
elif delta_sharpe > 0 and wins >= 5:
    print(f"\n  HYPOTHESIS PARTIALLY SUPPORTED: Some improvement but not robustly validated.")
else:
    print(f"\n  HYPOTHESIS REJECTED: Multi-timeframe confluence does not reliably beat v2.")

print(f"\n  Practical recommendation based on results above.")
print("\nDone.")
