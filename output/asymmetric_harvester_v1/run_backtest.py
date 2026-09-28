#!/usr/bin/env python3
"""
Asymmetric Opportunity Harvester — Full Backtest + Validation
==============================================================
Conditional strategy that goes aggressive when fear signals show
asymmetric upside, stays in cash otherwise.

Also builds:
  - Trend CTA (always-invested momentum benchmark)
  - Combined 50/50 blend (low correlation: momentum + contrarian)

Validation:
  - Permutation test (200 shuffles of signal dates)
  - Regime stability (green vs red months)
  - Sub-period stability (4 blocks)
  - Lag sensitivity (T-0 vs T-1 vs T-2)
  - Comparison to SPY B&H and Trend CTA

Output: /home/jupiter/Lvl3Quant/output/asymmetric_harvester_v1/
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json
import time

OUT = Path("/home/jupiter/Lvl3Quant/output/asymmetric_harvester_v1")
OUT.mkdir(parents=True, exist_ok=True)

COST_BPS = 10  # 10 bps per trade
MIN_HOLD_DAYS = 21  # Hold at least 21 days at Level 2-3
INITIAL_CAPITAL = 100_000
START_DATE = '2010-01-01'
END_DATE = '2026-07-21'

np.random.seed(42)

# ============================================================
# 1. DOWNLOAD DATA
# ============================================================
print("=" * 70)
print("STEP 1: Downloading data")
print("=" * 70)

tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SHY', 'HYG', 'LQD', 'IEF',
           'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB',
           '^VIX', '^VIX3M']

# UPRO data starts ~2009, use SPY * 3 leverage for earlier periods as fallback
tickers.append('UPRO')

raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)
close = raw['Close'].copy()

# Rename VIX columns
rename_map = {}
if '^VIX' in close.columns:
    rename_map['^VIX'] = 'VIX'
if '^VIX3M' in close.columns:
    rename_map['^VIX3M'] = 'VIX3M'
close.rename(columns=rename_map, inplace=True)

close = close.ffill()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

# Compute daily returns
spy_ret = close['SPY'].pct_change()
shy_ret = close['SHY'].pct_change()
upro_ret = close['UPRO'].pct_change() if 'UPRO' in close.columns else spy_ret * 3

# Sector ETFs for breadth
sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
sector_etfs = [s for s in sector_etfs if s in close.columns]

# ============================================================
# 2. COMPUTE SIGNALS (all T-1 shifted)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Computing signals")
print("=" * 70)

signals = pd.DataFrame(index=close.index)

# VIX term structure (backwardation = stress)
if 'VIX' in close.columns and 'VIX3M' in close.columns:
    signals['vix_term_ratio'] = close['VIX'] / close['VIX3M']

# VIX level + rolling percentile rank
if 'VIX' in close.columns:
    signals['vix_level'] = close['VIX']
    signals['vix_pctrank'] = close['VIX'].rolling(252, min_periods=60).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100, raw=False
    )

# IV-RV spread
spy_rvol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100
signals['iv_rv_spread'] = close['VIX'] - spy_rvol_21d
signals['iv_rv_pctrank'] = signals['iv_rv_spread'].rolling(252, min_periods=60).apply(
    lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100, raw=False
)

# SPY momentum
signals['spy_mom_6m'] = close['SPY'].pct_change(126)
signals['spy_mom_3m'] = close['SPY'].pct_change(63)
signals['spy_mom_1m'] = close['SPY'].pct_change(21)

# SPY vs 200d SMA
sma200 = close['SPY'].rolling(200).mean()
signals['spy_above_200sma'] = (close['SPY'] > sma200).astype(float)
signals['spy_below_200sma'] = (close['SPY'] < sma200).astype(float)

# Credit spread (HYG/LQD ratio change)
if 'HYG' in close.columns and 'LQD' in close.columns:
    credit_ratio = close['HYG'] / close['LQD']
    signals['credit_21d_chg'] = credit_ratio.pct_change(21)

# Breadth (% of sectors above 50d SMA)
if len(sector_etfs) > 5:
    breadth_df = pd.DataFrame()
    for s in sector_etfs:
        breadth_df[s] = (close[s] > close[s].rolling(50).mean()).astype(float)
    signals['breadth'] = breadth_df.mean(axis=1)

# Shift ALL signals by 1 day (T-1 only, no lookahead)
signals = signals.shift(1)

print(f"Computed {len(signals.columns)} signals")
print(f"Non-null rows: {signals.dropna().shape[0]}")

# ============================================================
# 3. DEFINE EXPOSURE LEVELS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Defining exposure levels")
print("=" * 70)

def classify_level(row):
    """
    Classify each day into exposure Level 0-3 based on T-1 signals.
    Returns (level, instrument) tuple.

    LEVEL 3 (max conviction, ~5% of time): VIX backwardation + weak mom, or backwardation alone
    LEVEL 2 (aggressive, ~15%): below 200SMA + low breadth, or high VIX + credit widening
    LEVEL 1 (moderate, ~30%): high IV-RV, or high VIX, or low breadth alone
    LEVEL 0 (cash, ~50%): default
    """

    # Check for NaN — default to cash
    if pd.isna(row.get('vix_term_ratio', np.nan)):
        return 0

    vix_backwardation = row.get('vix_term_ratio', 0) > 1.05
    weak_momentum = row.get('spy_mom_6m', 0) < 0
    high_vix = row.get('vix_pctrank', 0) > 0.70
    credit_widening = row.get('credit_21d_chg', 0) < -0.005  # -0.5%
    below_200sma = row.get('spy_below_200sma', 0) > 0.5
    low_breadth = row.get('breadth', 1) < 0.30
    high_iv_rv = row.get('iv_rv_pctrank', 0) > 0.80

    # LEVEL 3: Maximum conviction — blood in the streets
    if vix_backwardation and weak_momentum:
        return 3
    if vix_backwardation:
        return 3

    # LEVEL 2: Aggressive risk-on
    if below_200sma and low_breadth:
        return 2
    if high_vix and credit_widening:
        return 2

    # LEVEL 1: Moderate risk-on
    if high_iv_rv:
        return 1
    if high_vix:
        return 1
    if low_breadth:
        return 1

    # LEVEL 0: Cash (default)
    return 0


# Classify each day
levels = signals.apply(classify_level, axis=1)
levels.name = 'level'

print(f"\nExposure level distribution:")
level_counts = levels.value_counts().sort_index()
for lvl, cnt in level_counts.items():
    pct = cnt / len(levels) * 100
    print(f"  Level {lvl}: {cnt} days ({pct:.1f}%)")

# ============================================================
# 4. BACKTEST — ASYMMETRIC HARVESTER
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Backtesting Asymmetric Harvester")
print("=" * 70)

def run_harvester_backtest(levels_series, spy_ret_series, shy_ret_series,
                           upro_ret_series, cost_bps=COST_BPS,
                           min_hold=MIN_HOLD_DAYS, label="Harvester"):
    """
    Run the Asymmetric Harvester backtest.

    Level 0: 100% SHY
    Level 1: 50% SPY + 50% SHY
    Level 2: 75% SPY + 25% SHY
    Level 3: 100% UPRO (3x leveraged SPY)
    """

    # Align all series
    common_idx = levels_series.dropna().index.intersection(
        spy_ret_series.dropna().index
    ).intersection(shy_ret_series.dropna().index)

    levels_aligned = levels_series.loc[common_idx]
    spy_r = spy_ret_series.loc[common_idx]
    shy_r = shy_ret_series.loc[common_idx]
    upro_r = upro_ret_series.loc[common_idx]

    # Apply minimum hold constraint for Level 2-3
    effective_levels = levels_aligned.copy()
    current_level = 0
    hold_counter = 0

    for i in range(len(effective_levels)):
        raw_level = levels_aligned.iloc[i]

        if current_level >= 2 and hold_counter < min_hold:
            # Forced to stay at current level (or higher)
            effective_levels.iloc[i] = max(current_level, raw_level)
            hold_counter += 1
        else:
            if raw_level != current_level:
                current_level = raw_level
                hold_counter = 1 if raw_level >= 2 else 0
            else:
                hold_counter += 1
            effective_levels.iloc[i] = raw_level

    # Compute daily returns with costs
    portfolio_ret = pd.Series(0.0, index=common_idx)
    trades = 0
    prev_level = 0
    trade_log = []

    for i in range(len(common_idx)):
        dt = common_idx[i]
        lvl = effective_levels.iloc[i]

        # Daily return based on level
        if lvl == 0:
            daily_r = shy_r.iloc[i]
        elif lvl == 1:
            daily_r = 0.50 * spy_r.iloc[i] + 0.50 * shy_r.iloc[i]
        elif lvl == 2:
            daily_r = 0.75 * spy_r.iloc[i] + 0.25 * shy_r.iloc[i]
        elif lvl == 3:
            daily_r = upro_r.iloc[i]
        else:
            daily_r = shy_r.iloc[i]

        # Apply cost on level change
        if lvl != prev_level:
            # Cost proportional to allocation change
            alloc_change = abs(_get_equity_alloc(lvl) - _get_equity_alloc(prev_level))
            cost = alloc_change * cost_bps / 10000
            daily_r -= cost
            trades += 1
            trade_log.append({
                'date': str(dt.date()),
                'from_level': int(prev_level),
                'to_level': int(lvl),
                'cost_bps': cost * 10000
            })

        portfolio_ret.iloc[i] = daily_r
        prev_level = lvl

    # Compute equity curve
    equity = (1 + portfolio_ret).cumprod() * INITIAL_CAPITAL

    return {
        'returns': portfolio_ret,
        'equity': equity,
        'levels': effective_levels,
        'trades': trades,
        'trade_log': trade_log,
        'label': label,
    }


def _get_equity_alloc(level):
    """Return equity allocation fraction for a given level."""
    if level == 0: return 0.0
    elif level == 1: return 0.5
    elif level == 2: return 0.75
    elif level == 3: return 1.0
    return 0.0


# Run main backtest
result_harvester = run_harvester_backtest(
    levels, spy_ret, shy_ret, upro_ret,
    label="Asymmetric Harvester"
)

# ============================================================
# 5. TREND CTA BENCHMARK
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Building Trend CTA benchmark")
print("=" * 70)

def run_trend_cta(close_df, cost_bps=COST_BPS, label="Trend CTA"):
    """
    Simple Trend CTA:
    - Long SPY when SPY > 200d SMA AND 12m momentum > 0
    - Otherwise SHY (cash equivalent)
    - Monthly rebalance
    """
    spy_close = close_df['SPY'].dropna()
    sma200 = spy_close.rolling(200).mean()
    mom_12m = spy_close.pct_change(252)

    # Signal: long when above SMA and positive momentum (T-1)
    trend_signal = ((spy_close > sma200) & (mom_12m > 0)).shift(1).astype(float)

    spy_r = spy_close.pct_change()
    shy_r = close_df['SHY'].pct_change().reindex(spy_r.index)

    common = trend_signal.dropna().index.intersection(spy_r.dropna().index).intersection(shy_r.dropna().index)

    sig = trend_signal.loc[common]
    spy_r_a = spy_r.loc[common]
    shy_r_a = shy_r.loc[common].fillna(0)

    # Monthly rebalance only
    monthly_sig = sig.copy()
    prev_sig = 0
    month_counter = 0
    last_month = None

    for i in range(len(monthly_sig)):
        dt = common[i]
        current_month = (dt.year, dt.month)

        if current_month != last_month:
            # New month — can change signal
            last_month = current_month
            prev_sig = sig.iloc[i]

        monthly_sig.iloc[i] = prev_sig

    # Compute returns with costs
    port_ret = pd.Series(0.0, index=common)
    trades = 0
    prev = 0

    for i in range(len(common)):
        s = monthly_sig.iloc[i]
        daily_r = s * spy_r_a.iloc[i] + (1 - s) * shy_r_a.iloc[i]

        if s != prev:
            daily_r -= cost_bps / 10000
            trades += 1

        port_ret.iloc[i] = daily_r
        prev = s

    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL

    return {
        'returns': port_ret,
        'equity': equity,
        'signal': monthly_sig,
        'trades': trades,
        'label': label,
    }


result_cta = run_trend_cta(close, label="Trend CTA")

# ============================================================
# 6. COMBINED STRATEGY (50/50 blend)
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Building Combined strategy")
print("=" * 70)

# Align returns
common_idx = result_harvester['returns'].index.intersection(result_cta['returns'].index)
harv_ret = result_harvester['returns'].loc[common_idx]
cta_ret = result_cta['returns'].loc[common_idx]

combined_ret = 0.50 * harv_ret + 0.50 * cta_ret
combined_equity = (1 + combined_ret).cumprod() * INITIAL_CAPITAL

result_combined = {
    'returns': combined_ret,
    'equity': combined_equity,
    'label': "Combined (50/50)",
}

# SPY Buy & Hold
spy_bh_ret = spy_ret.loc[common_idx]
spy_bh_equity = (1 + spy_bh_ret).cumprod() * INITIAL_CAPITAL

result_spy = {
    'returns': spy_bh_ret,
    'equity': spy_bh_equity,
    'label': "SPY Buy & Hold",
}

# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Computing performance metrics")
print("=" * 70)

def compute_metrics(result_dict):
    """Compute comprehensive risk-adjusted metrics."""
    ret = result_dict['returns']
    eq = result_dict['equity']
    label = result_dict['label']

    # Filter to valid data
    ret = ret.dropna()
    if len(ret) < 252:
        return None

    # Basic stats
    total_days = len(ret)
    years = total_days / 252

    # Annualized return
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1

    # Volatility
    ann_vol = ret.std() * np.sqrt(252)

    # Sharpe (assume risk-free ~ SHY return)
    rf_daily = shy_ret.loc[ret.index].mean() if len(shy_ret.loc[ret.index].dropna()) > 0 else 0.0001
    excess_ret = ret - rf_daily
    sharpe = excess_ret.mean() / excess_ret.std() * np.sqrt(252) if excess_ret.std() > 0 else 0

    # Sortino
    downside = ret[ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (ret.mean() - rf_daily) * 252 / downside_vol

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    # Hit rate (% of positive days)
    hit_rate = (ret > 0).mean()

    # Profit factor
    gross_profit = ret[ret > 0].sum()
    gross_loss = abs(ret[ret < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Time invested (for Harvester)
    if 'levels' in result_dict:
        lvls = result_dict['levels']
        pct_invested = (lvls > 0).mean()
        pct_l1 = (lvls == 1).mean()
        pct_l2 = (lvls == 2).mean()
        pct_l3 = (lvls == 3).mean()
    elif 'signal' in result_dict:
        pct_invested = result_dict['signal'].mean()
        pct_l1 = pct_invested
        pct_l2 = 0
        pct_l3 = 0
    else:
        pct_invested = 1.0
        pct_l1 = 1.0
        pct_l2 = 0
        pct_l3 = 0

    trades = result_dict.get('trades', 0)

    # Monthly returns
    monthly = ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    best_month = monthly.max()
    worst_month = monthly.min()
    monthly_hit = (monthly > 0).mean()

    # Yearly returns
    yearly = ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    return {
        'label': label,
        'total_return': total_return,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'hit_rate_daily': hit_rate,
        'profit_factor': pf,
        'pct_invested': pct_invested,
        'pct_level_1': pct_l1,
        'pct_level_2': pct_l2,
        'pct_level_3': pct_l3,
        'trades': trades,
        'best_month': best_month,
        'worst_month': worst_month,
        'monthly_hit_rate': monthly_hit,
        'years': years,
        'final_equity': eq.iloc[-1],
        'yearly_returns': yearly,
        'monthly_returns': monthly,
    }


all_results = [result_spy, result_cta, result_harvester, result_combined]
all_metrics = {}

print(f"\n{'='*100}")
print(f"{'Metric':<25} {'SPY B&H':>14} {'Trend CTA':>14} {'Harvester':>14} {'Combined':>14}")
print(f"{'='*100}")

for res in all_results:
    m = compute_metrics(res)
    if m:
        all_metrics[m['label']] = m

metrics_order = ['SPY Buy & Hold', 'Trend CTA', 'Asymmetric Harvester', 'Combined (50/50)']
metric_rows = [
    ('CAGR', 'cagr', '{:.1%}'),
    ('Ann. Volatility', 'ann_vol', '{:.1%}'),
    ('Sharpe Ratio', 'sharpe', '{:.2f}'),
    ('Sortino Ratio', 'sortino', '{:.2f}'),
    ('Max Drawdown', 'max_dd', '{:.1%}'),
    ('Calmar Ratio', 'calmar', '{:.2f}'),
    ('Profit Factor', 'profit_factor', '{:.2f}'),
    ('Daily Hit Rate', 'hit_rate_daily', '{:.1%}'),
    ('Monthly Hit Rate', 'monthly_hit_rate', '{:.1%}'),
    ('% Time Invested', 'pct_invested', '{:.1%}'),
    ('Total Trades', 'trades', '{:.0f}'),
    ('Best Month', 'best_month', '{:.1%}'),
    ('Worst Month', 'worst_month', '{:.1%}'),
    ('Final Equity ($100K)', 'final_equity', '${:,.0f}'),
]

for row_name, key, fmt in metric_rows:
    vals = []
    for name in metrics_order:
        m = all_metrics.get(name, {})
        v = m.get(key, 0) if m else 0
        vals.append(fmt.format(v))
    print(f"  {row_name:<23} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")

print(f"{'='*100}")

# Hit rate per level for Harvester
if 'levels' in result_harvester:
    print("\n--- Harvester: Hit Rate by Exposure Level ---")
    harv_ret = result_harvester['returns']
    harv_lvls = result_harvester['levels']
    for lvl in sorted(harv_lvls.unique()):
        mask = harv_lvls == lvl
        if mask.sum() > 0:
            lvl_ret = harv_ret[mask]
            # Compute monthly returns for this level
            # Use daily hit rate as proxy
            daily_hit = (lvl_ret > 0).mean()
            avg_ret = lvl_ret.mean() * 252  # Annualized
            print(f"  Level {int(lvl)}: {mask.sum()} days ({mask.mean()*100:.1f}%), "
                  f"daily hit={daily_hit*100:.1f}%, ann. return={avg_ret*100:.1f}%")

# Correlation between strategies
print("\n--- Strategy Correlations (daily returns) ---")
corr_df = pd.DataFrame({
    'SPY': result_spy['returns'],
    'CTA': result_cta['returns'].reindex(result_spy['returns'].index),
    'Harvester': result_harvester['returns'].reindex(result_spy['returns'].index),
    'Combined': result_combined['returns'].reindex(result_spy['returns'].index),
}).dropna()

corr_matrix = corr_df.corr()
print(corr_matrix.round(3).to_string())

# ============================================================
# 8. VALIDATION — PERMUTATION TEST
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Permutation test (200 shuffles)")
print("=" * 70)

N_PERMS = 200
perm_sharpes = []
perm_cagrs = []

# Get actual Sharpe
actual_sharpe = all_metrics.get('Asymmetric Harvester', {}).get('sharpe', 0)
actual_cagr = all_metrics.get('Asymmetric Harvester', {}).get('cagr', 0)

print(f"Actual Harvester Sharpe: {actual_sharpe:.3f}, CAGR: {actual_cagr:.1%}")
print(f"Running {N_PERMS} permutations (shuffling signal dates, not returns)...")

t0 = time.time()

for perm_i in range(N_PERMS):
    # Shuffle the signal dates (keep signal values, shuffle when they occur)
    # This breaks the temporal relationship between signals and forward returns
    shuffled_idx = np.random.permutation(levels.dropna().index)
    shuffled_levels = pd.Series(levels.dropna().values, index=shuffled_idx).sort_index()
    shuffled_levels = shuffled_levels.reindex(levels.index)

    perm_result = run_harvester_backtest(
        shuffled_levels, spy_ret, shy_ret, upro_ret,
        label=f"perm_{perm_i}"
    )
    perm_m = compute_metrics(perm_result)
    if perm_m:
        perm_sharpes.append(perm_m['sharpe'])
        perm_cagrs.append(perm_m['cagr'])

    if (perm_i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  Completed {perm_i+1}/{N_PERMS} perms ({elapsed:.1f}s)")

perm_sharpes = np.array(perm_sharpes)
perm_cagrs = np.array(perm_cagrs)

p_value_sharpe = (perm_sharpes >= actual_sharpe).mean()
p_value_cagr = (perm_cagrs >= actual_cagr).mean()

print(f"\nPermutation test results:")
print(f"  Sharpe — Actual: {actual_sharpe:.3f}, Perm mean: {perm_sharpes.mean():.3f}, "
      f"Perm std: {perm_sharpes.std():.3f}, p-value: {p_value_sharpe:.4f}")
print(f"  CAGR   — Actual: {actual_cagr:.3%}, Perm mean: {perm_cagrs.mean():.3%}, "
      f"Perm std: {perm_cagrs.std():.3%}, p-value: {p_value_cagr:.4f}")

if p_value_sharpe < 0.05:
    print(f"  ** Sharpe is SIGNIFICANT at 5% level (p={p_value_sharpe:.4f}) **")
else:
    print(f"  !! Sharpe is NOT significant at 5% level (p={p_value_sharpe:.4f}) !!")

# ============================================================
# 9. VALIDATION — REGIME TEST (green vs red months)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Regime test (green vs red months)")
print("=" * 70)

# Define green/red months by SPY monthly return
spy_monthly = spy_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
harv_monthly = result_harvester['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
cta_monthly = result_cta['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
combined_monthly = result_combined['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)

# Align
common_months = spy_monthly.dropna().index.intersection(harv_monthly.dropna().index)
spy_m = spy_monthly.loc[common_months]
harv_m = harv_monthly.loc[common_months]
cta_m = cta_monthly.reindex(common_months).dropna()
comb_m = combined_monthly.reindex(common_months).dropna()

green_mask = spy_m > 0
red_mask = spy_m <= 0

print(f"Green months: {green_mask.sum()}, Red months: {red_mask.sum()}")

for name, strat_m in [('SPY B&H', spy_m), ('Trend CTA', cta_m),
                       ('Harvester', harv_m), ('Combined', comb_m)]:
    green_ret = strat_m[green_mask].mean() if green_mask.any() else 0
    red_ret = strat_m[red_mask].mean() if red_mask.any() else 0
    green_hit = (strat_m[green_mask] > 0).mean() if green_mask.any() else 0
    red_hit = (strat_m[red_mask] > 0).mean() if red_mask.any() else 0

    print(f"  {name:20s}: Green avg={green_ret*100:+.2f}%, hit={green_hit*100:.0f}% | "
          f"Red avg={red_ret*100:+.2f}%, hit={red_hit*100:.0f}%")

# ============================================================
# 10. VALIDATION — SUB-PERIOD STABILITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Sub-period stability (4 blocks)")
print("=" * 70)

harv_ret_full = result_harvester['returns'].dropna()
n = len(harv_ret_full)
block_size = n // 4

print(f"\n{'Block':<12} {'Period':<25} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Invested':>8}")
print("-" * 75)

for block_i in range(4):
    start_i = block_i * block_size
    end_i = start_i + block_size if block_i < 3 else n

    block_ret = harv_ret_full.iloc[start_i:end_i]
    block_eq = (1 + block_ret).cumprod()

    years_b = len(block_ret) / 252
    cagr_b = (block_eq.iloc[-1]) ** (1/years_b) - 1 if years_b > 0 else 0
    vol_b = block_ret.std() * np.sqrt(252)
    sharpe_b = block_ret.mean() / block_ret.std() * np.sqrt(252) if block_ret.std() > 0 else 0
    peak_b = block_eq.cummax()
    max_dd_b = ((block_eq - peak_b) / peak_b).min()

    # Time invested in this block
    block_lvls = result_harvester['levels'].iloc[start_i:end_i] if 'levels' in result_harvester else None
    invested_b = (block_lvls > 0).mean() if block_lvls is not None else 1.0

    period_str = f"{block_ret.index[0].date()} to {block_ret.index[-1].date()}"

    print(f"  Block {block_i+1:<5} {period_str:<25} {cagr_b:>7.1%} {sharpe_b:>7.2f} {max_dd_b:>7.1%} {invested_b:>7.1%}")

# ============================================================
# 11. VALIDATION — LAG SENSITIVITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Lag sensitivity (T-0, T-1, T-2)")
print("=" * 70)

for lag, lag_name in [(0, "T-0 (lookahead!)"), (1, "T-1 (standard)"), (2, "T-2 (extra lag)")]:
    # Re-shift signals
    signals_lagged = pd.DataFrame(index=close.index)

    if 'VIX' in close.columns and 'VIX3M' in close.columns:
        signals_lagged['vix_term_ratio'] = close['VIX'] / close['VIX3M']
    if 'VIX' in close.columns:
        signals_lagged['vix_level'] = close['VIX']
        signals_lagged['vix_pctrank'] = close['VIX'].rolling(252, min_periods=60).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100, raw=False
        )
    spy_rvol_21d_l = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    signals_lagged['iv_rv_spread'] = close['VIX'] - spy_rvol_21d_l
    signals_lagged['iv_rv_pctrank'] = signals_lagged['iv_rv_spread'].rolling(252, min_periods=60).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100, raw=False
    )
    signals_lagged['spy_mom_6m'] = close['SPY'].pct_change(126)
    signals_lagged['spy_below_200sma'] = (close['SPY'] < close['SPY'].rolling(200).mean()).astype(float)
    if 'HYG' in close.columns and 'LQD' in close.columns:
        signals_lagged['credit_21d_chg'] = (close['HYG'] / close['LQD']).pct_change(21)
    if len(sector_etfs) > 5:
        b_df = pd.DataFrame()
        for s in sector_etfs:
            b_df[s] = (close[s] > close[s].rolling(50).mean()).astype(float)
        signals_lagged['breadth'] = b_df.mean(axis=1)

    # Apply lag
    signals_lagged = signals_lagged.shift(lag)
    levels_lagged = signals_lagged.apply(classify_level, axis=1)

    lag_result = run_harvester_backtest(
        levels_lagged, spy_ret, shy_ret, upro_ret,
        label=f"Lag {lag_name}"
    )
    lag_m = compute_metrics(lag_result)
    if lag_m:
        print(f"  {lag_name:25s}: CAGR={lag_m['cagr']:.1%}, Sharpe={lag_m['sharpe']:.2f}, "
              f"MaxDD={lag_m['max_dd']:.1%}, Invested={lag_m['pct_invested']:.1%}")

# ============================================================
# 12. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Generating plots")
print("=" * 70)

fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 2, 2]})

# Plot 1: Equity curves
ax1 = axes[0]
for res in all_results:
    eq = res['equity']
    ax1.plot(eq.index, eq.values, label=res['label'], linewidth=1.5)
ax1.set_yscale('log')
ax1.set_title('Equity Curves (Log Scale, $100K start)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Portfolio Value ($)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Plot 2: Exposure levels
ax2 = axes[1]
if 'levels' in result_harvester:
    lvls = result_harvester['levels']
    colors = {0: '#d3d3d3', 1: '#90EE90', 2: '#4169E1', 3: '#FF4500'}
    for lvl in [0, 1, 2, 3]:
        mask = lvls == lvl
        ax2.fill_between(lvls.index, 0, 1, where=mask, alpha=0.7,
                        color=colors[lvl], label=f'Level {lvl}')
    ax2.set_title('Harvester Exposure Level Over Time', fontsize=12)
    ax2.legend(ncol=4, fontsize=9, loc='upper right')
    ax2.set_yticks([])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Plot 3: Drawdown comparison
ax3 = axes[2]
for res in all_results:
    eq = res['equity']
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax3.plot(dd.index, dd.values, label=res['label'], linewidth=1, alpha=0.8)
ax3.set_title('Drawdown (%)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylabel('Drawdown %')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Plot 4: Rolling 1-year Sharpe
ax4 = axes[3]
for res in all_results:
    ret = res['returns']
    rolling_sharpe = ret.rolling(252).mean() / ret.rolling(252).std() * np.sqrt(252)
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, label=res['label'], linewidth=1, alpha=0.8)
ax4.axhline(0, color='k', linewidth=0.5)
ax4.set_title('Rolling 1-Year Sharpe Ratio', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylabel('Sharpe')
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(OUT / 'equity_curves.png', dpi=150, bbox_inches='tight')
print(f"  Saved equity_curves.png")
plt.close()

# Plot 5: Permutation distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(perm_sharpes, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
ax1.axvline(actual_sharpe, color='red', linewidth=2, linestyle='--', label=f'Actual: {actual_sharpe:.3f}')
ax1.set_title(f'Permutation Test — Sharpe (p={p_value_sharpe:.4f})', fontsize=12)
ax1.set_xlabel('Sharpe Ratio')
ax1.legend()

ax2.hist(perm_cagrs * 100, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
ax2.axvline(actual_cagr * 100, color='red', linewidth=2, linestyle='--', label=f'Actual: {actual_cagr:.1%}')
ax2.set_title(f'Permutation Test — CAGR (p={p_value_cagr:.4f})', fontsize=12)
ax2.set_xlabel('CAGR (%)')
ax2.legend()

plt.tight_layout()
plt.savefig(OUT / 'permutation_test.png', dpi=150, bbox_inches='tight')
print(f"  Saved permutation_test.png")
plt.close()

# Plot 6: Monthly returns heatmap for Harvester
fig, ax = plt.subplots(figsize=(16, 8))
harv_monthly = result_harvester['returns'].copy()
harv_monthly.index = pd.to_datetime(harv_monthly.index)
monthly_rets = harv_monthly.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100

# Create year-month pivot
monthly_df = pd.DataFrame({'return': monthly_rets})
monthly_df['year'] = monthly_df.index.year
monthly_df['month'] = monthly_df.index.month

pivot = monthly_df.pivot_table(values='return', index='year', columns='month', aggfunc='first')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Add yearly total
yearly_total = monthly_rets.resample('YE').apply(lambda x: (1 + x/100).prod() - 1) * 100
pivot['Year'] = yearly_total.values[:len(pivot)] if len(yearly_total) >= len(pivot) else np.nan

im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                   color='black' if abs(val) < 5 else 'white')

ax.set_title('Asymmetric Harvester — Monthly Returns (%)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Return %')
plt.tight_layout()
plt.savefig(OUT / 'monthly_returns_heatmap.png', dpi=150, bbox_inches='tight')
print(f"  Saved monthly_returns_heatmap.png")
plt.close()

# ============================================================
# 13. SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: Saving results")
print("=" * 70)

# Save metrics summary
metrics_summary = {}
for name, m in all_metrics.items():
    metrics_summary[name] = {k: v for k, v in m.items()
                              if k not in ['yearly_returns', 'monthly_returns']}
    # Convert non-serializable types
    for k, v in metrics_summary[name].items():
        if isinstance(v, (np.floating, np.integer)):
            metrics_summary[name][k] = float(v)

# Add validation results
metrics_summary['validation'] = {
    'permutation_sharpe_pvalue': float(p_value_sharpe),
    'permutation_cagr_pvalue': float(p_value_cagr),
    'permutation_sharpe_mean': float(perm_sharpes.mean()),
    'permutation_sharpe_std': float(perm_sharpes.std()),
    'n_permutations': N_PERMS,
    'significant_at_5pct': bool(p_value_sharpe < 0.05),
}

# Strategy correlations
metrics_summary['correlations'] = {
    'harvester_vs_spy': float(corr_matrix.loc['Harvester', 'SPY']),
    'harvester_vs_cta': float(corr_matrix.loc['Harvester', 'CTA']),
    'cta_vs_spy': float(corr_matrix.loc['CTA', 'SPY']),
    'combined_vs_spy': float(corr_matrix.loc['Combined', 'SPY']),
}

with open(OUT / 'metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2, default=str)

# Save daily returns
daily_df = pd.DataFrame({
    'spy_bh': result_spy['returns'],
    'trend_cta': result_cta['returns'],
    'harvester': result_harvester['returns'],
    'combined': result_combined['returns'],
    'harvester_level': result_harvester.get('levels', pd.Series()),
}).dropna(how='all')
daily_df.to_csv(OUT / 'daily_returns.csv')

# Save trade log
with open(OUT / 'trade_log.json', 'w') as f:
    json.dump(result_harvester.get('trade_log', []), f, indent=2)

# Save yearly returns comparison
print("\n--- Yearly Returns Comparison ---")
yearly_comp = pd.DataFrame()
for name in metrics_order:
    m = all_metrics.get(name)
    if m and 'yearly_returns' in m:
        yearly_comp[name] = m['yearly_returns']

if len(yearly_comp) > 0:
    yearly_comp.index = yearly_comp.index.year
    yearly_comp.to_csv(OUT / 'yearly_returns.csv', float_format='%.4f')
    print(yearly_comp.applymap(lambda x: f"{x:.1%}" if not pd.isna(x) else "").to_string())

# ============================================================
# 14. GENERATE TEXT REPORT
# ============================================================
print("\n" + "=" * 70)
print("STEP 14: Generating summary report")
print("=" * 70)

report = []
report.append("=" * 80)
report.append("ASYMMETRIC OPPORTUNITY HARVESTER — BACKTEST REPORT")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append(f"Period: {START_DATE} to {END_DATE}")
report.append(f"Starting Capital: ${INITIAL_CAPITAL:,.0f}")
report.append("=" * 80)

report.append("\n\n1. STRATEGY DESCRIPTION")
report.append("-" * 40)
report.append("""
This is a CONDITIONAL strategy that goes aggressive when fear signals show
asymmetric upside and stays in cash/defensive otherwise.

Exposure Levels (based on T-1 signals):
  Level 0 (CASH): Default — hold SHY. Active ~50% of the time.
  Level 1 (MODERATE): 50% SPY + 50% SHY. Triggered by high IV-RV, high VIX, or low breadth alone.
  Level 2 (AGGRESSIVE): 75% SPY + 25% SHY. Triggered by below 200SMA + low breadth, or high VIX + credit widening.
  Level 3 (MAX CONVICTION): 100% UPRO (3x SPY). Triggered by VIX backwardation (+/- weak momentum).

Minimum hold: 21 trading days at Level 2-3.
Cost: 10 bps per trade.
""")

report.append("\n2. PERFORMANCE SUMMARY")
report.append("-" * 40)

header = f"  {'Metric':<25} {'SPY B&H':>14} {'Trend CTA':>14} {'Harvester':>14} {'Combined':>14}"
report.append(header)
report.append("  " + "-" * 83)

for row_name, key, fmt in metric_rows:
    vals = []
    for name in metrics_order:
        m = all_metrics.get(name, {})
        v = m.get(key, 0) if m else 0
        vals.append(fmt.format(v))
    report.append(f"  {row_name:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")

report.append(f"\n\n3. EXPOSURE LEVEL BREAKDOWN (Harvester)")
report.append("-" * 40)
if 'levels' in result_harvester:
    harv_ret_r = result_harvester['returns']
    harv_lvls_r = result_harvester['levels']
    for lvl in sorted(harv_lvls_r.unique()):
        mask = harv_lvls_r == lvl
        if mask.sum() > 0:
            lvl_ret = harv_ret_r[mask]
            daily_hit = (lvl_ret > 0).mean()
            avg_ret = lvl_ret.mean() * 252
            report.append(f"  Level {int(lvl)}: {mask.sum()} days ({mask.mean()*100:.1f}%), "
                         f"daily hit={daily_hit*100:.1f}%, ann. return={avg_ret*100:.1f}%")

report.append(f"\n\n4. STRATEGY CORRELATIONS")
report.append("-" * 40)
report.append(corr_matrix.round(3).to_string())

report.append(f"\n\n5. VALIDATION — PERMUTATION TEST ({N_PERMS} shuffles)")
report.append("-" * 40)
report.append(f"  Sharpe: Actual={actual_sharpe:.3f}, Perm mean={perm_sharpes.mean():.3f}, p={p_value_sharpe:.4f}")
report.append(f"  CAGR:   Actual={actual_cagr:.3%}, Perm mean={perm_cagrs.mean():.3%}, p={p_value_cagr:.4f}")
if p_value_sharpe < 0.05:
    report.append(f"  CONCLUSION: Signal timing is STATISTICALLY SIGNIFICANT (p={p_value_sharpe:.4f})")
else:
    report.append(f"  CONCLUSION: Signal timing is NOT statistically significant (p={p_value_sharpe:.4f})")

report.append(f"\n\n6. VALIDATION — REGIME TEST")
report.append("-" * 40)
for name, strat_m_r in [('SPY B&H', spy_m), ('Trend CTA', cta_m),
                          ('Harvester', harv_m), ('Combined', comb_m)]:
    green_ret = strat_m_r[green_mask].mean() if green_mask.any() else 0
    red_ret = strat_m_r[red_mask].mean() if red_mask.any() else 0
    report.append(f"  {name:20s}: Green avg={green_ret*100:+.2f}% | Red avg={red_ret*100:+.2f}%")

report.append(f"\n\n7. VALIDATION — SUB-PERIOD STABILITY")
report.append("-" * 40)
for block_i in range(4):
    start_i = block_i * block_size
    end_i = start_i + block_size if block_i < 3 else n
    block_ret = harv_ret_full.iloc[start_i:end_i]
    block_eq = (1 + block_ret).cumprod()
    years_b = len(block_ret) / 252
    cagr_b = (block_eq.iloc[-1]) ** (1/years_b) - 1 if years_b > 0 else 0
    sharpe_b = block_ret.mean() / block_ret.std() * np.sqrt(252) if block_ret.std() > 0 else 0
    peak_b = block_eq.cummax()
    max_dd_b = ((block_eq - peak_b) / peak_b).min()
    period_str = f"{block_ret.index[0].date()} to {block_ret.index[-1].date()}"
    report.append(f"  Block {block_i+1}: {period_str} — CAGR={cagr_b:.1%}, Sharpe={sharpe_b:.2f}, MaxDD={max_dd_b:.1%}")

report.append(f"\n\n8. KEY TAKEAWAYS")
report.append("-" * 40)

harv_m_final = all_metrics.get('Asymmetric Harvester', {})
cta_m_final = all_metrics.get('Trend CTA', {})
comb_m_final = all_metrics.get('Combined (50/50)', {})
spy_m_final = all_metrics.get('SPY Buy & Hold', {})

report.append(f"""
1. The Harvester is only invested {harv_m_final.get('pct_invested',0):.0%} of the time but targets
   fear-driven asymmetric opportunities for outsized returns.

2. Harvester Sharpe: {harv_m_final.get('sharpe',0):.2f} vs SPY B&H: {spy_m_final.get('sharpe',0):.2f}

3. The Combined (50% CTA + 50% Harvester) strategy blends momentum and contrarian:
   - CAGR: {comb_m_final.get('cagr',0):.1%} vs SPY: {spy_m_final.get('cagr',0):.1%}
   - Sharpe: {comb_m_final.get('sharpe',0):.2f} vs SPY: {spy_m_final.get('sharpe',0):.2f}
   - MaxDD: {comb_m_final.get('max_dd',0):.1%} vs SPY: {spy_m_final.get('max_dd',0):.1%}

4. Harvester-CTA correlation: {corr_matrix.loc['Harvester', 'CTA']:.3f} —
   {'LOW correlation confirms the blend adds diversification.' if abs(corr_matrix.loc['Harvester', 'CTA']) < 0.5 else 'Moderate correlation — some diversification benefit.'}

5. Permutation test p-value: {p_value_sharpe:.4f} —
   {'Signal timing is statistically significant.' if p_value_sharpe < 0.05 else 'Signal timing needs more investigation.'}
""")

report.append("\n" + "=" * 80)
report.append("END OF REPORT")
report.append("=" * 80)

report_text = "\n".join(report)
with open(OUT / 'backtest_report.txt', 'w') as f:
    f.write(report_text)

print(report_text)

print(f"\n\nAll outputs saved to: {OUT}")
print("Files generated:")
for f in sorted(OUT.glob('*')):
    print(f"  {f.name}")
