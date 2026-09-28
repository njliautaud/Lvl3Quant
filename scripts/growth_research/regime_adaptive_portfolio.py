#!/usr/bin/env python3
"""
Regime-Adaptive Portfolio Strategy
====================================
Switches allocations 2-6 times/year based on multi-signal macro regime detection.

Regimes (4-state):
  RISK-ON:  VIX <30th pctl, credit tight, momentum +  -> aggressive growth
  CAUTIOUS: VIX mid-range, mixed signals              -> balanced
  RISK-OFF: VIX >70th pctl, credit widening, mom -    -> defensive
  CRISIS:   VIX >90th pctl, credit stress, mom collapse -> capital preservation

Regime transitions require N consecutive days of signal confirmation (slow switching).
Backtest 2010-2026, $100K fixed capital, no DCA (HC #713), next-day execution.

Adversarial validation (HC #705):
  1. Permutation test (1000 shuffles)
  2. Sub-period consistency (3 equal blocks)
  3. Outlier removal (top 5% of days)
  4. R1 regime test (green/red SPY days)
  5. Walk-forward (rolling 2yr train, 6mo OOS) — SLIDING, not expanding (HC #0)

Cost assumptions:
  - Leveraged ETFs (UPRO/TQQQ): 10bp round-trip
  - Unleveraged ETFs: 4bp round-trip
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/regime_adaptive_portfolio')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
REBAL_COST_BPS = 10   # round-trip cost in bps for leveraged ETFs
UNLEV_COST_BPS = 4    # round-trip cost for unleveraged

# Regime detection parameters
VIX_PCTL_WINDOW = 252          # 1yr rolling window for VIX percentile
VIX_RISKON_PCTL = 30           # below this = risk-on
VIX_RISKOFF_PCTL = 70          # above this = risk-off
VIX_CRISIS_PCTL = 90           # above this = crisis
CREDIT_SPREAD_WINDOW = 63      # 3mo rolling z-score
CREDIT_STRESS_Z = -0.5         # below this = credit stress
CREDIT_CRISIS_Z = -1.5         # below this = credit crisis
MOM_WINDOW = 60                # 60-day momentum lookback
MOM_SHORT_WINDOW = 20          # 20-day short-term momentum
CONFIRMATION_DAYS = 7          # consecutive days to confirm regime switch

# Portfolio allocations per regime
ALLOCATIONS = {
    'RISK_ON':  {'UPRO': 0.70, 'QQQ': 0.20, 'GLD': 0.10},
    'CAUTIOUS': {'SPY': 0.50, 'GLD': 0.30, 'TLT': 0.20},
    'RISK_OFF': {'SPY': 0.20, 'TLT': 0.40, 'GLD': 0.30, 'SHY': 0.10},
    'CRISIS':   {'SHY': 1.00},
}

N_PERMUTATIONS = 1000

# Benchmark allocations
BENCHMARKS = {
    'SPY_BH':       {'SPY': 1.0},
    'UPRO_BH':      {'UPRO': 1.0},
    '60_40':        {'SPY': 0.60, 'TLT': 0.40},
}

REGIME_NAMES = {0: 'RISK_ON', 1: 'CAUTIOUS', 2: 'RISK_OFF', 3: 'CRISIS'}
REGIME_COLORS = {'RISK_ON': '#2ecc71', 'CAUTIOUS': '#f39c12', 'RISK_OFF': '#e74c3c', 'CRISIS': '#8e44ad'}

print("=" * 78)
print("  REGIME-ADAPTIVE PORTFOLIO STRATEGY")
print("  Multi-signal macro regime detection with slow switching")
print("=" * 78)

# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────
print("\n[1/7] Downloading data...")

ALL_TICKERS = ['SPY', 'UPRO', 'QQQ', 'TQQQ', 'GLD', 'TLT', 'IEF', 'HYG',
               'SHY', '^VIX', '^VIX3M', 'UUP', 'SLV']

cache_file = OUTPUT_DIR / 'price_cache.parquet'

# Always re-download to get latest data
dfs = {}
for t in ALL_TICKERS:
    try:
        d = yf.download(t, start='2010-01-01', progress=False)
        if len(d) > 100:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            col_name = t.replace('^', '')
            dfs[col_name] = d['Close']
            print(f"  {t}: {len(d)} rows ({d.index[0].strftime('%Y-%m-%d')} to {d.index[-1].strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"  {t}: FAILED - {e}")

prices = pd.DataFrame(dfs)
prices.index = pd.to_datetime(prices.index)
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
prices.to_parquet(cache_file)
print(f"  Combined: {prices.shape[0]} rows, {prices.shape[1]} columns")

# Check required tickers
required = ['VIX', 'SPY', 'UPRO', 'QQQ', 'GLD', 'TLT', 'IEF', 'HYG', 'SHY']
missing = [r for r in required if r not in prices.columns]
if missing:
    print(f"  FATAL: Missing tickers: {missing}")
    exit(1)

# Forward fill small gaps, drop rows without core data
prices = prices.ffill().dropna(subset=['VIX', 'SPY', 'HYG', 'IEF'])

# ─── REGIME DETECTION ────────────────────────────────────────────────────────
print("\n[2/7] Computing regime signals...")

# Signal 1: VIX Rolling Percentile
prices['vix_pctl'] = prices['VIX'].rolling(VIX_PCTL_WINDOW).apply(
    lambda x: stats.percentileofscore(x, x.iloc[-1]), raw=False
)

# Signal 2: Credit Spread (HYG/IEF ratio z-score)
# HYG outperforming IEF = credit tightening = risk-on
prices['credit_ratio'] = prices['HYG'] / prices['IEF']
credit_ma = prices['credit_ratio'].rolling(10).mean()
prices['credit_zscore'] = (
    (credit_ma - credit_ma.rolling(CREDIT_SPREAD_WINDOW).mean()) /
    credit_ma.rolling(CREDIT_SPREAD_WINDOW).std()
)

# Signal 3: Broad market momentum (SPY)
prices['mom_60d'] = prices['SPY'].pct_change(MOM_WINDOW)
prices['mom_20d'] = prices['SPY'].pct_change(MOM_SHORT_WINDOW)
# Combine into momentum score: positive if both positive, negative if both negative
prices['mom_score'] = (prices['mom_60d'] + prices['mom_20d']) / 2

# Signal 4: VIX term structure (if VIX3M available)
if 'VIX3M' in prices.columns:
    prices['vix_term'] = prices['VIX3M'] / prices['VIX']
    prices['vix_term'] = prices['vix_term'].rolling(5).mean()
else:
    prices['vix_term'] = np.nan

# Signal 5: Dollar strength (UUP) — rising dollar often = risk-off
if 'UUP' in prices.columns:
    prices['dollar_mom'] = prices['UUP'].pct_change(MOM_SHORT_WINDOW)
else:
    prices['dollar_mom'] = 0.0

# Signal 6: Gold momentum — rising gold often = fear/uncertainty
prices['gold_mom'] = prices['GLD'].pct_change(MOM_SHORT_WINDOW)

# ─── COMPOSITE REGIME SCORE ─────────────────────────────────────────────────
print("  Computing composite regime score...")

def compute_raw_regime(row):
    """
    Score from 0 (most risk-on) to 3 (crisis) based on multiple signals.
    Each signal contributes a score; we threshold the composite.
    """
    score = 0.0
    n_signals = 0

    # VIX percentile (primary signal, double-weighted)
    vp = row.get('vix_pctl', 50)
    if not np.isnan(vp):
        if vp >= VIX_CRISIS_PCTL:
            score += 6.0  # crisis-level contribution
        elif vp >= VIX_RISKOFF_PCTL:
            score += 4.0
        elif vp >= VIX_RISKON_PCTL:
            score += 2.0
        else:
            score += 0.0
        n_signals += 2  # double weight

    # Credit z-score
    cz = row.get('credit_zscore', 0)
    if not np.isnan(cz):
        if cz < CREDIT_CRISIS_Z:
            score += 3.0
        elif cz < CREDIT_STRESS_Z:
            score += 2.0
        elif cz < 0.5:
            score += 1.0
        else:
            score += 0.0
        n_signals += 1

    # Momentum
    ms = row.get('mom_score', 0)
    if not np.isnan(ms):
        if ms < -0.10:
            score += 3.0
        elif ms < -0.03:
            score += 2.0
        elif ms < 0.02:
            score += 1.0
        else:
            score += 0.0
        n_signals += 1

    # VIX term structure
    vt = row.get('vix_term', np.nan)
    if not np.isnan(vt):
        if vt < 0.85:       # deep backwardation = crisis
            score += 3.0
        elif vt < 0.95:     # mild backwardation = risk-off
            score += 2.0
        elif vt < 1.02:     # flat = cautious
            score += 1.0
        else:                # contango = risk-on
            score += 0.0
        n_signals += 1

    # Gold momentum (rising gold = defensive signal, lower weight)
    gm = row.get('gold_mom', 0)
    if not np.isnan(gm) and gm > 0.03:
        score += 0.5  # mild risk-off signal
        n_signals += 0.5

    if n_signals == 0:
        return np.nan

    # Normalize to 0-3 scale
    avg_score = score / n_signals
    if avg_score >= 2.5:
        return 3  # CRISIS
    elif avg_score >= 1.5:
        return 2  # RISK_OFF
    elif avg_score >= 0.7:
        return 1  # CAUTIOUS
    else:
        return 0  # RISK_ON


# Apply regime detection
signal_cols = ['vix_pctl', 'credit_zscore', 'mom_score', 'vix_term', 'gold_mom']
prices = prices.dropna(subset=['vix_pctl'])  # need at least VIX percentile

prices['raw_regime'] = prices[signal_cols].apply(
    lambda row: compute_raw_regime(row), axis=1
)

# ─── SLOW REGIME TRANSITIONS ────────────────────────────────────────────────
print(f"  Applying {CONFIRMATION_DAYS}-day confirmation filter for slow transitions...")

def apply_confirmation_filter(raw_regime_series, n_confirm):
    """
    Only switch regime after N consecutive days of new regime signal.
    Exception: switch to CRISIS immediately (1 day) for capital preservation.
    """
    confirmed = raw_regime_series.copy()
    current_regime = raw_regime_series.iloc[0]
    pending_regime = current_regime
    pending_count = 0

    for i in range(len(raw_regime_series)):
        raw = raw_regime_series.iloc[i]

        if np.isnan(raw):
            confirmed.iloc[i] = current_regime
            continue

        raw = int(raw)

        # CRISIS transitions are fast (2 days confirmation only)
        crisis_confirm = min(2, n_confirm)

        if raw == current_regime:
            # Same regime, reset pending
            pending_regime = current_regime
            pending_count = 0
            confirmed.iloc[i] = current_regime
        elif raw == pending_regime:
            # Continuing pending switch
            pending_count += 1
            threshold = crisis_confirm if raw == 3 else n_confirm
            if pending_count >= threshold:
                current_regime = raw
                pending_count = 0
            confirmed.iloc[i] = current_regime
        else:
            # New pending regime
            pending_regime = raw
            pending_count = 1
            # Immediate crisis escalation if already at 2+ days
            if raw == 3 and pending_count >= crisis_confirm:
                current_regime = raw
                pending_count = 0
            confirmed.iloc[i] = current_regime

    return confirmed


prices['regime'] = apply_confirmation_filter(prices['raw_regime'], CONFIRMATION_DAYS)

# Count transitions
regime_changes = (prices['regime'].diff().fillna(0) != 0).sum()
years = (prices.index[-1] - prices.index[0]).days / 365.25
switches_per_year = regime_changes / years

print(f"\n  Regime distribution:")
for code, name in REGIME_NAMES.items():
    n = (prices['regime'] == code).sum()
    pct = n / len(prices) * 100
    print(f"    {name:12s}: {n:5d} days ({pct:5.1f}%)")
print(f"  Total regime switches: {regime_changes} ({switches_per_year:.1f}/year)")

# ─── COMPUTE RETURNS ─────────────────────────────────────────────────────────
print("\n[3/7] Computing portfolio returns...")

# Daily returns for all ETFs
ret_cols = {}
for t in ['SPY', 'UPRO', 'QQQ', 'TQQQ', 'GLD', 'TLT', 'IEF', 'HYG', 'SHY']:
    if t in prices.columns:
        ret_cols[t] = prices[t].pct_change()

returns_df = pd.DataFrame(ret_cols, index=prices.index)

# Next-day execution: signal at close of day T, execute at day T+1
prices['regime_lagged'] = prices['regime'].shift(1)
prices = prices.dropna(subset=['regime_lagged'])

# Strategy returns: weighted sum of ETF returns based on lagged regime
def compute_strategy_returns(df, regime_col='regime_lagged', apply_costs=True):
    """Compute daily strategy returns based on regime-driven allocations."""
    strat_ret = pd.Series(0.0, index=df.index, dtype=float)

    # Align returns_df to df index
    aligned_returns = returns_df.reindex(df.index).fillna(0)

    for code, name in REGIME_NAMES.items():
        mask = (df[regime_col] == code).values
        if mask.sum() == 0:
            continue
        alloc = ALLOCATIONS[name]
        for ticker, weight in alloc.items():
            if ticker in aligned_returns.columns:
                strat_ret.values[mask] += weight * aligned_returns[ticker].values[mask]

    # Transaction costs on regime changes
    if apply_costs:
        signal_changes = df[regime_col].diff().fillna(0) != 0
        # Determine cost based on regime we're switching TO
        cost = pd.Series(0.0, index=df.index)
        for code, name in REGIME_NAMES.items():
            mask = signal_changes & (df[regime_col] == code)
            alloc = ALLOCATIONS[name]
            # Cost proportional to leveraged exposure
            has_leveraged = any(t in alloc for t in ['UPRO', 'TQQQ'])
            cost_bps = REBAL_COST_BPS if has_leveraged else UNLEV_COST_BPS
            cost[mask] = cost_bps / 10000
        strat_ret -= cost

    return strat_ret


prices['strat_ret'] = compute_strategy_returns(prices)

# Benchmark returns
for bm_name, bm_alloc in BENCHMARKS.items():
    bm_ret = pd.Series(0.0, index=prices.index, dtype=float)
    for ticker, weight in bm_alloc.items():
        if ticker in returns_df.columns:
            bm_ret += weight * returns_df.loc[prices.index, ticker].fillna(0)
    prices[f'bm_{bm_name}'] = bm_ret

# v4.4 macro-scaled proxy: SPY with leverage scaling based on VIX percentile
# (Simplified: 1.5x when VIX <30th pctl, 1.0x when 30-70, 0.5x when >70)
vp = prices['vix_pctl']
v44_lev = pd.Series(1.0, index=prices.index)
v44_lev[vp < 30] = 1.5
v44_lev[vp > 70] = 0.5
v44_lev[vp > 90] = 0.0
prices['bm_v44_proxy'] = v44_lev.shift(1) * returns_df.loc[prices.index, 'SPY'].fillna(0)


# ─── METRICS FUNCTIONS ───────────────────────────────────────────────────────
def compute_metrics(returns, name='Strategy', annual_rf=0.04):
    """Compute comprehensive portfolio metrics."""
    r = returns.dropna()
    if len(r) < 20:
        return {'name': name, 'error': 'insufficient data'}

    daily_rf = annual_rf / 252
    excess = r - daily_rf

    total_ret = (1 + r).prod() - 1
    n_years = len(r) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() - daily_rf) / r.std() * np.sqrt(252) if r.std() > 0 else 0

    downside = r[r < daily_rf] - daily_rf
    downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = (r.mean() - daily_rf) / (downside_vol / np.sqrt(252)) if downside_vol > 0 else 0

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (r > 0).mean()

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    return {
        'name': name,
        'total_return': total_ret,
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': wr,
        'profit_factor': pf,
        'n_days': len(r),
        'n_years': n_years,
    }


def print_metrics(m):
    """Print metrics in a clean format."""
    if 'error' in m:
        print(f"  {m['name']}: {m['error']}")
        return
    print(f"  {m['name']:30s}  Sharpe={m['sharpe']:+.2f}  Sortino={m['sortino']:+.2f}  "
          f"CAGR={m['cagr']*100:+.1f}%  MaxDD={m['max_dd']*100:.1f}%  "
          f"Calmar={m['calmar']:.2f}  WR={m['win_rate']*100:.1f}%  PF={m['profit_factor']:.2f}")


# ─── RESULTS ─────────────────────────────────────────────────────────────────
print("\n[4/7] Computing performance metrics...")
print("-" * 110)

strat_metrics = compute_metrics(prices['strat_ret'], 'Regime-Adaptive Portfolio')
print_metrics(strat_metrics)

bm_metrics = {}
for bm_name in list(BENCHMARKS.keys()) + ['v44_proxy']:
    col = f'bm_{bm_name}'
    if col in prices.columns:
        m = compute_metrics(prices[col], bm_name)
        bm_metrics[bm_name] = m
        print_metrics(m)

print("-" * 110)

# Per-regime performance breakdown
print("\n  Per-regime performance:")
print(f"  {'Regime':12s}  {'Days':>6s}  {'Sharpe':>7s}  {'Ann Ret':>8s}  {'Ann Vol':>8s}  {'WR':>6s}  {'Avg Daily':>10s}")
print("  " + "-" * 70)

for code, name in REGIME_NAMES.items():
    mask = prices['regime_lagged'] == code
    if mask.sum() < 10:
        print(f"  {name:12s}  {mask.sum():6d}  {'N/A':>7s}")
        continue
    regime_ret = prices.loc[mask, 'strat_ret']
    rm = compute_metrics(regime_ret, name)
    avg_daily = regime_ret.mean() * 100
    print(f"  {name:12s}  {mask.sum():6d}  {rm['sharpe']:+7.2f}  {rm['cagr']*100:+7.1f}%  "
          f"{rm['ann_vol']*100:7.1f}%  {rm['win_rate']*100:5.1f}%  {avg_daily:+9.4f}%")


# ─── ADVERSARIAL VALIDATION ─────────────────────────────────────────────────
print(f"\n[5/7] Adversarial validation (HC #705)...")

actual_sharpe = strat_metrics['sharpe']

# --- Test 1: Permutation test (1000 shuffles) ---
print(f"\n  1) Permutation test ({N_PERMUTATIONS} shuffles)...")
perm_sharpes = []
regime_series = prices['regime_lagged'].values.copy()
np.random.seed(42)

for i in range(N_PERMUTATIONS):
    shuffled = regime_series.copy()
    np.random.shuffle(shuffled)
    perm_prices = prices.copy()
    perm_prices['regime_lagged'] = shuffled
    perm_ret = compute_strategy_returns(perm_prices, apply_costs=False)
    r = perm_ret.dropna()
    if r.std() > 0:
        perm_sharpes.append(r.mean() / r.std() * np.sqrt(252))

perm_sharpes = np.array(perm_sharpes)
perm_p_value = (perm_sharpes >= actual_sharpe).mean()
print(f"     Actual Sharpe: {actual_sharpe:.3f}")
print(f"     Permuted mean: {perm_sharpes.mean():.3f}, std: {perm_sharpes.std():.3f}")
print(f"     p-value: {perm_p_value:.4f} {'PASS (<0.05)' if perm_p_value < 0.05 else 'FAIL (>=0.05)'}")

# --- Test 2: Sub-period consistency (3 equal blocks) ---
print("\n  2) Sub-period consistency (3 blocks)...")
n = len(prices)
block_size = n // 3
sub_sharpes = []
for i in range(3):
    start = i * block_size
    end = (i + 1) * block_size if i < 2 else n
    block = prices.iloc[start:end]
    block_ret = block['strat_ret']
    bm = compute_metrics(block_ret, f'Block {i+1}')
    sub_sharpes.append(bm['sharpe'])
    period = f"{block.index[0].strftime('%Y-%m-%d')} to {block.index[-1].strftime('%Y-%m-%d')}"
    print(f"     Block {i+1} ({period}): Sharpe={bm['sharpe']:+.2f}, CAGR={bm['cagr']*100:+.1f}%, MaxDD={bm['max_dd']*100:.1f}%")

all_positive = all(s > 0 for s in sub_sharpes)
sharpe_range = max(sub_sharpes) - min(sub_sharpes)
print(f"     All positive: {'YES' if all_positive else 'NO'}, Range: {sharpe_range:.2f} "
      f"{'PASS' if all_positive and sharpe_range < 1.5 else 'MARGINAL' if all_positive else 'FAIL'}")

# --- Test 3: Outlier removal (top 5% of daily returns) ---
print("\n  3) Outlier removal (top 5% of abs(daily returns))...")
abs_ret = prices['strat_ret'].abs()
threshold_95 = abs_ret.quantile(0.95)
non_outlier_mask = abs_ret <= threshold_95
outlier_ret = prices.loc[non_outlier_mask, 'strat_ret']
outlier_m = compute_metrics(outlier_ret, 'Excl Top 5%')
print(f"     Full Sharpe:    {actual_sharpe:.3f}")
print(f"     No-outlier Sharpe: {outlier_m['sharpe']:.3f}")
pct_change = (outlier_m['sharpe'] - actual_sharpe) / abs(actual_sharpe) * 100 if actual_sharpe != 0 else 0
print(f"     Change: {pct_change:+.1f}% {'PASS (robust)' if abs(pct_change) < 30 else 'FAIL (outlier-dependent)'}")

# --- Test 4: R1 regime test (green/red SPY days) ---
print("\n  4) R1 regime test (green vs red SPY days)...")
spy_ret = returns_df.loc[prices.index, 'SPY'].fillna(0)
green_mask = spy_ret > 0
red_mask = spy_ret < 0

green_ret = prices.loc[green_mask, 'strat_ret']
red_ret = prices.loc[red_mask, 'strat_ret']

green_m = compute_metrics(green_ret, 'Green days')
red_m = compute_metrics(red_ret, 'Red days')

print(f"     Green days Sharpe: {green_m['sharpe']:+.3f}")
print(f"     Red days Sharpe:   {red_m['sharpe']:+.3f}")

if max(abs(green_m['sharpe']), abs(red_m['sharpe'])) > 0:
    regime_imbalance = abs(green_m['sharpe'] - red_m['sharpe']) / max(abs(green_m['sharpe']), abs(red_m['sharpe']))
else:
    regime_imbalance = 0
print(f"     Regime imbalance: {regime_imbalance:.2f} "
      f"{'PASS (<0.50)' if regime_imbalance < 0.50 else 'FAIL (>=0.50, regime-tailored)'}")

# --- Test 5: Walk-forward (SLIDING 2yr train, 6mo OOS) ---
print("\n  5) Walk-forward validation (SLIDING 2yr train, 6mo OOS)...")

train_days = 504   # ~2 years
oos_days = 126     # ~6 months
step = oos_days

wf_results = []
wf_idx = 0

while wf_idx + train_days + oos_days <= len(prices):
    train_slice = prices.iloc[wf_idx:wf_idx + train_days]
    oos_slice = prices.iloc[wf_idx + train_days:wf_idx + train_days + oos_days]

    # In-sample metrics
    train_m = compute_metrics(train_slice['strat_ret'], f'Train_{wf_idx}')

    # Out-of-sample metrics (same strategy, no re-optimization)
    oos_m = compute_metrics(oos_slice['strat_ret'], f'OOS_{wf_idx}')

    oos_period = f"{oos_slice.index[0].strftime('%Y-%m-%d')} to {oos_slice.index[-1].strftime('%Y-%m-%d')}"

    wf_results.append({
        'oos_start': oos_slice.index[0].strftime('%Y-%m-%d'),
        'oos_end': oos_slice.index[-1].strftime('%Y-%m-%d'),
        'train_sharpe': train_m.get('sharpe', 0),
        'oos_sharpe': oos_m.get('sharpe', 0),
        'oos_cagr': oos_m.get('cagr', 0),
        'oos_maxdd': oos_m.get('max_dd', 0),
    })

    wf_idx += step

wf_df = pd.DataFrame(wf_results)
if len(wf_df) > 0:
    avg_oos_sharpe = wf_df['oos_sharpe'].mean()
    pct_positive = (wf_df['oos_sharpe'] > 0).mean() * 100
    print(f"     {len(wf_df)} OOS windows")
    print(f"     Avg OOS Sharpe: {avg_oos_sharpe:.3f}")
    print(f"     % OOS windows with positive Sharpe: {pct_positive:.0f}%")
    print(f"     OOS Sharpe range: [{wf_df['oos_sharpe'].min():.2f}, {wf_df['oos_sharpe'].max():.2f}]")
    print(f"     {'PASS' if pct_positive >= 60 and avg_oos_sharpe > 0 else 'FAIL'}")

    # Print individual windows
    print(f"\n     {'OOS Period':25s}  {'Train Sharpe':>13s}  {'OOS Sharpe':>11s}  {'OOS CAGR':>9s}  {'OOS MaxDD':>10s}")
    for _, row in wf_df.iterrows():
        period = f"{row['oos_start']} to {row['oos_end']}"
        print(f"     {period:25s}  {row['train_sharpe']:+12.2f}  {row['oos_sharpe']:+10.2f}  "
              f"{row['oos_cagr']*100:+8.1f}%  {row['oos_maxdd']*100:9.1f}%")

# ─── CHARTS ──────────────────────────────────────────────────────────────────
print("\n[6/7] Generating charts...")

fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 1.5, 1.5]})

# Chart 1: Equity curves
ax1 = axes[0]
cum_strat = (1 + prices['strat_ret']).cumprod() * INITIAL_CAPITAL
ax1.plot(prices.index, cum_strat, 'b-', linewidth=2, label=f"Regime-Adaptive (Sharpe={strat_metrics['sharpe']:.2f})")

colors = {'SPY_BH': 'gray', 'UPRO_BH': 'orange', '60_40': 'green', 'v44_proxy': 'purple'}
for bm_name in list(BENCHMARKS.keys()) + ['v44_proxy']:
    col = f'bm_{bm_name}'
    if col in prices.columns:
        cum_bm = (1 + prices[col]).cumprod() * INITIAL_CAPITAL
        bm_s = bm_metrics.get(bm_name, {}).get('sharpe', 0)
        ax1.plot(prices.index, cum_bm, color=colors.get(bm_name, 'gray'),
                 linewidth=1.2, alpha=0.7, label=f"{bm_name} (Sharpe={bm_s:.2f})")

ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title('Regime-Adaptive Portfolio vs Benchmarks', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Chart 2: Regime timeline
ax2 = axes[1]
for code, name in REGIME_NAMES.items():
    mask = prices['regime'] == code
    if mask.any():
        ax2.fill_between(prices.index, 0, 1, where=mask.values,
                         color=REGIME_COLORS[name], alpha=0.7, label=name)
ax2.set_ylabel('Regime')
ax2.set_yticks([])
ax2.legend(loc='upper right', ncol=4, fontsize=8)
ax2.set_title('Macro Regime Timeline', fontsize=11)

# Chart 3: Drawdown comparison
ax3 = axes[2]
cum_strat_dd = cum_strat / cum_strat.cummax() - 1
ax3.fill_between(prices.index, cum_strat_dd, 0, color='blue', alpha=0.3, label='Regime-Adaptive')
if 'bm_SPY_BH' in prices.columns:
    cum_spy = (1 + prices['bm_SPY_BH']).cumprod()
    spy_dd = cum_spy / cum_spy.cummax() - 1
    ax3.plot(prices.index, spy_dd, 'gray', linewidth=0.8, alpha=0.7, label='SPY B&H')
ax3.set_ylabel('Drawdown')
ax3.set_title('Drawdown Comparison', fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Chart 4: Rolling 1yr Sharpe
ax4 = axes[3]
rolling_sharpe = prices['strat_ret'].rolling(252).apply(
    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True
)
ax4.plot(prices.index, rolling_sharpe, 'b-', linewidth=1)
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax4.axhline(y=1, color='green', linestyle='--', alpha=0.3)
ax4.set_ylabel('Rolling 1yr Sharpe')
ax4.set_title('Rolling Sharpe Ratio', fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = OUTPUT_DIR / 'regime_adaptive_portfolio.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {chart_path}")

# Walk-forward chart
if len(wf_df) > 0:
    fig2, ax = plt.subplots(figsize=(14, 5))
    oos_dates = pd.to_datetime(wf_df['oos_start'])
    colors_wf = ['green' if s > 0 else 'red' for s in wf_df['oos_sharpe']]
    ax.bar(range(len(wf_df)), wf_df['oos_sharpe'], color=colors_wf, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=avg_oos_sharpe, color='blue', linewidth=1, linestyle='--',
               label=f'Avg OOS Sharpe={avg_oos_sharpe:.2f}')
    ax.set_xlabel('OOS Window')
    ax.set_ylabel('OOS Sharpe')
    ax.set_title('Walk-Forward OOS Sharpe (SLIDING 2yr/6mo)', fontsize=12)
    ax.set_xticks(range(len(wf_df)))
    ax.set_xticklabels([d.strftime('%Y') for d in oos_dates], rotation=45, fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wf_chart_path = OUTPUT_DIR / 'walk_forward_oos.png'
    plt.savefig(wf_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {wf_chart_path}")

# ─── SAVE RESULTS ────────────────────────────────────────────────────────────
print("\n[7/7] Saving results...")

results = {
    'strategy': strat_metrics,
    'benchmarks': bm_metrics,
    'regime_distribution': {
        REGIME_NAMES[code]: int((prices['regime_lagged'] == code).sum())
        for code in REGIME_NAMES
    },
    'regime_switches_total': int(regime_changes),
    'switches_per_year': float(switches_per_year),
    'adversarial': {
        'permutation_p_value': float(perm_p_value),
        'permutation_pass': perm_p_value < 0.05,
        'sub_period_sharpes': sub_sharpes,
        'sub_period_all_positive': all_positive,
        'outlier_removal_sharpe': outlier_m['sharpe'],
        'outlier_pct_change': float(pct_change),
        'r1_green_sharpe': green_m['sharpe'],
        'r1_red_sharpe': red_m['sharpe'],
        'r1_imbalance': float(regime_imbalance),
        'r1_pass': regime_imbalance < 0.50,
        'walk_forward_avg_oos_sharpe': float(avg_oos_sharpe) if len(wf_df) > 0 else None,
        'walk_forward_pct_positive': float(pct_positive) if len(wf_df) > 0 else None,
    },
    'parameters': {
        'vix_pctl_window': VIX_PCTL_WINDOW,
        'vix_riskon_pctl': VIX_RISKON_PCTL,
        'vix_riskoff_pctl': VIX_RISKOFF_PCTL,
        'vix_crisis_pctl': VIX_CRISIS_PCTL,
        'credit_spread_window': CREDIT_SPREAD_WINDOW,
        'confirmation_days': CONFIRMATION_DAYS,
        'mom_window': MOM_WINDOW,
        'allocations': ALLOCATIONS,
    },
    'run_timestamp': dt.datetime.now().isoformat(),
}

# Convert numpy types for JSON serialization
def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

results = convert_types(results)

results_path = OUTPUT_DIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: {results_path}")

# Save daily data for further analysis
daily_data = prices[['regime', 'regime_lagged', 'strat_ret', 'vix_pctl',
                      'credit_zscore', 'mom_score']].copy()
daily_data.to_parquet(OUTPUT_DIR / 'daily_data.parquet')

# Save walk-forward results
if len(wf_df) > 0:
    wf_df.to_csv(OUTPUT_DIR / 'walk_forward_results.csv', index=False)

# ─── FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("  FINAL SUMMARY")
print("=" * 78)
print(f"\n  Strategy: Regime-Adaptive Portfolio (4-state, {CONFIRMATION_DAYS}-day confirmation)")
print(f"  Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} ({strat_metrics['n_years']:.1f} years)")
print(f"  Regime switches: {regime_changes} total ({switches_per_year:.1f}/year)")
print(f"\n  {'Metric':20s}  {'Strategy':>10s}  {'SPY B&H':>10s}  {'60/40':>10s}  {'UPRO B&H':>10s}")
print("  " + "-" * 65)

metrics_to_show = [
    ('Sharpe', 'sharpe', '{:+.2f}'),
    ('Sortino', 'sortino', '{:+.2f}'),
    ('CAGR', 'cagr', '{:+.1%}'),
    ('Max Drawdown', 'max_dd', '{:.1%}'),
    ('Calmar', 'calmar', '{:.2f}'),
    ('Win Rate', 'win_rate', '{:.1%}'),
    ('Profit Factor', 'profit_factor', '{:.2f}'),
]

for label, key, fmt in metrics_to_show:
    s_val = fmt.format(strat_metrics.get(key, 0))
    spy_val = fmt.format(bm_metrics.get('SPY_BH', {}).get(key, 0))
    b60_val = fmt.format(bm_metrics.get('60_40', {}).get(key, 0))
    upro_val = fmt.format(bm_metrics.get('UPRO_BH', {}).get(key, 0))
    print(f"  {label:20s}  {s_val:>10s}  {spy_val:>10s}  {b60_val:>10s}  {upro_val:>10s}")

print(f"\n  Adversarial Validation Summary:")
print(f"    Permutation test:     {'PASS' if perm_p_value < 0.05 else 'FAIL'} (p={perm_p_value:.4f})")
print(f"    Sub-period:           {'PASS' if all_positive else 'FAIL'} (all blocks positive: {all_positive})")
print(f"    Outlier robustness:   {'PASS' if abs(pct_change) < 30 else 'FAIL'} ({pct_change:+.1f}% change)")
print(f"    R1 regime balance:    {'PASS' if regime_imbalance < 0.50 else 'FAIL'} (imbalance={regime_imbalance:.2f})")
if len(wf_df) > 0:
    print(f"    Walk-forward:         {'PASS' if pct_positive >= 60 and avg_oos_sharpe > 0 else 'FAIL'} "
          f"(avg OOS Sharpe={avg_oos_sharpe:.2f}, {pct_positive:.0f}% positive)")

n_pass = sum([
    perm_p_value < 0.05,
    all_positive,
    abs(pct_change) < 30,
    regime_imbalance < 0.50,
    len(wf_df) > 0 and pct_positive >= 60 and avg_oos_sharpe > 0,
])
print(f"\n  Overall: {n_pass}/5 adversarial tests passed")

final_val = cum_strat.iloc[-1]
print(f"\n  $100K -> ${final_val:,.0f} ({strat_metrics['cagr']*100:+.1f}% CAGR)")
print("=" * 78)
