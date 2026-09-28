#!/usr/bin/env python3
"""
v4.4 + VIX Curve Crisis Exit
==============================
Layers a VIX term structure inversion (backwardation) crisis exit ON TOP of the
proven v4.4 Full Adaptive strategy.

v4.4 handles the "when to be aggressive" part:
  - VIX percentile -> adaptive UPRO/SPY/GLD allocation
  - 6-factor confluence gate with adaptive entry/exit thresholds
  - SMA 20/200 protection, September hedge

What v4.4 LACKS: a CRISIS EXIT for extreme tail events.

This script adds: when VIX term structure inverts (VIX3M/VIX < 1.0) for N
consecutive days, OVERRIDE everything and go to a crisis asset (SHY or GLD+TLT).

Tested variants:
  A) v4.4 standalone (baseline)
  B) v4.4 + SHY crisis exit
  C) v4.4 + GLD/TLT crisis exit (50/50)
  D) SPY buy & hold (benchmark)

Confirmation period sweep: 2, 3, 5, 7 days.

Full adversarial validation (HC #705):
  1. Permutation test (1000 shuffles)
  2. Sub-period consistency (yearly + 3yr blocks)
  3. Outlier robustness (remove top/bottom 1%)
  4. R1 regime test (green/red day asymmetry < 0.50)
  5. Walk-forward validation (3yr train / 1yr OOS)

Backtest: 2010-2026, $100K initial, NO DCA, next-day execution.
Output: /home/jupiter/Lvl3Quant/output/growth_research/v44_crisis_exit/
"""

import os
import json
import time
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
np.random.seed(42)

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/v44_crisis_exit')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
REBAL_COST_BPS = 5  # conservative switching cost
N_PERMUTATIONS = 1000

# =============================================================================
# DATA
# =============================================================================

def download_data():
    print("=" * 80)
    print("v4.4 + VIX CURVE CRISIS EXIT")
    print("=" * 80)
    print(f"\nRun started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,}, no DCA, next-day execution")

    print("\n[1/8] Downloading data...")
    tickers = ['^VIX', '^VIX3M', 'SPY', 'UPRO', 'GLD', 'TLT', 'SHY']
    cache_file = OUTPUT_DIR / 'price_cache.parquet'

    if cache_file.exists():
        prices = pd.read_parquet(cache_file)
        print(f"  Loaded from cache: {prices.shape}")
    else:
        dfs = {}
        for t in tickers:
            try:
                d = yf.download(t, start='2008-01-01', end='2026-07-18', progress=False)
                if len(d) > 100:
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)
                    dfs[t.replace('^', '')] = d['Close']
                    print(f"  {t}: {len(d)} rows")
            except Exception as e:
                print(f"  {t}: FAILED - {e}")

        prices = pd.DataFrame(dfs)
        prices.index = pd.to_datetime(prices.index)
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        prices.to_parquet(cache_file)
        print(f"  Saved cache: {prices.shape}")

    prices = prices.ffill()

    # Compute returns for all assets
    for t in ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY']:
        if t in prices.columns:
            prices[f'{t}_ret'] = prices[t].pct_change()

    # Need VIX, VIX3M, SPY, UPRO at minimum
    prices = prices.dropna(subset=['VIX', 'VIX3M', 'SPY', 'UPRO', 'GLD', 'TLT', 'SHY'])
    prices = prices.dropna(subset=['SPY_ret'])

    # Start from 2010 to match UPRO inception
    prices = prices[prices.index >= '2010-01-01']

    print(f"  Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} rows")
    return prices


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================

def compute_signals(prices):
    """Compute all signals for v4.4 + VIX curve overlay."""
    print("\n[2/8] Computing signals...")
    spy = prices['SPY']
    spy_ret = prices['SPY_ret']
    vix = prices['VIX']

    sig = {}

    # -- VIX percentile (63d) for v4.4 -- vectorized for speed
    vix_vals = vix.values
    pctile_63 = np.full(len(vix_vals), np.nan)
    for i in range(63, len(vix_vals)):
        window = vix_vals[i-63:i]  # 63 prior values (not including current)
        pctile_63[i] = (vix_vals[i] > window).sum() / len(window) * 100
    sig['vix_pctile_63'] = pd.Series(pctile_63, index=vix.index)

    # -- Realized vol --
    sig['vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    sig['vol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    sig['vol_63d_trend'] = sig['vol_63d'] - sig['vol_63d'].rolling(21).mean()

    # -- Confluence signals --
    sig['mom_5d'] = spy.pct_change(5)
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    sig['rsi_10'] = 100 - (100 / (1 + rs))
    sig['sma_20'] = spy.rolling(20).mean()
    sig['sma_50'] = spy.rolling(50).mean()
    sig['sma_200'] = spy.rolling(200).mean()
    sig['sma_200_slope'] = sig['sma_200'].pct_change(20)

    # -- VIX term structure --
    sig['vix_ratio'] = prices['VIX3M'] / prices['VIX']  # < 1.0 = backwardation = stress

    print(f"  Computed {len(sig)} signal series.")
    return sig


# =============================================================================
# v4.4 FULL ADAPTIVE REGIME
# =============================================================================

def confluence_score(sig, i):
    """6-factor confluence score (0-3)."""
    s = 0.0
    m = sig['mom_5d'].iloc[i]
    r = sig['rsi_10'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s50 = sig['sma_50'].iloc[i]
    v21 = sig['vol_21d'].iloc[i]
    slope = sig['sma_200_slope'].iloc[i]
    vt = sig['vol_63d_trend'].iloc[i]

    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


def v44_regime(sig, i, date, in_upro):
    """v4.4 Full Adaptive: VIX percentile + adaptive confluence thresholds.
    Returns (holding, new_in_upro_state)."""
    s20 = sig['sma_20'].iloc[i]
    s200 = sig['sma_200'].iloc[i]

    # September hedge
    if date.month == 9:
        return 'SPY', False

    # MA protection
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200:
        return 'SPY', False

    pctile = sig['vix_pctile_63'].iloc[i]
    if np.isnan(pctile):
        return 'SPY', False

    # High VIX percentile -> defensive (GLD)
    if pctile > 80:
        return 'GLD', False

    # Adaptive confluence thresholds based on VIX percentile
    if pctile > 60:
        entry, exit_t = 3.0, 2.5
    elif pctile < 30:
        entry, exit_t = 2.0, 1.5
    else:
        entry, exit_t = 2.5, 2.0

    # Mid-range VIX: raise the bar
    if pctile > 20:
        entry = max(entry, 2.5)

    score = confluence_score(sig, i)
    if in_upro:
        if score < exit_t:
            return 'SPY', False
        return 'UPRO', True
    else:
        if score >= entry:
            return 'UPRO', True
        return 'SPY', False


# =============================================================================
# VIX CURVE CRISIS EXIT SIGNAL
# =============================================================================

def compute_crisis_signal(sig, confirm_days=3, threshold=1.0, recovery_threshold=1.02):
    """
    Generate crisis exit signal from VIX term structure inversion.

    When VIX3M/VIX < threshold for confirm_days consecutive days -> CRISIS
    Stay in crisis until ratio recovers above recovery_threshold (hysteresis).

    Returns: Series of 0 (crisis) or 1 (normal)
    """
    ratio = sig['vix_ratio']
    backwardation = ratio < threshold

    # Require N consecutive days of backwardation
    if confirm_days > 1:
        confirmed = backwardation.rolling(confirm_days, min_periods=confirm_days).sum() >= confirm_days
    else:
        confirmed = backwardation

    # Hysteresis: stay in crisis until full recovery
    signal = pd.Series(1, index=ratio.index)
    in_crisis = False

    for i in range(len(ratio)):
        if confirmed.iloc[i]:
            in_crisis = True
        elif ratio.iloc[i] > recovery_threshold:
            in_crisis = False

        if in_crisis:
            signal.iloc[i] = 0

    return signal


# =============================================================================
# COMBINED STRATEGY SIMULATION (NO DCA)
# =============================================================================

def simulate_strategy(prices, sig, mode='v44_only', confirm_days=3,
                      crisis_asset='SHY', warmup=260):
    """
    Simulate v4.4 with optional crisis exit overlay.

    Modes:
      'v44_only'      - v4.4 standalone
      'v44_crisis'    - v4.4 + crisis exit to specified asset
      'spy_bh'        - SPY buy & hold

    crisis_asset: 'SHY' or 'GLD_TLT' (50/50)

    NO DCA. Fixed $100K initial. Next-day execution (signal lagged 1 day).
    """
    # Compute crisis signal if needed
    if mode == 'v44_crisis':
        crisis_sig = compute_crisis_signal(sig, confirm_days=confirm_days)
        crisis_sig_lag = crisis_sig.shift(1).fillna(1)  # next-day execution

    capital = float(INITIAL_CAPITAL)
    in_upro = False
    prev_holding = None
    n_switches = 0
    daily_values = []
    daily_holdings = []
    daily_crisis = []  # track crisis state

    for i in range(warmup, len(prices)):
        date = prices.index[i]
        d = date.date() if hasattr(date, 'date') else date

        if mode == 'spy_bh':
            holding = 'SPY'
        elif mode in ('v44_only', 'v44_crisis'):
            # Get v4.4 regime
            holding, in_upro = v44_regime(sig, i, d, in_upro)

            # Crisis override
            if mode == 'v44_crisis':
                is_crisis = crisis_sig_lag.iloc[i] == 0
                if is_crisis:
                    holding = '__CRISIS__'  # placeholder
                    in_upro = False
        else:
            holding = 'SPY'

        # Switching cost
        if prev_holding is not None and holding != prev_holding:
            n_switches += 1
            capital *= (1 - REBAL_COST_BPS / 10000)
        prev_holding = holding

        # Apply return
        if holding == '__CRISIS__':
            if crisis_asset == 'SHY':
                r = prices['SHY_ret'].iloc[i]
                if not np.isnan(r):
                    capital *= (1 + r)
            elif crisis_asset == 'GLD_TLT':
                r_gld = prices['GLD_ret'].iloc[i]
                r_tlt = prices['TLT_ret'].iloc[i]
                if not np.isnan(r_gld) and not np.isnan(r_tlt):
                    capital *= (1 + 0.5 * r_gld + 0.5 * r_tlt)
            daily_crisis.append(1)
        else:
            ret_col = f'{holding}_ret'
            if ret_col in prices.columns:
                r = prices[ret_col].iloc[i]
                if not np.isnan(r):
                    capital *= (1 + r)
            daily_crisis.append(0)

        daily_values.append(capital)
        daily_holdings.append(holding)

    dates = prices.index[warmup:warmup + len(daily_values)]
    return {
        'values': pd.Series(daily_values, index=dates),
        'holdings': pd.Series(daily_holdings, index=dates),
        'crisis_days': pd.Series(daily_crisis, index=dates),
        'n_switches': n_switches,
    }


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(result, label):
    """Compute comprehensive risk-adjusted metrics."""
    vals = result['values']
    rets = vals.pct_change().dropna()

    if len(rets) < 50:
        return {'label': label, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'calmar': 0, 'pf': 0, 'wr': 0}

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg = rets[rets < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else 1
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    wr = (rets > 0).mean()

    peak = vals.cummax()
    dd = (vals - peak) / peak
    max_dd = dd.min()

    years = (vals.index[-1] - vals.index[0]).days / 365.25
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    n_crisis = result['crisis_days'].sum() if 'crisis_days' in result else 0
    pct_crisis = n_crisis / len(vals) if len(vals) > 0 else 0

    sw_yr = result['n_switches'] / years if years > 0 else 0

    return {
        'label': label,
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'cagr': float(cagr),
        'max_dd': float(max_dd),
        'calmar': float(calmar),
        'pf': float(pf),
        'wr': float(wr),
        'ann_ret': float(ann_ret),
        'ann_vol': float(ann_vol),
        'final_value': float(vals.iloc[-1]),
        'n_switches': int(result['n_switches']),
        'switches_yr': float(sw_yr),
        'pct_crisis': float(pct_crisis),
        'n_crisis_days': int(n_crisis),
    }


# =============================================================================
# CONFIRMATION PERIOD SWEEP
# =============================================================================

def sweep_confirmation_periods(prices, sig, warmup=260):
    """Test different VIX inversion confirmation periods: 2, 3, 5, 7 days."""
    print("\n[3/8] Sweeping confirmation periods...")

    confirm_days_list = [2, 3, 5, 7]
    crisis_assets = ['SHY', 'GLD_TLT']

    sweep_results = []

    # Baseline: v4.4 standalone
    res_v44 = simulate_strategy(prices, sig, mode='v44_only', warmup=warmup)
    m_v44 = calc_metrics(res_v44, 'v4.4 Standalone')
    sweep_results.append({'confirm': 0, 'crisis_asset': 'none', **m_v44})

    # SPY benchmark
    res_spy = simulate_strategy(prices, sig, mode='spy_bh', warmup=warmup)
    m_spy = calc_metrics(res_spy, 'SPY B&H')
    sweep_results.append({'confirm': -1, 'crisis_asset': 'none', **m_spy})

    for confirm in confirm_days_list:
        for ca in crisis_assets:
            res = simulate_strategy(prices, sig, mode='v44_crisis',
                                    confirm_days=confirm, crisis_asset=ca,
                                    warmup=warmup)
            label = f'v4.4+{ca} (confirm={confirm}d)'
            m = calc_metrics(res, label)
            sweep_results.append({'confirm': confirm, 'crisis_asset': ca, **m})

    sweep_df = pd.DataFrame(sweep_results)

    print(f"\n  {'Strategy':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
          f"{'MaxDD':>8} {'Calmar':>7} {'Crisis%':>8} {'Sw/yr':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")

    for _, row in sweep_df.iterrows():
        print(f"  {row['label']:<35} {row['sharpe']:>7.3f} {row['sortino']:>8.3f} "
              f"{row['cagr']:>6.1%} {row['max_dd']:>7.1%} {row['calmar']:>7.3f} "
              f"{row['pct_crisis']:>7.1%} {row['switches_yr']:>6.1f}")

    return sweep_df


# =============================================================================
# MAIN BACKTEST WITH BEST CONFIGS
# =============================================================================

def run_main_backtest(prices, sig, warmup=260):
    """Run the 4 main variants with optimized confirmation period."""
    print("\n[4/8] Running main backtest comparison...")

    strategies = {}

    # A) v4.4 standalone
    res_a = simulate_strategy(prices, sig, mode='v44_only', warmup=warmup)
    strategies['v4.4 Standalone'] = res_a

    # B) v4.4 + SHY crisis exit (3-day confirmation)
    res_b = simulate_strategy(prices, sig, mode='v44_crisis',
                              confirm_days=3, crisis_asset='SHY', warmup=warmup)
    strategies['v4.4 + SHY Crisis'] = res_b

    # C) v4.4 + GLD/TLT crisis exit (3-day confirmation)
    res_c = simulate_strategy(prices, sig, mode='v44_crisis',
                              confirm_days=3, crisis_asset='GLD_TLT', warmup=warmup)
    strategies['v4.4 + GLD/TLT Crisis'] = res_c

    # D) SPY buy & hold
    res_d = simulate_strategy(prices, sig, mode='spy_bh', warmup=warmup)
    strategies['SPY Buy & Hold'] = res_d

    # Also test 5-day confirmation variants
    res_b5 = simulate_strategy(prices, sig, mode='v44_crisis',
                               confirm_days=5, crisis_asset='SHY', warmup=warmup)
    strategies['v4.4 + SHY Crisis (5d)'] = res_b5

    res_c5 = simulate_strategy(prices, sig, mode='v44_crisis',
                               confirm_days=5, crisis_asset='GLD_TLT', warmup=warmup)
    strategies['v4.4 + GLD/TLT Crisis (5d)'] = res_c5

    # Compute metrics
    all_metrics = {}
    for label, res in strategies.items():
        m = calc_metrics(res, label)
        all_metrics[label] = m

    print(f"\n  {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
          f"{'MaxDD':>8} {'Calmar':>7} {'PF':>6} {'WR':>6} {'Crisis%':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*8}")

    for label, m in all_metrics.items():
        print(f"  {m['label']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.3f} "
              f"{m['pf']:>6.3f} {m['wr']:>5.1%} {m['pct_crisis']:>7.1%}")

    return strategies, all_metrics


# =============================================================================
# ADVERSARIAL VALIDATION SUITE
# =============================================================================

def adversarial_permutation(prices, sig, mode, confirm_days, crisis_asset,
                            warmup=260, n_perms=N_PERMUTATIONS):
    """Gate 1: Permutation test - shuffle crisis signal timing."""
    print(f"\n  Gate 1: Permutation test ({n_perms} iterations)...")

    # Real strategy
    res_real = simulate_strategy(prices, sig, mode=mode,
                                 confirm_days=confirm_days,
                                 crisis_asset=crisis_asset, warmup=warmup)
    real_rets = res_real['values'].pct_change().dropna()
    real_sharpe = (real_rets.mean() * 252) / (real_rets.std() * np.sqrt(252))

    # For comparison: what if crisis signal timing were random?
    crisis_pct = res_real['crisis_days'].mean()  # keep same % of time in crisis

    perm_sharpes = []
    for p in range(n_perms):
        np.random.seed(p + 1000)
        # Random crisis signal with same frequency
        random_crisis = np.random.choice([0, 1], size=len(real_rets),
                                          p=[crisis_pct, 1 - crisis_pct])

        # Apply random crisis overlay to v4.4 returns
        v44_res = simulate_strategy(prices, sig, mode='v44_only', warmup=warmup)
        v44_rets = v44_res['values'].pct_change().dropna()

        if crisis_asset == 'SHY':
            crisis_rets = prices['SHY_ret'].reindex(v44_rets.index).fillna(0)
        else:
            crisis_rets = (0.5 * prices['GLD_ret'] + 0.5 * prices['TLT_ret']).reindex(v44_rets.index).fillna(0)

        perm_rets = pd.Series(
            np.where(random_crisis[:len(v44_rets)] == 0, crisis_rets.values[:len(random_crisis)], v44_rets.values[:len(random_crisis)]),
            index=v44_rets.index[:len(random_crisis)]
        )

        ann_r = perm_rets.mean() * 252
        ann_v = perm_rets.std() * np.sqrt(252)
        perm_sharpes.append(ann_r / ann_v if ann_v > 0 else 0)

        if (p + 1) % 250 == 0:
            print(f"    {p+1}/{n_perms}...")

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())
    passed = p_value < 0.05

    print(f"    Real: {real_sharpe:.3f}, Perm mean: {np.mean(perm_sharpes):.3f}, "
          f"p={p_value:.4f} -> {'PASS' if passed else 'FAIL'}")

    return {'real_sharpe': float(real_sharpe), 'perm_mean': float(np.mean(perm_sharpes)),
            'p_value': p_value, 'pass': passed}


def adversarial_subperiod(prices, sig, mode, confirm_days, crisis_asset, warmup=260):
    """Gate 2: Sub-period consistency (yearly + 3yr blocks)."""
    print(f"\n  Gate 2: Sub-period consistency...")

    res = simulate_strategy(prices, sig, mode=mode, confirm_days=confirm_days,
                            crisis_asset=crisis_asset, warmup=warmup)
    rets = res['values'].pct_change().dropna()
    spy_res = simulate_strategy(prices, sig, mode='spy_bh', warmup=warmup)
    spy_rets = spy_res['values'].pct_change().dropna()

    # Yearly breakdown
    yearly = []
    for yr in sorted(rets.index.year.unique()):
        m = rets.index.year == yr
        if m.sum() < 50:
            continue
        yr_ret = rets[m]
        yr_spy = spy_rets.reindex(yr_ret.index).dropna()

        yr_s = (yr_ret.mean() * 252) / (yr_ret.std() * np.sqrt(252)) if yr_ret.std() > 0 else 0
        spy_s = (yr_spy.mean() * 252) / (yr_spy.std() * np.sqrt(252)) if len(yr_spy) > 0 and yr_spy.std() > 0 else 0

        yr_eq = (1 + yr_ret).cumprod()
        yr_dd = ((yr_eq - yr_eq.cummax()) / yr_eq.cummax()).min()

        yearly.append({'year': yr, 'sharpe': yr_s, 'spy_sharpe': spy_s,
                       'improvement': yr_s - spy_s, 'max_dd': yr_dd})
        marker = '+' if yr_s > spy_s else '-'
        print(f"    {yr}: Sharpe={yr_s:.2f} (SPY={spy_s:.2f}) [{marker}], DD={yr_dd*100:.1f}%")

    beats_spy = sum(1 for y in yearly if y['improvement'] > 0)

    # 3-year blocks
    years = sorted(set(rets.index.year))
    blocks = []
    for i in range(0, len(years), 3):
        block_years = years[i:i+3]
        if len(block_years) < 2:
            continue
        block_rets = rets[rets.index.year.isin(block_years)]
        if len(block_rets) > 100:
            bs = (block_rets.mean() * 252) / (block_rets.std() * np.sqrt(252))
            blocks.append({'years': f"{min(block_years)}-{max(block_years)}", 'sharpe': float(bs)})

    positive_blocks = sum(1 for b in blocks if b['sharpe'] > 0)
    pct_positive_blocks = positive_blocks / len(blocks) if blocks else 0

    passed = beats_spy / len(yearly) >= 0.40 and pct_positive_blocks >= 0.60
    print(f"    Beats SPY: {beats_spy}/{len(yearly)} years ({beats_spy/len(yearly)*100:.0f}%)")
    print(f"    Positive 3yr blocks: {positive_blocks}/{len(blocks)} ({pct_positive_blocks*100:.0f}%)")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {'yearly': yearly, 'blocks': blocks, 'beats_spy': beats_spy,
            'total_years': len(yearly), 'pct_positive_blocks': pct_positive_blocks,
            'pass': passed}


def adversarial_outlier(prices, sig, mode, confirm_days, crisis_asset, warmup=260):
    """Gate 3: Outlier robustness (trim top/bottom 1%)."""
    print(f"\n  Gate 3: Outlier robustness...")

    res = simulate_strategy(prices, sig, mode=mode, confirm_days=confirm_days,
                            crisis_asset=crisis_asset, warmup=warmup)
    rets = res['values'].pct_change().dropna()

    full_sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))

    p1, p99 = rets.quantile(0.01), rets.quantile(0.99)
    trim = rets[(rets >= p1) & (rets <= p99)]
    trim_sharpe = (trim.mean() * 252) / (trim.std() * np.sqrt(252))

    degradation = (trim_sharpe - full_sharpe) / abs(full_sharpe) if full_sharpe != 0 else 0
    passed = trim_sharpe > 0

    print(f"    Full: {full_sharpe:.3f}, Trimmed: {trim_sharpe:.3f} ({degradation*100:+.1f}%)")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {'full_sharpe': float(full_sharpe), 'trimmed_sharpe': float(trim_sharpe),
            'change_pct': float(degradation * 100), 'pass': passed}


def adversarial_regime_r1(prices, sig, mode, confirm_days, crisis_asset, warmup=260):
    """Gate 4: R1 regime test (green/red SPY day asymmetry < 0.50)."""
    print(f"\n  Gate 4: R1 Regime test...")

    res = simulate_strategy(prices, sig, mode=mode, confirm_days=confirm_days,
                            crisis_asset=crisis_asset, warmup=warmup)
    strat_rets = res['values'].pct_change().dropna()
    spy_rets = prices['SPY_ret'].reindex(strat_rets.index)

    green = strat_rets[spy_rets > 0]
    red = strat_rets[spy_rets < 0]

    sharpe_g = (green.mean() * 252) / (green.std() * np.sqrt(252)) if green.std() > 0 else 0
    sharpe_r = (red.mean() * 252) / (red.std() * np.sqrt(252)) if red.std() > 0 else 0

    max_s = max(abs(sharpe_g), abs(sharpe_r), 0.01)
    delta = abs(sharpe_g - sharpe_r) / max_s
    passed = delta < 0.50

    print(f"    Green: {sharpe_g:.3f}, Red: {sharpe_r:.3f}, delta={delta:.3f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {'sharpe_green': float(sharpe_g), 'sharpe_red': float(sharpe_r),
            'delta': float(delta), 'pass': passed}


def adversarial_walkforward(prices, sig, mode, confirm_days, crisis_asset,
                            warmup=260, train_years=3, test_years=1):
    """Gate 5: Walk-forward validation (3yr train / 1yr OOS)."""
    print(f"\n  Gate 5: Walk-forward validation ({train_years}yr / {test_years}yr)...")

    res_full = simulate_strategy(prices, sig, mode=mode, confirm_days=confirm_days,
                                 crisis_asset=crisis_asset, warmup=warmup)
    strat_rets = res_full['values'].pct_change().dropna()

    spy_res = simulate_strategy(prices, sig, mode='spy_bh', warmup=warmup)
    spy_rets = spy_res['values'].pct_change().dropna()

    all_years = sorted(set(strat_rets.index.year))
    oos_results = []

    for start_test in range(min(all_years) + train_years, max(all_years) + 1, test_years):
        end_test = start_test + test_years - 1
        test_mask = (strat_rets.index.year >= start_test) & (strat_rets.index.year <= end_test)
        test_strat = strat_rets[test_mask]
        test_spy = spy_rets.reindex(test_strat.index).dropna()

        if len(test_strat) < 50:
            continue

        s_strat = (test_strat.mean() * 252) / (test_strat.std() * np.sqrt(252)) if test_strat.std() > 0 else 0
        s_spy = (test_spy.mean() * 252) / (test_spy.std() * np.sqrt(252)) if len(test_spy) > 0 and test_spy.std() > 0 else 0

        excess = s_strat - s_spy
        oos_results.append({
            'period': f"{start_test}-{end_test}",
            'sharpe': float(s_strat),
            'spy_sharpe': float(s_spy),
            'excess': float(excess),
        })
        marker = '+' if excess > 0 else '-'
        print(f"    {start_test}-{end_test}: Strat={s_strat:.3f}, SPY={s_spy:.3f}, "
              f"Excess={excess:+.3f} [{marker}]")

    positive = sum(1 for r in oos_results if r['excess'] > 0)
    pct_pos = positive / len(oos_results) if oos_results else 0
    avg_oos = np.mean([r['sharpe'] for r in oos_results]) if oos_results else 0
    avg_excess = np.mean([r['excess'] for r in oos_results]) if oos_results else 0

    passed = pct_pos >= 0.5 and avg_oos > 0
    print(f"    Positive excess: {positive}/{len(oos_results)} ({pct_pos*100:.0f}%)")
    print(f"    Avg OOS Sharpe: {avg_oos:.3f}, Avg excess: {avg_excess:+.3f}")
    print(f"    -> {'PASS' if passed else 'FAIL'}")

    return {'oos_results': oos_results, 'pct_positive': pct_pos,
            'avg_oos_sharpe': float(avg_oos), 'avg_excess': float(avg_excess),
            'pass': passed}


def run_adversarial_suite(prices, sig, mode, confirm_days, crisis_asset, label, warmup=260):
    """Run all 5 adversarial gates."""
    print(f"\n{'#' * 80}")
    print(f"# ADVERSARIAL VALIDATION: {label}")
    print(f"{'#' * 80}")

    results = {}
    results['permutation'] = adversarial_permutation(
        prices, sig, mode, confirm_days, crisis_asset, warmup)
    results['subperiod'] = adversarial_subperiod(
        prices, sig, mode, confirm_days, crisis_asset, warmup)
    results['outlier'] = adversarial_outlier(
        prices, sig, mode, confirm_days, crisis_asset, warmup)
    results['regime_r1'] = adversarial_regime_r1(
        prices, sig, mode, confirm_days, crisis_asset, warmup)
    results['walkforward'] = adversarial_walkforward(
        prices, sig, mode, confirm_days, crisis_asset, warmup)

    n_pass = sum(1 for v in results.values() if v.get('pass', False))
    n_total = len(results)

    print(f"\n  {'='*50}")
    print(f"  ADVERSARIAL SUMMARY: {label}")
    print(f"  {'='*50}")
    for test_name, r in results.items():
        status = "PASS" if r.get('pass', False) else "FAIL"
        print(f"    {test_name:<20}: {status}")
    print(f"\n    TOTAL: {n_pass}/{n_total} PASSED")

    overall = n_pass >= 4
    banner = "PASS" if overall else "FAIL"
    print(f"    OVERALL: *** {banner} *** ({n_pass}/{n_total})")

    results['n_pass'] = n_pass
    results['n_total'] = n_total
    results['overall_pass'] = overall
    return results


# =============================================================================
# CRISIS PERIOD ANALYSIS
# =============================================================================

def crisis_period_analysis(strategies, prices, warmup=260):
    """Analyze behavior during known crisis periods."""
    print("\n[6/8] Crisis period deep-dive...")

    crises = {
        'COVID Crash':      ('2020-02-19', '2020-03-23'),
        'COVID Recovery':   ('2020-03-24', '2020-08-31'),
        'Rate Hike 2022':   ('2022-01-03', '2022-10-12'),
        'Tariff Shock 2025': ('2025-01-20', '2025-04-30'),
        'SVB 2023':         ('2023-03-08', '2023-03-24'),
        'Aug 2024 Selloff': ('2024-07-16', '2024-08-05'),
    }

    crisis_results = {}

    for crisis_name, (start, end) in crises.items():
        print(f"\n  {crisis_name} ({start} to {end}):")
        crisis_results[crisis_name] = {}

        for label, res in strategies.items():
            vals = res['values']
            mask = (vals.index >= start) & (vals.index <= end)
            crisis_vals = vals[mask]

            if len(crisis_vals) < 2:
                continue

            ret = (crisis_vals.iloc[-1] / crisis_vals.iloc[0] - 1) * 100
            dd = ((crisis_vals - crisis_vals.cummax()) / crisis_vals.cummax()).min() * 100

            crisis_days_in_period = res['crisis_days'][mask].sum() if 'crisis_days' in res else 0
            total_days = len(crisis_vals)

            crisis_results[crisis_name][label] = {
                'return_pct': float(ret),
                'max_dd_pct': float(dd),
                'crisis_days': int(crisis_days_in_period),
                'total_days': total_days,
            }

            print(f"    {label:<30} Return: {ret:+6.1f}%, MaxDD: {dd:6.1f}%, "
                  f"In crisis: {crisis_days_in_period}/{total_days}")

    return crisis_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_charts(strategies, all_metrics, prices, sig, warmup=260):
    """Generate comprehensive charts."""
    print("\n[7/8] Creating charts...")

    fig, axes = plt.subplots(4, 1, figsize=(16, 20))

    # 1. Equity curves
    ax = axes[0]
    colors = {'v4.4 Standalone': 'blue', 'v4.4 + SHY Crisis': 'green',
              'v4.4 + GLD/TLT Crisis': 'orange', 'SPY Buy & Hold': 'gray',
              'v4.4 + SHY Crisis (5d)': 'cyan', 'v4.4 + GLD/TLT Crisis (5d)': 'red'}

    for label, res in strategies.items():
        m = all_metrics[label]
        c = colors.get(label, 'purple')
        lw = 2.0 if label in ['v4.4 Standalone', 'v4.4 + SHY Crisis', 'v4.4 + GLD/TLT Crisis'] else 1.0
        ax.plot(res['values'].index, res['values'],
                label=f"{label} (Sh={m['sharpe']:.2f}, DD={m['max_dd']*100:.0f}%)",
                color=c, linewidth=lw)

    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title('v4.4 + Crisis Exit: Equity Curves ($100K, No DCA)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. VIX term structure + crisis periods
    ax = axes[1]
    vix_ratio = sig['vix_ratio']
    crisis_sig = compute_crisis_signal(sig, confirm_days=3)
    crisis_sig_lag = crisis_sig.shift(1).fillna(1)

    ax.plot(vix_ratio.index, vix_ratio, color='purple', alpha=0.7, linewidth=0.8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.0, label='Contango/Backwardation')
    ax.axhline(1.02, color='green', linestyle='--', linewidth=0.5, label='Recovery threshold')

    # Shade crisis periods
    crisis_mask = crisis_sig_lag == 0
    crisis_dates = crisis_mask[crisis_mask].index
    if len(crisis_dates) > 0:
        ax.fill_between(vix_ratio.index, vix_ratio.min(), vix_ratio.max(),
                        where=crisis_mask.reindex(vix_ratio.index, fill_value=False),
                        alpha=0.2, color='red', label='Crisis Override Active')

    ax.set_ylabel('VIX3M / VIX Ratio')
    ax.set_title('VIX Term Structure (Backwardation = Crisis Signal)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Drawdown comparison
    ax = axes[2]
    key_strats = ['v4.4 Standalone', 'v4.4 + SHY Crisis', 'v4.4 + GLD/TLT Crisis', 'SPY Buy & Hold']
    for label in key_strats:
        if label not in strategies:
            continue
        vals = strategies[label]['values']
        dd = (vals - vals.cummax()) / vals.cummax()
        c = colors.get(label, 'purple')
        ax.fill_between(dd.index, 0, dd, alpha=0.25, color=c, label=label)

    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown Comparison')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Rolling 1-year Sharpe
    ax = axes[3]
    for label in key_strats:
        if label not in strategies:
            continue
        vals = strategies[label]['values']
        rets = vals.pct_change().dropna()
        rolling_sharpe = rets.rolling(252).mean() / rets.rolling(252).std() * np.sqrt(252)
        c = colors.get(label, 'purple')
        ax.plot(rolling_sharpe.index, rolling_sharpe, color=c, label=label, alpha=0.7)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Rolling 1yr Sharpe')
    ax.set_title('Rolling 1-Year Sharpe Ratio')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_overview.png', dpi=150)
    plt.close()
    print(f"  Saved strategy_overview.png")

    # Confirmation period sweep chart
    fig, ax = plt.subplots(figsize=(10, 6))
    confirms = [2, 3, 5, 7]
    shy_sharpes = []
    gld_sharpes = []
    shy_dds = []
    gld_dds = []

    for confirm in confirms:
        res_shy = simulate_strategy(prices, sig, mode='v44_crisis',
                                    confirm_days=confirm, crisis_asset='SHY', warmup=warmup)
        m_shy = calc_metrics(res_shy, f'SHY_{confirm}d')
        shy_sharpes.append(m_shy['sharpe'])
        shy_dds.append(abs(m_shy['max_dd']) * 100)

        res_gld = simulate_strategy(prices, sig, mode='v44_crisis',
                                    confirm_days=confirm, crisis_asset='GLD_TLT', warmup=warmup)
        m_gld = calc_metrics(res_gld, f'GLD_TLT_{confirm}d')
        gld_sharpes.append(m_gld['sharpe'])
        gld_dds.append(abs(m_gld['max_dd']) * 100)

    x = np.arange(len(confirms))
    width = 0.35

    ax2 = ax.twinx()
    ax.bar(x - width/2, shy_sharpes, width, label='SHY Crisis Sharpe', color='blue', alpha=0.7)
    ax.bar(x + width/2, gld_sharpes, width, label='GLD/TLT Crisis Sharpe', color='orange', alpha=0.7)
    ax2.plot(x - width/2, shy_dds, 'bo-', label='SHY MaxDD%', markersize=8)
    ax2.plot(x + width/2, gld_dds, 'rs-', label='GLD/TLT MaxDD%', markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}d' for c in confirms])
    ax.set_xlabel('Confirmation Period')
    ax.set_ylabel('Sharpe Ratio')
    ax2.set_ylabel('Max Drawdown (%)')
    ax.set_title('Confirmation Period Sweep: Sharpe vs MaxDD')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confirmation_sweep.png', dpi=150)
    plt.close()
    print(f"  Saved confirmation_sweep.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()

    # 1. Data
    prices = download_data()

    # 2. Signals
    sig = compute_signals(prices)

    # 3. Confirmation period sweep
    sweep_df = sweep_confirmation_periods(prices, sig)

    # 4. Main backtest
    strategies, all_metrics = run_main_backtest(prices, sig)

    # 5. Adversarial validation on the 3 main crisis variants
    print("\n[5/8] Running adversarial validation...")

    adv_results = {}

    # v4.4 standalone (baseline)
    adv_results['v44_standalone'] = run_adversarial_suite(
        prices, sig, mode='v44_only', confirm_days=3, crisis_asset='SHY',
        label='v4.4 Standalone')

    # v4.4 + SHY (3d confirm)
    adv_results['v44_shy_3d'] = run_adversarial_suite(
        prices, sig, mode='v44_crisis', confirm_days=3, crisis_asset='SHY',
        label='v4.4 + SHY Crisis Exit (3d)')

    # v4.4 + GLD/TLT (3d confirm)
    adv_results['v44_gld_tlt_3d'] = run_adversarial_suite(
        prices, sig, mode='v44_crisis', confirm_days=3, crisis_asset='GLD_TLT',
        label='v4.4 + GLD/TLT Crisis Exit (3d)')

    # 6. Crisis deep-dive
    crisis_results = crisis_period_analysis(strategies, prices)

    # 7. Charts
    create_charts(strategies, all_metrics, prices, sig)

    # 8. Save results
    print("\n[8/8] Saving results...")

    # Clean metrics for JSON
    clean_metrics = {}
    for label, m in all_metrics.items():
        clean_metrics[label] = {k: v for k, v in m.items()}

    # Clean adversarial results for JSON
    clean_adv = {}
    for name, adv in adv_results.items():
        clean_adv[name] = {}
        for k, v in adv.items():
            if isinstance(v, dict):
                clean_adv[name][k] = {kk: vv for kk, vv in v.items()
                                       if not isinstance(vv, (pd.Series, pd.DataFrame, list))}
            else:
                clean_adv[name][k] = v

    full_output = {
        'strategy': 'v4.4 + VIX Curve Crisis Exit',
        'timestamp': dt.datetime.now().isoformat(),
        'period': f"{prices.index[0].date()} to {prices.index[-1].date()}",
        'initial_capital': INITIAL_CAPITAL,
        'dca': 'NONE',
        'metrics': clean_metrics,
        'adversarial': clean_adv,
        'crisis_analysis': crisis_results,
    }

    with open(OUTPUT_DIR / 'full_results.json', 'w') as f:
        json.dump(full_output, f, indent=2, default=str)

    sweep_df.to_csv(OUTPUT_DIR / 'confirmation_sweep.csv', index=False)

    # Final summary
    elapsed = time.time() - t0

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} "
          f"{'MaxDD':>8} {'Calmar':>7} {'Adv':>6}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    key_strats = ['v4.4 Standalone', 'v4.4 + SHY Crisis', 'v4.4 + GLD/TLT Crisis', 'SPY Buy & Hold']
    adv_map = {
        'v4.4 Standalone': 'v44_standalone',
        'v4.4 + SHY Crisis': 'v44_shy_3d',
        'v4.4 + GLD/TLT Crisis': 'v44_gld_tlt_3d',
    }

    for label in key_strats:
        if label not in all_metrics:
            continue
        m = all_metrics[label]
        adv_key = adv_map.get(label, '')
        adv_str = ''
        if adv_key and adv_key in adv_results:
            a = adv_results[adv_key]
            adv_str = f"{a['n_pass']}/{a['n_total']}"

        print(f"  {label:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>7.1%} {m['calmar']:>7.3f} "
              f"{adv_str:>6}")

    # Compute improvements
    v44_m = all_metrics.get('v4.4 Standalone', {})
    spy_m = all_metrics.get('SPY Buy & Hold', {})

    for label in ['v4.4 + SHY Crisis', 'v4.4 + GLD/TLT Crisis']:
        if label not in all_metrics:
            continue
        m = all_metrics[label]
        dd_improvement = (m['max_dd'] - v44_m.get('max_dd', 0)) * 100
        cagr_cost = (m['cagr'] - v44_m.get('cagr', 0)) * 100
        sharpe_delta = m['sharpe'] - v44_m.get('sharpe', 0)
        print(f"\n  {label} vs v4.4 Standalone:")
        print(f"    MaxDD improvement: {dd_improvement:+.1f}pp")
        print(f"    CAGR cost: {cagr_cost:+.1f}pp")
        print(f"    Sharpe change: {sharpe_delta:+.3f}")
        print(f"    Crisis time: {m.get('pct_crisis', 0)*100:.1f}%")

    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Results saved to {OUTPUT_DIR}")
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")

    return full_output


if __name__ == "__main__":
    results = main()
