#!/usr/bin/env python3
"""
IV Surface Shape Changes as Sector ETF Predictors
===================================================
Tests three hypotheses:
1. VIX term structure slope changes predict sector rotation
2. Cross-sector IV rank divergence predicts mean-reversion
3. Realized vs Implied volatility gap persistence predicts sector returns

Uses yfinance for data, 5-day holding period, permutation tests for significance.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────
SECTORS = ['XLF', 'XLE', 'XLU', 'XLK', 'XLY', 'XLP', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
DEFENSIVE = ['XLU', 'XLP', 'XLV']
CYCLICAL = ['XLF', 'XLE', 'XLK', 'XLY', 'XLI', 'XLB', 'XLC']
START = '2021-01-01'
END = '2026-08-15'
HOLD_DAYS = 5
VOL_WINDOW = 20
VOL_RANK_WINDOW = 252  # 1-year percentile lookback
PERM_ITERS = 1000

print("=" * 70)
print("IV SURFACE SHAPE → SECTOR ETF PREDICTOR RESEARCH")
print("=" * 70)

# ── Download Data ──────────────────────────────────────────────────────
print("\n[1/6] Downloading data...")

tickers = SECTORS + ['^VIX', '^VIX3M']
data = {}
for t in tickers:
    try:
        df = yf.download(t, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df
        print(f"  {t}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    except Exception as e:
        print(f"  {t}: FAILED - {e}")

# Build returns dataframe
closes = pd.DataFrame({t: data[t]['Close'] for t in SECTORS if t in data})
closes = closes.dropna()
returns_1d = closes.pct_change()
returns_5d = closes.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)  # forward 5-day return

print(f"\n  Common date range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
print(f"  Total trading days: {len(closes)}")


# ── Helper Functions ───────────────────────────────────────────────────
def compute_metrics(trade_returns, label=""):
    """Compute Sharpe, Sortino, PF, WR from array of trade returns."""
    if len(trade_returns) == 0:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'mean_ret': 0, 'label': label}

    tr = np.array(trade_returns)
    n = len(tr)
    mean_r = np.mean(tr)
    std_r = np.std(tr, ddof=1) if n > 1 else 1e-9

    # Annualize: ~50 5-day periods per year
    ann_factor = np.sqrt(252 / HOLD_DAYS)
    sharpe = (mean_r / std_r) * ann_factor if std_r > 1e-9 else 0

    downside = tr[tr < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_r / downside_std) * ann_factor if downside_std > 1e-9 else 0

    gains = tr[tr > 0].sum()
    losses = abs(tr[tr < 0].sum())
    pf = gains / losses if losses > 1e-9 else float('inf') if gains > 0 else 0

    wr = np.mean(tr > 0)

    return {
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'mean_ret_bps': round(mean_r * 10000, 2),
        'median_ret_bps': round(np.median(tr) * 10000, 2),
        'std_ret_bps': round(std_r * 10000, 2),
        'label': label
    }


def permutation_test(actual_returns, all_possible_returns, n_iters=PERM_ITERS):
    """Permutation test: is the signal's mean return significantly different from random?"""
    actual_mean = np.mean(actual_returns)
    n = len(actual_returns)
    if n == 0 or len(all_possible_returns) == 0:
        return 1.0

    count_better = 0
    all_arr = np.array(all_possible_returns)
    for _ in range(n_iters):
        perm_sample = np.random.choice(all_arr, size=n, replace=True)
        if np.mean(perm_sample) >= actual_mean:
            count_better += 1

    return count_better / n_iters


def regime_stratify(dates, trade_returns, spy_returns_1d):
    """Stratify trades by regime: green/red/flat days (based on SPY)."""
    results = {}
    for regime, cond_fn in [
        ('green', lambda r: r > 0.003),
        ('red', lambda r: r < -0.003),
        ('flat', lambda r: abs(r) <= 0.003)
    ]:
        mask = []
        for d in dates:
            if d in spy_returns_1d.index:
                mask.append(cond_fn(spy_returns_1d.loc[d]))
            else:
                mask.append(False)
        mask = np.array(mask)
        regime_rets = np.array(trade_returns)[mask]
        if len(regime_rets) > 5:
            results[regime] = compute_metrics(regime_rets, f"regime_{regime}")
        else:
            results[regime] = {'n_trades': len(regime_rets), 'sharpe': None, 'note': 'too few trades'}
    return results


# Get SPY for regime classification
spy = yf.download('SPY', start=START, end=END, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_ret_1d = spy['Close'].pct_change()

all_results = {}

# ══════════════════════════════════════════════════════════════════════
# TEST 1: VIX TERM STRUCTURE → SECTOR ROTATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[2/6] TEST 1: VIX Term Structure Slope → Sector Rotation")
print("=" * 70)

if '^VIX' in data and '^VIX3M' in data:
    vix = data['^VIX']['Close'].reindex(closes.index).ffill()
    vix3m = data['^VIX3M']['Close'].reindex(closes.index).ffill()

    vts_ratio = vix / vix3m  # >1 = backwardation (fear), <1 = contango (complacent)
    vts_ratio = vts_ratio.dropna()

    # Also compute rate of change
    vts_roc = vts_ratio.pct_change(5)  # 5-day change in ratio

    print(f"  VIX/VIX3M ratio stats: mean={vts_ratio.mean():.3f}, std={vts_ratio.std():.3f}")
    print(f"  Days in backwardation (>1.0): {(vts_ratio > 1.0).sum()} ({(vts_ratio > 1.0).mean()*100:.1f}%)")
    print(f"  Days in deep contango (<0.85): {(vts_ratio < 0.85).sum()} ({(vts_ratio < 0.85).mean()*100:.1f}%)")

    test1_results = {}

    # Signal 1a: Backwardation → long defensive, short cyclical
    for signal_name, condition, long_group, short_group in [
        ('backwardation_defense', lambda r: r > 1.0, DEFENSIVE, CYCLICAL),
        ('deep_contango_cyclical', lambda r: r < 0.85, CYCLICAL, DEFENSIVE),
        ('rapid_fear_spike', lambda roc: roc > 0.05, DEFENSIVE, CYCLICAL),  # 5% jump in ratio
        ('rapid_complacency', lambda roc: roc < -0.05, CYCLICAL, DEFENSIVE),
    ]:
        trade_rets = []
        trade_dates = []

        for i, dt in enumerate(closes.index):
            if dt not in vts_ratio.index:
                continue

            # Use ratio for first two, ROC for last two
            if 'rapid' in signal_name:
                if dt not in vts_roc.index or pd.isna(vts_roc.loc[dt]):
                    continue
                triggered = condition(vts_roc.loc[dt])
            else:
                triggered = condition(vts_ratio.loc[dt])

            if triggered:
                # Equal-weight long/short
                long_ret = returns_5d.loc[dt, [s for s in long_group if s in returns_5d.columns]].mean()
                short_ret = returns_5d.loc[dt, [s for s in short_group if s in returns_5d.columns]].mean()

                if pd.notna(long_ret) and pd.notna(short_ret):
                    trade_rets.append(long_ret - short_ret)  # L/S spread
                    trade_dates.append(dt)

        metrics = compute_metrics(trade_rets, signal_name)

        # Permutation test
        # Null: random 5-day L/S spread returns
        all_ls_rets = []
        for dt in closes.index:
            if dt in returns_5d.index:
                long_r = returns_5d.loc[dt, [s for s in long_group if s in returns_5d.columns]].mean()
                short_r = returns_5d.loc[dt, [s for s in short_group if s in returns_5d.columns]].mean()
                if pd.notna(long_r) and pd.notna(short_r):
                    all_ls_rets.append(long_r - short_r)

        p_val = permutation_test(trade_rets, all_ls_rets)
        metrics['p_value'] = round(p_val, 4)

        # Regime stratification
        metrics['regime'] = regime_stratify(trade_dates, trade_rets, spy_ret_1d)

        test1_results[signal_name] = metrics

        print(f"\n  {signal_name}:")
        print(f"    N={metrics['n_trades']}, Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
              f"PF={metrics['pf']}, WR={metrics['wr']:.1%}")
        print(f"    Mean={metrics['mean_ret_bps']}bps, p-value={p_val:.4f}")

    # Signal 1b: Long-only sector using VTS regime
    print("\n  --- Long-only sector signals by VTS regime ---")
    for regime_name, condition in [
        ('backwardation', lambda r: r > 1.0),
        ('contango', lambda r: r < 0.95),
    ]:
        for sector in SECTORS:
            trade_rets = []
            trade_dates = []
            for dt in closes.index:
                if dt not in vts_ratio.index:
                    continue
                if condition(vts_ratio.loc[dt]):
                    ret = returns_5d.loc[dt, sector] if sector in returns_5d.columns else np.nan
                    if pd.notna(ret):
                        trade_rets.append(ret)
                        trade_dates.append(dt)

            if len(trade_rets) > 20:
                m = compute_metrics(trade_rets, f"{regime_name}_{sector}")
                test1_results[f"{regime_name}_{sector}"] = m

    all_results['test1_vix_term_structure'] = test1_results
else:
    print("  SKIPPED: VIX or VIX3M data not available")
    all_results['test1_vix_term_structure'] = {'error': 'data unavailable'}


# ══════════════════════════════════════════════════════════════════════
# TEST 2: CROSS-SECTOR IV RANK DIVERGENCE → MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[3/6] TEST 2: Cross-Sector Vol Rank Divergence → Mean Reversion")
print("=" * 70)

# Compute realized vol for each sector
rv = {}
for s in SECTORS:
    if s in returns_1d.columns:
        rv[s] = returns_1d[s].rolling(VOL_WINDOW).std() * np.sqrt(252)

rv_df = pd.DataFrame(rv).dropna()

# Compute vol rank (percentile over trailing year)
vol_rank = pd.DataFrame(index=rv_df.index, columns=rv_df.columns, dtype=float)
for s in rv_df.columns:
    for i in range(VOL_RANK_WINDOW, len(rv_df)):
        window = rv_df[s].iloc[i-VOL_RANK_WINDOW:i]
        current = rv_df[s].iloc[i]
        vol_rank[s].iloc[i] = (window < current).mean()

vol_rank = vol_rank.dropna(how='all')
# Drop initial NaN rows
vol_rank = vol_rank.iloc[VOL_RANK_WINDOW:]

print(f"  Vol rank computed for {len(vol_rank)} days, {len(vol_rank.columns)} sectors")

# Cross-sectional z-score: how far is each sector's vol rank from the cross-sectional mean?
cs_mean = vol_rank.mean(axis=1)
cs_std = vol_rank.std(axis=1)
vol_zscore = vol_rank.sub(cs_mean, axis=0).div(cs_std, axis=0)

test2_results = {}

# Signal: go long sectors with z < -1.5 (unusually low vol rank = cheap), short z > 1.5 (expensive)
for threshold in [1.0, 1.5, 2.0]:
    trade_rets = []
    trade_dates = []
    trade_details = []

    for dt in vol_zscore.index:
        if dt not in returns_5d.index:
            continue

        z = vol_zscore.loc[dt].dropna()
        long_sectors = z[z < -threshold].index.tolist()
        short_sectors = z[z > threshold].index.tolist()

        if long_sectors or short_sectors:
            long_ret = returns_5d.loc[dt, long_sectors].mean() if long_sectors else 0
            short_ret = returns_5d.loc[dt, short_sectors].mean() if short_sectors else 0

            if pd.notna(long_ret) and pd.notna(short_ret):
                ls_ret = 0
                n_legs = 0
                if long_sectors:
                    ls_ret += long_ret
                    n_legs += 1
                if short_sectors:
                    ls_ret -= short_ret
                    n_legs += 1
                if n_legs > 0:
                    ls_ret /= n_legs  # average contribution
                    trade_rets.append(ls_ret)
                    trade_dates.append(dt)

    # Compute metrics
    metrics = compute_metrics(trade_rets, f"vol_divergence_z{threshold}")

    # Permutation test against random sector selection
    all_sector_rets = []
    for dt in returns_5d.index:
        for s in SECTORS:
            if s in returns_5d.columns:
                r = returns_5d.loc[dt, s]
                if pd.notna(r):
                    all_sector_rets.append(r)

    p_val = permutation_test(trade_rets, all_sector_rets)
    metrics['p_value'] = round(p_val, 4)
    metrics['regime'] = regime_stratify(trade_dates, trade_rets, spy_ret_1d)

    test2_results[f'z_threshold_{threshold}'] = metrics

    print(f"\n  Z-threshold = {threshold}:")
    print(f"    N={metrics['n_trades']}, Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
          f"PF={metrics['pf']}, WR={metrics['wr']:.1%}")
    print(f"    Mean={metrics['mean_ret_bps']}bps, p-value={p_val:.4f}")

# Also test long-only: buy cheap vol sectors
for threshold in [1.5, 2.0]:
    trade_rets = []
    trade_dates = []

    for dt in vol_zscore.index:
        if dt not in returns_5d.index:
            continue

        z = vol_zscore.loc[dt].dropna()
        long_sectors = z[z < -threshold].index.tolist()

        if long_sectors:
            ret = returns_5d.loc[dt, long_sectors].mean()
            if pd.notna(ret):
                trade_rets.append(ret)
                trade_dates.append(dt)

    metrics = compute_metrics(trade_rets, f"long_cheap_vol_z{threshold}")
    p_val = permutation_test(trade_rets, all_sector_rets)
    metrics['p_value'] = round(p_val, 4)
    metrics['regime'] = regime_stratify(trade_dates, trade_rets, spy_ret_1d)
    test2_results[f'long_cheap_vol_z{threshold}'] = metrics

    print(f"\n  Long cheap vol (z < -{threshold}):")
    print(f"    N={metrics['n_trades']}, Sharpe={metrics['sharpe']}, PF={metrics['pf']}, WR={metrics['wr']:.1%}")

all_results['test2_vol_rank_divergence'] = test2_results


# ══════════════════════════════════════════════════════════════════════
# TEST 3: VOLATILITY RISK PREMIUM (RV vs IV proxy) → SECTOR RETURNS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[4/6] TEST 3: Volatility Risk Premium → Sector Returns")
print("=" * 70)

# IV proxy: use 30-day forward realized vol as "what IV was pricing"
# Then VRP = IV_proxy - RV_current
# Better proxy: use VIX as market-wide IV, scale by sector beta to VIX

# Compute sector-specific VRP proxy:
# RV_20d = trailing 20-day realized vol
# FV_20d = forward 20-day realized vol (what vol actually turned out to be)
# VRP_proxy = RV_20d_rank - FV_20d_rank (positive = calm future relative to past)
# But we can't use FV in real-time. So instead:
# Use VIX level relative to sector RV as a proxy

vix_series = data['^VIX']['Close'].reindex(closes.index).ffill() if '^VIX' in data else None

test3_results = {}

if vix_series is not None:
    for s in SECTORS:
        if s not in rv_df.columns:
            continue

        # VRP proxy = VIX/100 - sector RV (annualized)
        # When VRP > 0, options are expensive relative to realized
        vrp = (vix_series / 100) - rv_df[s]
        vrp = vrp.dropna()

        # Percentile rank of VRP
        vrp_rank = pd.Series(index=vrp.index, dtype=float)
        for i in range(VOL_RANK_WINDOW, len(vrp)):
            window = vrp.iloc[i-VOL_RANK_WINDOW:i]
            vrp_rank.iloc[i] = (window < vrp.iloc[i]).mean()

        vrp_rank = vrp_rank.dropna()

        # Signal: High VRP (>70th pctile) → sector stays calm (short vol environment)
        # Low/negative VRP (<30th pctile) → big moves coming
        for signal_name, condition in [
            (f'high_vrp_{s}', lambda r: r > 0.70),
            (f'low_vrp_{s}', lambda r: r < 0.30),
        ]:
            trade_rets = []
            trade_dates = []

            for dt in vrp_rank.index:
                if pd.isna(vrp_rank.loc[dt]) or dt not in returns_5d.index:
                    continue
                if condition(vrp_rank.loc[dt]):
                    ret = returns_5d.loc[dt, s] if s in returns_5d.columns else np.nan
                    if pd.notna(ret):
                        trade_rets.append(ret)
                        trade_dates.append(dt)

            if len(trade_rets) > 20:
                metrics = compute_metrics(trade_rets, signal_name)

                # Also compute realized vol of the 5-day forward returns
                fwd_vol = np.std(trade_rets) * np.sqrt(252/HOLD_DAYS)
                metrics['fwd_realized_vol_ann'] = round(fwd_vol, 4)

                test3_results[signal_name] = metrics

    # Aggregate cross-sector VRP signal
    print("\n  --- Aggregate VRP Signal (cross-sector) ---")

    # Compute mean VRP rank across sectors
    vrp_ranks_all = {}
    for s in SECTORS:
        if s not in rv_df.columns:
            continue
        vrp = (vix_series / 100) - rv_df[s]
        vrp = vrp.dropna()
        vrp_rank_s = pd.Series(index=vrp.index, dtype=float)
        for i in range(VOL_RANK_WINDOW, len(vrp)):
            window = vrp.iloc[i-VOL_RANK_WINDOW:i]
            vrp_rank_s.iloc[i] = (window < vrp.iloc[i]).mean()
        vrp_ranks_all[s] = vrp_rank_s

    vrp_rank_df = pd.DataFrame(vrp_ranks_all).dropna(how='all')

    # For each sector, when its VRP rank is in top quartile, go long (calm expected)
    # When in bottom quartile, avoid (vol expansion expected)
    for direction, cond, label in [
        ('long_high_vrp', lambda r: r > 0.75, 'VRP>75pct → long (calm expected)'),
        ('long_low_vrp', lambda r: r < 0.25, 'VRP<25pct → long (vol expansion)'),
    ]:
        trade_rets = []
        trade_dates = []

        for dt in vrp_rank_df.index:
            if dt not in returns_5d.index:
                continue

            ranks = vrp_rank_df.loc[dt].dropna()
            selected = ranks[ranks.apply(cond)].index.tolist()

            if selected:
                ret = returns_5d.loc[dt, selected].mean()
                if pd.notna(ret):
                    trade_rets.append(ret)
                    trade_dates.append(dt)

        if len(trade_rets) > 20:
            metrics = compute_metrics(trade_rets, direction)
            p_val = permutation_test(trade_rets, all_sector_rets)
            metrics['p_value'] = round(p_val, 4)
            metrics['regime'] = regime_stratify(trade_dates, trade_rets, spy_ret_1d)
            test3_results[direction] = metrics

            print(f"\n  {label}:")
            print(f"    N={metrics['n_trades']}, Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
                  f"PF={metrics['pf']}, WR={metrics['wr']:.1%}")
            print(f"    Mean={metrics['mean_ret_bps']}bps, p-value={p_val:.4f}")

    # Key test: Does high VRP predict LOWER forward vol? (i.e., is VRP actually informative?)
    print("\n  --- VRP Predictiveness: Does high VRP → lower forward vol? ---")
    for s in SECTORS[:3]:  # Sample
        if s not in rv_df.columns:
            continue
        vrp = (vix_series / 100) - rv_df[s]
        fwd_vol = returns_1d[s].rolling(VOL_WINDOW).std().shift(-VOL_WINDOW) * np.sqrt(252)

        merged = pd.DataFrame({'vrp': vrp, 'fwd_vol': fwd_vol}).dropna()
        if len(merged) > 100:
            corr = merged['vrp'].corr(merged['fwd_vol'])
            high_vrp_fwd = merged[merged['vrp'] > merged['vrp'].quantile(0.75)]['fwd_vol'].mean()
            low_vrp_fwd = merged[merged['vrp'] < merged['vrp'].quantile(0.25)]['fwd_vol'].mean()
            print(f"    {s}: VRP↔fwd_vol corr={corr:.3f}, high_VRP_fwd_vol={high_vrp_fwd:.3f}, low_VRP_fwd_vol={low_vrp_fwd:.3f}")

else:
    print("  SKIPPED: VIX data unavailable")

all_results['test3_vrp'] = test3_results


# ══════════════════════════════════════════════════════════════════════
# TEST 4 (BONUS): COMBINED SIGNAL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[5/6] COMBINED SIGNAL: VTS + Vol Divergence")
print("=" * 70)

# Combine: In backwardation AND sector has low vol z-score → strong long signal
if '^VIX' in data and '^VIX3M' in data:
    combo_rets = []
    combo_dates = []

    for dt in vol_zscore.index:
        if dt not in vts_ratio.index or dt not in returns_5d.index:
            continue

        in_backwardation = vts_ratio.loc[dt] > 1.0
        z = vol_zscore.loc[dt].dropna()

        if in_backwardation:
            # In fear: long defensive with low vol, short cyclical with high vol
            long_secs = [s for s in DEFENSIVE if s in z.index and z[s] < -1.0]
            short_secs = [s for s in CYCLICAL if s in z.index and z[s] > 1.0]
        else:
            # In calm: long cyclical with low vol, short defensive with high vol
            long_secs = [s for s in CYCLICAL if s in z.index and z[s] < -1.0]
            short_secs = [s for s in DEFENSIVE if s in z.index and z[s] > 1.0]

        if long_secs or short_secs:
            l_ret = returns_5d.loc[dt, long_secs].mean() if long_secs else 0
            s_ret = returns_5d.loc[dt, short_secs].mean() if short_secs else 0

            if pd.notna(l_ret) and pd.notna(s_ret):
                combo_rets.append(l_ret - s_ret)
                combo_dates.append(dt)

    if combo_rets:
        metrics = compute_metrics(combo_rets, 'combined_vts_voldiv')
        p_val = permutation_test(combo_rets, all_sector_rets)
        metrics['p_value'] = round(p_val, 4)
        metrics['regime'] = regime_stratify(combo_dates, combo_rets, spy_ret_1d)
        all_results['test4_combined'] = metrics

        print(f"  Combined signal:")
        print(f"    N={metrics['n_trades']}, Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
              f"PF={metrics['pf']}, WR={metrics['wr']:.1%}")
        print(f"    Mean={metrics['mean_ret_bps']}bps, p-value={p_val:.4f}")


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[6/6] SUMMARY — TOP SIGNALS BY SHARPE")
print("=" * 70)

# Collect all signal metrics
all_signals = []
for test_name, test_data in all_results.items():
    if isinstance(test_data, dict):
        if 'n_trades' in test_data:
            # Single result
            test_data['test'] = test_name
            all_signals.append(test_data)
        else:
            # Nested results
            for sig_name, sig_data in test_data.items():
                if isinstance(sig_data, dict) and 'n_trades' in sig_data:
                    sig_data['test'] = test_name
                    sig_data['signal'] = sig_name
                    all_signals.append(sig_data)

# Sort by Sharpe
all_signals_valid = [s for s in all_signals if s.get('n_trades', 0) > 20 and s.get('sharpe') is not None]
all_signals_valid.sort(key=lambda x: abs(x.get('sharpe', 0)), reverse=True)

print(f"\n  Total signals tested: {len(all_signals_valid)}")
print(f"\n  {'Signal':<40} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} {'Mean':>7} {'p-val':>6}")
print(f"  {'-'*40} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")

for s in all_signals_valid[:20]:
    name = s.get('signal', s.get('label', s.get('test', '?')))[:40]
    p = s.get('p_value', '-')
    p_str = f"{p:.3f}" if isinstance(p, (int, float)) else str(p)
    print(f"  {name:<40} {s['n_trades']:>5} {s['sharpe']:>7.3f} {s['sortino']:>7.3f} {s['pf']:>6.3f} "
          f"{s['wr']:>5.1%} {s['mean_ret_bps']:>6.1f} {p_str:>6}")

# Determine significance
sig_signals = [s for s in all_signals_valid if isinstance(s.get('p_value'), (int, float)) and s['p_value'] < 0.05]
print(f"\n  Signals with p < 0.05: {len(sig_signals)} / {len(all_signals_valid)}")

# Determine if any signal survives all tests
print("\n  VERDICT:")
if sig_signals:
    best = sig_signals[0]
    name = best.get('signal', best.get('label', best.get('test', '?')))
    print(f"  Best statistically significant signal: {name}")
    print(f"    Sharpe={best['sharpe']}, WR={best['wr']:.1%}, p={best['p_value']:.4f}")

    # Check regime robustness
    if 'regime' in best:
        regime = best['regime']
        regime_sharpes = []
        for r_name, r_data in regime.items():
            if isinstance(r_data, dict) and r_data.get('sharpe') is not None:
                regime_sharpes.append((r_name, r_data['sharpe']))
                print(f"    {r_name}: Sharpe={r_data['sharpe']}, N={r_data.get('n_trades', '?')}")

        if len(regime_sharpes) >= 2:
            sharpe_vals = [abs(x[1]) for x in regime_sharpes]
            max_s = max(sharpe_vals) if sharpe_vals else 0
            if max_s > 0:
                regime_disparity = (max(sharpe_vals) - min(sharpe_vals)) / max_s
                print(f"    Regime disparity: {regime_disparity:.2f} (reject if >0.50)")
                if regime_disparity > 0.50:
                    print(f"    ⚠ REGIME-DEPENDENT — signal may be regime-tailored, not robust edge")
else:
    print(f"  NO signals achieved p < 0.05. IV surface shape changes (as proxied here)")
    print(f"  do NOT robustly predict sector ETF returns at the 5-day horizon.")

# Clean results for JSON serialization
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return None
    return obj

output = {
    'metadata': {
        'run_date': datetime.now().isoformat(),
        'data_range': f"{closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}",
        'trading_days': len(closes),
        'hold_period_days': HOLD_DAYS,
        'permutation_iterations': PERM_ITERS,
        'sectors': SECTORS
    },
    'results': clean_for_json(all_results),
    'top_signals': clean_for_json(all_signals_valid[:10]),
    'significant_signals': clean_for_json(sig_signals)
}

output_path = '/home/jupiter/Lvl3Quant/research/iv_surface_signals_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Results saved to {output_path}")
print("\nDONE.")
