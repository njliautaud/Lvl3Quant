#!/usr/bin/env python3
"""
Adversarial Validation for 3 Passing Correlation Breakdown Strategies
Tests: Inverse Signal, Random Timing, Cost Sensitivity, Sub-Period Stability,
       Parameter Robustness, Top-3 Sector Concentration
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

SECTOR_ETFS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLE', 'XLU', 'XLB', 'XLRE', 'XLV']
BENCHMARK = 'SPY'
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
BASE_COST_PCT = 0.001  # 0.10% RT
N_RANDOM_SHUFFLES = 1000

RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/scripts/growth_research/results/correlation_breakdown_adversarial.json')


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download all required price data."""
    tickers = SECTOR_ETFS + [BENCHMARK]
    print(f"Downloading {len(tickers)} tickers from {START_DATE} to {END_DATE}...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    close = data['Close'].dropna()
    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")
    return close


# ─── Signal Computation ─────────────────────────────────────────────────────

def compute_features(close):
    """Compute all features needed for the 3 strategies."""
    ret_5d = close.pct_change(5)

    # Cross-sectional dispersion of 5-day returns (sector ETFs only)
    sector_ret_5d = ret_5d[SECTOR_ETFS]
    dispersion = sector_ret_5d.std(axis=1)

    # RSI(14) for each sector
    rsi = pd.DataFrame(index=close.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        delta = close[etf].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi[etf] = 100 - (100 / (1 + rs))
    rsi = rsi.astype(float)

    # 20-day rolling correlation with SPY for each sector
    corr_20d = pd.DataFrame(index=close.index, columns=SECTOR_ETFS)
    spy_ret = close[BENCHMARK].pct_change()
    for etf in SECTOR_ETFS:
        etf_ret = close[etf].pct_change()
        corr_20d[etf] = etf_ret.rolling(20).corr(spy_ret)
    corr_20d = corr_20d.astype(float)

    return ret_5d, dispersion, rsi, corr_20d


# ─── Strategy Execution Engine ──────────────────────────────────────────────

def run_strategy_A(close, dispersion, rsi, disp_pct=80, rsi_thresh=35, hold_days=5,
                   tp=0.03, sl=-0.05, cost_pct=0.001, inverse=False):
    """High Dispersion Dip-Buy strategy."""
    disp_threshold = dispersion.rolling(252, min_periods=60).quantile(disp_pct / 100.0)

    trades = []
    for i in range(60, len(close)):
        date = close.index[i]
        d_val = dispersion.iloc[i]
        d_thresh = disp_threshold.iloc[i]

        if pd.isna(d_val) or pd.isna(d_thresh):
            continue

        if not inverse:
            # Normal: high dispersion + low RSI
            if d_val <= d_thresh:
                continue
            candidates = [etf for etf in SECTOR_ETFS
                         if not pd.isna(rsi[etf].iloc[i]) and rsi[etf].iloc[i] < rsi_thresh]
        else:
            # Inverse: low dispersion + high RSI
            if d_val >= d_thresh:
                continue
            candidates = [etf for etf in SECTOR_ETFS
                         if not pd.isna(rsi[etf].iloc[i]) and rsi[etf].iloc[i] > (100 - rsi_thresh)]

        if not candidates:
            continue

        if not inverse:
            best = min(candidates, key=lambda e: rsi[e].iloc[i])
        else:
            best = max(candidates, key=lambda e: rsi[e].iloc[i])

        entry_price = close[best].iloc[i]
        exit_idx = min(i + hold_days, len(close) - 1)

        # Check TP/SL during holding period
        final_exit_price = close[best].iloc[exit_idx]
        exit_price = final_exit_price
        for j in range(i + 1, exit_idx + 1):
            p = close[best].iloc[j]
            ret = (p - entry_price) / entry_price
            if ret >= tp:
                exit_price = entry_price * (1 + tp)
                break
            elif ret <= sl:
                exit_price = entry_price * (1 + sl)
                break
        else:
            exit_price = final_exit_price

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - cost_pct
        trades.append({'date': date, 'sector': best, 'return': net_ret})

    return trades


def run_strategy_B(close, corr_20d, rsi, corr_thresh=0.5, rsi_thresh=35, hold_days=5,
                   tp=0.03, sl=-0.05, cost_pct=0.001, inverse=False):
    """Correlation Breakdown strategy."""
    trades = []
    for i in range(60, len(close)):
        date = close.index[i]

        for etf in SECTOR_ETFS:
            corr_val = corr_20d[etf].iloc[i]
            rsi_val = rsi[etf].iloc[i]

            if pd.isna(corr_val) or pd.isna(rsi_val):
                continue

            if not inverse:
                # Normal: low correlation + low RSI
                if corr_val >= corr_thresh or rsi_val >= rsi_thresh:
                    continue
            else:
                # Inverse: high correlation + high RSI
                if corr_val <= corr_thresh or rsi_val <= (100 - rsi_thresh):
                    continue

            entry_price = close[etf].iloc[i]
            exit_idx = min(i + hold_days, len(close) - 1)

            final_exit_price = close[etf].iloc[exit_idx]
            exit_price = final_exit_price
            for j in range(i + 1, exit_idx + 1):
                p = close[etf].iloc[j]
                ret = (p - entry_price) / entry_price
                if ret >= tp:
                    exit_price = entry_price * (1 + tp)
                    break
                elif ret <= sl:
                    exit_price = entry_price * (1 + sl)
                    break
            else:
                exit_price = final_exit_price

            gross_ret = (exit_price - entry_price) / entry_price
            net_ret = gross_ret - cost_pct
            trades.append({'date': date, 'sector': etf, 'return': net_ret})

    return trades


def run_strategy_F(close, dispersion, rsi, disp_pct=80, rsi_thresh=35, hold_days=5,
                   tp=0.03, sl=-0.05, cost_pct=0.001, inverse=False):
    """Combined Bounce strategy (A + bounce confirmation)."""
    disp_threshold = dispersion.rolling(252, min_periods=60).quantile(disp_pct / 100.0)

    # 10-day low for bounce confirmation
    low_10d = close[SECTOR_ETFS].rolling(10).min()

    trades = []
    for i in range(60, len(close)):
        date = close.index[i]
        d_val = dispersion.iloc[i]
        d_thresh = disp_threshold.iloc[i]

        if pd.isna(d_val) or pd.isna(d_thresh):
            continue

        if not inverse:
            if d_val <= d_thresh:
                continue
            candidates = [etf for etf in SECTOR_ETFS
                         if not pd.isna(rsi[etf].iloc[i]) and rsi[etf].iloc[i] < rsi_thresh]
        else:
            if d_val >= d_thresh:
                continue
            candidates = [etf for etf in SECTOR_ETFS
                         if not pd.isna(rsi[etf].iloc[i]) and rsi[etf].iloc[i] > (100 - rsi_thresh)]

        if not candidates:
            continue

        # Bounce confirmation: yesterday near 10d low, today closes higher
        bounce_candidates = []
        for etf in candidates:
            if i < 1:
                continue
            yesterday_close = close[etf].iloc[i - 1]
            today_close = close[etf].iloc[i]
            ten_d_low = low_10d[etf].iloc[i - 1] if not pd.isna(low_10d[etf].iloc[i - 1]) else None

            if ten_d_low is None:
                continue

            if not inverse:
                # Normal: yesterday near 10d low, today higher
                near_low = (yesterday_close - ten_d_low) / ten_d_low < 0.01  # within 1%
                bounced = today_close > yesterday_close
                if near_low and bounced:
                    bounce_candidates.append(etf)
            else:
                # Inverse: skip bounce check for inverse
                bounce_candidates.append(etf)

        if not bounce_candidates:
            continue

        if not inverse:
            best = min(bounce_candidates, key=lambda e: rsi[e].iloc[i])
        else:
            best = max(bounce_candidates, key=lambda e: rsi[e].iloc[i])

        entry_price = close[best].iloc[i]
        exit_idx = min(i + hold_days, len(close) - 1)

        final_exit_price = close[best].iloc[exit_idx]
        exit_price = final_exit_price
        for j in range(i + 1, exit_idx + 1):
            p = close[best].iloc[j]
            ret = (p - entry_price) / entry_price
            if ret >= tp:
                exit_price = entry_price * (1 + tp)
                break
            elif ret <= sl:
                exit_price = entry_price * (1 + sl)
                break
        else:
            exit_price = final_exit_price

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - cost_pct
        trades.append({'date': date, 'sector': best, 'return': net_ret})

    return trades


# ─── Metric Computation ─────────────────────────────────────────────────────

def compute_sharpe(trades):
    """Annualized Sharpe from trade returns."""
    if len(trades) < 5:
        return 0.0
    rets = [t['return'] for t in trades]
    mean = np.mean(rets)
    std = np.std(rets, ddof=1)
    if std == 0:
        return 0.0
    # Annualize: assume ~50 trades/year as rough scaling
    trades_per_year = max(len(rets) / 4.5, 1)  # ~4.5 years of data
    return (mean / std) * np.sqrt(trades_per_year)


def compute_win_rate(trades):
    if len(trades) == 0:
        return 0.0
    wins = sum(1 for t in trades if t['return'] > 0)
    return wins / len(trades)


# ─── Adversarial Tests ──────────────────────────────────────────────────────

def test_inverse_signal(run_func, close, features, orig_sharpe, **kwargs):
    """Test 1: Flip entry direction. FAIL if inverse Sharpe > 0.5x original."""
    trades_inv = run_func(close, *features, inverse=True, **kwargs)
    inv_sharpe = compute_sharpe(trades_inv)
    ratio = inv_sharpe / orig_sharpe if orig_sharpe != 0 else float('inf')
    passed = ratio < 0.50
    return {
        'test': 'Inverse Signal',
        'original_sharpe': round(orig_sharpe, 3),
        'inverse_sharpe': round(inv_sharpe, 3),
        'inverse_trades': len(trades_inv),
        'ratio': round(ratio, 3),
        'threshold': '< 0.50',
        'PASS': passed
    }


def test_random_timing(trades, close, n_shuffles=N_RANDOM_SHUFFLES):
    """Test 2: Randomize entry dates, keep trade count. Must be > 90th pctile."""
    if len(trades) < 5:
        return {'test': 'Random Timing', 'PASS': False, 'reason': 'Too few trades'}

    real_sharpe = compute_sharpe(trades)
    n_trades = len(trades)

    # Get all available dates and sectors from trades
    all_dates = close.index[60:]
    random_sharpes = []

    for _ in range(n_shuffles):
        random_trades = []
        rand_indices = np.random.choice(len(all_dates), size=n_trades, replace=True)
        for idx in rand_indices:
            i = idx + 60  # offset for feature computation period
            if i >= len(close) - 5:
                continue
            etf = np.random.choice(SECTOR_ETFS)
            entry_price = close[etf].iloc[i]
            exit_idx = min(i + 5, len(close) - 1)
            exit_price = close[etf].iloc[exit_idx]
            ret = (exit_price - entry_price) / entry_price - BASE_COST_PCT
            random_trades.append({'return': ret})

        if len(random_trades) >= 5:
            random_sharpes.append(compute_sharpe(random_trades))

    if not random_sharpes:
        return {'test': 'Random Timing', 'PASS': False, 'reason': 'Could not generate random trades'}

    percentile = np.mean([1 for rs in random_sharpes if real_sharpe > rs]) * 100
    passed = percentile > 90

    return {
        'test': 'Random Timing',
        'real_sharpe': round(real_sharpe, 3),
        'random_mean_sharpe': round(np.mean(random_sharpes), 3),
        'random_median_sharpe': round(np.median(random_sharpes), 3),
        'percentile_rank': round(percentile, 1),
        'threshold': '> 90th percentile',
        'PASS': passed
    }


def test_cost_sensitivity(run_func, close, features, **kwargs):
    """Test 3: Run at multiple cost levels. PASS if Sharpe > 0.3 at 0.30%."""
    cost_levels = [0.001, 0.002, 0.003, 0.005]
    results = {}
    for cost in cost_levels:
        kw = {k: v for k, v in kwargs.items() if k != 'cost_pct'}
        trades = run_func(close, *features, cost_pct=cost, **kw)
        sharpe = compute_sharpe(trades)
        results[f'{cost*100:.1f}%'] = {'sharpe': round(sharpe, 3), 'n_trades': len(trades)}

    sharpe_at_030 = results['0.3%']['sharpe']
    passed = sharpe_at_030 > 0.3

    return {
        'test': 'Cost Sensitivity',
        'cost_results': results,
        'sharpe_at_030pct': round(sharpe_at_030, 3),
        'threshold': 'Sharpe > 0.3 at 0.30% cost',
        'PASS': passed
    }


def test_subperiod_stability(trades):
    """Test 4: Split into 4 equal periods. PASS if 3/4 have positive Sharpe."""
    if len(trades) < 20:
        return {'test': 'Sub-Period Stability', 'PASS': False, 'reason': 'Too few trades'}

    dates = [t['date'] for t in trades]
    min_date, max_date = min(dates), max(dates)
    total_days = (max_date - min_date).days
    quarter = total_days / 4

    period_sharpes = []
    for q in range(4):
        start = min_date + pd.Timedelta(days=int(quarter * q))
        end = min_date + pd.Timedelta(days=int(quarter * (q + 1)))
        period_trades = [t for t in trades if start <= t['date'] < end]
        if len(period_trades) >= 3:
            sharpe = compute_sharpe(period_trades)
        else:
            sharpe = 0.0
        period_sharpes.append(round(sharpe, 3))

    positive_periods = sum(1 for s in period_sharpes if s > 0)
    passed = positive_periods >= 3

    return {
        'test': 'Sub-Period Stability',
        'period_sharpes': period_sharpes,
        'positive_periods': positive_periods,
        'threshold': '>= 3 of 4 periods positive',
        'PASS': passed
    }


def test_parameter_robustness(strategy_type, close, features, run_func):
    """Test 5: Grid of parameter variations. PASS if >50% combos have Sharpe > 0.3."""
    if strategy_type in ('A', 'F'):
        disp_pcts = [70, 75, 80, 85, 90]
        rsi_thresholds = [30, 35, 40]
        hold_days_list = [3, 5, 7, 10]
    else:  # B
        corr_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        rsi_thresholds = [30, 35, 40]
        hold_days_list = [3, 5, 7, 10]

    total = 0
    above_threshold = 0
    all_sharpes = []

    if strategy_type in ('A', 'F'):
        for dp in disp_pcts:
            for rst in rsi_thresholds:
                for hd in hold_days_list:
                    trades = run_func(close, *features, disp_pct=dp, rsi_thresh=rst,
                                     hold_days=hd, cost_pct=BASE_COST_PCT)
                    sharpe = compute_sharpe(trades)
                    all_sharpes.append(sharpe)
                    total += 1
                    if sharpe > 0.3:
                        above_threshold += 1
    else:
        dispersion, rsi_df, corr_20d = features
        for ct in corr_thresholds:
            for rst in rsi_thresholds:
                for hd in hold_days_list:
                    trades = run_func(close, corr_20d, rsi_df, corr_thresh=ct, rsi_thresh=rst,
                                     hold_days=hd, cost_pct=BASE_COST_PCT)
                    sharpe = compute_sharpe(trades)
                    all_sharpes.append(sharpe)
                    total += 1
                    if sharpe > 0.3:
                        above_threshold += 1

    pct_above = above_threshold / total * 100 if total > 0 else 0
    passed = pct_above > 50

    return {
        'test': 'Parameter Robustness',
        'total_combos': total,
        'combos_above_0.3': above_threshold,
        'pct_above': round(pct_above, 1),
        'mean_sharpe': round(np.mean(all_sharpes), 3),
        'median_sharpe': round(np.median(all_sharpes), 3),
        'threshold': '> 50% combos with Sharpe > 0.3',
        'PASS': passed
    }


def test_sector_concentration(trades):
    """Test 6: Check if >50% trades from 3 sectors. FAIL if > 70%."""
    if len(trades) < 10:
        return {'test': 'Sector Concentration', 'PASS': False, 'reason': 'Too few trades'}

    sector_counts = Counter(t['sector'] for t in trades)
    top3 = sector_counts.most_common(3)
    top3_count = sum(c for _, c in top3)
    top3_pct = top3_count / len(trades) * 100

    passed = top3_pct <= 70

    return {
        'test': 'Top-3 Sector Concentration',
        'top3_sectors': [(s, c, round(c/len(trades)*100, 1)) for s, c in top3],
        'top3_concentration_pct': round(top3_pct, 1),
        'total_trades': len(trades),
        'threshold': '<= 70%',
        'PASS': passed
    }


# ─── Main Execution ─────────────────────────────────────────────────────────

def run_all_tests():
    # Download data
    close = download_data()

    # Compute features
    print("\nComputing features...")
    ret_5d, dispersion, rsi, corr_20d = compute_features(close)

    results = {}

    # ── Strategy A: High Dispersion Dip-Buy ──────────────────────────────
    print("\n" + "="*70)
    print("STRATEGY A: High Dispersion Dip-Buy")
    print("="*70)

    trades_A = run_strategy_A(close, dispersion, rsi, cost_pct=BASE_COST_PCT)
    sharpe_A = compute_sharpe(trades_A)
    wr_A = compute_win_rate(trades_A)
    print(f"  Baseline: {len(trades_A)} trades, Sharpe={sharpe_A:.3f}, WR={wr_A:.1%}")

    features_A = (dispersion, rsi)

    print("  [1/6] Inverse Signal...")
    t1 = test_inverse_signal(run_strategy_A, close, features_A, sharpe_A)
    print(f"    {'PASS' if t1['PASS'] else 'FAIL'} — Inverse Sharpe={t1['inverse_sharpe']}, Ratio={t1['ratio']}")

    print("  [2/6] Random Timing (1000 shuffles)...")
    t2 = test_random_timing(trades_A, close)
    print(f"    {'PASS' if t2['PASS'] else 'FAIL'} — Percentile={t2.get('percentile_rank', 'N/A')}")

    print("  [3/6] Cost Sensitivity...")
    t3 = test_cost_sensitivity(run_strategy_A, close, features_A)
    print(f"    {'PASS' if t3['PASS'] else 'FAIL'} — Sharpe@0.30%={t3['sharpe_at_030pct']}")

    print("  [4/6] Sub-Period Stability...")
    t4 = test_subperiod_stability(trades_A)
    print(f"    {'PASS' if t4['PASS'] else 'FAIL'} — Positive periods={t4.get('positive_periods', 'N/A')}/4")

    print("  [5/6] Parameter Robustness (60 combos)...")
    t5 = test_parameter_robustness('A', close, features_A, run_strategy_A)
    print(f"    {'PASS' if t5['PASS'] else 'FAIL'} — {t5['pct_above']}% above 0.3")

    print("  [6/6] Sector Concentration...")
    t6 = test_sector_concentration(trades_A)
    print(f"    {'PASS' if t6['PASS'] else 'FAIL'} — Top-3 concentration={t6.get('top3_concentration_pct', 'N/A')}%")

    tests_A = [t1, t2, t3, t4, t5, t6]
    passes_A = sum(1 for t in tests_A if t['PASS'])
    verdict_A = 'VALIDATED' if passes_A >= 5 else 'REJECTED'

    results['A_High_Dispersion_DipBuy'] = {
        'baseline': {'trades': len(trades_A), 'sharpe': round(sharpe_A, 3), 'win_rate': round(wr_A, 3)},
        'tests': tests_A,
        'passes': passes_A,
        'total_tests': 6,
        'verdict': verdict_A
    }

    # ── Strategy B: Correlation Breakdown ────────────────────────────────
    print("\n" + "="*70)
    print("STRATEGY B: Correlation Breakdown")
    print("="*70)

    trades_B = run_strategy_B(close, corr_20d, rsi, cost_pct=BASE_COST_PCT)
    sharpe_B = compute_sharpe(trades_B)
    wr_B = compute_win_rate(trades_B)
    print(f"  Baseline: {len(trades_B)} trades, Sharpe={sharpe_B:.3f}, WR={wr_B:.1%}")

    features_B = (dispersion, rsi, corr_20d)  # dispersion not used by B but kept for interface

    print("  [1/6] Inverse Signal...")
    t1 = test_inverse_signal(run_strategy_B, close, (corr_20d, rsi), sharpe_B)
    print(f"    {'PASS' if t1['PASS'] else 'FAIL'} — Inverse Sharpe={t1['inverse_sharpe']}, Ratio={t1['ratio']}")

    print("  [2/6] Random Timing (1000 shuffles)...")
    t2 = test_random_timing(trades_B, close)
    print(f"    {'PASS' if t2['PASS'] else 'FAIL'} — Percentile={t2.get('percentile_rank', 'N/A')}")

    print("  [3/6] Cost Sensitivity...")
    t3 = test_cost_sensitivity(run_strategy_B, close, (corr_20d, rsi))
    print(f"    {'PASS' if t3['PASS'] else 'FAIL'} — Sharpe@0.30%={t3['sharpe_at_030pct']}")

    print("  [4/6] Sub-Period Stability...")
    t4 = test_subperiod_stability(trades_B)
    print(f"    {'PASS' if t4['PASS'] else 'FAIL'} — Positive periods={t4.get('positive_periods', 'N/A')}/4")

    print("  [5/6] Parameter Robustness (60 combos)...")
    t5 = test_parameter_robustness('B', close, features_B, run_strategy_B)
    print(f"    {'PASS' if t5['PASS'] else 'FAIL'} — {t5['pct_above']}% above 0.3")

    print("  [6/6] Sector Concentration...")
    t6 = test_sector_concentration(trades_B)
    print(f"    {'PASS' if t6['PASS'] else 'FAIL'} — Top-3 concentration={t6.get('top3_concentration_pct', 'N/A')}%")

    tests_B = [t1, t2, t3, t4, t5, t6]
    passes_B = sum(1 for t in tests_B if t['PASS'])
    verdict_B = 'VALIDATED' if passes_B >= 5 else 'REJECTED'

    results['B_Correlation_Breakdown'] = {
        'baseline': {'trades': len(trades_B), 'sharpe': round(sharpe_B, 3), 'win_rate': round(wr_B, 3)},
        'tests': tests_B,
        'passes': passes_B,
        'total_tests': 6,
        'verdict': verdict_B
    }

    # ── Strategy F: Combined Bounce ──────────────────────────────────────
    print("\n" + "="*70)
    print("STRATEGY F: Combined Bounce")
    print("="*70)

    trades_F = run_strategy_F(close, dispersion, rsi, cost_pct=BASE_COST_PCT)
    sharpe_F = compute_sharpe(trades_F)
    wr_F = compute_win_rate(trades_F)
    print(f"  Baseline: {len(trades_F)} trades, Sharpe={sharpe_F:.3f}, WR={wr_F:.1%}")

    features_F = (dispersion, rsi)

    print("  [1/6] Inverse Signal...")
    t1 = test_inverse_signal(run_strategy_F, close, features_F, sharpe_F)
    print(f"    {'PASS' if t1['PASS'] else 'FAIL'} — Inverse Sharpe={t1['inverse_sharpe']}, Ratio={t1['ratio']}")

    print("  [2/6] Random Timing (1000 shuffles)...")
    t2 = test_random_timing(trades_F, close)
    print(f"    {'PASS' if t2['PASS'] else 'FAIL'} — Percentile={t2.get('percentile_rank', 'N/A')}")

    print("  [3/6] Cost Sensitivity...")
    t3 = test_cost_sensitivity(run_strategy_F, close, features_F)
    print(f"    {'PASS' if t3['PASS'] else 'FAIL'} — Sharpe@0.30%={t3['sharpe_at_030pct']}")

    print("  [4/6] Sub-Period Stability...")
    t4 = test_subperiod_stability(trades_F)
    print(f"    {'PASS' if t4['PASS'] else 'FAIL'} — Positive periods={t4.get('positive_periods', 'N/A')}/4")

    print("  [5/6] Parameter Robustness (60 combos)...")
    t5 = test_parameter_robustness('F', close, features_F, run_strategy_F)
    print(f"    {'PASS' if t5['PASS'] else 'FAIL'} — {t5['pct_above']}% above 0.3")

    print("  [6/6] Sector Concentration...")
    t6 = test_sector_concentration(trades_F)
    print(f"    {'PASS' if t6['PASS'] else 'FAIL'} — Top-3 concentration={t6.get('top3_concentration_pct', 'N/A')}%")

    tests_F = [t1, t2, t3, t4, t5, t6]
    passes_F = sum(1 for t in tests_F if t['PASS'])
    verdict_F = 'VALIDATED' if passes_F >= 5 else 'REJECTED'

    results['F_Combined_Bounce'] = {
        'baseline': {'trades': len(trades_F), 'sharpe': round(sharpe_F, 3), 'win_rate': round(wr_F, 3)},
        'tests': tests_F,
        'passes': passes_F,
        'total_tests': 6,
        'verdict': verdict_F
    }

    # ── Summary Table ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("="*70)

    for name, res in results.items():
        print(f"\n  {name}:")
        print(f"    Baseline: {res['baseline']['trades']} trades, Sharpe={res['baseline']['sharpe']}, WR={res['baseline']['win_rate']:.1%}")
        print(f"    {'Test':<30} {'Result':<8}")
        print(f"    {'-'*38}")
        for t in res['tests']:
            status = 'PASS' if t['PASS'] else 'FAIL'
            print(f"    {t['test']:<30} {status:<8}")
        print(f"    {'-'*38}")
        print(f"    Score: {res['passes']}/{res['total_tests']} — {res['verdict']}")

    # ── Save Results ─────────────────────────────────────────────────────
    # Convert timestamps for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    serializable = make_serializable(results)
    serializable['_meta'] = {
        'run_date': datetime.now().isoformat(),
        'data_range': f'{START_DATE} to {END_DATE}',
        'n_random_shuffles': N_RANDOM_SHUFFLES,
        'base_cost_pct': BASE_COST_PCT
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    run_all_tests()
