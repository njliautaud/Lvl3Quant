#!/usr/bin/env python3
"""
Adversarial Validation: Dynamic sector ETF dip-buying exits based on recovery speed prediction.

Six adversarial tests:
1. Inverse Signal — swap FAST/SLOW labels
2. Random Timing — permutation test (1000 shuffles)
3. Cost Sensitivity — Sharpe at 0.10%, 0.20%, 0.30%, 0.50% RT
4. Sub-Period Stability — 4 equal periods, Sharpe each
5. Parameter Robustness — 432-combo grid sweep
6. Sector Concentration — check if >70% trades in 3 sectors
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import product as iterproduct

warnings.filterwarnings('ignore')

# ─── CONFIG ────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
VIX_TICKER = '^VIX'
START = '2020-01-01'
END = datetime.now().strftime('%Y-%m-%d')
RSI_PERIOD = 14
RSI_THRESHOLD = 35
BASE_COST_PCT = 0.10 / 100
RESULTS_PATH = '/home/jupiter/Lvl3Quant/scripts/growth_research/results/recovery_speed_adversarial.json'

# Baseline parameters
BASE_VIX_FAST = 25
BASE_FAST_HOLD = 2
BASE_FAST_TP = 0.02
BASE_SLOW_HOLD = 10
BASE_SLOW_TP = 0.05
BASE_DEFAULT_HOLD = 5
BASE_DEFAULT_TP = 0.03
BASE_SL = -0.05
BASE_CAP_VOL = 1.5


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    print("Downloading data...")
    tickers = SECTOR_ETFS + ['SPY', VIX_TICKER]
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'].copy()
        volume = data['Volume'].copy()
    else:
        close = data
        volume = None

    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})
    if volume is not None and '^VIX' in volume.columns:
        volume = volume.rename(columns={'^VIX': 'VIX'})

    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close, volume


def find_entries(close, volume):
    """Find all RSI<35 dip entries across sector ETFs."""
    entries = []

    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue

        px = close[etf].dropna()
        vol = volume[etf].dropna() if volume is not None and etf in volume.columns else None
        rsi = compute_rsi(px, RSI_PERIOD)
        vix = close['VIX'].dropna() if 'VIX' in close.columns else None
        sma200 = px.rolling(200).mean()
        vol_ma20 = vol.rolling(20).mean() if vol is not None else None

        # RSI dip start detection
        rsi_below = rsi < RSI_THRESHOLD
        dip_starts = rsi_below & ~rsi_below.shift(1, fill_value=False)

        for date in dip_starts[dip_starts].index:
            idx = px.index.get_loc(date)
            if idx < 200:
                continue

            entry_price = px.iloc[idx]

            # VIX at entry
            vix_val = 20.0
            if vix is not None and len(vix.loc[:date]) > 0:
                vix_val = float(vix.loc[:date].iloc[-1])

            # Above 200 SMA
            sma200_val = sma200.iloc[idx]
            above_200 = bool(entry_price > sma200_val) if not pd.isna(sma200_val) else True

            # Capitulation volume
            cap_vol = False
            vol_ratio = 1.0
            if vol is not None and vol_ma20 is not None and not pd.isna(vol_ma20.iloc[idx]):
                vol_ratio = float(vol.iloc[idx] / vol_ma20.iloc[idx]) if vol_ma20.iloc[idx] > 0 else 1.0
                cap_vol = vol_ratio > BASE_CAP_VOL

            # Forward prices (up to 25 days to cover max hold)
            max_fwd = min(idx + 26, len(px))
            fwd_prices = px.iloc[idx:max_fwd].values

            entries.append({
                'etf': etf,
                'date': date,
                'entry_price': float(entry_price),
                'vix': vix_val,
                'above_200': above_200,
                'cap_vol': cap_vol,
                'vol_ratio': vol_ratio,
                'fwd_prices': fwd_prices,
                'fwd_len': len(fwd_prices),
            })

    print(f"  Found {len(entries)} dip entries across {len(set(e['etf'] for e in entries))} sectors")
    return entries


def classify_and_trade(entries, vix_fast=25, fast_hold=2, slow_hold=10,
                       fast_tp=0.02, slow_tp=0.05, default_hold=5, default_tp=0.03,
                       sl=-0.05, cost_pct=0.001, invert=False):
    """Classify entries and simulate trades. Returns list of trade returns."""
    trades = []

    for e in entries:
        # Classify
        is_fast = (e['vix'] > vix_fast) and e['cap_vol'] and e['above_200']
        is_slow = (not e['above_200']) and (not e['cap_vol'])

        if invert:
            # Swap: FAST gets SLOW params, SLOW gets FAST params
            if is_fast:
                hold, tp = slow_hold, slow_tp
            elif is_slow:
                hold, tp = fast_hold, fast_tp
            else:
                hold, tp = default_hold, default_tp
        else:
            if is_fast:
                hold, tp = fast_hold, fast_tp
            elif is_slow:
                hold, tp = slow_hold, slow_tp
            else:
                hold, tp = default_hold, default_tp

        # Simulate trade
        entry_px = e['fwd_prices'][0]
        exit_px = entry_px  # default: flat
        exit_day = hold

        for d in range(1, min(hold + 1, e['fwd_len'])):
            px = e['fwd_prices'][d]
            ret = (px - entry_px) / entry_px

            if ret >= tp:
                exit_px = entry_px * (1 + tp)
                exit_day = d
                break
            elif ret <= sl:
                exit_px = entry_px * (1 + sl)
                exit_day = d
                break
        else:
            # Hold expired — exit at last available price within hold window
            actual_exit = min(hold, e['fwd_len'] - 1)
            exit_px = e['fwd_prices'][actual_exit]
            exit_day = actual_exit

        gross_ret = (exit_px - entry_px) / entry_px
        net_ret = gross_ret - cost_pct

        classification = 'FAST' if (is_fast and not invert) or (is_slow and invert) else \
                         'SLOW' if (is_slow and not invert) or (is_fast and invert) else 'DEFAULT'

        trades.append({
            'etf': e['etf'],
            'date': e['date'],
            'classification': classification,
            'hold_days': exit_day,
            'gross_ret': gross_ret,
            'net_ret': net_ret,
        })

    return trades


def compute_metrics(trades):
    """Compute Sharpe, WR, PF from trade list."""
    if not trades:
        return {'sharpe': 0, 'wr': 0, 'pf': 0, 'n_trades': 0, 'avg_ret': 0}

    rets = np.array([t['net_ret'] for t in trades])
    n = len(rets)
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    avg_ret = float(np.mean(rets))
    std_ret = float(np.std(rets)) if len(rets) > 1 else 1e-6
    sharpe = avg_ret / std_ret * np.sqrt(252 / 5) if std_ret > 1e-8 else 0  # ~5 day avg hold
    wr = float(len(wins) / n) if n > 0 else 0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 1e-10 else 999

    return {
        'sharpe': round(float(sharpe), 3),
        'wr': round(wr * 100, 1),
        'pf': round(float(pf), 3),
        'n_trades': n,
        'avg_ret': round(avg_ret * 100, 3),
    }


def test_inverse_signal(entries):
    """Test 1: Swap FAST and SLOW labels."""
    print("\n" + "="*60)
    print("TEST 1: INVERSE SIGNAL")
    print("="*60)

    normal_trades = classify_and_trade(entries, invert=False)
    inverse_trades = classify_and_trade(entries, invert=True)

    normal_m = compute_metrics(normal_trades)
    inverse_m = compute_metrics(inverse_trades)

    ratio = inverse_m['sharpe'] / normal_m['sharpe'] if abs(normal_m['sharpe']) > 1e-6 else 999

    passed = ratio < 0.50
    print(f"  Normal Sharpe:  {normal_m['sharpe']:.3f}")
    print(f"  Inverse Sharpe: {inverse_m['sharpe']:.3f}")
    print(f"  Ratio:          {ratio:.3f}")
    print(f"  Result:         {'PASS' if passed else 'FAIL'} (need ratio < 0.50)")

    return {
        'test': 'Inverse Signal',
        'normal_sharpe': normal_m['sharpe'],
        'inverse_sharpe': inverse_m['sharpe'],
        'ratio': round(ratio, 3),
        'threshold': 0.50,
        'passed': passed,
        'normal_metrics': normal_m,
        'inverse_metrics': inverse_m,
    }


def test_random_timing(entries, close, n_shuffles=1000):
    """Test 2: Randomize entry dates, keep same count."""
    print("\n" + "="*60)
    print("TEST 2: RANDOM TIMING (1000 shuffles)")
    print("="*60)

    # Real strategy metrics
    real_trades = classify_and_trade(entries)
    real_m = compute_metrics(real_trades)
    real_sharpe = real_m['sharpe']

    # Get all valid trading dates for random entries
    all_dates = close.index[200:]  # skip first 200 for SMA warmup
    n_trades = len(entries)
    vix = close['VIX'].dropna() if 'VIX' in close.columns else None

    random_sharpes = []
    rng = np.random.RandomState(42)

    for shuffle_i in range(n_shuffles):
        # Random entries: pick random dates and random sectors
        rand_dates = rng.choice(all_dates, size=n_trades, replace=True)
        rand_etfs = rng.choice(SECTOR_ETFS, size=n_trades, replace=True)

        rand_entries = []
        for rd, retf in zip(rand_dates, rand_etfs):
            if retf not in close.columns:
                continue
            px_series = close[retf].dropna()
            if rd not in px_series.index:
                # Find nearest date
                valid = px_series.index[px_series.index >= rd]
                if len(valid) == 0:
                    continue
                rd = valid[0]

            idx = px_series.index.get_loc(rd)
            if idx < 200 or idx + 2 >= len(px_series):
                continue

            entry_price = float(px_series.iloc[idx])
            max_fwd = min(idx + 26, len(px_series))
            fwd_prices = px_series.iloc[idx:max_fwd].values

            vix_val = 20.0
            if vix is not None and len(vix.loc[:rd]) > 0:
                vix_val = float(vix.loc[:rd].iloc[-1])

            sma200_val = px_series.rolling(200).mean().iloc[idx]
            above_200 = bool(entry_price > sma200_val) if not pd.isna(sma200_val) else True

            rand_entries.append({
                'etf': retf,
                'date': rd,
                'entry_price': entry_price,
                'vix': vix_val,
                'above_200': above_200,
                'cap_vol': False,  # random entry unlikely capitulation
                'vol_ratio': 1.0,
                'fwd_prices': fwd_prices,
                'fwd_len': len(fwd_prices),
            })

        if len(rand_entries) > 10:
            rand_trades = classify_and_trade(rand_entries)
            rand_m = compute_metrics(rand_trades)
            random_sharpes.append(rand_m['sharpe'])

    random_sharpes = np.array(random_sharpes)
    percentile = float(np.mean(random_sharpes < real_sharpe) * 100)

    passed = percentile > 90
    print(f"  Real Sharpe:     {real_sharpe:.3f}")
    print(f"  Random median:   {np.median(random_sharpes):.3f}")
    print(f"  Random p95:      {np.percentile(random_sharpes, 95):.3f}")
    print(f"  Percentile:      {percentile:.1f}%")
    print(f"  Result:          {'PASS' if passed else 'FAIL'} (need > 90th percentile)")

    return {
        'test': 'Random Timing',
        'real_sharpe': real_sharpe,
        'random_median': round(float(np.median(random_sharpes)), 3),
        'random_p95': round(float(np.percentile(random_sharpes, 95)), 3),
        'percentile': round(percentile, 1),
        'threshold': 90,
        'passed': passed,
    }


def test_cost_sensitivity(entries):
    """Test 3: Run at multiple cost levels."""
    print("\n" + "="*60)
    print("TEST 3: COST SENSITIVITY")
    print("="*60)

    cost_levels = [0.10, 0.20, 0.30, 0.50]
    results_by_cost = {}

    for cost in cost_levels:
        cost_pct = cost / 100
        trades = classify_and_trade(entries, cost_pct=cost_pct)
        m = compute_metrics(trades)
        results_by_cost[f'{cost:.2f}%'] = m
        print(f"  Cost {cost:.2f}%: Sharpe={m['sharpe']:.3f}, WR={m['wr']:.1f}%, PF={m['pf']:.3f}")

    # Check Sharpe > 0.3 at 0.30%
    sharpe_030 = results_by_cost['0.30%']['sharpe']
    passed = sharpe_030 > 0.3

    print(f"  Sharpe at 0.30%: {sharpe_030:.3f}")
    print(f"  Result:          {'PASS' if passed else 'FAIL'} (need Sharpe > 0.3 at 0.30%)")

    return {
        'test': 'Cost Sensitivity',
        'cost_results': results_by_cost,
        'sharpe_at_030': sharpe_030,
        'threshold': 0.3,
        'passed': passed,
    }


def test_subperiod_stability(entries):
    """Test 4: Split into 4 equal time periods."""
    print("\n" + "="*60)
    print("TEST 4: SUB-PERIOD STABILITY")
    print("="*60)

    all_dates = sorted([e['date'] for e in entries])
    if len(all_dates) < 20:
        print("  Not enough trades for sub-period analysis")
        return {'test': 'Sub-Period Stability', 'passed': False, 'reason': 'insufficient trades'}

    # Split into 4 equal date ranges
    min_date = all_dates[0]
    max_date = all_dates[-1]
    total_days = (max_date - min_date).days
    quarter = total_days / 4

    periods = []
    for i in range(4):
        p_start = min_date + pd.Timedelta(days=int(quarter * i))
        p_end = min_date + pd.Timedelta(days=int(quarter * (i + 1)))
        period_entries = [e for e in entries if p_start <= e['date'] < p_end]
        if i == 3:  # include last day
            period_entries = [e for e in entries if p_start <= e['date'] <= p_end]
        periods.append((p_start, p_end, period_entries))

    period_results = []
    positive_count = 0

    for i, (p_start, p_end, p_entries) in enumerate(periods):
        trades = classify_and_trade(p_entries)
        m = compute_metrics(trades)
        is_positive = m['sharpe'] > 0
        if is_positive:
            positive_count += 1
        period_results.append({
            'period': f'P{i+1}',
            'start': str(p_start.date()),
            'end': str(p_end.date()),
            'n_trades': m['n_trades'],
            'sharpe': m['sharpe'],
            'wr': m['wr'],
            'positive': is_positive,
        })
        print(f"  P{i+1} ({p_start.date()} to {p_end.date()}): "
              f"N={m['n_trades']}, Sharpe={m['sharpe']:.3f}, WR={m['wr']:.1f}%")

    passed = positive_count >= 3
    print(f"  Positive periods: {positive_count}/4")
    print(f"  Result:           {'PASS' if passed else 'FAIL'} (need >= 3/4 positive)")

    return {
        'test': 'Sub-Period Stability',
        'periods': period_results,
        'positive_count': positive_count,
        'threshold': 3,
        'passed': passed,
    }


def test_parameter_robustness(entries):
    """Test 5: 432-combination grid sweep."""
    print("\n" + "="*60)
    print("TEST 5: PARAMETER ROBUSTNESS (432 combos)")
    print("="*60)

    vix_thresholds = [20, 25, 30, 35]
    fast_holds = [1, 2, 3]
    slow_holds = [7, 10, 15, 20]
    fast_tps = [0.015, 0.02, 0.03]
    slow_tps = [0.04, 0.05, 0.07]

    total = len(vix_thresholds) * len(fast_holds) * len(slow_holds) * len(fast_tps) * len(slow_tps)
    print(f"  Testing {total} parameter combinations...")

    sharpe_above_03 = 0
    all_sharpes = []
    best_sharpe = -999
    best_params = None

    for vt, fh, sh, ft, st in iterproduct(vix_thresholds, fast_holds, slow_holds, fast_tps, slow_tps):
        trades = classify_and_trade(
            entries, vix_fast=vt, fast_hold=fh, slow_hold=sh,
            fast_tp=ft, slow_tp=st
        )
        m = compute_metrics(trades)
        s = m['sharpe']
        all_sharpes.append(s)

        if s > 0.3:
            sharpe_above_03 += 1
        if s > best_sharpe:
            best_sharpe = s
            best_params = {'vix': vt, 'fast_hold': fh, 'slow_hold': sh,
                          'fast_tp': ft, 'slow_tp': st}

    pct_above = sharpe_above_03 / total * 100
    passed = pct_above > 50

    print(f"  Sharpe > 0.3:  {sharpe_above_03}/{total} ({pct_above:.1f}%)")
    print(f"  Median Sharpe: {np.median(all_sharpes):.3f}")
    print(f"  Best Sharpe:   {best_sharpe:.3f} (VIX={best_params['vix']}, "
          f"FH={best_params['fast_hold']}, SH={best_params['slow_hold']}, "
          f"FTP={best_params['fast_tp']}, STP={best_params['slow_tp']})")
    print(f"  Result:        {'PASS' if passed else 'FAIL'} (need > 50% with Sharpe > 0.3)")

    return {
        'test': 'Parameter Robustness',
        'total_combos': total,
        'above_threshold': sharpe_above_03,
        'pct_above': round(pct_above, 1),
        'median_sharpe': round(float(np.median(all_sharpes)), 3),
        'best_sharpe': round(best_sharpe, 3),
        'best_params': best_params,
        'threshold_pct': 50,
        'passed': passed,
    }


def test_sector_concentration(entries):
    """Test 6: Check if >70% trades from just 3 sectors."""
    print("\n" + "="*60)
    print("TEST 6: SECTOR CONCENTRATION")
    print("="*60)

    trades = classify_and_trade(entries)
    sector_counts = {}
    for t in trades:
        sector_counts[t['etf']] = sector_counts.get(t['etf'], 0) + 1

    total = len(trades)
    sorted_sectors = sorted(sector_counts.items(), key=lambda x: -x[1])

    print(f"  Total trades: {total}")
    print(f"  {'Sector':<8} {'Count':>6} {'Pct':>7}")
    for sec, cnt in sorted_sectors:
        print(f"  {sec:<8} {cnt:>6} {cnt/total*100:>6.1f}%")

    # Top 3 concentration
    top3_count = sum(cnt for _, cnt in sorted_sectors[:3])
    top3_pct = top3_count / total * 100

    passed = top3_pct <= 70
    print(f"\n  Top 3 sectors: {top3_pct:.1f}% of trades")
    print(f"  Result:        {'PASS' if passed else 'FAIL'} (need <= 70%)")

    return {
        'test': 'Sector Concentration',
        'sector_counts': {k: v for k, v in sorted_sectors},
        'top3_sectors': [s for s, _ in sorted_sectors[:3]],
        'top3_pct': round(top3_pct, 1),
        'threshold': 70,
        'passed': passed,
    }


def main():
    print("=" * 60)
    print("ADVERSARIAL VALIDATION: Recovery Speed Dynamic Exit")
    print("=" * 60)

    close, volume = download_data()
    entries = find_entries(close, volume)

    if len(entries) < 20:
        print("ERROR: Too few entries for meaningful adversarial testing")
        return

    # Run baseline first
    baseline_trades = classify_and_trade(entries)
    baseline = compute_metrics(baseline_trades)
    print(f"\n  BASELINE: Sharpe={baseline['sharpe']:.3f}, WR={baseline['wr']:.1f}%, "
          f"PF={baseline['pf']:.3f}, N={baseline['n_trades']}")

    # Run all 6 tests
    results = {}

    r1 = test_inverse_signal(entries)
    results['inverse_signal'] = r1

    r2 = test_random_timing(entries, close, n_shuffles=1000)
    results['random_timing'] = r2

    r3 = test_cost_sensitivity(entries)
    results['cost_sensitivity'] = r3

    r4 = test_subperiod_stability(entries)
    results['subperiod_stability'] = r4

    r5 = test_parameter_robustness(entries)
    results['parameter_robustness'] = r5

    r6 = test_sector_concentration(entries)
    results['sector_concentration'] = r6

    # Summary
    all_tests = [r1, r2, r3, r4, r5, r6]
    n_passed = sum(1 for t in all_tests if t['passed'])
    n_total = len(all_tests)

    print("\n" + "=" * 60)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Test':<30} {'Result':<10}")
    print("  " + "-" * 40)
    for t in all_tests:
        status = 'PASS' if t['passed'] else 'FAIL'
        print(f"  {t['test']:<30} {status:<10}")

    print(f"\n  OVERALL: {n_passed}/{n_total} tests passed")

    if n_passed >= 6:
        verdict = 'STRONG PASS — all tests passed'
    elif n_passed >= 5:
        verdict = 'PASS — 5/6, minor concern'
    elif n_passed >= 4:
        verdict = 'MARGINAL — review failed tests'
    else:
        verdict = 'FAIL — strategy has fundamental issues'

    print(f"  VERDICT: {verdict}")

    # Save results
    output = {
        'strategy': 'Recovery Speed Dynamic Exit',
        'timestamp': datetime.now().isoformat(),
        'data_range': f'{START} to {END}',
        'baseline': baseline,
        'tests': {},
        'summary': {
            'passed': n_passed,
            'total': n_total,
            'verdict': verdict,
        }
    }

    for key, r in results.items():
        # Remove non-serializable items
        clean = {}
        for k, v in r.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean[k] = v
        output['tests'][key] = clean

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
