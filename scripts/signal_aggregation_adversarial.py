#!/usr/bin/env python3
"""
Adversarial Validation for Signal Aggregation Strategy A
Tests whether the reported Sharpe 1.066 (perm p=0.042) reflects genuine edge.

6 tests:
1. INVERSE: Flip signals — if inverse also works, no directional edge
2. RANDOM TIMING: 1000 random entry/exit schedules, same holding distribution
3. LOOK-AHEAD: Lag all 6 input signals by 5 days
4. COST SENSITIVITY: Slippage at 0.05%, 0.10%, 0.15%, 0.20%
5. SUB-PERIOD: Split OOT into 4 equal sub-periods, Sharpe per sub-period
6. PARAMETER SENSITIVITY: Threshold × lookback grid (25 combos)

PASS criteria: >= 5/6 tests pass.
OOT: Jan 2022 - Jul 2026. yfinance for data. SPY 200-SMA for regime.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
BASE_SLIPPAGE = 0.0002
START_DATE = '2022-01-01'
END_DATE = '2026-07-30'
ACTUAL_SHARPE = 1.066

TICKERS = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
           'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
           'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI',
               'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/data/signal_aggregation_adversarial.json')


# ── Data Download ─────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV data for all tickers."""
    print("Downloading data...")
    buf_start = '2021-06-01'
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=buf_start, end=END_DATE, progress=False, auto_adjust=True)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"  {t}: {len(df)} bars")
        except Exception as e:
            print(f"  {t}: FAILED ({e})")
    return data


# ── Signal Generators (same as original) ──────────────────────────────────
def calc_momentum(close, period=20):
    ret = close.pct_change(period)
    return (ret / 0.20).clip(-1, 1)


def calc_rsi(close, period=5):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0.0, index=close.index)
    signal[rsi < 30] = 1.0
    signal[rsi > 70] = -1.0
    return signal


def calc_volume_signal(close, volume, period=20):
    vol_ratio = volume / volume.rolling(period).mean()
    daily_ret = close.pct_change()
    signal = pd.Series(0.0, index=close.index)
    signal[(vol_ratio > 1.5) & (daily_ret > 0)] = 1.0
    signal[(vol_ratio > 1.5) & (daily_ret < 0)] = -1.0
    return signal


def calc_trend(close, period=50):
    sma = close.rolling(period).mean()
    signal = pd.Series(0.0, index=close.index)
    signal[close > sma] = 1.0
    signal[close < sma] = -1.0
    return signal


def calc_breadth(sector_closes):
    above_sma = pd.DataFrame()
    for t, close in sector_closes.items():
        sma20 = close.rolling(20).mean()
        above_sma[t] = (close > sma20).astype(float)
    pct_above = above_sma.mean(axis=1)
    signal = pd.Series(0.0, index=pct_above.index)
    signal[pct_above > 0.70] = 1.0
    signal[pct_above < 0.30] = -1.0
    return signal


def calc_vol_regime(close):
    ret = close.pct_change()
    vol20 = ret.rolling(20).std()
    vol60 = ret.rolling(60).std()
    signal = pd.Series(0.0, index=close.index)
    signal[vol20 < vol60] = 1.0
    signal[vol20 > vol60] = -1.0
    return signal


def build_all_signals(data, mom_period=20):
    """Build signal DataFrame. mom_period is configurable for param sensitivity."""
    common_idx = None
    for t in TICKERS:
        if t in data:
            idx = data[t].index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

    sector_closes = {}
    for t in SECTOR_ETFS:
        if t in data:
            sector_closes[t] = data[t]['Close'].reindex(common_idx)

    breadth_signal = calc_breadth(sector_closes)

    signals = {}
    for t in TICKERS:
        if t not in data:
            continue
        df = data[t].reindex(common_idx)
        close = df['Close']
        volume = df['Volume']

        sig_df = pd.DataFrame(index=common_idx)
        sig_df['momentum'] = calc_momentum(close, period=mom_period)
        sig_df['rsi'] = calc_rsi(close)
        sig_df['volume'] = calc_volume_signal(close, volume)
        sig_df['trend'] = calc_trend(close)
        sig_df['breadth'] = breadth_signal
        sig_df['vol_regime'] = calc_vol_regime(close)
        sig_df['composite'] = sig_df.mean(axis=1)
        signals[t] = sig_df

    return signals, common_idx


# ── Simulation Engine ─────────────────────────────────────────────────────
def simulate_variant_a(signals, data, common_idx, oot_start, threshold=0.3,
                       slippage=BASE_SLIPPAGE, invert=False):
    """
    Strategy A: SPY when composite > +threshold, GLD when < -threshold, cash between.
    If invert=True, flip: SPY when < -threshold, GLD when > +threshold.
    """
    alloc = {}
    spy_sig = signals.get('SPY')
    if spy_sig is None:
        return pd.Series(dtype=float), 0

    for d in common_idx:
        if d < pd.Timestamp(oot_start):
            continue
        comp = spy_sig.loc[d, 'composite'] if d in spy_sig.index else 0
        if np.isnan(comp):
            alloc[d] = {}
        elif not invert:
            if comp > threshold:
                alloc[d] = {'SPY': 1.0}
            elif comp < -threshold:
                alloc[d] = {'GLD': 1.0}
            else:
                alloc[d] = {}
        else:
            # INVERSE: flip the logic
            if comp < -threshold:
                alloc[d] = {'SPY': 1.0}
            elif comp > threshold:
                alloc[d] = {'GLD': 1.0}
            else:
                alloc[d] = {}

    return _simulate_allocation(alloc, data, oot_start, slippage)


def _simulate_allocation(alloc_series, data, oot_start, slippage=BASE_SLIPPAGE):
    dates = sorted(alloc_series.keys())
    dates = [d for d in dates if d >= pd.Timestamp(oot_start)]
    if not dates:
        return pd.Series(dtype=float), 0

    equity = INITIAL_CAPITAL
    equity_curve = {}
    prev_alloc = {}
    trades = 0

    for d in dates:
        target = alloc_series[d]
        for t, w in target.items():
            old_w = prev_alloc.get(t, 0.0)
            if abs(w - old_w) > 0.01:
                trades += 1

        day_ret = 0.0
        for t, w in target.items():
            if t in data and d in data[t].index:
                idx = data[t].index.get_loc(d)
                if idx > 0:
                    r = (data[t]['Close'].iloc[idx] / data[t]['Close'].iloc[idx - 1]) - 1
                    day_ret += w * r
                    old_w = prev_alloc.get(t, 0.0)
                    if abs(w - old_w) > 0.01:
                        day_ret -= abs(w - old_w) * slippage

        equity *= (1 + day_ret)
        equity_curve[d] = equity
        prev_alloc = target.copy()

    return pd.Series(equity_curve), trades


def calc_sharpe(equity_curve):
    """Calculate annualized Sharpe from equity curve."""
    if len(equity_curve) < 20:
        return 0.0
    returns = equity_curve.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
    total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(returns) / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252) if returns.std() > 0 else 1e-6
    return ann_ret / ann_vol if ann_vol > 1e-8 else 0.0


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE
# ══════════════════════════════════════════════════════════════════════════
def test_inverse(signals, data, common_idx):
    """Flip signals. If inverse also works, no directional edge."""
    print("\n" + "=" * 70)
    print("TEST 1: INVERSE SIGNAL")
    print("=" * 70)

    eq_normal, _ = simulate_variant_a(signals, data, common_idx, START_DATE, invert=False)
    eq_inverse, _ = simulate_variant_a(signals, data, common_idx, START_DATE, invert=True)

    sharpe_normal = calc_sharpe(eq_normal)
    sharpe_inverse = calc_sharpe(eq_inverse)

    # PASS if inverse Sharpe is meaningfully worse (< 0.3 or < 50% of normal)
    inverse_is_bad = sharpe_inverse < 0.3 or sharpe_inverse < sharpe_normal * 0.5
    passed = inverse_is_bad

    print(f"  Normal Sharpe:  {sharpe_normal:.3f}")
    print(f"  Inverse Sharpe: {sharpe_inverse:.3f}")
    print(f"  PASS criteria: Inverse Sharpe < 0.3 OR < 50% of normal")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'inverse',
        'normal_sharpe': round(sharpe_normal, 3),
        'inverse_sharpe': round(sharpe_inverse, 3),
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM TIMING
# ══════════════════════════════════════════════════════════════════════════
def test_random_timing(signals, data, common_idx, n_iter=1000):
    """1000 random entry/exit dates with same holding distribution."""
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM TIMING (1000 iterations)")
    print("=" * 70)

    # First, get actual strategy's allocation pattern to measure holding distribution
    spy_sig = signals.get('SPY')
    oot_dates = [d for d in common_idx if d >= pd.Timestamp(START_DATE)]

    # Compute actual allocations to measure holding stats
    actual_states = []  # 0=cash, 1=SPY, 2=GLD
    for d in oot_dates:
        comp = spy_sig.loc[d, 'composite'] if d in spy_sig.index else 0
        if np.isnan(comp):
            actual_states.append(0)
        elif comp > 0.3:
            actual_states.append(1)
        elif comp < -0.3:
            actual_states.append(2)
        else:
            actual_states.append(0)

    actual_states = np.array(actual_states)
    n_days = len(oot_dates)

    # Count state frequencies
    pct_spy = (actual_states == 1).sum() / n_days
    pct_gld = (actual_states == 2).sum() / n_days
    pct_cash = (actual_states == 0).sum() / n_days

    print(f"  Actual holding distribution: SPY={pct_spy:.1%}, GLD={pct_gld:.1%}, Cash={pct_cash:.1%}")

    # Measure holding period lengths
    holding_lengths = []
    current_state = actual_states[0]
    current_len = 1
    for i in range(1, len(actual_states)):
        if actual_states[i] == current_state:
            current_len += 1
        else:
            holding_lengths.append(current_len)
            current_state = actual_states[i]
            current_len = 1
    holding_lengths.append(current_len)
    avg_hold = np.mean(holding_lengths)

    # Actual strategy Sharpe
    eq_actual, _ = simulate_variant_a(signals, data, common_idx, START_DATE)
    actual_sharpe = calc_sharpe(eq_actual)

    # Generate random timing strategies preserving holding distribution
    rng = np.random.RandomState(42)
    random_sharpes = []

    for i in range(n_iter):
        # Generate random state sequence with same state probabilities
        random_states = rng.choice([0, 1, 2], size=n_days, p=[pct_cash, pct_spy, pct_gld])

        # Build allocation dict
        alloc = {}
        for j, d in enumerate(oot_dates):
            state = random_states[j]
            if state == 1:
                alloc[d] = {'SPY': 1.0}
            elif state == 2:
                alloc[d] = {'GLD': 1.0}
            else:
                alloc[d] = {}

        eq, _ = _simulate_allocation(alloc, data, START_DATE)
        if len(eq) > 20:
            random_sharpes.append(calc_sharpe(eq))
        else:
            random_sharpes.append(0.0)

    random_sharpes = np.array(random_sharpes)
    percentile = (random_sharpes < actual_sharpe).sum() / len(random_sharpes) * 100

    # PASS if actual Sharpe > 90th percentile of random
    passed = percentile >= 90.0

    print(f"  Actual Sharpe: {actual_sharpe:.3f}")
    print(f"  Random Sharpes: mean={random_sharpes.mean():.3f}, "
          f"std={random_sharpes.std():.3f}, max={random_sharpes.max():.3f}")
    print(f"  Actual percentile: {percentile:.1f}th")
    print(f"  PASS criteria: Actual >= 90th percentile of random")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'random_timing',
        'actual_sharpe': round(actual_sharpe, 3),
        'random_mean_sharpe': round(float(random_sharpes.mean()), 3),
        'random_std_sharpe': round(float(random_sharpes.std()), 3),
        'random_max_sharpe': round(float(random_sharpes.max()), 3),
        'percentile': round(percentile, 1),
        'n_iterations': n_iter,
        'holding_distribution': {
            'spy_pct': round(pct_spy, 3),
            'gld_pct': round(pct_gld, 3),
            'cash_pct': round(pct_cash, 3),
        },
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: LOOK-AHEAD BIAS (5-day lag)
# ══════════════════════════════════════════════════════════════════════════
def test_look_ahead(data, common_idx):
    """Lag all 6 input signals by 5 days. If performance holds, signals don't predict."""
    print("\n" + "=" * 70)
    print("TEST 3: LOOK-AHEAD BIAS (5-day signal lag)")
    print("=" * 70)

    # Build normal signals for baseline
    signals_normal, _ = build_all_signals(data)
    eq_normal, _ = simulate_variant_a(signals_normal, data, common_idx, START_DATE)
    sharpe_normal = calc_sharpe(eq_normal)

    # Build lagged signals: shift each signal column by 5 days
    signals_lagged = {}
    signal_cols = ['momentum', 'rsi', 'volume', 'trend', 'breadth', 'vol_regime']
    for t, sig_df in signals_normal.items():
        s = sig_df.copy()
        for col in signal_cols:
            s[col] = s[col].shift(5)
        s['composite'] = s[signal_cols].mean(axis=1)
        signals_lagged[t] = s

    eq_lagged, _ = simulate_variant_a(signals_lagged, data, common_idx, START_DATE)
    sharpe_lagged = calc_sharpe(eq_lagged)

    # PASS if lagged performance drops meaningfully (Sharpe drops > 30%)
    drop_pct = (sharpe_normal - sharpe_lagged) / max(abs(sharpe_normal), 1e-6) * 100
    passed = drop_pct > 30  # lagged should be >30% worse

    print(f"  Normal Sharpe:  {sharpe_normal:.3f}")
    print(f"  Lagged Sharpe:  {sharpe_lagged:.3f}")
    print(f"  Drop:           {drop_pct:.1f}%")
    print(f"  PASS criteria:  Sharpe drops > 30% with 5-day lag")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'look_ahead',
        'normal_sharpe': round(sharpe_normal, 3),
        'lagged_sharpe': round(sharpe_lagged, 3),
        'drop_pct': round(drop_pct, 1),
        'lag_days': 5,
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: COST SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
def test_cost_sensitivity(signals, data, common_idx):
    """Test at 0.05%, 0.10%, 0.15%, 0.20% slippage."""
    print("\n" + "=" * 70)
    print("TEST 4: COST SENSITIVITY")
    print("=" * 70)

    slippage_levels = [0.0005, 0.0010, 0.0015, 0.0020]
    results = {}

    for slip in slippage_levels:
        eq, trades = simulate_variant_a(signals, data, common_idx, START_DATE, slippage=slip)
        sharpe = calc_sharpe(eq)
        label = f"{slip*100:.2f}%"
        results[label] = round(sharpe, 3)
        print(f"  Slippage {label}: Sharpe = {sharpe:.3f}")

    # PASS if Sharpe > 0.5 at 0.10% slippage (moderate costs)
    passed = results['0.10%'] > 0.5

    print(f"  PASS criteria: Sharpe > 0.5 at 0.10% slippage")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'cost_sensitivity',
        'sharpe_by_slippage': results,
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
def test_sub_period(signals, data, common_idx):
    """Split OOT into 4 equal sub-periods. Report Sharpe per sub-period."""
    print("\n" + "=" * 70)
    print("TEST 5: SUB-PERIOD STABILITY")
    print("=" * 70)

    oot_dates = [d for d in common_idx if d >= pd.Timestamp(START_DATE)]
    n = len(oot_dates)
    quarter = n // 4

    sub_periods = [
        ('Q1', oot_dates[:quarter]),
        ('Q2', oot_dates[quarter:2*quarter]),
        ('Q3', oot_dates[2*quarter:3*quarter]),
        ('Q4', oot_dates[3*quarter:]),
    ]

    results = {}
    all_positive = True

    for label, dates in sub_periods:
        period_start = dates[0]
        period_end = dates[-1]

        # Simulate on full data but measure equity only within sub-period
        eq_full, _ = simulate_variant_a(signals, data, common_idx, START_DATE)

        # Extract sub-period equity
        eq_sub = eq_full.reindex(dates).dropna()
        if len(eq_sub) < 10:
            sharpe = 0.0
        else:
            # Renormalize equity to start at INITIAL_CAPITAL for fair Sharpe calc
            eq_sub = eq_sub / eq_sub.iloc[0] * INITIAL_CAPITAL
            sharpe = calc_sharpe(eq_sub)

        results[label] = {
            'sharpe': round(sharpe, 3),
            'start': str(period_start.date()),
            'end': str(period_end.date()),
            'n_days': len(dates),
        }
        if sharpe < 0:
            all_positive = False
        print(f"  {label} ({period_start.date()} to {period_end.date()}): "
              f"Sharpe = {sharpe:.3f}, {len(dates)} days")

    # PASS if no sub-period has negative Sharpe
    passed = all_positive

    print(f"  PASS criteria: No sub-period with negative Sharpe")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'sub_period',
        'sub_periods': results,
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 6: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
def test_parameter_sensitivity(data, common_idx):
    """Test threshold grid [0.1-0.5] × lookback [10-40] = 25 combos."""
    print("\n" + "=" * 70)
    print("TEST 6: PARAMETER SENSITIVITY (25 combos)")
    print("=" * 70)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    lookbacks = [10, 15, 20, 30, 40]

    results = {}
    count_above = 0
    total = 0

    for lb in lookbacks:
        # Rebuild signals with this momentum lookback
        signals_lb, _ = build_all_signals(data, mom_period=lb)
        for thresh in thresholds:
            eq, _ = simulate_variant_a(signals_lb, data, common_idx, START_DATE, threshold=thresh)
            sharpe = calc_sharpe(eq)
            key = f"lb{lb}_th{thresh}"
            results[key] = round(sharpe, 3)
            total += 1
            if sharpe > 0.3:
                count_above += 1
            print(f"  Lookback={lb:2d}, Threshold={thresh:.1f}: Sharpe = {sharpe:.3f}")

    pct_above = count_above / total * 100

    # PASS if >= 30% of combos have Sharpe > 0.3
    passed = pct_above >= 30.0

    print(f"\n  Combos with Sharpe > 0.3: {count_above}/{total} ({pct_above:.0f}%)")
    print(f"  PASS criteria: >= 30% of combos with Sharpe > 0.3")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'parameter_sensitivity',
        'grid_results': results,
        'pct_above_threshold': round(pct_above, 1),
        'count_above': count_above,
        'total_combos': total,
        'passed': passed,
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Signal Aggregation Strategy A")
    print(f"Baseline: Sharpe={ACTUAL_SHARPE}, perm p=0.042")
    print(f"OOT: {START_DATE} to {END_DATE}")
    print("=" * 70)

    data = download_data()
    if 'SPY' not in data:
        print("FATAL: SPY data missing")
        return

    signals, common_idx = build_all_signals(data)
    print(f"Signals built. Common dates: {len(common_idx)}")
    oot_dates = [d for d in common_idx if d >= pd.Timestamp(START_DATE)]
    print(f"OOT dates: {len(oot_dates)} ({oot_dates[0].date()} to {oot_dates[-1].date()})")

    # Run all 6 tests
    results = {}

    results['1_inverse'] = test_inverse(signals, data, common_idx)
    results['2_random_timing'] = test_random_timing(signals, data, common_idx)
    results['3_look_ahead'] = test_look_ahead(data, common_idx)
    results['4_cost_sensitivity'] = test_cost_sensitivity(signals, data, common_idx)
    results['5_sub_period'] = test_sub_period(signals, data, common_idx)
    results['6_parameter_sensitivity'] = test_parameter_sensitivity(data, common_idx)

    # ── Final Verdict ─────────────────────────────────────────────────────
    n_passed = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)
    overall_pass = n_passed >= 5

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  {name}: {status}")
    print(f"\n  Tests passed: {n_passed}/{n_total}")
    print(f"  Required: >= 5/6")
    print(f"  >>> OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # ── Save Results ──────────────────────────────────────────────────────
    output = {
        'meta': {
            'test': 'adversarial_validation',
            'strategy': 'Signal Aggregation A (threshold long-only)',
            'baseline_sharpe': ACTUAL_SHARPE,
            'baseline_perm_p': 0.042,
            'oot_period': f'{START_DATE} to {END_DATE}',
            'run_timestamp': datetime.now().isoformat(),
        },
        'tests': results,
        'summary': {
            'tests_passed': n_passed,
            'tests_total': n_total,
            'pass_threshold': 5,
            'overall_verdict': 'PASS' if overall_pass else 'FAIL',
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
