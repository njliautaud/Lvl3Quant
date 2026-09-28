#!/usr/bin/env python3
"""
Permutation / Null Test for CNN De-Biased Strategy
===================================================

Answers: "Is Sharpe 3.89 statistically significant, or could random
predictions achieve this by chance?"

Method:
  1. Load the same OOS data used in debiased_sweep.py
  2. For each of 1000 permutations:
     - Shuffle the CNN predictions across bars (break pred-to-outcome mapping)
     - Re-run the best config backtest (vol>=80, conv>=1.5, 30min, morning_afternoon)
     - Compute Sharpe for the shuffled version
  3. Compare real Sharpe (3.89) against the null distribution
  4. Output: p-value, z-score, histogram data

The null hypothesis is that the CNN predictions contain no information
about future price movements. Under the null, any positive Sharpe is
from random alignment of entries with favorable price moves.

Usage:
    python alpha_discovery/null_test_cnn.py [--n-perms 1000] [--seed 42]
"""

import sys
import json
import gc
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Path setup ──
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from high_conviction_strategy import (
    load_all_dates, load_cnn_predictions, load_mbo_day,
    zscore_per_day, compute_trailing_vol, _precompute_vol_percentiles,
    simulate_trades, compute_metrics, HOLD_PERIODS, COST_STRUCTURES,
    PARAM_TUNE_DAYS
)

LVL3_ROOT = Path(__file__).parent.parent
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Best config from de-biased sweep ──
BEST_CONFIG = {
    'vol_pct': 80,
    'conv': 1.5,
    'hold': '30min',
    'time': 'morning_afternoon',
}

CNN_OFFSET = 99


def load_oos_data():
    """Load and prepare OOS data, matching debiased_sweep.py exactly."""
    print("Loading CNN predictions and MBO data...")
    all_dates = load_all_dates()
    cnn_data = load_cnn_predictions()

    cost = COST_STRUCTURES['ES_futures']
    cnn_dates_sorted = sorted(cnn_data.keys())

    all_days = []
    for date in cnn_dates_sorted:
        mbo = load_mbo_day(date)
        if mbo is None:
            continue
        mid, spread = mbo
        n_bars = len(mid)
        cp, ct = cnn_data[date]

        # Align CNN predictions with 99-bar offset
        cp_aligned = np.full(n_bars, np.nan)
        end_idx = min(CNN_OFFSET + len(cp), n_bars)
        cp_aligned[CNN_OFFSET:end_idx] = cp[:end_idx - CNN_OFFSET]

        signal = zscore_per_day(cp_aligned)
        conviction = np.abs(signal)
        n_agree = np.ones(n_bars, dtype=int)
        vol_pred = compute_trailing_vol(mid)
        vol_pct_thresholds = _precompute_vol_percentiles(vol_pred)

        all_days.append({
            'date': date,
            'mid': mid.astype(np.float32),
            'spread': spread.astype(np.float32),
            'n_bars': n_bars,
            '_signal': signal.astype(np.float32),
            '_conviction': conviction.astype(np.float32),
            '_n_agree': n_agree,
            '_vol_pred': vol_pred.astype(np.float32),
            '_vol_pct_thresholds': {k: v.astype(np.float32) for k, v in vol_pct_thresholds.items()},
            '_raw_cnn_aligned': cp_aligned.astype(np.float32),  # Keep raw for shuffling
        })
        del mid, spread, cp_aligned, signal, conviction, vol_pred, vol_pct_thresholds
        gc.collect()

    del cnn_data
    gc.collect()

    # Use full OOS (skip first 20 tune days)
    oos_days = all_days[PARAM_TUNE_DAYS:]
    print(f"Loaded {len(all_days)} total days, {len(oos_days)} OOS days")
    return oos_days


def run_backtest(oos_days, cfg):
    """Run backtest with given config, return Sharpe and metrics."""
    cost = COST_STRUCTURES['ES_futures']
    trades = []
    for day in oos_days:
        t = simulate_trades(
            day, day['_signal'], day['_conviction'], day['_n_agree'],
            day['_vol_pred'], hold_bars=HOLD_PERIODS[cfg['hold']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=cfg['conv'],
            min_agreement=1,
            vol_percentile_min=cfg['vol_pct'],
            time_filter=cfg['time'],
        )
        for tt in t:
            tt['date'] = day['date']
        trades.extend(t)

    m = compute_metrics(trades, label='null_test')
    return m


def shuffle_predictions_and_backtest(oos_days, cfg, rng):
    """
    Shuffle CNN predictions within each day, re-compute signal, run backtest.

    We shuffle the raw CNN predictions (before z-scoring) to preserve the
    temporal structure of prices/vol but break the prediction-to-outcome mapping.
    This is the correct null: the model's predictions are random with respect
    to future price movements, but the entry/exit mechanics and filters remain.
    """
    cost = COST_STRUCTURES['ES_futures']
    trades = []

    for day in oos_days:
        raw = day['_raw_cnn_aligned'].copy()

        # Shuffle only the non-NaN predictions (NaN = no CNN prediction at that bar)
        valid_mask = ~np.isnan(raw)
        valid_vals = raw[valid_mask].copy()
        rng.shuffle(valid_vals)
        raw[valid_mask] = valid_vals

        # Re-compute signal from shuffled predictions (same expanding z-score)
        signal = zscore_per_day(raw)
        conviction = np.abs(signal)

        t = simulate_trades(
            day, signal, conviction, day['_n_agree'],
            day['_vol_pred'], hold_bars=HOLD_PERIODS[cfg['hold']],
            cost_spread_ticks=cost['spread_ticks'],
            cost_comm_ticks=cost['comm_ticks'],
            conviction_threshold=cfg['conv'],
            min_agreement=1,
            vol_percentile_min=cfg['vol_pct'],
            time_filter=cfg['time'],
        )
        for tt in t:
            tt['date'] = day['date']
        trades.extend(t)

    m = compute_metrics(trades, label='perm')
    return m['sharpe'], m['total_trades'], m['net_pnl_ticks']


def main():
    parser = argparse.ArgumentParser(description='Permutation test for CNN strategy')
    parser.add_argument('--n-perms', type=int, default=1000,
                        help='Number of permutations (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    n_perms = args.n_perms
    seed = args.seed

    # Load data
    oos_days = load_oos_data()

    # Run real backtest
    print("\n" + "=" * 70)
    print("STEP 1: Real backtest (de-biased best config)")
    print("=" * 70)
    real_metrics = run_backtest(oos_days, BEST_CONFIG)
    real_sharpe = real_metrics['sharpe']
    print(f"  Real Sharpe: {real_sharpe:.2f}")
    print(f"  Real trades: {real_metrics['total_trades']}")
    print(f"  Real PnL:    {real_metrics['net_pnl_ticks']:+.1f} ticks (${real_metrics['net_pnl_dollars']:+,.0f})")
    print(f"  Win rate:    {real_metrics['win_rate']:.1f}%")

    # Run permutation tests
    print(f"\n" + "=" * 70)
    print(f"STEP 2: Running {n_perms} permutation tests (seed={seed})")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    null_sharpes = np.zeros(n_perms)
    null_trades = np.zeros(n_perms, dtype=int)
    null_pnls = np.zeros(n_perms)

    t0 = time.time()
    for i in range(n_perms):
        sharpe, trades, pnl = shuffle_predictions_and_backtest(oos_days, BEST_CONFIG, rng)
        null_sharpes[i] = sharpe
        null_trades[i] = trades
        null_pnls[i] = pnl

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            print(f"  {i+1:4d}/{n_perms} done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)  "
                  f"null Sharpe range: [{null_sharpes[:i+1].min():.2f}, {null_sharpes[:i+1].max():.2f}]")

    total_time = time.time() - t0

    # Compute statistics
    print(f"\n" + "=" * 70)
    print("RESULTS: Permutation Test")
    print("=" * 70)

    null_mean = null_sharpes.mean()
    null_std = null_sharpes.std()
    z_score = (real_sharpe - null_mean) / null_std if null_std > 0 else np.inf

    # p-value: fraction of null Sharpes >= real Sharpe (one-sided)
    p_value = (null_sharpes >= real_sharpe).sum() / n_perms

    # Also compute empirical CDF position
    rank = (null_sharpes < real_sharpe).sum()
    percentile = 100.0 * rank / n_perms

    print(f"  Real Sharpe:     {real_sharpe:+.2f}")
    print(f"  Null mean:       {null_mean:+.4f}")
    print(f"  Null std:        {null_std:.4f}")
    print(f"  Null range:      [{null_sharpes.min():+.2f}, {null_sharpes.max():+.2f}]")
    print(f"  Z-score:         {z_score:+.2f}")
    print(f"  p-value:         {p_value:.6f}  {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")
    print(f"  Percentile:      {percentile:.1f}th")
    print(f"  Time:            {total_time:.1f}s ({total_time/n_perms:.2f}s per perm)")

    # Histogram data (for plotting if needed)
    hist_counts, hist_edges = np.histogram(null_sharpes, bins=50)

    # Interpretation
    print(f"\n  INTERPRETATION:")
    if p_value < 0.001:
        verdict = (f"HIGHLY SIGNIFICANT (p < 0.001). Real Sharpe {real_sharpe:.2f} is "
                   f"{z_score:.1f} standard deviations above the null mean. "
                   f"Random predictions cannot achieve this performance. "
                   f"The CNN signal contains genuine predictive information.")
    elif p_value < 0.01:
        verdict = (f"SIGNIFICANT (p < 0.01). Real Sharpe {real_sharpe:.2f} is unlikely "
                   f"to arise from random predictions alone.")
    elif p_value < 0.05:
        verdict = (f"MARGINALLY SIGNIFICANT (p < 0.05). Some evidence of real signal, "
                   f"but not conclusive. More data or OOT validation needed.")
    else:
        verdict = (f"NOT SIGNIFICANT (p = {p_value:.3f}). Cannot reject the null "
                   f"hypothesis. The observed Sharpe could arise from random predictions. "
                   f"DO NOT paper trade.")
    print(f"  {verdict}")

    # Additional diagnostic: how many null permutations had similar trade counts?
    similar_trades_mask = np.abs(null_trades - real_metrics['total_trades']) <= 10
    n_similar = similar_trades_mask.sum()
    if n_similar > 0:
        similar_sharpes = null_sharpes[similar_trades_mask]
        print(f"\n  Trade count diagnostic:")
        print(f"    Real trades: {real_metrics['total_trades']}")
        print(f"    Perms with similar trade count (+/-10): {n_similar}")
        print(f"    Sharpe of similar-count perms: {similar_sharpes.mean():+.4f} +/- {similar_sharpes.std():.4f}")

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        'test': 'permutation_null_test',
        'timestamp': ts,
        'n_permutations': n_perms,
        'seed': seed,
        'config': BEST_CONFIG,
        'real': {
            'sharpe': real_sharpe,
            'trades': real_metrics['total_trades'],
            'pnl_ticks': real_metrics['net_pnl_ticks'],
            'pnl_dollars': real_metrics['net_pnl_dollars'],
            'win_rate': real_metrics['win_rate'],
        },
        'null_distribution': {
            'mean': float(null_mean),
            'std': float(null_std),
            'min': float(null_sharpes.min()),
            'max': float(null_sharpes.max()),
            'median': float(np.median(null_sharpes)),
            'p5': float(np.percentile(null_sharpes, 5)),
            'p25': float(np.percentile(null_sharpes, 25)),
            'p75': float(np.percentile(null_sharpes, 75)),
            'p95': float(np.percentile(null_sharpes, 95)),
        },
        'statistics': {
            'z_score': float(z_score),
            'p_value': float(p_value),
            'percentile': float(percentile),
        },
        'verdict': verdict,
        'histogram': {
            'counts': hist_counts.tolist(),
            'edges': hist_edges.tolist(),
        },
        'null_sharpes': null_sharpes.tolist(),
        'null_trades': null_trades.tolist(),
        'elapsed_sec': total_time,
    }

    outfile = RESULTS_DIR / f'null_test_cnn_{ts}.json'
    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {outfile}")


if __name__ == '__main__':
    main()
