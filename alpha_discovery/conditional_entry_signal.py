"""
Conditional Entry Signal Generator
===================================
Instead of single-signal thresholds, this creates a MULTI-GATE signal:
1. Microprice deviation must be extreme (top 2% of recent distribution)
2. Pressure imbalance must AGREE with microprice direction
3. Depth slope must be favorable (book supporting the direction)
4. Volatility must be in favorable regime (not too high, not too low)
5. Spread must be tight (1 tick — favorable for fills)

Only generates a signal when ALL gates pass simultaneously.
This dramatically reduces trade frequency but should increase win rate.

The hypothesis: individual signals lose money because they fire too often
on weak setups. Requiring 4-5 confirmations filters to only the strongest
setups where multiple independent microstructure features agree.

Usage:
    python alpha_discovery/conditional_entry_signal.py --max-days 50
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
SIGNAL_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature indices
IDX_MID = 0
IDX_SPREAD = 1
IDX_IMBALANCE = 2
IDX_MICROPRICE = 3
IDX_TRADE_IMBALANCE = 10
IDX_BID_PRESSURE = 18
IDX_ASK_PRESSURE = 19
IDX_PRESSURE_IMBALANCE = 20
IDX_BID_SLOPE = 22
IDX_ASK_SLOPE = 23
IDX_SPREAD_TICKS = 24


def rolling_percentile_rank(arr, lookback=5000):
    """Compute rolling percentile rank (0-100) for each element."""
    n = len(arr)
    pct_rank = np.zeros(n)
    for i in range(lookback, n):
        window = arr[i - lookback:i]
        pct_rank[i] = np.searchsorted(np.sort(window), arr[i]) / lookback * 100
    return pct_rank


def rolling_std(arr, lookback=3000):
    """Compute rolling standard deviation."""
    n = len(arr)
    result = np.zeros(n)
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr ** 2)
    for i in range(lookback, n):
        s = cumsum[i] - cumsum[i - lookback]
        s2 = cumsum2[i] - cumsum2[i - lookback]
        var = s2 / lookback - (s / lookback) ** 2
        result[i] = np.sqrt(max(var, 0))
    return result


def generate_conditional_signal(gf, mid, config):
    """
    Generate signal that only fires when multiple conditions align.

    Config keys:
      microprice_pct: percentile threshold for microprice deviation (e.g. 98)
      pressure_agree: require pressure_imbalance to agree with microprice direction
      depth_agree: require depth slope to be favorable
      vol_regime: 'calm' (only trade in low vol) or 'any'
      tight_spread: only trade when spread <= 1 tick
      lookback: rolling window size
    """
    n = len(gf)
    lookback = config.get('lookback', 5000)
    microprice_pct = config.get('microprice_pct', 98)

    # Core signals
    microprice_dev = gf[:, IDX_MICROPRICE] - mid
    pressure = gf[:, IDX_PRESSURE_IMBALANCE]
    bid_slope = gf[:, IDX_BID_SLOPE]
    ask_slope = gf[:, IDX_ASK_SLOPE]
    spread_ticks = gf[:, IDX_SPREAD_TICKS]

    # Compute rolling percentile ranks
    micro_pct = rolling_percentile_rank(microprice_dev, lookback)

    # Compute rolling volatility
    returns = np.diff(mid, prepend=mid[0]) / np.maximum(mid, 1.0)
    vol = rolling_std(returns, 3000)
    median_vol = np.median(vol[3000:]) if len(vol) > 3000 else 0.001

    signal = np.zeros(n)
    gate_stats = {'total': 0, 'micro_pass': 0, 'pressure_pass': 0,
                  'depth_pass': 0, 'vol_pass': 0, 'spread_pass': 0, 'all_pass': 0}

    for i in range(lookback, n):
        gate_stats['total'] += 1

        # Gate 1: Microprice must be extreme
        is_bullish = micro_pct[i] >= microprice_pct
        is_bearish = micro_pct[i] <= (100 - microprice_pct)
        if not (is_bullish or is_bearish):
            continue
        gate_stats['micro_pass'] += 1

        direction = 1.0 if is_bullish else -1.0

        # Gate 2: Pressure imbalance agrees
        if config.get('pressure_agree', True):
            if direction > 0 and pressure[i] <= 0:
                continue
            if direction < 0 and pressure[i] >= 0:
                continue
        gate_stats['pressure_pass'] += 1

        # Gate 3: Depth slope favorable
        if config.get('depth_agree', True):
            if direction > 0 and ask_slope[i] >= bid_slope[i]:
                continue  # Want ask slope < bid slope (thin asks = bullish)
            if direction < 0 and bid_slope[i] >= ask_slope[i]:
                continue  # Want bid slope < ask slope (thin bids = bearish)
        gate_stats['depth_pass'] += 1

        # Gate 4: Volatility regime
        if config.get('vol_regime', 'any') == 'calm':
            if vol[i] > median_vol * 1.5:
                continue  # Skip high-vol periods
        gate_stats['vol_pass'] += 1

        # Gate 5: Tight spread
        if config.get('tight_spread', True):
            if spread_ticks[i] > 1.5:
                continue  # Only trade when spread <= 1 tick
        gate_stats['spread_pass'] += 1

        gate_stats['all_pass'] += 1

        # Signal magnitude proportional to how extreme the microprice is
        extremity = abs(micro_pct[i] - 50) / 50  # 0 to 1
        signal[i] = direction * extremity * abs(pressure[i])

    return signal, gate_stats


CONFIGS = {
    'strict': {
        'microprice_pct': 98,
        'pressure_agree': True,
        'depth_agree': True,
        'vol_regime': 'calm',
        'tight_spread': True,
        'lookback': 5000,
    },
    'moderate': {
        'microprice_pct': 95,
        'pressure_agree': True,
        'depth_agree': True,
        'vol_regime': 'any',
        'tight_spread': True,
        'lookback': 5000,
    },
    'relaxed': {
        'microprice_pct': 90,
        'pressure_agree': True,
        'depth_agree': False,
        'vol_regime': 'any',
        'tight_spread': False,
        'lookback': 3000,
    },
    'micro_pressure_only': {
        'microprice_pct': 97,
        'pressure_agree': True,
        'depth_agree': False,
        'vol_regime': 'any',
        'tight_spread': False,
        'lookback': 5000,
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-days', type=int, default=50)
    parser.add_argument('--configs', type=str, default='strict,moderate,relaxed,micro_pressure_only')
    args = parser.parse_args()

    configs_to_run = args.configs.split(',')
    snap_files = sorted(SNAP_DIR.glob('*_snapshots.npz'))[:args.max_days]
    print(f"Generating conditional signals for {len(snap_files)} days, configs: {configs_to_run}")

    all_results = {}

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")
        print(f"  Params: {json.dumps(config, indent=2)}")

        day_stats = []
        generated = 0

        for f in snap_files:
            date_str = f.stem.replace('_snapshots', '')
            data = np.load(str(f))
            gf = data['global_features']
            mid = data['mid_prices']

            signal, gate_stats = generate_conditional_signal(gf, mid, config)
            signal = signal.astype(np.float64)

            # Stats
            nonzero = np.abs(signal) > 0.001
            nonzero_pct = float(nonzero.mean())

            # Quick IC check
            fwd_ret = np.zeros(len(mid))
            fwd_ret[:len(mid)-100] = mid[100:] - mid[:len(mid)-100]
            mask = np.isfinite(signal) & np.isfinite(fwd_ret) & (np.abs(signal) > 0.001)
            if mask.sum() > 30:
                s, r = signal[mask], fwd_ret[mask]
                s_dm = s - s.mean()
                r_dm = r - r.mean()
                denom = np.sqrt((s_dm**2).sum() * (r_dm**2).sum())
                ic = float((s_dm * r_dm).sum() / max(denom, 1e-12))
            else:
                ic = 0.0

            stats = {
                'date': date_str,
                'nonzero_pct': nonzero_pct,
                'ic_10s': ic,
                'signals_fired': int(nonzero.sum()),
                'gate_pass_rate': gate_stats['all_pass'] / max(gate_stats['total'], 1),
                'gate_stats': gate_stats,
            }
            day_stats.append(stats)

            # Save prediction file
            out = SIGNAL_DIR / f'cond_{config_name}_{date_str}.npz'
            np.savez_compressed(str(out), predictions=signal, mid_prices=mid)
            generated += 1

            if generated % 10 == 0:
                recent_ics = [s['ic_10s'] for s in day_stats[-10:]]
                recent_signals = [s['signals_fired'] for s in day_stats[-10:]]
                print(f"  [{generated}/{len(snap_files)}] "
                      f"mean IC={np.mean(recent_ics):+.4f}, "
                      f"avg signals/day={np.mean(recent_signals):.0f}, "
                      f"gate pass={day_stats[-1]['gate_pass_rate']:.3%}")

        # Summary
        ics = [s['ic_10s'] for s in day_stats]
        signals_per_day = [s['signals_fired'] for s in day_stats]
        gate_rates = [s['gate_pass_rate'] for s in day_stats]

        summary = {
            'config': config_name,
            'mean_ic': float(np.mean(ics)),
            'std_ic': float(np.std(ics)),
            'positive_pct': float((np.array(ics) > 0).mean()),
            't_stat': float(np.mean(ics) / max(np.std(ics) / np.sqrt(len(ics)), 1e-6)),
            'avg_signals_per_day': float(np.mean(signals_per_day)),
            'avg_gate_pass_rate': float(np.mean(gate_rates)),
            'days_generated': generated,
        }
        all_results[config_name] = summary

        print(f"\n  SUMMARY ({config_name}):")
        print(f"  Mean IC @ 10s: {summary['mean_ic']:+.4f}")
        print(f"  IC std: {summary['std_ic']:.4f}")
        print(f"  Positive IC days: {summary['positive_pct']:.1%}")
        print(f"  t-stat: {summary['t_stat']:.2f}")
        print(f"  Avg signals/day: {summary['avg_signals_per_day']:.0f}")
        print(f"  Avg gate pass rate: {summary['avg_gate_pass_rate']:.3%}")

    # Save summary
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'conditional_entry_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'experiment': 'conditional_entry',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Signal files: {SIGNAL_DIR}/cond_*.npz")
