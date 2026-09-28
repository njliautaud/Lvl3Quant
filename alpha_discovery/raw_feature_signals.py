"""
Raw Feature Signal Generator — Test individual features as direct trading signals.

Based on lead-lag analysis findings:
1. cancel_asym_5 LEADS OFI by 0.5s with 2.8x strength
2. depth_ratio_l1 has IC=+0.087 @10s (100% consistent)
3. ask_L1_orders has SLOWEST IC decay (25% remaining at 30s)
4. order_frag_asym has IC=+0.078 @10s

For each feature, we:
- Extract it directly as a signal
- Optionally gate by vol regime
- Optionally combine with complementary features
- Save as prediction NPZ files for Rust MBO sim

Usage:
    python alpha_discovery/raw_feature_signals.py
    python alpha_discovery/raw_feature_signals.py --signal cancel_asym --n-days 50
"""

import sys
import time
import json
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy.stats import pearsonr

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.mbo_features import get_feature_names

# Setup logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_log_file = RESULTS_DIR / f"raw_feature_signals_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("raw_signals")

# Constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
BARS_PER_SEC = 10

if platform.system() == 'Windows':
    FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
    SNAP_CACHE = LVL3_ROOT / "data" / "processed" / "medium_snapshots_cache"
else:
    FEATURE_CACHE = Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache"
    SNAP_CACHE = Path.home() / "lvl3quant" / "data" / "processed" / "medium_snapshots_cache"

OUTPUT_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build feature name -> index map
ALL_NAMES = get_feature_names()
NAME_TO_IDX = {name: idx for idx, name in enumerate(ALL_NAMES)}


# ============================================================================
# Signal Definitions
# ============================================================================

SIGNAL_CONFIGS = {
    'cancel_asym': {
        'description': 'Cancel asymmetry (leads OFI by 0.5s, 2.8x strength)',
        'feature': 'cancel_asym_5',
        'sign': -1,  # negative cancel_asym_5 = asks cancelling more = price going up? Check sign
        'gate_vol': True,
    },
    'cancel_asym_chain': {
        'description': 'Cancel asym + OFI causal chain combo',
        'features': ['cancel_asym_5', 'ofi_5'],
        'weights': [-0.5, 0.5],  # combine both signals
        'gate_vol': True,
    },
    'depth_ratio': {
        'description': 'L1 depth ratio (highest raw IC=0.087)',
        'feature': 'depth_ratio_l1',
        'sign': 1,
        'gate_vol': False,
    },
    'depth_ratio_z': {
        'description': 'Depth ratio z-score 500 (abnormal book imbalance)',
        'feature': 'depth_ratio_l1_zscore_500',
        'sign': 1,
        'gate_vol': False,
    },
    'ask_orders': {
        'description': 'Ask L1 orders (slowest IC decay, 25% at 30s)',
        'feature': 'ask_L1_orders',
        'sign': -1,  # more ask orders = price goes down
        'gate_vol': False,
    },
    'order_frag': {
        'description': 'Order fragmentation asymmetry (IC=0.078)',
        'feature': 'order_frag_asym',
        'sign': 1,
        'gate_vol': False,
    },
    'book_imb_z': {
        'description': 'Book imbalance z-score 50 (short-term abnormal imbalance)',
        'feature': 'book_imb_zscore_50',
        'sign': 1,
        'gate_vol': True,
    },
    'slow_decay_combo': {
        'description': 'Combo of slowest-decaying features (most tradeable)',
        'features': ['ask_L1_orders', 'weighted_book_imb', 'pressure_imbalance',
                     'book_imb_zscore_500', 'queue_depletion_asymmetry_5'],
        'weights': [-1.0, 1.0, 1.0, 1.0, -1.0],
        'gate_vol': True,
    },
    'causal_chain': {
        'description': 'Full causal chain: cancel_asym -> OFI -> depth_ratio',
        'features': ['cancel_asym_5', 'ofi_5', 'depth_ratio_l1', 'depth_ratio_l1_zscore_500'],
        'weights': [-0.3, 0.3, 0.2, 0.2],
        'gate_vol': True,
    },
    'top5_ensemble': {
        'description': 'Equal-weight ensemble of top 5 IC features',
        'features': ['depth_ratio_l1', 'depth_ratio_l1_zscore_500',
                     'depth_ratio_l1_zscore_50', 'order_frag_asym', 'book_imb_zscore_50'],
        'weights': [0.2, 0.2, 0.2, 0.2, 0.2],
        'gate_vol': False,
    },
}


def discover_days() -> List[Tuple[str, Path]]:
    """Find available days with feature caches."""
    days = []
    for f in sorted(FEATURE_CACHE.glob("*_mbo_features.npz")):
        date_str = f.stem.replace("_mbo_features", "")
        days.append((date_str, f))
    return days


def load_day(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one day of 340-feature data + mid prices."""
    data = np.load(str(fpath))
    features = data['mbo_features']  # (N, 340)
    mid_prices = features[:, 0].copy()  # col 0 = mid
    return features, mid_prices


def compute_forward_return(mid: np.ndarray, horizon_bars: int = 100) -> np.ndarray:
    """Compute forward return in ticks."""
    ret = np.full(len(mid), np.nan)
    if len(mid) > horizon_bars:
        ret[:-horizon_bars] = (mid[horizon_bars:] - mid[:-horizon_bars]) / TICK_SIZE
    return ret


def compute_vol_regime(features: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Compute rolling vol regime (high/low). Returns boolean mask for high vol."""
    # Compute 1-bar returns
    returns = np.diff(mid, prepend=mid[0]) / np.maximum(mid, 1e-10)

    # Rolling std over 500 bars (50 seconds)
    window = 500
    vol = np.full(len(returns), np.nan)
    cumsum = np.cumsum(returns ** 2)
    cumsum_padded = np.concatenate([[0], cumsum])
    for i in range(window, len(returns)):
        vol[i] = np.sqrt((cumsum_padded[i + 1] - cumsum_padded[i - window + 1]) / window)

    # Rolling percentile rank over 5000 bars
    rank_window = 5000
    is_high = np.zeros(len(returns), dtype=bool)
    for i in range(rank_window, len(returns)):
        if np.isnan(vol[i]):
            continue
        window_vals = vol[max(0, i - rank_window):i]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            rank = np.mean(valid <= vol[i])
            is_high[i] = rank >= 0.70  # top 30%

    return is_high


def generate_single_feature_signal(features: np.ndarray, config: dict,
                                   vol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate signal from a single feature."""
    feat_name = config['feature']
    idx = NAME_TO_IDX.get(feat_name)
    if idx is None:
        logger.warning(f"Feature {feat_name} not found in feature names!")
        return np.zeros(len(features))

    raw = features[:, idx].copy() * config.get('sign', 1)

    # Normalize to zero mean, unit std (rolling 5000-bar window)
    # Use separate array to avoid contaminating rolling stats with z-scored values
    signal = np.zeros_like(raw)
    window = 5000
    for i in range(window, len(raw)):
        chunk = raw[max(0, i - window):i]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 10:
            mu = np.mean(valid)
            std = np.std(valid)
            if std > 1e-10:
                signal[i] = (raw[i] - mu) / std
            else:
                signal[i] = 0.0
        else:
            signal[i] = 0.0

    # Vol gate
    if config.get('gate_vol', False) and vol_mask is not None:
        signal[~vol_mask] = 0.0

    return signal.astype(np.float32)


def generate_combo_signal(features: np.ndarray, config: dict,
                          vol_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Generate signal from weighted combination of features."""
    feat_names = config['features']
    weights = config['weights']

    signal = np.zeros(len(features), dtype=np.float64)

    for name, weight in zip(feat_names, weights):
        idx = NAME_TO_IDX.get(name)
        if idx is None:
            logger.warning(f"Feature {name} not found!")
            continue

        raw = features[:, idx].copy()

        # Normalize each feature independently (rolling)
        window = 5000
        normalized = np.zeros_like(raw)
        for i in range(window, len(raw)):
            chunk = raw[max(0, i - window):i]
            valid = chunk[np.isfinite(chunk)]
            if len(valid) > 10:
                mu = np.mean(valid)
                std = np.std(valid)
                if std > 1e-10:
                    normalized[i] = (raw[i] - mu) / std

        signal += weight * normalized

    # Vol gate
    if config.get('gate_vol', False) and vol_mask is not None:
        signal[~vol_mask] = 0.0

    return signal.astype(np.float32)


def compute_ic(signal: np.ndarray, target: np.ndarray) -> float:
    """Compute Pearson IC, handling NaN."""
    mask = np.isfinite(signal) & np.isfinite(target) & (signal != 0.0)
    if mask.sum() < 100:
        return 0.0
    try:
        corr, _ = pearsonr(signal[mask], target[mask])
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal', type=str, default='all',
                       choices=list(SIGNAL_CONFIGS.keys()) + ['all'])
    parser.add_argument('--n-days', type=int, default=0, help='0 = all')
    parser.add_argument('--horizon', type=int, default=100, help='Forward return horizon in bars')
    parser.add_argument('--skip-existing', action='store_true', help='Skip days with existing output')
    parser.add_argument('--force', action='store_true', help='Regenerate even if output exists')
    args = parser.parse_args()

    signals_to_run = list(SIGNAL_CONFIGS.keys()) if args.signal == 'all' else [args.signal]

    days = discover_days()
    if args.n_days > 0:
        days = days[:args.n_days]

    logger.info(f"Running {len(signals_to_run)} signal(s) on {len(days)} days")
    logger.info(f"Horizon: {args.horizon} bars ({args.horizon / BARS_PER_SEC:.1f}s)")
    logger.info(f"Output: {OUTPUT_DIR}")

    all_results = {}

    for sig_name in signals_to_run:
        config = SIGNAL_CONFIGS[sig_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Signal: {sig_name} — {config['description']}")
        logger.info(f"{'='*60}")

        ics = []
        n_active_bars_total = 0
        n_total_bars = 0

        for day_idx, (date_str, fpath) in enumerate(days):
            try:
                out_path = OUTPUT_DIR / f"{sig_name}_{date_str}.npz"
                if args.skip_existing and out_path.exists() and not args.force:
                    continue

                features, mid_prices = load_day(fpath)
                fwd_ret = compute_forward_return(mid_prices, args.horizon)

                # Compute vol mask if needed
                vol_mask = None
                if config.get('gate_vol', False):
                    vol_mask = compute_vol_regime(features, mid_prices)

                # Generate signal
                if 'features' in config:
                    signal = generate_combo_signal(features, config, vol_mask)
                else:
                    signal = generate_single_feature_signal(features, config, vol_mask)

                # Compute IC
                ic = compute_ic(signal, fwd_ret)
                active = np.sum(signal != 0.0)
                ics.append(ic)
                n_active_bars_total += active
                n_total_bars += len(features)

                # Save prediction file
                np.savez_compressed(str(out_path), predictions=signal)

                if (day_idx + 1) % 5 == 0 or day_idx == 0:
                    logger.info(f"  [{day_idx+1}/{len(days)}] {date_str}  IC={ic:+.4f}  "
                              f"active={active:,} ({100*active/len(features):.1f}%)")

                del features, mid_prices, fwd_ret, signal

            except Exception as e:
                logger.error(f"  {date_str}: ERROR: {e}")
                ics.append(0.0)

        # Summary
        ics_arr = np.array(ics)
        valid_ics = ics_arr[ics_arr != 0.0]
        n_valid = len(valid_ics)
        mean_ic = np.mean(valid_ics) if n_valid > 0 else 0.0
        std_ic = np.std(valid_ics) if n_valid > 0 else 0.0
        t_stat = mean_ic / (std_ic / np.sqrt(n_valid)) if std_ic > 0 and n_valid > 0 else 0.0
        pct_pos = 100 * np.mean(valid_ics > 0) if n_valid > 0 else 0.0

        logger.info(f"\n  {sig_name:25s} | n={n_valid}  mean_IC={mean_ic:+.4f}  "
                    f"std={std_ic:.4f}  t={t_stat:.2f}  pct+={pct_pos:.0f}%  "
                    f"active_rate={100*n_active_bars_total/max(n_total_bars,1):.1f}%")

        all_results[sig_name] = {
            'description': config['description'],
            'n_days': n_valid,
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            't_stat': float(t_stat),
            'pct_positive': float(pct_pos),
            'active_rate': float(n_active_bars_total / max(n_total_bars, 1)),
            'per_day_ic': [float(x) for x in ics],
        }

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL SUMMARY — Raw Feature Signals (IC vs {args.horizon/BARS_PER_SEC:.0f}s forward return)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Signal':25s} {'n':>5s} {'mean_IC':>10s} {'std':>8s} {'t-stat':>8s} {'pct+':>6s} {'active':>8s}")
    logger.info("-" * 80)

    sorted_results = sorted(all_results.items(), key=lambda x: abs(x[1]['mean_ic']), reverse=True)
    for name, r in sorted_results:
        logger.info(f"{name:25s} {r['n_days']:5d} {r['mean_ic']:+10.4f} {r['std_ic']:8.4f} "
                   f"{r['t_stat']:8.2f} {r['pct_positive']:5.0f}% {100*r['active_rate']:7.1f}%")

    logger.info(f"{'='*80}")

    # Save results
    out_file = RESULTS_DIR / f"raw_feature_signals_{_ts}.json"
    with open(str(out_file), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'horizon_bars': args.horizon,
            'n_days': len(days),
            'results': all_results,
        }, f, indent=2)
    logger.info(f"\nResults: {out_file}")
    logger.info(f"Predictions saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
