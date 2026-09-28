"""
Explore Untapped Feature Groups — Scan 150 features across 15 groups for directional IC.

Groups that are ALREADY in the 9-signal composite: A, B, E, F, I, J, O, T (partially)
Groups that are COMPLETELY UNTAPPED: C, D, G, H, K, M, N, P, Q, R, S, U, V, W, X

For each untapped feature, we compute rolling z-score signal and IC vs 10s forward return.
Features with IC > 0.03 and t-stat > 3 are candidates for the expanded composite.

Usage:
    python alpha_discovery/explore_untapped_features.py
    python alpha_discovery/explore_untapped_features.py --n-days 20  # quick scan
"""

import sys
import time
import json
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
from scipy.stats import pearsonr

sys.stdout.reconfigure(line_buffering=True)

LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))
from alpha_discovery.mbo_features import get_feature_names

# Logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_log_file = RESULTS_DIR / f"untapped_features_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("untapped")

TICK_SIZE = 0.25
BARS_PER_SEC = 10

if platform.system() == 'Windows':
    FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
else:
    FEATURE_CACHE = Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache"

OUTPUT_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_NAMES = get_feature_names()
NAME_TO_IDX = {name: idx for idx, name in enumerate(ALL_NAMES)}

# Define untapped feature groups and their features
# sign: +1 means higher value = bullish, -1 means higher = bearish
UNTAPPED_FEATURES = {
    # Group C: VPIN (informed trading probability)
    'vpin_20': {'idx': 111, 'sign': -1, 'group': 'C', 'desc': 'VPIN 20-bar (informed trading proxy)'},
    'vpin_50': {'idx': 112, 'sign': -1, 'group': 'C', 'desc': 'VPIN 50-bar'},
    'vpin_100': {'idx': 113, 'sign': -1, 'group': 'C', 'desc': 'VPIN 100-bar'},

    # Group D: Price momentum
    'ret_5': {'idx': 114, 'sign': 1, 'group': 'D', 'desc': '5-bar return (momentum)'},
    'ret_vel_5': {'idx': 115, 'sign': 1, 'group': 'D', 'desc': '5-bar return velocity'},
    'ret_10': {'idx': 116, 'sign': 1, 'group': 'D', 'desc': '10-bar return'},
    'ret_vel_10': {'idx': 117, 'sign': 1, 'group': 'D', 'desc': '10-bar return velocity'},
    'ret_20': {'idx': 118, 'sign': 1, 'group': 'D', 'desc': '20-bar return'},
    'ret_50': {'idx': 120, 'sign': 1, 'group': 'D', 'desc': '50-bar return'},
    'ret_100': {'idx': 122, 'sign': 1, 'group': 'D', 'desc': '100-bar return'},

    # Group G: Toxicity/adverse selection
    'kyle_lambda_20': {'idx': 140, 'sign': -1, 'group': 'G', 'desc': 'Kyle lambda 20-bar (market impact)'},
    'kyle_lambda_50': {'idx': 141, 'sign': -1, 'group': 'G', 'desc': 'Kyle lambda 50-bar'},
    'price_impact': {'idx': 142, 'sign': -1, 'group': 'G', 'desc': 'Price impact'},
    'adverse_sel': {'idx': 143, 'sign': -1, 'group': 'G', 'desc': 'Adverse selection'},
    'toxicity_score': {'idx': 144, 'sign': -1, 'group': 'G', 'desc': 'Toxicity composite'},

    # Group H: Microstructure strategies
    'iceberg_score': {'idx': 145, 'sign': 1, 'group': 'H', 'desc': 'Iceberg order detection'},
    'absorption_bid': {'idx': 146, 'sign': 1, 'group': 'H', 'desc': 'Bid absorption (buying)'},
    'absorption_ask': {'idx': 147, 'sign': -1, 'group': 'H', 'desc': 'Ask absorption (selling)'},
    'large_trade_rev': {'idx': 148, 'sign': 1, 'group': 'H', 'desc': 'Large trade reversal signal'},
    'spoof_score': {'idx': 149, 'sign': 1, 'group': 'H', 'desc': 'Spoofing detection'},
    'aggressive_burst': {'idx': 150, 'sign': 1, 'group': 'H', 'desc': 'Aggressive order burst'},
    'sweep_count': {'idx': 151, 'sign': 1, 'group': 'H', 'desc': 'Market sweep count'},
    'momentum_ignition': {'idx': 152, 'sign': 1, 'group': 'H', 'desc': 'Momentum ignition detection'},
    'book_flip': {'idx': 153, 'sign': 1, 'group': 'H', 'desc': 'Book side flip'},
    'hidden_liq': {'idx': 154, 'sign': 1, 'group': 'H', 'desc': 'Hidden liquidity'},

    # Group K: Vol-Direction interaction
    'vol_accel': {'idx': 196, 'sign': 1, 'group': 'K', 'desc': 'Vol acceleration'},
    'flow_during_vol': {'idx': 197, 'sign': 1, 'group': 'K', 'desc': 'Flow during high vol'},
    'cancel_asym_vol': {'idx': 198, 'sign': -1, 'group': 'K', 'desc': 'Cancel asymmetry x vol'},
    'aggr_momentum': {'idx': 199, 'sign': 1, 'group': 'K', 'desc': 'Aggressive momentum'},

    # Group M: Rolling book shape
    'entropy_change_5': {'idx': 200, 'sign': 1, 'group': 'M', 'desc': 'Book entropy change 5-bar'},
    'entropy_change_20': {'idx': 201, 'sign': 1, 'group': 'M', 'desc': 'Book entropy change 20-bar'},
    'cog_asym_5': {'idx': 202, 'sign': 1, 'group': 'M', 'desc': 'Center of gravity asymmetry 5-bar'},
    'cog_asym_20': {'idx': 203, 'sign': 1, 'group': 'M', 'desc': 'Center of gravity asymmetry 20-bar'},
    'depth_accel': {'idx': 204, 'sign': 1, 'group': 'M', 'desc': 'Depth acceleration'},
    'wall_bid_persist_20': {'idx': 206, 'sign': 1, 'group': 'M', 'desc': 'Bid wall persistence'},
    'wall_ask_persist_20': {'idx': 207, 'sign': -1, 'group': 'M', 'desc': 'Ask wall persistence'},
    'l1_ratio_shift_5': {'idx': 209, 'sign': 1, 'group': 'M', 'desc': 'L1 ratio shift'},

    # Group N: Rolling quote dynamics
    'l1_instability_5': {'idx': 210, 'sign': -1, 'group': 'N', 'desc': 'L1 instability 5-bar'},
    'depletion_momentum_5': {'idx': 212, 'sign': 1, 'group': 'N', 'desc': 'Depletion momentum 5-bar'},
    'depletion_dir_20': {'idx': 213, 'sign': 1, 'group': 'N', 'desc': 'Depletion direction 20-bar'},
    'sell_pressure_5': {'idx': 214, 'sign': -1, 'group': 'N', 'desc': 'Sell pressure 5-bar'},
    'sell_pressure_20': {'idx': 215, 'sign': -1, 'group': 'N', 'desc': 'Sell pressure 20-bar'},
    'reposition_intensity_5': {'idx': 217, 'sign': 1, 'group': 'N', 'desc': 'Repositioning intensity'},
    'institutional_flow_5': {'idx': 219, 'sign': 1, 'group': 'N', 'desc': 'Institutional flow 5-bar'},

    # Group P: Magnitude predictors
    'book_thinning': {'idx': 250, 'sign': 1, 'group': 'P', 'desc': 'Book thinning (volatility precursor)'},
    'stacked_depth_asym': {'idx': 251, 'sign': 1, 'group': 'P', 'desc': 'Stacked depth asymmetry'},
    'flow_acceleration': {'idx': 252, 'sign': 1, 'group': 'P', 'desc': 'Flow acceleration'},
    'cross_flow_agreement': {'idx': 254, 'sign': 1, 'group': 'P', 'desc': 'Cross-flow agreement'},
    'toxicity_spike': {'idx': 255, 'sign': -1, 'group': 'P', 'desc': 'Toxicity spike'},
    'institutional_signal': {'idx': 258, 'sign': 1, 'group': 'P', 'desc': 'Institutional signal'},

    # Group Q: Order flow sequences
    'run_direction_20': {'idx': 262, 'sign': 1, 'group': 'Q', 'desc': 'Run direction 20-bar'},
    'flow_reversal_5': {'idx': 263, 'sign': -1, 'group': 'Q', 'desc': 'Flow reversal 5-bar'},
    'flow_persistence_20': {'idx': 264, 'sign': 1, 'group': 'Q', 'desc': 'Flow persistence 20-bar'},
    'momentum_exhaustion': {'idx': 268, 'sign': -1, 'group': 'Q', 'desc': 'Momentum exhaustion'},
    'flow_regime_zscore': {'idx': 269, 'sign': 1, 'group': 'Q', 'desc': 'Flow regime z-score'},

    # Group R: Depth-weighted OFI + Staleness
    'dwfi_5': {'idx': 270, 'sign': 1, 'group': 'R', 'desc': 'Depth-weighted flow imbalance 5-bar'},
    'dwfi_20': {'idx': 271, 'sign': 1, 'group': 'R', 'desc': 'Depth-weighted flow imbalance 20-bar'},
    'ofi_l3_5': {'idx': 272, 'sign': 1, 'group': 'R', 'desc': 'L3 order flow imbalance 5-bar'},
    'ofi_deep_5': {'idx': 273, 'sign': 1, 'group': 'R', 'desc': 'Deep book OFI 5-bar'},
    'ofi_deep_vs_l1': {'idx': 274, 'sign': 1, 'group': 'R', 'desc': 'Deep OFI vs L1 divergence'},

    # Group S: Cross-signal interactions
    'return_autocorr_10': {'idx': 280, 'sign': 1, 'group': 'S', 'desc': 'Return autocorrelation 10-bar'},
    'return_autocorr_50': {'idx': 281, 'sign': 1, 'group': 'S', 'desc': 'Return autocorrelation 50-bar'},
    'imb_spread_interaction': {'idx': 284, 'sign': 1, 'group': 'S', 'desc': 'Imbalance x spread interaction'},
    'toxicity_imb_combo': {'idx': 286, 'sign': 1, 'group': 'S', 'desc': 'Toxicity x imbalance combo'},
    'multi_signal_strength': {'idx': 289, 'sign': 1, 'group': 'S', 'desc': 'Multi-signal agreement strength'},

    # Group U: Trade Clustering/Herding
    'burst_same_dir_5': {'idx': 300, 'sign': 1, 'group': 'U', 'desc': 'Same-direction burst 5-bar'},
    'burst_same_dir_20': {'idx': 301, 'sign': 1, 'group': 'U', 'desc': 'Same-direction burst 20-bar'},
    'trade_persistence_5': {'idx': 302, 'sign': 1, 'group': 'U', 'desc': 'Trade persistence 5-bar'},
    'herding_intensity_5': {'idx': 304, 'sign': 1, 'group': 'U', 'desc': 'Herding intensity 5-bar'},
    'herding_intensity_20': {'idx': 305, 'sign': 1, 'group': 'U', 'desc': 'Herding intensity 20-bar'},

    # Group V: Cross-Timescale Momentum
    'autocorr_1bar': {'idx': 310, 'sign': 1, 'group': 'V', 'desc': '1-bar autocorrelation'},
    'autocorr_10bar': {'idx': 311, 'sign': 1, 'group': 'V', 'desc': '10-bar autocorrelation'},
    'momentum_alignment_short': {'idx': 313, 'sign': 1, 'group': 'V', 'desc': 'Short-term momentum alignment'},
    'momentum_divergence': {'idx': 315, 'sign': 1, 'group': 'V', 'desc': 'Momentum divergence (multi-scale)'},
    'momentum_reversal_50': {'idx': 318, 'sign': -1, 'group': 'V', 'desc': 'Momentum reversal 50-bar'},
    'momentum_reversal_200': {'idx': 319, 'sign': -1, 'group': 'V', 'desc': 'Momentum reversal 200-bar'},

    # Group W: Spread Dynamics
    'spread_change_velocity_5': {'idx': 320, 'sign': 1, 'group': 'W', 'desc': 'Spread change velocity 5-bar'},
    'spread_return_corr_50': {'idx': 328, 'sign': 1, 'group': 'W', 'desc': 'Spread-return correlation 50-bar'},
    'spread_return_corr_200': {'idx': 329, 'sign': 1, 'group': 'W', 'desc': 'Spread-return correlation 200-bar'},

    # Group X: Size Classification
    'size_imbalance_5': {'idx': 333, 'sign': 1, 'group': 'X', 'desc': 'Order size imbalance 5-bar'},
    'size_imbalance_20': {'idx': 334, 'sign': 1, 'group': 'X', 'desc': 'Order size imbalance 20-bar'},
    'institutional_flow_proxy_5': {'idx': 335, 'sign': 1, 'group': 'X', 'desc': 'Institutional flow proxy 5-bar'},
    'institutional_flow_proxy_20': {'idx': 336, 'sign': 1, 'group': 'X', 'desc': 'Institutional flow proxy 20-bar'},
    'order_size_divergence': {'idx': 337, 'sign': 1, 'group': 'X', 'desc': 'Order size divergence'},
}


def discover_days() -> List[Tuple[str, Path]]:
    days = []
    for f in sorted(FEATURE_CACHE.glob("*_mbo_features.npz")):
        date_str = f.stem.replace("_mbo_features", "")
        days.append((date_str, f))
    return days


def rolling_zscore(raw: np.ndarray, window: int = 5000) -> np.ndarray:
    """Fully vectorized rolling z-score using cumulative sums — no Python loops."""
    n = len(raw)
    if n <= window:
        return np.zeros(n, dtype=np.float32)

    # Replace non-finite with 0
    clean = np.where(np.isfinite(raw), raw, 0.0).astype(np.float64)

    # Padded cumulative sums (prepend 0 for easy window subtraction)
    cs = np.concatenate([[0.0], np.cumsum(clean)])
    cs2 = np.concatenate([[0.0], np.cumsum(clean ** 2)])

    # For indices window..n-1: rolling sum over [i-window, i-1]
    idx = np.arange(window, n)
    roll_sum = cs[idx] - cs[idx - window]       # sum of clean[i-window..i-1]
    roll_sum2 = cs2[idx] - cs2[idx - window]    # sum of clean^2[i-window..i-1]

    mu = roll_sum / window
    var = roll_sum2 / window - mu ** 2

    # Z-score: (x[i] - mu) / std, only where var > 0
    signal = np.zeros(n, dtype=np.float32)
    valid = var > 1e-20
    std = np.sqrt(np.where(valid, var, 1.0))
    signal[window:] = np.where(valid, (clean[window:] - mu) / std, 0.0).astype(np.float32)

    return signal


def compute_ic(signal: np.ndarray, target: np.ndarray) -> float:
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
    parser.add_argument('--n-days', type=int, default=0, help='0 = all days')
    parser.add_argument('--horizon', type=int, default=100, help='Forward return horizon in bars (100=10s)')
    parser.add_argument('--save-signals', action='store_true',
                        help='Save prediction NPZ files for features with IC > 0.03')
    args = parser.parse_args()

    days = discover_days()
    if args.n_days > 0:
        days = days[:args.n_days]

    logger.info(f"Exploring {len(UNTAPPED_FEATURES)} untapped features across {len(days)} days")
    logger.info(f"Horizon: {args.horizon} bars ({args.horizon / BARS_PER_SEC:.1f}s)")
    logger.info(f"Feature cache: {FEATURE_CACHE}")

    # Track ICs per feature
    feature_ics = {name: [] for name in UNTAPPED_FEATURES}
    t_start = time.time()

    for day_idx, (date_str, fpath) in enumerate(days):
        try:
            data = np.load(str(fpath))
            features = data['mbo_features']
            mid = features[:, 0].copy()
            n_bars = len(features)

            # Forward return
            fwd_ret = np.full(n_bars, np.nan)
            if n_bars > args.horizon:
                fwd_ret[:-args.horizon] = (mid[args.horizon:] - mid[:-args.horizon]) / TICK_SIZE

            # Process each untapped feature
            for feat_name, config in UNTAPPED_FEATURES.items():
                idx = config['idx']
                raw = features[:, idx].copy() * config['sign']
                signal = rolling_zscore(raw)

                ic = compute_ic(signal, fwd_ret)
                feature_ics[feat_name].append(ic)

                # Save if requested and IC is decent
                if args.save_signals and abs(ic) > 0.02:
                    out_path = OUTPUT_DIR / f"{feat_name}_{date_str}.npz"
                    np.savez_compressed(str(out_path), predictions=signal)

            del features, mid, fwd_ret

            elapsed = time.time() - t_start
            rate = (day_idx + 1) / elapsed
            eta = (len(days) - day_idx - 1) / rate if rate > 0 else 0
            if (day_idx + 1) % 5 == 0 or day_idx == 0:
                logger.info(f"  [{day_idx+1}/{len(days)}] {date_str} ({elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            logger.error(f"  {date_str}: ERROR: {e}")
            for name in UNTAPPED_FEATURES:
                feature_ics[name].append(0.0)

    # Compute summary statistics
    logger.info(f"\n{'='*100}")
    logger.info(f"UNTAPPED FEATURE IC SCAN — {len(days)} days, horizon={args.horizon/BARS_PER_SEC:.0f}s")
    logger.info(f"{'='*100}")
    logger.info(f"{'Feature':35s} {'Group':>5s} {'n':>4s} {'mean_IC':>10s} {'std':>8s} {'t-stat':>8s} {'pct+':>6s} {'Sign':>5s}")
    logger.info("-" * 100)

    results = {}
    for name, config in UNTAPPED_FEATURES.items():
        ics = np.array(feature_ics[name])
        valid = ics[ics != 0.0]
        n = len(valid)
        mean_ic = np.mean(valid) if n > 0 else 0.0
        std_ic = np.std(valid) if n > 0 else 0.0
        t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 0 else 0.0
        pct_pos = 100 * np.mean(valid > 0) if n > 0 else 0.0

        results[name] = {
            'group': config['group'],
            'description': config['desc'],
            'sign': config['sign'],
            'idx': config['idx'],
            'n_days': int(n),
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            't_stat': float(t_stat),
            'pct_positive': float(pct_pos),
            'per_day_ic': [float(x) for x in ics],
        }

    # Sort by absolute IC
    sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['mean_ic']), reverse=True)
    for name, r in sorted_results:
        marker = " ***" if abs(r['mean_ic']) > 0.05 else " **" if abs(r['mean_ic']) > 0.03 else ""
        logger.info(f"{name:35s} {r['group']:>5s} {r['n_days']:4d} {r['mean_ic']:+10.4f} "
                   f"{r['std_ic']:8.4f} {r['t_stat']:8.2f} {r['pct_positive']:5.0f}% "
                   f"{'+' if r['sign']==1 else '-':>5s}{marker}")

    # Group summary
    logger.info(f"\n{'='*80}")
    logger.info("GROUP SUMMARY (best feature per group)")
    logger.info(f"{'='*80}")
    groups = {}
    for name, r in results.items():
        grp = r['group']
        if grp not in groups or abs(r['mean_ic']) > abs(groups[grp][1]['mean_ic']):
            groups[grp] = (name, r)

    for grp in sorted(groups.keys()):
        name, r = groups[grp]
        logger.info(f"  Group {grp}: {name:30s} IC={r['mean_ic']:+.4f} t={r['t_stat']:.1f}")

    # Candidates for composite
    candidates = [(n, r) for n, r in sorted_results if abs(r['mean_ic']) > 0.03 and abs(r['t_stat']) > 3]
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPOSITE CANDIDATES (|IC| > 0.03, |t| > 3): {len(candidates)}")
    logger.info(f"{'='*80}")
    for name, r in candidates:
        logger.info(f"  {name:35s} IC={r['mean_ic']:+.4f} t={r['t_stat']:.1f} group={r['group']}")

    # If --save-signals and candidates found, also generate for all 100 days
    if args.save_signals and candidates:
        logger.info(f"\nSaving prediction NPZ files for {len(candidates)} candidate features...")
        n_saved = 0
        for day_idx, (date_str, fpath) in enumerate(days):
            try:
                data = np.load(str(fpath))
                features = data['mbo_features']
                for name, r in candidates:
                    config = UNTAPPED_FEATURES[name]
                    raw = features[:, config['idx']].copy() * config['sign']
                    signal = rolling_zscore(raw)
                    out_path = OUTPUT_DIR / f"{name}_{date_str}.npz"
                    np.savez_compressed(str(out_path), predictions=signal)
                    n_saved += 1
                del features
            except Exception as e:
                logger.error(f"  Save error {date_str}: {e}")
        logger.info(f"  Saved {n_saved} prediction files")

    # Save results JSON
    out_file = RESULTS_DIR / f"untapped_features_{_ts}.json"
    with open(str(out_file), 'w') as f:
        json.dump({
            'timestamp': _ts,
            'horizon_bars': args.horizon,
            'n_days': len(days),
            'n_features_scanned': len(UNTAPPED_FEATURES),
            'n_candidates': len(candidates),
            'candidate_names': [n for n, _ in candidates],
            'results': results,
        }, f, indent=2)
    logger.info(f"\nResults saved: {out_file}")
    logger.info(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == '__main__':
    main()
