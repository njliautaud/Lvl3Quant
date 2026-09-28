"""
lgbm_feature_screen.py — Fast LGBM-based feature importance screen for ES book data.

Tests which derived features from the order book have predictive power for
ES futures 10-second (horizon_bars=100) forward price moves.

Usage:
    python lgbm_feature_screen.py                         # defaults: all features
    python lgbm_feature_screen.py --features raw          # only raw book features
    python lgbm_feature_screen.py --features all          # all derived features
    python lgbm_feature_screen.py --features ofi,spread   # specific derived features
    python lgbm_feature_screen.py --horizon 100 --train-days 60 --oot-days 20

Output:
    - Console: IC, top-20 features, feature group breakdown, ablation table
    - JSON: saved to results/autoresearch/lgbm_feature_screen_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- Path setup (mirrors research_harness.py) ----
ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

DEFAULT_BOOK_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_book_cache')
DEFAULT_OUTPUT_DIR = str(MODELS_DIR / 'results' / 'autoresearch')

logging.basicConfig(
    format='%(asctime)s [lgbm_screen] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('lgbm_screen')

# ---- Feature channel indices (post log1p transform) ----
FEAT_PRICE  = 0   # price_relative_to_mid (ticks, signed)
FEAT_DEPTH  = 1   # depth_lots (log1p)
FEAT_ORDERS = 2   # num_orders (log1p)
FEAT_AGE    = 3   # queue_age_seconds (log1p)
N_LEVELS    = 20
N_BID       = 10
EPS         = 1e-6

RAW_CH_NAMES = ['price_rel', 'depth', 'orders', 'age']

DERIVED_FEATURE_NAMES = [
    'order_flow_imbalance',
    'pressure_gradient',
    'spread',
    'queue_age_momentum',
    'depth_change_velocity',
    'book_asymmetry',
    'cumulative_depth_ratio',
]

# Short aliases accepted on command line
FEATURE_ALIASES = {
    'ofi': 'order_flow_imbalance',
    'pg': 'pressure_gradient',
    'sp': 'spread',
    'qam': 'queue_age_momentum',
    'dcv': 'depth_change_velocity',
    'ba': 'book_asymmetry',
    'cdr': 'cumulative_depth_ratio',
}


# ============================================================================
# Data loading (mirrors research_harness.py)
# ============================================================================

def get_available_dates(book_dir: str) -> List[str]:
    files = sorted(Path(book_dir).glob('*_book_tensors.npz'))
    return [f.name.replace('_book_tensors.npz', '') for f in files]


def load_book_data(book_dir: str, dates: List[str]):
    """
    Load and log1p-transform book tensors for the given dates.

    Returns:
        tensors:      (N, 20, 4) float32  — log1p applied to depth/orders/age
        mid_prices:   (N,)       float64
        day_boundaries: list of ints [0, n_day0, ...]
    """
    all_tensors, all_mids, boundaries = [], [], [0]
    for date in dates:
        fpath = Path(book_dir) / f'{date}_book_tensors.npz'
        if not fpath.exists():
            logger.warning(f'Missing: {fpath.name}, skipping')
            continue
        npz = np.load(fpath)
        t = npz['book_tensors'].astype(np.float32)
        # Log-transform depth / orders / age (same as BookTensorDataset)
        t[:, :, FEAT_DEPTH]  = np.log1p(t[:, :, FEAT_DEPTH])
        t[:, :, FEAT_ORDERS] = np.log1p(t[:, :, FEAT_ORDERS])
        t[:, :, FEAT_AGE]    = np.log1p(t[:, :, FEAT_AGE])
        all_tensors.append(t)
        all_mids.append(npz['mid_prices'])
        boundaries.append(boundaries[-1] + len(npz['mid_prices']))
    if not all_tensors:
        raise FileNotFoundError(f'No book tensor files found in {book_dir}')
    return (
        np.concatenate(all_tensors, axis=0),
        np.concatenate(all_mids, axis=0),
        boundaries,
    )


def compute_forward_return(
    mid_prices: np.ndarray,
    day_boundaries: List[int],
    horizon_bars: int = 100,
    tick_size: float = 0.25,
) -> np.ndarray:
    """
    Forward return in ticks at horizon_bars.  NaN where window crosses day boundary.
    """
    N = len(mid_prices)
    labels = np.full(N, np.nan, dtype=np.float32)
    n_days = len(day_boundaries) - 1
    for d in range(n_days):
        start = day_boundaries[d]
        end   = day_boundaries[d + 1]
        valid_end = end - horizon_bars
        if valid_end > start:
            labels[start:valid_end] = (
                (mid_prices[start + horizon_bars : end] - mid_prices[start:valid_end])
                / tick_size
            ).astype(np.float32)
    return labels


def split_dates(all_dates: List[str], train_days: int, oot_days: int):
    total = train_days + oot_days
    if len(all_dates) < total:
        logger.warning(
            f'Only {len(all_dates)} dates available; '
            f'using all with last {oot_days} as OOT.'
        )
        return all_dates[:-oot_days], all_dates[-oot_days:]
    subset = all_dates[-total:]
    return subset[:train_days], subset[train_days:]


# ============================================================================
# Feature engineering (pure numpy, mirrors feature_engineering.py semantics)
# ============================================================================

def _apply_log1p_if_needed(t: np.ndarray) -> np.ndarray:
    """Tensors loaded by load_book_data are already log1p'd — this is a no-op guard."""
    return t


# ---------- individual derived feature helpers ----------
# Each takes (N, 20, 4) and returns (N, n_cols) with named cols

def feat_order_flow_imbalance(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    bid_depth = t[:, :N_BID, FEAT_DEPTH]
    ask_depth = t[:, N_BID:, FEAT_DEPTH]
    sum_bid = bid_depth.sum(axis=1)
    sum_ask = ask_depth.sum(axis=1)
    ofi = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPS)
    return ofi[:, None], ['ofi']


def feat_pressure_gradient(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # Per-level depth gradient; summarised as mean + std per side
    bid_depth = t[:, :N_BID, FEAT_DEPTH]
    ask_depth = t[:, N_BID:, FEAT_DEPTH]
    bid_grad = np.diff(bid_depth, axis=1)  # (N, 9)
    ask_grad = np.diff(ask_depth, axis=1)
    cols = np.column_stack([
        bid_grad.mean(axis=1), bid_grad.std(axis=1),
        ask_grad.mean(axis=1), ask_grad.std(axis=1),
    ])
    names = ['pg_bid_mean', 'pg_bid_std', 'pg_ask_mean', 'pg_ask_std']
    return cols, names


def feat_spread(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # best_ask (level 10) price_rel - best_bid (level 9) price_rel
    spread = (t[:, N_BID, FEAT_PRICE] - t[:, N_BID - 1, FEAT_PRICE]).clip(min=0.0)
    return spread[:, None], ['spread_ticks']


def feat_queue_age_momentum(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # age delta: current bar vs prior bar (already the most-recent bar in a window context)
    # For single-bar tabular: we use the raw age values aggregated by side as proxy
    bid_age = t[:, :N_BID, FEAT_AGE]
    ask_age = t[:, N_BID:, FEAT_AGE]
    cols = np.column_stack([
        bid_age.mean(axis=1),
        ask_age.mean(axis=1),
        bid_age.max(axis=1),
        ask_age.max(axis=1),
        bid_age.mean(axis=1) - ask_age.mean(axis=1),   # age differential
    ])
    names = ['qam_bid_mean', 'qam_ask_mean', 'qam_bid_max', 'qam_ask_max', 'qam_diff']
    return cols, names


def feat_depth_change_velocity(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    # Single-bar tabular: use bid vs ask depth totals and best-level depth as proxies
    bid_total = t[:, :N_BID, FEAT_DEPTH].sum(axis=1)
    ask_total = t[:, N_BID:, FEAT_DEPTH].sum(axis=1)
    bid_best  = t[:, 0, FEAT_DEPTH]
    ask_best  = t[:, N_BID, FEAT_DEPTH]
    cols = np.column_stack([bid_total, ask_total, bid_best, ask_best, bid_total - ask_total])
    names = ['dcv_bid_total', 'dcv_ask_total', 'dcv_bid_best', 'dcv_ask_best', 'dcv_net']
    return cols, names


def feat_book_asymmetry(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    bid_depth = t[:, :N_BID, FEAT_DEPTH]
    ask_depth = t[:, N_BID:, FEAT_DEPTH]
    asym = bid_depth - ask_depth  # per-level
    cols = np.column_stack([
        asym.mean(axis=1),
        asym.std(axis=1),
        asym[:, 0],   # best-level asymmetry
        asym[:, 2],   # level-3 asymmetry
    ])
    names = ['asym_mean', 'asym_std', 'asym_best', 'asym_l3']
    return cols, names


def feat_cumulative_depth_ratio(t: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    bid_depth = t[:, :N_BID, FEAT_DEPTH]
    ask_depth = t[:, N_BID:, FEAT_DEPTH]
    cum_bid = np.cumsum(bid_depth, axis=1)  # (N, 10)
    cum_ask = np.cumsum(ask_depth, axis=1)
    ratio   = cum_bid / (cum_ask + EPS)
    cols = np.column_stack([
        ratio[:, 0],   # top-1 level ratio
        ratio[:, 2],   # top-3 level ratio
        ratio[:, 4],   # top-5 level ratio
        ratio[:, 9],   # full ratio
    ])
    names = ['cdr_top1', 'cdr_top3', 'cdr_top5', 'cdr_full']
    return cols, names


DERIVED_REGISTRY = {
    'order_flow_imbalance': feat_order_flow_imbalance,
    'pressure_gradient':    feat_pressure_gradient,
    'spread':               feat_spread,
    'queue_age_momentum':   feat_queue_age_momentum,
    'depth_change_velocity': feat_depth_change_velocity,
    'book_asymmetry':       feat_book_asymmetry,
    'cumulative_depth_ratio': feat_cumulative_depth_ratio,
}


# ============================================================================
# Tabular feature construction
# ============================================================================

def build_raw_features(tensors: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    80 raw features: all 20 levels x 4 channels.
    Feature name = level{i}_{channel}.
    """
    N = tensors.shape[0]
    names = []
    for lvl in range(N_LEVELS):
        side = 'bid' if lvl < N_BID else 'ask'
        side_lvl = lvl if lvl < N_BID else lvl - N_BID
        for ch, ch_name in enumerate(RAW_CH_NAMES):
            names.append(f'{side}_l{side_lvl}_{ch_name}')
    features = tensors.reshape(N, N_LEVELS * 4).astype(np.float32)
    return features, names


def build_aggregate_features(tensors: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate features:
      - mean/std/max/min per channel per side  (32)
      - depth imbalance, top-3, top-1, order imbalance  (4)
      - slope of depth across levels per side  (2)
      - weighted mid offset  (1)
      - queue age diff  (1)
    Total: 40
    """
    feature_list, names = [], []

    for side_name, sl in [('bid', slice(0, N_BID)), ('ask', slice(N_BID, N_LEVELS))]:
        side_data = tensors[:, sl, :]  # (N, 10, 4)
        for ch, ch_name in enumerate(RAW_CH_NAMES):
            cd = side_data[:, :, ch]
            feature_list += [cd.mean(1), cd.std(1), cd.max(1), cd.min(1)]
            names += [f'agg_{side_name}_{ch_name}_{s}' for s in ('mean', 'std', 'max', 'min')]

    # Depth imbalance
    bid_d = tensors[:, :N_BID, FEAT_DEPTH].sum(1)
    ask_d = tensors[:, N_BID:, FEAT_DEPTH].sum(1)
    tot_d = bid_d + ask_d + EPS
    feature_list.append(bid_d / tot_d);            names.append('agg_depth_imbalance')
    bid3  = tensors[:, :3, FEAT_DEPTH].sum(1)
    ask3  = tensors[:, N_BID:N_BID+3, FEAT_DEPTH].sum(1)
    feature_list.append(bid3 / (bid3 + ask3 + EPS)); names.append('agg_top3_imbalance')
    bid1  = tensors[:, 0, FEAT_DEPTH]
    ask1  = tensors[:, N_BID, FEAT_DEPTH]
    feature_list.append(bid1 / (bid1 + ask1 + EPS)); names.append('agg_top1_imbalance')

    # Order imbalance
    bid_o = tensors[:, :N_BID, FEAT_ORDERS].sum(1)
    ask_o = tensors[:, N_BID:, FEAT_ORDERS].sum(1)
    feature_list.append(bid_o / (bid_o + ask_o + EPS)); names.append('agg_order_imbalance')

    # Depth slope per side
    levels_x = np.arange(N_BID, dtype=np.float32) - (N_BID - 1) / 2
    x_var = (levels_x ** 2).sum()
    for side_name, sl in [('bid', slice(0, N_BID)), ('ask', slice(N_BID, N_LEVELS))]:
        depths = tensors[:, sl, FEAT_DEPTH]
        slope = (depths * levels_x[None, :]).sum(1) / max(x_var, EPS)
        feature_list.append(slope); names.append(f'agg_{side_name}_depth_slope')

    # Weighted mid offset
    bp = tensors[:, 0, FEAT_PRICE];  ap = tensors[:, N_BID, FEAT_PRICE]
    bd = tensors[:, 0, FEAT_DEPTH];  ad = tensors[:, N_BID, FEAT_DEPTH]
    simple_mid = (bp + ap) / 2.0
    denom = bd + ad
    wt_mid = np.where(denom > 0, (bp * ad + ap * bd) / denom, simple_mid)
    feature_list.append(wt_mid - simple_mid); names.append('agg_weighted_mid_offset')

    # Spread
    feature_list.append((ap - bp).clip(min=0.0)); names.append('agg_spread')

    # Queue age diff
    ba = tensors[:, :N_BID, FEAT_AGE].mean(1)
    aa = tensors[:, N_BID:, FEAT_AGE].mean(1)
    feature_list.append(ba - aa); names.append('agg_queue_age_diff')

    features = np.column_stack(feature_list).astype(np.float32)
    return features, names


def build_derived_features(
    tensors: np.ndarray,
    enabled: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Build derived features for the given list of enabled feature names."""
    if not enabled:
        return np.empty((len(tensors), 0), dtype=np.float32), []
    parts, names = [], []
    for feat_name in enabled:
        fn = DERIVED_REGISTRY[feat_name]
        arr, ns = fn(tensors)
        # Prefix with feature group name
        parts.append(arr.astype(np.float32))
        names.extend([f'{feat_name[:4]}_{n}' for n in ns])
    return np.concatenate(parts, axis=1), names


def build_all_features(
    tensors: np.ndarray,
    enabled_derived: List[str],
    include_raw: bool = True,
    include_agg: bool = True,
) -> Tuple[np.ndarray, List[str], Dict[str, List[int]]]:
    """
    Build the complete feature matrix.

    Returns:
        X:           (N, total_features) float32
        feat_names:  list of feature name strings
        group_idx:   dict mapping group name -> list of column indices
    """
    parts, names, group_idx = [], [], {}
    offset = 0

    if include_raw:
        raw_X, raw_names = build_raw_features(tensors)
        parts.append(raw_X)
        group_idx['raw'] = list(range(offset, offset + raw_X.shape[1]))
        names.extend(raw_names)
        offset += raw_X.shape[1]

    if include_agg:
        agg_X, agg_names = build_aggregate_features(tensors)
        parts.append(agg_X)
        group_idx['aggregate'] = list(range(offset, offset + agg_X.shape[1]))
        names.extend(agg_names)
        offset += agg_X.shape[1]

    if enabled_derived:
        der_X, der_names = build_derived_features(tensors, enabled_derived)
        parts.append(der_X)
        group_idx['derived'] = list(range(offset, offset + der_X.shape[1]))
        names.extend(der_names)
        offset += der_X.shape[1]

    X = np.concatenate(parts, axis=1) if parts else np.empty((len(tensors), 0), dtype=np.float32)
    return X, names, group_idx


# ============================================================================
# LGBM helpers
# ============================================================================

LGBM_PARAMS = {
    'objective':          'regression',
    'metric':             'rmse',
    'learning_rate':      0.05,
    'num_leaves':         63,
    'max_depth':          7,
    'min_child_samples':  200,    # larger floor for LOB data (very autocorrelated)
    'subsample':          0.8,
    'colsample_bytree':   0.8,
    'reg_alpha':          0.1,
    'reg_lambda':         1.0,
    'verbose':            -1,
    'n_jobs':             -1,
    'seed':               42,
}


def train_lgbm(X_train, y_train, X_oot, y_oot, feat_names: List[str]):
    import lightgbm as lgb
    from scipy.stats import spearmanr

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_names, free_raw_data=False)
    dval   = lgb.Dataset(X_oot,   label=y_oot,   feature_name=feat_names,
                         reference=dtrain, free_raw_data=False)

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    preds_oot   = model.predict(X_oot)
    preds_train = model.predict(X_train)

    def ic(p, y):
        mask = np.isfinite(p) & np.isfinite(y)
        if mask.sum() < 10:
            return 0.0
        r, _ = spearmanr(p[mask], y[mask])
        return float(r) if np.isfinite(r) else 0.0

    return model, ic(preds_oot, y_oot), ic(preds_train, y_train)


def feature_importance_report(
    model,
    feat_names: List[str],
    group_idx: Dict[str, List[int]],
    top_n: int = 20,
) -> Tuple[Dict, Dict]:
    raw_imp = model.feature_importance(importance_type='gain').astype(float)
    total   = raw_imp.sum()
    norm    = raw_imp / total if total > 0 else raw_imp

    # Top-N features
    top_idx = np.argsort(norm)[::-1][:top_n]
    top_features = {feat_names[i]: round(float(norm[i]), 5) for i in top_idx}

    # Group-level importance
    group_imp = {}
    for group, idxs in group_idx.items():
        group_imp[group] = round(float(norm[idxs].sum()), 5)

    return top_features, group_imp


# ============================================================================
# Main run logic
# ============================================================================

def parse_feature_arg(feat_arg: str) -> List[str]:
    """Parse --features argument into a list of canonical feature names."""
    feat_arg = feat_arg.strip().lower()
    if feat_arg in ('all', ''):
        return list(DERIVED_REGISTRY.keys())
    if feat_arg == 'none':
        return []
    parts = [p.strip() for p in feat_arg.replace(',', ' ').split()]
    resolved = []
    for p in parts:
        if p in FEATURE_ALIASES:
            p = FEATURE_ALIASES[p]
        if p not in DERIVED_REGISTRY:
            logger.warning(f'Unknown feature "{p}", skipping. Valid: {list(DERIVED_REGISTRY)}')
            continue
        if p not in resolved:
            resolved.append(p)
    return resolved


def run_screen(
    book_dir: str,
    horizon_bars: int,
    train_days: int,
    oot_days: int,
    enabled_derived: List[str],
    subsample: int,
    output_dir: str,
    top_n: int = 20,
) -> Dict:
    """Run the full LGBM feature screen and return a results dict."""
    try:
        import lightgbm  # noqa — just verify available
    except ImportError:
        logger.error(
            'lightgbm is not installed. Install with: pip install lightgbm\n'
            'Then re-run: python lgbm_feature_screen.py'
        )
        sys.exit(1)

    from scipy.stats import spearmanr  # noqa — verify scipy

    t0 = time.time()
    logger.info('=' * 65)
    logger.info('LGBM Feature Importance Screen')
    logger.info(f'  horizon_bars={horizon_bars}, train_days={train_days}, oot_days={oot_days}')
    logger.info(f'  derived features: {enabled_derived or ["(none)"] }')
    logger.info(f'  subsample every {subsample} bar(s)')
    logger.info('=' * 65)

    # ---- Load dates ----
    all_dates = get_available_dates(book_dir)
    if not all_dates:
        raise FileNotFoundError(f'No book tensor NPZ files found in: {book_dir}')
    train_dates, oot_dates = split_dates(all_dates, train_days, oot_days)
    logger.info(f'  Train: {train_dates[0]} .. {train_dates[-1]}  ({len(train_dates)} days)')
    logger.info(f'  OOT:   {oot_dates[0]} .. {oot_dates[-1]}  ({len(oot_dates)} days)')

    # ---- Load data ----
    logger.info('  Loading train data ...')
    train_t, train_mid, train_bounds = load_book_data(book_dir, train_dates)
    train_labels = compute_forward_return(train_mid, train_bounds, horizon_bars)

    logger.info('  Loading OOT data ...')
    oot_t, oot_mid, oot_bounds = load_book_data(book_dir, oot_dates)
    oot_labels = compute_forward_return(oot_mid, oot_bounds, horizon_bars)

    # ---- Filter valid rows ----
    train_mask = np.isfinite(train_labels)
    oot_mask   = np.isfinite(oot_labels)

    if subsample > 1:
        # Subsample by taking every Nth row (preserve time order)
        sub_idx_train = np.where(train_mask)[0][::subsample]
        # OOT also subsampled (same rate) to keep evaluation fast;
        # Spearman IC is stable with large N — we don't need all bars
        sub_idx_oot   = np.where(oot_mask)[0][::subsample]
    else:
        sub_idx_train = np.where(train_mask)[0]
        sub_idx_oot   = np.where(oot_mask)[0]

    # Hard cap: limit to 500K train / 200K OOT for speed.
    # Spearman IC on book data is stable well below these thresholds.
    MAX_TRAIN = 500_000
    MAX_OOT   = 200_000
    if len(sub_idx_train) > MAX_TRAIN:
        step = max(1, len(sub_idx_train) // MAX_TRAIN)
        sub_idx_train = sub_idx_train[::step]
    if len(sub_idx_oot) > MAX_OOT:
        step = max(1, len(sub_idx_oot) // MAX_OOT)
        sub_idx_oot = sub_idx_oot[::step]

    train_t_valid   = train_t[sub_idx_train]
    train_lbl_valid = train_labels[sub_idx_train]
    oot_t_valid     = oot_t[sub_idx_oot]
    oot_lbl_valid   = oot_labels[sub_idx_oot]

    logger.info(
        f'  Samples — train: {len(train_lbl_valid):,}  OOT: {len(oot_lbl_valid):,}'
    )

    # ---- Build features ----
    logger.info('  Building features ...')
    t_feat = time.time()
    X_train, feat_names, group_idx = build_all_features(
        train_t_valid, enabled_derived, include_raw=True, include_agg=True
    )
    X_oot, _, _ = build_all_features(
        oot_t_valid, enabled_derived, include_raw=True, include_agg=True
    )
    logger.info(
        f'  Feature matrix: {X_train.shape[1]} cols  '
        f'({time.time() - t_feat:.1f}s build time)'
    )
    logger.info(f'  Groups: { {k: len(v) for k, v in group_idx.items()} }')

    # Replace any NaN / inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_oot   = np.nan_to_num(X_oot,   nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Train full model ----
    logger.info('  Training LGBM (all features) ...')
    t_train = time.time()
    model_full, oot_ic, train_ic = train_lgbm(
        X_train, train_lbl_valid, X_oot, oot_lbl_valid, feat_names
    )
    elapsed_train = time.time() - t_train
    logger.info(f'  Done in {elapsed_train:.1f}s  |  OOT IC={oot_ic:.5f}  Train IC={train_ic:.5f}')

    top_features, group_imp = feature_importance_report(
        model_full, feat_names, group_idx, top_n=top_n
    )

    # ---- Ablation: raw-only ----
    ablation = {}
    if enabled_derived:
        logger.info('  Ablation: raw features only ...')
        X_raw_train, raw_names, raw_gidx = build_all_features(
            train_t_valid, [], include_raw=True, include_agg=True
        )
        X_raw_oot, _, _ = build_all_features(
            oot_t_valid, [], include_raw=True, include_agg=True
        )
        X_raw_train = np.nan_to_num(X_raw_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_raw_oot   = np.nan_to_num(X_raw_oot,   nan=0.0, posinf=0.0, neginf=0.0)
        _, raw_ic, raw_train_ic = train_lgbm(
            X_raw_train, train_lbl_valid, X_raw_oot, oot_lbl_valid, raw_names
        )
        ic_lift = oot_ic - raw_ic
        ablation = {
            'raw_only_ic':   round(raw_ic, 6),
            'full_ic':       round(oot_ic, 6),
            'ic_lift':       round(ic_lift, 6),
            'lift_pct':      round(ic_lift / abs(raw_ic) * 100, 2) if abs(raw_ic) > 1e-6 else None,
        }
        logger.info(
            f'  Raw IC={raw_ic:.5f}  Full IC={oot_ic:.5f}  '
            f'Lift={ic_lift:+.5f} ({ablation["lift_pct"]}%)'
        )

    # ---- Per-group ablation (leave-one-out group importance) ----
    per_derived_ic = {}
    if len(enabled_derived) > 1:
        logger.info('  Per-derived-feature IC (add-one) ...')
        for feat_name in enabled_derived:
            Xi_train, ni, _ = build_all_features(
                train_t_valid, [feat_name], include_raw=True, include_agg=True
            )
            Xi_oot, _, _ = build_all_features(
                oot_t_valid, [feat_name], include_raw=True, include_agg=True
            )
            Xi_train = np.nan_to_num(Xi_train, nan=0.0, posinf=0.0, neginf=0.0)
            Xi_oot   = np.nan_to_num(Xi_oot,   nan=0.0, posinf=0.0, neginf=0.0)
            _, ic_i, _ = train_lgbm(Xi_train, train_lbl_valid, Xi_oot, oot_lbl_valid, ni)
            per_derived_ic[feat_name] = round(ic_i, 6)
            logger.info(f'    {feat_name}: IC={ic_i:.5f}')

    elapsed_total = time.time() - t0

    # ---- Build result dict ----
    result = {
        'run_type':        'lgbm_feature_screen',
        'timestamp':       datetime.now().isoformat(timespec='seconds'),
        'horizon_bars':    horizon_bars,
        'train_days':      len(train_dates),
        'oot_days':        len(oot_dates),
        'train_range':     f'{train_dates[0]}..{train_dates[-1]}',
        'oot_range':       f'{oot_dates[0]}..{oot_dates[-1]}',
        'train_samples':   int(len(train_lbl_valid)),
        'oot_samples':     int(len(oot_lbl_valid)),
        'subsample':       subsample,
        'n_features':      int(X_train.shape[1]),
        'enabled_derived': enabled_derived,
        'oot_ic':          round(oot_ic, 6),
        'train_ic':        round(train_ic, 6),
        'overfit_ratio':   round(train_ic / oot_ic, 3) if abs(oot_ic) > 1e-6 else None,
        'best_iteration':  model_full.best_iteration,
        'feature_group_importance': group_imp,
        'top_features':    top_features,
        'ablation_raw_vs_full': ablation,
        'per_derived_feature_ic': per_derived_ic,
        'elapsed_seconds': round(elapsed_total, 1),
    }

    # ---- Pretty print to console ----
    _print_report(result)

    # ---- Save JSON ----
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(output_dir) / f'lgbm_feature_screen_{ts}.json'
    with open(out_path, 'w') as fh:
        json.dump(result, fh, indent=2)
    logger.info(f'\n  Results saved -> {out_path}')

    return result


def _print_report(r: Dict):
    sep = '-' * 65
    print(f'\n{"=" * 65}')
    print(f'  LGBM Feature Screen Results')
    print(f'{"=" * 65}')
    print(f'  Horizon:      {r["horizon_bars"]} bars ({r["horizon_bars"] * 0.1:.0f}s)')
    print(f'  Train period: {r["train_range"]}  ({r["train_days"]} days)')
    print(f'  OOT period:   {r["oot_range"]}  ({r["oot_days"]} days)')
    print(f'  Samples:      {r["train_samples"]:,} train  |  {r["oot_samples"]:,} OOT')
    print(f'  Features:     {r["n_features"]} total')
    print()
    print(f'  OOT IC   : {r["oot_ic"]:+.5f}')
    print(f'  Train IC : {r["train_ic"]:+.5f}')
    print(f'  Overfit  : {r["overfit_ratio"]}x')
    print(f'  Elapsed  : {r["elapsed_seconds"]}s')
    print()

    if r.get('feature_group_importance'):
        print(f'{sep}')
        print('  Feature Group Importance (gain %)')
        print(f'{sep}')
        for grp, imp in sorted(r['feature_group_importance'].items(), key=lambda x: -x[1]):
            bar = '#' * int(imp * 40)
            print(f'  {grp:<20} {imp*100:5.1f}%  {bar}')
        print()

    if r.get('ablation_raw_vs_full'):
        abl = r['ablation_raw_vs_full']
        print(f'{sep}')
        print('  Ablation: Raw+Aggregate vs Raw+Aggregate+Derived')
        print(f'{sep}')
        print(f'  Raw+Agg IC  : {abl["raw_only_ic"]:+.5f}')
        print(f'  Full IC     : {abl["full_ic"]:+.5f}')
        lift = abl["ic_lift"]
        pct  = abl["lift_pct"]
        print(f'  IC lift     : {lift:+.5f}  ({pct}%)')
        print()

    if r.get('per_derived_feature_ic'):
        print(f'{sep}')
        print('  Per-Derived Feature IC (raw+agg + each feature)')
        print(f'{sep}')
        for feat, ic_val in sorted(r['per_derived_feature_ic'].items(), key=lambda x: -x[1]):
            print(f'  {feat:<30} IC={ic_val:+.5f}')
        print()

    if r.get('top_features'):
        print(f'{sep}')
        print(f'  Top-{len(r["top_features"])} Features by Gain Importance')
        print(f'{sep}')
        for rank, (fname, imp) in enumerate(r['top_features'].items(), 1):
            bar = '#' * int(imp * 400)
            print(f'  {rank:>2}. {fname:<35} {imp*100:5.2f}%  {bar}')
        print()

    print('=' * 65)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LGBM feature importance screen for ES book data (10-second prediction).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lgbm_feature_screen.py
  python lgbm_feature_screen.py --features all
  python lgbm_feature_screen.py --features ofi,spread,ba
  python lgbm_feature_screen.py --horizon 100 --train-days 60 --oot-days 20 --subsample 5

Derived feature names (full or short alias):
  order_flow_imbalance (ofi)    pressure_gradient (pg)
  spread (sp)                   queue_age_momentum (qam)
  depth_change_velocity (dcv)   book_asymmetry (ba)
  cumulative_depth_ratio (cdr)
        """
    )
    parser.add_argument(
        '--book-dir', default=DEFAULT_BOOK_DIR,
        help=f'Directory of *_book_tensors.npz files (default: {DEFAULT_BOOK_DIR})'
    )
    parser.add_argument(
        '--horizon', type=int, default=100,
        help='Forward return horizon in bars (default: 100 = 10s at 100ms bars)'
    )
    parser.add_argument(
        '--train-days', type=int, default=60,
        help='Number of trading days for training set (default: 60)'
    )
    parser.add_argument(
        '--oot-days', type=int, default=20,
        help='Number of trading days for OOT test set (default: 20)'
    )
    parser.add_argument(
        '--features', default='all',
        help=(
            'Derived features to include: "all", "none", or comma-separated names. '
            'Default: all'
        )
    )
    parser.add_argument(
        '--subsample', type=int, default=3,
        help='Subsample training data (every Nth bar, speeds up run). Default: 3'
    )
    parser.add_argument(
        '--top-n', type=int, default=20,
        help='How many top features to report. Default: 20'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help=f'Where to save JSON results (default: {DEFAULT_OUTPUT_DIR})'
    )
    args = parser.parse_args()

    enabled_derived = parse_feature_arg(args.features)
    logger.info(f'Enabled derived features: {enabled_derived}')

    run_screen(
        book_dir=args.book_dir,
        horizon_bars=args.horizon,
        train_days=args.train_days,
        oot_days=args.oot_days,
        enabled_derived=enabled_derived,
        subsample=args.subsample,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == '__main__':
    main()
