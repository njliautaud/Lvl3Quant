"""
Deep Feature Importance Analysis — 340 MBO Features
=====================================================
Identifies which of the 340 MBO order-book features actually drive predictions,
using LightGBM split/gain, permutation importance, SHAP, feature-group ablation,
and per-feature raw IC.

Usage:
    python alpha_discovery/feature_importance_deep.py
    python alpha_discovery/feature_importance_deep.py --n-days 50
    python alpha_discovery/feature_importance_deep.py --quick

Output:
    alpha_discovery/results/feature_importance_{timestamp}.json
"""

import sys
import argparse
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

# ── project root on sys.path ──────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(LVL3_ROOT / "alpha_discovery"))

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
HORIZON_BARS = 100                   # 10 s at 100 ms/bar
MAX_TRAIN_ROWS = 500_000
SHAP_SAMPLE   = 50_000
TOP_N         = 30
TOP_RAW_IC    = 50

LGBM_PARAMS = dict(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 7,
    num_leaves        = 63,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    min_child_samples = 500,
    n_jobs            = -1,
    random_state      = 42,
    verbose           = -1,
)

# Feature indices to always exclude (leakage / price levels)
LEAKAGE_INDICES = {0, 1, 8, 9, 3, 26, 27, 28, 29}   # mid, spread, best_bid, best_ask,
                                                       # microprice, hour_norm, minute_norm,
                                                       # time_since_rth, time_to_close

# Feature groups defined by column ranges
# (label, start_inclusive, end_exclusive)
FEATURE_GROUPS = [
    ("A_static_snapshot",        0,   96),
    ("B_rolling_orderflow",     96,  111),
    ("C_vpin",                 111,  114),
    ("D_price_momentum",       114,  124),
    ("E_realized_vol",         124,  130),
    ("F_book_shape_dynamics",  130,  140),
    ("G_toxicity",             140,  145),
    ("H_microstructure_strat", 145,  155),
    ("I_mbo_enhanced_rolling", 155,  175),
    ("J_spatial_per_level",    175,  195),
    ("K_vol_direction",        195,  200),
    ("M_rolling_book_shape",   200,  210),
    ("N_rolling_quote_dyn",    210,  220),
    ("O_cross_timeframe_z",    220,  250),
    ("P_magnitude_predictors", 250,  260),
    ("Q_orderflow_sequences",  260,  270),
    ("R_depth_weighted_ofi",   270,  280),
    ("S_cross_signal_inter",   280,  290),
    ("T_queue_dynamics",       290,  300),
    ("U_trade_clustering",     300,  310),
    ("V_cross_timescale_mom",  310,  320),
    ("W_spread_dynamics",      320,  330),
    ("X_size_classification",  330,  340),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_ic(signal: np.ndarray, forward_ret: np.ndarray) -> float:
    """Pearson correlation (IC) between signal and forward return."""
    mask = np.isfinite(signal) & np.isfinite(forward_ret)
    if mask.sum() < 100:
        return 0.0
    s = signal[mask].astype(np.float64)
    r = forward_ret[mask].astype(np.float64)
    s -= s.mean()
    r -= r.mean()
    denom = np.sqrt((s ** 2).sum() * (r ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((s * r).sum() / denom)


def forward_returns(mid: np.ndarray, horizon: int = HORIZON_BARS) -> np.ndarray:
    """Compute forward log-returns at given horizon.  Last `horizon` rows are NaN."""
    n = len(mid)
    fwd = np.empty(n, dtype=np.float32)
    fwd[:] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mid[horizon:] / mid[:-horizon]
        fwd[: n - horizon] = np.where(ratio > 0, np.log(ratio).astype(np.float32), np.nan)
    return fwd


def load_data(n_days: int):
    """Load up to n_days of MBO features.  Returns (X, fwd_ret, feature_names)."""
    # Support both Windows and Linux paths
    win_path = Path(r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\processed\mbo_features_cache")
    lin_path = Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache"
    cache_dir = win_path if win_path.exists() else lin_path

    if not cache_dir.exists():
        raise FileNotFoundError(f"mbo_features_cache not found at {win_path} or {lin_path}")

    files = sorted(cache_dir.glob("*_mbo_features.npz"))
    if not files:
        raise FileNotFoundError(f"No *_mbo_features.npz files found in {cache_dir}")

    files = files[:n_days]
    print(f"  Loading {len(files)} days from {cache_dir}")

    all_X   = []
    all_fwd = []

    for f in files:
        try:
            data = np.load(str(f))
            X = data["mbo_features"]                   # (N, 340) float32
            if X.ndim != 2 or X.shape[1] != 340:
                print(f"  SKIP {f.name}: unexpected shape {X.shape}")
                continue
            mid = X[:, 0].astype(np.float64)
            fwd = forward_returns(mid)
            all_X.append(X)
            all_fwd.append(fwd)
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")

    if not all_X:
        raise RuntimeError("No valid data loaded.")

    X_all   = np.concatenate(all_X,   axis=0)
    fwd_all = np.concatenate(all_fwd, axis=0)
    print(f"  Total rows: {X_all.shape[0]:,}  ({len(all_X)} days)")
    return X_all, fwd_all


def get_feature_names_safe():
    """Import get_feature_names and return the list."""
    try:
        from mbo_features import get_feature_names
        return get_feature_names()
    except ImportError:
        # Fallback: generate generic names
        return [f"feat_{i:03d}" for i in range(340)]


def build_exclusion_mask(feature_names):
    """Return boolean mask: True means INCLUDE, False means EXCLUDE."""
    n = len(feature_names)
    mask = np.ones(n, dtype=bool)
    # Hard-coded leakage indices
    for idx in LEAKAGE_INDICES:
        if idx < n:
            mask[idx] = False
    # Name-based exclusions: rvol_* and vov_*
    for i, name in enumerate(feature_names):
        if name.startswith("rvol_") or name.startswith("vov_"):
            mask[i] = False
    return mask


def make_train_val_split(X, fwd, train_days_frac=0.75, day_size_est=None):
    """
    Split by first train_days_frac of rows (time-ordered).
    Returns indices (train_idx, val_idx).
    """
    n = len(fwd)
    split = int(n * train_days_frac)
    train_idx = np.arange(split)
    val_idx   = np.arange(split, n)
    return train_idx, val_idx


def prepare_arrays(X, fwd, incl_mask, idx):
    """Return (X_sub, y_sub) with finite rows only, downsampled if needed."""
    X_s = X[idx][:, incl_mask]
    y_s = fwd[idx]
    fin = np.isfinite(X_s).all(axis=1) & np.isfinite(y_s)
    X_s, y_s = X_s[fin], y_s[fin]
    if len(X_s) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(X_s), MAX_TRAIN_ROWS, replace=False)
        sel.sort()
        X_s, y_s = X_s[sel], y_s[sel]
    return X_s, y_s


# ─────────────────────────────────────────────────────────────────────────────
# A) LightGBM Split + Gain Importance
# ─────────────────────────────────────────────────────────────────────────────

def analysis_lgbm_importance(model, feature_names_clean):
    """Return dicts {name: score} for split and gain."""
    split_imp = dict(zip(feature_names_clean,
                         model.feature_importances_))  # default is 'split'
    # Re-train just for gain
    import lightgbm as lgb
    booster = model.booster_
    gain_raw  = booster.feature_importance(importance_type="gain")
    split_raw = booster.feature_importance(importance_type="split")

    gain_imp  = dict(zip(feature_names_clean, gain_raw))
    split_imp = dict(zip(feature_names_clean, split_raw))
    return split_imp, gain_imp


def print_top(label, imp_dict, n=TOP_N):
    ranked = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  --- Top {n} by {label} ---")
    for rank, (name, val) in enumerate(ranked[:n], 1):
        print(f"    {rank:3d}. {name:<45s}  {val:.4f}")
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# B) Permutation Importance
# ─────────────────────────────────────────────────────────────────────────────

def analysis_permutation(model, X_val, y_val, feature_names_clean, seed=42):
    """
    For each feature, shuffle it and measure IC drop.
    Returns dict {name: ic_drop}.
    """
    rng = np.random.default_rng(seed)
    y_pred_base = model.predict(X_val)
    ic_base = compute_ic(y_pred_base, y_val)
    print(f"  Baseline validation IC: {ic_base:.5f}")

    n_feats = X_val.shape[1]
    drops   = {}

    for i, name in enumerate(feature_names_clean):
        X_perm = X_val.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        y_pred_perm = model.predict(X_perm)
        ic_perm = compute_ic(y_pred_perm, y_val)
        drops[name] = float(ic_base - ic_perm)

    return drops, ic_base


# ─────────────────────────────────────────────────────────────────────────────
# C) SHAP Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analysis_shap(model, X_val, y_val, feature_names_clean):
    """Mean |SHAP| per feature and top-10 pairwise SHAP correlations."""
    import shap

    n_sample = min(SHAP_SAMPLE, len(X_val))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_val), n_sample, replace=False)
    idx.sort()
    X_s = X_val[idx]

    print(f"  Computing SHAP for {n_sample:,} rows …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)   # (n_sample, n_feats)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp = dict(zip(feature_names_clean, mean_abs_shap.tolist()))

    # Top-10 pairwise correlations of SHAP value columns
    n_top_shap = min(50, len(feature_names_clean))
    top_idx = np.argsort(mean_abs_shap)[::-1][:n_top_shap]
    top_names = [feature_names_clean[i] for i in top_idx]
    S = shap_values[:, top_idx]                # (n_sample, 50)

    # Correlate columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(S.T)                # (50, 50)

    # Extract top-10 off-diagonal pairs
    pairs = []
    for a in range(len(top_names)):
        for b in range(a + 1, len(top_names)):
            pairs.append((abs(corr[a, b]), top_names[a], top_names[b], float(corr[a, b])))
    pairs.sort(reverse=True)
    top_interactions = [{"feat_a": a, "feat_b": b, "shap_corr": c}
                        for _, a, b, c in pairs[:10]]

    return shap_imp, top_interactions


# ─────────────────────────────────────────────────────────────────────────────
# D) Feature Group Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analysis_feature_groups(X_train, y_train, X_val, y_val, incl_mask, n_feats_total=340):
    """
    For each feature group:
      1. Train LightGBM on ONLY that group → IC_group
      2. Train LightGBM with that group EXCLUDED → IC_drop
    Returns list of dicts.
    """
    import lightgbm as lgb

    # Baseline: all included features
    print("  Training baseline (all groups) …")
    base_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_val)
    ic_base = compute_ic(y_pred_base, y_val)
    print(f"  Baseline IC (all groups): {ic_base:.5f}")

    group_results = []

    for g_label, g_start, g_end in FEATURE_GROUPS:
        # Map global col indices → indices within incl_mask-filtered feature set
        group_global_cols = list(range(g_start, min(g_end, n_feats_total)))
        incl_indices = np.where(incl_mask)[0]   # global indices of included features

        # Indices within the filtered X that correspond to this group
        group_local = [j for j, gidx in enumerate(incl_indices) if gidx in set(group_global_cols)]

        if len(group_local) == 0:
            print(f"  SKIP {g_label}: all features excluded by leakage mask")
            group_results.append({"group": g_label, "n_features": 0,
                                   "ic_only_group": None, "ic_drop": None})
            continue

        n_group_feats = len(group_local)
        group_local_arr = np.array(group_local)

        # 1) Train on ONLY this group
        X_tr_only = X_train[:, group_local_arr]
        X_v_only  = X_val[:,   group_local_arr]
        fin_tr = np.isfinite(X_tr_only).all(axis=1)
        fin_v  = np.isfinite(X_v_only).all(axis=1)

        ic_only = None
        if fin_tr.sum() > 1000:
            m_only = lgb.LGBMRegressor(**LGBM_PARAMS)
            m_only.fit(X_tr_only[fin_tr], y_train[fin_tr])
            p = m_only.predict(X_v_only[fin_v])
            ic_only = compute_ic(p, y_val[fin_v])

        # 2) Train with this group EXCLUDED
        all_local = np.arange(X_train.shape[1])
        excl_mask_local = np.ones(len(all_local), dtype=bool)
        excl_mask_local[group_local_arr] = False
        remaining = all_local[excl_mask_local]

        ic_excl = None
        if len(remaining) > 0:
            X_tr_excl = X_train[:, remaining]
            X_v_excl  = X_val[:,   remaining]
            fin_tr2 = np.isfinite(X_tr_excl).all(axis=1)
            fin_v2  = np.isfinite(X_v_excl).all(axis=1)
            if fin_tr2.sum() > 1000:
                m_excl = lgb.LGBMRegressor(**LGBM_PARAMS)
                m_excl.fit(X_tr_excl[fin_tr2], y_train[fin_tr2])
                p2 = m_excl.predict(X_v_excl[fin_v2])
                ic_excl = compute_ic(p2, y_val[fin_v2])

        ic_drop = float(ic_base - ic_excl) if ic_excl is not None else None

        ic_excl_str = f"{ic_excl:+.5f}" if ic_excl is not None else "N/A"
        ic_drop_str = f"{ic_drop:+.5f}" if ic_drop is not None else "N/A"
        print(f"  {g_label:<30s}  n_feats={n_group_feats:3d}  "
              f"ic_only={ic_only:+.5f}  "
              f"ic_when_excl={ic_excl_str}  "
              f"ic_drop={ic_drop_str}")

        group_results.append({
            "group":          g_label,
            "n_features":     n_group_feats,
            "ic_only_group":  round(ic_only,  6) if ic_only  is not None else None,
            "ic_when_excl":   round(ic_excl,  6) if ic_excl  is not None else None,
            "ic_drop":        round(ic_drop,  6) if ic_drop  is not None else None,
            "ic_baseline":    round(ic_base,  6),
        })

    return group_results, ic_base


# ─────────────────────────────────────────────────────────────────────────────
# E) Pairwise Feature IC (raw Pearson IC vs forward return)
# ─────────────────────────────────────────────────────────────────────────────

def analysis_raw_ic(X_all, fwd_all, incl_mask, feature_names_clean, top_by_gain):
    """
    Compute raw IC for top 50 features by LightGBM gain.
    Returns dict {name: raw_ic}.
    """
    # Use all available rows
    fin = np.isfinite(X_all[:, :]).any(axis=1) & np.isfinite(fwd_all)
    X_f   = X_all[fin]
    fwd_f = fwd_all[fin]

    # Subset to included features
    incl_indices = np.where(incl_mask)[0]
    name_to_local = {name: j for j, name in enumerate(feature_names_clean)}

    raw_ic = {}
    top_names = [n for n, _ in top_by_gain[:TOP_RAW_IC]]

    for name in top_names:
        if name not in name_to_local:
            continue
        j = name_to_local[name]
        gidx = incl_indices[j]
        col  = X_f[:, gidx].astype(np.float64)
        ic   = compute_ic(col, fwd_f.astype(np.float64))
        raw_ic[name] = round(ic, 6)

    return raw_ic


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep feature importance analysis")
    parser.add_argument("--n-days", type=int, default=40,
                        help="Number of trading days to load (default: 40)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip SHAP and permutation importance (faster)")
    args = parser.parse_args()

    t0_total = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Deep Feature Importance Analysis")
    print(f"  n_days={args.n_days}  quick={args.quick}  horizon={HORIZON_BARS} bars (10s)")
    print("=" * 70)

    # ── 1. Load feature names ─────────────────────────────────────────────
    print("\n[1/6] Loading feature names …")
    feature_names = get_feature_names_safe()
    assert len(feature_names) == 340, f"Expected 340 feature names, got {len(feature_names)}"

    incl_mask = build_exclusion_mask(feature_names)
    feature_names_clean = [n for n, m in zip(feature_names, incl_mask) if m]
    n_incl = len(feature_names_clean)
    n_excl = 340 - n_incl
    print(f"  Features included: {n_incl}  (excluded by leakage mask: {n_excl})")

    # ── 2. Load data ──────────────────────────────────────────────────────
    print("\n[2/6] Loading MBO feature cache …")
    X_all, fwd_all = load_data(args.n_days)

    # Time-based split: first 75% train, last 25% validation
    n_total = len(X_all)
    split   = int(n_total * 0.75)
    train_idx = np.arange(split)
    val_idx   = np.arange(split, n_total)

    print(f"  Train: {len(train_idx):,} rows  |  Val: {len(val_idx):,} rows")

    X_train, y_train = prepare_arrays(X_all, fwd_all, incl_mask, train_idx)
    X_val,   y_val   = prepare_arrays(X_all, fwd_all, incl_mask, val_idx)

    print(f"  After filtering (finite rows): train={len(X_train):,}  val={len(X_val):,}")

    # ── 3. Train LightGBM ─────────────────────────────────────────────────
    print("\n[3/6] Training LightGBM …")
    import lightgbm as lgb
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=0)])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  (best iteration: {model.best_iteration_})")

    y_pred_val = model.predict(X_val)
    ic_overall = compute_ic(y_pred_val, y_val)
    print(f"  Overall validation IC: {ic_overall:.5f}")

    # ── 4A. Split + Gain Importance ───────────────────────────────────────
    print("\n[4A/6] LightGBM split + gain importance …")
    split_imp, gain_imp = analysis_lgbm_importance(model, feature_names_clean)
    ranked_split = print_top("split",  split_imp, TOP_N)
    ranked_gain  = print_top("gain",   gain_imp,  TOP_N)

    # ── 4B. Permutation Importance ────────────────────────────────────────
    perm_drops   = None
    ic_base_perm = None
    if not args.quick:
        print("\n[4B/6] Permutation importance (shuffling each feature) …")
        t0 = time.time()
        perm_drops, ic_base_perm = analysis_permutation(
            model, X_val, y_val, feature_names_clean)
        ranked_perm = print_top("permutation IC drop", perm_drops, TOP_N)
        print(f"  Permutation done in {time.time()-t0:.1f}s")
    else:
        print("\n[4B/6] Permutation importance SKIPPED (--quick)")

    # ── 4C. SHAP Analysis ─────────────────────────────────────────────────
    shap_imp          = None
    shap_interactions = None
    if not args.quick:
        print("\n[4C/6] SHAP analysis …")
        try:
            t0 = time.time()
            shap_imp, shap_interactions = analysis_shap(
                model, X_val, y_val, feature_names_clean)
            ranked_shap = print_top("mean |SHAP|", shap_imp, TOP_N)
            print(f"\n  Top-10 SHAP feature interactions (by SHAP value correlation):")
            for k, inter in enumerate(shap_interactions, 1):
                print(f"    {k:2d}. {inter['feat_a']:<40s} × {inter['feat_b']:<40s}  "
                      f"corr={inter['shap_corr']:+.4f}")
            print(f"  SHAP done in {time.time()-t0:.1f}s")
        except ImportError:
            print("  WARNING: shap not installed. Skipping SHAP analysis.")
            print("  Install with: pip install shap")
    else:
        print("\n[4C/6] SHAP analysis SKIPPED (--quick)")

    # ── 4D. Feature Group Ablation ────────────────────────────────────────
    print("\n[4D/6] Feature group ablation …")
    t0 = time.time()
    group_results, ic_base_group = analysis_feature_groups(
        X_train, y_train, X_val, y_val, incl_mask)
    print(f"  Group ablation done in {time.time()-t0:.1f}s")

    print(f"\n  --- Group Summary (sorted by ic_only_group) ---")
    sorted_groups = sorted(
        [g for g in group_results if g["ic_only_group"] is not None],
        key=lambda x: abs(x["ic_only_group"] or 0),
        reverse=True,
    )
    print(f"  {'Group':<30s}  {'n_feats':>7}  {'ic_only':>9}  {'ic_drop':>9}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*9}")
    for g in sorted_groups:
        ic_only = g["ic_only_group"]
        ic_drop = g["ic_drop"]
        print(f"  {g['group']:<30s}  {g['n_features']:>7d}  "
              f"{ic_only:>+9.5f}  "
              f"{ic_drop:>+9.5f if ic_drop is not None else 'N/A':>9}")

    # ── 4E. Raw Pearson IC per feature ────────────────────────────────────
    print("\n[4E/6] Raw Pearson IC for top 50 features (by LightGBM gain) …")
    t0 = time.time()
    raw_ic = analysis_raw_ic(X_all, fwd_all, incl_mask, feature_names_clean, ranked_gain)
    print(f"\n  --- Top 50 features: LightGBM gain vs raw IC ---")
    print(f"  {'Rank':>5}  {'Feature':<45s}  {'LGBM Gain':>10}  {'Raw IC':>10}")
    print(f"  {'-'*5}  {'-'*45}  {'-'*10}  {'-'*10}")
    for rank, (name, gain_val) in enumerate(ranked_gain[:TOP_RAW_IC], 1):
        ric = raw_ic.get(name, float("nan"))
        print(f"  {rank:>5d}  {name:<45s}  {gain_val:>10.1f}  {ric:>+10.6f}")
    print(f"  Raw IC done in {time.time()-t0:.1f}s")

    # ── 5. Comprehensive Summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 70)
    print(f"  Days loaded:           {args.n_days}")
    print(f"  Total rows:            {n_total:,}")
    print(f"  Training rows:         {len(X_train):,}")
    print(f"  Validation rows:       {len(X_val):,}")
    print(f"  Features (after excl): {n_incl}  ({n_excl} excluded)")
    print(f"  Validation IC (model): {ic_overall:+.5f}")

    # Overlap between top-30 split and top-30 gain
    top30_split = {n for n, _ in ranked_split[:30]}
    top30_gain  = {n for n, _ in ranked_gain[:30]}
    overlap = top30_split & top30_gain
    print(f"\n  Features in BOTH top-30 split AND top-30 gain ({len(overlap)}):")
    for name in sorted(overlap):
        print(f"    - {name}")

    if shap_imp:
        ranked_shap_list = sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)
        top30_shap = {n for n, _ in ranked_shap_list[:30]}
        triple = top30_gain & top30_shap
        if perm_drops:
            ranked_perm_list = sorted(perm_drops.items(), key=lambda x: x[1], reverse=True)
            top30_perm = {n for n, _ in ranked_perm_list[:30]}
            triple &= top30_perm
            print(f"\n  Features in top-30 of ALL 4 methods (gain + split + SHAP + perm):")
        else:
            print(f"\n  Features in top-30 of gain + SHAP:")
        for name in sorted(triple):
            print(f"    - {name}")

    # Best feature groups by predictive power (ic_only_group)
    print(f"\n  Top 5 feature groups by standalone IC:")
    for g in sorted_groups[:5]:
        print(f"    {g['group']:<32s}  ic_only={g['ic_only_group']:+.5f}  "
              f"ic_drop={g['ic_drop']:+.5f if g['ic_drop'] is not None else 'N/A'!s}")

    # Top 10 raw IC features
    raw_ic_sorted = sorted(raw_ic.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 10 features by raw Pearson IC (among top-{TOP_RAW_IC} by gain):")
    for rank, (name, ic_val) in enumerate(raw_ic_sorted[:10], 1):
        print(f"    {rank:2d}. {name:<45s}  raw_IC={ic_val:+.6f}")

    # ── 6. Save Results ───────────────────────────────────────────────────
    results_dir = LVL3_ROOT / "alpha_discovery" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"feature_importance_{timestamp}.json"

    def ranked_to_list(ranked, n=340):
        return [{"rank": r, "name": name, "score": float(score)}
                for r, (name, score) in enumerate(ranked[:n], 1)]

    output = {
        "timestamp":          timestamp,
        "n_days":             args.n_days,
        "quick_mode":         args.quick,
        "horizon_bars":       HORIZON_BARS,
        "total_rows":         int(n_total),
        "train_rows":         int(len(X_train)),
        "val_rows":           int(len(X_val)),
        "n_features_included": int(n_incl),
        "n_features_excluded": int(n_excl),
        "validation_ic":      round(ic_overall, 6),
        "lgbm_params":        LGBM_PARAMS,
        "A_lgbm_split":       ranked_to_list(ranked_split),
        "A_lgbm_gain":        ranked_to_list(ranked_gain),
        "B_permutation":      (ranked_to_list(
                                   sorted(perm_drops.items(), key=lambda x: x[1], reverse=True)
                               ) if perm_drops else None),
        "C_shap_mean_abs":    (ranked_to_list(
                                   sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)
                               ) if shap_imp else None),
        "C_shap_interactions": shap_interactions,
        "D_feature_groups":   group_results,
        "E_raw_ic_top50":     [{"name": n, "raw_ic": v}
                               for n, v in sorted(raw_ic.items(),
                                                   key=lambda x: abs(x[1]), reverse=True)],
        "excluded_features":  [n for n, m in zip(feature_names, incl_mask) if not m],
        "top30_overlap_gain_split": sorted(list(top30_gain & top30_split)),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    total_elapsed = time.time() - t0_total
    print(f"\n[6/6] Results saved to: {out_path}")
    print(f"  Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
