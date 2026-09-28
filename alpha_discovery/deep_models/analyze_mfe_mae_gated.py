#!/usr/bin/env python3
"""
analyze_mfe_mae_gated.py — MFE/MAE Analysis on XGB-Gated Trades
=================================================================
Loads CNN-Mamba v2 OOT predictions (10 folds) + XGBoost gate v1,
filters to Top5%/Top1% gated trades, and analyzes:
  - MFE (Max Favorable Excursion) at 1s, 5s, 10s horizons
  - MAE (Max Adverse Excursion) at 1s, 5s, 10s horizons
  - Optimal TP/SL levels derived from data
  - Hold time recommendations
  - Win rate at various TP/SL configs

Per HC #52: Commission = $4.70 RT = 0.376 ticks, TICK_VALUE = $12.50
Per HC #49: Exit rules must be data-driven from MFE/MAE analysis.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from constants import COMMISSION_TICKS, TICK_VALUE
except ImportError:
    COMMISSION_TICKS = 0.376
    TICK_VALUE = 12.50

# ─── Config ───────────────────────────────────────────────────────────────────
CNN_OOT_DIR = "output/cnn_mamba_v2_smart_v3_mar"
VOL_PRED_DIR = "output/vol_lgbm_v3"
GATE_DIR = "output/xgb_gate_v1"
OUTPUT_DIR = "output/mfe_mae_gated"

HORIZONS = ['1s', '5s', '10s']
HORIZON_IDX = {'1s': 0, '5s': 1, '10s': 2}
HORIZON_SECONDS = {'1s': 1, '5s': 5, '10s': 10}

# Gate tiers to analyze
TIERS = {
    'Top5%': 0.95,
    'Top1%': 0.99,
}


# ─── Reuse feature builder from gate v1 ──────────────────────────────────────

def build_features_from_predictions(preds_3h, vol_pred=None):
    """Build features from model predictions (no embeddings needed).
    Copied from train_xgb_gate_v1.py for consistency.
    """
    features = {}
    for i, h in enumerate(HORIZONS):
        features[f'pred_{h}'] = preds_3h[:, i]
        features[f'abs_pred_{h}'] = np.abs(preds_3h[:, i])
        features[f'sign_{h}'] = np.sign(preds_3h[:, i])

    signs = np.sign(preds_3h)
    features['horizon_agree_all'] = (np.abs(signs.sum(axis=1)) == 3).astype(float)
    features['horizon_agree_12'] = (signs[:, 0] == signs[:, 1]).astype(float)
    features['horizon_agree_13'] = (signs[:, 0] == signs[:, 2]).astype(float)
    features['horizon_agree_23'] = (signs[:, 1] == signs[:, 2]).astype(float)

    features['max_abs_pred'] = np.max(np.abs(preds_3h), axis=1)
    features['mean_abs_pred'] = np.mean(np.abs(preds_3h), axis=1)
    features['pred_range'] = np.max(preds_3h, axis=1) - np.min(preds_3h, axis=1)
    features['pred_std'] = np.std(preds_3h, axis=1)

    eps = 1e-8
    features['ratio_1s_10s'] = np.abs(preds_3h[:, 0]) / (np.abs(preds_3h[:, 2]) + eps)
    features['ratio_5s_10s'] = np.abs(preds_3h[:, 1]) / (np.abs(preds_3h[:, 2]) + eps)

    if vol_pred is not None:
        features['vol_pred'] = vol_pred
        features['abs_vol'] = np.abs(vol_pred)
        for i, h in enumerate(HORIZONS):
            features[f'pred_{h}_x_vol'] = preds_3h[:, i] * vol_pred
            features[f'abs_pred_{h}_x_vol'] = np.abs(preds_3h[:, i]) * np.abs(vol_pred)

    feat_names = sorted(features.keys())
    X = np.column_stack([features[k] for k in feat_names])
    return X, feat_names


def load_vol_for_date(date_str, n_samples):
    """Try to load vol predictions for a date, matching n_samples."""
    vol_file = os.path.join(VOL_PRED_DIR, f"vol_v3_{date_str}_predictions.npz")
    if not os.path.exists(vol_file):
        return None
    vd = np.load(vol_file, allow_pickle=True)
    for k in ['predictions', 'vol_pred', 'y_pred']:
        if k in vd:
            vp = vd[k]
            if vp.ndim > 1:
                vp = vp.flatten()
            if len(vp) == n_samples:
                return vp
            else:
                from scipy.interpolate import interp1d
                x_vol = np.linspace(0, 1, len(vp))
                x_cnn = np.linspace(0, 1, n_samples)
                f_interp = interp1d(x_vol, vp, kind='nearest', fill_value='extrapolate')
                return f_interp(x_cnn)
    return None


def load_oot_fold(fold_idx):
    """Load one OOT fold: predictions, labels, features for gate."""
    cnn_file = os.path.join(CNN_OOT_DIR, f"fold_{fold_idx:02d}_oot_predictions.npz")
    if not os.path.exists(cnn_file):
        return None

    cnn_data = np.load(cnn_file, allow_pickle=True)
    preds = cnn_data['predictions']   # (N, 3)
    labels = cnn_data['labels']       # (N, 3) — price moves in ticks at 1s, 5s, 10s

    oot_files = cnn_data['oot_files']
    if hasattr(oot_files, 'tolist'):
        oot_files = oot_files.tolist()
    date_str = str(oot_files[0]).split('/')[-1][:8] if isinstance(oot_files, list) else str(oot_files).split('/')[-1][:8]

    vol_pred = load_vol_for_date(date_str, len(preds))
    X, feat_names = build_features_from_predictions(preds, vol_pred)

    return {
        'X': X, 'preds': preds, 'labels': labels,
        'date': date_str, 'feat_names': feat_names, 'n': len(preds)
    }


def compute_mfe_mae_for_trades(preds_1s, labels_3h, direction_signs):
    """
    Compute MFE and MAE for each trade across 1s, 5s, 10s horizons.

    For a LONG trade (direction_sign > 0):
      - Favorable move = positive price change
      - Adverse move = negative price change
    For a SHORT trade (direction_sign < 0):
      - Favorable move = negative price change (profit when price drops)
      - Adverse move = positive price change

    labels_3h: (N, 3) — actual price moves in ticks at 1s, 5s, 10s
    direction_signs: (N,) — +1 for long, -1 for short (from 1s prediction sign)

    Returns dict with MFE/MAE arrays at each horizon.
    """
    N = len(labels_3h)

    # Aligned moves: positive = favorable, negative = adverse
    aligned_moves = labels_3h * direction_signs[:, None]  # (N, 3)

    # MFE at each horizon = max favorable move seen up to that point
    # Since we only have snapshots at 1s, 5s, 10s, MFE is the max of aligned moves up to that horizon
    mfe_1s = np.maximum(aligned_moves[:, 0], 0)  # best favorable by 1s
    mfe_5s = np.maximum(np.max(aligned_moves[:, :2], axis=1), 0)  # best favorable by 5s
    mfe_10s = np.maximum(np.max(aligned_moves[:, :3], axis=1), 0)  # best favorable by 10s

    # MAE at each horizon = max adverse move seen up to that point (positive = bad)
    mae_1s = np.maximum(-aligned_moves[:, 0], 0)  # worst adverse by 1s
    mae_5s = np.maximum(np.max(-aligned_moves[:, :2], axis=1), 0)  # worst adverse by 5s
    mae_10s = np.maximum(np.max(-aligned_moves[:, :3], axis=1), 0)  # worst adverse by 10s

    # Terminal P&L at each horizon (in ticks, before commission)
    pnl_1s = aligned_moves[:, 0]
    pnl_5s = aligned_moves[:, 1]
    pnl_10s = aligned_moves[:, 2]

    return {
        'mfe_1s': mfe_1s, 'mfe_5s': mfe_5s, 'mfe_10s': mfe_10s,
        'mae_1s': mae_1s, 'mae_5s': mae_5s, 'mae_10s': mae_10s,
        'pnl_1s': pnl_1s, 'pnl_5s': pnl_5s, 'pnl_10s': pnl_10s,
        'aligned_moves': aligned_moves,
    }


def distribution_stats(arr, name=""):
    """Return dict of distribution stats for an array."""
    if len(arr) == 0:
        return {}
    return {
        f'{name}_mean': float(np.mean(arr)),
        f'{name}_median': float(np.median(arr)),
        f'{name}_std': float(np.std(arr)),
        f'{name}_p25': float(np.percentile(arr, 25)),
        f'{name}_p75': float(np.percentile(arr, 75)),
        f'{name}_p90': float(np.percentile(arr, 90)),
        f'{name}_p95': float(np.percentile(arr, 95)),
        f'{name}_min': float(np.min(arr)),
        f'{name}_max': float(np.max(arr)),
    }


def simulate_tp_sl(aligned_moves_3h, tp_ticks, sl_ticks, commission=COMMISSION_TICKS):
    """
    Simulate TP/SL exit strategy across 1s, 5s, 10s snapshots (vectorized).

    For each trade, check at each horizon:
      - If aligned move >= TP → exit with profit (TP - commission)
      - If aligned move <= -SL → exit with loss (-SL - commission)
      - If neither by 10s → exit at terminal P&L - commission

    Returns: (net_pnl_per_trade, win_mask, hold_times, exit_types)
    """
    N = len(aligned_moves_3h)
    horizon_secs = np.array([1, 5, 10])

    net_pnl = np.full(N, np.nan)
    hold_times = np.full(N, 10.0)
    exit_type = np.full(N, 'timeout', dtype='U10')

    remaining = np.ones(N, dtype=bool)  # trades not yet exited

    for h_idx in range(3):
        if not remaining.any():
            break

        moves = aligned_moves_3h[remaining, h_idx]

        # TP hit
        tp_hit = moves >= tp_ticks
        if tp_hit.any():
            # Map back to full indices
            full_idx = np.where(remaining)[0][tp_hit]
            net_pnl[full_idx] = tp_ticks - commission
            hold_times[full_idx] = horizon_secs[h_idx]
            exit_type[full_idx] = 'TP'
            remaining[full_idx] = False

        # SL hit (only on still-remaining)
        still_remaining_moves = aligned_moves_3h[remaining, h_idx]
        sl_hit = still_remaining_moves <= -sl_ticks
        if sl_hit.any():
            full_idx = np.where(remaining)[0][sl_hit]
            net_pnl[full_idx] = -sl_ticks - commission
            hold_times[full_idx] = horizon_secs[h_idx]
            exit_type[full_idx] = 'SL'
            remaining[full_idx] = False

    # Timeout: exit at 10s terminal value
    if remaining.any():
        net_pnl[remaining] = aligned_moves_3h[remaining, 2] - commission

    wins = net_pnl > 0
    return net_pnl, wins, hold_times, exit_type


def main():
    os.chdir("/home/jupiter/Lvl3Quant")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, "run.log")
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log("=" * 80)
    log("MFE/MAE Analysis on XGB-Gated Trades")
    log(f"Commission: ${COMMISSION_TICKS * TICK_VALUE:.2f} RT = {COMMISSION_TICKS:.3f} ticks (HC #52)")
    log(f"Tick value: ${TICK_VALUE:.2f}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    # ─── Step 1: Load XGB gate model (1s horizon focus) ──────────────────────
    try:
        import xgboost as xgb
    except ImportError:
        log("ERROR: xgboost not installed")
        return

    gate_model_path = os.path.join(GATE_DIR, "xgb_gate_1s.json")
    if not os.path.exists(gate_model_path):
        log(f"ERROR: Gate model not found at {gate_model_path}")
        return

    gate_model = xgb.Booster()
    gate_model.load_model(gate_model_path)
    log(f"Loaded XGB gate: {gate_model_path}")
    log(f"Gate features: {gate_model.num_features()}")

    # ─── Step 2: Load all 10 OOT folds ──────────────────────────────────────
    log("\n--- Loading 10 OOT folds ---")
    folds = []
    for i in range(10):
        fd = load_oot_fold(i)
        if fd is not None:
            folds.append(fd)
            log(f"  Fold {i}: date={fd['date']}, n={fd['n']}")

    if not folds:
        log("ERROR: No folds loaded")
        return

    total_samples = sum(f['n'] for f in folds)
    log(f"Total samples across {len(folds)} folds: {total_samples:,}")

    # ─── Step 3: Score all predictions with gate, collect gated trades ───────
    log("\n--- Scoring with XGB gate (1s horizon focus) ---")

    # Collect MFE/MAE data per tier
    tier_data = {tier: defaultdict(list) for tier in TIERS}
    tier_data['All'] = defaultdict(list)  # baseline: all trades

    for fold in folds:
        feat_names = fold['feat_names']
        dmat = xgb.DMatrix(fold['X'], feature_names=feat_names)
        gate_scores = gate_model.predict(dmat)

        # Direction from 1s prediction
        direction_signs = np.sign(fold['preds'][:, 0])

        # Skip zero-prediction trades (no direction)
        valid_mask = direction_signs != 0
        if valid_mask.sum() == 0:
            continue

        # Compute MFE/MAE for ALL valid trades in this fold
        excursions = compute_mfe_mae_for_trades(
            fold['preds'][valid_mask, 0],
            fold['labels'][valid_mask],
            direction_signs[valid_mask]
        )

        # Store all trades for baseline
        for key in excursions:
            if isinstance(excursions[key], np.ndarray):
                tier_data['All'][key].append(excursions[key])

        # Filter by gate tiers
        valid_scores = gate_scores[valid_mask]
        for tier_name, tier_pct in TIERS.items():
            threshold = np.percentile(valid_scores, tier_pct * 100)
            tier_mask = valid_scores >= threshold
            n_gated = tier_mask.sum()

            if n_gated < 2:
                continue

            gated_excursions = compute_mfe_mae_for_trades(
                fold['preds'][valid_mask][tier_mask, 0],
                fold['labels'][valid_mask][tier_mask],
                direction_signs[valid_mask][tier_mask]
            )

            for key in gated_excursions:
                if isinstance(gated_excursions[key], np.ndarray):
                    tier_data[tier_name][key].append(gated_excursions[key])

    # Concatenate arrays across folds
    for tier_name in tier_data:
        for key in tier_data[tier_name]:
            tier_data[tier_name][key] = np.concatenate(tier_data[tier_name][key])

    # ─── Step 4: Report MFE/MAE distributions ───────────────────────────────
    results = {}

    for tier_name in ['All', 'Top5%', 'Top1%']:
        td = tier_data[tier_name]
        if 'mfe_1s' not in td or len(td['mfe_1s']) == 0:
            log(f"\n  {tier_name}: No data")
            continue

        n_trades = len(td['mfe_1s'])
        log(f"\n{'='*80}")
        log(f"TIER: {tier_name} — {n_trades:,} trades")
        log(f"{'='*80}")

        tier_results = {'n_trades': n_trades}

        # MFE distributions
        log(f"\n  --- MFE (Max Favorable Excursion, ticks) ---")
        for h in ['1s', '5s', '10s']:
            key = f'mfe_{h}'
            arr = td[key]
            stats = distribution_stats(arr, f'mfe_{h}')
            tier_results.update(stats)
            log(f"  MFE@{h}: mean={np.mean(arr):.2f}  med={np.median(arr):.2f}  "
                f"p75={np.percentile(arr,75):.2f}  p90={np.percentile(arr,90):.2f}  "
                f"p95={np.percentile(arr,95):.2f}")

        # MAE distributions
        log(f"\n  --- MAE (Max Adverse Excursion, ticks) ---")
        for h in ['1s', '5s', '10s']:
            key = f'mae_{h}'
            arr = td[key]
            stats = distribution_stats(arr, f'mae_{h}')
            tier_results.update(stats)
            log(f"  MAE@{h}: mean={np.mean(arr):.2f}  med={np.median(arr):.2f}  "
                f"p75={np.percentile(arr,75):.2f}  p90={np.percentile(arr,90):.2f}  "
                f"p95={np.percentile(arr,95):.2f}")

        # Terminal P&L distributions (before commission)
        log(f"\n  --- Terminal P&L (ticks, before commission) ---")
        for h in ['1s', '5s', '10s']:
            key = f'pnl_{h}'
            arr = td[key]
            stats = distribution_stats(arr, f'pnl_{h}')
            tier_results.update(stats)
            win_rate = (arr > 0).mean()
            profitable_after_cost = (arr - COMMISSION_TICKS > 0).mean()
            log(f"  P&L@{h}: mean={np.mean(arr):.3f}  med={np.median(arr):.3f}  "
                f"WR_raw={win_rate:.1%}  WR_net={profitable_after_cost:.1%}  "
                f"avg_net=${(np.mean(arr)-COMMISSION_TICKS)*TICK_VALUE:.2f}")

        # MFE peak analysis: where does favorable move peak?
        log(f"\n  --- Hold Time Recommendation (MFE peak analysis) ---")
        mfe_at = [np.mean(td['mfe_1s']), np.mean(td['mfe_5s']), np.mean(td['mfe_10s'])]
        peak_h = ['1s', '5s', '10s'][np.argmax(mfe_at)]
        log(f"  Mean MFE:  1s={mfe_at[0]:.3f}  5s={mfe_at[1]:.3f}  10s={mfe_at[2]:.3f}")
        log(f"  MFE peaks at: {peak_h}")

        # Check: does MAE grow faster than MFE after peak?
        mae_at = [np.mean(td['mae_1s']), np.mean(td['mae_5s']), np.mean(td['mae_10s'])]
        log(f"  Mean MAE:  1s={mae_at[0]:.3f}  5s={mae_at[1]:.3f}  10s={mae_at[2]:.3f}")

        # Reward/risk ratio at each horizon
        for h_idx, h in enumerate(['1s', '5s', '10s']):
            rr = mfe_at[h_idx] / mae_at[h_idx] if mae_at[h_idx] > 0 else float('inf')
            log(f"  R:R@{h} = {rr:.2f}")

        tier_results['mfe_peak_horizon'] = peak_h
        tier_results['rr_1s'] = mfe_at[0] / mae_at[0] if mae_at[0] > 0 else float('inf')
        tier_results['rr_5s'] = mfe_at[1] / mae_at[1] if mae_at[1] > 0 else float('inf')
        tier_results['rr_10s'] = mfe_at[2] / mae_at[2] if mae_at[2] > 0 else float('inf')

        # ─── Optimal TP/SL from data ─────────────────────────────────────────
        log(f"\n  --- Data-Driven TP/SL Recommendations ---")

        # TP candidates: p50, p75 of MFE at peak horizon
        mfe_peak = td[f'mfe_{peak_h}']
        tp_p50 = np.percentile(mfe_peak, 50)
        tp_p75 = np.percentile(mfe_peak, 75)
        tp_mean = np.mean(mfe_peak)

        # SL candidates: p50, p75 of MAE at peak horizon
        mae_peak = td[f'mae_{peak_h}']
        sl_p50 = np.percentile(mae_peak, 50)
        sl_p75 = np.percentile(mae_peak, 75)
        sl_p90 = np.percentile(mae_peak, 90)

        log(f"  MFE@{peak_h} distribution: p50={tp_p50:.2f}  p75={tp_p75:.2f}  mean={tp_mean:.2f}")
        log(f"  MAE@{peak_h} distribution: p50={sl_p50:.2f}  p75={sl_p75:.2f}  p90={sl_p90:.2f}")
        log(f"  Recommended TP = p50(MFE) = {tp_p50:.2f} ticks = ${tp_p50*TICK_VALUE:.2f}")
        log(f"  Recommended SL = p75(MAE) = {sl_p75:.2f} ticks = ${sl_p75*TICK_VALUE:.2f}")

        tier_results['recommended_tp_ticks'] = float(tp_p50)
        tier_results['recommended_sl_ticks'] = float(sl_p75)

        # ─── TP/SL Grid Simulation ───────────────────────────────────────────
        log(f"\n  --- TP/SL Grid Simulation (net of ${COMMISSION_TICKS*TICK_VALUE:.2f} commission) ---")
        log(f"  {'TP':>6} {'SL':>6} | {'WR':>6} {'AvgPnL':>8} {'PF':>6} {'Sortino':>8} | {'TP%':>5} {'SL%':>5} {'TO%':>5}")
        log(f"  {'-'*6} {'-'*6}-+-{'-'*6}-{'-'*8}-{'-'*6}-{'-'*8}-+-{'-'*5}-{'-'*5}-{'-'*5}")

        aligned = td['aligned_moves']  # (N, 3)
        best_sharpe = -999
        best_config = None

        tp_candidates = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        sl_candidates = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        grid_results = []
        for tp in tp_candidates:
            for sl in sl_candidates:
                net_pnl, wins, hold_times, exit_types = simulate_tp_sl(aligned, tp, sl)
                wr = wins.mean()
                avg_pnl = np.mean(net_pnl) * TICK_VALUE
                total_pnl = np.sum(net_pnl) * TICK_VALUE

                # Profit factor
                gross_profit = np.sum(net_pnl[net_pnl > 0])
                gross_loss = np.abs(np.sum(net_pnl[net_pnl < 0]))
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

                # Sortino
                if np.std(net_pnl) > 0:
                    neg = net_pnl[net_pnl < 0]
                    ds = neg.std() if len(neg) > 1 else 1.0
                    sortino = (np.mean(net_pnl) / ds) * np.sqrt(252) if ds > 0 else 0
                else:
                    sortino = 0

                # Exit type breakdown
                tp_pct = (exit_types == 'TP').mean()
                sl_pct = (exit_types == 'SL').mean()
                to_pct = (exit_types == 'timeout').mean()

                grid_results.append({
                    'tp': tp, 'sl': sl, 'wr': wr, 'avg_pnl': avg_pnl,
                    'pf': pf, 'sortino': sortino, 'total_pnl': total_pnl,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct, 'timeout_pct': to_pct,
                })

                log(f"  {tp:6.1f} {sl:6.1f} | {wr:5.1%} {avg_pnl:>7.2f}$ {pf:5.2f} {sortino:>7.1f}  | {tp_pct:4.0%} {sl_pct:4.0%} {to_pct:4.0%}")

                if sortino > best_sharpe and avg_pnl > 0:
                    best_sharpe = sortino
                    best_config = (tp, sl)

        if best_config:
            log(f"\n  BEST CONFIG: TP={best_config[0]:.1f} SL={best_config[1]:.1f} "
                f"(Sortino={best_sharpe:.1f})")
            tier_results['best_tp'] = best_config[0]
            tier_results['best_sl'] = best_config[1]
            tier_results['best_sortino'] = best_sharpe

        tier_results['grid_results'] = grid_results
        results[tier_name] = tier_results

    # ─── Step 5: Summary Recommendations ─────────────────────────────────────
    log(f"\n{'='*80}")
    log("SUMMARY RECOMMENDATIONS (HC #49: Data-driven exit rules)")
    log(f"{'='*80}")

    for tier_name in ['Top5%', 'Top1%']:
        if tier_name not in results:
            continue
        tr = results[tier_name]
        log(f"\n  {tier_name} ({tr['n_trades']:,} trades):")
        log(f"    MFE peaks at: {tr.get('mfe_peak_horizon', '?')}")
        log(f"    Recommended TP: {tr.get('recommended_tp_ticks', 0):.2f} ticks "
            f"(${tr.get('recommended_tp_ticks', 0)*TICK_VALUE:.2f})")
        log(f"    Recommended SL: {tr.get('recommended_sl_ticks', 0):.2f} ticks "
            f"(${tr.get('recommended_sl_ticks', 0)*TICK_VALUE:.2f})")
        if 'best_tp' in tr:
            log(f"    Grid-optimal TP/SL: {tr['best_tp']:.1f}/{tr['best_sl']:.1f} "
                f"(Sortino={tr['best_sortino']:.1f})")

    # ─── Step 6: Save results ────────────────────────────────────────────────
    # Clean results for JSON serialization
    save_results = {}
    for tier_name, tr in results.items():
        clean = {}
        for k, v in tr.items():
            if k == 'grid_results':
                clean[k] = v  # list of dicts, already clean
            elif isinstance(v, (float, int, str)):
                clean[k] = v
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
        save_results[tier_name] = clean

    results_path = os.path.join(OUTPUT_DIR, "mfe_mae_results.json")
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    log(f"\nResults saved to {results_path}")

    # ─── Step 7: Log to MLflow ───────────────────────────────────────────────
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("MFE_MAE_Gated")

        with mlflow.start_run(run_name=f"mfe_mae_gated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("gate_model", "xgb_gate_1s")
            mlflow.log_param("n_folds", len(folds))
            mlflow.log_param("total_samples", total_samples)
            mlflow.log_param("commission_ticks", COMMISSION_TICKS)

            for tier_name in ['Top5%', 'Top1%']:
                if tier_name not in results:
                    continue
                tr = results[tier_name]
                prefix = tier_name.replace('%', 'pct').replace(' ', '_')
                mlflow.log_metric(f"{prefix}_n_trades", tr['n_trades'])

                for h in ['1s', '5s', '10s']:
                    for stat in ['mean', 'median', 'p75', 'p90', 'p95']:
                        for metric in ['mfe', 'mae', 'pnl']:
                            key = f'{metric}_{h}_{stat}'
                            if key in tr:
                                mlflow.log_metric(f"{prefix}_{key}", tr[key])

                for key in ['recommended_tp_ticks', 'recommended_sl_ticks',
                            'best_tp', 'best_sl', 'best_sortino',
                            'rr_1s', 'rr_5s', 'rr_10s']:
                    if key in tr and not (isinstance(tr[key], float) and np.isinf(tr[key])):
                        mlflow.log_metric(f"{prefix}_{key}", tr[key])

            mlflow.log_artifact(log_path)
            mlflow.log_artifact(results_path)

        log("MLflow run logged successfully")
    except Exception as e:
        log(f"WARNING: MLflow logging failed: {e}")

    log(f"\n{'='*80}")
    log(f"DONE. Output in {OUTPUT_DIR}/")
    log(f"{'='*80}")
    log_file.close()


if __name__ == "__main__":
    main()
