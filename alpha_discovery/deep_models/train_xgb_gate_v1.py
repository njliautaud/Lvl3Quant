#!/usr/bin/env python3
"""
train_xgb_gate_v1.py — XGBoost Confidence Gate using Model Predictions
=======================================================================
Uses prediction outputs (no embeddings needed) from decay v4 analysis to:
  - Train on 10 original OOT dates (Feb 23 - Mar 5)
  - Validate on 39 NEW dates (Mar 16 - Apr 29) from decay analysis

Features:
  - CNN-Mamba v2 predictions (3 horizons: 1s, 5s, 10s)
  - |pred| magnitude per horizon (confidence proxy)
  - Horizon agreement (do 1s/5s/10s all point same direction?)
  - LGBM Vol predictions (volatility)
  - Derived: pred_sign × vol (interaction term)

Per HC #45: tree methods preferred when RL fails.
Per HC #52: Commission = $4.70 RT = 0.376 ticks.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from constants import COMMISSION_TICKS, TICK_VALUE
except ImportError:
    COMMISSION_TICKS = 0.376
    TICK_VALUE = 12.50

# ─── Config ───────────────────────────────────────────────────────────────────
CNN_OOT_DIR = "output/cnn_mamba_v2_smart_v3_mar"  # Original 10 OOT folds
DECAY_CNN_DIR = "output/decay_v4_comprehensive/CNN-Mamba_v2"
DECAY_VOL_DIR = "output/decay_v4_comprehensive/LGBM_Vol"
VOL_PRED_DIR = "output/vol_lgbm_v3"
OUTPUT_DIR = "output/xgb_gate_v1"

HORIZONS = ['1s', '5s', '10s']
HORIZON_IDX = {'1s': 0, '5s': 1, '10s': 2}

# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_features_from_predictions(preds_3h, vol_pred=None):
    """Build features from model predictions (no embeddings needed).

    preds_3h: (N, 3) — predictions for 1s, 5s, 10s
    vol_pred: (N,) or None — volatility prediction
    """
    features = {}

    # Raw predictions
    for i, h in enumerate(HORIZONS):
        features[f'pred_{h}'] = preds_3h[:, i]
        features[f'abs_pred_{h}'] = np.abs(preds_3h[:, i])
        features[f'sign_{h}'] = np.sign(preds_3h[:, i])

    # Horizon agreement
    signs = np.sign(preds_3h)  # (N, 3)
    features['horizon_agree_all'] = (np.abs(signs.sum(axis=1)) == 3).astype(float)
    features['horizon_agree_12'] = (signs[:, 0] == signs[:, 1]).astype(float)
    features['horizon_agree_13'] = (signs[:, 0] == signs[:, 2]).astype(float)
    features['horizon_agree_23'] = (signs[:, 1] == signs[:, 2]).astype(float)

    # Magnitude features
    features['max_abs_pred'] = np.max(np.abs(preds_3h), axis=1)
    features['mean_abs_pred'] = np.mean(np.abs(preds_3h), axis=1)
    features['pred_range'] = np.max(preds_3h, axis=1) - np.min(preds_3h, axis=1)
    features['pred_std'] = np.std(preds_3h, axis=1)

    # Ratio features (with epsilon to avoid div by zero)
    eps = 1e-8
    features['ratio_1s_10s'] = np.abs(preds_3h[:, 0]) / (np.abs(preds_3h[:, 2]) + eps)
    features['ratio_5s_10s'] = np.abs(preds_3h[:, 1]) / (np.abs(preds_3h[:, 2]) + eps)

    # Vol features
    if vol_pred is not None:
        features['vol_pred'] = vol_pred
        features['abs_vol'] = np.abs(vol_pred)
        for i, h in enumerate(HORIZONS):
            features[f'pred_{h}_x_vol'] = preds_3h[:, i] * vol_pred
            features[f'abs_pred_{h}_x_vol'] = np.abs(preds_3h[:, i]) * np.abs(vol_pred)

    # Stack into matrix
    feat_names = sorted(features.keys())
    X = np.column_stack([features[k] for k in feat_names])
    return X, feat_names


def load_oot_fold_data(fold_idx):
    """Load original OOT fold data (has labels)."""
    cnn_file = os.path.join(CNN_OOT_DIR, f"fold_{fold_idx:02d}_oot_predictions.npz")
    if not os.path.exists(cnn_file):
        return None

    cnn_data = np.load(cnn_file, allow_pickle=True)
    preds = cnn_data['predictions']  # (N, 3)
    labels = cnn_data['labels']      # (N, 3)

    # Get date
    oot_files = cnn_data['oot_files']
    if hasattr(oot_files, 'tolist'):
        oot_files = oot_files.tolist()
    date_str = str(oot_files[0]).split('/')[-1][:8] if isinstance(oot_files, list) else str(oot_files).split('/')[-1][:8]

    # Load vol
    vol_pred = load_vol_for_date(date_str, len(preds))

    X, feat_names = build_features_from_predictions(preds, vol_pred)
    return {
        'X': X, 'preds': preds, 'labels': labels,
        'date': date_str, 'feat_names': feat_names, 'n': len(preds)
    }


def load_decay_date_data(date_str):
    """Load decay v4 data for a new date (has labels from decay analysis)."""
    cnn_file = os.path.join(DECAY_CNN_DIR, date_str, "predictions.npz")
    if not os.path.exists(cnn_file):
        return None

    cnn_data = np.load(cnn_file, allow_pickle=True)
    preds = cnn_data['preds']  # (N, 3)

    labels = {}
    for h in HORIZONS:
        key = f'labels_{h}'
        if key in cnn_data:
            labels[h] = cnn_data[key]

    if not labels:
        return None

    # Stack labels into (N, 3) matching preds
    labels_arr = np.column_stack([labels.get(h, np.zeros(len(preds))) for h in HORIZONS])

    # Load vol
    vol_file = os.path.join(DECAY_VOL_DIR, date_str, "predictions.npz")
    vol_pred = None
    if os.path.exists(vol_file):
        vd = np.load(vol_file, allow_pickle=True)
        for k in ['preds', 'predictions', 'vol_pred', 'y_pred']:
            if k in vd:
                vp = vd[k]
                if vp.ndim > 1:
                    vp = vp.flatten()
                if len(vp) == len(preds):
                    vol_pred = vp
                else:
                    # Nearest-neighbor interpolation
                    from scipy.interpolate import interp1d
                    x_vol = np.linspace(0, 1, len(vp))
                    x_cnn = np.linspace(0, 1, len(preds))
                    f_interp = interp1d(x_vol, vp, kind='nearest', fill_value='extrapolate')
                    vol_pred = f_interp(x_cnn)
                break

    X, feat_names = build_features_from_predictions(preds, vol_pred)
    return {
        'X': X, 'preds': preds, 'labels': labels_arr,
        'date': date_str, 'feat_names': feat_names, 'n': len(preds)
    }


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


def evaluate_gate(preds_3h, labels_3h, gate_scores, date_str, horizon_idx=0):
    """Evaluate gate quality for a specific horizon."""
    horizon = HORIZONS[horizon_idx]
    p = preds_3h[:, horizon_idx]
    y = labels_3h[:, horizon_idx]

    # Directional correctness
    correct = (np.sign(p) == np.sign(y)).astype(float)
    baseline_da = correct.mean()

    results = {'date': date_str, 'horizon': horizon, 'n': len(p), 'baseline_da': baseline_da}

    # Evaluate at different gate thresholds
    for thr_name, thr_pct in [('Top10%', 0.90), ('Top5%', 0.95), ('Top1%', 0.99)]:
        threshold = np.percentile(gate_scores, thr_pct * 100)
        mask = gate_scores >= threshold
        n_gated = mask.sum()
        if n_gated < 5:
            continue

        da_gated = correct[mask].mean()
        lift = da_gated - baseline_da

        # PnL calculation (ticks)
        abs_move = np.abs(y[mask])
        wins = correct[mask].astype(bool)
        pnl_per_trade = np.where(wins, abs_move - COMMISSION_TICKS, -(abs_move + COMMISSION_TICKS))
        total_pnl = pnl_per_trade.sum()
        avg_pnl = pnl_per_trade.mean() * TICK_VALUE

        # Sortino
        if pnl_per_trade.std() > 0:
            neg_returns = pnl_per_trade[pnl_per_trade < 0]
            downside_std = neg_returns.std() if len(neg_returns) > 1 else 1.0
            sortino = (pnl_per_trade.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0

        # Profit factor
        gross_profit = pnl_per_trade[pnl_per_trade > 0].sum()
        gross_loss = np.abs(pnl_per_trade[pnl_per_trade < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        wr = wins.mean()

        results[f'{thr_name}_n'] = int(n_gated)
        results[f'{thr_name}_coverage'] = n_gated / len(p)
        results[f'{thr_name}_da'] = da_gated
        results[f'{thr_name}_lift'] = lift
        results[f'{thr_name}_sortino'] = sortino
        results[f'{thr_name}_avg_pnl'] = avg_pnl
        results[f'{thr_name}_pf'] = pf
        results[f'{thr_name}_wr'] = wr

    return results


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
    log("XGBoost Confidence Gate v1 — Prediction-Based Features")
    log(f"Commission: ${COMMISSION_TICKS * TICK_VALUE:.2f} RT = {COMMISSION_TICKS:.3f} ticks (HC #52)")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    # ─── Step 1: Load training data (10 OOT folds) ──────────────────────────
    log("\n--- Loading 10 OOT training folds ---")
    train_folds = []
    for i in range(10):
        fd = load_oot_fold_data(i)
        if fd is not None:
            train_folds.append(fd)
            log(f"  Fold {i}: date={fd['date']}, n={fd['n']}, features={fd['X'].shape[1]}")

    if len(train_folds) < 5:
        log(f"ERROR: Only {len(train_folds)} folds loaded, need at least 5")
        return

    feat_names = train_folds[0]['feat_names']
    log(f"\nTotal training folds: {len(train_folds)}")
    log(f"Features ({len(feat_names)}): {', '.join(feat_names[:10])}...")

    # ─── Step 2: Load validation data (39 new dates) ────────────────────────
    log("\n--- Loading 39 new dates from decay v4 ---")
    new_dates = sorted([d for d in os.listdir(DECAY_CNN_DIR)
                       if os.path.isdir(os.path.join(DECAY_CNN_DIR, d)) and d.isdigit()])

    val_folds = []
    for date_str in new_dates:
        fd = load_decay_date_data(date_str)
        if fd is not None:
            val_folds.append(fd)
            has_vol = 'vol_pred' in fd['feat_names']
            log(f"  {date_str}: n={fd['n']}, vol={'yes' if has_vol else 'no'}")

    log(f"\nTotal validation dates: {len(val_folds)}")

    # ─── Step 3: Train XGBoost gate (leave-one-out on training set) ──────────
    try:
        import xgboost as xgb
    except ImportError:
        log("ERROR: xgboost not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
        import xgboost as xgb

    log("\n" + "=" * 80)
    log("PHASE 1: Leave-One-Out CV on Training Dates")
    log("=" * 80)

    # For each horizon, train gate and evaluate
    for h_idx, horizon in enumerate(HORIZONS):
        log(f"\n{'='*60}")
        log(f"HORIZON: {horizon}")
        log(f"{'='*60}")

        # Leave-one-out CV
        loo_results = []
        for val_idx in range(len(train_folds)):
            # Train on all except val_idx
            X_trains = [f['X'] for j, f in enumerate(train_folds) if j != val_idx]

            # Label = 1 if CNN prediction direction is correct
            y_trains = []
            for j, f in enumerate(train_folds):
                if j != val_idx:
                    correct = (np.sign(f['preds'][:, h_idx]) == np.sign(f['labels'][:, h_idx])).astype(float)
                    y_trains.append(correct)

            X_train = np.vstack(X_trains)
            y_train = np.concatenate(y_trains)

            # Validation
            val_f = train_folds[val_idx]
            X_val = val_f['X']
            y_val_correct = (np.sign(val_f['preds'][:, h_idx]) == np.sign(val_f['labels'][:, h_idx])).astype(float)

            # Train XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
            dval = xgb.DMatrix(X_val, label=y_val_correct, feature_names=feat_names)

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 50,
                'seed': 42,
                'verbosity': 0,
                'nthread': 14,  # Jupiter has 16 cores
            }

            model = xgb.train(
                params, dtrain, num_boost_round=300,
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            # Get gate scores
            gate_scores = model.predict(dval)

            # Evaluate
            res = evaluate_gate(val_f['preds'], val_f['labels'], gate_scores, val_f['date'], h_idx)
            loo_results.append(res)

            # Print compact result
            t1 = res.get('Top1%_da', 0)
            t5 = res.get('Top5%_da', 0)
            t1_lift = res.get('Top1%_lift', 0)
            t5_lift = res.get('Top5%_lift', 0)
            log(f"  {val_f['date']}: baseline={res['baseline_da']:.1%}  "
                f"Top5%={t5:.1%}(+{t5_lift:.1%})  Top1%={t1:.1%}(+{t1_lift:.1%})")

        # Concat LOO results
        log(f"\n  LOO CONCAT ({horizon}):")
        for tier in ['Top10%', 'Top5%', 'Top1%']:
            das = [r.get(f'{tier}_da', 0) for r in loo_results if f'{tier}_da' in r]
            lifts = [r.get(f'{tier}_lift', 0) for r in loo_results if f'{tier}_lift' in r]
            sortinos = [r.get(f'{tier}_sortino', 0) for r in loo_results if f'{tier}_sortino' in r]
            pfs = [r.get(f'{tier}_pf', 0) for r in loo_results if f'{tier}_pf' in r]
            if das:
                log(f"    {tier}: DA={np.mean(das):.1%} (±{np.std(das):.1%})  "
                    f"Lift={np.mean(lifts):+.1%}  Sortino={np.mean(sortinos):.1f}  PF={np.mean(pfs):.2f}")

    # ─── Step 4: Train FULL model on all 10 dates, test on 39 new ───────────
    log("\n" + "=" * 80)
    log("PHASE 2: Full Model → 39 New Dates")
    log("=" * 80)

    if not val_folds:
        log("ERROR: No validation dates loaded")
        return

    for h_idx, horizon in enumerate(HORIZONS):
        log(f"\n{'='*60}")
        log(f"HORIZON: {horizon} — Testing on {len(val_folds)} new dates")
        log(f"{'='*60}")

        # Train on ALL 10 OOT folds
        X_train = np.vstack([f['X'] for f in train_folds])
        y_train = np.concatenate([
            (np.sign(f['preds'][:, h_idx]) == np.sign(f['labels'][:, h_idx])).astype(float)
            for f in train_folds
        ])

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 50,
            'seed': 42,
            'verbosity': 0,
            'nthread': 14,
        }

        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

        # Save model
        model.save_model(os.path.join(OUTPUT_DIR, f"xgb_gate_{horizon}.json"))

        # Feature importance
        importance = model.get_score(importance_type='gain')
        top_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        log(f"\n  Top features ({horizon}):")
        for fname, gain in top_feats:
            log(f"    {fname}: gain={gain:.1f}")

        # Test on each new date
        new_results = []
        for vf in val_folds:
            dtest = xgb.DMatrix(vf['X'], feature_names=feat_names)
            gate_scores = model.predict(dtest)
            res = evaluate_gate(vf['preds'], vf['labels'], gate_scores, vf['date'], h_idx)
            new_results.append(res)

            t5_da = res.get('Top5%_da', 0)
            t5_lift = res.get('Top5%_lift', 0)
            t1_da = res.get('Top1%_da', 0)
            t1_lift = res.get('Top1%_lift', 0)
            log(f"  {vf['date']}: baseline={res['baseline_da']:.1%}  "
                f"Top5%={t5_da:.1%}(+{t5_lift:.1%})  Top1%={t1_da:.1%}(+{t1_lift:.1%})")

        # Summary statistics
        log(f"\n  {'='*50}")
        log(f"  39-DATE VALIDATION SUMMARY ({horizon}):")
        log(f"  {'='*50}")

        baseline_das = [r['baseline_da'] for r in new_results]
        log(f"  Baseline DA: {np.mean(baseline_das):.1%} (±{np.std(baseline_das):.1%})")

        for tier in ['Top10%', 'Top5%', 'Top1%']:
            das = [r[f'{tier}_da'] for r in new_results if f'{tier}_da' in r]
            lifts = [r[f'{tier}_lift'] for r in new_results if f'{tier}_lift' in r]
            sortinos = [r[f'{tier}_sortino'] for r in new_results if f'{tier}_sortino' in r]
            pfs = [r[f'{tier}_pf'] for r in new_results if f'{tier}_pf' in r]
            wrs = [r[f'{tier}_wr'] for r in new_results if f'{tier}_wr' in r]
            avg_pnls = [r[f'{tier}_avg_pnl'] for r in new_results if f'{tier}_avg_pnl' in r]

            if das:
                log(f"\n  {tier} ({len(das)} dates):")
                log(f"    DA:      {np.mean(das):.1%} (±{np.std(das):.1%})")
                log(f"    Lift:    {np.mean(lifts):+.1%} (±{np.std(lifts):.1%})")
                log(f"    Sortino: {np.mean(sortinos):.1f} (±{np.std(sortinos):.1f})")
                log(f"    PF:      {np.mean(pfs):.2f} (±{np.std(pfs):.2f})")
                log(f"    WR:      {np.mean(wrs):.1%}")
                log(f"    AvgPnL:  ${np.mean(avg_pnls):.2f}")

                # Per-week breakdown for decay analysis
                dates_sorted = sorted([(r['date'], r.get(f'{tier}_da', 0), r.get(f'{tier}_lift', 0))
                                      for r in new_results if f'{tier}_da' in r])
                if len(dates_sorted) > 10:
                    third = len(dates_sorted) // 3
                    early = dates_sorted[:third]
                    mid = dates_sorted[third:2*third]
                    late = dates_sorted[2*third:]
                    log(f"    Early ({early[0][0]}-{early[-1][0]}): DA={np.mean([x[1] for x in early]):.1%}, Lift={np.mean([x[2] for x in early]):+.1%}")
                    log(f"    Mid   ({mid[0][0]}-{mid[-1][0]}): DA={np.mean([x[1] for x in mid]):.1%}, Lift={np.mean([x[2] for x in mid]):+.1%}")
                    log(f"    Late  ({late[0][0]}-{late[-1][0]}): DA={np.mean([x[1] for x in late]):.1%}, Lift={np.mean([x[2] for x in late]):+.1%}")

    # Save results
    all_results = {
        'train_dates': [f['date'] for f in train_folds],
        'val_dates': [f['date'] for f in val_folds],
        'n_features': len(feat_names),
        'feature_names': feat_names,
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"\n{'='*80}")
    log(f"DONE. Results saved to {OUTPUT_DIR}/")
    log(f"{'='*80}")
    log_file.close()


if __name__ == "__main__":
    main()
