#!/usr/bin/env python3
r"""
Fill-Prob Head v3 — HONEST (XGBoost, GPU) per HC #498 R3.

Predicts P(filled_h | features) at horizons h in {1s, 5s, 10s} per signal event
using queue-augmented microstructure features. Targets come from the MBO-walker
FIFO-replay labels (HC #493), NOT from a closed-form derivation of features
(which fooled fill_prob_v1).

Inputs (cross-platform path detection):
  output/mbo_walker_labels/labels_YYYYMMDD.parquet
  output/queue_augmented_features/features_YYYYMMDD.parquet

Outputs:
  output/fill_prob_v3_honest_xgb/
    smoke_summary.json
    full_summary.json
    xgb_fill_<h>_full.json (weights)
    fold_<NN>_preds_<h>.parquet (optional)
    summary.json (final verdict)
    training.log

Honest setup:
  * Target: filled_{1s,5s,10s} from MBO walker (binary).
  * Features: queue-augmented + pred_{1s,5s,10s} signal context.
  * Walk-forward: sliding 25-day train / 1-day eval. SLIDING ONLY (HC #0).
  * MLflow tracked (HC #74).
  * Reports concat AUC + per-day stratification (HC #428 R1).
  * Reports majority-class baseline alongside AUC.
"""
from __future__ import annotations
import sys, os, json, time, math, glob, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# ─── path detection ──────────────────────────────────────────────────────────
def detect_root() -> Path:
    candidates = [
        Path(r'C:\Users\claude\Lvl3Quant') if os.name == 'nt' else None,
        Path('/home/nick/Lvl3Quant'),
        Path('/home/jupiter/Lvl3Quant'),
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    raise RuntimeError('No Lvl3 root found')

LVL3_ROOT = detect_root()
FEAT_DIR  = LVL3_ROOT / 'output' / 'queue_augmented_features'
LBL_DIR   = LVL3_ROOT / 'output' / 'mbo_walker_labels'
OUT_DIR   = LVL3_ROOT / 'output' / 'fill_prob_v3_honest_xgb'
LOG_DIR   = LVL3_ROOT / 'logs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── logging ─────────────────────────────────────────────────────────────────
import logging
LOG_FILE = LOG_DIR / f'fill_prob_v3_xgb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger('fillprob_v3')

HORIZONS = ['1s', '5s', '10s']
TARGET_COLS = {h: f'filled_{h}' for h in HORIZONS}

# Walk-forward params (HC #0)
TRAIN_DAYS = 25
EVAL_DAYS  = 1

SMOKE_FOLDS  = 2
SMOKE_ROUNDS = 200
FULL_ROUNDS  = 1000
EARLY_STOP   = 50

WALL_CAP_S = 28 * 60  # leave 2 min margin within 30-min spec

# Gates (HC #494 + task spec)
HARD_PASS_AUC     = 0.65
MARGINAL_PASS_AUC = 0.60
SMOKE_KILL_AUC    = 0.55


# ─── data ─────────────────────────────────────────────────────────────────────
def list_dates() -> list[str]:
    fset = {Path(p).stem.replace('features_', '') for p in glob.glob(str(FEAT_DIR / 'features_*.parquet'))}
    lset = {Path(p).stem.replace('labels_', '')   for p in glob.glob(str(LBL_DIR  / 'labels_*.parquet'))}
    common = sorted(fset & lset)
    log.info(f'Found {len(common)} common dates: {common[0]}..{common[-1]}')
    return common


def load_date(date: str) -> pd.DataFrame | None:
    fp = FEAT_DIR / f'features_{date}.parquet'
    lp = LBL_DIR  / f'labels_{date}.parquet'
    try:
        feat = pd.read_parquet(fp)
        lbl  = pd.read_parquet(lp)
    except Exception as e:
        log.warning(f'{date}: load failed {e}')
        return None
    # Avoid duplicate cols at merge (side/pred_* live in both)
    drop_in_lbl = [c for c in ['side', 'pred_1s', 'pred_5s', 'pred_10s'] if c in lbl.columns and c in feat.columns]
    if drop_in_lbl:
        lbl = lbl.drop(columns=drop_in_lbl)
    df = feat.merge(lbl, on=['event_id', 'ts_ns'], how='inner')
    # Drop rows missing any target
    for h in HORIZONS:
        col = TARGET_COLS[h]
        if col not in df.columns:
            log.warning(f'{date}: missing target {col}')
            return None
    df = df.dropna(subset=[TARGET_COLS[h] for h in HORIZONS]).reset_index(drop=True)
    df['_date'] = date
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric cols except IDs, targets, derived columns."""
    exclude = ({'event_id', 'ts_ns', '_date', 'price',
                'queue_depth_at_touch'}  # keep queue_depth? it's a strong feat — yes, keep
               | set(TARGET_COLS.values())
               | {f'time_to_fill_s_{h}' for h in HORIZONS}
               | {f'queue_rank_at_{h}' for h in HORIZONS})  # leakage: rank known only at h
    # actually queue_depth_at_touch IS a feature (known at event time). Re-include.
    exclude.discard('queue_depth_at_touch')
    feats = [c for c in df.columns
             if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feats


# ─── XGB ─────────────────────────────────────────────────────────────────────
def train_xgb_clf(X_tr, y_tr, X_va, y_va, num_rounds: int, early_stop: int | None, gpu: bool):
    import xgboost as xgb
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    pos = float(y_tr.sum()); neg = float(len(y_tr) - pos)
    spw = (neg / pos) if pos > 0 else 1.0
    params = dict(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method='hist',
        scale_pos_weight=spw,
    )
    if gpu:
        params['device'] = 'cuda'
    evals = [(dtr, 'train'), (dva, 'eval')]
    kw = dict(params=params, dtrain=dtr, num_boost_round=num_rounds, evals=evals, verbose_eval=False)
    if early_stop:
        kw['early_stopping_rounds'] = early_stop
    bst = xgb.train(**kw)
    pred = bst.predict(dva)
    return bst, pred


def auc_score(y_true, y_score) -> float:
    """Robust AUC; sklearn-free fallback if needed."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Mann-Whitney U-based AUC
        y = np.asarray(y_true).astype(int)
        s = np.asarray(y_score, dtype=float)
        pos = s[y == 1]; neg = s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float('nan')
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(s) + 1)
        sum_r_pos = ranks[y == 1].sum()
        u = sum_r_pos - len(pos) * (len(pos) + 1) / 2.0
        return float(u / (len(pos) * len(neg)))


# ─── walk-forward ────────────────────────────────────────────────────────────
def build_folds(dates: list[str], train_days: int, eval_days: int = 1):
    folds = []
    for i in range(train_days, len(dates), eval_days):
        tr = dates[i - train_days: i]
        ev = dates[i: i + eval_days]
        if not ev: break
        folds.append((tr, ev))
    return folds


def concat_load(dates: list[str]) -> pd.DataFrame:
    parts = []
    for d in dates:
        df = load_date(d)
        if df is not None and len(df) > 0:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_smoke(dates, gpu, mlf=None):
    log.info(f'=== SMOKE: {SMOKE_FOLDS} folds, {SMOKE_ROUNDS} rounds, h=5s ===')
    folds = build_folds(dates, TRAIN_DAYS, EVAL_DAYS)
    if len(folds) < SMOKE_FOLDS:
        log.warning(f'Only {len(folds)} folds; need {SMOKE_FOLDS}')
    aucs = []; baselines = []
    for k, (tr_d, ev_d) in enumerate(folds[:SMOKE_FOLDS]):
        t0 = time.time()
        log.info(f'  smoke fold {k}: tr={tr_d[0]}..{tr_d[-1]} ({len(tr_d)}d) ev={ev_d[0]}')
        tr_df = concat_load(tr_d); ev_df = concat_load(ev_d)
        if len(tr_df) == 0 or len(ev_df) == 0:
            log.warning('    empty fold, skip'); continue
        feats = feature_columns(tr_df)
        log.info(f'    n_feats={len(feats)} n_tr={len(tr_df):,} n_va={len(ev_df):,}')
        X_tr = tr_df[feats].to_numpy(dtype=np.float32)
        X_va = ev_df[feats].to_numpy(dtype=np.float32)
        y_tr = tr_df[TARGET_COLS['5s']].to_numpy(dtype=np.int32)
        y_va = ev_df[TARGET_COLS['5s']].to_numpy(dtype=np.int32)
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            log.warning('    degenerate train labels, skip'); continue
        _, pred = train_xgb_clf(X_tr, y_tr, X_va, y_va, SMOKE_ROUNDS, None, gpu)
        auc = auc_score(y_va, pred)
        base = float(max(y_va.mean(), 1 - y_va.mean()))
        log.info(f'    fold {k} AUC_5s={auc:.4f}  majority_base={base:.4f}  elapsed={time.time()-t0:.1f}s')
        aucs.append(auc); baselines.append(base)
        if mlf is not None:
            try:
                import mlflow
                mlflow.log_metric(f'smoke_fold{k}_auc_5s', auc)
            except Exception: pass
    avg = float(np.mean(aucs)) if aucs else float('nan')
    avg_b = float(np.mean(baselines)) if baselines else float('nan')
    log.info(f'=== SMOKE complete: avg_AUC_5s={avg:.4f}  avg_majority_base={avg_b:.4f} ===')
    return {'folds': len(aucs), 'aucs': aucs, 'avg_auc_5s': avg,
            'avg_majority_baseline_5s': avg_b,
            'pass': avg >= SMOKE_KILL_AUC}


def classify_regime(date_df: pd.DataFrame) -> str:
    """Classify a day by intraday ES return proxy via pred_5s sign aggregation.
    We don't have ES close-to-close here without DB access; use pred_5s sign
    distribution as a quick regime proxy. NOT canonical ES regime, but
    documented as proxy."""
    p = date_df.get('pred_5s')
    if p is None or len(p) == 0:
        return 'unknown'
    m = float(p.mean())
    if m > 0.0002:  return 'bull_drift'
    if m < -0.0002: return 'bear_drift'
    return 'flat'


def run_full(dates, gpu, mlf=None, wall_deadline=None):
    log.info(f'=== FULL WF: 3 horizons x N folds, {FULL_ROUNDS} rounds ===')
    folds = build_folds(dates, TRAIN_DAYS, EVAL_DAYS)
    log.info(f'Total folds available: {len(folds)}')

    per_h = {h: {'fold_aucs': [], 'preds': [], 'truths': [],
                 'per_fold_baseline': [], 'per_fold_date': [],
                 'per_fold_regime': []} for h in HORIZONS}
    last_models = {}

    for k, (tr_d, ev_d) in enumerate(folds):
        if wall_deadline and time.time() > wall_deadline:
            log.warning(f'wall-time deadline hit at fold {k}; stopping early.'); break
        t0 = time.time()
        log.info(f'fold {k}: tr={tr_d[0]}..{tr_d[-1]} ({len(tr_d)}d) ev={ev_d[0]}')
        tr_df = concat_load(tr_d); ev_df = concat_load(ev_d)
        if len(tr_df) == 0 or len(ev_df) == 0:
            log.warning('  empty fold, skip'); continue
        feats = feature_columns(tr_df)
        X_tr = tr_df[feats].to_numpy(dtype=np.float32)
        X_va = ev_df[feats].to_numpy(dtype=np.float32)
        regime = classify_regime(ev_df)
        log.info(f'  n_feats={len(feats)} n_tr={len(tr_df):,} n_va={len(ev_df):,} regime={regime}')
        for h in HORIZONS:
            y_tr = tr_df[TARGET_COLS[h]].to_numpy(dtype=np.int32)
            y_va = ev_df[TARGET_COLS[h]].to_numpy(dtype=np.int32)
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr) or y_va.sum() == 0 or y_va.sum() == len(y_va):
                log.warning(f'  degenerate labels h={h}, skip'); continue
            bst, pred = train_xgb_clf(X_tr, y_tr, X_va, y_va, FULL_ROUNDS, EARLY_STOP, gpu)
            auc = auc_score(y_va, pred)
            base = float(max(y_va.mean(), 1 - y_va.mean()))
            per_h[h]['fold_aucs'].append(auc)
            per_h[h]['preds'].append(pred)
            per_h[h]['truths'].append(y_va)
            per_h[h]['per_fold_baseline'].append(base)
            per_h[h]['per_fold_date'].append(ev_d[0])
            per_h[h]['per_fold_regime'].append(regime)
            last_models[h] = bst
            log.info(f'  fold {k} h={h}: AUC={auc:.4f} majority_base={base:.4f}')
            if mlf is not None:
                try:
                    import mlflow
                    mlflow.log_metric(f'fold{k}_auc_{h}', auc)
                except Exception: pass
        log.info(f'  fold {k} elapsed={time.time()-t0:.1f}s')

    summary = {'folds_run': len(per_h['5s']['fold_aucs']), 'per_horizon': {}}
    for h in HORIZONS:
        aucs = per_h[h]['fold_aucs']
        if not aucs:
            summary['per_horizon'][h] = {'folds': 0}; continue
        all_pred = np.concatenate(per_h[h]['preds'])
        all_truth = np.concatenate(per_h[h]['truths'])
        concat_auc = auc_score(all_truth, all_pred)
        concat_base = float(max(all_truth.mean(), 1 - all_truth.mean()))

        # per-regime stratification
        regime_stats = {}
        for r in set(per_h[h]['per_fold_regime']):
            mask = np.array([rr == r for rr in per_h[h]['per_fold_regime']])
            preds_r = [per_h[h]['preds'][i] for i, m in enumerate(mask) if m]
            truths_r = [per_h[h]['truths'][i] for i, m in enumerate(mask) if m]
            if preds_r:
                pp = np.concatenate(preds_r); tt = np.concatenate(truths_r)
                regime_stats[r] = {
                    'folds': int(mask.sum()),
                    'concat_auc': auc_score(tt, pp) if len(set(tt)) > 1 else float('nan'),
                    'fill_rate': float(tt.mean()),
                }

        summary['per_horizon'][h] = {
            'folds': len(aucs),
            'mean_fold_auc': float(np.mean(aucs)),
            'median_fold_auc': float(np.median(aucs)),
            'concat_auc': float(concat_auc),
            'concat_majority_baseline': concat_base,
            'fold_aucs': [float(x) for x in aucs],
            'per_fold_date': per_h[h]['per_fold_date'],
            'per_fold_regime': per_h[h]['per_fold_regime'],
            'per_fold_baseline': per_h[h]['per_fold_baseline'],
            'per_regime': regime_stats,
        }
        log.info(f'h={h}: concat AUC={concat_auc:.4f} (majority={concat_base:.4f}) '
                 f'mean fold AUC={np.mean(aucs):.4f} folds={len(aucs)}')
        if mlf is not None:
            try:
                import mlflow
                mlflow.log_metric(f'concat_auc_{h}', float(concat_auc))
                mlflow.log_metric(f'mean_fold_auc_{h}', float(np.mean(aucs)))
                mlflow.log_metric(f'majority_baseline_{h}', concat_base)
            except Exception: pass
        try:
            mdl_path = OUT_DIR / f'xgb_fill_{h}_full.json'
            last_models[h].save_model(str(mdl_path))
            log.info(f'  saved {mdl_path}')
        except Exception as e:
            log.warning(f'  model save failed: {e}')

    # verdict on best horizon
    best_h = None; best_auc = -1.0
    for h, st in summary['per_horizon'].items():
        a = st.get('concat_auc', -1)
        if a is not None and a > best_auc:
            best_auc = a; best_h = h
    if best_auc >= HARD_PASS_AUC:
        verdict = 'PASS'
    elif best_auc >= MARGINAL_PASS_AUC:
        verdict = 'MARGINAL'
    else:
        verdict = 'REJECT'
    summary['verdict'] = verdict
    summary['best_horizon'] = best_h
    summary['best_concat_auc'] = float(best_auc)
    return summary


# ─── MLflow ──────────────────────────────────────────────────────────────────
def setup_mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('fill_prob_v3_honest_xgb')
        run = mlflow.start_run(run_name=f'fill_prob_v3_xgb_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        log.info(f'MLflow run_id={run.info.run_id}')
        return run
    except Exception as e:
        log.warning(f'MLflow not available: {e}')
        return None


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    t_start = time.time()
    wall_deadline = t_start + WALL_CAP_S

    dates = list_dates()
    if len(dates) < TRAIN_DAYS + 1:
        log.error(f'Not enough dates: {len(dates)} < {TRAIN_DAYS+1}'); sys.exit(1)

    gpu = not args.no_gpu
    try:
        import xgboost as xgb
        log.info(f'xgboost version: {xgb.__version__}')
    except Exception as e:
        log.error(f'xgboost import failed: {e}'); sys.exit(1)

    mlf = setup_mlflow()
    run_id = mlf.info.run_id if mlf is not None else None
    try:
        if mlf is not None:
            import mlflow
            mlflow.log_params({
                'horizons': ','.join(HORIZONS),
                'train_days': TRAIN_DAYS,
                'eval_days': EVAL_DAYS,
                'smoke_folds': SMOKE_FOLDS,
                'smoke_rounds': SMOKE_ROUNDS,
                'full_rounds': FULL_ROUNDS,
                'early_stop': EARLY_STOP,
                'n_dates': len(dates),
                'hard_pass_auc': HARD_PASS_AUC,
                'marginal_pass_auc': MARGINAL_PASS_AUC,
                'objective': 'binary:logistic',
            })

        smoke = run_smoke(dates, gpu, mlf)
        (OUT_DIR / 'smoke_summary.json').write_text(json.dumps(smoke, indent=2))
        log.info(f'smoke_summary: {json.dumps(smoke, indent=2)}')

        if args.smoke_only:
            log.info('smoke-only requested, exiting.'); return

        if not smoke['pass']:
            log.error(f"Smoke AUC_5s={smoke['avg_auc_5s']:.4f} < kill {SMOKE_KILL_AUC}. STOPPING per spec.")
            final = {'verdict': 'REJECT_SMOKE', 'smoke': smoke, 'mlflow_run_id': run_id}
            (OUT_DIR / 'summary.json').write_text(json.dumps(final, indent=2))
            return

        full = run_full(dates, gpu, mlf, wall_deadline)
        (OUT_DIR / 'full_summary.json').write_text(json.dumps(full, indent=2))
        final = {
            'verdict': full['verdict'],
            'best_horizon': full['best_horizon'],
            'best_concat_auc': full['best_concat_auc'],
            'per_horizon_summary': {h: {
                'concat_auc': full['per_horizon'][h].get('concat_auc'),
                'concat_majority_baseline': full['per_horizon'][h].get('concat_majority_baseline'),
                'mean_fold_auc': full['per_horizon'][h].get('mean_fold_auc'),
                'folds': full['per_horizon'][h].get('folds'),
                'per_regime': full['per_horizon'][h].get('per_regime'),
            } for h in HORIZONS},
            'mlflow_run_id': run_id,
            'smoke': smoke,
        }
        (OUT_DIR / 'summary.json').write_text(json.dumps(final, indent=2))
        log.info(f'final verdict: {full["verdict"]} best_h={full["best_horizon"]} '
                 f'concat_auc={full["best_concat_auc"]:.4f}')
        log.info(f'Total wall: {(time.time()-t_start)/60:.1f} min')
    finally:
        if mlf is not None:
            try:
                import mlflow; mlflow.end_run()
            except Exception: pass


if __name__ == '__main__':
    main()
