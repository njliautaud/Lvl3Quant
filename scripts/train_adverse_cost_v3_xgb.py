#!/usr/bin/env python3
r"""
Adverse-Cost Head v3 (XGBoost, GPU) — Microstructure path per HC #498 R2 + HC #499 R2.

Predicts realized adverse cost (in ticks) at horizons h in {1s, 5s, 10s} per signal
event using queue-augmented microstructure features + signal context features.

Inputs (on Razer):
  C:\Users\claude\Lvl3Quant\output\queue_augmented_features\features_YYYYMMDD.parquet
  C:\Users\claude\Lvl3Quant\output\adverse_cost_labels_v3\labels_YYYYMMDD.parquet

Outputs:
  C:\Users\claude\Lvl3Quant\output\adverse_cost_head_v3\
    smoke_summary.json
    full_summary.json
    fold_<NN>_oot_predictions_<h>.parquet (when full run)
    xgb_<h>_full.json (weights)
    training.log

Honest setup:
  * Target: realized adverse-cost ticks = max(0, -side * labels_h[event_id])
    (NOT closed-form of features).
  * Features: queue-augmented (~25 microstructure feats) + signal context (pred_1s/5s/10s)
    + simple engineered crosses, dropping (event_id, ts_ns, side, pred_*) when used as target inputs ONLY at training time per-horizon side handling.
  * Walk-forward: sliding 25-day train / 1-day eval. 13+ folds.
  * MLflow tracked.

Author: autonomous agent under HC #393, HC #420, HC #499 R2 binding.
"""
from __future__ import annotations
import sys, os, json, time, math, glob, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

LVL3_ROOT = Path(r'C:\Users\claude\Lvl3Quant') if os.name == 'nt' else Path('/home/jupiter/Lvl3Quant')
FEAT_DIR  = LVL3_ROOT / 'output' / 'queue_augmented_features'
LBL_DIR   = LVL3_ROOT / 'output' / 'adverse_cost_labels_v3'
OUT_DIR   = LVL3_ROOT / 'output' / 'adverse_cost_head_v3'
LOG_DIR   = LVL3_ROOT / 'logs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── logging ──────────────────────────────────────────────────────────────────
import logging
LOG_FILE = LOG_DIR / f'adverse_cost_v3_xgb_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger('advcost_v3')

HORIZONS = ['1s', '5s', '10s']
TARGET_COLS = {h: f'adverse_{h}' for h in HORIZONS}

# Successful gate from HC #498 R2 / task spec
V22_BASELINE_RMSE_5S = 4.78
STRONG_PASS_5S = 4.50

# HC #498 R3 honesty: realized adverse-cost has a fat right tail (FOMC/event days,
# max ~10000 ticks). v2.2 RMSE 4.78 was on a clipped/practical target — any real
# execution policy is bounded by a stop. Clip at 20 ticks (covers ~p99) so this
# head learns the executable cost surface, not lottery outliers. Documented.
TARGET_CLIP_TICKS = 20.0

# Walk-forward params (HC #0: SLIDING ONLY)
TRAIN_DAYS = 25
EVAL_DAYS  = 1

# Smoke params
SMOKE_FOLDS   = 2
SMOKE_ROUNDS  = 200
# Full params
FULL_ROUNDS   = 1000
EARLY_STOP    = 50

WALL_CAP_S    = 90 * 60  # 90 min total


# ─── data loading ─────────────────────────────────────────────────────────────
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
    # Join on (event_id, ts_ns)
    df = feat.merge(lbl, on=['event_id', 'ts_ns'], how='inner', suffixes=('', '_lbl'))
    # side from labels is canonical (queue feats might have a side too — keep labels')
    if 'side_lbl' in df.columns:
        df['side'] = df['side_lbl']; df.drop(columns=['side_lbl'], inplace=True)
    # Keep only valid
    if 'valid' in df.columns:
        df = df[df['valid'] == 1].reset_index(drop=True)
        df.drop(columns=['valid'], inplace=True)
    # Clip adverse-cost targets at executable bound (HC #498 R3 honesty)
    for h in HORIZONS:
        col = TARGET_COLS[h]
        if col in df.columns:
            df[col] = np.clip(df[col].to_numpy(dtype=np.float32), 0.0, TARGET_CLIP_TICKS)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric cols except identifiers and targets."""
    exclude = {'event_id', 'ts_ns'} | set(TARGET_COLS.values())
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feats


# ─── XGBoost training ─────────────────────────────────────────────────────────
def train_xgb(X_tr, y_tr, X_va, y_va, num_rounds: int, early_stop: int | None, gpu: bool):
    import xgboost as xgb
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    params = dict(
        objective='reg:pseudohubererror',  # robust to remaining tail
        huber_slope=1.0,
        eval_metric='rmse',
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method='hist',
    )
    if gpu:
        params['device'] = 'cuda'
    evals = [(dtr, 'train'), (dva, 'eval')]
    kwargs = dict(params=params, dtrain=dtr, num_boost_round=num_rounds, evals=evals, verbose_eval=False)
    if early_stop:
        kwargs['early_stopping_rounds'] = early_stop
    bst = xgb.train(**kwargs)
    pred_va = bst.predict(dva)
    rmse = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
    return bst, pred_va, rmse


# ─── walk-forward driver ──────────────────────────────────────────────────────
def build_folds(dates: list[str], train_days: int, eval_days: int = 1) -> list[tuple[list[str], list[str]]]:
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


def run_smoke(dates: list[str], gpu: bool, mlflow_run=None) -> dict:
    """2-fold smoke on h=5s only."""
    log.info(f'=== SMOKE: {SMOKE_FOLDS} folds, {SMOKE_ROUNDS} rounds, h=5s ===')
    folds = build_folds(dates, TRAIN_DAYS, EVAL_DAYS)
    if len(folds) < SMOKE_FOLDS:
        log.warning(f'Only {len(folds)} folds available, need {SMOKE_FOLDS}')
    smoke = folds[:SMOKE_FOLDS]
    rmses = []
    for k, (tr_d, ev_d) in enumerate(smoke):
        t0 = time.time()
        log.info(f'  smoke fold {k}: tr={tr_d[0]}..{tr_d[-1]} ({len(tr_d)}d) ev={ev_d[0]}')
        tr_df = concat_load(tr_d)
        ev_df = concat_load(ev_d)
        if len(tr_df) == 0 or len(ev_df) == 0:
            log.warning(f'    empty fold, skip')
            continue
        feats = feature_columns(tr_df)
        log.info(f'    n_feats={len(feats)} n_tr={len(tr_df):,} n_va={len(ev_df):,}')
        X_tr = tr_df[feats].to_numpy(dtype=np.float32)
        X_va = ev_df[feats].to_numpy(dtype=np.float32)
        y_tr = tr_df[TARGET_COLS['5s']].to_numpy(dtype=np.float32)
        y_va = ev_df[TARGET_COLS['5s']].to_numpy(dtype=np.float32)
        _, pred, rmse = train_xgb(X_tr, y_tr, X_va, y_va, SMOKE_ROUNDS, None, gpu)
        elapsed = time.time() - t0
        log.info(f'    fold {k} RMSE_5s={rmse:.4f}  elapsed={elapsed:.1f}s')
        rmses.append(rmse)
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metric(f'smoke_fold{k}_rmse_5s', rmse)
            except Exception:
                pass
    avg = float(np.mean(rmses)) if rmses else float('nan')
    log.info(f'=== SMOKE complete: avg_RMSE_5s={avg:.4f} vs v2.2 baseline {V22_BASELINE_RMSE_5S} ===')
    return {'folds': len(rmses), 'rmses': rmses, 'avg_rmse_5s': avg,
            'baseline_5s': V22_BASELINE_RMSE_5S, 'pass': avg < V22_BASELINE_RMSE_5S}


def run_full(dates: list[str], gpu: bool, mlflow_run=None, wall_deadline: float | None = None) -> dict:
    """Full walk-forward over all horizons."""
    log.info(f'=== FULL WALK-FORWARD: 3 horizons x N folds, {FULL_ROUNDS} rounds ===')
    folds = build_folds(dates, TRAIN_DAYS, EVAL_DAYS)
    log.info(f'Total folds available: {len(folds)}')

    per_h = {h: {'fold_rmses': [], 'preds': [], 'truths': []} for h in HORIZONS}
    last_models = {}

    for k, (tr_d, ev_d) in enumerate(folds):
        if wall_deadline and time.time() > wall_deadline:
            log.warning(f'wall-time deadline hit at fold {k}; stopping early.')
            break
        t0 = time.time()
        log.info(f'fold {k}: tr={tr_d[0]}..{tr_d[-1]} ({len(tr_d)}d) ev={ev_d[0]}')
        tr_df = concat_load(tr_d)
        ev_df = concat_load(ev_d)
        if len(tr_df) == 0 or len(ev_df) == 0:
            log.warning(f'  empty fold, skip'); continue
        feats = feature_columns(tr_df)
        X_tr = tr_df[feats].to_numpy(dtype=np.float32)
        X_va = ev_df[feats].to_numpy(dtype=np.float32)
        log.info(f'  n_feats={len(feats)} n_tr={len(tr_df):,} n_va={len(ev_df):,}')
        for h in HORIZONS:
            y_tr = tr_df[TARGET_COLS[h]].to_numpy(dtype=np.float32)
            y_va = ev_df[TARGET_COLS[h]].to_numpy(dtype=np.float32)
            bst, pred, rmse = train_xgb(X_tr, y_tr, X_va, y_va, FULL_ROUNDS, EARLY_STOP, gpu)
            per_h[h]['fold_rmses'].append(rmse)
            per_h[h]['preds'].append(pred)
            per_h[h]['truths'].append(y_va)
            last_models[h] = bst
            log.info(f'  fold {k} h={h}: RMSE={rmse:.4f}')
            if mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.log_metric(f'fold{k}_rmse_{h}', rmse)
                except Exception:
                    pass
        log.info(f'  fold {k} elapsed={time.time()-t0:.1f}s')

    summary = {'folds_run': len(per_h['5s']['fold_rmses']), 'per_horizon': {}}
    for h in HORIZONS:
        rmses = per_h[h]['fold_rmses']
        if not rmses:
            summary['per_horizon'][h] = {'folds': 0}
            continue
        all_pred = np.concatenate(per_h[h]['preds'])
        all_truth = np.concatenate(per_h[h]['truths'])
        concat_rmse = float(np.sqrt(np.mean((all_pred - all_truth) ** 2)))
        summary['per_horizon'][h] = {
            'folds': len(rmses),
            'mean_fold_rmse': float(np.mean(rmses)),
            'median_fold_rmse': float(np.median(rmses)),
            'concat_rmse': concat_rmse,
            'fold_rmses': [float(x) for x in rmses],
        }
        log.info(f'h={h}: concat RMSE={concat_rmse:.4f}, mean fold RMSE={np.mean(rmses):.4f}, folds={len(rmses)}')
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metric(f'concat_rmse_{h}', concat_rmse)
                mlflow.log_metric(f'mean_fold_rmse_{h}', float(np.mean(rmses)))
            except Exception:
                pass
        # Save last model
        try:
            mdl_path = OUT_DIR / f'xgb_{h}_full.json'
            last_models[h].save_model(str(mdl_path))
            log.info(f'  saved {mdl_path}')
        except Exception as e:
            log.warning(f'  model save failed: {e}')
    # verdict on 5s
    c5 = summary['per_horizon'].get('5s', {}).get('concat_rmse', math.nan)
    if c5 < STRONG_PASS_5S:
        verdict = 'STRONG_PASS'
    elif c5 < V22_BASELINE_RMSE_5S:
        verdict = 'PASS_MARGINAL'
    elif c5 < V22_BASELINE_RMSE_5S + 0.1:
        verdict = 'MARGINAL'
    else:
        verdict = 'REJECT'
    summary['verdict_5s'] = verdict
    summary['baseline_5s_v22'] = V22_BASELINE_RMSE_5S
    return summary


# ─── MLflow ───────────────────────────────────────────────────────────────────
def setup_mlflow():
    try:
        import mlflow
        # Try local first (remote MLflow endpoints have been timing out / 403)
        for uri in ['file:./mlruns']:
            try:
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment('adverse_cost_head_v3')
                run = mlflow.start_run(run_name=f'adv_cost_v3_xgb_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                log.info(f'MLflow tracking at {uri}, run_id={run.info.run_id}')
                return run
            except Exception as e:
                log.warning(f'MLflow uri {uri} failed: {e}')
        return None
    except Exception as e:
        log.warning(f'MLflow not available: {e}')
        return None


# ─── Sunday cutoff guard ──────────────────────────────────────────────────────
def check_sunday_cutoff():
    """Per task: pause if past Sunday 6 PM ET (markets reopening)."""
    from datetime import datetime
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo('America/New_York'))
    except Exception:
        # fallback: assume utc-4
        et = datetime.utcnow()
    # Sunday is weekday() == 6
    if et.weekday() == 6 and et.hour >= 18:
        log.error(f'PAST SUNDAY 6PM ET ({et}); paper trader must own GPU. Exiting.')
        sys.exit(2)


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true', help='Run smoke only then exit')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    check_sunday_cutoff()

    t_start = time.time()
    wall_deadline = t_start + WALL_CAP_S

    dates = list_dates()
    if len(dates) < TRAIN_DAYS + 1:
        log.error(f'Not enough dates: {len(dates)} < {TRAIN_DAYS+1}')
        sys.exit(1)

    # GPU check
    gpu = not args.no_gpu
    if gpu:
        try:
            import xgboost as xgb
            log.info(f'xgboost version: {xgb.__version__}')
        except Exception as e:
            log.error(f'xgboost import failed: {e}')
            sys.exit(1)

    mlf = setup_mlflow()
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
                'baseline_5s_v22': V22_BASELINE_RMSE_5S,
                'gate_strong_pass_5s': STRONG_PASS_5S,
            })

        smoke = run_smoke(dates, gpu, mlf)
        (OUT_DIR / 'smoke_summary.json').write_text(json.dumps(smoke, indent=2))
        log.info(f'smoke_summary: {json.dumps(smoke, indent=2)}')

        if args.smoke_only:
            log.info('smoke-only requested, exiting.')
            return

        if not smoke['pass']:
            log.warning(f"Smoke RMSE_5s={smoke['avg_rmse_5s']:.4f} >= v2.2 baseline {V22_BASELINE_RMSE_5S}. "
                        f"Proceeding to full per task spec (honest verdict).")

        full = run_full(dates, gpu, mlf, wall_deadline)
        (OUT_DIR / 'full_summary.json').write_text(json.dumps(full, indent=2))
        log.info(f'full_summary verdict: {full.get("verdict_5s")} (5s concat RMSE = '
                 f'{full["per_horizon"].get("5s", {}).get("concat_rmse", "nan")})')
        log.info(f'Total wall: {(time.time()-t_start)/60:.1f} min')
    finally:
        if mlf is not None:
            try:
                import mlflow; mlflow.end_run()
            except Exception:
                pass


if __name__ == '__main__':
    main()
