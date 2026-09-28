"""
train_vol_lgbm.py — Volume Profile + Order Flow LGBM for 10s prediction.
"""

import os, sys, time, logging, socket, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import scipy.stats

WINDOW      = int(os.environ.get("LGBM_WINDOW",       256))
STRIDE      = int(os.environ.get("LGBM_STRIDE",       500))
TRAIN_DAYS  = int(os.environ.get("LGBM_TRAIN_DAYS",    60))
TEST_DAYS   = int(os.environ.get("LGBM_TEST_DAYS",     15))
N_FOLDS     = int(os.environ.get("LGBM_N_FOLDS",        9))
N_EST       = int(os.environ.get("LGBM_N_ESTIMATORS", 300))
LGBM_LR     = float(os.environ.get("LGBM_LR",         0.05))
NUM_LEAVES  = int(os.environ.get("LGBM_NUM_LEAVES",    63))
MIN_CHILD   = int(os.environ.get("LGBM_MIN_CHILD",     50))
SUBSAMPLE   = float(os.environ.get("LGBM_SUBSAMPLE",   0.8))
N_JOBS      = int(os.environ.get("LGBM_N_JOBS",        10))
MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://neptune-win:5002")
MLFLOW_EXP  = os.environ.get("MLFLOW_EXPERIMENT",   "VolLGBM_10s")
DATA_DIR    = Path(os.environ.get("LGBM_DATA_DIR", "/home/jupiter/Lvl3Quant/data/processed/mbo_events"))
N_RAW = 6

class _FH(logging.StreamHandler):
    def emit(self, r):
        super().emit(r); self.flush()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[_FH(sys.stdout)])
log = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False
    log.warning("mlflow not found")

import lightgbm as lgb

COL_TIME=0; COL_TYPE=1; COL_SIDE=2; COL_PRICE=3; COL_QTY=4; COL_SPREAD=5
TYPE_CANCEL=2


def compute_features_batch(windows):
    N, W = windows.shape[0], windows.shape[1]
    f = np.empty((N, 20), dtype=np.float32)
    f[:, :6]   = windows.mean(1)
    f[:, 6:12] = windows.std(1) + 1e-8
    sides = windows[:, :, 2]
    qty   = np.exp(windows[:, :, 4]) - 1
    f[:, 12] = (sides * qty).sum(1)
    ic  = (windows[:, :, 1] == 2)
    cb  = (ic & (sides > 0)).sum(1).astype(np.float32)
    cs  = (ic & (sides < 0)).sum(1).astype(np.float32)
    f[:, 13] = (cb - cs) / (cb + cs + 1e-8)
    td = np.exp(windows[:, :, 0]) - 1
    f[:, 14] = 1.0 / (td.mean(1) + 1e-6)
    f[:, 15] = windows[:, -1, 3] - windows[:, 0, 3]
    f[:, 16] = (sides * qty).mean(1)
    pr = windows[:, :, 3]
    pm = (pr.max(1) + pr.min(1)) * 0.5
    ps = pr.std(1) + 1e-8
    f[:, 17] = (np.abs(pr - pm[:, None]) < ps[:, None]).mean(1).astype(np.float32)
    f[:, 18] = f[:, 12]
    h = max(1, W // 2)
    f[:, 19] = windows[:, h:, 5].mean(1) - windows[:, :h, 5].mean(1)
    return f

def compute_features(window):
    W = len(window)
    feats = []
    raw_mean = window.mean(0); raw_std = window.std(0) + 1e-8
    feats.extend(raw_mean.tolist()); feats.extend(raw_std.tolist())
    sides = window[:, COL_SIDE]; qty = np.exp(window[:, COL_QTY]) - 1
    ofi = float(np.sum(sides * qty)); feats.append(ofi)
    is_cancel = (window[:, COL_TYPE] == TYPE_CANCEL)
    cancel_buy = float(np.sum(is_cancel & (sides > 0))); cancel_sell = float(np.sum(is_cancel & (sides < 0)))
    cancel_asym = (cancel_buy - cancel_sell) / (cancel_buy + cancel_sell + 1e-8); feats.append(cancel_asym)
    time_deltas = np.exp(window[:, COL_TIME]) - 1; mean_delta = float(time_deltas.mean()) + 1e-6
    feats.append(1.0 / mean_delta)
    feats.append(float(window[-1, COL_PRICE] - window[0, COL_PRICE]))
    feats.append(float(np.mean(sides * qty)))
    prices = window[:, COL_PRICE]; near_poc = float(np.mean(np.abs(prices - (prices.max()+prices.min())/2.0) < (prices.std()+1e-8))); feats.append(near_poc)
    feats.append(float(np.sum(sides * (np.exp(window[:, COL_QTY]) - 1))))
    half = max(1, W//2); feats.append(float(window[half:, COL_SPREAD].mean()) - float(window[:half, COL_SPREAD].mean()))
    return np.array(feats, dtype=np.float32)

N_FEATURES = 20

def compute_stats(files):
    s, sq, n = np.zeros(N_RAW, np.float64), np.zeros(N_RAW, np.float64), 0
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            ev = (data["features"] if "features" in data else data["events"]).astype(np.float64)
            s += ev.sum(0); sq += (ev**2).sum(0); n += len(ev)
        except: pass
    if n == 0: return {"mean": np.zeros(N_RAW, np.float32), "std": np.ones(N_RAW, np.float32)}
    mean = (s/n).astype(np.float32); std = np.sqrt(np.maximum(sq/n-(s/n)**2, 1e-8)).astype(np.float32)
    return {"mean": mean, "std": std}

def build_xy(files, stats, window, stride):
    """Vectorized — ~100x faster."""
    mean, std = stats["mean"], stats["std"]
    all_X, all_y, chunk = [], [], 50_000
    ci_arr = np.arange(window, dtype=np.int32)
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            ev = (d["features"] if "features" in d else d["events"]).astype(np.float32)
            lb = d["labels_10s"].astype(np.float32)
        except Exception:
            continue
        en = (ev - mean) / (std + 1e-8)
        ne = len(en)
        if ne < window: continue
        starts = np.arange(0, ne - window + 1, stride, dtype=np.int32)
        lidxs  = starts + window - 1
        valid  = (lidxs < len(lb)) & ~np.isnan(lb[lidxs])
        starts, lidxs = starts[valid], lidxs[valid]
        if len(starts) == 0: continue
        for ci in range(0, len(starts), chunk):
            s = starts[ci:ci+chunk]; li = lidxs[ci:ci+chunk]
            all_X.append(compute_features_batch(en[s[:, None] + ci_arr[None, :]]))
            all_y.append(lb[li])
    if not all_X:
        return np.empty((0, N_FEATURES), dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(all_X), np.concatenate(all_y)


def ic(preds, labels):
    mask = ~(np.isnan(preds)|np.isnan(labels))
    if mask.sum() < 20: return float("nan")
    return float(scipy.stats.spearmanr(preds[mask], labels[mask])[0])

def run_wf(files, output_dir, run_name):
    n = len(files); folds = []
    test_start = TRAIN_DAYS
    while test_start + TEST_DAYS <= n:
        train_idx = list(range(test_start-TRAIN_DAYS, test_start))
        test_idx  = list(range(test_start, min(test_start+TEST_DAYS, n)))
        folds.append((len(folds), train_idx, test_idx)); test_start += TEST_DAYS
    if not folds: log.error(f"Not enough files ({n})"); return
    log.info(f"Folds: {len(folds)} | Train: {TRAIN_DAYS}d sliding | Test: {TEST_DAYS}d each")
    log.info(f"Config: W={WINDOW} S={STRIDE} n_est={N_EST} lr={LGBM_LR} leaves={NUM_LEAVES}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mlrun = None
    if MLFLOW_OK:
        mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(MLFLOW_EXP)
        mlrun = mlflow.start_run(run_name=run_name)
        mlflow.log_params({"window":WINDOW,"stride":STRIDE,"train_days":TRAIN_DAYS,"test_days":TEST_DAYS,"n_folds":len(folds),"n_files":n,"n_estimators":N_EST,"lr":LGBM_LR,"num_leaves":NUM_LEAVES,"min_child_samples":MIN_CHILD,"subsample":SUBSAMPLE,"node":socket.gethostname()})
    all_preds, all_labels = [], []
    feat_imp = np.zeros(N_FEATURES)
    for fold_idx, train_idx, test_idx in folds:
        t_files=[files[i] for i in train_idx]; oot_files=[files[i] for i in test_idx]
        log.info(f"\n{'='*60}\nFOLD {fold_idx:02d} | Train: {t_files[0].name}..{t_files[-1].name} | Test: {oot_files[0].name}..{oot_files[-1].name}")
        stats = compute_stats(t_files)
        t0=time.time(); X_tr,y_tr=build_xy(t_files,stats,WINDOW,STRIDE); X_ot,y_ot=build_xy(oot_files,stats,WINDOW,STRIDE)
        log.info(f"  Features built in {time.time()-t0:.1f}s | Train: {len(X_tr):,} | OOT: {len(X_ot):,}")
        if len(X_tr)==0 or len(X_ot)==0: log.warning(f"  Fold {fold_idx}: empty, skipping"); continue
        tr_mask=~np.isnan(y_tr); X_tr,y_tr=X_tr[tr_mask],y_tr[tr_mask]
        ot_mask=~np.isnan(y_ot); X_ot,y_ot=X_ot[ot_mask],y_ot[ot_mask]
        params={"objective":"regression","metric":"rmse","n_estimators":N_EST,"learning_rate":LGBM_LR,"num_leaves":NUM_LEAVES,"min_child_samples":MIN_CHILD,"subsample":SUBSAMPLE,"subsample_freq":1,"colsample_bytree":0.8,"reg_alpha":0.1,"reg_lambda":1.0,"n_jobs":N_JOBS,"verbose":-1,"random_state":42+fold_idx}
        t1=time.time(); model=lgb.LGBMRegressor(**params)
        model.fit(X_tr,y_tr,eval_set=[(X_ot,y_ot)],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(period=50)])
        fold_preds=model.predict(X_ot); fold_ic=ic(fold_preds,y_ot); train_ic=ic(model.predict(X_tr),y_tr)
        log.info(f"  {time.time()-t1:.1f}s | best_iter={model.best_iteration_} | TrainIC={train_ic:.4f} | OOT IC_10s={fold_ic:.4f}")
        feat_imp += model.feature_importances_
        if MLFLOW_OK and mlrun:
            mlflow.log_metrics({f"fold{fold_idx:02d}_oot_ic":fold_ic,f"fold{fold_idx:02d}_train_ic":train_ic,f"fold{fold_idx:02d}_best_iter":float(model.best_iteration_ or N_EST)},step=fold_idx)
        np.savez(output_dir/f"fold{fold_idx:02d}_preds.npz",preds=fold_preds,labels=y_ot)
        model.booster_.save_model(str(output_dir/f"fold{fold_idx:02d}_model.txt"))
        all_preds.append(fold_preds); all_labels.append(y_ot)
    if all_preds:
        concat_p=np.concatenate(all_preds); concat_l=np.concatenate(all_labels); concat_ic=ic(concat_p,concat_l)
        log.info(f"\n{'='*60}\nCONCAT IC_10s = {concat_ic:.4f}  ({len(concat_p):,} samples)")
        feat_names=([f"raw_mean_{i}" for i in range(N_RAW)]+[f"raw_std_{i}" for i in range(N_RAW)]+["ofi","cancel_asym","event_density","price_mom","qty_pressure","near_poc","cum_delta","spread_trend"])
        for rank,fi in enumerate(np.argsort(feat_imp)[::-1][:10]): log.info(f"  {rank+1:>2}. {feat_names[fi]:<25} {feat_imp[fi]:.0f}")
        if MLFLOW_OK and mlrun: mlflow.log_metric("concat_ic_10s",concat_ic); mlflow.log_params({"concat_ic_10s":concat_ic})
    if MLFLOW_OK and mlrun: mlflow.end_run()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default="/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/vol_lgbm"); ap.add_argument("--data-dir",default=str(DATA_DIR)); ap.add_argument("--run-name",default=None); args=ap.parse_args()
    output_dir=Path(args.output_dir); data_dir=Path(args.data_dir)
    files=sorted(data_dir.glob("*.npz"))
    if not files: log.error(f"No NPZ files in {data_dir}"); sys.exit(1)
    log.info(f"Found {len(files)} files: {files[0].name} .. {files[-1].name}")
    valid=[]
    for f in files:
        try:
            d=np.load(f,allow_pickle=True)
            if "labels_10s" in d and not np.all(np.isnan(d["labels_10s"])): valid.append(f)
        except: pass
    log.info(f"Valid: {len(valid)}"); files=valid
    run_name=args.run_name or f"VolLGBM_W{WINDOW}_{time.strftime('%H%M')}"
    run_wf(files, output_dir, run_name)

if __name__=="__main__": main()
