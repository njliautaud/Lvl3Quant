#!/usr/bin/env python3
"""
Direct First-Passage Binary Classifiers v1
For each (K,S), train XGBoost GPU predicting "TP_K before SL_S within 15s?"
Walk-forward sliding window (60-day train, 1-day OOT).
"""
import os, sys, json, time, warnings, traceback
import numpy as np
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

TICK = 0.25
HORIZON = 150  # 15s at 100ms
TRAIN_WIN = 60
CELLS = [(2,1), (3,1), (4,1), (5,1), (3,2), (4,2), (5,2), (5,4)]

MID_DIR = Path("/home/nick/Lvl3Quant/data/derived/mid_price_bars")
PRED_DIR = Path("/home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/direct_firstpassage_heads_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://localhost:5000"  # MLflow runs locally on Neptune

def discover_dates():
    mid = {f.stem for f in MID_DIR.glob("*.npz")}
    pred = {f.stem.replace("_unfiltered","") for f in PRED_DIR.glob("*_unfiltered.npz")}
    return sorted(mid & pred)

def gen_labels(mid_prices, K, S):
    """First-passage labels using stride_tricks. 1=TP first, 0=SL first."""
    n = len(mid_prices)
    cutoff = n - HORIZON
    labels = np.zeros(n, dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    if cutoff <= 0:
        return labels, valid

    tp_d = K * TICK
    sl_d = S * TICK

    # Process in chunks to limit memory (~50k rows x 150 cols x 4 bytes = 30MB per chunk)
    chunk = 50000
    for start in range(0, cutoff, chunk):
        end = min(start + chunk, cutoff)
        sz = end - start

        itemsize = mid_prices.strides[0]
        future = as_strided(mid_prices[start+1:], shape=(sz, HORIZON), strides=(itemsize, itemsize))
        entry = mid_prices[start:end].reshape(-1, 1)
        changes = (future - entry).astype(np.float32)

        tp_hit = changes >= tp_d
        sl_hit = changes <= -sl_d

        any_tp = tp_hit.any(axis=1)
        any_sl = sl_hit.any(axis=1)

        tp_first = np.where(any_tp, np.argmax(tp_hit, axis=1), HORIZON+1)
        sl_first = np.where(any_sl, np.argmax(sl_hit, axis=1), HORIZON+1)

        valid[start:end] = any_tp | any_sl
        labels[start:end] = (any_tp & (tp_first <= sl_first)).astype(np.int8)

    return labels, valid

def rolling_std(arr, w):
    n = len(arr)
    r = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr.astype(np.float64))
    cs2 = np.cumsum(arr.astype(np.float64)**2)
    cs = np.concatenate([[0], cs])
    cs2 = np.concatenate([[0], cs2])
    s = cs[w:n+1] - cs[:n+1-w]
    s2 = cs2[w:n+1] - cs2[:n+1-w]
    var = np.maximum(s2/w - (s/w)**2, 0)
    r[w-1:] = np.sqrt(var).astype(np.float32)
    return r

def build_features(mid, bid, ask, pred):
    n = len(mid)
    f = {}
    f['pred'] = pred.copy()
    f['pred_abs'] = np.abs(pred)
    f['pred_sq'] = pred**2
    f['spread'] = (ask - bid) / TICK

    for lag, nm in [(1,'r1'),(5,'r5'),(10,'r10'),(50,'r50')]:
        ret = np.zeros(n, dtype=np.float32)
        ret[lag:] = (mid[lag:] - mid[:-lag]) / TICK
        f[nm] = ret

    r1 = f['r1']
    for w, nm in [(10,'v10'),(50,'v50'),(100,'v100')]:
        f[nm] = rolling_std(r1, w)

    sign_ret = np.sign(r1)
    cs = np.cumsum(sign_ret)
    mom = np.zeros(n, dtype=np.float32)
    mom[20:] = (cs[20:] - cs[:-20]) / 20.0
    f['mom20'] = mom

    pd = np.zeros(n, dtype=np.float32)
    pd[5:] = pred[5:] - pred[:-5]
    f['ptrend5'] = pd

    names = sorted(f.keys())
    X = np.column_stack([f[k] for k in names]).astype(np.float32)
    return X, names

def main():
    t0 = time.time()
    print("=" * 70)
    print(f"First-Passage Heads v1 — {datetime.now()}")
    print(f"Cells: {CELLS}")
    print("=" * 70)

    dates = discover_dates()
    print(f"Dates: {len(dates)} ({dates[0]}-{dates[-1]})")

    # Load features
    print("Loading features...")
    cache = {}
    for i, d in enumerate(dates):
        md = np.load(MID_DIR / f"{d}.npz")
        prd = np.load(PRED_DIR / f"{d}_unfiltered.npz")
        mid = md['mid_prices']
        X, fnames = build_features(mid, md['bid_prices'], md['ask_prices'], prd['predictions'])
        cache[d] = {'mid': mid, 'X': X}
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(dates)} loaded")

    print(f"Features ({len(fnames)}): {fnames}")
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Quick label-gen speed test
    lt = time.time()
    tl, tv = gen_labels(cache[dates[0]]['mid'], 2, 1)
    print(f"Label speed test: 1 date in {time.time()-lt:.2f}s (valid={tv.sum():,})")

    # MLflow setup (non-blocking)
    use_mlflow = False
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("direct_firstpassage_heads_v1")
        use_mlflow = True
        print("MLflow connected")
    except Exception as e:
        print(f"MLflow unavailable ({e}), continuing without it")

    all_summaries = []
    parent_run = None

    if use_mlflow:
        parent_run = mlflow.start_run(run_name=f"fp_heads_{datetime.now().strftime('%Y%m%d_%H%M')}")
        mlflow.log_param("cells", str(CELLS))
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("train_window", TRAIN_WIN)
        mlflow.log_param("n_dates", len(dates))
        mlflow.log_param("n_features", len(fnames))

    for K, S in CELLS:
        cell = f"TP{K}_SL{S}"
        cdir = OUT_DIR / cell
        cdir.mkdir(exist_ok=True)
        ct0 = time.time()

        print(f"\n{'='*60}")
        print(f"Cell {cell} (K={K}, S={S})")
        print(f"{'='*60}")

        # Generate labels
        lt = time.time()
        labels = {}; valids = {}
        for d in dates:
            l, v = gen_labels(cache[d]['mid'], K, S)
            labels[d] = l; valids[d] = v

        n_valid = sum(v.sum() for v in valids.values())
        n_pos = sum(labels[d][valids[d]].sum() for d in dates)
        rate = n_pos / n_valid if n_valid > 0 else 0
        print(f"  Labels: {n_valid:,} valid, {n_pos:,} pos ({rate:.3f}), {time.time()-lt:.1f}s")

        # Walk-forward
        results = []
        min_train = 20

        for oot_idx in range(min_train, len(dates)):
            oot_date = dates[oot_idx]
            train_start = max(0, oot_idx - TRAIN_WIN)
            train_dates = dates[train_start:oot_idx]

            # Build train set
            Xt = []; yt = []
            for td in train_dates:
                mask = valids[td].copy()
                mask[:100] = False  # feature warm-up
                if mask.sum() > 0:
                    Xt.append(cache[td]['X'][mask])
                    yt.append(labels[td][mask])

            if not Xt:
                continue
            Xt = np.vstack(Xt); yt = np.concatenate(yt)

            # Build OOT set
            mask_o = valids[oot_date].copy()
            mask_o[:100] = False
            if mask_o.sum() < 100:
                continue
            Xo = cache[oot_date]['X'][mask_o]
            yo = labels[oot_date][mask_o]

            # Train XGBoost GPU
            spw = min((len(yt) - yt.sum()) / max(yt.sum(), 1), 10.0)
            dtrain = xgb.DMatrix(Xt, label=yt)
            doot = xgb.DMatrix(Xo, label=yo)

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'device': 'cuda',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 50,
                'scale_pos_weight': float(spw),
                'verbosity': 0,
            }

            bst = xgb.train(params, dtrain, num_boost_round=300,
                           evals=[(doot, 'oot')], early_stopping_rounds=30,
                           verbose_eval=False)

            yp = bst.predict(doot)
            yb = (yp >= 0.5).astype(int)

            try:
                auc = roc_auc_score(yo, yp)
            except:
                auc = 0.5

            acc = accuracy_score(yo, yb)
            prec = precision_score(yo, yb, zero_division=0)
            rec = recall_score(yo, yb, zero_division=0)

            r = {'date': oot_date, 'auc': float(auc), 'acc': float(acc),
                 'prec': float(prec), 'rec': float(rec),
                 'n_train': len(yt), 'n_oot': len(yo),
                 'n_train_days': len(train_dates),
                 'pos_rate_train': float(yt.mean()),
                 'pos_rate_oot': float(yo.mean()),
                 'best_iter': int(bst.best_iteration) if hasattr(bst, 'best_iteration') else -1}
            results.append(r)

            if oot_idx % 5 == 0 or oot_idx == len(dates)-1:
                print(f"  {oot_date}: AUC={auc:.4f} Acc={acc:.4f} P={prec:.4f} R={rec:.4f} "
                      f"(n_train={len(yt):,} n_oot={len(yo):,} pos={yo.mean():.3f})")

        # Save model + results
        if results:
            bst.save_model(str(cdir / "model.json"))
            with open(cdir / "results.json", "w") as f:
                json.dump(results, f, indent=2)

            aucs = [r['auc'] for r in results]
            summary = {
                'cell': cell, 'K': K, 'S': S,
                'n_folds': len(results),
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'median_auc': float(np.median(aucs)),
                'min_auc': float(np.min(aucs)),
                'max_auc': float(np.max(aucs)),
                'mean_acc': float(np.mean([r['acc'] for r in results])),
                'mean_prec': float(np.mean([r['prec'] for r in results])),
                'mean_rec': float(np.mean([r['rec'] for r in results])),
                'avg_pos_rate': float(np.mean([r['pos_rate_oot'] for r in results])),
                'train_time_s': float(time.time() - ct0),
            }
            with open(cdir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n  {cell}: AUC={summary['mean_auc']:.4f}+/-{summary['std_auc']:.4f} "
                  f"Acc={summary['mean_acc']:.4f} P={summary['mean_prec']:.4f} R={summary['mean_rec']:.4f} "
                  f"({time.time()-ct0:.0f}s)")

            if use_mlflow:
                try:
                    with mlflow.start_run(run_name=cell, nested=True):
                        mlflow.log_param("K", K)
                        mlflow.log_param("S", S)
                        mlflow.log_metric("mean_auc", summary['mean_auc'])
                        mlflow.log_metric("std_auc", summary['std_auc'])
                        mlflow.log_metric("mean_acc", summary['mean_acc'])
                        mlflow.log_metric("mean_prec", summary['mean_prec'])
                        mlflow.log_metric("mean_rec", summary['mean_rec'])
                        mlflow.log_metric("avg_pos_rate", summary['avg_pos_rate'])
                        mlflow.log_artifact(str(cdir / "model.json"))
                        mlflow.log_artifact(str(cdir / "summary.json"))
                except Exception as e:
                    print(f"  MLflow log failed: {e}")

            all_summaries.append(summary)

        del labels, valids

    # Final report
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Cell':<12} {'AUC':>8} {'Std':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'PosR':>8}")
    print("-" * 62)
    for s in all_summaries:
        print(f"{s['cell']:<12} {s['mean_auc']:>8.4f} {s['std_auc']:>8.4f} "
              f"{s['mean_acc']:>8.4f} {s['mean_prec']:>8.4f} "
              f"{s['mean_rec']:>8.4f} {s['avg_pos_rate']:>8.4f}")

    with open(OUT_DIR / "master_summary.json", "w") as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_time_s': total_time,
            'cells': all_summaries,
            'features': fnames,
            'config': {'horizon': HORIZON, 'train_window': TRAIN_WIN, 'tick': TICK,
                       'n_dates': len(dates), 'date_range': [dates[0], dates[-1]]},
        }, f, indent=2)

    if use_mlflow and parent_run:
        try:
            mlflow.log_metric("total_time_s", total_time)
            mlflow.log_artifact(str(OUT_DIR / "master_summary.json"))
            mlflow.end_run()
        except:
            pass

    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Output: {OUT_DIR}")

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
