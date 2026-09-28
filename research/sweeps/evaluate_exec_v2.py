#!/usr/bin/env python3
"""Evaluate supervised_exec_v2 OOT predictions across 8 folds."""
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/supervised_exec_v2")
HORIZONS = ["1s", "5s", "10s"]

# Collect per-horizon
concat_pnl_true = {h: [] for h in HORIZONS}
concat_pnl_pred = {h: [] for h in HORIZONS}
concat_cls_true = {h: [] for h in HORIZONS}
concat_cls_pred = {h: [] for h in HORIZONS}

for fold_idx in range(8):
    fold_dir = OUTPUT_DIR / f"fold_{fold_idx:02d}"
    pred_file = fold_dir / "oot_predictions.npz"
    if not pred_file.exists():
        print(f"  fold_{fold_idx:02d}: MISSING")
        continue
    
    data = np.load(str(pred_file), allow_pickle=True)
    pred_pnl = data['pred_pnl']        # (N, 3)
    pred_prob = data['pred_prob']       # (N, 3) 
    actual_pnl = data['actual_pnl']    # (N, 3)
    actual_prof = data['actual_profitable']  # (N, 3)
    
    n = len(pred_pnl)
    
    # Try to get eval dates
    try:
        eval_dates = data['eval_dates']
        print(f"\n=== fold_{fold_idx:02d} === N={n}, eval_dates={eval_dates}")
    except:
        print(f"\n=== fold_{fold_idx:02d} === N={n}")
    
    for hi, h in enumerate(HORIZONS):
        yt = actual_pnl[:, hi]
        yp = pred_pnl[:, hi]
        valid = np.isfinite(yt) & np.isfinite(yp)
        if valid.sum() < 10:
            continue
        
        spear, pval = spearmanr(yt[valid], yp[valid])
        
        yt_cls = actual_prof[:, hi]
        yp_cls = pred_prob[:, hi]
        valid_cls = np.isfinite(yt_cls) & np.isfinite(yp_cls)
        auc = roc_auc_score(yt_cls[valid_cls], yp_cls[valid_cls]) if len(np.unique(yt_cls[valid_cls])) > 1 else float('nan')
        
        print(f"  {h}: Spearman={spear:.4f}, AUC={auc:.4f}")
        
        concat_pnl_true[h].append(yt[valid])
        concat_pnl_pred[h].append(yp[valid])
        concat_cls_true[h].append(yt_cls[valid_cls])
        concat_cls_pred[h].append(yp_cls[valid_cls])

# CONCAT metrics
print("\n" + "="*60)
print("CONCAT METRICS (all 8 folds combined)")
print("="*60)

for hi, h in enumerate(HORIZONS):
    if not concat_pnl_true[h]:
        continue
    cat_true = np.concatenate(concat_pnl_true[h])
    cat_pred = np.concatenate(concat_pnl_pred[h])
    spear, pval = spearmanr(cat_true, cat_pred)
    pearson = np.corrcoef(cat_true, cat_pred)[0,1]
    
    cat_cls_t = np.concatenate(concat_cls_true[h])
    cat_cls_p = np.concatenate(concat_cls_pred[h])
    auc = roc_auc_score(cat_cls_t, cat_cls_p) if len(np.unique(cat_cls_t)) > 1 else float('nan')
    acc = accuracy_score(cat_cls_t, (cat_cls_p > 0.5).astype(int))
    
    print(f"\n--- {h} horizon ---")
    print(f"  Total samples: {len(cat_true)}")
    print(f"  Spearman: {spear:.4f} (p={pval:.2e})")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Base rate (profitable): {cat_cls_t.mean():.4f}")
    
    # Quintile analysis on regression predictions
    quintiles = np.quantile(cat_pred, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"  Quintile analysis (by predicted pnl):")
    for i in range(5):
        lo, hi_q = quintiles[i], quintiles[i+1]
        mask = (cat_pred >= lo) & (cat_pred < hi_q + (1e-9 if i==4 else 0))
        if mask.sum() > 0:
            avg = cat_true[mask].mean()
            wr = (cat_true[mask] > 0).mean()
            n_m = mask.sum()
            print(f"    Q{i+1} [{lo:+.3f},{hi_q:+.3f}]: N={n_m:>6d}, avg_pnl={avg:+.4f}t, WR={wr:.3f}")
    
    # Top/bottom decile
    p10 = np.percentile(cat_pred, 10)
    p90 = np.percentile(cat_pred, 90)
    
    bot = cat_true[cat_pred <= p10]
    top = cat_true[cat_pred >= p90]
    print(f"  Bottom decile (short signals): N={len(bot)}, avg_pnl={bot.mean():+.4f}t, WR_short={(bot < 0).mean():.3f}")
    print(f"  Top decile (long signals):     N={len(top)}, avg_pnl={top.mean():+.4f}t, WR_long={(top > 0).mean():.3f}")
    
    # Classification calibration
    print(f"  Probability calibration (classification head):")
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    for bi in range(len(bins)-1):
        mask = (cat_cls_p >= bins[bi]) & (cat_cls_p < bins[bi+1])
        if mask.sum() > 0:
            actual_wr = cat_cls_t[mask].mean()
            print(f"    p=[{bins[bi]:.1f},{bins[bi+1]:.1f}): N={mask.sum():>7d}, actual_WR={actual_wr:.3f}")

# VERDICT
print("\n" + "="*60)
print("VERDICT")
print("="*60)
for h in HORIZONS:
    if concat_pnl_true[h]:
        cat_true = np.concatenate(concat_pnl_true[h])
        cat_pred = np.concatenate(concat_pnl_pred[h])
        spear, _ = spearmanr(cat_true, cat_pred)
        cat_cls_t = np.concatenate(concat_cls_true[h])
        cat_cls_p = np.concatenate(concat_cls_pred[h])
        auc = roc_auc_score(cat_cls_t, cat_cls_p) if len(np.unique(cat_cls_t)) > 1 else float('nan')
        
        verdict = "STRONG" if spear > 0.15 else ("PROMISING" if spear > 0.08 else ("MARGINAL" if spear > 0.03 else "WEAK"))
        print(f"  {h}: Spearman={spear:.4f}, AUC={auc:.4f} => {verdict}")
