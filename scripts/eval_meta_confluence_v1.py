#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

pred_dir = Path("/home/nick/Lvl3Quant/output/razer_meta_confluence")
label_file = Path("/home/nick/Lvl3Quant/output/razer_classifier/class_features.parquet")
output_file = Path("/home/nick/Lvl3Quant/output/razer_meta_confluence/eval_summary_v1.csv")

print("[EVAL] Loading labels...", flush=True)
df_labels = pd.read_parquet(label_file)
df_labels['date'] = pd.to_datetime(df_labels['date']).dt.strftime('%Y%m%d')

dates = sorted(set(df_labels['date'].values))
print(f"[EVAL] Found {len(dates)} test dates: {dates[0]} to {dates[-1]}", flush=True)

n_pos = (df_labels['y_profitable_trigger'] == 1).sum()
n_total = len(df_labels)
base_rate = n_pos / n_total
print(f"[EVAL] Base rate: {base_rate:.4f} ({n_pos}/{n_total})", flush=True)

results = []
per_date_rows = []

for model_name in ['mlp', 'xgb']:
    print(f"\n[EVAL] === {model_name.upper()} ===", flush=True)
    auc_scores = []
    prec_1pct = []
    prec_5pct = []
    tp_1pct = []
    tp_5pct = []
    
    for date in dates:
        pred_file = pred_dir / f"meta_confl_preds_{model_name}_{date}.npz"
        if not pred_file.exists():
            continue
        
        data = np.load(pred_file)
        y_pred = data['prob'].flatten()
        
        df_date = df_labels[df_labels['date'] == date].copy()
        y_true = df_date['y_profitable_trigger'].values
        
        if len(y_pred) != len(y_true):
            print(f"[EVAL] WARN {date} {model_name}: len mismatch", flush=True)
            continue
        
        auc = roc_auc_score(y_true, y_pred)
        auc_scores.append(auc)
        
        thresh_1pct = np.percentile(y_pred, 99)
        mask_1pct = y_pred >= thresh_1pct
        p_1pct = y_true[mask_1pct].mean() if mask_1pct.sum() > 0 else np.nan
        prec_1pct.append(p_1pct)
        tp_1pct.append(int(y_true[mask_1pct].sum()))
        
        thresh_5pct = np.percentile(y_pred, 95)
        mask_5pct = y_pred >= thresh_5pct
        p_5pct = y_true[mask_5pct].mean() if mask_5pct.sum() > 0 else np.nan
        prec_5pct.append(p_5pct)
        tp_5pct.append(int(y_true[mask_5pct].sum()))
        
        uplift_1pct = (p_1pct / base_rate - 1) if not np.isnan(p_1pct) else np.nan
        uplift_5pct = (p_5pct / base_rate - 1) if not np.isnan(p_5pct) else np.nan
        
        per_date_rows.append({
            'date': date,
            'model': model_name,
            'auc': auc,
            'prec_top1pct': p_1pct,
            'uplift_1pct': uplift_1pct,
            'prec_top5pct': p_5pct,
            'uplift_5pct': uplift_5pct,
            'tp_1pct': int(tp_1pct[-1]),
            'tp_5pct': int(tp_5pct[-1]),
            'n_samples': len(y_true)
        })
        
        print(f"[EVAL] {date} {model_name}: AUC={auc:.4f}, top-1% prec={p_1pct:.4f} ({uplift_1pct:+.1%}), top-5% prec={p_5pct:.4f} ({uplift_5pct:+.1%})", flush=True)
    
    mean_auc = np.mean(auc_scores)
    mean_p1pct = np.nanmean(prec_1pct)
    mean_p5pct = np.nanmean(prec_5pct)
    mean_uplift_1pct = (mean_p1pct / base_rate - 1)
    mean_uplift_5pct = (mean_p5pct / base_rate - 1)
    
    print(f"[EVAL] {model_name.upper()} AGGREGATE across {len(auc_scores)} dates:", flush=True)
    print(f"       AUC={mean_auc:.4f}", flush=True)
    print(f"       top-1% prec={mean_p1pct:.4f} ({mean_uplift_1pct:+.1%} uplift vs base)", flush=True)
    print(f"       top-5% prec={mean_p5pct:.4f} ({mean_uplift_5pct:+.1%} uplift vs base)", flush=True)
    
    results.append({
        'model': model_name,
        'mean_auc': mean_auc,
        'mean_prec_top1pct': mean_p1pct,
        'mean_uplift_1pct': mean_uplift_1pct,
        'mean_prec_top5pct': mean_p5pct,
        'mean_uplift_5pct': mean_uplift_5pct,
        'n_dates': len(auc_scores)
    })

df_summary = pd.DataFrame(results)
df_summary.to_csv(output_file, index=False)
print(f"\n[EVAL] Summary CSV: {output_file}", flush=True)

df_per_date = pd.DataFrame(per_date_rows)
per_date_file = Path("/home/nick/Lvl3Quant/output/razer_meta_confluence/eval_per_date_v1.csv")
df_per_date.to_csv(per_date_file, index=False)
print(f"[EVAL] Per-date CSV: {per_date_file}", flush=True)
