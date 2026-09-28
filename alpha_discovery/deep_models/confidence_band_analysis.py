#!/usr/bin/env python3
"""
Confidence-Banded Performance Analysis
=======================================
Computes IC, DA, MagCorr at confidence tiers (All/Top50%/Top25%/Top10%/Top5%/Top1%)
with 95% bootstrap CIs, for all available models and horizons (1s/5s/10s).
"""

import numpy as np
import os
import sys
from pathlib import Path

RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results")
TIERS = [('All', 1.0), ('Top50%', 0.50), ('Top25%', 0.25), ('Top10%', 0.10), ('Top5%', 0.05), ('Top1%', 0.01)]
N_BOOTSTRAP = 1000
SEED = 42

# ─── Metric functions ───────────────────────────────────────────────────────

def ic(preds, labels):
    """Pearson correlation (Information Coefficient)."""
    if len(preds) < 5:
        return np.nan
    p, l = preds - preds.mean(), labels - labels.mean()
    denom = np.sqrt((p**2).sum() * (l**2).sum())
    if denom == 0:
        return 0.0
    return (p * l).sum() / denom

def da(preds, labels):
    """Directional accuracy: fraction where sign(pred) == sign(label)."""
    mask = (preds != 0) & (labels != 0)
    if mask.sum() < 5:
        return np.nan
    return (np.sign(preds[mask]) == np.sign(labels[mask])).mean()

def magcorr(preds, labels):
    """Magnitude correlation: Pearson corr of |pred| vs |actual|."""
    ap, al = np.abs(preds), np.abs(labels)
    return ic(ap, al)

def bootstrap_ci(preds, labels, metric_fn, n_boot=N_BOOTSTRAP, ci=0.025):
    """Bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(SEED)
    n = len(preds)
    if n < 10:
        val = metric_fn(preds, labels)
        return val, np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        vals[i] = metric_fn(preds[idx], labels[idx])
    point = metric_fn(preds, labels)
    lo, hi = np.nanpercentile(vals, [ci*100, (1-ci)*100])
    return point, lo, hi

def filter_to_tier(preds, labels, frac):
    """Filter to top `frac` fraction by |prediction| magnitude."""
    if frac >= 1.0:
        return preds, labels
    n_keep = max(5, int(len(preds) * frac))
    abs_p = np.abs(preds)
    threshold = np.sort(abs_p)[-n_keep]
    mask = abs_p >= threshold
    # If too many at threshold, subsample
    if mask.sum() > n_keep * 1.1:
        idx = np.where(mask)[0][:n_keep]
        return preds[idx], labels[idx]
    return preds[mask], labels[mask]


# ─── Model loaders ──────────────────────────────────────────────────────────

def load_lgbm_da(model_dir):
    """Load LGBM DA classifier predictions (binary).
    Returns dict: {horizon: (preds_continuous, labels_continuous, n_samples)}
    For classifiers: preds = prob - 0.5 (centered), labels = label*2-1 (to +/-1).
    """
    fold_files = sorted(model_dir.glob("fold*_preds.npz"))
    if not fold_files:
        return {}
    all_probs, all_labels = [], []
    for f in fold_files:
        d = np.load(f, allow_pickle=True)
        all_probs.append(d['probs'].astype(np.float64))
        all_labels.append(d['labels'].astype(np.float64))
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    # Convert to continuous for IC/MagCorr: center probs at 0, labels to +/-1
    preds_cont = probs - 0.5  # range [-0.5, 0.5]
    labels_cont = labels * 2 - 1  # 0->-1, 1->+1
    # DA is native since this is a classifier
    return {'10s': (preds_cont, labels_cont, len(preds_cont), probs, labels)}

def load_multi_horizon_concat(model_dir):
    """Load Neptune-style concat predictions with preds_Xs/labels_Xs keys."""
    concat_file = model_dir / "concat_oot_predictions.npz"
    if concat_file.exists():
        d = np.load(concat_file, allow_pickle=True)
        result = {}
        for h in ['1s', '5s', '10s']:
            pk, lk = f'preds_{h}', f'labels_{h}'
            if pk in d and lk in d:
                p = d[pk].astype(np.float64)
                l = d[lk].astype(np.float64)
                result[h] = (p, l, len(p), None, None)
        return result
    return {}

def load_multi_horizon_folds(model_dir):
    """Load Neptune-style fold predictions with (predictions, labels) shape=(N,3)."""
    fold_files = sorted(model_dir.glob("fold_*_oot_predictions.npz"))
    if not fold_files:
        return {}
    all_preds = {h: [] for h in ['1s', '5s', '10s']}
    all_labels = {h: [] for h in ['1s', '5s', '10s']}
    for f in fold_files:
        d = np.load(f, allow_pickle=True)
        if 'predictions' in d and d['predictions'].ndim == 2:
            horizons = list(d.get('horizons', ['1s', '5s', '10s']))
            for i, h in enumerate(horizons):
                if h in all_preds:
                    all_preds[h].append(d['predictions'][:, i].astype(np.float64))
                    all_labels[h].append(d['labels'][:, i].astype(np.float64))
    result = {}
    for h in ['1s', '5s', '10s']:
        if all_preds[h]:
            p = np.concatenate(all_preds[h])
            l = np.concatenate(all_labels[h])
            result[h] = (p, l, len(p), None, None)
    return result


# ─── Analysis ────────────────────────────────────────────────────────────────

def analyze_model(name, data_dict):
    """Run full confidence-banded analysis for one model."""
    print(f"\n{'='*90}")
    print(f"  MODEL: {name}")
    print(f"{'='*90}")

    for horizon in ['1s', '5s', '10s']:
        if horizon not in data_dict:
            continue
        preds, labels, n, probs, raw_labels = data_dict[horizon]

        # Check for degenerate data
        valid = np.isfinite(preds) & np.isfinite(labels)
        preds, labels = preds[valid], labels[valid]
        if probs is not None:
            probs = probs[valid]
            raw_labels = raw_labels[valid]

        print(f"\n  Horizon: {horizon}  |  N = {len(preds):,}")
        print(f"  {'Tier':<10} {'N':>8}  {'IC':>22}  {'DA':>22}  {'MagCorr':>22}")
        print(f"  {'-'*86}")

        for tier_name, tier_frac in TIERS:
            p_t, l_t = filter_to_tier(preds, labels, tier_frac)
            n_t = len(p_t)

            # IC
            ic_val, ic_lo, ic_hi = bootstrap_ci(p_t, l_t, ic)

            # DA - for classifiers use native, for regressors use sign
            if probs is not None:
                # Classifier: filter probs/raw_labels the same way
                pr_t, rl_t = filter_to_tier(preds, labels, tier_frac)
                # But use probs for DA filtering
                prob_t, rawl_t = filter_to_tier(probs - 0.5, raw_labels, tier_frac)
                # DA = (prob > 0.5 matches label==1)
                da_val = ((prob_t > 0) == (rawl_t == 1)).mean()
                # Bootstrap DA
                rng = np.random.RandomState(SEED)
                da_boots = []
                for _ in range(N_BOOTSTRAP):
                    idx = rng.randint(0, len(prob_t), len(prob_t))
                    da_boots.append(((prob_t[idx] > 0) == (rawl_t[idx] == 1)).mean())
                da_lo, da_hi = np.percentile(da_boots, [2.5, 97.5])
            else:
                da_val, da_lo, da_hi = bootstrap_ci(p_t, l_t, da)

            # MagCorr
            mc_val, mc_lo, mc_hi = bootstrap_ci(p_t, l_t, magcorr)

            def fmt(val, lo, hi):
                if np.isnan(val):
                    return f"{'N/A':>22}"
                if np.isnan(lo):
                    return f"{val:>8.4f} {'(no CI)':>13}"
                return f"{val:>8.4f}  [{lo:>7.4f}, {hi:>7.4f}]"

            print(f"  {tier_name:<10} {n_t:>8,}  {fmt(ic_val, ic_lo, ic_hi)}  {fmt(da_val, da_lo, da_hi)}  {fmt(mc_val, mc_lo, mc_hi)}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    models = {}

    # 1. LGBM DA smart_v2
    lgbm_sv2_dir = RESULTS_DIR / "lgbm_da_smart_v2"
    if lgbm_sv2_dir.exists():
        data = load_lgbm_da(lgbm_sv2_dir)
        if data:
            models["LGBM DA smart_v2 (classifier)"] = data

    # 2. LGBM DA classifier (original)
    lgbm_cls_dir = RESULTS_DIR / "lgbm_da_classifier"
    if lgbm_cls_dir.exists():
        data = load_lgbm_da(lgbm_cls_dir)
        if data:
            models["LGBM DA classifier (original)"] = data

    # 3. Event Mamba CUDA (concat available - use that)
    mamba_cuda_dir = RESULTS_DIR / "event_mamba_cuda"
    if mamba_cuda_dir.exists():
        data = load_multi_horizon_concat(mamba_cuda_dir)
        if data:
            models["Event Mamba CUDA"] = data

    # 4. Event Mamba Fast (concat available)
    mamba_fast_dir = RESULTS_DIR / "event_mamba_fast"
    if mamba_fast_dir.exists():
        data = load_multi_horizon_concat(mamba_fast_dir)
        if data:
            models["Event Mamba Fast"] = data

    # 5. Event Mamba CUDA v2 (fold-based, multi-horizon)
    mamba_v2_dir = RESULTS_DIR / "event_mamba_cuda_v2"
    if mamba_v2_dir.exists():
        data = load_multi_horizon_folds(mamba_v2_dir)
        if data:
            models["Event Mamba CUDA v2 (1 fold)"] = data

    # 6. CNN1D Neptune (fold-based, multi-horizon) - bonus for comparison
    cnn1d_dir = RESULTS_DIR / "cnn1d_neptune_20260418_1033"
    if cnn1d_dir.exists():
        data = load_multi_horizon_folds(cnn1d_dir)
        if data:
            models["EventCNN1D (Neptune Apr18)"] = data

    if not models:
        print("ERROR: No model predictions found!")
        sys.exit(1)

    print(f"\n{'#'*90}")
    print(f"  CONFIDENCE-BANDED PERFORMANCE ANALYSIS")
    print(f"  Models found: {len(models)}")
    print(f"  Bootstrap: {N_BOOTSTRAP} resamples, 95% CI")
    print(f"  Tiers: {', '.join(t[0] for t in TIERS)}")
    print(f"{'#'*90}")

    for name, data in models.items():
        analyze_model(name, data)

    # ─── Summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"  SUMMARY: Concat IC by Horizon (All data, with 95% CI)")
    print(f"{'='*90}")
    print(f"  {'Model':<35} {'IC_1s':>18} {'IC_5s':>18} {'IC_10s':>18}")
    print(f"  {'-'*89}")

    for name, data in models.items():
        row = f"  {name:<35}"
        for h in ['1s', '5s', '10s']:
            if h in data:
                p, l, n, _, _ = data[h]
                valid = np.isfinite(p) & np.isfinite(l)
                val, lo, hi = bootstrap_ci(p[valid], l[valid], ic)
                row += f" {val:>6.4f} [{lo:.4f},{hi:.4f}]"
            else:
                row += f" {'N/A':>18}"
        print(row)

    # ─── Top5% IC summary ──────────────────────────────────────────────
    print(f"\n  {'Model':<35} {'Top5% IC_1s':>18} {'Top5% IC_5s':>18} {'Top5% IC_10s':>18}")
    print(f"  {'-'*89}")

    for name, data in models.items():
        row = f"  {name:<35}"
        for h in ['1s', '5s', '10s']:
            if h in data:
                p, l, n, _, _ = data[h]
                valid = np.isfinite(p) & np.isfinite(l)
                p_f, l_f = filter_to_tier(p[valid], l[valid], 0.05)
                val, lo, hi = bootstrap_ci(p_f, l_f, ic)
                row += f" {val:>6.4f} [{lo:.4f},{hi:.4f}]"
            else:
                row += f" {'N/A':>18}"
        print(row)

    print()


if __name__ == "__main__":
    main()
