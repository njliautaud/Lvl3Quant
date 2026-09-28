#!/usr/bin/env python3
"""
Mamba + LGBM DA Fusion Analysis
================================
Aligns predictions from Mamba (event-level, stride=500) and LGBM DA (event-level, stride=250)
on overlapping OOT dates (Mar 2-3, 2026).

Tests simple ensemble methods before building full MLP stacker.
No leakage: all alignment is on OOT predictions only.
"""

import numpy as np
from pathlib import Path
from scipy import stats
import json
import os
import sys
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
MAMBA_DIR = BASE / "output" / "mamba_v4_sliding60d_raw6_v2"
LGBM_DIR = BASE / "alpha_discovery" / "deep_models" / "results" / "lgbm_da_classifier"
EVENT_DIR = BASE / "data" / "processed" / "mbo_events"
OUTPUT_DIR = BASE / "alpha_discovery" / "results" / "fusion_mamba_lgbm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
MAMBA_WINDOW = 1000
MAMBA_STRIDE = 500
LGBM_WINDOW = 500
LGBM_STRIDE = 250

# LGBM Fold 27: OOT dates Feb 27, Mar 2, 3, 4 (Feb 28 missing)
# Mamba Fold 0: Mar 2, Fold 1: Mar 3
LGBM_FOLD = 27
LGBM_OOT_DATES = ["20260227", "20260302", "20260303", "20260304"]
MAMBA_FOLDS = {
    "20260302": 0,
    "20260303": 1,
}


def load_mamba_fold(fold_idx):
    """Load Mamba OOT predictions for a fold."""
    path = MAMBA_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not path.exists():
        print(f"  [WARN] Mamba fold {fold_idx} not found: {path}")
        return None
    d = np.load(path)
    return {
        "predictions": d["predictions"],  # (N, 3) for 1s/5s/10s
        "labels": d["labels"],            # (N, 3)
        "embeddings": d["embeddings"],    # (N, 128)
    }


def load_lgbm_fold(fold_idx):
    """Load LGBM DA predictions for a fold."""
    path = LGBM_DIR / f"fold{fold_idx:02d}_preds.npz"
    if not path.exists():
        print(f"  [WARN] LGBM fold {fold_idx} not found: {path}")
        return None
    d = np.load(path)
    return {
        "probs": d["probs"],          # (N,) probability of UP
        "preds": d["preds"],          # (N,) predicted direction
        "labels": d["labels"],        # (N,) actual direction
        "confidence": d["confidence"],  # (N,) abs(prob - 0.5)
    }


def compute_lgbm_day_offsets(oot_dates):
    """Compute sample offsets for each OOT date in the LGBM fold.

    LGBM processes dates sequentially with window=500, stride=250.
    Returns dict of {date: (start_idx, n_samples)}.
    """
    offsets = {}
    cumulative = 0
    for dt in oot_dates:
        path = EVENT_DIR / f"{dt}_mbo_events.npz"
        if not path.exists():
            print(f"  [SKIP] {dt} — file not found")
            continue
        d = np.load(path)
        n_events = d["events"].shape[0]

        # Check for NaN labels (LGBM skips all-NaN files)
        labels = d.get("labels_10s", None)
        if labels is not None and np.all(np.isnan(labels)):
            print(f"  [SKIP] {dt} — all NaN labels")
            continue

        n_samples = max(0, (n_events - LGBM_WINDOW) // LGBM_STRIDE + 1)
        offsets[dt] = (cumulative, n_samples)
        cumulative += n_samples
        print(f"  {dt}: {n_events:,} events → {n_samples:,} LGBM samples (offset={cumulative - n_samples})")

    return offsets, cumulative


def align_predictions(mamba_data, lgbm_data, lgbm_day_offset, lgbm_day_samples, date_str):
    """Align Mamba and LGBM predictions by event position within a day.

    Mamba sample j predicts at event index: j * MAMBA_STRIDE + MAMBA_WINDOW
    LGBM sample i predicts at event index:  i * LGBM_STRIDE + LGBM_WINDOW

    Alignment: j*500 + 1000 = i*250 + 500 → i = 2j + 2
    """
    n_mamba = mamba_data["predictions"].shape[0]

    # For each Mamba sample, find corresponding LGBM sample
    mamba_indices = []
    lgbm_indices = []

    for j in range(n_mamba):
        lgbm_i = 2 * j + 2  # Alignment formula
        if 0 <= lgbm_i < lgbm_day_samples:
            mamba_indices.append(j)
            lgbm_indices.append(lgbm_day_offset + lgbm_i)

    mamba_indices = np.array(mamba_indices)
    lgbm_indices = np.array(lgbm_indices)

    print(f"  {date_str}: Aligned {len(mamba_indices):,} / {n_mamba:,} Mamba samples "
          f"to LGBM indices [{lgbm_indices[0]}..{lgbm_indices[-1]}]")

    return mamba_indices, lgbm_indices


def compute_metrics(preds, labels, tag=""):
    """Compute IC, DA, MagCorr at confidence tiers."""
    valid = ~np.isnan(preds) & ~np.isnan(labels)
    preds = preds[valid]
    labels = labels[valid]
    n = len(preds)

    if n < 100:
        return {"n": n, "error": "too few samples"}

    ic = np.corrcoef(preds, labels)[0, 1] if np.std(preds) > 0 else 0.0
    da = np.mean(np.sign(preds) == np.sign(labels))
    mag_corr = np.corrcoef(np.abs(preds), np.abs(labels))[0, 1] if np.std(np.abs(preds)) > 0 else 0.0

    results = {
        "All": {"IC": float(ic), "DA": float(da), "MagCorr": float(mag_corr), "N": n}
    }

    # Confidence tiers
    abs_preds = np.abs(preds)
    for pct, label in [(50, "Top50"), (25, "Top25"), (10, "Top10"), (5, "Top5")]:
        threshold = np.percentile(abs_preds, 100 - pct)
        mask = abs_preds >= threshold
        if mask.sum() < 50:
            continue
        p_tier = preds[mask]
        l_tier = labels[mask]
        tier_ic = np.corrcoef(p_tier, l_tier)[0, 1] if np.std(p_tier) > 0 else 0.0
        tier_da = np.mean(np.sign(p_tier) == np.sign(l_tier))
        tier_mc = np.corrcoef(np.abs(p_tier), np.abs(l_tier))[0, 1] if np.std(np.abs(p_tier)) > 0 else 0.0
        results[label] = {"IC": float(tier_ic), "DA": float(tier_da), "MagCorr": float(tier_mc), "N": int(mask.sum())}

    return results


def print_metrics_table(results, title):
    """Pretty-print metrics table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Tier':<8} {'IC':>8} {'DA':>8} {'MagCorr':>8} {'N':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for tier in ["All", "Top50", "Top25", "Top10", "Top5"]:
        if tier in results:
            r = results[tier]
            print(f"{tier:<8} {r['IC']:>8.4f} {r['DA']:>7.1%} {r['MagCorr']:>8.4f} {r['N']:>10,}")


def main():
    print(f"{'='*70}")
    print(f"  MAMBA + LGBM DA FUSION ANALYSIS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # ── Step 1: Load LGBM fold 27 ──────────────────────────────────────────
    print(f"\n[1] Loading LGBM DA fold {LGBM_FOLD}...")
    lgbm = load_lgbm_fold(LGBM_FOLD)
    if lgbm is None:
        print("FATAL: Cannot load LGBM predictions")
        return
    print(f"  Loaded {lgbm['probs'].shape[0]:,} total samples")

    # Compute per-day offsets
    print(f"\n[2] Computing LGBM per-day sample offsets...")
    day_offsets, total_expected = compute_lgbm_day_offsets(LGBM_OOT_DATES)
    print(f"  Expected total: {total_expected:,}, Actual: {lgbm['probs'].shape[0]:,}")

    # ── Step 2: Load and align Mamba predictions ───────────────────────────
    print(f"\n[3] Aligning Mamba + LGBM predictions...")

    all_mamba_preds = []  # (N, 3) multi-horizon
    all_lgbm_probs = []   # (N,) UP probability
    all_lgbm_conf = []    # (N,) confidence
    all_labels_10s = []   # (N,) 10s labels from Mamba
    all_mamba_emb = []    # (N, 128) embeddings

    for date_str, mamba_fold_idx in MAMBA_FOLDS.items():
        if date_str not in day_offsets:
            print(f"  [SKIP] {date_str} — not in LGBM fold {LGBM_FOLD}")
            continue

        mamba = load_mamba_fold(mamba_fold_idx)
        if mamba is None:
            continue

        lgbm_offset, lgbm_n = day_offsets[date_str]
        mamba_idx, lgbm_idx = align_predictions(mamba, lgbm, lgbm_offset, lgbm_n, date_str)

        # Validate LGBM indices are in range
        valid_mask = lgbm_idx < lgbm["probs"].shape[0]
        mamba_idx = mamba_idx[valid_mask]
        lgbm_idx = lgbm_idx[valid_mask]

        if len(mamba_idx) == 0:
            print(f"  [WARN] No aligned samples for {date_str}")
            continue

        all_mamba_preds.append(mamba["predictions"][mamba_idx])
        all_lgbm_probs.append(lgbm["probs"][lgbm_idx])
        all_lgbm_conf.append(lgbm["confidence"][lgbm_idx])
        all_labels_10s.append(mamba["labels"][mamba_idx, 2])  # 10s horizon
        all_mamba_emb.append(mamba["embeddings"][mamba_idx])

    if not all_mamba_preds:
        print("FATAL: No aligned predictions found!")
        return

    # Concatenate
    mamba_preds = np.concatenate(all_mamba_preds, axis=0)   # (N, 3)
    lgbm_probs = np.concatenate(all_lgbm_probs, axis=0)     # (N,)
    lgbm_conf = np.concatenate(all_lgbm_conf, axis=0)       # (N,)
    labels_10s = np.concatenate(all_labels_10s, axis=0)      # (N,)
    mamba_emb = np.concatenate(all_mamba_emb, axis=0)        # (N, 128)

    N = mamba_preds.shape[0]
    print(f"\n[4] Total aligned samples: {N:,}")

    # ── Step 3: Individual model metrics ───────────────────────────────────
    mamba_10s = mamba_preds[:, 2]  # 10s predictions

    # Convert LGBM prob to continuous signal: prob - 0.5 (centered)
    lgbm_signal = lgbm_probs - 0.5  # positive = UP, negative = DOWN

    print_metrics_table(
        compute_metrics(mamba_10s, labels_10s),
        "MAMBA (10s) — Individual"
    )

    print_metrics_table(
        compute_metrics(lgbm_signal, labels_10s),
        "LGBM DA (signal) — Individual"
    )

    # ── Step 4: Ensemble methods ───────────────────────────────────────────

    # Normalize both to z-scores for fair combination
    mamba_z = (mamba_10s - np.mean(mamba_10s)) / (np.std(mamba_10s) + 1e-10)
    lgbm_z = (lgbm_signal - np.mean(lgbm_signal)) / (np.std(lgbm_signal) + 1e-10)

    # Method 1: Simple average
    avg_signal = (mamba_z + lgbm_z) / 2
    print_metrics_table(
        compute_metrics(avg_signal, labels_10s),
        "ENSEMBLE: Simple Average (Mamba_z + LGBM_z) / 2"
    )

    # Method 2: Rank average
    mamba_rank = stats.rankdata(mamba_10s) / N
    lgbm_rank = stats.rankdata(lgbm_signal) / N
    rank_avg = (mamba_rank + lgbm_rank) / 2 - 0.5  # Center at 0
    print_metrics_table(
        compute_metrics(rank_avg, labels_10s),
        "ENSEMBLE: Rank Average"
    )

    # Method 3: Mamba-weighted by LGBM confidence
    # Use Mamba magnitude but filter by LGBM direction agreement
    mamba_dir = np.sign(mamba_10s)
    lgbm_dir = np.sign(lgbm_signal)
    agreement = mamba_dir == lgbm_dir  # Both agree on direction

    # Confluence: only keep signals where both models agree
    confluence_signal = np.where(agreement, mamba_10s, 0.0)
    print_metrics_table(
        compute_metrics(confluence_signal, labels_10s),
        "CONFLUENCE: Mamba signal where direction agrees with LGBM"
    )

    # Method 4: Confidence-weighted
    # Weight Mamba more when LGBM is confident
    lgbm_weight = lgbm_conf / (np.mean(lgbm_conf) + 1e-10)  # Normalize
    conf_weighted = mamba_z * (1 + lgbm_weight * lgbm_z.clip(-2, 2))
    print_metrics_table(
        compute_metrics(conf_weighted, labels_10s),
        "ENSEMBLE: Confidence-Weighted (Mamba * LGBM_confidence)"
    )

    # ── Step 5: Confluence statistics ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CONFLUENCE STATISTICS")
    print(f"{'='*70}")
    n_agree = agreement.sum()
    n_disagree = (~agreement).sum()
    print(f"Direction agreement: {n_agree:,} ({n_agree/N:.1%})")
    print(f"Direction disagreement: {n_disagree:,} ({n_disagree/N:.1%})")

    # When they agree, what's the DA?
    agree_labels = labels_10s[agreement]
    agree_preds = mamba_10s[agreement]
    if len(agree_labels) > 0:
        agree_da = np.mean(np.sign(agree_preds) == np.sign(agree_labels))
        agree_ic = np.corrcoef(agree_preds, agree_labels)[0, 1]
        print(f"When models AGREE:    DA={agree_da:.1%}, IC={agree_ic:.4f}, N={len(agree_labels):,}")

    # When they disagree
    disagree_labels = labels_10s[~agreement]
    disagree_preds = mamba_10s[~agreement]
    if len(disagree_labels) > 0:
        disagree_da = np.mean(np.sign(disagree_preds) == np.sign(disagree_labels))
        disagree_ic = np.corrcoef(disagree_preds, disagree_labels)[0, 1]
        print(f"When models DISAGREE: DA={disagree_da:.1%}, IC={disagree_ic:.4f}, N={len(disagree_labels):,}")

    # ── Step 6: Save results ───────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_aligned_samples": int(N),
        "overlap_dates": list(MAMBA_FOLDS.keys()),
        "individual": {
            "mamba_10s": compute_metrics(mamba_10s, labels_10s),
            "lgbm_da": compute_metrics(lgbm_signal, labels_10s),
        },
        "ensemble": {
            "simple_average": compute_metrics(avg_signal, labels_10s),
            "rank_average": compute_metrics(rank_avg, labels_10s),
            "confluence": compute_metrics(confluence_signal, labels_10s),
            "confidence_weighted": compute_metrics(conf_weighted, labels_10s),
        },
        "confluence_stats": {
            "agreement_rate": float(n_agree / N),
            "agree_da": float(agree_da) if n_agree > 0 else None,
            "agree_ic": float(agree_ic) if n_agree > 0 else None,
            "disagree_da": float(disagree_da) if n_disagree > 0 else None,
            "disagree_ic": float(disagree_ic) if n_disagree > 0 else None,
        }
    }

    out_json = OUTPUT_DIR / "fusion_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {out_json}")

    # Save aligned predictions for further analysis
    out_npz = OUTPUT_DIR / "aligned_predictions.npz"
    np.savez_compressed(out_npz,
        mamba_preds=mamba_preds,
        lgbm_probs=lgbm_probs,
        lgbm_confidence=lgbm_conf,
        labels_10s=labels_10s,
        mamba_embeddings=mamba_emb,
        ensemble_avg=avg_signal,
        ensemble_rank=rank_avg,
        confluence=confluence_signal,
    )
    print(f"[SAVED] {out_npz}")

    print(f"\n{'='*70}")
    print(f"  FUSION ANALYSIS COMPLETE — {N:,} aligned samples")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
