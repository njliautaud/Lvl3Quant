#!/usr/bin/env python3
"""
Multi-Model Confluence Analysis
================================
Tests whether agreement across multiple CNN-Mamba model versions improves
trade selection compared to any single model alone.

Models:
  1. CNN-Mamba v2 (per-date files)    — pred columns: predictions[:,0/1/2] for 1s/5s/10s
  2. CNN-Mamba v3.3 (uncertainty)     — pred columns: pred_log_ret_{1s,5s,10s}
  3. CNN-Mamba v3.4.2 (fixed MTL)    — pred columns: pred_log_ret_{1s,5s,10s}
  4. PatchTST (per-date files)        — pred columns: predictions[:,0/1/2] for 1s/5s/10s

Cost model (HC #512): 0.376 ticks round-trip commission ONLY. No spread cost.

Usage: python scripts/multi_model_confluence.py
"""

import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
V2_DIR = ROOT / "output" / "cnn_mamba_v2_all_oot"
V33_DIR = ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "oot_47day_perdate"
V342_DIR = ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
PT_DIR = ROOT / "output" / "patchtst_bulk_oot"
OUTPUT_DIR = ROOT / "output"

# ── Cost constant (HC #512) ─────────────────────────────────────────────────
COMMISSION_TICKS = 0.376  # round-trip, NO spread cost

HORIZONS = ["1s", "5s", "10s"]
HORIZON_IDX = {h: i for i, h in enumerate(HORIZONS)}


# ── Data loading ─────────────────────────────────────────────────────────────

def discover_dates():
    """Find dates available in all four models."""
    v2_dates = set()
    for f in os.listdir(V2_DIR):
        if f.endswith("_predictions.npz") and f[0].isdigit():
            v2_dates.add(f[:8])

    v33_dates = set()
    for f in os.listdir(V33_DIR):
        if f.startswith("oot_") and f.endswith(".npz"):
            v33_dates.add(f[4:12])

    v342_dates = set()
    for f in os.listdir(V342_DIR):
        if f.startswith("oot_") and f.endswith(".npz"):
            v342_dates.add(f[4:12])

    pt_dates = set()
    for f in os.listdir(PT_DIR):
        if f.endswith("_predictions.npz") and f[0].isdigit():
            pt_dates.add(f[:8])

    overlap = sorted(v2_dates & v33_dates & v342_dates & pt_dates)
    print(f"Date coverage: V2={len(v2_dates)}, V3.3={len(v33_dates)}, "
          f"V3.4.2={len(v342_dates)}, PatchTST={len(pt_dates)}")
    print(f"Overlapping dates (all 4 models): {len(overlap)}")
    return overlap


def load_date(date_str):
    """Load predictions and labels for one date from all models, aligned by index.

    Returns dict with keys: v2_pred, v33_pred, v342_pred, pt_pred, labels
    Each is shape (N, 3) for the 3 horizons [1s, 5s, 10s].
    N is the aligned (common) length.
    """
    # Load raw files
    v2 = np.load(V2_DIR / f"{date_str}_predictions.npz", allow_pickle=True)
    v33 = np.load(V33_DIR / f"oot_{date_str}.npz", allow_pickle=True)
    v342 = np.load(V342_DIR / f"oot_{date_str}.npz", allow_pickle=True)
    pt = np.load(PT_DIR / f"{date_str}_predictions.npz", allow_pickle=True)

    # Extract predictions (N_model, 3)
    v2_pred = v2["predictions"]  # (N_v2, 3) for 1s/5s/10s
    v2_lbl = v2["labels"]        # (N_v2, 3)

    v33_pred = np.column_stack([v33["pred_log_ret_1s"],
                                 v33["pred_log_ret_5s"],
                                 v33["pred_log_ret_10s"]])
    v33_lbl = np.column_stack([v33["target_log_ret_1s"],
                                v33["target_log_ret_5s"],
                                v33["target_log_ret_10s"]])

    v342_pred = np.column_stack([v342["pred_log_ret_1s"],
                                  v342["pred_log_ret_5s"],
                                  v342["pred_log_ret_10s"]])

    pt_pred = pt["predictions"]  # (N_pt, 3)
    pt_lbl = pt["labels"]        # (N_pt, 3)

    # ── Alignment ────────────────────────────────────────────────────────
    # Different models have different window sizes, so different sample counts.
    # All use stride=250 on the same event stream.
    # Models with larger windows produce fewer samples, starting later.
    # We align using cross-correlation of the 1s labels.
    # Use the model with fewest samples as the anchor (it starts latest).

    model_labels = {
        "v2": v2_lbl,
        "v33": v33_lbl,
        "pt": pt_lbl,
    }
    model_preds = {
        "v2": v2_pred,
        "v33": v33_pred,
        "v342": v342_pred,
        "pt": pt_pred,
    }

    # V3.3 and V3.4.2 are always perfectly aligned (same architecture, same window)
    # Find the model with fewest samples — that's the anchor
    counts = {"v2": len(v2_pred), "v33": len(v33_pred), "v342": len(v342_pred), "pt": len(pt_pred)}
    anchor = min(counts, key=counts.get)
    anchor_n = counts[anchor]

    # Get anchor labels (for 1s horizon)
    if anchor == "v2":
        anchor_lbl_1s = v2_lbl[:, 0]
    elif anchor in ("v33", "v342"):
        anchor_lbl_1s = v33_lbl[:, 0]  # v33 and v342 share same labels
    else:
        anchor_lbl_1s = pt_lbl[:, 0]

    # Find offset for each model relative to anchor
    offsets = {}
    for name in ["v2", "v33", "v342", "pt"]:
        if name == anchor or (name == "v342" and anchor == "v33") or (name == "v33" and anchor == "v342"):
            offsets[name] = 0
            continue

        n_model = counts[name]
        if n_model == anchor_n:
            offsets[name] = 0
            continue

        # The model with more samples starts earlier, so we skip its first (n_model - anchor_n) samples
        # But this is approximate — verify with correlation
        diff = n_model - anchor_n

        # Get model's label array for 1s
        if name == "v2":
            mlbl = v2_lbl[:, 0]
        elif name in ("v33", "v342"):
            mlbl = v33_lbl[:, 0]
        else:
            mlbl = pt_lbl[:, 0]

        # Search around the expected offset for best correlation
        best_corr, best_off = -1, diff
        for trial_off in range(max(0, diff - 5), diff + 6):
            if trial_off >= len(mlbl):
                continue
            seg = mlbl[trial_off: trial_off + anchor_n]
            n_check = min(len(seg), anchor_n)
            if n_check < 1000:
                continue
            m = ~np.isnan(seg[:n_check]) & ~np.isnan(anchor_lbl_1s[:n_check])
            if m.sum() < 500:
                continue
            c = np.corrcoef(seg[:n_check][m], anchor_lbl_1s[:n_check][m])[0, 1]
            if c > best_corr:
                best_corr, best_off = c, trial_off

        offsets[name] = best_off

    # v342 always has the same offset as v33
    if "v342" not in offsets or offsets.get("v342", -1) == -1:
        offsets["v342"] = offsets.get("v33", 0)
    if "v33" not in offsets or offsets.get("v33", -1) == -1:
        offsets["v33"] = offsets.get("v342", 0)

    # Compute common length
    common_n = min(counts[name] - offsets[name] for name in counts)

    # Slice aligned arrays
    result = {
        "v2_pred": v2_pred[offsets["v2"]: offsets["v2"] + common_n],
        "v33_pred": v33_pred[offsets["v33"]: offsets["v33"] + common_n],
        "v342_pred": v342_pred[offsets["v342"]: offsets["v342"] + common_n],
        "pt_pred": pt_pred[offsets["pt"]: offsets["pt"] + common_n],
        "n": common_n,
        "date": date_str,
    }

    # Use the labels from the model with most trustworthy alignment
    # V3.3 labels have zero NaNs, so prefer those
    result["labels"] = v33_lbl[offsets["v33"]: offsets["v33"] + common_n]

    return result


# ── Analysis functions ───────────────────────────────────────────────────────

def compute_ic(predictions, labels, mask=None):
    """Pearson correlation (IC) between predictions and labels."""
    if mask is None:
        mask = ~np.isnan(predictions) & ~np.isnan(labels)
    else:
        mask = mask & ~np.isnan(predictions) & ~np.isnan(labels)
    if mask.sum() < 30:
        return np.nan
    return np.corrcoef(predictions[mask], labels[mask])[0, 1]


def directional_accuracy(predictions, labels, mask=None):
    """Fraction of predictions where sign matches label sign (excluding zeros)."""
    if mask is None:
        mask = np.ones(len(predictions), dtype=bool)
    mask = mask & ~np.isnan(predictions) & ~np.isnan(labels)
    nonzero = mask & (labels != 0) & (predictions != 0)
    if nonzero.sum() < 10:
        return np.nan
    return (np.sign(predictions[nonzero]) == np.sign(labels[nonzero])).mean()


def mean_return_ticks(labels, mask):
    """Mean realized return in ticks for selected samples."""
    valid = mask & ~np.isnan(labels)
    if valid.sum() < 10:
        return np.nan
    return labels[valid].mean()


def net_ticks_per_trade(labels, mask, cost=COMMISSION_TICKS):
    """Mean realized return minus commission cost, per trade."""
    gross = mean_return_ticks(labels, mask)
    if np.isnan(gross):
        return np.nan
    return abs(gross) - cost  # trade in the direction of prediction


def sharpe_ratio(returns):
    """Annualized Sharpe (simple: mean/std, no annualization since horizon varies)."""
    if len(returns) < 10 or np.std(returns) == 0:
        return np.nan
    return np.mean(returns) / np.std(returns)


# ── Main confluence analysis ─────────────────────────────────────────────────

def run_confluence_analysis():
    """Main entry point."""
    print("=" * 80)
    print("MULTI-MODEL CONFLUENCE ANALYSIS")
    print("=" * 80)
    print()

    dates = discover_dates()
    if not dates:
        print("ERROR: No overlapping dates found across all 4 models.")
        sys.exit(1)
    print(f"Dates: {dates[0]} to {dates[-1]}")
    print()

    # Accumulate all predictions/labels across dates
    all_preds = {"v2": [], "v33": [], "v342": [], "pt": []}
    all_labels = []
    all_dates = []
    total_samples = 0

    for dt in dates:
        try:
            data = load_date(dt)
        except Exception as e:
            print(f"  SKIP {dt}: {e}")
            continue

        n = data["n"]
        all_preds["v2"].append(data["v2_pred"])
        all_preds["v33"].append(data["v33_pred"])
        all_preds["v342"].append(data["v342_pred"])
        all_preds["pt"].append(data["pt_pred"])
        all_labels.append(data["labels"])
        all_dates.extend([dt] * n)
        total_samples += n
        print(f"  {dt}: {n:,} aligned samples")

    print(f"\nTotal aligned samples: {total_samples:,} across {len(dates)} dates")
    print()

    # Concatenate
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)
    labels = np.concatenate(all_labels, axis=0)
    dates_arr = np.array(all_dates)

    # ── Step 1: Single-model baselines ───────────────────────────────────
    print("=" * 80)
    print("SINGLE-MODEL BASELINES")
    print("=" * 80)

    model_names = {"v2": "CNN-Mamba v2", "v33": "CNN-Mamba v3.3",
                   "v342": "CNN-Mamba v3.4.2", "pt": "PatchTST"}

    baseline_results = {}
    for model_key, model_name in model_names.items():
        preds = all_preds[model_key]
        print(f"\n  {model_name}:")
        model_res = {}
        for h_idx, horizon in enumerate(HORIZONS):
            p = preds[:, h_idx]
            l = labels[:, h_idx]
            mask = ~np.isnan(p) & ~np.isnan(l)
            ic = compute_ic(p, l, mask)
            da = directional_accuracy(p, l, mask)

            # Top-decile analysis (strongest predictions)
            abs_p = np.abs(p)
            abs_p[~mask] = 0
            thresh_90 = np.percentile(abs_p[mask], 90) if mask.sum() > 100 else np.inf
            top_mask = mask & (abs_p >= thresh_90)

            top_ic = compute_ic(p, l, top_mask)
            top_da = directional_accuracy(p, l, top_mask)

            # Directional return: trade in direction of prediction
            signed_ret = np.sign(p) * l
            top_signed_ret = signed_ret.copy()
            top_signed_ret[~top_mask] = np.nan
            top_mean = np.nanmean(top_signed_ret)
            top_net = top_mean - COMMISSION_TICKS if not np.isnan(top_mean) else np.nan
            top_count = top_mask.sum()

            # Per-trade returns for Sharpe
            valid_top = top_mask & ~np.isnan(signed_ret)
            if valid_top.sum() > 10:
                top_sharpe = sharpe_ratio(signed_ret[valid_top] - COMMISSION_TICKS)
            else:
                top_sharpe = np.nan

            model_res[horizon] = {
                "ic": round(float(ic), 4),
                "da": round(float(da), 4),
                "top10_ic": round(float(top_ic), 4),
                "top10_da": round(float(top_da), 4),
                "top10_gross_ticks": round(float(top_mean), 4),
                "top10_net_ticks": round(float(top_net), 4),
                "top10_sharpe": round(float(top_sharpe), 4),
                "top10_count": int(top_count),
            }
            print(f"    {horizon}: IC={ic:.4f}  DA={da:.4f}  "
                  f"Top10%: IC={top_ic:.4f} DA={top_da:.4f} "
                  f"gross={top_mean:.3f} net={top_net:.3f} ticks "
                  f"(n={top_count:,}, Sharpe={top_sharpe:.3f})")

        baseline_results[model_key] = model_res

    # ── Step 2: Confluence scoring ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONFLUENCE ANALYSIS")
    print("=" * 80)

    confluence_results = {}

    for h_idx, horizon in enumerate(HORIZONS):
        print(f"\n  Horizon: {horizon}")
        print(f"  {'─' * 70}")

        # Get signs of predictions from each model
        signs = {}
        for model_key in model_names:
            p = all_preds[model_key][:, h_idx]
            s = np.sign(p)
            s[np.isnan(p)] = 0  # treat NaN as no opinion
            signs[model_key] = s

        l = labels[:, h_idx]
        valid = ~np.isnan(l)

        # Agreement = number of models agreeing on direction
        # For each sample, count how many models predict positive vs negative
        sign_matrix = np.column_stack([signs[k] for k in model_names])  # (N, 4)

        n_long = (sign_matrix > 0).sum(axis=1)   # models predicting positive
        n_short = (sign_matrix < 0).sum(axis=1)  # models predicting negative
        n_active = (sign_matrix != 0).sum(axis=1)  # models with an opinion

        # Consensus direction: majority vote
        consensus_sign = np.where(n_long > n_short, 1,
                         np.where(n_short > n_long, -1, 0))

        # Agreement level: max(n_long, n_short) — how many models agree on the winning direction
        agreement = np.maximum(n_long, n_short)

        # Also compute average prediction magnitude across agreeing models
        avg_pred = np.zeros(len(l))
        for model_key in model_names:
            avg_pred += all_preds[model_key][:, h_idx]
        avg_pred /= len(model_names)

        horizon_results = {}

        # Note: with 4 models, a 2-2 split is a tie (no consensus direction),
        # so >=3 is the minimum meaningful agreement level. We test 3/4 and 4/4.
        for agree_level in [3, 4]:
            mask = valid & (agreement >= agree_level) & (consensus_sign != 0)
            n_trades = mask.sum()
            if n_trades < 30:
                print(f"    {agree_level}/4 agree: insufficient samples ({n_trades})")
                continue

            # Directional return: trade in consensus direction
            signed_ret = consensus_sign * l
            trade_returns = signed_ret[mask]

            gross_mean = trade_returns.mean()
            net_mean = gross_mean - COMMISSION_TICKS
            da = (trade_returns > 0).sum() / len(trade_returns)
            win_rate = da
            ic = compute_ic(avg_pred[mask], l[mask])
            trade_sharpe = sharpe_ratio(trade_returns - COMMISSION_TICKS)

            # Profit factor
            wins = trade_returns[trade_returns > COMMISSION_TICKS]
            losses = trade_returns[trade_returns <= COMMISSION_TICKS]
            if len(losses) > 0 and losses.sum() != 0:
                pf = wins.sum() / abs(losses.sum()) if len(wins) > 0 else 0
            else:
                pf = np.inf if len(wins) > 0 else np.nan

            # Long vs short breakdown
            long_mask = mask & (consensus_sign > 0)
            short_mask = mask & (consensus_sign < 0)
            long_ret = (consensus_sign * l)[long_mask].mean() if long_mask.sum() > 10 else np.nan
            short_ret = (consensus_sign * l)[short_mask].mean() if short_mask.sum() > 10 else np.nan
            long_net = long_ret - COMMISSION_TICKS if not np.isnan(long_ret) else np.nan
            short_net = short_ret - COMMISSION_TICKS if not np.isnan(short_ret) else np.nan

            # Top-magnitude within agreement (confluence + high conviction)
            abs_avg = np.abs(avg_pred)
            abs_avg[~mask] = 0
            if mask.sum() > 100:
                top_thresh = np.percentile(abs_avg[mask], 80)
                top_mask = mask & (abs_avg >= top_thresh)
            else:
                top_mask = mask
            top_ret = (consensus_sign * l)[top_mask]
            top_gross = top_ret.mean() if len(top_ret) > 10 else np.nan
            top_net = top_gross - COMMISSION_TICKS if not np.isnan(top_gross) else np.nan
            top_sharpe = sharpe_ratio(top_ret - COMMISSION_TICKS) if len(top_ret) > 10 else np.nan

            level_key = f"{agree_level}_of_4"
            horizon_results[level_key] = {
                "n_trades": int(n_trades),
                "n_long": int(long_mask.sum()),
                "n_short": int(short_mask.sum()),
                "gross_ticks": round(float(gross_mean), 4),
                "net_ticks": round(float(net_mean), 4),
                "win_rate": round(float(win_rate), 4),
                "ic": round(float(ic), 4),
                "sharpe": round(float(trade_sharpe), 4),
                "profit_factor": round(float(pf), 4) if not np.isinf(pf) else "inf",
                "long_net_ticks": round(float(long_net), 4) if not np.isnan(long_net) else None,
                "short_net_ticks": round(float(short_net), 4) if not np.isnan(short_net) else None,
                "top20_gross": round(float(top_gross), 4) if not np.isnan(top_gross) else None,
                "top20_net": round(float(top_net), 4) if not np.isnan(top_net) else None,
                "top20_sharpe": round(float(top_sharpe), 4) if not np.isnan(top_sharpe) else None,
                "top20_count": int(top_mask.sum()),
                "frac_of_total": round(n_trades / valid.sum(), 4),
            }

            print(f"    {agree_level}/4 agree: n={n_trades:>8,} ({n_trades/valid.sum()*100:.1f}%)  "
                  f"gross={gross_mean:+.3f}  net={net_mean:+.3f} ticks  "
                  f"WR={win_rate:.3f}  IC={ic:.4f}  Sharpe={trade_sharpe:.3f}  "
                  f"PF={pf:.2f}")
            print(f"      Long:  n={long_mask.sum():>7,}  net={long_net:+.3f} ticks" if not np.isnan(long_net) else "")
            print(f"      Short: n={short_mask.sum():>7,}  net={short_net:+.3f} ticks" if not np.isnan(short_net) else "")
            print(f"      Top20%+conf: n={top_mask.sum():>6,}  "
                  f"gross={top_gross:+.3f}  net={top_net:+.3f}  Sharpe={top_sharpe:.3f}"
                  if not np.isnan(top_gross) else "")

        confluence_results[horizon] = horizon_results

    # ── Step 3: Direction + magnitude confluence ─────────────────────────
    print("\n" + "=" * 80)
    print("MAGNITUDE-WEIGHTED CONFLUENCE (all models predict STRONG same direction)")
    print("=" * 80)

    magnitude_results = {}

    for h_idx, horizon in enumerate(HORIZONS):
        print(f"\n  Horizon: {horizon}")

        l = labels[:, h_idx]
        valid = ~np.isnan(l)

        # For each model, compute z-score of predictions (standardize per model)
        z_preds = {}
        for model_key in model_names:
            p = all_preds[model_key][:, h_idx]
            m = ~np.isnan(p)
            mu, sigma = p[m].mean(), p[m].std()
            z = (p - mu) / (sigma + 1e-10)
            z[~m] = 0
            z_preds[model_key] = z

        z_matrix = np.column_stack([z_preds[k] for k in model_names])  # (N, 4)
        avg_z = z_matrix.mean(axis=1)

        # All-strong-same-direction: all z-scores have same sign AND |z| > threshold
        z_thresh_results = {}
        for z_thresh in [0.5, 1.0, 1.5, 2.0]:
            # All models predict strongly positive
            all_strong_long = valid & np.all(z_matrix > z_thresh, axis=1)
            # All models predict strongly negative
            all_strong_short = valid & np.all(z_matrix < -z_thresh, axis=1)
            combined = all_strong_long | all_strong_short
            consensus = np.where(all_strong_long, 1, np.where(all_strong_short, -1, 0))

            n_trades = combined.sum()
            if n_trades < 20:
                print(f"    |z|>{z_thresh}: insufficient samples ({n_trades})")
                continue

            signed_ret = consensus * l
            trade_ret = signed_ret[combined]
            gross = trade_ret.mean()
            net = gross - COMMISSION_TICKS
            wr = (trade_ret > 0).mean()
            s = sharpe_ratio(trade_ret - COMMISSION_TICKS)

            z_thresh_results[f"z_{z_thresh}"] = {
                "n_trades": int(n_trades),
                "n_long": int(all_strong_long.sum()),
                "n_short": int(all_strong_short.sum()),
                "gross_ticks": round(float(gross), 4),
                "net_ticks": round(float(net), 4),
                "win_rate": round(float(wr), 4),
                "sharpe": round(float(s), 4),
                "frac_of_total": round(n_trades / valid.sum(), 4),
            }

            print(f"    |z|>{z_thresh}: n={n_trades:>7,} ({n_trades/valid.sum()*100:.1f}%)  "
                  f"gross={gross:+.3f}  net={net:+.3f} ticks  "
                  f"WR={wr:.3f}  Sharpe={s:.3f}  "
                  f"(L:{all_strong_long.sum():,} S:{all_strong_short.sum():,})")

        magnitude_results[horizon] = z_thresh_results

    # ── Step 4: Per-date analysis (stability) ────────────────────────────
    print("\n" + "=" * 80)
    print("PER-DATE CONFLUENCE STABILITY (4/4 agreement, 10s horizon)")
    print("=" * 80)

    h_idx = 2  # 10s
    sign_matrix = np.column_stack([np.sign(all_preds[k][:, h_idx]) for k in model_names])
    n_long = (sign_matrix > 0).sum(axis=1)
    n_short = (sign_matrix < 0).sum(axis=1)
    agreement = np.maximum(n_long, n_short)
    consensus = np.where(n_long > n_short, 1, np.where(n_short > n_long, -1, 0))
    l = labels[:, h_idx]
    valid = ~np.isnan(l)

    unique_dates = sorted(set(dates_arr))
    print(f"\n  {'Date':<12} {'N_4/4':>8} {'Gross':>8} {'Net':>8} {'WR':>6} {'Sharpe':>8} {'Dir':>5}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*5}")

    per_date_results = {}
    green_days = 0
    total_days = 0
    for dt in unique_dates:
        dt_mask = (dates_arr == dt) & valid & (agreement >= 4) & (consensus != 0)
        n_trades = dt_mask.sum()
        if n_trades < 5:
            continue
        total_days += 1
        signed_ret = (consensus * l)[dt_mask]
        gross = signed_ret.mean()
        net = gross - COMMISSION_TICKS
        wr = (signed_ret > 0).mean()
        s = sharpe_ratio(signed_ret - COMMISSION_TICKS)
        direction = "+" if net > 0 else "-"
        if net > 0:
            green_days += 1

        per_date_results[dt] = {
            "n_trades": int(n_trades),
            "gross": round(float(gross), 4),
            "net": round(float(net), 4),
            "wr": round(float(wr), 4),
            "sharpe": round(float(s), 4),
        }

        print(f"  {dt:<12} {n_trades:>8,} {gross:>+8.3f} {net:>+8.3f} {wr:>6.3f} {s:>+8.3f} {direction:>5}")

    if total_days > 0:
        print(f"\n  Green days: {green_days}/{total_days} ({green_days/total_days*100:.0f}%)")

    # ── Step 5: Confluence vs best single model comparison ───────────────
    print("\n" + "=" * 80)
    print("CONFLUENCE vs BEST SINGLE MODEL (head-to-head)")
    print("=" * 80)

    comparison = {}
    for h_idx, horizon in enumerate(HORIZONS):
        print(f"\n  Horizon: {horizon}")

        # Best single-model top-10% results
        best_single_net = -np.inf
        best_single_name = ""
        best_single_sharpe = -np.inf
        for model_key, model_name in model_names.items():
            res = baseline_results[model_key][horizon]
            if res["top10_net_ticks"] > best_single_net:
                best_single_net = res["top10_net_ticks"]
                best_single_name = model_name
            if res["top10_sharpe"] > best_single_sharpe:
                best_single_sharpe = res["top10_sharpe"]

        # Best confluence result
        conf_res = confluence_results.get(horizon, {})
        best_conf_net = -np.inf
        best_conf_level = ""
        best_conf_sharpe = -np.inf
        for level, res in conf_res.items():
            if res["net_ticks"] > best_conf_net:
                best_conf_net = res["net_ticks"]
                best_conf_level = level
            if res["sharpe"] > best_conf_sharpe:
                best_conf_sharpe = res["sharpe"]

        # Also check magnitude confluence
        mag_res = magnitude_results.get(horizon, {})
        best_mag_net = -np.inf
        best_mag_level = ""
        best_mag_sharpe = -np.inf
        for level, res in mag_res.items():
            if res["net_ticks"] > best_mag_net:
                best_mag_net = res["net_ticks"]
                best_mag_level = level
            if res["sharpe"] > best_mag_sharpe:
                best_mag_sharpe = res["sharpe"]

        conf_winner = best_conf_net > best_single_net
        mag_winner = best_mag_net > best_single_net

        comparison[horizon] = {
            "best_single_model": best_single_name,
            "best_single_net_ticks": round(float(best_single_net), 4),
            "best_single_sharpe": round(float(best_single_sharpe), 4),
            "best_confluence_net_ticks": round(float(best_conf_net), 4),
            "best_confluence_level": best_conf_level,
            "best_confluence_sharpe": round(float(best_conf_sharpe), 4),
            "best_magnitude_net_ticks": round(float(best_mag_net), 4),
            "best_magnitude_level": best_mag_level,
            "best_magnitude_sharpe": round(float(best_mag_sharpe), 4),
            "confluence_beats_single": conf_winner,
            "magnitude_beats_single": mag_winner,
        }

        print(f"    Best single model: {best_single_name} "
              f"net={best_single_net:+.3f} ticks, Sharpe={best_single_sharpe:.3f}")
        print(f"    Best direction confluence ({best_conf_level}): "
              f"net={best_conf_net:+.3f} ticks, Sharpe={best_conf_sharpe:.3f}"
              f"  {'BEATS SINGLE' if conf_winner else 'worse'}")
        print(f"    Best magnitude confluence ({best_mag_level}): "
              f"net={best_mag_net:+.3f} ticks, Sharpe={best_mag_sharpe:.3f}"
              f"  {'BEATS SINGLE' if mag_winner else 'worse'}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_dates": len(dates),
            "dates": dates,
            "total_samples": total_samples,
            "cost_ticks_rt": COMMISSION_TICKS,
            "models": list(model_names.values()),
        },
        "single_model_baselines": baseline_results,
        "direction_confluence": confluence_results,
        "magnitude_confluence": magnitude_results,
        "per_date_stability": per_date_results,
        "comparison": comparison,
    }

    out_path = OUTPUT_DIR / "multi_model_confluence_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for horizon in HORIZONS:
        c = comparison[horizon]
        verdict = "YES" if c["confluence_beats_single"] or c["magnitude_beats_single"] else "NO"
        print(f"  {horizon}: Confluence improves over single model? {verdict}")
        if c["confluence_beats_single"]:
            improvement = c["best_confluence_net_ticks"] - c["best_single_net_ticks"]
            print(f"    Direction confluence: +{improvement:.3f} ticks/trade improvement")
        if c["magnitude_beats_single"]:
            improvement = c["best_magnitude_net_ticks"] - c["best_single_net_ticks"]
            print(f"    Magnitude confluence: +{improvement:.3f} ticks/trade improvement")


if __name__ == "__main__":
    run_confluence_analysis()
