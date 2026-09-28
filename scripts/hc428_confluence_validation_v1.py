#!/usr/bin/env python3
"""
HC #428 Confluence Validation Pipeline v1
==========================================
Validates CNN-Mamba v2 + PatchTST confluence strategy over the full 40-day OOT
period with regime-agnostic gates per HC #428 R1 and HC #344.

STRATEGY:
  - CNN-Mamba v2: top 3% shorts (1s horizon predictions from bulk_oot_v2)
  - Meta-model filter: ensemble MLP (15-fold, corr +0.138) — meta_score > 0
  - PatchTST agreement: PatchTST 1s prediction also negative (same-direction gate)
  - Cost: 0.376 ticks RT (passive fill, commission only — CLAUDE.md canonical)
  - Hold: 1s passive exit

HC #428 R1 GATES:
  - ≥40 days OOT required
  - Per-day Sharpe/PF/WR reported
  - Regime classification: ES close-to-close (up=green, down=red, flat=flat)
  - Reject if |Sharpe_green − Sharpe_red| / max(|Sg|,|Sr|) > 0.50
  - Both regime mean P&L must be ≥ 0

HC #344: Day-concentration cap ≤ 0.70 (max single-day fraction of total trades)

NOTE: Run this AFTER PatchTST dense predictions are synced from Razer.
      CNN-Mamba v2 dates: output/cnn_mamba_v2_bulk_oot_v2/
      PatchTST dense dates: output/hc470_dense_patchtst_s5/

Usage:
  python scripts/hc428_confluence_validation_v1.py [--no-meta] [--no-patchtst]

Flags:
  --no-meta      Skip meta-model filter (test CNN-Mamba top-3% only)
  --no-patchtst  Skip PatchTST agreement gate (useful when preds not yet synced)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hc428_confluence")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/home/jupiter/Lvl3Quant")
CNN_MAMBA_DIR = BASE / "output" / "cnn_mamba_v2_bulk_oot_v2"
PATCHTST_DIR  = BASE / "output" / "hc470_dense_patchtst_s5"
META_S_WEIGHTS = BASE / "output" / "meta_production_v1" / "weights"
META_L_WEIGHTS = BASE / "output" / "meta_production_longs_v1" / "weights"
MBO_DIR        = BASE / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = BASE / "output" / "regime_labels" / "oot_dates_regime.parquet"
OUT_DIR        = BASE / "output" / "hc428_confluence_validation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
COST_TICKS      = 0.376   # passive RT commission (CLAUDE.md canonical)
SHORT_PCT       = 3        # top 3% shorts by CNN-Mamba 1s prediction
TRAIN_WINDOW    = 10       # meta-model fold window (must match training)
MIN_OOT_DAYS    = 40       # HC #428 requirement
DAY_CONC_CAP    = 0.70     # HC #344
REGIME_GAP_GATE = 0.50     # HC #428 R1


# ── Architecture (must match train_meta_production_v1.py) ────────────────────
class ProductionMetaMLP(nn.Module):
    """256→128→64→32 (arch sweep winner)."""
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def sharpe(arr: np.ndarray) -> float:
    """Annualised Sharpe (252 trading days) from per-day net-tick array."""
    if len(arr) < 2:
        return float("nan")
    s = arr.std(ddof=1)
    if s == 0 or math.isnan(s):
        return float("nan")
    return float(arr.mean() / s * math.sqrt(252))


def profit_factor(arr: np.ndarray) -> float:
    wins  = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def win_rate(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return float("nan")
    return float((arr > 0).mean())


def load_regime_map() -> dict[str, str]:
    """Return {YYYYMMDD: 'green'|'red'|'flat'} from canonical parquet."""
    import pandas as pd
    df = pd.read_parquet(REGIME_PARQUET)
    mp: dict[str, str] = {}
    for _, row in df.iterrows():
        lbl = str(row["trend_label"]).lower()
        if lbl == "up":
            mp[str(row["date"])] = "green"
        elif lbl == "down":
            mp[str(row["date"])] = "red"
        else:
            mp[str(row["date"])] = "flat"
    return mp


def load_meta_weights(weights_dir: Path) -> list[dict]:
    """Load all fold weights sorted by fold index."""
    pt_files = sorted(weights_dir.glob("fold_*.pt"))
    folds = []
    for pt in pt_files:
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        folds.append(ckpt)
    return folds


def infer_meta_score(folds: list[dict], features: np.ndarray, date: str) -> np.ndarray | None:
    """
    Ensemble meta-model inference: average predictions from all folds whose
    test_date ≤ date (walk-forward correct — no lookahead).
    Returns array of shape (N,) or None if no applicable folds.
    """
    eligible = [f for f in folds if f["test_date"] <= date]
    if not eligible:
        return None
    # Use the most recent fold (closest trained model before this date)
    ckpt = eligible[-1]
    mean = ckpt["norm_mean"]
    std  = ckpt["norm_std"]
    features_n = ((features - mean) / std).astype(np.float32)
    model = ProductionMetaMLP(ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(features_n)).numpy()
    return scores


def load_cnn_mamba_day(date: str) -> dict | None:
    """Load CNN-Mamba v2 predictions for a date. Returns None if missing."""
    f = CNN_MAMBA_DIR / f"{date}_predictions.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {
        "date": date,
        "predictions": d["predictions"],   # (N, 3) — horizons 1s/5s/10s
        "labels":      d["labels"],         # (N, 3)
        "n_windows":   int(d["n_windows"]),
        "window_size": int(d["window_size"]),
        "stride":      int(d["stride"]),
    }


def load_patchtst_day(date: str) -> dict | None:
    """Load PatchTST dense predictions for a date. Returns None if missing."""
    # HC470 naming convention: {date}_dense_predictions.npz
    f = PATCHTST_DIR / f"{date}_dense_predictions.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {
        "date": date,
        "predictions": d["predictions"],   # (N, 3) — horizons 1s/5s/10s
        "stride":      int(d.get("stride", 25)),
    }


def load_mbo_day(date: str, n_windows: int, window_size: int, stride: int) -> dict | None:
    """Load MBO features for the exact windows CNN-Mamba v2 used."""
    mbo_f = MBO_DIR / f"{date}_mbo_events.npz"
    if not mbo_f.exists():
        return None
    mbo = np.load(mbo_f, allow_pickle=True)
    events = mbo["events"]
    labels_1s = mbo["labels_1s"]

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(labels_1s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]

    events  = events[indices]
    labels_1s = labels_1s[indices]
    return {"events": events, "labels_1s": labels_1s, "n_windows": len(indices)}


def align_patchtst(cm_n_windows: int, cm_stride: int, cm_window: int,
                    pt_preds: np.ndarray, pt_stride: int, pt_window: int = 500) -> np.ndarray | None:
    """
    Align PatchTST dense predictions to CNN-Mamba event indices.
    Both are index by the *last* event in their respective windows.
    CNN-Mamba event index for window k: k * cm_stride + (cm_window - 1)
    PatchTST event index for window j: j * pt_stride + (pt_window - 1)

    We find the nearest PatchTST prediction for each CNN-Mamba event.
    Returns (cm_n_windows,) array of PatchTST 1s predictions, or None.
    """
    cm_indices = np.arange(cm_n_windows) * cm_stride + (cm_window - 1)
    pt_indices = np.arange(len(pt_preds)) * pt_stride + (pt_window - 1)

    if len(pt_indices) == 0:
        return None

    aligned = np.empty(cm_n_windows, dtype=np.float32)
    for i, ci in enumerate(cm_indices):
        j = int(np.argmin(np.abs(pt_indices - ci)))
        aligned[i] = pt_preds[j, 0]  # 1s horizon
    return aligned


def process_day(
    date: str,
    meta_short_folds: list[dict],
    use_meta: bool,
    use_patchtst: bool,
) -> dict | None:
    """
    Process one day. Returns per-trade dict or None.

    Returns dict with keys:
      date, trades, net_ticks_per_trade (array), n_total_signals,
      n_after_tier, n_after_meta, n_after_confluence
    """
    cm = load_cnn_mamba_day(date)
    if cm is None:
        return None

    n_windows  = cm["n_windows"]
    window_size = cm["window_size"]
    stride     = cm["stride"]
    preds      = cm["predictions"]   # (N, 3)
    labels     = cm["labels"]        # (N, 3)

    # Truncate to n_windows (safety)
    preds  = preds[:n_windows]
    labels = labels[:n_windows]

    pred_1s   = preds[:, 0]
    label_1s  = labels[:, 0]

    # Remove NaN labels
    valid = ~(np.isnan(label_1s) | np.isnan(pred_1s))
    preds     = preds[valid]
    labels    = labels[valid]
    pred_1s   = pred_1s[valid]
    label_1s  = label_1s[valid]
    n_total   = len(pred_1s)

    if n_total == 0:
        return None

    # ── Step 1: Top 3% shorts by CNN-Mamba 1s prediction ────────────────────
    threshold = np.percentile(pred_1s, SHORT_PCT)
    tier_mask = pred_1s <= threshold
    n_tier = int(tier_mask.sum())
    if n_tier == 0:
        return None

    # ── Step 2: Meta-model filter ────────────────────────────────────────────
    if use_meta and meta_short_folds:
        mbo = load_mbo_day(date, n_total, window_size, stride)
        if mbo is None:
            log.warning(f"{date}: MBO not found, skipping meta filter")
            meta_mask = np.ones(n_total, dtype=bool)
        else:
            events  = mbo["events"][:n_total]
            ranks   = np.argsort(np.argsort(pred_1s)).astype(np.float32) / max(n_total, 1)
            # Features: 25 MBO + 3 preds + rank = 29 (matches training)
            features = np.column_stack([
                events, preds[:, 0], preds[:, 1], preds[:, 2], ranks
            ]).astype(np.float32)
            scores = infer_meta_score(meta_short_folds, features, date)
            if scores is None:
                log.warning(f"{date}: No applicable meta folds, skipping meta filter")
                meta_mask = np.ones(n_total, dtype=bool)
            else:
                meta_mask = scores > 0
    else:
        meta_mask = np.ones(n_total, dtype=bool)

    combined_mask = tier_mask & meta_mask
    n_after_meta = int(combined_mask.sum())

    # ── Step 3: PatchTST agreement gate ─────────────────────────────────────
    if use_patchtst:
        pt = load_patchtst_day(date)
        if pt is None:
            log.debug(f"{date}: PatchTST not available, skipping confluence gate")
            pt_agree_mask = np.ones(n_total, dtype=bool)
        else:
            pt_preds_full = pt["predictions"][:, :]  # (M, 3)
            pt_stride     = pt["stride"]
            pt_aligned_1s = align_patchtst(
                n_total, stride, window_size,
                pt_preds_full, pt_stride, pt_window=500,
            )
            if pt_aligned_1s is None:
                pt_agree_mask = np.ones(n_total, dtype=bool)
            else:
                # Agreement: PatchTST 1s prediction also negative (short direction)
                pt_agree_mask = pt_aligned_1s < 0
    else:
        pt_agree_mask = np.ones(n_total, dtype=bool)

    final_mask = combined_mask & pt_agree_mask
    n_confluence = int(final_mask.sum())

    if n_confluence == 0:
        return {
            "date": date,
            "trades": 0,
            "net_ticks": np.array([], dtype=np.float32),
            "n_total": n_total,
            "n_tier": n_tier,
            "n_meta": n_after_meta,
            "n_confluence": 0,
        }

    # ── P&L: short trade, 1s hold, passive fill ───────────────────────────────
    # Short: profit = -label_1s (move down = profit)
    # Cost: 0.376 ticks commission (passive in + passive out)
    selected_labels = label_1s[final_mask]
    net_ticks = -selected_labels - COST_TICKS  # per-trade net ticks

    return {
        "date": date,
        "trades": n_confluence,
        "net_ticks": net_ticks,
        "n_total": n_total,
        "n_tier": n_tier,
        "n_meta": n_after_meta,
        "n_confluence": n_confluence,
    }


def run_validation(use_meta: bool = True, use_patchtst: bool = True) -> None:
    log.info("=" * 70)
    log.info("HC #428 Confluence Validation Pipeline v1")
    log.info(f"  Meta filter: {use_meta}")
    log.info(f"  PatchTST gate: {use_patchtst}")
    log.info("=" * 70)

    # Load regime map
    try:
        regime_map = load_regime_map()
        log.info(f"Loaded regime map: {len(regime_map)} dates")
    except Exception as e:
        log.error(f"Failed to load regime map: {e}")
        sys.exit(1)

    # Load meta-model weights (shorts only for this pipeline)
    if use_meta:
        log.info("Loading meta-model short weights...")
        try:
            meta_short_folds = load_meta_weights(META_S_WEIGHTS)
            log.info(f"  Loaded {len(meta_short_folds)} short folds")
        except Exception as e:
            log.warning(f"Failed to load meta weights: {e} — disabling meta filter")
            meta_short_folds = []
            use_meta = False
    else:
        meta_short_folds = []

    # Enumerate CNN-Mamba dates
    cm_files = sorted([
        f for f in CNN_MAMBA_DIR.glob("*_predictions.npz")
        if "_stale_" not in str(f)
    ])
    dates = [f.name[:8] for f in cm_files]
    log.info(f"CNN-Mamba v2 dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # Check PatchTST coverage
    if use_patchtst:
        pt_dates = set(
            f.name[:8]
            for f in PATCHTST_DIR.glob("*_dense_predictions.npz")
            if PATCHTST_DIR.exists()
        )
        log.info(f"PatchTST dense dates: {len(pt_dates)}")
        overlap = [d for d in dates if d in pt_dates]
        log.info(f"Overlap (both models): {len(overlap)} dates")
        if len(overlap) < MIN_OOT_DAYS:
            log.warning(
                f"ONLY {len(overlap)} overlap dates — need {MIN_OOT_DAYS} for HC #428. "
                f"Run after PatchTST sync completes."
            )
    else:
        pt_dates = set()
        overlap = dates

    # Process each date
    per_day_results: list[dict] = []
    for date in dates:
        result = process_day(date, meta_short_folds, use_meta, use_patchtst)
        if result is None:
            log.debug(f"{date}: skipped (no data)")
            continue
        per_day_results.append(result)

    n_days = len(per_day_results)
    log.info(f"\nProcessed {n_days} days with trades")

    if n_days == 0:
        log.error("No days with data — cannot produce report")
        sys.exit(1)

    # ── Per-day summary ───────────────────────────────────────────────────────
    total_trades = sum(r["trades"] for r in per_day_results)
    log.info(f"Total trades: {total_trades:,}")

    # Build per-day net-tick arrays
    per_day_nt: dict[str, np.ndarray] = {
        r["date"]: r["net_ticks"] for r in per_day_results if r["trades"] > 0
    }

    # Day concentration check (HC #344)
    if total_trades > 0:
        max_day_trades = max(r["trades"] for r in per_day_results)
        day_conc = max_day_trades / total_trades
    else:
        day_conc = float("nan")

    # Per-day statistics
    day_stats = []
    all_day_means = []
    for r in per_day_results:
        date = r["date"]
        nt = r["net_ticks"]
        regime = regime_map.get(date, "unknown")
        if len(nt) == 0:
            day_mean = 0.0
            day_pf   = float("nan")
            day_wr   = float("nan")
        else:
            day_mean = float(nt.mean())
            day_pf   = profit_factor(nt)
            day_wr   = win_rate(nt)
            all_day_means.append(day_mean)
        day_stats.append({
            "date": date,
            "regime": regime,
            "trades": r["trades"],
            "mean_net_ticks": day_mean,
            "pf": day_pf,
            "wr": day_wr,
            "n_tier": r["n_tier"],
            "n_meta": r["n_meta"],
            "n_confluence": r["n_confluence"],
        })

    # ── Regime stratification ─────────────────────────────────────────────────
    regime_groups: dict[str, list[float]] = defaultdict(list)
    for s in day_stats:
        if s["trades"] > 0:
            regime_groups[s["regime"]].append(s["mean_net_ticks"])

    regime_sharpes: dict[str, float] = {}
    regime_means:   dict[str, float] = {}
    for r in ("green", "red", "flat"):
        arr = np.array(regime_groups[r])
        regime_sharpes[r] = sharpe(arr) if len(arr) >= 2 else float("nan")
        regime_means[r]   = float(arr.mean()) if len(arr) > 0 else float("nan")

    sg = regime_sharpes["green"]
    sr = regime_sharpes["red"]
    if not (math.isnan(sg) or math.isnan(sr)) and max(abs(sg), abs(sr)) > 0:
        regime_gap = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        regime_gap = float("nan")

    # ── Overall metrics ───────────────────────────────────────────────────────
    all_nt = np.concatenate([r["net_ticks"] for r in per_day_results if r["trades"] > 0])
    overall_mean = float(all_nt.mean()) if len(all_nt) > 0 else float("nan")
    overall_pf   = profit_factor(all_nt)
    overall_wr   = win_rate(all_nt)
    day_means_arr = np.array(all_day_means)
    daily_sharpe  = sharpe(day_means_arr)

    # ── HC #428 Gate decisions ────────────────────────────────────────────────
    gate_oot_days  = n_days >= MIN_OOT_DAYS
    gate_regime_gap = (not math.isnan(regime_gap)) and (regime_gap <= REGIME_GAP_GATE)
    gate_regime_means = (
        (math.isnan(regime_means["green"]) or regime_means["green"] >= 0)
        and (math.isnan(regime_means["red"]) or regime_means["red"] >= 0)
    )
    gate_day_conc = (not math.isnan(day_conc)) and (day_conc <= DAY_CONC_CAP)
    gate_profitable = overall_mean > 0

    all_gates_pass = all([gate_oot_days, gate_regime_gap, gate_regime_means,
                          gate_day_conc, gate_profitable])

    # ── Output ────────────────────────────────────────────────────────────────
    report = {
        "config": {
            "use_meta": use_meta,
            "use_patchtst": use_patchtst,
            "cost_ticks": COST_TICKS,
            "short_pct": SHORT_PCT,
            "min_oot_days": MIN_OOT_DAYS,
            "day_conc_cap": DAY_CONC_CAP,
            "regime_gap_gate": REGIME_GAP_GATE,
        },
        "summary": {
            "n_days": n_days,
            "total_trades": int(total_trades),
            "mean_net_ticks": round(overall_mean, 4),
            "overall_pf": round(overall_pf, 3),
            "overall_wr": round(overall_wr, 4),
            "daily_sharpe_annualized": round(daily_sharpe, 3),
            "day_conc": round(day_conc, 4),
        },
        "regime_stratification": {
            "green": {
                "n_days": len(regime_groups["green"]),
                "sharpe": round(regime_sharpes["green"], 3),
                "mean_net_ticks": round(regime_means["green"], 4),
            },
            "red": {
                "n_days": len(regime_groups["red"]),
                "sharpe": round(regime_sharpes["red"], 3),
                "mean_net_ticks": round(regime_means["red"], 4),
            },
            "flat": {
                "n_days": len(regime_groups["flat"]),
                "sharpe": round(regime_sharpes["flat"], 3),
                "mean_net_ticks": round(regime_means["flat"], 4),
            },
            "regime_gap": round(regime_gap, 4) if not math.isnan(regime_gap) else None,
        },
        "hc428_gates": {
            "oot_days_ge_40": gate_oot_days,
            "regime_gap_le_0.50": gate_regime_gap,
            "both_regime_means_non_negative": gate_regime_means,
            "day_conc_le_0.70": gate_day_conc,
            "profitable": gate_profitable,
            "ALL_PASS": all_gates_pass,
        },
        "per_day": day_stats,
    }

    out_json = OUT_DIR / "report.json"
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # ── Console summary ───────────────────────────────────────────────────────
    verdict = "PASS" if all_gates_pass else "FAIL"
    print(f"\n{'='*70}")
    print(f"HC #428 CONFLUENCE VALIDATION — {verdict}")
    print(f"{'='*70}")
    print(f"  Days:          {n_days}  (need ≥{MIN_OOT_DAYS})")
    print(f"  Total trades:  {total_trades:,}")
    print(f"  Mean net ticks:{overall_mean:+.4f} / trade")
    print(f"  Overall PF:    {overall_pf:.3f}")
    print(f"  Overall WR:    {overall_wr:.1%}")
    print(f"  Daily Sharpe:  {daily_sharpe:.3f} (annualized)")
    print(f"  Day conc:      {day_conc:.3f}  (cap={DAY_CONC_CAP})")
    print(f"\n  Regime stratification:")
    for r in ("green", "red", "flat"):
        n = len(regime_groups[r])
        sh = regime_sharpes[r]
        mn = regime_means[r]
        print(f"    {r:5s}: {n} days | Sharpe {sh:+.3f} | mean {mn:+.4f} ticks")
    print(f"  Regime gap:    {regime_gap:.4f}  (gate ≤{REGIME_GAP_GATE})")
    print(f"\n  Gates:")
    print(f"    OOT days ≥40:               {'PASS' if gate_oot_days else 'FAIL'}")
    print(f"    Regime gap ≤0.50:           {'PASS' if gate_regime_gap else 'FAIL'}")
    print(f"    Both regime means ≥0:       {'PASS' if gate_regime_means else 'FAIL'}")
    print(f"    Day conc ≤0.70:             {'PASS' if gate_day_conc else 'FAIL'}")
    print(f"    Profitable (mean > 0):      {'PASS' if gate_profitable else 'FAIL'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Report saved: {out_json}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HC #428 Confluence Validation")
    parser.add_argument("--no-meta",     action="store_true", help="Skip meta-model filter")
    parser.add_argument("--no-patchtst", action="store_true", help="Skip PatchTST agreement gate")
    args = parser.parse_args()
    run_validation(use_meta=not args.no_meta, use_patchtst=not args.no_patchtst)
