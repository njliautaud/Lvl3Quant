#!/usr/bin/env python3
"""
PatchTST Confluence Filter for CNN-Mamba v2
=============================================
Tests whether PatchTST agreement as an entry filter improves
CNN-Mamba v2 risk-adjusted returns on 39 OOS decay dates.

MIDPOINT-BASED analysis (no FIFO sim).
Cost: 0.376 ticks RT ($4.70) per HC #52.

Strategies tested:
  1. Baseline       — CNN-Mamba signal only (|pred_10s| z > threshold)
  2. Confluence     — CNN-Mamba + PatchTST agrees on direction
  3. Strong conflu. — CNN-Mamba + PatchTST agrees AND PatchTST |z| > threshold
  4. Anti-confluence— CNN-Mamba + PatchTST DISAGREES (control, expect worse)

Alignment: valid_indices have different strides (~30k vs ~32.5k).
Uses nearest-neighbor matching with a max tolerance of 5000 indices.

Usage:
    python alpha_discovery/deep_models/test_patchtst_confluence.py
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
CM_DIR = LVL3_ROOT / "output" / "decay_v4_comprehensive" / "CNN-Mamba_v2"
PT_DIR = LVL3_ROOT / "output" / "decay_v4_comprehensive" / "PatchTST"
OUTPUT_DIR = LVL3_ROOT / "output" / "patchtst_confluence_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK_VALUE = 12.50       # ES tick value
COST_RT_TICKS = 0.376    # $4.70 / $12.50 — commission only (HC #52)
COST_RT_USD = 4.70
HORIZONS = ["1s", "5s", "10s"]
PRED_COL = 2             # index into preds (N,3) for 10s horizon
THRESHOLDS = [1.0, 1.5, 2.0, 2.5]  # z-score thresholds for CNN-Mamba
PT_THRESHOLD_FRAC = 0.5  # PatchTST threshold = this fraction of CM threshold for strong confluence
NN_MAX_GAP = 5000        # max index gap for nearest-neighbor match

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("confluence")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(OUTPUT_DIR / f"confluence_{_ts}.log"), mode="w")
_sh = logging.StreamHandler(sys.stdout)
for h in [_fh, _sh]:
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(h)


# ── Data classes ──

@dataclass
class TradeResult:
    pnl_ticks: float
    direction: int   # +1 long, -1 short
    pred_cm: float
    pred_pt: float
    label: float


@dataclass
class StrategyResult:
    name: str
    threshold: float
    n_trades: int
    win_rate: float
    sharpe: float
    sortino: float
    profit_factor: float
    rr_ratio: float
    total_pnl_ticks: float
    total_pnl_usd: float
    coverage_vs_baseline: float
    per_date: Dict[str, dict] = field(default_factory=dict)


# ── Alignment ──

def align_predictions(cm_data: dict, pt_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align CNN-Mamba and PatchTST predictions by valid_indices using
    nearest-neighbor matching with tolerance NN_MAX_GAP.

    Returns: (cm_preds_10s, pt_preds_10s, labels_10s, matched_count)
    """
    cm_idx = cm_data["valid_indices"]
    pt_idx = pt_data["valid_indices"]
    cm_preds = cm_data["preds"][:, PRED_COL]  # 10s horizon
    pt_preds = pt_data["preds"][:, PRED_COL]
    cm_labels = cm_data["labels_10s"]

    # For each CM index, find nearest PT index
    # Both arrays are sorted, so use searchsorted
    insert_pos = np.searchsorted(pt_idx, cm_idx)

    n_pt = len(pt_idx)
    matched_cm = []
    matched_pt = []
    matched_labels = []

    for i, pos in enumerate(insert_pos):
        best_dist = NN_MAX_GAP + 1
        best_j = -1

        # Check position and position-1
        for candidate in [pos, pos - 1]:
            if 0 <= candidate < n_pt:
                dist = abs(int(cm_idx[i]) - int(pt_idx[candidate]))
                if dist < best_dist:
                    best_dist = dist
                    best_j = candidate

        if best_dist <= NN_MAX_GAP:
            matched_cm.append(cm_preds[i])
            matched_pt.append(pt_preds[best_j])
            matched_labels.append(cm_labels[i])

    return (
        np.array(matched_cm, dtype=np.float32),
        np.array(matched_pt, dtype=np.float32),
        np.array(matched_labels, dtype=np.float32),
        len(matched_cm),
    )


# ── Metrics ──

def compute_metrics(pnls: np.ndarray) -> dict:
    """Compute Sharpe, Sortino, PF, win rate, R:R from tick PnLs."""
    if len(pnls) == 0:
        return dict(win_rate=0, sharpe=0, sortino=0, profit_factor=0, rr_ratio=0,
                    total_pnl=0, n_trades=0, mean_win=0, mean_loss=0)

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0

    mean = pnls.mean()
    std = pnls.std(ddof=1) if len(pnls) > 1 else 1e-9
    sharpe = mean / std if std > 1e-9 else 0

    downside = pnls[pnls < 0]
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-9
    sortino = mean / downside_std if downside_std > 1e-9 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    mean_win = wins.mean() if len(wins) > 0 else 0
    mean_loss = abs(losses.mean()) if len(losses) > 0 else 1e-9
    rr_ratio = mean_win / mean_loss if mean_loss > 1e-9 else float("inf")

    return dict(
        win_rate=win_rate,
        sharpe=sharpe,
        sortino=sortino,
        profit_factor=profit_factor,
        rr_ratio=rr_ratio,
        total_pnl=float(pnls.sum()),
        n_trades=len(pnls),
        mean_win=float(mean_win),
        mean_loss=float(mean_loss),
    )


# ── Per-date processing ──

def process_date(args) -> Optional[dict]:
    """Process a single date: load, align, run all strategies."""
    date_str, threshold = args
    cm_path = CM_DIR / date_str / "predictions.npz"
    pt_path = PT_DIR / date_str / "predictions.npz"

    if not cm_path.exists() or not pt_path.exists():
        return None

    try:
        cm_data = dict(np.load(cm_path))
        pt_data = dict(np.load(pt_path))
    except Exception as e:
        log.warning(f"Failed to load {date_str}: {e}")
        return None

    cm_preds, pt_preds, labels, n_matched = align_predictions(cm_data, pt_data)

    if n_matched < 10:
        log.warning(f"{date_str}: only {n_matched} matched samples, skipping")
        return None

    # Z-score normalize predictions (expanding would be ideal but we have
    # limited samples — use full-date z-score since this is OOS anyway)
    cm_std = cm_preds.std()
    pt_std = pt_preds.std()
    if cm_std < 1e-9 or pt_std < 1e-9:
        return None

    cm_z = (cm_preds - cm_preds.mean()) / cm_std
    pt_z = (pt_preds - pt_preds.mean()) / pt_std

    # PnL from midpoint: trade in direction of prediction, PnL = direction * label - cost
    # label is in prediction units (decay). Positive label = price went up.
    # We convert labels to ticks for PnL: labels are already normalized so we treat
    # raw label * direction as the "edge in prediction units" and use it directly.
    # The cost is in ticks of prediction space — we subtract COST_RT_TICKS from |pnl|.

    results = {}

    # ── Strategy 1: Baseline (CNN-Mamba only) ──
    mask_base = np.abs(cm_z) > threshold
    if mask_base.sum() > 0:
        direction = np.sign(cm_z[mask_base])
        raw_pnl = direction * labels[mask_base]
        pnl = raw_pnl - COST_RT_TICKS
        results["baseline"] = {
            "pnls": pnl.tolist(),
            "n_trades": int(mask_base.sum()),
            **compute_metrics(pnl),
        }
    else:
        results["baseline"] = {"pnls": [], "n_trades": 0, **compute_metrics(np.array([]))}

    # ── Strategy 2: Confluence (same sign) ──
    mask_conf = mask_base & (np.sign(cm_z) == np.sign(pt_z))
    if mask_conf.sum() > 0:
        direction = np.sign(cm_z[mask_conf])
        raw_pnl = direction * labels[mask_conf]
        pnl = raw_pnl - COST_RT_TICKS
        results["confluence"] = {
            "pnls": pnl.tolist(),
            "n_trades": int(mask_conf.sum()),
            **compute_metrics(pnl),
        }
    else:
        results["confluence"] = {"pnls": [], "n_trades": 0, **compute_metrics(np.array([]))}

    # ── Strategy 3: Strong confluence (same sign + PT above its threshold) ──
    pt_thresh = threshold * PT_THRESHOLD_FRAC
    mask_strong = mask_base & (np.sign(cm_z) == np.sign(pt_z)) & (np.abs(pt_z) > pt_thresh)
    if mask_strong.sum() > 0:
        direction = np.sign(cm_z[mask_strong])
        raw_pnl = direction * labels[mask_strong]
        pnl = raw_pnl - COST_RT_TICKS
        results["strong_confluence"] = {
            "pnls": pnl.tolist(),
            "n_trades": int(mask_strong.sum()),
            **compute_metrics(pnl),
        }
    else:
        results["strong_confluence"] = {"pnls": [], "n_trades": 0, **compute_metrics(np.array([]))}

    # ── Strategy 4: Anti-confluence (opposite sign — control) ──
    mask_anti = mask_base & (np.sign(cm_z) != np.sign(pt_z)) & (np.sign(pt_z) != 0)
    if mask_anti.sum() > 0:
        direction = np.sign(cm_z[mask_anti])
        raw_pnl = direction * labels[mask_anti]
        pnl = raw_pnl - COST_RT_TICKS
        results["anti_confluence"] = {
            "pnls": pnl.tolist(),
            "n_trades": int(mask_anti.sum()),
            **compute_metrics(pnl),
        }
    else:
        results["anti_confluence"] = {"pnls": [], "n_trades": 0, **compute_metrics(np.array([]))}

    return {
        "date": date_str,
        "threshold": threshold,
        "n_cm_samples": len(cm_data["preds"]),
        "n_pt_samples": len(pt_data["preds"]),
        "n_matched": n_matched,
        "strategies": results,
    }


# ── Aggregation ──

def aggregate_results(date_results: List[dict], threshold: float) -> Dict[str, StrategyResult]:
    """Aggregate per-date results into strategy-level summaries."""
    strategies = ["baseline", "confluence", "strong_confluence", "anti_confluence"]
    agg = {}

    # Get baseline trade count for coverage calculation
    baseline_trades = 0
    for dr in date_results:
        baseline_trades += dr["strategies"]["baseline"]["n_trades"]

    for strat in strategies:
        all_pnls = []
        per_date = {}
        for dr in date_results:
            s = dr["strategies"][strat]
            all_pnls.extend(s["pnls"])
            per_date[dr["date"]] = {
                "n_trades": s["n_trades"],
                "win_rate": round(s["win_rate"], 4),
                "total_pnl": round(s["total_pnl"], 4),
            }

        all_pnls = np.array(all_pnls, dtype=np.float64)
        m = compute_metrics(all_pnls)

        coverage = m["n_trades"] / baseline_trades if baseline_trades > 0 else 0

        agg[strat] = StrategyResult(
            name=strat,
            threshold=threshold,
            n_trades=m["n_trades"],
            win_rate=round(m["win_rate"], 4),
            sharpe=round(m["sharpe"], 4),
            sortino=round(m["sortino"], 4),
            profit_factor=round(m["profit_factor"], 4),
            rr_ratio=round(m["rr_ratio"], 4),
            total_pnl_ticks=round(m["total_pnl"], 4),
            total_pnl_usd=round(m["total_pnl"] * TICK_VALUE, 2),
            coverage_vs_baseline=round(coverage, 4),
            per_date=per_date,
        )

    return agg


def print_summary_table(all_results: Dict[float, Dict[str, StrategyResult]]):
    """Print a clean comparison table across thresholds and strategies."""
    strategies = ["baseline", "confluence", "strong_confluence", "anti_confluence"]
    header = f"{'Threshold':>9} | {'Strategy':>20} | {'Trades':>7} | {'WinRate':>7} | {'Sharpe':>7} | {'Sortino':>8} | {'PF':>6} | {'R:R':>5} | {'PnL(t)':>9} | {'PnL($)':>10} | {'Cover':>6}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  PATCHTST CONFLUENCE FILTER — CNN-MAMBA v2 (MIDPOINT-BASED)")
    print("  Cost: 0.376 ticks RT ($4.70 commission, no spread)")
    print("  39 OOS decay dates | 10s horizon")
    print("=" * len(header))
    print(header)
    print(sep)

    for thresh in sorted(all_results.keys()):
        for strat in strategies:
            r = all_results[thresh][strat]
            label = strat.replace("_", " ").title()
            print(
                f"{thresh:>9.1f} | {label:>20} | {r.n_trades:>7} | {r.win_rate:>7.1%} | "
                f"{r.sharpe:>7.3f} | {r.sortino:>8.3f} | {r.profit_factor:>6.2f} | "
                f"{r.rr_ratio:>5.2f} | {r.total_pnl_ticks:>9.2f} | {r.total_pnl_usd:>10.2f} | "
                f"{r.coverage_vs_baseline:>6.1%}"
            )
        print(sep)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="PatchTST confluence filter test")
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()),
                        help="Number of parallel workers (default: 16 or cpu_count)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS,
                        help="Z-score thresholds to test")
    args = parser.parse_args()

    # Discover dates (intersection of both model dirs)
    cm_dates = {d.name for d in CM_DIR.iterdir() if d.is_dir() and (d / "predictions.npz").exists()}
    pt_dates = {d.name for d in PT_DIR.iterdir() if d.is_dir() and (d / "predictions.npz").exists()}
    dates = sorted(cm_dates & pt_dates)

    log.info(f"Found {len(dates)} common dates (CM={len(cm_dates)}, PT={len(pt_dates)})")
    log.info(f"Thresholds: {args.thresholds}")
    log.info(f"Workers: {args.workers}")
    log.info(f"NN max gap for alignment: {NN_MAX_GAP}")

    if len(dates) == 0:
        log.error("No common dates found. Check paths.")
        sys.exit(1)

    # Build work items: (date, threshold) for all combos
    work_items = [(d, t) for t in args.thresholds for d in dates]
    log.info(f"Total work items: {len(work_items)} ({len(dates)} dates x {len(args.thresholds)} thresholds)")

    # Process in parallel
    t0 = datetime.now()
    with Pool(processes=args.workers) as pool:
        raw_results = pool.map(process_date, work_items)
    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Processing complete in {elapsed:.1f}s")

    # Group by threshold
    by_threshold = defaultdict(list)
    skipped = 0
    for r in raw_results:
        if r is None:
            skipped += 1
            continue
        by_threshold[r["threshold"]].append(r)

    if skipped > 0:
        log.warning(f"Skipped {skipped} date/threshold combos (missing data or too few matches)")

    # Log alignment stats
    for thresh in sorted(by_threshold.keys()):
        match_counts = [r["n_matched"] for r in by_threshold[thresh]]
        log.info(
            f"Threshold {thresh:.1f}: {len(match_counts)} dates, "
            f"matched samples: mean={np.mean(match_counts):.0f}, "
            f"min={np.min(match_counts)}, max={np.max(match_counts)}"
        )

    # Aggregate
    all_results = {}
    for thresh in sorted(by_threshold.keys()):
        all_results[thresh] = aggregate_results(by_threshold[thresh], thresh)

    # Print summary
    print_summary_table(all_results)

    # ── Save full results ──
    output = {
        "metadata": {
            "generated": _ts,
            "cost_model": "MIDPOINT-BASED (no FIFO sim)",
            "cost_rt_ticks": COST_RT_TICKS,
            "cost_rt_usd": COST_RT_USD,
            "horizon": "10s",
            "nn_max_gap": NN_MAX_GAP,
            "pt_threshold_frac": PT_THRESHOLD_FRAC,
            "n_dates": len(dates),
            "dates": dates,
        },
        "results": {},
    }

    for thresh, strats in all_results.items():
        output["results"][str(thresh)] = {
            name: asdict(sr) for name, sr in strats.items()
        }

    out_path = OUTPUT_DIR / f"confluence_results_{_ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Full results saved to {out_path}")

    # ── Per-date CSV for easy inspection ──
    csv_path = OUTPUT_DIR / f"confluence_per_date_{_ts}.csv"
    with open(csv_path, "w") as f:
        f.write("threshold,strategy,date,n_trades,win_rate,total_pnl_ticks\n")
        for thresh, strats in all_results.items():
            for sname, sr in strats.items():
                for date, info in sorted(sr.per_date.items()):
                    f.write(
                        f"{thresh},{sname},{date},{info['n_trades']},"
                        f"{info['win_rate']},{info['total_pnl']}\n"
                    )
    log.info(f"Per-date CSV saved to {csv_path}")

    # ── Key finding ──
    print("\n── KEY FINDING ──")
    for thresh in sorted(all_results.keys()):
        b = all_results[thresh]["baseline"]
        c = all_results[thresh]["confluence"]
        sc = all_results[thresh]["strong_confluence"]
        ac = all_results[thresh]["anti_confluence"]

        better = c.sharpe > b.sharpe
        verdict = "IMPROVES" if better else "DOES NOT IMPROVE"
        print(
            f"z>{thresh:.1f}: Confluence {verdict} Sharpe "
            f"({b.sharpe:.3f} -> {c.sharpe:.3f}), "
            f"Sortino ({b.sortino:.3f} -> {c.sortino:.3f}), "
            f"coverage={c.coverage_vs_baseline:.0%}, "
            f"anti-conf Sharpe={ac.sharpe:.3f}"
        )

    print(f"\nOutput: {OUTPUT_DIR}")
    print("NOTE: All PnL figures are MIDPOINT-BASED. Requires FIFO sim for realistic execution.\n")


if __name__ == "__main__":
    main()
