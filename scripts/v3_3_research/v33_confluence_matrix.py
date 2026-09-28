#!/usr/bin/env python3
"""
HC #363 deliverable 3: v3.3 head-pair confluence matrix.

Tests whether SIMULTANEOUS signals from multiple heads carry MORE FIFO edge
than each head alone. For every (head_i, head_j) pair at a fixed confidence
band (default SHORT Top1%), computes Sharpe + net t/fill on the intersection
of samples where BOTH heads are in their top-band on the same side.

Inputs:
  output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz

Outputs:
  output/v3_3_full_execution_analysis_20260514/confluence_matrix/
    confluence_short_top1pct.csv      (head_i, head_j, n_both, sharpe, net_t, lift_vs_max_solo)
    confluence_short_top1pct.md       (top-30 pairs)
    confluence_long_top1pct.csv
    confluence_long_top1pct.md
    matrix_short_top1pct.json         (32×32 sharpe + n_both grids)
    matrix_long_top1pct.json

Per HC #307D — NEW analysis script, no trainer mods.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/confluence_matrix"
BAND_TOP_FRAC = 0.01  # Top 1% per side per head
MIN_N_BOTH = 20      # require ≥20 co-signaled samples
COMMISSION_TICKS = 0.376  # ES round-trip commission cost


# Heads where HIGH prediction → SHORT signal (negative-direction sensitive)
# All other heads: HIGH prediction → LONG signal.
SHORT_HIGH_HEADS = {"p_reversal_15s", "p_reversal_30s", "p_reversal_60s"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PRED_NPZ.exists():
        print(f"ERR: predictions npz missing: {PRED_NPZ}", file=sys.stderr)
        return 1

    print(f"Loading {PRED_NPZ}")
    npz = np.load(PRED_NPZ, allow_pickle=True)
    keys = list(npz.keys())
    pred_keys = sorted([k for k in keys if k.startswith("pred_") and k != "pred_meta"])
    # Strip pred_ prefix
    heads = [k[len("pred_"):] for k in pred_keys]
    print(f"Found {len(heads)} prediction heads")

    target_fifo_net = None
    target_fifo_mask = None
    for cand in ("target_fifo_tp4sl3_net",):
        if cand in keys:
            target_fifo_net = npz[cand]
        cand_mask = cand.replace("target_", "mask_")
        if cand_mask in keys:
            target_fifo_mask = npz[cand_mask]
    if target_fifo_net is None:
        print("ERR: target_fifo_tp4sl3_net not found in NPZ", file=sys.stderr)
        return 1

    target_fifo_net = target_fifo_net.astype(np.float64)
    fifo_mask = (target_fifo_mask.astype(bool) if target_fifo_mask is not None
                 else ~np.isnan(target_fifo_net))
    n = target_fifo_net.shape[0]
    print(f"Samples={n:,}, fifo fillable={int(fifo_mask.sum()):,} ({100*fifo_mask.mean():.1f}%)")

    # Per-head SHORT-band + LONG-band membership masks (top X% per side).
    # Convention: SHORT band = lowest predictions for "high-means-long" heads,
    # highest predictions for SHORT_HIGH heads (e.g. p_reversal where high
    # means reversal-down is likely).
    short_in = {}
    long_in = {}
    solo_short = {}  # head -> {n, sharpe, net}
    solo_long = {}

    for h in heads:
        pred = npz[f"pred_{h}"].astype(np.float64)
        # Only consider samples with fifo_mask (these can be replayed)
        valid_pred = np.where(fifo_mask, pred, np.nan)
        # Threshold from fillable samples only
        finite = ~np.isnan(valid_pred)
        if finite.sum() < 100:
            continue
        if h in SHORT_HIGH_HEADS:
            short_thr = np.nanquantile(valid_pred, 1 - BAND_TOP_FRAC)
            long_thr = np.nanquantile(valid_pred, BAND_TOP_FRAC)
            s_mask = finite & (valid_pred >= short_thr)
            l_mask = finite & (valid_pred <= long_thr)
        else:
            short_thr = np.nanquantile(valid_pred, BAND_TOP_FRAC)
            long_thr = np.nanquantile(valid_pred, 1 - BAND_TOP_FRAC)
            s_mask = finite & (valid_pred <= short_thr)
            l_mask = finite & (valid_pred >= long_thr)
        short_in[h] = s_mask
        long_in[h] = l_mask

        # Solo Sharpe (passive: SHORT side flips sign, both pay commission)
        if s_mask.sum() >= 10:
            r = -target_fifo_net[s_mask] - COMMISSION_TICKS
            mu = float(np.mean(r))
            sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
            sh = mu / sd if sd > 1e-9 else 0.0
            solo_short[h] = {"n": int(s_mask.sum()), "sharpe": sh, "net_t": mu}
        if l_mask.sum() >= 10:
            r = target_fifo_net[l_mask] - COMMISSION_TICKS
            mu = float(np.mean(r))
            sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
            sh = mu / sd if sd > 1e-9 else 0.0
            solo_long[h] = {"n": int(l_mask.sum()), "sharpe": sh, "net_t": mu}

    eligible = [h for h in heads if h in short_in and h in long_in]
    print(f"Eligible heads (with both SHORT & LONG bands): {len(eligible)}")
    print(f"Solo SHORT cells: {len(solo_short)}  Solo LONG cells: {len(solo_long)}")

    def build(side: str):
        in_map = short_in if side == "SHORT" else long_in
        solo = solo_short if side == "SHORT" else solo_long
        rows = []
        H = len(eligible)
        sharpe_grid = np.full((H, H), np.nan, dtype=np.float64)
        n_grid = np.zeros((H, H), dtype=np.int64)
        for i, hi in enumerate(eligible):
            mi = in_map.get(hi)
            if mi is None:
                continue
            for j, hj in enumerate(eligible):
                if j < i:
                    continue
                mj = in_map.get(hj)
                if mj is None:
                    continue
                both = mi & mj
                nb = int(both.sum())
                n_grid[i, j] = nb
                n_grid[j, i] = nb
                if nb < MIN_N_BOTH:
                    continue
                if side == "SHORT":
                    r = -target_fifo_net[both] - COMMISSION_TICKS
                else:
                    r = target_fifo_net[both] - COMMISSION_TICKS
                mu = float(np.mean(r))
                sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
                sh = mu / sd if sd > 1e-9 else 0.0
                sharpe_grid[i, j] = sh
                sharpe_grid[j, i] = sh
                solo_i = solo.get(hi, {}).get("sharpe")
                solo_j = solo.get(hj, {}).get("sharpe")
                best_solo = max([s for s in [solo_i, solo_j] if s is not None], default=None)
                lift = sh - best_solo if best_solo is not None else None
                rows.append({
                    "head_i": hi, "head_j": hj, "n_both": nb,
                    "joint_sharpe": sh, "joint_net_t": mu,
                    "solo_sharpe_i": solo_i, "solo_sharpe_j": solo_j,
                    "lift_vs_best_solo": lift,
                })

        # Sort by joint sharpe desc
        rows.sort(key=lambda r: -(r["joint_sharpe"]))
        return rows, sharpe_grid, n_grid

    for side in ("SHORT", "LONG"):
        print(f"\n=== Building {side} matrix ===")
        rows, sharpe_grid, n_grid = build(side)
        tag = f"{side.lower()}_top{int(BAND_TOP_FRAC*100*10)/10}pct".replace(".0", "")

        # CSV
        import csv
        csv_path = OUT_DIR / f"confluence_{tag}.csv"
        cols = ["head_i", "head_j", "n_both", "joint_sharpe", "joint_net_t",
                "solo_sharpe_i", "solo_sharpe_j", "lift_vs_best_solo"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (f"{r[c]:.4f}" if isinstance(r[c], float) and r[c] is not None
                                else (r[c] if r[c] is not None else "")) for c in cols})
        print(f"Wrote {csv_path} ({len(rows)} pairs)")

        # Markdown summary (top 30 cross-head pairs, i!=j)
        cross_rows = [r for r in rows if r["head_i"] != r["head_j"]]
        md_path = OUT_DIR / f"confluence_{tag}.md"
        lines = []
        lines.append(f"# v3.3 CONFLUENCE MATRIX — {side} Top {BAND_TOP_FRAC*100:.1f}% per head\n")
        lines.append(f"Source: `{PRED_NPZ.name}` (5 OOT days, 241,351 events, 32 heads)")
        lines.append(f"Min n_both: {MIN_N_BOTH}.  Commission ticks: {COMMISSION_TICKS}.  Eligible heads: {len(eligible)}\n")
        lines.append(f"## Top 30 cross-head pairs by joint passive Sharpe\n")
        lines.append("| Pair | n_both | Joint Sharpe | Joint net t | Solo i | Solo j | Lift vs best solo |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in cross_rows[:30]:
            si = f"{r['solo_sharpe_i']:.3f}" if r['solo_sharpe_i'] is not None else "n/a"
            sj = f"{r['solo_sharpe_j']:.3f}" if r['solo_sharpe_j'] is not None else "n/a"
            lift = f"{r['lift_vs_best_solo']:+.3f}" if r['lift_vs_best_solo'] is not None else "n/a"
            lines.append(f"| {r['head_i']} ∧ {r['head_j']} | {r['n_both']} | {r['joint_sharpe']:.3f} | {r['joint_net_t']:.3f} | {si} | {sj} | {lift} |")
        lines.append("")
        # Solo reference for context
        lines.append(f"## Solo {side} Sharpe (Top {BAND_TOP_FRAC*100:.1f}% reference)\n")
        lines.append("| Head | n | Solo Sharpe | Solo net t |")
        lines.append("|---|---|---|---|")
        solo = solo_short if side == "SHORT" else solo_long
        for h, info in sorted(solo.items(), key=lambda kv: -kv[1]["sharpe"])[:20]:
            lines.append(f"| {h} | {info['n']} | {info['sharpe']:.3f} | {info['net_t']:.3f} |")
        md_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {md_path}")

        # JSON grid
        json_path = OUT_DIR / f"matrix_{tag}.json"
        json_path.write_text(json.dumps({
            "side": side,
            "band_top_frac": BAND_TOP_FRAC,
            "min_n_both": MIN_N_BOTH,
            "commission_ticks": COMMISSION_TICKS,
            "heads": eligible,
            "sharpe_grid": [[None if np.isnan(x) else float(x) for x in row] for row in sharpe_grid],
            "n_both_grid": n_grid.tolist(),
            "solo": {h: solo.get(h) for h in eligible},
        }, indent=2))
        print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
