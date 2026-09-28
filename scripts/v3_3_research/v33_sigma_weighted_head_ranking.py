#!/usr/bin/env python3
"""
HC #363 deliverable 2: v3.3 sigma-weighted head ranking.

Joins the v3.3 learned log_sigma uncertainty params (one per head, learned by
JointMultiHeadLossV33_UncertaintyWeighted during fold-0 training) with the
per-head dashboard's Sharpe/net metrics. Answers: "Does the model's confidence
(low sigma) match the heads that actually have FIFO Sharpe?"

Inputs:
  output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_sigma.json
  output/v3_3_full_execution_analysis_20260514/per_head_dashboard/per_head_master.csv

Output:
  output/v3_3_full_execution_analysis_20260514/sigma_weighted_ranking/
    sigma_x_sharpe.csv  (head, sigma, best_sharpe, best_band, best_side, confidence_rank, sharpe_rank, joint_rank)
    sigma_x_sharpe.md   (top-15 rankings + alignment commentary)
    ranking.json        (machine-readable)

Analysis tooling per HC #307D — NEW file, no trainer modifications.
"""
from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


ROOT = Path("/home/jupiter/Lvl3Quant")
SIGMA_JSON = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_sigma.json"
MASTER_CSV = ROOT / "output/v3_3_full_execution_analysis_20260514/per_head_dashboard/per_head_master.csv"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/sigma_weighted_ranking"


def safe_float(x: str) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("-inf")
        return v
    except (TypeError, ValueError):
        return float("-inf")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SIGMA_JSON.exists():
        print(f"ERR: sigma file missing: {SIGMA_JSON}", file=sys.stderr)
        return 1
    if not MASTER_CSV.exists():
        print(f"ERR: per-head master missing: {MASTER_CSV}", file=sys.stderr)
        return 1

    sigma_map = json.loads(SIGMA_JSON.read_text())
    print(f"Loaded sigma for {len(sigma_map)} heads")

    # Find best (head, side, band) cell per head by Sharpe with n_fills>=30
    per_head_best: dict[str, dict] = {}
    n_rows = 0
    with open(MASTER_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            head = row.get("head", "").strip()
            if not head:
                continue
            try:
                n_fills = int(float(row.get("fifo_tp4sl3_n_fills", "0") or 0))
            except (TypeError, ValueError):
                n_fills = 0
            if n_fills < 30:
                continue
            sharpe = safe_float(row.get("fifo_tp4sl3_sharpe", "nan"))
            if sharpe == float("-inf"):
                continue
            cur = per_head_best.get(head)
            if cur is None or sharpe > cur["sharpe"]:
                per_head_best[head] = {
                    "head": head,
                    "side": row.get("side", ""),
                    "band": row.get("band", ""),
                    "n_fills": n_fills,
                    "sharpe": sharpe,
                    "net_t_per_fill": safe_float(row.get("fifo_tp4sl3_net_mean_ticks", "nan")),
                    "passive_net": safe_float(row.get("passive_net_after_comm", "nan")),
                    "day_concentration": safe_float(row.get("per_day_concentration", "nan")),
                    "ci_low": safe_float(row.get("ci_low_95", "nan")),
                }
    print(f"Master CSV: {n_rows} rows, {len(per_head_best)} heads with ≥1 cell n_fills>=30")

    # Build joined rows
    joined = []
    for head, sigma_info in sigma_map.items():
        sigma = float(sigma_info["sigma"])
        best = per_head_best.get(head)
        joined.append({
            "head": head,
            "sigma": sigma,
            "log_sigma": float(sigma_info["log_sigma"]),
            "confidence": 1.0 / max(sigma, 1e-6),
            "best_sharpe": best["sharpe"] if best else float("-inf"),
            "best_side": best["side"] if best else "",
            "best_band": best["band"] if best else "",
            "best_n_fills": best["n_fills"] if best else 0,
            "best_net_t_per_fill": best["net_t_per_fill"] if best else float("-inf"),
            "best_passive_net": best["passive_net"] if best else float("-inf"),
            "best_day_conc": best["day_concentration"] if best else float("-inf"),
            "best_ci_low": best["ci_low"] if best else float("-inf"),
        })

    # Rankings
    by_conf = sorted(joined, key=lambda r: -r["confidence"])
    by_sharpe = sorted(joined, key=lambda r: -r["best_sharpe"])
    conf_rank = {r["head"]: i + 1 for i, r in enumerate(by_conf)}
    sharpe_rank = {r["head"]: i + 1 for i, r in enumerate(by_sharpe)}

    # Joint score: sharpe / sigma. Heads with -inf sharpe stay last.
    for r in joined:
        s = r["best_sharpe"]
        if s == float("-inf"):
            r["joint_score"] = float("-inf")
        else:
            r["joint_score"] = s / max(r["sigma"], 1e-6)
        r["conf_rank"] = conf_rank[r["head"]]
        r["sharpe_rank"] = sharpe_rank[r["head"]]
    by_joint = sorted(joined, key=lambda r: -r["joint_score"])
    for i, r in enumerate(by_joint):
        r["joint_rank"] = i + 1

    # Alignment correlation (rank-rank)
    # Spearman: rho = 1 - 6*sum(d^2) / (n*(n^2-1))
    n = len(joined)
    d2 = 0
    valid = [r for r in joined if r["best_sharpe"] != float("-inf")]
    nv = len(valid)
    if nv >= 3:
        # Re-rank within valid set
        v_by_conf = sorted(valid, key=lambda r: -r["confidence"])
        v_by_sharpe = sorted(valid, key=lambda r: -r["best_sharpe"])
        v_conf_rank = {r["head"]: i + 1 for i, r in enumerate(v_by_conf)}
        v_sharpe_rank = {r["head"]: i + 1 for i, r in enumerate(v_by_sharpe)}
        d2 = sum((v_conf_rank[r["head"]] - v_sharpe_rank[r["head"]]) ** 2 for r in valid)
        spearman = 1 - 6 * d2 / (nv * (nv * nv - 1))
    else:
        spearman = float("nan")

    # Write CSV
    csv_path = OUT_DIR / "sigma_x_sharpe.csv"
    cols = [
        "head", "sigma", "log_sigma", "confidence", "conf_rank",
        "best_side", "best_band", "best_n_fills",
        "best_sharpe", "sharpe_rank",
        "best_net_t_per_fill", "best_passive_net", "best_day_conc", "best_ci_low",
        "joint_score", "joint_rank",
    ]
    by_conf_sorted = sorted(joined, key=lambda r: r["conf_rank"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in by_conf_sorted:
            row = {c: r.get(c, "") for c in cols}
            # Replace -inf with "" for readability
            for c in cols:
                if row[c] == float("-inf"):
                    row[c] = ""
                elif isinstance(row[c], float):
                    row[c] = f"{row[c]:.4f}"
            w.writerow(row)
    print(f"Wrote {csv_path}")

    # JSON
    json_path = OUT_DIR / "ranking.json"
    json_path.write_text(json.dumps({
        "n_heads": n,
        "n_valid_for_spearman": nv,
        "spearman_confidence_vs_sharpe": spearman,
        "top10_by_confidence": [r["head"] for r in by_conf[:10]],
        "top10_by_sharpe": [r["head"] for r in by_sharpe[:10]],
        "top10_by_joint": [r["head"] for r in by_joint[:10]],
        "rows": [
            {**{k: (v if v != float("-inf") else None) for k, v in r.items()}}
            for r in by_conf_sorted
        ],
    }, indent=2, default=str))
    print(f"Wrote {json_path}")

    # Markdown summary
    md_path = OUT_DIR / "sigma_x_sharpe.md"
    lines = []
    lines.append("# v3.3 σ-WEIGHTED HEAD RANKING — HC #363 deliverable 2\n")
    lines.append(f"Source: `{SIGMA_JSON.name}` (32 learned σ params from fold-0 epoch=4 batch=22000)")
    lines.append(f"Joined with: `{MASTER_CSV.name}` (per-head best (side, band) cell by Sharpe, n_fills ≥ 30)\n")
    lines.append("## Alignment\n")
    lines.append(f"- Spearman ρ(confidence rank, Sharpe rank) over {nv} heads with valid Sharpe = **{spearman:.3f}**")
    if not math.isnan(spearman):
        if spearman > 0.5:
            lines.append("- Strong positive alignment — model's σ-confidence agrees with FIFO Sharpe ranking.")
        elif spearman > 0.2:
            lines.append("- Moderate positive alignment — σ-confidence partially predicts FIFO Sharpe.")
        elif spearman > -0.2:
            lines.append("- Weak alignment — σ-confidence and FIFO Sharpe are largely independent. σ may reflect TRAINING residual variance, not deployment edge.")
        else:
            lines.append("- NEGATIVE alignment — heads the model is most confident about are NOT the ones with FIFO Sharpe. Use Sharpe ranking for execution, ignore σ.")
    lines.append("")

    lines.append("## Top 15 by σ-confidence (lowest σ first — model's 'most trustworthy' heads)\n")
    lines.append("| Rank | Head | σ | Best (side, band, n) | FIFO Sharpe | Net t/fill | Passive net | Day-conc |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in by_conf[:15]:
        bs = r["best_sharpe"]
        bs_str = f"{bs:.3f}" if bs != float("-inf") else "n/a"
        nt = r["best_net_t_per_fill"]
        nt_str = f"{nt:.3f}" if nt != float("-inf") else "n/a"
        pn = r["best_passive_net"]
        pn_str = f"{pn:.3f}" if pn != float("-inf") else "n/a"
        dc = r["best_day_conc"]
        dc_str = f"{dc*100:.0f}%" if dc != float("-inf") else "n/a"
        cell = f"({r['best_side']}, {r['best_band']}, n={r['best_n_fills']})" if r["best_side"] else "no cells ≥30 fills"
        lines.append(f"| {r['conf_rank']} | {r['head']} | {r['sigma']:.4f} | {cell} | {bs_str} | {nt_str} | {pn_str} | {dc_str} |")
    lines.append("")

    lines.append("## Top 15 by FIFO Sharpe (best operational cell)\n")
    lines.append("| Rank | Head | σ | Best (side, band, n) | FIFO Sharpe | Joint score (Sharpe/σ) |")
    lines.append("|---|---|---|---|---|---|")
    for r in by_sharpe[:15]:
        bs = r["best_sharpe"]
        if bs == float("-inf"):
            break
        js = r["joint_score"]
        js_str = f"{js:.3f}" if js != float("-inf") else "n/a"
        cell = f"({r['best_side']}, {r['best_band']}, n={r['best_n_fills']})"
        lines.append(f"| {r['sharpe_rank']} | {r['head']} | {r['sigma']:.4f} | {cell} | {bs:.3f} | {js_str} |")
    lines.append("")

    lines.append("## Top 15 by JOINT score (Sharpe / σ — combined confidence × performance)\n")
    lines.append("| Rank | Head | σ | Best (side, band, n) | FIFO Sharpe | Joint score |")
    lines.append("|---|---|---|---|---|---|")
    for r in by_joint[:15]:
        if r["joint_score"] == float("-inf"):
            break
        cell = f"({r['best_side']}, {r['best_band']}, n={r['best_n_fills']})"
        lines.append(f"| {r['joint_rank']} | {r['head']} | {r['sigma']:.4f} | {cell} | {r['best_sharpe']:.3f} | {r['joint_score']:.3f} |")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    print(f"\nSPEARMAN ρ(confidence, Sharpe) = {spearman:.4f} over {nv} heads")
    print(f"Top conf head: {by_conf[0]['head']} (σ={by_conf[0]['sigma']:.4f})")
    print(f"Top Sharpe head: {by_sharpe[0]['head']} (Sharpe={by_sharpe[0]['best_sharpe']:.3f})")
    print(f"Top joint head: {by_joint[0]['head']} (score={by_joint[0]['joint_score']:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
