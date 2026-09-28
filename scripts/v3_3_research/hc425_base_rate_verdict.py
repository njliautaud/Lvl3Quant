"""
hc425_base_rate_verdict.py — HC #425 follow-up.

Reads the 4 alt-label geometries × 5 OOT dates produced by the session #63 alt-label
generator and computes a base-rate verdict: fill rate, TP-hit rate, mean net ticks,
WR, hold-time distribution. Writes a single markdown verdict.

Output: output/hc425_alternative_labels/base_rate_verdict.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant/output/hc425_alternative_labels")
DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
GEOMETRIES = ["time60", "tp4sl4", "tp6sl2", "tp8sl3"]


def summarize_side(d, side: str) -> dict:
    """side ∈ {'long', 'short'}."""
    filled = d[f"{side}_filled"]
    gross = d[f"{side}_gross_ticks"].astype(np.float64)
    hit_tp = d[f"{side}_hit_tp"]
    hit_sl = d[f"{side}_hit_sl"]
    hit_time = d[f"{side}_hit_time"]
    reason = d[f"{side}_exit_reason"]

    n_total = int(filled.size)
    n_filled = int(filled.sum())
    if n_filled == 0:
        return {"n_signals": n_total, "n_filled": 0, "fill_rate": 0.0}
    g = gross[filled]
    return {
        "n_signals": n_total,
        "n_filled": n_filled,
        "fill_rate": float(n_filled) / max(n_total, 1),
        "mean_gross_ticks": float(g.mean()),
        "median_gross_ticks": float(np.median(g)),
        "wr_pct": float((g > 0).mean() * 100),
        "pct_hit_tp": float(hit_tp[filled].mean() * 100),
        "pct_hit_sl": float(hit_sl[filled].mean() * 100),
        "pct_hit_time": float(hit_time[filled].mean() * 100),
        "p_pos_tail_p75": float(np.percentile(g[g > 0], 75)) if (g > 0).any() else 0.0,
        "p_neg_tail_p25": float(np.percentile(g[g < 0], 25)) if (g < 0).any() else 0.0,
    }


def main() -> int:
    verdict = {"by_geometry": {}, "ranking_by_long": [], "ranking_by_short": []}
    for geom in GEOMETRIES:
        gdir = ROOT / geom
        if not gdir.exists():
            print(f"[hc425] missing dir: {gdir}")
            continue
        agg_long = {"n_signals": 0, "n_filled": 0, "gross_sum": 0.0, "wr_num": 0}
        agg_short = {"n_signals": 0, "n_filled": 0, "gross_sum": 0.0, "wr_num": 0}
        per_day = {}
        for date in DATES:
            f = gdir / f"{date}_alt_labels.npz"
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            ls = summarize_side(d, "long")
            ss = summarize_side(d, "short")
            per_day[date] = {"long": ls, "short": ss}
            agg_long["n_signals"] += ls["n_signals"]
            agg_long["n_filled"] += ls["n_filled"]
            if ls["n_filled"] > 0:
                agg_long["gross_sum"] += ls["mean_gross_ticks"] * ls["n_filled"]
                agg_long["wr_num"] += (ls["wr_pct"] / 100) * ls["n_filled"]
            agg_short["n_signals"] += ss["n_signals"]
            agg_short["n_filled"] += ss["n_filled"]
            if ss["n_filled"] > 0:
                agg_short["gross_sum"] += ss["mean_gross_ticks"] * ss["n_filled"]
                agg_short["wr_num"] += (ss["wr_pct"] / 100) * ss["n_filled"]
        long_mean = agg_long["gross_sum"] / max(agg_long["n_filled"], 1)
        short_mean = agg_short["gross_sum"] / max(agg_short["n_filled"], 1)
        long_wr = (agg_long["wr_num"] / max(agg_long["n_filled"], 1)) * 100
        short_wr = (agg_short["wr_num"] / max(agg_short["n_filled"], 1)) * 100
        verdict["by_geometry"][geom] = {
            "long_total": {
                "n_signals": agg_long["n_signals"],
                "n_filled": agg_long["n_filled"],
                "fill_rate": agg_long["n_filled"] / max(agg_long["n_signals"], 1),
                "mean_gross_ticks": long_mean,
                "wr_pct": long_wr,
            },
            "short_total": {
                "n_signals": agg_short["n_signals"],
                "n_filled": agg_short["n_filled"],
                "fill_rate": agg_short["n_filled"] / max(agg_short["n_signals"], 1),
                "mean_gross_ticks": short_mean,
                "wr_pct": short_wr,
            },
            "per_day": per_day,
        }
    # Rankings by edge-after-cost (commission 0.376t passive, 1.376t IOC; we'll use 0.376 as floor).
    COST = 0.376
    for side in ("long", "short"):
        ranks = []
        for geom, stats in verdict["by_geometry"].items():
            mg = stats[f"{side}_total"]["mean_gross_ticks"]
            nf = stats[f"{side}_total"]["n_filled"]
            ranks.append({"geom": geom, "mean_gross": mg, "n_filled": nf,
                          "net_after_passive_cost": mg - COST})
        ranks.sort(key=lambda r: r["net_after_passive_cost"], reverse=True)
        verdict[f"ranking_by_{side}"] = ranks

    out_json = ROOT / "base_rate_verdict.json"
    out_md = ROOT / "base_rate_verdict.md"
    out_json.write_text(json.dumps(verdict, indent=2, default=str))

    lines = []
    lines.append("# HC #425 Alternative-Label Base-Rate Verdict\n")
    lines.append("**Dataset**: 5 OOT days (2026-02-23 → 2026-02-27), 4 label geometries (time-stop and 3 TP/SL combos).\n")
    lines.append("**Cost floor**: 0.376t passive commission (HC #426 R4). Edge = mean_gross_ticks − 0.376.\n\n")
    lines.append("## Aggregate (5-day total)\n\n")
    lines.append("| Geometry | Side  | n_signals | n_filled | fill % | mean_gross | WR %  | edge_after_cost |\n")
    lines.append("|----------|-------|----------:|---------:|-------:|-----------:|------:|----------------:|\n")
    for geom in GEOMETRIES:
        st = verdict["by_geometry"].get(geom, {})
        for side in ("long", "short"):
            s = st.get(f"{side}_total", {})
            if not s:
                continue
            lines.append(
                f"| {geom} | {side:5s} | {s['n_signals']:>9d} | {s['n_filled']:>8d} | "
                f"{s['fill_rate']*100:>5.1f}% | {s['mean_gross_ticks']:>+10.3f}t | "
                f"{s['wr_pct']:>5.1f}% | {s['mean_gross_ticks']-0.376:>+15.3f}t |\n"
            )
    lines.append("\n## Ranking by net-of-passive-commission edge\n\n")
    for side in ("long", "short"):
        lines.append(f"### {side.upper()} side\n\n")
        lines.append("| rank | geometry | mean_gross | n_filled | net_after_passive_cost |\n")
        lines.append("|-----:|----------|-----------:|---------:|-----------------------:|\n")
        for i, r in enumerate(verdict[f"ranking_by_{side}"], 1):
            lines.append(f"| {i} | {r['geom']} | {r['mean_gross']:>+10.3f}t | {r['n_filled']:>8d} | {r['net_after_passive_cost']:>+22.3f}t |\n")
        lines.append("\n")

    out_md.write_text("".join(lines))
    print(f"[hc425] wrote {out_md}")
    print(f"[hc425] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
