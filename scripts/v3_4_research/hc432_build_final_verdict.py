#!/usr/bin/env python3
"""
HC #432 — Build a combined final verdict markdown + leaderboard JSON across
all 5 candidates (4 v3.4.2 configs + ensemble + optional v2 baseline) using
each candidate's *_summary.json.

Output:
  HC432_FINAL_VERDICT.md       human-readable verdict + leaderboard table
  HC432_FINAL_LEADERBOARD.json machine-readable leaderboard rows

Failure-tolerant: missing summaries become explicit rows tagged "missing".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# config_name -> human-readable display name
CANDIDATES = [
    ("v342_long_1s_top0.5",                  "v3.4.2 1s long top-0.5% (existing)"),
    ("v342_short_5s_top0.5_t1422_R2fix",     "v3.4.2 5s short top-0.5% (t1422 R2-fixed)"),
    ("v342_short_10s_top0.5_t2831_R2fix",    "v3.4.2 10s short top-0.5% (t2831 R2-fixed)"),
    ("v342_long_5s_top0.5_for_ensemble",     "v3.4.2 5s long top-0.5% (ensemble leg)"),
    ("v342_lshort_5s_ensemble_50_50",        "5s long/short 50/50 ensemble"),
    ("v2_short_1s_top0.5_baseline",          "v2 1s short top-0.5% (sanity baseline)"),
]


def load_summary(out_dir: Path, cfg: str) -> Optional[dict]:
    p = out_dir / f"{cfg}_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def row_from_summary(cfg: str, display: str, s: Optional[dict]) -> Dict:
    if s is None:
        return {"config_name": cfg, "display_name": display, "missing": True,
                "n_fills": 0, "mean_tk": 0.0, "Sharpe_sqrtN": 0.0,
                "PF": 0.0, "WR": 0.0, "day_conc": 0.0,
                "r1_ratio": 0.0, "r1_pass": False, "r2_pass": False,
                "overall_pass": False, "n_dates_present": 0}
    o = s.get("overall", {})
    r1 = s.get("r1", {})
    r2 = s.get("r2", {})
    overall_pass = bool(s.get("combined_pass", False))
    return {
        "config_name": cfg,
        "display_name": display,
        "missing": False,
        "preliminary": bool(s.get("preliminary", False)),
        "n_dates_present": int(s.get("n_dates_present", 0)),
        "n_fills": int(o.get("n", 0)),
        "mean_tk": float(o.get("mean_tk", 0.0)),
        "sum_tk": float(o.get("sum_tk", 0.0)),
        "Sharpe_sqrtN": float(o.get("Sharpe_sqrtN", 0.0)),
        "Sortino": float(o.get("Sortino", 0.0)),
        "PF": float(o.get("PF", 0.0)),
        "WR": float(o.get("WR", 0.0)),
        "day_conc": float(r1.get("day_conc", 0.0)),
        "r1_ratio": float(r1.get("ratio", 0.0)),
        "r1_pass": bool(r1.get("pass", False)),
        "r2_pass": bool(r2.get("pass", False)),
        "overall_pass": overall_pass,
    }


def fmt(v: float, p: int = 3) -> str:
    if v != v:  # NaN
        return "n/a"
    if v == float("inf"):
        return "inf"
    return f"{v:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for cfg, display in CANDIDATES:
        s = load_summary(out_dir, cfg)
        rows.append(row_from_summary(cfg, display, s))

    # Detect v2 baseline status for Discord blurb
    v2 = next((r for r in rows if r["config_name"] == "v2_short_1s_top0.5_baseline"), None)
    v2_note = None
    if v2 is None or v2.get("missing"):
        v2_note = ("v2 sanity baseline did not run on this pass (separate v2 harness "
                   "not yet built). The v3.4.2 harness self-consistency is supported "
                   "by the existing long-1s run reproducing per-day metrics; treat "
                   "results below as preliminary until v2 closes the loop.")
    else:
        expected_v2 = 0.274
        got = v2["mean_tk"]
        gap = got - expected_v2
        if abs(gap) <= 0.05:
            v2_note = (f"v2 baseline reproduced within tolerance: got {got:+.3f} tk/fill "
                       f"vs HC #413 expected +{expected_v2:.3f} tk/fill (gap {gap:+.3f}). "
                       f"Harness is trusted.")
        else:
            v2_note = (f"v2 baseline DID NOT reproduce: got {got:+.3f} tk/fill vs "
                       f"HC #413 expected +{expected_v2:.3f} tk/fill (gap {gap:+.3f}). "
                       f"Harness may have a bug — review FIFO replay before trusting "
                       f"v3.4.2 verdicts.")

    leaderboard = {
        "rows": rows,
        "v2_baseline_note": v2_note,
    }
    (out_dir / "HC432_FINAL_LEADERBOARD.json").write_text(json.dumps(leaderboard, indent=2))

    # Build markdown
    md = []
    md.append("# HC #432 Final Verdict — 5-Candidate 47-Day Validation")
    md.append("")
    md.append("FIFO market replay (HC #74) + regime stratification (HC #428 R1) + "
              "MFE-within-horizon (HC #428 R2). All metrics on realized fills only.")
    md.append("")
    md.append("## Leaderboard")
    md.append("")
    md.append("| config | n_fills | net_tk/fill | Sharpe(sqrt-N) | PF | WR | day_conc | R1 ratio | R1 | R2 | OVERALL |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|")
    for r in rows:
        if r["missing"]:
            md.append(f"| {r['display_name']} | — | — | — | — | — | — | — | — | — | MISSING |")
            continue
        md.append(
            f"| {r['display_name']} | {r['n_fills']:,} | "
            f"{r['mean_tk']:+.3f} | {r['Sharpe_sqrtN']:+.2f} | "
            f"{fmt(r['PF'], 2)} | {r['WR']:.1f}% | "
            f"{r['day_conc']:.2f} | {r['r1_ratio']:.2f} | "
            f"{'P' if r['r1_pass'] else 'F'} | {'P' if r['r2_pass'] else 'F'} | "
            f"{'PASS' if r['overall_pass'] else 'FAIL'} |"
        )
    md.append("")
    md.append("## Gates")
    md.append("- **R1 (HC #428):** |Sh_green − Sh_red| / max ≤ 0.50 AND day_conc ≤ 0.70")
    md.append("- **R2 (HC #428):** TP ≤ p90(MFE@horizon), hold ≤ 1.5×h, cancel ≤ h")
    md.append("- **OVERALL:** R1 PASS and R2 PASS and Sharpe(sqrt-N) > 0")
    md.append("")
    if v2_note:
        md.append("## Harness sanity")
        md.append(v2_note)
        md.append("")
    md.append("## Per-candidate verdict files")
    for r in rows:
        if r["missing"]:
            md.append(f"- {r['display_name']}: (no summary found)")
        else:
            md.append(f"- {r['display_name']}: `{r['config_name']}_verdict.md`")
    md.append("")
    (out_dir / "HC432_FINAL_VERDICT.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_dir / 'HC432_FINAL_VERDICT.md'}")
    print(f"wrote {out_dir / 'HC432_FINAL_LEADERBOARD.json'}")


if __name__ == "__main__":
    main()
