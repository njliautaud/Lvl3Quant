"""
v33_config_search.py — HC #357/#358/#363/#368 candidate-config sweep.

WHAT:
  Consumes HC #363 v3.3 full execution analysis outputs at
  /home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/
  and emits ranked candidate framework configs into
  live_trading/v3_3_deploy_package/configs/candidate_configs/.

EACH CANDIDATE = a {head, side, band, order_type, time_exit, mfe_target, mae_stop}
combination that survived HC #357 full-cost stack (FIFO + queue-position-on-arrival
+ adverse selection + commission + cancel/replace).

RANKING:
  Primary: hc357_sharpe   (full-cost basis)
  Secondary: hc357_net    (net ticks / fill, full cost)
  Tiebreak: day_conc DESC (more diversified days first), then ci_low_95 (lower bound)

GATE (deploy-eligible only):
  - hc357_sharpe >= 0.50   (HC #344 PERF-GATING)
  - hc357_net > 0
  - n_fills >= 30          (statistical floor; weakest gate)
  - day_conc <= 0.95       (don't deploy a config that only fires on one day)
  - ci_low_95 > -0.5       (lower bound not catastrophic)
  - side == "SHORT"        (HC #354/#363 short-side > long-side evidence)

OUTPUT:
  configs/candidate_configs/<rank_NN>_<head>_<band>_<exit>s.json
    Each is a fully populated framework_config.json variant (copy of template
    with head_selection, entry_logic, exit_logic overridden).
  config_search_report.md  ← human-readable ranked summary

Authorized: HC #368. Read-only against HC #363 analysis outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Gates ---------------------------------------------------------------------
MIN_SHARPE_HC357 = 0.50
MIN_NET_TICKS = 0.0
MIN_N_FILLS = 30
MAX_DAY_CONC = 0.95
MIN_CI_LOW_95 = -0.5
REQUIRED_SIDE = "SHORT"

# Exit defaults per band (HC #363) ------------------------------------------
BAND_TO_TIME_EXIT = {
    "Top0.1%": 5,
    "Top0.5%": 7,
    "Top1%": 10,
    "Top5%": 15,
    "Top10%": 20,
}
BAND_TO_MFE = {
    "Top0.1%": 2.5,
    "Top0.5%": 2.0,
    "Top1%": 1.8,
    "Top5%": 1.5,
    "Top10%": 1.2,
}
BAND_TO_MAE = {
    "Top0.1%": -2.0,
    "Top0.5%": -2.0,
    "Top1%": -2.0,
    "Top5%": -2.5,
    "Top10%": -2.5,
}

# Percentile floor mapping --------------------------------------------------
BAND_TO_PCTILE = {
    "Top0.1%": 99.9,
    "Top0.5%": 99.5,
    "Top1%": 99.0,
    "Top5%": 95.0,
    "Top10%": 90.0,
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", s)


def _load_hc357_rankings(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                rows.append({
                    "head": r["head"],
                    "side": r["side"],
                    "band": r["band"],
                    "n_fills": int(r["n_fills"]),
                    "fifo_sharpe": float(r["fifo_sharpe"]),
                    "fifo_net_mean": float(r["fifo_net_mean"]),
                    "passive_net_after_comm": float(r["passive_net_after_comm"]),
                    "hc357_net": float(r["hc357_net"]),
                    "hc357_sharpe": float(r["hc357_sharpe"]),
                    "delta_net": float(r["delta_net"]),
                    "delta_sharpe": float(r["delta_sharpe"]),
                    "day_conc": float(r["day_conc"]),
                    "ci_low_95": float(r["ci_low_95"]),
                })
            except (KeyError, ValueError) as e:
                print(f"[warn] skip row {r}: {e}", file=sys.stderr)
    return rows


def _passes_gate(row: Dict) -> bool:
    return (
        row["side"] == REQUIRED_SIDE
        and row["hc357_sharpe"] >= MIN_SHARPE_HC357
        and row["hc357_net"] > MIN_NET_TICKS
        and row["n_fills"] >= MIN_N_FILLS
        and row["day_conc"] <= MAX_DAY_CONC
        and row["ci_low_95"] > MIN_CI_LOW_95
    )


def _gate_failure_reasons(row: Dict) -> List[str]:
    reasons = []
    if row["side"] != REQUIRED_SIDE:
        reasons.append(f"side={row['side']} (need SHORT)")
    if row["hc357_sharpe"] < MIN_SHARPE_HC357:
        reasons.append(f"hc357_sharpe={row['hc357_sharpe']:.3f} < {MIN_SHARPE_HC357}")
    if row["hc357_net"] <= MIN_NET_TICKS:
        reasons.append(f"hc357_net={row['hc357_net']:.3f} <= 0")
    if row["n_fills"] < MIN_N_FILLS:
        reasons.append(f"n_fills={row['n_fills']} < {MIN_N_FILLS}")
    if row["day_conc"] > MAX_DAY_CONC:
        reasons.append(f"day_conc={row['day_conc']:.3f} > {MAX_DAY_CONC}")
    if row["ci_low_95"] <= MIN_CI_LOW_95:
        reasons.append(f"ci_low_95={row['ci_low_95']:.3f} <= {MIN_CI_LOW_95}")
    return reasons


def _build_candidate_config(template: Dict, row: Dict, rank: int) -> Dict:
    cfg = json.loads(json.dumps(template))  # deep copy
    band = row["band"]

    # Head selection: this row's head is primary
    cfg["head_selection"]["primary_heads"] = [row["head"]]
    cfg["head_selection"]["_comment_hc363"] = (
        f"From HC #363 hc357 ranking: rank={rank}, hc357_sharpe={row['hc357_sharpe']:.3f}, "
        f"hc357_net={row['hc357_net']:.3f}t/fill over {row['n_fills']} fills, "
        f"day_conc={row['day_conc']:.3f}, ci_low_95={row['ci_low_95']:.3f}"
    )

    # Entry logic: side + band → percentile floor
    pctile = BAND_TO_PCTILE.get(band, 99.0)
    cfg["entry_logic"]["order_type"] = "passive_limit"
    cfg["entry_logic"]["passive_offset_ticks"] = 0
    cfg["entry_logic"]["side_bias"] = "short_only"
    cfg["entry_logic"]["suppress_long"] = True
    cfg["entry_logic"]["suppress_short"] = False
    cfg["entry_logic"]["min_percentile_1s"] = pctile
    cfg["entry_logic"]["min_percentile_5s"] = pctile
    cfg["entry_logic"]["min_percentile_10s"] = pctile

    # Exit logic from band
    cfg["exit_logic"]["mfe_target_ticks"] = BAND_TO_MFE.get(band, 2.0)
    cfg["exit_logic"]["mae_stop_ticks"] = BAND_TO_MAE.get(band, -2.5)
    cfg["exit_logic"]["time_exit_seconds"] = BAND_TO_TIME_EXIT.get(band, 10)
    cfg["exit_logic"]["time_exit_seconds_for_top1pct"] = BAND_TO_TIME_EXIT.get("Top1%", 10)

    # Annotate
    cfg["_meta"]["candidate_rank"] = rank
    cfg["_meta"]["hc357_row"] = row
    cfg["_meta"]["pctile_floor"] = pctile
    return cfg


def _render_report(passed: List[Dict], rejected: List[Dict], out_dir: Path) -> str:
    md = ["# v3.3 Config Search Report — HC #357/#363/#368", ""]
    md.append(f"- Output dir: `{out_dir}`")
    md.append(f"- Gate: side=SHORT, hc357_sharpe ≥ {MIN_SHARPE_HC357}, hc357_net > 0, "
              f"n_fills ≥ {MIN_N_FILLS}, day_conc ≤ {MAX_DAY_CONC}, ci_low_95 > {MIN_CI_LOW_95}")
    md.append("")
    md.append(f"## Deploy-eligible candidates ({len(passed)})")
    md.append("")
    md.append("| Rank | Head | Band | n_fills | hc357_sharpe | hc357_net | day_conc | ci_low_95 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(passed, 1):
        md.append(
            f"| {i} | {r['head']} | {r['band']} | {r['n_fills']} | "
            f"{r['hc357_sharpe']:.3f} | {r['hc357_net']:.3f} | "
            f"{r['day_conc']:.3f} | {r['ci_low_95']:.3f} |"
        )
    md.append("")
    md.append(f"## Rejected ({len(rejected)})")
    md.append("")
    md.append("| Head | Side | Band | Reasons |")
    md.append("|---|---|---|---|")
    for r in rejected[:40]:
        reasons = ", ".join(_gate_failure_reasons(r))
        md.append(f"| {r['head']} | {r['side']} | {r['band']} | {reasons} |")
    if len(rejected) > 40:
        md.append(f"\n_({len(rejected)-40} more rejections truncated)_")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hc357-csv",
        default="/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/"
                "hc357_overlay/hc357_adjusted_ranking.csv",
    )
    p.add_argument(
        "--template",
        default="/home/jupiter/Lvl3Quant/live_trading/v3_3_deploy_package/"
                "configs/v33_config_template.json",
    )
    p.add_argument(
        "--out-dir",
        default="/home/jupiter/Lvl3Quant/live_trading/v3_3_deploy_package/"
                "configs/candidate_configs",
    )
    p.add_argument("--max-candidates", type=int, default=10)
    args = p.parse_args()

    csv_path = Path(args.hc357_csv)
    if not csv_path.exists():
        print(f"[fatal] HC #357 ranking csv not found at {csv_path}", file=sys.stderr)
        return 2

    template_path = Path(args.template)
    template = json.loads(template_path.read_text())

    rows = _load_hc357_rankings(csv_path)
    print(f"[load] {len(rows)} rows from hc357 ranking csv")

    # Sort by hc357_sharpe DESC, then hc357_net DESC
    rows.sort(key=lambda r: (r["hc357_sharpe"], r["hc357_net"]), reverse=True)

    passed = [r for r in rows if _passes_gate(r)]
    rejected = [r for r in rows if not _passes_gate(r)]
    print(f"[gate] {len(passed)} pass / {len(rejected)} fail")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(passed[: args.max_candidates], 1):
        cfg = _build_candidate_config(template, row, i)
        slug = f"{i:02d}_{_slug(row['head'])}_{_slug(row['band'])}_{BAND_TO_TIME_EXIT.get(row['band'], 10)}s.json"
        out_path = out_dir / slug
        out_path.write_text(json.dumps(cfg, indent=2))
        print(f"[write] {out_path}")

    report = _render_report(passed[: args.max_candidates], rejected, out_dir)
    report_path = Path(args.out_dir).parent.parent / "config_search_report.md"
    report_path.write_text(report)
    print(f"[report] {report_path}")

    if not passed:
        print(
            "[result] NO deploy-eligible candidates. v2 stays live. "
            "Re-run after model improvements (more training data / σ recalibration / "
            "head-confluence re-evaluation).",
            file=sys.stderr,
        )
        return 1

    print(f"[result] {min(len(passed), args.max_candidates)} candidates written. "
          f"Review {report_path}, then deploy via .\\runbooks\\deploy_v33_to_razer.ps1 "
          f"-ConfigName <slug>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
