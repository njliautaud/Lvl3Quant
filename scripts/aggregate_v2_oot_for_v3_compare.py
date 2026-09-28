#!/usr/bin/env python3
"""
Aggregate v2 daily OOT IC into v3 weekly windows for apples-to-apples comparison.

v2: cnn_mamba_v2_smart_v3_mar — 11 daily folds (fold 0 OOT=20260223 ... fold 10 OOT=20260306)
v3: cnn_mamba_v3_smart_v3_fifo — 10 weekly folds (fold 0 OOT=20260223→20260227 ...)

Output: JSON with weighted-by-n_samples IC at each tier (All/Top50%/Top25%/Top10%/Top5%/Top1%)
        for v3 fold 0 and v3 fold 1 windows (where v2 has coverage).
        Saved to output/v2_weekly_for_v3_compare.json for fold-completion handler to read.

Per HC #286(E), HC #287(E). No code mods to live training systems.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

# v3 weekly OOT windows mapped to v2 daily OOT dates
V3_TO_V2_DATES = {
    "v3_fold_0": ["20260223", "20260224", "20260225", "20260226", "20260227"],
    "v3_fold_1": ["20260301", "20260302", "20260303", "20260304", "20260305"],
    "v3_fold_2_partial_first_day": ["20260306"],  # v2 only covers first day of v3 fold 2
}

# v2 daily folds are indexed: fold 0=20260223 ... fold 10=20260306
V2_DATE_TO_FOLD = {
    "20260223": 0, "20260224": 1, "20260225": 2, "20260226": 3,
    "20260227": 4, "20260301": 5, "20260302": 6, "20260303": 7,
    "20260304": 8, "20260305": 9, "20260306": 10,
}

V2_OUTPUT_DIR_REMOTE = "/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar"

# If running on Jupiter, scp the JSONs first via subprocess. For now assume
# we'll run this on Neptune OR rsync the analysis JSONs to Jupiter.
# Try Jupiter local first, fall back to ssh+cat.

import subprocess

def load_fold_analysis(fold_id: int) -> dict:
    """Read fold_NN_analysis.json from Neptune via ssh+cat."""
    remote_path = f"{V2_OUTPUT_DIR_REMOTE}/fold_{fold_id:02d}_analysis.json"
    # try local first
    local_candidate = Path("/tmp") / f"v2_fold_{fold_id:02d}_analysis.json"
    if not local_candidate.exists():
        try:
            r = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "nick@neptune",
                 f"cat {remote_path}"],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode != 0:
                return None
            local_candidate.write_text(r.stdout)
        except Exception as e:
            print(f"  fold {fold_id}: ssh err {e}", file=sys.stderr)
            return None
    try:
        return json.loads(local_candidate.read_text())
    except Exception as e:
        print(f"  fold {fold_id}: json err {e}", file=sys.stderr)
        return None

def aggregate_window(window_name: str, dates: list[str]) -> dict:
    """For a v3-style weekly window of v2 daily folds, compute weighted IC at each tier/horizon."""
    fold_analyses = []
    for d in dates:
        fold_id = V2_DATE_TO_FOLD.get(d)
        if fold_id is None:
            continue
        fa = load_fold_analysis(fold_id)
        if fa is None:
            print(f"  WARN window={window_name} date={d} fold={fold_id} missing analysis", file=sys.stderr)
            continue
        fold_analyses.append((d, fold_id, fa))
    if not fold_analyses:
        return {"window": window_name, "n_folds": 0, "error": "no v2 fold analyses found"}

    # aggregate by horizon × tier weighted by n_samples
    tiers = ["All", "Top50%", "Top25%", "Top10%", "Top5%", "Top1%"]
    horizons = ["1s", "5s", "10s"]
    out = {"window": window_name, "dates": dates, "n_folds_aggregated": len(fold_analyses), "horizons": {}}

    for h in horizons:
        out["horizons"][h] = {}
        for tier in tiers:
            num_ic = 0.0
            den = 0
            num_da = 0.0
            num_pnl = 0.0
            wins = 0
            losses = 0
            n_samples_total = 0
            for d, fold_id, fa in fold_analyses:
                if h not in fa.get("horizons", {}):
                    continue
                t = fa["horizons"][h].get(tier)
                if t is None:
                    continue
                n = t.get("n_samples", 0)
                if n == 0:
                    continue
                num_ic += t.get("IC", 0.0) * n
                num_da += t.get("DA", 0.0) * n
                num_pnl += t.get("net_pnl", 0.0)  # sum, not weighted
                den += n
                n_samples_total += n
            if den == 0:
                continue
            out["horizons"][h][tier] = {
                "weighted_IC": round(num_ic / den, 4),
                "weighted_DA": round(num_da / den, 4),
                "sum_net_pnl_ticks": round(num_pnl, 2),
                "n_samples_total": n_samples_total,
                "n_folds": len(fold_analyses),
            }
    return out

def main():
    out_dir = Path("/home/jupiter/Lvl3Quant/output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "v2_weekly_for_v3_compare.json"

    result = {
        "purpose": "Apples-to-apples comparison: aggregate v2 daily OOT IC into v3 weekly windows.",
        "v2_run": "cnn_mamba_v2_smart_v3_mar",
        "v3_run_target": "cnn_mamba_v3_smart_v3_fifo (MLflow run f29e0e28...)",
        "windows": {},
    }
    for w, dates in V3_TO_V2_DATES.items():
        result["windows"][w] = aggregate_window(w, dates)

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")

    # print headline
    for w, info in result["windows"].items():
        print(f"\n=== {w} ===")
        if "error" in info:
            print(f"  {info['error']}")
            continue
        print(f"  n_folds={info['n_folds_aggregated']} dates={info['dates']}")
        for h in ["1s", "5s", "10s"]:
            for tier in ["All", "Top25%", "Top10%"]:
                t = info["horizons"].get(h, {}).get(tier)
                if t is None:
                    continue
                print(f"    {h:>3s} {tier:>7s}: IC={t['weighted_IC']:+.4f} DA={t['weighted_DA']:.4f} sum_pnl={t['sum_net_pnl_ticks']:+.1f}t n={t['n_samples_total']}")

if __name__ == "__main__":
    main()
