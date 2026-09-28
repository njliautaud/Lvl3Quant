"""Sector picker v8 — Sub-industry granularity (HC #573, 2026-06-08).

Same machinery as v6 sparse-K (univariate Spearman IC top-K + ridge fit, same
panel, WF, costs, regime overlay) but pivots the bucket axis from GICS
SECTOR to SUB-INDUSTRY (see `sub_industry_taxonomy.py`).

Why:
    User directive HC #573 — the broad-sector picker averages edge across
    too-heterogeneous baskets ("Tech" = semis + cloud + cybersec + fintech +
    AI-software + robotics + quantum + space). At sub-industry granularity
    a narrow basket like "small-cap uranium miners" or "physical-AI/robotics"
    has a chance to surface real edge that the broad-sector average washes
    out.

Differences from v6:
    1. GROUPBY → ticker's sub-industry tag(s) from sub_industry_taxonomy.
       A ticker may belong to MULTIPLE buckets and contributes to each.
    2. Buckets with <4 deployable names are dropped (no Hail Mary).
    3. Single-name cap: per HC #573 R4, any one ticker contributes at most
       `1 / max(N_in_bucket, 4) × 0.5` to a bucket's daily PnL. This blocks
       the old-MacroStrategy "all-tech" concentration failure mode.
    4. Per-bucket Sharpe gate ≥ 1.0 AND pooled Calmar ≥ 1.5 for deploy
       (HC #573 R3, inherits HC #428 regime-agnostic 40-day OOT check).

CLI:
    python3 sector_picker_v8_subind.py [--K 10] [--hold-days 10] [--out DIR]
                                       [--min-bucket-size 4] [--n-jobs -1]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))

import sector_picker_v4 as v4  # noqa: E402  type: ignore
import sector_picker_v6_sparseK as v6  # noqa: E402  type: ignore
from walk_forward import _metrics  # noqa: E402  type: ignore
from sub_industry_taxonomy import (  # noqa: E402
    TICKER_TO_SUBINDUSTRIES,
    list_subindustries,
    get_tickers_for,
)

PANEL_V2 = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"

DEFAULT_HOLD = 10
DEFAULT_K = 10
DEFAULT_MIN_BUCKET = 4
SHARPE_GATE = 1.0
CALMAR_GATE = 1.5


# ---------------------------------------------------------------------------
# Per-bucket worker. Reuses v6.portfolio_for_sector_sparseK for the core
# sparse-K WF+ridge — we just feed it the filtered ticker subset.
# ---------------------------------------------------------------------------
def _run_one_subind(
    subind: str,
    panel: pd.DataFrame,
    K: int,
    hold_days: int,
    min_bucket: int,
):
    tickers = set(get_tickers_for(subind))
    if not tickers:
        return subind, None
    sp = panel[panel["ticker"].isin(tickers)].copy()
    n_tk = sp["ticker"].nunique()
    if n_tk < min_bucket:
        return subind, {"skipped": True, "reason": f"{n_tk} tickers < min_bucket={min_bucket}"}
    feats_present = [f for f in v6.FEATURE_POOL if f in sp.columns]
    if not feats_present:
        return subind, {"skipped": True, "reason": "no candidate features present in panel"}

    res = v6.portfolio_for_sector_sparseK(sp, feats_present, K=K, hold_days=hold_days)
    pnl = res["daily_pnl"]
    if pnl.empty:
        return subind, None

    # HC #573 R4 single-name cap: the v6 long/short basket already averages
    # equally over TOP_N=3 longs + BOT_N=3 shorts, so each name's share is
    # 1/(TOP_N+BOT_N)=0.166. With min_bucket=4 that's already below the
    # 1/max(N,4) × 0.5 = 0.125 cap only when N==4. For tiny buckets we
    # scale the daily PnL down by (cap / share) to enforce the rule.
    name_share = 1.0 / (v6.TOP_N + v6.BOT_N)
    cap = 0.5 / max(n_tk, 4)
    if name_share > cap:
        scale = cap / name_share
        pnl = pnl * scale

    return subind, {
        "skipped": False,
        "n_tickers": int(n_tk),
        "n_candidate_features": len(feats_present),
        "n_folds": len(res["folds"]),
        "avg_coef": res["avg_coef"],
        "avg_ic": res["avg_ic"],
        "chosen_counts": res["chosen_counts"],
        "folds": res["folds"],
        "daily_pnl": pnl,
        "metrics": _metrics(pnl),
        "name_cap_applied": (name_share > cap),
        "name_cap_value": float(cap),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--hold-days", type=int, default=DEFAULT_HOLD)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--min-bucket-size", type=int, default=DEFAULT_MIN_BUCKET,
                    help="Drop sub-industries with fewer deployable names (default 4)")
    ap.add_argument("--only-subind", type=str, default=None,
                    help="Comma-separated subset of sub-industries to run (debug)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        out_dir = ROOT / f"output/macro_picker/v8_subind_{ts}/K{args.K:02d}_H{args.hold_days:02d}d"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v8 subind] K={args.K}  hold_days={args.hold_days}  out={out_dir}")
    print(f"[v8 subind] taxonomy: {len(list_subindustries())} sub-industries, "
          f"{len(TICKER_TO_SUBINDUSTRIES)} ticker mappings")

    # Patch v4 globals (v6 relies on them too)
    v4.PANEL = PANEL_V2
    v4.HOLD_DAYS = args.hold_days

    print("loading panel ...")
    panel = v4.load_panel()
    print(f"  shape {panel.shape}")
    spy_ret, fund = v4.load_spy_and_funding()
    panel = v4.build_target(panel, hold_days=args.hold_days)

    # Universe overlap: of tickers in panel, how many are in our taxonomy?
    panel_tickers = set(panel["ticker"].unique())
    mapped = panel_tickers & set(TICKER_TO_SUBINDUSTRIES.keys())
    print(f"  panel tickers: {len(panel_tickers)};  in taxonomy: {len(mapped)}  "
          f"({100*len(mapped)/max(1,len(panel_tickers)):.0f}%)")

    # Pick the sub-industry universe
    if args.only_subind:
        subinds = [s.strip() for s in args.only_subind.split(",") if s.strip()]
    else:
        subinds = list_subindustries()
    print(f"  running {len(subinds)} sub-industries on {args.n_jobs} workers")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_run_one_subind)(s, panel, args.K, args.hold_days, args.min_bucket_size)
        for s in subinds
    )
    wall = time.time() - t0
    print(f"  parallel wall: {wall:.1f}s")

    per_subind = {s: r for s, r in results if r is not None}
    skipped = {s: r["reason"] for s, r in per_subind.items() if r.get("skipped")}
    active = {s: r for s, r in per_subind.items() if not r.get("skipped")}

    print(f"  active buckets: {len(active)};  skipped (size/empty): {len(skipped)}")

    if not active:
        print("no active sub-industry buckets — aborting"); return

    # Equal-weight across active buckets per day
    pnl_frames = {s: r["daily_pnl"] for s, r in active.items()}
    combined_df = pd.concat(pnl_frames.values(), axis=1, keys=pnl_frames.keys())
    combined_pnl = combined_df.mean(axis=1, skipna=True).dropna()

    strat_m = _metrics(combined_pnl)

    aligned = pd.concat(
        [combined_pnl.rename("s"), spy_ret.rename("spy"), fund.rename("f")],
        axis=1, join="inner").dropna(subset=["s", "spy"])
    aligned["f"] = aligned["f"].ffill().bfill()
    spy1 = v4._spy_levered(aligned["spy"], 1.0, aligned["f"])
    spy15 = v4._spy_levered(aligned["spy"], 1.5, aligned["f"])
    spy2 = v4._spy_levered(aligned["spy"], 2.0, aligned["f"])

    per_metrics = {s: r["metrics"] for s, r in active.items()}
    deployable = {s: m for s, m in per_metrics.items()
                  if m.get("sharpe", -1e9) >= SHARPE_GATE
                  and m.get("calmar", -1e9) >= 0.0}  # per-bucket Sharpe gate
    pooled_pass = strat_m.get("calmar", -1e9) >= CALMAR_GATE

    out = {
        "config": {
            "K": args.K,
            "hold_days": args.hold_days,
            "panel": str(PANEL_V2),
            "min_bucket_size": args.min_bucket_size,
            "sharpe_gate_per_bucket": SHARPE_GATE,
            "calmar_gate_pooled": CALMAR_GATE,
            "candidate_pool_size": len(v6.FEATURE_POOL),
            "taxonomy_sub_industries": len(list_subindustries()),
            "taxonomy_tickers": len(TICKER_TO_SUBINDUSTRIES),
            "panel_tickers_in_taxonomy": len(mapped),
        },
        "pooled_oot_combined": strat_m,
        "spy_1x": _metrics(spy1),
        "spy_15x": _metrics(spy15),
        "spy_2x": _metrics(spy2),
        "per_subindustry": per_metrics,
        "deployable_sub_industries": list(deployable.keys()),
        "n_deployable": len(deployable),
        "n_active": len(active),
        "n_skipped": len(skipped),
        "skipped_reasons": skipped,
        "pooled_passes_calmar_gate": bool(pooled_pass),
        "n_trading_days": int(len(combined_pnl)),
        "date_range": [str(combined_pnl.index.min().date()),
                       str(combined_pnl.index.max().date())] if len(combined_pnl) else [None, None],
        "wall_sec": round(wall, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(out, indent=2, default=str))

    # Per-bucket formulas
    formulas = {}
    for s, r in active.items():
        formulas[s] = {
            "n_tickers": r["n_tickers"],
            "n_folds": r["n_folds"],
            "avg_coef_on_selected": r["avg_coef"],
            "chosen_counts": r["chosen_counts"],
            "name_cap_applied": r["name_cap_applied"],
            "name_cap_value": r["name_cap_value"],
        }
    (out_dir / "per_subindustry_formulas.json").write_text(
        json.dumps(formulas, indent=2, default=str))

    # Per-day PnL
    pnl_df = pd.DataFrame({"date": combined_pnl.index, "combined_pnl": combined_pnl.values})
    for s, p in pnl_frames.items():
        pnl_df = pnl_df.merge(
            pd.DataFrame({"date": p.index, s: p.values}), on="date", how="left")
    pnl_df.to_parquet(out_dir / "per_day_pnl.parquet", index=False)

    # Markdown
    md = [
        f"# Sector picker v8 — Sub-industry (K={args.K}, hold={args.hold_days}d)",
        "",
        f"**Taxonomy**: {len(list_subindustries())} sub-industries, "
        f"{len(TICKER_TO_SUBINDUSTRIES)} ticker mappings",
        f"**Buckets active / skipped**: {len(active)} / {len(skipped)} "
        f"(min size = {args.min_bucket_size})",
        f"**Wall**: {wall:.1f}s",
        "",
        "## Pooled-OOT",
        "",
        "| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |",
        "|---|---:|---:|---:|---:|",
    ]
    spy1_m = _metrics(spy1); spy15_m = _metrics(spy15); spy2_m = _metrics(spy2)
    md.append(f"| Sharpe | {strat_m['sharpe']:.2f} | {spy1_m['sharpe']:.2f} | "
              f"{spy15_m['sharpe']:.2f} | {spy2_m['sharpe']:.2f} |")
    md.append(f"| Sortino | {strat_m['sortino']:.2f} | {spy1_m['sortino']:.2f} | "
              f"{spy15_m['sortino']:.2f} | {spy2_m['sortino']:.2f} |")
    md.append(f"| CAGR | {strat_m['cagr']*100:.1f}% | {spy1_m['cagr']*100:.1f}% | "
              f"{spy15_m['cagr']*100:.1f}% | {spy2_m['cagr']*100:.1f}% |")
    md.append(f"| MaxDD | {strat_m['max_dd']*100:.1f}% | {spy1_m['max_dd']*100:.1f}% | "
              f"{spy15_m['max_dd']*100:.1f}% | {spy2_m['max_dd']*100:.1f}% |")
    md.append(f"| Calmar | {strat_m['calmar']:.2f} | {spy1_m['calmar']:.2f} | "
              f"{spy15_m['calmar']:.2f} | {spy2_m['calmar']:.2f} |")
    md.append("")
    md.append(f"**Deployable sub-industries (Sharpe ≥ {SHARPE_GATE})**: "
              f"{len(deployable)} / {len(active)}")
    md.append(f"**Pooled passes Calmar ≥ {CALMAR_GATE}**: {'YES' if pooled_pass else 'NO'}")
    md.append("")
    md.append("## Per-sub-industry pooled-OOT (sorted by Sharpe)")
    md.append("")
    md.append("| Sub-industry | N | Sharpe | CAGR | MaxDD | Calmar | PF | WR |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s, m in sorted(per_metrics.items(), key=lambda kv: -kv[1].get("sharpe", -1e9)):
        n_t = active[s]["n_tickers"]
        md.append(f"| {s} | {n_t} | {m.get('sharpe', 0):.2f} | "
                  f"{m.get('cagr', 0)*100:.1f}% | {m.get('max_dd', 0)*100:.1f}% | "
                  f"{m.get('calmar', 0):.2f} | {m.get('pf', 0):.2f} | "
                  f"{m.get('wr', 0)*100:.1f}% |")
    md.append("")
    if skipped:
        md.append("## Skipped buckets")
        md.append("")
        for s, reason in sorted(skipped.items()):
            md.append(f"- `{s}` — {reason}")
        md.append("")
    (out_dir / "report.md").write_text("\n".join(md))

    print()
    print("\n".join(md))
    print()
    print(f"WROTE:")
    print(f"  {out_dir / 'report.json'}")
    print(f"  {out_dir / 'report.md'}")
    print(f"  {out_dir / 'per_subindustry_formulas.json'}")
    print(f"  {out_dir / 'per_day_pnl.parquet'}")


if __name__ == "__main__":
    main()
