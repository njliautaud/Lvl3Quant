#!/usr/bin/env python3
"""HC #443 filter stratification on canonical fills.

Slices the canonical _fifo_fills.csv files by:
  - time-of-day (30-min RTH buckets)
  - day-of-week
  - pred_strength quintiles (within-day percentile)
  - queue_ahead bucket (0, 1-5, 6-20, 21-100, 100+)
  - direction
  - hold_s bucket

For each slice computes: n, mean_tk, PF, WR, sharpe, sortino.

Goal: find a NON-EMPTY slice on the EXISTING (already-losing) signal universe
that, by itself, would have produced positive risk-adjusted P&L. If such a slice
exists, it becomes the new candidate filter for HC #442 R2 layered stack
(configs C/D from the roadmap).

Apples-to-apples: same fills, same canonical engine, just sub-selected. The
worst-case outcome is "no slice clears" which is itself a Friday-deliverable
finding (rules out the slicing axis).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
RESULTS_DIR = LVL3 / "output" / "hc432_v342_47day_validation"
OUT_DIR = LVL3 / "output" / "hc443_filter_strat"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def slice_metrics(s: pd.DataFrame) -> dict:
    if len(s) == 0:
        return {"n": 0, "mean_tk": np.nan, "sum_tk": 0.0,
                "PF": np.nan, "WR": np.nan,
                "sharpe": np.nan, "sortino": np.nan}
    nt = s["net_ticks"].to_numpy()
    wins = nt[nt > 0].sum()
    losses = -nt[nt < 0].sum()
    PF = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    WR = float((nt > 0).mean()) * 100.0
    mu = float(nt.mean())
    sd = float(nt.std(ddof=1)) if len(nt) > 1 else 0.0
    sharpe = (mu / sd) * np.sqrt(len(nt)) if sd > 0 else 0.0
    neg = nt[nt < 0]
    sortino_sd = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mu / sortino_sd) * np.sqrt(len(nt)) if sortino_sd > 0 else 0.0
    return {"n": int(len(nt)), "mean_tk": mu, "sum_tk": float(nt.sum()),
            "PF": float(PF) if np.isfinite(PF) else 999.0,
            "WR": WR, "sharpe": sharpe, "sortino": sortino}


def stratify(df: pd.DataFrame, config_name: str) -> dict:
    # Compute UTC entry-time → ET hour
    df = df.copy()
    df["entry_dt"] = pd.to_datetime(df["ts_entry_ns"], unit="ns", utc=True).dt.tz_convert("US/Eastern")
    df["et_hour"] = df["entry_dt"].dt.hour
    df["et_min30"] = df["entry_dt"].dt.minute // 30
    df["et_bucket"] = df["et_hour"].astype(str) + "_" + df["et_min30"].astype(str)
    df["dow"] = df["entry_dt"].dt.day_name()

    # Within-day pred_strength percentile (signal-relative-rank)
    df["pred_pct_in_day"] = df.groupby("date")["pred_strength"].rank(pct=True)

    out = {"config": config_name, "n_total": len(df), "slices": {}}

    # 1) Overall
    out["slices"]["__overall__"] = slice_metrics(df)

    # 2) Time-of-day
    tod = {}
    for bucket, sub in df.groupby("et_bucket"):
        if len(sub) >= 30:
            tod[bucket] = slice_metrics(sub)
    out["slices"]["time_of_day"] = tod

    # 3) Day of week
    dow = {}
    for day, sub in df.groupby("dow"):
        if len(sub) >= 30:
            dow[day] = slice_metrics(sub)
    out["slices"]["day_of_week"] = dow

    # 4) Pred-strength percentile within day
    quintiles = {}
    df["pred_qtl"] = pd.cut(df["pred_pct_in_day"],
                            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.001],
                            labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    for q, sub in df.groupby("pred_qtl", observed=False):
        if len(sub) >= 30:
            quintiles[str(q)] = slice_metrics(sub)
    out["slices"]["pred_strength_quintile_within_day"] = quintiles

    # 5) Queue-ahead bucket
    qbuckets = {}
    df["q_bucket"] = pd.cut(df["queue_ahead"].clip(0, 99999),
                            bins=[-0.5, 0.5, 5.5, 20.5, 100.5, 1e9],
                            labels=["q=0", "q=1-5", "q=6-20", "q=21-100", "q=100+"])
    for q, sub in df.groupby("q_bucket", observed=False):
        if len(sub) >= 30:
            qbuckets[str(q)] = slice_metrics(sub)
    out["slices"]["queue_ahead_bucket"] = qbuckets

    # 6) Hold-time bucket
    hbuckets = {}
    df["h_bucket"] = pd.cut(df["hold_s"],
                            bins=[-0.01, 0.5, 1.0, 1.5, 5.0, 30.0],
                            labels=["<0.5s", "0.5-1s", "1-1.5s", "1.5-5s", "5s+"])
    for h, sub in df.groupby("h_bucket", observed=False):
        if len(sub) >= 30:
            hbuckets[str(h)] = slice_metrics(sub)
    out["slices"]["hold_time_bucket"] = hbuckets

    # 7) Fill-type
    ftype = {}
    for ft, sub in df.groupby("fill_type"):
        if len(sub) >= 30:
            ftype[str(ft)] = slice_metrics(sub)
    out["slices"]["fill_type"] = ftype

    # 8) 2-way: time × pred-quintile (find best cell)
    cross = {}
    for (et, q), sub in df.groupby(["et_bucket", "pred_qtl"], observed=False):
        if len(sub) >= 50:
            cross[f"{et}__{q}"] = slice_metrics(sub)
    out["slices"]["et_x_pred_quintile"] = cross

    # 9) 2-way: time × queue
    cross2 = {}
    for (et, q), sub in df.groupby(["et_bucket", "q_bucket"], observed=False):
        if len(sub) >= 50:
            cross2[f"{et}__{q}"] = slice_metrics(sub)
    out["slices"]["et_x_queue"] = cross2

    return out


def top_slices_report(strats: list[dict], top_n: int = 15) -> str:
    rows = []
    for s in strats:
        cfg = s["config"]
        for slice_kind, content in s["slices"].items():
            if slice_kind == "__overall__":
                rows.append({"config": cfg, "slice_kind": "OVERALL",
                             "key": "all", **content})
            else:
                for k, m in content.items():
                    rows.append({"config": cfg, "slice_kind": slice_kind,
                                 "key": k, **m})
    df = pd.DataFrame(rows)
    df = df[df["n"] >= 30].copy()
    df = df[df["sharpe"].notna()]
    pos = df[(df["mean_tk"] > 0) & (df["PF"] >= 1.0)].sort_values(
        "sharpe", ascending=False).head(top_n)
    return df, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=False,
                    help="Specific fills CSVs (can repeat). Default = all hc44*_fifo_fills.csv with >=100 fills.")
    args = ap.parse_args()

    if args.csv:
        csv_paths = [Path(p) for p in args.csv]
    else:
        csv_paths = []
        for p in sorted(RESULTS_DIR.glob("hc44*_fifo_fills.csv")):
            try:
                # quick row count
                with open(p) as fh:
                    n = sum(1 for _ in fh) - 1
                if n >= 100:
                    csv_paths.append(p)
            except Exception:
                continue

    print(f"[strat] processing {len(csv_paths)} fills CSVs", file=sys.stderr)

    all_strats = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            if len(df) < 100:
                continue
            name = p.stem.replace("_fifo_fills", "")
            print(f"[strat] {name}: n={len(df)}", file=sys.stderr)
            s = stratify(df, name)
            all_strats.append(s)
            with open(OUT_DIR / f"{name}_strat.json", "w") as fh:
                json.dump(s, fh, indent=2, default=str)
        except Exception as e:
            print(f"[strat] failed on {p.name}: {e}", file=sys.stderr)

    # Build master table
    df_all, df_pos = top_slices_report(all_strats, top_n=25)
    df_all.to_csv(OUT_DIR / "all_slices.csv", index=False)
    df_pos.to_csv(OUT_DIR / "positive_slices_top25.csv", index=False)

    # Markdown summary
    lines = [f"# HC #443 Filter-Stratification Report", ""]
    lines.append(f"Slices computed: {len(df_all)} across {len(all_strats)} canonical fills CSVs.")
    lines.append(f"Positive-mean + PF≥1 slices (≥30 fills): **{len(df_pos)}**")
    lines.append("")
    if len(df_pos) > 0:
        lines.append("## Top 25 positive-mean slices (by sharpe)")
        lines.append("")
        lines.append("| config | slice_kind | key | n | mean_tk | PF | WR | sharpe |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for _, r in df_pos.iterrows():
            lines.append(f"| {r['config']} | {r['slice_kind']} | {r['key']} | "
                         f"{int(r['n'])} | {r['mean_tk']:+.3f} | {r['PF']:.2f} | "
                         f"{r['WR']:.1f}% | {r['sharpe']:+.2f} |")
    else:
        lines.append("## NO positive slices found.")
        lines.append("All slices with ≥30 fills are negative or PF<1.")
        lines.append("")
        lines.append("Implication: simple post-hoc filters on (time, conf-percentile,")
        lines.append("queue, hold, fill-type) cannot rescue the canonical loser.")
        lines.append("Friday fallback (live data collection) is the honest path.")

    with open(OUT_DIR / "report.md", "w") as fh:
        fh.write("\n".join(lines))

    print(f"[strat] DONE — saved {OUT_DIR}", file=sys.stderr)
    print(f"[strat] positive slices: {len(df_pos)}", file=sys.stderr)


if __name__ == "__main__":
    main()
