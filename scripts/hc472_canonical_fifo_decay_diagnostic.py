#!/usr/bin/env python3
"""HC #472 R1/R2 — Per-day decay trajectory + first/second half stratification
for the surviving canonical FIFO fills.

Reads output/stream_backtest_v2/surviving_canonical_fifo_fills.parquet (20,939
fills, 15 days with activity, 11 configs).

For each config + the union "ALL" set:
  - per-day trajectory: n_fills, net_ticks_sum, net_ticks_mean, wr, sharpe_daily
  - first_half vs second_half (split by median date)
  - long vs short net_ticks decomposition

Outputs:
  output/hc472_canonical_decay/per_config_per_day.parquet
  output/hc472_canonical_decay/per_config_halves.csv
  output/hc472_canonical_decay/summary.md

Diagnostic only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
SRC = ROOT / "output/stream_backtest_v2/surviving_canonical_fifo_fills.parquet"
OUT = ROOT / "output/hc472_canonical_decay"
OUT.mkdir(parents=True, exist_ok=True)


def metrics_for_subset(df):
    if len(df) == 0:
        return dict(n=0, mean_ticks=None, wr=None, sharpe_per_trade=None,
                    sharpe_daily=None, long_share=None, n_days=0,
                    sum_ticks=None, sortino_daily=None, pf=None)
    nt = df["net_ticks"].values
    is_long = (df["direction"].astype(str).str.lower() == "long").values \
        if "direction" in df.columns else np.zeros(len(df), dtype=bool)
    daily = df.groupby("date")["net_ticks"].sum()
    sharpe_pt = float(nt.mean() / nt.std(ddof=1)) if nt.std(ddof=1) > 0 else None
    sharpe_d = float(daily.mean() / daily.std(ddof=1)) if daily.std(ddof=1) > 0 \
        else None
    neg = daily[daily < 0]
    sortino_d = float(daily.mean() / neg.std(ddof=1)) if (len(neg) > 1 and
                                                          neg.std(ddof=1) > 0) \
        else None
    pos_sum = float(nt[nt > 0].sum())
    neg_sum = float(-nt[nt < 0].sum())
    pf = (pos_sum / neg_sum) if neg_sum > 0 else None
    return dict(
        n=int(len(df)),
        n_days=int(df["date"].nunique()),
        sum_ticks=float(nt.sum()),
        mean_ticks=float(nt.mean()),
        wr=float((nt > 0).mean()),
        sharpe_per_trade=sharpe_pt,
        sharpe_daily=sharpe_d,
        sortino_daily=sortino_d,
        pf=pf,
        long_share=float(is_long.mean()),
    )


def main():
    df = pd.read_parquet(SRC)
    print(f"[decay] {len(df)} fills, {df['date'].nunique()} days, "
          f"{df['config'].nunique()} configs")

    # Per-day per-config
    per_day = (df.groupby(["config", "date"])
                 .apply(lambda d: pd.Series({
                     "n": len(d),
                     "sum_ticks": float(d["net_ticks"].sum()),
                     "mean_ticks": float(d["net_ticks"].mean()),
                     "wr": float((d["net_ticks"] > 0).mean()),
                     "long_share": float((d["direction"].astype(str)
                                            .str.lower() == "long").mean()
                                          if "direction" in d.columns else 0.0),
                 })).reset_index())
    per_day.to_parquet(OUT / "per_config_per_day.parquet", index=False)

    # First/second half by date
    dates_sorted = np.sort(df["date"].unique())
    median_date = dates_sorted[len(dates_sorted) // 2]
    first_half = df[df["date"] < median_date]
    second_half = df[df["date"] >= median_date]

    rows = []
    for cfg in sorted(df["config"].unique()):
        sub = df[df["config"] == cfg]
        sub_fh = first_half[first_half["config"] == cfg]
        sub_sh = second_half[second_half["config"] == cfg]
        sub_long = sub[sub["direction"].astype(str).str.lower() == "long"]
        sub_short = sub[sub["direction"].astype(str).str.lower() == "short"]
        rows.append(dict(
            config=cfg,
            **{f"all_{k}": v for k, v in metrics_for_subset(sub).items()},
            **{f"fh_{k}": v for k, v in metrics_for_subset(sub_fh).items()},
            **{f"sh_{k}": v for k, v in metrics_for_subset(sub_sh).items()},
            **{f"long_{k}": v for k, v in metrics_for_subset(sub_long).items()},
            **{f"short_{k}": v for k, v in metrics_for_subset(sub_short).items()},
        ))
    halves = pd.DataFrame(rows)
    halves.to_csv(OUT / "per_config_halves.csv", index=False)

    # Summary markdown
    md = [
        "# HC #472 — Canonical FIFO Decay Diagnostic",
        f"\nSource: `{SRC.name}` (n={len(df)} fills, days={df['date'].nunique()})",
        f"\nMedian split date: `{median_date}` (first half < this, second ≥)",
        "\n## Headline (all configs, all fills)\n",
    ]
    all_m = metrics_for_subset(df)
    md.append(f"- n_fills={all_m['n']}, n_days={all_m['n_days']}, "
              f"sum_ticks={all_m['sum_ticks']:.1f}, "
              f"mean_ticks={all_m['mean_ticks']:.3f}, WR={all_m['wr']:.3f}")
    md.append(f"- Sharpe per-trade={all_m['sharpe_per_trade']}, "
              f"Sharpe daily={all_m['sharpe_daily']}, "
              f"Sortino daily={all_m['sortino_daily']}, PF={all_m['pf']}")
    md.append(f"- long_share={all_m['long_share']:.4f}  "
              f"({(1 - all_m['long_share']) * 100:.2f}% short)")

    md.append("\n## Per-config first-half vs second-half (HC #472 R2)\n")
    md.append("| config | all_n | all_mean | all_sharpe_d | fh_mean | sh_mean "
              "| fh_wr | sh_wr | long_sum | short_sum |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in halves.iterrows():
        md.append(
            f"| {r['config']} | {r['all_n']} | {r['all_mean_ticks']:.3f} "
            f"| {r['all_sharpe_daily']} "
            f"| {r['fh_mean_ticks']} | {r['sh_mean_ticks']} "
            f"| {r['fh_wr']} | {r['sh_wr']} "
            f"| {r['long_sum_ticks']} | {r['short_sum_ticks']} |"
        )

    md.append("\n## Interpretation per HC #472 R2\n")
    md.append("- If `fh_mean` and `sh_mean` differ by >2×, the full-window mean "
              "is REJECTED as a headline.")
    md.append("- Long-side sum near zero confirms model defect (HC #475 R1).")
    md.append("- Per-config configs whose `sh_sharpe_daily` is "
              "below acceptance fail HC #472 R2 retrain gate.")

    (OUT / "summary.md").write_text("\n".join(md))
    print(f"[decay] wrote {OUT/'summary.md'}")
    print(json.dumps(all_m, indent=2, default=str))


if __name__ == "__main__":
    main()
