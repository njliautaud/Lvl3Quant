"""
v33_composite_gates_v3.py — Composite regime gate analysis on K=2 LONG signals.

Reads:
  output/.../regime_analysis/k2_long_signals_with_regime_v2.csv  (per-event records w/ regime features)
  output/.../regime_analysis/per_day_features_v2.csv             (daily regime stats)

Tests composite gates of the form (vol, drift, trend_strength, intraday_drift, ToD bucket, spread)
applied to the LONG-fillable K=2 stack, replays via existing per-event net_ticks
(already from HC #357 canonical replay, passive_at_touch, 30s hold, commission-netted).

Writes:
  output/.../regime_analysis/composite_gates_v3.json

No external libs touched. Self-contained orchestrator.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/regime_analysis")
SIG_CSV = OUT_DIR / "k2_long_signals_with_regime_v2.csv"
DAY_CSV = OUT_DIR / "per_day_features_v2.csv"
OUT_JSON = OUT_DIR / "composite_gates_v3.json"


def gate_stats(df_filt: pd.DataFrame, label: str) -> dict:
    """Compute deploy-relevant stats for a filtered slice of LONG-fillable K=2 events."""
    filled = df_filt[df_filt.filled == True]
    n_signals = int(len(df_filt))
    n_fills = int(len(filled))
    if n_fills == 0:
        return dict(name=label, n_signals=n_signals, n_fills=0, t_per_fill=0.0, wr=0.0,
                    pf=0.0, total_ticks=0.0, max_day_conc=0.0, days_with_fills=0,
                    per_day={})
    net = filled.net_ticks.values
    wr = float((net > 0).mean())
    total = float(net.sum())
    t_per_fill = float(net.mean())
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    pf = float(pos / neg) if neg > 0 else float("inf")
    per_day = filled.groupby("date").net_ticks.agg(["count", "sum", "mean"]).to_dict("index")
    per_day = {str(k): {"n": int(v["count"]), "ticks": float(v["sum"]), "t_per_fill": float(v["mean"])}
               for k, v in per_day.items()}
    day_counts = filled.groupby("date").size()
    max_day_conc = float(day_counts.max() / n_fills) if n_fills else 0.0
    return dict(name=label, n_signals=n_signals, n_fills=n_fills,
                t_per_fill=t_per_fill, wr=wr, pf=pf, total_ticks=total,
                max_day_conc=max_day_conc, days_with_fills=int(len(per_day)),
                per_day=per_day)


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading {SIG_CSV}")
    if not SIG_CSV.exists():
        print(f"ERR: missing {SIG_CSV}")
        sys.exit(2)
    df = pd.read_csv(SIG_CSV)
    print(f"  n_events={len(df)}, n_filled={int(df.filled.sum())}, cols={list(df.columns)}")

    day_df = pd.read_csv(DAY_CSV)
    print(f"  day_features loaded ({len(day_df)} days)")
    day_feat = day_df.set_index("date").to_dict("index")
    # Map daily vol / trend to each event
    df["day_vol_1s"] = df.date.map(lambda d: day_feat.get(int(d), {}).get("vol_1s_ticks_std", np.nan))
    df["day_trend"] = df.date.map(lambda d: day_feat.get(int(d), {}).get("trend_strength", np.nan))
    df["day_drift"] = df.date.map(lambda d: day_feat.get(int(d), {}).get("drift_ticks", np.nan))

    results = {}

    # Baseline (no gate)
    results["BASELINE_no_gate"] = gate_stats(df, "BASELINE_no_gate")

    # 0225-like: low daily vol + strong uptrend
    g = df[(df.day_vol_1s < 1.0) & (df.day_trend > 0.5)]
    results["0225_like_vol<1.0_trend>0.5"] = gate_stats(g, "0225_like_vol<1.0_trend>0.5")

    # Daily vol filters
    for vol_max in [1.0, 1.1, 1.25, 1.4]:
        g = df[df.day_vol_1s < vol_max]
        results[f"day_vol_lt_{vol_max}"] = gate_stats(g, f"day_vol_lt_{vol_max}")

    # Daily trend filters (uptrend days only)
    for tr_min in [0.0, 0.3, 0.5, 0.7]:
        g = df[df.day_trend > tr_min]
        results[f"day_trend_gt_{tr_min}"] = gate_stats(g, f"day_trend_gt_{tr_min}")

    # Daily drift uptrend
    for drift_min in [0, 50, 100, 200]:
        g = df[df.day_drift > drift_min]
        results[f"day_drift_gt_{drift_min}"] = gate_stats(g, f"day_drift_gt_{drift_min}")

    # Intraday momentum (intraday_drift_ticks > 0)
    for idd in [0, 10, 25, 50, 100]:
        g = df[df.intraday_drift_ticks > idd]
        results[f"intra_drift_gt_{idd}"] = gate_stats(g, f"intra_drift_gt_{idd}")

    # Intraday vol regime (30s window) — short-term low vol
    for v in [1.0, 1.5, 2.0, 3.0]:
        g = df[df.vol_30s_ticks < v]
        results[f"intra_vol30s_lt_{v}"] = gate_stats(g, f"intra_vol30s_lt_{v}")

    # ToD bucket exclusion (skip open)
    for skip_bucket in ["open_0930_1030"]:
        g = df[df.bucket != skip_bucket]
        results[f"skip_{skip_bucket}"] = gate_stats(g, f"skip_{skip_bucket}")

    # ToD bucket isolation
    for keep_bucket in sorted(df.bucket.unique()):
        g = df[df.bucket == keep_bucket]
        results[f"only_{keep_bucket}"] = gate_stats(g, f"only_{keep_bucket}")

    # Composite: low-vol + uptrend + momentum
    composites = [
        ("comp_vol<1.0_trend>0.5_intra>0",
         (df.day_vol_1s < 1.0) & (df.day_trend > 0.5) & (df.intraday_drift_ticks > 0)),
        ("comp_vol<1.4_drift>0_intra>0",
         (df.day_vol_1s < 1.4) & (df.day_drift > 0) & (df.intraday_drift_ticks > 0)),
        ("comp_vol<1.25_trend>0.2",
         (df.day_vol_1s < 1.25) & (df.day_trend > 0.2)),
        ("comp_drift>50_intra>0",
         (df.day_drift > 50) & (df.intraday_drift_ticks > 0)),
        ("comp_skip_open_intra>0",
         (df.bucket != "open_0930_1030") & (df.intraday_drift_ticks > 0)),
    ]
    for name, mask in composites:
        results[name] = gate_stats(df[mask], name)

    # Compose into rankable list
    rows = []
    for k, v in results.items():
        rows.append({
            "gate": v["name"],
            "n_signals": v["n_signals"],
            "n_fills": v["n_fills"],
            "t_per_fill": round(v["t_per_fill"], 3),
            "wr": round(v["wr"], 3),
            "pf": round(v["pf"], 3) if v["pf"] != float("inf") else "inf",
            "total_t": round(v["total_ticks"], 1),
            "max_day_conc": round(v["max_day_conc"], 3),
            "days": v["days_with_fills"],
            "passes_hc344": v["max_day_conc"] <= 0.20 and v["n_fills"] >= 30,
            "passes_edge": v["t_per_fill"] > 0.0 and v["n_fills"] >= 30,
        })
    rank_df = pd.DataFrame(rows).sort_values(by=["passes_hc344", "passes_edge", "t_per_fill"],
                                              ascending=[False, False, False])
    print("\n=== TOP 15 BY t/fill (with gates) ===")
    print(rank_df.head(15).to_string(index=False))

    out = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        rule="K=2 LONG (p60 top20% AND p5m bot20%) entered LONG on tp4sl3_long_filled mask, passive_at_touch 30s hold, comm-netted.",
        n_events_total=int(len(df)),
        n_filled_total=int(df.filled.sum()),
        results=results,
        ranking=rank_df.to_dict(orient="records"),
        elapsed_sec=round(time.time() - t0, 2),
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {time.time()-t0:.1f}s — wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
