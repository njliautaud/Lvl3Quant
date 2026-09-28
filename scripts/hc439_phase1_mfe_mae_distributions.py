#!/usr/bin/env python3
"""HC #439 Phase 1 — MFE/MAE distribution analysis per confidence band, side,
horizon, and filter slice.

Reads all per-day parquets from output/hc439_deep_mfe_mae/signals/.
Computes:
  For each (side ∈ {long, short}, horizon ∈ {1s,5s,10s}):
    Confidence is pred_h for long, -pred_h for short.
    For each band ∈ {0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%} (top of conf):
      - n_signals
      - mean / p10 / p25 / p50 / p75 / p90 of side-adjusted MFE (favorable move
        in ticks; long → mfe_Ns_tk, short → mae_Ns_tk because mae is
        long-side adverse = short-side favorable when sign flipped)
      - mean / p10 / p25 / p50 / p75 / p90 of side-adjusted MAE
      - hit-rate: P(side-adj MFE >= 1 tick), P(MFE >= 2 ticks)
      - mean realized label (in ticks, sign-adjusted)
      - expected_value (mean label) NET of two cost regimes:
          passive_net = mean_label - 0.376  (commission only)
          market_net  = mean_label - 1.376  (commission + cross spread)

Then slices by each filter:
  - filter_vol_500ev_tk: terciles (Low/Mid/High)
  - filter_evt_per_sec_30s: terciles
  - filter_buy_aggr_50: terciles
  - tod_bucket: pre-defined buckets
  - dow: 0..4 (Mon..Fri)
  - filter_spread_proxy_tk: terciles

Outputs:
  output/hc439_deep_mfe_mae/phase1/
    overall_table.csv         (band × side × horizon)
    by_vol_table.csv
    by_evt_table.csv
    by_buyaggr_table.csv
    by_tod_table.csv
    by_dow_table.csv
    by_spread_table.csv
    summary.md                (top-line findings, best slices ranked by
                               passive-net EV and consistency)
"""
from __future__ import annotations
import os
import sys
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
SIG_DIR = LVL3 / "output/hc439_deep_mfe_mae/signals"
OUT_DIR = LVL3 / "output/hc439_deep_mfe_mae/phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants (canonical, see CLAUDE.md)
PASSIVE_COST_TK = 0.376  # commission only
MARKET_COST_TK = 1.376   # commission + 1.0 tick spread

CONF_BANDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
CONF_LABELS = ["top0.1%", "top0.5%", "top1%", "top2%", "top5%", "top10%", "top20%"]

HORIZONS = ["1s", "5s", "10s"]


def load_all() -> pd.DataFrame:
    files = sorted(glob.glob(str(SIG_DIR / "*.parquet")))
    print(f"Loading {len(files)} parquet files...")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total signals: {len(df):,}")
    print(f"Date range: {df['date'].min()} - {df['date'].max()}")
    return df


def compute_band_stats(df: pd.DataFrame, side: str, horizon: str,
                       extra_tag: str = "") -> list[dict]:
    """For one side and horizon, compute stats per confidence band.
    side: 'long' or 'short'
    horizon: '1s', '5s', '10s'
    extra_tag: appended to result rows (e.g. filter bucket name)
    """
    n = len(df)
    if n < 50:
        return []
    pred_col = f"pred_{horizon}"
    mfe_col = f"mfe_{horizon}_tk"
    mae_col = f"mae_{horizon}_tk"
    label_col = f"label_{horizon}"

    pred = df[pred_col].to_numpy()
    mfe = df[mfe_col].to_numpy()
    mae = df[mae_col].to_numpy()
    label = df[label_col].to_numpy()

    if side == "long":
        conf = pred
        side_mfe = mfe          # long favorable
        side_mae = mae          # long adverse
        side_label = label
    else:  # short
        conf = -pred
        side_mfe = mae          # short favorable (price down)
        side_mae = mfe          # short adverse (price up)
        side_label = -label

    rows = []
    # Sort by confidence descending once
    order = np.argsort(-conf, kind="quicksort")
    conf_s = conf[order]
    mfe_s = side_mfe[order]
    mae_s = side_mae[order]
    label_s = side_label[order]

    for band, lab in zip(CONF_BANDS, CONF_LABELS):
        k = max(1, int(n * band))
        m = mfe_s[:k]
        a = mae_s[:k]
        L = label_s[:k]
        if len(m) < 20:
            continue
        rows.append({
            "tag": extra_tag,
            "side": side,
            "horizon": horizon,
            "band": lab,
            "n_signals": int(k),
            "mfe_mean": float(m.mean()),
            "mfe_p10": float(np.percentile(m, 10)),
            "mfe_p50": float(np.percentile(m, 50)),
            "mfe_p90": float(np.percentile(m, 90)),
            "mae_mean": float(a.mean()),
            "mae_p50": float(np.percentile(a, 50)),
            "mae_p90": float(np.percentile(a, 90)),
            "label_mean": float(L.mean()),
            "label_median": float(np.percentile(L, 50)),
            "hit_1tk": float((m >= 1.0).mean()),
            "hit_2tk": float((m >= 2.0).mean()),
            "wr": float((L > 0).mean()),
            "passive_net": float(L.mean() - PASSIVE_COST_TK),
            "market_net": float(L.mean() - MARKET_COST_TK),
            # consistency proxy: positive-label fraction × mean-label
            "expectancy": float((L > 0).mean() * max(L[L > 0].mean(), 0) if (L > 0).any() else 0)
                          - float(((L < 0).mean()) * max(-L[L < 0].mean(), 0) if (L < 0).any() else 0),
        })
    return rows


def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ["long", "short"]:
        for h in HORIZONS:
            rows.extend(compute_band_stats(df, side, h, extra_tag="all"))
    return pd.DataFrame(rows)


def slice_terciles(df: pd.DataFrame, col: str) -> dict[str, pd.DataFrame]:
    """Split into Low / Mid / High terciles by col."""
    x = df[col].to_numpy()
    q33 = np.percentile(x, 33)
    q66 = np.percentile(x, 66)
    return {
        f"{col}_Low": df[x <= q33],
        f"{col}_Mid": df[(x > q33) & (x <= q66)],
        f"{col}_High": df[x > q66],
    }


def slice_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-tercile table for numeric filter col."""
    slices = slice_terciles(df, col)
    rows = []
    for name, sub in slices.items():
        for side in ["long", "short"]:
            for h in HORIZONS:
                rows.extend(compute_band_stats(sub, side, h, extra_tag=name))
    return pd.DataFrame(rows)


def slice_table_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for val, sub in df.groupby(col):
        for side in ["long", "short"]:
            for h in HORIZONS:
                rows.extend(compute_band_stats(sub, side, h,
                                               extra_tag=f"{col}={val}"))
    return pd.DataFrame(rows)


def write_summary(tables: dict[str, pd.DataFrame], path: Path):
    """Write a markdown summary picking the top slices by passive-net EV
    with n_signals >= 200 and consistency proxy."""
    lines = []
    lines.append("# HC #439 Phase 1 — MFE/MAE deep analysis summary")
    lines.append("")
    lines.append("Cost regime: passive_net = mean label − 0.376 ticks "
                 "(commission). market_net = mean label − 1.376 ticks "
                 "(commission + 1 tick spread).")
    lines.append("")

    # Overall — best (side, horizon, band) by passive_net with n>=500
    df = tables["overall"]
    df = df[df["n_signals"] >= 500].copy()
    df = df.sort_values("passive_net", ascending=False).head(20)
    lines.append("## Top 20 (side, horizon, band) by passive-net EV (n>=500)")
    lines.append("")
    lines.append(df[["side", "horizon", "band", "n_signals", "label_mean",
                     "passive_net", "market_net", "wr", "hit_1tk", "hit_2tk",
                     "mfe_mean", "mae_mean"]].to_markdown(index=False,
                                                          floatfmt=".3f"))
    lines.append("")

    # Best per-filter slices
    for tname, label in [
        ("by_vol", "Volatility (filter_vol_500ev_tk) terciles"),
        ("by_evt", "Event-rate (filter_evt_per_sec_30s) terciles"),
        ("by_buyaggr", "Buy-aggression (filter_buy_aggr_50) terciles"),
        ("by_spread", "Spread-proxy terciles"),
        ("by_tod", "Time-of-day buckets"),
        ("by_dow", "Day-of-week"),
    ]:
        t = tables[tname]
        if len(t) == 0:
            continue
        t = t[t["n_signals"] >= 200].copy()
        t = t.sort_values("passive_net", ascending=False).head(15)
        lines.append(f"## Top 15 slices — {label}")
        lines.append("")
        lines.append(t[["tag", "side", "horizon", "band", "n_signals",
                        "label_mean", "passive_net", "market_net", "wr",
                        "hit_1tk", "mfe_mean", "mae_mean"]
                      ].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    path.write_text("\n".join(lines))


def main():
    t0 = time.time()
    df = load_all()
    print(f"Loaded all parquets in {time.time()-t0:.1f}s")

    tables = {}

    print("Building overall table...")
    tables["overall"] = overall_table(df)
    tables["overall"].to_csv(OUT_DIR / "overall_table.csv", index=False)

    print("Slicing by volatility...")
    tables["by_vol"] = slice_table(df, "filter_vol_500ev_tk")
    tables["by_vol"].to_csv(OUT_DIR / "by_vol_table.csv", index=False)

    print("Slicing by event rate...")
    tables["by_evt"] = slice_table(df, "filter_evt_per_sec_30s")
    tables["by_evt"].to_csv(OUT_DIR / "by_evt_table.csv", index=False)

    print("Slicing by buy aggression...")
    tables["by_buyaggr"] = slice_table(df, "filter_buy_aggr_50")
    tables["by_buyaggr"].to_csv(OUT_DIR / "by_buyaggr_table.csv", index=False)

    print("Slicing by spread proxy...")
    tables["by_spread"] = slice_table(df, "filter_spread_proxy_tk")
    tables["by_spread"].to_csv(OUT_DIR / "by_spread_table.csv", index=False)

    print("Slicing by tod bucket...")
    tables["by_tod"] = slice_table_categorical(df, "tod_bucket")
    tables["by_tod"].to_csv(OUT_DIR / "by_tod_table.csv", index=False)

    print("Slicing by day-of-week...")
    tables["by_dow"] = slice_table_categorical(df, "dow")
    tables["by_dow"].to_csv(OUT_DIR / "by_dow_table.csv", index=False)

    print("Writing summary.md...")
    write_summary(tables, OUT_DIR / "summary.md")

    print(f"DONE in {time.time()-t0:.1f}s. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
