#!/usr/bin/env python3
"""
HC #444 R2 — Per-day cross-model agreement filter.

For each (config, fill_date), look up the daily-mean prediction sign on the
1s horizon for v3.4.2 and v3.3. Filter v2 fills to keep only those on days
where ALL THREE MODELS agree on direction (short configs => all negative,
long configs => all positive). Re-aggregate metrics.

This is the per-day coarse version of HC #444 R2's cross-model agreement
cell — a fast first-pass to decide if per-signal join is worth building.

If per-day passes: escalate to per-signal timestamp join.
If per-day fails: per-signal cannot rescue (strictly stricter filter).

Inputs:
- v3.4.2 per-date NPZs: /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz
- v3.3   per-date NPZs: /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate/oot_YYYYMMDD.npz
- v2 fills CSVs:        /home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/*_fifo_fills.csv

Output:
- /home/jupiter/Lvl3Quant/output/hc444_cross_model_perday/{config}_filtered.csv
- /home/jupiter/Lvl3Quant/output/hc444_cross_model_perday/summary.md
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
V342_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
V33_DIR  = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate"
FILLS_DIR = ROOT / "output/hc432_v342_47day_validation"
OUT_DIR = ROOT / "output/hc444_cross_model_perday"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants (HC #74 canonical)
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376

def date_to_str(d):
    if isinstance(d, (int, np.integer)):
        return f"{int(d):08d}"
    return str(d).strip()

def load_perday_signs(model_dir: Path) -> dict:
    """Return {date_str: {'mean_1s': float, 'mean_5s': float, 'mean_10s': float, 'n': int}}."""
    out = {}
    for npz_path in sorted(model_dir.glob("oot_*.npz")):
        date_str = npz_path.stem.replace("oot_", "")
        d = np.load(npz_path, allow_pickle=True)
        keys = set(d.files)
        if "pred_log_ret_1s" not in keys:
            continue
        p1 = d["pred_log_ret_1s"]
        # filter NaN-aware
        p1_valid = p1[np.isfinite(p1)]
        if p1_valid.size == 0:
            continue
        entry = {
            "n": int(p1_valid.size),
            "mean_1s": float(np.mean(p1_valid)),
            "median_1s": float(np.median(p1_valid)),
            "frac_short_1s": float(np.mean(p1_valid < 0)),
            "frac_top5pct_short_1s": float(np.mean(p1_valid[np.argsort(np.abs(p1_valid))[-max(1, p1_valid.size // 20):]] < 0)),
        }
        for h in ("5s", "10s"):
            k = f"pred_log_ret_{h}"
            if k in keys:
                p = d[k]
                pv = p[np.isfinite(p)]
                if pv.size > 0:
                    entry[f"mean_{h}"] = float(np.mean(pv))
                    entry[f"frac_short_{h}"] = float(np.mean(pv < 0))
        out[date_str] = entry
        d.close()
    return out

def main():
    print(f"[hc444] Loading v3.4.2 per-day signs from {V342_DIR}", flush=True)
    v342_signs = load_perday_signs(V342_DIR)
    print(f"[hc444]   loaded {len(v342_signs)} dates", flush=True)

    print(f"[hc444] Loading v3.3 per-day signs from {V33_DIR}", flush=True)
    v33_signs = load_perday_signs(V33_DIR)
    print(f"[hc444]   loaded {len(v33_signs)} dates", flush=True)

    # Common dates
    common_dates = sorted(set(v342_signs) & set(v33_signs))
    print(f"[hc444] common dates: {len(common_dates)}", flush=True)

    # Build a per-day cross-model sign frame
    rows = []
    for dt in common_dates:
        v342 = v342_signs[dt]
        v33  = v33_signs[dt]
        rows.append({
            "date": dt,
            "v342_mean_1s": v342["mean_1s"],
            "v342_frac_short_1s": v342["frac_short_1s"],
            "v33_mean_1s":  v33["mean_1s"],
            "v33_frac_short_1s":  v33["frac_short_1s"],
            "v342_n": v342["n"],
            "v33_n":  v33["n"],
        })
    cm_df = pd.DataFrame(rows)
    cm_df["both_short_meansign_1s"] = (cm_df["v342_mean_1s"] < 0) & (cm_df["v33_mean_1s"] < 0)
    cm_df["both_short_majority_1s"] = (cm_df["v342_frac_short_1s"] > 0.5) & (cm_df["v33_frac_short_1s"] > 0.5)
    cm_df["both_long_meansign_1s"]  = (cm_df["v342_mean_1s"] > 0) & (cm_df["v33_mean_1s"] > 0)
    cm_df["both_long_majority_1s"]  = (cm_df["v342_frac_short_1s"] < 0.5) & (cm_df["v33_frac_short_1s"] < 0.5)
    cm_df.to_csv(OUT_DIR / "per_day_cross_model_signs.csv", index=False)
    print(f"[hc444] cross-model per-day signs saved. Short-aligned days: meansign={cm_df['both_short_meansign_1s'].sum()}, majority={cm_df['both_short_majority_1s'].sum()}", flush=True)

    # Build dt → {meansign, majority} lookup
    short_meansign_dates = set(cm_df.loc[cm_df["both_short_meansign_1s"], "date"])
    short_majority_dates = set(cm_df.loc[cm_df["both_short_majority_1s"], "date"])
    long_meansign_dates  = set(cm_df.loc[cm_df["both_long_meansign_1s"],  "date"])
    long_majority_dates  = set(cm_df.loc[cm_df["both_long_majority_1s"],  "date"])

    # Walk every fills CSV
    fills_csvs = sorted(FILLS_DIR.glob("*_fifo_fills.csv"))
    print(f"[hc444] found {len(fills_csvs)} fills CSVs", flush=True)

    summary_rows = []
    for csv_path in fills_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  SKIP {csv_path.name}: {e}", flush=True)
            continue
        if df.empty or "date" not in df.columns or "net_ticks" not in df.columns or "direction" not in df.columns:
            continue
        n_total = len(df)
        if n_total < 50:
            # too few to evaluate
            continue
        df["date"] = df["date"].astype(str).str.zfill(8)
        # detect predominant side
        side_counts = df["direction"].value_counts()
        primary_side = side_counts.idxmax()
        # Choose the cross-model filter matching the side
        if primary_side == "short":
            keep_meansign = short_meansign_dates
            keep_majority = short_majority_dates
        else:
            keep_meansign = long_meansign_dates
            keep_majority = long_majority_dates

        def metrics(sub):
            n = len(sub)
            if n == 0:
                return dict(n=0, mean_tk=np.nan, pf=np.nan, wr=np.nan, sharpe=np.nan, days=0, pos_days=0, day_pct=np.nan)
            net = sub["net_ticks"].values.astype(float)
            wins = net[net > 0]
            losses = net[net < 0]
            pf = float(wins.sum() / -losses.sum()) if losses.size > 0 and losses.sum() < 0 else np.nan
            wr = float((net > 0).mean())
            mean_tk = float(net.mean())
            sharpe = float(net.mean() / net.std() * np.sqrt(n)) if net.std() > 0 else np.nan
            per_day = sub.groupby("date")["net_ticks"].sum()
            pos_days = int((per_day > 0).sum())
            return dict(n=n, mean_tk=mean_tk, pf=pf, wr=wr, sharpe=sharpe,
                        days=len(per_day), pos_days=pos_days,
                        day_pct=float(pos_days / max(1, len(per_day))))

        base = metrics(df)
        sub_meansign = df[df["date"].isin(keep_meansign)]
        m_meansign = metrics(sub_meansign)
        sub_majority = df[df["date"].isin(keep_majority)]
        m_majority = metrics(sub_majority)

        # Save filtered csvs
        config_name = csv_path.stem.replace("_fifo_fills", "")
        sub_meansign.to_csv(OUT_DIR / f"{config_name}__cm_meansign.csv", index=False)
        sub_majority.to_csv(OUT_DIR / f"{config_name}__cm_majority.csv", index=False)

        summary_rows.append({
            "config": config_name,
            "side": primary_side,
            "base_n": base["n"], "base_mean_tk": base["mean_tk"], "base_pf": base["pf"], "base_wr": base["wr"],
            "base_days": base["days"], "base_pos_days": base["pos_days"], "base_day_pct": base["day_pct"],
            "ms_n": m_meansign["n"], "ms_mean_tk": m_meansign["mean_tk"], "ms_pf": m_meansign["pf"], "ms_wr": m_meansign["wr"],
            "ms_days": m_meansign["days"], "ms_pos_days": m_meansign["pos_days"], "ms_day_pct": m_meansign["day_pct"],
            "ms_sharpe": m_meansign["sharpe"],
            "mj_n": m_majority["n"], "mj_mean_tk": m_majority["mean_tk"], "mj_pf": m_majority["pf"], "mj_wr": m_majority["wr"],
            "mj_days": m_majority["days"], "mj_pos_days": m_majority["pos_days"], "mj_day_pct": m_majority["day_pct"],
            "mj_sharpe": m_majority["sharpe"],
        })

    summary = pd.DataFrame(summary_rows)
    summary["ms_mean_tk_net"] = summary["ms_mean_tk"] - ES_RT_COMMISSION_TICKS
    summary["mj_mean_tk_net"] = summary["mj_mean_tk"] - ES_RT_COMMISSION_TICKS
    summary["base_mean_tk_net"] = summary["base_mean_tk"] - ES_RT_COMMISSION_TICKS

    summary = summary.sort_values("ms_mean_tk_net", ascending=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # Markdown summary
    lines = [
        "# HC #444 R2 — Per-Day Cross-Model Agreement Filter",
        "",
        f"Cross-model filter: keep v2 fills only on days where v3.4.2 AND v3.3 both lean SHORT (or both LONG, depending on config side) at the 1s horizon.",
        "",
        "- **meansign** filter: both model's day-mean pred_log_ret_1s has matching sign.",
        "- **majority** filter: both models have >50% of predictions in matching direction.",
        "",
        f"Cross-model day-counts: short-meansign={len(short_meansign_dates)}, short-majority={len(short_majority_dates)}, "
        f"long-meansign={len(long_meansign_dates)}, long-majority={len(long_majority_dates)} out of {len(common_dates)} common dates.",
        "",
        "## Results (sorted by meansign net tk/fill)",
        "",
        "| config | side | base n / mean_tk_net / day% | meansign n / mean_tk_net / pf / wr / day% / sharpe | majority n / mean_tk_net / pf / wr / day% / sharpe |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['config']} | {r['side']} | "
            f"{int(r['base_n'])} / {r['base_mean_tk_net']:.3f} / {r['base_day_pct']:.0%} | "
            f"{int(r['ms_n'])} / {r['ms_mean_tk_net']:.3f} / {r['ms_pf']:.2f} / {r['ms_wr']:.1%} / {r['ms_day_pct']:.0%} / {r['ms_sharpe']:.2f} | "
            f"{int(r['mj_n'])} / {r['mj_mean_tk_net']:.3f} / {r['mj_pf']:.2f} / {r['mj_wr']:.1%} / {r['mj_day_pct']:.0%} / {r['mj_sharpe']:.2f} |"
        )

    # Verdict
    lines += ["", "## Verdict", ""]
    has_winner = False
    winners = summary[(summary["ms_mean_tk_net"] > 0) & (summary["ms_n"] >= 100) & (summary["ms_day_pct"] >= 0.60)]
    if not winners.empty:
        has_winner = True
        lines.append("**PASS (meansign)**: configs surviving filter:")
        for _, r in winners.iterrows():
            lines.append(f"- `{r['config']}` (side {r['side']}): n={int(r['ms_n'])}, net={r['ms_mean_tk_net']:.3f} tk, PF={r['ms_pf']:.2f}, WR={r['ms_wr']:.1%}, day%={r['ms_day_pct']:.0%}, Sharpe={r['ms_sharpe']:.2f}")
    else:
        winners_mj = summary[(summary["mj_mean_tk_net"] > 0) & (summary["mj_n"] >= 100) & (summary["mj_day_pct"] >= 0.60)]
        if not winners_mj.empty:
            has_winner = True
            lines.append("**PASS (majority only)**: configs surviving stricter majority filter:")
            for _, r in winners_mj.iterrows():
                lines.append(f"- `{r['config']}` (side {r['side']}): n={int(r['mj_n'])}, net={r['mj_mean_tk_net']:.3f} tk, PF={r['mj_pf']:.2f}, WR={r['mj_wr']:.1%}, day%={r['mj_day_pct']:.0%}, Sharpe={r['mj_sharpe']:.2f}")
        else:
            lines.append("**NO CONFIG SURVIVES PER-DAY CROSS-MODEL FILTER**. Per-signal join would not rescue any config (strictly stricter filter). Ship HC #444 R4 fallback.")

    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"[hc444] wrote summary.md and summary.csv ({len(summary)} configs)", flush=True)
    print("\n" + "\n".join(lines[:40]))

    return 0 if has_winner else 2  # exit 2 = no winner, ship fallback

if __name__ == "__main__":
    sys.exit(main())
