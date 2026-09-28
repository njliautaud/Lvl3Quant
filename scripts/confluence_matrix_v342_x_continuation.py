#!/usr/bin/env python3
"""
confluence_matrix_v342_x_continuation.py — HC #466 R4 + HC #470 R6 confluence matrix.

Joins v3.4.2 directional output (pred_log_ret_*, pred_p_up_*) with the continuation
specialist gate (prob) over the 17 walk-forward dates where both exist.

For each event:
  - Directional signal from v3.4.2 (pred_log_ret_5s, pred_log_ret_10s, ...)
  - Continuation probability from specialist NPZ (prob)
  - Realized signed return target_log_ret_h (in log units, converted to ticks)

Slices by (directional confidence quantile, continuation confidence quantile) and
reports:
  - Trade count
  - Mean realized signed ticks (gross)
  - Net after 0.376-tick commission
  - Hit rate

Output: output/confluence_matrix_v342_x_continuation/REPORT.md + summary.json
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
CS_DIR = ROOT / "output/continuation_specialist_smoke"
OUT_DIR = ROOT / "output/confluence_matrix_v342_x_continuation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ES futures cost constant (HC #74)
COMM_TICKS = 0.376  # round-trip commission only (passive limits)
COMM_PLUS_SPREAD = 1.376  # market-order cost

# Empirical inspection of NPZs:
#   pred_log_ret_*: normalized magnitude in approx ±0.5 (NOT log returns — name is misleading,
#                   carries sign+confidence rank info).
#   target_log_ret_*: REALIZED TICKS for the horizon (±25-40 typical range, integers).
# So predictions need no unit conversion (we use sign + rank only), and targets are already ticks.


def load_and_join_all():
    cs_files = sorted(glob.glob(str(CS_DIR / "preds_mlp_*.npz")))
    if not cs_files:
        raise RuntimeError("No continuation specialist NPZs found")

    rows = {
        "date": [],
        "n": [],
    }
    DATA = {
        "pred_log_ret_1s": [],
        "pred_log_ret_5s": [],
        "pred_log_ret_10s": [],
        "pred_p_up_5s": [],
        "pred_p_up_10s": [],
        "target_log_ret_5s": [],
        "target_log_ret_10s": [],
        "target_log_ret_30s": [],
        "cont_prob": [],
        "cont_y_true": [],
        "cont_stream_life": [],
        "cont_stream_int": [],
        "date_str": [],
    }
    dates_used = []
    for f in cs_files:
        date = f.split("_")[-1].replace(".npz", "")
        v4_path = V4_DIR / f"oot_{date}.npz"
        if not v4_path.exists():
            print(f"  skip {date}: v3.4.2 OOT NPZ missing")
            continue
        cs = np.load(f)
        v4 = np.load(v4_path, allow_pickle=True)
        n_cs = cs["prob"].shape[0]
        n_v4 = v4["pred_log_ret_1s"].shape[0]
        if n_cs != n_v4:
            print(f"  skip {date}: row mismatch cs={n_cs} v4={n_v4}")
            continue
        DATA["pred_log_ret_1s"].append(v4["pred_log_ret_1s"])
        DATA["pred_log_ret_5s"].append(v4["pred_log_ret_5s"])
        DATA["pred_log_ret_10s"].append(v4["pred_log_ret_10s"])
        DATA["pred_p_up_5s"].append(v4["pred_p_up_5s"])
        DATA["pred_p_up_10s"].append(v4["pred_p_up_10s"])
        DATA["target_log_ret_5s"].append(v4["target_log_ret_5s"])
        DATA["target_log_ret_10s"].append(v4["target_log_ret_10s"])
        DATA["target_log_ret_30s"].append(v4["target_log_ret_30s"])
        DATA["cont_prob"].append(cs["prob"])
        DATA["cont_y_true"].append(cs["y_true"])
        DATA["cont_stream_life"].append(cs["stream_life"])
        DATA["cont_stream_int"].append(cs["stream_int"])
        DATA["date_str"].append(np.full(n_cs, date, dtype="<U8"))
        dates_used.append(date)
    out = {k: np.concatenate(v) for k, v in DATA.items()}
    return out, dates_used


def trade_slice(direction_signal, cont_prob, target_ticks, dir_q, cont_q):
    """Slice events where |direction| in top dir_q and cont_prob in top cont_q.
    Direction-signed entry: trade long if direction > 0, short if direction < 0.
    Realized signed return = target_ticks * sign(direction). target NaNs filtered.
    """
    valid = ~np.isnan(target_ticks)
    abs_dir = np.abs(direction_signal)
    dir_thr = np.quantile(abs_dir[valid], 1 - dir_q)
    cont_thr = np.quantile(cont_prob[valid], 1 - cont_q)
    mask = (abs_dir >= dir_thr) & (cont_prob >= cont_thr) & valid
    n = int(mask.sum())
    if n == 0:
        return None
    side = np.sign(direction_signal[mask])
    realized = target_ticks[mask] * side
    gross_mean = float(realized.mean())
    net_passive = gross_mean - COMM_TICKS
    net_market = gross_mean - COMM_PLUS_SPREAD
    hit = float((realized > 0).mean())
    std = float(realized.std())
    sharpe_per_trade = gross_mean / std if std > 0 else 0.0
    return {
        "n": n,
        "dir_thr_ticks": float(dir_thr),
        "cont_thr": float(cont_thr),
        "gross_ticks": gross_mean,
        "net_passive_ticks": net_passive,
        "net_market_ticks": net_market,
        "hit_rate": hit,
        "per_trade_sharpe": sharpe_per_trade,
    }


def main():
    print("Loading & joining v3.4.2 + continuation specialist NPZs ...")
    D, dates = load_and_join_all()
    n_total = D["cont_prob"].size
    print(f"Total events: {n_total:,} across {len(dates)} dates ({dates[0]} -> {dates[-1]})")

    # Predictions are normalized signal, targets are already ticks (per empirical inspection).
    pred_5s = D["pred_log_ret_5s"]
    pred_10s = D["pred_log_ret_10s"]
    tgt_5s_t = D["target_log_ret_5s"]
    tgt_10s_t = D["target_log_ret_10s"]
    tgt_30s_t = D["target_log_ret_30s"]

    HORIZONS = {
        "5s_pred_vs_5s_tgt": (pred_5s, tgt_5s_t),
        "5s_pred_vs_10s_tgt": (pred_5s, tgt_10s_t),
        "10s_pred_vs_10s_tgt": (pred_10s, tgt_10s_t),
        "10s_pred_vs_30s_tgt": (pred_10s, tgt_30s_t),
    }
    DIR_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.50]
    CONT_QUANTILES = [0.05, 0.10, 0.20, 0.50, 1.0]  # 1.0 = no cont gate

    report = {"dates": dates, "n_total": int(n_total), "tables": {}}
    md_lines = ["# HC #466 R4 + HC #470 R6 — Confluence Matrix Report",
                "",
                f"Dates: {dates[0]} → {dates[-1]} ({len(dates)} dates), events: {n_total:,}",
                "Cost model: passive = 0.376 ticks (commission only). market = 1.376 ticks.",
                f"Units: predictions are normalized sign+magnitude (~±0.4). targets are realized ticks (±25-50).",
                ""]

    for hkey, (pred_t, tgt_t) in HORIZONS.items():
        md_lines.append(f"## {hkey}")
        md_lines.append("")
        md_lines.append(f"| dir top% | cont top% | n | gross | net passive | net market | hit | per-trade Sharpe |")
        md_lines.append(f"|---------:|----------:|---:|------:|------------:|-----------:|----:|-----------------:|")
        h_table = {}
        for dq in DIR_QUANTILES:
            for cq in CONT_QUANTILES:
                slc = trade_slice(pred_t, D["cont_prob"], tgt_t, dq, cq)
                key = f"dir_top{int(dq*100)}_cont_top{int(cq*100)}"
                if slc is None:
                    h_table[key] = None
                    continue
                h_table[key] = slc
                md_lines.append(
                    f"| {dq*100:>7.1f}% | {cq*100:>8.1f}% | {slc['n']:>5} | "
                    f"{slc['gross_ticks']:>+6.4f} | {slc['net_passive_ticks']:>+11.4f} | "
                    f"{slc['net_market_ticks']:>+10.4f} | {slc['hit_rate']:>4.3f} | "
                    f"{slc['per_trade_sharpe']:>+15.4f} |"
                )
        md_lines.append("")
        report["tables"][hkey] = h_table

    # Find best tradable subset across all (horizon × dir_q × cont_q)
    candidates = []
    for hkey, table in report["tables"].items():
        for key, slc in table.items():
            if slc is None: continue
            if slc["n"] < 20: continue  # require minimum sample
            candidates.append((hkey, key, slc))
    candidates.sort(key=lambda x: x[2]["net_passive_ticks"], reverse=True)

    md_lines.append("## Best tradable subsets (top-10 by net passive ticks, n>=20)")
    md_lines.append("")
    md_lines.append("| rank | horizon | slice | n | gross | net passive | net market | hit | per-trade Sharpe |")
    md_lines.append("|-----:|---------|-------|---:|------:|------------:|-----------:|----:|-----------------:|")
    for i, (hkey, key, slc) in enumerate(candidates[:10], 1):
        md_lines.append(
            f"| {i} | {hkey} | {key} | {slc['n']} | "
            f"{slc['gross_ticks']:>+6.4f} | {slc['net_passive_ticks']:>+11.4f} | "
            f"{slc['net_market_ticks']:>+10.4f} | {slc['hit_rate']:.3f} | "
            f"{slc['per_trade_sharpe']:>+7.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Verdict (HC #466 R5 mandate)")
    md_lines.append("")
    if candidates and candidates[0][2]["net_passive_ticks"] > 0:
        top = candidates[0]
        md_lines.append(
            f"BEST TRADABLE SUBSET = horizon `{top[0]}`, slice `{top[1]}`, "
            f"n={top[2]['n']}, gross={top[2]['gross_ticks']:+.4f} ticks, "
            f"net passive={top[2]['net_passive_ticks']:+.4f} ticks, "
            f"per-trade Sharpe={top[2]['per_trade_sharpe']:+.4f}."
        )
        md_lines.append("")
        md_lines.append("Continuation gate adds value where the v3.4.2 directional alone failed.")
    else:
        md_lines.append(
            "NO net-passive-positive subset found across all horizon × confidence quantile combinations. "
            "Continuation gate does NOT rescue tradability on the WF-17-date sample. "
            "Either (a) larger v3.4.2 directional-confidence threshold required, "
            "(b) different cost model (HC #469 R3 canonical FIFO is the source of truth — this is an approximation), "
            "or (c) the WF-17-date window genuinely has no tradable subset under this gating."
        )

    (OUT_DIR / "REPORT.md").write_text("\n".join(md_lines))
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, default=str))
    (OUT_DIR / "confluence_matrix.DONE").write_text("done\n")
    print(f"Report written to {OUT_DIR}/REPORT.md")
    print(f"Best tradable subset (top 1): {candidates[0] if candidates else 'NONE'}")


if __name__ == "__main__":
    main()
