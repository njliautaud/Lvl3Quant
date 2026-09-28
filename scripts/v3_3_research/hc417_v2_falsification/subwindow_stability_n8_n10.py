"""
HC #417 — Extended HC #411 sub-window stability for cell `v2_1s_short_top05`.

Adds N=8 and N=10 to the existing N=3/4/6 falsification.  This is a NEW WRAPPER:
it imports the helper functions from the original `subwindow_stability.py` and
re-runs ONLY the single passing cell.  The original script is NOT modified.

Honest-power gating per task spec:
  - n_fills < 10 in a sub-window  ==>  underpowered (not a falsification)
  - n_fills >= 10 and net < 0      ==>  REGIME FLIP (true falsification)

Outputs:
  output/hc417_hc411_subwindow_v2_extended/sub_window_stability_n8_n10.csv
  output/hc417_hc411_subwindow_v2_extended/verdict.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts/hc413_scalping_backtester"))
sys.path.insert(0, str(LVL3 / "scripts/v3_3_research/hc417_v2_falsification"))

from fill_sim import (  # noqa: E402
    FillSimConfig, entry_filled_mask, entry_cost_ticks,
    load_fifo_for_dates,
)
from tp_sl_rules import thresholds_from_mfe_mae, resolve_exits, HORIZONS_ORDERED  # noqa: E402

# Reuse helpers from the original (DO NOT MODIFY) script
from subwindow_stability import (  # noqa: E402
    confidence_mask, mean_ci_lower, split_dates, compute_fill_arrays,
)

NPZ_PATH = LVL3 / "output/hc417_v2_full_oot_wrapped_for_hc413.npz"
MFE_CFG = LVL3 / "output/hc417_v2_native_mfe_matrix.csv"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/hc417_hc411_subwindow_v2_extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Only the single passing cell from HC #417 verdict
TARGET_CELLS = [("1s", "short", "top05")]
N_WINDOWS_LIST = [8, 10]
UNDERPOWER_FLOOR = 10  # n_fills < 10 => stat-underpowered, not a flip


def main():
    d = np.load(NPZ_PATH, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    n_dates = len(oot_dates)
    fifo = load_fifo_for_dates(LABELS_DIR, oot_dates)
    fifo_n_per_day = list(fifo["_n_per_day"])
    n_fifo = sum(fifo_n_per_day)
    n_use = min(n, n_fifo)
    cum = np.cumsum([0] + fifo_n_per_day)
    print(f"[sw_ext] n_dates={n_dates} n_use={n_use:,}")

    mfe_df = pd.read_csv(MFE_CFG)
    mfe_df = mfe_df[mfe_df["model"] == "v2"].reset_index(drop=True)

    rows = []
    cell_summary = []
    for horizon, side, tier in TARGET_CELLS:
        cell_id = f"v2_{horizon}_{side}_{tier}"
        mrow = mfe_df[(mfe_df["horizon"] == horizon) & (mfe_df["side"] == side)].iloc[0]
        mfe = float(mrow[f"mfe_{tier}"]); mae = float(mrow[f"mae_{tier}"])
        print(f"[sw_ext] cell={cell_id} mfe={mfe:.3f} mae={mae:.3f}")

        net, fill_idx = compute_fill_arrays(d, n_use, fifo, horizon, side, tier, mfe, mae)
        day_idx_per_fill = np.searchsorted(cum, fill_idx, side="right") - 1
        print(f"[sw_ext] total fills={len(net)}  net_mean={net.mean():+.4f}  n_active_days={len(set(day_idx_per_fill.tolist()))}")

        for n_win in N_WINDOWS_LIST:
            bounds = split_dates(n_dates, n_win)
            net_pos_all = True
            true_flip_count = 0       # n_fills>=floor AND net<0
            underpowered_count = 0    # n_fills<floor
            ci_pos_count = 0
            for w_idx, (lo, hi) in enumerate(bounds):
                in_win = (day_idx_per_fill >= lo) & (day_idx_per_fill < hi)
                w_net = net[in_win]
                n_w = w_net.size
                if n_w == 0:
                    mean_net = float("nan"); ci_low = float("nan"); wr = float("nan")
                    day_conc = float("nan"); pdpr = float("nan"); n_act_days = 0
                    label = "empty"
                else:
                    mean_net = float(w_net.mean())
                    ci_low = mean_ci_lower(w_net)
                    wr = float((w_net > 0).mean() * 100)
                    rel_days = day_idx_per_fill[in_win] - lo
                    counts = np.bincount(rel_days, minlength=hi - lo)
                    day_conc = float(counts.max() / n_w)
                    n_act_days = int((counts > 0).sum())
                    daily_pdpr = []
                    for dd in range(hi - lo):
                        if counts[dd] == 0:
                            continue
                        mask_dd = rel_days == dd
                        daily_pdpr.append(w_net[mask_dd].mean() > 0)
                    pdpr = float(np.mean(daily_pdpr)) if daily_pdpr else float("nan")
                    if n_w < UNDERPOWER_FLOOR:
                        label = "underpowered"
                    elif mean_net <= 0:
                        label = "regime_flip"
                    else:
                        label = "pass"
                # Tally bookkeeping
                if not (np.isfinite(mean_net) and mean_net > 0):
                    net_pos_all = False
                    if n_w >= UNDERPOWER_FLOOR:
                        true_flip_count += 1
                    elif n_w > 0:
                        underpowered_count += 1
                if np.isfinite(ci_low) and ci_low > 0:
                    ci_pos_count += 1
                rows.append(dict(
                    cell_id=cell_id, n_windows=n_win, sub_window=w_idx,
                    date_lo=oot_dates[lo] if lo < n_dates else "?",
                    date_hi=oot_dates[hi-1] if (hi-1) < n_dates else "?",
                    n_fills_per_sub_window=n_w,
                    n_active_days=n_act_days,
                    realized_net_per_fill=round(mean_net, 4) if np.isfinite(mean_net) else float("nan"),
                    ci_low_95_net=round(ci_low, 4) if np.isfinite(ci_low) else float("nan"),
                    day_conc=round(day_conc, 4) if np.isfinite(day_conc) else float("nan"),
                    wr_pct=round(wr, 2) if np.isfinite(wr) else float("nan"),
                    per_day_pass_rate=round(pdpr, 4) if np.isfinite(pdpr) else float("nan"),
                    power_label=label,
                ))
            cell_summary.append(dict(
                cell_id=cell_id, n_windows=n_win,
                all_net_positive=net_pos_all,
                true_regime_flip_count=true_flip_count,
                underpowered_count=underpowered_count,
                ci_pos_count=ci_pos_count,
                falsified=(true_flip_count > 0),
            ))

    df_sub = pd.DataFrame(rows)
    df_sub.to_csv(OUT_DIR / "sub_window_stability_n8_n10.csv", index=False)
    print(f"[sw_ext] wrote {OUT_DIR/'sub_window_stability_n8_n10.csv'} ({len(df_sub)} rows)")

    df_sum = pd.DataFrame(cell_summary)
    df_sum.to_csv(OUT_DIR / "cell_summary_n8_n10.csv", index=False)

    # verdict.md
    lines = []
    lines.append("# HC #417 — Extended HC #411 sub-window stability (N=8, N=10)\n")
    lines.append(f"Cell tested: `v2_1s_short_top05`")
    lines.append(f"NPZ: `{NPZ_PATH.name}`")
    lines.append(f"MFE config: V2-NATIVE (`{MFE_CFG.name}`)")
    lines.append(f"OOT: {oot_dates[0]} -> {oot_dates[-1]} ({n_dates} dates)")
    lines.append(f"Cost: passive_at_touch = 0.376 tk; canonical FIFO replay (HC #74/#377)\n")
    lines.append("**Power gate**: n_fills < 10 in a sub-window = STAT-UNDERPOWERED (not a falsification). "
                 "Only sub-windows with n_fills >= 10 AND net < 0 count as a true regime flip.\n")

    lines.append("## Summary table\n")
    lines.append("| N | all_net_pos | true_flips (n>=10, net<0) | underpowered (n<10, net<=0) | ci_pos | FALSIFIED? |")
    lines.append("|---:|:-:|---:|---:|---:|:-:|")
    for r in cell_summary:
        falsified = "YES" if r["falsified"] else "no"
        anp = "YES" if r["all_net_positive"] else "no"
        lines.append(f"| {r['n_windows']} | {anp} | {r['true_regime_flip_count']} | "
                     f"{r['underpowered_count']} | {r['ci_pos_count']}/{r['n_windows']} | {falsified} |")
    lines.append("")

    lines.append("## Sub-window detail\n")
    for n_win in N_WINDOWS_LIST:
        sub = df_sub[df_sub["n_windows"] == n_win]
        lines.append(f"### N={n_win}\n")
        lines.append("| win | dates | n_fills | n_days | net/fill | CI95lo | day_conc | pdpr | wr% | label |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|:-:|")
        for _, r in sub.iterrows():
            def f(x, fmt="{:.3f}"):
                return fmt.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"
            lines.append(
                f"| {int(r['sub_window'])} | {r['date_lo']}..{r['date_hi']} | "
                f"{int(r['n_fills_per_sub_window'])} | {int(r['n_active_days'])} | "
                f"{f(r['realized_net_per_fill'])} | {f(r['ci_low_95_net'])} | "
                f"{f(r['day_conc'])} | {f(r['per_day_pass_rate'])} | {f(r['wr_pct'])} | "
                f"{r['power_label']} |"
            )
        lines.append("")

    lines.append("## Bottom line\n")
    any_falsified = any(r["falsified"] for r in cell_summary)
    if any_falsified:
        first = next(r for r in cell_summary if r["falsified"])
        lines.append(f"- **FALSIFIED at N={first['n_windows']}**: at least one sub-window has n>=10 fills AND net<0.")
        lines.append("- Cell is NOT regime-stable beyond the previously-tested N<=6.")
    else:
        passed_n = [r["n_windows"] for r in cell_summary if not r["falsified"]]
        lines.append(f"- **Not falsified** at N in {passed_n}. No sub-window with n_fills>=10 had net<0.")
        lines.append("- Some sub-windows may be underpowered (n_fills<10); see power_label column.")
    (OUT_DIR / "verdict.md").write_text("\n".join(lines))
    print(f"[sw_ext] wrote {OUT_DIR/'verdict.md'}")


if __name__ == "__main__":
    main()
