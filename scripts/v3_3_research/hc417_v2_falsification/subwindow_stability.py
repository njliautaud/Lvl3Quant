"""
HC #417 falsification — Phase 3: HC #411 sub-window stability for v2 winning cells.

For each of the 5 winning cells (from v2-native MFE backtest), split the 36 OOT dates
into N non-overlapping contiguous sub-windows (try N=3, N=4, N=6), and per sub-window
compute:
  - n_fills, realized_net_per_fill, ci_low_95_net, day_conc, per_day_pass_rate, wr%

A cell is "regime_stable_v2" if:
  - net > 0 in every sub-window
  - CI95lo > 0 in at least 2/3 sub-windows (per task spec)

Outputs:
  output/hc417_hc411_subwindow_v2/sub_window_stability.csv
  output/hc417_hc411_subwindow_v2/verdict.md

Uses canonical FIFO market replay (HC #74/#377/#397B).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))
sys.path.insert(0, str(LVL3 / "scripts/hc413_scalping_backtester"))

from fill_sim import (  # noqa: E402
    FillSimConfig, entry_filled_mask, entry_cost_ticks,
    load_fifo_for_dates,
)
from tp_sl_rules import thresholds_from_mfe_mae, resolve_exits, HORIZONS_ORDERED  # noqa: E402

NPZ_PATH = LVL3 / "output/hc417_v2_full_oot_wrapped_for_hc413.npz"
MFE_CFG = LVL3 / "output/hc417_v2_native_mfe_matrix.csv"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/hc417_hc411_subwindow_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 winning cells from v2-native MFE backtest
WIN_CELLS = [
    ("1s",  "short", "top05"),
    ("1s",  "short", "top1"),
    ("5s",  "short", "top1"),
    ("5s",  "short", "top05"),
    ("10s", "short", "top1"),
]

CONF_PCT = {"top05": 0.5, "top1": 1.0, "top5": 5.0, "top10": 10.0}
N_WINDOWS_LIST = [3, 4, 6]


def confidence_mask(pred, mask, side, conf_tier):
    pct = CONF_PCT[conf_tier]
    cutoff_q = 100.0 - pct
    signed = pred if side == "long" else -pred
    valid = mask & np.isfinite(signed)
    if not valid.any():
        return np.zeros_like(valid)
    thr = np.percentile(signed[valid], cutoff_q)
    return valid & (signed >= thr)


def mean_ci_lower(arr, z=1.96):
    n = len(arr)
    if n < 2:
        return float("nan")
    se = arr.std(ddof=1) / np.sqrt(n)
    return float(arr.mean() - z * se)


def split_dates(n_dates: int, n_windows: int):
    base = n_dates // n_windows
    rem = n_dates % n_windows
    sizes = [base + (1 if i < rem else 0) for i in range(n_windows)]
    bounds = []
    cursor = 0
    for s in sizes:
        bounds.append((cursor, cursor + s))
        cursor += s
    return bounds


def compute_fill_arrays(d, n_use, fifo, horizon, side, tier, mfe, mae):
    pred = d[f"pred_log_ret_{horizon}"][:n_use].astype(np.float64)
    pmask = d[f"mask_log_ret_{horizon}"][:n_use].astype(bool) & np.isfinite(pred)
    sig_mask = confidence_mask(pred, pmask, side, tier)
    fill_cfg = FillSimConfig(order_type="passive_at_touch",
                             cancel_eval_window=40, side=side)
    entry_mask_all = entry_filled_mask(fifo, fill_cfg, seed=42)
    entry_mask = sig_mask & entry_mask_all[:n_use]
    fill_idx = np.where(entry_mask)[0]
    side_sign = +1.0 if side == "long" else -1.0
    targets = {}; masks = {}
    for h in HORIZONS_ORDERED:
        tk = f"target_log_ret_{h}"
        if tk in d.keys():
            arr = d[tk][:n_use][fill_idx]
            mk = d[f"mask_log_ret_{h}"][:n_use][fill_idx].astype(bool) & np.isfinite(arr)
            targets[h] = side_sign * arr
            masks[h] = mk
    thr = thresholds_from_mfe_mae(mfe, mae)
    gross, exit_code = resolve_exits(targets, masks, thr)
    cost = entry_cost_ticks("passive_at_touch")
    net = gross - cost
    valid = exit_code != 0
    return net[valid], fill_idx[valid]


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
    print(f"[sw] n_dates={n_dates} n_use={n_use:,}")

    mfe_df = pd.read_csv(MFE_CFG)
    mfe_df = mfe_df[mfe_df["model"] == "v2"].reset_index(drop=True)

    rows = []
    cell_summary = []  # one row per (cell, n_windows) with verdict
    for horizon, side, tier in WIN_CELLS:
        cell_id = f"v2_{horizon}_{side}_{tier}"
        mrow = mfe_df[(mfe_df["horizon"] == horizon) & (mfe_df["side"] == side)].iloc[0]
        mfe = float(mrow[f"mfe_{tier}"]); mae = float(mrow[f"mae_{tier}"])
        print(f"[sw] cell={cell_id} mfe={mfe:.3f} mae={mae:.3f}")

        net, fill_idx = compute_fill_arrays(d, n_use, fifo, horizon, side, tier, mfe, mae)
        # Map each fill to its date_idx
        day_idx_per_fill = np.searchsorted(cum, fill_idx, side="right") - 1

        for n_win in N_WINDOWS_LIST:
            bounds = split_dates(n_dates, n_win)
            net_pos_all = True
            ci_pos_count = 0
            per_window = []
            for w_idx, (lo, hi) in enumerate(bounds):
                in_win = (day_idx_per_fill >= lo) & (day_idx_per_fill < hi)
                w_net = net[in_win]
                n_w = w_net.size
                if n_w == 0:
                    mean_net = float("nan"); ci_low = float("nan"); wr = float("nan")
                    day_conc = float("nan"); pdpr = float("nan"); n_act_days = 0
                else:
                    mean_net = float(w_net.mean())
                    ci_low = mean_ci_lower(w_net)
                    wr = float((w_net > 0).mean() * 100)
                    rel_days = day_idx_per_fill[in_win] - lo
                    counts = np.bincount(rel_days, minlength=hi - lo)
                    day_conc = float(counts.max() / n_w)
                    # per_day_pass_rate
                    n_act_days = int((counts > 0).sum())
                    daily_pdpr = []
                    for dd in range(hi - lo):
                        if counts[dd] == 0:
                            continue
                        mask_dd = rel_days == dd
                        daily_pdpr.append(w_net[mask_dd].mean() > 0)
                    pdpr = float(np.mean(daily_pdpr)) if daily_pdpr else float("nan")
                # 2/3 rule: count windows where net>0 AND ci_low>0
                if not (np.isfinite(mean_net) and mean_net > 0):
                    net_pos_all = False
                if np.isfinite(ci_low) and ci_low > 0:
                    ci_pos_count += 1
                rows.append(dict(
                    cell_id=cell_id, n_windows=n_win, sub_window=w_idx,
                    window_date_lo_idx=lo, window_date_hi_excl=hi,
                    date_lo=oot_dates[lo] if lo < n_dates else "?",
                    date_hi=oot_dates[hi-1] if (hi-1) < n_dates else "?",
                    n_fills_per_sub_window=n_w,
                    n_active_days=n_act_days,
                    realized_net_per_fill=round(mean_net, 4) if np.isfinite(mean_net) else float("nan"),
                    ci_low_95_net=round(ci_low, 4) if np.isfinite(ci_low) else float("nan"),
                    day_conc=round(day_conc, 4) if np.isfinite(day_conc) else float("nan"),
                    wr_pct=round(wr, 2) if np.isfinite(wr) else float("nan"),
                    per_day_pass_rate=round(pdpr, 4) if np.isfinite(pdpr) else float("nan"),
                ))
                per_window.append((mean_net, ci_low, n_w))
            ci_threshold = max(2, int(np.ceil(2 * n_win / 3)))  # >= 2/3 of windows
            regime_stable = net_pos_all and (ci_pos_count >= ci_threshold)
            cell_summary.append(dict(
                cell_id=cell_id, n_windows=n_win,
                all_net_positive=net_pos_all,
                ci_pos_count=ci_pos_count,
                ci_threshold=ci_threshold,
                regime_stable_v2=regime_stable,
            ))

    df_sub = pd.DataFrame(rows)
    df_sub.to_csv(OUT_DIR / "sub_window_stability.csv", index=False)
    print(f"[sw] wrote {OUT_DIR/'sub_window_stability.csv'} ({len(df_sub)} rows)")

    df_sum = pd.DataFrame(cell_summary)
    df_sum.to_csv(OUT_DIR / "cell_summary.csv", index=False)
    print(f"[sw] wrote {OUT_DIR/'cell_summary.csv'}")

    # verdict.md
    lines = []
    lines.append("# HC #417 falsification — Phase 3: HC #411 sub-window stability (v2)\n")
    lines.append(f"NPZ: `{NPZ_PATH}`")
    lines.append(f"MFE config: `{MFE_CFG.name}` (V2-NATIVE)")
    lines.append(f"OOT: {oot_dates[0]} -> {oot_dates[-1]} ({n_dates} dates)")
    lines.append(f"Cost: passive_at_touch = 0.376 tk; canonical FIFO replay\n")

    lines.append("## Verdict per cell\n")
    lines.append("Regime-stable_v2 = net>0 in EVERY sub-window AND CI95lo>0 in >=2/3 sub-windows.\n")
    lines.append("| cell_id | N=3 stable | N=3 ci_pos | N=4 stable | N=4 ci_pos | N=6 stable | N=6 ci_pos |")
    lines.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|")
    for cell_id in [f"v2_{h}_{s}_{t}" for h, s, t in WIN_CELLS]:
        cells = {r["n_windows"]: r for r in cell_summary if r["cell_id"] == cell_id}
        def cell_v(n):
            r = cells.get(n, {})
            stable = "YES" if r.get("regime_stable_v2") else "no"
            ci = f"{r.get('ci_pos_count', 0)}/{n}"
            return stable, ci
        s3, c3 = cell_v(3); s4, c4 = cell_v(4); s6, c6 = cell_v(6)
        lines.append(f"| {cell_id} | {s3} | {c3} | {s4} | {c4} | {s6} | {c6} |")
    lines.append("")

    lines.append("## Sub-window detail\n")
    for cell_id in df_sub["cell_id"].unique():
        lines.append(f"### {cell_id}\n")
        for n_win in N_WINDOWS_LIST:
            sub = df_sub[(df_sub["cell_id"] == cell_id) & (df_sub["n_windows"] == n_win)]
            lines.append(f"**N={n_win}**\n")
            lines.append("| win | dates | n_fills | n_days | net/fill | CI95lo | day_conc | pdpr | wr% |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in sub.iterrows():
                def f(x, fmt="{:.3f}"):
                    return fmt.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else "n/a"
                lines.append(
                    f"| {int(r['sub_window'])} | {r['date_lo']}..{r['date_hi']} | "
                    f"{int(r['n_fills_per_sub_window'])} | {int(r['n_active_days'])} | "
                    f"{f(r['realized_net_per_fill'])} | {f(r['ci_low_95_net'])} | "
                    f"{f(r['day_conc'])} | {f(r['per_day_pass_rate'])} | {f(r['wr_pct'])} |"
                )
            lines.append("")

    lines.append("## Bottom line\n")
    survives_all = [r for r in cell_summary if r["n_windows"] == 4 and r["regime_stable_v2"]]
    lines.append(f"- {len(survives_all)} of {len(WIN_CELLS)} winning cells survive HC #411 sub-window stability at N=4.")
    lines.append("- See cell_summary.csv for the N=3, N=4, N=6 breakdown.")
    lines.append("- Cells that fail: net flipped negative in at least one sub-window OR <2/3 sub-windows had CI95lo>0.")
    (OUT_DIR / "verdict.md").write_text("\n".join(lines))
    print(f"[sw] wrote {OUT_DIR/'verdict.md'}")


if __name__ == "__main__":
    main()
