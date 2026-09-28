#!/usr/bin/env python3
"""HC #411 sub-window stability for v3.4.2 ep-1 5-date OOT.

With only 5 OOT dates, N=3 is the maximum meaningful split (other Ns leave
sub-windows with 1-2 dates, below the statistical floor). We run N=3 only.

For each of the 4 positive-net cells from the HC #413 backtest, check:
  - net > 0 in every sub-window
  - CI95lo > 0 in >=2/3 sub-windows
"""
from __future__ import annotations

import csv
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

NPZ = LVL3 / "output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz"
MFE_CFG = LVL3 / "output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/v342_ep1_eval"
OUT_CSV = OUT_DIR / "hc411_subwindow_results.csv"

# Positive-net cells from the HC #413 run
WIN_CELLS = [
    ("1s", "long", "top05"),
    ("1s", "long", "top1"),
    ("1s", "short", "top05"),
    ("1s", "short", "top1"),
]
CONF_PCT = {"top05": 0.5, "top1": 1.0, "top5": 5.0, "top10": 10.0}
N_LIST = [3]  # only 5 dates


def conf_mask(pred, mask, side, tier):
    pct = CONF_PCT[tier]
    q = 100.0 - pct
    signed = pred if side == "long" else -pred
    valid = mask & np.isfinite(signed)
    if not valid.any():
        return np.zeros_like(valid)
    thr = np.percentile(signed[valid], q)
    return valid & (signed >= thr)


def ci_low(x, z=1.96):
    n = len(x)
    if n < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - z * se)


def split_dates(n_dates, n_win):
    base, rem = divmod(n_dates, n_win)
    sizes = [base + (1 if i < rem else 0) for i in range(n_win)]
    out = []
    cur = 0
    for s in sizes:
        out.append((cur, cur + s))
        cur += s
    return out


def main():
    d = np.load(NPZ, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    fifo = load_fifo_for_dates(LABELS_DIR, oot_dates)
    n_fifo_per_day = list(fifo["_n_per_day"])
    n_use = min(n, sum(n_fifo_per_day))
    cum = np.cumsum([0] + n_fifo_per_day)

    mfe_df = pd.read_csv(MFE_CFG)
    mfe_df = mfe_df[mfe_df["model"] == "v3.4.2"].reset_index(drop=True)

    rows = []
    cell_summary = []
    for h, side, tier in WIN_CELLS:
        cell_id = f"v3.4.2_{h}_{side}_{tier}"
        mr = mfe_df[(mfe_df["horizon"] == h) & (mfe_df["side"] == side)].iloc[0]
        mfe = float(mr[f"mfe_{tier}"]); mae = float(mr[f"mae_{tier}"])

        pred = np.asarray(d[f"pred_log_ret_{h}"][:n_use], dtype=np.float64)
        pmsk = np.asarray(d[f"mask_log_ret_{h}"][:n_use], dtype=bool) & np.isfinite(pred)
        sig_mask = conf_mask(pred, pmsk, side, tier)
        cfg = FillSimConfig(order_type="passive_at_touch", cancel_eval_window=40, side=side)
        entry_mask_all = entry_filled_mask(fifo, cfg, seed=42)
        entry_mask = sig_mask & entry_mask_all[:n_use]
        fill_idx = np.where(entry_mask)[0]
        side_sign = +1.0 if side == "long" else -1.0
        targets, masks = {}, {}
        for hh in HORIZONS_ORDERED:
            tk = f"target_log_ret_{hh}"
            if tk in d.files:
                arr = np.asarray(d[tk][:n_use][fill_idx], dtype=np.float64)
                mk_arr = np.asarray(d[f"mask_log_ret_{hh}"][:n_use][fill_idx], dtype=bool) & np.isfinite(arr)
                targets[hh] = side_sign * arr
                masks[hh] = mk_arr
        thr = thresholds_from_mfe_mae(mfe, mae)
        gross, exit_code = resolve_exits(targets, masks, thr)
        cost = entry_cost_ticks("passive_at_touch")
        net = gross - cost
        valid = exit_code != 0
        net = net[valid]; fill_idx = fill_idx[valid]
        day_idx = np.searchsorted(cum, fill_idx, side="right") - 1

        for nw in N_LIST:
            bounds = split_dates(len(oot_dates), nw)
            all_pos = True; ci_pos = 0
            for w, (lo, hi) in enumerate(bounds):
                in_win = (day_idx >= lo) & (day_idx < hi)
                w_net = net[in_win]
                n_w = w_net.size
                if n_w == 0:
                    mean_net = float("nan"); cil = float("nan"); wr = float("nan")
                    day_conc = float("nan"); pdpr = float("nan"); n_act = 0
                else:
                    mean_net = float(w_net.mean())
                    cil = ci_low(w_net)
                    wr = float((w_net > 0).mean() * 100)
                    rel = day_idx[in_win] - lo
                    counts = np.bincount(rel, minlength=hi - lo)
                    day_conc = float(counts.max() / n_w)
                    n_act = int((counts > 0).sum())
                    daily = []
                    for dd in range(hi - lo):
                        if counts[dd] == 0:
                            continue
                        daily.append(w_net[rel == dd].mean() > 0)
                    pdpr = float(np.mean(daily)) if daily else float("nan")
                if not (np.isfinite(mean_net) and mean_net > 0):
                    all_pos = False
                if np.isfinite(cil) and cil > 0:
                    ci_pos += 1
                rows.append(dict(
                    cell_id=cell_id, n_windows=nw, sub_window=w,
                    date_lo=oot_dates[lo], date_hi=oot_dates[hi - 1],
                    n_fills=n_w, n_active_days=n_act,
                    realized_net_per_fill=round(mean_net, 4) if np.isfinite(mean_net) else float("nan"),
                    ci_low_95_net=round(cil, 4) if np.isfinite(cil) else float("nan"),
                    day_conc=round(day_conc, 4) if np.isfinite(day_conc) else float("nan"),
                    wr_pct=round(wr, 2) if np.isfinite(wr) else float("nan"),
                    per_day_pass_rate=round(pdpr, 4) if np.isfinite(pdpr) else float("nan"),
                ))
            ci_thr = max(2, int(np.ceil(2 * nw / 3)))
            cell_summary.append(dict(cell_id=cell_id, n_windows=nw,
                                     all_net_positive=all_pos,
                                     ci_pos_count=ci_pos, ci_threshold=ci_thr,
                                     regime_stable=all_pos and (ci_pos >= ci_thr)))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(cell_summary).to_csv(OUT_DIR / "hc411_subwindow_summary.csv", index=False)
    for cs in cell_summary:
        print(f"[hc411] {cs['cell_id']:35s} N={cs['n_windows']} "
              f"all_pos={cs['all_net_positive']} ci_pos={cs['ci_pos_count']}/{cs['n_windows']} "
              f"stable={cs['regime_stable']}")
    print(f"[hc411] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
