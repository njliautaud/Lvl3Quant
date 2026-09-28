#!/usr/bin/env python3
"""
v3.3 SIDE-FLIP + HOLD-HORIZON SWEEP

Discovery (2026-05-15): the v3.3 realistic-revalidation morning batch evaluated
14 configs as SHORT but the model is in fact predicting UPWARD moves on the
selected universe (realized log_ret_30s mean = +20.68 ticks on K=2). Treating
those signals as SHORT loses ~-25t/fill (commission+adverse). The signals may
be profitable on the LONG side, or on a shorter hold horizon.

THIS SCRIPT runs the canonical full-market-replay engine across:
  - 14 configs (5 solo + 9 stacked K∈{1,2,3} × band∈{0.05,0.10,0.20})
  - BOTH sides: LONG and SHORT
  - 4 hold horizons: 1s, 5s, 10s, 30s
  - 2 entry styles: passive_at_touch (40-eval cancel) and ioc_market

Per-config metrics: n_signals, n_filled, fill_rate, mean_ticks_per_fill/trade,
WR, PF, annualized Sharpe, annualized Sortino, toxic-5s fill rate, max single-day
concentration (HC #344 gate ≤20%).

OUTPUT:
  output/v3_3_full_execution_analysis_20260514/side_horizon_sweep/
    sweep_summary.csv

Pure analysis; reuses canonical full_market_replay primitives without modifying
them. Read-only on data dirs; writes only to its OUT_DIR.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    _load_fifo_labels,
    _queue_position_model,
    _adverse_selection,
    _annualized,
    _profit_factor,
    _max_drawdown_ticks,
    _entry_price_edge_ticks,
    _pick_exit_horizon,
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    PRICE_UNIT_TO_TICKS,
)

PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
FIFO_LABELS_DIR = ROOT / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/side_horizon_sweep"

# ---------------------------------------------------------------------------
# Config catalog — SAME 14 configs that v33_realistic_revalidation.py evaluated.
# Each "sign" here is the SHORT-orienting sign used to build the selection
# (i.e. top band-frac of sign*pred = SHORT-oriented universe). The SAME universe
# is then traded as LONG and SHORT below.
# ---------------------------------------------------------------------------
SOLO_CONFIGS = [
    ("solo_log_ret_60s_pos_0p001",      "pred_log_ret_60s",        +1, 0.001),
    ("solo_log_ret_5min_neg_0p001",     "pred_log_ret_5min",       -1, 0.001),
    ("solo_p_reversal_60s_neg_0p001",   "pred_p_reversal_60s",     -1, 0.001),
    ("solo_fifo_tp8sl5_net_neg_0p001",  "pred_fifo_tp8sl5_net",    -1, 0.001),
    ("solo_log_ret_5min_pos_0p001",     "pred_log_ret_5min",       +1, 0.001),
]
STACKED_HEADS_SIGNED = [
    ("pred_log_ret_60s",         +1),
    ("pred_log_ret_5min",        -1),
    ("pred_p_reversal_60s",      -1),
    ("pred_fifo_tp8sl5_net",     -1),
    ("pred_pred_mae_60s_ticks",  -1),
]
STACKED_K = [1, 2, 3]
STACKED_BANDS = [0.05, 0.10, 0.20]

SIDES = ["long", "short"]
HOLD_SECS = [1.0, 5.0, 10.0, 30.0]
ORDER_TYPES = [("passive_at_touch", 40), ("ioc_market", 1)]


def _load_all() -> dict:
    print(f"[load] {PRED_NPZ}", flush=True)
    d = np.load(PRED_NPZ, allow_pickle=True)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    needed_heads = {h for (h, _) in STACKED_HEADS_SIGNED}
    for _, h, _, _ in SOLO_CONFIGS:
        needed_heads.add(h)
    preds = {}
    for h in needed_heads:
        if h not in d.files:
            raise KeyError(f"Head missing from NPZ: {h}")
        preds[h] = d[h].astype(np.float64)

    tgt_lr, tgt_lr_mask = {}, {}
    for h in ("1s", "5s", "10s", "30s"):
        tk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        if tk in d.files:
            arr = d[tk].astype(np.float64)
            tgt_lr[h] = arr
            mask_arr = d[mk].astype(bool) if mk in d.files else None
            if mask_arr is None or mask_arr.sum() == 0:
                tgt_lr_mask[h] = np.isfinite(arr)
            else:
                tgt_lr_mask[h] = mask_arr & np.isfinite(arr)
        else:
            n = next(iter(preds.values())).shape[0]
            tgt_lr[h] = np.full(n, np.nan)
            tgt_lr_mask[h] = np.zeros(n, dtype=bool)

    oot_dates = [str(x) for x in d["oot_dates"]]
    target_fifo_net = d["target_fifo_tp4sl3_net"].astype(np.float64)
    n_total = next(iter(preds.values())).shape[0]
    return {
        "preds": preds, "fifo_mask": fifo_mask,
        "tgt_lr": tgt_lr, "tgt_lr_mask": tgt_lr_mask,
        "oot_dates": oot_dates, "target_fifo_net": target_fifo_net, "n": n_total,
    }


def signal_solo(data: dict, head: str, sign: int, band_frac: float):
    p = data["preds"][head]
    valid = data["fifo_mask"] & np.isfinite(p)
    s = sign * p
    s_valid = s[valid]
    thr = float(np.quantile(s_valid, 1.0 - band_frac))
    sel = valid & (s >= thr)
    return np.where(sel)[0]


def signal_stacked(data: dict, K: int, band_frac: float):
    chosen = STACKED_HEADS_SIGNED[:K]
    valid = data["fifo_mask"].copy()
    for h, _ in chosen:
        valid &= np.isfinite(data["preds"][h])
    sel = valid.copy()
    for h, sign in chosen:
        s = sign * data["preds"][h]
        thr = float(np.quantile(s[valid], 1.0 - band_frac))
        sel &= (s >= thr)
    return np.where(sel)[0]


def replay_for(
    data: dict, fifo: dict, sel_idx_full: np.ndarray, *,
    side: str, order_type: str, cancel_eval_window: int,
    hold_seconds: float,
    rt_commission_ticks: float = ES_RT_COMMISSION_TICKS_DEFAULT,
    spread_ticks_rth: float = ES_SPREAD_TICKS_RTH_DEFAULT,
) -> dict:
    n_total = data["n"]
    oot_dates = data["oot_dates"]
    n_fifo = int(sum(fifo["_n_per_day"]))
    n = min(n_total, n_fifo)
    sel = sel_idx_full[sel_idx_full < n]
    n_attempted = int(sel.size)

    nan_metrics = {
        "n_signals": n_attempted, "n_filled": 0, "fill_rate": 0.0,
        "mean_ticks_per_fill": float("nan"), "mean_ticks_per_trade": float("nan"),
        "wr_pct": float("nan"), "profit_factor": float("nan"),
        "sharpe_annualized": float("nan"), "sortino_annualized": float("nan"),
        "max_drawdown_ticks": 0.0,
        "adverse_selection_30s_avg_ticks": 0.0,
        "toxic_5s_pct": 0.0, "toxic_at_hold_pct": 0.0,
        "n_distinct_days": 0, "max_pct_single_day": 0.0,
        "hc344_gate_pass": False, "pnl_ticks_total": 0.0,
        "avg_queue_pos_on_arrival": float("nan"),
        "exit_horizon_used": _pick_exit_horizon(hold_seconds),
    }
    if n_attempted == 0:
        return nan_metrics

    side_sign = -1.0 if side == "short" else +1.0
    # FIFO label arrays are side-specific (tp4sl3_<side>_*); they govern queue
    # fill modeling on the simulated entry side.
    filled_lbl = fifo[f"tp4sl3_{side}_filled"][:n][sel]
    exit_reason_lbl = fifo[f"tp4sl3_{side}_exit_reason"][:n][sel]
    hold_time_lbl = fifo[f"tp4sl3_{side}_hold_time_ns"][:n][sel]
    date_idx_signal = fifo["_date_idx"][:n][sel]

    filled_mask, q_arrival, avg_q_pos = _queue_position_model(
        order_type, cancel_eval_window, filled_lbl, exit_reason_lbl, hold_time_lbl,
    )
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel[filled_idx_in_sel]
    n_filled = int(filled_mask.sum())
    if n_filled == 0:
        nan_metrics["n_signals"] = n_attempted
        nan_metrics["avg_queue_pos_on_arrival"] = avg_q_pos
        return nan_metrics

    fill_rate = n_filled / max(1, n_attempted)

    horizon_choice = _pick_exit_horizon(hold_seconds)
    lr_exit = data["tgt_lr"][horizon_choice][filled_global_idx]
    lr_mask_exit = data["tgt_lr_mask"][horizon_choice][filled_global_idx]
    edge_offset = _entry_price_edge_ticks(order_type, spread_ticks_rth)
    raw_pnl_ticks = side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - rt_commission_ticks
    raw_pnl_ticks = np.where(lr_mask_exit, raw_pnl_ticks, 0.0)

    pnl_total = float(raw_pnl_ticks.sum())
    mean_t_fill = pnl_total / n_filled
    mean_t_trade = pnl_total / max(1, n_attempted)
    sharpe = _annualized(raw_pnl_ticks, downside=False)
    sortino = _annualized(raw_pnl_ticks, downside=True)
    pf = _profit_factor(raw_pnl_ticks)
    wr = float((raw_pnl_ticks > 0).mean() * 100.0)
    mdd = _max_drawdown_ticks(raw_pnl_ticks)

    adv30 = _adverse_selection(filled_global_idx, side_sign,
                               data["tgt_lr"]["30s"], data["tgt_lr_mask"]["30s"])
    adv30_avg = float(np.nanmean(adv30)) if adv30.size and np.isfinite(np.nanmean(adv30)) else 0.0

    adv5 = _adverse_selection(filled_global_idx, side_sign,
                              data["tgt_lr"]["5s"], data["tgt_lr_mask"]["5s"])
    toxic5_pct = 0.0
    if adv5.size:
        toxic5_pct = float(((adv5 <= -1.0) & np.isfinite(adv5)).sum()) / n_filled * 100.0

    # Toxic AT HOLD horizon — adverse component at the exit horizon
    adv_h = _adverse_selection(filled_global_idx, side_sign,
                               data["tgt_lr"][horizon_choice],
                               data["tgt_lr_mask"][horizon_choice])
    toxic_at_hold = 0.0
    if adv_h.size:
        toxic_at_hold = float(((adv_h <= -1.0) & np.isfinite(adv_h)).sum()) / n_filled * 100.0

    n_per_day_filled = np.zeros(len(oot_dates), dtype=int)
    for di in range(len(oot_dates)):
        n_per_day_filled[di] = int(((date_idx_signal == di) & filled_mask).sum())
    n_distinct_days = int((n_per_day_filled > 0).sum())
    max_pct_single = float(n_per_day_filled.max() / n_filled)
    hc344_pass = (max_pct_single <= 0.20)

    return {
        "n_signals": int(n_attempted),
        "n_filled": int(n_filled),
        "fill_rate": fill_rate,
        "mean_ticks_per_fill": mean_t_fill,
        "mean_ticks_per_trade": mean_t_trade,
        "wr_pct": wr,
        "profit_factor": pf,
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
        "max_drawdown_ticks": mdd,
        "adverse_selection_30s_avg_ticks": adv30_avg,
        "toxic_5s_pct": toxic5_pct,
        "toxic_at_hold_pct": toxic_at_hold,
        "n_distinct_days": n_distinct_days,
        "max_pct_single_day": max_pct_single,
        "hc344_gate_pass": bool(hc344_pass),
        "pnl_ticks_total": pnl_total,
        "avg_queue_pos_on_arrival": avg_q_pos,
        "exit_horizon_used": horizon_choice,
    }


def run_all() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = _load_all()
    print(f"[load] n_total={data['n']:,}  fifo_fillable={int(data['fifo_mask'].sum()):,}  "
          f"oot_dates={data['oot_dates']}", flush=True)
    fifo = _load_fifo_labels(FIFO_LABELS_DIR, data["oot_dates"])
    print(f"[load] fifo labels: n_per_day={fifo['_n_per_day']}", flush=True)

    # Build the 14 selections ONCE (selections are side-agnostic)
    selections: list[tuple[str, str, np.ndarray]] = []
    for (label, head, sign, band) in SOLO_CONFIGS:
        sel = signal_solo(data, head, sign, band)
        selections.append(("solo", label, sel))
    for K in STACKED_K:
        for band in STACKED_BANDS:
            sel = signal_stacked(data, K, band)
            label = f"stacked_K{K}_band{band:.2f}"
            selections.append(("stacked", label, sel))

    print(f"[plan] {len(selections)} selections × {len(SIDES)} sides "
          f"× {len(HOLD_SECS)} holds × {len(ORDER_TYPES)} entries "
          f"= {len(selections)*len(SIDES)*len(HOLD_SECS)*len(ORDER_TYPES)} runs",
          flush=True)

    rows: list[dict] = []
    for grp, label, sel in selections:
        for side in SIDES:
            for hold in HOLD_SECS:
                for (otype, cancel_win) in ORDER_TYPES:
                    m = replay_for(
                        data, fifo, sel,
                        side=side, order_type=otype,
                        cancel_eval_window=cancel_win, hold_seconds=hold,
                    )
                    row = {
                        "group": grp,
                        "config_label": label,
                        "side": side,
                        "hold_seconds": hold,
                        "order_type": otype,
                        **m,
                    }
                    rows.append(row)

    # Rank by annualized Sharpe (descending), NaNs last
    def _key(r):
        s = r.get("sharpe_annualized", float("nan"))
        return (-s if np.isfinite(s) else 1e9)
    rows_sorted = sorted(rows, key=_key)

    csv_path = OUT_DIR / "sweep_summary.csv"
    field_order = [
        "config_label", "group", "side", "hold_seconds", "order_type",
        "n_signals", "n_filled", "fill_rate",
        "mean_ticks_per_fill", "mean_ticks_per_trade",
        "wr_pct", "profit_factor",
        "sharpe_annualized", "sortino_annualized", "max_drawdown_ticks",
        "adverse_selection_30s_avg_ticks",
        "toxic_5s_pct", "toxic_at_hold_pct",
        "n_distinct_days", "max_pct_single_day", "hc344_gate_pass",
        "exit_horizon_used", "pnl_ticks_total", "avg_queue_pos_on_arrival",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"\n[write] {csv_path}  ({len(rows)} rows)", flush=True)

    elapsed = time.time() - t0
    print(f"[done] elapsed: {elapsed:.1f}s", flush=True)

    # ── TOP-5 RANKED BY REALISTIC ANNUALIZED SHARPE ─────────────────────
    print(f"\n{'='*100}")
    print("TOP-5 by realistic annualized Sharpe (all configs, both sides, all horizons)")
    print(f"{'='*100}")
    finite_rows = [r for r in rows_sorted if np.isfinite(r.get("sharpe_annualized", float("nan")))]
    for r in finite_rows[:5]:
        gate = "PASS" if r["hc344_gate_pass"] else "FAIL"
        print(f"  {r['config_label']:<32}  side={r['side']:<5}  hold={r['hold_seconds']:>4.1f}s  "
              f"entry={r['order_type']:<17}  n_fill={r['n_filled']:>4}  "
              f"mean_t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
              f"PF={r['profit_factor']:5.2f}  Sharpe(ann)={r['sharpe_annualized']:+7.2f}  "
              f"max_day={100*r['max_pct_single_day']:5.1f}%  HC344={gate}")

    # ── HC #344 SURVIVORS (gate-passing + Sharpe > 0 + n_fill >= 30) ─────
    print(f"\n{'='*100}")
    print("HC #344 SURVIVORS — gate-pass AND Sharpe>0 AND n_filled>=30:")
    print(f"{'='*100}")
    survivors = [r for r in finite_rows
                 if r["hc344_gate_pass"] and r["sharpe_annualized"] > 0
                 and r["n_filled"] >= 30]
    if not survivors:
        print("  NONE.")
    else:
        for r in survivors[:15]:
            print(f"  PASS: {r['config_label']:<32}  side={r['side']:<5}  hold={r['hold_seconds']:>4.1f}s  "
                  f"entry={r['order_type']:<17}  n_fill={r['n_filled']:>4}  "
                  f"mean_t/fill={r['mean_ticks_per_fill']:+7.2f}  Sharpe={r['sharpe_annualized']:+7.2f}  "
                  f"max_day={100*r['max_pct_single_day']:5.1f}%")
    print(f"{'='*100}")
    return rows


def main() -> int:
    print(f"[v33_side_horizon_sweep] === SIDE × HOLD × ENTRY SWEEP ===", flush=True)
    print(f"[v33_side_horizon_sweep] commission={ES_RT_COMMISSION_TICKS_DEFAULT}t  "
          f"spread={ES_SPREAD_TICKS_RTH_DEFAULT}t", flush=True)
    run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
