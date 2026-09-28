#!/usr/bin/env python3
"""
v3.3 K=1 PER-DAY-CAPPED LONG SWEEP  +  K=2 EDGE STABILITY DIAGNOSTIC
====================================================================

CONTEXT (2026-05-15):
  v33_day_balanced_search found K=2 LONG +22t edge but only fires on 2 of 5
  OOT days (20260224 and 20260227 emit ZERO K-stack signals). HC #344
  (≤20% concentration) is mechanically impossible with K=2.

  The UNTESTED option is K=1 stacked-head selection with per-day caps.
  K=1 band 0.10 fires on every day (372/2011/2112/662/895 signals).

PART 1 — K=1 PER-DAY-CAPPED LONG SWEEP
  LONG, passive_at_touch, 30s hold. Single-head selection (K=1) for each of
  the 5 long-winning heads × bands {1,5,10,20}% × per-day caps
  {5,10,20,30,50,no_cap}. Output: full canonical metrics + per-day breakdown
  + HC#344 pass + statistical sufficiency flag (n_filled >= 30).

PART 2 — K=2 EDGE STABILITY DIAGNOSTIC
  For the K=2 stack winning rule (heads=pred_log_ret_60s pos AND
  pred_log_ret_5min neg, band=0.20), compute PER-DAY: mean_ticks/fill, WR,
  n_filled, mean ticks/trade. Is +22t stable across the 2 firing days, or
  is one day carrying everything?

OUTPUTS:
  output/v3_3_full_execution_analysis_20260514/k1_capped_search/results.csv
  output/v3_3_full_execution_analysis_20260514/k1_capped_search/k2_edge_stability.json
  logs/v33_k1_capped_<timestamp>.log  (via shell redirect)

CONFORMS TO:
  HC #0 (sliding/forward only), HC #344 (<=20% single day), HC #357 (canonical
  replay engine), HC #336 (queue + adverse selection always on).

Read-only on data dirs. Imports canonical primitives WITHOUT modification.
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
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/k1_capped_search"

# --- K=1 single-head catalog: each is a candidate solo head (sign matches the
# LONG-side flip rationale spelled out by the user). All are LONG-side trades.
# Order matches v33_day_balanced_search.STACKED_HEADS_SIGNED for K=2 stability.
K1_HEADS_SIGNED = [
    ("pred_log_ret_60s",         +1),  # head 0
    ("pred_log_ret_5min",        -1),  # head 1 (LONG when model says 5min DOWN)
    ("pred_p_reversal_60s",      -1),  # head 2 (LONG when reversal NOT predicted)
    ("pred_fifo_tp8sl5_net",     -1),  # head 3 (LONG when model says FIFO net NEG → flip)
    # head 4 (pred_pred_mae_60s_ticks, -1) not part of "long-winning" 5; skip.
]

BANDS = [0.01, 0.05, 0.10, 0.20]
CAPS = [5, 10, 20, 30, 50, None]  # None = no cap

# Execution config (LONG only)
SIDE = "long"
HOLD_SECONDS = 30.0
ORDER_TYPE = "passive_at_touch"
CANCEL_WINDOW = 40

# K=2 stack definition for stability diagnostic (same as day-balanced morning)
K2_HEADS = K1_HEADS_SIGNED[:2]  # log_ret_60s +1, log_ret_5min -1
K2_BAND = 0.20  # the morning's K=2 winning band


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_all() -> dict:
    print(f"[load] {PRED_NPZ}", flush=True)
    d = np.load(PRED_NPZ, allow_pickle=True)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    needed_heads = {h for (h, _) in K1_HEADS_SIGNED}
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
    n_total = next(iter(preds.values())).shape[0]
    return {
        "preds": preds,
        "fifo_mask": fifo_mask,
        "tgt_lr": tgt_lr,
        "tgt_lr_mask": tgt_lr_mask,
        "oot_dates": oot_dates,
        "n": n_total,
    }


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------
def signal_solo_global_capped(data, fifo, head, sign, band_frac, cap_per_day):
    """K=1: top-(band_frac) globally on sign*head, then cap top-N per day by score.
       If cap_per_day is None, no per-day cap is applied."""
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    date_idx = fifo["_date_idx"][:n]
    p = data["preds"][head][:n]
    valid = data["fifo_mask"][:n] & np.isfinite(p)
    s = sign * p
    if not valid.any():
        return np.array([], dtype=np.int64)
    thr = float(np.quantile(s[valid], 1.0 - band_frac))
    base_sel = valid & (s >= thr)
    if cap_per_day is None:
        return np.where(base_sel)[0]
    final = np.zeros(n, dtype=bool)
    for di in range(len(data["oot_dates"])):
        day_idx = np.where((date_idx == di) & base_sel)[0]
        if day_idx.size == 0:
            continue
        if day_idx.size <= cap_per_day:
            final[day_idx] = True
        else:
            top = day_idx[np.argsort(-s[day_idx])[:cap_per_day]]
            final[top] = True
    return np.where(final)[0]


def signal_k2_global(data, fifo, band_frac):
    """K=2 stacked, global band, no cap (mirror morning's winner)."""
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    valid = data["fifo_mask"][:n].copy()
    for h, _ in K2_HEADS:
        valid &= np.isfinite(data["preds"][h][:n])
    sel = valid.copy()
    for h, sign in K2_HEADS:
        s = sign * data["preds"][h][:n]
        thr = float(np.quantile(s[valid], 1.0 - band_frac))
        sel &= (s >= thr)
    return np.where(sel)[0]


# ---------------------------------------------------------------------------
# Replay (LONG, passive_at_touch, 30s)
# ---------------------------------------------------------------------------
def replay_for(
    data, fifo, sel_idx_full, *,
    side=SIDE, order_type=ORDER_TYPE,
    cancel_eval_window=CANCEL_WINDOW, hold_seconds=HOLD_SECONDS,
    rt_commission_ticks=ES_RT_COMMISSION_TICKS_DEFAULT,
    spread_ticks_rth=ES_SPREAD_TICKS_RTH_DEFAULT,
    return_per_trade=False,
):
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
        "per_day_filled": [0]*len(oot_dates),
        "per_day_signals": [0]*len(oot_dates),
        "per_day_pnl_ticks": [0.0]*len(oot_dates),
        "per_day_wins": [0]*len(oot_dates),
    }
    if n_attempted == 0:
        return nan_metrics

    side_sign = -1.0 if side == "short" else +1.0
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

    per_day_signals = [int((date_idx_signal == di).sum()) for di in range(len(oot_dates))]

    if n_filled == 0:
        nan_metrics["n_signals"] = n_attempted
        nan_metrics["avg_queue_pos_on_arrival"] = avg_q_pos
        nan_metrics["per_day_signals"] = per_day_signals
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
    adv_h = _adverse_selection(filled_global_idx, side_sign,
                               data["tgt_lr"][horizon_choice],
                               data["tgt_lr_mask"][horizon_choice])
    toxic_at_hold = 0.0
    if adv_h.size:
        toxic_at_hold = float(((adv_h <= -1.0) & np.isfinite(adv_h)).sum()) / n_filled * 100.0

    # Per-day breakdowns on FILLED trades
    date_idx_filled = date_idx_signal[filled_idx_in_sel]
    per_day_filled = [int((date_idx_filled == di).sum()) for di in range(len(oot_dates))]
    per_day_pnl_ticks = []
    per_day_wins = []
    for di in range(len(oot_dates)):
        mask_di = (date_idx_filled == di)
        if mask_di.any():
            day_pnl = raw_pnl_ticks[mask_di]
            per_day_pnl_ticks.append(float(day_pnl.sum()))
            per_day_wins.append(int((day_pnl > 0).sum()))
        else:
            per_day_pnl_ticks.append(0.0)
            per_day_wins.append(0)

    n_distinct_days = sum(1 for x in per_day_filled if x > 0)
    max_pct_single = max(per_day_filled) / n_filled if n_filled > 0 else 0.0
    hc344_pass = (max_pct_single <= 0.20)

    out = {
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
        "per_day_filled": per_day_filled,
        "per_day_signals": per_day_signals,
        "per_day_pnl_ticks": per_day_pnl_ticks,
        "per_day_wins": per_day_wins,
    }
    if return_per_trade:
        out["_raw_pnl_ticks"] = raw_pnl_ticks
        out["_date_idx_filled"] = date_idx_filled
    return out


# ---------------------------------------------------------------------------
# PART 1: K=1 capped sweep
# ---------------------------------------------------------------------------
def part1_k1_capped_sweep(data, fifo):
    dates = data["oot_dates"]
    rows: list[dict] = []
    print(f"\n[PART 1] K=1 single-head per-day-capped sweep "
          f"({len(K1_HEADS_SIGNED)} heads × {len(BANDS)} bands × {len(CAPS)} caps)", flush=True)
    for (head, sign) in K1_HEADS_SIGNED:
        for band in BANDS:
            for cap in CAPS:
                sel = signal_solo_global_capped(data, fifo, head, sign, band, cap)
                m = replay_for(data, fifo, sel)
                cap_label = "no_cap" if cap is None else str(cap)
                rule = f"K1:{head}_sign{int(sign):+d}_band{band:.2f}_cap{cap_label}"
                row = {
                    "rule": rule,
                    "head": head, "sign": int(sign), "band": band,
                    "cap_per_day": (cap if cap is not None else -1),
                    **m,
                }
                row["stat_sufficient"] = (row["n_filled"] >= 30)
                rows.append(row)

    # Inflate per-day columns
    for r in rows:
        for di, d in enumerate(dates):
            r[f"filled_{d}"] = r["per_day_filled"][di] if di < len(r["per_day_filled"]) else 0
            r[f"signals_{d}"] = r["per_day_signals"][di] if di < len(r["per_day_signals"]) else 0
            r[f"pnl_ticks_{d}"] = r["per_day_pnl_ticks"][di] if di < len(r["per_day_pnl_ticks"]) else 0.0
            r[f"wins_{d}"] = r["per_day_wins"][di] if di < len(r["per_day_wins"]) else 0
        r.pop("per_day_filled", None)
        r.pop("per_day_signals", None)
        r.pop("per_day_pnl_ticks", None)
        r.pop("per_day_wins", None)

    csv_path = OUT_DIR / "results.csv"
    field_order = [
        "rule", "head", "sign", "band", "cap_per_day",
        "n_signals", "n_filled", "fill_rate",
        "mean_ticks_per_fill", "mean_ticks_per_trade",
        "wr_pct", "profit_factor",
        "sharpe_annualized", "sortino_annualized", "max_drawdown_ticks",
        "adverse_selection_30s_avg_ticks",
        "toxic_5s_pct", "toxic_at_hold_pct",
        "n_distinct_days", "max_pct_single_day",
        "hc344_gate_pass", "stat_sufficient",
        "exit_horizon_used", "pnl_ticks_total", "avg_queue_pos_on_arrival",
    ]
    for d in dates:
        field_order += [f"filled_{d}", f"signals_{d}", f"pnl_ticks_{d}", f"wins_{d}"]

    def _key(r):
        s = r.get("sharpe_annualized", float("nan"))
        return (-s if np.isfinite(s) else 1e9)
    rows_sorted = sorted(rows, key=_key)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"[write] {csv_path}  ({len(rows)} K=1 capped rules)", flush=True)
    return rows_sorted


def print_part1_report(rows_sorted, dates):
    print(f"\n{'='*120}")
    print("PART 1 TOP-5 K=1 CAPPED CONFIGS by realistic annualized Sharpe (LONG, passive_at_touch, 30s)")
    print(f"{'='*120}")
    finite = [r for r in rows_sorted if np.isfinite(r.get("sharpe_annualized", float("nan")))]
    for r in finite[:5]:
        gate = "PASS" if r["hc344_gate_pass"] else "FAIL"
        stat = "OK" if r["stat_sufficient"] else "low_n"
        per_day_str = " ".join(f"{r.get(f'filled_{d}',0):>4d}" for d in dates)
        print(f"  {r['rule']:<58}  n_fill={r['n_filled']:>4}  "
              f"t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
              f"PF={r['profit_factor']:6.2f}  Sharpe={r['sharpe_annualized']:+8.1f}  "
              f"max_day={100*r['max_pct_single_day']:5.1f}% HC344={gate} n>=30={stat}  "
              f"day_fills=[{per_day_str}]  toxic5={r['toxic_5s_pct']:4.1f}%")

    print(f"\n{'='*120}")
    print("PART 1 TRIPLE-GATE SURVIVORS: Sharpe>0 AND HC#344(max_day<=20%) AND n_filled>=30")
    print(f"{'='*120}")
    survivors = [r for r in finite
                 if r.get("sharpe_annualized", -1e9) > 0
                 and r.get("hc344_gate_pass", False)
                 and r.get("n_filled", 0) >= 30]
    if not survivors:
        print("  NONE.")
        near = [r for r in finite
                if r.get("sharpe_annualized", -1e9) > 0
                and r.get("hc344_gate_pass", False)
                and r.get("n_filled", 0) >= 15]
        print(f"\n  Best near-misses (relaxing n>=30 to n>=15)  [count={len(near)}]:")
        for r in near[:5]:
            per_day_str = " ".join(f"{r.get(f'filled_{d}',0):>4d}" for d in dates)
            print(f"  NEAR: {r['rule']:<58}  n_fill={r['n_filled']:>4}  "
                  f"t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
                  f"Sharpe={r['sharpe_annualized']:+8.1f}  "
                  f"max_day={100*r['max_pct_single_day']:5.1f}%  "
                  f"day_fills=[{per_day_str}]")

        # Show best near-miss with full per-day breakdown
        if near:
            top = near[0]
            print(f"\n  Best near-miss FULL per-day breakdown: {top['rule']}")
            for d in dates:
                f_n = top.get(f'filled_{d}', 0)
                f_p = top.get(f'pnl_ticks_{d}', 0.0)
                f_w = top.get(f'wins_{d}', 0)
                tpf = (f_p / f_n) if f_n > 0 else float('nan')
                wr_d = (f_w / f_n * 100.0) if f_n > 0 else float('nan')
                print(f"    {d}: n_filled={f_n:>4}  t/fill={tpf:+7.2f}  WR={wr_d:5.1f}%  "
                      f"pnl_total_ticks={f_p:+8.2f}")
    else:
        for r in survivors[:10]:
            per_day_str = " ".join(f"{r.get(f'filled_{d}',0):>4d}" for d in dates)
            print(f"  PASS: {r['rule']:<58}  n_fill={r['n_filled']:>4}  "
                  f"t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
                  f"PF={r['profit_factor']:6.2f}  Sharpe={r['sharpe_annualized']:+8.1f}  "
                  f"max_day={100*r['max_pct_single_day']:5.1f}%  "
                  f"day_fills=[{per_day_str}]")
        top = survivors[0]
        print(f"\n  TOP SURVIVOR FULL per-day breakdown: {top['rule']}")
        for d in dates:
            f_n = top.get(f'filled_{d}', 0)
            f_p = top.get(f'pnl_ticks_{d}', 0.0)
            f_w = top.get(f'wins_{d}', 0)
            tpf = (f_p / f_n) if f_n > 0 else float('nan')
            wr_d = (f_w / f_n * 100.0) if f_n > 0 else float('nan')
            print(f"    {d}: n_filled={f_n:>4}  t/fill={tpf:+7.2f}  WR={wr_d:5.1f}%  "
                  f"pnl_total_ticks={f_p:+8.2f}")
    return survivors


# ---------------------------------------------------------------------------
# PART 2: K=2 edge stability
# ---------------------------------------------------------------------------
def part2_k2_edge_stability(data, fifo):
    print(f"\n{'='*120}")
    print(f"PART 2: K=2 STACK EDGE STABILITY across OOT days")
    print(f"  heads = {K2_HEADS}  band = {K2_BAND}  LONG / passive_at_touch / 30s hold")
    print(f"{'='*120}")
    sel = signal_k2_global(data, fifo, K2_BAND)
    m = replay_for(data, fifo, sel, return_per_trade=True)
    dates = data["oot_dates"]
    raw_pnl = m.pop("_raw_pnl_ticks", None)
    date_idx_filled = m.pop("_date_idx_filled", None)

    per_day_stats = {}
    print(f"  Overall: n_signals={m['n_signals']} n_filled={m['n_filled']} "
          f"t/fill={m['mean_ticks_per_fill']:+.2f} WR={m['wr_pct']:.1f}% "
          f"Sharpe={m['sharpe_annualized']:+.1f}")
    print(f"  Per-day breakdown (FILLED trades):")
    n_firing_days = 0
    for di, d in enumerate(dates):
        f_n = m["per_day_filled"][di]
        f_s = m["per_day_signals"][di]
        if raw_pnl is not None and date_idx_filled is not None and f_n > 0:
            mask = (date_idx_filled == di)
            day_pnl = raw_pnl[mask]
            t_fill = float(day_pnl.mean())
            wr_d = float((day_pnl > 0).mean() * 100.0)
            pnl_sum = float(day_pnl.sum())
        else:
            t_fill = float("nan"); wr_d = float("nan"); pnl_sum = 0.0
        if f_n > 0:
            n_firing_days += 1
        per_day_stats[d] = {
            "n_signals": int(f_s),
            "n_filled": int(f_n),
            "mean_ticks_per_fill": t_fill,
            "wr_pct": wr_d,
            "pnl_ticks_total": pnl_sum,
        }
        print(f"    {d}: n_signals={f_s:>5}  n_filled={f_n:>3}  "
              f"t/fill={t_fill:+7.2f}  WR={wr_d:5.1f}%  pnl_total_ticks={pnl_sum:+8.2f}")

    # Single-event artifact check
    firing_days_with_fills = [d for d, s in per_day_stats.items() if s["n_filled"] > 0]
    total_pnl = sum(s["pnl_ticks_total"] for s in per_day_stats.values())
    max_day_pnl_pct = 0.0
    dominant_day = None
    if total_pnl != 0:
        for d, s in per_day_stats.items():
            pct = abs(s["pnl_ticks_total"] / total_pnl) * 100.0 if total_pnl != 0 else 0.0
            if pct > max_day_pnl_pct:
                max_day_pnl_pct = pct
                dominant_day = d

    # Edge stability verdict
    if n_firing_days <= 1:
        verdict = "SINGLE_DAY_ARTIFACT"
    elif max_day_pnl_pct > 80.0:
        verdict = f"DOMINATED_BY_{dominant_day}_({max_day_pnl_pct:.0f}pct)"
    elif max_day_pnl_pct > 60.0:
        verdict = f"HEAVY_SKEW_TO_{dominant_day}_({max_day_pnl_pct:.0f}pct)"
    else:
        verdict = "STABLE_ACROSS_FIRING_DAYS"

    print(f"\n  Firing days (>=1 fill): {firing_days_with_fills}  ({n_firing_days}/{len(dates)})")
    print(f"  Max single-day pnl share: {max_day_pnl_pct:.1f}% ({dominant_day})")
    print(f"  VERDICT: {verdict}")

    stability = {
        "rule": f"K2_global_band{K2_BAND}_long_passive_at_touch_30s",
        "heads": [{"head": h, "sign": s} for h, s in K2_HEADS],
        "band": K2_BAND,
        "n_signals_total": int(m["n_signals"]),
        "n_filled_total": int(m["n_filled"]),
        "mean_ticks_per_fill_total": float(m["mean_ticks_per_fill"]) if np.isfinite(m["mean_ticks_per_fill"]) else None,
        "wr_pct_total": float(m["wr_pct"]) if np.isfinite(m["wr_pct"]) else None,
        "sharpe_annualized_total": float(m["sharpe_annualized"]) if np.isfinite(m["sharpe_annualized"]) else None,
        "n_firing_days": int(n_firing_days),
        "firing_days": firing_days_with_fills,
        "max_day_pnl_share_pct": float(max_day_pnl_pct),
        "dominant_day": dominant_day,
        "verdict": verdict,
        "per_day": per_day_stats,
    }
    out_path = OUT_DIR / "k2_edge_stability.json"
    with out_path.open("w") as fh:
        json.dump(stability, fh, indent=2, default=str)
    print(f"\n[write] {out_path}", flush=True)
    return stability


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[v33_k1_capped_search] === K=1 CAPPED SWEEP + K=2 EDGE STABILITY ===", flush=True)
    print(f"[cfg] side={SIDE}  hold={HOLD_SECONDS}s  order={ORDER_TYPE}  cancel_eval_window={CANCEL_WINDOW}", flush=True)
    print(f"[cfg] commission={ES_RT_COMMISSION_TICKS_DEFAULT}t  spread={ES_SPREAD_TICKS_RTH_DEFAULT}t", flush=True)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = _load_all()
    print(f"[load] n_total={data['n']:,}  fifo_mask={int(data['fifo_mask'].sum()):,}  "
          f"oot_dates={data['oot_dates']}", flush=True)
    fifo = _load_fifo_labels(FIFO_LABELS_DIR, data["oot_dates"])
    print(f"[load] fifo labels: n_per_day={fifo['_n_per_day']}", flush=True)

    rows_sorted = part1_k1_capped_sweep(data, fifo)
    survivors = print_part1_report(rows_sorted, data["oot_dates"])
    stability = part2_k2_edge_stability(data, fifo)

    print(f"\n{'='*120}")
    print(f"FINAL SUMMARY")
    print(f"{'='*120}")
    print(f"  PART 1: {len(rows_sorted)} K=1 capped configs tested. Triple-gate survivors: {len(survivors)}")
    print(f"  PART 2: K=2 stack edge verdict: {stability['verdict']}  "
          f"(firing on {stability['n_firing_days']}/5 days, "
          f"max single-day pnl share = {stability['max_day_pnl_share_pct']:.1f}%)")

    print(f"\n[done] elapsed: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
