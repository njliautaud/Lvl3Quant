#!/usr/bin/env python3
"""
v3.3 DAY-BALANCED LONG-ONLY SELECTION SEARCH
=============================================

CONTEXT (2026-05-15):
  v33_side_horizon_sweep proved LONG @ K=2 stack, 30s hold, passive_at_touch is
  +22.08 ticks/fill, 100% WR — but ALL 14 prior configs FAIL HC #344 (60–93%
  concentration on a single OOT day). Edge is real; day-balance is not.

THIS SCRIPT — pure analysis, day-balanced selection rule search:

  Step 1: Per-day signal breakdown for the 14 base configs (which day dominates).
  Step 2: Build & test 4 families of day-balanced rules, LONG-only, HC #357
          full replay (passive_at_touch, 30s hold; also a couple of variants):
    A. Per-day quantile selection (top X% WITHIN each day) for solo + stacked
       components. X ∈ {1, 5, 10, 20}%.
    B. Per-day top-X% on the K=2 stack (intersection within day).
    C. Floor-by-day: global stack rule capped at N per day.
    D. ToD bucket: per-bucket top X% (4 buckets per day from ts_ns quantiles).
  Step 3: Output CSV + console top-10 + flag full triple-gate passers.

OUTPUT:
  output/v3_3_full_execution_analysis_20260514/day_balanced_search/results.csv

CONFORMS TO:
  HC #0 (sliding/forward only), HC #344 (≤20% single day), HC #357 (canonical
  replay engine), HC #336 (queue + adverse selection always on).

Reuses canonical full_market_replay primitives without modifying them.
Read-only on data dirs.
"""
from __future__ import annotations

import csv
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
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/day_balanced_search"

# ---------------------------------------------------------------------------
# Base catalog (same 14 configs as side_horizon_sweep)
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

# Per-day search grid
PER_DAY_X = [0.01, 0.05, 0.10, 0.20]
BUCKET_X = [0.05, 0.10, 0.20]
N_BUCKETS = 4
CAPS = [5, 10, 20, 30]

# Default exec config (LONG only)
SIDE = "long"
HOLD_SECONDS = 30.0
ORDER_TYPE = "passive_at_touch"
CANCEL_WINDOW = 40


# ---------------------------------------------------------------------------
# Loaders / signal-builders (mirror v33_side_horizon_sweep)
# ---------------------------------------------------------------------------
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
    n_total = next(iter(preds.values())).shape[0]
    return {
        "preds": preds,
        "fifo_mask": fifo_mask,
        "tgt_lr": tgt_lr,
        "tgt_lr_mask": tgt_lr_mask,
        "oot_dates": oot_dates,
        "n": n_total,
    }


def signal_solo_global(data, head, sign, band_frac):
    p = data["preds"][head]
    valid = data["fifo_mask"] & np.isfinite(p)
    s = sign * p
    s_valid = s[valid]
    thr = float(np.quantile(s_valid, 1.0 - band_frac))
    sel = valid & (s >= thr)
    return np.where(sel)[0]


def signal_stacked_global(data, K, band_frac):
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


# ---------------------------------------------------------------------------
# Day-balanced selection builders
# ---------------------------------------------------------------------------
def _date_idx_global(data, fifo) -> np.ndarray:
    """Return a global date_idx array aligned to n_total = min(preds, fifo)."""
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    return fifo["_date_idx"][:n]


def signal_solo_per_day(data, fifo, head, sign, X):
    """Top-X within each day (global → masked by daily quantile)."""
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    date_idx = fifo["_date_idx"][:n]
    p = data["preds"][head][:n]
    valid = data["fifo_mask"][:n] & np.isfinite(p)
    s = sign * p
    sel = np.zeros(n, dtype=bool)
    for di in range(len(data["oot_dates"])):
        day_mask = (date_idx == di) & valid
        if not day_mask.any():
            continue
        s_day = s[day_mask]
        if s_day.size == 0:
            continue
        thr = float(np.quantile(s_day, 1.0 - X))
        sel |= day_mask & (s >= thr)
    return np.where(sel)[0]


def signal_stacked_per_day(data, fifo, K, X):
    """For each head in the stack, within each day require top-X%."""
    chosen = STACKED_HEADS_SIGNED[:K]
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    date_idx = fifo["_date_idx"][:n]
    valid = data["fifo_mask"][:n].copy()
    for h, _ in chosen:
        valid &= np.isfinite(data["preds"][h][:n])
    sel = valid.copy()
    for h, sign in chosen:
        s = sign * data["preds"][h][:n]
        head_sel = np.zeros(n, dtype=bool)
        for di in range(len(data["oot_dates"])):
            day_mask = (date_idx == di) & valid
            if not day_mask.any():
                continue
            s_day = s[day_mask]
            thr = float(np.quantile(s_day, 1.0 - X))
            head_sel |= day_mask & (s >= thr)
        sel &= head_sel
    return np.where(sel)[0]


def signal_stacked_global_capped(data, fifo, K, band, cap_per_day):
    """Global stack rule then cap top-N by combined score within each day."""
    chosen = STACKED_HEADS_SIGNED[:K]
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    date_idx = fifo["_date_idx"][:n]
    valid = data["fifo_mask"][:n].copy()
    for h, _ in chosen:
        valid &= np.isfinite(data["preds"][h][:n])
    sel = valid.copy()
    score = np.zeros(n, dtype=np.float64)
    for h, sign in chosen:
        s = sign * data["preds"][h][:n]
        thr = float(np.quantile(s[valid], 1.0 - band))
        sel &= (s >= thr)
        # standardized contribution to score for tie-breaking
        mu = float(s[valid].mean())
        sd = float(s[valid].std() + 1e-12)
        score += (s - mu) / sd
    final = np.zeros(n, dtype=bool)
    for di in range(len(data["oot_dates"])):
        day_idx = np.where((date_idx == di) & sel)[0]
        if day_idx.size == 0:
            continue
        if day_idx.size <= cap_per_day:
            final[day_idx] = True
        else:
            # top-N by score within day
            top = day_idx[np.argsort(-score[day_idx])[:cap_per_day]]
            final[top] = True
    return np.where(final)[0]


def signal_stacked_per_bucket(data, fifo, K, X, n_buckets=N_BUCKETS):
    """Split each day into n_buckets by ts_ns quantiles, top-X% per bucket on
    the K-stack intersection (each head's per-bucket threshold)."""
    chosen = STACKED_HEADS_SIGNED[:K]
    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    date_idx = fifo["_date_idx"][:n]
    ts = fifo["ts_ns"][:n]
    valid = data["fifo_mask"][:n].copy()
    for h, _ in chosen:
        valid &= np.isfinite(data["preds"][h][:n])

    # Assign bucket id (0..n_buckets-1) PER DAY by ts quantiles
    bucket = np.full(n, -1, dtype=np.int32)
    for di in range(len(data["oot_dates"])):
        day_idx = np.where(date_idx == di)[0]
        if day_idx.size == 0:
            continue
        ts_day = ts[day_idx]
        # Use quantile cutpoints
        if day_idx.size < n_buckets:
            bucket[day_idx] = 0
            continue
        qs = np.quantile(ts_day, np.linspace(0, 1, n_buckets + 1))
        b = np.searchsorted(qs[1:-1], ts_day, side="right")
        bucket[day_idx] = b.astype(np.int32)

    sel = valid.copy()
    for h, sign in chosen:
        s = sign * data["preds"][h][:n]
        head_sel = np.zeros(n, dtype=bool)
        for di in range(len(data["oot_dates"])):
            for b in range(n_buckets):
                bk_mask = (date_idx == di) & (bucket == b) & valid
                if not bk_mask.any():
                    continue
                s_bk = s[bk_mask]
                thr = float(np.quantile(s_bk, 1.0 - X))
                head_sel |= bk_mask & (s >= thr)
        sel &= head_sel
    return np.where(sel)[0]


# ---------------------------------------------------------------------------
# Replay (LONG, passive_at_touch, 30s) — vendored from sweep, but
# parametrized for hold/entry.
# ---------------------------------------------------------------------------
def replay_for(
    data, fifo, sel_idx_full, *,
    side=SIDE, order_type=ORDER_TYPE,
    cancel_eval_window=CANCEL_WINDOW, hold_seconds=HOLD_SECONDS,
    rt_commission_ticks=ES_RT_COMMISSION_TICKS_DEFAULT,
    spread_ticks_rth=ES_SPREAD_TICKS_RTH_DEFAULT,
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

    per_day_filled = [int(((date_idx_signal == di) & filled_mask).sum()) for di in range(len(oot_dates))]
    n_distinct_days = sum(1 for x in per_day_filled if x > 0)
    max_pct_single = max(per_day_filled) / n_filled if n_filled > 0 else 0.0
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
        "per_day_filled": per_day_filled,
        "per_day_signals": per_day_signals,
    }


# ---------------------------------------------------------------------------
# Step 1: Diagnose concentration of the 14 base configs
# ---------------------------------------------------------------------------
def diagnose_base_concentration(data, fifo):
    print(f"\n{'='*100}")
    print("STEP 1: per-day SIGNAL counts for the 14 base configs (global selection)")
    print(f"{'='*100}")
    rows = []
    dates = data["oot_dates"]
    print(f"{'config':<36} " + "  ".join(f"{d:>10}" for d in dates) + "    total")
    print("-"*120)

    selections = []
    for (label, head, sign, band) in SOLO_CONFIGS:
        sel = signal_solo_global(data, head, sign, band)
        selections.append(("solo", label, sel))
    for K in STACKED_K:
        for band in STACKED_BANDS:
            sel = signal_stacked_global(data, K, band)
            label = f"stacked_K{K}_band{band:.2f}"
            selections.append(("stacked", label, sel))

    n = min(data["n"], int(sum(fifo["_n_per_day"])))
    date_idx = fifo["_date_idx"][:n]
    for grp, label, sel in selections:
        sel_in = sel[sel < n]
        per_day = [int(((date_idx == di) & (np.isin(np.arange(n), sel_in))).sum()) for di in range(len(dates))]
        # Faster: but vectorize via bincount
        per_day = np.bincount(date_idx[sel_in], minlength=len(dates)).tolist()
        total = int(sum(per_day))
        row = {"group": grp, "config_label": label, "total_signals": total}
        for di, d in enumerate(dates):
            row[f"signals_{d}"] = per_day[di]
            row[f"pct_{d}"] = (per_day[di] / total * 100.0) if total > 0 else 0.0
        rows.append(row)
        print(f"{label:<36} " + "  ".join(f"{x:>10d}" for x in per_day) + f"    {total:>5d}")

    return rows


# ---------------------------------------------------------------------------
# Step 2: Build & test day-balanced selection rules
# ---------------------------------------------------------------------------
def build_and_test_rules(data, fifo) -> list[dict]:
    rows: list[dict] = []
    dates = data["oot_dates"]
    n_dates = len(dates)

    # Family A: per-day quantile on solo heads
    print(f"\n[A] per-day top-X% solo heads…", flush=True)
    for (label, head, sign, _band) in SOLO_CONFIGS:
        for X in PER_DAY_X:
            sel = signal_solo_per_day(data, fifo, head, sign, X)
            m = replay_for(data, fifo, sel)
            rule = f"A_solo:{label}_perDayX{X:g}"
            row = {"family": "A_perDay_solo", "rule": rule,
                   "head": head, "sign": sign, "X": X, "K": None, "band": None,
                   "cap": None, "bucket": None, **m}
            rows.append(row)

    # Family B: per-day top-X% on K=2 stacked heads (also K=1 and K=3)
    print(f"[B] per-day top-X% stacked heads…", flush=True)
    for K in STACKED_K:
        for X in BUCKET_X:
            sel = signal_stacked_per_day(data, fifo, K, X)
            m = replay_for(data, fifo, sel)
            rule = f"B_stack:K{K}_perDayX{X:g}"
            row = {"family": "B_perDay_stack", "rule": rule,
                   "head": None, "sign": None, "X": X, "K": K, "band": None,
                   "cap": None, "bucket": None, **m}
            rows.append(row)

    # Family C: global stack rule with per-day cap
    print(f"[C] global stack with per-day cap…", flush=True)
    for K in STACKED_K:
        for band in STACKED_BANDS:
            for cap in CAPS:
                sel = signal_stacked_global_capped(data, fifo, K, band, cap)
                m = replay_for(data, fifo, sel)
                rule = f"C_capped:K{K}_band{band:.2f}_cap{cap}"
                row = {"family": "C_global_capped", "rule": rule,
                       "head": None, "sign": None, "X": None, "K": K, "band": band,
                       "cap": cap, "bucket": None, **m}
                rows.append(row)

    # Family D: ToD bucket top-X% per bucket on stacked heads
    print(f"[D] ToD-bucket top-X% per bucket stacked heads…", flush=True)
    for K in STACKED_K:
        for X in BUCKET_X:
            sel = signal_stacked_per_bucket(data, fifo, K, X, n_buckets=N_BUCKETS)
            m = replay_for(data, fifo, sel)
            rule = f"D_bucket:K{K}_b{N_BUCKETS}_X{X:g}"
            row = {"family": "D_bucket_stack", "rule": rule,
                   "head": None, "sign": None, "X": X, "K": K, "band": None,
                   "cap": None, "bucket": N_BUCKETS, **m}
            rows.append(row)

    # Inflate per-day columns for CSV
    for r in rows:
        for di, d in enumerate(dates):
            r[f"filled_{d}"] = r["per_day_filled"][di] if di < len(r["per_day_filled"]) else 0
            r[f"signals_{d}"] = r["per_day_signals"][di] if di < len(r["per_day_signals"]) else 0
        r.pop("per_day_filled", None)
        r.pop("per_day_signals", None)
    return rows


# ---------------------------------------------------------------------------
# CSV / reporting
# ---------------------------------------------------------------------------
def write_results(rows, dates, base_concentration_rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Base concentration diagnostic
    base_csv = OUT_DIR / "base_config_per_day_concentration.csv"
    fields = ["group", "config_label", "total_signals"]
    for d in dates:
        fields += [f"signals_{d}", f"pct_{d}"]
    with base_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in base_concentration_rows:
            w.writerow(r)
    print(f"[write] {base_csv}", flush=True)

    # Rule search results
    csv_path = OUT_DIR / "results.csv"
    field_order = [
        "family", "rule", "head", "sign", "X", "K", "band", "cap", "bucket",
        "n_signals", "n_filled", "fill_rate",
        "mean_ticks_per_fill", "mean_ticks_per_trade",
        "wr_pct", "profit_factor",
        "sharpe_annualized", "sortino_annualized", "max_drawdown_ticks",
        "adverse_selection_30s_avg_ticks",
        "toxic_5s_pct", "toxic_at_hold_pct",
        "n_distinct_days", "max_pct_single_day", "hc344_gate_pass",
        "exit_horizon_used", "pnl_ticks_total", "avg_queue_pos_on_arrival",
    ]
    for d in dates:
        field_order += [f"filled_{d}", f"signals_{d}"]
    # Sort by realistic Sharpe desc (NaN last)
    def _key(r):
        s = r.get("sharpe_annualized", float("nan"))
        return (-s if np.isfinite(s) else 1e9)
    rows_sorted = sorted(rows, key=_key)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"[write] {csv_path}  ({len(rows)} rules)", flush=True)
    return rows_sorted


def print_top10_and_survivors(rows_sorted, dates):
    print(f"\n{'='*100}")
    print("TOP-10 by realistic annualized Sharpe (LONG, passive_at_touch, 30s)")
    print(f"{'='*100}")
    finite = [r for r in rows_sorted if np.isfinite(r.get("sharpe_annualized", float("nan")))]
    for r in finite[:10]:
        gate = "PASS" if r["hc344_gate_pass"] else "FAIL"
        per_day_str = " ".join(f"{r.get(f'filled_{d}',0):>3d}" for d in dates)
        print(f"  {r['rule']:<40}  n_fill={r['n_filled']:>4}  "
              f"t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
              f"PF={r['profit_factor']:6.2f}  Sharpe={r['sharpe_annualized']:+8.1f}  "
              f"max_day={100*r['max_pct_single_day']:5.1f}% HC344={gate}  "
              f"day_fills=[{per_day_str}]")

    print(f"\n{'='*100}")
    print("TRIPLE-GATE PASSERS: Sharpe>0 AND HC#344(max_day<=20%) AND n_filled>=30")
    print(f"{'='*100}")
    survivors = [r for r in finite
                 if r.get("sharpe_annualized", -1e9) > 0
                 and r.get("hc344_gate_pass", False)
                 and r.get("n_filled", 0) >= 30]
    if not survivors:
        print("  NONE.")
        print("\n  Best near-misses (relaxing n_filled>=30 to >=15):")
        near = [r for r in finite
                if r.get("sharpe_annualized", -1e9) > 0
                and r.get("hc344_gate_pass", False)
                and r.get("n_filled", 0) >= 15]
        for r in near[:10]:
            per_day_str = " ".join(f"{r.get(f'filled_{d}',0):>3d}" for d in dates)
            print(f"  NEAR: {r['rule']:<40}  n_fill={r['n_filled']:>4}  "
                  f"t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
                  f"Sharpe={r['sharpe_annualized']:+8.1f}  "
                  f"max_day={100*r['max_pct_single_day']:5.1f}%  "
                  f"day_fills=[{per_day_str}]")
    else:
        for r in survivors[:15]:
            per_day_str = " ".join(f"{r.get(f'filled_{d}',0):>3d}" for d in dates)
            print(f"  PASS: {r['rule']:<40}  n_fill={r['n_filled']:>4}  "
                  f"t/fill={r['mean_ticks_per_fill']:+7.2f}  WR={r['wr_pct']:5.1f}%  "
                  f"PF={r['profit_factor']:6.2f}  Sharpe={r['sharpe_annualized']:+8.1f}  "
                  f"max_day={100*r['max_pct_single_day']:5.1f}%  "
                  f"day_fills=[{per_day_str}]")
        # Explicit rule summary for top survivor
        top = survivors[0]
        print(f"\n{'='*100}")
        print(f"TOP TRIPLE-GATE SURVIVOR — full entry criteria:")
        print(f"{'='*100}")
        print(f"  rule_id        : {top['rule']}")
        print(f"  family         : {top['family']}")
        print(f"  side           : LONG")
        print(f"  hold_seconds   : {HOLD_SECONDS}")
        print(f"  order_type     : {ORDER_TYPE}  (cancel_eval_window={CANCEL_WINDOW})")
        if top["family"] == "A_perDay_solo":
            print(f"  heads          : {top['head']}  sign={int(top['sign'])}")
            print(f"  threshold      : within each OOT day, take top {100*top['X']:.0f}% by sign*pred")
        elif top["family"] == "B_perDay_stack":
            chosen = STACKED_HEADS_SIGNED[:int(top['K'])]
            print(f"  K              : {int(top['K'])}")
            print(f"  heads (signed) : {chosen}")
            print(f"  threshold      : within each OOT day, intersect top {100*top['X']:.0f}% per head")
        elif top["family"] == "C_global_capped":
            chosen = STACKED_HEADS_SIGNED[:int(top['K'])]
            print(f"  K              : {int(top['K'])}")
            print(f"  heads (signed) : {chosen}")
            print(f"  global band    : top {100*top['band']:.0f}% per head, intersect")
            print(f"  per-day cap    : keep top {int(top['cap'])} by composite z-score per day")
        elif top["family"] == "D_bucket_stack":
            chosen = STACKED_HEADS_SIGNED[:int(top['K'])]
            print(f"  K              : {int(top['K'])}")
            print(f"  heads (signed) : {chosen}")
            print(f"  buckets        : {N_BUCKETS} ToD buckets per day (by ts_ns quantiles)")
            print(f"  threshold      : within each (day, bucket), intersect top {100*top['X']:.0f}% per head")
        print(f"  expected per-day distribution (filled): "
              + " ".join(f"{d}={top.get(f'filled_{d}',0)}" for d in dates))
        print(f"  metrics        : n_fill={top['n_filled']}  mean_ticks/fill={top['mean_ticks_per_fill']:+.2f}  "
              f"WR={top['wr_pct']:.1f}%  PF={top['profit_factor']:.2f}  "
              f"Sharpe(ann)={top['sharpe_annualized']:+.1f}  "
              f"max_day={100*top['max_pct_single_day']:.1f}%")
    print(f"{'='*100}\n")


def main():
    print(f"[v33_day_balanced_search] === DAY-BALANCED LONG SELECTION SEARCH ===", flush=True)
    print(f"[cfg] side={SIDE}  hold={HOLD_SECONDS}s  order={ORDER_TYPE}  cancel_eval_window={CANCEL_WINDOW}", flush=True)
    print(f"[cfg] commission={ES_RT_COMMISSION_TICKS_DEFAULT}t  spread={ES_SPREAD_TICKS_RTH_DEFAULT}t", flush=True)
    t0 = time.time()
    data = _load_all()
    print(f"[load] n_total={data['n']:,}  fifo_mask={int(data['fifo_mask'].sum()):,}  "
          f"oot_dates={data['oot_dates']}", flush=True)
    fifo = _load_fifo_labels(FIFO_LABELS_DIR, data["oot_dates"])
    print(f"[load] fifo labels: n_per_day={fifo['_n_per_day']}", flush=True)

    # Step 1
    base_conc = diagnose_base_concentration(data, fifo)

    # Step 2 + 3
    rules = build_and_test_rules(data, fifo)
    rules_sorted = write_results(rules, data["oot_dates"], base_conc)
    print_top10_and_survivors(rules_sorted, data["oot_dates"])

    print(f"[done] elapsed: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
