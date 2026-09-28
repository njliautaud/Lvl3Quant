#!/usr/bin/env python3
"""
v3.3 REALISTIC RE-VALIDATION — Batch Full-Market-Replay (HC #357)

Discovery (2026-05-15, K=2 full market replay): the headline "Sharpe 7.59"
stacked-confluence config dies under realistic queue+adverse-selection replay
(35% fill rate, 7.95% WR, PF 0.057, mean -8.94 ticks/trade). This MAY invalidate
every other v3.3 execution config we found this morning.

THIS SCRIPT re-runs the realistic full_market_replay engine on every prior
"winner" config and writes a single ranking CSV.

CONFIGS RE-VALIDATED:
  Group A: 5 top SOLO single-head SHORT configs from solo_vol_full_oot_results.json
           (top-0.1% band — n in range 60-75)
  Group B: stacked confluence K∈{1,2,3} × band∈{0.05,0.10,0.20} = 9 configs
           (some collapse to identical universes; ~6 unique selections)

ENGINE:
  - Reuses scripts.v3_3_research.full_market_replay primitives:
       _load_fifo_labels, _queue_position_model, _adverse_selection,
       _annualized, _profit_factor, _max_drawdown_ticks, _entry_price_edge_ticks,
       _mfe_mae_per_fill, _pick_exit_horizon
  - Order type: passive_at_touch (40-eval cancel window) — same as K=2 replay.
  - Hold: 30s.
  - Short side only (matches solo_vol_full_oot semantics).

OUTPUT:
  output/v3_3_full_execution_analysis_20260514/realistic_revalidation/
    all_configs_summary.csv   <-- the single ranking artifact
    per_config/<label>_summary.json  (per-config full detail)

NOT MALWARE. Pure read-only analysis. Reuses canonical library without modifying it.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
    _mfe_mae_per_fill,
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    PRICE_UNIT_TO_TICKS,
)

PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
FIFO_LABELS_DIR = ROOT / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/realistic_revalidation"
PER_CFG_DIR = OUT_DIR / "per_config"

# ---------------------------------------------------------------------------
# Config catalog
# ---------------------------------------------------------------------------
# Group A: Top-5 solo SHORT configs from solo_vol_full_oot top25.
# Tuple: (label, head_name_in_npz, sign, band_frac)
# sign×pred → "more short-like": top band_frac on (sign×pred) = SHORT signal.
SOLO_CONFIGS = [
    ("solo_log_ret_60s_pos_0p001",       "pred_log_ret_60s",         +1, 0.001),
    ("solo_log_ret_5min_neg_0p001",      "pred_log_ret_5min",        -1, 0.001),
    ("solo_p_reversal_60s_neg_0p001",    "pred_p_reversal_60s",      -1, 0.001),
    ("solo_fifo_tp8sl5_net_neg_0p001",   "pred_fifo_tp8sl5_net",     -1, 0.001),
    ("solo_log_ret_5min_pos_0p001",      "pred_log_ret_5min",        +1, 0.001),
]

# Group B: Stacked confluence — the head_signed list from /tmp/v33_stacked_confluence.py
# (head_in_npz, sign) where sign×pred → more SHORT-like.
STACKED_HEADS_SIGNED = [
    ("pred_log_ret_60s",         +1),
    ("pred_log_ret_5min",        -1),
    ("pred_p_reversal_60s",      -1),
    ("pred_fifo_tp8sl5_net",     -1),
    ("pred_pred_mae_60s_ticks",  -1),
]
STACKED_K = [1, 2, 3]
STACKED_BANDS = [0.05, 0.10, 0.20]


# ---------------------------------------------------------------------------
# Predictions / common arrays loader (load once, reuse across configs)
# ---------------------------------------------------------------------------
def _load_all() -> dict:
    print(f"[load] {PRED_NPZ}", flush=True)
    d = np.load(PRED_NPZ, allow_pickle=True)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    # All prediction heads we'll need
    needed_heads = {h for (h, _) in STACKED_HEADS_SIGNED}
    for _, h, _, _ in SOLO_CONFIGS:
        needed_heads.add(h)
    preds = {}
    for h in needed_heads:
        if h not in d.files:
            raise KeyError(f"Head missing from NPZ: {h}")
        preds[h] = d[h].astype(np.float64)

    # Realized-target horizons for adverse-selection / exit pnl
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
        "preds": preds,
        "fifo_mask": fifo_mask,
        "tgt_lr": tgt_lr,
        "tgt_lr_mask": tgt_lr_mask,
        "oot_dates": oot_dates,
        "target_fifo_net": target_fifo_net,
        "n": n_total,
    }


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------
def signal_solo(data: dict, head: str, sign: int, band_frac: float):
    """Top band_frac of (sign × pred) within fifo-fillable universe."""
    p = data["preds"][head]
    valid = data["fifo_mask"] & np.isfinite(p)
    s = sign * p
    s_valid = s[valid]
    thr = float(np.quantile(s_valid, 1.0 - band_frac))
    sel = valid & (s >= thr)
    return np.where(sel)[0], {"thr_signed_pred": thr, "band_frac": band_frac,
                              "head": head, "sign": sign,
                              "n_valid_universe": int(valid.sum())}


def signal_stacked(data: dict, K: int, band_frac: float):
    """All K heads must be in top band_frac of own (sign×pred) within fifo-fillable universe."""
    chosen = STACKED_HEADS_SIGNED[:K]
    fifo_mask = data["fifo_mask"]
    # Build the validity mask = fifo & all preds finite
    valid = fifo_mask.copy()
    for h, _ in chosen:
        valid &= np.isfinite(data["preds"][h])
    sel = valid.copy()
    thrs = {}
    for h, sign in chosen:
        s = sign * data["preds"][h]
        thr = float(np.quantile(s[valid], 1.0 - band_frac))
        sel &= (s >= thr)
        thrs[h] = {"sign": sign, "thr_signed_pred": thr}
    return np.where(sel)[0], {"K": K, "band_frac": band_frac,
                              "thresholds": thrs,
                              "n_valid_universe": int(valid.sum())}


# ---------------------------------------------------------------------------
# Realistic replay for a given selection (always SHORT side)
# ---------------------------------------------------------------------------
def replay_for_selection(
    data: dict,
    fifo: dict,
    sel_idx_full: np.ndarray,
    *,
    order_type: str = "passive_at_touch",
    cancel_eval_window: int = 40,
    hold_seconds: float = 30.0,
    rt_commission_ticks: float = ES_RT_COMMISSION_TICKS_DEFAULT,
    spread_ticks_rth: float = ES_SPREAD_TICKS_RTH_DEFAULT,
) -> dict:
    """Run queue+adverse replay for a SHORT-side selection. Returns metrics dict."""
    n_total = data["n"]
    oot_dates = data["oot_dates"]
    n_fifo = int(sum(fifo["_n_per_day"]))
    n = min(n_total, n_fifo)
    sel = sel_idx_full[sel_idx_full < n]
    n_attempted = int(sel.size)

    if n_attempted == 0:
        return {
            "n_signals": 0, "n_filled": 0, "fill_rate": 0.0,
            "mean_ticks_per_fill": float("nan"),
            "mean_ticks_per_trade": float("nan"),
            "wr_pct": float("nan"), "profit_factor": float("nan"),
            "sharpe_annualized": float("nan"),
            "sortino_annualized": float("nan"),
            "max_drawdown_ticks": 0.0,
            "adverse_selection_30s_avg_ticks": 0.0,
            "toxic_5s_pct": 0.0,
            "n_distinct_days": 0,
            "max_pct_single_day": 0.0,
            "hc344_gate_pass": False,
            "naive_fifo_mean_ticks_net": float("nan"),
            "naive_fifo_per_trade_sharpe": float("nan"),
        }

    side = "short"
    side_sign = -1.0

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
    fill_rate = n_filled / max(1, n_attempted)

    horizon_choice = _pick_exit_horizon(hold_seconds)
    lr_exit = data["tgt_lr"][horizon_choice][filled_global_idx]
    lr_mask_exit = data["tgt_lr_mask"][horizon_choice][filled_global_idx]
    edge_offset = _entry_price_edge_ticks(order_type, spread_ticks_rth)
    raw_pnl_ticks = side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - rt_commission_ticks
    raw_pnl_ticks = np.where(lr_mask_exit, raw_pnl_ticks, 0.0)

    pnl_total = float(raw_pnl_ticks.sum())
    mean_t_fill = pnl_total / max(1, n_filled) if n_filled else float("nan")
    mean_t_trade = pnl_total / max(1, n_attempted)
    sharpe = _annualized(raw_pnl_ticks, downside=False) if n_filled else float("nan")
    sortino = _annualized(raw_pnl_ticks, downside=True) if n_filled else float("nan")
    pf = _profit_factor(raw_pnl_ticks) if n_filled else float("nan")
    wr = float((raw_pnl_ticks > 0).mean() * 100.0) if n_filled else float("nan")
    mdd = _max_drawdown_ticks(raw_pnl_ticks) if n_filled else 0.0

    adv30 = _adverse_selection(filled_global_idx, side_sign,
                               data["tgt_lr"]["30s"], data["tgt_lr_mask"]["30s"])
    adv30_avg = float(np.nanmean(adv30)) if adv30.size and np.isfinite(np.nanmean(adv30)) else 0.0

    adv5 = _adverse_selection(filled_global_idx, side_sign,
                              data["tgt_lr"]["5s"], data["tgt_lr_mask"]["5s"])
    toxic_pct = 0.0
    if adv5.size and n_filled:
        toxic_mask = (adv5 <= -1.0) & np.isfinite(adv5)
        toxic_pct = float(toxic_mask.sum()) / n_filled * 100.0

    # Day concentration on FILLED trades
    n_per_day_filled = np.zeros(len(oot_dates), dtype=int)
    for di in range(len(oot_dates)):
        n_per_day_filled[di] = int(((date_idx_signal == di) & filled_mask).sum())
    n_distinct_days = int((n_per_day_filled > 0).sum())
    max_pct_single = float(n_per_day_filled.max() / n_filled) if n_filled else 0.0
    hc344_pass = (max_pct_single <= 0.20) and (n_filled > 0)

    # Naive FIFO-baseline (no queue model): same selection, ticks net via
    # short_pnl = -target - commission (matches solo_vol_full_oot convention)
    tgt = data["target_fifo_net"][sel]
    naive_short_t = -tgt - rt_commission_ticks
    naive_mean = float(naive_short_t.mean())
    naive_std = float(naive_short_t.std(ddof=1)) if naive_short_t.size > 1 else 0.0
    naive_pertrade_sharpe = naive_mean / naive_std if naive_std > 1e-12 else float("nan")

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
        "toxic_5s_pct": toxic_pct,
        "n_distinct_days": n_distinct_days,
        "max_pct_single_day": max_pct_single,
        "hc344_gate_pass": bool(hc344_pass),
        "naive_fifo_mean_ticks_net": naive_mean,
        "naive_fifo_per_trade_sharpe": naive_pertrade_sharpe,
        "avg_queue_pos_on_arrival": avg_q_pos,
        "exit_horizon_used": horizon_choice,
        "pnl_ticks_total": pnl_total,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_all() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CFG_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = _load_all()
    print(f"[load] n_total={data['n']:,}  fifo_fillable={int(data['fifo_mask'].sum()):,}  "
          f"oot_dates={data['oot_dates']}", flush=True)

    fifo = _load_fifo_labels(FIFO_LABELS_DIR, data["oot_dates"])
    n_fifo = sum(fifo["_n_per_day"])
    print(f"[load] fifo labels: n={n_fifo:,}  n_per_day={fifo['_n_per_day']}", flush=True)

    rows: list[dict] = []
    rank = 0

    # Group A: Solo
    print(f"\n[grp A] === SOLO TOP-5 ===", flush=True)
    for (label, head, sign, band) in SOLO_CONFIGS:
        rank += 1
        t = time.time()
        sel_idx, sig_info = signal_solo(data, head, sign, band)
        metrics = replay_for_selection(data, fifo, sel_idx)
        elapsed = time.time() - t
        row = {
            "rank_order": rank,
            "group": "solo",
            "label": label,
            "head_or_K": head,
            "sign_or_K": sign,
            "band_frac": band,
            **metrics,
            "elapsed_sec": elapsed,
        }
        rows.append(row)
        _print_row(row)
        (PER_CFG_DIR / f"{label}_summary.json").write_text(
            json.dumps({"signal": sig_info, "metrics": metrics}, indent=2, default=float)
        )

    # Group B: Stacked
    print(f"\n[grp B] === STACKED K×band ===", flush=True)
    seen_sel_sig: dict[bytes, str] = {}  # dedupe identical selections
    for K in STACKED_K:
        for band in STACKED_BANDS:
            rank += 1
            label = f"stacked_K{K}_band{band:.2f}"
            t = time.time()
            sel_idx, sig_info = signal_stacked(data, K, band)
            sel_sig = sel_idx.tobytes()
            dup_of = seen_sel_sig.get(sel_sig)
            if dup_of is None:
                seen_sel_sig[sel_sig] = label
            metrics = replay_for_selection(data, fifo, sel_idx)
            elapsed = time.time() - t
            row = {
                "rank_order": rank,
                "group": "stacked",
                "label": label,
                "head_or_K": f"K={K}",
                "sign_or_K": K,
                "band_frac": band,
                **metrics,
                "duplicate_of": dup_of or "",
                "elapsed_sec": elapsed,
            }
            rows.append(row)
            _print_row(row)
            (PER_CFG_DIR / f"{label}_summary.json").write_text(
                json.dumps({"signal": sig_info, "metrics": metrics}, indent=2, default=float)
            )

    # Rank and write summary CSV
    csv_path = OUT_DIR / "all_configs_summary.csv"
    field_order = [
        "label", "group", "head_or_K", "sign_or_K", "band_frac",
        "n_signals", "n_filled", "fill_rate",
        "mean_ticks_per_fill", "mean_ticks_per_trade",
        "wr_pct", "profit_factor",
        "sharpe_annualized", "sortino_annualized", "max_drawdown_ticks",
        "adverse_selection_30s_avg_ticks", "toxic_5s_pct",
        "n_distinct_days", "max_pct_single_day", "hc344_gate_pass",
        "naive_fifo_mean_ticks_net", "naive_fifo_per_trade_sharpe",
        "exit_horizon_used", "pnl_ticks_total", "elapsed_sec",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (
            -(x["sharpe_annualized"] if np.isfinite(x.get("sharpe_annualized", np.nan)) else -1e9),
        )):
            w.writerow(r)
    print(f"\n[write] {csv_path}", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n[done] total elapsed: {elapsed_total:.1f}s   rows={len(rows)}", flush=True)

    # ── FINAL VERDICT ──────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("VERDICT — configs that pass ALL gates:")
    print("  (1) realistic annualized Sharpe > 0  AND")
    print("  (2) HC #344 day-conc gate (≤20% single day)  AND")
    print("  (3) ≥30 trades (filled) over 5-day OOT")
    print(f"{'='*78}")
    survivors = [
        r for r in rows
        if np.isfinite(r.get("sharpe_annualized", np.nan))
        and r["sharpe_annualized"] > 0
        and r["hc344_gate_pass"]
        and r["n_filled"] >= 30
    ]
    if not survivors:
        print("  NONE. All prior 'winners' fail realistic replay.")
    else:
        for r in sorted(survivors, key=lambda x: -x["sharpe_annualized"]):
            print(f"  PASS: {r['label']:<40}  Sharpe(ann)={r['sharpe_annualized']:+.2f}  "
                  f"n_filled={r['n_filled']}  WR={r['wr_pct']:.1f}%  "
                  f"PF={r['profit_factor']:.2f}  max_day={100*r['max_pct_single_day']:.1f}%")
    print(f"{'='*78}")

    return rows


def _print_row(r: dict) -> None:
    print(f"  [{r['label']:<40}]  n_sig={r['n_signals']:>5}  n_fill={r['n_filled']:>5}  "
          f"fill_rt={r['fill_rate']:.3f}  mean_t/fill={r['mean_ticks_per_fill']:+.3f}  "
          f"mean_t/trade={r['mean_ticks_per_trade']:+.3f}  WR={r['wr_pct']:.1f}%  "
          f"PF={r['profit_factor']:.2f}  Sharpe={r['sharpe_annualized']:+.2f}  "
          f"max_day={100*r['max_pct_single_day']:.1f}%  HC344={'PASS' if r['hc344_gate_pass'] else 'FAIL'}",
          flush=True)


def main() -> int:
    print(f"[v33_revalidate] === REALISTIC RE-VALIDATION (HC #357 BATCH) ===", flush=True)
    print(f"[v33_revalidate] order_type=passive_at_touch  cancel_window=40  hold=30s", flush=True)
    print(f"[v33_revalidate] commission={ES_RT_COMMISSION_TICKS_DEFAULT}t  spread={ES_SPREAD_TICKS_RTH_DEFAULT}t",
          flush=True)
    run_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
