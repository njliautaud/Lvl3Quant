#!/usr/bin/env python3
"""
K=2 STACKED CONFLUENCE — FULL MARKET REPLAY (HC #357)

Validates whether the v3.3 K=2 SHORT confluence config's headline Sharpe 7.59
survives full market replay with order-queue position + adverse selection
modeling. NOT FIFO-floor — uses the canonical full_market_replay machinery.

RULE (per user 2026-05-15):
  Head A: pred_log_ret_60s  POSITIVE → in TOP 20% of its (fillable) distribution
  Head B: pred_log_ret_5min NEGATIVE → in BOTTOM 20% of its (fillable) distribution
  Both must hit simultaneously → SHORT signal (fade the up-spike + 5min downtrend)

INPUTS:
  - output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz
  - data/processed/mbo_events_smart_v3_fifo_labels/<DATE>_fifo_labels.npz  (5 OOT days)

OUTPUT:
  - output/v3_3_full_execution_analysis_20260514/k2_market_replay/
        k2_replay_summary.json
        k2_replay_per_trade.csv

This script REUSES the queue-position + adverse-selection logic from
scripts.v3_3_research.full_market_replay (HC #357 canonical lib). It does NOT
modify that module. It only builds the K=2 dual-head signal mask (which the
public TradeConfig API does not natively support — that API is single-head
percentile gate only) and then runs the same queue/adverse pipeline.

NOT MALWARE. Pure analysis. Read-only on data dirs; writes only to output dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

# Reuse canonical machinery
from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    _load_fifo_labels,
    _queue_position_model,
    _adverse_selection,
    _annualized,
    _profit_factor,
    _max_drawdown_ticks,
    _entry_price_edge_ticks,
    _pick_exit_horizon,
    _horizon_to_sec,
    _mfe_mae_per_fill,
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    ES_TICK_VALUE_DEFAULT,
    PRICE_UNIT_TO_TICKS,
)

PRED_NPZ = ROOT / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
FIFO_LABELS_DIR = ROOT / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/k2_market_replay"


def build_k2_short_signals(npz_path: Path, band_top_frac: float = 0.20):
    """Build the K=2 SHORT confluence mask exactly as defined in the user rule.

    Returns:
      sel_idx_global: indices into the 241,351 prediction stream where signal fires
      thresholds: dict with both thresholds for audit
      preds_dict: arrays needed downstream (predictions + masks for adverse selection)
    """
    d = np.load(npz_path, allow_pickle=True)
    p60 = d["pred_log_ret_60s"].astype(np.float64)
    p5m = d["pred_log_ret_5min"].astype(np.float64)

    # Use isfinite + fifo_mask to define "fillable" universe (matches v33_confluence_matrix.py logic)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)
    finite = np.isfinite(p60) & np.isfinite(p5m)
    valid = finite & fifo_mask

    p60v = p60[valid]
    p5mv = p5m[valid]
    thr_60_top = float(np.quantile(p60v, 1.0 - band_top_frac))
    thr_5m_bot = float(np.quantile(p5mv, band_top_frac))

    sel_mask = valid & (p60 >= thr_60_top) & (p5m <= thr_5m_bot)
    sel_idx = np.where(sel_mask)[0]

    # Pre-load horizon arrays for adverse selection at 30s
    tgt_lr = {}
    tgt_lr_mask = {}
    for h in ("1s", "5s", "10s", "30s"):
        tk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        if tk in d.files:
            tgt_lr[h] = d[tk].astype(np.float64)
            mask_arr = d[mk].astype(bool) if mk in d.files else None
            if mask_arr is None or mask_arr.sum() == 0:
                # Empty masks in this NPZ for >=60s; default to finite-check
                tgt_lr_mask[h] = np.isfinite(tgt_lr[h])
            else:
                tgt_lr_mask[h] = mask_arr & np.isfinite(tgt_lr[h])
        else:
            tgt_lr[h] = np.full(len(p60), np.nan)
            tgt_lr_mask[h] = np.zeros(len(p60), dtype=bool)

    preds = {
        "p60": p60,
        "p5m": p5m,
        "tgt_lr": tgt_lr,
        "tgt_lr_mask": tgt_lr_mask,
        "n": len(p60),
        "oot_dates": [str(x) for x in d["oot_dates"]],
        "target_fifo_tp4sl3_net": d["target_fifo_tp4sl3_net"].astype(np.float64),
    }

    return sel_idx, {
        "thr_pred_log_ret_60s_top20": thr_60_top,
        "thr_pred_log_ret_5min_bot20": thr_5m_bot,
        "band_top_frac": band_top_frac,
        "n_valid_universe": int(valid.sum()),
        "n_signals_k2_short": int(sel_mask.sum()),
    }, preds, valid


def k2_full_market_replay(
    order_type: str = "passive_at_touch",
    cancel_eval_window: int = 40,
    hold_seconds: float = 30.0,
    band_top_frac: float = 0.20,
    rt_commission_ticks: float = ES_RT_COMMISSION_TICKS_DEFAULT,
    spread_ticks_rth: float = ES_SPREAD_TICKS_RTH_DEFAULT,
    verbose: bool = True,
) -> dict:
    """Run the full K=2 SHORT market replay analysis.

    Mirrors the structure of full_market_replay() but with a dual-head signal rule.
    """
    t_start = time.time()

    if verbose:
        print(f"[k2_replay] Loading predictions: {PRED_NPZ}", flush=True)
    sel_idx, thresholds, preds, valid_universe = build_k2_short_signals(
        PRED_NPZ, band_top_frac=band_top_frac,
    )
    n_signals = len(sel_idx)
    if verbose:
        print(f"[k2_replay] K=2 SHORT signals: n={n_signals}", flush=True)
        print(f"[k2_replay] Thresholds: {thresholds}", flush=True)

    # Load FIFO labels for all 5 OOT days, concatenated
    oot_dates = preds["oot_dates"]
    if verbose:
        print(f"[k2_replay] OOT dates: {oot_dates}", flush=True)
        print(f"[k2_replay] Loading FIFO labels…", flush=True)
    fifo = _load_fifo_labels(FIFO_LABELS_DIR, oot_dates)
    n_fifo = int(sum(fifo["_n_per_day"]))
    n = min(preds["n"], n_fifo)
    if verbose:
        print(f"[k2_replay] preds_n={preds['n']:,}  fifo_n={n_fifo:,}  using n={n:,}", flush=True)

    # Clip sel_idx to n (should be all within range; guard anyway)
    sel_idx_clip = sel_idx[sel_idx < n]
    n_attempted = len(sel_idx_clip)
    if verbose:
        print(f"[k2_replay] n_attempted (after clip)={n_attempted}", flush=True)

    # Pull label arrays for SHORT side at the sel_idx_clip positions
    side = "short"
    side_sign = -1.0
    filled_lbl = fifo[f"tp4sl3_{side}_filled"][:n][sel_idx_clip]
    exit_reason_lbl = fifo[f"tp4sl3_{side}_exit_reason"][:n][sel_idx_clip]
    hold_time_lbl = fifo[f"tp4sl3_{side}_hold_time_ns"][:n][sel_idx_clip]
    ts_signal = fifo["ts_ns"][:n][sel_idx_clip]
    date_idx_signal = fifo["_date_idx"][:n][sel_idx_clip]

    # Queue-position model — fills among attempted signals
    filled_mask, q_arrival, avg_q_pos = _queue_position_model(
        order_type, cancel_eval_window, filled_lbl, exit_reason_lbl, hold_time_lbl,
    )
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel_idx_clip[filled_idx_in_sel]
    n_filled = int(filled_mask.sum())
    n_filled_but_cancelled = int((filled_lbl & ~filled_mask).sum())
    fill_rate = n_filled / max(1, n_attempted)
    if verbose:
        print(f"[k2_replay] n_filled={n_filled}  fill_rate={fill_rate:.3f}  "
              f"n_filled_but_cancelled={n_filled_but_cancelled}  avg_q_pos={avg_q_pos:.2f}",
              flush=True)

    # PnL — hold_seconds market exit using realized log-ret at chosen horizon
    horizon_choice = _pick_exit_horizon(hold_seconds)
    lr_exit = preds["tgt_lr"][horizon_choice][filled_global_idx]
    lr_mask_exit = preds["tgt_lr_mask"][horizon_choice][filled_global_idx]
    edge_offset = _entry_price_edge_ticks(order_type, spread_ticks_rth)
    raw_pnl_ticks = (
        side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - rt_commission_ticks
    )
    raw_pnl_ticks = np.where(lr_mask_exit, raw_pnl_ticks, 0.0)
    pnl_filled = raw_pnl_ticks

    pnl_total = float(pnl_filled.sum())
    pnl_per_trade = pnl_total / max(1, n_attempted)
    pnl_per_fill = pnl_total / max(1, n_filled)
    sharpe = _annualized(pnl_filled, downside=False)
    sortino = _annualized(pnl_filled, downside=True)
    pf = _profit_factor(pnl_filled)
    wr = float((pnl_filled > 0).mean() * 100.0) if pnl_filled.size else float("nan")
    mdd = _max_drawdown_ticks(pnl_filled)

    # Adverse selection at +30s (toxic fills)
    adv = _adverse_selection(
        filled_global_idx, side_sign, preds["tgt_lr"]["30s"], preds["tgt_lr_mask"]["30s"],
    )
    adv_avg = float(np.nanmean(adv)) if adv.size and np.isfinite(np.nanmean(adv)) else 0.0
    # toxic fill = adverse component within 5s strongly against us; proxy via 5s horizon
    adv_5s_array = _adverse_selection(
        filled_global_idx, side_sign, preds["tgt_lr"]["5s"], preds["tgt_lr_mask"]["5s"],
    )
    toxic_thresh_ticks = -1.0  # >=1 tick against us in 5s
    if adv_5s_array.size:
        toxic_mask = (adv_5s_array <= toxic_thresh_ticks) & np.isfinite(adv_5s_array)
        toxic_count = int(toxic_mask.sum())
        toxic_rate = toxic_count / max(1, n_filled)
    else:
        toxic_count, toxic_rate = 0, 0.0

    # MFE / MAE
    mfe_arr, mae_arr = _mfe_mae_per_fill(
        filled_global_idx, side_sign, {"tgt_lr": preds["tgt_lr"], "tgt_lr_mask": preds["tgt_lr_mask"]},
        hold_seconds,
    )
    avg_mfe = float(np.nanmean(mfe_arr)) if mfe_arr.size and np.isfinite(np.nanmean(mfe_arr)) else float("nan")
    avg_mae = float(np.nanmean(mae_arr)) if mae_arr.size and np.isfinite(np.nanmean(mae_arr)) else float("nan")

    commission_total = float(rt_commission_ticks * n_filled)

    # Day-concentration (HC #344 gate: ≤20% on any single day)
    n_per_day_signals = np.zeros(len(oot_dates), dtype=int)
    n_per_day_filled = np.zeros(len(oot_dates), dtype=int)
    pnl_per_day = np.zeros(len(oot_dates), dtype=float)
    for di in range(len(oot_dates)):
        sig_mask_d = (date_idx_signal == di)
        n_per_day_signals[di] = int(sig_mask_d.sum())
        fill_mask_d = sig_mask_d & filled_mask
        n_per_day_filled[di] = int(fill_mask_d.sum())
        # Sum PnL on this day (pnl_filled is sized to n_filled, indexed parallel to filled_idx_in_sel)
        if n_per_day_filled[di] > 0:
            # Map back: which entries of pnl_filled correspond to this day?
            day_filled_in_sel = np.where(filled_mask & sig_mask_d)[0]
            # Position in filled_idx_in_sel for these entries
            pos_in_pnl = np.searchsorted(filled_idx_in_sel, day_filled_in_sel)
            pnl_per_day[di] = float(pnl_filled[pos_in_pnl].sum())

    n_distinct_days_with_trades = int((n_per_day_filled > 0).sum())
    max_pct_single_day = (
        float(n_per_day_filled.max() / max(1, n_filled)) if n_filled > 0 else 0.0
    )
    hc344_pass = max_pct_single_day <= 0.20

    # Headline naive (FIFO-fillable) numbers for comparison
    fifo_net = preds["target_fifo_tp4sl3_net"][sel_idx_clip]
    naive_short_t = -fifo_net - rt_commission_ticks
    naive_mean_t = float(naive_short_t.mean())
    naive_std = float(naive_short_t.std(ddof=1)) if naive_short_t.size > 1 else 0.0
    naive_sharpe_per_trade = naive_mean_t / naive_std if naive_std > 1e-12 else float("nan")
    naive_wr = float((naive_short_t > 0).mean())
    gw_n, gl_n = naive_short_t[naive_short_t > 0].sum(), -naive_short_t[naive_short_t < 0].sum()
    naive_pf = float(gw_n / gl_n) if gl_n > 1e-12 else float("inf")

    elapsed = time.time() - t_start

    # Build per-trade dataframe
    per_trade = pd.DataFrame({
        "signal_global_idx": sel_idx_clip,
        "signal_ts_ns": ts_signal,
        "date_idx": date_idx_signal,
        "date_str": [oot_dates[i] for i in date_idx_signal],
        "pred_log_ret_60s": preds["p60"][sel_idx_clip],
        "pred_log_ret_5min": preds["p5m"][sel_idx_clip],
        "label_fifo_filled": filled_lbl,
        "label_exit_reason": exit_reason_lbl,
        "label_hold_time_ns": hold_time_lbl,
        "queue_modeled_filled": filled_mask,
        "queue_pos_on_arrival": q_arrival,
        "net_ticks_after_queue_and_adv": np.where(
            filled_mask,
            np.concatenate([raw_pnl_ticks, np.zeros(n_attempted - n_filled)]),
            0.0,
        )[:n_attempted] if False else _scatter_filled(raw_pnl_ticks, filled_mask, n_attempted),
        "mfe_ticks": _scatter_filled(mfe_arr, filled_mask, n_attempted),
        "mae_ticks": _scatter_filled(mae_arr, filled_mask, n_attempted),
        "adv_sel_30s_ticks": _scatter_filled(adv, filled_mask, n_attempted),
        "adv_sel_5s_ticks": _scatter_filled(adv_5s_array, filled_mask, n_attempted),
        "naive_fifo_short_net_t": naive_short_t,
    })

    summary = {
        "config": {
            "rule": "K=2 SHORT: pred_log_ret_60s in TOP 20%  AND  pred_log_ret_5min in BOT 20%",
            "side": "short",
            "order_type": order_type,
            "cancel_eval_window": cancel_eval_window,
            "hold_seconds": hold_seconds,
            "exit_horizon_used": horizon_choice,
            "band_top_frac": band_top_frac,
            "rt_commission_ticks": rt_commission_ticks,
            "spread_ticks_rth": spread_ticks_rth,
            "oot_dates": oot_dates,
        },
        "thresholds": thresholds,
        "naive_fifo_fillable_baseline": {
            "n": n_attempted,
            "mean_net_t": naive_mean_t,
            "sharpe_per_trade": naive_sharpe_per_trade,
            "wr": naive_wr,
            "pf": naive_pf,
            "note": (
                "PER-TRADE Sharpe. The headline 7.59 from stacked_confluence_results.json "
                "is GROSS (no commission) per-trade Sharpe; with commission it is ~0.48."
            ),
        },
        "full_market_replay_results": {
            "n_signals": n_attempted,
            "n_filled": n_filled,
            "n_filled_but_cancelled_by_queue": n_filled_but_cancelled,
            "fill_rate": fill_rate,
            "avg_queue_pos_on_arrival": avg_q_pos,
            "pnl_ticks_total": pnl_total,
            "pnl_ticks_per_trade": pnl_per_trade,
            "pnl_ticks_per_fill": pnl_per_fill,
            "sharpe_annualized": sharpe,
            "sortino_annualized": sortino,
            "profit_factor": pf,
            "win_rate_pct": wr,
            "avg_mfe_ticks": avg_mfe,
            "avg_mae_ticks": avg_mae,
            "max_drawdown_ticks": mdd,
            "adverse_selection_30s_ticks_avg": adv_avg,
            "toxic_fills_5s_le_-1t": toxic_count,
            "toxic_fill_rate_5s": toxic_rate,
            "commission_ticks_total": commission_total,
        },
        "day_concentration": {
            "n_distinct_days_with_trades": n_distinct_days_with_trades,
            "n_per_day_signals": n_per_day_signals.tolist(),
            "n_per_day_filled": n_per_day_filled.tolist(),
            "pnl_per_day_ticks": pnl_per_day.tolist(),
            "max_pct_single_day_of_filled": max_pct_single_day,
            "hc344_gate_le_20pct": bool(hc344_pass),
        },
        "elapsed_sec": elapsed,
    }
    return summary, per_trade


def _scatter_filled(values: np.ndarray, mask: np.ndarray, n_total: int) -> np.ndarray:
    out = np.full(n_total, np.nan)
    idx = np.where(mask)[0]
    if values.size == idx.size:
        out[idx] = values
    else:
        k = min(values.size, idx.size)
        out[idx[:k]] = values[:k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order-type", default="passive_at_touch",
                    choices=["passive_at_touch", "passive_at_touch_plus_1",
                             "passive_at_touch_plus_2", "ioc_market"])
    ap.add_argument("--cancel-eval-window", type=int, default=40,
                    help="evals before cancel (40 = ~10s)")
    ap.add_argument("--hold-seconds", type=float, default=30.0)
    ap.add_argument("--band-top-frac", type=float, default=0.20)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[k2_replay] === K=2 SHORT FULL MARKET REPLAY (HC #357) ===", flush=True)
    print(f"[k2_replay] order_type={args.order_type} cancel_eval_window={args.cancel_eval_window}"
          f" hold_seconds={args.hold_seconds} band={args.band_top_frac}", flush=True)

    # Run primary config + variants
    runs = []
    primary = {
        "label": "primary_passive_at_touch_30s_hold",
        "order_type": args.order_type,
        "cancel_eval_window": args.cancel_eval_window,
        "hold_seconds": args.hold_seconds,
    }
    variants = [
        {"label": "passive_5s_hold",  "order_type": "passive_at_touch", "cancel_eval_window": 40, "hold_seconds": 5.0},
        {"label": "passive_10s_hold", "order_type": "passive_at_touch", "cancel_eval_window": 40, "hold_seconds": 10.0},
        {"label": "ioc_market_30s_hold", "order_type": "ioc_market", "cancel_eval_window": 1, "hold_seconds": 30.0},
        {"label": "passive_plus1_30s_hold", "order_type": "passive_at_touch_plus_1", "cancel_eval_window": 40, "hold_seconds": 30.0},
    ]

    all_results = {}
    for cfg in [primary] + variants:
        print(f"\n[k2_replay] >>> Running variant: {cfg['label']}", flush=True)
        summary, per_trade = k2_full_market_replay(
            order_type=cfg["order_type"],
            cancel_eval_window=cfg["cancel_eval_window"],
            hold_seconds=cfg["hold_seconds"],
            band_top_frac=args.band_top_frac,
            verbose=True,
        )
        all_results[cfg["label"]] = summary
        # Save per-trade only for primary
        if cfg["label"] == primary["label"]:
            csv_path = OUT_DIR / "k2_replay_per_trade.csv"
            per_trade.to_csv(csv_path, index=False)
            print(f"[k2_replay] Wrote per-trade CSV: {csv_path}", flush=True)

        # Print summary table for this variant
        r = summary["full_market_replay_results"]
        n_b = summary["naive_fifo_fillable_baseline"]
        print(f"\n[{cfg['label']}] RESULTS:")
        print(f"  Naive FIFO baseline:  n={n_b['n']}  mean_t={n_b['mean_net_t']:+.3f}  "
              f"WR={n_b['wr']:.3f}  PF={n_b['pf']:.3f}  Sharpe(per-trade)={n_b['sharpe_per_trade']:.3f}")
        print(f"  Realized post-queue: n_filled={r['n_filled']}  fill_rate={r['fill_rate']:.3f}  "
              f"mean_t/fill={r['pnl_ticks_per_fill']:+.3f}  mean_t/trade={r['pnl_ticks_per_trade']:+.3f}")
        print(f"                       WR={r['win_rate_pct']:.2f}%  PF={r['profit_factor']:.3f}  "
              f"Sharpe(ann)={r['sharpe_annualized']:.3f}  Sortino(ann)={r['sortino_annualized']:.3f}")
        print(f"                       MFE={r['avg_mfe_ticks']:.3f}t  MAE={r['avg_mae_ticks']:.3f}t  "
              f"MDD={r['max_drawdown_ticks']:.3f}t")
        print(f"                       Adv-30s avg={r['adverse_selection_30s_ticks_avg']:+.3f}t  "
              f"Toxic 5s (≤-1t): {r['toxic_fills_5s_le_-1t']}/{r['n_filled']} ({100*r['toxic_fill_rate_5s']:.1f}%)")
        d = summary["day_concentration"]
        print(f"  Days with trades: {d['n_distinct_days_with_trades']}/{len(summary['config']['oot_dates'])}  "
              f"max pct single day: {100*d['max_pct_single_day_of_filled']:.1f}%  "
              f"HC #344 gate: {'PASS' if d['hc344_gate_le_20pct'] else 'FAIL'}")

    json_path = OUT_DIR / "k2_replay_summary.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\n[k2_replay] Wrote summary JSON: {json_path}", flush=True)

    # ── FINAL VERDICT ──────────────────────────────────────────────────────
    primary_res = all_results[primary["label"]]["full_market_replay_results"]
    primary_baseline = all_results[primary["label"]]["naive_fifo_fillable_baseline"]
    print(f"\n{'='*72}")
    print(f"VERDICT — does headline Sharpe 7.59 hold under full market replay?")
    print(f"{'='*72}")
    print(f"  Reported headline (FIFO-fillable, GROSS):  Sharpe=7.59  PF=2.55  WR=77.2%  mean=+1.41t")
    print(f"  Same baseline, NET commission (per-trade Sharpe):")
    print(f"      Sharpe(per-trade)={primary_baseline['sharpe_per_trade']:.3f}  PF={primary_baseline['pf']:.3f}  "
          f"WR={primary_baseline['wr']:.3f}  mean_t={primary_baseline['mean_net_t']:+.3f}")
    print(f"  Full market replay (queue+adv-sel, passive_at_touch, 30s hold):")
    print(f"      fill_rate={primary_res['fill_rate']:.3f}  n_filled={primary_res['n_filled']}/{primary_res['n_signals']}")
    print(f"      mean_t/fill={primary_res['pnl_ticks_per_fill']:+.3f}  mean_t/trade={primary_res['pnl_ticks_per_trade']:+.3f}")
    print(f"      Sharpe(ann)={primary_res['sharpe_annualized']:.3f}  Sortino(ann)={primary_res['sortino_annualized']:.3f}")
    print(f"      PF={primary_res['profit_factor']:.3f}  WR={primary_res['win_rate_pct']:.2f}%")
    print(f"      Adverse-selection 30s avg: {primary_res['adverse_selection_30s_ticks_avg']:+.3f}t")
    print(f"      Toxic fill rate (5s ≤ -1t against us): {100*primary_res['toxic_fill_rate_5s']:.1f}%")
    d = all_results[primary["label"]]["day_concentration"]
    print(f"      Day concentration: max {100*d['max_pct_single_day_of_filled']:.1f}% on single day "
          f"({'PASS' if d['hc344_gate_le_20pct'] else 'FAIL'} HC #344 ≤20% gate)")
    print(f"{'='*72}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
