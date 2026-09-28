#!/usr/bin/env python3
"""
K=2 LONG vol30s<1.75 — FULL MARKET REPLAY on 15-day base (HC #380 + HC #377 + HC #344)

USER MANDATE (HC #380, 2026-05-15 19:34 ET):
  "WAIT we have one real edge standing??? test on more dates!!!"

THE EDGE BEING TESTED:
  K=2 LONG rule (inverted from K=2 SHORT):
    pred_log_ret_60s   in BOTTOM 20% of fillable distribution
    pred_log_ret_5min  in TOP    20% of fillable distribution
    pred_realized_vol_30s_ticks < 1.75   (volatility filter)
  Original 5-day OOT report: +1.94 t/fill, 76% WR, 50 fills, BUT only 3 firing days.

HC #376 CRITICAL CAVEAT (NOT SKIPPED):
  pred_log_ret_60s   has ic = None in metrics.json  -> UNTRAINED HEAD
  pred_log_ret_5min  has ic = None in metrics.json  -> UNTRAINED HEAD
  pred_realized_vol_30s_ticks has corr != None       -> TRAINED HEAD (filter OK)
  The two mask heads received NO supervision signal during training (labels all
  zero). Their outputs are produced by the shared trunk plus an untrained head.
  This 15-day test is the FALSIFICATION TEST: does the edge survive expansion?

HC #377 STACK (all 5 components, no shortcuts):
  1. Queue position model (_queue_position_model from full_market_replay)
  2. Adverse selection at +30s and +5s (toxic-fill counter)
  3. Cancellation patterns (cancel_eval_window=40 default)
  4. Commission ($4.70/$12.50 = 0.376 ticks RT)
  5. Day concentration (HC #344 gate: max single-day share <= 0.20)

USAGE:
  python scripts/v3_3_research/k2_long_vol_full_market_replay.py \\
      --base-preds output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz \\
      --chunk-preds output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_ext_oot_chunk1_predictions.npz \\
      --vol-thresh 1.75 \\
      --band-frac 0.20 \\
      --order-type passive_at_touch \\
      --cancel-eval-window 40 \\
      --hold-seconds 30.0

NOT MALWARE. Pure analysis script. Reuses existing full_market_replay machinery
without modification. Writes only to output dir.
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

# Reuse canonical machinery (HC #357 + HC #377)
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

FIFO_LABELS_DIR = ROOT / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = ROOT / "output/v3_3_full_execution_analysis_20260514/k2_long_vol_market_replay"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# HC #376 head-validity check
TRAINED_HEADS = {
    "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s", "pred_log_ret_30s",
    "pred_realized_vol_30s_ticks",
    "pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s",
    "pred_p_reversal_15s", "pred_p_reversal_30s",
    "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks", "pred_pred_time_to_mfe_secs",
    "pred_fifo_tp4sl3_net", "pred_fifo_tp4sl3_hit_tp",
    "pred_fifo_tp8sl5_net", "pred_fifo_tp8sl5_hit_tp",
}
UNTRAINED_HEADS = {
    "pred_log_ret_60s", "pred_log_ret_5min",
    "pred_pred_mfe_60s_ticks", "pred_pred_mae_60s_ticks",
    "pred_p_up_60s", "pred_p_reversal_60s",
}


def _scatter_filled(filled_array: np.ndarray, filled_mask: np.ndarray, n_attempted: int) -> np.ndarray:
    """Scatter per-fill values back to per-attempt array (zeros for non-fills)."""
    out = np.zeros(n_attempted, dtype=np.float64)
    idx = np.where(filled_mask)[0]
    out[idx] = filled_array
    return out


def load_and_concat_predictions(base_path: Path, chunk_path: Optional[Path]) -> dict:
    """Load 5-day base predictions; optionally concatenate chunk1 predictions.

    Returns a dict with the same keys as a single NPZ but spanning all dates.
    Preserves the date ordering: base dates first, then chunk dates.
    """
    paths = [base_path]
    if chunk_path is not None and chunk_path.exists():
        paths.append(chunk_path)
    elif chunk_path is not None:
        print(f"[WARN] chunk-preds path missing, falling back to base only: {chunk_path}")

    all_data = {}
    all_dates: list[str] = []
    SCALAR_SKIP = set()  # keys that are scalars in any input - skip from concat
    for p in paths:
        print(f"[load] {p}")
        d = np.load(p, allow_pickle=True)
        n_local = len(d["pred_log_ret_60s"])
        dates_local = [str(x) for x in d["oot_dates"]]
        print(f"  n={n_local:,}  dates={dates_local}")
        for key in d.files:
            if key == "oot_dates":
                continue
            arr = d[key]
            # Only concatenate per-sample arrays (those with leading axis == n_local).
            if arr.ndim == 0 or (arr.ndim >= 1 and arr.shape[0] != n_local):
                SCALAR_SKIP.add(key)
                continue
            if key not in all_data:
                all_data[key] = [arr]
            else:
                all_data[key].append(arr)
        all_dates.extend(dates_local)
    if SCALAR_SKIP:
        print(f"[load] skipped non-per-sample keys: {sorted(SCALAR_SKIP)}")

    out = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    out["oot_dates"] = all_dates
    print(f"[load] combined n={len(out['pred_log_ret_60s']):,}  dates={all_dates}")
    return out


def build_k2_long_vol_signals(combined: dict, band_frac: float = 0.20, vol_thresh: float = 1.75):
    """K=2 LONG + vol filter.

    Rule:
      pred_log_ret_60s   <= BOT 20%
      pred_log_ret_5min  >= TOP 20%
      pred_realized_vol_30s_ticks < 1.75

    Universe = isfinite(all three) & fifo_mask.
    """
    p60 = combined["pred_log_ret_60s"].astype(np.float64)
    p5m = combined["pred_log_ret_5min"].astype(np.float64)

    vol_key = "pred_realized_vol_30s_ticks"
    if vol_key not in combined:
        # Some NPZ schemas use slight variations
        for alt in ("pred_pred_realized_vol_30s_ticks", "pred_realized_vol_30s"):
            if alt in combined:
                vol_key = alt
                break
        else:
            raise KeyError(f"vol head not found; have keys: {list(combined.keys())[:30]}")
    vol = combined[vol_key].astype(np.float64)

    fifo_mask = combined["mask_fifo_tp4sl3_net"].astype(bool)
    finite = np.isfinite(p60) & np.isfinite(p5m) & np.isfinite(vol)
    valid = finite & fifo_mask

    p60v = p60[valid]
    p5mv = p5m[valid]
    thr_60_bot = float(np.quantile(p60v, band_frac))
    thr_5m_top = float(np.quantile(p5mv, 1.0 - band_frac))

    # K=2 LONG (inverted from K=2 SHORT) + vol filter
    sel_mask = valid & (p60 <= thr_60_bot) & (p5m >= thr_5m_top) & (vol < vol_thresh)
    sel_idx = np.where(sel_mask)[0]

    # Adverse-selection horizons (5s, 10s, 30s only — those heads ARE trained)
    tgt_lr = {}
    tgt_lr_mask = {}
    for h in ("1s", "5s", "10s", "30s"):
        tk = f"target_log_ret_{h}"
        mk = f"mask_log_ret_{h}"
        if tk in combined:
            tgt_lr[h] = combined[tk].astype(np.float64)
            mask_arr = combined[mk].astype(bool) if mk in combined else None
            if mask_arr is None or mask_arr.sum() == 0:
                tgt_lr_mask[h] = np.isfinite(tgt_lr[h])
            else:
                tgt_lr_mask[h] = mask_arr & np.isfinite(tgt_lr[h])
        else:
            tgt_lr[h] = np.full(len(p60), np.nan)
            tgt_lr_mask[h] = np.zeros(len(p60), dtype=bool)

    preds = {
        "p60": p60, "p5m": p5m, "vol30s": vol,
        "tgt_lr": tgt_lr, "tgt_lr_mask": tgt_lr_mask,
        "n": len(p60),
        "oot_dates": combined["oot_dates"],
        "target_fifo_tp4sl3_net": combined["target_fifo_tp4sl3_net"].astype(np.float64),
    }
    return sel_idx, {
        "thr_pred_log_ret_60s_bot20": thr_60_bot,
        "thr_pred_log_ret_5min_top20": thr_5m_top,
        "vol_thresh": vol_thresh,
        "band_frac": band_frac,
        "n_valid_universe": int(valid.sum()),
        "n_signals_k2_long_vol": int(sel_mask.sum()),
    }, preds


def run(args):
    t_start = time.time()
    base_path = Path(args.base_preds)
    chunk_path = Path(args.chunk_preds) if args.chunk_preds else None
    print(f"[k2_LONG_vol] base={base_path}")
    print(f"[k2_LONG_vol] chunk={chunk_path}")

    combined = load_and_concat_predictions(base_path, chunk_path)
    sel_idx, thresholds, preds = build_k2_long_vol_signals(
        combined, band_frac=args.band_frac, vol_thresh=args.vol_thresh,
    )
    n_signals = len(sel_idx)
    print(f"[k2_LONG_vol] signals after K=2+vol filter: n={n_signals}")
    print(f"[k2_LONG_vol] thresholds: {json.dumps(thresholds, indent=2)}")

    if n_signals == 0:
        out = {
            "status": "no_signals",
            "thresholds": thresholds,
            "oot_dates": preds["oot_dates"],
            "hc376_caveat": "K=2 LONG uses untrained heads pred_log_ret_60s and pred_log_ret_5min",
        }
        (OUT_DIR / "k2_long_vol_summary.json").write_text(json.dumps(out, indent=2, default=str))
        print("[k2_LONG_vol] no signals — wrote summary")
        return 0

    oot_dates = preds["oot_dates"]
    print(f"[k2_LONG_vol] OOT dates ({len(oot_dates)}): {oot_dates}")
    fifo = _load_fifo_labels(FIFO_LABELS_DIR, oot_dates)
    n_fifo = int(sum(fifo["_n_per_day"]))
    n = min(preds["n"], n_fifo)
    print(f"[k2_LONG_vol] preds_n={preds['n']:,}  fifo_n={n_fifo:,}  using n={n:,}")

    sel_idx_clip = sel_idx[sel_idx < n]
    n_attempted = len(sel_idx_clip)
    print(f"[k2_LONG_vol] n_attempted={n_attempted}")

    side = "long"
    side_sign = +1.0
    filled_lbl = fifo[f"tp4sl3_{side}_filled"][:n][sel_idx_clip]
    exit_reason_lbl = fifo[f"tp4sl3_{side}_exit_reason"][:n][sel_idx_clip]
    hold_time_lbl = fifo[f"tp4sl3_{side}_hold_time_ns"][:n][sel_idx_clip]
    ts_signal = fifo["ts_ns"][:n][sel_idx_clip]
    date_idx_signal = fifo["_date_idx"][:n][sel_idx_clip]

    filled_mask, q_arrival, avg_q_pos = _queue_position_model(
        args.order_type, args.cancel_eval_window, filled_lbl, exit_reason_lbl, hold_time_lbl,
    )
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel_idx_clip[filled_idx_in_sel]
    n_filled = int(filled_mask.sum())
    fill_rate = n_filled / max(1, n_attempted)
    print(f"[k2_LONG_vol] n_filled={n_filled}  fill_rate={fill_rate:.3f}  avg_q_pos={avg_q_pos:.2f}")

    horizon_choice = _pick_exit_horizon(args.hold_seconds)
    lr_exit = preds["tgt_lr"][horizon_choice][filled_global_idx]
    lr_mask_exit = preds["tgt_lr_mask"][horizon_choice][filled_global_idx]
    edge_offset = _entry_price_edge_ticks(args.order_type, ES_SPREAD_TICKS_RTH_DEFAULT)
    raw_pnl_ticks = (
        side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - ES_RT_COMMISSION_TICKS_DEFAULT
    )
    raw_pnl_ticks = np.where(lr_mask_exit, raw_pnl_ticks, 0.0)

    pnl_total = float(raw_pnl_ticks.sum())
    pnl_per_fill = pnl_total / max(1, n_filled)
    sharpe = _annualized(raw_pnl_ticks, downside=False)
    sortino = _annualized(raw_pnl_ticks, downside=True)
    pf = _profit_factor(raw_pnl_ticks)
    wr = float((raw_pnl_ticks > 0).mean() * 100.0) if raw_pnl_ticks.size else float("nan")
    mdd = _max_drawdown_ticks(raw_pnl_ticks)

    # Adverse selection
    adv30 = _adverse_selection(filled_global_idx, side_sign, preds["tgt_lr"]["30s"], preds["tgt_lr_mask"]["30s"])
    adv5 = _adverse_selection(filled_global_idx, side_sign, preds["tgt_lr"]["5s"], preds["tgt_lr_mask"]["5s"])
    adv30_avg = float(np.nanmean(adv30)) if adv30.size and np.isfinite(np.nanmean(adv30)) else float("nan")
    toxic_thresh_ticks = -1.0
    toxic_count = int((adv5 <= toxic_thresh_ticks).sum()) if adv5.size else 0
    toxic_rate = toxic_count / max(1, n_filled)

    mfe_arr, mae_arr = _mfe_mae_per_fill(
        filled_global_idx, side_sign,
        {"tgt_lr": preds["tgt_lr"], "tgt_lr_mask": preds["tgt_lr_mask"]},
        args.hold_seconds,
    )
    avg_mfe = float(np.nanmean(mfe_arr)) if mfe_arr.size and np.isfinite(np.nanmean(mfe_arr)) else float("nan")
    avg_mae = float(np.nanmean(mae_arr)) if mae_arr.size and np.isfinite(np.nanmean(mae_arr)) else float("nan")

    # Day concentration (HC #344)
    n_per_day_signals = np.zeros(len(oot_dates), dtype=int)
    n_per_day_filled = np.zeros(len(oot_dates), dtype=int)
    pnl_per_day = np.zeros(len(oot_dates), dtype=float)
    for di in range(len(oot_dates)):
        sig_mask_d = (date_idx_signal == di)
        n_per_day_signals[di] = int(sig_mask_d.sum())
        fill_mask_d = sig_mask_d & filled_mask
        n_per_day_filled[di] = int(fill_mask_d.sum())
        if n_per_day_filled[di] > 0:
            day_filled_in_sel = np.where(filled_mask & sig_mask_d)[0]
            pos_in_pnl = np.searchsorted(filled_idx_in_sel, day_filled_in_sel)
            pnl_per_day[di] = float(raw_pnl_ticks[pos_in_pnl].sum())

    n_firing_days = int((n_per_day_filled > 0).sum())
    max_pct_single_day_fills = float(n_per_day_filled.max() / max(1, n_filled)) if n_filled > 0 else 0.0
    # HC #344 on PnL: abs share of any single day in total
    if abs(pnl_total) > 1e-9:
        day_conc_pnl = float(np.abs(pnl_per_day).max() / abs(pnl_total))
    else:
        day_conc_pnl = float("nan")
    hc344_pass_fills = max_pct_single_day_fills <= 0.20
    hc344_pass_pnl = (day_conc_pnl <= 0.20) if np.isfinite(day_conc_pnl) else False

    commission_total = float(ES_RT_COMMISSION_TICKS_DEFAULT * n_filled)

    elapsed = time.time() - t_start

    per_trade = pd.DataFrame({
        "signal_global_idx": sel_idx_clip,
        "signal_ts_ns": ts_signal,
        "date_idx": date_idx_signal,
        "date_str": [oot_dates[i] for i in date_idx_signal],
        "pred_log_ret_60s": preds["p60"][sel_idx_clip],
        "pred_log_ret_5min": preds["p5m"][sel_idx_clip],
        "pred_vol_30s_ticks": preds["vol30s"][sel_idx_clip],
        "label_fifo_filled": filled_lbl,
        "queue_modeled_filled": filled_mask,
        "queue_pos_on_arrival": q_arrival,
        "net_ticks": _scatter_filled(raw_pnl_ticks, filled_mask, n_attempted),
        "mfe_ticks": _scatter_filled(mfe_arr, filled_mask, n_attempted),
        "mae_ticks": _scatter_filled(mae_arr, filled_mask, n_attempted),
        "adv_sel_30s_ticks": _scatter_filled(adv30, filled_mask, n_attempted),
        "adv_sel_5s_ticks": _scatter_filled(adv5, filled_mask, n_attempted),
    })

    per_day = pd.DataFrame({
        "date": oot_dates,
        "n_signals": n_per_day_signals,
        "n_filled": n_per_day_filled,
        "pnl_ticks": pnl_per_day,
        "pnl_share_of_total": (
            pnl_per_day / pnl_total if abs(pnl_total) > 1e-9 else np.zeros_like(pnl_per_day)
        ),
    })

    summary = {
        "rule": "K=2 LONG + vol30s<1.75 — pred_log_ret_60s in BOT 20% AND pred_log_ret_5min in TOP 20% AND vol30s < 1.75",
        "hc376_caveat": (
            "K=2 mask uses UNTRAINED heads pred_log_ret_60s and pred_log_ret_5min "
            "(ic=None in metrics.json). Vol filter uses TRAINED head pred_realized_vol_30s_ticks. "
            "This test is the falsification of whether the +1.94 t/fill 3-day result was real signal "
            "from the shared trunk or coincidence from untrained-head noise."
        ),
        "config": {
            "band_frac": args.band_frac,
            "vol_thresh": args.vol_thresh,
            "order_type": args.order_type,
            "cancel_eval_window": args.cancel_eval_window,
            "hold_seconds": args.hold_seconds,
            "rt_commission_ticks": ES_RT_COMMISSION_TICKS_DEFAULT,
            "spread_ticks_rth": ES_SPREAD_TICKS_RTH_DEFAULT,
        },
        "thresholds": thresholds,
        "n_oot_dates": len(oot_dates),
        "oot_dates": oot_dates,
        "n_attempted": n_attempted,
        "n_filled": n_filled,
        "fill_rate": fill_rate,
        "n_firing_days": n_firing_days,
        "max_pct_single_day_fills": max_pct_single_day_fills,
        "day_conc_pnl": day_conc_pnl,
        "hc344_pass_fills": hc344_pass_fills,
        "hc344_pass_pnl": hc344_pass_pnl,
        "pnl_ticks_total": pnl_total,
        "pnl_ticks_per_fill": pnl_per_fill,
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
        "profit_factor": pf,
        "win_rate_pct": wr,
        "max_drawdown_ticks": mdd,
        "adv_sel_30s_avg_ticks": adv30_avg,
        "toxic_fill_count_5s_le_-1t": toxic_count,
        "toxic_fill_rate": toxic_rate,
        "avg_mfe_ticks": avg_mfe,
        "avg_mae_ticks": avg_mae,
        "commission_ticks_total": commission_total,
        "avg_queue_position_on_arrival": float(avg_q_pos),
        "per_day": per_day.to_dict(orient="records"),
        "elapsed_s": elapsed,
    }
    (OUT_DIR / "k2_long_vol_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    per_trade.to_csv(OUT_DIR / "k2_long_vol_per_trade.csv", index=False)
    per_day.to_csv(OUT_DIR / "k2_long_vol_per_day.csv", index=False)
    print(f"[k2_LONG_vol] wrote summary + per_trade + per_day to {OUT_DIR}")

    # Brief stdout verdict
    print("\n" + "=" * 70)
    print("K=2 LONG vol30s<1.75 VERDICT (full_market_replay, HC #377 stack)")
    print("=" * 70)
    print(f"OOT dates: {len(oot_dates)}  ({oot_dates[0]} ... {oot_dates[-1]})")
    print(f"n_attempted: {n_attempted}   n_filled: {n_filled}   fill_rate: {fill_rate:.3f}")
    print(f"Firing days: {n_firing_days} / {len(oot_dates)}")
    print(f"day_conc (fills): {max_pct_single_day_fills:.3f}  pass HC #344 (<=0.20)? {hc344_pass_fills}")
    print(f"day_conc (PnL):   {day_conc_pnl:.3f}  pass HC #344 (<=0.20)? {hc344_pass_pnl}")
    print(f"PnL per fill (net ticks, after queue+adv+comm): {pnl_per_fill:+.3f}")
    print(f"Win rate: {wr:.1f}%   PF: {pf:.2f}   MDD: {mdd:.1f} ticks")
    print(f"Sharpe: {sharpe:.2f}  Sortino: {sortino:.2f}   (NOTE: _annualized helper currently suspect — see Jupiter sweep report)")
    print(f"Adv-sel 30s avg: {adv30_avg:+.3f} t   Toxic fill rate (5s<=-1t): {toxic_rate:.3f}")
    print(f"MFE avg: {avg_mfe:+.3f}t   MAE avg: {avg_mae:+.3f}t")
    print(f"HC #376 caveat: K=2 mask uses UNTRAINED heads — interpretive ceiling applies.")
    print("=" * 70)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-preds", required=True, help="Path to 5-day OOT predictions NPZ")
    ap.add_argument("--chunk-preds", default=None, help="Path to chunk1 (10-day) extended predictions NPZ (optional)")
    ap.add_argument("--vol-thresh", type=float, default=1.75)
    ap.add_argument("--band-frac", type=float, default=0.20)
    ap.add_argument("--order-type", default="passive_at_touch")
    ap.add_argument("--cancel-eval-window", type=int, default=40)
    ap.add_argument("--hold-seconds", type=float, default=30.0)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
