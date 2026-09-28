#!/usr/bin/env python3
"""
HC #437 Bug 2 — v2 (cnn_mamba_v2) FIFO sanity baseline under HC #413 BRACKET mode.

Mirrors `hc432_v2_baseline_runner.py` but invokes FIFOReplayEngine.simulate(
order_management='hc413_bracket') so exits are evaluated at HC #413
horizon-checkpoint TP1/TP2/SL brackets rather than per-event TP/SL price
tracking.

Per the HC #413 reference output (output/hc417_hc413_v2native_mfe/
scalping_backtest_results.csv:14), the v2 1s short top0.5 cell with
TP1=0.4782, TP2=0.9564, SL=0.5686 (derived from MFE=0.9564, MAE=0.5686 in
the v2-native MFE matrix) produces:
    n_fills=639, realized_net_per_fill=+0.274 tk, Sharpe_sqrtN=+12.77,
    PF=2.92, WR=85%, TP1=442, TP2=100, SL=95, time_stop=2.

If this runner reproduces those numbers within +/- 0.05 tk, the bracket
mode is verified and the HC #437 R1 sanity gate is satisfied under that
methodology.

Outputs (under output/hc437_harness_debug/):
    {config_name}_bracket_fifo_fills.csv
    {config_name}_bracket_fifo_summary.json

CLI mirrors hc432_v2_baseline_runner with extra bracket params:
    --tp1-ticks  --tp2-ticks  --sl-bracket-ticks
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

V2_OOT_DIR = LVL3 / "output" / "cnn_mamba_v2_bulk_oot_v2"
OUT_DIR = LVL3 / "output" / "hc437_harness_debug"
MBO_EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"

HORIZON_IDX = {"1": 0, "5": 1, "10": 2}
HORIZON_SEC = {"1": 1.0, "5": 5.0, "10": 10.0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hc437_v2_bracket")


def parse_conf_band(band: str) -> float:
    if not band.startswith("top"):
        raise ValueError(band)
    return float(band[3:]) / 100.0


def list_dates() -> List[str]:
    return [p.stem.split("_")[0]
            for p in sorted(V2_OOT_DIR.glob("2026*_predictions.npz"))]


def load_v2_day(date_str: str, horizon: str
                ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return (preds_at_horizon, labels_3xN, window_size, stride).

    labels_3xN is the FULL (N,3) realized log-return array (in ticks) for
    horizons 1s/5s/10s — needed for bracket mode.
    """
    p = V2_OOT_DIR / f"{date_str}_predictions.npz"
    d = np.load(p, allow_pickle=False)
    preds = d["predictions"][:, HORIZON_IDX[horizon]].astype(np.float64)
    labels_full = d["labels"].astype(np.float64)  # (N, 3): [1s, 5s, 10s]
    ws = int(d["window_size"])
    st = int(d["stride"])
    return preds, labels_full, ws, st


def select_signals_global(
    dates: List[str], horizon: str, side: str, conf_band: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    """
    Pool all dates, pick top-N% of side-aligned preds GLOBALLY, then group by date.

    Returns: { date: (idx_in_day, strengths, labels_full_at_idx (3xN_sel), ws, st) }
    """
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    day_meta: List[Tuple[str, int, int, int]] = []
    for d in dates:
        preds, labels_full, ws, st = load_v2_day(d, horizon)
        all_preds.append(preds)
        all_labels.append(labels_full)
        day_meta.append((d, preds.size, ws, st))

    preds_concat = np.concatenate(all_preds)
    mask_side = (preds_concat > 0) if side == "long" else (preds_concat < 0)
    strength = preds_concat if side == "long" else -preds_concat

    frac = parse_conf_band(conf_band)
    side_str = strength[mask_side]
    if side_str.size == 0:
        return {}
    k = max(1, int(side_str.size * frac))
    thresh = np.partition(side_str, -k)[-k]
    selected = mask_side & (strength >= thresh)
    log.info(f"  v2 selected {int(selected.sum()):,} signals "
             f"(k={k:,} thresh={thresh:.4g})")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = {}
    cursor = 0
    for (d, n, ws, st), labels_day in zip(day_meta, all_labels):
        sel_day = selected[cursor:cursor + n]
        if sel_day.any():
            idx = np.flatnonzero(sel_day)
            strs = strength[cursor:cursor + n][idx]
            labels_sel = labels_day[idx]  # (k, 3) for selected windows
            out[d] = (idx, strs.astype(np.float64),
                      labels_sel.astype(np.float64), ws, st)
        cursor += n
    return out


def map_to_ts(date_str: str, idx_in_day: np.ndarray,
              window: int, stride: int) -> np.ndarray:
    mbo = np.load(MBO_EVENT_DIR / f"{date_str}_mbo_events.npz",
                  allow_pickle=False)
    ts = mbo["timestamps"].astype(np.int64)
    n = len(ts)
    event_idx = np.minimum(idx_in_day * stride + window - 1, n - 1)
    return ts[event_idx]


def run_one_date(
    date_str: str, idx: np.ndarray, strengths: np.ndarray,
    labels_sel: np.ndarray,  # (k, 3) [1s, 5s, 10s] in ticks
    window: int, stride: int, side: str,
    tp1: float, tp2: float, sl: float,
    hold_s: float, cancel_s: float, order: str,
) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    mbo_p = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_p.exists():
        return [{"date": date_str, "error": "missing_mbo_events"}]

    ts_ns = map_to_ts(date_str, idx, window, stride)
    signals = []
    for i, (t, s) in enumerate(zip(ts_ns, strengths)):
        lbl_row = labels_sel[i]
        labels_by_h = {1: float(lbl_row[0]),
                       5: float(lbl_row[1]),
                       10: float(lbl_row[2])}
        signals.append({
            "ts_ns": int(t), "direction": side, "strength": float(s),
            "labels_by_h": labels_by_h,
        })

    engine_order = {"passive_at_touch": "limit",
                    "market": "market", "chase": "chase"}[order]
    cancel_ns = int(cancel_s * 1e9)
    hold_ns = int(hold_s * 1e9)
    try:
        engine = FIFOReplayEngine(date=date_str,
                                  cancel_after_ns=cancel_ns,
                                  max_hold_ns=hold_ns)
    except FileNotFoundError as e:
        return [{"date": date_str, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date_str, "error": f"engine_init: {e}"}]
    try:
        # tp_ticks/sl_ticks are IGNORED in bracket mode but the param is required.
        trades = engine.simulate(
            signals=signals,
            tp_ticks=tp2, sl_ticks=sl,  # benign placeholders for bracket mode
            order_type=engine_order,
            order_management='hc413_bracket',
            bracket_thresholds={'tp1': tp1, 'tp2': tp2, 'sl': sl},
            bracket_horizons_sec=(1.0, 5.0, 10.0),
        )
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]
    out = []
    for t in trades:
        out.append({
            "date": date_str,
            "ts_signal_ns": int(t.signal_ts_ns),
            "ts_entry_ns": int(t.entry_ts_ns or 0),
            "ts_exit_ns": int(t.exit_ts_ns or 0),
            "direction": t.direction,
            "order_type": t.order_type,
            "entry_raw": int(t.entry_price_raw or 0),
            "exit_raw": int(t.exit_price_raw or 0),
            "hold_s": (t.exit_ts_ns - t.entry_ts_ns) / 1e9
                      if t.entry_ts_ns and t.exit_ts_ns else 0.0,
            "fill_type": t.exit_reason,
            "gross_ticks": float(t.pnl_ticks),
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "queue_ahead": int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="1", choices=list(HORIZON_IDX.keys()))
    ap.add_argument("--side", default="short", choices=["long", "short"])
    ap.add_argument("--conf-band", default="top0.5")
    # HC #413 bracket thresholds (default = v2 1s short top0.5 reference cell)
    ap.add_argument("--tp1-ticks", type=float, default=0.4782)
    ap.add_argument("--tp2-ticks", type=float, default=0.9564)
    ap.add_argument("--sl-bracket-ticks", type=float, default=0.5686)
    ap.add_argument("--hold-s", type=float, default=10.0,
                    help="cancel/timeout window for the limit order (10s default "
                         "to match HC #413's cancel-eval-window of 40 evals)")
    ap.add_argument("--cancel-s", type=float, default=10.0)
    ap.add_argument("--order-type", default="passive_at_touch")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--config-name",
                    default="v2_short_1s_top0.5_bracket_HC437")
    ap.add_argument("--limit-dates", type=int, default=None)
    args = ap.parse_args()

    dates = list_dates()
    if args.limit_dates:
        dates = dates[:args.limit_dates]
    log.info(f"v2 bracket baseline: {len(dates)} dates  cfg={args.config_name}")
    log.info(f"  TP1={args.tp1_ticks}t  TP2={args.tp2_ticks}t  "
             f"SL={args.sl_bracket_ticks}t  order={args.order_type}")

    sig_by_date = select_signals_global(
        dates, args.horizon, args.side, args.conf_band)
    log.info(f"  signal days: {len(sig_by_date)}")

    fills_all: List[dict] = []
    errors: List[dict] = []
    tasks = [(d, idx, strs, lbls, ws, st)
             for d, (idx, strs, lbls, ws, st) in sig_by_date.items()]
    workers = max(1, min(args.workers, len(tasks)))

    if workers == 1 or len(tasks) <= 1:
        for d, idx, strs, lbls, ws, st in tasks:
            r = run_one_date(d, idx, strs, lbls, ws, st, args.side,
                             args.tp1_ticks, args.tp2_ticks,
                             args.sl_bracket_ticks,
                             args.hold_s, args.cancel_s, args.order_type)
            for f in r:
                (errors if "error" in f else fills_all).append(f)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(run_one_date, d, idx, strs, lbls, ws, st,
                              args.side,
                              args.tp1_ticks, args.tp2_ticks,
                              args.sl_bracket_ticks,
                              args.hold_s, args.cancel_s,
                              args.order_type): d
                    for d, idx, strs, lbls, ws, st in tasks}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    errors.append({"date": d, "error": f"future: {e}"})
                    continue
                for f in r:
                    (errors if "error" in f else fills_all).append(f)
                log.info(f"  {d}: {len([f for f in r if 'error' not in f])} fills")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{args.config_name}_bracket_fifo_fills.csv"
    if fills_all:
        pd.DataFrame(fills_all).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("")
    log.info(f"wrote {csv_path}  fills={len(fills_all)}  errors={len(errors)}")

    # First-pass aggregate metrics
    summary_metrics = {}
    if fills_all:
        df = pd.DataFrame(fills_all)
        nets = df["net_ticks"].to_numpy()
        gross = df["gross_ticks"].to_numpy()
        wins = (nets > 0).sum()
        gp = nets[nets > 0].sum()
        gl = -nets[nets < 0].sum()
        pf = float(gp / gl) if gl > 0 else float("inf")
        wr = float(100.0 * wins / len(nets))
        sharpe = float((nets.mean() / (nets.std(ddof=1) + 1e-12))
                       * np.sqrt(len(nets))) if len(nets) > 1 else 0.0
        # Exit reason counts
        rc = df["fill_type"].value_counts().to_dict()
        summary_metrics = {
            "n_fills": int(len(nets)),
            "mean_gross_ticks": float(gross.mean()),
            "mean_net_ticks": float(nets.mean()),
            "pf": pf, "wr_pct": wr, "sharpe_sqrtN": sharpe,
            "exit_counts": {str(k): int(v) for k, v in rc.items()},
        }
        log.info(
            f"FIRST-PASS bracket: n={len(nets)} "
            f"mean_gross={gross.mean():+.4f} mean_net={nets.mean():+.4f} "
            f"PF={pf:.3f} WR={wr:.2f}% Sharpe(sqrtN)={sharpe:.3f}  "
            f"exits={summary_metrics['exit_counts']}")

    (OUT_DIR / f"{args.config_name}_bracket_fifo_summary.json").write_text(
        json.dumps({
            "config_name": args.config_name,
            "mode": "hc413_bracket",
            "config": {
                "horizon": args.horizon, "side": args.side,
                "conf_band": args.conf_band,
                "tp1_ticks": args.tp1_ticks,
                "tp2_ticks": args.tp2_ticks,
                "sl_ticks": args.sl_bracket_ticks,
                "hold_s": args.hold_s, "cancel_s": args.cancel_s,
                "order_type": args.order_type,
            },
            "n_dates": len(dates),
            "n_signal_days": len(sig_by_date),
            "n_fills": len(fills_all),
            "n_errors": len(errors),
            "errors": errors[:20],
            "metrics": summary_metrics,
            "csv_path": str(csv_path),
        }, indent=2))


if __name__ == "__main__":
    main()
