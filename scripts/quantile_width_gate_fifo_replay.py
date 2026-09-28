#!/usr/bin/env python3
"""
quantile_width_gate_fifo_replay.py — FIFO market replay for DLinear quantile
width-gated short signals (HC #491 R4).

Tests whether the quantile model's confidence width (P90 - P10) can discriminate
high vs low quality trades under REALISTIC fills (FIFO queue simulation).

Proxy analysis showed:
  - Narrow-width shorts: +0.37 t/trade, 18/21 profitable days (5s horizon)
  - Wide-width shorts:   -0.12 t/trade, 10/21 profitable days
  - Width-gate lift:      +0.49 t/trade

This script validates whether that lift survives FIFO queue-position realism.

Model settings (from train_dlinear_quantile_v1.py):
  WINDOW = 500, OOT_STRIDE = 5
  So prediction i maps to MBO event index: i * 5 + 499

Cost model (canonical):
  Entry: passive limit at touch (ask for short)
  Exit: market at max_hold (1.5 * horizon)
  Commission: 0.376 t RT
  Market exit spread: +1.0 t crossing cost

Run:
  python3 scripts/quantile_width_gate_fifo_replay.py [--horizon 5] [--conf-pct 10] [--width-pct 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.deep_models.fifo_market_replay import (
    FIFOReplayEngine,
    COMMISSION_TICKS,
)

# --------- Config -----------------------------------------------------------
PRED_DIR = LVL3_ROOT / "output" / "hc488_dlinear_quantile_v1"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUT_DIR = LVL3_ROOT / "output" / "quantile_width_gate_fifo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DLinear quantile model inference settings
Q_WINDOW = 500
Q_OOT_STRIDE = 5

# FIFO settings per HC #428 R2
TP_TICKS_WIDE = 1000.0   # Wide so it never fires
SL_TICKS_WIDE = 1000.0   # Wide so it never fires
MARKET_EXIT_SPREAD = 1.0  # Spread crossing on market exit
ANNUAL_TRADING_DAYS = 252.0

# HC #428 deploy gates
GATE_NET = 0.10
GATE_SHARPE = 0.3
GATE_PDAYS_FRAC = 0.6875  # 11/16 scaled to any day count
GATE_REGIME_IMB = 0.50
GATE_DAYCONC = 0.70

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "fifo_replay.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("qwg_fifo")


# --------- Helpers ----------------------------------------------------------
def sharpe_per_day(day_means: np.ndarray) -> float:
    d = day_means[np.isfinite(day_means)]
    if d.size < 2:
        return float("nan")
    mu, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return mu / sd * math.sqrt(ANNUAL_TRADING_DAYS) if sd > 1e-12 else float("nan")


def sortino_per_day(day_means: np.ndarray) -> float:
    d = day_means[np.isfinite(day_means)]
    if d.size < 2:
        return float("nan")
    mu = float(np.mean(d))
    neg = d[d < 0]
    if neg.size == 0:
        return float("inf") if mu > 0 else float("nan")
    dd = float(np.sqrt(np.mean(neg**2)))
    return mu / dd * math.sqrt(ANNUAL_TRADING_DAYS) if dd > 1e-12 else float("nan")


def load_fold_predictions(fold_path: Path) -> dict:
    d = np.load(fold_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def select_width_gated_signals(
    fold_data: dict,
    horizon: str,
    conf_pct: float,
    width_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select short signals gated by confidence + width.

    Returns: (selected_indices, p50_values, width_values) — indices into the fold array.
    """
    p50 = fold_data[f"P50_{horizon}"]
    p10 = fold_data[f"P10_{horizon}"]
    p90 = fold_data[f"P90_{horizon}"]
    width = p90 - p10

    valid = np.isfinite(p50) & np.isfinite(width) & (width > 0)

    # Top conf_pct% most negative P50 = strongest short signals
    conf_thr = np.percentile(p50[valid], conf_pct)
    short_mask = valid & (p50 <= conf_thr)

    # Narrow width gate: bottom width_pct% of width among short signals
    w_short = width[short_mask]
    if len(w_short) < 10:
        return np.array([], dtype=np.int64), np.array([]), np.array([])

    width_thr = np.percentile(w_short, width_pct)
    final_mask = short_mask & (width <= width_thr)

    indices = np.where(final_mask)[0]
    return indices, p50[indices], width[indices]


def map_pred_idx_to_mbo_ts(date_str: str, pred_indices: np.ndarray) -> np.ndarray:
    """Map quantile prediction indices to nanosecond timestamps."""
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)

    # Each prediction i -> MBO event at i * OOT_STRIDE + WINDOW - 1
    event_idx = np.minimum(pred_indices * Q_OOT_STRIDE + Q_WINDOW - 1, n_events - 1)
    return ts_events[event_idx]


def run_one_day(
    date_str: str,
    pred_indices: np.ndarray,
    p50_vals: np.ndarray,
    horizon_sec: float,
) -> dict:
    """Run FIFO replay for one day. Returns summary dict."""
    ts_ns = map_pred_idx_to_mbo_ts(date_str, pred_indices)
    if ts_ns is None:
        return {"date": date_str, "error": "missing_mbo", "n_signaled": len(pred_indices)}

    signals = [
        {"ts_ns": int(t), "direction": "short", "strength": float(-p)}
        for t, p in zip(ts_ns, p50_vals)
    ]

    cancel_ns = int(round(horizon_sec * 1e9))
    max_hold_ns = int(round(1.5 * horizon_sec * 1e9))

    try:
        engine = FIFOReplayEngine(
            date=date_str,
            instrument_id=None,
            cancel_after_ns=cancel_ns,
            max_hold_ns=max_hold_ns,
            max_reprices=0,
            reprice_after_ns=int(1e9),
        )
    except Exception as e:
        return {"date": date_str, "error": f"engine_init: {e}",
                "n_signaled": len(pred_indices)}

    trades = engine.simulate(
        signals,
        tp_ticks=TP_TICKS_WIDE,
        sl_ticks=SL_TICKS_WIDE,
        order_type="limit",
        order_management="realtime_sl",
    )

    # Post-correct: market exit spread crossing
    # TradeResult is a dataclass with attrs: pnl_ticks_net, exit_reason,
    # entry_ts_ns, queue_wait_ns, etc.
    fills = []
    for tr in trades:
        net = tr.pnl_ticks_net  # already includes commission
        exit_reason = tr.exit_reason
        if exit_reason in ("max_hold", "eod"):
            net -= MARKET_EXIT_SPREAD
        fills.append({
            "date": date_str,
            "ts_ns": tr.entry_ts_ns,
            "net_ticks": net,
            "exit_reason": exit_reason,
            "time_to_fill_ms": tr.queue_wait_ns / 1e6,
        })

    n_filled = len(fills)
    fill_rate = n_filled / len(pred_indices) if len(pred_indices) > 0 else 0.0
    net_arr = np.array([f["net_ticks"] for f in fills]) if fills else np.array([0.0])
    mean_net = float(np.mean(net_arr)) if n_filled > 0 else float("nan")
    wr = float(np.mean(net_arr > 0)) if n_filled > 0 else float("nan")

    return {
        "date": date_str,
        "n_signaled": len(pred_indices),
        "n_filled": n_filled,
        "fill_rate": fill_rate,
        "mean_net_ticks": mean_net,
        "sum_net_ticks": float(np.sum(net_arr)),
        "wr": wr,
        "fills": fills,
    }


def deploy_verdict(day_results: List[dict]) -> dict:
    """HC #428 deploy gates."""
    valid = [d for d in day_results if d.get("n_filled", 0) > 0]
    if not valid:
        return {"pass": False, "reason": "no_fills"}

    day_nets = np.array([d["mean_net_ticks"] for d in valid])
    n_days = len(day_nets)
    avg_net = float(np.mean(day_nets))
    s = sharpe_per_day(day_nets)
    so = sortino_per_day(day_nets)
    pdays = int(np.sum(day_nets > 0))
    pdays_req = max(1, int(n_days * GATE_PDAYS_FRAC))

    total_ticks = sum(d.get("sum_net_ticks", 0) for d in valid)
    total_fills = sum(d.get("n_filled", 0) for d in valid)
    pool_avg = total_ticks / total_fills if total_fills > 0 else 0.0

    # Day concentration
    abs_nets = np.abs(day_nets)
    dayconc = float(np.max(abs_nets) / np.sum(abs_nets)) if abs_nets.sum() > 0 else 1.0

    # Fill rate
    total_signaled = sum(d.get("n_signaled", 0) for d in day_results)
    total_filled = sum(d.get("n_filled", 0) for d in day_results)
    fill_rate = total_filled / total_signaled if total_signaled > 0 else 0.0

    gates = {
        "net_t > 0.10": pool_avg > GATE_NET,
        f"sharpe > {GATE_SHARPE}": s > GATE_SHARPE,
        f"pdays >= {pdays_req}/{n_days}": pdays >= pdays_req,
        f"dayconc <= {GATE_DAYCONC}": dayconc <= GATE_DAYCONC,
    }
    passed = all(gates.values())

    return {
        "pass": passed,
        "pool_avg_net_t": pool_avg,
        "sharpe": s,
        "sortino": so,
        "profitable_days": f"{pdays}/{n_days}",
        "day_concentration": dayconc,
        "fill_rate": fill_rate,
        "total_fills": total_fills,
        "total_signaled": total_signaled,
        "gates": gates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=str, default="5s", help="Prediction horizon")
    parser.add_argument("--conf-pct", type=float, default=10.0,
                        help="Top N%% confidence (by P50 magnitude)")
    parser.add_argument("--width-pct", type=float, default=20.0,
                        help="Bottom N%% width (narrow gate)")
    parser.add_argument("--max-folds", type=int, default=999)
    args = parser.parse_args()

    horizon = args.horizon
    horizon_sec = float(horizon.replace("s", ""))
    h_idx = {"1s": 0, "5s": 1, "10s": 2}[horizon]

    log.info(f"=== FIFO Replay: short@{horizon}, top-{args.conf_pct}% conf, "
             f"narrow-{args.width_pct}% width ===")

    folds = sorted(PRED_DIR.glob("fold_*_preds.npz"))[:args.max_folds]
    log.info(f"Found {len(folds)} prediction folds")

    all_day_results = []

    for fi, fp in enumerate(folds):
        fold_data = load_fold_predictions(fp)
        date_str = str(fold_data["date"])

        indices, p50_vals, widths = select_width_gated_signals(
            fold_data, horizon, args.conf_pct, args.width_pct,
        )

        if len(indices) == 0:
            log.info(f"  fold {fi+1}: {date_str} — no signals selected, skipping")
            continue

        log.info(f"  fold {fi+1}: {date_str} — {len(indices):,} signals selected")

        t0 = time.time()
        result = run_one_day(date_str, indices, p50_vals, horizon_sec)
        elapsed = time.time() - t0

        if "error" in result:
            log.warning(f"    ERROR: {result['error']}")
            continue

        log.info(f"    filled {result['n_filled']:,}/{result['n_signaled']:,} "
                 f"({result['fill_rate']:.1%}) | net {result['mean_net_ticks']:+.4f} t/trade "
                 f"| WR {result['wr']:.1%} | {elapsed:.1f}s")
        all_day_results.append(result)

    if not all_day_results:
        log.error("No days with fills — cannot produce verdict.")
        return

    # Deploy verdict
    verdict = deploy_verdict(all_day_results)
    log.info("\n" + "=" * 70)
    log.info("DEPLOY VERDICT")
    log.info("=" * 70)
    for k, v in verdict.items():
        if k == "gates":
            for gk, gv in v.items():
                status = "PASS" if gv else "FAIL"
                log.info(f"  [{status}] {gk}")
        else:
            log.info(f"  {k}: {v}")
    log.info(f"\n  FINAL: {'PASS ✓' if verdict['pass'] else 'REJECT ✗'}")

    # Save results
    day_df = pd.DataFrame([{
        "date": d["date"],
        "n_signaled": d["n_signaled"],
        "n_filled": d["n_filled"],
        "fill_rate": d["fill_rate"],
        "mean_net_ticks": d["mean_net_ticks"],
        "wr": d["wr"],
    } for d in all_day_results])
    day_df.to_csv(OUT_DIR / "per_day_fifo.csv", index=False)

    # All fills
    all_fills = []
    for d in all_day_results:
        all_fills.extend(d.get("fills", []))
    if all_fills:
        fills_df = pd.DataFrame(all_fills)
        fills_df.to_csv(OUT_DIR / "per_trade_diagnostics.csv", index=False)

    # Summary
    with open(OUT_DIR / "verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    log.info(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
