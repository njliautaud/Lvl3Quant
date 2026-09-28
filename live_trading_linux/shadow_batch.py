#!/usr/bin/env python3
"""
shadow_batch.py — Vectorized shadow replay: batch features + batch predict.

Same contract as shadow_replay.py but 100-1000x faster because we use
NumPy-vectorized compute_derived() and a single LGBM batch predict. Used
for OOS validation when we don't need event-by-event streaming semantics.

Usage:
  python -m live_trading_linux.shadow_batch \
      --npz /home/jupiter/Lvl3Quant/data/processed/mbo_events/20260313_mbo_events.npz \
      --model /home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/labels_1s_lgbm.pkl \
      --calibration /home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/labels_1s_calibration.json \
      --tier top10 \
      --horizon-events 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

from live_trading_linux.lgbm_inference      import LGBMInference, TIER_ORDER
from live_trading_linux.streaming_features import _compute_derived_reference as compute_derived


log = logging.getLogger("shadow_batch")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


TICK_SIZE_PT = 0.25
POINT_USD    = 50.0


def simulate_trades(prices: np.ndarray, preds: np.ndarray, tier_ranks: np.ndarray,
                    tier_min_rank: int, horizon_events: int,
                    slip_ticks: int, fee: float):
    """Vectorized synthetic fills.

    For each entry event i where tier_ranks[i] >= tier_min_rank:
        side = sign(preds[i])
        entry_p = prices[i] +/- slip
        exit_i  = min(i + horizon_events, N-1)
        exit_p  = prices[exit_i] -/+ slip
        pnl_usd = (exit_p - entry_p) * side * POINT_USD - 2*fee
    """
    N = len(prices)
    mask = tier_ranks >= tier_min_rank
    mask &= preds != 0
    idx  = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    side   = np.sign(preds[idx]).astype(np.int8)
    slip_pt = slip_ticks * TICK_SIZE_PT
    entry_p = prices[idx] + slip_pt * side
    exit_i  = np.minimum(idx + horizon_events, N - 1)
    exit_p  = prices[exit_i] - slip_pt * side
    gross_pts = (exit_p - entry_p) * side
    gross_usd = gross_pts * POINT_USD
    net_usd   = gross_usd - 2 * fee
    return {
        "entry_i": idx, "exit_i": exit_i, "side": side,
        "entry_p": entry_p, "exit_p": exit_p,
        "pts": gross_pts, "gross": gross_usd, "net": net_usd,
    }


def aggregate(res: dict) -> dict:
    net = res["net"]; pts = res["pts"]; side = res["side"]
    wins = net > 0
    mu, sd = float(net.mean()), float(net.std())
    sharpe = mu / sd * np.sqrt(len(net)) if sd > 0 else 0.0
    downside = net[net < 0]
    sortino = mu / downside.std() * np.sqrt(len(net)) if len(downside) > 0 and downside.std() > 0 else 0.0
    return {
        "n_trades": int(len(net)),
        "hit_rate": float(wins.mean()),
        "mean_net_usd": mu, "std_net_usd": sd,
        "total_net_usd": float(net.sum()),
        "total_gross_usd": float(pts.sum() * POINT_USD),
        "sharpe_pertrade": sharpe,
        "sortino_pertrade": sortino,
        "avg_win_usd": float(net[wins].mean()) if wins.any() else 0.0,
        "avg_loss_usd": float(net[~wins].mean()) if (~wins).any() else 0.0,
        "long_pct": float((side > 0).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--tier", default="top10")
    ap.add_argument("--slip-ticks", type=int, default=1)
    ap.add_argument("--fee", type=float, default=0.5)
    ap.add_argument("--horizon-events", type=int, default=50)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    d = np.load(args.npz)
    ev = d["events"].astype(np.float64)
    if args.max_events: ev = ev[:args.max_events]
    log.info("Loaded %s (%d events) in %.1fs", Path(args.npz).name, len(ev), time.time() - t0)

    t0 = time.time()
    feats = compute_derived(ev)          # (N, 21)
    log.info("compute_derived: %.1fs  shape=%s", time.time() - t0, feats.shape)

    t0 = time.time()
    inf = LGBMInference(model_path=args.model, calibration_path=args.calibration)
    preds = inf.predict_batch(feats).astype(np.float32)
    log.info("batch predict: %.1fs", time.time() - t0)

    # Compute tier per event (vectorized)
    absp = np.abs(preds)
    ranks = np.zeros(len(preds), dtype=np.int8)
    # tier 'all'=0 always true; LGBMInference stores flat thresholds
    if inf.thresh_p50 > 0:
        ranks = np.where(absp >= inf.thresh_p50, 1, ranks)
    if inf.thresh_p75 > 0:
        ranks = np.where(absp >= inf.thresh_p75, 2, ranks)
    if inf.thresh_p90 > 0:
        ranks = np.where(absp >= inf.thresh_p90, 3, ranks)
    log.info("tier thresholds p50=%.4f p75=%.4f p90=%.4f", inf.thresh_p50, inf.thresh_p75, inf.thresh_p90)

    # pull price column (index 2 in events schema)
    prices = ev[:, 2].astype(np.float64)

    tier_min_rank = TIER_ORDER[args.tier]
    sim = simulate_trades(
        prices, preds, ranks,
        tier_min_rank  = tier_min_rank,
        horizon_events = args.horizon_events,
        slip_ticks     = args.slip_ticks,
        fee            = args.fee,
    )
    if sim is None:
        print(json.dumps({"n_trades": 0, "message": "No entries passed gate."}, indent=2))
        return 0

    res = aggregate(sim)
    res.update({
        "npz":       Path(args.npz).name,
        "model":     Path(args.model).name,
        "tier":      args.tier,
        "slip_ticks": args.slip_ticks,
        "fee":        args.fee,
        "horizon_events": args.horizon_events,
        "n_events":  int(len(ev)),
        "trade_rate_pct": float(100 * res["n_trades"] / len(ev)),
    })

    # Per-tier breakdown
    per_tier = {}
    for tname, rnk in [("all", 0), ("top50", 1), ("top25", 2), ("top10", 3)]:
        s = simulate_trades(prices, preds, ranks, rnk, args.horizon_events, args.slip_ticks, args.fee)
        if s is not None:
            per_tier[tname] = aggregate(s) | {"trade_rate_pct": 100 * s["net"].size / len(ev)}
    res["per_tier"] = per_tier

    print(json.dumps(res, indent=2, default=float))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
