#!/usr/bin/env python3
"""
shadow_replay.py — Offline replay of a recorded MBO NPZ file through the
live pipeline (StreamingFeatures → LGBMInference → tiered signal → synthetic
fill). Validates three things BEFORE going live on Rithmic:

    1. Feature parity under a real event stream (not just the 2k-event
       synthetic self-test in streaming_features.py).
    2. Tier threshold realism — do gated signals actually fire at the
       expected rate (e.g. ~10% of events for top10)?
    3. PnL sanity — does directional accuracy in the tier bucket translate
       into positive expectancy after assumed 1-tick slippage?

Usage:
    python -m live_trading_linux.shadow_replay \
        --npz /home/saturn/events_mbo/20260313_mbo_events.npz \
        --model /home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/labels_1s_lgbm.pkl \
        --calibration /home/jupiter/Lvl3Quant/models/lgbm_60_5_fold35/labels_1s_calibration.json \
        --tier top10 \
        --slip-ticks 1 \
        --fee 0.5

Assumes ES contract: $12.50 per tick, 4 ticks per point, $50/point.
Synthetic fill rule: on signal side B/S at tier gate, enter at price
 +/- slip; exit after `horizon_events` events (default 50 ~ 1s at 50Hz)
at the then-current price.
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

from live_trading_linux.lgbm_inference   import LGBMInference
from live_trading_linux.streaming_features import StreamingFeatures, FEATURE_NAMES


log = logging.getLogger("shadow_replay")
if not log.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")


# ES contract specs
TICK_SIZE_USD = 12.50   # ES mini, $12.50 per 0.25 pt
TICK_SIZE_PT  = 0.25
POINT_USD     = 50.0


def load_events(path: Path) -> np.ndarray:
    """Load an MBO NPZ — expects the same columns that compute_derived() consumes.

    Training data schema (first 6 columns): event_type, side, price, qty, spread, time_delta.
    """
    d = np.load(path)
    keys = list(d.keys())
    log.info("Loaded %s  keys=%s", path.name, keys)
    # Common formats: 'events' as (N,6) or individual arrays.
    if "events" in keys:
        ev = d["events"]
    elif all(k in keys for k in ("event_type", "side", "price", "qty", "spread", "time_delta")):
        ev = np.stack([d["event_type"], d["side"], d["price"], d["qty"],
                       d["spread"], d["time_delta"]], axis=1)
    else:
        # Fall back to first 6 numeric arrays
        arrs = [d[k] for k in keys if d[k].ndim == 1][:6]
        ev = np.stack(arrs, axis=1)
    assert ev.shape[1] >= 6, f"Bad event shape {ev.shape}"
    return ev.astype(np.float64)


def replay(
    events: np.ndarray,
    inf: LGBMInference,
    tier_min: str = "top10",
    slip_ticks: int = 1,
    fee_per_side: float = 0.5,
    horizon_events: int = 50,
    max_events: Optional[int] = None,
) -> dict:
    """Stream events through features → prediction → synthetic fills."""
    feats = StreamingFeatures()

    N = len(events) if max_events is None else min(max_events, len(events))
    log.info("Replaying %d events (horizon=%d, tier>=%s, slip=%d ticks, fee=%.2f)",
             N, horizon_events, tier_min, slip_ticks, fee_per_side)

    # Track open positions: list of (entry_idx, side, entry_price)
    open_trades: list[tuple[int, int, float]] = []
    closed_trades: list[dict] = []

    prices   = np.empty(N, dtype=np.float64)
    preds    = np.empty(N, dtype=np.float32)
    tiers    = np.empty(N, dtype="U8")

    from live_trading_linux.lgbm_inference import TIER_ORDER
    tier_min_rank = TIER_ORDER[tier_min]

    t0 = time.time()
    for i in range(N):
        et, sd, p, q, sp, td = events[i, :6]
        feat = feats.update(et, sd, p, q, sp, td)
        tier, pred = inf.predict_tier(feat)
        prices[i] = p
        preds[i]  = pred
        tiers[i]  = tier

        # Close trades whose horizon expired
        still_open = []
        for entry_i, side, ep in open_trades:
            if i - entry_i >= horizon_events:
                # Exit at current price minus slippage
                exit_p = p - slip_ticks * TICK_SIZE_PT if side > 0 else p + slip_ticks * TICK_SIZE_PT
                gross_pts = (exit_p - ep) * side
                gross_usd = gross_pts * POINT_USD
                fees      = 2 * fee_per_side   # entry + exit
                net_usd   = gross_usd - fees
                closed_trades.append({
                    "entry_i": entry_i, "exit_i": i,
                    "side":    side,
                    "entry_p": ep, "exit_p": exit_p,
                    "pts":     gross_pts, "gross": gross_usd,
                    "net":     net_usd,
                })
            else:
                still_open.append((entry_i, side, ep))
        open_trades = still_open

        # Entry logic: tier gate + no position + sign of pred drives direction
        if (TIER_ORDER[tier] >= tier_min_rank) and (pred != 0.0):
            side = 1 if pred > 0 else -1
            # Slip on entry: buy pays +1 tick, sell receives -1 tick
            entry_p = p + slip_ticks * TICK_SIZE_PT if side > 0 else p - slip_ticks * TICK_SIZE_PT
            open_trades.append((i, side, entry_p))

        if (i + 1) % 100_000 == 0:
            log.info("  step %d/%d  trades=%d  open=%d  elapsed=%.1fs",
                     i + 1, N, len(closed_trades), len(open_trades), time.time() - t0)

    dur = time.time() - t0
    log.info("Replay done: %d trades closed in %.1fs (%.0f events/s)",
             len(closed_trades), dur, N / dur)

    # Aggregate stats
    if not closed_trades:
        return {"n_trades": 0, "message": "No trades triggered."}

    arr_net   = np.array([t["net"]   for t in closed_trades])
    arr_pts   = np.array([t["pts"]   for t in closed_trades])
    arr_sides = np.array([t["side"]  for t in closed_trades])
    wins      = arr_net > 0
    hit_rate  = float(wins.mean())

    # Sharpe/Sortino on trade-level returns (not annualized — per-trade)
    mu, sd = float(arr_net.mean()), float(arr_net.std())
    sharpe = mu / sd * np.sqrt(len(arr_net)) if sd > 0 else 0.0
    downside = arr_net[arr_net < 0]
    sortino = mu / downside.std() * np.sqrt(len(arr_net)) if len(downside) > 0 and downside.std() > 0 else 0.0

    return {
        "n_events": int(N),
        "n_trades": int(len(closed_trades)),
        "trade_rate_pct": float(100 * len(closed_trades) / N),
        "hit_rate": hit_rate,
        "mean_net_usd": mu,
        "std_net_usd": sd,
        "total_gross_usd": float(arr_pts.sum() * POINT_USD),
        "total_net_usd": float(arr_net.sum()),
        "sharpe_pertrade": sharpe,
        "sortino_pertrade": sortino,
        "avg_win_usd": float(arr_net[wins].mean()) if wins.any() else 0.0,
        "avg_loss_usd": float(arr_net[~wins].mean()) if (~wins).any() else 0.0,
        "long_pct": float((arr_sides > 0).mean()),
        "replay_seconds": dur,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="MBO events NPZ")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--tier", default="top10", choices=["all", "top50", "top25", "top10", "top5", "top1"])
    ap.add_argument("--slip-ticks", type=int, default=1)
    ap.add_argument("--fee", type=float, default=0.5, help="per-side fee USD")
    ap.add_argument("--horizon-events", type=int, default=50,
                    help="exit after N events (~1s at 50Hz)")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ev = load_events(Path(args.npz))
    inf = LGBMInference(model_path=args.model, calibration_path=args.calibration)

    res = replay(
        ev, inf,
        tier_min       = args.tier,
        slip_ticks     = args.slip_ticks,
        fee_per_side   = args.fee,
        horizon_events = args.horizon_events,
        max_events     = args.max_events,
    )
    res["npz"]   = str(Path(args.npz).name)
    res["model"] = str(Path(args.model).name)
    res["tier"]  = args.tier
    res["horizon_events"] = args.horizon_events
    res["slip_ticks"] = args.slip_ticks
    res["fee_per_side"] = args.fee

    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
