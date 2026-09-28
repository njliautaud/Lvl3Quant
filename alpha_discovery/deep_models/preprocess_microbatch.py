"""
Microbatching Feature Engine for Mamba SSM

Converts raw MBO events (15M events/day × 6 features) into compressed
market state transitions (~150K steps/day × 20 features).

Raw event format:
  [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]

Event types: 0=add, 1=cancel, 2=modify, 3=trade, 4=fill

Output per microbatch (N=100 events):
  20 features capturing order flow, liquidity, intensity, and price dynamics.

Usage:
  python preprocess_microbatch.py                    # process all files
  python preprocess_microbatch.py --batch-size 100   # custom microbatch size
  python preprocess_microbatch.py --workers 8        # parallel processing
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Event type IDs
ADD, CANCEL, MODIFY, TRADE, FILL = 0, 1, 2, 3, 4

DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events"
)
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_microbatch"
)


def compute_microbatch_features(events: np.ndarray, batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw events into microbatched feature vectors.

    Args:
        events: (N, 6) raw MBO events
        batch_size: number of events per microbatch

    Returns:
        features: (M, 20) microbatch features where M = N // batch_size
        indices: (M,) index of last event in each microbatch (for label alignment)
    """
    n_events = len(events)
    n_batches = n_events // batch_size

    if n_batches == 0:
        return np.empty((0, 20), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Reshape into (n_batches, batch_size, 6)
    trimmed = events[:n_batches * batch_size]
    batches = trimmed.reshape(n_batches, batch_size, 6)

    # Extract columns: [time_delta_log, event_type_id, side_id, price_rel, qty_log, spread]
    time_delta = batches[:, :, 0]   # (M, B)
    event_type = batches[:, :, 1]   # (M, B)
    side       = batches[:, :, 2]   # (M, B)
    price_rel  = batches[:, :, 3]   # (M, B)
    qty_log    = batches[:, :, 4]   # (M, B)
    spread     = batches[:, :, 5]   # (M, B)

    # Convert qty from log to linear for volume calcs
    qty = np.exp(qty_log)

    features = np.zeros((n_batches, 20), dtype=np.float32)

    # === ORDER FLOW FEATURES ===

    # 1. Order Flow Imbalance (OFI): buy volume - sell volume
    #    side=1 is ask (sell side), side=0 is bid (buy side)
    buy_mask = (side < 0.5)  # bid side
    sell_mask = (side >= 0.5)  # ask side
    features[:, 0] = (qty * buy_mask).sum(axis=1) - (qty * sell_mask).sum(axis=1)

    # 2. Trade imbalance: buy trades / total trades
    trade_mask = (event_type >= 3)  # trade or fill
    buy_trade_mask = trade_mask & buy_mask
    n_trades = trade_mask.sum(axis=1).astype(np.float32)
    n_buy_trades = buy_trade_mask.sum(axis=1).astype(np.float32)
    features[:, 1] = np.where(n_trades > 0, n_buy_trades / n_trades, 0.5)

    # 3. Aggressor volume ratio: trade volume / total volume
    trade_vol = (qty * trade_mask).sum(axis=1)
    total_vol = qty.sum(axis=1)
    features[:, 2] = np.where(total_vol > 0, trade_vol / total_vol, 0.0)

    # === LIQUIDITY FEATURES ===

    # 4. Add/Cancel ratio: adds / (adds + cancels)
    add_mask = (event_type < 0.5)
    cancel_mask = ((event_type >= 0.5) & (event_type < 1.5))
    n_adds = add_mask.sum(axis=1).astype(np.float32)
    n_cancels = cancel_mask.sum(axis=1).astype(np.float32)
    features[:, 3] = np.where((n_adds + n_cancels) > 0, n_adds / (n_adds + n_cancels), 0.5)

    # 5. Cancel pressure: cancel volume / add volume
    add_vol = (qty * add_mask).sum(axis=1)
    cancel_vol = (qty * cancel_mask).sum(axis=1)
    features[:, 4] = np.where(add_vol > 0, cancel_vol / add_vol, 1.0)

    # 6. Book imbalance: bid add volume - ask add volume (net liquidity provision)
    bid_add_vol = (qty * add_mask * buy_mask).sum(axis=1)
    ask_add_vol = (qty * add_mask * sell_mask).sum(axis=1)
    features[:, 5] = bid_add_vol - ask_add_vol

    # === PRICE FEATURES ===

    # 7. Price velocity: last price - first price in microbatch
    features[:, 6] = price_rel[:, -1] - price_rel[:, 0]

    # 8. Price range: max - min price in microbatch
    features[:, 7] = price_rel.max(axis=1) - price_rel.min(axis=1)

    # 9. Volume-weighted price (microprice proxy)
    total_qty = qty.sum(axis=1)
    features[:, 8] = np.where(total_qty > 0,
                               (price_rel * qty).sum(axis=1) / total_qty,
                               price_rel.mean(axis=1))

    # 10. Trade-weighted price (where actual trades happen)
    trade_qty = (qty * trade_mask)
    trade_qty_sum = trade_qty.sum(axis=1)
    features[:, 9] = np.where(trade_qty_sum > 0,
                                (price_rel * trade_qty).sum(axis=1) / trade_qty_sum,
                                features[:, 8])

    # === SPREAD & LIQUIDITY GRADIENT ===

    # 11. Mean spread
    features[:, 10] = spread.mean(axis=1)

    # 12. Spread change: last - first
    features[:, 11] = spread[:, -1] - spread[:, 0]

    # 13. Spread volatility
    features[:, 12] = spread.std(axis=1)

    # === INTENSITY FEATURES ===

    # 14. Event intensity: 1 / mean(time_delta) = events per unit time
    mean_td = time_delta.mean(axis=1)
    features[:, 13] = np.where(mean_td > 0, 1.0 / (np.exp(mean_td)), 0.0)

    # 15. Trade intensity: trades per microbatch / batch_size
    features[:, 14] = n_trades / batch_size

    # 16. Real time span: sum of exp(time_delta) in seconds
    features[:, 15] = np.exp(time_delta).sum(axis=1)

    # === VOLUME FEATURES ===

    # 17. Total volume (log)
    features[:, 16] = np.log1p(total_vol)

    # 18. Volume concentration: max_qty / mean_qty (detects large orders)
    mean_qty = qty.mean(axis=1)
    max_qty = qty.max(axis=1)
    features[:, 17] = np.where(mean_qty > 0, max_qty / mean_qty, 1.0)

    # === EVENT TYPE DISTRIBUTION ===

    # 19. Event type entropy: diversity of event types
    for et in range(5):
        et_mask = ((event_type >= et - 0.5) & (event_type < et + 0.5))
        et_frac = et_mask.sum(axis=1).astype(np.float32) / batch_size
        # Contribution to entropy
        features[:, 18] -= np.where(et_frac > 0, et_frac * np.log(et_frac + 1e-10), 0.0)

    # 20. Signed volume momentum: sum of signed trade volumes
    #     positive = net buying, negative = net selling
    sign = np.where(buy_mask, 1.0, -1.0)
    features[:, 19] = (sign * qty * trade_mask).sum(axis=1)

    # Label alignment: use the last event index of each microbatch
    indices = np.arange(batch_size - 1, n_batches * batch_size, batch_size, dtype=np.int64)

    return features, indices


FEATURE_NAMES = [
    "ofi",                  # 0: Order Flow Imbalance
    "trade_imbalance",      # 1: Buy trades / total trades
    "aggressor_ratio",      # 2: Trade volume / total volume
    "add_cancel_ratio",     # 3: Adds / (adds + cancels)
    "cancel_pressure",      # 4: Cancel vol / add vol
    "book_imbalance",       # 5: Bid add vol - ask add vol
    "price_velocity",       # 6: Last price - first price
    "price_range",          # 7: Max - min price
    "vwap",                 # 8: Volume-weighted avg price
    "trade_price",          # 9: Trade-weighted avg price
    "spread_mean",          # 10: Mean spread
    "spread_change",        # 11: Spread change
    "spread_vol",           # 12: Spread volatility
    "event_intensity",      # 13: Events per unit time
    "trade_intensity",      # 14: Trades per microbatch
    "time_span",            # 15: Real time of microbatch (seconds)
    "total_volume_log",     # 16: Log total volume
    "volume_concentration", # 17: Max qty / mean qty
    "event_entropy",        # 18: Event type diversity
    "signed_volume",        # 19: Net signed trade volume
]


def process_one_file(args):
    """Process a single day's raw events into microbatched features."""
    npz_path, output_dir, batch_size = args
    fname = Path(npz_path).stem

    try:
        data = np.load(npz_path, allow_pickle=True)
        events = data["events"].astype(np.float32)

        features, indices = compute_microbatch_features(events, batch_size=batch_size)

        if len(features) == 0:
            return fname, 0, "empty"

        # Align labels to microbatch endpoints
        save_dict = {
            "features": features,
            "feature_names": np.array(FEATURE_NAMES),
            "batch_size": np.array(batch_size),
            "n_raw_events": np.array(len(events)),
        }

        for horizon in ["1s", "5s", "10s", "30s"]:
            key = f"labels_{horizon}"
            if key in data:
                labels = data[key].astype(np.float32)
                save_dict[key] = labels[indices]

        if "timestamps" in data:
            save_dict["timestamps"] = data["timestamps"][indices]

        out_path = Path(output_dir) / f"{fname}_microbatch.npz"
        np.savez_compressed(out_path, **save_dict)

        return fname, len(features), "ok"

    except Exception as e:
        return fname, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Microbatch MBO events for Mamba training")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Events per microbatch (default 100)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default 4)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No NPZ files in {data_dir}")
        sys.exit(1)

    logger.info(f"Processing {len(npz_files)} files → {output_dir}")
    logger.info(f"Microbatch size: {args.batch_size} events → 20 features")
    logger.info(f"Workers: {args.workers}")

    tasks = [(str(f), str(output_dir), args.batch_size) for f in npz_files]

    t0 = time.time()
    results = []

    if args.workers <= 1:
        for task in tasks:
            results.append(process_one_file(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one_file, t): t for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())

    elapsed = time.time() - t0

    # Summary
    ok = [r for r in results if r[2] == "ok"]
    failed = [r for r in results if r[2] not in ("ok", "empty")]
    total_steps = sum(r[1] for r in ok)

    logger.info(f"\n{'='*60}")
    logger.info(f"MICROBATCH PREPROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Files processed: {len(ok)} / {len(npz_files)}")
    logger.info(f"  Total microbatch steps: {total_steps:,}")
    logger.info(f"  Avg steps per day: {total_steps // max(len(ok), 1):,}")
    logger.info(f"  Features per step: 20")
    logger.info(f"  Microbatch size: {args.batch_size} events")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info(f"  Output: {output_dir}")

    if failed:
        logger.warning(f"  Failed: {len(failed)} files")
        for fname, _, err in failed:
            logger.warning(f"    {fname}: {err}")

    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
