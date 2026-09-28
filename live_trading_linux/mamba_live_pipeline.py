#!/usr/bin/env python3
"""
mamba_live_pipeline.py — Complete Mamba v7 live trading pipeline.

Wires together:
    1. MBO event stream (from recorder or live feed)
    2. StreamingFeaturesSmartV3 (25 features, exact match to training)
    3. MambaInferenceEngine (CPU inference, 270ms/prediction)
    4. Signal classification + execution decision

This is the core loop for Monday live/paper trading.

Usage:
    # Paper trading mode (replay from recorded MBO data):
    python mamba_live_pipeline.py --replay data/processed/mbo_events/20260313_mbo_events.npz

    # Live mode (reads from MBO recorder):
    python mamba_live_pipeline.py --live
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
from mamba_inference import MambaInferenceEngine


# ============================================================
# Constants
# ============================================================
TICK_SIZE = 0.25
POINT_VALUE = 50.0   # NQ
TICK_VALUE = 12.50
COMMISSION_PER_SIDE = 2.35  # AMP $4.70 RT / 2

# Default model paths
LVL3 = Path("/home/jupiter/Lvl3Quant")
DEFAULT_WEIGHTS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_best.pt"
DEFAULT_STATS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_feature_stats.npz"
DEFAULT_PREDS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/concat_oot_predictions.npz"


# ============================================================
# Signal Tracker — tracks model predictions over time
# ============================================================
class SignalTracker:
    """Tracks prediction history for signal-flip detection and execution decisions."""

    def __init__(self, history_size: int = 100):
        self.predictions: deque = deque(maxlen=history_size)
        self.directions: deque = deque(maxlen=history_size)
        self.timestamps: deque = deque(maxlen=history_size)
        self.confidences: deque = deque(maxlen=history_size)

    def add(self, pred: Dict, timestamp_ns: int = 0):
        self.predictions.append(pred)
        self.directions.append(pred["direction"])
        self.timestamps.append(timestamp_ns)
        self.confidences.append(pred["confidence_1s"])

    @property
    def current_direction(self) -> Optional[int]:
        if not self.directions:
            return None
        return self.directions[-1]

    @property
    def current_confidence(self) -> Optional[float]:
        if not self.confidences:
            return None
        return self.confidences[-1]

    @property
    def current_tier(self) -> Optional[str]:
        if not self.predictions:
            return None
        return self.predictions[-1].get("tier")

    def has_signal_flip(self) -> bool:
        """Check if direction just flipped from previous prediction."""
        if len(self.directions) < 2:
            return False
        return self.directions[-1] != self.directions[-2]

    def consecutive_same_direction(self) -> int:
        """Count how many consecutive predictions in same direction."""
        if not self.directions:
            return 0
        current = self.directions[-1]
        count = 0
        for d in reversed(self.directions):
            if d == current:
                count += 1
            else:
                break
        return count


# ============================================================
# Position Manager — tracks current position and P&L
# ============================================================
class PositionManager:
    """Manages position, P&L, and trade logging."""

    def __init__(self, max_position: int = 1):
        self.position: int = 0  # +1 = long, -1 = short, 0 = flat
        self.entry_price: float = 0.0
        self.entry_time_ns: int = 0
        self.max_position = max_position

        # P&L tracking
        self.realized_pnl: float = 0.0
        self.commission_paid: float = 0.0
        self.n_trades: int = 0
        self.n_winners: int = 0
        self.n_losers: int = 0
        self.total_gross: float = 0.0
        self.trades: List[Dict] = []

    @property
    def is_flat(self) -> bool:
        return self.position == 0

    @property
    def is_long(self) -> bool:
        return self.position > 0

    @property
    def is_short(self) -> bool:
        return self.position < 0

    def enter(self, direction: int, price: float, timestamp_ns: int,
              tier: str = None, confidence: float = 0.0):
        """Enter a position. direction: +1 for long, -1 for short."""
        if not self.is_flat:
            return  # Already in position

        self.position = direction
        self.entry_price = price
        self.entry_time_ns = timestamp_ns
        self.commission_paid += COMMISSION_PER_SIDE

        trade = {
            "type": "ENTRY",
            "direction": "LONG" if direction > 0 else "SHORT",
            "price": price,
            "time_ns": timestamp_ns,
            "tier": tier,
            "confidence": confidence,
        }
        self.trades.append(trade)

    def exit(self, price: float, timestamp_ns: int, reason: str = "signal_flip"):
        """Exit current position."""
        if self.is_flat:
            return None

        # Calculate P&L
        if self.is_long:
            gross_pnl = (price - self.entry_price) * POINT_VALUE
        else:
            gross_pnl = (self.entry_price - price) * POINT_VALUE

        commission = COMMISSION_PER_SIDE  # Exit side
        net_pnl = gross_pnl - COMMISSION_PER_SIDE * 2  # Both entry + exit commission

        hold_time_ns = timestamp_ns - self.entry_time_ns
        hold_time_s = hold_time_ns / 1e9 if hold_time_ns > 0 else 0

        self.realized_pnl += net_pnl
        self.commission_paid += COMMISSION_PER_SIDE
        self.total_gross += gross_pnl
        self.n_trades += 1
        if net_pnl > 0:
            self.n_winners += 1
        else:
            self.n_losers += 1

        trade = {
            "type": "EXIT",
            "reason": reason,
            "direction": "LONG" if self.position > 0 else "SHORT",
            "entry_price": self.entry_price,
            "exit_price": price,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "hold_time_s": round(hold_time_s, 2),
            "time_ns": timestamp_ns,
        }
        self.trades.append(trade)

        self.position = 0
        self.entry_price = 0.0
        self.entry_time_ns = 0

        return trade

    def summary(self) -> Dict:
        """Return trading summary."""
        win_rate = self.n_winners / max(self.n_trades, 1) * 100
        avg_pnl = self.realized_pnl / max(self.n_trades, 1)
        pf = self.total_gross / max(abs(self.total_gross - self.realized_pnl - self.commission_paid), 0.01)

        return {
            "n_trades": self.n_trades,
            "win_rate": round(win_rate, 1),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_commission": round(self.commission_paid, 2),
            "total_gross": round(self.total_gross, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
        }


# ============================================================
# Live Pipeline
# ============================================================
class MambaLivePipeline:
    """
    Complete Mamba v7 live trading pipeline.

    Event flow:
        Raw MBO event -> StreamingFeaturesSmartV3 (25 features)
                      -> MambaInferenceEngine (accumulate window, predict at stride)
                      -> Signal classification (confidence tier)
                      -> Execution decision (enter/exit/hold)
    """

    def __init__(
        self,
        weights_path: str = None,
        stats_path: str = None,
        preds_path: str = None,
        window_size: int = 1000,
        stride: int = 500,
        min_confidence_tier: str = "Top1%",
        entry_on_flip: bool = True,
        verbose: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride
        self.min_confidence_tier = min_confidence_tier
        self.entry_on_flip = entry_on_flip
        self.verbose = verbose

        # Tier ordering for comparison
        self._tier_order = {"Top5%": 1, "Top1%": 2, "Top0.5%": 3, "Top0.1%": 4}

        # Initialize components
        self.features = StreamingFeaturesSmartV3()
        # Auto-detect CUDA for GPU inference (10-50x faster than CPU)
        import torch as _torch
        _infer_device = "cuda" if _torch.cuda.is_available() else "cpu"

        self.engine = MambaInferenceEngine(
            weights_path=str(weights_path or DEFAULT_WEIGHTS),
            stats_path=str(stats_path or DEFAULT_STATS),
            window_size=window_size,
            stride=stride,
            device=_infer_device,
        )

        # Calibrate confidence thresholds
        preds = preds_path or DEFAULT_PREDS
        if Path(preds).exists():
            self.engine.calibrate_thresholds(str(preds))

        self.signal = SignalTracker()
        self.position = PositionManager()

        # Counters
        self.n_events = 0
        self.n_predictions = 0
        self.n_signals = 0  # Predictions meeting confidence threshold
        self.last_mid_price = 0.0

    def _tier_meets_minimum(self, tier: Optional[str]) -> bool:
        """Check if prediction tier meets minimum confidence threshold."""
        if tier is None:
            return False
        return self._tier_order.get(tier, 0) >= self._tier_order.get(self.min_confidence_tier, 0)

    def process_event(
        self,
        time_delta_log: float,
        event_type_id: int,
        side_id: int,
        price_rel_ticks: float,
        qty_log: float,
        spread_ticks: float,
        timestamp_ns: int = 0,
        mid_price: float = 0.0,
    ) -> Optional[Dict]:
        """
        Process one raw MBO event through the full pipeline.

        Returns prediction dict if a prediction was made, else None.
        """
        self.n_events += 1
        if mid_price > 0:
            self.last_mid_price = mid_price

        # Step 1: Compute streaming features
        feat_vec = self.features.update(
            time_delta_log=time_delta_log,
            event_type_id=event_type_id,
            side_id=side_id,
            price_rel_ticks=price_rel_ticks,
            qty_log=qty_log,
            spread_ticks=spread_ticks,
        )

        # Step 2: Feed to inference engine
        pred_result = self.engine.add_event(feat_vec)

        if pred_result is None:
            return None

        # Step 3: We have a prediction!
        self.n_predictions += 1
        self.signal.add(pred_result, timestamp_ns)

        # Step 4: Execution decision
        action = self._make_decision(pred_result, timestamp_ns)
        pred_result["action"] = action
        pred_result["position"] = self.position.position
        pred_result["realized_pnl"] = self.position.realized_pnl
        pred_result["n_trades"] = self.position.n_trades

        if self.verbose and action != "HOLD":
            dir_str = "LONG" if pred_result["direction"] == 1 else "SHORT"
            print(f"  [{self.n_events:,}] {action} | {dir_str} | "
                  f"1s={pred_result['pred_1s']:.4f} | "
                  f"conf={pred_result['confidence_1s']:.4f} | "
                  f"tier={pred_result.get('tier', 'None')} | "
                  f"P&L=${self.position.realized_pnl:.2f}")

        return pred_result

    def _make_decision(self, pred: Dict, timestamp_ns: int) -> str:
        """
        Make execution decision based on current prediction and position.

        Strategy: Top1%+ confidence entry on signal, signal-flip exit.
        """
        tier = pred.get("tier")
        direction = pred["direction"]
        meets_threshold = self._tier_meets_minimum(tier)

        # --- EXIT LOGIC ---
        if not self.position.is_flat:
            # Exit on signal flip (regardless of confidence)
            if self.signal.has_signal_flip():
                trade = self.position.exit(
                    self.last_mid_price, timestamp_ns, reason="signal_flip"
                )
                # After exit, check if we should immediately enter opposite
                if meets_threshold and self.entry_on_flip:
                    self.position.enter(
                        direction, self.last_mid_price, timestamp_ns,
                        tier=tier, confidence=pred["confidence_1s"]
                    )
                    return "FLIP"
                return "EXIT"

            # If same direction as position, hold
            if (self.position.is_long and direction > 0) or \
               (self.position.is_short and direction < 0):
                return "HOLD"

            # Direction changed but not signal flip? Shouldn't happen, but exit
            trade = self.position.exit(
                self.last_mid_price, timestamp_ns, reason="direction_change"
            )
            return "EXIT"

        # --- ENTRY LOGIC ---
        if self.position.is_flat and meets_threshold:
            self.n_signals += 1
            # Only enter if features are warm
            if not self.features.is_warm():
                return "WARM_UP"

            self.position.enter(
                direction, self.last_mid_price, timestamp_ns,
                tier=tier, confidence=pred["confidence_1s"]
            )
            return "ENTRY"

        return "HOLD"

    def summary(self) -> Dict:
        """Return pipeline summary."""
        pos = self.position.summary()
        return {
            "n_events": self.n_events,
            "n_predictions": self.n_predictions,
            "n_signals": self.n_signals,
            "features_warm": self.features.is_warm(),
            **pos,
        }


# ============================================================
# Replay mode — run pipeline on recorded MBO data
# ============================================================
def run_replay(npz_path: str, pipeline: MambaLivePipeline, max_events: int = None):
    """Replay recorded MBO events through the pipeline."""
    print(f"\nReplaying {npz_path}...")
    data = np.load(npz_path)
    events = data["events"]  # (N, 6) raw features
    timestamps = data.get("timestamps", np.zeros(len(events), dtype=np.int64))

    N = len(events) if max_events is None else min(len(events), max_events)
    print(f"  {N:,} events to process")

    t0 = time.time()
    predictions = []

    for i in range(N):
        ev = events[i]
        ts = int(timestamps[i]) if i < len(timestamps) else 0

        result = pipeline.process_event(
            time_delta_log=float(ev[0]),
            event_type_id=int(ev[1]),
            side_id=int(ev[2]),
            price_rel_ticks=float(ev[3]),
            qty_log=float(ev[4]),
            spread_ticks=float(ev[5]),
            timestamp_ns=ts,
            mid_price=0.0,  # Not available in raw events
        )

        if result is not None:
            predictions.append(result)

        # Progress every 50K events
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:,}/{N:,}] {rate:.0f} events/s | "
                  f"preds={len(predictions)} | P&L=${pipeline.position.realized_pnl:.2f}")

    elapsed = time.time() - t0
    rate = N / elapsed

    print(f"\n{'='*60}")
    print(f"REPLAY COMPLETE: {N:,} events in {elapsed:.1f}s ({rate:.0f} events/s)")
    summary = pipeline.summary()
    print(f"  Predictions: {summary['n_predictions']}")
    print(f"  Signals (>={pipeline.min_confidence_tier}): {summary['n_signals']}")
    print(f"  Trades: {summary['n_trades']}")
    print(f"  Win rate: {summary['win_rate']}%")
    print(f"  Realized P&L: ${summary['realized_pnl']}")
    print(f"  Commission: ${summary['total_commission']}")
    print(f"  Gross P&L: ${summary['total_gross']}")
    print(f"{'='*60}")

    return predictions, summary


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Mamba v7 Live Trading Pipeline")
    parser.add_argument("--replay", type=str, help="Replay from NPZ file (raw MBO events)")
    parser.add_argument("--replay-smart", type=str, help="Replay from smart_v3 NPZ (skip feature computation)")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--weights", type=str, default=None, help="Model weights path")
    parser.add_argument("--stats", type=str, default=None, help="Feature stats path")
    parser.add_argument("--preds", type=str, default=None, help="Historical predictions for calibration")
    parser.add_argument("--min-tier", type=str, default="Top1%",
                        choices=["Top5%", "Top1%", "Top0.5%", "Top0.1%"],
                        help="Minimum confidence tier for entry")
    parser.add_argument("--window", type=int, default=1000, help="Window size")
    parser.add_argument("--stride", type=int, default=500, help="Stride between predictions")
    parser.add_argument("--max-events", type=int, default=None, help="Max events to process")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    pipeline = MambaLivePipeline(
        weights_path=args.weights,
        stats_path=args.stats,
        preds_path=args.preds,
        window_size=args.window,
        stride=args.stride,
        min_confidence_tier=args.min_tier,
        verbose=args.verbose,
    )

    if args.replay:
        predictions, summary = run_replay(args.replay, pipeline, max_events=args.max_events)
    elif args.replay_smart:
        # Replay from smart_v3 data (features already computed)
        print(f"Replaying smart_v3 data: {args.replay_smart}")
        data = np.load(args.replay_smart)
        features = data["events"]  # Already smart_v3 normalized (N, 25)
        N = len(features) if args.max_events is None else min(len(features), args.max_events)

        t0 = time.time()
        for i in range(N):
            result = pipeline.engine.add_event(features[i])
            if result is not None:
                pipeline.signal.add(result, 0)
                pipeline.n_predictions += 1

        elapsed = time.time() - t0
        print(f"  {N:,} events in {elapsed:.1f}s, {pipeline.n_predictions} predictions")
    elif args.live:
        print("Live mode not yet implemented. Use --replay for testing.")
    else:
        print("Use --replay <file>, --replay-smart <file>, or --live")
        parser.print_help()


if __name__ == "__main__":
    main()
