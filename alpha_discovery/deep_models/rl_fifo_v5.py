#!/usr/bin/env python3
"""
RL Execution Agent v5 — FIFO Market Replay
=============================================
Trains a PPO agent on ACTUAL MBO event data with FIFO queue simulation.

Key improvements over v3.x (which mode-collapsed):
  1. FIFO book replay: reconstructs L2 book from MBO events, tracks queue position
  2. Multi-signal state: CNN-Mamba + book microstructure + position state
  3. Anti-collapse: entropy annealing, trade activity shaping, reward clipping
  4. Larger action space: entry method (limit/market/chase), TP/SL/hold adaptive
  5. Per-step rewards with Sortino shaping (not episodic which caused sparse gradients)
  6. More training data: use ALL available MBO days with matching predictions

Architecture:
  - Transformer-based policy (no LSTM state management issues)
  - State dim: ~48 (predictions + book features + position + trade history)
  - 5 action heads: gate(3) + entry_type(3) + tp(8) + sl(7) + hold(6)
  - PPO with high entropy coeff (0.05) annealing to 0.01

ES Futures constants (CANONICAL):
  - tick_value = $12.50
  - commission_rt = $4.70 = 0.376 ticks per side
  - spread: measured from actual book data

Usage:
  python rl_fifo_v5.py --epochs 500 --device cuda
  python rl_fifo_v5.py --eval --checkpoint output/rl_fifo_v5/best.pt
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
OUTPUT_DIR = LVL3_ROOT / "output" / "rl_fifo_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (CANONICAL ES futures) ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_PER_SIDE_TICKS = 0.376  # $4.70 / $12.50
COMMISSION_RT_TICKS = 2 * COMMISSION_PER_SIDE_TICKS  # 0.752 ticks

# MBO event encoding
ACTION_ADD = 0
ACTION_CANCEL = 1
ACTION_MODIFY = 2
ACTION_TRADE = 3
ACTION_FILL = 4
SIDE_BID = 0
SIDE_ASK = 1

# Action spaces
GATE_ACTIONS = ["SKIP", "ENTER_LONG", "ENTER_SHORT", "EXIT"]
ENTRY_TYPES = ["LIMIT_BBO", "LIMIT_AGGRESSIVE", "MARKET"]
TP_TICKS = np.array([2, 4, 6, 8, 10, 15, 20, 30], dtype=np.float32)
SL_TICKS = np.array([2, 3, 5, 8, 10, 15, 20], dtype=np.float32)
HOLD_SECS = np.array([5.0, 10.0, 30.0, 60.0, 120.0, 300.0], dtype=np.float32)

# Decision frequency: every 5000 events = ~10 prediction samples (~3 seconds)
# This gives ~2500 decision points per day (vs 25K before), making training feasible
DECISION_STRIDE = 5000

# ── Logging ──
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log = logging.getLogger("rl_fifo_v5")
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(OUTPUT_DIR / f"rl_fifo_v5_{_ts}.log"), mode="w")
_sh = logging.StreamHandler(sys.stdout)
for h in [_fh, _sh]:
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════
# SIMPLIFIED L2 BOOK FROM MBO EVENTS
# ═══════════════════════════════════════════════════════════════

class SimplifiedBook:
    """
    Tracks approximate L2 book state from MBO events using VECTORIZED batch updates.

    Instead of per-event Python loops, we process batches of events at once using numpy.
    This is ~100x faster than the per-event approach.
    """

    def __init__(self):
        self.spread_ticks = 1.0
        self.bid_depth = 0.0
        self.ask_depth = 0.0
        self.last_trade_side = 0
        self.trade_count = 0
        self.bid_queue_ahead = 0.0
        self.ask_queue_ahead = 0.0
        self.buy_vol_recent = 0.0
        self.sell_vol_recent = 0.0

    def update_batch(self, events: np.ndarray):
        """Update book state from a BATCH of MBO events (vectorized).
        events: (N, 6) array of [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
        """
        if len(events) == 0:
            return

        actions = events[:, 1].astype(np.int32)
        sides = events[:, 2].astype(np.int32)
        qty_logs = events[:, 4]
        spreads = events[:, 5]

        # Compute quantities (vectorized exp)
        qtys = np.where(qty_logs > 0, np.exp(np.clip(qty_logs, 0, 10)), 1.0)

        # Update spread from last non-zero spread
        nonzero_spread = spreads[spreads > 0]
        if len(nonzero_spread) > 0:
            self.spread_ticks = float(nonzero_spread[-1])

        # ADD events
        add_mask = actions == ACTION_ADD
        add_bid = add_mask & (sides == SIDE_BID)
        add_ask = add_mask & (sides == SIDE_ASK)
        self.bid_depth += qtys[add_bid].sum()
        self.ask_depth += qtys[add_ask].sum()
        # Queue: adds at best (price_rel=0)
        at_best_bid = add_bid & (events[:, 3] == 0)
        at_best_ask = add_ask & (events[:, 3] == 0)
        self.bid_queue_ahead += qtys[at_best_bid].sum()
        self.ask_queue_ahead += qtys[at_best_ask].sum()

        # CANCEL events
        cancel_mask = actions == ACTION_CANCEL
        cancel_bid = cancel_mask & (sides == SIDE_BID)
        cancel_ask = cancel_mask & (sides == SIDE_ASK)
        self.bid_depth = max(0, self.bid_depth - qtys[cancel_bid].sum())
        self.ask_depth = max(0, self.ask_depth - qtys[cancel_ask].sum())
        self.bid_queue_ahead = max(0, self.bid_queue_ahead - qtys[cancel_bid].sum() * 0.5)
        self.ask_queue_ahead = max(0, self.ask_queue_ahead - qtys[cancel_ask].sum() * 0.5)

        # TRADE events
        trade_mask = actions == ACTION_TRADE
        trade_bid = trade_mask & (sides == SIDE_BID)
        trade_ask = trade_mask & (sides == SIDE_ASK)
        n_trades = trade_mask.sum()
        self.trade_count += int(n_trades)

        # Buyer aggressor (side=BID) lifts asks
        self.ask_depth = max(0, self.ask_depth - qtys[trade_bid].sum())
        self.ask_queue_ahead = max(0, self.ask_queue_ahead - qtys[trade_bid].sum())
        # Seller aggressor (side=ASK) hits bids
        self.bid_depth = max(0, self.bid_depth - qtys[trade_ask].sum())
        self.bid_queue_ahead = max(0, self.bid_queue_ahead - qtys[trade_ask].sum())

        # Track recent trade flow (just this batch)
        self.buy_vol_recent = float(qtys[trade_bid].sum())
        self.sell_vol_recent = float(qtys[trade_ask].sum())

        if n_trades > 0:
            # Last trade side
            last_trade_idx = np.where(trade_mask)[0][-1]
            self.last_trade_side = int(sides[last_trade_idx])

    def get_features(self) -> np.ndarray:
        """Return 10 book microstructure features."""
        total_trade = self.buy_vol_recent + self.sell_vol_recent + 1e-8
        trade_imb = (self.buy_vol_recent - self.sell_vol_recent) / total_trade
        depth_total = self.bid_depth + self.ask_depth + 1e-8
        depth_imb = (self.bid_depth - self.ask_depth) / depth_total

        return np.array([
            min(self.spread_ticks, 10.0) / 5.0,
            np.clip(depth_imb, -1, 1),
            np.clip(trade_imb, -1, 1),
            min(self.bid_depth, 500) / 250.0,
            min(self.ask_depth, 500) / 250.0,
            min(self.bid_queue_ahead, 200) / 100.0,
            min(self.ask_queue_ahead, 200) / 100.0,
            min(self.trade_count, 100) / 50.0,
            float(self.last_trade_side),
            min(self.buy_vol_recent + self.sell_vol_recent, 100) / 50.0,
        ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# FIFO FILL SIMULATOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class Order:
    """A resting limit order with FIFO queue tracking."""
    side: int           # 0=bid (long), 1=ask (short)
    price_ticks: float  # absolute price in ticks from reference
    qty: int = 1
    queue_ahead: float = 0.0  # contracts ahead in queue
    submitted_at: int = 0     # event index when submitted
    is_filled: bool = False
    fill_event_idx: int = -1


class FIFOSimulator:
    """
    Simulates FIFO fills by tracking queue position.

    When we place a limit order:
    1. We join the queue at the back (queue_ahead = current depth at that level)
    2. As trades execute at our price level on the opposite side, queue_ahead decreases
    3. When queue_ahead <= 0, our order is filled

    For market orders: instant fill at current best opposite price.
    """

    def __init__(self):
        self.pending_order: Optional[Order] = None
        self.position: int = 0       # +1 long, -1 short, 0 flat
        self.entry_price: float = 0.0
        self.entry_event_idx: int = 0
        self.max_favorable: float = 0.0
        self.max_adverse: float = 0.0

    def submit_limit(self, side: int, queue_ahead: float, event_idx: int):
        """Submit a limit order at BBO with estimated queue position."""
        self.pending_order = Order(
            side=side,
            price_ticks=0,  # at current BBO
            queue_ahead=queue_ahead,
            submitted_at=event_idx,
        )

    def submit_market(self, side: int, event_idx: int):
        """Submit a market order (instant fill)."""
        direction = 1 if side == SIDE_BID else -1
        self.position = direction
        self.entry_price = 0.0  # reference price
        self.entry_event_idx = event_idx
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.pending_order = None

    def cancel_pending(self):
        """Cancel pending limit order."""
        self.pending_order = None

    def check_fill_batch(self, events: np.ndarray, start_idx: int) -> bool:
        """Check if pending order gets filled by events in this batch (vectorized)."""
        if self.pending_order is None or self.pending_order.is_filled:
            return False

        actions = events[:, 1].astype(np.int32)
        sides = events[:, 2].astype(np.int32)
        qty_logs = events[:, 4]
        qtys = np.where(qty_logs > 0, np.exp(np.clip(qty_logs, 0, 10)), 1.0)

        # Trades that consume our queue
        trade_mask = actions == ACTION_TRADE
        if self.pending_order.side == SIDE_BID:
            # Ask-side trades hitting bid fill us
            fill_trades = trade_mask & (sides == SIDE_ASK)
        else:
            # Bid-side trades hitting ask fill us
            fill_trades = trade_mask & (sides == SIDE_BID)

        fill_qty = qtys[fill_trades].sum()

        # Cancels at our side: we move up in queue
        cancel_mask = actions == ACTION_CANCEL
        same_side_cancels = cancel_mask & (sides == self.pending_order.side)
        cancel_benefit = qtys[same_side_cancels].sum() * 0.3

        self.pending_order.queue_ahead -= (fill_qty + cancel_benefit)

        # Check if too long waiting (>2 strides = ~1000 events)
        if start_idx - self.pending_order.submitted_at > 1000:
            self.cancel_pending()
            return False

        if self.pending_order.queue_ahead <= 0:
            self.pending_order.is_filled = True
            self.pending_order.fill_event_idx = start_idx
            direction = 1 if self.pending_order.side == SIDE_BID else -1
            self.position = direction
            self.entry_event_idx = start_idx
            self.max_favorable = 0.0
            self.max_adverse = 0.0
            return True

        return False

    def update_position_pnl(self, price_move_ticks: float):
        """Update position tracking with current price move."""
        if self.position == 0:
            return
        pnl = price_move_ticks * self.position
        self.max_favorable = max(self.max_favorable, pnl)
        self.max_adverse = min(self.max_adverse, pnl)


# ═══════════════════════════════════════════════════════════════
# ENVIRONMENT: FIFO REPLAY WITH MULTI-SIGNAL STATE
# ═══════════════════════════════════════════════════════════════

@dataclass
class DayData:
    """Pre-loaded data for one trading day."""
    date: str
    events: np.ndarray           # (N, 6) MBO events
    timestamps: np.ndarray       # (N,) nanosecond timestamps
    labels_10s: np.ndarray       # (N,) 10s forward return labels
    predictions: Optional[np.ndarray] = None  # (M, 3) CNN-Mamba predictions
    embeddings: Optional[np.ndarray] = None   # (M, 96) CNN-Mamba embeddings


class FIFOReplayEnv:
    """
    RL environment that replays MBO events with FIFO fill simulation.

    State (48-dim):
      - CNN-Mamba predictions at 3 horizons (3)
      - Prediction magnitude + direction (2)
      - Book microstructure features (10)
      - Position state (6): in_pos, direction, unrealized_pnl, hold_time, MFE, MAE
      - Trade history (5): last 5 trade PnLs
      - Session/time features (3)
      - Running stats (4): win_streak, loss_streak, cumulative_pnl, trade_count
      - Prediction z-scores (3): rolling z-score of predictions
      - Volatility features (2): recent vol, vol trend

    Actions (factorized):
      - Gate: SKIP(0), ENTER_LONG(1), ENTER_SHORT(2), EXIT(3)
      - Entry type: LIMIT_BBO(0), LIMIT_AGGRESSIVE(1), MARKET(2)
      - TP index: 0-7 -> [2,4,6,8,10,15,20,30] ticks
      - SL index: 0-6 -> [2,3,5,8,10,15,20] ticks
      - Hold index: 0-5 -> [5,10,30,60,120,300] seconds
    """

    STATE_DIM = 38  # Keep manageable

    def __init__(self, days: List[DayData], max_steps_per_day: int = 3000):
        self.days = days
        self.max_steps = max_steps_per_day
        self.current_day_idx = 0
        self.step_idx = 0

        # State tracking
        self.book = SimplifiedBook()
        self.sim = FIFOSimulator()
        self.trade_history: List[dict] = []
        self.recent_pnl = deque(maxlen=5)
        self.win_streak = 0
        self.loss_streak = 0
        self.cumulative_pnl = 0.0
        self.trade_count_today = 0

        # Rolling prediction stats
        self.pred_history = deque(maxlen=200)
        self.label_history = deque(maxlen=200)

        # Current decision state
        self.current_pred = np.zeros(3, dtype=np.float32)  # 1s, 5s, 10s
        self.pending_tp = 0.0
        self.pending_sl = 0.0
        self.pending_hold_secs = 0.0
        self.entry_timestamp = 0

        log.info(f"FIFOReplayEnv: {len(days)} days loaded")
        for d in days:
            n_preds = d.predictions.shape[0] if d.predictions is not None else 0
            log.info(f"  {d.date}: {d.events.shape[0]:,} events, {n_preds} predictions")

    def reset(self, day_idx: Optional[int] = None) -> np.ndarray:
        """Reset environment to start of a day."""
        if day_idx is not None:
            self.current_day_idx = day_idx
        else:
            self.current_day_idx = (self.current_day_idx + 1) % len(self.days)

        self.step_idx = 0
        self.book = SimplifiedBook()
        self.sim = FIFOSimulator()
        self.trade_count_today = 0
        self.pred_history.clear()
        self.label_history.clear()

        # Process initial warmup events (first 5000 events = ~10 predictions)
        day = self.days[self.current_day_idx]
        warmup = min(5000, len(day.events) // 10)
        self.book.update_batch(day.events[:warmup])

        self.step_idx = warmup // DECISION_STRIDE

        return self._get_state()

    def _get_prediction_at_step(self) -> np.ndarray:
        """Get CNN-Mamba prediction aligned to current event position."""
        day = self.days[self.current_day_idx]
        if day.predictions is None:
            return np.zeros(3, dtype=np.float32)

        # Predictions are at stride-500 from events, our decisions at stride-5000
        # So pred_idx = step_idx * (DECISION_STRIDE / 500)
        pred_idx = min(self.step_idx * (DECISION_STRIDE // 500), len(day.predictions) - 1)
        pred = day.predictions[pred_idx].astype(np.float32)
        return np.nan_to_num(pred, nan=0.0)

    def _get_state(self) -> np.ndarray:
        """Build state vector (38-dim)."""
        pred = self._get_prediction_at_step()
        self.current_pred = pred
        self.pred_history.append(pred[2])  # track 10s pred

        day = self.days[self.current_day_idx]
        event_idx = self.step_idx * DECISION_STRIDE
        n_events = len(day.events)

        # Prediction features (5)
        pred_mag = float(np.abs(pred[2]))
        pred_dir = float(np.sign(pred[2]))

        # Rolling z-score of predictions
        if len(self.pred_history) > 10:
            ph = np.array(self.pred_history)
            pred_mean = ph.mean()
            pred_std = max(ph.std(), 1e-8)
            pred_z = (pred[2] - pred_mean) / pred_std
        else:
            pred_z = 0.0

        # Book features (10)
        book_feat = self.book.get_features()

        # Position features (6)
        if self.sim.position != 0:
            # Estimate unrealized PnL from recent price moves
            # Use labels as proxy for price movement since entry
            entry_steps_ago = max(1, self.step_idx - self.sim.entry_event_idx // DECISION_STRIDE)
            in_pos = 1.0
            pos_dir = float(self.sim.position)
            # Use cumulative label since entry as unrealized PnL proxy
            recent_labels = day.labels_10s[
                max(0, event_idx - entry_steps_ago * DECISION_STRIDE):event_idx
            ]
            if len(recent_labels) > 0:
                unrealized = float(recent_labels.sum()) * self.sim.position * 10.0  # rough ticks
            else:
                unrealized = 0.0
            hold_frac = min(entry_steps_ago / 1000.0, 2.0)
            mfe = self.sim.max_favorable / 30.0
            mae = self.sim.max_adverse / 30.0
        else:
            in_pos = 0.0
            pos_dir = 0.0
            unrealized = 0.0
            hold_frac = 0.0
            mfe = 0.0
            mae = 0.0

        # Trade history (5)
        recent = list(self.recent_pnl) + [0.0] * (5 - len(self.recent_pnl))

        # Time features (3)
        day_progress = min(event_idx / max(n_events, 1), 1.0)
        # Estimate session from day progress
        # ~0.0-0.3 = overnight, 0.3-0.5 = pre-market, 0.5-0.7 = RTH core, 0.7-1.0 = close
        session_approx = day_progress

        # Volatility features (2)
        if len(self.label_history) > 10:
            lh = np.array(self.label_history)
            vol = float(lh.std())
            vol_trend = float(lh[-10:].std() - lh.std())
        else:
            vol = 0.0
            vol_trend = 0.0

        # Record label
        if event_idx < n_events:
            self.label_history.append(day.labels_10s[min(event_idx, n_events - 1)])

        # Running stats (4)
        state = np.array([
            # Predictions (5)
            pred[0], pred[1], pred[2],
            pred_mag, pred_dir,
            # Book (10)
            *book_feat,
            # Position (6)
            in_pos, pos_dir,
            np.clip(unrealized / 20.0, -2, 2),
            hold_frac, mfe, mae,
            # Trade history (5)
            *[p / 10.0 for p in recent[:5]],
            # Time (2)
            day_progress, session_approx,
            # Volatility (2)
            vol * 100, vol_trend * 100,
            # Running stats (4)
            min(self.win_streak, 10) / 5.0,
            min(self.loss_streak, 10) / 5.0,
            np.clip(self.cumulative_pnl / 100.0, -2, 2),
            min(self.trade_count_today, 50) / 25.0,
            # Pred z-score (1) -> total = 5+10+6+5+2+2+4+1 = 35...
            np.clip(pred_z / 3.0, -2, 2),
            # Padding to 38
            0.0, 0.0, 0.0,
        ], dtype=np.float32)

        # Sanitize: replace NaN/inf with 0
        state = np.nan_to_num(state, nan=0.0, posinf=2.0, neginf=-2.0)
        state = np.clip(state, -10.0, 10.0)

        return state[:self.STATE_DIM]

    def step(self, gate: int, entry_type: int, tp_idx: int, sl_idx: int,
             hold_idx: int) -> Tuple[float, np.ndarray, bool, dict]:
        """
        Execute one decision step.

        Returns: (reward, next_state, done, info)
        """
        day = self.days[self.current_day_idx]
        event_start = self.step_idx * DECISION_STRIDE
        event_end = min(event_start + DECISION_STRIDE, len(day.events))
        info = {"traded": False, "filled": False, "exit_reason": ""}

        reward = 0.0

        # Get batch of events for this stride
        batch_events = day.events[event_start:event_end]

        # ── Handle pending order fills (vectorized) ──
        if self.sim.pending_order is not None and not self.sim.pending_order.is_filled:
            if self.sim.check_fill_batch(batch_events, event_start):
                info["filled"] = True

        # ── Execute gate action ──
        if gate == 0:  # SKIP
            if self.sim.position == 0:
                reward = -0.002  # tiny idle penalty to encourage trading
            # If in position, SKIP = continue holding

        elif gate == 1 or gate == 2:  # ENTER_LONG or ENTER_SHORT
            if self.sim.position == 0 and self.sim.pending_order is None:
                direction_side = SIDE_BID if gate == 1 else SIDE_ASK
                tp = float(TP_TICKS[tp_idx])
                sl = float(SL_TICKS[sl_idx])
                hold_s = float(HOLD_SECS[hold_idx])

                self.pending_tp = tp
                self.pending_sl = sl
                self.pending_hold_secs = hold_s
                self.entry_timestamp = event_start

                if entry_type == 0:  # LIMIT at BBO
                    queue = self.book.bid_queue_ahead if direction_side == SIDE_BID else self.book.ask_queue_ahead
                    self.sim.submit_limit(direction_side, queue, event_start)
                elif entry_type == 1:  # LIMIT aggressive (join inside)
                    queue = max(0, (self.book.bid_queue_ahead if direction_side == SIDE_BID
                                    else self.book.ask_queue_ahead) * 0.3)
                    self.sim.submit_limit(direction_side, queue, event_start)
                elif entry_type == 2:  # MARKET
                    self.sim.submit_market(direction_side, event_start)
                    info["filled"] = True
                    # Market order cost: full spread
                    spread_cost = max(self.book.spread_ticks, 1.0) * 0.5
                    reward -= spread_cost * 0.1  # small immediate cost signal

        elif gate == 3:  # EXIT
            if self.sim.position != 0:
                # Market exit
                pnl = self._close_position(event_start, "exit_action")
                reward = pnl
                info["traded"] = True
                info["exit_reason"] = "exit_action"
                info["pnl_ticks"] = pnl

        # ── Update book state for this stride (vectorized) ──
        self.book.update_batch(batch_events)

        # ── Check TP/SL/hold timeout for open position ──
        if self.sim.position != 0:
            # Estimate price move from labels
            label_idx = min(self.step_idx, len(day.labels_10s) - 1)
            price_move = day.labels_10s[label_idx] * 10.0 * self.sim.position  # ticks
            self.sim.update_position_pnl(price_move)

            # Check TP
            if self.sim.max_favorable >= self.pending_tp:
                pnl = self._close_position(event_start, "tp")
                reward = pnl
                info["traded"] = True
                info["exit_reason"] = "tp"
                info["pnl_ticks"] = pnl

            # Check SL
            elif self.sim.max_adverse <= -self.pending_sl:
                pnl = self._close_position(event_start, "sl")
                reward = pnl
                info["traded"] = True
                info["exit_reason"] = "sl"
                info["pnl_ticks"] = pnl

            # Check hold timeout
            elif (event_start - self.entry_timestamp) > self.pending_hold_secs * 200:
                # ~200 events/sec approximate rate for ES
                pnl = self._close_position(event_start, "hold_timeout")
                reward = pnl
                info["traded"] = True
                info["exit_reason"] = "hold_timeout"
                info["pnl_ticks"] = pnl

        # ── Advance ──
        self.step_idx += 1
        done = (self.step_idx * DECISION_STRIDE >= len(day.events) - DECISION_STRIDE
                or self.step_idx >= self.max_steps)

        # Force close at end of day
        if done and self.sim.position != 0:
            pnl = self._close_position(event_start, "eod")
            reward = pnl
            info["traded"] = True
            info["exit_reason"] = "eod"

        # End-of-day trade activity bonus/penalty
        if done:
            if self.trade_count_today == 0:
                reward -= 2.0  # penalty for not trading at all
            elif self.trade_count_today >= 3:
                reward += 0.5  # small bonus for reasonable activity

        next_state = self._get_state() if not done else np.zeros(self.STATE_DIM, dtype=np.float32)
        return reward, next_state, done, info

    def _close_position(self, event_idx: int, reason: str) -> float:
        """Close current position and return net PnL in ticks."""
        if self.sim.position == 0:
            return 0.0

        # PnL from MFE/MAE tracking
        # Use a mix of max_favorable (capped at TP) and current unrealized
        if reason == "tp":
            raw_pnl = self.pending_tp
        elif reason == "sl":
            raw_pnl = -self.pending_sl
        elif reason == "hold_timeout" or reason == "eod" or reason == "exit_action":
            # Use recent label as proxy
            day = self.days[self.current_day_idx]
            label_idx = min(self.step_idx, len(day.labels_10s) - 1)
            # Sum labels since entry as cumulative move
            entry_step = self.sim.entry_event_idx // DECISION_STRIDE
            labels_since_entry = day.labels_10s[entry_step:self.step_idx]
            raw_pnl = float(labels_since_entry.sum()) * 10.0 * self.sim.position if len(labels_since_entry) > 0 else 0.0
            raw_pnl = np.clip(raw_pnl, -self.pending_sl, self.pending_tp)
        else:
            raw_pnl = 0.0

        # Apply costs
        entry_cost = COMMISSION_PER_SIDE_TICKS  # limit fill: commission only
        if self.sim.pending_order is None:  # was market entry
            entry_cost += max(self.book.spread_ticks, 1.0) * 0.5  # pay half spread

        exit_cost = COMMISSION_PER_SIDE_TICKS + max(self.book.spread_ticks, 1.0)  # market exit: spread + commission

        net_pnl = raw_pnl - entry_cost - exit_cost

        # Record trade
        trade = {
            "pnl_ticks": float(net_pnl),
            "raw_pnl": float(raw_pnl),
            "entry_cost": float(entry_cost),
            "exit_cost": float(exit_cost),
            "reason": reason,
            "hold_steps": self.step_idx - self.sim.entry_event_idx // DECISION_STRIDE,
            "tp": self.pending_tp,
            "sl": self.pending_sl,
            "direction": self.sim.position,
            "mfe": self.sim.max_favorable,
            "mae": self.sim.max_adverse,
        }
        self.trade_history.append(trade)
        self.recent_pnl.append(net_pnl)
        self.cumulative_pnl += net_pnl
        self.trade_count_today += 1

        if net_pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        # Reset position
        self.sim.position = 0
        self.sim.pending_order = None
        self.sim.max_favorable = 0.0
        self.sim.max_adverse = 0.0

        return net_pnl


# ═══════════════════════════════════════════════════════════════
# POLICY NETWORK
# ═══════════════════════════════════════════════════════════════

class ExecPolicyV5(nn.Module):
    """
    Multi-head actor-critic for execution decisions.

    Architecture: MLP backbone -> 5 action heads + value head
    No LSTM (avoids state management issues that plagued v3.x).
    Instead, trade history and rolling stats are in the state vector.
    """

    def __init__(self, state_dim: int = 38, hidden_dim: int = 256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        h = hidden_dim // 2  # 128

        # Action heads
        self.gate_head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 4))
        self.entry_head = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 3))
        self.tp_head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, len(TP_TICKS)))
        self.sl_head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, len(SL_TICKS)))
        self.hold_head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, len(HOLD_SECS)))

        # Value head
        self.value_head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller gain for action heads to start with uniform-ish distribution
        for head in [self.gate_head, self.entry_head, self.tp_head, self.sl_head, self.hold_head]:
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
        # Bias gate toward ENTER to prevent initial collapse to all-skip
        with torch.no_grad():
            self.gate_head[-1].bias[1] += 0.5  # ENTER_LONG
            self.gate_head[-1].bias[2] += 0.5  # ENTER_SHORT

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        return (
            self.gate_head(features),
            self.entry_head(features),
            self.tp_head(features),
            self.sl_head(features),
            self.hold_head(features),
            self.value_head(features).squeeze(-1),
        )

    def get_action(self, state: torch.Tensor, deterministic: bool = False,
                   temperature: float = 1.0):
        gate_l, entry_l, tp_l, sl_l, hold_l, value = self.forward(state)

        # Apply temperature
        gate_l = gate_l / temperature
        entry_l = entry_l / temperature
        tp_l = tp_l / temperature
        sl_l = sl_l / temperature
        hold_l = hold_l / temperature

        dists = [Categorical(logits=l) for l in [gate_l, entry_l, tp_l, sl_l, hold_l]]

        if deterministic:
            actions = [l.argmax(-1) for l in [gate_l, entry_l, tp_l, sl_l, hold_l]]
        else:
            actions = [d.sample() for d in dists]

        log_prob = sum(d.log_prob(a) for d, a in zip(dists, actions))
        entropy = sum(d.entropy() for d in dists)

        return [a.item() for a in actions], log_prob, value, entropy

    def evaluate_actions(self, states, actions_list):
        """Evaluate log probs for stored actions. actions_list: list of 5 action tensors."""
        gate_l, entry_l, tp_l, sl_l, hold_l, values = self.forward(states)
        dists = [Categorical(logits=l) for l in [gate_l, entry_l, tp_l, sl_l, hold_l]]
        log_probs = sum(d.log_prob(a) for d, a in zip(dists, actions_list))
        entropy = sum(d.entropy() for d in dists)
        return log_probs, values, entropy


# ═══════════════════════════════════════════════════════════════
# PPO TRAINER
# ═══════════════════════════════════════════════════════════════

class PPOTrainerV5:
    """PPO trainer with anti-collapse mechanisms."""

    def __init__(self, env: FIFOReplayEnv, lr: float = 3e-4,
                 gamma: float = 0.995, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 entropy_coeff_start: float = 0.08,
                 entropy_coeff_end: float = 0.01,
                 entropy_anneal_epochs: int = 200,
                 value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 batch_size: int = 512,
                 ppo_epochs: int = 4):

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end = entropy_coeff_end
        self.entropy_anneal_epochs = entropy_anneal_epochs
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        self.policy = ExecPolicyV5(
            state_dim=FIFOReplayEnv.STATE_DIM,
            hidden_dim=256,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=entropy_anneal_epochs, eta_min=lr * 0.1
        )

        n_params = sum(p.numel() for p in self.policy.parameters())
        log.info(f"Policy network: {n_params:,} parameters on {device}")

    def _get_entropy_coeff(self, epoch: int) -> float:
        """Anneal entropy coefficient."""
        if epoch >= self.entropy_anneal_epochs:
            return self.entropy_coeff_end
        frac = epoch / self.entropy_anneal_epochs
        return self.entropy_coeff_start + (self.entropy_coeff_end - self.entropy_coeff_start) * frac

    def collect_episode(self, day_idx: int = None, deterministic: bool = False,
                        temperature: float = 1.0) -> dict:
        """Collect one full day episode."""
        self.policy.eval()
        state = self.env.reset(day_idx)

        states, all_actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                actions, lp, val, ent = self.policy.get_action(
                    state_t, deterministic=deterministic, temperature=temperature
                )

            gate, entry_type, tp_idx, sl_idx, hold_idx = actions
            reward, next_state, done, info = self.env.step(gate, entry_type, tp_idx, sl_idx, hold_idx)

            states.append(state)
            all_actions.append(actions)
            rewards.append(reward)
            log_probs.append(lp.item())
            values.append(val.item())
            entropies.append(ent.item())

            state = next_state

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.int64),  # (T, 5)
            "rewards": np.array(rewards, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "entropies": np.array(entropies, dtype=np.float32),
        }

    def compute_gae(self, rewards, values):
        """GAE with higher gamma for longer-horizon rewards."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(self, rollout: dict, entropy_coeff: float) -> dict:
        """PPO update on collected rollout."""
        advantages, returns = self.compute_gae(rollout["rewards"], rollout["values"])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip rewards to prevent extreme gradients
        returns = np.clip(returns, -50, 50)

        states_t = torch.FloatTensor(rollout["states"]).to(device)
        actions_t = torch.LongTensor(rollout["actions"]).to(device)  # (T, 5)
        old_lp_t = torch.FloatTensor(rollout["log_probs"]).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)

        n = len(states_t)
        total_ploss = 0.0
        total_vloss = 0.0
        total_entropy = 0.0
        n_updates = 0

        self.policy.train()

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                b_idx = idx[start:end]

                action_list = [actions_t[b_idx, i] for i in range(5)]
                new_lp, new_val, entropy = self.policy.evaluate_actions(
                    states_t[b_idx], action_list
                )

                ratio = torch.exp(new_lp - old_lp_t[b_idx])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(ratio * adv_t[b_idx], clipped * adv_t[b_idx]).mean()
                value_loss = F.mse_loss(new_val, ret_t[b_idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coeff * value_loss + entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_ploss / max(1, n_updates),
            "value_loss": total_vloss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def train(self, n_epochs: int = 500, save_every: int = 10):
        """Main training loop."""
        best_sortino = -float("inf")
        best_pnl = -float("inf")
        history = []

        # MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri("http://localhost:5000")
                mlflow.set_experiment("rl_fifo_v5")
                mlflow.start_run(run_name=f"rl_fifo_v5_{_ts}")
                mlflow.log_params({
                    "n_epochs": n_epochs,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "gamma": self.gamma,
                    "entropy_start": self.entropy_coeff_start,
                    "entropy_end": self.entropy_coeff_end,
                    "state_dim": FIFOReplayEnv.STATE_DIM,
                    "n_days": len(self.env.days),
                    "device": str(device),
                })
            except Exception as e:
                log.warning(f"MLflow init failed: {e}")

        log.info("=" * 80)
        log.info("RL FIFO v5 TRAINING")
        log.info(f"  Epochs: {n_epochs}")
        log.info(f"  Days: {len(self.env.days)}")
        log.info(f"  Device: {device}")
        log.info(f"  Entropy: {self.entropy_coeff_start} -> {self.entropy_coeff_end}")
        log.info("=" * 80)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            entropy_coeff = self._get_entropy_coeff(epoch)

            # Collect episodes from multiple days
            all_rollouts = []
            epoch_trades = []
            epoch_pnl = 0.0

            # Train on each day
            for day_idx in range(len(self.env.days)):
                # Temperature: start high (1.5) to encourage exploration, anneal to 1.0
                temp = max(1.0, 1.5 - epoch / (n_epochs * 0.5))
                rollout = self.collect_episode(day_idx=day_idx, temperature=temp)
                all_rollouts.append(rollout)

                # Collect stats
                day_trades = [t for t in self.env.trade_history
                              if t not in epoch_trades]
                epoch_trades.extend(day_trades)

            # Merge rollouts
            merged = {
                "states": np.concatenate([r["states"] for r in all_rollouts]),
                "actions": np.concatenate([r["actions"] for r in all_rollouts]),
                "rewards": np.concatenate([r["rewards"] for r in all_rollouts]),
                "log_probs": np.concatenate([r["log_probs"] for r in all_rollouts]),
                "values": np.concatenate([r["values"] for r in all_rollouts]),
            }

            # PPO update
            losses = self.ppo_update(merged, entropy_coeff)
            self.scheduler.step()

            # Stats
            n_trades = len(epoch_trades)
            if n_trades > 0:
                pnls = np.array([t["pnl_ticks"] for t in epoch_trades])
                total_pnl = pnls.sum()
                win_rate = 100 * (pnls > 0).mean()
                avg_pnl = pnls.mean()
                downside = pnls[pnls < 0]
                if len(downside) > 1:
                    sortino = float(avg_pnl / (downside.std() + 1e-8))
                else:
                    sortino = float(avg_pnl) if avg_pnl > 0 else 0.0
            else:
                total_pnl = 0.0
                win_rate = 0.0
                avg_pnl = 0.0
                sortino = -10.0

            elapsed = time.time() - t0

            # Gate distribution
            if len(merged["actions"]) > 0:
                gate_dist = np.bincount(merged["actions"][:, 0], minlength=4) / len(merged["actions"])
                gate_str = f"S={gate_dist[0]:.0%} L={gate_dist[1]:.0%} S={gate_dist[2]:.0%} X={gate_dist[3]:.0%}"
            else:
                gate_str = "N/A"

            result = {
                "epoch": epoch,
                "n_trades": n_trades,
                "total_pnl_ticks": float(total_pnl),
                "total_pnl_usd": float(total_pnl * TICK_VALUE),
                "win_rate": float(win_rate),
                "avg_pnl": float(avg_pnl),
                "sortino": float(sortino),
                "entropy": losses["entropy"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy_coeff": entropy_coeff,
                "elapsed_s": elapsed,
                "gate_dist": gate_str,
            }
            history.append(result)

            # Log
            if epoch % 5 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"Trades: {n_trades:3d} | "
                    f"WR: {win_rate:5.1f}% | "
                    f"PnL: {total_pnl:+7.1f}t (${total_pnl * TICK_VALUE:+8,.0f}) | "
                    f"Sortino: {sortino:+.3f} | "
                    f"H: {losses['entropy']:.2f} | "
                    f"Gate: {gate_str} | "
                    f"{elapsed:.1f}s"
                )

                # Exit reason breakdown
                if epoch_trades:
                    reasons = {}
                    for t in epoch_trades:
                        r = t["reason"]
                        reasons[r] = reasons.get(r, 0) + 1
                    log.info(f"  Exit reasons: {reasons}")

            # MLflow
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        "n_trades": n_trades,
                        "total_pnl_ticks": float(total_pnl),
                        "win_rate": float(win_rate),
                        "sortino": float(sortino),
                        "entropy": losses["entropy"],
                        "policy_loss": losses["policy_loss"],
                        "trade_rate_pct": n_trades / max(1, len(merged["actions"])) * 100,
                    }, step=epoch)
                except Exception:
                    pass

            # Save best
            if n_trades >= 5 and sortino > best_sortino:
                best_sortino = sortino
                self._save_checkpoint("best.pt", epoch, result)
                log.info(f"  ** New best Sortino: {sortino:.4f} ({n_trades} trades)")

            if n_trades >= 5 and total_pnl > best_pnl:
                best_pnl = total_pnl
                self._save_checkpoint("best_pnl.pt", epoch, result)

            # Periodic save
            if epoch % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch}.pt", epoch, result)
                self._save_history(history)

            # Memory cleanup
            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final save
        self._save_checkpoint("final.pt", n_epochs, history[-1] if history else {})
        self._save_history(history)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass

        log.info(f"\nTraining complete. Best Sortino: {best_sortino:.4f}")
        return history

    def _save_checkpoint(self, filename: str, epoch: int, stats: dict):
        path = OUTPUT_DIR / filename
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "stats": stats,
        }, str(path))

    def _save_history(self, history: list):
        path = OUTPUT_DIR / "training_history.json"
        with open(str(path), "w") as f:
            json.dump(history, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_days(mbo_dir: Path, pred_dir: Path, max_days: int = 20) -> List[DayData]:
    """Load MBO event days with matching CNN-Mamba predictions."""
    days = []

    # Load CNN-Mamba fold predictions with date info
    pred_by_date = {}
    emb_by_date = {}
    for fold_file in sorted(pred_dir.glob("fold_*_oot_predictions.npz")):
        try:
            d = np.load(str(fold_file), allow_pickle=True)
            if "oot_files" in d:
                oot_path = str(d["oot_files"][0])
                date = oot_path.split("/")[-1].split("_")[0]
                pred_by_date[date] = d["predictions"]
                if "embeddings" in d:
                    emb_by_date[date] = d["embeddings"]
        except Exception as e:
            log.warning(f"Failed to load {fold_file}: {e}")

    log.info(f"Loaded predictions for {len(pred_by_date)} dates: {sorted(pred_by_date.keys())}")

    # Match with MBO event files
    for mbo_file in sorted(mbo_dir.glob("*_mbo_events.npz")):
        date = mbo_file.stem.split("_")[0]

        if date not in pred_by_date:
            continue

        if len(days) >= max_days:
            break

        try:
            mbo = np.load(str(mbo_file), allow_pickle=True)
            events = mbo["events"]
            timestamps = mbo["timestamps"]

            # Get labels (sanitize NaN)
            labels_10s = mbo["labels_10s"] if "labels_10s" in mbo else np.zeros(len(events))
            labels_10s = np.nan_to_num(labels_10s, nan=0.0)

            day = DayData(
                date=date,
                events=events,
                timestamps=timestamps,
                labels_10s=labels_10s,
                predictions=pred_by_date[date],
                embeddings=emb_by_date.get(date),
            )
            days.append(day)
            log.info(f"  Loaded {date}: {events.shape[0]:,} events, {len(pred_by_date[date])} preds")

        except Exception as e:
            log.warning(f"Failed to load {mbo_file}: {e}")

    return days


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL FIFO Execution Agent v5")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-days", type=int, default=10,
                        help="Max training days to load")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--entropy-start", type=float, default=0.08)
    parser.add_argument("--entropy-end", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    global device
    device = torch.device(dev)

    log.info(f"Device: {device}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    log.info("Loading MBO event data and predictions...")
    days = load_days(MBO_DIR, CNN_MAMBA_DIR, max_days=args.max_days)

    if not days:
        log.error("No matching days found! Check MBO_DIR and prediction files.")
        sys.exit(1)

    # Create environment
    env = FIFOReplayEnv(days, max_steps_per_day=3000)

    if args.eval:
        # Evaluation mode
        ckpt_path = args.checkpoint or str(OUTPUT_DIR / "best.pt")
        log.info(f"Evaluating checkpoint: {ckpt_path}")
        trainer = PPOTrainerV5(env, lr=args.lr)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.policy.load_state_dict(ckpt["model_state_dict"])

        for day_idx in range(len(days)):
            rollout = trainer.collect_episode(day_idx=day_idx, deterministic=True)
            trades = [t for t in env.trade_history]
            if trades:
                pnls = np.array([t["pnl_ticks"] for t in trades])
                log.info(f"Day {days[day_idx].date}: {len(trades)} trades, "
                         f"PnL={pnls.sum():+.1f}t, WR={100*(pnls>0).mean():.0f}%")
            env.trade_history.clear()
    else:
        # Training mode
        trainer = PPOTrainerV5(
            env,
            lr=args.lr,
            entropy_coeff_start=args.entropy_start,
            entropy_coeff_end=args.entropy_end,
            entropy_anneal_epochs=min(200, args.epochs // 2),
            batch_size=args.batch_size,
        )
        trainer.train(n_epochs=args.epochs)


if __name__ == "__main__":
    main()
