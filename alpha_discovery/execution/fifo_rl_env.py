#!/usr/bin/env python3
"""
FIFO-Aware RL Trading Environment for ES Futures Execution
===========================================================

Replays actual MBO event data event-by-event and simulates a FIFO queue
fill model for limit orders. This environment is the training ground for
SAC, PPO, and MLP execution models.

FIFO Queue Logic (critical to understand)
-----------------------------------------
When you place a limit order at the best bid (to buy) or best ask (to sell),
you join the BACK of that price level's queue. All contracts already resting
at that price must trade before yours fills.

- queue_position: how many contracts are ahead of you (you fill last)
- Each incoming trade (event_type=0) at your price level consumes some of
  those contracts. When cumulative volume consumed >= queue_position, you fill.
- If price moves away from your level before consuming queue_position, the
  order never fills (you cancel or it becomes stale).

Key data facts (confirmed from inspection):
- MBO event types: 0=trade, 1=add, 2=modify, 3=cancel, 4=clear
- Feature cols: 0=time_delta_log, 1=event_type, 2=side, 3=price_rel_ticks,
                4=qty_log, 5=spread_ticks, 6-24=book state features
- Side: +1 = buy (lift ask) / -1 = sell (hit bid)
- Labels in fold predictions: future mid-price move in ticks (N,3) for 1s/5s/10s
- Predictions (N,3): raw model output, scale ~0-1 (not in tick units)
- One prediction per stride (typically 50) events; window=1000 events
- Timestamps are nanoseconds since epoch

Cost constants (HC canonical, DO NOT CHANGE):
- TICK_SIZE = 0.25 points
- TICK_VALUE = $12.50
- COMMISSION_RT_TICKS = 0.376  ($4.70 / $12.50)
- NO spread crossing cost for limit fills (you're passive)
- 1.0 tick spread crossing cost for market orders (you cross the spread)

Observation space (48 dims total):
- [0:4]   Signal: pred_1s, pred_5s, pred_10s, confidence_tier (0-3)
- [4:8]   Book: best_bid_size_log, best_ask_size_log, book_imbalance, spread_ticks
- [8:10]  Price: price_rel_ticks (vs running mid), price_momentum_10
- [10:14] Position: position (0/1/-1), unrealized_pnl_ticks, time_in_pos_s, queue_frac_filled
- [14:16] MFE/MAE: max_fav_excursion_ticks, max_adv_excursion_ticks
- [16:19] Context: realized_vol_60s, tod_sin, tod_cos
- [19:24] Flow: event_density_10s, buy_vol_frac_10s, recent_flow_imbalance, order_cancel_rate, local_event_rate
- [24:29] Trade history: last 5 trade PnLs (normalized)
- [29:32] Stats: rolling_win_rate, rolling_sortino, consecutive_loss_count
- [32:34] Order: pending_order (0/1), pending_order_side (0/1/-1)
- [34:39] Embedding: top-5 PCA components from CNN-Mamba embedding (if available)
- [39:45] PatchTST confluence: pst_1s, pst_5s, pst_10s, confluence_1s, confluence_all, sweep_intensity
- [45:48] Alpha-awareness (HC #114): signal_remaining_frac, entry_signal_strength, current_alpha_alignment

Action space (discrete, 7 actions):
- 0: Do nothing / wait
- 1: Place limit BUY at bid (join back of bid queue)
- 2: Place limit SELL at ask (join back of ask queue)
- 3: Market BUY (lift ask, +1 tick crossing cost)
- 4: Market SELL (hit bid, +1 tick crossing cost)
- 5: Cancel pending order
- 6: Market exit (close position via market order)

Reward:
- Reward per step = 0 (no holding reward, except alpha-gate penalty for bad entries)
- On trade close: delta rolling Sortino (Sortino after close - Sortino before close)
- HC #114 Signal alignment: bonus for trading WITH alpha, penalty for trading AGAINST
- HC #114 Signal decay hold penalty: penalizes holding when alpha has decayed away
- HC #114 Alpha-gate penalty: penalizes entry attempts without meaningful signal
- Penalty: hold > 30s, consecutive losses, overtrading
- HC #104: NO short-side bias — any win = same reward

Author: Claude (Infrastructure Builder)
Date: 2026-05-02
"""

from __future__ import annotations

import os
import math
import time
import logging
from pathlib import Path
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("fifo_rl_env")

# ─── Paths ─────────────────────────────────────────────────────────────────────
LVL3 = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
PRED_DIR = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar"

# ─── Constants (HC canonical — DO NOT CHANGE) ───────────────────────────────────
TICK_SIZE = 0.25              # ES points per tick
TICK_VALUE = 12.50            # USD per tick
COMMISSION_RT_TICKS = 0.376   # $4.70 round-trip / $12.50 per tick
# HC #127 + HC #231(A): NO spread cost, NO crossing cost — full stop.
# User direct quote 2026-05-06: "there's no such thing as crossing spread, it's just
# the number we bought at and the number we sold at." Fill price already reflects
# bid/ask side. Adding any theoretical spread cost on top = double-counting.
# The ONLY cost in this env is the round-trip commission.
COMMISSION_COST = COMMISSION_RT_TICKS  # The ONLY cost. Period.
# DELETED 2026-05-06 per HC #231(A):
#   MARKET_ORDER_COST_TICKS  (was 1.376 — fictitious "commission + 1 tick crossing")
#   LIMIT_ORDER_COST_TICKS   (was 0.376 — duplicate of COMMISSION_RT_TICKS)
# Any importer must migrate to COMMISSION_COST. P&L = sell_price - buy_price - COMMISSION_COST.

# ─── Event type codes (from MBO data inspection) ───────────────────────────────
EVENT_TRADE  = 0  # A trade executed (consumes queue depth)
EVENT_ADD    = 1  # New limit order added to book
EVENT_MODIFY = 2  # Existing order modified
EVENT_CANCEL = 3  # Order cancelled
EVENT_CLEAR  = 4  # Book level cleared

# ─── Feature column indices (confirmed from data inspection) ───────────────────
COL_TIME_DELTA   = 0   # log(time since last event), normalized
COL_EVENT_TYPE   = 1   # 0=trade, 0.25=add, 0.5=modify, 0.75=cancel, 1=clear
COL_SIDE         = 2   # +1=buy, -1=sell
COL_PRICE_REL    = 3   # price relative to running mid, in ticks, capped ±2
COL_QTY_LOG      = 4   # log(quantity), raw (not normalized beyond log)
COL_SPREAD       = 5   # current spread in ticks

# ─── Observation space dimensions ──────────────────────────────────────────────
# 45 base + 3 HC #114 alpha-awareness features = 48
OBS_DIM = 48

# ─── FIFO queue constants ─────────────────────────────────────────────────────
# HC #145: Queue depth MUST come from real MBO data, NOT synthetic estimates.
# We reconstruct L1 book depth in real-time from event stream (adds/cancels/trades).
DEFAULT_QUEUE_DEPTH = 50     # Fallback ONLY for first few events before book builds
MIN_QUEUE_DEPTH     = 5      # Floor to prevent div-by-zero (real book can be thin)
MAX_HOLD_SECS       = 30.0   # Penalty kicks in after 30s hold
MAX_EPISODE_SECS    = 23400  # ~6.5h = one full RTH + pre/post market
RTH_START_SEC       = 9.5 * 3600   # 9:30 AM ET
RTH_END_SEC         = 16.0 * 3600  # 4:00 PM ET

# ─── Reward shaping ─────────────────────────────────────────────────────────────
SORTINO_BUFFER_SIZE = 20    # rolling window for Sortino calculation
MIN_TRADES_FOR_SORTINO = 3  # minimum trades before Sortino is meaningful
OVERTRADING_WINDOW = 60     # seconds to count trades for overtrading detection
OVERTRADING_LIMIT = 10      # max trades per minute before penalty
CONSECUTIVE_LOSS_PENALTY = 0.05  # ticks per consecutive loss after 3rd
SHORT_EDGE_BONUS = 0.0      # HC #104: NO short-side bias. Any win = same reward.

# ─── HC #114: Alpha-Aware RL Constraints ────────────────────────────────────────
# The agent should learn to USE the alpha, not learn patterns divorced from it.
# BUT: don't over-constrain. The model learns HOW to trade from inputs.
# We provide signal-awareness FEATURES (obs[45:48]) and SOFT guidance.
# No preconceived decay values hardcoded as penalties — let the model learn timing.
# Safeguards focus on: (1) signal must exist at entry, (2) mild alignment guidance.

# Alpha-gating: HARD-er gate per HC #230. Old value 0.05 was "any signal" — that's
# what produced 200k+ trades/epoch (overtrading). New value calibrated to roughly
# the historical Top10% confidence (CNN-Mamba pred mag). Tier-aware envs may pull
# from TIER_THRESHOLDS dict instead.
ALPHA_GATE_THRESHOLD = 0.50   # raised 2026-05-06 per HC #230 (was 0.05 — overtrading root cause)
ALPHA_GATE_PENALTY = -0.10    # raised from -0.01 — agent must actually pay for low-conf entries

# Signal-decay: NO hardcoded decay penalty. The model gets decay info as FEATURES
# (obs[45]=signal_remaining, obs[46]=entry_strength, obs[47]=current_alignment)
# and must learn when to exit on its own. We don't impose our decay assumptions.
SIGNAL_DECAY_HALFLIFE_S = 0.25   # used ONLY for obs[45] feature calculation, NOT for penalties

# Signal-alignment: SOFT guidance — mild bonus for with-signal, mild penalty for against
# These are gentle nudges, not hard constraints. The model can still learn freely.
SIGNAL_ALIGNMENT_BONUS = 0.05     # small bonus when trade direction matches signal at entry
SIGNAL_MISALIGNMENT_PENALTY = 0.08  # slightly stronger: "entering against signal is suspicious"


# ─── DST helper (module-level for speed) ──────────────────────────────────────

def _utc_month_from_epoch_s(epoch_s: int) -> int:
    """Return UTC month (1-12) from seconds since epoch, without datetime import overhead."""
    # Days since epoch
    days = epoch_s // 86400
    # Rough month via 365.25-day year (accurate to ±1 month for our purposes)
    year_approx = 1970 + days / 365.25
    frac = year_approx - int(year_approx)
    month = int(frac * 12) + 1
    return max(1, min(12, month))


# ─── Raw MBO action/side codes (match mbo_event_pipeline.py ACTION_MAP/SIDE_MAP) ─
# These are the codes stored in actions_raw / sides_raw arrays in the .npz files.
RAW_ACT_ADD    = 0   # 'A' in pipeline
RAW_ACT_CANCEL = 1   # 'C' in pipeline
RAW_ACT_MODIFY = 2   # 'M' in pipeline
RAW_ACT_TRADE  = 3   # 'T' in pipeline
RAW_ACT_FILL   = 4   # 'F' in pipeline

RAW_SIDE_BID  = 0    # 'B' in pipeline
RAW_SIDE_ASK  = 1    # 'A' in pipeline
RAW_SIDE_NONE = 2    # 'N' in pipeline

# Tick size in raw fixed-point (Databento uses 1e9 per point, ES tick = 0.25 pts)
TICK_SIZE_RAW = 250_000_000


# ─── Per-price-level FIFO order book ───────────────────────────────────────────

class _PriceLevel:
    """FIFO queue at one price level. Mirrors mbo_replay_server.PriceLevel."""
    __slots__ = ('price_raw', 'orders')

    def __init__(self, price_raw: int):
        self.price_raw = price_raw
        self.orders: OrderedDict = OrderedDict()  # order_id -> qty

    def add(self, oid: int, qty: int):
        self.orders[oid] = qty

    def cancel(self, oid: int):
        self.orders.pop(oid, None)

    def modify(self, oid: int, qty: int):
        if oid in self.orders:
            self.orders[oid] = qty

    def total_qty(self) -> int:
        return sum(self.orders.values())

    def qty_ahead_of(self, oid: int) -> int:
        """Return total qty resting ahead of oid in FIFO order."""
        qty = 0
        for o, q in self.orders.items():
            if o == oid:
                break
            qty += q
        return qty

    def consume(self, qty: int) -> list:
        """FIFO consume qty contracts. Returns list of fully-consumed order_ids."""
        consumed = []
        for oid in list(self.orders):
            if qty <= 0:
                break
            q = self.orders[oid]
            if q <= qty:
                qty -= q
                consumed.append(oid)
                del self.orders[oid]
            else:
                self.orders[oid] -= qty
                qty = 0
        return consumed

    def empty(self) -> bool:
        return not self.orders


class BookReconstructor:
    """
    Real per-price-level FIFO order book reconstructed from raw MBO events.

    Processes the raw arrays (order_ids, prices_raw, sizes_raw, sides_raw,
    actions_raw) saved by mbo_event_pipeline.py to maintain an accurate
    book with per-level FIFO queues and order_id tracking.

    This replaces the hacky 2-float exponential-decay approximation.
    """

    def __init__(self):
        self.bids: Dict[int, _PriceLevel] = {}  # price_raw -> _PriceLevel
        self.asks: Dict[int, _PriceLevel] = {}
        self._oid_side: Dict[int, int] = {}    # order_id -> side (RAW_SIDE_BID/ASK)
        self._oid_price: Dict[int, int] = {}   # order_id -> price_raw
        self._n_events_processed: int = 0

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self._oid_side.clear()
        self._oid_price.clear()
        self._n_events_processed = 0

    def _book_for_side(self, side: int) -> dict:
        return self.bids if side == RAW_SIDE_BID else self.asks

    @property
    def initialized(self) -> bool:
        """Book is considered initialized after processing enough events."""
        return self._n_events_processed > 200

    def process_event(self, action: int, side: int, price_raw: int,
                      size: int, order_id: int) -> None:
        """
        Process one raw MBO event to update the book state.

        Parameters
        ----------
        action : int — RAW_ACT_ADD/CANCEL/MODIFY/TRADE/FILL
        side : int — RAW_SIDE_BID/ASK/NONE
        price_raw : int — absolute fixed-point price
        size : int — quantity in contracts
        order_id : int — unique order identifier
        """
        self._n_events_processed += 1

        if action == RAW_ACT_ADD:
            if side == RAW_SIDE_NONE:
                return
            b = self._book_for_side(side)
            if price_raw not in b:
                b[price_raw] = _PriceLevel(price_raw)
            b[price_raw].add(order_id, size)
            self._oid_side[order_id] = side
            self._oid_price[order_id] = price_raw

        elif action == RAW_ACT_CANCEL:
            s = self._oid_side.pop(order_id, None)
            p = self._oid_price.pop(order_id, None)
            if s is not None and p is not None:
                b = self._book_for_side(s)
                if p in b:
                    b[p].cancel(order_id)
                    if b[p].empty():
                        del b[p]

        elif action == RAW_ACT_MODIFY:
            s = self._oid_side.get(order_id)
            old_p = self._oid_price.get(order_id)
            if s is None or old_p is None:
                return
            b = self._book_for_side(s)
            if price_raw != old_p:
                # Price change = cancel + re-add (loses queue priority)
                if old_p in b:
                    b[old_p].cancel(order_id)
                    if b[old_p].empty():
                        del b[old_p]
                if price_raw not in b:
                    b[price_raw] = _PriceLevel(price_raw)
                b[price_raw].add(order_id, size)
                self._oid_price[order_id] = price_raw
            else:
                # Same price — just update qty (keeps priority)
                if old_p in b:
                    b[old_p].modify(order_id, size)

        elif action == RAW_ACT_TRADE:
            # Trade: aggressor side is `side`, consumes PASSIVE side
            if side == RAW_SIDE_BID:
                # Buyer aggressor lifts ask → consume ask side
                passive_book = self.asks
            elif side == RAW_SIDE_ASK:
                # Seller aggressor hits bid → consume bid side
                passive_book = self.bids
            else:
                return
            if price_raw in passive_book:
                consumed = passive_book[price_raw].consume(size)
                if passive_book[price_raw].empty():
                    del passive_book[price_raw]
                for oid in consumed:
                    self._oid_side.pop(oid, None)
                    self._oid_price.pop(oid, None)

        elif action == RAW_ACT_FILL:
            # Fill events: similar to trade, consume the specific order
            s = self._oid_side.pop(order_id, None)
            p = self._oid_price.pop(order_id, None)
            if s is not None and p is not None:
                b = self._book_for_side(s)
                if p in b:
                    b[p].cancel(order_id)  # fully filled = remove
                    if b[p].empty():
                        del b[p]

    def best_bid(self) -> Optional[int]:
        """Best (highest) bid price, or None if no bids."""
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[int]:
        """Best (lowest) ask price, or None if no asks."""
        return min(self.asks) if self.asks else None

    def best_bid_depth(self) -> int:
        """Total qty resting at best bid price level."""
        bb = self.best_bid()
        if bb is None:
            return 0
        return self.bids[bb].total_qty()

    def best_ask_depth(self) -> int:
        """Total qty resting at best ask price level."""
        ba = self.best_ask()
        if ba is None:
            return 0
        return self.asks[ba].total_qty()

    def depth_at(self, side: int, price_raw: int) -> int:
        """Total qty at a specific price level."""
        b = self._book_for_side(side)
        return b[price_raw].total_qty() if price_raw in b else 0

    def total_depth(self, side: int, n_levels: int = 1) -> int:
        """
        Total qty across top n_levels on the given side.

        Parameters
        ----------
        side : RAW_SIDE_BID or RAW_SIDE_ASK
        n_levels : number of price levels from best to include
        """
        b = self._book_for_side(side)
        if not b:
            return 0
        if side == RAW_SIDE_BID:
            # Best bid = highest price
            sorted_prices = sorted(b.keys(), reverse=True)
        else:
            # Best ask = lowest price
            sorted_prices = sorted(b.keys())
        total = 0
        for p in sorted_prices[:n_levels]:
            total += b[p].total_qty()
        return total

    def queue_position_at_back(self, side: int) -> int:
        """
        If you place a limit order at the current best price on `side`,
        how many contracts are ahead of you in the FIFO queue?

        This is the total qty at the best price level (you join the back).
        """
        if side == RAW_SIDE_BID:
            return self.best_bid_depth()
        else:
            return self.best_ask_depth()

    def book_imbalance(self) -> float:
        """
        (best_bid_depth - best_ask_depth) / (best_bid_depth + best_ask_depth).
        Returns 0.0 if either side is empty.
        """
        bd = self.best_bid_depth()
        ad = self.best_ask_depth()
        total = bd + ad
        if total == 0:
            return 0.0
        return (bd - ad) / total


# ─── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Order:
    """A resting limit order in the FIFO queue."""
    side: int          # +1=buy (at bid), -1=sell (at ask)
    price_rel: float   # price level (relative ticks from mid, e.g. -0.5 for bid)
    placed_ts_ns: int  # nanosecond timestamp when placed
    queue_position: float  # contracts ahead of us when we joined (from queue snapshot)
    volume_consumed: float = 0.0  # volume traded through our level since we joined


@dataclass
class Trade:
    """A completed trade (entry + exit)."""
    direction: int       # +1=long, -1=short
    entry_ts_ns: int
    exit_ts_ns: int
    entry_price_rel: float
    exit_price_rel: float
    pnl_ticks: float     # net of all costs
    hold_secs: float
    exit_reason: str     # 'tp', 'sl', 'market_exit', 'cancel_stale', 'eod'
    mfe_ticks: float     # max favorable excursion during trade
    mae_ticks: float     # max adverse excursion during trade


class RollingSortino:
    """
    Maintains a rolling buffer of trade PnLs and computes Sortino ratio.
    Sortino = mean(returns) / downside_deviation
    Downside deviation = std of negative returns only (MAR = 0).
    """

    def __init__(self, window: int = SORTINO_BUFFER_SIZE):
        self.window = window
        self._buf: deque = deque(maxlen=window)

    def add(self, pnl_ticks: float) -> None:
        self._buf.append(pnl_ticks)

    def compute(self) -> float:
        """Returns rolling Sortino, or 0.0 if insufficient data."""
        if len(self._buf) < MIN_TRADES_FOR_SORTINO:
            return 0.0
        arr = np.array(self._buf, dtype=np.float64)
        mean_r = arr.mean()
        downside = arr[arr < 0]
        if len(downside) == 0:
            return mean_r * 10.0  # All wins — return high value
        dd = np.sqrt(np.mean(downside ** 2))
        if dd < 1e-8:
            return mean_r * 10.0
        return mean_r / dd

    def win_rate(self) -> float:
        if len(self._buf) == 0:
            return 0.5
        arr = np.array(self._buf)
        return float((arr > 0).mean())

    def __len__(self) -> int:
        return len(self._buf)


# ─── Main Environment ────────────────────────────────────────────────────────────

class FIFOExecutionEnv:
    """
    FIFO-aware RL environment for ES futures execution.

    Episode = one trading day of MBO event data.
    Step = one MBO event processed (event-by-event replay).
    Agent can act or wait at each step.
    Fill happens when FIFO queue conditions are met.

    FIFO Queue Simulation Detail
    ----------------------------
    When a limit BUY at bid is placed:
      - We estimate queue_position = current bid depth (from book features)
      - Each subsequent trade event where side=-1 (selling hits bid) at our
        price level adds its qty to volume_consumed
      - When volume_consumed >= queue_position: ORDER FILLS
      - If price drops below our bid (moves away): ORDER STALE → cancel or
        use adverse selection signal

    For a limit SELL at ask:
      - Same logic but for ask side
      - Each subsequent trade where side=+1 (buying lifts ask) consumes queue
      - If price rises above our ask: ORDER STALE

    Adverse Selection
    -----------------
    Passive limit fills naturally suffer adverse selection: if price moved
    through you, it means the market was going against your direction.
    We model this by tracking whether the price move AFTER our fill is adverse.
    This is captured in the labels/predictions — no special treatment needed
    beyond accurate FIFO fill simulation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        event_dir: Path = EVENT_DIR,
        pred_dir: Path = PRED_DIR,
        patchtst_pred_dir: Optional[Path] = None,
        pred_window: int = 1000,   # events per CNN window
        pred_stride: int = 50,     # events between predictions
        max_steps_per_episode: int = 2_000_000,
        rth_only: bool = True,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        event_dir : directory containing dated *_mbo_events.npz files
        pred_dir  : directory containing fold_XX_oot_predictions.npz files (CNN-Mamba)
        patchtst_pred_dir : directory containing PatchTST fold_XX_oot_predictions.npz
        pred_window : sliding window size used to generate predictions (events)
        pred_stride : stride between prediction indices (events)
        max_steps_per_episode : hard cap on steps per episode
        rth_only : if True, only step during RTH (9:30–16:00 ET)
        seed : random seed for reproducibility
        verbose : print debug info
        """
        self.event_dir = Path(event_dir)
        self.pred_dir = Path(pred_dir)
        self.patchtst_pred_dir = Path(patchtst_pred_dir) if patchtst_pred_dir else None
        self.pred_window = pred_window
        self.pred_stride = pred_stride
        self.max_steps = max_steps_per_episode
        self.rth_only = rth_only
        self.verbose = verbose

        self.rng = np.random.default_rng(seed)

        # Discover available episode files
        self._event_files = sorted(self.event_dir.glob("20??????_mbo_events.npz"))
        if not self._event_files:
            raise FileNotFoundError(f"No MBO event files found in {self.event_dir}")
        log.info(f"Found {len(self._event_files)} episode files in {self.event_dir}")

        # ── Build date→file index for fast prediction lookup ──
        self._pred_index: Dict[str, Path] = {}       # date_str → pred file path (CNN-Mamba)
        self._patchtst_index: Dict[str, Path] = {}   # date_str → pred file path (PatchTST)
        self._build_pred_index()

        # Current episode state (set by reset())
        self._events: Optional[np.ndarray] = None       # (N, 25) float32 features
        self._et_raw: Optional[np.ndarray] = None       # (N,) int8 event types
        self._timestamps: Optional[np.ndarray] = None  # (N,) int64 nanoseconds
        self._preds: Optional[np.ndarray] = None        # (M, 3) float32 predictions
        self._labels: Optional[np.ndarray] = None       # (M, 3) float32 labels
        self._embeddings: Optional[np.ndarray] = None  # (M, E) float32 embeddings
        self._patchtst_preds: Optional[np.ndarray] = None  # (M, 3) PatchTST predictions

        # Raw MBO arrays for proper FIFO book reconstruction
        self._order_ids: Optional[np.ndarray] = None     # (N,) int64
        self._prices_raw: Optional[np.ndarray] = None    # (N,) int64 fixed-point
        self._sizes_raw: Optional[np.ndarray] = None     # (N,) int32
        self._sides_raw: Optional[np.ndarray] = None     # (N,) int8
        self._actions_raw: Optional[np.ndarray] = None   # (N,) int8
        self._has_raw_book_data: bool = False  # True if .npz has raw MBO arrays

        self._n_events: int = 0
        self._n_preds: int = 0
        self._step_idx: int = 0     # current raw event index
        self._pred_idx: int = 0     # current prediction index
        self._episode_start_ts: int = 0
        self._episode_date: str = ""

        # Position state
        self._position: int = 0             # 0=flat, +1=long, -1=short
        self._entry_price_rel: float = 0.0  # price_rel at entry
        self._entry_ts_ns: int = 0
        self._mfe_ticks: float = 0.0        # max favorable excursion
        self._mae_ticks: float = 0.0        # max adverse excursion

        # Pending order
        self._pending_order: Optional[Order] = None

        # Running price tracker (reconstruct mid from relative prices)
        self._mid_price: float = 0.0   # current estimated mid in relative space
        self._last_mid_update: int = 0

        # Flow tracking buffers (last 10s of data)
        self._buy_vol_10s: deque = deque()   # (ts_ns, vol) tuples
        self._sell_vol_10s: deque = deque()
        self._event_ts_10s: deque = deque()  # event timestamps for density
        self._cancel_ts_10s: deque = deque()

        # Running sums (maintained incrementally to avoid O(N) sum each step)
        self._buy_vol_sum: float = 0.0
        self._sell_vol_sum: float = 0.0

        # Volatility buffer (last 60s of mid-price changes in ticks)
        self._vol_buf_60s: deque = deque()   # (ts_ns, price_change_ticks)
        self._vol_sum_sq: float = 0.0        # running sum of squares for std
        self._price_history_10: deque = deque(maxlen=10)  # last 10 trade prices (rel)

        # ── HC #145: Real per-price-level FIFO book from MBO events ─────
        # Uses BookReconstructor when raw MBO arrays are available.
        # Falls back to legacy 2-float approximation for old .npz files.
        self._book = BookReconstructor()
        # Legacy fallback fields (used ONLY when _has_raw_book_data is False)
        self._bid_depth: float = 0.0
        self._ask_depth: float = 0.0
        self._book_initialized: bool = False

        # Trade / reward tracking
        self._trade_history: List[Trade] = []
        self._sortino_tracker = RollingSortino(SORTINO_BUFFER_SIZE)
        self._last_sortino: float = 0.0
        self._consecutive_losses: int = 0
        self._episode_pnl_ticks: float = 0.0
        self._episode_trades: int = 0

        # Overtrading detection
        self._trade_ts_history: deque = deque()  # timestamps of recent trades

        # PCA components for embedding compression (fitted lazily)
        self._pca_components: Optional[np.ndarray] = None  # (5, E)

        # Observation space spec (for gym compatibility)
        self.observation_space = _BoxSpace(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = _DiscreteSpace(7)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        episode_file: Optional[Path] = None,
        pred_file: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Start a new episode.

        Parameters
        ----------
        episode_file : specific MBO event file to use; if None, sample randomly
        pred_file    : specific fold prediction file to use; if None, auto-discover

        Returns
        -------
        obs : np.ndarray (OBS_DIM,) initial observation
        """
        # Load event data
        if episode_file is None:
            episode_file = self.rng.choice(self._event_files)
        self._episode_date = episode_file.stem.replace("_mbo_events", "")

        if self.verbose:
            log.info(f"Loading episode: {episode_file.name}")

        # Try to load; if corrupt (BadZipFile or other error), skip and pick another
        _max_retries = 10
        for _attempt in range(_max_retries):
            try:
                data = np.load(episode_file)
                _ = data["events"]  # force zip read so corrupt files raise here
                break
            except Exception as _e:
                log.warning(f"Corrupt/unreadable episode file (attempt {_attempt+1}): "
                            f"{episode_file.name} — {_e}. Picking another.")
                episode_file = self.rng.choice(self._event_files)
                self._episode_date = episode_file.stem.replace("_mbo_events", "")
        else:
            raise RuntimeError(f"Could not find a valid episode file after {_max_retries} attempts.")

        self._events = data["events"]          # (N, 25) float32
        self._et_raw = data["event_type_raw"]  # (N,) int8
        self._timestamps = data["timestamps"]  # (N,) int64 nanoseconds
        self._n_events = len(self._events)
        self._episode_start_ts = int(self._timestamps[0])

        # Load raw MBO arrays for proper FIFO book reconstruction
        if "order_ids" in data and "prices_raw" in data and "actions_raw" in data:
            self._order_ids = data["order_ids"]       # (N,) int64
            self._prices_raw = data["prices_raw"]     # (N,) int64
            self._sizes_raw = data["sizes_raw"]       # (N,) int32
            self._sides_raw = data["sides_raw"]       # (N,) int8
            self._actions_raw = data["actions_raw"]   # (N,) int8
            self._has_raw_book_data = True
            if self.verbose:
                log.info(f"Loaded raw MBO arrays for FIFO book reconstruction ({self._n_events} events)")
        else:
            self._order_ids = None
            self._prices_raw = None
            self._sizes_raw = None
            self._sides_raw = None
            self._actions_raw = None
            self._has_raw_book_data = False
            log.warning(f"No raw MBO arrays in {episode_file.name} — falling back to legacy book tracking")

        # Load CNN-Mamba predictions if available
        self._preds = None
        self._labels = None
        self._embeddings = None
        self._patchtst_preds = None
        pred_path = self._find_pred_file(self._episode_date, pred_file)
        if pred_path is not None:
            try:
                pdata = np.load(pred_path, allow_pickle=True)
                self._preds = pdata["predictions"].astype(np.float32)     # (M, 3)
                self._labels = pdata["labels"].astype(np.float32)         # (M, 3)
                if "embeddings" in pdata:
                    self._embeddings = pdata["embeddings"].astype(np.float32)
                    self._maybe_fit_pca()
                self._n_preds = len(self._preds)
                if self.verbose:
                    log.info(f"Loaded {self._n_preds} predictions from {pred_path.name}")
            except Exception as e:
                log.warning(f"Could not load predictions from {pred_path}: {e}")
                self._preds = None

        # Load PatchTST predictions for confluence (HC #111, #108)
        if self.patchtst_pred_dir is not None:
            pst_path = self._find_pred_in_dir(self._episode_date, self.patchtst_pred_dir)
            if pst_path is not None:
                try:
                    pst_data = np.load(pst_path, allow_pickle=True)
                    pst_preds = pst_data["predictions"].astype(np.float32)
                    # Align to CNN-Mamba pred count (take min length)
                    if self._preds is not None:
                        n_align = min(len(pst_preds), self._n_preds)
                        self._patchtst_preds = pst_preds[:n_align]
                    else:
                        self._patchtst_preds = pst_preds
                    log.info(f"PatchTST confluence: {len(self._patchtst_preds)} preds from {pst_path.name} for {self._episode_date}")
                except Exception as e:
                    log.warning(f"Could not load PatchTST predictions: {e}")
                    self._patchtst_preds = None
            else:
                log.debug(f"No PatchTST predictions found for date {self._episode_date}")

        # Reset internal state
        self._step_idx = 0
        self._pred_idx = 0
        self._position = 0
        self._entry_price_rel = 0.0
        self._entry_ts_ns = 0
        self._mfe_ticks = 0.0
        self._mae_ticks = 0.0
        self._pending_order = None
        self._mid_price = 0.0
        self._last_mid_update = 0
        # HC #114: Track signal at entry for alignment and decay rewards
        self._entry_signal_1s = 0.0   # pred_1s at time of entry
        self._entry_signal_5s = 0.0   # pred_5s at time of entry
        self._entry_signal_10s = 0.0  # pred_10s at time of entry

        self._buy_vol_10s.clear()
        self._sell_vol_10s.clear()
        self._event_ts_10s.clear()
        self._cancel_ts_10s.clear()
        self._buy_vol_sum = 0.0
        self._sell_vol_sum = 0.0
        self._vol_buf_60s.clear()
        self._vol_sum_sq = 0.0
        self._price_history_10.clear()

        # HC #145: Reset book state
        self._book.reset()
        self._bid_depth = 0.0
        self._ask_depth = 0.0
        self._book_initialized = False

        self._trade_history = []
        self._last_closed_trade = None  # HC #221: for per-head reward computation
        self._last_entry_signal_1s = 0.0
        self._sortino_tracker = RollingSortino(SORTINO_BUFFER_SIZE)
        self._last_sortino = 0.0
        self._consecutive_losses = 0
        self._episode_pnl_ticks = 0.0
        self._episode_trades = 0
        self._trade_ts_history.clear()

        # Skip to first valid (RTH) event
        if self.rth_only:
            self._step_idx = self._find_rth_start()

        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Process one MBO event and optionally take an action.

        Parameters
        ----------
        action : int in {0..6}

        Returns
        -------
        obs    : np.ndarray (OBS_DIM,)
        reward : float
        done   : bool
        info   : dict with diagnostic information
        """
        if self._events is None:
            raise RuntimeError("Call reset() before step()")

        # ── Advance event pointer ──────────────────────────────────────────────
        if self._step_idx >= self._n_events:
            return self._get_obs(), 0.0, True, self._get_info()

        evt = self._events[self._step_idx]
        et  = int(self._et_raw[self._step_idx])
        ts  = int(self._timestamps[self._step_idx])

        # ── Update flow trackers ───────────────────────────────────��───────────
        self._update_flow_trackers(et, evt, ts)

        # ── HC #145: Update real L1 book depth from MBO events ────────────
        self._update_book_depth(et, evt)

        # ── Update prediction index ────────────────────────────────────────────
        if self._preds is not None:
            pred_idx_for_step = (self._step_idx - self.pred_window + 1) // self.pred_stride
            if pred_idx_for_step > self._pred_idx and pred_idx_for_step < self._n_preds:
                self._pred_idx = pred_idx_for_step

        # ── Check pending order fill ───────────────────────────────────────────
        fill_reward = 0.0
        if self._pending_order is not None and et == EVENT_TRADE:
            fill_reward = self._process_trade_event_for_fill(evt, ts)

        # ── Check if pending order is stale (price moved away) ────────────────
        stale_reward = 0.0
        if self._pending_order is not None:
            stale_reward = self._check_order_staleness(evt, ts)

        # ── Update MFE/MAE for open position ──────────────────────────────────
        if self._position != 0:
            self._update_mfe_mae(evt)

        # ── Apply agent action ────────────────────────────────────────────────
        action_reward = self._apply_action(action, evt, ts)

        # ── Advance to next prediction step ───────────────────────────────────
        self._step_idx += 1

        # ── Check end conditions ───────────────────────────────────────────────
        done = False
        eod_reward = 0.0
        if self._step_idx >= self._n_events:
            done = True
            if self._position != 0:
                eod_reward = self._force_close_position(ts, reason="eod")
        elif self._step_idx >= self.max_steps:
            done = True
        elif self.rth_only and self._is_after_rth(ts):
            done = True
            if self._position != 0:
                eod_reward = self._force_close_position(ts, reason="eod")

        total_reward = fill_reward + stale_reward + action_reward + eod_reward
        obs = self._get_obs()
        info = self._get_info()

        return obs, float(total_reward), done, info

    # ------------------------------------------------------------------
    # FIFO Fill Simulation
    # ------------------------------------------------------------------

    def _process_trade_event_for_fill(self, evt: np.ndarray, ts: int) -> float:
        """
        Check if a trade event fills our pending limit order.

        FIFO Queue Logic:
        - Our order joined the back of the queue at price level P
        - We recorded queue_position = volume ahead of us at that time
        - Each trade at our price level (or through it) consumes volume
        - When volume_consumed >= queue_position: WE FILL

        Returns reward from the fill (if any).
        """
        order = self._pending_order
        trade_side = float(evt[COL_SIDE])     # +1=buy, -1=sell
        trade_price_rel = float(evt[COL_PRICE_REL])  # relative to mid
        trade_qty = math.exp(float(evt[COL_QTY_LOG]))  # actual contracts

        # Check if this trade is at our order's price level
        # For a BUY limit at bid (side=+1): we fill when market sells hit our level
        #   → trade_side == -1 (seller) at price_rel == our bid level
        # For a SELL limit at ask (side=-1): we fill when market buys lift our level
        #   → trade_side == +1 (buyer) at price_rel == our ask level
        at_our_level = False
        if order.side == 1:  # buy limit at bid
            # Fill when sellers trade at or through our bid level
            # price_rel for bid level is negative (below mid), ask is positive
            # our bid: price_rel ≈ -spread/2 ≈ -0.5 ticks
            at_our_level = (
                trade_side == -1  # someone is selling (hitting bid)
                and abs(trade_price_rel - order.price_rel) < 0.1  # at our level ±0.1
            )
        else:  # sell limit at ask
            # Fill when buyers trade at or through our ask level
            at_our_level = (
                trade_side == 1   # someone is buying (lifting ask)
                and abs(trade_price_rel - order.price_rel) < 0.1  # at our level ±0.1
            )

        if at_our_level:
            order.volume_consumed += trade_qty

        # FILL condition: consumed volume has worked through our queue position
        if order.volume_consumed >= order.queue_position:
            return self._execute_fill(order, ts, fill_type="limit")

        return 0.0

    def _execute_fill(self, order: Order, ts: int, fill_type: str) -> float:
        """
        Execute an order fill: enter or exit position.

        Returns reward if this closes a position.
        """
        current_price_rel = self._events[min(self._step_idx, self._n_events - 1)][COL_PRICE_REL]

        if self._position == 0:
            # ── Opening a new position ─────────────────────────────────────────
            self._position = order.side
            self._entry_price_rel = order.price_rel  # filled at our limit price
            self._entry_ts_ns = ts
            self._mfe_ticks = 0.0
            self._mae_ticks = 0.0
            self._pending_order = None

            # HC #114: Capture signal at entry for alignment/decay tracking
            if self._preds is not None and self._pred_idx < self._n_preds:
                p = self._preds[self._pred_idx]
                self._entry_signal_1s = float(p[0])
                self._entry_signal_5s = float(p[1])
                self._entry_signal_10s = float(p[2])
            else:
                self._entry_signal_1s = 0.0
                self._entry_signal_5s = 0.0
                self._entry_signal_10s = 0.0

            if self.verbose:
                direction = "LONG" if order.side == 1 else "SHORT"
                log.debug(f"  FILL {fill_type}: opened {direction} at rel={order.price_rel:.3f} "
                          f"signal_1s={self._entry_signal_1s:.3f}")
            return 0.0

        else:
            # ── Closing an existing position ──────────────────────────────────
            direction = self._position
            exit_price_rel = order.price_rel if fill_type == "limit" else current_price_rel

            # PnL calculation — HC #127: only commission, NO spread cost
            # raw_move already reflects actual fill prices (bid/ask), spread is captured there
            raw_move = (exit_price_rel - self._entry_price_rel) * direction
            cost = COMMISSION_COST  # 0.376 ticks RT, regardless of order type

            pnl_ticks = raw_move - cost
            hold_secs = (ts - self._entry_ts_ns) / 1e9

            trade = Trade(
                direction=direction,
                entry_ts_ns=self._entry_ts_ns,
                exit_ts_ns=ts,
                entry_price_rel=self._entry_price_rel,
                exit_price_rel=exit_price_rel,
                pnl_ticks=pnl_ticks,
                hold_secs=hold_secs,
                exit_reason=fill_type,
                mfe_ticks=self._mfe_ticks,
                mae_ticks=self._mae_ticks,
            )
            self._trade_history.append(trade)
            self._episode_pnl_ticks += pnl_ticks
            self._episode_trades += 1
            self._trade_ts_history.append(ts)

            # HC #221: Store closed trade for info dict (per-head reward computation)
            self._last_closed_trade = trade
            self._last_entry_signal_1s = self._entry_signal_1s

            # Compute reward = delta Sortino
            sortino_before = self._sortino_tracker.compute()
            self._sortino_tracker.add(pnl_ticks)
            sortino_after = self._sortino_tracker.compute()
            delta_sortino = sortino_after - sortino_before

            # Track consecutive losses
            if pnl_ticks < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            # Reward shaping
            reward = delta_sortino

            # HC #104: NO short-side bias — any win is a win
            # (SHORT_EDGE_BONUS set to 0.0)

            # ── HC #114: Signal-Alignment Reward (SOFT guidance) ────────────────
            # Mild nudge toward trading with alpha. NOT a hard constraint.
            # The model has obs[45:48] showing signal decay/alignment as FEATURES
            # and must learn timing/exits on its own. We just gently reward
            # signal-aligned entries and gently discourage anti-signal entries.
            entry_signal = self._entry_signal_1s
            signal_mag = abs(entry_signal)
            if signal_mag > ALPHA_GATE_THRESHOLD:
                signal_sign = 1.0 if entry_signal > 0 else -1.0
                if direction == signal_sign:
                    # Trade aligned with alpha — small bonus
                    reward += SIGNAL_ALIGNMENT_BONUS * min(signal_mag, 1.0)
                else:
                    # Trade against alpha — mild discouragement
                    reward -= SIGNAL_MISALIGNMENT_PENALTY * min(signal_mag, 1.0)
            # NOTE: No penalty for weak-signal entries here — the entry gate
            # already handles that. Let the model learn if low-signal trades
            # can still be profitable through other features.

            # HC #114: NO hardcoded signal-decay hold penalty.
            # The model receives signal_remaining as obs[45] and must learn
            # when to exit on its own. We don't impose preconceived decay timing.

            # Penalty: consecutive losses (kicks in after 3rd)
            if self._consecutive_losses > 3:
                reward -= CONSECUTIVE_LOSS_PENALTY * (self._consecutive_losses - 3)

            # Penalty: excessive hold time (original 30s penalty — reasonable safeguard)
            if hold_secs > MAX_HOLD_SECS:
                overage = hold_secs - MAX_HOLD_SECS
                reward -= 0.01 * min(overage, 30.0)  # cap penalty

            # Penalty: overtrading (> OVERTRADING_LIMIT trades in last 60s)
            self._prune_trade_history_deque(ts)
            if len(self._trade_ts_history) > OVERTRADING_LIMIT:
                reward -= 0.05 * (len(self._trade_ts_history) - OVERTRADING_LIMIT)

            # Reset position state
            self._position = 0
            self._entry_price_rel = 0.0
            self._mfe_ticks = 0.0
            self._mae_ticks = 0.0
            self._pending_order = None
            self._last_sortino = sortino_after

            if self.verbose:
                log.debug(
                    f"  CLOSE {fill_type}: pnl={pnl_ticks:.3f} ticks "
                    f"hold={hold_secs:.1f}s reward={reward:.4f}"
                )

            return reward

    def _check_order_staleness(self, evt: np.ndarray, ts: int) -> float:
        """
        Cancel a pending limit order if price has moved away.

        If our BUY limit at bid is stalened (best bid moved down),
        the order is cancelled — no fill, no cost.
        """
        order = self._pending_order
        current_price_rel = float(evt[COL_PRICE_REL])
        spread = float(evt[COL_SPREAD])

        # Estimate current bid and ask levels
        # bid ≈ mid - spread/2, ask ≈ mid + spread/2
        half_spread = max(spread / 2.0, 0.5)
        current_bid_level = -half_spread
        current_ask_level = +half_spread

        stale = False
        if order.side == 1:  # buy limit at bid
            # Stale if best bid is now BELOW our order price (we're away from market)
            if current_bid_level < order.price_rel - 0.5:
                stale = True
        else:  # sell limit at ask
            # Stale if best ask is now ABOVE our order price
            if current_ask_level > order.price_rel + 0.5:
                stale = True

        # Also stale if held too long without fill (> 2x max hold time)
        hold_so_far = (ts - order.placed_ts_ns) / 1e9
        if hold_so_far > MAX_HOLD_SECS * 2:
            stale = True

        if stale:
            self._pending_order = None
            # No reward, no cost — just cancellation
            return 0.0

        return 0.0

    def _update_book_depth(self, et: int, evt: np.ndarray) -> None:
        """
        HC #145: Update book state from the current MBO event.

        Uses the proper BookReconstructor with per-price-level FIFO queues
        when raw MBO arrays are available. Falls back to the legacy 2-float
        approximation for old .npz files without raw arrays.
        """
        if self._has_raw_book_data:
            # ── Real FIFO book reconstruction from raw MBO arrays ────────
            idx = self._step_idx
            action = int(self._actions_raw[idx])
            side = int(self._sides_raw[idx])
            price_raw = int(self._prices_raw[idx])
            size = int(self._sizes_raw[idx])
            order_id = int(self._order_ids[idx])
            self._book.process_event(action, side, price_raw, size, order_id)
        else:
            # ── Legacy fallback: 2-float approximation with decay ────────
            side = float(evt[COL_SIDE])
            qty = math.exp(float(evt[COL_QTY_LOG]))

            if et == EVENT_ADD:
                if side > 0:
                    self._bid_depth += qty
                elif side < 0:
                    self._ask_depth += qty
            elif et == EVENT_CANCEL:
                if side > 0:
                    self._bid_depth = max(0.0, self._bid_depth - qty)
                elif side < 0:
                    self._ask_depth = max(0.0, self._ask_depth - qty)
            elif et == EVENT_TRADE:
                if side > 0:
                    self._ask_depth = max(0.0, self._ask_depth - qty)
                elif side < 0:
                    self._bid_depth = max(0.0, self._bid_depth - qty)
            elif et == EVENT_CLEAR:
                self._bid_depth = 0.0
                self._ask_depth = 0.0

            if self._step_idx % 1000 == 0 and self._step_idx > 0:
                decay = 0.87
                self._bid_depth *= decay
                self._ask_depth *= decay

            if not self._book_initialized and self._step_idx > 500:
                self._book_initialized = True

    def _get_queue_depth(self, side: int) -> float:
        """
        HC #145: Get queue depth at best price level for the given side.

        Parameters
        ----------
        side : +1 for buy (bid depth) or -1 for sell (ask depth)

        Returns
        -------
        Queue depth in contracts at the best price level. This is the number
        of contracts ahead of a new limit order placed at the current best.
        """
        if self._has_raw_book_data:
            # ── Real FIFO book ────────────────────────────────────────────
            if not self._book.initialized:
                return 0.0  # Book warming up, no fallback constant
            book_side = RAW_SIDE_BID if side == 1 else RAW_SIDE_ASK
            depth = self._book.queue_position_at_back(book_side)
            return max(float(MIN_QUEUE_DEPTH), float(depth))
        else:
            # ── Legacy fallback ───────────────────────────────────────────
            if not self._book_initialized:
                return float(DEFAULT_QUEUE_DEPTH)
            if side == 1:
                depth = self._bid_depth
            else:
                depth = self._ask_depth
            return max(MIN_QUEUE_DEPTH, depth)

    def _get_book_depth_for_obs(self) -> Tuple[float, float, float]:
        """
        Get bid depth, ask depth, and book imbalance for the observation vector.

        Returns
        -------
        (bid_depth, ask_depth, imbalance) where depths are raw contract counts
        and imbalance is in [-1, 1].
        """
        if self._has_raw_book_data and self._book.initialized:
            bd = float(self._book.best_bid_depth())
            ad = float(self._book.best_ask_depth())
            imbalance = self._book.book_imbalance()
            return bd, ad, imbalance
        elif not self._has_raw_book_data and self._book_initialized:
            bd = self._bid_depth
            ad = self._ask_depth
            total = bd + ad
            imbalance = (bd - ad) / total if total > 1e-8 else 0.0
            return bd, ad, imbalance
        else:
            return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # Action Application
    # ------------------------------------------------------------------

    def _apply_action(self, action: int, evt: np.ndarray, ts: int) -> float:
        """
        Apply the agent's chosen action.

        Returns immediate reward (0 for most actions; non-zero for market orders
        that close positions immediately).
        """
        spread = float(evt[COL_SPREAD])
        half_spread = max(spread / 2.0, 0.5)

        # ── HC #114: Alpha-Gating for Entry Actions ──────────────────────────
        # Entry actions (1-4) require meaningful alpha signal.
        # Without signal support, entries are penalized (not hard-blocked,
        # so the agent can still learn but is strongly discouraged).
        entry_penalty = 0.0
        if action in (1, 2, 3, 4) and self._position == 0:
            current_signal = 0.0
            if self._preds is not None and self._pred_idx < self._n_preds:
                current_signal = float(self._preds[self._pred_idx][0])  # pred_1s
            signal_mag = abs(current_signal)
            if signal_mag < ALPHA_GATE_THRESHOLD:
                # Weak/no signal — penalize entry attempt
                entry_penalty = ALPHA_GATE_PENALTY
            else:
                # Check direction alignment: buy (1,3) needs positive signal, sell (2,4) needs negative
                wants_long = action in (1, 3)
                signal_supports_long = current_signal > 0
                if wants_long != signal_supports_long:
                    # Trying to enter AGAINST the alpha — stronger penalty
                    entry_penalty = SIGNAL_MISALIGNMENT_PENALTY * min(signal_mag, 1.0)

        # Action 0: Do nothing
        if action == 0:
            return 0.0

        # Action 1: Place limit BUY at bid
        elif action == 1:
            if self._position == 0 and self._pending_order is None:
                bid_level = -half_spread
                queue_depth = self._get_queue_depth(1)
                self._pending_order = Order(
                    side=1,
                    price_rel=bid_level,
                    placed_ts_ns=ts,
                    queue_position=queue_depth,
                )
                if self.verbose:
                    log.debug(f"  ACTION: limit BUY at bid={bid_level:.2f}, queue={queue_depth:.0f}")
            return entry_penalty  # HC #114: penalize entries without alpha support

        # Action 2: Place limit SELL at ask
        elif action == 2:
            if self._position == 0 and self._pending_order is None:
                ask_level = +half_spread
                queue_depth = self._get_queue_depth(-1)
                self._pending_order = Order(
                    side=-1,
                    price_rel=ask_level,
                    placed_ts_ns=ts,
                    queue_position=queue_depth,
                )
                if self.verbose:
                    log.debug(f"  ACTION: limit SELL at ask={ask_level:.2f}, queue={queue_depth:.0f}")
            return entry_penalty  # HC #114: penalize entries without alpha support

        # Action 3: Market BUY (cross spread, immediately long)
        elif action == 3:
            if self._position == 0 and self._pending_order is None:
                ask_level = +half_spread
                # Create synthetic "filled" order
                order = Order(side=1, price_rel=ask_level, placed_ts_ns=ts, queue_position=0)
                order.volume_consumed = 1e9  # instantly filled
                self._execute_fill(order, ts, fill_type="market_entry")
            return entry_penalty  # HC #114: penalize entries without alpha support

        # Action 4: Market SELL (cross spread, immediately short)
        elif action == 4:
            if self._position == 0 and self._pending_order is None:
                bid_level = -half_spread
                order = Order(side=-1, price_rel=bid_level, placed_ts_ns=ts, queue_position=0)
                order.volume_consumed = 1e9
                self._execute_fill(order, ts, fill_type="market_entry")
            return entry_penalty  # HC #114: penalize entries without alpha support

        # Action 5: Cancel pending order
        elif action == 5:
            if self._pending_order is not None:
                self._pending_order = None
                if self.verbose:
                    log.debug("  ACTION: cancelled pending order")
            return 0.0

        # Action 6: Market exit (close position)
        elif action == 6:
            if self._position != 0:
                return self._force_close_position(ts, reason="market_exit")
            return 0.0

        return 0.0

    def _force_close_position(self, ts: int, reason: str) -> float:
        """Force-close an open position via market order."""
        if self._position == 0:
            return 0.0

        evt = self._events[min(self._step_idx, self._n_events - 1)]
        spread = float(evt[COL_SPREAD])
        half_spread = max(spread / 2.0, 0.5)
        current_price_rel = float(evt[COL_PRICE_REL])

        # Exit price: cross spread (we take liquidity)
        if self._position == 1:  # long → sell at bid
            exit_price_rel = current_price_rel - half_spread
        else:  # short → buy at ask
            exit_price_rel = current_price_rel + half_spread

        fake_order = Order(
            side=-self._position,
            price_rel=exit_price_rel,
            placed_ts_ns=ts,
            queue_position=0,
        )
        fake_order.volume_consumed = 1e9
        return self._execute_fill(fake_order, ts, fill_type=reason)

    # ------------------------------------------------------------------
    # Observation Construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Construct the 45-dim observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if self._step_idx >= self._n_events:
            return obs

        evt = self._events[self._step_idx]
        ts  = int(self._timestamps[self._step_idx])

        # ── [0:4] Signal features ──────────────────────────────────────────────
        pred_1s = pred_5s = pred_10s = 0.0
        confidence_tier = 0.0
        if self._preds is not None and self._pred_idx < self._n_preds:
            p = self._preds[self._pred_idx]
            pred_1s, pred_5s, pred_10s = float(p[0]), float(p[1]), float(p[2])
            # Confidence tier: 0=low, 1=mid, 2=high, 3=very high
            mag = abs(pred_1s)
            confidence_tier = float(min(3, int(mag / 0.25)))  # tier up every 0.25 units

        # Use inline clamp (max/min) instead of np.clip for scalars — much faster
        def clamp(x, lo, hi):
            return lo if x < lo else (hi if x > hi else x)

        obs[0] = clamp(pred_1s, -5.0, 5.0)
        obs[1] = clamp(pred_5s, -5.0, 5.0)
        obs[2] = clamp(pred_10s, -5.0, 5.0)
        obs[3] = confidence_tier / 3.0  # normalize to [0,1]

        # ── [4:8] Book state ───────────────────────────────────────────────────
        spread = float(evt[COL_SPREAD])
        half_spread = max(spread / 2.0, 0.5)

        # Real book depth from FIFO reconstruction (or legacy fallback)
        bid_depth, ask_depth, book_imbalance = self._get_book_depth_for_obs()
        bid_depth_log = math.log1p(bid_depth)
        ask_depth_log = math.log1p(ask_depth)

        # Also get volume sums for flow features used later
        buy_vol, sell_vol = self._get_10s_volumes()
        total_vol = buy_vol + sell_vol + 1e-8

        obs[4] = clamp(bid_depth_log / 5.0, -3.0, 3.0)
        obs[5] = clamp(ask_depth_log / 5.0, -3.0, 3.0)
        obs[6] = clamp(book_imbalance, -1.0, 1.0)
        obs[7] = clamp(spread / 4.0, 0.0, 2.0)

        # ── [8:10] Price features ──────────────────────────────────────────────
        price_rel = float(evt[COL_PRICE_REL])
        obs[8] = clamp(price_rel / 2.0, -1.0, 1.0)
        obs[9] = self._get_price_momentum()

        # ── [10:14] Position state ─────────────────────────────────────────────
        obs[10] = float(self._position)  # 0, +1, -1
        if self._position != 0:
            unrealized = (price_rel - self._entry_price_rel) * self._position
            hold_secs = (ts - self._entry_ts_ns) * 1e-9
            obs[11] = clamp(unrealized / 10.0, -3.0, 3.0)
            obs[12] = clamp(hold_secs / MAX_HOLD_SECS, 0.0, 3.0)
        else:
            obs[11] = 0.0
            obs[12] = 0.0

        # Queue fill fraction for pending order
        if self._pending_order is not None:
            order = self._pending_order
            queue_frac = min(1.0, order.volume_consumed / max(order.queue_position, 1.0))
            obs[13] = queue_frac
        else:
            obs[13] = 0.0

        # ── [14:16] MFE/MAE ───────────────────────────────────────────────────
        obs[14] = clamp(self._mfe_ticks / 10.0, 0.0, 3.0)
        obs[15] = clamp(self._mae_ticks / 10.0, 0.0, 3.0)

        # ── [16:19] Market context ─────────────────────────────────────────────
        obs[16] = clamp(self._get_realized_vol_60s() / 5.0, 0.0, 3.0)

        # Time of day: seconds since midnight as sin/cos encoding
        tod_sec = self._get_tod_seconds(ts)
        tod_frac = tod_sec / 86400.0
        obs[17] = math.sin(2.0 * math.pi * tod_frac)
        obs[18] = math.cos(2.0 * math.pi * tod_frac)

        # ── [19:24] Flow features ─────────────────────────────────────────────
        event_density = self._get_local_event_rate()
        obs[19] = clamp(event_density / 3.0, 0.0, 3.0)
        obs[20] = buy_vol / total_vol  # buy fraction [0,1]
        obs[21] = clamp(book_imbalance, -1.0, 1.0)
        cancel_rate = self._get_cancel_rate()
        obs[22] = clamp(cancel_rate / 3.0, 0.0, 3.0)
        obs[23] = clamp(event_density, 0.0, 5.0)
        obs[24] = 0.0  # reserved

        # ── [24:29] Trade history: last 5 PnLs ────────────────────────────────
        hist = self._trade_history
        n_hist = len(hist)
        for i in range(5):
            # Fill from most recent, zero-pad older slots
            idx = n_hist - 5 + i
            pnl = hist[idx].pnl_ticks if 0 <= idx < n_hist else 0.0
            obs[24 + i] = clamp(pnl / 10.0, -3.0, 3.0)

        # ── [29:32] Rolling stats ──────────────────────────────────────────────
        obs[29] = self._sortino_tracker.win_rate()
        sortino = self._sortino_tracker.compute()
        obs[30] = clamp(sortino / 5.0, -3.0, 3.0)
        obs[31] = clamp(float(self._consecutive_losses) / 5.0, 0.0, 3.0)

        # ── [32:34] Order state ────────────────────────────────────────────────
        obs[32] = 1.0 if self._pending_order is not None else 0.0
        obs[33] = float(self._pending_order.side) if self._pending_order is not None else 0.0

        # ── [34:39] CNN-Mamba embedding (top-5 PCA) ───────────────────────────
        if (self._embeddings is not None
                and self._pred_idx < len(self._embeddings)
                and self._pca_components is not None):
            emb = self._embeddings[self._pred_idx]  # (E,)
            pca_proj = self._pca_components @ emb    # (5,)
            obs[34:39] = np.clip(pca_proj / 3.0, -3.0, 3.0)

        # ── [39:45] PatchTST confluence + enhanced features (HC #111, #108) ──
        if (self._patchtst_preds is not None
                and self._pred_idx < len(self._patchtst_preds)):
            pst = self._patchtst_preds[self._pred_idx]
            pst_1s, pst_5s, pst_10s = float(pst[0]), float(pst[1]), float(pst[2])

            obs[39] = clamp(pst_1s, -5.0, 5.0)
            obs[40] = clamp(pst_5s, -5.0, 5.0)
            obs[41] = clamp(pst_10s, -5.0, 5.0)

            # Confluence: CNN-Mamba and PatchTST agreement score
            # +1 = both agree strongly on direction, -1 = disagree
            if abs(pred_1s) > 0.01 and abs(pst_1s) > 0.01:
                # Sign agreement × geometric mean of magnitudes (normalized)
                sign_agree = 1.0 if (pred_1s * pst_1s > 0) else -1.0
                mag = math.sqrt(abs(pred_1s) * abs(pst_1s))
                obs[42] = clamp(sign_agree * mag, -3.0, 3.0)
            else:
                obs[42] = 0.0

            # Confluence across all 3 horizons (avg sign agreement)
            agree_sum = 0.0
            for cm, ps in [(pred_1s, pst_1s), (pred_5s, pst_5s), (pred_10s, pst_10s)]:
                if abs(cm) > 0.01 and abs(ps) > 0.01:
                    agree_sum += 1.0 if (cm * ps > 0) else -1.0
            obs[43] = agree_sum / 3.0  # [-1, 1] avg agreement

        # Sweep intensity: ratio of trades to total events in last 10s
        n_trades_10s = sum(1 for ts_val in self._event_ts_10s
                          if abs(ts_val) < 1e18)  # placeholder
        obs[44] = clamp(self._get_local_event_rate() *
                        (buy_vol + sell_vol) / (total_vol * 10.0 + 1e-8), 0.0, 3.0)

        # ── [45:48] HC #114: Alpha-awareness features ─────────────────────────
        # These tell the agent explicitly how much signal support remains for
        # the current position, so it learns to exit when alpha decays.
        if self._position != 0:
            hold_secs = (ts - self._entry_ts_ns) * 1e-9

            # [45] Signal remaining: exponential decay from entry signal
            # At entry: 1.0, after 250ms: 0.5, after 1s: ~0.06, after 2s: ~0.004
            signal_remaining = math.exp(-0.693 * hold_secs / SIGNAL_DECAY_HALFLIFE_S)
            obs[45] = clamp(signal_remaining, 0.0, 1.0)

            # [46] Entry signal strength (how strong was the reason to enter)
            obs[46] = clamp(abs(self._entry_signal_1s), 0.0, 3.0)

            # [47] Current alpha alignment with position
            # Positive = current signal still supports position, negative = signal reversed
            if self._preds is not None and self._pred_idx < self._n_preds:
                current_pred = float(self._preds[self._pred_idx][0])
                obs[47] = clamp(current_pred * self._position, -3.0, 3.0)
            else:
                obs[47] = 0.0
        else:
            obs[45] = 0.0  # no position → no decay tracking
            obs[46] = 0.0
            obs[47] = 0.0

        return obs

    # ------------------------------------------------------------------
    # Flow / Market State Helpers
    # ------------------------------------------------------------------

    def _update_flow_trackers(self, et: int, evt: np.ndarray, ts: int) -> None:
        """
        Update 10s rolling volume and event counters.
        Maintains running sums for O(1) volume queries.
        """
        cutoff_10s = ts - 10_000_000_000  # 10 seconds in nanoseconds
        cutoff_60s = ts - 60_000_000_000  # 60 seconds

        qty = math.exp(float(evt[COL_QTY_LOG]))
        side = float(evt[COL_SIDE])
        price_rel = float(evt[COL_PRICE_REL])

        # Update 10s volume buffers with running sums
        if et == EVENT_TRADE:
            if side > 0:
                self._buy_vol_10s.append((ts, qty))
                self._buy_vol_sum += qty
            else:
                self._sell_vol_10s.append((ts, qty))
                self._sell_vol_sum += qty
            # Update price history for momentum
            self._price_history_10.append(price_rel)

        # Update event density buffer
        self._event_ts_10s.append(ts)

        # Update cancel buffer
        if et == EVENT_CANCEL:
            self._cancel_ts_10s.append(ts)

        # Update volatility buffer with running sum of squares
        if et == EVENT_TRADE and len(self._price_history_10) >= 2:
            ph = list(self._price_history_10)
            price_change = abs(price_rel - ph[-2])
            self._vol_buf_60s.append((ts, price_change))
            self._vol_sum_sq += price_change * price_change

        # Prune old entries from all buffers (subtract from running sums)
        while self._buy_vol_10s and self._buy_vol_10s[0][0] < cutoff_10s:
            self._buy_vol_sum -= self._buy_vol_10s.popleft()[1]
        while self._sell_vol_10s and self._sell_vol_10s[0][0] < cutoff_10s:
            self._sell_vol_sum -= self._sell_vol_10s.popleft()[1]
        while self._event_ts_10s and self._event_ts_10s[0] < cutoff_10s:
            self._event_ts_10s.popleft()
        while self._cancel_ts_10s and self._cancel_ts_10s[0] < cutoff_10s:
            self._cancel_ts_10s.popleft()
        while self._vol_buf_60s and self._vol_buf_60s[0][0] < cutoff_60s:
            self._vol_sum_sq -= self._vol_buf_60s.popleft()[1] ** 2

        # Clamp running sums to avoid negative drift from float precision
        self._buy_vol_sum  = max(0.0, self._buy_vol_sum)
        self._sell_vol_sum = max(0.0, self._sell_vol_sum)
        self._vol_sum_sq   = max(0.0, self._vol_sum_sq)

    def _update_mfe_mae(self, evt: np.ndarray) -> None:
        """Update max favorable / adverse excursion for open position."""
        if self._position == 0:
            return
        price_rel = float(evt[COL_PRICE_REL])
        unrealized = (price_rel - self._entry_price_rel) * self._position
        if unrealized > self._mfe_ticks:
            self._mfe_ticks = unrealized
        if unrealized < -self._mae_ticks:
            self._mae_ticks = -unrealized

    def _get_10s_volumes(self) -> Tuple[float, float]:
        """O(1) — returns running sums maintained by _update_flow_trackers."""
        return self._buy_vol_sum, self._sell_vol_sum

    def _get_local_event_rate(self) -> float:
        """Event rate over last 10s relative to global average (~500-2000 events/s)."""
        n = len(self._event_ts_10s)
        # Global average based on typical ES RTH: ~1000 events/sec
        return n / (10.0 * 1000.0)  # normalized: 1.0 = average rate

    def _get_cancel_rate(self) -> float:
        """Fraction of recent events that are cancellations."""
        n_events = len(self._event_ts_10s)
        n_cancels = len(self._cancel_ts_10s)
        if n_events == 0:
            return 0.0
        return n_cancels / n_events

    def _get_realized_vol_60s(self) -> float:
        """
        Realized volatility over last 60s as std of price changes (in ticks).
        O(1) computation via running sum of squares.
        """
        n = len(self._vol_buf_60s)
        if n < 2:
            return 1.0
        # E[X^2] - E[X]^2 = var; we use E[X^2] only (MAR=0 downside vol)
        mean_sq = self._vol_sum_sq / n
        return math.sqrt(max(0.0, mean_sq))

    def _get_price_momentum(self) -> float:
        """Normalized price momentum over last 10 trades."""
        hist = list(self._price_history_10)
        if len(hist) < 2:
            return 0.0
        recent = np.array(hist[-min(10, len(hist)):], dtype=np.float32)
        if len(recent) < 2:
            return 0.0
        momentum = float(recent[-1] - recent[0])
        return np.clip(momentum / 2.0, -1.0, 1.0)  # normalize to [-1, 1]

    def _get_tod_seconds(self, ts_ns: int) -> float:
        """
        Extract seconds since midnight Eastern Time from nanosecond UTC timestamp.

        Uses month-based DST detection:
        - EDT (UTC-4): March through November (approximate)
        - EST (UTC-5): December through February
        """
        SECONDS_PER_DAY = 86400
        total_secs = ts_ns // 1_000_000_000
        # Determine month for DST detection (rough but accurate enough)
        # Month from epoch: days = total_secs / 86400; no need for full datetime
        # Use UTC datetime month as proxy (off by <1 day at most)
        utc_month = _utc_month_from_epoch_s(total_secs)
        et_offset_s = 4 * 3600 if 3 <= utc_month <= 11 else 5 * 3600
        tod = (total_secs - et_offset_s) % SECONDS_PER_DAY
        return float(tod)

    def _prune_trade_history_deque(self, ts: int) -> None:
        """Remove trade timestamps older than 60s."""
        cutoff = ts - 60_000_000_000
        while self._trade_ts_history and self._trade_ts_history[0] < cutoff:
            self._trade_ts_history.popleft()

    # ------------------------------------------------------------------
    # Utility / Navigation
    # ------------------------------------------------------------------

    def _find_rth_start(self) -> int:
        """
        Find the index of the first event during RTH (9:30 ET).
        Uses binary search for efficiency on large files.
        """
        n = len(self._timestamps)
        if n == 0:
            return 0

        # Check if data even has RTH events
        # RTH is 9:30-16:00 ET (34200-57600 seconds from midnight)
        # Binary search: find first index where tod >= RTH_START_SEC
        lo, hi = 0, n - 1
        result = n  # sentinel: not found

        while lo <= hi:
            mid = (lo + hi) // 2
            tod = self._get_tod_seconds(int(self._timestamps[mid]))
            if tod >= RTH_START_SEC:
                result = mid
                hi = mid - 1
            else:
                lo = mid + 1

        if result >= n:
            # No RTH start found — return 0 so episode runs with whatever data is there
            log.warning(f"No RTH start found in {self._episode_date}, using index 0")
            return 0

        return result

    def _is_after_rth(self, ts_ns: int) -> bool:
        """Return True if timestamp is after RTH end (4pm ET)."""
        tod = self._get_tod_seconds(ts_ns)
        return tod >= RTH_END_SEC

    def _build_pred_index(self) -> None:
        """Build date→file index for CNN-Mamba and PatchTST predictions at startup.

        This scans all fold prediction files ONCE and builds a lookup dict,
        so reset() doesn't need to re-scan 1GB+ of .npz files every time.

        Supports two prediction file formats:
        1. fold_XX_oot_predictions.npz (walk-forward format, with 'oot_files' key)
        2. {YYYYMMDD}_predictions.npz (bulk OOT format, with 'date' key)
        """
        for name, search_dir, index in [
            ("CNN-Mamba", self.pred_dir, self._pred_index),
            ("PatchTST", self.patchtst_pred_dir, self._patchtst_index),
        ]:
            if search_dir is None or not search_dir.exists():
                continue

            # Scan fold-based prediction files (walk-forward format)
            pred_files = sorted(search_dir.glob("fold_*_oot_predictions.npz"))
            for pred_file in pred_files:
                try:
                    d = np.load(pred_file, allow_pickle=True)
                    if "oot_files" in d:
                        for of in d["oot_files"]:
                            of_str = str(of)
                            # Extract 8-digit date from filename like "20260301_mbo_events.npz"
                            for part in of_str.replace("\\", "/").split("/"):
                                if part[:8].isdigit() and len(part) >= 8:
                                    date_key = part[:8]
                                    if date_key not in index:
                                        index[date_key] = pred_file
                                    break
                except Exception as e:
                    log.warning(f"Could not index {pred_file.name}: {e}")
                    continue

            # Scan per-date prediction files (bulk OOT format: {YYYYMMDD}_predictions.npz)
            bulk_files = sorted(search_dir.glob("*_predictions.npz"))
            bulk_count = 0
            for pred_file in bulk_files:
                if pred_file.name.startswith("fold_") or pred_file.name.startswith("concat_"):
                    continue  # Skip fold-based files already handled above
                # Extract date from filename like "20260306_predictions.npz"
                date_key = pred_file.stem.split("_")[0]
                if len(date_key) == 8 and date_key.isdigit():
                    if date_key not in index:
                        index[date_key] = pred_file
                        bulk_count += 1

            total_sources = len(pred_files) + bulk_count
            log.info(f"{name} pred index: {len(index)} dates from {total_sources} files ({len(pred_files)} fold + {bulk_count} bulk) in {search_dir}")

    def _find_pred_file(
        self,
        date_str: str,
        override: Optional[Path] = None,
    ) -> Optional[Path]:
        """Find a prediction file for the given episode date."""
        if override is not None:
            return override if override.exists() else None

        # Use pre-built index (fast O(1) lookup)
        return self._pred_index.get(date_str)

    def _find_pred_in_dir(
        self,
        date_str: str,
        search_dir: Path,
    ) -> Optional[Path]:
        """Find a prediction file containing the given date. Uses pre-built index."""
        if search_dir is None or not search_dir.exists():
            return None

        # Check which index to use
        if search_dir == self.patchtst_pred_dir or (
            self.patchtst_pred_dir and search_dir.resolve() == self.patchtst_pred_dir.resolve()
        ):
            result = self._patchtst_index.get(date_str)
        else:
            result = self._pred_index.get(date_str)

        if result is not None:
            return result

        # Fallback: scan files (slow, only if index missed it)
        for pred_file in sorted(search_dir.glob("fold_*_oot_predictions.npz")):
            try:
                d = np.load(pred_file, allow_pickle=True)
                if "oot_files" in d:
                    for of in d["oot_files"]:
                        if date_str in str(of):
                            return pred_file
            except Exception:
                continue

        return None

    def _maybe_fit_pca(self) -> None:
        """
        Fit a simple PCA (5 components) on the episode's embeddings.
        Used to compress embeddings to 5-dim for observation space.
        Uses truncated SVD (power iteration) for efficiency.
        """
        if self._embeddings is None or self._pca_components is not None:
            return
        if len(self._embeddings) < 10:
            return

        E = self._embeddings.shape[1]
        n_components = min(5, E)

        # Subsample if embeddings are large
        n_samples = min(5000, len(self._embeddings))
        idx = np.random.choice(len(self._embeddings), n_samples, replace=False)
        X = self._embeddings[idx].astype(np.float64)
        X -= X.mean(axis=0)

        # Power iteration SVD (5 steps)
        rng = np.random.default_rng(42)
        Q = rng.standard_normal((E, n_components))
        for _ in range(5):
            Q, _ = np.linalg.qr(X.T @ (X @ Q))
        B = X @ Q
        U, _, Vt = np.linalg.svd(B, full_matrices=False)
        self._pca_components = (Q @ Vt).T  # (n_components, E)

    def _get_info(self) -> Dict[str, Any]:
        """Return diagnostic info dict.

        HC #221/#230: Enriched with per-trade data so train_split_dqn.py can
        compute per-head rewards (ENTRY/CANCEL/EXIT) from the info dict rather
        than using the single scalar reward for all heads.
        """
        info = {
            "step_idx": self._step_idx,
            "pred_idx": self._pred_idx,
            "position": self._position,
            "pending_order": self._pending_order is not None,
            "episode_pnl_ticks": self._episode_pnl_ticks,
            "episode_pnl_usd": self._episode_pnl_ticks * TICK_VALUE,
            "episode_trades": self._episode_trades,
            "win_rate": self._sortino_tracker.win_rate(),
            "rolling_sortino": self._sortino_tracker.compute(),
            "consecutive_losses": self._consecutive_losses,
            "mfe_ticks": self._mfe_ticks,
            "mae_ticks": self._mae_ticks,
            "date": self._episode_date,
        }
        # HC #221: Attach last_trade when a trade just closed (set by _execute_fill)
        if hasattr(self, '_last_closed_trade') and self._last_closed_trade is not None:
            t = self._last_closed_trade
            info["last_trade"] = {
                "pnl_ticks": t.pnl_ticks,
                "mfe_ticks": t.mfe_ticks,
                "mae_ticks": t.mae_ticks,
                "hold_secs": t.hold_secs,
                "direction": t.direction,
                "exit_reason": t.exit_reason,
                "entry_signal_1s": getattr(self, '_last_entry_signal_1s', 0.0),
            }
            self._last_closed_trade = None  # consume it
        return info

    # ------------------------------------------------------------------
    # Metrics / Summary
    # ------------------------------------------------------------------

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Compute full episode performance metrics.
        Call after episode is done (done=True from step()).
        """
        if not self._trade_history:
            return {
                "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "profit_factor": 0.0, "avg_pnl_ticks": 0.0, "total_pnl_ticks": 0.0,
                "total_pnl_usd": 0.0, "avg_hold_secs": 0.0, "avg_mfe_ticks": 0.0,
                "avg_mae_ticks": 0.0, "fill_rate": 0.0,
            }

        pnls = np.array([t.pnl_ticks for t in self._trade_history], dtype=np.float64)
        holds = np.array([t.hold_secs for t in self._trade_history], dtype=np.float64)
        mfes  = np.array([t.mfe_ticks for t in self._trade_history], dtype=np.float64)
        maes  = np.array([t.mae_ticks for t in self._trade_history], dtype=np.float64)

        n = len(pnls)
        win_rate = float((pnls > 0).mean())
        mean_pnl = float(pnls.mean())
        std_pnl  = float(pnls.std()) if n > 1 else 1.0

        # Sharpe (annualized rough estimate, per-trade basis)
        sharpe = mean_pnl / (std_pnl + 1e-8) * math.sqrt(n)

        # Sortino
        downside = pnls[pnls < 0]
        if len(downside) == 0:
            sortino = mean_pnl * 10.0
        else:
            dd = float(np.sqrt(np.mean(downside ** 2)))
            sortino = mean_pnl / (dd + 1e-8)

        # Profit factor
        gross_wins = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
        gross_losses = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
        pf = gross_wins / gross_losses

        return {
            "n_trades": n,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "sortino": sortino,
            "profit_factor": pf,
            "avg_pnl_ticks": mean_pnl,
            "total_pnl_ticks": float(pnls.sum()),
            "total_pnl_usd": float(pnls.sum()) * TICK_VALUE,
            "avg_hold_secs": float(holds.mean()),
            "avg_mfe_ticks": float(mfes.mean()),
            "avg_mae_ticks": float(maes.mean()),
            "fill_rate": 1.0,  # all tracked trades are fills; unfilled = no entry
        }


# ─── Minimal Gym-compatible spaces ────────────────────────────────────────────

class _BoxSpace:
    """Minimal Box space spec (avoids gym dependency)."""
    def __init__(self, low, high, shape, dtype):
        self.low   = np.full(shape, low,  dtype=dtype)
        self.high  = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        # For unbounded spaces, sample from standard normal (more useful than uniform)
        return np.random.standard_normal(self.shape).astype(self.dtype)


class _DiscreteSpace:
    """Minimal Discrete space spec."""
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(0, self.n)


# ─── Fast batch stepping helper ───────────────────────────────────────────────

def collect_episode(
    env: FIFOExecutionEnv,
    policy_fn,
    episode_file: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> Dict:
    """
    Run a full episode with the given policy function.

    Parameters
    ----------
    env         : FIFOExecutionEnv instance
    policy_fn   : callable(obs: np.ndarray) -> int
    episode_file: override episode file
    max_steps   : stop early (None = run to completion)

    Returns
    -------
    dict with observations, actions, rewards, dones, infos, and metrics
    """
    obs = env.reset(episode_file=episode_file)
    observations = [obs]
    actions, rewards, dones, infos = [], [], [], []

    step = 0
    while True:
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
        step += 1

        if done:
            break
        if max_steps is not None and step >= max_steps:
            break

    metrics = env.get_episode_metrics()
    return {
        "observations": np.array(observations[:-1], dtype=np.float32),
        "actions": np.array(actions, dtype=np.int32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
        "total_reward": float(np.sum(rewards)),
        "steps": step,
        "metrics": metrics,
    }


# ─── Main: smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import random

    log.setLevel(logging.INFO)
    log.info("=== FIFO RL Environment Smoke Test ===")

    # Verify event directory
    if not EVENT_DIR.exists():
        log.error(f"Event directory not found: {EVENT_DIR}")
        sys.exit(1)

    event_files = sorted(EVENT_DIR.glob("20??????_mbo_events.npz"))
    log.info(f"Found {len(event_files)} episode files")

    # Use a recent file for testing
    test_file = event_files[-1]
    log.info(f"Testing with: {test_file.name}")

    # Create environment
    env = FIFOExecutionEnv(
        event_dir=EVENT_DIR,
        pred_dir=PRED_DIR,
        rth_only=True,
        verbose=True,
        seed=42,
    )

    # Test 1: basic reset
    log.info("\n--- Test 1: reset() ---")
    obs = env.reset(episode_file=test_file)
    assert obs.shape == (OBS_DIM,), f"Expected obs shape ({OBS_DIM},), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in initial observation!"
    log.info(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    log.info(f"  obs[:8]: {obs[:8]}")
    log.info(f"  Initial info: {env._get_info()}")

    # Test 2: random policy episode (limited steps)
    log.info("\n--- Test 2: random policy (10,000 steps) ---")
    obs = env.reset(episode_file=test_file)
    total_reward = 0.0
    n_fills = 0
    step_count = 0
    t0 = time.time()

    for step in range(10_000):
        action = random.randint(0, 6)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if reward != 0.0:
            n_fills += 1
        step_count += 1
        if done:
            break

    elapsed = time.time() - t0
    log.info(f"  Steps: {step_count}, done: {done}")
    log.info(f"  Total reward: {total_reward:.4f}")
    log.info(f"  Non-zero rewards: {n_fills}")
    log.info(f"  Episode trades: {info['episode_trades']}")
    log.info(f"  Episode PnL: {info['episode_pnl_ticks']:.2f} ticks "
             f"(${info['episode_pnl_usd']:.2f})")
    log.info(f"  Speed: {step_count/elapsed:.0f} steps/sec")
    assert not np.any(np.isnan(obs)), "NaN in observation after 10k steps!"

    # Test 3: specific action sequences
    log.info("\n--- Test 3: structured action test ---")
    obs = env.reset(episode_file=test_file)
    # Step forward 500 events without acting
    for _ in range(500):
        obs, _, done, info = env.step(0)  # do nothing
        if done:
            break
    log.info(f"  After 500 wait steps: position={info['position']}, "
             f"pending={info['pending_order']}")

    # Place a limit buy
    obs, reward, done, info = env.step(1)
    log.info(f"  After limit BUY: pending={info['pending_order']}, reward={reward:.4f}")

    # Step forward and check for fill
    filled = False
    for step in range(10_000):
        obs, reward, done, info = env.step(0)
        if info["position"] == 1:
            filled = True
            log.info(f"  Limit BUY filled after {step+1} events! "
                     f"PnL so far: {info['episode_pnl_ticks']:.4f}")
            break
        if done:
            break

    if not filled:
        log.info("  Limit BUY not filled in 10k events (normal for back-of-queue)")
        # Cancel and try market order
        obs, _, _, _ = env.step(5)  # cancel
        obs, _, _, info = env.step(3)  # market buy
        log.info(f"  After market BUY: position={info['position']}, "
                 f"pending={info['pending_order']}")
        # Market exit
        obs, reward, _, info = env.step(6)
        log.info(f"  After market EXIT: reward={reward:.4f}, "
                 f"trades={info['episode_trades']}")

    # Test 4: episode with random policy (capped at 100k steps for speed)
    log.info("\n--- Test 4: random policy episode (100,000 steps) ---")
    rng = np.random.default_rng(123)
    result = collect_episode(
        env,
        policy_fn=lambda obs: int(rng.integers(0, 7)),
        episode_file=test_file,
        max_steps=100_000,
    )
    m = result["metrics"]
    log.info(f"  Episode steps: {result['steps']}")
    log.info(f"  Total reward: {result['total_reward']:.4f}")
    log.info(f"  Trades: {m['n_trades']}")
    log.info(f"  Win rate: {m['win_rate']:.1%}")
    log.info(f"  Sortino: {m['sortino']:.3f}")
    log.info(f"  Total PnL: {m['total_pnl_ticks']:.2f} ticks (${m['total_pnl_usd']:.2f})")
    log.info(f"  Avg hold: {m['avg_hold_secs']:.1f}s")
    assert result["observations"].shape[1] == OBS_DIM
    assert not np.any(np.isnan(result["observations"])), "NaN in episode observations!"

    # Test 5: observation space / action space
    log.info("\n--- Test 5: space checks ---")
    log.info(f"  Obs space shape: {env.observation_space.shape}")
    log.info(f"  Action space n: {env.action_space.n}")
    sample_obs = env.observation_space.sample()
    sample_act = env.action_space.sample()
    log.info(f"  Sample obs (finite check): {np.all(np.isfinite(sample_obs))}")
    log.info(f"  Sample action: {sample_act}")

    log.info("\n=== All tests passed ===")
    log.info(f"\nEnvironment summary:")
    log.info(f"  OBS_DIM: {OBS_DIM}")
    log.info(f"  Actions: 0=wait, 1=lim_buy, 2=lim_sell, 3=mkt_buy, "
             f"4=mkt_sell, 5=cancel, 6=exit")
    log.info(f"  TICK_VALUE: ${TICK_VALUE}")
    log.info(f"  COMMISSION_COST: {COMMISSION_COST:.3f} ticks (HC #127: only cost, no spread)")
    log.info(f"  NOTE: Spread already captured in fill prices (bid/ask). No separate spread cost.")
