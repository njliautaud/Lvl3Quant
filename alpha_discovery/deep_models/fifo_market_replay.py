#!/usr/bin/env python3
"""
FIFO Market Replay Evaluator for Execution Research
=====================================================
Replays raw MBO data event-by-event, reconstructing a full FIFO limit-order
book, and evaluates CNN-Mamba v2 predictions with realistic fill simulation.

WHY: All prior execution evaluation used midpoint PnL or probabilistic fills.
This module uses the actual CME FIFO queue from raw Databento MBO data.

WHAT IT DOES:
  1. Loads CNN-Mamba v2 OOT predictions (fold_XX_oot_predictions.npz)
  2. Maps each prediction to its nanosecond timestamp via MBO event data
  3. Replays raw DBN MBO data to reconstruct the order book event-by-event
  4. When prediction exceeds gate threshold: places a limit order at best bid/ask
  5. Tracks FIFO queue position, fills only when volume trades through
  6. Exits via TP/SL/max-hold-time/signal-decay/EOD
  7. Reports: Sharpe, Sortino, Profit Factor, Win Rate, R:R, fill rate, queue wait

DATA INPUTS:
  - CNN-Mamba v2 predictions: output/cnn_mamba_v2_smart_v3_mar/fold_XX_oot_predictions.npz
    Keys: predictions (N,3), labels (N,3), oot_files, embeddings (N,96)
  - Processed MBO events: data/processed/mbo_events/YYYYMMDD_mbo_events.npz
    Keys: events (M,6), timestamps (M,), labels_1s/5s/10s (M,)
  - Raw MBO DBN: data/raw/mbo/glbx-mdp3-YYYYMMDD.mbo.dbn.zst (Databento format)
  - Optional Exec MLP: output/exec_mlp_v1_9day/fold_XX_oot_predictions.npz
    Keys: gate_predictions (N,), confidence_predictions (N,)

ES FUTURES CONSTANTS (HC #52):
  Tick = $12.50 (0.25 pts), RT Commission = $4.70 = 0.376 ticks

Usage:
    python fifo_market_replay.py                           # default config
    python fifo_market_replay.py --gate-threshold 0.20     # higher gate
    python fifo_market_replay.py --tp-ticks 4 --sl-ticks 2 # wider TP/SL
    python fifo_market_replay.py --order-type chase         # chase orders
    python fifo_market_replay.py --use-exec-mlp             # use Exec MLP gate
"""

import numpy as np
import os
import sys
import json
import logging
import argparse
import time as time_mod
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('fifo_market_replay')

# ── ES Constants (HC #52) ────────────────────────────────────────────────────
TICK_RAW         = 250_000_000       # 0.25 pts in Databento fixed-point (1e9/pt)
TICK_USD         = 12.50             # $/tick
COMMISSION_RT    = 4.70              # $ round-trip
COMMISSION_TICKS = COMMISSION_RT / TICK_USD  # 0.376 ticks

# ── Prediction windowing (must match training) ──────────────────────────────
WINDOW_SIZE = 1000
STRIDE      = 500

# ── DBN action/side bytes ────────────────────────────────────────────────────
A_ADD    = b'A'
A_CANCEL = b'C'
A_MODIFY = b'M'
A_TRADE  = b'T'
A_FILL   = b'F'
A_RESET  = b'R'
S_BID    = b'B'
S_ASK    = b'A'
S_NONE   = b'N'

# ── Data paths ───────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parents[2]

CNN_PRED_DIR  = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
MBO_EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events'
EXEC_MLP_DIR  = LVL3_ROOT / 'output' / 'exec_mlp_v1_9day'
RESULTS_DIR   = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_MBO_DIRS = [
    str(LVL3_ROOT / 'data' / 'raw_mbo'),
    str(LVL3_ROOT / 'data' / 'raw' / 'mbo'),
]


# =============================================================================
# FIFO Order Book (from mbo_replay_server.py, kept self-contained)
# =============================================================================

class PriceLevel:
    """FIFO queue at one price level."""
    __slots__ = ('price_raw', 'orders')

    def __init__(self, price_raw: int):
        self.price_raw = price_raw
        self.orders: OrderedDict = OrderedDict()

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
        qty = 0
        for o, q in self.orders.items():
            if o == oid:
                break
            qty += q
        return qty

    def consume(self, qty: int) -> list:
        """FIFO consume qty. Returns list of fully-consumed order_ids."""
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


class OrderBook:
    """Full price-level FIFO order book for one instrument."""

    def __init__(self):
        self.bids: Dict[int, PriceLevel] = {}
        self.asks: Dict[int, PriceLevel] = {}
        self._oid_side:  Dict[int, bytes] = {}
        self._oid_price: Dict[int, int]   = {}

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self._oid_side.clear()
        self._oid_price.clear()

    def _book(self, side):
        return self.bids if side == S_BID else self.asks

    def add(self, oid: int, side, price: int, qty: int):
        if side == S_NONE:
            return
        b = self._book(side)
        if price not in b:
            b[price] = PriceLevel(price)
        b[price].add(oid, qty)
        self._oid_side[oid]  = side
        self._oid_price[oid] = price

    def cancel(self, oid: int):
        side  = self._oid_side.pop(oid, None)
        price = self._oid_price.pop(oid, None)
        if side is not None and price is not None:
            b = self._book(side)
            if price in b:
                b[price].cancel(oid)
                if b[price].empty():
                    del b[price]

    def modify(self, oid: int, new_qty: int, new_price: int):
        side      = self._oid_side.get(oid)
        old_price = self._oid_price.get(oid)
        if side is None or old_price is None:
            return
        b = self._book(side)
        if new_price != old_price:
            self.cancel(oid)
            self.add(oid, side, new_price, new_qty)
        else:
            if old_price in b:
                b[old_price].modify(oid, new_qty)

    def trade(self, aggressor_side, price: int, qty: int) -> list:
        passive = S_BID if aggressor_side == S_ASK else S_ASK
        b = self._book(passive)
        if price not in b:
            return []
        consumed = b[price].consume(qty)
        if b[price].empty():
            del b[price]
        for oid in consumed:
            self._oid_side.pop(oid, None)
            self._oid_price.pop(oid, None)
        return consumed

    def best_bid(self) -> Optional[int]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[int]:
        return min(self.asks) if self.asks else None

    def mid_raw(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    def spread_ticks(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) / TICK_RAW if bb is not None and ba is not None else None

    def qty_at(self, side, price: int) -> int:
        b = self._book(side)
        return b[price].total_qty() if price in b else 0

    def microprice_raw(self) -> Optional[float]:
        """HC #491 R5 — size-weighted microprice in raw price units.
        microprice = (bb * ask_sz + ba * bid_sz) / (bid_sz + ask_sz).
        Returns None if top of book is missing or zero-size on both sides."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        bid_sz = self.qty_at(S_BID, bb)
        ask_sz = self.qty_at(S_ASK, ba)
        denom = bid_sz + ask_sz
        if denom <= 0:
            return (bb + ba) / 2.0
        return (bb * ask_sz + ba * bid_sz) / denom


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class SimOrder:
    """A simulated order placed into the book."""
    sim_oid:          int
    direction:        str       # 'long' or 'short'
    signal_ts_ns:     int
    order_type:       str       # 'limit', 'market', 'chase'
    entry_price:      int       # raw price
    tp_price:         int
    sl_price:         int
    tp_ticks:         float
    sl_ticks:         float
    queue_ahead:      int
    mid_at_signal:    float
    cancel_after_ns:  int
    max_hold_ns:      int       # max hold time after fill
    pred_strength:    float     # |prediction| at signal time
    # Chase tracking
    chase_reprices:   int = 0
    max_reprices:     int = 3
    reprice_after_ns: int = 1_000_000_000
    last_reprice_ts:  int = 0
    # Fill tracking
    filled:           bool = False
    fill_price:       int  = 0
    fill_ts_ns:       int  = 0
    # HC #437 Bug 2 — per-signal realized labels at horizon checkpoints
    # (signed log-returns in ticks, +ve = price went up). Populated only when
    # order_management='hc413_bracket' is in use; otherwise None.
    labels_by_h:      Optional[Dict[int, float]] = None
    # HC #491 R5 / HC #492 R3 — TOD + microprice diagnostics captured at the
    # moment the signal was emitted / order was submitted.
    entry_time_ns:        int = 0           # ts_recv (UTC ns) at submit time
    microprice_at_entry:  float = 0.0       # size-weighted microprice at submit
    microprice_dir:       int = 0           # sign(microprice - last_trade_price)
    # HC #494 R3 — trailing-stop tracking (post-fill, exit-axis mechanic).
    # max_favorable_raw tracks running max favorable price excursion in raw
    # price units. trail_anchored flips True once the trailing trigger is
    # crossed and sl_price has been re-anchored. Backward compatible: when
    # trail_trigger_ticks==0 in simulate(), these stay inert.
    max_favorable_raw:    int = 0
    trail_anchored:       bool = False


@dataclass
class TradeResult:
    """Result of a completed trade (entry fill + exit)."""
    signal_ts_ns:     int
    direction:        str
    order_type:       str
    entry_price_raw:  int
    entry_ts_ns:      int
    exit_price_raw:   int
    exit_ts_ns:       int
    exit_reason:      str       # 'tp', 'sl', 'max_hold', 'eod', 'timeout'
    queue_ahead:      int
    queue_wait_ns:    int
    pnl_ticks:        float
    pnl_ticks_net:    float     # after commission
    pnl_dollars:      float
    slippage_ticks:   float
    mid_at_signal:    float
    spread_at_signal: Optional[float]
    pred_strength:    float
    hold_time_ns:     int
    # HC #491 R5 / HC #492 R3 — TOD-of-day + microprice-agreement diagnostics.
    # Captured at order-submit time so downstream slice analysis (TOD buckets,
    # microprice-direction agreement) can run without re-grading. Optional with
    # safe defaults so legacy callers and older fills.parquet readers continue
    # to function unchanged.
    entry_time_ns:        int = 0
    microprice_at_entry:  float = 0.0
    microprice_dir:       int = 0


# =============================================================================
# HC #413 bracket-exit resolver (HC #437 Bug 2 fix)
# =============================================================================

def _resolve_hc413_bracket(
    direction: str,
    labels_by_h: Dict[int, float],
    tp1: float,
    tp2: float,
    sl: float,
    horizons_sec: Tuple[float, ...] = (1.0, 5.0, 10.0),
) -> Tuple[str, float, float]:
    """
    HC #413 / HC #417 horizon-checkpoint TP1/TP2/SL bracket resolver.

    Mirrors `scripts/hc413_scalping_backtester/tp_sl_rules.resolve_exits`
    one-row variant. Exit priority at EACH checkpoint, in order:
        1. SL  if inpos(h) <= -sl                  -> gross = -sl
        2. TP2 if inpos(h) >= tp2 and tp2 > 0     -> gross = +tp2
        3. TP1 if inpos(h) >= tp1 and tp1 > 0     -> gross = +tp1
    Time-stop if no horizon hits: gross = inpos at latest finite horizon.

    SL-first ordering is the CONSERVATIVE choice per HC #413 docstring
    (under-counts TP-then-SL round-trip; never falsely converts an
    end-of-horizon adverse move into a TP).

    Parameters
    ----------
    direction : 'long' or 'short'
    labels_by_h : {h_sec_int -> signed log-return in ticks (+ve = price up)}
    tp1, tp2, sl : magnitudes in ticks (positive)
    horizons_sec : tuple of checkpoints to evaluate, in seconds, in order

    Returns
    -------
    (exit_reason, gross_ticks, hold_sec) where exit_reason is one of
    'tp1', 'tp2', 'sl', 'time_stop', or 'no_label' if no finite label found.
    gross_ticks is the realized signed P&L in ticks BEFORE commission. The
    caller is responsible for adding commission via TradeResult.pnl_ticks_net.
    """
    sign = +1.0 if direction == 'long' else -1.0
    last_inpos = None
    last_h = None
    for h_sec in horizons_sec:
        h_key = int(h_sec)
        if h_key not in labels_by_h:
            continue
        raw = labels_by_h[h_key]
        if raw is None or not np.isfinite(raw):
            continue
        inpos = sign * float(raw)
        # SL first
        if inpos <= -sl:
            return ('sl', -sl, h_sec)
        if tp2 > 0 and inpos >= tp2:
            return ('tp2', +tp2, h_sec)
        if tp1 > 0 and inpos >= tp1:
            return ('tp1', +tp1, h_sec)
        last_inpos = inpos
        last_h = h_sec
    if last_inpos is None:
        return ('no_label', 0.0, 0.0)
    return ('time_stop', last_inpos, last_h)


# =============================================================================
# Prediction loader
# =============================================================================

def load_fold_predictions(fold_path: Path, mbo_event_dir: Path) -> Optional[dict]:
    """
    Load one fold's OOT predictions and map to nanosecond timestamps.

    Each prediction i corresponds to the LAST event in its window:
        event_idx = min(i * STRIDE + WINDOW_SIZE - 1, n_events - 1)
    """
    d = np.load(fold_path, allow_pickle=True)
    preds  = d['predictions']    # (N, 3) — horizons 1s, 5s, 10s
    labels = d['labels']         # (N, 3)
    embeds = d.get('embeddings', None)

    # Extract OOT date
    oot_files = d.get('oot_files', None)
    if oot_files is None:
        return None
    if hasattr(oot_files, '__len__') and len(oot_files) > 0:
        fname = str(oot_files[0]).split('/')[-1].split('\\')[-1]
    else:
        fname = str(oot_files).split('/')[-1].split('\\')[-1]
    date_str = fname[:8]

    # Load timestamps from processed MBO events
    mbo_path = mbo_event_dir / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        log.warning(f"No MBO event file for {date_str}: {mbo_path}")
        return None

    mbo = np.load(mbo_path, allow_pickle=True)
    timestamps = mbo['timestamps']
    n_events   = len(timestamps)

    # Map prediction index to timestamp
    n_preds = len(preds)
    pred_ts = np.zeros(n_preds, dtype=np.int64)
    for i in range(n_preds):
        event_idx = min(i * STRIDE + WINDOW_SIZE - 1, n_events - 1)
        pred_ts[i] = timestamps[event_idx]

    return {
        'date':          date_str,
        'predictions':   preds,
        'labels':        labels,
        'embeddings':    embeds,
        'timestamps_ns': pred_ts,
        'n_preds':       n_preds,
        'n_events':      n_events,
    }


def load_decay_predictions(decay_dir: Path, mbo_event_dir: Path) -> List[dict]:
    """
    Load per-date decay predictions (from decay_v4_comprehensive).
    Each date dir has predictions.npz with:
      preds (N,3), labels_1s/5s/10s (N,), valid_indices (N,)
    """
    results = []
    for date_dir in sorted(decay_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        pred_file = date_dir / 'predictions.npz'
        if not pred_file.exists():
            continue

        d = np.load(pred_file, allow_pickle=True)
        preds = d['preds']              # (N, 3)
        valid_idx = d['valid_indices']   # (N,) — indices into MBO events

        # Build labels array (N, 3) from separate arrays
        labels = np.stack([d['labels_1s'], d['labels_5s'], d['labels_10s']], axis=1)

        # Load timestamps from processed MBO events
        mbo_path = mbo_event_dir / f"{date_str}_mbo_events.npz"
        if not mbo_path.exists():
            log.warning(f"No MBO event file for {date_str}: {mbo_path}")
            continue

        mbo = np.load(mbo_path, allow_pickle=True)
        timestamps = mbo['timestamps']
        n_events = len(timestamps)

        # Map valid_indices to timestamps
        safe_idx = np.clip(valid_idx, 0, n_events - 1)
        pred_ts = timestamps[safe_idx]

        results.append({
            'date':          date_str,
            'predictions':   preds,
            'labels':        labels,
            'embeddings':    None,
            'timestamps_ns': pred_ts,
            'n_preds':       len(preds),
            'n_events':      n_events,
        })

    return results


def load_exec_mlp_for_date(date_str: str, exec_mlp_dir: Path) -> Optional[dict]:
    """Load Exec MLP predictions for a specific date (if available)."""
    for f in exec_mlp_dir.glob("fold_*_oot_predictions.npz"):
        d = np.load(f, allow_pickle=True)
        if 'date' in d and str(d['date']) == date_str:
            return {
                'gate':       d['gate_predictions'],        # (N,)
                'confidence': d['confidence_predictions'],  # (N,)
            }
    return None


# =============================================================================
# Signal generation
# =============================================================================

def generate_signals(
    data: dict,
    gate_threshold: float = 0.05,
    horizon: int = 0,
    min_spread_ticks: float = 0.0,
    min_interval_ns: int = 500_000_000,
    exec_mlp: Optional[dict] = None,
    exec_mlp_gate_threshold: float = 0.5,
) -> List[dict]:
    """
    Generate trade signals from CNN-Mamba v2 predictions.

    Args:
        data: dict from load_fold_predictions
        gate_threshold: min |prediction| to trigger signal
        horizon: which horizon to use (0=1s, 1=5s, 2=10s)
        min_spread_ticks: skip if spread > this (0=no filter)
        min_interval_ns: minimum time between signals (anti-churn)
        exec_mlp: optional Exec MLP predictions dict
        exec_mlp_gate_threshold: min gate probability to trade

    Returns:
        List of signal dicts with ts_ns, direction, strength
    """
    preds = data['predictions']
    ts    = data['timestamps_ns']

    raw_signal = preds[:, horizon]
    strength   = np.abs(raw_signal)

    # Gate: prediction strength threshold
    mask = strength >= gate_threshold

    # Optional: Exec MLP gate
    if exec_mlp is not None:
        gate_preds = exec_mlp['gate']
        # Exec MLP may have different length; align by truncating
        min_len = min(len(mask), len(gate_preds))
        mask = mask[:min_len]
        mask &= gate_preds[:min_len] >= exec_mlp_gate_threshold

    indices = np.where(mask)[0]

    # Anti-churn: enforce minimum interval between signals
    signals = []
    last_ts = -np.inf
    for idx in indices:
        t = int(ts[idx])
        if t - last_ts < min_interval_ns:
            continue
        direction = 'long' if raw_signal[idx] > 0 else 'short'
        signals.append({
            'ts_ns':     t,
            'direction': direction,
            'strength':  float(strength[idx]),
        })
        last_ts = t

    return signals


# =============================================================================
# Raw DBN file finder
# =============================================================================

def find_dbn_path(date: str) -> str:
    """Find the raw MBO DBN file for a given date."""
    fname = f'glbx-mdp3-{date}.mbo.dbn.zst'
    for d in RAW_MBO_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'DBN file for {date} not found. Searched: {RAW_MBO_DIRS}')


# =============================================================================
# Core FIFO Replay Engine
# =============================================================================

class FIFOReplayEngine:
    """
    Full MBO FIFO replay fill simulator for one trading day.

    Replays all raw Databento MBO events, maintains the order book,
    and simulates limit/market/chase order fills with proper FIFO queue
    position tracking.
    """

    # ES instrument IDs by contract
    ES_INSTRUMENT_IDS = {
        'ESH6': 42140878,  # Mar 2026
        'ESM6': 51643782,  # Jun 2026
    }
    # Auto-detect: pick instrument with most events (handles contract rolls)
    DEFAULT_INSTR_ID = None  # None = auto-detect

    def __init__(
        self,
        date: str,
        instrument_id: Optional[int] = None,
        cancel_after_ns: int   = 30_000_000_000,   # 30s cancel unfilled orders
        max_hold_ns:     int   = 60_000_000_000,    # 60s max hold after fill
        max_reprices:    int   = 3,
        reprice_after_ns: int  = 1_000_000_000,     # 1s between chase reprices
    ):
        self.date             = date
        self.instrument_id    = instrument_id  # None = auto-detect
        self.cancel_after_ns  = cancel_after_ns
        self.max_hold_ns      = max_hold_ns
        self.max_reprices     = max_reprices
        self.reprice_after_ns = reprice_after_ns

        self._load(date)

    def _load(self, date: str):
        """Load and filter raw MBO records for the instrument."""
        try:
            import databento as db
        except ImportError:
            raise ImportError(
                "databento package required. Install with: pip install databento")

        path  = find_dbn_path(date)
        store = db.DBNStore.from_file(path)
        recs  = store.to_ndarray()

        # Auto-detect instrument ID: pick the one with most events
        if self.instrument_id is None:
            ids, counts = np.unique(recs['instrument_id'], return_counts=True)
            self.instrument_id = int(ids[np.argmax(counts)])
            log.info(f"  Auto-detected instrument_id={self.instrument_id} "
                     f"({counts.max():,} events, {len(ids)} instruments in file)")

        mask  = recs['instrument_id'] == self.instrument_id
        self.records = recs[mask]
        self._ts     = self.records['ts_recv'].astype(np.int64)
        log.info(f"  Loaded {len(self.records):,} MBO events for {date} "
                 f"(instr={self.instrument_id})")

    @staticmethod
    def _bytes(v) -> bytes:
        if isinstance(v, bytes):
            return v
        if isinstance(v, np.bytes_):
            return bytes(v)
        return bytes([v])

    def simulate(
        self,
        signals: List[dict],
        tp_ticks: float,
        sl_ticks: float,
        order_type: str = 'limit',
        order_management: str = 'realtime_sl',
        bracket_thresholds: Optional[Dict[str, float]] = None,
        bracket_horizons_sec: Optional[Tuple[float, ...]] = None,
        trail_trigger_ticks: float = 0.0,
        trail_anchor_offset_ticks: float = 0.0,
    ) -> List[TradeResult]:
        """
        Run the full FIFO replay simulation.

        Args:
            signals: list of {ts_ns, direction, strength,
                              labels_by_h?: {h_sec_int: tick}}
                     labels_by_h is REQUIRED per-signal for
                     order_management='hc413_bracket'. Each entry is the
                     realized SIGNED log-return at horizon h_sec from the
                     signal timestamp, in TICKS (matches v2/v3.x NPZ
                     target_log_ret_{1s,5s,10s} convention — positive =
                     price went UP).
            tp_ticks: take-profit in ticks from entry (used only in
                      'realtime_sl' mode — ignored in 'hc413_bracket')
            sl_ticks: stop-loss in ticks from entry (same caveat)
            order_type: 'limit', 'market', or 'chase'  — controls entry-side
                      fill mechanism. Entry-side FIFO replay is UNCHANGED
                      across order_management modes.
            order_management: 'realtime_sl' (default — original
                      intra-event TP/SL price tracking) OR 'hc413_bracket'
                      (HC #413 / HC #417 horizon-checkpoint TP1/TP2/SL
                      bracket evaluated against the realized per-signal
                      labels_by_h). The two modes coexist for side-by-side
                      methodology comparison (HC #437 Bug 2).
            bracket_thresholds: dict with keys 'tp1', 'tp2', 'sl' in TICKS,
                      required when order_management='hc413_bracket' AND
                      per-signal MFE/MAE is not provided. Magnitudes only;
                      direction is handled internally per side.
            bracket_horizons_sec: tuple of horizon checkpoints in SECONDS,
                      default (1.0, 5.0, 10.0). Each must have a matching
                      integer key in signal['labels_by_h'] (e.g. 1, 5, 10).
        """
        # ── HC #437 Bug 2 — bracket-mode validation ─────────────────────
        if order_management not in ('realtime_sl', 'hc413_bracket'):
            raise ValueError(
                f"order_management must be 'realtime_sl' or 'hc413_bracket'; "
                f"got {order_management!r}")
        use_bracket = (order_management == 'hc413_bracket')
        if use_bracket:
            if bracket_thresholds is None:
                raise ValueError(
                    "order_management='hc413_bracket' requires "
                    "bracket_thresholds={'tp1': float, 'tp2': float, 'sl': float}")
            br_tp1 = float(bracket_thresholds['tp1'])
            br_tp2 = float(bracket_thresholds['tp2'])
            br_sl  = float(bracket_thresholds['sl'])
            br_horizons = bracket_horizons_sec or (1.0, 5.0, 10.0)
        else:
            br_tp1 = br_tp2 = br_sl = 0.0
            br_horizons = ()

        signals = sorted(signals, key=lambda s: s['ts_ns'])
        results:  List[TradeResult] = []
        pending:  List[SimOrder]    = []
        filled:   List[SimOrder]    = []      # filled but not yet exited
        sim_oid_ctr = 900_000_000
        sig_idx  = 0
        book     = OrderBook()
        # HC #491 R5 — track most recent trade price for microprice-direction
        # diagnostics. Updated on every A_TRADE / A_FILL event.
        last_trade_price: int = 0

        # ── Local helper: emit bracket-mode TradeResult after fill ──────
        def _emit_bracket_result(o: SimOrder, fill_ts_ns: int) -> None:
            """Resolve HC #413 bracket exit at horizon checkpoints and
            append a TradeResult. Bypasses the intra-event price-tracking
            exit loop used by realtime_sl mode."""
            lbh = o.labels_by_h or {}
            exit_reason, gross_ticks, hold_sec = _resolve_hc413_bracket(
                direction=o.direction,
                labels_by_h=lbh,
                tp1=br_tp1, tp2=br_tp2, sl=br_sl,
                horizons_sec=br_horizons,
            )
            if exit_reason == 'no_label':
                # No finite label at any horizon — drop the trade.
                return
            # Reconstruct an exit price from gross_ticks and side.
            if o.direction == 'long':
                exit_price = o.fill_price + int(round(gross_ticks * TICK_RAW))
            else:
                exit_price = o.fill_price - int(round(gross_ticks * TICK_RAW))
            slip = ((o.fill_price - o.mid_at_signal) / TICK_RAW
                    if o.direction == 'long'
                    else (o.mid_at_signal - o.fill_price) / TICK_RAW)
            hold_ns = int(hold_sec * 1e9)
            results.append(TradeResult(
                signal_ts_ns=o.signal_ts_ns,
                direction=o.direction,
                order_type=o.order_type,
                entry_price_raw=o.fill_price,
                entry_ts_ns=fill_ts_ns,
                exit_price_raw=exit_price,
                exit_ts_ns=fill_ts_ns + hold_ns,
                # HC #413 exit codes: 'tp1'/'tp2'/'sl'/'time_stop'
                exit_reason=exit_reason,
                queue_ahead=o.queue_ahead,
                queue_wait_ns=fill_ts_ns - o.signal_ts_ns,
                pnl_ticks=gross_ticks,
                pnl_ticks_net=gross_ticks - COMMISSION_TICKS,
                pnl_dollars=(gross_ticks - COMMISSION_TICKS) * TICK_USD,
                slippage_ticks=slip,
                mid_at_signal=o.mid_at_signal,
                spread_at_signal=None,
                pred_strength=o.pred_strength,
                hold_time_ns=hold_ns,
                entry_time_ns=o.entry_time_ns,
                microprice_at_entry=o.microprice_at_entry,
                microprice_dir=o.microprice_dir,
            ))

        n_recs = len(self.records)
        report_interval = max(1, n_recs // 10)

        for rec_idx, rec in enumerate(self.records):
            if rec_idx % report_interval == 0 and rec_idx > 0:
                pct = rec_idx * 100 // n_recs
                log.debug(f"    replay {pct}%: {len(results)} trades, "
                          f"{len(pending)} pending, {len(filled)} open")

            ts     = int(rec['ts_recv'])
            action = self._bytes(rec['action'])
            side   = self._bytes(rec['side'])
            price  = int(rec['price'])
            qty    = int(rec['size'])
            oid    = int(rec['order_id'])

            # ── 1. Update order book ─────────────────────────────────────
            consumed_oids = []
            if action == A_RESET:
                book.reset()
            elif action == A_ADD:
                book.add(oid, side, price, qty)
            elif action == A_CANCEL:
                book.cancel(oid)
            elif action == A_MODIFY:
                book.modify(oid, qty, price)
            elif action in (A_TRADE, A_FILL):
                consumed_oids = book.trade(side, price, qty)
                # HC #491 R5 — refresh last trade price for microprice_dir.
                last_trade_price = price

            # ── 2. Accept new signals ────────────────────────────────────
            while sig_idx < len(signals) and signals[sig_idx]['ts_ns'] <= ts:
                sig = signals[sig_idx]
                sig_idx += 1

                mid = book.mid_raw()
                bb  = book.best_bid()
                ba  = book.best_ask()
                if mid is None or bb is None or ba is None:
                    continue

                direction = sig['direction']
                spread    = book.spread_ticks()

                # HC #491 R5 / HC #492 R3 — snapshot microprice + last-trade
                # direction at signal/submit time for downstream slice analysis.
                _mp_raw = book.microprice_raw()
                if _mp_raw is None:
                    _mp_at_entry = float(mid)
                    _mp_dir = 0
                else:
                    _mp_at_entry = float(_mp_raw)
                    if last_trade_price <= 0:
                        _mp_dir = 0
                    elif _mp_raw > last_trade_price:
                        _mp_dir = 1
                    elif _mp_raw < last_trade_price:
                        _mp_dir = -1
                    else:
                        _mp_dir = 0
                _entry_time_ns = int(ts)

                if order_type == 'market':
                    # Immediate fill at best ask (long) or best bid (short)
                    fill_price = ba if direction == 'long' else bb
                    slip = ((fill_price - mid) / TICK_RAW if direction == 'long'
                            else (mid - fill_price) / TICK_RAW)

                    o = SimOrder(
                        sim_oid=sim_oid_ctr, direction=direction,
                        signal_ts_ns=sig['ts_ns'], order_type='market',
                        entry_price=fill_price,
                        tp_price=(fill_price + int(tp_ticks * TICK_RAW)
                                  if direction == 'long'
                                  else fill_price - int(tp_ticks * TICK_RAW)),
                        sl_price=(fill_price - int(sl_ticks * TICK_RAW)
                                  if direction == 'long'
                                  else fill_price + int(sl_ticks * TICK_RAW)),
                        tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                        queue_ahead=0, mid_at_signal=mid,
                        cancel_after_ns=self.cancel_after_ns,
                        max_hold_ns=self.max_hold_ns,
                        pred_strength=sig.get('strength', 0),
                        filled=True, fill_price=fill_price, fill_ts_ns=ts,
                        labels_by_h=sig.get('labels_by_h'),
                        entry_time_ns=_entry_time_ns,
                        microprice_at_entry=_mp_at_entry,
                        microprice_dir=_mp_dir,
                    )
                    sim_oid_ctr += 1
                    if use_bracket:
                        # HC #413 bracket — resolve at horizon checkpoints
                        # and skip the per-event TP/SL price-tracking loop.
                        _emit_bracket_result(o, fill_ts_ns=ts)
                    else:
                        filled.append(o)
                    continue

                # ── Limit / Chase order ──────────────────────────────────
                entry_price  = bb if direction == 'long' else ba
                passive_side = S_BID if direction == 'long' else S_ASK
                queue_ahead  = book.qty_at(passive_side, entry_price)

                sim_oid = sim_oid_ctr
                sim_oid_ctr += 1
                book.add(sim_oid, passive_side, entry_price, 1)

                tp_price = (entry_price + int(tp_ticks * TICK_RAW)
                            if direction == 'long'
                            else entry_price - int(tp_ticks * TICK_RAW))
                sl_price = (entry_price - int(sl_ticks * TICK_RAW)
                            if direction == 'long'
                            else entry_price + int(sl_ticks * TICK_RAW))

                pending.append(SimOrder(
                    sim_oid=sim_oid, direction=direction,
                    signal_ts_ns=sig['ts_ns'], order_type=order_type,
                    entry_price=entry_price, tp_price=tp_price, sl_price=sl_price,
                    tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                    queue_ahead=queue_ahead, mid_at_signal=mid,
                    cancel_after_ns=self.cancel_after_ns,
                    max_hold_ns=self.max_hold_ns,
                    pred_strength=sig.get('strength', 0),
                    max_reprices=self.max_reprices,
                    reprice_after_ns=self.reprice_after_ns,
                    last_reprice_ts=ts,
                    labels_by_h=sig.get('labels_by_h'),
                    entry_time_ns=_entry_time_ns,
                    microprice_at_entry=_mp_at_entry,
                    microprice_dir=_mp_dir,
                ))

            # ── 3. Check pending orders for FIFO fills ───────────────────
            still_pending = []
            for o in pending:
                if o.sim_oid in consumed_oids:
                    o.filled     = True
                    o.fill_price = o.entry_price
                    o.fill_ts_ns = ts
                    if use_bracket:
                        # HC #413 bracket — resolve exit at horizon
                        # checkpoints from per-signal realized labels and
                        # bypass the per-event TP/SL tracking loop.
                        _emit_bracket_result(o, fill_ts_ns=ts)
                    else:
                        filled.append(o)
                    continue

                elapsed = ts - o.signal_ts_ns
                if elapsed > o.cancel_after_ns:
                    book.cancel(o.sim_oid)
                    continue

                # Chase: reprice toward current best
                if o.order_type == 'chase' and o.chase_reprices < o.max_reprices:
                    if ts - o.last_reprice_ts > o.reprice_after_ns:
                        new_price = (book.best_bid() if o.direction == 'long'
                                     else book.best_ask())
                        if new_price and new_price != o.entry_price:
                            book.cancel(o.sim_oid)
                            ps = S_BID if o.direction == 'long' else S_ASK
                            book.add(o.sim_oid, ps, new_price, 1)
                            o.entry_price = new_price
                            o.tp_price = (new_price + int(o.tp_ticks * TICK_RAW)
                                          if o.direction == 'long'
                                          else new_price - int(o.tp_ticks * TICK_RAW))
                            o.sl_price = (new_price - int(o.sl_ticks * TICK_RAW)
                                          if o.direction == 'long'
                                          else new_price + int(o.sl_ticks * TICK_RAW))
                            o.queue_ahead = book.qty_at(ps, new_price)
                            o.chase_reprices += 1
                            o.last_reprice_ts = ts

                still_pending.append(o)
            pending = still_pending

            # ── 4. Check filled orders for exit conditions ───────────────
            if action in (A_TRADE, A_FILL):
                still_filled = []
                # HC #494 R3 — trailing-stop config in raw price units
                _trail_on = trail_trigger_ticks > 0.0
                _trail_trig_raw = int(trail_trigger_ticks * TICK_RAW) if _trail_on else 0
                _trail_anchor_raw = int(trail_anchor_offset_ticks * TICK_RAW) if _trail_on else 0
                for o in filled:
                    exit_reason = None
                    exit_price  = None

                    # HC #494 R3 — update running MFE and re-anchor SL if
                    # trailing trigger crossed. Idempotent once anchored.
                    if _trail_on and not o.trail_anchored:
                        if o.direction == 'long':
                            fav_raw = price - o.fill_price
                        else:
                            fav_raw = o.fill_price - price
                        if fav_raw > o.max_favorable_raw:
                            o.max_favorable_raw = fav_raw
                        if o.max_favorable_raw >= _trail_trig_raw:
                            # Re-anchor SL to (entry ± anchor_offset).
                            # offset > 0 → above entry for long (locks profit);
                            # offset == 0 → exact breakeven (commission only).
                            if o.direction == 'long':
                                new_sl = o.fill_price + _trail_anchor_raw
                            else:
                                new_sl = o.fill_price - _trail_anchor_raw
                            o.sl_price = new_sl
                            o.trail_anchored = True

                    # Check TP/SL against trade price
                    if o.direction == 'long':
                        if price >= o.tp_price:
                            exit_reason = 'tp'
                            exit_price  = o.tp_price
                        elif price <= o.sl_price:
                            exit_reason = 'sl' if not o.trail_anchored else 'trail_sl'
                            exit_price  = o.sl_price
                    else:
                        if price <= o.tp_price:
                            exit_reason = 'tp'
                            exit_price  = o.tp_price
                        elif price >= o.sl_price:
                            exit_reason = 'sl' if not o.trail_anchored else 'trail_sl'
                            exit_price  = o.sl_price

                    # Check max hold time
                    if exit_reason is None and (ts - o.fill_ts_ns) > o.max_hold_ns:
                        exit_reason = 'max_hold'
                        mid = book.mid_raw()
                        exit_price = int(mid) if mid else o.entry_price

                    if exit_reason:
                        pnl_raw = (exit_price - o.fill_price
                                   if o.direction == 'long'
                                   else o.fill_price - exit_price)
                        pnl_ticks = pnl_raw / TICK_RAW
                        slip = ((o.fill_price - o.mid_at_signal) / TICK_RAW
                                if o.direction == 'long'
                                else (o.mid_at_signal - o.fill_price) / TICK_RAW)

                        results.append(TradeResult(
                            signal_ts_ns=o.signal_ts_ns,
                            direction=o.direction,
                            order_type=o.order_type,
                            entry_price_raw=o.fill_price,
                            entry_ts_ns=o.fill_ts_ns,
                            exit_price_raw=exit_price,
                            exit_ts_ns=ts,
                            exit_reason=exit_reason,
                            queue_ahead=o.queue_ahead,
                            queue_wait_ns=o.fill_ts_ns - o.signal_ts_ns,
                            pnl_ticks=pnl_ticks,
                            pnl_ticks_net=pnl_ticks - COMMISSION_TICKS,
                            pnl_dollars=(pnl_ticks - COMMISSION_TICKS) * TICK_USD,
                            slippage_ticks=slip,
                            mid_at_signal=o.mid_at_signal,
                            spread_at_signal=None,
                            pred_strength=o.pred_strength,
                            hold_time_ns=ts - o.fill_ts_ns,
                            entry_time_ns=o.entry_time_ns,
                            microprice_at_entry=o.microprice_at_entry,
                            microprice_dir=o.microprice_dir,
                        ))
                    else:
                        still_filled.append(o)
                filled = still_filled

        # ── EOD: close all remaining positions at mid ────────────────────
        last_ts = int(self._ts[-1]) if len(self._ts) else 0
        mid = book.mid_raw()

        for o in filled:
            exit_price = int(mid) if mid else o.fill_price
            pnl_raw = (exit_price - o.fill_price if o.direction == 'long'
                       else o.fill_price - exit_price)
            pnl_ticks = pnl_raw / TICK_RAW
            slip = ((o.fill_price - o.mid_at_signal) / TICK_RAW
                    if o.direction == 'long'
                    else (o.mid_at_signal - o.fill_price) / TICK_RAW)

            results.append(TradeResult(
                signal_ts_ns=o.signal_ts_ns,
                direction=o.direction,
                order_type=o.order_type,
                entry_price_raw=o.fill_price,
                entry_ts_ns=o.fill_ts_ns,
                exit_price_raw=exit_price,
                exit_ts_ns=last_ts,
                exit_reason='eod',
                queue_ahead=o.queue_ahead,
                queue_wait_ns=o.fill_ts_ns - o.signal_ts_ns,
                pnl_ticks=pnl_ticks,
                pnl_ticks_net=pnl_ticks - COMMISSION_TICKS,
                pnl_dollars=(pnl_ticks - COMMISSION_TICKS) * TICK_USD,
                slippage_ticks=slip,
                mid_at_signal=o.mid_at_signal,
                spread_at_signal=None,
                pred_strength=o.pred_strength,
                hold_time_ns=last_ts - o.fill_ts_ns,
                entry_time_ns=o.entry_time_ns,
                microprice_at_entry=o.microprice_at_entry,
                microprice_dir=o.microprice_dir,
            ))

        # Cancel remaining unfilled
        for o in pending:
            book.cancel(o.sim_oid)

        return results


# =============================================================================
# Metrics computation
# =============================================================================

def compute_metrics(
    results: List[TradeResult],
    n_signals: int,
) -> dict:
    """
    Compute comprehensive execution metrics from trade results.

    Returns dict with: Sharpe, Sortino, Profit Factor, Win Rate, avg R:R,
    fill rate, avg queue wait, avg hold time, exit breakdown.
    """
    if not results:
        return {
            'n_trades': 0, 'n_signals': n_signals, 'fill_rate': 0.0,
            'total_pnl_ticks': 0.0, 'total_pnl_dollars': 0.0,
            'sharpe': 0.0, 'sortino': 0.0, 'profit_factor': 0.0,
            'win_rate': 0.0, 'avg_rr': 0.0,
            'avg_queue_wait_ms': 0.0, 'avg_hold_time_ms': 0.0,
            'avg_queue_ahead': 0.0, 'avg_slippage_ticks': 0.0,
            'tp_rate': 0.0, 'sl_rate': 0.0, 'max_hold_rate': 0.0, 'eod_rate': 0.0,
            'avg_pred_strength': 0.0,
        }

    pnl = np.array([r.pnl_ticks_net for r in results])
    n   = len(pnl)

    # Win/loss breakdown
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    n_wins   = len(wins)
    n_losses = len(losses)

    # Sharpe (annualized from per-trade)
    mean_pnl = float(pnl.mean())
    std_pnl  = float(pnl.std()) if n > 1 else 1e-9
    sharpe   = mean_pnl / std_pnl if std_pnl > 1e-9 else 0.0

    # Sortino (downside deviation)
    neg = pnl[pnl < 0]
    downside_std = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-9
    sortino = mean_pnl / downside_std if downside_std > 1e-9 else 0.0

    # Profit Factor
    gross_profit = float(wins.sum()) if n_wins > 0 else 0.0
    gross_loss   = float(np.abs(losses.sum())) if n_losses > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    # Average R:R
    avg_win  = float(wins.mean())  if n_wins > 0  else 0.0
    avg_loss = float(np.abs(losses.mean())) if n_losses > 0 else 1e-9
    avg_rr   = avg_win / avg_loss

    # Exit breakdown
    exits = [r.exit_reason for r in results]
    tp_rate       = exits.count('tp') / n
    sl_rate       = exits.count('sl') / n
    max_hold_rate = exits.count('max_hold') / n
    eod_rate      = exits.count('eod') / n

    # Queue and timing
    queue_waits = np.array([r.queue_wait_ns for r in results])
    hold_times  = np.array([r.hold_time_ns for r in results])
    queue_ahead = np.array([r.queue_ahead for r in results])
    slippage    = np.array([r.slippage_ticks for r in results])
    strengths   = np.array([r.pred_strength for r in results])

    return {
        'n_trades':           n,
        'n_signals':          n_signals,
        'fill_rate':          n / max(1, n_signals),
        'total_pnl_ticks':    float(pnl.sum()),
        'total_pnl_dollars':  float(pnl.sum() * TICK_USD),
        'mean_pnl_ticks':     mean_pnl,
        'sharpe':             round(sharpe, 4),
        'sortino':            round(sortino, 4),
        'profit_factor':      round(profit_factor, 4),
        'win_rate':           round(n_wins / n, 4),
        'avg_rr':             round(avg_rr, 4),
        'avg_win_ticks':      round(avg_win, 3),
        'avg_loss_ticks':     round(avg_loss, 3),
        'n_wins':             n_wins,
        'n_losses':           n_losses,
        'avg_queue_wait_ms':  round(float(queue_waits.mean()) / 1e6, 1),
        'avg_hold_time_ms':   round(float(hold_times.mean()) / 1e6, 1),
        'avg_queue_ahead':    round(float(queue_ahead.mean()), 1),
        'avg_slippage_ticks': round(float(slippage.mean()), 3),
        'avg_pred_strength':  round(float(strengths.mean()), 4),
        'tp_rate':            round(tp_rate, 4),
        'sl_rate':            round(sl_rate, 4),
        'max_hold_rate':      round(max_hold_rate, 4),
        'eod_rate':           round(eod_rate, 4),
        'commission_ticks':   COMMISSION_TICKS,
    }


def compute_daily_metrics(daily_pnls: List[float]) -> dict:
    """Compute portfolio-level metrics from daily PnL array."""
    if not daily_pnls:
        return {}
    arr  = np.array(daily_pnls)
    n    = len(arr)
    mean = float(arr.mean())
    neg  = arr[arr < 0]

    std     = float(arr.std()) if n > 1 else 1e-9
    dstd    = float(np.sqrt(np.mean(neg**2))) if len(neg) > 0 else 1e-9
    sharpe  = mean / std if std > 1e-9 else 0.0
    sortino = mean / dstd if dstd > 1e-9 else 0.0

    # Monte Carlo bootstrap for confidence
    n_mc = 5000
    mc_sortinos = []
    for _ in range(n_mc):
        samp = np.random.choice(arr, size=n, replace=True)
        ng   = samp[samp < 0]
        ds   = float(np.sqrt(np.mean(ng**2))) if len(ng) > 0 else 1e-9
        mc_sortinos.append(float(samp.mean()) / ds)
    mc = np.array(mc_sortinos)

    return {
        'n_days':           n,
        'total_pnl':        round(float(arr.sum()), 2),
        'mean_daily_pnl':   round(mean, 2),
        'daily_sharpe':     round(sharpe, 4),
        'daily_sortino':    round(sortino, 4),
        'max_drawdown':     round(float((np.maximum.accumulate(np.cumsum(arr)) - np.cumsum(arr)).max()), 2),
        'win_days':         int((arr > 0).sum()),
        'loss_days':        int((arr < 0).sum()),
        'best_day':         round(float(arr.max()), 2),
        'worst_day':        round(float(arr.min()), 2),
        'mc_p5_sortino':    round(float(np.percentile(mc, 5)), 4),
        'mc_median_sortino': round(float(np.median(mc)), 4),
        'mc_p95_sortino':   round(float(np.percentile(mc, 95)), 4),
        'mc_prob_profit':   round(float((mc > 0).mean()), 4),
    }


# =============================================================================
# Main evaluation pipeline
# =============================================================================

def run_evaluation(args) -> dict:
    """
    Run the full FIFO market replay evaluation.

    Loads predictions, generates signals, replays MBO data,
    and computes comprehensive metrics.
    """
    log.info("=" * 80)
    log.info("FIFO MARKET REPLAY EVALUATOR")
    log.info("=" * 80)
    log.info(f"ES: tick=${TICK_USD}, commission=${COMMISSION_RT} RT "
             f"({COMMISSION_TICKS:.3f} ticks)")
    log.info(f"Config: gate={args.gate_threshold}, TP={args.tp_ticks}t, "
             f"SL={args.sl_ticks}t, order={args.order_type}, "
             f"max_hold={args.max_hold_ms}ms, cancel={args.cancel_ms}ms")
    log.info(f"Pred dir: {args.pred_dir}")
    log.info("")

    pred_dir     = Path(args.pred_dir)
    mbo_dir      = Path(args.mbo_event_dir)
    exec_mlp_dir = Path(args.exec_mlp_dir)

    # ── Load predictions ─────────────────────────────────────────────────
    # Try fold-based predictions first, then per-date decay predictions
    fold_files = sorted(pred_dir.glob("fold_*_oot_predictions.npz"))
    all_data = []

    if fold_files:
        log.info("Loading fold-based predictions...")
        for f in fold_files:
            data = load_fold_predictions(f, mbo_dir)
            if data:
                all_data.append(data)
                log.info(f"  {data['date']}: {data['n_preds']:,} predictions, "
                         f"{data['n_events']:,} events")
    else:
        # Try per-date decay format (each subdir = date with predictions.npz)
        date_dirs = [d for d in sorted(pred_dir.iterdir()) if d.is_dir() and (d / 'predictions.npz').exists()]
        if date_dirs:
            log.info("Loading per-date decay predictions...")
            all_data = load_decay_predictions(pred_dir, mbo_dir)
            for data in all_data:
                log.info(f"  {data['date']}: {data['n_preds']:,} predictions, "
                         f"{data['n_events']:,} events")
        else:
            log.error(f"No prediction files in {pred_dir}")
            return {}

    if not all_data:
        log.error("No valid data loaded")
        return {}

    log.info(f"\nLoaded {len(all_data)} days with predictions")

    # ── Evaluate each day ────────────────────────────────────────────────
    all_results    = []
    daily_pnls     = []
    daily_summaries = []
    total_signals  = 0
    total_trades   = 0

    for data in all_data:
        date = data['date']

        # Load optional Exec MLP
        exec_mlp = None
        if args.use_exec_mlp:
            exec_mlp = load_exec_mlp_for_date(date, exec_mlp_dir)
            if exec_mlp:
                log.info(f"  {date}: Exec MLP loaded ({len(exec_mlp['gate'])} predictions)")

        # Generate signals
        signals = generate_signals(
            data,
            gate_threshold=args.gate_threshold,
            horizon=args.horizon,
            min_interval_ns=int(args.min_interval_ms * 1e6),
            exec_mlp=exec_mlp,
            exec_mlp_gate_threshold=args.exec_mlp_gate,
        )
        total_signals += len(signals)

        if not signals:
            log.info(f"  {date}: 0 signals (gate={args.gate_threshold})")
            daily_pnls.append(0.0)
            continue

        # Run FIFO replay
        try:
            engine = FIFOReplayEngine(
                date=date,
                instrument_id=args.instrument_id,  # None = auto-detect
                cancel_after_ns=int(args.cancel_ms * 1e6),
                max_hold_ns=int(args.max_hold_ms * 1e6),
                max_reprices=args.max_reprices,
                reprice_after_ns=int(args.reprice_ms * 1e6),
            )
        except FileNotFoundError as e:
            log.warning(f"  {date}: {e}")
            continue

        trades = engine.simulate(
            signals,
            tp_ticks=args.tp_ticks,
            sl_ticks=args.sl_ticks,
            order_type=args.order_type,
        )
        all_results.extend(trades)
        total_trades += len(trades)

        # Daily metrics
        day_metrics = compute_metrics(trades, len(signals))
        day_pnl     = day_metrics['total_pnl_dollars']
        daily_pnls.append(day_pnl)

        log.info(
            f"  {date}: {len(signals)} signals -> {len(trades)} trades "
            f"({day_metrics['fill_rate']:.0%} fill) | "
            f"WR={day_metrics['win_rate']:.0%} | "
            f"PnL=${day_pnl:+.0f} ({day_metrics['total_pnl_ticks']:+.1f}t) | "
            f"TP={day_metrics['tp_rate']:.0%} SL={day_metrics['sl_rate']:.0%} "
            f"MH={day_metrics['max_hold_rate']:.0%} | "
            f"Q={day_metrics['avg_queue_ahead']:.0f} "
            f"wait={day_metrics['avg_queue_wait_ms']:.0f}ms"
        )

        daily_summaries.append({
            'date': date,
            'n_signals': len(signals),
            **day_metrics,
        })

    # ── Aggregate metrics ────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 80)

    overall = compute_metrics(all_results, total_signals)
    portfolio = compute_daily_metrics(daily_pnls)

    log.info(f"Signals:        {total_signals}")
    log.info(f"Trades:         {total_trades} ({overall.get('fill_rate', 0):.1%} fill rate)")
    if total_trades == 0:
        log.info("No trades executed — all signals missed fills or no matching DBN data.")
        log.info("Check that pred dates overlap with raw DBN data dates.")
    else:
        log.info(f"Total PnL:      ${overall['total_pnl_dollars']:+,.0f} "
                 f"({overall['total_pnl_ticks']:+.1f} ticks)")
        log.info(f"Win Rate:       {overall['win_rate']:.1%}")
        log.info(f"Avg R:R:        {overall.get('avg_rr', 0):.2f} "
                 f"(win={overall.get('avg_win_ticks', 0):.1f}t, loss={overall.get('avg_loss_ticks', 0):.1f}t)")
        log.info(f"Profit Factor:  {overall.get('profit_factor', 0):.2f}")
        log.info(f"Sharpe:         {overall.get('sharpe', 0):.3f}")
        log.info(f"Sortino:        {overall.get('sortino', 0):.3f}")
        log.info(f"Avg Queue:      {overall.get('avg_queue_ahead', 0):.1f} ahead, "
                 f"{overall.get('avg_queue_wait_ms', 0):.0f}ms wait")
        log.info(f"Avg Hold:       {overall.get('avg_hold_time_ms', 0):.0f}ms")
        log.info(f"Slippage:       {overall.get('avg_slippage_ticks', 0):.3f} ticks")

    if portfolio:
        log.info(f"\nPortfolio (daily):")
        log.info(f"  Days:         {portfolio['n_days']} "
                 f"(W={portfolio['win_days']}/L={portfolio['loss_days']})")
        log.info(f"  Daily Sortino: {portfolio['daily_sortino']:.3f}")
        log.info(f"  MC p5 Sortino: {portfolio['mc_p5_sortino']:.3f}")
        log.info(f"  Max Drawdown:  ${portfolio['max_drawdown']:,.0f}")
        log.info(f"  P(profit):     {portfolio['mc_prob_profit']:.1%}")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        'config': {
            'gate_threshold':   args.gate_threshold,
            'tp_ticks':         args.tp_ticks,
            'sl_ticks':         args.sl_ticks,
            'order_type':       args.order_type,
            'horizon':          args.horizon,
            'max_hold_ms':      args.max_hold_ms,
            'cancel_ms':        args.cancel_ms,
            'use_exec_mlp':     args.use_exec_mlp,
            'min_interval_ms':  args.min_interval_ms,
            'commission_ticks': COMMISSION_TICKS,
        },
        'overall':    overall,
        'portfolio':  portfolio,
        'daily':      daily_summaries,
    }

    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'fifo_replay_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_path}")

    return output


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='FIFO Market Replay Evaluator for CNN-Mamba v2')

    # Paths
    p.add_argument('--pred-dir', default=str(CNN_PRED_DIR),
                   help='CNN-Mamba v2 prediction directory')
    p.add_argument('--mbo-event-dir', default=str(MBO_EVENT_DIR),
                   help='Processed MBO event directory (for timestamps)')
    p.add_argument('--exec-mlp-dir', default=str(EXEC_MLP_DIR),
                   help='Exec MLP prediction directory')

    # Signal generation
    p.add_argument('--gate-threshold', type=float, default=0.10,
                   help='Min |prediction| to trigger signal (default: 0.10)')
    p.add_argument('--horizon', type=int, default=0,
                   help='Prediction horizon: 0=1s, 1=5s, 2=10s (default: 0)')
    p.add_argument('--min-interval-ms', type=float, default=500,
                   help='Min ms between signals (anti-churn, default: 500)')
    p.add_argument('--use-exec-mlp', action='store_true',
                   help='Use Exec MLP gate predictions')
    p.add_argument('--exec-mlp-gate', type=float, default=0.5,
                   help='Exec MLP gate threshold (default: 0.5)')

    # Order execution
    p.add_argument('--order-type', choices=['limit', 'market', 'chase'],
                   default='limit', help='Order type (default: limit)')
    p.add_argument('--tp-ticks', type=float, default=4.0,
                   help='Take profit in ticks (default: 4.0)')
    p.add_argument('--sl-ticks', type=float, default=2.0,
                   help='Stop loss in ticks (default: 2.0)')
    p.add_argument('--max-hold-ms', type=float, default=60000,
                   help='Max hold time in ms (default: 60000)')
    p.add_argument('--cancel-ms', type=float, default=30000,
                   help='Cancel unfilled orders after ms (default: 30000)')
    p.add_argument('--max-reprices', type=int, default=3,
                   help='Max chase reprices (default: 3)')
    p.add_argument('--reprice-ms', type=float, default=1000,
                   help='Chase reprice interval ms (default: 1000)')

    # Instrument
    p.add_argument('--instrument-id', type=int, default=None,
                   help='ES instrument ID (default: auto-detect)')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args)
