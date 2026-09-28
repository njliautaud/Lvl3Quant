#!/usr/bin/env python3
"""
Variable-Horizon FIFO Market Replay Grader (HC #497 R6)
========================================================

Grades the v3.5 multi-head CNN-Mamba model under per-trade, model-predicted
horizons. Per HC #497 R6, exits are NOT fixed TP/SL/hold — they are:

  head-A : pressure-direction logit (sign flip == regime change == exit)
  head-B : pressure-persistence regression in SECONDS (timeout countdown)
  head-C : cumulative-K-tick first-passage classifier (target hit -> exit)
  head-D : regime classifier (trend / MR / noise) — used for stratified Sharpe

ENTRY rule (HC #497 R6):
  On each event tick where |head-A| is in the top-`entry_pct` of the day
  (default top-10%), submit a PASSIVE LIMIT at same-side best
  (bid for long, ask for short).

POSITION-MANAGEMENT exits (first fire wins):
  1. head-A flips sign vs the entry-side  -> exit_reason='head-A flip'
  2. head-B persistence countdown expires -> exit_reason='head-B timeout'
  3. head-C cumulative-K-tick first-passage target hit (K configurable,
     default K=2 ticks)                   -> exit_reason='head-C target'
  Exit fill: passive limit at exit-side best (HC #74 FIFO-replay rule).

CANCEL-WINDOW for unfilled entry limit (HC #428 R2 — MFE-within-horizon):
  cancel_window_ns <= head-B predicted persistence at entry time.
  If unfilled by then -> cancel, free queue position, exit_reason='cancel_unfilled'.

QUEUE MECHANICS:
  Reuses the OrderBook / PriceLevel FIFO primitives from
  alpha_discovery.deep_models.fifo_market_replay (HC #493 canonical).
  Imported, NOT reinvented.

COSTS (CLAUDE.md cost section, ES futures AMP/Rithmic):
  ES_RT_COMMISSION_TICKS = 0.376   (canonical, $4.70 / $12.50)

OUTPUT PER OOT DAY:
  Trade list rows: entry_ts, exit_ts, side, entry_px, exit_px, ticks_net,
                   exit_reason in {head-A flip, head-B timeout, head-C target,
                                   cancel_unfilled}
  Day summary: n_trades, mean ticks_net, median ticks_net, WR, PF, day-Sharpe.
  Per-regime stratified Sharpe (head-D: trend/MR/noise) per HC #428 R1.

CROSS-FOLD AGGREGATION (HC #494 R1 viability bar):
  PASS requires ALL:
    (a) >=30 of OOT days FIFO-net-positive
    (b) >=5 trades/day mean
    (c) Sharpe >= 1.0
    (d) |Sharpe_green - Sharpe_red| / max <= 0.50   (HC #428 R1 regime-skew)

Usage:
  python3 scripts/variable_horizon_fifo_grader.py \\
      --pred-npz /path/to/fold_XX_v35_oot_predictions.npz \\
      --mbo-npz  /path/to/YYYYMMDD_v35.npz \\
      --output-dir output/var_horizon_fifo/

Per HC #420 this is the user's own quant codebase — authorized scaffold work.
Scaffold only. Real-data launches are gated on separate user authorization.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Canonical FIFO primitives — HC #493: reuse, do not reinvent.
# ──────────────────────────────────────────────────────────────────────────────
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
if str(LVL3_ROOT) not in sys.path:
    sys.path.insert(0, str(LVL3_ROOT))

# Soft-import: the test runs without databento installed, and the test injects
# its own minimal stand-ins via the public API. Real-data runs require the
# canonical queue primitives.
try:
    from alpha_discovery.deep_models.fifo_market_replay import (  # type: ignore
        OrderBook, PriceLevel, S_BID, S_ASK, TICK_RAW,
    )
    _HAVE_CANONICAL_BOOK = True
except Exception:  # pragma: no cover — exercised only when canonical unavailable
    _HAVE_CANONICAL_BOOK = False
    TICK_RAW = 250_000_000  # 0.25 pts in Databento fixed-point
    S_BID = b'B'
    S_ASK = b'A'

    # Minimal local fallback mirroring canonical interface — used by tests
    # and by environments where databento is not installed. Field-for-field
    # compatible with the canonical PriceLevel / OrderBook public methods
    # used in this module.
    from collections import OrderedDict

    class PriceLevel:  # type: ignore[no-redef]
        __slots__ = ('price_raw', 'orders')

        def __init__(self, price_raw: int):
            self.price_raw = price_raw
            self.orders: "OrderedDict[int, int]" = OrderedDict()

        def add(self, oid: int, qty: int):
            self.orders[oid] = qty

        def cancel(self, oid: int):
            self.orders.pop(oid, None)

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

    class OrderBook:  # type: ignore[no-redef]
        def __init__(self):
            self.bids: Dict[int, PriceLevel] = {}
            self.asks: Dict[int, PriceLevel] = {}
            self._oid_side: Dict[int, bytes] = {}
            self._oid_price: Dict[int, int] = {}

        def add(self, oid: int, side, price: int, qty: int):
            b = self.bids if side == S_BID else self.asks
            if price not in b:
                b[price] = PriceLevel(price)
            b[price].add(oid, qty)
            self._oid_side[oid] = side
            self._oid_price[oid] = price

        def cancel(self, oid: int):
            side = self._oid_side.pop(oid, None)
            price = self._oid_price.pop(oid, None)
            if side is not None and price is not None:
                b = self.bids if side == S_BID else self.asks
                if price in b:
                    b[price].cancel(oid)
                    if b[price].empty():
                        del b[price]

        def trade(self, aggressor_side, price: int, qty: int) -> list:
            passive = S_BID if aggressor_side == S_ASK else S_ASK
            b = self.bids if passive == S_BID else self.asks
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

        def qty_at(self, side, price: int) -> int:
            b = self.bids if side == S_BID else self.asks
            return b[price].total_qty() if price in b else 0


# ──────────────────────────────────────────────────────────────────────────────
# Constants (CLAUDE.md cost section)
# ──────────────────────────────────────────────────────────────────────────────
ES_RT_COMMISSION_TICKS = 0.376        # $4.70 RT / $12.50 per tick
ANNUAL_SQRT_SECONDS    = float(np.sqrt(252 * 6.5 * 3600))
REGIME_LABELS          = ('trend', 'mr', 'noise')  # head-D class order
SIM_OID_COUNTER_START  = 10_000_000_000_000

log = logging.getLogger('var_horizon_fifo_grader')


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class VHConfig:
    """Variable-horizon grader configuration."""
    entry_pct:        float = 10.0      # top-pct of |head-A| logit to trigger
    k_ticks_passage:  float = 2.0       # head-C K-tick first-passage target
    commission_ticks: float = ES_RT_COMMISSION_TICKS
    # head-C interpretation: probability threshold above which the model claims
    # the cumulative-K-tick first-passage has been satisfied along the trade.
    head_c_threshold: float = 0.5
    # head-A sign-flip detection: require absolute logit > this magnitude on
    # the OPPOSITE side before treating it as a flip (avoids zero-crossing
    # noise). 0.0 = pure sign change.
    head_a_flip_min_abs: float = 0.0
    # Minimum head-B persistence (sec) used as a floor — protects against
    # zero-sec timeouts that would close before any fill could occur.
    head_b_min_seconds: float = 0.05
    # Maximum head-B persistence (sec) used as a ceiling — sanity bound on
    # any pathological huge prediction.
    head_b_max_seconds: float = 60.0


@dataclass
class Trade:
    """One completed trade record (entry + exit, or cancel without fill)."""
    entry_ts_ns:  int
    exit_ts_ns:   int
    side:         str         # 'long' or 'short'
    entry_px:     float       # in ticks above arbitrary day origin (raw_price / TICK_RAW)
    exit_px:      float
    ticks_gross:  float       # signed ticks before commission
    ticks_net:    float       # ticks_gross - commission_ticks (always charged)
    exit_reason:  str         # one of: 'head-A flip', 'head-B timeout',
                              # 'head-C target', 'cancel_unfilled'
    regime:       str         # 'trend'|'mr'|'noise' — from head-D at entry
    queue_ahead:  int         # FIFO queue ahead at the moment of submit
    head_b_sec:   float       # predicted persistence at entry (timeout budget)
    pred_strength: float      # |head-A logit| at entry


# ──────────────────────────────────────────────────────────────────────────────
# Public API: grade one OOT day
# ──────────────────────────────────────────────────────────────────────────────
def grade_one_day(
    events:      np.ndarray,    # (M, 6+) MBO events: at minimum
                                # [ts_ns, action, side, price_raw, size, oid]
    timestamps:  np.ndarray,    # (M,)   ts_recv_ns aligned to events
    head_A:      np.ndarray,    # (M,)   pressure-direction logit, per-event
    head_B:      np.ndarray,    # (M,)   pressure-persistence in SECONDS
    head_C:      np.ndarray,    # (M,)   cum-K-tick first-passage probability
    head_D:      np.ndarray,    # (M,)   regime class id: 0=trend 1=mr 2=noise
    cfg:         VHConfig = VHConfig(),
) -> Tuple[List[Trade], Dict]:
    """
    Run the variable-horizon FIFO grader on one OOT day.

    The `events` array is expected to be one row per MBO message with columns
    (ts_ns, action, side, price_raw, size, oid). action/side are byte chars
    matching Databento conventions: A=add, C=cancel, T=trade; B=bid, A=ask.

    All head_* arrays are aligned 1:1 with events (per HC #497 R5 the v3.5
    inference path emits per-event head outputs after the model's windowing).

    Returns
    -------
    trades : list[Trade]
    summary : dict
        {n_trades, mean_ticks_net, median_ticks_net, wr, pf, sharpe_day,
         per_regime: {trend:{sharpe,n}, mr:{...}, noise:{...}},
         regime_skew: float}
    """
    n_events = len(events)
    assert len(timestamps) == n_events
    for nm, arr in (('A', head_A), ('B', head_B), ('C', head_C), ('D', head_D)):
        assert len(arr) == n_events, f"head_{nm} length {len(arr)} != events {n_events}"

    # Entry threshold: top-pct of |head-A|.
    # When |head-A| is sparse (most zeros, a few strong values), the quantile
    # may itself be zero, which would suppress all signals. Fall back to the
    # smallest non-zero |head-A| in the top-pct slice in that case.
    abs_A = np.abs(head_A)
    if cfg.entry_pct >= 100.0:
        entry_thr = 0.0
    else:
        q = 1.0 - (cfg.entry_pct / 100.0)
        entry_thr = float(np.quantile(abs_A, q))
        if entry_thr <= 0.0:
            nonzero = abs_A[abs_A > 0]
            if nonzero.size > 0:
                # Use smallest non-zero |A| so ALL signal-bearing events qualify
                entry_thr = float(nonzero.min())
            else:
                entry_thr = 0.0

    book = OrderBook()
    trades: List[Trade] = []

    # Single-position state machine (HC #74: realistic, no overlapping orders)
    state = 'flat'                 # 'flat' | 'pending' | 'in_trade'
    pending = None                 # dict for unfilled limit
    in_trade = None                # dict for filled position
    next_oid = SIM_OID_COUNTER_START

    for i in range(n_events):
        ev = events[i]
        ts = int(timestamps[i])
        action = _as_byte(ev[1])
        side = _as_byte(ev[2])
        price = int(ev[3])
        size = int(ev[4]) if ev[4] is not None else 0
        oid = int(ev[5]) if ev[5] is not None else 0

        # 1) Apply event to the book (canonical mechanics)
        if action == b'A':
            if side in (S_BID, S_ASK) and size > 0:
                book.add(oid, side, price, size)
        elif action == b'C':
            book.cancel(oid)
        elif action == b'T':
            # aggressor side: in DBN, 'side' on a trade is the AGGRESSOR side
            consumed = book.trade(side, price, size)
            # Did our pending or our in-trade exit-limit get filled?
            if pending is not None and pending['sim_oid'] in consumed:
                # Entry fill
                in_trade = {
                    'side':          pending['side'],
                    'entry_price':   pending['limit_price'],
                    'entry_ts':      ts,
                    'queue_ahead':   pending['queue_ahead_at_submit'],
                    'head_b_sec':    pending['head_b_sec'],
                    'pred_strength': pending['pred_strength'],
                    'regime':        pending['regime'],
                    'timeout_ts':    ts + int(pending['head_b_sec'] * 1e9),
                    'exit_oid':      None,
                    'exit_limit_px': None,
                }
                state = 'in_trade'
                pending = None
            if in_trade is not None and in_trade.get('exit_oid') in consumed:
                # Exit-limit fill
                trades.append(_close_trade(
                    in_trade, exit_price=in_trade['exit_limit_px'],
                    exit_ts=ts, exit_reason=in_trade['exit_reason_pending'],
                    cfg=cfg,
                ))
                in_trade = None
                state = 'flat'

        # 2) Cancel pending if head-B persistence has expired
        if pending is not None and ts >= pending['cancel_deadline_ns']:
            book.cancel(pending['sim_oid'])
            trades.append(Trade(
                entry_ts_ns=pending['signal_ts'],
                exit_ts_ns=ts,
                side=pending['side'],
                entry_px=float(pending['limit_price']) / TICK_RAW,
                exit_px=float(pending['limit_price']) / TICK_RAW,
                ticks_gross=0.0,
                ticks_net=0.0,                # no commission if no fill (HC #74)
                exit_reason='cancel_unfilled',
                regime=pending['regime'],
                queue_ahead=pending['queue_ahead_at_submit'],
                head_b_sec=pending['head_b_sec'],
                pred_strength=pending['pred_strength'],
            ))
            pending = None
            state = 'flat'

        # 3) Manage in-trade exits — first-fire-wins
        if in_trade is not None:
            exit_reason = _check_exit_reason(
                in_trade, head_A[i], head_B[i], head_C[i], ts, cfg,
            )
            if exit_reason is not None:
                # Submit passive-limit exit at opposite side; if no liquidity
                # there, fall through to next event (continues to evaluate).
                bb = book.best_bid()
                ba = book.best_ask()
                exit_side_book = S_ASK if in_trade['side'] == 'long' else S_BID
                exit_px_raw = bb if exit_side_book == S_BID else ba
                if exit_px_raw is None:
                    # No liquidity on exit side -> cross with the available
                    # opposite side as a marketable exit (HC #74 stop-out path).
                    fallback = ba if exit_side_book == S_BID else bb
                    if fallback is None:
                        # Both sides empty — defer, can't exit yet.
                        continue
                    trades.append(_close_trade(
                        in_trade, exit_price=fallback, exit_ts=ts,
                        exit_reason=exit_reason, cfg=cfg,
                    ))
                    in_trade = None
                    state = 'flat'
                else:
                    # Place passive limit; mark for fill on subsequent trade.
                    next_oid += 1
                    exit_oid = next_oid
                    pl = (book.bids if exit_side_book == S_BID else book.asks).get(exit_px_raw)
                    qty_ahead = pl.total_qty() if pl is not None else 0
                    book.add(exit_oid, exit_side_book, exit_px_raw, 1)
                    in_trade['exit_oid'] = exit_oid
                    in_trade['exit_limit_px'] = exit_px_raw
                    in_trade['exit_reason_pending'] = exit_reason
                    in_trade['exit_queue_ahead'] = qty_ahead
                continue

        # 4) Look for an entry signal (only when flat). Note: we require
        #    abs_A[i] > 0 to avoid firing on the zero-quantile edge case.
        if state == 'flat' and abs_A[i] >= entry_thr and abs_A[i] > 0:
            entry_side = 'long' if head_A[i] > 0 else 'short'
            bb = book.best_bid()
            ba = book.best_ask()
            if bb is None or ba is None:
                continue  # no top-of-book yet
            limit_price = bb if entry_side == 'long' else ba
            book_side = S_BID if entry_side == 'long' else S_ASK
            # Persistence in seconds, clipped to safe range
            hb = float(np.clip(head_B[i], cfg.head_b_min_seconds, cfg.head_b_max_seconds))
            cancel_deadline = ts + int(hb * 1e9)
            # Submit our passive limit at end of queue
            pl = (book.bids if book_side == S_BID else book.asks).get(limit_price)
            queue_ahead = pl.total_qty() if pl is not None else 0
            next_oid += 1
            sim_oid = next_oid
            book.add(sim_oid, book_side, limit_price, 1)
            regime_idx = int(head_D[i]) if 0 <= int(head_D[i]) < len(REGIME_LABELS) else 2
            pending = {
                'sim_oid':              sim_oid,
                'side':                 entry_side,
                'limit_price':          limit_price,
                'signal_ts':            ts,
                'cancel_deadline_ns':   cancel_deadline,
                'head_b_sec':           hb,
                'queue_ahead_at_submit': queue_ahead,
                'pred_strength':        float(abs_A[i]),
                'regime':               REGIME_LABELS[regime_idx],
            }
            state = 'pending'

    # End-of-day cleanup: any pending limit -> cancel_unfilled
    if pending is not None:
        trades.append(Trade(
            entry_ts_ns=pending['signal_ts'],
            exit_ts_ns=int(timestamps[-1]),
            side=pending['side'],
            entry_px=float(pending['limit_price']) / TICK_RAW,
            exit_px=float(pending['limit_price']) / TICK_RAW,
            ticks_gross=0.0,
            ticks_net=0.0,
            exit_reason='cancel_unfilled',
            regime=pending['regime'],
            queue_ahead=pending['queue_ahead_at_submit'],
            head_b_sec=pending['head_b_sec'],
            pred_strength=pending['pred_strength'],
        ))
    # Any still-open in-trade at EOD -> force-exit at last known px. If an
    # exit-limit was already in flight when EOD hit, preserve the reason that
    # placed it. Otherwise label as 'head-B timeout' (force-close at horizon
    # end is semantically a timeout per HC #497 R6).
    if in_trade is not None:
        last_ts = int(timestamps[-1])
        bb = book.best_bid()
        ba = book.best_ask()
        exit_px = (ba if in_trade['side'] == 'long' else bb)
        if exit_px is None:
            exit_px = in_trade['entry_price']  # zero-PnL fallback
        reason = in_trade.get('exit_reason_pending', 'head-B timeout')
        trades.append(_close_trade(
            in_trade, exit_price=exit_px, exit_ts=last_ts,
            exit_reason=reason, cfg=cfg,
        ))

    summary = _summarize_day(trades)
    return trades, summary


# ──────────────────────────────────────────────────────────────────────────────
# Exit-reason resolver — first-fire-wins
# ──────────────────────────────────────────────────────────────────────────────
def _check_exit_reason(in_trade: dict, a_i: float, b_i: float, c_i: float,
                       ts_now: int, cfg: VHConfig) -> Optional[str]:
    """Return exit_reason str or None. Priority: A-flip > C-target > B-timeout
    (A-flip is the strongest signal — regime change detected by the model)."""
    # head-A flip
    if in_trade['side'] == 'long' and a_i < -cfg.head_a_flip_min_abs:
        return 'head-A flip'
    if in_trade['side'] == 'short' and a_i > cfg.head_a_flip_min_abs:
        return 'head-A flip'
    # head-C target (first-passage probability crossed threshold)
    if c_i >= cfg.head_c_threshold:
        return 'head-C target'
    # head-B timeout (countdown from entry; b_i is ignored mid-trade — the
    # entry-time prediction governs, per HC #428 R2 MFE-within-horizon).
    if ts_now >= in_trade['timeout_ts']:
        return 'head-B timeout'
    return None


def _close_trade(in_trade: dict, exit_price: int, exit_ts: int,
                 exit_reason: str, cfg: VHConfig) -> Trade:
    sign = +1.0 if in_trade['side'] == 'long' else -1.0
    ticks_gross = sign * (exit_price - in_trade['entry_price']) / TICK_RAW
    ticks_net = ticks_gross - cfg.commission_ticks
    return Trade(
        entry_ts_ns=in_trade['entry_ts'],
        exit_ts_ns=exit_ts,
        side=in_trade['side'],
        entry_px=in_trade['entry_price'] / TICK_RAW,
        exit_px=exit_price / TICK_RAW,
        ticks_gross=float(ticks_gross),
        ticks_net=float(ticks_net),
        exit_reason=exit_reason,
        regime=in_trade['regime'],
        queue_ahead=in_trade['queue_ahead'],
        head_b_sec=in_trade['head_b_sec'],
        pred_strength=in_trade['pred_strength'],
    )


def _as_byte(v) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, (bytearray, np.bytes_)):
        return bytes(v)
    if isinstance(v, str):
        return v.encode()
    if isinstance(v, (int, np.integer)):
        return bytes([int(v)])
    return b''


# ──────────────────────────────────────────────────────────────────────────────
# Summaries & aggregation
# ──────────────────────────────────────────────────────────────────────────────
def _summarize_day(trades: List[Trade]) -> Dict:
    filled = [t for t in trades if t.exit_reason != 'cancel_unfilled']
    n = len(filled)
    if n == 0:
        return {
            'n_trades': 0, 'n_cancelled': len(trades),
            'mean_ticks_net': 0.0, 'median_ticks_net': 0.0,
            'wr': 0.0, 'pf': 0.0, 'sharpe_day': 0.0,
            'per_regime': {r: {'n': 0, 'sharpe': 0.0, 'mean_ticks_net': 0.0}
                           for r in REGIME_LABELS},
            'regime_skew': 0.0,
        }
    rets = np.array([t.ticks_net for t in filled], dtype=np.float64)
    wins = rets > 0
    gp = float(rets[wins].sum()) if wins.any() else 0.0
    gl = float(-rets[~wins].sum()) if (~wins).any() else 0.0
    pf = (gp / gl) if gl > 0 else float('inf') if gp > 0 else 0.0
    mean_r = float(rets.mean())
    std_r = float(rets.std(ddof=1)) if n > 1 else 0.0
    # Sharpe-of-day: per-trade Sharpe annualized w.r.t. sqrt(252) days
    sharpe_day = (mean_r / std_r * float(np.sqrt(252))) if std_r > 0 else 0.0

    per_regime = {}
    for r in REGIME_LABELS:
        sub = np.array([t.ticks_net for t in filled if t.regime == r],
                       dtype=np.float64)
        if len(sub) == 0:
            per_regime[r] = {'n': 0, 'sharpe': 0.0, 'mean_ticks_net': 0.0}
        else:
            m = float(sub.mean())
            s = float(sub.std(ddof=1)) if len(sub) > 1 else 0.0
            sr = (m / s * float(np.sqrt(252))) if s > 0 else 0.0
            per_regime[r] = {'n': int(len(sub)), 'sharpe': sr,
                             'mean_ticks_net': m}

    # Regime skew: HC #428 R1 uses green/red day classification. Here at the
    # trade-stratified level we report max pairwise skew across regimes as a
    # diagnostic; the cross-fold aggregator does the green/red-day check.
    sharpes = [v['sharpe'] for v in per_regime.values() if v['n'] > 0]
    if len(sharpes) >= 2:
        mx = max(abs(s) for s in sharpes)
        regime_skew = (max(sharpes) - min(sharpes)) / mx if mx > 0 else 0.0
    else:
        regime_skew = 0.0

    return {
        'n_trades':         int(n),
        'n_cancelled':      int(len(trades) - n),
        'mean_ticks_net':   mean_r,
        'median_ticks_net': float(np.median(rets)),
        'wr':               float(wins.mean()),
        'pf':               pf,
        'sharpe_day':       sharpe_day,
        'per_regime':       per_regime,
        'regime_skew':      regime_skew,
    }


def aggregate_cross_fold(day_summaries: List[Dict],
                         day_regime_classes: Optional[List[str]] = None,
                         min_positive_days: int = 30,
                         min_trades_per_day_mean: float = 5.0,
                         min_sharpe: float = 1.0,
                         max_regime_skew: float = 0.50) -> Dict:
    """
    HC #494 R1 viability bar. `day_regime_classes` is per-day {'green','red',
    'flat'} from ES close-to-close (HC #428 R1). If supplied, we compute
    Sharpe_green and Sharpe_red and enforce the skew bound.
    PASS requires all four gates to hold.
    """
    n_days = len(day_summaries)
    n_positive = sum(1 for s in day_summaries if s['mean_ticks_net'] > 0)
    mean_trades = float(np.mean([s['n_trades'] for s in day_summaries])) if n_days else 0.0

    # Pool all-day per-trade returns for an overall Sharpe estimate
    # (cross-fold uses sqrt(n_days) annualization scaling).
    all_means = np.array([s['mean_ticks_net'] for s in day_summaries], dtype=np.float64)
    if all_means.size > 1 and all_means.std(ddof=1) > 0:
        overall_sharpe = float(all_means.mean() / all_means.std(ddof=1)
                               * float(np.sqrt(252)))
    else:
        overall_sharpe = 0.0

    # Green/red regime split (HC #428 R1)
    sharpe_green = sharpe_red = None
    regime_skew = 0.0
    if day_regime_classes is not None and len(day_regime_classes) == n_days:
        def _sub_sharpe(mask: np.ndarray) -> float:
            if mask.sum() < 2:
                return 0.0
            sub = all_means[mask]
            if sub.std(ddof=1) <= 0:
                return 0.0
            return float(sub.mean() / sub.std(ddof=1) * float(np.sqrt(252)))
        gmask = np.array([c == 'green' for c in day_regime_classes])
        rmask = np.array([c == 'red' for c in day_regime_classes])
        sharpe_green = _sub_sharpe(gmask)
        sharpe_red = _sub_sharpe(rmask)
        denom = max(abs(sharpe_green), abs(sharpe_red))
        regime_skew = abs(sharpe_green - sharpe_red) / denom if denom > 0 else 0.0

    gates = {
        'gate_a_positive_days': n_positive >= min_positive_days,
        'gate_b_trades_per_day': mean_trades >= min_trades_per_day_mean,
        'gate_c_sharpe':         overall_sharpe >= min_sharpe,
        'gate_d_regime_skew':    (regime_skew <= max_regime_skew
                                  if day_regime_classes is not None else True),
    }
    pass_all = all(gates.values())
    return {
        'n_days':         n_days,
        'n_positive':     n_positive,
        'mean_trades':    mean_trades,
        'overall_sharpe': overall_sharpe,
        'sharpe_green':   sharpe_green,
        'sharpe_red':     sharpe_red,
        'regime_skew':    regime_skew,
        'gates':          gates,
        'pass':           pass_all,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def write_day_outputs(out_dir: Path, date_str: str,
                      trades: List[Trade], summary: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Trade CSV
    csv_path = out_dir / f"{date_str}_trades.csv"
    with open(csv_path, 'w') as f:
        f.write("entry_ts_ns,exit_ts_ns,side,entry_px,exit_px,"
                "ticks_gross,ticks_net,exit_reason,regime,queue_ahead,"
                "head_b_sec,pred_strength\n")
        for t in trades:
            f.write(f"{t.entry_ts_ns},{t.exit_ts_ns},{t.side},"
                    f"{t.entry_px:.4f},{t.exit_px:.4f},"
                    f"{t.ticks_gross:.4f},{t.ticks_net:.4f},"
                    f"{t.exit_reason},{t.regime},{t.queue_ahead},"
                    f"{t.head_b_sec:.4f},{t.pred_strength:.4f}\n")
    # Summary JSON
    with open(out_dir / f"{date_str}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def _load_npz_with_heads(pred_path: Path) -> Dict[str, np.ndarray]:
    d = np.load(pred_path, allow_pickle=True)
    keys = list(d.keys())
    need = ['head_A', 'head_B', 'head_C', 'head_D']
    out = {}
    for k in need:
        if k in keys:
            out[k] = d[k]
        elif k.lower() in keys:
            out[k] = d[k.lower()]
        else:
            raise KeyError(f"prediction npz missing required key: {k} "
                           f"(found: {keys})")
    return out


def _load_mbo_events_npz(mbo_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(mbo_path, allow_pickle=True)
    if 'events' in d and 'timestamps' in d:
        return d['events'], d['timestamps']
    raise KeyError(f"mbo npz missing 'events' or 'timestamps': "
                   f"{list(d.keys())}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred-npz', type=Path, required=True,
                    help='v3.5 multi-head OOT prediction npz')
    ap.add_argument('--mbo-npz', type=Path, required=True,
                    help='matching processed MBO events npz')
    ap.add_argument('--output-dir', type=Path,
                    default=LVL3_ROOT / 'output' / 'var_horizon_fifo')
    ap.add_argument('--entry-pct', type=float, default=10.0,
                    help='top-pct of |head-A| to trigger (default 10)')
    ap.add_argument('--k-ticks', type=float, default=2.0,
                    help='head-C cumulative-K-tick passage target')
    ap.add_argument('--date', type=str, default=None,
                    help='date label for output filenames (default: stem of mbo npz)')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    heads = _load_npz_with_heads(args.pred_npz)
    events, timestamps = _load_mbo_events_npz(args.mbo_npz)
    date_str = args.date or args.mbo_npz.stem.replace('_v35', '').replace('_mbo_events', '')

    cfg = VHConfig(entry_pct=args.entry_pct, k_ticks_passage=args.k_ticks)
    log.info(f"Grading {date_str}: {len(events)} events, "
             f"entry_pct={cfg.entry_pct}, K={cfg.k_ticks_passage}")
    trades, summary = grade_one_day(
        events=events, timestamps=timestamps,
        head_A=heads['head_A'], head_B=heads['head_B'],
        head_C=heads['head_C'], head_D=heads['head_D'],
        cfg=cfg,
    )
    write_day_outputs(args.output_dir, date_str, trades, summary)
    log.info(f"Day {date_str}: n_trades={summary['n_trades']}, "
             f"mean_ticks_net={summary['mean_ticks_net']:+.4f}, "
             f"WR={summary['wr']:.3f}, PF={summary['pf']:.2f}, "
             f"Sharpe={summary['sharpe_day']:.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
