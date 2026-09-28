"""
MBO Order-by-Order Replay Fill Simulator  v2
=============================================
Canonical fill sim. The ONLY fill sim — bar-level sims are retired.

Reads raw Databento MBO DBN files, reconstructs full order book,
and simulates realistic limit/market/chase order fills.

Architecture:
    MBOReplayEngine  — streams events, maintains OrderBook, dispatches to FillSimulator
    OrderBook        — full FIFO queue per price level
    FillSimulator    — accepts signals, places sim orders, tracks fills/exits
    run_fill_sim()   — day-level entry point, returns long/short/combined stats

Commission: $4.70/RT = $2.35/side = 0.376 ticks ES ($12.50/tick)

Order types:
    'limit'  — join queue at bid (long) or ask (short), passive fill only
    'market' — immediate fill at best ask (long) or best bid (short)
    'chase'  — limit order that reprices up to N times if not filled within reprice_after_ns

Usage:
    results = run_fill_sim(
        date='20260309',
        instrument_id=42140878,
        signal_ts_ns=[...],
        directions=['long', 'short', ...],
        tp_ticks=2.0, sl_ticks=1.0,
        order_type='limit',
        cancel_after_ns=30_000_000_000,
    )
    # results['long'], results['short'], results['combined']
"""

import numpy as np
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import databento as db

# ── ES constants ─────────────────────────────────────────────────────────────
TICK_RAW         = 250_000_000       # 0.25 points in Databento fixed-point (1e9 per point)
TICK_USD         = 12.50             # $ per tick
COMMISSION_RT    = 4.70              # $ round-trip per contract
COMMISSION_TICKS = COMMISSION_RT / TICK_USD   # 0.376 ticks

# ── DBN action/side bytes ─────────────────────────────────────────────────────
A_ADD    = b'A'
A_CANCEL = b'C'
A_MODIFY = b'M'
A_TRADE  = b'T'
A_FILL   = b'F'
A_RESET  = b'R'
S_BID    = b'B'
S_ASK    = b'A'
S_NONE   = b'N'

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_MBO_DIRS = [
    '/home/jupiter/Lvl3Quant/data/raw_mbo',
    '/home/jupiter/Lvl3Quant/data/raw/mbo',
    r'C:\Users\Footb\Documents\Github\Lvl3Quant\data\raw\mbo',
    r'C:\Users\nick\Lvl3Quant\data\raw\mbo',
]


def find_dbn_path(date: str) -> str:
    fname = f'glbx-mdp3-{date}.mbo.dbn.zst'
    for d in RAW_MBO_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'DBN file for {date} not found. Searched: {RAW_MBO_DIRS}')


# ── Order book ────────────────────────────────────────────────────────────────

class PriceLevel:
    """FIFO queue at one price level."""
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
        self.bids = {}  # price_raw -> PriceLevel
        self.asks = {}
        self._oid_side  = {}  # order_id -> side bytes
        self._oid_price = {}  # order_id -> price_raw

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
        """
        Trade at price with aggressor_side. Consumes passive side FIFO.
        Returns list of order_ids fully consumed.
        """
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

    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid_raw(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2.0 if bb and ba else None

    def spread_ticks(self):
        bb, ba = self.best_bid(), self.best_ask()
        return (ba - bb) / TICK_RAW if bb and ba else None

    def qty_at(self, side, price: int) -> int:
        b = self._book(side)
        return b[price].total_qty() if price in b else 0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SimOrder:
    sim_oid:          int
    direction:        str
    signal_ts_ns:     int
    order_type:       str
    entry_price:      int
    tp_price:         int
    sl_price:         int
    tp_ticks:         float
    sl_ticks:         float
    queue_ahead:      int
    mid_at_signal:    float
    cancel_after_ns:  int
    chase_reprices:   int = 0
    max_reprices:     int = 3
    reprice_after_ns: int = 1_000_000_000
    last_reprice_ts:  int = 0
    filled:           bool = False
    fill_price:       int  = 0
    fill_ts_ns:       int  = 0


@dataclass
class FillResult:
    signal_ts_ns:    int
    direction:       str
    order_type:      str
    entry_price:     int
    entry_ts_ns:     int
    exit_price:      int
    exit_ts_ns:      int
    exit_reason:     str    # 'tp' | 'sl' | 'timeout' | 'eod'
    queue_ahead:     int
    queue_wait_ns:   int
    pnl_ticks:       float
    pnl_ticks_net:   float
    slippage_ticks:  float
    mid_at_signal:   float
    spread_at_signal: object  # float or None


# ── Main simulator ────────────────────────────────────────────────────────────

class MBOFillSim:
    """
    Full MBO order-by-order replay fill simulator for one day.
    """

    def __init__(self, date: str, instrument_id: int,
                 cancel_after_ns: int = 30_000_000_000,
                 max_reprices: int = 3,
                 reprice_after_ns: int = 1_000_000_000):
        self.instrument_id   = instrument_id
        self.cancel_after_ns = cancel_after_ns
        self.max_reprices    = max_reprices
        self.reprice_after_ns = reprice_after_ns
        self._load(date)

    def _load(self, date: str):
        path  = find_dbn_path(date)
        store = db.DBNStore.from_file(path)
        recs  = store.to_ndarray()
        mask  = recs['instrument_id'] == self.instrument_id
        self.records = recs[mask]
        self._ts     = self.records['ts_recv'].astype(np.int64)
        print(f'Loaded {len(self.records):,} events for instr={self.instrument_id}')

    @staticmethod
    def _bytes(v) -> bytes:
        if isinstance(v, bytes):
            return v
        if isinstance(v, np.bytes_):
            return bytes(v)
        return bytes([v])

    def simulate(self, signals: list, tp_ticks: float, sl_ticks: float,
                 order_type: str = 'limit') -> list:
        signals = sorted(signals, key=lambda s: s['ts_ns'])
        results  = []
        pending  = []
        sim_oid_ctr = 900_000_000
        sig_idx  = 0
        book     = OrderBook()

        for rec in self.records:
            ts     = int(rec['ts_recv'])
            action = self._bytes(rec['action'])
            side   = self._bytes(rec['side'])
            price  = int(rec['price'])
            qty    = int(rec['size'])
            oid    = int(rec['order_id'])

            # 1. Update book
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

            # 2. Accept new signals
            while sig_idx < len(signals) and signals[sig_idx]['ts_ns'] <= ts:
                sig = signals[sig_idx]; sig_idx += 1
                mid = book.mid_raw()
                bb  = book.best_bid()
                ba  = book.best_ask()
                if mid is None or bb is None or ba is None:
                    continue

                direction = sig['direction']
                spread    = book.spread_ticks()

                if order_type == 'market':
                    fill_price = ba if direction == 'long' else bb
                    slip = ((fill_price - mid) / TICK_RAW if direction == 'long'
                            else (mid - fill_price) / TICK_RAW)
                    tp = (fill_price + int(tp_ticks * TICK_RAW) if direction == 'long'
                          else fill_price - int(tp_ticks * TICK_RAW))
                    sl = (fill_price - int(sl_ticks * TICK_RAW) if direction == 'long'
                          else fill_price + int(sl_ticks * TICK_RAW))
                    exit_p, exit_ts, exit_r = self._scan_exit(ts, direction, tp, sl, book)
                    pnl = self._pnl(fill_price, exit_p, direction)
                    results.append(FillResult(
                        signal_ts_ns=sig['ts_ns'], direction=direction,
                        order_type=order_type,
                        entry_price=fill_price, entry_ts_ns=ts,
                        exit_price=exit_p, exit_ts_ns=exit_ts, exit_reason=exit_r,
                        queue_ahead=0, queue_wait_ns=0,
                        pnl_ticks=pnl, pnl_ticks_net=pnl - COMMISSION_TICKS,
                        slippage_ticks=slip, mid_at_signal=mid,
                        spread_at_signal=spread,
                    ))
                    continue

                # limit / chase
                entry_price  = bb if direction == 'long' else ba
                passive_side = S_BID if direction == 'long' else S_ASK
                queue_ahead  = book.qty_at(passive_side, entry_price)

                sim_oid = sim_oid_ctr; sim_oid_ctr += 1
                book.add(sim_oid, passive_side, entry_price, 1)

                tp = (entry_price + int(tp_ticks * TICK_RAW) if direction == 'long'
                      else entry_price - int(tp_ticks * TICK_RAW))
                sl = (entry_price - int(sl_ticks * TICK_RAW) if direction == 'long'
                      else entry_price + int(sl_ticks * TICK_RAW))

                pending.append(SimOrder(
                    sim_oid=sim_oid, direction=direction,
                    signal_ts_ns=sig['ts_ns'], order_type=order_type,
                    entry_price=entry_price, tp_price=tp, sl_price=sl,
                    tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                    queue_ahead=queue_ahead, mid_at_signal=mid,
                    cancel_after_ns=self.cancel_after_ns,
                    max_reprices=self.max_reprices,
                    reprice_after_ns=self.reprice_after_ns,
                    last_reprice_ts=ts,
                ))

            # 3. Check pending orders
            still_pending = []
            for o in pending:
                if o.sim_oid in consumed_oids:
                    o.filled     = True
                    o.fill_price = o.entry_price
                    o.fill_ts_ns = ts

                if o.filled:
                    exit_p, exit_ts, exit_r = self._scan_exit(
                        o.fill_ts_ns, o.direction, o.tp_price, o.sl_price, book)
                    pnl  = self._pnl(o.fill_price, exit_p, o.direction)
                    slip = ((o.fill_price - o.mid_at_signal) / TICK_RAW
                            if o.direction == 'long'
                            else (o.mid_at_signal - o.fill_price) / TICK_RAW)
                    results.append(FillResult(
                        signal_ts_ns=o.signal_ts_ns, direction=o.direction,
                        order_type=o.order_type,
                        entry_price=o.fill_price, entry_ts_ns=o.fill_ts_ns,
                        exit_price=exit_p, exit_ts_ns=exit_ts, exit_reason=exit_r,
                        queue_ahead=o.queue_ahead,
                        queue_wait_ns=o.fill_ts_ns - o.signal_ts_ns,
                        pnl_ticks=pnl, pnl_ticks_net=pnl - COMMISSION_TICKS,
                        slippage_ticks=slip, mid_at_signal=o.mid_at_signal,
                        spread_at_signal=None,
                    ))
                    continue

                elapsed = ts - o.signal_ts_ns
                if elapsed > o.cancel_after_ns:
                    book.cancel(o.sim_oid)
                    continue

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
                            o.chase_reprices  += 1
                            o.last_reprice_ts  = ts

                still_pending.append(o)
            pending = still_pending

        for o in pending:
            book.cancel(o.sim_oid)

        return results

    def _scan_exit(self, fill_ts: int, direction: str,
                   tp: int, sl: int, book: OrderBook):
        start_idx = int(np.searchsorted(self._ts, fill_ts))
        for rec in self.records[start_idx:]:
            ts     = int(rec['ts_recv'])
            action = self._bytes(rec['action'])
            price  = int(rec['price'])
            if ts - fill_ts > self.cancel_after_ns:
                mid = book.mid_raw()
                return int(mid) if mid else tp, ts, 'timeout'
            if action in (A_TRADE, A_FILL):
                if direction == 'long':
                    if price >= tp: return tp, ts, 'tp'
                    if price <= sl: return sl, ts, 'sl'
                else:
                    if price <= tp: return tp, ts, 'tp'
                    if price >= sl: return sl, ts, 'sl'
        mid = book.mid_raw()
        last_ts = int(self.records['ts_recv'][-1]) if len(self.records) else fill_ts
        return int(mid) if mid else tp, last_ts, 'eod'

    @staticmethod
    def _pnl(entry: int, exit_p: int, direction: str) -> float:
        raw = exit_p - entry if direction == 'long' else entry - exit_p
        return raw / TICK_RAW


# ── Public API ────────────────────────────────────────────────────────────────

def run_fill_sim(date: str,
                 instrument_id: int,
                 signal_ts_ns: list,
                 directions: list,
                 tp_ticks: float = 2.0,
                 sl_ticks: float = 1.0,
                 order_type: str = 'limit',
                 cancel_after_ns: int = 30_000_000_000,
                 max_reprices: int = 3,
                 reprice_after_ns: int = 1_000_000_000) -> dict:
    """
    Run MBO replay fill sim for one day. Returns long/short/combined stats.
    """
    sim     = MBOFillSim(date, instrument_id, cancel_after_ns,
                         max_reprices, reprice_after_ns)
    signals = [{'ts_ns': t, 'direction': d}
               for t, d in zip(signal_ts_ns, directions)]
    results = sim.simulate(signals, tp_ticks, sl_ticks, order_type)

    n_long  = sum(1 for d in directions if d == 'long')
    n_short = len(directions) - n_long

    def stats(subset, n_sig):
        if not subset:
            return {'n': 0, 'n_signals': n_sig, 'fill_rate': 0.0}
        pnl   = np.array([r.pnl_ticks_net for r in subset])
        exits = [r.exit_reason for r in subset]
        neg   = pnl[pnl < 0]
        return {
            'n':                   len(subset),
            'n_signals':           n_sig,
            'fill_rate':           len(subset) / max(1, n_sig),
            'mean_pnl_net_ticks':  float(pnl.mean()),
            'std_pnl':             float(pnl.std()),
            'sortino':             (float(pnl.mean() / neg.std())
                                    if len(neg) > 1 else float('inf')),
            'win_rate':            float((pnl > 0).mean()),
            'mean_queue_ahead':    float(np.mean([r.queue_ahead for r in subset])),
            'mean_queue_wait_ms':  float(np.mean([r.queue_wait_ns for r in subset]) / 1e6),
            'mean_slippage_ticks': float(np.mean([r.slippage_ticks for r in subset])),
            'tp_rate':             exits.count('tp') / len(exits),
            'sl_rate':             exits.count('sl') / len(exits),
            'timeout_rate':       (exits.count('timeout') + exits.count('eod')) / len(exits),
            'commission_ticks':    COMMISSION_TICKS,
        }

    longs  = [r for r in results if r.direction == 'long']
    shorts = [r for r in results if r.direction == 'short']
    return {
        'long':        stats(longs,   n_long),
        'short':       stats(shorts,  n_short),
        'combined':    stats(results, len(directions)),
        'raw_results': results,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json

    date     = '20260309'
    instr_id = 42140878   # ESH6 on 2026-03-09

    # RTH: 09:30-16:00 ET = 14:30-21:00 UTC
    rth_start = 1_773_052_200_000_000_000
    rth_end   = 1_773_074_400_000_000_000
    n = 20
    ts_list = list(np.linspace(rth_start, rth_end, n, dtype=np.int64))
    dirs    = (['long', 'short'] * 20)[:n]

    print(f'Smoke test: {n} signals on {date}, TP2/SL1, limit orders')
    out = run_fill_sim(
        date=date, instrument_id=instr_id,
        signal_ts_ns=ts_list, directions=dirs,
        tp_ticks=2.0, sl_ticks=1.0,
        order_type='limit',
        cancel_after_ns=30_000_000_000,
    )
    for key in ('long', 'short', 'combined'):
        print(f'\n{key.upper()}:')
        d = {k: v for k, v in out[key].items() if k != 'raw_results'}
        print(json.dumps(d, indent=2, default=str))
    print(f"\nDONE. {len(out['raw_results'])} fills from {n} signals.")
