"""schema_adapter.py — Rithmic ES MBO  <->  Databento US Equities MBO

The 6-column event schema is the contract:
    [0] time_delta_log   = log1p(microseconds since previous event)
    [1] event_type_id    = {A:0, C:1, M:2, T:3, F:4}
    [2] side_id          = {B:0, A:1, N:2}
    [3] price_rel_ticks  = (price - mid) / tick_size
    [4] qty_log          = log(max(1, qty))
    [5] spread_ticks     = (ask - bid) / tick_size

This module gives a single translator + BBO tracker so the recorder code path
is identical regardless of source vendor. Databento delivers MBO with the SAME
action/side characters our ES schema already encodes; the only real work is
unit conversion (price scaling, tick size) and maintaining a running BBO.

Usage (Databento live):
    adapter = SchemaAdapter(tick_size=0.01, instrument="SPY")
    for db_record in client.mbo_stream("SPY"):
        row6, ts_ns = adapter.from_databento_mbo(db_record)
        if row6 is not None:
            events_buf.append(row6); ts_buf.append(ts_ns)

Usage (Rithmic live):
    adapter = SchemaAdapter(tick_size=0.25, instrument="ES")
    adapter.on_rithmic_bbo(bbo_event)        -> returns (row6, ts_ns) or None
    adapter.on_rithmic_trade(trade_event)    -> returns (row6, ts_ns)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Encoding constants — IDENTICAL to mbo_recorder.py metadata
ACTION_ENCODING = {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4}
SIDE_ENCODING = {"B": 0, "A": 1, "N": 2}

# Databento price scaling: prices are int64 * 1e-9 of the quote currency unit.
# For US equities the quote is USD per share, so dollars = raw_int / 1e9.
DATABENTO_PRICE_SCALE = 1e9


@dataclass
class AdaptedEvent:
    """Output of the adapter: the 6-column row plus the int64 ns timestamp."""
    row: list  # length 6, floats
    ts_ns: int


class SchemaAdapter:
    """Stateful adapter — tracks running BBO and previous timestamp.

    A single instance corresponds to one (vendor, symbol) stream. Do NOT share
    an instance across symbols: BBO state would interleave.
    """

    def __init__(self, tick_size: float, instrument: str = ""):
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = float(tick_size)
        self.instrument = instrument

        # Running BBO (in price units, e.g. dollars for SPY, points for ES)
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.best_bid_size: int = 0
        self.best_ask_size: int = 0
        self.mid_price: float = 0.0

        # For time_delta_log
        self.prev_ts_ns: int = 0

    # -----------------------------------------------------------------
    # BBO bookkeeping
    # -----------------------------------------------------------------
    def _update_bbo_on_event(self, side_id: int, action_id: int,
                             price: float, size: int) -> None:
        """Naive top-of-book update: ADD/MODIFY at a better price moves the BBO.

        This is a top-of-book approximation — full L2 book reconstruction
        belongs downstream (existing harness handles depth). The recorder only
        needs an approximate spread for the 6th column, identical to what the
        ES recorder does.
        """
        if action_id == ACTION_ENCODING["T"] or action_id == ACTION_ENCODING["F"]:
            # Trades do not move book quotes by themselves in this approximation.
            return
        if side_id == SIDE_ENCODING["B"]:  # bid
            # New best bid if higher; also update if exactly equal (size refresh)
            if action_id == ACTION_ENCODING["A"]:
                if price >= self.best_bid or self.best_bid == 0.0:
                    self.best_bid = price
                    self.best_bid_size = size
            elif action_id == ACTION_ENCODING["C"]:
                if price == self.best_bid:
                    # Cancel at the top — best_bid_size decays; if hit 0, we
                    # do not know the next level, so leave best_bid as last
                    # known (downstream will see a stale BBO until next ADD).
                    self.best_bid_size = max(0, self.best_bid_size - size)
            elif action_id == ACTION_ENCODING["M"]:
                if price >= self.best_bid:
                    self.best_bid = price
                    self.best_bid_size = size
        elif side_id == SIDE_ENCODING["A"]:  # ask
            if action_id == ACTION_ENCODING["A"]:
                if price <= self.best_ask or self.best_ask == 0.0:
                    self.best_ask = price
                    self.best_ask_size = size
            elif action_id == ACTION_ENCODING["C"]:
                if price == self.best_ask:
                    self.best_ask_size = max(0, self.best_ask_size - size)
            elif action_id == ACTION_ENCODING["M"]:
                if price <= self.best_ask or self.best_ask == 0.0:
                    self.best_ask = price
                    self.best_ask_size = size

        if self.best_bid > 0 and self.best_ask > 0:
            self.mid_price = (self.best_bid + self.best_ask) / 2.0

    def _spread_ticks(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0 and self.best_ask >= self.best_bid:
            return (self.best_ask - self.best_bid) / self.tick_size
        return 0.0

    # -----------------------------------------------------------------
    # Core encoder — identical math to mbo_recorder._encode
    # -----------------------------------------------------------------
    def _encode(self, ts_ns: int, action_id: int, side_id: int,
                price: float, qty: int) -> AdaptedEvent:
        if self.prev_ts_ns and ts_ns >= self.prev_ts_ns:
            delta_us = (ts_ns - self.prev_ts_ns) / 1000.0
        else:
            delta_us = 0.0
        self.prev_ts_ns = ts_ns

        if self.mid_price > 0 and price > 0:
            price_rel = (price - self.mid_price) / self.tick_size
        else:
            price_rel = 0.0

        row = [
            math.log1p(delta_us) if delta_us > 0 else 0.0,
            float(action_id),
            float(side_id),
            float(price_rel),
            math.log(max(1, qty)),
            self._spread_ticks(),
        ]
        return AdaptedEvent(row=row, ts_ns=ts_ns)

    # -----------------------------------------------------------------
    # Databento MBO record adapter
    # -----------------------------------------------------------------
    def from_databento_mbo(self, rec) -> Optional[AdaptedEvent]:
        """Translate one Databento MBO record.

        Accepts either:
          * an object exposing attributes (`rec.ts_event`, `rec.action`,
            `rec.side`, `rec.price`, `rec.size`) — the SDK's typed records, or
          * a plain dict with the same keys.

        Returns None if the record is undecodable (e.g. unknown action).
        """
        get = (lambda k: getattr(rec, k)) if not isinstance(rec, dict) else rec.__getitem__

        # ts_event is ns since epoch (int64 in databento SDK)
        ts_ns = int(get("ts_event"))
        action_raw = get("action")
        side_raw = get("side")
        price_raw = get("price")
        size_raw = get("size")

        # action/side may arrive as 1-char bytes or str; normalize
        if isinstance(action_raw, (bytes, bytearray)):
            action_raw = action_raw.decode("ascii", errors="ignore")
        if isinstance(side_raw, (bytes, bytearray)):
            side_raw = side_raw.decode("ascii", errors="ignore")
        action_raw = str(action_raw).strip()[:1].upper() or "A"
        side_raw = str(side_raw).strip()[:1].upper() or "N"

        action_id = ACTION_ENCODING.get(action_raw)
        side_id = SIDE_ENCODING.get(side_raw)
        if action_id is None or side_id is None:
            return None

        # Databento prices are int64 fixed-point; SDK exposes them either
        # already-floated or as int. Handle both.
        if isinstance(price_raw, (int,)) and abs(price_raw) > 1e6:
            price = price_raw / DATABENTO_PRICE_SCALE
        else:
            price = float(price_raw) if price_raw is not None else 0.0
        qty = int(size_raw) if size_raw is not None else 0

        # Update BBO BEFORE encoding so spread/mid reflect this event
        self._update_bbo_on_event(side_id, action_id, price, qty)
        return self._encode(ts_ns, action_id, side_id, price, qty)

    # -----------------------------------------------------------------
    # Rithmic adapters (kept for parity testing — recorder still owns these
    # in the existing code path; included so a single tool can replay both
    # sides against the same harness)
    # -----------------------------------------------------------------
    def from_rithmic_bbo(self, ev) -> Tuple[Optional[AdaptedEvent], Optional[AdaptedEvent]]:
        """Returns (bid_row, ask_row) — either may be None if not present."""
        ts_ns = int(ev.ssboe * 1_000_000_000 + ev.usecs * 1000)
        bid_out = ask_out = None
        if getattr(ev, "has_bid", False) and ev.bid_price > 0:
            self.best_bid = ev.bid_price
            self.best_bid_size = ev.bid_size
            if self.best_ask > 0:
                self.mid_price = (self.best_bid + self.best_ask) / 2.0
            bid_out = self._encode(ts_ns, ACTION_ENCODING["A"], SIDE_ENCODING["B"],
                                   ev.bid_price, ev.bid_size)
        if getattr(ev, "has_ask", False) and ev.ask_price > 0:
            self.best_ask = ev.ask_price
            self.best_ask_size = ev.ask_size
            if self.best_bid > 0:
                self.mid_price = (self.best_bid + self.best_ask) / 2.0
            ask_out = self._encode(ts_ns, ACTION_ENCODING["A"], SIDE_ENCODING["A"],
                                   ev.ask_price, ev.ask_size)
        return bid_out, ask_out

    def from_rithmic_trade(self, ev) -> AdaptedEvent:
        ts_ns = int(ev.ssboe * 1_000_000_000 + ev.usecs * 1000)
        # Rithmic aggressor: 1 = bid aggressor (=ask traded), 2 = ask aggressor
        side_id = SIDE_ENCODING["A"] if ev.aggressor == 1 else (
                  SIDE_ENCODING["B"] if ev.aggressor == 2 else SIDE_ENCODING["N"])
        return self._encode(ts_ns, ACTION_ENCODING["T"], side_id,
                            ev.trade_price, ev.trade_size)


# -----------------------------------------------------------------------
# Self-test (run: python feeds/schema_adapter.py)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=== SchemaAdapter self-test (SPY @ tick=0.01) ===")
    a = SchemaAdapter(tick_size=0.01, instrument="SPY")
    base_ns = int(time.time() * 1e9)
    fake_book = [
        # ts_ns,           action, side, price,   size
        (base_ns + 0,      "A",    "B",  580.10,  100),
        (base_ns + 1000,   "A",    "A",  580.11,  150),  # spread = 1 cent = 1 tick
        (base_ns + 5000,   "T",    "A",  580.11,   50),  # buyer hit ask
        (base_ns + 12000,  "C",    "B",  580.10,  100),
        (base_ns + 20000,  "A",    "B",  580.09,  200),
    ]
    for ts_ns, act, side, px, sz in fake_book:
        rec = {"ts_event": ts_ns, "action": act, "side": side,
               "price": px, "size": sz}
        out = a.from_databento_mbo(rec)
        print(f"  {act}/{side} {px:.2f} x{sz} -> "
              f"row={[round(v,4) for v in out.row]}  ts_ns={out.ts_ns}")
    print(f"  final BBO: bid={a.best_bid} ask={a.best_ask} "
          f"mid={a.mid_price:.4f} spread_ticks={a._spread_ticks():.2f}")
    print("OK")
