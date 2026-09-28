#!/usr/bin/env python3
"""
signal_engine.py — Orchestrator: Rithmic MD → features → LGBM → orders.

Pipeline per received event:
    1. Convert Rithmic BBOEvent / TradeEvent into a synthetic MBO event row
       (event_type, side, price, qty, spread, time_delta) that matches the
       training data schema.
    2. Call StreamingFeatures.update() → 21-dim feature vector.
    3. Call LGBMInference.predict_tier() → (tier, pred).
    4. Emit a signal if |pred| > threshold AND tier >= min_tier.
    5. Route the signal through RithmicClient.submit_order().

Position tracking:
    * Max 1 contract long or short; otherwise flat.
    * A flip signal (opposite side) closes the current position and opens
      the new one in a single MARKET order of size = 2 * contracts.
    * A timeout (default 30s) force-closes the position if no new signal.
    * Signals are written to a JSONL log for post-trade analysis.

CLI:
    python -m live_trading_linux.signal_engine \
        --model /path/to/fold11/labels_10s_lgbm.pkl \
        --calibration /path/to/calib.json \
        --symbol ESZ5 --exchange CME --paper

Rithmic credentials come from environment variables (see SETUP.md).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal as _sig
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from live_trading_linux.lgbm_inference   import LGBMInference, TIER_ORDER
from live_trading_linux.rithmic_client   import (
    BBOEvent, OrderEvent, RithmicClient, TradeEvent,
)
from live_trading_linux.streaming_features import StreamingFeatures


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path("/home/jupiter/Lvl3Quant/live_trading_linux/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("signal_engine")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / "signal_engine.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    timestamp:     float          # local recv_ts
    exchange_ts:   float          # ssboe + usecs*1e-6
    symbol:        str
    side:          str            # 'B' or 'S'
    pred:          float
    tier:          str
    features_hash: str
    model_version: str


@dataclass
class Position:
    size:         int = 0         # +N long, -N short
    side:         str = ""        # 'B' | 'S' | ''
    entry_price:  float = 0.0
    entry_ts:     float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class SignalEngine:
    """Streaming predictor + paper-trading order router."""

    def __init__(
        self,
        model_path:       str | Path,
        calibration_path: Optional[str | Path],
        symbol:           str,
        exchange:         str,
        threshold:        float = 0.0,
        min_tier:         str = "top25",
        position_timeout_s: float = 30.0,
        signals_log_path: Optional[str | Path] = None,
        dry_run:          bool = False,
    ) -> None:
        self.symbol    = symbol
        self.exchange  = exchange
        self.threshold = float(threshold)
        self.min_tier  = min_tier
        self.min_tier_rank = TIER_ORDER.get(min_tier, 2)
        self.position_timeout_s = float(position_timeout_s)
        self.dry_run   = dry_run

        self.features  = StreamingFeatures()
        self.model     = LGBMInference(model_path, calibration_path)
        self.client    = RithmicClient()

        # State
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.last_event_ts:  Optional[float] = None   # exchange time in seconds
        self.position: Position = Position()
        self._stop = asyncio.Event()

        # Signals log
        if signals_log_path is None:
            signals_log_path = _LOG_DIR / f"signals_{symbol}.jsonl"
        self.signals_log_path = Path(signals_log_path)
        self._signals_fh = open(self.signals_log_path, "a", buffering=1)

        # Wire up callbacks now; the client starts consuming once connect() runs
        self.client.set_md_callback(self._on_md)
        self.client.set_order_callback(self._on_order)

        # Housekeeping task for position timeout
        self._timeout_task: Optional[asyncio.Task] = None

        log.info(
            "SignalEngine configured: symbol=%s exch=%s threshold=%.4f min_tier=%s timeout=%.1fs dry_run=%s",
            symbol, exchange, threshold, min_tier, position_timeout_s, dry_run,
        )

    # ------------------------------------------------------------- lifecycle
    async def run(self) -> None:
        """Main entry — connects to Rithmic and runs until signalled to stop."""
        await self.client.connect()
        await self.client.subscribe_md(self.symbol, self.exchange)

        self._timeout_task = asyncio.create_task(
            self._position_timeout_loop(), name="position_timeout"
        )

        log.info("SignalEngine running. Waiting for MD...")
        try:
            await self._stop.wait()
        finally:
            if self._timeout_task is not None:
                self._timeout_task.cancel()
                try:
                    await self._timeout_task
                except Exception:
                    pass
            await self.client.disconnect()
            try:
                self._signals_fh.flush()
                self._signals_fh.close()
            except Exception:
                pass
            log.info("SignalEngine stopped.")

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- MD handler
    async def _on_md(self, event) -> None:
        """Callback from RithmicClient: handles BBOEvent and TradeEvent."""
        try:
            if isinstance(event, BBOEvent):
                await self._handle_bbo(event)
            elif isinstance(event, TradeEvent):
                await self._handle_trade(event)
        except Exception as e:
            log.exception("_on_md failed: %s", e)

    async def _handle_bbo(self, ev: BBOEvent) -> None:
        """BBO update → synthesise add/cancel events per changed side.

        BBO tells us the top-of-book price+size.  Since we don't have the
        full MBO stream, we approximate: whenever a side changes, we emit
        one synthetic 'add' event on that side with qty = delta-size (or 1 if
        only price changed).  This is a lossy proxy but sufficient as a v1
        pipeline — documented as a known gap in SETUP.md.
        """
        prev_bid = self.best_bid
        prev_ask = self.best_ask
        if ev.has_bid:
            self.best_bid = ev.bid_price
        if ev.has_ask:
            self.best_ask = ev.ask_price

        spread = self.best_ask - self.best_bid if (self.best_bid and self.best_ask) else 0.0
        now_exch = self._exchange_ts(ev.ssboe, ev.usecs)

        # Emit a synthetic add event for the side that changed
        if ev.has_bid and ev.bid_price != prev_bid:
            await self._process_event(
                event_type=0,        # add
                side=0,              # bid
                price=ev.bid_price,
                qty=max(1, int(ev.bid_size) if ev.bid_size else 1),
                spread=spread,
                exch_ts=now_exch,
                local_ts=ev.recv_ts,
                symbol=ev.symbol,
            )
        if ev.has_ask and ev.ask_price != prev_ask:
            await self._process_event(
                event_type=0,        # add
                side=1,              # ask
                price=ev.ask_price,
                qty=max(1, int(ev.ask_size) if ev.ask_size else 1),
                spread=spread,
                exch_ts=now_exch,
                local_ts=ev.recv_ts,
                symbol=ev.symbol,
            )

    async def _handle_trade(self, ev: TradeEvent) -> None:
        spread = self.best_ask - self.best_bid if (self.best_bid and self.best_ask) else 0.0
        # aggressor: 1=BUY (lifts offer → side=ask), 2=SELL (hits bid → side=bid)
        if ev.aggressor == 1:
            side = 1  # ask (trade lifted the offer)
        elif ev.aggressor == 2:
            side = 0  # bid (trade hit the bid)
        else:
            # Unknown aggressor — classify by price vs book
            if ev.trade_price >= self.best_ask and self.best_ask > 0:
                side = 1
            elif ev.trade_price <= self.best_bid and self.best_bid > 0:
                side = 0
            else:
                side = 1  # arbitrary tie-break
        now_exch = self._exchange_ts(ev.ssboe, ev.usecs)
        await self._process_event(
            event_type=3,            # trade
            side=side,
            price=ev.trade_price,
            qty=int(ev.trade_size),
            spread=spread,
            exch_ts=now_exch,
            local_ts=ev.recv_ts,
            symbol=ev.symbol,
        )

    @staticmethod
    def _exchange_ts(ssboe: int, usecs: int) -> float:
        return float(ssboe) + float(usecs) * 1e-6

    # ---------------------------------------------------------- core pipeline
    async def _process_event(
        self,
        event_type: int,
        side:       int,
        price:      float,
        qty:        int,
        spread:     float,
        exch_ts:    float,
        local_ts:   float,
        symbol:     str,
    ) -> None:
        """Compute features, predict, decide, route."""
        if self.last_event_ts is None:
            time_delta = 0.0
        else:
            time_delta = max(0.0, exch_ts - self.last_event_ts)
        self.last_event_ts = exch_ts

        feat = self.features.update(
            event_type=event_type,
            side=side,
            price=float(price),
            qty=float(qty),
            spread=float(spread),
            time_delta=float(time_delta),
        )
        tier, pred = self.model.predict_tier(feat)

        # Short-circuit: ignore until minimum warmup
        if self.features.n_events < self.features.W500:
            return

        # Threshold / tier gating
        if abs(pred) < self.threshold:
            return
        if TIER_ORDER[tier] < self.min_tier_rank:
            return

        desired_side = "B" if pred > 0 else "S"

        # Build + emit signal
        fh = hashlib.sha1(feat.tobytes()).hexdigest()[:12]
        sig = Signal(
            timestamp=local_ts,
            exchange_ts=exch_ts,
            symbol=symbol,
            side=desired_side,
            pred=float(pred),
            tier=tier,
            features_hash=fh,
            model_version=self.model.model_version,
        )
        self._write_signal(sig)
        log.info("SIGNAL %s %s pred=%+.4f tier=%s pos=%+d",
                 sig.side, sig.symbol, sig.pred, sig.tier, self.position.size)

        await self._act_on_signal(sig, ref_price=price, ref_ts=local_ts)

    def _write_signal(self, s: Signal) -> None:
        try:
            self._signals_fh.write(json.dumps(asdict(s)) + "\n")
        except Exception as e:
            log.warning("signal log write failed: %s", e)

    # ------------------------------------------------------- position management
    async def _act_on_signal(self, sig: Signal, ref_price: float, ref_ts: float) -> None:
        """Map signal → order(s), respecting max-1 position."""
        want_long  = (sig.side == "B")
        want_short = (sig.side == "S")
        pos = self.position

        if pos.size == 0:
            # Flat → open 1 contract in signal direction
            await self._submit(sig.side, qty=1, reason="open")
            pos.size = +1 if want_long else -1
            pos.side = sig.side
            pos.entry_price = ref_price
            pos.entry_ts    = ref_ts
            return

        same_direction = (pos.size > 0 and want_long) or (pos.size < 0 and want_short)
        if same_direction:
            # Already in the right direction, and cap is 1 contract — do nothing.
            return

        # Opposite signal → flip: close and re-open in one 2x market order.
        flip_qty = 2 * abs(pos.size)
        await self._submit(sig.side, qty=flip_qty, reason="flip")
        pos.size = +1 if want_long else -1
        pos.side = sig.side
        pos.entry_price = ref_price
        pos.entry_ts    = ref_ts

    async def _submit(self, side: str, qty: int, reason: str) -> None:
        log.info("ORDER.out %s %d %s (%s)", side, qty, self.symbol, reason)
        if self.dry_run:
            return
        try:
            await self.client.submit_order(
                symbol=self.symbol,
                side=side,
                qty=int(qty),
                price_type="MARKET",
                exchange=self.exchange,
            )
        except Exception as e:
            log.exception("submit_order failed: %s", e)

    async def _close_position(self, reason: str) -> None:
        pos = self.position
        if pos.size == 0:
            return
        close_side = "S" if pos.size > 0 else "B"
        await self._submit(close_side, qty=abs(pos.size), reason=f"close:{reason}")
        pos.size = 0
        pos.side = ""
        pos.entry_price = 0.0
        pos.entry_ts    = 0.0

    async def _position_timeout_loop(self) -> None:
        """Force-close position after timeout_s of no new signal."""
        try:
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                pos = self.position
                if pos.size != 0 and pos.entry_ts > 0:
                    age = time.time() - pos.entry_ts
                    if age >= self.position_timeout_s:
                        log.info("Position timeout (%.1fs) — closing.", age)
                        await self._close_position("timeout")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("position_timeout_loop crashed: %s", e)

    # ---------------------------------------------------------- order handler
    async def _on_order(self, ev: OrderEvent) -> None:
        """Called on every Rithmic/exchange order notification."""
        log.info("ORDER.in type=%d status=%s side=%s qty=%d price=%s fill=%s@%s text=%s",
                 ev.notify_type, ev.status, ev.side, ev.quantity,
                 ev.price, ev.fill_size, ev.fill_price, ev.text)
        # (We don't yet reconcile position state against fills — max-1 sizing
        # + MARKET orders keeps the risk bounded for v1.  See README_AUTONOMOUS.)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live Rithmic → LGBM → Paper-order engine")
    ap.add_argument("--model",       required=True, help="Path to LGBM .pkl (joblib)")
    ap.add_argument("--calibration", default=None,  help="Path to calibration JSON")
    ap.add_argument("--symbol",      required=True, help="e.g. ESZ5")
    ap.add_argument("--exchange",    required=True, help="e.g. CME")
    ap.add_argument("--threshold",   type=float, default=0.0,
                    help="Absolute prediction threshold")
    ap.add_argument("--min-tier",    default="top25",
                    choices=list(TIER_ORDER.keys()))
    ap.add_argument("--timeout",     type=float, default=30.0,
                    help="Position force-close timeout (seconds)")
    ap.add_argument("--paper", action="store_true",
                    help="Run against Rithmic paper system (no-op flag; Rithmic "
                         "paper/live is selected by RITHMIC_SYSTEM env var)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute signals but never submit orders")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    if args.paper and not os.environ.get("RITHMIC_SYSTEM"):
        log.warning("--paper passed but RITHMIC_SYSTEM is unset. "
                    "Set it to the AMP paper system name (e.g. 'Rithmic Paper Trading').")

    engine = SignalEngine(
        model_path=args.model,
        calibration_path=args.calibration,
        symbol=args.symbol,
        exchange=args.exchange,
        threshold=args.threshold,
        min_tier=args.min_tier,
        position_timeout_s=args.timeout,
        dry_run=args.dry_run,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_sig(*_a):
        log.info("Signal received, stopping...")
        engine.stop()

    for s in (_sig.SIGINT, _sig.SIGTERM):
        try:
            loop.add_signal_handler(s, _handle_sig)
        except NotImplementedError:
            # Windows fallback (we're on Linux so this shouldn't fire)
            _sig.signal(s, lambda *_: engine.stop())

    try:
        loop.run_until_complete(engine.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
