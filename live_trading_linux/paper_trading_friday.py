#!/usr/bin/env python3
"""
paper_trading_friday.py — Multi-strategy paper trading session for Friday.

Connects to Rithmic MBO feed (ESM6 on CME), runs streaming feature engine +
LGBM smart_v2 inference, and feeds signals to ALL 6 execution strategies
simultaneously. Compares strategy performance in real-time.

Architecture:
    Rithmic MBO Feed (ESM6/CME)
        -> Raw event encoding (6-col MBO format)
        -> StreamingSmartV2Features (22 features)
        -> LGBM predict_proba every 500 events
        -> Signal: P(up), confidence, LONG/SHORT/NEUTRAL
        -> StrategyRunner distributes to 6 strategies
        -> Each strategy tracks its own trades/P&L/equity
        -> Periodic comparison table every 5 min
        -> Final results saved to JSON

Strategies tested:
    1. Baseline Market Order (Top10% threshold)
    2. High Confidence Only (Top1% threshold)
    3. MFE-Informed TP/SL (Top5%+ with adaptive targets)
    4. Passive Entry via Limit Orders (Top5%+, 1-tick cost)
    5. Confidence-Scaled Position (1/2/3x virtual sizing)
    6. Signal Persistence (2+ consecutive signals required)

Usage:
    # Live trading session (connects to Rithmic):
    python3 paper_trading_friday.py

    # Replay from existing MBO NPZ file:
    python3 paper_trading_friday.py --replay /path/to/mbo_events.npz

    # Replay from existing signal JSONL:
    python3 paper_trading_friday.py --replay-signals /path/to/signals.jsonl

IMPORTANT: PAPER TRADE ONLY. No real orders are ever submitted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal as _sig
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lgbm_live_inference import (
    StreamingSmartV2Features,
    LGBMSmartV2Model,
    TICK_SIZE,
    POINT_VALUE,
    DEFAULT_MODEL_PATH,
    DEFAULT_STRIDE,
    DEFAULT_WARMUP,
    _json_safe,
)
from execution_strategies import (
    StrategyRunner,
    create_default_strategies,
    replay_from_signals_jsonl,
    CONF_TOP10,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path("/home/jupiter/Lvl3Quant/live_trading_linux/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_SESSION_DATE = datetime.now().strftime("%Y%m%d")

log = logging.getLogger("paper_friday")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / f"paper_friday_{_SESSION_DATE}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(sh)


# ═══════════════════════════════════════════════════════════════════════════════
# Friday Paper Trading Session
# ═══════════════════════════════════════════════════════════════════════════════

class FridayPaperSession:
    """Multi-strategy paper trading session.

    Connects to Rithmic, computes streaming features, runs LGBM inference,
    and feeds signals to all 6 execution strategies simultaneously.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        stride: int = DEFAULT_STRIDE,
        warmup: int = DEFAULT_WARMUP,
        symbol: str = "ESM6",
        exchange: str = "CME",
        stats_interval_s: float = 300.0,
        max_spread_ticks: float = 3.0,
    ):
        self.stride = stride
        self.warmup = warmup
        self.symbol = symbol
        self.exchange = exchange
        self.stats_interval_s = stats_interval_s
        self.max_spread_ticks = max_spread_ticks

        # Load LGBM model
        self.model = LGBMSmartV2Model(model_path)

        # Streaming feature engine
        self.features = StreamingSmartV2Features()

        # Create all 6 strategies
        self.strategies = create_default_strategies()
        self.runner = StrategyRunner(self.strategies)

        # Market state
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.mid_price: float = 0.0
        self.prev_ts_ns: int = 0

        # Counters
        self.events_processed: int = 0
        self.predictions_made: int = 0
        self.signals_generated: int = 0

        # Signal log (for replay)
        self._signals_path = _LOG_DIR / f"friday_signals_{symbol}_{_SESSION_DATE}.jsonl"
        self._signals_fh = open(self._signals_path, "a", buffering=1)

        # Results output
        self._results_path = (
            _LOG_DIR / f"friday_strategy_comparison_{symbol}_{_SESSION_DATE}.json"
        )

        # Asyncio
        self._stop = asyncio.Event()

        log.info("=" * 70)
        log.info("FRIDAY PAPER TRADING SESSION")
        log.info("=" * 70)
        log.info("  Model: %s", model_path)
        log.info("  Symbol: %s | Exchange: %s", symbol, exchange)
        log.info("  Stride: %d | Warmup: %d", stride, warmup)
        log.info("  Stats interval: %.0fs | Max spread: %.1f ticks",
                 stats_interval_s, max_spread_ticks)
        log.info("  Strategies: %d", len(self.strategies))
        for s in self.strategies:
            log.info("    - %s", s.name)
        log.info("  Signal log: %s", self._signals_path)
        log.info("  Results: %s", self._results_path)
        log.info("  *** PAPER TRADE MODE — NO REAL ORDERS ***")
        log.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    # Core: process one MBO event
    # ─────────────────────────────────────────────────────────────────────

    def _process_raw_event(
        self,
        time_delta_log: float,
        event_type_id: int,
        side_id: int,
        price_rel_ticks: float,
        qty_log: float,
        spread_ticks: float,
        timestamp_ns: int = 0,
    ):
        """Process one raw MBO event through feature engine + LGBM + strategies."""

        self.events_processed += 1

        # Compute streaming features
        feat = self.features.update(
            time_delta_log, event_type_id, side_id,
            price_rel_ticks, qty_log, spread_ticks,
        )

        n = self.features.n_events
        now = time.time()

        # Send price updates to all strategies (for TP/SL, limit fills, timeouts)
        if self.best_bid > 0 and self.best_ask > 0:
            self.runner.on_price_update(now, self.best_bid, self.best_ask)

        # Skip during warmup
        if n < self.warmup:
            return

        # Only infer every STRIDE events
        if (n - self.warmup) % self.stride != 0:
            return

        # Run LGBM inference
        prob_up = self.model.predict_proba(feat)
        confidence = abs(prob_up - 0.5)
        self.predictions_made += 1

        # Spread filter (don't trade if spread is too wide)
        spread = self.best_ask - self.best_bid if (
            self.best_bid > 0 and self.best_ask > 0) else 0.0
        if spread > self.max_spread_ticks * TICK_SIZE:
            return

        # Determine signal direction
        if confidence >= CONF_TOP10:
            signal = "LONG" if prob_up > 0.5 else "SHORT"
            self.signals_generated += 1
        else:
            signal = "NEUTRAL"

        # Log signal
        sig_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_ns": int(timestamp_ns),
            "event_num": int(n),
            "prob_up": round(float(prob_up), 6),
            "confidence": round(float(confidence), 6),
            "signal": signal,
            "bid": float(self.best_bid),
            "ask": float(self.best_ask),
            "mid": float(self.mid_price),
            "spread_ticks": float(spread_ticks),
        }
        self._signals_fh.write(json.dumps(sig_record) + "\n")

        # Feed signal to ALL strategies (they each decide independently)
        self.runner.on_signal(now, prob_up, confidence,
                              self.best_bid, self.best_ask)

        # Periodic logging
        if self.predictions_made % 20 == 0:
            log.info("Pred #%d | event=%d | P(up)=%.4f conf=%.4f signal=%s | "
                     "bid=%.2f ask=%.2f",
                     self.predictions_made, n, prob_up, confidence, signal,
                     self.best_bid, self.best_ask)

    # ─────────────────────────────────────────────────────────────────────
    # Live mode: connect to Rithmic
    # ─────────────────────────────────────────────────────────────────────

    async def run_live(self):
        """Connect to Rithmic for real-time multi-strategy paper trading."""
        from rithmic_client import RithmicClient, BBOEvent, TradeEvent
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")

        client = RithmicClient()

        async def on_md(ev):
            ts_ns = int(ev.ssboe * 1_000_000_000 + ev.usecs * 1000)

            if isinstance(ev, BBOEvent):
                if ev.has_bid and ev.bid_price > 0:
                    self.best_bid = ev.bid_price
                if ev.has_ask and ev.ask_price > 0:
                    self.best_ask = ev.ask_price
                if self.best_bid > 0 and self.best_ask > 0:
                    self.mid_price = (self.best_bid + self.best_ask) / 2.0

                if ev.has_bid:
                    self._encode_and_process(
                        ts_ns, 0.0, 0.0, ev.bid_price, ev.bid_size)
                if ev.has_ask:
                    self._encode_and_process(
                        ts_ns, 0.0, 1.0, ev.ask_price, ev.ask_size)

            elif isinstance(ev, TradeEvent):
                side = {1: 1.0, 2: 0.0}.get(ev.aggressor, 2.0)
                self._encode_and_process(
                    ts_ns, 3.0, side, ev.trade_price, ev.trade_size)

        client.set_md_callback(on_md)
        await client.connect()
        await client.subscribe_md(self.symbol, self.exchange)

        log.info("LIVE MODE: Connected to Rithmic. Waiting for market data...")
        log.info("  Account: DATA ONLY — no orders will be sent")

        # Background tasks
        async def stats_loop():
            while not self._stop.is_set():
                await asyncio.sleep(self.stats_interval_s)
                self._periodic_summary()

        async def price_update_loop():
            """Send periodic price updates for timeout/TP/SL checks."""
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                if self.best_bid > 0 and self.best_ask > 0:
                    self.runner.on_price_update(
                        time.time(), self.best_bid, self.best_ask)

        tasks = [
            asyncio.create_task(stats_loop()),
            asyncio.create_task(price_update_loop()),
        ]

        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await client.disconnect()
            self._final_report()

    def _encode_and_process(self, ts_ns: int, etype: float, side: float,
                             price: float, qty: int):
        """Encode a Rithmic event into raw MBO format and process."""
        delta_us = max(0, (ts_ns - self.prev_ts_ns) / 1000) if self.prev_ts_ns else 0
        self.prev_ts_ns = ts_ns

        time_delta_log = math.log1p(delta_us) if delta_us > 0 else 0.0
        price_rel = ((price - self.mid_price) / TICK_SIZE
                     if self.mid_price > 0 and price > 0 else 0.0)
        qty_log = math.log(max(1, qty))
        spread_ticks = ((self.best_ask - self.best_bid) / TICK_SIZE
                        if self.best_bid > 0 and self.best_ask > 0 else 0.0)

        self._process_raw_event(
            time_delta_log=time_delta_log,
            event_type_id=int(etype),
            side_id=int(side),
            price_rel_ticks=price_rel,
            qty_log=qty_log,
            spread_ticks=spread_ticks,
            timestamp_ns=ts_ns,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Replay mode: read from NPZ file
    # ─────────────────────────────────────────────────────────────────────

    def run_replay(self, npz_path: str | Path):
        """Replay events from a raw MBO NPZ file through all strategies."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        events = data["events"].astype(np.float32)
        timestamps = data.get("timestamps", np.zeros(len(events), dtype=np.int64))

        log.info("REPLAY MODE: %s | %d events", npz_path.name, len(events))

        if events.shape[1] > 6:
            log.info("Preprocessed data (%d cols) — using first 6 raw columns",
                     events.shape[1])
            events = events[:, :6]

        # Set synthetic reference price for paper trading
        ref_price = 5000.0
        self.mid_price = ref_price
        self.best_bid = ref_price - TICK_SIZE
        self.best_ask = ref_price + TICK_SIZE

        t0 = time.time()
        n_total = len(events)
        report_interval = max(1, n_total // 20)  # report 20 times during replay

        for i in range(n_total):
            ev = events[i]
            ts = int(timestamps[i]) if i < len(timestamps) else 0

            # Update synthetic BBO based on price movement
            price_rel = ev[3]
            if ev[1] == 3.0:  # trade event
                self.mid_price += price_rel * TICK_SIZE * 0.01
                self.best_bid = self.mid_price - ev[5] * TICK_SIZE / 2
                self.best_ask = self.mid_price + ev[5] * TICK_SIZE / 2

            self._process_raw_event(
                time_delta_log=float(ev[0]),
                event_type_id=int(ev[1]),
                side_id=int(ev[2]),
                price_rel_ticks=float(ev[3]),
                qty_log=float(ev[4]),
                spread_ticks=float(ev[5]),
                timestamp_ns=ts,
            )

            # Progress report
            if (i + 1) % report_interval == 0:
                pct = (i + 1) / n_total * 100
                log.info("Replay progress: %d/%d (%.0f%%)", i + 1, n_total, pct)

        # Close all open positions at end
        self.runner.close_all(time.time(), self.best_bid, self.best_ask)

        elapsed = time.time() - t0
        log.info("Replay complete: %d events in %.1fs (%.0f events/sec)",
                 n_total, elapsed, n_total / elapsed if elapsed > 0 else 0)

        self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────

    def _periodic_summary(self):
        """Print periodic comparison of all strategies."""
        log.info("")
        log.info("PERIODIC SUMMARY — %d events | %d predictions | %d signals",
                 self.events_processed, self.predictions_made, self.signals_generated)
        self.runner.print_comparison()

    def _final_report(self):
        """Generate and save final report."""
        log.info("")
        log.info("=" * 70)
        log.info("FINAL REPORT — FRIDAY PAPER TRADING SESSION")
        log.info("=" * 70)
        log.info("  Events processed: %d", self.events_processed)
        log.info("  Predictions made: %d", self.predictions_made)
        log.info("  Signals generated: %d", self.signals_generated)
        log.info("  Signal log: %s", self._signals_path)
        log.info("")

        # Print comparison table
        table = self.runner.print_comparison()

        # Save full results
        self.runner.save_results(self._results_path)
        log.info("Full results: %s", self._results_path)

        # Also save a summary to a separate file
        summary_path = _LOG_DIR / f"friday_summary_{self.symbol}_{_SESSION_DATE}.json"
        summary = {
            "session": {
                "date": _SESSION_DATE,
                "symbol": self.symbol,
                "events_processed": self.events_processed,
                "predictions_made": self.predictions_made,
                "signals_generated": self.signals_generated,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "strategies": [
                s.stats.summary_dict() for s in self.strategies
            ],
        }
        with open(summary_path, "w") as f:
            json.dump(_json_safe(summary), f, indent=2)
        log.info("Summary: %s", summary_path)

        # Close signal log
        try:
            self._signals_fh.flush()
            self._signals_fh.close()
        except Exception:
            pass

    def shutdown(self):
        """Signal shutdown."""
        self._stop.set()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Friday Paper Trading Session — multi-strategy execution "
                    "comparison on LGBM smart_v2 signal stream. "
                    "PAPER TRADE ONLY — no real orders.")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                    help="Path to LGBM model .txt file")
    ap.add_argument("--symbol", default="ESM6",
                    help="Trading symbol (default: ESM6)")
    ap.add_argument("--exchange", default="CME",
                    help="Exchange (default: CME)")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                    help=f"Inference every N events (default: {DEFAULT_STRIDE})")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help=f"Min events before first inference (default: {DEFAULT_WARMUP})")
    ap.add_argument("--stats-interval", type=float, default=300.0,
                    help="Stats comparison interval in seconds (default: 300)")
    ap.add_argument("--max-spread", type=float, default=3.0,
                    help="Max spread in ticks to trade (default: 3.0)")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--replay", type=str, default=None,
                      help="Replay from raw MBO NPZ file")
    mode.add_argument("--replay-signals", type=str, default=None,
                      help="Replay from signal JSONL file (skips feature/LGBM)")

    return ap.parse_args()


def main():
    args = _parse_args()

    # Signal-only replay mode (no feature engine or LGBM needed)
    if args.replay_signals:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        output = _LOG_DIR / f"friday_strategy_comparison_signals_{_SESSION_DATE}.json"
        replay_from_signals_jsonl(args.replay_signals, output_path=output)
        return

    # Create session
    session = FridayPaperSession(
        model_path=args.model,
        stride=args.stride,
        warmup=args.warmup,
        symbol=args.symbol,
        exchange=args.exchange,
        stats_interval_s=args.stats_interval,
        max_spread_ticks=args.max_spread,
    )

    if args.replay:
        # Replay mode: synchronous, no Rithmic needed
        session.run_replay(args.replay)
    else:
        # Live mode: async Rithmic connection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _handle_sig(*_a):
            log.info("Shutdown signal received — closing all positions...")
            session.shutdown()

        for s in (_sig.SIGINT, _sig.SIGTERM):
            try:
                loop.add_signal_handler(s, _handle_sig)
            except NotImplementedError:
                _sig.signal(s, lambda *_: session.shutdown())

        try:
            loop.run_until_complete(session.run_live())
        finally:
            loop.close()


if __name__ == "__main__":
    main()
