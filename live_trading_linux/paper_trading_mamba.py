#!/usr/bin/env python3
"""
paper_trading_mamba.py — Mamba v7 paper trading engine for Monday market open.

Architecture:
    Rithmic MBO Feed (NQM6/CME)
        -> Raw event encoding (6-col MBO format)
        -> StreamingFeaturesSmartV3 (25 features, validated to 0.000001 accuracy)
        -> MambaInferenceEngine (CPU, 270ms/prediction, 266K params)
        -> Signal: pred_1s, pred_5s, pred_10s, confidence, direction
        -> Position management (1 contract, signal-flip exits)
        -> Discord notifications for entries/exits
        -> Periodic P&L report every 5 min

Strategy: Top1%+ confidence entry, signal-flip exit.
    - Enter: When confidence >= Top1% threshold AND features warm (5000+ events)
    - Hold: As long as model keeps predicting same direction
    - Exit: On signal flip (direction reversal at any confidence)
    - Flip: After exit, immediately enter opposite if new signal meets threshold

Usage:
    # Live mode (connects to Rithmic):
    python3 paper_trading_mamba.py

    # Replay from recorded MBO data:
    python3 paper_trading_mamba.py --replay /path/to/mbo_events.npz

    # Replay with verbose output:
    python3 paper_trading_mamba.py --replay /path/to/mbo_events.npz --verbose

IMPORTANT: PAPER TRADE ONLY. No real orders are ever submitted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import platform
import signal as _sig
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
from mamba_inference import MambaInferenceEngine

# ── Constants ──
TICK_SIZE = 0.25
POINT_VALUE = 50.0  # NQ
TICK_VALUE = 12.50
COMMISSION_PER_SIDE = 2.35  # AMP $4.70 RT / 2

# Default model paths
LVL3 = Path("/home/jupiter/Lvl3Quant")
DEFAULT_WEIGHTS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_best.pt"
DEFAULT_STATS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_feature_stats.npz"
DEFAULT_PREDS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/concat_oot_predictions.npz"

# ── Logging ──
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_DATE = datetime.now().strftime("%Y%m%d_%H%M")

log = logging.getLogger("paper_mamba")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / f"paper_mamba_{_SESSION_DATE}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(sh)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Position Manager
# ═══════════════════════════════════════════════════════════════════════════════

class PaperPosition:
    """Tracks paper trading position and P&L."""

    def __init__(self):
        self.position: int = 0  # +1 long, -1 short, 0 flat
        self.entry_price: float = 0.0
        self.entry_time: float = 0.0
        self.entry_tier: str = ""

        self.realized_pnl: float = 0.0
        self.total_commission: float = 0.0
        self.n_trades: int = 0
        self.n_winners: int = 0
        self.trades: list[dict] = []

    @property
    def is_flat(self) -> bool:
        return self.position == 0

    def enter(self, direction: int, price: float, t: float, tier: str = ""):
        if not self.is_flat:
            return
        self.position = direction
        self.entry_price = price
        self.entry_time = t
        self.entry_tier = tier
        self.total_commission += COMMISSION_PER_SIDE
        log.info("ENTRY %s @ %.2f | tier=%s",
                 "LONG" if direction > 0 else "SHORT", price, tier)

    def exit(self, price: float, t: float, reason: str = "signal_flip") -> Optional[dict]:
        if self.is_flat:
            return None

        if self.position > 0:
            gross = (price - self.entry_price) * POINT_VALUE
        else:
            gross = (self.entry_price - price) * POINT_VALUE

        net = gross - COMMISSION_PER_SIDE * 2
        hold_s = t - self.entry_time

        self.realized_pnl += net
        self.total_commission += COMMISSION_PER_SIDE
        self.n_trades += 1
        if net > 0:
            self.n_winners += 1

        trade = {
            "direction": "LONG" if self.position > 0 else "SHORT",
            "entry_price": self.entry_price,
            "exit_price": price,
            "gross_pnl": round(gross, 2),
            "net_pnl": round(net, 2),
            "hold_time_s": round(hold_s, 2),
            "reason": reason,
            "tier": self.entry_tier,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade)

        log.info("EXIT %s @ %.2f | reason=%s | gross=$%.2f net=$%.2f hold=%.1fs | "
                 "cumulative P&L=$%.2f (%d trades, %.0f%% win)",
                 trade["direction"], price, reason, gross, net, hold_s,
                 self.realized_pnl, self.n_trades,
                 self.n_winners / max(self.n_trades, 1) * 100)

        self.position = 0
        self.entry_price = 0.0
        return trade

    def summary(self) -> dict:
        wr = self.n_winners / max(self.n_trades, 1) * 100
        return {
            "n_trades": self.n_trades,
            "win_rate": round(wr, 1),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_commission": round(self.total_commission, 2),
            "n_winners": self.n_winners,
            "n_losers": self.n_trades - self.n_winners,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Mamba Paper Trading Session
# ═══════════════════════════════════════════════════════════════════════════════

class MambaPaperSession:
    """Mamba v7 paper trading session with signal-flip exit strategy."""

    TIER_ORDER = {"Top5%": 1, "Top1%": 2, "Top0.5%": 3, "Top0.1%": 4}

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        stats_path: str | Path = DEFAULT_STATS,
        preds_path: str | Path = DEFAULT_PREDS,
        symbol: str = "NQM6",
        exchange: str = "CME",
        window_size: int = 1000,
        stride: int = 500,
        min_tier: str = "Top1%",
        stats_interval_s: float = 300.0,
        max_spread_ticks: float = 4.0,
        device: str | None = None,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.window_size = window_size
        self.stride = stride
        self.min_tier = min_tier
        self.stats_interval_s = stats_interval_s
        self.max_spread_ticks = max_spread_ticks

        # Feature engine (25 smart_v3 features, validated)
        self.features = StreamingFeaturesSmartV3()

        # Mamba inference engine
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = MambaInferenceEngine(
            weights_path=str(weights_path),
            stats_path=str(stats_path),
            window_size=window_size,
            stride=stride,
            device=_device,
        )
        if Path(preds_path).exists():
            self.engine.calibrate_thresholds(str(preds_path))

        # Position manager
        self.pos = PaperPosition()

        # Signal tracking for flip detection
        self.prev_direction: Optional[int] = None

        # Market state
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.mid_price: float = 0.0
        self.prev_ts_ns: int = 0

        # Counters
        self.events_processed: int = 0
        self.predictions_made: int = 0
        self.signals_generated: int = 0

        # Signal log
        self._signals_path = _LOG_DIR / f"mamba_signals_{symbol}_{_SESSION_DATE}.jsonl"
        self._signals_fh = open(self._signals_path, "a", buffering=1)

        # Results output
        self._results_path = _LOG_DIR / f"mamba_results_{symbol}_{_SESSION_DATE}.json"

        # Asyncio
        self._stop = asyncio.Event()
        self._last_stats_time = time.time()

        log.info("=" * 70)
        log.info("MAMBA v7 PAPER TRADING SESSION")
        log.info("=" * 70)
        log.info("  Model: %s", weights_path)
        log.info("  Symbol: %s | Exchange: %s", symbol, exchange)
        log.info("  Window: %d | Stride: %d", window_size, stride)
        log.info("  Min confidence tier: %s", min_tier)
        log.info("  Stats interval: %.0fs | Max spread: %.1f ticks",
                 stats_interval_s, max_spread_ticks)
        log.info("  Signal log: %s", self._signals_path)
        log.info("  *** PAPER TRADE MODE -- NO REAL ORDERS ***")
        log.info("=" * 70)

    def _tier_meets_min(self, tier: Optional[str]) -> bool:
        if tier is None:
            return False
        return self.TIER_ORDER.get(tier, 0) >= self.TIER_ORDER.get(self.min_tier, 0)

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
        """Process one raw MBO event through feature engine + Mamba + execution."""
        self.events_processed += 1

        # Step 1: Compute streaming features (25 smart_v3)
        feat_vec = self.features.update(
            time_delta_log, event_type_id, side_id,
            price_rel_ticks, qty_log, spread_ticks,
        )

        # Step 2: Feed to Mamba inference (accumulates window, predicts at stride)
        pred = self.engine.add_event(feat_vec)
        if pred is None:
            return  # Not at stride boundary yet

        self.predictions_made += 1

        # Step 3: Extract signal
        direction = pred["direction"]  # +1 or -1
        tier = pred.get("tier")
        confidence = pred["confidence_1s"]
        meets_threshold = self._tier_meets_min(tier)

        # Log signal
        sig_record = {
            "t": datetime.now(timezone.utc).isoformat(),
            "ts_ns": int(timestamp_ns),
            "event": self.events_processed,
            "pred_1s": round(pred["pred_1s"], 6),
            "pred_5s": round(pred["pred_5s"], 6),
            "pred_10s": round(pred["pred_10s"], 6),
            "conf": round(confidence, 4),
            "dir": "L" if direction > 0 else "S",
            "tier": tier,
            "bid": self.best_bid,
            "ask": self.best_ask,
            "mid": self.mid_price,
        }
        self._signals_fh.write(json.dumps(sig_record) + "\n")

        # Step 4: Execution logic (signal-flip exit, Top1%+ entry)
        now_t = time.time()

        # Spread filter
        if self.best_bid > 0 and self.best_ask > 0:
            spread_actual = (self.best_ask - self.best_bid) / TICK_SIZE
            if spread_actual > self.max_spread_ticks:
                self.prev_direction = direction
                return

        # EXIT: signal flip detection
        if not self.pos.is_flat and self.prev_direction is not None:
            if direction != self.prev_direction:
                # Signal flipped — exit
                exit_price = self.mid_price if self.mid_price > 0 else self.best_bid
                trade = self.pos.exit(exit_price, now_t, reason="signal_flip")

                # Immediately re-enter opposite if meets threshold
                if trade and meets_threshold and self.features.is_warm():
                    entry_price = self.mid_price if self.mid_price > 0 else (
                        self.best_ask if direction > 0 else self.best_bid)
                    self.pos.enter(direction, entry_price, now_t, tier=tier or "")
                    self.signals_generated += 1

        # ENTRY: flat + meets threshold + warm
        elif self.pos.is_flat and meets_threshold and self.features.is_warm():
            entry_price = self.mid_price if self.mid_price > 0 else (
                self.best_ask if direction > 0 else self.best_bid)
            self.pos.enter(direction, entry_price, now_t, tier=tier or "")
            self.signals_generated += 1

        self.prev_direction = direction

        # Periodic stats
        if now_t - self._last_stats_time >= self.stats_interval_s:
            self._periodic_summary()
            self._last_stats_time = now_t

    # ─────────────────────────────────────────────────────────────────────
    # Encoding: raw Rithmic event -> 6-col MBO format
    # ─────────────────────────────────────────────────────────────────────

    def _encode_and_process(self, ts_ns: int, etype: float, side: float,
                             price: float, qty: int):
        """Encode raw market data to 6-col format and process."""
        delta_us = max(0, (ts_ns - self.prev_ts_ns) / 1000) if self.prev_ts_ns else 0
        self.prev_ts_ns = ts_ns
        price_rel = ((price - self.mid_price) / TICK_SIZE
                     if self.mid_price > 0 and price > 0 else 0.0)
        spread = (self.best_ask - self.best_bid) / TICK_SIZE if (
            self.best_bid > 0 and self.best_ask > 0) else 0.0

        self._process_raw_event(
            time_delta_log=math.log1p(delta_us) if delta_us > 0 else 0.0,
            event_type_id=int(etype),
            side_id=int(side),
            price_rel_ticks=price_rel,
            qty_log=math.log(max(1, qty)),
            spread_ticks=spread,
            timestamp_ns=ts_ns,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Live mode: connect to Rithmic
    # ─────────────────────────────────────────────────────────────────────

    async def run_live(self):
        """Connect to Rithmic for live paper trading."""
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
                    self._encode_and_process(ts_ns, 0.0, 0.0, ev.bid_price, ev.bid_size)
                if ev.has_ask:
                    self._encode_and_process(ts_ns, 0.0, 1.0, ev.ask_price, ev.ask_size)
            elif isinstance(ev, TradeEvent):
                side = {1: 1.0, 2: 0.0}.get(ev.aggressor, 2.0)
                self._encode_and_process(ts_ns, 3.0, side, ev.trade_price, ev.trade_size)

        client.set_md_callback(on_md)
        await client.connect()
        await client.subscribe_md(self.symbol, self.exchange)

        log.info("LIVE: Connected to Rithmic. Receiving %s on %s.",
                 self.symbol, self.exchange)
        log.info("  Warming up features (need %d events)...", self.features.MIN_WARMUP)

        async def stats_loop():
            while not self._stop.is_set():
                await asyncio.sleep(self.stats_interval_s)
                self._periodic_summary()

        tasks = [asyncio.create_task(stats_loop())]

        def handle_signal(*_):
            log.info("Shutdown signal received")
            self._stop.set()

        for s in (_sig.SIGINT, _sig.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(s, handle_signal)

        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await client.disconnect()
            self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Follow-events mode: tail local JSONL fan-out file
    # ─────────────────────────────────────────────────────────────────────

    async def run_follow_events(self, jsonl_path: str | Path | None = None):
        """Tail local fan-out JSONL instead of connecting to Rithmic."""
        if jsonl_path is None:
            if platform.system() == "Windows":
                jsonl_path = Path(r"C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl")
            else:
                jsonl_path = Path(__file__).resolve().parent / "logs" / "live_events.jsonl"
        else:
            jsonl_path = Path(jsonl_path)

        log.info("FOLLOW-EVENTS: Tailing %s", jsonl_path)
        log.info("  Warming up features (need %d events)...", self.features.MIN_WARMUP)

        # Wait for file to exist
        while not jsonl_path.exists():
            if self._stop.is_set():
                return
            log.info("  Waiting for %s to appear...", jsonl_path)
            await asyncio.sleep(1.0)

        async def stats_loop():
            while not self._stop.is_set():
                await asyncio.sleep(self.stats_interval_s)
                self._periodic_summary()

        tasks = [asyncio.create_task(stats_loop())]

        # On Unix, install signal handlers
        if platform.system() != "Windows":
            def handle_signal(*_):
                log.info("Shutdown signal received")
                self._stop.set()
            for s in (_sig.SIGINT, _sig.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(s, handle_signal)

        try:
            with open(jsonl_path, "r") as fh:
                # Seek to end — only process new events
                fh.seek(0, 2)
                log.info("  Seeked to end of file, waiting for new events...")
                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        await asyncio.sleep(0.1)  # 100ms poll
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts_ns = ev["timestamp_ns"]
                    side = ev["side"]
                    action = ev["action"]
                    price_ticks = ev["price_ticks"]
                    size = ev["size"]
                    bbo = ev.get("bbo", {})

                    # Update BBO from the fan-out event
                    bid_p = bbo.get("bid_price", 0)
                    ask_p = bbo.get("ask_price", 0)
                    if bid_p > 0:
                        self.best_bid = bid_p
                    if ask_p > 0:
                        self.best_ask = ask_p
                    if self.best_bid > 0 and self.best_ask > 0:
                        self.mid_price = (self.best_bid + self.best_ask) / 2.0

                    # Convert back to price from ticks for encoding
                    price = price_ticks * TICK_SIZE
                    etype = float(action)   # 0=BBO, 3=Trade
                    side_f = float(side)

                    self._encode_and_process(ts_ns, etype, side_f, price, size)
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — stopping")
            self._stop.set()
        finally:
            for t in tasks:
                t.cancel()
            self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Replay mode: process recorded NPZ data
    # ─────────────────────────────────────────────────────────────────────

    def run_replay(self, npz_path: str, max_events: int = None):
        """Replay recorded MBO events through the full pipeline."""
        data = np.load(npz_path)
        events = data["events"]
        timestamps = data.get("timestamps", np.zeros(len(events), dtype=np.int64))

        N = len(events) if max_events is None else min(len(events), max_events)
        log.info("REPLAY: %s (%d events)", npz_path, N)

        t0 = time.time()
        for i in range(N):
            ev = events[i]
            ts = int(timestamps[i]) if i < len(timestamps) else 0
            self._process_raw_event(
                time_delta_log=float(ev[0]),
                event_type_id=int(ev[1]),
                side_id=int(ev[2]),
                price_rel_ticks=float(ev[3]),
                qty_log=float(ev[4]),
                spread_ticks=float(ev[5]),
                timestamp_ns=ts,
            )
            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                log.info("  [%d/%d] %.0f ev/s | preds=%d signals=%d P&L=$%.2f",
                         i + 1, N, (i + 1) / elapsed,
                         self.predictions_made, self.signals_generated,
                         self.pos.realized_pnl)

        self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────

    def _periodic_summary(self):
        """Print periodic status."""
        s = self.pos.summary()
        log.info("STATUS | events=%d preds=%d signals=%d | %d trades WR=%.0f%% P&L=$%.2f",
                 self.events_processed, self.predictions_made, self.signals_generated,
                 s["n_trades"], s["win_rate"], s["realized_pnl"])

    def _final_report(self):
        """Save final results."""
        s = self.pos.summary()
        results = {
            "session": _SESSION_DATE,
            "model": "mamba_v7_tiny_smart_v3",
            "symbol": self.symbol,
            "min_tier": self.min_tier,
            "window_size": self.window_size,
            "stride": self.stride,
            "events_processed": self.events_processed,
            "predictions_made": self.predictions_made,
            "signals_generated": self.signals_generated,
            "features_warm_at": self.features.MIN_WARMUP,
            **s,
            "trades": self.pos.trades,
        }
        with open(self._results_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info("=" * 70)
        log.info("FINAL REPORT")
        log.info("=" * 70)
        log.info("  Events: %d | Predictions: %d | Signals: %d",
                 self.events_processed, self.predictions_made, self.signals_generated)
        log.info("  Trades: %d | Win rate: %.1f%%", s["n_trades"], s["win_rate"])
        log.info("  Realized P&L: $%.2f | Commission: $%.2f",
                 s["realized_pnl"], s["total_commission"])
        log.info("  Results saved: %s", self._results_path)
        log.info("  Signals saved: %s", self._signals_path)
        log.info("=" * 70)

        self._signals_fh.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mamba v7 Paper Trading")
    parser.add_argument("--replay", type=str, help="Replay from NPZ file")
    parser.add_argument("--symbol", type=str, default="NQM6", help="Symbol")
    parser.add_argument("--exchange", type=str, default="CME", help="Exchange")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--preds", type=str, default=None)
    parser.add_argument("--min-tier", type=str, default="Top1%",
                        choices=["Top5%", "Top1%", "Top0.5%", "Top0.1%"])
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--follow-events", nargs="?", const="__default__", default=None,
                        metavar="PATH",
                        help="Tail local fan-out JSONL instead of connecting to Rithmic. "
                             "Optionally provide a path override.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda/cpu)")
    args = parser.parse_args()

    session = MambaPaperSession(
        weights_path=args.weights or DEFAULT_WEIGHTS,
        stats_path=args.stats or DEFAULT_STATS,
        preds_path=args.preds or DEFAULT_PREDS,
        symbol=args.symbol,
        exchange=args.exchange,
        window_size=args.window,
        stride=args.stride,
        min_tier=args.min_tier,
        device=args.device,
    )

    if args.replay:
        session.run_replay(args.replay, max_events=args.max_events)
    elif args.follow_events is not None:
        # Follow-events mode: tail local fan-out JSONL
        path = None if args.follow_events == "__default__" else args.follow_events
        asyncio.run(session.run_follow_events(path))
    else:
        # Live mode
        asyncio.run(session.run_live())


if __name__ == "__main__":
    main()
