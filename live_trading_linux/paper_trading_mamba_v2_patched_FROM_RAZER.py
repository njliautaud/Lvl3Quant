#!/usr/bin/env python3
"""
paper_trading_mamba_v2.py — CNN Mamba v2 paper trading engine for Razer (Windows).

v2.1 (HC #46): Full risk management integration.
  * Rule 1: Max hold time (15 min cap)
  * Rule 2: Signal decay exit (confidence drops below z=1.0)
  * Rule 3: Trailing stop (MFE > 4 ticks -> lock entry+1)
  * Rule 4: Time-of-day gate (no entries 9:25-9:40 AM ET)
  * Rule 5: Max daily loss (-$3,000 circuit breaker)
  * Rule 6: Consecutive loss limit (3 losers -> 10 min pause)
  * Rule 7: PatchTST reversal exit (secondary model confirmation)

Adapted from paper_trading_mamba.py (Jupiter/Linux/Mamba v7) with:
  * Windows paths (C:\\Users\\claude\\Lvl3Quant)
  * CNN Mamba v2 inference engine (CNNMambaV2Inference) instead of MambaInferenceEngine
  * CUDA device by default (RTX 3070 on Razer)
  * v2 weights at output\\cnn_mamba_v2_smart_v3_mar\\fold_10_best.pt
  * SKIP_NORMALIZE=1 + MAMBA_FEATURE_SET=smart_v3 set at module load

IMPORTANT: PAPER TRADE ONLY. No real orders are ever submitted.

Usage:
    # Live mode (connects to Rithmic):
    python paper_trading_mamba_v2.py

    # With PatchTST secondary model:
    python paper_trading_mamba_v2.py --patchtst-weights output\\patchtst_smartv3_25feat_2310\\fold_10_best.pt

    # Replay from recorded MBO data:
    python paper_trading_mamba_v2.py --replay C:\\path\\to\\mbo_events.npz
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

# Tell train_cnn_mamba_v2 it's smart_v3 (sets N_TOTAL_FEATURES=25)
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
from cnn_mamba_v2_inference import CNNMambaV2Inference
from patchtst_inference import PatchTSTInference
from risk_manager import RiskManager, RiskConfig

# ── Constants ──
TICK_SIZE = 0.25
POINT_VALUE = 50.0  # NQ
TICK_VALUE = 12.50
COMMISSION_PER_SIDE = 2.35  # AMP $4.70 RT / 2

# Default model paths — Razer Windows
LVL3 = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))
DEFAULT_WEIGHTS = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
DEFAULT_STATS = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"
DEFAULT_PREDS = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "concat_oot_predictions.npz"

# ── Logging ──
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_DATE = datetime.now().strftime("%Y%m%d_%H%M")

log = logging.getLogger("paper_mamba_v2")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / f"paper_mamba_v2_{_SESSION_DATE}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(sh)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Position Manager (extended with risk fields)
# ═══════════════════════════════════════════════════════════════════════════════

class PaperPosition:
    """Tracks paper trading position and P&L."""

    def __init__(self):
        self.position: int = 0  # +1 long, -1 short, 0 flat
        self.entry_price: float = 0.0
        self.entry_time: float = 0.0
        self.entry_tier: str = ""
        self.entry_confidence: float = 0.0  # NEW: for signal decay tracking

        self.realized_pnl: float = 0.0
        self.total_commission: float = 0.0
        self.n_trades: int = 0
        self.n_winners: int = 0
        self.trades: list[dict] = []

    @property
    def is_flat(self) -> bool:
        return self.position == 0

    def enter(self, direction: int, price: float, t: float, tier: str = "",
              confidence: float = 0.0):
        if not self.is_flat:
            return
        self.position = direction
        self.entry_price = price
        self.entry_time = t
        self.entry_tier = tier
        self.entry_confidence = confidence
        self.total_commission += COMMISSION_PER_SIDE
        log.info("ENTRY %s @ %.2f | tier=%s | confidence=%.4f",
                 "LONG" if direction > 0 else "SHORT", price, tier, confidence)

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
            "entry_confidence": round(self.entry_confidence, 6),
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
# CNN Mamba v2 Paper Trading Session (with Risk Management)
# ═══════════════════════════════════════════════════════════════════════════════

class MambaV2PaperSession:
    """CNN Mamba v2 paper trading session with full risk management (HC #46)."""

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
        device: str = "cuda",
        # Risk management config
        risk_config: RiskConfig = None,
        # PatchTST secondary model (Rule 7)
        patchtst_weights: str | Path = None,
        patchtst_stats: str | Path = None,
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

        # CNN Mamba v2 inference engine (RTX 3070 CUDA)
        self.engine = CNNMambaV2Inference(
            weights_path=str(weights_path),
            stats_path=str(stats_path),
            window_size=window_size,
            stride=stride,
            device=device,
        )
        if Path(preds_path).exists():
            self.engine.calibrate_thresholds(str(preds_path))
        else:
            log.warning("Predictions NPZ not found at %s — confidence tiers DISABLED. "
                        "All signals will fail _tier_meets_min until calibrated.", preds_path)

        # ── Risk Manager (HC #46) ──
        if risk_config is None:
            risk_config = RiskConfig()
        # If no PatchTST model, disable that rule
        if patchtst_weights is None:
            risk_config.patchtst_reversal_enabled = False
        self.risk = RiskManager(risk_config)

        # Calibrate signal decay threshold from predictions
        if Path(preds_path).exists():
            self.risk.calibrate_signal_decay_threshold(str(preds_path))

        # ── PatchTST Secondary Model (Rule 7) ──
        self.patchtst_engine: Optional[PatchTSTInference] = None
        if patchtst_weights is not None:
            ptst_stats = patchtst_stats or stats_path  # fallback to same stats
            try:
                self.patchtst_engine = PatchTSTInference(
                    weights_path=str(patchtst_weights),
                    stats_path=str(ptst_stats),
                    window_size=window_size,
                    stride=stride,
                    device=device,
                )
                log.info("PatchTST secondary model loaded (PatchTSTInference) from %s", patchtst_weights)
            except Exception as e:
                log.error("Failed to load PatchTST model: %s — Rule 7 DISABLED", e)
                risk_config.patchtst_reversal_enabled = False

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
        self.risk_exits: int = 0  # NEW: count of risk-triggered exits
        self.risk_blocked_entries: int = 0  # NEW: count of blocked entries

        # Signal log
        self._signals_path = _LOG_DIR / f"mamba_v2_signals_{symbol}_{_SESSION_DATE}.jsonl"
        self._signals_fh = open(self._signals_path, "a", buffering=1)

        # Results output
        self._results_path = _LOG_DIR / f"mamba_v2_results_{symbol}_{_SESSION_DATE}.json"

        # Asyncio
        self._stop = asyncio.Event()
        self._last_stats_time = time.time()

        log.info("=" * 70)
        log.info("CNN MAMBA v2 PAPER TRADING SESSION (Razer) — HC #46 Risk Management")
        log.info("=" * 70)
        log.info("  Model: %s", weights_path)
        log.info("  Device: %s", device)
        log.info("  Symbol: %s | Exchange: %s", symbol, exchange)
        log.info("  Window: %d | Stride: %d", window_size, stride)
        log.info("  Min confidence tier: %s", min_tier)
        log.info("  Stats interval: %.0fs | Max spread: %.1f ticks",
                 stats_interval_s, max_spread_ticks)
        log.info("  PatchTST: %s", "LOADED" if self.patchtst_engine else "DISABLED")
        log.info("  Signal log: %s", self._signals_path)
        log.info("  *** PAPER TRADE MODE -- NO REAL ORDERS ***")
        log.info("=" * 70)

    def _tier_meets_min(self, tier: Optional[str]) -> bool:
        if tier is None:
            return False
        return self.TIER_ORDER.get(tier, 0) >= self.TIER_ORDER.get(self.min_tier, 0)

    # ─────────────────────────────────────────────────────────────────────
    # Risk-aware exit helper
    # ─────────────────────────────────────────────────────────────────────

    def _risk_exit(self, reason: str, now_t: float) -> Optional[dict]:
        """Execute a risk-triggered exit. Returns trade dict or None."""
        if self.pos.is_flat:
            return None
        exit_price = self.mid_price if self.mid_price > 0 else self.best_bid
        trade = self.pos.exit(exit_price, now_t, reason=reason)
        if trade:
            self.risk.on_exit(exit_price, trade["net_pnl"], reason=reason)
            self.risk_exits += 1
        return trade

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
        """Process one raw MBO event through feature engine + CNN Mamba v2 + execution."""
        self.events_processed += 1

        # ── Rule 3: Check trailing stop on every price update ──
        if not self.pos.is_flat and self.mid_price > 0:
            price_exit_reason = self.risk.check_exit_on_price(self.mid_price)
            if price_exit_reason:
                self._risk_exit(price_exit_reason, time.time())
                return  # exit done, skip further processing this event

        # Step 1: Compute streaming features (25 smart_v3)
        feat_vec = self.features.update(
            time_delta_log, event_type_id, side_id,
            price_rel_ticks, qty_log, spread_ticks,
        )

        # Step 2: Feed to CNN Mamba v2 inference (accumulates window, predicts at stride)
        pred = self.engine.add_event(np.asarray(feat_vec, dtype=np.float32))
        if pred is None:
            return  # Not at stride boundary yet

        self.predictions_made += 1

        # Step 3: Extract signal
        direction = pred["direction"]
        tier = pred.get("tier")
        confidence = pred["confidence_1s"]
        meets_threshold = self._tier_meets_min(tier)

        # Step 3b: Get PatchTST direction (Rule 7)
        patchtst_direction = None
        if self.patchtst_engine is not None:
            ptst_pred = self.patchtst_engine.add_event(
                np.asarray(feat_vec, dtype=np.float32))
            if ptst_pred is not None:
                patchtst_direction = ptst_pred["direction"]

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
            "patchtst_dir": ("L" if patchtst_direction > 0 else "S")
                            if patchtst_direction is not None else None,
            "risk_status": self.risk.status_line(),
        }
        self._signals_fh.write(json.dumps(sig_record) + "\n")

        # Step 4: Execution logic WITH RISK MANAGEMENT
        now_t = time.time()

        # Spread filter
        if self.best_bid > 0 and self.best_ask > 0:
            spread_actual = (self.best_ask - self.best_bid) / TICK_SIZE
            if spread_actual > self.max_spread_ticks:
                self.prev_direction = direction
                return

        # ── RISK CHECK: prediction-based exits (Rules 1, 2, 5, 7) ──
        if not self.pos.is_flat:
            risk_reason = self.risk.check_exit_on_prediction(
                current_confidence=confidence,
                current_direction=direction,
                patchtst_direction=patchtst_direction,
            )
            if risk_reason:
                self._risk_exit(risk_reason, now_t)
                # After risk exit, we might re-enter if signal is strong
                # But check entry gating first
                if meets_threshold and self.features.is_warm():
                    can_enter, block_reason = self.risk.can_enter()
                    if can_enter:
                        entry_price = self.mid_price if self.mid_price > 0 else (
                            self.best_ask if direction > 0 else self.best_bid)
                        self.pos.enter(direction, entry_price, now_t,
                                       tier=tier or "", confidence=confidence)
                        self.risk.on_entry(entry_price, direction, confidence)
                        self.signals_generated += 1
                    else:
                        self.risk_blocked_entries += 1
                        log.info("RISK: Entry blocked after risk exit: %s", block_reason)
                self.prev_direction = direction
                return

        # EXIT: signal flip detection (original logic, kept as-is)
        if not self.pos.is_flat and self.prev_direction is not None:
            if direction != self.prev_direction:
                exit_price = self.mid_price if self.mid_price > 0 else self.best_bid
                trade = self.pos.exit(exit_price, now_t, reason="signal_flip")
                if trade:
                    self.risk.on_exit(exit_price, trade["net_pnl"], reason="signal_flip")

                # Immediately re-enter opposite if meets threshold AND risk allows
                if trade and meets_threshold and self.features.is_warm():
                    can_enter, block_reason = self.risk.can_enter()
                    if can_enter:
                        entry_price = self.mid_price if self.mid_price > 0 else (
                            self.best_ask if direction > 0 else self.best_bid)
                        self.pos.enter(direction, entry_price, now_t,
                                       tier=tier or "", confidence=confidence)
                        self.risk.on_entry(entry_price, direction, confidence)
                        self.signals_generated += 1
                    else:
                        self.risk_blocked_entries += 1
                        log.info("RISK: Re-entry blocked after flip: %s", block_reason)

        # ENTRY: flat + meets threshold + warm + RISK ALLOWS
        elif self.pos.is_flat and meets_threshold and self.features.is_warm():
            can_enter, block_reason = self.risk.can_enter()
            if can_enter:
                entry_price = self.mid_price if self.mid_price > 0 else (
                    self.best_ask if direction > 0 else self.best_bid)
                self.pos.enter(direction, entry_price, now_t,
                               tier=tier or "", confidence=confidence)
                self.risk.on_entry(entry_price, direction, confidence)
                self.signals_generated += 1
            else:
                self.risk_blocked_entries += 1
                if self.risk_blocked_entries % 20 == 1:  # don't spam
                    log.info("RISK: New entry blocked: %s", block_reason)

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

        # NEW: Background loop to enforce max hold time even between predictions
        async def risk_timeout_loop():
            while not self._stop.is_set():
                await asyncio.sleep(1.0)  # check every second
                if not self.pos.is_flat and self.pos.entry_time > 0:
                    hold = time.time() - self.pos.entry_time
                    if hold >= self.risk.cfg.max_hold_seconds:
                        log.warning("RISK TIMEOUT LOOP: Max hold %.0fs exceeded", hold)
                        self._risk_exit("max_hold_time", time.time())

        # STALE DATA WATCHDOG: self-terminate if no events for 5 min during RTH
        _STALE_THRESHOLD_S = 300  # 5 minutes
        _last_event_count = [0]
        _last_event_check_time = [time.time()]
        _stale_consecutive = [0]

        async def stale_data_watchdog():
            while not self._stop.is_set():
                await asyncio.sleep(60)  # check every 60 seconds
                now = time.time()
                current_count = self.events_processed
                if current_count == _last_event_count[0]:
                    # No new events since last check
                    stale_s = now - _last_event_check_time[0]
                    _stale_consecutive[0] += 1
                    if stale_s >= _STALE_THRESHOLD_S:
                        # Check if during RTH (Sun 6pm - Fri 5pm ET roughly)
                        import zoneinfo
                        try:
                            et_now = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
                        except Exception:
                            et_now = datetime.utcnow()
                        weekday = et_now.weekday()  # 0=Mon
                        hour = et_now.hour
                        # Market hours broadly: Mon-Fri, or Sun after 6pm
                        is_market = (0 <= weekday <= 4) or (weekday == 6 and hour >= 18)
                        if is_market:
                            log.error("STALE DATA WATCHDOG: No new events for %ds during market hours. "
                                      "events_processed=%d. SELF-TERMINATING for watchdog restart.",
                                      int(stale_s), current_count)
                            # Close any open position at last known price
                            if not self.pos.is_flat:
                                self._risk_exit("stale_data_shutdown", now)
                            self._stop.set()
                            return
                        else:
                            if _stale_consecutive[0] % 10 == 1:
                                log.info("STALE DATA: No events for %ds but outside market hours "
                                         "(day=%d hr=%d). Waiting.", int(stale_s), weekday, hour)
                else:
                    # Events are flowing
                    _last_event_count[0] = current_count
                    _last_event_check_time[0] = now
                    _stale_consecutive[0] = 0

        tasks = [
            asyncio.create_task(stats_loop()),
            asyncio.create_task(risk_timeout_loop()),
            asyncio.create_task(stale_data_watchdog()),
        ]

        def handle_signal(*_):
            log.info("Shutdown signal received")
            self._stop.set()

        try:
            for s in (_sig.SIGINT, _sig.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(s, handle_signal)
        except (NotImplementedError, AttributeError):
            pass

        try:
            await self._stop.wait()
        except KeyboardInterrupt:
            self._stop.set()
        finally:
            for t in tasks:
                t.cancel()
            await client.disconnect()
            self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Follow-events mode: tail recorder's live_events.jsonl
    # ─────────────────────────────────────────────────────────────────────

    async def run_follow(self, events_path: str = None):
        """Tail the recorder's live_events.jsonl for near-real-time inference."""
        if events_path is None:
            if sys.platform == "win32":
                events_path = r"C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl"
            else:
                events_path = str(Path(__file__).resolve().parent / "logs" / "live_events.jsonl")

        log.info("FOLLOW: Tailing %s for live events", events_path)
        log.info("  Warming up features (need %d events)...", self.features.MIN_WARMUP)

        async def stats_loop():
            while not self._stop.is_set():
                await asyncio.sleep(self.stats_interval_s)
                self._periodic_summary()

        async def risk_timeout_loop():
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                if not self.pos.is_flat and self.pos.entry_time > 0:
                    hold = time.time() - self.pos.entry_time
                    if hold >= self.risk.cfg.max_hold_seconds:
                        log.warning("RISK TIMEOUT LOOP: Max hold %.0fs exceeded", hold)
                        self._risk_exit("max_hold_time", time.time())

        tasks = [
            asyncio.create_task(stats_loop()),
            asyncio.create_task(risk_timeout_loop()),
        ]

        def handle_signal(*_):
            log.info("Shutdown signal received")
            self._stop.set()

        try:
            for s in (_sig.SIGINT, _sig.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(s, handle_signal)
        except (NotImplementedError, AttributeError):
            pass

        while not self._stop.is_set():
            if os.path.exists(events_path):
                break
            log.info("  Waiting for events file to appear...")
            await asyncio.sleep(2.0)

        try:
            with open(events_path, "r") as fh:
                fh.seek(0, 2)
                log.info("FOLLOW: Positioned at end of file, waiting for new events...")

                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        await asyncio.sleep(0.1)
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts_ns = ev.get("timestamp_ns", 0)

                    bbo = ev.get("bbo", {})
                    if bbo:
                        bid = bbo.get("bid_price", 0)
                        ask = bbo.get("ask_price", 0)
                        if bid > 0:
                            self.best_bid = bid
                        if ask > 0:
                            self.best_ask = ask
                        if self.best_bid > 0 and self.best_ask > 0:
                            self.mid_price = (self.best_bid + self.best_ask) / 2.0

                    action = ev.get("action", 0)
                    side = ev.get("side", 0)
                    price_ticks = ev.get("price_ticks", 0)
                    size = ev.get("size", 1)

                    price = self.mid_price + price_ticks * TICK_SIZE if self.mid_price > 0 else 0

                    etype = 3.0 if action >= 2 else 0.0
                    side_f = float(side)

                    self._encode_and_process(ts_ns, etype, side_f, price, size)

        except KeyboardInterrupt:
            self._stop.set()
        finally:
            for t in tasks:
                t.cancel()
            self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Replay mode
    # ─────────────────────────────────────────────────────────────────────

    def run_replay(self, npz_path: str, max_events: int = None):
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
                log.info("  [%d/%d] %.0f ev/s | preds=%d signals=%d P&L=$%.2f | %s",
                         i + 1, N, (i + 1) / elapsed,
                         self.predictions_made, self.signals_generated,
                         self.pos.realized_pnl, self.risk.status_line())

        self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────

    def _periodic_summary(self):
        s = self.pos.summary()
        log.info("STATUS | events=%d preds=%d signals=%d | %d trades WR=%.0f%% P&L=$%.2f "
                 "| risk_exits=%d blocked=%d | %s",
                 self.events_processed, self.predictions_made, self.signals_generated,
                 s["n_trades"], s["win_rate"], s["realized_pnl"],
                 self.risk_exits, self.risk_blocked_entries,
                 self.risk.status_line())

    def _final_report(self):
        s = self.pos.summary()
        risk_summary = self.risk.summary()
        results = {
            "session": _SESSION_DATE,
            "model": "cnn_mamba_v2_smart_v3",
            "symbol": self.symbol,
            "min_tier": self.min_tier,
            "window_size": self.window_size,
            "stride": self.stride,
            "events_processed": self.events_processed,
            "predictions_made": self.predictions_made,
            "signals_generated": self.signals_generated,
            "features_warm_at": self.features.MIN_WARMUP,
            "risk_exits": self.risk_exits,
            "risk_blocked_entries": self.risk_blocked_entries,
            "risk_summary": risk_summary,
            **s,
            "trades": self.pos.trades,
        }
        with open(self._results_path, "w") as f:
            json.dump(results, f, indent=2)

        log.info("=" * 70)
        log.info("FINAL REPORT (with HC #46 Risk Management)")
        log.info("=" * 70)
        log.info("  Events: %d | Predictions: %d | Signals: %d",
                 self.events_processed, self.predictions_made, self.signals_generated)
        log.info("  Trades: %d | Win rate: %.1f%%", s["n_trades"], s["win_rate"])
        log.info("  Realized P&L: $%.2f | Commission: $%.2f",
                 s["realized_pnl"], s["total_commission"])
        log.info("  Risk exits: %d | Blocked entries: %d",
                 self.risk_exits, self.risk_blocked_entries)
        log.info("  Exit reasons: %s", risk_summary.get("exits_by_reason", {}))
        log.info("  Results saved: %s", self._results_path)
        log.info("  Signals saved: %s", self._signals_path)
        log.info("=" * 70)

        self._signals_fh.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Single-instance lock (prevents duplicate paper traders) ──
    import msvcrt
    _lock_dir = Path(__file__).resolve().parent / "status"
    _lock_dir.mkdir(parents=True, exist_ok=True)
    LOCK_FILE = _lock_dir / "paper_trader.lock"
    try:
        _lock_fh = open(LOCK_FILE, "w")
        msvcrt.locking(_lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        _lock_fh.write(str(os.getpid()))
        _lock_fh.flush()
    except (IOError, OSError):
        print(f"FATAL: Another paper trader instance is already running (lock: {LOCK_FILE}). Exiting.")
        sys.exit(1)
    # Keep _lock_fh open for process lifetime (lock released on close/exit)

    parser = argparse.ArgumentParser(description="CNN Mamba v2 Paper Trading (Razer) — HC #46 Risk Mgmt")
    parser.add_argument("--replay", type=str, help="Replay from NPZ file")
    parser.add_argument("--symbol", type=str, default="NQM6", help="Symbol")
    parser.add_argument("--exchange", type=str, default="CME", help="Exchange")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--preds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--min-tier", type=str, default="Top1%",
                        choices=["Top5%", "Top1%", "Top0.5%", "Top0.1%"])
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--follow-events", nargs="?", const=True, default=None,
                        help="Tail recorder's live_events.jsonl instead of Rithmic. "
                             "Optionally pass a path to the JSONL file.")
    # Risk management CLI args
    parser.add_argument("--max-hold-minutes", type=float, default=15.0,
                        help="Max position hold time in minutes (default: 15)")
    parser.add_argument("--max-daily-loss", type=float, default=-3000.0,
                        help="Daily loss circuit breaker in dollars (default: -3000)")
    parser.add_argument("--trailing-mfe-ticks", type=float, default=4.0,
                        help="MFE trigger for trailing stop in ticks (default: 4)")
    parser.add_argument("--trailing-lock-ticks", type=float, default=1.0,
                        help="Trailing stop lock-in ticks above entry (default: 1)")
    parser.add_argument("--consec-loss-limit", type=int, default=3,
                        help="Consecutive losses before pause (default: 3)")
    parser.add_argument("--consec-loss-pause-min", type=float, default=10.0,
                        help="Pause duration after consecutive losses in minutes (default: 10)")
    # PatchTST
    parser.add_argument("--patchtst-weights", type=str, default=None,
                        help="Path to PatchTST model weights for Rule 7")
    parser.add_argument("--patchtst-stats", type=str, default=None,
                        help="Path to PatchTST feature stats")

    args = parser.parse_args()

    # Build risk config from CLI args
    risk_config = RiskConfig(
        max_hold_seconds=args.max_hold_minutes * 60,
        trailing_stop_mfe_trigger_ticks=args.trailing_mfe_ticks,
        trailing_stop_lock_ticks=args.trailing_lock_ticks,
        max_daily_loss=args.max_daily_loss,
        max_consecutive_losses=args.consec_loss_limit,
        pause_after_losses_seconds=args.consec_loss_pause_min * 60,
    )

    session = MambaV2PaperSession(
        weights_path=args.weights or DEFAULT_WEIGHTS,
        stats_path=args.stats or DEFAULT_STATS,
        preds_path=args.preds or DEFAULT_PREDS,
        symbol=args.symbol,
        exchange=args.exchange,
        window_size=args.window,
        stride=args.stride,
        min_tier=args.min_tier,
        device=args.device,
        risk_config=risk_config,
        patchtst_weights=args.patchtst_weights,
        patchtst_stats=args.patchtst_stats,
    )

    if args.replay:
        session.run_replay(args.replay, max_events=args.max_events)
    elif args.follow_events is not None:
        path = args.follow_events if isinstance(args.follow_events, str) else None
        asyncio.run(session.run_follow(events_path=path))
    else:
        asyncio.run(session.run_live())


if __name__ == "__main__":
    main()
