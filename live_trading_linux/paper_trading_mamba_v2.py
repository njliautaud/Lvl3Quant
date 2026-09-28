#!/usr/bin/env python3
"""
paper_trading_mamba_v2.py — CNN Mamba v2 paper trading engine for Razer (Windows).

Adapted from paper_trading_mamba.py (Jupiter/Linux/Mamba v7) with:
  * Windows paths (C:\\Users\\claude\\Lvl3Quant)
  * CNN Mamba v2 inference engine (CNNMambaV2Inference) instead of MambaInferenceEngine
  * CUDA device by default (RTX 3070 on Razer)
  * v2 weights at output\\cnn_mamba_v2_smart_v3_mar\\fold_10_best.pt
  * SKIP_NORMALIZE=1 + MAMBA_FEATURE_SET=smart_v3 set at module load

The execution logic (signal-flip exits, Top1%+ entries, position management,
Discord notifications, replay/live modes) is unchanged from the v7 engine.

IMPORTANT: PAPER TRADE ONLY. No real orders are ever submitted.

Usage:
    # Live mode (connects to Rithmic):
    python paper_trading_mamba_v2.py

    # Replay from recorded MBO data:
    python paper_trading_mamba_v2.py --replay C:\\path\\to\\mbo_events.npz

    # Replay with verbose:
    python paper_trading_mamba_v2.py --replay <npz> --verbose
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

from trade_journal import TradeJournal

# Tell train_cnn_mamba_v2 it's smart_v3 (sets N_TOTAL_FEATURES=25)
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
from cnn_mamba_v2_inference import CNNMambaV2Inference

# ── Optional RL Execution Agent ──
try:
    _LIVE_TRADING_DIR = str(Path(__file__).resolve().parent.parent / "live_trading")
    if _LIVE_TRADING_DIR not in sys.path:
        sys.path.insert(0, _LIVE_TRADING_DIR)
    from rl_execution_agent import RLPaperTradingWrapper, RLAgentConfig, build_rl_pipeline
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# ── Constants ──
TICK_SIZE = 0.25
POINT_VALUE = 12.50  # ES — $12.50 per tick
TICK_VALUE = 12.50
COMMISSION_PER_SIDE = 2.35  # AMP $4.70 RT / 2

# Default model paths — Razer Windows
LVL3 = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))
DEFAULT_WEIGHTS = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
DEFAULT_STATS = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"
# OOT predictions for threshold calibration — point at v2's concat preds when available
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
# Paper Position Manager (identical to v7 engine)
# ═══════════════════════════════════════════════════════════════════════════════

class PaperPosition:
    """Tracks paper trading position and P&L with hard exit/risk rules (HC #46)."""

    def __init__(self):
        self.position: int = 0  # +1 long, -1 short, 0 flat
        self.entry_price: float = 0.0
        self.entry_time: float = 0.0
        self.entry_tier: str = ""
        self.entry_zscore: float = 0.0

        self.realized_pnl: float = 0.0
        self.total_commission: float = 0.0
        self.n_trades: int = 0
        self.n_winners: int = 0
        self.trades: list[dict] = []

        # MFE/MAE tracking (HC #39, #46c)
        self.mfe_ticks: float = 0.0  # max favorable excursion
        self.mae_ticks: float = 0.0  # max adverse excursion
        self.trailing_stop_price: float = 0.0  # trailing stop activation price

        # Consecutive loss tracking (HC #46f)
        self.consecutive_losses: int = 0
        self.cooldown_until: float = 0.0  # timestamp when cooldown ends

        # Daily circuit breaker (HC #46e)
        self.daily_pnl: float = 0.0
        self.daily_halted: bool = False
        self.daily_date: str = ""

    @property
    def is_flat(self) -> bool:
        return self.position == 0

    def enter(self, direction: int, price: float, t: float, tier: str = "",
              zscore: float = 0.0):
        if not self.is_flat:
            return
        self.position = direction
        self.entry_price = price
        self.entry_time = t
        self.entry_tier = tier
        self.entry_zscore = zscore
        self.mfe_ticks = 0.0
        self.mae_ticks = 0.0
        self.trailing_stop_price = 0.0
        self.total_commission += COMMISSION_PER_SIDE
        log.info("ENTRY %s @ %.2f | tier=%s | z=%.2f",
                 "LONG" if direction > 0 else "SHORT", price, tier, zscore)

    def update_excursion(self, mid_price: float):
        """Track MFE/MAE during position (HC #39). Call on every price update."""
        if self.is_flat or mid_price <= 0:
            return
        if self.position > 0:
            unrealized_ticks = (mid_price - self.entry_price) / TICK_SIZE
        else:
            unrealized_ticks = (self.entry_price - mid_price) / TICK_SIZE
        self.mfe_ticks = max(self.mfe_ticks, unrealized_ticks)
        self.mae_ticks = min(self.mae_ticks, unrealized_ticks)

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
        self.daily_pnl += net
        self.total_commission += COMMISSION_PER_SIDE
        self.n_trades += 1
        if net > 0:
            self.n_winners += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        trade = {
            "direction": "LONG" if self.position > 0 else "SHORT",
            "entry_price": self.entry_price,
            "exit_price": price,
            "gross_pnl": round(gross, 2),
            "net_pnl": round(net, 2),
            "hold_time_s": round(hold_s, 2),
            "reason": reason,
            "tier": self.entry_tier,
            "entry_zscore": round(self.entry_zscore, 3),
            "mfe_ticks": round(self.mfe_ticks, 2),
            "mae_ticks": round(self.mae_ticks, 2),
            "time": datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade)

        log.info("EXIT %s @ %.2f | reason=%s | gross=$%.2f net=$%.2f hold=%.1fs | "
                 "MFE=%.1ft MAE=%.1ft | cumP&L=$%.2f (%d trades, %.0f%% win)",
                 trade["direction"], price, reason, gross, net, hold_s,
                 self.mfe_ticks, self.mae_ticks,
                 self.realized_pnl, self.n_trades,
                 self.n_winners / max(self.n_trades, 1) * 100)

        self.position = 0
        self.entry_price = 0.0
        self.entry_zscore = 0.0
        self.mfe_ticks = 0.0
        self.mae_ticks = 0.0
        return trade

    def check_daily_reset(self):
        """Reset daily P&L tracking at the start of each new day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_date:
            self.daily_date = today
            self.daily_pnl = 0.0
            self.daily_halted = False
            self.consecutive_losses = 0
            self.cooldown_until = 0.0

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
# CNN Mamba v2 Paper Trading Session
# ═══════════════════════════════════════════════════════════════════════════════

class MambaV2PaperSession:
    """CNN Mamba v2 paper trading with HARD EXIT RULES (HC #46).

    Hard exit rules are NON-NEGOTIABLE safety floors:
    (a) Max hold time — absolute cap, exit if trade hasn't hit TP
    (b) Signal decay exit — z-score drops below threshold during trade
    (c) Trailing stop — lock partial profit after MFE exceeds threshold
    (d) Time-of-day gate — no new entries 9:25-9:40 AM ET (HC #41)
    (e) Max daily loss — circuit breaker, stop trading for the day
    (f) Consecutive loss limit — pause after N consecutive losers
    (g) PatchTST reversal exit — (future: when PatchTST inference available)
    """

    TIER_ORDER = {"All": 0, "Top5%": 1, "Top1%": 2, "Top0.5%": 3, "Top0.1%": 4}

    # ── Hard Exit Rules (HC #46 + HC #49: DATA-DRIVEN from MFE/MAE analysis) ──
    # Source: strategy_optimizer_results_10s.json on 250K OOT predictions
    # Mean MFE: 4.37t, Mean MAE: 0.71t, optimal SL: 2t
    MAX_HOLD_SECONDS: float = 120.0       # 2 min — MFE data shows alpha exhausted by ~30-60s
    HARD_STOP_TICKS: float = 2.0          # 2 tick SL — optimal from MFE/MAE (Sortino=7.41)
    SIGNAL_DECAY_THRESHOLD: float = 0.5   # exit if model conviction drops below this
    TRAILING_STOP_ACTIVATION: float = 4.0 # mean MFE is 4.37t — activate trail at MFE mean
    TRAILING_STOP_OFFSET: float = 2.0     # trail by optimal SL (2t) from peak
    MAX_DAILY_LOSS: float = -500.0        # circuit breaker: stop after -$500 daily
    MAX_CONSECUTIVE_LOSSES: int = 3       # pause after 3 consecutive losers
    LOSS_COOLDOWN_SECONDS: float = 300.0  # 5 minute cooldown after consecutive losses

    # Time-of-day gate (HC #41): no new entries 9:25-9:40 AM ET
    # Stored as hours in ET (UTC-4 during EDT, UTC-5 during EST)
    TOD_BLOCK_START_HOUR: float = 9.0 + 25.0 / 60.0  # 9:25 AM ET
    TOD_BLOCK_END_HOUR: float = 9.0 + 40.0 / 60.0    # 9:40 AM ET

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS,
        stats_path: str | Path = DEFAULT_STATS,
        preds_path: str | Path = DEFAULT_PREDS,
        symbol: str = "ESM6",
        exchange: str = "CME",
        window_size: int = 1000,
        stride: int = 500,
        min_tier: str = "Top1%",
        stats_interval_s: float = 300.0,
        max_spread_ticks: float = 4.0,
        device: str = "cuda",
        rl_weights: Optional[str] = None,
        rl_hidden_dim: int = 256,
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

        # Position manager
        self.pos = PaperPosition()

        # Comprehensive trade journal (HC #142)
        config_name = f"rules_top01pct" if min_tier == "Top0.1%" else f"rules_top1pct" if min_tier == "Top1%" else f"rules_{min_tier.lower()}"
        self.journal = TradeJournal(config_name, log_dir=str(_LOG_DIR))

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

        # ── Feature Distribution Validator (catches train/live mismatch) ──
        self._feat_sum = np.zeros(N_FEATURES, dtype=np.float64)
        self._feat_sum_sq = np.zeros(N_FEATURES, dtype=np.float64)
        self._feat_min = np.full(N_FEATURES, np.inf)
        self._feat_max = np.full(N_FEATURES, -np.inf)
        self._feat_n = 0
        self._feat_last_check = 0
        self._FEAT_CHECK_INTERVAL = 10000  # validate every 10k events
        # Expected ranges from training data (feature index -> (expected_mean, expected_std))
        # If live mean deviates >3 std from expected, log WARNING
        self._FEAT_EXPECTED = {
            0: (0.3, 0.4),    # time_delta_log: [0,2], mean ~0.3 for 1-10ms gaps
            3: (0.0, 0.15),   # price_rel_ticks: centered ~0, std ~0.1-0.2
            4: (0.2, 0.4),    # qty_log: centered around small sizes
            9: (0.0, 1.0),    # price_mom_10: z-scored, should be ~N(0,1)
            10: (0.0, 1.0),   # qty_price_mom_50: z-scored
            11: (0.0, 0.3),   # price_sign_momentum_200: /100, should be near 0
            18: (0.0, 1.0),   # vol_weighted_pmom: z-scored
        }

        # Signal log
        self._signals_path = _LOG_DIR / f"mamba_v2_signals_{symbol}_{_SESSION_DATE}.jsonl"
        self._signals_fh = open(self._signals_path, "a", buffering=1)

        # Results output
        self._results_path = _LOG_DIR / f"mamba_v2_results_{symbol}_{_SESSION_DATE}.json"

        # Asyncio
        self._stop = asyncio.Event()
        self._last_stats_time = time.time()

        # ── RL Execution Agent (HC #116 — Monday live paper trading) ──
        self.rl_wrapper: Optional["RLPaperTradingWrapper"] = None
        self.rl_mode = False
        if rl_weights and RL_AVAILABLE:
            try:
                self.rl_wrapper = build_rl_pipeline(
                    weights_path=rl_weights,
                    hidden_dim=rl_hidden_dim,
                    device=device,
                    log_dir=str(_LOG_DIR),
                )
                self.rl_mode = True
                log.info("RL EXECUTION AGENT LOADED: %s (hidden=%d, device=%s)",
                         rl_weights, rl_hidden_dim, device)
            except Exception as e:
                log.error("Failed to load RL agent from %s: %s — falling back to rules",
                          rl_weights, e)
                self.rl_mode = False
        elif rl_weights and not RL_AVAILABLE:
            log.warning("RL weights specified but rl_execution_agent not importable. "
                        "Falling back to rule-based execution.")

        log.info("=" * 70)
        log.info("CNN MAMBA v2 PAPER TRADING SESSION (Razer)")
        log.info("  Execution mode: %s", "RL AGENT" if self.rl_mode else "RULE-BASED")
        log.info("=" * 70)
        log.info("  Model: %s", weights_path)
        log.info("  Device: %s", device)
        log.info("  Symbol: %s | Exchange: %s", symbol, exchange)
        log.info("  Window: %d | Stride: %d", window_size, stride)
        log.info("  Min confidence tier: %s", min_tier)
        log.info("  Stats interval: %.0fs | Max spread: %.1f ticks",
                 stats_interval_s, max_spread_ticks)
        log.info("  ── Hard Exit Rules (HC #46) ──")
        log.info("    Max hold: %.0fs | Hard SL: %.0ft | Signal decay z<%.1f | Trail: +%.0ft→%.0ft",
                 self.MAX_HOLD_SECONDS, self.HARD_STOP_TICKS,
                 self.SIGNAL_DECAY_THRESHOLD,
                 self.TRAILING_STOP_ACTIVATION, self.TRAILING_STOP_OFFSET)
        log.info("    ToD gate: 9:25-9:40 ET | Max daily loss: $%.0f | "
                 "Max consec losses: %d (%.0fs cooldown)",
                 abs(self.MAX_DAILY_LOSS), self.MAX_CONSECUTIVE_LOSSES,
                 self.LOSS_COOLDOWN_SECONDS)
        log.info("  Signal log: %s", self._signals_path)
        log.info("  *** PAPER TRADE MODE -- NO REAL ORDERS ***")
        log.info("=" * 70)

    def _tier_meets_min(self, tier: Optional[str]) -> bool:
        if self.min_tier == "All":
            return True  # Bypass tier gating entirely
        if tier is None:
            return False
        return self.TIER_ORDER.get(tier, 0) >= self.TIER_ORDER.get(self.min_tier, 0)

    # ─────────────────────────────────────────────────────────────────────
    # Hard Exit Rules (HC #46) — NON-NEGOTIABLE SAFETY FLOORS
    # ─────────────────────────────────────────────────────────────────────

    def _get_et_hour(self) -> float:
        """Current hour in Eastern Time (approximate: UTC-4 for EDT)."""
        utc_now = datetime.now(timezone.utc)
        et_hour = (utc_now.hour - 4) % 24 + utc_now.minute / 60.0
        return et_hour

    def _is_tod_blocked(self) -> bool:
        """HC #41: No new entries between 9:25-9:40 AM ET."""
        et_h = self._get_et_hour()
        return self.TOD_BLOCK_START_HOUR <= et_h <= self.TOD_BLOCK_END_HOUR

    def _check_hard_exits(self, now_t: float, current_zscore: float) -> Optional[str]:
        """Check all hard exit conditions. Returns exit reason or None.

        These are NON-NEGOTIABLE — cannot be overridden by the model.
        """
        if self.pos.is_flat:
            return None

        hold_time = now_t - self.pos.entry_time

        # (a) Max hold time — 2 min (data shows alpha exhausted by 30-60s)
        if hold_time >= self.MAX_HOLD_SECONDS:
            return "max_hold_time"

        # (a2) Hard stop loss — 2 ticks (optimal from MFE/MAE analysis)
        if self.pos.mae_ticks <= -self.HARD_STOP_TICKS:
            return "hard_stop_loss"

        # (b) Signal decay — z-score dropped below threshold
        if (abs(current_zscore) < self.SIGNAL_DECAY_THRESHOLD
                and abs(self.pos.entry_zscore) >= self.SIGNAL_DECAY_THRESHOLD):
            return "signal_decay"

        # (c) Trailing stop — lock profit after MFE exceeds activation
        if self.pos.mfe_ticks >= self.TRAILING_STOP_ACTIVATION:
            # Current unrealized P&L in ticks
            if self.pos.position > 0:
                unrealized = (self.mid_price - self.pos.entry_price) / TICK_SIZE
            else:
                unrealized = (self.pos.entry_price - self.mid_price) / TICK_SIZE
            stop_level = self.pos.mfe_ticks - self.TRAILING_STOP_OFFSET
            if unrealized <= stop_level:
                return "trailing_stop"

        return None

    def _can_enter_new_trade(self, now_t: float) -> tuple[bool, str]:
        """Check all entry gate conditions. Returns (can_enter, block_reason)."""
        # (d) Time-of-day gate (HC #41)
        if self._is_tod_blocked():
            return False, "tod_block_9:25-9:40"

        # (e) Max daily loss circuit breaker
        self.pos.check_daily_reset()
        if self.pos.daily_halted:
            return False, "daily_halt"
        if self.pos.daily_pnl <= self.MAX_DAILY_LOSS:
            self.pos.daily_halted = True
            log.warning("CIRCUIT BREAKER: Daily P&L $%.2f <= $%.2f. HALTING for day.",
                        self.pos.daily_pnl, self.MAX_DAILY_LOSS)
            return False, "daily_loss_circuit_breaker"

        # (f) Consecutive loss limit
        if self.pos.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            if now_t < self.pos.cooldown_until:
                return False, f"cooldown_{self.pos.consecutive_losses}_losses"
            else:
                # Cooldown expired — reset
                self.pos.cooldown_until = 0.0

        return True, ""

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

        # Step 1: Compute streaming features (25 smart_v3)
        feat_vec = self.features.update(
            time_delta_log, event_type_id, side_id,
            price_rel_ticks, qty_log, spread_ticks,
        )

        # ── Feature Distribution Validator ──
        # Track running statistics and alert on anomalies
        feat_arr = np.asarray(feat_vec, dtype=np.float64)
        self._feat_sum += feat_arr
        self._feat_sum_sq += feat_arr ** 2
        np.minimum(self._feat_min, feat_arr, out=self._feat_min)
        np.maximum(self._feat_max, feat_arr, out=self._feat_max)
        self._feat_n += 1

        if (self._feat_n - self._feat_last_check) >= self._FEAT_CHECK_INTERVAL:
            self._validate_feature_distributions()
            self._feat_last_check = self._feat_n

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

        # Step 4: Execution logic
        now_t = time.time()
        current_zscore = abs(pred.get("pred_10s", 0.0))  # z-score for signal decay check

        # ── RL Agent Execution Mode (HC #116) ──
        if self.rl_mode and self.rl_wrapper is not None:
            # Build market state dict for RL agent
            spread_actual = 1.0
            if self.best_bid > 0 and self.best_ask > 0:
                spread_actual = (self.best_ask - self.best_bid) / TICK_SIZE

            market_state = {
                "timestamp_s": now_t,
                "mid_price": self.mid_price,
                "best_bid": self.best_bid,
                "best_ask": self.best_ask,
                "best_bid_size": 10.0,  # placeholder — live MBO will provide real sizes
                "best_ask_size": 10.0,
                "spread_ticks": spread_actual,
                "price_rel_ticks": 0.0,
                "is_trade": (event_type_id == 0),
                "trade_side": side_id,
                "trade_qty": math.exp(qty_log) if qty_log > 0 else 1.0,
            }
            prediction_dict = {
                "pred_1s": pred["pred_1s"],
                "pred_5s": pred["pred_5s"],
                "pred_10s": pred["pred_10s"],
                "confidence_tier": self.TIER_ORDER.get(tier, 0) if tier else 0,
            }
            result = self.rl_wrapper.on_event(prediction_dict, market_state)
            if result and result.get("trade_completed"):
                pnl = result.get("pnl_ticks", 0)
                log.info("RL TRADE: %s | PnL: %.2f ticks | Hold: %.1fs",
                         result.get("direction", "?"), pnl, result.get("hold_time_s", 0))
            self.prev_direction = direction
            return

        # ── Rule-Based Execution (original, HC #46) ──

        # Spread filter
        if self.best_bid > 0 and self.best_ask > 0:
            spread_actual = (self.best_ask - self.best_bid) / TICK_SIZE
            if spread_actual > self.max_spread_ticks:
                self.prev_direction = direction
                return

        # UPDATE: Track MFE/MAE while in position (HC #39)
        if not self.pos.is_flat:
            self.pos.update_excursion(self.mid_price)

        # HARD EXIT CHECK (HC #46): max_hold, signal_decay, trailing_stop
        if not self.pos.is_flat:
            hard_exit_reason = self._check_hard_exits(now_t, current_zscore)
            if hard_exit_reason:
                exit_price = self.mid_price if self.mid_price > 0 else (
                        self.best_bid if self.pos.position > 0 else self.best_ask)
                trade = self.pos.exit(exit_price, now_t, reason=hard_exit_reason)
                if trade:
                    self.journal.record_exit(
                        exit_price=trade["exit_price"],
                        exit_reason=trade["reason"],
                        mfe_ticks=trade.get("mfe_ticks", 0),
                        mae_ticks=trade.get("mae_ticks", 0),
                        spread_at_exit=(self.best_ask - self.best_bid) / TICK_SIZE if self.best_bid > 0 and self.best_ask > 0 else 0,
                        bid_at_exit=self.best_bid,
                        ask_at_exit=self.best_ask,
                    )
                    log.info("HARD EXIT: %s after %.1fs hold",
                             hard_exit_reason, trade["hold_time_s"])
                    # Set cooldown if consecutive losses triggered
                    if (self.pos.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES
                            and self.pos.cooldown_until <= 0):
                        self.pos.cooldown_until = now_t + self.LOSS_COOLDOWN_SECONDS
                        log.warning("COOLDOWN: %d consecutive losses. Pausing entries for %.0fs.",
                                    self.pos.consecutive_losses, self.LOSS_COOLDOWN_SECONDS)
                self.prev_direction = direction
                return

        # EXIT: signal flip detection (original logic)
        if not self.pos.is_flat and self.prev_direction is not None:
            if direction != self.prev_direction:
                exit_price = self.mid_price if self.mid_price > 0 else (
                        self.best_bid if self.pos.position > 0 else self.best_ask)
                trade = self.pos.exit(exit_price, now_t, reason="signal_flip")
                if trade:
                    self.journal.record_exit(
                        exit_price=trade["exit_price"],
                        exit_reason=trade["reason"],
                        mfe_ticks=trade.get("mfe_ticks", 0),
                        mae_ticks=trade.get("mae_ticks", 0),
                        spread_at_exit=(self.best_ask - self.best_bid) / TICK_SIZE if self.best_bid > 0 and self.best_ask > 0 else 0,
                        bid_at_exit=self.best_bid,
                        ask_at_exit=self.best_ask,
                    )

                # Check entry gates before re-entering
                can_enter, block_reason = self._can_enter_new_trade(now_t)

                # Immediately re-enter opposite if meets threshold AND passes gates
                if trade and meets_threshold and self.features.is_warm() and can_enter:
                    entry_price = self.mid_price if self.mid_price > 0 else (
                        self.best_ask if direction > 0 else self.best_bid)
                    self.pos.enter(direction, entry_price, now_t, tier=tier or "",
                                   zscore=current_zscore)
                    self.journal.record_entry(
                        entry_price=entry_price,
                        direction="LONG" if direction > 0 else "SHORT",
                        signal_confidence=confidence,
                        signal_tier=tier or "",
                        entry_zscore=current_zscore,
                        pred_1s=pred.get("pred_1s", 0),
                        pred_5s=pred.get("pred_5s", 0),
                        pred_10s=pred.get("pred_10s", 0),
                        entry_action_type="limit",
                        spread_at_entry=(self.best_ask - self.best_bid) / TICK_SIZE if self.best_bid > 0 and self.best_ask > 0 else 0,
                        bid_at_entry=self.best_bid,
                        ask_at_entry=self.best_ask,
                    )
                    self.signals_generated += 1
                elif trade and not can_enter:
                    log.info("ENTRY BLOCKED after flip: %s", block_reason)

                # Set cooldown if consecutive losses triggered
                if (self.pos.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES
                        and self.pos.cooldown_until <= 0):
                    self.pos.cooldown_until = now_t + self.LOSS_COOLDOWN_SECONDS
                    log.warning("COOLDOWN: %d consecutive losses. Pausing entries for %.0fs.",
                                self.pos.consecutive_losses, self.LOSS_COOLDOWN_SECONDS)

        # ENTRY: flat + meets threshold + warm + passes all gates
        elif self.pos.is_flat and meets_threshold and self.features.is_warm():
            can_enter, block_reason = self._can_enter_new_trade(now_t)
            if can_enter:
                entry_price = self.mid_price if self.mid_price > 0 else (
                    self.best_ask if direction > 0 else self.best_bid)
                self.pos.enter(direction, entry_price, now_t, tier=tier or "",
                               zscore=current_zscore)
                self.journal.record_entry(
                    entry_price=entry_price,
                    direction="LONG" if direction > 0 else "SHORT",
                    signal_confidence=confidence,
                    signal_tier=tier or "",
                    entry_zscore=current_zscore,
                    pred_1s=pred.get("pred_1s", 0),
                    pred_5s=pred.get("pred_5s", 0),
                    pred_10s=pred.get("pred_10s", 0),
                    entry_action_type="limit",
                    spread_at_entry=(self.best_ask - self.best_bid) / TICK_SIZE if self.best_bid > 0 and self.best_ask > 0 else 0,
                    bid_at_entry=self.best_bid,
                    ask_at_entry=self.best_ask,
                )
                self.signals_generated += 1
            else:
                # Log first blocked entry per reason per minute to avoid spam
                log.debug("ENTRY BLOCKED: %s", block_reason)

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
        # Time delta in MILLISECONDS to match batch pipeline (mbo_event_pipeline.py)
        # Batch: time_deltas_ms = time_deltas_ns / 1_000_000.0; log1p(time_deltas_ms)
        # Previous bug: used microseconds (1000x too large), saturating feature 0
        delta_ns = max(0, ts_ns - self.prev_ts_ns) if self.prev_ts_ns else 0
        delta_ms = delta_ns / 1_000_000.0
        self.prev_ts_ns = ts_ns

        price_rel = ((price - self.mid_price) / TICK_SIZE
                     if self.mid_price > 0 and price > 0 else 0.0)
        spread = (self.best_ask - self.best_bid) / TICK_SIZE if (
            self.best_bid > 0 and self.best_ask > 0) else 0.0

        self._process_raw_event(
            # Matches batch: log1p(milliseconds), clipped [0, 8] / 4.0 in StreamingFeatures
            time_delta_log=math.log1p(delta_ms) if delta_ms > 0 else 0.0,
            event_type_id=int(etype),
            side_id=int(side),
            price_rel_ticks=price_rel,
            # Matches batch: log1p(qty) not log(qty). Batch min = log1p(1) = 0.693
            qty_log=math.log1p(max(qty, 1)),
            spread_ticks=spread,
            timestamp_ns=ts_ns,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Feature Distribution Validator — catches train/live mismatch early
    # ─────────────────────────────────────────────────────────────────────

    def _validate_feature_distributions(self):
        """Check live feature distributions against training expectations.

        Runs every 10K events. Logs WARNING if any feature is saturated or
        has mean/std far outside expected range. This would have caught the
        price_ticks bug (features 3,9,10,11,18 all saturated at +2.0).
        """
        n = self._feat_n
        if n < 1000:
            return

        means = self._feat_sum / n
        variances = (self._feat_sum_sq / n) - (means ** 2)
        stds = np.sqrt(np.maximum(variances, 0.0))

        FEAT_NAMES = [
            "time_delta_log", "event_type_id", "side_id", "price_rel_ticks",
            "qty_log", "spread_ticks", "cancel_side_asym", "rolling_ofi_500",
            "event_density", "price_mom_10", "qty_price_mom_50",
            "price_sign_mom_200", "entropy_200", "fill_add_restoration",
            "spread_velocity", "queue_replenishment", "mom_divergence",
            "ofi_x_spread", "vol_weighted_pmom", "buy_sell_intensity",
            "realized_vol", "sweep_intensity", "ofi_short_100",
            "ofi_long_2000", "ofi_acceleration",
        ]

        anomalies = []

        for idx, (exp_mean, exp_std) in self._FEAT_EXPECTED.items():
            live_mean = means[idx]
            # Flag if live mean deviates >4 expected-stds from expected mean
            if exp_std > 0 and abs(live_mean - exp_mean) > 4.0 * exp_std:
                name = FEAT_NAMES[idx] if idx < len(FEAT_NAMES) else f"feat_{idx}"
                anomalies.append(
                    f"  [{idx}] {name}: live_mean={live_mean:.4f} "
                    f"(expected ~{exp_mean:.2f}±{exp_std:.2f}), "
                    f"live_std={stds[idx]:.4f}, "
                    f"range=[{self._feat_min[idx]:.3f}, {self._feat_max[idx]:.3f}]"
                )

        # Check for saturated features (std near 0 but value at extreme)
        for idx in range(min(len(means), N_FEATURES)):
            if stds[idx] < 0.001 and abs(means[idx]) > 1.5:
                name = FEAT_NAMES[idx] if idx < len(FEAT_NAMES) else f"feat_{idx}"
                if not any(f"[{idx}]" in a for a in anomalies):
                    anomalies.append(
                        f"  [{idx}] {name}: SATURATED at {means[idx]:.4f} "
                        f"(std={stds[idx]:.6f}) — likely encoding bug!"
                    )

        if anomalies:
            log.warning("!! FEATURE DISTRIBUTION ANOMALY DETECTED (n=%d events):", n)
            for a in anomalies:
                log.warning(a)
            log.warning("  This may indicate a train/live feature mismatch. "
                        "Check _encode_and_process() and StreamingFeaturesSmartV3.")
        else:
            log.info("[OK] Feature distributions OK after %d events "
                     "(means: time_dt=%.3f, price_rel=%.3f, qty=%.3f)",
                     n, means[0], means[3], means[4])

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

        tasks = [asyncio.create_task(stats_loop())]

        def handle_signal(*_):
            log.info("Shutdown signal received")
            self._stop.set()

        # Windows asyncio doesn't support add_signal_handler. Use try/except.
        try:
            for s in (_sig.SIGINT, _sig.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(s, handle_signal)
        except (NotImplementedError, AttributeError):
            # Windows: rely on KeyboardInterrupt propagation only
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
            # Default path matches recorder's fan-out output
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

        tasks = [asyncio.create_task(stats_loop())]

        def handle_signal(*_):
            log.info("Shutdown signal received")
            self._stop.set()

        try:
            for s in (_sig.SIGINT, _sig.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(s, handle_signal)
        except (NotImplementedError, AttributeError):
            pass

        # Wait for the events file to appear
        while not self._stop.is_set():
            if os.path.exists(events_path):
                break
            log.info("  Waiting for events file to appear...")
            await asyncio.sleep(2.0)

        try:
            with open(events_path, "r") as fh:
                # Seek to end — we only process new events
                fh.seek(0, 2)
                log.info("FOLLOW: Positioned at end of file, waiting for new events...")

                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        await asyncio.sleep(0.1)  # Poll 10x/sec
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts_ns = ev.get("timestamp_ns", 0)

                    # Update BBO from event data
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

                    # Encode and process through the model
                    action = ev.get("action", 0)
                    side = ev.get("side", 0)
                    price_ticks = ev.get("price_ticks", 0)
                    size = ev.get("size", 1)

                    # price_ticks is ABSOLUTE (e.g. bid_price / TICK_SIZE = 28705), not relative
                    price = price_ticks * TICK_SIZE

                    # Map action to event_type: 0=BBO, 3=trade
                    etype = 3.0 if action >= 2 else 0.0  # action 0/1=add/modify/cancel, 2+=trade
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
                log.info("  [%d/%d] %.0f ev/s | preds=%d signals=%d P&L=$%.2f",
                         i + 1, N, (i + 1) / elapsed,
                         self.predictions_made, self.signals_generated,
                         self.pos.realized_pnl)

        self._final_report()

    # ─────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────

    def _periodic_summary(self):
        s = self.pos.summary()
        log.info("STATUS | events=%d preds=%d signals=%d | %d trades WR=%.0f%% P&L=$%.2f",
                 self.events_processed, self.predictions_made, self.signals_generated,
                 s["n_trades"], s["win_rate"], s["realized_pnl"])
        if self.journal.get_trade_count() > 0:
            log.info(self.journal.get_summary_str())

    def _final_report(self):
        s = self.pos.summary()

        # Exit reason breakdown (HC #44 execution checklist)
        exit_reasons = {}
        for t in self.pos.trades:
            r = t.get("reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # MFE/MAE summary
        mfes = [t.get("mfe_ticks", 0) for t in self.pos.trades]
        maes = [t.get("mae_ticks", 0) for t in self.pos.trades]
        hold_times = [t.get("hold_time_s", 0) for t in self.pos.trades]

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
            **s,
            "exit_reasons": exit_reasons,
            "hard_exit_rules": {
                "max_hold_seconds": self.MAX_HOLD_SECONDS,
                "signal_decay_threshold": self.SIGNAL_DECAY_THRESHOLD,
                "trailing_stop_activation": self.TRAILING_STOP_ACTIVATION,
                "trailing_stop_offset": self.TRAILING_STOP_OFFSET,
                "max_daily_loss": self.MAX_DAILY_LOSS,
                "max_consecutive_losses": self.MAX_CONSECUTIVE_LOSSES,
                "tod_block": "9:25-9:40 ET",
            },
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
        if self.pos.trades:
            log.info("  Avg hold: %.1fs | Avg MFE: %.1ft | Avg MAE: %.1ft",
                     sum(hold_times) / len(hold_times),
                     sum(mfes) / len(mfes),
                     sum(maes) / len(maes))
        log.info("  Exit reasons: %s", exit_reasons)
        log.info("  Results saved: %s", self._results_path)
        log.info("  Signals saved: %s", self._signals_path)
        log.info("=" * 70)

        self.journal.close()
        self._signals_fh.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CNN Mamba v2 Paper Trading (Razer)")
    parser.add_argument("--replay", type=str, help="Replay from NPZ file")
    parser.add_argument("--symbol", type=str, default="ESM6", help="Symbol")
    parser.add_argument("--exchange", type=str, default="CME", help="Exchange")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--preds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--min-tier", type=str, default="Top1%",
                        choices=["All", "Top5%", "Top1%", "Top0.5%", "Top0.1%"])
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=500)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--follow-events", nargs="?", const=True, default=None,
                        help="Tail recorder's live_events.jsonl instead of Rithmic. "
                             "Optionally pass a path to the JSONL file.")
    # RL execution agent (HC #116)
    parser.add_argument("--rl-weights", type=str, default=None,
                        help="Path to trained RL agent .pt weights. Enables RL execution mode.")
    parser.add_argument("--rl-hidden-dim", type=int, default=256,
                        help="Hidden dim of RL actor-critic (must match training)")
    args = parser.parse_args()

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
        rl_weights=args.rl_weights,
        rl_hidden_dim=args.rl_hidden_dim,
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
