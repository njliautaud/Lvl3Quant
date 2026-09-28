"""Risk Manager for paper trading (HC #46, #226, #230, #231C).

Manages:
- Max hold time enforcement
- Trailing stop based on MFE
- Consecutive loss gating (pause after N losses)
- Daily loss limit
- Signal decay threshold calibration
- PatchTST reversal detection (Rule 7)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Risk configuration with sane defaults per HC #226, #230, #231C."""
    # Max hold time (HC #226 + #231C: bounded by signal horizons ~30s)
    max_hold_seconds: float = 60.0  # default 1 min

    # Trailing stop (HC #46)
    trailing_stop_mfe_trigger_ticks: float = 2.0  # start trailing after +2 ticks MFE
    trailing_stop_lock_ticks: float = 1.0  # lock in MFE - this many ticks

    # Daily loss limit
    max_daily_loss: float = -200.0  # $ — stop trading for day

    # Consecutive loss gating (HC #230)
    max_consecutive_losses: int = 5
    pause_after_losses_seconds: float = 300.0  # 5 min cooldown

    # Signal decay threshold (calibrated from predictions)
    signal_decay_threshold: float = 0.3  # minimum confidence to stay in
    signal_decay_calibrated: bool = False

    # PatchTST reversal detection (Rule 7)
    patchtst_reversal_enabled: bool = True


class RiskManager:
    """Stateful risk manager — tracks position, losses, MFE for exit decisions."""

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

        # Position tracking
        self._entry_price: float = 0.0
        self._entry_time: float = 0.0
        self._direction: int = 0  # +1 long, -1 short
        self._confidence: float = 0.0
        self._mfe_ticks: float = 0.0  # max favorable excursion since entry
        self._in_position: bool = False

        # Loss tracking
        self._consecutive_losses: int = 0
        self._last_loss_time: float = 0.0
        self._daily_pnl: float = 0.0
        self._paused_until: float = 0.0
        self._total_risk_exits: int = 0
        self._exits_by_reason: dict = {}

    # ── Entry gating ──

    def can_enter(self) -> Tuple[bool, Optional[str]]:
        """Check if a new entry is allowed. Returns (allowed, reason_if_blocked)."""
        now = time.time()

        # Daily loss limit
        if self._daily_pnl <= self.cfg.max_daily_loss:
            return False, f"daily_loss_limit ({self._daily_pnl:.1f})"

        # Consecutive loss pause
        if now < self._paused_until:
            remaining = self._paused_until - now
            return False, f"consec_loss_pause ({remaining:.0f}s remaining)"

        return True, None

    def on_entry(self, price: float, direction: int, confidence: float):
        """Record new position entry."""
        self._entry_price = price
        self._entry_time = time.time()
        self._direction = direction
        self._confidence = confidence
        self._mfe_ticks = 0.0
        self._in_position = True

    # ── Exit checks ──

    def check_exit_on_price(self, current_price: float) -> Optional[str]:
        """Check price-based exit rules (trailing stop, max hold). Call on every tick."""
        if not self._in_position:
            return None

        TICK_SIZE = 0.25
        # Update MFE
        if self._direction > 0:
            excursion = (current_price - self._entry_price) / TICK_SIZE
        else:
            excursion = (self._entry_price - current_price) / TICK_SIZE

        if excursion > self._mfe_ticks:
            self._mfe_ticks = excursion

        # Rule 3: Trailing stop
        if self._mfe_ticks >= self.cfg.trailing_stop_mfe_trigger_ticks:
            drawdown_from_mfe = self._mfe_ticks - excursion
            lock_level = self._mfe_ticks - self.cfg.trailing_stop_lock_ticks
            if excursion <= lock_level:
                return f"trailing_stop (MFE={self._mfe_ticks:.1f}, now={excursion:.1f})"

        # Max hold time
        hold_time = time.time() - self._entry_time
        if hold_time >= self.cfg.max_hold_seconds:
            return f"max_hold ({hold_time:.0f}s >= {self.cfg.max_hold_seconds:.0f}s)"

        return None

    def check_exit_on_prediction(
        self,
        current_confidence: float,
        current_direction: int,
        patchtst_direction: Optional[int] = None,
    ) -> Optional[str]:
        """Check prediction-based exit rules. Call on each new prediction."""
        if not self._in_position:
            return None

        # Rule 1: Signal decay — confidence dropped below threshold
        if self.cfg.signal_decay_calibrated:
            if current_confidence < self.cfg.signal_decay_threshold:
                return f"signal_decay (conf={current_confidence:.3f} < {self.cfg.signal_decay_threshold:.3f})"

        # Rule 2: Direction reversal from model
        if current_direction != 0 and current_direction != self._direction:
            # Strong reversal signal — exit
            return f"direction_reversal (pos={self._direction}, signal={current_direction})"

        # Rule 7: PatchTST reversal detection
        if (self.cfg.patchtst_reversal_enabled and
                patchtst_direction is not None and
                patchtst_direction != 0 and
                patchtst_direction != self._direction):
            return f"patchtst_reversal (pos={self._direction}, patchtst={patchtst_direction})"

        return None

    def on_exit(self, exit_price: float, net_pnl: float, reason: str = "unknown"):
        """Record position exit and update loss tracking."""
        self._in_position = False
        self._daily_pnl += net_pnl
        self._total_risk_exits += 1
        self._exits_by_reason[reason] = self._exits_by_reason.get(reason, 0) + 1

        if net_pnl < 0:
            self._consecutive_losses += 1
            self._last_loss_time = time.time()
            if self._consecutive_losses >= self.cfg.max_consecutive_losses:
                self._paused_until = time.time() + self.cfg.pause_after_losses_seconds
                log.warning("RISK: %d consecutive losses — pausing %.0fs",
                            self._consecutive_losses, self.cfg.pause_after_losses_seconds)
        else:
            self._consecutive_losses = 0

    # ── Calibration ──

    def calibrate_signal_decay_threshold(self, preds_path: str):
        """Calibrate signal decay threshold from historical predictions file."""
        try:
            data = np.load(preds_path, allow_pickle=True)
            if "predictions" in data:
                preds = data["predictions"]
            elif "preds" in data:
                preds = data["preds"]
            else:
                log.warning("No predictions key found in %s, using default threshold", preds_path)
                return

            # Use 30th percentile of absolute predictions as decay threshold
            abs_preds = np.abs(preds.flatten())
            threshold = float(np.percentile(abs_preds, 30))
            self.cfg.signal_decay_threshold = max(threshold, 0.1)  # floor at 0.1
            self.cfg.signal_decay_calibrated = True
            log.info("Calibrated signal_decay_threshold = %.4f from %s", threshold, preds_path)
        except Exception as e:
            log.warning("Failed to calibrate signal decay from %s: %s", preds_path, e)

    # ── Reporting ──

    def status_line(self) -> str:
        """One-line status for heartbeat/logging."""
        parts = [f"pnl={self._daily_pnl:.1f}"]
        if self._consecutive_losses > 0:
            parts.append(f"consec_L={self._consecutive_losses}")
        if self._in_position:
            parts.append(f"MFE={self._mfe_ticks:.1f}t")
        if time.time() < self._paused_until:
            parts.append(f"PAUSED({self._paused_until - time.time():.0f}s)")
        return " | ".join(parts)

    def summary(self) -> dict:
        """End-of-session summary dict."""
        return {
            "daily_pnl": self._daily_pnl,
            "total_risk_exits": self._total_risk_exits,
            "exits_by_reason": dict(self._exits_by_reason),
            "max_consecutive_losses": self._consecutive_losses,
            "signal_decay_threshold": self.cfg.signal_decay_threshold,
            "signal_decay_calibrated": self.cfg.signal_decay_calibrated,
        }
