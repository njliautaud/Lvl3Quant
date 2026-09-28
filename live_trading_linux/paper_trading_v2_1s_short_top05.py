#!/usr/bin/env python3
"""
paper_trading_v2_1s_short_top05.py
==================================
Spec-compliant paper trading engine for the v2_1s_short_top05 strategy.

Source spec: output/hc417_v2_1s_short_top05_DEPLOYMENT_SPEC.md
HC reference: HC #418 (deploy directive) + HC #419 (proceed)

Strategy summary (all 19 spec gates implemented; ALL 9 kill-switches wired):

  Model:        CNN-Mamba v2 (fold_10_best.pt)                    [head 0 = pred_log_ret_1s]
  Direction:    SHORT only                                         [skip every long signal]
  Confidence:   per-day 99.5th percentile within current RTH AND  pred_log_ret_1s <= -0.6926
                cold-start: first 30 min uses GLOBAL floor only
  Entry:        passive_at_touch LIMIT @ best ASK, post-only      [10s cancel timer]
  Bracket:      TP1 +0.4782 tk (half size) | TP2 +0.9564 tk (half size) | SL -0.5686 tk (full)
                (For 1-contract size: single TP at TP2, hard SL at -0.5686 tk.)
  Hours:        RTH only (09:30-16:00 America/New_York, DST-aware via pytz)
  Sizing:       1 contract, max 1 concurrent, 5s re-entry cooldown after IDLE
  Cost:         TICK_SIZE=0.25, POINT_VALUE=$12.50 (ES), COMM_PER_SIDE=$2.35

Backtest expectation (from output/hc417_hc413_v2native_mfe/):
  net = +0.274 tk/fill ($3.43), n_fills = 639/25d, WR 84.8%, PF 2.92, day_conc 0.132

This module exports a pure  simulate_trades(predictions, mids, ts_ns, dates)  function
used by  scripts/v2_1s_short_top05_reproduce_backtest.py  to verify reproduction.

PAPER TRADE ONLY. No real orders submitted (paper account via rithmic_client). To go
live, the paper_account flag must be flipped manually with a code commit (not a CLI flag).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

# ── Env setup BEFORE importing v2 model code ───────────────────────────────────
os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")

# Make both the script's own dir AND the Lvl3Quant root importable so we can
# resolve cnn_mamba_v2_inference whether it lives in ./live_trading_linux/
# (Jupiter) or ../live_trading/ (Razer, where live_trading_linux/cnn_mamba_v2_inference.py
# is a shim that re-imports from live_trading.cnn_mamba_v2_inference).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — frozen per deploy spec §1, §2, §3
# ═══════════════════════════════════════════════════════════════════════════════

# Market microstructure (ES futures — DO NOT CHANGE without code commit)
TICK_SIZE: float = 0.25
POINT_VALUE: float = 12.50        # ES = $12.50/tick. NQ would be $5/tick — DO NOT use NQ value.
TICK_VALUE: float = 12.50
COMMISSION_PER_SIDE: float = 2.35  # AMP $4.70 round-trip / 2

# Entry cost already baked into backtested TP/SL (passive_at_touch = 0.376 tk)
ENTRY_COST_TICKS: float = 0.376

# Bracket parameters (v2_1s_short_top05 row from hc417_hc413_v2native_mfe/scalping_backtest_results.csv)
TP1_TICKS: float = 0.4782   # half size partial profit
TP2_TICKS: float = 0.9564   # full MFE target (final tp for 1-contract)
SL_TICKS:  float = 0.5686   # capped at MAE; stop-loss tick distance

# Confidence gating (per spec §1)
GLOBAL_CONFIDENCE_FLOOR_1S: float = -0.6926  # pred_log_ret_1s must be <= this
PERCENTILE_RANK_THRESHOLD: float = 99.5      # rank ≥ 99.5th within session
COLD_START_MINUTES: int = 30                 # use global floor only for first 30 min RTH

# Order / position management
CANCEL_WINDOW_SECONDS: float = 10.0          # 40 evals × 250ms stride
REENTRY_COOLDOWN_SECONDS: float = 5.0
MAX_CONCURRENT_POSITIONS: int = 1
POSITION_SIZE: int = 1                       # contracts; hard-coded, no scale-up

# RTH window (America/New_York)
RTH_OPEN  = dtime(9, 30)   # 09:30 ET
RTH_CLOSE = dtime(16, 0)   # 16:00 ET

# Stride / window (v2 model defaults)
WINDOW_SIZE: int = 1000
STRIDE: int = 250    # evals every 250ms

# Kill-switch thresholds (spec §3)
KS_DAILY_LOSS_TICKS: float = -10.0           # ≤ -10 tk realised PnL today  → halt until next day
KS_WEEKLY_LOSS_TICKS: float = -25.0          # ≤ -25 tk rolling 5-day      → halt until user re-enables
KS_CONSEC_LOSSES: int = 5                    # 5 consecutive losing fills  → 1h pause then re-arm
KS_CONSEC_PAUSE_SECONDS: float = 3600.0
KS_REALISED_DRIFT_NET: float = -0.5          # rolling net/fill on last 10 fills < -0.5 tk → halt+manual
KS_REALISED_DRIFT_WINDOW: int = 10
KS_IC_DRIFT_FLOOR: float = 0.15              # rolling 5d/100-fill IC < 0.15 → halt+manual
KS_IC_DRIFT_WINDOW: int = 100
KS_STALE_SIGNAL_SECONDS: float = 30.0        # no prediction >30s → pause new entries
KS_CONNECTIVITY_LOSS_SECONDS: float = 90.0   # HC #438 FIX: was 5.0; tail-mode "broker" = file tail.
                                             # heartbeat_loop fires every 30s; threshold must be >> that
                                             # or RTH-close event-stream slowdown self-halts the trader.
KS_PER_TRADE_SL_OVERRUN_SECONDS: float = 30.0  # SL not filled within 30s of trigger → market exit

# Canonical SHA-256 of fold_10_best.pt — set after first calibration; refuse start on mismatch.
# Leave None to skip check (NOT recommended; enable for production).
CANONICAL_CKPT_SHA256: Optional[str] = None

# Default file paths — Razer Windows (override via env / CLI)
LVL3 = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))
DEFAULT_WEIGHTS = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
DEFAULT_STATS   = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"

# Logs / heartbeat
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
HEARTBEAT_PATH = LVL3 / "output" / "v2_1s_short_top05_heartbeat.json"
HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
SESSION_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"v2_1s_short_top05_paper_{SESSION_TAG}.log"
JSONL_FILE = LOG_DIR / f"v2_1s_short_top05_paper_{SESSION_TAG}.jsonl"

# Logger
log = logging.getLogger("v2_1s_short_top05")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fh = logging.FileHandler(LOG_FILE)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_sh)


def jlog(event: str, **fields: Any) -> None:
    """Append a single JSON line to the structured log file."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    try:
        with JSONL_FILE.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # pragma: no cover
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# DISCORD ALERT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class DiscordNotifier:
    """Fires a webhook POST on every state change. Webhook URL via env DISCORD_WEBHOOK_URL."""

    def __init__(self, webhook_url: Optional[str] = None, throttle_seconds: float = 1.0):
        self.url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        self.throttle = throttle_seconds
        self._last_send_ts = 0.0
        self._enabled = bool(self.url)
        if not self._enabled:
            log.warning("DiscordNotifier: no webhook URL — alerts will only log to file.")

    def send(self, message: str, level: str = "info") -> None:
        jlog("discord_alert", level=level, message=message)
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_send_ts < self.throttle:
            return  # throttle to avoid spam
        self._last_send_ts = now
        try:
            import requests  # imported lazily; Razer should have it
            prefix = {"info": "ℹ️", "warn": "⚠️", "error": "🚨", "fill": "✅", "kill": "🛑"}.get(level, "•")
            payload = {"content": f"{prefix} [v2_1s_short_top05] {message}"}
            requests.post(self.url, json=payload, timeout=3)
        except Exception as e:  # pragma: no cover
            log.error("Discord webhook failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES — time zone, RTH gate, SHA-256, percentile rank
# ═══════════════════════════════════════════════════════════════════════════════

def now_et() -> datetime:
    """Current time in America/New_York (DST-aware)."""
    try:
        import pytz
        return datetime.now(pytz.timezone("America/New_York"))
    except ImportError:
        # Fallback: UTC-4 (EDT). Will be wrong in standard time. WARN at startup.
        log.warning("pytz not available — falling back to fixed UTC-4 offset (WRONG in Nov-Mar).")
        return datetime.now(timezone.utc) - timedelta(hours=4)


def is_rth(t_et: Optional[datetime] = None) -> bool:
    """09:30-16:00 ET, weekdays only."""
    t = t_et or now_et()
    if t.weekday() >= 5:  # Sat=5, Sun=6
        return False
    tod = t.time()
    return RTH_OPEN <= tod < RTH_CLOSE


def minutes_into_rth(t_et: Optional[datetime] = None) -> Optional[float]:
    """Minutes since 09:30 ET today; None if outside RTH."""
    t = t_et or now_et()
    if not is_rth(t):
        return None
    open_dt = t.replace(hour=9, minute=30, second=0, microsecond=0)
    return (t - open_dt).total_seconds() / 60.0


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# PER-DAY CONFIDENCE PERCENTILE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class PerDayPercentileTracker:
    """
    Tracks the running 99.5th percentile of `signed_short = -pred_log_ret_1s`
    within the current RTH session. Resets at the start of each new trading day.
    """

    def __init__(self, percentile: float = PERCENTILE_RANK_THRESHOLD):
        self.percentile = percentile
        self._day_key: Optional[str] = None
        self._values: List[float] = []
        self._cached_threshold: Optional[float] = None
        self._dirty = True

    def _reset_if_new_day(self, t_et: datetime) -> None:
        dk = t_et.strftime("%Y-%m-%d")
        if self._day_key != dk:
            self._day_key = dk
            self._values = []
            self._cached_threshold = None
            self._dirty = True

    def push(self, pred_log_ret_1s: float, t_et: Optional[datetime] = None) -> None:
        """Add a new signed-short observation."""
        t = t_et or now_et()
        self._reset_if_new_day(t)
        # We track signed_short = -pred_1s; rank-eligible only for short side
        if pred_log_ret_1s < 0:
            self._values.append(-pred_log_ret_1s)
            self._dirty = True

    @property
    def n_observations(self) -> int:
        return len(self._values)

    def threshold(self) -> Optional[float]:
        """99.5th percentile of signed_short. None if fewer than 200 obs (statistically thin)."""
        if not self._values or len(self._values) < 200:
            return None
        if self._dirty:
            arr = np.asarray(self._values)
            self._cached_threshold = float(np.percentile(arr, self.percentile))
            self._dirty = False
        return self._cached_threshold

    def passes(self, pred_log_ret_1s: float, t_et: Optional[datetime] = None) -> bool:
        """True if pred passes BOTH gates: global floor AND per-day percentile (if available).
        During first COLD_START_MINUTES, percentile is skipped and only global floor applies.
        """
        if pred_log_ret_1s > GLOBAL_CONFIDENCE_FLOOR_1S:
            return False  # not negative enough (or positive — we're SHORT only)
        t = t_et or now_et()
        mi = minutes_into_rth(t)
        if mi is None:
            return False
        if mi < COLD_START_MINUTES:
            return True  # cold-start: global floor alone
        thr = self.threshold()
        if thr is None:
            return True  # not enough obs yet — fall back to global floor
        return (-pred_log_ret_1s) >= thr


# ═══════════════════════════════════════════════════════════════════════════════
# KILL-SWITCH MANAGER — 9 switches per spec §3
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KillSwitchState:
    halted_until: Optional[float] = None      # unix ts; None=not halted
    halted_until_user_resume: bool = False    # requires manual user reset
    halted_reason: Optional[str] = None
    last_prediction_ts: Optional[float] = None
    last_broker_seen_ts: Optional[float] = None


class KillSwitchManager:

    def __init__(self, notifier: DiscordNotifier):
        self.n = notifier
        self.state = KillSwitchState()

        # Realised PnL ring buffer (for daily/weekly + drift checks)
        self.daily_net_ticks: float = 0.0
        self.daily_date: str = ""
        self.weekly_pnl_history: Deque[Tuple[str, float]] = deque(maxlen=10)  # (date, day_net_tk)

        # Per-fill ring buffers
        self.recent_fills: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.consec_losses: int = 0

        # IC tracking (rolling preds + realised log-rets at 1s)
        self.ic_preds: Deque[float] = deque(maxlen=KS_IC_DRIFT_WINDOW * 5)
        self.ic_targets: Deque[float] = deque(maxlen=KS_IC_DRIFT_WINDOW * 5)

    # --- state transitions -----------------------------------------------------

    def is_halted(self) -> bool:
        if self.state.halted_until_user_resume:
            return True
        if self.state.halted_until is not None and time.time() < self.state.halted_until:
            return True
        # Auto-clear a timed halt
        if self.state.halted_until is not None and time.time() >= self.state.halted_until:
            log.info("Kill-switch %s timed halt auto-cleared.", self.state.halted_reason)
            self.state.halted_until = None
            self.state.halted_reason = None
        return False

    def halt_temporary(self, seconds: float, reason: str) -> None:
        self.state.halted_until = time.time() + seconds
        self.state.halted_reason = reason
        log.warning("KILL-SWITCH TEMP HALT (%.0fs): %s", seconds, reason)
        self.n.send(f"TEMP HALT {seconds:.0f}s: {reason}", level="kill")

    def halt_manual(self, reason: str) -> None:
        self.state.halted_until_user_resume = True
        self.state.halted_reason = reason
        log.error("KILL-SWITCH MANUAL HALT: %s — user must re-enable.", reason)
        self.n.send(f"MANUAL HALT (USER ACTION NEEDED): {reason}", level="kill")

    def user_resume(self) -> None:
        self.state.halted_until = None
        self.state.halted_until_user_resume = False
        self.state.halted_reason = None
        log.info("Kill-switch state CLEARED by user resume.")
        self.n.send("Kill-switch CLEARED by user.", level="info")

    # --- event hooks -----------------------------------------------------------

    def on_prediction(self, pred_1s: float, target_1s: Optional[float] = None) -> None:
        self.state.last_prediction_ts = time.time()
        # IC drift tracking — only when realised label is available (lookback)
        if target_1s is not None and not (math.isnan(pred_1s) or math.isnan(target_1s)):
            self.ic_preds.append(pred_1s)
            self.ic_targets.append(target_1s)

    def on_broker_heartbeat(self) -> None:
        self.state.last_broker_seen_ts = time.time()

    def on_fill(self, kind: str, net_ticks: float, pred_1s: float = math.nan,
                realized_1s: float = math.nan) -> None:
        """kind ∈ {tp1, tp2, sl, time_stop, manual_exit, market_exit}"""
        today = now_et().strftime("%Y-%m-%d")
        if self.daily_date != today:
            # rollover
            if self.daily_date:
                self.weekly_pnl_history.append((self.daily_date, self.daily_net_ticks))
            self.daily_date = today
            self.daily_net_ticks = 0.0
        self.daily_net_ticks += net_ticks
        rec = {"ts": time.time(), "kind": kind, "net_tk": net_ticks,
               "pred_1s": pred_1s, "realized_1s": realized_1s}
        self.recent_fills.append(rec)
        if net_ticks <= 0:
            self.consec_losses += 1
        else:
            self.consec_losses = 0
        jlog("fill", **rec)
        self._evaluate_post_fill_killers()

    # --- periodic checks (call once per stride / event tick) -------------------

    def evaluate(self) -> None:
        """Check time-based and rolling-window kill-switches. Call frequently."""
        # Stale signal
        if self.state.last_prediction_ts is not None:
            if (time.time() - self.state.last_prediction_ts) > KS_STALE_SIGNAL_SECONDS:
                self.halt_temporary(60.0, f"stale_signal (>{KS_STALE_SIGNAL_SECONDS}s no prediction)")

        # Broker connectivity loss
        # HC #438 FIX: was halt_manual (sticky / requires user resume). In tail-mode the
        # "broker" is just the live_events.jsonl file tail — there is no broker session to
        # reconnect, and events naturally resume. A sticky halt here means the trader
        # silently dies every day at RTH close. Use a temporary halt that auto-clears so
        # if events resume the trader recovers on its own.
        if self.state.last_broker_seen_ts is not None:
            if (time.time() - self.state.last_broker_seen_ts) > KS_CONNECTIVITY_LOSS_SECONDS:
                self.halt_temporary(120.0, f"broker_connectivity_loss (>{KS_CONNECTIVITY_LOSS_SECONDS}s)")

        # Realised drift over last N fills
        if len(self.recent_fills) >= KS_REALISED_DRIFT_WINDOW:
            window = list(self.recent_fills)[-KS_REALISED_DRIFT_WINDOW:]
            avg = float(np.mean([r["net_tk"] for r in window]))
            if avg < KS_REALISED_DRIFT_NET:
                self.halt_manual(
                    f"realised_drift: last {KS_REALISED_DRIFT_WINDOW} fills avg={avg:+.3f} tk < {KS_REALISED_DRIFT_NET}"
                )

        # IC drift
        if len(self.ic_preds) >= KS_IC_DRIFT_WINDOW:
            p = np.asarray(list(self.ic_preds)[-KS_IC_DRIFT_WINDOW * 5:])
            t = np.asarray(list(self.ic_targets)[-KS_IC_DRIFT_WINDOW * 5:])
            if len(p) >= KS_IC_DRIFT_WINDOW:
                ic = float(np.corrcoef(p, t)[0, 1])
                if ic < KS_IC_DRIFT_FLOOR:
                    self.halt_manual(f"ic_drift: rolling IC={ic:+.3f} < {KS_IC_DRIFT_FLOOR}")

    def _evaluate_post_fill_killers(self) -> None:
        # Hard daily loss
        if self.daily_net_ticks <= KS_DAILY_LOSS_TICKS:
            # Halt until next trading day open
            end_of_day = now_et().replace(hour=23, minute=59, second=59)
            seconds_until = max(60.0, (end_of_day - now_et()).total_seconds())
            self.halt_temporary(seconds_until,
                                f"daily_loss_cap: today_net={self.daily_net_ticks:+.2f} tk")

        # Hard weekly loss
        week_net = self.daily_net_ticks + sum(d_net for _, d_net in list(self.weekly_pnl_history)[-4:])
        if week_net <= KS_WEEKLY_LOSS_TICKS:
            self.halt_manual(f"weekly_loss_cap: 5d_net={week_net:+.2f} tk")

        # Consecutive losses
        if self.consec_losses >= KS_CONSEC_LOSSES:
            self.halt_temporary(KS_CONSEC_PAUSE_SECONDS,
                                f"consec_losses={self.consec_losses}")
            self.consec_losses = 0  # reset after pause armed

    # --- summary ---------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "halted": self.is_halted(),
            "halt_reason": self.state.halted_reason,
            "halted_until": self.state.halted_until,
            "halted_until_user": self.state.halted_until_user_resume,
            "daily_net_tk": self.daily_net_ticks,
            "consec_losses": self.consec_losses,
            "n_recent_fills": len(self.recent_fills),
            "ic_window_n": len(self.ic_preds),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PURE simulate_trades — used by live engine AND by backtest reproduction script
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _SimState:
    pos: int = 0
    entry_px: float = 0.0
    entry_idx: int = -1
    entry_ts: float = 0.0


def simulate_trades(
    pred_log_ret_1s: np.ndarray,
    mid_ticks: np.ndarray,
    ts_ns: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
    *,
    global_floor: float = GLOBAL_CONFIDENCE_FLOOR_1S,
    percentile: float = PERCENTILE_RANK_THRESHOLD,
    tp1_tk: float = TP1_TICKS,
    tp2_tk: float = TP2_TICKS,
    sl_tk: float = SL_TICKS,
    entry_cost_tk: float = ENTRY_COST_TICKS,
    cancel_evals: int = int(CANCEL_WINDOW_SECONDS * 1000 / STRIDE),  # 40 evals @ 250ms stride
    cooldown_evals: int = int(REENTRY_COOLDOWN_SECONDS * 1000 / STRIDE),  # 20 evals
    cold_start_evals_per_day: int = int(COLD_START_MINUTES * 60_000 / STRIDE),
    per_day_pct: bool = True,
) -> Dict[str, Any]:
    """
    Pure-function simulation of the v2_1s_short_top05 strategy over a prediction stream.

    Inputs
    ------
    pred_log_ret_1s : (N,)  predicted 1s log-return (signed; negative = short signal)
    mid_ticks       : (N,)  mid price IN TICKS at each prediction index
    ts_ns           : (N,)  optional ns timestamps (used only for diagnostics)
    dates           : (N,)  optional per-prediction date key (str or int). Used to reset
                            per-day percentile + cold-start counter at day boundaries.

    Output dict carries: net_tk_per_fill, n_fills, wr, n_tp1, n_tp2, n_sl, n_time_stop,
    day_conc, ci95_lo_tk (bootstrap), per_day_pass_rate, fills (list of dicts).

    Logic matches the live engine: SHORT only, passive_at_touch (cost = 0.376 tk already
    baked into the bracket constants), enter at next-bar mid (proxy for fill at touch),
    exit on first of {TP2 hit, SL hit, time-stop @ cancel_evals after entry}.
    Half-size TP1 is recorded as a partial event but for 1-contract sizing we use the
    closer of {TP1, TP2} as effective exit price. Default: use TP2 (matches backtest).
    """
    N = int(len(pred_log_ret_1s))
    assert len(mid_ticks) == N
    pred = np.asarray(pred_log_ret_1s, dtype=np.float64)
    mid = np.asarray(mid_ticks, dtype=np.float64)
    if dates is None:
        dates = np.zeros(N, dtype=np.int64)
    else:
        dates = np.asarray(dates)

    fills: List[Dict[str, Any]] = []
    state = _SimState()

    # Per-day percentile state
    cur_day = None
    short_buf: List[float] = []
    cold_start_idx0 = 0
    last_exit_idx = -10**9

    for i in range(N):
        d = dates[i]
        if d != cur_day:
            cur_day = d
            short_buf = []
            cold_start_idx0 = i  # reset cold-start anchor at day boundary
            # If a position is still open across day boundaries, force flat (won't happen in RTH-only stream)
            if state.pos != 0:
                # Force flat at last available mid
                exit_px = mid[i - 1] if i > 0 else state.entry_px
                pnl_tk = (state.entry_px - exit_px) if state.pos < 0 else (exit_px - state.entry_px)
                net_tk = pnl_tk - entry_cost_tk
                fills.append({"date": int(d) if isinstance(d, (np.integer, int)) else str(d),
                              "kind": "day_flat", "net_tk": net_tk, "entry_idx": state.entry_idx,
                              "exit_idx": i - 1, "pred_1s": pred[state.entry_idx]})
                state = _SimState()

        # Confidence gate
        p = pred[i]
        if p < 0:
            short_buf.append(-p)

        passes_global = p <= global_floor
        passes_pct = True
        if per_day_pct:
            in_cold = (i - cold_start_idx0) < cold_start_evals_per_day
            if in_cold or len(short_buf) < 200:
                passes_pct = True
            else:
                thr = float(np.percentile(np.asarray(short_buf), percentile))
                passes_pct = (-p) >= thr

        signal_short = (state.pos == 0) and passes_global and passes_pct

        if state.pos == 0:
            # Cooldown check
            if (i - last_exit_idx) < cooldown_evals:
                continue
            if signal_short:
                # Enter SHORT at this mid (touch). Backtest assumption: passive limit at best ask fills here.
                state.pos = -1
                state.entry_px = mid[i]
                state.entry_idx = i
                state.entry_ts = float(ts_ns[i]) if ts_ns is not None else 0.0
                continue

        # In position — check bracket
        if state.pos != 0:
            mfe = state.entry_px - mid[i]  # in ticks; favorable for SHORT
            mae = mid[i] - state.entry_px  # adverse for SHORT
            kind: Optional[str] = None
            exit_px: float = mid[i]
            if mae >= sl_tk:
                kind = "sl"
                exit_px = state.entry_px + sl_tk
            elif mfe >= tp2_tk:
                kind = "tp2"
                exit_px = state.entry_px - tp2_tk
            elif (i - state.entry_idx) >= cancel_evals:
                kind = "time_stop"
                exit_px = mid[i]

            if kind is not None:
                pnl_tk = (state.entry_px - exit_px)  # SHORT
                net_tk = pnl_tk - entry_cost_tk
                fills.append({
                    "date": int(d) if isinstance(d, (np.integer, int)) else str(d),
                    "kind": kind, "net_tk": net_tk,
                    "entry_idx": state.entry_idx, "exit_idx": i,
                    "entry_px_tk": state.entry_px, "exit_px_tk": exit_px,
                    "hold_evals": i - state.entry_idx,
                    "pred_1s": pred[state.entry_idx],
                })
                state = _SimState()
                last_exit_idx = i

    # Aggregate metrics
    nets = np.array([f["net_tk"] for f in fills], dtype=np.float64)
    n = len(fills)
    if n == 0:
        return {"n_fills": 0, "net_tk_per_fill": 0.0, "wr": 0.0,
                "n_tp1": 0, "n_tp2": 0, "n_sl": 0, "n_time_stop": 0,
                "day_conc": 0.0, "ci95_lo_tk": 0.0, "per_day_pass_rate": 0.0,
                "fills": fills}
    by_kind = {k: sum(1 for f in fills if f["kind"] == k) for k in ("tp1", "tp2", "sl", "time_stop", "day_flat")}
    # Day concentration: fraction of fills on the most-active day
    per_day_n = {}
    for f in fills:
        per_day_n[f["date"]] = per_day_n.get(f["date"], 0) + 1
    day_conc = max(per_day_n.values()) / float(n) if n else 0.0
    # Per-day pass rate (rule 2): fraction of active days with net > 0
    per_day_net = {}
    for f in fills:
        per_day_net[f["date"]] = per_day_net.get(f["date"], 0.0) + f["net_tk"]
    n_active = len(per_day_net)
    n_pos_days = sum(1 for v in per_day_net.values() if v > 0)
    per_day_pass = n_pos_days / float(n_active) if n_active else 0.0
    # Bootstrap CI95 lower bound
    if n >= 30:
        rng = np.random.default_rng(20260518)
        boot = np.array([rng.choice(nets, size=n, replace=True).mean() for _ in range(2000)])
        ci95_lo = float(np.percentile(boot, 2.5))
    else:
        ci95_lo = float(nets.mean() - 1.96 * nets.std(ddof=1) / max(1.0, math.sqrt(n))) if n > 1 else 0.0

    return {
        "n_fills": n,
        "net_tk_per_fill": float(nets.mean()),
        "wr": float((nets > 0).mean()),
        "n_tp1": by_kind.get("tp1", 0),
        "n_tp2": by_kind.get("tp2", 0),
        "n_sl": by_kind.get("sl", 0),
        "n_time_stop": by_kind.get("time_stop", 0),
        "n_day_flat": by_kind.get("day_flat", 0),
        "day_conc": day_conc,
        "ci95_lo_tk": ci95_lo,
        "per_day_pass_rate": per_day_pass,
        "n_active_days": n_active,
        "fills": fills,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE POSITION / BRACKET STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class State:
    IDLE          = "IDLE"
    ENTRY_PENDING = "ENTRY_PENDING"
    IN_POSITION   = "IN_POSITION"
    EXIT_PENDING  = "EXIT_PENDING"


@dataclass
class LivePosition:
    state: str = State.IDLE
    side: int = 0          # -1 short, +1 long (we only do -1), 0 flat
    entry_px: float = 0.0
    entry_ts: float = 0.0
    entry_order_id: Optional[str] = None
    entry_submit_ts: float = 0.0
    exit_order_id: Optional[str] = None
    sl_trigger_ts: Optional[float] = None
    mfe_tk: float = 0.0
    mae_tk: float = 0.0
    qty: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAPER TRADER
# ═══════════════════════════════════════════════════════════════════════════════

class V2_1S_Short_Top05_PaperTrader:
    """
    Main live-trading engine. Two run modes:
      - replay(npz_path):   reads MBO events from saved NPZ, drives engine deterministically
      - live():             subscribes to Rithmic paper account live MBO stream

    In BOTH modes:
      * Same StreamingFeaturesSmartV3 feature pipeline
      * Same model inference
      * Same confidence gate
      * Same bracket / cancel / cooldown logic
      * Same kill-switches
      * Discord alerts wired
    """

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS,
        stats_path: Path = DEFAULT_STATS,
        symbol: str = "ESM6",
        device: str = "cuda",
        webhook_url: Optional[str] = None,
        canonical_sha256: Optional[str] = CANONICAL_CKPT_SHA256,
        shadow_mode: bool = False,
    ):
        self.weights_path = Path(weights_path)
        self.stats_path = Path(stats_path)
        self.symbol = symbol
        self.device = device
        self.shadow_mode = shadow_mode

        self.notifier = DiscordNotifier(webhook_url)
        self.killer = KillSwitchManager(self.notifier)
        self.pct = PerDayPercentileTracker()
        self.pos = LivePosition()

        # SHA-256 integrity (kill-switch #9)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        sha = file_sha256(self.weights_path)
        log.info("Weights SHA-256: %s", sha)
        if canonical_sha256 is not None and sha != canonical_sha256:
            raise SystemExit(
                f"REFUSE START: weights SHA-256 mismatch.\n  expected: {canonical_sha256}\n  got:      {sha}"
            )

        # Lazy-load heavy deps so import-only smoke test doesn't require CUDA
        self._engine = None
        self._streamer = None
        self._n_features = None

        self.last_heartbeat_ts = 0.0
        self.session_start_ts = time.time()
        self.fills_today = 0
        self.pnl_today_tk = 0.0

        # cooldown
        self.last_exit_ts: float = 0.0

        # Counters for shadow / live diagnostics
        self.n_signals_total = 0
        self.n_signals_passed_gate = 0

    # --- lazy init ------------------------------------------------------------

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        # Local import to avoid loading torch at script-import time
        from cnn_mamba_v2_inference import CNNMambaV2Inference
        from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
        self._engine = CNNMambaV2Inference(
            weights_path=str(self.weights_path),
            stats_path=str(self.stats_path),
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            device=self.device,
        )
        self._streamer = StreamingFeaturesSmartV3()
        self._n_features = N_FEATURES
        log.info("Engine loaded. features=%d window=%d stride=%d device=%s",
                 N_FEATURES, WINDOW_SIZE, STRIDE, self.device)
        self.notifier.send(f"v2_1s_short_top05 engine loaded. shadow={self.shadow_mode} symbol={self.symbol}",
                           level="info")

    # --- core decision step (called on every prediction) ----------------------

    def on_prediction(self, pred_dict: Dict[str, Any], best_bid: float, best_ask: float,
                      mid_price: float, event_ts: float) -> None:
        """Called every STRIDE events with a fresh model prediction.

        pred_dict: from CNNMambaV2Inference.predict / add_event
          {"pred_1s": ..., "pred_5s": ..., "pred_10s": ..., "direction": ±1, "tier": ...}
        best_bid/ask/mid: in TICKS or DOLLARS — we work in dollars here; convert to ticks on exit math.
        event_ts: unix-ish timestamp of the originating MBO event (seconds).
        """
        self.n_signals_total += 1
        pred_1s = float(pred_dict["pred_1s"])
        self.killer.on_prediction(pred_1s, target_1s=None)
        self.killer.evaluate()

        t_et = now_et()
        if not is_rth(t_et):
            return  # outside RTH — no entries
        if self.killer.is_halted():
            return

        # Confidence gate (BOTH global floor AND per-day percentile when ready)
        self.pct.push(pred_1s, t_et)
        if self.pos.state != State.IDLE:
            # Manage in-flight position (cancel / bracket / time-stop)
            self._manage_position(best_bid, best_ask, mid_price, event_ts)
            return

        # Cooldown
        if (time.time() - self.last_exit_ts) < REENTRY_COOLDOWN_SECONDS:
            return

        if not self.pct.passes(pred_1s, t_et):
            return
        self.n_signals_passed_gate += 1

        # Enter SHORT at passive_at_touch (LIMIT at best_ask).
        # For paper engine: same plumbing, paper account guarantees fill if touch occurs.
        self._submit_entry_limit_short(best_ask=best_ask, mid_price=mid_price, event_ts=event_ts,
                                       pred_1s=pred_1s, t_et=t_et)

    def _submit_entry_limit_short(self, *, best_ask: float, mid_price: float, event_ts: float,
                                  pred_1s: float, t_et: datetime) -> None:
        order_id = f"E-{uuid.uuid4().hex[:10]}"
        self.pos = LivePosition(
            state=State.ENTRY_PENDING,
            side=-1,
            entry_px=best_ask,   # provisional; updated on fill confirmation
            entry_ts=event_ts,
            entry_order_id=order_id,
            entry_submit_ts=time.time(),
            qty=POSITION_SIZE,
        )
        jlog("entry_submit", order_id=order_id, side="SHORT", limit_px=best_ask,
             mid=mid_price, pred_1s=pred_1s, t_et=t_et.isoformat(), shadow=self.shadow_mode)
        log.info("ENTRY SUBMIT SHORT @ %.4f (mid=%.4f, pred=%.4f) [%s%s]",
                 best_ask, mid_price, pred_1s, "SHADOW " if self.shadow_mode else "", order_id)
        if not self.shadow_mode:
            self._broker_submit_limit(side="SHORT", price=best_ask, qty=POSITION_SIZE, order_id=order_id)

    def _manage_position(self, best_bid: float, best_ask: float, mid_price: float, event_ts: float) -> None:
        # Update MFE/MAE
        if self.pos.state == State.IN_POSITION and self.pos.side == -1:
            mfe = (self.pos.entry_px - mid_price) / TICK_SIZE
            mae = (mid_price - self.pos.entry_px) / TICK_SIZE
            self.pos.mfe_tk = max(self.pos.mfe_tk, mfe)
            self.pos.mae_tk = max(self.pos.mae_tk, mae)

        # Cancel unfilled entry limit after 10s
        if self.pos.state == State.ENTRY_PENDING:
            age = time.time() - self.pos.entry_submit_ts
            if age > CANCEL_WINDOW_SECONDS:
                self._cancel_entry_and_reset(reason="cancel_10s")
                return

        # In position — check bracket
        if self.pos.state == State.IN_POSITION and self.pos.side == -1:
            mfe_now = (self.pos.entry_px - mid_price) / TICK_SIZE
            mae_now = (mid_price - self.pos.entry_px) / TICK_SIZE
            if mae_now >= SL_TICKS:
                self._submit_exit_market(reason="sl_hit", target_px=self.pos.entry_px + SL_TICKS * TICK_SIZE)
            elif mfe_now >= TP2_TICKS:
                # Use TP2 limit at best_bid (passive limit on exit to avoid recrossing)
                self._submit_exit_limit_short_cover(target_px=self.pos.entry_px - TP2_TICKS * TICK_SIZE,
                                                   reason="tp2_hit", best_bid=best_bid)
            elif (event_ts - self.pos.entry_ts) > (CANCEL_WINDOW_SECONDS * 4):
                # Time-stop fallback (40 evals × 4 = ~40s); spec lists 0.3% rate
                self._submit_exit_market(reason="time_stop", target_px=mid_price)

        # SL fill confirmation timeout
        if (self.pos.state == State.EXIT_PENDING and self.pos.sl_trigger_ts is not None
                and (time.time() - self.pos.sl_trigger_ts) > KS_PER_TRADE_SL_OVERRUN_SECONDS):
            log.error("Per-trade max risk: SL not filled in %.1fs — forcing MARKET exit",
                      KS_PER_TRADE_SL_OVERRUN_SECONDS)
            self.notifier.send("Per-trade max risk: forcing market exit", level="kill")
            self._broker_submit_market(side="BUY_TO_COVER", qty=self.pos.qty,
                                       order_id=f"X-{uuid.uuid4().hex[:8]}")
            # Treat as time_stop exit at current mid
            self._on_exit_filled(filled_px=mid_price, reason="forced_market_exit")

    # --- broker stubs (paper engine plugs in here) ----------------------------

    def _broker_submit_limit(self, *, side: str, price: float, qty: int, order_id: str) -> None:
        """Submit LIMIT order to paper broker. Plugin point — wire to rithmic_client paper account."""
        # PAPER mode: emit log; the live broker glue (run_loop_live) will route to RithmicClient.
        jlog("broker_submit_limit", side=side, price=price, qty=qty, order_id=order_id)

    def _broker_submit_market(self, *, side: str, qty: int, order_id: str) -> None:
        jlog("broker_submit_market", side=side, qty=qty, order_id=order_id)

    def _broker_cancel(self, order_id: str) -> None:
        jlog("broker_cancel", order_id=order_id)

    # --- order lifecycle ------------------------------------------------------

    def on_entry_fill(self, fill_px: float, fill_ts: float) -> None:
        """Called by broker callback when ENTRY_PENDING limit fills."""
        self.pos.entry_px = fill_px
        self.pos.entry_ts = fill_ts
        self.pos.state = State.IN_POSITION
        jlog("entry_filled", price=fill_px, order_id=self.pos.entry_order_id)
        self.notifier.send(f"ENTRY FILLED SHORT @ {fill_px:.4f}", level="fill")

    def _cancel_entry_and_reset(self, reason: str) -> None:
        oid = self.pos.entry_order_id
        if not self.shadow_mode and oid:
            self._broker_cancel(oid)
        jlog("entry_cancelled", reason=reason, order_id=oid)
        log.info("ENTRY CANCEL (%s) order=%s", reason, oid)
        self.pos = LivePosition()
        self.last_exit_ts = time.time()  # reuse cooldown

    def _submit_exit_limit_short_cover(self, *, target_px: float, reason: str, best_bid: float) -> None:
        oid = f"X-{uuid.uuid4().hex[:10]}"
        self.pos.exit_order_id = oid
        self.pos.state = State.EXIT_PENDING
        # Passive cover: BUY LIMIT @ target (or at best_bid, whichever is more aggressive but still passive)
        limit_px = max(target_px, best_bid)
        jlog("exit_submit_limit", reason=reason, price=limit_px, target=target_px, order_id=oid)
        log.info("EXIT SUBMIT (%s) BUY-COVER LIMIT @ %.4f (target %.4f)", reason, limit_px, target_px)
        if not self.shadow_mode:
            self._broker_submit_limit(side="BUY_TO_COVER", price=limit_px, qty=self.pos.qty, order_id=oid)

    def _submit_exit_market(self, *, reason: str, target_px: float) -> None:
        oid = f"X-{uuid.uuid4().hex[:10]}"
        self.pos.exit_order_id = oid
        self.pos.state = State.EXIT_PENDING
        self.pos.sl_trigger_ts = time.time() if reason == "sl_hit" else None
        jlog("exit_submit_market", reason=reason, target_px=target_px, order_id=oid)
        log.info("EXIT SUBMIT (%s) MARKET (target %.4f)", reason, target_px)
        if not self.shadow_mode:
            self._broker_submit_market(side="BUY_TO_COVER", qty=self.pos.qty, order_id=oid)
        # In shadow we synthesize fill at target_px immediately
        if self.shadow_mode:
            self._on_exit_filled(filled_px=target_px, reason=reason)

    def _on_exit_filled(self, filled_px: float, reason: str) -> None:
        if self.pos.state not in (State.IN_POSITION, State.EXIT_PENDING):
            return
        entry_px = self.pos.entry_px
        # SHORT P&L
        pnl_dollars = (entry_px - filled_px) * POINT_VALUE * self.pos.qty
        pnl_tk = (entry_px - filled_px) / TICK_SIZE
        net_tk = pnl_tk - ENTRY_COST_TICKS
        net_dollars = pnl_dollars - (COMMISSION_PER_SIDE * 2 * self.pos.qty)
        hold_s = time.time() - self.pos.entry_ts if self.pos.entry_ts else 0.0
        kind = {"tp2_hit": "tp2", "sl_hit": "sl", "time_stop": "time_stop",
                "forced_market_exit": "forced_market_exit"}.get(reason, reason)

        self.killer.on_fill(kind=kind, net_ticks=net_tk)
        self.fills_today += 1
        self.pnl_today_tk += net_tk

        rec = {
            "event": "fill_done", "reason": reason, "entry_px": entry_px, "exit_px": filled_px,
            "pnl_tk": pnl_tk, "net_tk": net_tk, "pnl_$": pnl_dollars, "net_$": net_dollars,
            "mfe_tk": self.pos.mfe_tk, "mae_tk": self.pos.mae_tk, "hold_s": hold_s,
            "entry_order": self.pos.entry_order_id, "exit_order": self.pos.exit_order_id,
        }
        jlog("trade_done", **rec)
        log.info(
            "EXIT FILLED (%s) @ %.4f | net=%+.2ftk ($%+.2f) | MFE %.1ftk MAE %.1ftk | hold=%.1fs | day_net=%+.2ftk (%d fills)",
            reason, filled_px, net_tk, net_dollars, self.pos.mfe_tk, self.pos.mae_tk, hold_s,
            self.pnl_today_tk, self.fills_today,
        )
        self.notifier.send(
            f"{kind.upper()} fill: net={net_tk:+.2f} tk (${net_dollars:+.2f}) | "
            f"day={self.pnl_today_tk:+.2f} tk / {self.fills_today} fills",
            level="fill",
        )
        self.pos = LivePosition()
        self.last_exit_ts = time.time()

    # --- heartbeat ------------------------------------------------------------

    def maybe_write_heartbeat(self, force: bool = False) -> None:
        if not force and (time.time() - self.last_heartbeat_ts) < 60.0:
            return
        self.last_heartbeat_ts = time.time()
        hb = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "state": self.pos.state,
            "side": self.pos.side,
            "fills_today": self.fills_today,
            "day_net_tk": self.pnl_today_tk,
            "n_signals_total": self.n_signals_total,
            "n_signals_passed_gate": self.n_signals_passed_gate,
            "pct_obs_today": self.pct.n_observations,
            "pct_threshold": self.pct.threshold(),
            "killer": self.killer.status(),
            "shadow_mode": self.shadow_mode,
            "symbol": self.symbol,
            "session_uptime_s": time.time() - self.session_start_ts,
        }
        try:
            HEARTBEAT_PATH.write_text(json.dumps(hb, indent=2, default=str))
        except Exception as e:  # pragma: no cover
            log.error("heartbeat write failed: %s", e)

    # --- run loops ------------------------------------------------------------

    def run_replay(self, npz_path: Path) -> Dict[str, Any]:
        """Drive the engine deterministically from a saved MBO events NPZ.

        Used for Step D shadow-mode validation and ad-hoc deterministic re-runs.
        """
        self._ensure_engine()
        d = np.load(str(npz_path), allow_pickle=True)
        log.info("Replay NPZ: %s keys=%s", npz_path, list(d.keys())[:10])
        # Expect: events (N×F) or per-event arrays; broker fills synthesized at best_ask/bid.
        # Conform to whichever schema the recorder produces — this is a stub.
        # In live integration, replace with the actual streaming_features_smart_v3 driver.
        raise NotImplementedError(
            "run_replay: wire to your specific MBO NPZ schema. "
            "Use simulate_trades() for prediction-stream replay (Step B reproduce script)."
        )

    async def run_live(self, events_path: Optional[str] = None) -> None:
        """Drive engine from the MBO recorder's `live_events.jsonl` file.

        Why tail-mode instead of a dedicated Rithmic subscription?
          The Razer host already runs ONE Rithmic session (PID 25512 = the existing
          paper_trading_mamba_v2). Opening a second concurrent session can clash
          (Rithmic rejects duplicate logins). The MBO recorder writes `live_events.jsonl`
          independently and is the canonical event stream for all live tooling
          (see paper_trading_mamba_v2.run_follow). Until cutover (Step E), we tail
          that file. In shadow mode we synthesize fills locally — never touching
          the real broker. The on_prediction() decision logic is IDENTICAL to live;
          only the broker-submit calls are short-circuited when shadow_mode=True.

        Schema of each JSONL line (per recorder):
            { timestamp_ns, side(0=bid|1=ask), action(0|1|2|3),
              price_ticks (relative to mid in 0.25-pt units),
              size, order_id, bbo: { bid_price, ask_price, bid_size, ask_size } }
        """
        import asyncio
        import math

        self._ensure_engine()

        # Default path = canonical recorder location on Razer
        if events_path is None:
            if sys.platform == "win32":
                events_path = r"C:\Users\claude\Lvl3Quant\live_trading\logs\live_events.jsonl"
            else:
                events_path = "/home/jupiter/Lvl3Quant/live_trading_linux/logs/live_events.jsonl"

        log.info("LIVE START symbol=%s shadow=%s events_path=%s",
                 self.symbol, self.shadow_mode, events_path)
        self.notifier.send(
            f"v2_1s_short_top05 LIVE START — symbol={self.symbol} shadow={self.shadow_mode}",
            level="info",
        )

        # Wait for events file to exist (recorder may not be up yet)
        for _ in range(60):
            if Path(events_path).exists():
                break
            log.info("  waiting for events file %s ...", events_path)
            await asyncio.sleep(2.0)
        if not Path(events_path).exists():
            raise FileNotFoundError(f"events file never appeared: {events_path}")

        # Book + encoder state
        best_bid = 0.0
        best_ask = 0.0
        mid_price = 0.0
        prev_ts_ns = 0
        n_events = 0
        n_preds = 0

        async def heartbeat_loop():
            while True:
                self.maybe_write_heartbeat()
                # HC #438 FIX: refresh broker-heartbeat BEFORE evaluate() so the
                # heartbeat_loop's own 30s tick satisfies KS_CONNECTIVITY_LOSS_SECONDS.
                # Previous order ran evaluate() against a stale last_broker_seen_ts and
                # could fire broker_connectivity_loss self-halt at RTH close.
                self.killer.on_broker_heartbeat()  # tail-mode treats recorder as broker
                self.killer.evaluate()
                await asyncio.sleep(30.0)

        hb_task = asyncio.create_task(heartbeat_loop())

        try:
            with open(events_path, "r") as fh:
                fh.seek(0, 2)  # tail from EOF
                log.info("LIVE: positioned at EOF; waiting for new events...")

                while True:
                    line = fh.readline()
                    if not line:
                        await asyncio.sleep(0.05)
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # In tail-mode we treat the recorder as the "broker" — every
                    # event we successfully decode is proof the upstream pipe is alive.
                    # This keeps KS_BROKER_CONNECTIVITY (5s timeout) satisfied at the
                    # event tick-rate (~100/s during RTH) instead of the 30s heartbeat.
                    self.killer.on_broker_heartbeat()

                    ts_ns = int(ev.get("timestamp_ns", 0))

                    # ── Book update ──
                    bbo = ev.get("bbo") or {}
                    if bbo:
                        b = float(bbo.get("bid_price", 0) or 0)
                        a = float(bbo.get("ask_price", 0) or 0)
                        if b > 0:
                            best_bid = b
                        if a > 0:
                            best_ask = a
                        if best_bid > 0 and best_ask > 0:
                            mid_price = 0.5 * (best_bid + best_ask)

                    # ── Encode 6-tuple ──
                    # HC #423 §3 ENCODER FIX (2026-05-19): three corrections
                    # documented in output/hc423_data_format_alignment_audit.md.
                    # Without these, live feat_vec diverges from training by
                    # 200-650σ on dims 0/3/8 → +0.30 pred bias → -$1,286 paper
                    # P&L overnight 2026-05-18→19.
                    action = int(ev.get("action", 0) or 0)
                    side_raw = int(ev.get("side", 0) or 0)
                    price_ticks = float(ev.get("price_ticks", 0) or 0)
                    size = int(ev.get("size", 1) or 1)

                    # FIX (b): Use raw action (0=Add, 1=Cancel, 2=Modify, 3=Trade,
                    # 4=Fill) directly as event_type_id. Old code collapsed to
                    # {0, 3} which broke dim 6 (cancel_side_asym → 100% zero) and
                    # dim 12 (event_type_entropy collapsed 0.80→0.16).
                    # NOTE: recorder upstream currently still drops cancels/modifies/
                    # fills (HC #423 §3 rec #3) — this fix becomes fully effective
                    # only once the recorder is patched to emit all 5 actions.
                    etype = action
                    # FIX (a): price_ticks from the recorder is ABSOLUTE price in
                    # ticks (audit observed mean=21,389 = $5,347). Training
                    # pipeline used price-relative-to-mid clipped to [−50,+50].
                    # Compute mid_ticks from bbo and subtract.
                    if best_bid > 0 and best_ask > 0:
                        mid_ticks = (best_bid + best_ask) / (2.0 * TICK_SIZE)
                        price_rel_ticks = float(price_ticks) - mid_ticks
                        # clip to training range
                        if price_rel_ticks > 50.0:
                            price_rel_ticks = 50.0
                        elif price_rel_ticks < -50.0:
                            price_rel_ticks = -50.0
                    else:
                        # No book yet — emit 0 (matches training warmup region)
                        price_rel_ticks = 0.0
                    spread_ticks = ((best_ask - best_bid) / TICK_SIZE
                                    if best_bid > 0 and best_ask > 0 else 0.0)
                    # FIX (c): training pipeline used log1p(delta_seconds), not
                    # log1p(delta_microseconds). Audit confirms training raw
                    # mean=0.0005, 99.95% zero → seconds unit (microseconds
                    # would have non-zero on every event since ticks are µs-spaced).
                    delta_s = max(0.0, (ts_ns - prev_ts_ns) / 1e9) if prev_ts_ns else 0.0
                    prev_ts_ns = ts_ns
                    time_delta_log = math.log1p(delta_s) if delta_s > 0 else 0.0
                    qty_log = math.log(max(1, size))

                    # ── Feature engine + inference ──
                    feat_vec = self._streamer.update(
                        time_delta_log, etype, side_raw,
                        price_rel_ticks, qty_log, spread_ticks,
                    )
                    n_events += 1
                    pred = self._engine.add_event(np.asarray(feat_vec, dtype=np.float32))
                    if pred is None:
                        continue
                    n_preds += 1

                    # ── Strategy decision (RTH + per-day pct + kill-switches in on_prediction) ──
                    self.on_prediction(
                        pred_dict=pred,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        mid_price=mid_price,
                        event_ts=ts_ns / 1e9 if ts_ns else time.time(),
                    )

                    # In shadow mode, synthesize entry fill on next mid touch
                    # (passive limit at best_ask gets touched whenever ask trades).
                    # We model this as: if state==ENTRY_PENDING and best_ask<=entry_px → fill.
                    if (self.shadow_mode
                            and self.pos.state == State.ENTRY_PENDING
                            and best_ask > 0
                            and self.pos.entry_px > 0
                            and best_ask <= self.pos.entry_px):
                        self.on_entry_fill(fill_px=self.pos.entry_px,
                                           fill_ts=ts_ns / 1e9 if ts_ns else time.time())

                    if n_preds % 100 == 0:
                        log.info("LIVE tick %d events / %d preds | mid=%.2f bid=%.2f ask=%.2f | "
                                 "passed_gate=%d fills_today=%d day_net=%+.2f tk",
                                 n_events, n_preds, mid_price, best_bid, best_ask,
                                 self.n_signals_passed_gate, self.fills_today, self.pnl_today_tk)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("LIVE: shutdown requested")
        finally:
            hb_task.cancel()
            self.maybe_write_heartbeat(force=True)
            self.notifier.send(
                f"v2_1s_short_top05 LIVE STOP — fills_today={self.fills_today} "
                f"day_net={self.pnl_today_tk:+.2f}tk shadow={self.shadow_mode}",
                level="info",
            )

    async def _run_live_unreachable(self) -> None:
        # Kept for IDE introspection; never called.
        raise NotImplementedError(
            "run_live: not used — tail-mode run_live is wired above."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2_1s_short_top05 paper trader (spec-compliant)")
    p.add_argument("--symbol", default="ESM6", help="Futures symbol (default ESM6)")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Model weights path")
    p.add_argument("--stats", default=str(DEFAULT_STATS), help="Feature stats NPZ path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--shadow", action="store_true",
                   help="Shadow mode: log signals + simulated fills, NO real orders")
    p.add_argument("--replay", type=str, default=None, help="Replay MBO NPZ (deterministic test)")
    p.add_argument("--webhook", default=os.environ.get("DISCORD_WEBHOOK_URL"),
                   help="Discord webhook URL (default $DISCORD_WEBHOOK_URL)")
    p.add_argument("--sha256", default=None, help="Canonical weights SHA-256 (refuse start on mismatch)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Import + spec self-check only; no engine load")
    return p.parse_args()


def _smoke_test() -> int:
    """Importable + constants sane. No CUDA / model load."""
    # Spec constants must match deploy spec exactly
    assert abs(TP1_TICKS - 0.4782) < 1e-6, "TP1 mismatch"
    assert abs(TP2_TICKS - 0.9564) < 1e-6, "TP2 mismatch"
    assert abs(SL_TICKS - 0.5686) < 1e-6, "SL mismatch"
    assert abs(GLOBAL_CONFIDENCE_FLOOR_1S + 0.6926) < 1e-6, "global floor mismatch"
    assert abs(POINT_VALUE - 12.50) < 1e-6, "POINT_VALUE mismatch — must be ES $12.50"
    assert abs(ENTRY_COST_TICKS - 0.376) < 1e-6, "entry cost mismatch"
    assert PERCENTILE_RANK_THRESHOLD == 99.5
    assert COLD_START_MINUTES == 30
    assert REENTRY_COOLDOWN_SECONDS == 5.0
    assert CANCEL_WINDOW_SECONDS == 10.0
    assert POSITION_SIZE == 1
    # Synthetic simulate_trades sanity: monotone short signal triggers fills
    rng = np.random.default_rng(0)
    N = 5000
    pred = rng.normal(0, 0.3, size=N)
    pred[1000:1010] = -1.5   # strong short signal block
    pred[2000:2010] = -1.5
    mids = np.cumsum(rng.normal(0, 0.1, size=N)) + 5000.0
    out = simulate_trades(pred, mids / TICK_SIZE, ts_ns=None, dates=np.zeros(N, dtype=np.int64),
                          cold_start_evals_per_day=0, per_day_pct=False)
    print(f"smoke-test simulate_trades: n_fills={out['n_fills']} "
          f"net/fill={out['net_tk_per_fill']:+.3f} tk wr={out['wr']:.1%}")
    print("smoke-test PASS — all 11 spec-constant assertions held.")
    return 0


def _main() -> int:
    args = _parse_args()
    if args.smoke_test:
        return _smoke_test()
    trader = V2_1S_Short_Top05_PaperTrader(
        weights_path=Path(args.weights),
        stats_path=Path(args.stats),
        symbol=args.symbol,
        device=args.device,
        webhook_url=args.webhook,
        canonical_sha256=args.sha256,
        shadow_mode=args.shadow,
    )
    if args.replay:
        trader.run_replay(Path(args.replay))
        return 0
    try:
        asyncio.run(trader.run_live())
    except KeyboardInterrupt:
        log.info("Shutdown by user.")
        trader.maybe_write_heartbeat(force=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
