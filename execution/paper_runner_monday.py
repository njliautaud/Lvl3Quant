#!/usr/bin/env python3
"""
paper_runner_monday.py — Multi-strategy paper trading runner for CNN-Mamba v2.

Runs 5 backtested execution strategies simultaneously on a shared signal stream.
Each strategy has independent position management, P&L tracking, and exit logic.

Two modes:
  --replay FILE   : replay saved MBO .npz data at configurable speed
  --live          : read from MBO recorder output (Rithmic feed)

Signal pipeline:
  raw MBO events -> precomputed smart_v3 features (25-dim)
                 -> sliding window of 1000 events
                 -> model inference every 500 events
                 -> [1s, 5s, 10s] predictions
                 -> z-score thresholding per strategy

Usage:
    # Replay Feb 25 at 500x speed
    python3 -m execution.paper_runner_monday \\
        --replay data/processed/mbo_events_smart_v3/20260225_mbo_events.npz \\
        --speed 500

    # Live mode (Monday market open)
    python3 -m execution.paper_runner_monday --live

    # Use specific model weights
    python3 -m execution.paper_runner_monday \\
        --replay data/processed/mbo_events_smart_v3/20260225_mbo_events.npz \\
        --weights output/cnn_mamba_v2/best.pt \\
        --stats output/cnn_mamba_v2/feature_stats.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal as _sig
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# ── Path setup ──
LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

# ── Logging ──
LOG_DIR = LVL3 / "execution" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("paper_runner")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "paper_runner_monday.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(sh)


# ============================================================
# Constants
# ============================================================
ES_TICK_SIZE = 0.25
ES_POINT_VALUE = 50.0       # $50 per point for ES
COMMISSION_PER_SIDE = 0.50   # AMP rate
CHASE_TICKS = 1              # chase 1 tick beyond BBO for fills
DEFAULT_SLIPPAGE_TICKS = 0.5 # half tick average slippage on chase entry
COOLDOWN_MS = 5000           # 5s cooldown between trades per strategy
WARMUP_EVENTS = 5000         # minimum events before trading

# Default model paths (Mamba v7 as fallback)
DEFAULT_WEIGHTS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_best.pt"
DEFAULT_STATS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_feature_stats.npz"
DEFAULT_PREDS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/concat_oot_predictions.npz"


# ============================================================
# Strategy Definitions
# ============================================================
STRATEGIES = [
    {
        "name": "midday_z2.5_30s",
        "z_threshold": 2.5,
        "hold_ms": 30000,
        "time_filter": (11, 0, 14, 0),  # 11:00-14:00 ET only
        "entry_mode": "chase",
        "exit_mode": "time",
        "bracket": None,
    },
    {
        "name": "open30_z2.5_30s",
        "z_threshold": 2.5,
        "hold_ms": 30000,
        "time_filter": (9, 30, 10, 0),  # first 30 min
        "entry_mode": "chase",
        "exit_mode": "time",
        "bracket": None,
    },
    {
        "name": "bracket_wide_z5.0",
        "z_threshold": 5.0,
        "hold_ms": 60000,
        "time_filter": None,  # all day
        "entry_mode": "chase",
        "exit_mode": "bracket",
        "bracket": {"sl_ticks": 4, "tp_ticks": 8},
    },
    {
        "name": "bracket_balanced_z5.0",
        "z_threshold": 5.0,
        "hold_ms": 60000,
        "time_filter": None,
        "entry_mode": "chase",
        "exit_mode": "bracket",
        "bracket": {"sl_ticks": 3, "tp_ticks": 4},
    },
    {
        "name": "baseline_z2.5_30s",
        "z_threshold": 2.5,
        "hold_ms": 30000,
        "time_filter": None,
        "entry_mode": "chase",
        "exit_mode": "time",
        "bracket": None,
    },
]


# ============================================================
# Model Abstraction
# ============================================================
class ModelInference:
    """Abstract model inference. Supports Mamba v7 and CNN-Mamba v2."""

    def __init__(self, weights_path: str, stats_path: str,
                 preds_path: Optional[str] = None,
                 window_size: int = 1000, stride: int = 500):
        import torch

        self.window_size = window_size
        self.stride = stride
        self.device = torch.device("cpu")
        self._prediction_count = 0

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        arch = ckpt.get("arch", {})
        model_type = arch.get("model_type", "mamba_v7")

        log.info("Loading model from %s", weights_path)
        log.info("  Type: %s, d_model=%s, n_layers=%s",
                 model_type, arch.get("d_model"), arch.get("n_layers"))

        if "cnn_mamba" in model_type.lower():
            self.model = self._build_cnn_mamba(ckpt, arch)
        else:
            # Default: Mamba v7
            self.model = self._build_mamba_v7(ckpt, arch)

        self.model.to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("  Parameters: %s", f"{n_params:,}")

        # Feature normalization stats
        stats = np.load(stats_path)
        self.feat_mean = torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)
        self.feat_std = torch.tensor(
            np.clip(stats["std"], 1e-8, None), dtype=torch.float32, device=self.device
        )

        # Confidence calibration from historical predictions
        self.z_mean = 0.0
        self.z_std = 1.0
        if preds_path and Path(preds_path).exists():
            self._calibrate(preds_path)

    def _build_mamba_v7(self, ckpt, arch):
        """Build Mamba v7 CPU-compatible model."""
        from live_trading_linux.mamba_inference import EventMambaCPU
        model = EventMambaCPU(
            n_features=25,
            d_model=arch.get("d_model", 96),
            d_state=arch.get("d_state", 32),
            n_layers=arch.get("n_layers", 3),
            d_conv=arch.get("d_conv", 4),
            dropout=arch.get("dropout", 0.1),
            n_targets=3,
        )
        model.load_state_dict(ckpt["model_state"])
        return model

    def _build_cnn_mamba(self, ckpt, arch):
        """Build CNN-Mamba v2 model. Falls back to Mamba v7 if class not found."""
        try:
            # Try importing CNN-Mamba v2 architecture
            from models.cnn_mamba_v2 import CNNMambaV2
            model = CNNMambaV2(**arch)
            model.load_state_dict(ckpt["model_state"])
            return model
        except (ImportError, KeyError):
            log.warning("CNN-Mamba v2 class not found, falling back to Mamba v7 loader")
            return self._build_mamba_v7(ckpt, arch)

    def _calibrate(self, preds_path: str):
        """Calibrate z-score normalization from historical predictions."""
        data = np.load(preds_path, allow_pickle=True)
        if "predictions" in data:
            preds_10s = data["predictions"][:, 2]  # 10s horizon
        elif "preds_10s" in data:
            preds_10s = data["preds_10s"]
        else:
            log.warning("No prediction keys in %s for calibration", preds_path)
            return
        self.z_mean = float(np.mean(preds_10s))
        self.z_std = float(max(np.std(preds_10s), 1e-8))
        log.info("  Calibrated: mean=%.6f, std=%.6f", self.z_mean, self.z_std)

    def predict(self, feature_window: np.ndarray) -> Dict:
        """Run inference on a (window_size, 25) feature array.

        Returns dict with pred_1s, pred_5s, pred_10s, z_score_10s, direction.
        """
        import torch
        with torch.no_grad():
            x = torch.tensor(feature_window, dtype=torch.float32, device=self.device)
            x = (x - self.feat_mean) / self.feat_std
            x = x.unsqueeze(0)
            preds = self.model(x).squeeze(0).cpu().numpy()

        pred_1s, pred_5s, pred_10s = float(preds[0]), float(preds[1]), float(preds[2])
        z_score = (pred_10s - self.z_mean) / self.z_std
        direction = 1 if pred_10s > 0 else -1

        self._prediction_count += 1
        return {
            "pred_1s": pred_1s,
            "pred_5s": pred_5s,
            "pred_10s": pred_10s,
            "z_score_10s": z_score,
            "abs_z": abs(z_score),
            "direction": direction,
            "prediction_num": self._prediction_count,
        }


# ============================================================
# Position & Trade Tracking (per strategy)
# ============================================================
@dataclass
class Position:
    size: int = 0             # +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    entry_ts_ns: int = 0      # nanosecond timestamp
    entry_z: float = 0.0
    entry_direction: int = 0
    sl_price: float = 0.0     # stop loss (bracket mode)
    tp_price: float = 0.0     # take profit (bracket mode)


@dataclass
class Trade:
    strategy: str
    entry_ts_ns: int
    exit_ts_ns: int
    direction: int            # +1 long, -1 short
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    hold_ms: float
    exit_reason: str          # "time", "bracket_tp", "bracket_sl", "signal_flip", "eod"
    entry_z: float


@dataclass
class StrategyState:
    """Per-strategy state."""
    config: Dict
    position: Position = field(default_factory=Position)
    trades: List[Trade] = field(default_factory=list)
    total_net_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    last_trade_ts_ns: int = 0  # for cooldown
    signals_seen: int = 0
    signals_traded: int = 0
    equity_high: float = 0.0
    equity_low: float = 0.0

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.net_pnl > 0)
        return wins / len(self.trades)

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_net_pnl / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in self.trades if t.net_pnl < 0))
        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def sortino(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        rets = [t.net_pnl for t in self.trades]
        mean_ret = statistics.mean(rets)
        downside = [r for r in rets if r < 0]
        if not downside:
            return float('inf') if mean_ret > 0 else 0.0
        ds = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
        if ds == 0:
            return 0.0
        trades_per_year = 50 * 252
        return (mean_ret / ds) * math.sqrt(trades_per_year)


# ============================================================
# Market state from event stream
# ============================================================
@dataclass
class MarketState:
    """Reconstructed market state from MBO event features."""
    mid_price: float = 0.0
    spread_ticks: float = 0.0
    last_ts_ns: int = 0
    event_count: int = 0


# ============================================================
# Paper Fill Simulator
# ============================================================
class FillSimulator:
    """Simulate fills for chase entries and bracket exits."""

    @staticmethod
    def chase_entry_price(direction: int, mid_price: float,
                          spread_ticks: float) -> float:
        """Simulate chase entry: cross spread + chase CHASE_TICKS beyond.
        For a buy: pay ask + chase. For a sell: hit bid - chase.
        """
        half_spread = (spread_ticks * ES_TICK_SIZE) / 2.0
        chase = CHASE_TICKS * ES_TICK_SIZE
        slippage = DEFAULT_SLIPPAGE_TICKS * ES_TICK_SIZE
        if direction == 1:  # buy
            return mid_price + half_spread + slippage
        else:  # sell
            return mid_price - half_spread - slippage

    @staticmethod
    def check_bracket(position: Position, mid_price: float) -> Optional[str]:
        """Check if bracket TP or SL is hit. Returns exit reason or None."""
        if position.size == 0:
            return None
        if position.size > 0:  # long
            if mid_price >= position.tp_price:
                return "bracket_tp"
            if mid_price <= position.sl_price:
                return "bracket_sl"
        else:  # short
            if mid_price <= position.tp_price:
                return "bracket_tp"
            if mid_price >= position.sl_price:
                return "bracket_sl"
        return None

    @staticmethod
    def bracket_exit_price(position: Position, exit_reason: str) -> float:
        """Get fill price for bracket exit."""
        if exit_reason == "bracket_tp":
            return position.tp_price
        elif exit_reason == "bracket_sl":
            return position.sl_price
        return 0.0


# ============================================================
# Paper Runner Engine
# ============================================================
class PaperRunnerMonday:
    """Multi-strategy paper trading engine."""

    def __init__(self, model: ModelInference, strategies: List[Dict],
                 discord_notify: bool = True):
        self.model = model
        self.fill_sim = FillSimulator()
        self.discord_notify = discord_notify

        # Initialize per-strategy state
        self.strategies: List[StrategyState] = []
        for cfg in strategies:
            self.strategies.append(StrategyState(config=cfg))

        # Event buffer for windowed inference
        self.event_buffer: List[np.ndarray] = []
        self.last_signal: Optional[Dict] = None
        self.market = MarketState()

        # Reporting
        self._last_report_ts = 0
        self._report_interval_ns = 5 * 60 * 1_000_000_000  # 5 minutes
        self._start_time = time.time()

        # Trade log
        self._trade_log_path = LOG_DIR / f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._trade_fh = open(self._trade_log_path, "w", buffering=1)

        log.info("Paper Runner initialized with %d strategies", len(self.strategies))
        for s in self.strategies:
            log.info("  Strategy: %s (z=%.1f, hold=%dms, exit=%s)",
                     s.name, s.config["z_threshold"], s.config["hold_ms"],
                     s.config["exit_mode"])

    def process_event(self, features: np.ndarray, ts_ns: int,
                      event_idx: int) -> None:
        """Process one MBO event (25-dim smart_v3 features + timestamp).

        This is the main entry point called for each event in both
        replay and live modes.
        """
        self.market.event_count += 1
        self.market.last_ts_ns = ts_ns

        # Reconstruct approximate mid price from features
        # Feature[3] = price_rel_ticks (normalized: clip[-50,50]/25)
        # Feature[5] = spread_ticks (normalized: clip[0,20]/5)
        # We track relative price movement, not absolute. For P&L we need
        # a reference price. We'll use a running price accumulator.
        price_rel = features[3] * 25.0  # un-normalize
        spread_t = features[5] * 5.0    # un-normalize
        self.market.spread_ticks = spread_t

        # Accumulate events for windowed inference
        self.event_buffer.append(features)

        # Check bracket exits on every event (before new signal)
        self._check_bracket_exits(ts_ns)

        # Check time-based exits on every event
        self._check_time_exits(ts_ns)

        # Run inference when we have enough events
        buf_len = len(self.event_buffer)
        if buf_len < self.model.window_size:
            return

        # Predict every `stride` events after first window
        events_past_window = buf_len - self.model.window_size
        if events_past_window % self.model.stride != 0:
            return

        # Build window and predict
        window = np.array(self.event_buffer[-self.model.window_size:])
        signal = self.model.predict(window)
        signal["ts_ns"] = ts_ns
        signal["event_idx"] = event_idx
        self.last_signal = signal

        # Apply signal to each strategy
        self._apply_signal(signal, ts_ns)

        # Trim buffer to prevent unbounded growth
        if buf_len > self.model.window_size * 3:
            self.event_buffer = self.event_buffer[-self.model.window_size:]

        # Periodic reporting
        if ts_ns - self._last_report_ts >= self._report_interval_ns:
            self._print_report(ts_ns)
            self._last_report_ts = ts_ns

    def _ts_to_et_hour_min(self, ts_ns: int) -> Tuple[int, int]:
        """Convert nanosecond timestamp to (hour, minute) in US/Eastern."""
        # Timestamps are in nanoseconds since epoch, in exchange time (ET)
        ts_s = ts_ns / 1e9
        dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
        # ET = UTC - 4 (EDT) or UTC - 5 (EST). April = EDT.
        et_offset = timedelta(hours=-4)
        dt_et = dt + et_offset
        return dt_et.hour, dt_et.minute

    def _in_time_filter(self, ts_ns: int, time_filter: Optional[Tuple]) -> bool:
        """Check if timestamp falls within strategy's time filter."""
        if time_filter is None:
            return True
        h, m = self._ts_to_et_hour_min(ts_ns)
        start_h, start_m, end_h, end_m = time_filter
        current = h * 60 + m
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        return start <= current < end

    def _apply_signal(self, signal: Dict, ts_ns: int) -> None:
        """Apply model signal to all strategies."""
        abs_z = signal["abs_z"]
        direction = signal["direction"]

        for strat in self.strategies:
            cfg = strat.config
            strat.signals_seen += 1

            # Time filter check
            if not self._in_time_filter(ts_ns, cfg.get("time_filter")):
                continue

            # Z-threshold check
            if abs_z < cfg["z_threshold"]:
                # Check signal-flip override: if in position and signal flips
                # with moderate confidence, exit
                if strat.position.size != 0:
                    pos_dir = strat.position.entry_direction
                    if direction != pos_dir and abs_z >= cfg["z_threshold"] * 0.7:
                        self._close_position(strat, ts_ns, "signal_flip")
                continue

            # Cooldown check
            if ts_ns - strat.last_trade_ts_ns < COOLDOWN_MS * 1_000_000:
                continue

            # Already in position?
            if strat.position.size != 0:
                # Same direction: hold
                if strat.position.entry_direction == direction:
                    continue
                # Opposite direction with strong signal: flip
                self._close_position(strat, ts_ns, "signal_flip")
                # Fall through to open new position

            # Open new position
            self._open_position(strat, direction, signal, ts_ns)

    def _open_position(self, strat: StrategyState, direction: int,
                       signal: Dict, ts_ns: int) -> None:
        """Open a new position for a strategy."""
        cfg = strat.config
        mid_price = self._get_reference_price(ts_ns)
        entry_price = self.fill_sim.chase_entry_price(
            direction, mid_price, self.market.spread_ticks
        )

        pos = strat.position
        pos.size = direction
        pos.entry_price = entry_price
        pos.entry_ts_ns = ts_ns
        pos.entry_z = signal["z_score_10s"]
        pos.entry_direction = direction

        # Set bracket levels if bracket mode
        if cfg["exit_mode"] == "bracket" and cfg["bracket"]:
            sl_ticks = cfg["bracket"]["sl_ticks"]
            tp_ticks = cfg["bracket"]["tp_ticks"]
            if direction == 1:  # long
                pos.sl_price = entry_price - sl_ticks * ES_TICK_SIZE
                pos.tp_price = entry_price + tp_ticks * ES_TICK_SIZE
            else:  # short
                pos.sl_price = entry_price + sl_ticks * ES_TICK_SIZE
                pos.tp_price = entry_price - tp_ticks * ES_TICK_SIZE

        strat.signals_traded += 1
        strat.last_trade_ts_ns = ts_ns

        dir_str = "LONG" if direction == 1 else "SHORT"
        h, m = self._ts_to_et_hour_min(ts_ns)
        log.info("[%s] OPEN %s @ %.2f (z=%.2f) %02d:%02d ET",
                 strat.name, dir_str, entry_price, signal["z_score_10s"], h, m)

    def _close_position(self, strat: StrategyState, ts_ns: int,
                        reason: str) -> None:
        """Close position and record trade."""
        pos = strat.position
        if pos.size == 0:
            return

        mid_price = self._get_reference_price(ts_ns)

        # Determine exit price
        if reason.startswith("bracket_"):
            exit_price = self.fill_sim.bracket_exit_price(pos, reason)
        else:
            # Time exit or signal flip: market exit
            exit_price = self.fill_sim.chase_entry_price(
                -pos.size, mid_price, self.market.spread_ticks
            )

        # Compute P&L
        if pos.size > 0:  # was long
            gross_pnl = (exit_price - pos.entry_price) * ES_POINT_VALUE
        else:  # was short
            gross_pnl = (pos.entry_price - exit_price) * ES_POINT_VALUE

        commission = COMMISSION_PER_SIDE * 2  # entry + exit
        net_pnl = gross_pnl - commission
        hold_ms = (ts_ns - pos.entry_ts_ns) / 1_000_000

        trade = Trade(
            strategy=strat.name,
            entry_ts_ns=pos.entry_ts_ns,
            exit_ts_ns=ts_ns,
            direction=pos.size,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            hold_ms=hold_ms,
            exit_reason=reason,
            entry_z=pos.entry_z,
        )
        strat.trades.append(trade)
        strat.total_net_pnl += net_pnl
        strat.total_gross_pnl += gross_pnl
        strat.last_trade_ts_ns = ts_ns

        # Drawdown tracking
        if strat.total_net_pnl > strat.peak_pnl:
            strat.peak_pnl = strat.total_net_pnl
        dd = strat.peak_pnl - strat.total_net_pnl
        if dd > strat.max_drawdown:
            strat.max_drawdown = dd

        # Equity high/low tracking
        if strat.total_net_pnl > strat.equity_high:
            strat.equity_high = strat.total_net_pnl
        if strat.total_net_pnl < strat.equity_low:
            strat.equity_low = strat.total_net_pnl

        # Log trade (convert numpy types for JSON)
        trade_dict = asdict(trade)
        for k, v in trade_dict.items():
            if isinstance(v, (np.floating, np.integer)):
                trade_dict[k] = float(v)
        self._trade_fh.write(json.dumps(trade_dict) + "\n")

        pnl_sym = "+" if net_pnl >= 0 else ""
        dir_str = "LONG" if pos.size > 0 else "SHORT"
        h, m = self._ts_to_et_hour_min(ts_ns)
        log.info("[%s] CLOSE %s @ %.2f (%s) %s$%.2f | hold=%.0fms | cumPnL=$%.2f  %02d:%02d ET",
                 strat.name, dir_str, exit_price, reason,
                 pnl_sym, net_pnl, hold_ms, strat.total_net_pnl, h, m)

        # Reset position
        pos.size = 0
        pos.entry_price = 0.0
        pos.entry_ts_ns = 0
        pos.entry_z = 0.0
        pos.entry_direction = 0
        pos.sl_price = 0.0
        pos.tp_price = 0.0

    def _check_bracket_exits(self, ts_ns: int) -> None:
        """Check all bracket strategies for TP/SL hits."""
        mid_price = self._get_reference_price(ts_ns)
        for strat in self.strategies:
            if strat.config["exit_mode"] != "bracket":
                continue
            if strat.position.size == 0:
                continue
            # Require minimum 500ms hold before checking brackets
            # to avoid spurious fills from price reference jitter
            hold_ns = ts_ns - strat.position.entry_ts_ns
            if hold_ns < 500_000_000:  # 500ms
                continue
            reason = self.fill_sim.check_bracket(strat.position, mid_price)
            if reason:
                self._close_position(strat, ts_ns, reason)

    def _check_time_exits(self, ts_ns: int) -> None:
        """Check all time-exit strategies for hold timeout."""
        for strat in self.strategies:
            if strat.position.size == 0:
                continue
            hold_ns = ts_ns - strat.position.entry_ts_ns
            hold_ms = hold_ns / 1_000_000
            if hold_ms >= strat.config["hold_ms"]:
                self._close_position(strat, ts_ns, "time")

    # ── Price tracking ──
    # Smart_v3 features encode price_rel_ticks (feature[3] * 25.0) which is
    # the tick-level price change relative to session reference. We track
    # a running reference price for P&L computation.

    def _init_reference_price(self, base_price: float) -> None:
        """Set the base reference price (e.g., session open)."""
        self._ref_price = base_price

    def _get_reference_price(self, ts_ns: int) -> float:
        """Get current mid price estimate."""
        return getattr(self, '_ref_price', 5000.0)

    def update_reference_price(self, price: float) -> None:
        """Update running reference price from raw event data."""
        self._ref_price = price

    # ── Reporting ──
    def _print_report(self, ts_ns: int) -> None:
        """Print 5-minute P&L report for all strategies."""
        h, m = self._ts_to_et_hour_min(ts_ns)
        elapsed = time.time() - self._start_time
        log.info("=" * 80)
        log.info("5-MIN REPORT | %02d:%02d ET | elapsed=%.0fs | events=%d",
                 h, m, elapsed, self.market.event_count)
        log.info("-" * 80)
        log.info("%-25s %6s %5s %8s %8s %8s %6s",
                 "Strategy", "Trades", "Win%", "NetPnL", "MaxDD", "PF", "Pos")
        log.info("-" * 80)

        total_pnl = 0.0
        for s in self.strategies:
            pos_str = f"{s.position.size:+d}" if s.position.size != 0 else "flat"
            wr = s.win_rate * 100 if s.trades else 0
            pf = s.profit_factor if s.trades else 0.0
            pf_str = f"{pf:.2f}" if pf < 100 else "inf"
            log.info("%-25s %6d %4.0f%% %+8.2f %8.2f %6s %6s",
                     s.name, len(s.trades), wr, s.total_net_pnl,
                     s.max_drawdown, pf_str, pos_str)
            total_pnl += s.total_net_pnl

        log.info("-" * 80)
        log.info("%-25s %6s %5s %+8.2f", "TOTAL", "", "", total_pnl)
        log.info("=" * 80)

    def print_final_report(self) -> None:
        """Print end-of-day summary."""
        log.info("\n" + "=" * 80)
        log.info("FINAL PAPER TRADING REPORT")
        log.info("=" * 80)

        total_pnl = 0.0
        total_trades = 0
        for s in self.strategies:
            n = len(s.trades)
            total_trades += n
            total_pnl += s.total_net_pnl
            log.info("\n--- %s ---", s.name)
            log.info("  Trades: %d  |  Win rate: %.1f%%", n, s.win_rate * 100)
            log.info("  Net P&L: $%.2f  |  Gross: $%.2f", s.total_net_pnl, s.total_gross_pnl)
            log.info("  Profit Factor: %.2f  |  Avg P&L: $%.2f",
                     s.profit_factor, s.avg_pnl)
            log.info("  Max Drawdown: $%.2f  |  Sortino: %.2f",
                     s.max_drawdown, s.sortino)
            log.info("  Signals seen: %d  |  Traded: %d",
                     s.signals_seen, s.signals_traded)
            if s.trades:
                holds = [t.hold_ms for t in s.trades]
                log.info("  Avg hold: %.0fms  |  Median: %.0fms",
                         statistics.mean(holds), statistics.median(holds))

                # Exit reason breakdown
                reasons = {}
                for t in s.trades:
                    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
                log.info("  Exit reasons: %s", dict(reasons))

        log.info("\n" + "=" * 80)
        log.info("TOTAL: %d trades across %d strategies | Net P&L: $%.2f",
                 total_trades, len(self.strategies), total_pnl)
        log.info("=" * 80)

        # Save results JSON
        results_path = LOG_DIR / f"paper_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "strategies": {},
        }
        for s in self.strategies:
            results["strategies"][s.name] = {
                "trades": len(s.trades),
                "win_rate": s.win_rate,
                "net_pnl": s.total_net_pnl,
                "gross_pnl": s.total_gross_pnl,
                "profit_factor": s.profit_factor,
                "max_drawdown": s.max_drawdown,
                "sortino": s.sortino,
                "avg_pnl": s.avg_pnl,
                "signals_seen": s.signals_seen,
                "signals_traded": s.signals_traded,
            }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        log.info("Results saved to %s", results_path)
        log.info("Trades log: %s", self._trade_log_path)

    def close(self) -> None:
        """Close all open positions (EOD) and finalize."""
        ts_ns = self.market.last_ts_ns
        for strat in self.strategies:
            if strat.position.size != 0:
                self._close_position(strat, ts_ns, "eod")
        self._trade_fh.close()


# ============================================================
# Replay Mode — read from saved NPZ
# ============================================================
def run_replay(npz_path: str, model: ModelInference, speed: float = 100.0,
               max_events: int = 0) -> PaperRunnerMonday:
    """Replay saved MBO events through the paper runner.

    Args:
        npz_path: path to smart_v3 NPZ file
        model: initialized model inference engine
        speed: replay speed multiplier (0 = as fast as possible)
        max_events: limit events (0 = all)
    """
    log.info("Loading replay data from %s", npz_path)
    data = np.load(npz_path)
    events = data["events"]         # (N, 25) float32 - precomputed smart_v3
    timestamps = data["timestamps"] # (N,) int64 - nanosecond timestamps
    N = len(events)
    if max_events > 0:
        N = min(N, max_events)
    log.info("  %d events, replaying %d at %.0fx speed", len(events), N,
             speed if speed > 0 else float('inf'))

    # Try to get raw events for price reference
    raw_npz = Path(npz_path).parent.parent / "mbo_events" / Path(npz_path).name
    raw_prices = None
    if raw_npz.exists():
        raw_data = np.load(str(raw_npz))
        raw_events = raw_data["events"]  # (N, 6): td, et, side, price_rel, qty_log, spread
        # Feature[3] in raw = price_rel_ticks (NOT normalized)
        # We need actual prices. Reconstruct from price_rel cumsum.
        # Actually, price_rel_ticks is the tick change from a rolling reference,
        # not cumulative. We'll use a synthetic base price for P&L tracking.
        raw_prices = raw_events[:N, 3]  # price_rel_ticks
        log.info("  Raw price data loaded for price reference")

    runner = PaperRunnerMonday(model, STRATEGIES)

    # Set initial reference price. Since we work with relative ticks,
    # we'll use a base of 5000 and accumulate price changes.
    base_price = 5800.0  # approximate ES price Feb 2026
    running_price = base_price
    runner._init_reference_price(running_price)

    prev_ts = timestamps[0]
    t0_wall = time.time()
    report_interval = max(N // 20, 100000)

    for i in range(N):
        feat = events[i]
        ts = int(timestamps[i])

        # Update reference price from raw relative ticks
        if raw_prices is not None:
            # price_rel_ticks from raw is the actual relative tick value
            price_change_pts = raw_prices[i] * ES_TICK_SIZE
            running_price = base_price + raw_prices[i] * ES_TICK_SIZE
            # Use a simple running mid from relative changes
            runner.update_reference_price(base_price + raw_prices[i] * ES_TICK_SIZE)
        else:
            # Approximate from normalized feature[3]
            price_rel = feat[3] * 25.0  # un-normalize
            runner.update_reference_price(base_price + price_rel * ES_TICK_SIZE)

        runner.process_event(feat, ts, i)

        # Speed control
        if speed > 0 and i > 0:
            dt_sim_ns = ts - prev_ts
            if dt_sim_ns > 0:
                dt_wall = dt_sim_ns / 1e9 / speed
                if dt_wall > 0.01:  # only sleep for > 10ms wall time
                    time.sleep(dt_wall)
        prev_ts = ts

        # Progress
        if (i + 1) % report_interval == 0:
            pct = (i + 1) / N * 100
            elapsed = time.time() - t0_wall
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            log.info("Progress: %d/%d (%.1f%%) | %.0f events/sec",
                     i + 1, N, pct, rate)

    # Close all positions
    runner.close()
    runner.print_final_report()

    elapsed = time.time() - t0_wall
    log.info("Replay complete: %d events in %.1fs (%.0f events/sec)",
             N, elapsed, N / elapsed if elapsed > 0 else 0)

    return runner


# ============================================================
# Live Mode — read from MBO recorder output
# ============================================================
def run_live(model: ModelInference, recorder_dir: Optional[str] = None) -> None:
    """Run live paper trading by tailing MBO recorder output.

    In live mode, we use StreamingFeaturesSmartV3 to compute features
    from raw MBO events coming from the Rithmic feed.
    """
    from live_trading_linux.streaming_features_smart_v3 import StreamingFeaturesSmartV3

    log.info("Starting LIVE paper trading mode")
    log.info("  Waiting for MBO events from recorder...")

    runner = PaperRunnerMonday(model, STRATEGIES)
    features_engine = StreamingFeaturesSmartV3()

    # If recorder_dir specified, tail the latest file
    if recorder_dir:
        _run_live_from_recorder(runner, model, features_engine, recorder_dir)
    else:
        # Placeholder: in real deployment, this would connect to Rithmic
        # via the RithmicClient and process events via callbacks
        log.info("No recorder directory specified. In production, connect to Rithmic.")
        log.info("Use --recorder-dir to tail MBO recorder output files.")
        log.info("Or use --replay to test on historical data.")
        return

    runner.close()
    runner.print_final_report()


def _run_live_from_recorder(runner: PaperRunnerMonday, model: ModelInference,
                            features_engine, recorder_dir: str) -> None:
    """Tail MBO recorder JSONL output and process events."""
    from pathlib import Path
    import glob

    rec_path = Path(recorder_dir)
    if not rec_path.exists():
        log.error("Recorder directory not found: %s", recorder_dir)
        return

    # Find latest JSONL file
    files = sorted(rec_path.glob("*.jsonl"))
    if not files:
        log.error("No JSONL files in %s", recorder_dir)
        return

    latest = files[-1]
    log.info("Tailing %s", latest)

    base_price = 5800.0
    runner._init_reference_price(base_price)

    with open(latest, "r") as f:
        # Seek to end for live tailing
        f.seek(0, 2)
        event_idx = 0

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.001)  # 1ms poll
                continue

            try:
                ev = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Extract raw fields from recorder event
            td_log = float(ev.get("time_delta_log", 0))
            et_id = int(ev.get("event_type", 0))
            side = int(ev.get("side", 0))
            price_rel = float(ev.get("price_rel_ticks", 0))
            qty_log = float(ev.get("qty_log", 0))
            spread = float(ev.get("spread_ticks", 0))
            ts_ns = int(ev.get("timestamp", time.time_ns()))

            # Compute smart_v3 features
            feat = features_engine.update(td_log, et_id, side,
                                          price_rel, qty_log, spread)

            # Update price reference
            runner.update_reference_price(base_price + price_rel * ES_TICK_SIZE)

            runner.process_event(feat, ts_ns, event_idx)
            event_idx += 1


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Multi-strategy paper trading runner for Monday market open. "
                    "ZERO real orders — paper fills only.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", metavar="NPZ",
                      help="Replay mode: path to smart_v3 NPZ file")
    mode.add_argument("--live", action="store_true",
                      help="Live mode: connect to MBO recorder")

    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help=f"Model weights path (default: Mamba v7)")
    ap.add_argument("--stats", default=str(DEFAULT_STATS),
                    help="Feature stats path for normalization")
    ap.add_argument("--preds", default=str(DEFAULT_PREDS),
                    help="Historical predictions for z-score calibration")
    ap.add_argument("--speed", type=float, default=0,
                    help="Replay speed (0 = max speed, default: 0)")
    ap.add_argument("--max-events", type=int, default=0,
                    help="Max events to process (0 = all)")
    ap.add_argument("--recorder-dir", default=None,
                    help="MBO recorder output directory (live mode)")
    ap.add_argument("--window", type=int, default=1000,
                    help="Inference window size (default: 1000)")
    ap.add_argument("--stride", type=int, default=500,
                    help="Inference stride (default: 500)")
    ap.add_argument("--no-discord", action="store_true",
                    help="Disable Discord notifications")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Paper Runner Monday — PAPER MODE ONLY, NO REAL ORDERS")
    log.info("  Mode: %s", "REPLAY" if args.replay else "LIVE")
    log.info("  Weights: %s", args.weights)

    # Validate paths
    if not Path(args.weights).exists():
        log.error("Weights not found: %s", args.weights)
        sys.exit(1)
    if not Path(args.stats).exists():
        log.error("Stats not found: %s", args.stats)
        sys.exit(1)

    # Load model
    preds_path = args.preds if Path(args.preds).exists() else None
    model = ModelInference(
        weights_path=args.weights,
        stats_path=args.stats,
        preds_path=preds_path,
        window_size=args.window,
        stride=args.stride,
    )

    # Run
    if args.replay:
        if not Path(args.replay).exists():
            log.error("Replay file not found: %s", args.replay)
            sys.exit(1)
        run_replay(args.replay, model, speed=args.speed, max_events=args.max_events)
    else:
        run_live(model, recorder_dir=args.recorder_dir)


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    _sig.signal(_sig.SIGINT, lambda *_: sys.exit(0))
    main()
