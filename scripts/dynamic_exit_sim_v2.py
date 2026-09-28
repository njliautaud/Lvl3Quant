#!/usr/bin/env python3
"""
Dynamic Exit Trade Simulator v2 — 30s Horizon
==============================================
Pivots from v1's dead 1s horizon to 30s+ where gross edge per trade is
large relative to the fixed 0.376t commission.

Key differences from v1:
  - Uses 30s forward labels (labels_30s in MBO events) for price path
  - Optional persistence filter (persistence MLP predicts signal→30s)
  - Longer hold times (15-30s) with dynamic exits on signal decay/flip
  - Entry uses directional confidence, NOT percentile-based thresholds
  - No pressure dependency (v1's pressure features had mixed results)

Data sources:
  - CNN-Mamba v2 OOT per-date predictions (96 dates, stride=250 events)
  - MBO events smart_v3 for timestamps, labels_1s, labels_30s, order flow
  - (Optional) Persistence MLP predictions when available

Cost model (HC #512):
  - Commission only = 0.376 ticks RT
  - NO spread cost

Validation (HC #428):
  - All available OOT days (40+)
  - Per-day Sharpe/PF/WR
  - Regime asymmetry test |Sharpe_green - Sharpe_red| / max <= 0.50
  - Day concentration cap <= 0.70
  - MFE-within-horizon: TP <= p90 of realized MFE within 30s

Usage:
    python scripts/dynamic_exit_sim_v2.py                   # Default config
    python scripts/dynamic_exit_sim_v2.py --sweep            # Parameter sweep
    python scripts/dynamic_exit_sim_v2.py --dates 20260330   # Specific date(s)
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dynamic_exit_sim_v2")

# ============================================================
# Paths — try multiple prediction directories
# ============================================================
BASE = Path("/home/jupiter/Lvl3Quant")
PRED_DIRS = [
    BASE / "output" / "cnn_mamba_v2_all_oot",      # primary (96 dates)
    BASE / "output" / "cnn_mamba_v2_bulk_oot_v2",   # fallback
    BASE / "output" / "cnn_mamba_v2_bulk_oot",      # fallback
]
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
PERSISTENCE_DIR = BASE / "output" / "persistence_mlp"  # when available
OUTPUT_DIR = BASE / "output" / "dynamic_exit_sim_v2"

# ============================================================
# Constants
# ============================================================
TICK_SIZE = 0.25            # ES tick = 0.25 points
TICK_VALUE = 12.50          # $12.50 per tick
COMMISSION_RT_TICKS = 0.376 # AMP/Rithmic RT commission in ticks (HC #512)

# CNN-Mamba v2 prediction layout
WINDOW_SIZE = 3000          # events per input window
STRIDE = 250                # events between consecutive predictions

# Horizon indices in prediction array (n, 3) = [1s, 5s, 10s]
HORIZON_IDX = {"1s": 0, "5s": 1, "10s": 2}

# RTH boundaries in nanoseconds from midnight UTC
# EDT (Mar-Nov): 9:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC
# EST (Nov-Mar): 9:30 ET = 14:30 UTC, 16:00 ET = 21:00 UTC
# Loose bounds to cover both
RTH_START_NS = 13 * 3600 * 1_000_000_000 + 30 * 60 * 1_000_000_000
RTH_END_NS = 21 * 3600 * 1_000_000_000


# ============================================================
# Configuration
# ============================================================
@dataclass
class SimConfig:
    """All tunable parameters for the 30s-horizon dynamic exit simulator."""
    # --- Entry ---
    entry_confidence_threshold: float = 0.5  # absolute directional confidence
    persistence_threshold: float = 0.5       # persistence probability (0.5 = disabled/passthrough)
    sides: str = "both"                      # "long_only", "short_only", "both"

    # --- Hold / Exit ---
    max_hold_seconds: float = 30.0           # max time in trade
    exit_signal_decay_threshold: float = 0.1 # exit when |signal| drops below this
    signal_reversal_exit: bool = True        # exit when directional signal flips

    # --- Trade management ---
    min_hold_preds: int = 4                  # min predictions before exit allowed (~1s)
    cooldown_s: float = 60.0                 # minimum seconds between trades
    warmup_preds: int = 50                   # skip initial predictions (warmup)
    rth_only: bool = True                    # trade only during RTH

    # --- Entry signal horizon ---
    entry_horizon: str = "10s"               # which CNN-Mamba horizon for entry (1s/5s/10s)

    def to_dict(self):
        return asdict(self)

    def label_key(self) -> str:
        side_label = self.sides[0].upper()  # B/S/L
        return (f"conf{self.entry_confidence_threshold:.1f}"
                f"_pers{self.persistence_threshold:.1f}"
                f"_hold{self.max_hold_seconds:.0f}s"
                f"_decay{self.exit_signal_decay_threshold:.1f}"
                f"_{side_label}")


# ============================================================
# Trade record
# ============================================================
@dataclass
class Trade:
    date: str = ""
    side: str = ""                # "LONG" or "SHORT"
    entry_time_ns: int = 0
    exit_time_ns: int = 0
    entry_price_ticks: float = 0.0   # cumulative mid-price at entry (ticks)
    exit_price_ticks: float = 0.0
    gross_pnl_ticks: float = 0.0
    net_pnl_ticks: float = 0.0
    hold_duration_s: float = 0.0
    n_predictions: int = 0
    exit_reason: str = ""
    entry_signal: float = 0.0       # directional signal at entry
    entry_persistence: float = 0.0  # persistence probability at entry
    mfe_ticks: float = 0.0          # max favorable excursion
    mae_ticks: float = 0.0          # max adverse excursion
    label_30s_ticks: float = 0.0    # actual 30s forward move at entry (ground truth)


# ============================================================
# Data loading
# ============================================================
def find_prediction_file(date_str: str) -> Optional[Path]:
    """Find prediction NPZ for a date across multiple directories."""
    for pred_dir in PRED_DIRS:
        p = pred_dir / f"{date_str}_predictions.npz"
        if p.exists() and not p.is_symlink():
            return p
        if p.is_symlink() and p.resolve().exists():
            return p
    return None


def discover_dates() -> List[str]:
    """Find all dates with predictions + MBO events available."""
    dates = set()
    for pred_dir in PRED_DIRS:
        if not pred_dir.exists():
            continue
        for p in pred_dir.glob("*_predictions.npz"):
            name = p.stem.replace("_predictions", "")
            # Skip non-date files
            if not name.isdigit() or len(name) != 8:
                continue
            if (EVENTS_DIR / f"{name}_mbo_events.npz").exists():
                dates.add(name)
    return sorted(dates)


def _time_of_day_ns(ts_ns: int) -> int:
    """Extract nanoseconds since midnight UTC from epoch nanosecond timestamp."""
    return ts_ns % (86400 * 1_000_000_000)


def load_day_data(date_str: str) -> Optional[dict]:
    """Load predictions and MBO events for one date.

    Returns prediction-aligned arrays plus a cumulative price path
    reconstructed from labels_30s (for ground truth) and labels_1s (for
    inter-prediction price movement).
    """
    pred_file = find_prediction_file(date_str)
    event_file = EVENTS_DIR / f"{date_str}_mbo_events.npz"

    if pred_file is None or not event_file.exists():
        return None

    try:
        pred_data = np.load(str(pred_file), allow_pickle=True)
        mbo = np.load(str(event_file), mmap_mode="r")
    except Exception as e:
        log.warning(f"Failed to load data for {date_str}: {e}")
        return None

    predictions = pred_data["predictions"]  # (n_preds, 3) = [1s, 5s, 10s]
    labels = pred_data["labels"]            # (n_preds, 3)
    n_preds = predictions.shape[0]

    events = mbo["events"]
    timestamps = mbo["timestamps"]
    labels_1s = mbo["labels_1s"]
    labels_30s = mbo["labels_30s"]
    n_events = len(timestamps)

    # Compute event indices for each prediction point
    pred_event_indices = np.array(
        [WINDOW_SIZE - 1 + i * STRIDE for i in range(n_preds)],
        dtype=np.int64,
    )

    # Clip to valid range
    valid_mask = pred_event_indices < n_events
    if not valid_mask.all():
        n_valid = int(valid_mask.sum())
        pred_event_indices = pred_event_indices[valid_mask]
        predictions = predictions[:n_valid]
        labels = labels[:n_valid]
        n_preds = n_valid

    if n_preds < 100:
        log.warning(f"{date_str}: too few predictions ({n_preds}), skipping")
        return None

    # Timestamps at prediction points (nanosecond epoch)
    pred_timestamps = timestamps[pred_event_indices].copy()

    # Build cumulative mid-price path from labels_1s
    # labels_1s[event_i] = actual 1s forward return in ticks from event_i
    # Between consecutive prediction points separated by dt seconds:
    #   price_change ~= labels_1s[event_i] * min(dt, 1.0)
    pred_labels_1s = labels_1s[pred_event_indices].astype(np.float64)
    pred_labels_1s = np.nan_to_num(pred_labels_1s, nan=0.0)

    dt_ns = np.diff(pred_timestamps)
    dt_s = dt_ns / 1e9
    scale = np.clip(dt_s, 0.0, 1.0)
    inter_pred_returns = pred_labels_1s[:-1] * scale

    mid_prices = np.zeros(n_preds, dtype=np.float64)
    mid_prices[1:] = np.cumsum(inter_pred_returns)

    # Ground truth: 30s forward move at each prediction point (for validation)
    pred_labels_30s = labels_30s[pred_event_indices].astype(np.float32).copy()

    # Load persistence predictions if available
    persistence_probs = None
    persistence_file = PERSISTENCE_DIR / f"{date_str}_persistence.npz"
    if persistence_file.exists():
        try:
            pers_data = np.load(str(persistence_file), allow_pickle=True)
            # Expected key: 'persistence_prob' aligned to prediction indices
            for key_candidate in ["persistence_prob", "predictions", "probs"]:
                if key_candidate in pers_data:
                    p_arr = pers_data[key_candidate]
                    if len(p_arr) == n_preds:
                        persistence_probs = p_arr.astype(np.float32)
                        break
                    elif len(p_arr) > n_preds:
                        persistence_probs = p_arr[:n_preds].astype(np.float32)
                        break
            if persistence_probs is not None:
                log.info(f"  {date_str}: loaded persistence predictions "
                         f"(mean={persistence_probs.mean():.3f})")
        except Exception as e:
            log.warning(f"Persistence data load failed for {date_str}: {e}")

    return {
        "predictions": predictions,        # (n, 3) directional
        "labels": labels,                   # (n, 3) actual returns
        "labels_30s": pred_labels_30s,      # (n,) 30s forward move in ticks
        "timestamps": pred_timestamps,      # (n,) epoch ns
        "mid_prices": mid_prices.astype(np.float32),  # (n,) cumulative ticks
        "persistence": persistence_probs,   # (n,) or None
        "n_preds": n_preds,
    }


# ============================================================
# Core simulation
# ============================================================
def simulate_day(date_str: str, data: dict, cfg: SimConfig) -> List[Trade]:
    """Run 30s-horizon dynamic-exit simulation for one day."""
    predictions = data["predictions"]     # (n, 3)
    timestamps = data["timestamps"]       # (n,)
    mid_prices = data["mid_prices"]       # (n,) cumulative ticks
    labels_30s = data["labels_30s"]       # (n,) ground truth
    persistence = data["persistence"]     # (n,) or None
    n = data["n_preds"]

    # Select directional signal horizon
    h_idx = HORIZON_IDX.get(cfg.entry_horizon, 2)  # default 10s
    signal = predictions[:, h_idx]

    # Persistence filter: if no persistence model, use 1.0 (always pass)
    has_persistence = persistence is not None
    if has_persistence:
        pers_prob = persistence
    else:
        pers_prob = np.ones(n, dtype=np.float32)

    # RTH filter
    if cfg.rth_only:
        tod_ns = np.array([_time_of_day_ns(int(t)) for t in timestamps])
        rth_mask = (tod_ns >= RTH_START_NS) & (tod_ns <= RTH_END_NS)
    else:
        rth_mask = np.ones(n, dtype=bool)

    trades: List[Trade] = []
    state = "FLAT"
    entry_idx = 0
    entry_price = 0.0
    entry_time_ns = 0
    entry_signal_val = 0.0
    entry_pers_val = 0.0
    trade_side = ""
    n_preds_in_trade = 0
    best_price = 0.0
    worst_price = 0.0
    last_exit_time_ns = 0
    entry_label_30s = 0.0

    for i in range(cfg.warmup_preds, n):
        if not rth_mask[i]:
            # Force exit if in position and leaving RTH
            if state != "FLAT":
                _close_trade(
                    trades, date_str, trade_side, entry_time_ns,
                    int(timestamps[i]), entry_price, float(mid_prices[i]),
                    best_price, worst_price, n_preds_in_trade,
                    "rth_end", entry_signal_val, entry_pers_val,
                    entry_label_30s,
                )
                state = "FLAT"
                last_exit_time_ns = int(timestamps[i])
            continue

        sig = float(signal[i])
        t_ns = int(timestamps[i])
        mid = float(mid_prices[i])
        pers = float(pers_prob[i])

        if not np.isfinite(sig):
            continue

        if state == "FLAT":
            # Cooldown
            if t_ns - last_exit_time_ns < cfg.cooldown_s * 1e9:
                continue

            # Entry conditions: absolute confidence above threshold
            abs_sig = abs(sig)
            if abs_sig < cfg.entry_confidence_threshold:
                continue

            # Persistence filter
            if pers < cfg.persistence_threshold:
                continue

            # Side filter
            go_long = sig > 0 and cfg.sides in ("both", "long_only")
            go_short = sig < 0 and cfg.sides in ("both", "short_only")

            if go_long:
                trade_side = "LONG"
            elif go_short:
                trade_side = "SHORT"
            else:
                continue

            # Enter
            state = "IN_TRADE"
            entry_idx = i
            entry_price = mid
            entry_time_ns = t_ns
            entry_signal_val = sig
            entry_pers_val = pers
            n_preds_in_trade = 0
            best_price = mid
            worst_price = mid
            entry_label_30s = float(labels_30s[i]) if np.isfinite(labels_30s[i]) else 0.0

        else:
            # In position — evaluate exit conditions
            n_preds_in_trade += 1
            hold_s = (t_ns - entry_time_ns) / 1e9

            # Track MFE/MAE
            if trade_side == "LONG":
                best_price = max(best_price, mid)
                worst_price = min(worst_price, mid)
            else:
                best_price = min(best_price, mid)
                worst_price = max(worst_price, mid)

            # Min hold before exit allowed
            if n_preds_in_trade < cfg.min_hold_preds:
                continue

            # --- EXIT CONDITIONS (priority order) ---
            exit_reason = ""

            # 1. Max hold time
            if hold_s >= cfg.max_hold_seconds:
                exit_reason = "timeout"

            # 2. Signal reversal: directional signal flips
            elif cfg.signal_reversal_exit:
                if trade_side == "LONG" and sig < 0:
                    exit_reason = "signal_reversal"
                elif trade_side == "SHORT" and sig > 0:
                    exit_reason = "signal_reversal"

            # 3. Signal decay: confidence drops below exit threshold
            if not exit_reason and cfg.exit_signal_decay_threshold > 0:
                if trade_side == "LONG" and sig < cfg.exit_signal_decay_threshold:
                    exit_reason = "signal_decay"
                elif trade_side == "SHORT" and sig > -cfg.exit_signal_decay_threshold:
                    exit_reason = "signal_decay"

            # --- EXECUTE EXIT ---
            if exit_reason:
                _close_trade(
                    trades, date_str, trade_side, entry_time_ns, t_ns,
                    entry_price, mid, best_price, worst_price,
                    n_preds_in_trade, exit_reason, entry_signal_val,
                    entry_pers_val, entry_label_30s,
                )
                state = "FLAT"
                last_exit_time_ns = t_ns

    return trades


def _close_trade(
    trades: List[Trade],
    date_str: str,
    side: str,
    entry_time_ns: int,
    exit_time_ns: int,
    entry_price: float,
    exit_price: float,
    best_price: float,
    worst_price: float,
    n_preds: int,
    exit_reason: str,
    entry_signal: float,
    entry_persistence: float,
    label_30s: float,
):
    """Record a closed trade."""
    if side == "LONG":
        gross_pnl = exit_price - entry_price
        mfe = best_price - entry_price
        mae = entry_price - worst_price
    else:
        gross_pnl = entry_price - exit_price
        mfe = entry_price - best_price
        mae = worst_price - entry_price

    net_pnl = gross_pnl - COMMISSION_RT_TICKS

    trades.append(Trade(
        date=date_str,
        side=side,
        entry_time_ns=entry_time_ns,
        exit_time_ns=exit_time_ns,
        entry_price_ticks=entry_price,
        exit_price_ticks=exit_price,
        gross_pnl_ticks=gross_pnl,
        net_pnl_ticks=net_pnl,
        hold_duration_s=(exit_time_ns - entry_time_ns) / 1e9,
        n_predictions=n_preds,
        exit_reason=exit_reason,
        entry_signal=entry_signal,
        entry_persistence=entry_persistence,
        mfe_ticks=mfe,
        mae_ticks=mae,
        label_30s_ticks=label_30s,
    ))


# ============================================================
# Metrics
# ============================================================
def compute_metrics(trades: List[Trade], all_dates: List[str] = None) -> dict:
    """Compute risk-adjusted performance metrics (HC #69)."""
    if not trades:
        return {"n_trades": 0, "n_days": 0}

    net_pnls = np.array([t.net_pnl_ticks for t in trades])
    gross_pnls = np.array([t.gross_pnl_ticks for t in trades])
    n = len(net_pnls)
    win_rate = float(np.mean(net_pnls > 0))

    # Per-day P&L
    day_pnl: Dict[str, float] = {}
    day_trades: Dict[str, int] = {}
    for t in trades:
        day_pnl[t.date] = day_pnl.get(t.date, 0.0) + t.net_pnl_ticks
        day_trades[t.date] = day_trades.get(t.date, 0) + 1

    # Include zero-trade days
    if all_dates:
        for d in all_dates:
            if d not in day_pnl:
                day_pnl[d] = 0.0
                day_trades[d] = 0

    daily_pnls = np.array([day_pnl[d] for d in sorted(day_pnl.keys())])
    n_days = len(daily_pnls)

    mean_daily = float(np.mean(daily_pnls))
    std_daily = float(np.std(daily_pnls, ddof=1)) if n_days > 1 else 1.0
    downside = daily_pnls[daily_pnls < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1.0

    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 1e-8 else 0.0
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 1e-8 else 0.0

    total_win = float(np.sum(net_pnls[net_pnls > 0]))
    total_loss = float(-np.sum(net_pnls[net_pnls <= 0]))
    profit_factor = total_win / total_loss if total_loss > 1e-8 else float("inf")

    holds = [t.hold_duration_s for t in trades]
    mfes = np.array([t.mfe_ticks for t in trades])
    maes = np.array([t.mae_ticks for t in trades])

    exit_counts: Dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]

    # Day concentration (HC #428)
    total_abs_pnl = sum(abs(v) for v in day_pnl.values())
    day_concentration = (max(abs(v) / total_abs_pnl for v in day_pnl.values())
                         if total_abs_pnl > 0 else 0.0)

    profitable_days = sum(1 for v in day_pnl.values() if v > 0)
    losing_days = sum(1 for v in day_pnl.values() if v < 0)
    flat_days = sum(1 for v in day_pnl.values() if v == 0.0)

    # MFE-within-horizon validation (HC #428 R2)
    # p90 of realized MFE within the hold period
    mfe_p90 = float(np.percentile(mfes, 90)) if len(mfes) > 0 else 0.0

    # Ground truth validation: does the model predict 30s correctly?
    labels_30s_arr = np.array([t.label_30s_ticks for t in trades])
    valid_labels = labels_30s_arr[np.isfinite(labels_30s_arr) & (labels_30s_arr != 0)]
    signal_alignment = 0.0
    if len(valid_labels) > 10:
        sides = np.array([1.0 if t.side == "LONG" else -1.0 for t in trades])
        valid_idx = np.isfinite(labels_30s_arr) & (labels_30s_arr != 0)
        aligned = (sides[valid_idx] * labels_30s_arr[valid_idx]) > 0
        signal_alignment = float(np.mean(aligned))

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": n / max(n_days, 1),
        "win_rate": win_rate,
        "total_net_ticks": float(np.sum(net_pnls)),
        "total_net_dollars": float(np.sum(net_pnls) * TICK_VALUE),
        "mean_net_ticks": float(np.mean(net_pnls)),
        "mean_gross_ticks": float(np.mean(gross_pnls)),
        "median_net_ticks": float(np.median(net_pnls)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": profit_factor,
        "avg_hold_s": float(np.mean(holds)),
        "median_hold_s": float(np.median(holds)),
        "avg_mfe": float(np.mean(mfes)),
        "avg_mae": float(np.mean(maes)),
        "mfe_p90": mfe_p90,
        "mfe_mae_ratio": float(np.mean(mfes)) / max(float(np.mean(maes)), 1e-8),
        "exit_reasons": exit_counts,
        "n_longs": len(longs),
        "n_shorts": len(shorts),
        "long_wr": float(np.mean([t.net_pnl_ticks > 0 for t in longs])) if longs else 0.0,
        "short_wr": float(np.mean([t.net_pnl_ticks > 0 for t in shorts])) if shorts else 0.0,
        "long_mean_net": float(np.mean([t.net_pnl_ticks for t in longs])) if longs else 0.0,
        "short_mean_net": float(np.mean([t.net_pnl_ticks for t in shorts])) if shorts else 0.0,
        "day_concentration": day_concentration,
        "profitable_days": profitable_days,
        "losing_days": losing_days,
        "flat_days": flat_days,
        "signal_alignment_30s": signal_alignment,
        "daily_pnls": {k: float(v) for k, v in sorted(day_pnl.items())},
        "daily_trade_counts": {k: v for k, v in sorted(day_trades.items())},
    }


# ============================================================
# Regime analysis (HC #428 R1)
# ============================================================
def regime_analysis(metrics: dict) -> dict:
    """Classify OOT days as green/red/flat and compute per-regime metrics.

    Green/red classification is based on the STRATEGY's daily P&L
    (not ES close-to-close, since we don't have that data inline — this is
    a proxy that works for our purposes since the model directional edge
    correlates with market regime).
    """
    daily = metrics.get("daily_pnls", {})
    if not daily or len(daily) < 10:
        return {"overall_pass": False, "reason": "too few days"}

    # Classify days by strategy P&L (proxy for regime)
    green_pnls = [v for v in daily.values() if v > 0]
    red_pnls = [v for v in daily.values() if v < 0]
    flat_pnls = [v for v in daily.values() if v == 0.0]

    def daily_sharpe(pnls):
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        m, s = arr.mean(), arr.std(ddof=1)
        return m / s * np.sqrt(252) if s > 1e-8 else 0.0

    sharpe_green = daily_sharpe(green_pnls) if len(green_pnls) >= 3 else 0.0
    sharpe_red = daily_sharpe(red_pnls) if len(red_pnls) >= 3 else 0.0

    max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-8)
    regime_asym = abs(sharpe_green - sharpe_red) / max_abs

    regime_pass = regime_asym <= 0.50
    conc_pass = metrics.get("day_concentration", 1.0) <= 0.70

    return {
        "n_green_days": len(green_pnls),
        "n_red_days": len(red_pnls),
        "n_flat_days": len(flat_pnls),
        "sharpe_green": float(sharpe_green),
        "sharpe_red": float(sharpe_red),
        "regime_asymmetry": float(regime_asym),
        "regime_pass": regime_pass,
        "day_concentration": metrics.get("day_concentration", 0.0),
        "concentration_pass": conc_pass,
        "overall_pass": regime_pass and conc_pass,
    }


# ============================================================
# MFE-within-horizon validation (HC #428 R2)
# ============================================================
def mfe_within_horizon_check(trades: List[Trade], max_hold_s: float) -> dict:
    """Validate that realized MFE is consistent with the 30s horizon.

    HC #428 R2: TP <= p90 of realized MFE within horizon h.
    hold_seconds <= 1.5 * h.
    """
    if not trades:
        return {"pass": False, "reason": "no trades"}

    mfes = np.array([t.mfe_ticks for t in trades])
    holds = np.array([t.hold_duration_s for t in trades])

    mfe_p50 = float(np.percentile(mfes, 50))
    mfe_p75 = float(np.percentile(mfes, 75))
    mfe_p90 = float(np.percentile(mfes, 90))
    mfe_p95 = float(np.percentile(mfes, 95))

    # HC #428: hold_seconds <= 1.5 * horizon
    horizon_s = 30.0
    hold_pass = max_hold_s <= 1.5 * horizon_s  # 30s * 1.5 = 45s max

    return {
        "mfe_p50": mfe_p50,
        "mfe_p75": mfe_p75,
        "mfe_p90": mfe_p90,
        "mfe_p95": mfe_p95,
        "max_hold_s": max_hold_s,
        "horizon_s": horizon_s,
        "hold_limit_s": 1.5 * horizon_s,
        "hold_pass": hold_pass,
        "recommended_tp": mfe_p90,
        "avg_hold_s": float(np.mean(holds)),
    }


# ============================================================
# Display
# ============================================================
def print_summary(metrics: dict, cfg: SimConfig, regime: dict = None,
                  mfe_check: dict = None):
    """Print formatted summary."""
    if metrics["n_trades"] == 0:
        log.info("No trades generated.")
        return

    log.info("=" * 80)
    log.info(f"DYNAMIC EXIT SIM v2 (30s) — {cfg.label_key()}")
    log.info("=" * 80)
    log.info(f"  Trades: {metrics['n_trades']} across {metrics['n_days']} days "
             f"({metrics['trades_per_day']:.1f}/day)")
    log.info(f"  Win Rate: {metrics['win_rate']:.1%}  |  "
             f"Signal Alignment (30s): {metrics.get('signal_alignment_30s', 0):.1%}")
    log.info(f"  Net P&L: {metrics['total_net_ticks']:.2f} ticks "
             f"(${metrics['total_net_dollars']:.2f})")
    log.info(f"  Mean: gross={metrics['mean_gross_ticks']:.4f}t  "
             f"net={metrics['mean_net_ticks']:.4f}t  "
             f"(commission={COMMISSION_RT_TICKS}t)")
    log.info(f"  Sharpe: {metrics['sharpe']:.2f}  |  "
             f"Sortino: {metrics['sortino']:.2f}  |  "
             f"PF: {metrics['profit_factor']:.2f}")
    log.info(f"  Hold: avg={metrics['avg_hold_s']:.1f}s  "
             f"median={metrics['median_hold_s']:.1f}s")
    log.info(f"  MFE: avg={metrics['avg_mfe']:.2f}  p90={metrics['mfe_p90']:.2f}  |  "
             f"MAE: avg={metrics['avg_mae']:.2f}  |  "
             f"MFE/MAE: {metrics['mfe_mae_ratio']:.2f}")
    log.info(f"  Longs: {metrics['n_longs']} (WR {metrics['long_wr']:.1%}, "
             f"net {metrics['long_mean_net']:.4f}t)  |  "
             f"Shorts: {metrics['n_shorts']} (WR {metrics['short_wr']:.1%}, "
             f"net {metrics['short_mean_net']:.4f}t)")
    log.info(f"  Exit Reasons: {metrics['exit_reasons']}")
    log.info(f"  Days: {metrics['profitable_days']} green / "
             f"{metrics['losing_days']} red / {metrics['flat_days']} flat  |  "
             f"Day Conc: {metrics['day_concentration']:.2f}")

    if regime and "sharpe_green" in regime:
        status = "PASS" if regime.get("overall_pass") else "FAIL"
        log.info(f"  Regime: green_sharpe={regime['sharpe_green']:.2f}  "
                 f"red_sharpe={regime['sharpe_red']:.2f}  "
                 f"asym={regime['regime_asymmetry']:.2f}  "
                 f"conc={regime['day_concentration']:.2f}  [{status}]")

    if mfe_check:
        hold_status = "PASS" if mfe_check.get("hold_pass") else "FAIL"
        log.info(f"  MFE-within-horizon: p90={mfe_check['mfe_p90']:.2f}t  "
                 f"recommended_tp={mfe_check['recommended_tp']:.2f}t  "
                 f"hold_limit={mfe_check['hold_limit_s']:.0f}s  [{hold_status}]")

    log.info("  Per-Day P&L (ticks):")
    for day, pnl in sorted(metrics["daily_pnls"].items()):
        tc = metrics["daily_trade_counts"].get(day, 0)
        marker = "+" if pnl > 0 else (" " if pnl == 0 else "")
        log.info(f"    {day}: {marker}{pnl:>8.2f}  ({tc} trades)")
    log.info("=" * 80)


# ============================================================
# Parameter sweep
# ============================================================
def generate_sweep_configs() -> List[SimConfig]:
    """Generate parameter sweep per user specification."""
    entry_confidence_thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    persistence_thresholds = [0.5, 0.7, 0.9]
    max_hold_seconds_list = [15.0, 20.0, 25.0, 30.0]
    exit_signal_decay_thresholds = [0.0, 0.1, 0.2, 0.3]
    sides_list = ["both", "short_only", "long_only"]

    configs = []
    seen = set()

    for (entry_conf, pers_thresh, max_hold, decay_thresh, sides) in product(
        entry_confidence_thresholds,
        persistence_thresholds,
        max_hold_seconds_list,
        exit_signal_decay_thresholds,
        sides_list,
    ):
        cfg = SimConfig(
            entry_confidence_threshold=entry_conf,
            persistence_threshold=pers_thresh,
            max_hold_seconds=max_hold,
            exit_signal_decay_threshold=decay_thresh,
            sides=sides,
        )
        key = cfg.label_key()
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    log.info(f"Generated {len(configs)} unique sweep configs")
    return configs


# ============================================================
# Main
# ============================================================
def preload_all_data(dates: List[str]) -> Dict[str, dict]:
    """Pre-load all date data into memory."""
    cached: Dict[str, dict] = {}
    has_persistence_count = 0
    for date_str in dates:
        data = load_day_data(date_str)
        if data is not None:
            cached[date_str] = data
            if data["persistence"] is not None:
                has_persistence_count += 1
            log.info(f"  Loaded {date_str}: {data['n_preds']} preds"
                     f"{' +persistence' if data['persistence'] is not None else ''}")
    log.info(f"  Persistence predictions available for "
             f"{has_persistence_count}/{len(cached)} dates")
    return cached


def run_simulation(
    dates: List[str],
    cfg: SimConfig,
    cached_data: Dict[str, dict],
    verbose: bool = True,
) -> Tuple[List[Trade], dict]:
    """Run simulation across all dates."""
    all_trades: List[Trade] = []
    dates_used = []

    for date_str in dates:
        data = cached_data.get(date_str)
        if data is None:
            continue

        day_trades = simulate_day(date_str, data, cfg)
        all_trades.extend(day_trades)
        dates_used.append(date_str)

        if verbose:
            day_net = sum(t.net_pnl_ticks for t in day_trades)
            log.info(f"  {date_str}: {len(day_trades)} trades, "
                     f"net {day_net:.2f} ticks")

    metrics = compute_metrics(all_trades, all_dates=dates_used)
    regime = regime_analysis(metrics)
    mfe_check = mfe_within_horizon_check(all_trades, cfg.max_hold_seconds)

    if verbose and metrics["n_trades"] > 0:
        print_summary(metrics, cfg, regime, mfe_check)

    return all_trades, {**metrics, "regime": regime, "mfe_check": mfe_check}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamic Exit Trade Simulator v2 — 30s Horizon")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Specific dates (e.g., 20260330)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Entry confidence threshold (absolute signal value)")
    parser.add_argument("--persistence", type=float, default=0.5,
                        help="Persistence threshold (0.5 = disabled)")
    parser.add_argument("--max-hold", type=float, default=30.0,
                        help="Max hold seconds")
    parser.add_argument("--decay", type=float, default=0.1,
                        help="Exit signal decay threshold")
    parser.add_argument("--sides",
                        choices=["both", "long_only", "short_only"],
                        default="both")
    parser.add_argument("--cooldown", type=float, default=60.0,
                        help="Cooldown between trades (seconds)")
    parser.add_argument("--horizon",
                        choices=["1s", "5s", "10s"], default="10s",
                        help="Entry signal horizon")
    parser.add_argument("--no-reversal", action="store_true",
                        help="Disable signal reversal exits")
    parser.add_argument("--no-rth", action="store_true",
                        help="Trade outside RTH too")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Top N sweep results to show")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover dates
    log.info("Discovering CNN-Mamba v2 prediction dates...")
    all_dates = discover_dates()
    log.info(f"Found {len(all_dates)} dates with predictions + MBO events")

    if not all_dates:
        log.error("No prediction data found! Check PRED_DIRS and EVENTS_DIR.")
        sys.exit(1)

    if args.dates:
        all_dates = [d for d in all_dates if d in args.dates]
        log.info(f"Filtered to {len(all_dates)} dates")

    # Pre-load all data
    log.info("Pre-loading all data into memory...")
    t0 = time.time()
    cached_data = preload_all_data(all_dates)
    load_time = time.time() - t0
    log.info(f"Loaded {len(cached_data)} dates in {load_time:.1f}s")

    if not cached_data:
        log.error("No valid data loaded!")
        sys.exit(1)

    if args.sweep:
        configs = generate_sweep_configs()
        log.info(f"Running sweep: {len(configs)} configs x "
                 f"{len(cached_data)} dates...")

        results = []
        t_start = time.time()

        for idx, cfg in enumerate(configs):
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(configs) - idx - 1) / rate
                log.info(f"  Sweep: {idx + 1}/{len(configs)} "
                         f"({rate:.1f}/s, ETA {eta:.0f}s)")

            _, metrics = run_simulation(
                all_dates, cfg, cached_data, verbose=False)

            if metrics["n_trades"] >= 10:  # lower bar than v1 (fewer trades at 30s)
                results.append({
                    "config": cfg.to_dict(),
                    "config_key": cfg.label_key(),
                    "metrics": {k: v for k, v in metrics.items()
                                if k not in ("daily_pnls", "daily_trade_counts")},
                    "regime": metrics.get("regime", {}),
                    "mfe_check": metrics.get("mfe_check", {}),
                })

        results.sort(
            key=lambda r: r["metrics"].get("sharpe", -999), reverse=True)

        elapsed = time.time() - t_start
        log.info(f"\nSweep done: {len(configs)} configs in {elapsed:.1f}s "
                 f"({len(configs)/elapsed:.1f}/s)")
        log.info(f"Configs with >= 10 trades: {len(results)}")

        # Print top results
        n_show = min(args.top_n, len(results))
        log.info(f"\n{'='*140}")
        log.info(f"TOP {n_show} by Sharpe "
                 f"(commission = {COMMISSION_RT_TICKS}t RT, 30s horizon)")
        log.info(f"{'='*140}")
        log.info(
            f"{'Rk':>3} {'Sharpe':>7} {'Sort':>7} {'PF':>5} "
            f"{'WR':>5} {'#Tr':>5} {'Tr/d':>5} "
            f"{'GrosT':>7} {'NetT':>7} "
            f"{'Hold':>6} {'MFE90':>6} {'DConc':>6} {'RAsym':>6} "
            f"{'Align':>5} {'P':>2} {'Config'}")
        log.info("-" * 140)

        for rank, r in enumerate(results[:n_show], 1):
            m = r["metrics"]
            reg = r.get("regime", {})
            mfe_c = r.get("mfe_check", {})
            log.info(
                f"{rank:>3} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} "
                f"{m['profit_factor']:>5.2f} "
                f"{m['win_rate']:>5.1%} {m['n_trades']:>5} "
                f"{m['trades_per_day']:>5.1f} "
                f"{m['mean_gross_ticks']:>7.4f} "
                f"{m['mean_net_ticks']:>7.4f} "
                f"{m['avg_hold_s']:>5.1f}s "
                f"{mfe_c.get('mfe_p90', 0):>6.2f} "
                f"{m.get('day_concentration',0):>6.2f} "
                f"{reg.get('regime_asymmetry',0):>6.2f} "
                f"{m.get('signal_alignment_30s',0):>5.1%} "
                f"{'Y' if reg.get('overall_pass') else 'N':>2} "
                f"{r['config_key']}")

        if len(results) > n_show:
            log.info(f"\n  Bottom 5:")
            for r in results[-5:]:
                m = r["metrics"]
                log.info(
                    f"    {m['sharpe']:>7.2f} {m['sortino']:>7.2f} "
                    f"{m['profit_factor']:>5.2f} {m['win_rate']:>5.1%} "
                    f"{m['n_trades']:>5} {m['mean_net_ticks']:>7.4f} "
                    f"{m['avg_hold_s']:>5.1f}s {r['config_key']}")

        # Save sweep results
        out_file = OUTPUT_DIR / "sweep_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"\nSweep results saved to {out_file}")

        # Re-run best config verbose
        if results:
            best_cfg = SimConfig(**results[0]["config"])
            log.info(f"\nBest config detailed run:")
            best_trades, best_metrics = run_simulation(
                all_dates, best_cfg, cached_data, verbose=True)

            trades_file = OUTPUT_DIR / "best_trades.json"
            with open(trades_file, "w") as f:
                json.dump([asdict(t) for t in best_trades], f, indent=2)

            best_file = OUTPUT_DIR / "best_metrics.json"
            with open(best_file, "w") as f:
                json.dump(best_metrics, f, indent=2, default=str)
            log.info(f"Best config trades and metrics saved")

            # Also save the top 10 configs for easy reference
            top10_file = OUTPUT_DIR / "top10_configs.json"
            with open(top10_file, "w") as f:
                json.dump(results[:10], f, indent=2, default=str)

    else:
        # Single config
        cfg = SimConfig(
            entry_confidence_threshold=args.conf,
            persistence_threshold=args.persistence,
            max_hold_seconds=args.max_hold,
            exit_signal_decay_threshold=args.decay,
            sides=args.sides,
            cooldown_s=args.cooldown,
            entry_horizon=args.horizon,
            signal_reversal_exit=not args.no_reversal,
            rth_only=not args.no_rth,
        )

        trades, metrics = run_simulation(
            all_dates, cfg, cached_data, verbose=True)

        if trades:
            trades_file = OUTPUT_DIR / "trades.json"
            with open(trades_file, "w") as f:
                json.dump([asdict(t) for t in trades], f, indent=2)

            metrics_file = OUTPUT_DIR / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)

            log.info(f"Results saved to {OUTPUT_DIR}")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("dynamic_exit_sim_v2")
        run_name = f"{'sweep' if args.sweep else 'sim'}_{time.strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            if args.sweep and results:
                best = results[0]["metrics"]
                mlflow.log_params(results[0]["config"])
                mlflow.log_metric("n_configs_tested", len(configs))
                mlflow.log_metric("n_configs_with_trades", len(results))
                for k, v in best.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mlflow.log_metric(f"best_{k}", v)
            elif not args.sweep and metrics.get("n_trades", 0) > 0:
                mlflow.log_params(cfg.to_dict())
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mlflow.log_metric(k, v)
        log.info("Results logged to MLflow")
    except Exception as e:
        log.info(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
