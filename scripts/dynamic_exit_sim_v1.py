#!/usr/bin/env python3
"""
Dynamic Exit Trade Simulator v1 — HC #515
==========================================
Continuous pressure monitoring during trades instead of static hold times.

Data sources:
  - CNN-Mamba v2 OOT per-date predictions (46 days, stride=250 events)
  - Smooth pressure targets (EOFI/NTPS/PDI/TIA) aligned to MBO events
  - MBO events smart_v3 for timestamps and order flow

Cost model (HC #512):
  - Bid/ask fill: commission only = 0.376 ticks RT
  - NEVER use 2.0, 1.752, etc.

Validation (HC #428):
  - 46 OOT days (> 40 required)
  - Per-day Sharpe/PF/WR
  - Regime asymmetry test
  - Day concentration cap <= 0.70

Price path reconstruction:
  - labels_1s[event_i] = actual 1s forward mid-price change from event i
  - Between prediction points separated by dt seconds, price change ~=
    labels_1s[event_i] * min(dt, 1.0) (proportional scaling for sub-1s gaps)
  - For dt > 1s (overnight gaps, etc.), we cap the multiplier at 1.0
    since the label only covers 1s of forward movement.

Usage:
    python scripts/dynamic_exit_sim_v1.py                  # Default config
    python scripts/dynamic_exit_sim_v1.py --sweep           # Parameter sweep
    python scripts/dynamic_exit_sim_v1.py --dates 20260330  # Specific date(s)
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dynamic_exit_sim")

# ============================================================
# Paths
# ============================================================
BASE = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = BASE / "output" / "cnn_mamba_v2_all_oot"
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
PRESSURE_DIR = BASE / "data" / "processed" / "smooth_pressure_targets"
OUTPUT_DIR = BASE / "output" / "dynamic_exit_sim_v1"

# ============================================================
# Constants
# ============================================================
TICK_SIZE = 0.25       # ES tick = 0.25 points
TICK_VALUE = 12.50     # $12.50 per tick
COMMISSION_RT_TICKS = 0.376  # AMP/Rithmic RT commission in ticks

# CNN-Mamba v2 all_oot format
WINDOW_SIZE = 3000     # events per input window
STRIDE = 250           # events between consecutive predictions (NOT 250ms)

# MBO smart_v3 feature columns (25-feature layout)
COL_PRICE_REL = 3      # (event_price - mid) / tick_size, NOT cumulative
COL_SPREAD = 5         # spread in ticks
COL_OFI_SHORT = 22     # ofi_short_100 (z-scored)
COL_OFI_LONG = 23      # ofi_long_2000 (z-scored)
COL_OFI_ACCEL = 24     # ofi_acceleration

# RTH boundaries in nanoseconds from midnight UTC
# RTH: 9:30 ET to 16:00 ET
# EDT (Mar-Nov): 9:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC
# EST (Nov-Mar): 9:30 ET = 14:30 UTC, 16:00 ET = 21:00 UTC
# We use loose bounds to cover both: 13:30-21:00 UTC
RTH_START_NS = 13 * 3600 * 1_000_000_000 + 30 * 60 * 1_000_000_000  # 13:30 UTC
RTH_END_NS = 21 * 3600 * 1_000_000_000                                # 21:00 UTC


# ============================================================
# Configuration
# ============================================================
@dataclass
class SimConfig:
    """All tunable parameters for the dynamic exit simulator."""
    # --- Entry ---
    confidence_pctile: float = 0.90    # top N% = 1 - this (0.90 = top 10%)
    entry_horizon: str = "1s"          # horizon for entry signal (1s/5s/10s)
    sides: str = "both"                # "long", "short", "both"

    # --- Dynamic exit: signal monitoring ---
    # NOTE: CNN-Mamba 1s/5s/10s predictions have near-zero autocorrelation
    # (sign flips ~43-50% of the time). Signal reversal / fade exits are
    # unreliable as primary dynamic exit. Pressure (EOFI) has autocorr ~0.42
    # at lag 1, making it the proper dynamic exit signal.
    signal_fade_exit: bool = False     # exit when signal fades (default OFF)
    signal_reversal_exit: bool = False # exit when signal flips (default OFF)
    fade_n: int = 8                    # consecutive weak-signal cycles before exit

    # --- Dynamic exit: pressure monitoring ---
    pressure_exit: bool = True         # exit when pressure reverses
    pressure_reversal_n: int = 5       # consecutive adverse pressure cycles
    pressure_feature: str = "eofi"     # pressure feature to use

    # --- Static bounds (ceiling, not target) ---
    tp_ticks: float = 6.0             # take-profit ceiling
    sl_ticks: float = 8.0             # stop-loss ceiling
    max_hold_s: float = 60.0          # absolute max hold

    # --- Trade management ---
    min_hold_preds: int = 2           # minimum predictions before exit
    cooldown_s: float = 30.0          # minimum SECONDS between trades
    warmup_preds: int = 50            # skip initial predictions
    rth_only: bool = True             # trade only during RTH

    def to_dict(self):
        return asdict(self)

    def label_key(self):
        top_pct = int((1 - self.confidence_pctile) * 100)
        return (f"top{top_pct}pct_{self.sides}"
                f"_tp{self.tp_ticks:.0f}_sl{self.sl_ticks:.0f}"
                f"_hold{self.max_hold_s:.0f}s"
                f"_pressN{self.pressure_reversal_n}"
                f"_fadeN{self.fade_n}")


# ============================================================
# Trade record
# ============================================================
@dataclass
class Trade:
    date: str = ""
    side: str = ""
    entry_time_ns: int = 0
    exit_time_ns: int = 0
    entry_price: float = 0.0   # cumulative price at entry (ticks from day start)
    exit_price: float = 0.0
    gross_pnl_ticks: float = 0.0
    net_pnl_ticks: float = 0.0
    hold_duration_s: float = 0.0
    n_predictions: int = 0
    exit_reason: str = ""
    entry_signal: float = 0.0
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0


# ============================================================
# Data loading
# ============================================================
def discover_dates() -> List[str]:
    """Find all per-date prediction files with matching MBO events."""
    dates = []
    for p in sorted(PRED_DIR.glob("*_predictions.npz")):
        name = p.stem.replace("_predictions", "")
        if name.startswith("fold_") or name.startswith("concat"):
            continue
        if (EVENTS_DIR / f"{name}_mbo_events.npz").exists():
            dates.append(name)
    return dates


def _time_of_day_ns(ts_ns: int) -> int:
    """Extract nanoseconds since midnight UTC from an epoch nanosecond timestamp."""
    # 86400 seconds = 1 day
    return ts_ns % (86400 * 1_000_000_000)


def load_day_data(date_str: str) -> Optional[dict]:
    """Load predictions, MBO events, and pressure data for one date.

    Returns aligned arrays at prediction-point resolution, plus a cumulative
    price path reconstructed from labels.
    """
    pred_file = PRED_DIR / f"{date_str}_predictions.npz"
    event_file = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    pressure_file = PRESSURE_DIR / f"{date_str}_pressure.npz"

    if not pred_file.exists() or not event_file.exists():
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
    mbo_labels_1s = mbo["labels_1s"]
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

    # Build cumulative mid-price path
    # labels_1s[event_i] = actual 1s forward return in ticks
    # Between consecutive prediction points with time gap dt:
    #   price_change ~= labels_1s[event_i] * min(dt/1.0, 1.0)
    # This scales the 1s return proportionally when preds are < 1s apart,
    # and caps at the full 1s return when preds are > 1s apart.
    pred_labels_1s = mbo_labels_1s[pred_event_indices].astype(np.float64)
    pred_labels_1s = np.nan_to_num(pred_labels_1s, nan=0.0)

    dt_ns = np.diff(pred_timestamps)
    dt_s = dt_ns / 1e9

    # Scale factor: min(dt/1.0, 1.0) — cap at 1.0 for gaps > 1s
    # For gaps > 1s, the label only covers the first 1s of movement
    scale = np.clip(dt_s, 0.0, 1.0)
    inter_pred_returns = pred_labels_1s[:-1] * scale

    mid_prices = np.zeros(n_preds, dtype=np.float64)
    mid_prices[1:] = np.cumsum(inter_pred_returns)

    # Extract OFI features at prediction points
    ofi_short = events[pred_event_indices, COL_OFI_SHORT].copy()

    # Load pressure data (event-aligned with MBO)
    pressure_data = {}
    if pressure_file.exists():
        try:
            pres = np.load(str(pressure_file), allow_pickle=True)
            for feat in ["eofi", "ntps", "pdi", "tia"]:
                if feat in pres:
                    pressure_data[feat] = pres[feat][pred_event_indices].copy()
        except Exception as e:
            log.warning(f"Pressure data load failed for {date_str}: {e}")

    return {
        "predictions": predictions,
        "labels": labels,
        "timestamps": pred_timestamps,
        "mid_prices": mid_prices.astype(np.float32),
        "ofi_short": ofi_short,
        "pressure": pressure_data,
        "n_preds": n_preds,
    }


# ============================================================
# Core simulation
# ============================================================
def simulate_day(date_str: str, data: dict, cfg: SimConfig) -> List[Trade]:
    """Run dynamic-exit trade simulation for one day."""
    predictions = data["predictions"]  # (n, 3)
    timestamps = data["timestamps"]
    mid_prices = data["mid_prices"]
    pressure = data["pressure"]
    n = data["n_preds"]

    # Select horizon for directional signal
    horizon_idx = {"1s": 0, "5s": 1, "10s": 2}[cfg.entry_horizon]
    signal = predictions[:, horizon_idx]

    # Get pressure feature if available
    has_pressure = cfg.pressure_exit and cfg.pressure_feature in pressure
    if has_pressure:
        pressure_signal = pressure[cfg.pressure_feature]
    else:
        pressure_signal = np.zeros(n, dtype=np.float32)

    # RTH filter: identify which prediction indices fall within RTH
    if cfg.rth_only:
        tod_ns = np.array([_time_of_day_ns(int(t)) for t in timestamps])
        rth_mask = (tod_ns >= RTH_START_NS) & (tod_ns <= RTH_END_NS)
    else:
        rth_mask = np.ones(n, dtype=bool)

    # Compute entry threshold from signal distribution within RTH
    rth_signal = signal[rth_mask & (np.arange(n) >= cfg.warmup_preds)]
    rth_signal = rth_signal[np.isfinite(rth_signal)]
    if len(rth_signal) < 50:
        return []
    abs_signal = np.abs(rth_signal)
    entry_threshold = float(np.percentile(abs_signal, cfg.confidence_pctile * 100))

    # Fade threshold: signal below this = edge is gone (40% of entry threshold)
    fade_threshold = entry_threshold * 0.4
    # Reversal threshold: signal must flip AND exceed this to trigger reversal exit
    # Use a meaningful fraction of entry threshold (not just crossing zero)
    reversal_threshold = entry_threshold * 0.3

    trades: List[Trade] = []
    state = "FLAT"
    entry_idx = 0
    entry_price = 0.0
    entry_time_ns = 0
    entry_signal_val = 0.0
    trade_side = ""
    fade_count = 0
    pressure_adverse_count = 0
    n_preds_in_trade = 0
    best_price = 0.0
    worst_price = 0.0
    last_exit_time_ns = 0

    for i in range(cfg.warmup_preds, n):
        if not rth_mask[i]:
            # If we're in a trade and go outside RTH, force exit
            if state != "FLAT":
                mid = float(mid_prices[i])
                t_ns = int(timestamps[i])
                # Force exit
                if trade_side == "LONG":
                    gross_pnl = mid - entry_price
                    mfe = best_price - entry_price
                    mae = entry_price - worst_price
                else:
                    gross_pnl = entry_price - mid
                    mfe = entry_price - best_price
                    mae = worst_price - entry_price

                trades.append(Trade(
                    date=date_str, side=trade_side,
                    entry_time_ns=entry_time_ns, exit_time_ns=t_ns,
                    entry_price=float(entry_price), exit_price=float(mid),
                    gross_pnl_ticks=float(gross_pnl),
                    net_pnl_ticks=float(gross_pnl - COMMISSION_RT_TICKS),
                    hold_duration_s=(t_ns - entry_time_ns) / 1e9,
                    n_predictions=n_preds_in_trade,
                    exit_reason="rth_end", entry_signal=entry_signal_val,
                    mfe_ticks=float(mfe), mae_ticks=float(mae),
                ))
                state = "FLAT"
                last_exit_time_ns = t_ns
            continue

        sig = signal[i]
        t_ns = int(timestamps[i])
        mid = float(mid_prices[i])
        pres = float(pressure_signal[i]) if has_pressure else 0.0

        if not np.isfinite(sig):
            continue

        if state == "FLAT":
            # Time-based cooldown
            if t_ns - last_exit_time_ns < cfg.cooldown_s * 1e9:
                continue

            # Entry conditions
            go_long = (cfg.sides in ("both", "long") and sig > entry_threshold)
            go_short = (cfg.sides in ("both", "short") and sig < -entry_threshold)

            # Pressure confirmation
            if has_pressure and np.isfinite(pres):
                if go_long and pres < 0:
                    go_long = False
                if go_short and pres > 0:
                    go_short = False

            if go_long:
                state = "LONG"
                trade_side = "LONG"
            elif go_short:
                state = "SHORT"
                trade_side = "SHORT"

            if state != "FLAT":
                entry_idx = i
                entry_price = mid
                entry_time_ns = t_ns
                entry_signal_val = float(sig)
                fade_count = 0
                pressure_adverse_count = 0
                n_preds_in_trade = 0
                best_price = mid
                worst_price = mid

        else:
            # In position — evaluate exit
            n_preds_in_trade += 1
            hold_s = (t_ns - entry_time_ns) / 1e9

            # Track MFE/MAE
            if trade_side == "LONG":
                best_price = max(best_price, mid)
                worst_price = min(worst_price, mid)
            else:
                best_price = min(best_price, mid)
                worst_price = max(worst_price, mid)

            # Min hold
            if n_preds_in_trade < cfg.min_hold_preds:
                continue

            # Current P&L
            if trade_side == "LONG":
                current_pnl = mid - entry_price
            else:
                current_pnl = entry_price - mid

            # --- EXIT CONDITIONS ---
            exit_reason = ""

            # (e) Stop-loss
            if current_pnl <= -cfg.sl_ticks:
                exit_reason = "stop_loss"

            # (c) Take-profit
            elif current_pnl >= cfg.tp_ticks:
                exit_reason = "take_profit"

            # (d) Max hold time
            elif hold_s >= cfg.max_hold_s:
                exit_reason = "timeout"

            # (b) Signal reversal — model now predicts OPPOSITE direction
            elif cfg.signal_reversal_exit:
                if trade_side == "LONG" and sig < -reversal_threshold:
                    exit_reason = "signal_reversal"
                elif trade_side == "SHORT" and sig > reversal_threshold:
                    exit_reason = "signal_reversal"

            # (b') Signal fade — signal magnitude drops (edge gone)
            if not exit_reason and cfg.signal_fade_exit:
                if trade_side == "LONG":
                    is_fading = sig < fade_threshold
                else:
                    is_fading = sig > -fade_threshold

                if is_fading:
                    fade_count += 1
                else:
                    fade_count = 0

                if fade_count >= cfg.fade_n:
                    exit_reason = "signal_fade"

            # (a) Pressure reversal
            if not exit_reason and has_pressure and np.isfinite(pres):
                if trade_side == "LONG":
                    adverse = pres < 0
                else:
                    adverse = pres > 0

                if adverse:
                    pressure_adverse_count += 1
                else:
                    pressure_adverse_count = 0

                if pressure_adverse_count >= cfg.pressure_reversal_n:
                    exit_reason = "pressure_reversal"

            # --- EXECUTE EXIT ---
            if exit_reason:
                exit_price = mid

                # Clamp TP/SL at trigger level
                if exit_reason == "stop_loss":
                    if trade_side == "LONG":
                        exit_price = max(exit_price, entry_price - cfg.sl_ticks)
                    else:
                        exit_price = min(exit_price, entry_price + cfg.sl_ticks)
                elif exit_reason == "take_profit":
                    if trade_side == "LONG":
                        exit_price = min(exit_price, entry_price + cfg.tp_ticks)
                    else:
                        exit_price = max(exit_price, entry_price - cfg.tp_ticks)

                if trade_side == "LONG":
                    gross_pnl = exit_price - entry_price
                    mfe = best_price - entry_price
                    mae = entry_price - worst_price
                else:
                    gross_pnl = entry_price - exit_price
                    mfe = entry_price - best_price
                    mae = worst_price - entry_price

                net_pnl = gross_pnl - COMMISSION_RT_TICKS

                trades.append(Trade(
                    date=date_str, side=trade_side,
                    entry_time_ns=entry_time_ns, exit_time_ns=t_ns,
                    entry_price=float(entry_price), exit_price=float(exit_price),
                    gross_pnl_ticks=float(gross_pnl),
                    net_pnl_ticks=float(net_pnl),
                    hold_duration_s=float(hold_s),
                    n_predictions=n_preds_in_trade,
                    exit_reason=exit_reason,
                    entry_signal=entry_signal_val,
                    mfe_ticks=float(mfe), mae_ticks=float(mae),
                ))

                state = "FLAT"
                last_exit_time_ns = t_ns

    return trades


# ============================================================
# Metrics
# ============================================================
def compute_metrics(trades: List[Trade], all_dates: List[str] = None) -> dict:
    """Compute risk-adjusted performance metrics."""
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
    mfes = [t.mfe_ticks for t in trades]
    maes = [t.mae_ticks for t in trades]

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

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": n / max(n_days, 1),
        "win_rate": win_rate,
        "total_net_ticks": float(np.sum(net_pnls)),
        "total_net_dollars": float(np.sum(net_pnls) * TICK_VALUE),
        "mean_net_ticks": float(np.mean(net_pnls)),
        "mean_gross_ticks": float(np.mean(gross_pnls)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": profit_factor,
        "avg_hold_s": float(np.mean(holds)),
        "median_hold_s": float(np.median(holds)),
        "avg_mfe": float(np.mean(mfes)),
        "avg_mae": float(np.mean(maes)),
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
        "daily_pnls": {k: float(v) for k, v in sorted(day_pnl.items())},
        "daily_trade_counts": {k: v for k, v in sorted(day_trades.items())},
    }


def regime_analysis(metrics: dict) -> dict:
    """HC #428 R1: Regime-agnostic validation.

    Classify days as green/red/flat and compute per-regime Sharpe.
    """
    daily = metrics.get("daily_pnls", {})
    if not daily or len(daily) < 10:
        return {"overall_pass": False, "reason": "too few days"}

    green_pnls = [v for v in daily.values() if v > 0]
    red_pnls = [v for v in daily.values() if v < 0]

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
        "n_flat_days": sum(1 for v in daily.values() if v == 0.0),
        "sharpe_green": float(sharpe_green),
        "sharpe_red": float(sharpe_red),
        "regime_asymmetry": float(regime_asym),
        "regime_pass": regime_pass,
        "day_concentration": metrics.get("day_concentration", 0.0),
        "concentration_pass": conc_pass,
        "overall_pass": regime_pass and conc_pass,
    }


def print_summary(metrics: dict, cfg: SimConfig, regime: dict = None):
    """Print formatted summary."""
    if metrics["n_trades"] == 0:
        log.info("No trades generated.")
        return

    log.info("=" * 75)
    log.info(f"DYNAMIC EXIT SIM — {cfg.label_key()}")
    log.info("=" * 75)
    log.info(f"  Trades: {metrics['n_trades']} across {metrics['n_days']} days "
             f"({metrics['trades_per_day']:.1f}/day)")
    log.info(f"  Win Rate: {metrics['win_rate']:.1%}")
    log.info(f"  Net P&L: {metrics['total_net_ticks']:.2f} ticks "
             f"(${metrics['total_net_dollars']:.2f})")
    log.info(f"  Mean Net: {metrics['mean_net_ticks']:.4f} ticks/trade "
             f"(gross: {metrics['mean_gross_ticks']:.4f})")
    log.info(f"  Sharpe: {metrics['sharpe']:.2f}  |  Sortino: {metrics['sortino']:.2f}  "
             f"|  PF: {metrics['profit_factor']:.2f}")
    log.info(f"  Avg Hold: {metrics['avg_hold_s']:.1f}s  |  "
             f"Median Hold: {metrics['median_hold_s']:.1f}s")
    log.info(f"  MFE: {metrics['avg_mfe']:.2f}  |  MAE: {metrics['avg_mae']:.2f}  "
             f"|  MFE/MAE: {metrics['mfe_mae_ratio']:.2f}")
    log.info(f"  Longs: {metrics['n_longs']} (WR {metrics['long_wr']:.1%}, "
             f"avg {metrics['long_mean_net']:.4f}t)  |  "
             f"Shorts: {metrics['n_shorts']} (WR {metrics['short_wr']:.1%}, "
             f"avg {metrics['short_mean_net']:.4f}t)")
    log.info(f"  Exit Reasons: {metrics['exit_reasons']}")
    log.info(f"  Days: {metrics['profitable_days']} green / "
             f"{metrics['losing_days']} red / {metrics['flat_days']} flat  |  "
             f"Day Conc: {metrics['day_concentration']:.2f}")

    if regime and "sharpe_green" in regime:
        log.info(f"  Regime: green_sharpe={regime['sharpe_green']:.2f} "
                 f"red_sharpe={regime['sharpe_red']:.2f} "
                 f"asym={regime['regime_asymmetry']:.2f} "
                 f"{'PASS' if regime.get('overall_pass') else 'FAIL'}")

    log.info("  Per-Day P&L (ticks):")
    for day, pnl in sorted(metrics["daily_pnls"].items()):
        tc = metrics["daily_trade_counts"].get(day, 0)
        marker = "+" if pnl > 0 else (" " if pnl == 0 else "")
        log.info(f"    {day}: {marker}{pnl:>8.2f}  ({tc} trades)")
    log.info("=" * 75)


# ============================================================
# Parameter sweep
# ============================================================
def generate_sweep_configs() -> List[SimConfig]:
    """Generate parameter sweep per user spec."""
    configs = []

    confidence_thresholds = [0.95, 0.90, 0.80]   # top 5%, 10%, 20%
    pressure_reversal_ns = [3, 5, 8, 12]
    max_hold_seconds = [30, 60, 120, 300]
    tp_ticks_list = [4, 6, 8, 12]
    sl_ticks_list = [4, 8, 12, 16]

    # Sweep 1: confidence x pressure_n x max_hold
    for conf_pct in confidence_thresholds:
        for press_n in pressure_reversal_ns:
            for max_hold in max_hold_seconds:
                configs.append(SimConfig(
                    confidence_pctile=conf_pct,
                    pressure_reversal_n=press_n,
                    max_hold_s=float(max_hold),
                    tp_ticks=6.0, sl_ticks=8.0,
                    sides="both",
                ))

    # Sweep 2: TP/SL grid
    for tp in tp_ticks_list:
        for sl in sl_ticks_list:
            for conf_pct in [0.95, 0.90]:
                configs.append(SimConfig(
                    confidence_pctile=conf_pct,
                    tp_ticks=float(tp), sl_ticks=float(sl),
                    max_hold_s=60.0, pressure_reversal_n=5,
                    sides="both",
                ))

    # Sweep 3: Short-only
    for conf_pct in confidence_thresholds:
        for press_n in [3, 5, 8]:
            for tp in [4, 6, 8]:
                configs.append(SimConfig(
                    confidence_pctile=conf_pct,
                    pressure_reversal_n=press_n,
                    tp_ticks=float(tp), sl_ticks=8.0,
                    max_hold_s=60.0, sides="short",
                ))

    # Sweep 4: No pressure (baseline)
    for conf_pct in [0.95, 0.90, 0.80]:
        configs.append(SimConfig(
            confidence_pctile=conf_pct,
            pressure_exit=False,
            tp_ticks=6.0, sl_ticks=8.0,
            max_hold_s=60.0, sides="both",
        ))

    # Sweep 5: Fade_n sensitivity
    for fade_n in [3, 5, 8, 12, 20]:
        for conf_pct in [0.95, 0.90]:
            configs.append(SimConfig(
                confidence_pctile=conf_pct,
                fade_n=fade_n,
                tp_ticks=6.0, sl_ticks=8.0,
                max_hold_s=60.0, pressure_reversal_n=5,
                sides="both",
            ))

    # Sweep 6: Cooldown sensitivity
    for cooldown in [10.0, 20.0, 30.0, 60.0]:
        for conf_pct in [0.95, 0.90]:
            configs.append(SimConfig(
                confidence_pctile=conf_pct,
                cooldown_s=cooldown,
                tp_ticks=6.0, sl_ticks=8.0,
                max_hold_s=60.0, pressure_reversal_n=5,
                sides="both",
            ))

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = c.label_key() + f"_cd{c.cooldown_s:.0f}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ============================================================
# Main
# ============================================================
def preload_all_data(dates: List[str]) -> Dict[str, dict]:
    """Pre-load all date data into memory."""
    cached: Dict[str, dict] = {}
    for date_str in dates:
        data = load_day_data(date_str)
        if data is not None:
            cached[date_str] = data
            log.info(f"  Loaded {date_str}: {data['n_preds']} predictions")
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

    if verbose and metrics["n_trades"] > 0:
        print_summary(metrics, cfg, regime)

    return all_trades, {**metrics, "regime": regime}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dynamic Exit Trade Simulator v1 (HC #515)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter sweep")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Specific dates (e.g., 20260330)")
    parser.add_argument("--conf", type=float, default=0.90,
                        help="Confidence percentile (0.90 = top 10%%)")
    parser.add_argument("--tp", type=float, default=6.0,
                        help="Take-profit ticks")
    parser.add_argument("--sl", type=float, default=8.0,
                        help="Stop-loss ticks")
    parser.add_argument("--max-hold", type=float, default=60.0,
                        help="Max hold seconds")
    parser.add_argument("--sides",
                        choices=["both", "long", "short"], default="both")
    parser.add_argument("--pressure-n", type=int, default=5,
                        help="Pressure reversal consecutive cycles")
    parser.add_argument("--fade-n", type=int, default=5,
                        help="Signal fade consecutive cycles")
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="Cooldown between trades (seconds)")
    parser.add_argument("--no-pressure", action="store_true",
                        help="Disable pressure exits")
    parser.add_argument("--no-rth", action="store_true",
                        help="Trade outside RTH too")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Top N sweep results to show")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover dates
    log.info("Discovering CNN-Mamba v2 per-date predictions...")
    all_dates = discover_dates()
    log.info(f"Found {len(all_dates)} dates with predictions + MBO events")

    if not all_dates:
        log.error("No prediction data found!")
        sys.exit(1)

    if args.dates:
        all_dates = [d for d in all_dates if d in args.dates]
        log.info(f"Filtered to {len(all_dates)} dates")

    # Pre-load all data
    log.info("Pre-loading all data into memory...")
    t0 = time.time()
    cached_data = preload_all_data(all_dates)
    log.info(f"Loaded {len(cached_data)} dates in {time.time() - t0:.1f}s")

    if args.sweep:
        configs = generate_sweep_configs()
        log.info(f"Running sweep: {len(configs)} configs x "
                 f"{len(cached_data)} dates...")

        results = []
        t_start = time.time()

        for idx, cfg in enumerate(configs):
            if (idx + 1) % 25 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(configs) - idx - 1) / rate
                log.info(f"  Sweep: {idx + 1}/{len(configs)} "
                         f"({rate:.1f}/s, ETA {eta:.0f}s)")

            _, metrics = run_simulation(
                all_dates, cfg, cached_data, verbose=False)

            if metrics["n_trades"] >= 20:
                results.append({
                    "config": cfg.to_dict(),
                    "config_key": cfg.label_key(),
                    "metrics": {k: v for k, v in metrics.items()
                                if k not in ("daily_pnls", "daily_trade_counts")},
                    "regime": metrics.get("regime", {}),
                })

        results.sort(
            key=lambda r: r["metrics"].get("sharpe", -999), reverse=True)

        elapsed = time.time() - t_start
        log.info(f"\nSweep done: {len(configs)} configs in {elapsed:.1f}s "
                 f"({len(configs)/elapsed:.1f}/s)")

        # Print top results
        n_show = min(args.top_n, len(results))
        log.info(f"\n{'='*130}")
        log.info(f"TOP {n_show} by Sharpe "
                 f"(commission = {COMMISSION_RT_TICKS} ticks RT, RTH only)")
        log.info(f"{'='*130}")
        log.info(
            f"{'Rk':>3} {'Sharpe':>7} {'Sort':>7} {'PF':>5} "
            f"{'WR':>5} {'#Tr':>5} {'Tr/d':>5} {'NetTr':>7} "
            f"{'Hold':>6} {'DConc':>6} {'RAsym':>6} "
            f"{'P':>2} {'Config'}")
        log.info("-" * 130)

        for rank, r in enumerate(results[:n_show], 1):
            m = r["metrics"]
            reg = r.get("regime", {})
            log.info(
                f"{rank:>3} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} "
                f"{m['profit_factor']:>5.2f} "
                f"{m['win_rate']:>5.1%} {m['n_trades']:>5} "
                f"{m['trades_per_day']:>5.1f} "
                f"{m['mean_net_ticks']:>7.4f} "
                f"{m['avg_hold_s']:>5.1f}s "
                f"{m.get('day_concentration',0):>6.2f} "
                f"{reg.get('regime_asymmetry',0):>6.2f} "
                f"{'Y' if reg.get('overall_pass') else 'N':>2} "
                f"{r['config_key']}")

        if len(results) > n_show:
            log.info(f"\nBottom 5:")
            for r in results[-5:]:
                m = r["metrics"]
                log.info(
                    f"    {m['sharpe']:>7.2f} {m['sortino']:>7.2f} "
                    f"{m['profit_factor']:>5.2f} {m['win_rate']:>5.1%} "
                    f"{m['n_trades']:>5} {m['mean_net_ticks']:>7.4f} "
                    f"{m['avg_hold_s']:>5.1f}s {r['config_key']}")

        # Save results
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

    else:
        # Single config
        cfg = SimConfig(
            confidence_pctile=args.conf,
            tp_ticks=args.tp, sl_ticks=args.sl,
            max_hold_s=args.max_hold,
            sides=args.sides,
            pressure_reversal_n=args.pressure_n,
            fade_n=args.fade_n,
            cooldown_s=args.cooldown,
            pressure_exit=not args.no_pressure,
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
        mlflow.set_experiment("dynamic_exit_sim_v1")
        run_name = f"{'sweep' if args.sweep else 'sim'}_{time.strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            if args.sweep and results:
                best = results[0]["metrics"]
                mlflow.log_params(results[0]["config"])
                mlflow.log_metric("n_configs_tested", len(configs))
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
