#!/usr/bin/env python3
"""
Dynamic Exit Trade Simulator v1 — CNN-Mamba v2 Predictions
==========================================================
Simulates trading with DYNAMIC exits governed by continuous signal monitoring.
No static hold times. Hold duration is VARIABLE based on real-time prediction updates.

Per HC #515: No static hold. Exit when signal says to exit.
Per HC #428 R2: TP/SL/hold bounded by model's predictive horizon.
Per HC #511/512: Commission = 0.376 ticks RT. Spread is NOT a separate cost.

Data:
  - CNN-Mamba v2 OOT predictions from walk-forward folds
  - MBO events (smart_v3 25-feature format) for timestamps, prices, order flow

Usage:
    python scripts/dynamic_exit_simulator_v1.py                  # Default config
    python scripts/dynamic_exit_simulator_v1.py --sweep           # Parameter sweep
    python scripts/dynamic_exit_simulator_v1.py --dates 20260223  # Specific date(s)
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
log = logging.getLogger("dynamic_exit_sim")

# ============================================================
# Paths
# ============================================================
BASE = Path("/home/jupiter/Lvl3Quant")
PRED_DIR = BASE / "output" / "cnn_mamba_v2_smart_v3_mar"
EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = BASE / "output" / "dynamic_exit_sim_v1"

# ============================================================
# Constants (canonical — CLAUDE.md)
# ============================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376  # AMP/Rithmic round-trip

# Model architecture constants — CNN-Mamba v2 on smart_v3
WINDOW_SIZE = 3000   # events per input window
STRIDE = 500         # events between consecutive predictions
# Each prediction point maps to event index: WINDOW_SIZE - 1 + pred_idx * STRIDE

# MBO feature columns (smart_v3 25-feature layout)
# 0: time_delta_log, 1: event_type_id, 2: side_id, 3: price_rel_ticks
# 4: qty_log, 5: spread_ticks
# Derived: 6-14 (v1 features), 15-21 (v2 features)
# 22: ofi_short_100 (z-scored), 23: ofi_long_2000 (z-scored), 24: ofi_acceleration
COL_PRICE_REL = 3
COL_SPREAD = 5
COL_OFI_SHORT = 22
COL_OFI_LONG = 23
COL_OFI_ACCEL = 24
COL_ROLLING_OFI_500 = 7  # derived feature 1 (offset by 6 base features)


# ============================================================
# Configuration
# ============================================================
@dataclass
class SimConfig:
    """All tunable parameters for the dynamic exit simulator."""
    # Entry
    entry_threshold: float = 0.50       # |pred_1s| threshold to enter
    entry_horizon: str = "1s"           # which horizon to use for entry signal (1s/5s/10s)
    sides: str = "both"                 # "long", "short", "both"

    # Dynamic exit — signal monitoring
    exit_threshold: float = 0.15        # exit when |pred| drops below this (edge gone)
    reversal_exit: bool = True          # exit when prediction flips direction
    fade_n: int = 3                     # consecutive weak predictions before exit

    # Dynamic exit — order flow pressure
    pressure_reversal: bool = True      # exit when OFI reverses against position
    pressure_threshold: float = 1.5     # |OFI z-score| threshold for pressure reversal

    # Static bounds (ceiling, not target)
    tp_ticks: float = 4.0              # take-profit ceiling
    sl_ticks: float = 2.0              # stop-loss ceiling
    max_hold_s: float = 30.0           # absolute max hold (HC #428: <= 1.5x horizon)

    # Misc
    min_hold_preds: int = 2            # minimum predictions to hold before allowing exit
    cooldown_preds: int = 10           # minimum predictions between trades
    warmup_preds: int = 100            # skip initial predictions for stability

    def to_dict(self):
        return asdict(self)

    def label_key(self):
        """Short label for sweep results."""
        h = {"1s": 0, "5s": 1, "10s": 2}[self.entry_horizon]
        return (f"ent{self.entry_threshold:.2f}_exit{self.exit_threshold:.2f}"
                f"_h{h}_tp{self.tp_ticks:.1f}_sl{self.sl_ticks:.1f}"
                f"_fade{self.fade_n}_press{self.pressure_threshold:.1f}")


# ============================================================
# Trade record
# ============================================================
@dataclass
class Trade:
    date: str = ""
    side: str = ""              # "LONG" or "SHORT"
    entry_time_ns: int = 0
    exit_time_ns: int = 0
    entry_price_ticks: float = 0.0   # mid price at entry in tick units
    exit_price_ticks: float = 0.0
    gross_pnl_ticks: float = 0.0
    net_pnl_ticks: float = 0.0
    hold_duration_s: float = 0.0
    n_predictions: int = 0
    exit_reason: str = ""
    entry_signal: float = 0.0   # prediction value at entry
    mfe_ticks: float = 0.0      # max favorable excursion
    mae_ticks: float = 0.0      # max adverse excursion


# ============================================================
# Data loading
# ============================================================
def discover_folds() -> List[Tuple[str, Path]]:
    """Find all fold prediction files and extract OOT dates."""
    folds = []
    for p in sorted(PRED_DIR.glob("fold_*_oot_predictions.npz")):
        try:
            d = np.load(str(p), allow_pickle=True)
            oot_file = str(d["oot_files"][0])
            # Extract date from path like '.../20260223_mbo_events.npz'
            date_str = oot_file.split("/")[-1].replace("_mbo_events.npz", "")
            folds.append((date_str, p))
        except Exception as e:
            log.warning(f"Skipping {p.name}: {e}")
    return folds


def load_fold_data(fold_path: Path, date_str: str) -> Optional[dict]:
    """Load predictions and corresponding MBO events for one fold/date.

    Price reconstruction: We build a cumulative price path from label_1s returns.
    label_1s[i] = actual price change (in ticks) over next 1 second from prediction
    point i. Since inter-prediction time averages ~1.1s (close to 1s horizon),
    we use label_1s as the inter-prediction return. This is approximately correct
    and unbiased (mean return near zero).

    Returns dict with:
        predictions: (n_preds, 3)  -- model output [1s, 5s, 10s]
        labels: (n_preds, 3)       -- actual returns [1s, 5s, 10s] in ticks
        timestamps: (n_preds,)     -- nanosecond timestamps at each prediction point
        mid_prices: (n_preds,)     -- cumulative price path (ticks from start)
        ofi_short: (n_preds,)      -- short-window OFI z-score at each prediction point
        ofi_long: (n_preds,)       -- long-window OFI z-score at each prediction point
        ofi_accel: (n_preds,)      -- OFI acceleration at each prediction point
    """
    # Load predictions
    pred_data = np.load(str(fold_path), allow_pickle=True)
    predictions = pred_data["predictions"]  # (n_preds, 3)
    labels = pred_data["labels"]            # (n_preds, 3)
    n_preds = predictions.shape[0]

    # Load MBO events for this date
    event_file = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    if not event_file.exists():
        log.warning(f"MBO events not found for {date_str}")
        return None

    try:
        mbo = np.load(str(event_file), mmap_mode="r")
    except Exception as e:
        log.warning(f"Failed to load MBO events for {date_str}: {e}")
        return None

    n_events = len(mbo["timestamps"])
    events = mbo["events"]       # (n_events, 25)
    timestamps = mbo["timestamps"]  # (n_events,) int64 nanoseconds

    # Compute event indices for each prediction
    # prediction i corresponds to the end of the i-th window
    pred_event_indices = np.array(
        [WINDOW_SIZE - 1 + i * STRIDE for i in range(n_preds)],
        dtype=np.int64,
    )

    # Clip to valid range
    max_valid = n_events - 1
    valid_mask = pred_event_indices <= max_valid
    if not valid_mask.all():
        n_valid = valid_mask.sum()
        log.info(f"  {date_str}: clipping {n_preds} -> {n_valid} preds (events exhausted)")
        pred_event_indices = pred_event_indices[valid_mask]
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        n_preds = n_valid

    if n_preds < 200:
        log.warning(f"  {date_str}: too few predictions ({n_preds}), skipping")
        return None

    # Extract timestamps at prediction points
    pred_timestamps = timestamps[pred_event_indices].copy()

    # Reconstruct mid price path from label_1s (inter-prediction returns)
    # label_1s[i] = actual forward 1s return in ticks from prediction point i
    # Since inter-prediction time ~1.1s (close to 1s), we treat label_1s as
    # the approximate inter-prediction price change.
    # Price path: price[0] = 0, price[i+1] = price[i] + label_1s[i]
    inter_pred_returns = labels[:, 0].astype(np.float64)  # 1s horizon labels
    mid_prices = np.zeros(n_preds, dtype=np.float64)
    mid_prices[1:] = np.cumsum(inter_pred_returns[:-1])

    # Extract order flow features at prediction points
    ofi_short = events[pred_event_indices, COL_OFI_SHORT].copy()
    ofi_long = events[pred_event_indices, COL_OFI_LONG].copy()
    ofi_accel = events[pred_event_indices, COL_OFI_ACCEL].copy()

    return {
        "predictions": predictions,
        "labels": labels,
        "timestamps": pred_timestamps,
        "mid_prices": mid_prices.astype(np.float32),
        "ofi_short": ofi_short,
        "ofi_long": ofi_long,
        "ofi_accel": ofi_accel,
        "pred_event_indices": pred_event_indices,
        "n_preds": n_preds,
    }


# ============================================================
# Core simulation
# ============================================================
def simulate_day(date_str: str, data: dict, cfg: SimConfig) -> List[Trade]:
    """Run dynamic-exit trade simulation for one day."""
    predictions = data["predictions"]  # (n, 3)
    labels = data["labels"]
    timestamps = data["timestamps"]
    mid_prices = data["mid_prices"]
    ofi_short = data["ofi_short"]
    ofi_long = data["ofi_long"]
    n = data["n_preds"]

    # Select horizon for entry signal
    horizon_idx = {"1s": 0, "5s": 1, "10s": 2}[cfg.entry_horizon]
    signal = predictions[:, horizon_idx]

    trades: List[Trade] = []
    state = "FLAT"
    entry_idx = 0
    entry_price = 0.0
    entry_time_ns = 0
    entry_signal_val = 0.0
    trade_side = ""
    fade_count = 0
    n_preds_in_trade = 0
    best_price = 0.0
    worst_price = 0.0
    last_trade_idx = -cfg.cooldown_preds

    for i in range(cfg.warmup_preds, n):
        sig = signal[i]
        t_ns = int(timestamps[i])
        mid = mid_prices[i]

        if state == "FLAT":
            # Cooldown check
            if i - last_trade_idx < cfg.cooldown_preds:
                continue

            # Entry conditions
            go_long = cfg.sides in ("both", "long") and sig > cfg.entry_threshold
            go_short = cfg.sides in ("both", "short") and sig < -cfg.entry_threshold

            if go_long:
                state = "LONG"
                trade_side = "LONG"
                entry_idx = i
                entry_price = mid  # fill at current mid (bid/ask = mid +/- 0.5 tick, cost already in commission)
                entry_time_ns = t_ns
                entry_signal_val = float(sig)
                fade_count = 0
                n_preds_in_trade = 0
                best_price = mid
                worst_price = mid
            elif go_short:
                state = "SHORT"
                trade_side = "SHORT"
                entry_idx = i
                entry_price = mid
                entry_time_ns = t_ns
                entry_signal_val = float(sig)
                fade_count = 0
                n_preds_in_trade = 0
                best_price = mid
                worst_price = mid

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

            # Skip exit evaluation during min hold period
            if n_preds_in_trade < cfg.min_hold_preds:
                continue

            # Current P&L in ticks
            if trade_side == "LONG":
                current_pnl = mid - entry_price
            else:
                current_pnl = entry_price - mid

            # --- EXIT CONDITIONS (evaluated in priority order) ---
            exit_reason = ""

            # (e) Stop-loss
            if current_pnl <= -cfg.sl_ticks:
                exit_reason = "stop_loss"

            # (d) Take-profit
            elif current_pnl >= cfg.tp_ticks:
                exit_reason = "take_profit"

            # (f) Max hold time
            elif hold_s >= cfg.max_hold_s:
                exit_reason = "timeout"

            # (a) Signal reversal — new prediction flips direction
            elif cfg.reversal_exit:
                if trade_side == "LONG" and sig < -cfg.exit_threshold:
                    exit_reason = "signal_reversal"
                elif trade_side == "SHORT" and sig > cfg.exit_threshold:
                    exit_reason = "signal_reversal"

            # (b) Signal fade — prediction magnitude drops
            if not exit_reason:
                if trade_side == "LONG":
                    is_fading = sig < cfg.exit_threshold
                else:
                    is_fading = sig > -cfg.exit_threshold

                if is_fading:
                    fade_count += 1
                else:
                    fade_count = 0

                if fade_count >= cfg.fade_n:
                    exit_reason = "signal_fade"

            # (c) Order flow pressure reversal
            if not exit_reason and cfg.pressure_reversal:
                ofi = ofi_short[i]
                if np.isfinite(ofi):
                    if trade_side == "LONG" and ofi < -cfg.pressure_threshold:
                        exit_reason = "pressure_reversal"
                    elif trade_side == "SHORT" and ofi > cfg.pressure_threshold:
                        exit_reason = "pressure_reversal"

            # --- EXECUTE EXIT ---
            if exit_reason:
                exit_price = mid  # exit at current mid

                # For TP/SL: clamp exit at the trigger level since these would
                # be continuously monitored in practice (not discrete).
                # Price can gap between prediction points but TP/SL orders
                # fill at the limit, not at the gap destination.
                if exit_reason == "stop_loss":
                    # SL fills at the loss limit (worst case = SL level)
                    if trade_side == "LONG":
                        exit_price = max(exit_price, entry_price - cfg.sl_ticks)
                    else:
                        exit_price = min(exit_price, entry_price + cfg.sl_ticks)
                elif exit_reason == "take_profit":
                    # TP fills at the profit target
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
                    date=date_str,
                    side=trade_side,
                    entry_time_ns=entry_time_ns,
                    exit_time_ns=t_ns,
                    entry_price_ticks=float(entry_price),
                    exit_price_ticks=float(exit_price),
                    gross_pnl_ticks=float(gross_pnl),
                    net_pnl_ticks=float(net_pnl),
                    hold_duration_s=float(hold_s),
                    n_predictions=n_preds_in_trade,
                    exit_reason=exit_reason,
                    entry_signal=entry_signal_val,
                    mfe_ticks=float(mfe),
                    mae_ticks=float(mae),
                ))

                state = "FLAT"
                last_trade_idx = i

    return trades


# ============================================================
# Metrics
# ============================================================
def compute_metrics(trades: List[Trade]) -> dict:
    """Compute risk-adjusted performance metrics from trades."""
    if not trades:
        return {"n_trades": 0}

    net_pnls = np.array([t.net_pnl_ticks for t in trades])
    gross_pnls = np.array([t.gross_pnl_ticks for t in trades])
    n = len(net_pnls)
    win_rate = float(np.mean(net_pnls > 0))

    # Per-day P&L for Sharpe/Sortino
    day_pnl: Dict[str, float] = {}
    for t in trades:
        day_pnl[t.date] = day_pnl.get(t.date, 0.0) + t.net_pnl_ticks
    daily_pnls = np.array(list(day_pnl.values()))

    mean_daily = float(np.mean(daily_pnls)) if len(daily_pnls) > 0 else 0.0
    std_daily = float(np.std(daily_pnls)) if len(daily_pnls) > 1 else 1.0
    downside = daily_pnls[daily_pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1.0

    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 1e-8 else 0.0
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 1e-8 else 0.0

    total_win = float(np.sum(net_pnls[net_pnls > 0]))
    total_loss = float(-np.sum(net_pnls[net_pnls <= 0]))
    profit_factor = total_win / total_loss if total_loss > 1e-8 else float("inf")

    holds = [t.hold_duration_s for t in trades]
    mfes = [t.mfe_ticks for t in trades]
    maes = [t.mae_ticks for t in trades]

    # Exit reason breakdown
    exit_counts: Dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Side breakdown
    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]

    return {
        "n_trades": n,
        "n_days": len(daily_pnls),
        "trades_per_day": n / max(len(daily_pnls), 1),
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
        "daily_pnls": {k: float(v) for k, v in day_pnl.items()},
    }


def print_summary(metrics: dict, cfg: SimConfig):
    """Print a formatted summary of simulation results."""
    if metrics["n_trades"] == 0:
        log.info("No trades generated.")
        return

    log.info("=" * 70)
    log.info(f"DYNAMIC EXIT SIMULATOR RESULTS — {cfg.label_key()}")
    log.info("=" * 70)
    log.info(f"  Trades: {metrics['n_trades']} across {metrics['n_days']} days "
             f"({metrics['trades_per_day']:.1f}/day)")
    log.info(f"  Win Rate: {metrics['win_rate']:.1%}")
    log.info(f"  Net P&L: {metrics['total_net_ticks']:.2f} ticks "
             f"(${metrics['total_net_dollars']:.2f})")
    log.info(f"  Mean Net: {metrics['mean_net_ticks']:.3f} ticks/trade "
             f"(gross: {metrics['mean_gross_ticks']:.3f})")
    log.info(f"  Sharpe: {metrics['sharpe']:.2f}  |  Sortino: {metrics['sortino']:.2f}  "
             f"|  PF: {metrics['profit_factor']:.2f}")
    log.info(f"  Avg Hold: {metrics['avg_hold_s']:.1f}s  |  "
             f"Median Hold: {metrics['median_hold_s']:.1f}s")
    log.info(f"  MFE: {metrics['avg_mfe']:.2f}  |  MAE: {metrics['avg_mae']:.2f}  "
             f"|  MFE/MAE: {metrics['mfe_mae_ratio']:.2f}")
    log.info(f"  Longs: {metrics['n_longs']} (WR {metrics['long_wr']:.1%}, "
             f"avg {metrics['long_mean_net']:.3f}t)  |  "
             f"Shorts: {metrics['n_shorts']} (WR {metrics['short_wr']:.1%}, "
             f"avg {metrics['short_mean_net']:.3f}t)")
    log.info(f"  Exit Reasons: {metrics['exit_reasons']}")

    # Per-day P&L
    log.info("  Per-Day P&L (ticks):")
    for day, pnl in sorted(metrics["daily_pnls"].items()):
        marker = "+" if pnl > 0 else " "
        log.info(f"    {day}: {marker}{pnl:.2f}")
    log.info("=" * 70)


# ============================================================
# Parameter sweep
# ============================================================
def generate_sweep_configs() -> List[SimConfig]:
    """Generate parameter sweep configurations."""
    configs = []

    entry_thresholds = [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
    exit_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    fade_ns = [2, 3, 5]
    tp_ticks_list = [2.0, 3.0, 4.0, 6.0]
    sl_ticks_list = [1.0, 1.5, 2.0, 3.0]
    pressure_thresholds = [1.0, 1.5, 2.0, 999.0]  # 999 = effectively disabled
    sides_list = ["both", "short"]
    horizons = ["1s"]  # 1s is strongest signal

    # Focused sweep: vary key parameters, keep others at reasonable defaults
    # Full grid would be enormous, so do targeted sweeps

    # Sweep 1: entry/exit thresholds (most important)
    for ent, ext in product(entry_thresholds, exit_thresholds):
        if ext >= ent:  # exit must be lower than entry
            continue
        configs.append(SimConfig(
            entry_threshold=ent, exit_threshold=ext,
            tp_ticks=4.0, sl_ticks=2.0, fade_n=3,
            pressure_threshold=1.5, sides="both",
        ))

    # Sweep 2: TP/SL with best entry/exit combos
    for tp, sl in product(tp_ticks_list, sl_ticks_list):
        for ent in [0.40, 0.60, 0.80]:
            configs.append(SimConfig(
                entry_threshold=ent, exit_threshold=0.15,
                tp_ticks=tp, sl_ticks=sl, fade_n=3,
                pressure_threshold=1.5, sides="both",
            ))

    # Sweep 3: short-only (signal is stronger on short side)
    for ent in [0.30, 0.40, 0.50, 0.60]:
        for ext in [0.05, 0.10, 0.15]:
            if ext >= ent:
                continue
            configs.append(SimConfig(
                entry_threshold=ent, exit_threshold=ext,
                tp_ticks=4.0, sl_ticks=2.0, fade_n=3,
                pressure_threshold=1.5, sides="short",
            ))

    # Sweep 4: pressure threshold sensitivity
    for press in pressure_thresholds:
        for ent in [0.40, 0.60]:
            configs.append(SimConfig(
                entry_threshold=ent, exit_threshold=0.15,
                tp_ticks=4.0, sl_ticks=2.0, fade_n=3,
                pressure_threshold=press, sides="both",
            ))

    # Sweep 5: fade_n sensitivity
    for fn in fade_ns:
        for ent in [0.40, 0.60]:
            configs.append(SimConfig(
                entry_threshold=ent, exit_threshold=0.15,
                tp_ticks=4.0, sl_ticks=2.0, fade_n=fn,
                pressure_threshold=1.5, sides="both",
            ))

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = c.label_key()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ============================================================
# Main
# ============================================================
def preload_data(
    folds: List[Tuple[str, Path]],
    date_filter: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Pre-load all fold data into memory. Returns {date_str: data_dict}."""
    cached: Dict[str, dict] = {}
    for date_str, fold_path in folds:
        if date_filter and date_str not in date_filter:
            continue
        if date_str in cached:
            continue
        data = load_fold_data(fold_path, date_str)
        if data is not None:
            cached[date_str] = data
    return cached


def run_simulation(
    folds: List[Tuple[str, Path]],
    cfg: SimConfig,
    date_filter: Optional[List[str]] = None,
    verbose: bool = True,
    cached_data: Optional[Dict[str, dict]] = None,
) -> Tuple[List[Trade], dict]:
    """Run simulation across all folds/dates with given config.

    If cached_data is provided, uses pre-loaded data instead of re-loading
    from disk (much faster for parameter sweeps).
    """
    all_trades: List[Trade] = []
    dates_processed = 0
    seen_dates = set()

    for date_str, fold_path in folds:
        if date_filter and date_str not in date_filter:
            continue
        # Skip duplicate dates (fold_10 duplicates fold_01)
        if date_str in seen_dates:
            continue
        seen_dates.add(date_str)

        if verbose:
            log.info(f"Processing {date_str} ({fold_path.name})...")

        if cached_data is not None:
            data = cached_data.get(date_str)
        else:
            data = load_fold_data(fold_path, date_str)

        if data is None:
            continue

        day_trades = simulate_day(date_str, data, cfg)
        all_trades.extend(day_trades)
        dates_processed += 1

        if verbose:
            day_net = sum(t.net_pnl_ticks for t in day_trades)
            log.info(f"  {date_str}: {len(day_trades)} trades, net {day_net:.2f} ticks")

    metrics = compute_metrics(all_trades)

    if verbose and metrics["n_trades"] > 0:
        print_summary(metrics, cfg)

    return all_trades, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Exit Trade Simulator v1")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--dates", nargs="+", default=None,
                        help="Specific dates to process (e.g., 20260223 20260224)")
    parser.add_argument("--entry", type=float, default=0.50, help="Entry threshold")
    parser.add_argument("--exit-thresh", type=float, default=0.15, help="Exit threshold")
    parser.add_argument("--tp", type=float, default=4.0, help="Take-profit in ticks")
    parser.add_argument("--sl", type=float, default=2.0, help="Stop-loss in ticks")
    parser.add_argument("--sides", choices=["both", "long", "short"], default="both")
    parser.add_argument("--top-n", type=int, default=20, help="Show top N sweep results")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover folds
    log.info("Discovering CNN-Mamba v2 OOT prediction folds...")
    folds = discover_folds()
    log.info(f"Found {len(folds)} folds: {[f[0] for f in folds]}")

    if not folds:
        log.error("No prediction folds found!")
        sys.exit(1)

    if args.sweep:
        # Parameter sweep — pre-load all data once for speed
        configs = generate_sweep_configs()
        log.info(f"Running sweep with {len(configs)} configurations...")

        log.info("Pre-loading all fold data...")
        cached_data = preload_data(folds, date_filter=args.dates)
        log.info(f"Loaded {len(cached_data)} dates into memory")

        results = []
        t_start = time.time()
        for idx, cfg in enumerate(configs):
            if (idx + 1) % 20 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(configs) - idx - 1) / rate
                log.info(f"  Sweep progress: {idx + 1}/{len(configs)} "
                         f"({rate:.1f} cfg/s, ETA {eta:.0f}s)")

            _, metrics = run_simulation(
                folds, cfg, date_filter=args.dates, verbose=False,
                cached_data=cached_data,
            )

            if metrics["n_trades"] >= 10:  # minimum trades for meaningful stats
                results.append({
                    "config": cfg.to_dict(),
                    "config_key": cfg.label_key(),
                    "metrics": metrics,
                })

        # Sort by Sharpe ratio
        results.sort(key=lambda r: r["metrics"].get("sharpe", -999), reverse=True)

        # Print top results
        log.info(f"\n{'='*80}")
        log.info(f"SWEEP RESULTS — Top {min(args.top_n, len(results))} by Sharpe")
        log.info(f"{'='*80}")
        log.info(f"{'Rank':>4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
                 f"{'Trades':>7} {'AvgNet':>8} {'AvgHold':>8} {'Config'}")
        log.info("-" * 110)

        for rank, r in enumerate(results[:args.top_n], 1):
            m = r["metrics"]
            log.info(
                f"{rank:>4} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                f"{m['profit_factor']:>6.2f} {m['win_rate']:>6.1%} "
                f"{m['n_trades']:>7} {m['mean_net_ticks']:>8.3f} "
                f"{m['avg_hold_s']:>7.1f}s {r['config_key']}"
            )

        # Also show worst (for contrast)
        if len(results) > args.top_n:
            log.info(f"\nBottom 5:")
            for r in results[-5:]:
                m = r["metrics"]
                log.info(
                    f"     {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
                    f"{m['profit_factor']:>6.2f} {m['win_rate']:>6.1%} "
                    f"{m['n_trades']:>7} {m['mean_net_ticks']:>8.3f} "
                    f"{m['avg_hold_s']:>7.1f}s {r['config_key']}"
                )

        # Save full results
        out_file = OUTPUT_DIR / "sweep_results.json"
        # Remove daily_pnls from saved results to keep file manageable
        for r in results:
            r["metrics"].pop("daily_pnls", None)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"\nFull sweep results saved to {out_file}")

        # Save trade log for best config
        if results:
            best_cfg = SimConfig(**results[0]["config"])
            log.info(f"\nRe-running best config for detailed trade log...")
            best_trades, best_metrics = run_simulation(
                folds, best_cfg, date_filter=args.dates, verbose=True,
                cached_data=cached_data,
            )
            trades_file = OUTPUT_DIR / "best_trades.json"
            with open(trades_file, "w") as f:
                json.dump([asdict(t) for t in best_trades], f, indent=2)
            log.info(f"Best config trades saved to {trades_file}")

    else:
        # Single config run
        cfg = SimConfig(
            entry_threshold=args.entry,
            exit_threshold=args.exit_thresh,
            tp_ticks=args.tp,
            sl_ticks=args.sl,
            sides=args.sides,
        )

        trades, metrics = run_simulation(folds, cfg, date_filter=args.dates, verbose=True)

        # Save trade log
        if trades:
            trades_file = OUTPUT_DIR / "trades.json"
            with open(trades_file, "w") as f:
                json.dump([asdict(t) for t in trades], f, indent=2)
            log.info(f"Trade log saved to {trades_file}")

            metrics_file = OUTPUT_DIR / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            log.info(f"Metrics saved to {metrics_file}")

    # Try MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("dynamic_exit_sim_v1")
        with mlflow.start_run(run_name=f"sim_{time.strftime('%Y%m%d_%H%M%S')}"):
            if args.sweep and results:
                best = results[0]["metrics"]
                mlflow.log_params(results[0]["config"])
                for k, v in best.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        mlflow.log_metric(k, v)
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
