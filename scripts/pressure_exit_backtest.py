#!/usr/bin/env python3
"""
Pressure-Based Exit Backtest
==============================
Instead of fixed TP/SL exits, use the CNN-Mamba v2 model's ongoing predictions
as a "pressure" signal to decide when to hold vs exit a position.

Enter with existing fill sim passive entries, then:
- If model keeps predicting high pressure (large magnitude) in our direction -> HOLD
- If pressure drops (model predicts small moves or reverses) -> EXIT early
- Uses the 250ms prediction stream as a real-time hold/exit decision engine

Data sources:
- Fill sim trades: fillsim_cnn_mamba_v2_h10s_t1.0_hold30000/sim_results/
- Multi-horizon predictions at 100ms bars: h1s/h5s/h10s per_day_preds
- Trade-print mid prices: mid_price_cache_hc439/

Cost model:
- Commission: 0.376 ticks RT (passive entry)
- Market exit spread: 0.5 ticks (half-spread, RTH)
- Total exit cost per trade: 0.376 + 0.5 = 0.876 ticks

Output: JSON summary with PF, WR, Sharpe, avg hold, net ticks per config.
"""
from __future__ import annotations
import json
import glob
import os
import sys
import time
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── PATHS ────────────────────────────────────────────────────────────────────
LVL3 = Path("/home/jupiter/Lvl3Quant")

# Fill sim trades (baseline: h10s, threshold=1.0, hold=30s, no TP/SL)
FILLSIM_BASE = LVL3 / "data/processed/fillsim_cnn_mamba_v2_h10s_t1.0_hold30000"
TRADES_DIR = FILLSIM_BASE / "sim_results"

# Multi-horizon predictions at 100ms bars
PRED_DIRS = {
    "1s": LVL3 / "data/processed/fillsim_cnn_mamba_v2_h1s_t1.0_hold30000/per_day_preds",
    "5s": LVL3 / "data/processed/fillsim_cnn_mamba_v2_h5s_t1.0_hold30000/per_day_preds",
    "10s": LVL3 / "data/processed/fillsim_cnn_mamba_v2_h10s_t1.0_hold30000/per_day_preds",
}

# Trade-print cache for mid-price bars
TRADE_CACHE_DIR = LVL3 / "data/derived/mid_price_cache_hc439"

# Output
OUT_DIR = LVL3 / "output/pressure_exit_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
TICK_SIZE = 0.25        # ES tick in points
TICK_VALUE = 12.50      # $ per tick
COMMISSION_RT = 0.376   # ticks, passive entry
MARKET_EXIT_SPREAD = 0.5  # ticks, half-spread for market exit
TOTAL_EXIT_COST = COMMISSION_RT + MARKET_EXIT_SPREAD  # 0.876 ticks

BAR_NS = 100_000_000   # 100ms per bar
N_RTH_BARS = 234_000   # 6.5h RTH at 100ms
PRED_STRIDE_BARS = 3   # predictions every ~250ms ≈ 2.5 bars, round to 3 for safety

ES_PX_RAW_MIN = 5000 * 1_000_000_000
ES_PX_RAW_MAX = 8000 * 1_000_000_000

# ─── SWEEP PARAMETERS ────────────────────────────────────────────────────────
PRESSURE_FADE_THRESHOLDS = [-0.05, -0.1, -0.2]
FADE_CONSECUTIVE = [2, 4, 8]
PRESSURE_REVERSAL_THRESHOLDS = [-0.3, -0.5]
REVERSAL_CONSECUTIVE = [1, 2, 4]
WEIGHT_SCHEMES = [
    (0.5, 0.3, 0.2),   # heavy on 1s
    (0.7, 0.2, 0.1),   # very heavy on 1s
    (0.3, 0.3, 0.4),   # balanced, slight 10s tilt
]

# Baseline fixed exit params (for comparison)
BASELINE_TP_TICKS = 8
BASELINE_SL_TICKS = 16
BASELINE_MAX_HOLD_S = 1800  # 30 min
BASELINE_MAX_HOLD_BARS = BASELINE_MAX_HOLD_S * 10  # 100ms bars

# Pressure exit safety params
HARD_SL_TICKS = 16
HARD_TP_TICKS = 8
MAX_HOLD_S = 1800      # 30 min absolute max
MAX_HOLD_BARS = MAX_HOLD_S * 10


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def get_rth_open_ns(date_str: str) -> int:
    """Get RTH open (9:30 AM ET) in nanoseconds for a given date."""
    import pytz
    et = pytz.timezone("US/Eastern")
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    dt = et.localize(datetime(year, month, day, 9, 30, 0))
    return int(dt.timestamp() * 1e9)


def ns_to_bar_index(ns_timestamp: int, rth_open_ns: int) -> int:
    """Convert nanosecond timestamp to 100ms bar index."""
    offset_ns = ns_timestamp - rth_open_ns
    if offset_ns < 0:
        return 0
    bar_idx = int(offset_ns // BAR_NS)
    return min(bar_idx, N_RTH_BARS - 1)


def load_mid_prices(date_str: str, rth_open_ns: int) -> np.ndarray:
    """Load trade prints and build 100ms bar mid-prices via last-trade sampling."""
    cache_file = TRADE_CACHE_DIR / f"{date_str}_trades.npz"
    if not cache_file.exists():
        print(f"  [WARN] No trade cache for {date_str}")
        return None

    d = np.load(cache_file)
    ts = d["ts_ns"].astype(np.int64)
    px = d["price_raw"].astype(np.int64)

    # Filter to ES futures by price range
    mask = (px > ES_PX_RAW_MIN) & (px < ES_PX_RAW_MAX)
    ts = ts[mask]
    px = px[mask]

    if len(ts) == 0:
        return None

    # Filter to RTH
    rth_close_ns = rth_open_ns + int(6.5 * 3600 * 1e9)
    rth_mask = (ts >= rth_open_ns) & (ts < rth_close_ns)
    ts_rth = ts[rth_mask]
    px_rth = px[rth_mask]

    if len(ts_rth) == 0:
        return None

    # Build 100ms bars via last-trade-price sampling
    bar_edges = np.arange(N_RTH_BARS + 1, dtype=np.int64) * BAR_NS + rth_open_ns
    bar_idx = np.searchsorted(ts_rth, bar_edges[1:], side="right") - 1
    bar_idx = np.clip(bar_idx, 0, len(px_rth) - 1)
    mid_bars = px_rth[bar_idx].astype(np.float64) / 1e9  # convert to ES price points

    return mid_bars


def load_predictions(date_str: str) -> dict:
    """Load multi-horizon predictions for a date. Returns dict of horizon -> array."""
    preds = {}
    for horizon, pred_dir in PRED_DIRS.items():
        npz_path = pred_dir / f"{date_str}_preds.npz"
        if not npz_path.exists():
            return None
        d = np.load(npz_path)
        preds[horizon] = d["predictions"]
    return preds


def load_trades(date_str: str) -> list[dict]:
    """Load fill sim trades for a date."""
    result_file = TRADES_DIR / f"{date_str}_result.json"
    if not result_file.exists():
        return []
    with open(result_file) as f:
        data = json.load(f)
    return data.get("trades", [])


# ─── PRESSURE COMPUTATION ────────────────────────────────────────────────────

def compute_pressure(preds: dict, bar_idx: int, side: str,
                     weights: tuple) -> float:
    """
    Compute pressure score at a given bar index.

    For a BUY position:
      - Positive prediction = price goes up = favorable pressure
      - Negative prediction = price goes down = adverse pressure
    For a SELL position:
      - Negative prediction = price goes down = favorable pressure
      - Positive prediction = price goes up = adverse pressure

    Since CNN-Mamba v2 predicts MAGNITUDE (unsigned), we interpret the sign
    of the prediction as the model's directional lean. The raw predictions
    from the fill sim are actually signed (see signal_strength in trades).

    Returns: pressure score (positive = favorable for position, negative = adverse)
    """
    if bar_idx < 0 or bar_idx >= N_RTH_BARS:
        return 0.0

    w1s, w5s, w10s = weights
    p1s = float(preds["1s"][bar_idx])
    p5s = float(preds["5s"][bar_idx])
    p10s = float(preds["10s"][bar_idx])

    # Weighted combo of raw predictions
    raw_pressure = w1s * p1s + w5s * p5s + w10s * p10s

    # Flip sign for SELL positions (negative prediction = favorable)
    if side == "SELL":
        raw_pressure = -raw_pressure

    return raw_pressure


# ─── EXIT SIMULATION ──────────────────────────────────────────────────────────

def simulate_baseline_exit(trade: dict, mid_prices: np.ndarray,
                           rth_open_ns: int) -> dict:
    """
    Simulate fixed TP8/SL16 exit on an existing fill.
    Returns trade result dict.
    """
    fill_bar = ns_to_bar_index(trade["fill_time_ns"], rth_open_ns)
    entry_price = trade["entry_price"]
    side = trade["side"]
    direction = 1.0 if side == "BUY" else -1.0

    best_pnl_ticks = 0.0
    worst_pnl_ticks = 0.0
    exit_bar = min(fill_bar + BASELINE_MAX_HOLD_BARS, N_RTH_BARS - 1)
    exit_reason = "HoldTimeout"
    exit_price = mid_prices[exit_bar]

    for bar in range(fill_bar + 1, min(fill_bar + BASELINE_MAX_HOLD_BARS + 1, N_RTH_BARS)):
        px = mid_prices[bar]
        pnl_ticks = (px - entry_price) * direction / TICK_SIZE

        best_pnl_ticks = max(best_pnl_ticks, pnl_ticks)
        worst_pnl_ticks = min(worst_pnl_ticks, pnl_ticks)

        # Check TP
        if pnl_ticks >= BASELINE_TP_TICKS:
            exit_bar = bar
            exit_price = entry_price + direction * BASELINE_TP_TICKS * TICK_SIZE
            exit_reason = "TakeProfit"
            break

        # Check SL
        if pnl_ticks <= -BASELINE_SL_TICKS:
            exit_bar = bar
            exit_price = entry_price - direction * BASELINE_SL_TICKS * TICK_SIZE
            exit_reason = "StopLoss"
            break
    else:
        exit_bar = min(fill_bar + BASELINE_MAX_HOLD_BARS, N_RTH_BARS - 1)
        exit_price = mid_prices[exit_bar]

    raw_pnl_ticks = (exit_price - entry_price) * direction / TICK_SIZE
    # Cost: commission RT + market exit spread
    net_pnl_ticks = raw_pnl_ticks - TOTAL_EXIT_COST
    hold_bars = exit_bar - fill_bar
    hold_s = hold_bars * 0.1

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "side": side,
        "raw_pnl_ticks": round(raw_pnl_ticks, 4),
        "net_pnl_ticks": round(net_pnl_ticks, 4),
        "hold_s": round(hold_s, 2),
        "exit_reason": exit_reason,
        "mfe_ticks": round(best_pnl_ticks, 2),
        "mae_ticks": round(worst_pnl_ticks, 2),
    }


def simulate_pressure_exit(trade: dict, mid_prices: np.ndarray,
                           preds: dict, rth_open_ns: int,
                           config: dict) -> dict:
    """
    Simulate pressure-based exit on an existing fill.

    Config keys:
      - fade_threshold: pressure score below this = fading (e.g., -0.1)
      - fade_consecutive: N consecutive readings below threshold to exit
      - reversal_threshold: pressure below this = reversal (e.g., -0.5)
      - reversal_consecutive: M consecutive readings for reversal exit
      - weights: (w1s, w5s, w10s) for pressure computation
    """
    fill_bar = ns_to_bar_index(trade["fill_time_ns"], rth_open_ns)
    entry_price = trade["entry_price"]
    side = trade["side"]
    direction = 1.0 if side == "BUY" else -1.0

    fade_thresh = config["fade_threshold"]
    fade_n = config["fade_consecutive"]
    rev_thresh = config["reversal_threshold"]
    rev_n = config["reversal_consecutive"]
    weights = config["weights"]

    # Prediction check interval: every ~250ms = every 2-3 bars
    # We check every 3 bars (300ms) to approximate the 250ms stride
    check_interval = 3

    fade_count = 0
    rev_count = 0
    best_pnl_ticks = 0.0
    worst_pnl_ticks = 0.0
    exit_bar = min(fill_bar + MAX_HOLD_BARS, N_RTH_BARS - 1)
    exit_reason = "HoldTimeout"
    exit_price = mid_prices[exit_bar]

    for bar in range(fill_bar + 1, min(fill_bar + MAX_HOLD_BARS + 1, N_RTH_BARS)):
        px = mid_prices[bar]
        pnl_ticks = (px - entry_price) * direction / TICK_SIZE

        best_pnl_ticks = max(best_pnl_ticks, pnl_ticks)
        worst_pnl_ticks = min(worst_pnl_ticks, pnl_ticks)

        # Hard TP ceiling
        if pnl_ticks >= HARD_TP_TICKS:
            exit_bar = bar
            exit_price = entry_price + direction * HARD_TP_TICKS * TICK_SIZE
            exit_reason = "HardTP"
            break

        # Hard SL safety net
        if pnl_ticks <= -HARD_SL_TICKS:
            exit_bar = bar
            exit_price = entry_price - direction * HARD_SL_TICKS * TICK_SIZE
            exit_reason = "HardSL"
            break

        # Check pressure at prediction intervals
        if (bar - fill_bar) % check_interval == 0:
            pressure = compute_pressure(preds, bar, side, weights)

            # Fade detection: pressure dropped below fade threshold
            if pressure < fade_thresh:
                fade_count += 1
            else:
                fade_count = 0

            # Reversal detection: pressure strongly negative
            if pressure < rev_thresh:
                rev_count += 1
            else:
                rev_count = 0

            # Exit on sustained fade
            if fade_count >= fade_n:
                exit_bar = bar
                exit_price = px
                exit_reason = "PressureFade"
                break

            # Exit on reversal
            if rev_count >= rev_n:
                exit_bar = bar
                exit_price = px
                exit_reason = "PressureReversal"
                break
    else:
        exit_bar = min(fill_bar + MAX_HOLD_BARS, N_RTH_BARS - 1)
        exit_price = mid_prices[exit_bar]

    raw_pnl_ticks = (exit_price - entry_price) * direction / TICK_SIZE
    net_pnl_ticks = raw_pnl_ticks - TOTAL_EXIT_COST
    hold_bars = exit_bar - fill_bar
    hold_s = hold_bars * 0.1

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "side": side,
        "raw_pnl_ticks": round(raw_pnl_ticks, 4),
        "net_pnl_ticks": round(net_pnl_ticks, 4),
        "hold_s": round(hold_s, 2),
        "exit_reason": exit_reason,
        "mfe_ticks": round(best_pnl_ticks, 2),
        "mae_ticks": round(worst_pnl_ticks, 2),
    }


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute summary metrics from a list of trade results."""
    if not results:
        return {"n_trades": 0}

    net_pnls = np.array([r["net_pnl_ticks"] for r in results])
    raw_pnls = np.array([r["raw_pnl_ticks"] for r in results])
    hold_times = np.array([r["hold_s"] for r in results])

    n = len(net_pnls)
    wins = net_pnls > 0
    losses = net_pnls < 0
    n_wins = int(wins.sum())
    n_losses = int(losses.sum())

    total_win_ticks = float(net_pnls[wins].sum()) if n_wins > 0 else 0.0
    total_loss_ticks = float(abs(net_pnls[losses].sum())) if n_losses > 0 else 0.0

    pf = total_win_ticks / total_loss_ticks if total_loss_ticks > 0 else float("inf")
    wr = n_wins / n * 100.0 if n > 0 else 0.0

    # Sharpe (per-trade, annualized assuming ~250 trading days, ~20 trades/day)
    if net_pnls.std() > 0:
        sharpe_per_trade = net_pnls.mean() / net_pnls.std()
        # Rough annualization: sqrt(N_trades_per_year)
        trades_per_year = n / len(set(r.get("date", "x") for r in results)) * 252
        sharpe_ann = sharpe_per_trade * np.sqrt(max(trades_per_year, 1))
    else:
        sharpe_per_trade = 0.0
        sharpe_ann = 0.0

    # Sortino
    downside = net_pnls[net_pnls < 0]
    if len(downside) > 0:
        downside_std = np.sqrt(np.mean(downside ** 2))
        sortino = net_pnls.mean() / downside_std if downside_std > 0 else float("inf")
    else:
        sortino = float("inf")

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for r in results:
        exit_reasons[r["exit_reason"]] += 1

    return {
        "n_trades": n,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 3),
        "total_net_ticks": round(float(net_pnls.sum()), 2),
        "total_net_usd": round(float(net_pnls.sum()) * TICK_VALUE, 2),
        "mean_net_ticks": round(float(net_pnls.mean()), 4),
        "mean_raw_ticks": round(float(raw_pnls.mean()), 4),
        "std_net_ticks": round(float(net_pnls.std()), 4),
        "sharpe_per_trade": round(sharpe_per_trade, 4),
        "sharpe_ann": round(sharpe_ann, 2),
        "sortino_per_trade": round(sortino, 4),
        "avg_hold_s": round(float(hold_times.mean()), 2),
        "median_hold_s": round(float(np.median(hold_times)), 2),
        "max_hold_s": round(float(hold_times.max()), 2),
        "avg_mfe_ticks": round(float(np.mean([r["mfe_ticks"] for r in results])), 2),
        "avg_mae_ticks": round(float(np.mean([r["mae_ticks"] for r in results])), 2),
        "exit_reasons": dict(exit_reasons),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("PRESSURE-BASED EXIT BACKTEST")
    print("=" * 70)

    # ── Discover available dates ──────────────────────────────────────────
    trade_files = sorted(glob.glob(str(TRADES_DIR / "*_result.json")))
    dates = []
    for f in trade_files:
        date_str = os.path.basename(f).split("_")[0]
        dates.append(date_str)
    print(f"\nDates with fill sim trades: {dates}")

    # ── Load all data ─────────────────────────────────────────────────────
    all_day_data = {}
    total_trades = 0
    for date_str in dates:
        rth_open_ns = get_rth_open_ns(date_str)
        mid_prices = load_mid_prices(date_str, rth_open_ns)
        preds = load_predictions(date_str)
        trades = load_trades(date_str)

        if mid_prices is None or preds is None or not trades:
            print(f"  [SKIP] {date_str}: missing data")
            continue

        all_day_data[date_str] = {
            "rth_open_ns": rth_open_ns,
            "mid_prices": mid_prices,
            "preds": preds,
            "trades": trades,
        }
        total_trades += len(trades)
        print(f"  [OK] {date_str}: {len(trades)} trades, "
              f"prices range {mid_prices.min():.2f}-{mid_prices.max():.2f}")

    print(f"\nLoaded {len(all_day_data)} days, {total_trades} total trades")

    if total_trades == 0:
        print("[ERROR] No trades to backtest!")
        return

    # ── Run baseline (fixed TP8/SL16) ────────────────────────────────────
    print("\n" + "─" * 70)
    print("BASELINE: Fixed TP=8 / SL=16 / MaxHold=30min")
    print("─" * 70)

    baseline_results = []
    for date_str, day in all_day_data.items():
        for trade in day["trades"]:
            result = simulate_baseline_exit(
                trade, day["mid_prices"], day["rth_open_ns"]
            )
            result["date"] = date_str
            baseline_results.append(result)

    baseline_metrics = compute_metrics(baseline_results)
    print(f"  Trades: {baseline_metrics['n_trades']}")
    print(f"  WR: {baseline_metrics['win_rate_pct']}%")
    print(f"  PF: {baseline_metrics['profit_factor']}")
    print(f"  Net ticks: {baseline_metrics['total_net_ticks']}")
    print(f"  Net USD: ${baseline_metrics['total_net_usd']}")
    print(f"  Sharpe (ann): {baseline_metrics['sharpe_ann']}")
    print(f"  Sortino: {baseline_metrics['sortino_per_trade']}")
    print(f"  Avg hold: {baseline_metrics['avg_hold_s']}s")
    print(f"  Exit reasons: {baseline_metrics['exit_reasons']}")

    # ── Sweep pressure configs ────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PRESSURE EXIT SWEEP")
    print("─" * 70)

    configs = []
    for fade_thresh in PRESSURE_FADE_THRESHOLDS:
        for fade_n in FADE_CONSECUTIVE:
            for rev_thresh in PRESSURE_REVERSAL_THRESHOLDS:
                for rev_n in REVERSAL_CONSECUTIVE:
                    for weights in WEIGHT_SCHEMES:
                        configs.append({
                            "fade_threshold": fade_thresh,
                            "fade_consecutive": fade_n,
                            "reversal_threshold": rev_thresh,
                            "reversal_consecutive": rev_n,
                            "weights": weights,
                        })

    print(f"Total configs to sweep: {len(configs)}")

    all_results = []
    best_pf = 0
    best_config_name = ""

    for i, config in enumerate(configs):
        w = config["weights"]
        config_name = (
            f"fade{config['fade_threshold']}_n{config['fade_consecutive']}_"
            f"rev{config['reversal_threshold']}_m{config['reversal_consecutive']}_"
            f"w{w[0]}_{w[1]}_{w[2]}"
        )

        pressure_results = []
        for date_str, day in all_day_data.items():
            for trade in day["trades"]:
                result = simulate_pressure_exit(
                    trade, day["mid_prices"], day["preds"],
                    day["rth_open_ns"], config
                )
                result["date"] = date_str
                pressure_results.append(result)

        metrics = compute_metrics(pressure_results)
        metrics["config_name"] = config_name
        metrics["config"] = {
            "fade_threshold": config["fade_threshold"],
            "fade_consecutive": config["fade_consecutive"],
            "reversal_threshold": config["reversal_threshold"],
            "reversal_consecutive": config["reversal_consecutive"],
            "weights_1s_5s_10s": list(config["weights"]),
        }

        all_results.append(metrics)

        # Track best
        pf = metrics["profit_factor"]
        if pf > best_pf and metrics["n_trades"] > 10:
            best_pf = pf
            best_config_name = config_name

        # Progress
        if (i + 1) % 27 == 0 or (i + 1) == len(configs):
            print(f"  [{i+1}/{len(configs)}] Last: PF={pf:.3f} WR={metrics['win_rate_pct']}% "
                  f"Net={metrics['total_net_ticks']:.1f}t Hold={metrics['avg_hold_s']:.1f}s "
                  f"({config_name[:40]}...)")

    # ── Sort by PF descending ─────────────────────────────────────────────
    all_results.sort(key=lambda x: x.get("profit_factor", 0), reverse=True)

    # ── Per-day breakdown for top 5 configs ──────────────────────────────
    print("\n" + "─" * 70)
    print("TOP 5 PRESSURE CONFIGS vs BASELINE")
    print("─" * 70)

    top5_with_daily = []
    for rank, result in enumerate(all_results[:5]):
        cfg = result["config"]
        config_obj = {
            "fade_threshold": cfg["fade_threshold"],
            "fade_consecutive": cfg["fade_consecutive"],
            "reversal_threshold": cfg["reversal_threshold"],
            "reversal_consecutive": cfg["reversal_consecutive"],
            "weights": tuple(cfg["weights_1s_5s_10s"]),
        }

        # Re-run to get per-day breakdown
        daily_pnl = defaultdict(float)
        daily_trades_count = defaultdict(int)
        for date_str, day in all_day_data.items():
            for trade in day["trades"]:
                r = simulate_pressure_exit(
                    trade, day["mid_prices"], day["preds"],
                    day["rth_open_ns"], config_obj
                )
                daily_pnl[date_str] += r["net_pnl_ticks"]
                daily_trades_count[date_str] += 1

        daily_pnls = [daily_pnl[d] for d in sorted(daily_pnl.keys())]
        daily_arr = np.array(daily_pnls)
        green_days = int((daily_arr > 0).sum())
        red_days = int((daily_arr < 0).sum())
        daily_sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)
                        if daily_arr.std() > 0 else 0)

        result["per_day"] = {d: {"net_ticks": round(daily_pnl[d], 2),
                                  "n_trades": daily_trades_count[d]}
                             for d in sorted(daily_pnl.keys())}
        result["green_days"] = green_days
        result["red_days"] = red_days
        result["daily_sharpe_ann"] = round(daily_sharpe, 2)

        print(f"\n  #{rank+1}: {result['config_name']}")
        print(f"    PF={result['profit_factor']:.3f}  WR={result['win_rate_pct']}%  "
              f"Net={result['total_net_ticks']:.1f}t  ${result['total_net_usd']:.0f}")
        print(f"    Sharpe(ann)={result['sharpe_ann']:.1f}  "
              f"Sortino={result['sortino_per_trade']:.3f}  "
              f"DailySharpe={daily_sharpe:.1f}")
        print(f"    Hold: avg={result['avg_hold_s']:.1f}s  "
              f"median={result['median_hold_s']:.1f}s")
        print(f"    Days: {green_days}G/{red_days}R  "
              f"Exits: {result['exit_reasons']}")
        for d in sorted(daily_pnl.keys()):
            marker = "+" if daily_pnl[d] > 0 else " "
            print(f"      {d}: {marker}{daily_pnl[d]:+7.1f}t  ({daily_trades_count[d]} trades)")

    # ── Also compute baseline per-day ─────────────────────────────────────
    print(f"\n  BASELINE (TP8/SL16/30min):")
    bl_daily_pnl = defaultdict(float)
    bl_daily_count = defaultdict(int)
    for r in baseline_results:
        bl_daily_pnl[r["date"]] += r["net_pnl_ticks"]
        bl_daily_count[r["date"]] += 1
    bl_daily_arr = np.array([bl_daily_pnl[d] for d in sorted(bl_daily_pnl.keys())])
    bl_green = int((bl_daily_arr > 0).sum())
    bl_red = int((bl_daily_arr < 0).sum())
    bl_daily_sharpe = (bl_daily_arr.mean() / bl_daily_arr.std() * np.sqrt(252)
                       if bl_daily_arr.std() > 0 else 0)
    print(f"    PF={baseline_metrics['profit_factor']:.3f}  "
          f"WR={baseline_metrics['win_rate_pct']}%  "
          f"Net={baseline_metrics['total_net_ticks']:.1f}t  "
          f"${baseline_metrics['total_net_usd']:.0f}")
    print(f"    Sharpe(ann)={baseline_metrics['sharpe_ann']:.1f}  "
          f"DailySharpe={bl_daily_sharpe:.1f}")
    print(f"    Days: {bl_green}G/{bl_red}R")
    for d in sorted(bl_daily_pnl.keys()):
        marker = "+" if bl_daily_pnl[d] > 0 else " "
        print(f"      {d}: {marker}{bl_daily_pnl[d]:+7.1f}t  ({bl_daily_count[d]} trades)")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {
            "n_dates": len(all_day_data),
            "dates": sorted(all_day_data.keys()),
            "total_trades": total_trades,
            "trade_source": str(TRADES_DIR),
        },
        "cost_model": {
            "commission_rt_ticks": COMMISSION_RT,
            "market_exit_spread_ticks": MARKET_EXIT_SPREAD,
            "total_exit_cost_ticks": TOTAL_EXIT_COST,
        },
        "pressure_exit_params": {
            "hard_tp_ticks": HARD_TP_TICKS,
            "hard_sl_ticks": HARD_SL_TICKS,
            "max_hold_s": MAX_HOLD_S,
            "check_interval_bars": 3,
            "check_interval_ms": 300,
        },
        "baseline": {
            "config": {
                "tp_ticks": BASELINE_TP_TICKS,
                "sl_ticks": BASELINE_SL_TICKS,
                "max_hold_s": BASELINE_MAX_HOLD_S,
            },
            "metrics": baseline_metrics,
            "per_day": {d: {"net_ticks": round(bl_daily_pnl[d], 2),
                           "n_trades": bl_daily_count[d]}
                       for d in sorted(bl_daily_pnl.keys())},
            "green_days": bl_green,
            "red_days": bl_red,
            "daily_sharpe_ann": round(bl_daily_sharpe, 2),
        },
        "top_10_pressure_configs": all_results[:10],
        "all_configs_summary": [
            {
                "config_name": r["config_name"],
                "pf": r["profit_factor"],
                "wr": r["win_rate_pct"],
                "net_ticks": r["total_net_ticks"],
                "sharpe_ann": r["sharpe_ann"],
                "avg_hold_s": r["avg_hold_s"],
                "exit_reasons": r["exit_reasons"],
            }
            for r in all_results
        ],
        "n_configs_tested": len(all_results),
    }

    out_path = OUT_DIR / "pressure_exit_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
