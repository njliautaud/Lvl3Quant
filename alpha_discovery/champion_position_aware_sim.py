#!/usr/bin/env python3
"""Position-aware trade simulation for champion config.

The sweep results show 14.8 trades/day — more than available 30-min bars.
This means positions overlap heavily. In reality, with a single contract,
you can only hold one position at a time.

This script reuses the saved walk-forward predictions from the sensitivity
sweep and re-simulates with proper position management:
  1. ONE_AT_A_TIME: Only enter when flat (most realistic for 1 contract)
  2. FIFO_QUEUE: Queue signals, enter next when current exits
  3. BEST_SIGNAL: When multiple signals pending, take the strongest

Also adds MFE/MAE tracking per trade for HC #361 price path reporting.

Run on Neptune: python champion_position_aware_sim.py
"""

import logging
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path("/home/nick/Lvl3Quant")
MINUTE_BAR_DIR = LVL3_ROOT / "data" / "processed" / "mbo_minute_bars_v1"
PRED_PATH = LVL3_ROOT / "output" / "champion_sensitivity_sweep" / "wf_predictions.parquet"
ENHANCED_DAILY = LVL3_ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"
OUT_DIR = LVL3_ROOT / "output" / "champion_position_aware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
COST_ENTRY_TICKS = 0.376
COST_EXIT_TICKS = 1.376
COST_RT_TICKS = COST_ENTRY_TICKS + COST_EXIT_TICKS  # 1.752

# Champion config
TP_TICKS = 25
SL_LONG_TICKS = 4
SL_SHORT_TICKS = 3
THRESHOLD = 0.05


def load_minute_bars() -> pd.DataFrame:
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    log.info(f"Loading {len(files)} minute bar files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "date" not in df.columns:
                fname = f.stem
                date_str = "".join(c for c in fname if c.isdigit())[:8]
                if len(date_str) == 8:
                    df["date"] = date_str
            dfs.append(df)
        except Exception as e:
            log.warning(f"Skip {f.name}: {e}")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort_values(["date", "ts_minute"], inplace=True)
    df_all["ts_minute"] = pd.to_datetime(df_all["ts_minute"])
    return df_all


def classify_days(minute_bars: pd.DataFrame) -> dict:
    daily_close = {}
    if ENHANCED_DAILY.exists():
        try:
            edf = pd.read_parquet(ENHANCED_DAILY)
            if "ES_close" in edf.columns and "date" in edf.columns:
                for _, row in edf.iterrows():
                    d = str(row["date"])[:8] if not isinstance(row["date"], str) else row["date"]
                    daily_close[d] = float(row["ES_close"])
        except Exception:
            pass
    if not daily_close:
        for date, grp in minute_bars.groupby("date"):
            if len(grp) > 0:
                daily_close[date] = float(grp.iloc[-1]["close"])
    dates_sorted = sorted(daily_close.keys())
    regimes = {}
    for i, d in enumerate(dates_sorted):
        if i == 0:
            regimes[d] = "flat"
            continue
        chg = daily_close[d] - daily_close[dates_sorted[i - 1]]
        if chg > 5.0:
            regimes[d] = "green"
        elif chg < -5.0:
            regimes[d] = "red"
        else:
            regimes[d] = "flat"
    return regimes


def simulate_position_aware(
    predictions: pd.DataFrame,
    minute_bars: pd.DataFrame,
    mode: str = "one_at_a_time",
) -> list[dict]:
    """Position-aware simulation: only one trade at a time.

    mode:
      'one_at_a_time': Skip signals while in position
      'best_signal': When flat, take strongest pending signal from current bar
    """
    # Index minute bars by date
    minute_bars_by_date = {
        date: grp.sort_values("ts_minute").reset_index(drop=True)
        for date, grp in minute_bars.groupby("date")
    }

    # Sort predictions by date and bar_30
    preds = predictions.copy()
    preds["bar_30"] = pd.to_datetime(preds["bar_30"])
    preds = preds.sort_values(["date", "bar_30"]).reset_index(drop=True)

    trades = []
    in_position = False
    current_exit_time = None

    for _, row in preds.iterrows():
        pred = row["prediction"]
        date = row["date"]
        bar_time = row["bar_30"]

        # Skip if below threshold
        if abs(pred) < THRESHOLD:
            continue

        # Position management: skip if still in position
        if in_position:
            if current_exit_time is not None and bar_time >= current_exit_time:
                in_position = False  # previous trade exited
            else:
                continue  # skip — still holding

        direction = 1 if pred > 0 else -1
        entry_price = row["close"]
        sl_ticks = SL_LONG_TICKS if direction == 1 else SL_SHORT_TICKS
        sl_pts = sl_ticks * ES_TICK_SIZE
        tp_pts = TP_TICKS * ES_TICK_SIZE

        if date not in minute_bars_by_date:
            continue
        day_mins = minute_bars_by_date[date]

        mask_after = day_mins["ts_minute"] > bar_time
        future_mins = day_mins[mask_after]
        if len(future_mins) == 0:
            continue

        # Track MFE/MAE (HC #361: price path)
        mfe_ticks = 0.0  # max favorable excursion
        mae_ticks = 0.0  # max adverse excursion
        exit_price = None
        exit_reason = None
        exit_time = None
        bars_held = 0

        # Price path at fixed offsets
        price_path = {}
        entry_ts = bar_time

        for _, mbar in future_mins.iterrows():
            bars_held += 1
            elapsed_min = (mbar["ts_minute"] - entry_ts).total_seconds() / 60

            # Track MFE/MAE
            if direction == 1:
                fav = (mbar["high"] - entry_price) / ES_TICK_SIZE
                adv = (entry_price - mbar["low"]) / ES_TICK_SIZE
            else:
                fav = (entry_price - mbar["low"]) / ES_TICK_SIZE
                adv = (mbar["high"] - entry_price) / ES_TICK_SIZE
            mfe_ticks = max(mfe_ticks, fav)
            mae_ticks = max(mae_ticks, adv)

            # Price path at standard offsets
            mark_ticks = (mbar["close"] - entry_price) * direction / ES_TICK_SIZE
            for offset in [1, 2, 5, 10, 15, 30, 60]:
                if offset not in price_path and elapsed_min >= offset:
                    price_path[offset] = round(mark_ticks, 2)

            # SL check first (conservative)
            if direction == 1:
                if mbar["low"] <= entry_price - sl_pts:
                    exit_price = entry_price - sl_pts
                    exit_reason = "SL"
                    exit_time = mbar["ts_minute"]
                    break
                if mbar["high"] >= entry_price + tp_pts:
                    exit_price = entry_price + tp_pts
                    exit_reason = "TP"
                    exit_time = mbar["ts_minute"]
                    break
            else:
                if mbar["high"] >= entry_price + sl_pts:
                    exit_price = entry_price + sl_pts
                    exit_reason = "SL"
                    exit_time = mbar["ts_minute"]
                    break
                if mbar["low"] <= entry_price - tp_pts:
                    exit_price = entry_price - tp_pts
                    exit_reason = "TP"
                    exit_time = mbar["ts_minute"]
                    break

            if bars_held >= 60:
                exit_price = mbar["close"]
                exit_reason = "TIMEOUT"
                exit_time = mbar["ts_minute"]
                break

        if exit_price is None:
            exit_price = future_mins.iloc[-1]["close"]
            exit_reason = "EOD"
            exit_time = future_mins.iloc[-1]["ts_minute"]
            bars_held = len(future_mins)

        # Mark position as held until exit
        in_position = True
        current_exit_time = exit_time

        raw_pnl_pts = (exit_price - entry_price) * direction
        raw_pnl_ticks = raw_pnl_pts / ES_TICK_SIZE
        net_pnl_ticks = raw_pnl_ticks - COST_RT_TICKS

        trade = {
            "date": date,
            "entry_time": str(bar_time),
            "exit_time": str(exit_time),
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "prediction": float(pred),
            "raw_pnl_ticks": round(float(raw_pnl_ticks), 4),
            "net_pnl_ticks": round(float(net_pnl_ticks), 4),
            "net_pnl_dollars": round(float(net_pnl_ticks * ES_TICK_VALUE), 2),
            "mfe_ticks": round(float(mfe_ticks), 2),
            "mae_ticks": round(float(mae_ticks), 2),
        }
        # Add price path offsets
        for offset in [1, 2, 5, 10, 15, 30, 60]:
            trade[f"path_{offset}m"] = price_path.get(offset, None)

        trades.append(trade)

    return trades


def compute_metrics(trades: list[dict], regimes: dict = None, label: str = "") -> dict:
    if not trades:
        return {"label": label, "n_trades": 0}

    pnls = np.array([t["net_pnl_ticks"] for t in trades])
    n = len(pnls)
    win_rate = float((pnls > 0).mean())
    avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls < 0].sum())
    pf = float(gross_wins / (gross_losses + 1e-8))

    # Daily Sharpe
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["date"]] += t["net_pnl_ticks"]
    daily_arr = np.array(list(daily_pnl.values()))
    n_days = len(daily_arr)
    sharpe = float(daily_arr.mean() / (daily_arr.std() + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    # Sortino
    downside = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
    sortino = float(daily_arr.mean() / (downside_std + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    # Max drawdown (daily)
    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    # Calmar
    annual_return = daily_arr.mean() * 252
    calmar = float(annual_return / (max_dd + 1e-8))

    # Win/loss streaks
    wins = pnls > 0
    max_win_streak = 0
    max_loss_streak = 0
    cur_streak = 0
    for w in wins:
        if w:
            cur_streak = max(cur_streak + 1, 0) if cur_streak >= 0 else 1
        else:
            cur_streak = min(cur_streak - 1, 0) if cur_streak <= 0 else -1
        max_win_streak = max(max_win_streak, cur_streak)
        max_loss_streak = min(max_loss_streak, cur_streak)

    # MFE/MAE stats (may not exist for independent sim trades)
    has_mfe = "mfe_ticks" in trades[0]
    mfes = np.array([t.get("mfe_ticks", 0) for t in trades]) if has_mfe else np.zeros(n)
    maes = np.array([t.get("mae_ticks", 0) for t in trades]) if has_mfe else np.zeros(n)

    # Regime analysis
    regime_gap = 999.0
    sharpe_green = 0
    sharpe_red = 0
    if regimes:
        green_pnl = defaultdict(float)
        red_pnl = defaultdict(float)
        for t in trades:
            r = regimes.get(t["date"], "flat")
            if r == "green":
                green_pnl[t["date"]] += t["net_pnl_ticks"]
            elif r == "red":
                red_pnl[t["date"]] += t["net_pnl_ticks"]
        if len(green_pnl) > 1 and len(red_pnl) > 1:
            g_arr = np.array(list(green_pnl.values()))
            r_arr = np.array(list(red_pnl.values()))
            sharpe_green = float(g_arr.mean() / (g_arr.std() + 1e-8) * np.sqrt(252))
            sharpe_red = float(r_arr.mean() / (r_arr.std() + 1e-8) * np.sqrt(252))
            max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-8)
            regime_gap = abs(sharpe_green - sharpe_red) / max_abs

    # Trades per day (only on days with trades)
    trades_per_day = n / max(n_days, 1)

    # Pct green days
    green_days = (daily_arr > 0).sum()
    pct_green = green_days / n_days * 100 if n_days > 0 else 0

    return {
        "label": label,
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(trades_per_day, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "pf": round(pf, 2),
        "wr": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl_ticks": round(float(pnls.sum()), 1),
        "total_pnl_dollars": round(float(pnls.sum() * ES_TICK_VALUE), 2),
        "max_dd_ticks": round(max_dd, 1),
        "calmar": round(calmar, 2),
        "pct_green_days": round(pct_green, 1),
        "regime_gap": round(regime_gap, 3),
        "sharpe_green": round(sharpe_green, 2),
        "sharpe_red": round(sharpe_red, 2),
        "mfe_mean": round(float(mfes.mean()), 2),
        "mfe_p90": round(float(np.percentile(mfes, 90)), 2),
        "mae_mean": round(float(maes.mean()), 2),
        "mae_p90": round(float(np.percentile(maes, 90)), 2),
    }


def print_price_path_summary(trades: list[dict]):
    """Print average price path at fixed time offsets (HC #361)."""
    offsets = [1, 2, 5, 10, 15, 30, 60]
    log.info("\n=== PRICE PATH (avg mark-to-market in ticks, direction-adjusted) ===")

    for side in ["ALL", "LONG", "SHORT", "TP", "SL"]:
        if side == "ALL":
            subset = trades
        elif side in ("LONG", "SHORT"):
            subset = [t for t in trades if t["direction"] == side]
        elif side == "TP":
            subset = [t for t in trades if t["exit_reason"] == "TP"]
        elif side == "SL":
            subset = [t for t in trades if t["exit_reason"] == "SL"]

        if not subset:
            continue

        path_str = f"  {side:5s} (n={len(subset):4d}): "
        for off in offsets:
            vals = [t[f"path_{off}m"] for t in subset if t[f"path_{off}m"] is not None]
            if vals:
                path_str += f" +{off}m={np.mean(vals):+.1f}"
        log.info(path_str)


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("CHAMPION POSITION-AWARE SIMULATION")
    log.info("=" * 70)

    # Load saved predictions
    log.info("Loading walk-forward predictions...")
    predictions = pd.read_parquet(PRED_PATH)
    log.info(f"Loaded {len(predictions):,} predictions")

    # Load minute bars
    minute_bars = load_minute_bars()

    # Classify days
    regimes = classify_days(minute_bars)

    # -------------------------------------------------------------------
    # Simulation 1: Independent trades (original, for comparison)
    # -------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Mode: INDEPENDENT (original — no position management)")
    log.info("=" * 70)

    # Quick independent sim for comparison
    from champion_sensitivity_sweep import simulate_trades as sim_independent
    from champion_sensitivity_sweep import get_prior_day_bias, aggregate_to_30min, engineer_features, compute_label

    bars_30 = aggregate_to_30min(minute_bars)
    bars_30 = engineer_features(bars_30)
    bars_30 = compute_label(bars_30)
    prior_day_up = get_prior_day_bias(bars_30)

    trades_indep = sim_independent(
        predictions, minute_bars,
        tp_ticks=TP_TICKS, sl_long_ticks=SL_LONG_TICKS, sl_short_ticks=SL_SHORT_TICKS,
        max_hold_minutes=60, threshold=THRESHOLD,
        daily_bias=True, bias_mult=1.5,
        prior_day_up=prior_day_up,
    )
    m_indep = compute_metrics(trades_indep, regimes, "independent")
    log.info(f"  Trades: {m_indep['n_trades']}, {m_indep['trades_per_day']}/day")
    log.info(f"  Sharpe: {m_indep['sharpe']}, Sortino: {m_indep['sortino']}, PF: {m_indep['pf']}, WR: {m_indep['wr']}%")
    log.info(f"  Total: {m_indep['total_pnl_ticks']} ticks (${m_indep['total_pnl_dollars']:,.0f})")
    log.info(f"  Green days: {m_indep['pct_green_days']}%, Max DD: {m_indep['max_dd_ticks']} ticks")
    log.info(f"  Regime gap: {m_indep['regime_gap']:.3f} (G: {m_indep['sharpe_green']}, R: {m_indep['sharpe_red']})")

    # -------------------------------------------------------------------
    # Simulation 2: One-at-a-time (realistic single contract)
    # -------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Mode: ONE-AT-A-TIME (single contract, skip signals while in position)")
    log.info("=" * 70)

    trades_oat = simulate_position_aware(predictions, minute_bars, mode="one_at_a_time")
    m_oat = compute_metrics(trades_oat, regimes, "one_at_a_time")
    log.info(f"  Trades: {m_oat['n_trades']}, {m_oat['trades_per_day']}/day")
    log.info(f"  Sharpe: {m_oat['sharpe']}, Sortino: {m_oat['sortino']}, PF: {m_oat['pf']}, WR: {m_oat['wr']}%")
    log.info(f"  Total: {m_oat['total_pnl_ticks']} ticks (${m_oat['total_pnl_dollars']:,.0f})")
    log.info(f"  Green days: {m_oat['pct_green_days']}%, Max DD: {m_oat['max_dd_ticks']} ticks")
    log.info(f"  Regime gap: {m_oat['regime_gap']:.3f} (G: {m_oat['sharpe_green']}, R: {m_oat['sharpe_red']})")
    log.info(f"  MFE mean: {m_oat['mfe_mean']} ticks, p90: {m_oat['mfe_p90']} ticks")
    log.info(f"  MAE mean: {m_oat['mae_mean']} ticks, p90: {m_oat['mae_p90']} ticks")

    # Price path
    print_price_path_summary(trades_oat)

    # -------------------------------------------------------------------
    # Exit reason breakdown
    # -------------------------------------------------------------------
    log.info("\n=== EXIT REASON BREAKDOWN ===")
    for mode_label, trade_list in [("independent", trades_indep), ("one_at_a_time", trades_oat)]:
        reasons = defaultdict(int)
        for t in trade_list:
            reasons[t.get("exit_reason", "UNKNOWN")] += 1
        reason_str = ", ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
        log.info(f"  {mode_label}: {reason_str}")

    # -------------------------------------------------------------------
    # Direction breakdown
    # -------------------------------------------------------------------
    log.info("\n=== DIRECTION BREAKDOWN (one-at-a-time) ===")
    for side in ["LONG", "SHORT"]:
        side_trades = [t for t in trades_oat if t["direction"] == side]
        if side_trades:
            m_side = compute_metrics(side_trades, regimes, side)
            log.info(f"  {side}: {m_side['n_trades']} trades, Sharpe {m_side['sharpe']}, "
                     f"PF {m_side['pf']}, WR {m_side['wr']}%, avg_win {m_side['avg_win']}, "
                     f"avg_loss {m_side['avg_loss']}")

    # -------------------------------------------------------------------
    # Comparison summary
    # -------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: Independent vs One-at-a-Time")
    log.info("=" * 70)
    reduction = (1 - m_oat['n_trades'] / m_indep['n_trades']) * 100
    pnl_reduction = (1 - m_oat['total_pnl_ticks'] / m_indep['total_pnl_ticks']) * 100
    log.info(f"  Trade count: {m_indep['n_trades']} → {m_oat['n_trades']} ({reduction:.0f}% fewer)")
    log.info(f"  Total P&L: {m_indep['total_pnl_ticks']:.0f} → {m_oat['total_pnl_ticks']:.0f} ticks ({pnl_reduction:.0f}% less)")
    log.info(f"  Sharpe: {m_indep['sharpe']} → {m_oat['sharpe']}")
    log.info(f"  PF: {m_indep['pf']} → {m_oat['pf']}")
    log.info(f"  WR: {m_indep['wr']} → {m_oat['wr']}%")
    log.info(f"  Regime gap: {m_indep['regime_gap']:.3f} → {m_oat['regime_gap']:.3f}")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    # Save trades
    trades_df = pd.DataFrame(trades_oat)
    trades_df.to_csv(OUT_DIR / "trades_one_at_a_time.csv", index=False)

    # Save summary
    summary = {
        "independent": m_indep,
        "one_at_a_time": m_oat,
        "reduction_pct_trades": round(reduction, 1),
        "reduction_pct_pnl": round(pnl_reduction, 1),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("Done.")


if __name__ == "__main__":
    main()
