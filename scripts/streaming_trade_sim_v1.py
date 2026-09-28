"""
Streaming Trade Simulator v1 — FIFO-based realistic ES futures backtester
=========================================================================
Uses streaming continuation intensity predictions (XGBoost, 4 horizons)
as a GATE for when to trade, combined with order-flow direction from
book features (cum_delta, depth_imbal, net_order_flow).

Design:
  - Intensity model predicts mfe_minus_mae (direction-agnostic pressure magnitude)
  - Direction comes from book features: cum_delta change + depth_imbal_5
  - Entry: intensity in top decile AND clear directional signal from order flow
  - Hold: while multi-horizon intensity stays above median
  - Exit: when intensity fades OR order flow reverses

FIFO Cost Model:
  - Passive entry (limit at best bid/ask): 0.376 ticks (commission only)
  - Market exit: 1.376 ticks (commission + 1 tick spread crossing)
  - Total round-trip: 1.752 ticks

HC #428: Regime-agnostic validation
HC #432: MFE-within-horizon validation
HC #69:  Report Sharpe/Sortino/PF/WR as primary metrics
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/home/nick/Lvl3Quant/data")
PRED_DIR = DATA_ROOT / "models/streaming_continuation_v1"
RELABEL_DIR = DATA_ROOT / "relabel"
EVENTS_DIR = DATA_ROOT / "processed/mbo_events_smart_v3"
BOOK_DIR = DATA_ROOT / "processed/mbo_book_features"
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/streaming_trade_sim_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "streaming_trade_sim_v1"

# ─── Cost Constants (ES Futures, AMP/Rithmic) ────────────────────────────
ES_TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376    # $4.70 / $12.50
SPREAD_TICKS = 1.0
PASSIVE_ENTRY_COST = COMMISSION_TICKS              # 0.376 ticks
MARKET_EXIT_COST = COMMISSION_TICKS + SPREAD_TICKS  # 1.376 ticks
TOTAL_RT_COST = PASSIVE_ENTRY_COST + MARKET_EXIT_COST  # 1.752 ticks

# ─── OOS Subsampling (must match training code exactly) ──────────────────
OOT_MAX_EVENTS = 500_000

# ─── Trade Logic Parameters ──────────────────────────────────────────────
ENTRY_PCTILE = 90    # top 10% intensity = strong pressure
HOLD_PCTILE = 50     # hold while above median
EXIT_PCTILE = 40     # exit when below this

# Order flow direction thresholds
# cum_delta change over lookback window determines direction
DIRECTION_LOOKBACK = 50       # events to look back for delta change
MIN_DELTA_CHANGE = 20.0       # minimum cum_delta change to confirm direction
MIN_DEPTH_IMBAL = 0.05        # minimum depth imbalance magnitude for confirmation

# Trade management
MIN_EVENTS_BETWEEN_TRADES = 200
MAX_HOLD_EVENTS = 1200        # ~5 min at 4 events/sec (bounded by h=30s horizon)
MIN_HOLD_EVENTS = 20
REVERSAL_STOP_TICKS = 4.0     # hard stop on adverse move
TRAILING_EXIT_FRAC = 0.5      # exit if given back 50% of MFE

# Passive fill
PASSIVE_FILL_WINDOW = 40

# Book feature column indices
COL_CUM_DELTA = 20
COL_ROLL_IMBAL = 21
COL_DEPTH_IMBAL = 23
COL_NET_ORDER_FLOW = 29
COL_MID_PX_CHG = 27


def replay_subsampling(date_str: str, horizon_s: int) -> np.ndarray:
    """Replay the exact subsampling used during OOT prediction."""
    rl_path = RELABEL_DIR / f"mfe_mae_h{horizon_s}s_{date_str}.parquet"
    if not rl_path.exists():
        return np.array([], dtype=np.int64)

    rl = pd.read_parquet(rl_path, columns=["mfe_minus_mae_ticks"])
    target = rl["mfe_minus_mae_ticks"].values
    valid_idx = np.where(~np.isnan(target))[0]

    if len(valid_idx) == 0:
        return np.array([], dtype=np.int64)

    if len(valid_idx) > OOT_MAX_EVENTS:
        rng = np.random.default_rng(hash(date_str) % 2**32)
        chosen = rng.choice(valid_idx, OOT_MAX_EVENTS, replace=False)
        chosen.sort()
    else:
        chosen = valid_idx
    return chosen


def load_day_data(date_str: str) -> Optional[pd.DataFrame]:
    """
    Load predictions + timestamps + mid prices + book features for one OOS date.
    Returns DataFrame with: event_idx, ts_ns, mid_t_ticks, pred_30s/60s/120s/300s,
                            cum_delta, depth_imbal, net_order_flow, roll_imbal
    """
    # Load predictions
    horizon_preds = {}
    horizon_indices = {}

    for h in [30, 60, 120, 300]:
        pred_path = PRED_DIR / f"oos_preds_h{h}s.parquet"
        if not pred_path.exists():
            continue
        preds_all = pd.read_parquet(pred_path)
        day_preds = preds_all[preds_all["date"] == date_str]
        if len(day_preds) == 0:
            del preds_all
            continue

        indices = replay_subsampling(date_str, h)
        if len(indices) != len(day_preds):
            print(f"  WARNING: {date_str} h={h}s index mismatch")
            del preds_all
            continue

        horizon_preds[h] = day_preds["pred"].values
        horizon_indices[h] = indices
        del preds_all, day_preds
        gc.collect()

    if 30 not in horizon_preds:
        return None

    base_indices = horizon_indices[30]

    # Load relabel for timestamps and mid prices
    rl_path = RELABEL_DIR / f"mfe_mae_h30s_{date_str}.parquet"
    if not rl_path.exists():
        return None
    rl = pd.read_parquet(rl_path, columns=["event_idx", "ts_ns", "mid_t_ticks"])

    # Load book features for directional signal
    bk_path = BOOK_DIR / f"{date_str}_book_features.npz"
    if not bk_path.exists():
        print(f"  No book features for {date_str}")
        return None
    bk = np.load(bk_path, allow_pickle=True)
    bk_feats = bk["features"]  # (N, 30)

    # Extract directional features for base indices
    cum_delta = bk_feats[base_indices, COL_CUM_DELTA].astype(np.float64)
    depth_imbal = bk_feats[base_indices, COL_DEPTH_IMBAL].astype(np.float64)
    net_order_flow = bk_feats[base_indices, COL_NET_ORDER_FLOW].astype(np.float64)
    roll_imbal = bk_feats[base_indices, COL_ROLL_IMBAL].astype(np.float64)
    del bk, bk_feats
    gc.collect()

    rl_subset = rl.iloc[base_indices].copy().reset_index(drop=True)

    df = pd.DataFrame({
        "event_idx": rl_subset["event_idx"].values,
        "ts_ns": rl_subset["ts_ns"].values,
        "mid_t_ticks": rl_subset["mid_t_ticks"].values.astype(np.float64),
        "pred_30s": horizon_preds[30],
        "cum_delta": cum_delta,
        "depth_imbal": depth_imbal,
        "net_order_flow": net_order_flow,
        "roll_imbal": roll_imbal,
    })

    # Map other horizons
    for h in [60, 120, 300]:
        col = f"pred_{h}s"
        if h not in horizon_preds:
            df[col] = np.nan
            continue
        idx_to_pred = dict(zip(horizon_indices[h], horizon_preds[h]))
        df[col] = df["event_idx"].map(idx_to_pred).astype(np.float32)

    df = df.sort_values("event_idx").reset_index(drop=True)
    del rl
    gc.collect()
    return df


def compute_percentile_thresholds(df: pd.DataFrame) -> dict:
    """Compute per-day percentile thresholds for each horizon."""
    thresholds = {}
    for h in [30, 60, 120, 300]:
        col = f"pred_{h}s"
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        thresholds[h] = {
            "entry": np.percentile(valid, ENTRY_PCTILE),
            "hold": np.percentile(valid, HOLD_PCTILE),
            "exit": np.percentile(valid, EXIT_PCTILE),
        }
    return thresholds


def get_direction(cum_delta: np.ndarray, depth_imbal: np.ndarray,
                  roll_imbal: np.ndarray, i: int) -> Optional[int]:
    """
    Determine trade direction from order flow features.
    Returns +1 (long), -1 (short), or None (no clear direction).

    Uses:
    1. cum_delta change over lookback (primary signal)
    2. depth_imbal_5 (confirmation — positive = more bid depth = bullish)
    3. roll_imbal_100 (confirmation)
    """
    if i < DIRECTION_LOOKBACK:
        return None

    # Primary: cum_delta change
    delta_chg = cum_delta[i] - cum_delta[i - DIRECTION_LOOKBACK]

    if abs(delta_chg) < MIN_DELTA_CHANGE:
        return None

    direction = 1 if delta_chg > 0 else -1

    # Confirmation: depth imbalance should agree
    # depth_imbal > 0 means more depth on ask side (bearish for us? No...)
    # Actually depth_imbal_5 = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    # Positive = more bid depth = support = bullish
    di = depth_imbal[i]
    if abs(di) >= MIN_DEPTH_IMBAL:
        di_dir = 1 if di > 0 else -1
        if di_dir != direction:
            # Conflicting signals — skip
            return None

    return direction


def simulate_passive_fill(mid_prices: np.ndarray, entry_idx: int,
                          direction: int) -> Optional[int]:
    """
    Simulate passive limit order fill.
    LONG: buy at bid (mid - 0.5), fill when mid <= limit
    SHORT: sell at ask (mid + 0.5), fill when mid >= limit
    """
    entry_mid = mid_prices[entry_idx]
    end_idx = min(entry_idx + PASSIVE_FILL_WINDOW, len(mid_prices))

    if direction == 1:
        limit_price = entry_mid - 0.5
        for i in range(entry_idx + 1, end_idx):
            if mid_prices[i] <= limit_price:
                return i
    else:
        limit_price = entry_mid + 0.5
        for i in range(entry_idx + 1, end_idx):
            if mid_prices[i] >= limit_price:
                return i
    return None


def run_day_simulation(df: pd.DataFrame, date_str: str,
                       thresholds: dict) -> list[dict]:
    """Run streaming trade simulation for one day."""
    trades = []
    n = len(df)
    i = 0
    last_exit_idx = -MIN_EVENTS_BETWEEN_TRADES

    # Extract arrays for speed
    mid_prices = df["mid_t_ticks"].values
    pred_30 = df["pred_30s"].values
    pred_60 = df["pred_60s"].values if "pred_60s" in df.columns else np.full(n, np.nan)
    pred_120 = df["pred_120s"].values if "pred_120s" in df.columns else np.full(n, np.nan)
    ts_ns = df["ts_ns"].values
    cum_delta = df["cum_delta"].values
    depth_imbal = df["depth_imbal"].values
    roll_imbal = df["roll_imbal"].values

    # Thresholds
    entry_30 = thresholds.get(30, {}).get("entry", np.inf)
    entry_60 = thresholds.get(60, {}).get("entry", np.inf)
    exit_30 = thresholds.get(30, {}).get("exit", -np.inf)
    hold_60 = thresholds.get(60, {}).get("hold", -np.inf)
    hold_120 = thresholds.get(120, {}).get("hold", -np.inf)

    while i < n - MAX_HOLD_EVENTS:
        # Cooldown between trades
        if i - last_exit_idx < MIN_EVENTS_BETWEEN_TRADES:
            i += 1
            continue

        # ── Entry Gate: high intensity predicted ──
        p30 = pred_30[i]
        p60 = pred_60[i]
        has_entry = (not np.isnan(p30) and p30 >= entry_30) or \
                    (not np.isnan(p60) and p60 >= entry_60)
        if not has_entry:
            i += 1
            continue

        # ── Direction from order flow ──
        direction = get_direction(cum_delta, depth_imbal, roll_imbal, i)
        if direction is None:
            i += 1
            continue

        # ── Passive Entry ──
        fill_idx = simulate_passive_fill(mid_prices, i, direction)
        if fill_idx is None:
            i += 1
            continue

        entry_price = mid_prices[fill_idx]
        entry_ts = ts_ns[fill_idx]
        entry_delta = cum_delta[fill_idx]

        # ── Hold & Exit Loop ──
        j = fill_idx + MIN_HOLD_EVENTS
        exit_reason = "max_hold"
        max_favorable = 0.0
        max_adverse = 0.0

        while j < min(fill_idx + MAX_HOLD_EVENTS, n):
            current_price = mid_prices[j]
            pnl_ticks = (current_price - entry_price) * direction

            if pnl_ticks > max_favorable:
                max_favorable = pnl_ticks
            if pnl_ticks < max_adverse:
                max_adverse = pnl_ticks

            # Exit 1: Hard stop
            if pnl_ticks < -REVERSAL_STOP_TICKS:
                exit_reason = "hard_stop"
                break

            # Exit 2: Trailing exit — gave back too much MFE
            if max_favorable > 2.0 and pnl_ticks < max_favorable * TRAILING_EXIT_FRAC:
                exit_reason = "trailing"
                break

            # Exit 3: Intensity fade (h=30s drops below exit threshold)
            p30_j = pred_30[j]
            if not np.isnan(p30_j) and p30_j < exit_30:
                exit_reason = "fade_30s"
                break

            # Exit 4: Multi-horizon disagreement
            p60_j = pred_60[j]
            p120_j = pred_120[j]
            if (not np.isnan(p60_j) and p60_j < hold_60 and
                not np.isnan(p120_j) and p120_j < hold_120):
                exit_reason = "multi_h_fade"
                break

            # Exit 5: Order flow reversal — cum_delta moving against us
            delta_since_entry = cum_delta[j] - entry_delta
            if direction == 1 and delta_since_entry < -MIN_DELTA_CHANGE * 2:
                exit_reason = "flow_reversal"
                break
            if direction == -1 and delta_since_entry > MIN_DELTA_CHANGE * 2:
                exit_reason = "flow_reversal"
                break

            j += 1

        # Record trade
        exit_idx = min(j, n - 1)
        exit_price = mid_prices[exit_idx]
        exit_ts = ts_ns[exit_idx]
        gross_ticks = (exit_price - entry_price) * direction
        net_ticks = gross_ticks - TOTAL_RT_COST
        hold_time_s = (exit_ts - entry_ts) / 1e9

        trades.append({
            "date": date_str,
            "direction": direction,
            "entry_idx": int(fill_idx),
            "exit_idx": int(exit_idx),
            "entry_price_t": float(entry_price),
            "exit_price_t": float(exit_price),
            "gross_ticks": float(gross_ticks),
            "net_ticks": float(net_ticks),
            "mfe_ticks": float(max_favorable),
            "mae_ticks": float(max_adverse),
            "hold_events": exit_idx - fill_idx,
            "hold_time_s": float(hold_time_s),
            "exit_reason": exit_reason,
            "entry_pred_30s": float(pred_30[i]),
            "entry_pred_60s": float(p60) if not np.isnan(p60) else None,
            "entry_cum_delta_chg": float(cum_delta[i] - cum_delta[max(0, i - DIRECTION_LOOKBACK)]),
            "entry_depth_imbal": float(depth_imbal[i]),
        })

        last_exit_idx = exit_idx
        i = exit_idx + 1

    return trades


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute risk-adjusted performance metrics."""
    if len(trades_df) == 0:
        return {"n_trades": 0}

    net = trades_df["net_ticks"].values
    gross = trades_df["gross_ticks"].values

    n_trades = len(net)
    total_gross = float(gross.sum())
    total_net = float(net.sum())
    winners = (net > 0).sum()
    losers = (net < 0).sum()
    win_rate = winners / n_trades

    gross_profit = float(net[net > 0].sum()) if winners > 0 else 0
    gross_loss = float(abs(net[net < 0].sum())) if losers > 0 else 1e-9
    profit_factor = gross_profit / gross_loss

    daily_pnl = trades_df.groupby("date")["net_ticks"].sum()
    if len(daily_pnl) > 1:
        daily_mean = daily_pnl.mean()
        daily_std = daily_pnl.std()
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0
        downside_returns = daily_pnl[daily_pnl < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else daily_std
        sortino = (daily_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    else:
        sharpe = sortino = 0

    avg_win = float(net[net > 0].mean()) if winners > 0 else 0
    avg_loss = float(net[net < 0].mean()) if losers > 0 else 0

    return {
        "n_trades": n_trades,
        "n_days": len(daily_pnl),
        "trades_per_day": n_trades / max(len(daily_pnl), 1),
        "total_gross_ticks": total_gross,
        "total_net_ticks": total_net,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_annualized": sharpe,
        "sortino_annualized": sortino,
        "avg_win_ticks": avg_win,
        "avg_loss_ticks": avg_loss,
        "avg_hold_s": float(trades_df["hold_time_s"].mean()),
        "avg_mfe_ticks": float(trades_df["mfe_ticks"].mean()),
        "avg_mae_ticks": float(trades_df["mae_ticks"].mean()),
        "total_net_dollars": total_net * ES_TICK_VALUE,
    }


def regime_analysis(trades_df: pd.DataFrame) -> dict:
    """Regime-agnostic validation per HC #428."""
    if len(trades_df) == 0:
        return {}

    trades_df = trades_df.copy()
    trades_df["month"] = trades_df["date"].str[:6]

    regime_metrics = {}
    for m in sorted(trades_df["month"].unique()):
        regime_metrics[m] = compute_metrics(trades_df[trades_df["month"] == m])

    sharpes = {m: v.get("sharpe_annualized", 0) for m, v in regime_metrics.items()
               if v.get("n_trades", 0) > 5}

    if len(sharpes) >= 2:
        vals = list(sharpes.values())
        max_abs = max(abs(s) for s in vals)
        divergence = (max(vals) - min(vals)) / max_abs if max_abs > 0 else 0
        regime_metrics["_divergence"] = divergence
        regime_metrics["_pass_hc428"] = divergence <= 0.50
    else:
        regime_metrics["_divergence"] = None
        regime_metrics["_pass_hc428"] = None

    return regime_metrics


def print_results(metrics: dict, regime: dict, trades_df: pd.DataFrame):
    """Pretty-print simulation results."""
    print("\n" + "=" * 70)
    print("STREAMING TRADE SIMULATOR v1 — RESULTS")
    print("=" * 70)

    print(f"\n  Total trades:      {metrics['n_trades']}")
    print(f"  Trading days:      {metrics.get('n_days', 0)}")
    print(f"  Trades/day:        {metrics.get('trades_per_day', 0):.1f}")
    print(f"  Win rate:          {metrics.get('win_rate', 0):.1%}")
    print(f"  Profit factor:     {metrics.get('profit_factor', 0):.2f}")
    print(f"\n  Sharpe (ann.):     {metrics.get('sharpe_annualized', 0):.2f}")
    print(f"  Sortino (ann.):    {metrics.get('sortino_annualized', 0):.2f}")
    print(f"\n  Total gross:       {metrics.get('total_gross_ticks', 0):.1f} ticks")
    print(f"  Total net:         {metrics.get('total_net_ticks', 0):.1f} ticks")
    print(f"  Total net ($):     ${metrics.get('total_net_dollars', 0):.2f}")
    print(f"\n  Avg win:           {metrics.get('avg_win_ticks', 0):.2f} ticks")
    print(f"  Avg loss:          {metrics.get('avg_loss_ticks', 0):.2f} ticks")
    print(f"  Avg hold:          {metrics.get('avg_hold_s', 0):.1f} seconds")
    print(f"  Avg MFE:           {metrics.get('avg_mfe_ticks', 0):.2f} ticks")
    print(f"  Avg MAE:           {metrics.get('avg_mae_ticks', 0):.2f} ticks")

    print(f"\n  Cost model:")
    print(f"    Passive entry:   {PASSIVE_ENTRY_COST:.3f} ticks")
    print(f"    Market exit:     {MARKET_EXIT_COST:.3f} ticks")
    print(f"    Total RT:        {TOTAL_RT_COST:.3f} ticks")
    print(f"    Total costs:     {metrics['n_trades'] * TOTAL_RT_COST:.1f} ticks")

    if len(trades_df) > 0:
        print(f"\n  Exit reasons:")
        for reason, count in trades_df["exit_reason"].value_counts().items():
            pct = count / len(trades_df)
            subset = trades_df[trades_df["exit_reason"] == reason]
            avg_net = subset["net_ticks"].mean()
            wr = (subset["net_ticks"] > 0).mean()
            print(f"    {reason:20s}: {count:5d} ({pct:.1%}) | WR={wr:.1%} | avg net: {avg_net:+.2f}t")

        print(f"\n  Direction breakdown:")
        for d, label in [(1, "LONG"), (-1, "SHORT")]:
            subset = trades_df[trades_df["direction"] == d]
            if len(subset) > 0:
                wr = (subset["net_ticks"] > 0).mean()
                avg = subset["net_ticks"].mean()
                total = subset["net_ticks"].sum()
                print(f"    {label:6s}: {len(subset):5d} trades | WR={wr:.1%} | "
                      f"avg={avg:+.2f}t | total={total:+.1f}t")

    print(f"\n  Regime analysis (HC #428):")
    for m, v in sorted(regime.items()):
        if m.startswith("_"):
            continue
        if isinstance(v, dict) and v.get("n_trades", 0) > 0:
            print(f"    {m}: {v['n_trades']:4d} trades | WR={v.get('win_rate',0):.1%} | "
                  f"Sharpe={v.get('sharpe_annualized',0):+.2f} | "
                  f"PF={v.get('profit_factor',0):.2f} | "
                  f"net={v.get('total_net_ticks',0):+.1f}t")

    div = regime.get("_divergence")
    passes = regime.get("_pass_hc428")
    if div is not None:
        status = "PASS" if passes else "FAIL"
        print(f"\n    Regime Sharpe divergence: {div:.2f} {status} (threshold: 0.50)")

    if len(trades_df) > 0:
        print(f"\n  Per-day breakdown:")
        daily = trades_df.groupby("date").agg(
            n_trades=("net_ticks", "count"),
            gross=("gross_ticks", "sum"),
            net=("net_ticks", "sum"),
            wr=("net_ticks", lambda x: (x > 0).mean()),
        )
        for dt, row in daily.iterrows():
            marker = "+" if row["net"] > 0 else "-"
            print(f"    {dt}: {int(row['n_trades']):3d} trades | "
                  f"gross={row['gross']:+7.1f}t | net={row['net']:+7.1f}t | "
                  f"WR={row['wr']:.0%} {marker}")

    print("\n" + "=" * 70)


def main():
    t_start = time.time()
    print("=" * 70)
    print("STREAMING TRADE SIMULATOR v1 — ORDER FLOW DIRECTION + INTENSITY GATE")
    print(f"Entry: intensity top {100-ENTRY_PCTILE}% AND order flow direction confirmed")
    print(f"  Direction: cum_delta change > {MIN_DELTA_CHANGE} over {DIRECTION_LOOKBACK} events")
    print(f"  Confirmation: depth_imbal_5 must agree (>= {MIN_DEPTH_IMBAL})")
    print(f"Hold:  intensity above p{HOLD_PCTILE}")
    print(f"Exit:  intensity fade | multi-h fade | flow reversal | trailing | hard stop")
    print(f"Cost:  {TOTAL_RT_COST:.3f}t RT (passive entry + market exit)")
    print(f"Stops: hard={REVERSAL_STOP_TICKS}t | trailing={TRAILING_EXIT_FRAC:.0%} of MFE")
    print("=" * 70)

    # Discover OOS dates
    preds_30 = pd.read_parquet(PRED_DIR / "oos_preds_h30s.parquet")
    oos_dates = sorted(preds_30["date"].unique())
    print(f"\nOOS dates ({len(oos_dates)}): {oos_dates}")
    del preds_30
    gc.collect()

    # Filter to dates with book features
    available_book_dates = {p.stem.split("_")[0]
                           for p in BOOK_DIR.glob("2026*_book_features.npz")}
    oos_dates = [d for d in oos_dates if d in available_book_dates]
    print(f"Dates with book features: {len(oos_dates)}")

    # MLflow setup
    HAS_MLFLOW_RUN = False
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run(run_name=f"sim_v1_flow_entry{ENTRY_PCTILE}_exit{EXIT_PCTILE}")
            mlflow.log_params({
                "entry_pctile": ENTRY_PCTILE,
                "hold_pctile": HOLD_PCTILE,
                "exit_pctile": EXIT_PCTILE,
                "direction_lookback": DIRECTION_LOOKBACK,
                "min_delta_change": MIN_DELTA_CHANGE,
                "min_depth_imbal": MIN_DEPTH_IMBAL,
                "min_events_between": MIN_EVENTS_BETWEEN_TRADES,
                "max_hold_events": MAX_HOLD_EVENTS,
                "reversal_stop_ticks": REVERSAL_STOP_TICKS,
                "trailing_exit_frac": TRAILING_EXIT_FRAC,
                "rt_cost_ticks": TOTAL_RT_COST,
            })
            HAS_MLFLOW_RUN = True
        except Exception as e:
            print(f"MLflow setup failed: {e}")

    # Process each day
    all_trades = []
    for date_str in oos_dates:
        t_day = time.time()
        print(f"\n--- {date_str} ---")

        df = load_day_data(date_str)
        if df is None:
            print(f"  No data, skip")
            continue

        n_30 = (~df["pred_30s"].isna()).sum()
        n_60 = (~df["pred_60s"].isna()).sum() if "pred_60s" in df.columns else 0
        n_120 = (~df["pred_120s"].isna()).sum() if "pred_120s" in df.columns else 0
        print(f"  Loaded {len(df):,} events | h30={n_30:,} | h60={n_60:,} | h120={n_120:,}")

        thresholds = compute_percentile_thresholds(df)
        if 30 not in thresholds:
            print(f"  No h=30s thresholds, skip")
            continue

        print(f"  Thresholds h=30s: entry={thresholds[30]['entry']:.2f} "
              f"hold={thresholds[30]['hold']:.2f} exit={thresholds[30]['exit']:.2f}")

        day_trades = run_day_simulation(df, date_str, thresholds)
        all_trades.extend(day_trades)

        if day_trades:
            net_sum = sum(t["net_ticks"] for t in day_trades)
            wr = sum(1 for t in day_trades if t["net_ticks"] > 0) / len(day_trades)
            n_long = sum(1 for t in day_trades if t["direction"] == 1)
            n_short = len(day_trades) - n_long
            print(f"  {len(day_trades)} trades (L={n_long} S={n_short}) | "
                  f"net={net_sum:+.1f}t | WR={wr:.1%} | ({time.time()-t_day:.1f}s)")
        else:
            print(f"  0 trades ({time.time()-t_day:.1f}s)")

        del df
        gc.collect()

    # Aggregate
    if not all_trades:
        print("\nNO TRADES GENERATED.")
        if HAS_MLFLOW_RUN:
            mlflow.log_metric("n_trades", 0)
            mlflow.end_run()
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_parquet(OUTPUT_DIR / "trades_v1.parquet", index=False)
    trades_df.to_csv(OUTPUT_DIR / "trades_v1.csv", index=False)
    print(f"\nSaved {len(trades_df)} trades")

    metrics = compute_metrics(trades_df)
    regime = regime_analysis(trades_df)
    print_results(metrics, regime, trades_df)

    results = {
        "params": {
            "entry_pctile": ENTRY_PCTILE, "hold_pctile": HOLD_PCTILE,
            "exit_pctile": EXIT_PCTILE, "direction_lookback": DIRECTION_LOOKBACK,
            "min_delta_change": MIN_DELTA_CHANGE, "min_depth_imbal": MIN_DEPTH_IMBAL,
            "min_events_between": MIN_EVENTS_BETWEEN_TRADES,
            "max_hold_events": MAX_HOLD_EVENTS,
            "reversal_stop_ticks": REVERSAL_STOP_TICKS,
            "trailing_exit_frac": TRAILING_EXIT_FRAC,
            "rt_cost_ticks": TOTAL_RT_COST,
        },
        "metrics": metrics,
        "regime": {k: v for k, v in regime.items()
                   if not isinstance(v, dict) or k.startswith("_")},
        "regime_detail": {k: v for k, v in regime.items()
                          if isinstance(v, dict) and not k.startswith("_")},
    }
    with open(OUTPUT_DIR / "results_v1.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if HAS_MLFLOW_RUN:
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(str(OUTPUT_DIR / "results_v1.json"))
            mlflow.log_artifact(str(OUTPUT_DIR / "trades_v1.csv"))
        except Exception as e:
            print(f"MLflow logging error: {e}")
        finally:
            mlflow.end_run()

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
