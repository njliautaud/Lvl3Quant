#!/usr/bin/env python3
"""
definitive_tick_replay.py — TRUE tick-level TP/SL replay using raw MBO data

Resolves the ambiguity in minute-bar backtests where TP and SL can both be
within the same bar's range. Uses Databento MBO data to reconstruct BBO
tick-by-tick and determine EXACT ordering of TP/SL hits.

Runs on Neptune (RTX 3090, Ubuntu) using py311-train conda env for databento.

Usage:
    /home/nick/miniconda3/envs/py311-train/bin/python3 definitive_tick_replay.py
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import databento as dbn
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    print("ERROR: databento not available. Use py311-train conda env.", file=sys.stderr)
    sys.exit(1)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LVL3 = Path("/home/nick/Lvl3Quant")
MBO_DIRS = [
    LVL3 / "data/raw/mbo_files",
    LVL3 / "data/raw/mbo",
]
MINUTE_BARS_DIR = LVL3 / "data/processed/mbo_minute_bars_v1"
TRADES_FILE = LVL3 / "output/integrated_pipeline_v1/best_trades.parquet"
PRED_FILE = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
OUTPUT_DIR = LVL3 / "output/definitive_tick_replay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25  # ES tick size in points
TICK_VALUE = 12.50  # $ per tick

# FIFO cost model (canonical from CLAUDE.md)
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50
SPREAD_TICKS = 1.0  # 1 tick crossing cost for market orders
COST_PASSIVE = COMMISSION_RT_TICKS  # limit fill: commission only
COST_MARKET = COMMISSION_RT_TICKS + SPREAD_TICKS  # market: commission + spread

# TP/SL configs to test
CONFIGS = [
    {"name": "TP20_SL10", "tp_ticks": 20, "sl_ticks": 10},
    {"name": "TP30_SL15", "tp_ticks": 30, "sl_ticks": 15},
    {"name": "TP10_SL5",  "tp_ticks": 10, "sl_ticks": 5},
]

MAX_HOLD_NS = 30 * 60 * 1_000_000_000  # 30 minutes in nanoseconds

# ES contract rollover (from process_missing_mbo.py)
ES_CONTRACTS = [
    ("2025-09-19", 14160),       # ESU5
    ("2025-12-19", 294973),      # ESZ5
    ("2026-03-20", 42140878),    # ESH6
    ("2026-06-19", None),        # ESM6
]

MLFLOW_URI = "http://localhost:5000"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE = OUTPUT_DIR / "definitive_tick_replay.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_instrument_id(date_str: str):
    """Get the front-month ES instrument_id for a given date."""
    d = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    for cutoff, iid in ES_CONTRACTS:
        if d < cutoff:
            return iid
    return ES_CONTRACTS[-1][1]


def find_mbo_file(date_str: str) -> Path:
    """Find the raw MBO file for a given date across both directories."""
    for mbo_dir in MBO_DIRS:
        fpath = mbo_dir / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        if fpath.exists():
            return fpath
    return None


def auto_detect_front_month(df, date_str: str):
    """Auto-detect front-month ES contract by trade volume."""
    # Filter to ES-like symbols (not spreads)
    es_mask = df["symbol"].str.match(r"^ES[A-Z]\d$", na=False)
    es_df = df[es_mask]
    if len(es_df) == 0:
        # Try broader match
        es_df = df[df["symbol"].str.startswith("ES") & ~df["symbol"].str.contains("-")]
    if len(es_df) == 0:
        return None

    trades = es_df[es_df["action"] == "T"]
    if len(trades) > 0:
        iid = trades["instrument_id"].value_counts().idxmax()
        return int(iid)
    iid = es_df["instrument_id"].value_counts().idxmax()
    return int(iid)


def load_mbo_mid_prices(date_str: str):
    """
    Load raw MBO data for a date and reconstruct BBO mid-price series.

    Returns:
        mid_prices: np.array of mid-prices (in ES points, e.g. 5780.25)
        timestamps: np.array of int64 nanosecond timestamps
        bid_prices: np.array of best bid prices
        ask_prices: np.array of best ask prices

    Returns (None, None, None, None) if data unavailable.
    """
    fpath = find_mbo_file(date_str)
    if fpath is None:
        return None, None, None, None

    store = dbn.DBNStore.from_file(str(fpath))
    df = store.to_df()
    df.columns = [c.lower() for c in df.columns]

    # Filter to front-month ES contract
    iid = get_instrument_id(date_str)
    if iid is None:
        iid = auto_detect_front_month(df, date_str)
    if iid is not None and "instrument_id" in df.columns:
        df = df[df["instrument_id"] == iid]

    if len(df) == 0:
        return None, None, None, None

    # Price scaling (databento fixed-point)
    if "price" in df.columns and len(df) > 0:
        sample = df["price"].dropna().iloc[0] if len(df["price"].dropna()) > 0 else 0
        if sample > 1e6:
            df["price"] = df["price"] * 1e-9

    df = df.sort_values("ts_event").reset_index(drop=True)
    df = df[df["action"].isin(["A", "C", "M", "T", "F"])].reset_index(drop=True)

    n = len(df)
    if n == 0:
        return None, None, None, None

    timestamps = df["ts_event"].values.astype("int64")
    prices = df["price"].values.astype(np.float64)
    actions = df["action"].values
    sides = df["side"].values

    # BBO tracking — reconstruct mid-prices
    mid_arr = np.full(n, np.nan, dtype=np.float64)
    bid_arr = np.full(n, np.nan, dtype=np.float64)
    ask_arr = np.full(n, np.nan, dtype=np.float64)

    best_bid = np.nan
    best_ask = np.nan

    # Use dict-based LOB for accuracy
    bid_levels = {}
    ask_levels = {}

    for i in range(n):
        p = prices[i]
        act = actions[i]
        side = sides[i]

        if not np.isnan(p) and p > 0:
            if act == "T" or act == "F":
                # Trade/fill: removes liquidity from the opposite side
                if side == "A":
                    # Buy aggressor hit the ask, so ask level was filled
                    best_bid = p  # trade at ask means bid was at least here
                elif side == "B":
                    # Sell aggressor hit the bid
                    best_ask = p
            elif act == "A":
                # Add order
                if side == "B":
                    bid_levels[p] = bid_levels.get(p, 0) + 1
                    if np.isnan(best_bid) or p > best_bid:
                        best_bid = p
                elif side == "A":
                    ask_levels[p] = ask_levels.get(p, 0) + 1
                    if np.isnan(best_ask) or p < best_ask:
                        best_ask = p
            elif act == "C":
                # Cancel order
                if side == "B":
                    if p in bid_levels:
                        bid_levels[p] -= 1
                        if bid_levels[p] <= 0:
                            del bid_levels[p]
                    if not np.isnan(best_bid) and p >= best_bid:
                        valid_bids = [k for k, v in bid_levels.items() if v > 0]
                        best_bid = max(valid_bids) if valid_bids else np.nan
                elif side == "A":
                    if p in ask_levels:
                        ask_levels[p] -= 1
                        if ask_levels[p] <= 0:
                            del ask_levels[p]
                    if not np.isnan(best_ask) and p <= best_ask:
                        valid_asks = [k for k, v in ask_levels.items() if v > 0]
                        best_ask = min(valid_asks) if valid_asks else np.nan
            elif act == "M":
                # Modify — treat as potential new level
                if side == "B":
                    if np.isnan(best_bid) or p > best_bid:
                        best_bid = p
                elif side == "A":
                    if np.isnan(best_ask) or p < best_ask:
                        best_ask = p

        if not np.isnan(best_bid) and not np.isnan(best_ask) and best_ask > best_bid:
            mid_arr[i] = (best_bid + best_ask) / 2.0
            bid_arr[i] = best_bid
            ask_arr[i] = best_ask
        elif i > 0:
            mid_arr[i] = mid_arr[i - 1]
            bid_arr[i] = bid_arr[i - 1]
            ask_arr[i] = ask_arr[i - 1]

        # Periodic cleanup of distant levels
        if i % 100000 == 0 and not np.isnan(mid_arr[i]):
            mid = mid_arr[i]
            radius = 50 * TICK_SIZE
            bid_levels = {k: v for k, v in bid_levels.items()
                         if abs(k - mid) <= radius and v > 0}
            ask_levels = {k: v for k, v in ask_levels.items()
                         if abs(k - mid) <= radius and v > 0}

    # Forward-fill NaNs
    for i in range(1, n):
        if np.isnan(mid_arr[i]):
            mid_arr[i] = mid_arr[i - 1]
            bid_arr[i] = bid_arr[i - 1]
            ask_arr[i] = ask_arr[i - 1]

    return mid_arr, timestamps, bid_arr, ask_arr


def build_30min_bars(date_str):
    """Build 30-min bars from minute data (same logic as tick_replay_validation_v2)."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None

    mbars = pd.read_parquet(fpath)
    mbars['ts_minute'] = pd.to_datetime(mbars['ts_minute'], utc=True)
    mbars = mbars.sort_values('ts_minute').reset_index(drop=True)
    mbars['bar_key'] = mbars['ts_minute'].dt.floor('30min')

    bars_30 = []
    for bar_key, grp in mbars.groupby('bar_key'):
        if len(grp) < 3:
            continue
        bars_30.append({
            'bar_key': bar_key,
            'open': grp['open'].iloc[0],
            'high': grp['high'].max(),
            'low': grp['low'].min(),
            'close': grp['close'].iloc[-1],
            'volume': grp['volume'].sum(),
        })

    return pd.DataFrame(bars_30).sort_values('bar_key').reset_index(drop=True)


def get_unique_bar_count(date_str, pred_dates):
    """Figure out how many UNIQUE bars this date has in predictions."""
    mask = pred_dates == date_str
    n_total = mask.sum()
    if n_total <= 15:
        return n_total
    if n_total == 28:
        return 14
    elif n_total == 21:
        return 7
    elif n_total == 16:
        return 8
    elif n_total == 18:
        return 9
    else:
        return min(n_total, 15)


def tick_replay_trade(mid_prices, timestamps, entry_price_points, direction,
                      entry_ts_ns, tp_ticks, sl_ticks, max_hold_ns):
    """
    TRUE tick-level replay of a single trade.

    Args:
        mid_prices: full day mid-price array (in ES points)
        timestamps: full day timestamp array (nanoseconds)
        entry_price_points: entry price in ES points
        direction: 1 (long) or -1 (short)
        entry_ts_ns: entry timestamp in nanoseconds
        tp_ticks: take-profit in ticks
        sl_ticks: stop-loss in ticks
        max_hold_ns: max hold time in nanoseconds

    Returns dict with trade result.
    """
    tp_points = tp_ticks * TICK_SIZE
    sl_points = sl_ticks * TICK_SIZE
    deadline_ns = entry_ts_ns + max_hold_ns

    # Find the starting index (first event at or after entry time)
    start_idx = np.searchsorted(timestamps, entry_ts_ns, side='left')
    if start_idx >= len(timestamps):
        return {
            'exit_type': 'no_data_after_entry',
            'exit_pnl_ticks': 0.0,
            'hold_ns': 0,
            'mfe_ticks': 0.0,
            'mae_ticks': 0.0,
            'n_ticks_scanned': 0,
        }

    mfe = 0.0  # max favorable excursion in ticks
    mae = 0.0  # max adverse excursion in ticks
    n_scanned = 0

    for i in range(start_idx, len(timestamps)):
        ts = timestamps[i]
        if ts > deadline_ns:
            # Timeout — exit at current mid
            exit_pnl = direction * (mid_prices[i] - entry_price_points) / TICK_SIZE
            return {
                'exit_type': 'timeout',
                'exit_pnl_ticks': float(exit_pnl),
                'exit_price': float(mid_prices[i]),
                'hold_ns': int(ts - entry_ts_ns),
                'mfe_ticks': float(mfe),
                'mae_ticks': float(mae),
                'n_ticks_scanned': n_scanned,
            }

        mid = mid_prices[i]
        if np.isnan(mid):
            continue

        n_scanned += 1
        move_ticks = direction * (mid - entry_price_points) / TICK_SIZE

        if move_ticks > mfe:
            mfe = move_ticks
        if -move_ticks > mae:
            mae = -move_ticks

        # Check TP
        if move_ticks >= tp_ticks:
            return {
                'exit_type': 'tp',
                'exit_pnl_ticks': float(tp_ticks),
                'exit_price': float(entry_price_points + direction * tp_points),
                'hold_ns': int(ts - entry_ts_ns),
                'mfe_ticks': float(mfe),
                'mae_ticks': float(mae),
                'n_ticks_scanned': n_scanned,
            }

        # Check SL
        if -move_ticks >= sl_ticks:
            return {
                'exit_type': 'sl',
                'exit_pnl_ticks': float(-sl_ticks),
                'exit_price': float(entry_price_points - direction * sl_points),
                'hold_ns': int(ts - entry_ts_ns),
                'mfe_ticks': float(mfe),
                'mae_ticks': float(mae),
                'n_ticks_scanned': n_scanned,
            }

    # Ran out of data before timeout
    exit_pnl = direction * (mid_prices[-1] - entry_price_points) / TICK_SIZE
    return {
        'exit_type': 'eod',
        'exit_pnl_ticks': float(exit_pnl),
        'exit_price': float(mid_prices[-1]),
        'hold_ns': int(timestamps[-1] - entry_ts_ns),
        'mfe_ticks': float(mfe),
        'mae_ticks': float(mae),
        'n_ticks_scanned': n_scanned,
    }


def apply_fifo_costs(exit_type, raw_pnl_ticks):
    """Apply FIFO cost model: passive entry, passive TP exit, market SL/timeout exit."""
    entry_cost = COST_PASSIVE  # Always passive limit entry
    if exit_type == 'tp':
        exit_cost = COST_PASSIVE  # TP is also passive limit
    else:
        exit_cost = COST_MARKET  # SL/timeout/eod are market orders

    total_cost = entry_cost + exit_cost
    net_pnl = raw_pnl_ticks - total_cost
    return net_pnl, total_cost


def compute_metrics(trades_df, label=""):
    """Compute strategy metrics from a DataFrame of trade results."""
    if len(trades_df) == 0:
        return {'label': label, 'n_trades': 0}

    n_trades = len(trades_df)
    total_pnl = float(trades_df['net_pnl_ticks'].sum())
    avg_pnl = float(trades_df['net_pnl_ticks'].mean())
    winners = (trades_df['net_pnl_ticks'] > 0).sum()
    wr = float(winners / n_trades)

    gross_profit = float(trades_df.loc[trades_df['net_pnl_ticks'] > 0, 'net_pnl_ticks'].sum())
    gross_loss = float(abs(trades_df.loc[trades_df['net_pnl_ticks'] < 0, 'net_pnl_ticks'].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9

    daily_pnl = trades_df.groupby('date')['net_pnl_ticks'].sum()
    n_days = len(daily_pnl)
    sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if n_days > 1 and daily_pnl.std() > 0 else 0.0

    downside = daily_pnl[daily_pnl < 0]
    sortino = float(daily_pnl.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else (99.9 if daily_pnl.mean() > 0 else 0.0)

    cumulative = daily_pnl.cumsum()
    max_dd = float((cumulative - cumulative.cummax()).min())

    # Day concentration
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else 1.0

    # Monthly breakdown
    trades_copy = trades_df.copy()
    trades_copy['month'] = trades_copy['date'].str[:6]
    monthly = trades_copy.groupby('month')['net_pnl_ticks'].agg(['sum', 'count', 'mean'])

    # Exit type distribution
    exit_dist = trades_df['exit_type'].value_counts().to_dict()

    return {
        'label': label,
        'n_trades': int(n_trades),
        'n_trading_days': int(n_days),
        'total_pnl_ticks': total_pnl,
        'total_pnl_dollars': total_pnl * TICK_VALUE,
        'avg_pnl_ticks': avg_pnl,
        'win_rate': wr,
        'profit_factor': min(float(pf), 99.9),
        'sharpe': sharpe,
        'sortino': min(float(sortino), 99.9),
        'max_dd_ticks': max_dd,
        'max_dd_dollars': max_dd * TICK_VALUE,
        'day_concentration': day_conc,
        'exit_distribution': exit_dist,
        'monthly_pnl': monthly.to_dict() if len(monthly) > 0 else {},
    }


def compute_regime_metrics(trades_df):
    """Regime stratification: green vs red vs flat days."""
    try:
        flow_df = pd.read_parquet(FLOW_FEATURES)
    except Exception:
        return {'regime_gap': np.nan, 'regime_pass': False}

    regime_map = {}
    for _, row in flow_df.iterrows():
        date_val = row.get('date', '')
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y%m%d')
        else:
            date_str = str(date_val).replace('-', '')[:8]
        cc = row.get('cc_return_ticks', 0)
        if pd.isna(cc):
            regime_map[date_str] = 'flat'
        elif cc > 20:
            regime_map[date_str] = 'green'
        elif cc < -20:
            regime_map[date_str] = 'red'
        else:
            regime_map[date_str] = 'flat'

    trades_copy = trades_df.copy()
    trades_copy['regime'] = trades_copy['date'].map(lambda d: regime_map.get(d, 'unknown'))

    results = {}
    for regime in ['green', 'red', 'flat']:
        rt = trades_copy[trades_copy['regime'] == regime]
        if len(rt) >= 3:
            results[regime] = compute_metrics(rt, regime)
        else:
            results[regime] = {'n_trades': len(rt), 'sharpe': 0}

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    mx = max(abs(sg), abs(sr), 0.01)
    gap = abs(sg - sr) / mx
    results['regime_gap'] = float(gap)
    results['regime_pass'] = bool(gap < 0.50)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 80)
    log.info("DEFINITIVE TICK-LEVEL REPLAY — Raw MBO BBO Reconstruction")
    log.info("=" * 80)

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("definitive_tick_replay")
            mlflow.start_run(run_name=f"tick_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            log.info("MLflow run started")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # ── Load trades ──
    trades = pd.read_parquet(TRADES_FILE)
    log.info(f"Loaded {len(trades)} curated trades from {trades['date'].nunique()} dates")

    # ── Load predictions for unfiltered set ──
    pred_data = np.load(PRED_FILE, allow_pickle=True)
    pred_preds = pred_data['preds']
    pred_dates = pred_data['dates']
    pred_confs = pred_data['confs']
    log.info(f"Loaded {len(pred_preds)} predictions for unfiltered analysis")

    # Build unfiltered trade list: every prediction becomes a trade
    # direction: pred > 0 means long, pred < 0 means short
    unfiltered_trades = []
    for i in range(len(pred_preds)):
        direction = 1 if pred_preds[i] > 0 else -1
        unfiltered_trades.append({
            'date': pred_dates[i],
            'direction': direction,
            'bar_idx': i,  # will be mapped per-date
            'pred': float(pred_preds[i]),
            'conf': float(pred_confs[i]),
        })
    unfiltered_df = pd.DataFrame(unfiltered_trades)
    # Reindex bar_idx within each date
    for date_str in unfiltered_df['date'].unique():
        mask = unfiltered_df['date'] == date_str
        n = mask.sum()
        unfiltered_df.loc[mask, 'bar_idx'] = list(range(n))
    unfiltered_df['bar_idx'] = unfiltered_df['bar_idx'].astype(int)
    log.info(f"Unfiltered: {len(unfiltered_df)} trades across {unfiltered_df['date'].nunique()} dates")

    # ── Unique bar counts for bar_idx mapping ──
    unique_bar_counts = {}
    for date_str in trades['date'].unique():
        unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)

    # ── Process each date ──
    all_dates = sorted(set(trades['date'].unique()) | set(unfiltered_df['date'].unique()))
    log.info(f"Processing {len(all_dates)} unique dates")

    # Cache: date -> (mid_prices, timestamps, bid_prices, ask_prices)
    date_results_curated = {cfg['name']: [] for cfg in CONFIGS}
    date_results_unfiltered = {cfg['name']: [] for cfg in CONFIGS}

    dates_with_data = 0
    dates_without_data = 0

    for di, date_str in enumerate(all_dates):
        log.info(f"[{di+1}/{len(all_dates)}] Processing {date_str}...")

        # Load 30-min bars for entry price mapping
        bars_30 = build_30min_bars(date_str)
        if bars_30 is None:
            log.warning(f"  No minute bars for {date_str}, skipping")
            dates_without_data += 1
            continue

        # Load raw MBO and reconstruct mid-prices
        mid_prices, timestamps, bid_prices, ask_prices = load_mbo_mid_prices(date_str)
        if mid_prices is None:
            log.warning(f"  No MBO data for {date_str}, skipping")
            dates_without_data += 1
            continue

        dates_with_data += 1
        valid_mids = np.count_nonzero(~np.isnan(mid_prices))
        log.info(f"  MBO: {len(mid_prices)} events, {valid_mids} valid mid-prices")
        log.info(f"  30-min bars: {len(bars_30)}")

        # Process curated trades for this date
        date_curated = trades[trades['date'] == date_str]
        for _, trade in date_curated.iterrows():
            bar_idx = trade['bar_idx']
            n_unique = unique_bar_counts.get(date_str, bar_idx + 1)
            actual_bar_idx = bar_idx % n_unique

            if actual_bar_idx >= len(bars_30):
                for cfg in CONFIGS:
                    date_results_curated[cfg['name']].append({
                        'date': date_str, 'direction': trade['direction'],
                        'bar_idx': bar_idx, 'exit_type': 'no_bar',
                        'net_pnl_ticks': 0.0, 'raw_pnl_ticks': 0.0,
                    })
                continue

            # Entry price: close of the 30-min bar, converted from ticks to points
            entry_price_ticks = bars_30.iloc[actual_bar_idx]['close']
            entry_price_points = entry_price_ticks * TICK_SIZE  # convert from tick-units to ES points

            # Entry time: end of the 30-min bar (start of next bar)
            bar_key = bars_30.iloc[actual_bar_idx]['bar_key']
            entry_time = bar_key + pd.Timedelta(minutes=30)
            entry_ts_ns = int(entry_time.value)

            direction = trade['direction']

            for cfg in CONFIGS:
                result = tick_replay_trade(
                    mid_prices, timestamps, entry_price_points, direction,
                    entry_ts_ns, cfg['tp_ticks'], cfg['sl_ticks'], MAX_HOLD_NS
                )
                net_pnl, cost = apply_fifo_costs(result['exit_type'], result['exit_pnl_ticks'])
                date_results_curated[cfg['name']].append({
                    'date': date_str,
                    'direction': direction,
                    'bar_idx': bar_idx,
                    'exit_type': result['exit_type'],
                    'raw_pnl_ticks': result['exit_pnl_ticks'],
                    'net_pnl_ticks': net_pnl,
                    'cost_ticks': cost,
                    'hold_ns': result.get('hold_ns', 0),
                    'mfe_ticks': result.get('mfe_ticks', 0),
                    'mae_ticks': result.get('mae_ticks', 0),
                    'n_ticks_scanned': result.get('n_ticks_scanned', 0),
                })

        # Process unfiltered trades for this date
        date_unfiltered = unfiltered_df[unfiltered_df['date'] == date_str]
        for _, trade in date_unfiltered.iterrows():
            bar_idx = trade['bar_idx']
            if bar_idx >= len(bars_30):
                for cfg in CONFIGS:
                    date_results_unfiltered[cfg['name']].append({
                        'date': date_str, 'direction': trade['direction'],
                        'bar_idx': bar_idx, 'exit_type': 'no_bar',
                        'net_pnl_ticks': 0.0, 'raw_pnl_ticks': 0.0,
                    })
                continue

            entry_price_ticks = bars_30.iloc[bar_idx]['close']
            entry_price_points = entry_price_ticks * TICK_SIZE

            bar_key = bars_30.iloc[bar_idx]['bar_key']
            entry_time = bar_key + pd.Timedelta(minutes=30)
            entry_ts_ns = int(entry_time.value)

            direction = trade['direction']

            for cfg in CONFIGS:
                result = tick_replay_trade(
                    mid_prices, timestamps, entry_price_points, direction,
                    entry_ts_ns, cfg['tp_ticks'], cfg['sl_ticks'], MAX_HOLD_NS
                )
                net_pnl, cost = apply_fifo_costs(result['exit_type'], result['exit_pnl_ticks'])
                date_results_unfiltered[cfg['name']].append({
                    'date': date_str,
                    'direction': direction,
                    'bar_idx': bar_idx,
                    'pred': trade.get('pred', np.nan),
                    'conf': trade.get('conf', np.nan),
                    'exit_type': result['exit_type'],
                    'raw_pnl_ticks': result['exit_pnl_ticks'],
                    'net_pnl_ticks': net_pnl,
                    'cost_ticks': cost,
                    'hold_ns': result.get('hold_ns', 0),
                    'mfe_ticks': result.get('mfe_ticks', 0),
                    'mae_ticks': result.get('mae_ticks', 0),
                    'n_ticks_scanned': result.get('n_ticks_scanned', 0),
                })

        if (di + 1) % 10 == 0:
            elapsed = time.time() - t0
            per_date = elapsed / (di + 1)
            remaining = per_date * (len(all_dates) - di - 1)
            log.info(f"  Progress: {di+1}/{len(all_dates)} dates, "
                     f"elapsed {elapsed/60:.1f}min, ETA {remaining/60:.1f}min")

    # ── Report Results ──
    log.info("")
    log.info("=" * 80)
    log.info("RESULTS")
    log.info("=" * 80)
    log.info(f"Dates with MBO data: {dates_with_data}, without: {dates_without_data}")

    all_metrics = {}

    for cfg in CONFIGS:
        cfg_name = cfg['name']
        log.info(f"\n{'─' * 60}")
        log.info(f"Config: {cfg_name} (TP={cfg['tp_ticks']}, SL={cfg['sl_ticks']})")
        log.info(f"{'─' * 60}")

        # Curated set
        curated_df = pd.DataFrame(date_results_curated[cfg_name])
        valid_curated = curated_df[curated_df['exit_type'].isin(['tp', 'sl', 'timeout', 'eod'])]
        log.info(f"\n  CURATED SET ({len(trades)} original, {len(valid_curated)} with tick data):")
        if len(valid_curated) > 0:
            m = compute_metrics(valid_curated, f"curated_{cfg_name}")
            all_metrics[f"curated_{cfg_name}"] = m
            log.info(f"    N trades:       {m['n_trades']}")
            log.info(f"    Sharpe:         {m['sharpe']:.2f}")
            log.info(f"    Sortino:        {m['sortino']:.2f}")
            log.info(f"    Win Rate:       {m['win_rate']:.1%}")
            log.info(f"    Profit Factor:  {m['profit_factor']:.2f}")
            log.info(f"    Avg PnL/trade:  {m['avg_pnl_ticks']:.2f} ticks")
            log.info(f"    Total PnL:      {m['total_pnl_ticks']:.1f} ticks (${m['total_pnl_dollars']:.0f})")
            log.info(f"    Max Drawdown:   {m['max_dd_ticks']:.1f} ticks (${m['max_dd_dollars']:.0f})")
            log.info(f"    Day Conc:       {m['day_concentration']:.2f}")
            log.info(f"    Exit dist:      {m['exit_distribution']}")

            # Regime
            regime = compute_regime_metrics(valid_curated)
            all_metrics[f"curated_{cfg_name}_regime"] = regime
            for r in ['green', 'red', 'flat']:
                if r in regime and regime[r].get('n_trades', 0) > 0:
                    log.info(f"    {r.upper():6s}: n={regime[r].get('n_trades',0):3d}, "
                             f"Sharpe={regime[r].get('sharpe',0):.2f}, "
                             f"WR={regime[r].get('win_rate',0):.1%}")
            log.info(f"    Regime gap:     {regime.get('regime_gap', 'N/A')}")
            log.info(f"    Regime pass:    {regime.get('regime_pass', 'N/A')}")

        # Unfiltered set
        unfiltered_result_df = pd.DataFrame(date_results_unfiltered[cfg_name])
        valid_unfiltered = unfiltered_result_df[unfiltered_result_df['exit_type'].isin(['tp', 'sl', 'timeout', 'eod'])]
        log.info(f"\n  UNFILTERED SET ({len(unfiltered_df)} original, {len(valid_unfiltered)} with tick data):")
        if len(valid_unfiltered) > 0:
            m = compute_metrics(valid_unfiltered, f"unfiltered_{cfg_name}")
            all_metrics[f"unfiltered_{cfg_name}"] = m
            log.info(f"    N trades:       {m['n_trades']}")
            log.info(f"    Sharpe:         {m['sharpe']:.2f}")
            log.info(f"    Sortino:        {m['sortino']:.2f}")
            log.info(f"    Win Rate:       {m['win_rate']:.1%}")
            log.info(f"    Profit Factor:  {m['profit_factor']:.2f}")
            log.info(f"    Avg PnL/trade:  {m['avg_pnl_ticks']:.2f} ticks")
            log.info(f"    Total PnL:      {m['total_pnl_ticks']:.1f} ticks (${m['total_pnl_dollars']:.0f})")
            log.info(f"    Max Drawdown:   {m['max_dd_ticks']:.1f} ticks (${m['max_dd_dollars']:.0f})")
            log.info(f"    Day Conc:       {m['day_concentration']:.2f}")
            log.info(f"    Exit dist:      {m['exit_distribution']}")

            regime = compute_regime_metrics(valid_unfiltered)
            all_metrics[f"unfiltered_{cfg_name}_regime"] = regime
            for r in ['green', 'red', 'flat']:
                if r in regime and regime[r].get('n_trades', 0) > 0:
                    log.info(f"    {r.upper():6s}: n={regime[r].get('n_trades',0):3d}, "
                             f"Sharpe={regime[r].get('sharpe',0):.2f}, "
                             f"WR={regime[r].get('win_rate',0):.1%}")
            log.info(f"    Regime gap:     {regime.get('regime_gap', 'N/A')}")
            log.info(f"    Regime pass:    {regime.get('regime_pass', 'N/A')}")

        # Save trade-level results
        curated_df.to_parquet(OUTPUT_DIR / f"curated_trades_{cfg_name}.parquet", index=False)
        unfiltered_result_df.to_parquet(OUTPUT_DIR / f"unfiltered_trades_{cfg_name}.parquet", index=False)

    # ── Monthly breakdown for primary config ──
    primary_cfg = CONFIGS[0]['name']
    log.info(f"\n{'─' * 60}")
    log.info(f"MONTHLY BREAKDOWN — {primary_cfg}")
    log.info(f"{'─' * 60}")

    for set_name, results_dict in [("CURATED", date_results_curated), ("UNFILTERED", date_results_unfiltered)]:
        df = pd.DataFrame(results_dict[primary_cfg])
        valid = df[df['exit_type'].isin(['tp', 'sl', 'timeout', 'eod'])]
        if len(valid) == 0:
            continue
        valid = valid.copy()
        valid['month'] = valid['date'].str[:6]
        log.info(f"\n  {set_name}:")
        log.info(f"  {'Month':8s} {'N':>5s} {'PnL':>8s} {'WR':>6s} {'AvgPnL':>8s}")
        for month, grp in valid.groupby('month'):
            n = len(grp)
            pnl = grp['net_pnl_ticks'].sum()
            wr = (grp['net_pnl_ticks'] > 0).mean()
            avg = grp['net_pnl_ticks'].mean()
            log.info(f"  {month:8s} {n:5d} {pnl:8.1f} {wr:6.1%} {avg:8.2f}")

    # ── Save summary ──
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'dates_processed': dates_with_data,
        'dates_missing': dates_without_data,
        'configs': [c['name'] for c in CONFIGS],
        'metrics': {},
    }
    for k, v in all_metrics.items():
        # Convert numpy types for JSON serialization
        summary['metrics'][k] = {
            mk: (float(mv) if isinstance(mv, (np.floating, np.integer)) else mv)
            for mk, mv in v.items()
        }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ── MLflow logging ──
    if MLFLOW_AVAILABLE:
        try:
            for k, v in all_metrics.items():
                if isinstance(v, dict):
                    for mk, mv in v.items():
                        if isinstance(mv, (int, float, np.floating, np.integer)):
                            mlflow.log_metric(f"{k}_{mk}", float(mv))
            mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))
            mlflow.log_artifact(str(LOG_FILE))
            mlflow.end_run()
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    log.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
