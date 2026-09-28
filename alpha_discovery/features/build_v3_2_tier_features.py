"""
CNN-Mamba v3.2 Tier 2 + Tier 3 Feature Builder

Authorized by HC #294 + #295 (DIRECTIVES.md) + user's verbatim in-channel grant
2026-05-11 ~19:42 ET: "Yes u have full permission to go through with it all..
ensure u engineer all 3 tiers PROEPRYLY..."

Reads raw MBO dbn.zst files (Databento format), produces:
  - Tier 2 parquet: per 100ms bucket × 14 order-flow features (per day partition)
  - Tier 3 parquet: per 1Hz snapshot × 24 session-context features (per day partition)
  - Daily ts_event_ns alignment file: maps smart_v3 event index → timestamp
    (existing npz files already have this in 'timestamps' key — we just verify)

Stores RAW values. Normalization (per-fold z-score) is applied in the dataloader.

Per-day script signature:
    python build_v3_2_tier_features.py --date YYYY-MM-DD
    python build_v3_2_tier_features.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

V1 scope (HC #295 compliance):
  - Tier 2: 14 features (subset of design doc's 18 — dropped 4 that need L2 book reconstruction:
            microprice_change, avg_spread, n_tob_changes, avg_top5_depth, large_order_count.
            Replaced with: trade_volume_log_z, n_order_events_log_z, avg_order_size_log_z.
            Net: 14 features instead of 18. Documented deviation from design §2.4.
  - Tier 3: 24 features as designed.
  - Storage: per-day parquet partitions, RAW values; per-fold stats computed separately.

Coverage: all RTH-relevant timestamps (4pm ET prior day prep → 4:15 ET next day) to support
training windows that cross session boundaries.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import databento as db
except ImportError:
    print("ERROR: databento library required. pip install databento", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# Constants
# ==============================================================================

TICK_SIZE = 0.25
TICK_VALUE_USD = 12.50
ET_TZ = "America/New_York"

# RTH bounds in ET
RTH_OPEN_HOUR_ET = 9
RTH_OPEN_MIN_ET = 30
RTH_CLOSE_HOUR_ET = 16
RTH_CLOSE_MIN_ET = 0
RTH_SECONDS_PER_DAY = (RTH_CLOSE_HOUR_ET - RTH_OPEN_HOUR_ET) * 3600 + (
    RTH_CLOSE_MIN_ET - RTH_OPEN_MIN_ET
) * 60  # 23,400

# Bucket cadences
TIER2_BUCKET_MS = 100  # 100ms
TIER2_BUCKET_NS = TIER2_BUCKET_MS * 1_000_000
TIER3_SNAP_NS = 1_000_000_000  # 1 second

# Lookbacks
HISTORY_DAYS_FOR_ZSCORE = 20  # rolling 20-day baseline for log-z features
HISTORY_DAYS_FOR_5D_EXTREME = 5
HISTORY_DAYS_FOR_LARGE_ORDER_PCTILE = 5  # not used in v1 (no large_order_count)

# Output paths
DATA_ROOT = Path("/home/jupiter/Lvl3Quant/data")
RAW_MBO_DIR = DATA_ROOT / "raw" / "mbo"
TIER2_OUT_ROOT = DATA_ROOT / "derived" / "tier2_orderflow_features_v1.parquet"
TIER3_OUT_ROOT = DATA_ROOT / "derived" / "tier3_session_features_v1.parquet"
PRIOR_SESSION_STATE_DIR = DATA_ROOT / "derived" / "v3_2_prior_session_state"
ROLLING_STATS_DIR = DATA_ROOT / "derived" / "v3_2_rolling_stats"
LOG_DIR = Path("/home/jupiter/Lvl3Quant/logs")

# Feature column names
TIER2_FEATURE_COLS = [
    "log_return_in_bucket_bps",       # (close - open) / open * 10000
    "bucket_mfe_ticks",                # high - open
    "bucket_mae_ticks",                # open - low
    "n_trades",                        # count of action=='T' (raw, will be log-z'd in loader)
    "trade_volume",                    # sum size where action=='T'
    "aggressor_buy_ratio",             # buyer-initiated vol / total vol
    "signed_volume",                   # buy_vol - sell_vol (raw, will be log-z + sign)
    "n_cancels",                       # count action=='C'
    "n_adds",                          # count action=='A'
    "cancel_add_ratio",                # n_cancels / (n_cancels + n_adds + 1)
    "n_order_events",                  # total non-trade events as activity proxy
    "avg_order_size",                  # mean size of add events
    "bucket_range_ticks",              # (high - low) / 0.25
    "seconds_since_rth_open",          # int seconds (0-23399 in RTH, else negative or > 23400)
]

TIER3_FEATURE_COLS = [
    # Price location / S-R (5)
    "dist_intraday_high_ticks",
    "dist_intraday_low_ticks",
    "dist_session_vwap_ticks",
    "dist_prior_session_close_ticks",
    "dist_prior_session_vwap_ticks",
    # Volume profile (5)
    "dist_intraday_vpoc_ticks",
    "dist_intraday_vah_ticks",
    "dist_intraday_val_ticks",
    "position_in_value_area",          # -1, 0, +1
    "volume_at_current_price_pctile",  # [0, 1]
    # Prior session levels (4)
    "dist_prior_session_high_ticks",
    "dist_prior_session_low_ticks",
    "dist_prior_session_vpoc_ticks",
    "dist_5d_extreme_ticks",
    # Path memory (5)
    "log_return_60s_bps",
    "log_return_5min_bps",
    "log_return_15min_bps",
    "realized_vol_5min_ticks",
    "trend_strength_5min",
    # Regime / time (5)
    "tod_sin",
    "tod_cos",
    "is_lunch_lull",
    "is_close_hour",
    "dow_sin",
    "dow_cos",
]


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(date_tag: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"build_v3_2_features_{date_tag}.log"
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("v3_2_features")


logger: Optional[logging.Logger] = None


# ==============================================================================
# Helpers
# ==============================================================================

def date_str_to_path(d: date) -> Path:
    """date(2026,4,27) -> /...glbx-mdp3-20260427.mbo.dbn.zst"""
    return RAW_MBO_DIR / f"glbx-mdp3-{d.strftime('%Y%m%d')}.mbo.dbn.zst"


def date_iter(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def rth_open_ns_for_date(d: date) -> int:
    """RTH open (9:30 ET) for date d, as ns UTC."""
    et_open = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                           hour=RTH_OPEN_HOUR_ET, minute=RTH_OPEN_MIN_ET,
                           tz=ET_TZ)
    return int(et_open.tz_convert("UTC").value)


def rth_close_ns_for_date(d: date) -> int:
    et_close = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                            hour=RTH_CLOSE_HOUR_ET, minute=RTH_CLOSE_MIN_ET,
                            tz=ET_TZ)
    return int(et_close.tz_convert("UTC").value)


def load_mbo_day(d: date) -> Optional[pd.DataFrame]:
    """Load one day of MBO data via databento.
    Returns df with cols: ts_event(int64 ns UTC), action(str), side(str),
    price(float), size(int), and adds 'rth_session_date' for grouping.
    """
    path = date_str_to_path(d)
    if not path.exists():
        logger.warning(f"Missing raw MBO file: {path.name}")
        return None
    try:
        store = db.DBNStore.from_file(str(path))
        df = store.to_df()
    except Exception as e:
        logger.error(f"Failed to load {path.name}: {e}")
        return None

    if len(df) == 0:
        return None

    # databento gives ts_recv as index, ts_event as a column. We use ts_event.
    df = df.reset_index(drop=True)

    # Filter to OUTRIGHT contracts only — exclude calendar spreads like 'ESM6-ESU6'
    # (spreads have wildly different prices and pollute volume profile / VWAP / H-L).
    # Outright symbols match e.g. 'ESZ6', 'ESM6' — single 'ES' + month-code letter + 1-2 year digits.
    df = df[~df["symbol"].astype(str).str.contains("-", regex=False)].copy()

    # Find dominant front-month symbol (most events) and keep only that to avoid mixing
    # roll-over days where two outrights both trade. Pick whichever has most events.
    sym_counts = df["symbol"].value_counts()
    if len(sym_counts) == 0:
        return None
    front = sym_counts.idxmax()
    df = df[df["symbol"] == front].copy()

    # ts_event arrives as pandas datetime; convert to int64 ns UTC
    if pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
        df["ts_event_ns"] = df["ts_event"].astype("int64")
    else:
        df["ts_event_ns"] = df["ts_event"].astype("int64")

    # Normalize action codes to strings if they came back as bytes/enum
    if df["action"].dtype == object:
        df["action"] = df["action"].astype(str)
    if df["side"].dtype == object:
        df["side"] = df["side"].astype(str)

    return df[["ts_event_ns", "action", "side", "price", "size"]]


# ==============================================================================
# Prior-session state (computed at end of each day, used by next day)
# ==============================================================================

def compute_prior_session_state(df_rth: pd.DataFrame, d: date) -> Dict:
    """Compute the closed-state summary of an RTH session for use by next session.
    df_rth must already be filtered to RTH events.
    Returns dict with: high, low, close, vwap, vpoc.
    """
    if df_rth is None or len(df_rth) == 0:
        return {"date": d.isoformat(), "high": np.nan, "low": np.nan,
                "close": np.nan, "vwap": np.nan, "vpoc": np.nan}

    trades = df_rth[df_rth["action"] == "T"]
    if len(trades) == 0:
        return {"date": d.isoformat(), "high": np.nan, "low": np.nan,
                "close": np.nan, "vwap": np.nan, "vpoc": np.nan}

    # Use trades-only for H/L/C/VWAP/VPOC (non-trade events may have invalid prices)
    vw = (trades["price"] * trades["size"]).sum() / trades["size"].sum()
    bin_volume = trades.groupby(np.round(trades["price"] / TICK_SIZE).astype(int))["size"].sum()
    vpoc_bin = bin_volume.idxmax()
    vpoc_price = vpoc_bin * TICK_SIZE

    return {
        "date": d.isoformat(),
        "high": float(trades["price"].max()),
        "low": float(trades["price"].min()),
        "close": float(trades["price"].iloc[-1]),
        "vwap": float(vw),
        "vpoc": float(vpoc_price),
    }


def save_prior_session_state(state: Dict, d: date):
    PRIOR_SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = PRIOR_SESSION_STATE_DIR / f"{d.strftime('%Y%m%d')}.json"
    with out.open("w") as f:
        json.dump(state, f)


def load_prior_session_state(d: date) -> Optional[Dict]:
    p = PRIOR_SESSION_STATE_DIR / f"{d.strftime('%Y%m%d')}.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_prior_session_state_chain(d: date, max_lookback_days: int = 14) -> List[Dict]:
    """Load up to N prior-session states going backward from d (exclusive of d)."""
    out = []
    cur = d - timedelta(days=1)
    tries = 0
    while len(out) < max_lookback_days and tries < max_lookback_days + 10:
        s = load_prior_session_state(cur)
        if s is not None and not np.isnan(s["high"]):
            out.append(s)
        cur -= timedelta(days=1)
        tries += 1
    return out


# ==============================================================================
# Tier 2: per 100ms bucket order-flow stats
# ==============================================================================

def build_tier2_for_day(df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Group MBO events into 100ms buckets and compute 14 stats per bucket.

    Returns DataFrame with one row per bucket that has events. Sparse representation
    (no row for buckets with no events). Dataloader pads with zeros where missing.
    """
    rth_open = rth_open_ns_for_date(d)

    # Bucket index = ns since RTH open // 100ms. Negative for pre-RTH events.
    df = df.copy()
    df["bucket_idx"] = ((df["ts_event_ns"] - rth_open) // TIER2_BUCKET_NS).astype(np.int64)
    df["seconds_since_rth_open"] = ((df["ts_event_ns"] - rth_open) / 1e9).astype(np.float32)

    # Trade subset
    df["is_trade"] = (df["action"] == "T").astype(np.int8)
    df["is_cancel"] = (df["action"] == "C").astype(np.int8)
    df["is_add"] = (df["action"] == "A").astype(np.int8)
    df["is_order_event"] = (df["action"] != "T").astype(np.int8)

    # Aggressor side for trades: Databento MBO trade events have side = AGGRESSOR side
    # 'B' = buy aggressor, 'A' = sell aggressor.
    # We compute buy_volume and sell_volume per bucket.
    df["trade_size"] = df["size"].where(df["is_trade"] == 1, 0).astype(np.float32)
    df["buy_volume"] = df["trade_size"].where(df["side"] == "B", 0)
    df["sell_volume"] = df["trade_size"].where(df["side"] == "A", 0)
    df["add_size"] = df["size"].where(df["is_add"] == 1, 0).astype(np.float32)

    # CRITICAL: price stats (open/high/low/close, MFE/MAE, range, log_return) must use
    # TRADE prices only. Add/cancel/modify events carry order-book limit prices that can
    # be deep in the book (10-40+ ticks away from mid) — including them in price stats
    # pollutes range/MFE/MAE wildly. Use NaN-masked trade prices and aggregate.
    df["trade_price"] = df["price"].where(df["is_trade"] == 1, np.nan).astype(np.float64)

    # First / last / max / min trade price per bucket (NaN-aware)
    grouped = df.groupby("bucket_idx", sort=True)

    agg = grouped.agg(
        open_price=("trade_price", "first"),     # first NON-NaN trade price; NaN if no trade
        close_price=("trade_price", "last"),     # last NON-NaN trade price
        high_price=("trade_price", "max"),       # NaN if no trade
        low_price=("trade_price", "min"),
        n_trades=("is_trade", "sum"),
        trade_volume=("trade_size", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        n_cancels=("is_cancel", "sum"),
        n_adds=("is_add", "sum"),
        n_order_events=("is_order_event", "sum"),
        add_volume_sum=("add_size", "sum"),
        seconds_since_rth_open=("seconds_since_rth_open", "first"),
    ).reset_index()
    # pandas .first()/.last() default to skipping NaN; .min()/.max() also skip NaN.
    # Buckets with zero trades will have NaN for all four — we'll convert these to
    # "no price movement" features below.

    # Build the 14 features
    out = pd.DataFrame()
    out["bucket_idx"] = agg["bucket_idx"]
    out["seconds_since_rth_open"] = agg["seconds_since_rth_open"].astype(np.float32)

    # Returns / range — TRADES-ONLY (NaN-masked where bucket had no trades)
    open_p = agg["open_price"].astype(np.float64).values
    close_p = agg["close_price"].astype(np.float64).values
    high_p = agg["high_price"].astype(np.float64).values
    low_p = agg["low_price"].astype(np.float64).values

    # Buckets with no trades: open_p/close_p/high_p/low_p are NaN. Set price features to 0
    # (no observed movement). The order-flow features below remain valid for these buckets.
    no_trade_mask = ~(np.isfinite(open_p) & np.isfinite(close_p))

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret_bps = np.where(
            np.isfinite(open_p) & (open_p > 0) & np.isfinite(close_p),
            np.log(close_p / np.maximum(open_p, 1e-9)) * 10000.0,
            0.0,
        )
    log_ret_bps = np.where(no_trade_mask, 0.0, log_ret_bps)
    out["log_return_in_bucket_bps"] = np.clip(log_ret_bps, -200, 200).astype(np.float32)

    mfe = np.where(np.isfinite(high_p) & np.isfinite(open_p), (high_p - open_p) / TICK_SIZE, 0.0)
    mae = np.where(np.isfinite(open_p) & np.isfinite(low_p), (open_p - low_p) / TICK_SIZE, 0.0)
    rng = np.where(np.isfinite(high_p) & np.isfinite(low_p), (high_p - low_p) / TICK_SIZE, 0.0)
    out["bucket_mfe_ticks"] = np.clip(mfe, 0, 40).astype(np.float32)
    out["bucket_mae_ticks"] = np.clip(mae, 0, 40).astype(np.float32)
    out["bucket_range_ticks"] = np.clip(rng, 0, 40).astype(np.float32)

    # Trade flow
    out["n_trades"] = agg["n_trades"].astype(np.float32)
    out["trade_volume"] = agg["trade_volume"].astype(np.float32)
    total_vol = (agg["buy_volume"] + agg["sell_volume"]).values
    out["aggressor_buy_ratio"] = np.where(
        total_vol > 0,
        agg["buy_volume"].values / np.maximum(total_vol, 1.0),
        0.5,
    ).astype(np.float32)
    out["signed_volume"] = (agg["buy_volume"] - agg["sell_volume"]).astype(np.float32)

    # Order events
    out["n_cancels"] = agg["n_cancels"].astype(np.float32)
    out["n_adds"] = agg["n_adds"].astype(np.float32)
    add_cancel = agg["n_adds"] + agg["n_cancels"]
    out["cancel_add_ratio"] = np.where(
        add_cancel > 0,
        agg["n_cancels"].values / np.maximum(add_cancel.values, 1.0),
        0.5,
    ).astype(np.float32)
    out["n_order_events"] = agg["n_order_events"].astype(np.float32)
    out["avg_order_size"] = np.where(
        agg["n_adds"] > 0,
        agg["add_volume_sum"].values / np.maximum(agg["n_adds"].values, 1.0),
        0.0,
    ).astype(np.float32)

    # Note: `date` column intentionally NOT written inside the parquet — partition key
    # `date=YYYY-MM-DD` carries that info. Writing it inside the file creates a schema
    # mismatch when pyarrow tries to merge dictionary-typed partition key with string
    # column in the file.
    cols_final = ["bucket_idx"] + TIER2_FEATURE_COLS
    out = out[cols_final]
    return out


# ==============================================================================
# Tier 3: per 1Hz session-context snapshot
# ==============================================================================

def build_tier3_for_day(df: pd.DataFrame, d: date,
                       prior_states: List[Dict]) -> pd.DataFrame:
    """Build per-second session-context snapshots for RTH of day d.

    Uses MBO trade events to compute intraday running stats (VWAP, VPOC, etc).
    Uses prior_states (most-recent first) for prior-session features.
    """
    rth_open = rth_open_ns_for_date(d)
    rth_close = rth_close_ns_for_date(d)

    # Filter to RTH events
    df_rth = df[(df["ts_event_ns"] >= rth_open) & (df["ts_event_ns"] < rth_close)].copy()
    if len(df_rth) == 0:
        return pd.DataFrame(columns=["second_idx"] + TIER3_FEATURE_COLS)

    # Build per-second timeline
    n_seconds = RTH_SECONDS_PER_DAY  # 23,400
    sec_idx = np.arange(n_seconds, dtype=np.int32)

    # Trades only (for VWAP, VPOC, volume profile)
    trades = df_rth[df_rth["action"] == "T"].copy()
    trades["sec_idx"] = ((trades["ts_event_ns"] - rth_open) // TIER3_SNAP_NS).astype(np.int64)
    trades["sec_idx"] = np.clip(trades["sec_idx"], 0, n_seconds - 1)
    trades["price_x_size"] = trades["price"] * trades["size"]
    trades["price_bin"] = np.round(trades["price"] / TICK_SIZE).astype(np.int64)

    # Last TRADE price per second (use trades only — non-trade events have invalid prices)
    df_rth["sec_idx"] = ((df_rth["ts_event_ns"] - rth_open) // TIER3_SNAP_NS).astype(np.int64)
    df_rth["sec_idx"] = np.clip(df_rth["sec_idx"], 0, n_seconds - 1)

    if len(trades) > 0:
        last_price_per_sec = trades.groupby("sec_idx")["price"].last()
        last_price_arr = np.full(n_seconds, np.nan)
        last_price_arr[last_price_per_sec.index.values.astype(int)] = last_price_per_sec.values
        last_price_arr = pd.Series(last_price_arr).ffill().bfill().values
    else:
        last_price_arr = np.full(n_seconds, 1.0)  # placeholder; no trades in RTH (rare)

    # Running cumulative buy/sell volume per second from trades
    trade_agg = trades.groupby("sec_idx").agg(
        vol_x_price=("price_x_size", "sum"),
        vol=("size", "sum"),
        high=("price", "max"),
        low=("price", "min"),
    )

    cum_vol = np.zeros(n_seconds, dtype=np.float64)
    cum_vol_x_price = np.zeros(n_seconds, dtype=np.float64)
    # Intraday high/low: running max/min of trade prices
    intraday_high = np.full(n_seconds, np.nan)
    intraday_low = np.full(n_seconds, np.nan)

    trade_idx = trade_agg.index.values.astype(int)
    vol_at_idx = np.zeros(n_seconds, dtype=np.float64)
    vxp_at_idx = np.zeros(n_seconds, dtype=np.float64)
    vol_at_idx[trade_idx] = trade_agg["vol"].values
    vxp_at_idx[trade_idx] = trade_agg["vol_x_price"].values

    cum_vol = np.cumsum(vol_at_idx)
    cum_vol_x_price = np.cumsum(vxp_at_idx)
    running_vwap = np.where(cum_vol > 0, cum_vol_x_price / np.maximum(cum_vol, 1e-9), last_price_arr)

    # Intraday running high/low: take cummax/cummin of last-observed price
    intraday_high = np.maximum.accumulate(last_price_arr)
    intraday_low = np.minimum.accumulate(last_price_arr)

    # Volume profile (running): for each second, the price bin with most cumulative volume
    # Compute bin range from TRADE prices only (non-trade events may have price=0 from clears/cancels)
    if len(trades) > 0:
        bin_min = int(np.floor(trades["price"].min() / TICK_SIZE))
        bin_max = int(np.ceil(trades["price"].max() / TICK_SIZE))
    else:
        bin_min = int(np.floor(np.nanmin(last_price_arr) / TICK_SIZE))
        bin_max = int(np.ceil(np.nanmax(last_price_arr) / TICK_SIZE))
    # Safety clip: ES daily range rarely exceeds 400 ticks. If we see something pathological,
    # cap at +/- 1000 ticks around median to prevent memory blowups.
    median_bin = int(np.round(np.nanmedian(last_price_arr) / TICK_SIZE))
    bin_min = max(bin_min, median_bin - 1000)
    bin_max = min(bin_max, median_bin + 1000)
    n_bins = max(bin_max - bin_min + 1, 1)
    # For each trade, increment its bin at its sec_idx, then cumsum across seconds.
    vol_by_bin_per_sec = np.zeros((n_seconds, n_bins), dtype=np.float64)
    if len(trades) > 0:
        rel_bin = (trades["price_bin"].values - bin_min).astype(np.int64)
        # Clip to valid range (some trades may be outside the median ±1000 safety window)
        rel_bin = np.clip(rel_bin, 0, n_bins - 1)
        sec_i = trades["sec_idx"].values.astype(np.int64)
        # Avoid python loop with np.add.at
        np.add.at(vol_by_bin_per_sec, (sec_i, rel_bin), trades["size"].values.astype(np.float64))
    cum_vol_by_bin = np.cumsum(vol_by_bin_per_sec, axis=0)  # [n_sec, n_bins]

    # VPOC per second = argmax bin (in price terms)
    vpoc_bin = cum_vol_by_bin.argmax(axis=1)  # [n_sec]
    vpoc_price = (vpoc_bin + bin_min) * TICK_SIZE

    # Value area: 70% volume around VPOC. We compute it cheaply:
    # for each second, find smallest contiguous range around VPOC that covers ≥70% of total vol.
    total_vol_per_sec = cum_vol_by_bin.sum(axis=1)
    vah_price = np.copy(vpoc_price)
    val_price = np.copy(vpoc_price)
    for s in range(n_seconds):
        if total_vol_per_sec[s] <= 0:
            continue
        bins = cum_vol_by_bin[s]
        target = 0.70 * total_vol_per_sec[s]
        # Expand from VPOC outward
        poc = int(vpoc_bin[s])
        lo = poc
        hi = poc
        acc = bins[poc]
        while acc < target and (lo > 0 or hi < n_bins - 1):
            left_vol = bins[lo - 1] if lo > 0 else -1
            right_vol = bins[hi + 1] if hi < n_bins - 1 else -1
            if left_vol >= right_vol and lo > 0:
                lo -= 1
                acc += bins[lo]
            elif hi < n_bins - 1:
                hi += 1
                acc += bins[hi]
            else:
                break
        vah_price[s] = (hi + bin_min) * TICK_SIZE
        val_price[s] = (lo + bin_min) * TICK_SIZE

    # Volume at current price percentile
    # For each second: rank of cum_vol_by_bin[s, current_price_bin] within cum_vol_by_bin[s, :]
    current_bin = (np.round(last_price_arr / TICK_SIZE).astype(np.int64) - bin_min)
    current_bin = np.clip(current_bin, 0, n_bins - 1)
    vol_at_current = cum_vol_by_bin[np.arange(n_seconds), current_bin]
    max_vol = cum_vol_by_bin.max(axis=1)
    vol_pctile = np.where(max_vol > 0, vol_at_current / np.maximum(max_vol, 1e-9), 0.0)

    # Position in value area
    pos_in_va = np.where(
        last_price_arr > vah_price, 1.0,
        np.where(last_price_arr < val_price, -1.0, 0.0),
    )

    # Prior session features (need at least 1 prior, ideally 5)
    if len(prior_states) > 0:
        ps = prior_states[0]
        prior_close = ps["close"]
        prior_vwap = ps["vwap"]
        prior_high = ps["high"]
        prior_low = ps["low"]
        prior_vpoc = ps["vpoc"]
    else:
        prior_close = np.nan
        prior_vwap = np.nan
        prior_high = np.nan
        prior_low = np.nan
        prior_vpoc = np.nan

    if len(prior_states) >= 5:
        recent5 = prior_states[:5]
        h5 = max(s["high"] for s in recent5)
        l5 = min(s["low"] for s in recent5)
    elif len(prior_states) > 0:
        h5 = max(s["high"] for s in prior_states)
        l5 = min(s["low"] for s in prior_states)
    else:
        h5 = np.nan
        l5 = np.nan

    # Build distance features in ticks
    def dist_ticks(target):
        if not np.isfinite(target):
            return np.zeros(n_seconds, dtype=np.float32)
        return np.clip((last_price_arr - target) / TICK_SIZE, -200, 200).astype(np.float32)

    out = pd.DataFrame()
    out["second_idx"] = sec_idx
    # Note: `date` intentionally NOT written inside parquet — partition key carries it.

    # Tier 3 features
    out["dist_intraday_high_ticks"] = np.clip((last_price_arr - intraday_high) / TICK_SIZE,
                                              -200, 200).astype(np.float32)
    out["dist_intraday_low_ticks"] = np.clip((last_price_arr - intraday_low) / TICK_SIZE,
                                              -200, 200).astype(np.float32)
    out["dist_session_vwap_ticks"] = np.clip((last_price_arr - running_vwap) / TICK_SIZE,
                                              -200, 200).astype(np.float32)
    out["dist_prior_session_close_ticks"] = dist_ticks(prior_close)
    out["dist_prior_session_vwap_ticks"] = dist_ticks(prior_vwap)

    out["dist_intraday_vpoc_ticks"] = np.clip((last_price_arr - vpoc_price) / TICK_SIZE,
                                              -200, 200).astype(np.float32)
    out["dist_intraday_vah_ticks"] = np.clip((last_price_arr - vah_price) / TICK_SIZE,
                                              -200, 200).astype(np.float32)
    out["dist_intraday_val_ticks"] = np.clip((last_price_arr - val_price) / TICK_SIZE,
                                              -200, 200).astype(np.float32)
    out["position_in_value_area"] = pos_in_va.astype(np.float32)
    out["volume_at_current_price_pctile"] = vol_pctile.astype(np.float32)

    out["dist_prior_session_high_ticks"] = dist_ticks(prior_high)
    out["dist_prior_session_low_ticks"] = dist_ticks(prior_low)
    out["dist_prior_session_vpoc_ticks"] = dist_ticks(prior_vpoc)
    # dist_5d_extreme: closer of {5d-high, 5d-low}
    if np.isfinite(h5) and np.isfinite(l5):
        d_h5 = np.abs(last_price_arr - h5)
        d_l5 = np.abs(last_price_arr - l5)
        nearer = np.where(d_h5 < d_l5, h5, l5)
        out["dist_5d_extreme_ticks"] = np.clip((last_price_arr - nearer) / TICK_SIZE,
                                               -200, 200).astype(np.float32)
    else:
        out["dist_5d_extreme_ticks"] = np.zeros(n_seconds, dtype=np.float32)

    # Path memory: log returns at 60s, 5min, 15min
    def lagged_log_return_bps(lag_seconds: int) -> np.ndarray:
        out = np.zeros(n_seconds, dtype=np.float64)
        if lag_seconds < n_seconds:
            ratio = last_price_arr[lag_seconds:] / np.maximum(last_price_arr[:-lag_seconds], 1e-9)
            out[lag_seconds:] = np.log(np.maximum(ratio, 1e-9)) * 10000.0
        return np.clip(out, -1000, 1000).astype(np.float32)

    out["log_return_60s_bps"] = lagged_log_return_bps(60)
    out["log_return_5min_bps"] = lagged_log_return_bps(300)
    out["log_return_15min_bps"] = lagged_log_return_bps(900)

    # Realized vol over last 5min in ticks
    s1_returns = np.zeros(n_seconds, dtype=np.float64)
    s1_returns[1:] = np.log(np.maximum(last_price_arr[1:] / np.maximum(last_price_arr[:-1], 1e-9), 1e-9))
    # rolling std over 300s window. Use pandas for simplicity.
    rv_5min = pd.Series(s1_returns).rolling(300, min_periods=10).std().fillna(0).values
    # convert log-returns std to "tick std" by multiplying by avg price / TICK_SIZE
    rv_5min_ticks = rv_5min * last_price_arr / TICK_SIZE
    out["realized_vol_5min_ticks"] = np.clip(rv_5min_ticks, 0, 50).astype(np.float32)

    # Trend strength: 5min directional move normalized by 5min realized vol → t-statistic.
    # Distinct from log_return_5min_bps (which is just the raw return).
    # = (P_T - P_{T-300}) / (realized_vol_5min_pts + eps), where vol is in price points.
    trend_raw = np.zeros(n_seconds, dtype=np.float64)
    if 300 < n_seconds:
        trend_raw[300:] = last_price_arr[300:] - last_price_arr[:-300]
    # rv_5min_ticks is already realized vol in ticks. Convert back to points (*TICK_SIZE).
    vol_pts = rv_5min_ticks * TICK_SIZE
    trend_t = trend_raw / np.maximum(vol_pts, 1e-3)  # 1e-3 pts = 0.004 ticks floor
    out["trend_strength_5min"] = np.clip(trend_t, -10.0, 10.0).astype(np.float32)

    # Regime / time
    seconds_in_rth = sec_idx.astype(np.float64)
    out["tod_sin"] = np.sin(2 * np.pi * seconds_in_rth / RTH_SECONDS_PER_DAY).astype(np.float32)
    out["tod_cos"] = np.cos(2 * np.pi * seconds_in_rth / RTH_SECONDS_PER_DAY).astype(np.float32)
    # lunch_lull: 11:30-13:30 ET. 11:30 is 2h after open = 7200s. 13:30 = 4h after open = 14400s.
    out["is_lunch_lull"] = ((seconds_in_rth >= 7200) & (seconds_in_rth < 14400)).astype(np.float32)
    # close_hour: last 30min = >= 21600s
    out["is_close_hour"] = (seconds_in_rth >= 21600).astype(np.float32)
    # day of week (Mon=0 .. Fri=4). Full sinusoidal encoding requires BOTH sin AND cos
    # so model can disambiguate each weekday uniquely (sin alone is ambiguous: sin(2π·1/5)=sin(2π·4/5)).
    dow = d.weekday()  # 0-6
    out["dow_sin"] = np.full(n_seconds, np.sin(2 * np.pi * dow / 5.0), dtype=np.float32)
    out["dow_cos"] = np.full(n_seconds, np.cos(2 * np.pi * dow / 5.0), dtype=np.float32)

    # Reorder cols (no `date` inside — partition key carries it)
    return out[["second_idx"] + TIER3_FEATURE_COLS]


# ==============================================================================
# Per-day processing
# ==============================================================================

def process_day(d: date, force: bool = False) -> bool:
    """Build Tier 2 + Tier 3 + prior-session state for one day. Returns True if successful."""
    t2_path = TIER2_OUT_ROOT / f"date={d.isoformat()}" / "part-0.parquet"
    t3_path = TIER3_OUT_ROOT / f"date={d.isoformat()}" / "part-0.parquet"
    ps_path = PRIOR_SESSION_STATE_DIR / f"{d.strftime('%Y%m%d')}.json"

    if not force and t2_path.exists() and t3_path.exists() and ps_path.exists():
        logger.info(f"[{d}] skipped (all artifacts exist)")
        return True

    logger.info(f"[{d}] loading MBO...")
    df = load_mbo_day(d)
    if df is None:
        return False
    logger.info(f"[{d}] loaded {len(df):,} events")

    # Build Tier 2 (across full day's events, sparse)
    logger.info(f"[{d}] building Tier 2...")
    t2 = build_tier2_for_day(df, d)
    t2_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t2, preserve_index=False), t2_path, compression="zstd")
    logger.info(f"[{d}] Tier 2 wrote {len(t2):,} buckets")

    # Build Tier 3 (RTH only)
    logger.info(f"[{d}] loading prior session chain...")
    prior_chain = load_prior_session_state_chain(d, max_lookback_days=14)
    logger.info(f"[{d}] {len(prior_chain)} prior sessions available")

    logger.info(f"[{d}] building Tier 3...")
    t3 = build_tier3_for_day(df, d, prior_chain)
    t3_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t3, preserve_index=False), t3_path, compression="zstd")
    logger.info(f"[{d}] Tier 3 wrote {len(t3):,} snapshots")

    # Compute and save prior-session state for next day
    rth_open = rth_open_ns_for_date(d)
    rth_close = rth_close_ns_for_date(d)
    df_rth = df[(df["ts_event_ns"] >= rth_open) & (df["ts_event_ns"] < rth_close)]
    state = compute_prior_session_state(df_rth, d)
    save_prior_session_state(state, d)
    logger.info(f"[{d}] saved prior-session state: VWAP={state['vwap']:.2f} H={state['high']:.2f} L={state['low']:.2f}")

    return True


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, help="single date YYYY-MM-DD")
    p.add_argument("--start-date", type=str, help="start date YYYY-MM-DD")
    p.add_argument("--end-date", type=str, help="end date YYYY-MM-DD (inclusive)")
    p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    return p.parse_args()


def main():
    args = parse_args()
    global logger
    tag = args.date or f"{args.start_date}_{args.end_date}"
    logger = setup_logging(tag.replace("-", ""))

    logger.info("=" * 70)
    logger.info("CNN-Mamba v3.2 Tier 2 + Tier 3 Feature Builder")
    logger.info(f"Output Tier 2: {TIER2_OUT_ROOT}")
    logger.info(f"Output Tier 3: {TIER3_OUT_ROOT}")
    logger.info(f"Prior-session state: {PRIOR_SESSION_STATE_DIR}")
    logger.info("=" * 70)

    if args.date:
        dates = [date.fromisoformat(args.date)]
    elif args.start_date and args.end_date:
        s = date.fromisoformat(args.start_date)
        e = date.fromisoformat(args.end_date)
        dates = list(date_iter(s, e))
    else:
        logger.error("Specify --date OR both --start-date and --end-date")
        sys.exit(2)

    success = 0
    failed = 0
    for d in dates:
        try:
            ok = process_day(d, force=args.force)
            if ok:
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            logger.warning("Interrupted")
            break
        except Exception as e:
            logger.exception(f"[{d}] FAILED: {e}")
            failed += 1

    logger.info("=" * 70)
    logger.info(f"DONE: {success} succeeded, {failed} failed, {len(dates)} total")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
