"""
CNN-Mamba v3.2.1 Tier 2 + Tier 3 Feature Builder — Priority-1 deliverable

Authorized by HC #295A / #296 / #297A / #299 / #300A (DIRECTIVES.md).
Author: Autonomous Claude (Opus 4), 2026-05-12, ~14h compressed work.

Builds on build_v3_2_tier_features.py with five Priority-1 changes:

  (1) Restore 5 missing T2 features via L2 book reconstruction (HC #295C):
        microprice_change_ticks, avg_spread_in_bucket_ticks,
        n_top_of_book_changes,  avg_top5_depth_volume, large_order_count.
      T2 grows 14 → 19 features.

  (2) Apply 19 HC #298 builder-side normalization fixes:
        - log1p for heavy-tail counts/volumes (n_trades, n_cancels, n_adds,
          n_order_events, trade_volume, avg_top5_depth_volume, large_order_count)
        - sign-preserving log-z for signed_volume
        - seconds_since_rth_open normalized to [0, 1] (/23400)
        - T3 distance features stored as clip([-200,200]) / 100  (NOT z-scored later)
        - cyclical / binary / bounded features passthrough (handled in dataloader)

  (3) Add LGBM vol predictions to T3 (Priority-1 item #3):
        - lgbm_vol_pred_5min  (closest available: LGBM 30s horizon)
        - lgbm_vol_pred_30min (closest available: LGBM 60s horizon)
      LGBM v3 trained horizons are [10s, 30s, 60s]; we use 30s/60s as proxies for
      5min/30min until a longer-horizon LGBM is trained. Documented in column doc.

  (4) Embed event-type as 8-dim learnable in T1 → trainer-side (NOT here).

  (5) Add session-phase one-hot to T3:
        phase_open      [0,    3600)
        phase_morning   [3600, 9000)
        phase_lunch     [9000, 14400)
        phase_afternoon [14400, +inf)
      Sum is exactly 1 per snapshot. T3 grows 25 → 31 (= 25 + 2 LGBM + 4 phase).
      Existing is_lunch_lull / is_close_hour preserved for backward compat.

Output paths (NEW — does NOT clobber v3.2):
  Tier 2: /home/jupiter/Lvl3Quant/data/processed/tier2_orderflow_v3_2_1/
  Tier 3: /home/jupiter/Lvl3Quant/data/processed/tier3_session_v3_2_1/

Per-day script signature (unchanged):
    python build_v3_2_1_tier_features.py --date YYYY-MM-DD
    python build_v3_2_1_tier_features.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

Coverage: all RTH-relevant timestamps (4pm ET prior day → 4:15 ET next day).
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
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
HISTORY_DAYS_FOR_ZSCORE = 20
HISTORY_DAYS_FOR_5D_EXTREME = 5
HISTORY_DAYS_FOR_LARGE_ORDER_PCTILE = 5  # rolling 5d 95th pctile of add sizes
LARGE_ORDER_PCTILE_Q = 0.95

# Top-of-book / depth tracking
TOP_N_DEPTH = 5  # avg top-5 depth volume per side

# Output paths (NEW — v3.2.1)
DATA_ROOT = Path("/home/jupiter/Lvl3Quant/data")
RAW_MBO_DIR = DATA_ROOT / "raw" / "mbo"
TIER2_OUT_ROOT = DATA_ROOT / "processed" / "tier2_orderflow_v3_2_1"
TIER3_OUT_ROOT = DATA_ROOT / "processed" / "tier3_session_v3_2_1"
PRIOR_SESSION_STATE_DIR = DATA_ROOT / "derived" / "v3_2_prior_session_state"  # SHARED w/ v3.2
ROLLING_STATS_DIR = DATA_ROOT / "derived" / "v3_2_1_rolling_stats"
LARGE_ORDER_PCTILE_DIR = DATA_ROOT / "derived" / "v3_2_1_large_order_pctile"
LOG_DIR = Path("/home/jupiter/Lvl3Quant/logs")

# LGBM vol model location
LGBM_VOL_DIR = Path("/home/jupiter/Lvl3Quant/output/vol_lgbm_v3")
LGBM_VOL_PREFIX = "vol_v3_"
LGBM_VOL_SUFFIX = "_models.pkl"
# Smart_v3 mbo_events (needed by LGBM feature builder)
SMART_V3_NPZ_ROOT = DATA_ROOT / "processed" / "mbo_events_smart_v3"
# LGBM v3 feature builder constants (mirror train_vol_lgbm_v3.py)
LGBM_WINDOW = 1000
LGBM_N_FEATURES = 28
LGBM_HORIZONS_S = [10.0, 30.0, 60.0]
LGBM_COL_TIME = 0
LGBM_COL_TYPE = 1
LGBM_COL_SIDE = 2
LGBM_COL_PRICE = 3
LGBM_COL_QTY = 4
LGBM_COL_SPREAD = 5
LGBM_TYPE_TRADE = 3
# Which LGBM horizon index corresponds to our T3 columns:
#   lgbm_vol_pred_5min  → 30s horizon (index 1) — closest available proxy
#   lgbm_vol_pred_30min → 60s horizon (index 2) — closest available proxy
LGBM_HORIZON_IDX_FOR_5MIN = 1
LGBM_HORIZON_IDX_FOR_30MIN = 2

# ------------------------------------------------------------------------------
# Feature column names (v3.2.1)
# ------------------------------------------------------------------------------
# TIER 2 — 19 features (was 14 in v3.2; +5 new for L2 book reconstruction).
# Each column lives in ONE of three normalization regimes (HC #298):
#   LOG1P  : log1p(x) written here; dataloader z-scores log-domain value
#   SIGNED : sign(x) * log1p(|x|) written here; dataloader z-scores
#   BOUNDED: stored already-bounded, dataloader can z-score normally (it's safe)
#   PASS   : stored already-normalized to [0,1] or similar; dataloader passthrough
TIER2_FEATURE_COLS = [
    "log_return_in_bucket_bps",        # bps, clip[-200,200], BOUNDED → trainer z-score
    "bucket_mfe_ticks",                # ticks, clip[0,40] (kept from v3.2 for compat), BOUNDED
    "bucket_mae_ticks",                # ticks, clip[0,40], BOUNDED
    "n_trades",                        # LOG1P (HC #298 fix #1)
    "trade_volume",                    # LOG1P (HC #298 fix #2)
    "aggressor_buy_ratio",             # [0,1], NaN→0.5, BOUNDED → trainer z-score
    "signed_volume",                   # SIGNED log-z: sign(x)*log1p(|x|)  (HC #298 fix #5)
    "n_cancels",                       # LOG1P (HC #298 fix #3)
    "n_adds",                          # LOG1P (HC #298 fix #4)
    "cancel_add_ratio",                # [0,1], NaN→0.5, BOUNDED
    "n_order_events",                  # LOG1P
    "avg_order_size",                  # ticks/contracts (mean of add events); not log'd (already small)
    "bucket_range_ticks",              # ticks, clip[0,40], BOUNDED
    "seconds_since_rth_open",          # PASS — already in [0,1] (= raw_secs / 23400)  (HC #298)
    # ---- NEW v3.2.1 (5 features, Priority-1 item #2) ----
    "microprice_change_ticks",         # ticks (signed): bucket-end micro - bucket-start micro
    "avg_spread_in_bucket_ticks",      # ticks, clip[0,20], BOUNDED
    "n_top_of_book_changes",           # LOG1P count of events that flipped best_bid OR best_ask
    "avg_top5_depth_volume",           # LOG1P (avg top-5 depth volume per side)
    "large_order_count",               # LOG1P count of orders > trailing-5d 95th pctile size
]
assert len(TIER2_FEATURE_COLS) == 19, f"T2 must be 19 features, got {len(TIER2_FEATURE_COLS)}"

# TIER 3 — 31 features (was 25 in v3.2; +2 LGBM + 4 session-phase; existing 25 kept).
# All distance features (14) stored as clip([-200,200]) / 100.0  per HC #298 fix.
TIER3_FEATURE_COLS = [
    # Price location / S-R (5) — DISTANCE (clip[-200,200] / 100, PASS in dataloader)
    "dist_intraday_high_ticks",
    "dist_intraday_low_ticks",
    "dist_session_vwap_ticks",
    "dist_prior_session_close_ticks",
    "dist_prior_session_vwap_ticks",
    # Volume profile (5) — 3 DISTANCE + 2 BOUNDED
    "dist_intraday_vpoc_ticks",
    "dist_intraday_vah_ticks",
    "dist_intraday_val_ticks",
    "position_in_value_area",          # categorical {-1, 0, +1} — PASS (no z-score)
    "volume_at_current_price_pctile",  # [0, 1] — PASS
    # Prior session levels (4) — all DISTANCE
    "dist_prior_session_high_ticks",
    "dist_prior_session_low_ticks",
    "dist_prior_session_vpoc_ticks",
    "dist_5d_extreme_ticks",
    # Path memory (5) — BOUNDED returns + bounded vol/t-stat
    "log_return_60s_bps",              # clip[-1000,1000] bps
    "log_return_5min_bps",             # clip[-1000,1000] bps
    "log_return_15min_bps",            # clip[-1000,1000] bps
    "realized_vol_5min_ticks",         # clip[0,50]
    "trend_strength_5min",             # clip[-10,10] t-stat
    # Regime / time (5) — 2 cyclical (PASS) + 2 binary (PASS) + 1 cyclical (PASS)
    "tod_sin",                         # [-1,+1] raw — PASS (HC #298 fix)
    "tod_cos",                         # [-1,+1] raw — PASS
    "is_lunch_lull",                   # {0,1} — PASS
    "is_close_hour",                   # {0,1} — PASS
    "dow_sin",                         # [-1,+1] raw — PASS
    "dow_cos",                         # [-1,+1] raw — PASS (HC #298 fix: was missing/zscored)
    # ---- NEW v3.2.1: LGBM vol predictions (2 features, Priority-1 item #3) ----
    "lgbm_vol_pred_5min",              # LGBM 30s-horizon vol prediction (proxy for 5min)
    "lgbm_vol_pred_30min",             # LGBM 60s-horizon vol prediction (proxy for 30min)
    # ---- NEW v3.2.1: session-phase one-hot (4 features, Priority-1 item #5) ----
    "phase_open",                      # binary {0,1}, 0 ≤ sec < 3600 (9:30-10:30)
    "phase_morning",                   # binary, 3600 ≤ sec < 9000 (10:30-12:00)
    "phase_lunch",                     # binary, 9000 ≤ sec < 14400 (12:00-13:30)
    "phase_afternoon",                 # binary, 14400 ≤ sec (13:30-close)
]
assert len(TIER3_FEATURE_COLS) == 31, f"T3 must be 31 features, got {len(TIER3_FEATURE_COLS)}"

# Indexed lists for the trainer dataloader to know which columns to passthrough.
TIER3_DISTANCE_COLS = [
    "dist_intraday_high_ticks", "dist_intraday_low_ticks", "dist_session_vwap_ticks",
    "dist_prior_session_close_ticks", "dist_prior_session_vwap_ticks",
    "dist_intraday_vpoc_ticks", "dist_intraday_vah_ticks", "dist_intraday_val_ticks",
    "dist_prior_session_high_ticks", "dist_prior_session_low_ticks",
    "dist_prior_session_vpoc_ticks", "dist_5d_extreme_ticks",
]  # 12 distances written as /100  (note: there are 12 dist_* + 2 inside volume profile = 14
   #  "tick distance" features; we list the 12 prefixed with dist_*; vah/val/vpoc are in here)
# (Above list intentionally enumerated by name. Trainer uses pattern dist_* match.)


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(date_tag: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"build_v3_2_1_features_{date_tag}.log"
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
    return logging.getLogger("v3_2_1_features")


logger: Optional[logging.Logger] = None


# ==============================================================================
# Helpers (mirror v3.2)
# ==============================================================================

def date_str_to_path(d: date) -> Path:
    return RAW_MBO_DIR / f"glbx-mdp3-{d.strftime('%Y%m%d')}.mbo.dbn.zst"


def date_iter(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def rth_open_ns_for_date(d: date) -> int:
    et_open = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                           hour=RTH_OPEN_HOUR_ET, minute=RTH_OPEN_MIN_ET, tz=ET_TZ)
    return int(et_open.tz_convert("UTC").value)


def rth_close_ns_for_date(d: date) -> int:
    et_close = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                            hour=RTH_CLOSE_HOUR_ET, minute=RTH_CLOSE_MIN_ET, tz=ET_TZ)
    return int(et_close.tz_convert("UTC").value)


def load_mbo_day(d: date) -> Optional[pd.DataFrame]:
    """Load one day of MBO data via databento. Same logic as v3.2."""
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
    df = df.reset_index(drop=True)
    # Outright only (exclude calendar spreads)
    df = df[~df["symbol"].astype(str).str.contains("-", regex=False)].copy()
    sym_counts = df["symbol"].value_counts()
    if len(sym_counts) == 0:
        return None
    front = sym_counts.idxmax()
    df = df[df["symbol"] == front].copy()
    if pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
        df["ts_event_ns"] = df["ts_event"].astype("int64")
    else:
        df["ts_event_ns"] = df["ts_event"].astype("int64")
    if df["action"].dtype == object:
        df["action"] = df["action"].astype(str)
    if df["side"].dtype == object:
        df["side"] = df["side"].astype(str)
    return df[["ts_event_ns", "action", "side", "price", "size"]]


# ==============================================================================
# Prior-session state (shared with v3.2 — read-only here)
# ==============================================================================

def compute_prior_session_state(df_rth: pd.DataFrame, d: date) -> Dict:
    if df_rth is None or len(df_rth) == 0:
        return {"date": d.isoformat(), "high": np.nan, "low": np.nan,
                "close": np.nan, "vwap": np.nan, "vpoc": np.nan}
    trades = df_rth[df_rth["action"] == "T"]
    if len(trades) == 0:
        return {"date": d.isoformat(), "high": np.nan, "low": np.nan,
                "close": np.nan, "vwap": np.nan, "vpoc": np.nan}
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
# L2 BOOK RECONSTRUCTION (NEW in v3.2.1 — needed for 5 missing T2 features)
# ==============================================================================
#
# We maintain a per-side dict {price_bin -> resting_qty} updated on each event:
#   action 'A' (add)    : book[side][price_bin] += size
#   action 'C' (cancel) : book[side][price_bin] -= size  (clamp >= 0; drop if 0)
#   action 'M' (modify) : treat as cancel then add (Databento may already split)
#   action 'T' (trade)  : trade consumes from the resting side at best price.
#                          In Databento MBO 'T' events the `side` is the AGGRESSOR
#                          side, so the resting side that's consumed is the OPPOSITE.
#                          If side='B' (buy aggressor) → consume from ask side.
#                          If side='A' (sell aggressor) → consume from bid side.
#   action 'F' (fill)   : same as T for our purposes
#   action 'R' (clear)  : reset side's book.
#
# We track best_bid_bin = max(bid_bins) and best_ask_bin = min(ask_bins).
# A "TOB change" is when best_bid_bin or best_ask_bin moves between successive events.
# microprice = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
# avg_spread = mean over the bucket of (best_ask - best_bid) in ticks
# avg_top5_depth = mean over the bucket of (sum of top-5 levels of bid or ask volume)
# ------------------------------------------------------------------------------

class _BookState:
    """Tiny L2 book maintainer. Uses Python dicts (price_bin → qty).
    Not the fastest possible — but on 60 days of daily MBO this completes overnight on Jupiter CPU.
    """
    __slots__ = ("bids", "asks")

    def __init__(self):
        # price_bin (int) -> aggregate size
        self.bids: Dict[int, float] = {}
        self.asks: Dict[int, float] = {}

    def add(self, side: str, price_bin: int, size: float):
        book = self.bids if side == "B" else (self.asks if side == "A" else None)
        if book is None:
            return
        book[price_bin] = book.get(price_bin, 0.0) + float(size)

    def cancel(self, side: str, price_bin: int, size: float):
        book = self.bids if side == "B" else (self.asks if side == "A" else None)
        if book is None:
            return
        cur = book.get(price_bin, 0.0)
        new = cur - float(size)
        if new <= 1e-9:
            book.pop(price_bin, None)
        else:
            book[price_bin] = new

    def consume_trade(self, aggressor_side: str, price_bin: int, size: float):
        # aggressor 'B' → consume from asks; aggressor 'A' → consume from bids.
        # Consume from the specified price level (Databento trades carry the exact level price).
        if aggressor_side == "B":
            book = self.asks
        elif aggressor_side == "A":
            book = self.bids
        else:
            return
        cur = book.get(price_bin, 0.0)
        new = cur - float(size)
        if new <= 1e-9:
            book.pop(price_bin, None)
        else:
            book[price_bin] = new

    def clear_side(self, side: str):
        if side == "B":
            self.bids.clear()
        elif side == "A":
            self.asks.clear()

    def best_bid_bin(self) -> Optional[int]:
        return max(self.bids) if self.bids else None

    def best_ask_bin(self) -> Optional[int]:
        return min(self.asks) if self.asks else None

    def best_bid_size(self) -> float:
        bb = self.best_bid_bin()
        return self.bids.get(bb, 0.0) if bb is not None else 0.0

    def best_ask_size(self) -> float:
        ba = self.best_ask_bin()
        return self.asks.get(ba, 0.0) if ba is not None else 0.0

    def top_n_depth_total(self, n: int = TOP_N_DEPTH) -> Tuple[float, float]:
        """Return (sum top-n bid volume, sum top-n ask volume)."""
        if self.bids:
            top_bid_bins = sorted(self.bids.keys(), reverse=True)[:n]
            sb = sum(self.bids[b] for b in top_bid_bins)
        else:
            sb = 0.0
        if self.asks:
            top_ask_bins = sorted(self.asks.keys())[:n]
            sa = sum(self.asks[b] for b in top_ask_bins)
        else:
            sa = 0.0
        return sb, sa


def _apply_event_to_book(book: _BookState, action: str, side: str,
                         price: float, size: float):
    """Apply one MBO event to the book state. Tolerant to unknown actions."""
    if not np.isfinite(price) or price <= 0 or size is None or size <= 0:
        # Some MBO events carry meta (clears, etc.) without valid price/size. Skip.
        if action == "R":  # clear-book
            # Databento 'R' clears the entire side
            book.clear_side(side)
        return
    price_bin = int(round(price / TICK_SIZE))
    if action == "A":
        book.add(side, price_bin, size)
    elif action == "C":
        book.cancel(side, price_bin, size)
    elif action == "M":
        # Modify is rare in MBO; treat conservatively as a no-op for total depth
        # (the prior add at this id is not separately tracked since we aggregate by price-bin).
        pass
    elif action in ("T", "F"):
        book.consume_trade(side, price_bin, size)
    elif action == "R":
        book.clear_side(side)
    # Unknown actions: ignore.


# ==============================================================================
# Large-order percentile (NEW in v3.2.1) — trailing 5-day rolling
# ==============================================================================
# We compute, for each day, the 95th percentile of add-event sizes from the prior
# 5 days' MBO data. Stored to disk so subsequent days don't re-scan. If insufficient
# history, we fall back to the 95th percentile of the current day's own add sizes.
# ==============================================================================

def _large_order_pctile_path(d: date) -> Path:
    LARGE_ORDER_PCTILE_DIR.mkdir(parents=True, exist_ok=True)
    return LARGE_ORDER_PCTILE_DIR / f"{d.strftime('%Y%m%d')}.json"


def _save_day_add_size_histogram(df_day: pd.DataFrame, d: date) -> None:
    """Save quantiles of the current day's add-event size distribution for later use."""
    adds = df_day[df_day["action"] == "A"]
    if len(adds) == 0:
        return
    sizes = adds["size"].to_numpy(dtype=np.float64)
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return
    # Save just the q95 and a few other useful quantiles. Simple JSON.
    info = {
        "date": d.isoformat(),
        "n_adds": int(sizes.size),
        "q50": float(np.percentile(sizes, 50)),
        "q90": float(np.percentile(sizes, 90)),
        "q95": float(np.percentile(sizes, 95)),
        "q99": float(np.percentile(sizes, 99)),
    }
    with _large_order_pctile_path(d).open("w") as f:
        json.dump(info, f)


def _load_trailing_5d_q95(d: date) -> Optional[float]:
    """Average q95 of add-sizes across the prior up-to-5 days. None if no history."""
    out = []
    cur = d - timedelta(days=1)
    tries = 0
    while len(out) < HISTORY_DAYS_FOR_LARGE_ORDER_PCTILE and tries < HISTORY_DAYS_FOR_LARGE_ORDER_PCTILE + 10:
        p = _large_order_pctile_path(cur)
        if p.exists():
            try:
                with p.open() as f:
                    info = json.load(f)
                if "q95" in info and np.isfinite(info["q95"]) and info["q95"] > 0:
                    out.append(float(info["q95"]))
            except Exception:
                pass
        cur -= timedelta(days=1)
        tries += 1
    if not out:
        return None
    return float(np.mean(out))


# ==============================================================================
# Tier 2 — v3.2.1
# ==============================================================================

def build_tier2_for_day_v321(df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Build T2 with 19 features per 100ms bucket. Stores already-normalized values
    where HC #298 says they need transformation (log1p, sign-preserved, /23400, etc).
    """
    rth_open = rth_open_ns_for_date(d)

    df = df.copy()
    df["bucket_idx"] = ((df["ts_event_ns"] - rth_open) // TIER2_BUCKET_NS).astype(np.int64)
    df["seconds_since_rth_open_raw"] = ((df["ts_event_ns"] - rth_open) / 1e9).astype(np.float32)

    df["is_trade"] = (df["action"] == "T").astype(np.int8)
    df["is_cancel"] = (df["action"] == "C").astype(np.int8)
    df["is_add"] = (df["action"] == "A").astype(np.int8)
    df["is_order_event"] = (df["action"] != "T").astype(np.int8)

    df["trade_size"] = df["size"].where(df["is_trade"] == 1, 0).astype(np.float32)
    df["buy_volume"] = df["trade_size"].where(df["side"] == "B", 0)
    df["sell_volume"] = df["trade_size"].where(df["side"] == "A", 0)
    df["add_size"] = df["size"].where(df["is_add"] == 1, 0).astype(np.float32)

    # TRADE prices only for OHLC (non-trade events have limit prices)
    df["trade_price"] = df["price"].where(df["is_trade"] == 1, np.nan).astype(np.float64)

    grouped = df.groupby("bucket_idx", sort=True)
    agg = grouped.agg(
        open_price=("trade_price", "first"),
        close_price=("trade_price", "last"),
        high_price=("trade_price", "max"),
        low_price=("trade_price", "min"),
        n_trades=("is_trade", "sum"),
        trade_volume=("trade_size", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        n_cancels=("is_cancel", "sum"),
        n_adds=("is_add", "sum"),
        n_order_events=("is_order_event", "sum"),
        add_volume_sum=("add_size", "sum"),
        seconds_since_rth_open_raw=("seconds_since_rth_open_raw", "first"),
    ).reset_index()

    out = pd.DataFrame()
    out["bucket_idx"] = agg["bucket_idx"]

    # ---- HC #298 fix: seconds_since_rth_open stored as /23400 (PASS) ----
    secs_raw = agg["seconds_since_rth_open_raw"].astype(np.float32).values
    out["seconds_since_rth_open"] = (secs_raw / np.float32(RTH_SECONDS_PER_DAY)).astype(np.float32)

    # ---- Prices / range (same as v3.2; clipped raw values, BOUNDED) ----
    open_p = agg["open_price"].astype(np.float64).values
    close_p = agg["close_price"].astype(np.float64).values
    high_p = agg["high_price"].astype(np.float64).values
    low_p = agg["low_price"].astype(np.float64).values
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
    mae = np.where(np.isfinite(open_p) & np.isfinite(low_p),  (open_p - low_p)  / TICK_SIZE, 0.0)
    rng = np.where(np.isfinite(high_p) & np.isfinite(low_p),  (high_p - low_p)  / TICK_SIZE, 0.0)
    out["bucket_mfe_ticks"]   = np.clip(mfe, 0, 40).astype(np.float32)
    out["bucket_mae_ticks"]   = np.clip(mae, 0, 40).astype(np.float32)
    out["bucket_range_ticks"] = np.clip(rng, 0, 40).astype(np.float32)

    # ---- HC #298 fix: heavy-tail counts/volumes as LOG1P ----
    out["n_trades"]        = np.log1p(agg["n_trades"].astype(np.float32).values).astype(np.float32)
    out["trade_volume"]    = np.log1p(agg["trade_volume"].astype(np.float32).values).astype(np.float32)
    out["n_cancels"]       = np.log1p(agg["n_cancels"].astype(np.float32).values).astype(np.float32)
    out["n_adds"]          = np.log1p(agg["n_adds"].astype(np.float32).values).astype(np.float32)
    out["n_order_events"]  = np.log1p(agg["n_order_events"].astype(np.float32).values).astype(np.float32)

    # ---- Aggressor / signed (HC #298 fix: SIGNED log-z = sign(x)*log1p(|x|)) ----
    total_vol = (agg["buy_volume"] + agg["sell_volume"]).values
    out["aggressor_buy_ratio"] = np.where(
        total_vol > 0,
        agg["buy_volume"].values / np.maximum(total_vol, 1.0),
        0.5,
    ).astype(np.float32)
    sv_raw = (agg["buy_volume"] - agg["sell_volume"]).astype(np.float32).values
    out["signed_volume"] = (np.sign(sv_raw) * np.log1p(np.abs(sv_raw))).astype(np.float32)

    # ---- cancel/add ratio ----
    add_cancel = agg["n_adds"] + agg["n_cancels"]
    out["cancel_add_ratio"] = np.where(
        add_cancel > 0,
        agg["n_cancels"].values / np.maximum(add_cancel.values, 1.0),
        0.5,
    ).astype(np.float32)
    # avg_order_size kept as raw (small range; not log'd in v3.2 either)
    out["avg_order_size"] = np.where(
        agg["n_adds"] > 0,
        agg["add_volume_sum"].values / np.maximum(agg["n_adds"].values, 1.0),
        0.0,
    ).astype(np.float32)

    # ==========================================================================
    # NEW v3.2.1 — 5 features from L2 book reconstruction
    # ==========================================================================
    n_buckets = len(agg)
    bucket_idx_arr = agg["bucket_idx"].values.astype(np.int64)

    # Determine large_order_size threshold from rolling 5-day q95 (with fallback)
    q95_threshold = _load_trailing_5d_q95(d)
    if q95_threshold is None or not np.isfinite(q95_threshold) or q95_threshold <= 0:
        # Fallback: current day's own q95 (note: this is a leakage caveat documented in builder logs)
        cur_day_adds = df.loc[df["action"] == "A", "size"].to_numpy(dtype=np.float64)
        cur_day_adds = cur_day_adds[cur_day_adds > 0]
        if cur_day_adds.size > 0:
            q95_threshold = float(np.percentile(cur_day_adds, 95))
            logger.warning(f"[{d}] no 5-day q95 history; falling back to current-day q95={q95_threshold:.0f} (slight leakage)")
        else:
            q95_threshold = 1e9  # effectively disables large_order_count
            logger.warning(f"[{d}] no add events; large_order_count → 0 for all buckets")

    # Walk events in time order. We collect per-bucket accumulators.
    # df is already in event order (load_mbo_day preserves it); make absolutely sure:
    df_sorted = df.sort_values("ts_event_ns", kind="stable")
    book = _BookState()

    # Per-bucket accumulators
    micro_first = np.full(n_buckets, np.nan, dtype=np.float64)
    micro_last  = np.full(n_buckets, np.nan, dtype=np.float64)
    spread_sum  = np.zeros(n_buckets, dtype=np.float64)
    spread_cnt  = np.zeros(n_buckets, dtype=np.int64)
    tob_changes = np.zeros(n_buckets, dtype=np.int64)
    depth_sum   = np.zeros(n_buckets, dtype=np.float64)  # sum (top5_bid+top5_ask)/2 per event
    depth_cnt   = np.zeros(n_buckets, dtype=np.int64)
    large_count = np.zeros(n_buckets, dtype=np.int64)

    # Map bucket_idx → row index in `agg`
    bidx_to_row = {int(b): i for i, b in enumerate(bucket_idx_arr)}

    prev_bb = None
    prev_ba = None
    # Iterate using numpy arrays for speed
    ev_action = df_sorted["action"].to_numpy()
    ev_side   = df_sorted["side"].to_numpy()
    ev_price  = df_sorted["price"].to_numpy(dtype=np.float64)
    ev_size   = df_sorted["size"].to_numpy(dtype=np.float64)
    ev_bidx   = df_sorted["bucket_idx"].to_numpy(dtype=np.int64)
    n_events = len(df_sorted)

    for i in range(n_events):
        action = ev_action[i]
        side = ev_side[i]
        price = ev_price[i]
        size = ev_size[i]
        bidx = int(ev_bidx[i])

        # Apply event to the book
        _apply_event_to_book(book, action, side, price, size)

        # Read post-event book state
        bb_bin = book.best_bid_bin()
        ba_bin = book.best_ask_bin()

        if bb_bin is not None and ba_bin is not None and ba_bin > bb_bin:
            spread_ticks = float(ba_bin - bb_bin)
            spread_ticks = min(spread_ticks, 20.0)  # clip [0, 20]
            spread_ticks = max(spread_ticks, 0.0)
            bb_size = book.best_bid_size()
            ba_size = book.best_ask_size()
            denom = bb_size + ba_size
            if denom > 0:
                micro = (bb_bin * TICK_SIZE * ba_size + ba_bin * TICK_SIZE * bb_size) / denom
            else:
                micro = (bb_bin + ba_bin) * 0.5 * TICK_SIZE

            row = bidx_to_row.get(bidx, -1)
            if row >= 0:
                if not np.isfinite(micro_first[row]):
                    micro_first[row] = micro
                micro_last[row] = micro
                spread_sum[row] += spread_ticks
                spread_cnt[row] += 1
                # top-5 depth (use mean of the two sides — keeps it symmetric)
                tb, ta = book.top_n_depth_total(TOP_N_DEPTH)
                depth_sum[row] += 0.5 * (tb + ta)
                depth_cnt[row] += 1

        # TOB change detection: did best_bid_bin OR best_ask_bin move?
        if (bb_bin is not None and prev_bb is not None and bb_bin != prev_bb) or \
           (ba_bin is not None and prev_ba is not None and ba_bin != prev_ba):
            row = bidx_to_row.get(bidx, -1)
            if row >= 0:
                tob_changes[row] += 1
        prev_bb, prev_ba = bb_bin, ba_bin

        # Large-order count: any event with size > q95_threshold (we count add events;
        # cancels of large orders also count as "large flow")
        if size > q95_threshold and action in ("A", "C"):
            row = bidx_to_row.get(bidx, -1)
            if row >= 0:
                large_count[row] += 1

    # microprice_change_ticks (bucket-end - bucket-start) in ticks
    with np.errstate(invalid="ignore"):
        micro_change_price = micro_last - micro_first
    micro_change_ticks = np.where(
        np.isfinite(micro_change_price),
        micro_change_price / TICK_SIZE,
        0.0,
    )
    # Clip to a sane range (±50 ticks); microprice rarely moves more than a few ticks intra-100ms
    out["microprice_change_ticks"] = np.clip(micro_change_ticks, -50.0, 50.0).astype(np.float32)

    avg_spread = np.where(spread_cnt > 0, spread_sum / np.maximum(spread_cnt, 1), 0.0)
    out["avg_spread_in_bucket_ticks"] = np.clip(avg_spread, 0.0, 20.0).astype(np.float32)

    # n_tob_changes: LOG1P
    out["n_top_of_book_changes"] = np.log1p(tob_changes.astype(np.float32)).astype(np.float32)

    avg_depth = np.where(depth_cnt > 0, depth_sum / np.maximum(depth_cnt, 1), 0.0)
    # LOG1P
    out["avg_top5_depth_volume"] = np.log1p(np.maximum(avg_depth, 0.0)).astype(np.float32)

    # large_order_count: LOG1P
    out["large_order_count"] = np.log1p(large_count.astype(np.float32)).astype(np.float32)

    # Persist this day's add-size histogram for future days' rolling q95
    try:
        _save_day_add_size_histogram(df, d)
    except Exception as e:
        logger.warning(f"[{d}] failed to save add-size histogram: {e}")

    cols_final = ["bucket_idx"] + TIER2_FEATURE_COLS
    out = out[cols_final]
    return out


# ==============================================================================
# LGBM Vol Inference Helper (NEW in v3.2.1)
# ==============================================================================

def _find_latest_lgbm_pkl() -> Optional[Path]:
    """Find the most-recent vol_v3_YYYYMMDD_models.pkl in LGBM_VOL_DIR."""
    if not LGBM_VOL_DIR.exists():
        return None
    candidates = sorted(LGBM_VOL_DIR.glob(f"{LGBM_VOL_PREFIX}*{LGBM_VOL_SUFFIX}"))
    if not candidates:
        return None
    return candidates[-1]


def _load_lgbm_vol_model(pkl_path: Path) -> Optional[Dict]:
    try:
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        # Expect keys: models (list of LGBMRegressor), feature_names, train_mean, train_std, horizons_s
        if "models" not in obj or "train_mean" not in obj or "train_std" not in obj:
            logger.warning(f"LGBM pkl missing expected keys: {pkl_path.name}")
            return None
        return obj
    except Exception as e:
        logger.warning(f"Failed to load LGBM pkl {pkl_path.name}: {e}")
        return None


def _compute_lgbm_28_features(window: np.ndarray) -> np.ndarray:
    """Vectorized 28-feature extractor — mirrors compute_features() in train_vol_lgbm_v3.py.
    Input: (W, 6) float32 window of smart_v3 normalized MBO events.
    Output: (28,) float32.
    """
    out = np.empty(LGBM_N_FEATURES, dtype=np.float32)
    W = window.shape[0]
    q4_start = W * 3 // 4
    half_start = W // 2

    prices  = window[:, LGBM_COL_PRICE]
    qty_log = window[:, LGBM_COL_QTY]
    qty     = np.maximum(np.exp(qty_log) - 1.0, 0.0)
    sides   = window[:, LGBM_COL_SIDE]
    spreads = window[:, LGBM_COL_SPREAD]
    types   = window[:, LGBM_COL_TYPE]
    tdelta  = np.maximum(np.exp(window[:, LGBM_COL_TIME]) - 1.0, 0.0)

    rets   = np.diff(prices)
    abs_r  = np.abs(rets)
    rv_full = abs_r.mean()
    rv_half = abs_r[half_start:].mean() if abs_r.size > half_start else rv_full
    rv_q4   = abs_r[q4_start:].mean()   if abs_r.size > q4_start   else rv_full
    rv_q1   = abs_r[:half_start].mean() if abs_r.size > half_start else rv_full
    rv_accel = rv_q4 * 4.0 - rv_full
    out[0] = rv_full; out[1] = rv_half; out[2] = rv_q4; out[3] = rv_q1; out[4] = rv_accel

    sp_mean = spreads.mean()
    sp_q4   = spreads[q4_start:].mean()
    sp_trend = spreads[half_start:].mean() - spreads[:half_start].mean()
    p_range = prices.max() - prices.min()
    out[5] = sp_mean; out[6] = sp_q4; out[7] = sp_trend; out[8] = p_range

    sgn = np.where(sides == 0, 1.0, np.where(sides == 1, -1.0, 0.0))
    sv = sgn * qty
    ofi_full = sv.sum()
    ofi_half = sv[half_start:].sum()
    ofi_q4_  = sv[q4_start:].sum()
    ofi_accel = ofi_q4_ * 4.0 - ofi_full
    total_qty = qty.sum() + 1e-8
    cum_delta_norm = ofi_full / total_qty
    out[9]  = ofi_full;  out[10] = ofi_half
    out[11] = ofi_q4_;   out[12] = ofi_accel
    out[13] = cum_delta_norm

    sec_full = tdelta.sum() + 1e-6
    sec_q4   = tdelta[q4_start:].sum() + 1e-6
    evt_d_full = float(W) / sec_full
    evt_d_q4   = float(W - q4_start) / sec_q4
    dens_accel = evt_d_q4 - evt_d_full
    is_trade   = (types == LGBM_TYPE_TRADE).astype(np.float32)
    trade_share = is_trade.mean()
    out[14] = evt_d_full; out[15] = evt_d_q4; out[16] = dens_accel; out[17] = trade_share

    vol_total = total_qty
    vol_q4 = qty[q4_start:].sum()
    vol_q4_share = vol_q4 / vol_total
    vol_std = qty.std()
    vol_med = float(np.median(qty))
    vol_skew = (qty.mean() - vol_med) / (vol_std + 1e-8)
    out[18] = np.log1p(vol_total)
    out[19] = vol_q4_share
    out[20] = np.log1p(vol_std)
    out[21] = vol_skew

    is_cancel = (types == 1).astype(np.float32)
    cancel_share = is_cancel.mean()
    cb = float((is_cancel * (sides == 0)).sum())
    ca = float((is_cancel * (sides == 1)).sum())
    cancel_asym = (cb - ca) / (cb + ca + 1e-6)
    cancel_q4 = is_cancel[q4_start:].mean()
    out[22] = cancel_share; out[23] = cancel_asym; out[24] = cancel_q4

    out[25] = sp_mean * evt_d_full
    out[26] = rv_full * sp_mean
    out[27] = abs(ofi_full) / total_qty

    return out


def _compute_lgbm_features_batch(windows: np.ndarray) -> np.ndarray:
    """Batch version: (N, W, 6) → (N, 28).
    Mirrors compute_features() in train_vol_lgbm_v3.py.
    """
    N, W, _ = windows.shape
    out = np.empty((N, LGBM_N_FEATURES), dtype=np.float32)
    q4_start = W * 3 // 4
    half_start = W // 2
    prices  = windows[:, :, LGBM_COL_PRICE]
    qty_log = windows[:, :, LGBM_COL_QTY]
    qty     = np.maximum(np.exp(qty_log) - 1.0, 0.0)
    sides   = windows[:, :, LGBM_COL_SIDE]
    spreads = windows[:, :, LGBM_COL_SPREAD]
    types   = windows[:, :, LGBM_COL_TYPE]
    tdelta  = np.maximum(np.exp(windows[:, :, LGBM_COL_TIME]) - 1.0, 0.0)

    rets   = np.diff(prices, axis=1)
    abs_r  = np.abs(rets)
    rv_full = abs_r.mean(axis=1)
    rv_half = abs_r[:, half_start:].mean(axis=1)
    rv_q4   = abs_r[:, q4_start:].mean(axis=1)
    rv_q1   = abs_r[:, :half_start].mean(axis=1)
    rv_accel = rv_q4 * 4.0 - rv_full
    out[:, 0] = rv_full; out[:, 1] = rv_half
    out[:, 2] = rv_q4;   out[:, 3] = rv_q1; out[:, 4] = rv_accel

    sp_mean = spreads.mean(axis=1)
    sp_q4   = spreads[:, q4_start:].mean(axis=1)
    sp_trend = spreads[:, half_start:].mean(axis=1) - spreads[:, :half_start].mean(axis=1)
    p_range = prices.max(axis=1) - prices.min(axis=1)
    out[:, 5] = sp_mean;  out[:, 6] = sp_q4
    out[:, 7] = sp_trend; out[:, 8] = p_range

    sgn = np.where(sides == 0, 1.0, np.where(sides == 1, -1.0, 0.0)).astype(np.float32)
    sv = sgn * qty
    ofi_full = sv.sum(axis=1)
    ofi_half = sv[:, half_start:].sum(axis=1)
    ofi_q4_  = sv[:, q4_start:].sum(axis=1)
    ofi_accel = ofi_q4_ * 4.0 - ofi_full
    total_qty = qty.sum(axis=1) + 1e-8
    cum_delta_norm = ofi_full / total_qty
    out[:, 9]  = ofi_full;  out[:, 10] = ofi_half
    out[:, 11] = ofi_q4_;   out[:, 12] = ofi_accel
    out[:, 13] = cum_delta_norm

    sec_full = tdelta.sum(axis=1) + 1e-6
    sec_q4   = tdelta[:, q4_start:].sum(axis=1) + 1e-6
    evt_d_full = float(W) / sec_full
    evt_d_q4   = float(W - q4_start) / sec_q4
    dens_accel = evt_d_q4 - evt_d_full
    is_trade   = (types == LGBM_TYPE_TRADE).astype(np.float32)
    trade_share = is_trade.mean(axis=1)
    out[:, 14] = evt_d_full; out[:, 15] = evt_d_q4
    out[:, 16] = dens_accel; out[:, 17] = trade_share

    vol_total = total_qty
    vol_q4 = qty[:, q4_start:].sum(axis=1)
    vol_q4_share = vol_q4 / vol_total
    vol_std = qty.std(axis=1)
    vol_med = np.median(qty, axis=1)
    vol_skew = (qty.mean(axis=1) - vol_med) / (vol_std + 1e-8)
    out[:, 18] = np.log1p(vol_total)
    out[:, 19] = vol_q4_share
    out[:, 20] = np.log1p(vol_std)
    out[:, 21] = vol_skew

    is_cancel = (types == 1).astype(np.float32)
    cancel_share = is_cancel.mean(axis=1)
    cb = (is_cancel * (sides == 0)).sum(axis=1)
    ca = (is_cancel * (sides == 1)).sum(axis=1)
    cancel_asym = (cb - ca) / (cb + ca + 1e-6)
    cancel_q4 = is_cancel[:, q4_start:].mean(axis=1)
    out[:, 22] = cancel_share; out[:, 23] = cancel_asym; out[:, 24] = cancel_q4

    out[:, 25] = sp_mean * evt_d_full
    out[:, 26] = rv_full * sp_mean
    out[:, 27] = np.abs(ofi_full) / total_qty
    return out


def _predict_lgbm_vol_per_second(d: date, n_seconds: int) -> Tuple[np.ndarray, np.ndarray]:
    """For one day, compute LGBM vol predictions for each second 0..n_seconds-1.
    Returns (pred_5min_arr, pred_30min_arr) each shape (n_seconds,) float32.

    Strategy:
      1. Load latest LGBM pkl (models + train_mean + train_std).
      2. Load smart_v3 mbo_events npz for this day (events 6-col + timestamps_ns).
      3. For each RTH second, find the last event index whose ts_ns ≤ second_end_ns,
         build a window of LGBM_WINDOW=1000 events ending there, compute the 28
         features, normalize (Xtr_mean, Xtr_std), predict each horizon.
      4. If any step fails, return zeros + log warning.

    If anchor index < LGBM_WINDOW-1 (insufficient events at start of day), fill 0.
    """
    zeros = np.zeros(n_seconds, dtype=np.float32)
    pkl_path = _find_latest_lgbm_pkl()
    if pkl_path is None:
        logger.warning(f"[{d}] no LGBM vol pkl found in {LGBM_VOL_DIR}; defaulting predictions to 0")
        return zeros, zeros
    obj = _load_lgbm_vol_model(pkl_path)
    if obj is None:
        logger.warning(f"[{d}] LGBM pkl load failed; defaulting predictions to 0")
        return zeros, zeros

    # Load smart_v3 mbo_events for this day
    date_str = d.strftime("%Y%m%d")
    npz_path = SMART_V3_NPZ_ROOT / f"{date_str}_mbo_events.npz"
    if not npz_path.exists():
        logger.warning(f"[{d}] smart_v3 events npz not found at {npz_path}; LGBM preds → 0")
        return zeros, zeros
    try:
        ev_npz = np.load(npz_path, allow_pickle=True)
        events = ev_npz["events"].astype(np.float32)  # (N, 6)
        ts_ns = ev_npz["timestamps"].astype(np.int64) if "timestamps" in ev_npz.files else None
    except Exception as e:
        logger.warning(f"[{d}] failed to load smart_v3 npz: {e}; LGBM preds → 0")
        return zeros, zeros
    if ts_ns is None or len(events) < LGBM_WINDOW + 10:
        logger.warning(f"[{d}] insufficient smart_v3 events for LGBM ({len(events)} < {LGBM_WINDOW+10})")
        return zeros, zeros

    rth_open_ns = rth_open_ns_for_date(d)

    # For each second s in [0, n_seconds), the cutoff timestamp is rth_open_ns + (s+1)*1e9 ns.
    # We use searchsorted to find anchor_idx = (last index < cutoff).
    second_idxs = np.arange(n_seconds, dtype=np.int64)
    cutoff_ns = rth_open_ns + (second_idxs + 1) * TIER3_SNAP_NS
    # last_event_idx[s] = position in ts_ns of last event with ts <= cutoff
    last_idxs = np.searchsorted(ts_ns, cutoff_ns, side="right") - 1
    # valid: have enough events for a full LGBM window
    valid_mask = last_idxs >= (LGBM_WINDOW - 1)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        logger.warning(f"[{d}] no valid LGBM anchors (insufficient pre-RTH event buildup)")
        return zeros, zeros

    valid_seconds = second_idxs[valid_mask]
    valid_anchors = last_idxs[valid_mask]

    # Build all windows: shape (n_valid, W, 6)
    # Memory-conscious chunking
    CHUNK = 4096
    train_mean = np.asarray(obj["train_mean"], dtype=np.float32)
    train_std = np.asarray(obj["train_std"], dtype=np.float32)
    eps = np.float32(1e-8)
    models = obj["models"]
    if len(models) < max(LGBM_HORIZON_IDX_FOR_5MIN, LGBM_HORIZON_IDX_FOR_30MIN) + 1:
        logger.warning(f"[{d}] LGBM model has only {len(models)} horizons; need ≥3. Defaulting preds → 0")
        return zeros, zeros

    pred_5min_full = np.zeros(n_seconds, dtype=np.float32)
    pred_30min_full = np.zeros(n_seconds, dtype=np.float32)

    col_idx = np.arange(LGBM_WINDOW, dtype=np.int64)
    for ci in range(0, n_valid, CHUNK):
        sub_anchor = valid_anchors[ci:ci + CHUNK]
        sub_starts = sub_anchor - (LGBM_WINDOW - 1)
        # Gather windows
        idx = sub_starts[:, None] + col_idx[None, :]  # (chunk, W)
        try:
            w = events[idx]  # (chunk, W, 6)
        except IndexError as e:
            logger.warning(f"[{d}] LGBM window gather failed at chunk {ci}: {e}; this chunk → 0")
            continue
        try:
            X = _compute_lgbm_features_batch(w)
            X_norm = np.clip((X - train_mean) / np.maximum(train_std + eps, eps), -8.0, 8.0).astype(np.float32)
            # NaN check
            if np.isnan(X_norm).any():
                X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred_5min  = models[LGBM_HORIZON_IDX_FOR_5MIN].predict(X_norm)
                pred_30min = models[LGBM_HORIZON_IDX_FOR_30MIN].predict(X_norm)
            sub_secs = valid_seconds[ci:ci + CHUNK]
            pred_5min_full[sub_secs]  = pred_5min.astype(np.float32)
            pred_30min_full[sub_secs] = pred_30min.astype(np.float32)
        except Exception as e:
            logger.warning(f"[{d}] LGBM inference chunk {ci} failed: {e}; chunk → 0")

    return pred_5min_full, pred_30min_full


# ==============================================================================
# Tier 3 — v3.2.1
# ==============================================================================

def build_tier3_for_day_v321(df: pd.DataFrame, d: date,
                             prior_states: List[Dict]) -> pd.DataFrame:
    """Build T3 with 31 features per RTH second.

    Changes vs v3.2:
      - Distance features written as clip([-200,200]) / 100.0  (HC #298 fix)
      - 2 LGBM vol predictions added (Priority-1 item #3)
      - 4 session-phase one-hot added (Priority-1 item #5)
      - is_lunch_lull, is_close_hour, tod_sin, tod_cos, dow_sin kept (PASS in dataloader)
      - dow_cos dropped — was in v3.2 but v3.2.1 only keeps dow_sin per task spec (25 + 6 = 31)
    """
    rth_open = rth_open_ns_for_date(d)
    rth_close = rth_close_ns_for_date(d)
    df_rth = df[(df["ts_event_ns"] >= rth_open) & (df["ts_event_ns"] < rth_close)].copy()
    if len(df_rth) == 0:
        return pd.DataFrame(columns=["second_idx"] + TIER3_FEATURE_COLS)

    n_seconds = RTH_SECONDS_PER_DAY
    sec_idx = np.arange(n_seconds, dtype=np.int32)

    trades = df_rth[df_rth["action"] == "T"].copy()
    trades["sec_idx"] = ((trades["ts_event_ns"] - rth_open) // TIER3_SNAP_NS).astype(np.int64)
    trades["sec_idx"] = np.clip(trades["sec_idx"], 0, n_seconds - 1)
    trades["price_x_size"] = trades["price"] * trades["size"]
    trades["price_bin"] = np.round(trades["price"] / TICK_SIZE).astype(np.int64)

    df_rth["sec_idx"] = ((df_rth["ts_event_ns"] - rth_open) // TIER3_SNAP_NS).astype(np.int64)
    df_rth["sec_idx"] = np.clip(df_rth["sec_idx"], 0, n_seconds - 1)

    if len(trades) > 0:
        last_price_per_sec = trades.groupby("sec_idx")["price"].last()
        last_price_arr = np.full(n_seconds, np.nan)
        last_price_arr[last_price_per_sec.index.values.astype(int)] = last_price_per_sec.values
        last_price_arr = pd.Series(last_price_arr).ffill().bfill().values
    else:
        last_price_arr = np.full(n_seconds, 1.0)

    # ===== Running VWAP + intraday H/L =====
    trade_agg = trades.groupby("sec_idx").agg(
        vol_x_price=("price_x_size", "sum"),
        vol=("size", "sum"),
        high=("price", "max"),
        low=("price", "min"),
    )
    trade_idx = trade_agg.index.values.astype(int)
    vol_at_idx = np.zeros(n_seconds, dtype=np.float64)
    vxp_at_idx = np.zeros(n_seconds, dtype=np.float64)
    vol_at_idx[trade_idx] = trade_agg["vol"].values
    vxp_at_idx[trade_idx] = trade_agg["vol_x_price"].values
    cum_vol = np.cumsum(vol_at_idx)
    cum_vol_x_price = np.cumsum(vxp_at_idx)
    running_vwap = np.where(cum_vol > 0, cum_vol_x_price / np.maximum(cum_vol, 1e-9), last_price_arr)
    intraday_high = np.maximum.accumulate(last_price_arr)
    intraday_low  = np.minimum.accumulate(last_price_arr)

    # ===== Volume profile (VPOC, VAH, VAL) =====
    if len(trades) > 0:
        bin_min = int(np.floor(trades["price"].min() / TICK_SIZE))
        bin_max = int(np.ceil(trades["price"].max() / TICK_SIZE))
    else:
        bin_min = int(np.floor(np.nanmin(last_price_arr) / TICK_SIZE))
        bin_max = int(np.ceil(np.nanmax(last_price_arr) / TICK_SIZE))
    median_bin = int(np.round(np.nanmedian(last_price_arr) / TICK_SIZE))
    bin_min = max(bin_min, median_bin - 1000)
    bin_max = min(bin_max, median_bin + 1000)
    n_bins = max(bin_max - bin_min + 1, 1)
    vol_by_bin_per_sec = np.zeros((n_seconds, n_bins), dtype=np.float64)
    if len(trades) > 0:
        rel_bin = (trades["price_bin"].values - bin_min).astype(np.int64)
        rel_bin = np.clip(rel_bin, 0, n_bins - 1)
        sec_i = trades["sec_idx"].values.astype(np.int64)
        np.add.at(vol_by_bin_per_sec, (sec_i, rel_bin), trades["size"].values.astype(np.float64))
    cum_vol_by_bin = np.cumsum(vol_by_bin_per_sec, axis=0)
    vpoc_bin = cum_vol_by_bin.argmax(axis=1)
    vpoc_price = (vpoc_bin + bin_min) * TICK_SIZE
    total_vol_per_sec = cum_vol_by_bin.sum(axis=1)
    vah_price = np.copy(vpoc_price)
    val_price = np.copy(vpoc_price)
    for s in range(n_seconds):
        if total_vol_per_sec[s] <= 0:
            continue
        bins = cum_vol_by_bin[s]
        target = 0.70 * total_vol_per_sec[s]
        poc = int(vpoc_bin[s])
        lo = poc
        hi = poc
        acc = bins[poc]
        while acc < target and (lo > 0 or hi < n_bins - 1):
            left_vol  = bins[lo - 1] if lo > 0 else -1
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

    current_bin = (np.round(last_price_arr / TICK_SIZE).astype(np.int64) - bin_min)
    current_bin = np.clip(current_bin, 0, n_bins - 1)
    vol_at_current = cum_vol_by_bin[np.arange(n_seconds), current_bin]
    max_vol = cum_vol_by_bin.max(axis=1)
    vol_pctile = np.where(max_vol > 0, vol_at_current / np.maximum(max_vol, 1e-9), 0.0)
    pos_in_va = np.where(
        last_price_arr > vah_price, 1.0,
        np.where(last_price_arr < val_price, -1.0, 0.0),
    )

    # ===== Prior session features =====
    if len(prior_states) > 0:
        ps = prior_states[0]
        prior_close = ps["close"]
        prior_vwap = ps["vwap"]
        prior_high = ps["high"]
        prior_low = ps["low"]
        prior_vpoc = ps["vpoc"]
    else:
        prior_close = prior_vwap = prior_high = prior_low = prior_vpoc = np.nan
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

    # ---- HC #298 fix: distance features stored as clip([-200,200]) / 100.0  ----
    def dist_norm(arr_diff: np.ndarray) -> np.ndarray:
        """Take a (last_price_arr - target) array, return clip([-200,200])/100 in ticks."""
        return (np.clip(arr_diff / TICK_SIZE, -200.0, 200.0) / 100.0).astype(np.float32)

    def dist_norm_const(target: float) -> np.ndarray:
        if not np.isfinite(target):
            return np.zeros(n_seconds, dtype=np.float32)
        return dist_norm(last_price_arr - target)

    out = pd.DataFrame()
    out["second_idx"] = sec_idx

    # Price location / S-R (5)
    out["dist_intraday_high_ticks"]       = dist_norm(last_price_arr - intraday_high)
    out["dist_intraday_low_ticks"]        = dist_norm(last_price_arr - intraday_low)
    out["dist_session_vwap_ticks"]        = dist_norm(last_price_arr - running_vwap)
    out["dist_prior_session_close_ticks"] = dist_norm_const(prior_close)
    out["dist_prior_session_vwap_ticks"]  = dist_norm_const(prior_vwap)

    # Volume profile (5)
    out["dist_intraday_vpoc_ticks"] = dist_norm(last_price_arr - vpoc_price)
    out["dist_intraday_vah_ticks"]  = dist_norm(last_price_arr - vah_price)
    out["dist_intraday_val_ticks"]  = dist_norm(last_price_arr - val_price)
    out["position_in_value_area"]            = pos_in_va.astype(np.float32)   # PASS
    out["volume_at_current_price_pctile"]    = vol_pctile.astype(np.float32)  # PASS

    # Prior session levels (4)
    out["dist_prior_session_high_ticks"] = dist_norm_const(prior_high)
    out["dist_prior_session_low_ticks"]  = dist_norm_const(prior_low)
    out["dist_prior_session_vpoc_ticks"] = dist_norm_const(prior_vpoc)
    if np.isfinite(h5) and np.isfinite(l5):
        d_h5 = np.abs(last_price_arr - h5)
        d_l5 = np.abs(last_price_arr - l5)
        nearer = np.where(d_h5 < d_l5, h5, l5)
        out["dist_5d_extreme_ticks"] = dist_norm(last_price_arr - nearer)
    else:
        out["dist_5d_extreme_ticks"] = np.zeros(n_seconds, dtype=np.float32)

    # Path memory (5) — bounded returns + vol/t-stat
    def lagged_log_return_bps(lag_seconds: int) -> np.ndarray:
        out_ = np.zeros(n_seconds, dtype=np.float64)
        if lag_seconds < n_seconds:
            ratio = last_price_arr[lag_seconds:] / np.maximum(last_price_arr[:-lag_seconds], 1e-9)
            out_[lag_seconds:] = np.log(np.maximum(ratio, 1e-9)) * 10000.0
        return np.clip(out_, -1000, 1000).astype(np.float32)
    out["log_return_60s_bps"]  = lagged_log_return_bps(60)
    out["log_return_5min_bps"] = lagged_log_return_bps(300)
    out["log_return_15min_bps"] = lagged_log_return_bps(900)

    s1_returns = np.zeros(n_seconds, dtype=np.float64)
    s1_returns[1:] = np.log(np.maximum(last_price_arr[1:] / np.maximum(last_price_arr[:-1], 1e-9), 1e-9))
    rv_5min = pd.Series(s1_returns).rolling(300, min_periods=10).std().fillna(0).values
    rv_5min_ticks = rv_5min * last_price_arr / TICK_SIZE
    out["realized_vol_5min_ticks"] = np.clip(rv_5min_ticks, 0, 50).astype(np.float32)
    trend_raw = np.zeros(n_seconds, dtype=np.float64)
    if 300 < n_seconds:
        trend_raw[300:] = last_price_arr[300:] - last_price_arr[:-300]
    vol_pts = rv_5min_ticks * TICK_SIZE
    trend_t = trend_raw / np.maximum(vol_pts, 1e-3)
    out["trend_strength_5min"] = np.clip(trend_t, -10.0, 10.0).astype(np.float32)

    # Regime / time (5) — cyclical + binary stored RAW (HC #298 fix: PASS in dataloader)
    seconds_in_rth = sec_idx.astype(np.float64)
    out["tod_sin"] = np.sin(2 * np.pi * seconds_in_rth / RTH_SECONDS_PER_DAY).astype(np.float32)
    out["tod_cos"] = np.cos(2 * np.pi * seconds_in_rth / RTH_SECONDS_PER_DAY).astype(np.float32)
    out["is_lunch_lull"]  = ((seconds_in_rth >= 7200) & (seconds_in_rth < 14400)).astype(np.float32)
    out["is_close_hour"]  = (seconds_in_rth >= 21600).astype(np.float32)
    dow = d.weekday()
    out["dow_sin"] = np.full(n_seconds, np.sin(2 * np.pi * dow / 5.0), dtype=np.float32)

    # ===== NEW v3.2.1: LGBM vol predictions (2) =====
    pred_5min, pred_30min = _predict_lgbm_vol_per_second(d, n_seconds)
    out["lgbm_vol_pred_5min"]  = pred_5min.astype(np.float32)
    out["lgbm_vol_pred_30min"] = pred_30min.astype(np.float32)

    # ===== NEW v3.2.1: Session-phase one-hot (4) =====
    # Bins: open [0,3600), morning [3600,9000), lunch [9000,14400), afternoon [14400,inf)
    out["phase_open"]      = ((seconds_in_rth >= 0)     & (seconds_in_rth < 3600)).astype(np.float32)
    out["phase_morning"]   = ((seconds_in_rth >= 3600)  & (seconds_in_rth < 9000)).astype(np.float32)
    out["phase_lunch"]     = ((seconds_in_rth >= 9000)  & (seconds_in_rth < 14400)).astype(np.float32)
    out["phase_afternoon"] = (seconds_in_rth >= 14400).astype(np.float32)

    # Sanity check: phase one-hot sums to exactly 1 every row
    phase_sum = (out["phase_open"] + out["phase_morning"]
                 + out["phase_lunch"] + out["phase_afternoon"]).values
    if not np.allclose(phase_sum, 1.0):
        n_bad = int((phase_sum != 1.0).sum())
        logger.warning(f"[{d}] phase one-hot sum != 1 for {n_bad} rows (should be 0)")

    return out[["second_idx"] + TIER3_FEATURE_COLS]


# ==============================================================================
# Per-day processing
# ==============================================================================

def process_day(d: date, force: bool = False) -> bool:
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

    logger.info(f"[{d}] building Tier 2 (19 features incl. L2 book reconstruction)...")
    t2 = build_tier2_for_day_v321(df, d)
    t2_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t2, preserve_index=False), t2_path, compression="zstd")
    logger.info(f"[{d}] Tier 2 wrote {len(t2):,} buckets × {len(TIER2_FEATURE_COLS)} feats")

    logger.info(f"[{d}] loading prior session chain...")
    prior_chain = load_prior_session_state_chain(d, max_lookback_days=14)
    logger.info(f"[{d}] {len(prior_chain)} prior sessions available")

    logger.info(f"[{d}] building Tier 3 (31 features incl. LGBM vol + session phase)...")
    t3 = build_tier3_for_day_v321(df, d, prior_chain)
    t3_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t3, preserve_index=False), t3_path, compression="zstd")
    logger.info(f"[{d}] Tier 3 wrote {len(t3):,} snapshots × {len(TIER3_FEATURE_COLS)} feats")

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
    p.add_argument("--date", type=str)
    p.add_argument("--start-date", type=str)
    p.add_argument("--end-date", type=str)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    global logger
    tag = args.date or f"{args.start_date}_{args.end_date}"
    logger = setup_logging(tag.replace("-", ""))

    logger.info("=" * 70)
    logger.info("CNN-Mamba v3.2.1 Tier 2 + Tier 3 Feature Builder (Priority-1)")
    logger.info(f"Output Tier 2: {TIER2_OUT_ROOT}  ({len(TIER2_FEATURE_COLS)} feats)")
    logger.info(f"Output Tier 3: {TIER3_OUT_ROOT}  ({len(TIER3_FEATURE_COLS)} feats)")
    logger.info(f"Prior-session state: {PRIOR_SESSION_STATE_DIR} (shared w/ v3.2)")
    logger.info(f"LGBM vol pkl source: {LGBM_VOL_DIR}")
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
