"""
Queue-Position-Aware Fill Model — ES Futures
=============================================

CONTEXT: Our hybrid execution simulator showed 61% of limit order exits fill
within 5s, but this was based on the simplistic assumption that "mid price
crosses our limit price = fill". In reality, ES futures use CME FIFO matching
and a limit order must wait in queue before filling.

CRITICAL FINDING (Feb 17, 2026):
  Actual MBO data analysis shows the DISPLAYED queue at best bid is
  only 1-3 contracts (median), NOT 1,500-3,000 as widely cited. The
  visible depth is low because 76% of orders are iceberg (display qty=1).
  The EFFECTIVE queue (volume traded before price moves) is ~20 contracts,
  but CME iceberg refills get NEW timestamps (BACK of FIFO queue), so
  our order jumps ahead of refills. Realistic queue position: 2-5 ahead.

THIS STUDY:
  Part 1 — Queue Depth Analysis
    - Load snapshot caches (NPZ) or raw .dbn MBO data
    - Extract inside queue depth (best bid/ask size in contracts)
    - Statistics: mean, median, p25, p75, p99
    - Variation by time-of-day and volatility

  Part 2 — Realistic Fill Probability Model
    - When we post at best bid: we join at BACK of queue
    - Fill probability depends on queue ahead + fill rate (contracts/sec)
    - Use MBO trade events to estimate actual fill rate at inside level
    - P(fill within T sec) given queue position Q ahead of us

  Part 3 — Queue-Adjusted PnL Estimation
    - Apply realistic fill rates to existing corrected-limit-study results
    - Scenarios: pessimistic (back of queue), middle, optimistic (front 25%)
    - Key question: does the strategy survive?

  Part 4 — Sensitivity Analysis
    - What queue position do we NEED to be profitable?
    - Break-even analysis

CONSTANTS (ES futures, CME FIFO):
  Tick size: 0.25 ($12.50/tick)
  Point value: $50
  Commission: $3.00 RT (0.24 ticks, AMP+Rithmic+CME)
  Half tick: 0.125

Usage:
    python alpha_discovery/run_queue_position_study.py
    python alpha_discovery/run_queue_position_study.py --fast
    python alpha_discovery/run_queue_position_study.py --max-files 3
"""

import sys, gc, json, time, logging, argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR

# ============================================================================
# LOGGING (ASCII-only for Windows cp1252)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / 'queue_position_study.log'),
                            mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger('queue_position_study')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE       = 0.25      # ES minimum price increment
TICK_VALUE      = 12.50     # Dollar value per tick
ES_POINT_VALUE  = 50        # Dollar value per point
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)      # Round-trip commission in dollars (AMP+Rithmic+CME fees)
HALF_TICK       = 0.125     # bid = mid - 0.125, ask = mid + 0.125
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks ($3.00 RT)
BARS_PER_SEC    = 10        # 100ms bars

# Node feature layout in NPZ (9 features per node):
# [0] price, [1] rel_price, [2] size (qty), [3] log_size,
# [4] level_idx, [5] side_flag, [6] order_count, [7] avg_order_size,
# [8] level_concentration
# First 10 nodes: bid levels (0=best bid), next 10 nodes: ask levels (0=best ask)
NODE_IDX_SIZE       = 2  # size (contracts) at level
NODE_IDX_ORDER_COUNT = 6  # number of distinct orders at level

# Global feature layout (45 total):
# [0] mid, [1] spread, [2] imbalance, [3] microprice, [4] total_bid_vol,
# [5] total_ask_vol, [6] mean_bid_size, [7] mean_ask_size, [8] best_bid, [9] best_ask
# Order flow [10-17], Microstructure [18-25], Temporal [26-30], MBO Enhanced [31-44]
# Temporal: [26] hour_norm, [27] minute_norm, [28] time_since_rth, [29] time_to_close, [30] event_density
# MBO Enhanced [31]: modify_count, [32] modify_to_add, [33] mean_lifetime, [34] fleeting_ratio,
#                    [35] aggr_buy_count, [36] aggr_sell_count, [37] aggr_imbalance, [38] max_trade_size,
#                    [39] cancel_bid_vol, [40] cancel_ask_vol, [41] cancel_side_imbalance,
#                    [42] tick_count, [43] sequence_gaps, [44] n_orders_completed
GLOBAL_COL_MID           = 0
GLOBAL_COL_SPREAD        = 1
GLOBAL_COL_TOTAL_BID_VOL = 4
GLOBAL_COL_TOTAL_ASK_VOL = 5
GLOBAL_COL_HOUR_NORM     = 26
GLOBAL_COL_TIME_SINCE_RTH = 28
GLOBAL_COL_TIME_TO_CLOSE = 29
GLOBAL_COL_BUY_VOL       = 11   # buy_volume in order flow section
GLOBAL_COL_SELL_VOL      = 12   # sell_volume in order flow section
GLOBAL_COL_TRADE_COUNT   = 15   # trade_count in order flow section
GLOBAL_COL_AGGR_BUY      = 35
GLOBAL_COL_AGGR_SELL      = 36
GLOBAL_COL_MAX_TRADE_SIZE = 38


# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================
def to_safe(obj):
    """Convert numpy types and NaN/Inf to JSON-serializable form."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [to_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ============================================================================
# DATA LOADING
# ============================================================================
def load_snapshot_data(max_files: Optional[int] = None) -> Optional[dict]:
    """
    Load snapshot data from NPZ cache or raw MBO files.

    Returns dict with:
      node_features: (N, 20, 9) - 10 bid + 10 ask levels, 9 features each
      global_features: (N, 45)
      mid_prices: (N,)
      day_boundaries: list of ints
      n_days: int
      source: 'cache' or 'raw'
    """
    CACHE_DIR = ROOT / "data" / "processed" / "medium_snapshots_cache"
    cache_files = sorted(CACHE_DIR.glob("file_*_snapshots.npz"))

    if cache_files:
        log.info(f"Loading from NPZ cache: {len(cache_files)} files in {CACHE_DIR}")
        return _load_from_cache(cache_files, max_files)
    else:
        log.info("No NPZ cache found — loading from raw MBO .dbn files")
        return _load_from_raw_mbo(max_files)


def _load_from_cache(cache_files: List[Path], max_files: Optional[int]) -> dict:
    """Load from pre-computed NPZ snapshot caches."""
    if max_files:
        cache_files = cache_files[:max_files]

    all_node = []
    all_global = []
    all_mid = []
    day_boundaries = [0]

    GLOBAL_COL_TIME_SINCE_RTH_LOCAL = 28

    for fi, fpath in enumerate(cache_files):
        log.info(f"  Loading cache {fi+1}/{len(cache_files)}: {fpath.name}")
        data = np.load(str(fpath), allow_pickle=True)
        gf = data['global_features']   # (N, 45)
        nf = data['node_features']     # (N, 20, 9) - 10 bid + 10 ask, 9 features each
        mp = data['mid_prices']        # (N,)
        data.close()

        # Detect day boundaries via time_since_rth drops
        if gf.shape[1] > GLOBAL_COL_TIME_SINCE_RTH_LOCAL:
            rth_col = gf[:, GLOBAL_COL_TIME_SINCE_RTH_LOCAL]
            diffs = np.diff(rth_col)
            transitions = np.where(diffs < -0.3)[0]
            day_starts = np.concatenate([[0], transitions + 1, [len(gf)]])
        else:
            day_starts = np.array([0, len(gf)])

        n_file_days = len(day_starts) - 1

        for d in range(n_file_days):
            ds, de = day_starts[d], day_starts[d + 1]

            # RTH filter
            if gf.shape[1] > GLOBAL_COL_TIME_SINCE_RTH_LOCAL:
                rth = gf[ds:de, GLOBAL_COL_TIME_SINCE_RTH_LOCAL]
                rth_mask = (rth >= 0.0) & (rth <= 1.0)
                gf_day = gf[ds:de][rth_mask]
                nf_day = nf[ds:de][rth_mask]
                mp_day = mp[ds:de][rth_mask]
            else:
                gf_day = gf[ds:de]
                nf_day = nf[ds:de]
                mp_day = mp[ds:de]

            if len(mp_day) < 100:
                continue

            # Skip FLAT days (weekends where book is static)
            price_range = mp_day.max() - mp_day.min()
            if price_range < 1.0:
                log.info(f"  File {fi} Day {d}: FLAT (range={price_range:.2f}pts), skipping")
                continue

            all_global.append(gf_day)
            all_node.append(nf_day)
            all_mid.append(mp_day)
            day_boundaries.append(day_boundaries[-1] + len(mp_day))

    if not all_mid:
        return None

    node_arr   = np.concatenate(all_node)
    global_arr = np.concatenate(all_global)
    mid_arr    = np.concatenate(all_mid)
    n_days     = len(day_boundaries) - 1

    log.info(f"Loaded from cache: {len(mid_arr):,} snapshots, {n_days} days")
    log.info(f"  node_features shape: {node_arr.shape}")
    log.info(f"  global_features shape: {global_arr.shape}")

    return {
        'node_features': node_arr,
        'global_features': global_arr,
        'mid_prices': mid_arr,
        'day_boundaries': np.array(day_boundaries),
        'n_days': n_days,
        'source': 'cache',
    }


def _load_from_raw_mbo(max_files: Optional[int]) -> dict:
    """
    Load from raw MBO .dbn files and reconstruct order book to get queue depths.

    This is the "slow but always works" path. It uses the OrderBook class
    and compute_features to generate the same data as the cache builder.
    """
    try:
        from src.data.ingest import iter_files, load_mbo_events, get_es_instrument_id
        from src.data.lob import OrderBook
        from src.features.engineering import compute_features
    except ImportError as e:
        log.error(f"Cannot import required modules: {e}")
        return None

    MBO_DIR = ROOT / "mbo"
    # INSTRUMENT_ID now determined per-file via get_es_instrument_id()
    SAMPLE_INTERVAL_MS = 100
    DEPTH_LEVELS = 10
    ET_OFFSET_HOURS = -4

    all_files = list(iter_files(str(MBO_DIR)))
    # Deduplicate by date (prefer .dbn over .dbn.zst)
    seen_dates = set()
    files = []
    for f in sorted(all_files, key=lambda p: (p.name.endswith('.zst'), p.name)):
        date_part = f.name.split('.')[0]
        if date_part not in seen_dates:
            seen_dates.add(date_part)
            files.append(f)

    if max_files:
        files = files[:max_files]

    log.info(f"Processing {len(files)} raw MBO files...")

    def is_within_rth(ts_ns: int) -> bool:
        if ts_ns == 0:
            return False
        try:
            dt_utc = datetime.utcfromtimestamp(ts_ns / 1e9)
            dt_et = dt_utc + timedelta(hours=ET_OFFSET_HOURS)
            tm = dt_et.hour * 60 + dt_et.minute
            return (9 * 60 + 30) <= tm < (16 * 60)
        except Exception:
            return False

    def to_ns(v) -> int:
        if hasattr(v, "value"):
            return int(v.value)
        return int(v) if v else 0

    all_node, all_global, all_mid = [], [], []
    day_boundaries = [0]

    for fi, fpath in enumerate(files):
        log.info(f"  [{fi+1}/{len(files)}] {fpath.name}")
        t0 = time.time()

        book = OrderBook(DEPTH_LEVELS)
        file_node, file_global, file_mid = [], [], []

        try:
            df = load_mbo_events(fpath, max_events=None, filter_instrument_id=INSTRUMENT_ID)
            if "ts_event" not in df.columns:
                log.warning(f"    No ts_event column, skipping")
                continue

            df = df.sort_values("ts_event")
            interval_ns = SAMPLE_INTERVAL_MS * 1_000_000
            next_snap_ts = None
            rth_count = 0

            for _, row in df.iterrows():
                ts = to_ns(row["ts_event"])
                if next_snap_ts is None:
                    next_snap_ts = ts
                event = row.to_dict()
                book.update(event)

                while ts >= next_snap_ts:
                    if is_within_rth(next_snap_ts):
                        snap = book.snapshot()
                        snap.timestamp_ns = next_snap_ts
                        if snap.mid == snap.mid:  # not NaN
                            nf, gf = compute_features(
                                snap, DEPTH_LEVELS,
                                include_order_flow=True,
                                include_microstructure=True,
                                include_temporal=True,
                            )
                            file_node.append(nf)
                            file_global.append(gf)
                            file_mid.append(snap.mid)
                            rth_count += 1
                    next_snap_ts += interval_ns

            del df
            gc.collect()

        except Exception as e:
            log.error(f"    Error: {e}")
            continue

        if not file_mid:
            log.warning(f"    No valid snapshots from {fpath.name}")
            continue

        file_node_arr   = np.array(file_node, dtype=np.float32)
        file_global_arr = np.array(file_global, dtype=np.float32)
        file_mid_arr    = np.array(file_mid, dtype=np.float32)

        all_node.append(file_node_arr)
        all_global.append(file_global_arr)
        all_mid.append(file_mid_arr)
        day_boundaries.append(day_boundaries[-1] + len(file_mid))

        elapsed = time.time() - t0
        log.info(f"    {rth_count:,} RTH snapshots [{elapsed:.0f}s]")

    if not all_mid:
        return None

    node_arr   = np.concatenate(all_node)
    global_arr = np.concatenate(all_global)
    mid_arr    = np.concatenate(all_mid)
    n_days     = len(day_boundaries) - 1

    log.info(f"Loaded from raw MBO: {len(mid_arr):,} snapshots, {n_days} days")
    return {
        'node_features': node_arr,
        'global_features': global_arr,
        'mid_prices': mid_arr,
        'day_boundaries': np.array(day_boundaries),
        'n_days': n_days,
        'source': 'raw_mbo',
    }


# ============================================================================
# PART 1: QUEUE DEPTH ANALYSIS
# ============================================================================
def analyze_queue_depth(data: dict) -> dict:
    """
    Analyze queue depth (contracts) at best bid and best ask across all snapshots.

    From node_features:
      - Bid levels: indices 0-9 (level 0 = best bid)
      - Ask levels: indices 10-19 (level 0 = best ask)
      - Feature index 2 = size (contracts at level)
      - Feature index 6 = order count (distinct orders)

    Returns stats on queue depth distribution, intraday variation, volatility impact.
    """
    log.info("=" * 60)
    log.info("PART 1: QUEUE DEPTH ANALYSIS")
    log.info("=" * 60)

    node_feat   = data['node_features']    # (N, 20, 9)
    global_feat = data['global_features']  # (N, 45)
    mid_prices  = data['mid_prices']       # (N,)
    day_boundaries = data['day_boundaries']
    N = len(mid_prices)

    # Extract best bid/ask sizes (contracts at level 0)
    # Bid side: node index 0 (best bid), size = feature index 2
    # Ask side: node index 10 (best ask), size = feature index 2
    has_node_data = node_feat.shape[1] >= 20 and node_feat.shape[2] >= 7

    if has_node_data:
        best_bid_size = node_feat[:, 0, NODE_IDX_SIZE].astype(float)      # (N,) contracts at best bid
        best_ask_size = node_feat[:, 10, NODE_IDX_SIZE].astype(float)     # (N,) contracts at best ask
        best_bid_orders = node_feat[:, 0, NODE_IDX_ORDER_COUNT].astype(float)  # (N,) orders at best bid
        best_ask_orders = node_feat[:, 10, NODE_IDX_ORDER_COUNT].astype(float) # (N,) orders at best ask
        log.info(f"  Loaded queue data from node_features: shape={node_feat.shape}")
    else:
        # Fallback: use global total_bid_vol / total_ask_vol divided by level count
        log.warning("  node_features too small, using global vol proxies")
        best_bid_size = global_feat[:, GLOBAL_COL_TOTAL_BID_VOL] / 10.0
        best_ask_size = global_feat[:, GLOBAL_COL_TOTAL_ASK_VOL] / 10.0
        best_bid_orders = np.zeros(N)
        best_ask_orders = np.zeros(N)

    # Filter out zeros and NaNs
    bid_valid = np.isfinite(best_bid_size) & (best_bid_size > 0)
    ask_valid = np.isfinite(best_ask_size) & (best_ask_size > 0)
    combined_valid = bid_valid & ask_valid

    bid_depths = best_bid_size[bid_valid]
    ask_depths = best_ask_size[ask_valid]
    inside_depth_avg = (best_bid_size[combined_valid] + best_ask_size[combined_valid]) / 2

    log.info(f"  Valid snapshots: {combined_valid.sum():,} / {N:,} "
             f"({100*combined_valid.mean():.1f}%)")

    # Overall statistics
    def pctile_stats(arr):
        if len(arr) == 0:
            return {}
        return {
            'mean':   float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std':    float(np.std(arr)),
            'p10':    float(np.percentile(arr, 10)),
            'p25':    float(np.percentile(arr, 25)),
            'p75':    float(np.percentile(arr, 75)),
            'p90':    float(np.percentile(arr, 90)),
            'p99':    float(np.percentile(arr, 99)),
            'min':    float(np.min(arr)),
            'max':    float(np.max(arr)),
            'n':      int(len(arr)),
        }

    bid_stats = pctile_stats(bid_depths)
    ask_stats = pctile_stats(ask_depths)
    inside_stats = pctile_stats(inside_depth_avg)

    log.info(f"\n  BEST BID QUEUE DEPTH (contracts):")
    log.info(f"    mean={bid_stats['mean']:.1f}  median={bid_stats['median']:.1f}  "
             f"p25={bid_stats['p25']:.0f}  p75={bid_stats['p75']:.0f}  p99={bid_stats['p99']:.0f}")
    log.info(f"  BEST ASK QUEUE DEPTH (contracts):")
    log.info(f"    mean={ask_stats['mean']:.1f}  median={ask_stats['median']:.1f}  "
             f"p25={ask_stats['p25']:.0f}  p75={ask_stats['p75']:.0f}  p99={ask_stats['p99']:.0f}")

    # ---- Intraday variation (by hour) ----
    log.info("\n  INTRADAY VARIATION (by hour ET):")
    has_time_col = global_feat.shape[1] > GLOBAL_COL_HOUR_NORM

    intraday = {}
    if has_time_col:
        hour_norm = global_feat[:, GLOBAL_COL_HOUR_NORM]  # 0-1 over trading day
        # Convert to approximate ET hour (9:30 = 0 -> 16:00 = 1, roughly 6.5 hours)
        # hour_norm 0.0 = 9:30 ET, 1.0 = 16:00 ET
        hour_et_approx = 9.5 + hour_norm * 6.5  # approximate

        hour_bins = [(9.5, 10.5, 'Open (9:30-10:30)'),
                     (10.5, 12.0, 'Morning (10:30-12:00)'),
                     (12.0, 13.5, 'Midday (12:00-13:30)'),
                     (13.5, 15.0, 'Afternoon (13:30-15:00)'),
                     (15.0, 16.0, 'Close (15:00-16:00)')]

        for h_lo, h_hi, label in hour_bins:
            mask = combined_valid & (hour_et_approx >= h_lo) & (hour_et_approx < h_hi)
            if mask.sum() < 10:
                continue
            avg_inside = (best_bid_size[mask] + best_ask_size[mask]) / 2
            s = pctile_stats(avg_inside)
            intraday[label] = s
            log.info(f"    {label:30s}: mean={s['mean']:.1f}  median={s['median']:.1f}  "
                     f"p75={s['p75']:.0f}")
    else:
        log.warning("  No time-of-day column available for intraday breakdown")

    # ---- Volatility-conditioned queue depth ----
    log.info("\n  VOLATILITY-CONDITIONED QUEUE DEPTH:")
    vol_bins_stats = {}
    n_days_vol = data['n_days']
    day_bounds = data['day_boundaries']

    # Compute 100-bar rolling return volatility as proxy
    mid_valid = np.isfinite(mid_prices)
    if mid_valid.sum() > 200:
        # Use rolling std of returns
        returns = np.zeros(N)
        for d in range(n_days_vol):
            ds, de = day_bounds[d], day_bounds[d + 1]
            if de - ds < 20:
                continue
            m = mid_prices[ds:de]
            # 1-bar returns
            r = np.diff(m, prepend=m[0])
            # 20-bar rolling std
            vol = np.zeros(len(r))
            for i in range(len(r)):
                lo = max(0, i - 20)
                vol[i] = np.std(r[lo:i+1]) if i > 0 else 0.0
            returns[ds:de] = vol

        vol_p33 = float(np.percentile(returns[returns > 0], 33))
        vol_p67 = float(np.percentile(returns[returns > 0], 67))

        for label, lo_v, hi_v in [
            ('Low vol (bottom 33%)', 0, vol_p33),
            ('Medium vol (33-67%)', vol_p33, vol_p67),
            ('High vol (top 33%)', vol_p67, 1e9),
        ]:
            mask = combined_valid & (returns >= lo_v) & (returns < hi_v)
            if mask.sum() < 10:
                continue
            avg_inside = (best_bid_size[mask] + best_ask_size[mask]) / 2
            s = pctile_stats(avg_inside)
            vol_bins_stats[label] = s
            log.info(f"    {label}: mean={s['mean']:.1f}  median={s['median']:.1f}  "
                     f"p75={s['p75']:.0f}")

    # ---- Order count stats (how many distinct orders at inside) ----
    log.info("\n  ORDERS AT BEST BID/ASK (distinct orders, not contracts):")
    bid_oc_valid = best_bid_orders[bid_valid & (best_bid_orders > 0)]
    ask_oc_valid = best_ask_orders[ask_valid & (best_ask_orders > 0)]
    if len(bid_oc_valid) > 0:
        log.info(f"    Bid order count: mean={np.mean(bid_oc_valid):.1f}  "
                 f"median={np.median(bid_oc_valid):.1f}  p75={np.percentile(bid_oc_valid, 75):.0f}")
    if len(ask_oc_valid) > 0:
        log.info(f"    Ask order count: mean={np.mean(ask_oc_valid):.1f}  "
                 f"median={np.median(ask_oc_valid):.1f}  p75={np.percentile(ask_oc_valid, 75):.0f}")

    return {
        'bid_depth_stats': bid_stats,
        'ask_depth_stats': ask_stats,
        'inside_depth_avg_stats': inside_stats,
        'intraday_by_session': intraday,
        'volatility_conditioned': vol_bins_stats,
        'bid_order_count_stats': pctile_stats(bid_oc_valid) if len(bid_oc_valid) > 0 else {},
        'ask_order_count_stats': pctile_stats(ask_oc_valid) if len(ask_oc_valid) > 0 else {},
        'n_valid_snapshots': int(combined_valid.sum()),
        'n_total_snapshots': int(N),
    }


# ============================================================================
# PART 2: REALISTIC FILL PROBABILITY MODEL
# ============================================================================
def analyze_fill_probability_queue(data: dict) -> dict:
    """
    Estimate P(fill within T seconds) given queue position.

    CME ES is FIFO: if we join at position Q in the queue, we need
    at least Q contracts to trade at that price level before we fill.

    Key estimates from MBO data:
    a) Queue depth at inside when signal fires
    b) Trade rate at inside level (contracts/second)
    c) P(fill | queue_ahead=Q, time=T) using Poisson approximation

    Poisson model: trades arrive at rate lambda (contracts/sec).
    P(fill within T) = P(Poisson(lambda*T) >= Q) = 1 - CDF(Q-1; lambda*T)

    For a simpler conservative model:
    P(fill within T) ~= P(lambda*T >= Q) -> depends on variability too.
    We use empirical: estimate lambda from MBO trade volume data.
    """
    log.info("\n" + "=" * 60)
    log.info("PART 2: REALISTIC FILL PROBABILITY MODEL")
    log.info("=" * 60)

    node_feat   = data['node_features']
    global_feat = data['global_features']
    day_bounds  = data['day_boundaries']
    mid_prices  = data['mid_prices']
    N = len(mid_prices)
    n_days = data['n_days']

    # Extract trade flow from global features
    # Global features [11] = buy_volume, [12] = sell_volume per 100ms interval
    # These are total volume traded (aggressor buy + sell)
    has_trade_data = global_feat.shape[1] > max(GLOBAL_COL_BUY_VOL, GLOBAL_COL_SELL_VOL,
                                                  GLOBAL_COL_TRADE_COUNT)

    if has_trade_data:
        buy_vol_per_bar  = global_feat[:, GLOBAL_COL_BUY_VOL]    # contracts bought per 100ms
        sell_vol_per_bar = global_feat[:, GLOBAL_COL_SELL_VOL]   # contracts sold per 100ms
        total_trade_per_bar = buy_vol_per_bar + sell_vol_per_bar
    else:
        log.warning("  Trade volume columns not available, using estimates")
        # ES typical: ~8-12 trades/second, avg size 4-6 contracts
        # At inside bid: roughly half of trades (sells) drain the bid queue
        # ~5 sells/sec * 5 contracts = 25 contracts/sec at bid (rough)
        total_trade_per_bar = np.full(N, 3.0)  # 3 contracts per 100ms = 30/sec conservative

    # Convert to per-second rates (multiply per-bar by BARS_PER_SEC)
    trade_rate_per_sec = total_trade_per_bar * BARS_PER_SEC  # contracts/second

    # Filter to valid (positive) trade rates
    valid_trade = np.isfinite(trade_rate_per_sec) & (trade_rate_per_sec > 0)
    trade_rates_valid = trade_rate_per_sec[valid_trade]

    log.info(f"\n  TRADE RATE AT INSIDE LEVEL (contracts/second from MBO):")
    log.info(f"    mean={np.mean(trade_rates_valid):.2f}  "
             f"median={np.median(trade_rates_valid):.2f}  "
             f"p25={np.percentile(trade_rates_valid, 25):.2f}  "
             f"p75={np.percentile(trade_rates_valid, 75):.2f}  "
             f"p99={np.percentile(trade_rates_valid, 99):.2f}")

    # Key insight: at the inside bid/ask, only a FRACTION of all trades actually
    # drain the queue there. Aggressive sellers hit the bid; buyers hit the ask.
    # For a buy limit at best bid: only aggressive SELL orders drain our queue.
    # Rough assumption: ~50% of volume at inside is on either side
    # This is actually ~50% of trades at the inside level (bid gets ~50% of flow)
    inside_fill_rate_per_sec = np.mean(trade_rates_valid) * 0.5  # conservative: 50% at our side

    # Also compute p25 (pessimistic) and p75 (optimistic) fill rates
    fill_rate_pessimistic = float(np.percentile(trade_rates_valid, 25)) * 0.5
    fill_rate_median      = float(np.median(trade_rates_valid)) * 0.5
    fill_rate_optimistic  = float(np.percentile(trade_rates_valid, 75)) * 0.5

    log.info(f"\n  ESTIMATED FILL RATES (contracts/sec at our queue position):")
    log.info(f"    Pessimistic (p25): {fill_rate_pessimistic:.2f} c/s")
    log.info(f"    Median:            {fill_rate_median:.2f} c/s")
    log.info(f"    Optimistic (p75):  {fill_rate_optimistic:.2f} c/s")

    # ---- Poisson fill probability model ----
    from scipy.stats import poisson

    def p_fill(queue_ahead: float, rate_per_sec: float, time_sec: float) -> float:
        """
        P(fill within time_sec seconds) given queue_ahead contracts ahead.

        Model: trades arrive as Poisson process with rate=rate_per_sec.
        We fill when cumulative trades >= queue_ahead.
        P(fill in T) = P(Poisson(rate*T) >= queue_ahead)
        """
        if queue_ahead <= 0:
            return 1.0
        if rate_per_sec <= 0:
            return 0.0
        lam = rate_per_sec * time_sec
        return float(1 - poisson.cdf(int(queue_ahead) - 1, lam))

    # ---- Compute P(fill) for various queue positions and time horizons ----
    log.info("\n  FILL PROBABILITY TABLE (Poisson FIFO model):")
    log.info(f"  {'Queue Ahead':>12s}  {'3s':>8s}  {'5s':>8s}  {'10s':>8s}  {'30s':>8s}  "
             f"[median fill rate = {fill_rate_median:.1f} c/s]")

    time_horizons = [3, 5, 10, 30, 60]
    queue_positions = [1, 5, 10, 25, 50, 100, 200, 500]

    fill_prob_table = {}
    for q in queue_positions:
        row = {}
        probs = []
        for t in time_horizons:
            p = p_fill(q, fill_rate_median, t)
            row[f'{t}s'] = p
            probs.append(p)
        fill_prob_table[q] = row
        if q <= 200:
            log.info(f"  Q={q:>4d} contracts ahead: " +
                     "  ".join(f"{p:.1%}" for p in probs[:4]))

    # ---- Estimate queue position when we post ----
    # When our signal fires and we post a limit order at best bid:
    # We join at the BACK of the DISPLAYED queue = current best bid size
    # KEY INSIGHT: This is the DISPLAYED depth (2-3 contracts typically),
    # NOT the effective depth (~20 contracts). The difference is icebergs:
    # 76% of orders show size=1 but are actually icebergs with hidden size.
    # When icebergs refill, CME gives them a NEW timestamp = BACK of FIFO.
    # So our order JUMPS AHEAD of iceberg refills! Our effective position
    # is just the displayed orders ahead of us (very favorable).
    has_bid_data = node_feat.shape[1] >= 20 and node_feat.shape[2] >= 3
    if has_bid_data:
        best_bid_sizes = node_feat[:, 0, NODE_IDX_SIZE].astype(float)
        best_ask_sizes = node_feat[:, 10, NODE_IDX_SIZE].astype(float)
        valid_sizes = np.isfinite(best_bid_sizes) & (best_bid_sizes > 0) & \
                      np.isfinite(best_ask_sizes) & (best_ask_sizes > 0)
        queue_at_entry_bid = best_bid_sizes[valid_sizes]
        queue_at_entry_ask = best_ask_sizes[valid_sizes]
    else:
        # Fallback: use typical ES DISPLAYED queue depth (mostly icebergs showing 1-3)
        queue_at_entry_bid = np.full(1000, 3.0)  # typical ES displayed inside queue
        queue_at_entry_ask = np.full(1000, 3.0)

    log.info(f"\n  QUEUE AHEAD WHEN WE POST (best bid size at signal time):")
    log.info(f"    mean={np.mean(queue_at_entry_bid):.1f}  "
             f"median={np.median(queue_at_entry_bid):.1f}  "
             f"p25={np.percentile(queue_at_entry_bid, 25):.0f}  "
             f"p75={np.percentile(queue_at_entry_bid, 75):.0f}")

    # ---- Realistic fill probabilities given ACTUAL queue depths ----
    log.info("\n  REALISTIC FILL PROBABILITY (using observed queue depths):")

    def compute_realistic_fill_probs(queue_arr, rate, time_horizons, percentile_label):
        """Given array of queue positions, compute P(fill | that queue) for each horizon."""
        results = {}
        for t in time_horizons:
            fill_probs = np.array([p_fill(q, rate, t) for q in queue_arr])
            results[f'{t}s'] = {
                'mean_p_fill': float(np.mean(fill_probs)),
                'median_p_fill': float(np.median(fill_probs)),
                'p25_p_fill': float(np.percentile(fill_probs, 25)),
                'p75_p_fill': float(np.percentile(fill_probs, 75)),
            }
        return results

    # Use p25 (pessimistic queue = join 25% position), median, p75
    queue_p25 = float(np.percentile(queue_at_entry_bid, 75))  # 75th pctile queue = harder to fill
    queue_p50 = float(np.median(queue_at_entry_bid))
    queue_p75 = float(np.percentile(queue_at_entry_bid, 25))  # smaller queue = easier

    scenarios = {
        'pessimistic_back_of_queue': {
            'queue_ahead': queue_p25,
            'fill_rate': fill_rate_pessimistic,
            'description': 'Always at back of queue, slow fill rate (p75 queue, p25 fill rate)'
        },
        'realistic_median': {
            'queue_ahead': queue_p50,
            'fill_rate': fill_rate_median,
            'description': 'Median queue position, median fill rate'
        },
        'optimistic_front_25pct': {
            'queue_ahead': queue_p75,
            'fill_rate': fill_rate_optimistic,
            'description': 'Front 25% of queue (queue-joining strategy), fast fill rate'
        },
    }

    scenario_results = {}
    for scenario_name, scenario in scenarios.items():
        q = scenario['queue_ahead']
        r = scenario['fill_rate']
        log.info(f"\n  Scenario: {scenario_name} (Q={q:.0f} ahead, rate={r:.1f} c/s)")
        scenario_probs = {}
        for t in time_horizons:
            p = p_fill(q, r, t)
            scenario_probs[f'{t}s'] = p
            log.info(f"    T={t}s: P(fill)={p:.1%}")
        scenario_results[scenario_name] = {
            'description': scenario['description'],
            'queue_ahead': q,
            'fill_rate_per_sec': r,
            'fill_probs': scenario_probs,
        }

    return {
        'trade_rate_stats': {
            'mean': float(np.mean(trade_rates_valid)),
            'median': float(np.median(trade_rates_valid)),
            'p25': float(np.percentile(trade_rates_valid, 25)),
            'p75': float(np.percentile(trade_rates_valid, 75)),
            'p99': float(np.percentile(trade_rates_valid, 99)),
        },
        'fill_rate_estimates': {
            'pessimistic': fill_rate_pessimistic,
            'median': fill_rate_median,
            'optimistic': fill_rate_optimistic,
        },
        'queue_at_entry_stats': {
            'mean': float(np.mean(queue_at_entry_bid)),
            'median': float(np.median(queue_at_entry_bid)),
            'p25': float(np.percentile(queue_at_entry_bid, 25)),
            'p75': float(np.percentile(queue_at_entry_bid, 75)),
            'p99': float(np.percentile(queue_at_entry_bid, 99)),
        },
        'fill_prob_table_median_rate': fill_prob_table,
        'scenario_results': scenario_results,
        'time_horizons_sec': time_horizons,
    }


# ============================================================================
# PART 3: QUEUE-ADJUSTED PnL ESTIMATION
# ============================================================================
def analyze_queue_adjusted_pnl(fill_prob_result: dict) -> dict:
    """
    Apply realistic fill probabilities to PnL estimates.

    From the corrected limit study (run_corrected_limit_study.py):
    - Fill rate at best bid/ask: ~6.5% in 3s, ~10.8% in 10s (MID-crossing assumption)
    - These were based on mid price touching the limit = fill (too optimistic!)
    - Actual fill requires the ENTIRE QUEUE AHEAD to drain first

    PnL breakdown:
    - Limit entry (passive): +0.5 tick = $6.25
    - Limit exit  (passive): +0.5 tick = $6.25 (if fills)
    - Market exit (crosses): -0.5 tick = -$6.25
    - Commission: -0.24 ticks = -$3.00
    - Directional PnL: variable (signal IC=0.1135 on 3s returns)

    Best case (both limit entry + limit exit): +1.0 tick - 0.24t commission = +$9.50
    Market exit (limit entry + market exit): 0.0 edge - 0.24t commission = -$3.00
    Need directional PnL > $3.00 just to break even on mkt exit

    From empirical data (corrected limit study):
    - Mean directional PnL at 10s hold, top 20% signal: ~+0.05 to +0.15 ticks
    - That's ~$0.63 to $1.88 — less than commission cost of $2.50!
    - Limit exit adds +0.5t = $6.25, making it viable IF fill probability is high

    KEY QUESTION: Given realistic fill probabilities, what is expected PnL?

    Expected PnL = P(fill_entry) * [dir_pnl + entry_edge + (P(fill_exit) * exit_edge
                                    + (1-P(fill_exit)) * (-exit_cost)) - commission]
    """
    log.info("\n" + "=" * 60)
    log.info("PART 3: QUEUE-ADJUSTED PnL ESTIMATION")
    log.info("=" * 60)

    scenario_results = fill_prob_result['scenario_results']
    time_horizons = fill_prob_result['time_horizons_sec']

    # Prior study results (from corrected limit study + hybrid execution sim)
    # These are the IDEALIZED (mid-crossing = fill) results we need to adjust
    # Based on IC=0.1135, top 20% signals, 10s hold, corrected 1-tick spread
    PRIOR_DIR_PNL_TICKS = 0.08   # mean directional PnL per trade (conservative, top 20%)
    PRIOR_DIR_STD_TICKS = 2.0    # std of directional PnL
    ENTRY_EDGE_TICKS    = 0.5    # passive limit entry earns half spread
    EXIT_EDGE_TICKS     = 0.5    # passive limit exit earns half spread
    EXIT_COST_TICKS     = 0.5    # market exit costs half spread
    COMMISSION_TICKS_   = COMMISSION_TICKS  # 0.24 ticks = $3.00 (AMP+CME)

    # Old (naive) entry fill probability: mid touches bid within 5s
    OLD_ENTRY_FILL_RATE_5S = 0.065  # ~6.5% from corrected_limit_study
    OLD_ENTRY_FILL_RATE_10S = 0.108
    OLD_EXIT_LIMIT_FILL = 0.61      # from hybrid_execution_sim (61% limit exits in 10s)

    log.info("\n  PRIOR (NAIVE) RESULTS:")
    log.info(f"    Entry fill rate (5s, mid-cross): {OLD_ENTRY_FILL_RATE_5S:.1%}")
    log.info(f"    Entry fill rate (10s, mid-cross): {OLD_ENTRY_FILL_RATE_10S:.1%}")
    log.info(f"    Exit limit fill rate (10s hold): {OLD_EXIT_LIMIT_FILL:.1%}")
    log.info(f"    Directional PnL (mean): {PRIOR_DIR_PNL_TICKS:+.4f}t = ${PRIOR_DIR_PNL_TICKS*TICK_VALUE:+.2f}")

    # Naive expected PnL per trade attempt:
    # = P(entry_fill) * [dir_pnl + entry_edge + P(exit_lmt)*exit_edge
    #                    + (1-P(exit_lmt))*(-exit_cost) - commission]
    naive_exit_edge_blend = (OLD_EXIT_LIMIT_FILL * EXIT_EDGE_TICKS
                             + (1 - OLD_EXIT_LIMIT_FILL) * (-EXIT_COST_TICKS))
    naive_pnl_given_fill = (PRIOR_DIR_PNL_TICKS + ENTRY_EDGE_TICKS
                             + naive_exit_edge_blend - COMMISSION_TICKS_)
    naive_pnl_per_attempt_5s = OLD_ENTRY_FILL_RATE_5S * naive_pnl_given_fill
    naive_pnl_per_attempt_10s = OLD_ENTRY_FILL_RATE_10S * naive_pnl_given_fill

    log.info(f"\n    Naive PnL given entry fill: {naive_pnl_given_fill:+.4f}t = ${naive_pnl_given_fill*TICK_VALUE:+.2f}")
    log.info(f"    Naive PnL per attempt (5s):  {naive_pnl_per_attempt_5s:+.4f}t = ${naive_pnl_per_attempt_5s*TICK_VALUE:+.2f}")
    log.info(f"    Naive PnL per attempt (10s): {naive_pnl_per_attempt_10s:+.4f}t = ${naive_pnl_per_attempt_10s*TICK_VALUE:+.2f}")

    log.info("\n  QUEUE-ADJUSTED RESULTS BY SCENARIO:")

    pnl_results = {}

    for scenario_name, scenario in scenario_results.items():
        desc = scenario['description']
        fill_probs = scenario['fill_probs']

        # For each time horizon, compute adjusted PnL
        for t in [3, 5, 10, 30]:
            t_key = f'{t}s'
            if t_key not in fill_probs:
                continue

            # ENTRY fill probability (queue-adjusted)
            p_entry = fill_probs[t_key]

            # EXIT fill probability: for the exit limit order, we also join a queue
            # The exit is at the opposing side. For a long position, exit = limit sell at ask.
            # The ask queue is similar to bid. Use same fill probability but for shorter horizon
            # (we only want to wait hold_horizon - fill_time for exit)
            # Conservative: use same P(fill) for exit
            p_exit_lmt_t5 = fill_probs.get('5s', 0.0)  # 5s for exit (within 10s hold)

            # Exit blend: P(lmt fill) * exit_edge + P(mkt) * (-exit_cost)
            exit_edge_blend = (p_exit_lmt_t5 * EXIT_EDGE_TICKS
                               + (1 - p_exit_lmt_t5) * (-EXIT_COST_TICKS))

            # PnL given entry fill
            pnl_given_fill = (PRIOR_DIR_PNL_TICKS + ENTRY_EDGE_TICKS
                              + exit_edge_blend - COMMISSION_TICKS_)

            # Expected PnL per attempt (entry fill probability * PnL given fill)
            expected_pnl_per_attempt = p_entry * pnl_given_fill

            if scenario_name not in pnl_results:
                pnl_results[scenario_name] = {
                    'description': desc,
                    'by_horizon': {}
                }

            pnl_results[scenario_name]['by_horizon'][t_key] = {
                'p_entry_fill': p_entry,
                'p_exit_limit_fill': p_exit_lmt_t5,
                'exit_edge_blend_ticks': exit_edge_blend,
                'pnl_given_fill_ticks': pnl_given_fill,
                'pnl_given_fill_dollars': pnl_given_fill * TICK_VALUE,
                'expected_pnl_per_attempt_ticks': expected_pnl_per_attempt,
                'expected_pnl_per_attempt_dollars': expected_pnl_per_attempt * TICK_VALUE,
                'vs_naive_pnl_per_attempt_dollars': (expected_pnl_per_attempt - naive_pnl_per_attempt_5s) * TICK_VALUE,
            }

            log.info(f"\n  [{scenario_name}] t={t}s:")
            log.info(f"    P(entry fill):          {p_entry:.2%}")
            log.info(f"    P(exit limit fill, 5s): {p_exit_lmt_t5:.2%}")
            log.info(f"    PnL given fill:         {pnl_given_fill:+.4f}t = ${pnl_given_fill*TICK_VALUE:+.2f}")
            log.info(f"    Expected PnL/attempt:   {expected_pnl_per_attempt:+.4f}t = ${expected_pnl_per_attempt*TICK_VALUE:+.2f}")

    # Comparison table
    log.info("\n  COMPARISON SUMMARY (at 5s fill horizon):")
    log.info(f"  {'Scenario':<35s} {'P(fill)':>8s} {'E[PnL]/attempt':>16s} {'vs naive':>12s}")
    log.info(f"  {'Naive (mid-cross)':35s} {OLD_ENTRY_FILL_RATE_5S:>8.2%} "
             f"${naive_pnl_per_attempt_5s*TICK_VALUE:>+14.2f}  {'baseline':>12s}")
    for scenario_name, r in pnl_results.items():
        hz = r['by_horizon'].get('5s', {})
        if not hz:
            continue
        p = hz.get('p_entry_fill', 0)
        e = hz.get('expected_pnl_per_attempt_dollars', 0)
        v = hz.get('vs_naive_pnl_per_attempt_dollars', 0)
        log.info(f"  {scenario_name[:35]:35s} {p:>8.2%} ${e:>+14.2f}  ${v:>+10.2f}")

    return {
        'naive_benchmark': {
            'entry_fill_rate_5s': OLD_ENTRY_FILL_RATE_5S,
            'entry_fill_rate_10s': OLD_ENTRY_FILL_RATE_10S,
            'exit_limit_fill_rate': OLD_EXIT_LIMIT_FILL,
            'pnl_given_fill_ticks': naive_pnl_given_fill,
            'pnl_per_attempt_5s_ticks': naive_pnl_per_attempt_5s,
            'pnl_per_attempt_5s_dollars': naive_pnl_per_attempt_5s * TICK_VALUE,
        },
        'queue_adjusted_scenarios': pnl_results,
        'assumptions': {
            'dir_pnl_ticks': PRIOR_DIR_PNL_TICKS,
            'dir_pnl_std_ticks': PRIOR_DIR_STD_TICKS,
            'entry_edge_ticks': ENTRY_EDGE_TICKS,
            'exit_edge_ticks': EXIT_EDGE_TICKS,
            'exit_cost_ticks': EXIT_COST_TICKS,
            'commission_ticks': COMMISSION_TICKS_,
        },
    }


# ============================================================================
# PART 4: SENSITIVITY ANALYSIS
# ============================================================================
def analyze_sensitivity(fill_prob_result: dict) -> dict:
    """
    Sensitivity analysis on queue position.

    Key question: What queue position do we NEED to be profitable?
    Break-even analysis: find the queue position where expected PnL = 0.

    Also: how does PnL scale with queue position improvement?
    If we could implement a smart queue-joining strategy (arriving before signals
    by anticipating order flow), how much could we improve?
    """
    log.info("\n" + "=" * 60)
    log.info("PART 4: SENSITIVITY ANALYSIS")
    log.info("=" * 60)

    from scipy.stats import poisson
    from scipy.optimize import brentq

    fill_rate_median = fill_prob_result['fill_rate_estimates']['median']
    fill_rate_pess   = fill_prob_result['fill_rate_estimates']['pessimistic']
    fill_rate_opt    = fill_prob_result['fill_rate_estimates']['optimistic']

    PRIOR_DIR_PNL_TICKS = 0.08
    ENTRY_EDGE_TICKS    = 0.5
    EXIT_EDGE_TICKS     = 0.5
    EXIT_COST_TICKS     = 0.5
    COMMISSION_TICKS_   = COMMISSION_TICKS  # 0.24 ticks = $3.00
    T_FILL_SEC = 5.0     # 5-second entry fill window
    T_EXIT_SEC = 5.0     # 5-second exit window (within 10s hold)

    def p_fill_q(queue_ahead, rate, t):
        if queue_ahead <= 0:
            return 1.0
        if rate <= 0:
            return 0.0
        lam = rate * t
        return float(1 - poisson.cdf(int(queue_ahead) - 1, lam))

    def expected_pnl_per_attempt(queue_ahead, fill_rate, t_entry=T_FILL_SEC, t_exit=T_EXIT_SEC,
                                  dir_pnl=PRIOR_DIR_PNL_TICKS):
        """Compute expected PnL per trading attempt given queue position."""
        p_entry = p_fill_q(queue_ahead, fill_rate, t_entry)
        p_exit_lmt = p_fill_q(queue_ahead, fill_rate, t_exit)  # exit queue similar to entry
        exit_blend = p_exit_lmt * EXIT_EDGE_TICKS + (1 - p_exit_lmt) * (-EXIT_COST_TICKS)
        pnl_given_fill = dir_pnl + ENTRY_EDGE_TICKS + exit_blend - COMMISSION_TICKS_
        return p_entry * pnl_given_fill

    # ---- Break-even queue position (where expected PnL = 0) ----
    log.info("\n  BREAK-EVEN QUEUE POSITION ANALYSIS:")
    log.info("  (How many contracts ahead of us can we tolerate and still be profitable?)")

    breakeven_results = {}
    for rate_label, rate in [('pessimistic', fill_rate_pess),
                               ('median', fill_rate_median),
                               ('optimistic', fill_rate_opt)]:
        # Find queue_ahead where expected_pnl_per_attempt = 0
        # Try binary search from 1 to 2000
        try:
            # If even Q=1 is negative, we're broken
            pnl_q1 = expected_pnl_per_attempt(1, rate)
            if pnl_q1 <= 0:
                breakeven_q = 0
                log.info(f"    {rate_label}: Even Q=1 is negative! "
                          f"PnL(Q=1) = {pnl_q1:+.4f}t = ${pnl_q1*TICK_VALUE:+.2f}")
            else:
                # Find breakeven
                f_lo = lambda q: expected_pnl_per_attempt(int(q), rate)
                # Find range where sign changes
                hi = 10
                while hi < 2000 and f_lo(hi) > 0:
                    hi *= 2
                if f_lo(hi) > 0:
                    breakeven_q = hi
                    log.info(f"    {rate_label}: Always profitable up to Q={hi} contracts!")
                else:
                    breakeven_q = int(brentq(f_lo, 1, hi, xtol=1.0))
                    log.info(f"    {rate_label}: Break-even at Q = {breakeven_q} contracts ahead  "
                              f"(rate={rate:.1f} c/s)")
            breakeven_results[rate_label] = breakeven_q
        except Exception as e:
            log.warning(f"    {rate_label}: Could not compute break-even: {e}")
            breakeven_results[rate_label] = None

    # ---- PnL vs Queue Position Grid ----
    log.info("\n  PnL vs QUEUE POSITION (expected PnL per attempt, $):")
    log.info(f"  {'Queue':>8s}  {'Pessimistic':>12s}  {'Median':>12s}  {'Optimistic':>12s}")

    queue_grid = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500]
    pnl_grid = {}

    for q in queue_grid:
        row = {}
        vals = []
        for rate_label, rate in [('pessimistic', fill_rate_pess),
                                   ('median', fill_rate_median),
                                   ('optimistic', fill_rate_opt)]:
            pnl = expected_pnl_per_attempt(q, rate)
            row[rate_label] = {
                'ticks': pnl,
                'dollars': pnl * TICK_VALUE,
                'p_entry': p_fill_q(q, rate, T_FILL_SEC),
            }
            vals.append(pnl * TICK_VALUE)
        pnl_grid[q] = row
        if q <= 300:
            log.info(f"  Q={q:>4d}: " +
                     "  ".join(f"${v:>+10.4f}" for v in vals))

    # ---- Directional PnL sensitivity ----
    log.info("\n  DIRECTIONAL EDGE REQUIRED for profitability (at median queue):")
    log.info("  (What IC do we need at current queue depth?)")

    q_typical = fill_prob_result['queue_at_entry_stats']['median']
    p_entry_typical = p_fill_q(q_typical, fill_rate_median, T_FILL_SEC)
    p_exit_typical  = p_fill_q(q_typical, fill_rate_median, T_EXIT_SEC)
    exit_blend_typical = (p_exit_typical * EXIT_EDGE_TICKS
                           + (1 - p_exit_typical) * (-EXIT_COST_TICKS))

    # Break-even directional PnL: entry_edge + exit_blend - commission = -dir_pnl
    breakeven_dir = -(ENTRY_EDGE_TICKS + exit_blend_typical - COMMISSION_TICKS_)
    breakeven_dir_dollars = breakeven_dir * TICK_VALUE

    log.info(f"    Typical queue: {q_typical:.0f} contracts ahead")
    log.info(f"    P(entry fill, 5s): {p_entry_typical:.2%}")
    log.info(f"    P(exit limit, 5s): {p_exit_typical:.2%}")
    log.info(f"    Exit edge blend: {exit_blend_typical:+.4f}t")
    log.info(f"    Break-even directional PnL: {breakeven_dir:+.4f}t = ${breakeven_dir_dollars:+.2f}")
    if breakeven_dir > 0:
        log.info(f"    => We need ADVERSE directional move of {breakeven_dir:+.4f}t")
        log.info(f"       The strategy CAN be profitable even with mean-reverting entry!")
    elif breakeven_dir < 0:
        log.info(f"    => We need directional PnL > {-breakeven_dir:.4f}t = ${-breakeven_dir_dollars:.2f}")
        log.info(f"       Current IC=0.1135 => mean dir PnL ~{PRIOR_DIR_PNL_TICKS:.4f}t")
        if PRIOR_DIR_PNL_TICKS > -breakeven_dir:
            log.info(f"       Strategy IS viable (dir_pnl > breakeven)")
        else:
            log.info(f"       Strategy is NOT viable at typical queue (need better IC or queue position)")
    else:
        log.info(f"    => Zero directional PnL needed (spread capture alone is sufficient)")

    # ---- Improvement needed ----
    log.info("\n  REQUIRED QUEUE POSITION IMPROVEMENT:")
    q_typical_int = max(1, int(q_typical))
    pnl_at_typical = expected_pnl_per_attempt(q_typical_int, fill_rate_median)
    log.info(f"    At typical queue (Q={q_typical_int}): E[PnL] = {pnl_at_typical:+.4f}t = ${pnl_at_typical*TICK_VALUE:+.2f}/attempt")

    q_targets = [1, 5, 10, 25, 50]
    log.info(f"    Target queue positions and their improvement:")
    for q_target in q_targets:
        if q_target >= q_typical_int:
            continue
        pnl_target = expected_pnl_per_attempt(q_target, fill_rate_median)
        improvement = (pnl_target - pnl_at_typical) * TICK_VALUE
        log.info(f"      Q={q_target:>3d}: E[PnL]={pnl_target:+.4f}t  improvement=${improvement:+.2f}")

    # ---- Summary verdict ----
    beq_median = breakeven_results.get('median', None)
    verdict = ""
    if beq_median is None or beq_median == 0:
        verdict = "BROKEN — strategy cannot be profitable at any queue position given current directional edge"
    elif beq_median < 10:
        verdict = f"VERY CHALLENGING — only profitable if we're in top {beq_median} contracts of queue"
    elif beq_median < 50:
        verdict = f"CHALLENGING — need to be in first {beq_median} contracts of queue (requires queue pre-positioning)"
    elif beq_median < 200:
        verdict = f"FEASIBLE — can be profitable up to Q={beq_median} contracts ahead (reasonable fill rate needed)"
    else:
        verdict = f"VIABLE — profitable even at Q={beq_median} contracts back (ample margin)"

    log.info(f"\n  VERDICT: {verdict}")

    return {
        'breakeven_queue_positions': breakeven_results,
        'pnl_vs_queue_grid': {str(q): v for q, v in pnl_grid.items()},
        'typical_queue_analysis': {
            'q_typical': float(q_typical),
            'p_entry_fill_5s': float(p_entry_typical),
            'p_exit_limit_5s': float(p_exit_typical),
            'exit_edge_blend_ticks': float(exit_blend_typical),
            'breakeven_dir_pnl_ticks': float(breakeven_dir),
            'breakeven_dir_pnl_dollars': float(breakeven_dir_dollars),
            'pnl_at_typical_queue_ticks': float(pnl_at_typical),
            'pnl_at_typical_queue_dollars': float(pnl_at_typical * TICK_VALUE),
        },
        'verdict': verdict,
    }


# ============================================================================
# SUMMARY FORMATTER
# ============================================================================
def format_summary(queue_depth: dict, fill_prob: dict,
                   pnl_adj: dict, sensitivity: dict) -> str:
    lines = [
        "", "=" * 70,
        "QUEUE-POSITION-AWARE FILL MODEL -- ES FUTURES",
        "CME FIFO: fill requires draining entire queue ahead of us",
        f"Tick: ${TICK_VALUE:.2f}  Commission: ${COMMISSION_RT:.2f} RT  Half-tick: ${HALF_TICK:.3f}",
        "=" * 70, "",
    ]

    # Part 1
    bd = queue_depth.get('bid_depth_stats', {})
    ad = queue_depth.get('ask_depth_stats', {})
    lines += [
        "PART 1 -- QUEUE DEPTH AT INSIDE (contracts):",
        f"  Best Bid: mean={bd.get('mean', 0):.0f}  median={bd.get('median', 0):.0f}  "
        f"p25={bd.get('p25', 0):.0f}  p75={bd.get('p75', 0):.0f}  p99={bd.get('p99', 0):.0f}",
        f"  Best Ask: mean={ad.get('mean', 0):.0f}  median={ad.get('median', 0):.0f}  "
        f"p25={ad.get('p25', 0):.0f}  p75={ad.get('p75', 0):.0f}  p99={ad.get('p99', 0):.0f}",
    ]

    intraday = queue_depth.get('intraday_by_session', {})
    if intraday:
        lines.append("  Intraday queue depth (avg inside):")
        for session, s in intraday.items():
            lines.append(f"    {session:35s}: mean={s.get('mean', 0):.0f}  median={s.get('median', 0):.0f}")

    # Part 2
    trade_stats = fill_prob.get('trade_rate_stats', {})
    fill_rates  = fill_prob.get('fill_rate_estimates', {})
    q_stats     = fill_prob.get('queue_at_entry_stats', {})
    lines += [
        "",
        "PART 2 -- REALISTIC FILL PROBABILITY MODEL:",
        f"  Trade rate at inside: mean={trade_stats.get('mean', 0):.1f}  "
        f"median={trade_stats.get('median', 0):.1f}  c/s",
        f"  Fill rate at our side: pessimistic={fill_rates.get('pessimistic', 0):.1f}  "
        f"median={fill_rates.get('median', 0):.1f}  optimistic={fill_rates.get('optimistic', 0):.1f} c/s",
        f"  Queue ahead when signal fires: mean={q_stats.get('mean', 0):.0f}  "
        f"median={q_stats.get('median', 0):.0f}  p75={q_stats.get('p75', 0):.0f}",
        "",
        "  Fill probabilities (Poisson FIFO model) by scenario:",
        f"  {'Scenario':<35s} {'3s':>6s}  {'5s':>6s}  {'10s':>6s}  {'30s':>6s}",
    ]
    for scenario_name, scenario in fill_prob.get('scenario_results', {}).items():
        probs = scenario.get('fill_probs', {})
        q = scenario.get('queue_ahead', 0)
        rate = scenario.get('fill_rate_per_sec', 0)
        lines.append(
            f"  {scenario_name[:35]:35s} "
            f"{probs.get('3s', 0):>5.1%}  {probs.get('5s', 0):>5.1%}  "
            f"{probs.get('10s', 0):>5.1%}  {probs.get('30s', 0):>5.1%}"
            f"  (Q={q:.0f}, r={rate:.1f}c/s)"
        )

    # Part 3
    naive = pnl_adj.get('naive_benchmark', {})
    lines += [
        "",
        "PART 3 -- QUEUE-ADJUSTED PnL vs NAIVE BENCHMARK:",
        f"  Naive (mid-cross): P(fill,5s)={naive.get('entry_fill_rate_5s', 0):.1%}  "
        f"E[PnL]=${naive.get('pnl_per_attempt_5s_dollars', 0):+.4f}/attempt",
        "",
        f"  {'Scenario':<35s} {'P(fill,5s)':>10s}  {'E[PnL]/attempt':>15s}",
    ]
    for scenario_name, r in pnl_adj.get('queue_adjusted_scenarios', {}).items():
        hz = r.get('by_horizon', {}).get('5s', {})
        if hz:
            p = hz.get('p_entry_fill', 0)
            e = hz.get('expected_pnl_per_attempt_dollars', 0)
            lines.append(f"  {scenario_name[:35]:35s} {p:>10.2%}  ${e:>+13.4f}")

    # Part 4
    beq = sensitivity.get('breakeven_queue_positions', {})
    typ = sensitivity.get('typical_queue_analysis', {})
    lines += [
        "",
        "PART 4 -- SENSITIVITY ANALYSIS:",
        f"  Break-even queue positions (max contracts ahead to be profitable):",
        f"    Pessimistic fill rate: Q <= {beq.get('pessimistic', 'N/A')}",
        f"    Median fill rate:      Q <= {beq.get('median', 'N/A')}",
        f"    Optimistic fill rate:  Q <= {beq.get('optimistic', 'N/A')}",
        "",
        f"  At typical queue (Q={typ.get('q_typical', 0):.0f} contracts ahead):",
        f"    P(entry fill, 5s):     {typ.get('p_entry_fill_5s', 0):.2%}",
        f"    P(exit limit, 5s):     {typ.get('p_exit_limit_5s', 0):.2%}",
        f"    Break-even dir PnL:    {typ.get('breakeven_dir_pnl_ticks', 0):+.4f}t = ${typ.get('breakeven_dir_pnl_dollars', 0):+.2f}",
        f"    E[PnL] at typical Q:   {typ.get('pnl_at_typical_queue_ticks', 0):+.4f}t = ${typ.get('pnl_at_typical_queue_dollars', 0):+.2f}/attempt",
        "",
        "VERDICT:",
        f"  {sensitivity.get('verdict', 'N/A')}",
        "",
        "KEY INSIGHTS (from MBO data analysis, Feb 17, 2026):",
        "  1. DISPLAYED queue at inside: median 2-3 contracts (NOT 1,500+ as widely cited)",
        "  2. EFFECTIVE queue (volume before price moves): ~20 contracts (icebergs!)",
        "  3. 76% of orders are icebergs (size=1 displayed). Iceberg refills go to BACK of FIFO.",
        "  4. Our order jumps ahead of iceberg refills — effective position = displayed queue only",
        "  5. Fill probability at position 3-5: 80-87% per price sweep (empirical from MBO)",
        "  6. Price level persists: median 842ms, mean 2.0s — ample time for fills",
        "  7. The queue is NOT the bottleneck. Signal strength (IC=0.11) IS the bottleneck.",
        "  8. Naive model (mid-cross = fill) is CLOSER to reality than Poisson model suggests",
        "=" * 70,
    ]

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Queue-Position-Aware Fill Model for ES Futures'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Load fewer files for quick testing')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of data files to process')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("QUEUE POSITION STUDY -- ES FUTURES MBO ALPHA")
    log.info("=" * 70)
    t_start = time.time()

    max_files = args.max_files
    if args.fast and max_files is None:
        max_files = 3  # Quick test with 3 files

    # ---- Load data ----
    log.info("Loading snapshot data...")
    data = load_snapshot_data(max_files=max_files)

    if data is None:
        log.error("Failed to load any data. Exiting.")
        return

    N = len(data['mid_prices'])
    n_days = data['n_days']
    log.info(f"Loaded: {N:,} snapshots, {n_days} days, source={data['source']}")
    log.info(f"  Mid price mean: ${np.nanmean(data['mid_prices']):.2f}")

    # ---- Part 1: Queue Depth Analysis ----
    queue_depth = analyze_queue_depth(data)
    gc.collect()

    # ---- Part 2: Fill Probability Model ----
    fill_prob = analyze_fill_probability_queue(data)
    gc.collect()

    # ---- Part 3: Queue-Adjusted PnL ----
    pnl_adj = analyze_queue_adjusted_pnl(fill_prob)
    gc.collect()

    # ---- Part 4: Sensitivity Analysis ----
    sensitivity = analyze_sensitivity(fill_prob)
    gc.collect()

    # ---- Format Summary ----
    summary_text = format_summary(queue_depth, fill_prob, pnl_adj, sensitivity)
    log.info("\n" + summary_text)

    # ---- Save Results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'queue_position_study_{timestamp}.json'

    output = {
        'timestamp': timestamp,
        'mode': 'fast' if args.fast else 'full',
        'max_files': max_files,
        'data_source': data['source'],
        'constants': {
            'tick_size': TICK_SIZE,
            'tick_value': TICK_VALUE,
            'es_point_value': ES_POINT_VALUE,
            'half_tick': HALF_TICK,
            'commission_rt': COMMISSION_RT,
            'commission_ticks': COMMISSION_TICKS,
            'bars_per_sec': BARS_PER_SEC,
        },
        'data_summary': {
            'n_snapshots': N,
            'n_days': n_days,
            'mid_price_mean': float(np.nanmean(data['mid_prices'])),
            'source': data['source'],
        },
        'part1_queue_depth': to_safe(queue_depth),
        'part2_fill_probability': to_safe(fill_prob),
        'part3_queue_adjusted_pnl': to_safe(pnl_adj),
        'part4_sensitivity': to_safe(sensitivity),
        'summary_text': summary_text,
        'total_elapsed_sec': time.time() - t_start,
    }

    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved: {out_file}")
    log.info(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    # Print final summary
    print("\n" + "=" * 70)
    print(summary_text)
    print(f"\nResults saved: {out_file}")

    # Print Discord-ready summary
    beq = sensitivity.get('breakeven_queue_positions', {})
    typ = sensitivity.get('typical_queue_analysis', {})
    q_stats = fill_prob.get('queue_at_entry_stats', {})
    fill_rates = fill_prob.get('fill_rate_estimates', {})
    scenarios = fill_prob.get('scenario_results', {})
    pess_5s = scenarios.get('pessimistic_back_of_queue', {}).get('fill_probs', {}).get('5s', 0)
    med_5s  = scenarios.get('realistic_median', {}).get('fill_probs', {}).get('5s', 0)
    opt_5s  = scenarios.get('optimistic_front_25pct', {}).get('fill_probs', {}).get('5s', 0)
    pnl_typ = typ.get('pnl_at_typical_queue_dollars', 0)

    discord_summary = (
        f"**Queue Position Study Complete**\n"
        f"Data: {N:,} snapshots, {n_days} days\n\n"
        f"**Part 1 — Queue Depth at Inside Bid/Ask:**\n"
        f"  Median: {queue_depth.get('bid_depth_stats', {}).get('median', 0):.0f} contracts  "
        f"p75: {queue_depth.get('bid_depth_stats', {}).get('p75', 0):.0f} contracts\n\n"
        f"**Part 2 — Fill Probability (Poisson FIFO, 5s window):**\n"
        f"  Pessimistic (back of queue): {pess_5s:.1%}\n"
        f"  Realistic (median):          {med_5s:.1%}\n"
        f"  Optimistic (front 25%):      {opt_5s:.1%}\n"
        f"  (vs naive mid-cross: 6.5%)\n\n"
        f"**Part 3 — E[PnL] at typical queue:**\n"
        f"  ${pnl_typ:+.4f}/attempt\n\n"
        f"**Part 4 — Break-even:**\n"
        f"  Max contracts ahead to be profitable:\n"
        f"  Pessimistic: Q <= {beq.get('pessimistic', 'N/A')}\n"
        f"  Median:      Q <= {beq.get('median', 'N/A')}\n"
        f"  Optimistic:  Q <= {beq.get('optimistic', 'N/A')}\n\n"
        f"**Verdict:** {sensitivity.get('verdict', 'N/A')}\n"
        f"Results: {out_file}"
    )
    print("\n" + discord_summary)

    return output


if __name__ == '__main__':
    main()
