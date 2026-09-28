#!/usr/bin/env python3
"""
mbo_trade_anatomy_v1.py — MBO-Level Trade Anatomy Analysis

PURPOSE: Use full MBO (market-by-order) data to understand WHY our champion
strategy's trades win or lose. The champion uses minute bars for entry/exit,
but we have tick-level order book data that reveals the microstructure context
around each trade.

RESEARCH QUESTIONS:
1. What does the order book look like at entry for winners vs losers?
   (book imbalance, recent sweeps, trade intensity, spread dynamics)
2. Can MBO features at entry predict which trades will win?
   (→ confluence filter to boost 23.3% WR)
3. What happens mid-trade at MBO level for winners vs losers?
   (→ early exit signals to cut losers before SL)
4. Can CNN-Mamba predictions add confluence value?
   (→ combine short-horizon AI signal with 30-min LightGBM)

DATA SOURCES:
- Raw MBO: /home/jupiter/Lvl3Quant/data/raw/mbo/*.dbn.zst (238 days)
- Processed events: /home/jupiter/Lvl3Quant/data/processed/mbo_events/*.npz
- Champion trades: reconstructed from multi_scale_combo_v1 output
- CNN-Mamba preds: /home/nick/Lvl3Quant/data/precomputed_obs/ (precomputed)

HC #0: SLIDING windows only.
HC #428: Regime-agnostic, MFE-within-horizon.
HC #433: Report plain English summaries.
"""

import os, sys, json, logging, warnings, time, gc
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

try:
    import databento as dbn
    HAS_DBN = True
except ImportError:
    HAS_DBN = False
    print("WARNING: databento not installed, will use processed events only")

warnings.filterwarnings('ignore')

# ── Paths ──
# Can run on either Jupiter or Neptune — auto-detect
if Path("/home/jupiter/Lvl3Quant").exists():
    ROOT = Path("/home/jupiter/Lvl3Quant")
else:
    ROOT = Path("/home/nick/Lvl3Quant")

RAW_MBO_DIR = ROOT / "data" / "raw" / "mbo"
PROCESSED_EVENTS_DIR = ROOT / "data" / "processed" / "mbo_events"
MINUTE_BARS_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
COMBO_OUTPUT = ROOT / "output" / "multi_scale_combo_v1"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
OUTPUT_DIR = ROOT / "output" / "mbo_trade_anatomy_v1"
LOG_FILE = ROOT / "logs" / "mbo_trade_anatomy_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MBO-ANATOMY] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 1.0
TICK_SIZE_PTS = 0.25
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

# Strategy params (champion config)
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD = 60
ENTRY_THRESHOLD = 0.05
CANCEL_WINDOW = 10
ENTRY_BAR_SIZE = 30
DAILY_BIAS_THRESHOLD_MULT = 1.5

# MBO feature extraction windows (in seconds)
MBO_LOOKBACK_WINDOWS = [5, 15, 30, 60, 120, 300]  # seconds before entry
MBO_LOOKAHEAD_WINDOWS = [5, 15, 30, 60]  # seconds after entry (mid-trade)


# ============================================================
# PART 1: RECONSTRUCT CHAMPION TRADES WITH TIMESTAMPS
# ============================================================

def load_champion_trades():
    """Reconstruct all 210 champion trades with exact fill timestamps."""
    log.info("=" * 70)
    log.info("PART 1: Reconstructing champion trades")
    log.info("=" * 70)

    # Load minute bars
    files = sorted(MINUTE_BARS_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df['date'] = f.stem
        frames.append(df)
    minute_df = pd.concat(frames, ignore_index=True)
    minute_df['ts_minute'] = pd.to_datetime(minute_df['ts_minute'], utc=True)
    minute_df = minute_df.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(minute_df):,} minute bars across {len(files)} days")

    # Build 30-min bars
    bars_30m = minute_df.groupby('date').apply(
        lambda g: g.iloc[::30] if len(g) >= 30 else g.iloc[:0]
    ).reset_index(drop=True)
    log.info(f"Built {len(bars_30m)} 30-min bars")

    # Load 30-min predictions
    data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    pred_30m = data['entry_preds']
    log.info(f"Loaded predictions: {len(pred_30m)} values, {np.sum(~np.isnan(pred_30m))} non-NaN")

    # Load daily features for bias filter
    daily_df = pd.read_parquet(FEATURES_PATH)
    daily_df['date'] = daily_df.index if isinstance(daily_df.index[0], str) else daily_df['date']
    if 'date' not in daily_df.columns:
        daily_df = daily_df.reset_index()
        daily_df.columns = ['date'] + list(daily_df.columns[1:])

    # Compute daily bias (from multi_scale_combo logic)
    if 'session_ofi' in daily_df.columns:
        ofi_col = daily_df['session_ofi']
        ofi_std = ofi_col.expanding(min_periods=20).std()
        ofi_mean = ofi_col.expanding(min_periods=20).mean()
        daily_df['ofi_z'] = (ofi_col - ofi_mean) / ofi_std.clip(lower=1e-6)
    log.info(f"Daily features: {len(daily_df)} days")

    # Reconstruct entry fills with FIFO logic
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - ENTRY_THRESHOLD)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], ENTRY_THRESHOLD)

    minute_lookup = {}
    for date_str, grp in minute_df.groupby('date'):
        minute_lookup[date_str] = grp.sort_values('ts_minute').reset_index(drop=True)

    bars_ts = bars_30m['ts_minute'].values if 'ts_minute' in bars_30m.columns else bars_30m['ts'].values
    bars_dates = bars_30m['date'].values
    bars_close = bars_30m['close'].values

    # Build daily bias lookup
    bias_threshold = DAILY_BIAS_THRESHOLD_MULT  # 1.5x std
    daily_bias_lookup = {}
    if 'ofi_z' in daily_df.columns:
        for _, row in daily_df.iterrows():
            d = str(row['date'])
            z = row['ofi_z']
            if pd.isna(z):
                daily_bias_lookup[d] = 'neutral'
            elif z > bias_threshold:
                daily_bias_lookup[d] = 'short'  # contrarian: high OFI -> short bias
            elif z < -bias_threshold:
                daily_bias_lookup[d] = 'long'   # contrarian: low OFI -> long bias
            else:
                daily_bias_lookup[d] = 'neutral'

    trades = []
    for i in range(len(bars_30m)):
        if i >= len(pred_30m) or np.isnan(pred_30m[i]):
            continue

        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        date_str = bars_dates[i]

        # Apply daily bias filter
        bias = daily_bias_lookup.get(str(date_str), 'neutral')
        if bias == 'short' and direction == 1:
            continue  # block longs on short-bias days
        if bias == 'long' and direction == -1:
            continue  # block shorts on long-bias days

        signal_ts = pd.Timestamp(bars_ts[i])
        signal_price = bars_close[i]

        if str(date_str) not in minute_lookup:
            continue

        day_minutes = minute_lookup[str(date_str)]
        day_ts = day_minutes['ts_minute'].values

        limit_price = signal_price
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        cancel_ts = signal_ts + pd.Timedelta(minutes=CANCEL_WINDOW + ENTRY_BAR_SIZE)

        fill_mask = (day_ts >= np.datetime64(signal_bar_end)) & (day_ts <= np.datetime64(cancel_ts))
        fill_candidates = day_minutes[fill_mask]

        if len(fill_candidates) == 0:
            continue

        filled = False
        fill_price = None
        fill_ts = None

        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    break

        if not filled:
            continue

        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]

        if len(remaining_minutes) < 2:
            continue

        # Simulate exit
        prices_close = remaining_minutes['close'].values
        prices_high = remaining_minutes['high'].values
        prices_low = remaining_minutes['low'].values

        tp_ticks = TP_LONG if direction == 1 else TP_SHORT
        sl_ticks = SL_LONG if direction == 1 else SL_SHORT
        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        exit_type = 'time'
        exit_pnl = 0.0
        exit_minute = 0

        max_check = min(MAX_HOLD, len(remaining_minutes))
        for m in range(1, max_check):
            bar_high = prices_high[m]
            bar_low = prices_low[m]
            sl_hit = tp_hit = False

            if direction == 1:
                if bar_low <= sl_price:
                    sl_hit = True
                if bar_high >= tp_price + TICK_SIZE:
                    tp_hit = True
            else:
                if bar_high >= sl_price:
                    sl_hit = True
                if bar_low <= tp_price - TICK_SIZE:
                    tp_hit = True

            if sl_hit and tp_hit:
                sl_hit = True
                tp_hit = False

            if sl_hit:
                exit_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                exit_type = 'sl'
                exit_minute = m
                break
            if tp_hit:
                exit_pnl = tp_ticks - RT_COMMISSION_TICKS
                exit_type = 'tp'
                exit_minute = m
                break

        if exit_type == 'time':
            exit_minute = min(MAX_HOLD, len(remaining_minutes) - 1)
            exit_minute = max(exit_minute, 1)
            exit_close = prices_close[exit_minute]
            exit_fill = exit_close - TICK_SIZE if direction == 1 else exit_close + TICK_SIZE
            exit_pnl = (exit_fill - fill_price) / TICK_SIZE * direction - RT_COMMISSION_TICKS

        trades.append({
            'idx': i,
            'date': str(date_str),
            'signal_ts': signal_ts,
            'fill_ts': pd.Timestamp(fill_ts),
            'fill_price': fill_price,
            'direction': direction,
            'pred_30m': float(pred_30m[i]),
            'exit_type': exit_type,
            'exit_pnl': exit_pnl,
            'exit_minute': exit_minute,
            'tp_ticks': tp_ticks,
            'sl_ticks': sl_ticks,
            'daily_bias': bias,
            'winner': exit_type == 'tp',
        })

    log.info(f"Reconstructed {len(trades)} champion trades")
    winners = sum(1 for t in trades if t['winner'])
    losers = len(trades) - winners
    log.info(f"  Winners: {winners} ({winners/len(trades)*100:.1f}%)")
    log.info(f"  Losers: {losers} ({losers/len(trades)*100:.1f}%)")
    return trades


# ============================================================
# PART 2: EXTRACT MBO FEATURES AROUND EACH TRADE
# ============================================================

def load_mbo_events_for_date(date_str):
    """Load processed MBO events for a given date."""
    # Try processed events first
    npz_path = PROCESSED_EVENTS_DIR / f"{date_str}.npz"
    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        events = data['events']
        timestamps = data['timestamps']
        metadata = data['metadata'].item() if 'metadata' in data else {}
        return events, timestamps, metadata

    # Try alternate naming
    for pattern in [f"*{date_str}*", f"*{date_str.replace('-', '')}*"]:
        matches = list(PROCESSED_EVENTS_DIR.glob(pattern))
        if matches:
            data = np.load(str(matches[0]), allow_pickle=True)
            return data['events'], data['timestamps'], data.get('metadata', {})

    return None, None, None


def extract_mbo_features_at_time(events, timestamps, target_ts_ns, lookback_secs,
                                 direction=None):
    """
    Extract MBO microstructure features around a specific timestamp.

    Features extracted:
    - Book imbalance (bid vs ask order flow)
    - Trade intensity (trades per second)
    - Sweep detection (aggressive large orders hitting multiple levels)
    - Spread dynamics (mean, std, widening/narrowing trend)
    - Order cancellation rate (cancel-to-add ratio)
    - Volume-weighted price pressure
    - Large order presence (institutional activity proxy)
    """
    # Event type encoding: A=0, C=1, M=2, T=3, F=4
    # Side encoding: B=0, A=1, N=2
    # Event columns: [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]

    window_start_ns = target_ts_ns - int(lookback_secs * 1e9)
    mask = (timestamps >= window_start_ns) & (timestamps < target_ts_ns)
    window_events = events[mask]

    n_events = len(window_events)
    features = {}
    prefix = f"mbo_{lookback_secs}s"

    if n_events < 5:
        # Not enough data — return NaN features
        for k in _get_mbo_feature_names(prefix):
            features[k] = np.nan
        return features

    event_types = window_events[:, 1].astype(int)
    sides = window_events[:, 2].astype(int)
    price_rel = window_events[:, 3]
    qty_log = window_events[:, 4]
    spread = window_events[:, 5]

    # Decode quantities from log
    qty = np.expm1(qty_log)

    # 1. BOOK IMBALANCE: bid-side vs ask-side activity
    bid_mask = sides == 0
    ask_mask = sides == 1
    add_mask = event_types == 0
    cancel_mask = event_types == 1
    trade_mask = (event_types == 3) | (event_types == 4)

    bid_add_qty = qty[bid_mask & add_mask].sum()
    ask_add_qty = qty[ask_mask & add_mask].sum()
    bid_cancel_qty = qty[bid_mask & cancel_mask].sum()
    ask_cancel_qty = qty[ask_mask & cancel_mask].sum()

    # Net book pressure: positive = more bid support (bullish)
    net_add = (bid_add_qty - ask_add_qty)
    total_add = bid_add_qty + ask_add_qty + 1e-8
    features[f'{prefix}_book_imbalance'] = net_add / total_add

    # Cancel imbalance: more bid cancels = weakening support (bearish)
    net_cancel = (bid_cancel_qty - ask_cancel_qty)
    total_cancel = bid_cancel_qty + ask_cancel_qty + 1e-8
    features[f'{prefix}_cancel_imbalance'] = net_cancel / total_cancel

    # Net flow = adds - cancels per side
    bid_net_flow = bid_add_qty - bid_cancel_qty
    ask_net_flow = ask_add_qty - ask_cancel_qty
    features[f'{prefix}_bid_net_flow'] = bid_net_flow
    features[f'{prefix}_ask_net_flow'] = ask_net_flow

    # 2. TRADE INTENSITY
    n_trades = trade_mask.sum()
    trade_qty = qty[trade_mask].sum()
    features[f'{prefix}_trade_rate'] = n_trades / lookback_secs
    features[f'{prefix}_trade_volume'] = trade_qty
    features[f'{prefix}_avg_trade_size'] = trade_qty / max(n_trades, 1)

    # Bid vs ask trade aggression
    bid_trade_qty = qty[trade_mask & bid_mask].sum()
    ask_trade_qty = qty[trade_mask & ask_mask].sum()
    total_trade = bid_trade_qty + ask_trade_qty + 1e-8
    features[f'{prefix}_trade_imbalance'] = (bid_trade_qty - ask_trade_qty) / total_trade

    # 3. SWEEP DETECTION (large orders at extreme prices)
    if n_trades > 0:
        trade_prices = np.abs(price_rel[trade_mask])
        trade_qtys = qty[trade_mask]
        # Sweeps = trades far from mid with large size
        sweep_mask = (trade_prices > 2) & (trade_qtys > np.median(trade_qtys) * 2)
        features[f'{prefix}_sweep_count'] = sweep_mask.sum()
        features[f'{prefix}_sweep_volume'] = trade_qtys[sweep_mask].sum()
    else:
        features[f'{prefix}_sweep_count'] = 0
        features[f'{prefix}_sweep_volume'] = 0

    # 4. SPREAD DYNAMICS
    valid_spread = spread[spread > 0]
    if len(valid_spread) > 5:
        features[f'{prefix}_spread_mean'] = valid_spread.mean()
        features[f'{prefix}_spread_std'] = valid_spread.std()
        # Spread trend: is it widening or narrowing?
        half = len(valid_spread) // 2
        features[f'{prefix}_spread_trend'] = valid_spread[half:].mean() - valid_spread[:half].mean()
    else:
        features[f'{prefix}_spread_mean'] = np.nan
        features[f'{prefix}_spread_std'] = np.nan
        features[f'{prefix}_spread_trend'] = np.nan

    # 5. CANCEL-TO-ADD RATIO (high ratio = spoofing/uncertainty)
    n_adds = add_mask.sum()
    n_cancels = cancel_mask.sum()
    features[f'{prefix}_cancel_add_ratio'] = n_cancels / max(n_adds, 1)

    # 6. LARGE ORDER PRESENCE (institutional proxy)
    if n_events > 10:
        q90 = np.quantile(qty, 0.9)
        large_mask = qty > q90
        features[f'{prefix}_large_order_frac'] = large_mask.sum() / n_events
        # Are large orders on same side as our trade?
        if direction is not None:
            aligned_side = 0 if direction == 1 else 1  # bid for long, ask for short
            large_aligned = (large_mask & (sides == aligned_side)).sum()
            large_opposed = (large_mask & (sides == (1 - aligned_side))).sum()
            features[f'{prefix}_large_order_alignment'] = (
                (large_aligned - large_opposed) / max(large_aligned + large_opposed, 1)
            )
        else:
            features[f'{prefix}_large_order_alignment'] = 0
    else:
        features[f'{prefix}_large_order_frac'] = np.nan
        features[f'{prefix}_large_order_alignment'] = np.nan

    # 7. EVENT RATE (overall activity)
    features[f'{prefix}_event_rate'] = n_events / lookback_secs

    # 8. PRICE PRESSURE (volume-weighted price position)
    if n_events > 0:
        features[f'{prefix}_vwap_rel'] = np.average(price_rel, weights=qty + 1e-8)
    else:
        features[f'{prefix}_vwap_rel'] = np.nan

    return features


def _get_mbo_feature_names(prefix):
    """Return all MBO feature names for a given prefix."""
    return [
        f'{prefix}_book_imbalance', f'{prefix}_cancel_imbalance',
        f'{prefix}_bid_net_flow', f'{prefix}_ask_net_flow',
        f'{prefix}_trade_rate', f'{prefix}_trade_volume',
        f'{prefix}_avg_trade_size', f'{prefix}_trade_imbalance',
        f'{prefix}_sweep_count', f'{prefix}_sweep_volume',
        f'{prefix}_spread_mean', f'{prefix}_spread_std',
        f'{prefix}_spread_trend', f'{prefix}_cancel_add_ratio',
        f'{prefix}_large_order_frac', f'{prefix}_large_order_alignment',
        f'{prefix}_event_rate', f'{prefix}_vwap_rel',
    ]


def extract_all_mbo_features(trades):
    """Extract MBO features for all trades across all lookback windows."""
    log.info("=" * 70)
    log.info("PART 2: Extracting MBO features for all trades")
    log.info("=" * 70)

    # Group trades by date
    trades_by_date = defaultdict(list)
    for i, t in enumerate(trades):
        trades_by_date[t['date']].append((i, t))

    all_features = []
    dates_processed = 0
    dates_failed = 0

    for date_str in sorted(trades_by_date.keys()):
        date_trades = trades_by_date[date_str]

        # Load MBO events for this date
        # Convert date format (YYYY-MM-DD -> YYYYMMDD for file lookup)
        date_clean = date_str.replace('-', '')
        events, timestamps, metadata = load_mbo_events_for_date(date_clean)

        if events is None:
            log.warning(f"  No MBO events for {date_str}, skipping {len(date_trades)} trades")
            dates_failed += 1
            for idx, trade in date_trades:
                feat_row = {'trade_idx': idx, 'date': date_str, 'has_mbo': False}
                all_features.append(feat_row)
            continue

        dates_processed += 1
        log.info(f"  {date_str}: {len(events):,} events, {len(date_trades)} trades")

        for idx, trade in date_trades:
            fill_ts = trade['fill_ts']
            # Convert fill timestamp to nanoseconds
            fill_ts_ns = int(fill_ts.value)  # pandas Timestamp.value is in nanoseconds

            feat_row = {
                'trade_idx': idx,
                'date': date_str,
                'has_mbo': True,
                'direction': trade['direction'],
                'winner': trade['winner'],
                'exit_type': trade['exit_type'],
                'exit_pnl': trade['exit_pnl'],
                'exit_minute': trade['exit_minute'],
                'pred_30m': trade['pred_30m'],
            }

            # Extract features at each lookback window
            for window_secs in MBO_LOOKBACK_WINDOWS:
                window_feats = extract_mbo_features_at_time(
                    events, timestamps, fill_ts_ns, window_secs,
                    direction=trade['direction']
                )
                feat_row.update(window_feats)

            all_features.append(feat_row)

        # Free memory
        del events, timestamps
        gc.collect()

    log.info(f"MBO features extracted: {dates_processed} dates processed, "
             f"{dates_failed} dates missing")
    log.info(f"Total feature rows: {len(all_features)}, "
             f"with MBO: {sum(1 for f in all_features if f.get('has_mbo', False))}")

    return pd.DataFrame(all_features)


# ============================================================
# PART 3: ANALYZE WINNERS VS LOSERS
# ============================================================

def analyze_winners_vs_losers(feature_df):
    """Statistical comparison of MBO features for winners vs losers."""
    log.info("=" * 70)
    log.info("PART 3: Winners vs Losers Analysis")
    log.info("=" * 70)

    df = feature_df[feature_df['has_mbo'] == True].copy()
    if len(df) < 20:
        log.warning("Too few trades with MBO data for analysis")
        return {}

    winners = df[df['winner'] == True]
    losers = df[df['winner'] == False]
    log.info(f"Analyzing: {len(winners)} winners vs {len(losers)} losers")

    results = {}
    significant_features = []

    # Get all MBO feature columns
    mbo_cols = [c for c in df.columns if c.startswith('mbo_')]

    for col in mbo_cols:
        w_vals = winners[col].dropna()
        l_vals = losers[col].dropna()
        if len(w_vals) < 5 or len(l_vals) < 5:
            continue

        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(w_vals, l_vals, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((w_vals.std()**2 + l_vals.std()**2) / 2)
        cohens_d = (w_vals.mean() - l_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Rank-biserial correlation (non-parametric effect size)
        try:
            u_stat, mann_p = stats.mannwhitneyu(w_vals, l_vals, alternative='two-sided')
            rank_biserial = 1 - 2 * u_stat / (len(w_vals) * len(l_vals))
        except:
            mann_p = 1.0
            rank_biserial = 0.0

        result = {
            'feature': col,
            'winner_mean': w_vals.mean(),
            'loser_mean': l_vals.mean(),
            'winner_median': w_vals.median(),
            'loser_median': l_vals.median(),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mann_whitney_p': mann_p,
            'rank_biserial': rank_biserial,
        }
        results[col] = result

        if p_value < 0.05 and abs(cohens_d) > 0.2:
            significant_features.append(result)

    # Sort significant features by effect size
    significant_features.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

    log.info(f"\nSignificant features (p<0.05, |d|>0.2): {len(significant_features)}")
    log.info("-" * 80)
    for sf in significant_features[:20]:
        direction = "WINNERS higher" if sf['cohens_d'] > 0 else "LOSERS higher"
        log.info(f"  {sf['feature']:45s}  d={sf['cohens_d']:+.3f}  p={sf['p_value']:.4f}  "
                 f"W={sf['winner_mean']:+.3f}  L={sf['loser_mean']:+.3f}  [{direction}]")

    return {
        'all_features': results,
        'significant_features': significant_features,
        'n_winners': len(winners),
        'n_losers': len(losers),
    }


# ============================================================
# PART 4: BUILD MBO CONFLUENCE FILTER
# ============================================================

def build_mbo_confluence_filter(feature_df):
    """
    Train LightGBM to predict winners using MBO features.
    Walk-forward validation (sliding window) per HC #0.
    """
    log.info("=" * 70)
    log.info("PART 4: MBO Confluence Filter (Walk-Forward)")
    log.info("=" * 70)

    df = feature_df[feature_df['has_mbo'] == True].copy()
    if len(df) < 50:
        log.warning("Too few trades with MBO data for walk-forward")
        return {}

    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)

    # MBO feature columns only
    mbo_cols = [c for c in df.columns if c.startswith('mbo_')]
    mbo_cols = [c for c in mbo_cols if df[c].notna().sum() > len(df) * 0.5]
    log.info(f"Using {len(mbo_cols)} MBO features (>50% non-NaN)")

    # Walk-forward: train on 30 days, predict 5 days, slide
    unique_dates = sorted(df['date'].unique())
    train_window = 30
    test_window = 5

    all_preds = []
    all_actuals = []
    all_dates = []
    all_indices = []

    for start in range(0, len(unique_dates) - train_window, test_window):
        train_dates = unique_dates[start:start + train_window]
        test_dates = unique_dates[start + train_window:start + train_window + test_window]

        if len(test_dates) == 0:
            break

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)

        X_train = df.loc[train_mask, mbo_cols].fillna(0).values
        y_train = df.loc[train_mask, 'winner'].astype(int).values
        X_test = df.loc[test_mask, mbo_cols].fillna(0).values
        y_test = df.loc[test_mask, 'winner'].astype(int).values

        if len(X_train) < 20 or len(X_test) < 3:
            continue
        if y_train.sum() < 3:  # need at least a few winners to learn from
            continue

        # Train LightGBM classifier
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 8,
            'max_depth': 3,
            'min_child_samples': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'is_unbalance': True,
        }

        train_data = lgb.Dataset(X_train, y_train)
        model = lgb.train(lgb_params, train_data, num_boost_round=100)

        preds = model.predict(X_test)
        all_preds.extend(preds.tolist())
        all_actuals.extend(y_test.tolist())
        all_dates.extend(df.loc[test_mask, 'date'].tolist())
        all_indices.extend(df.loc[test_mask].index.tolist())

    if len(all_preds) < 20:
        log.warning("Too few walk-forward predictions for evaluation")
        return {}

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # Evaluate: can MBO features predict winners?
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_actuals, all_preds)
    except:
        auc = 0.5

    log.info(f"Walk-forward AUC: {auc:.4f}")
    log.info(f"Predictions: {len(all_preds)}, Winners: {all_actuals.sum():.0f} "
             f"({all_actuals.mean()*100:.1f}%)")

    # Test as a filter: what if we only take trades where MBO score > threshold?
    filter_results = []
    for threshold_pct in [0.10, 0.20, 0.30, 0.40, 0.50]:
        threshold = np.quantile(all_preds, threshold_pct)
        passed_mask = all_preds >= threshold
        if passed_mask.sum() < 5:
            continue

        passed_actuals = all_actuals[passed_mask]
        blocked_actuals = all_actuals[~passed_mask]

        result = {
            'threshold_pct': threshold_pct,
            'threshold_val': threshold,
            'n_passed': passed_mask.sum(),
            'n_blocked': (~passed_mask).sum(),
            'passed_wr': passed_actuals.mean(),
            'blocked_wr': blocked_actuals.mean() if len(blocked_actuals) > 0 else 0,
            'base_wr': all_actuals.mean(),
            'wr_lift': passed_actuals.mean() - all_actuals.mean(),
        }
        filter_results.append(result)

        log.info(f"  Filter p>{threshold_pct:.0%}: pass {result['n_passed']}, "
                 f"WR {result['passed_wr']:.1%} vs base {result['base_wr']:.1%} "
                 f"(lift {result['wr_lift']:+.1%})")

    # Feature importance (last fold)
    try:
        importance = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(mbo_cols, importance), key=lambda x: x[1], reverse=True)
        log.info(f"\nTop 10 MBO features by importance:")
        for name, imp in feat_imp[:10]:
            log.info(f"  {name:45s}  importance={imp:.1f}")
    except:
        feat_imp = []

    return {
        'auc': auc,
        'n_predictions': len(all_preds),
        'filter_results': filter_results,
        'feature_importance': [(n, float(i)) for n, i in feat_imp[:20]],
    }


# ============================================================
# PART 5: MID-TRADE MBO ANALYSIS
# ============================================================

def analyze_mid_trade_mbo(trades, feature_df):
    """
    Analyze MBO patterns DURING trades (after entry, before exit).
    Can we detect losing trades early from order book changes?
    """
    log.info("=" * 70)
    log.info("PART 5: Mid-Trade MBO Pattern Analysis")
    log.info("=" * 70)

    # For each trade, extract MBO features at multiple points after entry
    # Compare the evolution of these features for winners vs losers

    df = feature_df[feature_df['has_mbo'] == True].copy()
    if len(df) < 20:
        log.warning("Too few trades for mid-trade analysis")
        return {}

    trades_by_date = defaultdict(list)
    for i, t in enumerate(trades):
        if i < len(df) and df.iloc[i].get('has_mbo', False):
            trades_by_date[t['date']].append((i, t))

    # For speed, just analyze at T+1min and T+2min after entry
    # (most SL hits happen in minute 1-2)
    checkpoints = [30, 60, 120]  # seconds after entry

    mid_trade_features = {cp: {'winners': [], 'losers': []} for cp in checkpoints}

    for date_str in sorted(trades_by_date.keys()):
        date_trades = trades_by_date[date_str]
        date_clean = date_str.replace('-', '')
        events, timestamps, _ = load_mbo_events_for_date(date_clean)

        if events is None:
            continue

        for idx, trade in date_trades:
            fill_ts_ns = int(trade['fill_ts'].value)

            for cp_secs in checkpoints:
                check_ts_ns = fill_ts_ns + int(cp_secs * 1e9)
                # Extract features looking back from the checkpoint
                feats = extract_mbo_features_at_time(
                    events, timestamps, check_ts_ns, lookback_secs=cp_secs,
                    direction=trade['direction']
                )
                if trade['winner']:
                    mid_trade_features[cp_secs]['winners'].append(feats)
                else:
                    mid_trade_features[cp_secs]['losers'].append(feats)

        del events, timestamps
        gc.collect()

    # Analyze differences at each checkpoint
    results = {}
    for cp_secs in checkpoints:
        w_list = mid_trade_features[cp_secs]['winners']
        l_list = mid_trade_features[cp_secs]['losers']

        if len(w_list) < 5 or len(l_list) < 5:
            continue

        w_df = pd.DataFrame(w_list)
        l_df = pd.DataFrame(l_list)

        sig_feats = []
        for col in w_df.columns:
            w_vals = w_df[col].dropna()
            l_vals = l_df[col].dropna()
            if len(w_vals) < 5 or len(l_vals) < 5:
                continue

            t_stat, p_value = stats.ttest_ind(w_vals, l_vals, equal_var=False)
            pooled_std = np.sqrt((w_vals.std()**2 + l_vals.std()**2) / 2)
            cohens_d = (w_vals.mean() - l_vals.mean()) / pooled_std if pooled_std > 0 else 0

            if p_value < 0.10 and abs(cohens_d) > 0.15:
                sig_feats.append({
                    'feature': col,
                    'checkpoint_secs': cp_secs,
                    'cohens_d': cohens_d,
                    'p_value': p_value,
                    'winner_mean': w_vals.mean(),
                    'loser_mean': l_vals.mean(),
                })

        sig_feats.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
        results[f't_plus_{cp_secs}s'] = sig_feats

        log.info(f"\nT+{cp_secs}s: {len(sig_feats)} significant features")
        for sf in sig_feats[:5]:
            direction = "higher in WINNERS" if sf['cohens_d'] > 0 else "higher in LOSERS"
            log.info(f"  {sf['feature']:45s}  d={sf['cohens_d']:+.3f}  p={sf['p_value']:.3f}  [{direction}]")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("MBO TRADE ANATOMY ANALYSIS v1")
    log.info(f"Started: {datetime.now()}")
    log.info("=" * 70)

    # Step 1: Reconstruct champion trades
    trades = load_champion_trades()

    # Step 2: Extract MBO features around each trade
    feature_df = extract_all_mbo_features(trades)

    # Save intermediate results
    feature_df.to_parquet(OUTPUT_DIR / "trade_mbo_features.parquet", index=False)
    log.info(f"Saved feature dataframe: {len(feature_df)} rows, {len(feature_df.columns)} columns")

    # Step 3: Winners vs Losers statistical analysis
    wl_analysis = analyze_winners_vs_losers(feature_df)

    # Step 4: Walk-forward MBO confluence filter
    confluence_results = build_mbo_confluence_filter(feature_df)

    # Step 5: Mid-trade MBO analysis
    mid_trade_results = analyze_mid_trade_mbo(trades, feature_df)

    # Save all results
    elapsed = time.time() - start_time

    summary = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_trades': len(trades),
        'n_trades_with_mbo': int(feature_df['has_mbo'].sum()) if 'has_mbo' in feature_df.columns else 0,
        'n_winners': sum(1 for t in trades if t['winner']),
        'n_losers': sum(1 for t in trades if not t['winner']),
        'winners_vs_losers': {
            'n_significant_features': len(wl_analysis.get('significant_features', [])),
            'top_features': [
                {'feature': sf['feature'], 'cohens_d': sf['cohens_d'], 'p_value': sf['p_value']}
                for sf in wl_analysis.get('significant_features', [])[:10]
            ],
        },
        'confluence_filter': confluence_results,
        'mid_trade_analysis': {
            k: [{'feature': sf['feature'], 'cohens_d': sf['cohens_d'], 'p_value': sf['p_value']}
                for sf in v[:5]]
            for k, v in mid_trade_results.items()
        } if mid_trade_results else {},
    }

    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("=" * 70)
    log.info(f"COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"Results saved to {OUTPUT_DIR}")
    log.info("=" * 70)

    # Print summary for Discord
    log.info("\n=== SUMMARY FOR USER ===")
    log.info(f"Analyzed {len(trades)} champion trades at MBO tick level")

    n_sig = len(wl_analysis.get('significant_features', []))
    log.info(f"\n1. WINNERS VS LOSERS: {n_sig} order book features significantly different")
    for sf in wl_analysis.get('significant_features', [])[:5]:
        feat_name = sf['feature'].replace('mbo_', '').replace('_', ' ')
        direction = "higher for winners" if sf['cohens_d'] > 0 else "higher for losers"
        log.info(f"   - {feat_name}: {direction} (effect={sf['cohens_d']:+.2f})")

    if confluence_results:
        auc = confluence_results.get('auc', 0.5)
        log.info(f"\n2. MBO CONFLUENCE FILTER: AUC={auc:.3f}")
        for fr in confluence_results.get('filter_results', []):
            log.info(f"   - Block bottom {fr['threshold_pct']:.0%}: "
                     f"WR {fr['passed_wr']:.1%} (base {fr['base_wr']:.1%}, "
                     f"lift {fr['wr_lift']:+.1%})")

    if mid_trade_results:
        log.info(f"\n3. MID-TRADE SIGNALS:")
        for k, v in mid_trade_results.items():
            if v:
                log.info(f"   {k}: {len(v)} distinguishing features found")

    return summary


if __name__ == '__main__':
    main()
