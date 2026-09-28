#!/usr/bin/env python3
"""
MBO Microstructure Feature Analysis — Tick-Level Alpha Research
================================================================
Computes order-flow-based microstructure features from raw MBO event data
and measures their predictive power (Information Coefficient) for short-term
price movements at 1s, 5s, 10s, 30s horizons.

Features computed:
  1. Order Flow Imbalance (OFI): net aggressive buy vs sell pressure
  2. Trade Flow Toxicity (VPIN-like): volume-synchronized probability of informed trading
  3. Book Depth Imbalance: bid vs ask depth at multiple levels
  4. Aggressive vs Passive Fill Ratio
  5. Large Order Detection: orders > 5x median size
  6. Trade intensity: trades per second
  7. Cancel-to-trade ratio
  8. Order arrival imbalance

Uses preprocessed MBO .npz files (tick-level) and aligned prediction indices.
Lightweight: reads existing files, no heavy computation.
"""

import os
import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Directories
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/preprocessed_mbo")
ALIGN_DIR = MBO_DIR / "pred_indices_aligned"
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/mbo_microstructure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MBO action codes (from preprocess_raw_mbo.py)
ACTION_ADD = 0
ACTION_CANCEL = 1
ACTION_MODIFY = 2
ACTION_TRADE = 3
ACTION_FILL = 4

# Side codes
SIDE_BID = 0
SIDE_ASK = 1

# Time windows for feature computation (in nanoseconds)
WINDOWS_NS = {
    '250ms': 250_000_000,
    '1s': 1_000_000_000,
    '5s': 5_000_000_000,
    '10s': 10_000_000_000,
}

# Forward return horizons (in nanoseconds)
HORIZONS_NS = {
    '1s': 1_000_000_000,
    '5s': 5_000_000_000,
    '10s': 10_000_000_000,
    '30s': 30_000_000_000,
}

# ES tick value
ES_TICK = 0.25


def get_available_dates():
    """Find dates that have both preprocessed MBO and aligned prediction indices."""
    mbo_dates = set()
    for f in os.listdir(MBO_DIR):
        if f.startswith('mbo_') and f.endswith('.npz'):
            mbo_dates.add(f.replace('mbo_', '').replace('.npz', ''))

    align_dates = set()
    if ALIGN_DIR.exists():
        for f in os.listdir(ALIGN_DIR):
            if f.startswith('aligned_') and f.endswith('.npz'):
                align_dates.add(f.replace('aligned_', '').replace('.npz', ''))

    both = sorted(mbo_dates & align_dates)
    return both


def load_mbo_day(date_str):
    """Load preprocessed MBO data for a single day."""
    path = MBO_DIR / f"mbo_{date_str}.npz"
    d = np.load(path, allow_pickle=True)
    return {
        'ts_ns': d['ts_ns'],
        'action': d['action'],
        'side': d['side'],
        'price': d['price'],
        'size': d['size'],
        'order_id': d['order_id'],
    }


def filter_rth(mbo, date_str):
    """
    Filter to RTH (Regular Trading Hours): 9:30 ET - 16:00 ET.
    MBO timestamps are in UTC nanoseconds.
    RTH = 13:30 - 20:00 UTC (ET is UTC-5 during EST, UTC-4 during EDT).
    For simplicity, use the middle of the day as reference.
    """
    ts = mbo['ts_ns']
    # Find the approximate day boundary by looking at price clustering
    # RTH has the most activity. Use trade events to find the dense period.
    trades_mask = mbo['action'] == ACTION_TRADE
    if trades_mask.sum() == 0:
        return mbo, np.ones(len(ts), dtype=bool)

    trade_ts = ts[trades_mask]
    # Find the densest 6.5-hour window (RTH length)
    # Simple approach: look at hour-level bins
    min_ts = ts.min()
    hours = (ts - min_ts) / 3.6e12  # convert to hours

    # RTH is typically the densest period. Use all data for now
    # (preprocessed data is already mostly RTH for ES)
    # Just filter out the first and last 5% by timestamp to avoid globex-only
    p5 = np.percentile(trade_ts, 2)
    p95 = np.percentile(trade_ts, 98)
    mask = (ts >= p5) & (ts <= p95)
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(ts) else v
            for k, v in mbo.items()}, mask


def compute_microstructure_features_at_points(mbo, eval_indices, window_ns=1_000_000_000):
    """
    Compute microstructure features at specific evaluation points.
    Each feature is computed over a lookback window ending at the eval point.

    Args:
        mbo: dict with ts_ns, action, side, price, size arrays
        eval_indices: array of indices into the MBO arrays where to evaluate
        window_ns: lookback window in nanoseconds

    Returns:
        dict of feature_name -> array of values at each eval point
    """
    ts = mbo['ts_ns']
    action = mbo['action']
    side = mbo['side']
    price = mbo['price']
    size = mbo['size']

    n_eval = len(eval_indices)
    n_events = len(ts)

    # Pre-compute masks
    is_trade = action == ACTION_TRADE
    is_fill = action == ACTION_FILL
    is_add = action == ACTION_ADD
    is_cancel = action == ACTION_CANCEL
    is_modify = action == ACTION_MODIFY
    is_bid = side == SIDE_BID
    is_ask = side == SIDE_ASK

    # Output arrays
    features = {
        'ofi_net': np.full(n_eval, np.nan),
        'ofi_imbalance': np.full(n_eval, np.nan),
        'trade_imbalance': np.full(n_eval, np.nan),
        'trade_volume_imbalance': np.full(n_eval, np.nan),
        'vpin_proxy': np.full(n_eval, np.nan),
        'bid_depth_adds': np.full(n_eval, np.nan),
        'ask_depth_adds': np.full(n_eval, np.nan),
        'depth_imbalance': np.full(n_eval, np.nan),
        'cancel_rate_bid': np.full(n_eval, np.nan),
        'cancel_rate_ask': np.full(n_eval, np.nan),
        'cancel_imbalance': np.full(n_eval, np.nan),
        'cancel_to_trade_ratio': np.full(n_eval, np.nan),
        'large_order_fraction': np.full(n_eval, np.nan),
        'large_order_side_imbalance': np.full(n_eval, np.nan),
        'trade_intensity': np.full(n_eval, np.nan),
        'avg_trade_size': np.full(n_eval, np.nan),
        'price_impact': np.full(n_eval, np.nan),
        'order_arrival_imbalance': np.full(n_eval, np.nan),
        'modify_intensity': np.full(n_eval, np.nan),
    }

    # Compute median trade size for large order detection
    trade_sizes = size[is_trade | is_fill]
    if len(trade_sizes) > 0:
        median_trade_size = np.median(trade_sizes)
        large_threshold = 5 * median_trade_size
    else:
        median_trade_size = 1
        large_threshold = 5

    # Use searchsorted for efficient window lookback
    for i, idx in enumerate(eval_indices):
        if idx < 0 or idx >= n_events:
            continue

        t_now = ts[idx]
        t_start = t_now - window_ns

        # Find window start using searchsorted
        win_start = np.searchsorted(ts, t_start, side='left')
        win_end = idx + 1  # inclusive of current event

        if win_start >= win_end:
            continue

        # Slice window
        w_action = action[win_start:win_end]
        w_side = side[win_start:win_end]
        w_price = price[win_start:win_end]
        w_size = size[win_start:win_end]
        w_ts = ts[win_start:win_end]

        # Masks within window
        w_is_trade = (w_action == ACTION_TRADE) | (w_action == ACTION_FILL)
        w_is_add = w_action == ACTION_ADD
        w_is_cancel = w_action == ACTION_CANCEL
        w_is_modify = w_action == ACTION_MODIFY
        w_is_bid = w_side == SIDE_BID
        w_is_ask = w_side == SIDE_ASK

        # --- OFI (Order Flow Imbalance) ---
        # Net aggressive buying: trades at ask (buyer-initiated) minus trades at bid (seller-initiated)
        buy_trades = w_is_trade & w_is_ask  # Trade at ask = buyer aggressor
        sell_trades = w_is_trade & w_is_bid  # Trade at bid = seller aggressor
        buy_vol = w_size[buy_trades].sum() if buy_trades.any() else 0
        sell_vol = w_size[sell_trades].sum() if sell_trades.any() else 0
        total_trade_vol = buy_vol + sell_vol

        features['ofi_net'][i] = buy_vol - sell_vol
        features['ofi_imbalance'][i] = (buy_vol - sell_vol) / max(total_trade_vol, 1)

        # --- Trade count imbalance ---
        n_buy = buy_trades.sum()
        n_sell = sell_trades.sum()
        n_total = n_buy + n_sell
        features['trade_imbalance'][i] = (n_buy - n_sell) / max(n_total, 1)
        features['trade_volume_imbalance'][i] = features['ofi_imbalance'][i]

        # --- VPIN proxy ---
        # Simplified VPIN: |buy_vol - sell_vol| / total_vol
        features['vpin_proxy'][i] = abs(buy_vol - sell_vol) / max(total_trade_vol, 1)

        # --- Book depth imbalance (add orders) ---
        bid_add_vol = w_size[w_is_add & w_is_bid].sum() if (w_is_add & w_is_bid).any() else 0
        ask_add_vol = w_size[w_is_add & w_is_ask].sum() if (w_is_add & w_is_ask).any() else 0
        features['bid_depth_adds'][i] = bid_add_vol
        features['ask_depth_adds'][i] = ask_add_vol
        total_add = bid_add_vol + ask_add_vol
        features['depth_imbalance'][i] = (bid_add_vol - ask_add_vol) / max(total_add, 1)

        # --- Cancel rates ---
        bid_cancel = (w_is_cancel & w_is_bid).sum()
        ask_cancel = (w_is_cancel & w_is_ask).sum()
        total_cancel = bid_cancel + ask_cancel
        features['cancel_rate_bid'][i] = bid_cancel
        features['cancel_rate_ask'][i] = ask_cancel
        features['cancel_imbalance'][i] = (bid_cancel - ask_cancel) / max(total_cancel, 1)
        features['cancel_to_trade_ratio'][i] = total_cancel / max(n_total, 1)

        # --- Large order detection ---
        large_mask = w_size >= large_threshold
        if w_is_trade.sum() > 0:
            features['large_order_fraction'][i] = (large_mask & w_is_trade).sum() / max(w_is_trade.sum(), 1)
            large_buy = (large_mask & buy_trades).sum()
            large_sell = (large_mask & sell_trades).sum()
            features['large_order_side_imbalance'][i] = (large_buy - large_sell) / max(large_buy + large_sell, 1)

        # --- Trade intensity (trades per second) ---
        window_duration_s = (w_ts[-1] - w_ts[0]) / 1e9 if len(w_ts) > 1 else 1.0
        features['trade_intensity'][i] = n_total / max(window_duration_s, 0.001)

        # --- Average trade size ---
        if n_total > 0:
            features['avg_trade_size'][i] = total_trade_vol / n_total

        # --- Price impact: price change over window in ticks ---
        trade_prices = w_price[w_is_trade]
        if len(trade_prices) >= 2:
            features['price_impact'][i] = (trade_prices[-1] - trade_prices[0]) / ES_TICK

        # --- Order arrival imbalance (all new orders bid vs ask) ---
        bid_new = (w_is_add & w_is_bid).sum()
        ask_new = (w_is_add & w_is_ask).sum()
        features['order_arrival_imbalance'][i] = (bid_new - ask_new) / max(bid_new + ask_new, 1)

        # --- Modify intensity ---
        features['modify_intensity'][i] = w_is_modify.sum() / max(window_duration_s, 0.001)

    return features


def compute_forward_returns_from_trades(mbo, eval_indices, horizons_ns):
    """
    Compute forward price returns at each evaluation point using trade prices.
    Returns dict of horizon_name -> array of forward returns in ticks.
    """
    ts = mbo['ts_ns']
    price = mbo['price']
    action = mbo['action']
    is_trade = (action == ACTION_TRADE) | (action == ACTION_FILL)

    # Build trade-only price/time arrays
    trade_mask = is_trade
    trade_ts = ts[trade_mask]
    trade_price = price[trade_mask]

    if len(trade_ts) == 0:
        return {h: np.full(len(eval_indices), np.nan) for h in horizons_ns}

    fwd_returns = {}
    for h_name, h_ns in horizons_ns.items():
        returns = np.full(len(eval_indices), np.nan)
        for i, idx in enumerate(eval_indices):
            if idx < 0 or idx >= len(ts):
                continue
            t_now = ts[idx]
            # Current price: last trade at or before t_now
            cur_idx = np.searchsorted(trade_ts, t_now, side='right') - 1
            if cur_idx < 0:
                continue
            cur_price = trade_price[cur_idx]

            # Future price: last trade at or before t_now + horizon
            t_future = t_now + h_ns
            fut_idx = np.searchsorted(trade_ts, t_future, side='right') - 1
            if fut_idx <= cur_idx:
                continue
            fut_price = trade_price[fut_idx]

            returns[i] = (fut_price - cur_price) / ES_TICK  # in ticks
        fwd_returns[h_name] = returns

    return fwd_returns


def compute_ic(feature_vals, return_vals):
    """Compute Spearman rank IC between feature and forward returns."""
    mask = np.isfinite(feature_vals) & np.isfinite(return_vals)
    n = mask.sum()
    if n < 30:
        return np.nan, np.nan, np.nan, 0

    f = feature_vals[mask]
    r = return_vals[mask]

    ic, p_value = stats.spearmanr(f, r)

    if abs(ic) >= 1.0:
        t_stat = np.inf * np.sign(ic)
    else:
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)

    return ic, t_stat, p_value, n


def analyze_single_date(date_str, window_ns=1_000_000_000, max_eval_points=5000):
    """
    Analyze microstructure features for a single date.
    Returns per-feature IC values at each horizon.
    """
    print(f"  Loading MBO data for {date_str}...")
    mbo = load_mbo_day(date_str)

    # Load aligned prediction indices
    align = np.load(ALIGN_DIR / f"aligned_{date_str}.npz", allow_pickle=True)
    eval_indices = align['event_indices']
    in_range = align['in_mbo_range']

    # Filter to in-range indices only
    eval_indices = eval_indices[in_range]

    # Subsample if too many eval points (for speed)
    if len(eval_indices) > max_eval_points:
        step = len(eval_indices) // max_eval_points
        eval_indices = eval_indices[::step]

    print(f"    Events: {len(mbo['ts_ns']):,}, Eval points: {len(eval_indices):,}")

    # Compute microstructure features at 1s window
    print(f"    Computing features (window={window_ns/1e9:.1f}s)...")
    features = compute_microstructure_features_at_points(mbo, eval_indices, window_ns)

    # Compute forward returns
    print(f"    Computing forward returns...")
    fwd_returns = compute_forward_returns_from_trades(mbo, eval_indices, HORIZONS_NS)

    return features, fwd_returns, len(eval_indices)


def main():
    dates = get_available_dates()
    print(f"Found {len(dates)} dates with both MBO and aligned predictions")
    print(f"Dates: {dates}")

    # Use a sample of dates for efficiency (lightweight analysis)
    # Pick every 3rd date for diversity
    sample_dates = dates[::3]
    if len(sample_dates) < 5:
        sample_dates = dates[:5]
    print(f"\nAnalyzing {len(sample_dates)} sample dates: {sample_dates}")

    # Collect all features and returns across dates
    all_features = defaultdict(list)
    all_returns = {h: [] for h in HORIZONS_NS}
    total_points = 0

    for date_str in sample_dates:
        try:
            features, fwd_returns, n_pts = analyze_single_date(
                date_str, window_ns=1_000_000_000, max_eval_points=3000
            )
            for feat_name, vals in features.items():
                all_features[feat_name].append(vals)
            for h_name, vals in fwd_returns.items():
                all_returns[h_name].append(vals)
            total_points += n_pts
        except Exception as e:
            print(f"  ERROR on {date_str}: {e}")
            continue

    print(f"\nTotal evaluation points across {len(sample_dates)} dates: {total_points:,}")

    # Concatenate
    concat_features = {k: np.concatenate(v) for k, v in all_features.items()}
    concat_returns = {k: np.concatenate(v) for k, v in all_returns.items()}

    # Compute IC for each feature at each horizon
    results = []
    print("\n" + "=" * 100)
    print("MICROSTRUCTURE FEATURE IC ANALYSIS (1s lookback window)")
    print("=" * 100)
    print(f"{'Feature':30s} {'Horizon':8s} {'IC':>10s} {'t-stat':>10s} {'p-value':>12s} {'n_obs':>8s} {'Sig?':>5s}")
    print("-" * 100)

    for feat_name in sorted(concat_features.keys()):
        feat_vals = concat_features[feat_name]
        for h_name in ['1s', '5s', '10s', '30s']:
            ret_vals = concat_returns[h_name]
            ic, t_stat, p_val, n = compute_ic(feat_vals, ret_vals)
            sig = "***" if (not np.isnan(ic) and abs(ic) > 0.02 and abs(t_stat) > 3.0) else ""
            results.append({
                'feature': feat_name,
                'horizon': h_name,
                'IC': round(float(ic), 6) if not np.isnan(ic) else None,
                't_stat': round(float(t_stat), 4) if not np.isnan(t_stat) else None,
                'p_value': round(float(p_val), 8) if not np.isnan(p_val) else None,
                'n_obs': int(n),
                'significant': bool(sig),
            })
            if not np.isnan(ic):
                print(f"  {feat_name:28s} {h_name:8s} {ic:+10.5f} {t_stat:+10.3f} {p_val:12.8f} {n:8d} {sig:>5s}")

    # Sort by |IC| descending
    results_sorted = sorted(results, key=lambda r: abs(r['IC']) if r['IC'] is not None else 0, reverse=True)

    # Save full results
    with open(OUTPUT_DIR / "ic_results.json", "w") as f:
        json.dump(results_sorted, f, indent=2)

    # Print top features
    print("\n" + "=" * 100)
    print("TOP 20 FEATURES BY |IC|")
    print("=" * 100)
    print(f"{'Feature':30s} {'Horizon':8s} {'IC':>10s} {'t-stat':>10s} {'p-value':>12s} {'n_obs':>8s}")
    print("-" * 100)
    for r in results_sorted[:20]:
        ic = r['IC'] if r['IC'] is not None else 0
        t = r['t_stat'] if r['t_stat'] is not None else 0
        p = r['p_value'] if r['p_value'] is not None else 1
        sig = "***" if r.get('significant') else ""
        print(f"  {r['feature']:28s} {r['horizon']:8s} {ic:+10.5f} {t:+10.3f} {p:12.8f} {r['n_obs']:8d} {sig}")

    # Significant features summary
    sig_features = [r for r in results_sorted if r.get('significant')]
    print(f"\n{'=' * 100}")
    print(f"SIGNIFICANT FEATURES (|IC| > 0.02, |t| > 3.0): {len(sig_features)}")
    print(f"{'=' * 100}")
    for r in sig_features:
        print(f"  {r['feature']:28s} @ {r['horizon']:4s}: IC={r['IC']:+.5f}, t={r['t_stat']:+.3f}")

    # Now run with multiple windows to see window sensitivity
    print(f"\n{'=' * 100}")
    print("WINDOW SENSITIVITY ANALYSIS (best features only)")
    print(f"{'=' * 100}")

    # Get top 5 features from 1s window
    top_feats = []
    seen = set()
    for r in results_sorted:
        if r['feature'] not in seen and r.get('significant'):
            top_feats.append(r['feature'])
            seen.add(r['feature'])
        if len(top_feats) >= 5:
            break

    # If we didn't get 5 significant features, take top 5 by |IC|
    if len(top_feats) < 5:
        for r in results_sorted:
            if r['feature'] not in seen:
                top_feats.append(r['feature'])
                seen.add(r['feature'])
            if len(top_feats) >= 5:
                break

    window_results = {}
    for win_name, win_ns in [('250ms', 250_000_000), ('1s', 1_000_000_000), ('5s', 5_000_000_000)]:
        print(f"\n  Window: {win_name}")
        all_f2 = defaultdict(list)
        all_r2 = {h: [] for h in HORIZONS_NS}

        for date_str in sample_dates[:5]:  # Use fewer dates for speed
            try:
                features, fwd_returns, _ = analyze_single_date(
                    date_str, window_ns=win_ns, max_eval_points=2000
                )
                for feat_name, vals in features.items():
                    if feat_name in top_feats:
                        all_f2[feat_name].append(vals)
                for h_name, vals in fwd_returns.items():
                    all_r2[h_name].append(vals)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        concat_f2 = {k: np.concatenate(v) for k, v in all_f2.items()}
        concat_r2 = {k: np.concatenate(v) for k, v in all_r2.items()}

        for feat_name in top_feats:
            if feat_name not in concat_f2:
                continue
            for h_name in ['1s', '5s', '10s']:
                ic, t_stat, p_val, n = compute_ic(concat_f2[feat_name], concat_r2[h_name])
                if not np.isnan(ic):
                    key = (feat_name, h_name, win_name)
                    window_results[key] = {'IC': ic, 't_stat': t_stat, 'n': n}
                    sig = "***" if abs(ic) > 0.02 and abs(t_stat) > 3 else ""
                    print(f"    {feat_name:28s} @ {h_name:4s}: IC={ic:+.5f}, t={t_stat:+.3f}, n={n} {sig}")

    # Save window sensitivity results
    win_res_list = []
    for (feat, h, w), vals in window_results.items():
        clean_vals = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in vals.items()}
        win_res_list.append({'feature': feat, 'horizon': h, 'window': w, **clean_vals})
    with open(OUTPUT_DIR / "window_sensitivity.json", "w") as f:
        json.dump(win_res_list, f, indent=2)

    # Per-date IC stability check (for top features)
    print(f"\n{'=' * 100}")
    print("PER-DATE IC STABILITY (top features @ 5s horizon, 1s window)")
    print(f"{'=' * 100}")

    for feat_name in top_feats[:3]:
        date_ics = []
        for j, date_str in enumerate(sample_dates):
            feat_vals = all_features[feat_name][j] if j < len(all_features[feat_name]) else None
            ret_vals = all_returns['5s'][j] if j < len(all_returns['5s']) else None
            if feat_vals is None or ret_vals is None:
                continue
            ic, _, _, n = compute_ic(feat_vals, ret_vals)
            if not np.isnan(ic):
                date_ics.append((date_str if j < len(sample_dates) else '?', ic, n))

        if date_ics:
            ics = [x[1] for x in date_ics]
            print(f"\n  {feat_name}:")
            for d, ic, n in date_ics:
                bar = '+' * int(abs(ic) * 500) if ic > 0 else '-' * int(abs(ic) * 500)
                print(f"    {d}: IC={ic:+.5f} (n={n:5d}) {bar}")
            print(f"    Mean IC: {np.mean(ics):+.5f}, Std: {np.std(ics):.5f}, "
                  f"IC>0 rate: {np.mean(np.array(ics) > 0):.1%}")

    # Feature statistics summary
    print(f"\n{'=' * 100}")
    print("FEATURE STATISTICS")
    print(f"{'=' * 100}")
    for feat_name in sorted(concat_features.keys()):
        vals = concat_features[feat_name]
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            print(f"  {feat_name:28s}: mean={np.mean(valid):+.4f}, std={np.std(valid):.4f}, "
                  f"median={np.median(valid):+.4f}, valid={len(valid)}/{len(vals)}")

    # Save comprehensive summary
    summary = {
        'analysis_date': '2026-07-20',
        'n_dates_analyzed': len(sample_dates),
        'dates': sample_dates,
        'total_eval_points': total_points,
        'window_ns': 1_000_000_000,
        'n_significant_features': len(sig_features),
        'significant_features': [{
            'feature': r['feature'],
            'horizon': r['horizon'],
            'IC': r['IC'],
            't_stat': r['t_stat'],
        } for r in sig_features],
        'top_20': [{
            'feature': r['feature'],
            'horizon': r['horizon'],
            'IC': r['IC'],
            't_stat': r['t_stat'],
        } for r in results_sorted[:20]],
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")
    return results_sorted


if __name__ == "__main__":
    main()
