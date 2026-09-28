"""
Feature engineering from raw MBO event sequences.

Computes 17 per-bar features from the same (N_bars, 200, 5) int16 event data
stored in dl_events_cache/*.npz files.

Raw event fields (axis=-1 of each 200x5 slice):
  0: event_type    — 0=add, 1=modify, 2=cancel, 3=trade
  1: price_level   — relative to mid in half-ticks (e.g. -4 = 2 ticks below mid)
  2: size          — order size in lots
  3: time_delta_ms — milliseconds since previous event
  4: side          — 0=bid, 1=ask

Output: (N_bars, 17) float32 array of engineered features.

All computation is numpy-vectorized for speed — no scipy dependency.
"""

import numpy as np


NUM_FEATURES = 17


def _entropy(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy of a distribution (per row). No scipy needed.

    Args:
        counts: (N, K) non-negative integer counts.

    Returns:
        (N,) entropy values. Zero when total count is 0.
    """
    totals = counts.sum(axis=1, keepdims=True).astype(np.float64)
    # Avoid division by zero
    safe_totals = np.where(totals > 0, totals, 1.0)
    probs = counts.astype(np.float64) / safe_totals
    # log(0) -> 0 contribution
    log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    ent = -(probs * log_probs).sum(axis=1)
    # Zero entropy when no events
    ent = np.where(totals.squeeze() > 0, ent, 0.0)
    return ent.astype(np.float32)


def compute_event_features(
    event_sequences: np.ndarray,
    sequence_lengths: np.ndarray,
) -> np.ndarray:
    """Compute 17 engineered features per bar from raw MBO events.

    Args:
        event_sequences: (N, 200, 5) int16 — raw events per bar.
        sequence_lengths: (N,) uint16 — valid event count per bar.

    Returns:
        features: (N, 17) float32 array.

    Feature list:
        0  order_flow_imbalance
        1  trade_flow_imbalance
        2  aggressive_ratio
        3  cancel_rate_bid
        4  cancel_rate_ask
        5  cancel_imbalance
        6  large_order_ratio
        7  sweep_count
        8  depth_add_rate_bid
        9  depth_add_rate_ask
        10 replenishment_speed
        11 event_rate
        12 size_entropy
        13 price_range
        14 time_clustering
        15 volume_acceleration  (placeholder — filled by caller for rolling)
        16 cancel_rate_trend    (placeholder — filled by caller for rolling)
    """
    N, max_len, _ = event_sequences.shape
    seq = event_sequences.astype(np.float32)  # (N, 200, 5)
    lengths = sequence_lengths.astype(np.int32)  # (N,)

    # Build a validity mask: (N, 200) True where event index < length
    idx = np.arange(max_len, dtype=np.int32)[None, :]  # (1, 200)
    valid = idx < lengths[:, None]  # (N, 200)
    valid_f = valid.astype(np.float32)
    n_valid = valid_f.sum(axis=1).clip(min=1.0)  # (N,) avoid /0

    # Extract fields
    event_type = seq[:, :, 0]   # (N, 200)
    price_level = seq[:, :, 1]  # (N, 200)
    size = seq[:, :, 2]         # (N, 200)
    time_delta = seq[:, :, 3]   # (N, 200)
    side = seq[:, :, 4]         # (N, 200)

    # Zero out invalid positions
    event_type = event_type * valid_f
    price_level = price_level * valid_f
    size = size * valid_f
    time_delta = time_delta * valid_f
    side = side * valid_f

    # ===== Masks for specific event types and sides =====
    is_add    = (seq[:, :, 0] == 0) & valid   # (N, 200)
    is_modify = (seq[:, :, 0] == 1) & valid
    is_cancel = (seq[:, :, 0] == 2) & valid
    is_trade  = (seq[:, :, 0] == 3) & valid

    is_bid = (seq[:, :, 4] == 0) & valid
    is_ask = (seq[:, :, 4] == 1) & valid

    features = np.zeros((N, NUM_FEATURES), dtype=np.float32)

    # ----- 0: order_flow_imbalance -----
    # (buy_volume - sell_volume) / total_volume
    # bid side = buy, ask side = sell (for size-weighted)
    buy_vol = (size * is_bid.astype(np.float32)).sum(axis=1)
    sell_vol = (size * is_ask.astype(np.float32)).sum(axis=1)
    total_vol = buy_vol + sell_vol
    features[:, 0] = np.where(total_vol > 0, (buy_vol - sell_vol) / total_vol, 0.0)

    # ----- 1: trade_flow_imbalance -----
    # Same but only trade events
    trade_buy = (size * (is_trade & is_bid).astype(np.float32)).sum(axis=1)
    trade_sell = (size * (is_trade & is_ask).astype(np.float32)).sum(axis=1)
    trade_total = trade_buy + trade_sell
    features[:, 1] = np.where(trade_total > 0, (trade_buy - trade_sell) / trade_total, 0.0)

    # ----- 2: aggressive_ratio -----
    n_trades = is_trade.astype(np.float32).sum(axis=1)
    features[:, 2] = n_trades / n_valid

    # ----- 3: cancel_rate_bid -----
    cancel_bid = (is_cancel & is_bid).astype(np.float32).sum(axis=1)
    total_bid = is_bid.astype(np.float32).sum(axis=1)
    features[:, 3] = np.where(total_bid > 0, cancel_bid / total_bid, 0.0)

    # ----- 4: cancel_rate_ask -----
    cancel_ask = (is_cancel & is_ask).astype(np.float32).sum(axis=1)
    total_ask = is_ask.astype(np.float32).sum(axis=1)
    features[:, 4] = np.where(total_ask > 0, cancel_ask / total_ask, 0.0)

    # ----- 5: cancel_imbalance -----
    total_cancels = cancel_bid + cancel_ask
    features[:, 5] = np.where(total_cancels > 0, (cancel_bid - cancel_ask) / total_cancels, 0.0)

    # ----- 6: large_order_ratio -----
    large = ((size > 5) & valid).astype(np.float32).sum(axis=1)
    features[:, 6] = large / n_valid

    # ----- 7: sweep_count -----
    # 3+ distinct price levels hit within a 50ms window.
    # Per bar, use a simple forward scan: at each event, look ahead up to 50ms
    # and count distinct price levels. If >= 3, it's a sweep.
    # Uses cumulative time_delta for O(N) per bar via prefix sums.
    sweep_counts = np.zeros(N, dtype=np.float32)
    for bar in range(N):
        n_ev = int(lengths[bar])
        if n_ev < 3:
            continue
        pl = event_sequences[bar, :n_ev, 1]  # int16
        td = event_sequences[bar, :n_ev, 3].astype(np.float64)
        # Prefix sum of time_deltas: cum[i] = sum(td[1:i+1])
        cum = np.zeros(n_ev + 1, dtype=np.float64)
        cum[1:] = np.cumsum(td)
        # Note: td[0] is time since start of bar, td[i] for i>0 is gap from i-1 to i.
        # Time span from event i to event j = cum[j+1] - cum[i+1]
        # (since cum[k+1] = sum of td[0..k])
        count = 0
        i = 0
        while i < n_ev - 2:
            # Find rightmost j such that time from i to j <= 50ms
            # time from i to j = cum[j+1] - cum[i+1] (ignoring td[i] itself)
            # Actually: elapsed between event i and event j = sum(td[i+1]..td[j])
            # = cum[j+1] - cum[i+1]
            levels = set()
            levels.add(int(pl[i]))
            found = False
            for j in range(i + 1, n_ev):
                elapsed = cum[j + 1] - cum[i + 1]
                if elapsed > 50.0:
                    break
                levels.add(int(pl[j]))
                if len(levels) >= 3:
                    count += 1
                    i = j + 1  # skip past this sweep
                    found = True
                    break
            if not found:
                i += 1
        sweep_counts[bar] = count
    features[:, 7] = sweep_counts

    # ----- 8: depth_add_rate_bid -----
    # Add events at best 3 bid levels (price_level in [-3, -2, -1] assuming
    # negative = bid side in half-ticks from mid)
    # Actually: price_level < 0 is bid side, best 3 = price_level in {-1, -2, -3}
    best_bid_mask = (price_level >= -3) & (price_level < 0)
    add_at_best_bid = (is_add & best_bid_mask).astype(np.float32).sum(axis=1)
    features[:, 8] = add_at_best_bid / n_valid

    # ----- 9: depth_add_rate_ask -----
    best_ask_mask = (price_level > 0) & (price_level <= 3)
    add_at_best_ask = (is_add & best_ask_mask).astype(np.float32).sum(axis=1)
    features[:, 9] = add_at_best_ask / n_valid

    # ----- 10: replenishment_speed -----
    # Average time_delta after cancel events at best levels (|price_level| <= 3)
    best_level_mask = (np.abs(price_level) <= 3) & valid
    cancel_at_best = is_cancel & best_level_mask  # (N, 200)
    # For each cancel at best level, look at the next event's time_delta
    # Shift cancel mask left by 1 to get "event after cancel"
    cancel_shifted = np.zeros_like(cancel_at_best)
    cancel_shifted[:, :-1] = cancel_at_best[:, 1:]
    # time_delta of events immediately following cancels at best levels
    replenish_times = time_delta * cancel_shifted.astype(np.float32)
    replenish_count = cancel_shifted.astype(np.float32).sum(axis=1).clip(min=1.0)
    features[:, 10] = replenish_times.sum(axis=1) / replenish_count

    # ----- 11: event_rate -----
    # total_events / time_span_ms
    time_span = time_delta.sum(axis=1).clip(min=1.0)
    features[:, 11] = n_valid / time_span

    # ----- 12: size_entropy -----
    # Entropy of order size distribution. Bucket sizes: [1, 2, 3, 4, 5+]
    size_clamped = np.clip(size * valid_f, 0, 5).astype(np.int32)  # (N, 200)
    # Build histogram per bar (6 bins: 0,1,2,3,4,5)
    size_hist = np.zeros((N, 6), dtype=np.int32)
    for b in range(6):
        size_hist[:, b] = ((size_clamped == b) & valid).sum(axis=1)
    # Drop bin 0 (unused/padding sizes)
    features[:, 12] = _entropy(size_hist[:, 1:])

    # ----- 13: price_range -----
    # Use masked min/max. Set invalid positions to 0, then compute.
    price_masked_high = np.where(valid, price_level, -9999.0)
    price_masked_low = np.where(valid, price_level, 9999.0)
    p_max = price_masked_high.max(axis=1)
    p_min = price_masked_low.min(axis=1)
    features[:, 13] = np.where(n_valid > 1, p_max - p_min, 0.0)

    # ----- 14: time_clustering -----
    # Std of time_deltas (bursty vs uniform arrival)
    # Mean per bar
    td_mean = time_delta.sum(axis=1) / n_valid
    td_diff = (time_delta - td_mean[:, None]) * valid_f
    td_var = (td_diff ** 2).sum(axis=1) / n_valid.clip(min=2.0)
    features[:, 14] = np.sqrt(td_var)

    # ----- 15: volume_acceleration (rolling, computed below) -----
    # ----- 16: cancel_rate_trend (rolling, computed below) -----
    # These require multi-bar context. Compute them here using a rolling approach.

    # Volume per bar (total size of all valid events)
    bar_volume = (size * valid_f).sum(axis=1)  # (N,)

    # Cancel rate per bar (total cancels / total events)
    bar_cancel_rate = (is_cancel.astype(np.float32).sum(axis=1)) / n_valid  # (N,)

    # 15: volume_acceleration = current bar volume / avg of last 10 bars
    # Use cumsum for efficient rolling average
    vol_cumsum = np.cumsum(bar_volume)
    vol_avg_10 = np.zeros(N, dtype=np.float32)
    for i in range(N):
        lookback = min(i, 10)
        if lookback > 0:
            start_sum = vol_cumsum[i - lookback] if (i - lookback) > 0 else 0.0
            vol_avg_10[i] = (vol_cumsum[i] - start_sum) / lookback
        else:
            vol_avg_10[i] = bar_volume[i] if bar_volume[i] > 0 else 1.0
    vol_avg_10 = np.where(vol_avg_10 > 0, vol_avg_10, 1.0)
    features[:, 15] = bar_volume / vol_avg_10

    # 16: cancel_rate_trend = change in cancel_rate over last 5 bars
    # (current cancel_rate - cancel_rate 5 bars ago)
    cancel_rate_5ago = np.zeros(N, dtype=np.float32)
    cancel_rate_5ago[5:] = bar_cancel_rate[:-5]
    cancel_rate_5ago[:5] = bar_cancel_rate[:5]
    features[:, 16] = bar_cancel_rate - cancel_rate_5ago

    return features


def compute_event_features_for_dataset(
    day_data_list: list,
) -> np.ndarray:
    """Convenience: compute features from a list of day dicts (as used in BarDataset).

    Each dict should have 'event_sequences' and 'sequence_lengths' keys.

    Returns:
        (N_total, 17) float32 array, concatenated across days.
    """
    all_feats = []
    for d in day_data_list:
        feats = compute_event_features(d['event_sequences'], d['sequence_lengths'])
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


if __name__ == '__main__':
    # Quick test with synthetic data
    N = 100
    seq = np.random.randint(0, 4, (N, 200, 5), dtype=np.int16)
    seq[:, :, 3] = np.random.randint(1, 100, (N, 200), dtype=np.int16)  # time deltas
    seq[:, :, 4] = np.random.randint(0, 2, (N, 200), dtype=np.int16)    # side
    seq[:, :, 2] = np.random.randint(1, 10, (N, 200), dtype=np.int16)   # size
    lengths = np.random.randint(50, 200, N, dtype=np.uint16)

    feats = compute_event_features(seq, lengths)
    print(f"Feature shape: {feats.shape}")  # (100, 17)
    print(f"Feature means: {feats.mean(axis=0)}")
    print(f"Feature stds:  {feats.std(axis=0)}")
    print(f"Any NaN: {np.isnan(feats).any()}")
