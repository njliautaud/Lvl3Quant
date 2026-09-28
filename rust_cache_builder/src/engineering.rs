/// Feature Engineering — Rust port of src/features/engineering.py
///
/// Extracts 96 global features and (20, 9) node features from BookSnapshot.
/// Output is float32 to match Python's np.float32 dtype.

use crate::lob::BookSnapshot;

pub const DEPTH_LEVELS: usize = 10;

// Feature dimension constants (must sum to 96)
pub const BASE_GLOBAL: usize = 10;
pub const ORDER_FLOW: usize = 8;
pub const MICROSTRUCTURE: usize = 8;
pub const TEMPORAL: usize = 5;
pub const MBO_ENHANCED: usize = 14;
pub const PHASE2_EVENT: usize = 25;
pub const PHASE3_BOOK_SHAPE: usize = 15;
pub const PHASE3_QUOTE_DYNAMICS: usize = 11;
pub const TOTAL_GLOBAL: usize = BASE_GLOBAL + ORDER_FLOW + MICROSTRUCTURE + TEMPORAL
    + MBO_ENHANCED + PHASE2_EVENT + PHASE3_BOOK_SHAPE + PHASE3_QUOTE_DYNAMICS; // = 96

pub const NODE_FEATURES: usize = 9;
pub const NUM_NODES: usize = DEPTH_LEVELS * 2; // 10 bid + 10 ask

/// Compute all features from a snapshot.
/// Returns (node_features: [NUM_NODES][NODE_FEATURES], global_features: [TOTAL_GLOBAL])
pub fn compute_features(
    snapshot: &BookSnapshot,
    _depth_levels: usize,  // Ignored; compile-time constant DEPTH_LEVELS is used
) -> ([f32; TOTAL_GLOBAL], [[f32; NODE_FEATURES]; NUM_NODES]) {
    let depth_levels = DEPTH_LEVELS;
    let mid = if snapshot.mid.is_finite() { snapshot.mid } else { 0.0 };

    // Pad bid/ask arrays to depth_levels
    let bid_prices = pad_prices(&snapshot.bid_prices, depth_levels, mid);
    let bid_sizes = pad_sizes(&snapshot.bid_sizes, depth_levels, 0.0);
    let ask_prices = pad_prices(&snapshot.ask_prices, depth_levels, mid);
    let ask_sizes = pad_sizes(&snapshot.ask_sizes, depth_levels, 0.0);

    // Pad order counts
    let bid_oc = pad_sizes(&snapshot.bid_order_counts, depth_levels, 0.0);
    let ask_oc = pad_sizes(&snapshot.ask_order_counts, depth_levels, 0.0);

    // Node features
    let total_bid: f64 = bid_sizes.iter().sum();
    let total_ask: f64 = ask_sizes.iter().sum();
    let bid_nodes = build_side_nodes(&bid_prices, &bid_sizes, &bid_oc, mid, 1.0, total_bid);
    let ask_nodes = build_side_nodes(&ask_prices, &ask_sizes, &ask_oc, mid, -1.0, total_ask);

    let mut node_features = [[0.0f32; NODE_FEATURES]; NUM_NODES];
    for i in 0..depth_levels {
        for j in 0..NODE_FEATURES {
            node_features[i][j] = bid_nodes[i][j];
            node_features[depth_levels + i][j] = ask_nodes[i][j];
        }
    }

    // Global features
    let mut global = [0.0f32; TOTAL_GLOBAL];
    let mut idx = 0;

    // Base (10)
    let base = build_global_features(snapshot, &bid_prices, &bid_sizes, &ask_prices, &ask_sizes);
    for v in base { global[idx] = v; idx += 1; }

    // Order flow (8)
    let of = build_order_flow_features(snapshot);
    for v in of { global[idx] = v; idx += 1; }

    // Microstructure (8)
    let ms = build_microstructure_features(snapshot, &bid_prices, &bid_sizes, &ask_prices, &ask_sizes);
    for v in ms { global[idx] = v; idx += 1; }

    // Temporal (5)
    let temp = build_temporal_features(snapshot.timestamp_ns, snapshot);
    for v in temp { global[idx] = v; idx += 1; }

    // MBO enhanced (14)
    let mbo = build_mbo_enhanced_features(snapshot);
    for v in mbo { global[idx] = v; idx += 1; }

    // Phase 2 (25)
    let p2 = build_phase2_event_features(snapshot);
    for v in p2 { global[idx] = v; idx += 1; }

    // Phase 3 book shape (15)
    let bs = build_book_shape_features(snapshot, &bid_prices, &bid_sizes, &ask_prices, &ask_sizes, &bid_oc, &ask_oc);
    for v in bs { global[idx] = v; idx += 1; }

    // Phase 3 quote dynamics (11)
    let qd = build_quote_dynamics_features(snapshot);
    for v in qd { global[idx] = v; idx += 1; }

    debug_assert_eq!(idx, TOTAL_GLOBAL, "Feature count mismatch: got {}, expected {}", idx, TOTAL_GLOBAL);

    (global, node_features)
}

// ============================================================================
// Node features
// ============================================================================

fn build_side_nodes(
    prices: &[f64],
    sizes: &[f64],
    order_counts: &[f64],
    mid: f64,
    side: f64,
    total_side_vol: f64,
) -> Vec<[f32; NODE_FEATURES]> {
    let n = prices.len();
    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let price = prices[i];
        let size = sizes[i];
        let oc = order_counts[i];

        let rel_price = (price - mid) / if mid != 0.0 { mid } else { 1.0 };
        let size_log = (1.0 + size).ln();
        let level_idx = if n <= 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
        let avg_order_size = if oc > 0.0 { size / oc } else { 0.0 };
        let level_conc = size / total_side_vol.max(1.0);

        nodes.push([
            price as f32,
            rel_price as f32,
            size as f32,
            size_log as f32,
            level_idx as f32,
            side as f32,
            oc as f32,
            avg_order_size as f32,
            level_conc as f32,
        ]);
    }
    nodes
}

// ============================================================================
// Global feature groups
// ============================================================================

fn build_global_features(
    snapshot: &BookSnapshot,
    bid_prices: &[f64],
    bid_sizes: &[f64],
    ask_prices: &[f64],
    ask_sizes: &[f64],
) -> [f32; BASE_GLOBAL] {
    let total_bid: f64 = bid_sizes.iter().sum();
    let total_ask: f64 = ask_sizes.iter().sum();
    let imbalance = (total_bid - total_ask) / (total_bid + total_ask).max(1.0);

    let microprice = compute_microprice(bid_prices, bid_sizes, ask_prices, ask_sizes);
    let mid = if snapshot.mid.is_finite() { snapshot.mid } else { 0.0 };
    let spread = if snapshot.spread.is_finite() { snapshot.spread } else { 0.0 };
    let best_bid = bid_prices.first().copied().unwrap_or(0.0);
    let best_ask = ask_prices.first().copied().unwrap_or(0.0);
    let mean_bid = if bid_sizes.is_empty() { 0.0 } else { total_bid / bid_sizes.len() as f64 };
    let mean_ask = if ask_sizes.is_empty() { 0.0 } else { total_ask / ask_sizes.len() as f64 };

    [
        mid as f32,
        spread as f32,
        imbalance as f32,
        microprice as f32,
        total_bid as f32,
        total_ask as f32,
        mean_bid as f32,
        mean_ask as f32,
        best_bid as f32,
        best_ask as f32,
    ]
}

fn build_order_flow_features(snapshot: &BookSnapshot) -> [f32; ORDER_FLOW] {
    let buy_vol = snapshot.buy_volume;
    let sell_vol = snapshot.sell_volume;
    let total_trade_vol = buy_vol + sell_vol;
    let trade_imbalance = (buy_vol - sell_vol) / total_trade_vol.max(1.0);

    let add = snapshot.add_count as f64;
    let cancel = snapshot.cancel_count as f64;
    let trade = snapshot.trade_count as f64;
    let cancel_to_add = cancel / add.max(1.0);
    let trade_to_add = trade / add.max(1.0);

    [
        trade_imbalance as f32,
        buy_vol as f32,
        sell_vol as f32,
        add as f32,
        cancel as f32,
        trade as f32,
        cancel_to_add as f32,
        trade_to_add as f32,
    ]
}

fn build_microstructure_features(
    snapshot: &BookSnapshot,
    bid_prices: &[f64],
    bid_sizes: &[f64],
    ask_prices: &[f64],
    ask_sizes: &[f64],
) -> [f32; MICROSTRUCTURE] {
    let mid = if snapshot.mid.is_finite() { snapshot.mid } else { 0.0 };
    let tick_size = 0.25_f64;

    let bid_pressure: f64 = bid_prices.iter().zip(bid_sizes.iter())
        .map(|(&p, &s)| s / ((mid - p).abs() + tick_size))
        .sum();
    let ask_pressure: f64 = ask_prices.iter().zip(ask_sizes.iter())
        .map(|(&p, &s)| s / ((p - mid).abs() + tick_size))
        .sum();
    let total_pressure = bid_pressure + ask_pressure;
    let pressure_imbalance = (bid_pressure - ask_pressure) / total_pressure.max(1.0);

    let total_bid: f64 = bid_sizes.iter().sum();
    let total_ask: f64 = ask_sizes.iter().sum();
    let best_bid_size = bid_sizes.first().copied().unwrap_or(0.0);
    let best_ask_size = ask_sizes.first().copied().unwrap_or(0.0);
    let bid_conc = best_bid_size / total_bid.max(1.0);
    let ask_conc = best_ask_size / total_ask.max(1.0);
    let depth_concentration = (bid_conc + ask_conc) / 2.0;

    let bid_slope = compute_depth_slope(bid_sizes);
    let ask_slope = compute_depth_slope(ask_sizes);
    let spread = if snapshot.spread.is_finite() { snapshot.spread } else { 0.0 };
    let spread_ticks = spread / tick_size;

    let bid_filled = bid_sizes.iter().filter(|&&s| s > 0.0).count() as f64 / bid_sizes.len().max(1) as f64;
    let ask_filled = ask_sizes.iter().filter(|&&s| s > 0.0).count() as f64 / ask_sizes.len().max(1) as f64;
    let depth_ratio = (bid_filled + ask_filled) / 2.0;

    [
        bid_pressure as f32,
        ask_pressure as f32,
        pressure_imbalance as f32,
        depth_concentration as f32,
        bid_slope as f32,
        ask_slope as f32,
        spread_ticks as f32,
        depth_ratio as f32,
    ]
}

fn build_temporal_features(timestamp_ns: u64, snapshot: &BookSnapshot) -> [f32; TEMPORAL] {
    if timestamp_ns == 0 { return [0.0f32; TEMPORAL]; }

    // Convert ns to seconds
    let ts_sec = timestamp_ns as f64 / 1e9;
    // UTC time components (simple integer math, no timezone library needed for CTish)
    let total_secs = ts_sec as u64;
    let hours_utc = (total_secs / 3600) % 24;
    let minutes = (total_secs / 60) % 60;

    // Convert UTC to CT: CT = UTC - 5 (CST) or UTC - 6 (CDT)
    // Python uses UTC-6, we'll use same constant for now (matches Python's hour_ct)
    let hour_ct = (hours_utc as i64 - 6).rem_euclid(24) as f64;

    // RTH: 8:30 AM CT = 510 minutes, close 15:15 CT = 915 minutes
    let rth_open_min = 8.0 * 60.0 + 30.0;
    let rth_duration = (15.0 * 60.0 + 15.0) - rth_open_min;
    let current_min = hour_ct * 60.0 + minutes as f64;

    let time_since_rth = ((current_min - rth_open_min) / rth_duration).clamp(-1.0, 1.0);
    let time_to_close = (1.0 - time_since_rth).clamp(0.0, 2.0);

    // Event density: log1p(events_per_sec) — matches Python
    let total_events = (snapshot.add_count + snapshot.cancel_count + snapshot.trade_count) as f64;
    let interval_sec = 0.1; // 100ms
    let events_per_sec = total_events / interval_sec;
    let event_density = (1.0 + events_per_sec).ln();

    [
        (hour_ct / 24.0) as f32,
        (minutes as f64 / 60.0) as f32,
        time_since_rth as f32,
        time_to_close as f32,
        event_density as f32,
    ]
}

fn build_mbo_enhanced_features(snapshot: &BookSnapshot) -> [f32; MBO_ENHANCED] {
    let modify_count = snapshot.modify_count as f64;
    let add_count = snapshot.add_count as f64;
    let modify_to_add = modify_count / add_count.max(1.0);

    // log1p-transformed lifetime (matches Python fix)
    let mean_lifetime_ms = (1.0 + snapshot.mean_lifetime_ms).ln();
    let fleeting_ratio = snapshot.fleeting_ratio;

    let aggr_buy = snapshot.aggressive_buy_count as f64;
    let aggr_sell = snapshot.aggressive_sell_count as f64;
    let total_aggr = aggr_buy + aggr_sell;
    let aggressive_imbalance = (aggr_buy - aggr_sell) / total_aggr.max(1.0);

    let max_trade_norm = (snapshot.max_trade_size / 10.0).min(50.0);

    let cancel_bid = snapshot.cancel_bid_vol;
    let cancel_ask = snapshot.cancel_ask_vol;
    let total_cancel = cancel_bid + cancel_ask;
    let cancel_side_imbalance = (cancel_bid - cancel_ask) / total_cancel.max(1.0);

    let tick_count = snapshot.tick_count as f64;
    let seq_gaps = snapshot.sequence_gaps as f64;
    let n_completed = snapshot.n_orders_completed as f64;

    [
        modify_count as f32,
        modify_to_add as f32,
        mean_lifetime_ms as f32,
        fleeting_ratio as f32,
        aggr_buy as f32,
        aggr_sell as f32,
        aggressive_imbalance as f32,
        max_trade_norm as f32,
        cancel_bid as f32,
        cancel_ask as f32,
        cancel_side_imbalance as f32,
        tick_count as f32,
        seq_gaps as f32,
        n_completed as f32,
    ]
}

fn build_phase2_event_features(snapshot: &BookSnapshot) -> [f32; PHASE2_EVENT] {
    let trade_count = snapshot.trade_count.max(1) as f64;
    let cancel_count = snapshot.cancel_count.max(1) as f64;
    let add_count = snapshot.add_count.max(1) as f64;
    let modify_count = snapshot.modify_count.max(1) as f64;

    // Trade size distribution (4)
    let large_trade_ratio = snapshot.n_large_trades as f64 / trade_count;
    let small_trade_ratio = snapshot.n_small_trades as f64 / trade_count;
    let trade_size_p75_log = (1.0 + snapshot.trade_size_p75).ln();
    let trade_heterogeneity = (snapshot.n_large_trades + snapshot.n_small_trades) as f64 / trade_count;

    // Cancel level structure (5)
    let cancel_l1_ratio = snapshot.cancel_at_l1_count as f64 / cancel_count;
    let total_cancel_vol = snapshot.cancel_at_l1_vol + snapshot.cancel_deep_vol;
    let cancel_l1_vol_frac = snapshot.cancel_at_l1_vol / total_cancel_vol.max(1.0);
    let cancel_deep_vol_frac = snapshot.cancel_deep_vol / total_cancel_vol.max(1.0);
    let add_l1_ratio = snapshot.add_at_l1_count as f64 / add_count;
    let l1_total = (snapshot.add_at_l1_count + snapshot.cancel_at_l1_count) as f64;
    let l1_net_activity = (snapshot.add_at_l1_count as f64 - snapshot.cancel_at_l1_count as f64) / l1_total.max(1.0);

    // Sweep signature (4)
    let consec_buy_log = (1.0 + snapshot.max_consec_buy_trades as f64).ln();
    let consec_sell_log = (1.0 + snapshot.max_consec_sell_trades as f64).ln();
    let max_streak = snapshot.max_consec_buy_trades.max(snapshot.max_consec_sell_trades) as f64;
    let total_streak = (snapshot.max_consec_buy_trades + snapshot.max_consec_sell_trades) as f64;
    let sweep_asymmetry = (snapshot.max_consec_buy_trades as f64 - snapshot.max_consec_sell_trades as f64)
        / total_streak.max(1.0);
    let sweep_intensity = max_streak / trade_count;

    // Trade timing (4)
    let mean_inter_trade_log = (1.0 + snapshot.mean_inter_trade_ms).ln();
    let min_inter_trade_log = (1.0 + snapshot.min_inter_trade_ms).ln();
    let trade_burstiness = if snapshot.mean_inter_trade_ms > 0.0 {
        snapshot.min_inter_trade_ms / snapshot.mean_inter_trade_ms.max(0.1)
    } else { 0.0 };
    let trade_rate = if snapshot.mean_inter_trade_ms > 0.0 {
        (1.0 + 1000.0 / snapshot.mean_inter_trade_ms.max(0.1)).ln()
    } else { 0.0 };

    // Modify chains (3)
    let modify_chain_ratio = snapshot.n_modify_chains as f64 / modify_count;
    let max_chain_log = (1.0 + snapshot.max_modify_chain_len as f64).ln();
    let chain_per_order = snapshot.n_modify_chains as f64 / add_count;

    // Multi-level trades (2)
    let multi_level_ratio = snapshot.trades_through_multiple_levels as f64 / trade_count;
    let multi_level_log = (1.0 + snapshot.trades_through_multiple_levels as f64).ln();

    // Event × context interactions (3)
    let large_sweep_combo = large_trade_ratio * sweep_intensity;
    let cancel_l1_trade_press = snapshot.cancel_at_l1_count as f64 / trade_count;
    let modify_chain_sweep = modify_chain_ratio * sweep_intensity;

    [
        // Trade size distribution (4)
        large_trade_ratio as f32,
        small_trade_ratio as f32,
        trade_size_p75_log as f32,
        trade_heterogeneity as f32,
        // Cancel level structure (5)
        cancel_l1_ratio as f32,
        cancel_l1_vol_frac as f32,
        cancel_deep_vol_frac as f32,
        add_l1_ratio as f32,
        l1_net_activity as f32,
        // Sweep signature (4)
        consec_buy_log as f32,
        consec_sell_log as f32,
        sweep_asymmetry as f32,
        sweep_intensity as f32,
        // Trade timing (4)
        mean_inter_trade_log as f32,
        min_inter_trade_log as f32,
        trade_burstiness as f32,
        trade_rate as f32,
        // Modify chains (3)
        modify_chain_ratio as f32,
        max_chain_log as f32,
        chain_per_order as f32,
        // Multi-level trades (2)
        multi_level_ratio as f32,
        multi_level_log as f32,
        // Interactions (3)
        large_sweep_combo as f32,
        cancel_l1_trade_press as f32,
        modify_chain_sweep as f32,
    ]
}

fn build_book_shape_features(
    snapshot: &BookSnapshot,
    bid_prices: &[f64],
    bid_sizes: &[f64],
    ask_prices: &[f64],
    ask_sizes: &[f64],
    bid_oc: &[f64],
    ask_oc: &[f64],
) -> [f32; PHASE3_BOOK_SHAPE] {
    let tick_size = 0.25_f64;
    let mid = if snapshot.mid.is_finite() { snapshot.mid } else { 0.0 };

    let bid_depth_skew = safe_skewness(bid_sizes);
    let ask_depth_skew = safe_skewness(ask_sizes);

    // Gap count
    let bid_gap_count = if bid_prices.len() >= 2 {
        let expected = ((bid_prices[0] - bid_prices[bid_prices.len() - 1]) / tick_size).round() as i64;
        (expected - bid_prices.len() as i64 + 1).max(0) as f64
    } else { 0.0 };
    let ask_gap_count = if ask_prices.len() >= 2 {
        let expected = ((ask_prices[ask_prices.len() - 1] - ask_prices[0]) / tick_size).round() as i64;
        (expected - ask_prices.len() as i64 + 1).max(0) as f64
    } else { 0.0 };

    let total_bid: f64 = bid_sizes.iter().sum();
    let total_ask: f64 = ask_sizes.iter().sum();
    let max_bid_wall_frac = bid_sizes.iter().cloned().fold(0.0_f64, f64::max) / total_bid.max(1.0);
    let max_ask_wall_frac = ask_sizes.iter().cloned().fold(0.0_f64, f64::max) / total_ask.max(1.0);

    // Center of gravity
    let bid_depth_cog = if !bid_prices.is_empty() && mid > 0.0 && total_bid > 0.0 {
        bid_prices.iter().zip(bid_sizes.iter())
            .map(|(&p, &s)| s * (mid - p).abs() / tick_size)
            .sum::<f64>() / total_bid
    } else { 0.0 };
    let ask_depth_cog = if !ask_prices.is_empty() && mid > 0.0 && total_ask > 0.0 {
        ask_prices.iter().zip(ask_sizes.iter())
            .map(|(&p, &s)| s * (p - mid).abs() / tick_size)
            .sum::<f64>() / total_ask
    } else { 0.0 };

    let bid_size_entropy = safe_entropy(bid_sizes);
    let ask_size_entropy = safe_entropy(ask_sizes);

    // Book symmetry (correlation)
    let min_len = bid_sizes.len().min(ask_sizes.len());
    let book_symmetry = if min_len >= 3 {
        let bc = &bid_sizes[..min_len];
        let ac = &ask_sizes[..min_len];
        let bc_mean = bc.iter().sum::<f64>() / min_len as f64;
        let ac_mean = ac.iter().sum::<f64>() / min_len as f64;
        let bc_c: Vec<f64> = bc.iter().map(|&x| x - bc_mean).collect();
        let ac_c: Vec<f64> = ac.iter().map(|&x| x - ac_mean).collect();
        let num: f64 = bc_c.iter().zip(ac_c.iter()).map(|(a, b)| a * b).sum();
        let denom = (bc_c.iter().map(|x| x * x).sum::<f64>()
            * ac_c.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if denom > 1e-10 { num / denom } else { 0.0 }
    } else { 0.0 };

    let total_depth_log = (1.0 + total_bid + total_ask).ln();

    let l1_l3_bid = if bid_sizes.len() >= 3 {
        bid_sizes[0] / bid_sizes[..3].iter().sum::<f64>().max(1.0)
    } else if !bid_sizes.is_empty() { 1.0 } else { 0.0 };
    let l1_l3_ask = if ask_sizes.len() >= 3 {
        ask_sizes[0] / ask_sizes[..3].iter().sum::<f64>().max(1.0)
    } else if !ask_sizes.is_empty() { 1.0 } else { 0.0 };

    let bid_oc_entropy = if !bid_oc.is_empty() { safe_entropy(bid_oc) } else { 0.0 };
    let ask_oc_entropy = if !ask_oc.is_empty() { safe_entropy(ask_oc) } else { 0.0 };
    let total_oc_ent = bid_oc_entropy + ask_oc_entropy;
    let order_frag_asym = (bid_oc_entropy - ask_oc_entropy) / total_oc_ent.max(0.01);

    [
        bid_depth_skew as f32,
        ask_depth_skew as f32,
        bid_gap_count as f32,
        ask_gap_count as f32,
        max_bid_wall_frac as f32,
        max_ask_wall_frac as f32,
        bid_depth_cog as f32,
        ask_depth_cog as f32,
        bid_size_entropy as f32,
        ask_size_entropy as f32,
        book_symmetry as f32,
        total_depth_log as f32,
        l1_l3_bid as f32,
        l1_l3_ask as f32,
        order_frag_asym as f32,
    ]
}

fn build_quote_dynamics_features(snapshot: &BookSnapshot) -> [f32; PHASE3_QUOTE_DYNAMICS] {
    let total_events = (snapshot.add_count + snapshot.cancel_count + snapshot.trade_count).max(1) as f64;
    let trade_count = snapshot.trade_count.max(1) as f64;

    let l1_bid_change_rate = snapshot.l1_bid_changes as f64 / total_events;
    let l1_ask_change_rate = snapshot.l1_ask_changes as f64 / total_events;
    let total_changes = (snapshot.l1_bid_changes + snapshot.l1_ask_changes) as f64;
    let l1_change_asymmetry = (snapshot.l1_bid_changes as f64 - snapshot.l1_ask_changes as f64)
        / total_changes.max(1.0);

    let total_depletions = snapshot.l1_bid_depleted + snapshot.l1_ask_depleted;
    let l1_depletion_log = (1.0 + total_depletions as f64).ln();
    let depletion_asymmetry = if total_depletions > 0 {
        (snapshot.l1_bid_depleted as f64 - snapshot.l1_ask_depleted as f64)
            / total_depletions as f64
    } else { 0.0 };

    let total_trades_side = (snapshot.n_trades_at_bid + snapshot.n_trades_at_ask).max(1) as f64;
    let trade_at_bid_frac = snapshot.n_trades_at_bid as f64 / total_trades_side;

    let avg_trade_size = (snapshot.buy_volume + snapshot.sell_volume) / trade_count;
    let trade_size_cv = if snapshot.trade_size_std > 0.0 {
        snapshot.trade_size_std / avg_trade_size.max(0.1)
    } else { 0.0 };
    let trade_size_skew = snapshot.trade_size_skew;

    let cancel_count = snapshot.cancel_count.max(1) as f64;
    let reposition_rate = snapshot.reposition_count as f64 / cancel_count;

    let mean_add_size_log = (1.0 + snapshot.mean_order_size_at_add).ln();
    let max_add_norm = if snapshot.mean_order_size_at_add > 0.0 {
        (snapshot.max_order_size_at_add / snapshot.mean_order_size_at_add.max(0.1)).min(50.0)
    } else { 0.0 };

    [
        l1_bid_change_rate as f32,
        l1_ask_change_rate as f32,
        l1_change_asymmetry as f32,
        l1_depletion_log as f32,
        depletion_asymmetry as f32,
        trade_at_bid_frac as f32,
        trade_size_cv as f32,
        trade_size_skew as f32,
        reposition_rate as f32,
        mean_add_size_log as f32,
        max_add_norm as f32,
    ]
}

// ============================================================================
// Helper functions
// ============================================================================

fn compute_microprice(
    bid_prices: &[f64],
    bid_sizes: &[f64],
    ask_prices: &[f64],
    ask_sizes: &[f64],
) -> f64 {
    if bid_prices.is_empty() || ask_prices.is_empty() { return 0.0; }
    let bid = bid_prices[0];
    let ask = ask_prices[0];
    let bid_sz = bid_sizes[0];
    let ask_sz = ask_sizes[0];
    let denom = bid_sz + ask_sz;
    if denom == 0.0 { (bid + ask) / 2.0 } else { (bid * ask_sz + ask * bid_sz) / denom }
}

fn compute_depth_slope(sizes: &[f64]) -> f64 {
    if sizes.len() < 2 { return 0.0; }
    let n = sizes.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = sizes.iter().sum::<f64>() / n;
    let numerator: f64 = sizes.iter().enumerate()
        .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
        .sum();
    let denominator: f64 = (0..sizes.len())
        .map(|i| (i as f64 - x_mean).powi(2))
        .sum();
    if denominator == 0.0 { 0.0 } else { (numerator / denominator) / y_mean.max(1.0) }
}

fn safe_skewness(arr: &[f64]) -> f64 {
    if arr.len() < 3 { return 0.0; }
    let mean = arr.iter().sum::<f64>() / arr.len() as f64;
    let variance = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / arr.len() as f64;
    let std = variance.sqrt();
    if std < 1e-10 { return 0.0; }
    arr.iter().map(|&x| ((x - mean) / std).powi(3)).sum::<f64>() / arr.len() as f64
}

fn safe_entropy(arr: &[f64]) -> f64 {
    if arr.len() < 2 { return 0.0; }
    let total: f64 = arr.iter().sum();
    if total < 1e-10 { return 0.0; }
    arr.iter()
        .map(|&x| x / total)
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

fn pad_prices(src: &[f64], n: usize, fill: f64) -> Vec<f64> {
    let mut v = src[..src.len().min(n)].to_vec();
    while v.len() < n { v.push(fill); }
    v
}

fn pad_sizes(src: &[f64], n: usize, fill: f64) -> Vec<f64> {
    let mut v = src[..src.len().min(n)].to_vec();
    while v.len() < n { v.push(fill); }
    v
}
