/// Enhanced Level 3 Order Book — Full MBO metadata extraction.
///
/// Direct Rust port of src/data/lob.py from Lvl3Quant.
/// Tracks every piece of information available from Databento MBO events.

use std::collections::{BTreeMap, HashMap, HashSet};
use ordered_float::OrderedFloat;

// ============================================================================
// Action / Side constants (match Python's sets)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Add,
    Cancel,
    Modify,
    Trade,
    Fill,
    Clear,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
    Unknown,
}

pub fn parse_action(a: u8) -> Action {
    // Databento uses ASCII byte values: A=65, C=67, M=77, T=84, F=70, R=82
    match a {
        b'A' | b'a' => Action::Add,
        b'C' | b'c' => Action::Cancel,
        b'M' | b'm' => Action::Modify,
        b'T' | b't' => Action::Trade,
        b'F' | b'f' => Action::Fill,
        b'R' | b'r' => Action::Clear,
        _ => Action::Unknown,
    }
}

pub fn parse_side(s: u8) -> Side {
    match s {
        b'B' | b'b' => Side::Bid,
        b'A' | b'a' => Side::Ask,
        _ => Side::Unknown,
    }
}

// Databento RecordFlags bit positions
const F_MAYBE_BAD_BOOK: u8 = 4;
const F_BAD_TS_RECV: u8 = 8;
const F_SNAPSHOT: u8 = 32;
const F_LAST: u8 = 128;

// ============================================================================
// BookSnapshot — matches Python's BookSnapshot dataclass
// ============================================================================

#[derive(Debug, Clone)]
pub struct BookSnapshot {
    // Core book state
    pub bid_prices: Vec<f64>,
    pub bid_sizes: Vec<f64>,
    pub ask_prices: Vec<f64>,
    pub ask_sizes: Vec<f64>,
    pub mid: f64,
    pub spread: f64,

    // Timestamp
    pub timestamp_ns: u64,

    // Event counts (since last snapshot)
    pub add_count: u32,
    pub cancel_count: u32,
    pub trade_count: u32,
    pub modify_count: u32,
    pub buy_volume: f64,
    pub sell_volume: f64,

    // Aggressive trade detail
    pub aggressive_buy_count: u32,
    pub aggressive_sell_count: u32,
    pub max_trade_size: f64,

    // Cancel by side
    pub cancel_bid_vol: f64,
    pub cancel_ask_vol: f64,

    // Order lifetime statistics
    pub mean_lifetime_ms: f64,
    pub median_lifetime_ms: f64,
    pub fleeting_ratio: f64,
    pub n_orders_completed: u32,

    // Per-level order counts
    pub bid_order_counts: Vec<f64>,
    pub ask_order_counts: Vec<f64>,

    // Data quality (from flags)
    pub bad_book_events: u32,
    pub bad_ts_events: u32,
    pub snapshot_init_events: u32,

    // Sequence tracking
    pub sequence_gaps: u32,

    // Batch/tick counting
    pub tick_count: u32,

    // Phase 2: Enhanced MBO fields
    pub n_large_trades: u32,
    pub n_small_trades: u32,
    pub trade_size_p75: f64,
    pub cancel_at_l1_count: u32,
    pub cancel_at_l1_vol: f64,
    pub cancel_deep_count: u32,
    pub cancel_deep_vol: f64,
    pub add_at_l1_count: u32,
    pub add_deep_count: u32,
    pub max_consec_buy_trades: u32,
    pub max_consec_sell_trades: u32,
    pub mean_inter_trade_ms: f64,
    pub min_inter_trade_ms: f64,
    pub n_modify_chains: u32,
    pub max_modify_chain_len: u32,
    pub trades_through_multiple_levels: u32,

    // Phase 3: Deep feature expansion
    pub l1_bid_changes: u32,
    pub l1_ask_changes: u32,
    pub l1_bid_depleted: u32,
    pub l1_ask_depleted: u32,
    pub trade_size_std: f64,
    pub trade_size_skew: f64,
    pub n_trades_at_bid: u32,
    pub n_trades_at_ask: u32,
    pub reposition_count: u32,
    pub mean_order_size_at_add: f64,
    pub max_order_size_at_add: f64,
}

impl BookSnapshot {
    pub fn is_valid(&self) -> bool {
        self.mid.is_finite() && !self.mid.is_nan()
    }
}

// ============================================================================
// OrderBook
// ============================================================================

pub struct OrderBook {
    pub depth_levels: usize,

    // Core book state
    // BTreeMap<OrderedFloat> allows sorted access:
    // bids: descending (largest price first) via .iter().rev()
    // asks: ascending (smallest price first) via .iter()
    bids: BTreeMap<OrderedFloat<f64>, f64>,
    asks: BTreeMap<OrderedFloat<f64>, f64>,

    // order_id -> (side, price, size)
    orders: HashMap<u64, (Side, f64, f64)>,

    // Per-level order counting
    bid_order_counts: HashMap<OrderedFloat<f64>, u32>,
    ask_order_counts: HashMap<OrderedFloat<f64>, u32>,

    // Order lifecycle tracking
    order_add_times: HashMap<u64, u64>,     // order_id -> add_timestamp_ns
    completed_lifetimes_ms: Vec<f64>,

    // Event counters (reset on snapshot)
    add_count: u32,
    cancel_count: u32,
    trade_count: u32,
    modify_count: u32,
    buy_volume: f64,
    sell_volume: f64,

    // Aggressive trade detail
    aggressive_buy_count: u32,
    aggressive_sell_count: u32,
    max_trade_size: f64,

    // Cancel by side
    cancel_bid_vol: f64,
    cancel_ask_vol: f64,

    // Flags tracking
    bad_book_events: u32,
    bad_ts_events: u32,
    snapshot_init_events: u32,

    // Sequence tracking
    last_sequence: i64,
    sequence_gaps: u32,

    // Batch/tick tracking
    tick_count: u32,

    // Last timestamp
    last_timestamp_ns: u64,

    // Phase 2: Enhanced tracking
    trade_sizes: Vec<f64>,
    cancel_l1_count: u32,
    cancel_l1_vol: f64,
    cancel_deep_count: u32,
    cancel_deep_vol: f64,
    add_l1_count: u32,
    add_deep_count: u32,
    consec_buy: u32,
    consec_sell: u32,
    max_consec_buy: u32,
    max_consec_sell: u32,
    trade_timestamps: Vec<u64>,
    modify_chain_counts: HashMap<u64, u32>,
    last_trade_price: f64,
    trades_multi_level: u32,
    rolling_median_trade_size: f64,

    // Phase 3: Deep feature expansion tracking
    prev_best_bid: f64,
    prev_best_ask: f64,
    l1_bid_changes: u32,
    l1_ask_changes: u32,
    l1_bid_depleted: u32,
    l1_ask_depleted: u32,
    n_trades_at_bid: u32,
    n_trades_at_ask: u32,
    cancelled_prices: HashSet<OrderedFloat<f64>>,
    reposition_count: u32,
    add_sizes: Vec<f64>,
    prev_l1_bid_size: f64,
    prev_l1_ask_size: f64,
}

impl OrderBook {
    pub fn new(depth_levels: usize) -> Self {
        OrderBook {
            depth_levels,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            bid_order_counts: HashMap::new(),
            ask_order_counts: HashMap::new(),
            order_add_times: HashMap::new(),
            completed_lifetimes_ms: Vec::new(),
            add_count: 0,
            cancel_count: 0,
            trade_count: 0,
            modify_count: 0,
            buy_volume: 0.0,
            sell_volume: 0.0,
            aggressive_buy_count: 0,
            aggressive_sell_count: 0,
            max_trade_size: 0.0,
            cancel_bid_vol: 0.0,
            cancel_ask_vol: 0.0,
            bad_book_events: 0,
            bad_ts_events: 0,
            snapshot_init_events: 0,
            last_sequence: -1,
            sequence_gaps: 0,
            tick_count: 0,
            last_timestamp_ns: 0,
            trade_sizes: Vec::new(),
            cancel_l1_count: 0,
            cancel_l1_vol: 0.0,
            cancel_deep_count: 0,
            cancel_deep_vol: 0.0,
            add_l1_count: 0,
            add_deep_count: 0,
            consec_buy: 0,
            consec_sell: 0,
            max_consec_buy: 0,
            max_consec_sell: 0,
            trade_timestamps: Vec::new(),
            modify_chain_counts: HashMap::new(),
            last_trade_price: 0.0,
            trades_multi_level: 0,
            rolling_median_trade_size: 1.0,
            prev_best_bid: 0.0,
            prev_best_ask: f64::INFINITY,
            l1_bid_changes: 0,
            l1_ask_changes: 0,
            l1_bid_depleted: 0,
            l1_ask_depleted: 0,
            n_trades_at_bid: 0,
            n_trades_at_ask: 0,
            cancelled_prices: HashSet::new(),
            reposition_count: 0,
            add_sizes: Vec::new(),
            prev_l1_bid_size: 0.0,
            prev_l1_ask_size: 0.0,
        }
    }

    fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|k| k.0)
    }

    fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|k| k.0)
    }

    fn add_to_level_counts(&mut self, side: Side, price: f64) {
        let key = OrderedFloat(price);
        match side {
            Side::Bid => *self.bid_order_counts.entry(key).or_insert(0) += 1,
            Side::Ask => *self.ask_order_counts.entry(key).or_insert(0) += 1,
            _ => {}
        }
    }

    fn remove_from_level_counts(&mut self, side: Side, price: f64) {
        let key = OrderedFloat(price);
        match side {
            Side::Bid => {
                if let Some(c) = self.bid_order_counts.get_mut(&key) {
                    if *c > 0 { *c -= 1; }
                    if *c == 0 { self.bid_order_counts.remove(&key); }
                }
            }
            Side::Ask => {
                if let Some(c) = self.ask_order_counts.get_mut(&key) {
                    if *c > 0 { *c -= 1; }
                    if *c == 0 { self.ask_order_counts.remove(&key); }
                }
            }
            _ => {}
        }
    }

    fn record_order_completion(&mut self, order_id: u64, timestamp_ns: u64) {
        if let Some(add_time) = self.order_add_times.remove(&order_id) {
            if timestamp_ns > 0 && add_time > 0 && timestamp_ns >= add_time {
                let lifetime_ms = (timestamp_ns - add_time) as f64 / 1e6;
                self.completed_lifetimes_ms.push(lifetime_ms);
            }
        }
    }

    fn detect_l1_changes(&mut self) {
        let cur_best_bid = self.best_bid().unwrap_or(0.0);
        let cur_best_ask = self.best_ask().unwrap_or(f64::INFINITY);

        if cur_best_bid != self.prev_best_bid && self.prev_best_bid > 0.0 {
            self.l1_bid_changes += 1;
        }
        if cur_best_ask != self.prev_best_ask && self.prev_best_ask < f64::INFINITY {
            self.l1_ask_changes += 1;
        }

        // Detect depletions
        if self.prev_best_bid > 0.0 && !self.bids.contains_key(&OrderedFloat(self.prev_best_bid)) {
            self.l1_bid_depleted += 1;
        }
        if self.prev_best_ask < f64::INFINITY && !self.asks.contains_key(&OrderedFloat(self.prev_best_ask)) {
            self.l1_ask_depleted += 1;
        }

        self.prev_best_bid = cur_best_bid;
        self.prev_best_ask = cur_best_ask;
    }

    /// Process a single MBO event. Call this for every event in sequence.
    pub fn update(
        &mut self,
        action_byte: u8,
        side_byte: u8,
        price_raw: i64,    // Fixed-point int64 from dbn — multiply by 1e-9 for actual price
        size: u32,
        order_id: u64,
        flags: u8,
        sequence: u32,
        timestamp_ns: u64,
    ) {
        let action = parse_action(action_byte);
        let side = parse_side(side_byte);
        // Databento prices are fixed-point with 1e-9 scale
        let price = price_raw as f64 * 1e-9;
        let size_f = size as f64;

        self.last_timestamp_ns = timestamp_ns;

        // Detect L1 changes before this update
        self.detect_l1_changes();

        // Parse flags
        let is_snapshot_event = (flags & F_SNAPSHOT) != 0;
        let is_last = (flags & F_LAST) != 0;

        if (flags & F_MAYBE_BAD_BOOK) != 0 { self.bad_book_events += 1; }
        if (flags & F_BAD_TS_RECV) != 0 { self.bad_ts_events += 1; }
        if is_snapshot_event { self.snapshot_init_events += 1; }
        if is_last { self.tick_count += 1; }

        // Sequence gap detection
        let seq = sequence as i64;
        if self.last_sequence >= 0
            && seq != self.last_sequence + 1
            && seq > self.last_sequence
        {
            self.sequence_gaps += 1;
        }
        self.last_sequence = seq;

        match action {
            Action::Add => {
                if !is_snapshot_event {
                    self.add_count += 1;

                    // Phase 2: Track L1 vs deep adds
                    let best_bid = self.best_bid().unwrap_or(0.0);
                    let best_ask = self.best_ask().unwrap_or(f64::INFINITY);
                    match side {
                        Side::Bid => {
                            if price >= best_bid { self.add_l1_count += 1; }
                            else { self.add_deep_count += 1; }
                        }
                        Side::Ask => {
                            if price <= best_ask { self.add_l1_count += 1; }
                            else { self.add_deep_count += 1; }
                        }
                        _ => {}
                    }
                }

                self.orders.insert(order_id, (side, price, size_f));
                match side {
                    Side::Bid => *self.bids.entry(OrderedFloat(price)).or_insert(0.0) += size_f,
                    Side::Ask => *self.asks.entry(OrderedFloat(price)).or_insert(0.0) += size_f,
                    _ => {}
                }
                self.add_to_level_counts(side, price);

                if !is_snapshot_event {
                    self.add_sizes.push(size_f);
                    if self.cancelled_prices.contains(&OrderedFloat(price)) {
                        self.reposition_count += 1;
                    }
                    self.order_add_times.insert(order_id, timestamp_ns);
                }
            }

            Action::Cancel => {
                let existing = match self.orders.remove(&order_id) {
                    Some(e) => e,
                    None => return, // Unknown order, skip
                };
                let (side_e, price_e, size_e) = existing;

                if !is_snapshot_event {
                    self.cancel_count += 1;

                    match side_e {
                        Side::Bid => self.cancel_bid_vol += size_e,
                        Side::Ask => self.cancel_ask_vol += size_e,
                        _ => {}
                    }

                    // Phase 2: L1 vs deep cancel tracking
                    let best_bid = self.best_bid().unwrap_or(0.0);
                    let best_ask = self.best_ask().unwrap_or(f64::INFINITY);
                    let is_l1 = match side_e {
                        Side::Bid => price_e >= best_bid,
                        Side::Ask => price_e <= best_ask,
                        _ => false,
                    };

                    if is_l1 {
                        self.cancel_l1_count += 1;
                        self.cancel_l1_vol += size_e;
                    } else {
                        self.cancel_deep_count += 1;
                        self.cancel_deep_vol += size_e;
                    }

                    // Phase 3: Track cancelled prices for reposition detection
                    self.cancelled_prices.insert(OrderedFloat(price_e));
                }

                // Update book
                match side_e {
                    Side::Bid => {
                        let key = OrderedFloat(price_e);
                        if let Some(v) = self.bids.get_mut(&key) {
                            *v = (*v - size_e).max(0.0);
                            if *v == 0.0 { self.bids.remove(&key); }
                        }
                    }
                    Side::Ask => {
                        let key = OrderedFloat(price_e);
                        if let Some(v) = self.asks.get_mut(&key) {
                            *v = (*v - size_e).max(0.0);
                            if *v == 0.0 { self.asks.remove(&key); }
                        }
                    }
                    _ => {}
                }
                self.remove_from_level_counts(side_e, price_e);
                self.record_order_completion(order_id, timestamp_ns);
            }

            Action::Modify => {
                self.modify_count += 1;
                *self.modify_chain_counts.entry(order_id).or_insert(0) += 1;

                let existing = match self.orders.get(&order_id).cloned() {
                    Some(e) => e,
                    None => return, // Unknown order, still count modify but skip book update
                };
                let (side_prev, price_prev, size_prev) = existing;

                // Remove from old location
                match side_prev {
                    Side::Bid => {
                        let key = OrderedFloat(price_prev);
                        if let Some(v) = self.bids.get_mut(&key) {
                            *v = (*v - size_prev).max(0.0);
                            if *v == 0.0 { self.bids.remove(&key); }
                        }
                    }
                    Side::Ask => {
                        let key = OrderedFloat(price_prev);
                        if let Some(v) = self.asks.get_mut(&key) {
                            *v = (*v - size_prev).max(0.0);
                            if *v == 0.0 { self.asks.remove(&key); }
                        }
                    }
                    _ => {}
                }
                self.remove_from_level_counts(side_prev, price_prev);

                // Add at new location
                self.orders.insert(order_id, (side, price, size_f));
                match side {
                    Side::Bid => *self.bids.entry(OrderedFloat(price)).or_insert(0.0) += size_f,
                    Side::Ask => *self.asks.entry(OrderedFloat(price)).or_insert(0.0) += size_f,
                    _ => {}
                }
                self.add_to_level_counts(side, price);
            }

            // Python ignores FILL events entirely (no handler in lob.py).
            // Only TRADE ('T') events are counted as trades.
            Action::Fill => {}

            Action::Trade => {
                self.trade_count += 1;
                let existing = match self.orders.get(&order_id).cloned() {
                    Some(e) => e,
                    None => return,
                };
                let (side_prev, price_prev, size_prev) = existing;
                let remaining = (size_prev - size_f).max(0.0);

                if remaining == 0.0 {
                    self.orders.remove(&order_id);
                    self.remove_from_level_counts(side_prev, price_prev);
                    self.record_order_completion(order_id, timestamp_ns);
                } else {
                    self.orders.insert(order_id, (side_prev, price_prev, remaining));
                }

                // Update book volume at price level
                match side_prev {
                    Side::Bid => {
                        let key = OrderedFloat(price_prev);
                        if let Some(v) = self.bids.get_mut(&key) {
                            *v = (*v - size_f).max(0.0);
                            if *v == 0.0 { self.bids.remove(&key); }
                        }
                    }
                    Side::Ask => {
                        let key = OrderedFloat(price_prev);
                        if let Some(v) = self.asks.get_mut(&key) {
                            *v = (*v - size_f).max(0.0);
                            if *v == 0.0 { self.asks.remove(&key); }
                        }
                    }
                    _ => {}
                }

                // Track direction (passive side tells us aggressor)
                match side_prev {
                    Side::Bid => {
                        // Passive bid was hit — aggressive sell
                        self.sell_volume += size_f;
                        self.aggressive_sell_count += 1;
                        self.n_trades_at_bid += 1;
                        self.consec_sell += 1;
                        self.consec_buy = 0;
                        if self.consec_sell > self.max_consec_sell {
                            self.max_consec_sell = self.consec_sell;
                        }
                    }
                    Side::Ask => {
                        // Passive ask was lifted — aggressive buy
                        self.buy_volume += size_f;
                        self.aggressive_buy_count += 1;
                        self.n_trades_at_ask += 1;
                        self.consec_buy += 1;
                        self.consec_sell = 0;
                        if self.consec_buy > self.max_consec_buy {
                            self.max_consec_buy = self.consec_buy;
                        }
                    }
                    _ => {}
                }

                if size_f > self.max_trade_size { self.max_trade_size = size_f; }

                self.trade_sizes.push(size_f);
                if timestamp_ns > 0 { self.trade_timestamps.push(timestamp_ns); }

                // Multi-level trade detection
                if self.last_trade_price > 0.0 && price_prev != self.last_trade_price {
                    self.trades_multi_level += 1;
                }
                self.last_trade_price = price_prev;

                // Update rolling median (EMA approximation)
                if self.rolling_median_trade_size <= 0.0 {
                    self.rolling_median_trade_size = size_f;
                } else {
                    self.rolling_median_trade_size = 0.99 * self.rolling_median_trade_size + 0.01 * size_f;
                }
            }

            Action::Clear => {
                self.bids.clear();
                self.asks.clear();
                self.orders.clear();
                self.bid_order_counts.clear();
                self.ask_order_counts.clear();
                self.order_add_times.clear();
                self.reset_event_counts();
            }

            Action::Unknown => {}
        }
    }

    fn reset_event_counts(&mut self) {
        self.add_count = 0;
        self.cancel_count = 0;
        self.trade_count = 0;
        self.modify_count = 0;
        self.buy_volume = 0.0;
        self.sell_volume = 0.0;
        self.aggressive_buy_count = 0;
        self.aggressive_sell_count = 0;
        self.max_trade_size = 0.0;
        self.cancel_bid_vol = 0.0;
        self.cancel_ask_vol = 0.0;
        self.bad_book_events = 0;
        self.bad_ts_events = 0;
        self.snapshot_init_events = 0;
        self.sequence_gaps = 0;
        self.tick_count = 0;
        self.completed_lifetimes_ms.clear();
        // Phase 2
        self.trade_sizes.clear();
        self.cancel_l1_count = 0;
        self.cancel_l1_vol = 0.0;
        self.cancel_deep_count = 0;
        self.cancel_deep_vol = 0.0;
        self.add_l1_count = 0;
        self.add_deep_count = 0;
        self.consec_buy = 0;
        self.consec_sell = 0;
        self.max_consec_buy = 0;
        self.max_consec_sell = 0;
        self.trade_timestamps.clear();
        self.trades_multi_level = 0;
        self.last_trade_price = 0.0;
        // Phase 3
        self.l1_bid_changes = 0;
        self.l1_ask_changes = 0;
        self.l1_bid_depleted = 0;
        self.l1_ask_depleted = 0;
        self.n_trades_at_bid = 0;
        self.n_trades_at_ask = 0;
        self.cancelled_prices.clear();
        self.reposition_count = 0;
        self.add_sizes.clear();
    }

    fn compute_lifetime_stats(&self) -> (f64, f64, f64, u32) {
        let lifetimes = &self.completed_lifetimes_ms;
        let n = lifetimes.len();
        if n == 0 { return (0.0, 0.0, 0.0, 0); }

        let mean_ms = lifetimes.iter().sum::<f64>() / n as f64;
        let fleeting = lifetimes.iter().filter(|&&x| x < 100.0).count() as f64 / n as f64;

        // Median via sort
        let mut sorted = lifetimes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_ms = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        (mean_ms, median_ms, fleeting, n as u32)
    }

    fn compute_phase2_stats(&self) -> Phase2Stats {
        let mut stats = Phase2Stats::default();

        // Trade size distribution
        if !self.trade_sizes.is_empty() {
            let arr = &self.trade_sizes;
            let median_size = self.rolling_median_trade_size.max(1.0);
            stats.n_large_trades = arr.iter().filter(|&&s| s > 2.0 * median_size).count() as u32;
            stats.n_small_trades = arr.iter().filter(|&&s| s < 0.5 * median_size).count() as u32;
            stats.trade_size_p75 = percentile(arr, 75.0);

            // Std and skew
            if arr.len() >= 2 {
                let mean = arr.iter().sum::<f64>() / arr.len() as f64;
                let variance = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / arr.len() as f64;
                stats.trade_size_std = variance.sqrt();
                if stats.trade_size_std > 1e-8 {
                    stats.trade_size_skew = arr.iter()
                        .map(|&x| ((x - mean) / stats.trade_size_std).powi(3))
                        .sum::<f64>() / arr.len() as f64;
                }
            }
        }

        // Inter-trade timing
        if self.trade_timestamps.len() >= 2 {
            let diffs: Vec<f64> = self.trade_timestamps.windows(2)
                .filter_map(|w| {
                    let diff = w[1].saturating_sub(w[0]);
                    if diff > 0 { Some(diff as f64 / 1e6) } else { None }
                })
                .collect();
            if !diffs.is_empty() {
                stats.mean_inter_trade_ms = diffs.iter().sum::<f64>() / diffs.len() as f64;
                stats.min_inter_trade_ms = diffs.iter().cloned().fold(f64::INFINITY, f64::min);
            }
        }

        // Modify chains
        if !self.modify_chain_counts.is_empty() {
            let chains: Vec<u32> = self.modify_chain_counts.values().cloned().collect();
            stats.n_modify_chains = chains.iter().filter(|&&c| c >= 3).count() as u32;
            stats.max_modify_chain_len = *chains.iter().max().unwrap_or(&0);
        }

        // Order size at add
        if !self.add_sizes.is_empty() {
            stats.mean_order_size_at_add = self.add_sizes.iter().sum::<f64>() / self.add_sizes.len() as f64;
            stats.max_order_size_at_add = self.add_sizes.iter().cloned().fold(0.0_f64, f64::max);
        }

        stats
    }

    fn get_level_order_counts(&self, prices: &[f64], is_bid: bool) -> Vec<f64> {
        let oc = if is_bid { &self.bid_order_counts } else { &self.ask_order_counts };
        prices.iter().map(|&p| {
            *oc.get(&OrderedFloat(p)).unwrap_or(&0) as f64
        }).collect()
    }

    /// Create a snapshot of current book state with all tracked metadata.
    /// This resets per-interval counters (mirrors Python's snapshot() + reset_event_counts()).
    pub fn snapshot(&mut self) -> BookSnapshot {
        // Collect top-N bid/ask levels
        let bid_prices: Vec<f64> = self.bids.keys().rev()
            .take(self.depth_levels)
            .map(|k| k.0)
            .collect();
        let ask_prices: Vec<f64> = self.asks.keys()
            .take(self.depth_levels)
            .map(|k| k.0)
            .collect();

        let bid_sizes: Vec<f64> = bid_prices.iter()
            .map(|&p| *self.bids.get(&OrderedFloat(p)).unwrap_or(&0.0))
            .collect();
        let ask_sizes: Vec<f64> = ask_prices.iter()
            .map(|&p| *self.asks.get(&OrderedFloat(p)).unwrap_or(&0.0))
            .collect();

        let best_bid = bid_prices.first().copied().unwrap_or(f64::NAN);
        let best_ask = ask_prices.first().copied().unwrap_or(f64::NAN);
        let mid = if best_bid.is_finite() && best_ask.is_finite() {
            (best_bid + best_ask) / 2.0
        } else {
            f64::NAN
        };
        let spread = if best_bid.is_finite() && best_ask.is_finite() {
            best_ask - best_bid
        } else {
            f64::NAN
        };

        let (mean_lt, median_lt, fleeting, n_completed) = self.compute_lifetime_stats();
        let bid_oc = self.get_level_order_counts(&bid_prices, true);
        let ask_oc = self.get_level_order_counts(&ask_prices, false);
        let p2 = self.compute_phase2_stats();

        let snap = BookSnapshot {
            bid_prices: bid_prices.clone(),
            bid_sizes: bid_sizes.clone(),
            ask_prices: ask_prices.clone(),
            ask_sizes: ask_sizes.clone(),
            mid,
            spread,
            timestamp_ns: self.last_timestamp_ns,
            add_count: self.add_count,
            cancel_count: self.cancel_count,
            trade_count: self.trade_count,
            modify_count: self.modify_count,
            buy_volume: self.buy_volume,
            sell_volume: self.sell_volume,
            aggressive_buy_count: self.aggressive_buy_count,
            aggressive_sell_count: self.aggressive_sell_count,
            max_trade_size: self.max_trade_size,
            cancel_bid_vol: self.cancel_bid_vol,
            cancel_ask_vol: self.cancel_ask_vol,
            mean_lifetime_ms: mean_lt,
            median_lifetime_ms: median_lt,
            fleeting_ratio: fleeting,
            n_orders_completed: n_completed,
            bid_order_counts: bid_oc,
            ask_order_counts: ask_oc,
            bad_book_events: self.bad_book_events,
            bad_ts_events: self.bad_ts_events,
            snapshot_init_events: self.snapshot_init_events,
            sequence_gaps: self.sequence_gaps,
            tick_count: self.tick_count,
            n_large_trades: p2.n_large_trades,
            n_small_trades: p2.n_small_trades,
            trade_size_p75: p2.trade_size_p75,
            cancel_at_l1_count: self.cancel_l1_count,
            cancel_at_l1_vol: self.cancel_l1_vol,
            cancel_deep_count: self.cancel_deep_count,
            cancel_deep_vol: self.cancel_deep_vol,
            add_at_l1_count: self.add_l1_count,
            add_deep_count: self.add_deep_count,
            max_consec_buy_trades: self.max_consec_buy,
            max_consec_sell_trades: self.max_consec_sell,
            mean_inter_trade_ms: p2.mean_inter_trade_ms,
            min_inter_trade_ms: p2.min_inter_trade_ms,
            n_modify_chains: p2.n_modify_chains,
            max_modify_chain_len: p2.max_modify_chain_len,
            trades_through_multiple_levels: self.trades_multi_level,
            l1_bid_changes: self.l1_bid_changes,
            l1_ask_changes: self.l1_ask_changes,
            l1_bid_depleted: self.l1_bid_depleted,
            l1_ask_depleted: self.l1_ask_depleted,
            trade_size_std: p2.trade_size_std,
            trade_size_skew: p2.trade_size_skew,
            n_trades_at_bid: self.n_trades_at_bid,
            n_trades_at_ask: self.n_trades_at_ask,
            reposition_count: self.reposition_count,
            mean_order_size_at_add: p2.mean_order_size_at_add,
            max_order_size_at_add: p2.max_order_size_at_add,
        };

        self.reset_event_counts();
        snap
    }

    // ========================================================================
    // Public accessors for deep learning modules
    // ========================================================================

    /// Read-only access to the bid side of the book.
    pub fn bids(&self) -> &BTreeMap<OrderedFloat<f64>, f64> {
        &self.bids
    }

    /// Read-only access to the ask side of the book.
    pub fn asks(&self) -> &BTreeMap<OrderedFloat<f64>, f64> {
        &self.asks
    }

    /// Read-only access to bid order counts per price level.
    pub fn bid_order_counts(&self) -> &HashMap<OrderedFloat<f64>, u32> {
        &self.bid_order_counts
    }

    /// Read-only access to ask order counts per price level.
    pub fn ask_order_counts(&self) -> &HashMap<OrderedFloat<f64>, u32> {
        &self.ask_order_counts
    }

    /// Get the last processed timestamp.
    pub fn last_timestamp_ns(&self) -> u64 {
        self.last_timestamp_ns
    }

    /// Compute the current mid price without taking a snapshot.
    pub fn current_mid(&self) -> f64 {
        let best_bid = self.bids.keys().next_back().map(|k| k.0);
        let best_ask = self.asks.keys().next().map(|k| k.0);
        match (best_bid, best_ask) {
            (Some(bid), Some(ask)) => (bid + ask) / 2.0,
            _ => f64::NAN,
        }
    }
}

// ============================================================================
// Helper structs
// ============================================================================

#[derive(Default)]
struct Phase2Stats {
    n_large_trades: u32,
    n_small_trades: u32,
    trade_size_p75: f64,
    mean_inter_trade_ms: f64,
    min_inter_trade_ms: f64,
    n_modify_chains: u32,
    max_modify_chain_len: u32,
    trade_size_std: f64,
    trade_size_skew: f64,
    mean_order_size_at_add: f64,
    max_order_size_at_add: f64,
}

/// Compute approximate percentile from unsorted data.
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    if data.len() == 1 { return data[0]; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}
