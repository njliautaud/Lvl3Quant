/// MBO Event-Level Fill Simulator
///
/// True event-by-event fill simulation using FIFO queue position tracking.
/// Processes raw .dbn MBO events, maintains a live order book, and determines
/// fills for virtual orders based on actual queue depletion from real trades.
///
/// Key design principles:
/// - NO shortcuts: every event is processed in nanosecond order
/// - Queue position = displayed book size at post time (icebergs get new timestamps)
/// - Fills are DETERMINISTIC: when contracts_ahead <= 0, we're filled
/// - Expected emergent fill rate: ~83% (validated against Feb 17 queue study)
/// - Trailing stops checked on every TRADE event, not just bar boundaries

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::lob::{self, Action, OrderBook, Side};
use crate::ingest::MboEvent;

// ============================================================================
// Constants
// ============================================================================

const TICK_SIZE: f64 = 0.25;      // ES tick size
const COMMISSION_RT_TICKS: f64 = 0.376;  // $4.70 / $12.50 per tick (AMP all-in)

/// Nanoseconds per 100ms bar
const BAR_NS: u64 = 100_000_000;

// ============================================================================
// Virtual Order — our simulated order in the book
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualOrderState {
    PendingEntry,
    InBook,       // Posted, waiting for fill
    Filled,       // Entry filled, position open
    PendingExit,  // Exit order posted
    Closed,       // Trade complete
    Cancelled,    // Timed out or cancelled
}

#[derive(Debug, Clone)]
pub struct VirtualOrder {
    pub id: u64,
    pub side: OrderSide,
    pub entry_price: f64,
    pub size: u32,
    pub state: VirtualOrderState,

    // Queue tracking
    pub queue_position: f64,          // Contracts ahead of us
    pub book_size_at_post: f64,       // Book size when we posted
    pub post_time_ns: u64,            // When we posted the order

    // Fill tracking
    pub fill_time_ns: Option<u64>,    // When entry was filled
    pub fill_price: Option<f64>,      // Actual fill price

    // Exit tracking
    pub exit_price: Option<f64>,
    pub exit_time_ns: Option<u64>,
    pub exit_reason: Option<ExitReason>,

    // Position management
    pub hold_until_ns: Option<u64>,   // Max hold time (from FILL, not signal)
    pub trailing_stop_price: Option<f64>,
    pub stop_loss_price: Option<f64>,    // Fixed stop loss (from entry price, never moves)
    pub take_profit_price: Option<f64>,
    pub best_price_seen: f64,         // For trailing stop tracking
    pub worst_price_seen: f64,       // For MAE tracking (worst adverse price)
    pub mae_ticks: f64,              // Max adverse excursion in ticks (from entry)
    pub mfe_ticks: f64,              // Max favorable excursion in ticks (from entry)

    // Vol-based exit tracking
    pub recent_adverse_ticks: f64,   // Adverse move in last N bars
    pub adverse_check_price: f64,    // Price at last adverse check window start

    // Ratchet stop tracking
    pub ratchet_stop_price: Option<f64>,  // Current ratchet stop level (only ratchets up)

    // Signal metadata
    pub signal_bar_ns: u64,           // When the signal fired
    pub signal_strength: f64,         // Model prediction value

    // Chase/reprice tracking
    pub original_entry_price: f64,    // First posted price (for chase distance calc)
    pub chase_reprices: u32,          // Number of cancel-replace cycles
    pub last_chase_check_ns: u64,     // Last time we checked for reprice

    // Conviction exit tracking
    pub conviction_flip_counter: u64,  // Consecutive bars of opposite signal
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExitReason {
    HoldTimeout,
    TrailingStop,
    StopLoss,     // Fixed stop loss from entry price (never moves)
    VolExit,      // Vol-based exit: fast adverse move detected
    TakeProfit,
    SignalFlip,
    EndOfDay,
    MarketExit,   // Forced market order (adds spread cost)
    RatchetStop,  // Progressive trailing stop based on MFE thresholds
    ConvictionExit,  // Delayed signal flip: opposite signal persisted for N bars
    MaeTimeExit,  // Exit when underwater by N ticks for M seconds
}

// ============================================================================
// Trade Result — output per completed trade
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeResult {
    pub order_id: u64,
    pub side: String,
    pub size: u32,
    pub signal_time_ns: u64,
    pub post_time_ns: u64,
    pub fill_time_ns: u64,
    pub exit_time_ns: u64,
    pub entry_price: f64,
    pub exit_price: f64,
    pub pnl_ticks: f64,
    pub pnl_dollars: f64,
    pub exit_reason: ExitReason,
    pub queue_position_at_post: f64,
    pub book_size_at_post: f64,
    pub fill_latency_ns: u64,       // fill_time - post_time
    pub hold_duration_ns: u64,      // exit_time - fill_time
    pub mae_ticks: f64,             // Max adverse excursion from entry (in ticks)
    pub mfe_ticks: f64,             // Max favorable excursion from entry (in ticks)
    pub signal_strength: f64,
}

// ============================================================================
// Simulation Configuration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub hold_ms: u64,               // Max hold period in ms (from FILL time)
    pub trailing_stop_ticks: Option<f64>,
    pub stop_loss_ticks: Option<f64>,   // Fixed stop loss from entry (never moves)
    pub take_profit_ticks: Option<f64>,
    /// Vol-based exit: if price moves X ticks against us within Y bars, exit.
    /// Detects abnormally fast adverse moves (flash crashes, news spikes).
    pub vol_exit_ticks: Option<f64>,     // Adverse ticks threshold
    pub vol_exit_bars: Option<u64>,      // Window in bars to measure (100ms each)
    pub max_wait_bars: u64,         // Max bars to wait for fill
    pub market_exit_spread_cost: f64,  // 0.5 ticks for crossing spread on market exit
    pub commission_ticks: f64,       // Per round-trip
    pub eod_close: bool,            // Force close at end of day
    pub signal_flip_exit: bool,     // Exit when signal flips sign
    /// Order submission latency in nanoseconds.
    /// Models the delay between signal generation and order appearing in the book.
    /// During this delay, the book can move and queue position may change.
    /// Typical values: 1-5ms co-located, 10-20ms direct, 50ms+ retail.
    /// Default: 0 (instant, optimistic).
    pub order_latency_ns: u64,
    /// Exit order latency in nanoseconds.
    /// Models delay for cancel/replace and market exit orders.
    /// Defaults to same as order_latency_ns if not set.
    pub exit_latency_ns: Option<u64>,
    /// When true, only submit signals during prime trading hours (10:30 AM - 2:30 PM ET).
    /// Signals outside this window are silently skipped.
    pub prime_hours_only: bool,
    /// Entry mode: when true, use market orders (cross the spread) instead of
    /// passive limit orders at BBO. Market orders fill instantly at the opposite
    /// BBO with full spread cost. Guarantees 100% fill rate but pays the spread.
    pub market_entry: bool,
    /// Entry spread cost in ticks for market orders. Default 1.0 (full spread).
    /// For aggressive limit (1 tick through), set to 0.5.
    pub market_entry_spread_ticks: f64,
    /// Mid-price entry: post limit orders at the midpoint (bid+ask)/2.
    /// Splits the spread — cheaper than market orders, better queue position than passive.
    /// The order sits between the BBO levels. Fill depends on price crossing mid.
    pub mid_price_entry: bool,

    // ---- Chase / Cancel-Replace Mode ----
    /// When true, unfilled passive orders will "chase" the BBO:
    /// if the BBO moves away from our resting price, we cancel and reprice
    /// to the new BBO (back of the new queue). This models a cancel-replace strategy.
    pub chase_entry: bool,
    /// How often to check for reprice opportunities, in nanoseconds.
    /// Default: 100ms (1 bar). The order won't reprice more often than this.
    pub chase_check_interval_ns: u64,
    /// Maximum number of ticks we're willing to chase away from the original entry.
    /// After chasing this far, the order is cancelled (or force-crossed if chase_force_cross).
    /// 0 = unlimited chase. Default: 2 ticks.
    pub chase_max_ticks: f64,
    /// Maximum number of cancel-replace cycles before giving up.
    /// 0 = unlimited. Default: 5.
    pub chase_max_reprices: u32,
    /// If true, after exceeding chase_max_ticks or chase_max_reprices, force a market
    /// order (cross the spread) instead of cancelling. Default: false.
    pub chase_force_cross: bool,
    /// Number of contracts per order. P&L and commissions scale linearly by this.
    /// Fill dynamics are unchanged (valid for <20 lots on ES). Default: 1.
    pub order_size: u32,
    /// Additional exit slippage in ticks, added to market exit spread cost.
    /// Models adverse price movement during exit latency.
    /// E.g., 0.1-0.2 ticks for ~150ms latency at typical ES volatility.
    /// Default: 0.0 (backward compatible).
    pub exit_slippage_ticks: f64,

    // ---- Ratcheting Trailing Stop ----
    /// When true, use progressive ratcheting stop based on MFE thresholds.
    /// As MFE crosses higher thresholds, the stop is locked at progressively
    /// higher profit levels. The stop only ratchets up (never moves back).
    pub ratchet_stop: bool,

    // ---- MAE + Time Exit ----
    // ---- Conviction Exit (delayed signal flip) ----
    /// Number of consecutive bars the signal must be opposite before triggering exit.
    /// 0 = disabled. 100 = 10 seconds (at 100ms bars). Replaces instant signal_flip_exit.
    pub conviction_exit_bars: u64,
    /// Minimum magnitude (absolute z-score) of opposite signal to count toward conviction.
    /// 0.0 = any opposite signal counts. 1.0 = require |z| >= 1.0.
    pub conviction_exit_mag: f64,

    /// If position is underwater by >= this many ticks AND held for >= mae_exit_hold_sec,
    /// exit at market. 0 = disabled. Typical: 10 ticks.
    pub mae_exit_ticks: f64,
    /// Minimum hold time in seconds before MAE exit triggers. 0 = disabled.
    /// Typical: 600 seconds (10 minutes).
    pub mae_exit_hold_sec: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            hold_ms: 10_000,
            trailing_stop_ticks: None,
            stop_loss_ticks: None,
            take_profit_ticks: None,
            vol_exit_ticks: None,
            vol_exit_bars: None,
            max_wait_bars: 100,
            market_exit_spread_cost: 0.5,
            commission_ticks: COMMISSION_RT_TICKS,
            eod_close: true,
            signal_flip_exit: false,
            order_latency_ns: 0,
            exit_latency_ns: None,
            prime_hours_only: false,
            market_entry: false,
            market_entry_spread_ticks: 1.0,
            mid_price_entry: false,
            chase_entry: false,
            chase_check_interval_ns: 100_000_000,  // 100ms = 1 bar
            chase_max_ticks: 2.0,
            chase_max_reprices: 5,
            chase_force_cross: false,
            order_size: 1,
            exit_slippage_ticks: 0.0,
            ratchet_stop: false,
            mae_exit_ticks: 0.0,
            mae_exit_hold_sec: 0.0,
            conviction_exit_bars: 0,
            conviction_exit_mag: 0.0,
        }
    }
}

// ============================================================================
// Signal — input from model predictions
// ============================================================================

#[derive(Debug, Clone)]
pub struct Signal {
    pub bar_ns: u64,          // Bar timestamp (100ms boundary)
    pub direction: f64,       // Positive = buy, negative = sell
    pub magnitude: f64,       // Predicted magnitude (for gating)
    pub confidence: f64,      // Model confidence (for position sizing)
}

// ============================================================================
// Free functions for position management (avoids borrow checker issues)
// ============================================================================

/// Close a position and set exit fields on the virtual order.
fn close_position_impl(
    config: &SimConfig,
    vo: &mut VirtualOrder,
    exit_price: f64,
    ts: u64,
    reason: ExitReason,
) {
    vo.state = VirtualOrderState::Closed;
    vo.exit_time_ns = Some(ts);
    vo.exit_reason = Some(reason);

    // Market/urgent exits add spread cost (crossing the spread) + exit slippage.
    // Trailing stops also pay spread — in fast markets when the stop fires,
    // you're crossing the spread to get out, not passively filled.
    //
    // EndOfDay: caller already passes bid (for longs) or ask (for shorts),
    // which already reflects crossing the spread. No additional spread penalty
    // is applied to avoid double-counting. Exit slippage still applies.
    let spread_penalty = match reason {
        ExitReason::EndOfDay => {
            // No spread cost (already in exit_price from bid/ask), only slippage
            config.exit_slippage_ticks * TICK_SIZE
        }
        ExitReason::HoldTimeout | ExitReason::MarketExit
        | ExitReason::TrailingStop | ExitReason::StopLoss | ExitReason::VolExit
        | ExitReason::SignalFlip | ExitReason::RatchetStop | ExitReason::MaeTimeExit | ExitReason::ConvictionExit => {
            (config.market_exit_spread_cost + config.exit_slippage_ticks) * TICK_SIZE
        }
        ExitReason::TakeProfit => 0.0, // TP is a resting limit order, passive fill
    };

    let raw_exit = match vo.side {
        OrderSide::Buy => exit_price - spread_penalty,
        OrderSide::Sell => exit_price + spread_penalty,
    };
    vo.exit_price = Some(raw_exit);
}

/// Ratcheting trailing stop table: (mfe_threshold_ticks, lock_profit_ticks).
/// When MFE crosses a threshold, the stop locks in the corresponding profit level.
/// The stop only ratchets up (never moves back to a lower level).
const RATCHET_TABLE: [(f64, f64); 6] = [
    (3.0,  0.0),    // MFE >= 3 ticks  -> stop at breakeven
    (8.0,  4.0),    // MFE >= 8 ticks  -> stop at +4 ticks
    (15.0, 8.0),    // MFE >= 15 ticks -> stop at +8 ticks
    (20.0, 10.0),   // MFE >= 20 ticks -> stop at +10 ticks
    (30.0, 15.0),   // MFE >= 30 ticks -> stop at +15 ticks
    (50.0, 30.0),   // MFE >= 50 ticks -> stop at +30 ticks
];

/// Compute the ratchet stop price given current MFE, entry price, and side.
/// Returns None if MFE hasn't reached any ratchet threshold.
fn compute_ratchet_stop(mfe_ticks: f64, entry_price: f64, side: OrderSide) -> Option<f64> {
    let mut best_lock: Option<f64> = None;
    for &(mfe_thresh, lock_profit) in &RATCHET_TABLE {
        if mfe_ticks >= mfe_thresh {
            best_lock = Some(lock_profit);
        }
    }
    best_lock.map(|lock_ticks| {
        match side {
            OrderSide::Buy  => entry_price + lock_ticks * TICK_SIZE,
            OrderSide::Sell => entry_price - lock_ticks * TICK_SIZE,
        }
    })
}

/// Check position exit conditions on a trade event (trailing stop, take profit).
/// Returns true if the position was closed.
fn check_exit_on_trade(
    config: &SimConfig,
    vo: &mut VirtualOrder,
    trade_price: f64,
    ts: u64,
) -> bool {
    if vo.state != VirtualOrderState::Filled {
        return false;
    }

    // Track MAE/MFE for all filled positions
    let entry = vo.entry_price;
    match vo.side {
        OrderSide::Buy => {
            let favorable = (trade_price - entry) / TICK_SIZE;
            let adverse = (entry - trade_price) / TICK_SIZE;
            if favorable > vo.mfe_ticks { vo.mfe_ticks = favorable; }
            if adverse > vo.mae_ticks { vo.mae_ticks = adverse; }
            if trade_price < vo.worst_price_seen { vo.worst_price_seen = trade_price; }

            // Update best price seen (for trailing stop)
            if trade_price > vo.best_price_seen {
                vo.best_price_seen = trade_price;
                if let Some(ts_ticks) = config.trailing_stop_ticks {
                    vo.trailing_stop_price = Some(
                        vo.best_price_seen - ts_ticks * TICK_SIZE
                    );
                }
            }

            // Ratcheting trailing stop: update and check
            if config.ratchet_stop {
                if let Some(new_stop) = compute_ratchet_stop(vo.mfe_ticks, entry, vo.side) {
                    // Only ratchet up — never move stop back to a lower level
                    let should_update = match vo.ratchet_stop_price {
                        Some(current) => new_stop > current,
                        None => true,
                    };
                    if should_update {
                        vo.ratchet_stop_price = Some(new_stop);
                    }
                }
                if let Some(ratchet) = vo.ratchet_stop_price {
                    if trade_price <= ratchet {
                        close_position_impl(config, vo, ratchet, ts, ExitReason::RatchetStop);
                        return true;
                    }
                }
            }

            // Check trailing stop hit
            if let Some(stop) = vo.trailing_stop_price {
                if trade_price <= stop {
                    close_position_impl(config, vo, stop, ts, ExitReason::TrailingStop);
                    return true;
                }
            }
            // Check fixed stop loss (from entry price, never moves)
            if let Some(sl) = vo.stop_loss_price {
                if trade_price <= sl {
                    close_position_impl(config, vo, sl, ts, ExitReason::StopLoss);
                    return true;
                }
            }
            // Check vol-based exit (abnormally fast adverse move)
            if let (Some(vol_ticks), Some(_vol_bars)) = (config.vol_exit_ticks, config.vol_exit_bars) {
                let adverse_from_entry = (entry - trade_price) / TICK_SIZE;
                if adverse_from_entry >= vol_ticks {
                    // Check speed: how fast did this adverse move happen?
                    let fill_ns = vo.fill_time_ns.unwrap_or(ts);
                    let elapsed_bars = (ts - fill_ns) / 100_000_000; // 100ms per bar
                    if let Some(vb) = config.vol_exit_bars {
                        if elapsed_bars <= vb {
                            // Fast adverse move — exit immediately
                            close_position_impl(config, vo, trade_price, ts, ExitReason::VolExit);
                            return true;
                        }
                    }
                }
            }
            // MAE + Time exit: if underwater by N ticks for M seconds, exit
            if config.mae_exit_ticks > 0.0 && config.mae_exit_hold_sec > 0.0 {
                let current_adverse = (entry - trade_price) / TICK_SIZE;
                if current_adverse >= config.mae_exit_ticks {
                    let fill_ns = vo.fill_time_ns.unwrap_or(ts);
                    let hold_secs = (ts.saturating_sub(fill_ns)) as f64 / 1_000_000_000.0;
                    if hold_secs >= config.mae_exit_hold_sec {
                        close_position_impl(config, vo, trade_price, ts, ExitReason::MaeTimeExit);
                        return true;
                    }
                }
            }
            // Check take profit
            if let Some(tp) = vo.take_profit_price {
                if trade_price >= tp {
                    close_position_impl(config, vo, tp, ts, ExitReason::TakeProfit);
                    return true;
                }
            }
        }
        OrderSide::Sell => {
            let favorable = (entry - trade_price) / TICK_SIZE;
            let adverse = (trade_price - entry) / TICK_SIZE;
            if favorable > vo.mfe_ticks { vo.mfe_ticks = favorable; }
            if adverse > vo.mae_ticks { vo.mae_ticks = adverse; }
            if trade_price > vo.worst_price_seen { vo.worst_price_seen = trade_price; }

            if trade_price < vo.best_price_seen {
                vo.best_price_seen = trade_price;
                if let Some(ts_ticks) = config.trailing_stop_ticks {
                    vo.trailing_stop_price = Some(
                        vo.best_price_seen + ts_ticks * TICK_SIZE
                    );
                }
            }

            // Ratcheting trailing stop: update and check (Sell side)
            if config.ratchet_stop {
                if let Some(new_stop) = compute_ratchet_stop(vo.mfe_ticks, entry, vo.side) {
                    // For shorts, "ratchet up" means stop moves DOWN (more favorable)
                    let should_update = match vo.ratchet_stop_price {
                        Some(current) => new_stop < current,
                        None => true,
                    };
                    if should_update {
                        vo.ratchet_stop_price = Some(new_stop);
                    }
                }
                if let Some(ratchet) = vo.ratchet_stop_price {
                    if trade_price >= ratchet {
                        close_position_impl(config, vo, ratchet, ts, ExitReason::RatchetStop);
                        return true;
                    }
                }
            }

            if let Some(stop) = vo.trailing_stop_price {
                if trade_price >= stop {
                    close_position_impl(config, vo, stop, ts, ExitReason::TrailingStop);
                    return true;
                }
            }
            // Check fixed stop loss (from entry price, never moves)
            if let Some(sl) = vo.stop_loss_price {
                if trade_price >= sl {
                    close_position_impl(config, vo, sl, ts, ExitReason::StopLoss);
                    return true;
                }
            }
            // Check vol-based exit (abnormally fast adverse move) — Sell side
            if let (Some(vol_ticks), Some(_vol_bars)) = (config.vol_exit_ticks, config.vol_exit_bars) {
                let adverse_from_entry = (trade_price - entry) / TICK_SIZE;
                if adverse_from_entry >= vol_ticks {
                    let fill_ns = vo.fill_time_ns.unwrap_or(ts);
                    let elapsed_bars = (ts - fill_ns) / 100_000_000;
                    if let Some(vb) = config.vol_exit_bars {
                        if elapsed_bars <= vb {
                            close_position_impl(config, vo, trade_price, ts, ExitReason::VolExit);
                            return true;
                        }
                    }
                }
            }
            // MAE + Time exit: if underwater by N ticks for M seconds, exit (Sell side)
            if config.mae_exit_ticks > 0.0 && config.mae_exit_hold_sec > 0.0 {
                let current_adverse = (trade_price - entry) / TICK_SIZE;
                if current_adverse >= config.mae_exit_ticks {
                    let fill_ns = vo.fill_time_ns.unwrap_or(ts);
                    let hold_secs = (ts.saturating_sub(fill_ns)) as f64 / 1_000_000_000.0;
                    if hold_secs >= config.mae_exit_hold_sec {
                        close_position_impl(config, vo, trade_price, ts, ExitReason::MaeTimeExit);
                        return true;
                    }
                }
            }
            if let Some(tp) = vo.take_profit_price {
                if trade_price <= tp {
                    close_position_impl(config, vo, tp, ts, ExitReason::TakeProfit);
                    return true;
                }
            }
        }
    }
    false
}

// ============================================================================
// Fill Simulator — the core engine
// ============================================================================

pub struct FillSimulator {
    pub config: SimConfig,
    pub book: OrderBook,
    pub virtual_orders: Vec<VirtualOrder>,
    pub completed_trades: Vec<TradeResult>,
    pub next_order_id: u64,

    // State tracking
    current_mid: f64,
    current_best_bid: f64,
    current_best_ask: f64,
    current_bar_ns: u64,

    // Statistics
    pub total_signals: u64,
    pub total_posted: u64,
    pub total_filled: u64,
    pub total_cancelled: u64,
}

impl FillSimulator {
    pub fn new(config: SimConfig) -> Self {
        FillSimulator {
            config,
            book: OrderBook::new(10),  // 10 depth levels
            virtual_orders: Vec::new(),
            completed_trades: Vec::new(),
            next_order_id: 1,
            current_mid: 0.0,
            current_best_bid: 0.0,
            current_best_ask: 0.0,
            current_bar_ns: 0,
            total_signals: 0,
            total_posted: 0,
            total_filled: 0,
            total_cancelled: 0,
        }
    }

    /// Process a single MBO event through the simulator.
    /// This is called for EVERY event in the .dbn file.
    pub fn process_event(&mut self, event: &MboEvent) {
        let action = lob::parse_action(event.action);
        let price = event.price_raw as f64 * 1e-9;

        // 1. Update the real order book
        self.book.update(
            event.action,
            event.side,
            event.price_raw,
            event.size,
            event.order_id,
            event.flags,
            event.sequence,
            event.ts_event,
        );

        // 2. Update current market state
        self.update_market_state();

        // 3. Activate pending orders whose latency has elapsed
        self.activate_pending_orders(event.ts_event);

        // 4. On TRADE events: check virtual order fills and trailing stops
        if action == Action::Trade {
            self.process_trade_event(price, event.size, event.ts_event, event.side);
        }

        // 5. Chase reprice: if enabled, check if unfilled orders need to move to new BBO
        if self.config.chase_entry {
            self.chase_reprice_orders(event.ts_event);
        }

        // 6. Check time-based exits on every event (cheap check)
        self.check_time_exits(event.ts_event);
    }

    /// Update bid/ask/mid from current book state
    fn update_market_state(&mut self) {
        if let Some((&bid_price, _)) = self.book.bids().iter().next_back() {
            self.current_best_bid = bid_price.into_inner();
        }
        if let Some((&ask_price, _)) = self.book.asks().iter().next() {
            self.current_best_ask = ask_price.into_inner();
        }
        if self.current_best_bid > 0.0 && self.current_best_ask > 0.0 {
            self.current_mid = (self.current_best_bid + self.current_best_ask) / 2.0;
        }
    }

    /// Submit a new virtual order based on a signal.
    /// If order_latency_ns > 0, the order starts in PendingEntry state
    /// and only enters the book after the latency delay elapses.
    /// Returns the order ID if posted, None if rejected.
    pub fn submit_signal(&mut self, signal: &Signal) -> Option<u64> {
        self.total_signals += 1;

        // Determine side and entry price based on entry mode
        let (side, entry_price) = if self.config.market_entry {
            // Market order: cross the spread. Buy at ask, sell at bid.
            if signal.direction > 0.0 {
                (OrderSide::Buy, self.current_best_ask)
            } else {
                (OrderSide::Sell, self.current_best_bid)
            }
        } else if self.config.mid_price_entry {
            // Mid-price: post at (bid+ask)/2, splitting the spread.
            // Cheaper than market, better position than passive BBO.
            let mid = self.current_mid;
            if signal.direction > 0.0 {
                (OrderSide::Buy, mid)
            } else {
                (OrderSide::Sell, mid)
            }
        } else {
            // Passive limit: post at BBO. Buy at bid, sell at ask.
            if signal.direction > 0.0 {
                (OrderSide::Buy, self.current_best_bid)
            } else {
                (OrderSide::Sell, self.current_best_ask)
            }
        };

        if entry_price <= 0.0 {
            return None;
        }

        let order_id = self.next_order_id;
        self.next_order_id += 1;

        // Market orders: instant fill, no queue
        if self.config.market_entry {
            let hold_ns = self.config.hold_ms * 1_000_000;
            let mut vo = VirtualOrder {
                id: order_id,
                side,
                entry_price,
                size: self.config.order_size,
                state: VirtualOrderState::Filled,
                queue_position: 0.0,
                book_size_at_post: 0.0,
                post_time_ns: signal.bar_ns,
                fill_time_ns: Some(signal.bar_ns),
                fill_price: Some(entry_price),
                exit_price: None,
                exit_time_ns: None,
                exit_reason: None,
                hold_until_ns: Some(signal.bar_ns + hold_ns),
                trailing_stop_price: None,
                stop_loss_price: None,
                take_profit_price: None,
                best_price_seen: entry_price,
                worst_price_seen: entry_price,
                mae_ticks: 0.0,
                mfe_ticks: 0.0,
                recent_adverse_ticks: 0.0,
                adverse_check_price: entry_price,
                signal_bar_ns: signal.bar_ns,
                signal_strength: signal.direction,
                original_entry_price: entry_price,
                ratchet_stop_price: None,
                chase_reprices: 0,
                last_chase_check_ns: signal.bar_ns,
                conviction_flip_counter: 0,
            };

            // Set trailing stop if configured
            if let Some(trail) = self.config.trailing_stop_ticks {
                let stop = match side {
                    OrderSide::Buy => entry_price - trail * TICK_SIZE,
                    OrderSide::Sell => entry_price + trail * TICK_SIZE,
                };
                vo.trailing_stop_price = Some(stop);
            }

            // Set fixed stop loss if configured (from entry, never moves)
            if let Some(sl) = self.config.stop_loss_ticks {
                let stop = match side {
                    OrderSide::Buy => entry_price - sl * TICK_SIZE,
                    OrderSide::Sell => entry_price + sl * TICK_SIZE,
                };
                vo.stop_loss_price = Some(stop);
            }

            // Set take profit if configured
            if let Some(tp) = self.config.take_profit_ticks {
                let tp_price = match side {
                    OrderSide::Buy => entry_price + tp * TICK_SIZE,
                    OrderSide::Sell => entry_price - tp * TICK_SIZE,
                };
                vo.take_profit_price = Some(tp_price);
            }

            self.virtual_orders.push(vo);
            self.total_posted += 1;
            self.total_filled += 1;
            return Some(order_id);
        }

        let latency = self.config.order_latency_ns;

        if latency == 0 {
            // Zero latency: order enters book immediately (original behavior)
            let book_size = match side {
                OrderSide::Buy => {
                    self.book.bids().get(&OrderedFloat(entry_price))
                        .copied().unwrap_or(0.0)
                }
                OrderSide::Sell => {
                    self.book.asks().get(&OrderedFloat(entry_price))
                        .copied().unwrap_or(0.0)
                }
            };

            let vo = VirtualOrder {
                id: order_id,
                side,
                entry_price,
                size: self.config.order_size,
                state: VirtualOrderState::InBook,
                queue_position: book_size,
                book_size_at_post: book_size,
                post_time_ns: signal.bar_ns,
                fill_time_ns: None,
                fill_price: None,
                exit_price: None,
                exit_time_ns: None,
                exit_reason: None,
                hold_until_ns: None,
                trailing_stop_price: None,
                stop_loss_price: None,
                take_profit_price: None,
                best_price_seen: entry_price,
                worst_price_seen: entry_price,
                mae_ticks: 0.0,
                mfe_ticks: 0.0,
                recent_adverse_ticks: 0.0,
                adverse_check_price: entry_price,
                signal_bar_ns: signal.bar_ns,
                signal_strength: signal.direction,
                original_entry_price: entry_price,
                ratchet_stop_price: None,
                chase_reprices: 0,
                last_chase_check_ns: signal.bar_ns,
                conviction_flip_counter: 0,
            };

            self.virtual_orders.push(vo);
            self.total_posted += 1;
        } else {
            // Latency > 0: order starts as PendingEntry, will be activated later.
            // We snapshot the intended price but queue position is determined
            // when the order actually arrives at the exchange (after latency).
            let vo = VirtualOrder {
                id: order_id,
                side,
                entry_price,
                size: self.config.order_size,
                state: VirtualOrderState::PendingEntry,
                queue_position: 0.0,       // Set when activated
                book_size_at_post: 0.0,    // Set when activated
                post_time_ns: signal.bar_ns + latency,  // Arrives after latency
                fill_time_ns: None,
                fill_price: None,
                exit_price: None,
                exit_time_ns: None,
                exit_reason: None,
                hold_until_ns: None,
                trailing_stop_price: None,
                stop_loss_price: None,
                take_profit_price: None,
                best_price_seen: entry_price,
                worst_price_seen: entry_price,
                mae_ticks: 0.0,
                mfe_ticks: 0.0,
                recent_adverse_ticks: 0.0,
                adverse_check_price: entry_price,
                signal_bar_ns: signal.bar_ns,
                signal_strength: signal.direction,
                original_entry_price: entry_price,
                ratchet_stop_price: None,
                chase_reprices: 0,
                last_chase_check_ns: signal.bar_ns + latency,
                conviction_flip_counter: 0,
            };

            self.virtual_orders.push(vo);
            self.total_posted += 1;
        }

        Some(order_id)
    }

    /// Activate pending orders whose latency delay has elapsed.
    /// Called on every event. When activated, we read the CURRENT book state
    /// (which may have changed during the latency window) to set queue position.
    /// If the BBO has moved away from our intended price, the order is still
    /// posted at the original price but deeper in queue or at a stale level.
    fn activate_pending_orders(&mut self, ts: u64) {
        for vo in self.virtual_orders.iter_mut() {
            if vo.state != VirtualOrderState::PendingEntry {
                continue;
            }
            if ts < vo.post_time_ns {
                continue;  // Latency not yet elapsed
            }

            // Order arrives at exchange now. Read current book for queue position.
            let book_size = match vo.side {
                OrderSide::Buy => {
                    self.book.bids().get(&OrderedFloat(vo.entry_price))
                        .copied().unwrap_or(0.0)
                }
                OrderSide::Sell => {
                    self.book.asks().get(&OrderedFloat(vo.entry_price))
                        .copied().unwrap_or(0.0)
                }
            };

            // Check if our price is still at or better than BBO
            let price_still_valid = match vo.side {
                OrderSide::Buy => vo.entry_price >= self.current_best_bid - TICK_SIZE * 0.01,
                OrderSide::Sell => vo.entry_price <= self.current_best_ask + TICK_SIZE * 0.01,
            };

            if !price_still_valid || book_size <= 0.0 {
                // BBO moved away during latency — cancel the order
                vo.state = VirtualOrderState::Cancelled;
                self.total_cancelled += 1;
                continue;
            }

            // Activate: join the back of the queue at the current book depth
            vo.state = VirtualOrderState::InBook;
            vo.queue_position = book_size;
            vo.book_size_at_post = book_size;
        }
    }

    /// Process a real trade event — check if it fills our virtual orders
    /// and update trailing stops.
    fn process_trade_event(&mut self, trade_price: f64, trade_size: u32, ts: u64, side_byte: u8) {
        let passive_side = lob::parse_side(side_byte);

        // Use index-based iteration to satisfy the borrow checker
        let config = self.config.clone();
        let n = self.virtual_orders.len();

        for i in 0..n {
            let vo = &mut self.virtual_orders[i];
            match vo.state {
                VirtualOrderState::InBook => {
                    // Check if this trade is at our price level on our side
                    let at_our_level = match vo.side {
                        // We're buying at bid — trades hit the bid (passive=Bid, aggressor=Sell)
                        OrderSide::Buy => {
                            passive_side == Side::Bid
                                && (trade_price - vo.entry_price).abs() < TICK_SIZE * 0.01
                        }
                        // We're selling at ask — trades hit the ask (passive=Ask, aggressor=Buy)
                        OrderSide::Sell => {
                            passive_side == Side::Ask
                                && (trade_price - vo.entry_price).abs() < TICK_SIZE * 0.01
                        }
                    };

                    if at_our_level {
                        // Decrement queue position
                        vo.queue_position -= trade_size as f64;

                        // FILL when queue is depleted
                        if vo.queue_position <= 0.0 {
                            vo.state = VirtualOrderState::Filled;
                            vo.fill_time_ns = Some(ts);
                            vo.fill_price = Some(vo.entry_price);
                            vo.best_price_seen = vo.entry_price;

                            // Hold timer starts at FILL time, not signal time
                            vo.hold_until_ns = Some(
                                ts + config.hold_ms * 1_000_000
                            );

                            // Set trailing stop from fill price
                            if let Some(ts_ticks) = config.trailing_stop_ticks {
                                vo.trailing_stop_price = Some(match vo.side {
                                    OrderSide::Buy => vo.entry_price - ts_ticks * TICK_SIZE,
                                    OrderSide::Sell => vo.entry_price + ts_ticks * TICK_SIZE,
                                });
                            }

                            // Set fixed stop loss from fill price (never moves)
                            if let Some(sl_ticks) = config.stop_loss_ticks {
                                vo.stop_loss_price = Some(match vo.side {
                                    OrderSide::Buy => vo.entry_price - sl_ticks * TICK_SIZE,
                                    OrderSide::Sell => vo.entry_price + sl_ticks * TICK_SIZE,
                                });
                            }

                            // Set take profit
                            if let Some(tp_ticks) = config.take_profit_ticks {
                                vo.take_profit_price = Some(match vo.side {
                                    OrderSide::Buy => vo.entry_price + tp_ticks * TICK_SIZE,
                                    OrderSide::Sell => vo.entry_price - tp_ticks * TICK_SIZE,
                                });
                            }

                            self.total_filled += 1;
                        }
                    }
                }
                VirtualOrderState::Filled => {
                    // Position is open — check trailing stop and take profit on EVERY trade
                    check_exit_on_trade(&config, vo, trade_price, ts);
                }
                _ => {}
            }
        }
    }

    /// Chase/reprice: for unfilled InBook orders, check if BBO has moved away.
    /// If so, cancel the old order and reprice to the new BBO (back of queue).
    /// Respects chase_check_interval_ns, chase_max_ticks, and chase_max_reprices.
    fn chase_reprice_orders(&mut self, ts: u64) {
        let interval = self.config.chase_check_interval_ns;
        let max_ticks = self.config.chase_max_ticks;
        let max_reprices = self.config.chase_max_reprices;
        let force_cross = self.config.chase_force_cross;

        for vo in self.virtual_orders.iter_mut() {
            if vo.state != VirtualOrderState::InBook {
                continue;
            }

            // Rate-limit chase checks
            if ts < vo.last_chase_check_ns + interval {
                continue;
            }
            vo.last_chase_check_ns = ts;

            // Check if BBO has moved away from our resting price
            let bbo_moved = match vo.side {
                // We're buying at bid. BBO moved if current best_bid > our price
                // (bid improved away from us — we're deeper in queue at a worse level)
                OrderSide::Buy => self.current_best_bid > vo.entry_price + TICK_SIZE * 0.01,
                // We're selling at ask. BBO moved if current best_ask < our price
                OrderSide::Sell => self.current_best_ask < vo.entry_price - TICK_SIZE * 0.01,
            };

            if !bbo_moved {
                continue; // Still at BBO, keep waiting
            }

            // BBO moved — check if we've exceeded chase limits
            let new_price = match vo.side {
                OrderSide::Buy => self.current_best_bid,
                OrderSide::Sell => self.current_best_ask,
            };
            let chase_distance = ((new_price - vo.original_entry_price).abs() / TICK_SIZE).round();
            let exceeded_ticks = max_ticks > 0.0 && chase_distance > max_ticks;
            let exceeded_reprices = max_reprices > 0 && vo.chase_reprices >= max_reprices;

            if exceeded_ticks || exceeded_reprices {
                if force_cross {
                    // Force market order: fill at opposite BBO
                    let fill_price = match vo.side {
                        OrderSide::Buy => self.current_best_ask,
                        OrderSide::Sell => self.current_best_bid,
                    };
                    if fill_price <= 0.0 {
                        vo.state = VirtualOrderState::Cancelled;
                        self.total_cancelled += 1;
                        continue;
                    }
                    vo.entry_price = fill_price;
                    vo.state = VirtualOrderState::Filled;
                    vo.fill_time_ns = Some(ts);
                    vo.fill_price = Some(fill_price);
                    vo.best_price_seen = fill_price;
                    let hold_ns = self.config.hold_ms * 1_000_000;
                    vo.hold_until_ns = Some(ts + hold_ns);

                    if let Some(trail) = self.config.trailing_stop_ticks {
                        vo.trailing_stop_price = Some(match vo.side {
                            OrderSide::Buy => fill_price - trail * TICK_SIZE,
                            OrderSide::Sell => fill_price + trail * TICK_SIZE,
                        });
                    }
                    if let Some(sl) = self.config.stop_loss_ticks {
                        vo.stop_loss_price = Some(match vo.side {
                            OrderSide::Buy => fill_price - sl * TICK_SIZE,
                            OrderSide::Sell => fill_price + sl * TICK_SIZE,
                        });
                    }
                    if let Some(tp) = self.config.take_profit_ticks {
                        vo.take_profit_price = Some(match vo.side {
                            OrderSide::Buy => fill_price + tp * TICK_SIZE,
                            OrderSide::Sell => fill_price - tp * TICK_SIZE,
                        });
                    }
                    self.total_filled += 1;
                } else {
                    // Cancel — too much chase
                    vo.state = VirtualOrderState::Cancelled;
                    self.total_cancelled += 1;
                }
                continue;
            }

            // Reprice to new BBO — join back of queue
            if new_price <= 0.0 {
                continue;
            }

            let new_book_size = match vo.side {
                OrderSide::Buy => {
                    self.book.bids().get(&OrderedFloat(new_price))
                        .copied().unwrap_or(0.0)
                }
                OrderSide::Sell => {
                    self.book.asks().get(&OrderedFloat(new_price))
                        .copied().unwrap_or(0.0)
                }
            };

            vo.entry_price = new_price;
            vo.queue_position = new_book_size; // Back of queue at new level
            vo.book_size_at_post = new_book_size;
            vo.chase_reprices += 1;
            // Each cancel-replace incurs round-trip latency (cancel msg + new order msg).
            // The next reprice cannot happen until order_latency_ns after this one,
            // on top of the normal chase_check_interval.
            vo.last_chase_check_ns = ts + self.config.order_latency_ns;
        }
    }

    /// Check if the current prediction signal has flipped sign relative to entry.
    /// Supports two modes:
    /// 1. Instant flip (signal_flip_exit=true, conviction_exit_bars=0): exit immediately on flip
    /// 2. Conviction exit (conviction_exit_bars>0): exit after N consecutive opposite bars
    ///    Optionally requires minimum magnitude (conviction_exit_mag>0).
    /// Returns true if any position was closed.
    pub fn check_signal_flip(&mut self, current_prediction: f64, ts: u64) -> bool {
        let use_instant = self.config.signal_flip_exit && self.config.conviction_exit_bars == 0;
        let use_conviction = self.config.conviction_exit_bars > 0;

        if !use_instant && !use_conviction {
            return false;
        }

        let config = self.config.clone();
        let current_mid = self.current_mid;
        let mut closed_any = false;

        for vo in self.virtual_orders.iter_mut() {
            if vo.state != VirtualOrderState::Filled {
                continue;
            }

            let entry_was_long = vo.signal_strength > 0.0;
            let current_is_short = current_prediction <= 0.0;
            let entry_was_short = vo.signal_strength < 0.0;
            let current_is_long = current_prediction >= 0.0;

            let flipped = (entry_was_long && current_is_short)
                || (entry_was_short && current_is_long);

            if use_instant {
                // Original behavior: exit immediately on any flip
                if flipped {
                    close_position_impl(&config, vo, current_mid, ts, ExitReason::SignalFlip);
                    closed_any = true;
                }
            } else if use_conviction {
                // Conviction mode: count consecutive opposite bars
                let mag_ok = config.conviction_exit_mag <= 0.0
                    || current_prediction.abs() >= config.conviction_exit_mag;

                if flipped && mag_ok {
                    vo.conviction_flip_counter += 1;
                } else {
                    // Reset counter: signal is back in our direction (or below magnitude)
                    vo.conviction_flip_counter = 0;
                }

                if vo.conviction_flip_counter >= config.conviction_exit_bars {
                    close_position_impl(&config, vo, current_mid, ts, ExitReason::ConvictionExit);
                    closed_any = true;
                }
            }
        }

        closed_any
    }

    /// Check time-based exits (hold timeout, entry timeout)
    fn check_time_exits(&mut self, ts: u64) {
        let config = self.config.clone();
        let current_mid = self.current_mid;

        for vo in self.virtual_orders.iter_mut() {
            match vo.state {
                VirtualOrderState::InBook => {
                    // Check if we've waited too long for a fill
                    let max_wait_ns = config.max_wait_bars * BAR_NS;
                    if ts > vo.post_time_ns + max_wait_ns {
                        vo.state = VirtualOrderState::Cancelled;
                        self.total_cancelled += 1;
                    }
                }
                VirtualOrderState::Filled => {
                    // Check hold timeout
                    if let Some(hold_until) = vo.hold_until_ns {
                        if ts >= hold_until {
                            close_position_impl(&config, vo, current_mid, ts, ExitReason::HoldTimeout);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Force close all open positions (end of day)
    pub fn close_all_positions(&mut self, ts: u64) {
        let config = self.config.clone();
        let best_bid = self.current_best_bid;
        let best_ask = self.current_best_ask;

        for vo in self.virtual_orders.iter_mut() {
            match vo.state {
                VirtualOrderState::Filled => {
                    let exit_price = match vo.side {
                        OrderSide::Buy => best_bid,   // Exit long at bid
                        OrderSide::Sell => best_ask,   // Exit short at ask
                    };
                    close_position_impl(&config, vo, exit_price, ts, ExitReason::EndOfDay);
                }
                VirtualOrderState::InBook => {
                    vo.state = VirtualOrderState::Cancelled;
                    self.total_cancelled += 1;
                }
                _ => {}
            }
        }
    }

    /// Collect completed trades into results
    pub fn collect_results(&mut self) -> Vec<TradeResult> {
        let mut results = Vec::new();

        for vo in self.virtual_orders.drain(..) {
            if vo.state == VirtualOrderState::Closed {
                let fill_price = vo.fill_price.unwrap_or(vo.entry_price);
                let exit_price = vo.exit_price.unwrap_or(self.current_mid);
                let fill_time = vo.fill_time_ns.unwrap_or(vo.post_time_ns);
                let exit_time = vo.exit_time_ns.unwrap_or(fill_time);

                // PnL calculation
                let raw_pnl_ticks = match vo.side {
                    OrderSide::Buy => (exit_price - fill_price) / TICK_SIZE,
                    OrderSide::Sell => (fill_price - exit_price) / TICK_SIZE,
                };
                // For market entries, the spread cost is already baked into
                // fill_price (buy at ask, sell at bid), so raw_pnl_ticks
                // naturally includes it. No extra deduction needed.
                // Commission and P&L scale by number of contracts (order_size).
                let size = vo.size as f64;
                let pnl_ticks = raw_pnl_ticks - self.config.commission_ticks;  // per-contract ticks
                let pnl_dollars = pnl_ticks * 12.50 * size; // ES $12.50 per tick per contract

                results.push(TradeResult {
                    order_id: vo.id,
                    side: match vo.side {
                        OrderSide::Buy => "BUY".to_string(),
                        OrderSide::Sell => "SELL".to_string(),
                    },
                    size: vo.size,
                    signal_time_ns: vo.signal_bar_ns,
                    post_time_ns: vo.post_time_ns,
                    fill_time_ns: fill_time,
                    exit_time_ns: exit_time,
                    entry_price: fill_price,
                    exit_price,
                    pnl_ticks,
                    pnl_dollars,
                    exit_reason: vo.exit_reason.unwrap_or(ExitReason::MarketExit),
                    queue_position_at_post: vo.book_size_at_post,
                    book_size_at_post: vo.book_size_at_post,
                    fill_latency_ns: fill_time.saturating_sub(vo.post_time_ns),
                    hold_duration_ns: exit_time.saturating_sub(fill_time),
                    mae_ticks: vo.mae_ticks,
                    mfe_ticks: vo.mfe_ticks,
                    signal_strength: vo.signal_strength,
                });
            }
        }

        self.completed_trades.extend(results.clone());
        results
    }

    /// Get simulation summary statistics
    pub fn summary(&self) -> SimSummary {
        let trades = &self.completed_trades;
        let n = trades.len() as f64;
        if n == 0.0 {
            return SimSummary::default();
        }

        let total_pnl: f64 = trades.iter().map(|t| t.pnl_dollars).sum();
        let winners: Vec<&TradeResult> = trades.iter().filter(|t| t.pnl_dollars > 0.0).collect();
        let losers: Vec<&TradeResult> = trades.iter().filter(|t| t.pnl_dollars <= 0.0).collect();

        let win_rate = winners.len() as f64 / n;
        let avg_win = if !winners.is_empty() {
            winners.iter().map(|t| t.pnl_dollars).sum::<f64>() / winners.len() as f64
        } else { 0.0 };
        let avg_loss = if !losers.is_empty() {
            losers.iter().map(|t| t.pnl_dollars).sum::<f64>() / losers.len() as f64
        } else { 0.0 };

        let gross_profit: f64 = winners.iter().map(|t| t.pnl_dollars).sum();
        let gross_loss: f64 = losers.iter().map(|t| t.pnl_dollars.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        // Per-trade Sharpe (simplified)
        let mean_pnl = total_pnl / n;
        let var_pnl: f64 = trades.iter()
            .map(|t| (t.pnl_dollars - mean_pnl).powi(2))
            .sum::<f64>() / n;
        let std_pnl = var_pnl.sqrt();
        let sharpe_per_trade = if std_pnl > 0.0 { mean_pnl / std_pnl } else { 0.0 };

        let fill_rate = self.total_filled as f64 / self.total_posted.max(1) as f64;
        let avg_queue = trades.iter().map(|t| t.queue_position_at_post).sum::<f64>() / n;
        let avg_fill_latency_ms = trades.iter()
            .map(|t| t.fill_latency_ns as f64 / 1_000_000.0)
            .sum::<f64>() / n;

        SimSummary {
            total_signals: self.total_signals,
            total_posted: self.total_posted,
            total_filled: self.total_filled,
            total_cancelled: self.total_cancelled,
            total_trades: trades.len() as u64,
            total_pnl_dollars: total_pnl,
            mean_pnl_per_trade: mean_pnl,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            sharpe_per_trade,
            fill_rate,
            avg_queue_position: avg_queue,
            avg_fill_latency_ms,
        }
    }
}

// ============================================================================
// Summary Statistics
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimSummary {
    pub total_signals: u64,
    pub total_posted: u64,
    pub total_filled: u64,
    pub total_cancelled: u64,
    pub total_trades: u64,
    pub total_pnl_dollars: f64,
    pub mean_pnl_per_trade: f64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    pub sharpe_per_trade: f64,
    pub fill_rate: f64,
    pub avg_queue_position: f64,
    pub avg_fill_latency_ms: f64,
}
