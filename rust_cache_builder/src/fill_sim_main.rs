/// fill_sim_cli — Rust MBO fill simulator CLI
///
/// Event-by-event FIFO queue fill simulation using real L3 order book data.
/// Reads predictions from NPZ, processes raw MBO events, outputs JSON results.
///
/// Usage:
///   fill_sim_cli --mbo-file data.dbn --predictions preds.npz --output result.json
///   fill_sim_cli --mbo-file data.dbn --predictions preds.npz --output result.json \
///     --hold-ms 30000 --trailing-ticks 4 --signal-threshold 0.7
///   fill_sim_cli --mbo-file data.dbn --predictions preds.npz --output result.json \
///     --config params.json --quiet

#[allow(dead_code)]

use std::path::PathBuf;
use std::io::Read;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Serialize, Deserialize};

mod lob;
mod engineering;
mod npz;
mod rth;
mod ingest;
mod processor;
mod event_tokens;
mod book_tensors;
mod trade_flow;
mod dl_processor;
mod fill_sim;

use ingest::{read_mbo_file, get_es_instrument_id};
use fill_sim::{FillSimulator, SimConfig, Signal, TradeResult, SimSummary};
use rth::{is_within_rth, is_within_prime_hours};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(
    name = "fill_sim_cli",
    about = "Rust MBO fill simulator — event-by-event FIFO queue simulation"
)]
struct Args {
    /// MBO .dbn or .dbn.zst file path
    #[arg(long)]
    mbo_file: PathBuf,

    /// Predictions NPZ file (must contain 'predictions' key)
    #[arg(long)]
    predictions: PathBuf,

    /// Output JSON file for results
    #[arg(long)]
    output: PathBuf,

    /// Config JSON file (optional, CLI flags override)
    #[arg(long)]
    config: Option<PathBuf>,

    /// Signal threshold — only trade when |prediction| > this value
    #[arg(long, default_value_t = 0.0)]
    signal_threshold: f64,

    /// Hold time in milliseconds (from fill time)
    #[arg(long)]
    hold_ms: Option<u64>,

    /// Trailing stop in ticks
    #[arg(long)]
    trailing_ticks: Option<f64>,

    /// Fixed stop loss in ticks (from entry price, never moves unlike trailing stop)
    #[arg(long)]
    stop_loss_ticks: Option<f64>,

    /// Take profit in ticks
    #[arg(long)]
    take_profit_ticks: Option<f64>,

    /// Vol-based exit: exit if price moves this many ticks against us within vol_exit_bars
    #[arg(long)]
    vol_exit_ticks: Option<f64>,

    /// Vol-based exit: window in bars (100ms each) to detect fast adverse move
    #[arg(long)]
    vol_exit_bars: Option<u64>,

    /// Max bars to wait for fill before cancelling
    #[arg(long)]
    max_wait_bars: Option<u64>,

    /// Order submission latency in milliseconds (default: 0 = instant)
    /// Models the delay between signal and order reaching the exchange.
    /// During this delay, the book can move and queue position may change.
    /// Typical: 1-5ms co-located, 10-20ms direct, 50ms+ retail.
    #[arg(long, default_value_t = 0)]
    latency_ms: u64,

    /// Exit position immediately when prediction signal flips sign (market exit, pays spread)
    #[arg(long, default_value_t = false)]
    signal_flip_exit: bool,

    /// Restrict trading to prime hours only: 10:30 AM - 2:30 PM ET.
    /// Signals outside this window are skipped. Shown in research to yield +2.7 Sharpe lift.
    #[arg(long, default_value_t = false)]
    prime_hours: bool,

    /// Use market orders (cross the spread) instead of passive limit orders.
    /// Guarantees 100% fill rate but entry is at opposite BBO (pays full spread).
    #[arg(long, default_value_t = false)]
    market_entry: bool,

    /// Mid-price entry: post limit orders at (bid+ask)/2, splitting the spread.
    /// Cheaper than market orders (saves ~0.5 ticks), but fill is not guaranteed.
    #[arg(long, default_value_t = false)]
    mid_price_entry: bool,

    /// Chase/cancel-replace mode: if BBO moves away from resting order,
    /// cancel and reprice to the new BBO. Middle ground between passive and market.
    #[arg(long, default_value_t = false)]
    chase_entry: bool,

    /// Max ticks to chase away from original entry price (0 = unlimited). Default: 2.
    #[arg(long, default_value_t = 2.0)]
    chase_max_ticks: f64,

    /// Max number of cancel-replace cycles (0 = unlimited). Default: 5.
    #[arg(long, default_value_t = 5)]
    chase_max_reprices: u32,

    /// After exceeding chase limits, force a market order instead of cancelling.
    #[arg(long, default_value_t = false)]
    chase_force_cross: bool,

    /// Chase check interval in milliseconds. Default: 100 (1 bar).
    #[arg(long, default_value_t = 100)]
    chase_interval_ms: u64,

    /// Number of contracts per order (default: 1). P&L and commissions scale linearly.
    /// Fill rate is unchanged (valid simplification for <20 lots on ES).
    #[arg(long, default_value_t = 1)]
    size: u32,

    /// Additional exit slippage in ticks, added to market exit spread cost.
    /// Models adverse price movement during exit latency window.
    /// E.g., 0.1-0.2 for ~150ms latency at typical ES volatility.
    #[arg(long, default_value_t = 0.0)]
    exit_slippage: f64,

    /// Enable ratcheting trailing stop based on MFE thresholds.
    /// As MFE crosses higher levels, the stop locks in progressively more profit.
    #[arg(long, default_value_t = false)]
    ratchet_stop: bool,

    /// MAE exit: exit if position is underwater by this many ticks (requires --mae-exit-hold-sec).
    /// Default: 0 (disabled). Typical: 10 ticks.
    #[arg(long, default_value_t = 0.0)]
    mae_exit_ticks: f64,

    /// MAE exit: minimum hold time in seconds before MAE exit triggers (requires --mae-exit-ticks).
    /// Default: 0 (disabled). Typical: 600 seconds.
    #[arg(long, default_value_t = 0.0)]
    mae_exit_hold_sec: f64,

    /// Conviction exit: number of consecutive bars the signal must be opposite
    /// before triggering exit. 0 = disabled. 100 = 10 seconds.
    /// When >0, replaces instant signal-flip-exit with delayed version.
    #[arg(long, default_value_t = 0)]
    conviction_exit_bars: u64,

    /// Conviction exit: minimum |z-score| of opposite signal to count.
    /// 0.0 = any opposite signal. 1.0 = require |prediction| >= 1.0.
    #[arg(long, default_value_t = 0.0)]
    conviction_exit_mag: f64,

    /// Queue position filter: skip trades where queue_position_at_post < this value.
    /// Use to study signals with good queue position only (e.g., near top-of-book).
    /// Default: 0 (no filter).
    #[arg(long, default_value_t = 0)]
    min_queue_pos: u32,

    /// Queue position filter: skip trades where queue_position_at_post > this value.
    /// Use to exclude signals posted far back in the queue.
    /// Default: 999999 (no filter).
    #[arg(long, default_value_t = 999_999)]
    max_queue_pos: u32,

    /// Time window start (ET, HH:MM format, e.g. "09:30"). Only simulate signals
    /// within this time window. Useful for time-of-day analysis (Track 2B).
    /// Default: empty string (no filter).
    #[arg(long, default_value_t = String::new())]
    time_window_start: String,

    /// Time window end (ET, HH:MM format, e.g. "16:00"). Must be paired with
    /// --time-window-start. Signals outside this window are skipped.
    /// Default: empty string (no filter).
    #[arg(long, default_value_t = String::new())]
    time_window_end: String,

    /// Suppress progress output
    #[arg(long)]
    quiet: bool,
}

// ============================================================================
// Config JSON format (matches Python sweep configs)
// ============================================================================

#[derive(Debug, Deserialize)]
struct ConfigJson {
    #[serde(default)]
    hold_ms: Option<u64>,
    #[serde(default)]
    trailing_stop_ticks: Option<f64>,
    #[serde(default)]
    take_profit_ticks: Option<f64>,
    #[serde(default)]
    signal_threshold: Option<f64>,
    #[serde(default)]
    max_wait_bars: Option<u64>,
    #[serde(default)]
    market_exit_spread_cost: Option<f64>,
    #[serde(default)]
    signal_flip_exit: Option<bool>,
    #[serde(default)]
    latency_ms: Option<u64>,
    #[serde(default)]
    prime_hours_only: Option<bool>,
    #[serde(default)]
    order_size: Option<u32>,
    #[serde(default)]
    exit_slippage_ticks: Option<f64>,
    #[serde(default)]
    ratchet_stop: Option<bool>,
    #[serde(default)]
    mae_exit_ticks: Option<f64>,
    #[serde(default)]
    mae_exit_hold_sec: Option<f64>,
    #[serde(default)]
    conviction_exit_bars: Option<u64>,
    #[serde(default)]
    conviction_exit_mag: Option<f64>,
}

// ============================================================================
// Output JSON format
// ============================================================================

#[derive(Serialize)]
struct OutputJson {
    // Metadata
    mbo_file: String,
    predictions_file: String,
    n_events: usize,
    n_rth_bars: usize,
    n_predictions: usize,

    // Config used
    config: SimConfig,
    signal_threshold: f64,

    // Order size (contracts per trade)
    order_size: u32,

    // Flat summary fields (for easy Python access)
    total_pnl_dollars: f64,
    total_trades: u64,
    total_signals: u64,
    total_posted: u64,
    total_filled: u64,
    total_cancelled: u64,
    win_rate: f64,
    fill_rate: f64,
    sharpe_per_trade: f64,
    profit_factor: f64,
    avg_queue_position: f64,
    avg_fill_latency_ms: f64,
    mean_pnl_per_trade: f64,
    avg_win: f64,
    avg_loss: f64,

    // Detailed
    summary: SimSummary,
    trades: Vec<TradeResult>,
    elapsed_secs: f64,
}

// ============================================================================
// NPZ reading (predictions format)
// ============================================================================

/// Read the 'predictions' array from an NPZ file as Vec<f64>.
fn read_predictions(path: &PathBuf) -> Result<Vec<f64>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open predictions: {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| "Invalid NPZ file")?;

    // Find the predictions.npy entry
    let entry_name = (0..archive.len())
        .filter_map(|i| {
            let f = archive.by_index(i).ok()?;
            let n = f.name().to_string();
            if n == "predictions.npy" {
                Some(n)
            } else {
                None
            }
        })
        .next()
        .ok_or_else(|| anyhow::anyhow!(
            "No 'predictions.npy' in NPZ. Available: {}",
            (0..archive.len())
                .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
                .collect::<Vec<_>>()
                .join(", ")
        ))?;

    let mut entry = archive.by_name(&entry_name)?;
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf)?;

    parse_npy_f64(&buf)
}

/// Parse a .npy buffer into Vec<f64>. Supports float32 and float64 dtypes.
fn parse_npy_f64(data: &[u8]) -> Result<Vec<f64>> {
    // Validate magic: \x93NUMPY
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        bail!("Invalid .npy magic bytes");
    }

    let major = data[6];
    let header_offset = if major == 1 {
        let hlen = u16::from_le_bytes([data[8], data[9]]) as usize;
        10 + hlen
    } else if major == 2 {
        let hlen = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        12 + hlen
    } else {
        bail!("Unsupported .npy version: {}", major);
    };

    let header = std::str::from_utf8(&data[if major == 1 { 10 } else { 12 }..header_offset])
        .unwrap_or("");

    let raw_data = &data[header_offset..];

    if header.contains("<f8") || header.contains("float64") {
        // float64 — direct read
        let n = raw_data.len() / 8;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * 8;
            let bytes: [u8; 8] = raw_data[start..start + 8].try_into().unwrap();
            result.push(f64::from_le_bytes(bytes));
        }
        Ok(result)
    } else if header.contains("<f4") || header.contains("float32") {
        // float32 — convert to f64
        let n = raw_data.len() / 4;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * 4;
            let bytes: [u8; 4] = raw_data[start..start + 4].try_into().unwrap();
            result.push(f32::from_le_bytes(bytes) as f64);
        }
        Ok(result)
    } else {
        bail!("Unsupported dtype in .npy header: {}", header);
    }
}

// ============================================================================
// Main
// ============================================================================

const BAR_NS: u64 = 100_000_000; // 100ms per bar

/// Parse "HH:MM" time string into minutes from midnight ET.
fn parse_hhmm(s: &str) -> Option<u32> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 { return None; }
    let h: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    if h > 23 || m > 59 { return None; }
    Some(h * 60 + m)
}

/// Check if a nanosecond UTC timestamp falls within a custom HH:MM time window (ET).
/// window_start_min and window_end_min are minutes from midnight ET.
fn is_within_time_window(timestamp_ns: u64, window_start_min: u32, window_end_min: u32) -> bool {
    if timestamp_ns == 0 { return false; }
    let ts_sec = (timestamp_ns / 1_000_000_000) as i64;
    // DST_END_2025: Nov 2, 2025 06:00 UTC
    const DST_END_2025_NS: u64 = 1_762_056_000_000_000_000;
    let et_offset: i64 = if timestamp_ns < DST_END_2025_NS { -4 } else { -5 };
    let et_sec = ts_sec + et_offset * 3600;
    let secs_in_day = et_sec.rem_euclid(86400) as u32;
    let time_minutes = secs_in_day / 60;
    time_minutes >= window_start_min && time_minutes < window_end_min
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.quiet {
        env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("info")
        ).init();
    }

    let t_start = Instant::now();

    // ---- Build SimConfig ----
    let mut sim_config = SimConfig::default();

    // From config file
    if let Some(config_path) = &args.config {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config: {}", config_path.display()))?;
        let cfg: ConfigJson = serde_json::from_str(&config_str)
            .with_context(|| format!("Invalid config JSON: {}", config_path.display()))?;

        if let Some(v) = cfg.hold_ms { sim_config.hold_ms = v; }
        if let Some(v) = cfg.trailing_stop_ticks { sim_config.trailing_stop_ticks = Some(v); }
        if let Some(v) = cfg.take_profit_ticks { sim_config.take_profit_ticks = Some(v); }
        if let Some(v) = cfg.max_wait_bars { sim_config.max_wait_bars = v; }
        if let Some(v) = cfg.market_exit_spread_cost { sim_config.market_exit_spread_cost = v; }
        if let Some(v) = cfg.signal_flip_exit { sim_config.signal_flip_exit = v; }
        if let Some(v) = cfg.latency_ms {
            sim_config.order_latency_ns = v * 1_000_000;
            sim_config.exit_latency_ns = Some(v * 1_000_000);
        }
        if let Some(v) = cfg.prime_hours_only { sim_config.prime_hours_only = v; }
        if let Some(v) = cfg.order_size { sim_config.order_size = v; }
        if let Some(v) = cfg.exit_slippage_ticks { sim_config.exit_slippage_ticks = v; }
        if let Some(v) = cfg.ratchet_stop { sim_config.ratchet_stop = v; }
        if let Some(v) = cfg.mae_exit_ticks { sim_config.mae_exit_ticks = v; }
        if let Some(v) = cfg.mae_exit_hold_sec { sim_config.mae_exit_hold_sec = v; }
        if let Some(v) = cfg.conviction_exit_bars { sim_config.conviction_exit_bars = v; }
        if let Some(v) = cfg.conviction_exit_mag { sim_config.conviction_exit_mag = v; }
    }

    // CLI overrides (highest priority)
    if let Some(v) = args.hold_ms { sim_config.hold_ms = v; }
    if let Some(v) = args.trailing_ticks { sim_config.trailing_stop_ticks = Some(v); }
    if let Some(v) = args.stop_loss_ticks { sim_config.stop_loss_ticks = Some(v); }
    if let Some(v) = args.take_profit_ticks { sim_config.take_profit_ticks = Some(v); }
    if let Some(v) = args.vol_exit_ticks { sim_config.vol_exit_ticks = Some(v); }
    if let Some(v) = args.vol_exit_bars { sim_config.vol_exit_bars = Some(v); }
    if let Some(v) = args.max_wait_bars { sim_config.max_wait_bars = v; }
    if args.latency_ms > 0 {
        sim_config.order_latency_ns = args.latency_ms * 1_000_000;
        sim_config.exit_latency_ns = Some(args.latency_ms * 1_000_000);
    }
    if args.signal_flip_exit {
        sim_config.signal_flip_exit = true;
    }
    if args.prime_hours {
        sim_config.prime_hours_only = true;
    }
    if args.market_entry {
        sim_config.market_entry = true;
        sim_config.market_entry_spread_ticks = 0.0; // Spread already in fill price
    }
    if args.mid_price_entry {
        sim_config.mid_price_entry = true;
    }
    if args.chase_entry {
        sim_config.chase_entry = true;
        sim_config.chase_max_ticks = args.chase_max_ticks;
        sim_config.chase_max_reprices = args.chase_max_reprices;
        sim_config.chase_force_cross = args.chase_force_cross;
        sim_config.chase_check_interval_ns = args.chase_interval_ms * 1_000_000;
    }
    if args.size > 0 {
        sim_config.order_size = args.size;
    }
    if args.exit_slippage > 0.0 {
        sim_config.exit_slippage_ticks = args.exit_slippage;
    }
    if args.ratchet_stop {
        sim_config.ratchet_stop = true;
    }
    if args.conviction_exit_bars > 0 {
        sim_config.conviction_exit_bars = args.conviction_exit_bars;
        sim_config.conviction_exit_mag = args.conviction_exit_mag;
    }
    if args.mae_exit_ticks > 0.0 {
        sim_config.mae_exit_ticks = args.mae_exit_ticks;
    }
    if args.mae_exit_hold_sec > 0.0 {
        sim_config.mae_exit_hold_sec = args.mae_exit_hold_sec;
    }

    let signal_threshold = args.signal_threshold;

    // Parse optional time window args (HH:MM ET format)
    let time_window: Option<(u32, u32)> = if !args.time_window_start.is_empty() && !args.time_window_end.is_empty() {
        match (parse_hhmm(&args.time_window_start), parse_hhmm(&args.time_window_end)) {
            (Some(start), Some(end)) => {
                if !args.quiet {
                    eprintln!("Time window filter: {:02}:{:02} - {:02}:{:02} ET",
                        start / 60, start % 60, end / 60, end % 60);
                }
                Some((start, end))
            }
            _ => {
                eprintln!("WARNING: Invalid --time-window-start/end format. Expected HH:MM. Filter disabled.");
                None
            }
        }
    } else {
        None
    };

    // Queue position filters (applied post-sim to completed_trades)
    let min_queue_pos = args.min_queue_pos as f64;
    let max_queue_pos = args.max_queue_pos as f64;
    let has_queue_filter = args.min_queue_pos > 0 || args.max_queue_pos < 999_999;
    if has_queue_filter && !args.quiet {
        eprintln!("Queue position filter: [{}, {}]", args.min_queue_pos, args.max_queue_pos);
    }

    let entry_mode = if sim_config.market_entry {
        "MARKET (cross spread)".to_string()
    } else if sim_config.mid_price_entry {
        "MID-PRICE (split spread)".to_string()
    } else if sim_config.chase_entry {
        format!("CHASE (max {}t, {} reprices, force_cross={})",
            sim_config.chase_max_ticks, sim_config.chase_max_reprices, sim_config.chase_force_cross)
    } else {
        "PASSIVE LIMIT (BBO)".to_string()
    };

    if !args.quiet {
        eprintln!("=== Rust Fill Sim CLI ===");
        eprintln!("Entry mode: {}", entry_mode);
        eprintln!("Config: hold={}ms trailing={:?}t threshold={:.2} prime_hours={} size={}",
            sim_config.hold_ms,
            sim_config.trailing_stop_ticks,
            signal_threshold,
            sim_config.prime_hours_only,
            sim_config.order_size);
    }

    // ---- Load MBO events ----
    let mbo_filename = args.mbo_file.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let instrument_id = get_es_instrument_id(&mbo_filename);

    if !args.quiet {
        eprintln!("Loading MBO: {} (instrument_id={:?})", args.mbo_file.display(), instrument_id);
    }

    let events = read_mbo_file(&args.mbo_file, instrument_id)?;
    let n_events = events.len();

    if !args.quiet {
        eprintln!("Loaded {} MBO events", n_events);
    }

    // ---- Load predictions ----
    if !args.quiet {
        eprintln!("Loading predictions: {}", args.predictions.display());
    }

    let predictions = read_predictions(&args.predictions)?;
    let n_predictions = predictions.len();

    if !args.quiet {
        let nonzero = predictions.iter().filter(|p| p.abs() > signal_threshold).count();
        eprintln!("Loaded {} predictions ({} above threshold {:.2})",
            n_predictions, nonzero, signal_threshold);
    }

    // ---- Run simulation ----
    let mut sim = FillSimulator::new(sim_config.clone());
    let mut rth_bar_index: usize = 0;     // Current bar within RTH (0-indexed)
    let mut last_bar_ns: u64 = 0;         // Last processed bar boundary (absolute)
    let mut in_rth = false;               // Currently within RTH?
    let mut has_position = false;         // Only one position at a time

    for event in &events {
        let event_bar_ns = event.ts_event / BAR_NS * BAR_NS; // Snap to bar boundary
        let event_in_rth = is_within_rth(event.ts_event);

        // Detect RTH transitions
        if event_in_rth && !in_rth {
            // Entering RTH — reset bar counter
            in_rth = true;
            last_bar_ns = event_bar_ns;
            rth_bar_index = 0;
        } else if !event_in_rth && in_rth {
            // Leaving RTH — close all positions
            in_rth = false;
            sim.close_all_positions(event.ts_event);
            sim.collect_results();
            has_position = false;
        }

        // Process event through the fill simulator
        sim.process_event(event);

        // Check if any position was closed or cancelled — update has_position
        let any_resolved = sim.virtual_orders.iter()
            .any(|vo| vo.state == fill_sim::VirtualOrderState::Closed
                   || vo.state == fill_sim::VirtualOrderState::Cancelled);
        if any_resolved {
            sim.collect_results();
            // Remove cancelled orders so they don't block new entries
            sim.virtual_orders.retain(|vo|
                vo.state != fill_sim::VirtualOrderState::Cancelled);
            has_position = sim.virtual_orders.iter()
                .any(|vo| vo.state == fill_sim::VirtualOrderState::Filled
                       || vo.state == fill_sim::VirtualOrderState::InBook
                       || vo.state == fill_sim::VirtualOrderState::PendingEntry);
        }

        // On new bar boundary within RTH: check predictions
        if in_rth && event_bar_ns > last_bar_ns {
            // Count bars since last check
            let bars_passed = ((event_bar_ns - last_bar_ns) / BAR_NS) as usize;
            last_bar_ns = event_bar_ns;

            // Process each bar we skipped over
            for _ in 0..bars_passed {
                if rth_bar_index < n_predictions {
                    let pred = predictions[rth_bar_index];

                    // Check signal flip exit before attempting new entry.
                    // If we have an open position and the signal has reversed sign,
                    // close it immediately (market exit, pays spread).
                    if has_position && (sim_config.signal_flip_exit || sim_config.conviction_exit_bars > 0) {
                        let flipped = sim.check_signal_flip(pred, event_bar_ns);
                        if flipped {
                            sim.collect_results();
                            sim.virtual_orders.retain(|vo|
                                vo.state != fill_sim::VirtualOrderState::Cancelled);
                            has_position = sim.virtual_orders.iter()
                                .any(|vo| vo.state == fill_sim::VirtualOrderState::Filled
                                       || vo.state == fill_sim::VirtualOrderState::InBook
                                       || vo.state == fill_sim::VirtualOrderState::PendingEntry);
                        }
                    }

                    // Submit signal if prediction exceeds threshold,
                    // we don't already have an open position,
                    // and (if prime_hours_only) the bar falls within prime hours.
                    let in_prime = !sim_config.prime_hours_only
                        || is_within_prime_hours(event_bar_ns);
                    let in_time_window = match time_window {
                        Some((start, end)) => is_within_time_window(event_bar_ns, start, end),
                        None => true,
                    };
                    if pred.abs() > signal_threshold && !has_position && in_prime && in_time_window {
                        let signal = Signal {
                            bar_ns: event_bar_ns,
                            direction: pred,
                            magnitude: pred.abs(),
                            confidence: pred.abs(),
                        };
                        if sim.submit_signal(&signal).is_some() {
                            has_position = true;
                        }
                    }
                }
                rth_bar_index += 1;
            }
        }

        // Update position tracking — check if our position was just filled or closed
        has_position = sim.virtual_orders.iter()
            .any(|vo| vo.state == fill_sim::VirtualOrderState::Filled
                   || vo.state == fill_sim::VirtualOrderState::InBook
                   || vo.state == fill_sim::VirtualOrderState::PendingEntry);
    }

    // End of day — close everything
    if let Some(last_event) = events.last() {
        sim.close_all_positions(last_event.ts_event);
    }
    sim.collect_results();

    // Apply queue position filter: remove trades outside [min_queue_pos, max_queue_pos].
    // queue_position_at_post is recorded per-trade in TradeResult; we post-filter here
    // so the fill simulator still runs normally (no skip during sim).
    if has_queue_filter {
        let before = sim.completed_trades.len();
        sim.completed_trades.retain(|t| {
            t.queue_position_at_post >= min_queue_pos
                && t.queue_position_at_post <= max_queue_pos
        });
        let after = sim.completed_trades.len();
        if !args.quiet {
            eprintln!("Queue pos filter [{:.0},{:.0}]: kept {}/{} trades",
                min_queue_pos, max_queue_pos, after, before);
        }
    }

    let summary = sim.summary();
    let elapsed = t_start.elapsed().as_secs_f64();

    // ---- Output results ----
    if !args.quiet {
        eprintln!("\n=== Results ===");
        eprintln!("Trades: {} (filled {} / posted {} / signals {})",
            summary.total_trades, summary.total_filled,
            summary.total_posted, summary.total_signals);
        eprintln!("PnL: ${:.2}", summary.total_pnl_dollars);
        eprintln!("Win rate: {:.1}%", summary.win_rate * 100.0);
        eprintln!("Fill rate: {:.1}%", summary.fill_rate * 100.0);
        eprintln!("Sharpe/trade: {:.3}", summary.sharpe_per_trade);
        eprintln!("Profit factor: {:.3}", summary.profit_factor);
        eprintln!("Avg queue pos: {:.1}", summary.avg_queue_position);
        eprintln!("Avg fill latency: {:.1}ms", summary.avg_fill_latency_ms);
        eprintln!("RTH bars processed: {}", rth_bar_index);
        eprintln!("Elapsed: {:.1}s ({:.0} events/sec)",
            elapsed, n_events as f64 / elapsed);
    }

    let output = OutputJson {
        mbo_file: mbo_filename,
        predictions_file: args.predictions.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string(),
        n_events,
        n_rth_bars: rth_bar_index,
        n_predictions,
        config: sim_config,
        signal_threshold,
        order_size: args.size,
        total_pnl_dollars: summary.total_pnl_dollars,
        total_trades: summary.total_trades,
        total_signals: summary.total_signals,
        total_posted: summary.total_posted,
        total_filled: summary.total_filled,
        total_cancelled: summary.total_cancelled,
        win_rate: summary.win_rate,
        fill_rate: summary.fill_rate,
        sharpe_per_trade: summary.sharpe_per_trade,
        profit_factor: summary.profit_factor,
        avg_queue_position: summary.avg_queue_position,
        avg_fill_latency_ms: summary.avg_fill_latency_ms,
        mean_pnl_per_trade: summary.mean_pnl_per_trade,
        avg_win: summary.avg_win,
        avg_loss: summary.avg_loss,
        summary,
        trades: sim.completed_trades.clone(),
        elapsed_secs: elapsed,
    };

    // Write output
    let json = serde_json::to_string_pretty(&output)?;

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, &json)?;

    if !args.quiet {
        eprintln!("\nResults written to: {}", args.output.display());
    }

    // Also print compact summary to stdout for scripting
    println!("{{\"pnl\":{:.2},\"trades\":{},\"wr\":{:.3},\"fr\":{:.3},\"sharpe\":{:.4},\"pf\":{:.3}}}",
        output.total_pnl_dollars,
        output.total_trades,
        output.win_rate,
        output.fill_rate,
        output.sharpe_per_trade,
        output.profit_factor);

    Ok(())
}
