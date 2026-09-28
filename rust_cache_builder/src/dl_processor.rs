/// Deep Learning Processor — replays MBO events and produces output for DL models.
///
/// This is an alternative to processor.rs that, instead of (or in addition to) computing
/// tabular features, collects raw data in formats suitable for deep learning:
///   - Event token sequences (for Transformers)
///   - Book snapshot tensors (for Spatial CNNs)
///   - Trade flow sequences (for LSTMs)
///
/// Each output mode produces its own NPZ file per trading day.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;

use crate::lob::{OrderBook, parse_action, parse_side, Action, Side};
use crate::rth::{is_within_rth, ts_to_et_date, is_weekday};
use crate::ingest::{read_mbo_file, get_es_instrument_id, get_es_contract_name};

use crate::event_tokens::{EventTokenCollector, EventTokenDayData, save_event_tokens_npz};
use crate::book_tensors::{
    BookTensorDayData, QueueAgeTracker, extract_book_tensor, save_book_tensors_npz,
};
use crate::trade_flow::{TradeFlowCollector, TradeFlowDayData, save_trade_flow_npz};

const MIN_SNAPSHOTS: usize = 100;
const MIN_PRICE_RANGE_PTS: f64 = 1.0;

/// Which deep learning output modes are enabled.
#[derive(Debug, Clone)]
pub struct DlOutputModes {
    pub events: bool,
    pub book: bool,
    pub trades: bool,
}

/// Result for a single date from deep learning processing.
#[derive(Debug)]
pub struct DlDateResult {
    pub date_str: String,
    pub n_bars: usize,
    pub price_range: f64,
    pub source_file: String,
    pub outputs_written: Vec<String>,
}

/// Per-date accumulator for all DL output types.
struct DlDayAccumulator {
    event_data: Option<EventTokenDayData>,
    book_data: Option<BookTensorDayData>,
    trade_data: Option<TradeFlowDayData>,
    mid_prices: Vec<f64>,
    n_bars: usize,
}

impl DlDayAccumulator {
    fn new(modes: &DlOutputModes) -> Self {
        DlDayAccumulator {
            event_data: if modes.events { Some(EventTokenDayData::new()) } else { None },
            book_data: if modes.book { Some(BookTensorDayData::new()) } else { None },
            trade_data: if modes.trades { Some(TradeFlowDayData::new()) } else { None },
            mid_prices: Vec::new(),
            n_bars: 0,
        }
    }
}

/// Process a single .dbn file for deep learning output.
///
/// Similar to processor::process_file but produces DL-format NPZ files.
pub fn process_file_dl(
    file_path: &Path,
    output_dir: &Path,
    depth_levels: usize,
    interval_ms: u64,
    canonical_owners: Arc<HashMap<String, String>>,
    modes: &DlOutputModes,
) -> Result<Vec<DlDateResult>> {
    let t_start = Instant::now();
    let filename = file_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    let instrument_id = get_es_instrument_id(&filename);
    let contract_name = get_es_contract_name(&filename);
    log::info!("[DL] Processing: {} (contract: {}, instrument_id: {:?})",
        filename, contract_name, instrument_id);

    if instrument_id.is_none() {
        log::warn!("[DL] Cannot determine instrument_id for {}, skipping", filename);
        return Ok(vec![]);
    }

    // Read all events
    let events = read_mbo_file(file_path, instrument_id)?;
    if events.is_empty() {
        log::warn!("[DL] No events loaded from {}", filename);
        return Ok(vec![]);
    }

    log::info!("[DL]   {} events loaded in {:.1}s", events.len(), t_start.elapsed().as_secs_f64());

    let interval_ns = interval_ms * 1_000_000;
    let mut book = OrderBook::new(depth_levels);

    // DL-specific collectors
    let mut event_collector = EventTokenCollector::new();
    let mut trade_collector = TradeFlowCollector::new();
    let mut queue_tracker = QueueAgeTracker::new();

    // Per-date accumulators
    let mut date_data: HashMap<String, DlDayAccumulator> = HashMap::new();

    let mut next_snapshot_ts: Option<u64> = None;
    let mut rth_count = 0usize;
    let t_replay = Instant::now();

    for (i, event) in events.iter().enumerate() {
        let ts = event.ts_event;

        if next_snapshot_ts.is_none() {
            next_snapshot_ts = Some(ts);
        }

        // Feed event into order book
        book.update(
            event.action,
            event.side,
            event.price_raw,
            event.size,
            event.order_id,
            event.flags,
            event.sequence,
            ts,
        );

        // Feed event into DL collectors
        let action = parse_action(event.action);
        let side = parse_side(event.side);

        // Event token collector: record all non-fill, non-clear events
        if modes.events {
            match action {
                Action::Add | Action::Cancel | Action::Modify | Action::Trade => {
                    event_collector.record_event(
                        event.action, event.side, event.price_raw, event.size, ts,
                    );
                }
                _ => {}
            }
        }

        // Trade flow collector: record only trades
        if modes.trades {
            if action == Action::Trade {
                // The side in the MBO record is the passive side.
                // We pass the original side byte through to the collector
                // which will interpret passive side -> aggressor direction.
                if let Some((_passive_side, _price, _size)) = get_order_info_for_trade(&book, event.order_id) {
                    trade_collector.record_trade(event.side, event.price_raw, event.size, ts);
                } else {
                    // Order not found in book but still record the trade
                    trade_collector.record_trade(event.side, event.price_raw, event.size, ts);
                }
            }
        }

        // Queue age tracker: track adds and removes for book tensor module
        if modes.book {
            let price = event.price_raw as f64 * 1e-9;
            match action {
                Action::Add => {
                    let is_bid = side == Side::Bid;
                    queue_tracker.on_add(event.order_id, price, is_bid, ts);
                }
                Action::Cancel | Action::Trade => {
                    queue_tracker.on_remove(event.order_id, book.bids(), book.asks());
                }
                Action::Clear => {
                    queue_tracker.reset();
                }
                _ => {}
            }
        }

        // Take snapshots at interval boundaries
        while let Some(snap_ts) = next_snapshot_ts {
            if ts < snap_ts { break; }

            if !is_within_rth(snap_ts) {
                // Even outside RTH, flush collectors to prevent stale data accumulation
                if modes.events { event_collector.flush_bar(0.0); }
                if modes.trades { trade_collector.flush_bar(0.0); }
                next_snapshot_ts = Some(snap_ts + interval_ns);
                continue;
            }

            let mid = book.current_mid();
            if mid.is_finite() && mid > 0.0 {
                let date_str = ts_to_et_date(snap_ts);
                let day = date_data.entry(date_str).or_insert_with(|| DlDayAccumulator::new(modes));

                // Event tokens
                if let Some(ref mut event_day) = day.event_data {
                    let tokens = event_collector.flush_bar(mid);
                    event_day.push_bar(&tokens, mid, snap_ts as i64);
                }

                // Book tensors
                if let Some(ref mut book_day) = day.book_data {
                    let tensor = extract_book_tensor(
                        book.bids(),
                        book.asks(),
                        book.bid_order_counts(),
                        book.ask_order_counts(),
                        &queue_tracker,
                        mid,
                        snap_ts,
                    );
                    book_day.push_bar(&tensor, mid, snap_ts as i64);
                }

                // Trade flow
                if let Some(ref mut trade_day) = day.trade_data {
                    let trades = trade_collector.flush_bar(mid);
                    trade_day.push_bar(&trades, mid, snap_ts as i64);
                }

                day.mid_prices.push(mid);
                day.n_bars += 1;
                rth_count += 1;
            } else {
                // Mid not valid: flush collectors anyway
                if modes.events { event_collector.flush_bar(0.0); }
                if modes.trades { trade_collector.flush_bar(0.0); }
            }

            next_snapshot_ts = Some(snap_ts + interval_ns);
        }

        // Log progress every 1M events
        if i > 0 && i % 1_000_000 == 0 {
            let elapsed = t_replay.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            log::info!(
                "[DL]     {}/{} events ({:.0}%), {:.0} evt/s, {} dates",
                i, events.len(), 100.0 * i as f64 / events.len() as f64,
                rate, date_data.len()
            );
        }
    }

    log::info!(
        "[DL]   Replay done: {} RTH bars across {} dates [{:.0}s]",
        rth_count, date_data.len(), t_start.elapsed().as_secs_f64()
    );

    // Validate and save each date's data
    let mut results = Vec::new();

    for (date_str, day) in &date_data {
        let n = day.n_bars;
        if n < MIN_SNAPSHOTS {
            log::info!("[DL]     {}: only {} bars (need {}), skipping", date_str, n, MIN_SNAPSHOTS);
            continue;
        }

        // Check price range
        let min_mid = day.mid_prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_mid = day.mid_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let price_range = max_mid - min_mid;

        if price_range < MIN_PRICE_RANGE_PTS {
            log::info!("[DL]     {}: flat (range={:.2}pts), skipping", date_str, price_range);
            continue;
        }

        if !is_weekday(date_str) {
            log::info!("[DL]     {}: weekend, skipping", date_str);
            continue;
        }

        // Check canonical ownership
        match canonical_owners.get(date_str) {
            None => {
                log::warn!("[DL]     {}: no canonical owner found, skipping", date_str);
                continue;
            }
            Some(owner) if owner == "existing" => {
                log::info!("[DL]     {}: already saved (resume mode), skipping", date_str);
                continue;
            }
            Some(owner) if owner != &filename => {
                log::info!("[DL]     {}: not canonical owner, skipping", date_str);
                continue;
            }
            Some(_) => { /* canonical owner, proceed */ }
        }

        let mut outputs_written = Vec::new();

        // Save event tokens NPZ
        if let Some(ref event_day) = day.event_data {
            if event_day.n_bars > 0 {
                let out_path = output_dir.join(format!("{}_event_tokens.npz", date_str));
                match save_event_tokens_npz(&out_path, event_day) {
                    Ok(_) => {
                        let size_mb = out_path.metadata().map(|m| m.len() as f64 / 1e6).ok();
                        log::info!("[DL]     SAVED {}_event_tokens.npz: {} bars, {:.1}MB",
                            date_str, event_day.n_bars, size_mb.unwrap_or(0.0));
                        outputs_written.push("event_tokens".to_string());
                    }
                    Err(e) => log::error!("[DL]     Failed to save event tokens for {}: {}", date_str, e),
                }
            }
        }

        // Save book tensors NPZ
        if let Some(ref book_day) = day.book_data {
            if book_day.n_bars > 0 {
                let out_path = output_dir.join(format!("{}_book_tensors.npz", date_str));
                match save_book_tensors_npz(&out_path, book_day) {
                    Ok(_) => {
                        let size_mb = out_path.metadata().map(|m| m.len() as f64 / 1e6).ok();
                        log::info!("[DL]     SAVED {}_book_tensors.npz: {} bars, {:.1}MB",
                            date_str, book_day.n_bars, size_mb.unwrap_or(0.0));
                        outputs_written.push("book_tensors".to_string());
                    }
                    Err(e) => log::error!("[DL]     Failed to save book tensors for {}: {}", date_str, e),
                }
            }
        }

        // Save trade flow NPZ
        if let Some(ref trade_day) = day.trade_data {
            if trade_day.n_bars > 0 {
                let out_path = output_dir.join(format!("{}_trade_flow.npz", date_str));
                match save_trade_flow_npz(&out_path, trade_day) {
                    Ok(_) => {
                        let size_mb = out_path.metadata().map(|m| m.len() as f64 / 1e6).ok();
                        log::info!("[DL]     SAVED {}_trade_flow.npz: {} bars, {:.1}MB",
                            date_str, trade_day.n_bars, size_mb.unwrap_or(0.0));
                        outputs_written.push("trade_flow".to_string());
                    }
                    Err(e) => log::error!("[DL]     Failed to save trade flow for {}: {}", date_str, e),
                }
            }
        }

        if !outputs_written.is_empty() {
            results.push(DlDateResult {
                date_str: date_str.clone(),
                n_bars: n,
                price_range,
                source_file: filename.clone(),
                outputs_written,
            });
        }
    }

    results.sort_by(|a, b| a.date_str.cmp(&b.date_str));
    Ok(results)
}

/// Helper to get order info from book for trade events.
/// Returns (passive_side, price, size) if the order exists.
fn get_order_info_for_trade(_book: &OrderBook, _order_id: u64) -> Option<(Side, f64, f64)> {
    // The order book's internal orders map is private, but we already have
    // the side/price from the MBO event itself. This function is kept as a
    // stub for potential future use when we need to cross-reference.
    None
}
