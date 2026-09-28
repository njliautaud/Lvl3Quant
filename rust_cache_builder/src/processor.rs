/// File processor — replays MBO events through OrderBook, takes snapshots, groups by date.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use anyhow::Result;

use crate::lob::OrderBook;
use crate::engineering::compute_features;
use crate::npz::{DayData, save_npz};
use crate::rth::{is_within_rth, ts_to_et_date, is_weekday};
use crate::ingest::{read_mbo_file, get_es_instrument_id, get_es_contract_name};

pub const SAMPLE_INTERVAL_MS: u64 = 100;
const MIN_SNAPSHOTS: usize = 100;
const MIN_PRICE_RANGE_PTS: f64 = 1.0;

#[derive(Debug)]
pub struct DateResult {
    pub date_str: String,
    pub n_snapshots: usize,
    pub price_range: f64,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub source_file: String,
}

/// Process a single .dbn/.dbn.zst file and save per-date NPZ files to output_dir.
/// Returns a list of DateResult for each saved date.
///
/// `canonical_owners` maps date_str → canonical_source_filename. This is pre-computed
/// at startup from the sorted file list. A file only writes NPZ output for dates where
/// its filename is the canonical owner (i.e., the filename date matches the output date).
///
/// This handles the case where each DBN file contains multiple days of RTH data:
/// the N+1 file starts before midnight UTC (evening of day N) so it "warms up" with
/// day N's book state and then captures day N's RTH as well as day N+1's RTH.
/// We must ensure only the canonical file (the one with the same date in its filename)
/// writes each date's NPZ.
///
/// Already-saved dates (from resume mode) are pre-populated in canonical_owners with
/// the value "existing" to prevent re-processing.
pub fn process_file(
    file_path: &Path,
    output_dir: &Path,
    depth_levels: usize,
    interval_ms: u64,
    canonical_owners: Arc<HashMap<String, String>>,
) -> Result<Vec<DateResult>> {
    let t_start = Instant::now();
    let filename = file_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    let instrument_id = get_es_instrument_id(&filename);
    let contract_name = get_es_contract_name(&filename);
    log::info!("Processing: {} (contract: {}, instrument_id: {:?})", filename, contract_name, instrument_id);

    if instrument_id.is_none() {
        log::warn!("Cannot determine instrument_id for {}, skipping", filename);
        return Ok(vec![]);
    }

    // Read all events
    let events = read_mbo_file(file_path, instrument_id)?;
    if events.is_empty() {
        log::warn!("No events loaded from {}", filename);
        return Ok(vec![]);
    }

    log::info!("  {} events loaded in {:.1}s", events.len(), t_start.elapsed().as_secs_f64());

    let interval_ns = interval_ms * 1_000_000;
    let mut book = OrderBook::new(depth_levels);

    // Accumulate snapshots per date
    // HashMap<date_str, DayData>
    let mut date_data: HashMap<String, DayData> = HashMap::new();

    let mut next_snapshot_ts: Option<u64> = None;
    let mut rth_count = 0usize;
    let mut non_rth_skipped = 0usize;
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

        // Take snapshots up to current timestamp
        while let Some(snap_ts) = next_snapshot_ts {
            if ts < snap_ts { break; }

            if !is_within_rth(snap_ts) {
                non_rth_skipped += 1;
                next_snapshot_ts = Some(snap_ts + interval_ns);
                continue;
            }

            let snapshot = book.snapshot();
            if snapshot.is_valid() {
                let (global, nodes) = compute_features(&snapshot, depth_levels);
                let date_str = ts_to_et_date(snap_ts);
                let day = date_data.entry(date_str).or_insert_with(DayData::new);
                day.push(&global, &nodes, snapshot.mid, snap_ts as i64);
                rth_count += 1;
            }

            next_snapshot_ts = Some(snap_ts + interval_ns);
        }

        // Log progress every 1M events
        if i > 0 && i % 1_000_000 == 0 {
            let elapsed = t_replay.elapsed().as_secs_f64();
            let rate = i as f64 / elapsed;
            let eta = (events.len() - i) as f64 / rate;
            log::info!(
                "    {}/{} events ({:.0}%), {:.0} evt/s, ETA {:.0}s, {} dates",
                i, events.len(), 100.0 * i as f64 / events.len() as f64,
                rate, eta, date_data.len()
            );
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    log::info!(
        "  Book replay done: {} RTH snapshots across {} dates, {} non-RTH skipped [{:.0}s]",
        rth_count, date_data.len(), non_rth_skipped, elapsed
    );

    // Validate and save each date's data
    let mut results = Vec::new();

    for (date_str, day) in &date_data {
        let n = day.n_rows;
        if n < MIN_SNAPSHOTS {
            log::info!("    {}: only {} snapshots (need {}), skipping", date_str, n, MIN_SNAPSHOTS);
            continue;
        }

        // Check price range
        let mids = &day.mid_prices;
        let min_mid = mids.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_mid = mids.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let price_range = max_mid - min_mid;

        if price_range < MIN_PRICE_RANGE_PTS {
            log::info!("    {}: flat (range={:.2}pts < {}), skipping", date_str, price_range, MIN_PRICE_RANGE_PTS);
            continue;
        }

        if !is_weekday(date_str) {
            log::info!("    {}: weekend, skipping", date_str);
            continue;
        }

        // Check canonical ownership: only write this date's NPZ if the current file
        // is the pre-assigned canonical owner for this date.
        //
        // canonical_owners was built at startup from the sorted file list, ensuring
        // that for any date appearing in multiple DBN files, only the file whose
        // filename date matches the output date (e.g., 20250714 file for 2025-07-14)
        // will write the NPZ. Adjacent files that contain this date's RTH data as
        // a "warm-up" overlap are silently skipped.
        match canonical_owners.get(date_str) {
            None => {
                // Date has no canonical owner registered - skip it.
                // This should not happen in normal operation.
                log::warn!("    {}: no canonical owner found, skipping", date_str);
                continue;
            }
            Some(owner) if owner == "existing" => {
                // Date was already saved in a previous run (resume mode).
                log::info!("    {}: already saved (resume mode), skipping", date_str);
                continue;
            }
            Some(owner) if owner != &filename => {
                // Current file is NOT the canonical owner for this date.
                log::info!(
                    "    {}: skipping (canonical owner is {}, current file is {})",
                    date_str, owner, filename
                );
                continue;
            }
            Some(_) => {
                // Current file IS the canonical owner. Proceed to save.
            }
        }

        // Save NPZ
        let out_path = output_dir.join(format!("{}_snapshots.npz", date_str));
        match save_npz(&out_path, day) {
            Ok(_) => {
                let size_mb = out_path.metadata().map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
                log::info!("    SAVED {}: {} snapshots, {:.1}MB", date_str, n, size_mb);
                results.push(DateResult {
                    date_str: date_str.clone(),
                    n_snapshots: n,
                    price_range,
                    open: *mids.first().unwrap_or(&0.0),
                    close: *mids.last().unwrap_or(&0.0),
                    high: max_mid,
                    low: min_mid,
                    source_file: filename.clone(),
                });
            }
            Err(e) => {
                log::error!("    Failed to save {}: {}", date_str, e);
            }
        }
    }

    results.sort_by(|a, b| a.date_str.cmp(&b.date_str));
    Ok(results)
}
