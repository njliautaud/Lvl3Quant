/// Debug processor — traces the book replay to diagnose zero event counts.
/// Run with: cargo run --bin debug_ingest -- path/to/file.dbn.zst

use std::path::Path;

// Bring in the book and RTH modules
mod lob;
mod rth;
mod ingest;
mod engineering;
mod npz;
mod processor;

use ingest::{read_mbo_file, get_es_instrument_id};
use lob::OrderBook;
use rth::is_within_rth;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: debug_ingest <file.dbn.zst>");
        std::process::exit(1);
    }

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn")
    ).init();

    let path = Path::new(&args[1]);
    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let instrument_id = get_es_instrument_id(filename);
    println!("File: {}", filename);
    println!("Instrument ID filter: {:?}", instrument_id);

    let events = read_mbo_file(path, instrument_id).expect("Failed to read file");
    println!("Loaded {} events", events.len());

    if events.is_empty() {
        println!("No events! Exiting.");
        return;
    }

    println!("\nFirst 5 events after sort:");
    for e in events.iter().take(5) {
        println!("  ts={} action={} side={} flags={} price_raw={} order_id={}",
                 e.ts_event, e.action, e.side, e.flags, e.price_raw, e.order_id);
    }

    // Count events by flags
    let n_snapshot = events.iter().filter(|e| (e.flags & 32) != 0).count();
    let n_live = events.iter().filter(|e| (e.flags & 32) == 0).count();
    println!("\nEvent breakdown:");
    println!("  F_SNAPSHOT set (flags & 32): {} events", n_snapshot);
    println!("  Live (flags & 32 == 0):      {} events", n_live);

    // Count by action
    let n_add    = events.iter().filter(|e| e.action == b'A').count();
    let n_cancel = events.iter().filter(|e| e.action == b'C').count();
    let n_modify = events.iter().filter(|e| e.action == b'M').count();
    let n_trade  = events.iter().filter(|e| e.action == b'T').count();
    let n_fill   = events.iter().filter(|e| e.action == b'F').count();
    let n_clear  = events.iter().filter(|e| e.action == b'R').count();
    println!("  ADD: {}, CANCEL: {}, MODIFY: {}, TRADE: {}, FILL: {}, CLEAR: {}",
             n_add, n_cancel, n_modify, n_trade, n_fill, n_clear);

    // Find first live ADD event
    if let Some(e) = events.iter().find(|e| e.action == b'A' && (e.flags & 32) == 0) {
        println!("\nFirst live ADD event:");
        println!("  ts={} action={} side={} flags={} price_raw={}",
                 e.ts_event, e.action, e.side, e.flags, e.price_raw);
        println!("  Is within RTH: {}", is_within_rth(e.ts_event));
    }

    // Find first RTH event
    if let Some(e) = events.iter().find(|e| is_within_rth(e.ts_event)) {
        println!("\nFirst RTH event:");
        println!("  ts={} action={} side={} flags={} price_raw={}",
                 e.ts_event, e.action, e.side, e.flags, e.price_raw);
    }

    // Simulate the processor loop and trace first 5 RTH snapshots
    let interval_ns: u64 = 100 * 1_000_000;
    let mut book = OrderBook::new(10);
    let mut next_snapshot_ts: Option<u64> = None;
    let mut rth_count = 0usize;

    println!("\n--- Processor trace (first 5 RTH snapshots) ---");

    for (i, event) in events.iter().enumerate() {
        let ts = event.ts_event;

        if next_snapshot_ts.is_none() {
            next_snapshot_ts = Some(ts);
        }

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

        while let Some(snap_ts) = next_snapshot_ts {
            if ts < snap_ts { break; }

            if !is_within_rth(snap_ts) {
                next_snapshot_ts = Some(snap_ts + interval_ns);
                continue;
            }

            let snapshot = book.snapshot();
            rth_count += 1;

            if rth_count <= 5 {
                println!("RTH snapshot #{} at ts={}", rth_count, snap_ts);
                println!("  mid={:.4}, spread={:.4}", snapshot.mid, snapshot.spread);
                println!("  add_count={}, cancel_count={}, trade_count={}",
                         snapshot.add_count, snapshot.cancel_count, snapshot.trade_count);
                println!("  buy_vol={}, sell_vol={}", snapshot.buy_volume, snapshot.sell_volume);
                println!("  book valid: {}", snapshot.is_valid());
                println!("  After event #{} at ts={}", i, ts);
            }

            if rth_count >= 5 {
                println!("\nGot 5 RTH snapshots. Total events processed: {}/{}", i+1, events.len());
                return;
            }

            next_snapshot_ts = Some(snap_ts + interval_ns);
        }
    }

    println!("\nTotal RTH snapshots: {}", rth_count);
}
