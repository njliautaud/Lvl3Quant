#![allow(dead_code)]
/// lob_cache_builder — High-performance Rust cache builder for Lvl3Quant.
///
/// Replaces the slow Python rebuild_snapshot_caches.py.
/// Expected speedup: 60-100x (4-6 hours → 3-5 minutes).
///
/// Usage:
///   lob_cache_builder --input-dir /path/to/mbo/ --output-dir /path/to/cache/
///   lob_cache_builder --input-dir ./mbo --output-dir ./data/processed/cache --workers 8
///   lob_cache_builder --input-dir ./mbo --output-dir ./cache --resume
///   lob_cache_builder --input-dir ./mbo --output-dir ./cache --mode events
///   lob_cache_builder --input-dir ./mbo --output-dir ./cache --mode all

use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
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

use ingest::{list_dbn_files, get_es_instrument_id, get_es_contract_name};
use processor::process_file;
use dl_processor::{process_file_dl, DlOutputModes};

#[derive(Parser, Debug)]
#[command(author, version, about = "Lvl3Quant NPZ snapshot cache builder (Rust)")]
struct Args {
    /// Directory containing .dbn or .dbn.zst MBO files
    #[arg(long, default_value = "./mbo")]
    input_dir: PathBuf,

    /// Directory for output .npz files
    #[arg(long, default_value = "./data/processed/medium_snapshots_cache")]
    output_dir: PathBuf,

    /// Order book depth levels
    #[arg(long, default_value_t = 10)]
    depth: usize,

    /// Snapshot interval in milliseconds
    #[arg(long, default_value_t = 100)]
    interval_ms: u64,

    /// Number of parallel workers (0 = auto = number of CPU cores)
    #[arg(long, default_value_t = 0)]
    workers: usize,

    /// Skip source files already listed in cache_manifest.json
    #[arg(long)]
    resume: bool,

    /// Show what would be done without doing it
    #[arg(long)]
    dry_run: bool,

    /// Output mode: features (default, 96+180 tabular), events (event token sequences),
    /// book (book snapshot tensors), trades (trade flow sequences), all (all 4 outputs)
    #[arg(long, default_value = "features")]
    mode: String,
}

// ============================================================================
// Manifest (matches Python's cache_manifest.json format)
// ============================================================================

#[derive(Debug, Serialize, Deserialize, Default)]
struct CacheManifest {
    format_version: u32,
    description: String,
    config: ManifestConfig,
    dates: HashMap<String, DateManifestEntry>,
    processed_files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    date_range: Option<DateRange>,
    total_days: usize,
    total_snapshots: u64,
    build_time_sec: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_updated: Option<String>,
    builder: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct ManifestConfig {
    interval_ms: u64,
    depth_levels: usize,
    rth_only: bool,
    rth_start: String,
    rth_end: String,
    min_snapshots: u32,
    min_price_range_pts: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DateManifestEntry {
    source_file: String,
    n_snapshots: usize,
    price_range: f64,
    open: f64,
    close: f64,
    high: f64,
    low: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_size_mb: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DateRange {
    first: String,
    last: String,
}

fn load_manifest(path: &Path) -> Option<CacheManifest> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn save_manifest(path: &Path, manifest: &CacheManifest) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(path, json)?;
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let args = Args::parse();

    // Parse mode flag
    let mode = args.mode.to_lowercase();
    let valid_modes = ["features", "events", "book", "trades", "all"];
    if !valid_modes.contains(&mode.as_str()) {
        eprintln!("Invalid mode '{}'. Valid modes: {}", mode, valid_modes.join(", "));
        std::process::exit(1);
    }

    let run_features = mode == "features" || mode == "all";
    let run_dl = mode == "events" || mode == "book" || mode == "trades" || mode == "all";

    let dl_modes = DlOutputModes {
        events: mode == "events" || mode == "all",
        book: mode == "book" || mode == "all",
        trades: mode == "trades" || mode == "all",
    };

    let input_dir = args.input_dir.canonicalize()
        .unwrap_or(args.input_dir.clone());
    let output_dir = args.output_dir.clone();

    let mode_desc = match mode.as_str() {
        "features" => "TABULAR FEATURES (96 global + 20x9 node)",
        "events" => "EVENT TOKEN SEQUENCES (Transformer)",
        "book" => "BOOK SNAPSHOT TENSORS (Spatial CNN)",
        "trades" => "TRADE FLOW SEQUENCES (LSTM)",
        "all" => "ALL OUTPUTS (features + events + book + trades)",
        _ => "UNKNOWN",
    };

    println!("======================================================================");
    println!("LOB CACHE BUILDER (Rust) — ONE FILE PER TRADING DAY");
    println!("  Input dir:  {}", input_dir.display());
    println!("  Output dir: {}", output_dir.display());
    println!("  Depth:      {} levels", args.depth);
    println!("  Interval:   {}ms", args.interval_ms);
    println!("  Mode:       {} ({})", mode, mode_desc);
    println!("======================================================================");

    // Discover files
    let mut files = list_dbn_files(&input_dir);
    files.sort();
    println!("Found {} unique source files", files.len());

    if args.dry_run {
        for (i, f) in files.iter().enumerate() {
            let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let iid = get_es_instrument_id(name);
            let cname = get_es_contract_name(name);
            println!("  [{}] {} -> {} (id={:?})", i, name, cname, iid);
        }
        println!("\nDRY RUN — {} files would be processed", files.len());
        return Ok(());
    }

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Load or init manifest
    let manifest_path = output_dir.join("cache_manifest.json");
    let mut manifest = if args.resume {
        load_manifest(&manifest_path).unwrap_or_default()
    } else {
        // Clean start: delete old files
        let old_npz: Vec<_> = std::fs::read_dir(&output_dir)?
            .flatten()
            .filter(|e| e.path().extension().map(|x| x == "npz").unwrap_or(false))
            .map(|e| e.path())
            .collect();
        if !old_npz.is_empty() {
            println!("Deleting {} old .npz files...", old_npz.len());
            for p in &old_npz { let _ = std::fs::remove_file(p); }
        }
        CacheManifest::default()
    };

    manifest.format_version = 3;
    manifest.description = "Per-trading-day snapshot caches. One file = one calendar day. \
        Built from Databento MBO data. Rust implementation.".to_string();
    manifest.config = ManifestConfig {
        interval_ms: args.interval_ms,
        depth_levels: args.depth,
        rth_only: true,
        rth_start: "09:30 ET".to_string(),
        rth_end: "16:00 ET".to_string(),
        min_snapshots: 100,
        min_price_range_pts: 1.0,
    };
    manifest.builder = format!("rust v{}", env!("CARGO_PKG_VERSION"));

    let processed_files: HashSet<String> = manifest.processed_files.iter().cloned().collect();
    let saved_dates: HashSet<String> = manifest.dates.keys().cloned().collect();

    // Filter files to process
    let files_to_process: Vec<_> = files.iter()
        .filter(|f| {
            let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("");
            !processed_files.contains(name)
        })
        .collect();

    let n_skip = files.len() - files_to_process.len();
    if n_skip > 0 {
        println!("Skipping {} already-processed files (resume mode)", n_skip);
    }
    println!("Processing {} files...\n", files_to_process.len());

    // Build canonical_owners: a pre-computed map from date_str -> canonical_source_filename.
    //
    // Background: each Databento daily DBN file starts at midnight UTC on day N, which is
    // typically early evening (8 PM ET) on day N-1. The initialization block warms up the
    // book with the previous day's state, so the file can generate RTH snapshots for BOTH
    // day N-1 and day N. Every file thus contains RTH data for the previous calendar day
    // as well as (sometimes several days back).
    //
    // The canonical owner for date "YYYY-MM-DD" is the source file named
    // "glbx-mdp3-YYYYMMDD.mbo.*" — i.e., the filename whose date matches the output date.
    // Files sorted alphabetically and processed in order means earlier filenames are
    // canonical for earlier dates. We build this map from ALL files (not just unprocessed)
    // so that parallel workers share a consistent view.
    //
    // Already-saved dates are marked "existing" to prevent resume-mode re-processing.
    let canonical_owners: HashMap<String, String> = {
        let mut owners: HashMap<String, String> = HashMap::new();

        // First, mark already-saved dates as "existing" (resume mode protection)
        for date in &saved_dates {
            owners.insert(date.clone(), "existing".to_string());
        }

        // Then assign canonical ownership from files to process.
        // The canonical file for date "YYYY-MM-DD" is the file named "glbx-mdp3-YYYYMMDD.*".
        // Since files is sorted, we can use the filename date extraction.
        for file in &files {
            let fname = file.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            // Extract the date from the filename (e.g., "glbx-mdp3-20250714.mbo.dbn" -> "2025-07-14")
            if let Some(date_str) = extract_date_from_fname(&fname) {
                // Only register as canonical if not already saved (resume mode)
                owners.entry(date_str).or_insert(fname);
            }
        }

        owners
    };
    let n_canonical = canonical_owners.len();
    println!("Canonical date assignments: {} dates (incl. {} already saved)", n_canonical, saved_dates.len());

    let canonical_owners_arc = Arc::new(canonical_owners);

    // Set up Rayon thread pool
    let n_workers = if args.workers == 0 {
        num_cpus_available()
    } else {
        args.workers
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build_global()
        .ok();
    println!("Using {} worker thread(s)", n_workers);

    let t_start = Instant::now();

    // Shared state for accumulating results
    let manifest_arc = Arc::new(Mutex::new(manifest));
    let saved_dates_arc = Arc::new(Mutex::new(saved_dates));
    let total_new = Arc::new(Mutex::new(0usize));
    let total_snapshots = Arc::new(Mutex::new(0u64));

    // Progress bar
    let pb = ProgressBar::new(files_to_process.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} files {msg}")
        .unwrap_or_else(|_| ProgressStyle::default_bar()));

    let output_dir_arc = output_dir.clone();
    let pb_arc = Arc::new(pb);
    let depth = args.depth;
    let interval_ms = args.interval_ms;

    // ========================================================================
    // Process files in parallel
    // ========================================================================

    // Deep learning output mode tracking
    let dl_modes_arc = Arc::new(dl_modes.clone());
    let total_dl_dates = Arc::new(Mutex::new(0usize));

    files_to_process
        .par_iter()
        .for_each(|file_path| {
            let fname = file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            log::info!("Starting: {}", fname);

            // ---- Tabular features mode (original behavior) ----
            if run_features {
                match process_file(file_path, &output_dir_arc, depth, interval_ms, Arc::clone(&canonical_owners_arc)) {
                    Ok(results) => {
                        let mut manifest = manifest_arc.lock().unwrap();
                        let mut saved = saved_dates_arc.lock().unwrap();
                        let mut new_count = total_new.lock().unwrap();
                        let mut snap_count = total_snapshots.lock().unwrap();

                        for r in results {
                            if saved.contains(&r.date_str) {
                                log::info!("  {}: duplicate date, skipping", r.date_str);
                                continue;
                            }

                            let npz_path = output_dir_arc.join(format!("{}_snapshots.npz", r.date_str));
                            let size_mb = npz_path.metadata().map(|m| m.len() as f64 / 1e6).ok();

                            saved.insert(r.date_str.clone());
                            *snap_count += r.n_snapshots as u64;
                            *new_count += 1;

                            manifest.dates.insert(r.date_str.clone(), DateManifestEntry {
                                source_file: r.source_file,
                                n_snapshots: r.n_snapshots,
                                price_range: (r.price_range * 100.0).round() / 100.0,
                                open: (r.open * 100.0).round() / 100.0,
                                close: (r.close * 100.0).round() / 100.0,
                                high: (r.high * 100.0).round() / 100.0,
                                low: (r.low * 100.0).round() / 100.0,
                                cache_size_mb: size_mb.map(|s| (s * 10.0).round() / 10.0),
                            });
                        }

                        manifest.processed_files.push(fname.clone());

                        let mut all_dates: Vec<String> = saved.iter().cloned().collect();
                        all_dates.sort();
                        if !all_dates.is_empty() {
                            manifest.date_range = Some(DateRange {
                                first: all_dates.first().cloned().unwrap_or_default(),
                                last: all_dates.last().cloned().unwrap_or_default(),
                            });
                        }
                        manifest.total_days = manifest.dates.len();
                        manifest.total_snapshots = *snap_count;
                        manifest.build_time_sec = t_start.elapsed().as_secs();
                        manifest.last_updated = Some(chrono::Utc::now().to_rfc3339());

                        let n_dates = manifest.dates.len();

                        drop(snap_count);
                        drop(new_count);
                        drop(saved);

                        if let Err(e) = save_manifest(&manifest_path, &manifest) {
                            log::error!("Failed to save manifest: {}", e);
                        }
                        drop(manifest);

                        pb_arc.set_message(format!("{} dates saved", n_dates));
                    }
                    Err(e) => {
                        log::error!("Error processing {} (features): {}", fname, e);
                    }
                }
            }

            // ---- Deep learning output modes ----
            if run_dl {
                match process_file_dl(
                    file_path, &output_dir_arc, depth, interval_ms,
                    Arc::clone(&canonical_owners_arc), &dl_modes_arc,
                ) {
                    Ok(results) => {
                        let mut dl_count = total_dl_dates.lock().unwrap();
                        for r in &results {
                            *dl_count += 1;
                            log::info!("[DL] {} -> {} outputs: {}",
                                r.date_str, r.n_bars,
                                r.outputs_written.join(", "));
                        }
                    }
                    Err(e) => {
                        log::error!("Error processing {} (DL): {}", fname, e);
                    }
                }
            }

            pb_arc.inc(1);
        });

    pb_arc.finish_with_message("Done!");

    // Final manifest (only for features mode)
    if run_features {
        let mut manifest = manifest_arc.lock().unwrap();
        manifest.completed = Some(chrono::Utc::now().to_rfc3339());
        manifest.build_time_sec = t_start.elapsed().as_secs();
        save_manifest(&manifest_path, &manifest)?;
    }

    // Summary
    let elapsed = t_start.elapsed();
    let new_count = *total_new.lock().unwrap();
    let snap_count = *total_snapshots.lock().unwrap();
    let dl_count = *total_dl_dates.lock().unwrap();

    println!("\n======================================================================");
    println!("COMPLETE  [mode: {}]", mode);

    if run_features {
        let manifest = manifest_arc.lock().unwrap();
        println!("  [features] Trading days saved: {}", new_count);
        println!("  [features] Total unique days:  {}", manifest.dates.len());
        println!("  [features] Total snapshots:    {}", snap_count);
        if let Some(ref dr) = manifest.date_range {
            println!("  [features] Date range:         {} to {}", dr.first, dr.last);
        }
    }

    if run_dl {
        println!("  [DL] Trading days saved:       {}", dl_count);
        if dl_modes.events {
            println!("  [DL] Event tokens:             {}_event_tokens.npz per day", "YYYY-MM-DD");
        }
        if dl_modes.book {
            println!("  [DL] Book tensors:             {}_book_tensors.npz per day", "YYYY-MM-DD");
        }
        if dl_modes.trades {
            println!("  [DL] Trade flow:               {}_trade_flow.npz per day", "YYYY-MM-DD");
        }
    }

    println!("  Time:                          {:.1}s ({:.1} min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    println!("======================================================================");

    if run_features {
        let manifest = manifest_arc.lock().unwrap();
        if let Some(verify_date) = manifest.dates.keys().next() {
            let verify_path = output_dir.join(format!("{}_snapshots.npz", verify_date));
            if verify_path.exists() {
                println!("\nVerification ({}):", verify_path.file_name().unwrap().to_str().unwrap());
                println!("  global_features: ({}, {})", manifest.dates[verify_date].n_snapshots, engineering::TOTAL_GLOBAL);
                println!("  node_features:   ({}, {}, {})", manifest.dates[verify_date].n_snapshots, engineering::NUM_NODES, engineering::NODE_FEATURES);
            }
        }
        println!("\nRun alpha scan with: python alpha_discovery/run_overnight_alpha.py --skip-cache-rebuild");
    }

    if run_dl {
        println!("\nDL outputs ready. Load in Python:");
        if dl_modes.events {
            println!("  events = np.load('YYYY-MM-DD_event_tokens.npz')");
            println!("    events['event_sequences'].shape  # (n_bars, 200, 5)");
            println!("    events['sequence_lengths'].shape  # (n_bars,)");
        }
        if dl_modes.book {
            println!("  book = np.load('YYYY-MM-DD_book_tensors.npz')");
            println!("    book['book_tensors'].shape  # (n_bars, 20, 4)");
        }
        if dl_modes.trades {
            println!("  trades = np.load('YYYY-MM-DD_trade_flow.npz')");
            println!("    trades['trade_sequences'].shape  # (n_bars, 50, 5)");
            println!("    trades['trade_lengths'].shape  # (n_bars,)");
        }
    }

    Ok(())
}

fn num_cpus_available() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Extract the date string "YYYY-MM-DD" from a DBN filename.
/// E.g., "glbx-mdp3-20250714.mbo.dbn" -> Some("2025-07-14")
fn extract_date_from_fname(filename: &str) -> Option<String> {
    let bytes = filename.as_bytes();
    for i in 0..bytes.len().saturating_sub(7) {
        if bytes[i..i + 8].iter().all(|b| b.is_ascii_digit()) {
            let s = &filename[i..i + 8];
            return Some(format!("{}-{}-{}", &s[..4], &s[4..6], &s[6..8]));
        }
    }
    None
}
