/// DBN file ingestion — reads .dbn and .dbn.zst files using the dbn crate v0.22.
///
/// ES contract rollover mapping (from Python's ingest.py):
///   ESU5 (instrument_id=14160)  until 2025-09-19
///   ESZ5 (instrument_id=294973) until 2025-12-19
///   ESH6 (unknown id)           after

use std::path::Path;
use anyhow::{Context, Result};

/// ES contract rollover schedule.
static ES_CONTRACT_SCHEDULE: &[(&str, u32, &str)] = &[
    ("2025-09-19", 14160,    "ESU5"),
    ("2025-12-19", 294973,   "ESZ5"),
    ("2026-03-20", 42140878, "ESH6"),
    ("2026-06-19", 51643782, "ESM6"),
];

pub fn get_es_instrument_id(filename: &str) -> Option<u32> {
    let date_str = extract_date_from_filename(filename)?;
    for &(cutoff, iid, _) in ES_CONTRACT_SCHEDULE {
        if date_str.as_str() < cutoff {
            return Some(iid);
        }
    }
    ES_CONTRACT_SCHEDULE.last().map(|&(_, iid, _)| iid)
}

pub fn get_es_contract_name(filename: &str) -> &'static str {
    let date_str = match extract_date_from_filename(filename) {
        Some(d) => d,
        None => return "unknown",
    };
    for &(cutoff, _, name) in ES_CONTRACT_SCHEDULE {
        if date_str.as_str() < cutoff {
            return name;
        }
    }
    ES_CONTRACT_SCHEDULE.last().map(|&(_, _, name)| name).unwrap_or("unknown")
}

fn extract_date_from_filename(filename: &str) -> Option<String> {
    let bytes = filename.as_bytes();
    for i in 0..bytes.len().saturating_sub(7) {
        if bytes[i..i + 8].iter().all(|b| b.is_ascii_digit()) {
            let s = &filename[i..i + 8];
            return Some(format!("{}-{}-{}", &s[..4], &s[4..6], &s[6..8]));
        }
    }
    None
}

/// A single MBO event extracted from DBN.
#[derive(Debug, Clone)]
pub struct MboEvent {
    pub action: u8,
    pub side: u8,
    pub price_raw: i64,    // Fixed-point: multiply by 1e-9
    pub size: u32,
    pub order_id: u64,
    pub flags: u8,
    pub sequence: u32,
    pub ts_event: u64,
    pub instrument_id: u32,
}

/// Read all MBO events from a .dbn or .dbn.zst file.
/// Returns events sorted by ts_event.
/// If filter_instrument_id finds 0 events, auto-detects the most common
/// instrument_id and retries.
pub fn read_mbo_file(
    path: &Path,
    filter_instrument_id: Option<u32>,
) -> Result<Vec<MboEvent>> {
    let events = read_mbo_file_inner(path, filter_instrument_id)?;

    // Auto-detect: if mapped ID found 0 events, try without filter
    if events.is_empty() && filter_instrument_id.is_some() {
        log::warn!(
            "  0 events with instrument_id={}. Auto-detecting...",
            filter_instrument_id.unwrap()
        );
        let all_events = read_mbo_file_inner(path, None)?;
        if all_events.is_empty() {
            return Ok(events);
        }

        // Find most common instrument_id
        let mut id_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for e in &all_events {
            *id_counts.entry(e.instrument_id).or_insert(0) += 1;
        }
        let (&best_id, &best_count) = id_counts.iter().max_by_key(|&(_, c)| c).unwrap();
        log::info!(
            "  Auto-detected instrument_id={} ({} events out of {})",
            best_id,
            best_count,
            all_events.len()
        );

        // Filter to most common ID
        let filtered: Vec<MboEvent> = all_events
            .into_iter()
            .filter(|e| e.instrument_id == best_id)
            .collect();
        return Ok(filtered);
    }

    Ok(events)
}

fn read_mbo_file_inner(
    path: &Path,
    filter_instrument_id: Option<u32>,
) -> Result<Vec<MboEvent>> {
    use dbn::{
        decode::{DynDecoder, DecodeRecord},
        MboMsg,
        VersionUpgradePolicy,
    };

    let filename = path.to_string_lossy().to_string();
    log::info!("Opening: {}", filename);

    let mut decoder = DynDecoder::from_file(path, VersionUpgradePolicy::AsIs)
        .with_context(|| format!("Failed to open DBN file: {}", path.display()))?;

    let mut events = Vec::new();
    let mut total_count = 0u64;

    loop {
        match decoder.decode_record::<MboMsg>() {
            Ok(Some(rec)) => {
                total_count += 1;
                let iid = rec.hd.instrument_id;

                if let Some(filter_id) = filter_instrument_id {
                    if iid != filter_id {
                        continue;
                    }
                }

                events.push(MboEvent {
                    action: rec.action as u8,
                    side: rec.side as u8,
                    price_raw: rec.price,
                    size: rec.size,
                    order_id: rec.order_id,
                    flags: rec.flags.raw(),
                    sequence: rec.sequence,
                    ts_event: rec.hd.ts_event,
                    instrument_id: iid,
                });
            }
            Ok(None) => break,
            Err(e) => {
                log::warn!("Error decoding record: {}", e);
                break;
            }
        }
    }

    if filter_instrument_id.is_some() {
        log::info!(
            "  Loaded {}/{} events (instrument_id={})",
            events.len(),
            total_count,
            filter_instrument_id.unwrap()
        );
    } else {
        log::info!("  Loaded {} events (all instruments)", total_count);
    }

    events.sort_unstable_by_key(|e| e.ts_event);
    Ok(events)
}

/// Resolve a .dbn path to the actual file.
///
/// Some downloads produce a directory named `foo.mbo.dbn/` containing a file
/// also named `foo.mbo.dbn` inside it.  This helper detects that pattern and
/// returns the inner file path.  If `path` is already a regular file it is
/// returned unchanged.
fn resolve_dbn_path(path: std::path::PathBuf) -> Option<std::path::PathBuf> {
    let meta = std::fs::metadata(&path).ok()?;
    if meta.is_file() {
        return Some(path);
    }
    if meta.is_dir() {
        // Look for a file with the same name inside the directory.
        let dir_name = path.file_name()?.to_owned();
        let inner = path.join(&dir_name);
        if inner.is_file() {
            log::debug!("Resolved directory {} -> {}", path.display(), inner.display());
            return Some(inner);
        }
        // Fallback: find any .dbn file inside the directory.
        if let Ok(entries) = std::fs::read_dir(&path) {
            for entry in entries.flatten() {
                let ep = entry.path();
                let en = ep.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                if ep.is_file() && (en.ends_with(".dbn") || en.ends_with(".dbn.zst")) {
                    log::debug!("Resolved directory {} -> {}", path.display(), ep.display());
                    return Some(ep);
                }
            }
        }
    }
    None
}

/// List all .dbn and .dbn.zst files in a directory, deduplicated by date prefix.
///
/// Handles both regular files and the case where a download produced a
/// directory named `foo.mbo.dbn/` that contains the actual file inside.
pub fn list_dbn_files(dir: &Path) -> Vec<std::path::PathBuf> {
    use std::collections::BTreeMap;

    let mut by_prefix: BTreeMap<String, std::path::PathBuf> = BTreeMap::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let raw_path = entry.path();
            let name = raw_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            if name.ends_with(".dbn") || name.ends_with(".dbn.zst") {
                // Resolve directories to the actual file path inside.
                let path = match resolve_dbn_path(raw_path) {
                    Some(p) => p,
                    None => continue,
                };

                let prefix = name.split('.').next().unwrap_or(&name).to_string();
                let is_zst = name.ends_with(".zst");
                let existing_is_zst = by_prefix.get(&prefix)
                    .map(|p| p.to_string_lossy().ends_with(".zst"))
                    .unwrap_or(true);

                if !by_prefix.contains_key(&prefix) || (!is_zst && existing_is_zst) {
                    by_prefix.insert(prefix, path);
                }
            }
        }
    }

    by_prefix.into_values().collect()
}
