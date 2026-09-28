// MBO Feature Expander — Rust port of alpha_discovery/mbo_features.py
// Suppress dead-code warnings (large function with many intermediate variables)
#![allow(dead_code, unused_variables, unused_imports)]
//
// Computes 290 features from the 96 global features and 20×9 node features
// stored in the NPZ snapshot cache files. Each day is processed independently
// to respect day boundaries (no cross-overnight leakage).

use std::io::{Read, Write, Seek, Cursor};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::sync::Arc;

use anyhow::{Result, Context};
use clap::Parser;
use rayon::prelude::*;
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions, CompressionMethod};
use std::fs::File;

// ============================================================================
// Constants matching Python mbo_features.py
// ============================================================================

pub const N_GLOBAL_FEATURES: usize = 96;
pub const TOTAL_FEATURES: usize = 290;
pub const DEPTH_LEVELS: usize = 10;
pub const NUM_NODES: usize = 20; // 2 * DEPTH_LEVELS
pub const NODE_FEATURES: usize = 9;

// Column indices for global_features_raw (matches Python COL_* constants)
const COL_MID: usize = 0;
const COL_SPREAD: usize = 1;
const COL_VOL_IMBALANCE: usize = 2;
const COL_MICROPRICE: usize = 3;
const COL_TOTAL_BID_VOL: usize = 4;
const COL_TOTAL_ASK_VOL: usize = 5;
const COL_MEAN_BID_SIZE: usize = 6;
const COL_MEAN_ASK_SIZE: usize = 7;
const COL_BEST_BID: usize = 8;
const COL_BEST_ASK: usize = 9;

const COL_TRADE_IMBALANCE: usize = 10;
const COL_BUY_VOL: usize = 11;
const COL_SELL_VOL: usize = 12;
const COL_ADD_COUNT: usize = 13;
const COL_CANCEL_COUNT: usize = 14;
const COL_TRADE_COUNT: usize = 15;
const COL_CANCEL_TO_ADD: usize = 16;
const COL_TRADE_TO_ADD: usize = 17;

const COL_BID_PRESSURE: usize = 18;
const COL_ASK_PRESSURE: usize = 19;
const COL_PRESSURE_IMBALANCE: usize = 20;
const COL_DEPTH_CONCENTRATION: usize = 21;
const COL_BID_SLOPE: usize = 22;
const COL_ASK_SLOPE: usize = 23;
const COL_SPREAD_TICKS: usize = 24;
const COL_DEPTH_RATIO: usize = 25;

const COL_HOUR_NORM: usize = 26;
const COL_MINUTE_NORM: usize = 27;
const COL_TIME_SINCE_RTH: usize = 28;
const COL_TIME_TO_CLOSE: usize = 29;
const COL_EVENT_DENSITY: usize = 30;

const COL_MODIFY_COUNT: usize = 31;
const COL_MODIFY_TO_ADD: usize = 32;
const COL_MEAN_LIFETIME: usize = 33;
const COL_FLEETING_RATIO: usize = 34;
const COL_AGGR_BUY_COUNT: usize = 35;
const COL_AGGR_SELL_COUNT: usize = 36;
const COL_AGGR_IMBALANCE: usize = 37;
const COL_MAX_TRADE_SIZE: usize = 38;
const COL_CANCEL_BID_VOL: usize = 39;
const COL_CANCEL_ASK_VOL: usize = 40;
const COL_CANCEL_SIDE_IMB: usize = 41;
const COL_TICK_COUNT: usize = 42;
const COL_SEQ_GAPS: usize = 43;
const COL_N_COMPLETED: usize = 44;

const COL_LARGE_TRADE_RATIO: usize = 45;
const COL_SMALL_TRADE_RATIO: usize = 46;
const COL_TRADE_SIZE_P75_LOG: usize = 47;
const COL_TRADE_HETEROGENEITY: usize = 48;
const COL_CANCEL_L1_RATIO: usize = 49;
const COL_CANCEL_L1_VOL_FRAC: usize = 50;
const COL_CANCEL_DEEP_VOL_FRAC: usize = 51;
const COL_ADD_L1_RATIO: usize = 52;
const COL_L1_NET_ACTIVITY: usize = 53;
const COL_CONSEC_BUY_LOG: usize = 54;
const COL_CONSEC_SELL_LOG: usize = 55;
const COL_SWEEP_ASYMMETRY: usize = 56;
const COL_SWEEP_INTENSITY: usize = 57;
const COL_MEAN_INTER_TRADE_LOG: usize = 58;
const COL_MIN_INTER_TRADE_LOG: usize = 59;
const COL_TRADE_BURSTINESS: usize = 60;
const COL_TRADE_RATE: usize = 61;
const COL_MODIFY_CHAIN_RATIO: usize = 62;
const COL_MAX_CHAIN_LOG: usize = 63;
const COL_CHAIN_PER_ORDER: usize = 64;
const COL_MULTI_LEVEL_RATIO: usize = 65;
const COL_MULTI_LEVEL_LOG: usize = 66;
const COL_LARGE_SWEEP_COMBO: usize = 67;
const COL_CANCEL_L1_TRADE_PRESS: usize = 68;
const COL_MODIFY_CHAIN_SWEEP: usize = 69;

const COL_BID_DEPTH_SKEW: usize = 70;
const COL_ASK_DEPTH_SKEW: usize = 71;
const COL_BID_GAP_COUNT: usize = 72;
const COL_ASK_GAP_COUNT: usize = 73;
const COL_MAX_BID_WALL_FRAC: usize = 74;
const COL_MAX_ASK_WALL_FRAC: usize = 75;
const COL_BID_DEPTH_COG: usize = 76;
const COL_ASK_DEPTH_COG: usize = 77;
const COL_BID_SIZE_ENTROPY: usize = 78;
const COL_ASK_SIZE_ENTROPY: usize = 79;
const COL_BOOK_SYMMETRY: usize = 80;
const COL_TOTAL_DEPTH_LOG: usize = 81;
const COL_L1_L3_BID_RATIO: usize = 82;
const COL_L1_L3_ASK_RATIO: usize = 83;
const COL_ORDER_FRAG_ASYM: usize = 84;

const COL_L1_BID_CHANGE_RATE: usize = 85;
const COL_L1_ASK_CHANGE_RATE: usize = 86;
const COL_L1_CHANGE_ASYMMETRY: usize = 87;
const COL_L1_DEPLETION_LOG: usize = 88;
const COL_DEPLETION_ASYMMETRY: usize = 89;
const COL_TRADE_AT_BID_FRAC: usize = 90;
const COL_TRADE_SIZE_CV: usize = 91;
const COL_TRADE_SIZE_SKEW_RAW: usize = 92;
const COL_REPOSITION_RATE: usize = 93;
const COL_MEAN_ADD_SIZE_LOG: usize = 94;
const COL_MAX_ADD_SIZE_NORM: usize = 95;

// Node feature indices
const NODE_SIZE_COL: usize = 2;
const NODE_ORDER_COUNT_COL: usize = 6;
const NODE_CONCENTRATION_COL: usize = 8;

const MAX_ROLLING_WINDOW: usize = 100;
const TICK_SIZE: f32 = 0.25;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug)]
#[command(author, version, about = "MBO Feature Expander — 96→290 features in Rust")]
pub struct ExpanderArgs {
    /// Directory containing per-day NPZ cache files (from lob_cache_builder)
    #[arg(long, default_value = "./data/processed/medium_snapshots_cache")]
    pub cache_dir: PathBuf,

    /// Output directory for expanded feature NPZ files
    #[arg(long, default_value = "./data/processed/mbo_features_cache")]
    pub output_dir: PathBuf,

    /// Number of parallel workers (0 = auto)
    #[arg(long, default_value_t = 0)]
    pub workers: usize,

    /// Resume: skip files that already have output
    #[arg(long)]
    pub resume: bool,

    /// Process only one specific date (for testing)
    #[arg(long)]
    pub date: Option<String>,
}

// ============================================================================
// NPZ I/O
// ============================================================================

/// Read a float32 array from an .npy entry in a zip archive.
/// Returns (data, shape) where shape is vec of dimensions.
fn read_npy_f32(archive: &mut ZipArchive<impl Read + Seek>, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let entry_name = format!("{}.npy", name);
    let mut entry = archive.by_name(&entry_name)
        .with_context(|| format!("Missing {} in NPZ", entry_name))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    parse_npy_f32(&bytes)
}

/// Read a float64 array from an .npy entry in a zip archive.
fn read_npy_f64(archive: &mut ZipArchive<impl Read + Seek>, name: &str) -> Result<(Vec<f64>, Vec<usize>)> {
    let entry_name = format!("{}.npy", name);
    let mut entry = archive.by_name(&entry_name)
        .with_context(|| format!("Missing {} in NPZ", entry_name))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    parse_npy_f64(&bytes)
}

/// Parse .npy bytes, returning (flat_data, shape).
fn parse_npy_header(bytes: &[u8]) -> Result<(Vec<usize>, usize, &str)> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        anyhow::bail!("Invalid .npy magic");
    }
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else {
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    };
    let prefix_len = if major == 1 { 10 } else { 12 };
    let header_str = std::str::from_utf8(&bytes[prefix_len..prefix_len + header_len])?;

    // Parse shape from header dict string
    let shape_start = header_str.find("'shape': (")
        .or_else(|| header_str.find("'shape':("))
        .with_context(|| "Cannot find shape in npy header")?;
    let shape_str_start = header_str[shape_start..].find('(').unwrap() + shape_start + 1;
    let shape_str_end = header_str[shape_str_start..].find(')').unwrap() + shape_str_start;
    let shape_str = &header_str[shape_str_start..shape_str_end];

    let shape: Vec<usize> = shape_str.split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let data_offset = prefix_len + header_len;

    // Determine dtype
    let dtype = if header_str.contains("'<f4'") || header_str.contains("\"<f4\"") {
        "<f4"
    } else if header_str.contains("'<f8'") || header_str.contains("\"<f8\"") {
        "<f8"
    } else if header_str.contains("'<i8'") || header_str.contains("\"<i8\"") {
        "<i8"
    } else {
        "<f4" // default
    };

    Ok((shape, data_offset, dtype))
}

fn parse_npy_f32(bytes: &[u8]) -> Result<(Vec<f32>, Vec<usize>)> {
    let (shape, offset, _dtype) = parse_npy_header(bytes)?;
    let n_elements: usize = shape.iter().product();
    let data_bytes = &bytes[offset..offset + n_elements * 4];
    let data: Vec<f32> = data_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((data, shape))
}

fn parse_npy_f64(bytes: &[u8]) -> Result<(Vec<f64>, Vec<usize>)> {
    let (shape, offset, dtype) = parse_npy_header(bytes)?;
    let n_elements: usize = shape.iter().product();
    // Handle both float32 and float64 storage (Python writes f32, Rust builder writes f64)
    if dtype == "<f4" || dtype == "|f4" {
        let data_bytes = &bytes[offset..offset + n_elements * 4];
        let data: Vec<f64> = data_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect();
        Ok((data, shape))
    } else {
        let data_bytes = &bytes[offset..offset + n_elements * 8];
        let data: Vec<f64> = data_bytes.chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
        Ok((data, shape))
    }
}

/// Load one day's NPZ file.
pub struct DayInput {
    pub global_features: Vec<f32>, // N * 96, row-major
    pub node_features: Vec<f32>,   // N * 20 * 9, row-major
    pub mid_prices: Vec<f64>,      // N
    pub n_bars: usize,
}

pub fn load_npz(path: &Path) -> Result<DayInput> {
    let cursor = Cursor::new(std::fs::read(path)?);
    let mut archive = ZipArchive::new(cursor)?;

    let (gf, gf_shape) = read_npy_f32(&mut archive, "global_features")?;
    let (nf, _nf_shape) = read_npy_f32(&mut archive, "node_features")?;
    let (mp, _mp_shape) = read_npy_f64(&mut archive, "mid_prices")?;

    let n = gf_shape[0];
    Ok(DayInput { global_features: gf, node_features: nf, mid_prices: mp, n_bars: n })
}

/// Write expanded features to NPZ.
pub fn write_features_npz(path: &Path, features: &[f32], n_bars: usize) -> Result<()> {
    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);

    // Build .npy bytes for mbo_features: (n_bars, TOTAL_FEATURES) float32
    let shape_str = format!("({}, {})", n_bars, TOTAL_FEATURES);
    let header_dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    let prefix_len = 10usize;
    let min_len = header_dict.len() + 1;
    let total_prefix = prefix_len + min_len;
    let pad_to = ((total_prefix + 63) / 64) * 64;
    let padding = pad_to - total_prefix;
    let header_padded = format!("{}{}\n", header_dict, " ".repeat(padding));
    let header_len = header_padded.len() as u16;

    let n_elements = n_bars * TOTAL_FEATURES;
    let mut npy_bytes = Vec::with_capacity(prefix_len + header_padded.len() + n_elements * 4);
    npy_bytes.extend_from_slice(b"\x93NUMPY");
    npy_bytes.push(1u8);
    npy_bytes.push(0u8);
    npy_bytes.extend_from_slice(&header_len.to_le_bytes());
    npy_bytes.extend_from_slice(header_padded.as_bytes());
    for &f in features {
        npy_bytes.extend_from_slice(&f.to_le_bytes());
    }

    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file("mbo_features.npy", options)?;
    zip.write_all(&npy_bytes)?;
    zip.finish()?;
    Ok(())
}

// ============================================================================
// Rolling window primitives (single-pass, O(n) each)
// ============================================================================

/// Causal rolling sum. For i < window, uses partial sum from [0..=i].
#[inline]
fn rolling_sum(arr: &[f64], window: usize) -> Vec<f64> {
    let n = arr.len();
    let mut cs = vec![0.0f64; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + arr[i];
    }
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        result[i] = cs[i + 1] - cs[start];
    }
    result
}

/// Causal rolling mean. For i < window, uses partial mean.
#[inline]
fn rolling_mean(arr: &[f64], window: usize) -> Vec<f64> {
    let n = arr.len();
    let mut cs = vec![0.0f64; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + arr[i];
    }
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let count = (i + 1 - start) as f64;
        result[i] = (cs[i + 1] - cs[start]) / count;
    }
    result
}

/// Causal rolling std. Uses Welford-style (mean of squares - square of means).
#[inline]
fn rolling_std(arr: &[f64], window: usize) -> Vec<f64> {
    let n = arr.len();
    let mut cs = vec![0.0f64; n + 1];
    let mut cs2 = vec![0.0f64; n + 1];
    for i in 0..n {
        cs[i + 1] = cs[i] + arr[i];
        cs2[i + 1] = cs2[i] + arr[i] * arr[i];
    }
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let count = (i + 1 - start) as f64;
        let mean = (cs[i + 1] - cs[start]) / count;
        let mean2 = (cs2[i + 1] - cs2[start]) / count;
        let var = (mean2 - mean * mean).max(0.0);
        result[i] = var.sqrt();
    }
    result
}

/// Causal rolling max using a monotonic deque for O(n) total.
fn rolling_max(arr: &[f64], window: usize) -> Vec<f64> {
    let n = arr.len();
    let mut result = vec![0.0f64; n];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    for i in 0..n {
        // Remove indices outside window
        while !deque.is_empty() && deque.front().unwrap() + window <= i {
            deque.pop_front();
        }
        // Maintain decreasing order
        while !deque.is_empty() && arr[*deque.back().unwrap()] <= arr[i] {
            deque.pop_back();
        }
        deque.push_back(i);
        result[i] = arr[*deque.front().unwrap()];
    }
    result
}

/// Shift array right by n (fill with first value). Equivalent to Python's _shift(arr, n).
#[inline]
fn shift(arr: &[f64], n: usize) -> Vec<f64> {
    let len = arr.len();
    if n == 0 {
        return arr.to_vec();
    }
    let mut result = vec![arr[0]; len];
    if n < len {
        result[n..].copy_from_slice(&arr[..len - n]);
    }
    result
}

/// Causal rolling Pearson correlation.
fn rolling_corr(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let xy: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).collect();
    let mx = rolling_mean(x, window);
    let my = rolling_mean(y, window);
    let mxy = rolling_mean(&xy, window);
    let sx = rolling_std(x, window);
    let sy = rolling_std(y, window);

    (0..n).map(|i| {
        let denom = sx[i] * sy[i];
        if denom > 1e-10 {
            ((mxy[i] - mx[i] * my[i]) / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }).collect()
}

/// Causal rolling OLS slope: y = a + b*x, returns b.
fn rolling_regression_slope(y: &[f64], x: &[f64], window: usize) -> Vec<f64> {
    let n = y.len();
    let xy: Vec<f64> = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).collect();
    let x2: Vec<f64> = x.iter().map(|&a| a * a).collect();
    let mx = rolling_mean(x, window);
    let my = rolling_mean(y, window);
    let mxy = rolling_mean(&xy, window);
    let mx2 = rolling_mean(&x2, window);

    (0..n).map(|i| {
        let denom = mx2[i] - mx[i] * mx[i];
        if denom.abs() > 1e-10 {
            ((mxy[i] - mx[i] * my[i]) / denom).clamp(-1e4, 1e4)
        } else {
            0.0
        }
    }).collect()
}

/// NumPy-compatible sign function: returns -1.0, 0.0, or +1.0.
/// Rust's f64::signum() returns +1.0 for +0.0 and -1.0 for -0.0 (IEEE 754),
/// but Python's np.sign(0.0) = 0.0. Use this to match Python behavior.
#[inline(always)]
fn py_sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

/// Compute run-length of consecutive True (>0.5) values.
/// Returns 0 if current bar is False, count of consecutive True bars up to current otherwise.
fn run_length(mask: &[f64]) -> Vec<f64> {
    let n = mask.len();
    let mut result = vec![0.0f64; n];
    let mut current = 0.0;
    for i in 0..n {
        if mask[i] > 0.5 {
            current += 1.0;
        } else {
            current = 0.0;
        }
        result[i] = current;
    }
    result
}

/// Count bars since last event (event[i] > 0.5). Capped at 500.
fn bars_since(event: &[f64]) -> Vec<f64> {
    let n = event.len();
    let mut result = vec![500.0f64; n];
    let mut last_event: i64 = -500;
    for i in 0..n {
        if event[i] > 0.5 {
            last_event = i as i64;
        }
        result[i] = ((i as i64 - last_event).min(500)) as f64;
    }
    result
}

/// Exponential moving average: alpha * x + (1-alpha) * prev
fn ema(arr: &[f64], alpha: f64) -> Vec<f64> {
    let n = arr.len();
    let mut result = vec![0.0f64; n];
    if n == 0 { return result; }
    result[0] = arr[0];
    let decay = 1.0 - alpha;
    for i in 1..n {
        result[i] = alpha * arr[i] + decay * result[i - 1];
    }
    result
}

/// Causal rolling skewness — matches Python's _rolling_skew exactly.
/// Python formula: diff[i] = arr[i] - rolling_mean(arr, w)[i]
///                 m3[i] = rolling_mean(diff^3, w)[i]
///                 skew[i] = m3[i] / rolling_std(arr, w)[i]^3
/// Note: This uses per-bar rolling means for centering (approximation, not exact central moment).
fn rolling_skew(arr: &[f64], window: usize) -> Vec<f64> {
    let n = arr.len();
    let mean = rolling_mean(arr, window);
    let std = rolling_std(arr, window);
    // diff[i] = arr[i] - mean[i]  (where mean[i] is the rolling mean at bar i)
    let diff3: Vec<f64> = (0..n).map(|i| {
        let d = arr[i] - mean[i];
        d * d * d
    }).collect();
    let m3 = rolling_mean(&diff3, window);
    (0..n).map(|i| {
        let s = std[i].max(1e-10);
        (m3[i] / (s * s * s)).clamp(-10.0, 10.0)
    }).collect()
}

// ============================================================================
// Diff (prepend first value, matches np.diff with prepend)
// ============================================================================

#[inline]
fn diff_prepend(arr: &[f64]) -> Vec<f64> {
    let n = arr.len();
    let mut result = vec![0.0f64; n];
    if n == 0 { return result; }
    result[0] = 0.0; // diff of first element with itself
    for i in 1..n {
        result[i] = arr[i] - arr[i - 1];
    }
    result
}

// ============================================================================
// Column accessor helpers
// ============================================================================

#[inline]
fn gcol(global: &[f32], n: usize, col: usize) -> Vec<f64> {
    (0..n).map(|i| global[i * N_GLOBAL_FEATURES + col] as f64).collect()
}

#[inline]
fn ncol(nodes: &[f32], n: usize, node_idx: usize, feat: usize) -> Vec<f64> {
    (0..n).map(|i| nodes[i * NUM_NODES * NODE_FEATURES + node_idx * NODE_FEATURES + feat] as f64).collect()
}

// ============================================================================
// Main feature computation for a single day
// ============================================================================

pub fn compute_mbo_features(input: &DayInput) -> Vec<f32> {
    let n = input.n_bars;
    let n_global = N_GLOBAL_FEATURES;
    let global = &input.global_features;
    let nodes = &input.node_features;
    let mid_prices = &input.mid_prices;

    // Output: N × TOTAL_FEATURES, initialized to NaN
    let mut features = vec![f32::NAN; n * TOTAL_FEATURES];

    // Helper closure to set a column
    let set_col = |features: &mut Vec<f32>, col: usize, values: &[f64]| {
        for i in 0..n {
            let v = values[i] as f32;
            features[i * TOTAL_FEATURES + col] = if v.is_nan() { f32::NAN } else if v.is_infinite() { 0.0 } else { v.clamp(-1e6, 1e6) };
        }
    };

    // =========================================================================
    // A: Static snapshot features (cols 0..96) — direct copy from global_features_raw
    // =========================================================================
    for i in 0..n {
        for j in 0..n_global {
            features[i * TOTAL_FEATURES + j] = global[i * n_global + j];
        }
    }
    let mut col = n_global; // = 96

    // =========================================================================
    // Extract commonly used arrays
    // =========================================================================
    let buy_volumes = gcol(global, n, COL_BUY_VOL);
    let sell_volumes = gcol(global, n, COL_SELL_VOL);
    let add_counts = gcol(global, n, COL_ADD_COUNT);
    let cancel_counts = gcol(global, n, COL_CANCEL_COUNT);
    let trade_counts = gcol(global, n, COL_TRADE_COUNT);
    let total_bid_vols = gcol(global, n, COL_TOTAL_BID_VOL);
    let total_ask_vols = gcol(global, n, COL_TOTAL_ASK_VOL);
    let bid_pressures = gcol(global, n, COL_BID_PRESSURE);
    let ask_pressures = gcol(global, n, COL_ASK_PRESSURE);
    let spreads = gcol(global, n, COL_SPREAD);

    let modify_counts = gcol(global, n, COL_MODIFY_COUNT);
    let fleeting_ratios = gcol(global, n, COL_FLEETING_RATIO);
    let aggr_buy_counts = gcol(global, n, COL_AGGR_BUY_COUNT);
    let aggr_sell_counts = gcol(global, n, COL_AGGR_SELL_COUNT);
    let aggr_imbalances = gcol(global, n, COL_AGGR_IMBALANCE);
    let max_trade_sizes = gcol(global, n, COL_MAX_TRADE_SIZE);
    let cancel_bid_vols = gcol(global, n, COL_CANCEL_BID_VOL);
    let cancel_ask_vols = gcol(global, n, COL_CANCEL_ASK_VOL);
    let cancel_side_imbs = gcol(global, n, COL_CANCEL_SIDE_IMB);
    let tick_counts = gcol(global, n, COL_TICK_COUNT);
    let n_completed = gcol(global, n, COL_N_COMPLETED);
    let mean_lifetimes = gcol(global, n, COL_MEAN_LIFETIME);

    // Node features: bid L1..L10, ask L1..L10
    let bid_sizes_l1 = ncol(nodes, n, 0, NODE_SIZE_COL);
    let ask_sizes_l1 = ncol(nodes, n, DEPTH_LEVELS, NODE_SIZE_COL);
    let bid_sizes_l3 = ncol(nodes, n, 2, NODE_SIZE_COL);
    let ask_sizes_l3 = ncol(nodes, n, DEPTH_LEVELS + 2, NODE_SIZE_COL);
    let bid_sizes_l5 = ncol(nodes, n, 4, NODE_SIZE_COL);
    let ask_sizes_l5 = ncol(nodes, n, DEPTH_LEVELS + 4, NODE_SIZE_COL);

    // Pre-compute common derived arrays
    let delta_bid = diff_prepend(&bid_sizes_l1);
    let delta_ask = diff_prepend(&ask_sizes_l1);
    let ofi_raw: Vec<f64> = (0..n).map(|i| delta_bid[i] - delta_ask[i]).collect();

    let total_trade_vol: Vec<f64> = (0..n).map(|i| buy_volumes[i] + sell_volumes[i]).collect();
    let safe_ttv: Vec<f64> = (0..n).map(|i| total_trade_vol[i].max(1.0)).collect();
    let trade_imb_raw: Vec<f64> = (0..n).map(|i| (buy_volumes[i] - sell_volumes[i]) / safe_ttv[i]).collect();

    // Compute signed_vol in float32 arithmetic to match Python (avoids float32→float64 sign flips)
    let signed_vol: Vec<f64> = (0..n).map(|i| {
        let bv = global[i * N_GLOBAL_FEATURES + COL_BUY_VOL];    // f32
        let sv = global[i * N_GLOBAL_FEATURES + COL_SELL_VOL];   // f32
        (bv - sv) as f64  // subtract as f32, then upcast
    }).collect();

    let log_mid: Vec<f64> = mid_prices.iter().map(|&m| m.max(1.0).ln()).collect();
    let log_ret_1 = diff_prepend(&log_mid);
    let price_change = diff_prepend(mid_prices);

    // =========================================================================
    // B: Rolling order flow (15) — windows [5, 20, 50]
    // =========================================================================
    for &w in &[5usize, 20, 50] {
        // ofi_w
        let ofi_w = rolling_sum(&ofi_raw, w);
        // trade_imb_w
        let timb_w = rolling_mean(&trade_imb_raw, w);
        // cancel_trade_w: rolling_sum(cancel) / max(1, rolling_sum(trade))
        let tc_sum = rolling_sum(&trade_counts, w);
        let cc_sum = rolling_sum(&cancel_counts, w);
        let cancel_trade: Vec<f64> = (0..n).map(|i| {
            if tc_sum[i] > 0.0 { cc_sum[i] / tc_sum[i].max(1.0) } else { 0.0 }
        }).collect();
        // net_flow_w: rolling_sum(add - cancel)
        let net: Vec<f64> = (0..n).map(|i| add_counts[i] - cancel_counts[i]).collect();
        let net_flow = rolling_sum(&net, w);
        // event_int_w: rolling_mean(add + cancel + trade)
        let aci: Vec<f64> = (0..n).map(|i| add_counts[i] + cancel_counts[i] + trade_counts[i]).collect();
        let event_int = rolling_mean(&aci, w);

        let tmp = [&ofi_w, &timb_w, &cancel_trade, &net_flow, &event_int];
        for arr in &tmp {
            set_col(&mut features, col, arr);
            col += 1;
        }
    }

    // =========================================================================
    // C: VPIN (3) — windows [20, 50, 100]
    // =========================================================================
    let abs_imb: Vec<f64> = (0..n).map(|i| (buy_volumes[i] - sell_volumes[i]).abs() / safe_ttv[i]).collect();
    for &w in &[20usize, 50, 100] {
        let vpin = rolling_mean(&abs_imb, w);
        set_col(&mut features, col, &vpin);
        col += 1;
    }

    // =========================================================================
    // D: Price momentum (10) — windows [5, 10, 20, 50, 100]
    // =========================================================================
    for &w in &[5usize, 10, 20, 50, 100] {
        let shifted = shift(&log_mid, w);
        let ret: Vec<f64> = (0..n).map(|i| log_mid[i] - shifted[i]).collect();
        let ret_vel: Vec<f64> = ret.iter().map(|&r| r / w.max(1) as f64).collect();
        set_col(&mut features, col, &ret);
        col += 1;
        set_col(&mut features, col, &ret_vel);
        col += 1;
    }

    // =========================================================================
    // E: Realized volatility (6) — windows [10, 20, 50]
    // =========================================================================
    for &w in &[10usize, 20, 50] {
        let rvol = rolling_std(&log_ret_1, w);
        let vov = rolling_std(&rvol, w);
        set_col(&mut features, col, &rvol);
        col += 1;
        set_col(&mut features, col, &vov);
        col += 1;
    }

    // =========================================================================
    // F: Book shape dynamics (10)
    // =========================================================================

    // depth_ratio_l1, depth_ratio_l3, depth_ratio_l5
    let f_start = col; // remember for later z-score indexing
    for (bs, asz) in &[
        (&bid_sizes_l1, &ask_sizes_l1),
        (&bid_sizes_l3, &ask_sizes_l3),
        (&bid_sizes_l5, &ask_sizes_l5),
    ] {
        let dr: Vec<f64> = (0..n).map(|i| {
            let total = bs[i] + asz[i];
            if total > 0.0 { bs[i] / total.max(1.0) } else { 0.5 }
        }).collect();
        set_col(&mut features, col, &dr);
        col += 1;
    }

    // weighted_book_imb
    let total_p: Vec<f64> = (0..n).map(|i| bid_pressures[i] + ask_pressures[i]).collect();
    let wbi: Vec<f64> = (0..n).map(|i| {
        if total_p[i] > 0.0 { (bid_pressures[i] - ask_pressures[i]) / total_p[i].max(1.0) } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &wbi);
    col += 1;

    // spread_change
    let spread_chg = diff_prepend(&spreads);
    let spread_chg_ticks: Vec<f64> = spread_chg.iter().map(|&s| s / TICK_SIZE as f64).collect();
    set_col(&mut features, col, &spread_chg_ticks);
    col += 1;

    // spread_zscore
    let sp_mean = rolling_mean(&spreads, 50);
    let sp_std = rolling_std(&spreads, 50);
    let sp_z: Vec<f64> = (0..n).map(|i| {
        if sp_std[i] > 1e-8 { (spreads[i] - sp_mean[i]) / sp_std[i] } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &sp_z);
    col += 1;

    // mid_ret_1
    let mid_ret_1 = {
        let shifted = shift(&log_mid, 1);
        let r: Vec<f64> = (0..n).map(|i| log_mid[i] - shifted[i]).collect();
        r
    };
    set_col(&mut features, col, &mid_ret_1);
    col += 1;

    // mid_ret_5
    let mid_ret_5 = {
        let shifted = shift(&log_mid, 5);
        (0..n).map(|i| log_mid[i] - shifted[i]).collect::<Vec<_>>()
    };
    set_col(&mut features, col, &mid_ret_5);
    col += 1;

    // mid_accel (ret1 - shift(ret1, 1))
    let ret1_shifted = shift(&mid_ret_1, 1);
    let mid_accel: Vec<f64> = (0..n).map(|i| mid_ret_1[i] - ret1_shifted[i]).collect();
    set_col(&mut features, col, &mid_accel);
    col += 1;

    // book_refresh: (add + cancel) / max(1, book_total)
    let book_total: Vec<f64> = (0..n).map(|i| total_bid_vols[i] + total_ask_vols[i]).collect();
    let book_refresh: Vec<f64> = (0..n).map(|i| {
        if book_total[i] > 0.0 {
            (add_counts[i] + cancel_counts[i]) / book_total[i].max(1.0)
        } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &book_refresh);
    col += 1;

    // =========================================================================
    // G: Toxicity / adverse selection (5)
    // =========================================================================

    // kyle_lambda_20, kyle_lambda_50
    for &w in &[20usize, 50] {
        let slope = rolling_regression_slope(&price_change, &signed_vol, w);
        set_col(&mut features, col, &slope);
        col += 1;
    }

    // price_impact
    let atv_20 = rolling_mean(&total_trade_vol, 20);
    let apc_20 = rolling_mean(&price_change.iter().map(|&x| x.abs()).collect::<Vec<_>>(), 20);
    let price_impact: Vec<f64> = (0..n).map(|i| {
        if atv_20[i] > 0.0 { apc_20[i] / atv_20[i].max(1e-8) } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &price_impact);
    col += 1;

    // adverse_sel: rolling_corr(sign(signed_vol), shift(price_change, 1), 50)
    let trade_dir: Vec<f64> = signed_vol.iter().map(|&x| py_sign(x)).collect();
    let past_ret = shift(&price_change, 1);
    let adv_sel = rolling_corr(&trade_dir, &past_ret, 50);
    set_col(&mut features, col, &adv_sel);
    col += 1;

    // toxicity_score: vpin_20 * max(0, -spread_chg / tick_size)
    // vpin_20 is at col = n_global + 15 (first VPIN)
    let vpin_20_col = n_global + 15;
    let toxicity_score: Vec<f64> = (0..n).map(|i| {
        let vpin_val = features[i * TOTAL_FEATURES + vpin_20_col] as f64;
        let spread_drop = (-spread_chg[i] / TICK_SIZE as f64).max(0.0);
        vpin_val * spread_drop
    }).collect();
    set_col(&mut features, col, &toxicity_score);
    col += 1;

    // =========================================================================
    // H: Microstructure strategy features (10)
    // =========================================================================

    // iceberg_score: rolling_sum(trade_counts * no_move, 10)
    let no_move: Vec<f64> = price_change.iter().map(|&pc| {
        if pc.abs() < TICK_SIZE as f64 * 0.5 { 1.0 } else { 0.0 }
    }).collect();
    let iceberg_raw: Vec<f64> = (0..n).map(|i| trade_counts[i] * no_move[i]).collect();
    let iceberg_score = rolling_sum(&iceberg_raw, 10);
    set_col(&mut features, col, &iceberg_score);
    col += 1;

    // absorption_bid: rolling_sum(sell_vol * (price_change >= 0), 10)
    let bid_absorb: Vec<f64> = (0..n).map(|i| {
        sell_volumes[i] * if price_change[i] >= 0.0 { 1.0 } else { 0.0 }
    }).collect();
    let absorption_bid = rolling_sum(&bid_absorb, 10);
    set_col(&mut features, col, &absorption_bid);
    col += 1;

    // absorption_ask: rolling_sum(buy_vol * (price_change <= 0), 10)
    let ask_absorb: Vec<f64> = (0..n).map(|i| {
        buy_volumes[i] * if price_change[i] <= 0.0 { 1.0 } else { 0.0 }
    }).collect();
    let absorption_ask = rolling_sum(&ask_absorb, 10);
    set_col(&mut features, col, &absorption_ask);
    col += 1;

    // large_trade_rev
    let avg_tv = rolling_mean(&total_trade_vol, 50);
    let large_mask: Vec<f64> = (0..n).map(|i| {
        if total_trade_vol[i] > 2.0 * avg_tv[i].max(1.0) { 1.0 } else { 0.0 }
    }).collect();
    let lagged_large = shift(&large_mask, 5);
    let lagged_sign: Vec<f64> = shift(&signed_vol.iter().map(|&x| py_sign(x)).collect::<Vec<_>>(), 5);
    let ret_since = {
        let shifted5 = shift(&log_mid, 5);
        (0..n).map(|i| log_mid[i] - shifted5[i]).collect::<Vec<_>>()
    };
    let large_trade_rev: Vec<f64> = (0..n).map(|i| {
        if lagged_large[i] > 0.5 { -ret_since[i] * lagged_sign[i] } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &large_trade_rev);
    col += 1;

    // spoof_score
    let add_5 = rolling_sum(&add_counts, 5);
    let cancel_5 = rolling_sum(&cancel_counts, 5);
    let spoof_score: Vec<f64> = (0..n).map(|i| {
        if add_5[i] > 0.0 {
            (cancel_5[i] / add_5[i].max(1.0)) * if cancel_5[i] > add_5[i] { 1.0 } else { 0.0 }
        } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &spoof_score);
    col += 1;

    // aggressive_burst: rolling_max(|signed_vol|, 5)
    let abs_signed_vol: Vec<f64> = signed_vol.iter().map(|&x| x.abs()).collect();
    let aggressive_burst = rolling_max(&abs_signed_vol, 5);
    set_col(&mut features, col, &aggressive_burst);
    col += 1;

    // sweep_count: where trade_count > 0, |price_change| / tick_size
    let sweep_count: Vec<f64> = (0..n).map(|i| {
        if trade_counts[i] > 0.0 { price_change[i].abs() / TICK_SIZE as f64 } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &sweep_count);
    col += 1;

    // momentum_ignition: trade_burst_3 * |mom_3|
    let trade_burst3 = rolling_sum(&trade_counts, 3);
    let mom_3 = {
        let shifted3 = shift(&log_mid, 3);
        (0..n).map(|i| (log_mid[i] - shifted3[i]).abs()).collect::<Vec<_>>()
    };
    let momentum_ignition: Vec<f64> = (0..n).map(|i| trade_burst3[i] * mom_3[i]).collect();
    set_col(&mut features, col, &momentum_ignition);
    col += 1;

    // book_flip: rolling_mean(|diff(sign(bid_vol - ask_vol))|, 20)
    let imb_sign: Vec<f64> = (0..n).map(|i| py_sign(total_bid_vols[i] - total_ask_vols[i])).collect();
    let sign_chg = diff_prepend(&imb_sign);
    let sign_chg_abs: Vec<f64> = sign_chg.iter().map(|&x| x.abs()).collect();
    let book_flip = rolling_mean(&sign_chg_abs, 20);
    set_col(&mut features, col, &book_flip);
    col += 1;

    // hidden_liq: total_trade_vol / max(1, bid_l1 + ask_l1)
    let visible: Vec<f64> = (0..n).map(|i| bid_sizes_l1[i] + ask_sizes_l1[i]).collect();
    let hidden_liq: Vec<f64> = (0..n).map(|i| {
        if visible[i] > 0.0 { total_trade_vol[i] / visible[i].max(1.0) } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &hidden_liq);
    col += 1;

    // =========================================================================
    // I: MBO-Enhanced rolling features (20) — windows [5, 20]
    // =========================================================================
    let i_start = col; // remember cancel_asym_5 position

    for &w in &[5usize, 20] {
        // modify_rate_w: rolling_sum(modify) / max(1, rolling_sum(add))
        let mod_sum = rolling_sum(&modify_counts, w);
        let add_sum = rolling_sum(&add_counts, w);
        let modify_rate: Vec<f64> = (0..n).map(|i| {
            if add_sum[i] > 0.0 { mod_sum[i] / add_sum[i].max(1.0) } else { 0.0 }
        }).collect();
        set_col(&mut features, col, &modify_rate);
        col += 1;

        // fleeting_ratio_w
        let fleeting = rolling_mean(&fleeting_ratios, w);
        set_col(&mut features, col, &fleeting);
        col += 1;

        // aggr_imb_w
        let aggr_imb = rolling_mean(&aggr_imbalances, w);
        set_col(&mut features, col, &aggr_imb);
        col += 1;

        // cancel_asym_w: rolling_mean(cancel_side_imb)
        let cancel_asym = rolling_mean(&cancel_side_imbs, w);
        set_col(&mut features, col, &cancel_asym);
        col += 1;

        // lifetime_mean_w
        let lifetime = rolling_mean(&mean_lifetimes, w);
        set_col(&mut features, col, &lifetime);
        col += 1;

        // max_trade_w: rolling_max
        let max_trade = rolling_max(&max_trade_sizes, w);
        set_col(&mut features, col, &max_trade);
        col += 1;

        // tick_density_w
        let tick_dens = rolling_mean(&tick_counts, w);
        set_col(&mut features, col, &tick_dens);
        col += 1;

        // orders_completed_w
        let orders_comp = rolling_mean(&n_completed, w);
        set_col(&mut features, col, &orders_comp);
        col += 1;

        // aggr_buy_rate_w: rolling_mean(aggr_buy / max(1, aggr_buy + aggr_sell))
        let total_aggr: Vec<f64> = (0..n).map(|i| (aggr_buy_counts[i] + aggr_sell_counts[i]).max(1.0)).collect();
        let aggr_buy_frac: Vec<f64> = (0..n).map(|i| aggr_buy_counts[i] / total_aggr[i]).collect();
        let aggr_buy_rate = rolling_mean(&aggr_buy_frac, w);
        set_col(&mut features, col, &aggr_buy_rate);
        col += 1;

        // cancel_bid_rate_w: rolling_mean(cancel_bid / max(1, cancel_bid + cancel_ask))
        let total_cancel: Vec<f64> = (0..n).map(|i| (cancel_bid_vols[i] + cancel_ask_vols[i]).max(1.0)).collect();
        let cancel_bid_frac: Vec<f64> = (0..n).map(|i| cancel_bid_vols[i] / total_cancel[i]).collect();
        let cancel_bid_rate = rolling_mean(&cancel_bid_frac, w);
        set_col(&mut features, col, &cancel_bid_rate);
        col += 1;
    }
    // i_start + 3 = cancel_asym_5 (4th feature in first window of I)

    // =========================================================================
    // J: Spatial per-level features (20)
    // =========================================================================
    // bid levels: L1..L5 order_count and concentration
    for lvl in 0..5usize {
        let oc = ncol(nodes, n, lvl, NODE_ORDER_COUNT_COL);
        let conc = ncol(nodes, n, lvl, NODE_CONCENTRATION_COL);
        set_col(&mut features, col, &oc);
        col += 1;
        set_col(&mut features, col, &conc);
        col += 1;
    }
    // ask levels: L1..L5
    for lvl in 0..5usize {
        let oc = ncol(nodes, n, DEPTH_LEVELS + lvl, NODE_ORDER_COUNT_COL);
        let conc = ncol(nodes, n, DEPTH_LEVELS + lvl, NODE_CONCENTRATION_COL);
        set_col(&mut features, col, &oc);
        col += 1;
        set_col(&mut features, col, &conc);
        col += 1;
    }

    // =========================================================================
    // K: Vol-Direction interaction features (5)
    // =========================================================================
    let rvol_5 = rolling_std(&log_ret_1, 5);
    let rvol_20 = rolling_std(&log_ret_1, 20);
    let rvol_100 = rolling_std(&log_ret_1, 100);

    // vol_regime: rvol_20 / rvol_100
    let k_start = col;
    let vol_regime: Vec<f64> = (0..n).map(|i| {
        if rvol_100[i] > 1e-10 { rvol_20[i] / rvol_100[i] } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &vol_regime);
    col += 1;

    // vol_accel: rvol_5 - shift(rvol_5, 5)
    let rvol5_shift5 = shift(&rvol_5, 5);
    let vol_accel: Vec<f64> = (0..n).map(|i| rvol_5[i] - rvol5_shift5[i]).collect();
    set_col(&mut features, col, &vol_accel);
    col += 1;

    // flow_during_vol: rolling_mean(trade_imb_raw, 5) * vol_ratio
    let flow_imb_5 = rolling_mean(&trade_imb_raw, 5);
    let flow_during_vol: Vec<f64> = (0..n).map(|i| flow_imb_5[i] * vol_regime[i]).collect();
    set_col(&mut features, col, &flow_during_vol);
    col += 1;

    // cancel_asym_vol: rolling_mean(cancel_side_imb, 5) * vol_ratio
    let casym_5 = rolling_mean(&cancel_side_imbs, 5);
    let cancel_asym_vol: Vec<f64> = (0..n).map(|i| casym_5[i] * vol_regime[i]).collect();
    set_col(&mut features, col, &cancel_asym_vol);
    col += 1;

    // aggr_momentum: rolling_mean(aggr_imbalances, 5)
    let aggr_momentum = rolling_mean(&aggr_imbalances, 5);
    set_col(&mut features, col, &aggr_momentum);
    col += 1;

    // =========================================================================
    // M: Rolling book shape dynamics (10)
    // =========================================================================
    let bid_entropy = gcol(global, n, COL_BID_SIZE_ENTROPY);
    let ask_entropy = gcol(global, n, COL_ASK_SIZE_ENTROPY);
    let total_entropy: Vec<f64> = (0..n).map(|i| bid_entropy[i] + ask_entropy[i]).collect();
    let bid_cog = gcol(global, n, COL_BID_DEPTH_COG);
    let ask_cog = gcol(global, n, COL_ASK_DEPTH_COG);
    let book_sym = gcol(global, n, COL_BOOK_SYMMETRY);
    let wall_bid = gcol(global, n, COL_MAX_BID_WALL_FRAC);
    let wall_ask = gcol(global, n, COL_MAX_ASK_WALL_FRAC);
    let bid_gaps = gcol(global, n, COL_BID_GAP_COUNT);
    let ask_gaps = gcol(global, n, COL_ASK_GAP_COUNT);
    let total_depth = gcol(global, n, COL_TOTAL_DEPTH_LOG);
    let l1_l3_bid = gcol(global, n, COL_L1_L3_BID_RATIO);
    let l1_l3_ask = gcol(global, n, COL_L1_L3_ASK_RATIO);

    // entropy_change_5: total_entropy - shift(total_entropy, 5)
    let ent_shift5 = shift(&total_entropy, 5);
    let entropy_change_5: Vec<f64> = (0..n).map(|i| total_entropy[i] - ent_shift5[i]).collect();
    set_col(&mut features, col, &entropy_change_5);
    col += 1;

    // entropy_change_20
    let ent_shift20 = shift(&total_entropy, 20);
    let entropy_change_20: Vec<f64> = (0..n).map(|i| total_entropy[i] - ent_shift20[i]).collect();
    set_col(&mut features, col, &entropy_change_20);
    col += 1;

    // cog_asym_5: rolling_mean(bid_cog - ask_cog, 5)
    let cog_asym: Vec<f64> = (0..n).map(|i| bid_cog[i] - ask_cog[i]).collect();
    let cog_asym_5 = rolling_mean(&cog_asym, 5);
    set_col(&mut features, col, &cog_asym_5);
    col += 1;

    // cog_asym_20
    let cog_asym_20 = rolling_mean(&cog_asym, 20);
    set_col(&mut features, col, &cog_asym_20);
    col += 1;

    // depth_accel: (depth - shift(depth, 1)) - shift(depth - shift(depth, 1), 1)
    let depth_shift1 = shift(&total_depth, 1);
    let depth_vel: Vec<f64> = (0..n).map(|i| total_depth[i] - depth_shift1[i]).collect();
    let depth_vel_shift1 = shift(&depth_vel, 1);
    let depth_accel: Vec<f64> = (0..n).map(|i| depth_vel[i] - depth_vel_shift1[i]).collect();
    set_col(&mut features, col, &depth_accel);
    col += 1;

    // symmetry_mean_20
    let sym_mean_20 = rolling_mean(&book_sym, 20);
    set_col(&mut features, col, &sym_mean_20);
    col += 1;

    // wall_bid_persist_20
    let wall_bid_persist = rolling_mean(&wall_bid, 20);
    set_col(&mut features, col, &wall_bid_persist);
    col += 1;

    // wall_ask_persist_20
    let wall_ask_persist = rolling_mean(&wall_ask, 20);
    set_col(&mut features, col, &wall_ask_persist);
    col += 1;

    // gap_trend_5: (bid_gaps + ask_gaps) - shift(bid_gaps + ask_gaps, 5)
    let total_gaps: Vec<f64> = (0..n).map(|i| bid_gaps[i] + ask_gaps[i]).collect();
    let gaps_shift5 = shift(&total_gaps, 5);
    let gap_trend_5: Vec<f64> = (0..n).map(|i| total_gaps[i] - gaps_shift5[i]).collect();
    set_col(&mut features, col, &gap_trend_5);
    col += 1;

    // l1_ratio_shift_5: (l1_l3_bid - l1_l3_ask) - shift(l1_l3_bid - l1_l3_ask, 5)
    let l1_asym: Vec<f64> = (0..n).map(|i| l1_l3_bid[i] - l1_l3_ask[i]).collect();
    let l1_asym_shift5 = shift(&l1_asym, 5);
    let l1_ratio_shift: Vec<f64> = (0..n).map(|i| l1_asym[i] - l1_asym_shift5[i]).collect();
    set_col(&mut features, col, &l1_ratio_shift);
    col += 1;

    // =========================================================================
    // N: Rolling quote dynamics (10)
    // =========================================================================
    let l1_bid_chg = gcol(global, n, COL_L1_BID_CHANGE_RATE);
    let l1_ask_chg = gcol(global, n, COL_L1_ASK_CHANGE_RATE);
    let depletion = gcol(global, n, COL_L1_DEPLETION_LOG);
    let depletion_asym = gcol(global, n, COL_DEPLETION_ASYMMETRY);
    let trade_bid_frac = gcol(global, n, COL_TRADE_AT_BID_FRAC);
    let trade_cv = gcol(global, n, COL_TRADE_SIZE_CV);
    let repos_rate = gcol(global, n, COL_REPOSITION_RATE);
    let add_size = gcol(global, n, COL_MEAN_ADD_SIZE_LOG);
    let max_add = gcol(global, n, COL_MAX_ADD_SIZE_NORM);

    // l1_instability_5, l1_instability_20
    let total_chg_rate: Vec<f64> = (0..n).map(|i| l1_bid_chg[i] + l1_ask_chg[i]).collect();
    let l1_inst5 = rolling_mean(&total_chg_rate, 5);
    set_col(&mut features, col, &l1_inst5);
    col += 1;
    let l1_inst20 = rolling_mean(&total_chg_rate, 20);
    set_col(&mut features, col, &l1_inst20);
    col += 1;

    // depletion_momentum_5
    let depl_sum5 = rolling_sum(&depletion, 5);
    set_col(&mut features, col, &depl_sum5);
    col += 1;

    // depletion_dir_20
    let depl_dir20 = rolling_mean(&depletion_asym, 20);
    set_col(&mut features, col, &depl_dir20);
    col += 1;

    // sell_pressure_5, sell_pressure_20
    let sell_press5 = rolling_mean(&trade_bid_frac, 5);
    set_col(&mut features, col, &sell_press5);
    col += 1;
    let sell_press20 = rolling_mean(&trade_bid_frac, 20);
    set_col(&mut features, col, &sell_press20);
    col += 1;

    // size_dispersion_20
    let size_disp20 = rolling_mean(&trade_cv, 20);
    set_col(&mut features, col, &size_disp20);
    col += 1;

    // reposition_intensity_5
    let repos_int5 = rolling_mean(&repos_rate, 5);
    set_col(&mut features, col, &repos_int5);
    col += 1;

    // add_size_trend_20
    let add_size_trend20 = rolling_mean(&add_size, 20);
    set_col(&mut features, col, &add_size_trend20);
    col += 1;

    // institutional_flow_5
    let inst_flow5 = rolling_mean(&max_add, 5);
    set_col(&mut features, col, &inst_flow5);
    col += 1;

    // =========================================================================
    // O: Cross-timeframe z-scores (30) — 10 features × 3 (z50, z500, diverge)
    // =========================================================================

    // Source arrays (extracted from already-computed columns or raw)
    // ofi_5 is at col n_global + 0
    let ofi_5_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + n_global + 0] as f64).collect();
    // trade_imb_5 is at col n_global + 1
    let trade_imb_5_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + n_global + 1] as f64).collect();
    // aggressive_imbalance: direct from raw
    let aggr_imb_arr = aggr_imbalances.clone();
    // depth_ratio_l1: at f_start
    let dr_l1_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + f_start] as f64).collect();
    // weighted_book_imb: at f_start + 3
    let wbi_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + f_start + 3] as f64).collect();
    // vpin_20: at n_global + 15
    let vpin_20_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + vpin_20_col] as f64).collect();
    // cancel_asym_5: at i_start + 3 (4th feature in first I window)
    let cancel_asym_5_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + i_start + 3] as f64).collect();
    // rvol_10: at n_global + 15 + 3 + 10 = n_global + 28 (after B:15, C:3, D:10)
    let e_start = n_global + 15 + 3 + 10;
    let rvol_10_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + e_start] as f64).collect();
    // event_density: direct from raw
    let event_density_arr = gcol(global, n, COL_EVENT_DENSITY);
    // bid_size_entropy: direct from raw
    let bid_entropy_arr = bid_entropy.clone();

    let zscore_sources = [
        &ofi_5_arr,
        &trade_imb_5_arr,
        &aggr_imb_arr,
        &dr_l1_arr,
        &wbi_arr,
        &vpin_20_arr,
        &cancel_asym_5_arr,
        &rvol_10_arr,
        &event_density_arr,
        &bid_entropy_arr,
    ];

    for src in &zscore_sources {
        // Short-term z-score (50 bars)
        // Python computes z in float32 (arr, mean, std all float32), then clips to [-5,5].
        // IMPORTANT: divergence uses the UNCLIPPED z values (Python local vars, not features array).
        let mean_50 = rolling_mean(src, 50);
        let std_50 = rolling_std(src, 50);
        // Compute unclipped z (in float32 to match Python)
        let z_50_raw: Vec<f64> = (0..n).map(|i| {
            if std_50[i] > 1e-8 {
                ((src[i] as f32 - mean_50[i] as f32) / std_50[i] as f32) as f64
            } else { 0.0 }
        }).collect();
        // Store clipped version
        let z_50_clipped: Vec<f64> = z_50_raw.iter().map(|&z| z.clamp(-5.0, 5.0)).collect();
        set_col(&mut features, col, &z_50_clipped);
        col += 1;

        // Long-term z-score (500 bars)
        let mean_500 = rolling_mean(src, 500);
        let std_500 = rolling_std(src, 500);
        let z_500_raw: Vec<f64> = (0..n).map(|i| {
            if std_500[i] > 1e-8 {
                ((src[i] as f32 - mean_500[i] as f32) / std_500[i] as f32) as f64
            } else { 0.0 }
        }).collect();
        let z_500_clipped: Vec<f64> = z_500_raw.iter().map(|&z| z.clamp(-5.0, 5.0)).collect();
        set_col(&mut features, col, &z_500_clipped);
        col += 1;

        // Divergence: uses UNCLIPPED z values (matching Python behavior where z_50/z_500
        // are local variables computed before clipping, not read back from features array)
        let diverge: Vec<f64> = (0..n).map(|i| (z_50_raw[i] - z_500_raw[i]).clamp(-10.0, 10.0)).collect();
        set_col(&mut features, col, &diverge);
        col += 1;
    }

    // =========================================================================
    // P: Magnitude predictors (10)
    // =========================================================================

    // book_thinning: l1_size / rolling_mean(l1_size, 100)
    let l1_size: Vec<f64> = (0..n).map(|i| bid_sizes_l1[i] + ask_sizes_l1[i]).collect();
    let l1_mean_100 = rolling_mean(&l1_size, 100);
    let book_thinning: Vec<f64> = (0..n).map(|i| {
        if l1_mean_100[i] > 1e-3 { l1_size[i] / l1_mean_100[i] } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &book_thinning);
    col += 1;

    // stacked_depth_asym: sum(bid L1-L3) / (sum(bid L1-L3) + sum(ask L1-L3))
    // Pre-extract the needed node columns to avoid O(N^2) allocations inside loop
    let bid_l0 = ncol(nodes, n, 0, NODE_SIZE_COL);
    let bid_l1_n = ncol(nodes, n, 1, NODE_SIZE_COL);
    let bid_l2 = ncol(nodes, n, 2, NODE_SIZE_COL);
    let ask_l0 = ncol(nodes, n, DEPTH_LEVELS + 0, NODE_SIZE_COL);
    let ask_l1_n = ncol(nodes, n, DEPTH_LEVELS + 1, NODE_SIZE_COL);
    let ask_l2 = ncol(nodes, n, DEPTH_LEVELS + 2, NODE_SIZE_COL);
    let bid_l1_l3: Vec<f64> = (0..n).map(|i| bid_l0[i] + bid_l1_n[i] + bid_l2[i]).collect();
    let ask_l1_l3: Vec<f64> = (0..n).map(|i| ask_l0[i] + ask_l1_n[i] + ask_l2[i]).collect();
    let stacked_asym: Vec<f64> = (0..n).map(|i| {
        let total = bid_l1_l3[i] + ask_l1_l3[i];
        if total > 0.0 { bid_l1_l3[i] / total.max(1.0) } else { 0.5 }
    }).collect();
    set_col(&mut features, col, &stacked_asym);
    col += 1;

    // flow_acceleration: d(OFI_5)/dt acceleration
    let ofi5_shift3 = shift(&ofi_5_arr, 3);
    let ofi_vel: Vec<f64> = (0..n).map(|i| ofi_5_arr[i] - ofi5_shift3[i]).collect();
    let ofi_vel_shift3 = shift(&ofi_vel, 3);
    let flow_accel: Vec<f64> = (0..n).map(|i| ofi_vel[i] - ofi_vel_shift3[i]).collect();
    set_col(&mut features, col, &flow_accel);
    col += 1;

    // spread_expansion: spreads / rolling_mean(spreads, 100)
    let spread_mean_100 = rolling_mean(&spreads, 100);
    let spread_expansion: Vec<f64> = (0..n).map(|i| {
        if spread_mean_100[i] > 1e-6 { spreads[i] / spread_mean_100[i] } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &spread_expansion);
    col += 1;

    // cross_flow_agreement: sign(ofi_5) * sign(ofi_50)
    let ofi_50_arr: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + n_global + 10] as f64).collect();
    let cross_flow: Vec<f64> = (0..n).map(|i| py_sign(ofi_5_arr[i]) * py_sign(ofi_50_arr[i])).collect();
    set_col(&mut features, col, &cross_flow);
    col += 1;

    // toxicity_spike: vpin_20 / rolling_mean(vpin_20, 100)
    let vpin_mean_100 = rolling_mean(&vpin_20_arr, 100);
    let toxicity_spike: Vec<f64> = (0..n).map(|i| {
        if vpin_mean_100[i] > 1e-6 { vpin_20_arr[i] / vpin_mean_100[i] } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &toxicity_spike);
    col += 1;

    // trade_rate_spike: trade_count / rolling_mean(trade_count, 100)
    let tc_mean_100 = rolling_mean(&trade_counts, 100);
    let trade_rate_spike: Vec<f64> = (0..n).map(|i| {
        if tc_mean_100[i] > 0.1 { trade_counts[i] / tc_mean_100[i].max(0.1) } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &trade_rate_spike);
    col += 1;

    // depletion_rate_5: rolling_sum(depletion, 5)
    let depl_rate5 = rolling_sum(&depletion, 5);
    set_col(&mut features, col, &depl_rate5);
    col += 1;

    // institutional_signal: large_trade_ratio * aggr_imbalance
    let lt_ratio = gcol(global, n, COL_LARGE_TRADE_RATIO);
    let inst_signal: Vec<f64> = (0..n).map(|i| lt_ratio[i] * aggr_imbalances[i]).collect();
    set_col(&mut features, col, &inst_signal);
    col += 1;

    // volatility_compression: rvol_5 / rvol_100
    let vol_compression: Vec<f64> = (0..n).map(|i| {
        if rvol_100[i] > 1e-10 { rvol_5[i] / rvol_100[i] } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &vol_compression);
    col += 1;

    // =========================================================================
    // Q: Order flow sequences (10)
    // =========================================================================

    let flow_sign: Vec<f64> = signed_vol.iter().map(|&x| py_sign(x)).collect();
    let buy_dom: Vec<f64> = flow_sign.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
    let sell_dom: Vec<f64> = flow_sign.iter().map(|&x| if x < 0.0 { 1.0 } else { 0.0 }).collect();

    let buy_run_len = run_length(&buy_dom);
    let sell_run_len = run_length(&sell_dom);

    // buy_run_max_5, sell_run_max_5
    let buy_run_max = rolling_max(&buy_run_len, 5);
    let sell_run_max = rolling_max(&sell_run_len, 5);
    set_col(&mut features, col, &buy_run_max);
    col += 1;
    set_col(&mut features, col, &sell_run_max);
    col += 1;

    // run_direction_20: rolling_sum(flow_sign, 20)
    let run_dir_20 = rolling_sum(&flow_sign, 20);
    set_col(&mut features, col, &run_dir_20);
    col += 1;

    // flow_reversal_5: sign changes in 5 bars
    let flow_diff = diff_prepend(&flow_sign);
    let sign_changes: Vec<f64> = flow_diff.iter().map(|&x| if x.abs() > 0.5 { 1.0 } else { 0.0 }).collect();
    let flow_reversal_5 = rolling_sum(&sign_changes, 5);
    set_col(&mut features, col, &flow_reversal_5);
    col += 1;

    // flow_persistence_20: rolling_corr(signed_vol, shift(signed_vol, 1), 20)
    let signed_vol_shift1 = shift(&signed_vol, 1);
    let flow_persistence_20 = rolling_corr(&signed_vol, &signed_vol_shift1, 20);
    let fp20_col = col;
    set_col(&mut features, col, &flow_persistence_20);
    col += 1;

    // aggressive_streak_5: max consecutive aggressive bars in 5
    let aggr_sign: Vec<f64> = (0..n).map(|i| py_sign(aggr_buy_counts[i] - aggr_sell_counts[i])).collect();
    let aggr_dom: Vec<f64> = aggr_sign.iter().map(|&x| if x != 0.0 { 1.0 } else { 0.0 }).collect();
    let aggr_run_len = run_length(&aggr_dom);
    let aggr_streak5 = rolling_max(&aggr_run_len, 5);
    set_col(&mut features, col, &aggr_streak5);
    col += 1;

    // buy_intensity_ratio, sell_intensity_ratio
    let buy_5 = rolling_sum(&buy_volumes, 5);
    let buy_50 = rolling_sum(&buy_volumes, 50);
    let sell_5 = rolling_sum(&sell_volumes, 5);
    let sell_50 = rolling_sum(&sell_volumes, 50);
    let buy_intensity: Vec<f64> = (0..n).map(|i| {
        if buy_50[i] > 0.1 { buy_5[i] / buy_50[i].max(0.1) * 10.0 } else { 1.0 }
    }).collect();
    let sell_intensity: Vec<f64> = (0..n).map(|i| {
        if sell_50[i] > 0.1 { sell_5[i] / sell_50[i].max(0.1) * 10.0 } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &buy_intensity);
    col += 1;
    set_col(&mut features, col, &sell_intensity);
    col += 1;

    // momentum_exhaustion: price_accel * clip(-flow_persistence, -1, 1)
    let log_ret_1_shift1 = shift(&log_ret_1, 1);
    let price_accel: Vec<f64> = (0..n).map(|i| log_ret_1[i] - log_ret_1_shift1[i]).collect();
    let momentum_exhaustion: Vec<f64> = (0..n).map(|i| {
        price_accel[i] * (-flow_persistence_20[i]).clamp(-1.0, 1.0)
    }).collect();
    set_col(&mut features, col, &momentum_exhaustion);
    col += 1;

    // flow_regime_zscore
    // flow_reversal_5 is at (col - 6) relative position
    // The raw flow_reversal_5 is at fp20_col - 1 (one before flow_persistence_20)
    let reversal_mean = rolling_mean(&sign_changes, 100);
    let reversal_std = rolling_std(&sign_changes, 100);
    let flow_reversal_5_raw = rolling_sum(&sign_changes, 5);
    let flow_regime_z: Vec<f64> = (0..n).map(|i| {
        if reversal_std[i] > 1e-6 {
            ((flow_reversal_5_raw[i] - reversal_mean[i] * 5.0) /
             (reversal_std[i] * 5.0f64.sqrt()).max(1e-6)).clamp(-5.0, 5.0)
        } else { 0.0 }
    }).collect();
    set_col(&mut features, col, &flow_regime_z);
    col += 1;

    // =========================================================================
    // R: Depth-weighted OFI + Staleness (10)
    // =========================================================================

    // dwfi_5, dwfi_20: depth-weighted flow imbalance over levels 0..5
    let mut dwfi_raw = vec![0.0f64; n];
    for lvl in 0..5usize {
        let bid_lvl = ncol(nodes, n, lvl, NODE_SIZE_COL);
        let ask_lvl = ncol(nodes, n, DEPTH_LEVELS + lvl, NODE_SIZE_COL);
        let d_bid = diff_prepend(&bid_lvl);
        let d_ask = diff_prepend(&ask_lvl);
        let weight = 1.0 / (lvl + 1) as f64;
        for i in 0..n {
            dwfi_raw[i] += (d_bid[i] - d_ask[i]) * weight;
        }
    }
    let dwfi_5 = rolling_sum(&dwfi_raw, 5);
    set_col(&mut features, col, &dwfi_5);
    col += 1;
    let dwfi_20 = rolling_sum(&dwfi_raw, 20);
    set_col(&mut features, col, &dwfi_20);
    col += 1;

    // ofi_l3_5: OFI at levels 0-2 combined, rolling 5 bars
    let mut ofi_l3_raw = vec![0.0f64; n];
    for lvl in 0..3usize {
        let bid_lvl = ncol(nodes, n, lvl, NODE_SIZE_COL);
        let ask_lvl = ncol(nodes, n, DEPTH_LEVELS + lvl, NODE_SIZE_COL);
        let d_bid = diff_prepend(&bid_lvl);
        let d_ask = diff_prepend(&ask_lvl);
        for i in 0..n {
            ofi_l3_raw[i] += d_bid[i] - d_ask[i];
        }
    }
    let ofi_l3_5 = rolling_sum(&ofi_l3_raw, 5);
    set_col(&mut features, col, &ofi_l3_5);
    col += 1;

    // ofi_deep_5: OFI at levels 3-9, rolling 5 bars
    let mut ofi_deep_raw = vec![0.0f64; n];
    for lvl in 3..DEPTH_LEVELS {
        let bid_lvl = ncol(nodes, n, lvl, NODE_SIZE_COL);
        let ask_lvl = ncol(nodes, n, DEPTH_LEVELS + lvl, NODE_SIZE_COL);
        let d_bid = diff_prepend(&bid_lvl);
        let d_ask = diff_prepend(&ask_lvl);
        for i in 0..n {
            ofi_deep_raw[i] += d_bid[i] - d_ask[i];
        }
    }
    let ofi_deep_5 = rolling_sum(&ofi_deep_raw, 5);
    set_col(&mut features, col, &ofi_deep_5);
    let ofi_deep_5_col = col;
    col += 1;

    // ofi_deep_vs_l1: deep_ofi_5 / max(|l1_ofi_5|, 0.01)
    let ofi_deep_vs_l1: Vec<f64> = (0..n).map(|i| {
        let safe_l1 = ofi_5_arr[i].abs().max(0.01);
        (features[i * TOTAL_FEATURES + ofi_deep_5_col] as f64 / safe_l1).clamp(-10.0, 10.0)
    }).collect();
    set_col(&mut features, col, &ofi_deep_vs_l1);
    col += 1;

    // bars_since_trade
    let has_trade: Vec<f64> = trade_counts.iter().map(|&t| if t > 0.0 { 1.0 } else { 0.0 }).collect();
    let bars_since_trade = bars_since(&has_trade);
    set_col(&mut features, col, &bars_since_trade);
    col += 1;

    // bars_since_big_trade: trade_size > 2 * rolling_mean(trade_size, 50)
    // Use f32 arithmetic to match Python (max_trade_sizes is f32, numpy ops stay f32)
    let median_trade = rolling_mean(&max_trade_sizes, 50);
    let has_big: Vec<f64> = (0..n).map(|i| {
        // Cast back to f32 to match Python's float32 comparison
        let mts_f32 = global[i * N_GLOBAL_FEATURES + COL_MAX_TRADE_SIZE]; // raw f32
        let thresh_f32 = (2.0f32 * (median_trade[i] as f32).max(0.1f32));
        if mts_f32 > thresh_f32 { 1.0 } else { 0.0 }
    }).collect();
    let bars_since_big = bars_since(&has_big);
    set_col(&mut features, col, &bars_since_big);
    col += 1;

    // bars_since_l1_change: bars since mid price changed
    let mid_changed: Vec<f64> = (0..n).map(|i| {
        if price_change[i].abs() > 1e-6 { 1.0 } else { 0.0 }
    }).collect();
    let bars_since_l1 = bars_since(&mid_changed);
    set_col(&mut features, col, &bars_since_l1);
    col += 1;

    // price_level_duration: consecutive bars at same mid
    let same_price: Vec<f64> = (0..n).map(|i| {
        if price_change[i].abs() < 1e-6 { 1.0 } else { 0.0 }
    }).collect();
    let price_dur = run_length(&same_price);
    set_col(&mut features, col, &price_dur);
    col += 1;

    // activity_halflife: EMA(trade_counts, alpha=0.1)
    let activity_hl = ema(&trade_counts, 0.1);
    set_col(&mut features, col, &activity_hl);
    col += 1;

    // =========================================================================
    // S: Cross-signal interactions (10)
    // =========================================================================

    // return_autocorr_10, return_autocorr_50
    let log_ret_shift1 = shift(&log_ret_1, 1);
    let ret_autocorr_10 = rolling_corr(&log_ret_1, &log_ret_shift1, 10);
    set_col(&mut features, col, &ret_autocorr_10);
    col += 1;
    let ret_autocorr_50 = rolling_corr(&log_ret_1, &log_ret_shift1, 50);
    set_col(&mut features, col, &ret_autocorr_50);
    col += 1;

    // vol_price_corr_20: rolling_corr(total_trade_vol, |log_ret_1|, 20)
    let abs_ret: Vec<f64> = log_ret_1.iter().map(|&r| r.abs()).collect();
    let vol_price_corr = rolling_corr(&total_trade_vol, &abs_ret, 20);
    set_col(&mut features, col, &vol_price_corr);
    col += 1;

    // spread_vol_interaction: spread_expansion * vol_ratio
    let s_vol_ratio: Vec<f64> = (0..n).map(|i| {
        if rvol_100[i] > 1e-10 { rvol_20[i] / rvol_100[i] } else { 1.0 }
    }).collect();
    let spread_vol_int: Vec<f64> = (0..n).map(|i| spread_expansion[i] * s_vol_ratio[i]).collect();
    set_col(&mut features, col, &spread_vol_int);
    col += 1;

    // imb_spread_interaction: (depth_ratio_l1 - 0.5) * spread_zscore
    // spread_zscore is at f_start + 5 (6th in F)
    let s_sp_z: Vec<f64> = (0..n).map(|i| features[i * TOTAL_FEATURES + f_start + 5] as f64).collect();
    let imb_spread_int: Vec<f64> = (0..n).map(|i| (dr_l1_arr[i] - 0.5) * s_sp_z[i]).collect();
    set_col(&mut features, col, &imb_spread_int);
    col += 1;

    // flow_vol_interaction: ofi_z50 * rvol_z50 (clipped)
    let s_ofi_mean = rolling_mean(&ofi_5_arr, 50);
    let s_ofi_std = rolling_std(&ofi_5_arr, 50);
    let s_ofi_z: Vec<f64> = (0..n).map(|i| {
        if s_ofi_std[i] > 1e-8 { (ofi_5_arr[i] - s_ofi_mean[i]) / s_ofi_std[i] } else { 0.0 }
    }).collect();
    let s_rvol_mean = rolling_mean(&rvol_5, 50);
    let s_rvol_std = rolling_std(&rvol_5, 50);
    let s_rvol_z: Vec<f64> = (0..n).map(|i| {
        if s_rvol_std[i] > 1e-10 { (rvol_5[i] - s_rvol_mean[i]) / s_rvol_std[i] } else { 0.0 }
    }).collect();
    let flow_vol_int: Vec<f64> = (0..n).map(|i| (s_ofi_z[i] * s_rvol_z[i]).clamp(-25.0, 25.0)).collect();
    set_col(&mut features, col, &flow_vol_int);
    col += 1;

    // toxicity_imb_combo: (vpin_z - 1) * (dr_l1 - 0.5) * 4
    let s_vpin_mean = rolling_mean(&vpin_20_arr, 100);
    let s_vpin_z: Vec<f64> = (0..n).map(|i| {
        if s_vpin_mean[i] > 1e-6 { vpin_20_arr[i] / s_vpin_mean[i] } else { 1.0 }
    }).collect();
    let toxicity_imb: Vec<f64> = (0..n).map(|i| {
        ((s_vpin_z[i] - 1.0) * (dr_l1_arr[i] - 0.5) * 4.0)
    }).collect();
    set_col(&mut features, col, &toxicity_imb);
    col += 1;

    // hurst_proxy_20: var(ret_1, 20) * 5 / var(ret_5, 20)
    let var_ret_1_20: Vec<f64> = rolling_std(&log_ret_1, 20).iter().map(|&s| s * s).collect();
    let ret_5: Vec<f64> = {
        let shift5 = shift(&log_mid, 5);
        (0..n).map(|i| log_mid[i] - shift5[i]).collect()
    };
    let var_ret_5_20: Vec<f64> = rolling_std(&ret_5, 20).iter().map(|&s| s * s).collect();
    let hurst_proxy: Vec<f64> = (0..n).map(|i| {
        if var_ret_5_20[i] > 1e-16 {
            (var_ret_1_20[i] * 5.0 / var_ret_5_20[i]).clamp(0.0, 5.0)
        } else { 1.0 }
    }).collect();
    set_col(&mut features, col, &hurst_proxy);
    col += 1;

    // return_skew_20
    let ret_skew_20 = rolling_skew(&log_ret_1, 20);
    set_col(&mut features, col, &ret_skew_20);
    col += 1;

    // multi_signal_strength: count of key features > 2σ
    let z_sources_for_ms = [
        &ofi_5_arr,
        &aggr_imb_arr,
        &dr_l1_arr,
        &vpin_20_arr,
        &cancel_asym_5_arr,
        &rvol_10_arr,
    ];
    let mut n_signals = vec![0.0f64; n];
    for src in &z_sources_for_ms {
        let z_std = rolling_std(src, 50);
        let z_mean = rolling_mean(src, 50);
        for i in 0..n {
            let z_val = if z_std[i] > 1e-8 { (src[i] - z_mean[i]) / z_std[i] } else { 0.0 };
            if z_val.abs() > 2.0 {
                n_signals[i] += 1.0;
            }
        }
    }
    set_col(&mut features, col, &n_signals);
    col += 1;

    // =========================================================================
    // Verify we filled exactly TOTAL_FEATURES columns
    // =========================================================================
    debug_assert_eq!(col, TOTAL_FEATURES,
        "Feature count mismatch: filled {}, expected {}", col, TOTAL_FEATURES);

    // =========================================================================
    // Day-boundary masking: first MAX_ROLLING_WINDOW (100) bars get NaN
    // for all rolling features (cols 96..290).
    // This single-day implementation masks bars 0..100 of the day.
    // When processing across multiple days concatenated, the caller handles
    // multi-day boundary masking. For per-day processing, only bar 0 is
    // the day boundary, so we mask 0..min(100, n).
    // =========================================================================
    let warmup_end = MAX_ROLLING_WINDOW.min(n);
    for i in 0..warmup_end {
        for j in n_global..TOTAL_FEATURES {
            features[i * TOTAL_FEATURES + j] = f32::NAN;
        }
    }

    features
}

// ============================================================================
// File discovery and processing
// ============================================================================

fn list_cache_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("Cannot read dir: {}", dir.display()))
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let s = name.to_str().unwrap_or("");
            s.ends_with("_snapshots.npz")
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

fn process_one_day(input_path: &Path, output_path: &Path) -> Result<(usize, f64)> {
    let t0 = Instant::now();

    let input = load_npz(input_path)?;
    let n = input.n_bars;

    if n < 10 {
        anyhow::bail!("Too few bars: {}", n);
    }

    let features = compute_mbo_features(&input);

    // Validate feature count
    assert_eq!(features.len(), n * TOTAL_FEATURES,
        "Feature vector size mismatch: got {}, expected {}", features.len(), n * TOTAL_FEATURES);

    write_features_npz(output_path, &features, n)?;

    let elapsed = t0.elapsed().as_secs_f64();
    Ok((n, elapsed))
}

// ============================================================================
// Main entry point (called from feature_expander_main.rs or standalone)
// ============================================================================

pub fn run_expander(args: ExpanderArgs) -> Result<()> {
    println!("======================================================================");
    println!("MBO FEATURE EXPANDER (Rust) — 96 → {} features", TOTAL_FEATURES);
    println!("  Cache dir:  {}", args.cache_dir.display());
    println!("  Output dir: {}", args.output_dir.display());
    println!("======================================================================");

    std::fs::create_dir_all(&args.output_dir)?;

    // Discover files
    let all_files = list_cache_files(&args.cache_dir);
    println!("Found {} cache files", all_files.len());

    // Filter to specific date if requested
    let files_to_process: Vec<PathBuf> = if let Some(ref date) = args.date {
        all_files.into_iter()
            .filter(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or("").starts_with(date))
            .collect()
    } else if args.resume {
        all_files.into_iter()
            .filter(|p| {
                let stem = p.file_stem().and_then(|n| n.to_str()).unwrap_or("");
                let date_part = stem.replace("_snapshots", "");
                let out_name = format!("{}_mbo_features.npz", date_part);
                !args.output_dir.join(out_name).exists()
            })
            .collect()
    } else {
        all_files
    };

    if files_to_process.is_empty() {
        println!("No files to process.");
        return Ok(());
    }
    println!("Processing {} files...", files_to_process.len());

    // Configure thread pool
    let n_workers = if args.workers == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    } else {
        args.workers
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build_global()
        .ok();
    println!("Using {} workers", n_workers);

    let t_start = Instant::now();
    let output_dir = Arc::new(args.output_dir.clone());

    // Progress tracking
    use std::sync::atomic::{AtomicUsize, Ordering};
    let completed = Arc::new(AtomicUsize::new(0));
    let total_bars = Arc::new(AtomicUsize::new(0));
    let n_files = files_to_process.len();

    let results: Vec<Result<(String, usize, f64)>> = files_to_process
        .par_iter()
        .map(|input_path| {
            let stem = input_path.file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .replace("_snapshots", "");

            let out_name = format!("{}_mbo_features.npz", stem);
            let output_path = output_dir.join(&out_name);

            match process_one_day(input_path, &output_path) {
                Ok((n, elapsed)) => {
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    total_bars.fetch_add(n, Ordering::Relaxed);
                    println!("  [{}/{}] {} → {} bars in {:.2}s",
                        done, n_files, stem, n, elapsed);
                    Ok((stem, n, elapsed))
                }
                Err(e) => {
                    eprintln!("  ERROR: {} — {}", stem, e);
                    Err(e)
                }
            }
        })
        .collect();

    let elapsed = t_start.elapsed();
    let n_ok = results.iter().filter(|r| r.is_ok()).count();
    let n_bars_total = total_bars.load(std::sync::atomic::Ordering::Relaxed);

    println!("\n======================================================================");
    println!("COMPLETE");
    println!("  Files processed: {}/{}", n_ok, n_files);
    println!("  Total bars:      {}", n_bars_total);
    println!("  Elapsed:         {:.1}s ({:.1} min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    if n_bars_total > 0 && elapsed.as_secs_f64() > 0.0 {
        println!("  Throughput:      {:.0}k bars/sec",
            n_bars_total as f64 / elapsed.as_secs_f64() / 1000.0);
    }
    println!("  Output:          {}", output_dir.display());
    println!("======================================================================");

    Ok(())
}
