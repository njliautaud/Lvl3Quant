/// Debug tool to inspect feature values for specific bars
use std::io::{Read, Seek, Cursor};
use zip::ZipArchive;
use std::fs::File;
use std::path::Path;

mod mbo_feature_expander;
use mbo_feature_expander::{load_npz, N_GLOBAL_FEATURES, TOTAL_FEATURES};

const COL_AGGR_BUY_COUNT: usize = 35;
const COL_AGGR_SELL_COUNT: usize = 36;
const COL_BUY_VOL: usize = 11;
const COL_SELL_VOL: usize = 12;

fn main() {
    let path = Path::new("../data/processed/medium_snapshots_cache/2025-07-14_snapshots.npz");
    let input = load_npz(path).expect("Failed to load NPZ");
    let n = input.n_bars;
    let global = &input.global_features;

    println!("n_bars = {}", n);
    println!("n_global_cols = {} (should be {})", global.len() / n, N_GLOBAL_FEATURES);

    // Print aggr_dom for all bars 0-110 to trace run_length
    println!("\naggr_dom for bars 0-110 (all, not filtered):");
    let mut run = 0.0f64;
    for i in 0..110.min(n) {
        let ab = global[i * N_GLOBAL_FEATURES + COL_AGGR_BUY_COUNT];
        let as_ = global[i * N_GLOBAL_FEATURES + COL_AGGR_SELL_COUNT];
        let aggr_diff = (ab as f64) - (as_ as f64);
        let aggr_sign = aggr_diff.signum();
        let aggr_dom = if aggr_sign.abs() > 0.0 { 1.0f64 } else { 0.0f64 };
        if aggr_dom > 0.5 {
            run += 1.0;
        } else {
            run = 0.0;
        }
        // Only print bars where something nonzero happens OR first 5 bars
        if i < 5 || aggr_dom > 0.5 || run > 0.0 {
            println!("  bar {:3}: ab={:.6} as_={:.6} aggr_diff={:.6} aggr_dom={} run={}",
                i, ab, as_, aggr_diff, aggr_dom, run);
        }
    }

    // Also check raw byte values at bars 0-5 for COL_AGGR_BUY_COUNT=35
    println!("\nRaw float32 values at col 35 (aggr_buy) bars 0-10:");
    for i in 0..10.min(n) {
        let val = global[i * N_GLOBAL_FEATURES + COL_AGGR_BUY_COUNT];
        let bits = val.to_bits();
        println!("  bar {:2}: f32={} bits=0x{:08X}", i, val, bits);
    }

    // Compute features
    let features = mbo_feature_expander::compute_mbo_features(&input);

    // Check aggressive_streak_5 (col 265) and buy_run_max_5 (col 260)
    println!("\naggressive_streak_5 (col 265) at bars 95-115:");
    for i in 95..115.min(n) {
        println!("  bar {}: {}", i, features[i * TOTAL_FEATURES + 265]);
    }

    println!("\nbuy_run_max_5 (col 260) at bars 95-115:");
    for i in 95..115.min(n) {
        println!("  bar {}: {}", i, features[i * TOTAL_FEATURES + 260]);
    }

    // Print raw aggr values to check stride
    println!("\nChecking stride: first 5 rows, cols 33-38:");
    for i in 0..5.min(n) {
        let base = i * N_GLOBAL_FEATURES;
        println!("  row {}: col33={} col34={} col35={} col36={} col37={} col38={}",
            i,
            global[base + 33],
            global[base + 34],
            global[base + 35],
            global[base + 36],
            global[base + 37],
            global[base + 38],
        );
    }
}
