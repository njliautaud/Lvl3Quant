/// Trade Flow Sequences — raw trade-by-trade data for LSTM models.
///
/// For each 100ms bar, outputs the sequence of trades:
/// - aggressor_side: i8 (-1=sell, +1=buy)
/// - size_lots: u16
/// - price_ticks_from_mid: i8
/// - time_delta_ms: u16
/// - cumulative_volume: u32
///
/// Output format: NPZ with:
/// - `trade_sequences`: shape (n_bars, MAX_TRADES_PER_BAR, 5) float32 padded
/// - `trade_lengths`: shape (n_bars,) u16 actual length before padding
/// - `mid_prices`: shape (n_bars,) f64 for target computation
/// - `timestamps`: shape (n_bars,) i64

use std::io::Write;
use zip::{write::SimpleFileOptions, ZipWriter, CompressionMethod};
use crate::lob::{parse_side, Side};

pub const MAX_TRADES_PER_BAR: usize = 50;
pub const TRADE_FEATURES: usize = 5;
const TICK_SIZE: f64 = 0.25;

/// A single trade within a bar.
#[derive(Debug, Clone, Copy, Default)]
pub struct TradeRecord {
    pub aggressor_side: f32,         // -1.0=sell, +1.0=buy
    pub size_lots: f32,              // trade size in lots
    pub price_ticks_from_mid: f32,   // relative to mid in ticks
    pub time_delta_ms: f32,          // ms since last trade in this bar
    pub cumulative_volume: f32,      // running total volume in this bar
}

/// Accumulated trade flow data for one trading day.
pub struct TradeFlowDayData {
    /// Flattened trade sequences: n_bars * MAX_TRADES_PER_BAR * TRADE_FEATURES as f32
    pub trade_sequences: Vec<f32>,
    /// Actual trade count per bar
    pub trade_lengths: Vec<u16>,
    /// Mid prices per bar
    pub mid_prices: Vec<f64>,
    /// Timestamps per bar
    pub timestamps: Vec<i64>,
    pub n_bars: usize,
}

impl TradeFlowDayData {
    pub fn new() -> Self {
        TradeFlowDayData {
            trade_sequences: Vec::new(),
            trade_lengths: Vec::new(),
            mid_prices: Vec::new(),
            timestamps: Vec::new(),
            n_bars: 0,
        }
    }

    /// Push a complete bar of trade records.
    pub fn push_bar(&mut self, trades: &[TradeRecord], mid: f64, ts: i64) {
        let actual_len = trades.len().min(MAX_TRADES_PER_BAR);

        for i in 0..actual_len {
            let t = &trades[i];
            self.trade_sequences.push(t.aggressor_side);
            self.trade_sequences.push(t.size_lots);
            self.trade_sequences.push(t.price_ticks_from_mid);
            self.trade_sequences.push(t.time_delta_ms);
            self.trade_sequences.push(t.cumulative_volume);
        }

        // Pad remaining positions with zeros
        let padding = MAX_TRADES_PER_BAR - actual_len;
        for _ in 0..(padding * TRADE_FEATURES) {
            self.trade_sequences.push(0.0);
        }

        self.trade_lengths.push(actual_len as u16);
        self.mid_prices.push(mid);
        self.timestamps.push(ts);
        self.n_bars += 1;
    }
}

/// Collector that accumulates trade events within a bar.
pub struct TradeFlowCollector {
    /// Raw trades in the current bar
    current_bar_trades: Vec<RawTrade>,
    /// Cumulative volume within this bar
    cumulative_vol: u32,
}

#[derive(Debug, Clone)]
struct RawTrade {
    /// The passive side of the trade (from the resting order).
    /// Bid passive = aggressive sell; Ask passive = aggressive buy.
    passive_side: u8,
    /// Trade price (fixed-point raw)
    price_raw: i64,
    /// Trade size in lots
    size: u32,
    /// Event timestamp (nanoseconds)
    ts_event: u64,
}

impl TradeFlowCollector {
    pub fn new() -> Self {
        TradeFlowCollector {
            current_bar_trades: Vec::with_capacity(64),
            cumulative_vol: 0,
        }
    }

    /// Record a trade event. Only call this for Action::Trade events.
    ///
    /// `passive_side` is the side of the resting order that was hit:
    ///   Side::Bid = aggressive seller hit the bid
    ///   Side::Ask = aggressive buyer lifted the ask
    pub fn record_trade(
        &mut self,
        passive_side: u8,
        price_raw: i64,
        size: u32,
        ts_event: u64,
    ) {
        self.cumulative_vol += size;
        self.current_bar_trades.push(RawTrade {
            passive_side,
            price_raw,
            size,
            ts_event,
        });
    }

    /// Flush the current bar: produce trade records relative to the given mid price.
    /// Resets internal state for the next bar.
    pub fn flush_bar(&mut self, mid_price: f64) -> Vec<TradeRecord> {
        if self.current_bar_trades.is_empty() {
            self.cumulative_vol = 0;
            return Vec::new();
        }

        let mid_ticks = if mid_price.is_finite() && mid_price > 0.0 {
            mid_price / TICK_SIZE
        } else {
            0.0
        };

        let mut records = Vec::with_capacity(self.current_bar_trades.len().min(MAX_TRADES_PER_BAR));
        let mut last_ts = self.current_bar_trades[0].ts_event;
        let mut running_vol: u32 = 0;

        for (idx, trade) in self.current_bar_trades.iter().enumerate() {
            if idx >= MAX_TRADES_PER_BAR {
                break;
            }

            let side = parse_side(trade.passive_side);
            // Passive bid = aggressive sell (-1), passive ask = aggressive buy (+1)
            let aggressor_side: f32 = match side {
                Side::Bid => -1.0,   // seller aggressor
                Side::Ask => 1.0,    // buyer aggressor
                _ => 0.0,            // unknown
            };

            let price = trade.price_raw as f64 * 1e-9;
            let price_ticks = price / TICK_SIZE;
            let price_rel = if mid_ticks > 0.0 {
                (price_ticks - mid_ticks) as f32
            } else {
                0.0
            };
            // Clamp to reasonable range
            let price_rel = price_rel.clamp(-20.0, 20.0);

            // Time delta (ms since last trade in bar)
            let delta_ns = trade.ts_event.saturating_sub(last_ts);
            let delta_ms = (delta_ns as f64 / 1_000_000.0).min(100.0) as f32;
            last_ts = trade.ts_event;

            running_vol += trade.size;

            records.push(TradeRecord {
                aggressor_side,
                size_lots: trade.size as f32,
                price_ticks_from_mid: price_rel,
                time_delta_ms: delta_ms,
                cumulative_volume: running_vol as f32,
            });
        }

        // Reset for next bar
        self.current_bar_trades.clear();
        self.cumulative_vol = 0;

        records
    }

    /// Reset state (e.g., on book clear).
    pub fn reset(&mut self) {
        self.current_bar_trades.clear();
        self.cumulative_vol = 0;
    }
}

/// Write TradeFlowDayData to NPZ file.
pub fn save_trade_flow_npz(
    path: &std::path::Path,
    day: &TradeFlowDayData,
) -> anyhow::Result<()> {

    let n = day.n_bars;
    if n == 0 {
        return Err(anyhow::anyhow!("No data to save"));
    }

    let file = std::fs::File::create(path)?;
    let mut zip = ZipWriter::new(file);

    // trade_sequences: (n_bars, MAX_TRADES_PER_BAR, TRADE_FEATURES) float32
    {
        let bytes: Vec<u8> = day.trade_sequences.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        write_npy_to_zip(
            &mut zip,
            "trade_sequences",
            "<f4",
            &[n, MAX_TRADES_PER_BAR, TRADE_FEATURES],
            &bytes,
        )?;
    }

    // trade_lengths: (n_bars,) u16
    {
        let bytes: Vec<u8> = day.trade_lengths.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        write_npy_to_zip(&mut zip, "trade_lengths", "<u2", &[n], &bytes)?;
    }

    // mid_prices: (n_bars,) f64
    {
        let bytes: Vec<u8> = day.mid_prices.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        write_npy_to_zip(&mut zip, "mid_prices", "<f8", &[n], &bytes)?;
    }

    // timestamps: (n_bars,) i64
    {
        let bytes: Vec<u8> = day.timestamps.iter()
            .flat_map(|&i| i.to_le_bytes())
            .collect();
        write_npy_to_zip(&mut zip, "timestamps", "<i8", &[n], &bytes)?;
    }

    zip.finish()?;
    Ok(())
}

/// Write a single .npy entry into a zip writer.
fn write_npy_to_zip<W: Write + std::io::Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    dtype_str: &str,
    shape: &[usize],
    data: &[u8],
) -> anyhow::Result<()> {
    let shape_str = shape.iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let shape_tuple = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape_str)
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype_str, shape_tuple
    );

    let prefix_len = 6 + 2 + 2;
    let min_len = header_dict.len() + 1;
    let total_prefix = prefix_len + min_len;
    let pad_to = ((total_prefix + 63) / 64) * 64;
    let padding = pad_to - total_prefix;
    let header_padded = format!("{}{}\n", header_dict, " ".repeat(padding));
    let header_len = header_padded.len() as u16;

    let mut npy_bytes = Vec::with_capacity(prefix_len + header_padded.len() + data.len());
    npy_bytes.extend_from_slice(b"\x93NUMPY");
    npy_bytes.push(1u8);
    npy_bytes.push(0u8);
    npy_bytes.extend_from_slice(&header_len.to_le_bytes());
    npy_bytes.extend_from_slice(header_padded.as_bytes());
    npy_bytes.extend_from_slice(data);

    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file(format!("{}.npy", name), options)?;
    zip.write_all(&npy_bytes)?;
    Ok(())
}
