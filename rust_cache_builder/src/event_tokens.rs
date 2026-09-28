/// Event Tokenizer — converts raw MBO events into tokenized sequences for Transformer models.
///
/// For each 100ms bar, outputs a SEQUENCE of the raw events that occurred:
/// - event_type: u8 (0=add_bid, 1=add_ask, 2=cancel_bid, 3=cancel_ask, 4=modify_bid, 5=modify_ask, 6=trade_bid, 7=trade_ask)
/// - price_level: i8 (relative to mid, in ticks: -10 to +10)
/// - size_bucket: u8 (0=1-5 lots, 1=6-20, 2=21-50, 3=51-100, 4=100+)
/// - time_delta_ms: u16 (ms since last event in this bar, capped at 100)
/// - sequence_position: u16 (position within bar)
///
/// Output format: NPZ with:
/// - `event_sequences`: shape (n_bars, MAX_EVENTS_PER_BAR, 5) i16 padded
/// - `sequence_lengths`: shape (n_bars,) u16 actual length before padding
/// - `mid_prices`: shape (n_bars,) f64 for target computation

use std::io::Write;
use zip::{write::SimpleFileOptions, ZipWriter, CompressionMethod};
use crate::lob::{parse_action, parse_side, Action, Side};

pub const MAX_EVENTS_PER_BAR: usize = 200;
pub const TOKEN_FEATURES: usize = 5;
const TICK_SIZE: f64 = 0.25;

/// A single tokenized event within a bar.
#[derive(Debug, Clone, Copy, Default)]
pub struct EventToken {
    pub event_type: i16,
    pub price_level: i16,
    pub size_bucket: i16,
    pub time_delta_ms: i16,
    pub sequence_position: i16,
}

/// Accumulated event token data for one trading day.
pub struct EventTokenDayData {
    /// Flattened event sequences: n_bars * MAX_EVENTS_PER_BAR * TOKEN_FEATURES as i16
    pub event_sequences: Vec<i16>,
    /// Actual sequence length per bar
    pub sequence_lengths: Vec<u16>,
    /// Mid prices per bar
    pub mid_prices: Vec<f64>,
    /// Timestamps per bar
    pub timestamps: Vec<i64>,
    pub n_bars: usize,
}

impl EventTokenDayData {
    pub fn new() -> Self {
        EventTokenDayData {
            event_sequences: Vec::new(),
            sequence_lengths: Vec::new(),
            mid_prices: Vec::new(),
            timestamps: Vec::new(),
            n_bars: 0,
        }
    }

    /// Push a complete bar of event tokens.
    /// `tokens` is the list of events in this bar (will be padded/truncated to MAX_EVENTS_PER_BAR).
    pub fn push_bar(&mut self, tokens: &[EventToken], mid: f64, ts: i64) {
        let actual_len = tokens.len().min(MAX_EVENTS_PER_BAR);

        // Write actual events
        for i in 0..actual_len {
            let t = &tokens[i];
            self.event_sequences.push(t.event_type);
            self.event_sequences.push(t.price_level);
            self.event_sequences.push(t.size_bucket);
            self.event_sequences.push(t.time_delta_ms);
            self.event_sequences.push(t.sequence_position);
        }

        // Pad remaining positions with zeros
        let padding = MAX_EVENTS_PER_BAR - actual_len;
        for _ in 0..(padding * TOKEN_FEATURES) {
            self.event_sequences.push(0);
        }

        self.sequence_lengths.push(actual_len as u16);
        self.mid_prices.push(mid);
        self.timestamps.push(ts);
        self.n_bars += 1;
    }
}

/// Collector that accumulates raw events within a bar and produces tokens on snapshot.
pub struct EventTokenCollector {
    /// Events accumulated in the current bar
    current_bar_events: Vec<RawBarEvent>,
    /// Timestamp of the first event in this bar (nanoseconds)
    bar_start_ns: u64,
}

/// Raw event data collected during a bar, before tokenization.
#[derive(Debug, Clone)]
struct RawBarEvent {
    action: u8,
    side: u8,
    price_raw: i64,
    size: u32,
    ts_event: u64,
}

impl EventTokenCollector {
    pub fn new() -> Self {
        EventTokenCollector {
            current_bar_events: Vec::with_capacity(256),
            bar_start_ns: 0,
        }
    }

    /// Record a raw MBO event in the current bar.
    pub fn record_event(
        &mut self,
        action: u8,
        side: u8,
        price_raw: i64,
        size: u32,
        ts_event: u64,
    ) {
        if self.bar_start_ns == 0 {
            self.bar_start_ns = ts_event;
        }
        self.current_bar_events.push(RawBarEvent {
            action,
            side,
            price_raw,
            size,
            ts_event,
        });
    }

    /// Flush the current bar: tokenize all accumulated events relative to the given mid price.
    /// Returns the token sequence. Resets internal state for the next bar.
    pub fn flush_bar(&mut self, mid_price: f64) -> Vec<EventToken> {
        if self.current_bar_events.is_empty() {
            return Vec::new();
        }

        let mid_ticks = mid_price / TICK_SIZE;
        let bar_start = self.bar_start_ns;
        let mut last_ts = bar_start;

        let mut tokens = Vec::with_capacity(self.current_bar_events.len().min(MAX_EVENTS_PER_BAR));

        for (seq_pos, event) in self.current_bar_events.iter().enumerate() {
            if seq_pos >= MAX_EVENTS_PER_BAR {
                break;
            }

            let action = parse_action(event.action);
            let side = parse_side(event.side);

            // Compute event_type: combine action + side
            let event_type = match (action, side) {
                (Action::Add, Side::Bid) => 0,
                (Action::Add, Side::Ask) => 1,
                (Action::Cancel, Side::Bid) => 2,
                (Action::Cancel, Side::Ask) => 3,
                (Action::Modify, Side::Bid) => 4,
                (Action::Modify, Side::Ask) => 5,
                (Action::Trade, Side::Bid) => 6,  // passive side is bid = aggressive sell
                (Action::Trade, Side::Ask) => 7,  // passive side is ask = aggressive buy
                (Action::Fill, _) => continue,     // skip fill events
                (Action::Clear, _) => continue,    // skip clear events
                _ => continue,                     // skip unknown
            };

            // Compute price_level: relative to mid in ticks, clamped to [-10, +10]
            let price = event.price_raw as f64 * 1e-9;
            let price_ticks = price / TICK_SIZE;
            let price_level = (price_ticks - mid_ticks).round() as i16;
            let price_level = price_level.clamp(-10, 10);

            // Compute size_bucket
            let size_bucket = match event.size {
                1..=5 => 0,
                6..=20 => 1,
                21..=50 => 2,
                51..=100 => 3,
                _ => 4,
            };

            // Compute time_delta_ms: ms since last event, capped at 100
            let delta_ns = event.ts_event.saturating_sub(last_ts);
            let delta_ms = (delta_ns / 1_000_000) as u16;
            let time_delta_ms = delta_ms.min(100) as i16;
            last_ts = event.ts_event;

            tokens.push(EventToken {
                event_type,
                price_level,
                size_bucket,
                time_delta_ms,
                sequence_position: seq_pos as i16,
            });
        }

        // Reset for next bar
        self.current_bar_events.clear();
        self.bar_start_ns = 0;

        tokens
    }

    /// Reset state (e.g., on book clear).
    pub fn reset(&mut self) {
        self.current_bar_events.clear();
        self.bar_start_ns = 0;
    }
}

/// Write EventTokenDayData to NPZ file.
pub fn save_event_tokens_npz(
    path: &std::path::Path,
    day: &EventTokenDayData,
) -> anyhow::Result<()> {
    let n = day.n_bars;
    if n == 0 {
        return Err(anyhow::anyhow!("No data to save"));
    }

    let file = std::fs::File::create(path)?;
    let mut zip = ZipWriter::new(file);

    // event_sequences: (n_bars, MAX_EVENTS_PER_BAR, TOKEN_FEATURES) i16
    {
        let bytes: Vec<u8> = day.event_sequences.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        write_npy_to_zip(
            &mut zip,
            "event_sequences",
            "<i2",
            &[n, MAX_EVENTS_PER_BAR, TOKEN_FEATURES],
            &bytes,
        )?;
    }

    // sequence_lengths: (n_bars,) u16
    {
        let bytes: Vec<u8> = day.sequence_lengths.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        write_npy_to_zip(&mut zip, "sequence_lengths", "<u2", &[n], &bytes)?;
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

/// Write a single .npy entry into a zip writer. (Same as npz.rs but local to avoid coupling.)
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
