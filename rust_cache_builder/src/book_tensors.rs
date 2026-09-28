/// Book Snapshot Tensors — raw 10-level order book state as 2D tensors for Spatial CNN.
///
/// For each 100ms bar, outputs the full book state:
/// - 10 bid levels x 4 features: (price_relative_to_mid, depth_lots, num_orders, queue_age_ms)
/// - 10 ask levels x 4 features: same
/// - Total: 20 levels x 4 features per bar, structured as 2D
///
/// Output format: NPZ with:
/// - `book_tensors`: shape (n_bars, 20, 4) float32
/// - `mid_prices`: shape (n_bars,) float64 for target computation
/// - `timestamps`: shape (n_bars,) int64

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use ordered_float::OrderedFloat;
use zip::{write::SimpleFileOptions, ZipWriter, CompressionMethod};

pub const DEPTH_LEVELS: usize = 10;
pub const TOTAL_LEVELS: usize = DEPTH_LEVELS * 2;  // 10 bid + 10 ask
pub const LEVEL_FEATURES: usize = 4;  // price_rel, depth, num_orders, queue_age

const TICK_SIZE: f64 = 0.25;

/// Accumulated book tensor data for one trading day.
pub struct BookTensorDayData {
    /// Flattened book tensors: n_bars * TOTAL_LEVELS * LEVEL_FEATURES as f32
    pub book_tensors: Vec<f32>,
    /// Mid prices per bar
    pub mid_prices: Vec<f64>,
    /// Timestamps per bar
    pub timestamps: Vec<i64>,
    pub n_bars: usize,
}

impl BookTensorDayData {
    pub fn new() -> Self {
        BookTensorDayData {
            book_tensors: Vec::new(),
            mid_prices: Vec::new(),
            timestamps: Vec::new(),
            n_bars: 0,
        }
    }

    /// Push a complete bar of book tensor data.
    /// `tensor` is [TOTAL_LEVELS][LEVEL_FEATURES] in row-major order.
    pub fn push_bar(&mut self, tensor: &[[f32; LEVEL_FEATURES]; TOTAL_LEVELS], mid: f64, ts: i64) {
        for level in tensor.iter() {
            self.book_tensors.extend_from_slice(level);
        }
        self.mid_prices.push(mid);
        self.timestamps.push(ts);
        self.n_bars += 1;
    }
}

/// Tracks queue ages for order book levels to produce the queue_age feature.
///
/// For each price level, we track when it first appeared (or was last refreshed).
/// Queue age = time since the level was first populated with its current orders.
pub struct QueueAgeTracker {
    /// Maps price level -> earliest add timestamp (ns) for currently live orders at that level
    bid_level_first_add: HashMap<OrderedFloat<f64>, u64>,
    ask_level_first_add: HashMap<OrderedFloat<f64>, u64>,
    /// Maps order_id -> (price, side) for tracking which level an order belongs to
    order_levels: HashMap<u64, (f64, bool)>,  // bool: true=bid, false=ask
}

impl QueueAgeTracker {
    pub fn new() -> Self {
        QueueAgeTracker {
            bid_level_first_add: HashMap::new(),
            ask_level_first_add: HashMap::new(),
            order_levels: HashMap::new(),
        }
    }

    /// Record an add event. Updates the first-add timestamp for the level if needed.
    pub fn on_add(&mut self, order_id: u64, price: f64, is_bid: bool, ts_ns: u64) {
        let key = OrderedFloat(price);
        if is_bid {
            self.bid_level_first_add.entry(key).or_insert(ts_ns);
        } else {
            self.ask_level_first_add.entry(key).or_insert(ts_ns);
        }
        self.order_levels.insert(order_id, (price, is_bid));
    }

    /// Record a cancel/trade that removes volume from a level.
    /// If the level is now empty, remove the timestamp tracking.
    pub fn on_remove(
        &mut self,
        order_id: u64,
        bids: &BTreeMap<OrderedFloat<f64>, f64>,
        asks: &BTreeMap<OrderedFloat<f64>, f64>,
    ) {
        if let Some((price, is_bid)) = self.order_levels.remove(&order_id) {
            let key = OrderedFloat(price);
            if is_bid {
                // If level no longer exists in the book, remove tracking
                if !bids.contains_key(&key) {
                    self.bid_level_first_add.remove(&key);
                }
            } else {
                if !asks.contains_key(&key) {
                    self.ask_level_first_add.remove(&key);
                }
            }
        }
    }

    /// Get the queue age in milliseconds for a given price level.
    pub fn get_queue_age_ms(&self, price: f64, is_bid: bool, current_ts_ns: u64) -> f64 {
        let key = OrderedFloat(price);
        let first_add = if is_bid {
            self.bid_level_first_add.get(&key)
        } else {
            self.ask_level_first_add.get(&key)
        };

        match first_add {
            Some(&ts) if current_ts_ns >= ts => {
                (current_ts_ns - ts) as f64 / 1_000_000.0  // ns -> ms
            }
            _ => 0.0,
        }
    }

    /// Reset all tracking (e.g., on book clear).
    pub fn reset(&mut self) {
        self.bid_level_first_add.clear();
        self.ask_level_first_add.clear();
        self.order_levels.clear();
    }
}

/// Extract a book tensor from the current order book state.
///
/// Returns a [TOTAL_LEVELS][LEVEL_FEATURES] array:
///   Rows 0..9   = bid levels (best to worst)
///   Rows 10..19 = ask levels (best to worst)
///
/// Features per level:
///   [0] price_relative_to_mid (in ticks)
///   [1] depth_lots
///   [2] num_orders
///   [3] queue_age_seconds
pub fn extract_book_tensor(
    bids: &BTreeMap<OrderedFloat<f64>, f64>,
    asks: &BTreeMap<OrderedFloat<f64>, f64>,
    bid_order_counts: &HashMap<OrderedFloat<f64>, u32>,
    ask_order_counts: &HashMap<OrderedFloat<f64>, u32>,
    queue_tracker: &QueueAgeTracker,
    mid: f64,
    timestamp_ns: u64,
) -> [[f32; LEVEL_FEATURES]; TOTAL_LEVELS] {
    let mut tensor = [[0.0f32; LEVEL_FEATURES]; TOTAL_LEVELS];

    if !mid.is_finite() || mid == 0.0 {
        return tensor;
    }

    // Bid levels: descending price (best bid first)
    let bid_prices: Vec<f64> = bids.keys().rev().take(DEPTH_LEVELS).map(|k| k.0).collect();
    for (i, &price) in bid_prices.iter().enumerate() {
        let size = *bids.get(&OrderedFloat(price)).unwrap_or(&0.0);
        let num_orders = *bid_order_counts.get(&OrderedFloat(price)).unwrap_or(&0) as f64;
        let queue_age_ms = queue_tracker.get_queue_age_ms(price, true, timestamp_ns);

        tensor[i][0] = ((price - mid) / TICK_SIZE) as f32;   // relative to mid in ticks
        tensor[i][1] = size as f32;                           // depth in lots
        tensor[i][2] = num_orders as f32;                     // number of orders
        tensor[i][3] = (queue_age_ms / 1000.0) as f32;       // queue age in seconds
    }

    // Ask levels: ascending price (best ask first)
    let ask_prices: Vec<f64> = asks.keys().take(DEPTH_LEVELS).map(|k| k.0).collect();
    for (i, &price) in ask_prices.iter().enumerate() {
        let size = *asks.get(&OrderedFloat(price)).unwrap_or(&0.0);
        let num_orders = *ask_order_counts.get(&OrderedFloat(price)).unwrap_or(&0) as f64;
        let queue_age_ms = queue_tracker.get_queue_age_ms(price, false, timestamp_ns);

        let row = DEPTH_LEVELS + i;
        tensor[row][0] = ((price - mid) / TICK_SIZE) as f32;
        tensor[row][1] = size as f32;
        tensor[row][2] = num_orders as f32;
        tensor[row][3] = (queue_age_ms / 1000.0) as f32;
    }

    tensor
}

/// Write BookTensorDayData to NPZ file.
pub fn save_book_tensors_npz(
    path: &std::path::Path,
    day: &BookTensorDayData,
) -> anyhow::Result<()> {

    let n = day.n_bars;
    if n == 0 {
        return Err(anyhow::anyhow!("No data to save"));
    }

    let file = std::fs::File::create(path)?;
    let mut zip = ZipWriter::new(file);

    // book_tensors: (n_bars, TOTAL_LEVELS, LEVEL_FEATURES) float32
    {
        let bytes: Vec<u8> = day.book_tensors.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        write_npy_to_zip(
            &mut zip,
            "book_tensors",
            "<f4",
            &[n, TOTAL_LEVELS, LEVEL_FEATURES],
            &bytes,
        )?;
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
