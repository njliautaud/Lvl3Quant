/// NPZ writer — writes numpy-compatible .npz files.
///
/// NPZ format: ZIP archive where each array is stored as a .npy file.
/// We write:
///   - global_features.npy  : (N, 96) float32
///   - node_features.npy    : (N, 20, 9) float32
///   - mid_prices.npy       : (N,) float64
///   - timestamps.npy       : (N,) int64
///
/// NumPy .npy format v1.0:
///   Magic: \x93NUMPY (6 bytes)
///   Major: \x01 Minor: \x00 (2 bytes)
///   Header len: little-endian uint16 (2 bytes)
///   Header: ASCII Python dict string, padded to 64-byte alignment with spaces,
///           then a single '\n'
///   Data: raw binary in C order (row-major)

use std::io::{self, Write};
use zip::{write::SimpleFileOptions, ZipWriter, CompressionMethod};
use std::path::Path;
use std::fs::File;

/// Write a single .npy array entry into a zip writer.
/// `data` is the raw bytes (already in the correct dtype, C-order).
/// `dtype_str` is numpy dtype string e.g. "<f4" (float32 LE), "<f8" (float64 LE), "<i8" (int64 LE)
/// `shape` is the array dimensions.
fn write_npy_to_zip<W: Write + io::Seek>(
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
    // Python dict header — must match numpy's expectation exactly
    let shape_tuple = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape_str)
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype_str, shape_tuple
    );

    // Header must be padded so that (6 + 2 + 2 + header_len) % 64 == 0
    // where header includes the dict + padding spaces + trailing '\n'
    let prefix_len = 6 + 2 + 2; // magic + version + header_len field
    let min_len = header_dict.len() + 1; // +1 for '\n'
    let total_prefix = prefix_len + min_len;
    let pad_to = ((total_prefix + 63) / 64) * 64;
    let padding = pad_to - total_prefix;
    let header_padded = format!("{}{}\n", header_dict, " ".repeat(padding));

    let header_len = header_padded.len() as u16;

    // Build .npy bytes
    let mut npy_bytes = Vec::with_capacity(prefix_len + header_padded.len() + data.len());
    npy_bytes.extend_from_slice(b"\x93NUMPY");  // magic
    npy_bytes.push(1u8);                         // major version
    npy_bytes.push(0u8);                         // minor version
    npy_bytes.extend_from_slice(&header_len.to_le_bytes());
    npy_bytes.extend_from_slice(header_padded.as_bytes());
    npy_bytes.extend_from_slice(data);

    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    zip.start_file(format!("{}.npy", name), options)?;
    zip.write_all(&npy_bytes)?;
    Ok(())
}

pub struct DayData {
    /// Global features: rows × 96 float32 (C order)
    pub global_features: Vec<f32>,
    /// Node features: rows × 20 × 9 float32 (C order)
    pub node_features: Vec<f32>,
    /// Mid prices: rows float64
    pub mid_prices: Vec<f64>,
    /// Timestamps: rows int64
    pub timestamps: Vec<i64>,
    pub n_rows: usize,
}

impl DayData {
    pub fn new() -> Self {
        DayData {
            global_features: Vec::new(),
            node_features: Vec::new(),
            mid_prices: Vec::new(),
            timestamps: Vec::new(),
            n_rows: 0,
        }
    }

    pub fn push(
        &mut self,
        global: &[f32; crate::engineering::TOTAL_GLOBAL],
        nodes: &[[f32; crate::engineering::NODE_FEATURES]; crate::engineering::NUM_NODES],
        mid: f64,
        ts: i64,
    ) {
        self.global_features.extend_from_slice(global);
        for node in nodes.iter() {
            self.node_features.extend_from_slice(node);
        }
        self.mid_prices.push(mid);
        self.timestamps.push(ts);
        self.n_rows += 1;
    }
}

/// Save a DayData to an .npz file.
/// Output matches Python's np.savez_compressed format.
pub fn save_npz(path: &Path, day: &DayData) -> anyhow::Result<()> {
    let n = day.n_rows;
    if n == 0 {
        return Err(anyhow::anyhow!("No data to save"));
    }

    let file = File::create(path)?;
    let mut zip = ZipWriter::new(file);

    // global_features: (N, 96) float32
    {
        let bytes: Vec<u8> = day.global_features.iter()
            .flat_map(|&f| <f32>::to_le_bytes(f).into_iter())
            .collect();
        write_npy_to_zip(
            &mut zip,
            "global_features",
            "<f4",
            &[n, crate::engineering::TOTAL_GLOBAL],
            &bytes,
        )?;
    }

    // node_features: (N, 20, 9) float32
    {
        let bytes: Vec<u8> = day.node_features.iter()
            .flat_map(|&f| <f32>::to_le_bytes(f).into_iter())
            .collect();
        write_npy_to_zip(
            &mut zip,
            "node_features",
            "<f4",
            &[n, crate::engineering::NUM_NODES, crate::engineering::NODE_FEATURES],
            &bytes,
        )?;
    }

    // mid_prices: (N,) float64
    {
        let bytes: Vec<u8> = day.mid_prices.iter()
            .flat_map(|&f| <f64>::to_le_bytes(f).into_iter())
            .collect();
        write_npy_to_zip(&mut zip, "mid_prices", "<f8", &[n], &bytes)?;
    }

    // timestamps: (N,) int64
    {
        let bytes: Vec<u8> = day.timestamps.iter()
            .flat_map(|&i| <i64>::to_le_bytes(i).into_iter())
            .collect();
        write_npy_to_zip(&mut zip, "timestamps", "<i8", &[n], &bytes)?;
    }

    zip.finish()?;
    Ok(())
}
