"""
v3.4.2 MEMORY-SAFE dataset variant — HC #395 root-cause fix for kernel OOM
on Neptune (32 GB RAM) when running 60d sliding-window training.

ROOT CAUSE (identified 2026-05-16):
    SmartV34DualTrunkDataset.__init__ loaded EVERY date's book pyramid
    fully into self.book_data as a dict of in-memory float32 arrays.
    Per-day pyramid (N, 5, 4) float32 ~ 520 MB (N ≈ 6.5M rows). At
    wf_train_days=60 → 31 GB resident set just for book features.
    journalctl OOM at 11:21:31 was killing python at anon-rss=29.9 GB,
    consistent with this calculation. v3.3 ran fine because v3.3 had
    no BookCNN dual-trunk and no per-date book array.

FIX (this file):
    SmartV34MemmapDualTrunkDataset replaces SmartV34DualTrunkDataset.
    For each date the preprocessed (N, 5, 4) float32 pyramid is built
    ONCE and persisted as a tiny .npy header + raw float32 mmap file
    under BOOK_PYRAMID_CACHE_DIR. Subsequent dataset instantiations
    open the file via np.memmap (read-only) — pages are pulled into
    the OS page cache only as windows are actually sliced. The Python
    process holds no dense copy. Expected resident-set drop from 31 GB
    to <2 GB at 60d setup. Disk cost ~31 GB once (one-time amortised).

    First-time cache build per date streams the NPZ -> sidecar without
    holding more than one day in RAM at a time, so cache construction
    is also OOM-safe.

This file is a SUBCLASS / DROP-IN replacement — does NOT modify
train_cnn_mamba_v3_4.py or dispatch_v34_2_fixedmtl.py. The launcher
monkey-patches the dispatch module to use this class.

HC #386 patches (book_gate init = 0.5, head audit, confidence-band
dump) are preserved by the calling launcher — they are independent
of this dataset change.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from alpha_discovery.deep_models.train_cnn_mamba_v3_4 import (
    SmartV34DualTrunkDataset,
    DEFAULT_BOOK_FEATURES_DIR,
    WINDOW_SIZE_T2,
    N_BOOK_LEVELS,
    N_BOOK_FEATURES_PER_LEVEL,
    BOOK_BID_PRICE_IDX,
    BOOK_BID_SIZE_IDX,
    BOOK_ASK_PRICE_IDX,
    BOOK_ASK_SIZE_IDX,
)

logger = logging.getLogger("v342_memsafe")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_BOOK_PYRAMID_CACHE_DIR = os.environ.get(
    "V342_BOOK_PYRAMID_CACHE_DIR",
    "/home/nick/Lvl3Quant/data/processed/mbo_book_features_pyramid_cache",
)

# Cache format version — bump if preprocessing math changes.
CACHE_VERSION = 1


def _cache_paths(cache_dir: Path, date_str: str):
    base = cache_dir / f"{date_str}_pyramid_v{CACHE_VERSION}"
    return base.with_suffix(".bin"), base.with_suffix(".meta.json")


def _build_pyramid_for_date(npz_path: Path, bin_path: Path, meta_path: Path) -> dict:
    """Stream the NPZ → preprocessed pyramid → raw float32 file. Returns meta dict.

    Preprocessing replicated EXACTLY from SmartV34DualTrunkDataset.__init__ so
    runtime numeric behaviour is identical.
    """
    with np.load(npz_path) as d:
        arr = d["features"]  # (N, 30) float32 — already in RAM after this, but only 1 day
        bp_arr = arr[:, BOOK_BID_PRICE_IDX]
        bs_arr = arr[:, BOOK_BID_SIZE_IDX]
        ap_arr = arr[:, BOOK_ASK_PRICE_IDX]
        as_arr = arr[:, BOOK_ASK_SIZE_IDX]
        pyramid = np.stack([bp_arr, bs_arr, ap_arr, as_arr], axis=-1).astype(np.float32)
        ref_mid = (pyramid[:, 0, 0] + pyramid[:, 0, 2]) / 2.0
        pyramid[:, :, 0] = (pyramid[:, :, 0] - ref_mid[:, None]) / 0.25
        pyramid[:, :, 2] = (pyramid[:, :, 2] - ref_mid[:, None]) / 0.25
        pyramid[:, :, 1] = np.log1p(pyramid[:, :, 1])
        pyramid[:, :, 3] = np.log1p(pyramid[:, :, 3])

    tmp_path = bin_path.with_suffix(".bin.tmp")
    # Write raw float32 — no header. Shape stored in sidecar JSON.
    pyramid.tofile(str(tmp_path))
    os.replace(str(tmp_path), str(bin_path))
    meta = {
        "version": CACHE_VERSION,
        "shape": list(pyramid.shape),
        "dtype": "float32",
        "n_levels": N_BOOK_LEVELS,
        "n_features_per_level": N_BOOK_FEATURES_PER_LEVEL,
        "source_npz": str(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    # Free 1-day temp arrays before returning
    del pyramid, arr, bp_arr, bs_arr, ap_arr, as_arr, ref_mid
    return meta


class SmartV34MemmapDualTrunkDataset(SmartV34DualTrunkDataset):
    """Drop-in replacement for SmartV34DualTrunkDataset.

    Key differences from parent:
      * `self.book_data[date] = (memmap_array, shape)` instead of dense ndarray.
      * Per-date preprocessing materialised to disk ONCE (sidecar .bin + .meta.json)
        — the Python process never holds more than one day at a time during build.
      * `_book_window` slices the memmap and returns a small float32 copy of
        just the (window, 5, 4) region — no dense day arrays in RAM.

    Preprocessing math is COPIED verbatim from the parent so output is bit-identical.
    """
    def __init__(
        self,
        v32_inner,
        book_features_dir: str = DEFAULT_BOOK_FEATURES_DIR,
        window_size_book: int = WINDOW_SIZE_T2,
        log_size: bool = True,
        cache_dir: Optional[str] = None,
        dry_run_skip_train_init: bool = False,
    ):
        # Intentionally do NOT call super().__init__ — parent's init eagerly loads
        # everything into RAM, which is exactly what we are eliminating.
        self.inner = v32_inner
        self.window_size_book = window_size_book
        self.book_dir = Path(book_features_dir)
        self.cache_dir = Path(cache_dir or DEFAULT_BOOK_PYRAMID_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # date_str -> (np.memmap, (N, 5, 4))
        self.book_data: Dict[str, tuple] = {}
        loaded_dates = 0
        built_dates = 0
        missing_dates = []

        dates = list(getattr(self.inner, "dates", []))
        if log_size:
            logger.info(f"SmartV34MemmapDualTrunkDataset: building/opening pyramid cache "
                        f"for {len(dates)} dates in {self.cache_dir}")

        for date_str in dates:
            npz_path = self.book_dir / f"{date_str}_book_features.npz"
            if not npz_path.exists():
                missing_dates.append(date_str)
                continue
            bin_path, meta_path = _cache_paths(self.cache_dir, date_str)
            if not (bin_path.exists() and meta_path.exists()):
                try:
                    _build_pyramid_for_date(npz_path, bin_path, meta_path)
                    built_dates += 1
                except Exception as e:
                    logger.error(f"  pyramid build failed for {date_str}: {e}")
                    missing_dates.append(date_str)
                    continue
            try:
                meta = json.loads(meta_path.read_text())
                shape = tuple(meta["shape"])
                mm = np.memmap(str(bin_path), dtype=np.float32, mode="r", shape=shape)
                self.book_data[date_str] = (mm, shape)
                loaded_dates += 1
            except Exception as e:
                logger.error(f"  memmap open failed for {date_str}: {e}")
                missing_dates.append(date_str)

        if log_size:
            logger.info(f"SmartV34MemmapDualTrunkDataset: "
                        f"loaded={loaded_dates} built={built_dates} missing={len(missing_dates)}")
            if missing_dates:
                logger.warning(f"  missing book dates: {missing_dates[:10]}"
                               f"{' ...' if len(missing_dates) > 10 else ''}")

    def _book_window(self, date_str: str, row_idx: int) -> np.ndarray:
        """Returns (window_size_book, 5_levels, 4_features) np.float32 — small copy of mmap slice."""
        entry = self.book_data.get(date_str)
        if entry is None:
            return np.zeros((self.window_size_book, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL),
                            dtype=np.float32)
        mm, shape = entry
        end = row_idx + 1
        start = max(0, end - self.window_size_book)
        # np.array(...) forces a SMALL contiguous copy out of mmap — only the window
        # is pulled. This is critical: returning the memmap directly would force the
        # collate path to retain a reference to the mapping, defeating the eviction.
        win = np.array(mm[start:end], dtype=np.float32, copy=True)
        if win.shape[0] < self.window_size_book:
            pad = np.zeros((self.window_size_book - win.shape[0], N_BOOK_LEVELS,
                            N_BOOK_FEATURES_PER_LEVEL), dtype=np.float32)
            win = np.concatenate([pad, win], axis=0)
        return win


def estimate_rss_savings(dates_count: int) -> dict:
    """Quick back-of-envelope so launcher can log expected savings."""
    per_day_mb = 6_500_000 * N_BOOK_LEVELS * N_BOOK_FEATURES_PER_LEVEL * 4 / 1e6  # ~520 MB
    return {
        "dates": dates_count,
        "old_rss_book_gb": round(per_day_mb * dates_count / 1024, 2),
        "new_rss_book_gb": round(0.05, 2),  # negligible — just memmap headers
        "disk_cache_gb": round(per_day_mb * dates_count / 1024, 2),
    }


__all__ = [
    "SmartV34MemmapDualTrunkDataset",
    "DEFAULT_BOOK_PYRAMID_CACHE_DIR",
    "estimate_rss_savings",
]
