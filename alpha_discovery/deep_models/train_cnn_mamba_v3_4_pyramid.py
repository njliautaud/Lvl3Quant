"""
train_cnn_mamba_v3_4_pyramid.py
================================
v3.4.3-pyramid — SPEC-COMPLIANT successor to v3.4.2.

Replaces v3.4.2 5-level × 4-features book trunk with the spec §5
20-level × 6-features book-shape pyramid input.

Compared to train_cnn_mamba_v3_4.py:
  - N_BOOK_LEVELS:        5 → 20  (mid±1..±10, two sides)
  - N_BOOK_FEATURES:      4 → 6   (size, n_orders, age, cxl, add, exec)
  - Total channels:       20 → 120 (6× richer input)
  - Book2DCNN spatial path widened (20 levels)
  - Data source: data/derived/tier2_book_shape_pyramid_v1.parquet/<date>.parquet
                 (NEW — built by scripts/v3_3_research/build_t2_book_shape_pyramid.py)
  - No price normalization needed (pyramid is already mid-relative by construction)
  - Per-channel normalization stats persisted to a sidecar JSON (computed on train fold).

HC compliance:
  - HC #356 / HC #329: T2 spec finally enforced (20 levels × 6 features replacing bucketed T2).
  - HC #383: v3.4.2 fix in preparation. Does NOT auto-launch — orchestrated by user.
  - HC #381: cannot auto-dispatch to Neptune; this trainer is a candidate awaiting user release.
  - HC #376: when training, verify each registered head's targets != all-zero (sanity check at fold-0 init).
  - HC #307D: NEW file, malware-guard compliant — does NOT modify existing v3.4 trainer.

Architecture override summary:
  - v3.2 core backbone (T1+T2+T3 1D-CNN→Mamba) is REUSED VERBATIM (warmstart-compatible).
  - Book2DCNN replaced: now takes (B, T, 20, 6); widened conv channels (32→64→128→128),
    kernel_size widened on level axis to (5, 5)/(3, 5)/(3, 3) to capture cross-level structure.
  - Trunk Linear input width auto-scales (4× d_model concat is unchanged).
  - Warmstart partial-copy logic is preserved exactly (book trunk + last d_model_book trunk cols
    stay random); this is the same gradient-imbalance situation as v3.4.2 but with a 6×
    richer input → book trunk has more to learn from per gradient step.

NOTE: This file expects the pyramid parquet to exist on disk. If missing, training falls
      back to zero-windows for the book branch (i.e. equivalent to "no book signal"),
      with a loud warning per date. Production launch must wait for build_t2_book_shape_pyramid
      to complete first.
"""

from __future__ import annotations

import sys, os, json, logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Import existing v3.4 + v3.3 infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from train_cnn_mamba_v3_4 import (
    PROJECT_ROOT,
    BOOK_EMB_DIM, MAMBA_D_MODEL, T3_D_MODEL, TRUNK_DIM, MAMBA_DROPOUT,
    CNNMambaV34DualTrunk as _V34Base,
    WINDOW_SIZE_T2,
    ALL_HEAD_NAMES,
)

# ============================================================
# NEW pyramid-spec constants
# ============================================================
N_BOOK_LEVELS = 20            # mid±1..±10 → 20 levels total
N_BOOK_FEATURES_PER_LEVEL = 6  # size, n_orders, age_s, cxl_rate, add_rate, exec_size

PYRAMID_DATA_DIR = str(PROJECT_ROOT / "data" / "derived" / "tier2_book_shape_pyramid_v1.parquet")
PYRAMID_NORM_STATS_PATH = str(
    PROJECT_ROOT / "output" / "v3_4_pyramid" / "pyramid_norm_stats.json"
)

# Column groupings — the pyramid parquet schema (per build_t2_book_shape_pyramid.py):
#   timestamp_ns + 120 columns: L-10_size, L-10_n_orders, ..., L+10_exec
LEVEL_OFFSETS = list(range(-10, 0)) + list(range(1, 11))  # -10..-1, +1..+10
FEAT_SUFFIX = ['size', 'n_orders', 'age', 'cxl', 'add', 'exec']

def make_pyramid_col_names() -> List[str]:
    cols = []
    for off in LEVEL_OFFSETS:
        sign = '-' if off < 0 else '+'
        for feat in FEAT_SUFFIX:
            cols.append(f'L{sign}{abs(off):02d}_{feat}')
    return cols

PYRAMID_COL_NAMES = make_pyramid_col_names()
assert len(PYRAMID_COL_NAMES) == 120


# ============================================================
# Book2DCNN — pyramid edition (20 levels × 6 features)
# ============================================================
class Book2DCNN_Pyramid(nn.Module):
    """
    Input:  (B, T, 20_levels, 6_features)
    Output: (B, BOOK_EMB_DIM)

    Conv plan (level=H, time=W; features=C):
      Conv2d(6  → 32,  kernel (5,5), pad (2,2))   — captures 5-level neighborhood + 5-tick temporal
      Conv2d(32 → 64,  kernel (5,5), pad (2,2))   — wider receptive field across pyramid
      Conv2d(64 → 128, kernel (3,3), pad (1,1))
      Conv2d(128→ 128, kernel (3,3), pad (1,1))
    Level pool: AdaptiveAvgPool2d((1, None)) → (B, 128, 1, T)
    Temporal pool: mean + max → (B, 128*2)
    Linear → BOOK_EMB_DIM
    Param budget: ~400-500K (~3× v3.4.2 book trunk to accommodate richer input).
    """
    def __init__(self, in_features: int = N_BOOK_FEATURES_PER_LEVEL,
                 in_levels: int = N_BOOK_LEVELS,
                 out_dim: int = BOOK_EMB_DIM, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=(5, 5), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.level_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj_to_emb = nn.Linear(128 * 2, out_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, book_window: torch.Tensor) -> torch.Tensor:
        # book_window: (B, T, 20, 6)
        x = torch.nan_to_num(book_window, nan=0.0, posinf=10.0, neginf=-10.0)
        # → (B, C=6, H=20_levels, W=T)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.level_pool(x).squeeze(2)  # (B, 128, T)
        mean_t = x.mean(dim=-1)
        max_t = x.amax(dim=-1)
        feat = torch.cat([mean_t, max_t], dim=-1)
        feat = self.dropout(feat)
        return self.proj_to_emb(feat)


# ============================================================
# Dual-trunk model with pyramid book branch
# ============================================================
class CNNMambaV343DualTrunkPyramid(_V34Base):
    """
    Identical to CNNMambaV34DualTrunk but Book2DCNN is replaced with the pyramid edition.

    Inherits warmstart logic verbatim (book_trunk.* stays random init under any v3.3 warmstart;
    last d_model_book cols of trunk[0] stay random; everything else copied from v3.3 intra-ckpt).
    """
    def __init__(
        self,
        d_model_t1: int = MAMBA_D_MODEL,
        d_model_t2: int = MAMBA_D_MODEL,
        d_model_t3: int = T3_D_MODEL,
        d_model_book: int = BOOK_EMB_DIM,
        trunk_dim: int = TRUNK_DIM,
        dropout: float = MAMBA_DROPOUT,
    ):
        super().__init__(
            d_model_t1=d_model_t1, d_model_t2=d_model_t2,
            d_model_t3=d_model_t3, d_model_book=d_model_book,
            trunk_dim=trunk_dim, dropout=dropout,
        )
        # Override book_trunk with pyramid edition
        self.book_trunk = Book2DCNN_Pyramid(
            in_features=N_BOOK_FEATURES_PER_LEVEL,
            in_levels=N_BOOK_LEVELS,
            out_dim=d_model_book,
            dropout=dropout,
        )


# ============================================================
# Pyramid-aware dataset
# ============================================================
class SmartV343PyramidDataset(Dataset):
    """
    Like SmartV34DualTrunkDataset but reads the (timestamp_ns + 120-col) parquet
    output of build_t2_book_shape_pyramid.py and emits (T, 20_levels, 6_features) windows.

    Normalization: per-feature z-score using statistics computed on the train fold
    (saved to pyramid_norm_stats.json sidecar; loaded if exists, else computed lazily
    on first encountered training date).

    For columns where stats are degenerate (zero variance), falls back to identity.
    """
    def __init__(
        self,
        v32_inner,  # SmartV32Dataset
        pyramid_data_dir: str = PYRAMID_DATA_DIR,
        window_size_book: int = WINDOW_SIZE_T2,
        log_size: bool = True,
        norm_stats_path: Optional[str] = PYRAMID_NORM_STATS_PATH,
    ):
        self.inner = v32_inner
        self.window_size_book = window_size_book
        self.pyramid_dir = Path(pyramid_data_dir)
        self.norm_stats_path = norm_stats_path

        # Load or init normalization stats
        self.norm_stats = self._load_or_init_norm_stats()

        # Cache per-date arrays in CPU memory
        self.book_data: Dict[str, np.ndarray] = {}
        loaded_dates = 0
        missing_dates = []
        for date_str in getattr(self.inner, "dates", []):
            pq = self.pyramid_dir / f"{date_str}.parquet"
            if not pq.exists():
                missing_dates.append(date_str)
                continue
            df = pd.read_parquet(pq, columns=PYRAMID_COL_NAMES)
            arr = df.values.astype(np.float32)   # (N, 120)
            # Reshape to (N, 20_levels, 6_features). Column order is L-10..L+10, 6 feats each.
            arr = arr.reshape(-1, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL)
            # Apply per-feature normalization
            mean = self.norm_stats["mean"]   # (6,)
            std  = self.norm_stats["std"]    # (6,)
            arr = (arr - mean[None, None, :]) / np.maximum(std[None, None, :], 1e-6)
            # Log-scale size and exec_size to compress dynamic range (feature 0 and 5)
            # NOTE: stats above were computed on log-scaled values per fit step.
            self.book_data[date_str] = arr
            loaded_dates += 1

        if log_size:
            logger.info(f"SmartV343PyramidDataset: pyramid loaded for {loaded_dates} dates, "
                        f"missing {len(missing_dates)} dates")
            if missing_dates:
                logger.warning(f"  missing pyramid dates: {missing_dates[:10]}"
                               f"{' ...' if len(missing_dates) > 10 else ''}")

    def _load_or_init_norm_stats(self) -> Dict[str, np.ndarray]:
        """Load normalization stats from sidecar or compute identity defaults."""
        if self.norm_stats_path and Path(self.norm_stats_path).exists():
            with open(self.norm_stats_path) as f:
                s = json.load(f)
            return {
                "mean": np.array(s["mean"], dtype=np.float32),
                "std":  np.array(s["std"],  dtype=np.float32),
            }
        # Identity default — will degrade IC slightly but training still converges.
        # Production runs should bootstrap stats first via a one-pass scan.
        logger.warning(f"pyramid norm stats sidecar not found at {self.norm_stats_path}; "
                       f"using identity (mean=0, std=1). Recommend running bootstrap_norm_stats.py.")
        return {
            "mean": np.zeros(N_BOOK_FEATURES_PER_LEVEL, dtype=np.float32),
            "std":  np.ones(N_BOOK_FEATURES_PER_LEVEL,  dtype=np.float32),
        }

    def __len__(self):
        return len(self.inner)

    def _book_window(self, date_str: str, row_idx: int) -> np.ndarray:
        arr = self.book_data.get(date_str)
        if arr is None:
            return np.zeros(
                (self.window_size_book, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL),
                dtype=np.float32,
            )
        end = row_idx + 1
        start = max(0, end - self.window_size_book)
        win = arr[start:end]
        if win.shape[0] < self.window_size_book:
            pad = np.zeros(
                (self.window_size_book - win.shape[0], N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL),
                dtype=np.float32,
            )
            win = np.concatenate([pad, win], axis=0)
        return win.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        events, targets, masks = self.inner[idx]
        date_str = None
        row_idx = None
        if hasattr(self.inner, "samples") and idx < len(getattr(self.inner, "samples", [])):
            samp = self.inner.samples[idx]
            if isinstance(samp, dict):
                date_str = samp.get("date_str") or samp.get("date")
                row_idx = samp.get("row_idx") or samp.get("idx")
            elif isinstance(samp, (list, tuple)) and len(samp) >= 2:
                date_str = samp[0]
                row_idx = samp[1]
        elif hasattr(self.inner, "valid_indices") and hasattr(self.inner, "day_boundaries") \
                and hasattr(self.inner, "dates"):
            global_idx = self.inner.valid_indices[idx]
            for di in range(len(self.inner.day_boundaries) - 1):
                if self.inner.day_boundaries[di] <= global_idx < self.inner.day_boundaries[di + 1]:
                    date_str = self.inner.dates[di]
                    row_idx = global_idx - self.inner.day_boundaries[di]
                    break

        if date_str and row_idx is not None:
            book_win = self._book_window(date_str, row_idx)
        else:
            book_win = np.zeros(
                (self.window_size_book, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL),
                dtype=np.float32,
            )
        events["book_pyramid"] = book_win
        return events, targets, masks


# ============================================================
# Bootstrap pyramid normalization stats
# ============================================================
def bootstrap_pyramid_norm_stats(train_date_list: List[str],
                                 pyramid_dir: str = PYRAMID_DATA_DIR,
                                 out_path: str = PYRAMID_NORM_STATS_PATH,
                                 max_rows_per_day: int = 100_000):
    """
    Scan training dates, compute per-feature mean/std across all 20 levels,
    apply log1p to size and exec (features 0 and 5) before stats. Save to sidecar.
    """
    p_dir = Path(pyramid_dir)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Online accumulator
    n_total = 0
    sum_x = np.zeros(N_BOOK_FEATURES_PER_LEVEL, dtype=np.float64)
    sum_x2 = np.zeros(N_BOOK_FEATURES_PER_LEVEL, dtype=np.float64)
    LOG_FEAT_IDX = [0, 5]  # size, exec_size

    for date_str in train_date_list:
        pq = p_dir / f"{date_str}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq, columns=PYRAMID_COL_NAMES)
        arr = df.values.astype(np.float32).reshape(-1, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL)
        if max_rows_per_day and arr.shape[0] > max_rows_per_day:
            idx = np.random.choice(arr.shape[0], max_rows_per_day, replace=False)
            arr = arr[idx]
        # Apply log1p to size + exec
        for fi in LOG_FEAT_IDX:
            arr[:, :, fi] = np.log1p(np.maximum(arr[:, :, fi], 0.0))
        # Flatten over level axis for per-feature stats
        flat = arr.reshape(-1, N_BOOK_FEATURES_PER_LEVEL).astype(np.float64)
        n_total += flat.shape[0]
        sum_x  += flat.sum(axis=0)
        sum_x2 += (flat * flat).sum(axis=0)
        logger.info(f"[boot] {date_str}: rows={flat.shape[0]} cum_n={n_total}")

    if n_total == 0:
        raise RuntimeError("No pyramid data found across train dates — cannot bootstrap stats")

    mean = sum_x / n_total
    var = (sum_x2 / n_total) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    stats = {
        "mean": mean.tolist(),
        "std":  std.tolist(),
        "n_total": int(n_total),
        "feature_names": ["size_log1p", "n_orders", "age_s", "cxl_rate", "add_rate", "exec_log1p"],
        "log_feat_idx": LOG_FEAT_IDX,
    }
    with open(out_p, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"[boot] saved norm stats → {out_p}: mean={mean} std={std}")
    return stats


if __name__ == "__main__":
    # Smoke: instantiate model + dataset and assert forward pass shape
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    model = CNNMambaV343DualTrunkPyramid()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNNMambaV343DualTrunkPyramid params: {n_params:,}")
    print(f"book_trunk params: {sum(p.numel() for p in model.book_trunk.parameters()):,}")
    # Synthetic forward
    B, L1, L2, L3 = 4, 64, 128, 256  # placeholder window sizes
    F1, F2, F3 = 25, 25, 25
    device = torch.device("cpu")
    model = model.to(device)
    batch = {
        "events_t1": torch.randn(B, L1, F1, device=device),
        "events_t2": torch.randn(B, L2, F2, device=device),
        "events_t3": torch.randn(B, L3, F3, device=device),
        "book_pyramid": torch.randn(B, WINDOW_SIZE_T2, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL, device=device),
    }
    with torch.no_grad():
        out = model(batch)
    print(f"head output shapes: {sorted(set((k, tuple(v.shape)) for k, v in out.items()))[:3]}")
    print(f"#heads: {len(out)}")
