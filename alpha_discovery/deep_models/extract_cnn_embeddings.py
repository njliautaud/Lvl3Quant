#!/usr/bin/env python3
"""
EXP-TS-01: CNN Embedding Extraction for LSTM Temporal Model
============================================================
Loads fold_74 Wider BookSpatialCNN checkpoint (spatial=(64,128,256,512), temporal=512)
and extracts 512-dim embeddings from temporal_pool for all 173 days of bar data.

Output per day: {date}_embeddings.npz with keys:
  - embeddings: (N_valid_bars, 512) float32 — 512-dim temporal_pool vectors
  - predictions: (N_valid_bars,) float32  — scalar z-score predictions from head
  - targets: (N_valid_bars,) float32      — mfe_net targets (100-bar horizon, 0.25 tick)

Output dir: C:/Users/Footb/Documents/Github/Lvl3Quant/data/processed/cnn_embeddings_fold74/
"""

import sys
import os
import gc
import time
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime

# BELOW_NORMAL process priority (Windows)
try:
    import ctypes
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        BELOW_NORMAL_PRIORITY_CLASS
    )
    print("Process priority set to BELOW_NORMAL", flush=True)
except Exception as e:
    print(f"Could not set priority: {e}", flush=True)

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from book_spatial_cnn import BookSpatialCNN

CHECKPOINT = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn' / 'checkpoints' / 'fold_74_2025-11-03.pt'
DATA_DIR   = ROOT / 'data' / 'processed' / 'dl_book_cache'
OUT_DIR    = ROOT / 'data' / 'processed' / 'cnn_embeddings_fold74'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW_SIZE = 20
BATCH_SIZE  = 512
CHUNK_SIZE  = 2000
HORIZON     = 100   # mfe_net target horizon in bars
TICK_SIZE   = 0.25  # ES tick size

# ── Wider CNN architecture ────────────────────────────────────────────────────
class WiderBookSpatialCNN(BookSpatialCNN):
    """2x wider BookSpatialCNN — matches fold_74 training config."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels'] = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout'] = 0.15
        super().__init__(**kwargs)


# ── Embedding hook ────────────────────────────────────────────────────────────
_captured_embeddings = []

def _hook_fn(module, input, output):
    """Forward hook on temporal_pool: captures the squeezed (B, 512) tensor."""
    # output is (B, 512, 1) from AdaptiveAvgPool1d(1)
    _captured_embeddings.append(output.squeeze(-1).float().cpu())


def compute_mfe_net_targets(mid_prices: np.ndarray, horizon: int = 100, tick_size: float = 0.25) -> np.ndarray:
    """
    MFE-net target: maximum favorable excursion in ticks over [1, horizon] bars,
    signed by direction of price at horizon.

    Vectorized: uses stride tricks to build (n, horizon) future-price matrix.
    Returns float32 array of length n. Last `horizon` bars get target=0.
    """
    n = len(mid_prices)
    targets = np.zeros(n, dtype=np.float32)
    valid_n = n - horizon  # bars with full horizon
    if valid_n <= 0:
        return targets

    mid = mid_prices.astype(np.float64)

    # Build (valid_n, horizon) matrix of future prices using stride tricks
    # future_prices[i, k] = mid[i + 1 + k] for k in [0, horizon-1]
    idx = np.arange(valid_n)[:, None] + np.arange(1, horizon + 1)[None, :]  # (valid_n, horizon)
    future_prices = mid[idx]  # (valid_n, horizon)

    origin = mid[:valid_n, None]  # (valid_n, 1)
    delta_at_h = (future_prices[:, -1] - mid[:valid_n]) / tick_size  # (valid_n,)
    mfe = np.max(np.abs(future_prices - origin), axis=1) / tick_size  # (valid_n,)

    sign = np.sign(delta_at_h)
    sign[sign == 0] = 0.0
    targets[:valid_n] = (mfe * sign).astype(np.float32)
    return targets


def process_day(model, hook_handle, device, bt, mid):
    """
    Run inference on one day's data. Returns (embeddings, predictions, targets)
    aligned to valid bar indices [WINDOW_SIZE-1 : n-HORIZON].

    bt:  (n, 20, 4) float32 — log-transformed book tensors
    mid: (n,)       float64 — mid prices
    """
    n = len(bt)
    # Full-array predictions and embedding accumulators
    preds = np.zeros(n, dtype=np.float32)
    # We collect embeddings per batch, then concatenate
    all_embs = []   # list of (batch, 512) arrays

    # We only need windows from [WINDOW_SIZE-1 .. n-HORIZON-1] for valid targets
    # But we run full inference for all valid windows, then slice
    valid_start = WINDOW_SIZE - 1    # first bar with a full window
    valid_end   = n - HORIZON        # last bar with a full target (exclusive → n - HORIZON)
    if valid_end <= valid_start:
        return None, None, None

    model.eval()
    with torch.no_grad():
        for cstart in range(0, n, CHUNK_SIZE):
            cend = min(cstart + CHUNK_SIZE, n)
            slice_start = max(0, cstart - WINDOW_SIZE + 1)
            bt_chunk = bt[slice_start:cend]
            bt_gpu   = torch.from_numpy(bt_chunk).to(device)

            if len(bt_gpu) < WINDOW_SIZE:
                del bt_gpu
                continue

            # (n_windows, WINDOW_SIZE, 20, 4)
            windowed = bt_gpu.unfold(0, WINDOW_SIZE, 1).permute(0, 3, 1, 2).contiguous()
            pred_offset = slice_start + WINDOW_SIZE - 1

            _captured_embeddings.clear()

            for bstart in range(0, len(windowed), BATCH_SIZE):
                bend = min(bstart + BATCH_SIZE, len(windowed))
                batch = windowed[bstart:bend]

                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model(batch).squeeze(-1)  # (B,)
                else:
                    out = model(batch).squeeze(-1)

                g_start = pred_offset + bstart
                g_end   = pred_offset + bend
                preds[g_start:g_end] = out.cpu().float().numpy()

            # _captured_embeddings now has one tensor per batch in this chunk
            if _captured_embeddings:
                chunk_embs = torch.cat(_captured_embeddings, dim=0).numpy()  # (n_windows_in_chunk, 512)
                all_embs.append((pred_offset, chunk_embs))
                _captured_embeddings.clear()

            del bt_gpu, windowed
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Assemble full embedding array (size n, aligned by pred_offset)
    emb_full = np.zeros((n, 512), dtype=np.float32)
    for (offset, emb_arr) in all_embs:
        end_idx = offset + len(emb_arr)
        if end_idx > n:
            emb_arr = emb_arr[:n - offset]
            end_idx = n
        emb_full[offset:end_idx] = emb_arr

    # Compute targets
    targets = compute_mfe_net_targets(mid, HORIZON, TICK_SIZE)

    # Slice to valid range
    emb_out  = emb_full[valid_start:valid_end]
    pred_out = preds[valid_start:valid_end]
    tgt_out  = targets[valid_start:valid_end]

    return emb_out, pred_out, tgt_out


def main():
    print("=" * 70, flush=True)
    print("EXP-TS-01: CNN Embedding Extraction (fold_74 Wider BookSpatialCNN)", flush=True)
    print(f"  Checkpoint : {CHECKPOINT}", flush=True)
    print(f"  Data dir   : {DATA_DIR}", flush=True)
    print(f"  Output dir : {OUT_DIR}", flush=True)
    print(f"  Window={WINDOW_SIZE}, Batch={BATCH_SIZE}, Chunk={CHUNK_SIZE}", flush=True)
    print(f"  Target horizon={HORIZON} bars, tick={TICK_SIZE}", flush=True)
    print("=" * 70, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = WiderBookSpatialCNN(
        window_size=WINDOW_SIZE,
        num_levels=20,
        num_features=4,
        num_classes=1,
    ).to(device)

    state = torch.load(str(CHECKPOINT), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} params", flush=True)

    # ── Register hook on temporal_pool ────────────────────────────────────────
    hook_handle = model.temporal_pool.register_forward_hook(_hook_fn)
    print("Forward hook registered on temporal_pool", flush=True)

    # ── Discover files ────────────────────────────────────────────────────────
    files = sorted(DATA_DIR.glob('*_book_tensors.npz'))
    print(f"Found {len(files)} days in {DATA_DIR}", flush=True)

    # Find already-completed days (skip re-processing)
    completed = {f.name.replace('_embeddings.npz', '') for f in OUT_DIR.glob('*_embeddings.npz')}
    todo = [f for f in files if f.name.replace('_book_tensors.npz', '') not in completed]
    if completed:
        print(f"  {len(completed)} days already done, {len(todo)} remaining", flush=True)

    t0 = time.time()
    n_done = 0
    n_skipped = 0

    for fi, f in enumerate(todo):
        # RAM safety check
        ram_gb = psutil.virtual_memory().available / 1e9
        if ram_gb < 3.0:
            print(f"ABORT: Only {ram_gb:.1f} GB RAM free. Stopping safely.", flush=True)
            break

        date = f.name.replace('_book_tensors.npz', '')
        out_path = OUT_DIR / f'{date}_embeddings.npz'

        td = time.time()

        # Load and log-transform
        npz = np.load(str(f))
        bt  = npz['book_tensors'].astype(np.float32)  # (n, 20, 4)
        mid = npz['mid_prices'].astype(np.float64)     # (n,)
        npz.close()

        np.log1p(bt[:, :, 1], out=bt[:, :, 1])
        np.log1p(bt[:, :, 2], out=bt[:, :, 2])
        np.log1p(bt[:, :, 3], out=bt[:, :, 3])

        n_bars = len(bt)

        # Process
        emb, pred, tgt = process_day(model, hook_handle, device, bt, mid)

        if emb is None:
            print(f"[{fi+1}/{len(todo)}] {date}: SKIP (too few bars: {n_bars})", flush=True)
            n_skipped += 1
            del bt, mid
            gc.collect()
            continue

        # Save
        np.savez_compressed(str(out_path),
                            embeddings=emb,    # (N_valid, 512)
                            predictions=pred,  # (N_valid,)
                            targets=tgt)       # (N_valid,)

        day_time = time.time() - td
        total_time = time.time() - t0
        ram_gb = psutil.virtual_memory().available / 1e9
        n_done += 1

        print(f"[{fi+1}/{len(todo)}] {date}: {n_bars:,} bars -> {len(emb):,} valid "
              f"emb={emb.shape} pred=[{pred.min():.3f},{pred.max():.3f}] "
              f"tgt=[{tgt.min():.2f},{tgt.max():.2f}] "
              f"{day_time:.1f}s RAM:{ram_gb:.0f}GB (total {total_time:.0f}s)",
              flush=True)

        del bt, mid, emb, pred, tgt
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    hook_handle.remove()
    total_time = time.time() - t0
    completed_final = list(OUT_DIR.glob('*_embeddings.npz'))
    print("=" * 70, flush=True)
    print(f"DONE: {n_done} days processed, {n_skipped} skipped in {total_time:.0f}s", flush=True)
    print(f"Total files in output dir: {len(completed_final)}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)
    print("EXP-TS-01 COMPLETE — ready for LSTM training (EXP-TS-002)", flush=True)


if __name__ == '__main__':
    main()
