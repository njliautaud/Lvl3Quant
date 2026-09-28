#!/usr/bin/env python3
"""
EXP-TS-02: Leak-Free CNN Embedding Extraction for LSTM Temporal Model
======================================================================
For each WF fold's test date, loads the PREVIOUS fold's checkpoint weights.
This guarantees the CNN has never seen that date — fully OOT embeddings.

Leak-free guarantee:
  Fold N test date D  →  load fold_(N-1) checkpoint
  The CNN in fold_(N-1) was trained on days BEFORE D, never on D.

Available checkpoints (main dir, clean):
  fold_72_2025-10-29.pt  →  embeds fold_73 test date (2025-10-30)
  fold_73_2025-10-30.pt  →  embeds fold_74 test date (2025-10-31)
  fold_74_2025-11-03.pt  →  embeds fold_75 test date (2025-11-03)  [NOTE: fold_75 test = 2025-11-03]
  fold_75_2025-11-04.pt  →  embeds fold_76 test date (2025-11-04)

For folds 37-72 (test dates 2025-09-10 to 2025-10-29):
  No per-fold checkpoints exist. fold_72 (train_days=76) has seen all those
  dates as training data — we CANNOT use it for those folds without leakage.
  Strategy: SKIP folds 37-72 and report clearly. The fold_74 OOT extraction
  already covers dates 2025-11-05+ (days 87-173). The LSTM will be trained on
  the 4 leak-free WF dates + the fold_74 OOT dates.

Output dir: data/processed/cnn_embeddings_wf/   (separate from cnn_embeddings_fold74/)
Output per day: {date}_embeddings.npz
  embeddings:   (N_valid, 512) float32
  predictions:  (N_valid,)     float32
  targets:      (N_valid,)     float32
  fold:         scalar int     — which fold this date belongs to
  checkpoint_fold: scalar int  — which fold's weights were used

Architecture: BookSpatialCNN spatial=(64,128,256,512), temporal=512 (Wider)
"""

import sys
import os
import gc
import json
import time
import glob
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime

# ── BELOW_NORMAL process priority (Windows) ───────────────────────────────────
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
SCRIPT_DIR   = Path(__file__).resolve().parent
ROOT         = SCRIPT_DIR.parent.parent
CKPT_DIR     = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn' / 'checkpoints'
CHECKPOINT_JSON = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn' / 'checkpoint_book_20260323_231723.json'
DATA_DIR     = ROOT / 'data' / 'processed' / 'dl_book_cache'
OUT_DIR      = ROOT / 'data' / 'processed' / 'cnn_embeddings_wf'

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from book_spatial_cnn import BookSpatialCNN

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW_SIZE = 20
BATCH_SIZE  = 512
CHUNK_SIZE  = 2000
HORIZON     = 100
TICK_SIZE   = 0.25

# ── Wider CNN architecture ────────────────────────────────────────────────────
class WiderBookSpatialCNN(BookSpatialCNN):
    """2x wider BookSpatialCNN — matches WF training config."""
    def __init__(self, **kwargs):
        kwargs['spatial_channels']  = (64, 128, 256, 512)
        kwargs['temporal_channels'] = 512
        kwargs['dropout']           = 0.15
        super().__init__(**kwargs)


# ── Embedding hook ────────────────────────────────────────────────────────────
_captured_embeddings = []

def _hook_fn(module, input, output):
    """Forward hook on temporal_pool: captures (B, 512) from (B, 512, 1)."""
    _captured_embeddings.append(output.squeeze(-1).float().cpu())


# ── Target computation ────────────────────────────────────────────────────────
def compute_mfe_net_targets(mid_prices: np.ndarray, horizon: int = 100, tick_size: float = 0.25) -> np.ndarray:
    """
    MFE-net target: maximum favorable excursion in ticks over [1, horizon] bars,
    signed by direction of price at horizon.
    Last `horizon` bars get target=0.
    """
    n = len(mid_prices)
    targets  = np.zeros(n, dtype=np.float32)
    valid_n  = n - horizon
    if valid_n <= 0:
        return targets

    mid = mid_prices.astype(np.float64)
    idx          = np.arange(valid_n)[:, None] + np.arange(1, horizon + 1)[None, :]
    future_prices = mid[idx]
    origin        = mid[:valid_n, None]
    delta_at_h    = (future_prices[:, -1] - mid[:valid_n]) / tick_size
    mfe           = np.max(np.abs(future_prices - origin), axis=1) / tick_size
    sign          = np.sign(delta_at_h)
    sign[sign == 0] = 0.0
    targets[:valid_n] = (mfe * sign).astype(np.float32)
    return targets


# ── Per-day inference ─────────────────────────────────────────────────────────
def process_day(model, device, bt, mid):
    """
    Run inference on one day's data.
    Returns (embeddings, predictions, targets) for valid bar range,
    or (None, None, None) if insufficient data.
    """
    n = len(bt)
    preds    = np.zeros(n, dtype=np.float32)
    all_embs = []

    valid_start = WINDOW_SIZE - 1
    valid_end   = n - HORIZON
    if valid_end <= valid_start:
        return None, None, None

    model.eval()
    with torch.no_grad():
        for cstart in range(0, n, CHUNK_SIZE):
            cend        = min(cstart + CHUNK_SIZE, n)
            slice_start = max(0, cstart - WINDOW_SIZE + 1)
            bt_chunk    = bt[slice_start:cend]
            bt_gpu      = torch.from_numpy(bt_chunk).to(device)

            if len(bt_gpu) < WINDOW_SIZE:
                del bt_gpu
                continue

            # (n_windows, WINDOW_SIZE, 20, 4) → permute to (n_windows, 4, WINDOW_SIZE, 20)
            windowed    = bt_gpu.unfold(0, WINDOW_SIZE, 1).permute(0, 3, 1, 2).contiguous()
            pred_offset = slice_start + WINDOW_SIZE - 1

            _captured_embeddings.clear()

            for bstart in range(0, len(windowed), BATCH_SIZE):
                bend  = min(bstart + BATCH_SIZE, len(windowed))
                batch = windowed[bstart:bend]

                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model(batch).squeeze(-1)
                else:
                    out = model(batch).squeeze(-1)

                g_start = pred_offset + bstart
                g_end   = pred_offset + bend
                preds[g_start:g_end] = out.cpu().float().numpy()

            if _captured_embeddings:
                chunk_embs = torch.cat(_captured_embeddings, dim=0).numpy()
                all_embs.append((pred_offset, chunk_embs))
                _captured_embeddings.clear()

            del bt_gpu, windowed
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Assemble full embedding array aligned by pred_offset
    emb_full = np.zeros((n, 512), dtype=np.float32)
    for (offset, emb_arr) in all_embs:
        end_idx = offset + len(emb_arr)
        if end_idx > n:
            emb_arr = emb_arr[:n - offset]
            end_idx = n
        emb_full[offset:end_idx] = emb_arr

    targets  = compute_mfe_net_targets(mid, HORIZON, TICK_SIZE)
    emb_out  = emb_full[valid_start:valid_end]
    pred_out = preds[valid_start:valid_end]
    tgt_out  = targets[valid_start:valid_end]

    return emb_out, pred_out, tgt_out


# ── Checkpoint discovery ───────────────────────────────────────────────────────
def find_checkpoint(fold_num: int) -> Path | None:
    """
    Find the checkpoint file for a given fold number.
    Pattern: fold_{fold_num}_YYYY-MM-DD.pt in CKPT_DIR (main, clean checkpoints only).
    Returns Path or None if not found.
    """
    matches = list(CKPT_DIR.glob(f'fold_{fold_num}_*.pt'))
    if matches:
        return matches[0]
    return None


# ── Build the fold→(test_date, prior_checkpoint) map ─────────────────────────
def build_work_plan(fold_details: list) -> list:
    """
    For each fold N (37-76), determine:
      - test_date
      - which checkpoint to use: fold N-1's weights (the PREVIOUS fold)
      - whether that checkpoint file exists

    Returns list of dicts with keys:
      fold, test_date, ckpt_fold, ckpt_path, skipped, skip_reason
    """
    plan = []
    for entry in fold_details:
        fold_n    = entry['fold']
        test_date = entry['test_date']
        ckpt_fold = fold_n - 1
        ckpt_path = find_checkpoint(ckpt_fold)

        if ckpt_path is None:
            plan.append({
                'fold':        fold_n,
                'test_date':   test_date,
                'ckpt_fold':   ckpt_fold,
                'ckpt_path':   None,
                'skipped':     True,
                'skip_reason': f'fold_{ckpt_fold} checkpoint not found (only folds 72-75 saved)',
            })
        else:
            plan.append({
                'fold':        fold_n,
                'test_date':   test_date,
                'ckpt_fold':   ckpt_fold,
                'ckpt_path':   ckpt_path,
                'skipped':     False,
                'skip_reason': '',
            })
    return plan


def main():
    print("=" * 75, flush=True)
    print("EXP-TS-02: Leak-Free CNN Embedding Extraction (Per-Fold WF)", flush=True)
    print(f"  Checkpoint dir : {CKPT_DIR}", flush=True)
    print(f"  Data dir       : {DATA_DIR}", flush=True)
    print(f"  Output dir     : {OUT_DIR}", flush=True)
    print(f"  Window={WINDOW_SIZE}, Batch={BATCH_SIZE}, Chunk={CHUNK_SIZE}", flush=True)
    print(f"  Target horizon={HORIZON} bars, tick_size={TICK_SIZE}", flush=True)
    print("=" * 75, flush=True)

    # ── Load fold details from checkpoint JSON ─────────────────────────────────
    with open(CHECKPOINT_JSON, 'r') as f:
        ckpt_data = json.load(f)
    fold_details = ckpt_data['fold_details']
    print(f"Loaded {len(fold_details)} folds from checkpoint JSON "
          f"(folds {fold_details[0]['fold']} to {fold_details[-1]['fold']})", flush=True)

    # ── Build work plan ────────────────────────────────────────────────────────
    plan = build_work_plan(fold_details)

    runnable = [p for p in plan if not p['skipped']]
    skipped  = [p for p in plan if p['skipped']]

    print(f"\nWork plan summary:", flush=True)
    print(f"  Runnable (checkpoint available): {len(runnable)}", flush=True)
    print(f"  Skipped  (no checkpoint):        {len(skipped)}", flush=True)
    if skipped:
        print(f"  Skipped folds: {[p['fold'] for p in skipped]}", flush=True)
        print(f"  Skip reason: {skipped[0]['skip_reason']}", flush=True)

    print(f"\nRunnable folds:", flush=True)
    for p in runnable:
        print(f"  Fold {p['fold']} test={p['test_date']} "
              f"<- fold_{p['ckpt_fold']} weights ({p['ckpt_path'].name})", flush=True)

    if not runnable:
        print("\nNO runnable folds. Exiting.", flush=True)
        return

    # ── Device setup ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    # ── Find already-completed dates (resume support) ──────────────────────────
    completed_dates = {
        f.name.replace('_embeddings.npz', '')
        for f in OUT_DIR.glob('*_embeddings.npz')
    }
    if completed_dates:
        print(f"\nResume: {len(completed_dates)} dates already in output dir, will skip.", flush=True)

    # ── Main extraction loop ───────────────────────────────────────────────────
    t0          = time.time()
    n_done      = 0
    n_already   = 0
    n_data_miss = 0

    # We keep the model in memory and reload weights per fold to avoid
    # repeated model instantiation overhead.
    print("\nInstantiating model architecture...", flush=True)
    model = WiderBookSpatialCNN(
        window_size=WINDOW_SIZE,
        num_levels=20,
        num_features=4,
        num_classes=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params", flush=True)

    # Register forward hook on temporal_pool (stays for all iterations)
    hook_handle = model.temporal_pool.register_forward_hook(_hook_fn)
    print("Forward hook registered on temporal_pool", flush=True)

    current_ckpt_fold = None  # track which fold's weights are currently loaded

    for item in runnable:
        fold_n    = item['fold']
        test_date = item['test_date']
        ckpt_fold = item['ckpt_fold']
        ckpt_path = item['ckpt_path']

        # ── Resume check ─────────────────────────────────────────────────────
        out_path = OUT_DIR / f'{test_date}_embeddings.npz'
        if test_date in completed_dates:
            print(f"[FOLD {fold_n}] {test_date}: already done, skipping", flush=True)
            n_already += 1
            continue

        # ── RAM safety ────────────────────────────────────────────────────────
        ram_gb = psutil.virtual_memory().available / 1e9
        if ram_gb < 3.0:
            print(f"ABORT: Only {ram_gb:.1f} GB RAM free. Stopping safely.", flush=True)
            break

        # ── Load checkpoint (only if different from what's already loaded) ────
        if current_ckpt_fold != ckpt_fold:
            print(f"\n[FOLD {fold_n}] Loading checkpoint: {ckpt_path.name}", flush=True)
            state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            current_ckpt_fold = ckpt_fold
            print(f"  Checkpoint fold_{ckpt_fold} loaded.", flush=True)
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # ── Load bar tensor data for test_date ────────────────────────────────
        data_file = DATA_DIR / f'{test_date}_book_tensors.npz'
        if not data_file.exists():
            print(f"[FOLD {fold_n}] {test_date}: DATA MISSING ({data_file.name}), skipping", flush=True)
            n_data_miss += 1
            continue

        td = time.time()
        npz = np.load(str(data_file))
        bt  = npz['book_tensors'].astype(np.float32)   # (n, 20, 4)
        mid = npz['mid_prices'].astype(np.float64)      # (n,)
        npz.close()
        n_bars = len(bt)

        # Log-transform bid/ask size columns (columns 1, 2, 3)
        np.log1p(bt[:, :, 1], out=bt[:, :, 1])
        np.log1p(bt[:, :, 2], out=bt[:, :, 2])
        np.log1p(bt[:, :, 3], out=bt[:, :, 3])

        # ── Run inference ─────────────────────────────────────────────────────
        print(f"[FOLD {fold_n}] {test_date}: {n_bars:,} bars | "
              f"checkpoint=fold_{ckpt_fold} | running inference...", flush=True)

        emb, pred, tgt = process_day(model, device, bt, mid)

        if emb is None:
            print(f"[FOLD {fold_n}] {test_date}: SKIP — too few bars ({n_bars})", flush=True)
            del bt, mid
            gc.collect()
            continue

        # ── Save output ───────────────────────────────────────────────────────
        np.savez_compressed(
            str(out_path),
            embeddings      = emb,                          # (N_valid, 512)
            predictions     = pred,                         # (N_valid,)
            targets         = tgt,                          # (N_valid,)
            fold            = np.array(fold_n, dtype=np.int32),
            checkpoint_fold = np.array(ckpt_fold, dtype=np.int32),
        )

        day_time   = time.time() - td
        total_time = time.time() - t0
        ram_gb     = psutil.virtual_memory().available / 1e9
        n_done    += 1

        print(f"  -> {len(emb):,} valid embeddings | "
              f"emb={emb.shape} | "
              f"pred=[{pred.min():.3f},{pred.max():.3f}] | "
              f"tgt=[{tgt.min():.2f},{tgt.max():.2f}] | "
              f"{day_time:.1f}s | RAM:{ram_gb:.0f}GB | total:{total_time:.0f}s",
              flush=True)

        del bt, mid, emb, pred, tgt
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Cleanup & summary ─────────────────────────────────────────────────────
    hook_handle.remove()
    total_time     = time.time() - t0
    final_files    = list(OUT_DIR.glob('*_embeddings.npz'))

    print("\n" + "=" * 75, flush=True)
    print(f"EXP-TS-02 COMPLETE", flush=True)
    print(f"  Processed (new):    {n_done}", flush=True)
    print(f"  Already done:       {n_already}", flush=True)
    print(f"  Data missing:       {n_data_miss}", flush=True)
    print(f"  Skipped (no ckpt):  {len(skipped)}", flush=True)
    print(f"  Total output files: {len(final_files)}", flush=True)
    print(f"  Wall time:          {total_time:.0f}s ({total_time/60:.1f}min)", flush=True)
    print(f"  Output dir:         {OUT_DIR}", flush=True)
    print("", flush=True)
    print("LEAK-FREE GUARANTEE: Each date embedded using weights that were", flush=True)
    print("trained ONLY on prior dates. Zero look-ahead bias.", flush=True)
    print("", flush=True)
    print("NEXT STEP: Combine cnn_embeddings_wf/ + cnn_embeddings_fold74/", flush=True)
    print("  WF dates (folds 73-76):  cnn_embeddings_wf/  (4 dates, this script)", flush=True)
    print("  OOT dates (days 87-173): cnn_embeddings_fold74/ (Razer extraction)", flush=True)
    print("  These two sets are BOTH leak-free and can be combined for LSTM training.", flush=True)
    print("=" * 75, flush=True)


if __name__ == '__main__':
    main()
