#!/usr/bin/env python3
"""
MC Dropout Inference for BookSpatialCNN
========================================
Runs N stochastic forward passes per bar with Dropout active (model.eval() +
re-enabled Dropout/Dropout2d) to produce per-bar (pred_mean, pred_std).

High pred_std = model disagrees with itself = low-confidence bar.
Prior finding: pred_std anti-correlates with IC at r=-0.77 (gating_model work).
When pred_std > 0.15, IC drops. This script is the first step to exploiting that.

Usage (single checkpoint):
    python mc_dropout_inference.py \
        --checkpoint checkpoints/cnn_wf_fold20_2025-12-29.pt \
        --data-dir ../../data/processed/dl_book_cache_oot \
        --output-dir results/mc_dropout/

Usage (all WF checkpoints, matched to their prediction date):
    python mc_dropout_inference.py \
        --wf-mode \
        --checkpoints-dir checkpoints \
        --data-dir ../../data/processed/dl_book_cache_oot \
        --output-dir results/mc_dropout/

Usage (static OOT checkpoint, all days):
    python mc_dropout_inference.py \
        --checkpoint results/best_cnn_oot_20260311_015004.pt \
        --data-dir ../../data/processed/dl_book_cache_oot \
        --output-dir results/mc_dropout/ \
        --all-days

Outputs:
    results/mc_dropout/{date}_mc_preds.npz
        pred_mean   : (n_bars,) float32 — mean prediction across MC passes
        pred_std    : (n_bars,) float32 — std across MC passes (uncertainty)
        mid_prices  : (n_bars,) float64 — from source NPZ

    results/mc_dropout/mc_summary.json
        Per-day stats: mean_uncertainty, IC, corr(pred_std, |pred_mean|), etc.
"""

import sys
import os
import gc
import json
import time
import argparse
import logging
import re
import numpy as np
import psutil
from pathlib import Path
from datetime import datetime
from scipy import stats

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Resolve Lvl3Quant root so the script works from any cwd
_THIS = Path(__file__).resolve()
ROOT = _THIS.parent.parent.parent          # …/Lvl3Quant
sys.path.insert(0, str(_THIS.parent))      # so we can import book_spatial_cnn

import torch
import torch.nn as nn
from book_spatial_cnn import BookSpatialCNN

# ---------------------------------------------------------------------------
# Constants matching the production inference pipeline
# ---------------------------------------------------------------------------
WINDOW_SIZE = 20
NUM_LEVELS  = 20
NUM_FEATURES = 4
CHUNK       = 2000   # GPU chunk size (bars at a time)
BATCH_SIZE  = 512    # mini-batch inside each chunk
CNN_OFFSET  = WINDOW_SIZE - 1   # 19 — first valid prediction index
HORIZON     = 100    # bars for IC calculation (~10s)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'mc_dropout_{_ts}.log'
    logger = logging.getLogger('mc_dropout')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(ch)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    return logger


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> BookSpatialCNN:
    """Load BookSpatialCNN from a state_dict checkpoint."""
    model = BookSpatialCNN(
        window_size=WINDOW_SIZE,
        num_levels=NUM_LEVELS,
        num_features=NUM_FEATURES,
        spatial_channels=(32, 64, 128, 256),
        temporal_channels=256,
        dropout=0.1,
        num_classes=1,
    ).to(device)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def enable_mc_dropout(model: nn.Module) -> int:
    """
    Set model to eval() (freezes BN, disables Dropout by default),
    then re-enable ONLY Dropout and Dropout2d layers for stochastic inference.

    Returns the count of re-enabled dropout modules.
    """
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pre-processing (mirrors run_oot_inference.py exactly)
# ---------------------------------------------------------------------------

def preprocess_book_tensor(bt: np.ndarray) -> np.ndarray:
    """Log-transform depth/orders/age features in-place. Returns float32."""
    bt = bt.astype(np.float32, copy=True)
    np.log1p(bt[:, :, 1], out=bt[:, :, 1])   # depth_lots
    np.log1p(bt[:, :, 2], out=bt[:, :, 2])   # num_orders
    np.log1p(bt[:, :, 3], out=bt[:, :, 3])   # queue_age_seconds
    return bt


# ---------------------------------------------------------------------------
# Core MC inference — single day
# ---------------------------------------------------------------------------

def mc_inference_day(
    bt: np.ndarray,
    model: nn.Module,
    device: torch.device,
    n_passes: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N stochastic forward passes on a full day of book tensors.

    Args:
        bt          : (n_bars, 20, 4) float32, already log-transformed
        model       : BookSpatialCNN with MC dropout active
        device      : torch device
        n_passes    : number of stochastic forward passes (default 30)

    Returns:
        pred_mean   : (n_bars,) float32 — mean prediction
        pred_std    : (n_bars,) float32 — std (uncertainty estimate)
    """
    n = len(bt)
    # Accumulate predictions across passes: shape (n_passes, n_bars)
    all_preds = np.zeros((n_passes, n), dtype=np.float32)

    for pass_idx in range(n_passes):
        preds = np.zeros(n, dtype=np.float32)

        for cstart in range(0, n, CHUNK):
            cend = min(cstart + CHUNK, n)
            slice_start = max(0, cstart - WINDOW_SIZE + 1)
            bt_chunk = bt[slice_start:cend]
            bt_gpu = torch.from_numpy(bt_chunk).to(device)

            if len(bt_gpu) < WINDOW_SIZE:
                del bt_gpu
                continue

            # unfold: (n_windows, window_size, 20, 4) -> permute -> (n_windows, 4, 20, window_size)?
            # Must match the inference code: unfold(0, ws, 1).permute(0, 3, 1, 2)
            # unfold(0, ws, 1): (n_windows, 20, 4, ws) — NO, unfold on dim 0 yields (n_windows, 20, 4)
            # Actually in run_oot_inference.py:
            #   windowed = bt_gpu.unfold(0, ws, 1).permute(0, 3, 1, 2).contiguous()
            # bt_gpu shape: (L, 20, 4)
            # unfold(0, ws=20, step=1): (n_windows, 20, 4, 20) — last dim is the unfolded window
            # permute(0, 3, 1, 2): (n_windows, 20, 20, 4) = (batch, window_size, num_levels, num_features) ✓
            windowed = bt_gpu.unfold(0, WINDOW_SIZE, 1).permute(0, 3, 1, 2).contiguous()
            pred_offset = slice_start + WINDOW_SIZE - 1

            for bstart in range(0, len(windowed), BATCH_SIZE):
                bend = min(bstart + BATCH_SIZE, len(windowed))
                batch = windowed[bstart:bend]

                # No torch.no_grad() here — dropout must be active (and it is via .train())
                # Using autocast for speed; dropout is not affected by precision
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = model(batch).squeeze(-1)
                else:
                    out = model(batch).squeeze(-1)

                g_start = pred_offset + bstart
                g_end   = pred_offset + bend
                preds[g_start:g_end] = out.detach().cpu().float().numpy()

            del bt_gpu, windowed
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        all_preds[pass_idx] = preds

    pred_mean = all_preds.mean(axis=0)
    pred_std  = all_preds.std(axis=0)
    return pred_mean, pred_std


# ---------------------------------------------------------------------------
# IC calculation (mirrors run_wf_fill_sim.py)
# ---------------------------------------------------------------------------

def compute_ic(pred_mean: np.ndarray, mid: np.ndarray, horizon: int = HORIZON) -> float | None:
    """Spearman rank correlation of pred_mean vs horizon-bar forward return."""
    n = len(mid)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        if mid[i] > 0:
            fwd_ret[i] = (mid[i + horizon] - mid[i]) / mid[i]
    valid = (~np.isnan(pred_mean) & ~np.isnan(fwd_ret)
             & (pred_mean != 0) & np.isfinite(fwd_ret))
    if valid.sum() < 50:
        return None
    ic, _ = stats.spearmanr(pred_mean[valid].astype(np.float64), fwd_ret[valid])
    return float(ic)


def compute_uncertainty_ic_corr(pred_mean: np.ndarray, pred_std: np.ndarray,
                                 mid: np.ndarray, horizon: int = HORIZON) -> dict:
    """
    Compute correlation between per-bar uncertainty (pred_std) and
    absolute prediction magnitude (|pred_mean|), and between pred_std and
    whether the prediction was correct (sign matches future return).

    Key hypothesis from gating_model work:
        - pred_std anti-correlates with IC at r=-0.77 (day-level)
        - Here we compute bar-level version of the same idea

    Returns dict with:
        corr_std_abspred  : Spearman(pred_std, |pred_mean|)  — are high-confidence bars also high-magnitude?
        corr_std_correct  : Spearman(pred_std, bar_correct)   — do low-uncertainty bars get direction right?
        mean_uncertainty  : mean(pred_std) over valid bars
        std_uncertainty   : std(pred_std) over valid bars
        pct_high_uncert   : fraction of bars with pred_std > threshold_75pct
    """
    n = len(mid)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        if mid[i] > 0:
            fwd_ret[i] = (mid[i + horizon] - mid[i]) / mid[i]

    valid = (~np.isnan(pred_mean) & ~np.isnan(fwd_ret)
             & (pred_mean != 0) & np.isfinite(fwd_ret)
             & np.isfinite(pred_std))

    result = {
        'corr_std_abspred': None,
        'corr_std_correct': None,
        'mean_uncertainty': float(pred_std[valid].mean()) if valid.sum() > 0 else None,
        'std_uncertainty':  float(pred_std[valid].std())  if valid.sum() > 0 else None,
        'pct_high_uncert':  None,
        'n_valid':          int(valid.sum()),
    }

    if valid.sum() < 50:
        return result

    pm = pred_mean[valid].astype(np.float64)
    ps = pred_std[valid].astype(np.float64)
    fr = fwd_ret[valid]

    # corr(pred_std, |pred_mean|) — are confident bars also high-signal?
    r1, _ = stats.spearmanr(ps, np.abs(pm))
    result['corr_std_abspred'] = float(r1)

    # bar_correct: 1 if sign(pred_mean) == sign(fwd_ret), 0 otherwise
    bar_correct = ((pm > 0) == (fr > 0)).astype(np.float64)
    r2, _ = stats.spearmanr(ps, bar_correct)
    result['corr_std_correct'] = float(r2)

    # fraction with high uncertainty (> 75th pct of std)
    thr75 = np.percentile(ps, 75)
    result['pct_high_uncert'] = float((ps > thr75).mean())
    result['std_threshold_75pct'] = float(thr75)

    return result


# ---------------------------------------------------------------------------
# WF checkpoint matching — parse date from filename
# ---------------------------------------------------------------------------

def parse_wf_checkpoint_date(ckpt_path: Path) -> str | None:
    """
    Parse the date from a WF checkpoint name.
    Pattern: cnn_wf_fold{N}_{YYYY-MM-DD}.pt  ->  'YYYY-MM-DD'
    """
    m = re.search(r'(\d{4}-\d{2}-\d{2})', ckpt_path.stem)
    return m.group(1) if m else None


def build_wf_checkpoint_map(checkpoints_dir: Path) -> dict[str, Path]:
    """
    Build a map from date -> checkpoint path for all WF CNN checkpoints.
    Skips GNN checkpoints (those don't have a date in their name).
    """
    ckpt_map = {}
    for f in sorted(checkpoints_dir.glob('cnn_wf_*.pt')):
        date = parse_wf_checkpoint_date(f)
        if date:
            ckpt_map[date] = f
    return ckpt_map


# ---------------------------------------------------------------------------
# Uncertainty-gated signal for fill_sim
# ---------------------------------------------------------------------------

def create_gated_signal(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    uncertainty_threshold: float,
    gate_mode: str = 'zero',
) -> np.ndarray:
    """
    Create uncertainty-gated signal for the Rust fill_sim pipeline.

    Args:
        pred_mean             : (n_bars,) raw MC mean predictions
        pred_std              : (n_bars,) MC uncertainty per bar
        uncertainty_threshold : bars with pred_std > this get filtered
        gate_mode             : 'zero'  — set filtered bars to 0 (no signal)
                                'scale' — multiply by (1 - pred_std/max_std)

    Returns:
        gated_signal : (n_bars,) float32 — same shape as pred_mean, filtered
    """
    gated = pred_mean.copy().astype(np.float32)
    high_uncert = pred_std > uncertainty_threshold

    if gate_mode == 'zero':
        gated[high_uncert] = 0.0
    elif gate_mode == 'scale':
        max_std = pred_std.max() if pred_std.max() > 0 else 1.0
        scale = np.clip(1.0 - pred_std / max_std, 0.0, 1.0)
        gated = gated * scale.astype(np.float32)

    return gated


def save_gated_predictions_for_fillsim(
    date: str,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    output_dir: Path,
    uncertainty_thresholds: list[float],
    n_bars: int,
) -> dict[float, Path]:
    """
    Produce per-day NPZ files compatible with run_wf_fill_sim.py's 'predictions' key.
    One file per uncertainty threshold, named: {date}_mc_unc{thr}.npz

    The signal stored is the raw pred_mean with high-uncertainty bars zeroed out.
    The fill_sim pipeline then applies its own z-score + vol gate on top.
    """
    saved = {}
    # Align to CNN offset (same as run_wf_fill_sim.py)
    for thr in uncertainty_thresholds:
        gated = create_gated_signal(pred_mean, pred_std, thr, gate_mode='zero')
        # Zero-pad to n_bars with CNN offset alignment
        aligned = np.zeros(n_bars, dtype=np.float64)
        end_idx = min(CNN_OFFSET + len(gated), n_bars)
        aligned[CNN_OFFSET:end_idx] = gated[:end_idx - CNN_OFFSET].astype(np.float64)

        thr_str = f'{thr:.3f}'.replace('.', 'p')
        out_file = output_dir / f'{date}_mc_unc{thr_str}.npz'
        np.savez_compressed(str(out_file), predictions=aligned)
        saved[thr] = out_file

    return saved


# ---------------------------------------------------------------------------
# Main inference loop — single checkpoint, all days
# ---------------------------------------------------------------------------

def run_inference_single_checkpoint(
    checkpoint: Path,
    data_dir: Path,
    output_dir: Path,
    n_passes: int,
    uncertainty_thresholds: list[float],
    logger: logging.Logger,
) -> list[dict]:
    """Run MC inference for a single checkpoint on all days in data_dir."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    logger.info(f"Loading checkpoint: {checkpoint}")
    model = load_model(checkpoint, device)
    n_dropout = enable_mc_dropout(model)
    logger.info(f"MC mode: {n_dropout} dropout modules re-enabled (model.eval() + Dropout.train())")
    logger.info(f"MC passes per day: {n_passes}")

    npz_files = sorted(data_dir.glob('*_book_tensors.npz'))
    if not npz_files:
        logger.error(f"No book tensor NPZ files in {data_dir}")
        return []
    logger.info(f"{len(npz_files)} days found in {data_dir}")

    gated_dir = output_dir / 'gated_predictions'
    gated_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    t0 = time.time()

    for fi, f in enumerate(npz_files):
        ram = psutil.virtual_memory().available / 1e9
        if ram < 3.0:
            logger.error(f"ABORT: Only {ram:.1f}GB RAM free")
            break

        date = f.name.replace('_book_tensors.npz', '')
        td = time.time()

        # Load and preprocess
        npz = np.load(str(f))
        bt  = preprocess_book_tensor(npz['book_tensors'])
        mid = npz['mid_prices'].copy().astype(np.float64)
        npz.close()
        n_bars = len(bt)

        logger.info(f"[{fi+1}/{len(npz_files)}] {date}: {n_bars:,} bars, "
                    f"RAM={ram:.0f}GB — running {n_passes} MC passes...")

        # MC inference
        pred_mean, pred_std = mc_inference_day(bt, model, device, n_passes=n_passes)

        # IC
        ic = compute_ic(pred_mean, mid)
        ic_str = f'{ic:+.4f}' if ic is not None else 'N/A'

        # Uncertainty stats
        unc_stats = compute_uncertainty_ic_corr(pred_mean, pred_std, mid)

        # Save per-day NPZ
        out_npz = output_dir / f'{date}_mc_preds.npz'
        np.savez_compressed(
            str(out_npz),
            pred_mean=pred_mean.astype(np.float32),
            pred_std=pred_std.astype(np.float32),
            mid_prices=mid,
        )

        # Save gated predictions for fill_sim
        gated_files = save_gated_predictions_for_fillsim(
            date, pred_mean, pred_std, gated_dir,
            uncertainty_thresholds, n_bars,
        )

        day_time = time.time() - td
        total    = time.time() - t0
        logger.info(
            f"  IC={ic_str} | mean_unc={unc_stats['mean_uncertainty']:.4f} "
            f"| corr(std,correct)={unc_stats['corr_std_correct'] or 'N/A'} "
            f"| {day_time:.1f}s ({total:.0f}s total)"
        )

        day_summary = {
            'date': date,
            'n_bars': n_bars,
            'ic': ic,
            'n_passes': n_passes,
            **unc_stats,
            'checkpoint': str(checkpoint),
        }
        summary.append(day_summary)

        del bt, mid, pred_mean, pred_std
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Main inference loop — WF mode (each day uses its own checkpoint)
# ---------------------------------------------------------------------------

def run_inference_wf_mode(
    checkpoints_dir: Path,
    data_dir: Path,
    output_dir: Path,
    n_passes: int,
    uncertainty_thresholds: list[float],
    logger: logging.Logger,
) -> list[dict]:
    """
    Walk-forward mode: for each prediction day, load the checkpoint trained
    on data up to that day (cnn_wf_fold{N}_{date}.pt).

    WF checkpoints are named by the LAST training date, which is the date
    they start predicting FROM. So cnn_wf_fold20_2025-12-29.pt predicts
    on 2025-12-29 and subsequent days until the next checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    ckpt_map = build_wf_checkpoint_map(checkpoints_dir)
    if not ckpt_map:
        logger.error(f"No WF CNN checkpoints found in {checkpoints_dir}")
        return []
    ckpt_dates = sorted(ckpt_map.keys())
    logger.info(f"WF checkpoints: {len(ckpt_map)} ({ckpt_dates[0]} to {ckpt_dates[-1]})")

    npz_files = sorted(data_dir.glob('*_book_tensors.npz'))
    if not npz_files:
        logger.error(f"No book tensor NPZ files in {data_dir}")
        return []

    gated_dir = output_dir / 'gated_predictions'
    gated_dir.mkdir(parents=True, exist_ok=True)

    # Match each prediction day to the most recent checkpoint <= that day
    def get_checkpoint_for_date(pred_date: str) -> Path | None:
        eligible = [d for d in ckpt_dates if d <= pred_date]
        return ckpt_map[eligible[-1]] if eligible else None

    summary = []
    current_ckpt_path = None
    model = None
    t0 = time.time()

    for fi, f in enumerate(npz_files):
        ram = psutil.virtual_memory().available / 1e9
        if ram < 3.0:
            logger.error(f"ABORT: Only {ram:.1f}GB RAM free")
            break

        date = f.name.replace('_book_tensors.npz', '')

        ckpt_path = get_checkpoint_for_date(date)
        if ckpt_path is None:
            logger.warning(f"No checkpoint available for {date}, skipping")
            continue

        # Reload model only when checkpoint changes
        if ckpt_path != current_ckpt_path:
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            logger.info(f"Loading checkpoint: {ckpt_path.name}")
            model = load_model(ckpt_path, device)
            n_dropout = enable_mc_dropout(model)
            logger.info(f"  MC mode: {n_dropout} dropout modules active")
            current_ckpt_path = ckpt_path

        td = time.time()
        npz = np.load(str(f))
        bt  = preprocess_book_tensor(npz['book_tensors'])
        mid = npz['mid_prices'].copy().astype(np.float64)
        npz.close()
        n_bars = len(bt)

        logger.info(f"[{fi+1}/{len(npz_files)}] {date}: {n_bars:,} bars "
                    f"[ckpt={ckpt_path.stem}] RAM={ram:.0f}GB")

        pred_mean, pred_std = mc_inference_day(bt, model, device, n_passes=n_passes)

        ic = compute_ic(pred_mean, mid)
        ic_str = f'{ic:+.4f}' if ic is not None else 'N/A'
        unc_stats = compute_uncertainty_ic_corr(pred_mean, pred_std, mid)

        out_npz = output_dir / f'{date}_mc_preds.npz'
        np.savez_compressed(
            str(out_npz),
            pred_mean=pred_mean.astype(np.float32),
            pred_std=pred_std.astype(np.float32),
            mid_prices=mid,
        )

        gated_files = save_gated_predictions_for_fillsim(
            date, pred_mean, pred_std, gated_dir,
            uncertainty_thresholds, n_bars,
        )

        day_time = time.time() - td
        total    = time.time() - t0
        logger.info(
            f"  IC={ic_str} | mean_unc={unc_stats['mean_uncertainty']:.4f} "
            f"| corr(std,correct)={unc_stats['corr_std_correct'] or 'N/A'} "
            f"| {day_time:.1f}s ({total:.0f}s total)"
        )

        day_summary = {
            'date': date,
            'n_bars': n_bars,
            'ic': ic,
            'n_passes': n_passes,
            'checkpoint': str(ckpt_path),
            **unc_stats,
        }
        summary.append(day_summary)

        del bt, mid, pred_mean, pred_std
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return summary


# ---------------------------------------------------------------------------
# Summary reporting and analysis
# ---------------------------------------------------------------------------

def report_and_save_summary(
    summary: list[dict],
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """
    Print per-day stats, compute aggregate analysis, save mc_summary.json.

    Key analysis outputs:
      - Mean IC with/without high-uncertainty bars
      - Corr between day-level mean_uncertainty and day-level IC
      - Distribution of uncertainty values
    """
    if not summary:
        logger.warning("No summary data to report.")
        return {}

    logger.info("\n" + "=" * 90)
    logger.info("MC DROPOUT INFERENCE SUMMARY")
    logger.info(f"Days: {len(summary)} | MC passes: {summary[0]['n_passes']}")
    logger.info("=" * 90)
    logger.info(f"{'Date':<14} {'IC':>8} {'MeanUnc':>10} {'CorrStdCrt':>12} {'N_bars':>10}")
    logger.info("-" * 60)

    valid_ics = []
    mean_uncs = []
    corr_std_corrects = []

    for s in summary:
        ic_str  = f"{s['ic']:+.4f}"  if s['ic'] is not None else '  N/A  '
        munc    = s['mean_uncertainty']
        csc     = s['corr_std_correct']
        munc_str = f"{munc:.5f}" if munc is not None else '  N/A  '
        csc_str  = f"{csc:+.4f}" if csc is not None else '  N/A  '

        logger.info(f"  {s['date']:<12} {ic_str:>8} {munc_str:>12} {csc_str:>12} {s['n_bars']:>10,}")

        if s['ic'] is not None:
            valid_ics.append(s['ic'])
        if munc is not None:
            mean_uncs.append(munc)
        if csc is not None:
            corr_std_corrects.append(csc)

    logger.info("-" * 60)

    # Aggregate IC stats
    agg = {}
    if valid_ics:
        ic_arr = np.array(valid_ics)
        agg['ic_mean']     = float(np.mean(ic_arr))
        agg['ic_std']      = float(np.std(ic_arr))
        agg['ic_tstat']    = float(np.mean(ic_arr) / (np.std(ic_arr) / np.sqrt(len(ic_arr))))
        agg['ic_pct_pos']  = float(np.mean(ic_arr > 0) * 100)
        agg['ic_n_days']   = len(ic_arr)
        logger.info(f"\nIC: mean={agg['ic_mean']:+.4f}, t={agg['ic_tstat']:.2f}, "
                    f"{agg['ic_pct_pos']:.0f}% positive ({agg['ic_n_days']} days)")

    # Day-level corr(mean_uncertainty, IC)
    if len(mean_uncs) >= 5 and len(valid_ics) >= 5:
        min_len = min(len(mean_uncs), len(valid_ics))
        r, p = stats.spearmanr(mean_uncs[:min_len], valid_ics[:min_len])
        agg['day_corr_uncertainty_ic'] = float(r)
        agg['day_corr_uncertainty_ic_pval'] = float(p)
        logger.info(f"Day-level Spearman(mean_uncertainty, IC): r={r:+.3f}, p={p:.4f}")
        logger.info("  (Hypothesis from gating model: expect r ≈ -0.77)")

    if mean_uncs:
        agg['mean_uncertainty_across_days'] = float(np.mean(mean_uncs))
        agg['std_uncertainty_across_days']  = float(np.std(mean_uncs))
        logger.info(f"Mean uncertainty: {agg['mean_uncertainty_across_days']:.5f} "
                    f"± {agg['std_uncertainty_across_days']:.5f}")

    if corr_std_corrects:
        agg['mean_corr_std_correct'] = float(np.mean(corr_std_corrects))
        logger.info(f"Mean bar-level corr(pred_std, correct_direction): "
                    f"{agg['mean_corr_std_correct']:+.4f}")
        logger.info("  (Negative = high uncertainty bars more often wrong)")

    # Save
    out_json = output_dir / 'mc_summary.json'
    output_data = {
        'timestamp': _ts,
        'n_days': len(summary),
        'n_passes': summary[0]['n_passes'],
        'aggregate': agg,
        'per_day': summary,
    }
    with open(str(out_json), 'w') as fh:
        json.dump(output_data, fh, indent=2, default=str)
    logger.info(f"\nSummary saved: {out_json}")
    logger.info("=" * 90)

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='MC Dropout Inference for BookSpatialCNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single static checkpoint (OOT), all days:
  python mc_dropout_inference.py \\
    --checkpoint results/best_cnn_oot_20260311_015004.pt \\
    --data-dir ../../data/processed/dl_book_cache_oot \\
    --output-dir results/mc_dropout/ \\
    --all-days

  # Walk-forward mode (each day uses its matched WF checkpoint):
  python mc_dropout_inference.py \\
    --wf-mode \\
    --checkpoints-dir checkpoints \\
    --data-dir ../../data/processed/dl_book_cache_oot \\
    --output-dir results/mc_dropout/

  # Quick test — 5 passes on first 3 days:
  python mc_dropout_inference.py \\
    --checkpoint results/best_cnn_oot_20260311_015004.pt \\
    --data-dir ../../data/processed/dl_book_cache_oot \\
    --output-dir results/mc_dropout/ \\
    --n-passes 5 --max-days 3
        """
    )

    # Checkpoint selection — one of these two must be provided
    ckpt_grp = p.add_mutually_exclusive_group(required=True)
    ckpt_grp.add_argument(
        '--checkpoint', type=Path, metavar='PATH',
        help='Single .pt checkpoint for all days'
    )
    ckpt_grp.add_argument(
        '--wf-mode', action='store_true',
        help='Walk-forward mode: match each day to its trained WF checkpoint'
    )

    p.add_argument(
        '--checkpoints-dir', type=Path,
        default=_THIS.parent / 'checkpoints',
        metavar='DIR',
        help='Directory with cnn_wf_fold*.pt files (--wf-mode only, '
             'default: <script_dir>/checkpoints)'
    )
    p.add_argument(
        '--data-dir', type=Path, required=True, metavar='DIR',
        help='Directory with *_book_tensors.npz files'
    )
    p.add_argument(
        '--output-dir', type=Path, required=True, metavar='DIR',
        help='Output directory for per-day NPZ + summary JSON'
    )
    p.add_argument(
        '--n-passes', type=int, default=30, metavar='N',
        help='Number of stochastic MC forward passes per day (default: 30)'
    )
    p.add_argument(
        '--all-days', action='store_true',
        help='Process all days in data-dir (default when using single checkpoint)'
    )
    p.add_argument(
        '--max-days', type=int, default=None, metavar='N',
        help='Limit to first N days (useful for testing)'
    )
    p.add_argument(
        '--uncertainty-thresholds', type=str,
        default='0.05,0.10,0.15,0.20,0.25',
        metavar='FLOATS',
        help='Comma-separated uncertainty thresholds for gated signal output '
             '(default: "0.05,0.10,0.15,0.20,0.25")'
    )
    p.add_argument(
        '--skip-existing', action='store_true',
        help='Skip days that already have output NPZ files'
    )

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=" * 65)
    logger.info("MC Dropout Inference — BookSpatialCNN")
    logger.info(f"  Mode:       {'Walk-Forward' if args.wf_mode else 'Single checkpoint'}")
    if not args.wf_mode:
        logger.info(f"  Checkpoint: {args.checkpoint}")
    else:
        logger.info(f"  Ckpt dir:   {args.checkpoints_dir}")
    logger.info(f"  Data dir:   {args.data_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  MC passes:  {args.n_passes}")
    logger.info(f"  Max days:   {args.max_days or 'all'}")
    logger.info("=" * 65)

    # Parse uncertainty thresholds
    unc_thrs = [float(x) for x in args.uncertainty_thresholds.split(',')]
    logger.info(f"Uncertainty gating thresholds: {unc_thrs}")

    # Validate inputs
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not args.wf_mode and not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Optionally limit days by patching the glob
    if args.max_days is not None:
        orig_glob = args.data_dir.glob

        def limited_glob(pattern):
            files = sorted(orig_glob(pattern))
            return iter(files[:args.max_days])

        args.data_dir.glob = limited_glob

    # Run
    if args.wf_mode:
        if not args.checkpoints_dir.exists():
            logger.error(f"Checkpoints directory not found: {args.checkpoints_dir}")
            sys.exit(1)
        summary = run_inference_wf_mode(
            checkpoints_dir=args.checkpoints_dir,
            data_dir=args.data_dir,
            output_dir=output_dir,
            n_passes=args.n_passes,
            uncertainty_thresholds=unc_thrs,
            logger=logger,
        )
    else:
        summary = run_inference_single_checkpoint(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=output_dir,
            n_passes=args.n_passes,
            uncertainty_thresholds=unc_thrs,
            logger=logger,
        )

    agg = report_and_save_summary(summary, output_dir, logger)

    logger.info("\nDone.")
    logger.info(f"Per-day NPZ files: {output_dir}/<date>_mc_preds.npz")
    logger.info(f"Gated signals for fill_sim: {output_dir}/gated_predictions/<date>_mc_unc*.npz")
    logger.info(f"Summary: {output_dir}/mc_summary.json")


if __name__ == '__main__':
    main()
