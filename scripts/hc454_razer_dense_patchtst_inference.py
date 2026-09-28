#!/usr/bin/env python3
"""
HC #454 R3(b)(1) — DENSE-STRIDE PatchTST INFERENCE on RAZER GPU.

Joint-gate diagnostic (2026-05-21 07:23 ET) showed 5s + 10s PatchTST horizons
PASS the HC #455 R1 joint smoothness/informativeness gate. HC #454 R3 decision
tree → "dispatch Razer to densify PatchTST inference (stride < 250) to test
whether stream cadence reduces flicker; HC #453 R8 may be overturned".

This script:
  - Loads fold_15_best.pt PatchTST checkpoint (best IC=0.0632, training default).
  - Iterates over recent N days of MBO smart_v3 events available on Razer.
  - Runs dense-stride inference (stride=25, 10x denser than training default 250).
  - Saves one prediction NPZ per day in output/hc454_dense_patchtst/.
  - GPU device='cuda', batch_size tuned for RTX 3070 8GB.

Acceptance:
  - Razer 3070 must show >40% utilization within 2 minutes of launch.
  - Per-day NPZs land in output/hc454_dense_patchtst/.
  - HC #453 R8 stream-coherence question is testable: feed these dense predictions
    back through the HC #455 R1 joint-gate diagnostic to compare smoothness vs
    the legacy stride=250 baseline.

Usage on Razer:
  python scripts/hc454_razer_dense_patchtst_inference.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hc454_dense_ptst")

# ─── Paths (RAZER) ──────────────────────────────────────────────────────────
REPO = Path(os.environ.get("LVL3_ROOT", r"C:\Users\claude\Lvl3Quant"))
MBO_DIR = REPO / "data" / "processed" / "mbo_events_smart_v3"
WEIGHTS = REPO / "models" / "patchtst_razer_weights" / "fold_15_best.pt"
OUT_DIR = REPO / "output" / os.environ.get("OUT_DIR_NAME", "hc454_dense_patchtst")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Inference params ───────────────────────────────────────────────────────
WINDOW = 500
STRIDE_DENSE = int(os.environ.get("STRIDE_DENSE", "25"))   # 10x denser than training default 250
BATCH = 64          # RTX 3070 8GB safe batch
N_DAYS = int(os.environ.get("N_DAYS", "30"))  # how many recent days to process
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── PatchTST architecture (matches fold_15_best.pt) ────────────────────────
class ALiBiAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        slopes = torch.tensor(self._get_slopes(n_heads), dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    @staticmethod
    def _get_slopes(n):
        def slopes_pow2(k):
            start = 2 ** (-2 ** -(np.log2(k) - 3))
            return [start * (start ** i) for i in range(k)]
        if (n & (n - 1)) == 0:
            return slopes_pow2(n)
        closest = 2 ** int(np.floor(np.log2(n)))
        return slopes_pow2(closest) + ALiBiAttention._get_slopes(2 * closest)[0::2][: n - closest]

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        pos = torch.arange(L, device=x.device)
        bias = -(pos[None, :] - pos[:, None]).abs().float()
        bias = bias.unsqueeze(0).unsqueeze(0) * self.alibi_slopes.view(1, -1, 1, 1)
        attn = attn + bias
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ALiBiAttention(d_model, n_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PatchTST(nn.Module):
    def __init__(self, n_features=25, patch_size=25, d_model=256, n_heads=4,
                 head_dim=64, n_layers=4, ffn_dim=1024, dropout=0.1,
                 n_targets=3, window_size=500):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size
        patch_dim = patch_size * n_features
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, head_dim, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x):
        B, W, F = x.shape
        x = x.reshape(B, self.n_patches, self.patch_size * F)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def load_patchtst(path: Path, device: str) -> PatchTST:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    arch = ckpt.get("arch", {})
    kwargs = {k: arch[k] for k in (
        "n_features", "patch_size", "d_model", "n_heads", "head_dim",
        "n_layers", "ffn_dim", "dropout", "window_size") if k in arch}
    log.info(f"Loading PatchTST arch={kwargs}")
    log.info(f"  fold={ckpt.get('fold')} val_ic_10s={ckpt.get('val_ic_10s', 0):.4f}")
    model = PatchTST(**kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical = [k for k in missing if not k.endswith("alibi_slopes")]
    if critical:
        raise RuntimeError(f"Critical missing keys: {critical}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected}")
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    log.info(f"PatchTST loaded: {n/1e6:.2f}M params on {device}")
    return model


@torch.no_grad()
def run_day(model: PatchTST, events: np.ndarray,
            labels_1s: np.ndarray, labels_5s: np.ndarray, labels_10s: np.ndarray,
            window: int, stride: int, batch: int, device: str):
    n = len(events)
    valid_starts = [
        s for s in range(0, n - window + 1, stride)
        if (not np.isnan(labels_1s[s + window - 1])
            and not np.isnan(labels_5s[s + window - 1])
            and not np.isnan(labels_10s[s + window - 1]))
    ]
    N = len(valid_starts)
    if N == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    events_t = torch.from_numpy(events).to(device)
    preds_all = []
    labels_all = []
    for i in range(0, N, batch):
        bs = valid_starts[i: i + batch]
        xb = torch.stack([events_t[s: s + window] for s in bs])  # (B, W, F)
        out = model(xb)
        preds_all.append(out.cpu().numpy())
        labels_all.append(np.array([
            [labels_1s[s + window - 1], labels_5s[s + window - 1], labels_10s[s + window - 1]]
            for s in bs
        ], dtype=np.float32))
        if (i // batch) % 50 == 0 and i > 0:
            log.info(f"    {i:,}/{N:,} ({100*i/N:.0f}%)")

    return np.concatenate(preds_all, 0), np.concatenate(labels_all, 0)


def main():
    if not WEIGHTS.exists():
        log.error(f"Weights not found: {WEIGHTS}")
        sys.exit(2)
    if not MBO_DIR.exists():
        log.error(f"MBO dir not found: {MBO_DIR}")
        sys.exit(2)

    log.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)} "
                 f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    model = load_patchtst(WEIGHTS, DEVICE)

    all_files = sorted(MBO_DIR.glob("*_mbo_events.npz"))
    # Latest N days (skip today since it may be mid-session)
    target = all_files[-N_DAYS-1: -1] if len(all_files) > N_DAYS else all_files
    log.info(f"Processing {len(target)} days, stride={STRIDE_DENSE} (10x dense)")

    summary = []
    t0 = time.time()
    for i, f in enumerate(target):
        date = f.name.split("_")[0]
        out_npz = OUT_DIR / f"{date}_dense_predictions.npz"
        if out_npz.exists():
            log.info(f"[{i+1}/{len(target)}] {date} — already done, skipping")
            continue

        log.info(f"[{i+1}/{len(target)}] {date} — loading…")
        try:
            d = np.load(f, allow_pickle=True)
            events = d["events"].astype(np.float32)
            l1 = d["labels_1s"].astype(np.float32)
            l5 = d["labels_5s"].astype(np.float32)
            l10 = d["labels_10s"].astype(np.float32)
        except Exception as e:
            log.error(f"  load failed: {e}")
            continue

        t_day = time.time()
        try:
            preds, labels = run_day(model, events, l1, l5, l10,
                                    WINDOW, STRIDE_DENSE, BATCH, DEVICE)
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"  CUDA OOM at batch={BATCH}, day skipped: {e}")
            torch.cuda.empty_cache()
            continue

        # IC per horizon
        ics = []
        for col in range(3):
            v = ~(np.isnan(preds[:, col]) | np.isnan(labels[:, col]))
            if v.sum() >= 10:
                import scipy.stats
                ic, _ = scipy.stats.spearmanr(preds[v, col], labels[v, col])
                ics.append(float(ic))
            else:
                ics.append(float("nan"))

        np.savez(out_npz,
                 predictions=preds.astype(np.float32),
                 labels=labels.astype(np.float32),
                 horizons=np.array([1, 5, 10], np.int32),
                 ic_1s=ics[0], ic_5s=ics[1], ic_10s=ics[2],
                 stride=STRIDE_DENSE, window=WINDOW,
                 source_file=str(f.name))
        dt = time.time() - t_day
        log.info(f"  {date}: N={len(preds):,} IC=({ics[0]:.4f},{ics[1]:.4f},{ics[2]:.4f}) "
                 f"in {dt:.1f}s → {out_npz.name}")
        summary.append((date, len(preds), ics))

    log.info(f"=== DONE in {(time.time()-t0)/60:.1f} min ===")
    log.info(f"Days processed: {len(summary)}")
    for date, n, ics in summary[-5:]:
        log.info(f"  {date}: N={n:,} IC=({ics[0]:.4f},{ics[1]:.4f},{ics[2]:.4f})")


if __name__ == "__main__":
    main()
