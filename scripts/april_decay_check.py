#!/usr/bin/env python3
"""
CNN-Mamba v2 Model Decay Check — April 2026
Computes IC on recent April dates to detect signal decay.
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
DATA_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
WEIGHTS = ROOT / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
OUTPUT_FILE = ROOT / "output" / "model_decay_april_check.json"

TEST_DATES = ["20260421", "20260422", "20260423", "20260424", "20260428", "20260429"]
WINDOW = 1000
STRIDE = 3000  # Large stride for CPU feasibility
BATCH_SIZE = 64
MAX_WINDOWS_PER_DAY = 500  # Cap to keep runtime reasonable on CPU

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Model Definition (inline for portability)
# =============================================================================

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4):
        super().__init__()
        self.d_model, self.d_state, self.dt_rank, self.d_conv = d_model, d_state, dt_rank, d_conv
        self.d_inner = d_model * 2
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=0, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x, time_delta=None):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_conv = F.pad(x_branch.transpose(1,2).contiguous(), (self.d_conv-1, 0))
        x_branch = F.silu(self.conv1d(x_conv).transpose(1,2).contiguous())
        x_proj = self.x_proj(x_branch)
        dt_x = x_proj[:,:,:self.dt_rank]
        B_sel = x_proj[:,:,self.dt_rank:self.dt_rank+self.d_state]
        C_sel = x_proj[:,:,self.dt_rank+self.d_state:]
        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log)
        y = self._scan(x_branch, dt, A, B_sel, C_sel, self.D, time_delta)
        return self.out_proj(y * F.silu(z))

    def _scan(self, x, dt, A, B, C, D, time_delta=None):
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        if time_delta is not None:
            td = time_delta.unsqueeze(-1).unsqueeze(-1)
            dA = dA * torch.exp(-F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0) * td.abs())
        dBx = (B.unsqueeze(2) * dt.unsqueeze(-1)) * x.unsqueeze(-1)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = dA[:,t] * h + dBx[:,t]
            outputs.append(torch.einsum("bn,bdn->bd", C[:,t], h))
        return torch.stack(outputs, dim=1) + x * D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, dt_rank, d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_delta=None):
        return x + self.dropout(self.ssm(self.norm(x), time_delta))


class CNNMambaV2(nn.Module):
    CNN_KERNELS = [3, 7, 15, 31]

    def __init__(self, n_smart_features=25, d_model=96, d_state=32, n_layers=3,
                 dt_rank=6, d_conv=4, dropout=0.1, n_targets=3,
                 feature_mlp_hidden=128, feature_mlp_out=64, cnn_channels_per_scale=16):
        super().__init__()
        self.d_model, self.n_targets = d_model, n_targets
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_smart_features, feature_mlp_hidden), nn.GELU(),
            nn.Dropout(dropout*0.5),
            nn.Linear(feature_mlp_hidden, feature_mlp_out), nn.LayerNorm(feature_mlp_out)
        )
        self.cnn_kernels = self.CNN_KERNELS
        self.temporal_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_smart_features, cnn_channels_per_scale, kernel_size=k, padding=0, bias=True),
                nn.GELU()
            ) for k in self.cnn_kernels
        ])
        self.cnn_out_dim = cnn_channels_per_scale * len(self.cnn_kernels)
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_mlp_out + self.cnn_out_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, dt_rank, d_conv, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_targets)
        )

    def forward(self, smart):
        B, L, _ = smart.shape
        time_delta = smart[:,:,0]
        feat_out = self.feature_mlp(smart)
        x_t = smart.transpose(1,2)
        cnn_out = torch.cat([
            conv(F.pad(x_t, (k-1, 0))).transpose(1,2)
            for k, conv in zip(self.cnn_kernels, self.temporal_cnns)
        ], dim=-1)
        x = self.fusion_proj(torch.cat([feat_out, cnn_out], dim=-1))
        for block in self.blocks:
            x = block(x, time_delta)
        return self.head(self.final_norm(x)[:, -1, :])


# =============================================================================
# IC computation
# =============================================================================

def compute_ic(preds, labels):
    """Pearson correlation (Information Coefficient)."""
    if len(preds) < 10:
        return float('nan')
    p = preds - preds.mean()
    l = labels - labels.mean()
    denom = (np.sqrt((p**2).sum()) * np.sqrt((l**2).sum()))
    if denom < 1e-12:
        return float('nan')
    return float((p * l).sum() / denom)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("CNN-Mamba v2 Model Decay Check — April 2026")
    print("=" * 70)

    # Load model
    print(f"\nLoading weights from: {WEIGHTS}")
    device = torch.device("cpu")
    ckpt = torch.load(str(WEIGHTS), map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    state = ckpt["model_state"]

    # Infer dt_rank from weight shapes
    d_state = arch.get("d_state", 32)
    if "blocks.0.ssm.x_proj.weight" in state:
        dt_rank = state["blocks.0.ssm.x_proj.weight"].shape[0] - 2 * d_state
    else:
        dt_rank = 6

    print(f"  d_model={arch.get('d_model',96)}, d_state={d_state}, n_layers={arch.get('n_layers',3)}, dt_rank={dt_rank}")

    model = CNNMambaV2(
        d_model=arch.get("d_model", 96),
        d_state=d_state,
        n_layers=arch.get("n_layers", 3),
        dt_rank=dt_rank,
        d_conv=arch.get("d_conv", 4),
        dropout=arch.get("dropout", 0.1),
        n_targets=3,
        feature_mlp_hidden=arch.get("feature_mlp_hidden", 128),
        feature_mlp_out=arch.get("feature_mlp_out", 64),
        cnn_channels_per_scale=arch.get("cnn_channels_per_scale", 16),
    )
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"  WARNING: missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  WARNING: unexpected keys: {load_result.unexpected_keys[:5]}")
    model.eval()
    print("  Model loaded successfully.")

    # Process each date
    results = {"dates": {}, "concat": {}, "baseline_oot": {"IC_1s": 0.20, "IC_5s": 0.10, "IC_10s": 0.07}}
    all_preds = {h: [] for h in ["1s", "5s", "10s"]}
    all_labels = {h: [] for h in ["1s", "5s", "10s"]}

    torch.set_num_threads(8)

    for date_str in TEST_DATES:
        fpath = DATA_DIR / f"{date_str}_mbo_events.npz"
        if not fpath.exists():
            print(f"\n  {date_str}: FILE NOT FOUND, skipping")
            results["dates"][date_str] = {"error": "file not found"}
            continue

        t0 = time.time()
        print(f"\n  Processing {date_str}...", end="", flush=True)

        d = np.load(fpath)
        events = d["events"]
        labels_1s = d["labels_1s"]
        labels_5s = d["labels_5s"]
        labels_10s = d["labels_10s"]
        N = len(events)

        # Create windows (capped for CPU feasibility)
        starts = list(range(0, N - WINDOW + 1, STRIDE))
        if not starts:
            starts = [0]
        if len(starts) > MAX_WINDOWS_PER_DAY:
            # Uniformly sample to cap
            step = len(starts) // MAX_WINDOWS_PER_DAY
            starts = starts[::step][:MAX_WINDOWS_PER_DAY]

        # Run inference batch by batch
        preds_at_end = np.zeros((N, 3), dtype=np.float64)
        count_at_end = np.zeros(N, dtype=np.float64)

        with torch.no_grad():
            for bi in range(0, len(starts), BATCH_SIZE):
                bs = starts[bi:bi+BATCH_SIZE]
                batch = torch.tensor(
                    np.array([events[s:s+WINDOW] for s in bs]),
                    dtype=torch.float32
                )
                out = model(batch)
                out = out[0] if isinstance(out, tuple) else out
                preds_np = out.cpu().numpy()
                for i, s in enumerate(bs):
                    idx = s + WINDOW - 1
                    preds_at_end[idx] += preds_np[i]
                    count_at_end[idx] += 1

        valid = count_at_end > 0
        final_preds = np.zeros((N, 3), dtype=np.float32)
        final_preds[valid] = (preds_at_end[valid] / count_at_end[valid, np.newaxis]).astype(np.float32)

        # Compute IC
        p1 = final_preds[valid, 0]
        p5 = final_preds[valid, 1]
        p10 = final_preds[valid, 2]
        l1 = labels_1s[:N][valid]
        l5 = labels_5s[:N][valid]
        l10 = labels_10s[:N][valid]

        ic_1s = compute_ic(p1, l1)
        ic_5s = compute_ic(p5, l5)
        ic_10s = compute_ic(p10, l10)

        elapsed = time.time() - t0
        print(f" done ({elapsed:.1f}s, {int(valid.sum())} windows)")
        print(f"    IC_1s={ic_1s:.4f}  IC_5s={ic_5s:.4f}  IC_10s={ic_10s:.4f}")

        results["dates"][date_str] = {
            "IC_1s": round(ic_1s, 5),
            "IC_5s": round(ic_5s, 5),
            "IC_10s": round(ic_10s, 5),
            "n_events": int(N),
            "n_valid_windows": int(valid.sum()),
            "elapsed_s": round(elapsed, 1),
        }

        # Accumulate for concat IC
        all_preds["1s"].append(p1)
        all_preds["5s"].append(p5)
        all_preds["10s"].append(p10)
        all_labels["1s"].append(l1)
        all_labels["5s"].append(l5)
        all_labels["10s"].append(l10)

    # Concat IC across all dates
    print("\n" + "=" * 70)
    print("CONCAT IC (all April dates combined):")
    for h in ["1s", "5s", "10s"]:
        if all_preds[h]:
            concat_p = np.concatenate(all_preds[h])
            concat_l = np.concatenate(all_labels[h])
            ic = compute_ic(concat_p, concat_l)
            results["concat"][f"IC_{h}"] = round(ic, 5)
            print(f"  IC_{h} = {ic:.4f}")
        else:
            results["concat"][f"IC_{h}"] = None
            print(f"  IC_{h} = N/A (no data)")

    # Compare to baseline
    print("\nBASELINE OOT (Feb/Mar training period):")
    print(f"  IC_1s=0.200  IC_5s=0.100  IC_10s=0.070")

    print("\nDECAY ASSESSMENT:")
    for h in ["1s", "5s", "10s"]:
        baseline = results["baseline_oot"][f"IC_{h}"]
        current = results["concat"].get(f"IC_{h}")
        if current is not None:
            pct_change = (current - baseline) / baseline * 100
            status = "OK" if pct_change > -30 else "DECAYED" if pct_change > -50 else "SEVERE DECAY"
            print(f"  IC_{h}: {current:.4f} vs {baseline:.3f} baseline → {pct_change:+.1f}% [{status}]")
            results["concat"][f"decay_pct_{h}"] = round(pct_change, 1)
            results["concat"][f"status_{h}"] = status

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
