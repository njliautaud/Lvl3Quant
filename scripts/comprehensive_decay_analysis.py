"""
Comprehensive per-date decay analysis for CNN-Mamba v2
Runs inference on EVERY date from Mar 6 - Apr 29 to find exact decay onset.
Reports IC, DA, MagCorr at All/Top50/Top25/Top10 confidence bands.
Also logs vol_1s for each date to correlate with IC.

Uses fold_10_best.pt (deployed model: d_model=96, d_state=32, dt_rank=6, n_layers=3)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr

# ============ MODEL ARCHITECTURE (exact match to fold_10_best.pt weights) ============

class MambaBlock(nn.Module):
    def __init__(self, d_model=96, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv-1, groups=d_model)
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_branch = F.silu(x_conv)

        x_dbl = self.x_proj(x_branch)
        dt, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)

        # Selective scan (sequential for correctness)
        y = torch.zeros_like(x_branch)
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)

        for t in range(seq_len):
            dt_t = dt[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]
            x_t = x_branch[:, t, :]

            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = h * dA + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
            y[:, t, :] = y_t + self.D * x_t

        y = y * F.silu(z)
        output = self.out_proj(y)
        return residual + self.dropout(output)


class CNNFrontEnd(nn.Module):
    def __init__(self, input_dim, d_model=96, n_layers=3, kernel_size=5, channels=64):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i in range(n_layers):
            out_ch = channels if i < n_layers - 1 else d_model
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)  # back to (batch, seq, d_model)


class CNNMambaV2(nn.Module):
    def __init__(self, input_dim=25, d_model=96, d_state=32, n_layers=3,
                 dt_rank=6, d_conv=4, dropout=0.1, cnn_channels=64, cnn_kernel=5, cnn_layers=3):
        super().__init__()
        self.cnn = CNNFrontEnd(input_dim, d_model, cnn_layers, cnn_kernel, cnn_channels)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dt_rank, d_conv, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 3 horizons: 1s, 5s, 10s
        )

    def forward(self, x):
        x = self.cnn(x)
        for layer in self.mamba_layers:
            x = layer(x)
        # Take last position prediction
        x = x[:, -1, :]
        return self.head(x)


def load_model(checkpoint_path, device='cuda'):
    """Load model, inferring architecture from weight shapes."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Infer params from weights
    input_dim = state_dict['cnn.net.0.weight'].shape[1]  # First conv input channels
    d_model = state_dict['head.0.weight'].shape[0]  # LayerNorm in head

    # Infer dt_rank from x_proj
    x_proj_out = state_dict['mamba_layers.0.x_proj.weight'].shape[0]
    d_state = state_dict['mamba_layers.0.A_log.weight' if 'mamba_layers.0.A_log.weight' in state_dict
                          else 'mamba_layers.0.A_log'].shape[1]
    dt_rank = x_proj_out - 2 * d_state
    n_layers = sum(1 for k in state_dict if 'mamba_layers.' in k and '.A_log' in k)

    print(f"Model arch: input_dim={input_dim}, d_model={d_model}, d_state={d_state}, "
          f"dt_rank={dt_rank}, n_layers={n_layers}")

    model = CNNMambaV2(input_dim=input_dim, d_model=d_model, d_state=d_state,
                       n_layers=n_layers, dt_rank=dt_rank)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference_on_date(model, data_path, window_size=3000, stride=500, device='cuda', max_windows=500):
    """Run inference on a single date file. Returns predictions and labels."""
    data = np.load(data_path)
    events = data['events']
    labels_1s = data['labels_1s']
    labels_5s = data['labels_5s']
    labels_10s = data['labels_10s']

    n_events = len(events)
    if n_events < window_size:
        return None, None

    # Create windows
    all_preds = []
    all_labels = []

    indices = list(range(0, n_events - window_size, stride))
    if len(indices) > max_windows:
        # Evenly sample
        step = len(indices) // max_windows
        indices = indices[::step][:max_windows]

    batch_size = 32
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]

        windows = []
        batch_labels = []
        for idx in batch_indices:
            end = idx + window_size
            w = events[idx:end].astype(np.float32)
            # Replace NaN with 0
            w = np.nan_to_num(w, nan=0.0)
            windows.append(w)

            # Labels at the END of window
            l1 = labels_1s[end - 1] if end - 1 < len(labels_1s) else np.nan
            l5 = labels_5s[end - 1] if end - 1 < len(labels_5s) else np.nan
            l10 = labels_10s[end - 1] if end - 1 < len(labels_10s) else np.nan
            batch_labels.append([l1, l5, l10])

        windows = np.array(windows)
        x = torch.from_numpy(windows).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            preds = model(x).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(np.array(batch_labels))

    if not all_preds:
        return None, None

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(preds, labels, horizon_idx=0):
    """Compute IC, DA, MagCorr for a specific horizon."""
    p = preds[:, horizon_idx]
    l = labels[:, horizon_idx]
    valid = ~np.isnan(l) & ~np.isnan(p) & np.isfinite(l) & np.isfinite(p)
    p, l = p[valid], l[valid]

    if len(p) < 20:
        return {'ic': 0, 'da': 0, 'mag_corr': 0, 'n': len(p)}

    ic = spearmanr(p, l)[0]
    da = np.mean(np.sign(p) == np.sign(l))
    try:
        mag_corr = np.corrcoef(np.abs(p), np.abs(l))[0, 1]
    except:
        mag_corr = 0

    return {'ic': float(ic), 'da': float(da), 'mag_corr': float(mag_corr), 'n': int(len(p))}


def compute_confidence_metrics(preds, labels, horizon_idx=0):
    """Compute metrics at different confidence bands."""
    p = preds[:, horizon_idx]
    l = labels[:, horizon_idx]
    valid = ~np.isnan(l) & ~np.isnan(p) & np.isfinite(l) & np.isfinite(p)
    p, l = p[valid], l[valid]

    if len(p) < 20:
        return {}

    conf = np.abs(p)
    results = {}

    for band_name, pct in [('All', 0), ('Top50%', 50), ('Top25%', 75), ('Top10%', 90)]:
        if pct > 0:
            thresh = np.percentile(conf, pct)
            mask = conf >= thresh
        else:
            mask = np.ones(len(conf), dtype=bool)

        p_band, l_band = p[mask], l[mask]
        n = len(p_band)

        if n < 10:
            results[band_name] = {'ic': 0, 'da': 0, 'mag_corr': 0, 'n': n}
            continue

        ic = spearmanr(p_band, l_band)[0]
        da = np.mean(np.sign(p_band) == np.sign(l_band))
        try:
            mag_corr = np.corrcoef(np.abs(p_band), np.abs(l_band))[0, 1]
        except:
            mag_corr = 0

        results[band_name] = {'ic': float(ic), 'da': float(da), 'mag_corr': float(mag_corr), 'n': int(n)}

    return results


def main():
    # Paths
    data_dir = Path('/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3')
    checkpoint = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_10_best.pt')
    output_file = Path('/home/nick/Lvl3Quant/output/comprehensive_decay_analysis.json')

    if not checkpoint.exists():
        # Try fold_09
        checkpoint = Path('/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_09_best.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")

    # Load model
    model = load_model(str(checkpoint), device)

    # Get all dates from Mar 6 onward
    all_files = sorted(data_dir.glob('*.npz'))
    target_dates = []
    for f in all_files:
        date_str = f.name[:8]
        if date_str >= '20260306':  # Everything after last training OOT
            target_dates.append(f)

    print(f"\nRunning inference on {len(target_dates)} dates (Mar 6 - Apr 29)")
    print(f"{'Date':<10} {'vol_1s':>7} {'IC_1s':>7} {'IC_5s':>7} {'IC_10s':>7} {'DA_1s':>6} {'DA_T10':>6} {'MagC':>6} {'n':>6}")
    print('-' * 75)

    results = {}

    for date_file in target_dates:
        date_str = date_file.name[:8]
        t0 = time.time()

        # Get vol info
        data = np.load(date_file)
        l1_all = data['labels_1s']
        valid_l1 = l1_all[~np.isnan(l1_all)]
        if len(valid_l1) == 0:
            print(f"{date_str} SKIP (NaN labels)")
            results[date_str] = {'status': 'nan_labels'}
            continue

        vol_1s = float(np.std(valid_l1))
        n_events = len(data['events'])
        del data

        # Run inference
        preds, labels = run_inference_on_date(model, str(date_file), device=device, max_windows=500)

        if preds is None:
            print(f"{date_str} SKIP (too short)")
            results[date_str] = {'status': 'too_short'}
            continue

        # Compute metrics
        m_1s = compute_metrics(preds, labels, 0)
        m_5s = compute_metrics(preds, labels, 1)
        m_10s = compute_metrics(preds, labels, 2)

        # Confidence bands
        conf_1s = compute_confidence_metrics(preds, labels, 0)
        conf_5s = compute_confidence_metrics(preds, labels, 1)
        conf_10s = compute_confidence_metrics(preds, labels, 2)

        da_t10 = conf_1s.get('Top10%', {}).get('da', 0)

        elapsed = time.time() - t0
        print(f"{date_str} {vol_1s:>7.2f} {m_1s['ic']:>7.4f} {m_5s['ic']:>7.4f} {m_10s['ic']:>7.4f} "
              f"{m_1s['da']:>6.3f} {da_t10:>6.3f} {m_1s['mag_corr']:>6.4f} {m_1s['n']:>6} ({elapsed:.1f}s)")

        results[date_str] = {
            'vol_1s': vol_1s,
            'n_events': n_events,
            'n_windows': m_1s['n'],
            'metrics_1s': m_1s,
            'metrics_5s': m_5s,
            'metrics_10s': m_10s,
            'confidence_1s': conf_1s,
            'confidence_5s': conf_5s,
            'confidence_10s': conf_10s,
        }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    valid_dates = {k: v for k, v in results.items() if 'metrics_1s' in v}
    if valid_dates:
        ics = [v['metrics_1s']['ic'] for v in valid_dates.values()]
        vols = [v['vol_1s'] for v in valid_dates.values()]

        print(f"\n=== SUMMARY ===")
        print(f"Dates analyzed: {len(valid_dates)}")
        print(f"IC_1s range: {min(ics):.4f} - {max(ics):.4f}")
        print(f"IC_1s mean: {np.mean(ics):.4f}")
        print(f"Vol_1s range: {min(vols):.2f} - {max(vols):.2f}")

        # Correlation between vol and IC
        corr = np.corrcoef(vols, ics)[0, 1]
        print(f"Correlation(vol_1s, IC_1s): {corr:.4f}")


if __name__ == '__main__':
    main()
