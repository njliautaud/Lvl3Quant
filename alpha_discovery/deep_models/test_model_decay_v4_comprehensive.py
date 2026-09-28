#!/usr/bin/env python3
"""
Model Decay Test V4 — COMPREHENSIVE, PARALLELIZED, SAVES RAW PREDICTIONS.

Implements the FULL Model Evaluation Checklist (HC #44/63):
- IC, DA, Magnitude IC per horizon (1s, 5s, 10s)
- Conditional IC/DA at top-10%, top-5%, top-1%, top-0.5% confidence bands
- Per-decile hit rate table
- Signal autocorrelation
- Long/Short bias and per-direction IC
- Coverage at each z-score threshold
- Per-date breakdown AND weekly aggregation

Saves raw predictions+labels to disk so metrics can be recomputed without re-running inference.

Usage:
    python3 alpha_discovery/deep_models/test_model_decay_v4_comprehensive.py
    python3 alpha_discovery/deep_models/test_model_decay_v4_comprehensive.py --workers 6
    python3 alpha_discovery/deep_models/test_model_decay_v4_comprehensive.py --models cnn_mamba  # single model
"""

import os
import sys
import time
import json
import pickle
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")
import functools
print = functools.partial(print, flush=True)

# Add project paths
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decay_metrics import (
    compute_all_metrics, format_metrics_table,
    compute_ic, compute_directional_accuracy, compute_magnitude_ic,
    compute_conditional_ic, compute_conditional_da,
    compute_decile_hit_rates, compute_signal_autocorrelation,
    compute_long_short_analysis, compute_coverage
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR_SMART = ROOT / "data" / "processed" / "mbo_events_smart_v3"
DATA_DIR_RAW   = ROOT / "data" / "processed" / "mbo_events"
OUTPUT_DIR     = ROOT / "output" / "decay_v4_comprehensive"

TEST_DATES = [
    "20260316", "20260317", "20260318", "20260319", "20260320",
    "20260322", "20260323", "20260324", "20260325", "20260326", "20260327",
    "20260329", "20260330", "20260331",
    "20260401", "20260402", "20260403",
    "20260405", "20260406", "20260407", "20260408", "20260409", "20260410",
    "20260412", "20260413", "20260414", "20260415", "20260416", "20260417",
    "20260419", "20260420",
    "20260421", "20260422", "20260423", "20260424",
    "20260426", "20260427", "20260428", "20260429",
]

CNNMAMBA_WEIGHTS = ROOT / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
CNNMAMBA_WINDOW  = 3000
CNNMAMBA_STRIDE  = 3000

PATCHTST_WEIGHTS = ROOT / "output" / "patchtst_razer_weights" / "fold_17_best.pt"
PATCHTST_WINDOW  = 500
PATCHTST_STRIDE  = 500

LGBM_MODEL_DIR = ROOT / "models" / "lgbm_60_5_fold35"
LGBM_HORIZONS  = ["1s", "5s", "10s", "30s"]

BASELINE_IC = {
    "CNN-Mamba v2": {"IC_1s": 0.040, "IC_5s": 0.065, "IC_10s": 0.085},
    "PatchTST":     {"IC_1s": 0.035, "IC_5s": 0.055, "IC_10s": 0.075},
    "LGBM Vol":     {"IC_1s": 0.122, "IC_5s": 0.055, "IC_10s": 0.041},
}

MAX_WINDOWS_PER_DAY = 500
DEEP_HORIZONS = ["1s", "5s", "10s"]


# =============================================================================
# Data loading
# =============================================================================

def load_day_data_smart(date_str):
    fpath = DATA_DIR_SMART / f"{date_str}_mbo_events.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")
    d = np.load(fpath)
    return {k: d[k] for k in ["events", "labels_1s", "labels_5s", "labels_10s"]}

def load_day_data_raw(date_str):
    fpath = DATA_DIR_RAW / f"{date_str}_mbo_events.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"Not found: {fpath}")
    d = np.load(fpath)
    result = {k: d[k] for k in ["events", "labels_1s", "labels_5s", "labels_10s"]}
    if "labels_30s" in d:
        result["labels_30s"] = d["labels_30s"]
    return result

def get_days_since(date_str):
    return (datetime.strptime(date_str, "%Y%m%d") - datetime(2026, 3, 15)).days

def get_week_group(date_str):
    days = get_days_since(date_str)
    if days <= 7: return "Week1 (d1-7)"
    elif days <= 14: return "Week2 (d8-14)"
    elif days <= 21: return "Week3 (d15-21)"
    elif days <= 28: return "Week4 (d22-28)"
    elif days <= 35: return "Week5 (d29-35)"
    elif days <= 42: return "Week6 (d36-42)"
    else: return "Week7+ (d43+)"


# =============================================================================
# Model builders (inline for multiprocessing pickling)
# =============================================================================

def _build_cnn_mamba_v2():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
            xz = self.in_proj(x); x_branch, z = xz.chunk(2, dim=-1)
            x_conv = F.pad(x_branch.transpose(1,2).contiguous(), (self.d_conv-1, 0))
            x_branch = F.silu(self.conv1d(x_conv).transpose(1,2).contiguous())
            x_proj = self.x_proj(x_branch)
            dt_x, B_sel, C_sel = x_proj[:,:,:self.dt_rank], x_proj[:,:,self.dt_rank:self.dt_rank+self.d_state], x_proj[:,:,self.dt_rank+self.d_state:]
            dt = F.softplus(self.dt_proj(dt_x)); A = -torch.exp(self.A_log)
            y = self._scan(x_branch, dt, A, B_sel, C_sel, self.D, time_delta)
            return self.out_proj(y * F.silu(z))
        def _scan(self, x, dt, A, B, C, D, time_delta=None):
            batch, seq_len, d_inner = x.shape; d_state = A.shape[1]
            dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
            if time_delta is not None:
                td = time_delta.unsqueeze(-1).unsqueeze(-1)
                dA = dA * torch.exp(-F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0) * td.abs())
            dBx = (B.unsqueeze(2) * dt.unsqueeze(-1)) * x.unsqueeze(-1)
            h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype); outputs = []
            for t in range(seq_len):
                h = dA[:,t] * h + dBx[:,t]
                outputs.append(torch.einsum("bn,bdn->bd", C[:,t], h))
            return torch.stack(outputs, dim=1) + x * D.unsqueeze(0).unsqueeze(0)

    class MambaBlock(nn.Module):
        def __init__(self, d_model, d_state=32, dt_rank=6, d_conv=4, dropout=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(d_model); self.ssm = SelectiveSSM(d_model, d_state, dt_rank, d_conv); self.dropout = nn.Dropout(dropout)
        def forward(self, x, time_delta=None):
            return x + self.dropout(self.ssm(self.norm(x), time_delta))

    class CNNMambaV2(nn.Module):
        CNN_KERNELS = [3, 7, 15, 31]
        def __init__(self, n_smart_features=25, d_model=96, d_state=32, n_layers=3, dt_rank=6, d_conv=4, dropout=0.1, n_targets=3, feature_mlp_hidden=128, feature_mlp_out=64, cnn_channels_per_scale=16):
            super().__init__()
            self.d_model, self.n_targets = d_model, n_targets
            self.feature_mlp = nn.Sequential(nn.Linear(n_smart_features, feature_mlp_hidden), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(feature_mlp_hidden, feature_mlp_out), nn.LayerNorm(feature_mlp_out))
            self.cnn_kernels = self.CNN_KERNELS
            self.temporal_cnns = nn.ModuleList([nn.Sequential(nn.Conv1d(n_smart_features, cnn_channels_per_scale, kernel_size=k, padding=0, bias=True), nn.GELU()) for k in self.cnn_kernels])
            self.cnn_out_dim = cnn_channels_per_scale * len(self.cnn_kernels)
            self.fusion_proj = nn.Sequential(nn.Linear(feature_mlp_out + self.cnn_out_dim, d_model), nn.LayerNorm(d_model))
            self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, dt_rank, d_conv, dropout) for _ in range(n_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_targets))
        def forward(self, smart):
            B, L, _ = smart.shape; time_delta = smart[:,:,0]
            feat_out = self.feature_mlp(smart); x_t = smart.transpose(1,2)
            cnn_out = torch.cat([conv(F.pad(x_t, (k-1,0))).transpose(1,2) for k, conv in zip(self.cnn_kernels, self.temporal_cnns)], dim=-1)
            x = self.fusion_proj(torch.cat([feat_out, cnn_out], dim=-1))
            for block in self.blocks: x = block(x, time_delta)
            return self.head(self.final_norm(x)[:, -1, :])

    device = torch.device("cpu")
    ckpt = torch.load(str(CNNMAMBA_WEIGHTS), map_location=device, weights_only=False)
    arch, state = ckpt.get("arch", {}), ckpt["model_state"]
    d_state = arch.get("d_state", 32)
    dt_rank = state["blocks.0.ssm.x_proj.weight"].shape[0] - 2*d_state if "blocks.0.ssm.x_proj.weight" in state else 6
    model = CNNMambaV2(d_model=arch.get("d_model",96), d_state=d_state, n_layers=arch.get("n_layers",3), dt_rank=dt_rank, d_conv=arch.get("d_conv",4), dropout=arch.get("dropout",0.1), n_targets=3, feature_mlp_hidden=arch.get("feature_mlp_hidden",128), feature_mlp_out=arch.get("feature_mlp_out",64), cnn_channels_per_scale=arch.get("cnn_channels_per_scale",16))
    model.load_state_dict(state, strict=False); model.to(device).eval()
    return model

def _build_patchtst():
    import torch
    os.environ["TST_FEATURE_SET"] = "smart_v3"; os.environ["SKIP_NORMALIZE"] = "1"; os.environ["DISABLE_MLFLOW"] = "1"
    _dir = str(Path(__file__).resolve().parent)
    if _dir not in sys.path: sys.path.insert(0, _dir)
    import logging; logging.disable(logging.INFO)
    from train_event_patchtst import PatchTST; logging.disable(logging.NOTSET)
    device = torch.device("cpu")
    ckpt = torch.load(str(PATCHTST_WEIGHTS), map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = PatchTST(n_features=arch.get("n_features",25), patch_size=arch.get("patch_size",25), d_model=arch.get("d_model",256), n_heads=arch.get("n_heads",4), head_dim=arch.get("head_dim",64), n_layers=arch.get("n_layers",4), ffn_dim=arch.get("ffn_dim",1024), dropout=arch.get("dropout",0.1), n_targets=3, window_size=arch.get("window_size",500))
    model.load_state_dict(ckpt["model_state"], strict=False); model.to(device).eval()
    return model


# =============================================================================
# LGBM features (same as v3)
# =============================================================================

def compute_lgbm_features(ev):
    N = len(ev)
    td, et, side, price, qty, sprd = [ev[:,i].astype('f4') for i in range(6)]
    W20, W50, W100, W200, W500 = [np.ones(w,'f4')/w for w in [20,50,100,200,500]]
    is_t, is_c, is_a = [(et==v).astype('f4') for v in [3,1,0]]
    is_bid, is_ask = (side==0).astype('f4'), (side==1).astype('f4')
    ofi = is_t*is_ask*qty - is_t*is_bid*qty
    rofi = np.convolve(ofi,W100,'full')[:N]
    casym = np.convolve(is_c*is_bid - is_c*is_ask, W100,'full')[:N]
    dens = np.convolve(np.where(td>1e-9, 1./(td+1e-6), 1.).astype('f4'), W50,'full')[:N]
    pdiff = np.zeros(N,'f4'); pdiff[20:] = price[20:] - price[:-20]
    pmom = np.convolve(pdiff, W20,'full')[:N]
    qpmom = np.convolve(qty*np.sign(pdiff), W20,'full')[:N]
    cr = np.cumsum(ofi).astype('f4'); crma = np.convolve(cr, W500,'full')[:N]
    crstd = np.sqrt(np.maximum(np.convolve(cr**2, W500,'full')[:N] - crma**2, 1e-8))
    cd = (cr - crma) / (crstd + 1e-6)
    rd5 = np.convolve(ofi, W500,'full')[:N]; os20 = np.convolve(ofi, W20,'full')[:N]
    crate = np.convolve(is_c, W100,'full')[:N]; trate = np.convolve(is_t, W50,'full')[:N]
    aasym = np.convolve(is_a*is_bid - is_a*is_ask, W100,'full')[:N]
    sdiff = np.zeros(N,'f4'); sdiff[1:] = sprd[1:] - sprd[:-1]
    svel = np.convolve(sdiff, W50,'full')[:N]
    baq, aaq = is_a*is_bid*qty, is_a*is_ask*qty
    qai = np.convolve(baq-aaq, W100,'full')[:N] / (np.convolve(baq+aaq, W100,'full')[:N] + 1e-6)
    psm = np.convolve(np.sign(pdiff), W200,'full')[:N]
    br = np.convolve(is_t*is_bid, W20,'full')[:N] * is_a*is_bid
    ar = np.convolve(is_t*is_ask, W20,'full')[:N] * is_a*is_ask
    fr = np.convolve(br+ar, W20,'full')[:N]
    d = np.stack([rofi,casym,dens,pmom,qpmom,cd,rd5,os20,crate,trate,aasym,svel,qai,psm,fr], axis=1)
    return np.concatenate([ev, d], axis=1).astype('f4')


# =============================================================================
# Worker: process one date for a deep model — SAVES RAW PREDICTIONS
# =============================================================================

def _process_deep_date(args):
    model_name, date_str, window, stride, batch_size, threads = args
    import torch; torch.set_num_threads(threads)
    t0 = time.time()

    model = _build_cnn_mamba_v2() if model_name == "CNN-Mamba v2" else _build_patchtst()

    try:
        data = load_day_data_smart(date_str)
    except FileNotFoundError as e:
        return {"date": date_str, "model": model_name, "error": str(e)}

    events = data["events"]; N, F = events.shape; n_targets = 3
    if N < window:
        padded = np.zeros((window, F), dtype=np.float32); padded[-N:] = events; events = padded; N = window

    starts = list(range(0, N - window + 1, stride))
    if not starts: starts = [0]
    if len(starts) > MAX_WINDOWS_PER_DAY:
        step = len(starts) // MAX_WINDOWS_PER_DAY
        starts = starts[::step][:MAX_WINDOWS_PER_DAY]

    pred_sum = np.zeros((N, n_targets), dtype=np.float64)
    pred_count = np.zeros(N, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for bi in range(0, len(starts), batch_size):
            bs = starts[bi:bi+batch_size]
            batch = torch.tensor(np.array([events[s:s+window] for s in bs]), dtype=torch.float32)
            out = model(batch); out = out[0] if isinstance(out, tuple) else out
            preds_np = out.cpu().numpy()
            for i, s in enumerate(bs):
                pred_sum[s+window-1] += preds_np[i]; pred_count[s+window-1] += 1

    valid = pred_count > 0
    result_preds = np.zeros((N, n_targets), dtype=np.float32)
    result_preds[valid] = (pred_sum[valid] / pred_count[valid, np.newaxis]).astype(np.float32)

    # Save raw predictions + labels
    save_dir = OUTPUT_DIR / model_name.replace(" ", "_") / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "predictions.npz",
        preds=result_preds[valid],
        labels_1s=data["labels_1s"][:len(result_preds)][valid],
        labels_5s=data["labels_5s"][:len(result_preds)][valid],
        labels_10s=data["labels_10s"][:len(result_preds)][valid],
        valid_indices=np.where(valid)[0],
    )

    # Compute ALL metrics per horizon
    days_since = get_days_since(date_str)
    date_result = {"date": date_str, "model": model_name, "days_since_training": days_since,
                   "week_group": get_week_group(date_str), "n_events": int(len(data["events"])),
                   "n_windows": len(starts), "elapsed_s": 0, "metrics_by_horizon": {}}

    for hi, horizon in enumerate(DEEP_HORIZONS):
        labels = data[f"labels_{horizon}"][:len(result_preds)]
        p, l = result_preds[valid, hi], labels[valid]
        date_result["metrics_by_horizon"][horizon] = compute_all_metrics(p, l, horizon)

    date_result["elapsed_s"] = round(time.time() - t0, 1)
    return date_result


def _process_lgbm_date(args):
    date_str, threads = args
    t0 = time.time()

    lgbm_models = {}
    for hz in LGBM_HORIZONS:
        pkl_path = LGBM_MODEL_DIR / f"labels_{hz}_lgbm.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f: lgbm_models[hz] = pickle.load(f)

    try:
        data = load_day_data_raw(date_str)
    except FileNotFoundError as e:
        return {"date": date_str, "model": "LGBM Vol", "error": str(e)}

    features = compute_lgbm_features(data["events"])
    horizons = list(lgbm_models.keys()); N = len(data["events"])
    preds = np.zeros((N, len(horizons)), dtype=np.float32)
    for hi, hz in enumerate(horizons):
        preds[:, hi] = lgbm_models[hz].predict(features).astype(np.float32)

    # Save raw predictions
    save_dir = OUTPUT_DIR / "LGBM_Vol" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dict = {"preds": preds}
    for hz in horizons:
        lk = f"labels_{hz}"
        if lk in data: save_dict[lk] = data[lk]
    np.savez_compressed(save_dir / "predictions.npz", **save_dict)

    days_since = get_days_since(date_str)
    date_result = {"date": date_str, "model": "LGBM Vol", "days_since_training": days_since,
                   "week_group": get_week_group(date_str), "n_events": N,
                   "elapsed_s": 0, "metrics_by_horizon": {}}

    for hi, hz in enumerate(horizons):
        lk = f"labels_{hz}"
        if lk not in data: continue
        labels = data[lk][:N]; finite = np.isfinite(labels)
        p, l = preds[finite, hi], labels[finite]
        date_result["metrics_by_horizon"][hz] = compute_all_metrics(p, l, hz)

    date_result["elapsed_s"] = round(time.time() - t0, 1)
    return date_result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (6 default, gentler on RAM)")
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--models", type=str, default="all", help="cnn_mamba, patchtst, lgbm, or all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_cores = os.cpu_count() or 16
    print(f"System: {total_cores} cores | Workers: {args.workers} x {args.threads_per_worker} threads")
    print(f"Dates: {len(TEST_DATES)} | Output: {OUTPUT_DIR}")
    print(f"Models: {args.models}")
    print()

    start_time = time.time()
    all_results = defaultdict(dict)

    models_to_run = []
    if args.models in ("all", "cnn_mamba"):
        models_to_run.append(("CNN-Mamba v2", CNNMAMBA_WINDOW, CNNMAMBA_STRIDE, 4))
    if args.models in ("all", "patchtst"):
        models_to_run.append(("PatchTST", PATCHTST_WINDOW, PATCHTST_STRIDE, 32))

    # Phase 1: Deep models
    for model_name, window, stride, bsz in models_to_run:
        print(f"\n{'='*80}")
        print(f"  {model_name} — {len(TEST_DATES)} dates x {args.workers} workers (COMPREHENSIVE)")
        print(f"{'='*80}")
        t0_model = time.time()

        tasks = [(model_name, d, window, stride, bsz, args.threads_per_worker) for d in TEST_DATES]
        completed = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_deep_date, t): t[1] for t in tasks}
            for future in as_completed(futures):
                date_str = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    if result and "error" not in result:
                        all_results[model_name][date_str] = result
                        m10s = result["metrics_by_horizon"].get("10s", {})
                        ic = m10s.get("IC", 0); da = m10s.get("DA", 0.5)
                        mag_ic = m10s.get("magnitude_IC", 0)
                        cic1 = m10s.get("condIC_top1pct", 0)
                        cda1 = m10s.get("condDA_top1pct", 0.5)
                        print(f"  [{completed}/{len(TEST_DATES)}] {date_str} d+{result['days_since_training']} "
                              f"IC={ic:+.4f} DA={da:.3f} MagIC={mag_ic:+.4f} "
                              f"Top1%IC={cic1:+.4f} Top1%DA={cda1:.3f} ({result['elapsed_s']:.0f}s)")
                    elif result:
                        print(f"  [{completed}/{len(TEST_DATES)}] {date_str} SKIP: {result.get('error','?')}")
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{len(TEST_DATES)}] {date_str} ERROR: {e}")

        print(f"\n  {model_name} done: {time.time()-t0_model:.0f}s")

    # Phase 2: LGBM
    if args.models in ("all", "lgbm"):
        print(f"\n{'='*80}")
        print(f"  LGBM Vol — {len(TEST_DATES)} dates x {args.workers} workers (COMPREHENSIVE)")
        print(f"{'='*80}")
        t0_lgbm = time.time()

        tasks = [(d, args.threads_per_worker) for d in TEST_DATES]
        completed = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_lgbm_date, t): t[0] for t in tasks}
            for future in as_completed(futures):
                date_str = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    if result and "error" not in result:
                        all_results["LGBM Vol"][date_str] = result
                        m10s = result["metrics_by_horizon"].get("10s", {})
                        ic = m10s.get("IC", 0); da = m10s.get("DA", 0.5)
                        mag_ic = m10s.get("magnitude_IC", 0)
                        print(f"  [{completed}/{len(TEST_DATES)}] {date_str} d+{result['days_since_training']} "
                              f"IC={ic:+.4f} DA={da:.3f} MagIC={mag_ic:+.4f} ({result['elapsed_s']:.0f}s)")
                    elif result:
                        print(f"  [{completed}/{len(TEST_DATES)}] {date_str} SKIP: {result.get('error','?')}")
                except Exception as e:
                    completed += 1
                    print(f"  [{completed}/{len(TEST_DATES)}] {date_str} ERROR: {e}")

        print(f"\n  LGBM Vol done: {time.time()-t0_lgbm:.0f}s")

    # =========================================================================
    # Comprehensive Summary Tables
    # =========================================================================
    all_model_names = [mn for mn in ["CNN-Mamba v2", "PatchTST", "LGBM Vol"] if mn in all_results]
    week_order = ["Week1 (d1-7)", "Week2 (d8-14)", "Week3 (d15-21)", "Week4 (d22-28)",
                  "Week5 (d29-35)", "Week6 (d36-42)", "Week7+ (d43+)"]

    print(f"\n\n{'='*120}")
    print("COMPREHENSIVE DECAY ANALYSIS — ALL METRICS × ALL HORIZONS × ALL CONFIDENCE BANDS")
    print(f"{'='*120}")

    for mn in all_model_names:
        horizons = DEEP_HORIZONS if mn != "LGBM Vol" else LGBM_HORIZONS

        print(f"\n{'─'*120}")
        print(f"  MODEL: {mn}")
        print(f"{'─'*120}")

        # Per-date table with key metrics
        header = (f"  {'Date':<10} {'Day+':>4} {'Hz':<4} {'IC':>7} {'DA':>6} {'MagIC':>7} "
                  f"{'T10%IC':>7} {'T5%IC':>7} {'T1%IC':>7} {'T0.5%IC':>8} "
                  f"{'T1%DA':>6} {'L/S%':>7} {'AC_1':>6}")
        print(header)
        print("  " + "-" * 115)

        for d in TEST_DATES:
            if d not in all_results.get(mn, {}):
                continue
            r = all_results[mn][d]
            for hz in horizons:
                if hz not in r.get("metrics_by_horizon", {}):
                    continue
                m = r["metrics_by_horizon"][hz]
                ls = m.get("long_short_analysis", {}); ac = m.get("signal_autocorrelation", {})
                print(f"  {d:<10} {r['days_since_training']:>4} {hz:<4} "
                      f"{m['IC']:>+7.4f} {m['DA']:>6.3f} {m['magnitude_IC']:>+7.4f} "
                      f"{m['condIC_top10pct']:>+7.4f} {m['condIC_top5pct']:>+7.4f} "
                      f"{m['condIC_top1pct']:>+7.4f} {m.get('condIC_top0.5pct',0):>+8.4f} "
                      f"{m.get('condDA_top1pct',0.5):>6.3f} {ls.get('pct_long',50):>5.1f}%L "
                      f"{ac.get('lag_1',0):>+6.3f}")

        # Weekly aggregation
        print(f"\n  WEEKLY DECAY CURVE — {mn}")
        print(f"  {'Week':<20} {'Hz':<4} {'AvgIC':>7} {'AvgDA':>6} {'AvgMagIC':>8} "
              f"{'AvgT1%IC':>8} {'AvgT1%DA':>8} {'n_dates':>7} {'vs_BL':>8}")
        print("  " + "-" * 95)

        for week in week_order:
            week_dates = [d for d in TEST_DATES if d in all_results.get(mn, {}) and all_results[mn][d].get("week_group") == week]
            if not week_dates:
                continue
            for hz in horizons:
                ics, das, mags, cic1s, cda1s = [], [], [], [], []
                for d in week_dates:
                    m = all_results[mn][d].get("metrics_by_horizon", {}).get(hz, {})
                    if m:
                        ics.append(m["IC"]); das.append(m["DA"]); mags.append(m["magnitude_IC"])
                        cic1s.append(m.get("condIC_top1pct", 0)); cda1s.append(m.get("condDA_top1pct", 0.5))
                if ics:
                    bl = BASELINE_IC.get(mn, {}).get(f"IC_{hz}", 0)
                    avg_ic = np.mean(ics)
                    vs_bl = ((avg_ic - bl) / abs(bl) * 100) if bl else 0
                    print(f"  {week:<20} {hz:<4} {avg_ic:>+7.4f} {np.mean(das):>6.3f} {np.mean(mags):>+8.4f} "
                          f"{np.mean(cic1s):>+8.4f} {np.mean(cda1s):>8.3f} {len(ics):>7} {vs_bl:>+7.0f}%")
            print("  " + "-" * 95)

    # Decay verdict
    print(f"\n{'='*80}")
    print("DECAY VERDICT & RETRAIN RECOMMENDATION")
    print(f"{'='*80}")
    for mn in all_model_names:
        model_dates = [all_results[mn][d] for d in TEST_DATES if d in all_results.get(mn, {})]
        if not model_dates:
            continue
        for hz in (DEEP_HORIZONS if mn != "LGBM Vol" else ["1s", "5s", "10s"]):
            all_ics = [r["metrics_by_horizon"][hz]["IC"] for r in model_dates if hz in r.get("metrics_by_horizon", {})]
            all_mag = [r["metrics_by_horizon"][hz]["magnitude_IC"] for r in model_dates if hz in r.get("metrics_by_horizon", {})]
            all_cic1 = [r["metrics_by_horizon"][hz].get("condIC_top1pct", 0) for r in model_dates if hz in r.get("metrics_by_horizon", {})]
            if not all_ics:
                continue
            mean_ic = np.mean(all_ics); mean_mag = np.mean(all_mag); mean_cic1 = np.mean(all_cic1)
            bl = BASELINE_IC.get(mn, {}).get(f"IC_{hz}", 0)
            decay_pct = ((mean_ic - bl) / abs(bl) * 100) if bl else 0
            print(f"  {mn:<16} {hz:<4}: meanIC={mean_ic:+.4f} ({decay_pct:+.0f}% vs BL={bl:+.4f}) "
                  f"magIC={mean_mag:+.4f} top1%IC={mean_cic1:+.4f}")

    # Save full results as JSON
    out_json = OUTPUT_DIR / "decay_v4_results.json"
    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_safe(i) for i in obj]
        return obj
    with open(out_json, "w") as f:
        json.dump(_safe(dict(all_results)), f, indent=2)
    print(f"\nFull results saved to {out_json}")
    print(f"Raw predictions saved to {OUTPUT_DIR}/*/YYYYMMDD/predictions.npz")
    print(f"\nTotal runtime: {(time.time()-start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
