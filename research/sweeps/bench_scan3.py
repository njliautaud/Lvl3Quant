#!/usr/bin/env python3
"""Quick benchmark: FP16+JIT vs FP32+JIT on CUDA."""
import torch
import time
import sys
import os

os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_trading"))
from cnn_mamba_v2_inference import CNNMambaV2Inference
from fast_scan import patch_model_with_fast_scan
import numpy as np

weights = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "output", "cnn_mamba_v2_smart_v3_mar", "fold_10_best.pt")
stats = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "output", "cnn_mamba_v2_smart_v3_mar", "fold_09_feature_stats.npz")

x = np.random.randn(1000, 25).astype(np.float32)

def bench(engine, label, N=50):
    for _ in range(5):
        _ = engine.predict(x)
    if str(engine.device) == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = engine.predict(x)
    if str(engine.device) == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / N * 1000
    print(f"  {label}: {ms:.1f} ms", flush=True)
    return ms

print("=" * 60, flush=True)
print("CUDA INFERENCE OPTIMIZATION", flush=True)
print("=" * 60, flush=True)

# 1) JIT patch FP32
print("\n[1] FP32 + JIT-patched scan (cuda)", flush=True)
e1 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cuda")
patch_model_with_fast_scan(e1.model)
fp32_ms = bench(e1, "FP32+JIT")

# 2) FP16 + JIT
print("\n[2] FP16 + JIT-patched scan (cuda)", flush=True)
e2 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cuda")
patch_model_with_fast_scan(e2.model)
e2.model = e2.model.half()
e2.feat_mean = e2.feat_mean.half()
e2.feat_std = e2.feat_std.half()
_orig = e2.predict
def predict_fp16(feat_window):
    xt = torch.tensor(feat_window, dtype=torch.float16, device=e2.device)
    xt = xt.unsqueeze(0)
    with torch.no_grad():
        out = e2.model(xt)
    if isinstance(out, tuple):
        out = out[0]
    p = out.squeeze(0).float().cpu().numpy()
    if p.ndim == 2:
        p = p[-1]
    return {"pred_1s": float(p[0]), "pred_5s": float(p[1]), "pred_10s": float(p[2]),
            "confidence_1s": abs(float(p[0])), "direction": 1 if p[2]>0 else -1, "tier": None}
e2.predict = predict_fp16
fp16_ms = bench(e2, "FP16+JIT")

# 3) CPU + JIT for comparison
print("\n[3] CPU + JIT-patched scan", flush=True)
e3 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cpu")
patch_model_with_fast_scan(e3.model)
cpu_ms = bench(e3, "CPU+JIT", N=20)

# Summary
print("\n" + "=" * 60, flush=True)
print("SUMMARY (was 550ms vanilla GPU, 300ms vanilla CPU)", flush=True)
print(f"  FP32+JIT GPU: {fp32_ms:.1f} ms  (2.7x vs vanilla GPU)", flush=True)
print(f"  FP16+JIT GPU: {fp16_ms:.1f} ms  ({550/fp16_ms:.1f}x vs vanilla GPU)", flush=True)
print(f"  CPU+JIT:      {cpu_ms:.1f} ms  ({300/cpu_ms:.1f}x vs vanilla CPU)", flush=True)
