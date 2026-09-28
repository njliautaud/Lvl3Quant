#!/usr/bin/env python3
"""Benchmark: torch.compile on full model vs JIT scan patch."""
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
print("CUDA OPTIMIZATION COMPARISON", flush=True)
print("=" * 60, flush=True)

# 1) JIT patch only
print("\n[1] JIT-patched scan (cuda)", flush=True)
e1 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cuda")
patch_model_with_fast_scan(e1.model)
jit_ms = bench(e1, "JIT-patched")

# 2) torch.compile the whole model
print("\n[2] torch.compile (reduce-overhead mode, cuda)", flush=True)
e2 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cuda")
patch_model_with_fast_scan(e2.model)
try:
    e2.model = torch.compile(e2.model, mode="reduce-overhead")
    compile_ms = bench(e2, "compile+JIT", N=50)
except Exception as ex:
    print(f"  torch.compile failed: {ex}", flush=True)
    compile_ms = None

# 3) torch.compile max-autotune
print("\n[3] torch.compile (max-autotune mode, cuda)", flush=True)
e3 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cuda")
patch_model_with_fast_scan(e3.model)
try:
    e3.model = torch.compile(e3.model, mode="max-autotune")
    autotune_ms = bench(e3, "max-autotune+JIT", N=30)
except Exception as ex:
    print(f"  torch.compile max-autotune failed: {ex}", flush=True)
    autotune_ms = None

# 4) Half precision (fp16) with JIT
print("\n[4] FP16 + JIT-patched (cuda)", flush=True)
e4 = CNNMambaV2Inference(weights_path=weights, stats_path=stats, device="cuda")
patch_model_with_fast_scan(e4.model)
e4.model = e4.model.half()
e4.feat_mean = e4.feat_mean.half()
e4.feat_std = e4.feat_std.half()
# Override predict to use fp16
_orig_predict = e4.predict
def predict_fp16(feat_window):
    xt = torch.tensor(feat_window, dtype=torch.float16, device=e4.device)
    xt = (xt - e4.feat_mean) / e4.feat_std if not e4._skip_normalize else xt
    xt = xt.unsqueeze(0)
    with torch.no_grad():
        out = e4.model(xt)
    if isinstance(out, tuple):
        out = out[0]
    preds = out.squeeze(0).float().cpu().numpy()
    return {"pred_1s": float(preds[0]), "pred_5s": float(preds[1]), "pred_10s": float(preds[2]),
            "confidence_1s": abs(float(preds[0])), "direction": 1 if preds[2] > 0 else -1, "tier": None}
e4.predict = predict_fp16
fp16_ms = bench(e4, "FP16+JIT", N=50)

# Summary
print("\n" + "=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  JIT-patched:           {jit_ms:.1f} ms", flush=True)
if compile_ms:
    print(f"  compile+JIT:           {compile_ms:.1f} ms ({jit_ms/compile_ms:.1f}x vs JIT)", flush=True)
if autotune_ms:
    print(f"  max-autotune+JIT:      {autotune_ms:.1f} ms ({jit_ms/autotune_ms:.1f}x vs JIT)", flush=True)
print(f"  FP16+JIT:              {fp16_ms:.1f} ms ({jit_ms/fp16_ms:.1f}x vs JIT)", flush=True)
print(f"\n  Original vanilla was:  ~550 ms", flush=True)
