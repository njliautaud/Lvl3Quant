#!/usr/bin/env python3
"""Benchmark selective scan: JIT-patched vs vanilla, GPU vs CPU."""
import torch
import torch.jit
import time
import sys
import os

os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
os.environ.setdefault("SKIP_NORMALIZE", "1")

def bench_full_model():
    """Benchmark full CNN Mamba v2 inference: vanilla vs JIT-patched."""
    print("=" * 60)
    print("FULL MODEL INFERENCE BENCHMARK (CNN Mamba v2, 290k params)")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_trading"))
    from cnn_mamba_v2_inference import CNNMambaV2Inference
    from fast_scan import patch_model_with_fast_scan
    import numpy as np

    weights = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "cnn_mamba_v2_smart_v3_mar", "fold_10_best.pt")
    stats = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "output", "cnn_mamba_v2_smart_v3_mar", "fold_09_feature_stats.npz")

    x = np.random.randn(1000, 25).astype(np.float32)

    for device_name in ["cuda", "cpu"]:
        if device_name == "cuda" and not torch.cuda.is_available():
            continue

        print(f"\n  --- {device_name.upper()} ---")

        # Vanilla (no patch)
        engine = CNNMambaV2Inference(
            weights_path=weights, stats_path=stats, device=device_name)
        for _ in range(3):
            _ = engine.predict(x)
        if device_name == "cuda":
            torch.cuda.synchronize()

        N = 50 if device_name == "cuda" else 10
        t0 = time.perf_counter()
        for _ in range(N):
            _ = engine.predict(x)
        if device_name == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        vanilla_ms = (t1 - t0) / N * 1000
        print(f"  Vanilla:     {vanilla_ms:.1f} ms/inference")

        # JIT-patched
        patch_model_with_fast_scan(engine.model)
        for _ in range(3):
            _ = engine.predict(x)
        if device_name == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            _ = engine.predict(x)
        if device_name == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        patched_ms = (t1 - t0) / N * 1000
        speedup = vanilla_ms / patched_ms
        print(f"  JIT-patched: {patched_ms:.1f} ms/inference  ({speedup:.1f}x speedup)")

if __name__ == "__main__":
    bench_full_model()
    print("\nDone.")
