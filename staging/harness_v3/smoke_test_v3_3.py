#!/usr/bin/env python3
"""
smoke_test_v3_3.py — offline validation of v3_3_inference adapter on Jupiter.

Loads the v3.3 ckpt + feature_stats from the canonical paths, runs four
test cases against the V33Inference.predict() API, prints PASS/FAIL per
case, and exits non-zero if any case fails.

This is an OFFLINE smoke test on CPU only — it does NOT touch Razer, the
live feed, or any production process.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from statistics import median

import numpy as np

REPO_ROOT = "/home/jupiter/Lvl3Quant"
HARNESS_DIR = os.path.join(REPO_ROOT, "staging", "harness_v3")
for p in (REPO_ROOT, HARNESS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

WEIGHTS_PATH = os.path.join(
    REPO_ROOT,
    "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt",
)
STATS_PATH = os.path.join(
    REPO_ROOT,
    "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_feature_stats.npz",
)

TRADE_HEADS = (
    "pred_log_ret_1s",
    "pred_log_ret_5s",
    "pred_log_ret_10s",
    "pred_log_ret_30s",
)

results = {}


def _report(test, ok, msg=""):
    status = "PASS" if ok else "FAIL"
    results[test] = ok
    print(f"[{status}] {test}: {msg}", flush=True)


def _summarize_out(out):
    if not isinstance(out, dict):
        return f"type={type(out).__name__}"
    parts = []
    for k, v in out.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:+.4e}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def main():
    print("=" * 72, flush=True)
    print(f"smoke_test_v3_3 :: python={sys.version.split()[0]}", flush=True)
    print(f"  weights={WEIGHTS_PATH}", flush=True)
    print(f"  stats  ={STATS_PATH}", flush=True)
    print("=" * 72, flush=True)

    # ---- Instantiate adapter ----
    try:
        from v3_3_inference import V33Inference  # noqa: E402
        t0 = time.perf_counter()
        eng = V33Inference(
            weights_path=WEIGHTS_PATH,
            stats_path=STATS_PATH,
            device="cpu",
        )
        load_ms = (time.perf_counter() - t0) * 1e3
        print(f"[init] adapter loaded in {load_ms:.1f} ms", flush=True)
    except Exception as e:
        print(f"[FATAL] adapter init failed: {e}", flush=True)
        traceback.print_exc()
        return 2

    rng = np.random.default_rng(seed=42)

    # ---- Test A: flat (1500, 25) input ----
    try:
        x_flat = rng.standard_normal((1500, 25)).astype(np.float32)
        out_a = eng.predict(x_flat)
        assert isinstance(out_a, dict), f"return type was {type(out_a).__name__}"
        present = [k for k in TRADE_HEADS if k in out_a]
        assert len(present) >= 1, f"no trade heads in return: keys={list(out_a.keys())}"

        nan_heads = [k for k in TRADE_HEADS if np.isnan(out_a.get(k, float('nan')))]
        if nan_heads:
            reason = out_a.get("reason", "<missing reason>")
            msg = f"NaN in {nan_heads} reason={reason}"
            _report("A_flat_25", False, msg)
        else:
            print(f"[A] full output: {_summarize_out(out_a)}", flush=True)
            _report("A_flat_25", True,
                    f"keys={list(out_a.keys())} 1s={out_a['pred_log_ret_1s']:+.3e}")
        test_a_out = out_a
    except Exception as e:
        traceback.print_exc()
        _report("A_flat_25", False, f"{type(e).__name__}: {e}")
        test_a_out = None

    # ---- Test B: structured dict input ----
    try:
        e1 = rng.standard_normal((1500, 39)).astype(np.float32)
        e2 = rng.standard_normal((1500, 14)).astype(np.float32)
        e3 = rng.standard_normal((1500, 25)).astype(np.float32)
        batch_dict = {"events_t1": e1, "events_t2": e2, "events_t3": e3}
        out_b = eng.predict(batch_dict)
        assert isinstance(out_b, dict), f"return type was {type(out_b).__name__}"
        nan_heads = [k for k in TRADE_HEADS if np.isnan(out_b.get(k, float('nan')))]
        if nan_heads:
            reason = out_b.get("reason", "<missing reason>")
            _report("B_dict_full", False, f"NaN in {nan_heads} reason={reason}")
        else:
            print(f"[B] full output: {_summarize_out(out_b)}", flush=True)
            _report("B_dict_full", True,
                    f"1s={out_b['pred_log_ret_1s']:+.3e} reason={out_b.get('reason')}")
    except Exception as e:
        traceback.print_exc()
        _report("B_dict_full", False, f"{type(e).__name__}: {e}")

    # ---- Test C: latency over 50 calls ----
    try:
        x_flat = rng.standard_normal((1500, 25)).astype(np.float32)
        # warm-up
        _ = eng.predict(x_flat)
        lat_ms = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = eng.predict(x_flat)
            lat_ms.append((time.perf_counter() - t0) * 1e3)
        lat_ms.sort()
        med = median(lat_ms)
        p95 = lat_ms[int(0.95 * len(lat_ms)) - 1]
        print(f"[C] latency over 50 calls: median={med:.1f} ms  p95={p95:.1f} ms  "
              f"min={lat_ms[0]:.1f}  max={lat_ms[-1]:.1f}", flush=True)
        _report("C_latency", True, f"median={med:.1f}ms p95={p95:.1f}ms")
    except Exception as e:
        traceback.print_exc()
        _report("C_latency", False, f"{type(e).__name__}: {e}")

    # ---- Test D: malformed input ----
    try:
        bad = rng.standard_normal((100, 5)).astype(np.float32)
        out_d = eng.predict(bad)
        assert isinstance(out_d, dict), f"return type was {type(out_d).__name__}"
        all_nan = all(np.isnan(out_d.get(k, 0.0)) for k in TRADE_HEADS)
        reason = out_d.get("reason", "")
        assert all_nan, f"expected NaN heads, got {out_d}"
        assert reason and reason != "ok", f"expected non-ok reason, got '{reason}'"
        _report("D_malformed", True, f"NaN heads + reason='{reason}'")
    except Exception as e:
        traceback.print_exc()
        _report("D_malformed", False, f"{type(e).__name__}: {e}")

    # ---- Summary ----
    print("=" * 72, flush=True)
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} passed  |  results={results}", flush=True)

    if test_a_out is not None and not any(
        np.isnan(test_a_out.get(k, float('nan'))) for k in TRADE_HEADS
    ):
        mags = [abs(test_a_out[k]) for k in TRADE_HEADS]
        print(f"[A] pred magnitudes: 1s={mags[0]:.3e} 5s={mags[1]:.3e} "
              f"10s={mags[2]:.3e} 30s={mags[3]:.3e}", flush=True)
        max_mag = max(mags)
        print(f"[A] max|pred|={max_mag:.3e}  "
              f"({'in_range_<1e-2' if max_mag < 1e-2 else 'OUT_OF_RANGE'})",
              flush=True)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
