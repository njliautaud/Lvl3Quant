#!/usr/bin/env python3
"""
Monday Dry-Run Smoke Test — Full End-to-End Pipeline on CPU
=============================================================

Verifies the complete inference pipeline BEFORE Monday's live deployment on Razer:

  MBO events → CNN-Mamba v2 inference → meta-model filter → stacked filter → trade/no-trade

Tests on Jupiter (CPU-only) using recorded MBO data. Reports:
  - Component load success/failure
  - Latency per event and per batch
  - Filter pass rates at each gate
  - Memory usage
  - Any errors or warnings

Usage:
    python -m live_trading.monday_dryrun_test
    python live_trading/monday_dryrun_test.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil  # type: ignore

# Ensure Lvl3Quant is on the path
_THIS_DIR = Path(__file__).resolve().parent
_LVL3 = _THIS_DIR.parent
sys.path.insert(0, str(_LVL3))

import torch

# ── Configuration ──────────────────────────────────────────────────────────────

# CNN-Mamba v2 weights (production fold)
CNN_MAMBA_WEIGHTS = _LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"
CNN_MAMBA_FEATURE_STATS = _LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"

# Meta-model weights directory (15 folds)
META_WEIGHTS_DIR = _LVL3 / "output" / "meta_production_v1" / "weights"

# Stacked filter config
STACKED_CONFIG = _LVL3 / "live_trading" / "configs" / "monday_stacked_v1.json"

# MBO event data (use a recent day for realism)
MBO_DATA_DIR = _LVL3 / "data" / "processed" / "mbo_events_smart_v3"

# Test parameters
EVENT_WINDOW = 1000  # CNN-Mamba v2 actual window size (from arch dict in checkpoint)
N_TEST_EVENTS = 5000 # Events to process (~100 predictions at stride=50, enough to test all gates)
PRED_STRIDE = 50     # Predict every N events (matches ~250ms at live event rate)
DEVICE = "cpu"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_mem_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def fmt_time(ms: float) -> str:
    if ms < 1.0:
        return f"{ms * 1000:.0f}us"
    return f"{ms:.2f}ms"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# ── Test Components ────────────────────────────────────────────────────────────

class DryRunResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: list[str] = []

    def record_pass(self, msg: str):
        self.passed += 1
        ok(msg)

    def record_fail(self, msg: str):
        self.failed += 1
        self.errors.append(msg)
        fail(msg)

    def record_warn(self, msg: str):
        self.warnings += 1
        warn(msg)


def test_load_cnn_mamba(results: DryRunResults) -> torch.nn.Module | None:
    """Load CNN-Mamba v2 model weights onto CPU.

    Uses the correct architecture from batch_inference_cpu.py which matches
    the actual fold_10_best.pt checkpoint (Feature MLP + Multi-Scale Temporal
    CNN + Mamba backbone with time-delta gating).

    NOTE: The file live_trading/cnn_mamba_v2_model.py has an OLDER architecture
    (plain CNN front-end) that does NOT match fold_10_best.pt. The production
    weights use feature_mlp + temporal_cnns + fusion_proj, loaded by
    alpha_discovery/execution/batch_inference_cpu.py:load_cnn_mamba().
    """
    section("1. CNN-Mamba v2 Model Load")
    mem_before = get_mem_mb()
    t0 = time.perf_counter()

    try:
        # Use the correct loader that matches fold_10_best.pt architecture
        sys.path.insert(0, str(_LVL3 / "alpha_discovery" / "execution"))
        from batch_inference_cpu import load_cnn_mamba

        if not CNN_MAMBA_WEIGHTS.exists():
            results.record_fail(f"Weights file not found: {CNN_MAMBA_WEIGHTS.name}")
            return None

        model = load_cnn_mamba(str(CNN_MAMBA_WEIGHTS))
        model.to(DEVICE)
        dt = (time.perf_counter() - t0) * 1000
        mem_after = get_mem_mb()

        n_params = sum(p.numel() for p in model.parameters())
        results.record_pass(f"Loaded CNN-Mamba v2: {n_params:,} params")
        info(f"Load time: {fmt_time(dt)}")
        info(f"Memory delta: +{mem_after - mem_before:.1f} MB (total RSS: {mem_after:.0f} MB)")

        # Verify model is in eval mode
        if model.training:
            results.record_warn("Model not in eval mode — switching")
            model.eval()
        else:
            results.record_pass("Model is in eval mode")

        # Quick forward pass sanity check with correct window size
        dummy = torch.randn(1, EVENT_WINDOW, 25)
        with torch.no_grad():
            out = model(dummy)
        if out.shape == (1, 3):
            results.record_pass(f"Forward pass OK — output shape {tuple(out.shape)} (1s, 5s, 10s)")
        else:
            results.record_fail(f"Unexpected output shape: {tuple(out.shape)}, expected (1, 3)")

        return model

    except Exception as e:
        results.record_fail(f"CNN-Mamba load failed: {e}")
        traceback.print_exc()
        return None


def test_load_feature_stats(results: DryRunResults) -> tuple[np.ndarray, np.ndarray] | None:
    """Load feature normalization stats."""
    section("2. Feature Normalization Stats")

    try:
        if not CNN_MAMBA_FEATURE_STATS.exists():
            results.record_warn(f"Feature stats not found: {CNN_MAMBA_FEATURE_STATS.name} — will use raw features")
            return None

        stats = np.load(str(CNN_MAMBA_FEATURE_STATS))
        feat_mean = stats["mean"].astype(np.float32)
        feat_std = stats["std"].astype(np.float32)

        results.record_pass(f"Feature stats loaded: mean/std shapes {feat_mean.shape}")

        # Check for SKIP_NORMALIZE pattern (smart_v3 features are pre-normalized)
        is_skip = (np.allclose(feat_mean, 0.0, atol=1e-3) and np.allclose(feat_std, 1.0, atol=1e-3))
        if is_skip:
            results.record_pass("Feature stats are SKIP_NORMALIZE (mean~0, std~1) — smart_v3 pre-normalized")
        else:
            info(f"Feature stats have non-trivial normalization (mean range: [{feat_mean.min():.3f}, {feat_mean.max():.3f}])")

        # Sanity: std should be > 0 everywhere
        zero_std = (feat_std < 1e-8).sum()
        if zero_std > 0:
            results.record_warn(f"{zero_std} features have near-zero std — could cause NaN")
        else:
            results.record_pass("All feature stds are non-zero")

        return feat_mean, feat_std

    except Exception as e:
        results.record_fail(f"Feature stats load failed: {e}")
        return None


def test_load_meta_model(results: DryRunResults) -> list | None:
    """Load meta-model ensemble (15 folds)."""
    section("3. Meta-Model Ensemble (15 folds)")
    mem_before = get_mem_mb()
    t0 = time.perf_counter()

    try:
        from live_trading.stacked_filter import ProductionMetaMLP

        if not META_WEIGHTS_DIR.exists():
            results.record_fail(f"Meta weights dir not found")
            return None

        pt_files = sorted(META_WEIGHTS_DIR.glob("fold_*.pt"))
        if not pt_files:
            results.record_fail("No fold_*.pt files found in meta weights dir")
            return None

        info(f"Found {len(pt_files)} fold weight files")

        models = []
        load_errors = 0
        for pt_file in pt_files:
            try:
                ckpt = torch.load(pt_file, map_location=DEVICE, weights_only=False)
                input_dim = ckpt["input_dim"]
                model = ProductionMetaMLP(input_dim, dropout=0.0)
                model.load_state_dict(ckpt["model_state"])
                model.to(DEVICE)
                model.eval()
                norm_mean = ckpt["norm_mean"].astype(np.float32)
                norm_std = ckpt["norm_std"].astype(np.float32)
                models.append((model, norm_mean, norm_std))
            except Exception as e:
                load_errors += 1
                results.record_warn(f"Failed to load {pt_file.name}: {e}")

        dt = (time.perf_counter() - t0) * 1000
        mem_after = get_mem_mb()

        if models:
            results.record_pass(f"Loaded {len(models)}/{len(pt_files)} meta-model folds")
            if load_errors:
                results.record_warn(f"{load_errors} folds failed to load")
            info(f"Load time: {fmt_time(dt)}")
            info(f"Memory delta: +{mem_after - mem_before:.1f} MB")
            info(f"Input dim: {models[0][1].shape[0]}")

            # Quick inference test
            dummy_feat = np.random.randn(29).astype(np.float32)
            m, nm, ns = models[0]
            feat_n = (dummy_feat - nm) / (ns + 1e-8)
            with torch.no_grad():
                out = m(torch.from_numpy(feat_n).unsqueeze(0))
            results.record_pass(f"Meta-model forward pass OK — output: {out.item():.6f}")
        else:
            results.record_fail("No meta-model folds loaded successfully")
            return None

        return models

    except Exception as e:
        results.record_fail(f"Meta-model load failed: {e}")
        traceback.print_exc()
        return None


def test_load_stacked_filter(results: DryRunResults):
    """Load the full stacked filter with all 3 gates."""
    section("4. Stacked Filter (3-gate pipeline)")
    mem_before = get_mem_mb()
    t0 = time.perf_counter()

    try:
        from live_trading.stacked_filter import StackedFilter

        if not STACKED_CONFIG.exists():
            results.record_fail(f"Config not found: {STACKED_CONFIG.name}")
            return None

        sf = StackedFilter(
            config_path=str(STACKED_CONFIG),
            preset="balanced",
            meta_weights_dir=str(META_WEIGHTS_DIR),
            device=DEVICE,
        )
        dt = (time.perf_counter() - t0) * 1000
        mem_after = get_mem_mb()

        results.record_pass(f"StackedFilter initialized: {sf}")
        info(f"Load time: {fmt_time(dt)}")
        info(f"Memory delta: +{mem_after - mem_before:.1f} MB")
        info(f"Config: {json.dumps(sf.config_summary, indent=2)}")

        # Test a single should_trade call with synthetic data
        fake_result = sf.should_trade(
            pred_1s=-0.15,
            pred_5s=-0.08,
            pred_10s=-0.04,
            event_features=np.random.randn(25).astype(np.float32),
            ofi_book_1s=-50.0,
        )
        results.record_pass(f"should_trade() call OK — result: should_trade={fake_result.should_trade}")
        info(f"Gate reasons: {json.dumps({k: v.get('passed', '?') for k, v in fake_result.gate_reasons.items()})}")

        return sf

    except Exception as e:
        results.record_fail(f"StackedFilter load failed: {e}")
        traceback.print_exc()
        return None


def test_load_mbo_data(results: DryRunResults) -> np.ndarray | None:
    """Load a sample of recorded MBO event data."""
    section("5. MBO Event Data Load")

    try:
        if not MBO_DATA_DIR.exists():
            results.record_fail("MBO data directory not found")
            return None

        # Pick the most recent available day
        npz_files = sorted(MBO_DATA_DIR.glob("*_mbo_events.npz"))
        npz_files = [f for f in npz_files if not str(f).startswith(str(MBO_DATA_DIR / "_"))]  # skip backup dirs

        if not npz_files:
            results.record_fail("No MBO event .npz files found")
            return None

        data_file = npz_files[-1]  # Most recent
        info(f"Loading: {data_file.name}")

        t0 = time.perf_counter()
        data = np.load(str(data_file))
        events = data["events"]  # shape: (N, 25)
        dt = (time.perf_counter() - t0) * 1000

        results.record_pass(f"Loaded {events.shape[0]:,} events, {events.shape[1]} features from {data_file.name}")
        info(f"Load time: {fmt_time(dt)}")
        info(f"Data dtype: {events.dtype}")
        info(f"Memory: {events.nbytes / 1024 / 1024:.1f} MB")

        # Basic sanity checks
        nan_count = np.isnan(events).sum()
        inf_count = np.isinf(events).sum()
        if nan_count > 0:
            results.record_warn(f"{nan_count} NaN values in event data")
        else:
            results.record_pass("No NaN values in event data")

        if inf_count > 0:
            results.record_warn(f"{inf_count} Inf values in event data")
        else:
            results.record_pass("No Inf values in event data")

        # Check labels are present too
        if "labels_1s" in data:
            labels = data["labels_1s"]
            results.record_pass(f"Labels available: {labels.shape[0]:,} 1s labels, mean={labels.mean():.6f}")
        else:
            results.record_warn("No labels_1s in data file (not needed for inference, but noted)")

        return events

    except Exception as e:
        results.record_fail(f"MBO data load failed: {e}")
        traceback.print_exc()
        return None


def test_full_pipeline(
    results: DryRunResults,
    model: torch.nn.Module,
    stacked_filter,
    events: np.ndarray,
    feature_stats: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Run full end-to-end pipeline: events -> CNN-Mamba -> meta -> stacked filter -> decision."""
    section("6. Full Pipeline End-to-End Test")

    n_total = min(events.shape[0], N_TEST_EVENTS + EVENT_WINDOW)
    n_predictions = 0
    n_trades = 0
    n_gate1_pass = 0
    n_gate2_pass = 0
    n_gate3_pass = 0
    latencies_inference = []
    latencies_filter = []
    latencies_total = []
    errors = []

    mem_before = get_mem_mb()
    info(f"Processing {n_total - EVENT_WINDOW} events (window={EVENT_WINDOW}, stride={PRED_STRIDE})")
    info(f"RSS before pipeline: {mem_before:.0f} MB")

    feat_mean, feat_std = feature_stats if feature_stats else (None, None)

    t_pipeline_start = time.perf_counter()

    # Simulate streaming: slide window through events
    for i in range(EVENT_WINDOW, min(n_total, EVENT_WINDOW + N_TEST_EVENTS), PRED_STRIDE):
        try:
            t_total_start = time.perf_counter()

            # Extract window
            window = events[i - EVENT_WINDOW : i]  # (500, 25)

            # Normalize if stats available
            if feat_mean is not None and feat_std is not None:
                window = (window - feat_mean) / (feat_std + 1e-8)

            # CNN-Mamba inference
            t_inf_start = time.perf_counter()
            with torch.no_grad():
                x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)  # (1, 500, 25)
                preds = model(x)  # (1, 3)
            pred_1s = preds[0, 0].item()
            pred_5s = preds[0, 1].item()
            pred_10s = preds[0, 2].item()
            t_inf_end = time.perf_counter()
            latencies_inference.append((t_inf_end - t_inf_start) * 1000)

            # Current event features for meta-model (last event in window, raw)
            current_event = events[i - 1]  # 25-dim raw features

            # Compute a simple OFI proxy from the event data
            # Feature index 2 is typically 'side', index 4 is 'qty_log'
            # Use a simple rolling imbalance as OFI proxy
            recent = events[max(0, i - 50) : i]
            sides = recent[:, 2]  # side feature
            qtys = recent[:, 4]   # qty feature
            # Approximate: negative = sell pressure
            ofi_proxy = float(np.sum(qtys[sides < 0.5]) - np.sum(qtys[sides >= 0.5]))

            # Stacked filter decision
            t_filt_start = time.perf_counter()
            gate_result = stacked_filter.should_trade(
                pred_1s=pred_1s,
                pred_5s=pred_5s,
                pred_10s=pred_10s,
                event_features=current_event,
                ofi_book_1s=ofi_proxy,
            )
            t_filt_end = time.perf_counter()
            latencies_filter.append((t_filt_end - t_filt_start) * 1000)

            t_total_end = time.perf_counter()
            latencies_total.append((t_total_end - t_total_start) * 1000)

            n_predictions += 1
            if gate_result.should_trade:
                n_trades += 1

            # Track per-gate pass rates from gate_reasons
            reasons = gate_result.gate_reasons
            if "signal_gate" in reasons and reasons["signal_gate"].get("passed"):
                n_gate1_pass += 1
            if "meta_gate" in reasons and reasons["meta_gate"].get("passed"):
                n_gate2_pass += 1
            if "ofi_gate" in reasons and reasons["ofi_gate"].get("passed"):
                n_gate3_pass += 1

        except Exception as e:
            errors.append(f"Event {i}: {e}")
            if len(errors) <= 3:
                traceback.print_exc()

    t_pipeline_total = (time.perf_counter() - t_pipeline_start) * 1000
    mem_after = get_mem_mb()

    # Report results
    if errors:
        results.record_fail(f"{len(errors)} errors during pipeline execution")
        for e in errors[:5]:
            info(f"  Error: {e}")
    else:
        results.record_pass(f"Pipeline completed {n_predictions} predictions with 0 errors")

    info(f"")
    info(f"── Pipeline Summary ──")
    info(f"  Total predictions:    {n_predictions}")
    info(f"  Trade signals:        {n_trades} ({n_trades / max(n_predictions, 1) * 100:.1f}%)")
    info(f"  Gate 1 (signal) pass: {n_gate1_pass} ({n_gate1_pass / max(n_predictions, 1) * 100:.1f}%)")
    info(f"  Gate 2 (meta) pass:   {n_gate2_pass} ({n_gate2_pass / max(n_predictions, 1) * 100:.1f}%)")
    info(f"  Gate 3 (OFI) pass:    {n_gate3_pass} ({n_gate3_pass / max(n_predictions, 1) * 100:.1f}%)")
    info(f"")

    if latencies_inference:
        lat_inf = np.array(latencies_inference)
        info(f"── Latency: CNN-Mamba Inference (CPU) ──")
        info(f"  Mean:   {lat_inf.mean():.2f} ms")
        info(f"  Median: {np.median(lat_inf):.2f} ms")
        info(f"  P95:    {np.percentile(lat_inf, 95):.2f} ms")
        info(f"  P99:    {np.percentile(lat_inf, 99):.2f} ms")
        info(f"  Max:    {lat_inf.max():.2f} ms")
        info(f"")

    if latencies_filter:
        lat_filt = np.array(latencies_filter)
        info(f"── Latency: Stacked Filter ──")
        info(f"  Mean:   {lat_filt.mean():.2f} ms")
        info(f"  Median: {np.median(lat_filt):.2f} ms")
        info(f"  P95:    {np.percentile(lat_filt, 95):.2f} ms")
        info(f"  Max:    {lat_filt.max():.2f} ms")
        info(f"")

    if latencies_total:
        lat_tot = np.array(latencies_total)
        info(f"── Latency: Full Pipeline (inference + filter) ──")
        info(f"  Mean:   {lat_tot.mean():.2f} ms")
        info(f"  Median: {np.median(lat_tot):.2f} ms")
        info(f"  P95:    {np.percentile(lat_tot, 95):.2f} ms")
        info(f"  P99:    {np.percentile(lat_tot, 99):.2f} ms")
        info(f"  Max:    {lat_tot.max():.2f} ms")
        info(f"  Total wall time: {t_pipeline_total:.0f} ms for {n_predictions} predictions")
        info(f"")

    info(f"── Memory ──")
    info(f"  RSS before pipeline: {mem_before:.0f} MB")
    info(f"  RSS after pipeline:  {mem_after:.0f} MB")
    info(f"  Delta:               +{mem_after - mem_before:.1f} MB")

    # Check latency budget: on GPU (Razer) we need < 250ms per prediction (stride interval)
    # CPU will be slower, but flag if it's extreme
    if latencies_total:
        median_total = np.median(latencies_total)
        if median_total > 500:
            results.record_warn(f"CPU median latency {median_total:.0f}ms — expected to be faster on Razer GPU")
        elif median_total > 250:
            results.record_warn(f"CPU median latency {median_total:.0f}ms — tight for 250ms stride, but GPU should be fine")
        else:
            results.record_pass(f"CPU median latency {median_total:.0f}ms — well within budget for GPU deployment")


def test_prediction_distribution(
    results: DryRunResults,
    model: torch.nn.Module,
    events: np.ndarray,
    feature_stats: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    """Check that model predictions look reasonable (not all zeros, not all NaN)."""
    section("7. Prediction Sanity Check")

    feat_mean, feat_std = feature_stats if feature_stats else (None, None)

    # Run predictions on a batch of windows
    n_samples = min(50, (events.shape[0] - EVENT_WINDOW) // PRED_STRIDE)
    preds_1s = []
    preds_5s = []
    preds_10s = []

    indices = np.linspace(EVENT_WINDOW, min(events.shape[0], EVENT_WINDOW + N_TEST_EVENTS) - 1, n_samples, dtype=int)

    for idx in indices:
        window = events[idx - EVENT_WINDOW : idx].copy()
        if feat_mean is not None and feat_std is not None:
            window = (window - feat_mean) / (feat_std + 1e-8)
        with torch.no_grad():
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            out = model(x)
        preds_1s.append(out[0, 0].item())
        preds_5s.append(out[0, 1].item())
        preds_10s.append(out[0, 2].item())

    preds_1s = np.array(preds_1s)
    preds_5s = np.array(preds_5s)
    preds_10s = np.array(preds_10s)

    # Check for NaN/Inf
    for name, arr in [("1s", preds_1s), ("5s", preds_5s), ("10s", preds_10s)]:
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan > 0:
            results.record_fail(f"pred_{name}: {n_nan}/{len(arr)} NaN predictions")
        elif n_inf > 0:
            results.record_fail(f"pred_{name}: {n_inf}/{len(arr)} Inf predictions")
        else:
            results.record_pass(f"pred_{name}: no NaN/Inf — range [{arr.min():.4f}, {arr.max():.4f}], mean={arr.mean():.4f}, std={arr.std():.4f}")

    # Check predictions have variance (not collapsed)
    if preds_1s.std() < 1e-6:
        results.record_fail("pred_1s has near-zero variance — model may be broken")
    else:
        results.record_pass(f"pred_1s has healthy variance (std={preds_1s.std():.4f})")

    # Check short signals exist (should be roughly symmetric or slightly skewed)
    n_short = (preds_1s < 0).sum()
    info(f"Short signals: {n_short}/{len(preds_1s)} ({n_short / len(preds_1s) * 100:.0f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  MONDAY DRY-RUN SMOKE TEST")
    print("  Full pipeline: MBO → CNN-Mamba → Meta → Stacked Filter")
    print(f"  Device: {DEVICE} (CPU-only test on Jupiter)")
    print(f"  Torch version: {torch.__version__}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = DryRunResults()
    mem_start = get_mem_mb()
    info(f"Starting RSS: {mem_start:.0f} MB")

    # 1. Load CNN-Mamba v2
    model = test_load_cnn_mamba(results)

    # 2. Load feature stats
    feature_stats = test_load_feature_stats(results)

    # 3. Load meta-model ensemble
    meta_models = test_load_meta_model(results)

    # 4. Load stacked filter (also loads its own copy of meta-models)
    stacked_filter = test_load_stacked_filter(results)

    # 5. Load MBO event data
    events = test_load_mbo_data(results)

    # 6. Full pipeline test (only if all components loaded)
    if model is not None and stacked_filter is not None and events is not None:
        test_full_pipeline(results, model, stacked_filter, events, feature_stats)
    else:
        section("6. Full Pipeline End-to-End Test")
        results.record_fail("Skipped — missing components (see failures above)")

    # 7. Prediction distribution sanity check
    if model is not None and events is not None:
        test_prediction_distribution(results, model, events, feature_stats)
    else:
        section("7. Prediction Sanity Check")
        results.record_fail("Skipped — missing model or data")

    # Final summary
    mem_end = get_mem_mb()
    section("FINAL RESULTS")
    info(f"Total memory: {mem_end:.0f} MB (delta: +{mem_end - mem_start:.0f} MB)")
    print()
    print(f"  Passed:   {results.passed}")
    print(f"  Failed:   {results.failed}")
    print(f"  Warnings: {results.warnings}")
    print()

    if results.failed == 0:
        print("  >>> ALL CHECKS PASSED — Pipeline is ready for Monday deployment <<<")
        return 0
    else:
        print("  >>> FAILURES DETECTED — Fix before Monday deployment <<<")
        for err in results.errors:
            print(f"    - {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
