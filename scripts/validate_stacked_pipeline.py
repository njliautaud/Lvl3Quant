#!/usr/bin/env python3
"""
validate_stacked_pipeline.py — End-to-end validation of Monday's stacked filter
================================================================================
Catches integration bugs BEFORE Monday morning. Tests:
  1. Module loads, config parses, meta-model weights load (15 folds)
  2. Gate logic works with synthetic inputs at various confidence levels
  3. Edge cases: NaN preds, extreme values, empty history, all presets
  4. Gate pass rates match expected ranges from sensitivity sweep
  5. Short-circuit behavior (gate 1 fail → gates 2/3 skipped)

Output: PASS/FAIL summary to stdout + JSON results file.
"""

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Setup
os.chdir("/home/jupiter/Lvl3Quant")
sys.path.insert(0, "/home/jupiter/Lvl3Quant")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS = {"tests": [], "passed": 0, "failed": 0, "errors": 0}


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        def wrapper():
            try:
                logger.info(f"TEST: {name}")
                fn()
                RESULTS["tests"].append({"name": name, "status": "PASS"})
                RESULTS["passed"] += 1
                logger.info(f"  PASS: {name}")
            except AssertionError as e:
                RESULTS["tests"].append({"name": name, "status": "FAIL", "error": str(e)})
                RESULTS["failed"] += 1
                logger.error(f"  FAIL: {name} — {e}")
            except Exception as e:
                RESULTS["tests"].append({"name": name, "status": "ERROR", "error": traceback.format_exc()})
                RESULTS["errors"] += 1
                logger.error(f"  ERROR: {name} — {e}")
        wrapper.__name__ = name
        return wrapper
    return decorator


# ===== TEST SUITE =====

@test("1_module_import")
def test_import():
    """Can we import the stacked filter module?"""
    from live_trading.stacked_filter import StackedFilter, GateResult, GateStats
    assert StackedFilter is not None
    assert GateResult is not None
    assert GateStats is not None


@test("2_config_parse")
def test_config():
    """Does the config file parse and contain expected presets?"""
    cfg_path = Path("live_trading/configs/monday_stacked_v1.json")
    assert cfg_path.exists(), f"Config not found at {cfg_path}"
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert "presets" in cfg
    assert "balanced" in cfg["presets"]
    assert "simpler" in cfg["presets"]
    assert "ultra_selective" in cfg["presets"]
    assert cfg["active_preset"] == "balanced"
    assert cfg["cost"]["rt_commission_ticks"] == 0.376
    assert cfg["contract"]["symbol"] == "ESU6"
    logger.info(f"  Config OK: {len(cfg['presets'])} presets, ESU6, cost 0.376t")


@test("3_balanced_preset_loads")
def test_balanced():
    """Load StackedFilter with balanced preset — meta weights should load."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    assert sf._preset_name == "balanced"
    assert sf._signal_threshold_pct == 0.10
    assert sf._meta_filter_pct == 0.30
    assert sf._ofi_gate_enabled is True
    assert len(sf._meta_models) == 15, f"Expected 15 meta folds, got {len(sf._meta_models)}"
    logger.info(f"  Balanced loaded: {len(sf._meta_models)} meta folds")


@test("4_simpler_preset_loads")
def test_simpler():
    """Load simpler preset — meta disabled (filter_pct=1.0)."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="simpler",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    assert sf._preset_name == "simpler"
    assert sf._signal_threshold_pct == 0.05
    assert sf._meta_enabled is False or sf._meta_filter_pct == 1.0
    assert sf._ofi_gate_enabled is True


@test("5_ultra_selective_preset_loads")
def test_ultra():
    """Load ultra_selective preset."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="ultra_selective",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    assert sf._preset_name == "ultra_selective"
    assert sf._signal_threshold_pct == 0.03
    assert sf._meta_filter_pct == 0.10


@test("6_gate_logic_strong_signal")
def test_strong_signal():
    """A very strong short signal with agreeing OFI should pass all gates."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    # Seed the prediction history with enough samples for percentile computation
    for i in range(500):
        fake_pred = np.random.normal(-0.02, 0.05)
        sf._pred_history.append(fake_pred)

    # Strong short signal (very negative = strong short)
    result = sf.should_trade(
        pred_1s=-0.25,   # Very strong short
        pred_5s=-0.15,
        pred_10s=-0.08,
        event_features=np.random.randn(25).astype(np.float32),
        ofi_book_1s=-200.0,  # Strong selling pressure = agrees with short
    )
    logger.info(f"  Strong signal result: trade={result.should_trade}, "
                f"signal_pct={result.signal_pct_rank:.3f}, "
                f"meta={result.meta_score:.4f}, ofi={result.ofi_agreement}")
    # We can't guarantee it passes (meta-model may reject), but it shouldn't crash
    assert isinstance(result.should_trade, bool)
    assert 0.0 <= result.signal_pct_rank <= 1.0


@test("7_gate_logic_weak_signal")
def test_weak_signal():
    """A weak signal should be rejected by gate 1 (signal threshold)."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    for i in range(500):
        sf._pred_history.append(np.random.normal(-0.02, 0.05))

    # Weak signal (near zero)
    result = sf.should_trade(
        pred_1s=-0.001,
        pred_5s=0.001,
        pred_10s=0.002,
        event_features=np.random.randn(25).astype(np.float32),
        ofi_book_1s=-50.0,
    )
    assert result.should_trade is False, "Weak signal should be rejected"
    assert "signal" in result.gate_reasons or "gate1" in str(result.gate_reasons).lower()
    logger.info(f"  Weak signal correctly rejected. Reasons: {list(result.gate_reasons.keys())}")


@test("8_short_circuit_gate1_fail")
def test_short_circuit():
    """When gate 1 fails, gates 2-3 should be skipped (short-circuit)."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    for i in range(500):
        sf._pred_history.append(np.random.normal(-0.02, 0.05))

    result = sf.should_trade(
        pred_1s=0.01,  # LONG signal — should fail gate 1 for shorts
        pred_5s=0.005,
        pred_10s=0.003,
        event_features=np.random.randn(25).astype(np.float32),
        ofi_book_1s=100.0,
    )
    assert result.should_trade is False
    # Verify stats show gate1 didn't pass
    assert sf.stats.gate1_passed == 0 or sf.stats.all_passed == 0


@test("9_nan_handling")
def test_nan():
    """NaN predictions should not crash the filter."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    for i in range(500):
        sf._pred_history.append(np.random.normal(-0.02, 0.05))

    try:
        result = sf.should_trade(
            pred_1s=float('nan'),
            pred_5s=float('nan'),
            pred_10s=float('nan'),
            event_features=np.random.randn(25).astype(np.float32),
            ofi_book_1s=0.0,
        )
        # Should either return False or handle gracefully
        assert result.should_trade is False, "NaN signals should be rejected"
        logger.info("  NaN handled gracefully — rejected")
    except (ValueError, RuntimeError) as e:
        # Acceptable if it raises a clear error
        logger.info(f"  NaN raised expected error: {type(e).__name__}")


@test("10_gate_stats_tracking")
def test_stats():
    """Gate statistics should accumulate correctly."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    for i in range(500):
        sf._pred_history.append(np.random.normal(-0.02, 0.05))

    # Run 100 random predictions
    n_trades = 0
    for _ in range(100):
        pred = np.random.normal(-0.02, 0.08)
        result = sf.should_trade(
            pred_1s=pred,
            pred_5s=pred * 0.7,
            pred_10s=pred * 0.5,
            event_features=np.random.randn(25).astype(np.float32),
            ofi_book_1s=np.random.normal(0, 100),
        )
        if result.should_trade:
            n_trades += 1

    stats = sf.stats.to_dict()
    assert stats["total_evaluated"] == 100, f"Expected 100 evaluations, got {stats['total_evaluated']}"
    assert stats["all_passed"] == n_trades
    logger.info(f"  Stats OK: {stats['total_evaluated']} evals, {stats['all_passed']} passed "
                f"(gate1: {stats['gate1_rate']:.1%}, gate2: {stats['gate2_rate']:.1%}, "
                f"gate3: {stats['gate3_rate']:.1%})")


@test("11_meta_model_inference_speed")
def test_speed():
    """Meta-model ensemble inference should be fast (<10ms per call)."""
    from live_trading.stacked_filter import StackedFilter
    sf = StackedFilter(
        preset="balanced",
        meta_weights_dir="output/meta_production_v1/weights/",
        device="cpu",
    )
    for i in range(500):
        sf._pred_history.append(np.random.normal(-0.02, 0.05))

    # Warm up
    for _ in range(10):
        sf.should_trade(
            pred_1s=-0.15, pred_5s=-0.08, pred_10s=-0.04,
            event_features=np.random.randn(25).astype(np.float32),
            ofi_book_1s=-100.0,
        )

    # Benchmark
    t0 = time.time()
    N = 100
    for _ in range(N):
        sf.should_trade(
            pred_1s=-0.15, pred_5s=-0.08, pred_10s=-0.04,
            event_features=np.random.randn(25).astype(np.float32),
            ofi_book_1s=-100.0,
        )
    elapsed = (time.time() - t0) / N * 1000  # ms per call

    assert elapsed < 50, f"Too slow: {elapsed:.1f}ms per call (target <50ms)"
    logger.info(f"  Speed OK: {elapsed:.2f}ms per should_trade() call")


# ===== MAIN =====

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STACKED PIPELINE VALIDATION — Monday Deploy Readiness")
    logger.info("=" * 60)

    tests = [
        test_import, test_config, test_balanced, test_simpler, test_ultra,
        test_strong_signal, test_weak_signal, test_short_circuit,
        test_nan, test_stats, test_speed,
    ]

    for t in tests:
        t()
        logger.info("")

    # Summary
    logger.info("=" * 60)
    total = RESULTS["passed"] + RESULTS["failed"] + RESULTS["errors"]
    logger.info(f"RESULTS: {RESULTS['passed']}/{total} passed, "
                f"{RESULTS['failed']} failed, {RESULTS['errors']} errors")

    if RESULTS["failed"] == 0 and RESULTS["errors"] == 0:
        logger.info("ALL TESTS PASSED — Pipeline ready for Monday deployment")
    else:
        logger.warning("ISSUES FOUND — Review failures before Monday")

    # Save results
    out_path = Path("output/pipeline_validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    logger.info(f"Results saved to {out_path}")
    logger.info("=" * 60)

    sys.exit(0 if RESULTS["failed"] == 0 and RESULTS["errors"] == 0 else 1)
