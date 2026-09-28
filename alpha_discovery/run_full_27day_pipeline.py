"""
Full 27-Day Pipeline — Runs everything in sequence after cache rebuild.

Steps:
1. Verify all 27 cache files exist
2. Build feature cache (features_alldays.npz) via run_mbo_scan.py --no-cache
3. Run corrected limit study with full data
4. Run hybrid execution sim with full data
5. Run multi-bar alpha scan (if script exists)
6. Save combined results and summary

Usage:
    python alpha_discovery/run_full_27day_pipeline.py
    python alpha_discovery/run_full_27day_pipeline.py --skip-feature-cache
"""

import gc
import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data" / "processed" / "medium_snapshots_cache"
RESULTS_DIR = ROOT / "alpha_discovery" / "results"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'pipeline_27day.log', mode='w'),
    ]
)
logger = logging.getLogger("pipeline")


def run_script(script_path, args=None, timeout_minutes=120):
    """Run a Python script and capture output."""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    logger.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_minutes * 60,
        cwd=str(ROOT),
        env={**os.environ, 'PYTHONUNBUFFERED': '1'},
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Script failed (exit code {result.returncode}) after {elapsed:.0f}s")
        logger.error(f"STDERR: {result.stderr[-2000:] if result.stderr else 'none'}")
        logger.error(f"STDOUT (last 1000): {result.stdout[-1000:] if result.stdout else 'none'}")
        return None

    logger.info(f"Script completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return result.stdout


def verify_caches():
    """Check that all snapshot cache files exist."""
    cache_files = sorted(CACHE_DIR.glob("file_*_snapshots.npz"))
    logger.info(f"Found {len(cache_files)} cache files in {CACHE_DIR}")

    if len(cache_files) < 20:
        logger.warning(f"Only {len(cache_files)} cache files — expected 27. Cache rebuild may be incomplete.")

    total_size = sum(f.stat().st_size for f in cache_files)
    logger.info(f"Total cache size: {total_size / 1e9:.2f} GB")

    return len(cache_files)


def main():
    parser = argparse.ArgumentParser(description='Full 27-Day Pipeline')
    parser.add_argument('--skip-feature-cache', action='store_true',
                        help='Skip feature cache rebuild (use existing)')
    parser.add_argument('--skip-scan', action='store_true',
                        help='Skip the alpha scan (just run limit study + hybrid)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    t_pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("FULL 27-DAY PIPELINE")
    logger.info(f"  Started: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    results = {}

    # Step 1: Verify caches
    logger.info("\n--- STEP 1: Verify Cache Files ---")
    n_files = verify_caches()
    results['n_cache_files'] = n_files

    if n_files < 15:
        logger.error("Too few cache files. Run rebuild_snapshot_caches.py first!")
        sys.exit(1)

    # Step 2: Build feature cache
    if not args.skip_feature_cache:
        logger.info("\n--- STEP 2: Build Feature Cache ---")
        scan_script = ROOT / "alpha_discovery" / "run_mbo_scan.py"
        output = run_script(scan_script, ['--no-cache'], timeout_minutes=30)
        if output:
            results['feature_cache'] = 'success'
            # Save the scan output
            with open(RESULTS_DIR / f'scan_27day_{timestamp}.txt', 'w') as f:
                f.write(output)
        else:
            logger.error("Feature cache build FAILED")
            results['feature_cache'] = 'failed'
    else:
        logger.info("\n--- STEP 2: Skipped (using existing feature cache) ---")

    gc.collect()

    # Step 3: Corrected Limit Study
    logger.info("\n--- STEP 3: Corrected Limit Study (Full Data) ---")
    limit_script = ROOT / "alpha_discovery" / "run_corrected_limit_study.py"
    if limit_script.exists():
        output = run_script(limit_script, timeout_minutes=60)
        if output:
            results['corrected_limit_study'] = 'success'
        else:
            results['corrected_limit_study'] = 'failed'
    else:
        logger.warning("run_corrected_limit_study.py not found, skipping")
        results['corrected_limit_study'] = 'missing'

    gc.collect()

    # Step 4: Hybrid Execution Sim
    logger.info("\n--- STEP 4: Hybrid Execution Sim (Full Data) ---")
    hybrid_script = ROOT / "alpha_discovery" / "run_hybrid_execution_sim.py"
    if hybrid_script.exists():
        output = run_script(hybrid_script, timeout_minutes=60)
        if output:
            results['hybrid_execution'] = 'success'
        else:
            results['hybrid_execution'] = 'failed'
    else:
        logger.warning("run_hybrid_execution_sim.py not found, skipping")
        results['hybrid_execution'] = 'missing'

    gc.collect()

    # Step 5: Multi-bar Alpha Scan (if exists)
    logger.info("\n--- STEP 5: Multi-Bar Alpha Scan ---")
    multibar_script = ROOT / "alpha_discovery" / "run_multibar_alpha_scan.py"
    if multibar_script.exists() and not args.skip_scan:
        output = run_script(multibar_script, timeout_minutes=90)
        if output:
            results['multibar_scan'] = 'success'
        else:
            results['multibar_scan'] = 'failed'
    else:
        logger.info("Multi-bar scan script not found or skipped")
        results['multibar_scan'] = 'skipped'

    gc.collect()

    # Step 5b: Adverse Selection Study (4-tier fill model comparison)
    logger.info("\n--- STEP 5b: Adverse Selection Study ---")
    adverse_script = ROOT / "alpha_discovery" / "run_adverse_selection_study.py"
    if adverse_script.exists():
        output = run_script(adverse_script, timeout_minutes=60)
        if output:
            results['adverse_selection'] = 'success'
        else:
            results['adverse_selection'] = 'failed'
    else:
        logger.info("Adverse selection study not found, skipping")
        results['adverse_selection'] = 'skipped'

    gc.collect()

    # Step 6: Queue Position Study (if exists)
    logger.info("\n--- STEP 6: Queue Position Study ---")
    queue_script = ROOT / "alpha_discovery" / "run_queue_position_study.py"
    if queue_script.exists():
        output = run_script(queue_script, timeout_minutes=30)
        if output:
            results['queue_position'] = 'success'
        else:
            results['queue_position'] = 'failed'
    else:
        logger.info("Queue position script not found, skipping")
        results['queue_position'] = 'skipped'

    # Step 7: Compile Realistic Assessment
    logger.info("\n--- STEP 7: Realistic Assessment ---")
    assessment_script = ROOT / "alpha_discovery" / "compile_realistic_assessment.py"
    if assessment_script.exists():
        output = run_script(assessment_script, timeout_minutes=5)
        if output:
            results['realistic_assessment'] = 'success'
            # Save the assessment output
            with open(RESULTS_DIR / f'assessment_27day_{timestamp}.txt', 'w') as f:
                f.write(output)
        else:
            results['realistic_assessment'] = 'failed'
    else:
        logger.info("Assessment script not found, skipping")
        results['realistic_assessment'] = 'skipped'

    # Summary
    elapsed = time.time() - t_pipeline_start
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Results: {json.dumps(results, indent=2)}")
    logger.info("=" * 70)

    # Save pipeline results
    results['timestamp'] = timestamp
    results['elapsed_seconds'] = elapsed
    results['n_cache_files'] = n_files

    result_path = RESULTS_DIR / f'pipeline_27day_{timestamp}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Pipeline results saved to {result_path}")

    # Collect all result files for summary
    logger.info("\nRecent result files:")
    for f in sorted(RESULTS_DIR.glob("*_2026021*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        logger.info(f"  {f.name} ({f.stat().st_size / 1e3:.1f} KB)")


if __name__ == '__main__':
    main()
