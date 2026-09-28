#!/usr/bin/env python3
"""
Parallel wrapper for precompute_features_smart_v2.py
Splits remaining files across N workers for higher CPU utilization.
Each worker processes a disjoint subset of files.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

INPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v2")
SCRIPT = "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/precompute_features_smart_v2.py"

N_WORKERS = int(os.environ.get("N_WORKERS", 4))

def get_remaining_files():
    """Get input files that haven't been processed yet."""
    input_files = sorted(INPUT_DIR.glob("*_mbo_events.npz"))
    done_files = {f.name for f in OUTPUT_DIR.glob("*.npz")}
    remaining = [f for f in input_files if f.name not in done_files]
    return remaining

def process_file(input_path: Path):
    """Process a single file using the v2 preprocessing logic."""
    import numpy as np
    # Import the processing functions from the v2 script
    sys.path.insert(0, str(Path(SCRIPT).parent))

    # We need to do the actual processing inline since importing is complex
    # Instead, use a simpler approach: each worker runs the full script
    # but with a restricted file list via env var
    output_path = OUTPUT_DIR / input_path.name
    if output_path.exists():
        return f"SKIP {input_path.name}"

    # Actually process using subprocess with file filter
    env = os.environ.copy()
    env["SMART_SINGLE_FILE"] = str(input_path)
    env["SMART_OUTPUT_DIR"] = str(OUTPUT_DIR)
    env["SMART_INPUT_DIR"] = str(INPUT_DIR)

    result = subprocess.run(
        [sys.executable, SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300  # 5 min max per file
    )

    if result.returncode == 0:
        return f"OK {input_path.name}"
    else:
        return f"FAIL {input_path.name}: {result.stderr[-200:]}"

def process_chunk(file_list):
    """Process a chunk of files sequentially (one worker's share)."""
    results = []
    for f in file_list:
        output_path = OUTPUT_DIR / f.name
        if output_path.exists():
            continue
        try:
            # Run the v2 script with SMART_SINGLE_FILE env var
            env = os.environ.copy()
            env["SMART_SINGLE_FILE"] = str(f)
            env["SMART_OUTPUT_DIR"] = str(OUTPUT_DIR)
            env["SMART_INPUT_DIR"] = str(INPUT_DIR)

            result = subprocess.run(
                [sys.executable, SCRIPT],
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )

            if output_path.exists():
                results.append(f"OK {f.name}")
            else:
                results.append(f"FAIL {f.name}")
        except subprocess.TimeoutExpired:
            results.append(f"TIMEOUT {f.name}")
        except Exception as e:
            results.append(f"ERROR {f.name}: {e}")
    return results

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    remaining = get_remaining_files()
    print(f"Remaining files: {len(remaining)}, Workers: {N_WORKERS}")

    if not remaining:
        print("All files already processed!")
        sys.exit(0)

    # Split files across workers
    chunks = [[] for _ in range(N_WORKERS)]
    for i, f in enumerate(remaining):
        chunks[i % N_WORKERS].append(f)

    for i, chunk in enumerate(chunks):
        print(f"  Worker {i}: {len(chunk)} files ({chunk[0].name}..{chunk[-1].name})")

    print(f"\nStarting {N_WORKERS} parallel workers...")
    start = time.time()

    with Pool(N_WORKERS) as pool:
        all_results = pool.map(process_chunk, chunks)

    elapsed = time.time() - start
    total_ok = sum(1 for r in sum(all_results, []) if r.startswith("OK"))
    total_fail = sum(1 for r in sum(all_results, []) if not r.startswith("OK"))

    print(f"\nDone in {elapsed:.0f}s: {total_ok} OK, {total_fail} failed")
    print(f"Total v2 files: {len(list(OUTPUT_DIR.glob('*.npz')))}/211")
