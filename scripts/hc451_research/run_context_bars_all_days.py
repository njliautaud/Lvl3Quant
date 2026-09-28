#!/usr/bin/env python3
"""HC #451 R1 — Orchestrator: run precompute_context_bars.py across all OOT days.

Day list is canonically derived from the existing salience parquets (per the task
instruction: same days as salience). Multiprocessing.Pool with cpu_count-2 workers.
"""
import json
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

SCRIPT_DIR = Path("/home/jupiter/Lvl3Quant/scripts/hc451_research")
sys.path.insert(0, str(SCRIPT_DIR))

from precompute_context_bars import process_date  # noqa: E402

SAL_DIR = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags/per_day")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc451_context_bars/per_day")
LOG_PATH = Path("/home/jupiter/Lvl3Quant/output/hc451_context_bars/run_log.jsonl")
FAILED_PATH = Path("/home/jupiter/Lvl3Quant/output/hc451_context_bars/failed_days.txt")

# Reduced 2026-05-21 00:38 ET — 12 workers exhausted 2GB swap (each worker loads
# 200-650MB MBO .dbn.zst into RAM and processes rolling windows; total footprint
# ~36+ GB ate all 46 GB RAM + spilled to swap → thrashing, 0 dates completed in
# 22 min). 6 workers keeps total RAM at ~18-20 GB, comfortable headroom.
import os as _os
N_WORKERS = int(_os.environ.get("CONTEXT_BARS_N_WORKERS", "6"))


def list_dates():
    pat = re.compile(r"(\d{8})_salience\.parquet$")
    dates = []
    for f in sorted(SAL_DIR.iterdir()):
        m = pat.match(f.name)
        if m:
            dates.append(m.group(1))
    return dates


def worker(date_str):
    try:
        return process_date(date_str, OUT_DIR, force=False)
    except Exception as e:
        return {"date": date_str, "status": "error", "error": repr(e)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = list_dates()
    print(f"[driver] {len(dates)} dates queued, {N_WORKERS} workers", flush=True)

    t0 = time.time()
    results = []
    with open(LOG_PATH, "w") as logf, Pool(N_WORKERS) as pool:
        for i, st in enumerate(pool.imap_unordered(worker, dates), 1):
            results.append(st)
            logf.write(json.dumps(st) + "\n")
            logf.flush()
            if i % 5 == 0 or i == len(dates):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(dates) - i) / rate if rate > 0 else float("inf")
                print(f"[driver] {i}/{len(dates)} done — elapsed {el:.1f}s "
                      f"ETA {eta:.0f}s ({eta/60:.1f}min)", flush=True)

    wall = time.time() - t0
    failed = [r for r in results if r.get("status") not in ("ok", "skipped_exists")]
    if failed:
        with open(FAILED_PATH, "w") as f:
            for r in failed:
                f.write(f"{r.get('date')}\t{r.get('status')}\t{r.get('error', '')}\n")
    ok = [r for r in results if r.get("status") == "ok"]
    skip = [r for r in results if r.get("status") == "skipped_exists"]
    print(f"[driver] DONE: ok={len(ok)} skipped={len(skip)} failed={len(failed)} "
          f"wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
