#!/usr/bin/env python3
"""HC #451 — Orchestrator: run precompute_salience_tags.py across all 238 OOT days.

Bounded driver. Does NOT modify the per-day script — calls process_date() in a
multiprocessing.Pool capped at 6 workers (Ray runs on Jupiter; do not starve it).

Outputs:
  - per_day/<YYYYMMDD>_salience.parquet
  - per_day/<YYYYMMDD>_stats.json
  - failed_days.txt
  - run_log.jsonl
"""
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Make per-day script importable
SCRIPT_DIR = Path("/home/jupiter/Lvl3Quant/scripts/hc451_research")
sys.path.insert(0, str(SCRIPT_DIR))

from precompute_salience_tags import process_date, RAW_DIR  # noqa: E402

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags/per_day")
LOG_PATH = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags/run_log.jsonl")
FAILED_PATH = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags/failed_days.txt")

N_WORKERS = 6


def list_dates():
    pat = re.compile(r"glbx-mdp3-(\d{8})\.mbo\.dbn\.zst$")
    dates = []
    for f in sorted(RAW_DIR.iterdir()):
        m = pat.match(f.name)
        if m:
            dates.append(m.group(1))
    return dates


def worker(date_str):
    try:
        stats = process_date(date_str, OUT_DIR, force=False)
        return stats
    except Exception as e:
        return {"date": date_str, "status": "error", "error": repr(e)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = list_dates()
    print(f"[driver] {len(dates)} dates queued, {N_WORKERS} workers")

    t0 = time.time()
    results = []
    with open(LOG_PATH, "w") as logf, Pool(N_WORKERS) as pool:
        for i, st in enumerate(pool.imap_unordered(worker, dates), 1):
            results.append(st)
            logf.write(json.dumps(st) + "\n")
            logf.flush()
            if i % 10 == 0 or i == len(dates):
                el = time.time() - t0
                print(f"[driver] {i}/{len(dates)} done — elapsed {el:.1f}s")

    wall = time.time() - t0

    # Collect failures
    failed = [r for r in results
              if r.get("status") not in ("ok", "skipped_exists")]
    if failed:
        with open(FAILED_PATH, "w") as f:
            for r in failed:
                f.write(f"{r.get('date')}\t{r.get('status')}\t{r.get('error', '')}\n")

    ok = [r for r in results if r.get("status") == "ok"]
    skip = [r for r in results if r.get("status") == "skipped_exists"]
    print(f"[driver] DONE: ok={len(ok)} skipped={len(skip)} failed={len(failed)} "
          f"wall={wall:.1f}s")
    return wall, results


if __name__ == "__main__":
    main()
