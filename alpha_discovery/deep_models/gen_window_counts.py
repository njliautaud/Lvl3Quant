#!/usr/bin/env python3
"""Generate window_counts.json sidecar for precomputed tensor directories."""
import sys, json, time
from pathlib import Path
import torch

def gen_counts(tensor_dir: str):
    d = Path(tensor_dir)
    pt_files = sorted(d.glob("*.pt"))
    if not pt_files:
        print(f"[WARN] No .pt files in {d}")
        return
    out = {}
    t0 = time.time()
    for i, f in enumerate(pt_files):
        data = torch.load(f, map_location="cpu", weights_only=True)
        if "events" not in data: continue
        n = int(data["events"].shape[0])
        out[f.name] = n
        del data
        if (i+1) % 10 == 0 or (i+1) == len(pt_files):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pt_files)}] {f.name}: {n} windows | elapsed={elapsed:.0f}s", flush=True)
    counts_path = d / "window_counts.json"
    with open(counts_path, "w") as fh:
        json.dump(out, fh)
    total = sum(out.values())
    print(f"[DONE] {d.name}: {len(out)} files, {total:,} total windows -> {counts_path}")

if __name__ == "__main__":
    dirs = sys.argv[1:]
    for d in dirs:
        print(f"\n=== Processing {d} ===")
        gen_counts(d)
