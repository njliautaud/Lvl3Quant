#!/usr/bin/env python3
"""Generate window_counts.json sidecar for precomputed tensor directories."""
import sys, json, time
from pathlib import Path
import torch

def gen_counts(tensor_dir: str):
    d = Path(tensor_dir)
    # Only process date-prefixed mbo_events files, skip stats.pt/metadata.json
    pt_files = sorted(f for f in d.glob("*_mbo_events.pt"))
    if not pt_files:
        print(f"[WARN] No *_mbo_events.pt files in {d}")
        return
    out = {}
    t0 = time.time()
    for i, f in enumerate(pt_files):
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "events" in data:
                n = int(data["events"].shape[0])
            elif isinstance(data, dict):
                # fallback: try first tensor-valued key
                for k, v in data.items():
                    if hasattr(v, "shape"):
                        n = int(v.shape[0])
                        break
                else:
                    n = 0
            else:
                n = int(data.shape[0])
            out[f.name] = n
            del data
        except Exception as e:
            print(f"  [WARN] {f.name}: {e} — skipping")
            out[f.name] = 0
        if (i+1) % 10 == 0 or (i+1) == len(pt_files):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(pt_files)}] {f.name}: {out[f.name]} windows | elapsed={elapsed:.0f}s", flush=True)
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
