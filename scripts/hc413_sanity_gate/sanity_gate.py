#!/usr/bin/env python3
"""HC #413 step-4 model sanity-gate harness.

Decides whether a trained model checkpoint + its prediction NPZ is safe to
wire into the live paper-trader on Razer.

Usage:
  python3 sanity_gate.py \
      --npz <path/to/fold_00_predictions.npz> \
      --model-family v3_3|v3_4_2 \
      [--ckpt <path/to/fold_00_intra_ckpt.pt>] \
      [--data-dir <path>] \
      --output-dir <dir>

Exit code: 0 if overall pass, 1 if any required check failed.
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from checks import (
    npz_health,
    distribution,
    calibration,
    direction,
    book_gate,
    input_features,
    e2e_smoke,
)
from report import write_reports


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to predictions NPZ")
    p.add_argument("--model-family", required=True, choices=["v3_3", "v3_4_2"])
    p.add_argument("--ckpt", default=None, help="Path to model .pt (required for v3_4_2 book_gate check)")
    p.add_argument("--data-dir", default=None, help="Optional path to data dir for input feature checks")
    p.add_argument("--output-dir", required=True, help="Directory to write report.json and report.md")
    return p.parse_args()


def _run(name: str, fn):
    try:
        return fn()
    except Exception as e:  # noqa
        return {
            "check": name,
            "passed": False,
            "failures": [f"check raised: {e!r}"],
            "details": {"traceback": traceback.format_exc().splitlines()[-6:]},
        }


def main() -> int:
    args = parse_args()
    t0 = time.time()
    if not os.path.exists(args.npz):
        print(f"ERROR: NPZ not found: {args.npz}", file=sys.stderr)
        return 2

    try:
        npz = np.load(args.npz, allow_pickle=True)
    except Exception as e:  # noqa
        print(f"ERROR: failed to load NPZ: {e!r}", file=sys.stderr)
        return 2

    family = args.model_family
    results = []
    results.append(_run("npz_health", lambda: npz_health.run(npz, family)))
    results.append(_run("distribution", lambda: distribution.run(npz, family)))
    results.append(_run("calibration", lambda: calibration.run(npz, family)))
    results.append(_run("direction", lambda: direction.run(npz, family)))
    results.append(_run("book_gate", lambda: book_gate.run(args.ckpt, family)))
    results.append(_run("input_features", lambda: input_features.run(args.npz, family, args.data_dir)))
    results.append(_run("e2e_smoke", lambda: e2e_smoke.run(npz, family)))

    meta = {
        "npz": os.path.abspath(args.npz),
        "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
        "model_family": family,
        "data_dir": args.data_dir,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    json_path, md_path = write_reports(results, meta, args.output_dir)

    overall_pass = all(r.get("passed", False) for r in results)
    print(f"[hc413-sanity] overall_pass={overall_pass}")
    print(f"[hc413-sanity] json: {json_path}")
    print(f"[hc413-sanity] md:   {md_path}")
    for r in results:
        status = "SKIP" if r.get("skipped") else ("PASS" if r.get("passed") else "FAIL")
        fails = ("; ".join(r.get("failures") or []))[:160]
        print(f"  {r['check']:18s} {status}  {fails}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
