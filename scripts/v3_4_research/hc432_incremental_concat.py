#!/usr/bin/env python3
"""
HC #432 — Incremental concat of v3.4.2 per-date OOT NPZs from Neptune.

Pulls (or reads in-place) per-date NPZs produced by Neptune for the 47-day
OOT window and concatenates them into a single fold_00-style NPZ on Jupiter,
with explicit `oot_dates` (unique dates) and `sample_dates` (per-sample) keys
so downstream HC #428 R1 regime stratification can join on date.

Inputs:
  Neptune path : /home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz
  Each NPZ     : 105 keys = pred_*/target_*/mask_* (49530-ish samples per day)
                 + oot_dates (1,) + sample_dates (N,) + metric_*/metrics_loss scalars

Output:
  /home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/
      fold_00_ep1_oot_inference_47day_hc432.npz

Idempotent — safe to re-run. If Neptune SSH fails, exits cleanly with non-zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

NEPTUNE_HOST = "nick@neptune"
NEPTUNE_NPZ_DIR = "/home/nick/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
LOCAL_OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation")
LOCAL_NPZ_CACHE = LOCAL_OUT_DIR / "_npz_cache_neptune"
OUT_PATH = LOCAL_OUT_DIR / "fold_00_ep1_oot_inference_47day_hc432.npz"
MANIFEST_PATH = LOCAL_OUT_DIR / "concat_manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("hc432_concat")


def ssh_list_remote_npzs(timeout: int = 15) -> List[str]:
    cmd = ["ssh", "-o", f"ConnectTimeout={timeout}", "-o", "StrictHostKeyChecking=no",
           NEPTUNE_HOST, f"ls {NEPTUNE_NPZ_DIR}/oot_*.npz 2>/dev/null || true"]
    try:
        out = subprocess.check_output(cmd, timeout=timeout + 10).decode().strip()
    except subprocess.SubprocessError as e:
        log.error(f"SSH list failed: {e}")
        return []
    return sorted([line.strip() for line in out.splitlines() if line.strip()])


def rsync_pull(remote_paths: List[str], timeout: int = 120) -> List[Path]:
    """Rsync any missing NPZs from Neptune to local cache. Returns local paths."""
    LOCAL_NPZ_CACHE.mkdir(parents=True, exist_ok=True)
    if not remote_paths:
        return []
    fetched: List[Path] = []
    missing: List[str] = []
    for rp in remote_paths:
        fname = Path(rp).name
        local = LOCAL_NPZ_CACHE / fname
        if local.exists() and local.stat().st_size > 0:
            fetched.append(local)
        else:
            missing.append(rp)
    if missing:
        log.info(f"rsyncing {len(missing)} missing NPZs from Neptune...")
        cmd = ["rsync", "-az", "--timeout=60",
               f"{NEPTUNE_HOST}:{NEPTUNE_NPZ_DIR}/{{" + ",".join(Path(p).name for p in missing) + "}}",
               str(LOCAL_NPZ_CACHE) + "/"]
        # rsync brace-expansion isn't safe through subprocess list mode; pull one-by-one
        for rp in missing:
            fname = Path(rp).name
            local = LOCAL_NPZ_CACHE / fname
            cmd = ["rsync", "-az", "--timeout=60", f"{NEPTUNE_HOST}:{rp}", str(local)]
            try:
                subprocess.check_call(cmd, timeout=timeout)
                fetched.append(local)
            except subprocess.SubprocessError as e:
                log.warning(f"rsync failed for {fname}: {e}")
    return sorted(fetched)


def discover_npzs(use_remote: bool) -> List[Path]:
    if use_remote:
        remote = ssh_list_remote_npzs()
        if not remote:
            log.warning("No NPZs visible on Neptune (SSH ok but list empty).")
            return []
        local = rsync_pull(remote)
        return local
    # local-cache-only mode
    if not LOCAL_NPZ_CACHE.exists():
        return []
    return sorted(LOCAL_NPZ_CACHE.glob("oot_*.npz"))


def classify_keys(sample_npz: np.lib.npyio.NpzFile) -> Tuple[List[str], List[str], List[str]]:
    """Split keys into (concatenable_arrays, per_run_scalars, special)."""
    concat_keys: List[str] = []
    scalar_keys: List[str] = []
    special: List[str] = []
    n_samples = None
    # detect N from sample_dates
    if "sample_dates" in sample_npz.files:
        n_samples = sample_npz["sample_dates"].shape[0]
    for k in sample_npz.files:
        a = sample_npz[k]
        if k in ("oot_dates", "sample_dates"):
            special.append(k)
            continue
        if a.ndim == 0 or a.shape == ():
            scalar_keys.append(k)
            continue
        if n_samples is not None and a.shape[0] == n_samples:
            concat_keys.append(k)
        else:
            # safety: treat anything not matching N as scalar/meta
            scalar_keys.append(k)
    return concat_keys, scalar_keys, special


def concat_npzs(paths: List[Path]) -> Dict[str, np.ndarray]:
    if not paths:
        return {}
    log.info(f"Concatenating {len(paths)} per-date NPZs...")
    first = np.load(paths[0], allow_pickle=False)
    concat_keys, scalar_keys, special = classify_keys(first)
    log.info(f"  concat_keys={len(concat_keys)} scalar_keys={len(scalar_keys)} special={special}")

    # accumulate buffers
    buffers: Dict[str, List[np.ndarray]] = {k: [] for k in concat_keys}
    sample_dates_buf: List[np.ndarray] = []
    oot_dates_set: List[str] = []
    per_date_n: Dict[str, int] = {}
    scalar_collect: Dict[str, List[float]] = {k: [] for k in scalar_keys}

    for p in paths:
        d = np.load(p, allow_pickle=False)
        # required keys must exist
        if "sample_dates" not in d.files:
            log.warning(f"  skip {p.name}: missing sample_dates")
            continue
        sd = d["sample_dates"]
        n = sd.shape[0]
        # find date string
        if "oot_dates" in d.files and d["oot_dates"].size > 0:
            date_str = str(d["oot_dates"][0])
        else:
            # fall back to filename
            date_str = p.stem.replace("oot_", "")
        per_date_n[date_str] = n
        oot_dates_set.append(date_str)
        sample_dates_buf.append(sd.astype("<U8"))
        for k in concat_keys:
            if k in d.files and d[k].shape[0] == n:
                buffers[k].append(d[k])
            else:
                # pad with zeros + log
                buffers[k].append(np.zeros(n, dtype=d[k].dtype if k in d.files else "float32"))
                log.warning(f"  {p.name}: key {k} missing/mismatched, padded zeros")
        for k in scalar_keys:
            if k in d.files:
                try:
                    scalar_collect[k].append(float(d[k]))
                except Exception:
                    pass

    out: Dict[str, np.ndarray] = {}
    for k in concat_keys:
        out[k] = np.concatenate(buffers[k], axis=0)
    out["sample_dates"] = np.concatenate(sample_dates_buf, axis=0).astype("<U8")
    out["oot_dates"] = np.array(sorted(set(oot_dates_set)), dtype="<U8")
    # average scalar metrics across days (informational)
    for k, vals in scalar_collect.items():
        if vals:
            out[k] = np.array(np.mean(vals), dtype="float64")

    total_n = out["sample_dates"].shape[0]
    log.info(f"  total samples: {total_n:,}  unique dates: {out['oot_dates'].shape[0]}")
    return out


def write_output(arrs: Dict[str, np.ndarray]):
    LOCAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_written = OUT_PATH.with_name(OUT_PATH.stem + ".writing.npz")
    # np.savez_compressed appends .npz only if missing — pass base WITHOUT .npz
    tmp_base = OUT_PATH.with_name(OUT_PATH.stem + ".writing")
    np.savez_compressed(str(tmp_base), **arrs)
    if not tmp_written.exists():
        raise RuntimeError(f"np.savez_compressed did not produce {tmp_written}")
    os.replace(tmp_written, OUT_PATH)
    # cleanup stale tmps
    for stale in OUT_PATH.parent.glob(OUT_PATH.stem + ".*.npz"):
        if stale != OUT_PATH:
            try:
                stale.unlink()
            except OSError:
                pass
    manifest = {
        "out_path": str(OUT_PATH),
        "size_bytes": OUT_PATH.stat().st_size,
        "n_samples": int(arrs["sample_dates"].shape[0]),
        "n_unique_dates": int(arrs["oot_dates"].shape[0]),
        "dates": [str(d) for d in arrs["oot_dates"]],
        "written_utc": datetime.utcnow().isoformat() + "Z",
        "n_keys": len(arrs),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    log.info(f"wrote {OUT_PATH}  ({manifest['size_bytes']/1e6:.1f} MB)")
    log.info(f"manifest: {MANIFEST_PATH}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12, help="(reserved for future parallel concat)")
    ap.add_argument("--remote", action="store_true", default=True,
                    help="Pull from Neptune via rsync (default)")
    ap.add_argument("--no-remote", dest="remote", action="store_false",
                    help="Use local cache only — skip Neptune SSH")
    ap.add_argument("--require", type=int, default=0,
                    help="Require N dates merged or exit 2 (0=accept any)")
    args = ap.parse_args()

    paths = discover_npzs(use_remote=args.remote)
    if not paths:
        log.error("No NPZs available (Neptune unreachable and local cache empty).")
        sys.exit(2)

    log.info(f"discovered {len(paths)}/47 per-date NPZs")
    if args.require and len(paths) < args.require:
        log.error(f"only {len(paths)} dates available, required {args.require}")
        sys.exit(2)

    arrs = concat_npzs(paths)
    if not arrs:
        log.error("Concat produced empty arrays.")
        sys.exit(2)

    manifest = write_output(arrs)
    print(json.dumps({"n_dates": manifest["n_unique_dates"], "n_samples": manifest["n_samples"]}))


if __name__ == "__main__":
    main()
