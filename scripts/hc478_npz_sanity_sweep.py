#!/usr/bin/env python3
"""
HC #478 R2 — Standard sanity sweep on every NPZ artifact under output/.

For each NPZ file, for each array, compute:
  shape, dtype, n_total, n_nan, n_inf, n_zero, min, max, mean, std, all_zero_flag

Flag any array with:
  - std == 0 (constant column — like the HC #477 long-horizon labels)
  - all_nan
  - all_zero
  - n_nan / n_total > 0.10
  - non-finite min/max in float arrays

Output:
  - reports/hc478_audit/<timestamp>/per_file.jsonl   (one record per NPZ)
  - reports/hc478_audit/<timestamp>/flagged.md       (human-readable summary of problems)
  - reports/hc478_audit/<timestamp>/SUMMARY.md       (counts + headline)

Author: HC #478 R2 (2026-05-21).
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_ROOT = ROOT / "output"
REPORT_ROOT = ROOT / "reports" / "hc478_audit"

# Skip these path patterns (scalers, tmp dirs, etc.)
SKIP_PATTERNS = ("scaler", "/tmp/", "/.checkpoints/")

# Flags
NAN_FRACTION_FLAG = 0.10
CONSTANT_STD_EPS = 0.0  # exactly zero std = constant column


def audit_array(arr: np.ndarray) -> dict:
    """Return a sanity-sweep record for one numpy array."""
    rec = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_total": int(arr.size),
    }
    if arr.size == 0:
        rec["empty"] = True
        return rec

    is_floating = np.issubdtype(arr.dtype, np.floating)
    if is_floating:
        finite = np.isfinite(arr)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int((~finite & ~np.isnan(arr)).sum())
        n_finite = int(finite.sum())
        rec["n_nan"] = n_nan
        rec["n_inf"] = n_inf
        rec["nan_fraction"] = n_nan / arr.size if arr.size else 0.0
        if n_finite > 0:
            finite_vals = arr[finite]
            rec["min"] = float(finite_vals.min())
            rec["max"] = float(finite_vals.max())
            rec["mean"] = float(finite_vals.mean())
            rec["std"] = float(finite_vals.std())
            rec["n_zero"] = int((finite_vals == 0).sum())
            rec["zero_fraction"] = rec["n_zero"] / arr.size
        else:
            rec["all_non_finite"] = True
    else:
        # Integer / bool / other
        rec["n_nan"] = 0
        rec["n_inf"] = 0
        rec["nan_fraction"] = 0.0
        try:
            rec["min"] = float(arr.min())
            rec["max"] = float(arr.max())
            rec["mean"] = float(arr.mean())
            rec["std"] = float(arr.std())
            rec["n_zero"] = int((arr == 0).sum())
            rec["zero_fraction"] = rec["n_zero"] / arr.size
        except Exception as e:  # object arrays, strings, etc.
            rec["non_numeric"] = True
            rec["error"] = str(e)
            return rec

    # Flags
    flags = []
    if rec.get("nan_fraction", 0.0) >= 1.0:
        flags.append("all_nan")
    elif rec.get("nan_fraction", 0.0) > NAN_FRACTION_FLAG:
        flags.append(f"high_nan({rec['nan_fraction']:.2%})")
    if rec.get("zero_fraction", 0.0) >= 1.0:
        flags.append("all_zero")
    if "std" in rec and rec["std"] == CONSTANT_STD_EPS and not rec.get("all_non_finite"):
        flags.append("constant_std0")
    if rec.get("n_inf", 0) > 0:
        flags.append(f"has_inf({rec['n_inf']})")
    rec["flags"] = flags
    return rec


def audit_npz(path: Path) -> dict:
    """Audit every array inside one NPZ file. Returns flat record."""
    out = {"file": str(path), "size_bytes": path.stat().st_size}
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {}
            for key in data.files:
                try:
                    arr = data[key]
                    arrays[key] = audit_array(arr)
                except Exception as e:
                    arrays[key] = {"error": f"load_failed: {e}"}
            out["arrays"] = arrays
    except Exception as e:
        out["error"] = f"open_failed: {e}"
        out["traceback"] = traceback.format_exc()
    return out


def is_skipped(p: Path) -> bool:
    s = str(p)
    return any(sp in s for sp in SKIP_PATTERNS)


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = REPORT_ROOT / ts
    report_dir.mkdir(parents=True, exist_ok=True)

    per_file_path = report_dir / "per_file.jsonl"
    flagged_md = report_dir / "flagged.md"
    summary_md = report_dir / "SUMMARY.md"
    log_path = report_dir / "audit.log"

    def log(msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log(f"HC #478 R2 sanity sweep starting. report_dir={report_dir}")

    npz_files = []
    for p in OUTPUT_ROOT.rglob("*.npz"):
        if not is_skipped(p):
            npz_files.append(p)
    log(f"Discovered {len(npz_files)} NPZ files for audit")

    n_audited = 0
    n_with_flags = 0
    flagged_records = []  # (file, array_name, flags, summary_stats)

    with per_file_path.open("w") as fout:
        for p in sorted(npz_files):
            rec = audit_npz(p)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_audited += 1

            # Pull flagged arrays
            file_flag_count = 0
            for arr_name, arr_rec in rec.get("arrays", {}).items():
                flags = arr_rec.get("flags") or []
                if flags:
                    file_flag_count += 1
                    flagged_records.append({
                        "file": rec["file"].replace(str(ROOT), ""),
                        "array": arr_name,
                        "flags": flags,
                        "shape": arr_rec.get("shape"),
                        "n_total": arr_rec.get("n_total"),
                        "std": arr_rec.get("std"),
                        "min": arr_rec.get("min"),
                        "max": arr_rec.get("max"),
                        "nan_fraction": arr_rec.get("nan_fraction"),
                        "zero_fraction": arr_rec.get("zero_fraction"),
                    })
            if file_flag_count > 0:
                n_with_flags += 1
                log(f"  FLAGGED {file_flag_count} arrays in {p.name}")
            if n_audited % 25 == 0:
                log(f"  progress: {n_audited}/{len(npz_files)} files audited, "
                    f"{n_with_flags} flagged")

    log(f"Audit complete. {n_audited} files audited, {n_with_flags} files with flags, "
        f"{len(flagged_records)} flagged arrays total")

    # Write flagged.md (human-readable)
    with flagged_md.open("w") as f:
        f.write(f"# HC #478 R2 — Flagged arrays ({ts})\n\n")
        f.write(f"**Total NPZ files audited**: {n_audited}\n")
        f.write(f"**Files with at least one flagged array**: {n_with_flags}\n")
        f.write(f"**Total flagged arrays**: {len(flagged_records)}\n\n")
        f.write("## Flag legend\n")
        f.write("- `constant_std0` — std=0, array is a single constant value (like the HC #477 60s/5min labels)\n")
        f.write("- `all_zero` — every element is zero\n")
        f.write("- `all_nan` — every element is NaN\n")
        f.write("- `high_nan(p%)` — more than 10% NaN\n")
        f.write("- `has_inf(n)` — array contains infinity values\n\n")

        # Group by flag type for triage
        by_flag = {}
        for r in flagged_records:
            for fl in r["flags"]:
                # Strip parens for grouping
                key = fl.split("(")[0]
                by_flag.setdefault(key, []).append(r)

        for flag_key, recs in sorted(by_flag.items(),
                                     key=lambda kv: -len(kv[1])):
            f.write(f"## Flag: `{flag_key}` ({len(recs)} arrays)\n\n")
            f.write("| File | Array | Shape | Std | Min | Max | NaN frac | Zero frac |\n")
            f.write("|------|-------|-------|-----|-----|-----|----------|-----------|\n")
            for r in recs[:200]:  # cap per flag-type
                std = f"{r['std']:.3g}" if r['std'] is not None else "-"
                mn = f"{r['min']:.3g}" if r['min'] is not None else "-"
                mx = f"{r['max']:.3g}" if r['max'] is not None else "-"
                nf = f"{r['nan_fraction']:.2%}" if r['nan_fraction'] is not None else "-"
                zf = f"{r['zero_fraction']:.2%}" if r['zero_fraction'] is not None else "-"
                f.write(f"| {r['file']} | `{r['array']}` | {r['shape']} | {std} | {mn} | {mx} | {nf} | {zf} |\n")
            if len(recs) > 200:
                f.write(f"\n_(+{len(recs)-200} more rows truncated; see per_file.jsonl)_\n")
            f.write("\n")

    with summary_md.open("w") as f:
        f.write(f"# HC #478 R2 — Sanity Sweep SUMMARY ({ts})\n\n")
        f.write(f"- NPZ files audited: **{n_audited}**\n")
        f.write(f"- Files with flagged arrays: **{n_with_flags}**\n")
        f.write(f"- Flagged arrays total: **{len(flagged_records)}**\n\n")
        f.write("See `flagged.md` for grouped triage list, `per_file.jsonl` for raw records.\n")

    log("Reports written.")
    log(f"  per_file.jsonl  ({per_file_path.stat().st_size} bytes)")
    log(f"  flagged.md      ({flagged_md.stat().st_size} bytes)")
    log(f"  SUMMARY.md      ({summary_md.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
