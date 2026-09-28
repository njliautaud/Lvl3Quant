#!/usr/bin/env python3
"""
HC #419 Step A2 — Reproduce the v2_1s_short_top05 backtest verdict.

Purpose
-------
Before bringing v2_1s_short_top05 live on Razer, prove the +0.274 tk/fill,
n=639, WR 84.8% headline numbers can be REGENERATED on demand from the
existing prediction NPZ + FIFO labels + MFE config. If this reproduces,
we have high confidence the live spec is grounded in a stable backtest.

What it does (pure orchestration — no algorithmic code)
-------------------------------------------------------
1. Re-invokes the canonical HC #413 scalping backtester as a subprocess
   (the same script that produced the original verdict CSV).
2. Targets:
       NPZ        : output/hc417_v2_full_oot_wrapped_for_hc413.npz
       MFE config : output/hc411_regime_agnostic_20260517_215211/...
       Cell       : v2_1s_short_top05  (1s × short × top0.5%)
3. Parses the resulting scalping_backtest_results.csv.
4. Asserts headline numbers vs the deploy-spec values from
   output/hc417_v2_1s_short_top05_DEPLOYMENT_SPEC.md §2.
5. Prints PASS/FAIL.

Pass criteria (deploy spec §2 headline tolerances)
--------------------------------------------------
  n_fills            = 639      ± 5
  net_per_fill (tk)  = +0.2743  ± 0.01
  wr (%)             = 84.82    ± 1.0
  day_conc           = 0.1315   ± 0.02
  ci95_lo (tk)       = +0.2318  ± 0.02

Optional second pass (Step B.2 deliverable, deferred)
-----------------------------------------------------
If --extended-npz <path> is supplied, also re-wrap and re-run on the
56d/61d NPZ once Jupiter PID 960545 finishes the missing-dates gap-fill.
For now (Step A2 / Step B), we just verify the wrapped_for_hc413 NPZ.

Usage
-----
    python scripts/v2_1s_short_top05_reproduce_backtest.py
    python scripts/v2_1s_short_top05_reproduce_backtest.py --output-dir output/repro_$(date +%Y%m%d_%H%M%S)

Exit code 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

LVL3 = Path("/home/jupiter/Lvl3Quant")

# -- Inputs (canonical) -------------------------------------------------------
DEFAULT_WRAPPED_NPZ = LVL3 / "output/hc417_v2_full_oot_wrapped_for_hc413.npz"
DEFAULT_MFE_CONFIG = (
    LVL3 / "output/hc417_v2_native_mfe_matrix.csv"
)
DEFAULT_LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
BACKTESTER = LVL3 / "scripts/hc413_scalping_backtester/backtester.py"

# -- Expected values from deploy spec §2 + §9 appendix row --------------------
TARGET_CELL_ID = "v2_1s_short_top05"
EXPECTED: Dict[str, Tuple[float, float]] = {
    # field name -> (expected_value, abs_tolerance)
    "n_fills":              (639.0,    5.0),
    "realized_net_per_fill":(0.27428,  0.01),
    "wr":                   (84.82,    1.0),
    "day_conc":             (0.13153,  0.02),
    "ci_low_95_net":        (0.23179,  0.02),
    "n_tp1_hits":           (100.0,   15.0),  # softer tolerance
    "n_tp2_hits":           (442.0,   20.0),
    "n_sl_hits":            (95.0,    15.0),
}

# Risk-adjusted metrics we ALSO log (not gate-blocking, but reported)
REPORT_ONLY = ["sharpe_sqrtN", "sortino_sqrtN", "pf",
               "tp1", "tp2", "sl", "entry_cost_ticks"]


def run_backtester(npz: Path, out_dir: Path, mfe_cfg: Path,
                   labels_dir: Path, seed: int = 42) -> Path:
    """Invoke the hc413 backtester targeting cell v2_1s_short_top05.

    Returns the path to scalping_backtest_results.csv.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(BACKTESTER),
        "--npz", str(npz),
        "--mfe-config", str(mfe_cfg),
        "--output-dir", str(out_dir),
        "--confidence-tier", "top05",
        "--horizon", "1s",
        "--side", "short",
        "--model", "v2",
        "--order-type", "passive_at_touch",
        "--cancel-eval-window", "40",
        "--labels-dir", str(labels_dir),
        "--seed", str(seed),
    ]
    print(f"[repro] CMD: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    if rc.returncode != 0:
        sys.exit(f"[repro] FAIL: backtester exited rc={rc.returncode}")
    print(f"[repro] backtester completed in {dt:.1f}s")
    csv_path = out_dir / "scalping_backtest_results.csv"
    if not csv_path.exists():
        sys.exit(f"[repro] FAIL: expected CSV not found: {csv_path}")
    return csv_path


def load_cell_row(csv_path: Path, cell_id: str) -> Dict[str, str]:
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r["cell_id"] == cell_id:
                return r
    sys.exit(f"[repro] FAIL: cell_id={cell_id!r} not found in {csv_path}")


def check_pass(row: Dict[str, str]) -> Tuple[bool, list]:
    """Return (overall_pass, per_field_results list).

    per_field_results: list of dict(name, actual, expected, tol, pass).
    """
    results = []
    overall = True
    for name, (expected, tol) in EXPECTED.items():
        raw = row.get(name, "")
        try:
            actual = float(raw)
        except (TypeError, ValueError):
            results.append({"name": name, "actual": raw, "expected": expected,
                            "tol": tol, "pass": False, "err": "non-numeric"})
            overall = False
            continue
        ok = abs(actual - expected) <= tol
        if not ok:
            overall = False
        results.append({"name": name, "actual": actual, "expected": expected,
                        "tol": tol, "pass": ok})
    return overall, results


def print_report(row: Dict[str, str], results: list, overall: bool,
                 csv_path: Path) -> None:
    print()
    print("=" * 72)
    print(f"HC #419 STEP A2 — v2_1s_short_top05 BACKTEST REPRODUCTION")
    print("=" * 72)
    print(f"Source CSV: {csv_path}")
    print(f"Cell:       {row['cell_id']}  (entry={row.get('order_type', '?')}, "
          f"cost={row.get('entry_cost_ticks', '?')} tk)")
    print()
    print(f"{'FIELD':<28} {'ACTUAL':>14} {'EXPECTED':>14} {'TOL':>10}  RESULT")
    print("-" * 72)
    for r in results:
        a = r["actual"]; e = r["expected"]; t = r["tol"]
        astr = f"{a:.5f}" if isinstance(a, float) else str(a)
        marker = "PASS" if r["pass"] else "FAIL"
        print(f"{r['name']:<28} {astr:>14} {e:>14.5f} {t:>10.3f}  {marker}")
    print()
    print("Report-only metrics (not gated):")
    for k in REPORT_ONLY:
        v = row.get(k, "?")
        try:
            vf = float(v)
            print(f"  {k:<24} = {vf:.4f}")
        except (TypeError, ValueError):
            print(f"  {k:<24} = {v}")
    print()
    # Risk-adjusted summary
    try:
        net_tk = float(row["realized_net_per_fill"])
        wr     = float(row["wr"])
        pf     = float(row["pf"])
        sharpe = float(row["sharpe_sqrtN"])
        sortino = float(row["sortino_sqrtN"])
        n      = int(float(row["n_fills"]))
        dollars = net_tk * 12.50
        print(f"Headline: n={n}  net=+{net_tk:.3f}tk (+${dollars:.2f}/fill)  "
              f"WR={wr:.1f}%  PF={pf:.2f}  Sharpe√N={sharpe:.2f}  Sortino√N={sortino:.1f}")
    except (KeyError, ValueError, TypeError) as e:
        print(f"(headline summary unavailable: {e})")
    print()
    print("=" * 72)
    print(f"OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(DEFAULT_WRAPPED_NPZ),
                    help="Wrapped-for-hc413 prediction NPZ (default: 36d v2 OOT)")
    ap.add_argument("--mfe-config", default=str(DEFAULT_MFE_CONFIG),
                    help="MFE-at-confidence matrix CSV")
    ap.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR),
                    help="FIFO labels directory")
    ap.add_argument("--output-dir",
                    default=str(LVL3 / f"output/hc419_step_a2_repro_{time.strftime('%Y%m%d_%H%M%S')}"),
                    help="Where to drop the CSV + verdict.md")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    npz = Path(args.npz)
    mfe = Path(args.mfe_config)
    labels = Path(args.labels_dir)
    out_dir = Path(args.output_dir)

    # Pre-flight: every input must exist.
    for label, path in (("NPZ", npz), ("MFE config", mfe),
                        ("Labels dir", labels), ("Backtester", BACKTESTER)):
        if not path.exists():
            sys.exit(f"[repro] FAIL: {label} not found: {path}")

    print(f"[repro] inputs OK. running hc413 backtester...")
    csv_path = run_backtester(npz, out_dir, mfe, labels, seed=args.seed)
    row = load_cell_row(csv_path, TARGET_CELL_ID)
    overall, results = check_pass(row)
    print_report(row, results, overall, csv_path)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
