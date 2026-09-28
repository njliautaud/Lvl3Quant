#!/usr/bin/env python3
"""
HC #432 — Merge two single-leg FIFO fills CSVs (long + short) into a single
weighted ensemble fills CSV that the existing hc432_validate_full.py can
consume.

Approach (50/50 ensemble):
  - Each long fill contributes weight w (default 0.5) of its tick PnL.
  - Each short fill contributes weight (1-w) of its tick PnL.
  - We DO NOT collapse fills onto each other (they happen on different signals
    and different timestamps); instead we keep each fill as a separate row and
    scale its net_ticks / net_dollars / pred_strength by the leg weight.
  - This represents trading both legs simultaneously with 0.5-contract sizing
    on each side. Per-fill metrics (Sharpe, PF, WR) computed on the merged
    stream therefore reflect the joint distribution of half-size fills.

CLI:
  --long-csv   path to long-leg fills CSV
  --short-csv  path to short-leg fills CSV
  --weight     long-leg weight (default 0.5; short-leg gets 1 - weight)
  --out-csv    output merged CSV
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_or_empty(p: Path) -> pd.DataFrame:
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-csv", required=True)
    ap.add_argument("--short-csv", required=True)
    ap.add_argument("--weight", type=float, default=0.5,
                    help="long-leg weight (short = 1 - weight)")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    w_long = float(args.weight)
    w_short = 1.0 - w_long
    if not (0.0 < w_long < 1.0):
        print(f"[ensemble] weight must be in (0,1), got {w_long}", file=sys.stderr)
        return 2

    long_df = load_or_empty(Path(args.long_csv))
    short_df = load_or_empty(Path(args.short_csv))

    if long_df.empty and short_df.empty:
        print("[ensemble] both legs empty; writing empty CSV", file=sys.stderr)
        Path(args.out_csv).write_text("")
        return 0

    parts = []
    if not long_df.empty:
        long_df = long_df.copy()
        long_df["leg_weight"] = w_long
        for col in ("net_ticks", "net_dollars"):
            if col in long_df.columns:
                long_df[col] = long_df[col].astype(float) * w_long
        long_df["leg"] = "long"
        parts.append(long_df)

    if not short_df.empty:
        short_df = short_df.copy()
        short_df["leg_weight"] = w_short
        for col in ("net_ticks", "net_dollars"):
            if col in short_df.columns:
                short_df[col] = short_df[col].astype(float) * w_short
        short_df["leg"] = "short"
        parts.append(short_df)

    merged = pd.concat(parts, ignore_index=True, sort=False)
    # Stable sort by date then entry timestamp for downstream day-grouping
    if "ts_entry_ns" in merged.columns:
        merged = merged.sort_values(["date", "ts_entry_ns"]).reset_index(drop=True)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_p, index=False)

    n_long = len(long_df) if not long_df.empty else 0
    n_short = len(short_df) if not short_df.empty else 0
    print(f"[ensemble] wrote {out_p}  long={n_long}  short={n_short}  total={len(merged)}  "
          f"w_long={w_long}  w_short={w_short}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
