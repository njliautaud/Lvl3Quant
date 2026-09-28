#!/usr/bin/env python3
"""
hc471_fifo_v342_x_continuation.py — HC #471 R4 LITERAL DELIVERABLE.

Takes the +3.40-ticks-net-passive headline setup (HC #467/#470 mask) and re-runs
it through CANONICAL FIFO MARKET REPLAY (HC #74 / HC #469 R3) to answer:

  Does the +3.40 ticks net survive REALISTIC queue position?

Mask: (10s pred top-1% directional) × (continuation prob top-20%).
Direction: long if pred_log_ret_10s > 0, short if < 0.
Per-day quantile cuts (no look-ahead — each day's thresholds from its own dist).

FIFO config (same as surviving_confluence_canonical_fifo.py):
  TP=4 ticks, SL=3 ticks, hold=30s, cancel=10s, passive_at_touch (limit at touch).

Inputs:
  /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_*.npz
  /home/jupiter/Lvl3Quant/output/continuation_specialist_smoke/preds_mlp_*.npz

Outputs:
  /home/jupiter/Lvl3Quant/output/hc471_fifo_v342_x_continuation/
    - REPORT.md            (plain-English summary)
    - fills.parquet        (per-trade fills)
    - per_day.parquet      (per-day aggregation)
    - summary.json         (headline numbers)
    - DONE                 (completion marker)

Comparison columns per HC #469 R6: (a) naive fill-at-touch always,
(b) canonical FIFO with realistic queue position. The gap is what HC #471 cares about.

Run AFTER surviving_confluence_canonical_fifo.py finishes (shares disk/IO).
"""
from __future__ import annotations
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v3_4_research"))

from hc432_fifo_full_market_replay import run_one_date  # noqa: E402

# Constants
V4_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
CS_DIR = ROOT / "output/continuation_specialist_smoke"
OUT_DIR = ROOT / "output/hc471_fifo_v342_x_continuation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ES futures cost constants (HC #74)
COMM_TICKS = 0.376  # round-trip commission only (passive limits)

# HC #471 R4 mask: top-1% directional × top-20% continuation
DIR_TOP_PCT = 0.01
CONT_TOP_PCT = 0.20

# FIFO replay config (same as surviving_confluence_canonical_fifo.py)
TP_TICKS = 4.0
SL_TICKS = 3.0
HOLD_S = 30.0
CANCEL_S = 10.0
ORDER_TYPE = "passive_at_touch"


def find_day_pairs() -> List[tuple]:
    """Return [(date_str, cs_path, v4_path), ...] for days where BOTH exist."""
    pairs = []
    for cs_path in sorted(glob.glob(str(CS_DIR / "preds_mlp_*.npz"))):
        date = Path(cs_path).stem.split("_")[-1]
        v4_path = V4_DIR / f"oot_{date}.npz"
        if v4_path.exists():
            pairs.append((date, cs_path, str(v4_path)))
    return pairs


def build_mask(pred_dir: np.ndarray, cont_prob: np.ndarray) -> tuple:
    """Per-day top-1% directional × top-20% continuation. Returns (long_mask, short_mask)."""
    # Directional: top-1% by |pred_dir|
    abs_pred = np.abs(pred_dir)
    dir_thr = np.quantile(abs_pred, 1.0 - DIR_TOP_PCT)
    dir_mask = abs_pred >= dir_thr

    # Continuation: top-20% by prob (higher prob = more likely to continue)
    cont_thr = np.quantile(cont_prob, 1.0 - CONT_TOP_PCT)
    cont_mask = cont_prob >= cont_thr

    combined = dir_mask & cont_mask
    long_mask = combined & (pred_dir > 0)
    short_mask = combined & (pred_dir < 0)
    return long_mask, short_mask, dir_thr, cont_thr


def run_day(date: str, v4_path: str, cs_path: str) -> Dict:
    """Build mask, run FIFO replay, return per-day aggregation + fills."""
    v4 = np.load(v4_path, allow_pickle=True)
    cs = np.load(cs_path)
    pred_10s = v4["pred_log_ret_10s"]
    cont_prob = cs["prob"]
    target_30s = v4["target_log_ret_30s"]  # realized 30s ticks (per existing header doc)

    # Align: continuation NPZ has same per-day index as v3.4.2 per-day NPZ?
    # Continuation pred_npz length should match v4 per-day length (both built off same events).
    n_v4 = pred_10s.shape[0]
    n_cs = cont_prob.shape[0]
    if n_v4 != n_cs:
        # Truncate to min — confluence_matrix_v342_x_continuation does the same
        n = min(n_v4, n_cs)
        pred_10s = pred_10s[:n]
        cont_prob = cont_prob[:n]
        target_30s = target_30s[:n]

    long_mask, short_mask, dir_thr, cont_thr = build_mask(pred_10s, cont_prob)
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    print(f"  [{date}] dir_thr={dir_thr:.4f} cont_thr={cont_thr:.4f} long={n_long} short={n_short}")

    naive_fills = []
    fifo_fills = []

    for side, mask in [("long", long_mask), ("short", short_mask)]:
        if not mask.any():
            continue
        idx_in_day = np.flatnonzero(mask).astype(np.int64)
        strength = np.ones(idx_in_day.size, dtype=np.float64)

        # (a) Naive: fill-at-touch assumed, P&L = realized target_log_ret_30s in ticks * sign
        sign = 1.0 if side == "long" else -1.0
        naive_ticks = sign * target_30s[idx_in_day]
        for i, t in zip(idx_in_day, naive_ticks):
            naive_fills.append({
                "date": date, "side": side, "idx": int(i),
                "gross_ticks": float(t),
                "net_passive_ticks": float(t - COMM_TICKS),
            })

        # (b) Canonical FIFO
        fills = run_one_date(
            date_str=date,
            idx_in_day=idx_in_day,
            direction=side,
            strength=strength,
            tp_ticks=TP_TICKS,
            sl_ticks=SL_TICKS,
            hold_s=HOLD_S,
            cancel_s=CANCEL_S,
            order_type=ORDER_TYPE,
        )
        for f in fills:
            if "error" in f:
                print(f"    FIFO error {date} {side}: {f['error']}")
                continue
            f["date"] = date
            f["side"] = side
            fifo_fills.append(f)

    return {"date": date, "naive": naive_fills, "fifo": fifo_fills}


def aggregate(naive_all: List[dict], fifo_all: List[dict]) -> Dict:
    naive_df = pd.DataFrame(naive_all) if naive_all else pd.DataFrame()
    fifo_df = pd.DataFrame(fifo_all) if fifo_all else pd.DataFrame()

    out = {"naive": {}, "fifo": {}}
    if not naive_df.empty:
        out["naive"] = {
            "n_trades": int(len(naive_df)),
            "gross_ticks_total": float(naive_df["gross_ticks"].sum()),
            "gross_ticks_mean": float(naive_df["gross_ticks"].mean()),
            "net_passive_ticks_total": float(naive_df["net_passive_ticks"].sum()),
            "net_passive_ticks_mean": float(naive_df["net_passive_ticks"].mean()),
            "hit_rate": float((naive_df["gross_ticks"] > 0).mean()),
        }
    if not fifo_df.empty and "net_ticks" in fifo_df.columns:
        filled = fifo_df[fifo_df.get("fill_type", pd.Series([], dtype=str)) != "no_fill"]
        out["fifo"] = {
            "n_signals": int(len(fifo_df)),
            "n_filled": int(len(filled)),
            "fill_rate": float(len(filled) / len(fifo_df)) if len(fifo_df) else 0.0,
            "net_ticks_total": float(filled["net_ticks"].sum()) if len(filled) else 0.0,
            "net_ticks_mean": float(filled["net_ticks"].mean()) if len(filled) else 0.0,
            "hit_rate": float((filled["net_ticks"] > 0).mean()) if len(filled) else 0.0,
        }
    return out, naive_df, fifo_df


def write_report(summary: Dict, n_days: int, wall_s: float) -> None:
    lines = [
        "# HC #471 R4 — DOES +3.40 TICKS NET SURVIVE CANONICAL FIFO REPLAY?",
        "",
        f"Setup: 10s pred top-1% × continuation top-20%. Per-day quantiles.",
        f"FIFO config: TP=4, SL=3, hold=30s, cancel=10s, passive_at_touch.",
        f"Days: {n_days}. Wall: {wall_s:.0f}s.",
        "",
        "## (a) NAIVE fill-at-touch (the +3.40 headline assumption)",
    ]
    naive = summary.get("naive", {})
    if naive:
        lines += [
            f"  n_trades={naive['n_trades']}",
            f"  gross_mean={naive['gross_ticks_mean']:+.3f} ticks",
            f"  net_passive_mean={naive['net_passive_ticks_mean']:+.3f} ticks",
            f"  net_passive_total={naive['net_passive_ticks_total']:+.2f} ticks",
            f"  hit_rate={naive['hit_rate']:.1%}",
        ]
    else:
        lines += ["  (no trades)"]
    lines += ["", "## (b) CANONICAL FIFO with realistic queue position"]
    fifo = summary.get("fifo", {})
    if fifo:
        lines += [
            f"  n_signals={fifo['n_signals']}",
            f"  n_filled={fifo['n_filled']}",
            f"  fill_rate={fifo['fill_rate']:.1%}",
            f"  net_ticks_mean={fifo['net_ticks_mean']:+.3f} ticks",
            f"  net_ticks_total={fifo['net_ticks_total']:+.2f} ticks",
            f"  hit_rate={fifo['hit_rate']:.1%}",
        ]
    else:
        lines += ["  (no fills)"]
    lines += ["", "## VERDICT"]
    if naive and fifo:
        gap = naive["net_passive_ticks_mean"] - fifo["net_ticks_mean"]
        lines.append(f"  Naive→FIFO gap = {gap:+.3f} ticks (lost to queue dynamics).")
        if fifo["net_ticks_mean"] > 0:
            lines.append("  ✅ POSITIVE under canonical FIFO. Alpha survives queue position.")
        else:
            lines.append("  ❌ NEGATIVE under canonical FIFO. Queue position kills the edge.")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    t0 = time.time()
    pairs = find_day_pairs()
    print(f"Found {len(pairs)} day-pairs (continuation × v3.4.2)")
    if not pairs:
        print("ERROR: no day pairs"); sys.exit(1)

    naive_all: List[dict] = []
    fifo_all: List[dict] = []
    for date, cs_path, v4_path in pairs:
        try:
            d = run_day(date, v4_path, cs_path)
            naive_all.extend(d["naive"])
            fifo_all.extend(d["fifo"])
        except Exception as e:
            print(f"  ERROR {date}: {e}")

    summary, naive_df, fifo_df = aggregate(naive_all, fifo_all)
    if not naive_df.empty:
        naive_df.to_parquet(OUT_DIR / "naive_fills.parquet", index=False)
    if not fifo_df.empty:
        fifo_df.to_parquet(OUT_DIR / "fifo_fills.parquet", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    wall_s = time.time() - t0
    write_report(summary, len(pairs), wall_s)
    (OUT_DIR / "DONE").write_text(f"completed {time.strftime('%Y-%m-%d %H:%M:%S ET')}\nwall_s: {wall_s:.0f}\n")


if __name__ == "__main__":
    main()
