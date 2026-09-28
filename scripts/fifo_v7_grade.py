#!/usr/bin/env python3
"""
HC #493 R3 — Canonical FIFO replay grade for v7 production meta-classifier.

Per HC #493 R1: no alpha claim without FIFO replay.
Reuses alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine (HC #74).
Reuses pattern from scripts/hc450_research/hc450_canonical_replay.py.

What this does:
  1. Load v7 prod concat OOT predictions (1.34M signed preds + dates).
  2. For each OOT date, find per-day v2 NPZ for window_size/stride and validate
     length alignment (v7 concat for that date should equal v2's n_windows).
  3. For each date, select top-pct strongest preds per side (long/short).
  4. Map prediction idx -> MBO ts_ns via the v2 window_size/stride convention.
  5. Feed signals to FIFOReplayEngine — passive limit at touch, h=1s bounds.
  6. Aggregate per-day Sharpe/Sortino/PF/WR + regime stratification.

Outputs:
  output/fifo_v7_grade/summary.csv
  output/fifo_v7_grade/summary.md
  output/fifo_v7_grade/fills.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_DIR = LVL3_ROOT / "output" / "fifo_v7_grade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V7_PRED_NPZ = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

# HC #428 R2 bounds for h=1s (v7's stated horizon)
H_SEC = 1.0
HOLD_S = 1.5
CANCEL_S = 1.0

# Entry params (passive limit at touch)
TP_TICKS = 2.0
SL_TICKS = 1.0

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # canonical

LOG_PATH = OUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("fifo_v7_grade")


# ──────────────────────────────────────────────────────────────────────
# Load + per-day group
# ──────────────────────────────────────────────────────────────────────
def load_v7_perday() -> Dict[str, Dict]:
    """Returns {date: {'preds': np.ndarray (n_windows,), 'window_size': int, 'stride': int}}
    Validates that v7's per-date count matches the v2 per-day file length."""
    log.info(f"Loading v7 concat preds from {V7_PRED_NPZ}")
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    preds = d["predictions"].astype(np.float32)
    dates = d["dates"].astype(str)
    log.info(f"  v7 total preds: {preds.size:,}, unique dates: {len(np.unique(dates))}")

    perday = {}
    for date_str in np.unique(dates):
        mask = dates == date_str
        v7_preds_d = preds[mask]
        v2_npz = V2_DIR / f"{date_str}_predictions.npz"
        if not v2_npz.exists():
            log.warning(f"  {date_str}: no v2 NPZ, skip")
            continue
        v2 = np.load(v2_npz, allow_pickle=False)
        ws = int(v2["window_size"])
        st = int(v2["stride"])
        nw = int(v2["n_windows"])
        if v7_preds_d.size > nw:
            log.warning(f"  {date_str}: v7 has {v7_preds_d.size} preds > v2 n_windows {nw} — SKIP (unexpected)")
            continue
        # v7 commonly has 8-15 fewer windows than v2 due to label-horizon trim in meta training.
        # Treat v7 as the first v7.size windows; idx returned by selector remains valid against v2 ws/stride.
        if v7_preds_d.size < nw:
            log.info(f"  {date_str}: v7={v7_preds_d.size} v2={nw} (trim {nw - v7_preds_d.size}, OK)")
        perday[date_str] = {
            "preds": v7_preds_d,
            "window_size": ws,
            "stride": st,
        }
    log.info(f"  Aligned dates: {len(perday)}")
    return perday


# ──────────────────────────────────────────────────────────────────────
# Per-day top-pct selector
# ──────────────────────────────────────────────────────────────────────
def select_topk_perday(
    perday: Dict[str, Dict],
    side: str,
    pct: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, int, int]]:
    """For each day, select top `pct` per side by |signed strength|."""
    out = {}
    for d, rec in perday.items():
        x = rec["preds"]
        if x.size == 0:
            continue
        if side == "long":
            mask = x > 0
            strength = x
        else:
            mask = x < 0
            strength = -x
        side_s = strength[mask]
        if side_s.size == 0:
            continue
        k = max(1, int(side_s.size * pct))
        thresh = np.partition(side_s, -k)[-k]
        selected = mask & (strength >= thresh)
        idx = np.where(selected)[0]
        if idx.size == 0:
            continue
        out[d] = (idx, strength[idx], rec["window_size"], rec["stride"])
    return out


# ──────────────────────────────────────────────────────────────────────
# Map sample idx → MBO ts_ns
# ──────────────────────────────────────────────────────────────────────
def map_idx_to_ts(date_str: str, idx_in_day: np.ndarray, window_size: int, stride: int) -> Optional[np.ndarray]:
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)
    return ts_events[event_idx]


# ──────────────────────────────────────────────────────────────────────
# Per-day FIFO replay worker
# ──────────────────────────────────────────────────────────────────────
def run_one_date(
    date_str: str,
    idx_in_day: np.ndarray,
    direction: str,
    strength: np.ndarray,
    window_size: int,
    stride: int,
) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    ts_ns = map_idx_to_ts(date_str, idx_in_day, window_size, stride)
    if ts_ns is None:
        return [{"date": date_str, "error": "missing_mbo_events"}]

    signals = [
        {"ts_ns": int(t), "direction": direction, "strength": float(strength[i])}
        for i, t in enumerate(ts_ns)
    ]
    if not signals:
        return []

    cancel_ns = int(CANCEL_S * 1_000_000_000)
    hold_ns = int(HOLD_S * 1_000_000_000)

    try:
        engine = FIFOReplayEngine(
            date=date_str,
            cancel_after_ns=cancel_ns,
            max_hold_ns=hold_ns,
        )
    except FileNotFoundError as e:
        return [{"date": date_str, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date_str, "error": f"engine_init: {e}"}]

    try:
        trades = engine.simulate(
            signals=signals,
            tp_ticks=TP_TICKS,
            sl_ticks=SL_TICKS,
            order_type="limit",
        )
    except Exception as e:
        return [{"date": date_str, "error": f"simulate: {e}"}]

    fills = []
    for t in trades:
        hold_s = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        fills.append({
            "date": date_str,
            "direction": t.direction,
            "hold_s": hold_s,
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "queue_ahead": int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
        })
    return fills


# ──────────────────────────────────────────────────────────────────────
# Regime
# ──────────────────────────────────────────────────────────────────────
def load_regime_labels() -> Optional[pd.DataFrame]:
    if not REGIME_PARQUET.exists():
        log.warning("Regime parquet missing — skipping regime stratification.")
        return None
    df = pd.read_parquet(REGIME_PARQUET)
    df["date"] = df["date"].astype(str).str.zfill(8)
    def classify(r):
        delta = r["close_minus_open_ticks"]
        if delta >= 4: return "green"
        if delta <= -4: return "red"
        return "flat"
    df["regime"] = df.apply(classify, axis=1)
    return df[["date", "regime", "close_minus_open_ticks"]]


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def metrics_for(fills_df: pd.DataFrame) -> dict:
    if fills_df.empty:
        return {k: np.nan for k in [
            "n", "n_days", "mean_tk_net", "sharpe_ann",
            "sortino_ann", "pf", "wr", "day_pos_pct",
        ]}
    nets = fills_df["net_ticks"].values.astype(np.float64)
    n = len(nets)
    daily = fills_df.groupby("date")["net_ticks"].sum()
    n_days = daily.size
    mean_tk = float(nets.mean())
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 and n_days > 1 else float("nan")
    down = daily[daily < 0]
    sortino = float(daily.mean() / down.std(ddof=1) * np.sqrt(252)) if down.size >= 1 and down.std(ddof=1) > 0 else (float("inf") if daily.mean() > 0 else float("nan"))
    wins = nets[nets > 0].sum()
    losses = -nets[nets < 0].sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    wr = float((nets > 0).mean())
    day_pos = float((daily > 0).mean())
    return {
        "n": n, "n_days": n_days, "mean_tk_net": mean_tk,
        "sharpe_ann": sharpe, "sortino_ann": sortino,
        "pf": pf, "wr": wr, "day_pos_pct": day_pos,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=0.05, help="top fraction per day (e.g. 0.05 = top 5%)")
    ap.add_argument("--side", choices=["short", "long", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    log.info(f"=== HC #493 R3 — v7 prod FIFO grade ===")
    log.info(f"  side={args.side}, pct={args.pct}, workers={args.workers}")
    log.info(f"  TP={TP_TICKS}t SL={SL_TICKS}t hold={HOLD_S}s cancel={CANCEL_S}s")

    perday = load_v7_perday()
    if not perday:
        log.error("No aligned dates — aborting.")
        sys.exit(1)

    sides = ["short", "long"] if args.side == "both" else [args.side]

    all_fills_rows: List[dict] = []
    for side in sides:
        selected = select_topk_perday(perday, side, args.pct)
        log.info(f"  side={side}: {len(selected)} dates with selections")
        direction = "short" if side == "short" else "long"

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {
                ex.submit(run_one_date, d, idx, direction, strength, ws, st): d
                for d, (idx, strength, ws, st) in selected.items()
            }
            for i, fut in enumerate(as_completed(futures)):
                d = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    log.error(f"  {d}: worker crashed: {e}")
                    continue
                all_fills_rows.extend(rows)
                if (i + 1) % 5 == 0:
                    log.info(f"  side={side}: {i+1}/{len(selected)} dates done")

    if not all_fills_rows:
        log.error("No fills produced. Check engine / data paths.")
        sys.exit(1)

    df = pd.DataFrame(all_fills_rows)
    err_mask = "error" in df.columns and df["error"].notna() if "error" in df.columns else None
    if "error" in df.columns:
        err_df = df[df["error"].notna()]
        log.warning(f"Errored dates: {len(err_df)} — {err_df['date'].unique()[:5].tolist()}...")
        df = df[df["error"].isna()].drop(columns=["error"])

    if df.empty:
        log.error("All dates errored — abort.")
        sys.exit(1)

    df.to_parquet(OUT_DIR / "fills.parquet")
    log.info(f"  Wrote fills.parquet: {len(df):,} rows")

    # Overall + per-side metrics
    rows = []
    rows.append({"cohort": "overall", **metrics_for(df)})
    for side, sdir in [("short", "short"), ("long", "long")]:
        m = metrics_for(df[df["direction"] == sdir])
        rows.append({"cohort": f"{side}_only", **m})

    # Regime stratification
    reg = load_regime_labels()
    if reg is not None:
        merged = df.merge(reg, on="date", how="left")
        for regime in ["green", "red", "flat"]:
            m = metrics_for(merged[merged["regime"] == regime])
            rows.append({"cohort": f"regime_{regime}", **m})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    log.info("\n" + summary.to_string(index=False))

    # Markdown summary
    md = ["# HC #493 R3 — v7 prod FIFO replay grade", ""]
    md.append(f"- side: {args.side}, top-pct per day: {args.pct}")
    md.append(f"- TP={TP_TICKS}t SL={SL_TICKS}t hold≤{HOLD_S}s cancel≤{CANCEL_S}s")
    md.append(f"- Engine: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74)")
    md.append(f"- v7 prod concat preds: {V7_PRED_NPZ}")
    md.append("")
    md.append(summary.to_markdown(index=False))
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    log.info("Wrote summary.md")
    log.info("DONE.")


if __name__ == "__main__":
    main()
