#!/usr/bin/env python3
"""
HC #493 R3 — Canonical FIFO replay grade for the symmetric quantile-pinball
DLinear LONG-side claim (+0.65 ticks/trade lift at h=1s, 20260427+20260428).

Reuses alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine (HC #74),
same harness used for v7 regrade and confluence-short regrade earlier today.

Setup definition (per SESSION_STATE_ARCHIVE_2026-W21.md @ 2026-05-22 19:30 ET):
  - Predictions:  output/hc488_dlinear_quantile_v1/fold_{21,22}_preds.npz
                  (fold_21 = 20260427, fold_22 = 20260428)
  - Selection:    top-1% LONG per day by P50_1s (median quantile prediction)
                  HORIZON h = 1s (matches the "+0.65 lift at 1s" claim)
  - Window/stride: WINDOW=500, OOT_STRIDE=5 (per train_dlinear_mse_baseline_v1.py)
                   index in NPZ -> event index via valid_starts() recomputation
  - TP/SL/hold:   TP=2.0 ticks, SL=1.0 ticks, hold<=1.5s, cancel<=1.0s
                   (HC #428 R2 compliant for 1s horizon; identical to v7)
  - Costs:        ES_RT_COMMISSION_TICKS=0.376 (canonical)
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

OUT_DIR = LVL3_ROOT / "output" / "fifo_quantile_long_grade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = LVL3_ROOT / "output" / "hc488_dlinear_quantile_v1"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

# Setup fixed parameters
WINDOW = 500
OOT_STRIDE = 5
DATE_FOLDS = [("20260427", 21), ("20260428", 22)]

# HC #428 R2 bounds for h=1s
H_SEC = 1.0
HOLD_S = 1.5
CANCEL_S = 1.0

TP_TICKS = 2.0
SL_TICKS = 1.0

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376

LOG_PATH = OUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("fifo_quantile_long_grade")


def recompute_valid_starts(date: str) -> np.ndarray:
    """Recompute the same valid_starts array as train_dlinear used at OOT time."""
    f = MBO_EVENT_DIR / f"{date}_mbo_events.npz"
    d = np.load(f, allow_pickle=False)
    l1 = d["labels_1s"]
    l5 = d["labels_5s"]
    l10 = d["labels_10s"]
    n = len(l1)
    starts = np.arange(0, n - WINDOW + 1, OOT_STRIDE)
    end_idx = starts + WINDOW - 1
    mask = ~(np.isnan(l1[end_idx]) | np.isnan(l5[end_idx]) | np.isnan(l10[end_idx]))
    return starts[mask].astype(np.int64)


def load_p50_1s(fold_id: int) -> np.ndarray:
    f = PRED_DIR / f"fold_{fold_id:02d}_preds.npz"
    d = np.load(f, allow_pickle=False)
    return d["P50_1s"].astype(np.float32)


def select_topk_long(p50: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray]:
    """Top-pct LONG by P50_1s — highest predictions."""
    n = p50.size
    k = max(1, int(n * pct))
    # argpartition for top-k
    top_idx = np.argpartition(p50, -k)[-k:]
    # Threshold gate (strict): all selected have score >= threshold
    return top_idx, p50[top_idx]


def map_npz_idx_to_event_ts(date: str, npz_idx: np.ndarray) -> np.ndarray:
    """Map index within the NPZ -> ts_ns of the LAST event in the window
    (event index = starts[npz_idx] + WINDOW - 1)."""
    starts = recompute_valid_starts(date)
    if len(starts) == 0:
        return None
    f = MBO_EVENT_DIR / f"{date}_mbo_events.npz"
    d = np.load(f, allow_pickle=False)
    ts = d["timestamps"].astype(np.int64)
    event_idx = starts[npz_idx] + WINDOW - 1
    event_idx = np.minimum(event_idx, len(ts) - 1)
    return ts[event_idx]


def run_one_date(date: str, fold_id: int, pct: float) -> List[dict]:
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    p50 = load_p50_1s(fold_id)
    n = p50.size
    log.info(f"  {date}: fold_{fold_id:02d} P50_1s n={n}, "
             f"min={float(p50.min()):.3f}, max={float(p50.max()):.3f}, mean={float(p50.mean()):.3f}")
    log.info(f"  {date}: first 3 P50_1s = {p50[:3].tolist()}")

    top_idx, top_strength = select_topk_long(p50, pct)
    log.info(f"  {date}: selected {len(top_idx)} long signals "
             f"(threshold P50_1s>={float(top_strength.min()):.4f}, "
             f"max strength={float(top_strength.max()):.4f})")

    ts_ns = map_npz_idx_to_event_ts(date, top_idx)
    if ts_ns is None:
        return [{"date": date, "error": "missing_mbo_events"}]

    signals = [
        {"ts_ns": int(t), "direction": "long", "strength": float(top_strength[i])}
        for i, t in enumerate(ts_ns)
    ]
    if not signals:
        return []

    log.info(f"  {date}: first 3 signal ts_ns = "
             f"{[s['ts_ns'] for s in signals[:3]]}")

    cancel_ns = int(CANCEL_S * 1_000_000_000)
    hold_ns = int(HOLD_S * 1_000_000_000)

    try:
        engine = FIFOReplayEngine(
            date=date,
            cancel_after_ns=cancel_ns,
            max_hold_ns=hold_ns,
        )
    except FileNotFoundError as e:
        return [{"date": date, "error": f"no_dbn: {e}"}]
    except Exception as e:
        return [{"date": date, "error": f"engine_init: {e}"}]

    try:
        trades = engine.simulate(
            signals=signals,
            tp_ticks=TP_TICKS,
            sl_ticks=SL_TICKS,
            order_type="limit",
        )
    except Exception as e:
        return [{"date": date, "error": f"simulate: {e}"}]

    fills = []
    for t in trades:
        hold_s = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        fills.append({
            "date": date,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=0.01, help="top-pct LONG per day (default 0.01)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    log.info("=== HC #493 R3 — quantile-pinball LONG (+0.65 lift) FIFO grade ===")
    log.info(f"  side=long, pct={args.pct}, workers={args.workers}")
    log.info(f"  TP={TP_TICKS}t SL={SL_TICKS}t hold={HOLD_S}s cancel={CANCEL_S}s")
    log.info(f"  Dates: {[d for d,_ in DATE_FOLDS]}")
    log.info(f"  Pred dir: {PRED_DIR}")

    # Per-date FIFO replay
    all_fills_rows: List[dict] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(run_one_date, d, f, args.pct): d
            for d, f in DATE_FOLDS
        }
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                log.error(f"  {d}: worker crashed: {e}")
                continue
            all_fills_rows.extend(rows)
            log.info(f"  {d}: produced {len(rows)} fill rows")

    if not all_fills_rows:
        log.error("No fills produced. Check engine / data paths.")
        sys.exit(1)

    df = pd.DataFrame(all_fills_rows)
    if "error" in df.columns:
        err_df = df[df["error"].notna()]
        if len(err_df):
            log.warning(f"Errored: {err_df.to_dict(orient='records')}")
        df = df[df["error"].isna()].drop(columns=["error"])

    if df.empty:
        log.error("All dates errored — abort.")
        sys.exit(1)

    df.to_parquet(OUT_DIR / "fills.parquet")
    log.info(f"  Wrote fills.parquet: {len(df):,} rows")

    # Per-day breakdown
    daily = df.groupby("date").agg(
        n=("net_ticks", "size"),
        sum_ticks=("net_ticks", "sum"),
        mean_ticks=("net_ticks", "mean"),
        wr=("net_ticks", lambda s: (s > 0).mean()),
    ).reset_index()
    daily.to_csv(OUT_DIR / "per_day.csv", index=False)
    log.info("\nPer-day:\n" + daily.to_string(index=False))

    # Overall
    rows = []
    rows.append({"cohort": "overall_long", **metrics_for(df)})

    # Regime
    if REGIME_PARQUET.exists():
        reg = pd.read_parquet(REGIME_PARQUET)
        reg["date"] = reg["date"].astype(str).str.zfill(8)
        def classify(r):
            delta = r["close_minus_open_ticks"]
            if delta >= 4: return "green"
            if delta <= -4: return "red"
            return "flat"
        reg["regime"] = reg.apply(classify, axis=1)
        merged = df.merge(reg[["date", "regime"]], on="date", how="left")
        for regime in ["green", "red", "flat"]:
            m = metrics_for(merged[merged["regime"] == regime])
            rows.append({"cohort": f"regime_{regime}", **m})

    # Exit reason breakdown
    if "fill_type" in df.columns:
        for fr in df["fill_type"].unique():
            sub = df[df["fill_type"] == fr]
            rows.append({"cohort": f"exit_{fr}", **metrics_for(sub)})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    log.info("\nSummary:\n" + summary.to_string(index=False))

    # Markdown
    md = ["# HC #493 R3 — Quantile-pinball LONG FIFO Regrade", ""]
    md.append(f"- side: long, top-pct per day: {args.pct}")
    md.append(f"- horizon h=1s, TP={TP_TICKS}t SL={SL_TICKS}t hold<={HOLD_S}s cancel<={CANCEL_S}s")
    md.append(f"- Engine: `alpha_discovery.deep_models.fifo_market_replay.FIFOReplayEngine` (HC #74)")
    md.append(f"- Predictions: `output/hc488_dlinear_quantile_v1/fold_{{21,22}}_preds.npz`")
    md.append(f"- Dates: 20260427, 20260428 (the two days the +0.65 claim was made on)")
    md.append("")
    md.append("## Per-day net (FIFO)")
    md.append("")
    md.append(daily.to_markdown(index=False))
    md.append("")
    md.append("## Cohort summary")
    md.append("")
    md.append(summary.to_markdown(index=False))
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    log.info("Wrote summary.md")
    log.info("DONE.")


if __name__ == "__main__":
    main()
