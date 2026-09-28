#!/usr/bin/env python3
"""
Optimal Filter Sweep — Combinatorial Search Over Existing Trades
=================================================================

Per HC #248 + #247.D: We have 42K simulated trades from the conditional sweep.
Each trade has: signal_strength, queue_position_at_post, book_size_at_post,
fill_latency_ns, mae_ticks, mfe_ticks, pnl_ticks.

Instead of re-running the rust simulator (which can't take new params), we
post-process: for each filter combination, ask "if we ONLY kept trades
matching this filter, what's the WR / PF / Sortino / total P&L?"

Filters tested:
  - confidence_min: |pred_1s| >= X
  - queue_max: queue_position_at_post <= X (front-of-queue only)
  - fill_latency_max_s: cancel orders that take > X seconds (HC #248)
  - direction: long-only / short-only / both

Goal: find filter combos that flip total P&L positive, identify the
high-conf + front-of-queue + fast-fill regime where edge exists.

Usage:
  python3 optimal_filter_sweep.py \\
    --trades-parquet output/fill_wait_mfe_v1/all_trades.parquet \\
    --output-dir output/optimal_filter_sweep_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [FILT] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ES_TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376


def evaluate_filter(df: pd.DataFrame, conf_min: float, queue_max: int,
                    fill_max_s: float, side: str = "both") -> dict:
    """Apply filter combination, return aggregate stats."""
    sel = df[df["exit_reason"] != "cancelled"].copy()
    if sel.empty:
        return {"n": 0}

    sel["abs_conf"] = sel["signal_strength"].abs()
    mask = (sel["abs_conf"] >= conf_min) & \
           (sel["queue_position_at_post"] <= queue_max) & \
           (sel["fill_latency_ns"] / 1e9 <= fill_max_s)
    if side == "long":
        mask &= sel["side"] == "buy"
    elif side == "short":
        mask &= sel["side"] == "sell"
    sub = sel[mask]
    if len(sub) < 30:
        return {"n": len(sub)}

    pnl_net = sub["pnl_ticks"] - COMMISSION_TICKS
    wr = (sub["pnl_ticks"] > 0).mean()
    avg_net = pnl_net.mean()
    total_pnl_net = pnl_net.sum()
    wins = sub[sub["pnl_ticks"] > 0]["pnl_ticks"].sum()
    losses = -sub[sub["pnl_ticks"] <= 0]["pnl_ticks"].sum()
    pf = (wins / losses) if losses > 0 else float("inf")
    if len(sub) > 1:
        sortino = pnl_net.mean() / max(pnl_net[pnl_net < 0].std(), 1e-9)
    else:
        sortino = 0.0
    # Per-day breakdown for greenness
    by_day = sub.groupby("date").apply(lambda g: (g["pnl_ticks"] - COMMISSION_TICKS).sum())
    n_days = len(by_day)
    n_green = (by_day > 0).sum()
    return {
        "n": len(sub), "wr": wr, "avg_net_ticks": avg_net,
        "total_net_ticks": total_pnl_net,
        "total_dollars": total_pnl_net * ES_TICK_VALUE,
        "pf": pf, "sortino": sortino,
        "n_days": n_days, "n_green": n_green,
        "green_pct": n_green / max(n_days, 1),
        "trades_per_day": len(sub) / max(n_days, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.trades_parquet)
    log.info(f"Loaded {len(df):,} trades")
    log.info(f"Sides: {df['side'].value_counts().to_dict()}")
    log.info(f"Exit reasons: {df['exit_reason'].value_counts().to_dict()}")

    # Confidence percentile thresholds (use abs |pred_1s|)
    abs_conf = df[df["exit_reason"] != "cancelled"]["signal_strength"].abs()
    conf_thresholds = {
        "p0":   0.0,
        "p50":  abs_conf.quantile(0.50),
        "p75":  abs_conf.quantile(0.75),
        "p90":  abs_conf.quantile(0.90),
        "p95":  abs_conf.quantile(0.95),
        "p99":  abs_conf.quantile(0.99),
    }
    log.info(f"Confidence thresholds: {conf_thresholds}")

    queue_thresholds = [10, 25, 50, 100, 250, 1_000_000]  # last = no filter
    fill_max_seconds = [0.25, 0.5, 1.0, 2.0, 5.0, 999.0]
    sides = ["both", "long", "short"]

    results = []
    log.info("Sweeping filter combinations...")
    for cname, cmin in conf_thresholds.items():
        for q in queue_thresholds:
            for f in fill_max_seconds:
                for s in sides:
                    r = evaluate_filter(df, cmin, q, f, s)
                    if r.get("n", 0) < 30:
                        continue
                    r.update({"conf_pct": cname, "conf_min": cmin,
                              "queue_max": q, "fill_max_s": f, "side": s})
                    results.append(r)

    res = pd.DataFrame(results)
    log.info(f"Evaluated {len(res):,} filter combinations with n>=30")

    res.to_csv(out_dir / "all_filter_results.csv", index=False)

    # Top by total_net_ticks (must have meaningful trade count)
    log.info("\n" + "=" * 110)
    log.info("TOP 30 BY TOTAL NET TICKS (n >= 100, n_days >= 5):")
    log.info("=" * 110)
    qual = res[(res["n"] >= 100) & (res["n_days"] >= 5)].copy()
    top = qual.sort_values("total_net_ticks", ascending=False).head(args.top_n)
    log.info(f"{'conf':>5} {'qmax':>8} {'fmax_s':>7} {'side':>6} {'n':>7} {'tpd':>6} "
             f"{'WR':>6} {'avg_net':>9} {'total_net':>10} {'$total':>10} {'PF':>5} "
             f"{'green%':>7} {'sortino':>8}")
    log.info("-" * 110)
    for _, r in top.iterrows():
        log.info(f"{r['conf_pct']:>5} {r['queue_max']:>8.0f} {r['fill_max_s']:>7.2f} "
                 f"{r['side']:>6} {r['n']:>7.0f} {r['trades_per_day']:>6.1f} "
                 f"{r['wr']:>5.1%} {r['avg_net_ticks']:>+9.3f} "
                 f"{r['total_net_ticks']:>+10.1f} ${r['total_dollars']:>+9.0f} "
                 f"{r['pf']:>5.2f} {r['green_pct']:>6.1%} {r['sortino']:>+8.3f}")

    # Top by Sortino with n >= 200
    log.info("\n" + "=" * 110)
    log.info("TOP 20 BY SORTINO (n >= 200, total_net > 0):")
    log.info("=" * 110)
    qual2 = res[(res["n"] >= 200) & (res["total_net_ticks"] > 0)].copy()
    top2 = qual2.sort_values("sortino", ascending=False).head(20)
    if not top2.empty:
        log.info(f"{'conf':>5} {'qmax':>8} {'fmax_s':>7} {'side':>6} {'n':>7} {'tpd':>6} "
                 f"{'WR':>6} {'avg_net':>9} {'$total':>10} {'sortino':>8}")
        log.info("-" * 110)
        for _, r in top2.iterrows():
            log.info(f"{r['conf_pct']:>5} {r['queue_max']:>8.0f} {r['fill_max_s']:>7.2f} "
                     f"{r['side']:>6} {r['n']:>7.0f} {r['trades_per_day']:>6.1f} "
                     f"{r['wr']:>5.1%} {r['avg_net_ticks']:>+9.3f} "
                     f"${r['total_dollars']:>+9.0f} {r['sortino']:>+8.3f}")
    else:
        log.warning("NO profitable filter combinations found with n>=200 and total>0")

    # Summary: how many combos hit positive total
    pos = res[res["total_net_ticks"] > 0]
    log.info(f"\nSummary: {len(pos):,} of {len(res):,} ({len(pos)/max(len(res),1):.1%}) "
             f"filter combos have positive total P&L")
    log.info(f"Save: {out_dir/'all_filter_results.csv'}")


if __name__ == "__main__":
    main()
