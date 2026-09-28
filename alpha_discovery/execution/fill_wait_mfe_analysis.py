#!/usr/bin/env python3
"""
Fill-Wait-Time + MFE/MAE-by-Confidence Analysis
================================================

Per HC #247 (2026-05-08 10:15 ET):
  (A) Fill-wait-time distribution + correlation to win/loss
  (B) MFE/MAE distribution by confidence threshold for TP/SL derivation

Reads existing fill_sim_cli JSON outputs (no new sims needed). Each trade has:
  - signal_time_ns, post_time_ns, fill_time_ns, exit_time_ns
  - fill_latency_ns (= fill_time - post_time)
  - mae_ticks, mfe_ticks
  - signal_strength (CNN-Mamba |pred_1s|)
  - queue_position_at_post, book_size_at_post
  - pnl_ticks, exit_reason

Hypothesis (from HC #248): orders that wait > 1-2s to fill are losers (signal decays).

Usage:
  python3 fill_wait_mfe_analysis.py \\
    --input-dirs output/conditional_entry_v1 output/fifo_validate_v2 output/fifo_validate_v3 \\
    --output-dir output/fill_wait_mfe_v1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [FW_MFE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ES_TICK_VALUE = 12.50
ES_RT_COMM_TICKS = 0.376


def load_trades(json_paths: List[Path]) -> pd.DataFrame:
    """Load all trades from a list of fill_sim JSON outputs into one DataFrame."""
    rows = []
    n_files = 0
    n_skip = 0
    for p in json_paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as e:
            n_skip += 1
            continue
        # Handle both {trades:[...], config:...} and bare list-of-trades formats
        if isinstance(data, list):
            trades = data
            cfg = p.stem.rsplit("_", 1)[0]  # config = filename minus date
        elif isinstance(data, dict):
            trades = data.get("trades") or []
            cfg = data.get("config", p.stem.rsplit("_", 1)[0])
        else:
            n_skip += 1
            continue
        if not trades:
            n_skip += 1
            continue
        date = p.stem.split("_")[-1]
        for t in trades:
            t["config"] = cfg
            t["date"] = date
            t["source_file"] = p.name
            rows.append(t)
        n_files += 1
    log.info(f"Loaded {len(rows):,} trades from {n_files} files ({n_skip} skipped/empty)")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize dtypes
    for col in ("fill_latency_ns", "hold_duration_ns", "signal_time_ns",
                "post_time_ns", "fill_time_ns", "exit_time_ns"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fill_latency_s"] = df["fill_latency_ns"] / 1e9
    df["hold_duration_s"] = df["hold_duration_ns"] / 1e9
    df["abs_signal"] = df["signal_strength"].abs()
    df["is_win"] = df["pnl_ticks"] > 0
    df["pnl_net"] = df["pnl_ticks"] - ES_RT_COMM_TICKS  # net of commission
    return df


def fill_wait_distribution(df: pd.DataFrame, out_dir: Path):
    """Distribution of fill_latency and correlation to outcome."""
    log.info("\n" + "=" * 70)
    log.info("(A) FILL-WAIT-TIME ANALYSIS")
    log.info("=" * 70)

    fill = df[df["exit_reason"] != "cancelled"].copy()
    if "fill_latency_s" not in fill.columns or fill.empty:
        log.warning("No filled trades")
        return None

    log.info(f"Filled trades: {len(fill):,}")
    pcts = [10, 25, 50, 75, 90, 95, 99]
    qs = np.percentile(fill["fill_latency_s"], pcts)
    log.info("Fill-latency percentiles (seconds):")
    for p, q in zip(pcts, qs):
        log.info(f"  p{p:>2}: {q:8.3f}s")

    # Bucket by fill latency, measure outcome
    bins = [0, 0.05, 0.10, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0, 9999.0]
    labels = ["<50ms", "50-100ms", "100-250ms", "250-500ms",
              "500ms-1s", "1-2s", "2-5s", "5-15s", "15-60s", ">60s"]
    fill["wait_bin"] = pd.cut(fill["fill_latency_s"], bins=bins, labels=labels,
                              include_lowest=True, right=False)
    grp = fill.groupby("wait_bin", observed=True).agg(
        n=("pnl_ticks", "size"),
        wr=("is_win", "mean"),
        avg_pnl=("pnl_ticks", "mean"),
        avg_net=("pnl_net", "mean"),
        avg_mfe=("mfe_ticks", "mean"),
        avg_mae=("mae_ticks", "mean"),
    ).reset_index()

    log.info("\nFill-wait bucket vs outcome:")
    log.info(f"{'wait_bin':<11} {'n':>8} {'WR':>7} {'avg_pnl':>9} {'net_pnl':>9} {'avg_mfe':>9} {'avg_mae':>9}")
    log.info("-" * 70)
    for _, r in grp.iterrows():
        log.info(f"{str(r['wait_bin']):<11} {r['n']:>8.0f} {r['wr']:>6.1%} "
                 f"{r['avg_pnl']:>+9.3f} {r['avg_net']:>+9.3f} "
                 f"{r['avg_mfe']:>+9.3f} {r['avg_mae']:>+9.3f}")

    # Correlation
    corr_pnl = fill[["fill_latency_s", "pnl_ticks"]].corr().iloc[0, 1]
    corr_mfe = fill[["fill_latency_s", "mfe_ticks"]].corr().iloc[0, 1]
    log.info(f"\nPearson(fill_latency, pnl)  = {corr_pnl:+.4f}")
    log.info(f"Pearson(fill_latency, mfe)  = {corr_mfe:+.4f}")
    log.info(f"(Negative correlation supports HC #248: long waits = adverse selection.)")

    grp.to_csv(out_dir / "fill_wait_distribution.csv", index=False)
    return grp


def mfe_mae_by_confidence(df: pd.DataFrame, out_dir: Path):
    """MFE/MAE distribution by confidence bucket — used to derive optimal TP/SL."""
    log.info("\n" + "=" * 70)
    log.info("(B) MFE/MAE BY CONFIDENCE — TP/SL DERIVATION")
    log.info("=" * 70)

    fill = df[df["exit_reason"] != "cancelled"].copy()
    if fill.empty:
        log.warning("No fills")
        return None
    fill["abs_conf"] = fill["signal_strength"].abs()

    pcts = np.percentile(fill["abs_conf"], [50, 75, 90, 95, 99, 99.5])
    log.info(f"Confidence percentiles: p50={pcts[0]:.3f} p75={pcts[1]:.3f} "
             f"p90={pcts[2]:.3f} p95={pcts[3]:.3f} p99={pcts[4]:.3f} p99.5={pcts[5]:.3f}")

    # Bucket by confidence percentile
    fill["conf_pct"] = fill["abs_conf"].rank(pct=True) * 100
    buckets = [
        ("0-50%",   0,    50),
        ("50-75%",  50,   75),
        ("75-90%",  75,   90),
        ("90-95%",  90,   95),
        ("95-99%",  95,   99),
        ("99-99.5%", 99,  99.5),
        ("99.5-100%", 99.5, 100.001),
    ]

    rows = []
    log.info(f"\n{'bucket':<11} {'n':>7} {'WR':>6} {'avg_pnl':>9} {'mfe_p25':>9} {'mfe_med':>9} "
             f"{'mfe_p75':>9} {'mfe_p90':>9} {'mae_p10':>9} {'mae_med':>9} {'mae_p90':>9}")
    log.info("-" * 110)
    for name, lo, hi in buckets:
        sel = fill[(fill["conf_pct"] >= lo) & (fill["conf_pct"] < hi)]
        if len(sel) < 50:
            continue
        mfe_p = np.percentile(sel["mfe_ticks"], [25, 50, 75, 90])
        mae_p = np.percentile(sel["mae_ticks"], [10, 50, 90])
        wr = (sel["pnl_ticks"] > 0).mean()
        log.info(f"{name:<11} {len(sel):>7} {wr:>5.1%} {sel['pnl_ticks'].mean():>+9.3f} "
                 f"{mfe_p[0]:>+9.3f} {mfe_p[1]:>+9.3f} {mfe_p[2]:>+9.3f} {mfe_p[3]:>+9.3f} "
                 f"{mae_p[0]:>+9.3f} {mae_p[1]:>+9.3f} {mae_p[2]:>+9.3f}")
        rows.append({
            "bucket": name, "n": len(sel), "wr": wr,
            "avg_pnl": sel["pnl_ticks"].mean(),
            "mfe_p25": mfe_p[0], "mfe_p50": mfe_p[1], "mfe_p75": mfe_p[2], "mfe_p90": mfe_p[3],
            "mae_p10": mae_p[0], "mae_p50": mae_p[1], "mae_p90": mae_p[2],
        })

    # Optimal TP/SL: rule of thumb — TP at mfe_p75, SL at mae_p75 (asymmetric to favor high-conf)
    log.info("\nSuggested TP/SL per bucket (TP = MFE p75, SL = MAE p75):")
    for r in rows:
        log.info(f"  {r['bucket']:<11}: TP={r['mfe_p75']:.2f}  SL={r['mae_p90']:.2f}  "
                 f"(R:R = {r['mfe_p75']/max(abs(r['mae_p90']),0.01):.2f})")

    pd.DataFrame(rows).to_csv(out_dir / "mfe_mae_by_confidence.csv", index=False)
    return rows


def queue_position_x_confidence(df: pd.DataFrame, out_dir: Path):
    """Bivariate: queue position × confidence → outcome (per HC #247.D)."""
    log.info("\n" + "=" * 70)
    log.info("(C) QUEUE POSITION × CONFIDENCE → OUTCOME")
    log.info("=" * 70)

    fill = df[df["exit_reason"] != "cancelled"].copy()
    if fill.empty or "queue_position_at_post" not in fill.columns:
        log.warning("No queue data")
        return None

    fill["q_rank"] = fill["queue_position_at_post"] / fill["book_size_at_post"].clip(lower=1)
    fill["abs_conf"] = fill["signal_strength"].abs()
    fill["conf_pct"] = fill["abs_conf"].rank(pct=True) * 100

    log.info(f"{'conf_pct':<10} {'qrank':<10} {'n':>6} {'WR':>6} {'pnl':>8} {'fill_lat_s':>11}")
    log.info("-" * 60)
    rows = []
    for cname, clo, chi in [("0-90", 0, 90), ("90-99", 90, 99), ("99-100", 99, 100.001)]:
        for qname, qlo, qhi in [("front", 0, 0.10), ("mid", 0.10, 0.50), ("back", 0.50, 1.01)]:
            sel = fill[(fill["conf_pct"] >= clo) & (fill["conf_pct"] < chi)
                       & (fill["q_rank"] >= qlo) & (fill["q_rank"] < qhi)]
            if len(sel) < 20:
                continue
            wr = (sel["pnl_ticks"] > 0).mean()
            log.info(f"{cname:<10} {qname:<10} {len(sel):>6} {wr:>5.1%} "
                     f"{sel['pnl_ticks'].mean():>+8.3f} {sel['fill_latency_s'].mean():>11.3f}")
            rows.append({"conf": cname, "qrank": qname, "n": len(sel), "wr": wr,
                         "avg_pnl": sel["pnl_ticks"].mean(),
                         "avg_fill_lat_s": sel["fill_latency_s"].mean()})

    pd.DataFrame(rows).to_csv(out_dir / "queue_x_confidence.csv", index=False)
    return rows


def aggressive_cancel_simulation(df: pd.DataFrame, out_dir: Path):
    """Simulate HC #248: cancel orders with fill_latency > 1.0s, > 2.0s.
    For trades that DID fill, ask: what % of P&L came from quick fills (< X)?
    """
    log.info("\n" + "=" * 70)
    log.info("(D) AGGRESSIVE-CANCEL SIMULATION (HC #248)")
    log.info("=" * 70)

    fill = df[df["exit_reason"] != "cancelled"].copy()
    if fill.empty:
        return None

    log.info(f"{'cancel_thresh':<15} {'kept_n':>10} {'kept_pct':>10} {'avg_pnl':>9} "
             f"{'WR':>6} {'total_pnl_t':>12}")
    log.info("-" * 70)
    rows = []
    for thresh in [0.5, 1.0, 2.0, 5.0, 15.0, 9999.0]:
        kept = fill[fill["fill_latency_s"] <= thresh]
        if kept.empty:
            continue
        avg_pnl = kept["pnl_ticks"].mean()
        wr = (kept["pnl_ticks"] > 0).mean()
        total = kept["pnl_ticks"].sum()
        thresh_label = f"<= {thresh}s" if thresh < 9999 else "no cancel"
        log.info(f"{thresh_label:<15} {len(kept):>10} {len(kept)/len(fill):>9.1%} "
                 f"{avg_pnl:>+9.3f} {wr:>5.1%} {total:>+12.1f}")
        rows.append({"cancel_thresh_s": thresh, "kept_n": len(kept),
                     "kept_pct": len(kept)/len(fill), "avg_pnl": avg_pnl, "wr": wr,
                     "total_pnl_ticks": total})

    pd.DataFrame(rows).to_csv(out_dir / "aggressive_cancel_sim.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Directories with fill_sim JSON outputs")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of input files (for testing)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = []
    for d in args.input_dirs:
        json_paths.extend(sorted(Path(d).glob("*.json")))
    if args.max_files:
        json_paths = json_paths[: args.max_files]
    log.info(f"Found {len(json_paths)} JSON files across {len(args.input_dirs)} dirs")

    df = load_trades(json_paths)
    if df.empty:
        log.error("No trades loaded — aborting")
        return

    # Coerce mixed-type columns to JSON strings before saving
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(50)
            if any(isinstance(x, (dict, list)) for x in sample):
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
    try:
        df.to_parquet(out_dir / "all_trades.parquet", index=False)
        log.info(f"Saved {len(df):,} trades to {out_dir/'all_trades.parquet'}")
    except Exception as e:
        log.warning(f"Parquet save failed ({e}); falling back to CSV")
        df.to_csv(out_dir / "all_trades.csv", index=False)

    fill_wait_distribution(df, out_dir)
    mfe_mae_by_confidence(df, out_dir)
    queue_position_x_confidence(df, out_dir)
    aggressive_cancel_simulation(df, out_dir)

    log.info("\nAnalysis complete. Results in: " + str(out_dir))


if __name__ == "__main__":
    main()
