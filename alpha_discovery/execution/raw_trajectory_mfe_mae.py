#!/usr/bin/env python3
"""
Raw-Trajectory MFE/MAE Analyzer (Unclamped)
============================================
Per HC #248 follow-up: derive TRUE optimal TP/SL distributions per confidence tier
using actual ES futures mid-price trajectories, not the rust fill_sim outputs which
clamp MFE/MAE to TP/SL bounds.

For each trade in our cached parquet (entry_price, signal_time_ns, side, signal_strength):
  1. Locate the raw MBO file for that date
  2. Build a mid-price time series from ASK/BID levels
  3. Look forward up to N seconds from signal_time
  4. Compute raw MFE = max favorable excursion (ticks, signed by side)
  5. Compute raw MAE = max adverse excursion (ticks)
  6. Aggregate per confidence-tier bucket

Outputs:
  - per_trade_raw_mfe_mae.parquet  — every trade enriched with raw MFE/MAE
  - mfe_mae_by_tier.csv            — bucket distributions, percentiles
  - tp_sl_recommendations.csv      — TP=p75(MFE), SL=p75(MAE) per tier

Usage:
  python3 raw_trajectory_mfe_mae.py \\
    --trades-parquet output/fill_wait_mfe_v1/all_trades.parquet \\
    --raw-mbo-dir data/raw/mbo \\
    --output-dir output/raw_trajectory_v1 \\
    --horizon-sec 30 \\
    --max-dates 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [RAW_TRAJ] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ES_TICK = 0.25  # ES futures tick size in points


def load_mid_price_series(dbn_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read MBO file, return (ts_ns, mid_price) arrays sorted by timestamp.
    Builds top-of-book bid/ask via order book reconstruction."""
    import databento as db
    log.info(f"Reading {dbn_path.name} ...")
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df()  # this can be HEAVY; consider chunked iter for prod

    # MBO records have side (B/A), price, size, action (Add/Cancel/Modify/Trade/Fill)
    # We approximate top of book: track best bid/ask via running min/max with cancellations.
    # Simpler proxy: use Trade prints (action == 'T') as last-price; mid = last trade price.
    # This is a coarse approximation but sufficient for MFE/MAE direction analysis.
    if "action" in df.columns:
        trades = df[df["action"] == "T"].copy()
    else:
        trades = df.copy()
    if trades.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    ts_col = "ts_event" if "ts_event" in trades.columns else "ts_recv"
    ts_ns = trades[ts_col].astype("int64").to_numpy()
    px = trades["price"].astype("float64").to_numpy()
    # databento prices are in fixed-point 1e-9 if stored raw; if .to_df() returns float points,
    # values look like e.g. 5230.25.  We sanity check magnitude.
    if px.max() > 1e6:
        px = px / 1e9
    log.info(f"  Trade prints: {len(ts_ns):,} | px range {px.min():.2f}-{px.max():.2f}")
    order = np.argsort(ts_ns)
    return ts_ns[order], px[order].astype(np.float32)


def compute_raw_mfe_mae(ts_arr: np.ndarray, px_arr: np.ndarray,
                        signal_ts: int, side: str, entry_price: float,
                        horizon_ns: int) -> tuple[float, float, int]:
    """Return (mfe_ticks_signed_for_side, mae_ticks_signed_for_side, n_obs)."""
    if len(ts_arr) == 0:
        return np.nan, np.nan, 0
    lo = np.searchsorted(ts_arr, signal_ts, side="left")
    hi = np.searchsorted(ts_arr, signal_ts + horizon_ns, side="right")
    if hi <= lo:
        return np.nan, np.nan, 0
    px_window = px_arr[lo:hi]
    if side.upper() in ("BUY", "B", "LONG", "L"):
        mfe = (px_window.max() - entry_price) / ES_TICK
        mae = (entry_price - px_window.min()) / ES_TICK
    else:  # SELL / SHORT
        mfe = (entry_price - px_window.min()) / ES_TICK
        mae = (px_window.max() - entry_price) / ES_TICK
    return float(mfe), float(mae), int(hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-parquet", required=True)
    ap.add_argument("--raw-mbo-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--horizon-sec", type=int, default=30)
    ap.add_argument("--max-dates", type=int, default=0,
                    help="0 = all available; otherwise process only first N dates")
    ap.add_argument("--sample-per-date", type=int, default=0,
                    help="0 = all trades; otherwise sample N trades per date")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_mbo_dir)

    df = pd.read_parquet(args.trades_parquet)
    df = df[df["exit_reason"] != "cancelled"].copy()
    df["date_str"] = df["date"].astype(str)
    log.info(f"Loaded {len(df):,} trades; {df['date_str'].nunique()} unique dates")

    horizon_ns = args.horizon_sec * 1_000_000_000
    enriched = []

    dates = sorted(df["date_str"].unique())
    if args.max_dates > 0:
        dates = dates[: args.max_dates]
    log.info(f"Processing {len(dates)} dates: {dates[:5]}...")

    for d in dates:
        # filename pattern: glbx-mdp3-YYYYMMDD.mbo.dbn.zst (date is YYYYMMDD)
        fn = raw_dir / f"glbx-mdp3-{d}.mbo.dbn.zst"
        if not fn.exists():
            log.warning(f"No raw file for {d}, skipping")
            continue
        try:
            ts_arr, px_arr = load_mid_price_series(fn)
        except Exception as e:
            log.warning(f"Failed to load {fn}: {e}")
            continue
        if len(ts_arr) == 0:
            continue

        sub = df[df["date_str"] == d]
        if args.sample_per_date > 0 and len(sub) > args.sample_per_date:
            sub = sub.sample(n=args.sample_per_date, random_state=42)
        log.info(f"{d}: computing MFE/MAE on {len(sub):,} trades...")

        for _, t in sub.iterrows():
            mfe, mae, n_obs = compute_raw_mfe_mae(
                ts_arr, px_arr,
                int(t["signal_time_ns"]),
                str(t["side"]),
                float(t["entry_price"]),
                horizon_ns,
            )
            enriched.append({
                "date": d,
                "order_id": t["order_id"],
                "side": t["side"],
                "signal_strength": t["signal_strength"],
                "abs_signal": abs(float(t["signal_strength"])),
                "queue_position_at_post": t["queue_position_at_post"],
                "fill_latency_s": t["fill_latency_s"],
                "raw_mfe_ticks": mfe,
                "raw_mae_ticks": mae,
                "n_obs": n_obs,
                # original clamped fill-sim outputs for comparison:
                "fillsim_mfe_ticks": t["mfe_ticks"],
                "fillsim_mae_ticks": t["mae_ticks"],
                "fillsim_pnl_ticks": t["pnl_ticks"],
            })

    if not enriched:
        log.error("No enriched trades produced — abort")
        return
    edf = pd.DataFrame(enriched).dropna(subset=["raw_mfe_ticks"])
    log.info(f"\nEnriched {len(edf):,} trades")

    edf.to_parquet(out / "per_trade_raw_mfe_mae.parquet", index=False)

    # Confidence-tier bucketing
    qs = [0.0, 0.50, 0.75, 0.90, 0.95, 0.99]
    thresh = {f"p{int(q*100)}": edf["abs_signal"].quantile(q) for q in qs}
    log.info(f"Confidence thresholds: {thresh}")

    rows = []
    for tier_lo, tier_hi in [("p0", "p50"), ("p50", "p75"), ("p75", "p90"),
                             ("p90", "p95"), ("p95", "p99"), ("p99", None)]:
        lo_v = thresh[tier_lo]
        hi_v = thresh[tier_hi] if tier_hi is not None else float("inf")
        sub = edf[(edf["abs_signal"] >= lo_v) & (edf["abs_signal"] < hi_v)]
        if len(sub) < 10:
            continue
        rows.append({
            "tier": f"{tier_lo}-{tier_hi or 'p100'}",
            "n": len(sub),
            "mfe_p25": sub["raw_mfe_ticks"].quantile(0.25),
            "mfe_med": sub["raw_mfe_ticks"].median(),
            "mfe_p75": sub["raw_mfe_ticks"].quantile(0.75),
            "mfe_p90": sub["raw_mfe_ticks"].quantile(0.90),
            "mae_p25": sub["raw_mae_ticks"].quantile(0.25),
            "mae_med": sub["raw_mae_ticks"].median(),
            "mae_p75": sub["raw_mae_ticks"].quantile(0.75),
            "mae_p90": sub["raw_mae_ticks"].quantile(0.90),
            "wr_at_breakeven": (sub["raw_mfe_ticks"] > sub["raw_mae_ticks"]).mean(),
        })
    by_tier = pd.DataFrame(rows)
    by_tier.to_csv(out / "mfe_mae_by_tier.csv", index=False)

    log.info("\n" + "=" * 100)
    log.info("RAW (UNCLAMPED) MFE/MAE BY CONFIDENCE TIER:")
    log.info("=" * 100)
    log.info(by_tier.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    # TP/SL recommendation: TP = MFE p75, SL = MAE p75 (cap losses where most reach)
    rec = by_tier[["tier", "n", "mfe_p75", "mae_p75"]].copy()
    rec["recommended_TP"] = rec["mfe_p75"].clip(lower=1.0)
    rec["recommended_SL"] = rec["mae_p75"].clip(lower=1.0)
    rec["RR_ratio"] = rec["recommended_TP"] / rec["recommended_SL"]
    rec.to_csv(out / "tp_sl_recommendations.csv", index=False)
    log.info("\nTP/SL RECOMMENDATIONS (TP=MFE p75, SL=MAE p75):")
    log.info(rec.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    log.info(f"\nSaved: {out}/per_trade_raw_mfe_mae.parquet")
    log.info(f"Saved: {out}/mfe_mae_by_tier.csv")
    log.info(f"Saved: {out}/tp_sl_recommendations.csv")


if __name__ == "__main__":
    main()
