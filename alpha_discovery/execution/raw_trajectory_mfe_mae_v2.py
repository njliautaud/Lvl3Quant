#!/usr/bin/env python3
"""Raw-Trajectory MFE/MAE Analyzer v2 — adds per-trade price-range filter
to exclude non-ES instruments contaminating the MBO file."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [RAW_TRAJ2] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
ES_TICK = 0.25
PX_TOLERANCE = 50.0  # filter prints to ±50 points of trade entry (ES is ~7400 ± 200)


def load_trade_prints(dbn_path: Path):
    import databento as db
    log.info(f"Reading {dbn_path.name} ...")
    store = db.DBNStore.from_file(str(dbn_path))
    df = store.to_df()
    if "action" in df.columns:
        df = df[df["action"] == "T"]
    if df.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32), None
    ts_col = "ts_event" if "ts_event" in df.columns else "ts_recv"
    ts_ns = df[ts_col].astype("int64").to_numpy()
    px = df["price"].astype("float64").to_numpy()
    if px.max() > 1e6:
        px = px / 1e9
    sym = df["symbol"].astype(str).to_numpy() if "symbol" in df.columns else None
    o = np.argsort(ts_ns)
    log.info(f"  Trade prints: {len(ts_ns):,} | px range {px.min():.2f}-{px.max():.2f}")
    return ts_ns[o], px[o].astype(np.float32), (sym[o] if sym is not None else None)


def compute_filtered(ts_arr, px_arr, sym_arr, signal_ts, side, entry_price, horizon_ns):
    if len(ts_arr) == 0:
        return np.nan, np.nan, 0
    lo = np.searchsorted(ts_arr, signal_ts, side="left")
    hi = np.searchsorted(ts_arr, signal_ts + horizon_ns, side="right")
    if hi <= lo:
        return np.nan, np.nan, 0
    pxw = px_arr[lo:hi]
    mask = np.abs(pxw - entry_price) <= PX_TOLERANCE
    pxw = pxw[mask]
    if len(pxw) == 0:
        return np.nan, np.nan, 0
    if side.upper() in ("BUY", "B", "LONG", "L"):
        mfe = (pxw.max() - entry_price) / ES_TICK
        mae = (entry_price - pxw.min()) / ES_TICK
    else:
        mfe = (entry_price - pxw.min()) / ES_TICK
        mae = (pxw.max() - entry_price) / ES_TICK
    return float(mfe), float(mae), len(pxw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades-parquet", required=True)
    ap.add_argument("--raw-mbo-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--horizon-sec", type=int, default=30)
    ap.add_argument("--max-dates", type=int, default=5)
    ap.add_argument("--sample-per-date", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_mbo_dir)
    df = pd.read_parquet(args.trades_parquet)
    df = df[df["exit_reason"] != "cancelled"].copy()
    df["date_str"] = df["date"].astype(str)

    horizon_ns = args.horizon_sec * 1_000_000_000
    rows = []
    dates = sorted(df["date_str"].unique())
    if args.max_dates > 0:
        dates = dates[: args.max_dates]
    log.info(f"Processing {len(dates)} dates")

    for d in dates:
        fn = raw_dir / f"glbx-mdp3-{d}.mbo.dbn.zst"
        if not fn.exists():
            log.warning(f"No file for {d}")
            continue
        try:
            ts_arr, px_arr, _ = load_trade_prints(fn)
        except Exception as e:
            log.warning(f"Load error {d}: {e}")
            continue
        sub = df[df["date_str"] == d]
        if args.sample_per_date and len(sub) > args.sample_per_date:
            sub = sub.sample(n=args.sample_per_date, random_state=42)
        log.info(f"{d}: {len(sub):,} trades")
        for _, t in sub.iterrows():
            mfe, mae, n_obs = compute_filtered(
                ts_arr, px_arr, None,
                int(t["signal_time_ns"]), str(t["side"]),
                float(t["entry_price"]), horizon_ns)
            rows.append({
                "date": d,
                "side": t["side"],
                "abs_signal": abs(float(t["signal_strength"])),
                "queue_position_at_post": float(t["queue_position_at_post"]),
                "fill_latency_s": float(t["fill_latency_s"]),
                "raw_mfe_ticks": mfe,
                "raw_mae_ticks": mae,
                "n_obs": n_obs,
                "fillsim_mfe": t["mfe_ticks"],
                "fillsim_mae": t["mae_ticks"],
                "fillsim_pnl": t["pnl_ticks"],
            })

    edf = pd.DataFrame(rows).dropna(subset=["raw_mfe_ticks"])
    log.info(f"Enriched {len(edf):,} trades (post-filter)")
    edf.to_parquet(out / "per_trade_raw_mfe_mae_v2.parquet", index=False)

    qs = [0.0, 0.50, 0.75, 0.90, 0.95, 0.99]
    th = {f"p{int(q*100)}": edf["abs_signal"].quantile(q) for q in qs}
    log.info(f"Conf thresholds: { {k: round(v,4) for k,v in th.items()} }")

    out_rows = []
    for lo_k, hi_k in [("p0","p50"),("p50","p75"),("p75","p90"),
                        ("p90","p95"),("p95","p99"),("p99",None)]:
        lo = th[lo_k]; hi = th[hi_k] if hi_k else float("inf")
        s = edf[(edf["abs_signal"] >= lo) & (edf["abs_signal"] < hi)]
        if len(s) < 10: continue
        # also break by side
        for side_label, side_filter in [("both", s),
                                         ("long", s[s["side"].str.upper().isin(["BUY","B","LONG"])]),
                                         ("short", s[s["side"].str.upper().isin(["SELL","S","SHORT"])])]:
            if len(side_filter) < 10: continue
            out_rows.append({
                "tier": f"{lo_k}-{hi_k or 'p100'}",
                "side": side_label,
                "n": len(side_filter),
                "mfe_med": side_filter["raw_mfe_ticks"].median(),
                "mfe_p75": side_filter["raw_mfe_ticks"].quantile(0.75),
                "mfe_p90": side_filter["raw_mfe_ticks"].quantile(0.90),
                "mae_med": side_filter["raw_mae_ticks"].median(),
                "mae_p75": side_filter["raw_mae_ticks"].quantile(0.75),
                "mae_p90": side_filter["raw_mae_ticks"].quantile(0.90),
                # Probability MFE >= K and MAE < K (one-sided breakout)
                "p_mfe_ge_2": (side_filter["raw_mfe_ticks"] >= 2).mean(),
                "p_mfe_ge_4": (side_filter["raw_mfe_ticks"] >= 4).mean(),
                "p_mfe_ge_6": (side_filter["raw_mfe_ticks"] >= 6).mean(),
                "p_mae_le_2": (side_filter["raw_mae_ticks"] <= 2).mean(),
                "p_mae_le_4": (side_filter["raw_mae_ticks"] <= 4).mean(),
                "wr_at_be":  (side_filter["raw_mfe_ticks"] > side_filter["raw_mae_ticks"]).mean(),
            })
    by_tier = pd.DataFrame(out_rows)
    by_tier.to_csv(out / "mfe_mae_by_tier_v2.csv", index=False)
    log.info("\n" + "=" * 110)
    log.info("RAW UNCLAMPED MFE/MAE BY (CONF TIER × SIDE):")
    log.info("=" * 110)
    log.info(by_tier.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
