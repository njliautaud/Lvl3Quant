#!/usr/bin/env python3
"""HC #451 R2 — Salience-tag sanity check.

Given a precomputed salience parquet and the raw MBO file:
  - Take all sweep_tag=True events on front-month trade rows.
  - For each, compute the 5-second forward displacement (signed in direction of
    the sweep's aggressor side: positive = price moved WITH the aggressor).
  - Compare distribution vs background (random sample of trades on same day).

Background = same number of randomly sampled NON-sweep trades, with the same
forward-window definition. We want to see whether sweep events systematically
precede larger directional moves than background trades.

Reports:
  - n_sweep, n_bg
  - mean / median / std forward 5s displacement, in ticks, for sweep vs bg
  - mean abs displacement (volatility proxy) sweep vs bg
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import databento as db


TICK_RAW = 250_000_000  # 0.25 pt
A_TRADE = ord('T')
SIDE_A = ord('A')  # trade hit ASK order → aggressor was BUYER (price up)
SIDE_B = ord('B')  # trade hit BID order → aggressor was SELLER (price down)

RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
FORWARD_NS = 5_000_000_000  # 5 seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--salience-dir", required=True)
    ap.add_argument("--n-bg-sample", type=int, default=2000,
                    help="size of background sample of non-sweep trades")
    ap.add_argument("--n-sweep-sample", type=int, default=2000,
                    help="size of sweep sample (caps to keep runtime ~seconds)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    salience = pl.read_parquet(
        Path(args.salience_dir) / f"{args.date}_salience.parquet"
    )
    sweep_flags = salience["sweep_tag"].to_numpy()
    large_flags = salience["large_print_tag"].to_numpy()

    dbn = RAW_DIR / f"glbx-mdp3-{args.date}.mbo.dbn.zst"
    store = db.DBNStore.from_file(str(dbn))
    arr = store.to_ndarray()
    actions = arr['action'].view(np.uint8)
    sides = arr['side'].view(np.uint8)
    iids = arr['instrument_id']
    ts_event = arr['ts_event'].astype(np.int64)
    prices = arr['price'].astype(np.int64)

    # Identify front-month trade indices, same logic as precompute
    t_mask_all = actions == A_TRADE
    es_mask = (prices > 5_000_000_000_000) & (prices < 8_000_000_000_000)
    cand = t_mask_all & es_mask
    iid_cand = iids[cand]
    uniq, cnt = np.unique(iid_cand, return_counts=True)
    front_id = int(uniq[np.argmax(cnt)])

    front_trade_mask = t_mask_all & (iids == front_id)
    tr_event_idx = np.flatnonzero(front_trade_mask)
    tr_ts = ts_event[tr_event_idx]
    tr_px = prices[tr_event_idx]
    tr_side = sides[tr_event_idx]
    tr_sweep = sweep_flags[tr_event_idx]
    tr_large = large_flags[tr_event_idx]

    # Build a trade-price timeseries (used to read forward price after 5s as last
    # trade price <= t+5s, matching MFE/MAE convention).
    print(f"front_month_id={front_id} n_trades_front={len(tr_ts)}")

    sweep_trade_pos = np.flatnonzero(tr_sweep)
    nonsweep_trade_pos = np.flatnonzero(~tr_sweep)
    print(f"sweep trade events: {len(sweep_trade_pos)}, "
          f"non-sweep: {len(nonsweep_trade_pos)}")
    print(f"large_print trade events: {int(tr_large.sum())}")

    rng = np.random.default_rng(args.seed)
    sw_pick = rng.choice(sweep_trade_pos,
                         size=min(args.n_sweep_sample, len(sweep_trade_pos)),
                         replace=False)
    bg_pick = rng.choice(nonsweep_trade_pos,
                         size=min(args.n_bg_sample, len(nonsweep_trade_pos)),
                         replace=False)

    def forward_signed_disp(picks: np.ndarray) -> dict:
        """For each pick (index into tr_ts arrays), find tr price at ts+5s
        (last trade with ts <= sweep_ts+5s) and return signed displacement
        in direction of aggressor (side A=buy=+, side B=sell=-)."""
        disp_signed_tk = np.empty(len(picks), dtype=np.float64)
        disp_abs_tk = np.empty(len(picks), dtype=np.float64)
        for k, i in enumerate(picks):
            t_now = tr_ts[i]
            p_now = tr_px[i]
            t_fwd = t_now + FORWARD_NS
            # Find last trade index j with tr_ts[j] <= t_fwd
            j = np.searchsorted(tr_ts, t_fwd, side='right') - 1
            if j <= i:
                disp_signed_tk[k] = 0.0
                disp_abs_tk[k] = 0.0
                continue
            p_fwd = tr_px[j]
            raw_disp = p_fwd - p_now  # in raw fixed-point units
            tk = raw_disp / TICK_RAW
            sign = +1.0 if tr_side[i] == SIDE_A else (-1.0 if tr_side[i] == SIDE_B else 0.0)
            disp_signed_tk[k] = sign * tk
            disp_abs_tk[k] = abs(tk)
        return {
            "n": len(picks),
            "mean_signed_tk": float(np.mean(disp_signed_tk)),
            "median_signed_tk": float(np.median(disp_signed_tk)),
            "std_signed_tk": float(np.std(disp_signed_tk)),
            "mean_abs_tk": float(np.mean(disp_abs_tk)),
            "p90_abs_tk": float(np.quantile(disp_abs_tk, 0.9)),
        }

    sw_stats = forward_signed_disp(sw_pick)
    bg_stats = forward_signed_disp(bg_pick)

    # Also small-N robustness: 50-event check the task asked for
    sw_pick50 = rng.choice(sweep_trade_pos,
                           size=min(50, len(sweep_trade_pos)), replace=False)
    sw50 = forward_signed_disp(sw_pick50)

    elapsed = time.time() - t0
    report = {
        "date": args.date,
        "front_month_id": front_id,
        "n_trades_front_month": int(len(tr_ts)),
        "n_sweep_trades": int(len(sweep_trade_pos)),
        "n_large_print_trades": int(tr_large.sum()),
        "sweep_sample_2000": sw_stats,
        "background_sample_2000_non_sweep": bg_stats,
        "sweep_sample_50_robustness": sw50,
        "delta_mean_signed_sweep_minus_bg_tk":
            sw_stats["mean_signed_tk"] - bg_stats["mean_signed_tk"],
        "delta_mean_abs_sweep_minus_bg_tk":
            sw_stats["mean_abs_tk"] - bg_stats["mean_abs_tk"],
        "elapsed_s": round(elapsed, 2),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
