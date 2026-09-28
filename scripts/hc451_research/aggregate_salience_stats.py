#!/usr/bin/env python3
"""HC #451 — Aggregate salience-tag stats across all per-day outputs.

Reads per-day stats JSONs (from precompute) for counts and rates.
Replicates the validate_salience_tags forward-5s signed-displacement
methodology to compute sweep-vs-background averaged across days.

Outputs aggregate JSON + SUMMARY.md.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl
import databento as db

OUT_ROOT = Path("/home/jupiter/Lvl3Quant/output/hc451_salience_tags")
PER_DAY = OUT_ROOT / "per_day"
RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")

TICK_RAW = 250_000_000
A_TRADE = ord('T')
SIDE_A = ord('A')
SIDE_B = ord('B')
FORWARD_NS = 5_000_000_000

N_SWEEP = 2000
N_BG = 2000
SEED = 42

N_WORKERS = 6


def per_day_displacement(date_str):
    """Replicates validate_salience_tags forward_signed_disp for one date."""
    try:
        parq = PER_DAY / f"{date_str}_salience.parquet"
        dbn = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        if not parq.exists() or not dbn.exists():
            return {"date": date_str, "status": "missing"}

        salience = pl.read_parquet(parq)
        sweep_flags = salience["sweep_tag"].to_numpy()

        store = db.DBNStore.from_file(str(dbn))
        arr = store.to_ndarray()
        actions = arr['action'].view(np.uint8)
        sides = arr['side'].view(np.uint8)
        iids = arr['instrument_id']
        ts_event = arr['ts_event'].astype(np.int64)
        prices = arr['price'].astype(np.int64)

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

        sweep_pos = np.flatnonzero(tr_sweep)
        nonsweep_pos = np.flatnonzero(~tr_sweep)
        if len(sweep_pos) == 0 or len(nonsweep_pos) == 0:
            return {"date": date_str, "status": "empty_sweeps",
                    "n_sweep": int(len(sweep_pos))}

        rng = np.random.default_rng(SEED)
        sw_pick = rng.choice(sweep_pos,
                             size=min(N_SWEEP, len(sweep_pos)), replace=False)
        bg_pick = rng.choice(nonsweep_pos,
                             size=min(N_BG, len(nonsweep_pos)), replace=False)

        def fsd(picks):
            disp_signed = np.empty(len(picks))
            disp_abs = np.empty(len(picks))
            for k, i in enumerate(picks):
                t_now = tr_ts[i]
                p_now = tr_px[i]
                t_fwd = t_now + FORWARD_NS
                j = np.searchsorted(tr_ts, t_fwd, side='right') - 1
                if j <= i:
                    disp_signed[k] = 0.0
                    disp_abs[k] = 0.0
                    continue
                raw = tr_px[j] - p_now
                tk = raw / TICK_RAW
                s = 1.0 if tr_side[i] == SIDE_A else (-1.0 if tr_side[i] == SIDE_B else 0.0)
                disp_signed[k] = s * tk
                disp_abs[k] = abs(tk)
            return float(np.mean(disp_signed)), float(np.mean(disp_abs)), len(picks)

        sw_mean_signed, sw_mean_abs, sw_n = fsd(sw_pick)
        bg_mean_signed, bg_mean_abs, bg_n = fsd(bg_pick)

        return {
            "date": date_str,
            "status": "ok",
            "n_trades_front": int(len(tr_ts)),
            "n_sweep_total": int(len(sweep_pos)),
            "sw_n": sw_n,
            "bg_n": bg_n,
            "sw_mean_signed_tk": sw_mean_signed,
            "sw_mean_abs_tk": sw_mean_abs,
            "bg_mean_signed_tk": bg_mean_signed,
            "bg_mean_abs_tk": bg_mean_abs,
        }
    except Exception as e:
        return {"date": date_str, "status": "error", "error": repr(e)}


def main():
    # Load all per-day stats JSONs
    stats_files = sorted(PER_DAY.glob("*_stats.json"))
    per_day_stats = [json.loads(p.read_text()) for p in stats_files]
    dates = [s["date"] for s in per_day_stats if s.get("status") == "ok"]
    print(f"[agg] {len(dates)} day stats loaded")

    # Counts & rates
    sweep_counts = np.array([s["sweep_tag_count"] for s in per_day_stats])
    large_counts = np.array([s["large_print_tag_count"] for s in per_day_stats])
    n_trades = np.array([s["n_trades_front"] for s in per_day_stats])
    sweep_rates = np.array([s["sweep_tag_rate_per_trade"] for s in per_day_stats])
    large_rates = np.array([s["large_print_tag_rate_per_trade"] for s in per_day_stats])
    elapsed_total = sum(s["elapsed_total_s"] for s in per_day_stats)

    print(f"[agg] starting forward-displacement on {len(dates)} dates "
          f"with {N_WORKERS} workers")
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        disp_results = list(pool.imap_unordered(per_day_displacement, dates))
    disp_wall = time.time() - t0
    print(f"[agg] displacement done in {disp_wall:.1f}s")

    disp_ok = [r for r in disp_results if r.get("status") == "ok"]
    disp_fail = [r for r in disp_results if r.get("status") != "ok"]
    sw_signed = np.array([r["sw_mean_signed_tk"] for r in disp_ok])
    bg_signed = np.array([r["bg_mean_signed_tk"] for r in disp_ok])
    sw_abs = np.array([r["sw_mean_abs_tk"] for r in disp_ok])
    bg_abs = np.array([r["bg_mean_abs_tk"] for r in disp_ok])

    agg = {
        "n_days_processed": len(per_day_stats),
        "n_days_displacement_ok": len(disp_ok),
        "n_days_displacement_failed": len(disp_fail),
        "total_sweep_count": int(sweep_counts.sum()),
        "total_large_print_count": int(large_counts.sum()),
        "total_trades_front": int(n_trades.sum()),
        "per_day_sweep_count_mean": float(sweep_counts.mean()),
        "per_day_sweep_count_std": float(sweep_counts.std()),
        "per_day_large_print_count_mean": float(large_counts.mean()),
        "per_day_large_print_count_std": float(large_counts.std()),
        "sweep_fire_rate_per_trade_mean": float(sweep_rates.mean()),
        "sweep_fire_rate_per_trade_std": float(sweep_rates.std()),
        "large_print_fire_rate_per_trade_mean": float(large_rates.mean()),
        "across_days_sweep_5s_signed_disp_mean_tk":
            float(sw_signed.mean()),
        "across_days_sweep_5s_signed_disp_std_tk":
            float(sw_signed.std()),
        "across_days_bg_5s_signed_disp_mean_tk":
            float(bg_signed.mean()),
        "across_days_bg_5s_signed_disp_std_tk":
            float(bg_signed.std()),
        "across_days_sweep_minus_bg_signed_tk":
            float(sw_signed.mean() - bg_signed.mean()),
        "across_days_sweep_5s_abs_disp_mean_tk": float(sw_abs.mean()),
        "across_days_bg_5s_abs_disp_mean_tk": float(bg_abs.mean()),
        "across_days_sweep_minus_bg_abs_tk":
            float(sw_abs.mean() - bg_abs.mean()),
        "cpu_seconds_sum_precompute": round(elapsed_total, 1),
        "displacement_wall_s": round(disp_wall, 1),
    }
    (OUT_ROOT / "aggregate_stats.json").write_text(json.dumps(agg, indent=2))
    if disp_fail:
        (OUT_ROOT / "displacement_failed.txt").write_text(
            "\n".join(f"{r['date']}\t{r.get('status')}\t{r.get('error','')}" for r in disp_fail)
        )
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
