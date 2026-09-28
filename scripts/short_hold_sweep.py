#!/usr/bin/env python3
"""
short_hold_sweep.py -- Test short hold durations [10s, 30s, 1min, 2min, 5min].
Cards: C1, C4, C5, C7 on OOT dates.
Verifies whether ANY short hold is profitable with the current model.
Uses chase entry (same as HFT card setup).
"""
import json
import logging
import subprocess
import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

# Paths
FILL_SIM = Path("/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli")
MBO_DIR  = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/short_hold_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 8)
WORKERS   = 14

# Hold durations in ms
HOLD_CONFIGS = [
    ("10s",  10_000),
    ("30s",  30_000),
    ("1min", 60_000),
    ("2min", 120_000),
    ("5min", 300_000),
]

CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "tp_ticks": 8,
        "desc": "Card1: book predstdExit conv1.5 vol50, TP8",
    },
    "c4": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "desc": "Card4: book predstdExit conv2.0 vol70, TP20",
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "signal_threshold": 0.1,
        "tp_ticks": None,
        "desc": "Card5: raw rawExit conv0.05 ethr0.5 vol0, no TP",
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": None,
        "desc": "Card7: smooth smoothExit conv1.5 vol70, no TP",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("short_hold")


def discover_dates():
    dates = []
    d = OOT_START
    while d <= OOT_END:
        mbo = MBO_DIR / f"glbx-mdp3-{d.strftime('%Y%m%d')}.mbo.dbn.zst"
        if mbo.exists():
            dates.append(d)
        d = d + datetime.timedelta(days=1)
    return dates


def run_fill_sim(card_id, card_def, hold_label, hold_ms, date_obj):
    date_iso = date_obj.isoformat()
    date_num = date_obj.strftime("%Y%m%d")
    mbo_file  = MBO_DIR  / f"glbx-mdp3-{date_num}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date_iso}_{card_def['pred_suffix']}.npz"

    if not mbo_file.exists() or not pred_file.exists():
        return None

    card_out_dir = OUT_DIR / card_id / hold_label
    card_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = card_out_dir / f"{date_iso}.json"

    if out_path.exists():
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception:
            pass

    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_path),
        "--signal-threshold", str(card_def["signal_threshold"]),
        "--hold-ms", str(hold_ms),
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--latency-ms", "50",
    ]
    if card_def.get("tp_ticks") is not None:
        cmd += ["--take-profit-ticks", str(card_def["tp_ticks"])]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Error {card_id} {hold_label} {date_iso}: {e}")
    return None


def sharpe_from_daily(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    m = float(np.mean(daily_pnls))
    s = float(np.std(daily_pnls, ddof=1))
    if s == 0:
        return 0.0
    return m / s * (252 ** 0.5)


def compute_metrics(results):
    if not results:
        return {"n_trades": 0, "sharpe": 0.0, "total_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0}
    daily = defaultdict(float)
    all_pnls = []
    for r in results:
        for trade in r.get("trades", []):
            pnl = trade.get("pnl_dollars", 0.0)
            all_pnls.append(pnl)
            daily[r.get("date", "unknown")] += pnl
    daily_vals = list(daily.values())
    sh = sharpe_from_daily(daily_vals)
    return {
        "n_trades": len(all_pnls),
        "sharpe": round(sh, 4),
        "total_pnl": round(sum(all_pnls), 2),
        "win_rate": round(sum(1 for p in all_pnls if p > 0) / len(all_pnls), 4) if all_pnls else 0,
        "avg_pnl": round(float(np.mean(all_pnls)), 2) if all_pnls else 0,
        "fill_rate": round(
            sum(r.get("total_filled", 0) for r in results) /
            max(1, sum(r.get("total_posted", 1) for r in results)), 4
        ),
    }


def main():
    dates = discover_dates()
    log.info("Short hold sweep starting")
    log.info(f"Cards: {list(CARDS.keys())}")
    log.info(f"Hold durations: {[h for h,_ in HOLD_CONFIGS]}")
    log.info(f"OOT dates: {len(dates)}")

    jobs = []
    for card_id, card_def in CARDS.items():
        for hold_label, hold_ms in HOLD_CONFIGS:
            for d in dates:
                jobs.append((card_id, card_def, hold_label, hold_ms, d))

    log.info(f"Total jobs: {len(jobs)}")

    results_by_card_hold = defaultdict(list)
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_fill_sim, *j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            card_id, card_def, hold_label, hold_ms, d = job
            try:
                r = fut.result()
                if r is not None:
                    r["date"] = d.isoformat()
                    results_by_card_hold[(card_id, hold_label)].append(r)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.warning(f"Job failed {card_id} {hold_label} {d}: {e}")
            done += 1
            if done % 100 == 0:
                log.info(f"  {done}/{len(jobs)} done, {errors} errors")

    log.info(f"All jobs done. {errors} errors total.")

    summary = {}
    for card_id, card_def in CARDS.items():
        summary[card_id] = {
            "desc": card_def["desc"],
            "hold_results": {}
        }
        log.info(f"\n{'='*60}")
        log.info(f"Card: {card_id} -- {card_def['desc']}")
        log.info(f"{'='*60}")
        log.info(f"  {'Hold':>6}  {'Trades':>7}  {'Sharpe':>8}  {'PnL':>12}  {'WinRate':>8}  {'FillRate':>9}")
        log.info(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*9}")
        for hold_label, _ in HOLD_CONFIGS:
            rlist = results_by_card_hold.get((card_id, hold_label), [])
            m = compute_metrics(rlist)
            summary[card_id]["hold_results"][hold_label] = m
            profitable = " PROFIT" if m["sharpe"] > 0 and m["total_pnl"] > 0 else ""
            log.info(f"  {hold_label:>6}  {m['n_trades']:>7}  {m['sharpe']:>8.3f}  ${m['total_pnl']:>10,.0f}  {m['win_rate']:>7.1%}  {m.get('fill_rate', 0):>8.1%}{profitable}")
        # Summary: any profitable?
        profitable_holds = [(h, summary[card_id]["hold_results"][h]) for h, _ in HOLD_CONFIGS
                           if summary[card_id]["hold_results"][h]["sharpe"] > 0]
        if profitable_holds:
            best = max(profitable_holds, key=lambda x: x[1]["sharpe"])
            log.info(f"  >> BEST PROFITABLE: {best[0]} Sharpe={best[1]['sharpe']:.3f}")
            summary[card_id]["best_hold"] = best[0]
        else:
            log.info(f"  >> NO profitable short holds found for {card_id}")
            summary[card_id]["best_hold"] = None

    out_file = OUT_DIR / "short_hold_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nResults saved to {out_file}")
    log.info(f"[short_hold] DONE -- {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
