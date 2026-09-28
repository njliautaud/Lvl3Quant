#!/usr/bin/env python3
"""
tp_optimization_sweep.py -- Test different take-profit levels per card.
Cards: C1, C4, C5, C7 on OOT dates.
TP values: [4, 6, 8, 10, 13, 15, 20, 25, 30]
Uses existing fill_sim_cli binary and stacked predictions.
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
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/tp_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 8)
WORKERS   = 14

TP_VALUES = [4, 6, 8, 10, 13, 15, 20, 25, 30]

CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "hold_ms": 7200000,
        "base_tp": 8,
        "desc": "Card1: book predstdExit conv1.5 vol50",
    },
    "c4": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.1,
        "hold_ms": 7200000,
        "base_tp": 20,
        "desc": "Card4: book predstdExit conv2.0 vol70",
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "signal_threshold": 0.1,
        "hold_ms": 3600000,
        "base_tp": None,
        "desc": "Card5: raw rawExit conv0.05 ethr0.5 vol0",
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "hold_ms": 3600000,
        "base_tp": None,
        "desc": "Card7: smooth smoothExit conv1.5 vol70",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tp_sweep")


def discover_dates():
    dates = []
    d = OOT_START
    while d <= OOT_END:
        mbo = MBO_DIR / f"glbx-mdp3-{d.strftime('%Y%m%d')}.mbo.dbn.zst"
        if mbo.exists():
            dates.append(d)
        d = d + datetime.timedelta(days=1)
    return dates


def run_fill_sim(card_id, card_def, tp_ticks, date_obj):
    date_iso = date_obj.isoformat()
    date_num = date_obj.strftime("%Y%m%d")
    mbo_file  = MBO_DIR  / f"glbx-mdp3-{date_num}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date_iso}_{card_def['pred_suffix']}.npz"

    if not mbo_file.exists() or not pred_file.exists():
        return None

    tp_label = f"TP{tp_ticks}" if tp_ticks is not None else "TPnone"
    card_out_dir = OUT_DIR / card_id / tp_label
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
        "--hold-ms", str(card_def["hold_ms"]),
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--latency-ms", "50",
    ]
    if tp_ticks is not None:
        cmd += ["--take-profit-ticks", str(tp_ticks)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Error {card_id} TP{tp_ticks} {date_iso}: {e}")
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
    }


def main():
    dates = discover_dates()
    log.info("TP optimization sweep starting")
    log.info(f"Cards: {list(CARDS.keys())}")
    log.info(f"TP values: {TP_VALUES}")
    log.info(f"OOT dates: {len(dates)}")

    jobs = []
    for card_id, card_def in CARDS.items():
        for tp in TP_VALUES:
            for d in dates:
                jobs.append((card_id, card_def, tp, d))

    log.info(f"Total jobs: {len(jobs)}")

    results_by_card_tp = defaultdict(list)
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_fill_sim, *j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            card_id, card_def, tp, d = job
            try:
                r = fut.result()
                if r is not None:
                    r["date"] = d.isoformat()
                    results_by_card_tp[(card_id, tp)].append(r)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.warning(f"Job failed {card_id} TP{tp} {d}: {e}")
            done += 1
            if done % 100 == 0:
                log.info(f"  {done}/{len(jobs)} done, {errors} errors")

    log.info(f"All jobs done. {errors} errors total.")

    summary = {}
    for card_id, card_def in CARDS.items():
        summary[card_id] = {
            "desc": card_def["desc"],
            "base_tp": card_def["base_tp"],
            "tp_results": {}
        }
        log.info(f"\n{'='*60}")
        log.info(f"Card: {card_id} -- {card_def['desc']}")
        log.info(f"{'='*60}")
        log.info(f"  {'TP':>6}  {'Trades':>7}  {'Sharpe':>8}  {'PnL':>12}  {'WinRate':>8}")
        log.info(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*12}  {'-'*8}")
        for tp in TP_VALUES:
            rlist = results_by_card_tp.get((card_id, tp), [])
            m = compute_metrics(rlist)
            summary[card_id]["tp_results"][f"TP{tp}"] = m
            marker = " <-- current" if tp == card_def.get("base_tp") else ""
            log.info(f"  TP{tp:>4}  {m['n_trades']:>7}  {m['sharpe']:>8.3f}  ${m['total_pnl']:>10,.0f}  {m['win_rate']:>7.1%}{marker}")
        best_tp = max(TP_VALUES, key=lambda t: summary[card_id]["tp_results"][f"TP{t}"]["sharpe"])
        best_m = summary[card_id]["tp_results"][f"TP{best_tp}"]
        summary[card_id]["best_tp"] = best_tp
        summary[card_id]["best_sharpe"] = best_m["sharpe"]
        log.info(f"  >> BEST: TP{best_tp} = Sharpe {best_m['sharpe']:.3f} (PnL ${best_m['total_pnl']:,.0f})")

    out_file = OUT_DIR / "tp_sweep_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nResults saved to {out_file}")
    log.info(f"[tp_sweep] DONE -- {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
