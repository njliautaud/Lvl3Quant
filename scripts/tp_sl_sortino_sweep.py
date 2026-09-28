#!/usr/bin/env python3
"""tp_sl_sortino_sweep.py -- 2D TP x SL sweep, Sortino-maximizing."""
import json, logging, subprocess, datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import numpy as np

FILL_SIM = Path("/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli")
MBO_DIR  = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/tp_sl_sortino_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 8)
WORKERS   = 14
TP_VALUES = [8, 10, 13, 15, 20]
SL_VALUES = [10, 15, 20, 25, 30]
CARDS = {
    "c1": {"pred_suffix": "book_predstdExit_conv1.5_vol50", "signal_threshold": 0.1, "hold_ms": 7200000, "desc": "Card1"},
    "c4": {"pred_suffix": "book_predstdExit_conv2.0_vol70", "signal_threshold": 0.1, "hold_ms": 7200000, "desc": "Card4"},
    "c5": {"pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0", "signal_threshold": 0.1, "hold_ms": 3600000, "desc": "Card5"},
    "c7": {"pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70", "signal_threshold": 0.1, "hold_ms": 3600000, "desc": "Card7"},
}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tp_sl_sweep")

def discover_dates():
    dates = []
    d = OOT_START
    while d <= OOT_END:
        mbo = MBO_DIR / ("glbx-mdp3-" + d.strftime("%Y%m%d") + ".mbo.dbn.zst")
        if mbo.exists():
            dates.append(d)
        d += datetime.timedelta(days=1)
    return dates

def run_fill_sim(card_id, card_def, tp_ticks, sl_ticks, date_obj):
    date_iso = date_obj.isoformat()
    date_num = date_obj.strftime("%Y%m%d")
    mbo_file  = MBO_DIR  / ("glbx-mdp3-" + date_num + ".mbo.dbn.zst")
    pred_file = PRED_DIR / (date_iso + "_" + card_def["pred_suffix"] + ".npz")
    if not mbo_file.exists() or not pred_file.exists():
        return None
    label = "TP" + str(tp_ticks) + "_SL" + str(sl_ticks)
    card_out_dir = OUT_DIR / card_id / label
    card_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = card_out_dir / (date_iso + ".json")
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
        "--take-profit-ticks", str(tp_ticks),
        "--stop-loss-ticks", str(sl_ticks),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except Exception as e:
        log.warning("Error " + card_id + " TP" + str(tp_ticks) + "/SL" + str(sl_ticks) + " " + date_iso + ": " + str(e))
    return None

def sortino_from_daily(daily_pnls, mar=0.0):
    if len(daily_pnls) < 2:
        return 0.0
    arr = np.array(daily_pnls, dtype=float)
    excess = arr - mar
    mean_excess = float(np.mean(excess))
    downside = excess[excess < 0]
    if len(downside) == 0:
        downside_std = 1e-6
    else:
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return 0.0
    return mean_excess / downside_std * (252 ** 0.5)

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
        return {"n_trades": 0, "sortino": 0.0, "sharpe": 0.0, "total_pnl": 0.0,
                "win_rate": 0.0, "avg_pnl": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "win_loss_ratio": 0.0, "n_days": 0}
    daily = defaultdict(float)
    all_pnls = []
    for r in results:
        for trade in r.get("trades", []):
            pnl = trade.get("pnl_dollars", 0.0)
            all_pnls.append(pnl)
            daily[r.get("date", "?")] += pnl
    daily_vals = list(daily.values())
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    return {
        "n_trades": len(all_pnls), "n_days": len(daily_vals),
        "sortino": round(sortino_from_daily(daily_vals), 4),
        "sharpe":  round(sharpe_from_daily(daily_vals), 4),
        "total_pnl": round(sum(all_pnls), 2),
        "win_rate": round(sum(1 for p in all_pnls if p > 0) / len(all_pnls), 4) if all_pnls else 0,
        "avg_pnl": round(float(np.mean(all_pnls)), 2) if all_pnls else 0,
        "avg_win":  round(avg_win, 2), "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(win_loss, 3),
    }

def main():
    dates = discover_dates()
    log.info("=" * 70)
    log.info("TP/SL 2D Sortino Sweep")
    log.info(str(len(CARDS)) + " cards, TP=" + str(TP_VALUES) + ", SL=" + str(SL_VALUES))
    log.info(str(len(dates)) + " OOT days (" + str(dates[0]) + " to " + str(dates[-1]) + ")")
    total_jobs = len(CARDS) * len(TP_VALUES) * len(SL_VALUES) * len(dates)
    log.info("Total jobs: " + str(total_jobs))
    log.info("=" * 70)
    jobs = []
    for card_id, card_def in CARDS.items():
        for tp in TP_VALUES:
            for sl in SL_VALUES:
                for d in dates:
                    jobs.append((card_id, card_def, tp, sl, d))
    results_by_key = defaultdict(list)
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_fill_sim, *j): j for j in jobs}
        for fut in as_completed(futures):
            card_id, card_def, tp, sl, d = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    r["date"] = d.isoformat()
                    results_by_key[(card_id, tp, sl)].append(r)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.warning("Failed " + card_id + " TP" + str(tp) + "/SL" + str(sl) + " " + str(d) + ": " + str(e))
            done += 1
            if done % 200 == 0:
                pct = int(done / len(jobs) * 100)
                log.info("  " + str(done) + "/" + str(len(jobs)) + " (" + str(pct) + "%) " + str(errors) + " errors")
    log.info("Jobs done. " + str(errors) + " errors.")
    summary = {}
    for card_id, card_def in CARDS.items():
        summary[card_id] = {"desc": card_def["desc"], "matrix": {}, "best_tp": None, "best_sl": None, "best_sortino": -999.0}
        log.info("=" * 70)
        log.info("Card: " + card_id + " " + card_def["desc"])
        log.info("  Config         N  Sortino   Sharpe          PnL      WR   W/L")
        for tp in TP_VALUES:
            for sl in SL_VALUES:
                m = compute_metrics(results_by_key.get((card_id, tp, sl), []))
                key = "TP" + str(tp) + "_SL" + str(sl)
                summary[card_id]["matrix"][key] = m
                if m["sortino"] > summary[card_id]["best_sortino"] and m["n_trades"] >= 50:
                    summary[card_id]["best_sortino"] = m["sortino"]
                    summary[card_id]["best_tp"] = tp
                    summary[card_id]["best_sl"] = sl
                log.info("  " + key.ljust(12) + "  " + str(m["n_trades"]).rjust(6) + "  " +
                         "{:.3f}".format(m["sortino"]).rjust(7) + "  " +
                         "{:.3f}".format(m["sharpe"]).rjust(7) + "  " +
                         "${:>10,.0f}".format(m["total_pnl"]) + "  " +
                         "{:.1%}".format(m["win_rate"]).rjust(5) + "  " +
                         "{:.2f}".format(m["win_loss_ratio"]).rjust(5))
        bt = summary[card_id]["best_tp"]
        bs = summary[card_id]["best_sl"]
        bm = summary[card_id]["matrix"].get("TP" + str(bt) + "_SL" + str(bs), {})
        log.info("  >> BEST: TP" + str(bt) + "/SL" + str(bs) +
                 " Sortino=" + "{:.3f}".format(bm.get("sortino", 0)) +
                 " PnL=${:,.0f}".format(bm.get("total_pnl", 0)) +
                 " W/L=" + "{:.2f}".format(bm.get("win_loss_ratio", 0)))
    log.info("=" * 70)
    log.info("FINAL RECOMMENDATIONS (Sortino-max, min 50 trades):")
    for cid in CARDS:
        s = summary[cid]
        log.info("  " + cid + ": TP=" + str(s["best_tp"]) + " SL=" + str(s["best_sl"]) +
                 " Sortino=" + "{:.3f}".format(s["best_sortino"]))
    out_file = OUT_DIR / "tp_sl_sortino_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved: " + str(out_file))
    log.info("[tp_sl_sweep] DONE " + str(datetime.datetime.now()))

if __name__ == "__main__":
    main()
