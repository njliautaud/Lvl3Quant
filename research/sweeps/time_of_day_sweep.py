#!/usr/bin/env python3
"""
Time-of-day + conviction exit sweep.
Answers: does 10s CNN edge cluster in specific intraday windows? Does conviction exit help?
Relevant for 30s CNN: if edge is time-clustered, longer-horizon model may behave differently.
"""
import sys, json, time, subprocess, os, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

WORKERS = 12
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "time_of_day_sweep"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70
OOT_START = "2025-12-01"
OOT_END   = "2026-03-08"

# Configs: (name, pred_type, sig, tp, sl, hold_ms, mae_t, mae_s)
BASE_CONFIGS = [
    ("c1_tp13_s01",   "book_predstdExit_conv1.5_vol50",    0.1,  13,   None, 7200000,  None, None),
    ("c4_tp20_s03",   "book_predstdExit_conv2.0_vol70",    0.3,  20,   None, 7200000,  None, None),
    ("c5_raw_s01",    "raw_rawExit_conv0.05_ethr0.5_vol0", 0.1,  None, None, 3600000,  None, None),
    ("c7_mae15_s015", "smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.15, None, 20, 3600000, 15, 300),
]

# Session variants: (label, prime_hours, conviction_exit_bars)
SESSION_VARIANTS = [
    ("full_session",    False, 0),
    ("prime_hours",     True,  0),
    ("conviction_3s",   False, 30),
    ("conviction_10s",  False, 100),
    ("conviction_30s",  False, 300),
    ("prime_conv10s",   True,  100),
]

def find_dates_for_pred(pred_type):
    dates = {}
    for f in sorted(PRED_DIR.glob(f"*_{pred_type}.npz")):
        date = f.stem[:10]
        if not (OOT_START <= date <= OOT_END):
            continue
        nodash = date.replace("-","")
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
        if mbo.exists():
            dates[date] = (f, mbo)
    return dates

def run_one(args):
    cfg_name, pred_type, sig, tp, sl, hold_ms, mae_t, mae_s, sess_label, prime, conv_bars, date, pred_file, mbo_file = args
    tag = f"{cfg_name}_{sess_label}"
    out_file = OUT_BASE / f"{date}_{tag}.json"
    if out_file.exists():
        try:
            d = json.loads(out_file.read_text())
            return (tag, date, d)
        except:
            pass
    cmd = [str(BINARY),
           "--mbo-file", str(mbo_file),
           "--predictions", str(pred_file),
           "--output", str(out_file),
           "--signal-threshold", str(sig),
           "--hold-ms", str(hold_ms),
           "--size", "1",
    ]
    if tp: cmd += ["--take-profit-ticks", str(tp)]
    if sl: cmd += ["--stop-loss-ticks", str(sl)]
    if mae_t and mae_s:
        cmd += ["--mae-exit-ticks", str(mae_t), "--mae-exit-hold-sec", str(mae_s)]
    if prime: cmd.append("--prime-hours")
    if conv_bars > 0:
        cmd += ["--conviction-exit-bars", str(conv_bars)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and out_file.exists():
            return (tag, date, json.loads(out_file.read_text()))
    except Exception as e:
        pass
    return (tag, date, None)

def aggregate(results_by_tag):
    summary = {}
    for tag, days in results_by_tag.items():
        valid = [d for d in days if d and d.get("total_trades", 0) > 0]
        if not valid:
            summary[tag] = {"n_days": 0, "sharpe": None, "total_pnl": 0, "n_trades": 0}
            continue
        daily_pnls = [d.get("total_pnl_dollars", 0) for d in valid]
        n_trades = sum(d.get("total_trades", 0) for d in valid)
        total_pnl = sum(daily_pnls)
        if len(daily_pnls) > 1 and statistics.stdev(daily_pnls) > 0:
            sharpe = (statistics.mean(daily_pnls) / statistics.stdev(daily_pnls)) * (252**0.5)
        else:
            sharpe = 0
        summary[tag] = {
            "n_days": len(valid),
            "n_trades": n_trades,
            "total_pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 3),
            "trades_per_day": round(n_trades / len(valid), 1),
        }
    return summary

def main():
    print(f"[tod_sweep] Starting time-of-day + conviction sweep -- {datetime.now()}", flush=True)

    all_tasks = []
    for cfg_name, pred_type, sig, tp, sl, hold_ms, mae_t, mae_s in BASE_CONFIGS:
        dates = find_dates_for_pred(pred_type)
        print(f"[tod_sweep] {cfg_name}: {len(dates)} dates for {pred_type}", flush=True)
        for sess_label, prime, conv_bars in SESSION_VARIANTS:
            for date, (pred_file, mbo_file) in dates.items():
                all_tasks.append((cfg_name, pred_type, sig, tp, sl, hold_ms, mae_t, mae_s,
                                  sess_label, prime, conv_bars, date, pred_file, mbo_file))

    print(f"[tod_sweep] Total tasks: {len(all_tasks)}", flush=True)

    results_by_tag = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_one, t): t for t in all_tasks}
        for fut in as_completed(futures):
            tag, date, res = fut.result()
            if tag not in results_by_tag:
                results_by_tag[tag] = []
            results_by_tag[tag].append(res)
            done += 1
            if done % 100 == 0:
                print(f"[tod_sweep] Progress: {done}/{len(all_tasks)}", flush=True)

    summary = aggregate(results_by_tag)

    print("\n" + "="*90, flush=True)
    print("TIME-OF-DAY + CONVICTION SWEEP RESULTS", flush=True)
    print("="*90, flush=True)
    for cfg_name, pred_type, sig, tp, sl, hold_ms, mae_t, mae_s in BASE_CONFIGS:
        print(f"\n  {cfg_name}", flush=True)
        print(f"  {'Session':<20} {'Sharpe':>8} {'PnL':>12} {'Trades':>8} {'T/Day':>8}", flush=True)
        print(f"  {'-'*60}", flush=True)
        for sess_label, prime, conv_bars in SESSION_VARIANTS:
            tag = f"{cfg_name}_{sess_label}"
            s = summary.get(tag, {})
            sharpe = s.get("sharpe", "N/A")
            pnl = s.get("total_pnl_dollars", "N/A")
            nt = s.get("total_trades", 0)
            tpd = s.get("trades_per_day", "N/A")
            print(f"  {sess_label:<20} {str(sharpe):>8} {str(pnl):>12} {nt:>8} {str(tpd):>8}", flush=True)

    out_path = OUT_BASE / "tod_sweep_summary.json"
    out_path.write_text(json.dumps({"timestamp": datetime.now().isoformat(), "summary": summary}, indent=2))
    print(f"\n[tod_sweep] Results saved to {out_path}", flush=True)
    print(f"[tod_sweep] DONE -- {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
