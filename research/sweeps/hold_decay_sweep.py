#!/usr/bin/env python3
"""
Hold-time decay sweep — sweep hold_ms across top card configs.
Answers: what is the optimal hold time for each card's 10s CNN predictions?
Key question for 30s CNN validation: will longer holds (matched to 30s horizon) hurt or help?
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
OUT_BASE = LVL3_ROOT / "data" / "processed" / "hold_decay_sweep"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

OOT_START = "2025-12-01"
OOT_END   = "2026-03-08"

# Top card configs from targeted opt — use their best pred_type + params, sweep hold_ms
# Format: (name, pred_type, sig_thr, tp_ticks, sl_ticks, mae_ticks, mae_hold_sec, prime_hours)
CARDS = [
    # C1 best: TP13, sig0.1, short-only direction (Sharpe 4.91 in targeted opt)
    ("c1_tp13_s01", "book_predstdExit_conv1.5_vol50", 0.1, 13, None, None, None, False),
    # C4 best: sig0.3, tp20, long-only (Sharpe 3.76)
    ("c4_tp20_s03", "book_predstdExit_conv2.0_vol70", 0.3, 20, None, None, None, False),
    # C5 best: sig0.1 baseline (Sharpe 3.66)
    ("c5_raw_s01", "raw_rawExit_conv0.05_ethr0.5_vol0", 0.1, None, None, None, None, False),
    # C7 best: prime hours + MAE15t/300s (Sharpe 4.22)
    ("c7_prime_mae15", "smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.15, None, 20, 15, 300, True),
]

# Hold times to sweep: from 10s (100 bars, native horizon) to 4hr
HOLD_TIMES_MS = [10000, 30000, 60000, 120000, 300000, 600000, 1800000, 3600000, 7200000, 14400000]
HOLD_LABELS   = ["10s", "30s", "60s", "2min", "5min", "10min", "30min", "1hr", "2hr", "4hr"]

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
    card_name, pred_type, sig, tp, sl, mae_t, mae_s, prime, hold_ms, hold_label, date, pred_file, mbo_file = args
    tag = f"{card_name}_{hold_label}"
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
            summary[tag] = {"n_days": 0, "sharpe": None}
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
            "trades_per_day": round(n_trades / len(valid), 1) if valid else 0,
        }
    return summary

def main():
    print(f"[hold_decay] Starting hold-time decay sweep -- {datetime.now()}", flush=True)
    print(f"[hold_decay] Cards: {len(CARDS)}, Hold times: {len(HOLD_TIMES_MS)}", flush=True)

    all_tasks = []
    for card_name, pred_type, sig, tp, sl, mae_t, mae_s, prime in CARDS:
        dates = find_dates_for_pred(pred_type)
        print(f"[hold_decay] {card_name}: {len(dates)} dates found for {pred_type}", flush=True)
        for hold_ms, hold_label in zip(HOLD_TIMES_MS, HOLD_LABELS):
            for date, (pred_file, mbo_file) in dates.items():
                all_tasks.append((card_name, pred_type, sig, tp, sl, mae_t, mae_s, prime,
                                  hold_ms, hold_label, date, pred_file, mbo_file))

    print(f"[hold_decay] Total tasks: {len(all_tasks)}", flush=True)

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
                print(f"[hold_decay] Progress: {done}/{len(all_tasks)}", flush=True)

    summary = aggregate(results_by_tag)

    print("\n" + "="*90, flush=True)
    print("HOLD-TIME DECAY ANALYSIS RESULTS", flush=True)
    print("="*90, flush=True)
    for card_name, pred_type, sig, tp, sl, mae_t, mae_s, prime in CARDS:
        print(f"\n  {card_name} ({pred_type})", flush=True)
        print(f"  {'Hold':<10} {'Sharpe':>8} {'PnL':>12} {'Trades/Day':>12} {'N Days':>8}", flush=True)
        print(f"  {'-'*54}", flush=True)
        for hold_ms, hold_label in zip(HOLD_TIMES_MS, HOLD_LABELS):
            tag = f"{card_name}_{hold_label}"
            s = summary.get(tag, {})
            sharpe = s.get("sharpe", "N/A")
            pnl = s.get("total_pnl_dollars", "N/A")
            tpd = s.get("trades_per_day", "N/A")
            nd = s.get("n_days", 0)
            print(f"  {hold_label:<10} {str(sharpe):>8} {str(pnl):>12} {str(tpd):>12} {nd:>8}", flush=True)

    out_path = OUT_BASE / "hold_decay_summary.json"
    out_path.write_text(json.dumps({"timestamp": datetime.now().isoformat(), "summary": summary}, indent=2))
    print(f"\n[hold_decay] Results saved to {out_path}", flush=True)
    print(f"[hold_decay] DONE -- {datetime.now()}", flush=True)

if __name__ == "__main__":
    main()
