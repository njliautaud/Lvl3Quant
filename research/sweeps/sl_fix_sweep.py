#!/usr/bin/env python3
"""
Jupiter SL Fix Sweep v2 â€” fix payoff asymmetry in c1/c4 configs
================================================================
MC validation failed: avg_loss ($983) >> avg_win ($157).
Goal: tighter SL to fix payoff ratio. No --commission-ticks (not a valid arg).
Binary handles commission internally.
"""
import sys, json, time, subprocess, os, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

WORKERS = 20
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "sl_fix_sweep"
OUT_BASE.mkdir(parents=True, exist_ok=True)

LOG = LVL3_ROOT / "sl_fix_sweep.log"
SUMMARY_FILE = OUT_BASE / "sl_fix_summary.json"

OOT_START = "2025-12-01"
OOT_END   = "2026-03-08"

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(str(LOG), 'a') as f:
        f.write(line + "\n")

# Best configs from time_of_day_sweep â€” selective entry (3-7 trades/day)
# c1_tp13_s01 full_session: Sharpe=4.87, c4_tp20_s03 full_session: Sharpe=3.20
BASE_CONFIGS = [
    ("book_predstdExit_conv1.5_vol50", 0.1, "c1"),
    ("book_predstdExit_conv2.0_vol70", 0.3, "c4"),
]

# SL/TP sweep â€” fix payoff asymmetry
# Original: no SL â†’ avg_loss $983. Need SL to cap it.
TP_TICKS = [8, 12, 16, 20]
SL_TICKS = [4, 6, 8, 10, 12]
HOLD_TIMES_MS = [600000, 1800000, 3600000]  # 10min, 30min, 1hr

# Build date -> MBO map
mbo_map = {}
for f in sorted(MBO_DIR.glob("*.mbo.dbn.zst")):
    stem = f.stem.replace('.mbo.dbn', '')
    parts = stem.split('-')
    d = parts[-1] if parts else ''
    if len(d) == 8:
        date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if OOT_START <= date <= OOT_END:
            mbo_map[date] = str(f)

log(f"Found {len(mbo_map)} MBO dates")

# Build pred map
pred_map = defaultdict(dict)
for f in PRED_DIR.glob("*.npz"):
    parts = f.stem.split("_", 1)
    if len(parts) == 2:
        date, pred_type = parts
        pred_map[date][pred_type] = str(f)

log(f"Found {len(pred_map)} pred dates")

def run_sim(date, mbo_file, pred_file, tp, sl, hold_ms, sig, config_label):
    tag = f"{date}_{config_label}_tp{tp}_sl{sl}_h{hold_ms//60000}min"
    out_file = OUT_BASE / f"{tag}.json"
    if out_file.exists():
        return None, str(out_file)

    cmd = [
        str(BINARY),
        "--mbo-file", mbo_file,
        "--predictions", pred_file,
        "--output", str(out_file),
        "--signal-threshold", str(sig),
        "--hold-ms", str(hold_ms),
        "--take-profit-ticks", str(tp),
        "--stop-loss-ticks", str(sl),
        "--size", "1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and out_file.exists():
            return None, str(out_file)
        return f"Exit {r.returncode}: {r.stderr[:200]}", None
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None
    except Exception as e:
        return str(e), None

# Build tasks
tasks = []
for pred_type, sig, name in BASE_CONFIGS:
    for tp in TP_TICKS:
        for sl in SL_TICKS:
            if sl >= tp:
                continue
            for hold_ms in HOLD_TIMES_MS:
                for date, mbo_file in mbo_map.items():
                    if pred_type in pred_map.get(date, {}):
                        tasks.append((date, mbo_file, pred_map[date][pred_type],
                                      tp, sl, hold_ms, sig, name))

log(f"Total tasks: {len(tasks)}")

errors = 0
completed = 0
start_t = time.time()

def do_task(t):
    return run_sim(*t)

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(do_task, t): t for t in tasks}
    for future in as_completed(futures):
        err, out = future.result()
        completed += 1
        if err:
            errors += 1
        if completed % 500 == 0:
            elapsed = time.time() - start_t
            rate = completed / max(elapsed, 0.01)
            eta = (len(tasks) - completed) / rate
            log(f"Progress: {completed}/{len(tasks)} ({errors} errors, {errors/completed*100:.0f}%) ETA: {eta:.0f}s")

log(f"Done. {completed} tasks, {errors} errors ({errors/max(completed,1)*100:.0f}%)")

# === AGGREGATE ===
log("Aggregating results...")
config_days = defaultdict(list)

for result_file in OUT_BASE.glob("*.json"):
    if result_file.name == "sl_fix_summary.json":
        continue
    try:
        with open(result_file) as f:
            d = json.load(f)
        stem = result_file.stem
        parts = stem.split("_")
        date = parts[0]
        config_key = "_".join(parts[1:])
        config_days[config_key].append({
            "date": date,
            "pnl": d.get("total_pnl_dollars", 0) or 0,
            "trades": d.get("total_trades", 0) or 0,
            "wr": d.get("win_rate", 0) or 0,
            "avg_win": d.get("avg_win", 0) or 0,
            "avg_loss": d.get("avg_loss", 0) or 0,
        })
    except Exception:
        continue

agg = []
for ck, days in config_days.items():
    if len(days) < 10:
        continue
    pnls = [d["pnl"] for d in days]
    n = len(pnls)
    mean_p = sum(pnls) / n
    std_p = statistics.stdev(pnls) if n > 1 else 1e-8
    neg = [p for p in pnls if p < 0]
    ds = statistics.stdev(neg) if len(neg) > 1 else 1e-8
    sortino = mean_p / (ds + 1e-8)
    pct_pos = sum(1 for p in pnls if p > 0) / n
    wins = [d["avg_win"] for d in days if d["avg_win"] and d["avg_win"] > 0]
    losses = [d["avg_loss"] for d in days if d["avg_loss"] and d["avg_loss"] < 0]
    mean_win = sum(wins)/len(wins) if wins else 0
    mean_loss = sum(losses)/len(losses) if losses else 0
    payoff = abs(mean_win/mean_loss) if mean_loss != 0 else 0
    wrs = [d["wr"] for d in days if d["wr"] is not None]

    agg.append({
        "config_key": ck, "n_days": n,
        "total_pnl": round(sum(pnls), 2),
        "mean_daily_pnl": round(mean_p, 2),
        "sortino": round(sortino, 4),
        "pct_pos_days": round(pct_pos, 3),
        "mean_wr": round(sum(wrs)/len(wrs), 4) if wrs else 0,
        "payoff_ratio": round(payoff, 4),
        "mean_avg_win": round(mean_win, 2),
        "mean_avg_loss": round(mean_loss, 2),
    })

agg.sort(key=lambda x: x["sortino"], reverse=True)
positive = [r for r in agg if r["sortino"] > 0 and r["total_pnl"] > 0]

log(f"Aggregated {len(agg)} configs, {len(positive)} positive")
log("Top 10:")
for r in agg[:10]:
    log(f"  {r['config_key']}: Sortino={r['sortino']:.3f} PnL=${r['total_pnl']:.0f} "
        f"WR={r['mean_wr']:.1%} Payoff={r['payoff_ratio']:.2f} {r['n_days']}d {r['pct_pos_days']*100:.0f}%pos")

summary = {
    "timestamp": datetime.now().isoformat(),
    "n_configs": len(agg), "n_positive": len(positive),
    "best_sortino": agg[0]["sortino"] if agg else 0,
    "best_config": agg[0]["config_key"] if agg else "",
    "top_20": agg[:20],
    "elapsed_sec": round(time.time() - start_t, 1),
}
with open(str(SUMMARY_FILE), "w") as f:
    json.dump(summary, f, indent=2)

log(f"Saved to {SUMMARY_FILE}")
log(f"VERDICT: {'PROMISING' if positive else 'NEEDS_WIDER_SEARCH'}")
