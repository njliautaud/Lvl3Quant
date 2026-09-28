#!/usr/bin/env python3
"""Data-driven MAE exit + breakeven lock test for Cards 1,2,4.
Tests MAE thresholds derived from actual winner/loser MAE distributions."""
import json, subprocess, statistics, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BINARY = Path("/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli")
MBO = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUT = Path("/home/jupiter/Lvl3Quant/data/processed/mae_threshold_test")
OUT.mkdir(parents=True, exist_ok=True)

CARDS = [
    ("card1", "book_predstdExit_conv1.5_vol50", 0.1, 8),
    ("card2", "book_predstdExit_conv1.5_vol50", 0.5, 15),
    ("card4", "book_predstdExit_conv2.0_vol70", 0.5, 20),
]

# Data-driven configs to test
MODES = [
    # (name, mae_ticks, mae_hold_sec, ratchet, hold_ms)
    ("baseline_30m", 0, 0, False, 1800000),
    ("baseline_2h", 0, 0, False, 7200000),
    ("mae35_10m", 35, 600, False, 7200000),
    ("mae40_10m", 40, 600, False, 7200000),
    ("mae50_15m", 50, 900, False, 7200000),
    ("mae30_5m", 30, 300, False, 7200000),
    ("mae25_10m", 25, 600, False, 7200000),
    ("breakeven_only", 0, 0, True, 7200000),  # ratchet at 3t = breakeven
    ("mae35_10m_breakeven", 35, 600, True, 7200000),
    ("mae40_10m_breakeven", 40, 600, True, 7200000),
]

def get_dates():
    dates = set()
    for f in PRED.iterdir():
        if f.suffix == ".npz" and len(f.name) >= 10:
            dates.add(f.name[:10])
    return sorted(d for d in dates if "2025-12-01" <= d <= "2026-03-08")

def run_one(date, card, pred_type, sig, tp, mode_name, mae_t, mae_s, ratchet, hold_ms):
    pred = list(PRED.glob(f"{date}_{pred_type}*"))
    if not pred: return None
    mbo = MBO / f"glbx-mdp3-{date.replace('-','')}.mbo.dbn.zst"
    if not mbo.exists(): return None
    out_file = OUT / f"{card}_{mode_name}_{date}.json"
    if out_file.exists():
        try: return json.loads(out_file.read_text())
        except: pass
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred[0]),
           "--signal-threshold", str(sig), "--take-profit-ticks", str(tp),
           "--hold-ms", str(hold_ms), "--output", str(out_file)]
    if mae_t > 0:
        cmd += ["--mae-exit-ticks", str(mae_t), "--mae-exit-hold-sec", str(mae_s)]
    if ratchet:
        cmd += ["--ratchet-stop"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out_file.exists():
            return json.loads(out_file.read_text())
    except: pass
    return None

dates = get_dates()
print(f"Testing {len(CARDS)} cards x {len(MODES)} modes x {len(dates)} dates = {len(CARDS)*len(MODES)*len(dates)} jobs", flush=True)

results = {}
done = 0
total = len(CARDS) * len(MODES) * len(dates)
import time
t0 = time.time()

with ThreadPoolExecutor(max_workers=10) as pool:
    futs = {}
    for card, pred_type, sig, tp in CARDS:
        for mode_name, mae_t, mae_s, ratchet, hold_ms in MODES:
            for date in dates:
                key = f"{card}_{mode_name}"
                f = pool.submit(run_one, date, card, pred_type, sig, tp, mode_name, mae_t, mae_s, ratchet, hold_ms)
                futs[f] = key
    for f in as_completed(futs):
        done += 1
        key = futs[f]
        r = f.result()
        if r:
            if key not in results: results[key] = []
            results[key].append(r)
        if done % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{done}/{total}] {done/elapsed*60:.0f}/min", flush=True)

print(f"\n{'='*90}", flush=True)
print(f"MAE THRESHOLD TEST RESULTS ({len(dates)} OOT days)", flush=True)
print(f"{'='*90}", flush=True)
print(f"{'Config':<35} {'Sharpe':>7} {'PnL':>10} {'Trades':>7} {'WR':>6} {'AvgLoss':>8} {'MaxDD':>8} {'Timeouts':>8}", flush=True)
print("-"*90, flush=True)

summary = []
for card, _, _, _ in CARDS:
    for mode_name, _, _, _, _ in MODES:
        key = f"{card}_{mode_name}"
        days = results.get(key, [])
        if not days: continue
        pnls = [d.get("total_pnl_dollars", 0) for d in days]
        trades = sum(d.get("total_trades", 0) for d in days)
        avg = statistics.mean(pnls) if pnls else 0
        std = statistics.stdev(pnls) if len(pnls) > 1 else 1
        sharpe = avg / std * (252**0.5) if std > 0 else 0
        wrs = [d.get("win_rate", 0) for d in days if d.get("total_trades", 0) > 0]
        wr = statistics.mean(wrs) * 100 if wrs else 0
        # Get avg loss and timeout count from trade details
        losses = []
        timeouts = 0
        for d in days:
            for t in d.get("trades", []):
                if t.get("pnl_ticks", 0) <= 0:
                    losses.append(t.get("pnl_ticks", 0) * 12.50)
                if t.get("exit_reason") in ("HoldTimeout", "HOLD_TIMEOUT"):
                    timeouts += 1
        avg_loss = statistics.mean(losses) if losses else 0
        max_dd = max((max(0, sum(pnls[:i]) - sum(pnls[:j])) for i in range(len(pnls)) for j in range(i)), default=0) if pnls else 0
        
        entry = {"config": key, "sharpe": round(sharpe,2), "pnl": round(sum(pnls),2),
                 "trades": trades, "wr": round(wr,1), "avg_loss": round(avg_loss,2),
                 "timeouts": timeouts}
        summary.append(entry)
        print(f"{key:<35} {sharpe:>7.2f} ${sum(pnls):>9,.0f} {trades:>7} {wr:>5.1f}% ${avg_loss:>7,.0f} ${max_dd:>7,.0f} {timeouts:>8}", flush=True)
    print(flush=True)

Path(OUT / "mae_threshold_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSaved to {OUT}/mae_threshold_summary.json", flush=True)

