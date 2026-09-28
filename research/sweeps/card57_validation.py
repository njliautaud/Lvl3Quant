#!/usr/bin/env python3
"""Card 5-7 OOT Validation - flat npz + correct CLI flags."""
import sys, json, time, subprocess, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "card57_validation"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70
OOT_START = "2025-12-01"
OOT_END = "2026-03-08"

ALL_CONFIGS = [
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.1, None, None, 3600000, "c5_raw_e05_v0_s01", "card5"),
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.3, None, None, 3600000, "c5_raw_e05_v0_s03", "card5"),
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.5, None, None, 3600000, "c5_raw_e05_v0_s05", "card5"),
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.1, 20, 25, 3600000, "c6_raw_c015_v70_s01", "card6"),
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.3, 20, 25, 3600000, "c6_raw_c015_v70_s03", "card6"),
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.5, 20, 25, 3600000, "c6_raw_c015_v70_s05", "card6"),
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.1, None, 20, 3600000, "c7_smooth_c15_v70_s01", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.3, None, 20, 3600000, "c7_smooth_c15_v70_s03", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.5, None, 20, 3600000, "c7_smooth_c15_v70_s05", "card7"),
]

def get_pred_dates():
    dates = set()
    for f in PRED_DIR.iterdir():
        if f.suffix == ".npz" and len(f.name) >= 10:
            date = f.name[:10]
            if OOT_START <= date <= OOT_END:
                dates.add(date)
    return sorted(dates)

def find_pred_file(date, pred_pattern):
    for f in PRED_DIR.glob(f"{date}_{pred_pattern}*"):
        if f.suffix == ".npz":
            return f
    return None

def get_mbo_path(date):
    nodash = date.replace("-", "")
    return MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"

def run_one(date, pred_pattern, sig_thr, tp, sl, hold_ms, name, card_tag):
    pred_file = find_pred_file(date, pred_pattern)
    if not pred_file:
        return None
    mbo_path = get_mbo_path(date)
    if not mbo_path.exists():
        return None
    out_file = OUT_BASE / f"{name}_{date}.json"
    if out_file.exists():
        try:
            return json.loads(out_file.read_text())
        except Exception:
            pass
    cmd = [str(BINARY), "--mbo-file", str(mbo_path), "--predictions", str(pred_file),
           "--signal-threshold", str(sig_thr), "--hold-ms", str(hold_ms),
           "--output", str(out_file)]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl is not None:
        cmd += ["--stop-loss-ticks", str(sl)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out_file.exists():
            return json.loads(out_file.read_text())
    except Exception as e:
        print(f"  ERR {name} {date}: {e}", file=sys.stderr, flush=True)
    return None

def main():
    if not BINARY.exists():
        print(f"ERROR: binary not found: {BINARY}", flush=True)
        sys.exit(1)
    dates = get_pred_dates()
    if not dates:
        print("ERROR: no prediction dates found!", flush=True)
        sys.exit(1)
    total = len(ALL_CONFIGS) * len(dates)
    print(f"Found {len(dates)} OOT dates ({dates[0]} to {dates[-1]})", flush=True)
    print(f"{len(ALL_CONFIGS)} configs x {len(dates)} dates = {total} jobs, {WORKERS} workers", flush=True)

    results = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = []
        for cfg in ALL_CONFIGS:
            for date in dates:
                jobs.append((date,) + cfg)
        futs = {pool.submit(run_one, *j): j for j in jobs}
        for fut in as_completed(futs):
            done += 1
            j = futs[fut]
            r = fut.result()
            name = j[6]
            if r:
                if name not in results:
                    results[name] = {"config": name, "card": j[7], "days": []}
                results[name]["days"].append(r)
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed * 60
                eta = (total - done) / (done / elapsed) if done > 0 else 0
                print(f"  [{done}/{total}] {rate:.0f}/min ETA {eta/60:.1f}m", flush=True)

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"CARD 5-7 VALIDATION ({len(dates)} OOT days)", flush=True)
    print("=" * 70, flush=True)
    summary = []
    for name, data in sorted(results.items()):
        days = data["days"]
        pnls = [d.get("net_pnl", d.get("pnl_after_costs", 0)) for d in days]
        total_pnl = sum(pnls)
        trades = sum(d.get("total_trades", 0) for d in days)
        avg = statistics.mean(pnls) if pnls else 0
        std = statistics.stdev(pnls) if len(pnls) > 1 else 1
        sharpe = avg / std * (252 ** 0.5) if std > 0 else 0
        wr_vals = [d.get("win_rate", 0) for d in days if d.get("total_trades", 0) > 0]
        wr = statistics.mean(wr_vals) if wr_vals else 0
        entry = {"config": name, "card": data["card"], "sharpe": round(sharpe, 2),
                 "pnl": round(total_pnl, 2), "trades": trades, "wr": round(wr * 100, 1),
                 "days": len(days), "avg_daily": round(avg, 2)}
        summary.append(entry)
        print(f"  {name}: Sharpe={sharpe:.2f} PnL=${total_pnl:,.0f} Trades={trades} WR={wr*100:.1f}%", flush=True)
    Path(OUT_BASE / "card57_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved to {OUT_BASE}/card57_summary.json", flush=True)

if __name__ == "__main__":
    main()
