#!/usr/bin/env python3
"""Card 3 Extended Parameter Sweep.
Uses raw_rawExit_conv0.05_ethr0.0_vol70 (best Card 3 pred type).
Also tests vol0/vol50 variants to find if different vol filters help.
Best from card3_new: TP15_trail25 (Sharpe 3.1), TP15_slN (Sharpe 2.78).
Now exploring: more TP levels, signal thresholds, SL variants, other vol filters.
~2900 total jobs, 14 workers, ~35-50 min.
"""
import json, os, subprocess, time, sys, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "card3_extended"
WORKERS = 14
TICK_VALUE = 12.50
COMMISSION_RT = 4.70


def find_dates(pred_suffix):
    preds = {}
    mbo_dates = set()
    for f in sorted(PRED_DIR.glob("*_" + pred_suffix + ".npz")):
        date = f.stem[:10]
        preds[date] = f
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        nodash = f.name.split("-")[2].split(".")[0]
        date = nodash[:4] + "-" + nodash[4:6] + "-" + nodash[6:8]
        mbo_dates.add(date)
    return sorted(set(preds.keys()) & mbo_dates), preds


def get_mbo(date):
    return MBO_DIR / ("glbx-mdp3-" + date.replace("-", "") + ".mbo.dbn.zst")


def run_sim(cmd, out_path, label):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out_path).exists():
            return (label, True, None)
        return (label, False, r.stderr[:200] if r.stderr else "rc=" + str(r.returncode))
    except subprocess.TimeoutExpired:
        return (label, False, "TIMEOUT")
    except Exception as e:
        return (label, False, str(e)[:200])


# CONFIGS for vol70 (the proven winner) - exploring around best results
# Format: (name, tp, trail, sl, sig)
CONFIGS_VOL70 = [
    # TP variations around TP15 (baseline)
    ("tp10_slN_sig0.1",      10, None, None, 0.1),
    ("tp12_slN_sig0.1",      12, None, None, 0.1),
    ("tp15_slN_sig0.1",      15, None, None, 0.1),
    ("tp18_slN_sig0.1",      18, None, None, 0.1),
    ("tp20_slN_sig0.1",      20, None, None, 0.1),
    ("tp25_slN_sig0.1",      25, None, None, 0.1),
    ("tp30_slN_sig0.1",      30, None, None, 0.1),
    # Trail variations around best (trail25 won)
    ("tp15_trail15_sig0.1",  15,  15, None, 0.1),
    ("tp15_trail20_sig0.1",  15,  20, None, 0.1),
    ("tp15_trail25_sig0.1",  15,  25, None, 0.1),
    ("tp15_trail30_sig0.1",  15,  30, None, 0.1),
    ("tp15_trail35_sig0.1",  15,  35, None, 0.1),
    ("tp20_trail25_sig0.1",  20,  25, None, 0.1),
    ("tp20_trail20_sig0.1",  20,  20, None, 0.1),
    ("tp25_trail25_sig0.1",  25,  25, None, 0.1),
    ("tp25_trail20_sig0.1",  25,  20, None, 0.1),
    # Signal threshold sweep (raw preds may be sensitive to threshold)
    ("tp15_slN_sig0.05",     15, None, None, 0.05),
    ("tp15_slN_sig0.15",     15, None, None, 0.15),
    ("tp15_slN_sig0.2",      15, None, None, 0.2),
    ("tp15_slN_sig0.3",      15, None, None, 0.3),
    ("tp15_slN_sig0.5",      15, None, None, 0.5),
    ("tp15_trail25_sig0.05", 15,  25, None, 0.05),
    ("tp15_trail25_sig0.15", 15,  25, None, 0.15),
    ("tp15_trail25_sig0.2",  15,  25, None, 0.2),
    ("tp15_trail25_sig0.3",  15,  25, None, 0.3),
    # Hard stop loss variants
    ("tp15_sl20_sig0.1",     15, None,  20, 0.1),
    ("tp15_sl30_sig0.1",     15, None,  30, 0.1),
    ("tp15_sl50_sig0.1",     15, None,  50, 0.1),
    ("tp20_sl30_sig0.1",     20, None,  30, 0.1),
    ("tp20_sl50_sig0.1",     20, None,  50, 0.1),
    # Best combos: high TP + trail + varied threshold
    ("tp20_trail25_sig0.15", 20,  25, None, 0.15),
    ("tp20_trail25_sig0.2",  20,  25, None, 0.2),
    ("tp25_trail25_sig0.15", 25,  25, None, 0.15),
    ("tp30_trail25_sig0.1",  30,  25, None, 0.1),
    ("tp30_trail30_sig0.1",  30,  30, None, 0.1),
]

# CONFIGS for vol0 variant (denser signals) - quick scan
CONFIGS_VOL0 = [
    ("tp15_slN_sig0.1",      15, None, None, 0.1),
    ("tp15_slN_sig0.3",      15, None, None, 0.3),
    ("tp15_slN_sig0.5",      15, None, None, 0.5),
    ("tp15_trail25_sig0.1",  15,  25, None, 0.1),
    ("tp15_trail25_sig0.3",  15,  25, None, 0.3),
    ("tp15_trail25_sig0.5",  15,  25, None, 0.5),
    ("tp20_slN_sig0.3",      20, None, None, 0.3),
    ("tp20_trail25_sig0.3",  20,  25, None, 0.3),
    ("tp10_slN_sig0.3",      10, None, None, 0.3),
    ("tp10_trail25_sig0.3",  10,  25, None, 0.3),
]

# CONFIGS for vol50 variant
CONFIGS_VOL50 = [
    ("tp15_slN_sig0.1",      15, None, None, 0.1),
    ("tp15_slN_sig0.3",      15, None, None, 0.3),
    ("tp15_trail25_sig0.1",  15,  25, None, 0.1),
    ("tp15_trail25_sig0.3",  15,  25, None, 0.3),
    ("tp20_slN_sig0.1",      20, None, None, 0.1),
    ("tp20_trail25_sig0.1",  20,  25, None, 0.1),
    ("tp10_slN_sig0.1",      10, None, None, 0.1),
    ("tp10_trail25_sig0.1",  10,  25, None, 0.1),
]


def build_cmd(date, pred_path, out_path, tp, trail, sl, sig):
    cmd = [
        str(BINARY),
        "--mbo-file", str(get_mbo(date)),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", str(sig),
        "--hold-ms", "3600000",
        "--max-wait-bars", "50",
        "--latency-ms", "50",
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--quiet",
    ]
    if tp is not None:
        cmd.extend(["--take-profit-ticks", str(tp)])
    if trail is not None:
        cmd.extend(["--trailing-ticks", str(trail)])
    if sl is not None:
        cmd.extend(["--stop-loss-ticks", str(sl)])
    return cmd


def aggregate_config(cfg_dir, dates):
    daily_pnls = []
    total_trades = 0
    total_wins = 0
    gross_win = 0.0
    gross_loss = 0.0

    for d in dates:
        p = cfg_dir / (d + ".json")
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        trades = data.get("trades", [])
        day_pnl = 0.0
        for t in trades:
            pnl = t.get("pnl_ticks", 0)
            day_pnl += pnl
            total_trades += 1
            if pnl > 0:
                total_wins += 1
                gross_win += pnl
            else:
                gross_loss += abs(pnl)
        if not trades:
            day_pnl = data.get("total_pnl_ticks", 0)
            n_t = data.get("total_trades", 0)
            total_trades += n_t
        daily_pnls.append(day_pnl)

    if not daily_pnls:
        return None
    n = len(daily_pnls)
    mean_d = sum(daily_pnls) / n
    var_d = sum((x - mean_d) ** 2 for x in daily_pnls) / max(n - 1, 1)
    std_d = var_d ** 0.5
    sharpe = (mean_d / std_d) * (252 ** 0.5) if std_d > 0 else 0.0
    wr = total_wins / max(total_trades, 1) * 100
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    total_pnl = sum(daily_pnls)
    net_usd = total_pnl * TICK_VALUE - total_trades * COMMISSION_RT
    pos_days = sum(1 for p in daily_pnls if p > 0)
    return {
        "n_days": n,
        "total_trades": total_trades,
        "total_pnl_ticks": round(total_pnl, 2),
        "net_usd": round(net_usd, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "pos_days": pos_days,
        "pos_day_pct": round(pos_days / n * 100, 1),
        "avg_daily_ticks": round(mean_d, 2),
        "std_daily_ticks": round(std_d, 2),
        "trades_per_day": round(total_trades / n, 1),
    }


def main():
    print("[" + datetime.now().strftime("%H:%M:%S") + "] Card 3 Extended Sweep Starting", flush=True)

    if not BINARY.exists():
        print("ERROR: Binary not found: " + str(BINARY), flush=True)
        sys.exit(1)

    dates_v70, preds_v70 = find_dates("raw_rawExit_conv0.05_ethr0.0_vol70")
    dates_v0,  preds_v0  = find_dates("raw_rawExit_conv0.05_ethr0.0_vol0")
    dates_v50, preds_v50 = find_dates("raw_rawExit_conv0.05_ethr0.0_vol50")

    print("vol70 dates: " + str(len(dates_v70)) +
          ", vol0 dates: " + str(len(dates_v0)) +
          ", vol50 dates: " + str(len(dates_v50)), flush=True)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = 0

    for cfg_name, tp, trail, sl, sig in CONFIGS_VOL70:
        cfg_dir = OUT_BASE / "vol70" / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for d in dates_v70:
            out = cfg_dir / (d + ".json")
            if out.exists():
                skipped += 1
                continue
            cmd = build_cmd(d, preds_v70[d], out, tp, trail, sl, sig)
            jobs.append((cmd, str(out), "v70/" + cfg_name + "/" + d))

    for cfg_name, tp, trail, sl, sig in CONFIGS_VOL0:
        cfg_dir = OUT_BASE / "vol0" / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for d in dates_v0:
            out = cfg_dir / (d + ".json")
            if out.exists():
                skipped += 1
                continue
            cmd = build_cmd(d, preds_v0[d], out, tp, trail, sl, sig)
            jobs.append((cmd, str(out), "v0/" + cfg_name + "/" + d))

    for cfg_name, tp, trail, sl, sig in CONFIGS_VOL50:
        cfg_dir = OUT_BASE / "vol50" / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for d in dates_v50:
            out = cfg_dir / (d + ".json")
            if out.exists():
                skipped += 1
                continue
            cmd = build_cmd(d, preds_v50[d], out, tp, trail, sl, sig)
            jobs.append((cmd, str(out), "v50/" + cfg_name + "/" + d))

    total_configs = len(CONFIGS_VOL70) + len(CONFIGS_VOL0) + len(CONFIGS_VOL50)
    print("Total configs: " + str(total_configs) +
          " | Total jobs: " + str(len(jobs)) + " (skipped " + str(skipped) + " existing)", flush=True)

    if jobs:
        done = 0
        failed = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_sim, cmd, out, lbl): lbl for cmd, out, lbl in jobs}
            for f in as_completed(futures):
                label, success, err = f.result()
                done += 1
                if not success:
                    failed += 1
                    print("  FAIL [" + str(done) + "/" + str(len(jobs)) + "] " + label + ": " + str(err), flush=True)
                elif done % 100 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print("  [" + str(done) + "/" + str(len(jobs)) + "] " + label + " OK (" + str(int(elapsed)) + "s, ETA " + str(int(eta)) + "s)", flush=True)

        print("\nSweep done: " + str(done - failed) + "/" + str(len(jobs)) + " in " + str(int(time.time() - t0)) + "s", flush=True)
    else:
        print("All jobs already done!", flush=True)

    # ANALYSIS
    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY (Card 3 Extended Sweep)", flush=True)
    print("=" * 70, flush=True)

    all_results = []

    for vol, pred_suffix, configs, dates in [
        ("vol70", "raw_rawExit_conv0.05_ethr0.0_vol70", CONFIGS_VOL70, dates_v70),
        ("vol0",  "raw_rawExit_conv0.05_ethr0.0_vol0",  CONFIGS_VOL0,  dates_v0),
        ("vol50", "raw_rawExit_conv0.05_ethr0.0_vol50", CONFIGS_VOL50, dates_v50),
    ]:
        print("\n--- " + pred_suffix + " (" + str(len(dates)) + " days) ---", flush=True)
        print("  " + "Config".ljust(28) + "Sharpe".rjust(7) + "Net$".rjust(10) + "WR%".rjust(6) + "PF".rjust(6) + "T/Day".rjust(6) + "POS%".rjust(6), flush=True)
        print("  " + "-" * 28 + "-" * 7 + "-" * 10 + "-" * 6 + "-" * 6 + "-" * 6 + "-" * 6, flush=True)

        vol_results = []
        for cfg_name, tp, trail, sl, sig in configs:
            cfg_dir = OUT_BASE / vol / cfg_name
            m = aggregate_config(cfg_dir, dates)
            if m:
                all_results.append({"pred": pred_suffix, "vol": vol, "config": cfg_name, **m})
                vol_results.append((cfg_name, m))
                flag = " **BEST**" if m["sharpe"] > 4.0 else (" *" if m["sharpe"] > 3.0 else "")
                print("  " + cfg_name.ljust(28) + str(round(m["sharpe"], 2)).rjust(7) + ("$" + str(int(m["net_usd"]))).rjust(10) + str(round(m["win_rate"], 1)).rjust(6) + str(round(m["profit_factor"], 2)).rjust(6) + str(round(m["trades_per_day"], 1)).rjust(6) + str(round(m["pos_day_pct"], 1)).rjust(6) + flag, flush=True)

        if vol_results:
            best = sorted(vol_results, key=lambda x: x[1]["sharpe"], reverse=True)[:5]
            print("\n  TOP 5 by Sharpe:", flush=True)
            for name, m in best:
                print("    " + name + ": Sharpe=" + str(m["sharpe"]) + ", Net=$" + str(m["net_usd"]) + ", WR=" + str(m["win_rate"]) + "%, PF=" + str(m["profit_factor"]), flush=True)

    out_path = OUT_BASE / "extended_summary.json"
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print("\nResults saved to: " + str(out_path), flush=True)
    print("Completion: " + datetime.now().strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
