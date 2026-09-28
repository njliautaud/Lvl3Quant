#!/usr/bin/env python3
"""Card 1/3/4 Fix Sweep - OOT Validation.

CARD 1/4 FIX:
- Problem: book_predstdExit_conv2.5_vol70 has near-zero signal
- Fix: Test lower conviction thresholds (conv1.5, conv2.0) AND ema_bookExit_conv2.5

CARD 3 FIX:
- Problem: raw_rawExit IS Sharpe 3.03 but OOT Sharpe -0.45 (overfitting)
- Fix: Test with more robust exit params (higher sig threshold, simpler configs)
"""
import sys, json, time, subprocess, os, math, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "card_fix_sweep"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

# Format: (pred_type, sig_thr, tp, trail, sl, hold_ms, name)
CARD14_CONFIGS = [
    ("book_predstdExit_conv1.5_vol70", 0.1, 8,  None, None, 3600000, "book_c15_v70_tp8_slN"),
    ("book_predstdExit_conv1.5_vol70", 0.1, 15, None, None, 3600000, "book_c15_v70_tp15_slN"),
    ("book_predstdExit_conv1.5_vol70", 0.1, 30, None, None, 3600000, "book_c15_v70_tp30_slN"),
    ("book_predstdExit_conv1.5_vol70", 0.1, 8,  None, 10,  3600000, "book_c15_v70_tp8_sl10"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 8,  None, None, 3600000, "book_c20_v70_tp8_slN"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 15, None, None, 3600000, "book_c20_v70_tp15_slN"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 30, None, None, 3600000, "book_c20_v70_tp30_slN"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 8,  None, 10,  3600000, "book_c20_v70_tp8_sl10"),
    ("book_predstdExit_conv2.0_vol50", 0.1, 8,  None, None, 3600000, "book_c20_v50_tp8_slN"),
    ("book_predstdExit_conv2.0_vol50", 0.1, 15, None, None, 3600000, "book_c20_v50_tp15_slN"),
    ("ema_bookExit_conv2.5_vol70",     0.1, 8,  None, None, 3600000, "ema_c25_v70_tp8_slN"),
    ("ema_bookExit_conv2.5_vol70",     0.1, 15, None, None, 3600000, "ema_c25_v70_tp15_slN"),
    ("ema_bookExit_conv2.5_vol70",     0.1, 30, None, None, 3600000, "ema_c25_v70_tp30_slN"),
    ("ema_bookExit_conv2.5_vol70",     0.1, 8,  None, 10,  3600000, "ema_c25_v70_tp8_sl10"),
    ("book_predstdExit_conv1.5_vol50", 0.1, 8,  None, None, 3600000, "book_c15_v50_tp8_slN"),
    ("book_predstdExit_conv1.5_vol50", 0.1, 30, None, None, 3600000, "book_c15_v50_tp30_slN"),
]

CARD3_CONFIGS = [
    ("raw_rawExit_conv0.05_ethr0.0_vol50",    0.5, 15, None, None, 3600000, "raw_v50_s05_tp15_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol50",    0.3, 15, None, None, 3600000, "raw_v50_s03_tp15_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol50",    0.1, 15, None, None, 3600000, "raw_v50_s01_tp15_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol50",    0.5, 10, None, None, 3600000, "raw_v50_s05_tp10_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol50",    0.3, 10, None, None, 3600000, "raw_v50_s03_tp10_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol50",    0.5, 8,  None, None, 3600000, "raw_v50_s05_tp8_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol70",    0.5, 15, None, None, 3600000, "raw_v70_s05_tp15_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol70",    0.3, 15, None, None, 3600000, "raw_v70_s03_tp15_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol70",    0.5, 10, None, None, 3600000, "raw_v70_s05_tp10_slN"),
    ("raw_rawExit_conv0.05_ethr0.0_vol70",    0.3, 10, None, None, 3600000, "raw_v70_s03_tp10_slN"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol50", 0.5, 15, None, None, 3600000, "rsmooth_v50_s05_tp15_slN"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol50", 0.3, 15, None, None, 3600000, "rsmooth_v50_s03_tp15_slN"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol70", 0.5, 15, None, None, 3600000, "rsmooth_v70_s05_tp15_slN"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol70", 0.3, 15, None, None, 3600000, "rsmooth_v70_s03_tp15_slN"),
    ("mom_emaExit_conv0.3_ethr0.0_vol70",     0.5, 15, None, None, 3600000, "mom_v70_s05_tp15_slN"),
    ("mom_emaExit_conv0.3_ethr0.0_vol70",     0.3, 15, None, None, 3600000, "mom_v70_s03_tp15_slN"),
    ("mom_emaExit_conv0.3_ethr0.0_vol50",     0.5, 15, None, None, 3600000, "mom_v50_s05_tp15_slN"),
    ("mom_emaExit_conv0.3_ethr0.0_vol50",     0.3, 15, None, None, 3600000, "mom_v50_s03_tp15_slN"),
]

ALL_CONFIGS = [("card14_fix", c) for c in CARD14_CONFIGS] + [("card3_fix", c) for c in CARD3_CONFIGS]


def get_mbo_path(date):
    nodash = date.replace("-", "")
    return MBO_DIR / "glbx-mdp3-{}.mbo.dbn.zst".format(nodash)


def find_dates_for_pred(pred_type):
    dates = {}
    for f in sorted(PRED_DIR.glob("*_{}.npz".format(pred_type))):
        date = f.stem[:10]
        mbo = get_mbo_path(date)
        if mbo.exists():
            dates[date] = f
    return dates


def run_sim(cmd, out_path, label):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and Path(out_path).exists():
            return (label, True, None)
        return (label, False, r.stderr[:200] if r.stderr else "rc={}".format(r.returncode))
    except subprocess.TimeoutExpired:
        return (label, False, "TIMEOUT")
    except Exception as e:
        return (label, False, str(e)[:200])


def build_cmd(date, pred_path, out_path, sig_thr, tp, trail, sl, hold_ms):
    cmd = [
        str(BINARY),
        "--mbo-file", str(get_mbo_path(date)),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", str(sig_thr),
        "--hold-ms", str(hold_ms),
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


def get_month(date_str):
    m = int(date_str[5:7])
    return {12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar"}.get(m, "M{}".format(m))


def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def analyze_config(config_name, results_by_date):
    all_daily_pnls = []
    total_trades = 0
    total_wins = 0
    all_trade_pnls = []
    regime = {}
    for date in sorted(results_by_date.keys()):
        data = results_by_date[date]
        if data is None:
            continue
        trades = data.get("trades", [])
        day_pnl = sum(t.get("pnl_ticks", 0) for t in trades)
        day_wins = sum(1 for t in trades if t.get("pnl_ticks", 0) > 0)
        all_daily_pnls.append(day_pnl)
        total_trades += len(trades)
        total_wins += day_wins
        all_trade_pnls.extend(t.get("pnl_ticks", 0) for t in trades)
        m = get_month(date)
        if m not in regime:
            regime[m] = {"pnl": 0, "trades": 0, "days": 0}
        regime[m]["pnl"] += day_pnl
        regime[m]["trades"] += len(trades)
        regime[m]["days"] += 1
    n = len(all_daily_pnls)
    if n < 2 or total_trades == 0:
        return None
    mean_d = sum(all_daily_pnls) / n
    std_d = (sum((x - mean_d)**2 for x in all_daily_pnls) / (n - 1))**0.5
    sharpe = (mean_d / std_d * (252**0.5)) if std_d > 0 else 0
    gross_p = sum(p for p in all_trade_pnls if p > 0)
    gross_l = abs(sum(p for p in all_trade_pnls if p < 0))
    pf = gross_p / gross_l if gross_l > 0 else 0
    wr = 100.0 * total_wins / total_trades
    net_pnl = sum(all_daily_pnls) * TICK_VALUE - total_trades * COMMISSION_RT
    pos_days = sum(1 for p in all_daily_pnls if p > 0)
    return {
        "config": config_name, "n_days": n, "trades": total_trades,
        "sharpe": round(sharpe, 2), "wr": round(wr, 1), "pf": round(pf, 2),
        "net_pnl": round(net_pnl, 0), "pos_days": pos_days, "regime": regime,
    }


def main():
    print("[{}] Card Fix Sweep Starting".format(datetime.now()), flush=True)
    if not BINARY.exists():
        print("ERROR: binary not found: {}".format(BINARY), flush=True)
        sys.exit(1)

    jobs = []
    skipped = 0
    pred_cache = {}

    for card_tag, (pred_type, sig_thr, tp, trail, sl, hold_ms, name) in ALL_CONFIGS:
        if pred_type not in pred_cache:
            pred_cache[pred_type] = find_dates_for_pred(pred_type)
        dates = pred_cache[pred_type]
        if not dates:
            print("WARNING: No dates for: {}".format(pred_type), flush=True)
            continue
        out_dir = OUT_BASE / card_tag / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for date, pred_path in sorted(dates.items()):
            out_path = out_dir / "{}.json".format(date)
            if out_path.exists():
                skipped += 1
                continue
            cmd = build_cmd(date, pred_path, out_path, sig_thr, tp, trail, sl, hold_ms)
            jobs.append((cmd, str(out_path), "{}/{}".format(name, date)))

    print("Total jobs: {} (skipped {})".format(len(jobs), skipped), flush=True)

    if jobs:
        done = 0
        failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_sim, cmd, out, label): label for cmd, out, label in jobs}
            for f in as_completed(futures):
                label, ok, err_msg = f.result()
                done += 1
                if not ok:
                    failed += 1
                    print("  FAIL [{}/{}] {}: {}".format(done, len(jobs), label, err_msg), flush=True)
                elif done % 100 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 1
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print("  [{}/{}] {:.1f}/s ETA {:.0f}s".format(done, len(jobs), rate, eta), flush=True)
        print("Sweep done: {}/{} ok, {} failed in {:.0f}s".format(
            done - failed, len(jobs), failed, time.time() - t0), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    for card_tag, (pred_type, sig_thr, tp, trail, sl, hold_ms, name) in ALL_CONFIGS:
        out_dir = OUT_BASE / card_tag / name
        dates = pred_cache.get(pred_type, {})
        results_by_date = {}
        for date in sorted(dates.keys()):
            out_path = out_dir / "{}.json".format(date)
            results_by_date[date] = load_result(out_path)
        a = analyze_config(name, results_by_date)
        if a:
            all_results.append((card_tag, a))

    hdr = "{:<3} {:<32} {:>6} {:>6} {:>5} {:>5} {:>8} {:>7}"
    print(hdr.format("TAG", "Config", "Trades", "Sharpe", "WR%", "PF", "Net$", "PosD"), flush=True)
    print("-" * 75, flush=True)
    row = "{:<3} {:<32} {:>6} {:>6.2f} {:>5.1f} {:>5.2f} {:>8,.0f} {:>4}/{:<2}"
    for card_tag, a in sorted(all_results, key=lambda x: -x[1]["sharpe"]):
        tag = "C14" if card_tag == "card14_fix" else "C3 "
        print(row.format(tag, a["config"][:32], a["trades"], a["sharpe"], a["wr"],
                         a["pf"], a["net_pnl"], a["pos_days"], a["n_days"]), flush=True)
        for m in ["Dec", "Jan", "Feb"]:
            if m in a["regime"]:
                r = a["regime"][m]
                avg = r["pnl"] / r["days"] if r["days"] > 0 else 0
                print("     {} {:>3}d {:>4}t avg {:.1f}tk/d".format(
                    m, r["days"], r["trades"], avg), flush=True)

    summary_path = OUT_BASE / "fix_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump([{"tag": t, "analysis": a} for t, a in all_results], f, indent=2)
    print("\nSaved: {}".format(summary_path), flush=True)
    print("SWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
