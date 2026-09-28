#!/usr/bin/env python3
"""Cards 5-7 OOT Validation - FULL 68-day period (Dec 1, 2025 - Mar 6, 2026).

Card 5: raw_rawExit_conv0.05_ethr0.5_vol0  (avg win > avg loss, WR ~59%, no TP/SL)
Card 6: raw_rawExit_conv0.15_vol70          (coin-flip WR, W/L 1.16x, TP20/SL25)
Card 7: smooth_smoothExit_conv1.5_vol70     (pure trend-follower, WR ~48%, no TP/SL20)

Uses fill_sim_cli binary, same pipeline as Cards 1-4.
Skips already-done dates. Extended OOT window to Mar 6, 2026.
"""
import sys, json, time, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "card_oot_validation"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

# FULL 68-day OOT period
OOT_START = "2025-12-01"
OOT_END   = "2026-03-06"

# Card 5: raw_rawExit_conv0.05_ethr0.5_vol0, no TP/SL
# Card 6: raw_rawExit_conv0.15_vol70, TP20/SL25  (sweep both ethr 0.0 and 0.5)
# Card 7: smooth_smoothExit_conv1.5_vol70, no TP/SL20 (sweep both ethr 0.0 and 0.5)
# sig thresholds: [0.1, 0.3, 0.5, 0.7]

ALL_CONFIGS = [
    # --- Card 5: raw_rawExit_conv0.05_ethr0.5_vol0, no TP/SL ---
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.1, None, None, None, 3600000, "c5_raw_e05_v0_tpN_slN_s01", "card5"),
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.3, None, None, None, 3600000, "c5_raw_e05_v0_tpN_slN_s03", "card5"),
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.5, None, None, None, 3600000, "c5_raw_e05_v0_tpN_slN_s05", "card5"),
    ("raw_rawExit_conv0.05_ethr0.5_vol0", 0.7, None, None, None, 3600000, "c5_raw_e05_v0_tpN_slN_s07", "card5"),

    # --- Card 6: raw_rawExit_conv0.15 ethr0.0 vol70, TP20/SL25 ---
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.1, 20, None, 25, 3600000, "c6_raw_c015_e00_v70_tp20_sl25_s01", "card6"),
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.3, 20, None, 25, 3600000, "c6_raw_c015_e00_v70_tp20_sl25_s03", "card6"),
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.5, 20, None, 25, 3600000, "c6_raw_c015_e00_v70_tp20_sl25_s05", "card6"),
    ("raw_rawExit_conv0.15_ethr0.0_vol70", 0.7, 20, None, 25, 3600000, "c6_raw_c015_e00_v70_tp20_sl25_s07", "card6"),

    # --- Card 6: raw_rawExit_conv0.15 ethr0.5 vol70, TP20/SL25 ---
    ("raw_rawExit_conv0.15_ethr0.5_vol70", 0.1, 20, None, 25, 3600000, "c6_raw_c015_e05_v70_tp20_sl25_s01", "card6"),
    ("raw_rawExit_conv0.15_ethr0.5_vol70", 0.3, 20, None, 25, 3600000, "c6_raw_c015_e05_v70_tp20_sl25_s03", "card6"),
    ("raw_rawExit_conv0.15_ethr0.5_vol70", 0.5, 20, None, 25, 3600000, "c6_raw_c015_e05_v70_tp20_sl25_s05", "card6"),
    ("raw_rawExit_conv0.15_ethr0.5_vol70", 0.7, 20, None, 25, 3600000, "c6_raw_c015_e05_v70_tp20_sl25_s07", "card6"),

    # --- Card 7: smooth_smoothExit_conv1.5 ethr0.0 vol70, no TP / SL20 ---
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.1, None, None, 20, 3600000, "c7_smooth_c15_e00_v70_tpN_sl20_s01", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.3, None, None, 20, 3600000, "c7_smooth_c15_e00_v70_tpN_sl20_s03", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.5, None, None, 20, 3600000, "c7_smooth_c15_e00_v70_tpN_sl20_s05", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.0_vol70", 0.7, None, None, 20, 3600000, "c7_smooth_c15_e00_v70_tpN_sl20_s07", "card7"),

    # --- Card 7: smooth_smoothExit_conv1.5 ethr0.5 vol70, no TP / SL20 ---
    ("smooth_smoothExit_conv1.5_ethr0.5_vol70", 0.1, None, None, 20, 3600000, "c7_smooth_c15_e05_v70_tpN_sl20_s01", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.5_vol70", 0.3, None, None, 20, 3600000, "c7_smooth_c15_e05_v70_tpN_sl20_s03", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.5_vol70", 0.5, None, None, 20, 3600000, "c7_smooth_c15_e05_v70_tpN_sl20_s05", "card7"),
    ("smooth_smoothExit_conv1.5_ethr0.5_vol70", 0.7, None, None, 20, 3600000, "c7_smooth_c15_e05_v70_tpN_sl20_s07", "card7"),
]


def get_mbo_path(date):
    nodash = date.replace("-", "")
    return MBO_DIR / "glbx-mdp3-{}.mbo.dbn.zst".format(nodash)


def is_oot_date(date):
    return OOT_START <= date <= OOT_END


def find_dates_for_pred(pred_type):
    dates = {}
    for f in sorted(PRED_DIR.glob("*_{}.npz".format(pred_type))):
        date = f.stem[:10]
        if not is_oot_date(date):
            continue
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
        mo = get_month(date)
        if mo not in regime:
            regime[mo] = {"pnl": 0, "trades": 0, "days": 0}
        regime[mo]["pnl"] += day_pnl
        regime[mo]["trades"] += len(trades)
        regime[mo]["days"] += 1
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

    # Max drawdown in ticks
    cumsum = []
    s = 0
    for p in all_daily_pnls:
        s += p
        cumsum.append(s)
    peak = cumsum[0]
    max_dd = 0
    for v in cumsum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    return {
        "config": config_name, "n_days": n, "trades": total_trades,
        "sharpe": round(sharpe, 2), "wr": round(wr, 1), "pf": round(pf, 2),
        "net_pnl": round(net_pnl, 0), "pos_days": pos_days,
        "max_dd_ticks": round(max_dd, 1),
        "avg_tpd": round(total_trades / n, 1),
        "regime": regime,
    }


def main():
    print("[{}] Cards 5-7 FULL OOT Validation".format(datetime.now()), flush=True)
    print("OOT range: {} to {} (target 68 trading days)".format(OOT_START, OOT_END), flush=True)
    if not BINARY.exists():
        print("ERROR: binary not found: {}".format(BINARY), flush=True)
        sys.exit(1)

    jobs = []
    skipped = 0
    pred_cache = {}

    for (pred_type, sig_thr, tp, trail, sl, hold_ms, name, card_tag) in ALL_CONFIGS:
        if pred_type not in pred_cache:
            pred_cache[pred_type] = find_dates_for_pred(pred_type)
        dates = pred_cache[pred_type]
        if pred_type not in [c[0] for c in ALL_CONFIGS[:ALL_CONFIGS.index((pred_type, sig_thr, tp, trail, sl, hold_ms, name, card_tag))]]:
            print("  {} => {} OOT dates available".format(pred_type, len(dates)), flush=True)
        if not dates:
            print("  WARNING: No OOT dates for: {}".format(pred_type), flush=True)
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

    print("\nTotal NEW jobs: {} (skipped {} already-done)".format(len(jobs), skipped), flush=True)

    t0 = time.time()
    if jobs:
        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_sim, cmd, out, label): label for cmd, out, label in jobs}
            for f in as_completed(futures):
                label, ok, err_msg = f.result()
                done += 1
                if not ok:
                    failed += 1
                    print("  FAIL [{}/{}] {}: {}".format(done, len(jobs), label, err_msg), flush=True)
                elif done % 50 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 1
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print("  [{}/{}] {:.1f}/s ETA {:.0f}s".format(
                        done, len(jobs), rate, eta), flush=True)
        print("Sweep done: {}/{} ok, {} failed in {:.0f}s".format(
            done - failed, len(jobs), failed, time.time() - t0), flush=True)
    else:
        print("All jobs already done - loading results only.", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("CARDS 5-7 OOT RESULTS (Dec 2025 - Mar 2026, FULL period)", flush=True)
    print("=" * 100, flush=True)

    all_results = []
    for (pred_type, sig_thr, tp, trail, sl, hold_ms, name, card_tag) in ALL_CONFIGS:
        out_dir = OUT_BASE / card_tag / name
        dates = pred_cache.get(pred_type, {})
        results_by_date = {}
        for date in sorted(dates.keys()):
            out_path = out_dir / "{}.json".format(date)
            results_by_date[date] = load_result(out_path)
        a = analyze_config(name, results_by_date)
        if a:
            all_results.append((card_tag, a))

    hdr = "{:<8} {:<44} {:>5} {:>6} {:>5} {:>5} {:>9} {:>5} {:>7} {:>5}"
    print(hdr.format("CARD", "Config", "Days", "Sharpe", "WR%", "PF", "Net$", "PosD", "MaxDD", "TPD"), flush=True)
    print("-" * 105, flush=True)
    row_fmt = "{:<8} {:<44} {:>5} {:>6.2f} {:>5.1f} {:>5.2f} {:>9,.0f} {:>3}/{:<2} {:>6.0f} {:>5.1f}"

    prev_tag = None
    for card_tag, a in sorted(all_results, key=lambda x: (x[0], -x[1]["sharpe"])):
        if card_tag != prev_tag:
            print("", flush=True)
            prev_tag = card_tag
        print(row_fmt.format(card_tag, a["config"][:44], a["n_days"], a["sharpe"],
                             a["wr"], a["pf"], a["net_pnl"], a["pos_days"], a["n_days"],
                             a["max_dd_ticks"], a["avg_tpd"]),
              flush=True)
        for mo in ["Dec", "Jan", "Feb", "Mar"]:
            if mo in a["regime"]:
                r = a["regime"][mo]
                avg = r["pnl"] / r["days"] if r["days"] > 0 else 0
                net_mo = r["pnl"] * TICK_VALUE - r["trades"] * COMMISSION_RT
                print("     {} {:>3}d {:>4}t avg {:.1f}tk/d  net ${:,.0f}".format(
                    mo, r["days"], r["trades"], avg, net_mo), flush=True)

    # Coverage report
    print("\n" + "=" * 100, flush=True)
    print("DATE COVERAGE REPORT", flush=True)
    print("=" * 100, flush=True)
    seen = set()
    for pred_type in [c[0] for c in ALL_CONFIGS]:
        if pred_type in seen:
            continue
        seen.add(pred_type)
        dates = sorted(pred_cache.get(pred_type, {}).keys())
        if dates:
            print("  {}: {} dates ({} to {})".format(pred_type, len(dates), dates[0], dates[-1]), flush=True)
        else:
            print("  {}: NO DATES".format(pred_type), flush=True)

    max_dates = max(len(d) for d in pred_cache.values()) if pred_cache else 0
    if max_dates < 68:
        print("\n  NOTE: Predictions cover {} of 68 target OOT dates.".format(max_dates), flush=True)
        print("  Missing: predictions after Feb 17, 2026 not yet generated for these model types.", flush=True)

    summary_path = OUT_BASE / "card567_oot_summary.json"
    with open(summary_path, "w") as f:
        json.dump([{"tag": t, "analysis": a} for t, a in all_results], f, indent=2)
    print("\nSaved: {}".format(summary_path), flush=True)
    print("\nSWEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
