#!/usr/bin/env python3
"""Full OOT Execution Engine Validation - Baseline vs Trailing Stop vs Vol Gate
Runs across ALL available OOT dates (Dec 2025 - Mar 2026).
"""
import sys, json, time, subprocess, os, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "exit_engine_validation"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

OOT_START = "2025-12-01"
OOT_END   = "2026-03-31"

ALL_CONFIGS = [
    # ===== BASELINE CONFIGS (current production) =====
    # Card1 baseline: TP8, no SL, 30min hold
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, None, None, 1800000, "c1_base_tp8_30m", "baseline"),
    ("book_predstdExit_conv1.5_vol50", 0.3, 8, None, None, 1800000, "c1_base_tp8_30m_s03", "baseline"),
    ("book_predstdExit_conv1.5_vol50", 0.5, 8, None, None, 1800000, "c1_base_tp8_30m_s05", "baseline"),

    # Card2 baseline: TP15, no SL, 30min hold
    ("book_predstdExit_conv2.0_vol50", 0.1, 15, None, None, 1800000, "c2_base_tp15_30m", "baseline"),
    ("book_predstdExit_conv2.0_vol50", 0.3, 15, None, None, 1800000, "c2_base_tp15_30m_s03", "baseline"),

    # Card3 baseline: TP15, no SL, 30min hold (vol70 AND vol50)
    ("raw_smoothExit_conv0.05_ethr0.0_vol70", 0.3, 15, None, None, 1800000, "c3_base_tp15_30m_v70", "baseline"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol50", 0.3, 15, None, None, 1800000, "c3_base_tp15_30m_v50", "baseline"),

    # Card4 baseline: TP20, no SL, 30min hold (vol70 AND vol50)
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, None, None, 1800000, "c4_base_tp20_30m_v70", "baseline"),
    ("book_predstdExit_conv2.0_vol50", 0.1, 20, None, None, 1800000, "c4_base_tp20_30m_v50", "baseline"),

    # ===== WITH TRAILING STOP (ratcheting exit) =====
    # Card1: TP8, trailing 5t, 2hr hold
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, 5, None, 7200000, "c1_trail5_tp8_2h", "trailing"),
    ("book_predstdExit_conv1.5_vol50", 0.3, 8, 5, None, 7200000, "c1_trail5_tp8_2h_s03", "trailing"),
    ("book_predstdExit_conv1.5_vol50", 0.5, 8, 5, None, 7200000, "c1_trail5_tp8_2h_s05", "trailing"),

    # Card1: TP8, trailing 3t, 2hr hold (tighter trail)
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, 3, None, 7200000, "c1_trail3_tp8_2h", "trailing"),

    # Card2: TP15, trailing 8t, 2hr hold
    ("book_predstdExit_conv2.0_vol50", 0.1, 15, 8, None, 7200000, "c2_trail8_tp15_2h", "trailing"),
    ("book_predstdExit_conv2.0_vol50", 0.3, 15, 8, None, 7200000, "c2_trail8_tp15_2h_s03", "trailing"),

    # Card2: TP15, trailing 5t, 2hr hold (tighter)
    ("book_predstdExit_conv2.0_vol50", 0.1, 15, 5, None, 7200000, "c2_trail5_tp15_2h", "trailing"),

    # Card4: TP20, trailing 10t, 2hr hold
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, 10, None, 7200000, "c4_trail10_tp20_2h_v70", "trailing"),
    ("book_predstdExit_conv2.0_vol50", 0.1, 20, 10, None, 7200000, "c4_trail10_tp20_2h_v50", "trailing"),

    # Card4: TP20, trailing 8t, 2hr hold (tighter)
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, 8, None, 7200000, "c4_trail8_tp20_2h_v70", "trailing"),

    # ===== HOLD TIME COMPARISON (MAE-time proxy) =====
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, None, None, 600000,  "c1_base_tp8_10m", "holdtime"),
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, None, None, 3600000, "c1_base_tp8_60m", "holdtime"),
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, None, None, 7200000, "c1_base_tp8_2h", "holdtime"),

    ("book_predstdExit_conv2.0_vol70", 0.1, 20, None, None, 600000,  "c4_base_tp20_10m_v70", "holdtime"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, None, None, 3600000, "c4_base_tp20_60m_v70", "holdtime"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, None, None, 7200000, "c4_base_tp20_2h_v70", "holdtime"),

    # ===== VOL GATE COMPARISON (vol50 vs vol70) =====
    ("raw_smoothExit_conv0.05_ethr0.0_vol50", 0.3, 15, None, None, 1800000, "c3_v50_tp15_30m", "volgate"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol70", 0.3, 15, None, None, 1800000, "c3_v70_tp15_30m", "volgate"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol50", 0.5, 15, None, None, 1800000, "c3_v50_tp15_30m_s05", "volgate"),
    ("raw_smoothExit_conv0.05_ethr0.0_vol70", 0.5, 15, None, None, 1800000, "c3_v70_tp15_30m_s05", "volgate"),

    ("book_predstdExit_conv2.0_vol50", 0.1, 20, None, None, 1800000, "c4_v50_tp20_30m", "volgate"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, None, None, 1800000, "c4_v70_tp20_30m", "volgate"),
    ("book_predstdExit_conv2.0_vol50", 0.3, 20, None, None, 1800000, "c4_v50_tp20_30m_s03", "volgate"),
    ("book_predstdExit_conv2.0_vol70", 0.3, 20, None, None, 1800000, "c4_v70_tp20_30m_s03", "volgate"),

    # ===== WITH FIXED STOP LOSS =====
    ("book_predstdExit_conv1.5_vol50", 0.1, 8, None, 6, 1800000, "c1_tp8_sl6_30m", "stoploss"),
    ("book_predstdExit_conv2.0_vol50", 0.1, 15, None, 10, 1800000, "c2_tp15_sl10_30m", "stoploss"),
    ("book_predstdExit_conv2.0_vol70", 0.1, 20, None, 12, 1800000, "c4_tp20_sl12_30m_v70", "stoploss"),
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
    total_losses = 0
    all_trade_pnls = []
    all_mae_winners = []
    all_mae_losers = []
    all_mfe = []
    all_hold_winners_ms = []
    all_hold_losers_ms = []
    all_wins_pnl = []
    all_losses_pnl = []
    exit_reasons = {}
    equity_curve = []
    running_equity = 0

    for date in sorted(results_by_date.keys()):
        data = results_by_date[date]
        if data is None:
            continue
        trades = data.get("trades", [])
        day_pnl = 0
        for t in trades:
            pnl = t.get("pnl_ticks", 0)
            pnl_dollars = t.get("pnl_dollars", pnl * TICK_VALUE)
            day_pnl += pnl
            all_trade_pnls.append(pnl)
            total_trades += 1

            mae = t.get("mae_ticks", 0)
            mfe = t.get("mfe_ticks", 0)
            hold_ns = t.get("hold_duration_ns", 0)
            hold_ms = hold_ns / 1e6 if hold_ns else 0

            all_mfe.append(mfe)

            reason = t.get("exit_reason", "Unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            if pnl > 0:
                total_wins += 1
                all_wins_pnl.append(pnl)
                all_mae_winners.append(mae)
                all_hold_winners_ms.append(hold_ms)
            elif pnl < 0:
                total_losses += 1
                all_losses_pnl.append(pnl)
                all_mae_losers.append(mae)
                all_hold_losers_ms.append(hold_ms)

            running_equity += pnl_dollars - COMMISSION_RT
            equity_curve.append(running_equity)

        all_daily_pnls.append(day_pnl)

    n = len(all_daily_pnls)
    if n < 2 or total_trades == 0:
        return None

    mean_d = sum(all_daily_pnls) / n
    std_d = (sum((x - mean_d)**2 for x in all_daily_pnls) / (n - 1))**0.5
    sharpe = (mean_d / std_d * (252**0.5)) if std_d > 0 else 0

    gross_p = sum(p for p in all_trade_pnls if p > 0)
    gross_l = abs(sum(p for p in all_trade_pnls if p < 0))
    pf = gross_p / gross_l if gross_l > 0 else float('inf')
    wr = 100.0 * total_wins / total_trades if total_trades > 0 else 0

    net_pnl = sum(all_daily_pnls) * TICK_VALUE - total_trades * COMMISSION_RT

    avg_win = sum(all_wins_pnl) / len(all_wins_pnl) * TICK_VALUE if all_wins_pnl else 0
    avg_loss = sum(all_losses_pnl) / len(all_losses_pnl) * TICK_VALUE if all_losses_pnl else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    avg_hold_win = sum(all_hold_winners_ms) / len(all_hold_winners_ms) / 1000 if all_hold_winners_ms else 0
    avg_hold_loss = sum(all_hold_losers_ms) / len(all_hold_losers_ms) / 1000 if all_hold_losers_ms else 0

    avg_mae_win = sum(all_mae_winners) / len(all_mae_winners) if all_mae_winners else 0
    avg_mae_loss = sum(all_mae_losers) / len(all_mae_losers) if all_mae_losers else 0
    avg_mfe = sum(all_mfe) / len(all_mfe) if all_mfe else 0

    max_dd = 0
    peak = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    trades_per_day = total_trades / n if n > 0 else 0
    pos_days = sum(1 for p in all_daily_pnls if p > 0)

    return {
        "config": config_name,
        "n_days": n,
        "trades": total_trades,
        "trades_per_day": round(trades_per_day, 1),
        "sharpe": round(sharpe, 2),
        "wr": round(wr, 1),
        "pf": round(pf, 2) if pf != float('inf') else 999.0,
        "net_pnl": round(net_pnl, 0),
        "avg_win": round(avg_win, 1),
        "avg_loss": round(avg_loss, 1),
        "wl_ratio": round(wl_ratio, 2) if wl_ratio != float('inf') else 999.0,
        "avg_hold_win_s": round(avg_hold_win, 1),
        "avg_hold_loss_s": round(avg_hold_loss, 1),
        "avg_mae_win": round(avg_mae_win, 2),
        "avg_mae_loss": round(avg_mae_loss, 2),
        "avg_mfe": round(avg_mfe, 2),
        "max_drawdown": round(max_dd, 0),
        "pos_days": pos_days,
        "exit_reasons": exit_reasons,
    }


def main():
    print("[{}] Exit Engine Full Validation Starting".format(datetime.now()), flush=True)
    print("Date range: {} to {}".format(OOT_START, OOT_END), flush=True)
    if not BINARY.exists():
        print("ERROR: binary not found: {}".format(BINARY), flush=True)
        sys.exit(1)

    jobs = []
    skipped = 0
    pred_cache = {}

    for (pred_type, sig_thr, tp, trail, sl, hold_ms, name, group) in ALL_CONFIGS:
        if pred_type not in pred_cache:
            pred_cache[pred_type] = find_dates_for_pred(pred_type)
        dates = pred_cache[pred_type]
        if not dates:
            print("WARNING: No dates for: {}".format(pred_type), flush=True)
            continue
        print("  {} -> {} dates".format(name, len(dates)), flush=True)
        out_dir = OUT_BASE / group / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for date, pred_path in sorted(dates.items()):
            out_path = out_dir / "{}.json".format(date)
            if out_path.exists():
                skipped += 1
                continue
            cmd = build_cmd(date, pred_path, out_path, sig_thr, tp, trail, sl, hold_ms)
            jobs.append((cmd, str(out_path), "{}/{}".format(name, date)))

    print("Total jobs: {} (skipped {} already-done)".format(len(jobs), skipped), flush=True)
    print("Configs: {}".format(len(ALL_CONFIGS)), flush=True)

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
                elif done % 100 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 1
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    print("  [{}/{}] {:.1f}/s ETA {:.0f}s".format(
                        done, len(jobs), rate, eta), flush=True)
        print("Sweep done: {}/{} ok, {} failed in {:.0f}s".format(
            done - failed, len(jobs), failed, time.time() - t0), flush=True)
    else:
        print("All jobs already done.", flush=True)

    # ========== ANALYSIS ==========
    print("\n" + "=" * 150, flush=True)
    print("EXIT ENGINE VALIDATION RESULTS - ALL OOT DATES", flush=True)
    print("=" * 150, flush=True)

    all_results = []
    for (pred_type, sig_thr, tp, trail, sl, hold_ms, name, group) in ALL_CONFIGS:
        out_dir = OUT_BASE / group / name
        dates = pred_cache.get(pred_type, {})
        results_by_date = {}
        for date in sorted(dates.keys()):
            out_path = out_dir / "{}.json".format(date)
            results_by_date[date] = load_result(out_path)
        a = analyze_config(name, results_by_date)
        if a:
            a["group"] = group
            all_results.append(a)

    groups_order = ["baseline", "trailing", "holdtime", "volgate", "stoploss"]
    group_labels = {
        "baseline": "BASELINE (current production configs)",
        "trailing": "WITH TRAILING STOP (ratcheting exit)",
        "holdtime": "HOLD TIME COMPARISON",
        "volgate":  "VOL GATE COMPARISON (vol50 vs vol70)",
        "stoploss": "WITH FIXED STOP LOSS",
    }

    hdr = "{:<30} {:>6} {:>5} {:>7} {:>5} {:>6} {:>10} {:>8} {:>8} {:>5} {:>7} {:>7} {:>6} {:>6} {:>5} {:>8}"
    cols = ("Config", "Trd", "T/d", "Sharpe", "WR%", "PF", "Net$", "AvgWin", "AvgLos", "W/L", "HldW_s", "HldL_s", "MAEw", "MAEl", "MFE", "MaxDD$")

    for grp in groups_order:
        grp_results = [r for r in all_results if r["group"] == grp]
        if not grp_results:
            continue
        print("\n--- {} ---".format(group_labels.get(grp, grp)), flush=True)
        print(hdr.format(*cols), flush=True)
        print("-" * 150, flush=True)
        for a in sorted(grp_results, key=lambda x: -x["sharpe"]):
            pf_str = "{:.2f}".format(a["pf"]) if a["pf"] < 900 else "INF"
            wl_str = "{:.2f}".format(a["wl_ratio"]) if a["wl_ratio"] < 900 else "INF"
            print(hdr.format(
                a["config"][:30],
                a["trades"],
                a["trades_per_day"],
                a["sharpe"],
                a["wr"],
                pf_str,
                "${:,.0f}".format(a["net_pnl"]),
                "${:.0f}".format(a["avg_win"]),
                "${:.0f}".format(a["avg_loss"]),
                wl_str,
                a["avg_hold_win_s"],
                a["avg_hold_loss_s"],
                a["avg_mae_win"],
                a["avg_mae_loss"],
                a["avg_mfe"],
                "${:,.0f}".format(a["max_drawdown"]),
            ), flush=True)
            reasons = a.get("exit_reasons", {})
            if reasons:
                parts = ["{}: {}".format(k, v) for k, v in sorted(reasons.items(), key=lambda x: -x[1])]
                print("    Exits: {}  |  Days: {}/{}".format(", ".join(parts), a["pos_days"], a["n_days"]), flush=True)

    summary_path = OUT_BASE / "full_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved: {}".format(summary_path), flush=True)
    print("\nVALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
