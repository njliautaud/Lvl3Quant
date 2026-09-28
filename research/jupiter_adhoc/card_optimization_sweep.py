#!/usr/bin/env python3
"""Card optimization sweep — runs on Jupiter."""
import json, os, subprocess, time, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import math

BINARY = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/card_optimizations"
WORKERS = 14

# 54 OOT dates (prediction file dates in YYYY-MM-DD format)
DATES = [
    "2025-12-01","2025-12-02","2025-12-03","2025-12-04","2025-12-05",
    "2025-12-08","2025-12-09","2025-12-10","2025-12-11","2025-12-12",
    "2025-12-15","2025-12-16","2025-12-17","2025-12-18","2025-12-19",
    "2025-12-22","2025-12-23","2025-12-24","2025-12-26",
    "2025-12-30","2025-12-31",
    "2026-01-02","2026-01-05","2026-01-06","2026-01-07","2026-01-08","2026-01-09",
    "2026-01-12","2026-01-13","2026-01-14","2026-01-15","2026-01-16",
    "2026-01-19","2026-01-20","2026-01-21","2026-01-22","2026-01-23",
    "2026-01-26","2026-01-27","2026-01-28","2026-01-29","2026-01-30",
    "2026-02-02","2026-02-03","2026-02-04","2026-02-05","2026-02-06",
    "2026-02-09","2026-02-10","2026-02-11","2026-02-12","2026-02-13",
    "2026-02-16","2026-02-17",
]

def date_to_mbo(d):
    """YYYY-MM-DD -> glbx-mdp3-YYYYMMDD.mbo.dbn.zst"""
    return f"glbx-mdp3-{d.replace('-','')}.mbo.dbn.zst"

def run_job(job):
    """Run a single fill_sim job. Returns (test_name, config_key, date, result_dict)."""
    test_name, config_key, date, cmd, outfile = job
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return (test_name, config_key, date, {"error": result.stderr[:200]})
        if os.path.exists(outfile):
            with open(outfile) as f:
                data = json.load(f)
            return (test_name, config_key, date, data)
        else:
            return (test_name, config_key, date, {"error": "no output file"})
    except subprocess.TimeoutExpired:
        return (test_name, config_key, date, {"error": "timeout"})
    except Exception as e:
        return (test_name, config_key, date, {"error": str(e)[:200]})

def build_jobs():
    """Build all 918 jobs across 4 tests."""
    jobs = []
    os.makedirs(OUT_DIR, exist_ok=True)

    for d in DATES:
        mbo = f"{MBO_DIR}/{date_to_mbo(d)}"
        # Verify MBO exists
        if not os.path.exists(mbo):
            print(f"WARNING: MBO missing for {d}, skipping")
            continue

        # ── Test 1: Signal threshold on Card 1 (conv2.5_vol70 + TP3 + trail25 + wb50 + chase1t/3r) ──
        pred1 = f"{PRED_DIR}/{d}_book_predstdExit_conv2.5_vol70.npz"
        if os.path.exists(pred1):
            for sig in [0.05, 0.1, 0.15, 0.2, 0.3]:
                outfile = f"{OUT_DIR}/t1_sig{sig}_{d}.json"
                cmd = [
                    BINARY,
                    "--mbo-file", mbo,
                    "--predictions", pred1,
                    "--output", outfile,
                    "--signal-threshold", str(sig),
                    "--hold-ms", "3600000",
                    "--max-wait-bars", "50",
                    "--latency-ms", "50",
                    "--chase-entry",
                    "--chase-max-ticks", "1",
                    "--chase-max-reprices", "3",
                    "--take-profit-ticks", "3",
                    "--trailing-ticks", "25",
                    "--quiet",
                ]
                jobs.append(("test1_sig", f"sig{sig}", d, cmd, outfile))

        # ── Test 2: Chase params on Card 1 (sig0.1 fixed) ──
        if os.path.exists(pred1):
            chase_configs = [
                ("chase_none", []),  # no chase flags at all
                ("chase_1t3r", ["--chase-entry", "--chase-max-ticks", "1", "--chase-max-reprices", "3"]),
                ("chase_2t5r", ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5"]),
                ("chase_1t5r", ["--chase-entry", "--chase-max-ticks", "1", "--chase-max-reprices", "5"]),
            ]
            for cname, cflags in chase_configs:
                outfile = f"{OUT_DIR}/t2_{cname}_{d}.json"
                cmd = [
                    BINARY,
                    "--mbo-file", mbo,
                    "--predictions", pred1,
                    "--output", outfile,
                    "--signal-threshold", "0.1",
                    "--hold-ms", "3600000",
                    "--max-wait-bars", "50",
                    "--latency-ms", "50",
                    "--take-profit-ticks", "3",
                    "--trailing-ticks", "25",
                    "--quiet",
                ] + cflags
                jobs.append(("test2_chase", cname, d, cmd, outfile))

        # ── Test 3: Signal threshold on Card 2 (conv1.5_vol50 + TP15 + no SL) ──
        pred2 = f"{PRED_DIR}/{d}_book_predstdExit_conv1.5_vol50.npz"
        if os.path.exists(pred2):
            for sig in [0.05, 0.1, 0.15, 0.2]:
                outfile = f"{OUT_DIR}/t3_sig{sig}_{d}.json"
                cmd = [
                    BINARY,
                    "--mbo-file", mbo,
                    "--predictions", pred2,
                    "--output", outfile,
                    "--signal-threshold", str(sig),
                    "--hold-ms", "3600000",
                    "--max-wait-bars", "50",
                    "--latency-ms", "50",
                    "--chase-entry",
                    "--chase-max-ticks", "1",
                    "--chase-max-reprices", "3",
                    "--take-profit-ticks", "15",
                    "--quiet",
                ]
                jobs.append(("test3_sig", f"sig{sig}", d, cmd, outfile))

        # ── Test 4: Chase params on Card 2 (sig0.1, TP15, no SL) ──
        if os.path.exists(pred2):
            chase_configs = [
                ("chase_none", []),
                ("chase_1t3r", ["--chase-entry", "--chase-max-ticks", "1", "--chase-max-reprices", "3"]),
                ("chase_2t5r", ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5"]),
                ("chase_1t5r", ["--chase-entry", "--chase-max-ticks", "1", "--chase-max-reprices", "5"]),
            ]
            for cname, cflags in chase_configs:
                outfile = f"{OUT_DIR}/t4_{cname}_{d}.json"
                cmd = [
                    BINARY,
                    "--mbo-file", mbo,
                    "--predictions", pred2,
                    "--output", outfile,
                    "--signal-threshold", "0.1",
                    "--hold-ms", "3600000",
                    "--max-wait-bars", "50",
                    "--latency-ms", "50",
                    "--take-profit-ticks", "15",
                    "--quiet",
                ] + cflags
                jobs.append(("test4_chase", cname, d, cmd, outfile))

    return jobs

def extract_metrics(data):
    """Extract key metrics from fill_sim output JSON."""
    if "error" in data:
        return None
    try:
        pnl = data.get("total_pnl_ticks", 0)
        trades = data.get("total_trades", 0)
        wins = data.get("winning_trades", 0)
        losses = data.get("losing_trades", 0)
        wr = wins / trades * 100 if trades > 0 else 0
        gross_win = data.get("gross_winning_ticks", 0)
        gross_loss = abs(data.get("gross_losing_ticks", 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf') if gross_win > 0 else 0
        return {
            "pnl_ticks": pnl,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "wr": wr,
            "pf": pf,
            "gross_win": gross_win,
            "gross_loss": gross_loss,
        }
    except Exception:
        return None

def aggregate_results(results_by_test):
    """Aggregate per-config metrics across all dates."""
    summary = {}
    for test_name, config_results in results_by_test.items():
        summary[test_name] = {}
        for config_key, date_results in config_results.items():
            metrics_list = []
            for date, data in date_results.items():
                m = extract_metrics(data)
                if m:
                    metrics_list.append(m)

            if not metrics_list:
                summary[test_name][config_key] = {"error": "no valid results"}
                continue

            n_days = len(metrics_list)
            total_pnl = sum(m["pnl_ticks"] for m in metrics_list)
            total_trades = sum(m["trades"] for m in metrics_list)
            total_wins = sum(m["wins"] for m in metrics_list)
            total_losses = sum(m["losses"] for m in metrics_list)
            total_gross_win = sum(m["gross_win"] for m in metrics_list)
            total_gross_loss = sum(m["gross_loss"] for m in metrics_list)

            wr = total_wins / total_trades * 100 if total_trades > 0 else 0
            pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float('inf') if total_gross_win > 0 else 0
            trades_per_day = total_trades / n_days if n_days > 0 else 0

            # Daily PnL for Sharpe
            daily_pnls = [m["pnl_ticks"] for m in metrics_list]
            mean_daily = sum(daily_pnls) / len(daily_pnls) if daily_pnls else 0
            if len(daily_pnls) > 1:
                var = sum((x - mean_daily)**2 for x in daily_pnls) / (len(daily_pnls) - 1)
                std_daily = math.sqrt(var) if var > 0 else 0.001
            else:
                std_daily = 0.001
            sharpe = (mean_daily / std_daily) * math.sqrt(252) if std_daily > 0 else 0

            # PnL in dollars ($12.50 per tick)
            total_pnl_dollars = total_pnl * 12.50
            # Subtract commissions ($4.70 per RT)
            net_pnl_dollars = total_pnl_dollars - (total_trades * 4.70)

            summary[test_name][config_key] = {
                "n_days": n_days,
                "total_pnl_ticks": round(total_pnl, 2),
                "total_pnl_dollars": round(total_pnl_dollars, 2),
                "net_pnl_dollars": round(net_pnl_dollars, 2),
                "total_trades": total_trades,
                "trades_per_day": round(trades_per_day, 1),
                "win_rate": round(wr, 1),
                "profit_factor": round(pf, 3) if pf != float('inf') else 999,
                "sharpe": round(sharpe, 2),
                "mean_daily_ticks": round(mean_daily, 2),
                "std_daily_ticks": round(std_daily, 2),
            }
    return summary

def main():
    print("=" * 60)
    print("CARD OPTIMIZATION SWEEP — Jupiter")
    print("=" * 60)

    # Build all jobs
    jobs = build_jobs()
    print(f"\nTotal jobs: {len(jobs)}")

    # Count per test
    test_counts = defaultdict(int)
    for j in jobs:
        test_counts[j[0]] += 1
    for t, c in sorted(test_counts.items()):
        print(f"  {t}: {c} jobs")

    print(f"\nRunning with {WORKERS} workers...")
    start = time.time()

    # Execute all jobs
    results_by_test = defaultdict(lambda: defaultdict(dict))
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            test_name, config_key, date, data = future.result()
            results_by_test[test_name][config_key][date] = data
            completed += 1
            if "error" in data:
                errors += 1
            if completed % 100 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(jobs) - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed}/{len(jobs)} ({errors} errors) — {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

    elapsed = time.time() - start
    print(f"\nCompleted {completed}/{len(jobs)} in {elapsed:.0f}s ({errors} errors)")

    # Aggregate
    summary = aggregate_results(results_by_test)

    # Save raw summary
    summary_file = f"{OUT_DIR}/card_optimization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Baselines for comparison
    baselines = {}

    for test_name in sorted(summary.keys()):
        configs = summary[test_name]
        print(f"\n{'─' * 60}")
        print(f"  {test_name.upper()}")
        print(f"{'─' * 60}")
        print(f"  {'Config':<16} {'PnL$':>8} {'Net$':>8} {'Trades':>7} {'T/Day':>6} {'WR%':>6} {'PF':>6} {'Sharpe':>7}")
        print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

        for config_key in sorted(configs.keys()):
            c = configs[config_key]
            if "error" in c:
                print(f"  {config_key:<16} ERROR: {c['error']}")
                continue

            flag = ""
            # Store baseline
            if test_name == "test1_sig" and config_key == "sig0.1":
                baselines["card1"] = c
            elif test_name == "test2_chase" and config_key == "chase_1t3r":
                baselines["card1_chase"] = c
            elif test_name == "test3_sig" and config_key == "sig0.1":
                baselines["card2"] = c
            elif test_name == "test4_chase" and config_key == "chase_1t3r":
                baselines["card2_chase"] = c

            # Check improvement vs baseline
            base_key = "card1" if "test1" in test_name or "test2" in test_name else "card2"
            if base_key in baselines:
                base = baselines[base_key]
                if base.get("profit_factor", 0) > 0 and c.get("profit_factor", 0) > 0:
                    pf_imp = (c["profit_factor"] - base["profit_factor"]) / base["profit_factor"] * 100
                    sh_imp = (c["sharpe"] - base["sharpe"]) / abs(base["sharpe"]) * 100 if base["sharpe"] != 0 else 0
                    if pf_imp > 10 or sh_imp > 10:
                        flag = " ★"

            print(f"  {config_key:<16} {c['total_pnl_dollars']:>8.0f} {c['net_pnl_dollars']:>8.0f} {c['total_trades']:>7} {c['trades_per_day']:>6.1f} {c['win_rate']:>6.1f} {c['profit_factor']:>6.2f} {c['sharpe']:>7.2f}{flag}")

    # Print improvement analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT vs BASELINE (sig0.1 / chase_1t3r)")
    print("=" * 80)

    for test_name in sorted(summary.keys()):
        configs = summary[test_name]
        base_key = "card1" if "test1" in test_name else "card1_chase" if "test2" in test_name else "card2" if "test3" in test_name else "card2_chase"
        if base_key not in baselines:
            continue
        base = baselines[base_key]
        if "error" in base:
            continue

        print(f"\n  {test_name}: baseline = {base_key} (PF={base['profit_factor']:.2f}, Sharpe={base['sharpe']:.2f})")
        for config_key in sorted(configs.keys()):
            c = configs[config_key]
            if "error" in c:
                continue
            if base["profit_factor"] > 0:
                pf_chg = (c["profit_factor"] - base["profit_factor"]) / base["profit_factor"] * 100
            else:
                pf_chg = 0
            sh_chg = (c["sharpe"] - base["sharpe"]) / abs(base["sharpe"]) * 100 if base["sharpe"] != 0 else 0
            net_chg = c["net_pnl_dollars"] - base["net_pnl_dollars"]

            markers = []
            if abs(pf_chg) > 10:
                markers.append(f"PF {'+'if pf_chg>0 else ''}{pf_chg:.0f}%")
            if abs(sh_chg) > 10:
                markers.append(f"Sharpe {'+'if sh_chg>0 else ''}{sh_chg:.0f}%")

            marker_str = f"  ★ {', '.join(markers)}" if markers else ""
            print(f"    {config_key:<16} PF={c['profit_factor']:.2f} ({pf_chg:+.0f}%), Sharpe={c['sharpe']:.2f} ({sh_chg:+.0f}%), Net${net_chg:+.0f}{marker_str}")

    print("\n✓ Card optimization sweep complete.")

if __name__ == "__main__":
    main()

