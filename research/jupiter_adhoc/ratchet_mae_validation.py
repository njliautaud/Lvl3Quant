#!/usr/bin/env python3
"""
Ratchet Stop + MAE Exit Validation Sweep
=========================================
Full 68-day OOT validation: Baseline vs Ratchet vs MAE vs Full optimization
for all 4 production cards.

4 cards x 4 modes x ~54 dates = ~864 jobs
Uses 6 workers (alongside existing sweeps on Saturn).

Cards:
  Card1: book_predstdExit_conv1.5_vol50, sig=0.1, TP8
  Card2: book_predstdExit_conv1.5_vol50, sig=0.1, TP15
  Card3: raw_smoothExit_conv0.05_ethr0.0_vol70, sig=0.5, TP15
  Card4: book_predstdExit_conv2.0_vol70, sig=0.5, TP20

Modes:
  A (baseline): Fixed hold 1hr, no ratchet, no MAE
  B (+ratchet): Ratchet stop, 2hr hold
  C (+MAE):     MAE exit 10t/600s, 2hr hold (no ratchet)
  D (full):     Ratchet + MAE exit 10t/600s, 2hr hold
"""

import subprocess, os, json, glob, time, sys, math
from multiprocessing import Pool
from datetime import datetime
from collections import defaultdict

# ── Paths ──
FILL_SIM = "/home/saturn/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli_v2"
PRED_DIR = "/home/saturn/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
MBO_DIRS = [
    "/home/saturn/Lvl3Quant/data/raw/mbo",
    "/home/saturn/Lvl3Quant/mbo_oot",
]
OUT_DIR = "/home/saturn/Lvl3Quant/data/processed/ratchet_mae_validation"
os.makedirs(OUT_DIR, exist_ok=True)

WORKERS = 6  # Conservative alongside existing sweeps

# ── Card Definitions ──
CARDS = {
    "card1": {
        "pred_model": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "take_profit_ticks": 8,
    },
    "card2": {
        "pred_model": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "take_profit_ticks": 15,
    },
    "card3": {
        "pred_model": "raw_smoothExit_conv0.05_ethr0.0_vol70",
        "signal_threshold": 0.5,
        "take_profit_ticks": 15,
    },
    "card4": {
        "pred_model": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.5,
        "take_profit_ticks": 20,
    },
}

# ── Mode Definitions ──
# Common: chase-entry with 1t/3r, latency 50ms, max-wait 50 bars
MODES = {
    "A_baseline": {
        "hold_ms": 3600000,  # 1hr
        "ratchet_stop": False,
        "mae_exit_ticks": 0,
        "mae_exit_hold_sec": 0,
    },
    "B_ratchet": {
        "hold_ms": 7200000,  # 2hr
        "ratchet_stop": True,
        "mae_exit_ticks": 0,
        "mae_exit_hold_sec": 0,
    },
    "C_mae": {
        "hold_ms": 7200000,  # 2hr
        "ratchet_stop": False,
        "mae_exit_ticks": 10,
        "mae_exit_hold_sec": 600,
    },
    "D_full": {
        "hold_ms": 7200000,  # 2hr
        "ratchet_stop": True,
        "mae_exit_ticks": 10,
        "mae_exit_hold_sec": 600,
    },
}


def find_mbo(date_str):
    """Find MBO file for a date across all MBO directories."""
    mbo_date = date_str.replace("-", "")
    for d in MBO_DIRS:
        path = os.path.join(d, f"glbx-mdp3-{mbo_date}.mbo.dbn.zst")
        if os.path.exists(path):
            return path
    return None


def find_dates_for_card(card_name):
    """Find all dates where both prediction and MBO exist for a card."""
    card = CARDS[card_name]
    pattern = os.path.join(PRED_DIR, f"*_{card['pred_model']}.npz")
    pred_files = sorted(glob.glob(pattern))
    dates = []
    for pf in pred_files:
        basename = os.path.basename(pf)
        # Extract date: everything before the model name
        date_str = basename.split(f"_{card['pred_model']}")[0]
        mbo_file = find_mbo(date_str)
        if mbo_file:
            dates.append((date_str, pf, mbo_file))
    return dates


def run_sim(args):
    """Run a single fill simulation."""
    card_name, mode_name, date_str, pred_file, mbo_file = args
    card = CARDS[card_name]
    mode = MODES[mode_name]

    out_file = os.path.join(OUT_DIR, f"{card_name}_{mode_name}_{date_str}.json")

    # Skip if already done
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                data = json.load(f)
            if "summary" in data:
                return {"card": card_name, "mode": mode_name, "date": date_str, "status": "skip"}
        except:
            pass

    cmd = [
        FILL_SIM,
        "--mbo-file", mbo_file,
        "--predictions", pred_file,
        "--output", out_file,
        "--signal-threshold", str(card["signal_threshold"]),
        "--hold-ms", str(mode["hold_ms"]),
        "--take-profit-ticks", str(card["take_profit_ticks"]),
        "--latency-ms", "50",
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--max-wait-bars", "50",
        "--quiet",
    ]

    if mode["ratchet_stop"]:
        cmd.append("--ratchet-stop")
    if mode["mae_exit_ticks"] > 0:
        cmd.extend(["--mae-exit-ticks", str(mode["mae_exit_ticks"])])
    if mode["mae_exit_hold_sec"] > 0:
        cmd.extend(["--mae-exit-hold-sec", str(mode["mae_exit_hold_sec"])])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(out_file):
            return {"card": card_name, "mode": mode_name, "date": date_str, "status": "ok"}
        else:
            return {"card": card_name, "mode": mode_name, "date": date_str, "status": "error",
                    "stderr": (result.stderr or "")[:300]}
    except subprocess.TimeoutExpired:
        return {"card": card_name, "mode": mode_name, "date": date_str, "status": "timeout"}
    except Exception as e:
        return {"card": card_name, "mode": mode_name, "date": date_str, "status": "error",
                "stderr": str(e)[:300]}


def aggregate_results():
    """Read all result JSONs and compute per-card/mode aggregates."""
    results = defaultdict(lambda: defaultdict(list))

    for card_name in CARDS:
        for mode_name in MODES:
            pattern = os.path.join(OUT_DIR, f"{card_name}_{mode_name}_*.json")
            files = sorted(glob.glob(pattern))
            for f in files:
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    results[card_name][mode_name].append(data)
                except:
                    pass

    return results


def compute_sharpe(pnls):
    """Annualized Sharpe from per-day PnL list."""
    if len(pnls) < 2:
        return 0.0
    mean_pnl = sum(pnls) / len(pnls)
    var = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mean_pnl / std) * math.sqrt(252)


def print_table(results):
    """Print clean comparison table."""
    print("\n" + "=" * 120)
    print("RATCHET STOP + MAE EXIT VALIDATION — FULL OOT RESULTS")
    print("=" * 120)

    header = f"{'Card':>8} | {'Metric':>12} | {'A_baseline':>14} | {'B_ratchet':>14} | {'C_mae':>14} | {'D_full':>14}"
    print(header)
    print("-" * 120)

    for card_name in sorted(CARDS.keys()):
        card_results = results[card_name]
        stats = {}

        for mode_name in MODES:
            data_list = card_results.get(mode_name, [])
            if not data_list:
                stats[mode_name] = None
                continue

            daily_pnls = [d.get("total_pnl_dollars", 0) for d in data_list]
            total_trades = sum(d.get("total_trades", 0) for d in data_list)
            total_pnl = sum(daily_pnls)

            # Per-trade PnL
            all_trades = []
            for d in data_list:
                for t in d.get("trades", []):
                    all_trades.append(t.get("pnl_dollars", 0))

            win_count = sum(1 for p in all_trades if p > 0)
            wr = (win_count / len(all_trades) * 100) if all_trades else 0

            # MAE/MFE
            maes = [t.get("mae_ticks", 0) for t in all_trades if "mae_ticks" in t]
            mfes = [t.get("mfe_ticks", 0) for t in all_trades if "mfe_ticks" in t]
            avg_mae = sum(maes) / len(maes) if maes else 0
            avg_mfe = sum(mfes) / len(mfes) if mfes else 0

            # Max drawdown (cumulative PnL)
            cum = 0
            peak = 0
            max_dd = 0
            for p in all_trades:
                cum += p
                peak = max(peak, cum)
                dd = peak - cum
                max_dd = max(max_dd, dd)

            # Per-trade stats
            avg_pnl = sum(all_trades) / len(all_trades) if all_trades else 0
            wins = [p for p in all_trades if p > 0]
            losses = [p for p in all_trades if p < 0]
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            sharpe = compute_sharpe(daily_pnls)

            stats[mode_name] = {
                "sharpe": sharpe,
                "total_pnl": total_pnl,
                "trades": total_trades,
                "wr": wr,
                "avg_pnl": avg_pnl,
                "avg_mae": avg_mae,
                "avg_mfe": avg_mfe,
                "max_dd": max_dd,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "days": len(data_list),
            }

        def fmt(mode, key, fmt_str="{:.2f}"):
            s = stats.get(mode)
            if s is None:
                return "N/A".rjust(14)
            return fmt_str.format(s[key]).rjust(14)

        metrics = [
            ("Sharpe", "sharpe", "{:.2f}"),
            ("PnL($)", "total_pnl", "{:.0f}"),
            ("Trades", "trades", "{:.0f}"),
            ("WR(%)", "wr", "{:.1f}"),
            ("$/trade", "avg_pnl", "{:.2f}"),
            ("AvgMAE(t)", "avg_mae", "{:.2f}"),
            ("AvgMFE(t)", "avg_mfe", "{:.2f}"),
            ("MaxDD($)", "max_dd", "{:.0f}"),
            ("AvgWin($)", "avg_win", "{:.2f}"),
            ("AvgLoss($)", "avg_loss", "{:.2f}"),
            ("Days", "days", "{:.0f}"),
        ]

        for i, (label, key, fs) in enumerate(metrics):
            cn = card_name if i == 0 else ""
            line = f"{cn:>8} | {label:>12} | {fmt('A_baseline', key, fs)} | {fmt('B_ratchet', key, fs)} | {fmt('C_mae', key, fs)} | {fmt('D_full', key, fs)}"
            print(line)
        print("-" * 120)

    # Summary line
    print("\nKey:")
    print("  A_baseline = TP only, 1hr hold")
    print("  B_ratchet  = Ratcheting trailing stop, 2hr hold")
    print("  C_mae      = MAE exit (10t after 600s), 2hr hold")
    print("  D_full     = Ratchet + MAE exit, 2hr hold")
    print(f"\nCompleted: {datetime.now().isoformat()}")


def main():
    print(f"{'='*80}")
    print("RATCHET STOP + MAE EXIT — FULL OOT VALIDATION")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Binary: {FILL_SIM}")
    print(f"Workers: {WORKERS}")
    print(f"{'='*80}")

    # Build task list
    tasks = []
    for card_name in sorted(CARDS.keys()):
        dates = find_dates_for_card(card_name)
        print(f"\n{card_name}: {len(dates)} dates with predictions + MBO")
        for mode_name in sorted(MODES.keys()):
            for date_str, pred_file, mbo_file in dates:
                tasks.append((card_name, mode_name, date_str, pred_file, mbo_file))

    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Cards: {list(CARDS.keys())}")
    print(f"Modes: {list(MODES.keys())}")

    # Run
    t_start = time.time()
    ok = 0
    skip = 0
    fail = 0

    with Pool(WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(run_sim, tasks)):
            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "skip":
                skip += 1
            else:
                fail += 1

            total_done = ok + skip + fail
            if total_done % 10 == 0 or total_done == len(tasks):
                elapsed = time.time() - t_start
                rate = (ok + skip) / elapsed * 60 if elapsed > 0 else 0
                remaining = len(tasks) - total_done
                eta_min = remaining / (rate / 60) if rate > 0 else 0
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Progress: {total_done}/{len(tasks)} "
                      f"({100*total_done/len(tasks):.1f}%) | OK={ok} SKIP={skip} FAIL={fail} | "
                      f"{rate:.1f}/min | ETA: {eta_min/60:.1f}h")

    elapsed = time.time() - t_start
    print(f"\n{'='*80}")
    print(f"Sweep complete in {elapsed/3600:.1f}h")
    print(f"OK={ok}, SKIP={skip}, FAIL={fail}")

    # Aggregate and print results
    results = aggregate_results()
    print_table(results)

    # Save summary JSON
    summary_file = os.path.join(OUT_DIR, "_summary.json")
    summary = {}
    for card_name in CARDS:
        summary[card_name] = {}
        for mode_name in MODES:
            data_list = results[card_name].get(mode_name, [])
            if not data_list:
                continue
            daily_pnls = [d.get("total_pnl_dollars", 0) for d in data_list]
            all_trades_pnl = []
            for d in data_list:
                for t in d.get("trades", []):
                    all_trades_pnl.append(t.get("pnl_dollars", 0))
            summary[card_name][mode_name] = {
                "sharpe": compute_sharpe(daily_pnls),
                "total_pnl": sum(daily_pnls),
                "trades": sum(d.get("total_trades", 0) for d in data_list),
                "win_rate": sum(1 for p in all_trades_pnl if p > 0) / max(len(all_trades_pnl), 1),
                "days": len(data_list),
            }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
