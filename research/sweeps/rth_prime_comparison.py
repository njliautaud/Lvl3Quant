#!/usr/bin/env python3
"""RTH vs Prime Hours Comparison for all 4 cards.

Runs each card config with and without --prime-hours flag across all OOT dates.
The fill_sim already restricts to RTH (9:30-4:00 ET).
--prime-hours further restricts to 10:30 AM - 2:30 PM ET.

This comparison shows whether the volatile open/close periods help or hurt.

Card1: book_predstdExit_conv1.5_vol50, sig=0.1, TP8, hold 1hr
Card2: book_predstdExit_conv1.5_vol50, sig=0.1, TP15, hold 1hr
Card3: raw_smoothExit_conv0.05_ethr0.0_vol70, sig=0.5, TP15, hold 1hr
Card4: book_predstdExit_conv2.0_vol70, sig=0.5, TP20, hold 1hr

8 configs (4 cards x 2 modes) x 54 dates = 432 jobs, 14 workers
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
OUT_BASE = LVL3_ROOT / "data" / "processed" / "rth_validation"

TICK_VALUE = 12.50
COMMISSION_RT = 4.70

CARDS = [
    {
        "name": "Card1",
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "take_profit": 8,
        "hold_ms": 3600000,
    },
    {
        "name": "Card2",
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "take_profit": 15,
        "hold_ms": 3600000,
    },
    {
        "name": "Card3",
        "pred_suffix": "raw_smoothExit_conv0.05_ethr0.0_vol70",
        "signal_threshold": 0.5,
        "take_profit": 15,
        "hold_ms": 3600000,
    },
    {
        "name": "Card4",
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.5,
        "take_profit": 20,
        "hold_ms": 3600000,
    },
]


def find_dates_for_card(pred_suffix):
    """Find all dates with both prediction and MBO files."""
    preds = {}
    mbo_dates = set()
    for f in sorted(PRED_DIR.glob(f"*_{pred_suffix}.npz")):
        date = f.stem[:10]
        preds[date] = f
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        nodash = f.name.split("-")[2].split(".")[0]
        date = f"{nodash[:4]}-{nodash[4:6]}-{nodash[6:8]}"
        mbo_dates.add(date)
    dates = sorted(set(preds.keys()) & mbo_dates)
    return dates, preds


def get_mbo_path(date):
    nodash = date.replace("-", "")
    return MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"


def run_sim(cmd, out_path, label):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and Path(out_path).exists():
            return (label, True, None)
        return (label, False, r.stderr[:300] if r.stderr else f"rc={r.returncode}")
    except subprocess.TimeoutExpired:
        return (label, False, "TIMEOUT")
    except Exception as e:
        return (label, False, str(e)[:200])


def build_cmd(date, pred_path, out_path, card, prime_hours=False):
    cmd = [
        str(BINARY),
        "--mbo-file", str(get_mbo_path(date)),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", str(card["signal_threshold"]),
        "--hold-ms", str(card["hold_ms"]),
        "--take-profit-ticks", str(card["take_profit"]),
        "--max-wait-bars", "50",
        "--latency-ms", "50",
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--quiet",
    ]
    if prime_hours:
        cmd.append("--prime-hours")
    return cmd


def analyze_results(result_dir, card_name, mode):
    """Aggregate results from all dates for a card/mode."""
    all_trades = []
    daily_pnl = []
    n_days = 0
    total_signals = 0
    total_filled = 0
    total_cancelled = 0

    for f in sorted(result_dir.glob("*.json")):
        try:
            d = json.load(open(f))
        except:
            continue
        n_days += 1
        day_pnl = d.get("total_pnl_dollars", 0)
        daily_pnl.append(day_pnl)
        total_signals += d.get("total_signals", 0)
        total_filled += d.get("total_filled", 0)
        total_cancelled += d.get("total_cancelled", 0)
        for t in d.get("trades", []):
            all_trades.append(t)

    if not all_trades or n_days == 0:
        return {
            "card": card_name, "mode": mode, "n_days": n_days,
            "trades": 0, "sharpe": 0, "net_pnl": 0, "wr": 0, "pf": 0,
            "avg_win": 0, "avg_loss": 0, "trades_per_day": 0,
            "avg_hold_min": 0, "avg_mae": 0, "avg_mfe": 0,
            "fill_rate": 0, "total_signals": total_signals,
        }

    wins = [t["pnl_dollars"] for t in all_trades if t["pnl_dollars"] > 0]
    losses = [t["pnl_dollars"] for t in all_trades if t["pnl_dollars"] <= 0]
    total_pnl = sum(t["pnl_dollars"] for t in all_trades)
    wr = len(wins) / len(all_trades) * 100 if all_trades else 0
    avg_win = statistics.mean(wins) if wins else 0
    avg_loss = statistics.mean(losses) if losses else 0
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe: annualized daily PnL
    if len(daily_pnl) > 1 and statistics.stdev(daily_pnl) > 0:
        sharpe = (statistics.mean(daily_pnl) / statistics.stdev(daily_pnl)) * math.sqrt(252)
    else:
        sharpe = 0

    # Average hold time in minutes
    hold_times = []
    for t in all_trades:
        if "hold_duration_ns" in t and t["hold_duration_ns"]:
            hold_times.append(t["hold_duration_ns"] / 1e9 / 60)
    avg_hold = statistics.mean(hold_times) if hold_times else 0

    # MAE/MFE
    mae_vals = [t.get("mae_ticks", 0) for t in all_trades]
    mfe_vals = [t.get("mfe_ticks", 0) for t in all_trades]
    avg_mae = statistics.mean(mae_vals) if mae_vals else 0
    avg_mfe = statistics.mean(mfe_vals) if mfe_vals else 0

    fill_rate = total_filled / total_signals * 100 if total_signals > 0 else 0

    return {
        "card": card_name,
        "mode": mode,
        "n_days": n_days,
        "trades": len(all_trades),
        "sharpe": round(sharpe, 2),
        "net_pnl": round(total_pnl, 2),
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "trades_per_day": round(len(all_trades) / n_days, 1),
        "avg_hold_min": round(avg_hold, 1),
        "avg_mae": round(avg_mae, 2),
        "avg_mfe": round(avg_mfe, 2),
        "fill_rate": round(fill_rate, 1),
        "total_signals": total_signals,
        "max_dd": round(max_drawdown(daily_pnl), 2) if daily_pnl else 0,
        "pos_days": sum(1 for p in daily_pnl if p > 0),
    }


def max_drawdown(daily_pnl):
    """Max drawdown in dollars from daily PnL series."""
    cumulative = 0
    peak = 0
    max_dd = 0
    for p in daily_pnl:
        cumulative += p
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    return max_dd


def main():
    t_start = time.time()

    # Build all jobs
    jobs = []
    for card in CARDS:
        dates, preds = find_dates_for_card(card["pred_suffix"])
        print(f"{card['name']}: {len(dates)} dates found")

        for mode_name, use_prime in [("full_rth", False), ("prime_only", True)]:
            out_dir = OUT_BASE / card["name"] / mode_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for date in dates:
                pred_path = preds[date]
                out_path = out_dir / f"{date}.json"

                # Skip if already done
                if out_path.exists():
                    try:
                        d = json.load(open(out_path))
                        if "total_pnl_dollars" in d:
                            continue
                    except:
                        pass

                cmd = build_cmd(date, pred_path, out_path, card, prime_hours=use_prime)
                label = f"{card['name']}_{mode_name}_{date}"
                jobs.append((cmd, str(out_path), label))

    print(f"\n{len(jobs)} jobs to run with {WORKERS} workers")
    if not jobs:
        print("All jobs already complete!")
    else:
        # Run jobs
        done = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(run_sim, cmd, out, label): label
                       for cmd, out, label in jobs}
            for fut in as_completed(futures):
                label, ok, err = fut.result()
                done += 1
                if not ok:
                    failed += 1
                    print(f"  FAIL [{done}/{len(jobs)}] {label}: {err}")
                elif done % 50 == 0 or done == len(jobs):
                    print(f"  [{done}/{len(jobs)}] done...")

        print(f"\nCompleted: {done - failed}/{done} succeeded, {failed} failed")

    # Analyze and compare
    print("\n" + "=" * 120)
    print("RTH (9:30-4:00) vs PRIME HOURS (10:30-2:30) COMPARISON")
    print("=" * 120)

    results = []
    for card in CARDS:
        for mode in ["full_rth", "prime_only"]:
            result_dir = OUT_BASE / card["name"] / mode
            r = analyze_results(result_dir, card["name"], mode)
            results.append(r)

    # Print side-by-side comparison
    header = f"{'Card':<8} {'Mode':<12} {'Days':>5} {'Trades':>7} {'Sharpe':>7} {'Net PnL':>10} {'WR%':>6} {'PF':>6} {'AvgWin':>8} {'AvgLoss':>8} {'T/Day':>6} {'Hold':>6} {'MAE':>6} {'MFE':>6} {'Fill%':>6} {'MaxDD':>8} {'Pos%':>5}"
    print(header)
    print("-" * len(header))

    for i in range(0, len(results), 2):
        rth = results[i]
        prime = results[i + 1]
        for r in [rth, prime]:
            pos_pct = round(r["pos_days"] / r["n_days"] * 100, 0) if r["n_days"] > 0 else 0
            print(f"{r['card']:<8} {r['mode']:<12} {r['n_days']:>5} {r['trades']:>7} {r['sharpe']:>7.2f} {r['net_pnl']:>10.0f} {r['wr']:>5.1f}% {r['pf']:>6.2f} {r['avg_win']:>8.1f} {r['avg_loss']:>8.1f} {r['trades_per_day']:>6.1f} {r['avg_hold_min']:>5.0f}m {r['avg_mae']:>6.2f} {r['avg_mfe']:>6.2f} {r['fill_rate']:>5.1f}% {r['max_dd']:>8.0f} {pos_pct:>4.0f}%")

        # Print delta
        if rth["trades"] > 0 and prime["trades"] > 0:
            d_sharpe = prime["sharpe"] - rth["sharpe"]
            d_pnl = prime["net_pnl"] - rth["net_pnl"]
            d_wr = prime["wr"] - rth["wr"]
            d_mae = prime["avg_mae"] - rth["avg_mae"]
            d_mfe = prime["avg_mfe"] - rth["avg_mfe"]
            d_tpd = prime["trades_per_day"] - rth["trades_per_day"]
            direction = "BETTER" if d_sharpe > 0 else "WORSE" if d_sharpe < 0 else "SAME"
            print(f"  >> DELTA: Sharpe {d_sharpe:+.2f} ({direction}), PnL {d_pnl:+.0f}, WR {d_wr:+.1f}%, MAE {d_mae:+.2f}, MFE {d_mfe:+.2f}, T/Day {d_tpd:+.1f}")
        print()

    # Save summary
    summary_path = OUT_BASE / "rth_vs_prime_summary.json"
    json.dump(results, open(summary_path, "w"), indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
