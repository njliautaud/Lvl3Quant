#!/usr/bin/env python3
"""
targeted_card_optimizations.py — Data-driven optimization sweep for all cards.

Based on full_card_optimization.py analysis results, tests targeted improvements
using ONLY flags supported by fill_sim_cli:

  1. MAE exit at optimal cutoffs (--mae-exit-ticks + --mae-exit-hold-sec)
  2. Prime hours filter (--prime-hours, 10:30-14:30 ET)
  3. Higher signal thresholds (--signal-threshold)
  4. Shorter/longer hold times (--hold-ms)
  5. TP/SL adjustments (--take-profit-ticks, --stop-loss-ticks)
  6. Trailing stop variants (--trailing-ticks)
  7. Signal flip exit (--signal-flip-exit)
  8. Ratchet stop (--ratchet-stop)
  9. Combined MAE + prime hours, MAE + signal gate, etc.

NOTE: fill_sim does NOT support --side-filter or --exclude-hours.
Side analysis (long vs short) must be done post-hoc from trade-level JSON.

Run on Jupiter with 14 workers:
  nohup python3 targeted_card_optimizations.py 2>&1 | tee targeted_opt.log &

Output: /home/jupiter/Lvl3Quant/data/processed/card_targeted_opt/
"""

import os
import sys
import json
import time
import subprocess
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
ANALYSIS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/card_deep_analysis")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/card_targeted_opt")

OOT_START = "2025-12-01"
OOT_END = "2026-03-08"
WORKERS = 14

# ── Base Card Definitions ─────────────────────────────────────────────────────
CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "tp_ticks": 8,
        "sl_ticks": None,
        "hold_ms": 7200000,
        "conviction_bars": None,
        "conviction_mag": None,
    },
    "c3": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.3,
        "tp_ticks": 10,
        "sl_ticks": None,
        "hold_ms": 3600000,
        "conviction_bars": 100,
        "conviction_mag": 0.8,
    },
    "c4": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "sl_ticks": None,
        "hold_ms": 7200000,
        "conviction_bars": None,
        "conviction_mag": None,
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "signal_threshold": 0.1,
        "tp_ticks": None,
        "sl_ticks": None,
        "hold_ms": 3600000,
        "conviction_bars": None,
        "conviction_mag": None,
    },
    "c6": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "sl_ticks": 25,
        "hold_ms": 3600000,
        "conviction_bars": 60,
        "conviction_mag": 1.5,
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": None,
        "sl_ticks": 20,
        "hold_ms": 3600000,
        "conviction_bars": None,
        "conviction_mag": None,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("targeted_opt")


def build_optimization_grid():
    """Build data-driven optimization grid using only supported fill_sim flags.

    Supported flags:
      --mae-exit-ticks + --mae-exit-hold-sec
      --prime-hours
      --signal-threshold
      --hold-ms
      --take-profit-ticks, --stop-loss-ticks
      --trailing-ticks
      --signal-flip-exit
      --ratchet-stop
      --conviction-exit-bars, --conviction-exit-mag
    """

    variants = {}

    for card_id, card_def in CARDS.items():
        card_variants = []

        # Baseline
        card_variants.append({"label": "baseline", "overrides": {}})

        # --- 1. MAE EXIT SWEEPS (from analysis: optimal MAE cutoffs vary per card) ---
        # MAE exit requires both ticks AND hold time. Test combinations.
        for mae_ticks in [10, 15, 20, 25, 30, 40, 50]:
            for mae_hold_sec in [60, 300, 600]:
                label = f"mae{mae_ticks}t_{mae_hold_sec}s"
                card_variants.append({
                    "label": label,
                    "overrides": {
                        "mae_exit_ticks": mae_ticks,
                        "mae_exit_hold_sec": mae_hold_sec,
                    },
                })

        # --- 2. PRIME HOURS (10:30 AM - 2:30 PM ET) ---
        card_variants.append({
            "label": "prime_hours",
            "overrides": {"prime_hours": True},
        })

        # --- 3. SIGNAL THRESHOLD SWEEPS ---
        base_sig = card_def["signal_threshold"]
        for sig in [0.15, 0.2, 0.3, 0.5, 1.0, 1.5]:
            if sig != base_sig:
                card_variants.append({
                    "label": f"sig{sig}",
                    "overrides": {"signal_threshold": sig},
                })

        # --- 4. HOLD TIME SWEEPS ---
        base_hold = card_def["hold_ms"]
        for hold_ms in [300000, 600000, 900000, 1800000, 3600000, 7200000]:
            if hold_ms != base_hold:
                label = f"hold{hold_ms // 60000}m"
                card_variants.append({
                    "label": label,
                    "overrides": {"hold_ms": hold_ms},
                })

        # --- 5. TP ADJUSTMENTS ---
        base_tp = card_def["tp_ticks"]
        if base_tp is not None:
            for tp in [max(1, base_tp - 5), max(1, base_tp - 3), base_tp + 5, base_tp + 10, base_tp * 2]:
                if tp != base_tp and tp > 0:
                    card_variants.append({
                        "label": f"tp{tp}",
                        "overrides": {"tp_ticks": tp},
                    })
        else:
            for tp in [10, 15, 20, 30, 50]:
                card_variants.append({
                    "label": f"tp{tp}",
                    "overrides": {"tp_ticks": tp},
                })

        # --- 6. SL ADJUSTMENTS ---
        base_sl = card_def["sl_ticks"]
        if base_sl is not None:
            for sl in [10, 15, 20, 25, 30, 40]:
                if sl != base_sl:
                    card_variants.append({
                        "label": f"sl{sl}",
                        "overrides": {"sl_ticks": sl},
                    })
        else:
            for sl in [15, 20, 25, 30, 40, 50]:
                card_variants.append({
                    "label": f"sl{sl}",
                    "overrides": {"sl_ticks": sl},
                })

        # --- 7. TRAILING STOP (different from fixed SL) ---
        for trail in [10, 15, 20, 25, 30]:
            card_variants.append({
                "label": f"trail{trail}",
                "overrides": {"trailing_ticks": trail},
            })

        # --- 8. SIGNAL FLIP EXIT ---
        card_variants.append({
            "label": "flip_exit",
            "overrides": {"signal_flip_exit": True},
        })

        # --- 9. RATCHET STOP ---
        card_variants.append({
            "label": "ratchet",
            "overrides": {"ratchet_stop": True},
        })

        # --- 10. COMBINED: Best MAE + prime hours ---
        for mae_ticks in [15, 20, 25]:
            card_variants.append({
                "label": f"mae{mae_ticks}t_300s_prime",
                "overrides": {
                    "mae_exit_ticks": mae_ticks,
                    "mae_exit_hold_sec": 300,
                    "prime_hours": True,
                },
            })

        # --- 11. COMBINED: Higher signal + shorter hold ---
        for sig in [0.3, 0.5]:
            if sig != base_sig:
                for hold_ms in [1800000, 3600000]:
                    if hold_ms != base_hold:
                        card_variants.append({
                            "label": f"sig{sig}_hold{hold_ms//60000}m",
                            "overrides": {
                                "signal_threshold": sig,
                                "hold_ms": hold_ms,
                            },
                        })

        # --- 12. COMBINED: MAE + signal gate ---
        for mae_ticks in [20, 25]:
            for sig in [0.3, 0.5]:
                if sig != base_sig:
                    card_variants.append({
                        "label": f"mae{mae_ticks}t_300s_sig{sig}",
                        "overrides": {
                            "mae_exit_ticks": mae_ticks,
                            "mae_exit_hold_sec": 300,
                            "signal_threshold": sig,
                        },
                    })

        # --- 13. COMBINED: Ratchet + shorter hold ---
        for hold_ms in [1800000, 3600000]:
            if hold_ms != base_hold:
                card_variants.append({
                    "label": f"ratchet_hold{hold_ms//60000}m",
                    "overrides": {
                        "ratchet_stop": True,
                        "hold_ms": hold_ms,
                    },
                })

        # Deduplicate by label
        seen = set()
        deduped = []
        for v in card_variants:
            if v["label"] not in seen:
                seen.add(v["label"])
                deduped.append(v)
        variants[card_id] = deduped

    return variants


def discover_dates():
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        name = f.stem.split(".")[0]
        date_raw = name.split("-")[-1]
        if len(date_raw) == 8:
            date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            if OOT_START <= date_str <= OOT_END:
                dates.append(date_str)
    return dates


def build_task(card_id, card_def, date_str, variant):
    """Build fill_sim command with variant overrides."""
    date_compact = date_str.replace("-", "")
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date_str}_{card_def['pred_suffix']}.npz"

    if not mbo_file.exists() or not pred_file.exists():
        return None

    out_dir = OUTPUT_DIR / f"{card_id}_{variant['label']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"

    if out_file.exists():
        return None

    overrides = variant.get("overrides", {})

    sig = overrides.get("signal_threshold", card_def["signal_threshold"])
    hold = overrides.get("hold_ms", card_def["hold_ms"])
    tp = overrides.get("tp_ticks", card_def["tp_ticks"])
    sl = overrides.get("sl_ticks", card_def["sl_ticks"])

    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(sig),
        "--hold-ms", str(hold),
        "--quiet",
    ]

    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl is not None:
        cmd += ["--stop-loss-ticks", str(sl)]

    # Conviction exit from base card (always applied)
    conv_bars = card_def.get("conviction_bars")
    conv_mag = card_def.get("conviction_mag")
    if conv_bars is not None:
        cmd += ["--conviction-exit-bars", str(conv_bars)]
    if conv_mag is not None:
        cmd += ["--conviction-exit-mag", str(conv_mag)]

    # MAE exit
    if "mae_exit_ticks" in overrides:
        cmd += ["--mae-exit-ticks", str(overrides["mae_exit_ticks"])]
        cmd += ["--mae-exit-hold-sec", str(overrides.get("mae_exit_hold_sec", 300))]

    # Trailing stop
    if "trailing_ticks" in overrides:
        cmd += ["--trailing-ticks", str(overrides["trailing_ticks"])]

    # Boolean flags
    if overrides.get("prime_hours"):
        cmd += ["--prime-hours"]
    if overrides.get("signal_flip_exit"):
        cmd += ["--signal-flip-exit"]
    if overrides.get("ratchet_stop"):
        cmd += ["--ratchet-stop"]

    return {
        "cmd": cmd,
        "card": card_id,
        "date": date_str,
        "variant": variant["label"],
        "out_file": str(out_file),
    }


def run_task(task):
    try:
        result = subprocess.run(task["cmd"], capture_output=True, timeout=600)
        if result.returncode != 0:
            return {"status": "FAIL", **task, "error": result.stderr.decode(errors='replace')[:200]}
        return {"status": "OK", **task}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", **task}
    except Exception as e:
        return {"status": "ERROR", **task, "error": str(e)[:200]}


def aggregate_results(card_id, variant_label):
    result_dir = OUTPUT_DIR / f"{card_id}_{variant_label}"
    if not result_dir.exists():
        return None

    all_trades = []
    daily_pnls = []
    n_days = 0

    for f in sorted(result_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            n_days += 1
            pnl = data.get("total_pnl_dollars", 0)
            daily_pnls.append(pnl)
            trades = data.get("trades", [])
            all_trades.extend(trades)
        except Exception:
            continue

    if n_days == 0:
        return None

    total_pnl = sum(daily_pnls)
    n_trades = len(all_trades)
    wins = [t for t in all_trades if t.get("pnl_dollars", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl_dollars", 0) <= 0]

    daily_arr = np.array(daily_pnls)
    sharpe = float(np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)) if np.std(daily_arr) > 0 else 0

    gross_wins = sum(t.get("pnl_dollars", 0) for t in wins)
    gross_losses = abs(sum(t.get("pnl_dollars", 0) for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    cumulative = np.cumsum(daily_arr)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0

    exit_reasons = {}
    for t in all_trades:
        reason = t.get("exit_reason", "Unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Side breakdown
    longs = [t for t in all_trades if t.get("side") == "BUY"]
    shorts = [t for t in all_trades if t.get("side") == "SELL"]
    long_pnl = sum(t.get("pnl_dollars", 0) for t in longs)
    short_pnl = sum(t.get("pnl_dollars", 0) for t in shorts)

    return {
        "card": card_id,
        "variant": variant_label,
        "n_days": n_days,
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "daily_sharpe": round(sharpe, 2),
        "win_rate": round(len(wins) / n_trades * 100, 1) if n_trades > 0 else 0,
        "avg_win": round(gross_wins / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_losses / len(losses), 2) if losses else 0,
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "pos_days": sum(1 for p in daily_pnls if p > 0),
        "neg_days": sum(1 for p in daily_pnls if p <= 0),
        "exit_reasons": exit_reasons,
        "trades_per_day": round(n_trades / max(n_days, 1), 1),
        "long_pnl": round(long_pnl, 2),
        "short_pnl": round(short_pnl, 2),
        "n_longs": len(longs),
        "n_shorts": len(shorts),
    }


def print_results(all_summary):
    """Print sorted results per card."""
    print("\n" + "=" * 170)
    print("TARGETED OPTIMIZATION RESULTS — SORTED BY SHARPE (per card)")
    print("=" * 170)

    cards_seen = sorted(set(s["card"] for s in all_summary))

    for card_id in cards_seen:
        card_results = [s for s in all_summary if s["card"] == card_id]
        card_results.sort(key=lambda x: x["daily_sharpe"], reverse=True)

        baseline = next((s for s in card_results if s["variant"] == "baseline"), None)
        baseline_sharpe = baseline["daily_sharpe"] if baseline else 0

        print(f"\n{'─'*170}")
        print(f"  {card_id.upper()} (baseline Sharpe = {baseline_sharpe:.2f})")
        print(f"{'─'*170}")

        header = (
            f"{'Rank':>4} {'Variant':>25} | {'Days':>4} {'Trades':>6} {'T/Day':>5} "
            f"{'PnL':>10} {'Sharpe':>7} {'dS':>6} {'PF':>6} {'WR%':>5} "
            f"{'AvgW':>8} {'AvgL':>8} {'MaxDD':>9} | {'L$':>9} {'S$':>9} | Exit"
        )
        print(header)
        print("-" * 170)

        for rank, s in enumerate(card_results, 1):
            delta = s["daily_sharpe"] - baseline_sharpe
            delta_str = f"{delta:>+5.2f}"
            pf_str = f"{s['profit_factor']:>6.2f}" if s['profit_factor'] < 999 else "   inf"
            exits = ",".join(f"{k}:{v}" for k, v in sorted(s.get("exit_reasons", {}).items()))[:40]
            marker = " ***" if rank <= 3 and s["variant"] != "baseline" else ""
            print(
                f"{rank:>4} {s['variant']:>25} | {s['n_days']:>4} {s['n_trades']:>6} {s['trades_per_day']:>5.1f} "
                f"{s['total_pnl']:>10.2f} {s['daily_sharpe']:>7.2f} {delta_str} {pf_str} {s['win_rate']:>5.1f} "
                f"{s['avg_win']:>8.2f} {s['avg_loss']:>8.2f} {s['max_drawdown']:>9.2f} | "
                f"{s['long_pnl']:>9.2f} {s['short_pnl']:>9.2f} | {exits}{marker}"
            )

            if rank >= 25:
                remaining = len(card_results) - 25
                if remaining > 0:
                    print(f"  ... and {remaining} more variants")
                break

    # Cross-card top 15
    print(f"\n{'='*170}")
    print("TOP 15 IMPROVEMENTS OVERALL (sorted by Sharpe delta vs baseline)")
    print("=" * 170)

    baselines = {}
    for s in all_summary:
        if s["variant"] == "baseline":
            baselines[s["card"]] = s["daily_sharpe"]

    non_baseline = [s for s in all_summary if s["variant"] != "baseline"]
    for s in non_baseline:
        s["_delta"] = s["daily_sharpe"] - baselines.get(s["card"], 0)

    top15 = sorted(non_baseline, key=lambda x: x["_delta"], reverse=True)[:15]
    for rank, s in enumerate(top15, 1):
        print(
            f"  #{rank}: {s['card'].upper()} {s['variant']} — "
            f"Sharpe={s['daily_sharpe']:.2f} (delta={s['_delta']:+.2f}), "
            f"PnL=${s['total_pnl']:,.2f}, Trades={s['n_trades']}, "
            f"WR={s['win_rate']:.1f}%, PF={s['profit_factor']:.2f}, "
            f"L$={s['long_pnl']:,.2f}, S$={s['short_pnl']:,.2f}"
        )

    # Per-card BEST non-baseline
    print(f"\n{'='*170}")
    print("BEST OPTIMIZATION PER CARD")
    print("=" * 170)
    for card_id in cards_seen:
        card_non_bl = [s for s in non_baseline if s["card"] == card_id and s["n_trades"] >= 10]
        if card_non_bl:
            best = max(card_non_bl, key=lambda x: x["daily_sharpe"])
            bl_sharpe = baselines.get(card_id, 0)
            print(
                f"  {card_id.upper()}: {best['variant']} — "
                f"Sharpe {bl_sharpe:.2f} -> {best['daily_sharpe']:.2f} ({best['daily_sharpe']-bl_sharpe:+.2f}), "
                f"PnL ${best['total_pnl']:,.2f}, {best['n_trades']} trades"
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = discover_dates()
    log.info(f"Found {len(dates)} OOT dates: {dates[0]} to {dates[-1]}")

    variants = build_optimization_grid()
    total_variants = sum(len(v) for v in variants.values())
    log.info(f"Optimization grid: {total_variants} variants across {len(variants)} cards")
    for card_id, card_vars in variants.items():
        log.info(f"  {card_id}: {len(card_vars)} variants")

    # Build tasks
    tasks = []
    skipped = 0
    for card_id, card_vars in variants.items():
        card_def = CARDS[card_id]
        for variant in card_vars:
            for date_str in dates:
                task = build_task(card_id, card_def, date_str, variant)
                if task is not None:
                    tasks.append(task)
                else:
                    skipped += 1

    total_possible = total_variants * len(dates)
    log.info(f"Tasks: {len(tasks)} new + {skipped} skipped = {total_possible} total")
    log.info(f"Workers: {WORKERS}")

    if tasks:
        t0 = time.time()
        ok = fail = 0

        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result["status"] == "OK":
                    ok += 1
                else:
                    fail += 1
                    if fail <= 20:
                        log.warning(f"  FAIL: {result['card']} {result['date']} {result['variant']}: "
                                    f"{result.get('error', '')[:100]}")

                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed * 60
                    eta_min = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    log.info(
                        f"Progress: {i+1}/{len(tasks)} ({ok} OK, {fail} FAIL) | "
                        f"{rate:.0f}/min | ETA: {eta_min:.1f}min"
                    )

        log.info(f"\nExecution: {ok} OK, {fail} FAIL in {(time.time()-t0)/60:.1f}min")
    else:
        log.info("All tasks already completed!")

    # Aggregate
    log.info("\nAggregating results...")
    all_summary = []
    for card_id, card_vars in variants.items():
        for variant in card_vars:
            agg = aggregate_results(card_id, variant["label"])
            if agg:
                all_summary.append(agg)

    if all_summary:
        print_results(all_summary)

        summary_file = OUTPUT_DIR / "targeted_opt_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "oot_range": f"{OOT_START} to {OOT_END}",
                "n_variants_per_card": {k: len(v) for k, v in variants.items()},
                "total_variants": total_variants,
                "results": all_summary,
            }, f, indent=2, default=str)
        log.info(f"\nSaved: {summary_file}")
    else:
        log.warning("No results to aggregate!")


if __name__ == "__main__":
    main()
