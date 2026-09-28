#!/usr/bin/env python3
"""
FIFO Market Replay Validation — Vol-Conditioned Execution Parameters
====================================================================
Runs fill_sim_cli (Rust FIFO queue simulator) across all OOT prediction folds
with vol-conditioned params vs a generic baseline. Compares per-config & per-date.

Configs:
  - low_vol:  TP=12, SL=2,  hold=5000ms, threshold=0.8
  - med_vol:  TP=8,  SL=3,  hold=5000ms, threshold=0.3
  - high_vol: TP=12, SL=6,  hold=5000ms, threshold=0.3
  - baseline: TP=6,  SL=4,  hold=10000ms, threshold=0.5
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar"
MBO_DIR  = "/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR  = "/home/jupiter/Lvl3Quant/output/fifo_vol_validation"
VOL_TABLE = "/home/jupiter/Lvl3Quant/output/vol_conditioned_exec/vol_regime_lookup_table.json"

MAX_WORKERS = 16
TICK_VALUE = 12.50       # ES tick value
COMMISSION_RT = 4.70     # AMP round-trip commission per contract

# ── Configs ────────────────────────────────────────────────────────────
CONFIGS = {
    "low_vol": {
        "take_profit_ticks": 12,
        "stop_loss_ticks": 2,
        "hold_ms": 5000,
        "signal_threshold": 0.8,
    },
    "med_vol": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 3,
        "hold_ms": 5000,
        "signal_threshold": 0.3,
    },
    "high_vol": {
        "take_profit_ticks": 12,
        "stop_loss_ticks": 6,
        "hold_ms": 5000,
        "signal_threshold": 0.3,
    },
    "baseline": {
        "take_profit_ticks": 6,
        "stop_loss_ticks": 4,
        "hold_ms": 10000,
        "signal_threshold": 0.5,
    },
}


def discover_folds():
    """Find all fold prediction files and extract dates + matching MBO files."""
    folds = []
    for fname in sorted(os.listdir(PRED_DIR)):
        m = re.match(r"fold_(\d+)_oot_predictions\.npz", fname)
        if not m:
            continue
        fold_idx = int(m.group(1))
        pred_path = os.path.join(PRED_DIR, fname)

        # Extract date from oot_files key
        npz = np.load(pred_path, allow_pickle=True)
        oot_files = npz["oot_files"]
        # Path like: .../20260223_mbo_events.npz
        date_match = re.search(r"(\d{8})_mbo_events", str(oot_files[0]))
        if not date_match:
            print(f"  WARN: cannot parse date from fold {fold_idx}: {oot_files[0]}")
            continue
        date_str = date_match.group(1)

        # Find matching MBO file
        mbo_file = os.path.join(MBO_DIR, f"glbx-mdp3-{date_str}.mbo.dbn.zst")
        if not os.path.exists(mbo_file):
            print(f"  WARN: no MBO file for {date_str}, skipping fold {fold_idx}")
            continue

        folds.append({
            "fold_idx": fold_idx,
            "date": date_str,
            "pred_path": pred_path,
            "mbo_file": mbo_file,
        })
    return folds


def prepare_single_horizon_npz(pred_path, horizon_idx=2):
    """
    Extract a single horizon from the (N,3) predictions array.
    horizon_idx=2 corresponds to 10s horizon (columns: 1s, 5s, 10s).
    Returns path to temp .npz file.
    """
    npz = np.load(pred_path, allow_pickle=True)
    preds = npz["predictions"][:, horizon_idx].astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False, dir=OUT_DIR)
    np.savez(tmp.name, predictions=preds)
    tmp.close()
    return tmp.name


def run_single(task):
    """Run fill_sim_cli for one (fold, config) pair. Called in worker process."""
    fold = task["fold"]
    config_name = task["config_name"]
    config = task["config"]
    temp_pred = task["temp_pred"]

    output_json = os.path.join(
        OUT_DIR, f"fold_{fold['fold_idx']:02d}_{fold['date']}_{config_name}.json"
    )

    cmd = [
        FILL_SIM,
        "--mbo-file", fold["mbo_file"],
        "--predictions", temp_pred,
        "--output", output_json,
        "--take-profit-ticks", str(config["take_profit_ticks"]),
        "--stop-loss-ticks", str(config["stop_loss_ticks"]),
        "--hold-ms", str(config["hold_ms"]),
        "--signal-threshold", str(config["signal_threshold"]),
        "--prime-hours",
        "--quiet",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {
                "fold_idx": fold["fold_idx"],
                "date": fold["date"],
                "config": config_name,
                "error": result.stderr[:500],
            }

        # Parse output JSON
        if os.path.exists(output_json):
            with open(output_json) as f:
                data = json.load(f)
            data["fold_idx"] = fold["fold_idx"]
            data["date"] = fold["date"]
            data["config"] = config_name
            return data
        else:
            return {
                "fold_idx": fold["fold_idx"],
                "date": fold["date"],
                "config": config_name,
                "error": "no output file produced",
            }
    except subprocess.TimeoutExpired:
        return {
            "fold_idx": fold["fold_idx"],
            "date": fold["date"],
            "config": config_name,
            "error": "timeout (600s)",
        }
    except Exception as e:
        return {
            "fold_idx": fold["fold_idx"],
            "date": fold["date"],
            "config": config_name,
            "error": str(e),
        }


def compute_sortino(daily_pnls):
    """Sortino ratio from a list of daily PnL values."""
    if len(daily_pnls) < 2:
        return 0.0
    arr = np.array(daily_pnls)
    mean_ret = np.mean(arr)
    downside = arr[arr < 0]
    if len(downside) == 0:
        return 999.0 if mean_ret > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std == 0:
        return 999.0 if mean_ret > 0 else 0.0
    return mean_ret / downside_std * np.sqrt(252)


def compute_sharpe(daily_pnls):
    """Annualized Sharpe from daily PnL values."""
    if len(daily_pnls) < 2:
        return 0.0
    arr = np.array(daily_pnls)
    std = np.std(arr)
    if std == 0:
        return 999.0 if np.mean(arr) > 0 else 0.0
    return np.mean(arr) / std * np.sqrt(252)


def aggregate_results(all_results):
    """Aggregate per-config results across all dates."""
    # Group by config
    by_config = {}
    for r in all_results:
        cfg = r.get("config", "unknown")
        if cfg not in by_config:
            by_config[cfg] = []
        by_config[cfg].append(r)

    print("\n" + "=" * 100)
    print("FIFO MARKET REPLAY VALIDATION — VOL-CONDITIONED EXECUTION PARAMS")
    print("=" * 100)

    summary = {}

    for config_name in ["low_vol", "med_vol", "high_vol", "baseline"]:
        results = by_config.get(config_name, [])
        if not results:
            print(f"\n--- {config_name}: NO RESULTS ---")
            continue

        errors = [r for r in results if "error" in r]
        valid = [r for r in results if "error" not in r]

        if errors:
            print(f"\n  {config_name} ERRORS:")
            for e in errors:
                print(f"    fold {e['fold_idx']} ({e['date']}): {e['error']}")

        if not valid:
            print(f"\n--- {config_name}: ALL FAILED ---")
            continue

        # Aggregate metrics
        total_trades = 0
        total_wins = 0
        total_gross_profit = 0.0
        total_gross_loss = 0.0
        total_pnl = 0.0
        daily_pnls = []

        print(f"\n{'─' * 100}")
        print(f"  CONFIG: {config_name}")
        params = CONFIGS[config_name]
        print(f"  TP={params['take_profit_ticks']} SL={params['stop_loss_ticks']} "
              f"Hold={params['hold_ms']}ms Threshold={params['signal_threshold']}")
        print(f"{'─' * 100}")
        print(f"  {'Date':<12} {'Trades':>7} {'WR':>7} {'PF':>8} {'PnL($)':>10} {'AvgPnL':>8}")
        print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*8} {'─'*10} {'─'*8}")

        for r in sorted(valid, key=lambda x: x["date"]):
            n_trades = r.get("total_trades", r.get("n_trades", 0))
            wins = r.get("winning_trades", r.get("wins", 0))
            gross_profit = r.get("gross_profit", 0.0)
            gross_loss = abs(r.get("gross_loss", 0.0))
            pnl = r.get("net_pnl", r.get("total_pnl", r.get("pnl", 0.0)))

            # Handle case where pnl might be in ticks — check for dollars key
            if "net_pnl_dollars" in r:
                pnl = r["net_pnl_dollars"]
            elif "total_pnl_dollars" in r:
                pnl = r["total_pnl_dollars"]

            total_trades += n_trades
            total_wins += wins
            total_gross_profit += gross_profit
            total_gross_loss += gross_loss
            total_pnl += pnl
            daily_pnls.append(pnl)

            wr = wins / n_trades * 100 if n_trades > 0 else 0
            pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
            avg_pnl = pnl / n_trades if n_trades > 0 else 0

            print(f"  {r['date']:<12} {n_trades:>7} {wr:>6.1f}% {pf:>8.2f} {pnl:>10.2f} {avg_pnl:>8.2f}")

        # Totals
        overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        overall_pf = total_gross_profit / total_gross_loss if total_gross_loss > 0 else 999.0
        sharpe = compute_sharpe(daily_pnls)
        sortino = compute_sortino(daily_pnls)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Post-commission
        commission_total = total_trades * COMMISSION_RT
        net_after_comm = total_pnl - commission_total

        print(f"  {'─'*12} {'─'*7} {'─'*7} {'─'*8} {'─'*10} {'─'*8}")
        print(f"  {'TOTAL':<12} {total_trades:>7} {overall_wr:>6.1f}% {overall_pf:>8.2f} {total_pnl:>10.2f} {avg_pnl:>8.2f}")
        print(f"  Commission: ${commission_total:.2f} ({total_trades} trades x ${COMMISSION_RT})")
        print(f"  Net after commission: ${net_after_comm:.2f}")
        print(f"  Sharpe: {sharpe:.2f}  |  Sortino: {sortino:.2f}")
        print(f"  Avg daily PnL: ${np.mean(daily_pnls):.2f}  |  Days: {len(daily_pnls)}")

        summary[config_name] = {
            "total_trades": total_trades,
            "win_rate": overall_wr,
            "profit_factor": overall_pf,
            "total_pnl": total_pnl,
            "net_after_commission": net_after_comm,
            "sharpe": sharpe,
            "sortino": sortino,
            "n_days": len(daily_pnls),
            "avg_pnl_per_trade": avg_pnl,
            "daily_pnls": daily_pnls,
        }

    # ── Vol-conditioned composite vs baseline ──────────────────────────
    print("\n" + "=" * 100)
    print("VOL-CONDITIONED COMPOSITE vs BASELINE")
    print("=" * 100)

    vol_configs = ["low_vol", "med_vol", "high_vol"]
    vol_present = [c for c in vol_configs if c in summary]

    if vol_present and "baseline" in summary:
        # Composite: sum across vol regimes
        comp_trades = sum(summary[c]["total_trades"] for c in vol_present)
        comp_pnl = sum(summary[c]["total_pnl"] for c in vol_present)
        comp_comm = comp_trades * COMMISSION_RT
        comp_net = comp_pnl - comp_comm

        bl = summary["baseline"]
        bl_comm = bl["total_trades"] * COMMISSION_RT
        bl_net = bl["total_pnl"] - bl_comm

        # Combine daily PnLs for composite (sum across configs per day)
        # Since each config runs independently on every date, composite daily PnL
        # is the sum of all vol-regime PnLs for that date
        all_dates_pnl = {}
        for cfg_name in vol_present:
            valid_results = [r for r in by_config.get(cfg_name, []) if "error" not in r]
            for r in valid_results:
                d = r["date"]
                pnl_val = r.get("net_pnl", r.get("total_pnl", r.get("pnl", r.get("net_pnl_dollars", r.get("total_pnl_dollars", 0)))))
                if "net_pnl_dollars" in r:
                    pnl_val = r["net_pnl_dollars"]
                elif "total_pnl_dollars" in r:
                    pnl_val = r["total_pnl_dollars"]
                all_dates_pnl[d] = all_dates_pnl.get(d, 0) + pnl_val

        comp_daily = list(all_dates_pnl.values())
        comp_sharpe = compute_sharpe(comp_daily)
        comp_sortino = compute_sortino(comp_daily)

        print(f"\n  {'Metric':<25} {'Vol-Conditioned':>18} {'Baseline':>18} {'Delta':>18}")
        print(f"  {'─'*25} {'─'*18} {'─'*18} {'─'*18}")
        print(f"  {'Total Trades':<25} {comp_trades:>18} {bl['total_trades']:>18} {comp_trades - bl['total_trades']:>+18}")
        print(f"  {'Gross PnL ($)':<25} {comp_pnl:>18.2f} {bl['total_pnl']:>18.2f} {comp_pnl - bl['total_pnl']:>+18.2f}")
        print(f"  {'Net PnL ($)':<25} {comp_net:>18.2f} {bl_net:>18.2f} {comp_net - bl_net:>+18.2f}")
        print(f"  {'Sharpe':<25} {comp_sharpe:>18.2f} {bl['sharpe']:>18.2f} {comp_sharpe - bl['sharpe']:>+18.2f}")
        print(f"  {'Sortino':<25} {comp_sortino:>18.2f} {bl['sortino']:>18.2f} {comp_sortino - bl['sortino']:>+18.2f}")

        # Edge per trade
        comp_edge = comp_net / comp_trades if comp_trades > 0 else 0
        bl_edge = bl_net / bl["total_trades"] if bl["total_trades"] > 0 else 0
        print(f"  {'Edge/Trade ($)':<25} {comp_edge:>18.2f} {bl_edge:>18.2f} {comp_edge - bl_edge:>+18.2f}")

    # Save summary
    summary_clean = {}
    for k, v in summary.items():
        s = dict(v)
        s["daily_pnls"] = [float(x) for x in s["daily_pnls"]]
        summary_clean[k] = s

    summary_path = os.path.join(OUT_DIR, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_clean, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    return summary


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("FIFO Vol-Conditioned Validation")
    print(f"  fill_sim_cli: {FILL_SIM}")
    print(f"  Workers: {MAX_WORKERS}")
    print()

    # 1. Discover folds
    folds = discover_folds()
    print(f"Found {len(folds)} OOT folds with matching MBO files:")
    for f in folds:
        print(f"  Fold {f['fold_idx']:02d}: {f['date']}  MBO: {os.path.basename(f['mbo_file'])}")

    if not folds:
        print("ERROR: No valid folds found. Exiting.")
        sys.exit(1)

    # 2. Prepare single-horizon prediction files (10s = index 2)
    print("\nPreparing single-horizon (10s) prediction files...")
    temp_preds = {}
    for fold in folds:
        tmp_path = prepare_single_horizon_npz(fold["pred_path"], horizon_idx=2)
        temp_preds[fold["fold_idx"]] = tmp_path
        print(f"  Fold {fold['fold_idx']:02d}: {fold['pred_path']} -> {tmp_path}")

    # 3. Build task list: each fold x each config
    tasks = []
    for fold in folds:
        for config_name, config in CONFIGS.items():
            tasks.append({
                "fold": fold,
                "config_name": config_name,
                "config": config,
                "temp_pred": temp_preds[fold["fold_idx"]],
            })

    print(f"\nTotal tasks: {len(tasks)} ({len(folds)} folds x {len(CONFIGS)} configs)")
    print("Launching parallel execution...\n")

    # 4. Run in parallel
    all_results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            all_results.append(result)
            task = futures[future]
            status = "ERROR" if "error" in result else "OK"
            n_trades = result.get("total_trades", result.get("n_trades", "?"))
            print(f"  [{completed}/{len(tasks)}] fold={task['fold']['fold_idx']:02d} "
                  f"date={task['fold']['date']} config={task['config_name']:<10} "
                  f"status={status} trades={n_trades}")

    # 5. Aggregate and report
    summary = aggregate_results(all_results)

    # 6. Clean up temp files
    for tmp_path in temp_preds.values():
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
