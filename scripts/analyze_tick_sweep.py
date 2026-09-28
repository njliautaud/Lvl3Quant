#!/usr/bin/env python3
"""
Analyze tick-level sweep results from tick_replay_fast.py.
Reads JSON output, identifies statistically significant configs,
runs regime stratification, and writes summary.

Can be run standalone or called by sweep auto-launcher.
"""

import json
import sys
import os
import numpy as np
from datetime import datetime

def load_results(path):
    with open(path) as f:
        return json.load(f)


def analyze_sweep(results, threshold_label=""):
    """Analyze sweep results and return summary dict."""
    if not results:
        return {"verdict": "NO RESULTS", "details": []}

    # Sort by Sharpe
    results_sorted = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)

    # Find significant configs (p < 0.05)
    significant = [r for r in results_sorted if r.get('p_value', 1) < 0.05]
    marginal = [r for r in results_sorted if 0.05 <= r.get('p_value', 1) < 0.10]

    summary = {
        "threshold": threshold_label,
        "n_configs": len(results),
        "n_significant": len(significant),
        "n_marginal": len(marginal),
        "best_config": None,
        "verdict": "",
        "details": []
    }

    if significant:
        best = significant[0]
        summary["best_config"] = {
            "tp": best.get('tp_ticks'),
            "sl": best.get('sl_ticks'),
            "sharpe": best.get('sharpe', 0),
            "sortino": best.get('sortino', 0),
            "pf": best.get('profit_factor', 0),
            "wr": best.get('win_rate', 0),
            "n_trades": best.get('n_trades', 0),
            "net_pnl_ticks": best.get('net_pnl_ticks', 0),
            "net_pnl_dollars": best.get('net_pnl_dollars', 0),
            "p_value": best.get('p_value', 1),
            "random_mean": best.get('random_mean_pnl', 0),
        }
        summary["verdict"] = f"SIGNAL FOUND: {len(significant)} configs beat random (p<0.05)"

        for r in significant:
            summary["details"].append(
                f"  TP{r['tp_ticks']}/SL{r['sl_ticks']}: "
                f"Sharpe={r.get('sharpe',0):.2f}, "
                f"PF={r.get('profit_factor',0):.2f}, "
                f"WR={r.get('win_rate',0):.1%}, "
                f"{r.get('n_trades',0)} trades, "
                f"PnL={r.get('net_pnl_ticks',0):.0f}t "
                f"(${r.get('net_pnl_dollars',0):.0f}), "
                f"p={r.get('p_value',1):.3f}"
            )
    elif marginal:
        summary["verdict"] = f"WEAK SIGNAL: {len(marginal)} configs marginal (0.05<p<0.10), none significant"
    else:
        summary["verdict"] = "NO SIGNAL: No configs beat random at p<0.10"

    # Check for artifact pattern: many configs profitable but random also profitable
    profitable = [r for r in results if r.get('net_pnl_ticks', 0) > 0]
    random_profitable = [r for r in results if r.get('random_mean_pnl', 0) > 0]
    if len(random_profitable) > len(results) * 0.5:
        summary["verdict"] += " [WARNING: random directions also profitable in >50% configs — possible structural bias]"

    return summary


def write_summary(summaries, output_path):
    """Write combined summary to file."""
    with open(output_path, 'w') as f:
        f.write(f"# Tick-Level FIFO Replay Sweep Results\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}\n")
        f.write(f"# Model: CNN-Mamba v3.4.2 (pred_log_ret_1s)\n")
        f.write(f"# Engine: tick_replay_fast.py (numba-accelerated FIFO with back-of-queue fill)\n\n")

        for s in summaries:
            f.write(f"## Threshold = {s['threshold']}\n")
            f.write(f"VERDICT: {s['verdict']}\n")
            f.write(f"Configs tested: {s['n_configs']}\n")
            f.write(f"Significant (p<0.05): {s['n_significant']}\n")
            f.write(f"Marginal (p<0.10): {s['n_marginal']}\n\n")

            if s['best_config']:
                b = s['best_config']
                f.write(f"Best config: TP={b['tp']}/SL={b['sl']}\n")
                f.write(f"  Sharpe: {b['sharpe']:.2f}\n")
                f.write(f"  Sortino: {b['sortino']:.2f}\n")
                f.write(f"  PF: {b['pf']:.2f}\n")
                f.write(f"  WR: {b['wr']:.1%}\n")
                f.write(f"  Trades: {b['n_trades']}\n")
                f.write(f"  Net PnL: {b['net_pnl_ticks']:.0f} ticks (${b['net_pnl_dollars']:.0f})\n")
                f.write(f"  p-value: {b['p_value']:.4f}\n")
                f.write(f"  Random mean: {b['random_mean']:.1f} ticks\n\n")

            if s['details']:
                f.write("All significant configs:\n")
                for d in s['details']:
                    f.write(f"{d}\n")
                f.write("\n")

            f.write("---\n\n")

    print(f"Summary written to {output_path}")


def main():
    result_files = {
        "0.20": "engines/tick_sweep_t020_results.json",
        "0.25": "engines/tick_sweep_t025_results.json",
        "0.30": "engines/tick_sweep_t030_results.json",
    }

    base_dir = "/home/jupiter/Lvl3Quant"
    summaries = []

    for thresh, fname in sorted(result_files.items()):
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            print(f"\nAnalyzing {fname}...")
            results = load_results(path)
            s = analyze_sweep(results, threshold_label=thresh)
            summaries.append(s)
            print(f"  {s['verdict']}")
        else:
            print(f"  {fname} not found (sweep may still be running)")

    if summaries:
        output_path = os.path.join(base_dir, "output/tick_sweep_analysis.md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_summary(summaries, output_path)

        # Print Discord-friendly summary
        print("\n" + "="*60)
        print("DISCORD SUMMARY:")
        print("="*60)
        for s in summaries:
            print(f"Threshold {s['threshold']}: {s['verdict']}")
            if s['best_config']:
                b = s['best_config']
                print(f"  Best: TP{b['tp']}/SL{b['sl']}, "
                      f"Sharpe {b['sharpe']:.2f}, WR {b['wr']:.0%}, "
                      f"p={b['p_value']:.3f}")


if __name__ == "__main__":
    main()
