#!/usr/bin/env python3
"""
Fillsim Validation of Data-Driven Exit Rules (HC #49)
=====================================================
Validates the deployed paper trading exit config through Rust fill_sim_cli.

Exit rules derived from MFE/MAE analysis (250K OOT predictions):
  - Mean MFE: 4.37 ticks → trailing stop activation at 4t
  - Mean MAE: 0.71 ticks → stop loss at 2-4t range
  - Alpha exhaustion: ~30-60s → max hold 120s
  - Signal decay: exit if |z-score| < 0.5

Compares deployed config vs baselines to quantify exit rule impact.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR    = Path("/home/jupiter/Lvl3Quant")
FILL_SIM    = BASE_DIR / "rust_cache_builder/target/release/fill_sim_cli"
PRED_DIR    = BASE_DIR / "data/processed/fillsim_cnn_mamba_v2/per_day_preds"
RAW_MBO_DIR = BASE_DIR / "data/raw/mbo"
OUTPUT_DIR  = BASE_DIR / "execution/results/exit_rule_validation"

# OOT dates from CNN-Mamba v2 walk-forward
OOT_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304", "20260305"
]

# Configurations to compare
CONFIGS = {
    "baseline_signal_flip": {
        "desc": "Signal-flip only (no hard exit rules)",
        "args": [
            "--signal-flip-exit",
            "--hold-ms", "600000",  # 10 min fallback
            "--signal-threshold", "0.5",
            "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
            "--prime-hours",
            "--latency-ms", "10",
        ]
    },
    "deployed_v1": {
        "desc": "Deployed config: SL=3t, hold=120s, trail=4t/2t, signal-flip",
        "args": [
            "--signal-flip-exit",
            "--hold-ms", "120000",       # max hold 120s (data-driven: alpha dies by 60s)
            "--stop-loss-ticks", "3",     # SL=3t (Top1% best: SL_3t PF=1.97)
            "--trailing-ticks", "2",      # trail offset 2t after MFE≥4t
            "--signal-threshold", "0.5",
            "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
            "--prime-hours",
            "--latency-ms", "10",
        ]
    },
    "deployed_tight_sl": {
        "desc": "Tighter SL=2t (Top1% best PF=2.24), hold=120s",
        "args": [
            "--signal-flip-exit",
            "--hold-ms", "120000",
            "--stop-loss-ticks", "2",
            "--trailing-ticks", "2",
            "--signal-threshold", "0.5",
            "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
            "--prime-hours",
            "--latency-ms", "10",
        ]
    },
    "deployed_wide_sl": {
        "desc": "Wider SL=4t (Top1% best WR=42.7%), hold=120s",
        "args": [
            "--signal-flip-exit",
            "--hold-ms", "120000",
            "--stop-loss-ticks", "4",
            "--trailing-ticks", "2",
            "--signal-threshold", "0.5",
            "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
            "--prime-hours",
            "--latency-ms", "10",
        ]
    },
    "ratchet_stop": {
        "desc": "Ratcheting stop (MFE-based progressive trailing), hold=120s",
        "args": [
            "--signal-flip-exit",
            "--hold-ms", "120000",
            "--stop-loss-ticks", "3",
            "--ratchet-stop",
            "--signal-threshold", "0.5",
            "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
            "--prime-hours",
            "--latency-ms", "10",
        ]
    },
    "conviction_exit": {
        "desc": "Conviction exit (delayed signal flip, 10 bars), SL=3t, hold=120s",
        "args": [
            "--conviction-exit-bars", "100",   # 10 seconds
            "--conviction-exit-mag", "0.5",    # require |z| >= 0.5 to count
            "--hold-ms", "120000",
            "--stop-loss-ticks", "3",
            "--trailing-ticks", "2",
            "--signal-threshold", "0.5",
            "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
            "--prime-hours",
            "--latency-ms", "10",
        ]
    },
}

# Confidence tiers (raw prediction units, NOT z-scores)
# p95 ≈ 0.75, p99 ≈ 1.0 across dates
THRESHOLDS = {
    "top25pct": "0.3",    # ~top 25% of predictions
    "top5pct":  "0.5",    # ~top 5%
    "top1pct":  "0.8",    # ~top 1%
    "top0.5pct": "1.0",   # ~top 0.5%
}


def run_sim(date_str: str, config_name: str, config_args: list, threshold: str = "2.0") -> dict | None:
    """Run fill_sim_cli for one date with given config."""
    pred_path = PRED_DIR / f"{date_str}_preds.npz"
    raw_mbo = RAW_MBO_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"

    if not pred_path.exists():
        return None
    if not raw_mbo.exists():
        return None

    out_dir = OUTPUT_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}_result.json"

    # Build command — override threshold
    cmd = [str(FILL_SIM), "--mbo-file", str(raw_mbo), "--predictions", str(pred_path), "--output", str(out_path)]

    # Add config args but override signal-threshold
    args_copy = list(config_args)
    # Remove existing threshold if any
    for i in range(len(args_copy) - 1):
        if args_copy[i] == "--signal-threshold":
            args_copy[i+1] = threshold
            break
    else:
        args_copy.extend(["--signal-threshold", threshold])

    cmd.extend(args_copy)
    cmd.append("--quiet")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"    FAIL {date_str}: {result.stderr[:200]}")
            return None
        if out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"    ERROR {date_str}: {e}")
    return None


def summarize_results(results: list[dict]) -> dict:
    """Summarize across multiple days."""
    if not results:
        return {}

    total_pnl = sum(r.get("total_pnl_dollars", 0) for r in results)
    total_trades = sum(r.get("total_fills", r.get("total_trades", 0)) for r in results)
    total_signals = sum(r.get("total_signals", 0) for r in results)

    daily_pnls = [r.get("total_pnl_dollars", 0) for r in results]
    avg_daily = np.mean(daily_pnls) if daily_pnls else 0
    std_daily = np.std(daily_pnls) if len(daily_pnls) > 1 else 1

    win_rates = [r.get("win_rate_pct", 0) for r in results if r.get("total_fills", r.get("total_trades", 0)) > 0]
    avg_wr = np.mean(win_rates) if win_rates else 0

    profit_factors = [r.get("profit_factor", 0) for r in results if (r.get("profit_factor") or 0) > 0]
    avg_pf = np.mean(profit_factors) if profit_factors else 0

    sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

    # Sortino
    downside = [p for p in daily_pnls if p < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_daily
    sortino = (avg_daily / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    pos_days = sum(1 for p in daily_pnls if p > 0)

    fill_rates = [r.get("fill_rate_pct", r.get("fill_rate", 0) * 100 if "fill_rate" in r else 0) for r in results]
    avg_fill_rate = np.mean(fill_rates) if fill_rates else 0

    return {
        "total_pnl": total_pnl,
        "avg_daily_pnl": avg_daily,
        "total_trades": total_trades,
        "total_signals": total_signals,
        "avg_win_rate": avg_wr,
        "avg_profit_factor": avg_pf,
        "sharpe": sharpe,
        "sortino": sortino,
        "pos_days": f"{pos_days}/{len(daily_pnls)}",
        "avg_fill_rate": avg_fill_rate,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check which prediction files exist
    available = [d for d in OOT_DATES if (PRED_DIR / f"{d}_preds.npz").exists()]
    print(f"Available prediction files: {len(available)}/{len(OOT_DATES)}")
    if not available:
        print("No prediction files found! Run deep_pred_to_fillsim.py first with --folds 0-9")
        sys.exit(1)
    print(f"  Dates: {', '.join(available)}")
    print()

    # Run all configs across all dates
    all_summaries = {}

    for config_name, config in CONFIGS.items():
        print(f"{'='*60}")
        print(f"Config: {config_name}")
        print(f"  {config['desc']}")
        print(f"{'='*60}")

        results = []
        for date_str in available:
            r = run_sim(date_str, config_name, config["args"])
            if r:
                pnl = r.get("total_pnl_dollars", 0)
                trades = r.get("total_fills", r.get("total_trades", 0))
                wr = r.get("win_rate_pct", 0)
                print(f"    {date_str}: PnL=${pnl:>8,.0f} | Trades={trades:>4} | WR={wr:.1f}%")
                results.append(r)
            else:
                print(f"    {date_str}: skipped")

        summary = summarize_results(results)
        all_summaries[config_name] = {
            "desc": config["desc"],
            "summary": summary,
            "per_day": results,
        }

        if summary:
            print(f"\n  SUMMARY: PnL=${summary['total_pnl']:>10,.0f} | "
                  f"Trades={summary['total_trades']} | "
                  f"WR={summary['avg_win_rate']:.1f}% | "
                  f"PF={summary['avg_profit_factor']:.2f} | "
                  f"Sharpe={summary['sharpe']:.2f} | "
                  f"Sortino={summary['sortino']:.2f} | "
                  f"Pos={summary['pos_days']}")
        print()

    # Confidence tier sweep on best config
    print(f"\n{'='*60}")
    print("CONFIDENCE TIER SWEEP (deployed_v1)")
    print(f"{'='*60}")

    for tier_name, threshold in THRESHOLDS.items():
        results = []
        for date_str in available:
            r = run_sim(date_str, f"tier_{tier_name}", CONFIGS["deployed_v1"]["args"], threshold=threshold)
            if r:
                results.append(r)

        summary = summarize_results(results)
        all_summaries[f"tier_{tier_name}"] = {"summary": summary}

        if summary:
            print(f"  {tier_name}: PnL=${summary['total_pnl']:>10,.0f} | "
                  f"Trades={summary['total_trades']} | "
                  f"WR={summary['avg_win_rate']:.1f}% | "
                  f"Sortino={summary['sortino']:.2f} | "
                  f"Fill={summary['avg_fill_rate']:.1f}%")

    # Save full results
    out_file = OUTPUT_DIR / "validation_results.json"
    with open(out_file, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
