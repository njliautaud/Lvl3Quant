#!/usr/bin/env python3
"""
Tight Stop-Loss Sweep — FIFO Fill Sim
======================================
Tests tighter SL configurations to find profitable TP/SL ratios.
Hypothesis: tighter SL improves PF even if WR drops (avg loss shrinks).
Reuses existing pred_npzs from optimizer_gate_validation.
"""

import os
import sys
import json
import subprocess
import datetime
import numpy as np
from pathlib import Path

# ============ PATHS ============
BASE = Path("/home/nick/Lvl3Quant")
MBO_DIR = BASE / "data" / "raw" / "mbo"
FILL_SIM = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
PRED_NPZ_DIR = BASE / "output" / "optimizer_gate_validation" / "pred_npzs"
OUTPUT_DIR = BASE / "output" / "tight_sl_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OOT dates (from existing pred_npzs)
OOT_DATES = sorted([
    f.stem.replace("_unfiltered", "")
    for f in PRED_NPZ_DIR.glob("*_unfiltered.npz")
])

# ============ SWEEP CONFIGS ============
CONFIGS = {
    "TP8_SL8_30m": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 8,
        "hold_ms": 1800000,
        "signal_threshold": 0.3,
    },
    "TP8_SL10_30m": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 10,
        "hold_ms": 1800000,
        "signal_threshold": 0.3,
    },
    "TP8_SL12_30m": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 12,
        "hold_ms": 1800000,
        "signal_threshold": 0.3,
    },
    "TP6_SL6_15m": {
        "take_profit_ticks": 6,
        "stop_loss_ticks": 6,
        "hold_ms": 900000,
        "signal_threshold": 0.3,
    },
    "TP6_SL8_15m": {
        "take_profit_ticks": 6,
        "stop_loss_ticks": 8,
        "hold_ms": 900000,
        "signal_threshold": 0.3,
    },
    "TP4_SL4_10m": {
        "take_profit_ticks": 4,
        "stop_loss_ticks": 4,
        "hold_ms": 600000,
        "signal_threshold": 0.3,
    },
    "TP8_SL8_30m_sig50": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 8,
        "hold_ms": 1800000,
        "signal_threshold": 0.5,
    },
    "TP8_SL8_30m_sig70": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 8,
        "hold_ms": 1800000,
        "signal_threshold": 0.7,
    },
}

# Also run the baseline for comparison
BASELINE = {
    "TP8_SL16_30m_baseline": {
        "take_profit_ticks": 8,
        "stop_loss_ticks": 16,
        "hold_ms": 1800000,
        "signal_threshold": 0.3,
    },
}
CONFIGS = {**BASELINE, **CONFIGS}


def run_fillsim(mbo_path, pred_npz_path, output_path, config):
    """Run fill_sim_cli and return parsed results."""
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz_path),
        "--output", str(output_path),
        "--signal-threshold", str(config["signal_threshold"]),
        "--take-profit-ticks", str(config["take_profit_ticks"]),
        "--stop-loss-ticks", str(config["stop_loss_ticks"]),
        "--hold-ms", str(config["hold_ms"]),
        "--latency-ms", "10",
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  fill_sim ERROR: {result.stderr[:200]}", flush=True)
            return None
        if os.path.exists(output_path):
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"  fill_sim EXCEPTION: {e}", flush=True)
    return None


def compute_metrics(results_list):
    """Compute aggregate metrics across all days."""
    if not results_list:
        return {}

    all_trade_pnls = []
    total_pnl = 0
    total_trades = 0
    total_signals = 0
    total_filled = 0
    daily_pnls = []

    for r in results_list:
        pnl = r.get("total_pnl_dollars", 0)
        total_pnl += pnl
        total_trades += r.get("total_trades", 0)
        total_signals += r.get("total_signals", 0)
        total_filled += r.get("total_filled", 0)
        daily_pnls.append(pnl)
        for t in r.get("trades", []):
            all_trade_pnls.append(t.get("pnl_dollars", 0))

    if not all_trade_pnls:
        return {"total_pnl_dollars": total_pnl, "n_trades": 0}

    arr = np.array(all_trade_pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    mean_pnl = arr.mean()
    std_pnl = arr.std() if len(arr) > 1 else 1.0
    sharpe_trade = mean_pnl / std_pnl if std_pnl > 0 else 0

    daily_arr = np.array(daily_pnls)
    daily_sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if len(daily_arr) > 1 and daily_arr.std() > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = len(wins) / len(arr)

    net_ticks = total_pnl / 12.50
    ticks_per_trade = (total_pnl / 12.50) / len(arr) if len(arr) > 0 else 0

    downside = arr[arr < 0]
    downside_std = downside.std() if len(downside) > 1 else 1.0
    sortino = mean_pnl / downside_std if downside_std > 0 else 0

    return {
        "total_pnl_dollars": round(total_pnl, 2),
        "n_trades": total_trades,
        "n_signals": total_signals,
        "n_filled": total_filled,
        "fill_rate": round(total_filled / total_signals, 4) if total_signals > 0 else 0,
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "sharpe_per_trade": round(sharpe_trade, 4),
        "daily_sharpe_ann": round(daily_sharpe, 2),
        "sortino_per_trade": round(sortino, 4),
        "net_ticks": round(net_ticks, 2),
        "ticks_per_trade": round(ticks_per_trade, 4),
        "avg_win_dollars": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss_dollars": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "n_days": len(results_list),
        "n_green_days": int((daily_arr > 0).sum()),
        "n_red_days": int((daily_arr < 0).sum()),
    }


def main():
    print("=" * 70, flush=True)
    print("TIGHT STOP-LOSS SWEEP — FIFO Fill Sim", flush=True)
    print(f"OOT dates: {len(OOT_DATES)} ({OOT_DATES[0]} to {OOT_DATES[-1]})", flush=True)
    print(f"Configs: {len(CONFIGS)}", flush=True)
    print("=" * 70, flush=True)

    # Try MLflow logging
    use_mlflow = False
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("tight_sl_sweep")
        use_mlflow = True
        print("MLflow tracking enabled", flush=True)
    except Exception as e:
        print(f"MLflow not available: {e}", flush=True)

    all_metrics = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {cfg_name} (TP={cfg['take_profit_ticks']}, SL={cfg['stop_loss_ticks']}, "
              f"hold={cfg['hold_ms']//1000}s, sig={cfg['signal_threshold']})", flush=True)
        print(f"{'='*60}", flush=True)

        cfg_dir = OUTPUT_DIR / cfg_name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        results_list = []
        for date in OOT_DATES:
            pred_path = PRED_NPZ_DIR / f"{date}_unfiltered.npz"
            mbo_path = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"

            if not pred_path.exists():
                print(f"  {date}: pred NPZ missing, skipping", flush=True)
                continue
            if not mbo_path.exists():
                print(f"  {date}: MBO file missing, skipping", flush=True)
                continue

            out_path = cfg_dir / f"fillsim_{date}.json"
            r = run_fillsim(mbo_path, pred_path, out_path, cfg)
            if r is not None:
                results_list.append(r)
                pnl = r.get("total_pnl_dollars", 0)
                trades = r.get("total_trades", 0)
                wr = r.get("win_rate", 0)
                print(f"  {date}: PnL=${pnl:>7.0f}  trades={trades:>4d}  WR={wr*100:>5.1f}%", flush=True)
            else:
                print(f"  {date}: fill_sim FAILED", flush=True)

        metrics = compute_metrics(results_list)
        all_metrics[cfg_name] = metrics

        # Log to MLflow
        if use_mlflow and metrics.get("n_trades", 0) > 0:
            try:
                with mlflow.start_run(run_name=cfg_name):
                    mlflow.log_params({
                        "take_profit_ticks": cfg["take_profit_ticks"],
                        "stop_loss_ticks": cfg["stop_loss_ticks"],
                        "hold_ms": cfg["hold_ms"],
                        "signal_threshold": cfg["signal_threshold"],
                        "latency_ms": 10,
                        "n_oot_dates": len(OOT_DATES),
                    })
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v)
            except Exception as e:
                print(f"  MLflow logging failed: {e}", flush=True)

    # ============ SUMMARY TABLE ============
    print("\n" + "=" * 120, flush=True)
    print("SUMMARY: TIGHT SL SWEEP RESULTS", flush=True)
    print("=" * 120, flush=True)

    header = (f"{'Config':<25} {'Trades':>7} {'WR':>7} {'PF':>7} {'Sharpe':>8} {'Sortino':>8} "
              f"{'NetTicks':>10} {'Tick/Trd':>9} {'PnL$':>10} {'Green':>6} {'Red':>5}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for cfg_name, m in all_metrics.items():
        if m.get("n_trades", 0) == 0:
            print(f"{cfg_name:<25}  NO TRADES", flush=True)
            continue
        print(
            f"{cfg_name:<25} "
            f"{m.get('n_trades', 0):>7d} "
            f"{m.get('win_rate', 0)*100:>6.1f}% "
            f"{m.get('profit_factor', 0):>7.2f} "
            f"{m.get('sharpe_per_trade', 0):>8.4f} "
            f"{m.get('sortino_per_trade', 0):>8.4f} "
            f"{m.get('net_ticks', 0):>10.1f} "
            f"{m.get('ticks_per_trade', 0):>9.4f} "
            f"${m.get('total_pnl_dollars', 0):>9.0f} "
            f"{m.get('n_green_days', 0):>4d}/{m.get('n_days', 0)} "
            f"{m.get('n_red_days', 0):>4d}",
            flush=True,
        )

    # Comparison vs baseline
    baseline = all_metrics.get("TP8_SL16_30m_baseline", {})
    if baseline.get("n_trades", 0) > 0:
        print(f"\n{'='*80}", flush=True)
        print("DELTA vs BASELINE (TP8/SL16/30m)", flush=True)
        print(f"{'='*80}", flush=True)
        bl_pf = baseline.get("profit_factor", 1)
        bl_wr = baseline.get("win_rate", 0)
        bl_sharpe = baseline.get("sharpe_per_trade", 0)
        bl_tpt = baseline.get("ticks_per_trade", 0)

        for cfg_name, m in all_metrics.items():
            if cfg_name == "TP8_SL16_30m_baseline" or m.get("n_trades", 0) == 0:
                continue
            pf_d = m.get("profit_factor", 0) - bl_pf
            wr_d = (m.get("win_rate", 0) - bl_wr) * 100
            sh_d = m.get("sharpe_per_trade", 0) - bl_sharpe
            tpt_d = m.get("ticks_per_trade", 0) - bl_tpt
            print(f"  {cfg_name:<25} PF:{pf_d:>+7.2f}  WR:{wr_d:>+6.1f}pp  Sharpe:{sh_d:>+8.4f}  Tick/Trd:{tpt_d:>+8.4f}", flush=True)

    # Save all results
    summary_path = OUTPUT_DIR / "sweep_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nResults saved to {summary_path}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
