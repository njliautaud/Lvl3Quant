#!/usr/bin/env python3
"""
FIFO Fill Sim Validation — Test regime rules with realistic FIFO fills
=======================================================================

Takes the best rules from focused_gate_v1 and validates them through the
Rust FIFO fill simulator. This is the final hurdle: do the midpoint-based
edges survive realistic queue position, fill probability, and adverse selection?

Tests the top rule configs:
1. Long-only, high confidence, with trailing stop
2. Various take-profit / stop-loss combinations
3. Signal persistence filter

Author: Claude (Infrastructure Builder)
Date: 2026-05-08
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

LOG = logging.getLogger("FIFO_VALIDATE")
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))

FILL_SIM_BIN = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"

COMMISSION_TICKS = 0.376  # $4.70 / $12.50


def find_mbo_file(date_str: str) -> Path | None:
    """Find MBO .dbn file for a date."""
    # Try date formats
    for pattern in [
        f"glbx-mdp3-{date_str}.mbo.dbn.zst",
        f"*{date_str}*.dbn.zst",
        f"*{date_str}*.dbn",
    ]:
        matches = list(MBO_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def create_gated_predictions(
    pred_file: Path,
    output_file: Path,
    min_confidence: float = 0.75,
    require_agreement: bool = True,
    min_persistence: float = 0.5,
    long_only: bool = True,
) -> int:
    """
    Create a gated prediction file that zeros out predictions that don't
    pass the regime gate. The fill sim will only trade non-zero predictions.

    Returns number of signals passing the gate.
    """
    data = np.load(str(pred_file), allow_pickle=True)
    predictions = data["predictions"].copy()  # (N, 3)
    labels = data["labels"]

    n_total = len(predictions)
    n_passed = 0

    for i in range(n_total):
        p1, p5, p10 = predictions[i]

        # Gate: zero out predictions that don't pass
        pass_gate = True

        # Confidence gate
        if abs(p1) < min_confidence:
            pass_gate = False

        # Long-only gate
        if long_only and p1 < 0:
            pass_gate = False

        # Agreement gate: all horizons same sign
        if require_agreement:
            if not (np.sign(p1) == np.sign(p5) == np.sign(p10)) or np.sign(p1) == 0:
                pass_gate = False

        # Persistence gate: signal persists from 1s to 10s
        if min_persistence is not None and abs(p1) > 1e-6:
            ratio = p10 / (p1 + 1e-8)
            if ratio < min_persistence:
                pass_gate = False

        if not pass_gate:
            predictions[i] = [0.0, 0.0, 0.0]
        else:
            n_passed += 1

    # Save gated predictions
    np.savez_compressed(
        str(output_file),
        predictions=predictions,
        labels=labels,
        **{k: data[k] for k in data.keys() if k not in ("predictions", "labels")},
    )

    return n_passed


def run_fill_sim(
    mbo_file: Path,
    pred_file: Path,
    output_file: Path,
    signal_threshold: float = 0.0,
    hold_ms: int = 30000,
    trailing_ticks: float = 0,
    stop_loss_ticks: float = 8,
    take_profit_ticks: float = 4,
    max_wait_bars: int = 100,
    latency_ms: int = 50,
) -> dict | None:
    """Run the Rust fill sim binary."""
    cmd = [
        str(FILL_SIM_BIN),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--signal-threshold", str(signal_threshold),
        "--hold-ms", str(hold_ms),
        "--stop-loss-ticks", str(stop_loss_ticks),
        "--take-profit-ticks", str(take_profit_ticks),
        "--max-wait-bars", str(max_wait_bars),
        "--latency-ms", str(latency_ms),
    ]

    if trailing_ticks > 0:
        cmd.extend(["--trailing-ticks", str(trailing_ticks)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            LOG.warning(f"Fill sim failed: {result.stderr[:200]}")
            return None

        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        LOG.warning(f"Fill sim timed out")
    except Exception as e:
        LOG.warning(f"Fill sim error: {e}")

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output" / "fifo_validate_v1"))
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir) + ".log", mode="w"),
        ],
    )

    LOG.info("=" * 70)
    LOG.info("FIFO FILL SIM VALIDATION — Regime Rules")
    LOG.info("=" * 70)

    # Find dates with both MBO and prediction files
    pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))
    valid_dates = []
    for pf in pred_files:
        if pf.stem.startswith("fold_"):
            continue
        date = pf.stem.replace("_predictions", "")
        mbo = find_mbo_file(date)
        if mbo is not None:
            valid_dates.append((date, pf, mbo))

    LOG.info(f"Found {len(valid_dates)} dates with both MBO + predictions")
    for d, pf, mbo in valid_dates[:5]:
        LOG.info(f"  {d}: {pf.name} + {mbo.name}")

    if not valid_dates:
        LOG.error("No valid dates found!")
        return

    # Define regime configs to test
    configs = [
        # Baseline: no gate, various exit params
        {"name": "baseline_tp4_sl8", "conf": 0.0, "agree": False, "persist": None,
         "long_only": False, "tp": 4, "sl": 8, "hold": 30000, "trail": 0},

        # Long-only + high confidence
        {"name": "long_c075_tp4_sl8", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": True, "tp": 4, "sl": 8, "hold": 30000, "trail": 0},

        {"name": "long_c100_tp4_sl8", "conf": 1.0, "agree": True, "persist": None,
         "long_only": True, "tp": 4, "sl": 8, "hold": 30000, "trail": 0},

        # Different exit strategies for the best regime
        {"name": "long_c075_tp2_sl4", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": True, "tp": 2, "sl": 4, "hold": 15000, "trail": 0},

        {"name": "long_c075_tp3_sl6", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": True, "tp": 3, "sl": 6, "hold": 20000, "trail": 0},

        {"name": "long_c075_tp6_sl4", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": True, "tp": 6, "sl": 4, "hold": 30000, "trail": 0},

        {"name": "long_c075_trail2", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": True, "tp": 0, "sl": 6, "hold": 30000, "trail": 2},

        {"name": "long_c075_trail3", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": True, "tp": 0, "sl": 8, "hold": 60000, "trail": 3},

        # Short-only comparison
        {"name": "short_c075_tp4_sl8", "conf": 0.75, "agree": True, "persist": 0.5,
         "long_only": False, "tp": 4, "sl": 8, "hold": 30000, "trail": 0,
         "_short_only": True},

        # Very high confidence
        {"name": "long_c150_tp4_sl8", "conf": 1.5, "agree": True, "persist": 0.8,
         "long_only": True, "tp": 4, "sl": 8, "hold": 30000, "trail": 0},
    ]

    # Run all configs across all dates
    all_results = {}

    for cfg in configs:
        LOG.info(f"\n{'='*60}")
        LOG.info(f"CONFIG: {cfg['name']}")
        LOG.info(f"  conf≥{cfg['conf']}, agree={cfg['agree']}, persist={cfg.get('persist')}")
        LOG.info(f"  long_only={cfg['long_only']}, TP={cfg['tp']}, SL={cfg['sl']}, "
                 f"hold={cfg['hold']}ms, trail={cfg['trail']}")
        LOG.info(f"{'='*60}")

        config_dir = output_dir / cfg["name"]
        config_dir.mkdir(exist_ok=True)

        day_results = []
        total_trades = 0
        total_pnl = 0.0
        total_signals = 0

        for date, pred_file, mbo_file in valid_dates:
            # Create gated predictions
            gated_pred = config_dir / f"{date}_gated.npz"

            is_short_only = cfg.get("_short_only", False)

            if is_short_only:
                # Invert: gate for short-only
                n_signals = create_gated_predictions(
                    pred_file, gated_pred,
                    min_confidence=cfg["conf"],
                    require_agreement=cfg["agree"],
                    min_persistence=cfg.get("persist"),
                    long_only=False,  # Allow shorts
                )
                # Also need to filter out longs - hack: zero positive predictions
                data = np.load(str(gated_pred), allow_pickle=True)
                preds = data["predictions"].copy()
                preds[preds[:, 0] > 0] = 0  # zero longs
                np.savez_compressed(
                    str(gated_pred),
                    predictions=preds,
                    **{k: data[k] for k in data.keys() if k != "predictions"},
                )
                n_signals = (preds[:, 0] != 0).sum()
            else:
                n_signals = create_gated_predictions(
                    pred_file, gated_pred,
                    min_confidence=cfg["conf"],
                    require_agreement=cfg["agree"],
                    min_persistence=cfg.get("persist"),
                    long_only=cfg["long_only"],
                )

            if n_signals == 0:
                continue

            total_signals += n_signals

            # Run fill sim
            result_file = config_dir / f"{date}_result.json"
            result = run_fill_sim(
                mbo_file=mbo_file,
                pred_file=gated_pred,
                output_file=result_file,
                signal_threshold=0.01,  # Any non-zero prediction
                hold_ms=cfg["hold"],
                trailing_ticks=cfg["trail"],
                stop_loss_ticks=cfg["sl"],
                take_profit_ticks=cfg["tp"],
                max_wait_bars=100,
                latency_ms=50,
            )

            if result:
                n_trades = result.get("total_trades", 0)
                # Fill sim already includes commission in total_pnl_dollars
                pnl_dollars = result.get("total_pnl_dollars", 0)
                pnl_ticks = pnl_dollars / 12.50  # Convert $ to ticks
                wr = result.get("win_rate", 0)
                pf = result.get("profit_factor", 0)
                avg_queue = result.get("avg_queue_position", 0)
                avg_fill_lat = result.get("avg_fill_latency_ms", 0)

                total_trades += n_trades
                total_pnl += pnl_ticks

                day_results.append({
                    "date": date,
                    "signals": n_signals,
                    "trades": n_trades,
                    "fill_rate": n_trades / max(n_signals, 1),
                    "pnl_dollars": pnl_dollars,
                    "pnl_ticks": pnl_ticks,
                    "wr": wr,
                    "pf": pf,
                    "avg_queue": avg_queue,
                    "avg_fill_lat_ms": avg_fill_lat,
                })

                LOG.info(f"  {date}: {n_signals:4d} signals → {n_trades:3d} trades, "
                         f"WR={wr:.3f}, PnL=${pnl_dollars:+.0f} ({pnl_ticks:+.1f}tk), "
                         f"PF={pf:.2f}, Q={avg_queue:.0f}, lat={avg_fill_lat:.0f}ms "
                         f"(fill_rate={n_trades/max(n_signals,1):.1%})")
            else:
                LOG.warning(f"  {date}: fill sim failed")

        # Summary
        n_days = len(day_results)
        if n_days > 0:
            day_pnls = [d["pnl_ticks"] for d in day_results]
            green_days = sum(1 for p in day_pnls if p > 0)
            avg_daily_pnl = np.mean(day_pnls)
            std_daily_pnl = np.std(day_pnls)
            neg_pnls = [p for p in day_pnls if p < 0]
            sortino = avg_daily_pnl / (np.sqrt(np.mean([p**2 for p in neg_pnls])) if neg_pnls else 1e-8)

            avg_fill_rate = np.mean([d["fill_rate"] for d in day_results])
            avg_wr = np.mean([d["wr"] for d in day_results if d["trades"] > 0])

            LOG.info(f"\n  SUMMARY for {cfg['name']}:")
            LOG.info(f"    Days: {n_days}, Green: {green_days}/{n_days} ({green_days/n_days*100:.0f}%)")
            LOG.info(f"    Total trades: {total_trades}, Total signals: {total_signals}")
            LOG.info(f"    Fill rate: {avg_fill_rate:.1%}")
            LOG.info(f"    Avg WR: {avg_wr:.3f}")
            LOG.info(f"    Total net PnL: {total_pnl:+.1f} ticks")
            LOG.info(f"    Avg daily PnL: {avg_daily_pnl:+.1f}tk, Std: {std_daily_pnl:.1f}tk")
            LOG.info(f"    Sortino: {sortino:+.3f}")

            all_results[cfg["name"]] = {
                "n_days": n_days,
                "green_days": green_days,
                "total_trades": total_trades,
                "total_signals": total_signals,
                "fill_rate": float(avg_fill_rate),
                "avg_wr": float(avg_wr),
                "total_pnl_ticks": float(total_pnl),
                "avg_daily_pnl": float(avg_daily_pnl),
                "sortino": float(sortino),
                "day_results": day_results,
            }

    # Final comparison
    LOG.info(f"\n{'='*70}")
    LOG.info(f"FINAL COMPARISON — All Configs")
    LOG.info(f"{'='*70}")
    LOG.info(f"  {'Config':>25s}  {'Days':>4s}  {'Trades':>6s}  {'FillR':>6s}  {'WR':>6s}  "
             f"{'PnLtk':>8s}  {'DlyPnL':>7s}  {'Sortino':>8s}  {'Green%':>6s}")

    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["sortino"]):
        LOG.info(f"  {name:>25s}  {r['n_days']:4d}  {r['total_trades']:6d}  "
                 f"{r['fill_rate']:5.1%}  {r['avg_wr']:.3f}  "
                 f"{r['total_pnl_ticks']:+7.1f}  {r['avg_daily_pnl']:+6.1f}  "
                 f"{r['sortino']:+7.3f}  "
                 f"{r['green_days']/max(r['n_days'],1)*100:5.0f}%")

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    LOG.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
