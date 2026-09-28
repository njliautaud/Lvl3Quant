#!/usr/bin/env python3
"""
FIFO Fill Sim Validation v3 — Chase Entry + Conviction Exit
============================================================

Based on proven findings from Apr 26 + May 1:
1. Chase entry = ONLY profitable entry mode in FIFO
2. Conviction exit (30 bars opposing signal) >> signal-flip exit
3. SHORT-ONLY has edge in FIFO (462 profitable configs vs 1 long)
4. 5ms latency was used in the breakthrough result

Tests CNN-Mamba v2 predictions (IC=0.222, much better than LGBM IC=0.13)
with the proven chase+conviction formula across many more dates.

Author: Claude (Infrastructure Builder)
Date: 2026-05-08
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

LOG = logging.getLogger("FIFO_V3")
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))

FILL_SIM_BIN = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"


def find_mbo_file(date_str: str) -> "Path | None":
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
    min_confidence: float = 0.0,
    long_only: bool = False,
    short_only: bool = False,
) -> int:
    """Minimal gating — just confidence + side filter."""
    data = np.load(str(pred_file), allow_pickle=True)
    predictions = data["predictions"].copy()
    n_passed = 0

    for i in range(len(predictions)):
        p1 = predictions[i, 0]
        if abs(p1) < min_confidence:
            predictions[i] = [0.0, 0.0, 0.0]
        elif long_only and p1 < 0:
            predictions[i] = [0.0, 0.0, 0.0]
        elif short_only and p1 > 0:
            predictions[i] = [0.0, 0.0, 0.0]
        else:
            n_passed += 1

    np.savez_compressed(
        str(output_file),
        predictions=predictions,
        labels=data.get("labels", np.array([])),
        **{k: data[k] for k in data.keys() if k not in ("predictions", "labels")},
    )
    return n_passed


def run_fill_sim(
    mbo_file: Path,
    pred_file: Path,
    output_file: Path,
    signal_threshold: float = 0.01,
    hold_ms: int = 60000,
    stop_loss_ticks: float = 6,
    take_profit_ticks: float = 0,
    latency_ms: int = 5,
    chase_entry: bool = True,
    chase_max_ticks: float = 2,
    chase_force_cross: bool = False,
    conviction_exit_bars: int = 30,
    conviction_exit_mag: float = 0.5,
    prime_hours: bool = False,
    trailing_ticks: float = 0,
) -> "dict | None":
    """Run fill sim with chase+conviction config."""
    cmd = [
        str(FILL_SIM_BIN),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--signal-threshold", str(signal_threshold),
        "--hold-ms", str(hold_ms),
        "--stop-loss-ticks", str(stop_loss_ticks),
        "--latency-ms", str(latency_ms),
        "--max-wait-bars", "200",
    ]

    if take_profit_ticks > 0:
        cmd.extend(["--take-profit-ticks", str(take_profit_ticks)])
    if trailing_ticks > 0:
        cmd.extend(["--trailing-ticks", str(trailing_ticks)])
    if chase_entry:
        cmd.append("--chase-entry")
        cmd.extend(["--chase-max-ticks", str(chase_max_ticks)])
        if chase_force_cross:
            cmd.append("--chase-force-cross")
    if conviction_exit_bars > 0:
        cmd.extend(["--conviction-exit-bars", str(conviction_exit_bars)])
        cmd.extend(["--conviction-exit-mag", str(conviction_exit_mag)])
    if prime_hours:
        cmd.append("--prime-hours")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            LOG.warning(f"Fill sim failed: {result.stderr[:200]}")
            return None
        if Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        LOG.warning("Fill sim timed out")
    except Exception as e:
        LOG.warning(f"Fill sim error: {e}")
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output" / "fifo_validate_v3"))
    parser.add_argument("--max-dates", type=int, default=40)
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
    LOG.info("FIFO v3 — Chase Entry + Conviction Exit (PROVEN FORMULA)")
    LOG.info("=" * 70)

    # Find dates
    pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))
    valid_dates = []
    for pf in pred_files:
        if pf.stem.startswith("fold_"):
            continue
        date = pf.stem.replace("_predictions", "")
        mbo = find_mbo_file(date)
        if mbo is not None:
            valid_dates.append((date, pf, mbo))

    valid_dates = valid_dates[:args.max_dates]
    LOG.info(f"Found {len(valid_dates)} dates with both MBO + predictions")

    # ========================================================================
    # CONFIGS — Based on Apr 26 breakthrough + May 1 SHORT-ONLY finding
    # ========================================================================
    # Breakthrough config: chase + conviction(30, mag=0.5) + hold=60s + 5ms lat
    # Test: multiple confidence levels, SHORT vs LONG vs BOTH, TP/SL combos
    # ========================================================================

    # CNN-Mamba v2 prediction scale: range [-1.4, 1.5], std=0.27
    # |pred| > 0.3 = top ~25%, |pred| > 0.5 = top ~10%, |pred| > 0.75 = top ~2%
    # Use CONFIDENCE GATE (min_confidence) instead of fill sim threshold
    # Fill sim threshold = 0.01 (catch all gated signals)

    configs = [
        # === CHASE + CONVICTION: Core formula, SHORT-ONLY ===
        # Sweep confidence levels
        {"name": "chase_conv30_short_c03",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        {"name": "chase_conv30_short_c05",
         "conf": 0.5, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        {"name": "chase_conv30_short_c075",
         "conf": 0.75, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        # === CHASE + CONVICTION: BOTH SIDES ===
        {"name": "chase_conv30_both_c03",
         "conf": 0.3, "short_only": False, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        {"name": "chase_conv30_both_c05",
         "conf": 0.5, "short_only": False, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        # === CHASE + CONVICTION: LONG-ONLY (comparison) ===
        {"name": "chase_conv30_long_c03",
         "conf": 0.3, "short_only": False, "long_only": True,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        {"name": "chase_conv30_long_c05",
         "conf": 0.5, "short_only": False, "long_only": True,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        # === CHASE + FORCE CROSS ===
        {"name": "chase_force_conv30_short_c03",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": True,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        # === CONVICTION VARIANTS ===
        {"name": "chase_conv20_short_c03",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 20, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        {"name": "chase_conv50_short_c03",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 50, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},

        # === WITH TAKE PROFIT ===
        {"name": "chase_conv30_short_c03_tp6_sl4",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 6, "sl": 4, "hold": 60000, "trail": 0,
         "latency": 5},

        {"name": "chase_conv30_short_c03_tp4_sl4",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0,
         "latency": 5},

        # === TRAILING STOP ===
        {"name": "chase_conv30_short_c03_trail2",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 2,
         "latency": 5},

        # === PRIME HOURS ===
        {"name": "chase_conv30_short_c03_prime",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5, "prime": True},

        # === LATENCY COMPARISON ===
        {"name": "chase_conv30_short_c03_50ms",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": True, "chase_max": 2, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 50},

        # === PASSIVE BASELINE (for comparison) ===
        {"name": "passive_short_c03_conv30",
         "conf": 0.3, "short_only": True, "long_only": False,
         "threshold": 0.01,
         "chase": False, "chase_max": 0, "chase_force": False,
         "conv_bars": 30, "conv_mag": 0.1,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "latency": 5},
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        entry_type = "CHASE" if cfg["chase"] else "PASSIVE"
        if cfg.get("chase_force"):
            entry_type = "CHASE+FORCE"

        side = "SHORT" if cfg["short_only"] else ("LONG" if cfg["long_only"] else "BOTH")

        LOG.info(f"\n{'='*60}")
        LOG.info(f"CONFIG: {cfg_name}")
        LOG.info(f"  Entry: {entry_type}, Side: {side}, Threshold: {cfg['threshold']}")
        LOG.info(f"  Conviction: {cfg['conv_bars']} bars, mag={cfg['conv_mag']}")
        LOG.info(f"  TP={cfg['tp']}, SL={cfg['sl']}, Hold={cfg['hold']}ms, "
                 f"Trail={cfg['trail']}, Latency={cfg['latency']}ms")
        if cfg.get("prime"):
            LOG.info(f"  Prime hours only")
        LOG.info(f"{'='*60}")

        config_dir = output_dir / cfg_name
        config_dir.mkdir(exist_ok=True)

        day_results = []
        total_trades = 0
        total_pnl = 0.0
        total_signals = 0

        for date, pred_file, mbo_file in valid_dates:
            gated_pred = config_dir / f"{date}_gated.npz"

            # Apply side filter and confidence gate
            n_signals = create_gated_predictions(
                pred_file, gated_pred,
                min_confidence=cfg["conf"],
                long_only=cfg["long_only"],
                short_only=cfg["short_only"],
            )

            if n_signals == 0:
                continue

            total_signals += n_signals

            result_file = config_dir / f"{date}_result.json"
            result = run_fill_sim(
                mbo_file=mbo_file,
                pred_file=gated_pred,
                output_file=result_file,
                signal_threshold=cfg["threshold"],
                hold_ms=cfg["hold"],
                stop_loss_ticks=cfg["sl"],
                take_profit_ticks=cfg["tp"],
                latency_ms=cfg["latency"],
                chase_entry=cfg["chase"],
                chase_max_ticks=cfg["chase_max"],
                chase_force_cross=cfg["chase_force"],
                conviction_exit_bars=cfg["conv_bars"],
                conviction_exit_mag=cfg["conv_mag"],
                prime_hours=cfg.get("prime", False),
                trailing_ticks=cfg["trail"],
            )

            if result:
                n_trades = result.get("total_trades", 0)
                pnl_dollars = result.get("total_pnl_dollars", 0)
                pnl_ticks = pnl_dollars / 12.50
                wr = result.get("win_rate", 0)
                pf = result.get("profit_factor", 0)
                fr = result.get("fill_rate", 0)

                total_trades += n_trades
                total_pnl += pnl_ticks

                day_results.append({
                    "date": date,
                    "signals": n_signals,
                    "trades": n_trades,
                    "fill_rate": fr,
                    "pnl_dollars": pnl_dollars,
                    "pnl_ticks": pnl_ticks,
                    "wr": wr,
                    "pf": pf,
                })

                LOG.info(f"  {date}: {n_signals:5d} sig → {n_trades:3d} tr, "
                         f"WR={wr:.3f}, PnL=${pnl_dollars:+.0f} ({pnl_ticks:+.1f}tk), "
                         f"PF={pf:.2f}, FR={fr:.1%}")
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
            total_wr = np.mean([d["wr"] for d in day_results if d["trades"] > 0])
            avg_fr = np.mean([d["fill_rate"] for d in day_results if d["fill_rate"] > 0])

            LOG.info(f"\n  SUMMARY for {cfg_name}:")
            LOG.info(f"    Entry: {entry_type}, Side: {side}")
            LOG.info(f"    Days: {n_days}, Green: {green_days}/{n_days} ({green_days/n_days*100:.0f}%)")
            LOG.info(f"    Total trades: {total_trades}, Total signals: {total_signals}")
            LOG.info(f"    Avg fill rate: {avg_fr:.1%}")
            LOG.info(f"    Avg WR: {total_wr:.3f}")
            LOG.info(f"    Total PnL: {total_pnl:+.1f}tk (${total_pnl*12.50:+,.0f})")
            LOG.info(f"    Avg daily PnL: {avg_daily_pnl:+.1f}tk (${avg_daily_pnl*12.50:+.0f}/day)")
            LOG.info(f"    Sortino: {sortino:+.3f}")
            LOG.info(f"    Per trade: {total_pnl/max(total_trades,1):+.2f}tk")

            all_results[cfg_name] = {
                "entry_type": entry_type,
                "side": side,
                "n_days": n_days,
                "green_days": green_days,
                "green_pct": green_days / n_days * 100,
                "total_trades": total_trades,
                "total_signals": total_signals,
                "avg_fill_rate": float(avg_fr),
                "avg_wr": float(total_wr),
                "total_pnl_ticks": float(total_pnl),
                "total_pnl_dollars": float(total_pnl * 12.50),
                "avg_daily_pnl_ticks": float(avg_daily_pnl),
                "avg_daily_pnl_dollars": float(avg_daily_pnl * 12.50),
                "std_daily_pnl": float(std_daily_pnl),
                "sortino": float(sortino),
                "per_trade_pnl": float(total_pnl / max(total_trades, 1)),
            }

    # Final comparison
    LOG.info(f"\n\n{'='*110}")
    LOG.info("FINAL COMPARISON — Chase + Conviction Exit Sweep v3")
    LOG.info(f"{'='*110}")
    LOG.info(f"{'Config':<38} {'Entry':>6} {'Side':>5} {'Trades':>6} {'FR':>5} {'WR':>5} "
             f"{'PnL(tk)':>8} {'$/day':>7} {'Green%':>6} {'Sortino':>7} {'$/tr':>6}")
    LOG.info("-" * 110)

    for name, r in sorted(all_results.items(),
                          key=lambda x: x[1].get("sortino", -99), reverse=True):
        LOG.info(
            f"{name:<38} {r['entry_type']:>6} {r['side']:>5} {r['total_trades']:>6} "
            f"{r['avg_fill_rate']:>4.1%} {r['avg_wr']:>5.3f} "
            f"{r['total_pnl_ticks']:>+8.1f} "
            f"{r['avg_daily_pnl_dollars']:>+7.0f} "
            f"{r['green_pct']:>5.0f}% "
            f"{r['sortino']:>+7.3f} "
            f"{r['per_trade_pnl']:>+6.2f}"
        )

    # Save
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    LOG.info(f"\nResults saved to {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
