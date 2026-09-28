#!/usr/bin/env python3
"""
FIFO Fill Sim Validation v2 — Execution Tactics Sweep
======================================================

v1 showed passive limit orders lose money due to adverse selection.
v2 tests: market entry, chase entry, symmetric TP/SL, signal-flip exit.

Key question: can different execution tactics capture the proven signal edge?

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

import numpy as np

LOG = logging.getLogger("FIFO_V2")
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))

FILL_SIM_BIN = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"

COMMISSION_TICKS = 0.376


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
    min_confidence: float = 0.75,
    require_agreement: bool = True,
    min_persistence: float = None,
    long_only: bool = True,
    short_only: bool = False,
) -> int:
    """Zero out predictions that don't pass the gate."""
    data = np.load(str(pred_file), allow_pickle=True)
    predictions = data["predictions"].copy()
    n_total = len(predictions)
    n_passed = 0

    for i in range(n_total):
        p1, p5, p10 = predictions[i]
        pass_gate = True

        if abs(p1) < min_confidence:
            pass_gate = False
        if long_only and p1 < 0:
            pass_gate = False
        if short_only and p1 > 0:
            pass_gate = False
        if require_agreement:
            if not (np.sign(p1) == np.sign(p5) == np.sign(p10)) or np.sign(p1) == 0:
                pass_gate = False
        if min_persistence is not None and abs(p1) > 1e-6:
            ratio = p10 / (p1 + 1e-8)
            if ratio < min_persistence:
                pass_gate = False

        if not pass_gate:
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
    hold_ms: int = 30000,
    trailing_ticks: float = 0,
    stop_loss_ticks: float = 8,
    take_profit_ticks: float = 4,
    max_wait_bars: int = 100,
    latency_ms: int = 50,
    market_entry: bool = False,
    chase_entry: bool = False,
    chase_max_ticks: float = 2,
    chase_force_cross: bool = False,
    signal_flip_exit: bool = False,
    prime_hours: bool = False,
) -> "dict | None":
    """Run the Rust fill sim binary with extended options."""
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
    if market_entry:
        cmd.append("--market-entry")
    if chase_entry:
        cmd.append("--chase-entry")
        cmd.extend(["--chase-max-ticks", str(chase_max_ticks)])
        if chase_force_cross:
            cmd.append("--chase-force-cross")
    if signal_flip_exit:
        cmd.append("--signal-flip-exit")
    if prime_hours:
        cmd.append("--prime-hours")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            LOG.warning(f"Fill sim failed: {result.stderr[:200]}")
            return None
        if output_file.exists():
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
                       default=str(LVL3_ROOT / "output" / "fifo_validate_v2"))
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
    LOG.info("FIFO FILL SIM v2 — Execution Tactics Sweep")
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
    if not valid_dates:
        LOG.error("No valid dates found!")
        return

    # ========================================================================
    # CONFIGS — Testing execution tactics
    # ========================================================================
    # v1 showed: passive limit + TP=4/SL=8 = Sortino -0.39 (best gated)
    # Key insight: 52% WR with 2:1 risk/reward can't work. Need either:
    #   1. Higher WR (market entry avoids adverse selection)
    #   2. Better risk/reward ratio (symmetric TP/SL)
    #   3. Dynamic exits (signal-flip)
    # ========================================================================

    configs = [
        # === PASSIVE SYMMETRIC TP/SL (v1 used 2:1 risk/reward requiring 67% WR) ===
        # Symmetric TP=SL requires only ~51% WR to break even after commission
        {"name": "pass_long_c075_tp3_sl3",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 3, "sl": 3, "hold": 20000, "trail": 0},

        {"name": "pass_long_c075_tp4_sl4",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0},

        {"name": "pass_long_c075_tp2_sl2",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 2, "sl": 2, "hold": 10000, "trail": 0},

        {"name": "pass_long_c100_tp3_sl3",
         "conf": 1.0, "agree": True, "persist": None, "long_only": True,
         "tp": 3, "sl": 3, "hold": 20000, "trail": 0},

        {"name": "pass_long_c100_tp4_sl4",
         "conf": 1.0, "agree": True, "persist": None, "long_only": True,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0},

        # === HIGHER TP THAN SL (let winners run, cut losers) ===
        {"name": "pass_long_c075_tp6_sl3",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 6, "sl": 3, "hold": 30000, "trail": 0},

        {"name": "pass_long_c100_tp6_sl3",
         "conf": 1.0, "agree": True, "persist": None, "long_only": True,
         "tp": 6, "sl": 3, "hold": 30000, "trail": 0},

        # === TRAILING STOP (capture momentum) ===
        {"name": "pass_long_c075_trail2_sl4",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 0, "sl": 4, "hold": 30000, "trail": 2},

        {"name": "pass_long_c075_trail3_sl6",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 3},

        # === SIGNAL-FLIP EXIT (exit when prediction reverses) ===
        {"name": "pass_long_c075_flipex_sl6",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 0, "sl": 6, "hold": 60000, "trail": 0,
         "signal_flip_exit": True},

        {"name": "pass_long_c100_flipex_sl4",
         "conf": 1.0, "agree": True, "persist": None, "long_only": True,
         "tp": 0, "sl": 4, "hold": 30000, "trail": 0,
         "signal_flip_exit": True},

        # === PRIME HOURS ONLY (10:30-14:30 ET, lower noise) ===
        {"name": "pass_long_c075_prime_tp3_sl3",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 3, "sl": 3, "hold": 20000, "trail": 0,
         "prime_hours": True},

        {"name": "pass_long_c075_prime_tp4_sl4",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0,
         "prime_hours": True},

        # === CHASE ENTRY (post limit, reprice if BBO moves) ===
        {"name": "chase_long_c075_tp4_sl4",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0,
         "chase_entry": True, "chase_max_ticks": 2, "chase_force_cross": False},

        {"name": "chase_force_long_c075_tp4_sl4",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": True,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0,
         "chase_entry": True, "chase_max_ticks": 2, "chase_force_cross": True},

        # === BOTH SIDES for comparison ===
        {"name": "pass_both_c075_tp3_sl3",
         "conf": 0.75, "agree": True, "persist": 0.5, "long_only": False,
         "tp": 3, "sl": 3, "hold": 20000, "trail": 0},

        # === VERY HIGH CONFIDENCE ===
        {"name": "pass_long_c200_tp4_sl4",
         "conf": 2.0, "agree": True, "persist": 0.8, "long_only": True,
         "tp": 4, "sl": 4, "hold": 30000, "trail": 0},
    ]

    all_results = {}

    for cfg in configs:
        cfg_name = cfg["name"]
        is_market = cfg.get("market_entry", False)
        is_chase = cfg.get("chase_entry", False)
        is_flip = cfg.get("signal_flip_exit", False)
        is_prime = cfg.get("prime_hours", False)

        LOG.info(f"\n{'='*60}")
        LOG.info(f"CONFIG: {cfg_name}")
        entry_type = "MARKET" if is_market else ("CHASE" if is_chase else "PASSIVE")
        LOG.info(f"  Entry: {entry_type}, conf≥{cfg['conf']}, agree={cfg['agree']}, "
                 f"persist={cfg.get('persist')}")
        LOG.info(f"  long_only={cfg['long_only']}, TP={cfg['tp']}, SL={cfg['sl']}, "
                 f"hold={cfg['hold']}ms, trail={cfg['trail']}")
        if is_flip:
            LOG.info(f"  signal_flip_exit=True")
        if is_prime:
            LOG.info(f"  prime_hours_only=True")
        LOG.info(f"{'='*60}")

        config_dir = output_dir / cfg_name
        config_dir.mkdir(exist_ok=True)

        day_results = []
        total_trades = 0
        total_pnl = 0.0
        total_signals = 0

        for date, pred_file, mbo_file in valid_dates:
            gated_pred = config_dir / f"{date}_gated.npz"

            short_only = cfg.get("_short_only", False)
            n_signals = create_gated_predictions(
                pred_file, gated_pred,
                min_confidence=cfg["conf"],
                require_agreement=cfg["agree"],
                min_persistence=cfg.get("persist"),
                long_only=cfg["long_only"],
                short_only=short_only,
            )

            if n_signals == 0:
                continue

            total_signals += n_signals

            result_file = config_dir / f"{date}_result.json"
            result = run_fill_sim(
                mbo_file=mbo_file,
                pred_file=gated_pred,
                output_file=result_file,
                signal_threshold=0.01,
                hold_ms=cfg["hold"],
                trailing_ticks=cfg["trail"],
                stop_loss_ticks=cfg["sl"],
                take_profit_ticks=cfg["tp"],
                max_wait_bars=100,
                latency_ms=50,
                market_entry=is_market,
                chase_entry=is_chase,
                chase_max_ticks=cfg.get("chase_max_ticks", 2),
                chase_force_cross=cfg.get("chase_force_cross", False),
                signal_flip_exit=is_flip,
                prime_hours=is_prime,
            )

            if result:
                n_trades = result.get("total_trades", 0)
                pnl_dollars = result.get("total_pnl_dollars", 0)
                pnl_ticks = pnl_dollars / 12.50
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

                LOG.info(f"  {date}: {n_signals:4d} sig → {n_trades:3d} tr, "
                         f"WR={wr:.3f}, PnL=${pnl_dollars:+.0f} ({pnl_ticks:+.1f}tk), "
                         f"PF={pf:.2f}, fillrate={n_trades/max(n_signals,1):.1%}")
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

            LOG.info(f"\n  SUMMARY for {cfg_name}:")
            LOG.info(f"    Entry: {entry_type}")
            LOG.info(f"    Days: {n_days}, Green: {green_days}/{n_days} ({green_days/n_days*100:.0f}%)")
            LOG.info(f"    Total trades: {total_trades}, Total signals: {total_signals}")
            LOG.info(f"    Fill rate: {total_trades/max(total_signals,1):.1%}")
            LOG.info(f"    Avg WR: {total_wr:.3f}")
            LOG.info(f"    Total net PnL: {total_pnl:+.1f} ticks (${total_pnl*12.50:+,.0f})")
            LOG.info(f"    Avg daily PnL: {avg_daily_pnl:+.1f}tk, Std: {std_daily_pnl:.1f}tk")
            LOG.info(f"    Sortino: {sortino:+.3f}")
            LOG.info(f"    Per trade: {total_pnl/max(total_trades,1):+.2f}tk")

            all_results[cfg_name] = {
                "entry_type": entry_type,
                "n_days": n_days,
                "green_days": green_days,
                "green_pct": green_days / n_days * 100,
                "total_trades": total_trades,
                "total_signals": total_signals,
                "fill_rate": total_trades / max(total_signals, 1),
                "avg_wr": float(total_wr),
                "total_pnl_ticks": float(total_pnl),
                "avg_daily_pnl": float(avg_daily_pnl),
                "std_daily_pnl": float(std_daily_pnl),
                "sortino": float(sortino),
                "per_trade_pnl": float(total_pnl / max(total_trades, 1)),
            }
        else:
            LOG.warning(f"  No results for {cfg_name}")

    # Final comparison table
    LOG.info(f"\n\n{'='*100}")
    LOG.info("FINAL COMPARISON — Execution Tactics Sweep v2")
    LOG.info(f"{'='*100}")
    LOG.info(f"{'Config':<38} {'Entry':>7} {'Trades':>6} {'Fill%':>5} {'WR':>5} "
             f"{'PnL(tk)':>8} {'$/day':>7} {'Green%':>6} {'Sortino':>7} {'$/tr':>6}")
    LOG.info("-" * 100)

    for name, r in sorted(all_results.items(),
                          key=lambda x: x[1].get("sortino", -99), reverse=True):
        LOG.info(
            f"{name:<38} {r['entry_type']:>7} {r['total_trades']:>6} "
            f"{r['fill_rate']:>4.1%} {r['avg_wr']:>5.3f} "
            f"{r['total_pnl_ticks']:>+8.1f} "
            f"{r['avg_daily_pnl']*12.50:>+7.0f} "
            f"{r['green_pct']:>5.0f}% "
            f"{r['sortino']:>+7.3f} "
            f"{r['per_trade_pnl']:>+6.2f}"
        )

    # Save results
    results_file = output_dir / "all_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    LOG.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
