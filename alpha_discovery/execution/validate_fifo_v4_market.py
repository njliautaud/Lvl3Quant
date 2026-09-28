#!/usr/bin/env python3
"""
FIFO Fill Sim v4 — Market Entry at Extreme Confidence
=====================================================

v1-v3 tested passive limits → all negative (adverse selection).
v4 tests: MARKET ORDERS at very high confidence.

Logic: If XGBoost top 1% signals have +0.49 ticks edge on midpoint,
and market order cost = 0.376 ticks commission (no spread crossing cost per HC #231),
then net edge = +0.49 - 0.376 = +0.11 ticks/trade.

But wait — with market orders we AVOID adverse selection entirely because
we get filled immediately at the ask (for longs) or bid (for shorts).
The question is whether the fill price is good enough.

Tests:
1. Market entry at various confidence thresholds (top 0.5%, 1%, 2%, 5%)
2. Various exit strategies (fixed TP/SL, trailing, signal-flip, time-stop)
3. Long-only vs both sides
4. Prime hours vs all hours
5. With cooldown (no re-entry for N bars after exit)

Author: Claude
Date: 2026-05-08
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

LOG = logging.getLogger("FIFO_V4")
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))

# Use local paths — this runs on Jupiter or Neptune
FILL_SIM_BIN = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"

COMMISSION_TICKS = 0.376  # $4.70 / $12.50


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
    long_only: bool = False,
    short_only: bool = False,
    percentile_threshold: float = None,
) -> int:
    """Zero out predictions that don't pass the gate.

    If percentile_threshold is set, use the top X% of |predictions|
    instead of an absolute confidence threshold.
    """
    data = np.load(str(pred_file), allow_pickle=True)
    predictions = data["predictions"].copy()
    n_total = len(predictions)

    # Calculate percentile threshold if requested
    abs_preds = np.abs(predictions[:, 0])  # 1s horizon
    if percentile_threshold is not None:
        pct_value = np.percentile(abs_preds[abs_preds > 0], 100 - percentile_threshold)
        min_confidence = max(min_confidence, pct_value)
        LOG.info(f"  Percentile threshold {percentile_threshold}% → min_conf={min_confidence:.4f}")

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
    latency_ms: int = 5,
    market_entry: bool = False,
    chase_entry: bool = False,
    signal_flip_exit: bool = False,
    prime_hours: bool = False,
    cooldown_bars: int = 0,
) -> "dict | None":
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
    if market_entry:
        cmd.append("--market-entry")
    if chase_entry:
        cmd.append("--chase-entry")
    if signal_flip_exit:
        cmd.append("--signal-flip-exit")
    if prime_hours:
        cmd.append("--prime-hours")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            LOG.warning(f"Fill sim failed: {result.stderr[:300]}")
            return None
        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        LOG.warning("Fill sim timed out")
    except Exception as e:
        LOG.warning(f"Fill sim error: {e}")
    return None


def analyze_results(all_results: list, config_name: str):
    """Compute aggregate metrics across dates."""
    if not all_results:
        return None

    total_trades = 0
    total_pnl = 0.0
    total_winners = 0
    daily_pnls = []
    all_trade_pnls = []

    for date, result in all_results:
        trades = result.get("trades", [])
        if not trades:
            daily_pnls.append(0.0)
            continue

        day_pnl = 0.0
        for t in trades:
            pnl_ticks = t.get("pnl_ticks", 0.0) - COMMISSION_TICKS
            day_pnl += pnl_ticks
            total_pnl += pnl_ticks
            total_trades += 1
            all_trade_pnls.append(pnl_ticks)
            if pnl_ticks > 0:
                total_winners += 1
        daily_pnls.append(day_pnl)

    if total_trades == 0:
        return None

    wr = total_winners / total_trades
    avg_pnl = total_pnl / total_trades
    green_days = sum(1 for p in daily_pnls if p > 0)
    green_pct = green_days / len(daily_pnls) if daily_pnls else 0

    # Sortino
    daily_arr = np.array(daily_pnls)
    mean_daily = daily_arr.mean()
    downside = daily_arr[daily_arr < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-6
    sortino = mean_daily / downside_std if downside_std > 1e-6 else 0.0

    # Profit factor
    gross_win = sum(p for p in all_trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Winners vs losers breakdown
    winners = [p for p in all_trade_pnls if p > 0]
    losers = [p for p in all_trade_pnls if p <= 0]
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0

    return {
        "config": config_name,
        "trades": total_trades,
        "trades_per_day": total_trades / len(daily_pnls),
        "wr": wr,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "pf": pf,
        "sortino": sortino,
        "green_days": green_days,
        "total_days": len(daily_pnls),
        "green_pct": green_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "dates": len(all_results),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output" / "fifo_validate_v4_market"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir / "run.log"), mode="w"),
        ],
    )

    LOG.info("=" * 70)
    LOG.info("FIFO FILL SIM v4 — Market Entry at Extreme Confidence")
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
    # CONFIGS — Market entry tests at extreme confidence
    # ========================================================================
    # The KEY INSIGHT: market orders avoid adverse selection entirely.
    # Cost = commission only (0.376 ticks).
    # If top 1% signals have +0.49 ticks midpoint edge, net = +0.11 ticks.
    #
    # We test various exit strategies to see which captures the edge.
    # ========================================================================

    configs = [
        # === MARKET ENTRY, VARIOUS CONFIDENCE LEVELS ===
        # Short hold times (5-15s) to match signal horizon

        # Top 1% confidence, market entry, various exits
        {"name": "mkt_top1pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_top1pct_tp4_sl4_hold15s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 4, "sl": 4, "hold": 15000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_top1pct_tp2_sl2_hold5s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 2, "sl": 2, "hold": 5000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_top1pct_trail2_sl4_hold15s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 99, "sl": 4, "hold": 15000, "trail": 2, "market": True, "latency": 5},

        {"name": "mkt_top1pct_flipex_sl4_hold30s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 99, "sl": 4, "hold": 30000, "trail": 0, "market": True, "latency": 5,
         "flipex": True},

        # Top 0.5% confidence
        {"name": "mkt_top05pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 0.5, "agree": True, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_top05pct_tp4_sl4_hold15s",
         "conf": 0.0, "pct": 0.5, "agree": True, "long_only": False,
         "tp": 4, "sl": 4, "hold": 15000, "trail": 0, "market": True, "latency": 5},

        # Top 2% confidence
        {"name": "mkt_top2pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 2.0, "agree": True, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_top2pct_tp4_sl4_hold15s",
         "conf": 0.0, "pct": 2.0, "agree": True, "long_only": False,
         "tp": 4, "sl": 4, "hold": 15000, "trail": 0, "market": True, "latency": 5},

        # Top 5% confidence
        {"name": "mkt_top5pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 5.0, "agree": True, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        # === LONG-ONLY MARKET ENTRY (longs have been better) ===
        {"name": "mkt_long_top1pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": True,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_long_top1pct_tp4_sl4_hold15s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": True,
         "tp": 4, "sl": 4, "hold": 15000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_long_top05pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 0.5, "agree": True, "long_only": True,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        # === PRIME HOURS ONLY (10:30-14:30 ET, less noise) ===
        {"name": "mkt_prime_top1pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5,
         "prime": True},

        {"name": "mkt_prime_top1pct_tp4_sl4_hold15s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 4, "sl": 4, "hold": 15000, "trail": 0, "market": True, "latency": 5,
         "prime": True},

        # === VERY SHORT HOLD (signal decays fast) ===
        {"name": "mkt_top1pct_tp2_sl2_hold3s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 2, "sl": 2, "hold": 3000, "trail": 0, "market": True, "latency": 5},

        {"name": "mkt_top1pct_tp1_sl1_hold2s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 1, "sl": 1, "hold": 2000, "trail": 0, "market": True, "latency": 5},

        # === NO AGREEMENT REQUIREMENT (more trades) ===
        {"name": "mkt_noagree_top1pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 1.0, "agree": False, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": True, "latency": 5},

        # === PASSIVE LIMIT COMPARISON (same confidence, for reference) ===
        {"name": "pass_top1pct_tp3_sl3_hold10s",
         "conf": 0.0, "pct": 1.0, "agree": True, "long_only": False,
         "tp": 3, "sl": 3, "hold": 10000, "trail": 0, "market": False, "latency": 5},
    ]

    # ========================================================================
    # Run all configs
    # ========================================================================
    results_summary = []

    for ci, cfg in enumerate(configs):
        config_name = cfg["name"]
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Config {ci+1}/{len(configs)}: {config_name}")
        LOG.info(f"{'='*60}")

        config_dir = output_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        for date, pred_file, mbo_file in valid_dates:
            # Create gated predictions
            gated_file = config_dir / f"{date}_gated.npz"
            n_signals = create_gated_predictions(
                pred_file, gated_file,
                min_confidence=cfg.get("conf", 0.0),
                require_agreement=cfg.get("agree", True),
                long_only=cfg.get("long_only", False),
                percentile_threshold=cfg.get("pct", None),
            )

            if n_signals == 0:
                continue

            # Run fill sim
            sim_output = config_dir / f"{date}_trades.json"
            result = run_fill_sim(
                mbo_file=mbo_file,
                pred_file=gated_file,
                output_file=sim_output,
                signal_threshold=0.01,
                hold_ms=cfg.get("hold", 10000),
                trailing_ticks=cfg.get("trail", 0),
                stop_loss_ticks=cfg.get("sl", 4),
                take_profit_ticks=cfg.get("tp", 4),
                max_wait_bars=200,
                latency_ms=cfg.get("latency", 5),
                market_entry=cfg.get("market", False),
                signal_flip_exit=cfg.get("flipex", False),
                prime_hours=cfg.get("prime", False),
            )

            if result:
                n_trades = len(result.get("trades", []))
                if n_trades > 0:
                    all_results.append((date, result))
                    LOG.info(f"  {date}: {n_signals} signals → {n_trades} trades")

        # Analyze
        metrics = analyze_results(all_results, config_name)
        if metrics:
            results_summary.append(metrics)
            LOG.info(f"\n  *** {config_name} ***")
            LOG.info(f"  Trades: {metrics['trades']} ({metrics['trades_per_day']:.1f}/day)")
            LOG.info(f"  WR: {metrics['wr']:.1%}")
            LOG.info(f"  Avg P&L: {metrics['avg_pnl']:.3f} ticks")
            LOG.info(f"  Total P&L: {metrics['total_pnl']:.1f} ticks")
            LOG.info(f"  PF: {metrics['pf']:.3f}")
            LOG.info(f"  Sortino: {metrics['sortino']:.3f}")
            LOG.info(f"  Green: {metrics['green_days']}/{metrics['total_days']} ({metrics['green_pct']:.0%})")
            LOG.info(f"  Avg Win: +{metrics['avg_win']:.2f}  Avg Loss: {metrics['avg_loss']:.2f}")
        else:
            LOG.info(f"  {config_name}: NO TRADES")

    # ========================================================================
    # Summary
    # ========================================================================
    LOG.info("\n" + "=" * 100)
    LOG.info("FINAL SUMMARY — ALL CONFIGS")
    LOG.info("=" * 100)

    # Sort by Sortino (best first)
    results_summary.sort(key=lambda x: x["sortino"], reverse=True)

    header = f"{'Config':<45} {'Trades':>7} {'WR':>6} {'AvgPnL':>8} {'PF':>6} {'Sortino':>8} {'Green%':>7} {'AvgWin':>7} {'AvgLoss':>8}"
    LOG.info(header)
    LOG.info("-" * len(header))

    for m in results_summary:
        line = (f"{m['config']:<45} {m['trades']:>7} {m['wr']:>6.1%} "
                f"{m['avg_pnl']:>+8.3f} {m['pf']:>6.3f} {m['sortino']:>+8.3f} "
                f"{m['green_pct']:>6.0%} {m['avg_win']:>+7.2f} {m['avg_loss']:>+8.2f}")
        LOG.info(line)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    LOG.info(f"\nSummary saved to {summary_file}")

    # Check for ANY profitable config
    profitable = [m for m in results_summary if m["avg_pnl"] > 0]
    if profitable:
        LOG.info(f"\n🏆 FOUND {len(profitable)} PROFITABLE CONFIGS!")
        for p in profitable:
            LOG.info(f"  ✅ {p['config']}: Avg PnL = +{p['avg_pnl']:.3f} ticks, PF={p['pf']:.2f}, Sortino={p['sortino']:.3f}")
    else:
        LOG.info("\n❌ NO PROFITABLE CONFIGS FOUND")
        LOG.info("   Next step: need fundamentally different approach to execution")


if __name__ == "__main__":
    main()
