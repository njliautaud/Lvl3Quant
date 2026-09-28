#!/usr/bin/env python3
"""
FIFO Adverse Selection Analysis
================================

Runs single-date fill sims with various configs and analyzes the per-trade
data to understand:
1. How does queue position affect trade outcome?
2. What's the fill latency distribution?
3. Do high-confidence signals get filled differently?
4. Is adverse selection uniform or concentrated in certain conditions?

This informs whether we can build a "fill quality" predictor.

Author: Claude (Infrastructure Builder)
Date: 2026-05-08
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

LOG = logging.getLogger("ADVERSE_ANALYSIS")
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


def run_fill_sim_raw(mbo_file, pred_file, output_file,
                     tp=4, sl=4, hold_ms=30000, latency_ms=50):
    """Run fill sim and return full JSON including trades."""
    cmd = [
        str(FILL_SIM_BIN),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--signal-threshold", "0.01",
        "--hold-ms", str(hold_ms),
        "--stop-loss-ticks", str(sl),
        "--take-profit-ticks", str(tp),
        "--max-wait-bars", "100",
        "--latency-ms", str(latency_ms),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            LOG.warning(f"Fill sim failed: {result.stderr[:200]}")
            return None
        if Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        LOG.warning(f"Fill sim error: {e}")
    return None


def analyze_trades(trades, config_name):
    """Deep analysis of per-trade data."""
    if not trades:
        return {}

    # Extract fields
    pnl_ticks = [t["pnl_ticks"] for t in trades]
    queue_pos = [t.get("queue_position_at_post", 0) for t in trades]
    book_size = [t.get("book_size_at_post", 0) for t in trades]
    fill_lat_ns = [t.get("fill_latency_ns", 0) for t in trades]
    hold_dur_ns = [t.get("hold_duration_ns", 0) for t in trades]
    mae = [t.get("mae_ticks", 0) for t in trades]
    mfe = [t.get("mfe_ticks", 0) for t in trades]
    signals = [abs(t.get("signal_strength", 0)) for t in trades]
    sides = [t.get("side", "") for t in trades]
    exit_reasons = [t.get("exit_reason", "") for t in trades]

    n = len(trades)
    winners = [t for t in trades if t["pnl_ticks"] > 0]
    losers = [t for t in trades if t["pnl_ticks"] < 0]

    results = {
        "n_trades": n,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "wr": len(winners) / max(n, 1),
        "total_pnl_ticks": sum(pnl_ticks),
        "avg_pnl_ticks": np.mean(pnl_ticks) if pnl_ticks else 0,
        "avg_winner_ticks": np.mean([t["pnl_ticks"] for t in winners]) if winners else 0,
        "avg_loser_ticks": np.mean([t["pnl_ticks"] for t in losers]) if losers else 0,
    }

    # Queue position analysis
    if queue_pos and max(queue_pos) > 0:
        q_arr = np.array(queue_pos)
        pnl_arr = np.array(pnl_ticks)

        # Split by queue position terciles
        q_terciles = np.percentile(q_arr[q_arr > 0], [33, 66]) if sum(q > 0 for q in queue_pos) > 3 else [5, 15]
        front = pnl_arr[q_arr <= q_terciles[0]]
        mid = pnl_arr[(q_arr > q_terciles[0]) & (q_arr <= q_terciles[1])]
        back = pnl_arr[q_arr > q_terciles[1]]

        results["queue_analysis"] = {
            "front_tercile_avg_pnl": float(np.mean(front)) if len(front) > 0 else None,
            "front_tercile_wr": float(np.mean(front > 0)) if len(front) > 0 else None,
            "front_tercile_n": int(len(front)),
            "mid_tercile_avg_pnl": float(np.mean(mid)) if len(mid) > 0 else None,
            "mid_tercile_wr": float(np.mean(mid > 0)) if len(mid) > 0 else None,
            "mid_tercile_n": int(len(mid)),
            "back_tercile_avg_pnl": float(np.mean(back)) if len(back) > 0 else None,
            "back_tercile_wr": float(np.mean(back > 0)) if len(back) > 0 else None,
            "back_tercile_n": int(len(back)),
            "tercile_boundaries": [float(q_terciles[0]), float(q_terciles[1])],
        }

        # Queue position vs PnL correlation
        from scipy.stats import spearmanr
        try:
            corr, pval = spearmanr(queue_pos, pnl_ticks)
            results["queue_pnl_spearman"] = float(corr)
            results["queue_pnl_pval"] = float(pval)
        except:
            pass

    # Signal strength analysis
    if signals and max(signals) > 0:
        sig_arr = np.array(signals)
        pnl_arr = np.array(pnl_ticks)

        sig_med = np.median(sig_arr)
        high_sig = pnl_arr[sig_arr > sig_med]
        low_sig = pnl_arr[sig_arr <= sig_med]

        results["signal_analysis"] = {
            "high_signal_avg_pnl": float(np.mean(high_sig)) if len(high_sig) > 0 else None,
            "high_signal_wr": float(np.mean(high_sig > 0)) if len(high_sig) > 0 else None,
            "high_signal_n": int(len(high_sig)),
            "low_signal_avg_pnl": float(np.mean(low_sig)) if len(low_sig) > 0 else None,
            "low_signal_wr": float(np.mean(low_sig > 0)) if len(low_sig) > 0 else None,
            "low_signal_n": int(len(low_sig)),
            "median_signal": float(sig_med),
        }

        from scipy.stats import spearmanr
        try:
            corr, pval = spearmanr(signals, pnl_ticks)
            results["signal_pnl_spearman"] = float(corr)
            results["signal_pnl_pval"] = float(pval)
        except:
            pass

    # Fill latency analysis
    fill_lat_ms = [ns / 1e6 for ns in fill_lat_ns if ns > 0]
    if fill_lat_ms:
        results["fill_latency_ms"] = {
            "p50": float(np.percentile(fill_lat_ms, 50)),
            "p75": float(np.percentile(fill_lat_ms, 75)),
            "p90": float(np.percentile(fill_lat_ms, 90)),
            "p95": float(np.percentile(fill_lat_ms, 95)),
            "mean": float(np.mean(fill_lat_ms)),
        }

        # Fast fill vs slow fill
        lat_arr = np.array(fill_lat_ms)
        pnl_arr = np.array(pnl_ticks[:len(lat_arr)])
        lat_med = np.median(lat_arr)
        fast = pnl_arr[lat_arr < lat_med]
        slow = pnl_arr[lat_arr >= lat_med]
        results["fill_speed_analysis"] = {
            "fast_fill_avg_pnl": float(np.mean(fast)) if len(fast) > 0 else None,
            "fast_fill_wr": float(np.mean(fast > 0)) if len(fast) > 0 else None,
            "slow_fill_avg_pnl": float(np.mean(slow)) if len(slow) > 0 else None,
            "slow_fill_wr": float(np.mean(slow > 0)) if len(slow) > 0 else None,
        }

    # MFE/MAE analysis
    if mfe and mae:
        results["mfe_mae"] = {
            "avg_mfe": float(np.mean(mfe)),
            "avg_mae": float(np.mean(mae)),
            "mfe_capture": float(np.mean([t["pnl_ticks"] / max(t["mfe_ticks"], 0.01)
                                          for t in trades if t["mfe_ticks"] > 0])),
            "avg_mfe_winners": float(np.mean([t["mfe_ticks"] for t in winners])) if winners else 0,
            "avg_mae_winners": float(np.mean([t["mae_ticks"] for t in winners])) if winners else 0,
            "avg_mfe_losers": float(np.mean([t["mfe_ticks"] for t in losers])) if losers else 0,
            "avg_mae_losers": float(np.mean([t["mae_ticks"] for t in losers])) if losers else 0,
        }

    # Exit reason breakdown
    exit_pnl = defaultdict(list)
    for t in trades:
        exit_pnl[t.get("exit_reason", "unknown")].append(t["pnl_ticks"])
    results["exit_breakdown"] = {
        reason: {
            "n": len(pnls),
            "avg_pnl": float(np.mean(pnls)),
            "wr": float(np.mean([p > 0 for p in pnls])),
        }
        for reason, pnls in exit_pnl.items()
    }

    # Side breakdown
    buy_trades = [t for t in trades if t.get("side") == "BUY"]
    sell_trades = [t for t in trades if t.get("side") == "SELL"]
    results["side_breakdown"] = {
        "BUY": {
            "n": len(buy_trades),
            "avg_pnl": float(np.mean([t["pnl_ticks"] for t in buy_trades])) if buy_trades else 0,
            "wr": float(np.mean([t["pnl_ticks"] > 0 for t in buy_trades])) if buy_trades else 0,
        },
        "SELL": {
            "n": len(sell_trades),
            "avg_pnl": float(np.mean([t["pnl_ticks"] for t in sell_trades])) if sell_trades else 0,
            "wr": float(np.mean([t["pnl_ticks"] > 0 for t in sell_trades])) if sell_trades else 0,
        },
    }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output" / "adverse_analysis_v1"))
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
    LOG.info("FIFO Adverse Selection Analysis")
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
    LOG.info(f"Analyzing {len(valid_dates)} dates")

    # Run fill sim for each date with symmetric TP/SL
    configs = [
        ("tp3_sl3", 3, 3, 20000),
        ("tp4_sl4", 4, 4, 30000),
        ("tp4_sl8", 4, 8, 30000),
    ]

    all_trades_by_config = {name: [] for name, _, _, _ in configs}

    for date, pred_file, mbo_file in valid_dates:
        LOG.info(f"\nDate: {date}")

        for cfg_name, tp, sl, hold_ms in configs:
            result_file = output_dir / f"{date}_{cfg_name}_result.json"
            result = run_fill_sim_raw(mbo_file, pred_file, result_file,
                                      tp=tp, sl=sl, hold_ms=hold_ms)
            if result and "trades" in result:
                trades = result["trades"]
                n = len(trades)
                pnl_dollars = result.get("total_pnl_dollars", 0)
                wr = result.get("win_rate", 0)
                LOG.info(f"  {cfg_name}: {n} trades, WR={wr:.3f}, PnL=${pnl_dollars:+.0f}")
                all_trades_by_config[cfg_name].extend(trades)
            else:
                LOG.info(f"  {cfg_name}: no trades")

    # Aggregate analysis
    LOG.info(f"\n\n{'='*70}")
    LOG.info("AGGREGATE ANALYSIS — All Dates Combined")
    LOG.info(f"{'='*70}")

    all_analysis = {}
    for cfg_name, _, _, _ in configs:
        trades = all_trades_by_config[cfg_name]
        LOG.info(f"\n{'='*50}")
        LOG.info(f"CONFIG: {cfg_name} ({len(trades)} trades)")
        LOG.info(f"{'='*50}")

        analysis = analyze_trades(trades, cfg_name)
        all_analysis[cfg_name] = analysis

        LOG.info(f"  Total trades: {analysis.get('n_trades', 0)}")
        LOG.info(f"  Winners: {analysis.get('n_winners', 0)}, Losers: {analysis.get('n_losers', 0)}")
        LOG.info(f"  WR: {analysis.get('wr', 0):.3f}")
        LOG.info(f"  Total PnL: {analysis.get('total_pnl_ticks', 0):+.1f}tk")
        LOG.info(f"  Avg PnL/trade: {analysis.get('avg_pnl_ticks', 0):+.2f}tk")
        LOG.info(f"  Avg winner: {analysis.get('avg_winner_ticks', 0):+.2f}tk")
        LOG.info(f"  Avg loser: {analysis.get('avg_loser_ticks', 0):+.2f}tk")

        if "queue_analysis" in analysis:
            qa = analysis["queue_analysis"]
            LOG.info(f"\n  QUEUE POSITION ANALYSIS (terciles at {qa['tercile_boundaries']}):")
            LOG.info(f"    Front (queue ≤ {qa['tercile_boundaries'][0]:.0f}): "
                     f"n={qa['front_tercile_n']}, WR={qa['front_tercile_wr']:.3f}, "
                     f"avg PnL={qa['front_tercile_avg_pnl']:+.2f}tk")
            LOG.info(f"    Mid: n={qa['mid_tercile_n']}, WR={qa['mid_tercile_wr']:.3f}, "
                     f"avg PnL={qa['mid_tercile_avg_pnl']:+.2f}tk")
            LOG.info(f"    Back (queue > {qa['tercile_boundaries'][1]:.0f}): "
                     f"n={qa['back_tercile_n']}, WR={qa['back_tercile_wr']:.3f}, "
                     f"avg PnL={qa['back_tercile_avg_pnl']:+.2f}tk")

        if "queue_pnl_spearman" in analysis:
            LOG.info(f"    Queue→PnL Spearman: {analysis['queue_pnl_spearman']:+.3f} "
                     f"(p={analysis['queue_pnl_pval']:.4f})")

        if "signal_analysis" in analysis:
            sa = analysis["signal_analysis"]
            LOG.info(f"\n  SIGNAL STRENGTH ANALYSIS (median={sa['median_signal']:.3f}):")
            LOG.info(f"    High signal: n={sa['high_signal_n']}, WR={sa['high_signal_wr']:.3f}, "
                     f"avg PnL={sa['high_signal_avg_pnl']:+.2f}tk")
            LOG.info(f"    Low signal: n={sa['low_signal_n']}, WR={sa['low_signal_wr']:.3f}, "
                     f"avg PnL={sa['low_signal_avg_pnl']:+.2f}tk")

        if "signal_pnl_spearman" in analysis:
            LOG.info(f"    Signal→PnL Spearman: {analysis['signal_pnl_spearman']:+.3f} "
                     f"(p={analysis['signal_pnl_pval']:.4f})")

        if "fill_speed_analysis" in analysis:
            fsa = analysis["fill_speed_analysis"]
            LOG.info(f"\n  FILL SPEED ANALYSIS:")
            LOG.info(f"    Fast fills: WR={fsa['fast_fill_wr']:.3f}, avg PnL={fsa['fast_fill_avg_pnl']:+.2f}tk")
            LOG.info(f"    Slow fills: WR={fsa['slow_fill_wr']:.3f}, avg PnL={fsa['slow_fill_avg_pnl']:+.2f}tk")

        if "fill_latency_ms" in analysis:
            fl = analysis["fill_latency_ms"]
            LOG.info(f"    Latency: p50={fl['p50']:.0f}ms, p90={fl['p90']:.0f}ms, p95={fl['p95']:.0f}ms")

        if "mfe_mae" in analysis:
            mm = analysis["mfe_mae"]
            LOG.info(f"\n  MFE/MAE ANALYSIS:")
            LOG.info(f"    Avg MFE: {mm['avg_mfe']:.2f}tk, Avg MAE: {mm['avg_mae']:.2f}tk")
            LOG.info(f"    MFE capture ratio: {mm['mfe_capture']:.2f}")
            LOG.info(f"    Winners: MFE={mm['avg_mfe_winners']:.2f}, MAE={mm['avg_mae_winners']:.2f}")
            LOG.info(f"    Losers: MFE={mm['avg_mfe_losers']:.2f}, MAE={mm['avg_mae_losers']:.2f}")

        if "exit_breakdown" in analysis:
            LOG.info(f"\n  EXIT REASON BREAKDOWN:")
            for reason, stats in sorted(analysis["exit_breakdown"].items()):
                LOG.info(f"    {reason}: n={stats['n']}, WR={stats['wr']:.3f}, "
                         f"avg PnL={stats['avg_pnl']:+.2f}tk")

        if "side_breakdown" in analysis:
            LOG.info(f"\n  SIDE BREAKDOWN:")
            for side, stats in analysis["side_breakdown"].items():
                LOG.info(f"    {side}: n={stats['n']}, WR={stats['wr']:.3f}, "
                         f"avg PnL={stats['avg_pnl']:+.2f}tk")

    # Save all analysis
    results_file = output_dir / "analysis_results.json"
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_file, "w") as f:
        json.dump(all_analysis, f, indent=2, default=convert)
    LOG.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
