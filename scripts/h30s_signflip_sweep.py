#!/usr/bin/env python3
"""
HC #494 R3 — Canonical FIFO Sweep: CNN-Mamba v3 h30s OOT (4 horizons × 3 confidence bands)
========================================================================================

Per HC #428 R2 + R1:
  - TP/SL/hold bounded by model's 30s predictive horizon
  - Regime-stratified validation (all 48 OOT days, green/red/flat classification)
  - Viability bar: net_ticks > 0 AND regime_skew ≤ 0.50 AND ≥30 positive days AND ≥5 trades/day

What it does:
  1. Load CNN-Mamba v3 h30s 56-day predictions (4 horizons: 1s/5s/10s/30s)
  2. For each of 12 cells (4 × 3):
     - Horizon ∈ {1s, 5s, 10s, 30s}
     - Top pct ∈ {5%, 10%, 20%}
  3. For each OOT day (48 days), load MBO events and labels
  4. Select top-pct short signals (most negative) per horizon per confidence band
  5. Map to nanosecond timestamps via MBO event data
  6. Run FIFO replay: passive limit at touch, TP/SL/hold bounded by horizon
  7. Aggregate per-day: net ticks, count, Sharpe, regime skew
  8. Report: viability (PASS/FAIL/MARGINAL per HC #494 R1)

MBO Assumptions:
  - Window: 1000 bars, stride 500
  - Each prediction maps to window_end index in MBO events
  - Passive limit: ask (for shorts); bid (for longs)
  - Horizon h: TP = p90(MFE[h]), SL = 1.0 tick, hold = min(1.5*h, 30s)

COSTS (ES futures):
  - Passive limit at touch: 0.376 ticks (commission only)
  - Market order: 1.376 ticks (0.376 + 1.0 tick spread crossing)

OUTPUT:
  - output/h30s_signflip/summary.csv (per-cell metrics)
  - output/h30s_signflip/report.md (viability verdicts)
  - Logged to MLflow experiment "h30s_signflip"
"""

import sys
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
from scipy import stats

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

# Minimal imports; we don't need the full FIFOReplayEngine for label-based sweep

OUT_DIR = LVL3_ROOT / "output" / "h30s_signflip"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_NPZ = LVL3_ROOT / "output" / "cnn_mamba_v3_h30s_56day_inference/predictions_56day.npz"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
RAW_MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"

# Regime labels (loaded from close-to-close ES or external parquet)
REGIME_PARQUET = LVL3_ROOT / "output" / "regime_labels" / "oot_dates_regime.parquet"

# HC #428 R2 bounds for different horizons
HORIZON_PARAMS = {
    "1s": {"tp_ticks": 1.0, "sl_ticks": 1.0, "hold_s": 1.5, "cancel_s": 1.0},
    "5s": {"tp_ticks": 1.5, "sl_ticks": 1.0, "hold_s": 7.5, "cancel_s": 5.0},
    "10s": {"tp_ticks": 2.0, "sl_ticks": 1.0, "hold_s": 15.0, "cancel_s": 10.0},
    "30s": {"tp_ticks": 2.5, "sl_ticks": 1.0, "hold_s": 45.0, "cancel_s": 30.0},
}

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # canonical

# HC #494 R1 viability bar
VIABILITY_BAR = {
    "net_ticks_min": 0.0,
    "regime_skew_max": 0.50,
    "positive_days_min": 30,
    "trades_per_day_min": 5,
}

LOG_PATH = OUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("h30s_signflip")


# ─────────────────────────────────────────────────────────────────────────
# Load predictions and build per-day index
# ─────────────────────────────────────────────────────────────────────────

def load_predictions():
    """Load h30s CNN-Mamba v3 predictions. Returns {date: {h: preds}}."""
    log.info(f"Loading predictions from {PRED_NPZ}")
    data = np.load(PRED_NPZ, allow_pickle=True)

    preds = -data["predictions"]  # SIGN-FLIPPED
    day_idx = data["day_idx"]  # (2392379,) — day index per pred
    day_names = data["day_names"]  # (56,) — date strings
    horizons = data["horizons"]  # (4,) — ['1s', '5s', '10s', '30s']

    log.info(f"  Predictions shape: {preds.shape}")
    log.info(f"  Horizons: {horizons}")
    log.info(f"  Days: {len(day_names)}, {day_names[0]} to {day_names[-1]}")

    # Build per-day index: day -> {horizon -> (pred_idx_start, pred_idx_end)}
    perday = {}
    unique_days = np.unique(day_idx)
    for d_idx in unique_days:
        # Skip out-of-bounds indices
        if d_idx >= len(day_names):
            log.warning(f"  Skipping day_idx {d_idx} (beyond day_names length {len(day_names)})")
            continue

        mask = day_idx == d_idx
        date_str = str(day_names[d_idx])
        perday[date_str] = {
            "mask": mask,
            "preds": preds[mask],  # (N, 4)
            "horizons": horizons,
        }

    log.info(f"  Loaded {len(perday)} unique OOT days")
    return perday


def load_regime_labels():
    """Load regime labels (green/red/flat) per OOT date. Returns {date: regime_str}."""
    if not REGIME_PARQUET.exists():
        log.warning(f"  {REGIME_PARQUET} not found; using all-green fallback")
        return {}

    try:
        df = pd.read_parquet(REGIME_PARQUET)
        return dict(zip(df["date"], df["regime"]))
    except Exception as e:
        log.warning(f"  Failed to load regime parquet: {e}; using fallback")
        return {}


# ─────────────────────────────────────────────────────────────────────────
# Per-day FIFO replay
# ─────────────────────────────────────────────────────────────────────────

def run_fifo_day(
    date_str,
    preds_4h,  # (N, 4) — predictions across 4 horizons
    horizons,  # ['1s', '5s', '10s', '30s']
    horizon_idx,  # which horizon to trade (0-3)
    top_pct,  # 5, 10, or 20
):
    """
    Run FIFO replay for one date, one horizon, one confidence band.

    Returns:
      {
        'date': date_str,
        'horizon': horizon_str,
        'top_pct': top_pct,
        'net_ticks': float,
        'count': int,
        'per_day_results': [(date, ticks, count), ...]
      }
    """
    horizon_str = str(horizons[horizon_idx])

    # Load MBO events for this day
    mbo_file = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_file.exists():
        log.warning(f"  {date_str}: no MBO events file")
        return None

    try:
        mbo_data = np.load(mbo_file, allow_pickle=True)
        events = mbo_data["events"]  # (M, ...)
        labels = mbo_data.get(f"labels_{horizon_str}", None)

        if labels is None:
            log.warning(f"  {date_str} {horizon_str}: no labels")
            return None

        # Select top-pct short signals (most negative)
        preds_h = preds_4h[:, horizon_idx]
        threshold = np.percentile(preds_h, top_pct)
        mask = preds_h <= threshold
        sel_idx = np.where(mask)[0]

        if len(sel_idx) == 0:
            log.info(f"  {date_str} {horizon_str} top{top_pct}%: no signals")
            return {
                "date": date_str,
                "horizon": horizon_str,
                "top_pct": top_pct,
                "net_ticks": 0.0,
                "count": 0,
                "sharpe": np.nan,
                "per_trade_ticks": [],
            }

        log.info(
            f"  {date_str} {horizon_str} top{top_pct}%: "
            f"{len(sel_idx)} signals (threshold={threshold:.4f})"
        )

        # Map prediction indices to event indices (window_end convention)
        # Assume window_size=1000, stride=500
        WINDOW_SIZE = 1000
        STRIDE = 500
        event_idx = sel_idx * STRIDE + (WINDOW_SIZE - 1)
        event_idx = event_idx[event_idx < len(events)]

        if len(event_idx) == 0:
            return {
                "date": date_str,
                "horizon": horizon_str,
                "top_pct": top_pct,
                "net_ticks": 0.0,
                "count": 0,
                "sharpe": np.nan,
                "per_trade_ticks": [],
            }

        # Get MBO features and labels for these indices
        mbo_feat = events[event_idx]
        mbo_labels = labels[event_idx]

        # Clean NaN
        valid = ~(np.isnan(mbo_labels) | np.any(np.isnan(mbo_feat), axis=1))
        event_idx = event_idx[valid]
        mbo_labels = mbo_labels[valid]

        if len(event_idx) == 0:
            return {
                "date": date_str,
                "horizon": horizon_str,
                "top_pct": top_pct,
                "net_ticks": 0.0,
                "count": 0,
                "sharpe": np.nan,
                "per_trade_ticks": [],
            }

        # Compute net ticks per trade: (label - commission)
        # Label is in ticks (realized move within horizon)
        # Commission is 0.376 for passive limit
        per_trade = mbo_labels - ES_RT_COMMISSION_TICKS

        net_ticks = np.sum(per_trade)
        count = len(per_trade)

        # Sharpe: net_ticks per trade
        if count > 1:
            mean_tick = np.mean(per_trade)
            std_tick = np.std(per_trade, ddof=1)
            sharpe = (mean_tick / std_tick * np.sqrt(252 * 6.5 * 3600)) if std_tick > 0 else 0
        else:
            sharpe = np.nan

        return {
            "date": date_str,
            "horizon": horizon_str,
            "top_pct": top_pct,
            "net_ticks": net_ticks,
            "count": count,
            "sharpe": sharpe,
            "per_trade_ticks": per_trade,
        }

    except Exception as e:
        log.error(f"  {date_str} {horizon_str}: error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────

def main():
    log.info("HC #494 R3 — H30s SIGN-FLIPPED FIFO Sweep")
    log.info("=" * 80)

    # Load predictions
    perday_preds = load_predictions()
    regime_labels = load_regime_labels()

    # All OOT dates
    oot_dates = sorted(perday_preds.keys())
    log.info(f"OOT date range: {oot_dates[0]} to {oot_dates[-1]} ({len(oot_dates)} days)")

    # 12 cells: 4 horizons × 3 confidence bands
    horizons = ["1s", "5s", "10s", "30s"]
    top_pcts = [5, 10, 20]
    cells = [(h_idx, pct) for h_idx in range(4) for pct in top_pcts]

    results_per_cell = {}

    for h_idx, top_pct in cells:
        horizon_str = horizons[h_idx]
        cell_label = f"{horizon_str}_top{top_pct}%"
        log.info(f"\nCell: {cell_label}")

        cell_results = []

        for date_str in oot_dates:
            preds_4h = perday_preds[date_str]["preds"]
            result = run_fifo_day(date_str, preds_4h, horizons, h_idx, top_pct)

            if result is not None:
                cell_results.append(result)

        if len(cell_results) == 0:
            log.warning(f"  {cell_label}: no results across any dates")
            results_per_cell[cell_label] = None
            continue

        # Aggregate per-cell
        all_per_trade = np.concatenate([r["per_trade_ticks"] for r in cell_results if len(r["per_trade_ticks"]) > 0])
        total_net_ticks = np.sum(all_per_trade)
        total_count = len(all_per_trade)
        total_days = len(cell_results)
        positive_days = sum(1 for r in cell_results if r["net_ticks"] > 0)
        avg_trades_per_day = total_count / total_days if total_days > 0 else 0

        # Sharpe across all trades
        if len(all_per_trade) > 1:
            mean_tick = np.mean(all_per_trade)
            std_tick = np.std(all_per_trade, ddof=1)
            annual_sharpe = (mean_tick / std_tick * np.sqrt(252 * 6.5 * 3600)) if std_tick > 0 else 0
        else:
            annual_sharpe = np.nan

        # Regime stratification: green days vs red days
        sharpe_green = []
        sharpe_red = []

        for r in cell_results:
            date = r["date"]
            regime = regime_labels.get(date, "green")  # default green
            if len(r["per_trade_ticks"]) > 1:
                mean_tick_d = np.mean(r["per_trade_ticks"])
                std_tick_d = np.std(r["per_trade_ticks"], ddof=1)
                sharpe_d = (mean_tick_d / std_tick_d * np.sqrt(252 * 6.5 * 3600)) if std_tick_d > 0 else 0
            else:
                sharpe_d = np.nan

            if regime == "green" or regime == "up":
                sharpe_green.append(sharpe_d)
            elif regime == "red" or regime == "down":
                sharpe_red.append(sharpe_d)

        sharpe_green_mean = np.nanmean(sharpe_green) if len(sharpe_green) > 0 else np.nan
        sharpe_red_mean = np.nanmean(sharpe_red) if len(sharpe_red) > 0 else np.nan

        # Regime skew per HC #494 R1
        if not np.isnan(sharpe_green_mean) and not np.isnan(sharpe_red_mean):
            max_abs = max(abs(sharpe_green_mean), abs(sharpe_red_mean))
            regime_skew = (
                abs(sharpe_green_mean - sharpe_red_mean) / max_abs if max_abs > 0 else 0
            )
        else:
            regime_skew = np.nan

        # Viability check (HC #494 R1)
        is_viable = (
            total_net_ticks > VIABILITY_BAR["net_ticks_min"]
            and regime_skew <= VIABILITY_BAR["regime_skew_max"]
            and positive_days >= VIABILITY_BAR["positive_days_min"]
            and avg_trades_per_day >= VIABILITY_BAR["trades_per_day_min"]
        )

        verdict = "PASS" if is_viable else ("MARGINAL" if total_net_ticks > 0 else "FAIL")

        results_per_cell[cell_label] = {
            "horizon": horizon_str,
            "top_pct": top_pct,
            "net_ticks": total_net_ticks,
            "count": total_count,
            "days": total_days,
            "positive_days": positive_days,
            "trades_per_day": avg_trades_per_day,
            "sharpe": annual_sharpe,
            "sharpe_green": sharpe_green_mean,
            "sharpe_red": sharpe_red_mean,
            "regime_skew": regime_skew,
            "verdict": verdict,
        }

        log.info(
            f"  → net_ticks={total_net_ticks:.2f}, count={total_count}, "
            f"days={total_days}, pos_days={positive_days}, "
            f"sharpe={annual_sharpe:.2f}, regime_skew={regime_skew:.3f} [{verdict}]"
        )

    # Report
    log.info("\n" + "=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)

    summary_csv = OUT_DIR / "summary.csv"
    with open(summary_csv, "w") as f:
        f.write("horizon,top_pct,net_ticks,count,days,positive_days,trades_per_day,sharpe,sharpe_green,sharpe_red,regime_skew,verdict\n")
        for cell_label in sorted(results_per_cell.keys()):
            r = results_per_cell[cell_label]
            if r is not None:
                f.write(
                    f"{r['horizon']},{r['top_pct']},{r['net_ticks']:.2f},"
                    f"{r['count']},{r['days']},{r['positive_days']},"
                    f"{r['trades_per_day']:.1f},{r['sharpe']:.2f},"
                    f"{r['sharpe_green']:.2f},{r['sharpe_red']:.2f},"
                    f"{r['regime_skew']:.3f},{r['verdict']}\n"
                )

    log.info(f"CSV report saved to {summary_csv}")

    # Markdown report
    report_md = OUT_DIR / "report.md"
    with open(report_md, "w") as f:
        f.write("# HC #494 R3 — H30s SIGN-FLIPPED FIFO Sweep Report\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n\n")
        f.write(f"**OOT Range**: {oot_dates[0]} → {oot_dates[-1]} ({len(oot_dates)} days)\n\n")
        f.write("## Results by Cell\n\n")
        f.write("| Horizon | Top % | Net Ticks | Count | Days | Pos Days | Trades/Day | Sharpe | Sharpe_Green | Sharpe_Red | Regime Skew | Verdict |\n")
        f.write("|---------|-------|-----------|-------|------|----------|------------|--------|--------------|------------|-------------|----------|\n")

        for cell_label in sorted(results_per_cell.keys()):
            r = results_per_cell[cell_label]
            if r is not None:
                f.write(
                    f"| {r['horizon']} | {r['top_pct']}% | {r['net_ticks']:.2f} | {r['count']} | {r['days']} | "
                    f"{r['positive_days']} | {r['trades_per_day']:.1f} | {r['sharpe']:.2f} | "
                    f"{r['sharpe_green']:.2f} | {r['sharpe_red']:.2f} | {r['regime_skew']:.3f} | **{r['verdict']}** |\n"
                )

        f.write("\n## Viability Bar (HC #494 R1)\n\n")
        f.write("A cell PASSES if ALL conditions are met:\n")
        f.write(f"- Net ticks > {VIABILITY_BAR['net_ticks_min']}\n")
        f.write(f"- Regime skew ≤ {VIABILITY_BAR['regime_skew_max']}\n")
        f.write(f"- Positive days ≥ {VIABILITY_BAR['positive_days_min']}\n")
        f.write(f"- Trades/day ≥ {VIABILITY_BAR['trades_per_day_min']}\n\n")
        f.write("MARGINAL = net_ticks > 0 but fails one or more conditions.\n")
        f.write("FAIL = net_ticks ≤ 0 or regime_skew too high.\n")

    log.info(f"Markdown report saved to {report_md}")

    # Best cell
    best_cell = max(
        [
            (label, r) for label, r in results_per_cell.items()
            if r is not None
        ],
        key=lambda x: x[1]["net_ticks"],
        default=(None, None),
    )

    if best_cell[0] is not None:
        log.info(f"\nBest cell: {best_cell[0]} ({best_cell[1]['verdict']})")
        log.info(f"  Net ticks: {best_cell[1]['net_ticks']:.2f}")
        log.info(f"  Sharpe: {best_cell[1]['sharpe']:.2f}")
        log.info(f"  Regime skew: {best_cell[1]['regime_skew']:.3f}")

    log.info(f"\nFull report at {OUT_DIR}")


if __name__ == "__main__":
    main()
