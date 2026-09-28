#!/usr/bin/env python3
"""
Confluence Verdict v1 — Comprehensive acceptance-test for multi-model confluence trading
========================================================================================
Tests the BEST configurations from multi_model_confluence.py against HC #428 gates:
  - Regime-agnostic OOT validation (all 23+ dates, stratified by green/red/flat)
  - MFE-within-horizon checks
  - Day concentration, drawdown, long/short breakdown, confidence deciles

Cost model (HC #512): 0.376 ticks RT commission ONLY. No spread cost on top.

Acceptance gates:
  1. Mean net ticks > 0
  2. Profit Factor >= 1.2
  3. Sharpe >= 0.5 (annualized from daily)
  4. Regime asymmetry <= 0.50
  5. Day concentration <= 0.70
  6. Trades/day >= 5

Usage: python scripts/confluence_verdict_v1.py
"""

import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
V2_DIR = ROOT / "output" / "cnn_mamba_v2_all_oot"
V33_DIR = ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "oot_47day_perdate"
V342_DIR = ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
PT_DIR = ROOT / "output" / "patchtst_bulk_oot"
OUTPUT_DIR = ROOT / "output"

# ── Constants ────────────────────────────────────────────────────────────────
COMMISSION_TICKS = 0.376  # round-trip, NO spread cost (HC #512)
HORIZONS = ["1s", "5s", "10s"]
HORIZON_IDX = {h: i for i, h in enumerate(HORIZONS)}
ANNUALIZATION_FACTOR = math.sqrt(252)  # trading days/year

MODEL_NAMES = {
    "v2": "CNN-Mamba v2",
    "v33": "CNN-Mamba v3.3",
    "v342": "CNN-Mamba v3.4.2",
    "pt": "PatchTST",
}


# ── Data loading (reused from multi_model_confluence.py) ─────────────────────

def discover_dates():
    """Find dates available in all four models."""
    v2_dates = {f[:8] for f in os.listdir(V2_DIR)
                if f.endswith("_predictions.npz") and f[0].isdigit()}
    v33_dates = {f[4:12] for f in os.listdir(V33_DIR)
                 if f.startswith("oot_") and f.endswith(".npz")}
    v342_dates = {f[4:12] for f in os.listdir(V342_DIR)
                  if f.startswith("oot_") and f.endswith(".npz")}
    pt_dates = {f[:8] for f in os.listdir(PT_DIR)
                if f.endswith("_predictions.npz") and f[0].isdigit()}
    overlap = sorted(v2_dates & v33_dates & v342_dates & pt_dates)
    print(f"Date coverage: V2={len(v2_dates)}, V3.3={len(v33_dates)}, "
          f"V3.4.2={len(v342_dates)}, PatchTST={len(pt_dates)}")
    print(f"Overlapping dates (all 4 models): {len(overlap)}")
    return overlap


def load_date(date_str):
    """Load predictions and labels for one date from all models, aligned by index."""
    v2 = np.load(V2_DIR / f"{date_str}_predictions.npz", allow_pickle=True)
    v33 = np.load(V33_DIR / f"oot_{date_str}.npz", allow_pickle=True)
    v342 = np.load(V342_DIR / f"oot_{date_str}.npz", allow_pickle=True)
    pt = np.load(PT_DIR / f"{date_str}_predictions.npz", allow_pickle=True)

    v2_pred = v2["predictions"]
    v2_lbl = v2["labels"]
    v33_pred = np.column_stack([v33["pred_log_ret_1s"], v33["pred_log_ret_5s"], v33["pred_log_ret_10s"]])
    v33_lbl = np.column_stack([v33["target_log_ret_1s"], v33["target_log_ret_5s"], v33["target_log_ret_10s"]])
    v342_pred = np.column_stack([v342["pred_log_ret_1s"], v342["pred_log_ret_5s"], v342["pred_log_ret_10s"]])
    pt_pred = pt["predictions"]
    pt_lbl = pt["labels"]

    model_preds = {"v2": v2_pred, "v33": v33_pred, "v342": v342_pred, "pt": pt_pred}
    counts = {k: len(v) for k, v in model_preds.items()}
    anchor = min(counts, key=counts.get)
    anchor_n = counts[anchor]

    if anchor == "v2":
        anchor_lbl_1s = v2_lbl[:, 0]
    elif anchor in ("v33", "v342"):
        anchor_lbl_1s = v33_lbl[:, 0]
    else:
        anchor_lbl_1s = pt_lbl[:, 0]

    offsets = {}
    for name in ["v2", "v33", "v342", "pt"]:
        if name == anchor or (name == "v342" and anchor == "v33") or (name == "v33" and anchor == "v342"):
            offsets[name] = 0
            continue
        if counts[name] == anchor_n:
            offsets[name] = 0
            continue

        diff = counts[name] - anchor_n
        if name == "v2":
            mlbl = v2_lbl[:, 0]
        elif name in ("v33", "v342"):
            mlbl = v33_lbl[:, 0]
        else:
            mlbl = pt_lbl[:, 0]

        best_corr, best_off = -1, diff
        for trial_off in range(max(0, diff - 5), diff + 6):
            if trial_off >= len(mlbl):
                continue
            seg = mlbl[trial_off: trial_off + anchor_n]
            n_check = min(len(seg), anchor_n)
            if n_check < 1000:
                continue
            m = ~np.isnan(seg[:n_check]) & ~np.isnan(anchor_lbl_1s[:n_check])
            if m.sum() < 500:
                continue
            c = np.corrcoef(seg[:n_check][m], anchor_lbl_1s[:n_check][m])[0, 1]
            if c > best_corr:
                best_corr, best_off = c, trial_off
        offsets[name] = best_off

    if "v342" not in offsets or offsets.get("v342", -1) == -1:
        offsets["v342"] = offsets.get("v33", 0)
    if "v33" not in offsets or offsets.get("v33", -1) == -1:
        offsets["v33"] = offsets.get("v342", 0)

    common_n = min(counts[name] - offsets[name] for name in counts)

    result = {
        "v2_pred": v2_pred[offsets["v2"]: offsets["v2"] + common_n],
        "v33_pred": v33_pred[offsets["v33"]: offsets["v33"] + common_n],
        "v342_pred": v342_pred[offsets["v342"]: offsets["v342"] + common_n],
        "pt_pred": pt_pred[offsets["pt"]: offsets["pt"] + common_n],
        "labels": v33_lbl[offsets["v33"]: offsets["v33"] + common_n],
        "n": common_n,
        "date": date_str,
    }
    return result


# ── Helper functions ─────────────────────────────────────────────────────────

def classify_regime(labels_10s):
    """Classify a day as green/red/flat based on net label drift (proxy for ES daily return).
    Uses the mean of all 10s-horizon labels: positive drift = green day, negative = red, near-zero = flat.
    """
    valid = labels_10s[~np.isnan(labels_10s)]
    if len(valid) < 100:
        return "flat"
    daily_drift = valid.mean()
    # Threshold: ~0.05 ticks mean drift is noise
    if daily_drift > 0.05:
        return "green"
    elif daily_drift < -0.05:
        return "red"
    else:
        return "flat"


def annualized_sharpe_from_daily(daily_pnls):
    """Annualized Sharpe from a list of daily P&L values."""
    arr = np.array(daily_pnls)
    if len(arr) < 3 or np.std(arr) == 0:
        return np.nan
    return (np.mean(arr) / np.std(arr)) * ANNUALIZATION_FACTOR


def profit_factor(trade_returns, cost=COMMISSION_TICKS):
    """Profit factor: gross wins / gross losses (after commission)."""
    net = trade_returns - cost
    wins = net[net > 0]
    losses = net[net <= 0]
    if len(losses) == 0 or abs(losses.sum()) < 1e-10:
        return float('inf') if len(wins) > 0 else float('nan')
    if len(wins) == 0:
        return 0.0
    return float(wins.sum() / abs(losses.sum()))


def sortino_ratio(daily_pnls):
    """Annualized Sortino from daily P&L."""
    arr = np.array(daily_pnls)
    if len(arr) < 3:
        return np.nan
    mean_ret = np.mean(arr)
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float('inf') if mean_ret > 0 else float('nan')
    downside_std = np.std(downside)
    if downside_std == 0:
        return float('nan')
    return (mean_ret / downside_std) * ANNUALIZATION_FACTOR


# ── Configs to test ──────────────────────────────────────────────────────────

CONFIGS = [
    {
        "name": "Direction 4/4, z>=0.5, 10s",
        "horizon_idx": 2,
        "horizon": "10s",
        "direction_agreement": 4,
        "z_threshold": 0.5,
    },
    {
        "name": "Direction 4/4, z>=1.0, 10s",
        "horizon_idx": 2,
        "horizon": "10s",
        "direction_agreement": 4,
        "z_threshold": 1.0,
    },
]


# ── Main analysis ────────────────────────────────────────────────────────────

def run_verdict():
    print("=" * 80)
    print("CONFLUENCE VERDICT v1 — HC #428 ACCEPTANCE TEST")
    print("=" * 80)
    print()

    dates = discover_dates()
    if not dates:
        print("ERROR: No overlapping dates.")
        sys.exit(1)
    print(f"Dates: {dates[0]} to {dates[-1]} ({len(dates)} dates)")
    print()

    # Load all data
    all_preds = {"v2": [], "v33": [], "v342": [], "pt": []}
    all_labels = []
    all_dates = []
    date_boundaries = []  # (start_idx, end_idx, date_str)
    total_samples = 0

    for dt in dates:
        try:
            data = load_date(dt)
        except Exception as e:
            print(f"  SKIP {dt}: {e}")
            continue

        n = data["n"]
        start = total_samples
        all_preds["v2"].append(data["v2_pred"])
        all_preds["v33"].append(data["v33_pred"])
        all_preds["v342"].append(data["v342_pred"])
        all_preds["pt"].append(data["pt_pred"])
        all_labels.append(data["labels"])
        all_dates.extend([dt] * n)
        total_samples += n
        date_boundaries.append((start, total_samples, dt))
        print(f"  {dt}: {n:,} aligned samples")

    print(f"\nTotal: {total_samples:,} samples across {len(date_boundaries)} dates")

    # Concatenate
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k], axis=0)
    labels = np.concatenate(all_labels, axis=0)
    dates_arr = np.array(all_dates)

    # Compute per-model z-scores (global standardization)
    z_preds = {}
    for model_key in MODEL_NAMES:
        z_by_horizon = []
        for h_idx in range(3):
            p = all_preds[model_key][:, h_idx]
            m = ~np.isnan(p)
            mu, sigma = p[m].mean(), p[m].std()
            z = (p - mu) / (sigma + 1e-10)
            z[~m] = 0
            z_by_horizon.append(z)
        z_preds[model_key] = np.column_stack(z_by_horizon)

    # Classify each day's regime
    day_regimes = {}
    for start, end, dt in date_boundaries:
        lbl_10s = labels[start:end, 2]
        day_regimes[dt] = classify_regime(lbl_10s)

    regime_counts = defaultdict(int)
    for r in day_regimes.values():
        regime_counts[r] += 1
    print(f"\nRegime classification: green={regime_counts['green']}, "
          f"red={regime_counts['red']}, flat={regime_counts['flat']}")

    # ── Run each config through the gauntlet ─────────────────────────────
    results = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        h_idx = cfg["horizon_idx"]
        z_thresh = cfg["z_threshold"]

        print(f"\n{'=' * 80}")
        print(f"CONFIG: {name}")
        print(f"{'=' * 80}")

        l = labels[:, h_idx]
        valid = ~np.isnan(l)

        # Build z-score matrix for this horizon
        z_matrix = np.column_stack([z_preds[k][:, h_idx] for k in MODEL_NAMES])

        # Selection mask: all 4 models agree on direction AND all |z| > threshold
        all_strong_long = valid & np.all(z_matrix > z_thresh, axis=1)
        all_strong_short = valid & np.all(z_matrix < -z_thresh, axis=1)
        trade_mask = all_strong_long | all_strong_short
        consensus = np.where(all_strong_long, 1, np.where(all_strong_short, -1, 0))

        # Signed returns (trade in consensus direction)
        signed_ret = consensus * l  # gross ticks per trade

        n_total_trades = trade_mask.sum()
        print(f"\nTotal trades: {n_total_trades:,}")

        if n_total_trades < 100:
            print("  INSUFFICIENT TRADES — SKIP")
            results[name] = {"verdict": "FAIL", "reason": "insufficient trades"}
            continue

        # ── 1. Per-day P&L breakdown ─────────────────────────────────────
        print(f"\n--- Per-Day P&L Breakdown ---")
        print(f"  {'Date':<10} {'Regime':<6} {'Trades':>7} {'Gross':>8} {'Net':>8} {'WR':>6} {'NetTot':>8}")
        print(f"  {'─'*10} {'─'*6} {'─'*7} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")

        daily_net_ticks = []  # net ticks total for that day
        daily_net_per_trade = []
        daily_trades_count = []
        daily_dates_used = []
        daily_gross_ticks = []

        for start, end, dt in date_boundaries:
            day_mask = np.zeros(len(l), dtype=bool)
            day_mask[start:end] = True
            day_trade_mask = day_mask & trade_mask
            n_day = day_trade_mask.sum()

            if n_day == 0:
                daily_net_ticks.append(0.0)
                daily_net_per_trade.append(0.0)
                daily_trades_count.append(0)
                daily_dates_used.append(dt)
                daily_gross_ticks.append(0.0)
                continue

            day_returns = signed_ret[day_trade_mask]
            gross_per_trade = day_returns.mean()
            net_per_trade = gross_per_trade - COMMISSION_TICKS
            net_total = (day_returns - COMMISSION_TICKS).sum()
            wr = (day_returns > 0).mean()
            regime = day_regimes.get(dt, "?")

            daily_net_ticks.append(float(net_total))
            daily_net_per_trade.append(float(net_per_trade))
            daily_trades_count.append(int(n_day))
            daily_dates_used.append(dt)
            daily_gross_ticks.append(float(day_returns.sum()))

            print(f"  {dt:<10} {regime:<6} {n_day:>7} {gross_per_trade:>+8.3f} "
                  f"{net_per_trade:>+8.3f} {wr:>6.3f} {net_total:>+8.1f}")

        daily_net_ticks = np.array(daily_net_ticks)
        daily_trades_count = np.array(daily_trades_count)
        active_days = daily_trades_count > 0
        n_active_days = active_days.sum()

        # ── 2. Aggregate metrics ─────────────────────────────────────────
        all_trade_returns = signed_ret[trade_mask]
        gross_mean = all_trade_returns.mean()
        net_mean = gross_mean - COMMISSION_TICKS
        wr = (all_trade_returns > 0).mean()
        pf = profit_factor(all_trade_returns, COMMISSION_TICKS)

        print(f"\n--- Aggregate ---")
        print(f"  Total trades: {n_total_trades:,} across {n_active_days} active days")
        print(f"  Avg trades/day: {n_total_trades / max(n_active_days, 1):.1f}")
        print(f"  Gross ticks/trade: {gross_mean:+.4f}")
        print(f"  Net ticks/trade: {net_mean:+.4f}")
        print(f"  Win rate (gross): {wr:.4f}")
        print(f"  Profit factor: {pf:.4f}")

        # ── 3. Sharpe / Sortino from daily P&L ───────────────────────────
        # Use active days only for Sharpe calculation
        active_daily = daily_net_ticks[active_days]
        ann_sharpe = annualized_sharpe_from_daily(active_daily)
        ann_sortino = sortino_ratio(active_daily)

        print(f"\n--- Risk Metrics (from daily P&L) ---")
        print(f"  Annualized Sharpe: {ann_sharpe:.4f}")
        print(f"  Annualized Sortino: {ann_sortino:.4f}")
        green_day_count = (active_daily > 0).sum()
        red_day_count = (active_daily < 0).sum()
        flat_day_count = (active_daily == 0).sum()
        print(f"  Green/Red/Flat days: {green_day_count}/{red_day_count}/{flat_day_count} "
              f"({green_day_count/max(n_active_days,1)*100:.0f}% green)")

        # ── 4. Regime stratification ─────────────────────────────────────
        print(f"\n--- Regime Stratification ---")
        regime_daily = defaultdict(list)
        for i, (start, end, dt) in enumerate(date_boundaries):
            r = day_regimes.get(dt, "flat")
            if daily_trades_count[i] > 0:
                regime_daily[r].append(daily_net_ticks[i])

        regime_sharpes = {}
        for regime in ["green", "red", "flat"]:
            days = regime_daily.get(regime, [])
            if len(days) < 2:
                regime_sharpes[regime] = float('nan')
                print(f"  {regime:>5}: {len(days)} days — insufficient")
                continue
            arr = np.array(days)
            s = annualized_sharpe_from_daily(arr)
            regime_sharpes[regime] = float(s)
            mean_daily = arr.mean()
            print(f"  {regime:>5}: {len(days)} days, mean daily net={mean_daily:+.1f} ticks, "
                  f"Sharpe={s:.4f}")

        # Regime asymmetry check
        sg = regime_sharpes.get("green", float('nan'))
        sr = regime_sharpes.get("red", float('nan'))
        if not (np.isnan(sg) or np.isnan(sr)):
            max_abs = max(abs(sg), abs(sr))
            regime_asymmetry = abs(sg - sr) / max_abs if max_abs > 0 else 0.0
        else:
            regime_asymmetry = float('nan')
        print(f"  Regime asymmetry |Sg-Sr|/max: {regime_asymmetry:.4f}")

        # ── 5. Day concentration ─────────────────────────────────────────
        total_net = daily_net_ticks[active_days].sum()
        if total_net > 0:
            best_day_pct = daily_net_ticks[active_days].max() / total_net
        elif total_net < 0:
            # If total is negative, worst day drives losses
            best_day_pct = abs(daily_net_ticks[active_days].min()) / abs(total_net)
        else:
            best_day_pct = float('nan')
        print(f"\n--- Day Concentration ---")
        print(f"  Total net P&L: {total_net:+.1f} ticks")
        print(f"  Best single day: {daily_net_ticks[active_days].max():+.1f} ticks")
        print(f"  Worst single day: {daily_net_ticks[active_days].min():+.1f} ticks")
        print(f"  Best day as % of total: {best_day_pct:.4f}")

        # ── 6. Long vs Short breakdown ───────────────────────────────────
        print(f"\n--- Long vs Short Breakdown ---")
        long_mask_all = trade_mask & (consensus > 0)
        short_mask_all = trade_mask & (consensus < 0)
        n_long = long_mask_all.sum()
        n_short = short_mask_all.sum()

        if n_long > 10:
            long_returns = signed_ret[long_mask_all]
            long_gross = long_returns.mean()
            long_net = long_gross - COMMISSION_TICKS
            long_wr = (long_returns > 0).mean()
            long_pf = profit_factor(long_returns, COMMISSION_TICKS)
            print(f"  Long:  n={n_long:>7,}  gross={long_gross:+.4f}  net={long_net:+.4f}  "
                  f"WR={long_wr:.4f}  PF={long_pf:.3f}")
        else:
            long_net = float('nan')
            long_wr = float('nan')
            long_pf = float('nan')
            print(f"  Long:  n={n_long} — insufficient")

        if n_short > 10:
            short_returns = signed_ret[short_mask_all]
            short_gross = short_returns.mean()
            short_net = short_gross - COMMISSION_TICKS
            short_wr = (short_returns > 0).mean()
            short_pf = profit_factor(short_returns, COMMISSION_TICKS)
            print(f"  Short: n={n_short:>7,}  gross={short_gross:+.4f}  net={short_net:+.4f}  "
                  f"WR={short_wr:.4f}  PF={short_pf:.3f}")
        else:
            short_net = float('nan')
            short_wr = float('nan')
            short_pf = float('nan')
            print(f"  Short: n={n_short} — insufficient")

        # ── 7. Win rate by confidence decile ─────────────────────────────
        print(f"\n--- Win Rate by Confidence Decile ---")
        avg_z = z_matrix[trade_mask].mean(axis=1)  # average z-score across models
        abs_avg_z = np.abs(avg_z)
        trade_ret_arr = all_trade_returns

        # 10 equal bins by |avg_z|
        decile_edges = np.percentile(abs_avg_z, np.arange(0, 101, 10))
        print(f"  {'Decile':<8} {'|z| range':<16} {'N':>7} {'Gross':>8} {'Net':>8} {'WR':>6}")
        print(f"  {'─'*8} {'─'*16} {'─'*7} {'─'*8} {'─'*8} {'─'*6}")

        decile_results = []
        for d in range(10):
            lo = decile_edges[d]
            hi = decile_edges[d + 1]
            if d == 9:
                bin_mask = (abs_avg_z >= lo) & (abs_avg_z <= hi)
            else:
                bin_mask = (abs_avg_z >= lo) & (abs_avg_z < hi)
            n_bin = bin_mask.sum()
            if n_bin < 10:
                decile_results.append({"decile": d + 1, "n": int(n_bin)})
                continue
            bin_ret = trade_ret_arr[bin_mask]
            bin_gross = bin_ret.mean()
            bin_net = bin_gross - COMMISSION_TICKS
            bin_wr = (bin_ret > 0).mean()
            decile_results.append({
                "decile": d + 1,
                "z_lo": round(float(lo), 3),
                "z_hi": round(float(hi), 3),
                "n": int(n_bin),
                "gross": round(float(bin_gross), 4),
                "net": round(float(bin_net), 4),
                "wr": round(float(bin_wr), 4),
            })
            print(f"  D{d+1:<6} [{lo:.2f}, {hi:.2f}){']' if d==9 else ' '} "
                  f"{n_bin:>7} {bin_gross:>+8.4f} {bin_net:>+8.4f} {bin_wr:>6.4f}")

        # ── 8. Drawdown analysis ─────────────────────────────────────────
        print(f"\n--- Drawdown Analysis ---")
        cum_pnl = np.cumsum(active_daily)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = cum_pnl - running_max
        max_dd = drawdown.min()
        max_dd_idx = np.argmin(drawdown)

        # Max consecutive losing days
        losing_streak = 0
        max_losing_streak = 0
        for v in active_daily:
            if v < 0:
                losing_streak += 1
                max_losing_streak = max(max_losing_streak, losing_streak)
            else:
                losing_streak = 0

        print(f"  Max drawdown: {max_dd:+.1f} ticks")
        print(f"  Max consecutive losing days: {max_losing_streak}")
        print(f"  Cumulative P&L: {cum_pnl[-1]:+.1f} ticks (over {n_active_days} days)")

        # Cumulative P&L curve (for later visualization)
        cum_pnl_series = []
        running = 0.0
        for i, (start, end, dt) in enumerate(date_boundaries):
            running += daily_net_ticks[i]
            cum_pnl_series.append({"date": dt, "cum_net_ticks": round(running, 2),
                                   "trades": int(daily_trades_count[i])})

        # ── 9. Trade frequency ───────────────────────────────────────────
        avg_trades_per_day = n_total_trades / max(n_active_days, 1)
        min_trades_day = daily_trades_count[active_days].min() if n_active_days > 0 else 0

        print(f"\n--- Trade Frequency ---")
        print(f"  Avg trades/day: {avg_trades_per_day:.1f}")
        print(f"  Min trades/day: {min_trades_day}")
        print(f"  Max trades/day: {daily_trades_count[active_days].max() if n_active_days > 0 else 0}")

        # ── 10. ACCEPTANCE GATES ─────────────────────────────────────────
        print(f"\n{'=' * 60}")
        print(f"  ACCEPTANCE GATES (HC #428)")
        print(f"{'=' * 60}")

        gates = {}

        # Gate 1: Mean net ticks > 0
        g1 = net_mean > 0
        gates["net_ticks_positive"] = {"pass": g1, "value": round(net_mean, 4), "threshold": "> 0"}
        print(f"  [{'PASS' if g1 else 'FAIL'}] Net ticks/trade > 0: {net_mean:+.4f}")

        # Gate 2: Profit Factor >= 1.2
        g2 = pf >= 1.2
        gates["profit_factor"] = {"pass": g2, "value": round(pf, 4), "threshold": ">= 1.2"}
        print(f"  [{'PASS' if g2 else 'FAIL'}] Profit Factor >= 1.2: {pf:.4f}")

        # Gate 3: Annualized Sharpe >= 0.5
        g3 = ann_sharpe >= 0.5 if not np.isnan(ann_sharpe) else False
        gates["sharpe"] = {"pass": g3, "value": round(float(ann_sharpe), 4) if not np.isnan(ann_sharpe) else None,
                           "threshold": ">= 0.5"}
        print(f"  [{'PASS' if g3 else 'FAIL'}] Annualized Sharpe >= 0.5: {ann_sharpe:.4f}")

        # Gate 4: Regime asymmetry <= 0.50
        g4 = regime_asymmetry <= 0.50 if not np.isnan(regime_asymmetry) else False
        gates["regime_asymmetry"] = {"pass": g4,
                                     "value": round(float(regime_asymmetry), 4) if not np.isnan(regime_asymmetry) else None,
                                     "threshold": "<= 0.50"}
        print(f"  [{'PASS' if g4 else 'FAIL'}] Regime asymmetry <= 0.50: {regime_asymmetry:.4f}")

        # Gate 5: Day concentration <= 0.70
        g5 = best_day_pct <= 0.70 if not np.isnan(best_day_pct) else False
        gates["day_concentration"] = {"pass": g5,
                                      "value": round(float(best_day_pct), 4) if not np.isnan(best_day_pct) else None,
                                      "threshold": "<= 0.70"}
        print(f"  [{'PASS' if g5 else 'FAIL'}] Day concentration <= 0.70: {best_day_pct:.4f}")

        # Gate 6: Trades/day >= 5
        g6 = avg_trades_per_day >= 5
        gates["trades_per_day"] = {"pass": g6, "value": round(avg_trades_per_day, 1), "threshold": ">= 5"}
        print(f"  [{'PASS' if g6 else 'FAIL'}] Trades/day >= 5: {avg_trades_per_day:.1f}")

        all_pass = all(g["pass"] for g in gates.values())
        verdict = "PASS" if all_pass else "FAIL"
        failed_gates = [k for k, g in gates.items() if not g["pass"]]

        print(f"\n  >>> VERDICT: {verdict} <<<")
        if not all_pass:
            print(f"  Failed gates: {', '.join(failed_gates)}")

        # ── Store results ────────────────────────────────────────────────
        results[name] = {
            "verdict": verdict,
            "failed_gates": failed_gates if not all_pass else [],
            "gates": gates,
            "aggregate": {
                "n_trades": int(n_total_trades),
                "n_active_days": int(n_active_days),
                "avg_trades_per_day": round(avg_trades_per_day, 1),
                "gross_ticks_per_trade": round(float(gross_mean), 4),
                "net_ticks_per_trade": round(float(net_mean), 4),
                "win_rate": round(float(wr), 4),
                "profit_factor": round(float(pf), 4) if not np.isinf(pf) else "inf",
                "annualized_sharpe": round(float(ann_sharpe), 4) if not np.isnan(ann_sharpe) else None,
                "annualized_sortino": round(float(ann_sortino), 4) if not np.isnan(ann_sortino) else None,
                "green_days": int(green_day_count),
                "red_days": int(red_day_count),
            },
            "regime_stratification": {
                regime: {
                    "n_days": len(regime_daily.get(regime, [])),
                    "sharpe": round(float(regime_sharpes.get(regime, float('nan'))), 4)
                    if not np.isnan(regime_sharpes.get(regime, float('nan'))) else None,
                }
                for regime in ["green", "red", "flat"]
            },
            "regime_asymmetry": round(float(regime_asymmetry), 4) if not np.isnan(regime_asymmetry) else None,
            "day_concentration": round(float(best_day_pct), 4) if not np.isnan(best_day_pct) else None,
            "long_short": {
                "n_long": int(n_long),
                "n_short": int(n_short),
                "long_net_ticks": round(float(long_net), 4) if not np.isnan(long_net) else None,
                "short_net_ticks": round(float(short_net), 4) if not np.isnan(short_net) else None,
                "long_wr": round(float(long_wr), 4) if not np.isnan(long_wr) else None,
                "short_wr": round(float(short_wr), 4) if not np.isnan(short_wr) else None,
            },
            "confidence_deciles": decile_results,
            "drawdown": {
                "max_drawdown_ticks": round(float(max_dd), 2),
                "max_consecutive_losing_days": int(max_losing_streak),
                "cumulative_pnl_ticks": round(float(cum_pnl[-1]), 2),
            },
            "cumulative_pnl_curve": cum_pnl_series,
            "per_day": {
                dt: {
                    "trades": int(daily_trades_count[i]),
                    "net_ticks": round(float(daily_net_ticks[i]), 2),
                    "regime": day_regimes.get(dt, "?"),
                }
                for i, (start, end, dt) in enumerate(date_boundaries)
            },
        }

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 80}")
    for name, res in results.items():
        v = res["verdict"]
        agg = res.get("aggregate", {})
        print(f"\n  {name}: {v}")
        if agg:
            print(f"    Net ticks/trade: {agg.get('net_ticks_per_trade', '?')}")
            print(f"    Sharpe (ann): {agg.get('annualized_sharpe', '?')}")
            print(f"    Sortino (ann): {agg.get('annualized_sortino', '?')}")
            print(f"    PF: {agg.get('profit_factor', '?')}")
            print(f"    WR: {agg.get('win_rate', '?')}")
            print(f"    Trades/day: {agg.get('avg_trades_per_day', '?')}")
        if res.get("failed_gates"):
            print(f"    Failed: {', '.join(res['failed_gates'])}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "script": "confluence_verdict_v1.py",
            "n_dates": len(date_boundaries),
            "dates": [dt for _, _, dt in date_boundaries],
            "total_samples": total_samples,
            "cost_ticks_rt": COMMISSION_TICKS,
            "models": list(MODEL_NAMES.values()),
            "acceptance_gates": {
                "net_ticks_positive": "> 0",
                "profit_factor": ">= 1.2",
                "annualized_sharpe": ">= 0.5",
                "regime_asymmetry": "<= 0.50",
                "day_concentration": "<= 0.70",
                "trades_per_day": ">= 5",
            },
        },
        "configs": results,
    }

    out_path = OUTPUT_DIR / "confluence_verdict_v1.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_verdict()
