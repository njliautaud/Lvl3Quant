#!/usr/bin/env python3
"""
tier3_profit_eval.py -- Comprehensive Tier 3 Profit Translation Evaluation
===========================================================================
Computes profit-based metrics from OOT prediction files (.npz format).

Context: ES futures (MBO event-driven). Predictions at 1s/5s/10s horizons.
Evaluated at confidence tiers: All / 50% / 25% / 10% / 5% / 1%.

Cost assumptions for ES futures (AMP Futures):
  - Tick size: $12.50 per tick (0.25 points)
  - Commission: $4.70 RT = 0.376 ticks (HC #52)
  - Spread: VARIABLE — measure from data, do NOT assume fixed
  - Default cost: 0.376 ticks (commission only)

Tiers:
  Tier 1: IC, DA, MagCorr at each confidence tier
  Tier 2: MFE/MAE proxies, win/loss ratio, long/short breakdown
  Tier 3: Cost-adjusted P&L, Sortino, profit factor, drawdown, equity curve

Usage (CLI):
    python tier3_profit_eval.py --pred-file /path/to/concat_oot_predictions.npz --horizon 10s --cost 0.376
    python tier3_profit_eval.py --pred-dir /path/to/fold_predictions/ --horizon 10s --cost 0.376

Usage (importable):
    from evaluation.tier3_profit_eval import evaluate_tier3, evaluate_all_tiers
    results = evaluate_tier3(predictions, labels, cost_ticks=0.376)
    full = evaluate_all_tiers(predictions, labels, cost_ticks=0.376)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

# ---- Constants ---------------------------------------------------------------

TICK_VALUE_USD = 12.50
DEFAULT_COST_TICKS = 0.376  # Commission only ($4.70 RT / $12.50 tick). Spread is variable — add separately per order type.
TRADING_HOURS_PER_DAY = 6.5
TRADING_DAYS_PER_YEAR = 252

# Confidence tiers: name -> percentile threshold (of |prediction| magnitude)
CONFIDENCE_TIERS = {
    "All":  0,
    "50%":  50,
    "25%":  75,
    "10%":  90,
    "5%":   95,
    "1%":   99,
}

# ---- NPZ Loading -------------------------------------------------------------

# Known key patterns for predictions and labels across the codebase
_PRED_KEY_PATTERNS = [
    "preds_{h}", "predictions_{h}", "pred_{h}", "y_pred_{h}",
    "preds", "predictions", "pred", "y_pred",
]
_LABEL_KEY_PATTERNS = [
    "labels_{h}", "label_{h}", "y_true_{h}",
    "labels", "label", "y_true",
]


def _find_key(npz_keys: list, patterns: list, horizon: Optional[str] = None) -> Optional[str]:
    """Find the first matching key from pattern list."""
    for pat in patterns:
        if horizon:
            candidate = pat.format(h=horizon)
        else:
            candidate = pat.replace("_{h}", "")
        if candidate in npz_keys:
            return candidate
    return None


def load_predictions(
    path: str | Path,
    horizon: str = "10s",
    horizon_idx: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load predictions and labels from an .npz file.

    Handles multiple formats:
      - preds_10s / labels_10s  (concat files with named horizon arrays)
      - predictions / labels with shape (N, 3) where col index selects horizon
      - predictions / labels with shape (N,) — single horizon
      - Date-keyed files: '2025-12-01_predictions', '2025-12-01_labels'

    Args:
        path: Path to .npz file
        horizon: Horizon string like '1s', '5s', '10s'
        horizon_idx: If predictions are (N,3), which column (0=1s, 1=5s, 2=10s).
                     Auto-detected from horizon if not provided.

    Returns:
        (predictions, labels) as 1D float arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    keys = list(data.keys())

    # Auto-detect horizon index
    if horizon_idx is None:
        horizon_map = {"1s": 0, "5s": 1, "10s": 2}
        horizon_idx = horizon_map.get(horizon, 2)

    # Strategy 1: Horizon-specific keys (preds_10s / labels_10s)
    pred_key = _find_key(keys, _PRED_KEY_PATTERNS, horizon)
    label_key = _find_key(keys, _LABEL_KEY_PATTERNS, horizon)

    if pred_key and label_key:
        preds = np.asarray(data[pred_key], dtype=np.float64).ravel()
        labels = np.asarray(data[label_key], dtype=np.float64).ravel()
        return _clean_arrays(preds, labels)

    # Strategy 2: Generic keys with multi-column shape
    pred_key = _find_key(keys, _PRED_KEY_PATTERNS, horizon=None)
    label_key = _find_key(keys, _LABEL_KEY_PATTERNS, horizon=None)

    if pred_key and label_key:
        preds_raw = np.asarray(data[pred_key], dtype=np.float64)
        labels_raw = np.asarray(data[label_key], dtype=np.float64)

        if preds_raw.ndim == 2 and preds_raw.shape[1] >= horizon_idx + 1:
            preds = preds_raw[:, horizon_idx]
            labels = labels_raw[:, horizon_idx]
        else:
            preds = preds_raw.ravel()
            labels = labels_raw.ravel()

        return _clean_arrays(preds, labels)

    # Strategy 3: Date-keyed files (e.g., '2025-12-01_predictions')
    date_preds = []
    date_labels = []
    for k in sorted(keys):
        if k.endswith("_predictions") or k.endswith("_preds"):
            date_str = k.rsplit("_", 1)[0]
            label_candidates = [f"{date_str}_labels", f"{date_str}_label"]
            for lc in label_candidates:
                if lc in keys:
                    p = np.asarray(data[k], dtype=np.float64).ravel()
                    l = np.asarray(data[lc], dtype=np.float64).ravel()
                    date_preds.append(p)
                    date_labels.append(l)
                    break

    if date_preds:
        preds = np.concatenate(date_preds)
        labels = np.concatenate(date_labels)
        return _clean_arrays(preds, labels)

    raise ValueError(
        f"Cannot find prediction/label keys in {path.name}. "
        f"Available keys: {keys}"
    )


def load_predictions_dir(
    pred_dir: str | Path,
    horizon: str = "10s",
    concat_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate predictions from a directory of .npz files.

    Prefers 'concat_oot_predictions.npz' if it exists.
    Otherwise concatenates all fold_*_oot_predictions.npz files.
    """
    pred_dir = Path(pred_dir)
    if not pred_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {pred_dir}")

    # Prefer concat file
    concat_file = pred_dir / "concat_oot_predictions.npz"
    if concat_file.exists():
        return load_predictions(concat_file, horizon)

    # Also check parent for concat
    for parent_concat in pred_dir.parent.glob("*concat*oot*predictions*.npz"):
        try:
            return load_predictions(parent_concat, horizon)
        except ValueError:
            continue

    # Fall back to individual fold files
    fold_files = sorted(pred_dir.glob("fold_*_oot*.npz"))
    if not fold_files:
        fold_files = sorted(pred_dir.glob("*.npz"))

    if not fold_files:
        raise FileNotFoundError(f"No .npz files found in {pred_dir}")

    all_preds, all_labels = [], []
    for f in fold_files:
        try:
            p, l = load_predictions(f, horizon)
            all_preds.append(p)
            all_labels.append(l)
        except (ValueError, KeyError) as e:
            warnings.warn(f"Skipping {f.name}: {e}")

    if not all_preds:
        raise ValueError(f"No valid prediction files loaded from {pred_dir}")

    return np.concatenate(all_preds), np.concatenate(all_labels)


def _clean_arrays(preds: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN/Inf values from both arrays."""
    valid = np.isfinite(preds) & np.isfinite(labels)
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        warnings.warn(f"Removed {n_invalid:,} NaN/Inf values ({n_invalid/len(preds)*100:.1f}%)")
    return preds[valid], labels[valid]


# ---- Tier 1: Statistical Metrics ---------------------------------------------

def compute_tier1(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Tier 1 metrics at each confidence tier.
      - IC (Spearman rank correlation)
      - DA (Directional Accuracy)
      - MagCorr (Pearson correlation of magnitudes)
    """
    results = {}
    abs_preds = np.abs(predictions)

    for tier_name, pct in CONFIDENCE_TIERS.items():
        if pct > 0:
            thresh = np.percentile(abs_preds, pct)
            mask = abs_preds >= thresh
        else:
            mask = np.ones(len(predictions), dtype=bool)

        p = predictions[mask]
        l = labels[mask]
        n = len(p)

        if n < 10:
            results[tier_name] = {"n": n, "ic": float("nan"), "da": float("nan"), "mag_corr": float("nan")}
            continue

        # IC (Spearman)
        if spearmanr is not None:
            ic_val, _ = spearmanr(p, l)
            ic_val = float(ic_val) if np.isfinite(ic_val) else float("nan")
        else:
            # Fallback: Pearson on ranks
            rp = np.argsort(np.argsort(p)).astype(float)
            rl = np.argsort(np.argsort(l)).astype(float)
            ic_val = float(np.corrcoef(rp, rl)[0, 1])

        # DA (Directional Accuracy)
        pred_dir = np.sign(p)
        label_dir = np.sign(l)
        # Exclude zero-label events from DA calc
        nz = label_dir != 0
        da = float(np.mean(pred_dir[nz] == label_dir[nz])) if nz.sum() > 0 else float("nan")

        # MagCorr (Pearson correlation of |pred| vs |label|)
        mag_corr = float(np.corrcoef(np.abs(p), np.abs(l))[0, 1])
        if not np.isfinite(mag_corr):
            mag_corr = float("nan")

        results[tier_name] = {
            "n": n,
            "ic": round(ic_val, 4),
            "da": round(da, 4),
            "mag_corr": round(mag_corr, 4),
        }

    return results


# ---- Tier 2: Trade Quality Metrics -------------------------------------------

def compute_tier2(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Tier 2 metrics at each confidence tier.
      - Win rate (direction correct)
      - Avg winner / avg loser (in ticks, raw)
      - Win/loss ratio
      - Long-only DA / Short-only DA
      - MFE proxy: avg positive outcome for correct-direction trades
      - MAE proxy: avg negative outcome for wrong-direction trades
    """
    results = {}
    abs_preds = np.abs(predictions)

    for tier_name, pct in CONFIDENCE_TIERS.items():
        if pct > 0:
            thresh = np.percentile(abs_preds, pct)
            mask = abs_preds >= thresh
        else:
            mask = np.ones(len(predictions), dtype=bool)

        p = predictions[mask]
        l = labels[mask]
        n = len(p)

        if n < 10:
            results[tier_name] = {"n": n}
            continue

        # Per-trade raw P&L: sign(pred) * label
        # If we trade in the predicted direction, our raw pnl is sign(pred) * actual_move
        raw_pnl = np.sign(p) * l

        winners = raw_pnl[raw_pnl > 0]
        losers = raw_pnl[raw_pnl < 0]
        flat = raw_pnl[raw_pnl == 0]

        win_rate = len(winners) / n if n > 0 else float("nan")
        avg_winner = float(np.mean(winners)) if len(winners) > 0 else 0.0
        avg_loser = float(np.mean(losers)) if len(losers) > 0 else 0.0
        wl_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else float("inf")

        # Long/short breakdown
        long_mask = p > 0
        short_mask = p < 0

        long_da = float("nan")
        short_da = float("nan")
        if long_mask.sum() > 10:
            long_correct = (l[long_mask] > 0).sum()
            long_da = float(long_correct / long_mask.sum())
        if short_mask.sum() > 10:
            short_correct = (l[short_mask] < 0).sum()
            short_da = float(short_correct / short_mask.sum())

        # MFE/MAE proxies
        correct_dir = raw_pnl > 0
        mfe_proxy = float(np.mean(l[correct_dir] * np.sign(p[correct_dir]))) if correct_dir.sum() > 0 else 0.0
        wrong_dir = raw_pnl < 0
        mae_proxy = float(np.mean(l[wrong_dir] * np.sign(p[wrong_dir]))) if wrong_dir.sum() > 0 else 0.0

        results[tier_name] = {
            "n": n,
            "win_rate": round(win_rate, 4),
            "avg_winner_ticks": round(avg_winner, 3),
            "avg_loser_ticks": round(avg_loser, 3),
            "wl_ratio": round(wl_ratio, 3),
            "n_winners": int(len(winners)),
            "n_losers": int(len(losers)),
            "n_flat": int(len(flat)),
            "long_da": round(long_da, 4),
            "short_da": round(short_da, 4),
            "mfe_proxy_ticks": round(mfe_proxy, 3),
            "mae_proxy_ticks": round(mae_proxy, 3),
        }

    return results


# ---- Tier 3: Profit-Based Metrics --------------------------------------------

def compute_tier3(
    predictions: np.ndarray,
    labels: np.ndarray,
    cost_ticks: float = DEFAULT_COST_TICKS,
) -> dict:
    """
    Tier 3 profit metrics at each confidence tier.
      - Raw & cost-adjusted P&L per trade
      - Cost-adjusted win rate
      - Net expectancy
      - Sortino ratio (annualized)
      - Profit factor
      - Max drawdown (from cumulative P&L)
      - Avg winner / avg loser (after costs)
      - Equity curve data
    """
    results = {}
    abs_preds = np.abs(predictions)

    for tier_name, pct in CONFIDENCE_TIERS.items():
        if pct > 0:
            thresh = np.percentile(abs_preds, pct)
            mask = abs_preds >= thresh
        else:
            mask = np.ones(len(predictions), dtype=bool)

        p = predictions[mask]
        l = labels[mask]
        n = len(p)

        if n < 10:
            results[tier_name] = {
                "n": n,
                "raw_mean_pnl": float("nan"),
                "cost_adj_mean_pnl": float("nan"),
                "cost_adj_win_rate": float("nan"),
                "net_expectancy": float("nan"),
                "sortino": float("nan"),
                "profit_factor": float("nan"),
                "max_drawdown_ticks": float("nan"),
                "avg_winner_ticks": float("nan"),
                "avg_loser_ticks": float("nan"),
                "total_pnl_ticks": float("nan"),
                "total_pnl_usd": float("nan"),
            }
            continue

        # Raw P&L per trade: sign(prediction) * actual_move
        raw_pnl = np.sign(p) * l

        # Cost-adjusted P&L: deduct cost from each trade
        cost_adj_pnl = raw_pnl - cost_ticks

        # Raw stats
        raw_mean = float(np.mean(raw_pnl))
        raw_total = float(np.sum(raw_pnl))

        # Cost-adjusted stats
        cost_adj_mean = float(np.mean(cost_adj_pnl))
        cost_adj_total = float(np.sum(cost_adj_pnl))

        # Cost-adjusted win rate: trades where cost_adj_pnl > 0
        cost_adj_winners = cost_adj_pnl[cost_adj_pnl > 0]
        cost_adj_losers = cost_adj_pnl[cost_adj_pnl <= 0]
        cost_adj_win_rate = len(cost_adj_winners) / n

        # Net expectancy = mean cost-adjusted P&L
        net_expectancy = cost_adj_mean

        # Avg winner / avg loser (after costs)
        avg_winner = float(np.mean(cost_adj_winners)) if len(cost_adj_winners) > 0 else 0.0
        avg_loser = float(np.mean(cost_adj_losers)) if len(cost_adj_losers) > 0 else 0.0

        # Sortino ratio (annualized)
        # Downside deviation: std of negative returns only
        negative_returns = cost_adj_pnl[cost_adj_pnl < 0]
        if len(negative_returns) > 1:
            downside_dev = float(np.std(negative_returns, ddof=1))
        else:
            downside_dev = 0.0

        if downside_dev > 1e-10:
            # Annualize: assume each "trade" is roughly independent
            # trades_per_day depends on tier selectivity
            # We use sqrt(trades_per_year) for annualization
            # Approximate: total N trades happened over some period.
            # For simplicity, annualize as: mean / downside_dev * sqrt(252)
            # This treats each trade as a daily return (conservative for HF)
            sortino = float(cost_adj_mean / downside_dev * np.sqrt(TRADING_DAYS_PER_YEAR))
        else:
            sortino = float("inf") if cost_adj_mean > 0 else float("nan")

        # Profit factor
        gross_profit = float(np.sum(cost_adj_winners)) if len(cost_adj_winners) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(cost_adj_losers))) if len(cost_adj_losers) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")

        # Max drawdown from cumulative P&L curve
        equity_curve = np.cumsum(cost_adj_pnl)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - running_max
        max_dd = float(np.min(drawdowns))  # most negative

        results[tier_name] = {
            "n": n,
            "cost_ticks": cost_ticks,
            "raw_mean_pnl": round(raw_mean, 4),
            "raw_total_pnl": round(raw_total, 2),
            "cost_adj_mean_pnl": round(cost_adj_mean, 4),
            "cost_adj_total_pnl": round(cost_adj_total, 2),
            "cost_adj_win_rate": round(cost_adj_win_rate, 4),
            "net_expectancy": round(net_expectancy, 4),
            "sortino": round(sortino, 3) if np.isfinite(sortino) else sortino,
            "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else profit_factor,
            "max_drawdown_ticks": round(max_dd, 2),
            "max_drawdown_usd": round(max_dd * TICK_VALUE_USD, 2),
            "avg_winner_ticks": round(avg_winner, 3),
            "avg_loser_ticks": round(avg_loser, 3),
            "total_pnl_ticks": round(cost_adj_total, 2),
            "total_pnl_usd": round(cost_adj_total * TICK_VALUE_USD, 2),
            "gross_profit_ticks": round(gross_profit, 2),
            "gross_loss_ticks": round(gross_loss, 2),
            "n_winners": int(len(cost_adj_winners)),
            "n_losers": int(len(cost_adj_losers)),
            # Equity curve stored separately (large array)
            "_equity_curve": equity_curve,
        }

    return results


# ---- Combined Evaluation ------------------------------------------------------

def evaluate_tier3(
    predictions: np.ndarray,
    labels: np.ndarray,
    cost_ticks: float = DEFAULT_COST_TICKS,
) -> dict:
    """
    Main entry point: compute all Tier 3 profit metrics.
    Returns dict keyed by confidence tier.
    """
    return compute_tier3(predictions, labels, cost_ticks)


def evaluate_all_tiers(
    predictions: np.ndarray,
    labels: np.ndarray,
    cost_ticks: float = DEFAULT_COST_TICKS,
) -> dict:
    """
    Run Tier 1 + Tier 2 + Tier 3 evaluation.
    Returns: { 'tier1': {...}, 'tier2': {...}, 'tier3': {...}, 'summary': {...} }
    """
    t1 = compute_tier1(predictions, labels)
    t2 = compute_tier2(predictions, labels)
    t3 = compute_tier3(predictions, labels, cost_ticks)

    # Build summary: one row per confidence tier with key metrics from all tiers
    summary = {}
    for tier_name in CONFIDENCE_TIERS:
        row = {"tier": tier_name}
        if tier_name in t1:
            row.update({f"t1_{k}": v for k, v in t1[tier_name].items()})
        if tier_name in t2:
            row.update({f"t2_{k}": v for k, v in t2[tier_name].items()})
        if tier_name in t3:
            # Exclude equity curve from summary
            row.update({
                f"t3_{k}": v for k, v in t3[tier_name].items()
                if not k.startswith("_")
            })
        summary[tier_name] = row

    return {
        "tier1": t1,
        "tier2": t2,
        "tier3": t3,
        "summary": summary,
    }


# ---- Formatting: Console Table -----------------------------------------------

def print_tier1_table(tier1: dict) -> str:
    """Print Tier 1 table and return as string."""
    lines = []
    lines.append("")
    lines.append("=" * 65)
    lines.append("TIER 1: Statistical Signal Quality")
    lines.append("=" * 65)
    lines.append(f"{'Tier':>6}  {'N':>10}  {'IC':>8}  {'DA':>7}  {'MagCorr':>8}")
    lines.append("-" * 65)

    for tier_name in CONFIDENCE_TIERS:
        d = tier1.get(tier_name, {})
        n = d.get("n", 0)
        ic = d.get("ic", float("nan"))
        da = d.get("da", float("nan"))
        mc = d.get("mag_corr", float("nan"))
        lines.append(
            f"{tier_name:>6}  {n:>10,}  {_fmt(ic, 4):>8}  {_fmt(da, 3):>7}  {_fmt(mc, 4):>8}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def print_tier2_table(tier2: dict) -> str:
    """Print Tier 2 table and return as string."""
    lines = []
    lines.append("")
    lines.append("=" * 95)
    lines.append("TIER 2: Trade Quality (Raw, Before Costs)")
    lines.append("=" * 95)
    lines.append(
        f"{'Tier':>6}  {'N':>10}  {'WinRate':>7}  {'AvgWin':>7}  {'AvgLoss':>7}  "
        f"{'W/L':>5}  {'LongDA':>7}  {'ShortDA':>7}  {'MFE':>7}  {'MAE':>7}"
    )
    lines.append("-" * 95)

    for tier_name in CONFIDENCE_TIERS:
        d = tier2.get(tier_name, {})
        n = d.get("n", 0)
        wr = d.get("win_rate", float("nan"))
        aw = d.get("avg_winner_ticks", float("nan"))
        al = d.get("avg_loser_ticks", float("nan"))
        wl = d.get("wl_ratio", float("nan"))
        lda = d.get("long_da", float("nan"))
        sda = d.get("short_da", float("nan"))
        mfe = d.get("mfe_proxy_ticks", float("nan"))
        mae = d.get("mae_proxy_ticks", float("nan"))
        lines.append(
            f"{tier_name:>6}  {n:>10,}  {_fmt(wr, 3):>7}  {_fmt(aw, 2):>7}  {_fmt(al, 2):>7}  "
            f"{_fmt(wl, 2):>5}  {_fmt(lda, 3):>7}  {_fmt(sda, 3):>7}  {_fmt(mfe, 2):>7}  {_fmt(mae, 2):>7}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def print_tier3_table(tier3: dict, cost_ticks: float = DEFAULT_COST_TICKS) -> str:
    """Print Tier 3 table and return as string."""
    lines = []
    lines.append("")
    lines.append("=" * 115)
    lines.append(f"TIER 3: Profit Translation (cost = {cost_ticks:.1f} ticks = ${cost_ticks * TICK_VALUE_USD:.2f} per trade)")
    lines.append("=" * 115)
    lines.append(
        f"{'Tier':>6}  {'N':>10}  {'RawPnL':>8}  {'CostAdj':>8}  {'CAWinR':>7}  "
        f"{'Expect':>8}  {'Sortino':>8}  {'PF':>6}  {'MaxDD':>9}  "
        f"{'AvgWin':>7}  {'AvgLoss':>7}  {'TotalPnL':>10}"
    )
    lines.append("-" * 115)

    for tier_name in CONFIDENCE_TIERS:
        d = tier3.get(tier_name, {})
        n = d.get("n", 0)
        raw = d.get("raw_mean_pnl", float("nan"))
        ca = d.get("cost_adj_mean_pnl", float("nan"))
        cawr = d.get("cost_adj_win_rate", float("nan"))
        exp = d.get("net_expectancy", float("nan"))
        sort = d.get("sortino", float("nan"))
        pf = d.get("profit_factor", float("nan"))
        mdd = d.get("max_drawdown_ticks", float("nan"))
        aw = d.get("avg_winner_ticks", float("nan"))
        al = d.get("avg_loser_ticks", float("nan"))
        total = d.get("total_pnl_usd", float("nan"))
        lines.append(
            f"{tier_name:>6}  {n:>10,}  {_fmt(raw, 3):>8}  {_fmt(ca, 3):>8}  {_fmt(cawr, 3):>7}  "
            f"{_fmt(exp, 4):>8}  {_fmt(sort, 2):>8}  {_fmt(pf, 2):>6}  {_fmt(mdd, 1):>9}  "
            f"{_fmt(aw, 2):>7}  {_fmt(al, 2):>7}  {'$'+_fmt(total,0):>10}"
        )

    text = "\n".join(lines)
    print(text)
    return text


def print_full_report(
    results: dict,
    cost_ticks: float = DEFAULT_COST_TICKS,
    title: str = "Model Evaluation Report",
) -> str:
    """Print all three tier tables. Returns full text."""
    lines = []
    lines.append("")
    lines.append("#" * 115)
    lines.append(f"  {title}")
    lines.append(f"  N = {results['tier1'].get('All', {}).get('n', '?'):,} predictions")
    lines.append("#" * 115)

    t1 = print_tier1_table(results["tier1"])
    t2 = print_tier2_table(results["tier2"])
    t3 = print_tier3_table(results["tier3"], cost_ticks)

    return "\n".join(lines) + t1 + t2 + t3


# ---- Formatting: Discord Markdown --------------------------------------------

def format_for_discord(
    results: dict,
    cost_ticks: float = DEFAULT_COST_TICKS,
    title: str = "Model Evaluation",
    horizon: str = "10s",
) -> str:
    """
    Produce a clean Discord-ready markdown summary.
    Concise enough to fit in a single Discord message (~2000 char limit).
    """
    t1 = results.get("tier1", {})
    t2 = results.get("tier2", {})
    t3 = results.get("tier3", {})

    n_total = t1.get("All", {}).get("n", 0)
    all_ic = t1.get("All", {}).get("ic", float("nan"))
    all_da = t1.get("All", {}).get("da", float("nan"))

    lines = [
        f"**{title}** (horizon={horizon}, cost={cost_ticks}t, N={n_total:,})",
        "",
        f"**Tier 1 Signal** | IC={_fmt(all_ic,3)} | DA={_fmt(all_da,3)}",
        "```",
        f"{'Tier':>5} | {'N':>8} | {'IC':>6} | {'DA':>5} | {'WR':>5} | {'Expect':>7} | {'Sortino':>7} | {'PF':>5} | {'MaxDD':>7}",
        "-" * 75,
    ]

    for tier_name in CONFIDENCE_TIERS:
        d1 = t1.get(tier_name, {})
        d2 = t2.get(tier_name, {})
        d3 = t3.get(tier_name, {})

        n = d1.get("n", 0)
        ic = d1.get("ic", float("nan"))
        da = d1.get("da", float("nan"))
        wr = d3.get("cost_adj_win_rate", float("nan"))
        exp = d3.get("net_expectancy", float("nan"))
        sort = d3.get("sortino", float("nan"))
        pf = d3.get("profit_factor", float("nan"))
        mdd = d3.get("max_drawdown_ticks", float("nan"))

        lines.append(
            f"{tier_name:>5} | {n:>8,} | {_fmt(ic,3):>6} | {_fmt(da,3):>5} | "
            f"{_fmt(wr,3):>5} | {_fmt(exp,4):>7} | {_fmt(sort,2):>7} | "
            f"{_fmt(pf,2):>5} | {_fmt(mdd,1):>7}"
        )

    lines.append("```")

    # Bottom line: best tier by expectancy
    best_tier = None
    best_exp = -float("inf")
    for tier_name in CONFIDENCE_TIERS:
        d3 = t3.get(tier_name, {})
        exp = d3.get("net_expectancy", float("-inf"))
        if isinstance(exp, (int, float)) and np.isfinite(exp) and exp > best_exp:
            best_exp = exp
            best_tier = tier_name

    if best_tier:
        d3_best = t3.get(best_tier, {})
        total_usd = d3_best.get("total_pnl_usd", 0)
        lines.append(
            f"Best tier: **{best_tier}** | "
            f"Expectancy={_fmt(best_exp, 4)}t/trade | "
            f"Total=${_fmt(total_usd, 0)}"
        )

    # Profitability verdict
    all_exp = t3.get("All", {}).get("net_expectancy", float("nan"))
    if isinstance(all_exp, (int, float)) and np.isfinite(all_exp):
        if all_exp > 0:
            lines.append("Verdict: PROFITABLE after costs at All tier")
        elif best_exp > 0:
            lines.append(f"Verdict: Profitable ONLY at {best_tier} tier (needs selectivity)")
        else:
            lines.append("Verdict: NOT PROFITABLE at any tier after costs")

    return "\n".join(lines)


def _fmt(val, decimals: int = 2) -> str:
    """Format a float, handling NaN/Inf gracefully."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if np.isnan(val):
            return "N/A"
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
    try:
        if decimals == 0:
            return f"{val:,.0f}"
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


# ---- Save Results ------------------------------------------------------------

def save_results(
    results: dict,
    output_path: str | Path,
) -> Path:
    """Save results to JSON, stripping non-serializable equity curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {
                k: _clean(v) for k, v in obj.items()
                if not k.startswith("_")
            }
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(_clean(results), f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    return output_path


# ---- CLI Entry Point ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tier 3 Profit Translation Evaluation for ES futures predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tier3_profit_eval.py --pred-file concat_oot_predictions.npz --horizon 10s
  python tier3_profit_eval.py --pred-dir results/event_mamba_cuda/ --horizon 10s --cost 2.5
  python tier3_profit_eval.py --pred-file predictions.npz --horizon 5s --save results.json
        """,
    )
    parser.add_argument("--pred-file", type=str, help="Path to a single .npz prediction file")
    parser.add_argument("--pred-dir", type=str, help="Path to directory of .npz prediction files")
    parser.add_argument("--horizon", type=str, default="10s", choices=["1s", "5s", "10s"],
                        help="Prediction horizon (default: 10s)")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST_TICKS,
                        help=f"Cost per trade in ticks (default: {DEFAULT_COST_TICKS})")
    parser.add_argument("--save", type=str, default=None, help="Path to save JSON results")
    parser.add_argument("--discord", action="store_true", help="Print Discord-formatted output")
    parser.add_argument("--tier", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Which tier(s) to compute (default: all)")
    parser.add_argument("--title", type=str, default=None, help="Title for the report")

    args = parser.parse_args()

    if not args.pred_file and not args.pred_dir:
        parser.error("Must specify --pred-file or --pred-dir")

    # Load predictions
    print(f"Loading predictions (horizon={args.horizon})...")
    if args.pred_file:
        predictions, labels = load_predictions(args.pred_file, args.horizon)
        source = Path(args.pred_file).name
    else:
        predictions, labels = load_predictions_dir(args.pred_dir, args.horizon)
        source = Path(args.pred_dir).name

    print(f"Loaded {len(predictions):,} prediction-label pairs from {source}")
    title = args.title or f"{source} Evaluation"

    # Compute requested tiers
    if args.tier == "all":
        results = evaluate_all_tiers(predictions, labels, args.cost)
        if args.discord:
            print(format_for_discord(results, args.cost, title, args.horizon))
        else:
            print_full_report(results, args.cost, title)
    elif args.tier == "1":
        results = {"tier1": compute_tier1(predictions, labels)}
        print_tier1_table(results["tier1"])
    elif args.tier == "2":
        results = {"tier2": compute_tier2(predictions, labels)}
        print_tier2_table(results["tier2"])
    elif args.tier == "3":
        results = {"tier3": compute_tier3(predictions, labels, args.cost)}
        print_tier3_table(results["tier3"], args.cost)

    # Save if requested
    if args.save:
        save_results(results, args.save)

    return results


if __name__ == "__main__":
    main()
