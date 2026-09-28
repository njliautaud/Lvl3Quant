#!/usr/bin/env python3
"""
Combined Model Fill Simulator with Adaptive TP/SL
===================================================
Combines predictions from CNN1D, Mamba, and LGBM into an ensemble,
then runs a fill simulation with confidence-adaptive take-profit
and stop-loss levels.

Key idea:
  - CNN/LGBM = direction specialists (high DA at confidence)
  - Mamba = magnitude specialist (high IC on big moves)
  - Combined = good direction + good magnitude = tradeable signal

Adaptive TP/SL:
  - Higher ensemble confidence → wider TP (let winners run), tighter SL (cut fast)
  - Lower confidence → skip trade entirely
  - Confidence = |ensemble_prediction| percentile

Usage:
    python alpha_discovery/combined_model_fill_sim.py
    python alpha_discovery/combined_model_fill_sim.py --cnn-dir results/cnn1d_5fold_embed --mamba-dir results/mamba_v4_d192_6L --lgbm-dir results/lgbm_book_features
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from scipy.stats import spearmanr

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results" / "combined_fill_sim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / f'fill_sim_{_ts}.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants (ES futures) ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.00   # round-trip commission + fees
SLIPPAGE_TICKS = 0.5   # conservative: half a tick each way


@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    pnl_ticks: float
    pnl_dollars: float
    hold_events: int
    exit_reason: str  # 'tp', 'sl', 'timeout', 'trailing'
    confidence: float
    tp_ticks: float
    sl_ticks: float


@dataclass
class SimConfig:
    """Adaptive TP/SL configuration."""
    # Confidence thresholds (percentile of |prediction|)
    min_confidence_pct: float = 75.0   # skip below this percentile

    # Base TP/SL in ticks (for median-confidence trades)
    base_tp_ticks: float = 8.0
    base_sl_ticks: float = 6.0

    # Adaptive scaling (multiplier at top confidence)
    # High confidence: TP widens (let winners run), SL can tighten
    tp_scale_at_top: float = 2.5   # top 1% conf → TP = base * 2.5 = 20 ticks
    sl_scale_at_top: float = 0.6   # top 1% conf → SL = base * 0.6 = 3.6 ticks

    # Timeout
    max_hold_events: int = 50000   # ~5 min of events (safety valve)

    # Trailing stop (activate after partial profit)
    trailing_stop: bool = True
    trailing_activate_ticks: float = 4.0  # activate after 4 tick profit
    trailing_distance_ticks: float = 3.0  # trail by 3 ticks

    # Cooldown between trades
    cooldown_events: int = 1000  # ~100ms bars → 100 seconds

    # Model weights for ensemble (will be optimized)
    cnn_weight: float = 0.35
    mamba_weight: float = 0.35
    lgbm_weight: float = 0.30


def load_fold_predictions(fold_dir: Path, model_type: str) -> Dict:
    """Load all fold predictions from a model directory.

    Returns dict with keys: predictions, labels, embeddings (if available)
    """
    fold_files = sorted(fold_dir.glob("fold_*_predictions.npz"))
    if not fold_files:
        # Try concat file
        concat = fold_dir / "concat_oot_predictions.npz"
        if concat.exists():
            d = np.load(concat)
            result = {'predictions': d['predictions'], 'labels': d['labels']}
            if 'embeddings' in d:
                result['embeddings'] = d['embeddings']
            if 'leaf_indices' in d:
                result['leaf_indices'] = d['leaf_indices']
            log.info(f"  {model_type}: loaded concat file, {len(result['predictions'])} samples")
            return result
        return {}

    all_preds, all_labels, all_embeds = [], [], []

    for fp in fold_files:
        d = np.load(fp)
        preds = d['predictions']
        labels = d['labels']

        # Handle multi-horizon predictions (CNN/Mamba: shape (N, 3) for 1s/5s/10s)
        if preds.ndim == 2 and preds.shape[1] == 3:
            # Use 10s horizon (index 2) as primary — best for trading
            preds_use = preds[:, 2]
            labels_use = labels[:, 2]
        else:
            preds_use = preds.flatten()
            labels_use = labels.flatten()

        all_preds.append(preds_use)
        all_labels.append(labels_use)

        if 'embeddings' in d:
            all_embeds.append(d['embeddings'])

        log.info(f"  {model_type}: {fp.name} → {len(preds_use)} samples")

    result = {
        'predictions': np.concatenate(all_preds),
        'labels': np.concatenate(all_labels),
    }
    if all_embeds:
        result['embeddings'] = np.concatenate(all_embeds)

    log.info(f"  {model_type} total: {len(result['predictions'])} samples")
    return result


def normalize_predictions(preds: np.ndarray) -> np.ndarray:
    """Z-score normalize predictions for combining across models."""
    valid = ~np.isnan(preds)
    if valid.sum() < 10:
        return preds
    mu = np.mean(preds[valid])
    sigma = np.std(preds[valid])
    if sigma < 1e-10:
        return np.zeros_like(preds)
    return (preds - mu) / sigma


def combine_predictions(
    cnn_preds: Optional[np.ndarray],
    mamba_preds: Optional[np.ndarray],
    lgbm_preds: Optional[np.ndarray],
    config: SimConfig,
) -> np.ndarray:
    """Weighted ensemble of normalized predictions.

    All inputs must be same length (aligned by event).
    Missing models get zero weight (renormalized).
    """
    n = 0
    available = []
    weights = []

    if cnn_preds is not None:
        n = len(cnn_preds)
        available.append(normalize_predictions(cnn_preds))
        weights.append(config.cnn_weight)
    if mamba_preds is not None:
        n = max(n, len(mamba_preds))
        available.append(normalize_predictions(mamba_preds))
        weights.append(config.mamba_weight)
    if lgbm_preds is not None:
        n = max(n, len(lgbm_preds))
        available.append(normalize_predictions(lgbm_preds))
        weights.append(config.lgbm_weight)

    if not available:
        raise ValueError("No predictions available!")

    # Renormalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    ensemble = np.zeros(n)
    for preds, w in zip(available, weights):
        ensemble[:len(preds)] += w * preds

    return ensemble


def compute_adaptive_tp_sl(
    confidence_pct: float,  # 0-100 percentile of this trade's confidence
    config: SimConfig,
) -> Tuple[float, float]:
    """Compute adaptive TP/SL based on confidence percentile.

    Higher confidence → wider TP (let winners run) + tighter SL (cut losses fast).
    The idea: when the model is very confident, it's worth giving the trade room
    to develop. When less confident, take quick profits.
    """
    # Linear interpolation from min_confidence (1.0x) to 100th pct (scale_at_top)
    conf_frac = (confidence_pct - config.min_confidence_pct) / (100.0 - config.min_confidence_pct)
    conf_frac = np.clip(conf_frac, 0, 1)

    tp_scale = 1.0 + conf_frac * (config.tp_scale_at_top - 1.0)
    sl_scale = 1.0 + conf_frac * (config.sl_scale_at_top - 1.0)  # sl_scale < 1 → tighter

    tp = config.base_tp_ticks * tp_scale
    sl = config.base_sl_ticks * sl_scale

    return tp, sl


def simulate_trades(
    ensemble_preds: np.ndarray,
    labels: np.ndarray,  # actual price changes in ticks
    config: SimConfig,
) -> List[TradeResult]:
    """Run fill simulation with adaptive TP/SL.

    Uses actual labels (price changes) to simulate realistic fills.
    Entry: when |ensemble_pred| > confidence threshold
    Direction: sign of prediction
    Exit: TP hit, SL hit, trailing stop, or timeout
    """
    n = len(ensemble_preds)
    abs_preds = np.abs(ensemble_preds)

    # Compute confidence threshold
    conf_threshold = np.percentile(abs_preds[~np.isnan(abs_preds)], config.min_confidence_pct)
    log.info(f"Confidence threshold ({config.min_confidence_pct}th pct): {conf_threshold:.4f}")

    trades = []
    i = 0
    last_exit = -config.cooldown_events

    while i < n - config.max_hold_events:
        # Skip if below confidence threshold or in cooldown
        if abs_preds[i] < conf_threshold or i - last_exit < config.cooldown_events:
            i += 1
            continue

        # Entry signal
        direction = 1 if ensemble_preds[i] > 0 else -1
        confidence_pct = (abs_preds[:i+1] < abs_preds[i]).mean() * 100  # expanding percentile

        # Adaptive TP/SL
        tp_ticks, sl_ticks = compute_adaptive_tp_sl(confidence_pct, config)

        # Simulate trade: walk forward through actual price changes
        entry_price = 0.0  # relative
        cum_pnl = 0.0
        max_favorable = 0.0
        trailing_active = False
        trailing_stop_price = None
        exit_reason = 'timeout'
        exit_idx = i + config.max_hold_events

        for j in range(i + 1, min(i + config.max_hold_events, n)):
            # Actual price move for this event
            if np.isnan(labels[j]):
                continue
            cum_pnl += direction * labels[j]

            # Track max favorable excursion
            if cum_pnl > max_favorable:
                max_favorable = cum_pnl

            # Check TP
            if cum_pnl >= tp_ticks:
                exit_reason = 'tp'
                exit_idx = j
                break

            # Check SL
            if cum_pnl <= -sl_ticks:
                exit_reason = 'sl'
                exit_idx = j
                break

            # Trailing stop
            if config.trailing_stop and cum_pnl >= config.trailing_activate_ticks:
                trailing_active = True
                if trailing_stop_price is None:
                    trailing_stop_price = cum_pnl - config.trailing_distance_ticks
                else:
                    trailing_stop_price = max(trailing_stop_price, cum_pnl - config.trailing_distance_ticks)

                if cum_pnl <= trailing_stop_price:
                    exit_reason = 'trailing'
                    exit_idx = j
                    break

        # Apply costs
        net_pnl_ticks = cum_pnl - SLIPPAGE_TICKS - (COMMISSION_RT / TICK_VALUE)
        net_pnl_dollars = net_pnl_ticks * TICK_VALUE

        trades.append(TradeResult(
            entry_idx=i,
            exit_idx=exit_idx,
            direction=direction,
            entry_price=0.0,
            exit_price=cum_pnl * TICK_SIZE,
            pnl_ticks=net_pnl_ticks,
            pnl_dollars=net_pnl_dollars,
            hold_events=exit_idx - i,
            exit_reason=exit_reason,
            confidence=confidence_pct,
            tp_ticks=tp_ticks,
            sl_ticks=sl_ticks,
        ))

        last_exit = exit_idx
        i = exit_idx + 1

    return trades


def compute_metrics(trades: List[TradeResult]) -> Dict:
    """Compute trading metrics from trade results."""
    if not trades:
        return {'error': 'no trades'}

    pnls = np.array([t.pnl_dollars for t in trades])
    pnl_ticks = np.array([t.pnl_ticks for t in trades])

    winners = pnls > 0
    losers = pnls < 0

    total_pnl = float(np.sum(pnls))
    n_trades = len(trades)
    win_rate = float(np.mean(winners)) if n_trades > 0 else 0

    avg_win = float(np.mean(pnls[winners])) if winners.any() else 0
    avg_loss = float(np.mean(pnls[losers])) if losers.any() else 0

    # Sortino ratio (annualized, assuming ~250 trading days, ~20 trades/day)
    daily_pnl = total_pnl / max(1, n_trades / 20)  # rough daily PnL
    downside_returns = pnls[pnls < 0]
    downside_std = float(np.std(downside_returns)) if len(downside_returns) > 1 else 1.0
    sortino = float(np.mean(pnls) / downside_std * np.sqrt(250 * 20)) if downside_std > 0 else 0

    # Sharpe
    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(250 * 20)) if np.std(pnls) > 0 else 0

    # Profit factor
    gross_profit = float(np.sum(pnls[winners])) if winners.any() else 0
    gross_loss = float(np.abs(np.sum(pnls[losers]))) if losers.any() else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Confidence tier analysis
    conf_tiers = {}
    for label, lo, hi in [('all', 0, 100), ('top50', 50, 100), ('top25', 75, 100), ('top10', 90, 100), ('top5', 95, 100)]:
        tier_trades = [t for t in trades if lo <= t.confidence < hi or (hi == 100 and t.confidence >= lo)]
        if not tier_trades:
            continue
        tier_pnls = np.array([t.pnl_dollars for t in tier_trades])
        tier_wins = tier_pnls > 0
        conf_tiers[label] = {
            'n_trades': len(tier_trades),
            'win_rate': float(np.mean(tier_wins)),
            'avg_pnl': float(np.mean(tier_pnls)),
            'total_pnl': float(np.sum(tier_pnls)),
            'avg_pnl_ticks': float(np.mean([t.pnl_ticks for t in tier_trades])),
        }

    return {
        'total_pnl': total_pnl,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sortino': sortino,
        'sharpe': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'avg_hold_events': float(np.mean([t.hold_events for t in trades])),
        'exit_reasons': exit_reasons,
        'confidence_tiers': conf_tiers,
    }


def sweep_configs(
    ensemble_preds: np.ndarray,
    labels: np.ndarray,
) -> List[Dict]:
    """Sweep TP/SL configs to find optimal adaptive parameters."""
    configs = []

    for base_tp in [6, 8, 10, 13, 16, 20]:
        for base_sl in [4, 6, 8, 10]:
            for min_conf in [70, 75, 80, 85, 90]:
                for tp_scale in [1.5, 2.0, 2.5]:
                    for sl_scale in [0.5, 0.7, 1.0]:
                        cfg = SimConfig(
                            min_confidence_pct=min_conf,
                            base_tp_ticks=float(base_tp),
                            base_sl_ticks=float(base_sl),
                            tp_scale_at_top=tp_scale,
                            sl_scale_at_top=sl_scale,
                        )
                        configs.append(cfg)

    return configs


def run_quick_sweep(ensemble_preds, labels):
    """Quick sweep of key parameters, report best configs."""
    log.info("\n" + "="*60)
    log.info("ADAPTIVE TP/SL SWEEP")
    log.info("="*60)

    best_sortino = -999
    best_config = None
    best_metrics = None
    all_results = []

    # Focused sweep (not exhaustive — ~180 configs)
    for base_tp in [8, 12, 16, 20]:
        for base_sl in [4, 6, 8]:
            for min_conf in [75, 80, 85, 90]:
                for trailing in [True, False]:
                    cfg = SimConfig(
                        min_confidence_pct=min_conf,
                        base_tp_ticks=float(base_tp),
                        base_sl_ticks=float(base_sl),
                        trailing_stop=trailing,
                    )

                    trades = simulate_trades(ensemble_preds, labels, cfg)
                    metrics = compute_metrics(trades)

                    if 'error' in metrics:
                        continue

                    result = {
                        'base_tp': base_tp,
                        'base_sl': base_sl,
                        'min_conf': min_conf,
                        'trailing': trailing,
                        'sortino': metrics['sortino'],
                        'sharpe': metrics['sharpe'],
                        'total_pnl': metrics['total_pnl'],
                        'n_trades': metrics['n_trades'],
                        'win_rate': metrics['win_rate'],
                        'profit_factor': metrics['profit_factor'],
                        'max_dd': metrics['max_drawdown'],
                    }
                    all_results.append(result)

                    if metrics['sortino'] > best_sortino and metrics['n_trades'] >= 20:
                        best_sortino = metrics['sortino']
                        best_config = cfg
                        best_metrics = metrics

    # Sort by Sortino
    all_results.sort(key=lambda x: x['sortino'], reverse=True)

    log.info(f"\nTop 10 configs by Sortino (min 20 trades):")
    log.info(f"{'TP':>4} {'SL':>4} {'Conf%':>5} {'Trail':>5} | {'Sortino':>8} {'Sharpe':>8} {'PnL':>10} {'Trades':>6} {'WR%':>6} {'PF':>6}")
    log.info("-" * 80)
    for r in all_results[:10]:
        if r['n_trades'] >= 20:
            log.info(f"{r['base_tp']:4.0f} {r['base_sl']:4.0f} {r['min_conf']:5.0f} {str(r['trailing']):>5} | "
                    f"{r['sortino']:8.2f} {r['sharpe']:8.2f} ${r['total_pnl']:9.0f} {r['n_trades']:6d} "
                    f"{r['win_rate']*100:5.1f}% {r['profit_factor']:5.2f}")

    return best_config, best_metrics, all_results


def main():
    parser = argparse.ArgumentParser(description="Combined Model Fill Simulator")
    parser.add_argument('--cnn-dir', type=str,
                       default=str(LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'cnn1d_5fold_embed'))
    parser.add_argument('--mamba-dir', type=str,
                       default=str(LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'mamba_v4_d192_6L'))
    parser.add_argument('--lgbm-dir', type=str,
                       default=str(LVL3_ROOT / 'alpha_discovery' / 'results' / 'lgbm_book_features'))
    parser.add_argument('--sweep', action='store_true', help='Run TP/SL parameter sweep')
    parser.add_argument('--cnn-weight', type=float, default=0.35)
    parser.add_argument('--mamba-weight', type=float, default=0.35)
    parser.add_argument('--lgbm-weight', type=float, default=0.30)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("COMBINED MODEL FILL SIMULATOR — Adaptive TP/SL")
    log.info("=" * 60)

    config = SimConfig(
        cnn_weight=args.cnn_weight,
        mamba_weight=args.mamba_weight,
        lgbm_weight=args.lgbm_weight,
    )

    # ── Load predictions from each model ──
    cnn_data, mamba_data, lgbm_data = None, None, None

    cnn_dir = Path(args.cnn_dir)
    if cnn_dir.exists():
        log.info(f"Loading CNN1D predictions from {cnn_dir}")
        cnn_data = load_fold_predictions(cnn_dir, "CNN1D")
    else:
        log.warning(f"CNN dir not found: {cnn_dir}")

    mamba_dir = Path(args.mamba_dir)
    if mamba_dir.exists():
        log.info(f"Loading Mamba predictions from {mamba_dir}")
        mamba_data = load_fold_predictions(mamba_dir, "Mamba")
    else:
        log.warning(f"Mamba dir not found: {mamba_dir}")

    lgbm_dir = Path(args.lgbm_dir)
    if lgbm_dir.exists():
        log.info(f"Loading LGBM predictions from {lgbm_dir}")
        lgbm_data = load_fold_predictions(lgbm_dir, "LGBM")
    else:
        log.warning(f"LGBM dir not found: {lgbm_dir}")

    # Count available models
    available = sum(1 for d in [cnn_data, mamba_data, lgbm_data] if d)
    log.info(f"\nModels available: {available}/3")

    if available == 0:
        log.error("No predictions found! Training runs may still be in progress.")
        log.info("Expected prediction dirs:")
        log.info(f"  CNN:   {args.cnn_dir}")
        log.info(f"  Mamba: {args.mamba_dir}")
        log.info(f"  LGBM:  {args.lgbm_dir}")
        return

    # ── Align predictions (use shortest available) ──
    lengths = []
    if cnn_data: lengths.append(('CNN', len(cnn_data['predictions'])))
    if mamba_data: lengths.append(('Mamba', len(mamba_data['predictions'])))
    if lgbm_data: lengths.append(('LGBM', len(lgbm_data['predictions'])))

    min_len = min(l for _, l in lengths)
    log.info(f"Prediction lengths: {lengths}")
    log.info(f"Aligning to shortest: {min_len}")

    cnn_preds = cnn_data['predictions'][:min_len] if cnn_data else None
    mamba_preds = mamba_data['predictions'][:min_len] if mamba_data else None
    lgbm_preds = lgbm_data['predictions'][:min_len] if lgbm_data else None

    # Use labels from whichever model has them
    labels = None
    for data in [cnn_data, mamba_data, lgbm_data]:
        if data and 'labels' in data:
            labels = data['labels'][:min_len]
            break

    if labels is None:
        log.error("No labels found in any prediction file!")
        return

    # ── Combine predictions ──
    log.info(f"\nEnsemble weights: CNN={config.cnn_weight}, Mamba={config.mamba_weight}, LGBM={config.lgbm_weight}")
    ensemble = combine_predictions(cnn_preds, mamba_preds, lgbm_preds, config)

    # ── Report individual and ensemble IC ──
    log.info("\n--- Prediction Quality ---")
    valid = ~np.isnan(labels)
    if cnn_preds is not None:
        ic = spearmanr(cnn_preds[valid], labels[valid]).correlation
        log.info(f"  CNN IC:      {ic:.4f}")
    if mamba_preds is not None:
        ic = spearmanr(mamba_preds[valid], labels[valid]).correlation
        log.info(f"  Mamba IC:    {ic:.4f}")
    if lgbm_preds is not None:
        ic = spearmanr(lgbm_preds[valid], labels[valid]).correlation
        log.info(f"  LGBM IC:     {ic:.4f}")
    ensemble_ic = spearmanr(ensemble[valid], labels[valid]).correlation
    log.info(f"  Ensemble IC: {ensemble_ic:.4f}")

    # DA
    if cnn_preds is not None:
        da = np.mean(np.sign(cnn_preds[valid]) == np.sign(labels[valid]))
        log.info(f"  CNN DA:      {da:.4f}")
    if mamba_preds is not None:
        da = np.mean(np.sign(mamba_preds[valid]) == np.sign(labels[valid]))
        log.info(f"  Mamba DA:    {da:.4f}")
    if lgbm_preds is not None:
        da = np.mean(np.sign(lgbm_preds[valid]) == np.sign(labels[valid]))
        log.info(f"  LGBM DA:     {da:.4f}")
    ensemble_da = np.mean(np.sign(ensemble[valid]) == np.sign(labels[valid]))
    log.info(f"  Ensemble DA: {ensemble_da:.4f}")

    # ── Run simulation ──
    if args.sweep:
        best_config, best_metrics, all_results = run_quick_sweep(ensemble, labels)
        if best_metrics:
            log.info(f"\n{'='*60}")
            log.info(f"BEST CONFIG: Sortino={best_metrics['sortino']:.2f}")
            log.info(f"  TP={best_config.base_tp_ticks}, SL={best_config.base_sl_ticks}, "
                    f"Conf>={best_config.min_confidence_pct}%, Trail={best_config.trailing_stop}")
            log.info(f"  PnL=${best_metrics['total_pnl']:,.0f}, Trades={best_metrics['n_trades']}, "
                    f"WR={best_metrics['win_rate']*100:.1f}%")

            # Save sweep results
            sweep_path = RESULTS_DIR / f'sweep_results_{_ts}.json'
            json.dump(all_results, open(sweep_path, 'w'), indent=2, default=str)
            log.info(f"Saved sweep results → {sweep_path}")
    else:
        log.info(f"\nRunning simulation with default config...")
        log.info(f"  TP={config.base_tp_ticks}, SL={config.base_sl_ticks}, "
                f"Conf>={config.min_confidence_pct}%, Trail={config.trailing_stop}")

        trades = simulate_trades(ensemble, labels, config)
        metrics = compute_metrics(trades)

        if 'error' in metrics:
            log.error(f"Simulation failed: {metrics['error']}")
            return

        log.info(f"\n{'='*60}")
        log.info(f"SIMULATION RESULTS")
        log.info(f"{'='*60}")
        log.info(f"  Total PnL:      ${metrics['total_pnl']:,.0f}")
        log.info(f"  Trades:          {metrics['n_trades']}")
        log.info(f"  Win Rate:        {metrics['win_rate']*100:.1f}%")
        log.info(f"  Avg Win:         ${metrics['avg_win']:,.0f}")
        log.info(f"  Avg Loss:        ${metrics['avg_loss']:,.0f}")
        log.info(f"  Sortino:         {metrics['sortino']:.2f}")
        log.info(f"  Sharpe:          {metrics['sharpe']:.2f}")
        log.info(f"  Profit Factor:   {metrics['profit_factor']:.2f}")
        log.info(f"  Max Drawdown:    ${metrics['max_drawdown']:,.0f}")
        log.info(f"  Avg Hold:        {metrics['avg_hold_events']:.0f} events")
        log.info(f"  Exit Reasons:    {metrics['exit_reasons']}")

        log.info(f"\n--- Confidence Tier Analysis ---")
        for tier, data in metrics.get('confidence_tiers', {}).items():
            log.info(f"  [{tier:6s}] trades={data['n_trades']:5d} WR={data['win_rate']*100:5.1f}% "
                    f"avg_pnl=${data['avg_pnl']:7.0f} total=${data['total_pnl']:10,.0f} "
                    f"avg_ticks={data['avg_pnl_ticks']:+.2f}")

        # Save results
        result_path = RESULTS_DIR / f'sim_results_{_ts}.json'
        json.dump(metrics, open(result_path, 'w'), indent=2, default=str)
        log.info(f"\nSaved results → {result_path}")

    # ── Save combined predictions + embeddings for meta-model ──
    save_dict = {
        'ensemble_predictions': ensemble,
        'labels': labels,
    }
    if cnn_data and 'embeddings' in cnn_data:
        save_dict['cnn_embeddings'] = cnn_data['embeddings'][:min_len]
    if mamba_data and 'embeddings' in mamba_data:
        save_dict['mamba_embeddings'] = mamba_data['embeddings'][:min_len]
    if lgbm_data and 'leaf_indices' in lgbm_data:
        save_dict['lgbm_leaf_indices'] = lgbm_data['leaf_indices'][:min_len]

    combined_path = RESULTS_DIR / f'combined_predictions_{_ts}.npz'
    np.savez_compressed(combined_path, **save_dict)
    log.info(f"Saved combined predictions + all embeddings → {combined_path}")


if __name__ == '__main__':
    main()
