#!/usr/bin/env python3
"""
Rules-Based Execution Optimizer — OOT Dates (Mar 16 → Apr 29)
===============================================================
Finds profitable rule combinations across ALL OOT dates using
CNN-Mamba v2 decay predictions with actual forward returns.

Tests:
  - Conviction thresholds (z-score gates)
  - Horizon selection (1s, 5s, 10s)
  - Direction filters (long-only, short-only, both)
  - Minimum interval between trades
  - Multi-horizon agreement (confluence)
  - Time-of-day effects (via timestamp analysis)

Metrics: Sharpe, Sortino, PF, WR, avg P&L per trade, per-date breakdown.
All costs at $4.70 RT = 0.376 ticks (HC #52).

Usage:
    python execution/rules_optimizer_oot.py --workers 14
    python execution/rules_optimizer_oot.py --top 20
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50 (HC #52)
LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
PRED_DIR = LVL3_ROOT / 'output' / 'decay_v4_comprehensive' / 'CNN-Mamba_v2'
MBO_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('rules_opt')

HORIZON_MAP = {0: 'labels_1s', 1: 'labels_5s', 2: 'labels_10s'}
HORIZON_NAMES = {0: '1s', 1: '5s', 2: '10s'}


@dataclass
class RuleConfig:
    config_id: int
    horizon: int            # 0=1s, 1=5s, 2=10s
    threshold_pct: float    # percentile threshold (e.g. 0.95 = top 5%)
    direction: str          # 'both', 'long', 'short'
    min_interval_idx: int   # min predictions between trades
    require_confluence: bool # require 2+ horizons agree on direction


@dataclass
class RuleResult:
    config_id: int
    horizon: str
    threshold_pct: float
    direction: str
    min_interval_idx: int
    require_confluence: bool
    # Aggregate
    n_dates: int
    n_trades: int
    trades_per_day: float
    win_rate: float
    avg_pnl_ticks: float
    total_pnl_ticks: float
    total_pnl_dollars: float
    profit_factor: float
    sharpe: float
    sortino: float
    # Robustness
    win_days: int
    loss_days: int
    daily_sharpe: float
    daily_sortino: float
    max_drawdown_dollars: float
    avg_daily_pnl: float
    # Distribution
    avg_win_ticks: float
    avg_loss_ticks: float
    rr_ratio: float
    best_day_dollars: float
    worst_day_dollars: float
    pct_profitable_days: float


def load_all_dates() -> Dict[str, dict]:
    """Load all decay predictions with labels."""
    data = {}
    for date_dir in sorted(PRED_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        pred_file = date_dir / 'predictions.npz'
        if not pred_file.exists():
            continue
        date_str = date_dir.name
        d = np.load(pred_file, allow_pickle=True)
        preds = d['preds']  # (N, 3) — predictions for 1s, 5s, 10s
        labels_1s = d['labels_1s']
        labels_5s = d['labels_5s']
        labels_10s = d['labels_10s']

        data[date_str] = {
            'preds': preds,
            'labels': np.stack([labels_1s, labels_5s, labels_10s], axis=1),
            'n': len(preds),
        }
    return data


def evaluate_config(args) -> Optional[RuleResult]:
    """Evaluate one rule configuration across all dates."""
    config, all_data = args

    all_trade_pnls = []  # individual trade PnLs in ticks
    daily_pnls = []      # per-day total PnL in dollars
    n_dates = 0

    for date_str, dd in sorted(all_data.items()):
        preds = dd['preds']
        labels = dd['labels']
        n = dd['n']

        if n < 50:
            continue
        n_dates += 1

        # Get predictions and labels for this horizon
        h = config.horizon
        pred_h = preds[:, h]  # predictions for chosen horizon
        label_h = labels[:, h]  # actual forward returns (ticks)

        # Compute z-scores for thresholding
        mu = np.mean(pred_h)
        sigma = np.std(pred_h)
        if sigma < 1e-8:
            daily_pnls.append(0.0)
            continue
        z = (pred_h - mu) / sigma

        # Find absolute threshold for this percentile
        abs_z = np.abs(z)
        thresh = np.percentile(abs_z, config.threshold_pct * 100)

        # Confluence check: if required, check that other horizons agree
        if config.require_confluence:
            # At least 2 of 3 horizons must agree on direction
            signs = np.sign(preds)
            agreement = np.sum(signs == np.sign(preds[:, h:h+1]), axis=1)
            confluence_mask = agreement >= 2
        else:
            confluence_mask = np.ones(n, dtype=bool)

        # Generate signals
        day_trades = []
        last_trade_idx = -config.min_interval_idx

        for i in range(n):
            if i - last_trade_idx < config.min_interval_idx:
                continue
            if abs_z[i] < thresh:
                continue
            if not confluence_mask[i]:
                continue

            # Direction filter
            direction = 1 if z[i] > 0 else -1
            if config.direction == 'long' and direction < 0:
                continue
            if config.direction == 'short' and direction > 0:
                continue

            # Trade PnL = direction * actual_return - commission
            trade_pnl_ticks = direction * label_h[i] - COMMISSION_RT_TICKS
            day_trades.append(trade_pnl_ticks)
            last_trade_idx = i

        all_trade_pnls.extend(day_trades)
        day_total = sum(day_trades) * TICK_VALUE
        daily_pnls.append(day_total)

    if not all_trade_pnls or n_dates == 0:
        return None

    # Compute metrics
    trades = np.array(all_trade_pnls)
    n_trades = len(trades)
    wins = trades[trades > 0]
    losses = trades[trades <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_pnl = np.mean(trades)
    total_pnl = np.sum(trades)
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-8
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Sharpe/Sortino on per-trade basis
    if np.std(trades) > 0:
        sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252 * max(n_trades / n_dates, 1))
    else:
        sharpe = 0
    downside = trades[trades < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.mean(trades) / np.std(downside) * np.sqrt(252 * max(n_trades / n_dates, 1))
    else:
        sortino = sharpe

    # Daily metrics
    daily = np.array(daily_pnls)
    win_days = int(np.sum(daily > 0))
    loss_days = int(np.sum(daily < 0))
    avg_daily = np.mean(daily)

    if np.std(daily) > 0:
        daily_sharpe = np.mean(daily) / np.std(daily) * np.sqrt(252)
    else:
        daily_sharpe = 0
    daily_down = daily[daily < 0]
    if len(daily_down) > 0 and np.std(daily_down) > 0:
        daily_sortino = np.mean(daily) / np.std(daily_down) * np.sqrt(252)
    else:
        daily_sortino = daily_sharpe

    # Max drawdown
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = np.max(dd) if len(dd) > 0 else 0

    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 1e-8
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    return RuleResult(
        config_id=config.config_id,
        horizon=HORIZON_NAMES[config.horizon],
        threshold_pct=config.threshold_pct,
        direction=config.direction,
        min_interval_idx=config.min_interval_idx,
        require_confluence=config.require_confluence,
        n_dates=n_dates,
        n_trades=n_trades,
        trades_per_day=n_trades / n_dates,
        win_rate=win_rate,
        avg_pnl_ticks=avg_pnl,
        total_pnl_ticks=total_pnl,
        total_pnl_dollars=total_pnl * TICK_VALUE,
        profit_factor=pf,
        sharpe=sharpe,
        sortino=sortino,
        win_days=win_days,
        loss_days=loss_days,
        daily_sharpe=daily_sharpe,
        daily_sortino=daily_sortino,
        max_drawdown_dollars=max_dd,
        avg_daily_pnl=avg_daily,
        avg_win_ticks=avg_win,
        avg_loss_ticks=float(np.mean(np.abs(losses))) if len(losses) > 0 else 0,
        rr_ratio=rr,
        best_day_dollars=float(np.max(daily)) if len(daily) > 0 else 0,
        worst_day_dollars=float(np.min(daily)) if len(daily) > 0 else 0,
        pct_profitable_days=win_days / n_dates * 100 if n_dates > 0 else 0,
    )


def build_configs() -> List[RuleConfig]:
    """Build comprehensive rule configurations."""
    configs = []
    cid = 0

    horizons = [0, 1, 2]  # 1s, 5s, 10s
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    directions = ['both', 'long', 'short']
    intervals = [1, 5, 10, 20, 50]  # min predictions between trades
    confluences = [False, True]

    for h, thresh, dirn, interval, conf in itertools.product(
        horizons, thresholds, directions, intervals, confluences
    ):
        configs.append(RuleConfig(
            config_id=cid,
            horizon=h,
            threshold_pct=thresh,
            direction=dirn,
            min_interval_idx=interval,
            require_confluence=conf,
        ))
        cid += 1

    return configs


def main():
    parser = argparse.ArgumentParser(description="Rules-Based Execution Optimizer (OOT)")
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--top', type=int, default=30, help="Show top N results")
    parser.add_argument('--min-trades', type=int, default=50, help="Min trades to include")
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    log.info("Loading decay predictions for all OOT dates...")
    all_data = load_all_dates()
    log.info(f"  Loaded {len(all_data)} dates: {sorted(all_data.keys())[0]}..{sorted(all_data.keys())[-1]}")

    total_preds = sum(d['n'] for d in all_data.values())
    log.info(f"  Total predictions: {total_preds:,}")

    configs = build_configs()
    log.info(f"  Testing {len(configs)} rule configurations with {args.workers} workers...")

    # Run evaluation
    results = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_config, (c, all_data)): c for c in configs}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - start
                log.info(f"  Progress: {done}/{len(configs)} ({elapsed:.0f}s)")
            result = future.result()
            if result is not None and result.n_trades >= args.min_trades:
                results.append(result)

    elapsed = time.time() - start
    log.info(f"  Completed {len(configs)} configs in {elapsed:.1f}s → {len(results)} valid results")

    if not results:
        log.error("No valid results! All configs had fewer than --min-trades trades.")
        return

    # Sort by Sortino (risk-adjusted return)
    results.sort(key=lambda r: r.daily_sortino, reverse=True)

    # Print top results
    log.info(f"\n{'='*120}")
    log.info(f"TOP {args.top} RULE COMBINATIONS BY DAILY SORTINO (across {len(all_data)} OOT dates, Mar 16 → Apr 29)")
    log.info(f"{'='*120}")
    log.info(f"{'#':>3} {'Hz':>3} {'Thresh':>7} {'Dir':>6} {'Intv':>4} {'Conf':>4} | "
             f"{'Trades':>6} {'T/Day':>5} {'WR%':>5} {'PF':>5} {'R:R':>4} | "
             f"{'Sharpe':>7} {'Sortino':>7} {'D.Srtn':>7} | "
             f"{'PnL$':>9} {'AvgD$':>7} {'MaxDD$':>8} {'W/L':>5} {'%Days+':>6}")

    for i, r in enumerate(results[:args.top]):
        log.info(
            f"{i+1:>3} {r.horizon:>3} {r.threshold_pct:>7.2f} {r.direction:>6} {r.min_interval_idx:>4} "
            f"{'Y' if r.require_confluence else 'N':>4} | "
            f"{r.n_trades:>6} {r.trades_per_day:>5.1f} {r.win_rate*100:>5.1f} {r.profit_factor:>5.2f} {r.rr_ratio:>4.2f} | "
            f"{r.sharpe:>7.2f} {r.sortino:>7.2f} {r.daily_sortino:>7.2f} | "
            f"{r.total_pnl_dollars:>9.0f} {r.avg_daily_pnl:>7.0f} {r.max_drawdown_dollars:>8.0f} "
            f"{r.win_days}/{r.loss_days:>2} {r.pct_profitable_days:>5.1f}%"
        )

    # Also show top by Sharpe
    by_sharpe = sorted(results, key=lambda r: r.daily_sharpe, reverse=True)
    log.info(f"\n{'='*120}")
    log.info(f"TOP 10 BY DAILY SHARPE")
    log.info(f"{'='*120}")
    for i, r in enumerate(by_sharpe[:10]):
        log.info(
            f"{i+1:>3} {r.horizon:>3} {r.threshold_pct:>7.2f} {r.direction:>6} {r.min_interval_idx:>4} "
            f"{'Y' if r.require_confluence else 'N':>4} | "
            f"{r.n_trades:>6} {r.trades_per_day:>5.1f} {r.win_rate*100:>5.1f} {r.profit_factor:>5.2f} | "
            f"{r.daily_sharpe:>7.2f} {r.daily_sortino:>7.2f} | "
            f"${r.total_pnl_dollars:>8.0f} {r.pct_profitable_days:>5.1f}%"
        )

    # Top by profit factor
    by_pf = sorted(results, key=lambda r: r.profit_factor, reverse=True)
    log.info(f"\n{'='*120}")
    log.info(f"TOP 10 BY PROFIT FACTOR (min 100 trades)")
    log.info(f"{'='*120}")
    for i, r in enumerate([x for x in by_pf if x.n_trades >= 100][:10]):
        log.info(
            f"{i+1:>3} {r.horizon:>3} {r.threshold_pct:>7.2f} {r.direction:>6} {r.min_interval_idx:>4} "
            f"{'Y' if r.require_confluence else 'N':>4} | "
            f"{r.n_trades:>6} PF={r.profit_factor:>5.2f} WR={r.win_rate*100:.1f}% | "
            f"${r.total_pnl_dollars:>8.0f} Sortino={r.daily_sortino:.2f}"
        )

    # Save all results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = args.output or str(LVL3_ROOT / 'execution' / 'results' / f'rules_opt_oot_{ts}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    output = {
        'timestamp': ts,
        'n_dates': len(all_data),
        'date_range': f"{sorted(all_data.keys())[0]}..{sorted(all_data.keys())[-1]}",
        'n_configs_tested': len(configs),
        'n_valid_results': len(results),
        'cost_ticks_rt': COMMISSION_RT_TICKS,
        'results': [asdict(r) for r in results],
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    log.info(f"\nResults saved to {out_path}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    profitable = [r for r in results if r.total_pnl_dollars > 0]
    log.info(f"  Profitable configs: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.1f}%)")
    if profitable:
        best = max(profitable, key=lambda r: r.daily_sortino)
        log.info(f"  Best config: horizon={best.horizon}, thresh={best.threshold_pct:.2f}, "
                 f"dir={best.direction}, interval={best.min_interval_idx}, confluence={'Y' if best.require_confluence else 'N'}")
        log.info(f"    → {best.n_trades} trades, WR={best.win_rate*100:.1f}%, PF={best.profit_factor:.2f}, "
                 f"Sortino={best.daily_sortino:.2f}, PnL=${best.total_pnl_dollars:.0f}")
        log.info(f"    → {best.pct_profitable_days:.0f}% profitable days, MaxDD=${best.max_drawdown_dollars:.0f}")


if __name__ == '__main__':
    main()
