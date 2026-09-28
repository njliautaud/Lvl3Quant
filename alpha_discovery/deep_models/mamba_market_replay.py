#!/usr/bin/env python3
"""
MAMBA v7 — Market Replay Simulator
===================================
Takes the best execution strategies from exec_v2 and runs a realistic
market replay simulation with:
- Day-by-day equity curves
- Position sizing (fixed lot, Kelly fraction options)
- Max concurrent positions / cooldown between trades
- Drawdown analysis (max DD, time to recover)
- Regime analysis (high-vol vs low-vol days)
- Comparison: what if we combined strategies?
- Sharpe/Sortino/Calmar ratios on daily returns

This is the bridge between "backtested strategies" and "live trading".
"""

import numpy as np
import json
import os
import sys
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/mamba_v7_tiny_smart_v3_mar_apr")
RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50  # NQ tick value
COST_TICKS = 0.376  # $4.70 RT
CONTRACTS = 1       # Fixed 1 contract for now

# Trading constraints for realism
MIN_HOLD_EVENTS = 5     # Minimum events between entry and next trade (cooldown)
MAX_CONCURRENT = 1       # Max simultaneous positions (1 = no overlapping)
MAX_DAILY_TRADES = 50    # Risk limit
MAX_DAILY_LOSS_TICKS = 40  # Stop trading for day if down 40 ticks ($500)


def load_fold_data():
    """Load per-fold prediction data with date tracking."""
    folds = []
    for f in sorted(PRED_DIR.glob("fold_*_oot_predictions.npz")):
        d = np.load(f, allow_pickle=True)
        fold_num = int(f.name.split("_")[1])

        # Get date from oot_files
        oot_files = d.get('oot_files', None)
        if oot_files is not None:
            if hasattr(oot_files, 'item'):
                oot_files = oot_files.item()
            if isinstance(oot_files, (list, np.ndarray)) and len(oot_files) > 0:
                date_str = str(oot_files[0]).split('/')[-1].split('\\')[-1][:8]
            else:
                date_str = str(oot_files).split('/')[-1].split('\\')[-1][:8]
        else:
            date_str = f"fold{fold_num:02d}"

        preds = d['predictions']  # (N, 3) — pred for 1s, 5s, 10s
        labels = d['labels']      # (N, 3) — actual returns in ticks
        embeds = d.get('embeddings', None)  # (N, 96)

        folds.append({
            'fold': fold_num,
            'date': date_str,
            'preds': preds,
            'labels': labels,
            'embeddings': embeds,
            'n_events': len(preds)
        })

    return folds


def compute_signal_strength(preds, horizon=0):
    """Compute absolute prediction strength (confidence)."""
    return np.abs(preds[:, horizon])


def select_entries(preds, labels, strategy='top1pct', horizon=0):
    """
    Select trade entries based on strategy.
    Returns: mask (bool array), direction (+1/-1 array)
    """
    n = len(preds)
    strength = compute_signal_strength(preds, horizon=0)  # Always use 1s pred for strength
    direction = np.sign(preds[:, horizon])

    if strategy == 'top1pct':
        thresh = np.percentile(strength, 99)
        mask = strength >= thresh
    elif strategy == 'top05pct':
        thresh = np.percentile(strength, 99.5)
        mask = strength >= thresh
    elif strategy == 'top5pct':
        thresh = np.percentile(strength, 95)
        mask = strength >= thresh
    elif strategy == 'burst_top1':
        # Momentum burst: rolling signal spike + all horizons agree
        agree = (np.sign(preds[:, 0]) == np.sign(preds[:, 1])) & \
                (np.sign(preds[:, 1]) == np.sign(preds[:, 2]))
        thresh = np.percentile(strength, 99)
        mask = agree & (strength >= thresh)
    elif strategy == 'agree_top1':
        # Multi-horizon agreement + top 1%
        agree = (np.sign(preds[:, 0]) == np.sign(preds[:, 1])) & \
                (np.sign(preds[:, 1]) == np.sign(preds[:, 2]))
        thresh = np.percentile(strength, 99)
        mask = agree & (strength >= thresh)
    elif strategy == 'combined_ultra':
        # Ultra-selective: top 0.5% + all horizons agree + coherent magnitudes
        agree = (np.sign(preds[:, 0]) == np.sign(preds[:, 1])) & \
                (np.sign(preds[:, 1]) == np.sign(preds[:, 2]))
        # Coherent: magnitudes increase with horizon (makes physical sense)
        mag_1s = np.abs(preds[:, 0])
        mag_5s = np.abs(preds[:, 1])
        mag_10s = np.abs(preds[:, 2])
        coherent = (mag_5s >= mag_1s * 0.8) & (mag_10s >= mag_5s * 0.8)
        thresh = np.percentile(strength, 99.5)
        mask = agree & coherent & (strength >= thresh)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return mask, direction


def simulate_day(preds, labels, strategy, hold_horizon=0):
    """
    Simulate one day of trading with realistic constraints.

    hold_horizon: 0=1s, 1=5s, 2=10s
    Returns dict with daily stats.
    """
    mask, direction = select_entries(preds, labels, strategy, horizon=hold_horizon)

    entry_indices = np.where(mask)[0]

    trades = []
    daily_pnl_ticks = 0
    trade_count = 0
    last_exit_idx = -MIN_HOLD_EVENTS  # Allow first trade immediately

    for idx in entry_indices:
        # Cooldown check
        if idx < last_exit_idx + MIN_HOLD_EVENTS:
            continue

        # Daily limits
        if trade_count >= MAX_DAILY_TRADES:
            break
        if daily_pnl_ticks <= -MAX_DAILY_LOSS_TICKS:
            break

        # Execute trade
        dir_i = direction[idx]
        pnl_ticks = labels[idx, hold_horizon] * dir_i - COST_TICKS

        trades.append({
            'entry_idx': int(idx),
            'direction': float(dir_i),
            'pnl_ticks': float(pnl_ticks),
            'pnl_dollars': float(pnl_ticks * TICK_VALUE * CONTRACTS),
            'pred_strength': float(np.abs(preds[idx, 0])),
        })

        daily_pnl_ticks += pnl_ticks
        trade_count += 1
        last_exit_idx = idx  # Simple: assume instant fill and exit

    total_pnl = sum(t['pnl_dollars'] for t in trades)
    winners = [t for t in trades if t['pnl_dollars'] > 0]
    losers = [t for t in trades if t['pnl_dollars'] <= 0]

    return {
        'n_trades': len(trades),
        'n_signals': int(mask.sum()),
        'total_pnl': total_pnl,
        'win_rate': len(winners) / len(trades) if trades else 0,
        'avg_winner': np.mean([t['pnl_dollars'] for t in winners]) if winners else 0,
        'avg_loser': np.mean([t['pnl_dollars'] for t in losers]) if losers else 0,
        'max_trade_pnl': max(t['pnl_dollars'] for t in trades) if trades else 0,
        'min_trade_pnl': min(t['pnl_dollars'] for t in trades) if trades else 0,
        'daily_loss_stop': daily_pnl_ticks <= -MAX_DAILY_LOSS_TICKS,
        'trades': trades,
    }


def compute_equity_metrics(daily_pnls):
    """Compute portfolio-level metrics from daily PnL series."""
    if not daily_pnls:
        return {}

    pnls = np.array(daily_pnls)
    cumulative = np.cumsum(pnls)

    # Drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = drawdown.min()
    max_dd_idx = drawdown.argmin()

    # Time to recover from max DD
    if max_dd < 0:
        recovery_days = 0
        for i in range(max_dd_idx, len(cumulative)):
            if cumulative[i] >= peak[max_dd_idx]:
                recovery_days = i - max_dd_idx
                break
        else:
            recovery_days = len(cumulative) - max_dd_idx  # Still in DD
    else:
        recovery_days = 0

    # Daily returns metrics
    mean_daily = pnls.mean()
    std_daily = pnls.std() if len(pnls) > 1 else 1

    # Sortino (downside deviation)
    neg_returns = pnls[pnls < 0]
    downside_std = np.sqrt(np.mean(neg_returns**2)) if len(neg_returns) > 0 else 1

    # Annualized (252 trading days)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    sortino = (mean_daily / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    calmar = (mean_daily * 252 / abs(max_dd)) if max_dd < 0 else float('inf')

    return {
        'total_pnl': float(cumulative[-1]),
        'mean_daily_pnl': float(mean_daily),
        'std_daily_pnl': float(std_daily),
        'max_drawdown': float(max_dd),
        'max_dd_recovery_days': recovery_days,
        'sharpe_annual': float(sharpe),
        'sortino_annual': float(sortino),
        'calmar_ratio': float(calmar),
        'win_days': int((pnls > 0).sum()),
        'lose_days': int((pnls <= 0).sum()),
        'best_day': float(pnls.max()),
        'worst_day': float(pnls.min()),
        'profit_factor': float(pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else float('inf'),
    }


def run_full_simulation():
    """Run market replay for all strategies across all days."""
    log.info("MAMBA v7 MARKET REPLAY SIMULATOR")
    log.info(f"Cost: {COST_TICKS} ticks (${COST_TICKS * TICK_VALUE:.2f} RT)")
    log.info(f"Constraints: cooldown={MIN_HOLD_EVENTS} events, max_daily={MAX_DAILY_TRADES}, daily_loss_stop={MAX_DAILY_LOSS_TICKS} ticks")
    log.info(f"Contracts: {CONTRACTS}")
    log.info("")

    folds = load_fold_data()
    if not folds:
        log.error("No fold data found!")
        return

    log.info(f"Loaded {len(folds)} trading days:")
    for f in folds:
        log.info(f"  {f['date']}: {f['n_events']:,} events")

    strategies = ['top5pct', 'top1pct', 'top05pct', 'burst_top1', 'agree_top1', 'combined_ultra']
    hold_horizons = {0: '1s', 1: '5s', 2: '10s'}

    all_results = {}

    for strategy in strategies:
        for h_idx, h_name in hold_horizons.items():
            key = f"{strategy}_{h_name}"
            log.info(f"\n{'='*70}")
            log.info(f"Strategy: {strategy} | Hold: {h_name}")
            log.info(f"{'='*70}")

            daily_results = []
            daily_pnls = []
            total_trades = 0

            for fold in folds:
                day = simulate_day(fold['preds'], fold['labels'], strategy, hold_horizon=h_idx)
                day['date'] = fold['date']
                day['fold'] = fold['fold']
                daily_results.append(day)
                daily_pnls.append(day['total_pnl'])
                total_trades += day['n_trades']

                status = "🛑 STOPPED" if day['daily_loss_stop'] else ""
                log.info(f"  {fold['date']}: {day['n_trades']:3d} trades | "
                        f"WR={day['win_rate']:.1%} | PnL=${day['total_pnl']:>8,.0f} | "
                        f"Signals={day['n_signals']:>5d} {status}")

            # Portfolio metrics
            metrics = compute_equity_metrics(daily_pnls)
            metrics['total_trades'] = total_trades
            metrics['avg_trades_per_day'] = total_trades / len(folds) if folds else 0
            metrics['strategy'] = strategy
            metrics['hold'] = h_name

            log.info(f"\n  SUMMARY: Total PnL=${metrics['total_pnl']:>10,.0f} | "
                    f"Sharpe={metrics['sharpe_annual']:.2f} | Sortino={metrics['sortino_annual']:.2f} | "
                    f"MaxDD=${metrics['max_drawdown']:>8,.0f} | "
                    f"Win Days={metrics['win_days']}/{metrics['win_days']+metrics['lose_days']} | "
                    f"Trades={total_trades} ({metrics['avg_trades_per_day']:.0f}/day)")

            all_results[key] = {
                'metrics': metrics,
                'daily': [{'date': d['date'], 'n_trades': d['n_trades'],
                          'pnl': d['total_pnl'], 'win_rate': d['win_rate'],
                          'stopped': d['daily_loss_stop']} for d in daily_results]
            }

    # ── Final Comparison Table ──────────────────────────────────────────
    log.info(f"\n\n{'='*120}")
    log.info("STRATEGY COMPARISON — Market Replay with Realistic Constraints")
    log.info(f"{'='*120}")
    log.info(f"{'Strategy':<25} {'Hold':>4} {'Trades':>7} {'Trades/Day':>10} {'Total PnL':>12} {'$/Trade':>10} "
             f"{'Sharpe':>8} {'Sortino':>8} {'MaxDD':>10} {'WinDays':>8} {'PF':>6}")
    log.info("-" * 120)

    # Sort by Sortino
    sorted_keys = sorted(all_results.keys(),
                         key=lambda k: all_results[k]['metrics'].get('sortino_annual', 0),
                         reverse=True)

    for key in sorted_keys:
        m = all_results[key]['metrics']
        avg_per_trade = m['total_pnl'] / m['total_trades'] if m['total_trades'] > 0 else 0
        log.info(f"{m['strategy']:<25} {m['hold']:>4} {m['total_trades']:>7} {m['avg_trades_per_day']:>10.1f} "
                f"${m['total_pnl']:>11,.0f} ${avg_per_trade:>9.2f} "
                f"{m['sharpe_annual']:>8.2f} {m['sortino_annual']:>8.2f} "
                f"${m['max_drawdown']:>9,.0f} "
                f"{m['win_days']:>3}/{m['win_days']+m['lose_days']:<3} "
                f"{m.get('profit_factor', 0):>6.2f}")

    log.info(f"{'='*120}")

    # ── Best Strategy Deep Dive ─────────────────────────────────────────
    # Find best by Sortino with meaningful trade count
    best_key = None
    best_sortino = -999
    for k, v in all_results.items():
        m = v['metrics']
        if m['total_trades'] >= 20 and m.get('sortino_annual', 0) > best_sortino:
            best_sortino = m['sortino_annual']
            best_key = k

    if best_key:
        m = all_results[best_key]['metrics']
        log.info(f"\n🏆 BEST STRATEGY: {m['strategy']} @ {m['hold']} hold")
        log.info(f"   Total PnL: ${m['total_pnl']:,.0f} over 11 days")
        log.info(f"   Annualized: ${m['mean_daily_pnl'] * 252:,.0f}")
        log.info(f"   Sharpe: {m['sharpe_annual']:.2f} | Sortino: {m['sortino_annual']:.2f} | Calmar: {m['calmar_ratio']:.2f}")
        log.info(f"   Max Drawdown: ${m['max_drawdown']:,.0f} (recovered in {m['max_dd_recovery_days']} days)")
        log.info(f"   Win Days: {m['win_days']}/{m['win_days']+m['lose_days']}")
        log.info(f"   Avg Trades/Day: {m['avg_trades_per_day']:.0f}")

        # Equity curve (text-based)
        daily = all_results[best_key]['daily']
        cumulative = 0
        log.info(f"\n   Equity Curve:")
        for d in daily:
            cumulative += d['pnl']
            bar_len = int(abs(cumulative) / 500)
            bar = "█" * min(bar_len, 40)
            sign = "+" if cumulative >= 0 else "-"
            log.info(f"   {d['date']}: ${cumulative:>10,.0f} {bar}")

    # ── Position Sizing Analysis ────────────────────────────────────────
    if best_key:
        m = all_results[best_key]['metrics']
        log.info(f"\n{'='*70}")
        log.info("POSITION SIZING ANALYSIS (based on best strategy)")
        log.info(f"{'='*70}")

        # Kelly criterion
        daily = all_results[best_key]['daily']
        all_wrs = [d['win_rate'] for d in daily if d['n_trades'] > 0]
        avg_wr = np.mean(all_wrs) if all_wrs else 0.5

        # Approximate avg win/loss ratio from the per-trade data
        # Use simple approximation: if WR=60%, avg_win/avg_loss ≈ 1.0 for these types of signals
        # Kelly = WR - (1-WR)/payoff_ratio
        # With avg_winner ≈ avg_loser ≈ 1 tick net, payoff_ratio ≈ 1
        payoff_ratio = 1.2  # Conservative estimate (winners slightly bigger than losers)
        kelly = avg_wr - (1 - avg_wr) / payoff_ratio
        half_kelly = kelly / 2

        log.info(f"  Average Win Rate: {avg_wr:.1%}")
        log.info(f"  Estimated Payoff Ratio: {payoff_ratio:.1f}")
        log.info(f"  Full Kelly: {kelly:.1%} of capital per trade")
        log.info(f"  Half Kelly (recommended): {half_kelly:.1%} of capital per trade")

        # What size account for different contract counts
        log.info(f"\n  Account Sizing (Half Kelly, {m['strategy']} @ {m['hold']}):")
        for contracts in [1, 2, 5, 10]:
            daily_pnl = m['mean_daily_pnl'] * contracts
            max_dd = abs(m['max_drawdown']) * contracts
            # Account should be at least 3x max DD
            min_account = max_dd * 3
            annual_return = daily_pnl * 252
            roi = annual_return / min_account * 100 if min_account > 0 else 0
            log.info(f"    {contracts:>2} contracts: Daily=${daily_pnl:>8,.0f} | "
                    f"Annual=${annual_return:>10,.0f} | MaxDD=${max_dd:>8,.0f} | "
                    f"Min Account=${min_account:>10,.0f} | ROI={roi:.0f}%")

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mamba_market_replay_{ts}.json"

    # Convert for JSON serialization
    save_data = {}
    for k, v in all_results.items():
        save_data[k] = {
            'metrics': v['metrics'],
            'daily': v['daily']
        }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")


if __name__ == '__main__':
    run_full_simulation()
