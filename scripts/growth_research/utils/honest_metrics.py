#!/usr/bin/env python3
"""Honest Metrics — Correct Sharpe/Sortino calculation for compounding strategies.

BUG FOUND 2026-07-26: All $645 compounding strategies had inflated Sharpe because
monthly_return = monthly_pnl / initial_capital (fixed $645), instead of dividing
by current equity at the start of each month.

This module provides CORRECT metric calculations. Import and use instead of
computing Sharpe/Sortino inline.

Usage:
    from utils.honest_metrics import compute_honest_metrics

    metrics = compute_honest_metrics(
        trades,        # list of dicts with 'entry' date and 'pnl' dollar amount
        initial_cap,   # starting capital (e.g., 645.0)
        eq_curve=None  # optional equity curve for MaxDD
    )
"""
import numpy as np
import pandas as pd


def compute_honest_metrics(trades, initial_cap, eq_curve=None, name='strategy'):
    """Compute risk-adjusted metrics using honest equity-based returns.

    Args:
        trades: list of dicts, each with at least 'entry' (date str) and 'pnl' (float)
        initial_cap: starting capital ($)
        eq_curve: optional list/array of equity values over time
        name: strategy name for the result dict

    Returns:
        dict with honest metrics
    """
    if not trades:
        return {'name': name, 'n_trades': 0, 'verdict': 'NO TRADES'}

    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['date'].dt.to_period('M')

    # Build monthly equity and returns
    monthly_pnl = tdf.groupby('month')['pnl'].sum().sort_index()

    # Track equity at START of each month
    equity = initial_cap
    monthly_returns = []
    monthly_equities = []
    for m, pnl in monthly_pnl.items():
        monthly_equities.append(equity)
        ret = pnl / equity  # HONEST: divide by current equity, not initial
        monthly_returns.append(ret)
        equity += pnl

    mr = np.array(monthly_returns)
    n_years = max(len(mr) / 12, 0.5)

    # Sharpe (annualized)
    if len(mr) > 3 and mr.std() > 1e-10:
        sharpe = (mr.mean() / mr.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    # Sortino (annualized, downside deviation)
    downside = mr[mr < 0]
    if len(downside) > 1:
        sortino = (mr.mean() / downside.std()) * np.sqrt(12)
    else:
        sortino = float('inf') if mr.mean() > 0 else 0.0

    # CAGR
    final_equity = initial_cap + sum(t['pnl'] for t in trades)
    cagr = (final_equity / initial_cap) ** (1 / n_years) - 1

    # MaxDD from equity curve
    if eq_curve is not None:
        eq = np.array(eq_curve)
    else:
        # Build from trades
        eq = [initial_cap]
        for t in sorted(trades, key=lambda x: x['entry']):
            eq.append(eq[-1] + t['pnl'])
        eq = np.array(eq)

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-10)
    maxdd = float(dd.min())

    # Win rate, profit factor
    n = len(trades)
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    wr = wins / n * 100
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    pf = gross_profit / (gross_loss + 1e-10)

    # For comparison: compute the INFLATED Sharpe too
    mr_inflated = monthly_pnl.values / initial_cap
    if len(mr_inflated) > 3 and mr_inflated.std() > 1e-10:
        sharpe_inflated = (mr_inflated.mean() / mr_inflated.std()) * np.sqrt(12)
    else:
        sharpe_inflated = 0.0

    inflation_ratio = sharpe_inflated / sharpe if abs(sharpe) > 0.01 else 1.0

    return {
        'name': name,
        'n_trades': n,
        'sharpe_honest': round(sharpe, 2),
        'sharpe_inflated': round(sharpe_inflated, 2),
        'inflation_ratio': round(inflation_ratio, 2),
        'sortino_honest': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(maxdd * 100, 1),
        'win_rate_pct': round(wr, 1),
        'profit_factor': round(pf, 2),
        'final_equity': round(final_equity, 2),
        'n_months': len(mr),
        'n_years': round(n_years, 1),
    }


def compute_regime_split(trades, spy_sma_200, name='strategy'):
    """Split trades into bull/bear by SPY 200d SMA and compute metrics for each.

    Args:
        trades: list of dicts with 'entry', 'pnl', and optionally 'regime'
        spy_sma_200: pd.Series of SPY 200-day SMA values indexed by date
        name: strategy name

    Returns:
        dict with bull_metrics, bear_metrics, r1_gap
    """
    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['entry'])

    if 'regime' not in tdf.columns:
        # Classify by SPY vs 200d SMA
        for i, row in tdf.iterrows():
            d = row['date']
            nearest = spy_sma_200.index.asof(d)
            if pd.isna(nearest):
                tdf.at[i, 'regime'] = 'unknown'
            else:
                tdf.at[i, 'regime'] = 'bull' if spy_sma_200.loc[nearest] > 0 else 'bear'

    bull = tdf[tdf['regime'] == 'bull']
    bear = tdf[tdf['regime'] == 'bear']

    bull_wr = bull['pnl'].gt(0).mean() * 100 if len(bull) > 0 else 0
    bear_wr = bear['pnl'].gt(0).mean() * 100 if len(bear) > 0 else 0

    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1e-10)

    return {
        'bull_n': len(bull),
        'bear_n': len(bear),
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_gap < 0.50,
    }


def compute_yearly_sharpe(trades, initial_cap):
    """Compute per-year Sharpe to check consistency.

    Returns:
        dict with yearly_sharpe and consistency metrics
    """
    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['entry'])
    tdf['year'] = tdf['date'].dt.year

    yearly = {}
    for yr, grp in tdf.groupby('year'):
        pnls = grp['pnl'].values
        if len(pnls) < 4:
            continue
        # Use honest returns (but approximate — divide by initial for yearly)
        total = pnls.sum()
        mean_pnl = pnls.mean()
        std_pnl = pnls.std()
        yearly[str(yr)] = {
            'n_trades': len(pnls),
            'total_pnl': round(total, 2),
            'mean_pnl': round(mean_pnl, 2),
            'sharpe_approx': round(mean_pnl / (std_pnl + 1e-10) * np.sqrt(52), 2),  # ~weekly
            'win_rate': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
        }

    n_profitable = sum(1 for y in yearly.values() if y['total_pnl'] > 0)
    n_years = len(yearly)

    return {
        'yearly': yearly,
        'n_profitable_years': n_profitable,
        'n_total_years': n_years,
        'pct_profitable': round(n_profitable / max(n_years, 1) * 100, 1),
        'consistency_pass': n_profitable / max(n_years, 1) >= 0.60,
    }
