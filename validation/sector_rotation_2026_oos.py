#!/usr/bin/env python3
"""
TRUE Out-of-Sample Lockbox Validation: Sector Rotation (AVO v24, score 6.50)
============================================================================
Tests the AVO-evolved sector rotation strategy on 2026 data (Jan 2 - Aug 22)
that was NEVER seen during evolution. The strategy evolved on 2022-2025
walk-forward folds.

Features are computed from yfinance to cover the full 2026 period (the feature
store only goes to 2026-06-05). Pre-2026 data is warmup only.

Runs twice: once with the strategy's MAX_CONCURRENT (1), and once forcing
MAX_CONCURRENT=1 (same in this case, but kept for template correctness).
"""

import sys
import os
import json
import importlib.util
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STRATEGY_PATH = '/home/jupiter/teleclaude-main/runs/sector_rotation_wf-20260824-014631/work/strategy.py'
RESULTS_PATH = '/home/jupiter/Lvl3Quant/validation/sector_rotation_lockbox_results.json'

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'

OOS_START = '2026-01-02'
OOS_END = '2026-08-22'
WARMUP_START = '2025-01-01'  # enough for 60d returns + indicators

CAPITAL = 10000.0

# ---------------------------------------------------------------------------
# Load strategy module
# ---------------------------------------------------------------------------
def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Build feature DataFrame matching what the strategy expects
# ---------------------------------------------------------------------------
def build_features(prices_df, spy_series):
    """
    Compute the features the strategy needs from raw prices:
      - ret_1d, ret_20d, ret_60d
      - rel_strength_spy  (etf ret_20d - spy ret_20d)
      - rs_rank_among_sectors (rank of rel_strength across sectors per day)
      - lead_lag_score_5d  (correlation of etf 5d returns with spy 5d returns, 20d rolling)
      - momentum_cross_20_60  (+1 if ret_20d > ret_60d and wasn't yesterday, -1 vice versa, 0 else)
    """
    spy_ret_1d = spy_series.pct_change()
    spy_ret_20d = spy_series.pct_change(20)
    spy_ret_5d = spy_series.pct_change(5)

    rows = []
    for etf in SECTOR_ETFS:
        if etf not in prices_df.columns:
            continue
        px = prices_df[etf].dropna()
        if len(px) < 61:
            continue

        ret_1d = px.pct_change()
        ret_5d = px.pct_change(5)
        ret_20d = px.pct_change(20)
        ret_60d = px.pct_change(60)

        # Relative strength vs SPY (20d return differential)
        rs = ret_20d - spy_ret_20d

        # Lead-lag score: rolling 20d correlation of etf 5d returns with spy 5d returns
        ll = ret_5d.rolling(20).corr(spy_ret_5d)

        # Momentum cross: +1 when 20d > 60d and wasn't before, -1 opposite
        ma_diff = ret_20d - ret_60d
        ma_diff_prev = ma_diff.shift(1)
        mom_cross = pd.Series(0.0, index=px.index)
        mom_cross[(ma_diff > 0) & (ma_diff_prev <= 0)] = 1.0
        mom_cross[(ma_diff < 0) & (ma_diff_prev >= 0)] = -1.0

        for date in px.index:
            if pd.isna(ret_1d.get(date)) or pd.isna(ret_20d.get(date)):
                continue
            rows.append({
                'date': date,
                'etf': etf,
                'ret_1d': ret_1d[date],
                'ret_20d': ret_20d[date],
                'ret_60d': ret_60d.get(date, np.nan),
                'rel_strength_spy': rs.get(date, np.nan),
                'lead_lag_score_5d': ll.get(date, np.nan),
                'momentum_cross_20_60': mom_cross.get(date, 0.0),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Compute rs_rank_among_sectors per day
    def rank_rs(group):
        group = group.copy()
        group['rs_rank_among_sectors'] = group['rel_strength_spy'].rank(ascending=False, method='min')
        return group

    df = df.groupby('date', group_keys=False).apply(rank_rs)
    return df


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
def run_simulation(strategy_mod, feature_data, prices, max_concurrent_override=None):
    """
    Run the strategy on 2026 OOS data. Train on pre-2026 data for fit().
    """
    capital = CAPITAL
    max_per_trade = getattr(strategy_mod, 'MAX_PER_TRADE', 2000.0)
    max_concurrent = max_concurrent_override or getattr(strategy_mod, 'MAX_CONCURRENT', 1)
    slippage_pct = getattr(strategy_mod, 'SLIPPAGE_PCT', 0.0001)

    # Train on 2025 data (warmup period)
    train_mask = feature_data['date'] < OOS_START
    train_data = feature_data[train_mask].copy()
    if len(train_data) == 0:
        print("WARNING: No training data available!")
        return [], pd.Series(dtype=float), pd.Series(dtype=float)

    print(f"  Training on {len(train_data)} rows ({train_data['date'].min().strftime('%Y-%m-%d')} to {train_data['date'].max().strftime('%Y-%m-%d')})")
    params = strategy_mod.fit(train_data)

    # Generate signals on ALL data (strategy filters by date internally)
    test_mask = (feature_data['date'] >= OOS_START) & (feature_data['date'] <= OOS_END)
    test_data = feature_data[test_mask].copy()
    print(f"  Test data: {len(test_data)} rows ({test_data['date'].min().strftime('%Y-%m-%d')} to {test_data['date'].max().strftime('%Y-%m-%d')})")

    signals = strategy_mod.generate_signals(test_data, params)
    print(f"  Signals generated: {len(signals)} entries")

    # Build signal lookup
    signal_lookup = {}
    if signals is not None and len(signals) > 0:
        for _, row in signals.iterrows():
            d = pd.Timestamp(row['date'])
            if d not in signal_lookup:
                signal_lookup[d] = []
            signal_lookup[d].append((row['etf'], row.get('score', 1.0)))

    # Get OOS trading dates
    oos_mask = (prices.index >= OOS_START) & (prices.index <= OOS_END)
    dates = prices.index[oos_mask]

    trades = []
    open_positions = []
    daily_pnl = pd.Series(0.0, index=dates)
    equity_curve = pd.Series(CAPITAL, index=dates)
    peak_equity = CAPITAL
    pending_entries = []

    for i, date in enumerate(dates):
        curr_equity = equity_curve.iloc[max(0, i - 1)] if i > 0 else CAPITAL
        portfolio_dd = (curr_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        # -- Process exits --
        new_open = []
        for pos in open_positions:
            ticker = pos['ticker']
            if ticker not in prices.columns:
                new_open.append(pos)
                continue
            curr_price = prices.loc[date, ticker]
            if pd.isna(curr_price):
                new_open.append(pos)
                continue

            # Update high water mark
            if curr_price > pos.get('high_water_mark', pos['entry_price_adj']):
                pos['high_water_mark'] = curr_price

            do_exit = strategy_mod.should_exit(pos, curr_price, date, portfolio_dd)

            if do_exit:
                exit_price = curr_price * (1.0 - slippage_pct)
                pnl = (exit_price - pos['entry_price_adj']) * pos['shares']
                capital += pos['size_dollars'] + pnl
                daily_pnl.iloc[i] += pnl

                entry_p = pos['entry_price_adj']
                pnl_pct = (curr_price - entry_p) / entry_p
                days_held = int(np.busday_count(
                    np.datetime64(pos['entry_date'], 'D'),
                    np.datetime64(date, 'D')))

                # Determine exit reason
                if pnl_pct >= strategy_mod.TAKE_PROFIT_PCT:
                    exit_reason = 'take_profit'
                elif days_held >= strategy_mod.MAX_HOLD_DAYS:
                    exit_reason = 'max_hold'
                elif days_held >= strategy_mod.UNDERWATER_CUT_DAYS and pnl_pct < strategy_mod.UNDERWATER_CUT_TOL:
                    exit_reason = 'underwater_cut'
                else:
                    exit_reason = 'trailing_stop'

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(pos['entry_date'])[:10],
                    'exit_date': str(date)[:10],
                    'entry_price': round(float(pos['entry_price_adj']), 4),
                    'exit_price': round(float(exit_price), 4),
                    'shares': pos['shares'],
                    'pnl': round(float(pnl), 2),
                    'return_pct': round(float(pnl / pos['size_dollars'] * 100), 4),
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })
            else:
                # Mark-to-market
                if i > 0:
                    prev_price = prices.loc[dates[i - 1], ticker]
                    if not pd.isna(prev_price):
                        daily_pnl.iloc[i] += (curr_price - prev_price) * pos['shares']
                new_open.append(pos)

        open_positions = new_open

        # -- Process pending entries (next-day execution) --
        for ticker, score in pending_entries:
            if len(open_positions) >= max_concurrent:
                break
            if any(p['ticker'] == ticker for p in open_positions):
                continue
            if ticker not in prices.columns:
                continue

            price = prices.loc[date, ticker]
            if pd.isna(price) or price <= 0:
                continue

            size = min(max_per_trade, capital * 0.95)
            if size < 10:
                continue

            entry_price = price * (1.0 + slippage_pct)
            shares = int(size / entry_price)
            if shares < 1:
                continue

            actual_cost = shares * entry_price
            capital -= actual_cost
            open_positions.append({
                'ticker': ticker,
                'entry_date': date,
                'entry_price_adj': entry_price,
                'high_water_mark': entry_price,
                'shares': shares,
                'size_dollars': actual_cost,
            })

        pending_entries = []

        # -- Check for new signals (execute NEXT day) --
        if date in signal_lookup and len(open_positions) < max_concurrent:
            candidates = sorted(signal_lookup[date], key=lambda x: x[1], reverse=True)
            for ticker, score in candidates:
                if not any(p['ticker'] == ticker for p in open_positions):
                    pending_entries.append((ticker, score))

        # -- Update equity curve --
        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] + daily_pnl.iloc[i]
        else:
            equity_curve.iloc[i] = CAPITAL + daily_pnl.iloc[i]
        peak_equity = max(peak_equity, equity_curve.iloc[i])

    # -- Force close remaining positions --
    last_date = dates[-1]
    for pos in open_positions:
        ticker = pos['ticker']
        if ticker not in prices.columns:
            continue
        price = prices.loc[last_date, ticker]
        if pd.isna(price):
            continue
        exit_price = price * (1.0 - slippage_pct)
        pnl = (exit_price - pos['entry_price_adj']) * pos['shares']
        days_held = int(np.busday_count(
            np.datetime64(pos['entry_date'], 'D'),
            np.datetime64(last_date, 'D')))
        trades.append({
            'ticker': ticker,
            'entry_date': str(pos['entry_date'])[:10],
            'exit_date': str(last_date)[:10],
            'entry_price': round(float(pos['entry_price_adj']), 4),
            'exit_price': round(float(exit_price), 4),
            'shares': pos['shares'],
            'pnl': round(float(pnl), 2),
            'return_pct': round(float(pnl / pos['size_dollars'] * 100), 4),
            'days_held': days_held,
            'exit_reason': 'force_close',
        })

    return trades, daily_pnl, equity_curve


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades, daily_pnl, equity_curve, label=""):
    n = len(trades)
    result = {
        'label': label,
        'total_trades': n,
    }

    if n == 0:
        result.update({
            'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'sortino': 0,
            'total_pnl': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0,
            'avg_days_held': 0, 'avg_return_pct': 0,
        })
        return result

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / n
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    ann = np.sqrt(252)
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    sharpe = float(mean_r / std_r * ann) if std_r > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    ds_std = downside.std() if len(downside) > 5 else std_r
    sortino = float(mean_r / ds_std * ann) if ds_std > 0 else 0.0

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    max_dd = float(dd.min()) * 100

    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - CAPITAL) / CAPITAL * 100

    avg_days = np.mean([t['days_held'] for t in trades])
    avg_ret = np.mean([t['return_pct'] for t in trades])

    result.update({
        'win_rate': round(win_rate, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_return, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'final_equity': round(float(final_equity), 2),
        'avg_days_held': round(float(avg_days), 1),
        'avg_return_pct': round(float(avg_ret), 4),
        'wins': len(wins),
        'losses': len(losses),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
    })

    return result


def regime_analysis(trades, daily_pnl, prices, vix_series):
    """Stratify performance by SPY regime (green/red/flat days) and VIX regime."""
    results = {}

    # SPY daily returns for regime classification
    spy = prices[BENCHMARK]
    spy_ret = spy.pct_change()

    # VIX regime classification
    vix_aligned = vix_series.reindex(daily_pnl.index).ffill().fillna(20.0)

    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    # By VIX regime
    for regime, label in [('low', 'VIX < 16'), ('mid', 'VIX 16-25'), ('high', 'VIX > 25')]:
        if regime == 'low':
            mask = vix_aligned < 16
        elif regime == 'mid':
            mask = (vix_aligned >= 16) & (vix_aligned <= 25)
        else:
            mask = vix_aligned > 25

        regime_rets = daily_ret[mask]
        n_days = len(regime_rets)
        if n_days < 5 or regime_rets.std() == 0:
            results[f'vix_{regime}'] = {'label': label, 'days': n_days, 'sharpe': 0.0}
        else:
            s = float(regime_rets.mean() / regime_rets.std() * np.sqrt(252))
            results[f'vix_{regime}'] = {'label': label, 'days': n_days, 'sharpe': round(s, 4)}

    # By SPY direction
    spy_daily = spy_ret.reindex(daily_pnl.index).fillna(0)
    for direction, label in [('green', 'SPY up days'), ('red', 'SPY down days'), ('flat', 'SPY flat')]:
        if direction == 'green':
            mask = spy_daily > 0.001
        elif direction == 'red':
            mask = spy_daily < -0.001
        else:
            mask = (spy_daily >= -0.001) & (spy_daily <= 0.001)

        dir_rets = daily_ret[mask]
        n_days = len(dir_rets)
        if n_days < 5 or dir_rets.std() == 0:
            results[f'spy_{direction}'] = {'label': label, 'days': n_days, 'sharpe': 0.0}
        else:
            s = float(dir_rets.mean() / dir_rets.std() * np.sqrt(252))
            results[f'spy_{direction}'] = {'label': label, 'days': n_days, 'sharpe': round(s, 4)}

    return results


def sector_breakdown(trades):
    """P&L and trade count by sector."""
    result = {}
    for etf in sorted(set(t['ticker'] for t in trades)):
        etf_trades = [t for t in trades if t['ticker'] == etf]
        etf_pnl = sum(t['pnl'] for t in etf_trades)
        etf_wins = len([t for t in etf_trades if t['pnl'] > 0])
        result[etf] = {
            'trades': len(etf_trades),
            'wins': etf_wins,
            'pnl': round(etf_pnl, 2),
            'win_rate': round(etf_wins / len(etf_trades), 3) if etf_trades else 0,
        }
    return result


def exit_breakdown(trades):
    """P&L by exit reason."""
    result = {}
    for reason in sorted(set(t['exit_reason'] for t in trades)):
        r_trades = [t for t in trades if t['exit_reason'] == reason]
        r_pnl = sum(t['pnl'] for t in r_trades)
        result[reason] = {
            'count': len(r_trades),
            'pnl': round(r_pnl, 2),
            'avg_return_pct': round(np.mean([t['return_pct'] for t in r_trades]), 4),
        }
    return result


def monthly_equity(equity_curve):
    """Monthly equity snapshots."""
    result = {}
    for month in sorted(set(str(d)[:7] for d in equity_curve.index)):
        month_mask = [str(d)[:7] == month for d in equity_curve.index]
        month_vals = equity_curve[month_mask]
        if len(month_vals) > 0:
            result[month] = round(float(month_vals.iloc[-1]), 2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TRUE OUT-OF-SAMPLE LOCKBOX: Sector Rotation (AVO v24, score 6.50)")
    print("=" * 70)
    print(f"OOS period: {OOS_START} to {OOS_END}")
    print(f"Capital: ${CAPITAL:,.0f}")
    print()

    # Load strategy
    strategy_mod = load_strategy()
    print(f"Strategy loaded: MAX_CONCURRENT={strategy_mod.MAX_CONCURRENT}, "
          f"SLIPPAGE={strategy_mod.SLIPPAGE_PCT}, MAX_HOLD={strategy_mod.MAX_HOLD_DAYS}d")

    # Download prices
    print("\nDownloading price data...")
    import yfinance as yf
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    raw = yf.download(tickers, start=WARMUP_START, end='2026-08-23',
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [c[0] if isinstance(c, tuple) else c for c in prices.columns]

    prices = prices.ffill().dropna(how='all')
    print(f"Prices: {prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}, {len(prices)} days")

    # VIX
    vix_cols = [c for c in prices.columns if 'VIX' in str(c).upper()]
    vix = prices[vix_cols[0]] if vix_cols else pd.Series(20.0, index=prices.index)
    spy = prices[BENCHMARK]

    # Build features
    print("\nBuilding features from price data...")
    feature_data = build_features(prices, spy)
    print(f"Feature rows: {len(feature_data)}, ETFs: {sorted(feature_data['etf'].unique())}")
    print(f"Feature date range: {feature_data['date'].min().strftime('%Y-%m-%d')} to {feature_data['date'].max().strftime('%Y-%m-%d')}")

    # Also try using the feature store for comparison (if it covers)
    # But we compute from yfinance for the full 2026 period

    all_results = {}

    # ---- Run 1: Strategy's default MAX_CONCURRENT ----
    print(f"\n{'='*70}")
    print(f"RUN 1: MAX_CONCURRENT = {strategy_mod.MAX_CONCURRENT} (strategy default)")
    print(f"{'='*70}")

    trades1, dpnl1, eq1 = run_simulation(strategy_mod, feature_data, prices)
    metrics1 = compute_metrics(trades1, dpnl1, eq1, f"MAX_CONCURRENT={strategy_mod.MAX_CONCURRENT}")

    print(f"\n  Total trades:    {metrics1['total_trades']}")
    print(f"  Win rate:        {metrics1['win_rate']:.1%}")
    print(f"  Profit factor:   {metrics1['profit_factor']:.2f}")
    print(f"  Sharpe:          {metrics1['sharpe']:.2f}")
    print(f"  Sortino:         {metrics1['sortino']:.2f}")
    print(f"  Total P&L:       ${metrics1['total_pnl']:,.2f}")
    print(f"  Total return:    {metrics1['total_return_pct']:.1f}%")
    print(f"  Max drawdown:    {metrics1['max_drawdown_pct']:.1f}%")
    print(f"  Avg days held:   {metrics1['avg_days_held']:.1f}")

    if trades1:
        regime1 = regime_analysis(trades1, dpnl1, prices, vix)
        sector1 = sector_breakdown(trades1)
        exit1 = exit_breakdown(trades1)
        monthly1 = monthly_equity(eq1)

        print("\n  VIX Regime Sharpes:")
        for k, v in regime1.items():
            if k.startswith('vix_'):
                print(f"    {v['label']}: Sharpe={v['sharpe']:.2f} ({v['days']} days)")

        print("\n  SPY Direction Sharpes:")
        for k, v in regime1.items():
            if k.startswith('spy_'):
                print(f"    {v['label']}: Sharpe={v['sharpe']:.2f} ({v['days']} days)")

        print("\n  By Sector:")
        for etf, info in sector1.items():
            print(f"    {etf}: {info['trades']} trades, {info['wins']}/{info['trades']} wins, P&L ${info['pnl']:.2f}")

        print("\n  By Exit Reason:")
        for reason, info in exit1.items():
            print(f"    {reason}: {info['count']} trades, P&L ${info['pnl']:.2f}, avg {info['avg_return_pct']:.2f}%")

        print("\n  Monthly Equity:")
        for month, eq in monthly1.items():
            print(f"    {month}: ${eq:,.2f}")

        all_results['default_concurrent'] = {
            'metrics': metrics1,
            'regime': regime1,
            'sector': sector1,
            'exit_reasons': exit1,
            'monthly_equity': monthly1,
            'trades': trades1,
        }
    else:
        print("\n  NO TRADES GENERATED!")
        all_results['default_concurrent'] = {'metrics': metrics1, 'trades': []}

    # ---- Run 2: MAX_CONCURRENT=1 (if different from default) ----
    if strategy_mod.MAX_CONCURRENT != 1:
        print(f"\n{'='*70}")
        print(f"RUN 2: MAX_CONCURRENT = 1 (concentrated)")
        print(f"{'='*70}")

        trades2, dpnl2, eq2 = run_simulation(strategy_mod, feature_data, prices, max_concurrent_override=1)
        metrics2 = compute_metrics(trades2, dpnl2, eq2, "MAX_CONCURRENT=1")

        print(f"\n  Total trades:    {metrics2['total_trades']}")
        print(f"  Win rate:        {metrics2['win_rate']:.1%}")
        print(f"  Profit factor:   {metrics2['profit_factor']:.2f}")
        print(f"  Sharpe:          {metrics2['sharpe']:.2f}")
        print(f"  Sortino:         {metrics2['sortino']:.2f}")
        print(f"  Total P&L:       ${metrics2['total_pnl']:,.2f}")
        print(f"  Total return:    {metrics2['total_return_pct']:.1f}%")
        print(f"  Max drawdown:    {metrics2['max_drawdown_pct']:.1f}%")

        if trades2:
            regime2 = regime_analysis(trades2, dpnl2, prices, vix)
            all_results['concentrated'] = {
                'metrics': metrics2,
                'regime': regime2,
                'trades': trades2,
            }
        else:
            all_results['concentrated'] = {'metrics': metrics2, 'trades': []}
    else:
        print(f"\n  (Strategy default is already MAX_CONCURRENT=1, skipping duplicate run)")

    # ---- Save results ----
    save_results = {
        'strategy': 'sector_rotation_wf',
        'avo_version': 'v24',
        'avo_score': 6.50,
        'oos_period': f'{OOS_START} to {OOS_END}',
        'capital': CAPITAL,
        'strategy_params': {
            'MAX_CONCURRENT': strategy_mod.MAX_CONCURRENT,
            'SLIPPAGE_PCT': strategy_mod.SLIPPAGE_PCT,
            'MAX_PER_TRADE': strategy_mod.MAX_PER_TRADE,
            'MAX_HOLD_DAYS': strategy_mod.MAX_HOLD_DAYS,
            'TRAILING_STOP_PCT': strategy_mod.TRAILING_STOP_PCT,
            'TAKE_PROFIT_PCT': strategy_mod.TAKE_PROFIT_PCT,
            'UNDERWATER_CUT_DAYS': strategy_mod.UNDERWATER_CUT_DAYS,
            'UNDERWATER_CUT_TOL': strategy_mod.UNDERWATER_CUT_TOL,
            'PORTFOLIO_DD_CUT': strategy_mod.PORTFOLIO_DD_CUT,
            'MIN_COMPOSITE_SCORE': strategy_mod.MIN_COMPOSITE_SCORE,
        },
        'timestamp': datetime.now().isoformat(),
    }

    for key in all_results:
        run_data = all_results[key]
        save_run = {k: v for k, v in run_data.items() if k != 'trades'}
        if 'trades' in run_data:
            save_run['trade_count'] = len(run_data['trades'])
            save_run['trades_detail'] = run_data['trades'][:200]  # cap for JSON size
        save_results[key] = save_run

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("LOCKBOX VALIDATION SUMMARY")
    print(f"{'='*70}")
    m = metrics1
    print(f"Strategy:        Sector Rotation (AVO v24, score 6.50)")
    print(f"OOS Period:      {OOS_START} to {OOS_END}")
    print(f"Trades:          {m['total_trades']}")
    print(f"Win Rate:        {m['win_rate']:.1%}")
    print(f"Profit Factor:   {m['profit_factor']:.2f}")
    print(f"Sharpe:          {m['sharpe']:.2f}")
    print(f"Sortino:         {m['sortino']:.2f}")
    print(f"Total Return:    {m['total_return_pct']:.1f}%")
    print(f"Max Drawdown:    {m['max_drawdown_pct']:.1f}%")

    verdict = "PASS" if m['sharpe'] > 0.5 and m['profit_factor'] > 1.0 and m['total_trades'] >= 10 else "FAIL"
    print(f"\nVERDICT: {verdict}")
    if verdict == "PASS":
        print("  Strategy shows positive risk-adjusted returns on truly unseen 2026 data.")
    else:
        reasons = []
        if m['sharpe'] <= 0.5:
            reasons.append(f"Sharpe {m['sharpe']:.2f} <= 0.5")
        if m['profit_factor'] <= 1.0:
            reasons.append(f"PF {m['profit_factor']:.2f} <= 1.0")
        if m['total_trades'] < 10:
            reasons.append(f"Only {m['total_trades']} trades (< 10)")
        print(f"  Reason: {'; '.join(reasons)}")


if __name__ == '__main__':
    main()
