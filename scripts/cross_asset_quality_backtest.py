#!/usr/bin/env python3
"""
Cross-Asset Signals for Quality Stock Timing Backtest

Uses bond/gold/dollar/credit signals to FILTER quality stock mean-reversion entries.
When credit markets and safe havens confirm a stock dip is just noise (not systemic risk),
enter with more confidence.

6 Variants (all start with QMR base: stock drops >5% from 20d high AND RSI<35):
  A: QMR baseline (no cross-asset filter) — CONTROL
  B: QMR + HYG not in downtrend (HYG > 10d SMA)
  C: QMR + TLT 5d change < 2% (bonds not surging)
  D: QMR + GLD 5d change < 1% (gold not surging)
  E: QMR + HYG/IEF ratio rising (credit spread tightening)
  F: Combined — QMR + at least 2 of 3: HYG>10SMA, TLT not surging, GLD not surging
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
QUALITY_STOCKS = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
CROSS_ASSETS = ['TLT', 'GLD', 'UUP', 'HYG', 'IEF']
ALL_TICKERS = QUALITY_STOCKS + CROSS_ASSETS

OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
FETCH_START = '2021-06-01'  # extra lookback for indicators

STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
HOLD_DAYS = 10

# QMR base parameters
DRAWDOWN_PCT = 0.05  # 5% drop from 20-day high
RSI_THRESHOLD = 35
RSI_PERIOD = 14
HIGH_LOOKBACK = 20

# 5-Gate thresholds
SHARPE_MIN = 0.5
PERM_P_MAX = 0.05
GAP_MAX = 0.5
MDD_MIN = -0.50  # max drawdown > -50%
MIN_TRADES = 20
N_PERMUTATIONS = 2000


def fetch_data():
    """Download all price data."""
    print(f"Fetching data for {len(ALL_TICKERS)} tickers...")
    data = yf.download(ALL_TICKERS, start=FETCH_START, end=OOT_END, progress=False, auto_adjust=True)
    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    # Handle multi-level columns from yfinance
    if isinstance(close.columns, pd.MultiIndex):
        close = data['Close']
    close = close.ffill().dropna(how='all')
    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")
    return close


def compute_rsi(series, period=RSI_PERIOD):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(close):
    """Compute all needed indicators."""
    indicators = {}

    # Stock indicators
    for sym in QUALITY_STOCKS:
        if sym not in close.columns:
            continue
        px = close[sym]
        indicators[f'{sym}_rsi'] = compute_rsi(px)
        indicators[f'{sym}_high20'] = px.rolling(HIGH_LOOKBACK).max()
        indicators[f'{sym}_drawdown'] = (px / indicators[f'{sym}_high20']) - 1

    # Cross-asset indicators
    if 'HYG' in close.columns:
        indicators['HYG_sma10'] = close['HYG'].rolling(10).mean()
        indicators['HYG_above_sma'] = close['HYG'] > indicators['HYG_sma10']

    if 'TLT' in close.columns:
        indicators['TLT_5d_chg'] = close['TLT'].pct_change(5) * 100

    if 'GLD' in close.columns:
        indicators['GLD_5d_chg'] = close['GLD'].pct_change(5) * 100

    if 'HYG' in close.columns and 'IEF' in close.columns:
        ratio = close['HYG'] / close['IEF']
        indicators['HYG_IEF_ratio'] = ratio
        indicators['HYG_IEF_rising'] = ratio > ratio.shift(5)

    return pd.DataFrame(indicators, index=close.index)


def qmr_base_signal(indicators, sym, date):
    """Check QMR base condition: stock drops >5% from 20d high AND RSI<35."""
    dd_key = f'{sym}_drawdown'
    rsi_key = f'{sym}_rsi'
    if dd_key not in indicators.columns or rsi_key not in indicators.columns:
        return False
    try:
        dd = indicators.loc[date, dd_key]
        rsi = indicators.loc[date, rsi_key]
        if pd.isna(dd) or pd.isna(rsi):
            return False
        return dd <= -DRAWDOWN_PCT and rsi < RSI_THRESHOLD
    except (KeyError, TypeError):
        return False


def cross_asset_filter(variant, indicators, date):
    """Apply cross-asset filter based on variant."""
    if variant == 'A':
        return True  # No filter

    try:
        if variant == 'B':
            # HYG not in downtrend
            return bool(indicators.loc[date, 'HYG_above_sma'])

        elif variant == 'C':
            # TLT 5d change < 2%
            tlt_chg = indicators.loc[date, 'TLT_5d_chg']
            return not pd.isna(tlt_chg) and tlt_chg < 2.0

        elif variant == 'D':
            # GLD 5d change < 1%
            gld_chg = indicators.loc[date, 'GLD_5d_chg']
            return not pd.isna(gld_chg) and gld_chg < 1.0

        elif variant == 'E':
            # HYG/IEF ratio rising
            return bool(indicators.loc[date, 'HYG_IEF_rising'])

        elif variant == 'F':
            # At least 2 of 3: HYG>10SMA, TLT not surging, GLD not surging
            checks = []
            checks.append(bool(indicators.loc[date, 'HYG_above_sma']))

            tlt_chg = indicators.loc[date, 'TLT_5d_chg']
            checks.append(not pd.isna(tlt_chg) and tlt_chg < 2.0)

            gld_chg = indicators.loc[date, 'GLD_5d_chg']
            checks.append(not pd.isna(gld_chg) and gld_chg < 1.0)

            return sum(checks) >= 2

    except (KeyError, TypeError):
        return False

    return False


def run_backtest(variant, close, indicators, oot_dates):
    """Run backtest for a single variant."""
    capital = STARTING_CAPITAL
    positions = []  # list of {sym, entry_date, entry_price, shares, exit_idx}
    trades = []
    equity_curve = [STARTING_CAPITAL]
    dates_curve = [oot_dates[0]]

    for i, date in enumerate(oot_dates):
        # Check exits
        new_positions = []
        for pos in positions:
            days_held = (date - pos['entry_date']).days
            if days_held >= HOLD_DAYS:
                # Exit
                if pos['sym'] in close.columns:
                    try:
                        exit_price = close.loc[date, pos['sym']]
                        if pd.isna(exit_price):
                            new_positions.append(pos)
                            continue
                    except KeyError:
                        new_positions.append(pos)
                        continue
                    exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price_adj - pos['entry_price']) * pos['shares']
                    capital += pos['shares'] * exit_price_adj
                    trades.append({
                        'sym': pos['sym'],
                        'entry_date': str(pos['entry_date'].date()),
                        'exit_date': str(date.date()),
                        'entry_price': round(pos['entry_price'], 4),
                        'exit_price': round(exit_price_adj, 4),
                        'shares': pos['shares'],
                        'pnl': round(pnl, 2),
                        'ret_pct': round(pnl / (pos['entry_price'] * pos['shares']) * 100, 2)
                    })
                else:
                    new_positions.append(pos)
            else:
                new_positions.append(pos)
        positions = new_positions

        # Check entries
        if len(positions) < MAX_CONCURRENT:
            for sym in QUALITY_STOCKS:
                if len(positions) >= MAX_CONCURRENT:
                    break
                # Skip if already holding this stock
                if any(p['sym'] == sym for p in positions):
                    continue
                if qmr_base_signal(indicators, sym, date):
                    if cross_asset_filter(variant, indicators, date):
                        try:
                            entry_price = close.loc[date, sym]
                            if pd.isna(entry_price):
                                continue
                        except KeyError:
                            continue
                        entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)
                        alloc = min(MAX_PER_TRADE, capital * 0.95)
                        if alloc < 10:
                            continue
                        shares = alloc / entry_price_adj
                        cost = shares * entry_price_adj
                        if cost > capital:
                            continue
                        capital -= cost
                        positions.append({
                            'sym': sym,
                            'entry_date': date,
                            'entry_price': entry_price_adj,
                            'shares': shares
                        })

        # Mark to market for equity curve
        mtm = capital
        for pos in positions:
            try:
                px = close.loc[date, pos['sym']]
                if not pd.isna(px):
                    mtm += pos['shares'] * px
            except KeyError:
                mtm += pos['shares'] * pos['entry_price']
        equity_curve.append(mtm)
        dates_curve.append(date)

    # Force-close any remaining positions at last date
    last_date = oot_dates[-1]
    for pos in positions:
        try:
            exit_price = close.loc[last_date, pos['sym']]
            if pd.isna(exit_price):
                exit_price = pos['entry_price']
        except KeyError:
            exit_price = pos['entry_price']
        exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
        pnl = (exit_price_adj - pos['entry_price']) * pos['shares']
        capital += pos['shares'] * exit_price_adj
        trades.append({
            'sym': pos['sym'],
            'entry_date': str(pos['entry_date'].date()),
            'exit_date': str(last_date.date()),
            'entry_price': round(pos['entry_price'], 4),
            'exit_price': round(exit_price_adj, 4),
            'shares': pos['shares'],
            'pnl': round(pnl, 2),
            'ret_pct': round(pnl / (pos['entry_price'] * pos['shares']) * 100, 2)
        })

    return trades, equity_curve, dates_curve


def compute_metrics(trades, equity_curve):
    """Compute performance metrics."""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'avg_pnl': 0, 'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
            'max_drawdown_pct': 0, 'final_equity': STARTING_CAPITAL,
            'total_return_pct': 0
        }

    pnls = [t['pnl'] for t in trades]
    rets = [t['ret_pct'] / 100 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0

    # Sharpe from trade returns
    if len(rets) > 1 and np.std(rets) > 0:
        # Annualize: assume ~25 trades/year as rough estimate
        trades_per_year = max(len(trades) / 4.5, 1)  # 4.5 year OOT
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = [r for r in rets if r < 0]
    if downside and np.std(downside) > 0 and len(rets) > 1:
        trades_per_year = max(len(trades) / 4.5, 1)
        sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = 0

    # Profit factor
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.01
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = drawdowns.min()

    final_eq = equity_curve[-1]
    total_ret = (final_eq - STARTING_CAPITAL) / STARTING_CAPITAL * 100

    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'final_equity': round(final_eq, 2),
        'total_return_pct': round(total_ret, 2)
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle trade signs to get p-value."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t['pnl'] for t in trades])
    observed = np.mean(pnls)
    count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        if np.mean(pnls * signs) >= observed:
            count += 1
    return count / n_perms


def five_gate_validation(metrics, trades):
    """Apply 5-gate validation."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['sharpe_pass'] = metrics['sharpe'] > SHARPE_MIN
    gates['sharpe_value'] = metrics['sharpe']

    # Gate 2: Permutation p-value < 0.05
    p_val = permutation_test(trades)
    gates['perm_p_pass'] = p_val < PERM_P_MAX
    gates['perm_p_value'] = round(p_val, 4)

    # Gate 3: Gap < 0.5 (not directly applicable without IS, use Sharpe stability proxy)
    # Use ratio of worst-half Sharpe to full Sharpe as gap proxy
    if len(trades) >= 10:
        pnls = [t['pnl'] for t in trades]
        mid = len(pnls) // 2
        first_half = pnls[:mid]
        second_half = pnls[mid:]
        s1 = np.mean(first_half) / (np.std(first_half) + 1e-10)
        s2 = np.mean(second_half) / (np.std(second_half) + 1e-10)
        gap = abs(s1 - s2) / (abs(max(s1, s2)) + 1e-10)
    else:
        gap = 1.0
    gates['gap_pass'] = gap < GAP_MAX
    gates['gap_value'] = round(gap, 4)

    # Gate 4: Max drawdown > -50%
    gates['mdd_pass'] = metrics['max_drawdown_pct'] > MDD_MIN * 100
    gates['mdd_value'] = metrics['max_drawdown_pct']

    # Gate 5: Minimum trades >= 20
    gates['min_trades_pass'] = metrics['total_trades'] >= MIN_TRADES
    gates['min_trades_value'] = metrics['total_trades']

    gates['all_pass'] = all([
        gates['sharpe_pass'], gates['perm_p_pass'], gates['gap_pass'],
        gates['mdd_pass'], gates['min_trades_pass']
    ])

    return gates


def main():
    # Fetch data
    close = fetch_data()

    # Compute indicators
    print("Computing indicators...")
    indicators = compute_indicators(close)

    # Filter to OOT period
    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    oot_dates = close.index[oot_mask].tolist()
    print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} trading days)")

    variants = {
        'A': 'QMR baseline (no cross-asset filter)',
        'B': 'QMR + HYG not in downtrend (HYG > 10d SMA)',
        'C': 'QMR + TLT 5d change < 2% (bonds not surging)',
        'D': 'QMR + GLD 5d change < 1% (gold not surging)',
        'E': 'QMR + HYG/IEF ratio rising (credit spread tightening)',
        'F': 'Combined — QMR + at least 2 of 3 cross-asset filters'
    }

    results = {}
    all_results = {}

    print("\n" + "=" * 90)
    print("CROSS-ASSET QUALITY STOCK TIMING BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL} | Max/trade: ${MAX_PER_TRADE}")
    print(f"Universe: {len(QUALITY_STOCKS)} quality stocks | Hold: {HOLD_DAYS} days | Slippage: {SLIPPAGE_PCT*100:.2f}%/side")
    print("=" * 90)

    for var_id in ['A', 'B', 'C', 'D', 'E', 'F']:
        print(f"\n--- Variant {var_id}: {variants[var_id]} ---")
        trades, equity_curve, dates_curve = run_backtest(var_id, close, indicators, oot_dates)
        metrics = compute_metrics(trades, equity_curve)
        gates = five_gate_validation(metrics, trades)

        results[var_id] = {
            'description': variants[var_id],
            'metrics': metrics,
            'gates': gates,
            'trade_count': len(trades)
        }
        all_results[var_id] = {
            'description': variants[var_id],
            'metrics': metrics,
            'gates': gates,
            'trades': trades
        }

        passed = "PASS" if gates['all_pass'] else "FAIL"
        print(f"  Trades: {metrics['total_trades']:>4} | WR: {metrics['win_rate']:>5.1f}% | "
              f"PnL: ${metrics['total_pnl']:>8.2f} | Sharpe: {metrics['sharpe']:>6.3f} | "
              f"Sortino: {metrics['sortino']:>6.3f} | PF: {metrics['profit_factor']:>5.2f} | "
              f"MDD: {metrics['max_drawdown_pct']:>6.2f}% | 5-Gate: {passed}")
        print(f"  Final equity: ${metrics['final_equity']:.2f} | Return: {metrics['total_return_pct']:.1f}%")
        gate_details = []
        for g in ['sharpe', 'perm_p', 'gap', 'mdd', 'min_trades']:
            status = "✓" if gates[f'{g}_pass'] else "✗"
            gate_details.append(f"{g}={gates[f'{g}_value']}({status})")
        print(f"  Gates: {' | '.join(gate_details)}")

    # Summary comparison vs baseline
    baseline = results['A']['metrics']
    print("\n" + "=" * 90)
    print("SUMMARY: VARIANT PERFORMANCE vs BASELINE (A)")
    print("=" * 90)
    print(f"{'Var':<4} {'Description':<55} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'PF':>6} {'Return%':>8} {'5Gate':>6}")
    print("-" * 90)
    for var_id in ['A', 'B', 'C', 'D', 'E', 'F']:
        m = results[var_id]['metrics']
        g = results[var_id]['gates']
        passed = "PASS" if g['all_pass'] else "FAIL"
        desc = variants[var_id][:53]
        print(f"  {var_id}  {desc:<55} {m['total_trades']:>5} {m['win_rate']:>6.1f} {m['sharpe']:>7.3f} "
              f"{m['profit_factor']:>6.2f} {m['total_return_pct']:>7.1f}% {passed:>6}")

    # Highlight improvements
    print("\n--- Cross-Asset Filter Impact ---")
    for var_id in ['B', 'C', 'D', 'E', 'F']:
        m = results[var_id]['metrics']
        sharpe_delta = m['sharpe'] - baseline['sharpe']
        wr_delta = m['win_rate'] - baseline['win_rate']
        trade_delta = m['total_trades'] - baseline['total_trades']
        direction = "BETTER" if sharpe_delta > 0 else "WORSE" if sharpe_delta < 0 else "SAME"
        print(f"  {var_id}: Sharpe {sharpe_delta:>+.3f} ({direction}), WR {wr_delta:>+.1f}pp, "
              f"Trades {trade_delta:>+d} (filtered {baseline['total_trades'] - m['total_trades']} signals)")

    # Save results
    output_path = Path('/home/jupiter/Lvl3Quant/data/cross_asset_quality_results.json')
    save_data = {
        'metadata': {
            'description': 'Cross-Asset Signals for Quality Stock Timing',
            'oot_period': f'{OOT_START} to {OOT_END}',
            'starting_capital': STARTING_CAPITAL,
            'max_per_trade': MAX_PER_TRADE,
            'max_concurrent': MAX_CONCURRENT,
            'slippage_pct_per_side': SLIPPAGE_PCT,
            'hold_days': HOLD_DAYS,
            'universe': QUALITY_STOCKS,
            'cross_assets': CROSS_ASSETS,
            'run_date': datetime.now().isoformat()
        },
        'variants': {}
    }
    for var_id in ['A', 'B', 'C', 'D', 'E', 'F']:
        save_data['variants'][var_id] = all_results[var_id]

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
