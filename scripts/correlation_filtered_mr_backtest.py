#!/usr/bin/env python3
"""
Correlation-Filtered Mean Reversion Backtest
=============================================
Tests whether filtering out trades based on stock-SPY correlation
improves mean reversion performance on quality names.

Thesis: Idiosyncratic dips (stock down, market flat) recover more reliably
than systematic dips (stock down because market down).

6 Variants:
  A: Baseline (no filter)
  B: Low-Correlation Only (corr < 0.5)
  C: High-Correlation Only (corr > 0.7)
  D: Stock Underperformed SPY (20d relative)
  E: SPY-Neutral Dip (SPY flat/up last 5d)
  F: Market Correction Dip (SPY down >2% last 10d)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
START_DATE = '2021-06-01'  # Extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
HOLD_DAYS = 10
RSI_PERIOD = 14
RSI_THRESHOLD = 35
HIGH_LOOKBACK = 20
DROP_PCT = 0.05
CONSEC_RED_DAYS = 3
CORR_WINDOW = 20
PERM_ITERATIONS = 1000

OUTPUT_PATH = '/home/jupiter/Lvl3Quant/data/correlation_filtered_mr_results.json'


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    """Download price data for universe + SPY."""
    tickers = UNIVERSE + ['SPY']
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
    close = data['Close']
    # Ensure all tickers present
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")
    close = close.dropna(how='all')
    close = close.ffill()
    return close


def generate_base_signals(close):
    """Generate Dual Signal D entries for each stock."""
    spy_close = close['SPY']
    spy_ret = spy_close.pct_change()
    spy_sma200 = spy_close.rolling(200).mean()

    # Precompute SPY-level features for filters
    spy_5d_ret = spy_close.pct_change(5)
    spy_10d_ret = spy_close.pct_change(10)

    all_signals = []

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        px = close[ticker]
        ret = px.pct_change()

        # RSI
        rsi = compute_rsi(px, RSI_PERIOD)

        # 20-day high
        high_20 = px.rolling(HIGH_LOOKBACK).max()

        # Drop from 20-day high
        drop_from_high = (px - high_20) / high_20

        # Consecutive red days
        is_red = ret < 0
        is_green = ret > 0
        consec_red = is_red.astype(int)
        # Count consecutive reds ending yesterday
        consec_count = pd.Series(0, index=px.index)
        count = 0
        for i in range(len(px)):
            if is_red.iloc[i]:
                count += 1
            else:
                count = 0
            consec_count.iloc[i] = count

        # Rolling correlation with SPY
        rolling_corr = ret.rolling(CORR_WINDOW).corr(spy_ret)

        # Stock 20d return vs SPY 20d return
        stock_20d_ret = px.pct_change(20)
        spy_20d_ret_series = spy_close.pct_change(20)

        for i in range(1, len(px)):
            date = px.index[i]
            if date < pd.Timestamp(OOT_START) or date > pd.Timestamp(OOT_END):
                continue

            # Dual Signal D conditions:
            # 1. Stock dropped >5% from 20-day high
            if pd.isna(drop_from_high.iloc[i]) or drop_from_high.iloc[i] > -DROP_PCT:
                continue
            # 2. RSI < 35
            if pd.isna(rsi.iloc[i]) or rsi.iloc[i] >= RSI_THRESHOLD:
                continue
            # 3. First green after 3+ consecutive red days
            #    Yesterday had consec_red >= 3, today is green
            if i < 1:
                continue
            if consec_count.iloc[i-1] < CONSEC_RED_DAYS:
                continue
            if not is_green.iloc[i]:
                continue

            # Collect filter features
            corr_val = rolling_corr.iloc[i] if not pd.isna(rolling_corr.iloc[i]) else 0.5
            stock_20d = stock_20d_ret.iloc[i] if not pd.isna(stock_20d_ret.iloc[i]) else 0.0
            spy_20d = spy_20d_ret_series.iloc[i] if not pd.isna(spy_20d_ret_series.iloc[i]) else 0.0
            spy_5d = spy_5d_ret.iloc[i] if not pd.isna(spy_5d_ret.iloc[i]) else 0.0
            spy_10d = spy_10d_ret.iloc[i] if not pd.isna(spy_10d_ret.iloc[i]) else 0.0
            spy_sma_val = spy_sma200.iloc[i] if not pd.isna(spy_sma200.iloc[i]) else spy_close.iloc[i]
            is_bull = spy_close.iloc[i] > spy_sma_val

            all_signals.append({
                'date': date,
                'ticker': ticker,
                'entry_price': px.iloc[i],
                'corr_20d': corr_val,
                'stock_20d_ret': stock_20d,
                'spy_20d_ret': spy_20d,
                'spy_5d_ret': spy_5d,
                'spy_10d_ret': spy_10d,
                'is_bull': is_bull,
                'rsi': rsi.iloc[i],
                'drop_pct': drop_from_high.iloc[i],
            })

    signals_df = pd.DataFrame(all_signals)
    if len(signals_df) > 0:
        signals_df = signals_df.sort_values('date').reset_index(drop=True)
    print(f"Generated {len(signals_df)} base signals across {signals_df['ticker'].nunique() if len(signals_df) > 0 else 0} tickers")
    return signals_df


def apply_filter(signals_df, variant):
    """Apply variant-specific filter to signals."""
    if variant == 'A':
        return signals_df.copy()
    elif variant == 'B':
        return signals_df[signals_df['corr_20d'] < 0.5].copy()
    elif variant == 'C':
        return signals_df[signals_df['corr_20d'] > 0.7].copy()
    elif variant == 'D':
        return signals_df[signals_df['stock_20d_ret'] < signals_df['spy_20d_ret']].copy()
    elif variant == 'E':
        return signals_df[signals_df['spy_5d_ret'] >= 0].copy()
    elif variant == 'F':
        return signals_df[signals_df['spy_10d_ret'] < -0.02].copy()
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_backtest(signals_df, close):
    """Run portfolio backtest with position sizing and concurrency limits."""
    if len(signals_df) == 0:
        return {
            'trades': [],
            'equity_curve': [STARTING_CAPITAL],
            'sharpe': 0, 'sortino': 0, 'win_rate': 0,
            'profit_factor': 0, 'max_drawdown': 0, 'total_return': 0,
            'num_trades': 0, 'bull_sharpe': 0, 'bear_sharpe': 0,
        }

    trades = []
    capital = STARTING_CAPITAL
    daily_returns = []
    active_positions = []  # list of (ticker, entry_date, entry_price, shares, exit_date_target)
    equity_curve = [capital]

    # Get all trading days in OOT period
    oot_dates = close.loc[OOT_START:OOT_END].index
    spy_close = close['SPY']
    spy_sma200 = spy_close.rolling(200).mean()

    signals_by_date = {}
    for _, row in signals_df.iterrows():
        d = row['date']
        if d not in signals_by_date:
            signals_by_date[d] = []
        signals_by_date[d].append(row)

    for date in oot_dates:
        # Check exits
        new_active = []
        day_pnl = 0
        for pos in active_positions:
            ticker, entry_date, entry_price, shares, exit_target = pos
            if date >= exit_target:
                # Exit
                if date in close.index and ticker in close.columns:
                    exit_price = close.loc[date, ticker]
                    if pd.isna(exit_price):
                        new_active.append(pos)
                        continue
                    # Apply slippage on exit (selling)
                    exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price_adj - entry_price) * shares
                    capital += exit_price_adj * shares
                    day_pnl += pnl
                    trades.append({
                        'ticker': ticker,
                        'entry_date': str(entry_date.date()),
                        'exit_date': str(date.date()),
                        'entry_price': round(entry_price, 4),
                        'exit_price': round(exit_price_adj, 4),
                        'shares': round(shares, 4),
                        'pnl': round(pnl, 2),
                        'return_pct': round((exit_price_adj / entry_price - 1) * 100, 2),
                    })
                else:
                    new_active.append(pos)
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check new entries
        if date in signals_by_date:
            for sig in signals_by_date[date]:
                if len(active_positions) >= MAX_CONCURRENT:
                    break
                ticker = sig['ticker']
                # Don't double up on same ticker
                if any(p[0] == ticker for p in active_positions):
                    continue
                entry_price = sig['entry_price']
                # Apply slippage on entry (buying)
                entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)
                position_size = min(MAX_PER_TRADE, capital * 0.95)  # Keep 5% buffer
                if position_size < 10:
                    continue
                shares = position_size / entry_price_adj
                capital -= entry_price_adj * shares

                # Find exit date (HOLD_DAYS trading days later)
                future_dates = oot_dates[oot_dates > date]
                if len(future_dates) >= HOLD_DAYS:
                    exit_target = future_dates[HOLD_DAYS - 1]
                else:
                    exit_target = future_dates[-1] if len(future_dates) > 0 else date + timedelta(days=14)

                active_positions.append((ticker, date, entry_price_adj, shares, exit_target))

        # Calculate daily portfolio value
        port_value = capital
        for pos in active_positions:
            ticker, _, _, shares, _ = pos
            if date in close.index and ticker in close.columns:
                current_price = close.loc[date, ticker]
                if not pd.isna(current_price):
                    port_value += current_price * shares

        daily_ret = (port_value / equity_curve[-1]) - 1 if equity_curve[-1] > 0 else 0
        daily_returns.append(daily_ret)
        equity_curve.append(port_value)

    # Force close any remaining positions at end
    for pos in active_positions:
        ticker, entry_date, entry_price, shares, _ = pos
        last_date = oot_dates[-1]
        if ticker in close.columns:
            exit_price = close.loc[last_date, ticker]
            if not pd.isna(exit_price):
                exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price_adj - entry_price) * shares
                trades.append({
                    'ticker': ticker,
                    'entry_date': str(entry_date.date()),
                    'exit_date': str(last_date.date()),
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(exit_price_adj, 4),
                    'shares': round(shares, 4),
                    'pnl': round(pnl, 2),
                    'return_pct': round((exit_price_adj / entry_price - 1) * 100, 2),
                })

    # Compute metrics
    daily_returns = np.array(daily_returns)
    equity_arr = np.array(equity_curve)

    # Sharpe (annualized)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0.0

    # Win rate
    trade_pnls = [t['pnl'] for t in trades]
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    win_rate = len(wins) / len(trade_pnls) * 100 if trade_pnls else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    max_dd = np.min(drawdown) * 100

    # Total return
    total_return = (equity_arr[-1] / equity_arr[0] - 1) * 100

    # Regime analysis (Bull vs Bear based on SPY > 200-SMA)
    bull_returns = []
    bear_returns = []
    for i, date in enumerate(oot_dates):
        if i >= len(daily_returns):
            break
        spy_val = spy_close.loc[date] if date in spy_close.index else np.nan
        sma_val = spy_sma200.loc[date] if date in spy_sma200.index else np.nan
        if pd.isna(spy_val) or pd.isna(sma_val):
            continue
        if spy_val > sma_val:
            bull_returns.append(daily_returns[i])
        else:
            bear_returns.append(daily_returns[i])

    bull_returns = np.array(bull_returns) if bull_returns else np.array([0.0])
    bear_returns = np.array(bear_returns) if bear_returns else np.array([0.0])

    bull_sharpe = (np.mean(bull_returns) / np.std(bull_returns) * np.sqrt(252)) if np.std(bull_returns) > 0 else 0
    bear_sharpe = (np.mean(bear_returns) / np.std(bear_returns) * np.sqrt(252)) if np.std(bear_returns) > 0 else 0

    return {
        'trades': trades,
        'equity_curve': [round(e, 2) for e in equity_curve[::20]],  # Sample every 20 days
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown': round(max_dd, 1),
        'total_return': round(total_return, 1),
        'num_trades': len(trades),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'final_equity': round(equity_arr[-1], 2),
    }


def permutation_test(signals_df, close, actual_sharpe, n_iter=PERM_ITERATIONS):
    """Permutation test: shuffle signal dates and re-run backtest."""
    if len(signals_df) == 0 or actual_sharpe == 0:
        return 1.0

    perm_sharpes = []
    all_dates = close.loc[OOT_START:OOT_END].index
    n_signals = len(signals_df)

    for _ in range(n_iter):
        # Create shuffled signals with random dates
        perm_df = signals_df.copy()
        random_indices = np.random.choice(len(all_dates), size=n_signals, replace=True)
        perm_df['date'] = [all_dates[i] for i in random_indices]
        # Update entry prices to match random dates
        for idx in perm_df.index:
            ticker = perm_df.loc[idx, 'ticker']
            date = perm_df.loc[idx, 'date']
            if date in close.index and ticker in close.columns:
                price = close.loc[date, ticker]
                if not pd.isna(price):
                    perm_df.loc[idx, 'entry_price'] = price
        perm_df = perm_df.sort_values('date').reset_index(drop=True)
        result = run_backtest(perm_df, close)
        perm_sharpes.append(result['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    return round(p_value, 4)


def five_gate_validation(result, p_value):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': result['sharpe'] > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': abs(result['bull_sharpe'] - result['bear_sharpe']) / max(abs(result['bull_sharpe']), abs(result['bear_sharpe']), 0.001) < 0.5,
        'mdd_gt_neg50': result['max_drawdown'] > -50,
        'trades_gte_20': result['num_trades'] >= 20,
    }
    gates['all_passed'] = all(gates.values())
    return gates


def main():
    print("=" * 70)
    print("CORRELATION-FILTERED MEAN REVERSION BACKTEST")
    print("=" * 70)

    # Download data
    close = download_data()
    print(f"Data range: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"Tickers available: {[t for t in UNIVERSE if t in close.columns]}")

    # Generate base signals
    signals_df = generate_base_signals(close)
    if len(signals_df) == 0:
        print("ERROR: No signals generated. Check data and parameters.")
        return

    # Print signal statistics
    print(f"\nSignal Statistics:")
    print(f"  Total signals: {len(signals_df)}")
    print(f"  Date range: {signals_df['date'].min().date()} to {signals_df['date'].max().date()}")
    print(f"  Mean 20d correlation: {signals_df['corr_20d'].mean():.3f}")
    print(f"  Signals with corr < 0.5: {(signals_df['corr_20d'] < 0.5).sum()}")
    print(f"  Signals with corr > 0.7: {(signals_df['corr_20d'] > 0.7).sum()}")

    # Run all variants
    variants = {
        'A': 'Baseline (no filter)',
        'B': 'Low-Correlation Only (corr < 0.5)',
        'C': 'High-Correlation Only (corr > 0.7)',
        'D': 'Stock Underperformed SPY (20d)',
        'E': 'SPY-Neutral Dip (SPY flat/up 5d)',
        'F': 'Market Correction Dip (SPY down >2% 10d)',
    }

    results = {}

    for variant_key, variant_name in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {variant_key}: {variant_name}")
        print(f"{'─' * 60}")

        filtered = apply_filter(signals_df, variant_key)
        print(f"  Signals after filter: {len(filtered)}")

        if len(filtered) == 0:
            print(f"  SKIPPED: No signals pass filter")
            results[variant_key] = {
                'name': variant_name,
                'num_signals': 0,
                'num_trades': 0,
                'sharpe': 0, 'sortino': 0, 'win_rate': 0,
                'profit_factor': 0, 'max_drawdown': 0, 'total_return': 0,
                'bull_sharpe': 0, 'bear_sharpe': 0,
                'p_value': 1.0,
                'gates': {'all_passed': False},
            }
            continue

        bt_result = run_backtest(filtered, close)

        print(f"  Trades: {bt_result['num_trades']}")
        print(f"  Sharpe: {bt_result['sharpe']}")
        print(f"  Sortino: {bt_result['sortino']}")
        print(f"  Win Rate: {bt_result['win_rate']}%")
        print(f"  Profit Factor: {bt_result['profit_factor']}")
        print(f"  Max DD: {bt_result['max_drawdown']}%")
        print(f"  Total Return: {bt_result['total_return']}%")
        print(f"  Final Equity: ${bt_result['final_equity']}")
        print(f"  Bull Sharpe: {bt_result['bull_sharpe']} | Bear Sharpe: {bt_result['bear_sharpe']}")

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...")
        p_value = permutation_test(filtered, close, bt_result['sharpe'], PERM_ITERATIONS)
        print(f"  Permutation p-value: {p_value}")

        # 5-gate validation
        gates = five_gate_validation(bt_result, p_value)
        print(f"  5-Gate: {'PASS' if gates['all_passed'] else 'FAIL'} — {gates}")

        results[variant_key] = {
            'name': variant_name,
            'num_signals': len(filtered),
            'num_trades': bt_result['num_trades'],
            'sharpe': bt_result['sharpe'],
            'sortino': bt_result['sortino'],
            'win_rate': bt_result['win_rate'],
            'profit_factor': bt_result['profit_factor'],
            'max_drawdown': bt_result['max_drawdown'],
            'total_return': bt_result['total_return'],
            'final_equity': bt_result['final_equity'],
            'bull_sharpe': bt_result['bull_sharpe'],
            'bear_sharpe': bt_result['bear_sharpe'],
            'p_value': p_value,
            'gates': gates,
            'equity_curve_sampled': bt_result['equity_curve'],
            'sample_trades': bt_result['trades'][:10],  # First 10 trades
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<8} {'Name':<40} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR%':>5} {'PF':>6} {'MDD%':>6} {'Ret%':>7} {'p-val':>6} {'5G':>4}")
    print("-" * 110)
    for k, v in results.items():
        passed = 'PASS' if v['gates'].get('all_passed', False) else 'FAIL'
        print(f"{k:<8} {v['name'][:40]:<40} {v['num_trades']:>6} {v['sharpe']:>7.3f} {v['sortino']:>8.3f} {v['win_rate']:>5.1f} {v['profit_factor']:>6.3f} {v['max_drawdown']:>6.1f} {v['total_return']:>7.1f} {v['p_value']:>6.4f} {passed:>4}")

    # Save results
    output = {
        'metadata': {
            'strategy': 'Correlation-Filtered Mean Reversion',
            'base_signal': 'Dual Signal D',
            'universe': UNIVERSE,
            'oot_period': f'{OOT_START} to {OOT_END}',
            'starting_capital': STARTING_CAPITAL,
            'max_per_trade': MAX_PER_TRADE,
            'max_concurrent': MAX_CONCURRENT,
            'slippage_pct': SLIPPAGE_PCT,
            'hold_days': HOLD_DAYS,
            'run_timestamp': datetime.now().isoformat(),
        },
        'variants': results,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
