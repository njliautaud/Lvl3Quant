#!/usr/bin/env python3
"""
TRUE Out-of-Sample Validation: Vol Compression Sector Rotation
==============================================================
Tests the AVO-evolved strategy (v17) on 2026 YTD data that the
evolution NEVER scored on. Evolution used 2022H1-2025H2 walk-forward.

Downloads real market data via yfinance.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, date

# ── Import the evolved strategy directly ──────────────────────────────────────
STRATEGY_DIR = '/home/jupiter/teleclaude-main/runs/vol_compression-20260823-052034/work'
sys.path.insert(0, STRATEGY_DIR)
import strategy as strat

# ── Constants ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
SLIPPAGE_BPS = 1  # 1 bps per side
WARMUP_START = '2025-09-01'
OOS_START = '2026-01-02'
OOS_END = None  # today

TICKERS = strat.SECTOR_ETFS + [strat.BENCHMARK]
VIX_TICKER = '^VIX'


def download_data():
    """Download all required data from yfinance."""
    import yfinance as yf

    print("Downloading market data...")
    all_tickers = TICKERS + [VIX_TICKER]
    data = yf.download(all_tickers, start=WARMUP_START, end=None, auto_adjust=True, progress=False)

    # yf.download returns MultiIndex columns (Price, Ticker) for multiple tickers
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].copy()
    else:
        prices = data[['Close']].copy()

    # Separate VIX
    vix = prices[VIX_TICKER].copy() if VIX_TICKER in prices.columns else None
    prices = prices.drop(columns=[VIX_TICKER], errors='ignore')

    # Forward-fill gaps, drop rows where all NaN
    prices = prices.ffill().dropna(how='all')
    if vix is not None:
        vix = vix.ffill()

    print(f"  Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Total trading days: {len(prices)}")
    print(f"  Tickers: {list(prices.columns)}")

    return prices, vix


def run_simulation(prices, vix):
    """
    Full portfolio simulation with position sizing, concurrent limits,
    trailing stops, profit floors, dead money exits.
    """
    # Generate signals across entire dataset (needs warmup)
    spy = prices[strat.BENCHMARK]
    sector_prices = prices[strat.SECTOR_ETFS]
    signals = strat.generate_signals(sector_prices, spy, vix)

    # Only simulate from OOS start
    oos_mask = signals.index >= OOS_START
    oos_dates = signals.index[oos_mask]

    print(f"\n  OOS period: {oos_dates[0].date()} to {oos_dates[-1].date()}")
    print(f"  OOS trading days: {len(oos_dates)}")

    # Count raw signals before any filtering
    raw_signal_count = signals.loc[oos_mask].sum().sum()
    print(f"  Raw entry signals (pre-capacity): {int(raw_signal_count)}")

    # Portfolio state
    capital = INITIAL_CAPITAL
    positions = []  # list of dicts
    trades = []     # completed trades
    equity_curve = []

    peak_equity = capital
    portfolio_dd = 0.0

    for day in oos_dates:
        day_str = day.strftime('%Y-%m-%d')

        # ── Check exits first ─────────────────────────────────────────────
        positions_to_keep = []
        for pos in positions:
            etf = pos['etf']
            if etf not in prices.columns:
                positions_to_keep.append(pos)
                continue
            current_price = prices.at[day, etf]
            if pd.isna(current_price):
                positions_to_keep.append(pos)
                continue

            if strat.should_exit(pos, current_price, day_str, portfolio_dd):
                # Exit with slippage
                exit_price = current_price * (1 - SLIPPAGE_BPS / 10000)
                shares = pos['shares']
                pnl = (exit_price - pos['entry_price_adj']) * shares
                pnl_pct = (exit_price - pos['entry_price_adj']) / pos['entry_price_adj']
                capital += pos['notional'] + pnl

                # Determine VIX regime at entry
                entry_vix = vix.get(pos['entry_date'], 20) if vix is not None else 20
                if pd.isna(entry_vix):
                    entry_vix = 20

                trades.append({
                    'etf': etf,
                    'entry_date': pos['entry_date'],
                    'exit_date': day_str,
                    'entry_price': pos['entry_price_adj'],
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl_dollar': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': np.busday_count(
                        np.datetime64(pos['entry_date'], 'D'),
                        np.datetime64(day_str, 'D')),
                    'entry_vix': entry_vix,
                })
            else:
                # Update HWM tracking
                current_price_val = prices.at[day, etf]
                if not pd.isna(current_price_val):
                    hwm_key = 'hwm' if 'hwm' in pos else 'high_water'
                    if current_price_val > pos.get(hwm_key, pos['entry_price_adj']):
                        pos[hwm_key] = current_price_val
                positions_to_keep.append(pos)

        positions = positions_to_keep

        # ── Check entries ─────────────────────────────────────────────────
        if len(positions) < strat.MAX_CONCURRENT:
            day_signals = signals.loc[day]
            active_etfs = {p['etf'] for p in positions}
            candidates = [etf for etf in strat.SECTOR_ETFS
                          if day_signals.get(etf, False) and etf not in active_etfs]

            for etf in candidates:
                if len(positions) >= strat.MAX_CONCURRENT:
                    break
                if capital < strat.MAX_PER_TRADE * 0.5:
                    break  # not enough capital

                price = prices.at[day, etf]
                if pd.isna(price):
                    continue

                # Entry with slippage
                entry_price = price * (1 + SLIPPAGE_BPS / 10000)
                notional = min(strat.MAX_PER_TRADE, capital)
                shares = notional / entry_price
                capital -= notional

                positions.append({
                    'etf': etf,
                    'entry_date': day_str,
                    'entry_price_adj': entry_price,
                    'shares': shares,
                    'notional': notional,
                    'hwm': entry_price,
                })

        # ── Mark to market ────────────────────────────────────────────────
        pos_value = 0.0
        for pos in positions:
            etf = pos['etf']
            cp = prices.at[day, etf] if etf in prices.columns else pos['entry_price_adj']
            if pd.isna(cp):
                cp = pos['entry_price_adj']
            pos_value += cp * pos['shares']

        total_equity = capital + pos_value
        peak_equity = max(peak_equity, total_equity)
        portfolio_dd = (total_equity - peak_equity) / peak_equity if peak_equity > 0 else 0

        equity_curve.append({
            'date': day,
            'equity': total_equity,
            'cash': capital,
            'positions': len(positions),
            'drawdown': portfolio_dd,
        })

    # Close any remaining positions at last price
    last_day = oos_dates[-1]
    last_day_str = last_day.strftime('%Y-%m-%d')
    for pos in positions:
        etf = pos['etf']
        cp = prices.at[last_day, etf] if etf in prices.columns else pos['entry_price_adj']
        if pd.isna(cp):
            cp = pos['entry_price_adj']
        exit_price = cp * (1 - SLIPPAGE_BPS / 10000)
        shares = pos['shares']
        pnl = (exit_price - pos['entry_price_adj']) * shares
        pnl_pct = (exit_price - pos['entry_price_adj']) / pos['entry_price_adj']
        capital += pos['notional'] + pnl

        entry_vix = vix.get(pos['entry_date'], 20) if vix is not None else 20
        if pd.isna(entry_vix):
            entry_vix = 20

        trades.append({
            'etf': etf,
            'entry_date': pos['entry_date'],
            'exit_date': last_day_str,
            'entry_price': pos['entry_price_adj'],
            'exit_price': exit_price,
            'shares': shares,
            'pnl_dollar': pnl,
            'pnl_pct': pnl_pct,
            'days_held': np.busday_count(
                np.datetime64(pos['entry_date'], 'D'),
                np.datetime64(last_day_str, 'D')),
            'entry_vix': entry_vix,
        })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)


def compute_spy_benchmark(prices):
    """Buy-and-hold SPY over the OOS period."""
    spy = prices[strat.BENCHMARK]
    oos = spy[spy.index >= OOS_START]
    spy_ret = oos / oos.iloc[0]
    return spy_ret


def compute_metrics(equity_df, trades_df, spy_bh):
    """Compute all required performance metrics."""
    print("\n" + "=" * 70)
    print("TRUE OUT-OF-SAMPLE RESULTS: 2026 YTD")
    print("Strategy: Vol Compression Sector Rotation (AVO v17)")
    print("Evolution training period: 2022H1 - 2025H2")
    print("Validation period: 2026 YTD (NEVER seen by evolution)")
    print("=" * 70)

    eq = equity_df.set_index('date')['equity']
    daily_ret = eq.pct_change().dropna()

    # ── Portfolio-level metrics ───────────────────────────────────────────
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe (annualized)
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    print(f"\n{'PORTFOLIO METRICS':=^50}")
    print(f"  Initial Capital:    ${INITIAL_CAPITAL:,.0f}")
    print(f"  Final Equity:       ${eq.iloc[-1]:,.2f}")
    print(f"  Total Return:       {total_return:.2%}")
    print(f"  CAGR:               {cagr:.2%}")
    print(f"  Sharpe Ratio:       {sharpe:.2f}", end="")
    if sharpe > 1.0:
        print("  (strong risk-adjusted return)")
    elif sharpe > 0.5:
        print("  (acceptable risk-adjusted return)")
    elif sharpe > 0:
        print("  (weak but positive)")
    else:
        print("  (negative risk-adjusted return)")
    print(f"  Sortino Ratio:      {sortino:.2f}")
    print(f"  Max Drawdown:       {max_dd:.2%}")
    print(f"  OOS Period:         {eq.index[0].date()} to {eq.index[-1].date()} ({days} days)")

    # ── Trade-level metrics ───────────────────────────────────────────────
    if len(trades_df) > 0:
        n_trades = len(trades_df)
        winners = trades_df[trades_df['pnl_dollar'] > 0]
        losers = trades_df[trades_df['pnl_dollar'] <= 0]
        win_rate = len(winners) / n_trades

        avg_pnl = trades_df['pnl_dollar'].mean()
        avg_pnl_pct = trades_df['pnl_pct'].mean()
        avg_winner = winners['pnl_dollar'].mean() if len(winners) > 0 else 0
        avg_loser = losers['pnl_dollar'].mean() if len(losers) > 0 else 0

        gross_profit = winners['pnl_dollar'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_dollar'].sum()) if len(losers) > 0 else 0.001
        profit_factor = gross_profit / gross_loss

        avg_days = trades_df['days_held'].mean()

        print(f"\n{'TRADE METRICS':=^50}")
        print(f"  Total Trades:       {n_trades}")
        print(f"  Win Rate:           {win_rate:.1%} ({len(winners)}W / {len(losers)}L)")
        print(f"  Avg Profit/Trade:   ${avg_pnl:.2f} ({avg_pnl_pct:.2%})")
        print(f"  Avg Winner:         ${avg_winner:.2f}")
        print(f"  Avg Loser:          ${avg_loser:.2f}")
        print(f"  Profit Factor:      {profit_factor:.2f}  (for every $1 lost, earned ${profit_factor:.2f})")
        print(f"  Avg Days Held:      {avg_days:.1f}")
        print(f"  Total P&L:          ${trades_df['pnl_dollar'].sum():.2f}")

        # ── By sector ────────────────────────────────────────────────────
        print(f"\n{'BY SECTOR':=^50}")
        sector_stats = trades_df.groupby('etf').agg(
            trades=('pnl_dollar', 'count'),
            total_pnl=('pnl_dollar', 'sum'),
            avg_pnl=('pnl_pct', 'mean'),
            win_rate=('pnl_dollar', lambda x: (x > 0).mean()),
        ).sort_values('total_pnl', ascending=False)
        print(sector_stats.to_string())

        # ── By VIX regime ────────────────────────────────────────────────
        print(f"\n{'BY VIX REGIME':=^50}")

        def vix_regime(v):
            if v < 16:
                return 'Low (<16)'
            elif v <= 25:
                return 'Mid (16-25)'
            elif v <= 40:
                return 'High (25-40)'
            else:
                return 'Crisis (>40)'

        trades_df['regime'] = trades_df['entry_vix'].apply(vix_regime)
        regime_stats = trades_df.groupby('regime').agg(
            trades=('pnl_dollar', 'count'),
            total_pnl=('pnl_dollar', 'sum'),
            avg_pnl_pct=('pnl_pct', 'mean'),
            win_rate=('pnl_dollar', lambda x: (x > 0).mean()),
            avg_days=('days_held', 'mean'),
        )
        for regime in ['Low (<16)', 'Mid (16-25)', 'High (25-40)', 'Crisis (>40)']:
            if regime in regime_stats.index:
                r = regime_stats.loc[regime]
                print(f"  {regime:15s}: {int(r['trades']):3d} trades, "
                      f"WR={r['win_rate']:.0%}, "
                      f"avg={r['avg_pnl_pct']:.2%}, "
                      f"total=${r['total_pnl']:.2f}, "
                      f"avg hold={r['avg_days']:.1f}d")
            else:
                print(f"  {regime:15s}: no trades")

        # ── By month ─────────────────────────────────────────────────────
        print(f"\n{'BY MONTH':=^50}")
        trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
        month_stats = trades_df.groupby('month').agg(
            trades=('pnl_dollar', 'count'),
            total_pnl=('pnl_dollar', 'sum'),
            win_rate=('pnl_dollar', lambda x: (x > 0).mean()),
        )
        for month, row in month_stats.iterrows():
            bar = '+' * int(max(0, row['total_pnl'] / 5)) + '-' * int(max(0, -row['total_pnl'] / 5))
            print(f"  {month}: {int(row['trades']):3d} trades, "
                  f"WR={row['win_rate']:.0%}, "
                  f"P&L=${row['total_pnl']:>8.2f}  {bar}")

    else:
        print("\n  *** NO TRADES GENERATED ***")
        print("  The strategy produced zero entry signals during 2026.")

    # ── SPY benchmark comparison ─────────────────────────────────────────
    print(f"\n{'SPY BUY-AND-HOLD COMPARISON':=^50}")
    spy_return = (spy_bh.iloc[-1] / spy_bh.iloc[0]) - 1
    spy_daily = spy_bh.pct_change().dropna()
    spy_sharpe = (spy_daily.mean() / spy_daily.std()) * np.sqrt(252) if spy_daily.std() > 0 else 0
    spy_peak = spy_bh.cummax()
    spy_dd = ((spy_bh - spy_peak) / spy_peak).min()

    print(f"  SPY Total Return:   {spy_return:.2%}")
    print(f"  SPY Sharpe:         {spy_sharpe:.2f}")
    print(f"  SPY Max Drawdown:   {spy_dd:.2%}")
    print(f"  Strategy Return:    {total_return:.2%}")
    excess = total_return - spy_return
    print(f"  Excess Return:      {excess:+.2%} {'(outperformed)' if excess > 0 else '(underperformed)'}")
    # Capital utilization note
    print(f"\n  NOTE: Strategy uses max 75% of capital at any time (3 x $2,500 / $10,000)")
    print(f"        Remaining 25% sits in cash. SPY is 100% invested.")

    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'n_trades': len(trades_df) if len(trades_df) > 0 else 0,
        'win_rate': win_rate if len(trades_df) > 0 else 0,
        'profit_factor': profit_factor if len(trades_df) > 0 else 0,
    }


def main():
    # Download data
    prices, vix = download_data()

    # Run simulation
    print("\nRunning portfolio simulation...")
    equity_df, trades_df = run_simulation(prices, vix)

    # SPY benchmark
    spy_bh = compute_spy_benchmark(prices)

    # Compute and print metrics
    metrics = compute_metrics(equity_df, trades_df, spy_bh)

    # Save trade log
    csv_path = '/home/jupiter/Lvl3Quant/validation/vol_compression_2026_trades.csv'
    if len(trades_df) > 0:
        trades_df.to_csv(csv_path, index=False)
        print(f"\nTrade log saved to: {csv_path}")
    else:
        print(f"\nNo trades to save.")

    # Save equity curve
    eq_path = '/home/jupiter/Lvl3Quant/validation/vol_compression_2026_equity.csv'
    equity_df.to_csv(eq_path, index=False)
    print(f"Equity curve saved to: {eq_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
