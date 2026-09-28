#!/usr/bin/env python3
"""
Strategy 4: Volatility Breakout (Mean-Reversion of Volatility)
- When realized vol drops to multi-month lows, it tends to explode
- Buy stocks with compressed vol + positive momentum for direction
- Walk-forward backtest
"""
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/aggressive"

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    'ADBE', 'NFLX', 'AVGO', 'QCOM', 'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY',
    'JPM', 'BAC', 'WFC', 'GS', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD',
    'HD', 'CAT', 'BA', 'HON', 'XOM', 'CVX', 'COP', 'V', 'MA', 'DIS',
    'ORCL', 'ACN', 'IBM', 'NOW', 'ISRG', 'REGN', 'LIN', 'NEE', 'UNP', 'RTX',
]

def download_data():
    print("Downloading data for vol breakout strategy...")
    end = datetime(2026, 7, 1)
    start = datetime(2019, 1, 1)
    data = yf.download(UNIVERSE + ['SPY'], start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')
    return close

def compute_vol_signals(close, vol_window=20, vol_lookback=120, vol_percentile=20):
    """
    For each stock, compute:
    - Current realized vol (vol_window days)
    - Percentile rank vs last vol_lookback days
    - Momentum (20-day return)
    Returns signal DataFrame with buy signals
    """
    signals = {}

    for ticker in close.columns:
        if ticker == 'SPY':
            continue

        prices = close[ticker].dropna()
        if len(prices) < vol_lookback + vol_window:
            continue

        # Realized vol (annualized)
        log_ret = np.log(prices / prices.shift(1))
        current_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

        # Vol percentile rank over lookback
        vol_pctile = current_vol.rolling(vol_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Momentum (20-day return)
        momentum = prices.pct_change(20)

        # Buy signal: vol at low percentile AND positive momentum
        buy_signal = (vol_pctile < vol_percentile / 100) & (momentum > 0)

        signals[ticker] = pd.DataFrame({
            'vol': current_vol,
            'vol_pctile': vol_pctile,
            'momentum': momentum,
            'buy': buy_signal.astype(int)
        })

    return signals

def backtest_vol_breakout(close, signals, hold_days=20, max_positions=2):
    """Walk-forward backtest of vol breakout strategy"""
    all_dates = close.index
    port_returns = pd.Series(0.0, index=all_dates)

    positions = []  # (ticker, entry_date, exit_date_idx)
    trades = []

    for i, date in enumerate(all_dates):
        # Check for exit
        active = [p for p in positions if p['exit_idx'] > i]

        # Check for new entries (only if under max positions)
        if len(active) < max_positions:
            candidates = []
            for ticker, sig in signals.items():
                if date in sig.index and sig.loc[date, 'buy'] == 1:
                    # Check not already in position
                    active_tickers = [p['ticker'] for p in active]
                    if ticker not in active_tickers:
                        candidates.append((ticker, sig.loc[date, 'vol_pctile']))

            # Pick lowest vol percentile (most compressed)
            candidates.sort(key=lambda x: x[1])

            for ticker, _ in candidates[:max_positions - len(active)]:
                exit_idx = min(i + hold_days, len(all_dates) - 1)
                entry_price = close[ticker].iloc[i] if ticker in close.columns else None
                exit_price = close[ticker].iloc[exit_idx] if ticker in close.columns and exit_idx < len(close) else None

                if entry_price and exit_price and entry_price > 0:
                    positions.append({
                        'ticker': ticker,
                        'entry_idx': i,
                        'exit_idx': exit_idx,
                        'entry_date': date,
                    })
                    trades.append({
                        'ticker': ticker,
                        'entry_date': str(date.date()),
                        'exit_date': str(all_dates[exit_idx].date()),
                        'entry_price': float(entry_price),
                        'exit_price': float(exit_price),
                        'return': float(exit_price / entry_price - 1),
                    })

        # Compute daily return from active positions
        active = [p for p in positions if p['entry_idx'] <= i <= p['exit_idx']]
        if active:
            daily_ret = 0
            for p in active:
                ticker = p['ticker']
                if i > 0 and ticker in close.columns:
                    prev = close[ticker].iloc[i-1]
                    curr = close[ticker].iloc[i]
                    if prev > 0:
                        daily_ret += (curr / prev - 1) / max_positions
            port_returns.iloc[i] = daily_ret

    trades_df = pd.DataFrame(trades)

    # Trim to active period
    first_signal = None
    for ticker, sig in signals.items():
        first_buy = sig[sig['buy'] == 1].index.min()
        if first_buy is not None:
            if first_signal is None or first_buy < first_signal:
                first_signal = first_buy

    if first_signal:
        port_returns = port_returns.loc[first_signal:]

    return port_returns, trades_df

def analyze(returns, trades_df, name, spy_close):
    if len(returns) < 252:
        return None

    n_days = len(returns)
    n_years = n_days / 252
    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade stats
    n_trades = len(trades_df)
    if n_trades > 0:
        wr = (trades_df['return'] > 0).mean()
        avg_ret = trades_df['return'].mean()
        wins = trades_df.loc[trades_df['return'] > 0, 'return']
        losses = trades_df.loc[trades_df['return'] < 0, 'return']
        pf = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else float('inf')
    else:
        wr = avg_ret = pf = 0

    # R1 regime
    spy_monthly = spy_close.resample('ME').last().pct_change()
    strat_monthly = returns.resample('ME').sum()
    common_idx = strat_monthly.index.intersection(spy_monthly.index)

    r1_pass = False
    sharpe_green = sharpe_red = regime_gap = 0
    if len(common_idx) > 10:
        sm = strat_monthly.loc[common_idx]
        spy_m = spy_monthly.loc[common_idx]
        green = sm[spy_m > 0.01]
        red = sm[spy_m < -0.01]
        sharpe_green = green.mean() / green.std() * np.sqrt(12) if len(green) > 2 and green.std() > 0 else 0
        sharpe_red = red.mean() / red.std() * np.sqrt(12) if len(red) > 2 and red.std() > 0 else 0
        max_s = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_s if max_s > 0 else 0
        r1_pass = regime_gap <= 0.50

    final_441 = 441 * (1 + cum_ret)

    return {
        'strategy': name,
        'n_trades': n_trades,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'trade_win_rate': round(wr * 100, 1),
        'avg_trade_return': round(avg_ret * 100, 2),
        'profit_factor': round(pf, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'cum_return': round(cum_ret * 100, 2),
        'final_441': round(final_441, 2),
        'R1_regime_gap': round(regime_gap, 3),
        'R1_pass': r1_pass,
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'feasible_441': True,
    }

def main():
    close = download_data()
    spy = close['SPY']

    configs = [
        {'vol_window': 20, 'vol_lookback': 120, 'vol_percentile': 20, 'hold_days': 20, 'max_positions': 2,
         'name': 'VolBreak_20d_p20_hold20'},
        {'vol_window': 20, 'vol_lookback': 120, 'vol_percentile': 10, 'hold_days': 20, 'max_positions': 2,
         'name': 'VolBreak_20d_p10_hold20'},
        {'vol_window': 10, 'vol_lookback': 60, 'vol_percentile': 20, 'hold_days': 10, 'max_positions': 2,
         'name': 'VolBreak_10d_p20_hold10'},
        {'vol_window': 20, 'vol_lookback': 120, 'vol_percentile': 15, 'hold_days': 30, 'max_positions': 1,
         'name': 'VolBreak_20d_p15_hold30_1pos'},
        {'vol_window': 20, 'vol_lookback': 120, 'vol_percentile': 20, 'hold_days': 40, 'max_positions': 2,
         'name': 'VolBreak_20d_p20_hold40'},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        hold = cfg.pop('hold_days')
        max_pos = cfg.pop('max_positions')
        print(f"\nTesting {name}...")

        signals = compute_vol_signals(close, **cfg)
        n_signals = sum(sig['buy'].sum() for sig in signals.values())
        print(f"  Generated {n_signals} buy signals")

        ret, trades = backtest_vol_breakout(close, signals, hold_days=hold, max_positions=max_pos)
        if len(ret) > 0:
            r = analyze(ret, trades, name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, WR={r['trade_win_rate']}%, "
                      f"MaxDD={r['max_drawdown']}%, Trades={r['n_trades']}, R1={'PASS' if r['R1_pass'] else 'FAIL'}")

    with open(f"{OUTPUT_DIR}/strategy4_vol_breakout_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("STRATEGY 4: VOLATILITY BREAKOUT — RESULTS")
    print("="*80)
    if results:
        df = pd.DataFrame(results)
        print(df[['strategy', 'CAGR', 'sharpe', 'sortino', 'n_trades', 'trade_win_rate',
                  'max_drawdown', 'R1_pass', 'final_441']].to_string(index=False))

    return results

if __name__ == '__main__':
    results = main()
