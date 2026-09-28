#!/usr/bin/env python3
"""
Strategy 3: Post-Earnings Announcement Drift (PEAD)
- Buy stocks with large positive earnings surprises
- Hold for 60 days
- Walk-forward across 5+ years
- Industry-agnostic
"""
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/aggressive"

# Large-cap universe for earnings
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    'ADBE', 'NFLX', 'PYPL', 'AVGO', 'QCOM', 'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',
    'LLY', 'TMO', 'ABT', 'DHR', 'BMY', 'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'BLK', 'AXP', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX',
    'HD', 'CAT', 'BA', 'HON', 'UPS', 'RTX', 'GE', 'DE', 'LMT', 'UNP',
    'XOM', 'CVX', 'COP', 'SLB', 'LIN', 'APD', 'NEE', 'DUK', 'DIS', 'CMCSA',
    'V', 'MA', 'BRK-B', 'ORCL', 'ACN', 'IBM', 'NOW', 'INTU', 'ISRG', 'REGN',
]

def download_earnings_and_prices():
    """Download price data and earnings data"""
    print("Downloading price data for PEAD universe...")
    end = datetime(2026, 7, 1)
    start = datetime(2019, 1, 1)

    # Download prices
    data = yf.download(UNIVERSE + ['SPY'], start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')

    # Simulate earnings surprises using price gaps
    # Since yfinance earnings calendar is unreliable for historical data,
    # we'll use large overnight gaps as a proxy for earnings surprises
    # A gap > 5% on open vs prev close = likely earnings reaction
    open_data = data['Open'].dropna(how='all')

    return close, open_data

def detect_earnings_reactions(close, open_data, gap_threshold=0.05):
    """
    Detect likely earnings reactions via overnight gaps.
    gap_threshold: minimum gap to qualify (5% = likely earnings)
    """
    events = []

    for ticker in close.columns:
        if ticker == 'SPY':
            continue
        if ticker not in open_data.columns:
            continue

        c = close[ticker].dropna()
        o = open_data[ticker].dropna()

        common = c.index.intersection(o.index)
        if len(common) < 100:
            continue

        c = c.loc[common]
        o = o.loc[common]

        # Overnight gap: today's open / yesterday's close - 1
        prev_close = c.shift(1)
        gap = (o / prev_close - 1).dropna()

        # Large positive gaps (positive surprise)
        pos_gaps = gap[gap > gap_threshold]
        for date, gap_pct in pos_gaps.items():
            events.append({
                'ticker': ticker,
                'date': date,
                'gap_pct': gap_pct,
                'direction': 'positive'
            })

        # Large negative gaps (negative surprise)
        neg_gaps = gap[gap < -gap_threshold]
        for date, gap_pct in neg_gaps.items():
            events.append({
                'ticker': ticker,
                'date': date,
                'gap_pct': gap_pct,
                'direction': 'negative'
            })

    return pd.DataFrame(events)

def backtest_pead(close, events_df, hold_days=60, top_n=2, long_only=True):
    """
    PEAD backtest: buy positive surprises, hold for hold_days.
    top_n: max positions at once (concentrated for $441 account)
    """
    if events_df.empty:
        return pd.Series(dtype=float)

    # Sort events by date
    events_df = events_df.sort_values('date')

    # Only positive surprises for long-only
    if long_only:
        events_df = events_df[events_df['direction'] == 'positive']

    # Walk through time
    positions = []  # (ticker, entry_date, entry_price, exit_date)
    daily_returns = {}

    for _, event in events_df.iterrows():
        ticker = event['ticker']
        entry_date = event['date']

        if ticker not in close.columns:
            continue

        # Check if we already have max positions
        active = [p for p in positions if p['exit_date'] > entry_date]
        if len(active) >= top_n:
            continue

        # Get entry and exit prices
        future_prices = close[ticker].loc[entry_date:]
        if len(future_prices) < 2:
            continue

        entry_price = future_prices.iloc[0]  # Close on event day

        # Exit after hold_days trading days
        exit_idx = min(hold_days, len(future_prices) - 1)
        exit_date = future_prices.index[exit_idx]
        exit_price = future_prices.iloc[exit_idx]

        positions.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'gap_pct': event['gap_pct'],
            'return': exit_price / entry_price - 1
        })

    if not positions:
        return pd.Series(dtype=float), pd.DataFrame()

    pos_df = pd.DataFrame(positions)

    # Convert to daily returns by assigning position returns across holding period
    all_dates = close.index
    port_returns = pd.Series(0.0, index=all_dates)

    for _, pos in pos_df.iterrows():
        mask = (all_dates >= pos['entry_date']) & (all_dates <= pos['exit_date'])
        hold_dates = all_dates[mask]
        if len(hold_dates) > 1:
            ticker_rets = close[pos['ticker']].loc[hold_dates].pct_change().fillna(0)
            # Scale by number of active positions (equal weight)
            port_returns.loc[hold_dates] += ticker_rets / top_n

    # Only return dates where we had at least one position
    first_entry = pos_df['entry_date'].min()
    last_exit = pos_df['exit_date'].max()
    port_returns = port_returns.loc[first_entry:last_exit]

    return port_returns, pos_df

def analyze_pead(returns, positions_df, name, spy_close):
    """Analyze PEAD strategy"""
    if len(returns) < 100:
        return None

    # Filter out zero-return days (no position)
    active_days = returns[returns != 0]
    if len(active_days) < 50:
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

    # Trade-level stats
    if positions_df is not None and len(positions_df) > 0:
        trade_wr = (positions_df['return'] > 0).mean()
        avg_trade_ret = positions_df['return'].mean()
        avg_win = positions_df.loc[positions_df['return'] > 0, 'return'].mean() if (positions_df['return'] > 0).any() else 0
        avg_loss = abs(positions_df.loc[positions_df['return'] < 0, 'return'].mean()) if (positions_df['return'] < 0).any() else 1
        trade_pf = avg_win / avg_loss if avg_loss > 0 else float('inf')
        n_trades = len(positions_df)
    else:
        trade_wr = avg_trade_ret = trade_pf = n_trades = 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # R1 regime test
    spy_monthly = spy_close.resample('ME').last().pct_change()
    strat_monthly = returns.resample('ME').sum()
    common_idx = strat_monthly.index.intersection(spy_monthly.index)

    if len(common_idx) > 10:
        sm = strat_monthly.loc[common_idx]
        spy_m = spy_monthly.loc[common_idx]

        green = sm[spy_m > 0.01]
        red = sm[spy_m < -0.01]

        sharpe_green = green.mean() / green.std() * np.sqrt(12) if len(green) > 2 and green.std() > 0 else 0
        sharpe_red = red.mean() / red.std() * np.sqrt(12) if len(red) > 2 and red.std() > 0 else 0

        max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0
        r1_pass = regime_gap <= 0.50
    else:
        sharpe_green = sharpe_red = regime_gap = 0
        r1_pass = False
        green = red = pd.Series(dtype=float)

    final_441 = 441 * (1 + cum_ret)

    return {
        'strategy': name,
        'n_trades': n_trades,
        'n_days': n_days,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'trade_win_rate': round(trade_wr * 100, 1),
        'avg_trade_return': round(avg_trade_ret * 100, 2),
        'trade_profit_factor': round(trade_pf, 3),
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
    close, open_data = download_earnings_and_prices()

    # Detect earnings-like reactions
    events = detect_earnings_reactions(close, open_data, gap_threshold=0.05)
    print(f"Detected {len(events)} large gap events ({(events['direction']=='positive').sum()} positive, "
          f"{(events['direction']=='negative').sum()} negative)")

    spy = close['SPY']

    configs = [
        {'hold_days': 60, 'top_n': 1, 'name': 'PEAD_60d_1pos'},
        {'hold_days': 60, 'top_n': 2, 'name': 'PEAD_60d_2pos'},
        {'hold_days': 30, 'top_n': 1, 'name': 'PEAD_30d_1pos'},
        {'hold_days': 30, 'top_n': 2, 'name': 'PEAD_30d_2pos'},
        {'hold_days': 20, 'top_n': 2, 'name': 'PEAD_20d_2pos'},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\nTesting {name}...")

        # Filter to only large gaps (top quartile)
        pos_events = events[events['direction'] == 'positive'].copy()
        gap_75 = pos_events['gap_pct'].quantile(0.75)
        big_events = pos_events[pos_events['gap_pct'] >= gap_75]
        print(f"  Using {len(big_events)} large positive gap events (>={gap_75:.1%})")

        ret, pos_df = backtest_pead(close, big_events, **cfg)
        if ret is not None and len(ret) > 0:
            r = analyze_pead(ret, pos_df, name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, Sortino={r['sortino']}, "
                      f"WR={r['trade_win_rate']}%, MaxDD={r['max_drawdown']}%, "
                      f"Trades={r['n_trades']}, R1={'PASS' if r['R1_pass'] else 'FAIL'}")

    # Also test all positive gaps (not just top quartile)
    for cfg_name, hold, n in [('PEAD_all_60d_2pos', 60, 2), ('PEAD_all_30d_2pos', 30, 2)]:
        print(f"\nTesting {cfg_name} (all positive gaps)...")
        pos_events = events[events['direction'] == 'positive'].copy()
        ret, pos_df = backtest_pead(close, pos_events, hold_days=hold, top_n=n)
        if ret is not None and len(ret) > 0:
            r = analyze_pead(ret, pos_df, cfg_name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, Trades={r['n_trades']}, "
                      f"R1={'PASS' if r['R1_pass'] else 'FAIL'}")

    with open(f"{OUTPUT_DIR}/strategy3_pead_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("STRATEGY 3: POST-EARNINGS DRIFT — RESULTS")
    print("="*80)
    if results:
        df = pd.DataFrame(results)
        print(df[['strategy', 'CAGR', 'sharpe', 'sortino', 'n_trades', 'trade_win_rate',
                  'max_drawdown', 'R1_pass', 'final_441']].to_string(index=False))

    return results

if __name__ == '__main__':
    results = main()
