#!/usr/bin/env python3
"""
Round 2 Strategy 1: Post-Earnings Drift (PEAD)
Academic edge: stocks drift in surprise direction for 60 days after earnings.
With $441: buy 1-2 stocks per earnings season.

Walk-forward sliding window backtest.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import warnings
import os
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/aggressive/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ROUND 2 — STRATEGY 1: POST-EARNINGS DRIFT (PEAD)")
print("=" * 70)

# Get S&P 500 tickers
print("\n[1/6] Downloading S&P 500 constituent list...")
sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
try:
    sp500_table = pd.read_html(sp500_url)[0]
    tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
except:
    # Fallback to a representative sample
    tickers = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','JPM','JNJ','V',
               'PG','UNH','HD','MA','DIS','PYPL','BAC','ADBE','CRM','NFLX',
               'CMCSA','XOM','COST','TMO','ABT','PEP','AVGO','ACN','NKE','MRK',
               'LLY','WMT','DHR','TXN','PM','QCOM','LOW','UNP','LIN','NEE',
               'BMY','RTX','MDT','AMGN','HON','SBUX','IBM','CAT','GE','DE',
               'CVX','COP','MCD','INTC','AMD','ISRG','NOW','BKNG','SYK','GILD',
               'BLK','MDLZ','ADP','TJX','VRTX','REGN','ZTS','CI','MMC','PLD',
               'SCHW','CB','BDX','SO','DUK','ICE','NSC','AON','BSX','FIS',
               'CL','SHW','MCO','HUM','PNC','USB','TFC','MET','AIG','ALL']

print(f"  Got {len(tickers)} tickers")

# Download SPY as market benchmark for regime classification
print("\n[2/6] Downloading SPY for regime classification...")
spy = yf.download('SPY', start='2016-01-01', end='2026-07-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_monthly = spy['Close'].resample('ME').last()
spy_monthly_ret = spy_monthly.pct_change()

# Classify months as green (>=0) or red (<0)
regime_map = {}
for dt, ret in spy_monthly_ret.items():
    if pd.notna(ret):
        regime_map[dt.strftime('%Y-%m')] = 'green' if ret >= 0 else 'red'

print(f"  SPY regime map: {sum(1 for v in regime_map.values() if v=='green')} green, {sum(1 for v in regime_map.values() if v=='red')} red months")

# Download earnings data
print("\n[3/6] Downloading earnings surprise data (this takes a while)...")
# We'll use yfinance earnings data - get actual vs expected EPS
# For a practical backtest, we'll simulate earnings surprises from price reactions

# Strategy: use price reaction on earnings day as proxy for surprise
# (In practice, this is what matters - the market's reaction)

# Download price data for all tickers
all_prices = {}
batch_size = 50
failed = 0
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    batch_str = ' '.join(batch)
    try:
        data = yf.download(batch_str, start='2016-01-01', end='2026-07-01', progress=False, group_by='ticker')
        for t in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[t]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df.dropna()) > 252:
                    all_prices[t] = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            except:
                failed += 1
    except:
        failed += 1
    if (i // batch_size) % 2 == 0:
        print(f"  Downloaded {min(i+batch_size, len(tickers))}/{len(tickers)} tickers...")

print(f"  Got price data for {len(all_prices)} stocks ({failed} failed)")

# Identify earnings events: large volume spikes + price gaps as proxy
print("\n[4/6] Identifying earnings events and computing surprise scores...")

def find_earnings_events(prices_df, ticker):
    """
    Identify likely earnings dates by volume spikes + absolute return.
    Returns list of (date, surprise_score, return_1d)
    """
    df = prices_df.copy()
    df['ret'] = df['Close'].pct_change()
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma20']
    df['abs_ret'] = df['ret'].abs()

    # Earnings proxy: days with volume > 2x average AND |return| > 2%
    # This catches most actual earnings days
    earnings_mask = (df['vol_ratio'] > 2.0) & (df['abs_ret'] > 0.02)

    # Cluster nearby days (within 3 days) and keep the highest volume one
    events = []
    df_events = df[earnings_mask].copy()

    if len(df_events) == 0:
        return events

    # Remove events too close together (within 30 days)
    last_event_date = None
    for idx, row in df_events.iterrows():
        if last_event_date is not None and (idx - last_event_date).days < 30:
            continue
        surprise_score = row['ret'] * row['vol_ratio']  # Signed surprise
        events.append({
            'date': idx,
            'surprise_score': surprise_score,
            'day_return': row['ret'],
            'vol_ratio': row['vol_ratio'],
            'ticker': ticker
        })
        last_event_date = idx

    return events

all_events = []
for ticker, prices in all_prices.items():
    events = find_earnings_events(prices, ticker)
    all_events.extend(events)

events_df = pd.DataFrame(all_events)
if len(events_df) == 0:
    print("  ERROR: No earnings events found!")
    exit(1)

events_df = events_df.sort_values('date')
print(f"  Found {len(events_df)} earnings-like events across {events_df['ticker'].nunique()} stocks")
print(f"  Date range: {events_df['date'].min().strftime('%Y-%m-%d')} to {events_df['date'].max().strftime('%Y-%m-%d')}")

# Walk-forward PEAD backtest
print("\n[5/6] Running walk-forward PEAD backtest...")

def run_pead_backtest(events_df, all_prices, hold_days=60, n_positions=2,
                      train_years=2, surprise_threshold_pct=80, long_only=True):
    """
    Walk-forward PEAD strategy.
    - Training window: trailing train_years of earnings events
    - Each quarter: rank earnings surprises, buy top N positive surprises
    - Hold for hold_days
    """
    # Group events by quarter
    events_df = events_df.copy()
    events_df['quarter'] = events_df['date'].dt.to_period('Q')

    quarters = sorted(events_df['quarter'].unique())

    # Need at least train_years*4 quarters for training
    min_train_quarters = train_years * 4

    trades = []

    for q_idx in range(min_train_quarters, len(quarters)):
        test_quarter = quarters[q_idx]

        # Training data: last train_years*4 quarters
        train_quarters = quarters[q_idx - min_train_quarters:q_idx]

        # Get training events to calibrate what constitutes a "big" surprise
        train_events = events_df[events_df['quarter'].isin(train_quarters)]

        if len(train_events) < 20:
            continue

        # Calibrate: what surprise threshold historically led to drift?
        # Compute post-event returns for training period
        train_drifts = []
        for _, ev in train_events.iterrows():
            ticker = ev['ticker']
            ev_date = ev['date']
            if ticker not in all_prices:
                continue
            p = all_prices[ticker]

            # Find entry price (next day open)
            future_dates = p.index[p.index > ev_date]
            if len(future_dates) < 2:
                continue
            entry_date = future_dates[0]
            entry_price = p.loc[entry_date, 'Open']

            # Find exit price (hold_days later)
            exit_dates = future_dates[future_dates >= entry_date + pd.Timedelta(days=hold_days)]
            if len(exit_dates) == 0:
                exit_dates = future_dates[-1:]
            exit_date = exit_dates[0]
            exit_price = p.loc[exit_date, 'Close']

            drift_return = (exit_price / entry_price) - 1
            train_drifts.append({
                'surprise': ev['surprise_score'],
                'day_return': ev['day_return'],
                'drift': drift_return
            })

        if len(train_drifts) < 10:
            continue

        train_drift_df = pd.DataFrame(train_drifts)

        # Find optimal surprise threshold from training data
        # We want surprises where drift is consistently positive (for longs)
        pos_surprises = train_drift_df[train_drift_df['surprise'] > 0]
        if len(pos_surprises) < 5:
            continue

        # Use top quintile of positive surprises
        surprise_threshold = pos_surprises['surprise'].quantile(0.60)

        # Now apply to test quarter
        test_events = events_df[events_df['quarter'] == test_quarter]

        if long_only:
            candidates = test_events[test_events['surprise_score'] > surprise_threshold]
        else:
            # Also short the big negative surprises
            pos_candidates = test_events[test_events['surprise_score'] > surprise_threshold]
            neg_candidates = test_events[test_events['surprise_score'] < -surprise_threshold]
            candidates = pos_candidates  # For now, long only

        if len(candidates) == 0:
            continue

        # Rank by absolute surprise score, take top N
        candidates = candidates.sort_values('surprise_score', ascending=False).head(n_positions)

        for _, ev in candidates.iterrows():
            ticker = ev['ticker']
            ev_date = ev['date']
            if ticker not in all_prices:
                continue
            p = all_prices[ticker]

            future_dates = p.index[p.index > ev_date]
            if len(future_dates) < 2:
                continue
            entry_date = future_dates[0]
            entry_price = p.loc[entry_date, 'Open']

            # Exit after hold_days
            exit_candidates = future_dates[future_dates >= entry_date + pd.Timedelta(days=hold_days)]
            if len(exit_candidates) == 0:
                exit_candidates = future_dates[-1:]
            exit_date = exit_candidates[0]
            exit_price = p.loc[exit_date, 'Close']

            trade_return = (exit_price / entry_price) - 1

            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'ticker': ticker,
                'surprise': ev['surprise_score'],
                'day_return': ev['day_return'],
                'trade_return': trade_return,
                'hold_days': (exit_date - entry_date).days,
                'quarter': str(test_quarter)
            })

    return pd.DataFrame(trades)


def compute_equity_curve(trades_df, initial_capital=441):
    """Convert trades to daily equity curve assuming equal weight, sequential positions."""
    if len(trades_df) == 0:
        return pd.Series(dtype=float), {}

    trades_df = trades_df.sort_values('entry_date')

    # Build daily equity curve
    all_dates = pd.date_range(trades_df['entry_date'].min(), trades_df['exit_date'].max(), freq='B')
    equity = pd.Series(0.0, index=all_dates)

    capital = initial_capital
    daily_returns = []

    # Simple approach: each trade uses equal fraction of capital at entry
    # Since positions are few, treat them sequentially
    position_log = []

    for _, trade in trades_df.iterrows():
        position_log.append({
            'entry': trade['entry_date'],
            'exit': trade['exit_date'],
            'return': trade['trade_return']
        })

    # Convert to daily equity
    equity_val = initial_capital
    daily_rets = pd.Series(0.0, index=all_dates)

    for dt in all_dates:
        active_positions = [p for p in position_log if p['entry'] <= dt <= p['exit']]
        if active_positions:
            # Daily return = weighted avg of position daily returns
            # Approximate: spread trade return evenly across hold period
            day_ret = 0
            for p in active_positions:
                hold_bdays = len(pd.date_range(p['entry'], p['exit'], freq='B'))
                if hold_bdays > 0:
                    daily_trade_ret = (1 + p['return']) ** (1/hold_bdays) - 1
                    day_ret += daily_trade_ret / max(len(active_positions), 1)
            daily_rets[dt] = day_ret

    equity_curve = initial_capital * (1 + daily_rets).cumprod()
    return equity_curve, daily_rets


def compute_metrics(daily_returns, equity_curve, trades_df, initial_capital=441):
    """Compute all required performance metrics."""
    if len(daily_returns) == 0 or daily_returns.std() == 0:
        return None

    dr = daily_returns[daily_returns != 0]  # Only days with positions
    if len(dr) == 0:
        dr = daily_returns

    ann_factor = np.sqrt(252)
    sharpe = dr.mean() / dr.std() * ann_factor if dr.std() > 0 else 0

    neg_rets = dr[dr < 0]
    downside_std = neg_rets.std() if len(neg_rets) > 0 else dr.std()
    sortino = dr.mean() / downside_std * ann_factor if downside_std > 0 else 0

    # CAGR
    n_years = len(daily_returns) / 252
    if n_years > 0 and equity_curve.iloc[-1] > 0:
        cagr = (equity_curve.iloc[-1] / initial_capital) ** (1/n_years) - 1
    else:
        cagr = 0

    # Max drawdown
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trade stats
    if len(trades_df) > 0:
        wins = trades_df[trades_df['trade_return'] > 0]
        losses = trades_df[trades_df['trade_return'] <= 0]
        wr = len(wins) / len(trades_df) * 100

        gross_profit = wins['trade_return'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['trade_return'].sum()) if len(losses) > 0 else 0.001
        pf = gross_profit / gross_loss
    else:
        wr, pf = 0, 0

    return {
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'n_trades': len(trades_df),
        'n_years': round(n_years, 1),
        'final_equity': round(equity_curve.iloc[-1], 2),
        'cum_return': round((equity_curve.iloc[-1] / initial_capital - 1) * 100, 2),
    }


def regime_test(daily_returns, spy_data):
    """R1: regime-agnostic test."""
    dr = daily_returns.copy()

    # Classify each day by its month's regime
    green_rets = []
    red_rets = []

    for dt, ret in dr.items():
        ym = dt.strftime('%Y-%m')
        if ym in regime_map:
            if regime_map[ym] == 'green':
                green_rets.append(ret)
            else:
                red_rets.append(ret)

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    if len(green_rets) < 10 or len(red_rets) < 10:
        return None, None, None, False

    sharpe_green = green_rets.mean() / green_rets.std() * np.sqrt(252) if green_rets.std() > 0 else 0
    sharpe_red = red_rets.mean() / red_rets.std() * np.sqrt(252) if red_rets.std() > 0 else 0

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    if max_abs > 0:
        regime_gap = abs(sharpe_green - sharpe_red) / max_abs
    else:
        regime_gap = 0

    r1_pass = regime_gap <= 0.50

    return sharpe_green, sharpe_red, regime_gap, r1_pass


def permutation_test(daily_returns, n_perms=100):
    """Permutation test: shuffle returns, compute Sharpe, check if real > random."""
    real_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    count_better = 0
    rets = daily_returns.values

    for _ in range(n_perms):
        shuffled = np.random.permutation(rets)
        shuf_sharpe = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        if shuf_sharpe >= real_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return p_value


# Run multiple PEAD configurations
configs = [
    {'hold_days': 60, 'n_positions': 1, 'name': 'PEAD_60d_1pos'},
    {'hold_days': 60, 'n_positions': 2, 'name': 'PEAD_60d_2pos'},
    {'hold_days': 30, 'n_positions': 2, 'name': 'PEAD_30d_2pos'},
    {'hold_days': 45, 'n_positions': 2, 'name': 'PEAD_45d_2pos'},
    {'hold_days': 60, 'n_positions': 3, 'name': 'PEAD_60d_3pos'},
]

results = []
for cfg in configs:
    print(f"\n  Testing {cfg['name']}...")
    trades = run_pead_backtest(events_df, all_prices,
                                hold_days=cfg['hold_days'],
                                n_positions=cfg['n_positions'])

    if len(trades) == 0:
        print(f"    No trades generated!")
        continue

    print(f"    {len(trades)} trades, {trades['entry_date'].min().strftime('%Y-%m-%d')} to {trades['entry_date'].max().strftime('%Y-%m-%d')}")

    equity, daily_rets = compute_equity_curve(trades)

    if len(equity) == 0:
        continue

    metrics = compute_metrics(daily_rets, equity, trades)
    if metrics is None:
        continue

    # Regime test
    sg, sr, rg, rp = regime_test(daily_rets, spy)

    # Permutation test
    nonzero_rets = daily_rets[daily_rets != 0]
    if len(nonzero_rets) > 20:
        perm_p = permutation_test(nonzero_rets)
    else:
        perm_p = 1.0

    result = {
        'strategy': cfg['name'],
        **metrics,
        'sharpe_green': round(sg, 3) if sg is not None else None,
        'sharpe_red': round(sr, 3) if sr is not None else None,
        'R1_regime_gap': round(rg, 3) if rg is not None else None,
        'R1_pass': str(rp),
        'perm_p_value': round(perm_p, 3),
        'perm_pass': str(perm_p < 0.05),
    }
    results.append(result)

    print(f"    Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['CAGR']:.1f}%, WR={metrics['win_rate']:.1f}%")
    print(f"    MaxDD={metrics['max_drawdown']:.1f}%, Calmar={metrics['calmar']:.3f}")
    print(f"    R1 regime gap={rg:.3f} ({'PASS' if rp else 'FAIL'}), Perm p={perm_p:.3f} ({'PASS' if perm_p < 0.05 else 'FAIL'})")

# Save results
print("\n[6/6] Saving results...")
with open(os.path.join(OUTPUT_DIR, 'r2_strategy1_pead.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}r2_strategy1_pead.json")
print("\n" + "=" * 70)
print("PEAD SUMMARY")
print("=" * 70)
for r in results:
    flags = []
    if r['R1_pass'] == 'True': flags.append('R1-PASS')
    else: flags.append('R1-FAIL')
    if r['perm_pass'] == 'True': flags.append('PERM-PASS')
    else: flags.append('PERM-FAIL')
    print(f"  {r['strategy']}: Sharpe={r['sharpe']:.3f}, CAGR={r['CAGR']:.1f}%, "
          f"WR={r['win_rate']:.0f}%, MaxDD={r['max_drawdown']:.1f}%, "
          f"[{', '.join(flags)}]")
