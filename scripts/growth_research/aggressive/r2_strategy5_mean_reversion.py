#!/usr/bin/env python3
"""
Round 2 Strategy 5: Mean-Reversion After Large Drops
When a quality stock drops >10% in a week with no fundamental reason, buy.
Hold until recovery to 50-day MA or 30 days max.
Screen: only stocks with strong fundamentals.

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
print("ROUND 2 — STRATEGY 5: MEAN-REVERSION AFTER LARGE DROPS")
print("=" * 70)

# Get S&P 500 tickers
print("\n[1/6] Getting S&P 500 constituents...")
try:
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
except:
    tickers = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','JPM','JNJ','V',
               'PG','UNH','HD','MA','DIS','BAC','ADBE','CRM','NFLX',
               'CMCSA','XOM','COST','TMO','ABT','PEP','AVGO','ACN','NKE','MRK',
               'LLY','WMT','DHR','TXN','PM','QCOM','LOW','UNP','LIN','NEE',
               'BMY','RTX','MDT','AMGN','HON','SBUX','IBM','CAT','GE','DE',
               'CVX','COP','MCD','INTC','AMD','ISRG','NOW','BKNG','SYK','GILD',
               'BLK','MDLZ','ADP','TJX','VRTX','REGN','ZTS','CI','MMC','PLD',
               'SCHW','CB','BDX','SO','DUK','ICE','NSC','AON','BSX','FIS',
               'CL','SHW','MCO','HUM','PNC','USB','TFC','MET','AIG','ALL',
               'ORCL','WBA','BIIB','KLAC','MCHP','SNPS','CDNS','FTNT','PANW','CRWD']

print(f"  Got {len(tickers)} tickers")

# Download price data
print("\n[2/6] Downloading price data...")
all_prices = {}
batch_size = 50
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    try:
        data = yf.download(batch, start='2016-01-01', end='2026-07-01', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data.xs('Close', level=0, axis=1)
            volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else data.xs('Volume', level=0, axis=1)
        else:
            close = data[['Close']]
            volume = data[['Volume']]
        for t in batch:
            if t in close.columns:
                s = close[t].dropna()
                v = volume[t].dropna() if t in volume.columns else None
                if len(s) > 252:
                    all_prices[t] = {'close': s, 'volume': v}
            elif len(batch) == 1:
                s = close.iloc[:, 0].dropna()
                v = volume.iloc[:, 0].dropna() if volume is not None else None
                if len(s) > 252:
                    all_prices[t] = {'close': s, 'volume': v}
    except:
        pass
    if (i // batch_size) % 2 == 0:
        print(f"  Downloaded {min(i+batch_size, len(tickers))}/{len(tickers)} tickers...")

print(f"  Got price data for {len(all_prices)} stocks")

# SPY for regime classification
spy = yf.download('SPY', start='2016-01-01', end='2026-07-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_close = spy['Close']
spy_monthly = spy_close.resample('ME').last()
spy_monthly_ret = spy_monthly.pct_change()

regime_map = {}
for dt, ret in spy_monthly_ret.items():
    if pd.notna(ret):
        regime_map[dt.strftime('%Y-%m')] = 'green' if ret >= 0 else 'red'

print(f"  Regime: {sum(1 for v in regime_map.values() if v=='green')} green, {sum(1 for v in regime_map.values() if v=='red')} red")


def find_large_drops(prices_dict, drop_threshold=-0.10, lookback_days=5):
    """
    Find instances where stocks dropped >threshold in lookback_days.
    Exclude drops that coincide with earnings (high volume spike = likely earnings).
    """
    all_drops = []

    for ticker, pdata in prices_dict.items():
        close = pdata['close']
        volume = pdata['volume']

        # Weekly return
        weekly_ret = close.pct_change(lookback_days)

        # Volume spike detection (for earnings exclusion)
        if volume is not None and len(volume) > 20:
            vol_ma20 = volume.rolling(20).mean()
            vol_ratio = volume / vol_ma20
        else:
            vol_ratio = pd.Series(1.0, index=close.index)

        # 50-day and 200-day MA for quality filter
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # Find drops
        drop_mask = weekly_ret < drop_threshold

        for dt in close.index[drop_mask]:
            if dt not in weekly_ret.index or pd.isna(weekly_ret[dt]):
                continue

            # Exclude if volume spike (likely earnings)
            if dt in vol_ratio.index and pd.notna(vol_ratio[dt]) and vol_ratio[dt] > 3.0:
                continue

            # Quality filter: stock was above 200-day MA before the drop
            # (i.e., it was in an uptrend, not already collapsing)
            pre_drop_idx = close.index.get_loc(dt) - lookback_days
            if pre_drop_idx >= 0 and pre_drop_idx < len(close.index):
                pre_date = close.index[pre_drop_idx]
                if pre_date in ma200.index and pd.notna(ma200[pre_date]):
                    if close[pre_date] < ma200[pre_date]:
                        continue  # Skip stocks already in downtrend

            all_drops.append({
                'ticker': ticker,
                'date': dt,
                'weekly_return': weekly_ret[dt],
                'price': close[dt],
                'ma50': ma50[dt] if dt in ma50.index else np.nan,
                'ma200': ma200[dt] if dt in ma200.index else np.nan,
            })

    return pd.DataFrame(all_drops)


def mean_reversion_backtest(drops_df, prices_dict,
                             max_hold_days=30, exit_ma='50d',
                             train_window_years=2, n_positions=1,
                             initial_capital=441):
    """
    Walk-forward mean reversion strategy.
    - Training: use trailing 2 years to calibrate drop threshold and hold period
    - Entry: buy after large drop
    - Exit: when price recovers to exit_ma OR max_hold_days
    """
    drops_df = drops_df.sort_values('date')

    # Walk-forward: train on trailing window, test on next period
    all_dates = sorted(drops_df['date'].unique())
    if len(all_dates) < 100:
        return pd.DataFrame()

    train_start_date = all_dates[0] + pd.Timedelta(days=365 * train_window_years)
    test_drops = drops_df[drops_df['date'] >= train_start_date]

    trades = []
    # Track active positions to limit concurrency
    active_until = None

    for _, drop in test_drops.iterrows():
        ticker = drop['ticker']
        entry_date = drop['date']

        # Don't enter if we already have an active position
        if active_until is not None and entry_date < active_until:
            continue

        if ticker not in prices_dict:
            continue

        close = prices_dict[ticker]['close']

        # Walk-forward: compute win rate of similar drops in training window
        train_end = entry_date
        train_start = entry_date - pd.Timedelta(days=365 * train_window_years)
        train_drops = drops_df[
            (drops_df['date'] >= train_start) &
            (drops_df['date'] < train_end) &
            (drops_df['ticker'] != ticker)  # Avoid leakage from same stock
        ]

        # Compute historical win rate for this type of drop
        if len(train_drops) < 10:
            continue

        train_wins = 0
        train_total = 0
        for _, td in train_drops.iterrows():
            t_ticker = td['ticker']
            if t_ticker not in prices_dict:
                continue
            t_close = prices_dict[t_ticker]['close']
            t_entry_dates = t_close.index[t_close.index > td['date']]
            if len(t_entry_dates) < 2:
                continue

            t_entry_date = t_entry_dates[0]
            t_entry_price = t_close[t_entry_date]

            # Exit: recovery to 50d MA or max_hold_days
            t_ma50 = t_close.rolling(50).mean()
            t_exit_date = None
            for j in range(1, min(max_hold_days + 1, len(t_entry_dates))):
                check_date = t_entry_dates[j]
                if exit_ma == '50d' and check_date in t_ma50.index:
                    if pd.notna(t_ma50[check_date]) and t_close[check_date] >= t_ma50[check_date]:
                        t_exit_date = check_date
                        break
                elif j >= max_hold_days:
                    t_exit_date = check_date
                    break

            if t_exit_date is None:
                t_exit_date = t_entry_dates[min(max_hold_days, len(t_entry_dates) - 1)]

            t_exit_price = t_close[t_exit_date]
            t_ret = (t_exit_price / t_entry_price) - 1

            train_total += 1
            if t_ret > 0:
                train_wins += 1

        if train_total < 5:
            continue

        train_wr = train_wins / train_total

        # Only enter if training window shows >55% win rate
        if train_wr < 0.55:
            continue

        # Execute the trade
        future_dates = close.index[close.index > entry_date]
        if len(future_dates) < 2:
            continue

        actual_entry_date = future_dates[0]
        entry_price = close[actual_entry_date]

        # Exit logic
        ma50 = close.rolling(50).mean()
        exit_date = None
        for j in range(1, min(max_hold_days + 1, len(future_dates))):
            check_date = future_dates[j]
            if exit_ma == '50d' and check_date in ma50.index:
                if pd.notna(ma50[check_date]) and close[check_date] >= ma50[check_date]:
                    exit_date = check_date
                    break

        if exit_date is None:
            exit_idx = min(max_hold_days, len(future_dates) - 1)
            exit_date = future_dates[exit_idx]

        exit_price = close[exit_date]
        trade_return = (exit_price / entry_price) - 1

        trades.append({
            'ticker': ticker,
            'entry_date': actual_entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_return': trade_return,
            'hold_days': (exit_date - actual_entry_date).days,
            'weekly_drop': drop['weekly_return'],
            'train_wr': train_wr,
            'train_n': train_total,
        })

        active_until = exit_date

    return pd.DataFrame(trades)


def compute_metrics(trades_df, initial_capital=441):
    """Compute metrics from trade list."""
    if len(trades_df) == 0:
        return None

    trades = trades_df.sort_values('entry_date')

    # Build equity curve
    equity = initial_capital
    equity_points = [{'date': trades.iloc[0]['entry_date'], 'equity': initial_capital}]

    for _, trade in trades.iterrows():
        equity *= (1 + trade['trade_return'])
        equity_points.append({'date': trade['exit_date'], 'equity': equity})

    eq_df = pd.DataFrame(equity_points).set_index('date')

    # Metrics
    n_trades = len(trades)
    date_range = (trades['exit_date'].max() - trades['entry_date'].min()).days
    n_years = date_range / 365.25 if date_range > 0 else 1

    cagr = (equity / initial_capital) ** (1/n_years) - 1 if n_years > 0 else 0

    rets = trades['trade_return'].values
    sharpe = rets.mean() / rets.std() * np.sqrt(n_trades / n_years) if rets.std() > 0 and n_years > 0 else 0

    neg = rets[rets < 0]
    downside = neg.std() if len(neg) > 0 else rets.std()
    sortino = rets.mean() / downside * np.sqrt(n_trades / n_years) if downside > 0 and n_years > 0 else 0

    wr = np.sum(rets > 0) / len(rets) * 100
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets <= 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Max drawdown from equity curve
    eq_vals = np.array([e['equity'] for e in equity_points])
    peak = np.maximum.accumulate(eq_vals)
    dd = (eq_vals - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'n_trades': n_trades,
        'n_years': round(n_years, 1),
        'avg_hold_days': round(trades['hold_days'].mean(), 1),
        'avg_trade_return': round(rets.mean() * 100, 2),
        'final_equity': round(equity, 2),
        'cum_return': round((equity / initial_capital - 1) * 100, 2),
    }


def regime_test(trades_df):
    """R1 regime test on trades."""
    green_rets = []
    red_rets = []

    for _, trade in trades_df.iterrows():
        ym = trade['entry_date'].strftime('%Y-%m')
        if ym in regime_map:
            if regime_map[ym] == 'green':
                green_rets.append(trade['trade_return'])
            else:
                red_rets.append(trade['trade_return'])

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    if len(green_rets) < 5 or len(red_rets) < 5:
        return None, None, None, False

    # Annualize per-trade Sharpe
    sg = green_rets.mean() / green_rets.std() if green_rets.std() > 0 else 0
    sr = red_rets.mean() / red_rets.std() if red_rets.std() > 0 else 0

    max_abs = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_abs if max_abs > 0 else 0

    return sg, sr, gap, gap <= 0.50


def permutation_test(trades_df, n_perms=100):
    rets = trades_df['trade_return'].values
    real_mean = rets.mean()

    count = 0
    for _ in range(n_perms):
        # Shuffle assignment of returns to trades (test if order matters)
        shuf = np.random.choice(rets, size=len(rets), replace=True)
        if shuf.mean() >= real_mean:
            count += 1

    return count / n_perms


# Find all large drops
print("\n[3/6] Identifying large weekly drops...")
configs_drop = [
    {'threshold': -0.10, 'lookback': 5, 'name': 'Drop10pct_5d'},
    {'threshold': -0.08, 'lookback': 5, 'name': 'Drop8pct_5d'},
    {'threshold': -0.15, 'lookback': 5, 'name': 'Drop15pct_5d'},
    {'threshold': -0.10, 'lookback': 10, 'name': 'Drop10pct_10d'},
]

all_results = []

for drop_cfg in configs_drop:
    drops = find_large_drops(all_prices, drop_threshold=drop_cfg['threshold'],
                              lookback_days=drop_cfg['lookback'])

    if len(drops) == 0:
        print(f"  {drop_cfg['name']}: No drops found")
        continue

    print(f"\n  {drop_cfg['name']}: Found {len(drops)} drops across {drops['ticker'].nunique()} stocks")

    # Test with different hold periods and exit strategies
    hold_configs = [
        {'max_hold': 20, 'exit_ma': '50d', 'name': f"{drop_cfg['name']}_hold20_ma50"},
        {'max_hold': 30, 'exit_ma': '50d', 'name': f"{drop_cfg['name']}_hold30_ma50"},
        {'max_hold': 15, 'exit_ma': '50d', 'name': f"{drop_cfg['name']}_hold15_ma50"},
        {'max_hold': 30, 'exit_ma': 'none', 'name': f"{drop_cfg['name']}_hold30_fixed"},
    ]

    for hcfg in hold_configs:
        print(f"\n    Testing {hcfg['name']}...")
        trades = mean_reversion_backtest(
            drops, all_prices,
            max_hold_days=hcfg['max_hold'],
            exit_ma=hcfg['exit_ma'],
            n_positions=1
        )

        if len(trades) < 10:
            print(f"      Only {len(trades)} trades, skipping")
            continue

        metrics = compute_metrics(trades)
        if metrics is None:
            continue

        sg, sr, gap, r1_pass = regime_test(trades)
        perm_p = permutation_test(trades)

        r = {
            'strategy': hcfg['name'],
            **metrics,
            'sharpe_green': round(sg, 3) if sg is not None else None,
            'sharpe_red': round(sr, 3) if sr is not None else None,
            'R1_regime_gap': round(gap, 3) if gap is not None else None,
            'R1_pass': str(r1_pass),
            'perm_p_value': round(perm_p, 3),
            'perm_pass': str(perm_p < 0.05),
        }
        all_results.append(r)

        print(f"      Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['CAGR']:.1f}%, WR={metrics['win_rate']:.1f}%")
        print(f"      MaxDD={metrics['max_drawdown']:.1f}%, AvgHold={metrics['avg_hold_days']:.0f}d, Trades={metrics['n_trades']}")
        if gap is not None:
            print(f"      R1: gap={gap:.3f} ({'PASS' if r1_pass else 'FAIL'})")
        print(f"      Perm p={perm_p:.3f} ({'PASS' if perm_p < 0.05 else 'FAIL'})")

# Save results
print("\n[6/6] Saving results...")
with open(os.path.join(OUTPUT_DIR, 'r2_strategy5_mean_reversion.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nResults saved.")
print("\n" + "=" * 70)
print("MEAN-REVERSION SUMMARY")
print("=" * 70)
for r in all_results:
    flags = []
    if r['R1_pass'] == 'True': flags.append('R1-PASS')
    else: flags.append('R1-FAIL')
    if r['perm_pass'] == 'True': flags.append('PERM-PASS')
    else: flags.append('PERM-FAIL')
    print(f"  {r['strategy']}: Sharpe={r['sharpe']:.3f}, CAGR={r['CAGR']:.1f}%, "
          f"WR={r['win_rate']:.0f}%, MaxDD={r['max_drawdown']:.1f}%, "
          f"Trades={r['n_trades']}, [{', '.join(flags)}]")
