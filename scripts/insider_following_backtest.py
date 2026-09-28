#!/usr/bin/env python3
"""
Insider Trading Following Strategy Backtest
HC #763 — New strategy development

Strategy: Buy stocks when corporate insiders (CEO, CFO, directors) make open-market
purchases. Hold for 30-60 days. Academic literature shows 3-8% outperformance.

Signal filters:
  - Open market purchases only (not grants, exercises, gifts)
  - C-suite and directors (Position contains Officer, Director, CEO, CFO, etc.)
  - Large purchases: Value > $50K (relaxed from $100K to get more signals)
  - Cluster bonus: multiple insiders buying within 14 days

Regime hedge: half-size when SPY < 200-SMA

Cost model: $0 commission (RH shares), 0.02% slippage each way

OOT: All available data from yfinance insider transactions (~2 years)
Universe: S&P 500 stocks

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (100 iterations)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Configuration ──
HOLD_DAYS = 45  # Middle of 30-60 range
MIN_PURCHASE_VALUE = 50_000  # $50K minimum purchase
CLUSTER_WINDOW_DAYS = 14
CLUSTER_MIN_INSIDERS = 2
SLIPPAGE_PCT = 0.0002  # 0.02% each way (0.04% RT)
POSITION_SIZE = 0.02  # 2% of portfolio per trade
REGIME_HEDGE_FACTOR = 0.5  # Half-size when SPY < 200-SMA
INITIAL_CAPITAL = 1_000_000
N_PERMUTATIONS = 200  # For permutation test

# S&P 500 tickers (representative subset — full 500 would take too long)
# Use ~200 of the most liquid / likely to have insider activity
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers():
    """Get S&P 500 tickers. Falls back to a hardcoded list."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
    except Exception:
        # Fallback: large-cap tickers known to have insider activity
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM',
            'PFE', 'CSCO', 'VZ', 'ADBE', 'CRM', 'CMCSA', 'NFLX', 'INTC',
            'ABT', 'KO', 'PEP', 'TMO', 'NKE', 'MRK', 'WMT', 'CVX', 'LLY',
            'MDT', 'DHR', 'AVGO', 'TXN', 'NEE', 'QCOM', 'LOW', 'BMY', 'UNP',
            'PM', 'RTX', 'HON', 'IBM', 'AMGN', 'CAT', 'GE', 'BA', 'MMM',
            'GS', 'AXP', 'BLK', 'SCHW', 'MS', 'USB', 'PNC', 'TFC', 'COF',
            'C', 'WFC', 'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'CB', 'AFL',
            'GM', 'F', 'TM', 'RIVN', 'DAL', 'UAL', 'AAL', 'LUV', 'FDX',
            'UPS', 'DE', 'EMR', 'ITW', 'ETN', 'PH', 'ROK', 'CMI', 'DOV',
            'ORCL', 'ACN', 'INTU', 'NOW', 'SNOW', 'PLTR', 'PANW', 'CRWD',
            'ZS', 'DDOG', 'MDB', 'NET', 'FTNT', 'WDAY', 'TEAM', 'VEEV',
            'LMT', 'NOC', 'GD', 'HII', 'LHX', 'TDG',
            'SHW', 'ECL', 'APD', 'LIN', 'DD', 'DOW', 'PPG', 'FCX',
            'CL', 'EL', 'CLX', 'CHD', 'SJM', 'K', 'GIS', 'HSY', 'MKC',
            'AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'SPG', 'O', 'WELL',
            'ISRG', 'SYK', 'EW', 'BSX', 'ZBH', 'BDX', 'BAX', 'GEHC',
            'SO', 'DUK', 'AEP', 'EXC', 'SRE', 'D', 'ED', 'XEL', 'WEC',
            'T', 'TMUS', 'CHTR',
            'CI', 'ELV', 'HUM', 'CNC', 'MOH',
            'WM', 'RSG', 'WCN',
            'COST', 'TGT', 'DG', 'DLTR', 'ROST', 'TJX',
            'ADP', 'PAYX', 'CTAS', 'RHI',
            'ICE', 'CME', 'NDAQ', 'SPGI', 'MCO', 'MSCI',
            'SQ', 'PYPL', 'FIS', 'FISV', 'GPN',
            'ABNB', 'BKNG', 'MAR', 'HLT', 'H',
            'OXY', 'COP', 'EOG', 'SLB', 'HAL', 'DVN', 'MPC', 'VLO', 'PSX',
        ]


def fetch_insider_data(ticker, retries=2):
    """Fetch insider transaction data for a single ticker."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            df = t.insider_transactions
            if df is not None and len(df) > 0:
                df = df.copy()
                df['ticker'] = ticker
                return df
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5)
            continue
    return None


def extract_purchases(all_insider_data):
    """Filter for open-market purchases only."""
    if all_insider_data.empty:
        return pd.DataFrame()

    # Filter for purchases via Text field
    mask_purchase = all_insider_data['Text'].str.contains(
        'Purchase', case=False, na=False
    )

    # Exclude option exercises, grants, gifts
    mask_exclude = all_insider_data['Text'].str.contains(
        'Exercise|Grant|Gift|Conversion|Disposition|Automatic',
        case=False, na=False
    )

    purchases = all_insider_data[mask_purchase & ~mask_exclude].copy()

    if purchases.empty:
        return pd.DataFrame()

    # Parse date
    purchases['date'] = pd.to_datetime(purchases['Start Date'], errors='coerce')
    purchases = purchases.dropna(subset=['date'])

    # Filter by value
    purchases['Value'] = pd.to_numeric(purchases['Value'], errors='coerce')
    purchases = purchases[purchases['Value'] >= MIN_PURCHASE_VALUE]

    # Filter by position (C-suite and directors)
    c_suite_keywords = ['CEO', 'CFO', 'COO', 'CTO', 'Chief', 'Officer', 'Director', 'President', 'Chairman']
    mask_csuite = purchases['Position'].apply(
        lambda x: any(kw.lower() in str(x).lower() for kw in c_suite_keywords)
        if pd.notna(x) else False
    )
    purchases = purchases[mask_csuite]

    return purchases


def extract_purchases_relaxed(all_insider_data, min_value=25_000):
    """Same as extract_purchases but with custom min value."""
    if all_insider_data.empty:
        return pd.DataFrame()
    mask_purchase = all_insider_data['Text'].str.contains('Purchase', case=False, na=False)
    mask_exclude = all_insider_data['Text'].str.contains(
        'Exercise|Grant|Gift|Conversion|Disposition|Automatic', case=False, na=False
    )
    purchases = all_insider_data[mask_purchase & ~mask_exclude].copy()
    if purchases.empty:
        return pd.DataFrame()
    purchases['date'] = pd.to_datetime(purchases['Start Date'], errors='coerce')
    purchases = purchases.dropna(subset=['date'])
    purchases['Value'] = pd.to_numeric(purchases['Value'], errors='coerce')
    purchases = purchases[purchases['Value'] >= min_value]
    c_suite_keywords = ['CEO', 'CFO', 'COO', 'CTO', 'Chief', 'Officer', 'Director', 'President', 'Chairman']
    mask_csuite = purchases['Position'].apply(
        lambda x: any(kw.lower() in str(x).lower() for kw in c_suite_keywords) if pd.notna(x) else False
    )
    purchases = purchases[mask_csuite]
    return purchases


def detect_clusters(purchases):
    """Detect cluster buys: multiple insiders buying same stock within CLUSTER_WINDOW_DAYS."""
    if purchases.empty:
        return purchases

    clusters = []
    for ticker in purchases['ticker'].unique():
        ticker_buys = purchases[purchases['ticker'] == ticker].sort_values('date')
        if len(ticker_buys) < 2:
            continue

        for i, row in ticker_buys.iterrows():
            window_start = row['date'] - timedelta(days=CLUSTER_WINDOW_DAYS)
            window_end = row['date'] + timedelta(days=CLUSTER_WINDOW_DAYS)
            nearby = ticker_buys[
                (ticker_buys['date'] >= window_start) &
                (ticker_buys['date'] <= window_end)
            ]
            unique_insiders = nearby['Insider'].nunique()
            if unique_insiders >= CLUSTER_MIN_INSIDERS:
                clusters.append(i)

    purchases = purchases.copy()
    purchases['is_cluster'] = purchases.index.isin(clusters)
    return purchases


def get_price_data(tickers, start_date, end_date):
    """Fetch price data for all tickers + SPY."""
    all_tickers = list(set(tickers + ['SPY']))
    print(f"  Fetching price data for {len(all_tickers)} tickers...")

    # Download in batches
    prices = {}
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )
            if len(batch) == 1:
                sym = batch[0]
                if 'Close' in data.columns:
                    prices[sym] = data[['Close']].rename(columns={'Close': sym})
            else:
                for sym in batch:
                    try:
                        if sym in data.columns.get_level_values(0):
                            close = data[sym]['Close'].dropna()
                            if len(close) > 0:
                                prices[sym] = close
                    except Exception:
                        continue
        except Exception as e:
            print(f"  Warning: batch download failed: {e}")
            continue

        if i + batch_size < len(all_tickers):
            time.sleep(0.2)

    return prices


def compute_spy_regime(spy_prices):
    """Compute regime: SPY > 200-SMA = bull, else bear."""
    spy_sma200 = spy_prices.rolling(200).mean()
    regime = pd.Series('bull', index=spy_prices.index)
    regime[spy_prices < spy_sma200] = 'bear'
    return regime


def run_backtest(purchases, prices, spy_regime, shuffle_dates=False):
    """
    Run the insider following backtest.
    If shuffle_dates=True, randomize entry dates for permutation test.
    """
    if purchases.empty:
        return {
            'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'n_trades': 0, 'total_return': 0, 'max_dd': -1.0,
            'cagr': 0, 'trades': []
        }

    # Generate trade signals from purchases
    # Group by (ticker, date) to avoid duplicate entries on same day
    signals = purchases.groupby(['ticker', purchases['date'].dt.date]).agg({
        'Value': 'sum',
        'Insider': 'nunique',
        'is_cluster': 'any',
        'Position': 'first'
    }).reset_index()
    signals.columns = ['ticker', 'signal_date', 'total_value', 'n_insiders',
                        'is_cluster', 'position']
    signals['signal_date'] = pd.to_datetime(signals['signal_date'])

    if shuffle_dates:
        # Permutation: shuffle the signal dates
        all_trading_days = sorted(set(
            d for p in prices.values()
            for d in (p.index if isinstance(p, pd.Series) else p.index)
        ))
        if len(all_trading_days) > HOLD_DAYS * 2:
            valid_days = all_trading_days[:-HOLD_DAYS]
            signals['signal_date'] = np.random.choice(
                valid_days, size=len(signals), replace=True
            )
            signals['signal_date'] = pd.to_datetime(signals['signal_date'])

    # Execute trades
    trades = []
    for _, sig in signals.iterrows():
        ticker = sig['ticker']
        entry_date = sig['signal_date']

        if ticker not in prices or ticker == 'SPY':
            continue

        price_data = prices[ticker]
        if isinstance(price_data, pd.DataFrame):
            price_data = price_data.iloc[:, 0]

        # Find entry: next trading day after signal
        valid_dates = price_data.index[price_data.index > entry_date]
        if len(valid_dates) < 2:
            continue
        actual_entry_date = valid_dates[0]
        entry_price = price_data.loc[actual_entry_date]

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Find exit: HOLD_DAYS trading days later
        entry_idx = price_data.index.get_loc(actual_entry_date)
        exit_idx = min(entry_idx + HOLD_DAYS, len(price_data) - 1)
        actual_exit_date = price_data.index[exit_idx]
        exit_price = price_data.iloc[exit_idx]

        if pd.isna(exit_price) or exit_price <= 0:
            continue

        # If we don't have enough days, it's an incomplete trade — still count it
        if exit_idx - entry_idx < 10:
            continue

        # Cost: slippage on entry + exit
        entry_cost = entry_price * SLIPPAGE_PCT
        exit_cost = exit_price * SLIPPAGE_PCT

        # Regime-based sizing
        regime = 'bull'
        if 'SPY' in prices:
            spy_p = prices['SPY']
            if isinstance(spy_p, pd.DataFrame):
                spy_p = spy_p.iloc[:, 0]
            if actual_entry_date in spy_regime.index:
                regime = spy_regime.loc[actual_entry_date]

        size_mult = 1.0 if regime == 'bull' else REGIME_HEDGE_FACTOR

        # Cluster bonus: 1.5x size for cluster buys
        if sig['is_cluster']:
            size_mult *= 1.5

        # Return calculation
        gross_return = (exit_price - entry_price) / entry_price
        cost_return = (entry_cost + exit_cost) / entry_price
        net_return = gross_return - cost_return

        # Position-sized return
        sized_return = net_return * POSITION_SIZE * size_mult

        trades.append({
            'ticker': ticker,
            'entry_date': str(actual_entry_date.date()),
            'exit_date': str(actual_exit_date.date()),
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(exit_price), 2),
            'gross_return_pct': round(gross_return * 100, 2),
            'net_return_pct': round(net_return * 100, 2),
            'sized_return_pct': round(sized_return * 100, 4),
            'regime': regime,
            'is_cluster': bool(sig['is_cluster']),
            'n_insiders': int(sig['n_insiders']),
            'total_insider_value': float(sig['total_value']),
            'hold_days': int(exit_idx - entry_idx),
        })

    if not trades:
        return {
            'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'n_trades': 0, 'total_return': 0, 'max_dd': -1.0,
            'cagr': 0, 'trades': []
        }

    trades_df = pd.DataFrame(trades)
    returns = trades_df['net_return_pct'].values / 100.0

    # Metrics
    n_trades = len(trades)
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    std_return = returns.std() if len(returns) > 1 else 1e-9

    # Annualize: assume avg hold of HOLD_DAYS calendar days
    trades_per_year = 252 / HOLD_DAYS
    ann_return = avg_return * trades_per_year
    ann_std = std_return * np.sqrt(trades_per_year)

    sharpe = ann_return / ann_std if ann_std > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    ann_downside_std = downside_std * np.sqrt(trades_per_year)
    sortino = ann_return / ann_downside_std if ann_downside_std > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (from cumulative equity curve of sized returns)
    sized_returns = trades_df.sort_values('entry_date')['sized_return_pct'].values / 100.0
    equity = np.cumprod(1 + sized_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = drawdowns.min()

    # Total return
    total_return = (equity[-1] - 1) * 100

    # CAGR
    first_date = pd.to_datetime(trades_df['entry_date'].min())
    last_date = pd.to_datetime(trades_df['exit_date'].max())
    years = (last_date - first_date).days / 365.25
    cagr = ((equity[-1]) ** (1/years) - 1) * 100 if years > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(win_rate * 100, 1),
        'n_trades': n_trades,
        'total_return': round(total_return, 2),
        'max_dd': round(max_dd * 100, 2),
        'cagr': round(cagr, 2),
        'avg_return_pct': round(avg_return * 100, 2),
        'median_return_pct': round(float(np.median(returns) * 100), 2),
        'trades': trades if not shuffle_dates else [],
    }


def regime_analysis(trades):
    """Compute regime-stratified Sharpe."""
    if not trades:
        return {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 1.0}

    df = pd.DataFrame(trades)
    trades_per_year = 252 / HOLD_DAYS

    results = {}
    for regime in ['bull', 'bear']:
        regime_trades = df[df['regime'] == regime]
        if len(regime_trades) < 3:
            results[f'{regime}_sharpe'] = 0
            results[f'{regime}_n'] = 0
            continue
        rets = regime_trades['net_return_pct'].values / 100.0
        ann_ret = rets.mean() * trades_per_year
        ann_std = rets.std() * np.sqrt(trades_per_year) if rets.std() > 0 else 1e-9
        results[f'{regime}_sharpe'] = round(ann_ret / ann_std, 3)
        results[f'{regime}_n'] = len(regime_trades)

    bull_s = abs(results.get('bull_sharpe', 0))
    bear_s = abs(results.get('bear_sharpe', 0))
    max_s = max(bull_s, bear_s)
    regime_gap = abs(bull_s - bear_s) / max_s if max_s > 0 else 0
    results['regime_gap'] = round(regime_gap, 3)

    return results


def permutation_test(purchases, prices, spy_regime, actual_sharpe, n_perms=N_PERMUTATIONS):
    """Run permutation test: shuffle entry dates, recompute Sharpe."""
    print(f"\n  Running {n_perms} permutations...")
    perm_sharpes = []
    for i in range(n_perms):
        if (i + 1) % 50 == 0:
            print(f"    Permutation {i+1}/{n_perms}...")
        result = run_backtest(purchases, prices, spy_regime, shuffle_dates=True)
        perm_sharpes.append(result['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()

    return {
        'p_value': round(float(p_value), 4),
        'perm_mean_sharpe': round(float(perm_sharpes.mean()), 3),
        'perm_std_sharpe': round(float(perm_sharpes.std()), 3),
        'perm_max_sharpe': round(float(perm_sharpes.max()), 3),
        'actual_vs_perm_z': round(
            float((actual_sharpe - perm_sharpes.mean()) / perm_sharpes.std())
            if perm_sharpes.std() > 0 else 0, 3
        ),
    }


def main():
    print("=" * 60)
    print("INSIDER TRADING FOLLOWING STRATEGY BACKTEST")
    print("=" * 60)

    # Step 1: Get universe
    print("\n[1/6] Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"  Universe: {len(tickers)} tickers")

    # Step 2: Fetch insider data
    print(f"\n[2/6] Fetching insider transaction data...")
    all_data = []
    errors = 0
    batch_size = 20

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_insider_data, t): t for t in batch}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)

        pct = min(100, (i + batch_size) * 100 // len(tickers))
        n_with_data = len(all_data)
        print(f"  Progress: {pct}% ({n_with_data} tickers with insider data)", end='\r')
        time.sleep(0.3)  # Rate limit

    print(f"\n  Fetched insider data for {len(all_data)} tickers")

    if not all_data:
        print("ERROR: No insider data fetched. Aborting.")
        sys.exit(1)

    all_insider_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total insider transactions: {len(all_insider_df)}")

    # Step 3: Extract purchases
    print(f"\n[3/6] Filtering for open-market purchases...")
    purchases = extract_purchases(all_insider_df)
    print(f"  Open-market purchases (>=${MIN_PURCHASE_VALUE/1000:.0f}K, C-suite/directors): {len(purchases)}")

    if purchases.empty:
        print("ERROR: No qualifying insider purchases found.")
        # Try relaxing filters
        print("  Relaxing value filter to $25K...")
        purchases = extract_purchases_relaxed(all_insider_df, 25_000)
        print(f"  Purchases with relaxed filter: {len(purchases)}")

    if purchases.empty:
        print("FATAL: No insider purchases found even with relaxed filters.")
        result = {
            'strategy': 'Insider Trading Following',
            'status': 'FAILED - No data',
            'error': 'No qualifying insider purchases found in yfinance data',
            'timestamp': datetime.now().isoformat(),
        }
        with open('/home/jupiter/Lvl3Quant/data/insider_following_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    # Detect clusters
    purchases = detect_clusters(purchases)
    n_cluster = purchases['is_cluster'].sum()
    print(f"  Cluster buys (>={CLUSTER_MIN_INSIDERS} insiders within {CLUSTER_WINDOW_DAYS}d): {n_cluster}")

    # Show sample
    print("\n  Sample purchases:")
    sample = purchases[['ticker', 'date', 'Insider', 'Position', 'Value', 'is_cluster']].head(10)
    print(sample.to_string(index=False))

    # Step 4: Get price data
    print(f"\n[4/6] Fetching price data...")
    trade_tickers = purchases['ticker'].unique().tolist()
    min_date = purchases['date'].min() - timedelta(days=250)  # Need 200-SMA lookback
    max_date = datetime.now()
    prices = get_price_data(trade_tickers, min_date, max_date)
    print(f"  Got price data for {len(prices)} tickers")

    if 'SPY' not in prices:
        print("  WARNING: No SPY data — regime hedge disabled")
        spy_regime = pd.Series('bull', index=pd.date_range(min_date, max_date, freq='B'))
    else:
        spy_p = prices['SPY']
        if isinstance(spy_p, pd.DataFrame):
            spy_p = spy_p.iloc[:, 0]
        spy_regime = compute_spy_regime(spy_p)

    # Step 5: Run backtest
    print(f"\n[5/6] Running backtest...")
    result = run_backtest(purchases, prices, spy_regime, shuffle_dates=False)
    print(f"\n  === BACKTEST RESULTS ===")
    print(f"  Trades: {result['n_trades']}")
    print(f"  Win Rate: {result['wr']}%")
    print(f"  Avg Return/Trade: {result.get('avg_return_pct', 0):.2f}%")
    print(f"  Median Return/Trade: {result.get('median_return_pct', 0):.2f}%")
    print(f"  Sharpe: {result['sharpe']}")
    print(f"  Sortino: {result['sortino']}")
    print(f"  Profit Factor: {result['pf']}")
    print(f"  Max DD: {result['max_dd']}%")
    print(f"  Total Return: {result['total_return']}%")
    print(f"  CAGR: {result['cagr']}%")

    # Regime analysis
    regime = regime_analysis(result['trades'])
    print(f"\n  === REGIME ANALYSIS ===")
    print(f"  Bull Sharpe: {regime.get('bull_sharpe', 'N/A')} (n={regime.get('bull_n', 0)})")
    print(f"  Bear Sharpe: {regime.get('bear_sharpe', 'N/A')} (n={regime.get('bear_n', 0)})")
    print(f"  Regime Gap: {regime.get('regime_gap', 'N/A')}")

    # Step 6: Permutation test
    print(f"\n[6/6] Running permutation test ({N_PERMUTATIONS} iterations)...")
    perm_results = permutation_test(purchases, prices, spy_regime, result['sharpe'])
    print(f"  p-value: {perm_results['p_value']}")
    print(f"  Actual Sharpe: {result['sharpe']} vs Perm Mean: {perm_results['perm_mean_sharpe']}")
    print(f"  Z-score: {perm_results['actual_vs_perm_z']}")

    # ── 5-Gate Validation ──
    print("\n" + "=" * 60)
    print("5-GATE VALIDATION")
    print("=" * 60)

    gates = {
        'G1_sharpe_gt_0.5': result['sharpe'] > 0.5,
        'G2_perm_p_lt_0.05': perm_results['p_value'] < 0.05,
        'G3_regime_gap_lt_0.5': regime.get('regime_gap', 1) < 0.5,
        'G4_maxdd_gt_neg50': result['max_dd'] > -50,
        'G5_n_trades_gte_20': result['n_trades'] >= 20,
    }

    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate}: {status}")

    gates_passed = sum(gates.values())
    total_gates = len(gates)
    print(f"\n  RESULT: {gates_passed}/{total_gates} gates passed")

    if gates_passed == total_gates:
        verdict = "VALIDATED"
    elif gates_passed >= 4:
        verdict = "INTERESTING - near miss"
    elif gates_passed >= 3:
        verdict = "MARGINAL"
    else:
        verdict = "DEAD"

    print(f"  VERDICT: {verdict}")

    # ── Save Results ──
    output = {
        'strategy': 'Insider Trading Following',
        'description': 'Buy stocks when C-suite/directors make open-market purchases >$50K. Hold 45 days. Regime hedge: half-size when SPY < 200-SMA.',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'hold_days': HOLD_DAYS,
            'min_purchase_value': MIN_PURCHASE_VALUE,
            'cluster_window_days': CLUSTER_WINDOW_DAYS,
            'cluster_min_insiders': CLUSTER_MIN_INSIDERS,
            'slippage_pct': SLIPPAGE_PCT,
            'position_size_pct': POSITION_SIZE * 100,
            'regime_hedge_factor': REGIME_HEDGE_FACTOR,
        },
        'results': {
            'n_trades': result['n_trades'],
            'win_rate': result['wr'],
            'avg_return_pct': result.get('avg_return_pct', 0),
            'median_return_pct': result.get('median_return_pct', 0),
            'sharpe': result['sharpe'],
            'sortino': result['sortino'],
            'profit_factor': result['pf'],
            'max_drawdown_pct': result['max_dd'],
            'total_return_pct': result['total_return'],
            'cagr_pct': result['cagr'],
        },
        'regime_analysis': regime,
        'permutation_test': perm_results,
        'gates': {k: bool(v) for k, v in gates.items()},
        'gates_passed': f'{gates_passed}/{total_gates}',
        'verdict': verdict,
        'top_trades': sorted(
            result['trades'],
            key=lambda x: x['net_return_pct'],
            reverse=True
        )[:10] if result['trades'] else [],
        'worst_trades': sorted(
            result['trades'],
            key=lambda x: x['net_return_pct']
        )[:5] if result['trades'] else [],
    }

    results_path = '/home/jupiter/Lvl3Quant/data/insider_following_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {results_path}")
    print("=" * 60)

    return output


if __name__ == '__main__':
    main()
