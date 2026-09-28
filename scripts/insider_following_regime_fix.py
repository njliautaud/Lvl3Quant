#!/usr/bin/env python3
"""
Insider Following Strategy — Regime Fix Variants
HC #763 follow-up

Problem: Original strategy has regime gap 0.671 (fails <0.5 gate).
Bear Sharpe (1.843) >> Bull Sharpe (0.607) — inverted from typical.
Insiders buying in bear markets = stronger signal (undervalued companies).
Current half-size-in-bear hedge HURTS by reducing exposure when signal is strongest.

Tests 6 variants (A-F) to fix the regime gap while preserving edge.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Configuration ──
HOLD_DAYS = 45
MIN_PURCHASE_VALUE = 50_000
CLUSTER_WINDOW_DAYS = 14
CLUSTER_MIN_INSIDERS = 2
SLIPPAGE_PCT = 0.0002
POSITION_SIZE = 0.02
INITIAL_CAPITAL = 645  # Agentic account
N_PERMUTATIONS = 500
OOT_START = '2022-01-01'
OOT_END = '2026-07-25'

SP500_TICKERS = [
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


def get_sp500_tickers():
    """Get S&P 500 tickers with Wikipedia fallback."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        return df['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception:
        return SP500_TICKERS


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
    return None


def extract_purchases(all_insider_data, min_value=MIN_PURCHASE_VALUE):
    """Filter for open-market C-suite/director purchases."""
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

    c_suite_keywords = ['CEO', 'CFO', 'COO', 'CTO', 'Chief', 'Officer', 'Director',
                        'President', 'Chairman']
    mask_csuite = purchases['Position'].apply(
        lambda x: any(kw.lower() in str(x).lower() for kw in c_suite_keywords)
        if pd.notna(x) else False
    )
    purchases = purchases[mask_csuite]
    return purchases


def detect_clusters(purchases):
    """Detect cluster buys: 2+ insiders buying same stock within 14 days."""
    if purchases.empty:
        return purchases

    clusters = set()
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
            if nearby['Insider'].nunique() >= CLUSTER_MIN_INSIDERS:
                clusters.add(i)

    purchases = purchases.copy()
    purchases['is_cluster'] = purchases.index.isin(clusters)
    return purchases


def get_price_data(tickers, start_date, end_date):
    """Fetch price data for all tickers + SPY + VIX (^VIX)."""
    all_tickers = list(set(tickers + ['SPY', '^VIX']))
    print(f"  Fetching price data for {len(all_tickers)} tickers...")

    prices = {}
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        try:
            data = yf.download(batch, start=start_date, end=end_date,
                               progress=False, group_by='ticker',
                               auto_adjust=True, threads=True)
            if len(batch) == 1:
                sym = batch[0]
                if 'Close' in data.columns:
                    prices[sym] = data['Close']
                elif isinstance(data.columns, pd.MultiIndex):
                    prices[sym] = data[sym]['Close'].dropna()
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
        if i + batch_size < len(all_tickers):
            time.sleep(0.2)

    return prices


def compute_spy_regime(spy_prices):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy_sma200 = spy_prices.rolling(200).mean()
    regime = pd.Series('bull', index=spy_prices.index)
    regime[spy_prices < spy_sma200] = 'bear'
    return regime


def get_stock_sma50(prices, ticker, date):
    """Check if stock is above its own 50-SMA on given date."""
    if ticker not in prices:
        return None
    p = prices[ticker]
    if isinstance(p, pd.DataFrame):
        p = p.iloc[:, 0]
    # Get data up to and including date
    available = p[p.index <= date]
    if len(available) < 50:
        return None
    sma50 = available.tail(50).mean()
    current = available.iloc[-1]
    return current > sma50


def get_vix_on_date(prices, date):
    """Get VIX value on a given date (nearest available)."""
    if '^VIX' not in prices:
        return 20.0  # Default
    vix = prices['^VIX']
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    # Find nearest date
    available = vix[vix.index <= date]
    if len(available) == 0:
        return 20.0
    return float(available.iloc[-1])


def run_backtest_variant(purchases, prices, spy_regime, variant='original',
                         shuffle_dates=False):
    """
    Run backtest with different regime-sizing rules.

    Variants:
    - 'original': half-size in bear (the broken baseline)
    - 'A_inverse': double size in bear, half in bull
    - 'B_bull_reduce': 40% position in bull, full in bear
    - 'C_vix_inverse': size = base * max(0.3, VIX/30)
    - 'D_bear_only': only trade when SPY < 200-SMA
    - 'E_cluster_bull': in bull require cluster; in bear single OK
    - 'F_quality_bull': in bull require stock > 50-SMA; in bear no filter
    """
    if purchases.empty:
        return _empty_result()

    # Generate signals
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

    trades = []
    for _, sig in signals.iterrows():
        ticker = sig['ticker']
        entry_date = sig['signal_date']

        if ticker not in prices or ticker == 'SPY' or ticker == '^VIX':
            continue

        price_data = prices[ticker]
        if isinstance(price_data, pd.DataFrame):
            price_data = price_data.iloc[:, 0]

        valid_dates = price_data.index[price_data.index > entry_date]
        if len(valid_dates) < 2:
            continue
        actual_entry_date = valid_dates[0]
        entry_price = price_data.loc[actual_entry_date]

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        entry_idx = price_data.index.get_loc(actual_entry_date)
        exit_idx = min(entry_idx + HOLD_DAYS, len(price_data) - 1)
        actual_exit_date = price_data.index[exit_idx]
        exit_price = price_data.iloc[exit_idx]

        if pd.isna(exit_price) or exit_price <= 0:
            continue
        if exit_idx - entry_idx < 10:
            continue

        # Determine regime
        regime = 'bull'
        if 'SPY' in prices:
            if actual_entry_date in spy_regime.index:
                regime = spy_regime.loc[actual_entry_date]

        # ── Variant-specific logic ──
        size_mult = 1.0
        skip_trade = False

        if variant == 'original':
            # Original: half-size in bear
            size_mult = 1.0 if regime == 'bull' else 0.5

        elif variant == 'A_inverse':
            # INVERSE: double in bear, half in bull
            size_mult = 0.5 if regime == 'bull' else 2.0

        elif variant == 'B_bull_reduce':
            # 40% in bull, full in bear
            size_mult = 0.4 if regime == 'bull' else 1.0

        elif variant == 'C_vix_inverse':
            # Size proportional to VIX
            vix = get_vix_on_date(prices, actual_entry_date)
            size_mult = max(0.3, vix / 30.0)

        elif variant == 'D_bear_only':
            # Skip bull trades entirely
            if regime == 'bull':
                skip_trade = True
            size_mult = 1.0

        elif variant == 'E_cluster_bull':
            # In bull: require cluster buy. In bear: single insider OK
            if regime == 'bull' and not sig['is_cluster']:
                skip_trade = True
            size_mult = 1.0

        elif variant == 'F_quality_bull':
            # In bull: require stock > 50-SMA. In bear: no filter
            if regime == 'bull':
                above_sma = get_stock_sma50(prices, ticker, actual_entry_date)
                if above_sma is None or not above_sma:
                    skip_trade = True
            size_mult = 1.0

        if skip_trade:
            continue

        # Cluster bonus: 1.5x for cluster buys (all variants)
        if sig['is_cluster']:
            size_mult *= 1.5

        # Compute return
        entry_cost = entry_price * SLIPPAGE_PCT
        exit_cost = exit_price * SLIPPAGE_PCT
        gross_return = (exit_price - entry_price) / entry_price
        cost_return = (entry_cost + exit_cost) / entry_price
        net_return = gross_return - cost_return
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
            'size_mult': round(size_mult, 2),
        })

    if not trades:
        return _empty_result()

    return _compute_metrics(trades, shuffle_dates)


def _empty_result():
    return {
        'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
        'n_trades': 0, 'total_return': 0, 'max_dd': -100.0,
        'cagr': 0, 'trades': [],
    }


def _compute_metrics(trades, shuffle_dates=False):
    """Compute all risk-adjusted metrics from a list of trades."""
    trades_df = pd.DataFrame(trades)
    returns = trades_df['net_return_pct'].values / 100.0

    n_trades = len(trades)
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    std_return = returns.std() if len(returns) > 1 else 1e-9

    trades_per_year = 252 / HOLD_DAYS
    ann_return = avg_return * trades_per_year
    ann_std = std_return * np.sqrt(trades_per_year) if std_return > 0 else 1e-9
    sharpe = ann_return / ann_std if ann_std > 0 else 0

    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    ann_downside_std = downside_std * np.sqrt(trades_per_year)
    sortino = ann_return / ann_downside_std if ann_downside_std > 0 else 0

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    sized_returns = trades_df.sort_values('entry_date')['sized_return_pct'].values / 100.0
    equity = np.cumprod(1 + sized_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = drawdowns.min()

    total_return = (equity[-1] - 1) * 100

    first_date = pd.to_datetime(trades_df['entry_date'].min())
    last_date = pd.to_datetime(trades_df['exit_date'].max())
    years = (last_date - first_date).days / 365.25
    cagr = ((equity[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0

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
    """Compute per-regime Sharpe and regime gap."""
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
    results['regime_gap'] = round(abs(bull_s - bear_s) / max_s if max_s > 0 else 0, 3)

    return results


def permutation_test(purchases, prices, spy_regime, actual_sharpe, variant,
                     n_perms=N_PERMUTATIONS):
    """Shuffle entry dates, recompute Sharpe, get p-value."""
    print(f"    Running {n_perms} permutations for {variant}...")
    perm_sharpes = []
    for i in range(n_perms):
        if (i + 1) % 100 == 0:
            print(f"      Permutation {i + 1}/{n_perms}...")
        result = run_backtest_variant(purchases, prices, spy_regime,
                                      variant=variant, shuffle_dates=True)
        perm_sharpes.append(result['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()
    perm_std = perm_sharpes.std() if perm_sharpes.std() > 0 else 1e-9

    return {
        'p_value': round(float(p_value), 4),
        'perm_mean_sharpe': round(float(perm_sharpes.mean()), 3),
        'perm_std_sharpe': round(float(perm_sharpes.std()), 3),
        'perm_max_sharpe': round(float(perm_sharpes.max()), 3),
        'z_score': round(float((actual_sharpe - perm_sharpes.mean()) / perm_std), 3),
    }


def five_gate_validation(result, regime, perm):
    """Apply 5-gate validation."""
    gates = {
        'G1_sharpe_gt_0.5': result['sharpe'] > 0.5,
        'G2_perm_p_lt_0.05': perm['p_value'] < 0.05,
        'G3_regime_gap_lt_0.5': regime.get('regime_gap', 1) < 0.5,
        'G4_maxdd_gt_neg50': result['max_dd'] > -50,
        'G5_n_trades_gte_20': result['n_trades'] >= 20,
    }
    passed = sum(gates.values())
    return gates, passed


def main():
    print("=" * 70)
    print("INSIDER FOLLOWING STRATEGY — REGIME FIX VARIANTS")
    print("=" * 70)

    # ── Step 1: Get universe ──
    print("\n[1/5] Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"  Universe: {len(tickers)} tickers")

    # ── Step 2: Fetch insider transaction data ──
    print(f"\n[2/5] Fetching insider transaction data...")
    all_data = []
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_insider_data, t): t for t in batch}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_data.append(result)
        pct = min(100, (i + batch_size) * 100 // len(tickers))
        print(f"  Progress: {pct}% ({len(all_data)} tickers with data)", end='\r')
        time.sleep(0.3)

    print(f"\n  Fetched insider data for {len(all_data)} tickers")

    if not all_data:
        print("ERROR: No insider data fetched. Aborting.")
        sys.exit(1)

    all_insider_df = pd.concat(all_data, ignore_index=True)
    print(f"  Total insider transactions: {len(all_insider_df)}")

    # ── Step 3: Extract purchases + detect clusters ──
    print(f"\n[3/5] Filtering for open-market purchases...")
    purchases = extract_purchases(all_insider_df)
    print(f"  Qualifying purchases (>=$50K, C-suite/directors): {len(purchases)}")

    if len(purchases) < 10:
        print("  Relaxing to $25K...")
        purchases = extract_purchases(all_insider_df, min_value=25_000)
        print(f"  Purchases with relaxed filter: {len(purchases)}")

    if purchases.empty:
        print("FATAL: No insider purchases found.")
        sys.exit(1)

    purchases = detect_clusters(purchases)
    n_cluster = purchases['is_cluster'].sum()
    print(f"  Cluster buys (2+ insiders within 14d): {n_cluster}")

    # ── Step 4: Fetch price data (single call for consistency) ──
    print(f"\n[4/5] Fetching price data...")
    trade_tickers = purchases['ticker'].unique().tolist()
    min_date = purchases['date'].min() - timedelta(days=250)
    max_date = datetime.now()
    prices = get_price_data(trade_tickers, min_date, max_date)
    print(f"  Got price data for {len(prices)} tickers")

    # Compute regime
    if 'SPY' not in prices:
        print("  WARNING: No SPY data — using bull default")
        spy_regime = pd.Series('bull', index=pd.date_range(min_date, max_date, freq='B'))
    else:
        spy_p = prices['SPY']
        if isinstance(spy_p, pd.DataFrame):
            spy_p = spy_p.iloc[:, 0]
        spy_regime = compute_spy_regime(spy_p)

    # ── Step 5: Run all 6 variants + original baseline ──
    print(f"\n[5/5] Running 7 backtests (original + 6 variants)...")

    variants = {
        'original': 'Original (half-size in bear)',
        'A_inverse': 'INVERSE regime: 2x bear, 0.5x bull',
        'B_bull_reduce': '40% in bull, full in bear',
        'C_vix_inverse': 'VIX-inverse sizing: base * max(0.3, VIX/30)',
        'D_bear_only': 'Bear-only: skip bull trades',
        'E_cluster_bull': 'Cluster-only in bull, single OK in bear',
        'F_quality_bull': 'Bull: stock > 50-SMA filter; Bear: no filter',
    }

    all_results = {}

    for variant_key, variant_desc in variants.items():
        print(f"\n  ── {variant_key}: {variant_desc} ──")

        result = run_backtest_variant(purchases, prices, spy_regime,
                                      variant=variant_key, shuffle_dates=False)

        print(f"    Trades: {result['n_trades']}, WR: {result['wr']}%, "
              f"Sharpe: {result['sharpe']}, Sortino: {result['sortino']}, "
              f"PF: {result['pf']}, MaxDD: {result['max_dd']}%")

        regime = regime_analysis(result['trades'])
        print(f"    Bull Sharpe: {regime.get('bull_sharpe', 0)} (n={regime.get('bull_n', 0)}), "
              f"Bear Sharpe: {regime.get('bear_sharpe', 0)} (n={regime.get('bear_n', 0)}), "
              f"Gap: {regime.get('regime_gap', 'N/A')}")

        # Permutation test (skip if <20 trades)
        if result['n_trades'] >= 20:
            perm = permutation_test(purchases, prices, spy_regime,
                                    result['sharpe'], variant_key)
            print(f"    Perm p={perm['p_value']}, z={perm['z_score']}")
        else:
            perm = {'p_value': 1.0, 'perm_mean_sharpe': 0, 'perm_std_sharpe': 0,
                    'perm_max_sharpe': 0, 'z_score': 0}
            print(f"    Skipped permutation test (<20 trades)")

        gates, n_passed = five_gate_validation(result, regime, perm)
        gate_str = ' '.join([
            f"{'P' if v else 'F'}" for v in gates.values()
        ])
        print(f"    Gates: {gate_str} => {n_passed}/5")

        if n_passed == 5:
            verdict = "VALIDATED"
        elif n_passed == 4:
            verdict = "NEAR MISS"
        elif n_passed == 3:
            verdict = "MARGINAL"
        else:
            verdict = "DEAD"
        print(f"    Verdict: {verdict}")

        all_results[variant_key] = {
            'description': variant_desc,
            'metrics': {
                'n_trades': result['n_trades'],
                'win_rate': result['wr'],
                'avg_return_pct': result.get('avg_return_pct', 0),
                'median_return_pct': result.get('median_return_pct', 0),
                'sharpe': result['sharpe'],
                'sortino': result['sortino'],
                'profit_factor': result['pf'],
                'max_drawdown_pct': result['max_dd'],
                'total_return_pct': result['total_return'],
                'cagr_pct': result.get('cagr', 0),
            },
            'regime_analysis': regime,
            'permutation_test': perm,
            'gates': {k: bool(v) for k, v in gates.items()},
            'gates_passed': f'{n_passed}/5',
            'verdict': verdict,
        }

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY — ALL VARIANTS")
    print("=" * 70)
    print(f"{'Variant':<18} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
          f"{'MaxDD':>7} {'#Tr':>5} {'Gap':>6} {'Gates':>6} {'Verdict':<12}")
    print("-" * 95)

    best_variant = None
    best_gates = -1

    for vk, vr in all_results.items():
        m = vr['metrics']
        ra = vr['regime_analysis']
        gp = int(vr['gates_passed'].split('/')[0])
        print(f"{vk:<18} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.3f} "
              f"{m['win_rate']:>5.1f}% {m['max_drawdown_pct']:>6.2f}% {m['n_trades']:>5} "
              f"{ra.get('regime_gap', 0):>6.3f} {vr['gates_passed']:>6} {vr['verdict']:<12}")

        if gp > best_gates or (gp == best_gates and m['sharpe'] > all_results.get(best_variant, {}).get('metrics', {}).get('sharpe', 0)):
            best_gates = gp
            best_variant = vk

    print(f"\n  BEST VARIANT: {best_variant} ({all_results[best_variant]['verdict']})")
    print(f"    {all_results[best_variant]['description']}")

    # ── Save results ──
    output = {
        'strategy': 'Insider Following — Regime Fix',
        'problem': 'Original regime gap 0.671 (fails <0.5 gate). Bear Sharpe (1.843) >> Bull (0.607). '
                   'Half-size-in-bear hedge hurts by reducing exposure when signal is strongest.',
        'timestamp': datetime.now().isoformat(),
        'oot_period': f'{OOT_START} to {OOT_END}',
        'initial_capital': INITIAL_CAPITAL,
        'n_permutations': N_PERMUTATIONS,
        'insider_data_source': 'yfinance Ticker.insider_transactions (Form 4 filings)',
        'n_tickers_with_insider_data': len(all_data),
        'n_qualifying_purchases': len(purchases),
        'n_cluster_buys': int(n_cluster),
        'variants': all_results,
        'best_variant': best_variant,
        'best_variant_verdict': all_results[best_variant]['verdict'],
        'recommendation': (
            f"Use variant {best_variant}: {all_results[best_variant]['description']}. "
            f"Gates: {all_results[best_variant]['gates_passed']}. "
            f"Sharpe={all_results[best_variant]['metrics']['sharpe']}, "
            f"Regime gap={all_results[best_variant]['regime_analysis'].get('regime_gap', 'N/A')}."
        ),
    }

    results_path = '/home/jupiter/Lvl3Quant/data/insider_following_regime_fix_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {results_path}")
    print("=" * 70)

    return output


if __name__ == '__main__':
    main()
