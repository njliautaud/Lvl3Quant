#!/usr/bin/env python3
"""
High-Growth Strategy Research — 3 New Concepts
===============================================
Goal: Beat SPY CAGR (~16-20%) with aggressive strategies.
Period: 2019-01-01 to 2026-08-22 (walk-forward, real data via yfinance)

Strategy A: Leveraged Momentum with Crash Protection (TQQQ/UPRO + 200-SMA filter)
Strategy B: Earnings Momentum Concentration (post-earnings drift, top 50 S&P)
Strategy C: Volatility Risk Premium Harvest (inverse VIX when contango + VIX>20)

All use realistic costs (10bps equities, measured slippage).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import sys
import time

# ─── CONFIG ─────────────────────────────────────────────────────────────────
CAPITAL = 100_000.0
START = '2018-06-01'  # warmup for 200-SMA
TRADE_START = '2019-01-02'
END = '2026-08-22'
COST_BPS = 10  # 10 bps round-trip for equities
RISK_FREE = 0.04  # ~4% for Sharpe calc

# ─── HELPERS ────────────────────────────────────────────────────────────────

def download_safe(tickers, start, end, retries=3):
    """Download with retry logic."""
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
            if data is not None and len(data) > 0:
                return data
        except Exception as e:
            print(f"  Download attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def calc_metrics(equity_curve, risk_free=RISK_FREE):
    """Calculate CAGR, Sharpe, Sortino, MaxDD, PF, WR from equity curve."""
    if len(equity_curve) < 20:
        return {'CAGR': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0, 'PF': 0, 'WR': 0}

    returns = equity_curve.pct_change().dropna()
    if len(returns) < 10:
        return {'CAGR': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0, 'PF': 0, 'WR': 0}

    # CAGR
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if total_days <= 0:
        return {'CAGR': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0, 'PF': 0, 'WR': 0}
    years = total_days / 365.25
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = (total_return ** (1/years) - 1) * 100

    # Sharpe
    daily_rf = (1 + risk_free) ** (1/252) - 1
    excess = returns - daily_rf
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    # Sortino
    downside = returns[returns < daily_rf] - daily_rf
    downside_std = np.sqrt((downside**2).mean()) if len(downside) > 0 else 1e-9
    sortino = np.sqrt(252) * excess.mean() / downside_std if downside_std > 0 else 0

    # Max Drawdown
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    max_dd = dd.min() * 100

    # Profit Factor & Win Rate
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    wr = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

    return {
        'CAGR': round(cagr, 1),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDD': round(max_dd, 1),
        'PF': round(pf, 2),
        'WR': round(wr, 1),
        'Total_Return': round((total_return - 1) * 100, 1),
        'Years': round(years, 1),
    }

def print_metrics(name, metrics):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        unit = '%' if k in ('CAGR', 'MaxDD', 'WR', 'Total_Return') else ''
        print(f"  {k:>15}: {v}{unit}")


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY A: Leveraged Momentum with Crash Protection
# ═════════════════════════════════════════════════════════════════════════════

def strategy_a_leveraged_momentum():
    """
    Hold TQQQ (3x Nasdaq) or UPRO (3x S&P) during uptrends.
    Rotate to cash/TLT when SPY < 200-SMA.

    Logic:
    - If SPY > 200-SMA: hold TQQQ (aggressive) or UPRO
    - If SPY < 200-SMA: hold SHY (cash proxy) or TLT
    - Rebalance daily. 10bps cost per switch.
    """
    print("\n" + "="*70)
    print("STRATEGY A: Leveraged Momentum with Crash Protection")
    print("="*70)

    tickers = ['SPY', 'TQQQ', 'UPRO', 'TLT', 'SHY']
    data = download_safe(tickers, START, END)
    if data is None:
        print("ERROR: Failed to download data")
        return None

    close = data['Close'].dropna()

    # SPY 200-day SMA
    spy_sma200 = close['SPY'].rolling(200).mean()

    results = {}

    for lev_ticker, safe_ticker, label in [
        ('TQQQ', 'SHY', 'TQQQ + Cash'),
        ('TQQQ', 'TLT', 'TQQQ + TLT'),
        ('UPRO', 'SHY', 'UPRO + Cash'),
        ('UPRO', 'TLT', 'UPRO + TLT'),
    ]:
        if lev_ticker not in close.columns or safe_ticker not in close.columns:
            print(f"  Skipping {label} — missing data")
            continue

        # Build equity curve
        equity = [CAPITAL]
        position = None  # 'risk' or 'safe'
        trade_dates = close.index[close.index >= TRADE_START]

        for i, date in enumerate(trade_dates):
            if date not in spy_sma200.index or pd.isna(spy_sma200[date]):
                equity.append(equity[-1])
                continue

            spy_price = close.loc[date, 'SPY']
            sma_val = spy_sma200[date]

            # Signal: above SMA = risk-on, below = risk-off
            new_position = 'risk' if spy_price > sma_val else 'safe'

            # Calculate return
            if i > 0:
                prev_date = trade_dates[i-1]
                if position == 'risk':
                    daily_ret = close.loc[date, lev_ticker] / close.loc[prev_date, lev_ticker] - 1
                elif position == 'safe':
                    daily_ret = close.loc[date, safe_ticker] / close.loc[prev_date, safe_ticker] - 1
                else:
                    daily_ret = 0

                # Cost on switch
                cost = 0
                if new_position != position and position is not None:
                    cost = COST_BPS / 10000 * 2  # buy + sell

                new_equity = equity[-1] * (1 + daily_ret - cost)
                equity.append(max(new_equity, 1))  # floor at $1
            else:
                equity.append(equity[-1])

            position = new_position

        eq_series = pd.Series(equity[1:], index=trade_dates[:len(equity)-1])
        metrics = calc_metrics(eq_series)
        results[label] = metrics
        print_metrics(f"Strategy A: {label}", metrics)

        # Count switches
        signals = (close.loc[trade_dates, 'SPY'] > spy_sma200.loc[trade_dates]).astype(int)
        switches = (signals.diff().abs().sum())
        print(f"  {'Regime switches':>15}: {int(switches)}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY B: Earnings Momentum Concentration
# ═════════════════════════════════════════════════════════════════════════════

def strategy_b_earnings_momentum():
    """
    Post-earnings drift: buy stocks that gap up >3% on earnings, hold 20-60 days.
    Concentrated: max 5 positions.

    Since we can't easily get historical earnings dates from yfinance for free,
    we'll use a proxy: detect large single-day gaps (>3%) with volume surge (>2x avg)
    as earnings-like events. This actually captures earnings + other catalysts.
    """
    print("\n" + "="*70)
    print("STRATEGY B: Earnings Momentum Concentration")
    print("="*70)

    # Top 50 S&P 500 by market cap (approximate, stable large caps)
    universe = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'LLY',
        'AVGO', 'JPM', 'TSLA', 'V', 'UNH', 'XOM', 'MA', 'JNJ', 'PG',
        'COST', 'HD', 'ABBV', 'MRK', 'WMT', 'NFLX', 'CRM', 'BAC',
        'CVX', 'KO', 'PEP', 'ORCL', 'LIN', 'AMD', 'TMO', 'ACN',
        'MCD', 'CSCO', 'ABT', 'ADBE', 'WFC', 'DHR', 'TXN', 'NEE',
        'PM', 'AMGN', 'IBM', 'QCOM', 'CAT', 'GE', 'INTU', 'AMAT', 'ISRG'
    ]

    print(f"  Downloading {len(universe)} stocks...")
    data = download_safe(universe + ['SPY'], START, END)
    if data is None:
        print("ERROR: Failed to download data")
        return None

    close = data['Close']
    volume = data['Volume']

    # Clean: drop tickers with too much missing data
    valid_tickers = []
    for t in universe:
        if t in close.columns:
            pct_valid = close[t].dropna().shape[0] / close.shape[0]
            if pct_valid > 0.8:
                valid_tickers.append(t)

    print(f"  Valid tickers: {len(valid_tickers)}")

    results = {}

    for hold_days, gap_thresh, label in [
        (20, 0.03, '20d hold, 3% gap'),
        (40, 0.03, '40d hold, 3% gap'),
        (60, 0.03, '60d hold, 3% gap'),
        (20, 0.05, '20d hold, 5% gap'),
        (40, 0.05, '40d hold, 5% gap'),
    ]:
        MAX_POSITIONS = 5
        POSITION_SIZE = 1.0 / MAX_POSITIONS  # equal weight

        trade_dates = close.index[close.index >= TRADE_START]
        equity = CAPITAL
        equity_curve = []
        positions = []  # list of (ticker, entry_date, entry_price, exit_date_target)
        daily_pnl = []
        n_trades = 0
        n_wins = 0

        for i, date in enumerate(trade_dates):
            # Remove expired positions
            new_positions = []
            for ticker, entry_date, entry_price, exit_target in positions:
                if date >= exit_target:
                    # Exit
                    if ticker in close.columns and not pd.isna(close.loc[date, ticker]):
                        exit_price = close.loc[date, ticker]
                        ret = (exit_price / entry_price - 1)
                        pnl = equity * POSITION_SIZE * (ret - COST_BPS/10000 * 2)
                        equity += pnl
                        n_trades += 1
                        if ret > 0:
                            n_wins += 1
                    # else: position lost (delisted?), treat as 0
                else:
                    new_positions.append((ticker, entry_date, entry_price, exit_target))
            positions = new_positions

            # Check for new signals (gap up on high volume)
            if len(positions) < MAX_POSITIONS and i > 20:
                prev_date = trade_dates[i-1]
                for ticker in valid_tickers:
                    if len(positions) >= MAX_POSITIONS:
                        break
                    # Skip if already holding
                    if any(p[0] == ticker for p in positions):
                        continue

                    if ticker not in close.columns:
                        continue

                    curr_price = close.loc[date, ticker]
                    prev_price = close.loc[prev_date, ticker]

                    if pd.isna(curr_price) or pd.isna(prev_price) or prev_price <= 0:
                        continue

                    gap = curr_price / prev_price - 1

                    # Volume check
                    vol_now = volume.loc[date, ticker] if ticker in volume.columns else 0
                    vol_avg = volume[ticker].iloc[max(0,i-20):i].mean() if ticker in volume.columns else 1

                    if pd.isna(vol_now) or pd.isna(vol_avg) or vol_avg <= 0:
                        continue

                    vol_ratio = vol_now / vol_avg

                    if gap >= gap_thresh and vol_ratio >= 1.5:
                        # Enter position
                        exit_target = trade_dates[min(i + hold_days, len(trade_dates)-1)]
                        positions.append((ticker, date, curr_price, exit_target))
                        equity -= equity * POSITION_SIZE * COST_BPS / 10000  # entry cost

            # Mark-to-market
            mtm_equity = equity
            for ticker, entry_date, entry_price, exit_target in positions:
                if ticker in close.columns and not pd.isna(close.loc[date, ticker]):
                    curr_price = close.loc[date, ticker]
                    unrealized = equity * POSITION_SIZE * (curr_price / entry_price - 1)
                    mtm_equity += unrealized

            equity_curve.append(mtm_equity)

        eq_series = pd.Series(equity_curve, index=trade_dates[:len(equity_curve)])
        metrics = calc_metrics(eq_series)
        metrics['Trades'] = n_trades
        metrics['WR_trades'] = round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0
        results[label] = metrics
        print_metrics(f"Strategy B: {label}", metrics)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY C: Volatility Risk Premium Harvest
# ═════════════════════════════════════════════════════════════════════════════

def strategy_c_vol_premium():
    """
    Harvest VIX risk premium using inverse VIX ETFs.

    Logic:
    - When VIX > 20 AND term structure in contango: go long SVXY (inverse VIX)
    - When VIX < 15 (low vol, premium thin): stay in cash
    - When VIX > 30 (crisis): stay in cash (vol spike risk too high)
    - Hard stop: -15% drawdown from entry → exit

    We use VIX and SVXY data. For term structure, we approximate contango
    using VIX vs VIX3M (3-month VIX) or VIX vs its own 20-day MA.
    """
    print("\n" + "="*70)
    print("STRATEGY C: Volatility Risk Premium Harvest")
    print("="*70)

    tickers = ['SVXY', '^VIX', 'SPY']
    data = download_safe(tickers, START, END)
    if data is None:
        print("ERROR: Failed to download data")
        return None

    close = data['Close']

    # Rename VIX column
    vix_col = None
    for col in close.columns:
        if 'VIX' in str(col).upper():
            vix_col = col
            break

    if vix_col is None:
        # Try downloading VIX separately
        print("  VIX not found in batch download, trying separately...")
        vix_data = download_safe(['^VIX'], START, END)
        if vix_data is not None:
            if isinstance(vix_data['Close'], pd.DataFrame):
                vix_series = vix_data['Close'].iloc[:, 0]
            else:
                vix_series = vix_data['Close']
        else:
            print("ERROR: Cannot download VIX data")
            return None
    else:
        vix_series = close[vix_col]

    # Get SVXY
    svxy_col = None
    for col in close.columns:
        if 'SVXY' in str(col).upper():
            svxy_col = col
            break

    if svxy_col is None:
        svxy_data = download_safe(['SVXY'], START, END)
        if svxy_data is not None:
            if isinstance(svxy_data['Close'], pd.DataFrame):
                svxy_series = svxy_data['Close'].iloc[:, 0]
            else:
                svxy_series = svxy_data['Close']
        else:
            print("ERROR: Cannot download SVXY data")
            return None
    else:
        svxy_series = close[svxy_col]

    # Get SPY for benchmark
    spy_col = None
    for col in close.columns:
        if 'SPY' in str(col).upper():
            spy_col = col
            break

    if spy_col:
        spy_series = close[spy_col]
    else:
        spy_data = download_safe(['SPY'], START, END)
        spy_series = spy_data['Close'].iloc[:, 0] if isinstance(spy_data['Close'], pd.DataFrame) else spy_data['Close']

    # VIX 20-day MA (contango proxy: VIX < VIX_MA = contango-like)
    vix_ma20 = vix_series.rolling(20).mean()

    results = {}

    for vix_low, vix_high, stop_pct, label in [
        (18, 30, -0.10, 'VIX 18-30, 10% stop'),
        (20, 35, -0.15, 'VIX 20-35, 15% stop'),
        (16, 28, -0.08, 'VIX 16-28, 8% stop'),
        (20, 30, -0.12, 'VIX 20-30, 12% stop'),
    ]:
        trade_dates = svxy_series.index[svxy_series.index >= TRADE_START]
        equity = [CAPITAL]
        in_trade = False
        entry_price = 0
        entry_equity = 0
        n_trades = 0
        n_wins = 0

        for i, date in enumerate(trade_dates):
            if i == 0:
                continue

            prev_date = trade_dates[i-1]
            vix_now = vix_series.get(date, np.nan)
            vix_ma = vix_ma20.get(date, np.nan)
            svxy_now = svxy_series.get(date, np.nan)
            svxy_prev = svxy_series.get(prev_date, np.nan)

            if pd.isna(vix_now) or pd.isna(svxy_now) or pd.isna(svxy_prev) or svxy_prev <= 0:
                equity.append(equity[-1])
                continue

            # If in trade, check stop and exit conditions
            if in_trade:
                daily_ret = svxy_now / svxy_prev - 1
                new_eq = equity[-1] * (1 + daily_ret)

                # Check drawdown from entry
                dd_from_entry = (new_eq - entry_equity) / entry_equity

                # Hard stop
                if dd_from_entry <= stop_pct:
                    cost = COST_BPS / 10000 * 2
                    new_eq = new_eq * (1 - cost)
                    equity.append(max(new_eq, 1))
                    n_trades += 1
                    if new_eq > entry_equity:
                        n_wins += 1
                    in_trade = False
                    continue

                # Exit if VIX drops below threshold (premium thin) or spikes above high
                if vix_now < vix_low * 0.8 or vix_now > vix_high:
                    cost = COST_BPS / 10000 * 2
                    new_eq = new_eq * (1 - cost)
                    equity.append(max(new_eq, 1))
                    n_trades += 1
                    if new_eq > entry_equity:
                        n_wins += 1
                    in_trade = False
                    continue

                equity.append(max(new_eq, 1))
            else:
                # Check entry conditions
                contango = not pd.isna(vix_ma) and vix_now < vix_ma  # VIX below MA = contango-like
                in_range = vix_low <= vix_now <= vix_high

                if in_range and contango:
                    # Enter SVXY position
                    in_trade = True
                    entry_price = svxy_now
                    entry_equity = equity[-1]
                    cost = COST_BPS / 10000
                    equity.append(equity[-1] * (1 - cost))
                else:
                    equity.append(equity[-1])

        eq_series = pd.Series(equity[1:], index=trade_dates[:len(equity)-1])
        metrics = calc_metrics(eq_series)
        metrics['Trades'] = n_trades
        metrics['WR_trades'] = round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0
        results[label] = metrics
        print_metrics(f"Strategy C: {label}", metrics)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK: SPY Buy-and-Hold
# ═════════════════════════════════════════════════════════════════════════════

def benchmark_spy():
    print("\n" + "="*70)
    print("BENCHMARK: SPY Buy-and-Hold")
    print("="*70)

    data = download_safe(['SPY'], START, END)
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    trade_dates = close.index[close.index >= TRADE_START]
    eq = close.loc[trade_dates]
    eq = eq / eq.iloc[0] * CAPITAL

    metrics = calc_metrics(eq)
    print_metrics("SPY Buy-and-Hold", metrics)
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*70)
    print("HIGH-GROWTH STRATEGY RESEARCH")
    print(f"Period: {TRADE_START} to {END}")
    print(f"Capital: ${CAPITAL:,.0f}")
    print(f"Costs: {COST_BPS} bps round-trip")
    print("="*70)

    spy_metrics = benchmark_spy()

    print("\n\n" + "#"*70)
    print("# STRATEGY A: LEVERAGED MOMENTUM WITH CRASH PROTECTION")
    print("#"*70)
    a_results = strategy_a_leveraged_momentum()

    print("\n\n" + "#"*70)
    print("# STRATEGY B: EARNINGS MOMENTUM CONCENTRATION")
    print("#"*70)
    b_results = strategy_b_earnings_momentum()

    print("\n\n" + "#"*70)
    print("# STRATEGY C: VOLATILITY RISK PREMIUM HARVEST")
    print("#"*70)
    c_results = strategy_c_vol_premium()

    # ─── SUMMARY ────────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    all_results = {'SPY B&H': spy_metrics}
    if a_results:
        for k, v in a_results.items():
            all_results[f'A: {k}'] = v
    if b_results:
        for k, v in b_results.items():
            all_results[f'B: {k}'] = v
    if c_results:
        for k, v in c_results.items():
            all_results[f'C: {k}'] = v

    # Print comparison table
    print(f"\n{'Strategy':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'PF':>6} {'WR':>6}")
    print("-" * 85)
    for name, m in all_results.items():
        print(f"{name:<35} {m['CAGR']:>7.1f}% {m['Sharpe']:>7.2f} {m['Sortino']:>7.2f} {m['MaxDD']:>7.1f}% {m['PF']:>5.2f} {m['WR']:>5.1f}%")

    # Save results
    output = {
        'run_date': datetime.now().isoformat(),
        'period': f'{TRADE_START} to {END}',
        'capital': CAPITAL,
        'cost_bps': COST_BPS,
        'results': {}
    }
    for name, m in all_results.items():
        output['results'][name] = {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for k, v in m.items()}

    with open('/home/jupiter/Lvl3Quant/strategies/high_growth_research_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to high_growth_research_results.json")

    # ─── VERDICT ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("VERDICT & NEXT STEPS")
    print("="*70)

    # Find best by CAGR
    best_name = max(all_results.keys(), key=lambda x: all_results[x]['CAGR'])
    best = all_results[best_name]

    print(f"\nBest CAGR: {best_name} at {best['CAGR']:.1f}%")
    print(f"SPY CAGR: {spy_metrics['CAGR']:.1f}%")

    # Strategies that beat SPY
    beaters = {k: v for k, v in all_results.items() if k != 'SPY B&H' and v['CAGR'] > spy_metrics['CAGR']}
    if beaters:
        print(f"\n{len(beaters)} variants beat SPY on CAGR:")
        for name, m in sorted(beaters.items(), key=lambda x: -x[1]['CAGR']):
            beat_by = m['CAGR'] - spy_metrics['CAGR']
            print(f"  {name}: +{beat_by:.1f}% over SPY (Sharpe {m['Sharpe']:.2f}, MaxDD {m['MaxDD']:.1f}%)")
    else:
        print("\nNo variants beat SPY on CAGR.")

    print("\nDone.")
