#!/usr/bin/env python3
"""
Earnings Options Strategy v1 — Systematic Pre-Earnings Plays
==============================================================

From earnings_move_predictor_v1 (entry #916):
- MSFT: 85% gap-up rate, +2.0% post-gap drift → BULL bias
- PG: 70% gap-DOWN → BEAR bias
- AAPL: 75% gap-DOWN, worse when pre-earnings momentum high → BEAR bias
- AMZN: 55% up, largest moves ±1.3% → NEUTRAL
- Expected moves small (0.4-1.3%) for mega-caps

STRATEGIES:
1. Directional bull/bear spreads based on historical gap direction
2. Iron condors on low-move names (sell premium when expected move < spread width)
3. Pre-earnings momentum filter: only take directional trades AGAINST momentum
   (contrarian = better for earnings because momentum = already priced in)
4. Straddle buying on high-move names (AMZN, volatile small-caps)

Test on mega-caps with 20+ quarters of earnings data.
Use actual earnings dates from yfinance.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'earnings_options_strategy_v1_results.json'

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass


def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_earnings_dates(ticker, close_series):
    """
    Detect earnings dates from price data using gap detection.
    Earnings cause abnormal overnight gaps (>1.5% for mega-caps).
    Returns list of (pre_earnings_date, post_earnings_date) tuples.
    """
    rets = close_series.pct_change()
    # Detect large overnight gaps (earnings-like moves)
    # Mega-caps typically move 1-5% on earnings
    threshold = 0.015  # 1.5% gap

    gaps = rets[abs(rets) > threshold]

    # Filter to quarterly cadence (at least 60 days apart)
    earnings_dates = []
    last_date = None
    for date in gaps.index:
        if last_date is None or (date - last_date).days > 60:
            # Pre-earnings = day before, post = gap day
            pre_idx = close_series.index.get_loc(date) - 1
            if pre_idx >= 0:
                pre_date = close_series.index[pre_idx]
                earnings_dates.append({
                    'pre_date': pre_date,
                    'post_date': date,
                    'gap_pct': float(rets.loc[date]) * 100,
                    'direction': 'up' if rets.loc[date] > 0 else 'down',
                })
            last_date = date

    return earnings_dates


def compute_earnings_stats(earnings_events, close_series, lookback=20):
    """Compute statistics for each earnings event."""
    for event in earnings_events:
        pre_date = event['pre_date']

        # Pre-earnings momentum (20-day return before earnings)
        pre_idx = close_series.index.get_loc(pre_date)
        if pre_idx >= lookback:
            pre_px = float(close_series.iloc[pre_idx - lookback])
            curr_px = float(close_series.iloc[pre_idx])
            event['pre_momentum'] = (curr_px / pre_px - 1) * 100
        else:
            event['pre_momentum'] = 0

        # Post-earnings drift (5-day return after gap)
        post_idx = close_series.index.get_loc(event['post_date'])
        if post_idx + 5 < len(close_series):
            post_px = float(close_series.iloc[post_idx])
            drift_px = float(close_series.iloc[post_idx + 5])
            event['post_drift'] = (drift_px / post_px - 1) * 100
        else:
            event['post_drift'] = 0

    return earnings_events


def simulate_earnings_strategy(ticker, close_series, vix_close, spy_close,
                               strategy='directional', capital=10000,
                               spread_pct=3.0, dte=7, name='base',
                               min_history=8):
    """
    Simulate earnings options strategy.

    strategy types:
    - 'directional': bull/bear spreads based on historical direction bias
    - 'contrarian': trade AGAINST pre-earnings momentum
    - 'iron_condor': sell IC on low-move names
    - 'straddle': buy straddles on high-move names
    - 'combined': directional + contrarian filter
    """
    fprint(f"\n--- {name} ({ticker}) ---")

    spy_sma200 = spy_close.rolling(200).mean()

    # Detect earnings
    earnings = get_earnings_dates(ticker, close_series)
    earnings = compute_earnings_stats(earnings, close_series)

    if len(earnings) < min_history + 4:
        fprint(f"  Only {len(earnings)} earnings events detected, need {min_history + 4}")
        return None

    trades = []
    equity = capital

    # Walk-forward: use first min_history events to compute stats,
    # then trade subsequent events
    for i in range(min_history, len(earnings)):
        event = earnings[i]
        pre_date = event['pre_date']
        post_date = event['post_date']

        if pre_date not in close_series.index or post_date not in close_series.index:
            continue

        S = float(close_series.loc[pre_date])
        vix = float(vix_close.loc[pre_date]) if pre_date in vix_close.index else 20
        sigma = vix / 100 * 1.5  # Earnings IV is typically 1.5x normal

        spy_val = float(spy_close.loc[pre_date]) if pre_date in spy_close.index else 0
        sma_val = float(spy_sma200.loc[pre_date]) if pre_date in spy_sma200.index else spy_val
        regime = 'bear' if spy_val < sma_val else 'bull'

        # Historical stats from prior events
        prior = earnings[:i]
        up_rate = sum(1 for e in prior if e['direction'] == 'up') / len(prior)
        avg_gap = np.mean([abs(e['gap_pct']) for e in prior])
        avg_gap_up = np.mean([e['gap_pct'] for e in prior if e['direction'] == 'up']) if up_rate > 0 else 0
        avg_gap_down = np.mean([e['gap_pct'] for e in prior if e['direction'] == 'down']) if up_rate < 1 else 0

        T = dte / 365
        actual_gap = event['gap_pct']

        if strategy == 'directional':
            # Trade in direction of historical bias (>60% or <40%)
            if up_rate >= 0.60:
                # Bull call spread
                K_long = round(S)
                K_short = round(S * (1 + spread_pct / 100))
                cost = (bs_price(S, K_long, T, sigma, opt='call') -
                       bs_price(S, K_short, T, sigma, opt='call'))

                if cost <= 0 or cost * 100 > equity * 0.15:
                    continue

                # Settlement
                S_post = float(close_series.loc[post_date])
                long_val = max(0, S_post - K_long)
                short_val = max(0, S_post - K_short)
                pnl = ((long_val - short_val) - cost) * 100 - 1.30

            elif up_rate <= 0.40:
                # Bear put spread
                K_long = round(S)
                K_short = round(S * (1 - spread_pct / 100))
                cost = (bs_price(S, K_long, T, sigma, opt='put') -
                       bs_price(S, K_short, T, sigma, opt='put'))

                if cost <= 0 or cost * 100 > equity * 0.15:
                    continue

                S_post = float(close_series.loc[post_date])
                long_val = max(0, K_long - S_post)
                short_val = max(0, K_short - S_post)
                pnl = ((long_val - short_val) - cost) * 100 - 1.30
            else:
                continue  # No clear bias

        elif strategy == 'contrarian':
            # Trade AGAINST pre-earnings momentum (contrarian)
            pre_mom = event['pre_momentum']

            if pre_mom > 5:  # Stock up big → buy bear put spread
                K_long = round(S)
                K_short = round(S * (1 - spread_pct / 100))
                cost = (bs_price(S, K_long, T, sigma, opt='put') -
                       bs_price(S, K_short, T, sigma, opt='put'))

                if cost <= 0 or cost * 100 > equity * 0.15:
                    continue

                S_post = float(close_series.loc[post_date])
                long_val = max(0, K_long - S_post)
                short_val = max(0, K_short - S_post)
                pnl = ((long_val - short_val) - cost) * 100 - 1.30

            elif pre_mom < -5:  # Stock down big → buy bull call spread
                K_long = round(S)
                K_short = round(S * (1 + spread_pct / 100))
                cost = (bs_price(S, K_long, T, sigma, opt='call') -
                       bs_price(S, K_short, T, sigma, opt='call'))

                if cost <= 0 or cost * 100 > equity * 0.15:
                    continue

                S_post = float(close_series.loc[post_date])
                long_val = max(0, S_post - K_long)
                short_val = max(0, S_post - K_short)
                pnl = ((long_val - short_val) - cost) * 100 - 1.30
            else:
                continue  # No strong momentum to fade

        elif strategy == 'iron_condor':
            # Sell IC on earnings (sell premium)
            if avg_gap > 3:  # Skip high-move names
                continue

            call_short = round(S * 1.03)
            call_long = round(S * 1.06)
            put_short = round(S * 0.97)
            put_long = round(S * 0.94)

            sc = bs_price(S, call_short, T, sigma, opt='call')
            lc = bs_price(S, call_long, T, sigma, opt='call')
            sp = bs_price(S, put_short, T, sigma, opt='put')
            lp = bs_price(S, put_long, T, sigma, opt='put')
            credit = (sc - lc) + (sp - lp)

            max_loss = (call_long - call_short) - credit
            if max_loss * 100 > equity * 0.15 or credit < 0.20:
                continue

            S_post = float(close_series.loc[post_date])
            call_stl = max(0, S_post - call_short) - max(0, S_post - call_long)
            put_stl = max(0, put_short - S_post) - max(0, put_long - S_post)
            settlement = call_stl + put_stl
            pnl = (credit - settlement) * 100 - 2.60

        elif strategy == 'straddle':
            # Buy straddle on high-move names
            if avg_gap < 2:  # Skip low-move names
                continue

            K = round(S)
            call_cost = bs_price(S, K, T, sigma, opt='call')
            put_cost = bs_price(S, K, T, sigma, opt='put')
            total_cost = call_cost + put_cost

            if total_cost * 100 > equity * 0.15:
                continue

            S_post = float(close_series.loc[post_date])
            call_val = max(0, S_post - K)
            put_val = max(0, K - S_post)
            pnl = ((call_val + put_val) - total_cost) * 100 - 1.30

        elif strategy == 'combined':
            # Directional + contrarian filter
            pre_mom = event['pre_momentum']

            if up_rate >= 0.60 and pre_mom < 10:  # Bullish bias, not over-extended
                K_long = round(S)
                K_short = round(S * (1 + spread_pct / 100))
                cost = (bs_price(S, K_long, T, sigma, opt='call') -
                       bs_price(S, K_short, T, sigma, opt='call'))

                if cost <= 0 or cost * 100 > equity * 0.15:
                    continue

                S_post = float(close_series.loc[post_date])
                long_val = max(0, S_post - K_long)
                short_val = max(0, S_post - K_short)
                pnl = ((long_val - short_val) - cost) * 100 - 1.30

            elif up_rate <= 0.40 and pre_mom > -10:  # Bearish bias, not over-sold
                K_long = round(S)
                K_short = round(S * (1 - spread_pct / 100))
                cost = (bs_price(S, K_long, T, sigma, opt='put') -
                       bs_price(S, K_short, T, sigma, opt='put'))

                if cost <= 0 or cost * 100 > equity * 0.15:
                    continue

                S_post = float(close_series.loc[post_date])
                long_val = max(0, K_long - S_post)
                short_val = max(0, K_short - S_post)
                pnl = ((long_val - short_val) - cost) * 100 - 1.30
            else:
                continue
        else:
            continue

        equity += pnl
        trades.append({
            'entry': str(pre_date), 'exit': str(post_date),
            'pnl': round(pnl, 2), 'win': pnl > 0, 'regime': regime,
            'actual_gap': round(actual_gap, 2),
            'hist_up_rate': round(up_rate, 2),
            'pre_momentum': round(event['pre_momentum'], 1),
        })

    if not trades:
        fprint("  No trades!")
        return None

    # Metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100
    total_pnl = sum(t['pnl'] for t in trades)

    tdf = pd.DataFrame(trades)
    tdf['quarter'] = pd.to_datetime(tdf['entry']).dt.to_period('Q')
    quarterly = tdf.groupby('quarter')['pnl'].sum() / capital
    n_years = len(quarterly) / 4

    sharpe = (quarterly.mean() * 4) / (quarterly.std() * np.sqrt(4) + 1e-10) if len(quarterly) > 3 else 0
    cagr = (1 + total_pnl / capital) ** (1 / max(n_years, 0.5)) - 1

    cum = np.cumsum([t['pnl'] for t in trades])
    peak = np.maximum.accumulate(cum + capital)
    dd = (cum + capital - peak) / peak
    maxdd = dd.min()

    pf = abs(sum(t['pnl'] for t in trades if t['win']) / (sum(t['pnl'] for t in trades if not t['win']) + 1e-10))

    bull_t = [t for t in trades if t['regime'] == 'bull']
    bear_t = [t for t in trades if t['regime'] == 'bear']
    bull_wr = sum(1 for t in bull_t if t['win']) / max(len(bull_t), 1) * 100
    bear_wr = sum(1 for t in bear_t if t['win']) / max(len(bear_t), 1) * 100
    r1_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    result = {
        'name': name, 'ticker': ticker, 'strategy': strategy,
        'n_trades': n_trades, 'win_rate': round(wr, 1),
        'total_pnl': round(total_pnl, 2), 'final_equity': round(equity, 2),
        'cagr_pct': round(cagr * 100, 1), 'sharpe': round(sharpe, 2),
        'maxdd_pct': round(maxdd * 100, 1), 'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3), 'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr, 1), 'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_t), 'bear_trades': len(bear_t),
        'quarterly_returns': quarterly.tolist(),
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}")
    fprint(f"    ${capital:,} → ${equity:,.0f} | R1 gap {r1_gap:.3f}")

    return result


def permutation_test(returns, n_perms=1000):
    if len(returns) < 5:
        return 1.0
    real = np.mean(returns) / (np.std(returns) + 1e-10)
    count = sum(1 for _ in range(n_perms)
                if np.mean(returns * np.random.choice([-1, 1], len(returns))) /
                (np.std(returns) + 1e-10) >= real)
    return count / n_perms


def main():
    import yfinance as yf

    fprint(f"Earnings Options Strategy v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Mega-caps with most earnings history
    tickers = ['MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'V', 'PG', 'JNJ', 'UNH', 'JPM']

    fprint("Downloading data...")
    raw = yf.download(tickers + ['SPY', '^VIX'], start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna()
    spy = close['SPY'].dropna()

    fprint(f"Data: {len(spy)} days")

    # Detect earnings for each ticker
    fprint("\nDetecting earnings dates...")
    for ticker in tickers:
        if ticker in close.columns:
            events = get_earnings_dates(ticker, close[ticker].dropna())
            fprint(f"  {ticker}: {len(events)} earnings events detected")

    strategies = ['directional', 'contrarian', 'iron_condor', 'straddle', 'combined']

    results = []

    if MLFLOW_OK:
        exp_name = 'earnings_options_strategy_v1'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except:
            pass
        mlflow.set_experiment(exp_name)

    # Test each strategy on a portfolio of mega-caps
    for strat in strategies:
        fprint(f"\n{'='*40}")
        fprint(f"STRATEGY: {strat.upper()}")
        fprint(f"{'='*40}")

        # Run on each ticker individually
        all_trades_pnl = []
        ticker_results = []

        for ticker in tickers:
            if ticker not in close.columns:
                continue
            tc = close[ticker].dropna()
            common = tc.index.intersection(vix.index).intersection(spy.index)

            r = simulate_earnings_strategy(
                ticker, tc.loc[common], vix.loc[common], spy.loc[common],
                strategy=strat, capital=10000,
                name=f"{strat}_{ticker}"
            )
            if r:
                ticker_results.append(r)

        if not ticker_results:
            continue

        # Aggregate across all tickers (portfolio approach)
        total_trades = sum(r['n_trades'] for r in ticker_results)
        total_pnl = sum(r['total_pnl'] for r in ticker_results)
        total_wins = sum(int(r['n_trades'] * r['win_rate'] / 100) for r in ticker_results)
        avg_sharpe = np.mean([r['sharpe'] for r in ticker_results])
        avg_maxdd = np.mean([r['maxdd_pct'] for r in ticker_results])

        portfolio = {
            'name': f"PORTFOLIO_{strat}",
            'strategy': strat,
            'n_tickers': len(ticker_results),
            'total_trades': total_trades,
            'avg_trades_per_ticker': round(total_trades / len(ticker_results), 1),
            'portfolio_wr': round(total_wins / max(total_trades, 1) * 100, 1),
            'total_pnl': round(total_pnl, 2),
            'avg_sharpe': round(avg_sharpe, 2),
            'avg_maxdd': round(avg_maxdd, 1),
            'avg_pf': round(np.mean([r['pf'] for r in ticker_results]), 2),
            'tickers_profitable': sum(1 for r in ticker_results if r['total_pnl'] > 0),
        }

        # Combine quarterly returns for permutation test
        all_qrets = []
        for r in ticker_results:
            all_qrets.extend(r['quarterly_returns'])
        all_qrets = np.array(all_qrets)

        if len(all_qrets) >= 5:
            portfolio['perm_p'] = round(permutation_test(all_qrets), 3)
            portfolio['g1_pass'] = portfolio['perm_p'] < 0.05
        else:
            portfolio['perm_p'] = 1.0
            portfolio['g1_pass'] = False

        # R1 from individual results
        avg_r1 = np.mean([r['r1_gap'] for r in ticker_results])
        portfolio['avg_r1_gap'] = round(avg_r1, 3)
        portfolio['g2_pass'] = avg_r1 <= 0.50

        results.append(portfolio)

        fprint(f"\n  PORTFOLIO {strat}: {total_trades} trades across {len(ticker_results)} tickers, "
               f"WR {portfolio['portfolio_wr']:.0f}%, Avg Sharpe {avg_sharpe:.2f}, "
               f"Total PnL ${total_pnl:,.0f}")
        fprint(f"    Perm p={portfolio['perm_p']}, R1 gap avg {avg_r1:.3f}")
        fprint(f"    {portfolio['tickers_profitable']}/{len(ticker_results)} tickers profitable")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — Earnings Options Strategy v1")
    fprint("=" * 70)
    fprint(f"{'Strategy':<20} {'Trades':>6} {'WR':>6} {'AvgSharpe':>9} {'TotalPnL':>10} "
           f"{'AvgDD':>7} {'PermP':>6} {'Profitable':>10}")
    fprint("-" * 85)
    for r in sorted(results, key=lambda x: x['avg_sharpe'], reverse=True):
        fprint(f"{r['strategy']:<20} {r['total_trades']:>6} {r['portfolio_wr']:>5.0f}% "
               f"{r['avg_sharpe']:>9.2f} ${r['total_pnl']:>9,.0f} "
               f"{r['avg_maxdd']:>6.1f}% {r['perm_p']:>6.3f} "
               f"{r['tickers_profitable']}/{r['n_tickers']}")

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
