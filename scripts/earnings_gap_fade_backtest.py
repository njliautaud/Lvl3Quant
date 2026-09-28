#!/usr/bin/env python3
"""
Earnings Gap Fade Backtest — Contrarian strategy
Academic basis: Stocks gapping DOWN on earnings (not catastrophically) tend to recover
over 10-20 days when selloff is sentiment-driven rather than fundamental.

6 Variants:
A. Small Gap Fade: Buy gap-down 3-10%, hold 10d
B. Recovery Filter: Same + must recover 1% from day low by close, hold 10d
C. Quality Dip: Gap-down 3-10% + above 200-SMA pre-earnings, hold 20d
D. Volume Exhaustion: Gap-down + volume >3x 20d avg, hold 10d
E. Sector Leaders Only: Fade gaps in mega-cap leaders only, hold 20d
F. Portfolio Fade: Monthly top-3 worst earnings reactions, equal weight, hold 20d

Universe: 24 major US stocks
OOT: Jan 2022 - Jul 2026
Starting capital: $645, $0 commission, 0.02% slippage
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
    'CRM', 'ADBE', 'NFLX', 'AVGO', 'COST', 'PEP', 'LLY', 'UNH',
    'V', 'MA', 'JPM', 'HD', 'INTC', 'MU', 'QCOM', 'PYPL'
]

SECTOR_LEADERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']

START_DATE = '2021-06-01'  # extra buffer for 200-SMA warmup
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%

N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/earnings_gap_fade_results.json'


def download_data():
    """Download price/volume data for universe + SPY."""
    tickers = UNIVERSE + ['SPY']
    print(f"Downloading data for {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 100:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
    print(f"  Downloaded {len(data)} tickers successfully.")
    return data


def compute_sma(series, window):
    return series.rolling(window=window, min_periods=window).mean()


def identify_earnings_events(data):
    """
    Proxy for earnings days: |daily return| > 3% AND volume > 2x 20-day average.
    Returns list of (ticker, date, gap_pct, volume_ratio, day_low, day_close, prev_close).
    """
    events = []
    for ticker, df in data.items():
        if ticker == 'SPY':
            continue
        df = df.copy()
        df['Return'] = df['Close'].pct_change()
        df['VolMA20'] = df['Volume'].rolling(20).mean()
        df['SMA200'] = compute_sma(df['Close'], 200)

        for i in range(21, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            ret = row['Return']
            vol_ratio = row['Volume'] / row['VolMA20'] if row['VolMA20'] > 0 else 0

            if abs(ret) > 0.03 and vol_ratio > 2.0:
                events.append({
                    'ticker': ticker,
                    'date': df.index[i],
                    'gap_pct': ret,
                    'volume_ratio': vol_ratio,
                    'day_low': row['Low'],
                    'day_close': row['Close'],
                    'prev_close': prev['Close'],
                    'pre_earnings_sma200': prev['SMA200'],
                    'pre_earnings_close': prev['Close'],
                })
    return events


def compute_forward_return(data, ticker, entry_date, hold_days):
    """Compute forward return from entry_date over hold_days trading days."""
    df = data.get(ticker)
    if df is None:
        return None
    dates = df.index
    idx = dates.get_loc(entry_date) if entry_date in dates else None
    if idx is None:
        # find nearest
        mask = dates >= entry_date
        if not mask.any():
            return None
        idx = mask.argmax()

    exit_idx = min(idx + hold_days, len(dates) - 1)
    if exit_idx <= idx:
        return None

    entry_price = df.iloc[idx]['Close'] * (1 + SLIPPAGE_PCT)  # buy slippage
    exit_price = df.iloc[exit_idx]['Close'] * (1 - SLIPPAGE_PCT)  # sell slippage
    return (exit_price - entry_price) / entry_price


def get_spy_regime(spy_df, date):
    """Bull if SPY > 200-SMA, else Bear."""
    if date not in spy_df.index:
        # find nearest prior
        prior = spy_df.index[spy_df.index <= date]
        if len(prior) == 0:
            return 'Bull'
        date = prior[-1]
    row_idx = spy_df.index.get_loc(date)
    if row_idx < 200:
        return 'Bull'
    sma200 = spy_df['Close'].iloc[max(0, row_idx-199):row_idx+1].mean()
    return 'Bull' if spy_df['Close'].iloc[row_idx] > sma200 else 'Bear'


def run_variant(variant_name, events, data, spy_df):
    """Run a single variant and return trades list."""
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    if variant_name == 'A':
        # Small Gap Fade: gap down 3-10%, hold 10d
        for ev in events:
            if ev['date'] < oot_start or ev['date'] > oot_end:
                continue
            gap = ev['gap_pct']
            if -0.10 <= gap <= -0.03:
                ret = compute_forward_return(data, ev['ticker'], ev['date'], 10)
                if ret is not None:
                    regime = get_spy_regime(spy_df, ev['date'])
                    trades.append({'ticker': ev['ticker'], 'date': str(ev['date'].date()),
                                   'gap_pct': gap, 'return': ret, 'regime': regime, 'hold': 10})

    elif variant_name == 'B':
        # Recovery Filter: gap down 3-10% + recover 1% from low by close
        for ev in events:
            if ev['date'] < oot_start or ev['date'] > oot_end:
                continue
            gap = ev['gap_pct']
            if -0.10 <= gap <= -0.03:
                recovery = (ev['day_close'] - ev['day_low']) / ev['day_low'] if ev['day_low'] > 0 else 0
                if recovery >= 0.01:
                    ret = compute_forward_return(data, ev['ticker'], ev['date'], 10)
                    if ret is not None:
                        regime = get_spy_regime(spy_df, ev['date'])
                        trades.append({'ticker': ev['ticker'], 'date': str(ev['date'].date()),
                                       'gap_pct': gap, 'return': ret, 'regime': regime, 'hold': 10})

    elif variant_name == 'C':
        # Quality Dip: gap down 3-10% + above 200-SMA before earnings, hold 20d
        for ev in events:
            if ev['date'] < oot_start or ev['date'] > oot_end:
                continue
            gap = ev['gap_pct']
            if -0.10 <= gap <= -0.03:
                sma200 = ev.get('pre_earnings_sma200')
                pre_close = ev.get('pre_earnings_close')
                if sma200 is not None and pre_close is not None and not np.isnan(sma200) and pre_close > sma200:
                    ret = compute_forward_return(data, ev['ticker'], ev['date'], 20)
                    if ret is not None:
                        regime = get_spy_regime(spy_df, ev['date'])
                        trades.append({'ticker': ev['ticker'], 'date': str(ev['date'].date()),
                                       'gap_pct': gap, 'return': ret, 'regime': regime, 'hold': 20})

    elif variant_name == 'D':
        # Volume Exhaustion: gap down + volume >3x 20d avg, hold 10d
        for ev in events:
            if ev['date'] < oot_start or ev['date'] > oot_end:
                continue
            gap = ev['gap_pct']
            if -0.10 <= gap <= -0.03 and ev['volume_ratio'] > 3.0:
                ret = compute_forward_return(data, ev['ticker'], ev['date'], 10)
                if ret is not None:
                    regime = get_spy_regime(spy_df, ev['date'])
                    trades.append({'ticker': ev['ticker'], 'date': str(ev['date'].date()),
                                   'gap_pct': gap, 'return': ret, 'regime': regime, 'hold': 10})

    elif variant_name == 'E':
        # Sector Leaders Only: only AAPL, MSFT, GOOGL, AMZN, NVDA, META, hold 20d
        for ev in events:
            if ev['date'] < oot_start or ev['date'] > oot_end:
                continue
            if ev['ticker'] not in SECTOR_LEADERS:
                continue
            gap = ev['gap_pct']
            if -0.10 <= gap <= -0.03:
                ret = compute_forward_return(data, ev['ticker'], ev['date'], 20)
                if ret is not None:
                    regime = get_spy_regime(spy_df, ev['date'])
                    trades.append({'ticker': ev['ticker'], 'date': str(ev['date'].date()),
                                   'gap_pct': gap, 'return': ret, 'regime': regime, 'hold': 20})

    elif variant_name == 'F':
        # Portfolio Fade: each month, buy up to 3 worst earnings reactions, equal weight, hold 20d
        monthly_events = {}
        for ev in events:
            if ev['date'] < oot_start or ev['date'] > oot_end:
                continue
            gap = ev['gap_pct']
            if gap < -0.03:  # any gap down >3%
                month_key = ev['date'].strftime('%Y-%m')
                if month_key not in monthly_events:
                    monthly_events[month_key] = []
                monthly_events[month_key].append(ev)

        for month_key in sorted(monthly_events.keys()):
            month_evs = sorted(monthly_events[month_key], key=lambda x: x['gap_pct'])  # most negative first
            selected = month_evs[:3]
            for ev in selected:
                ret = compute_forward_return(data, ev['ticker'], ev['date'], 20)
                if ret is not None:
                    regime = get_spy_regime(spy_df, ev['date'])
                    trades.append({'ticker': ev['ticker'], 'date': str(ev['date'].date()),
                                   'gap_pct': ev['gap_pct'], 'return': ret, 'regime': regime, 'hold': 20})

    return trades


def compute_metrics(trades, starting_capital):
    """Compute strategy metrics from list of trades."""
    if not trades:
        return None

    returns = np.array([t['return'] for t in trades])
    n_trades = len(returns)

    # Equity curve (sequential, equal allocation per trade)
    equity = starting_capital
    equity_curve = [starting_capital]
    for r in returns:
        equity *= (1 + r)
        equity_curve.append(equity)
    equity_curve = np.array(equity_curve)

    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    win_rate = np.mean(returns > 0)
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0
    avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0
    profit_factor = abs(np.sum(returns[returns > 0]) / np.sum(returns[returns < 0])) if np.any(returns < 0) and np.sum(returns[returns < 0]) != 0 else np.inf

    # Annualized Sharpe (assume ~60 trades/year as rough normalization, or use actual)
    # More honest: use per-trade returns, annualize by sqrt(trades_per_year)
    years = max((pd.Timestamp(OOT_END) - pd.Timestamp(OOT_START)).days / 365.25, 0.5)
    trades_per_year = n_trades / years
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-9 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-9 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = np.min(dd)

    # Regime breakdown
    bull_returns = [t['return'] for t in trades if t['regime'] == 'Bull']
    bear_returns = [t['return'] for t in trades if t['regime'] == 'Bear']
    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_returns) > 1:
        bull_sharpe = (np.mean(bull_returns) / np.std(bull_returns, ddof=1)) * np.sqrt(max(len(bull_returns)/years, 1))
    if len(bear_returns) > 1:
        bear_sharpe = (np.mean(bear_returns) / np.std(bear_returns, ddof=1)) * np.sqrt(max(len(bear_returns)/years, 1))

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    return {
        'n_trades': n_trades,
        'total_return_pct': round(total_return * 100, 2),
        'final_equity': round(equity_curve[-1], 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3) if profit_factor != np.inf else 999.0,
        'win_rate': round(win_rate * 100, 2),
        'avg_win_pct': round(avg_win * 100, 2),
        'avg_loss_pct': round(avg_loss * 100, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_trades': len(bull_returns),
        'bear_trades': len(bear_returns),
        'returns': returns.tolist(),
    }


def permutation_test(returns, n_perms=1000):
    """Test if mean return is significantly different from 0 by shuffling signs."""
    if len(returns) < 5:
        return 1.0
    observed = np.mean(returns)
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(returns))
        perm_mean = np.mean(returns * signs)
        if perm_mean >= observed:
            count += 1
    return count / n_perms


def validate_5gate(metrics, perm_p):
    """Apply 5-gate validation."""
    gates = {}
    gates['sharpe_gt_0.5'] = metrics['sharpe'] > 0.5
    gates['perm_p_lt_0.05'] = perm_p < 0.05
    gates['regime_gap_lt_0.5'] = metrics['regime_gap'] < 0.5
    gates['max_dd_gt_neg50'] = metrics['max_dd_pct'] > -50
    gates['min_20_trades'] = metrics['n_trades'] >= 20
    gates['all_pass'] = all(gates.values())
    return gates


def main():
    print("=" * 80)
    print("EARNINGS GAP FADE BACKTEST — Contrarian Overreaction Strategy")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print("=" * 80)

    # Download data
    data = download_data()
    spy_df = data.get('SPY')
    if spy_df is None:
        print("ERROR: Could not download SPY data.")
        return

    # Identify earnings events
    events = identify_earnings_events(data)
    gap_down_events = [e for e in events if e['gap_pct'] < -0.03]
    print(f"\nIdentified {len(events)} total earnings-proxy events, {len(gap_down_events)} gap-down events (>3%)")

    # Show date range distribution
    oot_events = [e for e in gap_down_events if pd.Timestamp(OOT_START) <= e['date'] <= pd.Timestamp(OOT_END)]
    print(f"Gap-down events in OOT window: {len(oot_events)}")

    variant_names = {
        'A': 'Small Gap Fade (3-10% down, 10d hold)',
        'B': 'Recovery Filter (+1% bounce from low, 10d hold)',
        'C': 'Quality Dip (above 200-SMA, 20d hold)',
        'D': 'Volume Exhaustion (vol >3x avg, 10d hold)',
        'E': 'Sector Leaders Only (mega-caps, 20d hold)',
        'F': 'Portfolio Fade (top-3 worst/month, 20d hold)',
    }

    results = {}
    all_results = {}

    print("\n" + "=" * 80)
    print("VARIANT RESULTS")
    print("=" * 80)

    for var_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        trades = run_variant(var_key, events, data, spy_df)
        var_label = f"Variant {var_key}: {variant_names[var_key]}"

        if not trades:
            print(f"\n{var_label}")
            print("  NO TRADES — skipped")
            results[var_key] = {'variant': var_label, 'status': 'NO_TRADES'}
            continue

        metrics = compute_metrics(trades, STARTING_CAPITAL)
        returns_arr = np.array(metrics['returns'])
        perm_p = permutation_test(returns_arr, N_PERMUTATIONS)
        gates = validate_5gate(metrics, perm_p)

        # Remove raw returns from saved metrics (too large)
        metrics_save = {k: v for k, v in metrics.items() if k != 'returns'}
        metrics_save['perm_p_value'] = round(perm_p, 4)
        metrics_save['gates'] = gates

        results[var_key] = {
            'variant': var_label,
            'metrics': metrics_save,
            'trades': trades,
        }

        status = "PASS" if gates['all_pass'] else "FAIL"

        print(f"\n{'─' * 70}")
        print(f"  {var_label}")
        print(f"{'─' * 70}")
        print(f"  Trades: {metrics['n_trades']}  |  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Total Return: {metrics['total_return_pct']:+.1f}%  |  Final Equity: ${metrics['final_equity']:.2f}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}  |  PF: {metrics['profit_factor']:.2f}")
        print(f"  Avg Win: {metrics['avg_win_pct']:+.2f}%  |  Avg Loss: {metrics['avg_loss_pct']:+.2f}%")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f} ({metrics['bull_trades']} trades)")
        print(f"  Bear Sharpe: {metrics['bear_sharpe']:.3f} ({metrics['bear_trades']} trades)")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f}")
        print(f"  Permutation p-value: {perm_p:.4f}")
        print(f"  5-Gate: {status}")
        for gate, passed in gates.items():
            if gate != 'all_pass':
                marker = "PASS" if passed else "FAIL"
                print(f"    [{marker}] {gate}")

        # Collect for summary
        all_results[var_key] = {
            'name': variant_names[var_key],
            'sharpe': metrics['sharpe'],
            'sortino': metrics['sortino'],
            'win_rate': metrics['win_rate'],
            'total_return': metrics['total_return_pct'],
            'max_dd': metrics['max_dd_pct'],
            'n_trades': metrics['n_trades'],
            'pf': metrics['profit_factor'],
            'perm_p': perm_p,
            'regime_gap': metrics['regime_gap'],
            '5gate': status,
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Var':<4} {'Name':<42} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'Return%':>8} {'MaxDD%':>7} {'Trades':>6} {'Gate':>5}")
    print("-" * 95)
    for var_key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if var_key in all_results:
            r = all_results[var_key]
            print(f"{var_key:<4} {r['name']:<42} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate']:>5.1f}% {r['total_return']:>+7.1f}% {r['max_dd']:>6.1f}% {r['n_trades']:>6} {r['5gate']:>5}")
        else:
            print(f"{var_key:<4} NO TRADES")

    # Save results (without raw trade lists for JSON size)
    save_results = {}
    for var_key, res in results.items():
        save_entry = {
            'variant': res.get('variant', ''),
            'status': res.get('status', 'OK'),
        }
        if 'metrics' in res:
            save_entry['metrics'] = res['metrics']
            save_entry['n_trades'] = len(res.get('trades', []))
            # Save sample trades (first 5)
            save_entry['sample_trades'] = res.get('trades', [])[:5]
        save_results[var_key] = save_entry

    save_payload = {
        'strategy': 'Earnings Gap Fade (Contrarian)',
        'oot_period': f'{OOT_START} to {OOT_END}',
        'starting_capital': STARTING_CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'universe_size': len(UNIVERSE),
        'total_gap_down_events_oot': len(oot_events),
        'run_timestamp': datetime.now().isoformat(),
        'variants': save_results,
        'summary': all_results,
    }

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_payload, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("Done.")


if __name__ == '__main__':
    main()
