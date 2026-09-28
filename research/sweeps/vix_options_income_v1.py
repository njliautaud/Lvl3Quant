#!/usr/bin/env python3
"""
VIX Options Income Strategy v1
===============================
Systematic VIX mean-reversion income strategy using VIX options.

Approach:
- Sell VIX call spreads when VIX > 20 (mean reversion down = collect premium)
- Buy VIX puts when VIX > 25 (capture mean reversion actively)
- Buy VIX calls when VIX < 15 (cheap crash insurance that pays off on spikes)

Uses ^VIX historical data + Black-Scholes pricing with VIX-of-VIX proxy.

Adversarial gates: permutation (200 shuffles), regime R1, sub-period, outlier.

Per HC #747 (VIX options as trading instrument).
Per HC #741 (minimum 10-15% annual return bar).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Try MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    print("MLflow connected")
except:
    print("MLflow unavailable, continuing without tracking")

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'vix_options_income_v1_results.json'


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_vix_data():
    import yfinance as yf
    vix = yf.download('^VIX', start='2010-01-01', end='2026-07-24', progress=False)
    if vix.index.tz is not None:
        vix.index = vix.index.tz_convert(None)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    spy = yf.download('SPY', start='2010-01-01', end='2026-07-24', progress=False)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    return vix, spy


def compute_vvix_proxy(vix_close, window=20):
    log_ret = np.log(vix_close / vix_close.shift(1))
    vvix = log_ret.rolling(window).std() * np.sqrt(252)
    return vvix


def classify_regime(spy_close):
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series('bull', index=spy_close.index)
    regime[spy_close < sma200] = 'bear'
    return regime


def run_strategy(vix_df, spy_df, strategy_type='call_spread_sell',
                 entry_vix_high=20, entry_vix_low=15,
                 hold_days=21, capital=100000,
                 max_position_pct=0.05):
    vix_close = vix_df['Close'].copy()
    spy_close = spy_df['Close'].reindex(vix_close.index, method='ffill')
    vvix = compute_vvix_proxy(vix_close)
    regime = classify_regime(spy_close)
    r = 0.04

    trades = []
    equity = capital
    equity_curve = pd.Series(dtype=float)
    dates = vix_close.index[252:]

    i = 0
    while i < len(dates):
        date = dates[i]
        vix_val = vix_close.loc[date]
        vvix_val = vvix.get(date, np.nan)
        if pd.isna(vvix_val) or vvix_val <= 0:
            vvix_val = 0.8

        if np.isnan(vix_val):
            equity_curve[date] = equity
            i += 1
            continue

        max_risk = equity * max_position_pct
        trade = None

        if strategy_type in ('call_spread_sell', 'combined'):
            if vix_val > entry_vix_high:
                K_short = round(vix_val)
                K_long = K_short + 5
                T = hold_days / 252
                iv = max(vvix_val * 1.2, 0.5)
                short_call = bs_call(vix_val, K_short, T, r, iv)
                long_call = bs_call(vix_val, K_long, T, r, iv)
                credit = (short_call - long_call) * 100
                max_loss = (K_long - K_short) * 100 - credit

                if credit > 20 and max_loss > 0:
                    n_contracts = max(1, int(max_risk / max_loss))
                    total_credit = credit * n_contracts

                    exit_idx = min(i + hold_days, len(dates) - 1)
                    exit_date = dates[exit_idx]
                    vix_exit = vix_close.loc[exit_date]

                    short_payoff = max(vix_exit - K_short, 0) * 100
                    long_payoff = max(vix_exit - K_long, 0) * 100
                    pnl_per = credit - (short_payoff - long_payoff)
                    pnl = pnl_per * n_contracts

                    commission = n_contracts * 4.0
                    pnl -= commission

                    trade = {
                        'type': 'call_spread_sell',
                        'entry_date': str(date.date()),
                        'exit_date': str(exit_date.date()),
                        'vix_entry': round(float(vix_val), 2),
                        'vix_exit': round(float(vix_exit), 2),
                        'strike_short': K_short,
                        'strike_long': K_long,
                        'credit': round(float(total_credit), 2),
                        'pnl': round(float(pnl), 2),
                        'contracts': n_contracts,
                        'regime': regime.get(date, 'unknown'),
                    }
                    equity += pnl
                    i = exit_idx + 1
                    trades.append(trade)
                    equity_curve[date] = equity
                    continue

        if strategy_type in ('put_buy', 'combined'):
            if vix_val > entry_vix_high and (not trade or strategy_type == 'put_buy'):
                K = round(vix_val) - 2
                T = hold_days / 252
                iv = max(vvix_val * 1.2, 0.5)
                put_price = bs_put(vix_val, K, T, r, iv) * 100

                if put_price > 10:
                    n_contracts = max(1, int(max_risk / put_price))
                    cost = put_price * n_contracts

                    exit_idx = min(i + hold_days, len(dates) - 1)
                    exit_date = dates[exit_idx]
                    vix_exit = vix_close.loc[exit_date]

                    payoff = max(K - vix_exit, 0) * 100 * n_contracts
                    pnl = payoff - cost
                    commission = n_contracts * 2.0
                    pnl -= commission

                    trade = {
                        'type': 'put_buy',
                        'entry_date': str(date.date()),
                        'exit_date': str(exit_date.date()),
                        'vix_entry': round(float(vix_val), 2),
                        'vix_exit': round(float(vix_exit), 2),
                        'strike': K,
                        'cost': round(float(cost), 2),
                        'pnl': round(float(pnl), 2),
                        'contracts': n_contracts,
                        'regime': regime.get(date, 'unknown'),
                    }
                    equity += pnl
                    i = exit_idx + 1
                    trades.append(trade)
                    equity_curve[date] = equity
                    continue

        if strategy_type in ('call_buy_insurance', 'combined'):
            if vix_val < entry_vix_low and (not trade or strategy_type == 'call_buy_insurance'):
                K = round(vix_val) + 3
                T = hold_days / 252
                iv = max(vvix_val * 1.0, 0.4)
                call_price = bs_call(vix_val, K, T, r, iv) * 100

                if call_price > 5:
                    insurance_budget = max_risk * 0.3
                    n_contracts = max(1, int(insurance_budget / call_price))
                    cost = call_price * n_contracts

                    exit_idx = min(i + hold_days, len(dates) - 1)
                    exit_date = dates[exit_idx]
                    vix_exit = vix_close.loc[exit_date]

                    payoff = max(vix_exit - K, 0) * 100 * n_contracts
                    pnl = payoff - cost
                    commission = n_contracts * 2.0
                    pnl -= commission

                    trade = {
                        'type': 'call_buy_insurance',
                        'entry_date': str(date.date()),
                        'exit_date': str(exit_date.date()),
                        'vix_entry': round(float(vix_val), 2),
                        'vix_exit': round(float(vix_exit), 2),
                        'strike': K,
                        'cost': round(float(cost), 2),
                        'pnl': round(float(pnl), 2),
                        'contracts': n_contracts,
                        'regime': regime.get(date, 'unknown'),
                    }
                    equity += pnl
                    i = exit_idx + 1
                    trades.append(trade)
                    equity_curve[date] = equity
                    continue

        equity_curve[date] = equity
        i += 1

    return trades, equity_curve


def compute_metrics(trades, equity_curve, capital=100000):
    if not trades:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'cagr': 0, 'maxdd': 0, 'n_trades': 0,
                'total_pnl': 0, 'avg_pnl': 0, 'final_equity': capital}

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    wr = len(wins) / len(pnls) * 100
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')

    if len(equity_curve) > 1:
        daily_ret = equity_curve.pct_change().dropna()
        if len(daily_ret) > 0 and daily_ret.std() > 0:
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            downside = daily_ret[daily_ret < 0]
            sortino = daily_ret.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
        else:
            sharpe = sortino = 0
    else:
        sharpe = sortino = 0

    final = equity_curve.iloc[-1] if len(equity_curve) > 0 else capital
    years = len(equity_curve) / 252 if len(equity_curve) > 0 else 1
    cagr = ((final / capital) ** (1 / max(years, 0.01)) - 1) * 100

    if len(equity_curve) > 0:
        peak = equity_curve.expanding().max()
        dd = (equity_curve - peak) / peak
        maxdd = dd.min() * 100
    else:
        maxdd = 0

    return {
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'pf': round(float(min(pf, 999)), 2),
        'wr': round(float(wr), 1),
        'cagr': round(float(cagr), 1),
        'maxdd': round(float(maxdd), 1),
        'n_trades': len(trades),
        'total_pnl': round(sum(pnls), 2),
        'avg_pnl': round(float(np.mean(pnls)), 2),
        'final_equity': round(float(final), 2),
    }


def permutation_test(trades, n_perms=200):
    if len(trades) < 10:
        return 1.0, 0

    pnls = np.array([t['pnl'] for t in trades])
    real_sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

    # Proper permutation: shuffle which VIX levels trigger entries
    # Since we can't re-simulate easily, we shuffle P&L signs
    # (which trades would have been wins vs losses if timing was random)
    null_sharpes = []
    for _ in range(n_perms):
        # Randomly flip signs of P&L (simulates random entry direction)
        signs = np.random.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        null_sharpes.append(s)

    p_value = np.mean([ns >= real_sharpe for ns in null_sharpes])
    return float(p_value), float(np.std(null_sharpes))


def regime_test(trades):
    bull = [t['pnl'] for t in trades if t.get('regime') == 'bull']
    bear = [t['pnl'] for t in trades if t.get('regime') == 'bear']

    if len(bull) < 5 or len(bear) < 5:
        return None, None, None, 'SKIP'

    s_bull = np.mean(bull) / np.std(bull) if np.std(bull) > 0 else 0
    s_bear = np.mean(bear) / np.std(bear) if np.std(bear) > 0 else 0

    denom = max(abs(s_bull), abs(s_bear))
    gap = abs(s_bull - s_bear) / denom if denom > 0 else 0

    result = 'PASS' if gap < 0.50 else 'FAIL'
    return round(float(s_bull), 2), round(float(s_bear), 2), round(float(gap), 3), result


def sub_period_test(trades):
    if len(trades) < 10:
        return 'SKIP'
    mid = len(trades) // 2
    return 'PASS' if sum(t['pnl'] for t in trades[:mid]) > 0 and sum(t['pnl'] for t in trades[mid:]) > 0 else 'FAIL'


def outlier_test(trades):
    if len(trades) < 20:
        return 'SKIP'
    pnls = sorted([t['pnl'] for t in trades])
    n_remove = max(1, int(len(pnls) * 0.05))
    return 'PASS' if sum(pnls[:-n_remove]) > 0 else 'FAIL'


def main():
    print("=" * 60)
    print("VIX OPTIONS INCOME STRATEGY v1")
    print("=" * 60)

    print("\nDownloading VIX and SPY data...")
    vix_df, spy_df = get_vix_data()
    print(f"VIX: {len(vix_df)} days ({vix_df.index[0].date()} to {vix_df.index[-1].date()})")
    print(f"Current VIX: {vix_df['Close'].iloc[-1]:.1f}")

    variants = {
        'call_spread_sell_20': {'strategy_type': 'call_spread_sell', 'entry_vix_high': 20, 'hold_days': 21,
                                 'desc': 'Sell call spreads when VIX>20, 21d hold'},
        'call_spread_sell_22': {'strategy_type': 'call_spread_sell', 'entry_vix_high': 22, 'hold_days': 21,
                                 'desc': 'Sell call spreads when VIX>22, 21d hold'},
        'call_spread_sell_25': {'strategy_type': 'call_spread_sell', 'entry_vix_high': 25, 'hold_days': 21,
                                 'desc': 'Sell call spreads when VIX>25, 21d hold'},
        'call_spread_sell_20_14d': {'strategy_type': 'call_spread_sell', 'entry_vix_high': 20, 'hold_days': 14,
                                     'desc': 'Sell call spreads when VIX>20, 14d hold'},
        'put_buy_25': {'strategy_type': 'put_buy', 'entry_vix_high': 25, 'hold_days': 21,
                        'desc': 'Buy puts when VIX>25, 21d hold'},
        'put_buy_30': {'strategy_type': 'put_buy', 'entry_vix_high': 30, 'hold_days': 21,
                        'desc': 'Buy puts when VIX>30, 21d hold'},
        'call_buy_15': {'strategy_type': 'call_buy_insurance', 'entry_vix_low': 15, 'hold_days': 21,
                         'desc': 'Buy calls when VIX<15, 21d hold (insurance)'},
        'call_buy_13': {'strategy_type': 'call_buy_insurance', 'entry_vix_low': 13, 'hold_days': 30,
                         'desc': 'Buy calls when VIX<13, 30d hold (insurance)'},
        'combined_20_15': {'strategy_type': 'combined', 'entry_vix_high': 20, 'entry_vix_low': 15, 'hold_days': 21,
                            'desc': 'Combined: sell spreads VIX>20 + buy calls VIX<15'},
        'combined_22_14': {'strategy_type': 'combined', 'entry_vix_high': 22, 'entry_vix_low': 14, 'hold_days': 21,
                            'desc': 'Combined: sell spreads VIX>22 + buy calls VIX<14'},
    }

    results = {}
    best_sharpe = -999
    best_variant = None

    if MLFLOW_OK:
        try:
            mlflow.set_experiment('vix_options_income_v1')
        except:
            pass

    for name, params in variants.items():
        print(f"\n--- {name}: {params['desc']} ---")

        trades, eq_curve = run_strategy(
            vix_df, spy_df,
            strategy_type=params['strategy_type'],
            entry_vix_high=params.get('entry_vix_high', 20),
            entry_vix_low=params.get('entry_vix_low', 15),
            hold_days=params.get('hold_days', 21),
        )

        metrics = compute_metrics(trades, eq_curve)

        perm_p, perm_std = permutation_test(trades)
        bull_s, bear_s, r1_gap, r1_result = regime_test(trades)
        sub_result = sub_period_test(trades)
        outlier_result = outlier_test(trades)

        gates = {
            'permutation': {'p_value': perm_p, 'result': 'PASS' if perm_p < 0.05 else 'FAIL'},
            'regime_r1': {'bull_sharpe': bull_s, 'bear_sharpe': bear_s, 'gap': r1_gap, 'result': r1_result},
            'sub_period': sub_result,
            'outlier': outlier_result,
        }
        gates_passed = sum([
            1 if gates['permutation']['result'] == 'PASS' else 0,
            1 if r1_result == 'PASS' else 0,
            1 if sub_result == 'PASS' else 0,
            1 if outlier_result == 'PASS' else 0,
        ])

        results[name] = {
            'description': params['desc'],
            'metrics': metrics,
            'gates': gates,
            'gates_passed': f"{gates_passed}/4",
        }

        print(f"  Trades: {metrics['n_trades']}, WR: {metrics['wr']}%, PF: {metrics['pf']}")
        print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, CAGR: {metrics['cagr']}%")
        print(f"  MaxDD: {metrics['maxdd']}%, Final: ${metrics['final_equity']:,.0f}")
        print(f"  Gates: {gates_passed}/4 (perm={'PASS' if perm_p<0.05 else 'FAIL'} p={perm_p:.3f}, "
              f"R1={r1_result}, sub={sub_result}, outlier={outlier_result})")

        if metrics['sharpe'] > best_sharpe and metrics['n_trades'] >= 10:
            best_sharpe = metrics['sharpe']
            best_variant = name

        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f'vix_{name}'):
                    mlflow.log_params({
                        'strategy_type': params['strategy_type'],
                        'entry_vix_high': str(params.get('entry_vix_high', 'N/A')),
                        'entry_vix_low': str(params.get('entry_vix_low', 'N/A')),
                        'hold_days': params.get('hold_days', 21),
                    })
                    mlflow.log_metrics({
                        'sharpe': metrics['sharpe'],
                        'sortino': metrics['sortino'],
                        'pf': min(metrics['pf'], 999),
                        'wr': metrics['wr'],
                        'cagr': metrics['cagr'],
                        'maxdd': metrics['maxdd'],
                        'n_trades': metrics['n_trades'],
                        'perm_p': perm_p,
                        'gates_passed': gates_passed,
                    })
            except:
                pass

    print("\n" + "=" * 60)
    print("SUMMARY — VIX OPTIONS INCOME v1")
    print("=" * 60)
    print(f"\nBest variant: {best_variant} (Sharpe {best_sharpe:.2f})")

    ranked = sorted(results.items(), key=lambda x: x[1]['metrics']['sharpe'], reverse=True)
    print(f"\n{'Variant':<30} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'#Trades':>8} {'Gates':>6}")
    print("-" * 80)
    for name, r in ranked:
        m = r['metrics']
        print(f"{name:<30} {m['sharpe']:>7.2f} {m['cagr']:>6.1f}% {m['maxdd']:>6.1f}% {m['wr']:>5.1f}% {m['n_trades']:>8} {r['gates_passed']:>6}")

    output = {
        'strategy': 'VIX Options Income v1',
        'run_date': str(datetime.now()),
        'data_range': f"{vix_df.index[0].date()} to {vix_df.index[-1].date()}",
        'best_variant': best_variant,
        'best_sharpe': best_sharpe,
        'variants': results,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved.")
    return output


if __name__ == '__main__':
    results = main()
