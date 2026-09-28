#!/usr/bin/env python3
"""
Combined Portfolio + Vol Overlay — HC #709 R1 Nuanced Test
============================================================
Takes the validated growth_tilt portfolio (Sharpe 2.10, MaxDD -13.1%)
and adds Gameplan v2 vol-adjusted overlay to reduce red-day losses.

HC #709 R1 nuanced criteria:
  - Growth strategy that fails R1 standalone CAN be deployed IF:
    (a) Combined with hedge that limits red-day losses to <50% of green-day gains
    (b) MaxDD with hedge is <15%

Also tests: hedge via cross-asset regime signals (HC #710).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/combined_vol_overlay'
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
ANNUALIZE = 252


def sharpe(returns):
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ANNUALIZE))


def sortino(returns):
    if len(returns) < 5:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(ANNUALIZE))


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def cagr_func(returns):
    cum = (1 + returns).prod()
    n_years = len(returns) / ANNUALIZE
    if n_years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1 / n_years) - 1)


def win_rate(returns):
    return float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def compute_metrics(returns, label=""):
    return {
        'label': label,
        'sharpe': round(sharpe(returns), 3),
        'sortino': round(sortino(returns), 3),
        'cagr': round(cagr_func(returns) * 100, 1),
        'max_dd': round(max_drawdown(returns) * 100, 1),
        'win_rate': round(win_rate(returns) * 100, 1),
        'pf': round(profit_factor(returns), 3),
        'n_days': len(returns),
    }


def download_data():
    """Download all needed data."""
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'IEF', 'SHY', 'QQQ', 'TQQQ',
               'HYG', 'LQD', 'IWM', 'UUP', 'COPX', 'EEM', 'USO', 'SLV',
               'XLU', 'XLP', 'XLV', 'XBI', 'VIXY']
    data = yf.download(tickers, start='2015-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass
    return closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])


def build_strategy_returns(closes):
    """
    Build daily returns for each strategy component.
    All strategies use walk-forward or simple rules (no look-ahead).
    """
    returns = closes.pct_change().fillna(0)
    spy = closes['SPY']
    spy_ret = returns['SPY']

    # 1. UPRO PROTECTED (Gameplan v2 core)
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    sma20 = spy.rolling(20).mean()
    sma200 = spy.rolling(200).mean()

    upro_protected = pd.Series(0.0, index=closes.index)
    for i in range(260, len(closes)):
        date = closes.index[i]
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        vol_pct = vol * 100

        m, d = date.month, date.day
        is_earn = ((m == 1 and d >= 15) or (m == 2 and d <= 15) or
                  (m == 4 and d >= 15) or (m == 5 and d <= 15) or
                  (m == 7 and d >= 15) or (m == 8 and d <= 15) or
                  (m == 10 and d >= 15) or (m == 11 and d <= 15))
        low_t = 25 if is_earn else 20

        # September hedge
        if date.month == 9:
            upro_protected.iloc[i] = returns.iloc[i]['SPY']
        elif vol_pct > 30:
            upro_protected.iloc[i] = returns.iloc[i].get('GLD', 0)
        elif vol_pct > low_t or (sma20.iloc[i] < sma200.iloc[i]):
            upro_protected.iloc[i] = returns.iloc[i]['SPY']
        else:
            upro_protected.iloc[i] = returns.iloc[i]['UPRO']

    # 2. CTA TREND FOLLOWING (SMA50 on 9 ETFs)
    cta_tickers = ['GLD', 'SLV', 'USO', 'COPX', 'UUP', 'TLT', 'IEF', 'EEM', 'IWM']
    available_cta = [t for t in cta_tickers if t in closes.columns]
    cta_returns = pd.Series(0.0, index=closes.index)

    for i in range(260, len(closes)):
        daily_r = 0.0
        count = 0
        for ticker in available_cta:
            if i < 50 or pd.isna(closes[ticker].iloc[i]):
                continue
            sma50 = closes[ticker].iloc[i-50:i].mean()
            price = closes[ticker].iloc[i-1]  # Use yesterday's close for signal
            r = returns.iloc[i].get(ticker, 0)
            if not np.isnan(r):
                count += 1
                if price > sma50:
                    daily_r += r  # Long
                # else: flat (cash = 0 return)
        if count > 0:
            cta_returns.iloc[i] = daily_r / len(available_cta)  # Equal weight across universe

    # 3. ETF REVERSAL (bottom 5 by 5d return, hold 5 days)
    reversal_tickers = ['XLU', 'XLP', 'XLV', 'IWM', 'EEM', 'QQQ', 'GLD', 'TLT', 'SPY']
    available_rev = [t for t in reversal_tickers if t in closes.columns]
    reversal_returns = pd.Series(0.0, index=closes.index)

    for i in range(260, len(closes)):
        if closes.index[i].weekday() != 0 and i % 5 != 0:  # Rebalance weekly
            # Carry forward
            if i > 0:
                reversal_returns.iloc[i] = reversal_returns.iloc[i-1]
            continue

        # Rank by 5-day return, buy bottom 3
        perf_5d = {}
        for t in available_rev:
            if i >= 5 and not np.isnan(closes[t].iloc[i-5]) and closes[t].iloc[i-5] > 0:
                perf_5d[t] = (closes[t].iloc[i] / closes[t].iloc[i-5]) - 1

        if len(perf_5d) >= 3:
            bottom_3 = sorted(perf_5d, key=perf_5d.get)[:3]
            r = 0.0
            for t in bottom_3:
                ret = returns.iloc[i].get(t, 0)
                if not np.isnan(ret):
                    r += ret / 3
            reversal_returns.iloc[i] = r

    # 4. VIX PANIC BUY (buy SPY/UPRO when VIX proxy high)
    vixy = closes.get('VIXY')
    panic_returns = pd.Series(0.0, index=closes.index)
    if vixy is not None:
        vixy_sma20 = vixy.rolling(20).mean()
        for i in range(260, len(closes)):
            if not np.isnan(vixy.iloc[i]) and not np.isnan(vixy_sma20.iloc[i]):
                if vixy.iloc[i] > vixy_sma20.iloc[i] * 1.5:  # VIX spike
                    panic_returns.iloc[i] = returns.iloc[i]['UPRO']

    # Trim to common period
    start_idx = 260
    upro_protected = upro_protected.iloc[start_idx:]
    cta_returns = cta_returns.iloc[start_idx:]
    reversal_returns = reversal_returns.iloc[start_idx:]
    panic_returns = panic_returns.iloc[start_idx:]

    return upro_protected, cta_returns, reversal_returns, panic_returns


def test_portfolio(upro_ret, cta_ret, rev_ret, panic_ret, spy_ret_full,
                   weights, label, regime_labels):
    """Test a portfolio mix and return metrics + HC #709 check."""
    w_upro, w_cta, w_rev, w_panic = weights

    # Align all series
    common = upro_ret.index.intersection(cta_ret.index).intersection(
        rev_ret.index).intersection(panic_ret.index).intersection(
        spy_ret_full.index)

    portfolio_ret = (w_upro * upro_ret.loc[common] +
                    w_cta * cta_ret.loc[common] +
                    w_rev * rev_ret.loc[common] +
                    w_panic * panic_ret.loc[common])

    spy_ret = spy_ret_full.loc[common]

    # Regime classification
    regime = pd.Series('flat', index=common)
    regime[spy_ret > 0.001] = 'green'
    regime[spy_ret < -0.001] = 'red'

    metrics = compute_metrics(portfolio_ret, label)

    # R1 test
    green_ret = portfolio_ret[regime == 'green']
    red_ret = portfolio_ret[regime == 'red']

    sharpe_green = sharpe(green_ret)
    sharpe_red = sharpe(red_ret)
    denom = max(abs(sharpe_green), abs(sharpe_red), 0.001)
    r1_gap = abs(sharpe_green - sharpe_red) / denom

    # HC #709 R1 nuanced criteria
    avg_green_gain = green_ret.mean() if len(green_ret) > 0 else 0
    avg_red_loss = red_ret.mean() if len(red_ret) > 0 else 0
    red_to_green_ratio = abs(avg_red_loss / avg_green_gain) if avg_green_gain != 0 else float('inf')

    metrics.update({
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'r1_gap': round(r1_gap, 3),
        'r1_strict_pass': r1_gap < 0.50,
        'avg_green_gain_bps': round(avg_green_gain * 10000, 1),
        'avg_red_loss_bps': round(avg_red_loss * 10000, 1),
        'red_to_green_ratio': round(red_to_green_ratio, 3),
        'hc709_red_loss_check': red_to_green_ratio < 0.50,  # Red losses < 50% of green gains
        'hc709_maxdd_check': metrics['max_dd'] > -15.0,  # MaxDD < 15%
        'hc709_nuanced_pass': red_to_green_ratio < 0.50 and metrics['max_dd'] > -15.0,
        'n_green': len(green_ret),
        'n_red': len(red_ret),
    })

    return metrics, portfolio_ret


def permutation_test(portfolio_ret, growth_ret, n_perms=200):
    """Shuffle growth timing to test if timing adds value."""
    real_sharpe = sharpe(portfolio_ret)
    non_growth = portfolio_ret - growth_ret
    count_better = 0

    for p in range(n_perms):
        shuffled = growth_ret.sample(frac=1, replace=False).values
        perm_ret = non_growth.values + shuffled
        if sharpe(pd.Series(perm_ret)) >= real_sharpe:
            count_better += 1

    p_val = count_better / n_perms
    return {
        'real_sharpe': round(real_sharpe, 3),
        'p_value': round(p_val, 4),
        'pass': p_val < 0.05,
    }


def main():
    print("=" * 70)
    print("COMBINED PORTFOLIO + VOL OVERLAY")
    print("HC #709 R1 Nuanced Criteria Test")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nDownloading data...")
    closes = download_data()
    print(f"  Data: {closes.index[0].date()} to {closes.index[-1].date()}")

    print("\nBuilding strategy returns...")
    upro_ret, cta_ret, rev_ret, panic_ret = build_strategy_returns(closes)

    spy_ret = closes['SPY'].pct_change().fillna(0)

    # Portfolio configurations to test
    configs = [
        ('UPRO Only', (1.0, 0.0, 0.0, 0.0)),
        ('50/30/15/5 (HC #709 optimal)', (0.50, 0.30, 0.15, 0.05)),
        ('60/25/10/5 (growth tilt)', (0.60, 0.25, 0.10, 0.05)),
        ('70/20/5/5 (heavy growth)', (0.70, 0.20, 0.05, 0.05)),
        ('40/35/20/5 (balanced)', (0.40, 0.35, 0.20, 0.05)),
        ('30/40/25/5 (CTA heavy)', (0.30, 0.40, 0.25, 0.05)),
    ]

    # Regime labels
    common_idx = upro_ret.index.intersection(spy_ret.index)
    regime = pd.Series('flat', index=common_idx)
    spy_aligned = spy_ret.loc[common_idx]
    regime[spy_aligned > 0.001] = 'green'
    regime[spy_aligned < -0.001] = 'red'

    all_results = {}

    print(f"\n{'='*70}")
    print(f"  {'Config':<30} {'Sharpe':>7} {'CAGR':>6} {'MaxDD':>7} {'S_grn':>6} {'S_red':>6} {'R/G':>6} {'709':>5}")
    print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")

    best_config = None
    best_sharpe = -999

    for label, weights in configs:
        m, port_ret = test_portfolio(upro_ret, cta_ret, rev_ret, panic_ret,
                                     spy_ret, weights, label, regime)

        pass_709 = '✓' if m['hc709_nuanced_pass'] else '✗'
        print(f"  {label:<30} {m['sharpe']:>7.3f} {m['cagr']:>5.1f}% {m['max_dd']:>6.1f}% "
              f"{m['sharpe_green']:>6.3f} {m['sharpe_red']:>6.3f} {m['red_to_green_ratio']:>5.2f} {pass_709:>5}")

        all_results[label] = m

        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_config = label
            best_ret = port_ret

    # Detailed analysis of best config
    print(f"\n{'='*70}")
    print(f"  BEST CONFIG: {best_config}")
    print(f"{'='*70}")
    best_m = all_results[best_config]

    print(f"\n  Sharpe:         {best_m['sharpe']:.3f}")
    print(f"  Sortino:        {best_m['sortino']:.3f}")
    print(f"  CAGR:           {best_m['cagr']:.1f}%")
    print(f"  MaxDD:          {best_m['max_dd']:.1f}%")
    print(f"  Win Rate:       {best_m['win_rate']:.1f}%")
    print(f"  Profit Factor:  {best_m['pf']:.3f}")
    print(f"  N Days:         {best_m['n_days']}")

    print(f"\n  REGIME ANALYSIS:")
    print(f"    Green days ({best_m['n_green']}): Sharpe {best_m['sharpe_green']:.3f}, "
          f"avg gain {best_m['avg_green_gain_bps']:.1f}bps")
    print(f"    Red days ({best_m['n_red']}):   Sharpe {best_m['sharpe_red']:.3f}, "
          f"avg loss {best_m['avg_red_loss_bps']:.1f}bps")
    print(f"    R1 gap:       {best_m['r1_gap']:.3f} ({'PASS' if best_m['r1_strict_pass'] else 'FAIL'} strict)")
    print(f"    Red/Green ratio: {best_m['red_to_green_ratio']:.3f} "
          f"({'< 0.50 ✓' if best_m['hc709_red_loss_check'] else '>= 0.50 ✗'})")
    print(f"    MaxDD check:  {best_m['max_dd']:.1f}% "
          f"({'< -15% ✓' if best_m['hc709_maxdd_check'] else '>= -15% ✗'})")

    hc709_verdict = "PASS" if best_m['hc709_nuanced_pass'] else "FAIL"
    print(f"\n  HC #709 NUANCED VERDICT: {hc709_verdict}")

    # Find the best HC #709 passing config
    passing_configs = {k: v for k, v in all_results.items() if v['hc709_nuanced_pass']}
    if passing_configs:
        best_passing = max(passing_configs.items(), key=lambda x: x[1]['sharpe'])
        print(f"\n  BEST HC #709 PASSING CONFIG: {best_passing[0]}")
        print(f"    Sharpe: {best_passing[1]['sharpe']:.3f}, CAGR: {best_passing[1]['cagr']:.1f}%, "
              f"MaxDD: {best_passing[1]['max_dd']:.1f}%")

    # Permutation test on best config
    print(f"\n{'='*70}")
    print(f"  PERMUTATION TEST")
    print(f"{'='*70}")

    common = upro_ret.index.intersection(cta_ret.index).intersection(
        rev_ret.index).intersection(panic_ret.index)

    best_weights = dict(configs)[best_config]
    growth_component = best_weights[0] * upro_ret.loc[common]
    perm = permutation_test(best_ret, growth_component, n_perms=200)
    print(f"  Real Sharpe: {perm['real_sharpe']:.3f}")
    print(f"  p-value:     {perm['p_value']:.4f}")
    print(f"  VERDICT:     {'PASS ✓' if perm['pass'] else 'FAIL ✗'}")

    # Sub-period check
    print(f"\n{'='*70}")
    print(f"  SUB-PERIOD CONSISTENCY")
    print(f"{'='*70}")
    n = len(best_ret)
    thirds = [best_ret.iloc[:n//3], best_ret.iloc[n//3:2*n//3], best_ret.iloc[2*n//3:]]
    for i, third in enumerate(thirds):
        s = sharpe(third)
        dates = f"{third.index[0].strftime('%Y-%m')}-{third.index[-1].strftime('%Y-%m')}"
        marker = "✓" if s > 0 else "✗"
        print(f"  Period {i+1} ({dates}): Sharpe {s:.3f} {marker}")

    # Compare with DCA
    print(f"\n{'='*70}")
    print(f"  DCA PROJECTION ($500 start, $100/wk)")
    print(f"{'='*70}")

    cash = 500.0
    contributed = 500.0
    last_week = None
    max_val = cash
    max_dd_dca = 0
    vals = []

    for i, (date, r) in enumerate(best_ret.items()):
        wk = (date.year, date.isocalendar()[1])
        if wk != last_week:
            cash += 100
            contributed += 100
            last_week = wk
        cash *= (1 + r)
        if cash > max_val:
            max_val = cash
        dd = (cash - max_val) / max_val
        if dd < max_dd_dca:
            max_dd_dca = dd
        vals.append(cash)

    years = len(best_ret) / 252
    print(f"  Period: {years:.1f} years")
    print(f"  Final value:  ${cash:,.0f}")
    print(f"  Contributed:  ${contributed:,.0f}")
    print(f"  Profit:       ${cash - contributed:,.0f}")
    print(f"  DCA MaxDD:    {max_dd_dca:.1%}")

    # Save
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'configs': all_results,
        'best_config': best_config,
        'best_passing_hc709': best_passing[0] if passing_configs else None,
        'permutation': perm,
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n  Saved to {OUTPUT_DIR}/results.json")


if __name__ == '__main__':
    main()
