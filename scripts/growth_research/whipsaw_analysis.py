#!/usr/bin/env python3
"""
Whipsaw Analysis — False Signal Cost
======================================
How often does the vol-adjusted system generate false signals (switch to cash
or SPY, then switch back within N days)? What's the cost?

Key questions:
1. How many regime switches are "false" (reverse within 5/10/20 days)?
2. What's the average cost of each whipsaw?
3. Do whipsaws cluster in certain periods?
4. Can we add a confirmation delay to reduce whipsaws?
5. How much does a 2-day or 5-day confirmation delay improve things?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/whipsaw'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
    data = yf.download(tickers, start='2012-01-01', period='max',
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


def simulate_with_delay(closes, confirm_days=0, name=""):
    """Simulate vol-adjusted system with optional confirmation delay."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    pending_regime = None
    pending_count = 0
    daily_values = []

    switches = []
    regimes_over_time = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            signal_regime = 'CASH'
            signal_target = {'SPY': 1.0}
        elif vol < 0.20:
            signal_regime = 'UPRO'
            signal_target = {'UPRO': 1.0}
        elif vol < 0.30:
            signal_regime = 'SPY'
            signal_target = {'SPY': 1.0}
        else:
            signal_regime = 'SAFE'
            signal_target = {'GLD': 1.0}

        # Confirmation delay logic
        if confirm_days > 0:
            if signal_regime != last_regime:
                if pending_regime == signal_regime:
                    pending_count += 1
                else:
                    pending_regime = signal_regime
                    pending_count = 1

                if pending_count >= confirm_days:
                    # Confirmed — execute switch
                    regime = signal_regime
                    target = signal_target
                    pending_regime = None
                    pending_count = 0
                else:
                    regime = last_regime
                    target = None
            else:
                pending_regime = None
                pending_count = 0
                regime = signal_regime
                target = None
        else:
            regime = signal_regime
            target = signal_target if signal_regime != last_regime else None

        if regime != last_regime and target:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items()
                       if w > 0 and t in closes.columns}
            cash = 0
            switches.append({
                'date': date,
                'from': last_regime,
                'to': regime,
                'vol': vol,
                'portfolio_val': total_val,
            })
            last_regime = regime
        elif cash > 50 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)
        regimes_over_time.append(regime or last_regime)

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])
    return portfolio, total_contributed, switches


def main():
    print("="*70)
    print("WHIPSAW ANALYSIS — FALSE SIGNAL COST")
    print("="*70)

    closes = download_data()
    print(f"  Data: {len(closes)} days")

    # --- Baseline (no delay) ---
    portfolio, total, switches = simulate_with_delay(closes, confirm_days=0)

    print(f"\n  BASELINE (no confirmation delay):")
    print(f"    Total switches: {len(switches)}")
    years = len(portfolio) / 252
    print(f"    Switches/year: {len(switches)/years:.1f}")

    # Classify whipsaws
    for threshold in [5, 10, 20, 30]:
        whipsaws = 0
        for i in range(len(switches) - 1):
            days_between = (switches[i+1]['date'] - switches[i]['date']).days
            if days_between <= threshold:
                whipsaws += 1
        print(f"    Whipsaws (reverse within {threshold}d): {whipsaws} "
              f"({whipsaws/len(switches)*100:.0f}% of switches)")

    # Switch type breakdown
    from collections import Counter
    switch_types = Counter(f"{s['from']}→{s['to']}" for s in switches)
    print(f"\n    Switch type breakdown:")
    for stype, count in switch_types.most_common():
        print(f"      {stype}: {count}")

    # Whipsaw cost estimate
    print(f"\n    WHIPSAW COST ESTIMATE:")
    upro_ret = closes['UPRO'].pct_change()
    whipsaw_costs = []
    for i in range(len(switches) - 1):
        days_between = (switches[i+1]['date'] - switches[i]['date']).days
        if days_between <= 10:
            # Cost = what UPRO did during the whipsaw period
            start_date = switches[i]['date']
            end_date = switches[i+1]['date']
            upro_period = closes['UPRO'].loc[start_date:end_date]
            if len(upro_period) > 1:
                missed_return = upro_period.iloc[-1] / upro_period.iloc[0] - 1
                whipsaw_costs.append({
                    'date': start_date,
                    'days': days_between,
                    'missed_return': missed_return,
                    'from': switches[i]['from'],
                    'to': switches[i]['to'],
                })

    if whipsaw_costs:
        avg_cost = np.mean([abs(w['missed_return']) for w in whipsaw_costs])
        total_cost = sum(abs(w['missed_return']) for w in whipsaw_costs)
        positive = sum(1 for w in whipsaw_costs if w['missed_return'] > 0)
        negative = sum(1 for w in whipsaw_costs if w['missed_return'] < 0)
        print(f"    Whipsaws (≤10 days): {len(whipsaw_costs)}")
        print(f"    Avg missed UPRO return per whipsaw: {avg_cost*100:.2f}%")
        print(f"    Beneficial exits (UPRO fell): {negative} ({negative/len(whipsaw_costs)*100:.0f}%)")
        print(f"    Costly exits (UPRO rose): {positive} ({positive/len(whipsaw_costs)*100:.0f}%)")

    # --- Confirmation delay tests ---
    print(f"\n{'='*70}")
    print("CONFIRMATION DELAY TESTS")
    print(f"{'='*70}")

    delay_results = {}
    for delay in [0, 1, 2, 3, 5, 10]:
        portfolio, total, switches = simulate_with_delay(closes, confirm_days=delay)
        r = portfolio.pct_change().dropna()
        final = portfolio.iloc[-1]
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        peak = portfolio.expanding().max()
        max_dd = ((portfolio - peak) / peak).min()

        delay_results[delay] = {
            'final_value': float(final),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd * 100),
            'switches': len(switches),
        }

        print(f"\n  Delay {delay}d:")
        print(f"    Final: ${final:,.0f} | Sharpe {sharpe:.3f} | MaxDD {max_dd*100:.1f}% | "
              f"Switches: {len(switches)} ({len(switches)/years:.1f}/yr)")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY — CONFIRMATION DELAY IMPACT")
    print(f"{'='*70}")

    print(f"\n  {'Delay':>6s} {'Final $':>10s} {'Sharpe':>7s} {'MaxDD':>7s} {'Switches':>9s} {'SW/yr':>6s}")
    print("  " + "-"*47)
    for delay, m in sorted(delay_results.items()):
        print(f"  {delay:>5d}d ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['max_dd']:>6.1f}% "
              f"{m['switches']:>9d} {m['switches']/years:>5.1f}")

    baseline_val = delay_results[0]['final_value']
    print(f"\n  vs no delay:")
    for delay, m in sorted(delay_results.items()):
        if delay == 0:
            continue
        val_diff = m['final_value'] - baseline_val
        sharpe_diff = m['sharpe'] - delay_results[0]['sharpe']
        sw_diff = m['switches'] - delay_results[0]['switches']
        print(f"    {delay}d delay: value {val_diff:>+10,.0f}, Sharpe {sharpe_diff:+.3f}, "
              f"switches {sw_diff:+d}")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'delay_results': delay_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'whipsaw_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
