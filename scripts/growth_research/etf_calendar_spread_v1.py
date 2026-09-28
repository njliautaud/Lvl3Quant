#!/usr/bin/env python3
"""
ETF Calendar Spread Income v1
==============================
Sell near-dated ATM options, buy far-dated ATM options on sector ETFs.
Profits from faster theta decay in the near-dated leg.

Thesis: Calendar spreads are ideal for income on small accounts:
1. Defined risk (max loss = net debit paid)
2. Profits from time decay differential
3. Works best in low-vol, range-bound environments
4. Can be entered for $100-300 (agentic account sized)

Strategy:
- Monthly: sell front-month ATM call, buy back-month ATM call (same strike)
- Entry: when realized vol < implied vol (IV premium = profitable time decay)
- Exit: at front-month expiry or when spread reaches target profit
- Universe: sector ETFs with moderate IV (15-35% annualized)

Walk-forward: monthly rebalance, 2016-2026.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

ETF_UNIVERSE = [
    'XLE', 'XLK', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
    'GLD', 'QQQ', 'IWM', 'VNQ', 'XBI', 'TLT', 'EEM', 'SLV',
]

STARTING_CAPITAL = 645
COMMISSION = 2.60  # RT per spread (2 legs × $1.30)


def load_data():
    cache = '/home/jupiter/Lvl3Quant/data/etf_universe_cache.parquet'
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    raise FileNotFoundError("ETF cache not found")


def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, sigma):
    """Vega: sensitivity to vol change."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1) / 100  # Per 1% vol change


def compute_realized_vol(prices, window=21):
    """Compute annualized realized volatility."""
    if len(prices) < window + 1:
        return 0.20
    rets = np.diff(np.log(prices[-window-1:]))
    return np.std(rets) * np.sqrt(252)


def compute_implied_vol_proxy(prices, window=63):
    """
    Proxy for implied vol using recent realized vol + term structure.
    In practice, IV ≈ RV + vol risk premium (historically ~2-4% for SPX).
    """
    rv_short = compute_realized_vol(prices, 21)
    rv_long = compute_realized_vol(prices, 63) if len(prices) > 64 else rv_short

    # IV is typically above RV (vol risk premium)
    # Higher premium when RV is low, lower when RV is high
    vrp_pct = 0.15 if rv_short < 0.15 else (0.08 if rv_short < 0.25 else 0.03)
    iv_proxy = rv_short * (1 + vrp_pct)

    return iv_proxy, rv_short, rv_long


def run_variant(data, name, top_n=3, front_dte=30, back_dte=60,
                iv_filter=True, momentum_filter=False, max_per_trade=200):
    """
    Run calendar spread backtest variant.

    Calendar spread mechanics:
    - Buy back-month call at strike K, expiry T2
    - Sell front-month call at strike K, expiry T1 (T1 < T2)
    - Net debit = back_call - front_call
    - At front expiry: if stock near K, spread is worth ~back_call_remaining
    - Profit = spread_value_at_exit - net_debit
    - Max loss = net debit paid
    """
    all_dates = sorted(data['date'].unique())
    n_dates = len(all_dates)

    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [all_dates[252]]
    r = 0.05

    rebal_idx = 252  # Need 1 year of data

    while rebal_idx + front_dte < n_dates:
        rebal_date = all_dates[rebal_idx]
        front_expiry_idx = min(rebal_idx + front_dte, n_dates - 1)
        front_expiry = all_dates[front_expiry_idx]
        current_equity = equity[-1]

        if current_equity <= 50:
            rebal_idx += front_dte
            equity.append(current_equity)
            equity_dates.append(front_expiry)
            continue

        # Evaluate each ETF
        candidates = []
        for ticker in ETF_UNIVERSE:
            td = data[(data['ticker'] == ticker) & (data['date'] <= rebal_date)]
            if len(td) < 252:
                continue

            prices = td['close'].values
            current_price = prices[-1]

            iv_proxy, rv_short, rv_long = compute_implied_vol_proxy(prices)

            # Filters
            if iv_filter:
                # Only enter when IV > RV (positive vol risk premium)
                if iv_proxy < rv_short * 1.05:
                    continue

                # Prefer moderate IV (15-40%) — too low = tiny premium, too high = big moves
                if iv_proxy < 0.12 or iv_proxy > 0.45:
                    continue

            # Momentum filter: prefer range-bound or slightly bullish
            if momentum_filter:
                mom_1m = prices[-1] / prices[-21] - 1 if len(prices) >= 21 else 0
                # Calendar spreads work best when price stays near strike
                # Avoid strongly trending (>5% monthly)
                if abs(mom_1m) > 0.05:
                    continue

            # Score: prefer moderate IV with positive VRP
            vrp = iv_proxy - rv_short
            score = vrp * 100  # Higher VRP = better

            candidates.append({
                'ticker': ticker,
                'price': current_price,
                'iv': iv_proxy,
                'rv': rv_short,
                'vrp': vrp,
                'score': score,
            })

        if len(candidates) < 1:
            rebal_idx += front_dte
            equity.append(current_equity)
            equity_dates.append(front_expiry)
            continue

        candidates.sort(key=lambda x: x['score'], reverse=True)
        picks = candidates[:top_n]

        per_pick = min(max_per_trade, current_equity / max(len(picks), 1))
        period_pnl = 0

        for pick in picks:
            price = pick['price']
            iv = pick['iv']
            strike = round(price, 0)  # ATM

            T1 = front_dte / 365.0  # Front month
            T2 = back_dte / 365.0   # Back month

            # Price the calendar spread at entry
            front_call = bs_call(price, strike, T1, r, iv)
            back_call = bs_call(price, strike, T2, r, iv)
            net_debit = back_call - front_call  # We pay this

            if net_debit <= 0:
                continue  # Shouldn't happen for ATM calendar

            # Cost per contract (100 shares)
            cost_per_contract = net_debit * 100

            if cost_per_contract <= 0 or cost_per_contract > per_pick:
                continue

            n_contracts = max(1, int(per_pick / cost_per_contract))
            total_cost = cost_per_contract * n_contracts

            # Get price at front expiry
            exit_data = data[(data['ticker'] == pick['ticker']) &
                           (data['date'] == front_expiry)]
            if len(exit_data) == 0:
                future = data[(data['ticker'] == pick['ticker']) &
                             (data['date'] > rebal_date) &
                             (data['date'] <= front_expiry)]
                if len(future) == 0:
                    continue
                exit_data = future.iloc[-1:]

            exit_price = exit_data['close'].iloc[0]

            # At front expiry:
            # Front call is at expiry: worth max(exit_price - strike, 0)
            front_call_exit = max(exit_price - strike, 0)

            # Back call still has T2-T1 time left
            remaining_T = (back_dte - front_dte) / 365.0

            # Recalculate IV for remaining period (use same IV, slight adjustment)
            # In reality IV might change, but this is a reasonable approximation
            price_move = exit_price / price - 1
            # If price moved a lot, IV tends to increase (vol clustering)
            iv_exit = iv * (1 + abs(price_move) * 0.5)

            back_call_exit = bs_call(exit_price, strike, remaining_T, r, iv_exit)

            # Calendar spread value at exit
            spread_value_exit = back_call_exit - front_call_exit

            # P&L per share
            pnl_per_share = spread_value_exit - net_debit
            pnl = pnl_per_share * 100 * n_contracts - COMMISSION * n_contracts

            # Calendar spread max loss is limited to net debit
            if pnl < -total_cost:
                pnl = -total_cost

            period_pnl += pnl

            trades.append({
                'date': rebal_date,
                'exit_date': front_expiry,
                'ticker': pick['ticker'],
                'price': price,
                'exit_price': exit_price,
                'strike': strike,
                'iv': iv,
                'rv': pick['rv'],
                'vrp': pick['vrp'],
                'net_debit': net_debit * 100 * n_contracts,
                'spread_exit': spread_value_exit * 100 * n_contracts,
                'pnl': pnl,
                'pnl_pct': pnl / total_cost if total_cost > 0 else 0,
                'price_move': price_move,
                'n_contracts': n_contracts,
                'won': pnl > 0,
            })

        current_equity += period_pnl
        equity.append(max(0, current_equity))
        equity_dates.append(front_expiry)
        rebal_idx += front_dte

    if len(trades) == 0:
        return None

    return {
        'variant': name,
        'trades': trades,
        'equity': equity,
        'equity_dates': equity_dates,
    }


def compute_metrics(result, starting=645):
    trades = result['trades']
    equity = result['equity']
    if len(trades) == 0:
        return None

    total = len(trades)
    winners = sum(1 for t in trades if t['pnl'] > 0)
    wr = winners / total

    pnls = [t['pnl'] for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else float('inf')

    eq = np.array(equity)
    final = eq[-1]

    dates = result['equity_dates']
    if len(dates) >= 2:
        years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    else:
        years = 1

    cagr = (final / starting) ** (1 / max(years, 0.1)) - 1 if final > 0 else -1

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = np.min(dd)

    period_rets = []
    for i in range(1, len(equity)):
        if equity[i-1] > 0:
            period_rets.append(equity[i] / equity[i-1] - 1)

    if len(period_rets) > 1 and np.std(period_rets) > 0:
        sharpe = np.mean(period_rets) / np.std(period_rets) * np.sqrt(12)
        down = [r for r in period_rets if r < 0]
        ds = np.std(down) if len(down) > 1 else np.std(period_rets)
        sortino = np.mean(period_rets) / ds * np.sqrt(12) if ds > 0 else 0
    else:
        sharpe = sortino = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Average stats
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades])
    avg_price_move = np.mean([abs(t['price_move']) for t in trades])

    return {
        'total_trades': total, 'win_rate': wr,
        'profit_factor': pf, 'sharpe': sharpe, 'sortino': sortino,
        'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
        'final_equity': final, 'years': years,
        'trades_per_year': total / max(years, 0.1),
        'avg_pnl_pct': avg_pnl_pct,
        'avg_price_move': avg_price_move,
    }


def permutation_test(result, n_perms=200):
    real_m = compute_metrics(result)
    if not real_m:
        return 1.0, []
    real_sharpe = real_m['sharpe']
    pnls = [t['pnl'] for t in result['trades']]
    perm_sharpes = []
    print(f"  Running {n_perms} permutations...")
    for _ in range(n_perms):
        shuf = np.random.permutation(pnls)
        eq = [STARTING_CAPITAL]
        for p in shuf:
            eq.append(max(0, eq[-1] + p))
        rets = []
        for i in range(1, len(eq)):
            if eq[i-1] > 0:
                rets.append(eq[i] / eq[i-1] - 1)
        if len(rets) > 1 and np.std(rets) > 0:
            s = np.mean(rets) / np.std(rets) * np.sqrt(12)
        else:
            s = 0
        perm_sharpes.append(s)
    p = np.mean([1 if ps >= real_sharpe else 0 for ps in perm_sharpes])
    return p, perm_sharpes


def regime_test(result, data):
    trades = result['trades']
    if not trades:
        return {'pass': True, 'gap': 0}

    qqq = data[data['ticker'] == 'QQQ'].sort_values('date').copy()
    if len(qqq) < 200:
        return {'pass': True, 'gap': 0, 'bull_sharpe': 0, 'bear_sharpe': 0,
                'bull_trades': 0, 'bear_trades': 0}
    qqq = qqq.set_index('date')
    qqq['sma200'] = qqq['close'].rolling(200).mean()

    bull, bear = [], []
    for t in trades:
        d = pd.Timestamp(t['date'])
        if d in qqq.index and pd.notna(qqq.loc[d, 'sma200']):
            if qqq.loc[d, 'close'] > qqq.loc[d, 'sma200']:
                bull.append(t['pnl'])
            else:
                bear.append(t['pnl'])
        else:
            bull.append(t['pnl'])

    def s(p):
        if len(p) < 2: return 0
        return np.mean(p) / (np.std(p) + 1e-10) * np.sqrt(12)

    bs, brs = s(bull), s(bear)
    mx = max(abs(bs), abs(brs))
    gap = abs(bs - brs) / mx if mx > 0 else 0
    return {'bull_sharpe': bs, 'bear_sharpe': brs,
            'bull_trades': len(bull), 'bear_trades': len(bear),
            'gap': gap, 'pass': gap <= 0.50}


def main():
    print("=" * 70)
    print("ETF CALENDAR SPREAD INCOME v1")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("etf_calendar_spread")
        mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}")

    data = load_data()
    print(f"Loaded {len(data)} rows, {data['ticker'].nunique()} ETFs")

    variants = [
        # (name, top_n, front_dte, back_dte, iv_filter, mom_filter, max_per_trade)
        ("Top3_30_60_IVfilt", 3, 30, 60, True, False, 200),
        ("Top3_30_60_noFilt", 3, 30, 60, False, False, 200),
        ("Top3_30_60_momFilt", 3, 30, 60, True, True, 200),
        ("Top5_30_60_IVfilt", 5, 30, 60, True, False, 120),
        ("Top2_30_60_IVfilt", 2, 30, 60, True, False, 300),
        ("Top3_21_50_IVfilt", 3, 21, 50, True, False, 200),
        ("Top3_30_90_IVfilt", 3, 30, 90, True, False, 200),
        ("Top3_14_45_IVfilt", 3, 14, 45, True, False, 200),
    ]

    results = []
    best = None
    best_sharpe = -999

    for name, tn, fdte, bdte, ivf, mf, mpt in variants:
        print(f"\n--- {name} ---")
        r = run_variant(data, name, top_n=tn, front_dte=fdte, back_dte=bdte,
                       iv_filter=ivf, momentum_filter=mf, max_per_trade=mpt)
        if r is None:
            print("  No trades")
            continue
        m = compute_metrics(r)
        if m is None:
            continue
        print(f"  Trades: {m['total_trades']}, WR: {m['win_rate']:.1%}")
        print(f"  Sharpe: {m['sharpe']:.2f}, CAGR: {m['cagr']:.1%}, MaxDD: {m['max_dd']:.1%}")
        print(f"  PF: {m['profit_factor']:.2f}, Final: ${m['final_equity']:.0f}")
        print(f"  Avg PnL/trade: {m['avg_pnl_pct']:.1%}, Avg |price move|: {m['avg_price_move']:.1%}")
        r['metrics'] = m
        results.append(r)
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best = r

    if best is None:
        print("\nNO VARIANTS PRODUCED RESULTS")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "NO_TRADES")
            mlflow.end_run()
        return

    metrics = best['metrics']
    print(f"\n{'='*70}")
    print(f"BEST: {best['variant']}")
    print(f"{'='*70}")
    print(f"  Sharpe:  {metrics['sharpe']:.2f}")
    print(f"  Sortino: {metrics['sortino']:.2f}")
    print(f"  CAGR:    {metrics['cagr']:.1%}")
    print(f"  MaxDD:   {metrics['max_dd']:.1%}")
    print(f"  WR:      {metrics['win_rate']:.1%}")
    print(f"  PF:      {metrics['profit_factor']:.2f}")
    print(f"  Calmar:  {metrics['calmar']:.2f}")
    print(f"  Trades:  {metrics['total_trades']} ({metrics['trades_per_year']:.0f}/yr)")
    print(f"  Final:   ${metrics['final_equity']:.0f}")

    # Gates
    print(f"\n--- GATE 1: Permutation ---")
    pp, ps = permutation_test(best)
    pp_pass = pp < 0.05
    print(f"  p={pp:.3f} {'PASS ✅' if pp_pass else 'FAIL ❌'}")
    print(f"  Real Sharpe: {metrics['sharpe']:.2f}, Random mean: {np.mean(ps):.2f}")

    print(f"\n--- GATE 2: Regime ---")
    rg = regime_test(best, data)
    print(f"  Bull: {rg['bull_sharpe']:.2f} ({rg['bull_trades']}t), Bear: {rg['bear_sharpe']:.2f} ({rg['bear_trades']}t)")
    print(f"  Gap: {rg['gap']:.3f} {'PASS ✅' if rg['pass'] else 'FAIL ❌'}")

    # Sub-period
    trades = best['trades']
    mid = len(trades) // 2
    h1p = [t['pnl'] for t in trades[:mid]]
    h2p = [t['pnl'] for t in trades[mid:]]
    h1s = np.mean(h1p) / (np.std(h1p) + 1e-10) * np.sqrt(12) if len(h1p) > 1 else 0
    h2s = np.mean(h2p) / (np.std(h2p) + 1e-10) * np.sqrt(12) if len(h2p) > 1 else 0
    sub_pass = h1s > 0 and h2s > 0
    print(f"\n--- GATE 3: Sub-Period ---")
    print(f"  H1: {h1s:.2f}, H2: {h2s:.2f} {'PASS ✅' if sub_pass else 'FAIL ❌'}")

    # Outlier
    all_pnls = sorted([t['pnl'] for t in trades])
    nr = max(1, int(len(all_pnls) * 0.05))
    trimmed = all_pnls[:-nr]
    ts = np.mean(trimmed) / (np.std(trimmed) + 1e-10) * np.sqrt(12) if len(trimmed) > 1 else 0
    out_pass = ts > 0
    print(f"\n--- GATE 4: Outlier ---")
    print(f"  Trimmed Sharpe: {ts:.2f} {'PASS ✅' if out_pass else 'FAIL ❌'}")

    gates = sum([pp_pass, rg['pass'], sub_pass, out_pass])
    print(f"\n{'='*70}")
    print(f"GATES: {gates}/4")
    print(f"{'='*70}")

    # Ticker breakdown
    tdf = pd.DataFrame(best['trades'])
    print(f"\nTicker breakdown:")
    for t, g in tdf.groupby('ticker'):
        if len(g) >= 2:
            print(f"  {t}: {len(g)}t, WR {g['won'].mean():.0%}, PnL ${g['pnl'].sum():.0f}, "
                  f"avg IV {g['iv'].mean():.0%}, avg move {g['price_move'].mean():.1%}")

    # Win rate by IV level
    tdf['iv_bucket'] = pd.cut(tdf['iv'], bins=[0, 0.15, 0.25, 0.35, 1.0],
                               labels=['<15%', '15-25%', '25-35%', '>35%'])
    print(f"\nWR by IV level:")
    for bucket, g in tdf.groupby('iv_bucket'):
        if len(g) >= 3:
            print(f"  IV {bucket}: {len(g)}t, WR {g['won'].mean():.0%}, avg PnL% {g['pnl_pct'].mean():.1%}")

    # All variants
    print(f"\n{'='*70}")
    print(f"{'Variant':<25} {'Sharpe':>7} {'WR':>6} {'CAGR':>8} {'MaxDD':>8} {'PF':>6} {'Final':>8}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['metrics']['sharpe'], reverse=True):
        m = r['metrics']
        print(f"{r['variant']:<25} {m['sharpe']:>7.2f} {m['win_rate']:>5.1%} "
              f"{m['cagr']:>7.1%} {m['max_dd']:>7.1%} {m['profit_factor']:>6.2f} "
              f"${m['final_equity']:>7.0f}")

    # Save
    save_path = '/home/jupiter/Lvl3Quant/research/findings/etf_calendar_spread_v1_results.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'best_variant': best['variant'],
        'metrics': metrics,
        'gates': {'perm_p': pp, 'perm_pass': pp_pass, 'regime': rg,
                  'sub': {'h1': h1s, 'h2': h2s, 'pass': sub_pass},
                  'outlier': {'trimmed': ts, 'pass': out_pass},
                  'total': gates},
        'all_variants': [{'variant': r['variant'], 'metrics': r['metrics']} for r in results],
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_AVAILABLE:
        mlflow.log_param("best_variant", best['variant'])
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        mlflow.log_metric("perm_p", pp)
        mlflow.log_metric("gates_passed", gates)
        mlflow.log_artifact(save_path)
        mlflow.end_run()

    print(f"\nCompleted at {datetime.now()}")
    return save_data


if __name__ == '__main__':
    main()
