#!/usr/bin/env python3
"""
Momentum Call Buying on Sector ETFs v1
=======================================
Buy 1-month ATM calls on top momentum sector ETFs.

KEY THESIS: Our sector ETF momentum signal is validated (Sharpe 3.96, 90% WR).
Previous options tests failed because:
1. Dip-buying has elevated IV → expensive premium → theta kills edge
2. Individual stock signals were weak

This test is DIFFERENT because:
1. Momentum signal is 4x stronger than dip signals
2. Buying INTO strength → IV is normal/low (not elevated like dip stocks)
3. Sector ETFs have lower IV than individual stocks → cheaper options
4. Monthly rebalance aligns with standard options expiry

Walk-forward: 252d train, 21d test (monthly). Starting capital $645 (agentic account).
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


ETF_UNIVERSE = [
    'XLE', 'XLK', 'XLF', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
    'XLC', 'GLD', 'SLV', 'DBC', 'TLT', 'IEF', 'HYG', 'QQQ', 'IWM', 'EEM',
    'VNQ', 'XBI',
]

STARTING_CAPITAL = 645
TRAIN_DAYS = 252
HOLD_DAYS = 21  # Monthly cycle
COMMISSION = 1.30  # Per contract per leg


def load_data():
    cache = '/home/jupiter/Lvl3Quant/data/etf_universe_cache.parquet'
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    raise FileNotFoundError("Run momentum_put_spread_etf_v1.py first")


def compute_momentum_score(df_ticker):
    """Same scoring as validated sector_etf_momentum_v1."""
    close = df_ticker['close'].values
    n = len(close)
    if n < 252:
        return None, None

    # Core features
    mom_12_1 = close[-21] / close[-252] - 1 if n >= 252 else 0
    rets_6m = close[-1] / close[-126] - 1 if n >= 126 else 0
    rets_1m = close[-1] / close[-21] - 1 if n >= 21 else 0
    rets_3m = close[-1] / close[-63] - 1 if n >= 63 else 0
    mom_accel = rets_1m - (rets_3m / 3) if n >= 63 else 0

    daily_rets = np.diff(close[-63:]) / close[-63:-1] if n >= 63 else np.array([0])
    vol_60d = np.std(daily_rets) * np.sqrt(252) if len(daily_rets) > 1 else 0.20
    sharpe_6m = rets_6m / (vol_60d + 1e-8)

    if n >= 63:
        window = close[-63:]
        peak = np.maximum.accumulate(window)
        dd = (window - peak) / peak
        maxdd_63d = np.min(dd)
    else:
        maxdd_63d = 0

    vol_ratio = 1
    if 'volume' in df_ticker.columns and n >= 42:
        vol = df_ticker['volume'].values
        vol_ratio = np.mean(vol[-21:]) / (np.mean(vol[-42:-21]) + 1)

    score = (
        0.30 * mom_12_1 +
        0.25 * sharpe_6m +
        0.20 * mom_accel +
        0.15 * (1 + maxdd_63d) +
        0.10 * vol_ratio
    )

    return score, vol_60d


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def run_variant(data, name, top_n=3, hold_days=21, moneyness='atm',
                max_per_trade=200, starting_capital=645):
    """
    Run a single variant of the momentum call buying strategy.
    moneyness: 'atm', 'itm' (5% ITM), 'otm' (5% OTM)
    """
    all_dates = sorted(data['date'].unique())
    n_dates = len(all_dates)

    trades = []
    equity = [starting_capital]
    equity_dates = [all_dates[TRAIN_DAYS]]

    rebal_idx = TRAIN_DAYS

    while rebal_idx + hold_days < n_dates:
        rebal_date = all_dates[rebal_idx]
        exit_date = all_dates[min(rebal_idx + hold_days, n_dates - 1)]
        current_equity = equity[-1]

        if current_equity <= 50:  # Stop if too low
            rebal_idx += hold_days
            equity.append(current_equity)
            equity_dates.append(exit_date)
            continue

        # Rank ETFs
        train_start = all_dates[max(0, rebal_idx - TRAIN_DAYS)]
        rankings = []

        for ticker in ETF_UNIVERSE:
            td = data[(data['ticker'] == ticker) &
                     (data['date'] >= train_start) &
                     (data['date'] <= rebal_date)]
            if len(td) < 126:
                continue

            score, vol = compute_momentum_score(td)
            if score is None:
                continue

            price = td['close'].iloc[-1]
            rankings.append({
                'ticker': ticker,
                'score': score,
                'price': price,
                'vol': max(vol, 0.10),
            })

        if len(rankings) < top_n:
            rebal_idx += hold_days
            equity.append(current_equity)
            equity_dates.append(exit_date)
            continue

        rankings.sort(key=lambda x: x['score'], reverse=True)
        picks = rankings[:top_n]

        # Allocate capital equally
        per_pick = min(max_per_trade, current_equity / top_n)
        period_pnl = 0

        for pick in picks:
            price = pick['price']
            vol = pick['vol']

            # Set strike based on moneyness
            if moneyness == 'atm':
                strike = price
            elif moneyness == 'itm':
                strike = price * 0.95  # 5% ITM
            elif moneyness == 'otm':
                strike = price * 1.05  # 5% OTM
            else:
                strike = price

            T = 30 / 365.0  # 1 month
            r = 0.05

            # Entry call price
            entry_call = bs_call_price(price, strike, T, r, vol)
            if entry_call <= 0:
                continue

            # How many contracts? (each contract = 100 shares)
            cost_per_contract = entry_call * 100
            if cost_per_contract <= 0:
                continue

            n_contracts = max(1, int(per_pick / cost_per_contract))
            total_cost = cost_per_contract * n_contracts

            if total_cost > current_equity * 0.5:  # Don't risk more than 50% on one trade
                n_contracts = max(1, int(current_equity * 0.5 / cost_per_contract))
                total_cost = cost_per_contract * n_contracts

            # Get exit price
            exit_data = data[(data['ticker'] == pick['ticker']) & (data['date'] == exit_date)]
            if len(exit_data) == 0:
                future = data[(data['ticker'] == pick['ticker']) &
                             (data['date'] > rebal_date) &
                             (data['date'] <= exit_date)]
                if len(future) == 0:
                    continue
                exit_data = future.iloc[-1:]

            exit_price = exit_data['close'].iloc[0]

            # Exit call price (less time remaining)
            remaining_T = max(0, (30 - hold_days)) / 365.0
            exit_call = bs_call_price(exit_price, strike, remaining_T, r, vol)

            # P&L = (exit_call - entry_call) * 100 * n_contracts - commissions
            pnl = (exit_call - entry_call) * 100 * n_contracts - COMMISSION * 2 * n_contracts

            # Cap loss at total cost (can't lose more than you paid)
            if pnl < -total_cost:
                pnl = -total_cost

            period_pnl += pnl

            trades.append({
                'date': rebal_date,
                'exit_date': exit_date,
                'ticker': pick['ticker'],
                'score': pick['score'],
                'price': price,
                'exit_price': exit_price,
                'strike': strike,
                'vol': vol,
                'entry_call': entry_call,
                'exit_call': exit_call,
                'n_contracts': n_contracts,
                'cost': total_cost,
                'pnl': pnl,
                'return_pct': exit_price / price - 1,
                'won': pnl > 0,
            })

        current_equity += period_pnl
        equity.append(max(0, current_equity))
        equity_dates.append(exit_date)
        rebal_idx += hold_days

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

    if len(result['equity_dates']) >= 2:
        first = pd.Timestamp(result['equity_dates'][0])
        last = pd.Timestamp(result['equity_dates'][-1])
        years = (last - first).days / 365.25
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

    return {
        'total_trades': total, 'win_rate': wr,
        'profit_factor': pf, 'sharpe': sharpe, 'sortino': sortino,
        'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
        'final_equity': final, 'years': years,
        'trades_per_year': total / max(years, 0.1),
    }


def permutation_test(data, result, n_perms=200):
    real_m = compute_metrics(result)
    if not real_m:
        return 1.0, []
    real_sharpe = real_m['sharpe']
    pnls = [t['pnl'] for t in result['trades']]
    perm_sharpes = []
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
    qqq = data[data['ticker'] == 'QQQ'].sort_values('date')
    if len(qqq) < 200:
        return {'pass': True, 'gap': 0, 'bull_sharpe': 0, 'bear_sharpe': 0}
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
    print("MOMENTUM CALL BUYING ON SECTOR ETFs v1")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("momentum_call_etf")
        mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}")

    data = load_data()
    print(f"Loaded {len(data)} rows, {data['ticker'].nunique()} ETFs")

    variants = [
        # (name, top_n, hold_days, moneyness, max_per_trade)
        ("Top3_21d_ATM_200", 3, 21, 'atm', 200),
        ("Top3_21d_ITM_200", 3, 21, 'itm', 200),
        ("Top3_21d_OTM_200", 3, 21, 'otm', 200),
        ("Top2_21d_ATM_300", 2, 21, 'atm', 300),
        ("Top3_14d_ATM_200", 3, 14, 'atm', 200),
        ("Top1_21d_ATM_500", 1, 21, 'atm', 500),
        ("Top3_10d_ATM_200", 3, 10, 'atm', 200),
        ("Top5_21d_ATM_120", 5, 21, 'atm', 120),
    ]

    results = []
    best = None
    best_sharpe = -999

    for name, top_n, hd, money, max_pt in variants:
        print(f"\n--- {name} ---")
        r = run_variant(data, name, top_n=top_n, hold_days=hd,
                       moneyness=money, max_per_trade=max_pt)
        if r is None:
            print("  No trades")
            continue
        m = compute_metrics(r)
        if m is None:
            continue
        print(f"  Trades: {m['total_trades']}, WR: {m['win_rate']:.1%}")
        print(f"  Sharpe: {m['sharpe']:.2f}, CAGR: {m['cagr']:.1%}, MaxDD: {m['max_dd']:.1%}")
        print(f"  PF: {m['profit_factor']:.2f}, Final: ${m['final_equity']:.0f}")
        r['metrics'] = m
        results.append(r)
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best = r

    if best is None:
        print("NO VARIANTS PRODUCED RESULTS")
        if MLFLOW_AVAILABLE:
            mlflow.log_param("status", "FAILED")
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
    pp, ps = permutation_test(data, best)
    pp_pass = pp < 0.05
    print(f"  p={pp:.3f} {'PASS ✅' if pp_pass else 'FAIL ❌'}")

    print(f"\n--- GATE 2: Regime ---")
    rg = regime_test(best, data)
    print(f"  Bull: {rg['bull_sharpe']:.2f} ({rg['bull_trades']}t), Bear: {rg['bear_sharpe']:.2f} ({rg['bear_trades']}t)")
    print(f"  Gap: {rg['gap']:.3f} {'PASS ✅' if rg['pass'] else 'FAIL ❌'}")

    # Sub-period
    trades = best['trades']
    mid = len(trades) // 2
    h1_pnls = [t['pnl'] for t in trades[:mid]]
    h2_pnls = [t['pnl'] for t in trades[mid:]]
    h1s = np.mean(h1_pnls) / (np.std(h1_pnls) + 1e-10) * np.sqrt(12) if len(h1_pnls) > 1 else 0
    h2s = np.mean(h2_pnls) / (np.std(h2_pnls) + 1e-10) * np.sqrt(12) if len(h2_pnls) > 1 else 0
    sub_pass = h1s > 0 and h2s > 0
    print(f"\n--- GATE 3: Sub-Period ---")
    print(f"  H1: {h1s:.2f}, H2: {h2s:.2f} {'PASS ✅' if sub_pass else 'FAIL ❌'}")

    # Outlier
    all_pnls = sorted([t['pnl'] for t in trades])
    n_rm = max(1, int(len(all_pnls) * 0.05))
    trimmed = all_pnls[:-n_rm]
    ts = np.mean(trimmed) / (np.std(trimmed) + 1e-10) * np.sqrt(12) if len(trimmed) > 1 else 0
    out_pass = ts > 0
    print(f"\n--- GATE 4: Outlier ---")
    print(f"  Trimmed Sharpe: {ts:.2f} {'PASS ✅' if out_pass else 'FAIL ❌'}")

    gates = sum([pp_pass, rg['pass'], sub_pass, out_pass])
    print(f"\n{'='*70}")
    print(f"GATES: {gates}/4")
    print(f"{'='*70}")

    # Top tickers
    tdf = pd.DataFrame(best['trades'])
    print(f"\nTop tickers:")
    for t, g in tdf.groupby('ticker'):
        if len(g) >= 3:
            print(f"  {t}: {len(g)} trades, WR {g['won'].mean():.0%}, PnL ${g['pnl'].sum():.0f}, avg return {g['return_pct'].mean():.1%}")

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
    save_path = '/home/jupiter/Lvl3Quant/research/findings/momentum_call_etf_v1_results.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'best_variant': best['variant'],
        'metrics': metrics,
        'gates': {'perm_p': pp, 'perm_pass': pp_pass, 'regime': rg,
                  'sub_period': {'h1': h1s, 'h2': h2s, 'pass': sub_pass},
                  'outlier': {'trimmed_sharpe': ts, 'pass': out_pass},
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
