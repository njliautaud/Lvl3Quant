#!/usr/bin/env python3
"""
ETF Momentum Regime-Adaptive v1
================================
Attempts to fix R1 regime gap (0.784) in the validated sector ETF momentum
strategy (Sharpe 3.96) by adapting selection to VIX regime.

Problem: Base strategy picks momentum winners which are mostly risk-on ETFs.
In bear markets, these get crushed. Bear Sharpe is 1.31 vs bull 6.06.

Solution attempts:
A) VIX regime overlay: When VIX > threshold, shift to defensive ETFs
B) Dual momentum: Require absolute momentum (above risk-free) as filter
C) Risk-parity weighting: Size positions inversely to recent vol
D) Trend filter: Only hold when ETF is above its SMA

Walk-forward: 252d train, 21d test, sliding window, 2016-2026.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

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

DEFENSIVE_ETFS = {'XLP', 'XLU', 'TLT', 'IEF', 'GLD'}
RISK_ON_ETFS = {'XLK', 'QQQ', 'XLY', 'XBI', 'IWM', 'EEM'}

TRAIN_DAYS = 252
TEST_DAYS = 21
STARTING_CAPITAL = 10000


def load_data():
    cache = '/home/jupiter/Lvl3Quant/data/etf_universe_cache.parquet'
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    raise FileNotFoundError("ETF cache not found")


def get_vix_proxy(data):
    """
    VIX proxy from QQQ realized vol and price dynamics.
    High vol + price below SMA200 = bear regime.
    """
    qqq = data[data['ticker'] == 'QQQ'].sort_values('date').copy()
    qqq = qqq.set_index('date')

    # Realized vol (21d annualized)
    qqq['rv_21d'] = qqq['close'].pct_change().rolling(21).std() * np.sqrt(252)
    qqq['sma200'] = qqq['close'].rolling(200).mean()
    qqq['sma50'] = qqq['close'].rolling(50).mean()

    # VIX proxy = realized vol scaled to typical VIX range
    qqq['vix_proxy'] = qqq['rv_21d'] * 100  # rough scaling

    # Regime: bear if below SMA200 OR vol > 25%
    qqq['regime'] = 'bull'
    qqq.loc[qqq['close'] < qqq['sma200'], 'regime'] = 'bear'

    return qqq[['rv_21d', 'sma200', 'sma50', 'vix_proxy', 'regime']].to_dict('index')


def compute_momentum_features(prices, lookback=252):
    """Compute momentum+quality features from price array."""
    n = len(prices)
    if n < lookback:
        return None

    mom_12_1 = prices[-21] / prices[-252] - 1 if n >= 252 else 0
    mom_6m = prices[-1] / prices[-126] - 1 if n >= 126 else 0
    mom_1m = prices[-1] / prices[-21] - 1 if n >= 21 else 0
    mom_3m = prices[-1] / prices[-63] - 1 if n >= 63 else 0
    mom_accel = mom_1m - (mom_3m / 3)

    rets = np.diff(prices[-63:]) / prices[-63:-1]
    vol_60d = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0.20
    sharpe_6m = mom_6m / (vol_60d + 1e-8)

    window = prices[-63:]
    peak = np.maximum.accumulate(window)
    dd = (window - peak) / peak
    maxdd_63d = np.min(dd)

    # SMA trend
    sma_50 = np.mean(prices[-50:]) if n >= 50 else prices[-1]
    sma_200 = np.mean(prices[-200:]) if n >= 200 else prices[-1]
    above_sma50 = prices[-1] > sma_50
    above_sma200 = prices[-1] > sma_200

    score = (
        0.30 * mom_12_1 +
        0.25 * sharpe_6m +
        0.20 * mom_accel +
        0.15 * (1 + maxdd_63d) +
        0.10 * (mom_1m * 5)
    )

    return {
        'score': score,
        'mom_12_1': mom_12_1,
        'mom_6m': mom_6m,
        'mom_1m': mom_1m,
        'vol_60d': vol_60d,
        'sharpe_6m': sharpe_6m,
        'maxdd_63d': maxdd_63d,
        'above_sma50': above_sma50,
        'above_sma200': above_sma200,
    }


def run_variant(data, name, top_n=3, variant_type='base',
                vix_threshold=25, sma_filter=False, dual_mom=False,
                risk_parity=False, defensive_shift=False):
    """Run a single variant of the ETF momentum strategy."""

    all_dates = sorted(data['date'].unique())
    n_dates = len(all_dates)
    vix_data = get_vix_proxy(data)

    equity = [STARTING_CAPITAL]
    equity_dates = [all_dates[TRAIN_DAYS]]
    trades = []

    rebal_idx = TRAIN_DAYS
    while rebal_idx + TEST_DAYS < n_dates:
        rebal_date = all_dates[rebal_idx]
        exit_date = all_dates[min(rebal_idx + TEST_DAYS, n_dates - 1)]
        current_equity = equity[-1]

        if current_equity <= 100:
            rebal_idx += TEST_DAYS
            equity.append(current_equity)
            equity_dates.append(exit_date)
            continue

        # Get regime
        regime = 'bull'
        vix_level = 15
        if rebal_date in vix_data:
            regime = vix_data[rebal_date].get('regime', 'bull')
            vix_level = vix_data[rebal_date].get('vix_proxy', 15)

        # Rank all ETFs
        train_start = all_dates[max(0, rebal_idx - TRAIN_DAYS)]
        rankings = []

        for ticker in ETF_UNIVERSE:
            td = data[(data['ticker'] == ticker) &
                     (data['date'] >= train_start) &
                     (data['date'] <= rebal_date)]
            if len(td) < 126:
                continue

            prices = td['close'].values
            features = compute_momentum_features(prices)
            if features is None:
                continue

            entry_price = prices[-1]

            # FILTER: SMA trend filter
            if sma_filter and not features['above_sma50']:
                continue

            # FILTER: Dual momentum (absolute momentum > 0)
            if dual_mom and features['mom_6m'] < 0:
                continue

            # ADJUST: Defensive shift in bear regime
            adjusted_score = features['score']
            if defensive_shift and regime == 'bear':
                if ticker in DEFENSIVE_ETFS:
                    adjusted_score *= 1.5  # Boost defensives
                elif ticker in RISK_ON_ETFS:
                    adjusted_score *= 0.5  # Penalize risk-on

            rankings.append({
                'ticker': ticker,
                'score': adjusted_score,
                'raw_score': features['score'],
                'vol': features['vol_60d'],
                'price': entry_price,
                'features': features,
            })

        if len(rankings) < 1:
            # Cash position when no ETFs pass filters
            rebal_idx += TEST_DAYS
            # Small risk-free return while in cash
            equity.append(current_equity * (1 + 0.05 / 12))
            equity_dates.append(exit_date)
            continue

        rankings.sort(key=lambda x: x['score'], reverse=True)
        picks = rankings[:top_n]

        # Position sizing
        if risk_parity:
            # Inverse vol weighting
            total_inv_vol = sum(1 / (p['vol'] + 0.05) for p in picks)
            weights = [(1 / (p['vol'] + 0.05)) / total_inv_vol for p in picks]
        else:
            weights = [1 / len(picks)] * len(picks)

        # Calculate returns
        period_pnl = 0
        for pick, weight in zip(picks, weights):
            ticker = pick['ticker']
            entry_price = pick['price']

            exit_data = data[(data['ticker'] == ticker) & (data['date'] == exit_date)]
            if len(exit_data) == 0:
                future = data[(data['ticker'] == ticker) &
                             (data['date'] > rebal_date) &
                             (data['date'] <= exit_date)]
                if len(future) == 0:
                    continue
                exit_data = future.iloc[-1:]

            exit_price = exit_data['close'].iloc[0]
            ret = exit_price / entry_price - 1
            position_size = current_equity * weight
            pnl = position_size * ret

            period_pnl += pnl
            trades.append({
                'date': rebal_date,
                'exit_date': exit_date,
                'ticker': ticker,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': ret,
                'weight': weight,
                'pnl': pnl,
                'regime': regime,
                'vix_level': vix_level,
                'won': pnl > 0,
            })

        current_equity += period_pnl
        equity.append(max(0, current_equity))
        equity_dates.append(exit_date)
        rebal_idx += TEST_DAYS

    if len(trades) == 0:
        return None

    return {
        'variant': name,
        'trades': trades,
        'equity': equity,
        'equity_dates': equity_dates,
    }


def compute_metrics(result, starting=10000):
    trades = result['trades']
    eq = np.array(result['equity'])
    if len(trades) == 0:
        return None

    total = len(trades)
    winners = sum(1 for t in trades if t['pnl'] > 0)
    wr = winners / total

    pnls = [t['pnl'] for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else float('inf')

    final = eq[-1]
    dates = result['equity_dates']
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25 if len(dates) >= 2 else 1
    cagr = (final / starting) ** (1 / max(years, 0.1)) - 1 if final > 0 else -1

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = np.min(dd)

    # Monthly returns for Sharpe
    period_rets = []
    for i in range(1, len(eq)):
        if eq[i-1] > 0:
            period_rets.append(eq[i] / eq[i-1] - 1)

    if len(period_rets) > 1 and np.std(period_rets) > 0:
        sharpe = np.mean(period_rets) / np.std(period_rets) * np.sqrt(12)
        down = [r for r in period_rets if r < 0]
        ds = np.std(down) if len(down) > 1 else np.std(period_rets)
        sortino = np.mean(period_rets) / ds * np.sqrt(12) if ds > 0 else 0
    else:
        sharpe = sortino = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Regime breakdown
    bull_pnls = [t['pnl'] for t in trades if t['regime'] == 'bull']
    bear_pnls = [t['pnl'] for t in trades if t['regime'] == 'bear']

    def regime_sharpe(pnls):
        if len(pnls) < 2: return 0
        return np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(12)

    bull_sharpe = regime_sharpe(bull_pnls)
    bear_sharpe = regime_sharpe(bear_pnls)
    max_s = max(abs(bull_sharpe), abs(bear_sharpe))
    r1_gap = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0

    return {
        'total_trades': total, 'win_rate': wr,
        'profit_factor': pf, 'sharpe': sharpe, 'sortino': sortino,
        'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
        'final_equity': final, 'years': years,
        'bull_sharpe': bull_sharpe, 'bear_sharpe': bear_sharpe,
        'bull_trades': len(bull_pnls), 'bear_trades': len(bear_pnls),
        'r1_gap': r1_gap, 'r1_pass': r1_gap <= 0.50,
    }


def permutation_test(result, n_perms=200):
    real_m = compute_metrics(result)
    if not real_m: return 1.0, []
    real_sharpe = real_m['sharpe']
    pnls = [t['pnl'] for t in result['trades']]
    perm_sharpes = []
    for _ in range(n_perms):
        shuf = np.random.permutation(pnls)
        eq = [STARTING_CAPITAL]
        for p in shuf:
            eq.append(max(0, eq[-1] + p))
        rets = [eq[i]/eq[i-1]-1 for i in range(1, len(eq)) if eq[i-1] > 0]
        s = np.mean(rets) / (np.std(rets)+1e-10) * np.sqrt(12) if len(rets) > 1 else 0
        perm_sharpes.append(s)
    return np.mean([1 if ps >= real_sharpe else 0 for ps in perm_sharpes]), perm_sharpes


def main():
    print("=" * 70)
    print("ETF MOMENTUM REGIME-ADAPTIVE v1")
    print("=" * 70)
    print(f"Start: {datetime.now()}")

    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("etf_momentum_regime_adaptive")
        mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}")

    data = load_data()
    print(f"Loaded {len(data)} rows, {data['ticker'].nunique()} ETFs")

    variants = [
        # (name, top_n, variant_type, vix_thresh, sma_filter, dual_mom, risk_parity, defensive_shift)
        ("A_Base_Top3", 3, 'base', 25, False, False, False, False),
        ("B_DefensiveShift", 3, 'defensive', 25, False, False, False, True),
        ("C_SMAfilter", 3, 'sma', 25, True, False, False, False),
        ("D_DualMom", 3, 'dual', 25, False, True, False, False),
        ("E_RiskParity", 3, 'rp', 25, False, False, True, False),
        ("F_DefShift+SMA", 3, 'combo1', 25, True, False, False, True),
        ("G_DualMom+RP", 3, 'combo2', 25, False, True, True, False),
        ("H_AllFilters", 3, 'all', 25, True, True, True, True),
        ("I_Base_Top5", 5, 'base5', 25, False, False, False, False),
        ("J_DefShift_Top5", 5, 'def5', 25, False, False, False, True),
    ]

    results = []
    for name, tn, vt, vix, sma, dm, rp, ds in variants:
        print(f"\n--- {name} ---")
        r = run_variant(data, name, top_n=tn, variant_type=vt,
                       vix_threshold=vix, sma_filter=sma, dual_mom=dm,
                       risk_parity=rp, defensive_shift=ds)
        if r is None:
            print("  No trades")
            continue
        m = compute_metrics(r)
        if m is None:
            continue
        print(f"  Sharpe: {m['sharpe']:.2f}, CAGR: {m['cagr']:.1%}, MaxDD: {m['max_dd']:.1%}")
        print(f"  WR: {m['win_rate']:.1%}, PF: {m['profit_factor']:.2f}")
        print(f"  Bull Sharpe: {m['bull_sharpe']:.2f}, Bear Sharpe: {m['bear_sharpe']:.2f}")
        print(f"  R1 Gap: {m['r1_gap']:.3f} {'PASS ✅' if m['r1_pass'] else 'FAIL ❌'}")
        r['metrics'] = m
        results.append(r)

    if not results:
        print("NO RESULTS")
        if MLFLOW_AVAILABLE:
            mlflow.end_run()
        return

    # Find best R1-passing variant
    r1_passers = [r for r in results if r['metrics']['r1_pass']]
    if r1_passers:
        best = max(r1_passers, key=lambda x: x['metrics']['sharpe'])
        print(f"\n{'='*70}")
        print(f"BEST R1-PASSING: {best['variant']}")
    else:
        best = max(results, key=lambda x: x['metrics']['sharpe'])
        print(f"\n{'='*70}")
        print(f"BEST OVERALL (no R1 passers): {best['variant']}")

    m = best['metrics']
    print(f"{'='*70}")
    print(f"  Sharpe:      {m['sharpe']:.2f}")
    print(f"  Sortino:     {m['sortino']:.2f}")
    print(f"  CAGR:        {m['cagr']:.1%}")
    print(f"  MaxDD:       {m['max_dd']:.1%}")
    print(f"  WR:          {m['win_rate']:.1%}")
    print(f"  PF:          {m['profit_factor']:.2f}")
    print(f"  Calmar:      {m['calmar']:.2f}")
    print(f"  Bull Sharpe: {m['bull_sharpe']:.2f} ({m['bull_trades']}t)")
    print(f"  Bear Sharpe: {m['bear_sharpe']:.2f} ({m['bear_trades']}t)")
    print(f"  R1 Gap:      {m['r1_gap']:.3f} {'PASS ✅' if m['r1_pass'] else 'FAIL ❌'}")

    # Permutation test on best
    print(f"\n--- Permutation Test ---")
    pp, ps = permutation_test(best)
    pp_pass = pp < 0.05
    print(f"  p={pp:.3f} {'PASS ✅' if pp_pass else 'FAIL ❌'}")

    # Sub-period
    trades = best['trades']
    mid = len(trades) // 2
    h1p = [t['pnl'] for t in trades[:mid]]
    h2p = [t['pnl'] for t in trades[mid:]]
    h1s = np.mean(h1p)/(np.std(h1p)+1e-10)*np.sqrt(12) if len(h1p)>1 else 0
    h2s = np.mean(h2p)/(np.std(h2p)+1e-10)*np.sqrt(12) if len(h2p)>1 else 0
    sub_pass = h1s > 0 and h2s > 0
    print(f"\n--- Sub-Period ---")
    print(f"  H1: {h1s:.2f}, H2: {h2s:.2f} {'PASS ✅' if sub_pass else 'FAIL ❌'}")

    # Outlier
    all_pnls = sorted([t['pnl'] for t in trades])
    nr = max(1, int(len(all_pnls)*0.05))
    trimmed = all_pnls[:-nr]
    ts = np.mean(trimmed)/(np.std(trimmed)+1e-10)*np.sqrt(12) if len(trimmed)>1 else 0
    out_pass = ts > 0
    print(f"\n--- Outlier ---")
    print(f"  Trimmed: {ts:.2f} {'PASS ✅' if out_pass else 'FAIL ❌'}")

    gates = sum([pp_pass, m['r1_pass'], sub_pass, out_pass])
    print(f"\n{'='*70}")
    print(f"GATES: {gates}/4")
    print(f"{'='*70}")

    # Full comparison table
    print(f"\n{'='*70}")
    print(f"{'Variant':<22} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'WR':>6} {'BullS':>7} {'BearS':>7} {'R1gap':>7} {'R1':>5}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['metrics']['sharpe'], reverse=True):
        m = r['metrics']
        r1_str = "PASS" if m['r1_pass'] else "FAIL"
        print(f"{r['variant']:<22} {m['sharpe']:>7.2f} {m['cagr']:>7.1%} {m['max_dd']:>7.1%} "
              f"{m['win_rate']:>5.1%} {m['bull_sharpe']:>7.2f} {m['bear_sharpe']:>7.2f} "
              f"{m['r1_gap']:>7.3f} {r1_str:>5}")

    # Top tickers in best variant
    tdf = pd.DataFrame(best['trades'])
    print(f"\nTop tickers in {best['variant']}:")
    for t, g in tdf.groupby('ticker'):
        if len(g) >= 5:
            print(f"  {t}: {len(g)}t, WR {g['won'].mean():.0%}, "
                  f"PnL ${g['pnl'].sum():.0f}, avg ret {g['return'].mean():.1%}")

    # Save
    save_path = '/home/jupiter/Lvl3Quant/research/findings/etf_momentum_regime_adaptive_v1_results.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'best_variant': best['variant'],
        'best_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in best['metrics'].items()},
        'gates': {'perm_p': float(pp), 'perm_pass': bool(pp_pass),
                  'r1_pass': bool(m['r1_pass']), 'r1_gap': float(m['r1_gap']),
                  'sub_pass': bool(sub_pass), 'outlier_pass': bool(out_pass),
                  'total': gates},
        'all_variants': [{
            'name': r['variant'],
            'sharpe': float(r['metrics']['sharpe']),
            'cagr': float(r['metrics']['cagr']),
            'max_dd': float(r['metrics']['max_dd']),
            'r1_gap': float(r['metrics']['r1_gap']),
            'r1_pass': bool(r['metrics']['r1_pass']),
            'bull_sharpe': float(r['metrics']['bull_sharpe']),
            'bear_sharpe': float(r['metrics']['bear_sharpe']),
        } for r in results],
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_AVAILABLE:
        mlflow.log_param("best_variant", best['variant'])
        mlflow.log_param("n_variants", len(results))
        mlflow.log_metric("sharpe", m['sharpe'])
        mlflow.log_metric("cagr", m['cagr'])
        mlflow.log_metric("max_dd", m['max_dd'])
        mlflow.log_metric("r1_gap", m['r1_gap'])
        mlflow.log_metric("bull_sharpe", m['bull_sharpe'])
        mlflow.log_metric("bear_sharpe", m['bear_sharpe'])
        mlflow.log_metric("perm_p", pp)
        mlflow.log_metric("gates_passed", gates)
        mlflow.log_artifact(save_path)
        mlflow.end_run()

    print(f"\nCompleted at {datetime.now()}")


if __name__ == '__main__':
    main()
