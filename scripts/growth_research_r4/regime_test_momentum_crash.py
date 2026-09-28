#!/usr/bin/env python3
"""
REGIME TEST for Momentum + Crash Filter (HC #428)
The best growth candidate from R2: Sharpe 2.10, CAGR 35.8%

Classifies each OOT day as green/red/flat based on SPY close-to-close.
Computes Sharpe per regime. Checks regime gap < 0.50.
If it fails, this strategy is just a bull market proxy.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r4'
os.makedirs(OUT_DIR, exist_ok=True)

START = '2015-01-01'
END = '2026-07-14'


def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 10:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'win_rate': 0, 'pf': 0, 'n_days': len(rets)}
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = rets.mean() * 252 / downside if downside > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    return {'name': name, 'sharpe': float(sharpe), 'sortino': float(sortino),
            'cagr': float(cagr), 'max_dd': float(max_dd), 'win_rate': float(wr),
            'pf': float(pf), 'n_days': len(rets), 'total_return_pct': float(total_ret * 100)}


def classify_regime(spy_ret_series, threshold=0.003):
    """Classify each day as green/red/flat based on SPY close-to-close."""
    regimes = pd.Series(index=spy_ret_series.index, dtype=str)
    regimes[spy_ret_series > threshold] = 'green'
    regimes[spy_ret_series < -threshold] = 'red'
    regimes[(spy_ret_series >= -threshold) & (spy_ret_series <= threshold)] = 'flat'
    return regimes


def main():
    print("=" * 70)
    print("REGIME TEST — MOMENTUM + CRASH FILTER (HC #428)")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ─── Reproduce the Momentum + Crash Filter strategy ─────────────────
    mom_tickers = ['QQQ', 'XLK', 'XLC', 'XLY', 'XLI', 'SMH', 'IGV', 'IWF']
    crash_tickers = ['^VIX', 'TLT', 'HYG', 'LQD', 'SPY']

    all_tickers = list(set(mom_tickers + crash_tickers))
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START, end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df
            print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")

    spy = data['SPY']
    common_idx = spy.index

    # Momentum signals
    mom_rets = pd.DataFrame()
    for t in mom_tickers:
        if t in data:
            mom_rets[t] = data[t]['Close'].reindex(common_idx).pct_change()

    # 12-1 month momentum
    mom_12_1 = pd.DataFrame()
    for t in mom_tickers:
        if t in data:
            p = data[t]['Close'].reindex(common_idx)
            mom_12_1[t] = p.shift(21) / p.shift(252) - 1

    # Crash indicators
    crash = pd.DataFrame(index=common_idx)
    if '^VIX' in data:
        vix = data['^VIX']['Close'].reindex(common_idx).ffill()
        crash['vix_above_25'] = (vix > 25).astype(int)
    if 'HYG' in data and 'LQD' in data:
        crash['credit_stress'] = (data['LQD']['Close'].reindex(common_idx).pct_change(20) -
                                   data['HYG']['Close'].reindex(common_idx).pct_change(20))
        crash['credit_alarm'] = (crash['credit_stress'] > crash['credit_stress'].rolling(252).quantile(0.9)).astype(int)

    crash['spy_below_ma200'] = (data['SPY']['Close'].reindex(common_idx) <
                                 data['SPY']['Close'].reindex(common_idx).rolling(200).mean()).astype(int)

    crash_filter = (crash.get('vix_above_25', pd.Series(0, index=common_idx)).astype(int) |
                   crash.get('credit_alarm', pd.Series(0, index=common_idx)).astype(int) |
                   crash['spy_below_ma200'].astype(int))

    # Monthly rebalance
    rebal_dates = common_idx[::21]

    strat_rets_list = []
    spy_daily_ret = data['SPY']['Close'].reindex(common_idx).pct_change()

    for i in range(1, len(rebal_dates)):
        dt = rebal_dates[i]
        prev_dt = rebal_dates[i-1]

        if dt not in mom_12_1.index:
            continue

        scores = mom_12_1.loc[dt].dropna()
        if len(scores) < 3:
            continue

        top3 = scores.nlargest(3).index.tolist()

        period_mask = (common_idx > prev_dt) & (common_idx <= dt)
        period_rets = mom_rets.loc[period_mask, top3].mean(axis=1)

        crash_mask = crash_filter.reindex(period_rets.index).fillna(0).astype(bool)
        period_rets[crash_mask] = 0  # go to cash

        strat_rets_list.append(period_rets)

    strat_all = pd.concat(strat_rets_list)
    strat_all = strat_all[~strat_all.index.duplicated(keep='first')]

    # ─── SPY regime classification ──────────────────────────────────────
    spy_ret_aligned = spy_daily_ret.reindex(strat_all.index)
    regimes = classify_regime(spy_ret_aligned)

    print(f"\n{'='*70}")
    print("REGIME CLASSIFICATION (SPY close-to-close)")
    print(f"{'='*70}")

    for r in ['green', 'red', 'flat']:
        n = (regimes == r).sum()
        pct = n / len(regimes) * 100
        print(f"  {r:>5}: {n:>5} days ({pct:.1f}%)")

    # ─── Per-regime Sharpe ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-REGIME PERFORMANCE")
    print(f"{'='*70}")

    regime_results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        r_rets = strat_all[mask]
        m = calc_metrics(r_rets, f'Mom+Crash ({r})')
        regime_results[r] = m
        print(f"\n  {r.upper()} days ({m['n_days']} days):")
        print(f"    Sharpe:  {m['sharpe']:.3f}")
        print(f"    Sortino: {m['sortino']:.3f}")
        print(f"    CAGR:    {m['cagr']:.1%}")
        print(f"    MaxDD:   {m['max_dd']:.1%}")
        print(f"    WinRate: {m['win_rate']:.1%}")
        print(f"    PF:      {m['pf']:.2f}")

    # ─── Regime Gap Test (HC #428) ──────────────────────────────────────
    sharpe_green = regime_results['green']['sharpe']
    sharpe_red = regime_results['red']['sharpe']
    sharpe_flat = regime_results['flat']['sharpe']

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else float('inf')

    print(f"\n{'='*70}")
    print("HC #428 REGIME-AGNOSTIC TEST")
    print(f"{'='*70}")
    print(f"  Sharpe_green: {sharpe_green:.3f}")
    print(f"  Sharpe_red:   {sharpe_red:.3f}")
    print(f"  Sharpe_flat:  {sharpe_flat:.3f}")
    print(f"  Regime gap:   {regime_gap:.3f}")
    print(f"  Threshold:    0.500")
    print(f"  PASS:         {'YES' if regime_gap < 0.50 else 'NO'}")

    if regime_gap >= 0.50:
        print(f"\n  VERDICT: FAIL — regime gap {regime_gap:.3f} >= 0.50")
        print(f"  This strategy is regime-dependent (likely a bull market proxy).")
        if sharpe_green > sharpe_red:
            print(f"  Sharpe is {sharpe_green:.2f} on green days vs {sharpe_red:.2f} on red days.")
            print(f"  The 'edge' is mostly directional equity exposure, not alpha.")
        else:
            print(f"  Sharpe is {sharpe_red:.2f} on red days vs {sharpe_green:.2f} on green days.")
            print(f"  Interesting — better on red days. May have genuine crash protection value.")
    else:
        print(f"\n  VERDICT: PASS — regime gap {regime_gap:.3f} < 0.50")
        print(f"  Strategy works across regimes. Proceed with deployment research.")

    # ─── Full period stats for comparison ───────────────────────────────
    full_m = calc_metrics(strat_all, 'Mom+Crash (full)')
    spy_m = calc_metrics(spy_ret_aligned.dropna(), 'SPY B&H')

    print(f"\n{'='*70}")
    print("FULL PERIOD COMPARISON")
    print(f"{'='*70}")
    print(f"  Strategy: Sharpe={full_m['sharpe']:.2f}, CAGR={full_m['cagr']:.1%}, "
          f"MaxDD={full_m['max_dd']:.1%}, Sortino={full_m['sortino']:.2f}")
    print(f"  SPY B&H:  Sharpe={spy_m['sharpe']:.2f}, CAGR={spy_m['cagr']:.1%}, "
          f"MaxDD={spy_m['max_dd']:.1%}, Sortino={spy_m['sortino']:.2f}")

    # ─── Per-day breakdown (for deeper analysis) ────────────────────────
    print(f"\n{'='*70}")
    print("PER-DAY SHARPE BREAKDOWN (rolling 21d)")
    print(f"{'='*70}")

    rolling_sharpe = strat_all.rolling(21).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    for regime in ['green', 'red', 'flat']:
        mask = regimes == regime
        rs = rolling_sharpe[mask].dropna()
        if len(rs) > 0:
            print(f"  {regime:>5}: median rolling Sharpe = {rs.median():.2f}, "
                  f"mean = {rs.mean():.2f}, pct_positive = {(rs > 0).mean():.1%}")

    # ─── Day concentration check (HC #344) ──────────────────────────────
    # Top 10% of days shouldn't contribute > 70% of returns
    sorted_rets = strat_all.sort_values(ascending=False)
    n_top = max(1, int(len(sorted_rets) * 0.10))
    top_10_pct_contribution = sorted_rets.head(n_top).sum() / sorted_rets[sorted_rets > 0].sum()

    print(f"\n{'='*70}")
    print("DAY CONCENTRATION (HC #344)")
    print(f"{'='*70}")
    print(f"  Top 10% of days contribute: {top_10_pct_contribution:.1%} of total gains")
    print(f"  Threshold: <= 70%")
    print(f"  {'PASS' if top_10_pct_contribution <= 0.70 else 'FAIL'}")

    # ─── Save results ───────────────────────────────────────────────────
    output = {
        'run_date': datetime.now().isoformat(),
        'strategy': 'Momentum + Crash Filter',
        'source': 'growth_new_signals_v1 / strategy 4',
        'full_period': full_m,
        'spy_benchmark': spy_m,
        'regime_results': regime_results,
        'regime_gap': float(regime_gap),
        'regime_gap_threshold': 0.50,
        'regime_pass': regime_gap < 0.50,
        'day_concentration': float(top_10_pct_contribution),
        'day_concentration_pass': top_10_pct_contribution <= 0.70,
        'overall_pass': regime_gap < 0.50 and top_10_pct_contribution <= 0.70,
        'regime_distribution': {
            'green': int((regimes == 'green').sum()),
            'red': int((regimes == 'red').sum()),
            'flat': int((regimes == 'flat').sum()),
        }
    }

    with open(os.path.join(OUT_DIR, 'regime_test_momentum_crash.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved.")
    return output


if __name__ == '__main__':
    main()
