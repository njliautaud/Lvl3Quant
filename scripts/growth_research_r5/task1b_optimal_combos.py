#!/usr/bin/env python3
"""
R5 TASK 1B: OPTIMAL COMBINATIONS
Based on findings from Task 1:
- Best base: 4-asset (SPY/TLT/GLD/DBC), regime gap 0.175
- Best tilt: momentum (Sharpe 0.870, CAGR 14.9%)
- Composite + vol target gives best regime gap (0.048!)
- Leverage scales linearly: 3x hits 17.9% CAGR, 3.5x hits 20.4%
- Leveraged ETFs beat margin (no carrying cost drag)

This script tests the top combinations more precisely, including:
1. Momentum tilt at various leverage levels
2. Composite tilt + vol target at various leverage levels
3. Best of both worlds: momentum tilt + vol target
4. Real-world leveraged ETF simulation
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
import json
import os

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r5'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2010-01-01'
END = '2026-07-14'


def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
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
    if max_dd < -1.0:
        max_dd = -1.0
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    return {'name': name, 'sharpe': float(sharpe), 'sortino': float(sortino),
            'cagr': float(cagr), 'max_dd': float(max_dd), 'win_rate': float(wr),
            'pf': float(pf), 'n_days': int(len(rets)),
            'cagr_pct': float(cagr * 100)}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        m = calc_metrics(strat_rets[mask], f'{name} ({r})')
        results[r] = m
    sg = results['green']['sharpe']
    sr = results['red']['sharpe']
    max_s = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_s if max_s > 0 else float('inf')
    return {
        'per_regime': results,
        'sharpe_green': float(sg), 'sharpe_red': float(sr),
        'sharpe_flat': float(results['flat']['sharpe']),
        'regime_gap': float(gap), 'regime_pass': gap < 0.50,
    }


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def run_rp(rets, prices, assets, leverage, tilt_mode='none', vol_target=None):
    """Run walk-forward risk parity. No margin cost (leveraged ETF implementation)."""
    common_idx = rets.index

    # Pre-compute signals
    mom_12_1 = prices.shift(21) / prices.shift(252) - 1
    carry_63 = prices.pct_change(63)
    rsi_21 = pd.DataFrame({a: compute_rsi(prices[a], 21) for a in assets})
    realized_vol_63 = rets.rolling(63).std() * np.sqrt(252)

    strat_rets_list = []
    rebal_dates = list(range(TRAIN_DAYS + 252, len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        lookback = rets[assets].iloc[max(0, i-63):i]
        vols = lookback.std()
        vols = vols.replace(0, np.nan).dropna()
        if len(vols) == 0:
            continue
        inv_vol = 1.0 / vols
        base_w = inv_vol / inv_vol.sum() * leverage

        if tilt_mode == 'none':
            final_w = base_w
        else:
            tilt_scores = pd.Series(0.0, index=assets)
            for a in assets:
                score = 0.0
                n = 0
                if tilt_mode in ['momentum', 'mom_voltgt', 'composite']:
                    m = mom_12_1[a].iloc[i] if i < len(mom_12_1) and a in mom_12_1 else np.nan
                    if not np.isnan(m):
                        score += np.clip(m, -0.5, 0.5)
                        n += 1
                if tilt_mode in ['carry', 'composite']:
                    c = carry_63[a].iloc[i] if i < len(carry_63) and a in carry_63 else np.nan
                    if not np.isnan(c):
                        score += np.clip(c, -0.5, 0.5)
                        n += 1
                if tilt_mode in ['meanrev', 'composite']:
                    r = rsi_21[a].iloc[i] if i < len(rsi_21) and a in rsi_21 else np.nan
                    if not np.isnan(r):
                        score += (50 - r) / 100
                        n += 1
                tilt_scores[a] = score / max(n, 1)

            tilt_rank = tilt_scores.rank()
            if tilt_rank.std() > 0:
                tilt_factor = 1.0 + (tilt_rank - tilt_rank.mean()) / tilt_rank.std() * 0.3
            else:
                tilt_factor = pd.Series(1.0, index=assets)
            tilt_factor = tilt_factor.clip(0.5, 1.5)
            final_w = base_w * tilt_factor
            final_w = final_w / final_w.sum() * leverage

        # Vol targeting
        if vol_target is not None:
            port_vol = (lookback * base_w).sum(axis=1).std() * np.sqrt(252)
            if port_vol > 0:
                vol_scale = vol_target / port_vol
                vol_scale = np.clip(vol_scale, 0.5, 3.0)
                final_w = final_w * vol_scale

        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            next_ret = rets[assets].iloc[j + 1] if j + 1 < len(rets) else pd.Series(0, index=assets)
            port_ret = (final_w * next_ret).sum()
            strat_rets_list.append({'date': common_idx[j], 'ret': port_ret})

    strat_s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()
    return strat_s


def main():
    print("=" * 70)
    print("R5 TASK 1B: OPTIMAL RISK PARITY COMBINATIONS")
    print("=" * 70)

    tickers = ['SPY', 'TLT', 'GLD', 'DBC']
    print("\nDownloading data...")
    data = {}
    for t in tickers:
        df = yf.download(t, start=START, end=END, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df
        print(f"  {t}: {len(df)} days")

    common_idx = data['SPY'].index
    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for a in tickers:
        prices[a] = data[a]['Close'].reindex(common_idx)
        rets[a] = prices[a].pct_change()
    spy_ret = rets['SPY']

    # Test grid
    configs = []
    for lev in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for tilt in ['none', 'momentum', 'composite', 'mom_voltgt']:
            for vt in [None, 0.12, 0.15, 0.20]:
                label = f'L{lev}_{tilt}_vt{vt}'
                configs.append((label, lev, tilt, vt))

    results = []
    for label, lev, tilt, vt in configs:
        s = run_rp(rets, prices, tickers, leverage=lev, tilt_mode=tilt, vol_target=vt)
        m = calc_metrics(s, label)
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), label)
        results.append({
            'label': label, 'leverage': lev, 'tilt': tilt, 'vol_target': vt,
            'sharpe': m['sharpe'], 'sortino': m['sortino'],
            'cagr_pct': m['cagr_pct'], 'max_dd': m['max_dd'],
            'win_rate': m['win_rate'], 'pf': m['pf'],
            'regime_gap': rt['regime_gap'],
            'sharpe_green': rt['sharpe_green'], 'sharpe_red': rt['sharpe_red'],
        })

    df = pd.DataFrame(results)

    # Sort by CAGR within regime-viable (<0.50 gap)
    viable = df[df['regime_gap'] < 0.50].sort_values('cagr_pct', ascending=False)
    all_sorted = df.sort_values('sharpe', ascending=False)

    print(f"\n{'='*90}")
    print("TOP 10 REGIME-VIABLE CONFIGS (gap < 0.50):")
    print(f"{'='*90}")
    print(f"{'Label':<35} {'Sharpe':>7} {'Sort':>6} {'CAGR%':>7} {'MaxDD%':>8} {'WR%':>5} {'PF':>5} {'Gap':>6}")
    print("-" * 90)
    for _, r in viable.head(10).iterrows():
        print(f"{r['label']:<35} {r['sharpe']:>7.3f} {r['sortino']:>6.3f} {r['cagr_pct']:>7.1f} "
              f"{r['max_dd']*100:>8.1f} {r['win_rate']*100:>5.1f} {r['pf']:>5.2f} {r['regime_gap']:>6.3f}")

    print(f"\n{'='*90}")
    print("TOP 10 BY SHARPE (ALL):")
    print(f"{'='*90}")
    print(f"{'Label':<35} {'Sharpe':>7} {'Sort':>6} {'CAGR%':>7} {'MaxDD%':>8} {'WR%':>5} {'PF':>5} {'Gap':>6}")
    print("-" * 90)
    for _, r in all_sorted.head(10).iterrows():
        print(f"{r['label']:<35} {r['sharpe']:>7.3f} {r['sortino']:>6.3f} {r['cagr_pct']:>7.1f} "
              f"{r['max_dd']*100:>8.1f} {r['win_rate']*100:>5.1f} {r['pf']:>5.2f} {r['regime_gap']:>6.3f}")

    # EFFICIENT FRONTIER: for each leverage level, show the best tilt
    print(f"\n{'='*90}")
    print("EFFICIENT FRONTIER BY LEVERAGE:")
    print(f"{'='*90}")
    for lev in [1.5, 2.0, 2.5, 3.0, 3.5]:
        lev_df = df[df['leverage'] == lev]
        best_sharpe = lev_df.loc[lev_df['sharpe'].idxmax()]
        best_cagr_viable = lev_df[lev_df['regime_gap'] < 0.50]
        if len(best_cagr_viable) > 0:
            best_cv = best_cagr_viable.loc[best_cagr_viable['cagr_pct'].idxmax()]
            print(f"  {lev}x: Best Sharpe={best_sharpe['label']} (S={best_sharpe['sharpe']:.3f}, "
                  f"CAGR={best_sharpe['cagr_pct']:.1f}%, Gap={best_sharpe['regime_gap']:.3f}) | "
                  f"Best viable CAGR={best_cv['label']} (CAGR={best_cv['cagr_pct']:.1f}%, Gap={best_cv['regime_gap']:.3f})")
        else:
            print(f"  {lev}x: Best Sharpe={best_sharpe['label']} (S={best_sharpe['sharpe']:.3f}, "
                  f"CAGR={best_sharpe['cagr_pct']:.1f}%, Gap={best_sharpe['regime_gap']:.3f}) | No viable CAGR configs")

    # Save
    out_file = os.path.join(OUT_DIR, 'task1b_optimal_combos.json')
    df.to_json(out_file, orient='records', indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
