#!/usr/bin/env python3
"""
Multi-Strategy Portfolio v4 — With Decorrelated Alt Trend
==========================================================

Prior findings:
- Multi-Strategy Portfolio v3: Regime-Adaptive Sharpe 1.00, MaxDD -15.7%, 2/4 gates
  - Growth strategies 92% correlated, limiting diversification
  - VIX strategies decorrelated but low returns

NEW: Alt Trend Following v1 achieved Corr(SPY) = 0.10-0.14, 4/4 gates.
Adding this should reduce portfolio drawdown and improve risk-adjusted returns.

Components:
1. Sector ETF Momentum v2 (LightGBM + defensive shift) — GROWTH
2. ETF Regime-Adaptive v1 (defensive shift top 5) — GROWTH (different method)
3. Alt Trend Following (SMA200 vol-target, bonds/commodities) — DECORRELATOR
4. Cross-Asset Trend (dual mom vol-target, all assets) — MODERATE DECORRELATOR
5. VIX Mean-Rev (optional income component)

The key question: does adding the decorrelated component improve portfolio-level
risk metrics enough to justify the lower absolute returns?

Tests:
A. 50/50 equity momentum + alt trend
B. 60/40 equity momentum + alt trend
C. 70/20/10 equity momentum + alt trend + VIX income
D. Risk parity across all strategies
E. Min-variance walk-forward
F. Regime-adaptive (more alt trend in bears, more equity in bulls)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_etf_momentum(close, spy_close, top_k=3, cost_bps=10):
    """Simplified ETF momentum with defensive shift (replicating validated strategy)."""
    monthly = close.resample('ME').last().dropna(how='all')
    monthly_rets = monthly.pct_change()

    sma200 = close.rolling(200).mean().resample('ME').last()
    spy_monthly = spy_close.resample('ME').last()
    spy_sma = spy_close.rolling(200).mean().resample('ME').last()

    DEFENSIVE = {'GLD', 'TLT', 'XLU', 'XLP'}
    RISK_ON = {'XLK', 'QQQ', 'XLY', 'IWM', 'EEM'}

    returns = []
    dates = []
    regimes = []

    for i in range(13, len(monthly) - 1):
        date = monthly.index[i]

        # 12-1 momentum
        scores = {}
        for etf in monthly.columns:
            try:
                p12 = float(monthly.iloc[i-12][etf])
                p1 = float(monthly.iloc[i-1][etf])
                p0 = float(monthly.iloc[i][etf])
                if any(pd.isna(x) or x == 0 for x in [p12, p1, p0]):
                    continue
                mom = (p1 / p12) - 1  # 12-1 momentum
                scores[etf] = mom
            except:
                continue

        if len(scores) < top_k:
            returns.append(0)
            dates.append(date)
            regimes.append('bull')
            continue

        # Regime detection
        sp = spy_monthly.iloc[i] if i < len(spy_monthly) else np.nan
        ss = spy_sma.iloc[i] if i < len(spy_sma) else np.nan
        is_bear = not pd.isna(sp) and not pd.isna(ss) and sp < ss
        regime = 'bear' if is_bear else 'bull'

        # Defensive shift in bears
        if is_bear:
            for etf in scores:
                if etf in DEFENSIVE:
                    scores[etf] += 0.05
                elif etf in RISK_ON:
                    scores[etf] -= 0.03

        # Pick top K
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        picks = [r[0] for r in ranked]

        # Return
        period_rets = []
        for etf in picks:
            if etf in monthly_rets.columns and i + 1 < len(monthly_rets):
                r = monthly_rets.iloc[i+1][etf]
                if not pd.isna(r):
                    period_rets.append(r)

        ret = np.mean(period_rets) if period_rets else 0
        ret -= cost_bps / 10000 * 2 * 0.3  # ~30% turnover

        returns.append(ret)
        dates.append(date)
        regimes.append(regime)

    return np.array(returns), dates, np.array(regimes)


def simulate_alt_trend(close, spy_close, vol_target=0.08, cost_bps=10):
    """Non-equity trend following (bonds/commodities/gold)."""
    ALT_ASSETS = ['TLT', 'IEF', 'TIP', 'SHY', 'LQD', 'GLD', 'SLV', 'DBC', 'USO', 'VNQ', 'UUP']
    valid = [a for a in ALT_ASSETS if a in close.columns]

    alt_close = close[valid].dropna(how='all')
    monthly = alt_close.resample('ME').last().dropna(how='all')
    monthly_rets = monthly.pct_change()
    sma200 = alt_close.rolling(200).mean().resample('ME').last()

    daily_rets = alt_close.pct_change()
    rolling_vol = (daily_rets.rolling(63).std() * np.sqrt(252)).resample('ME').last()

    spy_monthly = spy_close.resample('ME').last()
    spy_sma = spy_close.rolling(200).mean().resample('ME').last()

    returns = []
    dates = []
    regimes = []

    for i in range(13, len(monthly) - 1):
        date = monthly.index[i]

        # Find assets above SMA200
        longs = {}
        for asset in monthly.columns:
            try:
                price = float(monthly.iloc[i][asset])
                sma = float(sma200.iloc[i][asset])
                if pd.isna(price) or pd.isna(sma):
                    continue
                if price > sma:
                    v = float(rolling_vol.iloc[i][asset]) if asset in rolling_vol.columns else 0.12
                    if pd.isna(v) or v < 0.01:
                        v = 0.12
                    longs[asset] = v
            except:
                continue

        regime = 'bull'
        sp = spy_monthly.iloc[i] if i < len(spy_monthly) else np.nan
        ss = spy_sma.iloc[i] if i < len(spy_sma) else np.nan
        if not pd.isna(sp) and not pd.isna(ss) and sp < ss:
            regime = 'bear'

        if not longs:
            returns.append(0)
            dates.append(date)
            regimes.append(regime)
            continue

        # Vol-target weights
        inv_vols = {a: 1.0/v for a, v in longs.items()}
        total = sum(inv_vols.values())
        raw = {a: iv/total for a, iv in inv_vols.items()}
        port_vol = sum(raw[a] * longs[a] for a in raw)
        scale = min(vol_target / (port_vol + 1e-10), 1.5)
        weights = {a: w * scale for a, w in raw.items()}
        tw = sum(weights.values())
        if tw > 1.0:
            weights = {a: w/tw for a, w in weights.items()}

        ret = 0
        for a, w in weights.items():
            if a in monthly_rets.columns and i+1 < len(monthly_rets):
                r = monthly_rets.iloc[i+1][a]
                if not pd.isna(r):
                    ret += w * r

        ret -= cost_bps / 10000 * 2 * 0.25
        returns.append(ret)
        dates.append(date)
        regimes.append(regime)

    return np.array(returns), dates, np.array(regimes)


def compute_metrics(returns, regimes, name):
    """Compute standard metrics."""
    total_ret = np.prod(1 + returns) - 1
    n_years = len(returns) / 12
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = np.std(returns) * np.sqrt(12)
    sharpe = (np.mean(returns) * 12) / (ann_vol + 1e-10)

    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-10
    sortino = (np.mean(returns) * 12) / (downside_vol + 1e-10)

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    maxdd = dd.min()

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) * 100
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999
    calmar = cagr / abs(maxdd) if maxdd != 0 else 999

    # R1
    bull_rets = returns[regimes == 'bull']
    bear_rets = returns[regimes == 'bear']
    bull_sharpe = (np.mean(bull_rets)*12) / (np.std(bull_rets)*np.sqrt(12) + 1e-10) if len(bull_rets) > 3 else 0
    bear_sharpe = (np.mean(bear_rets)*12) / (np.std(bear_rets)*np.sqrt(12) + 1e-10) if len(bear_rets) > 3 else 0
    r1_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

    return {
        'name': name,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr*100, 1),
        'maxdd_pct': round(maxdd*100, 1),
        'wr_pct': round(wr, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'n_months': len(returns),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_gap <= 0.50,
        'returns': returns.tolist(),
    }


def permutation_test(returns, n_perms=1000):
    real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
    block_size = 3
    n_blocks = len(returns) // block_size
    if n_blocks < 4:
        return 1.0
    blocked = [returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    count = 0
    for _ in range(n_perms):
        perm = np.random.permutation(n_blocks)
        pr = np.concatenate([blocked[i] for i in perm])
        shift = np.random.randint(1, len(pr))
        pr = np.roll(pr, shift)
        for i in range(0, len(pr), block_size):
            if np.random.random() < 0.5:
                pr[i:i+block_size] = -pr[i:i+block_size]
        if np.mean(pr) / (np.std(pr) + 1e-10) >= real_sharpe:
            count += 1
    return count / n_perms


def main():
    import yfinance as yf

    fprint(f"Multi-Strategy Portfolio v4 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    # Download all needed data
    ALL_TICKERS = [
        'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
        'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
        'VNQ', 'DBC', 'SPY', 'IEF', 'TIP', 'SHY', 'LQD', 'SLV', 'USO', 'UUP',
    ]

    fprint("Downloading data...")
    raw = yf.download(ALL_TICKERS, start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy_close = close['SPY']

    # Simulate component strategies
    fprint("\nSimulating component strategies...")

    fprint("  1. Sector ETF Momentum (defensive shift, top 3)...")
    eq_ret, eq_dates, eq_regimes = simulate_etf_momentum(close, spy_close, top_k=3)

    fprint("  2. Alt Trend Following (non-equity, vol-target 8%)...")
    alt_ret, alt_dates, alt_regimes = simulate_alt_trend(close, spy_close, vol_target=0.08)

    # Align dates
    eq_dates_set = set(str(d) for d in eq_dates)
    alt_dates_set = set(str(d) for d in alt_dates)
    common_dates = sorted(eq_dates_set & alt_dates_set)

    fprint(f"\n  Equity momentum: {len(eq_ret)} months, Sharpe {np.mean(eq_ret)*12/(np.std(eq_ret)*np.sqrt(12)+1e-10):.2f}")
    fprint(f"  Alt trend: {len(alt_ret)} months, Sharpe {np.mean(alt_ret)*12/(np.std(alt_ret)*np.sqrt(12)+1e-10):.2f}")

    # Map returns by date
    eq_map = dict(zip([str(d) for d in eq_dates], zip(eq_ret, eq_regimes)))
    alt_map = dict(zip([str(d) for d in alt_dates], zip(alt_ret, alt_regimes)))

    # Build aligned arrays
    eq_aligned = np.array([eq_map[d][0] for d in common_dates])
    alt_aligned = np.array([alt_map[d][0] for d in common_dates])
    regimes = np.array([eq_map[d][1] for d in common_dates])

    # Correlation between strategies
    corr = np.corrcoef(eq_aligned, alt_aligned)[0, 1]
    fprint(f"\n  Correlation between strategies: {corr:.3f}")
    fprint(f"  (Previous growth-growth correlation was 0.92)")

    # Portfolio variants
    variants = [
        ('A_50_50', 0.50, 0.50),
        ('B_60_40', 0.60, 0.40),
        ('C_70_30', 0.70, 0.30),
        ('D_40_60', 0.40, 0.60),
        ('E_80_20', 0.80, 0.20),
    ]

    results = []

    for name, w_eq, w_alt in variants:
        port_ret = w_eq * eq_aligned + w_alt * alt_aligned
        r = compute_metrics(port_ret, regimes, name)
        r['w_equity'] = w_eq
        r['w_alt'] = w_alt
        r['corr'] = round(corr, 3)
        results.append(r)
        fprint(f"\n  {name} ({w_eq:.0%} equity, {w_alt:.0%} alt):")
        fprint(f"    Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%, "
               f"R1 gap {r['r1_gap']:.3f} {'PASS' if r['r1_pass'] else 'FAIL'}")

    # Regime-adaptive: more alt in bears, more equity in bulls
    fprint("\n  Regime-adaptive portfolio...")
    regime_ret = []
    for i in range(len(regimes)):
        if regimes[i] == 'bear':
            w_e, w_a = 0.30, 0.70  # More alts in bears
        else:
            w_e, w_a = 0.70, 0.30  # More equity in bulls
        regime_ret.append(w_e * eq_aligned[i] + w_a * alt_aligned[i])
    regime_ret = np.array(regime_ret)
    r = compute_metrics(regime_ret, regimes, 'F_RegimeAdaptive')
    r['corr'] = round(corr, 3)
    results.append(r)
    fprint(f"  F_RegimeAdaptive (70/30 bull, 30/70 bear):")
    fprint(f"    Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%, "
           f"R1 gap {r['r1_gap']:.3f} {'PASS' if r['r1_pass'] else 'FAIL'}")

    # Risk-parity: weight by inverse vol
    fprint("\n  Risk-parity portfolio...")
    rp_ret = []
    lookback = 12
    for i in range(lookback, len(eq_aligned)):
        eq_vol = np.std(eq_aligned[i-lookback:i]) * np.sqrt(12)
        alt_vol = np.std(alt_aligned[i-lookback:i]) * np.sqrt(12)
        if eq_vol < 0.01: eq_vol = 0.15
        if alt_vol < 0.01: alt_vol = 0.10
        w_e = (1/eq_vol) / (1/eq_vol + 1/alt_vol)
        w_a = 1 - w_e
        rp_ret.append(w_e * eq_aligned[i] + w_a * alt_aligned[i])
    rp_ret = np.array(rp_ret)
    rp_regimes = regimes[lookback:]
    r = compute_metrics(rp_ret, rp_regimes, 'G_RiskParity')
    r['corr'] = round(corr, 3)
    results.append(r)
    fprint(f"  G_RiskParity:")
    fprint(f"    Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%, "
           f"R1 gap {r['r1_gap']:.3f} {'PASS' if r['r1_pass'] else 'FAIL'}")

    # Adversarial validation
    fprint("\n" + "=" * 70)
    fprint("ADVERSARIAL VALIDATION")
    fprint("=" * 70)

    for r in results:
        rets = np.array(r['returns'])
        perm_p = permutation_test(rets)
        r['perm_p'] = round(perm_p, 3)
        r['g1_pass'] = perm_p < 0.05
        r['g2_pass'] = r['r1_pass']

        # Sub-period
        n = len(rets)
        chunk = n // 3
        subs = [np.mean(rets[i*chunk:(i+1)*chunk])*12 / (np.std(rets[i*chunk:(i+1)*chunk])*np.sqrt(12)+1e-10) for i in range(3)]
        r['g3_pass'] = all(s > 0 for s in subs)
        r['sub_sharpes'] = [round(s, 2) for s in subs]

        # Outlier
        n_trim = max(1, int(n * 0.05))
        trimmed = np.sort(rets)[n_trim:-n_trim]
        trim_s = np.mean(trimmed) / (np.std(trimmed) + 1e-10)
        orig_s = np.mean(rets) / (np.std(rets) + 1e-10)
        r['g4_pass'] = trim_s > 0 and (trim_s / (orig_s + 1e-10)) > 0.5
        r['trimmed_sharpe'] = round(trim_s, 2)

        gates = sum([r['g1_pass'], r['g2_pass'], r['g3_pass'], r['g4_pass']])
        r['gates_passed'] = gates

        fprint(f"\n{r['name']}:")
        fprint(f"  G1 Perm: {'PASS' if r['g1_pass'] else 'FAIL'} (p={r['perm_p']})")
        fprint(f"  G2 R1:   {'PASS' if r['g2_pass'] else 'FAIL'} (gap={r['r1_gap']})")
        fprint(f"  G3 Sub:  {'PASS' if r['g3_pass'] else 'FAIL'} (sharpes={r['sub_sharpes']})")
        fprint(f"  G4 Out:  {'PASS' if r['g4_pass'] else 'FAIL'} (trimmed={r['trimmed_sharpe']})")
        fprint(f"  GATES: {gates}/4")

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — MULTI-STRATEGY PORTFOLIO v4")
    fprint("=" * 70)
    fprint(f"Strategy correlation: {corr:.3f} (was 0.92 in v3)")
    fprint(f"\n{'Name':<25} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Sort':>7} {'R1':>7} {'Gates':>6}")
    fprint("-" * 75)

    # Also show standalone components for comparison
    eq_metrics = compute_metrics(eq_aligned, regimes, 'Equity_Momentum_Only')
    alt_metrics = compute_metrics(alt_aligned, regimes, 'Alt_Trend_Only')
    fprint(f"{'[Equity Mom Only]':<25} {eq_metrics['sharpe']:>7.2f} {eq_metrics['cagr_pct']:>6.1f}% "
           f"{eq_metrics['maxdd_pct']:>6.1f}% {eq_metrics['sortino']:>6.2f} {eq_metrics['r1_gap']:>6.3f}    ---")
    fprint(f"{'[Alt Trend Only]':<25} {alt_metrics['sharpe']:>7.2f} {alt_metrics['cagr_pct']:>6.1f}% "
           f"{alt_metrics['maxdd_pct']:>6.1f}% {alt_metrics['sortino']:>6.2f} {alt_metrics['r1_gap']:>6.3f}    ---")
    fprint("-" * 75)

    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['sortino']:>6.2f} {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    # KEY VERDICT
    fprint("\n" + "=" * 70)
    fprint("KEY FINDINGS")
    fprint("=" * 70)
    best = max(results, key=lambda x: x['gates_passed'] * 10 + x['sharpe'])
    fprint(f"Best portfolio: {best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, "
           f"MaxDD {best['maxdd_pct']}%, {best['gates_passed']}/4 gates")
    fprint(f"Strategy correlation: {corr:.3f} (down from 0.92 in v3)")
    fprint(f"Diversification benefit: MaxDD improved from {eq_metrics['maxdd_pct']:.1f}% (equity only) "
           f"to {best['maxdd_pct']:.1f}% (combined)")

    # Save
    save = [{k: v for k, v in r.items() if k != 'returns'} for r in results]
    RESULTS_PATH = RESULTS_DIR / 'multi_strategy_portfolio_v4_results.json'
    with open(RESULTS_PATH, 'w') as f:
        json.dump({'correlation': corr, 'results': save,
                   'equity_standalone': {k: v for k, v in eq_metrics.items() if k != 'returns'},
                   'alt_standalone': {k: v for k, v in alt_metrics.items() if k != 'returns'}},
                  f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
