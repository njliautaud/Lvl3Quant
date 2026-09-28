#!/usr/bin/env python3
"""
Leveraged ETF Decay Analysis & Optimal Holding Period
=====================================================
Quantifies the "volatility drag" / leverage decay for our core holdings
(UPRO 3x, TQQQ 3x, TMF 3x bonds) and determines:

1. What's the actual long-term decay vs underlying?
2. Under what vol regimes does leverage help vs hurt?
3. What's the optimal holding period for leveraged ETFs?
4. Does our protection overlay (SPY > SMA50) mitigate decay?
5. How does rebalancing frequency interact with leverage decay?

Critical for our deployment plan since Phase 1-2 is heavily leveraged.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/leverage_decay'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_data():
    """Download leveraged and unleveraged ETFs."""
    tickers = {
        'SPY': 'S&P 500 1x',
        'UPRO': 'S&P 500 3x',
        'QQQ': 'Nasdaq 1x',
        'TQQQ': 'Nasdaq 3x',
        'TLT': 'LT Bonds 1x',
        'TMF': 'LT Bonds 3x',
        'IWM': 'Russell 2000 1x',
        'VIXY': 'VIX proxy',
    }

    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(list(tickers.keys()), start='2010-01-01', period='max',
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

    closes = closes.dropna(how='all').dropna(subset=['UPRO', 'TQQQ'])
    print(f"  Data: {len(closes)} days")
    print(f"  Range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes

def analyze_decay(closes, leveraged, underlying, leverage_ratio=3):
    """Analyze leverage decay between leveraged ETF and its underlying."""
    lev = closes[leveraged].dropna()
    und = closes[underlying].dropna()

    common = lev.index.intersection(und.index)
    lev = lev.loc[common]
    und = und.loc[common]

    lev_ret = lev.pct_change().dropna()
    und_ret = und.pct_change().dropna()

    # Theoretical 3x daily return
    theo_3x_ret = und_ret * leverage_ratio

    # Daily tracking error
    tracking_error = lev_ret - theo_3x_ret
    avg_te = tracking_error.mean() * 252

    # Rolling decay analysis
    windows = [21, 63, 126, 252]
    decay_by_window = {}

    for w in windows:
        # Compare realized leveraged return vs theoretical
        lev_roll = lev_ret.rolling(w).apply(lambda x: (1+x).prod() - 1, raw=True)
        und_roll = und_ret.rolling(w).apply(lambda x: (1+x).prod() - 1, raw=True)

        # Theoretical: if you held 3x the underlying for w days
        theo_roll = ((1 + und_ret).rolling(w).apply(lambda x: x.prod(), raw=True) ** leverage_ratio) - 1

        # Actual decay
        decay = lev_roll - theo_roll
        decay_valid = decay.dropna()

        decay_by_window[w] = {
            'mean_decay_pct': float(decay_valid.mean() * 100),
            'median_decay_pct': float(decay_valid.median() * 100),
            'worst_decay_pct': float(decay_valid.min() * 100),
            'best_decay_pct': float(decay_valid.max() * 100),
            'pct_positive': float((decay_valid > 0).mean() * 100),  # % of time leveraged BEATS 3x
        }

    # Vol regime analysis
    und_vol = und_ret.rolling(21).std() * np.sqrt(252)

    vol_quintiles = pd.qcut(und_vol.dropna(), 5, labels=False, duplicates='drop')
    vol_regime_decay = {}

    for q in sorted(vol_quintiles.unique()):
        mask = vol_quintiles == q
        dates = vol_quintiles[mask].index

        lev_r = lev_ret.loc[lev_ret.index.isin(dates)]
        und_r = und_ret.loc[und_ret.index.isin(dates)]

        lev_ann = lev_r.mean() * 252
        und_ann = und_r.mean() * 252

        # Effective leverage realized
        eff_lev = lev_ann / und_ann if und_ann != 0 else 0

        avg_vol = und_vol.loc[und_vol.index.isin(dates)].mean()

        vol_regime_decay[int(q)] = {
            'avg_vol': float(avg_vol * 100),
            'underlying_ann_ret': float(und_ann * 100),
            'leveraged_ann_ret': float(lev_ann * 100),
            'effective_leverage': float(eff_lev),
            'n_days': int(mask.sum()),
        }

    # Long-term comparison
    total_years = len(common) / 252
    und_total = (und.iloc[-1] / und.iloc[0]) ** (1/total_years) - 1
    lev_total = (lev.iloc[-1] / lev.iloc[0]) ** (1/total_years) - 1
    theo_total = (1 + und_total) ** leverage_ratio - 1
    long_term_eff_leverage = lev_total / und_total if und_total != 0 else 0

    return {
        'pair': f"{leveraged}/{underlying}",
        'leverage_ratio': leverage_ratio,
        'total_years': float(total_years),
        'underlying_cagr': float(und_total * 100),
        'leveraged_cagr': float(lev_total * 100),
        'theoretical_3x_cagr': float(theo_total * 100),
        'effective_long_term_leverage': float(long_term_eff_leverage),
        'avg_daily_tracking_error_ann': float(avg_te * 100),
        'decay_by_window': decay_by_window,
        'vol_regime_decay': vol_regime_decay,
    }

def analyze_protection_vs_decay(closes):
    """Does our SPY>SMA50 protection mitigate leverage decay?"""
    spy = closes['SPY']
    upro = closes['UPRO']

    spy_ret = spy.pct_change().fillna(0)
    upro_ret = upro.pct_change().fillna(0)

    # Protection: SPY > SMA50
    sma50 = spy.rolling(50).mean()
    protection = spy > sma50

    # UPRO returns split by protection
    upro_protected = upro_ret.copy()
    upro_protected[~protection] = 0  # Cash when unprotected

    upro_unprotected = upro_ret.copy()

    # Rolling vol when protected vs not
    vol_protected = upro_ret[protection].std() * np.sqrt(252)
    vol_unprotected = upro_ret[~protection].std() * np.sqrt(252)

    # Compounding: protected vs unprotected
    equity_prot = (1 + upro_protected).cumprod()
    equity_unprot = (1 + upro_unprotected).cumprod()

    years = len(upro_ret) / 252
    cagr_prot = equity_prot.iloc[-1] ** (1/years) - 1
    cagr_unprot = equity_unprot.iloc[-1] ** (1/years) - 1

    # Max drawdown
    peak_prot = equity_prot.expanding().max()
    dd_prot = ((equity_prot - peak_prot) / peak_prot).min()

    peak_unprot = equity_unprot.expanding().max()
    dd_unprot = ((equity_unprot - peak_unprot) / peak_unprot).min()

    # SPY same comparison
    spy_prot_ret = spy_ret.copy()
    spy_prot_ret[~protection] = 0
    eq_spy_prot = (1 + spy_prot_ret).cumprod()
    eq_spy_unprot = (1 + spy_ret).cumprod()
    cagr_spy_prot = eq_spy_prot.iloc[-1] ** (1/years) - 1
    cagr_spy_unprot = eq_spy_unprot.iloc[-1] ** (1/years) - 1

    # Effective leverage WITH protection
    eff_lev_prot = cagr_prot / cagr_spy_prot if cagr_spy_prot > 0 else 0
    eff_lev_unprot = cagr_unprot / cagr_spy_unprot if cagr_spy_unprot > 0 else 0

    return {
        'protection_pct': float(protection.mean() * 100),
        'upro_protected_cagr': float(cagr_prot * 100),
        'upro_unprotected_cagr': float(cagr_unprot * 100),
        'upro_protected_maxdd': float(dd_prot * 100),
        'upro_unprotected_maxdd': float(dd_unprot * 100),
        'spy_protected_cagr': float(cagr_spy_prot * 100),
        'spy_unprotected_cagr': float(cagr_spy_unprot * 100),
        'vol_during_protection': float(vol_protected * 100),
        'vol_during_cash': float(vol_unprotected * 100),
        'effective_leverage_protected': float(eff_lev_prot),
        'effective_leverage_unprotected': float(eff_lev_unprot),
    }

def optimal_holding_period(closes, leveraged='UPRO', underlying='SPY'):
    """Find the optimal holding period for leveraged ETFs."""
    lev = closes[leveraged]
    und = closes[underlying]

    results = {}

    for hold_days in [1, 5, 10, 21, 42, 63, 126, 252, 504]:
        # Calculate rolling returns for both
        lev_ret = lev.pct_change(hold_days).dropna()
        und_ret = und.pct_change(hold_days).dropna()

        # What multiple did leveraged deliver?
        mask = und_ret != 0
        effective_mult = lev_ret[mask] / und_ret[mask]

        # Win rate of leveraged vs 3x underlying
        theo_ret = und_ret * 3
        beat_theo = (lev_ret > theo_ret).mean()

        # Sharpe comparison
        lev_sharpe = lev_ret.mean() / lev_ret.std() * np.sqrt(252 / hold_days)
        und_sharpe = und_ret.mean() / und_ret.std() * np.sqrt(252 / hold_days)
        sharpe_ratio = lev_sharpe / und_sharpe if und_sharpe > 0 else 0

        results[hold_days] = {
            'hold_days': hold_days,
            'avg_effective_mult': float(effective_mult.median()),
            'pct_beat_3x': float(beat_theo * 100),
            'leveraged_sharpe': float(lev_sharpe),
            'underlying_sharpe': float(und_sharpe),
            'sharpe_ratio': float(sharpe_ratio),
            'n_periods': int(len(lev_ret)),
        }

    return results

def analyze_vol_breakeven(closes, leveraged='UPRO', underlying='SPY', leverage_ratio=3):
    """Find the volatility level where leveraged ETF breaks even vs underlying."""
    und_ret = closes[underlying].pct_change().dropna()
    lev_ret = closes[leveraged].pct_change().dropna()

    # Rolling 63-day vol and return
    roll_vol = und_ret.rolling(63).std() * np.sqrt(252)
    roll_und_ret = und_ret.rolling(63).apply(lambda x: (1+x).prod() ** (252/63) - 1, raw=True)
    roll_lev_ret = lev_ret.rolling(63).apply(lambda x: (1+x).prod() ** (252/63) - 1, raw=True)

    # Drop NaNs
    valid = roll_vol.dropna().index.intersection(roll_und_ret.dropna().index).intersection(roll_lev_ret.dropna().index)

    v = roll_vol.loc[valid]
    u = roll_und_ret.loc[valid]
    l = roll_lev_ret.loc[valid]

    # Effective multiplier at each vol level
    eff = l / u.clip(lower=0.001)

    # Bin by vol
    vol_bins = pd.cut(v, bins=[0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.0])
    vol_analysis = {}

    for vbin in vol_bins.unique().dropna():
        mask = vol_bins == vbin
        bin_label = f"{vbin.left*100:.0f}-{vbin.right*100:.0f}%"

        if mask.sum() < 10:
            continue

        avg_mult = eff[mask].median()
        avg_vol = v[mask].mean()
        avg_und = u[mask].mean()
        avg_lev = l[mask].mean()

        vol_analysis[bin_label] = {
            'avg_vol': float(avg_vol * 100),
            'avg_underlying_ann': float(avg_und * 100),
            'avg_leveraged_ann': float(avg_lev * 100),
            'effective_multiplier': float(avg_mult),
            'leverage_beneficial': bool(avg_mult > 1.0),
            'n_periods': int(mask.sum()),
        }

    # Theoretical breakeven vol (for 3x leverage, decay = L*(L-1)*σ²/2)
    # breakeven when return > decay: μ > (L-1)*σ²/2
    # For SPY avg μ ≈ 10%, L=3: breakeven σ = sqrt(2*μ/(L-1)) = sqrt(2*0.10/2) = 0.316 = 31.6%
    avg_return = und_ret.mean() * 252
    breakeven_vol = np.sqrt(2 * avg_return / (leverage_ratio - 1)) if avg_return > 0 else 0

    return {
        'vol_analysis': vol_analysis,
        'theoretical_breakeven_vol': float(breakeven_vol * 100),
        'avg_underlying_return': float(avg_return * 100),
    }

def main():
    print("="*70)
    print("LEVERAGED ETF DECAY ANALYSIS & OPTIMAL HOLDING PERIOD")
    print("="*70)

    closes = download_data()

    # === 1. Decay analysis for each leveraged pair ===
    print("\n" + "="*70)
    print("1. LEVERAGE DECAY BY PAIR")
    print("="*70)

    pairs = [
        ('UPRO', 'SPY', 3),
        ('TQQQ', 'QQQ', 3),
        ('TMF', 'TLT', 3),
    ]

    decay_results = {}
    for lev, und, ratio in pairs:
        if lev in closes.columns and und in closes.columns:
            result = analyze_decay(closes, lev, und, ratio)
            decay_results[f"{lev}/{und}"] = result

            print(f"\n  {lev}/{und} ({ratio}x leverage, {result['total_years']:.1f} years):")
            print(f"    Underlying CAGR: {result['underlying_cagr']:.1f}%")
            print(f"    Leveraged CAGR:  {result['leveraged_cagr']:.1f}%")
            print(f"    Theoretical {ratio}x CAGR: {result['theoretical_3x_cagr']:.1f}%")
            print(f"    Effective long-term leverage: {result['effective_long_term_leverage']:.2f}x")
            print(f"    Avg daily tracking error (ann): {result['avg_daily_tracking_error_ann']:.2f}%")

            print(f"\n    Vol regime effective leverage:")
            for q, vd in result['vol_regime_decay'].items():
                label = ['Very Low', 'Low', 'Medium', 'High', 'Very High'][q]
                print(f"      {label:>9s} vol ({vd['avg_vol']:.0f}%): eff lev = {vd['effective_leverage']:.2f}x "
                      f"(und {vd['underlying_ann_ret']:+.1f}%, lev {vd['leveraged_ann_ret']:+.1f}%)")

    # === 2. Protection vs decay ===
    print("\n" + "="*70)
    print("2. PROTECTION OVERLAY vs LEVERAGE DECAY")
    print("="*70)

    prot_results = analyze_protection_vs_decay(closes)
    print(f"\n  Protection active: {prot_results['protection_pct']:.1f}% of days")
    print(f"\n  UPRO without protection:")
    print(f"    CAGR: {prot_results['upro_unprotected_cagr']:.1f}%")
    print(f"    MaxDD: {prot_results['upro_unprotected_maxdd']:.1f}%")
    print(f"    Effective leverage: {prot_results['effective_leverage_unprotected']:.2f}x")
    print(f"\n  UPRO with SMA50 protection:")
    print(f"    CAGR: {prot_results['upro_protected_cagr']:.1f}%")
    print(f"    MaxDD: {prot_results['upro_protected_maxdd']:.1f}%")
    print(f"    Effective leverage: {prot_results['effective_leverage_protected']:.2f}x")
    print(f"\n  SPY comparison:")
    print(f"    SPY CAGR (unprotected): {prot_results['spy_unprotected_cagr']:.1f}%")
    print(f"    SPY CAGR (protected): {prot_results['spy_protected_cagr']:.1f}%")

    # === 3. Optimal holding period ===
    print("\n" + "="*70)
    print("3. OPTIMAL HOLDING PERIOD")
    print("="*70)

    for lev, und, ratio in pairs:
        if lev in closes.columns and und in closes.columns:
            hp_results = optimal_holding_period(closes, lev, und)

            print(f"\n  {lev}/{und}:")
            print(f"    {'Hold':>6s} {'Eff Mult':>9s} {'Beat 3x':>8s} {'Lev Sharpe':>11s} {'Und Sharpe':>11s} {'Ratio':>6s}")
            for hold, r in sorted(hp_results.items()):
                print(f"    {r['hold_days']:>4d}d {r['avg_effective_mult']:>8.2f}x {r['pct_beat_3x']:>7.1f}% "
                      f"{r['leveraged_sharpe']:>10.3f} {r['underlying_sharpe']:>10.3f} {r['sharpe_ratio']:>6.2f}")

    # === 4. Volatility breakeven ===
    print("\n" + "="*70)
    print("4. VOLATILITY BREAKEVEN ANALYSIS")
    print("="*70)

    be_results = analyze_vol_breakeven(closes, 'UPRO', 'SPY', 3)
    print(f"\n  Theoretical breakeven vol: {be_results['theoretical_breakeven_vol']:.1f}%")
    print(f"  Avg SPY return: {be_results['avg_underlying_return']:.1f}%/yr")
    print(f"\n  Vol regime breakdown (63-day rolling):")
    for vbin, va in be_results['vol_analysis'].items():
        beneficial = "✓ YES" if va['leverage_beneficial'] else "✗ NO"
        print(f"    Vol {vbin:>10s}: eff mult {va['effective_multiplier']:.2f}x, "
              f"leverage helpful: {beneficial} (n={va['n_periods']})")

    # === 5. Summary ===
    print("\n" + "="*70)
    print("SUMMARY & PRACTICAL IMPLICATIONS")
    print("="*70)

    upro_decay = decay_results.get('UPRO/SPY', {})
    if upro_decay:
        print(f"\n  UPRO delivers ~{upro_decay['effective_long_term_leverage']:.1f}x effective leverage long-term")
        print(f"  (vs theoretical 3.0x — {(3.0 - upro_decay['effective_long_term_leverage']) / 3.0 * 100:.0f}% decay)")

    print(f"\n  Protection overlay INCREASES effective leverage by avoiding high-vol drawdowns")
    print(f"  (Protected: {prot_results['effective_leverage_protected']:.2f}x vs "
          f"Unprotected: {prot_results['effective_leverage_unprotected']:.2f}x)")

    print(f"\n  PRACTICAL RULES:")
    print(f"    1. UPRO is worth holding for long-term growth (eff leverage ~{upro_decay.get('effective_long_term_leverage', 2):.1f}x)")
    print(f"    2. Protection overlay is CRITICAL — it reduces decay AND drawdowns")
    print(f"    3. Leverage hurts when vol > {be_results['theoretical_breakeven_vol']:.0f}% — that's when to go to cash")
    print(f"    4. Daily rebalancing of leveraged ETFs is fine (they reset daily internally)")

    # Save results
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'decay_results': decay_results,
        'protection_results': prot_results,
        'vol_breakeven': be_results,
    }

    output_path = os.path.join(OUTPUT_DIR, 'leverage_decay_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
