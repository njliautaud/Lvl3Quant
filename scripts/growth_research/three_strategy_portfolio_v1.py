#!/usr/bin/env python3
"""
Three-Strategy Combined Portfolio v1
======================================

Combines three validated, truly different return streams:
1. Sector ETF Momentum (directional, trend-following) — Sharpe 0.79, CAGR 13.1%
2. Alt Trend Following (non-equity bonds/commodities) — Sharpe 0.75, CAGR 2.9%, Corr(SPY) 0.11
3. SPY Iron Condor Income (non-directional, theta decay) — Sharpe 2.43, CAGR 11.7%

These should have LOW mutual correlation:
- Momentum vs Alt Trend: 0.38 (measured in v4)
- Momentum vs Iron Condor: should be low (non-directional)
- Alt Trend vs Iron Condor: should be very low

Tests:
A. Equal weight 33/33/33
B. Growth tilt 50/20/30 (more equity momentum)
C. Income tilt 30/20/50 (more iron condor)
D. Risk parity
E. Regime-adaptive (shift weights in bear markets)
F. Min-variance walk-forward
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE',
    'XLC', 'QQQ', 'IWM', 'MDY', 'EFA', 'EEM', 'GLD', 'TLT', 'HYG', 'IYR',
    'VNQ', 'DBC',
]
DEFENSIVE = {'GLD', 'TLT', 'XLU', 'XLP'}
RISK_ON = {'XLK', 'QQQ', 'XLY', 'IWM', 'EEM'}
ALT_ASSETS = ['TLT', 'IEF', 'TIP', 'SHY', 'LQD', 'GLD', 'SLV', 'DBC', 'USO', 'VNQ', 'UUP']


def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S-K) if opt=='call' else max(0, K-S)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt=='call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def strike_at_delta(S, T, sigma, target_delta, r=0.04, opt='call'):
    if T <= 0: return S
    lo, hi = (S*0.8, S*1.5) if opt=='call' else (S*0.5, S*1.2)
    for _ in range(50):
        mid = (lo+hi)/2
        d1 = (np.log(S/mid) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        if opt=='call':
            if norm.cdf(d1) > target_delta: lo=mid
            else: hi=mid
        else:
            if abs(-norm.cdf(-d1)) > target_delta: hi=mid
            else: lo=mid
    return round((lo+hi)/2)


def simulate_equity_momentum(close, spy_close):
    """Monthly equity momentum with defensive shift, top 3."""
    monthly = close.resample('ME').last().dropna(how='all')
    monthly_rets = monthly.pct_change()
    spy_monthly = spy_close.resample('ME').last()
    spy_sma = spy_close.rolling(200).mean().resample('ME').last()

    returns, dates, regimes = [], [], []
    for i in range(13, len(monthly) - 1):
        date = monthly.index[i]
        scores = {}
        for etf in monthly.columns:
            try:
                p12 = float(monthly.iloc[i-12][etf])
                p1 = float(monthly.iloc[i-1][etf])
                if pd.isna(p12) or pd.isna(p1) or p12 == 0: continue
                scores[etf] = (p1/p12) - 1
            except: continue

        if len(scores) < 3:
            returns.append(0); dates.append(date); regimes.append('bull'); continue

        sp = spy_monthly.iloc[i]; ss = spy_sma.iloc[i]
        is_bear = not pd.isna(sp) and not pd.isna(ss) and sp < ss
        regime = 'bear' if is_bear else 'bull'

        if is_bear:
            for etf in scores:
                if etf in DEFENSIVE: scores[etf] += 0.05
                elif etf in RISK_ON: scores[etf] -= 0.03

        picks = [r[0] for r in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        rets = [monthly_rets.iloc[i+1][e] for e in picks if e in monthly_rets.columns and not pd.isna(monthly_rets.iloc[i+1][e])]
        ret = np.mean(rets) - 0.002 * 0.3 if rets else 0

        returns.append(ret); dates.append(date); regimes.append(regime)

    return np.array(returns), dates, np.array(regimes)


def simulate_alt_trend(close, spy_close):
    """Non-equity SMA200 trend, vol-target 8%."""
    valid = [a for a in ALT_ASSETS if a in close.columns]
    alt_close = close[valid].dropna(how='all')
    monthly = alt_close.resample('ME').last().dropna(how='all')
    monthly_rets = monthly.pct_change()
    sma200 = alt_close.rolling(200).mean().resample('ME').last()
    daily_rets = alt_close.pct_change()
    rolling_vol = (daily_rets.rolling(63).std() * np.sqrt(252)).resample('ME').last()
    spy_monthly = spy_close.resample('ME').last()
    spy_sma = spy_close.rolling(200).mean().resample('ME').last()

    returns, dates, regimes = [], [], []
    for i in range(13, len(monthly) - 1):
        date = monthly.index[i]
        longs = {}
        for a in monthly.columns:
            try:
                p = float(monthly.iloc[i][a]); s = float(sma200.iloc[i][a])
                if pd.isna(p) or pd.isna(s): continue
                if p > s:
                    v = float(rolling_vol.iloc[i][a]) if a in rolling_vol.columns else 0.12
                    if pd.isna(v) or v < 0.01: v = 0.12
                    longs[a] = v
            except: continue

        sp = spy_monthly.iloc[i]; ss = spy_sma.iloc[i]
        regime = 'bear' if (not pd.isna(sp) and not pd.isna(ss) and sp < ss) else 'bull'

        if not longs:
            returns.append(0); dates.append(date); regimes.append(regime); continue

        inv_vols = {a: 1.0/v for a, v in longs.items()}
        total = sum(inv_vols.values())
        raw = {a: iv/total for a, iv in inv_vols.items()}
        port_vol = sum(raw[a]*longs[a] for a in raw)
        scale = min(0.08/(port_vol+1e-10), 1.5)
        weights = {a: w*scale for a, w in raw.items()}
        tw = sum(weights.values())
        if tw > 1.0: weights = {a: w/tw for a, w in weights.items()}

        ret = sum(weights.get(a, 0) * (monthly_rets.iloc[i+1][a] if not pd.isna(monthly_rets.iloc[i+1][a]) else 0)
                  for a in weights if a in monthly_rets.columns and i+1 < len(monthly_rets))
        ret -= 0.002 * 0.25

        returns.append(ret); dates.append(date); regimes.append(regime)

    return np.array(returns), dates, np.array(regimes)


def simulate_iron_condor(spy_close, vix_close):
    """Monthly iron condor returns (16-delta, $5 wide, IV rank > 30%)."""
    vix_rank = vix_close.rolling(252).apply(lambda x: (x.iloc[-1]-x.min())/(x.max()-x.min()+1e-10)*100)
    spy_sma200 = spy_close.rolling(200).mean()

    # Group into monthly periods
    monthly_dates = spy_close.resample('ME').last().index[13:]  # Skip first year

    returns, dates, regimes = [], [], []

    for m_date in monthly_dates[:-1]:
        # Check IV rank at month start
        loc = spy_close.index.searchsorted(m_date)
        if loc >= len(spy_close) - 22: break

        # Use start of month for entry
        entry_date = spy_close.index[max(0, loc - 20)]
        if entry_date not in vix_rank.index: continue
        ivr = vix_rank.loc[entry_date]
        if pd.isna(ivr) or ivr < 30:
            returns.append(0); dates.append(m_date)
            regime = 'bear' if float(spy_close.loc[entry_date]) < float(spy_sma200.loc[entry_date]) else 'bull'
            regimes.append(regime); continue

        S = float(spy_close.loc[entry_date])
        vix = float(vix_close.loc[entry_date])
        sigma = vix / 100
        T = 30/252

        cs = strike_at_delta(S, T, sigma, 0.16, opt='call')
        ps = strike_at_delta(S, T, sigma, 0.16, opt='put')
        cl = cs + 5; pl = ps - 5

        credit = (bs_price(S, cs, T, sigma, opt='call') - bs_price(S, cl, T, sigma, opt='call') +
                  bs_price(S, ps, T, sigma, opt='put') - bs_price(S, pl, T, sigma, opt='put'))

        if credit <= 0:
            returns.append(0); dates.append(m_date)
            regimes.append('bull'); continue

        # Check outcome at expiry
        exp_loc = min(loc, len(spy_close) - 1)
        S_exp = float(spy_close.iloc[exp_loc])

        call_loss = max(0, S_exp - cs) - max(0, S_exp - cl)
        put_loss = max(0, ps - S_exp) - max(0, pl - S_exp)
        pnl = credit - call_loss - put_loss

        # Apply 50% profit target (approximate — take 50% of credit if position profitable)
        if pnl > 0:
            pnl = min(pnl, credit * 0.5)

        max_loss = 5 - credit
        ret = pnl / max_loss  # Return on risk

        regime = 'bear' if S < float(spy_sma200.loc[entry_date]) else 'bull'
        returns.append(ret); dates.append(m_date); regimes.append(regime)

    return np.array(returns), dates, np.array(regimes)


def compute_metrics(returns, regimes, name):
    total_ret = np.prod(1+returns)-1
    n_years = len(returns)/12
    cagr = (1+total_ret)**(1/max(n_years,0.01))-1
    ann_vol = np.std(returns)*np.sqrt(12)
    sharpe = (np.mean(returns)*12)/(ann_vol+1e-10)
    down = returns[returns<0]
    sortino = (np.mean(returns)*12)/(np.std(down)*np.sqrt(12)+1e-10) if len(down)>0 else 999
    cum = np.cumprod(1+returns); peak = np.maximum.accumulate(cum)
    maxdd = ((cum-peak)/peak).min()
    wins = returns[returns>0]; losses = returns[returns<0]
    wr = len(wins)/len(returns)*100
    pf = abs(wins.sum()/(losses.sum()+1e-10))
    br = returns[regimes=='bull']; ber = returns[regimes=='bear']
    bs = (np.mean(br)*12)/(np.std(br)*np.sqrt(12)+1e-10) if len(br)>3 else 0
    bes = (np.mean(ber)*12)/(np.std(ber)*np.sqrt(12)+1e-10) if len(ber)>3 else 0
    r1 = abs(bs-bes)/max(abs(bs),abs(bes),0.01)
    return {'name':name,'sharpe':round(sharpe,2),'sortino':round(sortino,2),
            'cagr_pct':round(cagr*100,1),'maxdd_pct':round(maxdd*100,1),
            'wr_pct':round(wr,1),'pf':round(pf,2),'r1_gap':round(r1,3),
            'r1_pass':r1<=0.50,'bull_sharpe':round(bs,2),'bear_sharpe':round(bes,2),
            'returns':returns.tolist()}


def permutation_test(returns, n_perms=1000):
    real = np.mean(returns)/(np.std(returns)+1e-10)
    count = sum(1 for _ in range(n_perms)
                if np.mean(returns*np.random.choice([-1,1],len(returns)))/(np.std(returns)+1e-10) >= real)
    return count/n_perms


def main():
    import yfinance as yf
    fprint(f"Three-Strategy Portfolio v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("="*70)

    ALL = list(set(EQUITY_UNIVERSE + ALT_ASSETS + ['SPY', '^VIX']))
    raw = yf.download(ALL, start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex): close = raw['Close']
    else: close = raw

    spy = close['SPY']
    vix = close['^VIX'] if '^VIX' in close.columns else None

    fprint("\nSimulating strategies...")
    eq_ret, eq_dates, eq_reg = simulate_equity_momentum(close, spy)
    fprint(f"  Equity momentum: {len(eq_ret)} months, Sharpe {np.mean(eq_ret)*12/(np.std(eq_ret)*np.sqrt(12)+1e-10):.2f}")

    alt_ret, alt_dates, alt_reg = simulate_alt_trend(close, spy)
    fprint(f"  Alt trend: {len(alt_ret)} months, Sharpe {np.mean(alt_ret)*12/(np.std(alt_ret)*np.sqrt(12)+1e-10):.2f}")

    ic_ret, ic_dates, ic_reg = simulate_iron_condor(spy, vix)
    fprint(f"  Iron condor: {len(ic_ret)} months, Sharpe {np.mean(ic_ret)*12/(np.std(ic_ret)*np.sqrt(12)+1e-10):.2f}")

    # Align
    eq_map = dict(zip([str(d) for d in eq_dates], zip(eq_ret, eq_reg)))
    alt_map = dict(zip([str(d) for d in alt_dates], zip(alt_ret, alt_reg)))
    ic_map = dict(zip([str(d) for d in ic_dates], zip(ic_ret, ic_reg)))

    common = sorted(set(eq_map.keys()) & set(alt_map.keys()) & set(ic_map.keys()))
    fprint(f"  Common months: {len(common)}")

    eq_a = np.array([eq_map[d][0] for d in common])
    alt_a = np.array([alt_map[d][0] for d in common])
    ic_a = np.array([ic_map[d][0] for d in common])
    regs = np.array([eq_map[d][1] for d in common])

    # Correlations
    corr_eq_alt = np.corrcoef(eq_a, alt_a)[0,1]
    corr_eq_ic = np.corrcoef(eq_a, ic_a)[0,1]
    corr_alt_ic = np.corrcoef(alt_a, ic_a)[0,1]

    fprint(f"\n  Correlations:")
    fprint(f"    Equity ↔ Alt Trend: {corr_eq_alt:.3f}")
    fprint(f"    Equity ↔ Iron Condor: {corr_eq_ic:.3f}")
    fprint(f"    Alt Trend ↔ Iron Condor: {corr_alt_ic:.3f}")

    # Portfolio variants
    variants = [
        ('A_EqualWeight', 0.333, 0.333, 0.334),
        ('B_GrowthTilt', 0.50, 0.20, 0.30),
        ('C_IncomeTilt', 0.30, 0.20, 0.50),
        ('D_BalancedGrowth', 0.40, 0.25, 0.35),
        ('E_ConservIncome', 0.20, 0.30, 0.50),
    ]

    results = []
    for name, w_eq, w_alt, w_ic in variants:
        port = w_eq*eq_a + w_alt*alt_a + w_ic*ic_a
        r = compute_metrics(port, regs, name)
        r['w_equity'] = w_eq; r['w_alt'] = w_alt; r['w_ic'] = w_ic
        results.append(r)
        fprint(f"\n  {name} ({w_eq:.0%}/{w_alt:.0%}/{w_ic:.0%}):")
        fprint(f"    Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%, R1 {r['r1_gap']:.3f}")

    # Regime-adaptive
    regime_ret = []
    for i in range(len(regs)):
        if regs[i] == 'bear':
            port = 0.15*eq_a[i] + 0.35*alt_a[i] + 0.50*ic_a[i]  # More income/alts
        else:
            port = 0.45*eq_a[i] + 0.20*alt_a[i] + 0.35*ic_a[i]  # More growth
        regime_ret.append(port)
    r = compute_metrics(np.array(regime_ret), regs, 'F_RegimeAdaptive')
    results.append(r)
    fprint(f"\n  F_RegimeAdaptive: Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%")

    # Risk parity
    lookback = 12
    rp_ret = []
    for i in range(lookback, len(eq_a)):
        ev = np.std(eq_a[i-lookback:i])*np.sqrt(12)+1e-10
        av = np.std(alt_a[i-lookback:i])*np.sqrt(12)+1e-10
        iv = np.std(ic_a[i-lookback:i])*np.sqrt(12)+1e-10
        ie, ia, ii = 1/ev, 1/av, 1/iv
        tot = ie+ia+ii
        rp_ret.append((ie/tot)*eq_a[i] + (ia/tot)*alt_a[i] + (ii/tot)*ic_a[i])
    r = compute_metrics(np.array(rp_ret), regs[lookback:], 'G_RiskParity')
    results.append(r)
    fprint(f"\n  G_RiskParity: Sharpe {r['sharpe']}, CAGR {r['cagr_pct']}%, MaxDD {r['maxdd_pct']}%")

    # Adversarial
    fprint("\n" + "="*70 + "\nADVERSARIAL VALIDATION\n" + "="*70)
    for r in results:
        rets = np.array(r['returns'])
        r['perm_p'] = round(permutation_test(rets), 3)
        r['g1_pass'] = r['perm_p'] < 0.05
        r['g2_pass'] = r['r1_pass']
        n = len(rets); chunk = max(n//3,1)
        subs = [np.mean(rets[i*chunk:(i+1)*chunk])*12/(np.std(rets[i*chunk:(i+1)*chunk])*np.sqrt(12)+1e-10) for i in range(3)]
        r['g3_pass'] = all(s > 0 for s in subs)
        r['sub_sharpes'] = [round(s,2) for s in subs]
        if n > 5:
            nt = max(1,int(n*0.05)); tr = np.sort(rets)[nt:-nt]
            o = np.mean(rets)/(np.std(rets)+1e-10)
            t = np.mean(tr)/(np.std(tr)+1e-10)
            r['g4_pass'] = t > 0 and t/(o+1e-10) > 0.5
        else: r['g4_pass'] = False
        r['gates_passed'] = sum([r['g1_pass'],r['g2_pass'],r['g3_pass'],r['g4_pass']])
        fprint(f"  {r['name']}: G1={'PASS' if r['g1_pass'] else 'FAIL'}, G2={'PASS' if r['g2_pass'] else 'FAIL'}, "
               f"G3={'PASS' if r['g3_pass'] else 'FAIL'}, G4={'PASS' if r['g4_pass'] else 'FAIL'} → {r['gates_passed']}/4")

    # Summary
    fprint("\n" + "="*70 + "\nSUMMARY — THREE-STRATEGY PORTFOLIO\n" + "="*70)
    fprint(f"Correlations: Eq↔Alt={corr_eq_alt:.2f}, Eq↔IC={corr_eq_ic:.2f}, Alt↔IC={corr_alt_ic:.2f}")

    # Standalone comparison
    eq_m = compute_metrics(eq_a, regs, 'Equity_Only')
    alt_m = compute_metrics(alt_a, regs, 'Alt_Only')
    ic_m = compute_metrics(ic_a, regs, 'IC_Only')
    fprint(f"\n{'Name':<25} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Sort':>7} {'R1':>7} {'Gates':>6}")
    fprint("-"*75)
    fprint(f"{'[Equity Only]':<25} {eq_m['sharpe']:>7.2f} {eq_m['cagr_pct']:>6.1f}% {eq_m['maxdd_pct']:>6.1f}% {eq_m['sortino']:>6.2f} {eq_m['r1_gap']:>6.3f}    ---")
    fprint(f"{'[Alt Trend Only]':<25} {alt_m['sharpe']:>7.2f} {alt_m['cagr_pct']:>6.1f}% {alt_m['maxdd_pct']:>6.1f}% {alt_m['sortino']:>6.2f} {alt_m['r1_gap']:>6.3f}    ---")
    fprint(f"{'[Iron Condor Only]':<25} {ic_m['sharpe']:>7.2f} {ic_m['cagr_pct']:>6.1f}% {ic_m['maxdd_pct']:>6.1f}% {ic_m['sortino']:>6.2f} {ic_m['r1_gap']:>6.3f}    ---")
    fprint("-"*75)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['sharpe']:>7.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['sortino']:>6.2f} {r['r1_gap']:>6.3f} {r['gates_passed']:>4}/4")

    best = max(results, key=lambda x: x['gates_passed']*10 + x['sharpe'])
    fprint(f"\nBEST: {best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, MaxDD {best['maxdd_pct']}%, {best['gates_passed']}/4")

    save = {'correlations': {'eq_alt': round(corr_eq_alt,3), 'eq_ic': round(corr_eq_ic,3), 'alt_ic': round(corr_alt_ic,3)},
            'results': [{k:v for k,v in r.items() if k!='returns'} for r in results]}
    with open(RESULTS_DIR / 'three_strategy_portfolio_v1_results.json', 'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    main()
