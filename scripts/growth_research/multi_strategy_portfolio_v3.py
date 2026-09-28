#!/usr/bin/env python3
"""
Multi-Strategy Portfolio v3 — Comprehensive Portfolio Combining ALL Validated Strategies
Tests 6 allocation schemes with full adversarial validation.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from datetime import datetime
import json, os, sys
import yfinance as yf

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False

print(f"Multi-Strategy Portfolio v3 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


def dl(tickers, start='2018-01-01', end='2026-07-25'):
    """Download and return Close prices, handling yfinance quirks."""
    raw = yf.download(tickers if isinstance(tickers, list) else [tickers],
                      start=start, end=end, progress=False)
    close = raw['Close']
    if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
        return close.iloc[:, 0]
    return close


ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        'XLRE', 'XLC', 'QQQ', 'DIA', 'IWM', 'EEM', 'EFA', 'GLD', 'SLV',
        'DBC', 'TLT', 'HYG', 'LQD']
DEFENSIVE = {'GLD', 'TLT', 'XLU', 'XLP', 'LQD'}


def get_spy_regime():
    spy = dl('SPY')
    spy_m = spy.resample('ME').last()
    spy_sma = spy.rolling(200).mean().resample('ME').last()
    bear = (spy_m < spy_sma).astype(bool)
    return spy_m, bear


def momentum_ranking(monthly, top_k, bear_flags, defensive_boost=False):
    mom_12_1 = monthly.pct_change(12) - monthly.pct_change(1)
    rets, dates = [], []

    for i in range(13, len(monthly) - 1):
        dt = monthly.index[i]
        dt_next = monthly.index[i + 1]
        scores = {}
        for etf in monthly.columns:
            try:
                m = mom_12_1.loc[dt, etf]
                if pd.isna(m): continue
                score = float(m)
                if defensive_boost and dt in bear_flags.index and bool(bear_flags.loc[dt]):
                    score += 0.05 if etf in DEFENSIVE else -0.02
                scores[etf] = score
            except:
                continue

        if len(scores) < top_k: continue
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        month_rets = []
        for ticker, _ in top:
            try:
                r = float(monthly.loc[dt_next, ticker] / monthly.loc[dt, ticker] - 1)
                if not np.isnan(r): month_rets.append(r)
            except:
                continue

        if month_rets:
            rets.append(np.mean(month_rets) - 0.002)
            dates.append(dt_next)

    return pd.Series(rets, index=dates)


def build_sector_etf_momentum_v2(etf_data, bear_flags):
    print("\n[1/5] Sector ETF Momentum v2 (Top3 + Defensive Shift)...")
    monthly = etf_data.resample('ME').last().dropna(how='all')
    s = momentum_ranking(monthly, top_k=3, bear_flags=bear_flags, defensive_boost=True)
    s.name = 'SectorMom_v2'
    _ps(s)
    return s


def build_etf_regime_adaptive(etf_data, bear_flags):
    print("\n[2/5] ETF Regime-Adaptive (Top5 + Defensive Shift)...")
    monthly = etf_data.resample('ME').last().dropna(how='all')
    s = momentum_ranking(monthly, top_k=5, bear_flags=bear_flags, defensive_boost=True)
    s.name = 'RegimeAdaptive'
    _ps(s)
    return s


def build_vix_call_spread_income(bear_flags):
    print("\n[3/5] VIX Call Spread Income...")
    vix = dl('^VIX')
    vix_m = vix.resample('ME').last()
    rets, dates = [], []
    for i in range(1, len(vix_m) - 1):
        dt, dt_next = vix_m.index[i], vix_m.index[i + 1]
        v = float(vix_m.iloc[i])
        if v > 20:
            v_next = float(vix_m.iloc[i + 1])
            chg = (v_next - v) / v
            if chg < -0.05: ret = 0.03
            elif chg < 0.10: ret = 0.02 * (1 - chg / 0.10)
            else: ret = -0.05 * min(chg / 0.20, 1.0)
            rets.append(ret)
        else:
            rets.append(0.0)
        dates.append(dt_next)
    s = pd.Series(rets, index=dates, name='VIXCallSpread')
    _ps(s)
    return s


def build_vix_mean_reversion():
    print("\n[4/5] VIX Mean-Reversion...")
    vix = dl('^VIX')
    spy = dl('SPY')
    vix_m, spy_m = vix.resample('ME').last(), spy.resample('ME').last()
    common = vix_m.index.intersection(spy_m.index)
    rets, dates = [], []
    for i in range(1, len(common) - 1):
        dt, dt_next = common[i], common[i + 1]
        if float(vix_m.loc[dt]) > 30:
            rets.append(float(spy_m.loc[dt_next] / spy_m.loc[dt] - 1) - 0.001)
        else:
            rets.append(0.0)
        dates.append(dt_next)
    s = pd.Series(rets, index=dates, name='VIXMeanRev')
    _ps(s)
    return s


def build_covered_call_overlay(etf_data, bear_flags):
    print("\n[5/5] Covered Call Overlay...")
    monthly = etf_data.resample('ME').last().dropna(how='all')
    mom_12_1 = monthly.pct_change(12) - monthly.pct_change(1)
    rets, dates = [], []
    for i in range(13, len(monthly) - 1):
        dt, dt_next = monthly.index[i], monthly.index[i + 1]
        scores = {}
        for etf in monthly.columns:
            try:
                m = float(mom_12_1.loc[dt, etf])
                if not np.isnan(m): scores[etf] = m
            except:
                continue
        if len(scores) < 3: continue
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        month_rets = []
        for ticker, _ in top3:
            try:
                r = float(monthly.loc[dt_next, ticker] / monthly.loc[dt, ticker] - 1)
                month_rets.append(min(r, 0.03) + 0.011)  # cap + premium
            except:
                continue
        if month_rets:
            rets.append(np.mean(month_rets) - 0.002)
            dates.append(dt_next)
    s = pd.Series(rets, index=dates, name='CoveredCall')
    _ps(s)
    return s


def _ps(s):
    if len(s) > 1:
        sh = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
        cagr = ((1 + s).prod()) ** (12 / len(s)) - 1
        print(f"  {len(s)} months | Sharpe {sh:.2f} | CAGR {cagr:.1%}")
    else:
        print("  WARNING: Insufficient data")


def risk_parity_weights(cov):
    n = cov.shape[0]
    w0 = np.ones(n) / n
    def obj(w):
        pv = w @ cov @ w
        rc = w * (cov @ w)
        return np.sum((rc - pv / n) ** 2)
    res = minimize(obj, w0, method='SLSQP',
                   bounds=[(0.05, 0.60)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
    return res.x if res.success else w0


def min_var_weights(cov):
    n = cov.shape[0]
    w0 = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w, w0, method='SLSQP',
                   bounds=[(0.05, 0.60)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
    return res.x if res.success else w0


def max_sharpe_weights(ret_df):
    n = ret_df.shape[1]
    mu, cov = ret_df.mean().values, ret_df.cov().values
    w0 = np.ones(n) / n
    res = minimize(lambda w: -(w @ mu) / (np.sqrt(w @ cov @ w) + 1e-10), w0,
                   method='SLSQP', bounds=[(0.05, 0.60)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
    return res.x if res.success else w0


def adversarial_validation(returns, name, spy_ret_m):
    print(f"\n{'='*60}")
    print(f"VALIDATION: {name}")
    gates = 0

    real_sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

    # G1: Permutation
    vals = returns.values
    perm_count = sum(1 for _ in range(1000)
                     if np.mean(s := np.random.permutation(vals)) / (np.std(s) + 1e-10) * np.sqrt(12) >= real_sharpe)
    p_val = perm_count / 1000
    g1 = p_val < 0.05
    gates += g1
    print(f"  G1 Perm: p={p_val:.3f} {'✅' if g1 else '❌'}")

    # G2: Regime
    common = returns.index.intersection(spy_ret_m.index)
    bull = [d for d in common if float(spy_ret_m.loc[d]) > 0]
    bear = [d for d in common if float(spy_ret_m.loc[d]) <= 0]
    bull_sh = returns.loc[bull].mean() / returns.loc[bull].std() * np.sqrt(12) if len(bull) > 3 and returns.loc[bull].std() > 0 else 0
    bear_sh = returns.loc[bear].mean() / returns.loc[bear].std() * np.sqrt(12) if len(bear) > 3 and returns.loc[bear].std() > 0 else 0
    gap = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
    g2 = gap < 0.50
    gates += g2
    print(f"  G2 Regime: Bull {bull_sh:.2f}, Bear {bear_sh:.2f}, gap {gap:.3f} {'✅' if g2 else '❌'}")

    # G3: Sub-period
    mid = len(returns) // 2
    h1 = returns.iloc[:mid]
    h2 = returns.iloc[mid:]
    h1_sh = h1.mean() / h1.std() * np.sqrt(12) if h1.std() > 0 else 0
    h2_sh = h2.mean() / h2.std() * np.sqrt(12) if h2.std() > 0 else 0
    g3 = h1_sh > 0 and h2_sh > 0
    gates += g3
    print(f"  G3 Sub: H1 {h1_sh:.2f}, H2 {h2_sh:.2f} {'✅' if g3 else '❌'}")

    # G4: Outlier
    trimmed = returns[(returns >= returns.quantile(0.01)) & (returns <= returns.quantile(0.99))]
    trim_sh = trimmed.mean() / trimmed.std() * np.sqrt(12) if len(trimmed) > 3 and trimmed.std() > 0 else 0
    g4 = trim_sh > 0
    gates += g4
    print(f"  G4 Outlier: Trimmed {trim_sh:.2f} {'✅' if g4 else '❌'}")

    # Metrics
    cum = (1 + returns).cumprod()
    years = len(returns) / 12
    cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else 0
    max_dd = float(((cum / cum.cummax()) - 1).min())
    down = returns[returns < 0]
    sortino = float(returns.mean() / down.std() * np.sqrt(12)) if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    pf = gains / losses if losses > 0 else 99
    wr = float((returns > 0).mean())

    print(f"  Sharpe {real_sharpe:.2f} | Sortino {sortino:.2f} | CAGR {cagr:.1%} | MaxDD {max_dd:.1%} | "
          f"Calmar {calmar:.2f} | PF {pf:.2f} | WR {wr:.1%} | Gates {gates}/4")

    return {'sharpe': float(real_sharpe), 'sortino': sortino, 'cagr': cagr, 'max_dd': max_dd,
            'calmar': calmar, 'pf': pf, 'wr': wr, 'gates': gates, 'perm_p': float(p_val),
            'r1_gap': float(gap), 'bull_sharpe': float(bull_sh), 'bear_sharpe': float(bear_sh),
            'h1_sharpe': float(h1_sh), 'h2_sharpe': float(h2_sh), 'trimmed_sharpe': float(trim_sh)}


def main():
    print("Downloading data...")
    etf_data = dl(ETFS)
    spy_m, bear_flags = get_spy_regime()
    spy_ret = spy_m.pct_change()

    s1 = build_sector_etf_momentum_v2(etf_data, bear_flags)
    s2 = build_etf_regime_adaptive(etf_data, bear_flags)
    s3 = build_vix_call_spread_income(bear_flags)
    s4 = build_vix_mean_reversion()
    s5 = build_covered_call_overlay(etf_data, bear_flags)

    df = pd.DataFrame({'SectorMom_v2': s1, 'RegimeAdaptive': s2,
                       'VIXCallSpread': s3, 'VIXMeanRev': s4, 'CoveredCall': s5}).dropna()

    print(f"\n{'='*70}")
    print(f"COMBINED: {len(df)} months ({df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')})")
    print(f"\nCORRELATIONS:\n{df.corr().round(3).to_string()}")

    print(f"\nINDIVIDUAL STATS:")
    for c in df.columns:
        s = df[c]
        sh = s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0
        cagr = ((1 + s).prod()) ** (12 / len(s)) - 1
        mdd = ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()
        print(f"  {c:20s}: Sharpe {sh:6.2f}, CAGR {cagr:7.1%}, MaxDD {mdd:7.1%}")

    results = {}
    n = len(df.columns)

    # A. Equal Weight
    w = np.ones(n) / n
    print(f"\n{'='*70}\nA. EQUAL WEIGHT")
    results['A_EqualWeight'] = adversarial_validation(
        pd.Series((df.values * w).sum(axis=1), index=df.index), "Equal Weight", spy_ret)

    # B. Risk Parity
    w = risk_parity_weights(df.cov().values)
    print(f"\n{'='*70}\nB. RISK PARITY: {dict(zip(df.columns, w.round(3)))}")
    results['B_RiskParity'] = adversarial_validation(
        pd.Series((df.values * w).sum(axis=1), index=df.index), "Risk Parity", spy_ret)

    # C. WF Min Variance
    lb = 12
    wf_r = []
    wf_d = []
    for i in range(lb, len(df)):
        w = min_var_weights(df.iloc[i-lb:i].cov().values)
        wf_r.append((df.iloc[i].values * w).sum())
        wf_d.append(df.index[i])
    print(f"\n{'='*70}\nC. WF MIN VARIANCE")
    results['C_MinVar_WF'] = adversarial_validation(pd.Series(wf_r, index=wf_d), "WF Min Var", spy_ret)

    # D. WF Max Sharpe
    ms_r, ms_d = [], []
    for i in range(lb, len(df)):
        w = max_sharpe_weights(df.iloc[i-lb:i])
        ms_r.append((df.iloc[i].values * w).sum())
        ms_d.append(df.index[i])
    print(f"\n{'='*70}\nD. WF MAX SHARPE")
    results['D_MaxSharpe_WF'] = adversarial_validation(pd.Series(ms_r, index=ms_d), "WF Max Sharpe", spy_ret)

    # E. Growth-Tilted
    w = np.array([0.30, 0.30, 0.15, 0.10, 0.15])
    print(f"\n{'='*70}\nE. GROWTH-TILTED 60/40")
    results['E_GrowthTilted'] = adversarial_validation(
        pd.Series((df.values * w).sum(axis=1), index=df.index), "Growth-Tilted", spy_ret)

    # F. Regime-Adaptive
    ra = []
    for i in range(len(df)):
        dt = df.index[i]
        is_bear = bool(bear_flags.loc[dt]) if dt in bear_flags.index else False
        w = np.array([0.10, 0.15, 0.25, 0.25, 0.25]) if is_bear else np.array([0.35, 0.30, 0.10, 0.05, 0.20])
        ra.append((df.iloc[i].values * w).sum())
    print(f"\n{'='*70}\nF. REGIME-ADAPTIVE")
    results['F_RegimeAdaptive'] = adversarial_validation(pd.Series(ra, index=df.index), "Regime-Adaptive", spy_ret)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<22s} {'Sharpe':>7s} {'Sort':>6s} {'CAGR':>7s} {'MaxDD':>7s} {'Cal':>6s} {'PF':>5s} {'WR':>5s} {'G':>3s}")
    print("-" * 75)
    for name in sorted(results):
        r = results[name]
        print(f"{name:<22s} {r['sharpe']:7.2f} {r['sortino']:6.2f} {r['cagr']:7.1%} "
              f"{r['max_dd']:7.1%} {r['calmar']:6.2f} {r['pf']:5.2f} {r['wr']:5.1%} {r['gates']:>2d}/4")

    best = max(results.items(), key=lambda x: x[1]['sharpe'])
    best_v = max(results.items(), key=lambda x: (x[1]['gates'], x[1]['sharpe']))
    print(f"\n🏆 HIGHEST SHARPE: {best[0]} — {best[1]['sharpe']:.2f}, CAGR {best[1]['cagr']:.1%}, Gates {best[1]['gates']}/4")
    if best_v[0] != best[0]:
        print(f"🏅 BEST VALIDATED: {best_v[0]} — {best_v[1]['sharpe']:.2f}, Gates {best_v[1]['gates']}/4")

    # Save
    output = {
        'strategy': 'Multi-Strategy Portfolio v3',
        'run_date': datetime.now().isoformat(),
        'n_months': len(df),
        'date_range': f"{df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}",
        'correlations': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in df.corr().to_dict().items()},
        'variants': results,
        'winner': best[0], 'best_validated': best_v[0]
    }
    save_path = '/home/jupiter/Lvl3Quant/research/findings/multi_strategy_portfolio_v3_results.json'
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o)
    print(f"\nSaved → {save_path}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('multi_strategy_portfolio_v3')
            with mlflow.start_run(run_name=f'pv3_{datetime.now().strftime("%Y%m%d_%H%M")}'):
                for name, r in results.items():
                    for k in ['sharpe', 'cagr', 'max_dd', 'gates']:
                        mlflow.log_metric(f'{name}_{k}', r[k])
                mlflow.log_artifact(save_path)
            print("MLflow ✅")
        except Exception as e:
            print(f"MLflow: {e}")

    return results


if __name__ == '__main__':
    main()
