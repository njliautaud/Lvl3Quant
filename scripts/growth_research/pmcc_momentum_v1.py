#!/usr/bin/env python3
"""
PMCC Momentum v1 — Poor Man's Covered Call + LightGBM Sector Ranking
Buy 70-delta LEAPS (stock substitute) on top-ranked sector ETFs, sell
short-term OTM calls for income. $645 starting capital.
Variants: A-F (15d/30d delta, T1/T3 sectors, adaptive, biweekly).
Walk-forward LightGBM (252d train, 21d test). BS pricing, 15% haircut.
"""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
warnings.filterwarnings('ignore')

def fprint(*a, **kw): print(*a, **kw, flush=True)

BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / 'output' / 'growth_research' / 'pmcc_momentum_v1'
OUTPUT.mkdir(parents=True, exist_ok=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: pass

ETFS = ['XLK','XLF','XLV','XLE','XLI','XLY','XLP','XLU','XLB','XLRE','XLC']
CAP, MAX_POS, MAX_CONC = 645, 300, 2
COMM = 0.65 * 2 * 2  # $2.60 per PMCC round-trip
LEAPS_DTE, ROLL_DTE, SHORT_DTE, HAIRCUT = 180, 90, 30, 0.15

def bs_price(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0: return max(0, S - K)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_delta(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0: return 1.0 if S > K else 0.0
    return norm.cdf((np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))

def strike_for_d(S, T, sigma, td, r=0.04):
    lo, hi = S*0.5, S*2.0
    for _ in range(60):
        mid = (lo+hi)/2
        d = bs_delta(S, mid, T, sigma, r)
        if d > td: lo = mid
        else: hi = mid
        if abs(d - td) < 1e-6: break
    return (lo+hi)/2

def fetch_data():
    cache = OUTPUT / 'price_cache.parquet'
    if cache.exists(): return pd.read_parquet(cache)
    import yfinance as yf
    raw = yf.download(ETFS + ['SPY','^VIX'], start='2008-01-01', end='2026-07-25',
                      auto_adjust=True, threads=True, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    try: close.columns = close.columns.droplevel(1)
    except: pass
    close = close.dropna(how='all'); close.to_parquet(cache)
    fprint(f"  Data: {close.shape}, {close.index[0].date()} to {close.index[-1].date()}")
    return close

def build_feat(px, end):
    px = px.loc[:end].dropna()
    if len(px) < 252: return None
    r = [px.iloc[-1]/px.iloc[-d]-1 for d in [21,63,126,252]]
    r.append(r[3]-r[0])  # mom_12_1
    pct = px.pct_change()
    r += [pct.iloc[-20:].std()*np.sqrt(252), pct.iloc[-60:].std()*np.sqrt(252)]
    pk = px.iloc[-252:].expanding().max(); r.append(((px.iloc[-252:]-pk)/pk).min())
    rets60 = pct.iloc[-60:].dropna()
    r += [rets60.kurtosis() if len(rets60)>10 else 0, rets60.skew() if len(rets60)>10 else 0]
    d14 = px.diff().iloc[-14:]
    g, l = d14.clip(lower=0).mean(), -d14.clip(upper=0).mean()
    r.append(100-100/(1+g/(l+1e-10)) if l>0 else 50)
    r.append(rets60.mean()/(rets60.std()+1e-10)*np.sqrt(252) if len(rets60)>20 else 0)
    return r if not any(np.isnan(v) for v in r) else None

def precompute(close_df, etfs):
    fprint("  Precomputing features...")
    mes = close_df.resample('ME').last().dropna(how='all').index
    cache = {}
    for i, me in enumerate(mes):
        feats = {e: build_feat(close_df[e], me) for e in etfs if e in close_df.columns}
        feats = {k: v for k, v in feats.items() if v is not None}
        idx = i
        if idx+1 < len(mes):
            nme = mes[idx+1]
            feats['_fwd'] = {e: close_df[e].loc[:nme].dropna().iloc[-1]/close_df[e].loc[:me].dropna().iloc[-1]-1
                             for e in etfs if e in close_df.columns and len(close_df[e].loc[:me].dropna())>0
                             and len(close_df[e].loc[:nme].dropna())>0}
        cache[me] = feats
    fprint(f"  {len(cache)} month-ends cached"); return cache

def lgbm_rank(fc, dt, tm=12):
    try: import lightgbm as lgb
    except ImportError: return _simple_rank(fc, dt)
    ms = sorted([k for k in fc if k <= dt and isinstance(k, pd.Timestamp)])
    if len(ms) < 24: return _simple_rank(fc, dt)
    X, y = [], []
    for me in ms[max(0,len(ms)-tm):-1]:
        fwd = fc[me].get('_fwd',{})
        for e in fc[me]:
            if e.startswith('_') or e not in fwd: continue
            X.append(fc[me][e]); y.append(fwd[e])
    if len(X) < 30: return _simple_rank(fc, dt)
    m = lgb.LGBMRegressor(n_estimators=100,max_depth=4,learning_rate=0.1,
        subsample=0.8,colsample_bytree=0.8,min_child_samples=5,verbose=-1,n_jobs=4)
    m.fit(np.array(X), np.array(y))
    cur = ms[-1]; sc = {e: m.predict(np.array([fc[cur][e]]))[0] for e in fc[cur] if not e.startswith('_')}
    return sorted(sc, key=sc.get, reverse=True)

def _simple_rank(fc, dt):
    ms = sorted([k for k in fc if k <= dt and isinstance(k, pd.Timestamp)])
    if not ms: return ETFS[:3]
    sc = {e: fc[ms[-1]][e][3]-fc[ms[-1]][e][0] for e in fc[ms[-1]] if not e.startswith('_')}
    return sorted(sc, key=sc.get, reverse=True)

def simulate(cdf, vix, fc, top_n=1, sd=0.15, adaptive=False, biweek=False, cap=CAP, name=''):
    fprint(f"\n  {name}: top_n={top_n} sd={sd} adaptive={adaptive}")
    dates = cdf.index[cdf.index >= cdf.index[252]]
    rdates = dates.to_series().resample('2W' if biweek else 'ME').last().dropna().values
    eq, pos, rets, inc, grw, nt = float(cap), [], [], 0.0, 0.0, 0
    prev, ri, ranks = eq, 0, []
    for date in dates:
        date = pd.Timestamp(date)
        v = float(vix.loc[date]) if date in vix.index else 18.0
        sig = max(v/100, 0.05)
        if ri < len(rdates) and date >= pd.Timestamp(rdates[ri]):
            ranks = lgbm_rank(fc, date); ri += 1
        for p in pos: p['ld'] -= 1; p['sdt'] -= 1
        # Handle expired shorts
        for p in list(pos):
            if p['sdt'] > 0: continue
            px = float(cdf[p['t']].loc[date]) if date in cdf.index else p['ep']
            if px > p['sk']:  # assigned
                lv = bs_price(px, p['lk'], max(p['ld'],1)/252, sig)*(1-HAIRCUT)
                pnl = p['sp'] - (px-p['sk']) + lv - p['lc'] - COMM
                eq += pnl; grw += lv-p['lc']; inc += p['sp']-(px-p['sk']); nt += 1
                pos.remove(p)
            else:  # expired OTM
                inc += p['sp']; eq += p['sp']; nt += 1
                d = 0.30 if adaptive and v<20 else (0.15 if adaptive else sd)
                nk = strike_for_d(px, SHORT_DTE/252, sig, d)
                p['sk'] = nk; p['sdt'] = SHORT_DTE
                p['sp'] = bs_price(px, nk, SHORT_DTE/252, sig)*(1-HAIRCUT)
                eq -= 0.65*2  # commission for new short
        # Roll LEAPS
        for p in list(pos):
            if p['ld'] > ROLL_DTE or p['ld'] <= 0: continue
            px = float(cdf[p['t']].loc[date])
            ov = bs_price(px, p['lk'], p['ld']/252, sig)*(1-HAIRCUT)
            nk = strike_for_d(px, LEAPS_DTE/252, sig, 0.70)
            nc = bs_price(px, nk, LEAPS_DTE/252, sig)*(1+HAIRCUT)
            rc = nc - ov + 0.65*2
            if eq > rc+50:
                eq -= rc; grw += ov-p['lc']
                p['lk'],p['ld'],p['lc'],p['ep'] = nk, LEAPS_DTE, nc, px
        # Open new positions on rebalance
        if ri > 0 and date == pd.Timestamp(rdates[min(ri-1, len(rdates)-1)]):
            held = {p['t'] for p in pos}
            for e in [t for t in ranks if t in ETFS][:top_n]:
                if e in held or len(pos) >= MAX_CONC or e not in cdf.columns: continue
                px = float(cdf[e].loc[date])
                lk = strike_for_d(px, LEAPS_DTE/252, sig, 0.70)
                lc = bs_price(px, lk, LEAPS_DTE/252, sig)*(1+HAIRCUT)
                if lc > MAX_POS or lc > eq-50: continue
                d = 0.30 if adaptive and v<20 else (0.15 if adaptive else sd)
                sk = strike_for_d(px, SHORT_DTE/252, sig, d)
                sp = bs_price(px, sk, SHORT_DTE/252, sig)*(1-HAIRCUT)
                eq -= lc + COMM; nt += 1
                pos.append({'t':e,'lk':lk,'ld':LEAPS_DTE,'lc':lc,'sk':sk,'sdt':SHORT_DTE,'sp':sp,'ep':px,'ed':date})
        # MTM
        pv = sum(bs_price(float(cdf[p['t']].loc[date]), p['lk'], max(p['ld'],1)/252, sig)*(1-HAIRCUT)
                 - bs_price(float(cdf[p['t']].loc[date]), p['sk'], max(p['sdt'],1)/252, sig)*(1-HAIRCUT)
                 for p in pos if p['t'] in cdf.columns and date in cdf.index)
        te = eq + pv
        rets.append(te/prev-1 if prev > 0 else 0); prev = te
    # Metrics
    r = np.array(rets); r = r[~np.isnan(r)]
    yrs = len(r)/252; feq = cap * np.prod(1+r) if len(r) else cap
    sh = r.mean()/(r.std()+1e-10)*np.sqrt(252)
    dn = r[r<0]; so = r.mean()/(dn.std()+1e-10)*np.sqrt(252) if len(dn) else 0
    cum = np.cumprod(1+r); mdd = ((cum/np.maximum.accumulate(cum))-1).min() if len(cum) else 0
    cagr = (feq/cap)**(1/yrs)-1 if yrs > 0 else 0
    mr = pd.Series(r).groupby(np.arange(len(r))//21).sum()
    wr = (mr>0).mean(); pf = mr[mr>0].sum()/(-mr[mr<0].sum()+1e-10)
    m = dict(name=name, sharpe=round(sh,3), sortino=round(so,3), cagr=round(cagr*100,2),
             maxdd=round(mdd*100,2), wr=round(wr*100,1), pf=round(pf,2), final_equity=round(feq,2),
             n_trades=nt, income_total=round(inc,2), growth_total=round(grw,2),
             avg_monthly_income=round(inc/max(yrs*12,1),2), total_years=round(yrs,1))
    fprint(f"    Sharpe={sh:.3f} Sortino={so:.3f} CAGR={cagr*100:.1f}% MaxDD={mdd*100:.1f}% "
           f"WR={wr*100:.0f}% PF={pf:.2f} Final=${feq:.0f} Inc=${inc:.0f}")
    return m, r

def perm_test(cdf, vix, fc, real_sh, n=50):
    fprint(f"\n  Permutation test ({n} shuffles)...")
    ps = []
    for i in range(n):
        sc = {}
        for dt, feats in fc.items():
            es = [k for k in feats if not k.startswith('_')]
            if not es: sc[dt] = feats; continue
            s = dict(feats); np.random.seed(i*1000+hash(str(dt))%10000)
            pk = list(np.random.permutation(es)); vs = [feats[k] for k in es]
            for j,k in enumerate(pk): s[k] = vs[j]
            sc[dt] = s
        _, r = simulate(cdf, vix, sc, name=f'perm_{i}')
        ps.append(r.mean()/(r.std()+1e-10)*np.sqrt(252))
    ps = np.array(ps); pv = (ps >= real_sh).mean()
    fprint(f"  p-value={pv:.4f} (real={real_sh:.3f}, perm_mean={ps.mean():.3f})"); return pv

def regime_check(dr, sr):
    n = min(len(dr), len(sr)); dr, sr = dr[:n], sr[:n]
    g, r = dr[sr>=0], dr[sr<0]
    sg = g.mean()/(g.std()+1e-10)*np.sqrt(252) if len(g)>10 else 0
    sr2 = r.mean()/(r.std()+1e-10)*np.sqrt(252) if len(r)>10 else 0
    gap = abs(sg-sr2)/(max(abs(sg),abs(sr2))+1e-10); ok = gap < 0.50
    fprint(f"  R1: Sharpe_green={sg:.3f} Sharpe_red={sr2:.3f} gap={gap:.3f} {'PASS' if ok else 'FAIL'}")
    return dict(sharpe_green=round(sg,3), sharpe_red=round(sr2,3), gap=round(gap,3), passed=ok)

def main():
    fprint("="*70); fprint("PMCC Momentum v1"); fprint("="*70); t0 = datetime.now()
    cdf = fetch_data()
    vc = '^VIX' if '^VIX' in cdf.columns else 'VIX'
    vix = cdf[vc].dropna() if vc in cdf.columns else pd.Series(18, index=cdf.index)
    spy = cdf['SPY'].dropna()
    ec = [c for c in ETFS if c in cdf.columns]; fprint(f"  {len(ec)} sector ETFs")
    fc = precompute(cdf, ec)
    variants = {'A_15d_T1':(1,.15,False,False), 'B_30d_T1':(1,.30,False,False),
                'C_15d_T3':(3,.15,False,False), 'D_30d_T3':(3,.30,False,False),
                'E_Adaptive':(1,.15,True,False), 'F_BiWeekly':(1,.15,False,True)}
    res, bsh, bn, br = {}, -999, '', None
    for vn,(tn,sd,ad,bw) in variants.items():
        m, r = simulate(cdf, vix, fc, top_n=tn, sd=sd, adaptive=ad, biweek=bw, name=vn)
        res[vn] = m
        if m['sharpe'] > bsh: bsh, bn, br = m['sharpe'], vn, r
    fprint(f"\n  Best: {bn} (Sharpe={bsh:.3f})")
    spy_r = spy.pct_change().dropna().values[-len(br):]
    r1 = regime_check(br, spy_r); res[bn]['regime_check'] = r1
    # Sub-period
    ch = len(br)//3
    sps = [round((br[i*ch:(i+1)*ch].mean()/(br[i*ch:(i+1)*ch].std()+1e-10)*np.sqrt(252)),3) for i in range(3)]
    fprint(f"  Sub-periods: {sps}"); res[bn]['subperiod_sharpes'] = sps
    # Outlier removal
    lo,hi = np.percentile(br,1), np.percentile(br,99)
    cl = np.clip(br,lo,hi); so,sc2 = br.mean()/(br.std()+1e-10)*np.sqrt(252), cl.mean()/(cl.std()+1e-10)*np.sqrt(252)
    fprint(f"  Outlier: orig={so:.3f} clipped={sc2:.3f}"); res[bn]['outlier_check'] = dict(original=round(so,3), clipped=round(sc2,3))
    pv = perm_test(cdf, vix, fc, bsh); res[bn]['perm_p_value'] = round(pv, 4)
    # Summary
    fprint(f"\n{'='*70}\n{'Variant':<16} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>6} {'MaxDD%':>7} {'WR%':>5} {'PF':>5} {'Final$':>8} {'Inc$':>7}")
    fprint("-"*78)
    for v,m in sorted(res.items()):
        fprint(f"{v:<16} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>6.1f} {m['maxdd']:>7.1f} "
               f"{m['wr']:>5.1f} {m['pf']:>5.2f} {m['final_equity']:>8.0f} {m['income_total']:>7.0f}")
    rp = OUTPUT / 'pmcc_momentum_v1_results.json'
    with open(rp, 'w') as f: json.dump(res, f, indent=2, default=str)
    fprint(f"\n  Saved {rp}")
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('pmcc_momentum_v1')
            with mlflow.start_run(run_name=f'pmcc_v1_{datetime.now():%Y%m%d_%H%M}'):
                b = res[bn]
                for k in ['sharpe','sortino','cagr','maxdd','wr','pf','final_equity','income_total']:
                    mlflow.log_metric(k, b[k])
                mlflow.log_metric('regime_gap', r1['gap']); mlflow.log_metric('perm_p_value', pv)
                mlflow.log_param('best_variant', bn); mlflow.log_param('capital', CAP)
                try: mlflow.log_artifact(str(rp))
                except: pass
            fprint("  MLflow logged")
        except Exception as e: fprint(f"  MLflow error: {e}")
    fprint(f"\n  Done in {(datetime.now()-t0).total_seconds():.0f}s")

if __name__ == '__main__':
    main()
