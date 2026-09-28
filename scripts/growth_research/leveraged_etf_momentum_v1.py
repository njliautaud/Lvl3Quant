#!/usr/bin/env python3
"""
Leveraged ETF Momentum Rotation v1 — LightGBM timing of 3x ETFs
Uses actual leveraged ETF prices (decay embedded). WF: 252d train, bi-weekly rebalance.
$645 start. 7 variants: Lev_Always_T3, Lev_Bull_Only, Lev_VIX_Scale,
Lev_Momentum, Lev_Inverse_Bear, Regular_Baseline, Lev_VolTarget.
"""
import sys, json, time, warnings
import numpy as np, pandas as pd, yfinance as yf, lightgbm as lgb
from pathlib import Path
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / 'output' / 'growth_research' / 'leveraged_etf_momentum_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except: pass

REGULAR = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','SPY','QQQ']
LEVERAGED = {'QQQ':'TQQQ','SPY':'UPRO','XLK':'TECL','XLF':'FAS','XLE':'ERX'}
INVERSE = {'QQQ':'SQQQ','SPY':'SPXU'}
ALL_TICKERS = list(set(REGULAR + list(LEVERAGED.values()) + list(INVERSE.values()) + ['SOXL','SOXS','^VIX']))
START, COST_BPS, REBAL, TOP_K = 645.0, 5, 10, 3

def download_data():
    fprint("Downloading data...")
    data = yf.download(ALL_TICKERS, start='2009-01-01', end='2026-07-25', auto_adjust=True, progress=False)
    prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    if '^VIX' in prices.columns: prices = prices.rename(columns={'^VIX':'VIX'})
    prices = prices.ffill().dropna(how='all')
    fprint(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    return prices

def build_features(close, spy, vix):
    lr = np.log(close / close.shift(1)); slr = np.log(spy / spy.shift(1))
    f = pd.DataFrame({
        'r5': close.pct_change(5), 'r10': close.pct_change(10), 'r21': close.pct_change(21),
        'r63': close.pct_change(63), 'r126': close.pct_change(126), 'r252': close.pct_change(252),
        'mom12_1': close.pct_change(252) - close.pct_change(21),
        'v20': lr.rolling(20).std()*np.sqrt(252), 'v60': lr.rolling(60).std()*np.sqrt(252),
        'vr': lr.rolling(20).std()/(lr.rolling(60).std()+1e-10),
        'sr63': lr.rolling(63).mean()/(lr.rolling(63).std()+1e-10),
        'mdd63': (close/close.rolling(63).max()-1).rolling(63).min(),
        'sk63': lr.rolling(63).skew(),
        'a50': (close > close.rolling(50).mean()).astype(int),
        'a200': (close > close.rolling(200).mean()).astype(int),
        'd200': (close - close.rolling(200).mean())/(close.rolling(200).mean()+1e-10),
        'spy21': spy.pct_change(21), 'spya200': (spy > spy.rolling(200).mean()).astype(int),
        'corr': lr.rolling(63).corr(slr),
    }, index=close.index)
    if vix is not None:
        va = vix.reindex(close.index).ffill()
        f['vix'] = va; f['vix20'] = va.rolling(20).mean(); f['vixp'] = va.rolling(252).rank(pct=True)
    return f

def lgbm_walkforward(prices):
    spy = prices['SPY']; vix = prices.get('VIX')
    fa, la, ma = [], [], []
    ci = prices[REGULAR].dropna(how='all').index
    for etf in REGULAR:
        cl = prices[etf].reindex(ci)
        if cl.isna().sum() > len(cl)*0.5: continue
        feat = build_features(cl, spy.reindex(ci), vix.reindex(ci) if vix is not None else None)
        feat['fwd'] = cl.pct_change(REBAL).shift(-REBAL); feat['etf'] = etf; feat['date'] = feat.index
        v = feat.dropna(subset=['fwd'])
        fc = [c for c in feat.columns if c not in ['fwd','etf','date']]
        fa.append(v[fc]); la.append(v['fwd']); ma.append(v[['etf','date']])
    X = pd.concat(fa).astype(float); y = pd.concat(la); meta = pd.concat(ma)
    fc = list(X.columns); dates = sorted(meta['date'].unique()); rdates = dates[252::REBAL]
    rankings = {}; fprint(f"  WF: {len(rdates)} rebalance dates")
    par = {'objective':'regression','metric':'rmse','num_leaves':31,'learning_rate':0.05,
           'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,'n_jobs':-1}
    for rd in rdates:
        ri = dates.index(rd); t0 = dates[max(0, ri-252)]
        tr = (meta['date'] >= t0) & (meta['date'] < rd); te = meta['date'] == rd
        Xtr, ytr, Xte, mte = X[tr], y[tr], X[te], meta[te]
        if len(Xtr) < 50 or len(Xte) < 2: continue
        Xtr = Xtr.replace([np.inf,-np.inf], np.nan); Xte = Xte.replace([np.inf,-np.inf], np.nan)
        for c in fc:
            med = Xtr[c].median(); Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)
        mdl = lgb.train(par, lgb.Dataset(Xtr, label=ytr), num_boost_round=100)
        pdf = pd.DataFrame({'etf': mte['etf'].values, 'pred': mdl.predict(Xte)})
        rankings[rd] = pdf.sort_values('pred', ascending=False)['etf'].tolist()
    return rankings

def is_bull(prices, date):
    s = prices['SPY']; loc = s.index.searchsorted(date)
    return True if loc < 200 else s.iloc[loc-1] >= s.iloc[max(0,loc-200):loc].mean()

def get_vix(prices, date):
    if 'VIX' not in prices.columns: return 20.0
    v = prices['VIX']; loc = v.index.searchsorted(date)
    return (v.iloc[loc-1] if loc > 0 and not np.isnan(v.iloc[loc-1]) else 20.0)

def rvol(prices, tk, date, w=21):
    if tk not in prices.columns: return 0.2
    cl = prices[tk]; loc = cl.index.searchsorted(date)
    if loc < w+1: return 0.2
    lr = np.log(cl.iloc[max(0,loc-w):loc]/cl.iloc[max(0,loc-w):loc].shift(1)).dropna()
    return lr.std()*np.sqrt(252) if len(lr) > 5 else 0.2

lev = lambda e: LEVERAGED.get(e, e)
inv = lambda e: INVERSE.get(e, None)

def simulate(prices, rankings, variant):
    rdates = sorted(rankings.keys()); equity = START; curve = []
    for i, rd in enumerate(rdates):
        top = rankings[rd][:TOP_K]
        end = rdates[i+1] if i+1 < len(rdates) else None
        if end is None: break
        bull = is_bull(prices, rd); vx = get_vix(prices, rd); h = []
        if variant == 'Regular_Baseline':
            h = [(e, 1.0/TOP_K) for e in top]
        elif variant == 'Lev_Always_T3':
            h = [(lev(e), 1.0/TOP_K) for e in top]
        elif variant == 'Lev_Bull_Only':
            h = [(lev(e) if bull else e, 1.0/TOP_K) for e in top]
        elif variant == 'Lev_VIX_Scale':
            if vx < 20: h = [(lev(e), 1.0/TOP_K) for e in top]
            elif vx < 30:
                for e in top: h += [(lev(e), 0.66/TOP_K), (e, 0.34/TOP_K)]
            else: h = [(e, 1.0/TOP_K) for e in top]
        elif variant == 'Lev_Momentum':
            h = [(lev(top[0]), 1.0/TOP_K)] + [(e, 1.0/TOP_K) for e in top[1:]]
        elif variant == 'Lev_Inverse_Bear':
            if bull: h = [(lev(e), 1.0/TOP_K) for e in top]
            else:
                for e in top:
                    iv = inv(e)
                    h.append((iv, 1.0/TOP_K) if iv and iv in prices.columns else (e, 1.0/TOP_K))
        elif variant == 'Lev_VolTarget':
            for e in top:
                lt = lev(e); rv = rvol(prices, lt, rd); sc = min(0.20/(rv+1e-10), 1.0)
                h.append((lt, sc/TOP_K) if sc > 0.66 else (e, 1.0/TOP_K))
        pr = 0.0
        for tk, wt in h:
            if tk not in prices.columns:
                orig = [k for k,v in LEVERAGED.items() if v == tk]
                tk = orig[0] if orig else top[0]
            cl = prices[tk]; l0 = cl.index.searchsorted(rd); l1 = cl.index.searchsorted(end)
            if l0 >= len(cl) or l1 >= len(cl) or l0 == l1: continue
            pr += wt * (cl.iloc[l1]/cl.iloc[l0] - 1)
        pr -= len(h) * COST_BPS / 10000; equity *= (1 + pr)
        curve.append({'date': rd, 'equity': equity, 'ret': pr})
    return pd.DataFrame(curve).set_index('date') if curve else pd.DataFrame()

def calc_metrics(c):
    if c.empty or len(c) < 10: return {}
    r = c['ret']; n = len(r); ppy = 252/REBAL
    ar = (c['equity'].iloc[-1]/START)**(ppy/n) - 1; av = r.std()*np.sqrt(ppy)
    sh = ar/av if av > 0 else 0; ds = r[r<0].std()*np.sqrt(ppy) if (r<0).any() else 1e-6
    so = ar/ds; pk = c['equity'].cummax(); mdd = ((c['equity']-pk)/pk).min()
    yrs = n/ppy; cagr = (c['equity'].iloc[-1]/START)**(1/yrs)-1 if yrs > 0 else 0
    cm = c['ret'].resample('ME').apply(lambda x: (1+x).prod()-1)
    wr = (cm > 0).mean() if len(cm) > 0 else 0
    gp = r[r>0].sum(); gl = abs(r[r<0].sum()); pf = gp/gl if gl > 0 else float('inf')
    return {'sharpe':round(sh,3),'sortino':round(so,3),'cagr':round(cagr*100,2),
            'maxdd':round(mdd*100,2),'wr_monthly':round(wr*100,1),'pf':round(pf,3),
            'final_equity':round(c['equity'].iloc[-1],2),'n_periods':n,'ann_vol':round(av*100,2)}

def adversarial(prices, rankings, variant, curve, metrics):
    res = {}
    # Permutation test
    perm_sh = []
    etf_list = list(set(e for r in rankings.values() for e in r))
    for _ in range(100):
        sh = {}
        for d in rankings:
            np.random.shuffle(etf_list); sh[d] = etf_list[:len(rankings[d])]
        pm = calc_metrics(simulate(prices, sh, variant))
        perm_sh.append(pm.get('sharpe', 0))
    res['perm_p'] = round(np.mean([s >= metrics.get('sharpe',0) for s in perm_sh]), 4)
    res['perm_mean_sh'] = round(np.mean(perm_sh), 3)
    # R1 regime
    rdates = sorted(rankings.keys()); bull_r, bear_r = [], []
    for i, d in enumerate(rdates[:-1]):
        if i >= len(curve): continue
        (bull_r if is_bull(prices, d) else bear_r).append(curve['ret'].iloc[i])
    if bull_r and bear_r:
        ppy = 252/REBAL
        sb = np.mean(bull_r)/(np.std(bull_r)+1e-10)*np.sqrt(ppy)
        sr = np.mean(bear_r)/(np.std(bear_r)+1e-10)*np.sqrt(ppy)
        gap = abs(sb-sr)/max(abs(sb),abs(sr),1e-10)
        res.update({'r1_bull':round(sb,3),'r1_bear':round(sr,3),'r1_gap':round(gap,3),'r1_pass':gap<=0.50})
    else: res['r1_pass'] = False
    # Sub-period
    mid = len(curve)//2
    if mid > 10:
        m1 = calc_metrics(curve.iloc[:mid].copy()); m2 = calc_metrics(curve.iloc[mid:].copy())
        res['sub_sh1'] = m1.get('sharpe',0); res['sub_sh2'] = m2.get('sharpe',0)
        res['sub_stable'] = min(m1.get('sharpe',0), m2.get('sharpe',0)) > 0
    # Drop best month
    mo = curve['ret'].resample('ME').apply(lambda x: (1+x).prod()-1)
    if len(mo) > 3:
        bi = mo.idxmax(); dropped = curve.loc[curve.index.to_period('M') != bi.to_period('M')]
        dm = calc_metrics(dropped)
        res['nobest_sh'] = dm.get('sharpe',0); res['nobest_cagr'] = dm.get('cagr',0)
    return res

def main():
    t0 = time.time()
    fprint("="*70); fprint("Leveraged ETF Momentum Rotation v1"); fprint("="*70)
    prices = download_data()
    fprint("\nRunning LightGBM walk-forward on regular ETFs...")
    rankings = lgbm_walkforward(prices)
    fprint(f"  Got rankings for {len(rankings)} dates")
    variants = ['Regular_Baseline','Lev_Always_T3','Lev_Bull_Only',
                'Lev_VIX_Scale','Lev_Momentum','Lev_Inverse_Bear','Lev_VolTarget']
    results = {}
    if MLFLOW_OK: mlflow.set_experiment('leveraged_etf_momentum_v1')
    for v in variants:
        fprint(f"\n--- {v} ---")
        curve = simulate(prices, rankings, v); m = calc_metrics(curve)
        fprint(f"  Sharpe={m.get('sharpe')} Sortino={m.get('sortino')} CAGR={m.get('cagr')}% "
               f"MaxDD={m.get('maxdd')}% WR={m.get('wr_monthly')}% PF={m.get('pf')} ${m.get('final_equity')}")
        adv = adversarial(prices, rankings, v, curve, m)
        fprint(f"  Perm p={adv.get('perm_p')} R1 gap={adv.get('r1_gap')} R1={adv.get('r1_pass')}")
        results[v] = {**m, **adv}
        if not curve.empty: curve.to_csv(OUT_DIR / f'{v}_equity.csv')
        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=v):
                    for k2,v2 in {**m,**adv}.items():
                        if isinstance(v2,(int,float)): mlflow.log_metric(k2, v2)
                    mlflow.log_param('variant',v); mlflow.log_param('top_k',TOP_K)
                    mlflow.log_param('rebal_days',REBAL); mlflow.log_param('cost_bps',COST_BPS)
            except: pass
    fprint("\n"+"="*70); fprint("SUMMARY"); fprint("="*70)
    fprint(f"{'Variant':<22} {'Sharpe':>7} {'Sort':>7} {'CAGR%':>7} {'MDD%':>7} {'WR%':>5} {'PF':>6} {'Final$':>10} {'Pp':>6} {'R1':>3}")
    fprint("-"*90)
    for v in variants:
        r = results.get(v,{})
        fprint(f"{v:<22} {r.get('sharpe',''):>7} {r.get('sortino',''):>7} {r.get('cagr',''):>7} "
               f"{r.get('maxdd',''):>7} {r.get('wr_monthly',''):>5} {r.get('pf',''):>6} "
               f"{r.get('final_equity',''):>10} {r.get('perm_p',''):>6} {'Y' if r.get('r1_pass') else 'N':>3}")
    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nSaved to {OUT_DIR}. Time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
