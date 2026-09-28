#!/usr/bin/env python3
"""Multi-Asset Momentum Options v1 — Extend sector rotation to broader asset classes.

Our sector rotation (11 ETFs) gives Sharpe 4.20. Can we improve by adding:
- Commodities: GLD, SLV, USO, DBA, UNG
- Bonds: TLT, HYG, LQD, TIP
- International: EFA, EEM, VWO
- Crypto: BITO
- Alternatives: VNQ (REITs beyond XLRE), AMLP (MLPs)

Variants:
  A: Sector-only baseline (11 ETFs, replicate integrated v2)
  B: Broad universe (all 25+ ETFs)
  C: Sectors + Commodities only
  D: Sectors + International only
  E: Sectors + Bonds (decorrelation play)
  F: Best multi-asset with quality-momentum features
  G: Concentrated top-3 from broad universe

Uses quality-momentum features (winning config from integrated v2).
$645 starting capital, ATR pricing + 15% haircut, bi-weekly rebalance.
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'multi_asset_momentum_options_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

# Asset universes
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
COMMODITIES = ['GLD','SLV','USO','DBA']  # Removed UNG (too volatile/decay)
BONDS = ['TLT','HYG','LQD','TIP']
INTERNATIONAL = ['EFA','EEM','VWO']
ALTERNATIVES = ['VNQ','AMLP']
CRYPTO = ['BITO']

ALL_ASSETS = SECTORS + COMMODITIES + BONDS + INTERNATIONAL + ALTERNATIVES + CRYPTO
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

# Quality-momentum features (winning config)
QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']

def download_data(universe):
    import yfinance as yf
    fprint(f"Downloading {len(universe)} assets...")
    tickers = list(set(universe + ['SPY', '^VIX']))
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    available = [c for c in universe if c in close.columns and close[c].dropna().shape[0] > 500]
    fprint(f"  Available: {len(available)}/{len(universe)} assets (>500 days data)")
    if not available: return None, None, None, None, None, None, []

    sc = close[available].dropna(how='all')
    sh = high[[c for c in available if c in high.columns]].dropna(how='all')
    sl = low[[c for c in available if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in available if c in volume.columns]].dropna(how='all')

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    if len(sh) > 0: ix = ix.intersection(sh.index)
    if len(sl) > 0: ix = ix.intersection(sl.index)
    fprint(f"  Data: {len(ix)} days, {len(available)} assets")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix], available

def compute_features(px, vol_data=None):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    # Quality
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax(); mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    up = rets[rets > 0]; dn = rets[rets < 0]
    f['up_capture'] = float(up.iloc[-63:].mean()/(up.mean()+1e-10)) if len(up) > 10 else 1.0
    f['dn_capture'] = float(dn.iloc[-63:].mean()/(dn.mean()+1e-10)) if len(dn) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values+1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean()/(vol_data.iloc[-63:].mean()+1e-10)) if vol_data is not None and len(vol_data) >= 63 else 1.0
    return f

def build_rankings(sc, sv, rebal_dates):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats: continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns: df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[QM_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))
        try:
            try:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, device='gpu', verbose=-1)
                m.fit(Xt, yt)
            except:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
                m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

def simulate(name, rankings, sc, sh, sl, spy, vix, top_k=3, vix_min=20):
    sma200 = spy.rolling(200).mean()
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        if vix_min is not None and cv < vix_min: continue

        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]
        max_pos = min(200, equity/3)
        if max_pos < 30: eq_curve.append(equity); continue

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt); ei = min(di+30, len(sc)-1); Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S*1.03)
            lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
            sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
            val = lp-sp; width = K2-K1; cost = val*100+SPREAD_COMM
            mx_prof = (width-val)*100-SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

            pnl, aei = None, ei
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci]); dh = ci-di; rd = max(0,30-dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/30.0)
                si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                cp = si - cost
                if cp >= mx_prof*0.50 or rd < 7: pnl = cp; aei = ci; break
            if pnl is None:
                pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - val*100 - SPREAD_COMM
            equity += pnl; n_ent += 1
            trades.append({'entry': str(dt.date()), 'ticker': tk, 'pnl': round(pnl,2),
                           'win': pnl>0, 'regime': 'bull' if bull else 'bear'})
        eq_curve.append(equity)
    return trades, equity, eq_curve

def metrics_validate(trades, final_eq, eq_curve, name):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    tdf = pd.DataFrame(trades); tdf['m'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    mr = tdf.groupby('m')['pnl'].sum()/CAP; ny = max(len(mr)/12, 0.5)
    sh = (mr.mean()*12)/(mr.std()*np.sqrt(12)+1e-10) if len(mr) > 3 else 0
    dn = mr[mr<0]; so = (mr.mean()*12)/(dn.std()*np.sqrt(12)+1e-10) if len(dn) > 1 else 0
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0)); pf = gp/(gl+1e-10)
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    rets = np.array(mr.values.tolist()); gates = 0
    pp, rg = 1.0, 1.0
    if len(rets) >= 10:
        rs = np.mean(rets)/(np.std(rets)+1e-10)
        pp = sum(1 for _ in range(2000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/2000
        g1 = pp < 0.05; gates += g1
        rg = abs(bw-brw)/max(bw,brw,1); g2 = rg < 0.50; gates += g2
        mid = len(rets)//2
        h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
        h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
        g3 = h1 > 0 and h2 > 0; gates += g3
        tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0 if len(rets) > 5 else False; gates += g4
    else:
        g1, g2, g3, g4, h1, h2 = False, False, False, False, 0, 0

    # Ticker breakdown
    tdf_tk = tdf.groupby('ticker').agg({'pnl': ['mean','sum','count'], 'win': 'mean'})
    top_tickers = tdf_tk.sort_values(('pnl','sum'), ascending=False).head(5)

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1), 'sharpe': round(sh,2),
         'sortino': round(so,2), 'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'avg_pnl': round(np.mean(pnls),2), 'final_equity': round(final_eq,2),
         'n_assets': len(tdf['ticker'].unique()),
         'bull_wr': round(bw,1), 'bear_wr': round(brw,1),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3)}
    fprint(f"  {name}: {n} trades ({r['n_assets']} assets) | WR {wr:.1f}% | Sh {sh:.2f} | "
           f"CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | PF {pf:.2f} | ${CAP:.0f}->${final_eq:.0f} | Gates {gates}/4")
    fprint(f"    Perm p={pp:.4f} | R1 gap={rg:.3f} | Sub: {h1:.2f}/{h2:.2f}")
    return r

def main():
    t0 = datetime.now()
    fprint(f"Multi-Asset Momentum Options v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Extending sector rotation to commodities, bonds, international, alts")
    fprint(f"{'='*80}")

    # Download for each universe
    universes = {
        'A_Sectors': SECTORS,
        'B_Broad': ALL_ASSETS,
        'C_Sec+Commod': SECTORS + COMMODITIES,
        'D_Sec+Intl': SECTORS + INTERNATIONAL,
        'E_Sec+Bonds': SECTORS + BONDS,
        'F_Broad_QM': ALL_ASSETS,  # Same universe, different top_k
        'G_Broad_Top3': ALL_ASSETS,
    }

    # Download all data once
    fprint("\nDownloading all assets...")
    import yfinance as yf
    all_tickers = list(set(ALL_ASSETS + ['SPY', '^VIX']))
    raw = yf.download(all_tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close_all = raw['Close'] if mi else raw
    high_all = raw['High'] if mi else raw
    low_all = raw['Low'] if mi else raw
    vol_all = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close_all.columns else 'VIX'
    vix = close_all[vc].dropna()
    spy = close_all['SPY'].dropna()

    results = []
    configs = [
        ('A_Sectors_Baseline', SECTORS, 3, 20),
        ('B_Broad_Universe', ALL_ASSETS, 3, 20),
        ('C_Sectors+Commodities', SECTORS+COMMODITIES, 3, 20),
        ('D_Sectors+International', SECTORS+INTERNATIONAL, 3, 20),
        ('E_Sectors+Bonds', SECTORS+BONDS, 3, 20),
        ('F_Broad_AllVIX', ALL_ASSETS, 3, None),  # No VIX filter
        ('G_Broad_Concentrated', ALL_ASSETS, 2, 20),  # Top-2 only
    ]

    for var_name, universe, top_k, vix_min in configs:
        fprint(f"\n{'='*60}")
        fprint(f"VARIANT: {var_name} ({len(universe)} assets, top-{top_k})")
        fprint(f"{'='*60}")

        available = [c for c in universe if c in close_all.columns and close_all[c].dropna().shape[0] > 500]
        if len(available) < 3:
            fprint(f"  SKIPPED: only {len(available)} assets available")
            continue

        sc = close_all[available].dropna(how='all')
        sh = high_all[[c for c in available if c in high_all.columns]].dropna(how='all')
        sl = low_all[[c for c in available if c in low_all.columns]].dropna(how='all')
        sv = vol_all[[c for c in available if c in vol_all.columns]].dropna(how='all')

        ix = sc.index.intersection(vix.index).intersection(spy.index)
        if len(sh) > 0: ix = ix.intersection(sh.index)
        if len(sl) > 0: ix = ix.intersection(sl.index)
        sc = sc.loc[ix]; sh = sh.loc[ix]; sl = sl.loc[ix]; sv = sv.reindex(ix)

        fprint(f"  Data: {len(ix)} days, {len(available)} assets")
        bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
        rankings = build_rankings(sc, sv, bd)
        if not rankings:
            fprint(f"  SKIPPED: no rankings")
            continue

        tr, eq, cu = simulate(var_name, rankings, sc, sh, sl, spy, vix, top_k=top_k, vix_min=vix_min)
        r = metrics_validate(tr, eq, cu, var_name)
        if r: results.append(r)

    if not results: fprint("No results"); return

    # Summary
    fprint(f"\n{'='*105}")
    fprint(f"SUMMARY — Multi-Asset Momentum Options v1")
    fprint(f"{'='*105}")
    fprint(f"{'Variant':<28} {'#':>5} {'Assets':>6} {'WR':>6} {'Sh':>6} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>9} {'G':>4}")
    fprint("-"*105)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<28} {r['n_trades']:>5} {r['n_assets']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} "
               f"${r['final_equity']:>8.0f} {r['gates']:>3}/4")

    bs = max(results, key=lambda x: x['sharpe'])
    bl = next((r for r in results if 'Baseline' in r['name']), results[0])
    fprint(f"\nBASELINE (sectors only): Sharpe {bl['sharpe']}, CAGR {bl['cagr_pct']}%")
    fprint(f"BEST: {bs['name']} — Sharpe {bs['sharpe']}, CAGR {bs['cagr_pct']}%")
    delta = bs['sharpe'] - bl['sharpe']
    if delta > 0.10:
        fprint(f"VERDICT: Broader universe IMPROVES Sharpe by {delta:.2f}. Worth expanding.")
    elif delta > -0.10:
        fprint(f"VERDICT: Broader universe ~NEUTRAL (Sharpe delta {delta:+.2f}). Sectors alone may be sufficient.")
    else:
        fprint(f"VERDICT: Broader universe HURTS (Sharpe delta {delta:+.2f}). Stick with sectors.")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'results': results, 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'multi_asset_momentum_options_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"ma_mom_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'n_variants': len(results), 'capital': CAP, 'pricing': 'ATR'})
                for r in results:
                    p = r['name'][:18].replace(' ','_').replace('+','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_cagr': r['cagr_pct'],
                                        f'{p}_mdd': r['maxdd_pct'], f'{p}_wr': r['win_rate'],
                                        f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Multi-Asset Momentum Options v1\n{'='*80}")

if __name__ == '__main__':
    main()
