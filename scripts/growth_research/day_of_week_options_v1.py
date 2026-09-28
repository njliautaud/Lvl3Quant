#!/usr/bin/env python3
"""Day-of-Week Options Timing v1 — Does entry timing matter for sector bull spreads?

Tests whether entering on specific days of the week improves risk-adjusted returns.
Also tests: monthly timing (beginning/middle/end), VIX-level gating, and combo rules.

Directly actionable for the $645 agentic account.
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
RESULTS_PATH = RESULTS_DIR / 'day_of_week_options_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15
FEAT_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

# ==================== CORE FUNCTIONS ====================
def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

def compute_features(px, idx):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0.0
    pk = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    return f

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    time_prem = atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)
    return intrinsic + time_prem

def lgbm_rankings(sc, rebal_dates):
    fprint("  Running LGBM walk-forward rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            feats = compute_features(px, idx)
            if not feats: continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[FEAT_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[FEAT_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"    Rankings for {len(rankings)} dates")
    return rankings

# ==================== SIMULATION ====================
def simulate_with_filter(name, rankings, sc, sh, sl, spy, vix,
                          day_filter=None, month_pos_filter=None,
                          vix_max=None, vix_min=None,
                          spread_pct=3.0, dte=30, top_k=3):
    """Simulate with optional entry filters."""
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    skipped = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue

        # Day-of-week filter (0=Mon, 4=Fri)
        if day_filter is not None and dt.dayofweek not in day_filter:
            skipped += 1; continue

        # Month position filter
        if month_pos_filter is not None:
            dom = dt.day
            if month_pos_filter == 'early' and dom > 10: skipped += 1; continue
            if month_pos_filter == 'mid' and (dom < 10 or dom > 20): skipped += 1; continue
            if month_pos_filter == 'late' and dom < 20: skipped += 1; continue

        cv = float(vix.loc[dt])
        # VIX filters
        if vix_max is not None and cv > vix_max: skipped += 1; continue
        if vix_min is not None and cv < vix_min: skipped += 1; continue

        sv = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv
        bull = sv >= sm
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
            di = sc.index.get_loc(dt); ei = min(di+dte, len(sc)-1); Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S*(1+spread_pct/100))
            lp = atr_premium(S, K1, dte, av, cv, 'call')*(1+HAIRCUT)
            sp = atr_premium(S, K2, dte, av, cv, 'call')*(1-HAIRCUT)
            val = lp-sp; width = K2-K1; cost = val*100+SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue
            mx_prof = (width-val)*100-SPREAD_COMM
            pnl, aei = None, ei
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci]); dh = ci-di; rd = max(0,dte-dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))
                si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                cp = si - cost
                if cp >= mx_prof*0.50 or rd < 7: pnl = cp; aei = ci; break
            if pnl is None:
                pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - val*100 - SPREAD_COMM
            equity += pnl; n_ent += 1
            trades.append({'entry': str(dt.date()), 'ticker': tk, 'pnl': round(pnl,2),
                           'win': pnl>0, 'regime': 'bull' if bull else 'bear',
                           'dow': dt.day_name(), 'vix': round(cv,1)})
        eq_curve.append(equity)
    return trades, equity, eq_curve, skipped

def metrics_and_validate(trades, final_eq, eq_curve, name):
    if not trades: return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    tdf = pd.DataFrame(trades); tdf['m'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    mr = tdf.groupby('m')['pnl'].sum()/CAP; ny = max(len(mr)/12, 0.5)
    sh = (mr.mean()*12)/(mr.std()*np.sqrt(12)+1e-10) if len(mr) > 3 else 0
    dn = mr[mr<0]; so = (mr.mean()*12)/(dn.std()*np.sqrt(12)+1e-10) if len(dn) > 1 else 0
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10); ap = np.mean(pnls)
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100
    # Validation gates
    rets = np.array(mr.values.tolist())
    gates = 0
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
        pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1), 'sharpe': round(sh,2),
         'sortino': round(so,2), 'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'avg_pnl': round(ap,2), 'final_equity': round(final_eq,2),
         'bull_wr': round(bw,1), 'bear_wr': round(brw,1), 'bull_n': len(bt), 'bear_n': len(brt),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'monthly_returns': rets.tolist()}
    return r

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Day-of-Week Options Timing v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")

    sc, sh, sl, spy, vix = download_data()
    # Use all trading days as potential entry points (daily rebalancing)
    # But filter by day-of-week in simulation
    all_days = sc.index[sc.index >= '2010-01-01']
    # Sample every 5 days to make rankings tractable
    sample_days = all_days[::5]
    rankings = lgbm_rankings(sc, sample_days)

    if not rankings:
        fprint("FATAL: No rankings"); return

    results = []
    configs = [
        # Baseline: all days
        ('A_AllDays_Baseline', None, None, None, None),
        # Individual days
        ('B_Monday_Only', [0], None, None, None),
        ('C_Tuesday_Only', [1], None, None, None),
        ('D_Wednesday_Only', [2], None, None, None),
        ('E_Thursday_Only', [3], None, None, None),
        ('F_Friday_Only', [4], None, None, None),
        # Day combos
        ('G_Mon_Wed', [0,2], None, None, None),
        ('H_Tue_Thu', [1,3], None, None, None),
        ('I_Mon_Thu', [0,3], None, None, None),
        # Month position
        ('J_Early_Month', None, 'early', None, None),
        ('K_Mid_Month', None, 'mid', None, None),
        ('L_Late_Month', None, 'late', None, None),
        # VIX filters
        ('M_LowVIX_Under20', None, None, 20, None),
        ('N_MidVIX_15to25', None, None, 25, 15),
        ('O_HighVIX_Over20', None, None, None, 20),
        # Best combos
        ('P_Mon_LowVIX', [0], None, 20, None),
        ('Q_Thu_MidVIX', [3], None, 25, 15),
    ]

    fprint(f"\nRunning {len(configs)} timing variants...")
    for nm, day_f, month_f, vix_max, vix_min in configs:
        tr, eq, cu, skip = simulate_with_filter(nm, rankings, sc, sh, sl, spy, vix,
                                                 day_filter=day_f, month_pos_filter=month_f,
                                                 vix_max=vix_max, vix_min=vix_min)
        r = metrics_and_validate(tr, eq, cu, nm)
        if r:
            r['skipped'] = skip
            results.append(r)
            fprint(f"  {nm}: {r['n_trades']} trades | WR {r['win_rate']:.1f}% | Sh {r['sharpe']:.2f} | "
                   f"CAGR {r['cagr_pct']:.1f}% | MDD {r['maxdd_pct']:.1f}% | Gates {r['gates']}/4 | "
                   f"Skipped {skip}")
        else:
            fprint(f"  {nm}: No trades (all skipped)")

    if not results: fprint("No results"); return

    # Day-of-week breakdown for baseline
    fprint(f"\n{'='*90}")
    fprint(f"DAY-OF-WEEK ANALYSIS (from baseline trades)")
    fprint(f"{'='*90}")
    baseline = next((r for r in results if r['name'] == 'A_AllDays_Baseline'), None)
    if baseline:
        # Analyze the baseline trades
        bl_trades = []
        tr, eq, cu, _ = simulate_with_filter('tmp', rankings, sc, sh, sl, spy, vix)
        if tr:
            tdf = pd.DataFrame(tr)
            fprint(f"\n{'Day':<12} {'#':>5} {'WR':>6} {'AvgPnL':>8} {'MedPnL':>8} {'TotalPnL':>10}")
            fprint("-"*55)
            for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
                dd = tdf[tdf['dow']==day]
                if len(dd) == 0: continue
                fprint(f"{day:<12} {len(dd):>5} {(dd['win'].mean()*100):>5.1f}% "
                       f"${dd['pnl'].mean():>7.2f} ${dd['pnl'].median():>7.2f} ${dd['pnl'].sum():>9.2f}")

            # VIX band analysis
            fprint(f"\n{'VIX Band':<15} {'#':>5} {'WR':>6} {'AvgPnL':>8} {'MedPnL':>8}")
            fprint("-"*45)
            for lo, hi, label in [(0,15,'VIX<15'),(15,20,'VIX 15-20'),(20,25,'VIX 20-25'),(25,35,'VIX 25-35'),(35,100,'VIX>35')]:
                vv = tdf[(tdf['vix']>=lo) & (tdf['vix']<hi)]
                if len(vv) == 0: continue
                fprint(f"{label:<15} {len(vv):>5} {(vv['win'].mean()*100):>5.1f}% "
                       f"${vv['pnl'].mean():>7.2f} ${vv['pnl'].median():>7.2f}")

    # Summary
    fprint(f"\n{'='*100}")
    fprint(f"SUMMARY — Day-of-Week Options Timing v1")
    fprint(f"{'='*100}")
    fprint(f"{'Variant':<25} {'#':>5} {'WR':>6} {'Sh':>6} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>9} {'G':>4} {'Skip':>6}")
    fprint("-"*100)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} "
               f"${r['final_equity']:>8.0f} {r['gates']:>3}/4 {r.get('skipped',0):>5}")

    bs = max(results, key=lambda x: x['sharpe'])
    bl = next((r for r in results if 'Baseline' in r['name']), results[0])
    fprint(f"\nBASELINE: {bl['name']} — Sharpe {bl['sharpe']}, CAGR {bl['cagr_pct']}%")
    fprint(f"BEST: {bs['name']} — Sharpe {bs['sharpe']}, CAGR {bs['cagr_pct']}%")
    delta = bs['sharpe'] - bl['sharpe']
    fprint(f"TIMING VALUE: {'+' if delta > 0 else ''}{delta:.2f} Sharpe")
    if abs(delta) < 0.20:
        fprint("VERDICT: Timing adds MINIMAL value. Entry day doesn't matter much — stick with bi-weekly baseline.")
    elif delta > 0.20:
        fprint(f"VERDICT: {bs['name']} timing improves Sharpe by {delta:.2f}. Consider as execution rule.")
    else:
        fprint("VERDICT: Timing filters HURT performance. Don't add complexity.")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'results': results, 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'day_of_week_options_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"dow_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'n_variants': len(results), 'capital': CAP})
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Day-of-Week Options Timing v1\n{'='*80}")

if __name__ == '__main__':
    main()
