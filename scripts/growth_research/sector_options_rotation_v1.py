#!/usr/bin/env python3
"""Sector Options Rotation v1 — Realistic ATR-based execution test for $645 account.
Combines proven LightGBM sector ranking with options strategies using ATR-based
pricing (NOT Black-Scholes) with 15% bid-ask haircut for realistic friction.

Variants: A=BullSpread 3%/30d, B=BullSpread 5%/45d, C=PutCredit 5%/30d,
D=Dynamic (regime-switch), E=LGBM Quality filter, F=BiWeekly rebalance.
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
import yfinance as yf, lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'sector_options_rotation_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4 * LEG_COMM  # $2.60 RT
HAIRCUT = 0.15  # 15% bid-ask haircut on premiums
FEAT_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

def download_data():
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
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

# === LightGBM Walk-Forward Ranking ===
def compute_features(px_daily, idx, ticker):
    px = px_daily[ticker].iloc[:idx+1].dropna()
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0.0
    px63 = px.iloc[-63:]; pk = px63.cummax()
    f['maxdd_63d'] = float(((px63/pk)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    return f

def lgbm_wf(px_daily, rebal_dates, train_periods=12):
    fprint("Running LightGBM walk-forward...")
    records = []
    for dt in rebal_dates:
        idx = px_daily.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in px_daily.columns:
            feats = compute_features(px_daily, idx, tk)
            if not feats: continue
            fi = min(idx+21, len(px_daily)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(px_daily[tk].iloc[fi]/px_daily[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(train_periods, len(dates)):
        td = dates[max(0, i-train_periods):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[FEAT_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[FEAT_COLS].values.astype(np.float32))
        try:
            try:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, device='gpu', verbose=-1)
                m.fit(Xt, yt)
            except Exception:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
                m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except Exception: continue
    fprint(f"  Rankings for {len(rankings)} dates")
    return rankings

# === ATR-Based Options Pricing (NOT Black-Scholes) ===
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    """ATR-based option premium. More conservative than B-S for realistic fills."""
    T = dte / 252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val / 20.0)
    time_prem = atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S-K)/S)
    return intrinsic + time_prem

def price_spread(S, spread_pct, dte, atr, vix_val, stype='bull'):
    """Price bull call or put credit spread. Returns (cost_or_credit, max_profit, max_loss, K1, K2)."""
    if stype == 'bull':
        K1, K2 = round(S), round(S*(1+spread_pct/100))  # long ATM, short OTM
        lp = atr_premium(S, K1, dte, atr, vix_val, 'call') * (1+HAIRCUT)
        sp = atr_premium(S, K2, dte, atr, vix_val, 'call') * (1-HAIRCUT)
        debit = lp - sp; width = K2 - K1
        return debit, (width-debit)*100-SPREAD_COMM, debit*100+SPREAD_COMM, K1, K2
    else:  # put_credit: sell 5% OTM put, buy 8% OTM put (3% wide)
        K1 = round(S*(1-5.0/100)); K2 = round(S*(1-8.0/100))  # K1=short, K2=long
        sp = atr_premium(S, K1, dte, atr, vix_val, 'put') * (1-HAIRCUT)
        lp = atr_premium(S, K2, dte, atr, vix_val, 'put') * (1+HAIRCUT)
        credit = sp - lp; width = K1 - K2
        return credit, credit*100-SPREAD_COMM, (width-credit)*100+SPREAD_COMM, K1, K2

# === Simulation Engine ===
def simulate(name, rankings, sc, sh, sl, spy, vix, stype='bull', spct=3.0,
             dte=30, top_k=3, quality=False):
    fprint(f"\n--- {name} ---")
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        sv = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv
        bull = sv >= sm; cv = float(vix.loc[dt])
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if quality:
            med = np.median([s for _,s in ranked])
            ranked = [(t,s) for t,s in ranked if s > med]
        picks = [t for t,_ in ranked[:top_k]]
        # Strategy selection
        strat = ('bull' if bull else 'put') if stype=='dynamic' else ('bull' if stype=='bull_spread' else 'put')
        max_pos = min(200, equity/3)
        if max_pos < 30: eq_curve.append(equity); continue
        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt); ei = min(di+dte, len(sc)-1)
            Se = float(sc[tk].iloc[ei])
            val, mx_prof, mx_loss, K1, K2 = price_spread(S, spct, dte, av, cv, strat)
            cost = mx_loss if strat=='put' else val*100+SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue
            # Walk forward checking for early exit (50% profit or DTE<7)
            pnl, aei = None, ei
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci]); dh = ci-di; rd = max(0, dte-dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))
                if strat == 'bull':
                    si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                    cp = si - (val*100+SPREAD_COMM)
                else:  # put credit
                    liab = (max(0,K1-Sc)-max(0,K2-Sc))*100
                    cp = val*100*(1-tm) - liab - SPREAD_COMM
                if cp >= mx_prof*0.50 or rd < 7: pnl = cp; aei = ci; break
            if pnl is None:  # hold to expiry
                if strat == 'bull':
                    pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - val*100 - SPREAD_COMM
                else:
                    pnl = val*100 - (max(0,K1-Se)-max(0,K2-Se))*100 - SPREAD_COMM
            equity += pnl; n_ent += 1
            trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                'ticker': tk, 'strategy': strat, 'pnl': round(pnl,2), 'win': pnl>0,
                'regime': 'bull' if bull else 'bear', 'underlying_ret': round((Se/S-1)*100,2)})
        eq_curve.append(equity)
    return trades, equity, eq_curve

# === Metrics & Validation ===
def metrics(trades, final_eq, eq_curve, name):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]; total = sum(pnls)
    tdf = pd.DataFrame(trades); tdf['m'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    mr = tdf.groupby('m')['pnl'].sum() / CAP; ny = max(len(mr)/12, 0.5)
    sh = (mr.mean()*12)/(mr.std()*np.sqrt(12)+1e-10) if len(mr) > 3 else 0
    dn = mr[mr<0]; so = (mr.mean()*12)/(dn.std()*np.sqrt(12)+1e-10) if len(dn) > 1 else 0
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10); ap = np.mean(pnls)
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100
    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1), 'sharpe': round(sh,2),
         'sortino': round(so,2), 'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'avg_pnl': round(ap,2), 'avg_trades_yr': round(n/ny,1),
         'final_equity': round(final_eq,2), 'total_pnl': round(total,2),
         'bull_wr': round(bw,1), 'bear_wr': round(brw,1), 'bull_n': len(bt), 'bear_n': len(brt),
         'monthly_returns': mr.values.tolist()}
    fprint(f"  {name}: {n} trades | WR {wr:.1f}% | Sharpe {sh:.2f} | Sort {so:.2f} | "
           f"CAGR {cagr*100:.1f}% | MaxDD {mdd*100:.1f}% | PF {pf:.2f} | ${CAP:.0f}->${final_eq:.0f}")
    return r

def validate(r):
    if r is None: return r
    rets = np.array(r['monthly_returns'])
    if len(rets) < 10: r.update({'gates':0,'perm_p':1.0,'r1_gap':1.0}); return r
    gates = 0; rs = np.mean(rets)/(np.std(rets)+1e-10)
    # G1: Permutation
    pp = sum(1 for _ in range(1000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/1000
    g1 = pp < 0.05; gates += g1
    # G2: Regime
    rg = abs(r['bull_wr']-r['bear_wr'])/max(r['bull_wr'],r['bear_wr'],1)
    g2 = rg < 0.50; gates += g2
    # G3: Sub-period
    mid = len(rets)//2
    h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
    h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
    g3 = h1 > 0 and h2 > 0; gates += g3
    # G4: Outlier removal
    g4 = False
    if len(rets) > 5:
        tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0
    gates += g4
    r.update({'gates': gates, 'perm_p': round(pp,3), 'r1_gap': round(rg,3),
              'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
              'h1_sh': round(h1,2), 'h2_sh': round(h2,2)})
    fprint(f"  Gates: G1={'P' if g1 else 'F'}(p={pp:.3f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'} => {gates}/4")
    return r

# === Main ===
def main():
    fprint(f"Sector Options Rotation v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*70}\nCapital: ${CAP:.0f} | Pricing: ATR+{HAIRCUT:.0%} haircut (NOT B-S) | Comm: ${SPREAD_COMM}/spread RT")
    fprint("="*70)
    sc, sh, sl, spy, vix = download_data()
    md = pd.DatetimeIndex(sc.index.to_series().resample('ME').last().dropna().values)
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    fprint("\n--- LGBM Monthly ---"); rm = lgbm_wf(sc, md)
    fprint("\n--- LGBM Biweekly ---"); rb = lgbm_wf(sc, bd)
    if not rm: fprint("FATAL: No rankings"); return
    results = []
    # A-F variants
    configs = [
        ('A_BullSpread_3pct_30d', rm, 'bull_spread', 3.0, 30, 3, False),
        ('B_BullSpread_5pct_45d', rm, 'bull_spread', 5.0, 45, 3, False),
        ('C_PutCredit_5pct_30d',  rm, 'put_credit',  5.0, 30, 3, False),
        ('D_Dynamic',             rm, 'dynamic',     3.0, 30, 3, False),
        ('E_LGBM_Quality',        rm, 'bull_spread', 3.0, 30, 3, True),
        ('F_BiWeekly',            rb, 'bull_spread', 3.0, 30, 3, False),
    ]
    for nm, rnk, st, sp, dt, tk, qf in configs:
        tr, eq, cu = simulate(nm, rnk, sc, sh, sl, spy, vix, st, sp, dt, tk, qf)
        r = metrics(tr, eq, cu, nm)
        if r: results.append(validate(r))
    if not results: fprint("No results"); return
    # Summary
    fprint(f"\n{'='*85}\nSUMMARY — Sector Options Rotation v1 (ATR-based)\n{'='*85}")
    fprint(f"{'Variant':<25} {'#':>5} {'WR':>6} {'Sh':>6} {'So':>6} {'CAGR':>7} {'MDD':>7} {'PF':>5} {'Final$':>8} {'G':>4}")
    fprint("-"*85)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} "
               f"{r['sortino']:>6.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>5.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4")
    bs = max(results, key=lambda x: x['sharpe'])
    vl = [r for r in results if r['gates'] >= 3]
    fprint(f"\nBEST SHARPE: {bs['name']} — Sharpe {bs['sharpe']}, CAGR {bs['cagr_pct']}%, Gates {bs['gates']}/4")
    if vl:
        bv = max(vl, key=lambda x: x['sharpe'])
        fprint(f"BEST VALIDATED (3+): {bv['name']} — Sharpe {bv['sharpe']}, CAGR {bv['cagr_pct']}%")
    else:
        fprint("NO variant passes 3+ gates with realistic ATR pricing.")
    fprint(f"\nNOTE: ATR pricing with {HAIRCUT:.0%} haircut. Compare to B-S backtests for friction sensitivity.")
    # MLflow
    if MLFLOW_OK:
        try:
            en = 'sector_options_rotation_v1'
            try:
                if not mlflow.get_experiment_by_name(en): mlflow.create_experiment(en)
            except Exception: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"rot_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'pricing':'ATR','haircut':HAIRCUT,'capital':CAP,'comm_leg':LEG_COMM})
                for r in results:
                    p = r['name'].split('_')[0]
                    for k in ['sharpe','cagr_pct','maxdd_pct','gates','win_rate']:
                        mlflow.log_metric(f"{p}_{k.replace('_pct','')}", r[k])
                try: mlflow.log_artifact(str(RESULTS_PATH))
                except Exception: pass
            fprint("MLflow: logged")
        except Exception as e: fprint(f"MLflow failed: {e}")
    # Save
    save = {'strategy': 'Sector Options Rotation v1 — ATR-based', 'run_date': datetime.now().isoformat(),
            'pricing': f'ATR + {HAIRCUT:.0%} haircut (NOT B-S)', 'capital': CAP,
            'commission': f'${LEG_COMM}/leg, ${SPREAD_COMM}/spread RT',
            'variants': [{k:v for k,v in r.items() if k!='monthly_returns'} for r in results],
            'best_sharpe': bs['name'], 'best_validated': vl[0]['name'] if vl else 'NONE'}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save, f, indent=2, default=lambda o: float(o) if hasattr(o,'__float__') else str(o))
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    main()
