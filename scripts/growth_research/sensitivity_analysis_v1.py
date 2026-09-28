#!/usr/bin/env python3
"""Sensitivity Analysis v1 — Is Sharpe 4.70 robust or fragile?

Tests how our best strategy (broad universe momentum with VIX>20 filter)
responds to parameter perturbations. If small changes kill the edge → overfitting.
If edge persists → robust.

Dimensions tested:
  1. Spread width: 2%, 3% (baseline), 4%, 5%, 7%
  2. DTE: 14, 21, 30 (baseline), 45, 60
  3. Top-K: 1, 2, 3 (baseline), 4, 5
  4. VIX threshold: 15, 18, 20 (baseline), 22, 25, none
  5. Rebalance frequency: weekly, bi-weekly (baseline), monthly
  6. Lookback (training): 6, 12 (baseline), 18, 24 periods
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
RESULTS_PATH = RESULTS_DIR / 'sensitivity_analysis_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

# Broad universe (from multi-asset v1 winner)
UNIVERSE = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC',
            'GLD','SLV','USO','DBA','TLT','HYG','LQD','TIP','EFA','EEM','VWO',
            'VNQ','AMLP','BITO']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']

def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = list(set(UNIVERSE + ['SPY', '^VIX']))
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    avail = [c for c in UNIVERSE if c in close.columns and close[c].dropna().shape[0] > 500]
    sc = close[avail].dropna(how='all')
    sh = high[[c for c in avail if c in high.columns]].dropna(how='all')
    sl = low[[c for c in avail if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in avail if c in volume.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(avail)} assets")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]

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

def build_rankings(sc, sv, rebal_dates, train_periods=12, fwd_days=14):
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats: continue
            fi = min(idx+fwd_days, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns: df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(train_periods, len(dates)):
        td = dates[max(0,i-train_periods):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[QM_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    return rankings

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val):
    T = dte/252.0
    if T <= 0: return max(0, S-K)
    return max(0, S-K) + atr*np.sqrt(T)*max(0.3, vix_val/20.0)*np.exp(-3.0*abs(S-K)/S)

def simulate(rankings, sc, sh, sl, spy, vix, spread_pct=3.0, dte=30, top_k=3, vix_min=20):
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
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
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        max_pos = min(200, equity/3)
        if max_pos < 30: eq_curve.append(equity); continue
        n_ent = 0
        for tk, _ in ranked:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt); ei = min(di+dte, len(sc)-1); Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S*(1+spread_pct/100))
            lp = atr_premium(S, K1, dte, av, cv)*(1+HAIRCUT)
            sp = atr_premium(S, K2, dte, av, cv)*(1-HAIRCUT)
            val = lp-sp; cost = val*100+SPREAD_COMM; width = K2-K1
            mx_prof = (width-val)*100-SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue
            pnl = None
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci]); rd = max(0,dte-(ci-di))
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))
                si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                if si-cost >= mx_prof*0.50 or rd < 7: pnl = si-cost; break
            if pnl is None:
                pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - val*100 - SPREAD_COMM
            equity += pnl; n_ent += 1
            trades.append({'pnl': pnl, 'win': pnl>0, 'regime': 'bull' if bull else 'bear'})
        eq_curve.append(equity)
    return trades, equity, eq_curve

def quick_metrics(trades, final_eq, eq_curve):
    if not trades or len(trades) < 10: return None
    n = len(trades); wr = sum(1 for t in trades if t['win'])/n*100
    pnls = [t['pnl'] for t in trades]
    # Monthly aggregation
    # Simple: assume trades evenly distributed
    chunk = max(1, n//60)  # ~60 months in 5yr
    monthly = [sum(pnls[i:i+chunk]) for i in range(0, n, chunk)]
    mr = np.array(monthly)/CAP
    ny = max(len(mr)/12, 0.5)
    sh = (np.mean(mr)*12)/(np.std(mr)*np.sqrt(12)+1e-10) if len(mr) > 3 else 0
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq)
    mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100
    rg = abs(bw-brw)/max(bw,brw,1)
    # Quick perm
    rs = np.mean(mr)/(np.std(mr)+1e-10)
    pp = sum(1 for _ in range(500) if np.mean(mr*np.random.choice([-1,1],len(mr)))/(np.std(mr)+1e-10)>=rs)/500
    return {'n': n, 'wr': round(wr,1), 'sharpe': round(sh,2), 'cagr': round(cagr*100,1),
            'mdd': round(mdd*100,1), 'pf': round(pf,2), 'final': round(final_eq,0),
            'perm_p': round(pp,3), 'r1_gap': round(rg,3), 'pass': pp < 0.05 and rg < 0.50}

def main():
    t0 = datetime.now()
    fprint(f"Sensitivity Analysis v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Testing robustness of Sharpe 4.70 broad-universe strategy")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix = download_data()

    # Pre-build rankings for different rebalance frequencies
    fprint("\nBuilding rankings for different frequencies...")
    bd_w = pd.DatetimeIndex(sc.index.to_series().resample('W-FRI').last().dropna().values)
    bd_bw = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    bd_m = pd.DatetimeIndex(sc.index.to_series().resample('ME').last().dropna().values)

    # Build rankings with different training lookbacks
    fprint("  Bi-weekly (baseline, 12-period lookback)...")
    rank_bw_12 = build_rankings(sc, sv, bd_bw, train_periods=12)
    fprint(f"    {len(rank_bw_12)} dates")

    fprint("  Weekly...")
    rank_w = build_rankings(sc, sv, bd_w, train_periods=12)
    fprint(f"    {len(rank_w)} dates")

    fprint("  Monthly...")
    rank_m = build_rankings(sc, sv, bd_m, train_periods=12)
    fprint(f"    {len(rank_m)} dates")

    fprint("  Bi-weekly 6-period lookback...")
    rank_bw_6 = build_rankings(sc, sv, bd_bw, train_periods=6)
    fprint(f"    {len(rank_bw_6)} dates")

    fprint("  Bi-weekly 18-period lookback...")
    rank_bw_18 = build_rankings(sc, sv, bd_bw, train_periods=18)
    fprint(f"    {len(rank_bw_18)} dates")

    fprint("  Bi-weekly 24-period lookback...")
    rank_bw_24 = build_rankings(sc, sv, bd_bw, train_periods=24)
    fprint(f"    {len(rank_bw_24)} dates")

    results = {}

    # 1. Spread width sensitivity
    fprint("\n=== SPREAD WIDTH ===")
    for spw in [2, 3, 4, 5, 7]:
        tr, eq, cu = simulate(rank_bw_12, sc, sh, sl, spy, vix, spread_pct=spw)
        m = quick_metrics(tr, eq, cu)
        key = f"spread_{spw}pct"
        results[key] = m
        if m: fprint(f"  {spw}%: Sh {m['sharpe']:.2f} | WR {m['wr']:.1f}% | CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | {'PASS' if m['pass'] else 'FAIL'}")

    # 2. DTE sensitivity
    fprint("\n=== DTE ===")
    for dte in [14, 21, 30, 45, 60]:
        tr, eq, cu = simulate(rank_bw_12, sc, sh, sl, spy, vix, dte=dte)
        m = quick_metrics(tr, eq, cu)
        key = f"dte_{dte}"
        results[key] = m
        if m: fprint(f"  {dte}d: Sh {m['sharpe']:.2f} | WR {m['wr']:.1f}% | CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | {'PASS' if m['pass'] else 'FAIL'}")

    # 3. Top-K sensitivity
    fprint("\n=== TOP-K ===")
    for tk in [1, 2, 3, 4, 5]:
        tr, eq, cu = simulate(rank_bw_12, sc, sh, sl, spy, vix, top_k=tk)
        m = quick_metrics(tr, eq, cu)
        key = f"topk_{tk}"
        results[key] = m
        if m: fprint(f"  Top-{tk}: Sh {m['sharpe']:.2f} | WR {m['wr']:.1f}% | CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | {'PASS' if m['pass'] else 'FAIL'}")

    # 4. VIX threshold sensitivity
    fprint("\n=== VIX THRESHOLD ===")
    for vt in [None, 15, 18, 20, 22, 25]:
        tr, eq, cu = simulate(rank_bw_12, sc, sh, sl, spy, vix, vix_min=vt)
        m = quick_metrics(tr, eq, cu)
        key = f"vix_{vt if vt else 'none'}"
        results[key] = m
        lbl = f"VIX>{vt}" if vt else "No filter"
        if m: fprint(f"  {lbl}: Sh {m['sharpe']:.2f} | WR {m['wr']:.1f}% | #{m['n']} | CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | {'PASS' if m['pass'] else 'FAIL'}")

    # 5. Rebalance frequency sensitivity
    fprint("\n=== REBALANCE FREQUENCY ===")
    for freq, rnk, label in [('weekly', rank_w, 'Weekly'), ('biweekly', rank_bw_12, 'Bi-weekly'), ('monthly', rank_m, 'Monthly')]:
        tr, eq, cu = simulate(rnk, sc, sh, sl, spy, vix)
        m = quick_metrics(tr, eq, cu)
        results[f"freq_{freq}"] = m
        if m: fprint(f"  {label}: Sh {m['sharpe']:.2f} | WR {m['wr']:.1f}% | #{m['n']} | CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | {'PASS' if m['pass'] else 'FAIL'}")

    # 6. Lookback sensitivity
    fprint("\n=== LOOKBACK PERIODS ===")
    for lb, rnk in [(6, rank_bw_6), (12, rank_bw_12), (18, rank_bw_18), (24, rank_bw_24)]:
        tr, eq, cu = simulate(rnk, sc, sh, sl, spy, vix)
        m = quick_metrics(tr, eq, cu)
        results[f"lookback_{lb}"] = m
        if m: fprint(f"  {lb} periods: Sh {m['sharpe']:.2f} | WR {m['wr']:.1f}% | CAGR {m['cagr']:.1f}% | MDD {m['mdd']:.1f}% | {'PASS' if m['pass'] else 'FAIL'}")

    # Robustness score
    fprint(f"\n{'='*80}")
    fprint(f"ROBUSTNESS SUMMARY")
    fprint(f"{'='*80}")
    valid = {k: v for k, v in results.items() if v is not None}
    n_pass = sum(1 for v in valid.values() if v['pass'])
    n_total = len(valid)
    sharpes = [v['sharpe'] for v in valid.values()]
    fprint(f"Configurations tested: {n_total}")
    fprint(f"Pass rate: {n_pass}/{n_total} ({n_pass/n_total*100:.0f}%)")
    fprint(f"Sharpe range: {min(sharpes):.2f} — {max(sharpes):.2f}")
    fprint(f"Sharpe mean: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
    fprint(f"Sharpe median: {np.median(sharpes):.2f}")

    # Find most/least sensitive dimensions
    dims = {
        'Spread width': [v['sharpe'] for k, v in valid.items() if k.startswith('spread')],
        'DTE': [v['sharpe'] for k, v in valid.items() if k.startswith('dte')],
        'Top-K': [v['sharpe'] for k, v in valid.items() if k.startswith('topk')],
        'VIX threshold': [v['sharpe'] for k, v in valid.items() if k.startswith('vix')],
        'Rebalance freq': [v['sharpe'] for k, v in valid.items() if k.startswith('freq')],
        'Lookback': [v['sharpe'] for k, v in valid.items() if k.startswith('lookback')],
    }
    fprint(f"\n{'Dimension':<18} {'Range':>8} {'Mean±Std':>12} {'Sensitive?':>12}")
    fprint("-"*55)
    for dim, shs in sorted(dims.items(), key=lambda x: max(x[1])-min(x[1]), reverse=True):
        rng = max(shs) - min(shs)
        sens = "YES" if rng > 1.0 else "MODERATE" if rng > 0.5 else "NO"
        fprint(f"{dim:<18} {rng:>7.2f} {np.mean(shs):>5.2f}±{np.std(shs):>4.2f} {sens:>12}")

    if np.mean(sharpes) > 3.0 and n_pass/n_total > 0.7:
        fprint(f"\nVERDICT: ROBUST. Strategy maintains Sharpe > 3 across {n_pass/n_total*100:.0f}% of parameter perturbations.")
    elif np.mean(sharpes) > 2.0:
        fprint(f"\nVERDICT: MODERATELY ROBUST. Mean Sharpe {np.mean(sharpes):.2f} but some sensitivity found.")
    else:
        fprint(f"\nVERDICT: FRAGILE. Edge is parameter-dependent. Use baseline config only.")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'results': {k: v for k, v in results.items() if v},
                 'robustness': {'n_pass': n_pass, 'n_total': n_total, 'sharpe_mean': round(np.mean(sharpes),2),
                                'sharpe_std': round(np.std(sharpes),2), 'sharpe_min': round(min(sharpes),2),
                                'sharpe_max': round(max(sharpes),2)},
                 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'sensitivity_analysis_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"sens_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'n_configs': n_total, 'capital': CAP, 'universe_size': len(UNIVERSE)})
                mlflow.log_metrics({'pass_rate': n_pass/n_total, 'sharpe_mean': np.mean(sharpes),
                                    'sharpe_std': np.std(sharpes), 'sharpe_min': min(sharpes), 'sharpe_max': max(sharpes)})
                mlflow.log_artifact(str(RESULTS_PATH))
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Sensitivity Analysis v1\n{'='*80}")

if __name__ == '__main__':
    main()
