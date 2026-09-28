#!/usr/bin/env python3
"""Honest Sharpe Re-Run — Re-run top strategies with CORRECT equity-based Sharpe.

The issue: All our options backtests compute Sharpe as:
    monthly_return = sum(monthly_pnl) / INITIAL_CAPITAL ($645)

When account grows from $645 to $39K, this makes a $500/month look like 77% return
instead of the honest ~1.3% of current equity.

This script re-runs the EXACT same backtest logic from multi_asset_momentum_options_v1.py
but computes Sharpe on equity-based returns.
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
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

CAP = 645.0
SPREAD_COMM = 4 * 0.65  # $2.60 round-trip
HAIRCUT = 0.15

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
ALL_ASSETS = SECTORS + ['GLD','SLV','USO','DBA','TLT','HYG','LQD','TIP','EFA','EEM','VWO','VNQ','AMLP','BITO']

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']

def compute_features(px, vol_s=None):
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
    h52 = px.iloc[-252:].max() if len(px) >= 252 else px.max()
    f['pct_52w_high'] = float(px.iloc[-1]/h52)
    r21 = rets.iloc[-21:]
    r63b = rets.iloc[-63:-21] if len(rets) > 63 else rets.iloc[:21]
    f['mom_accel'] = float(r21.mean() - r63b.mean()) if len(r63b) > 5 else 0.0
    monthly = px.resample('ME').last().pct_change().dropna().iloc[-12:]
    f['pct_pos_months_12m'] = float((monthly > 0).mean()) if len(monthly) > 3 else 0.5
    dr = r63[r63<0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    dd63 = float(f['maxdd_63d'])
    ann_ret = float((px.iloc[-1]/px.iloc[-252]-1)) if len(px) >= 252 else float(px.pct_change().mean()*252)
    f['calmar_1y'] = float(ann_ret / abs(dd63)) if abs(dd63) > 0.001 else 0.0
    f['up_capture'] = 1.0
    f['dn_capture'] = 1.0
    x = np.arange(63)
    y = np.log(px.iloc[-63:].values+1e-10)
    if len(y) == 63:
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = float(r_val**2)
        f['trend_slope_63d'] = float(slope*252)
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0
    v21 = float(rets.iloc[-21:].std())
    v63 = float(rets.iloc[-63:].std())
    f['rel_vol_21d'] = float(v21/(v63+1e-10))
    return f

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

def build_rankings(sc, sv, rebal_dates):
    rankings = {}
    train_periods = 12
    for i, rd in enumerate(rebal_dates):
        if i < train_periods or rd not in sc.index: continue
        rd_idx = sc.index.get_loc(rd)

        train_data = []
        for pi in range(max(0, i-train_periods), i):
            prd = rebal_dates[pi]
            if prd not in sc.index: continue
            prd_idx = sc.index.get_loc(prd)
            next_idx = sc.index.get_loc(rebal_dates[pi+1]) if pi+1 < len(rebal_dates) and rebal_dates[pi+1] in sc.index else min(prd_idx+10, len(sc)-1)
            for tk in sc.columns:
                if pd.isna(sc[tk].iloc[prd_idx]) or pd.isna(sc[tk].iloc[next_idx]): continue
                hist = sc[tk].iloc[:prd_idx+1].dropna()
                feats = compute_features(hist)
                if feats is None: continue
                fwd = float(sc[tk].iloc[next_idx]/sc[tk].iloc[prd_idx]-1)
                row = feats.copy()
                row['fwd_ret'] = fwd
                row['ticker'] = tk
                train_data.append(row)

        if len(train_data) < 20: continue
        tdf = pd.DataFrame(train_data)
        fc = [c for c in QM_COLS if c in tdf.columns]
        X = np.nan_to_num(tdf[fc].values.astype(np.float32))
        y = np.nan_to_num(tdf['fwd_ret'].values.astype(np.float32))

        try:
            ds = lgb.Dataset(X, label=y, free_raw_data=False)
            model = lgb.train({'objective':'regression','num_leaves':15,'learning_rate':0.05,
                              'min_child_samples':5,'verbose':-1,'n_jobs':4,
                              'feature_fraction':0.7,'bagging_fraction':0.7,'bagging_freq':5},
                             ds, num_boost_round=100)
        except: continue

        current = []
        for tk in sc.columns:
            if pd.isna(sc[tk].iloc[rd_idx]): continue
            hist = sc[tk].iloc[:rd_idx+1].dropna()
            feats = compute_features(hist)
            if feats is None: continue
            row = feats.copy(); row['ticker'] = tk; current.append(row)

        if not current: continue
        cdf = pd.DataFrame(current)
        Xc = np.nan_to_num(cdf[fc].values.astype(np.float32))
        preds = model.predict(Xc)
        rankings[rd] = dict(zip(cdf['ticker'], preds))

    return rankings

def simulate_with_honest_sharpe(name, rankings, sc, sh, sl, spy, vix, top_k=3, vix_min=20):
    """EXACT same simulation as original, but track equity properly."""
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
                           'win': pnl>0, 'regime': 'bull' if bull else 'bear',
                           'equity_before': round(equity-pnl, 2)})
        eq_curve.append(equity)

    return trades, equity, eq_curve

def compute_both_sharpes(trades, eq_curve, name):
    """Compute BOTH inflated and honest Sharpe for comparison."""
    if not trades or len(trades) < 10:
        fprint(f"  {name}: Too few trades ({len(trades)})")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp/(gl+1e-10)

    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['date'].dt.to_period('M')

    # INFLATED Sharpe (original method): monthly_pnl / CAP
    mr_inflated = tdf.groupby('month')['pnl'].sum() / CAP
    ny = max(len(mr_inflated)/12, 0.5)
    sh_inflated = (mr_inflated.mean()*12)/(mr_inflated.std()*np.sqrt(12)+1e-10) if len(mr_inflated) > 3 else 0
    dn_i = mr_inflated[mr_inflated<0]
    so_inflated = (mr_inflated.mean()*12)/(dn_i.std()*np.sqrt(12)+1e-10) if len(dn_i) > 1 else 0

    # HONEST Sharpe: monthly_pnl / equity_at_month_start
    equity_track = CAP
    month_start_equity = {}
    current_month = None
    monthly_pnl = {}

    for _, row in tdf.iterrows():
        m = row['month']
        if m != current_month:
            month_start_equity[m] = equity_track
            current_month = m
            monthly_pnl[m] = 0
        monthly_pnl[m] += row['pnl']
        equity_track += row['pnl']

    months = sorted(monthly_pnl.keys())
    mr_honest = pd.Series([monthly_pnl[m]/max(month_start_equity[m], 1.0) for m in months], index=months)
    sh_honest = (mr_honest.mean()*12)/(mr_honest.std()*np.sqrt(12)+1e-10) if len(mr_honest) > 3 else 0
    dn_h = mr_honest[mr_honest<0]
    so_honest = (mr_honest.mean()*12)/(dn_h.std()*np.sqrt(12)+1e-10) if len(dn_h) > 1 else 0

    # Equity curve metrics
    eq = np.array(eq_curve)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq-pk)/(pk+1e-10)).min())*100
    final = eq[-1]
    cagr = (final/CAP)**(1/ny)-1

    # Regime analysis
    bt = [t for t in trades if t['regime']=='bull']
    brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100
    r1_gap = abs(bw-brw)/max(bw,brw,1)

    # Adversarial gates on HONEST returns
    honest_rets = mr_honest.values
    gates = 0

    # G1: Permutation test
    perm_p = 1.0
    if len(honest_rets) >= 10:
        real_sr = np.mean(honest_rets)/(np.std(honest_rets)+1e-10)
        count = sum(1 for _ in range(5000) if
                    np.mean(honest_rets*np.random.choice([-1,1],len(honest_rets)))/(np.std(honest_rets)+1e-10)>=real_sr)
        perm_p = count/5000
    g1 = perm_p < 0.05; gates += g1

    # G2: R1 regime gap
    g2 = r1_gap < 0.50; gates += g2

    # G3: Sub-period stability
    mid = len(honest_rets)//2
    h1 = np.mean(honest_rets[:mid])/(np.std(honest_rets[:mid])+1e-10) if mid > 3 else 0
    h2 = np.mean(honest_rets[mid:])/(np.std(honest_rets[mid:])+1e-10) if len(honest_rets)-mid > 3 else 0
    g3 = h1 > 0 and h2 > 0; gates += g3

    # G4: Drop best month, still positive
    if len(honest_rets) > 5:
        tr = np.sort(honest_rets)[:-1]
        g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0
    else:
        g4 = False
    gates += g4

    inflation = sh_inflated/(sh_honest+1e-10)

    result = {
        'name': name,
        'sharpe_honest': round(sh_honest, 2),
        'sharpe_inflated': round(sh_inflated, 2),
        'inflation_ratio': round(inflation, 2),
        'sortino_honest': round(so_honest, 2),
        'sortino_inflated': round(so_inflated, 2),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'cagr_pct': round(cagr*100, 1),
        'maxdd_pct': round(mdd, 1),
        'final_equity': round(final, 2),
        'n_trades': n,
        'n_months': len(months),
        'bull_wr': round(bw, 1),
        'bear_wr': round(brw, 1),
        'r1_gap': round(r1_gap, 3),
        'perm_p': round(perm_p, 4),
        'sub_period_h1': round(h1, 3),
        'sub_period_h2': round(h2, 3),
        'gates': gates,
        'gate_detail': f"perm={'P' if g1 else 'F'} R1={'P' if g2 else 'F'} stable={'P' if g3 else 'F'} robust={'P' if g4 else 'F'}"
    }

    fprint(f"\n  {name}:")
    fprint(f"    {'INFLATED Sharpe':>20}: {sh_inflated:.2f}  (monthly_pnl / $645)")
    fprint(f"    {'HONEST Sharpe':>20}: {sh_honest:.2f}  (monthly_pnl / current_equity)")
    fprint(f"    {'Inflation':>20}: {inflation:.1f}x")
    fprint(f"    {'Sortino (honest)':>20}: {so_honest:.2f}")
    fprint(f"    {'WR':>20}: {wr:.1f}%  |  PF: {pf:.2f}")
    fprint(f"    {'CAGR':>20}: {cagr*100:.1f}%  |  MaxDD: {mdd:.1f}%")
    fprint(f"    {'Final equity':>20}: ${final:,.0f}  from  ${CAP}")
    fprint(f"    {'Bull WR':>20}: {bw:.1f}%  |  Bear WR: {brw:.1f}%  |  R1 gap: {r1_gap:.3f}")
    fprint(f"    {'Perm p':>20}: {perm_p:.4f}  |  Sub-period: H1={h1:.3f}, H2={h2:.3f}")
    fprint(f"    {'Gates':>20}: {gates}/4 ({result['gate_detail']})")

    return result

def main():
    t0 = datetime.now()
    fprint("="*70)
    fprint(f"HONEST SHARPE RE-RUN — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("="*70)
    fprint()
    fprint("Re-running top strategies with CORRECTED Sharpe calculation.")
    fprint("Comparing monthly_pnl/INITIAL_CAPITAL vs monthly_pnl/CURRENT_EQUITY.")
    fprint()

    import yfinance as yf
    tickers = list(set(ALL_ASSETS + ['SPY','^VIX']))
    fprint(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start='2008-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close_all = raw['Close'] if mi else raw
    high_all = raw['High'] if mi else raw
    low_all = raw['Low'] if mi else raw
    vol_all = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close_all.columns else 'VIX'
    vix = close_all[vc].dropna()
    spy = close_all['SPY'].dropna()

    configs = [
        ('B_Broad_Universe', ALL_ASSETS, 3, 20),        # Our best
        ('A_Sectors_Baseline', SECTORS, 3, 20),          # Original baseline
        ('E_Sectors_Bonds', SECTORS+['TLT','HYG','LQD','TIP'], 3, 20),  # Lowest DD
        ('F_Broad_AllVIX', ALL_ASSETS, 3, None),         # No VIX filter
    ]

    if MLFLOW_OK:
        mlflow.set_experiment("honest_sharpe_rerun_v1")
        mlflow.start_run(run_name=f"honest_{t0.strftime('%Y%m%d_%H%M')}")

    all_results = []

    for var_name, universe, top_k, vix_min in configs:
        fprint(f"\n{'='*60}")
        fprint(f"VARIANT: {var_name} ({len(universe)} assets, top-{top_k}, VIX>={vix_min or 'none'})")
        fprint(f"{'='*60}")

        available = [c for c in universe if c in close_all.columns and close_all[c].dropna().shape[0] > 500]
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

        tr, eq, cu = simulate_with_honest_sharpe(var_name, rankings, sc, sh, sl, spy, vix, top_k=top_k, vix_min=vix_min)
        r = compute_both_sharpes(tr, cu, var_name)
        if r:
            all_results.append(r)
            if MLFLOW_OK:
                try:
                    p = var_name[:18]
                    mlflow.log_metrics({
                        f'{p}_sh_honest': r['sharpe_honest'],
                        f'{p}_sh_inflated': r['sharpe_inflated'],
                        f'{p}_inflation': r['inflation_ratio'],
                        f'{p}_gates': r['gates'],
                        f'{p}_wr': r['win_rate'],
                        f'{p}_cagr': r['cagr_pct'],
                        f'{p}_mdd': r['maxdd_pct']
                    })
                except: pass

    # SUMMARY
    fprint(f"\n{'='*90}")
    fprint("SUMMARY: HONEST vs INFLATED SHARPE")
    fprint(f"{'='*90}")
    fprint(f"{'Variant':<25} {'Inflated':>10} {'Honest':>10} {'Inflate':>8} {'WR':>6} {'CAGR':>7} {'MDD':>7} {'Gates':>7}")
    fprint("-"*90)
    for r in sorted(all_results, key=lambda x: x['sharpe_honest'], reverse=True):
        inf_tag = "✅" if r['inflation_ratio'] < 2 else "⚠️" if r['inflation_ratio'] < 5 else "❌"
        fprint(f"{r['name']:<25} {r['sharpe_inflated']:>10.2f} {r['sharpe_honest']:>10.2f} {r['inflation_ratio']:>7.1f}x {r['win_rate']:>5.1f}% {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['gates']:>5}/4 {inf_tag}")

    # KEY FINDINGS
    if all_results:
        best = max(all_results, key=lambda x: x['sharpe_honest'])
        avg_infl = np.mean([r['inflation_ratio'] for r in all_results])
        fprint(f"\n{'='*70}")
        fprint("KEY FINDINGS")
        fprint(f"{'='*70}")
        fprint(f"  Average Sharpe inflation: {avg_infl:.1f}x")
        fprint(f"  Best HONEST Sharpe: {best['name']} = {best['sharpe_honest']:.2f}")
        fprint(f"    (was reported as {best['sharpe_inflated']:.2f} — inflation {best['inflation_ratio']:.1f}x)")
        fprint(f"  Strategies passing 4/4 gates (on honest returns): {sum(1 for r in all_results if r['gates']>=4)}/{len(all_results)}")

        # Is the strategy STILL good after honest Sharpe?
        if best['sharpe_honest'] >= 1.0:
            fprint(f"\n  ✅ VERDICT: Strategy IS REAL. Honest Sharpe {best['sharpe_honest']:.2f} is still strong.")
            fprint(f"     The inflated number ({best['sharpe_inflated']:.2f}) overstates it, but the edge is legitimate.")
        elif best['sharpe_honest'] >= 0.5:
            fprint(f"\n  ⚠️ VERDICT: Strategy has a MODEST edge. Honest Sharpe {best['sharpe_honest']:.2f}.")
            fprint(f"     The inflated {best['sharpe_inflated']:.2f} is misleading. Proceed with caution.")
        else:
            fprint(f"\n  ❌ VERDICT: Strategy may be FAKE. Honest Sharpe only {best['sharpe_honest']:.2f}.")
            fprint(f"     The {best['sharpe_inflated']:.2f} was entirely due to compounding inflation.")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save results
    output = {
        'audit_date': t0.isoformat(),
        'issue': 'Sharpe inflated by dividing monthly PnL by initial capital instead of current equity',
        'results': all_results,
        'runtime_s': round(elapsed, 1)
    }
    out_path = RESULTS_DIR / 'honest_sharpe_rerun_results.json'
    out_path.write_text(json.dumps(output, indent=2, default=str))
    fprint(f"Results saved.")

    if MLFLOW_OK:
        try:
            mlflow.log_artifact(str(out_path))
            mlflow.end_run()
        except: pass

    fprint("\nAUDIT COMPLETE.")

if __name__ == '__main__':
    main()
