#!/usr/bin/env python3
"""
Multi-Timeframe Momentum Ensemble V1 (2026-07-28)
===================================================
3 independent LGBM models (weekly/monthly/quarterly) rank sector ETFs.
Only trade when models agree. 6 variants, 5-gate validation.
Walk-forward: 252d sliding train. Data from 2015. Capital $645, zero commission.
Output: output/growth_research/multi_tf_ensemble_v1/
MLflow: multi_tf_ensemble_v1
"""
import json, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

_bp = print
def fprint(*a, **k): _bp(*a, **k); sys.stdout.flush()

BASE = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
fprint(f"Running on: {BASE}"); sys.path.insert(0, str(BASE))
OUTPUT_DIR = BASE / "output" / "growth_research" / "multi_tf_ensemble_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI, EXP_NAME, MLFLOW_OK = "http://jupiter:5000", "multi_tf_ensemble_v1", False
try:
    import mlflow; mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(EXP_NAME); MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}")
except Exception as e: fprint(f"MLflow not available: {e}")

HAS_LGBM = False
try: import lightgbm as lgb; HAS_LGBM = True
except ImportError: fprint("WARNING: LightGBM unavailable, using momentum fallback.")

# ── CONFIG ──
ETFS = ['XLK','XLV','XLF','XLE','XLI','XLC','XLY','XLP','XLU','XLRE','XLB']
CAP, DATA_START, TRAIN_W, REBAL_D, SAMP_EVERY = 645.0, '2015-01-01', 252, 21, 21
SHARED = ['corr_to_spy_63d','beta_to_spy_63d','rel_strength_vs_mean','vol_ratio_21_63','rsi_14']
WEEKLY_F = ['ret_5d','ret_10d','ret_21d','vol_5d','vol_10d','vol_21d','sharpe_21d','maxdd_21d',
    'mom_accel_5_10','mom_accel_5_21','pct_pos_days_10','pct_pos_days_21','trend_slope_21d',
    'trend_r2_21d','roc_5d','roc_10d','rank_5d'] + SHARED
MONTHLY_F = ['ret_5d','ret_10d','ret_21d','ret_63d','vol_21d','vol_63d','sharpe_63d','maxdd_63d',
    'mom_accel_21_63','pct_pos_days_21','pct_pos_days_63','trend_slope_63d','trend_r2_63d',
    'sortino_63d','calmar_63d','roc_21d','rank_21d'] + SHARED
QUARTERLY_F = ['ret_21d','ret_63d','ret_126d','ret_252d','vol_63d','vol_126d','sharpe_126d',
    'maxdd_126d','mom_accel_63_126','pct_pos_months_6m','pct_pos_months_12m','trend_slope_126d',
    'trend_r2_126d','sortino_126d','calmar_1y','pct_52w_high','rank_63d'] + SHARED

VNAMES = {'A':'Ensemble Top-2 Monthly','B':'2/3 Agreement Top-2','C':'3/3 Agreement Top-1',
          'D':'Ensemble+15% TrailStop','E':'Ensemble+VIX Filter','F':'Ensemble+Short Bot-1'}

def download_data():
    import yfinance as yf
    tickers = ETFS + ['SPY', '^VIX']
    fprint(f"Downloading {len(tickers)} tickers from {DATA_START}...")
    raw = yf.download(tickers, start=DATA_START, progress=False)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close.columns, pd.MultiIndex): close.columns = close.columns.get_level_values(-1)
    close = close.ffill().rename(columns={'^VIX': 'VIX'})
    vix, spy = close['VIX'].dropna(), close['SPY'].dropna()
    avail = [c for c in ETFS if c in close.columns]
    fc = close[avail].ffill().dropna(how='all')
    ix = fc.index.intersection(spy.index).intersection(vix.index)
    fprint(f"Data: {ix[0].date()} to {ix[-1].date()}, {len(ix)} days, {len(avail)} ETFs")
    return fc.loc[ix], spy.loc[ix], vix.loc[ix]

def _trend(px, lb):
    if len(px) < lb: return 0.0, 0.0
    y = np.log(px.iloc[-lb:].values + 1e-10)
    s, _, r, _, _ = stats.linregress(np.arange(len(y)), y)
    return s * 252, r ** 2

def compute_feats(px, spy_px=None, all_r21=None):
    if len(px) < 260: return None
    rets, f = px.pct_change().dropna(), {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    for lb, nm in [(5,'vol_5d'),(10,'vol_10d'),(21,'vol_21d'),(63,'vol_63d'),(126,'vol_126d')]:
        f[nm] = float(rets.iloc[-lb:].std() * np.sqrt(252)) if len(rets) >= lb else 0.2
    f['vol_ratio_21_63'] = f['vol_21d'] / (f['vol_63d'] + 1e-10)
    for lb, nm in [(21,'sharpe_21d'),(63,'sharpe_63d'),(126,'sharpe_126d')]:
        r = rets.iloc[-lb:]
        f[nm] = float(r.mean() / (r.std() + 1e-10) * np.sqrt(252)) if len(r) > 10 else 0.0
    for lb, nm in [(63,'sortino_63d'),(126,'sortino_126d')]:
        r = rets.iloc[-lb:]; dr = r[r < 0]
        f[nm] = float(r.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    for lb, nm in [(21,'maxdd_21d'),(63,'maxdd_63d'),(126,'maxdd_126d')]:
        p = px.iloc[-lb:]; f[nm] = float(((p / p.cummax()) - 1).min()) if len(p) > 1 else 0.0
    f['mom_accel_5_10'] = f['ret_5d'] - f['ret_10d'] / 2
    f['mom_accel_5_21'] = f['ret_5d'] - f['ret_21d'] / 4
    f['mom_accel_21_63'] = f['ret_21d'] - f['ret_63d'] / 3
    f['mom_accel_63_126'] = f['ret_63d'] - f['ret_126d'] / 2
    for lb, nm in [(10,'pct_pos_days_10'),(21,'pct_pos_days_21'),(63,'pct_pos_days_63')]:
        f[nm] = float((rets.iloc[-lb:] > 0).mean()) if len(rets) >= lb else 0.5
    mo = rets.resample('ME').sum()
    f['pct_pos_months_6m'] = float((mo.iloc[-6:] > 0).mean()) if len(mo) >= 6 else 0.5
    f['pct_pos_months_12m'] = float((mo.iloc[-12:] > 0).mean()) if len(mo) >= 12 else 0.5
    for lb, sn, rn in [(21,'trend_slope_21d','trend_r2_21d'),(63,'trend_slope_63d','trend_r2_63d'),
                        (126,'trend_slope_126d','trend_r2_126d')]:
        f[sn], f[rn] = _trend(px, lb)
    f['roc_5d'], f['roc_10d'], f['roc_21d'] = f['ret_5d'], f['ret_10d'], f['ret_21d']
    f['calmar_63d'] = f['ret_63d'] / (abs(f['maxdd_63d']) + 1e-10)
    f['calmar_1y'] = f.get('ret_252d', 0) / (abs(f['maxdd_126d']) + 1e-10)
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max()) if len(px) >= 252 else 1.0
    delta = px.diff(); gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi_14'] = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10))) if len(px) > 14 else 50.0
    if spy_px is not None and len(spy_px) >= 63:
        sr = spy_px.pct_change().dropna(); ml = min(len(rets), len(sr))
        if ml >= 63:
            f['corr_to_spy_63d'] = float(rets.iloc[-63:].corr(sr.iloc[-63:]))
            cv = np.cov(rets.iloc[-63:].values, sr.iloc[-63:].values)
            f['beta_to_spy_63d'] = float(cv[0, 1] / (cv[1, 1] + 1e-10))
        else: f['corr_to_spy_63d'], f['beta_to_spy_63d'] = 0.0, 1.0
    else: f['corr_to_spy_63d'], f['beta_to_spy_63d'] = 0.0, 1.0
    f['rel_strength_vs_mean'] = f['ret_21d'] - np.mean(list(all_r21.values())) if all_r21 else 0.0
    f['rank_5d'] = f['rank_21d'] = f['rank_63d'] = 0.5
    return f

def train_predict(fc, spy, idx_end, feat_cols, fwd_days):
    if not HAS_LGBM:
        r = fc.iloc[:idx_end+1].pct_change(min(fwd_days, 21)).iloc[-1]
        return dict(r.sort_values(ascending=False))
    records, start_i = [], max(260, idx_end - TRAIN_W)
    spy_px = spy.iloc[:idx_end+1]
    for i in list(range(start_i, idx_end))[::SAMP_EVERY][:-1]:
        ar = {tk: float(fc[tk].iloc[:i+1].dropna().iloc[-1] / fc[tk].iloc[:i+1].dropna().iloc[-21] - 1)
              for tk in fc.columns if len(fc[tk].iloc[:i+1].dropna()) > 21}
        for tk in fc.columns:
            px = fc[tk].iloc[:i+1].dropna()
            ft = compute_feats(px, spy_px=spy_px.iloc[:i+1], all_r21=ar)
            if not ft: continue
            fi = min(i + fwd_days, len(fc) - 1)
            ft['fwd_ret'] = float(fc[tk].iloc[fi] / fc[tk].iloc[i] - 1)
            records.append({c: ft.get(c, 0.0) for c in feat_cols + ['fwd_ret']})
    if len(records) < 50:
        r = fc.iloc[:idx_end+1].pct_change(21).iloc[-1]
        return dict(r.sort_values(ascending=False))
    df = pd.DataFrame(records)
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
    df[feat_cols] = df[feat_cols].fillna(0.0)
    X = np.nan_to_num(df[feat_cols].values.astype(np.float32))
    y = df['fwd_ret'].rank(pct=True).values.astype(np.float32)
    m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
    m.fit(X, y)
    ar = {tk: float(fc[tk].iloc[:idx_end+1].dropna().iloc[-1] / fc[tk].iloc[:idx_end+1].dropna().iloc[-21] - 1)
          for tk in fc.columns if len(fc[tk].iloc[:idx_end+1].dropna()) > 21}
    scores = {}
    for tk in fc.columns:
        px = fc[tk].iloc[:idx_end+1].dropna()
        ft = compute_feats(px, spy_px=spy.iloc[:idx_end+1], all_r21=ar)
        if ft:
            v = np.nan_to_num(np.array([[ft.get(c, 0.0) for c in feat_cols]], dtype=np.float32))
            scores[tk] = float(m.predict(v)[0])
    return scores

def get_ensemble(fc, spy, vix, idx_end):
    ws = train_predict(fc, spy, idx_end, WEEKLY_F, 5)
    ms = train_predict(fc, spy, idx_end, MONTHLY_F, 21)
    qs = train_predict(fc, spy, idx_end, QUARTERLY_F, 63)
    def ranks(d):
        items = sorted(d.items(), key=lambda x: x[1]); n = len(items)
        return {tk: (i+1)/n for i, (tk, _) in enumerate(items)} if items else {}
    wr, mr, qr = ranks(ws), ranks(ms), ranks(qs)
    tks = set(wr) | set(mr) | set(qr)
    ens = {tk: 0.3*wr.get(tk,.5) + 0.4*mr.get(tk,.5) + 0.3*qr.get(tk,.5) for tk in tks}
    return ens, wr, mr, qr

def run_variant(fc, spy, vix, variant, verbose=False):
    dates, n = fc.index, len(fc)
    si = max(280, TRAIN_W + 30)
    eq_curve, holdings, trades = [], {}, []
    last_ri, n_reb = None, 0
    spy_sh = CAP / float(spy.iloc[si])
    for di in range(si, n):
        today = dates[di]
        # Trailing stop (D)
        if variant == 'D':
            for tk in [tk for tk, p in holdings.items() if p['side'] == 'long']:
                cp = float(fc[tk].iloc[di])
                if np.isnan(cp): continue
                pos = holdings[tk]
                pos['peak_price'] = max(pos.get('peak_price', pos['entry_price']), cp)
                if (pos['peak_price'] - cp) / pos['peak_price'] >= 0.15:
                    pnl = pos['shares'] * (cp - pos['entry_price'])
                    trades.append(dict(ticker=tk, side='long', exit_reason='trailing_stop',
                        entry_date=str(pos['entry_date'].date()), exit_date=str(today.date()),
                        entry_price=pos['entry_price'], exit_price=cp, shares=pos['shares'],
                        pnl=round(pnl, 4), holding_days=di - pos['entry_idx']))
                    del holdings[tk]
        # MTM (guard NaN)
        mtm = 0.0
        for tk, p in holdings.items():
            cp = float(fc[tk].iloc[di])
            if np.isnan(cp): continue
            mtm += p['shares'] * (cp - p['entry_price']) * (1 if p['side']=='long' else -1)
        realized = sum(t['pnl'] for t in trades)
        ceq = CAP + mtm + realized
        eq_curve.append(dict(date=today, equity=ceq, spy_equity=spy_sh*float(spy.iloc[di]), n_holdings=len(holdings)))
        # Rebalance?
        if last_ri is None: do = di >= si + 5
        else: do = di - last_ri >= REBAL_D
        if not do: continue
        last_ri, n_reb = di, n_reb + 1
        ens, wr, mr, qr = get_ensemble(fc, spy, vix, di)
        if not ens: continue
        ranked = sorted(ens.items(), key=lambda x: x[1], reverse=True)
        tl, ts = {}, {}
        if variant in ('A', 'D'):
            for tk, _ in ranked[:2]: tl[tk] = 0.5
        elif variant == 'B':
            wt2 = set(tk for tk, _ in sorted(wr.items(), key=lambda x: x[1], reverse=True)[:2])
            mt2 = set(tk for tk, _ in sorted(mr.items(), key=lambda x: x[1], reverse=True)[:2])
            qt2 = set(tk for tk, _ in sorted(qr.items(), key=lambda x: x[1], reverse=True)[:2])
            cands = {tk: ens[tk] for tk in wt2 | mt2 | qt2 if sum([tk in wt2, tk in mt2, tk in qt2]) >= 2}
            if cands:
                top = sorted(cands.items(), key=lambda x: x[1], reverse=True)[:2]
                for tk, _ in top: tl[tk] = 1.0 / len(top)
        elif variant == 'C':
            tops = [sorted(r.items(), key=lambda x: x[1], reverse=True)[0][0] for r in [wr, mr, qr]]
            if tops[0] == tops[1] == tops[2]: tl[tops[0]] = 1.0
        elif variant == 'E':
            cv = float(vix.iloc[di]) if di < len(vix) else 20.0
            v80 = float(vix.iloc[max(0, di-252):di+1].quantile(0.80))
            if cv <= v80:
                for tk, _ in ranked[:2]: tl[tk] = 0.5
        elif variant == 'F':
            for tk, _ in ranked[:2]: tl[tk] = 0.40
            ts[ranked[-1][0]] = 0.20
        # Close positions not in target
        for tk in [tk for tk in holdings if not ((tk in tl and holdings[tk]['side']=='long') or
                                                  (tk in ts and holdings[tk]['side']=='short'))]:
            pos = holdings.pop(tk); cp = float(fc[tk].iloc[di])
            if np.isnan(cp): cp = pos['entry_price']  # fallback
            pnl = pos['shares'] * ((cp - pos['entry_price']) if pos['side']=='long' else (pos['entry_price'] - cp))
            trades.append(dict(ticker=tk, side=pos['side'], exit_reason='rebalance',
                entry_date=str(pos['entry_date'].date()), exit_date=str(today.date()),
                entry_price=pos['entry_price'], exit_price=cp, shares=pos['shares'],
                pnl=round(pnl, 4), holding_days=di - pos['entry_idx']))
        # Open new
        avail = CAP + sum(t['pnl'] for t in trades)
        for tk, w in list(tl.items()) + list(ts.items()):
            side = 'long' if tk in tl else 'short'
            if tk in holdings or tk not in fc.columns: continue
            cp = float(fc[tk].iloc[di])
            if np.isnan(cp) or cp <= 0: continue
            sh = (avail * w) / cp
            if sh < 0.001: continue
            holdings[tk] = dict(shares=sh, entry_price=cp, entry_date=today, entry_idx=di, side=side, peak_price=cp)
        if verbose and n_reb <= 3:
            fprint(f"  Rebal {n_reb} ({today.date()}): L[{','.join(tl)}] S[{','.join(ts) or 'none'}] eq=${ceq:.2f}")
    # Close remaining
    for tk, pos in list(holdings.items()):
        cp = float(fc[tk].iloc[-1])
        pnl = pos['shares'] * ((cp - pos['entry_price']) if pos['side']=='long' else (pos['entry_price'] - cp))
        trades.append(dict(ticker=tk, side=pos['side'], exit_reason='end_of_backtest',
            entry_date=str(pos['entry_date'].date()), exit_date=str(dates[-1].date()),
            entry_price=pos['entry_price'], exit_price=cp, shares=pos['shares'],
            pnl=round(pnl, 4), holding_days=n - 1 - pos['entry_idx']))
    edf = pd.DataFrame(eq_curve)
    if len(edf) > 0: edf = edf.set_index('date'); edf = edf[~edf.index.duplicated(keep='last')]
    return dict(trades=trades, equity_curve=edf, metrics=calc_metrics(trades, edf, variant),
                variant=variant, n_rebalances=n_reb)

def calc_metrics(trades, edf, vn):
    z = dict(variant=vn, n_trades=0, sharpe=0, sortino=0, pf=0, wr=0, mdd=0,
             total_return=0, cagr=0, total_pnl=0, mean_pnl=0, wins=0, losses=0,
             spy_total_return=0, spy_sharpe=0, alpha_vs_spy=0)
    if not trades: return z
    pnls = [t['pnl'] for t in trades]; n = len(pnls)
    wins = sum(1 for p in pnls if p > 0); tp = sum(pnls)
    if len(edf) > 5:
        dr = edf['equity'].pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)
        sh = float(dr.mean() / (dr.std() + 1e-10) * np.sqrt(252))
        ds = dr[dr < 0]; so = float(dr.mean() / (ds.std() + 1e-10) * np.sqrt(252)) if len(ds) > 1 else sh
    else: sh = so = 0.0
    gw = sum(p for p in pnls if p > 0); gl = abs(sum(p for p in pnls if p < 0))
    pf = gw / (gl + 1e-10)
    mdd = float(((edf['equity'] - edf['equity'].cummax()) / edf['equity'].cummax()).min()) if len(edf) > 0 else 0
    tr = tp / CAP
    yrs = (edf.index[-1] - edf.index[0]).days / 365.25 if len(edf) > 1 else 1
    cagr = (1 + tr) ** (1/yrs) - 1 if yrs > 0 and (1 + tr) > 0 else 0
    sptr = sshr = alpha = 0
    if 'spy_equity' in edf.columns and len(edf) > 5:
        sptr = float(edf['spy_equity'].iloc[-1] / edf['spy_equity'].iloc[0] - 1)
        sd = edf['spy_equity'].pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)
        sshr = float(sd.mean() / (sd.std() + 1e-10) * np.sqrt(252)); alpha = tr - sptr
    return dict(variant=vn, n_trades=n, sharpe=round(sh, 3), sortino=round(so, 3), pf=round(pf, 3),
                wr=round(wins/n*100, 1), mdd=round(mdd*100, 2), total_return=round(tr*100, 2),
                cagr=round(cagr*100, 2), total_pnl=round(tp, 2), mean_pnl=round(np.mean(pnls), 4),
                wins=wins, losses=n-wins, spy_total_return=round(sptr*100, 2),
                spy_sharpe=round(sshr, 3), alpha_vs_spy=round(alpha*100, 2))

def regime_breakdown(trades, spy):
    if not trades: return {}
    rt = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        ps = spy[(spy.index >= pd.Timestamp(t['entry_date'])) & (spy.index <= pd.Timestamp(t['exit_date']))]
        pr = float(ps.iloc[-1] / ps.iloc[0] - 1) if len(ps) >= 2 else 0.0
        rt['green' if pr > 0.005 else ('red' if pr < -0.005 else 'flat')].append(t)
    bd = {}
    for reg, trs in rt.items():
        if not trs: bd[reg] = dict(n=0, sharpe=0, wr=0, mean_pnl=0, total_pnl=0); continue
        p = [t['pnl'] for t in trs]; nn = len(p); w = sum(1 for x in p if x > 0)
        mp, sp = np.mean(p), np.std(p) if nn > 1 else 1.0
        bd[reg] = dict(n=nn, sharpe=round((mp/(sp+1e-10))*np.sqrt(13), 2), wr=round(w/nn*100, 1),
                       mean_pnl=round(mp, 4), total_pnl=round(sum(p), 2))
    return bd

def validate(result, spy):
    m, trades, gates = result['metrics'], result['trades'], {}
    gates['sharpe_gt_1'] = dict(value=m['sharpe'], threshold=1.0, **{'pass': m['sharpe'] > 1.0})
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades]); rng = np.random.RandomState(42)
        pm = np.array([np.mean(rng.choice(pnls, len(pnls), replace=True)) for _ in range(100)])
        pv = float(np.mean(pm <= 0)) if np.mean(pnls) > 0 else 1.0
    else: pv = 1.0
    gates['perm_p_lt_005'] = dict(value=round(pv, 4), threshold=0.05, **{'pass': pv < 0.05})
    gates['wr_gt_40'] = dict(value=m['wr'], threshold=40.0, **{'pass': m['wr'] > 40.0})
    bd = regime_breakdown(trades, spy)
    sg, sr = bd.get('green', {}).get('sharpe', 0), bd.get('red', {}).get('sharpe', 0)
    rs = abs(sg - sr) / max(abs(sg), abs(sr), 0.01)
    gates['regime_balance'] = dict(value=round(rs, 3), threshold=0.50, sharpe_green=sg, sharpe_red=sr,
                                    **{'pass': rs <= 0.50})
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades]); rng = np.random.RandomState(123)
        mc = np.array([np.sum(rng.choice(pnls, len(pnls), replace=True)) for _ in range(5000)])
        cl, cu = float(np.percentile(mc, 2.5)), float(np.percentile(mc, 97.5))
    else: cl = cu = 0
    gates['mc_ci_positive'] = dict(value=round(cl, 2), ci_upper=round(cu, 2), threshold=0.0,
                                    **{'pass': cl > 0})
    np_ = sum(1 for g in gates.values() if g['pass'])
    return dict(gates=gates, n_pass=np_, n_total=len(gates), all_pass=np_==len(gates), regime_breakdown=bd)

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("  MULTI-TIMEFRAME MOMENTUM ENSEMBLE V1")
    fprint("  Hypothesis: Multi-TF agreement filters noise, catches persistent trends")
    fprint("=" * 70)
    fc, spy, vix = download_data()
    oot = len(fc) - max(280, TRAIN_W + 30)
    fprint(f"OOT days: {oot}")
    results, vals = {}, {}
    for v in 'ABCDEF':
        fprint(f"\n{'='*60}\n  VARIANT {v}: {VNAMES[v]}\n{'='*60}")
        ts = time.time()
        results[v] = run_variant(fc, spy, vix, v, verbose=True)
        m = results[v]['metrics']
        fprint(f"  Trades:{m['n_trades']} Sharpe:{m['sharpe']:.3f} Sortino:{m['sortino']:.3f} "
               f"PF:{m['pf']:.3f} WR:{m['wr']:.1f}% MDD:{m['mdd']:.2f}% "
               f"Ret:{m['total_return']:.2f}% CAGR:{m['cagr']:.2f}% Alpha:{m['alpha_vs_spy']:.2f}% "
               f"({time.time()-ts:.1f}s)")
        vals[v] = validate(results[v], spy)
        fprint(f"  5-Gate: {vals[v]['n_pass']}/{vals[v]['n_total']} PASS")
        for gn, g in vals[v]['gates'].items():
            fprint(f"    {gn}: {'PASS' if g['pass'] else 'FAIL'} ({g['value']} vs {g['threshold']})")
    # Summary
    fprint("\n" + "=" * 95 + "\n  COMPARISON SUMMARY\n" + "=" * 95)
    fprint(f"{'Var':<30} {'N':>5} {'Shp':>6} {'Sort':>6} {'PF':>5} {'WR%':>5} {'MDD%':>6} {'Ret%':>7} {'CAGR%':>6} {'Gate':>5}")
    fprint("-" * 95)
    for v in 'ABCDEF':
        m = results[v]['metrics']
        fprint(f"{v}: {VNAMES[v]:<27} {m['n_trades']:>4} {m['sharpe']:>6.3f} {m['sortino']:>6.3f} "
               f"{m['pf']:>5.2f} {m['wr']:>5.1f} {m['mdd']:>6.2f} {m['total_return']:>7.2f} "
               f"{m['cagr']:>6.2f} {vals[v]['n_pass']:>2}/{vals[v]['n_total']}")
    sm = results['A']['metrics']
    fprint(f"{'SPY Buy-Hold':<33} {'':>4} {sm['spy_sharpe']:>6.3f} {'':>6} {'':>5} {'':>5} {'':>6} "
           f"{sm['spy_total_return']:>7.2f}")
    # Regime
    fprint("\n  REGIME BREAKDOWNS")
    for v in 'ABCDEF':
        bd = vals[v]['regime_breakdown']
        fprint(f"  {v} ({VNAMES[v]}):")
        for r in ['green','red','flat']:
            b = bd.get(r, {})
            fprint(f"    {r:5s}: n={b.get('n',0):3d} Shp={b.get('sharpe',0):5.2f} WR={b.get('wr',0):5.1f}% "
                   f"mean=${b.get('mean_pnl',0):.4f} total=${b.get('total_pnl',0):.2f}")
    bv = max('ABCDEF', key=lambda v: results[v]['metrics']['sharpe'])
    fprint(f"\n  BEST: {bv} ({VNAMES[bv]}) Sharpe {results[bv]['metrics']['sharpe']:.3f}")
    ap = any(vals[v]['all_pass'] for v in 'ABCDEF')
    if ap: fprint(f"  VERDICT: Multi-TF ensemble passes all 5 gates!")
    elif results[bv]['metrics']['sharpe'] > 1: fprint(f"  VERDICT: Edge found but fails some gates.")
    else: fprint(f"  VERDICT: No strong edge with this config.")
    elapsed = time.time() - t0
    fprint(f"\nRuntime: {elapsed:.0f}s")
    # Save
    save = dict(timestamp=datetime.now().isoformat(), runtime_s=round(elapsed, 1),
                data_range=f"{fc.index[0].date()} to {fc.index[-1].date()}", n_days=len(fc),
                oot_days=oot, capital=CAP, etfs=ETFS, train_window=TRAIN_W,
                models=dict(weekly=dict(fwd=5, w=0.3), monthly=dict(fwd=21, w=0.4),
                            quarterly=dict(fwd=63, w=0.3)),
                metrics={v: results[v]['metrics'] for v in 'ABCDEF'},
                validations={v: dict(n_pass=vals[v]['n_pass'], n_total=vals[v]['n_total'],
                    all_pass=vals[v]['all_pass'], gates={gn: dict(gv) for gn, gv in vals[v]['gates'].items()})
                    for v in 'ABCDEF'},
                regimes={v: vals[v]['regime_breakdown'] for v in 'ABCDEF'})
    rp = OUTPUT_DIR / "backtest_results.json"
    with open(rp, 'w') as f: json.dump(save, f, indent=2, default=str)
    for v in 'ABCDEF':
        with open(OUTPUT_DIR / f"trades_{v}.json", 'w') as f: json.dump(results[v]['trades'], f, indent=2, default=str)
        results[v]['equity_curve'].to_csv(OUTPUT_DIR / f"equity_{v}.csv")
    fprint(f"Results saved to {rp}")
    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name=f"multi_tf_ens_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_param("capital", CAP); mlflow.log_param("train_window", TRAIN_W)
                mlflow.log_param("etfs", ','.join(ETFS)); mlflow.log_param("best", f"{bv}_{VNAMES[bv]}")
                for v in 'ABCDEF':
                    m = results[v]['metrics']
                    for k in ['sharpe','sortino','pf','wr','mdd','total_return','cagr','n_trades','alpha_vs_spy']:
                        mlflow.log_metric(f"v{v}_{k}", m[k])
                    mlflow.log_metric(f"v{v}_gates", vals[v]['n_pass'])
                mlflow.log_artifact(str(rp))
            fprint("MLflow logged")
        except Exception as e: fprint(f"MLflow failed: {e}")
    fprint("Done.")

if __name__ == '__main__':
    main()
