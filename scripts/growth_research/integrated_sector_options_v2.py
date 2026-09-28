#!/usr/bin/env python3
"""Integrated Sector Options v2 — Production Config combining ALL validated findings.

Merges tonight's research breakthroughs into one definitive strategy:
1. Multi-factor LGBM ranking (Sharpe 3.87 > 3.57 momentum-only) — from multifactor v1
2. VIX>20 entry timing (+0.23 Sharpe) — from day-of-week v1
3. Bi-weekly rebalancing (beats monthly) — from sector rotation v1
4. Confluence gating: min 2 confirming signals (HC #750)
5. Tiered position sizing (scales with equity) — from position scaling v1

Also tests:
  A: All improvements combined (the "full stack")
  B: Without VIX filter (trade all VIX levels)
  C: With concentrated top-2 (fewer positions, less MaxDD)
  D: With adaptive sizing (sqrt scaling)
  E: Conservative (VIX 20-35 only, no extreme vol)
  F: Quality-momentum only (simpler feature set)

$645 starting capital, ATR pricing + 15% haircut.
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
RESULTS_PATH = RESULTS_DIR / 'integrated_sector_options_v2_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

# ==================== MULTI-FACTOR FEATURES ====================
MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
VALUE_COLS = ['dist_sma200','dist_sma50','rsi_14','bb_pct','price_zscore']
BREADTH_COLS = ['rel_str_21d','rel_str_63d']
ALL_COLS = MOM_COLS + QUALITY_COLS + VALUE_COLS + BREADTH_COLS
QM_COLS = MOM_COLS + QUALITY_COLS  # Quality-momentum subset

def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in SECTORS if c in volume.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]

def compute_all_features(px, vol_data, spy_slice):
    """Compute all factor features for a single ticker."""
    if len(px) < 260: return None
    f = {}
    # Momentum
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
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]; dn_days = rets[rets < 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean()+1e-10)) if len(up_days) > 10 else 1.0
    f['dn_capture'] = float(dn_days.iloc[-63:].mean() / (dn_days.mean()+1e-10)) if len(dn_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean()/(vol_data.iloc[-63:].mean()+1e-10)) if vol_data is not None and len(vol_data) >= 63 else 1.0

    # Value
    sma200 = px.rolling(200).mean()
    f['dist_sma200'] = float(px.iloc[-1]/sma200.iloc[-1]-1) if not pd.isna(sma200.iloc[-1]) else 0.0
    sma50 = px.rolling(50).mean()
    f['dist_sma50'] = float(px.iloc[-1]/sma50.iloc[-1]-1) if not pd.isna(sma50.iloc[-1]) else 0.0
    gains = rets.clip(lower=0).rolling(14).mean()
    losses = (-rets.clip(upper=0)).rolling(14).mean()
    rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
    f['rsi_14'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    sma20 = px.rolling(20).mean(); std20 = px.rolling(20).std()
    f['bb_pct'] = float((px.iloc[-1]-sma20.iloc[-1])/(2*std20.iloc[-1])) if not pd.isna(std20.iloc[-1]) and std20.iloc[-1] > 0 else 0.0
    f['price_zscore'] = float((px.iloc[-1]-px.iloc[-252:].mean())/(px.iloc[-252:].std()+1e-10)) if len(px) >= 252 else 0.0

    # Breadth (relative strength vs SPY)
    if len(spy_slice) >= 63:
        rel = px / spy_slice
        f['rel_str_21d'] = float(rel.iloc[-1]/rel.iloc[-21]-1) if len(rel) > 21 else 0.0
        f['rel_str_63d'] = float(rel.iloc[-1]/rel.iloc[-63]-1) if len(rel) > 63 else 0.0
    else:
        f['rel_str_21d'] = 0.0; f['rel_str_63d'] = 0.0

    return f

# ==================== LGBM RANKING ====================
def build_rankings(sc, sv, spy, rebal_dates, feat_cols, name=''):
    fprint(f"  Building {name} rankings ({len(feat_cols)} features)...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            spy_s = spy.iloc[:idx+1]
            feats = compute_all_features(px, vol_d, spy_s)
            if not feats: continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
    df[feat_cols] = df[feat_cols].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[feat_cols].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[feat_cols].values.astype(np.float32))
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

# ==================== CONFLUENCE GATE ====================
def confluence_check(tk, dt, sc, spy, vix):
    """HC #750: min 2 confirming signals. Returns (pass, n_signals, signals_list)."""
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 21: return False, 0, []
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False, 0, []

    # Signal 1: Momentum (21d return > 0)
    ret_21 = float(px.iloc[-1]/px.iloc[-21]-1) if len(px) > 21 else 0
    if ret_21 > 0: signals.append('mom_21d')

    # Signal 2: Trend (price > 50-SMA)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]:
        signals.append('above_sma50')

    # Signal 3: Relative strength vs SPY
    spy_s = spy.iloc[:idx+1]
    if len(spy_s) > 21:
        rel = px / spy_s
        if len(rel) > 21 and float(rel.iloc[-1]/rel.iloc[-21]-1) > 0:
            signals.append('rel_str_pos')

    # Signal 4: RSI not overbought (RSI < 80)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if r < 80: signals.append('rsi_ok')

    # Signal 5: VIX context (VIX > 15 = enough premium)
    cv = float(vix.loc[dt]) if dt in vix.index else 20
    if cv > 15: signals.append('vix_premium')

    return len(signals) >= 2, len(signals), signals

# ==================== OPTIONS PRICING ====================
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

# ==================== SIMULATION ====================
def simulate(name, rankings, sc, sh, sl, spy, vix,
             spread_pct=3.0, dte=30, top_k=3,
             vix_min=None, vix_max=None,
             use_confluence=True, sizing='fixed'):
    fprint(f"\n--- {name} ---")
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    confluence_blocks, confluence_passes = 0, 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])

        # VIX filter
        if vix_min is not None and cv < vix_min: continue
        if vix_max is not None and cv > vix_max: continue

        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]

        # Position sizing
        if sizing == 'fixed':
            max_pos = min(200, equity/3)
        elif sizing == 'sqrt':
            max_pos = min(2000, 200 * np.sqrt(equity/CAP))
        elif sizing == 'tiered':
            if equity < 2000: max_pos = 200
            elif equity < 10000: max_pos = 500
            else: max_pos = 1000
        else:
            max_pos = min(200, equity/3)

        if max_pos < 30: eq_curve.append(equity); continue
        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue

            # Confluence gate (HC #750)
            if use_confluence:
                passes, n_sig, sigs = confluence_check(tk, dt, sc, spy, vix)
                if not passes:
                    confluence_blocks += 1; continue
                confluence_passes += 1

            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt); ei = min(di+dte, len(sc)-1); Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S*(1+spread_pct/100))
            lp = atr_premium(S, K1, dte, av, cv, 'call')*(1+HAIRCUT)
            sp = atr_premium(S, K2, dte, av, cv, 'call')*(1-HAIRCUT)
            val = lp-sp; width = K2-K1
            cost = val*100+SPREAD_COMM; mx_prof = (width-val)*100-SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

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
            trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                           'ticker': tk, 'pnl': round(pnl,2), 'win': pnl>0,
                           'regime': 'bull' if bull else 'bear', 'vix': round(cv,1)})
        eq_curve.append(equity)

    if use_confluence:
        total = confluence_passes + confluence_blocks
        fprint(f"  Confluence: {confluence_passes} passed, {confluence_blocks} blocked "
               f"({confluence_blocks/max(total,1)*100:.1f}% filtered)")
    return trades, equity, eq_curve

# ==================== METRICS & VALIDATION ====================
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
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    # 4-gate validation
    rets = np.array(mr.values.tolist()); gates = 0
    pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0
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

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1), 'sharpe': round(sh,2),
         'sortino': round(so,2), 'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'avg_pnl': round(np.mean(pnls),2), 'final_equity': round(final_eq,2),
         'bull_wr': round(bw,1), 'bear_wr': round(brw,1), 'bull_n': len(bt), 'bear_n': len(brt),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}
    fprint(f"  {name}: {n} trades | WR {wr:.1f}% | Sh {sh:.2f} | Sort {so:.2f} | "
           f"CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | PF {pf:.2f} | ${CAP:.0f}->${final_eq:.0f}")
    fprint(f"  Gates: G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'} => {gates}/4")
    return r

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Integrated Sector Options v2 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"PRODUCTION CONFIG: Multi-factor + VIX timing + confluence + tiered sizing")
    fprint(f"Capital: ${CAP:.0f} | ATR+{HAIRCUT:.0%} haircut | Comm: ${SPREAD_COMM}/spread")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix = download_data()
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)

    # Build two ranking sets
    rank_mf = build_rankings(sc, sv, spy, bd, ALL_COLS, 'multi-factor')
    rank_qm = build_rankings(sc, sv, spy, bd, QM_COLS, 'quality-momentum')

    if not rank_mf: fprint("FATAL: No rankings"); return

    # Variants
    fprint(f"\n=== Simulating {6} Production Variants ===")
    results = []
    configs = [
        # name, rankings, spread%, dte, top_k, vix_min, vix_max, confluence, sizing
        ('A_FullStack',        rank_mf, 3.0, 30, 3, 20, None, True,  'fixed'),
        ('B_NoVIXFilter',      rank_mf, 3.0, 30, 3, None, None, True, 'fixed'),
        ('C_Concentrated_T2',  rank_mf, 3.0, 30, 2, 20, None, True,  'fixed'),
        ('D_SqrtSizing',       rank_mf, 3.0, 30, 3, 20, None, True,  'sqrt'),
        ('E_Conservative',     rank_mf, 3.0, 30, 3, 20, 35, True,   'fixed'),
        ('F_QualMom_Simple',   rank_qm, 3.0, 30, 3, 20, None, True,  'fixed'),
    ]

    for nm, rnk, sp, dt, tk, vmin, vmax, conf, sz in configs:
        tr, eq, cu = simulate(nm, rnk, sc, sh, sl, spy, vix,
                               spread_pct=sp, dte=dt, top_k=tk,
                               vix_min=vmin, vix_max=vmax,
                               use_confluence=conf, sizing=sz)
        r = metrics_validate(tr, eq, cu, nm)
        if r: results.append(r)

    if not results: fprint("No results"); return

    # Compare to prior best
    fprint(f"\n{'='*100}")
    fprint(f"SUMMARY — Integrated Sector Options v2 (Production Config)")
    fprint(f"{'='*100}")
    fprint(f"{'Variant':<25} {'#':>5} {'WR':>6} {'Sh':>6} {'So':>6} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>9} {'G':>4}")
    fprint("-"*100)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} "
               f"{r['sortino']:>6.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>6.2f} ${r['final_equity']:>8.0f} {r['gates']:>3}/4")

    bs = max(results, key=lambda x: x['sharpe'])
    fprint(f"\n=== PRODUCTION RECOMMENDATION ===")
    fprint(f"BEST: {bs['name']} — Sharpe {bs['sharpe']}, CAGR {bs['cagr_pct']}%, MaxDD {bs['maxdd_pct']}%, Gates {bs['gates']}/4")

    # Compare to historical baselines
    fprint(f"\n=== VS HISTORICAL BASELINES ===")
    fprint(f"Sector Rotation v1 F_BiWeekly (momentum only): Sharpe 3.76, CAGR 32.6%")
    fprint(f"Multi-Factor v1 B_MultiFactor:                  Sharpe 3.87, CAGR 32.6%")
    fprint(f"Day-of-Week v1 O_HighVIX:                       Sharpe 4.30, CAGR 87.6%")
    fprint(f"INTEGRATED v2 {bs['name']}:           Sharpe {bs['sharpe']}, CAGR {bs['cagr_pct']}%")

    # Trade log for last 52 weeks
    full_stack = next((r for r in results if r['name']=='A_FullStack'), bs)
    tr_fs, eq_fs, cu_fs = simulate('_log', rank_mf, sc, sh, sl, spy, vix,
                                    vix_min=20, use_confluence=True, sizing='fixed')
    if tr_fs:
        tdf = pd.DataFrame(tr_fs)
        tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
        recent = tdf[tdf['entry_dt'] >= '2025-07-01']
        if len(recent) > 0:
            fprint(f"\n=== LAST 52 WEEKS TRADE LOG ({len(recent)} trades) ===")
            wr_recent = recent['win'].mean()*100
            avg_pnl = recent['pnl'].mean()
            fprint(f"Recent WR: {wr_recent:.1f}% | Recent avg PnL: ${avg_pnl:.2f}")
            # Monthly breakdown
            recent['mo'] = recent['entry_dt'].dt.to_period('M')
            fprint(f"\n{'Month':<10} {'#':>4} {'WR':>6} {'PnL':>8}")
            fprint("-"*30)
            for mo in sorted(recent['mo'].unique()):
                md = recent[recent['mo']==mo]
                fprint(f"{str(mo):<10} {len(md):>4} {md['win'].mean()*100:>5.1f}% ${md['pnl'].sum():>7.2f}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save
    save_data = {'timestamp': t0.isoformat(), 'capital': CAP, 'results': results, 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"Results saved")

    # MLflow
    if MLFLOW_OK:
        try:
            en = 'integrated_sector_options_v2'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"integ_v2_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'pricing':'ATR','haircut':HAIRCUT,'capital':CAP,
                    'features':'multi-factor','vix_filter':'20+','confluence':'HC750','sizing':'fixed'})
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_cagr': r['cagr_pct'],
                                        f'{p}_mdd': r['maxdd_pct'], f'{p}_wr': r['win_rate'],
                                        f'{p}_pf': r['pf'], f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Integrated Sector Options v2\n{'='*80}")

if __name__ == '__main__':
    main()
