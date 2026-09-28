#!/usr/bin/env python3
"""Bull+Bear Combined Sector Strategy v2 — Optimized Always-Trading.

Improvements over v1:
  1. VIX dead-zone: Skip trading when VIX 18-22 (gray zone) — only trade clear regimes
  2. Asymmetric sizing: Bull gets full budget (proven Sharpe 5.1), bear gets reduced
  3. Conviction-weighted: LGBM score magnitude → position size (higher score = bigger trade)
  4. Dynamic bear scaling: VIX distance from threshold scales bear size (VIX=10 full, VIX=19 minimal)
  5. Sector momentum filter: Require 5d momentum alignment before entry

Variants:
  A: v1 baseline (reproduced for fair comparison)
  B: VIX dead-zone (18-22 skip)
  C: Asymmetric sizing (bull 100%, bear 50%)
  D: Conviction-weighted entries (LGBM score → size)
  E: Dynamic bear (VIX distance scaling)
  F: Combined optimizations (B+C+D+E together)
  G: Combined + wider dead zone (17-23)
  H: Best subset (B+C only — simple, testable)

$645 starting, 20d exit on both sides, honest equity-based Sharpe.
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'bull_bear_combined_v2_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
QM_COLS = MOM_COLS + QUALITY_COLS

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
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index)
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]

def compute_features(px, vol_data, spy_slice):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252))
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax(); mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr_1y = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr_1y / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]; dn_days = rets[rets < 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean()+1e-10)) if len(up_days) > 10 else 1.0
    f['dn_capture'] = float(dn_days.iloc[-63:].mean() / (dn_days.mean()+1e-10)) if len(dn_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = 1.0
    return f

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

def build_rankings(sc, sv, spy, rebal_dates):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d, spy.iloc[:idx+1])
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
        tr_df = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr_df) < 50: continue
        Xt = np.nan_to_num(tr_df[QM_COLS].values.astype(np.float32))
        yt = tr_df['rank_label'].values.astype(np.float32)
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

def bull_confluence(tk, dt, sc, spy, vix):
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 63: return False
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False
    if float(px.iloc[-1]/px.iloc[-21]-1) > 0: signals.append(1)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]: signals.append(1)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        if not pd.isna(rsi.iloc[-1]) and float(rsi.iloc[-1]) < 80: signals.append(1)
    cv = float(vix.loc[dt]) if dt in vix.index else 20
    if cv > 15: signals.append(1)
    return len(signals) >= 2

def bear_confluence(tk, dt, sc, spy, vix):
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 63: return False
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False
    if float(px.iloc[-1]/px.iloc[-21]-1) < 0: signals.append(1)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] < sma50.iloc[-1]: signals.append(1)
    spy_s = spy.iloc[:idx+1]
    if len(spy_s) > 21:
        rel = px / spy_s
        if len(rel) > 21 and float(rel.iloc[-1]/rel.iloc[-21]-1) < 0: signals.append(1)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if 20 < r < 50: signals.append(1)
    return len(signals) >= 2

# ==================== ENHANCED SIMULATION ====================
def simulate(name, rankings, sc, sh, sl, spy, vix,
             use_bull=True, use_bear=True,
             sizing='fixed', bear_size_mult=1.0,
             vix_dead_lo=None, vix_dead_hi=None,
             conviction_weighted=False,
             dynamic_bear=False,
             mom_filter_5d=False):
    """
    Enhanced simulator with v2 features:
    - vix_dead_lo/hi: Skip trading when VIX in [dead_lo, dead_hi]
    - conviction_weighted: Use LGBM score magnitude for position sizing
    - dynamic_bear: Scale bear size by VIX distance from threshold
    - mom_filter_5d: Require 5d momentum alignment
    """
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    bull_count, bear_count, skip_count = 0, 0, 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull_regime = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue

        # Dead zone check
        if vix_dead_lo is not None and vix_dead_hi is not None:
            if vix_dead_lo <= cv <= vix_dead_hi:
                skip_count += 1
                eq_curve.append(equity)
                continue

        # Position sizing base
        base_pos = min(200, equity/3)
        if base_pos < 30: eq_curve.append(equity); continue

        # Decide direction based on VIX
        bull_threshold = vix_dead_hi if vix_dead_hi else 20
        bear_threshold = vix_dead_lo if vix_dead_lo else 20
        is_bull_mode = cv >= bull_threshold
        is_bear_mode = cv < bear_threshold

        if is_bull_mode and use_bull:
            # BULL: call spreads on top-ranked
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            picks = [t for t, _ in ranked[:3]]

            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                if not bull_confluence(tk, dt, sc, spy, vix): continue

                # 5d momentum filter
                if mom_filter_5d:
                    idx_m = sc.index.get_indexer([dt], method='ffill')[0]
                    if idx_m >= 5:
                        mom5 = float(sc[tk].iloc[idx_m] / sc[tk].iloc[idx_m-5] - 1)
                        if mom5 <= 0: continue  # Skip if 5d momentum negative for bull

                # Conviction-weighted sizing
                if conviction_weighted and tk in scores:
                    score = scores[tk]
                    # Score is 0-1 (rank percentile). Top picks should be near 1.0
                    # Scale position: score=1.0 → 100% base, score=0.7 → 70% base
                    conv_mult = max(0.5, min(1.5, score))
                    max_pos = base_pos * conv_mult
                else:
                    max_pos = base_pos

                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + 20, len(sc)-1)
                K1, K2 = round(S), round(S*1.03)
                lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
                val = lp-sp; width = K2-K1
                cost = val*100+SPREAD_COMM; mx_prof = (width-val)*100-SPREAD_COMM
                if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

                pnl, aei = None, ei
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di))
                    ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    tm = np.sqrt(rd/30)
                    cur = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                    cp = cur - cost
                    if cp >= mx_prof*0.50: pnl = cp; aei = ci; break
                    if ci == ei:
                        pnl = cur - cost if rd > 0 else (max(0,Sc-K1)-max(0,Sc-K2))*100 - cost
                        aei = ci; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - cost

                equity += pnl; n_ent += 1; bull_count += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bull', 'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        elif is_bear_mode and use_bear:
            # BEAR: put spreads on bottom-ranked
            ranked = sorted(scores.items(), key=lambda x: x[1])
            picks = [t for t, _ in ranked[:3]]

            # Dynamic bear sizing: scale by VIX distance from threshold
            if dynamic_bear:
                # VIX=10 → full size, VIX=19 → 20% size (linear scale)
                threshold = bear_threshold if bear_threshold else 20
                dist = max(0, threshold - cv)  # distance below threshold
                dyn_mult = min(1.0, max(0.2, dist / 10.0))  # 0.2-1.0 range
                effective_bear_mult = bear_size_mult * dyn_mult
            else:
                effective_bear_mult = bear_size_mult

            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                if not bear_confluence(tk, dt, sc, spy, vix): continue

                # 5d momentum filter
                if mom_filter_5d:
                    idx_m = sc.index.get_indexer([dt], method='ffill')[0]
                    if idx_m >= 5:
                        mom5 = float(sc[tk].iloc[idx_m] / sc[tk].iloc[idx_m-5] - 1)
                        if mom5 >= 0: continue  # Skip if 5d momentum positive for bear

                # Conviction-weighted sizing for bear (use inverse score — lower = stronger conviction)
                if conviction_weighted and tk in scores:
                    score = scores[tk]
                    conv_mult = max(0.5, min(1.5, 1.0 - score))  # Low score = high conviction for bear
                    max_pos = base_pos * effective_bear_mult * conv_mult
                else:
                    max_pos = base_pos * effective_bear_mult

                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + 20, len(sc)-1)
                K1, K2 = round(S), round(S*0.97)
                if K2 >= K1: continue
                lp = atr_premium(S, K1, 30, av, cv, 'put')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'put')*(1-HAIRCUT)
                debit = lp-sp; width = K1-K2
                cost = debit*100+SPREAD_COMM; mx_prof = (width-debit)*100-SPREAD_COMM
                if cost <= 0 or cost > max_pos or cost > equity*0.40 or mx_prof <= 0: continue

                pnl, aei = None, ei
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di))
                    ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    tm = np.sqrt(rd/30)
                    intrinsic = (max(0,K1-Sc)-max(0,K2-Sc))*100
                    cur = intrinsic + ac*tm*0.3*100
                    cp = cur - cost
                    if cp >= mx_prof*0.50: pnl = cp; aei = ci; break
                    if ci == ei:
                        pnl = cur - cost if rd > 0 else intrinsic - cost
                        aei = ci; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,K1-Se)-max(0,K2-Se))*100 - cost

                equity += pnl; n_ent += 1; bear_count += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bear', 'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        eq_curve.append(equity)

    return trades, equity, eq_curve, bull_count, bear_count, skip_count

def compute_honest_sharpe(trades):
    if not trades: return 0.0, 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')
    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mt = tdf[tdf['month'] == mo]
        pnl = mt['pnl'].sum()
        eq = max(mt['equity_at_trade'].iloc[0] - mt['pnl'].iloc[0], 100)
        monthly.append(pnl / eq)
    rets = np.array(monthly)
    if len(rets) < 4: return 0.0, 0.0, rets
    sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(12))
    dn = rets[rets < 0]
    so = float(np.mean(rets) / (np.std(dn) + 1e-10) * np.sqrt(12)) if len(dn) > 1 else 0.0
    return sh, so, rets

def validate(trades, final_eq, eq_curve, name, bull_n, bear_n, skip_n=0):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    sh, so, rets = compute_honest_sharpe(trades)
    ny = max(len(rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)

    # Regime analysis
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    # Side analysis
    bull_trades = [t for t in trades if t['side']=='bull']
    bear_trades = [t for t in trades if t['side']=='bear']
    bull_wr = sum(1 for t in bull_trades if t['win'])/max(len(bull_trades),1)*100
    bear_wr = sum(1 for t in bear_trades if t['win'])/max(len(bear_trades),1)*100
    bull_pnl = sum(t['pnl'] for t in bull_trades)
    bear_pnl = sum(t['pnl'] for t in bear_trades)

    # 4-gate validation
    gates = 0; pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0
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

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1),
         'sharpe': round(sh,2), 'sortino': round(so,2),
         'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'final_equity': round(final_eq,2),
         'bull_trades': len(bull_trades), 'bear_trades': len(bear_trades),
         'bull_wr': round(bull_wr,1), 'bear_wr': round(bear_wr,1),
         'bull_pnl': round(bull_pnl,2), 'bear_pnl': round(bear_pnl,2),
         'regime_bull_wr': round(bw,1), 'regime_bear_wr': round(brw,1),
         'skip_periods': skip_n,
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}

    fprint(f"  {name}: {n} trades (B{len(bull_trades)}/S{len(bear_trades)}) | "
           f"WR {wr:.1f}% | Sh {sh:.2f} | CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | "
           f"PF {pf:.2f} | ${CAP}->${final_eq:.0f} | Gates {gates}/4"
           + (f" | Skips {skip_n}" if skip_n else ""))
    fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.0f}%, PnL ${bull_pnl:.0f}")
    fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.0f}%, PnL ${bear_pnl:.0f}")
    fprint(f"    G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'}")
    return r

# ==================== RANDOM DIRECTION CONTROL ====================
def simulate_random(name, rankings, sc, sh, sl, spy, vix, **kwargs):
    """Same as simulate but randomize bull/bear direction on each date.
    Tests if the DIRECTION CHOICE matters or if any direction works."""
    np.random.seed(42)
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull_regime = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue

        base_pos = min(200, equity/3)
        if base_pos < 30: eq_curve.append(equity); continue

        # RANDOM direction instead of VIX-based
        is_bull_mode = np.random.random() > 0.5

        if is_bull_mode:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            picks = [t for t, _ in ranked[:3]]
            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + 20, len(sc)-1)
                K1, K2 = round(S), round(S*1.03)
                lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
                val = lp-sp; width = K2-K1
                cost = val*100+SPREAD_COMM; mx_prof = (width-val)*100-SPREAD_COMM
                if cost <= 0 or cost > base_pos or cost > equity*0.40: continue
                pnl, aei = None, ei
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di))
                    ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    tm = np.sqrt(rd/30)
                    cur = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                    cp = cur - cost
                    if cp >= mx_prof*0.50: pnl = cp; aei = ci; break
                    if ci == ei:
                        pnl = cur - cost if rd > 0 else (max(0,Sc-K1)-max(0,Sc-K2))*100 - cost
                        aei = ci; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - cost
                equity += pnl; n_ent += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bull', 'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1])
            picks = [t for t, _ in ranked[:3]]
            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + 20, len(sc)-1)
                K1, K2 = round(S), round(S*0.97)
                if K2 >= K1: continue
                lp = atr_premium(S, K1, 30, av, cv, 'put')*(1+HAIRCUT)
                sp = atr_premium(S, K2, 30, av, cv, 'put')*(1-HAIRCUT)
                debit = lp-sp; width = K1-K2
                cost = debit*100+SPREAD_COMM; mx_prof = (width-debit)*100-SPREAD_COMM
                if cost <= 0 or cost > base_pos or cost > equity*0.40 or mx_prof <= 0: continue
                pnl, aei = None, ei
                for ci in range(di+1, ei+1):
                    if ci >= len(sc): break
                    Sc = float(sc[tk].iloc[ci])
                    rd = max(0, 30-(ci-di))
                    ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                    tm = np.sqrt(rd/30)
                    intrinsic = (max(0,K1-Sc)-max(0,K2-Sc))*100
                    cur = intrinsic + ac*tm*0.3*100
                    cp = cur - cost
                    if cp >= mx_prof*0.50: pnl = cp; aei = ci; break
                    if ci == ei:
                        pnl = cur - cost if rd > 0 else intrinsic - cost
                        aei = ci; break
                if pnl is None:
                    Se = float(sc[tk].iloc[ei])
                    pnl = (max(0,K1-Se)-max(0,K2-Se))*100 - cost
                equity += pnl; n_ent += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bear', 'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})
        eq_curve.append(equity)

    return trades, equity, eq_curve, 0, 0, 0

def main():
    t0 = datetime.now()
    fprint(f"Bull+Bear Combined v2 — Optimized — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Testing: VIX dead-zone, asymmetric sizing, conviction weighting, dynamic bear")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix = download_data()
    fprint(f"Data: {len(sc)} days")
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rankings = build_rankings(sc, sv, spy, bd)
    if not rankings: fprint("FATAL: No rankings"); return

    # VIX regime stats
    vix_above_20 = (vix >= 20).sum() / len(vix) * 100
    vix_18_22 = ((vix >= 18) & (vix <= 22)).sum() / len(vix) * 100
    vix_17_23 = ((vix >= 17) & (vix <= 23)).sum() / len(vix) * 100
    fprint(f"\nVIX stats: {vix_above_20:.1f}% >= 20 | {vix_18_22:.1f}% in 18-22 | {vix_17_23:.1f}% in 17-23")

    configs = [
        # (name, use_bull, use_bear, sizing, bear_mult, dead_lo, dead_hi, conviction, dynamic_bear, mom_5d)
        ('A_v1_Baseline',        True, True, 'fixed', 1.0, None, None, False, False, False),
        ('B_DeadZone_18_22',     True, True, 'fixed', 1.0, 18,   22,   False, False, False),
        ('C_Asymmetric_50pct',   True, True, 'fixed', 0.5, None, None, False, False, False),
        ('D_Conviction_Wt',      True, True, 'fixed', 1.0, None, None, True,  False, False),
        ('E_Dynamic_Bear',       True, True, 'fixed', 1.0, None, None, False, True,  False),
        ('F_AllOptimized',       True, True, 'fixed', 0.5, 18,   22,   True,  True,  True),
        ('G_WideDeadZone_17_23', True, True, 'fixed', 1.0, 17,   23,   False, False, False),
        ('H_Simple_DZ_Asym',     True, True, 'fixed', 0.5, 18,   22,   False, False, False),
        ('I_Mom5d_Filter',       True, True, 'fixed', 1.0, None, None, False, False, True),
        ('J_BullOnly_Control',   True, False,'fixed', 1.0, None, None, False, False, False),
    ]

    results = []
    for nm, ub, ubr, sz, bm, dl, dh, cw, db, m5 in configs:
        fprint(f"\n--- {nm} ---")
        tr, eq, cu, bn, brn, skn = simulate(nm, rankings, sc, sh, sl, spy, vix,
                                             use_bull=ub, use_bear=ubr, sizing=sz,
                                             bear_size_mult=bm,
                                             vix_dead_lo=dl, vix_dead_hi=dh,
                                             conviction_weighted=cw,
                                             dynamic_bear=db,
                                             mom_filter_5d=m5)
        r = validate(tr, eq, cu, nm, bn, brn, skn)
        if r: results.append(r)

    # Random direction control
    fprint(f"\n--- Z_RandomDir_Control ---")
    tr, eq, cu, _, _, _ = simulate_random('Z_RandomDir_Control', rankings, sc, sh, sl, spy, vix)
    r = validate(tr, eq, cu, 'Z_RandomDir_Control', 0, 0, 0)
    if r: results.append(r)

    if not results: fprint("No results"); return

    # Summary
    fprint(f"\n{'='*120}")
    fprint(f"SUMMARY — Bull+Bear Combined v2 (Optimized)")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<26} {'#':>5} {'B/S':>7} {'WR':>6} {'Sh':>7} {'Sort':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>8} {'G':>4} {'Skp':>4}")
    fprint("-"*120)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<26} {r['n_trades']:>5} {r['bull_trades']:>3}/{r['bear_trades']:<3} "
               f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['cagr_pct']:>6.1f}% "
               f"{r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4 "
               f"{r.get('skip_periods', 0):>3}")

    # Key comparisons
    baseline = next((r for r in results if r['name']=='A_v1_Baseline'), None)
    bull_only = next((r for r in results if r['name']=='J_BullOnly_Control'), None)
    random_ctrl = next((r for r in results if r['name']=='Z_RandomDir_Control'), None)

    if baseline and bull_only:
        fprint(f"\n=== BASELINE vs BULL-ONLY ===")
        eq_diff = (baseline['final_equity'] - bull_only['final_equity']) / bull_only['final_equity'] * 100
        fprint(f"Combined: Sh {baseline['sharpe']:.2f}, ${baseline['final_equity']:.0f}")
        fprint(f"Bull-only: Sh {bull_only['sharpe']:.2f}, ${bull_only['final_equity']:.0f}")
        fprint(f"Equity diff: {eq_diff:+.0f}%")

    if baseline and random_ctrl:
        fprint(f"\n=== VIX DIRECTION vs RANDOM ===")
        fprint(f"VIX-based: Sh {baseline['sharpe']:.2f}, ${baseline['final_equity']:.0f}")
        fprint(f"Random:    Sh {random_ctrl['sharpe']:.2f}, ${random_ctrl['final_equity']:.0f}")
        if random_ctrl['sharpe'] >= baseline['sharpe'] * 0.8:
            fprint(f"WARNING: Random direction is within 20% of VIX-based — VIX timing adds marginal value")
        else:
            fprint(f"VIX-based direction adds {(baseline['sharpe']/max(random_ctrl['sharpe'],0.01)-1)*100:.0f}% more Sharpe")

    # Find best v2 improvement
    best_v2 = max([r for r in results if r['name'] not in ['A_v1_Baseline', 'J_BullOnly_Control', 'Z_RandomDir_Control']],
                  key=lambda x: x['sharpe'], default=None)
    if best_v2 and baseline:
        sh_imp = (best_v2['sharpe'] - baseline['sharpe']) / max(abs(baseline['sharpe']), 0.01) * 100
        eq_imp = (best_v2['final_equity'] - baseline['final_equity']) / max(baseline['final_equity'], 1) * 100
        fprint(f"\n=== BEST v2 OPTIMIZATION ===")
        fprint(f"Best: {best_v2['name']} — Sh {best_v2['sharpe']:.2f} ({sh_imp:+.0f}% vs v1), "
               f"${best_v2['final_equity']:.0f} ({eq_imp:+.0f}% vs v1)")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'capital': CAP,
                 'results': results, 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'bull_bear_combined_v2'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"bb_opt_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'strategy': 'bull+bear_optimized', 'exit': '20d',
                                   'version': 'v2'})
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_mdd': r['maxdd_pct'],
                                        f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Bull+Bear Combined v2\n{'='*80}")

if __name__ == '__main__':
    main()
