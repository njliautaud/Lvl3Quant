#!/usr/bin/env python3
"""Multi-Asset Bull+Bear Combined v1 — Broader Universe Always-Trading.

Combines TWO proven results:
1. Multi-asset momentum (25 ETFs) beat sectors-only (Sharpe 4.70 vs 4.18)
2. Bull+Bear combined generates 73% more equity than bull-only ($48K vs $28K)

Question: Does combining BOTH — broader universe + always-trading — create the best growth strategy?

Variants:
  A: Broad combined (25 ETFs, bull VIX>=20, bear VIX<20)
  B: Broad bull-only (25 ETFs, VIX>=20 only) — control
  C: Sector combined (11 ETFs, bull VIX>=20, bear VIX<20) — v1 baseline
  D: Sector bull-only (11 ETFs, VIX>=20 only) — control
  E: Broad combined + 20d exit (early exit like v1)
  F: Sec+Bonds combined (decorrelation play)
  G: Sec+Commodities combined (inflation hedge)
  H: Random direction control (broad universe)

$645 starting, 20d exit, honest equity-based Sharpe.
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
RESULTS_PATH = RESULTS_DIR / 'multi_asset_bull_bear_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

# Asset universes
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
COMMODITIES = ['GLD','SLV','USO','DBA']
BONDS = ['TLT','HYG','LQD','TIP']
INTERNATIONAL = ['EFA','EEM','VWO']
ALTERNATIVES = ['VNQ','AMLP']
CRYPTO = ['BITO']
ALL_ASSETS = SECTORS + COMMODITIES + BONDS + INTERNATIONAL + ALTERNATIVES + CRYPTO

CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']

def compute_features(px, vol_data=None):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63)>10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:]>0).mean()) if len(monthly)>=12 else 0.5
    dr = r63[r63<0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr)>3 else 0.0
    pk = px.iloc[-252:].cummax(); mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px)>=252 else 0.0
    f['calmar_1y'] = cagr/(abs(mdd)+1e-10)
    up = rets[rets>0]; dn = rets[rets<0]
    f['up_capture'] = float(up.iloc[-63:].mean()/(up.mean()+1e-10)) if len(up)>10 else 1.0
    f['dn_capture'] = float(dn.iloc[-63:].mean()/(dn.mean()+1e-10)) if len(dn)>10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values+1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean()/(vol_data.iloc[-63:].mean()+1e-10)) if vol_data is not None and len(vol_data)>=63 else 1.0
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

def build_rankings(sc, sv, rebal_dates, available):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in available:
            if tk not in sc.columns: continue
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

def simulate(name, rankings, sc, sh, sl, spy, vix,
             use_bull=True, use_bear=True, exit_days=20, random_dir=False):
    sma200 = spy.rolling(200).mean()
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])
    equity, trades, eq_curve = CAP, [], [CAP]
    bull_count, bear_count = 0, 0

    rng = np.random.RandomState(42) if random_dir else None

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

        if random_dir:
            is_bull_mode = rng.random() > 0.5
        else:
            is_bull_mode = cv >= 20

        if is_bull_mode and use_bull:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            picks = [t for t, _ in ranked[:3]]
            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                if not random_dir and not bull_confluence(tk, dt, sc, spy, vix): continue

                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + exit_days, len(sc)-1)
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

                equity += pnl; n_ent += 1; bull_count += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bull', 'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        elif not is_bull_mode and use_bear:
            ranked = sorted(scores.items(), key=lambda x: x[1])
            picks = [t for t, _ in ranked[:3]]
            n_ent = 0
            for tk in picks:
                if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
                if not random_dir and not bear_confluence(tk, dt, sc, spy, vix): continue

                S = float(sc[tk].loc[dt])
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                di = sc.index.get_loc(dt)
                ei = min(di + exit_days, len(sc)-1)
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

                equity += pnl; n_ent += 1; bear_count += 1
                trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                               'ticker': tk, 'side': 'bear', 'pnl': round(pnl,2), 'win': pnl>0,
                               'hold_days': aei-di, 'regime': 'bull' if bull_regime else 'bear',
                               'vix': round(cv,1), 'equity_at_trade': round(equity,2)})

        eq_curve.append(equity)

    return trades, equity, eq_curve, bull_count, bear_count

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
    sh = float(np.mean(rets)/(np.std(rets)+1e-10)*np.sqrt(12))
    dn = rets[rets < 0]
    so = float(np.mean(rets)/(np.std(dn)+1e-10)*np.sqrt(12)) if len(dn) > 1 else 0.0
    return sh, so, rets

def validate(trades, final_eq, eq_curve, name, bull_n, bear_n):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    sh, so, rets = compute_honest_sharpe(trades)
    ny = max(len(rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)

    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    bull_trades = [t for t in trades if t['side']=='bull']
    bear_trades = [t for t in trades if t['side']=='bear']
    bull_wr = sum(1 for t in bull_trades if t['win'])/max(len(bull_trades),1)*100
    bear_wr = sum(1 for t in bear_trades if t['win'])/max(len(bear_trades),1)*100
    bull_pnl = sum(t['pnl'] for t in bull_trades)
    bear_pnl = sum(t['pnl'] for t in bear_trades)

    # Unique tickers
    tickers_used = set(t['ticker'] for t in trades)

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
         'n_unique_tickers': len(tickers_used),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}

    fprint(f"  {name}: {n} trades (B{len(bull_trades)}/S{len(bear_trades)}) [{len(tickers_used)} tickers] | "
           f"WR {wr:.1f}% | Sh {sh:.2f} | Sort {so:.2f} | CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | "
           f"PF {pf:.2f} | ${CAP}->${final_eq:.0f} | Gates {gates}/4")
    fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.0f}%, PnL ${bull_pnl:.0f}")
    fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.0f}%, PnL ${bear_pnl:.0f}")
    fprint(f"    G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'}")

    # Ticker breakdown
    tdf = pd.DataFrame(trades)
    tk_pnl = tdf.groupby('ticker')['pnl'].agg(['sum','count','mean'])
    tk_pnl['wr'] = tdf.groupby('ticker')['win'].mean()*100
    tk_pnl = tk_pnl.sort_values('sum', ascending=False)
    top5 = ', '.join(f"{t} ${tk_pnl.loc[t,'sum']:.0f}({tk_pnl.loc[t,'count']:.0f}t)" for t in tk_pnl.index[:5])
    bot5 = ', '.join(f"{t} ${tk_pnl.loc[t,'sum']:.0f}({tk_pnl.loc[t,'count']:.0f}t)" for t in tk_pnl.index[-5:])
    fprint(f"    Top 5 tickers: {top5}")
    fprint(f"    Bot 5 tickers: {bot5}")
    return r

def main():
    t0 = datetime.now()
    fprint(f"Multi-Asset Bull+Bear v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Combining broader universe (25 ETFs) with always-trading (bull+bear)")
    fprint(f"{'='*80}")

    # Download all data once
    import yfinance as yf
    fprint("\nDownloading all assets...")
    all_tickers = list(set(ALL_ASSETS + ['SPY', '^VIX']))
    raw = yf.download(all_tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    vix_above_20 = (vix >= 20).sum() / len(vix) * 100
    fprint(f"\nVIX stats: {vix_above_20:.1f}% of days VIX>=20")

    # Build universes
    def make_subset(universe_list):
        avail = [c for c in universe_list if c in close.columns and close[c].dropna().shape[0] > 500]
        sc = close[avail].dropna(how='all')
        sh = high[[c for c in avail if c in high.columns]].dropna(how='all')
        sl = low[[c for c in avail if c in low.columns]].dropna(how='all')
        sv = volume[[c for c in avail if c in volume.columns]].dropna(how='all')
        ix = sc.index.intersection(vix.index).intersection(spy.index)
        if len(sh) > 0: ix = ix.intersection(sh.index)
        return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix], avail

    fprint("\nBuilding universe subsets...")
    sc_broad, sh_broad, sl_broad, sv_broad, spy_b, vix_b, avail_broad = make_subset(ALL_ASSETS)
    sc_sect, sh_sect, sl_sect, sv_sect, spy_s, vix_s, avail_sect = make_subset(SECTORS)
    sc_bonds, sh_bonds, sl_bonds, sv_bonds, spy_bo, vix_bo, avail_bonds = make_subset(SECTORS + BONDS)
    sc_comm, sh_comm, sl_comm, sv_comm, spy_co, vix_co, avail_comm = make_subset(SECTORS + COMMODITIES)

    fprint(f"  Broad: {len(avail_broad)} assets, {len(sc_broad)} days")
    fprint(f"  Sectors: {len(avail_sect)} assets, {len(sc_sect)} days")
    fprint(f"  Sec+Bonds: {len(avail_bonds)} assets, {len(sc_bonds)} days")
    fprint(f"  Sec+Commod: {len(avail_comm)} assets, {len(sc_comm)} days")

    # Build rankings for each universe
    bd = pd.DatetimeIndex(sc_broad.index.to_series().resample('2W-FRI').last().dropna().values)

    fprint("\nBuilding rankings for each universe...")
    rank_broad = build_rankings(sc_broad, sv_broad, bd, avail_broad)
    rank_sect = build_rankings(sc_sect, sv_sect, bd, avail_sect)
    rank_bonds = build_rankings(sc_bonds, sv_bonds, bd, avail_bonds)
    rank_comm = build_rankings(sc_comm, sv_comm, bd, avail_comm)

    configs = [
        # (name, rankings, sc, sh, sl, spy, vix, use_bull, use_bear, exit_days, random_dir)
        ('A_Broad_Combined',     rank_broad, sc_broad, sh_broad, sl_broad, spy_b, vix_b, True, True, 20, False),
        ('B_Broad_BullOnly',     rank_broad, sc_broad, sh_broad, sl_broad, spy_b, vix_b, True, False, 20, False),
        ('C_Sector_Combined',    rank_sect,  sc_sect,  sh_sect,  sl_sect,  spy_s, vix_s, True, True, 20, False),
        ('D_Sector_BullOnly',    rank_sect,  sc_sect,  sh_sect,  sl_sect,  spy_s, vix_s, True, False, 20, False),
        ('E_Broad_Comb_30dExit', rank_broad, sc_broad, sh_broad, sl_broad, spy_b, vix_b, True, True, 30, False),
        ('F_SecBonds_Combined',  rank_bonds, sc_bonds, sh_bonds, sl_bonds, spy_bo, vix_bo, True, True, 20, False),
        ('G_SecCommod_Combined', rank_comm,  sc_comm,  sh_comm,  sl_comm,  spy_co, vix_co, True, True, 20, False),
        ('H_Random_Dir',         rank_broad, sc_broad, sh_broad, sl_broad, spy_b, vix_b, True, True, 20, True),
    ]

    results = []
    for nm, rnk, sc_, sh_, sl_, sp_, vx_, ub, ubr, ed, rd in configs:
        fprint(f"\n--- {nm} ---")
        tr, eq, cu, bn, brn = simulate(nm, rnk, sc_, sh_, sl_, sp_, vx_,
                                        use_bull=ub, use_bear=ubr, exit_days=ed, random_dir=rd)
        r = validate(tr, eq, cu, nm, bn, brn)
        if r: results.append(r)

    if not results: fprint("No results"); return

    # Summary
    fprint(f"\n{'='*130}")
    fprint(f"SUMMARY — Multi-Asset Bull+Bear Combined v1")
    fprint(f"{'='*130}")
    fprint(f"{'Variant':<26} {'#':>5} {'B/S':>7} {'Tkrs':>4} {'WR':>6} {'Sh':>7} {'Sort':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>8} {'G':>4}")
    fprint("-"*130)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<26} {r['n_trades']:>5} {r['bull_trades']:>3}/{r['bear_trades']:<3} "
               f"{r['n_unique_tickers']:>4} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} "
               f"${r['final_equity']:>7.0f} {r['gates']:>3}/4")

    # Key comparisons
    broad_comb = next((r for r in results if r['name']=='A_Broad_Combined'), None)
    sector_comb = next((r for r in results if r['name']=='C_Sector_Combined'), None)
    broad_bull = next((r for r in results if r['name']=='B_Broad_BullOnly'), None)
    random_dir = next((r for r in results if r['name']=='H_Random_Dir'), None)

    if broad_comb and sector_comb:
        fprint(f"\n=== BROAD vs SECTOR (both combined) ===")
        eq_diff = (broad_comb['final_equity'] - sector_comb['final_equity']) / sector_comb['final_equity'] * 100
        sh_diff = broad_comb['sharpe'] - sector_comb['sharpe']
        fprint(f"Broad:  Sh {broad_comb['sharpe']:.2f}, ${broad_comb['final_equity']:.0f}, {broad_comb['n_unique_tickers']} tickers")
        fprint(f"Sector: Sh {sector_comb['sharpe']:.2f}, ${sector_comb['final_equity']:.0f}, {sector_comb['n_unique_tickers']} tickers")
        fprint(f"Diff: Sharpe {sh_diff:+.2f}, Equity {eq_diff:+.0f}%")

    if broad_comb and broad_bull:
        fprint(f"\n=== COMBINED vs BULL-ONLY (both broad) ===")
        eq_diff = (broad_comb['final_equity'] - broad_bull['final_equity']) / broad_bull['final_equity'] * 100
        fprint(f"Combined: Sh {broad_comb['sharpe']:.2f}, ${broad_comb['final_equity']:.0f}")
        fprint(f"BullOnly: Sh {broad_bull['sharpe']:.2f}, ${broad_bull['final_equity']:.0f}")
        fprint(f"Equity diff: {eq_diff:+.0f}%")

    if broad_comb and random_dir:
        fprint(f"\n=== VIX-DIRECTED vs RANDOM (both broad) ===")
        fprint(f"VIX-based: Sh {broad_comb['sharpe']:.2f}, ${broad_comb['final_equity']:.0f}")
        fprint(f"Random:    Sh {random_dir['sharpe']:.2f}, ${random_dir['final_equity']:.0f}")
        if random_dir['sharpe'] >= broad_comb['sharpe'] * 0.8:
            fprint(f"WARNING: Random is within 20% of VIX-based — direction choice adds marginal value")
        else:
            fprint(f"VIX direction adds {(broad_comb['sharpe']/max(random_dir['sharpe'],0.01)-1)*100:.0f}% Sharpe")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'capital': CAP,
                 'results': results, 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'multi_asset_bull_bear_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"mabb_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'strategy': 'multi_asset_bull_bear', 'version': 'v1'})
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_mdd': r['maxdd_pct'],
                                        f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Multi-Asset Bull+Bear v1\n{'='*80}")

if __name__ == '__main__':
    main()
