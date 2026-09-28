#!/usr/bin/env python3
"""Sector + PEAD Combined Portfolio v1 — Two Complementary Growth Engines.

Combines TWO validated strategies on a SINGLE $645 equity curve:
  1. Sector Bull/Bear Spreads (biweekly, VIX-regime-based) — Sharpe 3.10, 4/4 gates
  2. PEAD Call Spreads (quarterly, post-earnings drift) — Sharpe 1.38, 4/4 gates

Why combine:
  - Different TIMING: Sector fires every 2 weeks, PEAD fires 4x/year during earnings season
  - Different STOCKS: Sector = 11 ETFs, PEAD = 30 individual mega-caps
  - Different EDGE: Sector = VIX timing + momentum ranking, PEAD = earnings gap + drift
  - Should diversify returns and reduce drawdowns

Variants:
  A: Full combined (sector bull+bear + PEAD gap>3%)
  B: Full combined (sector + PEAD gap>5%, higher quality)
  C: Sector only control (reproduce prior v1 baseline)
  D: PEAD only control
  E: Combined with PEAD half-size (reduce PEAD MDD contribution)
  F: Combined with tiered sizing

$645 starting, honest equity-based Sharpe.
"""
import json, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import norm
import lightgbm as lgb

BASE = Path('/home/jupiter/Lvl3Quant')
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'sector_pead_combined_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
PEAD_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','AMD','INTC',
    'BA','DIS','SBUX','HD','LOW','MCD','NKE','COST','WMT',
    'JPM','GS','BAC','MS','JNJ','PG','KO','UNH','ABBV','CRM','NOW',
]
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15
EARNINGS_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}

MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
QM_COLS = MOM_COLS + QUALITY_COLS

# ==================== DATA ====================
def download_sector_data():
    import yfinance as yf
    fprint("Downloading sector data...")
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

def download_pead_data():
    import yfinance as yf
    cache = BASE / 'output' / 'pead_options_v1' / 'prices_cache.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
        fprint(f"  PEAD prices from cache: {len(df)} rows")
        return df
    fprint(f"  Downloading PEAD tickers...")
    all_data = []
    for tk in PEAD_TICKERS + ['SPY']:
        try:
            d = yf.download(tk, start='2016-01-01', end='2026-07-26', progress=False, auto_adjust=True)
            if len(d) > 0:
                d = d.reset_index()
                d.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in d.columns]
                d['ticker'] = tk
                all_data.append(d[['date','ticker','open','high','low','close','volume']])
        except: pass
        time.sleep(0.1)
    df = pd.concat(all_data, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

# ==================== SECTOR FUNCTIONS (from bull_bear_combined_v1) ====================
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
    else: f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
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

def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == 'call': return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

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

# ==================== PEAD FUNCTIONS ====================
def detect_earnings(prices_df, ticker):
    tk_data = prices_df[prices_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
    if len(tk_data) < 60: return []
    events = []
    for i in range(1, len(tk_data)):
        row = tk_data.iloc[i]; prev = tk_data.iloc[i-1]
        if row['date'].month not in EARNINGS_MONTHS: continue
        gap = (row['open'] - prev['close']) / prev['close']
        if abs(gap) < 0.02: continue
        if any((row['date'] - e['date']).days < 60 for e in events): continue
        events.append({'date': row['date'], 'ticker': ticker, 'gap_pct': gap*100,
                       'open_price': row['open'], 'idx': i})
    return events

def value_spread(S, K1, K2, T, vol):
    if T <= 0: return (max(0, S-K1) - max(0, S-K2)) * 100
    return (bs_price(S, K1, T, vol, opt='call') - bs_price(S, K2, T, vol, opt='call')) * 100

# ==================== COMBINED SIMULATION ====================
def simulate(name, rankings, sc, sh, sl, spy, vix, pead_events, pead_prices,
             use_sector=True, use_pead=True, gap_threshold=3.0,
             sizing='fixed', pead_size_mult=1.0):

    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    sector_count, pead_count = 0, 0

    # Build a timeline of all events
    all_dates = set()
    if use_sector:
        all_dates.update(rankings.keys())
    if use_pead:
        all_dates.update(e['date'] for e in pead_events if e['gap_pct'] >= gap_threshold)

    for dt in sorted(all_dates):
        if dt not in spy.index: continue

        # Position sizing
        if sizing == 'tiered':
            if equity < 2000: base_pos = 200
            elif equity < 10000: base_pos = 500
            else: base_pos = 1000
        else:
            base_pos = min(200, equity / 3)
        if base_pos < 30: eq_curve.append(equity); continue

        cv = float(vix.loc[dt]) if dt in vix.index else 18
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull_regime = sv_val >= sm

        # === SECTOR TRADES (biweekly rebalance dates) ===
        if use_sector and dt in rankings:
            scores = rankings[dt]
            if scores:
                is_bull_mode = cv >= 20

                if is_bull_mode:
                    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    for tk, _ in ranked:
                        if tk not in sc.columns or tk not in atr_d: continue
                        if not bull_confluence(tk, dt, sc, spy, vix): continue
                        S = float(sc[tk].loc[dt])
                        av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                        di = sc.index.get_loc(dt); ei = min(di+20, len(sc)-1)
                        K1, K2 = round(S), round(S*1.03)
                        lp = atr_premium(S, K1, 30, av, cv, 'call')*(1+HAIRCUT)
                        sp = atr_premium(S, K2, 30, av, cv, 'call')*(1-HAIRCUT)
                        val = lp-sp; cost = val*100+SPREAD_COMM; mx = (K2-K1-val)*100-SPREAD_COMM
                        if cost <= 0 or cost > base_pos or cost > equity*0.40: continue
                        pnl = None
                        for ci in range(di+1, ei+1):
                            if ci >= len(sc): break
                            Sc = float(sc[tk].iloc[ci]); rd = max(0,30-(ci-di))
                            ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                            cur = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*np.sqrt(rd/30)*0.3*100
                            if cur-cost >= mx*0.50: pnl = cur-cost; ei = ci; break
                            if ci == ei: pnl = cur-cost if rd>0 else (max(0,Sc-K1)-max(0,Sc-K2))*100-cost; break
                        if pnl is None:
                            Se = float(sc[tk].iloc[ei]); pnl = (max(0,Se-K1)-max(0,Se-K2))*100-cost
                        equity += pnl; sector_count += 1
                        trades.append({'entry': str(dt.date()), 'ticker': tk, 'strategy': 'sector_bull',
                                       'pnl': round(pnl,2), 'win': pnl>0, 'regime': 'bull' if bull_regime else 'bear',
                                       'equity_at_trade': round(equity,2)})
                else:
                    ranked = sorted(scores.items(), key=lambda x: x[1])[:3]
                    for tk, _ in ranked:
                        if tk not in sc.columns or tk not in atr_d: continue
                        if not bear_confluence(tk, dt, sc, spy, vix): continue
                        S = float(sc[tk].loc[dt])
                        av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
                        di = sc.index.get_loc(dt); ei = min(di+20, len(sc)-1)
                        K1, K2 = round(S), round(S*0.97)
                        if K2 >= K1: continue
                        lp = atr_premium(S, K1, 30, av, cv, 'put')*(1+HAIRCUT)
                        sp = atr_premium(S, K2, 30, av, cv, 'put')*(1-HAIRCUT)
                        debit = lp-sp; cost = debit*100+SPREAD_COMM; mx = (K1-K2-debit)*100-SPREAD_COMM
                        if cost <= 0 or cost > base_pos or cost > equity*0.40 or mx <= 0: continue
                        pnl = None
                        for ci in range(di+1, ei+1):
                            if ci >= len(sc): break
                            Sc = float(sc[tk].iloc[ci]); rd = max(0,30-(ci-di))
                            ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                            intr = (max(0,K1-Sc)-max(0,K2-Sc))*100
                            cur = intr + ac*np.sqrt(rd/30)*0.3*100
                            if cur-cost >= mx*0.50: pnl = cur-cost; ei = ci; break
                            if ci == ei: pnl = cur-cost if rd>0 else intr-cost; break
                        if pnl is None:
                            Se = float(sc[tk].iloc[ei]); pnl = (max(0,K1-Se)-max(0,K2-Se))*100-cost
                        equity += pnl; sector_count += 1
                        trades.append({'entry': str(dt.date()), 'ticker': tk, 'strategy': 'sector_bear',
                                       'pnl': round(pnl,2), 'win': pnl>0, 'regime': 'bull' if bull_regime else 'bear',
                                       'equity_at_trade': round(equity,2)})

        # === PEAD TRADES (earnings gap days) ===
        if use_pead:
            day_events = [e for e in pead_events if e['date'] == dt and e['gap_pct'] >= gap_threshold]
            for event in day_events[:2]:  # Max 2 PEAD trades per day
                tk = event['ticker']
                tk_data = pead_prices[pead_prices['ticker'] == tk].sort_values('date').reset_index(drop=True)
                idx = event['idx']
                if idx + 40 >= len(tk_data): continue

                pead_pos = base_pos * pead_size_mult
                if pead_pos < 50: continue

                S = event['open_price']
                pre_vol = tk_data['close'].iloc[max(0,idx-21):idx].pct_change().std() * np.sqrt(252)
                post_vol = pre_vol * 0.65

                K1 = round(S, 2); K2 = round(S * 1.05, 2)
                lp = bs_price(S, K1, 45/365, post_vol, opt='call') * (1+HAIRCUT)
                sp = bs_price(S, K2, 45/365, post_vol, opt='call') * (1-HAIRCUT)
                debit = lp - sp; cost = debit*100 + SPREAD_COMM
                mx = (K2 - K1 - debit)*100 - SPREAD_COMM

                if cost <= 0 or cost > pead_pos or cost > equity*0.35 or mx <= 0: continue

                pnl, aei = None, min(idx+30, len(tk_data)-1)
                for di in range(1, 31):
                    ci = idx + di
                    if ci >= len(tk_data): break
                    Sc = tk_data['close'].iloc[ci]
                    T_rem = max(1, 45 - di) / 365.0
                    cv2 = post_vol * (0.9 + 0.1 * di/30)
                    sv2 = value_spread(Sc, K1, K2, T_rem, cv2)
                    if sv2 - cost >= mx * 0.50: pnl = sv2 - cost; aei = ci; break

                if pnl is None:
                    ci = min(idx+30, len(tk_data)-1)
                    Sc = tk_data['close'].iloc[ci]
                    T_rem = max(1, 15) / 365.0
                    sv2 = value_spread(Sc, K1, K2, T_rem, post_vol*0.95)
                    pnl = sv2 - cost

                equity += pnl; pead_count += 1
                trades.append({'entry': str(dt.date()), 'ticker': tk, 'strategy': 'pead',
                               'pnl': round(pnl,2), 'win': pnl>0, 'regime': 'bull' if bull_regime else 'bear',
                               'gap_pct': round(event['gap_pct'],1),
                               'equity_at_trade': round(equity,2)})

        eq_curve.append(equity)

    return trades, equity, eq_curve, sector_count, pead_count

# ==================== VALIDATION ====================
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

def validate(trades, final_eq, eq_curve, name, s_n, p_n):
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

    # Strategy breakdown
    strats = {}
    for s in ['sector_bull', 'sector_bear', 'pead']:
        st = [t for t in trades if t.get('strategy') == s]
        if st:
            strats[s] = {'n': len(st), 'wr': sum(1 for t in st if t['win'])/len(st)*100,
                         'pnl': sum(t['pnl'] for t in st)}

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
         'sector_trades': s_n, 'pead_trades': p_n,
         'strategy_breakdown': strats,
         'regime_bull_wr': round(bw,1), 'regime_bear_wr': round(brw,1),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}

    fprint(f"  {name}: {n} trades (Sec:{s_n}/PEAD:{p_n}) | "
           f"WR {wr:.1f}% | Sh {sh:.2f} | So {so:.2f} | CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | "
           f"PF {pf:.2f} | ${CAP}->${final_eq:.0f} | Gates {gates}/4")
    for sn, sd in strats.items():
        fprint(f"    {sn}: {sd['n']} trades, WR {sd['wr']:.0f}%, PnL ${sd['pnl']:.0f}")
    fprint(f"    G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'}")
    return r

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Sector + PEAD Combined v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*90}")

    sc, sh, sl, sv, spy, vix = download_sector_data()
    pead_prices = download_pead_data()
    fprint(f"Sector: {len(sc)} days | PEAD: {len(pead_prices)} rows")

    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rankings = build_rankings(sc, sv, spy, bd)
    if not rankings: fprint("FATAL: No rankings"); return

    # Detect PEAD events (only use dates within sector data range)
    fprint("\nDetecting PEAD events...")
    sector_start = sc.index[0]
    all_pead = []
    for tk in PEAD_TICKERS:
        events = detect_earnings(pead_prices, tk)
        # Filter to sector data range
        events = [e for e in events if e['date'] >= sector_start]
        all_pead.extend(events)
    fprint(f"  {len(all_pead)} PEAD events | Gap up >3%: {sum(1 for e in all_pead if e['gap_pct']>3)} | >5%: {sum(1 for e in all_pead if e['gap_pct']>5)}")

    configs = [
        # name, use_sector, use_pead, gap_thresh, sizing, pead_mult
        ('A_Combined_Gap3',    True,  True,  3.0, 'fixed',  1.0),
        ('B_Combined_Gap5',    True,  True,  5.0, 'fixed',  1.0),
        ('C_SectorOnly',       True,  False, 3.0, 'fixed',  1.0),
        ('D_PEADOnly',         False, True,  3.0, 'fixed',  1.0),
        ('E_Combined_HalfPEAD',True,  True,  3.0, 'fixed',  0.5),
        ('F_Combined_Tiered',  True,  True,  3.0, 'tiered', 1.0),
    ]

    results = []
    for nm, us, up, gt, sz, pm in configs:
        fprint(f"\n--- {nm} ---")
        tr, eq, cu, sn, pn = simulate(nm, rankings, sc, sh, sl, spy, vix,
                                       all_pead, pead_prices,
                                       use_sector=us, use_pead=up, gap_threshold=gt,
                                       sizing=sz, pead_size_mult=pm)
        r = validate(tr, eq, cu, nm, sn, pn)
        if r: results.append(r)

    if not results: fprint("No results"); return

    fprint(f"\n{'='*120}")
    fprint(f"SUMMARY — Sector + PEAD Combined v1")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<24} {'#':>5} {'S/P':>7} {'WR':>6} {'Sh':>7} {'So':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>8} {'G':>4}")
    fprint("-"*120)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<24} {r['n_trades']:>5} {r['sector_trades']:>3}/{r['pead_trades']:<3} "
               f"{r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['cagr_pct']:>6.1f}% "
               f"{r['maxdd_pct']:>6.1f}% {r['pf']:>6.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4")

    # Key comparison
    combined = next((r for r in results if r['name']=='A_Combined_Gap3'), None)
    sector = next((r for r in results if r['name']=='C_SectorOnly'), None)
    pead = next((r for r in results if r['name']=='D_PEADOnly'), None)

    if combined and sector:
        fprint(f"\n=== COMBINED vs COMPONENTS ===")
        fprint(f"Sector only:  Sh {sector['sharpe']:.2f} | CAGR {sector['cagr_pct']:.1f}% | MDD {sector['maxdd_pct']:.1f}% | ${sector['final_equity']:.0f}")
        if pead:
            fprint(f"PEAD only:    Sh {pead['sharpe']:.2f} | CAGR {pead['cagr_pct']:.1f}% | MDD {pead['maxdd_pct']:.1f}% | ${pead['final_equity']:.0f}")
        fprint(f"Combined:     Sh {combined['sharpe']:.2f} | CAGR {combined['cagr_pct']:.1f}% | MDD {combined['maxdd_pct']:.1f}% | ${combined['final_equity']:.0f}")

        eq_add = (combined['final_equity'] - sector['final_equity']) / max(sector['final_equity'],1) * 100
        fprint(f"PEAD adds: {eq_add:+.0f}% equity")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'capital': CAP, 'results': results,
                 'runtime_s': round(elapsed,1), 'n_pead_events': len(all_pead)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'sector_pead_combined_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"sec_pead_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'strategy': 'sector+pead', 'n_pead_events': len(all_pead)})
                for r in results:
                    p = r['name'][:20].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_mdd': r['maxdd_pct'],
                                        f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*90}\nDONE — Sector + PEAD Combined v1\n{'='*90}")

if __name__ == '__main__':
    main()
