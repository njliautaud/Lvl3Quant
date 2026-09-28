#!/usr/bin/env python3
"""VIX-Filtered Sector Spreads v1 — Sector rotation bull call spreads (Sharpe 3.76)
+ VIX regime filter + LSTM spike prediction. 7 variants A-G.
MLflow: vix_filtered_sector_spreads_v1, server http://jupiter:5000
"""
import json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
warnings.filterwarnings('ignore')
def fprint(*a, **kw): print(*a, **kw, flush=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / 'research' / 'findings'; OUT_DIR.mkdir(parents=True, exist_ok=True)
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
DEFENSIVE = ['XLP','XLU']
CAP = 645.0; SPREAD_COMM = 2.60; HAIRCUT = 0.15
MAX_POS, MAX_CONC = 200, 3
TRAIN_D, TEST_D, SEQ_LEN = 252, 21, 20
SPIKE_TH, HORIZON = 25, 5
EPOCHS, BATCH, LR = 50, 256, 1e-3
LGBM_FEAT = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

# === 1. DATA ===
def download_data():
    import yfinance as yf
    fprint("[1/7] Downloading data...")
    tickers = SECTORS + ['SPY','^VIX','TLT','GLD','HYG','QQQ','IWM']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close, high, low = (raw['Close'], raw['High'], raw['Low']) if mi else (raw, raw, raw)
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    extras = {tk: close[tk].dropna() for tk in ['TLT','GLD','HYG','QQQ','IWM'] if tk in close.columns}
    ix = sc.index
    for s in [vix, spy, sh, sl] + list(extras.values()): ix = ix.intersection(s.index)
    fprint(f"  {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix], {k: v.loc[ix] for k, v in extras.items()}

# === 2. LGBM SECTOR RANKING ===
def sector_features(px, idx, tk):
    p = px[tk].iloc[:idx+1].dropna()
    if len(p) < 260: return None
    f = {nm: float(p.iloc[-1]/p.iloc[-lb]-1) if len(p) > lb else 0.0
         for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]}
    r = p.pct_change().dropna()
    f['vol_21d'] = float(r.iloc[-21:].std()*np.sqrt(252)) if len(r) > 21 else 0.2
    f['vol_63d'] = float(r.iloc[-63:].std()*np.sqrt(252)) if len(r) > 63 else 0.2
    r63 = r.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0
    p63 = p.iloc[-63:]; f['maxdd_63d'] = float(((p63/p63.cummax())-1).min())
    f['pct_52w_high'] = float(p.iloc[-1]/p.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    return f

def lgbm_wf(px, dates, tp=12):
    fprint("[2/7] LightGBM walk-forward ranking...")
    recs = []
    for dt in dates:
        idx = px.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in px.columns:
            f = sector_features(px, idx, tk)
            if not f: continue
            fi = min(idx+21, len(px)-1)
            f.update({'date': dt, 'ticker': tk, 'fwd_ret': float(px[tk].iloc[fi]/px[tk].iloc[idx]-1)})
            recs.append(f)
    df = pd.DataFrame(recs)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    udates = sorted(df['date'].unique()); ranks = {}
    for i in range(tp, len(udates)):
        td = udates[max(0,i-tp):i]; tdate = udates[i]
        tr, te = df[df['date'].isin(td)], df[df['date']==tdate].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[LGBM_FEAT].values.astype(np.float32))
        Xe = np.nan_to_num(te[LGBM_FEAT].values.astype(np.float32))
        try:
            try: m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_samples=5, device='gpu', verbose=-1); m.fit(Xt, tr['rank_label'].values)
            except: m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_samples=5, verbose=-1); m.fit(Xt, tr['rank_label'].values)
            te['score'] = m.predict(Xe); ranks[tdate] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"  Rankings: {len(ranks)} dates"); return ranks

# === 3. LSTM VIX SPIKE PREDICTOR ===
def build_vix_features(spy, vix, extras):
    f = pd.DataFrame(index=vix.index)
    tlt, gld, hyg, qqq, iwm = [extras.get(k, spy) for k in ['TLT','GLD','HYG','QQQ','IWM']]
    sr, tr = spy.pct_change(), tlt.pct_change()
    f['vix_level'] = vix
    for w in [5,10,20]: f[f'vix_chg_{w}d'] = vix - vix.shift(w)
    f['vix_pctile_1y'] = vix.rolling(252).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
    f['vix_ma20_ratio'] = vix / vix.rolling(20).mean()
    f['vix_dist_high_20d'] = vix / vix.rolling(20).max() - 1
    f['vix_dist_low_20d'] = vix / vix.rolling(20).min() - 1
    f['vix_rising_days'] = (vix.diff() > 0).rolling(10).sum()
    f['vix_accel'] = vix.diff().diff()
    for w in [5,10,20]: f[f'spy_ret_{w}d'] = spy / spy.shift(w) - 1
    for w in [10,20,60]: f[f'spy_rvol_{w}d'] = sr.rolling(w).std() * np.sqrt(252)
    f['spy_qqq_vol_ratio'] = sr.rolling(20).std() / qqq.pct_change().rolling(20).std()
    f['spy_consec_down'] = (sr < 0).rolling(10).sum()
    f['spy_dist_sma200'] = spy / spy.rolling(200).mean() - 1
    f['iwm_spy_spread_10d'] = (iwm/iwm.shift(10)-1) - (spy/spy.shift(10)-1)
    for w in [5,10]: f[f'hyg_ret_{w}d'] = hyg / hyg.shift(w) - 1
    f['hyg_tlt_spread_chg_10d'] = (hyg/hyg.shift(10)) - (tlt/tlt.shift(10))
    f['credit_stress_rising'] = (f['hyg_tlt_spread_chg_10d'] < f['hyg_tlt_spread_chg_10d'].shift(5)).astype(float)
    f['gld_ret_10d'] = gld/gld.shift(10)-1; f['tlt_ret_10d'] = tlt/tlt.shift(10)-1
    f['spy_tlt_corr_20d'] = sr.rolling(20).corr(tr)
    f['vix_slope_proxy'] = vix.rolling(5).mean() - vix.rolling(20).mean()
    f['day_of_week'] = pd.Series(vix.index.dayofweek, index=vix.index).astype(float)
    f['month'] = pd.Series(vix.index.month, index=vix.index).astype(float)
    sm = (vix >= SPIKE_TH).astype(int); ds = pd.Series(np.nan, index=vix.index); last = -999
    for i, (idx, val) in enumerate(sm.items()):
        if val == 1: last = i
        ds.iloc[i] = i - last if last >= 0 else 999
    f['days_since_spike'] = ds.clip(upper=252)
    tgt = ((vix.shift(-1).rolling(HORIZON).max().shift(-(HORIZON-1))) >= SPIKE_TH).astype(float)
    f = f.replace([np.inf, -np.inf], np.nan)
    v = f.dropna().index.intersection(tgt.dropna().index)
    fprint(f"  VIX features: {len(f.columns)} cols, {len(v)} samples")
    return f.loc[v], tgt.loc[v]

class FocalLoss(nn.Module):
    def __init__(self, a=0.75, g=2.0): super().__init__(); self.a, self.g = a, g
    def forward(self, lo, tg):
        bce = nn.functional.binary_cross_entropy_with_logits(lo, tg, reduction='none')
        pt = tg*torch.sigmoid(lo) + (1-tg)*(1-torch.sigmoid(lo))
        return ((tg*self.a+(1-tg)*(1-self.a))*(1-pt)**self.g*bce).mean()

class LSTMModel(nn.Module):
    def __init__(self, nf, sl=20):
        super().__init__()
        self.lstm = nn.LSTM(nf, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))
    def forward(self, x): return self.head(self.lstm(x)[0][:,-1,:]).squeeze(-1)

def train_lstm(mdl, Xt, yt, Xv, yv):
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); crit = FocalLoss()
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True)
    ba, bs, w = 0.0, None, 0
    for _ in range(EPOCHS):
        mdl.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(mdl(xb), yb).backward(); nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
        sch.step(); mdl.eval()
        with torch.no_grad():
            vp = torch.sigmoid(mdl(Xv)).cpu().numpy(); yn = yv.cpu().numpy()
            if len(np.unique(yn)) < 2: continue
            a = roc_auc_score(yn, vp)
            if a > ba: ba = a; bs = {k: v.clone() for k, v in mdl.state_dict().items()}; w = 0
            else:
                w += 1
                if w >= 10: break
    if bs: mdl.load_state_dict(bs)

def gen_spike_probs(feats, tgt):
    fprint("[3/7] LSTM VIX spike walk-forward...")
    n, nf = len(feats), len(feats.columns); ap, ad, nfold = [], [], 0
    for ts in range(TRAIN_D+SEQ_LEN, n-TEST_D, TEST_D):
        t0 = max(0, ts-TRAIN_D-SEQ_LEN); te = min(ts+TEST_D, n)
        Xr, yr = feats.iloc[t0:te].values, tgt.iloc[t0:te].values; sp = ts-t0
        sc = StandardScaler(); Xs = sc.fit_transform(Xr[:sp]); Xt = sc.transform(Xr[sp:])
        def mk(d, l, o):
            s, la = [], []
            for i in range(SEQ_LEN, len(d)): s.append(d[i-SEQ_LEN:i]); la.append(l[o+i])
            return np.array(s), np.array(la)
        Xts, yts = mk(Xs, yr, 0)
        Xc = np.vstack([Xs[-SEQ_LEN:], Xt]); yc = yr[sp-SEQ_LEN:]
        Xes, yes = [], []
        for i in range(SEQ_LEN, len(Xc)):
            Xes.append(Xc[i-SEQ_LEN:i])
            if i < len(yc): yes.append(yc[i])
        Xes = Xes[:len(yes)]; Xes = np.array(Xes) if Xes else np.empty((0,SEQ_LEN,nf)); yes = np.array(yes)
        if len(Xts) < 50 or len(Xes) < 5: continue
        m = LSTMModel(nf, SEQ_LEN).to(DEVICE)
        train_lstm(m, torch.tensor(Xts,dtype=torch.float32).to(DEVICE), torch.tensor(yts,dtype=torch.float32).to(DEVICE),
                   torch.tensor(Xes,dtype=torch.float32).to(DEVICE), torch.tensor(yes,dtype=torch.float32).to(DEVICE))
        m.eval()
        with torch.no_grad(): p = torch.sigmoid(m(torch.tensor(Xes,dtype=torch.float32).to(DEVICE))).cpu().numpy()
        ap.extend(p.tolist()); ad.extend(feats.index[ts:ts+len(p)].tolist()); nfold += 1
        if nfold % 25 == 0: fprint(f"    Fold {nfold}, {len(ap)} preds")
    ps = pd.Series(ap, index=pd.DatetimeIndex(ad), name='spike_prob')
    cm = ps.index.intersection(tgt.index)
    auc = roc_auc_score(tgt.loc[cm], ps.loc[cm]) if len(cm) > 100 else 0
    fprint(f"  LSTM: {nfold} folds, {len(ap)} preds, AUC={auc:.3f}"); return ps, auc

# === 4. ATR OPTIONS PRICING ===
def compute_atr(h, l, c, p=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(p).mean()

def atr_prem(S, K, dte, atr, vx, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    return (max(0,S-K) if opt=='call' else max(0,K-S)) + atr*np.sqrt(T)*max(0.3,vx/20)*np.exp(-3*abs(S-K)/S)

def price_spread(S, spct, dte, atr, vx):
    K1, K2 = round(S), round(S*(1+spct/100))
    lp = atr_prem(S, K1, dte, atr, vx, 'call')*(1+HAIRCUT)
    sp = atr_prem(S, K2, dte, atr, vx, 'call')*(1-HAIRCUT)
    deb = lp-sp; w = K2-K1
    return deb, (w-deb)*100-SPREAD_COMM, deb*100+SPREAD_COMM, K1, K2

# === 5. SIMULATION ===
def simulate(var, ranks, sc, sh, sl, spy, vix, extras, sprobs, atr_d):
    eq, trades, curve = CAP, [], [CAP]; sma200 = spy.rolling(200).mean()
    for dt in sorted(ranks.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt]); scores = ranks[dt]
        if not scores: curve.append(eq); continue
        sp = float(sprobs.loc[dt]) if dt in sprobs.index else 0.0
        skip, smul, use_def, hedge = False, 1.0, False, False
        if var == 'B_VIX_Cautious' and cv > 30: skip = True
        elif var == 'C_VIX_Defensive' and cv > 25: use_def = True
        elif var == 'D_LSTM_Filter' and sp >= 0.5: skip = True
        elif var == 'E_LSTM_Size' and sp > 0.5: smul = 0.5
        elif var == 'F_Combined' and (cv > 30 or sp >= 0.7): skip = True
        elif var == 'G_Hedged' and sp > 0.7: hedge = True
        if skip: curve.append(eq); continue
        if use_def:
            picks = sorted([(tk, float(sc[tk].iloc[sc.index.get_loc(dt)]/sc[tk].iloc[max(0,sc.index.get_loc(dt)-21)]-1))
                            for tk in DEFENSIVE if tk in sc.columns], key=lambda x: x[1], reverse=True)
            picks = [t for t,_ in picks[:3]]
        else:
            picks = [t for t,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]]
        mx = min(MAX_POS*smul, eq/3)
        if mx < 30: curve.append(eq); continue
        bull = float(spy.loc[dt]) >= (float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else float(spy.loc[dt]))
        ne = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or ne >= MAX_CONC: continue
            S = float(sc[tk].loc[dt]); di = sc.index.get_loc(dt); dte = 14
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            ei = min(di+dte, len(sc)-1); Se = float(sc[tk].iloc[ei])
            val, mxp, mxl, K1, K2 = price_spread(S, 3.0, dte, av, cv)
            cost = val*100+SPREAD_COMM
            if cost <= 0 or cost > mx or cost > eq*0.40: continue
            pnl, aei = None, ei
            for ci in range(di+3, ei+1):
                Sc = float(sc[tk].iloc[ci]); rd = max(0, dte-(ci-di))
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))
                if var in ('D_LSTM_Filter','F_Combined'):
                    cdt = sc.index[ci]; csp = float(sprobs.loc[cdt]) if cdt in sprobs.index else 0
                    if csp >= 0.7:
                        pnl = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100 - cost; aei = ci; break
                si = (max(0,Sc-K1)-max(0,Sc-K2))*100 + ac*tm*0.3*100
                if si-cost >= mxp*0.5 or rd < 7: pnl = si-cost; aei = ci; break
            if pnl is None: pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - cost
            eq += pnl; ne += 1
            trades.append({'entry':str(dt.date()),'exit':str(sc.index[aei].date()),'ticker':tk,
                'pnl':round(pnl,2),'win':pnl>0,'regime':'bull' if bull else 'bear','vix':round(cv,1),'lstm':round(sp,3)})
        if hedge and eq > 50:  # VIX call spread hedge
            hc = min(max(cv*0.05,1.0)*10, eq*0.05); di = sc.index.get_loc(dt); fi = min(di+14,len(vix)-1)
            fvm = vix.iloc[di:fi+1].max()
            hp = min((fvm-cv)*3, hc*3)-hc if fvm > cv*1.15 else -hc
            eq += hp
            trades.append({'entry':str(dt.date()),'exit':str(sc.index[fi].date()),'ticker':'VIX_HEDGE',
                'pnl':round(hp,2),'win':hp>0,'regime':'hedge','vix':round(cv,1),'lstm':round(sp,3)})
        curve.append(eq)
    return trades, eq, curve

# === 6. METRICS & VALIDATION ===
def metrics(trades, feq, curve, name):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    tdf = pd.DataFrame(trades); tdf['m'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    mr = tdf.groupby('m')['pnl'].sum()/CAP; ny = max(len(mr)/12, 0.5)
    sh = (mr.mean()*12)/(mr.std()*np.sqrt(12)+1e-10) if len(mr) > 3 else 0
    dn = mr[mr<0]; so = (mr.mean()*12)/(dn.std()*np.sqrt(12)+1e-10) if len(dn) > 1 else 0
    cagr = (feq/CAP)**(1/ny)-1
    eq = np.array(curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0)); pf = gp/(gl+1e-10)
    mcl = c = 0
    for t in trades:
        if not t['win']: c += 1; mcl = max(mcl, c)
        else: c = 0
    bt = [t for t in trades if t.get('regime')=='bull']; brt = [t for t in trades if t.get('regime')=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100; brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100
    r = {'name':name,'n_trades':n,'win_rate':round(wr,1),'sharpe':round(sh,2),'sortino':round(so,2),
         'cagr_pct':round(cagr*100,1),'maxdd_pct':round(mdd*100,1),'pf':round(pf,2),'avg_pnl':round(np.mean(pnls),2),
         'final_equity':round(feq,2),'total_pnl':round(sum(pnls),2),'max_consec_loss':mcl,
         'bull_wr':round(bw,1),'bear_wr':round(brw,1),'bull_n':len(bt),'bear_n':len(brt),'monthly_returns':mr.values.tolist()}
    fprint(f"  {name:20s} | {n:4d} | WR {wr:5.1f}% | Sh {sh:5.2f} | So {so:5.2f} | "
           f"CAGR {cagr*100:5.1f}% | MDD {mdd*100:5.1f}% | PF {pf:5.2f} | ${CAP:.0f}->${feq:.0f} | MCL {mcl}")
    return r

def validate(r):
    if r is None: return r
    rets = np.array(r['monthly_returns'])
    if len(rets) < 10: r.update({'gates':0,'perm_p':1.0,'r1_gap':1.0}); return r
    gates = 0; rs = np.mean(rets)/(np.std(rets)+1e-10)
    pp = sum(1 for _ in range(1000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/1000
    g1 = pp < 0.05; gates += g1
    rg = abs(r['bull_wr']-r['bear_wr'])/max(r['bull_wr'],r['bear_wr'],1); g2 = rg < 0.50; gates += g2
    mid = len(rets)//2
    h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
    h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
    g3 = h1 > 0 and h2 > 0; gates += g3
    g4 = False
    if len(rets) > 5: tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0
    gates += g4
    r.update({'gates':gates,'perm_p':round(pp,3),'r1_gap':round(rg,3),'g1_perm':g1,'g2_regime':g2,
              'g3_sub':g3,'g4_outlier':g4,'h1_sh':round(h1,2),'h2_sh':round(h2,2)})
    fprint(f"    Gates: P={'Y' if g1 else 'N'}({pp:.3f}) R1={'Y' if g2 else 'N'}({rg:.2f}) "
           f"Sub={'Y' if g3 else 'N'}({h1:.2f}/{h2:.2f}) Out={'Y' if g4 else 'N'} => {gates}/4")
    return r

# === 7. MAIN ===
def main():
    t0 = time.time()
    fprint(f"VIX-Filtered Sector Spreads v1 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    fprint(f"{'='*75}\nCap: ${CAP:.0f} | ATR+{HAIRCUT:.0%} haircut | Comm: ${SPREAD_COMM}/spread | Device: {DEVICE}\n{'='*75}")
    sc, sh, sl, spy, vix, extras = download_data()
    rdates = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    ranks = lgbm_wf(sc, rdates)
    if not ranks: fprint("FATAL: No rankings"); return
    vf, vt = build_vix_features(spy, vix, extras)
    sprobs, auc = gen_spike_probs(vf, vt)
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    fprint(f"\n[4/7] Simulating 7 variants...")
    variants = ['A_Baseline','B_VIX_Cautious','C_VIX_Defensive','D_LSTM_Filter','E_LSTM_Size','F_Combined','G_Hedged']
    results = []
    for v in variants:
        tr, eq, cu = simulate(v, ranks, sc, sh, sl, spy, vix, extras, sprobs, atr_d)
        r = metrics(tr, eq, cu, v)
        if r: results.append(validate(r))
    if not results: fprint("No results"); return
    # Summary
    fprint(f"\n{'='*95}\n{'Variant':<22} {'#':>5} {'WR':>6} {'Sh':>6} {'So':>6} {'CAGR':>7} {'MDD':>7} {'PF':>5} {'$':>8} {'MCL':>4} {'G':>4}\n{'-'*95}")
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} {r['sortino']:>6.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>5.2f} ${r['final_equity']:>7.0f} {r['max_consec_loss']:>3} {r['gates']:>3}/4")
    bs = max(results, key=lambda x: x['sharpe']); bl = next((r for r in results if r['name']=='A_Baseline'), None)
    vl = [r for r in results if r['gates'] >= 3]
    fprint(f"\nBEST: {bs['name']} Sharpe={bs['sharpe']} CAGR={bs['cagr_pct']}% MDD={bs['maxdd_pct']}%")
    if bl and bs['name'] != 'A_Baseline':
        fprint(f"vs BASELINE: Sharpe {bs['sharpe']-bl['sharpe']:+.2f}, MDD {(1-abs(bs['maxdd_pct'])/max(abs(bl['maxdd_pct']),0.01))*100:+.1f}%")
    if vl: bv = max(vl, key=lambda x: x['sharpe']); fprint(f"BEST VALID (3+): {bv['name']} Sharpe={bv['sharpe']}")
    fprint(f"LSTM AUC: {auc:.3f}")
    # MLflow
    if MLFLOW_OK:
        try:
            en = 'vix_filtered_sector_spreads_v1'
            try:
                if not mlflow.get_experiment_by_name(en): mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"vfs_v1_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_params({'pricing':'ATR','haircut':HAIRCUT,'capital':CAP,'lstm_auc':round(auc,3),'device':str(DEVICE)})
                for r in results:
                    for k in ['sharpe','sortino','cagr_pct','maxdd_pct','win_rate','pf','gates']:
                        try: mlflow.log_metric(f"{r['name']}_{k}", r[k])
                        except: pass
            fprint("MLflow: logged")
        except Exception as e: fprint(f"MLflow: {e}")
    # Save
    op = OUT_DIR / 'vix_filtered_sector_spreads_v1_results.json'
    save = {'strategy':'VIX-Filtered Sector Spreads v1','run_date':datetime.now().isoformat(),
            'lstm_auc':round(auc,3),'capital':CAP,'pricing':f'ATR+{HAIRCUT:.0%}',
            'variants':[{k:v for k,v in r.items() if k!='monthly_returns'} for r in results],
            'best':bs['name'],'best_valid':vl[0]['name'] if vl else 'NONE'}
    with open(op, 'w') as f: json.dump(save, f, indent=2, default=lambda o: float(o) if hasattr(o,'__float__') else str(o))
    fprint(f"\nDone — {time.time()-t0:.0f}s on {DEVICE}")

if __name__ == '__main__':
    main()
