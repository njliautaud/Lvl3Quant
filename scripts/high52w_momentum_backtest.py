"""
52-Week High Momentum Backtest v1 (George & Hwang 2004)
========================================================
Stocks near 52w high continue outperforming. Different from standard
momentum — works because anchoring bias makes traders underreact to
stocks near highs. Academic evidence suggests more robust across regimes.

Variants:
A) Top 3 closest to 52w high monthly, hold 1 month
B) Close within 5% of 52w high + RSI>60, hold 10d
C) New 52w high + volume confirmation (>1.5x avg)
D) Relative 52w high rank + above 200SMA filter
E) Top 3 closest to 52w high but EXCLUDE if >0% (already AT high)
F) 52w high nearness + low vol (quiet breakout, not euphoria)
"""

import numpy as np, pandas as pd, yfinance as yf, json, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

UNIVERSE = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AMD',
            'NFLX','CRM','PLTR','SOFI','HOOD','SNAP','PINS','COIN','RBLX','UBER']
ACCOUNT = 669.0
OOT_S, OOT_E = '2022-01-01', '2026-07-30'
N_PERMS = 1000

def load():
    d = {}
    for t in UNIVERSE:
        try:
            df = yf.download(t, start='2021-01-01', end=OOT_E, progress=False)
            if len(df)>200:
                df.columns=[c[0] if isinstance(c,tuple) else c for c in df.columns]
                d[t]=df
        except: pass
    spy = yf.download('SPY', start='2021-01-01', end=OOT_E, progress=False)
    spy.columns=[c[0] if isinstance(c,tuple) else c for c in spy.columns]
    spy['sma200']=spy['Close'].rolling(200).mean()
    spy['bull']=spy['Close']>spy['sma200']
    return d, spy

def features(df):
    df=df.copy()
    df['high52w']=df['High'].rolling(252).max()
    df['pct_from_high']=(df['Close']-df['high52w'])/df['high52w']
    df['nearness']=df['Close']/df['high52w']  # 1.0 = at high
    delta=df['Close'].diff()
    g=delta.where(delta>0,0).rolling(14).mean()
    l=(-delta.where(delta<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/l.replace(0,np.nan)))
    df['vol_ratio']=df['Volume']/df['Volume'].rolling(20).mean()
    df['sma200']=df['Close'].rolling(200).mean()
    df['above200']=df['Close']>df['sma200']
    df['atr_pct']=(df['High']-df['Low']).rolling(20).mean()/df['Close']
    return df

def gen_monthly_rankings(data):
    """Generate monthly cross-sectional rankings by nearness to 52w high."""
    all_features = {}
    for t, df in data.items():
        df = features(df)
        all_features[t] = df
    
    # Get all trading dates in OOT
    dates = pd.date_range(OOT_S, OOT_E, freq='B')
    
    signals = []
    # Monthly rebalance: first trading day of each month
    months_seen = set()
    for d in dates:
        m = (d.year, d.month)
        if m in months_seen: continue
        months_seen.add(m)
        
        scores = []
        for t, df in all_features.items():
            if d not in df.index:
                # Find nearest prior date
                prior = df.index[df.index <= d]
                if len(prior) == 0: continue
                d_use = prior[-1]
            else:
                d_use = d
            
            row = df.loc[d_use]
            if pd.isna(row.get('nearness')): continue
            
            # Forward return: 21 trading days
            fwd_idx = df.index[df.index > d_use]
            if len(fwd_idx) < 21: continue
            fwd_price = df.loc[fwd_idx[20], 'Close']
            fwd_ret = fwd_price / row['Close'] - 1
            
            scores.append({
                'date': d_use, 'ticker': t, 'close': row['Close'],
                'nearness': row['nearness'], 'pct_from_high': row['pct_from_high'],
                'rsi': row['rsi'], 'vol_ratio': row['vol_ratio'],
                'above200': row['above200'], 'atr_pct': row['atr_pct'],
                'fwd_ret_21d': fwd_ret
            })
        
        if len(scores) > 0:
            sdf = pd.DataFrame(scores).sort_values('nearness', ascending=False)
            for _, s in sdf.iterrows():
                s_dict = s.to_dict()
                s_dict['rank'] = list(sdf['ticker']).index(s['ticker']) + 1
                signals.append(s_dict)
    
    return pd.DataFrame(signals)

def vA(s): return s[s['rank']<=3]  # Top 3 nearest to 52w high
def vB(s): return s[(s['pct_from_high']>-0.05) & (s['rsi']>60)]
def vC(s): return s[(s['nearness']>=1.0) & (s['vol_ratio']>1.5)]  # New highs + volume
def vD(s): return s[(s['rank']<=3) & (s['above200']==True)]
def vE(s): return s[(s['rank']<=3) & (s['nearness']<1.0)]  # Near but not at high
def vF(s): return s[(s['rank']<=3) & (s['atr_pct']<s['atr_pct'].median())]  # Low vol

def bt(trades, hold=21):
    if trades is None or len(trades)==0: return None
    t = trades.dropna(subset=['fwd_ret_21d']).sort_values('date')
    
    # For monthly signals, allow multiple positions per month (portfolio)
    # Group by date, take top positions
    pnls = []
    for date in t['date'].unique():
        day_trades = t[t['date']==date]
        for _, r in day_trades.iterrows():
            sh = int(ACCOUNT * 0.3 / r['close'])  # 30% per position (3 stocks = 90%)
            if sh < 1: continue
            cost = sh * r['close']
            pnl = cost * r['fwd_ret_21d'] - cost * 0.001 * 2
            pnls.append({'date': r['date'], 'ticker': r['ticker'], 'pnl': pnl, 
                         'ret': r['fwd_ret_21d'] - 0.002})
    
    if len(pnls) < 10: return None
    p = pd.DataFrame(pnls)
    
    # Aggregate by month
    p['month'] = p['date'].apply(lambda x: (x.year, x.month) if hasattr(x,'year') else (pd.Timestamp(x).year, pd.Timestamp(x).month))
    monthly = p.groupby('month').agg({'pnl':'sum','ret':'mean'}).reset_index()
    
    r = monthly['ret'].values
    sh = np.mean(r)/np.std(r)*np.sqrt(12) if np.std(r)>0 else 0  # annualized
    dr = r[r<0]; ds = np.std(dr) if len(dr)>1 else np.std(r)
    so = np.mean(r)/ds*np.sqrt(12) if ds>0 else 0
    gp = monthly[monthly['pnl']>0]['pnl'].sum()
    gl = abs(monthly[monthly['pnl']<0]['pnl'].sum())
    pf = gp/gl if gl>0 else 999
    cum = monthly['pnl'].cumsum(); mdd = (cum-cum.cummax()).min()
    
    return {'n': len(monthly), 'n_trades': len(p), 'pnl': round(monthly['pnl'].sum(),2),
            'ret_pct': round(monthly['pnl'].sum()/ACCOUNT*100,2),
            'wr': round((monthly['pnl']>0).mean()*100,1), 'pf': round(pf,2),
            'sharpe': round(sh,3), 'sortino': round(so,3),
            'mdd_pct': round(mdd/ACCOUNT*100,1), 'returns': r.tolist()}

def regime(trades, spy):
    if trades is None or len(trades)==0: return 0,0,99
    t = trades.dropna(subset=['fwd_ret_21d']).sort_values('date')
    bull_d = spy['bull'].to_dict()
    br,rr=[],[]
    for _,r in t.iterrows():
        d=r['date']
        ib=True
        for sd,b in bull_d.items():
            if (isinstance(sd,str) and pd.Timestamp(sd)<=d) or (not isinstance(sd,str) and sd<=d): ib=b
        (br if ib else rr).append(r['fwd_ret_21d']-0.002)
    def s(a):
        a=np.array(a)
        return float(np.mean(a)/np.std(a)*np.sqrt(12)) if len(a)>2 and np.std(a)>0 else 0
    sb,sr=s(br),s(rr); g=abs(sb-sr)/max(abs(sb),abs(sr),0.001)
    return round(sb,3),round(sr,3),round(g,3)

def perm(trades, all_s, obs_sh, n=N_PERMS):
    if trades is None or len(trades)<10: return 1.0
    nt=len(trades); better=0
    vs=all_s.dropna(subset=['fwd_ret_21d'])
    for _ in range(n):
        samp=vs.sample(n=min(nt,len(vs)),replace=False)
        r=bt(samp)
        if r and r['sharpe']>=obs_sh: better+=1
    return round(better/n,3)

def main():
    print("="*60)
    print("52-WEEK HIGH MOMENTUM v1")
    print("="*60)
    data, spy = load()
    print(f"Loaded {len(data)} tickers")
    
    sigs = gen_monthly_rankings(data)
    print(f"Total signal rows: {len(sigs)}")
    
    variants = {'A_Top3_Nearest':vA, 'B_Near_RSI':vB, 'C_New_High_Vol':vC,
                'D_Top3_Quality':vD, 'E_Near_NotAt':vE, 'F_Top3_LowVol':vF}
    
    results = {}
    for name, func in variants.items():
        print(f"\n{'='*50}\n{name}")
        tr = func(sigs)
        print(f"  Signals: {len(tr)}")
        r = bt(tr)
        if r is None:
            print("  SKIP"); results[name]={'variant':name,'status':'TOO_FEW'}; continue
        
        sb,sr,gap = regime(tr, spy)
        print(f"  Perms...")
        pp = perm(tr, sigs, r['sharpe'])
        
        g=[r['sharpe']>0.5, pp<0.05, gap<0.5, r['mdd_pct']>-50, r['n']>=20]
        passed=sum(g)
        gs=f"{'✓' if g[0] else '✗'}sh {'✓' if g[1] else '✗'}perm {'✓' if g[2] else '✗'}gap {'✓' if g[3] else '✗'}mdd {'✓' if g[4] else '✗'}n"
        
        results[name]={
            'variant':name,'n_months':r['n'],'n_trades':r['n_trades'],
            'total_pnl':r['pnl'],'total_return_pct':r['ret_pct'],
            'win_rate':r['wr'],'profit_factor':r['pf'],
            'sharpe':r['sharpe'],'sortino':r['sortino'],'mdd_pct':r['mdd_pct'],
            'sharpe_bull':sb,'sharpe_bear':sr,'regime_gap':gap,
            'perm_p':pp,'gates_passed':str(passed),'gate_detail':gs
        }
        print(f"  M:{r['n']} T:{r['n_trades']} PnL:${r['pnl']} Sh:{r['sharpe']} WR:{r['wr']}% PF:{r['pf']} MDD:{r['mdd_pct']}%")
        print(f"  Bull:{sb} Bear:{sr} Gap:{gap} Perm:{pp}")
        print(f"  Gates: {passed}/5 — {gs}")
    
    valid={k:v for k,v in results.items() if 'sharpe' in v}
    champ=max(valid,key=lambda k:int(valid[k].get('gates_passed','0'))*100+valid[k].get('sharpe',-99)) if valid else "NONE"
    
    with open('/home/jupiter/Lvl3Quant/data/high52w_momentum_results.json','w') as f:
        json.dump({'metadata':{'script':'high52w_momentum_backtest.py','run_date':datetime.now().isoformat()},
            'variant_results':results,'champion':champ}, f, indent=2, default=str)
    
    print(f"\n{'='*60}\nCHAMPION: {champ}\n{'='*60}")

if __name__=='__main__': main()
