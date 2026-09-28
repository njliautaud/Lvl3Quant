"""
Breakout After Consolidation Backtest v1
==========================================
Hypothesis: Stocks that consolidate (low volatility) then break out
tend to trend. Bollinger Band squeeze → expansion.

Variants:
A) BB squeeze + upside breakout (close > upper BB after squeeze)
B) Volume breakout — close > 20d high + volume > 2x avg
C) Range contraction — 10d ATR < 50% of 40d ATR, then close > 10d high
D) BB squeeze + RSI>50 (momentum confirmation)
E) Price > 20d high + above 200SMA (quality breakout)
F) BB squeeze + sector leader (relative strength > 1.0 vs SPY)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, warnings
warnings.filterwarnings('ignore')

UNIVERSE = ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AMD',
            'NFLX','CRM','PLTR','SOFI','HOOD','SNAP','PINS','COIN','RBLX','UBER']
ACCOUNT = 669.0
OOT_START, OOT_END = '2022-01-01', '2026-07-30'
HOLD = 10
N_PERMS = 1000

def load_data():
    data = {}
    spy = yf.download('SPY', start='2021-01-01', end=OOT_END, progress=False)
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy['sma200'] = spy['Close'].rolling(200).mean()
    spy['bull'] = spy['Close'] > spy['sma200']
    spy_ret = spy['Close'].pct_change(20)
    
    for t in UNIVERSE:
        try:
            df = yf.download(t, start='2021-01-01', end=OOT_END, progress=False)
            if len(df) > 200:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[t] = df
        except: pass
    return data, spy

def compute(df, spy):
    df = df.copy()
    df['ret'] = df['Close'].pct_change()
    df['sma20'] = df['Close'].rolling(20).mean()
    df['std20'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['sma20'] + 2 * df['std20']
    df['bb_lower'] = df['sma20'] - 2 * df['std20']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
    df['bb_width_pctile'] = df['bb_width'].rolling(100).rank(pct=True)
    df['squeeze'] = df['bb_width_pctile'] < 0.20  # bottom 20% of 100d range
    df['squeeze_prev'] = df['squeeze'].shift(1)
    
    df['high_20d'] = df['High'].rolling(20).max()
    df['high_10d'] = df['High'].rolling(10).max()
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    df['atr_10'] = (df['High'] - df['Low']).rolling(10).mean()
    df['atr_40'] = (df['High'] - df['Low']).rolling(40).mean()
    df['atr_ratio'] = df['atr_10'] / df['atr_40']
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    
    df['sma200'] = df['Close'].rolling(200).mean()
    df['above_200'] = df['Close'] > df['sma200']
    
    # RS vs SPY
    spy_close = spy['Close'].reindex(df.index, method='ffill')
    df['rs_vs_spy'] = (df['Close'].pct_change(20)) / (spy_close.pct_change(20).replace(0, np.nan))
    
    df['fwd_ret'] = df['Close'].shift(-HOLD) / df['Close'] - 1
    return df

def gen_signals(data, spy):
    rows = []
    for ticker, df in data.items():
        df = compute(df, spy)
        oot = df.loc[OOT_START:OOT_END]
        for i, (idx, r) in enumerate(oot.iterrows()):
            if pd.isna(r['fwd_ret']) or pd.isna(r.get('bb_width_pctile')): continue
            rows.append({
                'date': idx, 'ticker': ticker, 'close': r['Close'],
                'squeeze': r['squeeze'], 'squeeze_prev': r.get('squeeze_prev', False),
                'above_bb': r['Close'] > r['bb_upper'] if not pd.isna(r['bb_upper']) else False,
                'above_20d_high': r['Close'] >= r['high_20d'] if not pd.isna(r['high_20d']) else False,
                'above_10d_high': r['Close'] >= r['high_10d'] if not pd.isna(r['high_10d']) else False,
                'vol_ratio': r['vol_ratio'] if not pd.isna(r['vol_ratio']) else 1,
                'atr_ratio': r['atr_ratio'] if not pd.isna(r['atr_ratio']) else 1,
                'rsi': r['rsi'] if not pd.isna(r['rsi']) else 50,
                'above_200': r['above_200'] if not pd.isna(r['above_200']) else True,
                'rs_vs_spy': r['rs_vs_spy'] if not pd.isna(r['rs_vs_spy']) else 1,
                'fwd_ret': r['fwd_ret'],
            })
    return pd.DataFrame(rows)

def vA(s): return s[(s['squeeze_prev']==True) & (s['above_bb']==True)]
def vB(s): return s[(s['above_20d_high']==True) & (s['vol_ratio']>2.0)]
def vC(s): return s[(s['atr_ratio']<0.5) & (s['above_10d_high']==True)]
def vD(s): return s[(s['squeeze_prev']==True) & (s['above_bb']==True) & (s['rsi']>50)]
def vE(s): return s[(s['above_20d_high']==True) & (s['above_200']==True)]
def vF(s): return s[(s['squeeze_prev']==True) & (s['above_bb']==True) & (s['rs_vs_spy']>1.0)]

def bt(trades):
    if trades is None or len(trades) == 0: return None
    t = trades.dropna(subset=['fwd_ret']).sort_values('date')
    pnls, last = [], None
    for _, r in t.iterrows():
        if last and (r['date'] - last).days < HOLD: continue
        sh = int(ACCOUNT * 0.9 / r['close'])
        if sh < 1: continue
        cost = sh * r['close']
        pnl = cost * r['fwd_ret'] - cost * 0.001 * 2
        pnls.append({'date': r['date'], 'pnl': pnl, 'ret': r['fwd_ret'] - 0.002})
        last = r['date']
    if len(pnls) < 5: return None
    p = pd.DataFrame(pnls)
    r = p['ret'].values
    sh = np.mean(r)/np.std(r)*np.sqrt(len(r)/4.5) if np.std(r)>0 else 0
    dr = r[r<0]; ds = np.std(dr) if len(dr)>1 else np.std(r)
    so = np.mean(r)/ds*np.sqrt(len(r)/4.5) if ds>0 else 0
    gp = p[p['pnl']>0]['pnl'].sum(); gl = abs(p[p['pnl']<0]['pnl'].sum())
    pf = gp/gl if gl>0 else 999
    cum = p['pnl'].cumsum(); mdd = (cum - cum.cummax()).min()
    return {'n': len(p), 'pnl': round(p['pnl'].sum(),2), 'ret_pct': round(p['pnl'].sum()/ACCOUNT*100,2),
            'wr': round((p['pnl']>0).mean()*100,1), 'pf': round(pf,2),
            'sharpe': round(sh,3), 'sortino': round(so,3),
            'mdd_pct': round(mdd/ACCOUNT*100,1), 'returns': r.tolist()}

def regime(trades, spy_bull):
    if trades is None or len(trades)==0: return 0,0,99
    t = trades.dropna(subset=['fwd_ret']).sort_values('date')
    br, rr = [], []
    for _, r in t.iterrows():
        d = r['date']
        ib = True
        for sd, b in spy_bull.items():
            if (isinstance(sd, str) and pd.Timestamp(sd) <= d) or (not isinstance(sd, str) and sd <= d): ib = b
        (br if ib else rr).append(r['fwd_ret']-0.002)
    def s(a):
        a=np.array(a)
        return float(np.mean(a)/np.std(a)*np.sqrt(len(a)/4.5)) if len(a)>2 and np.std(a)>0 else 0
    sb,sr = s(br),s(rr)
    g = abs(sb-sr)/max(abs(sb),abs(sr),0.001)
    return round(sb,3),round(sr,3),round(g,3)

def perm(trades, all_s, obs_sh, n=N_PERMS):
    if trades is None or len(trades)<5: return 1.0
    nt = len(trades); better = 0
    vs = all_s.dropna(subset=['fwd_ret'])
    for _ in range(n):
        samp = vs.sample(n=min(nt, len(vs)), replace=False)
        r = bt(samp)
        if r and r['sharpe'] >= obs_sh: better += 1
    return round(better/n, 3)

def main():
    print("="*60)
    print("BREAKOUT AFTER CONSOLIDATION v1")
    print("="*60)
    
    data, spy = load_data()
    print(f"Loaded {len(data)} tickers")
    
    sigs = gen_signals(data, spy)
    print(f"Total signal rows: {len(sigs)}")
    
    spy_bull = spy['bull'].to_dict()
    
    variants = {'A_BB_Squeeze_Breakout': vA, 'B_Volume_Breakout': vB, 'C_Range_Contract': vC,
                'D_Squeeze_RSI': vD, 'E_Quality_Breakout': vE, 'F_Squeeze_Leader': vF}
    
    results = {}
    for name, func in variants.items():
        print(f"\n{'='*50}\nVariant: {name}")
        tr = func(sigs)
        print(f"  Signals: {len(tr)}")
        
        r = bt(tr)
        if r is None:
            print("  SKIP — too few"); results[name] = {'variant':name,'status':'TOO_FEW'}; continue
        
        sb,sr,gap = regime(tr, spy_bull)
        print(f"  Running perms...")
        pp = perm(tr, sigs, r['sharpe'])
        
        g = [r['sharpe']>0.5, pp<0.05, gap<0.5, r['mdd_pct']>-50, r['n']>=20]
        passed = sum(g)
        gs = f"{'✓' if g[0] else '✗'} sh | {'✓' if g[1] else '✗'} perm | {'✓' if g[2] else '✗'} gap | {'✓' if g[3] else '✗'} mdd | {'✓' if g[4] else '✗'} n"
        
        results[name] = {
            'variant':name, 'n_trades':r['n'], 'total_pnl':r['pnl'], 'total_return_pct':r['ret_pct'],
            'win_rate':r['wr'], 'profit_factor':r['pf'], 'sharpe':r['sharpe'], 'sortino':r['sortino'],
            'max_drawdown_pct':r['mdd_pct'], 'sharpe_bull':sb, 'sharpe_bear':sr, 'regime_gap':gap,
            'perm_p':pp, 'gates_passed':str(passed), 'gate_detail':gs
        }
        print(f"  N:{r['n']} PnL:${r['pnl']} Sharpe:{r['sharpe']} WR:{r['wr']}% PF:{r['pf']} MDD:{r['mdd_pct']}%")
        print(f"  Bull:{sb} Bear:{sr} Gap:{gap} Perm:{pp}")
        print(f"  Gates: {passed}/5 — {gs}")
    
    valid = {k:v for k,v in results.items() if 'sharpe' in v}
    champ = max(valid, key=lambda k: int(valid[k].get('gates_passed','0'))*100 + valid[k].get('sharpe',-99)) if valid else "NONE"
    
    with open('/home/jupiter/Lvl3Quant/data/breakout_consolidation_results.json','w') as f:
        json.dump({'metadata':{'script':'breakout_consolidation_backtest.py','run_date':datetime.now().isoformat(),
            'account':ACCOUNT,'oot':f"{OOT_START} to {OOT_END}",'hold':HOLD,'n_perms':N_PERMS},
            'variant_results':results,'champion':champ,
            'summary':{k:{'passed':v.get('gates_passed','0'),'sharpe':v.get('sharpe',0),'pnl':v.get('total_pnl',0)}
                for k,v in results.items() if 'sharpe' in v}}, f, indent=2, default=str)
    
    print(f"\n{'='*60}\nCHAMPION: {champ}\n{'='*60}")

if __name__=='__main__': main()
