#!/usr/bin/env python3
"""
Vol Compression Breakout — LONG-ONLY Portfolio v3

v2 showed: 10d with longs+shorts = 11.3% CAGR but regime gap 1.91 (FAIL).
The SHORT side dragged bear Sharpe to -0.92.
Hypothesis: LONG-ONLY breakouts should fix regime gap since the original
per-trade stats showed longs work in both regimes.

Also tests: what if we only take UPWARD breakouts from vol compression?
"""
import numpy as np, pandas as pd, yfinance as yf, json, os, sys, time, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/nick/Lvl3Quant/output/vol_compression_longonly_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)

def get_tickers():
    cache = '/home/nick/Lvl3Quant/output/sp500_tickers.json'
    if os.path.exists(cache):
        with open(cache) as f: return json.load(f)
    return ['SPY']

def download_data(tickers, start='2010-01-01', end='2026-07-22'):
    all_data = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        log(f'  Downloading batch {i//50+1}/{(len(tickers)-1)//50+1}')
        try:
            data = yf.download(batch, start=start, end=end, progress=False, auto_adjust=True, threads=True)
            if data is None or data.empty: continue
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        if t in data.columns.get_level_values(1):
                            td = data.xs(t, level=1, axis=1)
                        elif t in data.columns.get_level_values(0):
                            td = data[t]
                        else: continue
                        if 'Close' in td.columns and len(td.dropna(subset=['Close'])) > 100:
                            all_data[t] = td[['Open','High','Low','Close','Volume']].dropna()
                    except: pass
            else:
                if len(batch)==1 and 'Close' in data.columns:
                    all_data[batch[0]] = data[['Open','High','Low','Close','Volume']].dropna()
        except: pass
        time.sleep(0.3)
    return all_data

def detect_breakouts(prices_dict, vol_pct=10, lookback=63, direction_filter='long'):
    signals = []
    for ticker, df in prices_dict.items():
        if len(df) < lookback+20: continue
        df = df.copy()
        df['ret'] = df['Close'].pct_change()
        df['vol_20d'] = df['ret'].rolling(20).std()
        df['vol_pctile'] = df['vol_20d'].rolling(lookback).apply(lambda x: (x < x.iloc[-1]).sum()/len(x)*100, raw=False)
        df['compressed'] = df['vol_pctile'] < vol_pct
        df['breakout'] = df['compressed'].shift(1) & ~df['compressed']
        df['direction'] = np.where(df['ret'] > 0, 'long', 'short')
        
        for day in df[df['breakout']].index:
            idx = df.index.get_loc(day)
            if idx+10 >= len(df): continue
            d = df.iloc[idx]['direction']
            if direction_filter and d != direction_filter: continue
            signals.append({'ticker': ticker, 'date': day, 'entry_price': df.iloc[idx]['Close'], 'direction': d})
    return pd.DataFrame(signals)

def run_sim(signals_df, prices_dict, max_pos=10, hold_days=10, init_cap=100000, cost_ps=0.005, slip=0.0005):
    signals_df = signals_df.sort_values('date').copy()
    spy = prices_dict.get('SPY')
    if spy is not None:
        spy = spy.copy()
        spy['sma200'] = spy['Close'].rolling(200).mean()
        spy['regime'] = np.where(spy['Close'] > spy['sma200'], 'bull', 'bear')
    
    all_dates = sorted(set().union(*[set(df.index) for df in prices_dict.values()]))
    all_dates = [d for d in all_dates if d >= signals_df['date'].min()]
    
    pf = {'cash': init_cap, 'positions': [], 'nav_history': [], 'trades': []}
    sig_by_date = signals_df.groupby('date')
    
    for date in all_dates:
        new_pos = []
        for pos in pf['positions']:
            dh = len([d for d in all_dates if pos['entry_date'] <= d <= date])
            if dh >= hold_days:
                ep = prices_dict[pos['ticker']].loc[date,'Close'] if pos['ticker'] in prices_dict and date in prices_dict[pos['ticker']].index else pos.get('lp', pos['entry_price'])
                pnl = (ep - pos['entry_price']) * pos['shares']
                cost = cost_ps * pos['shares'] * 2 + ep * pos['shares'] * slip
                pnl -= cost
                pf['cash'] += ep * pos['shares'] + pnl
                pf['trades'].append({'ticker': pos['ticker'], 'pnl': pnl, 'ret': pnl/(pos['entry_price']*pos['shares'])*100,
                    'entry_date': pos['entry_date'], 'exit_date': date})
            else: new_pos.append(pos)
        pf['positions'] = new_pos
        
        if date in sig_by_date.groups and len(pf['positions']) < max_pos:
            slots = max_pos - len(pf['positions'])
            for _, sig in sig_by_date.get_group(date).head(slots).iterrows():
                if sig['ticker'] not in prices_dict or date not in prices_dict[sig['ticker']].index: continue
                ep = sig['entry_price']
                pv = min(pf['cash'] / max(slots,1), pf['cash'] * 0.95)
                if pv < 100 or ep < 1: continue
                sh = int(pv / ep)
                if sh < 1: continue
                cost = cost_ps * sh + ep * sh * slip
                pf['cash'] -= ep * sh + cost
                pf['positions'].append({'ticker': sig['ticker'], 'entry_price': ep, 'entry_date': date, 'shares': sh})
        
        pv = sum(prices_dict[p['ticker']].loc[date,'Close']*p['shares'] if p['ticker'] in prices_dict and date in prices_dict[p['ticker']].index else p.get('lp',p['entry_price'])*p['shares'] for p in pf['positions'])
        for p in pf['positions']:
            if p['ticker'] in prices_dict and date in prices_dict[p['ticker']].index:
                p['lp'] = prices_dict[p['ticker']].loc[date,'Close']
        
        regime = spy.loc[date,'regime'] if spy is not None and date in spy.index else 'unknown'
        pf['nav_history'].append({'date': date, 'nav': pf['cash']+pv, 'regime': regime})
    
    return pf

def analyze(pf, ic=100000):
    ndf = pd.DataFrame(pf['nav_history'])
    ndf['date'] = pd.to_datetime(ndf['date'])
    ndf = ndf.set_index('date')
    ndf['dr'] = ndf['nav'].pct_change()
    
    yrs = len(ndf)/252
    fn = ndf['nav'].iloc[-1]
    cagr = (fn/ic)**(1/yrs)-1
    dr = ndf['dr'].dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    neg = dr[dr<0]
    sortino = dr.mean()/neg.std()*np.sqrt(252) if len(neg)>0 and neg.std()>0 else 0
    dd = (ndf['nav'] - ndf['nav'].cummax())/ndf['nav'].cummax()
    maxdd = dd.min()
    
    tr = pf['trades']
    w = [t for t in tr if t['pnl']>0]
    l = [t for t in tr if t['pnl']<=0]
    wr = len(w)/len(tr)*100 if tr else 0
    pf_ratio = abs(sum(t['pnl'] for t in w))/abs(sum(t['pnl'] for t in l)) if l else float('inf')
    
    bull = ndf[ndf['regime']=='bull']['dr'].dropna()
    bear = ndf[ndf['regime']=='bear']['dr'].dropna()
    bs = bull.mean()/bull.std()*np.sqrt(252) if len(bull)>10 and bull.std()>0 else 0
    brs = bear.mean()/bear.std()*np.sqrt(252) if len(bear)>10 and bear.std()>0 else 0
    rg = abs(bs-brs)/max(abs(bs),abs(brs),0.001)
    
    ndf['year'] = ndf.index.year
    yearly = [{'year':yr, 'ret':(g['nav'].iloc[-1]/g['nav'].iloc[0]-1)*100} for yr,g in ndf.groupby('year')]
    
    return {'cagr':cagr*100,'maxdd':maxdd*100,'sharpe':sharpe,'sortino':sortino,'wr':wr,'pf':pf_ratio,
            'trades':len(tr),'bull_sharpe':bs,'bear_sharpe':brs,'regime_gap':rg,'final_nav':fn,'years':yrs,'yearly':yearly}

def perm_test(signals_df, prices_dict, real_sharpe, n=200, **kw):
    log(f'Running {n} permutation shuffles...')
    all_dates = sorted(set().union(*[set(df.index) for df in prices_dict.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp('2011-01-01')]
    ps = []
    for i in range(n):
        sh = signals_df.copy()
        sh['date'] = np.random.choice(all_dates, size=len(sh), replace=True)
        try:
            p = run_sim(sh.sort_values('date'), prices_dict, **kw)
            ndf = pd.DataFrame(p['nav_history'])
            dr = ndf['nav'].pct_change().dropna()
            s = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
            ps.append(s)
        except: pass
        if (i+1)%50==0: log(f'  Perm {i+1}/{n}')
    if ps:
        pv = sum(1 for s in ps if s >= real_sharpe)/len(ps)
        return pv, np.mean(ps), np.std(ps)
    return 1.0, 0, 0

def main():
    log('='*60)
    log('VOL COMPRESSION — LONG-ONLY PORTFOLIO v3')
    log('='*60)
    
    tickers = get_tickers()
    if 'SPY' not in tickers: tickers.append('SPY')
    
    log(f'Downloading {len(tickers)} tickers...')
    prices = download_data(tickers)
    log(f'Got {len(prices)} tickers')
    
    for hold in [5, 10, 21]:
        log(f'\n{"="*40}')
        log(f'LONG-ONLY, {hold}d hold, max 10 positions')
        log(f'{"="*40}')
        
        signals = detect_breakouts(prices, direction_filter='long')
        log(f'Found {len(signals)} LONG breakout signals')
        
        portfolio = run_sim(signals, prices, max_pos=10, hold_days=hold)
        results = analyze(portfolio)
        
        log(f'\n  CAGR: {results["cagr"]:.1f}%')
        log(f'  MaxDD: {results["maxdd"]:.1f}%')
        log(f'  Sharpe: {results["sharpe"]:.3f}')
        log(f'  Sortino: {results["sortino"]:.3f}')
        log(f'  WR: {results["wr"]:.1f}%')
        log(f'  PF: {results["pf"]:.2f}')
        log(f'  Trades: {results["trades"]}')
        log(f'  Bull Sharpe: {results["bull_sharpe"]:.3f}')
        log(f'  Bear Sharpe: {results["bear_sharpe"]:.3f}')
        log(f'  Regime Gap: {results["regime_gap"]:.3f} {"PASS" if results["regime_gap"]<0.50 else "FAIL"}')
        log(f'  Final NAV: ${results["final_nav"]:,.0f}')
        
        for yr in results['yearly']:
            m = '+' if yr['ret']>0 else ''
            log(f'    {yr["year"]}: {m}{yr["ret"]:.1f}%')
        
        py = sum(1 for yr in results['yearly'] if yr['ret']>0)
        log(f'  {py}/{len(results["yearly"])} years profitable')
        
        pp, pm, ps2 = perm_test(signals, prices, results['sharpe'], n=200, max_pos=10, hold_days=hold)
        log(f'\n  PERM p={pp:.3f} {"PASS" if pp<0.05 else "FAIL"}')
        log(f'  Null Sharpe: {pm:.3f} ± {ps2:.3f}')
        log(f'  Observed: {results["sharpe"]:.3f}')
        
        results['perm_p'] = pp
        with open(f'{OUTPUT_DIR}/results_{hold}d.json','w') as f:
            json.dump(results, f, indent=2, default=str)
    
    log('\n' + '='*60)
    log('COMPLETE')
    log('='*60)

if __name__ == '__main__':
    main()
