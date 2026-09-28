#!/usr/bin/env python3
"""
Stat-Arb Pairs Trading v1 — Market-Neutral Sector ETF Pairs
============================================================
Universe: 11 sector ETFs + SPY, QQQ, IWM
Method: Rolling cointegration → z-score mean reversion → walk-forward OOT
Capital: $645, OOT: 2022-01-01 to 2026-07-25
"""

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
import json, os, warnings, traceback
from datetime import datetime
warnings.filterwarnings('ignore')

UNIVERSE = ['XLK','XLF','XLE','XLV','XLC','XLI','XLY','XLP','XLU','XLRE','XLB']
EXTRAS = ['SPY','QQQ','IWM']
ALL_TICKERS = UNIVERSE + EXTRAS
TRAIN_WINDOW = 252
OOT_START = '2022-01-01'
OOT_END = '2026-07-25'
CAPITAL = 645.0
COINT_PVALUE = 0.05
MAX_PAIRS = 3
ALLOC_PER_PAIR = 0.30
SLIPPAGE_BPS = 5
N_PERMS = 100
RESULTS_DIR = '/home/nick/Lvl3Quant/research/findings'
os.makedirs(RESULTS_DIR, exist_ok=True)

def download_data():
    print(f"Downloading {len(ALL_TICKERS)} tickers...")
    data = yf.download(ALL_TICKERS, start='2018-01-01', end=OOT_END, auto_adjust=True, progress=False)
    prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    prices = prices.dropna(how='all')
    print(f"Got {len(prices)} days, {prices.shape[1]} tickers")
    return prices

def compute_half_life(spread):
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    if len(spread_lag) < 10: return 999
    try:
        beta = np.polyfit(spread_lag, spread_diff, 1)[0]
        return max(1, min(-np.log(2)/beta, 999)) if beta < 0 else 999
    except: return 999

def find_cointegrated_pairs(prices_window, p_threshold=COINT_PVALUE):
    tickers = prices_window.columns.tolist()
    pairs = []
    for t1, t2 in combinations(tickers, 2):
        s1, s2 = prices_window[t1].dropna(), prices_window[t2].dropna()
        common = s1.index.intersection(s2.index)
        if len(common) < 120: continue
        s1, s2 = s1.loc[common], s2.loc[common]
        try:
            _, pvalue, _ = coint(s1.values, s2.values)
            if pvalue < p_threshold:
                beta = np.polyfit(s2.values, s1.values, 1)[0]
                spread = s1.values - beta * s2.values
                adf_stat, adf_p, *_ = adfuller(spread, maxlag=10)
                if adf_p < 0.05:
                    pairs.append({'ticker1':t1,'ticker2':t2,'coint_pvalue':pvalue,
                                  'adf_pvalue':adf_p,'beta':beta,
                                  'half_life':compute_half_life(spread),
                                  'spread_mean':np.mean(spread),'spread_std':np.std(spread)})
        except: continue
    pairs.sort(key=lambda x: x['coint_pvalue'])
    return pairs

def run_pairs_backtest(prices, variant='A', seed=42):
    params = {'A':(2.0,0.5,4.0,False,False), 'B':(1.5,0.3,3.5,False,False),
              'C':(2.5,0.8,5.0,False,False), 'D':(2.0,0.5,4.0,True,False),
              'E':(2.0,0.5,4.0,False,True)}
    entry_z, exit_z, stop_z, dynamic_hedge, vol_weight = params.get(variant, params['A'])

    oot_dates = prices.index[prices.index >= OOT_START]
    if len(oot_dates) < 40: return None

    equity = CAPITAL
    equity_curve = []
    trades = []
    open_positions = []
    current_pairs = []

    for i, date in enumerate(oot_dates):
        date_idx = prices.index.get_loc(date)
        if date_idx < TRAIN_WINDOW:
            equity_curve.append(equity)
            continue

        train_prices = prices.iloc[date_idx-TRAIN_WINDOW:date_idx]
        today_prices = prices.iloc[date_idx]

        if i % 5 == 0:
            current_pairs = find_cointegrated_pairs(train_prices)

        # Update open positions
        to_close = []
        for pi, pos in enumerate(open_positions):
            t1, t2 = pos['ticker1'], pos['ticker2']
            if t1 not in today_prices.index or t2 not in today_prices.index: continue
            p1, p2 = today_prices[t1], today_prices[t2]
            if pd.isna(p1) or pd.isna(p2): continue

            if dynamic_hedge:
                recent = prices.iloc[max(0,date_idx-60):date_idx]
                if t1 in recent.columns and t2 in recent.columns:
                    s1r, s2r = recent[t1].dropna(), recent[t2].dropna()
                    c = s1r.index.intersection(s2r.index)
                    if len(c) > 20:
                        pos['beta'] = np.polyfit(s2r.loc[c].values, s1r.loc[c].values, 1)[0]

            spread_now = p1 - pos['beta'] * p2
            z_now = (spread_now - pos['spread_mean']) / pos['spread_std'] if pos['spread_std'] > 0 else 0
            days_held = (date - pos['entry_date']).days

            should_close = False
            if pos['direction'] == 'long_spread':
                if z_now >= -exit_z or z_now >= 0 or z_now < -stop_z: should_close = True
            else:
                if z_now <= exit_z or z_now <= 0 or z_now > stop_z: should_close = True
            if days_held > max(20, 2*pos.get('half_life',20)): should_close = True

            if should_close:
                leg1_pnl = pos['leg1_shares'] * (p1 - pos['leg1_entry'])
                leg2_pnl = pos['leg2_shares'] * (p2 - pos['leg2_entry'])
                exit_cost = (abs(pos['leg1_shares'])*p1 + abs(pos['leg2_shares'])*p2) * SLIPPAGE_BPS/10000
                total_pnl = leg1_pnl + leg2_pnl - exit_cost - pos['entry_cost']
                equity += total_pnl
                trades.append({'entry_date':pos['entry_date'].strftime('%Y-%m-%d'),
                              'exit_date':date.strftime('%Y-%m-%d'),
                              'pair':f"{t1}/{t2}",'direction':pos['direction'],
                              'pnl':total_pnl,'days_held':days_held})
                to_close.append(pi)

        for idx in sorted(to_close, reverse=True): open_positions.pop(idx)

        # Open new positions
        if len(open_positions) < MAX_PAIRS and current_pairs:
            for pair_info in current_pairs[:5]:
                if len(open_positions) >= MAX_PAIRS: break
                t1, t2 = pair_info['ticker1'], pair_info['ticker2']
                active = {p['ticker1'] for p in open_positions} | {p['ticker2'] for p in open_positions}
                if t1 in active or t2 in active: continue
                if t1 not in today_prices.index or t2 not in today_prices.index: continue
                p1, p2 = today_prices[t1], today_prices[t2]
                if pd.isna(p1) or pd.isna(p2): continue

                spread_now = p1 - pair_info['beta'] * p2
                z_now = (spread_now - pair_info['spread_mean']) / pair_info['spread_std'] if pair_info['spread_std'] > 0 else 0

                direction = None
                if z_now > entry_z: direction = 'short_spread'
                elif z_now < -entry_z: direction = 'long_spread'
                if not direction: continue

                alloc = equity * ALLOC_PER_PAIR
                if vol_weight and pair_info['half_life'] < 999:
                    alloc *= min(2.0, 20.0/max(pair_info['half_life'],5))
                alloc = min(alloc, equity*0.40)

                if direction == 'long_spread':
                    l1s, l2s = alloc/(2*p1), -alloc/(2*p2)
                else:
                    l1s, l2s = -alloc/(2*p1), alloc/(2*p2)

                ec = (abs(l1s)*p1 + abs(l2s)*p2) * SLIPPAGE_BPS/10000
                open_positions.append({'ticker1':t1,'ticker2':t2,'beta':pair_info['beta'],
                    'spread_mean':pair_info['spread_mean'],'spread_std':pair_info['spread_std'],
                    'half_life':pair_info.get('half_life',20),'direction':direction,
                    'entry_date':date,'entry_z':z_now,'leg1_shares':l1s,'leg1_entry':p1,
                    'leg2_shares':l2s,'leg2_entry':p2,'entry_cost':ec})

        equity_curve.append(equity)

    if not equity_curve: return None
    ea = np.array(equity_curve)
    rets = np.diff(ea)/ea[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 40 or np.std(rets) == 0: return None

    sharpe = np.mean(rets)/np.std(rets)*np.sqrt(252)
    neg = rets[rets<0]
    sortino = np.mean(rets)/(np.std(neg)*np.sqrt(252)) if len(neg)>0 and np.std(neg)>0 else 0
    cagr = (ea[-1]/CAPITAL)**(252/len(rets))-1
    peak = np.maximum.accumulate(ea)
    mdd = np.min((ea-peak)/peak)*100

    if trades:
        pnls = [t['pnl'] for t in trades]
        wr = sum(1 for p in pnls if p>0)/len(pnls)*100
        wins = [p for p in pnls if p>0]
        losses = [abs(p) for p in pnls if p<0]
        pf = sum(wins)/sum(losses) if losses else 999
    else:
        wr, pf = 0, 0

    return {'sharpe':round(sharpe,3),'sortino':round(sortino,3),'cagr':round(cagr*100,2),
            'mdd':round(mdd,2),'wr':round(wr,2),'pf':round(pf,3),'n_trades':len(trades),
            'n_days':len(rets),'final_equity':round(ea[-1],2),
            'equity_curve':ea.tolist(),
            'oot_dates':[d.strftime('%Y-%m-%d') for d in oot_dates[:len(ea)]],
            'trades':trades}

def regime_stratify(metrics, prices):
    if 'SPY' not in prices.columns: return {'regime_balance_ok':False}
    spy = prices['SPY'].dropna()
    oot_spy = spy[spy.index >= OOT_START]
    spy_rets = oot_spy.pct_change().dropna()
    ea = np.array(metrics['equity_curve'])
    oot_dates = pd.to_datetime(metrics['oot_dates'])
    strat_rets = pd.Series(np.diff(ea)/ea[:-1], index=oot_dates[1:len(ea)])

    green = spy_rets[spy_rets>0.001].index
    red = spy_rets[spy_rets<-0.001].index
    flat = spy_rets[(spy_rets>=-0.001)&(spy_rets<=0.001)].index

    def ss(rs, ds):
        s = rs[rs.index.isin(ds)]
        if len(s)<10 or np.std(s)==0: return 0.0, len(s)
        return round(np.mean(s)/np.std(s)*np.sqrt(252),3), len(s)

    sg, ng = ss(strat_rets, green)
    sr, nr = ss(strat_rets, red)
    sf, nf = ss(strat_rets, flat)
    mx = max(abs(sg), abs(sr), 0.001)
    gap = abs(sg-sr)/mx
    return {'sharpe_green':sg,'n_green':ng,'sharpe_red':sr,'n_red':nr,
            'sharpe_flat':sf,'n_flat':nf,'regime_gap':round(gap,3),
            'regime_balance_ok':gap<=0.50}

def permutation_test(prices, variant, n_perms=N_PERMS):
    actual = run_pairs_backtest(prices, variant=variant)
    if not actual: return None, None
    actual_sharpe = actual['sharpe']
    perm_sharpes = []
    for i in range(n_perms):
        shuffled = prices.copy()
        for col in shuffled.columns:
            r = shuffled[col].pct_change().dropna()
            sr = r.sample(frac=1, random_state=i).values
            base = shuffled[col].iloc[0]
            np_arr = [base]
            for rv in sr: np_arr.append(np_arr[-1]*(1+rv))
            shuffled[col] = np_arr[:len(shuffled)]
        pr = run_pairs_backtest(shuffled, variant=variant, seed=i)
        if pr: perm_sharpes.append(pr['sharpe'])
    if not perm_sharpes: return actual, {'p_value':1.0,'significant':False}
    pv = np.mean([s>=actual_sharpe for s in perm_sharpes])
    return actual, {'actual_sharpe':actual_sharpe,'perm_mean':round(np.mean(perm_sharpes),3),
                    'perm_std':round(np.std(perm_sharpes),3),'p_value':round(pv,3),
                    'significant':pv<0.05}

def main():
    print("="*70)
    print("STAT-ARB PAIRS TRADING v1")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*70)

    prices = download_data()

    print("\n--- Cointegration Scan ---")
    full_pairs = find_cointegrated_pairs(prices)
    print(f"Found {len(full_pairs)} cointegrated pairs")
    for p in full_pairs[:10]:
        print(f"  {p['ticker1']}/{p['ticker2']}: p={p['coint_pvalue']:.4f}, hl={p['half_life']:.1f}d")

    variants = [('A','Classic Z-Score (2.0/0.5)'),('B','Tight Z-Score (1.5/0.3)'),
                ('C','Wide Z-Score (2.5/0.8)'),('D','Dynamic Hedge Ratio'),
                ('E','Vol-Weighted Half-Life')]

    results = {}
    for v, label in variants:
        print(f"\n{'='*60}\nVariant {v}: {label}\n{'='*60}")
        try:
            r = run_pairs_backtest(prices, variant=v)
            if r:
                regime = regime_stratify(r, prices)
                print(f"  Sharpe={r['sharpe']}, Sortino={r['sortino']}, CAGR={r['cagr']}%, "
                      f"Trades={r['n_trades']}, WR={r['wr']}%, PF={r['pf']}, MDD={r['mdd']}%")
                print(f"  Regime: green={regime['sharpe_green']}, red={regime['sharpe_red']}, "
                      f"gap={regime['regime_gap']}, pass={regime['regime_balance_ok']}")
                rc = {k:v for k,v in r.items() if k not in ('equity_curve','oot_dates','trades')}
                results[v] = {'variant':v,'label':label,'metrics':rc,'regime':regime,
                             'trades_sample':r.get('trades',[])[:10]}
            else:
                print(f"  No valid results")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    # Perm test on best
    if results:
        best = max(results, key=lambda v: results[v]['metrics']['sharpe'])
        print(f"\n--- Perm Test on Variant {best} ---")
        _, perm = permutation_test(prices, variant=best, n_perms=N_PERMS)
        if perm:
            results[best]['permutation'] = perm
            print(f"  p={perm['p_value']}, sig={perm['significant']}")

    final = {'strategy':'Stat-Arb Pairs Trading v1','run_date':datetime.now().isoformat(),
             'capital':CAPITAL,'universe':ALL_TICKERS,'oot_start':OOT_START,'oot_end':OOT_END,
             'n_cointegrated_pairs':len(full_pairs),
             'top_pairs':[{'pair':f"{p['ticker1']}/{p['ticker2']}",
                          'coint_p':round(p['coint_pvalue'],4),
                          'half_life':round(p['half_life'],1)} for p in full_pairs[:10]],
             'variants':results}

    out = os.path.join(RESULTS_DIR, 'stat_arb_pairs_v1_results.json')
    with open(out,'w') as f: json.dump(final, f, indent=2, default=str)

    print(f"\n{'='*70}\nRESULTS SAVED: {out}\n{'='*70}")
    for v,r in sorted(results.items()):
        m = r['metrics']
        rg = r['regime']
        ok = "PASS" if rg.get('regime_balance_ok') else "FAIL"
        print(f"  {v} ({r['label']}): Sharpe={m['sharpe']}, CAGR={m['cagr']}%, "
              f"WR={m['wr']}%, Regime={ok}(gap={rg.get('regime_gap','?')})")
    print(f"Completed: {datetime.now().isoformat()}")

if __name__ == '__main__':
    main()
