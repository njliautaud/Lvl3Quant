#!/usr/bin/env python3
"""
LGBM-Directed Premium Selling on Sector ETFs v1
================================================
KEY INSIGHT: All our call-buying strategies lose to theta. But iron condors 
(Sharpe 3.55) and premium selling work great. What if we combine:
  - Our validated LGBM sector ranking (strongest signal)
  - Premium selling (most profitable strategy type)

Strategy: Sell bull put spreads on top-ranked sectors (collect premium with 
directional edge) and bear call spreads on bottom-ranked sectors.

This is CREDIT-based, not DEBIT-based. We COLLECT theta instead of fighting it.

Variants:
A: Bull put spreads on top-2 sectors (monthly)
B: Bear call spreads on bottom-2 sectors (monthly)  
C: Iron condor on top sector (bull put + bear call on same)
D: Directional: bull put on top-2 + bear call on bottom-2 (paired)
E: VIX-filtered: only sell when VIX > 16 (premium is higher)
F: Kelly-sized with LGBM confidence
"""

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
from datetime import datetime
import os

START_CAPITAL = 645.0
OOT_START = '2022-01-01'
SECTOR_ETFS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
COMMISSION_RT = 2.60  # RT spread commission on RH
HAIRCUT = 0.85

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S-K, 0)
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(K-S, 0)
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def price_bull_put_spread(S, otm_pct, width_pct, dte, vol, r=0.05):
    """Sell higher put, buy lower put → collect credit"""
    T = dte / 365.0
    K_short = S * (1 - otm_pct)  # short put (closer to ATM)
    K_long = S * (1 - otm_pct - width_pct)  # long put (further OTM)
    
    short_put = bs_put(S, K_short, T, r, vol) * 100
    long_put = bs_put(S, K_long, T, r, vol) * 100
    
    credit = (short_put - long_put) * HAIRCUT  # what we collect
    max_loss = (K_short - K_long) * 100 - credit  # spread width - credit
    max_profit = credit - COMMISSION_RT
    
    return credit, max_loss, max_profit, K_short, K_long

def price_bear_call_spread(S, otm_pct, width_pct, dte, vol, r=0.05):
    """Sell lower call, buy higher call → collect credit"""
    T = dte / 365.0
    K_short = S * (1 + otm_pct)  # short call (closer to ATM)
    K_long = S * (1 + otm_pct + width_pct)  # long call (further OTM)
    
    short_call = bs_call(S, K_short, T, r, vol) * 100
    long_call = bs_call(S, K_long, T, r, vol) * 100
    
    credit = (short_call - long_call) * HAIRCUT
    max_loss = (K_long - K_short) * 100 - credit
    max_profit = credit - COMMISSION_RT
    
    return credit, max_loss, max_profit, K_short, K_long

def settle_bull_put(S_exit, K_short, K_long, credit_received):
    """Settle bull put spread at expiry"""
    if S_exit >= K_short:
        pnl = credit_received - COMMISSION_RT  # full profit
    elif S_exit <= K_long:
        pnl = credit_received - (K_short - K_long) * 100 - COMMISSION_RT  # max loss
    else:
        pnl = credit_received - (K_short - S_exit) * 100 - COMMISSION_RT  # partial loss
    return pnl

def settle_bear_call(S_exit, K_short, K_long, credit_received):
    """Settle bear call spread at expiry"""
    if S_exit <= K_short:
        pnl = credit_received - COMMISSION_RT  # full profit
    elif S_exit >= K_long:
        pnl = credit_received - (K_long - K_short) * 100 - COMMISSION_RT  # max loss
    else:
        pnl = credit_received - (S_exit - K_short) * 100 - COMMISSION_RT  # partial loss
    return pnl

def load_data():
    price_df = pd.read_parquet('research/cache/sector_etf_daily_data.parquet')
    price_df.index = pd.to_datetime(price_df.index)
    
    vix_df = pd.read_parquet('data/cache/regime_macro/VIX.parquet')
    vix_df.index = pd.to_datetime(vix_df.index)
    vix_aligned = vix_df['close'].reindex(price_df.index, method='ffill').fillna(18)
    
    return price_df.sort_index(), vix_aligned

def precompute_scores(price_df):
    ret5 = price_df / price_df.shift(5) - 1
    ret21 = price_df / price_df.shift(21) - 1
    ret63 = price_df / price_df.shift(63) - 1
    score = ret5*20 + ret21*35 + ret63*15
    
    # Volatility for pricing
    log_ret = np.log(price_df / price_df.shift(1))
    vol_21d = log_ret.rolling(21).std() * np.sqrt(252)
    
    return score, vol_21d

def run_variant(price_df, scores, vol_df, vix, name, cfg):
    dates = price_df.index
    oot_start = np.argmax(dates >= pd.Timestamp(OOT_START))
    
    top_n = cfg.get('top_n', 2)
    bottom_n = cfg.get('bottom_n', 0)
    rebal_freq = cfg.get('rebal_freq', 21)  # monthly
    hold_days = cfg.get('hold_days', 21)  # hold to near-expiry
    dte = cfg.get('dte', 28)
    otm_pct = cfg.get('otm_pct', 0.04)  # 4% OTM
    width_pct = cfg.get('width_pct', 0.03)  # 3% wide
    vix_min = cfg.get('vix_min', 0)  # min VIX to sell premium
    use_iron_condor = cfg.get('use_iron_condor', False)
    max_risk_pct = cfg.get('max_risk_pct', 0.30)  # max 30% of capital at risk per trade
    
    capital = START_CAPITAL
    equity = [capital]
    trades = []
    positions = []  # list of dicts
    
    for idx in range(oot_start, len(dates)):
        date = dates[idx]
        vix_val = float(vix.iloc[idx]) if idx < len(vix) else 18
        
        # Check exits (hold to expiry approximation)
        new_positions = []
        for pos in positions:
            days_held = (date - pos['entry_date']).days
            if days_held >= hold_days:
                # Settle at current price
                exit_p = price_df.iloc[idx].get(pos['etf'], np.nan)
                if np.isnan(exit_p): 
                    new_positions.append(pos)
                    continue
                
                if pos['type'] == 'bull_put':
                    pnl = settle_bull_put(exit_p, pos['K_short'], pos['K_long'], pos['credit'])
                elif pos['type'] == 'bear_call':
                    pnl = settle_bear_call(exit_p, pos['K_short'], pos['K_long'], pos['credit'])
                else:
                    pnl = 0
                
                capital += pnl
                # Release collateral
                capital += pos['collateral']
                trades.append({
                    'pnl': pnl, 'type': pos['type'], 'etf': pos['etf'],
                    'vix': pos.get('entry_vix', 18), 'days': days_held,
                    'credit': pos['credit'], 'exit_price': exit_p,
                    'entry_price': pos['entry_price'],
                })
            else:
                new_positions.append(pos)
        positions = new_positions
        
        # Track equity (capital + unreturned collateral in positions)
        total_collateral = sum(p['collateral'] for p in positions)
        equity.append(capital + total_collateral)
        
        # Rebalance check
        if (idx - oot_start) % rebal_freq != 0:
            continue
        
        # VIX filter
        if vix_val < vix_min:
            continue
        
        # Get rankings
        row = scores.iloc[idx]
        valid = row.dropna()
        if len(valid) < 5:
            continue
        ranked = valid.sort_values(ascending=False)
        
        # SELL BULL PUT SPREADS on top-ranked (bullish) sectors
        entries = 0
        for etf in ranked.index[:top_n]:
            price = price_df.iloc[idx].get(etf, np.nan)
            vol = vol_df.iloc[idx].get(etf, np.nan)
            if np.isnan(price) or np.isnan(vol) or vol <= 0:
                continue
            vol = max(0.10, min(vol, 0.60))
            
            credit, max_loss, max_profit, K_s, K_l = price_bull_put_spread(
                price, otm_pct, width_pct, dte, vol
            )
            
            if credit < 3 or max_profit < 1:  # not worth it
                continue
            
            # Collateral requirement = max_loss (spread width - credit)
            collateral = max_loss
            if collateral > capital * max_risk_pct or collateral > capital - 50:
                continue
            
            capital -= collateral  # lock up collateral
            positions.append({
                'entry_date': date, 'etf': etf, 'type': 'bull_put',
                'entry_price': price, 'K_short': K_s, 'K_long': K_l,
                'credit': credit, 'collateral': collateral,
                'entry_vix': vix_val,
            })
            entries += 1
        
        # SELL BEAR CALL SPREADS on bottom-ranked (bearish) sectors
        if bottom_n > 0:
            bottom_etfs = ranked.index[-bottom_n:]
            for etf in bottom_etfs:
                price = price_df.iloc[idx].get(etf, np.nan)
                vol = vol_df.iloc[idx].get(etf, np.nan)
                if np.isnan(price) or np.isnan(vol) or vol <= 0:
                    continue
                vol = max(0.10, min(vol, 0.60))
                
                credit, max_loss, max_profit, K_s, K_l = price_bear_call_spread(
                    price, otm_pct, width_pct, dte, vol
                )
                
                if credit < 3 or max_profit < 1:
                    continue
                
                collateral = max_loss
                if collateral > capital * max_risk_pct or collateral > capital - 50:
                    continue
                
                capital -= collateral
                positions.append({
                    'entry_date': date, 'etf': etf, 'type': 'bear_call',
                    'entry_price': price, 'K_short': K_s, 'K_long': K_l,
                    'credit': credit, 'collateral': collateral,
                    'entry_vix': vix_val,
                })
        
        # IRON CONDOR on top sector
        if use_iron_condor and len(ranked) > 0:
            etf = ranked.index[0]
            price = price_df.iloc[idx].get(etf, np.nan)
            vol = vol_df.iloc[idx].get(etf, np.nan)
            if not np.isnan(price) and not np.isnan(vol) and vol > 0:
                vol = max(0.10, min(vol, 0.60))
                
                # Bull put side
                bp_credit, bp_loss, bp_profit, bp_Ks, bp_Kl = price_bull_put_spread(
                    price, otm_pct, width_pct, dte, vol
                )
                # Bear call side
                bc_credit, bc_loss, bc_profit, bc_Ks, bc_Kl = price_bear_call_spread(
                    price, otm_pct, width_pct, dte, vol
                )
                
                total_credit = bp_credit + bc_credit
                total_collateral = max(bp_loss, bc_loss)  # iron condor collateral = max of one side
                
                if total_credit > 5 and total_collateral <= capital * max_risk_pct:
                    capital -= total_collateral
                    positions.append({
                        'entry_date': date, 'etf': etf, 'type': 'bull_put',
                        'entry_price': price, 'K_short': bp_Ks, 'K_long': bp_Kl,
                        'credit': bp_credit, 'collateral': total_collateral / 2,
                        'entry_vix': vix_val,
                    })
                    positions.append({
                        'entry_date': date, 'etf': etf, 'type': 'bear_call',
                        'entry_price': price, 'K_short': bc_Ks, 'K_long': bc_Kl,
                        'credit': bc_credit, 'collateral': total_collateral / 2,
                        'entry_vix': vix_val,
                    })
    
    # Close remaining at end
    for pos in positions:
        exit_p = price_df.iloc[-1].get(pos['etf'], np.nan)
        if np.isnan(exit_p): continue
        days_held = (dates[-1] - pos['entry_date']).days
        if pos['type'] == 'bull_put':
            pnl = settle_bull_put(exit_p, pos['K_short'], pos['K_long'], pos['credit'])
        else:
            pnl = settle_bear_call(exit_p, pos['K_short'], pos['K_long'], pos['credit'])
        capital += pnl + pos['collateral']
        trades.append({'pnl': pnl, 'type': pos['type'], 'etf': pos['etf'], 'vix': 0, 'days': days_held,
                       'credit': pos['credit'], 'exit_price': exit_p, 'entry_price': pos['entry_price']})
    
    equity.append(capital)
    
    if len(trades) < 10:
        return {'variant': name, 'trades': len(trades), 'sharpe': -999, 'skip': True,
                'reason': f'Only {len(trades)} trades'}
    
    # Metrics
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins)/len(pnls)*100
    pf = wins.sum()/abs(losses.sum()) if len(losses)>0 and losses.sum()!=0 else 999
    
    eq = np.array(equity)
    rets = np.diff(eq)/(np.abs(eq[:-1])+1e-10)
    rets = rets[rets!=0]
    ann_f = np.sqrt(252/max(rebal_freq,1))
    sharpe = np.mean(rets)/(np.std(rets)+1e-10)*ann_f if len(rets)>5 else 0
    neg = rets[rets<0]
    sortino = np.mean(rets)/(np.std(neg)+1e-10)*ann_f if len(neg)>2 else sharpe
    
    pk = eq[0]; mdd = 0
    for v in eq:
        pk = max(pk,v)
        mdd = min(mdd, (v-pk)/(pk+1e-10))
    
    years = len(equity)/252
    cagr = (max(eq[-1],1)/START_CAPITAL)**(1/max(years,0.1))-1
    
    # Regime gap
    avg_sector = price_df[SECTOR_ETFS].mean(axis=1)
    mkt_ret = avg_sector.pct_change()
    green_pnl, red_pnl = [], []
    for t in trades:
        # Use entry_price date approximation
        entry_idx = dates.get_indexer([t.get('entry_date', dates[0])], method='nearest')[0] if 'entry_date' in t else 0
        d = dates[min(entry_idx, len(dates)-1)]
        if d in mkt_ret.index:
            r = mkt_ret.loc[d]
            if r > 0: green_pnl.append(t['pnl'])
            else: red_pnl.append(t['pnl'])
    
    regime_gap = 999
    if green_pnl and red_pnl:
        sg = np.mean(green_pnl)/(np.std(green_pnl)+1e-10)
        sr = np.mean(red_pnl)/(np.std(red_pnl)+1e-10)
        regime_gap = abs(sg-sr)/(max(abs(sg),abs(sr))+1e-10)
    
    # Win/loss by type
    bp_trades = [t for t in trades if t['type']=='bull_put']
    bc_trades = [t for t in trades if t['type']=='bear_call']
    bp_wr = len([t for t in bp_trades if t['pnl']>0])/max(len(bp_trades),1)*100
    bc_wr = len([t for t in bc_trades if t['pnl']>0])/max(len(bc_trades),1)*100
    
    return {
        'variant': name, 'trades': len(trades),
        'final_equity': round(eq[-1], 2), 'cagr': round(cagr*100,1),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 2),
        'wr': round(wr,1), 'pf': round(pf, 2), 'max_dd': round(mdd*100,1),
        'regime_gap': round(regime_gap, 3),
        'bp_trades': len(bp_trades), 'bp_wr': round(bp_wr,1),
        'bc_trades': len(bc_trades), 'bc_wr': round(bc_wr,1),
        'avg_credit': round(np.mean([t['credit'] for t in trades]),1),
        'avg_pnl': round(np.mean(pnls),1),
        'trade_pnls': pnls.tolist(),
    }

def permutation_test(pnls, n=1000):
    obs = np.mean(pnls)
    p = np.array(pnls)
    count = sum(1 for _ in range(n) if np.mean(p*np.random.choice([-1,1],len(p))) >= obs)
    return count/n

def random_baseline(price_df, vol_df, n_sims=200, rebal_freq=21, hold_days=21):
    dates = price_df.index
    oot_start = np.argmax(dates >= pd.Timestamp(OOT_START))
    etfs = list(price_df.columns)
    sharpes = []
    for sim in range(n_sims):
        np.random.seed(sim*31)
        capital = START_CAPITAL
        eq = [capital]
        for idx in range(oot_start, len(dates), rebal_freq):
            if idx+hold_days >= len(dates): break
            picks = np.random.choice(etfs, 2, replace=False)
            for etf in picks:
                ep = price_df.iloc[idx].get(etf, np.nan)
                xp = price_df.iloc[min(idx+hold_days, len(dates)-1)].get(etf, np.nan)
                vol = vol_df.iloc[idx].get(etf, np.nan)
                if np.isnan(ep) or np.isnan(xp) or np.isnan(vol): continue
                vol = max(0.10, min(vol, 0.60))
                credit, ml, mp, Ks, Kl = price_bull_put_spread(ep, 0.04, 0.03, 28, vol)
                if credit < 3 or ml > capital*0.30: continue
                pnl = settle_bull_put(xp, Ks, Kl, credit)
                capital += pnl
            eq.append(capital)
        r = np.diff(eq)/(np.abs(eq[:-1])+1e-10)
        r = r[r!=0]
        if len(r)>3:
            sharpes.append(np.mean(r)/(np.std(r)+1e-10)*np.sqrt(12))
    return np.mean(sharpes) if sharpes else 0

def main():
    t0 = datetime.now()
    print("="*70)
    print("  LGBM-DIRECTED PREMIUM SELLING v1")
    print("  Sell credit spreads on LGBM-ranked sectors (theta harvesting)")
    print("="*70)
    
    price_df, vix = load_data()
    scores, vol_df = precompute_scores(price_df)
    print(f"  {len(price_df.columns)} sectors, {len(price_df)} days")
    
    variants = {
        'A_BullPut_Top2': {'top_n': 2, 'bottom_n': 0, 'rebal_freq': 21, 'hold_days': 21,
                           'otm_pct': 0.04, 'width_pct': 0.03},
        'B_BearCall_Bot2': {'top_n': 0, 'bottom_n': 2, 'rebal_freq': 21, 'hold_days': 21,
                            'otm_pct': 0.04, 'width_pct': 0.03},
        'C_Paired_2x2': {'top_n': 2, 'bottom_n': 2, 'rebal_freq': 21, 'hold_days': 21,
                         'otm_pct': 0.04, 'width_pct': 0.03},
        'D_IronCondor_Top1': {'top_n': 0, 'bottom_n': 0, 'use_iron_condor': True,
                               'rebal_freq': 21, 'hold_days': 21, 'otm_pct': 0.04, 'width_pct': 0.03},
        'E_VixFiltered': {'top_n': 2, 'bottom_n': 2, 'rebal_freq': 21, 'hold_days': 21,
                          'otm_pct': 0.04, 'width_pct': 0.03, 'vix_min': 16},
        'F_Wider_Spread': {'top_n': 2, 'bottom_n': 2, 'rebal_freq': 21, 'hold_days': 21,
                           'otm_pct': 0.05, 'width_pct': 0.05},
    }
    
    results = []
    for name, cfg in variants.items():
        print(f"\n  {name}...", end=" ")
        try:
            r = run_variant(price_df, scores, vol_df, vix, name, cfg)
            if r and not r.get('skip'):
                results.append(r)
                print(f"Sharpe {r['sharpe']:.3f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | "
                      f"MDD {r['max_dd']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t "
                      f"[BP:{r['bp_trades']}t {r['bp_wr']:.0f}%WR | BC:{r['bc_trades']}t {r['bc_wr']:.0f}%WR] "
                      f"avg credit ${r['avg_credit']:.0f}")
            elif r:
                print(f"SKIP: {r.get('reason')}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
    
    if not results:
        print("\n❌ ALL FAILED")
        return
    
    print(f"\n  Random baseline (200 sims)...", end=" ")
    rand_s = random_baseline(price_df, vol_df)
    print(f"Sharpe {rand_s:.3f}")
    
    print(f"\n{'='*70}")
    print("  5-GATE EVALUATION")
    print(f"{'='*70}")
    
    for r in sorted(results, key=lambda x: -x['sharpe']):
        g = 0
        g1 = r['sharpe'] > 0.5; g += g1
        pv = permutation_test(r['trade_pnls']) if len(r['trade_pnls'])>=10 else 1.0
        g2 = pv < 0.05; g += g2
        g3 = r['max_dd'] > -25; g += g3  # tighter for income strategies
        g4 = r['regime_gap'] < 0.50; g += g4
        g5 = r['sharpe'] > rand_s + 0.10; g += g5
        
        status = '✅ PASS' if g >= 4 else ('⚠️' if g >= 3 else '❌ FAIL')
        print(f"\n  {r['variant']} — {status} ({g}/5)")
        print(f"    Sharpe {r['sharpe']:.3f} | Sortino {r['sortino']:.2f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f}")
        print(f"    MDD {r['max_dd']:.1f}% | CAGR {r['cagr']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t")
        print(f"    Bull puts: {r['bp_trades']}t ({r['bp_wr']:.0f}% WR) | Bear calls: {r['bc_trades']}t ({r['bc_wr']:.0f}% WR)")
        print(f"    Avg credit: ${r['avg_credit']:.0f} | Avg P&L: ${r['avg_pnl']:.1f} | Regime gap: {r['regime_gap']:.3f}")
        print(f"    G1 Sharpe>0.5: {'✅' if g1 else '❌'} | G2 Perm: {'✅' if g2 else '❌'} (p={pv:.3f})")
        print(f"    G3 MDD>-25%: {'✅' if g3 else '❌'} | G4 Regime<0.50: {'✅' if g4 else '❌'} | G5 >Random: {'✅' if g5 else '❌'}")
    
    best = max(results, key=lambda x: x['sharpe'])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['variant']} Sharpe {best['sharpe']:.3f}")
    print(f"  Random: {rand_s:.3f}")
    print(f"  Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")
    
    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        exp = mlflow.set_experiment("lgbm_directed_premium_v1")
        with mlflow.start_run(run_name=f"premium_{datetime.now().strftime('%H%M')}"):
            mlflow.log_param("strategy", "lgbm_directed_premium_v1")
            mlflow.log_param("best_variant", best['variant'])
            mlflow.log_metric("best_sharpe", best['sharpe'])
            mlflow.log_metric("best_sortino", best['sortino'])
            mlflow.log_metric("best_wr", best['wr'])
            mlflow.log_metric("best_pf", best['pf'])
            mlflow.log_metric("best_mdd", best['max_dd'])
            mlflow.log_metric("random_sharpe", rand_s)
            for r in results:
                mlflow.log_metric(f"{r['variant']}_sharpe", r['sharpe'])
        print(f"  MLflow: {exp.name}")
    except Exception as e:
        print(f"  MLflow: {e}")

if __name__ == '__main__':
    main()
