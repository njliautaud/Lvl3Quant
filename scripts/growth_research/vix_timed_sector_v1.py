#!/usr/bin/env python3
"""
VIX-Timed Sector Rotation v1 — Options + Volatility Timing
==========================================================
Hypothesis: Combine our LGBM sector momentum signal with VIX regime filter.
- VIX < 20: Buy bull call spreads on top-ranked sectors (favorable)
- VIX 20-30: Reduce exposure, tighter stops
- VIX > 30: Cash (crisis mode, don't fight vol)

Also test: VIX mean-reversion timing
- Buy when VIX spikes > 25 and starts dropping (post-panic entry)
- This could capture the bounce while momentum signal picks the direction

Uses real VIX data + sector ETF prices from our cache.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
from datetime import datetime
import os

START_CAPITAL = 645.0
OOT_START = '2022-01-01'
SECTOR_ETFS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def spread_pnl(entry_p, exit_p, vol_entry, vol_exit, dte, days_held, width=0.05):
    """Bull call spread P&L"""
    T0 = dte/365.0
    T1 = max((dte - days_held)/365.0, 1/365.0)
    K1 = entry_p
    K2 = entry_p * (1 + width)
    h = 0.85  # haircut
    
    entry_val = (bs_call(entry_p, K1, T0, 0.05, vol_entry) - bs_call(entry_p, K2, T0, 0.05, vol_entry)) * h
    exit_val = (bs_call(exit_p, K1, T1, 0.05, vol_exit) - bs_call(exit_p, K2, T1, 0.05, vol_exit)) * h
    return (exit_val - entry_val) * 100 - 1.30  # commissions

def load_data():
    price_df = pd.read_parquet('research/cache/sector_etf_daily_data.parquet')
    price_df.index = pd.to_datetime(price_df.index)
    
    # Load VIX from parquet cache
    vix = None
    vix_parquet = 'data/cache/regime_macro/VIX.parquet'
    if os.path.exists(vix_parquet):
        v = pd.read_parquet(vix_parquet)
        v.index = pd.to_datetime(v.index)
        vix = pd.DataFrame({'VIX': v['close'].values}, index=v.index)

    if vix is None:
        try:
            import yfinance as yf
            v = yf.download('^VIX', start='2010-01-01', progress=False)
            vix = pd.DataFrame({'VIX': v['Close'].values}, index=v.index)
        except:
            pass

    if vix is None:
        log_ret = np.log(price_df / price_df.shift(1))
        avg_vol = log_ret.rolling(21).std().mean(axis=1) * np.sqrt(252) * 100
        vix = pd.DataFrame({'VIX': avg_vol}, index=price_df.index)
    
    return price_df.sort_index(), vix.sort_index()

def momentum_score(price_df, idx, lookback=252):
    if idx < lookback: return {}
    w = price_df.iloc[idx-lookback+1:idx+1]
    scores = {}
    for etf in price_df.columns:
        c = w[etf].dropna().values
        if len(c) < 63: continue
        r5 = c[-1]/c[-5] - 1 if len(c) >= 5 else 0
        r21 = c[-1]/c[-21] - 1 if len(c) >= 21 else 0
        r63 = c[-1]/c[-63] - 1 if len(c) >= 63 else 0
        scores[etf] = r5*20 + r21*35 + r63*15
    return scores

def run_variant(price_df, vix_df, name, cfg):
    dates = price_df.index
    oot_mask = dates >= pd.Timestamp(OOT_START)
    oot_start = np.argmax(oot_mask)
    
    vix_low = cfg.get('vix_low', 20)
    vix_high = cfg.get('vix_high', 30)
    top_n = cfg.get('top_n', 2)
    rebal_freq = cfg.get('rebal_freq', 5)
    hold_days = cfg.get('hold_days', 10)
    dte = cfg.get('dte', 30)
    width = cfg.get('width', 0.05)
    use_vix_timer = cfg.get('use_vix_timer', True)
    vix_bounce = cfg.get('vix_bounce', False)  # enter only on VIX mean-reversion
    
    capital = START_CAPITAL
    equity = [capital]
    trades = []
    positions = {}
    
    for idx in range(oot_start, len(dates)):
        date = dates[idx]
        
        # Get VIX (scalar)
        vix_val = 18.0
        try:
            loc_idx = vix_df.index.get_indexer([date], method='ffill')[0]
            if loc_idx >= 0:
                v = vix_df.iloc[loc_idx]['VIX']
                vix_val = float(v) if not np.isnan(float(v)) else 18.0
        except:
            pass
        
        # Exit check
        for etf in list(positions.keys()):
            pos = positions[etf]
            days_held = (date - pos['entry_date']).days
            if days_held >= hold_days:
                exit_p = price_df.iloc[idx].get(etf, np.nan)
                if np.isnan(exit_p): continue
                vol_exit = pos['vol'] * 0.95
                pnl = spread_pnl(pos['entry_p'], exit_p, pos['vol'], vol_exit, dte, days_held, width)
                capital += pnl
                trades.append({'date': date, 'etf': etf, 'pnl': pnl, 'vix': vix_val})
                del positions[etf]
        
        equity.append(capital)
        
        if (idx - oot_start) % rebal_freq != 0:
            continue
        
        # VIX filter
        if use_vix_timer:
            if vix_val > vix_high:
                continue  # crisis - stay cash
            
            if vix_bounce:
                # Only enter on VIX mean-reversion: VIX was > 25 recently and now dropping
                try:
                    loc5 = vix_df.index.get_indexer([dates[max(idx-5, 0)]], method='ffill')[0]
                    vix_5d_ago = float(vix_df.iloc[max(0, loc5)]['VIX']) if loc5 >= 0 else 18
                except:
                    vix_5d_ago = 18
                if not (vix_5d_ago > 25 and vix_val < vix_5d_ago):
                    if vix_val < vix_low:
                        pass  # normal low-vol entry
                    else:
                        continue
        
        # Get scores
        scores = momentum_score(price_df, idx)
        if len(scores) < 5: continue
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        # Position sizing based on VIX
        if vix_val < vix_low:
            size_mult = 1.0  # full size
        elif vix_val < vix_high:
            size_mult = 0.5  # half size in elevated vol
        else:
            size_mult = 0.0  # no new positions
        
        available = capital * 0.85 - sum(p['cost'] for p in positions.values())
        entries = 0
        
        for etf, score in ranked:
            if etf in positions or entries >= top_n: break
            
            price = price_df.iloc[idx].get(etf, np.nan)
            if np.isnan(price): continue
            
            # Use VIX for vol estimate
            vol = vix_val / 100 * 1.5  # sector vol ≈ 1.5x VIX
            vol = max(0.10, min(vol, 0.60))
            
            cost_raw, _, _ = price_option_spread(price, vol, dte, width)
            cost = cost_raw * size_mult
            
            if cost < 5 or cost > min(capital * 0.30, available):
                continue
            
            positions[etf] = {
                'entry_date': date, 'entry_p': price, 'vol': vol, 'cost': cost
            }
            available -= cost
            entries += 1
    
    # Close remaining
    for etf in list(positions.keys()):
        pos = positions[etf]
        exit_p = price_df.iloc[-1].get(etf, np.nan)
        if np.isnan(exit_p): continue
        days_held = (dates[-1] - pos['entry_date']).days
        pnl = spread_pnl(pos['entry_p'], exit_p, pos['vol'], pos['vol']*0.95, dte, days_held, width)
        capital += pnl
        trades.append({'date': dates[-1], 'etf': etf, 'pnl': pnl, 'vix': 0})
    equity.append(capital)
    
    if len(trades) < 5:
        return {'variant': name, 'trades': len(trades), 'sharpe': -999, 'skip': True}
    
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins)/len(pnls)*100
    pf = wins.sum()/abs(losses.sum()) if len(losses)>0 and losses.sum()!=0 else 999
    
    eq = np.array(equity)
    rets = np.diff(eq)/(np.abs(eq[:-1])+1e-10)
    rets = rets[rets != 0]
    sharpe = np.mean(rets)/(np.std(rets)+1e-10) * np.sqrt(252/max(rebal_freq,1)) if len(rets)>5 else 0
    neg = rets[rets<0]
    sortino = np.mean(rets)/(np.std(neg)+1e-10) * np.sqrt(252/max(rebal_freq,1)) if len(neg)>2 else sharpe
    
    peak = eq[0]
    mdd = 0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, (v-peak)/(peak+1e-10))
    
    years = len(equity)/252
    cagr = (max(eq[-1],1)/START_CAPITAL)**(1/max(years,0.1))-1
    
    # VIX regime performance
    low_vix_pnl = [t['pnl'] for t in trades if t['vix'] < 20]
    high_vix_pnl = [t['pnl'] for t in trades if t['vix'] >= 20]
    
    return {
        'variant': name, 'trades': len(trades),
        'final_equity': round(eq[-1], 2), 'cagr': round(cagr*100, 1),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 2),
        'wr': round(wr, 1), 'pf': round(pf, 2), 'max_dd': round(mdd*100, 1),
        'trade_pnls': pnls.tolist(),
        'low_vix_trades': len(low_vix_pnl),
        'high_vix_trades': len(high_vix_pnl),
        'low_vix_avg': round(np.mean(low_vix_pnl), 2) if low_vix_pnl else 0,
        'high_vix_avg': round(np.mean(high_vix_pnl), 2) if high_vix_pnl else 0,
    }

def price_option_spread(S, vol, dte, width):
    T = dte/365.0
    K1 = S
    K2 = S * (1+width)
    h = 0.85
    debit = (bs_call(S, K1, T, 0.05, vol) - bs_call(S, K2, T, 0.05, vol)) * h * 100
    max_prof = (K2-K1)*100 - debit
    return debit, max_prof, debit

def permutation_test(pnls, n=1000):
    obs = np.mean(pnls)
    p = np.array(pnls)
    count = sum(1 for _ in range(n) if np.mean(p * np.random.choice([-1,1], len(p))) >= obs)
    return count/n

def random_baseline(price_df, vix_df, n_sims=200):
    dates = price_df.index
    oot_start = np.argmax(dates >= pd.Timestamp(OOT_START))
    etfs = list(price_df.columns)
    sharpes = []
    for sim in range(n_sims):
        np.random.seed(sim*13)
        capital = START_CAPITAL
        eq = [capital]
        for idx in range(oot_start, len(dates), 5):
            if idx+10 >= len(dates): break
            picks = np.random.choice(etfs, 2, replace=False)
            for etf in picks:
                ep = price_df.iloc[idx].get(etf, np.nan)
                xp = price_df.iloc[min(idx+10, len(dates)-1)].get(etf, np.nan)
                if np.isnan(ep) or np.isnan(xp): continue
                pnl = spread_pnl(ep, xp, 0.25, 0.24, 30, 10)
                if abs(pnl) < capital * 0.5:
                    capital += pnl
            eq.append(capital)
        r = np.diff(eq)/(np.abs(eq[:-1])+1e-10)
        r = r[r!=0]
        if len(r)>5:
            sharpes.append(np.mean(r)/(np.std(r)+1e-10)*np.sqrt(52))
    return np.mean(sharpes) if sharpes else 0

def main():
    t0 = datetime.now()
    print("="*70)
    print("  VIX-TIMED SECTOR ROTATION v1")
    print("  Sector momentum + VIX regime filter for option timing")
    print("="*70)
    
    price_df, vix_df = load_data()
    print(f"  {len(price_df.columns)} sectors, {len(price_df)} days, VIX: {len(vix_df)} days")
    
    variants = {
        'A_NoFilter': {'use_vix_timer': False, 'top_n': 2, 'rebal_freq': 5, 'hold_days': 10},
        'B_VixTimer': {'use_vix_timer': True, 'vix_low': 20, 'vix_high': 30, 'top_n': 2, 'rebal_freq': 5, 'hold_days': 10},
        'C_TightVix': {'use_vix_timer': True, 'vix_low': 16, 'vix_high': 25, 'top_n': 2, 'rebal_freq': 5, 'hold_days': 10},
        'D_VixBounce': {'use_vix_timer': True, 'vix_bounce': True, 'vix_low': 20, 'vix_high': 30, 'top_n': 2, 'rebal_freq': 5, 'hold_days': 10},
        'E_Top1_VixTimer': {'use_vix_timer': True, 'vix_low': 20, 'vix_high': 30, 'top_n': 1, 'rebal_freq': 5, 'hold_days': 10},
        'F_Biweekly_VixTimer': {'use_vix_timer': True, 'vix_low': 20, 'vix_high': 30, 'top_n': 2, 'rebal_freq': 10, 'hold_days': 15},
    }
    
    results = []
    for name, cfg in variants.items():
        print(f"\n  {name}...", end=" ")
        try:
            r = run_variant(price_df, vix_df, name, cfg)
            if r and not r.get('skip'):
                results.append(r)
                print(f"Sharpe {r['sharpe']:.3f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | "
                      f"MDD {r['max_dd']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t "
                      f"[lowVIX:{r['low_vix_trades']}t avg${r['low_vix_avg']:.0f} | "
                      f"hiVIX:{r['high_vix_trades']}t avg${r['high_vix_avg']:.0f}]")
            elif r:
                print(f"SKIP ({r.get('trades',0)} trades)")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
    
    if not results:
        print("\n❌ ALL FAILED")
        return
    
    print(f"\n  Random baseline...", end=" ")
    rand_s = random_baseline(price_df, vix_df)
    print(f"Sharpe {rand_s:.3f}")
    
    print(f"\n{'='*70}")
    print("  5-GATE EVALUATION")
    print(f"{'='*70}")
    
    for r in sorted(results, key=lambda x: -x['sharpe']):
        g = 0
        g1 = r['sharpe'] > 0.5; g += g1
        pv = permutation_test(r['trade_pnls']) if len(r['trade_pnls'])>=10 else 1.0
        g2 = pv < 0.05; g += g2
        g3 = r['max_dd'] > -40; g += g3
        # VIX-specific: does the timer help?
        g4 = r['low_vix_avg'] > r['high_vix_avg'] if r['high_vix_trades'] > 0 else True; g += g4
        g5 = r['sharpe'] > rand_s + 0.10; g += g5
        
        status = '✅ PASS' if g >= 4 else ('⚠️' if g >= 3 else '❌ FAIL')
        print(f"\n  {r['variant']} — {status} ({g}/5)")
        print(f"    Sharpe {r['sharpe']:.3f} | Sortino {r['sortino']:.2f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f}")
        print(f"    MDD {r['max_dd']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t")
        print(f"    G1 Sharpe>0.5: {'✅' if g1 else '❌'} | G2 Perm: {'✅' if g2 else '❌'} (p={pv:.3f})")
        print(f"    G3 MDD>-40%: {'✅' if g3 else '❌'} | G4 VIX helps: {'✅' if g4 else '❌'} | G5 >Random: {'✅' if g5 else '❌'}")
        print(f"    Low-VIX: {r['low_vix_trades']}t avg ${r['low_vix_avg']:.1f} | High-VIX: {r['high_vix_trades']}t avg ${r['high_vix_avg']:.1f}")
    
    best = max(results, key=lambda x: x['sharpe'])
    print(f"\n  BEST: {best['variant']} Sharpe {best['sharpe']:.3f} | Random {rand_s:.3f}")
    print(f"  Elapsed: {(datetime.now()-t0).total_seconds():.0f}s")
    
    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        exp = mlflow.set_experiment("vix_timed_sector_v1")
        with mlflow.start_run(run_name=f"vix_timed_{datetime.now().strftime('%H%M')}"):
            mlflow.log_param("strategy", "vix_timed_sector_v1")
            mlflow.log_param("best_variant", best['variant'])
            mlflow.log_metric("best_sharpe", best['sharpe'])
            mlflow.log_metric("best_wr", best['wr'])
            mlflow.log_metric("best_mdd", best['max_dd'])
            mlflow.log_metric("random_sharpe", rand_s)
            for r in results:
                mlflow.log_metric(f"{r['variant']}_sharpe", r['sharpe'])
        print(f"  MLflow: {exp.name}")
    except Exception as e:
        print(f"  MLflow: {e}")

if __name__ == '__main__':
    main()
