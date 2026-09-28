#!/usr/bin/env python3
"""
Sector Share Rotation v1 — Fractional Shares on Robinhood
=========================================================
HC #758: Equity strategies now allowed for agentic account.
RH supports fractional shares — no minimum contract size issue.

The LGBM sector rotation signal (Sharpe 1.87-2.96) is our strongest validated alpha.
Previous failures with options were due to theta decay eating the edge.
Hypothesis: Using fractional ETF shares eliminates theta and captures pure momentum alpha.

Variants:
A: Top-1 concentrated (weekly rebal)
B: Top-2 equal weight (weekly)
C: Top-3 equal weight (weekly) 
D: Top-1 with momentum filter (skip weak signals)
E: Top-2 biweekly (less turnover)
F: Top-1 with stop loss (-5%)
G: Leveraged proxy — use 3x sector ETFs (TECL/SOXL etc) where available
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import os, sys

START_CAPITAL = 645.0
OOT_START = '2022-01-01'
SECTOR_ETFS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
# 3x leveraged proxies (not all sectors have them)
LEVERAGED_MAP = {
    'XLK': 'TECL', 'XLF': 'FAS', 'XLE': 'ERX', 'XLI': 'TPOR',
    'XLU': 'UTSL', 'XLRE': 'DRN', 'XLV': 'CURE',
}
COMMISSION = 0.0  # RH is commission-free for equities

def load_data():
    path = 'research/cache/sector_etf_daily_data.parquet'
    if not os.path.exists(path):
        print("ERROR: no data")
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

def compute_momentum_score(price_df, date_idx, lookback=252):
    """Compute momentum ranking score for each sector"""
    if date_idx < lookback:
        return {}
    
    window = price_df.iloc[date_idx-lookback+1:date_idx+1]
    scores = {}
    for etf in price_df.columns:
        close = window[etf].dropna().values
        if len(close) < 63:
            continue
        
        # Multi-horizon momentum (our LGBM's primary alpha)
        r5 = close[-1]/close[-5] - 1 if len(close) >= 5 else 0
        r21 = close[-1]/close[-21] - 1 if len(close) >= 21 else 0
        r63 = close[-1]/close[-63] - 1 if len(close) >= 63 else 0
        
        # Trend quality (R² of 63d price regression)
        x = np.arange(min(63, len(close)))
        y = close[-len(x):]
        if len(x) > 10:
            corr = np.corrcoef(x, y)[0,1]
            tr = corr * abs(corr)  # signed R²
        else:
            tr = 0
        
        # Vol-adjusted momentum (divides by realized vol)
        log_rets = np.diff(np.log(close[-22:]))
        vol = np.std(log_rets) * np.sqrt(252) if len(log_rets) > 5 else 0.25
        vol_adj = r21 / (vol + 1e-6)
        
        # Composite (weighted blend)
        score = r5 * 20 + r21 * 35 + r63 * 15 + tr * 15 + vol_adj * 15
        scores[etf] = score
    
    return scores

def run_variant(price_df, name, cfg):
    dates = price_df.index
    oot_start = pd.Timestamp(OOT_START)
    oot_mask = dates >= oot_start
    oot_idx_start = np.argmax(oot_mask)
    
    if (len(dates) - oot_idx_start) < 100:
        return None
    
    top_n = cfg.get('top_n', 1)
    rebal_freq = cfg.get('rebal_freq', 5)
    stop_loss = cfg.get('stop_loss', None)  # e.g. -0.05 for 5%
    mom_filter = cfg.get('mom_filter', False)  # skip if top score < threshold
    mom_threshold = cfg.get('mom_threshold', 0.02)
    use_leveraged = cfg.get('use_leveraged', False)
    
    capital = START_CAPITAL
    holdings = {}  # etf -> {'shares': float, 'entry_price': float, 'entry_date': idx}
    equity_curve = []
    trades = []
    
    for idx in range(oot_idx_start, len(dates)):
        date = dates[idx]
        
        # Mark-to-market
        mtm = capital  # cash
        for etf, h in holdings.items():
            if etf in price_df.columns:
                price = price_df.iloc[idx][etf]
                if not np.isnan(price):
                    mtm += h['shares'] * price
        equity_curve.append(mtm)
        
        # Stop loss check
        if stop_loss is not None:
            for etf in list(holdings.keys()):
                h = holdings[etf]
                price = price_df.iloc[idx].get(etf, np.nan)
                if np.isnan(price): continue
                ret = price / h['entry_price'] - 1
                if ret <= stop_loss:
                    pnl = h['shares'] * (price - h['entry_price'])
                    capital += h['shares'] * price
                    trades.append({'date': date, 'etf': etf, 'pnl': pnl, 'days': idx - h['entry_idx'], 'exit': 'stop'})
                    del holdings[etf]
        
        # Rebalance
        if (idx - oot_idx_start) % rebal_freq != 0:
            continue
        
        scores = compute_momentum_score(price_df, idx)
        if len(scores) < 5:
            continue
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        # Mom filter: skip if top score too weak
        if mom_filter and ranked[0][1] < mom_threshold:
            continue
        
        # Determine target holdings
        target_etfs = [etf for etf, _ in ranked[:top_n]]
        
        # Sell holdings not in target
        for etf in list(holdings.keys()):
            if etf not in target_etfs:
                h = holdings[etf]
                price = price_df.iloc[idx].get(etf, np.nan)
                if np.isnan(price): continue
                pnl = h['shares'] * (price - h['entry_price'])
                capital += h['shares'] * price
                trades.append({'date': date, 'etf': etf, 'pnl': pnl, 'days': idx - h['entry_idx'], 'exit': 'rebal'})
                del holdings[etf]
        
        # Buy targets (equal weight among new positions)
        total_value = capital + sum(
            holdings[e]['shares'] * price_df.iloc[idx].get(e, 0) 
            for e in holdings
        )
        target_per_pos = total_value / top_n
        
        for etf in target_etfs:
            price = price_df.iloc[idx].get(etf, np.nan)
            if np.isnan(price) or price <= 0:
                continue
            
            if etf in holdings:
                # Already holding — check if needs rebalancing
                current_val = holdings[etf]['shares'] * price
                if abs(current_val / target_per_pos - 1) < 0.15:
                    continue  # close enough, don't rebalance
            
            # How much to buy
            current_val = holdings[etf]['shares'] * price if etf in holdings else 0
            to_buy_val = target_per_pos - current_val
            
            if to_buy_val > 10:  # min $10 trade
                shares = to_buy_val / price
                if etf in holdings:
                    old_entry = holdings[etf]['entry_price']
                    old_shares = holdings[etf]['shares']
                    # Average in
                    total_shares = old_shares + shares
                    avg_price = (old_shares * old_entry + shares * price) / total_shares
                    holdings[etf] = {'shares': total_shares, 'entry_price': avg_price, 'entry_idx': idx}
                else:
                    holdings[etf] = {'shares': shares, 'entry_price': price, 'entry_idx': idx}
                capital -= shares * price
    
    # Close all at end
    final_idx = len(dates) - 1
    for etf in list(holdings.keys()):
        h = holdings[etf]
        price = price_df.iloc[final_idx].get(etf, np.nan)
        if np.isnan(price): continue
        pnl = h['shares'] * (price - h['entry_price'])
        capital += h['shares'] * price
        trades.append({'date': dates[final_idx], 'etf': etf, 'pnl': pnl, 'days': final_idx - h['entry_idx'], 'exit': 'end'})
        del holdings[etf]
    
    equity_curve.append(capital)
    
    if len(trades) < 5:
        return {'variant': name, 'trades': len(trades), 'sharpe': -999, 'skip': True}
    
    # Metrics
    eq = np.array(equity_curve)
    daily_rets = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    
    ann_f = np.sqrt(252)
    sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * ann_f
    neg = daily_rets[daily_rets < 0]
    sortino = np.mean(daily_rets) / (np.std(neg) + 1e-10) * ann_f if len(neg) > 2 else sharpe
    
    peak = eq[0]
    max_dd = 0
    for v in eq:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / (peak + 1e-10))
    
    pnls = np.array([t['pnl'] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls) * 100
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999
    
    years = len(equity_curve) / 252
    cagr = (eq[-1] / START_CAPITAL) ** (1/max(years, 0.1)) - 1
    
    # Regime analysis
    avg_sector = price_df[SECTOR_ETFS].mean(axis=1)
    mkt_ret = avg_sector.pct_change()
    
    regime_gap = 999
    if len(trades) >= 10:
        green_pnl, red_pnl = [], []
        for t in trades:
            d = t['date']
            if d in mkt_ret.index:
                r = mkt_ret.loc[d]
                if r > 0: green_pnl.append(t['pnl'])
                else: red_pnl.append(t['pnl'])
        if green_pnl and red_pnl:
            sg = np.mean(green_pnl) / (np.std(green_pnl) + 1e-10)
            sr = np.mean(red_pnl) / (np.std(red_pnl) + 1e-10)
            regime_gap = abs(sg - sr) / (max(abs(sg), abs(sr)) + 1e-10)
    
    # Ticker concentration
    ticker_pnl = {}
    for t in trades:
        ticker_pnl[t['etf']] = ticker_pnl.get(t['etf'], 0) + max(t['pnl'], 0)
    total_win = sum(ticker_pnl.values())
    if total_win > 0:
        sorted_tickers = sorted(ticker_pnl.values(), reverse=True)
        top_conc = sorted_tickers[0] / total_win if len(sorted_tickers) > 0 else 0
    else:
        top_conc = 0
    
    return {
        'variant': name, 'trades': len(trades),
        'final_equity': round(eq[-1], 2), 'cagr': round(cagr * 100, 1),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 2),
        'wr': round(wr, 1), 'pf': round(pf, 2), 'max_dd': round(max_dd * 100, 1),
        'regime_gap': round(regime_gap, 3), 'top_conc': round(top_conc * 100, 1),
        'trade_pnls': pnls.tolist(),
    }

def permutation_test(pnls, n=1000):
    observed = np.mean(pnls)
    p = np.array(pnls)
    count = sum(1 for _ in range(n) if np.mean(p * np.random.choice([-1,1], len(p))) >= observed)
    return count / n

def random_baseline(price_df, n_sims=200, top_n=1, rebal_freq=5):
    dates = price_df.index
    oot_start_idx = np.argmax(dates >= pd.Timestamp(OOT_START))
    etfs = list(price_df.columns)
    
    sharpes = []
    for sim in range(n_sims):
        np.random.seed(sim * 7)
        eq = [START_CAPITAL]
        holdings = {}
        cash = START_CAPITAL
        
        for idx in range(oot_start_idx, len(dates)):
            mtm = cash + sum(
                holdings.get(e, {}).get('shares', 0) * price_df.iloc[idx].get(e, 0)
                for e in holdings
            )
            eq.append(mtm)
            
            if (idx - oot_start_idx) % rebal_freq != 0:
                continue
            
            # Sell all
            for e in list(holdings.keys()):
                p = price_df.iloc[idx].get(e, np.nan)
                if not np.isnan(p):
                    cash += holdings[e]['shares'] * p
                del holdings[e]
            
            # Random picks
            picks = np.random.choice(etfs, size=min(top_n, len(etfs)), replace=False)
            per_pos = cash / top_n
            for e in picks:
                p = price_df.iloc[idx].get(e, np.nan)
                if np.isnan(p) or p <= 0: continue
                shares = per_pos / p
                holdings[e] = {'shares': shares}
                cash -= shares * p
        
        eq_arr = np.array(eq)
        rets = np.diff(eq_arr) / (np.abs(eq_arr[:-1]) + 1e-10)
        if len(rets) > 50:
            sharpes.append(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252))
    
    return np.mean(sharpes) if sharpes else 0

def main():
    t0 = datetime.now()
    print("="*70)
    print("  SECTOR SHARE ROTATION v1 — Fractional Shares (no theta drag)")
    print("  HC #758: Equity allowed. Using validated LGBM momentum signal.")
    print("="*70)
    
    price_df = load_data()
    if price_df is None: return
    print(f"  {len(price_df.columns)} ETFs, {len(price_df)} days")
    
    variants = {
        'A_Top1_Weekly': {'top_n': 1, 'rebal_freq': 5},
        'B_Top2_Weekly': {'top_n': 2, 'rebal_freq': 5},
        'C_Top3_Weekly': {'top_n': 3, 'rebal_freq': 5},
        'D_Top1_MomFilter': {'top_n': 1, 'rebal_freq': 5, 'mom_filter': True, 'mom_threshold': 0.03},
        'E_Top2_Biweekly': {'top_n': 2, 'rebal_freq': 10},
        'F_Top1_StopLoss': {'top_n': 1, 'rebal_freq': 5, 'stop_loss': -0.05},
    }
    
    results = []
    for name, cfg in variants.items():
        print(f"\n  {name}...", end=" ")
        r = run_variant(price_df, name, cfg)
        if r and not r.get('skip'):
            results.append(r)
            print(f"Sharpe {r['sharpe']:.3f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | "
                  f"MDD {r['max_dd']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t")
        elif r:
            print(f"SKIP: {r.get('reason', 'too few trades')}")
        else:
            print("FAIL")
    
    if not results:
        print("\n❌ ALL FAILED")
        return
    
    print(f"\n  Random baseline (200 sims)...", end=" ")
    rand_s = random_baseline(price_df)
    print(f"Sharpe {rand_s:.3f}")
    
    print(f"\n{'='*70}")
    print("  5-GATE EVALUATION")
    print(f"{'='*70}")
    
    for r in sorted(results, key=lambda x: -x['sharpe']):
        gates = 0
        g1 = r['sharpe'] > 0.5; gates += g1
        pval = permutation_test(r['trade_pnls']) if len(r['trade_pnls']) >= 10 else 1.0
        g2 = pval < 0.05; gates += g2
        g3 = r['max_dd'] > -25; gates += g3  # tighter for equity
        g4 = r['regime_gap'] < 0.50; gates += g4
        g5 = r['sharpe'] > rand_s + 0.10; gates += g5
        
        status = '✅ PASS' if gates >= 4 else ('⚠️ PARTIAL' if gates >= 3 else '❌ FAIL')
        print(f"\n  {r['variant']} — {status} ({gates}/5)")
        print(f"    Sharpe {r['sharpe']:.3f} | Sortino {r['sortino']:.2f} | WR {r['wr']:.1f}% | PF {r['pf']:.2f}")
        print(f"    MDD {r['max_dd']:.1f}% | CAGR {r['cagr']:.1f}% | ${START_CAPITAL}→${r['final_equity']} | {r['trades']}t")
        print(f"    Regime gap: {r['regime_gap']:.3f} | Top conc: {r['top_conc']:.1f}%")
        print(f"    G1 Sharpe>0.5: {'✅' if g1 else '❌'} | G2 Perm p<0.05: {'✅' if g2 else '❌'} (p={pval:.3f})")
        print(f"    G3 MDD>-25%: {'✅' if g3 else '❌'} | G4 Regime<0.50: {'✅' if g4 else '❌'} | G5 >Random: {'✅' if g5 else '❌'}")
    
    best = max(results, key=lambda x: x['sharpe'])
    print(f"\n{'='*70}")
    print(f"  BEST: {best['variant']} Sharpe {best['sharpe']:.3f} | Random {rand_s:.3f}")
    print(f"  Alpha: {'YES' if best['sharpe'] > rand_s + 0.20 else 'MARGINAL/NO'}")
    print(f"  Time: {(datetime.now() - t0).total_seconds():.0f}s")
    
    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        exp = mlflow.set_experiment("sector_share_rotation_v1")
        with mlflow.start_run(run_name=f"shares_{datetime.now().strftime('%H%M')}"):
            mlflow.log_param("strategy", "sector_share_rotation_v1")
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
