#!/usr/bin/env python3
"""
Drawdown Control Overlay for Sector Bull Spreads v1
=====================================================
Key finding from our research: ML adds RISK MANAGEMENT value, not alpha.
- ML vs Random: Sharpe 4.59 vs 3.50, but MaxDD -3.1% vs -16.7% (5x safer)
- Exit optimization: 20-day exit doubles Sharpe (0.55 vs 0.29)

This experiment tests SIMPLE drawdown control rules on top of our
validated sector bull spread strategy:

1. Position scaling: reduce position size when in drawdown
2. Skip filter: skip trades entirely during severe drawdowns
3. Recovery detection: increase size when recovering from drawdown
4. ML for risk only: use LightGBM rankings to AVOID bad sectors, not pick winners

Combined with 20-day exit rule from exit_optimization_v1.

$645 starting capital, B-S pricing, commissions, 4-gate audit.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import json
import os
from scipy.stats import norm

try:
    import mlflow; mlflow.set_tracking_uri("http://localhost:5000"); HAS_MLFLOW = True
except: HAS_MLFLOW = False

try:
    import lightgbm as lgb
except:
    print("LightGBM required"); exit(1)

UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
START_CAP = 645.0
COMMISSION_RT = 4.70
HAIRCUT = 0.15

CONFIGS = {
    'A_Baseline_NoControl': {
        'dd_scale': False, 'dd_skip_pct': None, 'recovery_boost': False,
        'ml_risk_filter': False, 'exit_days': 30, 'vix_min': 20,
        'desc': 'Baseline: no drawdown control, 30d hold'
    },
    'B_20dExit_NoControl': {
        'dd_scale': False, 'dd_skip_pct': None, 'recovery_boost': False,
        'ml_risk_filter': False, 'exit_days': 20, 'vix_min': 20,
        'desc': '20-day exit only (from exit_optimization finding)'
    },
    'C_DDScale_50pct': {
        'dd_scale': True, 'dd_threshold': -0.05, 'dd_min_alloc': 0.50,
        'dd_skip_pct': None, 'recovery_boost': False,
        'ml_risk_filter': False, 'exit_days': 20, 'vix_min': 20,
        'desc': 'Scale to 50% position when DD > 5%'
    },
    'D_DDScale_25pct': {
        'dd_scale': True, 'dd_threshold': -0.03, 'dd_min_alloc': 0.25,
        'dd_skip_pct': None, 'recovery_boost': False,
        'ml_risk_filter': False, 'exit_days': 20, 'vix_min': 20,
        'desc': 'Scale to 25% position when DD > 3% (aggressive risk control)'
    },
    'E_DDSkip_10pct': {
        'dd_scale': False, 'dd_skip_pct': -0.10,
        'recovery_boost': False, 'ml_risk_filter': False,
        'exit_days': 20, 'vix_min': 20,
        'desc': 'Skip ALL trades when DD > 10%'
    },
    'F_MLRisk_20d': {
        'dd_scale': False, 'dd_skip_pct': None,
        'recovery_boost': False, 'ml_risk_filter': True,
        'exit_days': 20, 'vix_min': 20,
        'desc': 'ML filter: skip bottom-3 sectors (use ML for risk, not alpha)'
    },
    'G_Full_Stack': {
        'dd_scale': True, 'dd_threshold': -0.05, 'dd_min_alloc': 0.50,
        'dd_skip_pct': -0.15, 'recovery_boost': True,
        'ml_risk_filter': True, 'exit_days': 20, 'vix_min': 20,
        'desc': 'FULL: DD scaling + skip + recovery boost + ML risk filter + 20d exit'
    },
    'H_Conservative': {
        'dd_scale': True, 'dd_threshold': -0.03, 'dd_min_alloc': 0.30,
        'dd_skip_pct': -0.08, 'recovery_boost': False,
        'ml_risk_filter': True, 'exit_days': 20, 'vix_min': 20,
        'desc': 'CONSERVATIVE: tight DD control + ML risk + 20d exit'
    },
}

def bs_call(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def spread_value(S, K_long, K_short, T, sigma):
    if T <= 0.001:
        return max(S - K_long, 0) - max(S - K_short, 0)
    return bs_call(S, K_long, T, sigma) - bs_call(S, K_short, T, sigma)

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def load_data():
    import yfinance as yf
    cache_dir = '/home/jupiter/Lvl3Quant/research/cache'
    os.makedirs(cache_dir, exist_ok=True)
    daily_cache = os.path.join(cache_dir, 'sector_etf_daily_data.parquet')
    vix_cache = os.path.join(cache_dir, 'vix_daily_data.parquet')

    if os.path.exists(daily_cache) and (datetime.now().timestamp() - os.path.getmtime(daily_cache)) / 3600 < 24:
        close = pd.read_parquet(daily_cache)
        vix = pd.read_parquet(vix_cache) if os.path.exists(vix_cache) else None
        return close, vix

    tickers = UNIVERSE + ['^VIX']
    data = yf.download(tickers, start='2010-01-01', progress=False, auto_adjust=True)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    vc = '^VIX' if '^VIX' in close.columns else None
    vix = close[[vc]].rename(columns={vc: 'VIX'}) if vc else None
    close = close.drop(columns=[vc], errors='ignore') if vc else close
    close.to_parquet(daily_cache)
    if vix is not None: vix.to_parquet(vix_cache)
    return close, vix

def build_monthly_features(close, vix):
    monthly = close.resample('ME').last().dropna(how='all')
    mvix = vix.resample('ME').last().dropna() if vix is not None else None
    fl = []
    for t in UNIVERSE:
        if t not in monthly.columns: continue
        px = monthly[t].dropna()
        if len(px) < 15: continue
        f = pd.DataFrame(index=px.index)
        f['ticker'] = t; f['price'] = px
        for m in [1,2,3,6,12]: f[f'ret_{m}m'] = px.pct_change(m)
        r1 = px.pct_change()
        for m in [3,6,12]: f[f'vol_{m}m'] = r1.rolling(m).std()
        f['sharpe_6m'] = f['ret_6m']/(f['vol_6m']+1e-8)
        f['sharpe_12m'] = f['ret_12m']/(f['vol_12m']+1e-8)
        f['rsi_14'] = compute_rsi(px, 14)
        for m in [5,10,20]:
            sma = px.rolling(m).mean()
            f[f'above_sma{m}'] = (px > sma).astype(float)
        ew = monthly[UNIVERSE].pct_change(3).mean(axis=1)
        f['rel_strength_3m'] = f['ret_3m'] - ew
        rm = px.rolling(12).max()
        dd = (px - rm) / rm
        f['max_dd_12m'] = dd.rolling(12).min()
        f['calmar_12m'] = f['ret_12m']/(-f['max_dd_12m']+1e-8)
        if mvix is not None and 'VIX' in mvix.columns:
            v = mvix['VIX'].reindex(f.index, method='ffill')
            f['vix'] = v
        f['target'] = px.pct_change().shift(-1)
        fl.append(f)
    return pd.concat(fl)

def run_backtest(daily_close, vix, features, config):
    feat_cols = [c for c in features.columns if c not in ['ticker','price','target','vix']]
    dates = sorted(features.index.unique())

    exit_days = config['exit_days']
    vix_min = config['vix_min']
    dd_scale = config.get('dd_scale', False)
    dd_threshold = config.get('dd_threshold', -0.05)
    dd_min_alloc = config.get('dd_min_alloc', 0.50)
    dd_skip_pct = config.get('dd_skip_pct')
    recovery_boost = config.get('recovery_boost', False)
    ml_risk_filter = config.get('ml_risk_filter', False)

    equity = START_CAP
    peak_equity = START_CAP
    trades = []
    equity_curve = []
    train_periods = 12
    top_k = 3

    for i in range(train_periods, len(dates) - 1):
        date = dates[i]

        # Track drawdown
        current_dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0

        # DD skip: don't trade at all during severe drawdowns
        if dd_skip_pct is not None and current_dd < dd_skip_pct:
            equity_curve.append({'date': date, 'equity': equity, 'dd': current_dd, 'action': 'skip_dd'})
            continue

        # VIX filter
        if vix is not None:
            vv = vix.loc[:date, 'VIX']
            current_vix = vv.iloc[-1] if len(vv) > 0 else 15
            if current_vix < vix_min:
                equity_curve.append({'date': date, 'equity': equity, 'dd': current_dd, 'action': 'skip_vix'})
                continue
        else:
            current_vix = 15

        # Train LightGBM
        train_start = dates[max(0, i - train_periods)]
        train_mask = (features.index >= train_start) & (features.index < date)
        train_data = features[train_mask].copy()
        pred_data = features[features.index == date].copy()

        if len(train_data) < 15 or len(pred_data) == 0:
            equity_curve.append({'date': date, 'equity': equity, 'dd': current_dd})
            continue

        tX = train_data[feat_cols].replace([np.inf,-np.inf], np.nan)
        ty = train_data['target'].values
        valid = ~(tX.isna().any(axis=1) | np.isnan(ty))
        tX = tX[valid]; ty = ty[valid]

        if len(tX) < 10:
            equity_curve.append({'date': date, 'equity': equity, 'dd': current_dd})
            continue

        pX = pred_data[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)

        try:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1, random_state=42
            )
            model.fit(tX, ty)
            preds = model.predict(pX)
        except:
            equity_curve.append({'date': date, 'equity': equity, 'dd': current_dd})
            continue

        pred_df = pred_data[['ticker','price']].copy()
        pred_df['pred_ret'] = preds
        pred_df = pred_df.sort_values('pred_ret', ascending=False)

        # ML risk filter: exclude bottom-3 sectors
        if ml_risk_filter:
            bottom_tickers = pred_df.tail(3)['ticker'].values
            pred_df = pred_df[~pred_df['ticker'].isin(bottom_tickers)]

        # Take top K from remaining
        top_sectors = pred_df.head(top_k)
        next_date = dates[i + 1]

        # Position sizing with drawdown control
        base_alloc = 0.30  # 30% of equity per period, split across top_k

        if dd_scale and current_dd < dd_threshold:
            # Linear interpolation: at threshold → min_alloc, at 0 → full alloc
            scale = max(dd_min_alloc, 1.0 + (1.0 - dd_min_alloc) * current_dd / dd_threshold)
            alloc = base_alloc * scale
        elif recovery_boost and current_dd < 0 and current_dd > dd_threshold:
            # Recovering from drawdown: slightly larger positions
            alloc = base_alloc * 1.1
        else:
            alloc = base_alloc

        entry_date_candidates = daily_close.index[daily_close.index > date]
        if len(entry_date_candidates) == 0: continue
        entry_date = entry_date_candidates[0]

        for _, row in top_sectors.iterrows():
            ticker = row['ticker']
            if ticker not in daily_close.columns: continue

            S_entry = daily_close.loc[entry_date, ticker] if entry_date in daily_close.index else None
            if S_entry is None or pd.isna(S_entry): continue

            # IV estimate
            hist = daily_close[ticker].loc[:entry_date].tail(60)
            iv = hist.pct_change().std() * np.sqrt(252) * 1.1 if len(hist) > 10 else 0.25

            K_long = S_entry * 0.985
            K_short = S_entry * 1.015
            T_entry = 30 / 365.0

            entry_val = spread_value(S_entry, K_long, K_short, T_entry, iv)
            entry_cost = entry_val * (1 + HAIRCUT) * 100
            if entry_cost <= 0: continue

            max_spread = (K_short - K_long) * 100
            max_trade = min(equity * alloc / top_k, 200)
            n_contracts = max(1, int(max_trade / (entry_cost / 100 + 0.01)))

            # Daily exit monitoring with exit_days limit
            exit_date = None; exit_price = None

            for d in daily_close.index:
                if d <= entry_date: continue
                days_held = (d - entry_date).days
                if days_held > exit_days:
                    exit_date = d; break

                if ticker not in daily_close.columns or d not in daily_close.index: continue
                S_now = daily_close.loc[d, ticker]
                if pd.isna(S_now): continue

                T_rem = max((30 - days_held) / 365.0, 0.001)
                curr_val = spread_value(S_now, K_long, K_short, T_rem, iv) * (1 - HAIRCUT) * 100
                unrealized = curr_val - entry_cost

                # 50% take profit
                if unrealized >= 0.50 * (max_spread - entry_cost):
                    exit_date = d; exit_price = S_now; break

            if exit_date is None:
                possible = daily_close.index[daily_close.index >= entry_date + timedelta(days=exit_days)]
                exit_date = possible[0] if len(possible) > 0 else None
            if exit_date is None: continue

            if exit_price is None:
                exit_price = daily_close.loc[exit_date, ticker] if exit_date in daily_close.index and ticker in daily_close.columns else None
            if exit_price is None or pd.isna(exit_price): continue

            days_held = (exit_date - entry_date).days
            T_exit = max((30 - days_held) / 365.0, 0)

            if T_exit <= 0.001:
                exit_val = (max(exit_price - K_long, 0) - max(exit_price - K_short, 0)) * 100
            else:
                exit_val = spread_value(exit_price, K_long, K_short, T_exit, iv) * (1 - HAIRCUT) * 100

            pnl = (exit_val - entry_cost - COMMISSION_RT) * n_contracts
            actual_cost = entry_cost * n_contracts / 100
            if pnl < -actual_cost: pnl = -actual_cost

            equity += pnl
            if equity > peak_equity: peak_equity = equity

            # Regime
            mkt = daily_close[UNIVERSE].mean(axis=1)
            mkt_ret = (mkt.loc[exit_date] - mkt.loc[entry_date]) / mkt.loc[entry_date] if entry_date in mkt.index and exit_date in mkt.index else 0

            trades.append({
                'date': entry_date, 'exit_date': exit_date, 'ticker': ticker,
                'pnl': pnl, 'cost': actual_cost, 'n_contracts': n_contracts,
                'days_held': days_held, 'dd_at_entry': current_dd,
                'alloc_pct': alloc, 'vix': current_vix,
                'regime': 'bull' if mkt_ret > 0 else 'bear',
                'equity_after': equity,
            })

        equity_curve.append({'date': date, 'equity': equity, 'dd': current_dd})

    return trades, equity_curve

def adversarial_audit(trades, equity_curve, name):
    if len(trades) < 10:
        return {'name': name, 'n_trades': len(trades), 'gates_passed': 0, 'error': 'Too few trades'}

    td = pd.DataFrame(trades)
    pnls = td['pnl'].values
    n = len(pnls); wins = (pnls > 0).sum()
    wr = wins/n*100
    pf = abs(pnls[pnls>0].sum() / pnls[pnls<0].sum()) if (pnls<0).sum() != 0 else 999

    eq = pd.DataFrame(equity_curve)
    eq['date'] = pd.to_datetime(eq['date'])
    eq = eq.set_index('date')
    meq = eq['equity'].resample('ME').last().dropna()
    mret = meq.pct_change().dropna()

    sharpe = mret.mean()/(mret.std()+1e-8)*np.sqrt(12) if len(mret) > 1 else 0
    neg = mret[mret<0]
    sortino = mret.mean()/(neg.std()+1e-8)*np.sqrt(12) if len(neg) > 0 else sharpe*1.5

    feq = eq['equity'].iloc[-1] if len(eq) > 0 else START_CAP
    yrs = max((eq.index[-1]-eq.index[0]).days/365.25, 0.1)
    cagr = (feq/START_CAP)**(1/yrs)-1
    maxdd = ((eq['equity']-eq['equity'].cummax())/eq['equity'].cummax()).min()
    calmar = cagr/(-maxdd+1e-8) if maxdd < 0 else 0

    # G1: Permutation
    ps = []
    for _ in range(500):
        sh = pnls.copy(); np.random.shuffle(sh)
        ep = np.cumsum(sh)+START_CAP
        nm = max(1,len(ep)//3); ch = np.array_split(ep,nm)
        mr = []; pv = START_CAP
        for c in ch:
            if len(c)>0: mr.append((c[-1]-pv)/pv); pv=c[-1]
        if len(mr)>1: a=np.array(mr); ps.append(a.mean()/(a.std()+1e-8)*np.sqrt(12))
    pp = np.mean(np.array(ps)>=sharpe) if ps else 1.0
    g1 = pp < 0.05

    # G2: R1
    bull=td[td['regime']=='bull']['pnl'].values
    bear=td[td['regime']=='bear']['pnl'].values
    if len(bull)>5 and len(bear)>5:
        bs=bull.mean()/(bull.std()+1e-8)*np.sqrt(12)
        brs=bear.mean()/(bear.std()+1e-8)*np.sqrt(12)
        r1=abs(bs-brs)/max(abs(bs),abs(brs),1e-8)
        bwr=(bull>0).mean()*100; brwr=(bear>0).mean()*100
    else:
        bs=brs=sharpe;r1=0;bwr=brwr=wr
    g2 = r1 < 0.50

    # G3
    t=len(td)//3; ss=[]
    for s,e in [(0,t),(t,2*t),(2*t,len(td))]:
        sp=td.iloc[s:e]['pnl'].values
        if len(sp)>3: ss.append(sp.mean()/(sp.std()+1e-8)*np.sqrt(12))
    g3 = len(ss)>=2 and all(x>0 for x in ss)

    # G4
    p5,p95=np.percentile(pnls,[5,95])
    tr=pnls[(pnls>=p5)&(pnls<=p95)]
    g4 = (tr.mean()/(tr.std()+1e-8)*np.sqrt(12)>0) if len(tr)>3 else False

    # DD stats
    dd_trades = td[td['dd_at_entry'] < -0.03] if 'dd_at_entry' in td.columns else pd.DataFrame()
    dd_wr = (dd_trades['pnl']>0).mean()*100 if len(dd_trades) > 0 else 0

    avg_hold = td['days_held'].mean() if 'days_held' in td.columns else 30

    return {
        'name': name, 'n_trades': n, 'win_rate': round(wr,1),
        'avg_win': round(pnls[pnls>0].mean(),2) if wins>0 else 0,
        'avg_loss': round(pnls[pnls<0].mean(),2) if (pnls<0).sum()>0 else 0,
        'total_pnl': round(pnls.sum(),2), 'final_equity': round(feq,2),
        'cagr_pct': round(cagr*100,1), 'sharpe': round(sharpe,2),
        'sortino': round(sortino,2), 'maxdd_pct': round(maxdd*100,1),
        'calmar': round(calmar,2), 'pf': round(pf,2),
        'r1_gap': round(r1,3), 'bull_wr': round(bwr,1), 'bear_wr': round(brwr,1),
        'bull_trades': len(bull), 'bear_trades': len(bear),
        'perm_p': round(pp,4), 'g1_pass': g1, 'g2_pass': g2,
        'g3_pass': g3, 'g4_pass': g4,
        'sub_sharpes': [round(x,2) for x in ss],
        'gates_passed': sum([g1,g2,g3,g4]),
        'avg_hold_days': round(avg_hold,1),
        'trades_during_dd': len(dd_trades),
        'dd_wr': round(dd_wr,1),
    }

def main():
    print("="*70)
    print("Drawdown Control Overlay v1")
    print("="*70)
    t0 = datetime.now()

    close, vix = load_data()
    features = build_monthly_features(close, vix)
    print(f"Data: {close.shape[0]} days, {len(features)} feature rows")

    if HAS_MLFLOW:
        exp = mlflow.set_experiment("drawdown_control_v1")
        exp_id = exp.experiment_id

    results = []
    best = None; best_sharpe = -999

    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name} — {config['desc']}")

        trades, eq = run_backtest(close, vix, features, config)

        if len(trades) < 10:
            print(f"  SKIP: {len(trades)} trades")
            results.append({'name': name, 'n_trades': len(trades), 'gates_passed': 0})
            continue

        r = adversarial_audit(trades, eq, name)
        results.append(r)

        gates = f"{r['gates_passed']}/4"
        status = "✅" if r['gates_passed']==4 else "⚠️" if r['gates_passed']>=2 else "❌"

        print(f"  {status} {gates} | Sharpe {r['sharpe']:.2f} | WR {r['win_rate']:.1f}% | "
              f"CAGR {r['cagr_pct']:.1f}% | MDD {r['maxdd_pct']:.1f}% | "
              f"PF {r['pf']:.2f} | {r['n_trades']} trades | ${START_CAP}→${r['final_equity']:,.0f}")
        print(f"  R1 gap {r['r1_gap']:.3f} | Perm p={r['perm_p']:.4f} | "
              f"Hold {r['avg_hold_days']:.1f}d | DD trades: {r.get('trades_during_dd',0)} (WR {r.get('dd_wr',0):.0f}%)")

        if HAS_MLFLOW:
            with mlflow.start_run(experiment_id=exp_id, run_name=name):
                for k,v in r.items():
                    if isinstance(v,(int,float)): mlflow.log_metric(k,v)
                mlflow.log_params({k: str(v) for k,v in config.items() if k != 'desc'})

        if r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']; best = r

    runtime = (datetime.now() - t0).total_seconds()

    print(f"\n{'='*70}")
    print("SUMMARY — Drawdown Control v1")
    print(f"{'='*70}")

    p4 = sum(1 for r in results if r.get('gates_passed',0)==4)
    print(f"Configs: {len(results)} | 4/4: {p4} | Runtime: {runtime:.0f}s")

    # Improvement over baseline
    baseline = next((r for r in results if 'Baseline' in r.get('name','')), None)
    if baseline and best:
        print(f"\nBaseline: Sharpe {baseline['sharpe']:.2f}, MDD {baseline['maxdd_pct']:.1f}%")
        print(f"Best:     Sharpe {best['sharpe']:.2f}, MDD {best['maxdd_pct']:.1f}% ({best['name']})")
        print(f"Improvement: {best['sharpe'] - baseline['sharpe']:+.2f} Sharpe, "
              f"{best['maxdd_pct'] - baseline['maxdd_pct']:+.1f}% MDD")

    print(f"\n--- COMPARISON ---")
    for r in sorted(results, key=lambda x: x.get('sharpe',-999), reverse=True):
        if 'error' not in r:
            bl = baseline['sharpe'] if baseline else 0
            print(f"  {r['name']:25s} | Sharpe {r['sharpe']:5.2f} | MDD {r['maxdd_pct']:6.1f}% | "
                  f"WR {r['win_rate']:5.1f}% | Δ {r['sharpe']-bl:+.2f}")

    # Save
    fd = '/home/jupiter/Lvl3Quant/research/findings'
    os.makedirs(fd, exist_ok=True)
    with open(os.path.join(fd, 'drawdown_control_v1_results.json'), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(), 'experiment': 'drawdown_control_v1',
            'results': results, 'best': best, 'baseline': baseline, 'runtime_s': runtime,
        }, f, indent=2, default=str)

    print(f"\nResults saved.")
    return results

if __name__ == '__main__':
    main()
