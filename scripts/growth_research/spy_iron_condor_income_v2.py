#!/usr/bin/env python3
"""
SPY Iron Condor Income v2 — Larger Capital + More Variants
===========================================================

v1 finding: Only 20-delta variant fit $645 account (4/4 gates, Sharpe 1.38, WR 72%).
Problem: $645 too small for most iron condor widths — margin requirement exceeds capital.

v2: Test with $10K baseline (realistic portfolio allocation for IC sleeve).
Also: fix the margin filter, test more delta/DTE combos, add VIX timing.

This addresses the INCOME component of the portfolio.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'spy_iron_condor_income_v2_results.json'

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass


def bs_price(S, K, T, sigma, r=0.04, opt='call'):
    if T <= 0 or sigma <= 0:
        return max(0, S-K) if opt=='call' else max(0, K-S)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt=='call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def strike_at_delta(S, T, sigma, target_delta, r=0.04, opt='call'):
    if T <= 0: return S
    lo, hi = (S*0.8, S*1.5) if opt=='call' else (S*0.5, S*1.2)
    for _ in range(50):
        mid = (lo+hi)/2
        d1 = (np.log(S/mid) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        if opt=='call':
            if norm.cdf(d1) > target_delta: lo=mid
            else: hi=mid
        else:
            if abs(-norm.cdf(-d1)) > target_delta: hi=mid
            else: lo=mid
    return round((lo+hi)/2)


def simulate_ic(spy_close, vix_close, delta=0.16, width=5, iv_rank_min=30,
                dte=30, profit_target=0.50, name='base', capital=10000,
                max_concurrent=3, max_risk_pct=0.15):
    """Simulate iron condor with proper position sizing."""
    fprint(f"\n--- {name} ---")

    vix_rank = vix_close.rolling(252).apply(
        lambda x: (x.iloc[-1]-x.min()) / (x.max()-x.min()+1e-10) * 100
    )
    spy_sma200 = spy_close.rolling(200).mean()
    dates = spy_close.index[252:]

    trades = []
    open_trades = []  # Track concurrent positions
    equity = capital

    for i_idx in range(len(dates) - dte - 1):
        date = dates[i_idx]

        # Close expired/exited trades
        new_open = []
        for ot in open_trades:
            if date >= ot['check_dates'][-1]:
                # Force close
                S_now = float(spy_close.loc[date])
                vix_now = float(vix_close.loc[date]) if date in vix_close.index else ot['vix']
                days_left = max(1, (ot['expiry_date'] - date).days)
                T_now = max(days_left/252, 0.001)
                sigma_now = vix_now / 100

                cost = (bs_price(S_now, ot['call_short'], T_now, sigma_now, opt='call') -
                        bs_price(S_now, ot['call_long'], T_now, sigma_now, opt='call') +
                        bs_price(S_now, ot['put_short'], T_now, sigma_now, opt='put') -
                        bs_price(S_now, ot['put_long'], T_now, sigma_now, opt='put'))

                pnl = (ot['credit'] - cost) * 100 - 5  # per contract
                equity += pnl
                regime = 'bear' if S_now < float(spy_sma200.loc[date]) else 'bull'
                trades.append({
                    'entry': str(ot['entry_date']), 'exit': str(date),
                    'pnl': round(pnl, 2), 'credit': round(ot['credit'], 2),
                    'win': pnl > 0, 'regime': regime
                })
            else:
                # Check for early exit
                S_now = float(spy_close.loc[date])
                vix_now = float(vix_close.loc[date]) if date in vix_close.index else ot['vix']
                days_left = max(1, (ot['expiry_date'] - date).days)
                T_now = max(days_left/252, 0.001)
                sigma_now = vix_now / 100

                cost = (bs_price(S_now, ot['call_short'], T_now, sigma_now, opt='call') -
                        bs_price(S_now, ot['call_long'], T_now, sigma_now, opt='call') +
                        bs_price(S_now, ot['put_short'], T_now, sigma_now, opt='put') -
                        bs_price(S_now, ot['put_long'], T_now, sigma_now, opt='put'))

                current_pnl = ot['credit'] - cost

                # 50% profit target
                if current_pnl >= ot['credit'] * profit_target:
                    pnl = current_pnl * 100 - 5
                    equity += pnl
                    regime = 'bear' if S_now < float(spy_sma200.loc[date]) else 'bull'
                    trades.append({
                        'entry': str(ot['entry_date']), 'exit': str(date),
                        'pnl': round(pnl, 2), 'credit': round(ot['credit'], 2),
                        'win': True, 'regime': regime
                    })
                    continue

                # Stop: short strike breached with > 2x loss
                if current_pnl < -ot['credit'] * 2:
                    pnl = current_pnl * 100 - 5
                    equity += pnl
                    regime = 'bear' if S_now < float(spy_sma200.loc[date]) else 'bull'
                    trades.append({
                        'entry': str(ot['entry_date']), 'exit': str(date),
                        'pnl': round(pnl, 2), 'credit': round(ot['credit'], 2),
                        'win': False, 'regime': regime
                    })
                    continue

                new_open.append(ot)

        open_trades = new_open

        # Skip if at max concurrent
        if len(open_trades) >= max_concurrent:
            continue

        # Check IV rank
        if date not in vix_rank.index:
            continue
        ivr = vix_rank.loc[date]
        if pd.isna(ivr) or ivr < iv_rank_min:
            continue

        # Only open on Fridays or Mondays (avoid mid-week noise)
        if date.weekday() not in [0, 4]:
            continue

        S = float(spy_close.loc[date])
        vix = float(vix_close.loc[date])
        sigma = vix / 100
        T = dte / 252

        call_short = strike_at_delta(S, T, sigma, delta, opt='call')
        put_short = strike_at_delta(S, T, sigma, delta, opt='put')
        call_long = call_short + width
        put_long = put_short - width

        sc = bs_price(S, call_short, T, sigma, opt='call')
        lc = bs_price(S, call_long, T, sigma, opt='call')
        sp = bs_price(S, put_short, T, sigma, opt='put')
        lp = bs_price(S, put_long, T, sigma, opt='put')

        credit = (sc - lc) + (sp - lp)
        max_loss = width - credit

        if credit <= 0.30 or max_loss <= 0:
            continue

        # Position sizing: risk max_risk_pct of equity per trade
        margin = max_loss * 100
        if margin > equity * max_risk_pct:
            continue

        # Number of contracts
        n_contracts = max(1, int(equity * max_risk_pct / margin))
        n_contracts = min(n_contracts, 2)  # Cap at 2 for small accounts

        expiry = date + pd.Timedelta(days=dte)
        check_dates = pd.bdate_range(date + pd.Timedelta(days=1), expiry)

        open_trades.append({
            'entry_date': date,
            'expiry_date': expiry,
            'call_short': call_short, 'call_long': call_long,
            'put_short': put_short, 'put_long': put_long,
            'credit': credit,
            'max_loss': max_loss,
            'n_contracts': n_contracts,
            'vix': vix,
            'check_dates': check_dates,
        })

    # Close any remaining open trades at last date
    for ot in open_trades:
        S_now = float(spy_close.iloc[-1])
        pnl = ot['credit'] * 100 - 5  # Assume expired worthless (optimistic)
        equity += pnl
        trades.append({
            'entry': str(ot['entry_date']), 'exit': str(spy_close.index[-1]),
            'pnl': round(pnl, 2), 'credit': round(ot['credit'], 2),
            'win': pnl > 0, 'regime': 'bull'
        })

    if not trades:
        fprint("  No trades!")
        return None

    # Metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n_trades * 100

    total_pnl = sum(t['pnl'] for t in trades)
    avg_win = np.mean([t['pnl'] for t in trades if t['win']]) if wins > 0 else 0
    avg_loss = np.mean([t['pnl'] for t in trades if not t['win']]) if wins < n_trades else 0

    # Monthly P&L
    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum() / capital
    n_years = len(monthly) / 12

    sharpe = (monthly.mean() * 12) / (monthly.std() * np.sqrt(12) + 1e-10) if len(monthly) > 3 else 0
    cagr = (1 + total_pnl/capital) ** (1/max(n_years, 0.01)) - 1

    down = monthly[monthly < 0]
    sortino = (monthly.mean()*12) / (down.std()*np.sqrt(12)+1e-10) if len(down) > 0 else 999

    cum = np.cumsum([t['pnl'] for t in trades])
    peak = np.maximum.accumulate(cum + capital)
    dd = (cum + capital - peak) / peak
    maxdd = dd.min()

    pf = abs(sum(t['pnl'] for t in trades if t['win']) / (sum(t['pnl'] for t in trades if not t['win'])+1e-10))

    bull_t = [t for t in trades if t['regime']=='bull']
    bear_t = [t for t in trades if t['regime']=='bear']
    bull_wr = sum(1 for t in bull_t if t['win'])/max(len(bull_t),1)*100
    bear_wr = sum(1 for t in bear_t if t['win'])/max(len(bear_t),1)*100
    r1_gap = abs(bull_wr-bear_wr)/max(bull_wr,bear_wr,1)

    result = {
        'name': name, 'n_trades': n_trades, 'win_rate': round(wr,1),
        'avg_win': round(avg_win,2), 'avg_loss': round(avg_loss,2),
        'total_pnl': round(total_pnl,2), 'final_equity': round(equity,2),
        'cagr_pct': round(cagr*100,1), 'sharpe': round(sharpe,2),
        'sortino': round(sortino,2), 'maxdd_pct': round(maxdd*100,1),
        'pf': round(pf,2), 'r1_gap': round(r1_gap,3), 'r1_pass': r1_gap <= 0.50,
        'bull_wr': round(bull_wr,1), 'bear_wr': round(bear_wr,1),
        'bull_trades': len(bull_t), 'bear_trades': len(bear_t),
        'monthly_returns': monthly.tolist(),
    }

    fprint(f"  {name}: {n_trades} trades, WR {wr:.1f}%, Sharpe {sharpe:.2f}, "
           f"CAGR {cagr*100:.1f}%, MaxDD {maxdd*100:.1f}%, PF {pf:.2f}")
    fprint(f"    ${capital:,} → ${equity:,.0f}, Bull WR {bull_wr:.0f}% ({len(bull_t)}), Bear WR {bear_wr:.0f}% ({len(bear_t)})")

    return result


def permutation_test(returns, n_perms=1000):
    real = np.mean(returns)/(np.std(returns)+1e-10)
    count = sum(1 for _ in range(n_perms)
                if np.mean(returns * np.random.choice([-1,1], len(returns))) /
                (np.std(returns)+1e-10) >= real)
    return count / n_perms


def main():
    import yfinance as yf
    fprint(f"SPY Iron Condor Income v2 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    raw = yf.download(['SPY','^VIX'], start='2008-01-01', end='2026-07-25', progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    spy = close['SPY'].dropna()
    vix = close['^VIX'].dropna() if '^VIX' in close.columns else close['VIX'].dropna()
    common = spy.index.intersection(vix.index)
    spy, vix = spy.loc[common], vix.loc[common]

    fprint(f"Data: {len(spy)} days")

    variants = [
        # (name, delta, width, ivr_min, dte, capital)
        ('A_16d_5w_ivr30', 0.16, 5, 30, 30, 10000),
        ('B_16d_10w_ivr30', 0.16, 10, 30, 30, 10000),
        ('C_10d_5w_ivr30', 0.10, 5, 30, 30, 10000),
        ('D_16d_5w_ivr50', 0.16, 5, 50, 30, 10000),
        ('E_20d_5w_ivr30', 0.20, 5, 30, 30, 10000),
        ('F_16d_5w_45dte', 0.16, 5, 30, 45, 10000),
        ('G_16d_5w_nofilter', 0.16, 5, 0, 30, 10000),
        ('H_20d_5w_645', 0.20, 5, 30, 30, 645),  # Agentic account size
    ]

    results = []

    if MLFLOW_OK:
        exp_name = 'spy_iron_condor_income_v2'
        try:
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(exp_name)
        except: pass
        mlflow.set_experiment(exp_name)

    for vname, d, w, ivr, dte_val, cap in variants:
        try:
            r = simulate_ic(spy, vix, delta=d, width=w, iv_rank_min=ivr,
                           dte=dte_val, name=vname, capital=cap)
            if r:
                if MLFLOW_OK:
                    with mlflow.start_run(run_name=vname):
                        mlflow.log_params({'delta':d,'width':w,'ivr_min':ivr,'dte':dte_val,'capital':cap})
                        mlflow.log_metrics({k:v for k,v in r.items() if isinstance(v,(int,float))})
                results.append(r)
        except Exception as e:
            fprint(f"  ERROR {vname}: {e}")
            import traceback; traceback.print_exc()

    if not results:
        fprint("No results!"); return

    # Adversarial
    fprint("\n" + "="*70 + "\nADVERSARIAL VALIDATION\n" + "="*70)
    for r in results:
        rets = np.array(r['monthly_returns'])
        r['perm_p'] = round(permutation_test(rets), 3)
        r['g1_pass'] = r['perm_p'] < 0.05
        r['g2_pass'] = r['r1_pass']
        n = len(rets); chunk = max(n//3,1)
        subs = [np.mean(rets[i*chunk:(i+1)*chunk])*12/(np.std(rets[i*chunk:(i+1)*chunk])*np.sqrt(12)+1e-10) for i in range(3)]
        r['g3_pass'] = all(s > 0 for s in subs)
        r['sub_sharpes'] = [round(s,2) for s in subs]
        if n > 5:
            nt = max(1,int(n*0.05))
            tr = np.sort(rets)[nt:-nt] if nt < n//2 else rets
            r['g4_pass'] = np.mean(tr)/(np.std(tr)+1e-10) > 0 and np.mean(tr)/(np.std(tr)+1e-10)/(np.mean(rets)/(np.std(rets)+1e-10)+1e-10) > 0.5
        else:
            r['g4_pass'] = False
        r['gates_passed'] = sum([r['g1_pass'],r['g2_pass'],r['g3_pass'],r['g4_pass']])
        fprint(f"\n{r['name']}: G1={'PASS' if r['g1_pass'] else 'FAIL'}(p={r['perm_p']}), "
               f"G2={'PASS' if r['g2_pass'] else 'FAIL'}(gap={r['r1_gap']}), "
               f"G3={'PASS' if r['g3_pass'] else 'FAIL'}, G4={'PASS' if r['g4_pass'] else 'FAIL'} → {r['gates_passed']}/4")

    fprint("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    fprint(f"{'Name':<25} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'PF':>5} {'Gates':>6}")
    fprint("-"*75)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['n_trades']:>6} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>5.2f} {r['gates_passed']:>4}/4")

    save = [{k:v for k,v in r.items() if k!='monthly_returns'} for r in results]
    with open(RESULTS_PATH,'w') as f:
        json.dump(save, f, indent=2, default=str)
    fprint(f"\nDone — {datetime.now().strftime('%H:%M:%S')}")

if __name__ == '__main__':
    main()
