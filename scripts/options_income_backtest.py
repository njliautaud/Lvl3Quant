#!/usr/bin/env python3
"""
Options Income Backtest — Black-Scholes Simulated Options Strategies
Target: Small Robinhood account ($645, Level 2 options)
OOT: 2022-01-01 to 2026-07-30

REALISM NOTES:
- $645 can't buy ATM calls on QQQ ($1000+/contract) or cash-secure puts ($30k+ collateral)
- Must use spreads (defined risk, $100-200 max loss) or cheap underlyings
- SOXL drops to $7-10 in late 2022, making wheel viable then
- Mega-cap earnings often move 5-15%, $2-wing iron butterflies get blown through
- All signals lagged 1 day, $0.65/contract/leg fees
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings('ignore')

STARTING_CAPITAL = 645.0
RF_RATE = 0.045
DIV_YIELD = 0.005
RH_FEE = 0.65
IV_MULT = 1.1
MAX_RISK = 0.20
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
N_PERM = 1000


def bs_call(S, K, T, r, q, s):
    if T <= 0: return max(S-K, 0)
    d1 = (np.log(S/K) + (r-q+0.5*s**2)*T) / (s*np.sqrt(T))
    d2 = d1 - s*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, q, s):
    if T <= 0: return max(K-S, 0)
    d1 = (np.log(S/K) + (r-q+0.5*s**2)*T) / (s*np.sqrt(T))
    d2 = d1 - s*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def strike_from_delta(S, T, r, q, s, delta, otype='put'):
    if T <= 0 or s <= 0: return S
    if otype == 'put':
        nd1 = min(max(abs(delta)/np.exp(-q*T), 0.001), 0.999)
        d1 = -norm.ppf(nd1)
    else:
        nd1 = min(max(delta/np.exp(-q*T), 0.001), 0.999)
        d1 = norm.ppf(nd1)
    return S * np.exp(-(d1*s*np.sqrt(T) - (r-q+0.5*s**2)*T))

def download_data():
    print("Downloading data...")
    tickers = ['QQQ','SPY','SOXL','TQQQ','^VIX','AAPL','MSFT','AMZN','GOOGL','NVDA','META']
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start='2021-06-01', end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return data

def add_indicators(df):
    c = df['Close']
    r = c.pct_change()
    d = c.diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    rs = g / l.replace(0, np.nan)
    df['RSI'] = 100 - (100/(1+rs))
    df['HV20'] = r.rolling(20).std() * np.sqrt(252)
    df['IV'] = df['HV20'] * IV_MULT
    df['Return'] = r
    return df

def giv(prev, default=0.25):
    v = prev.get('IV', default)
    return default if (pd.isna(v) or v <= 0) else v


# ══════════════════════════════════════════════════════════════════════════
# A: QQQ Put Credit Spread — RSI<40, $1 wide, 30-delta
# ══════════════════════════════════════════════════════════════════════════
def strategy_a(data):
    print("\n=== A: QQQ Put Credit Spread (RSI<40) ===")
    df = add_indicators(data['QQQ'].copy()).loc[OOT_START:]
    cap = STARTING_CAPITAL
    trades, active = [], False
    eq, di = [cap], [df.index[0]]
    cd = 0
    for i in range(1, len(df)):
        S, iv, rsi = df.iloc[i]['Close'], giv(df.iloc[i-1], 0.22), df.iloc[i-1].get('RSI', 50)
        date = df.index[i]
        if pd.isna(rsi): rsi = 50
        if cd > 0: cd -= 1

        if not active and cd == 0 and rsi < 40:
            T = 30/252; w = 1
            Ks = round(strike_from_delta(S, T, RF_RATE, DIV_YIELD, iv, -0.30, 'put'))
            Kb = Ks - w
            cr = (bs_put(S, Ks, T, RF_RATE, DIV_YIELD, iv) - bs_put(S, Kb, T, RF_RATE, DIV_YIELD, iv)) * 100
            ml = w*100 - cr; cr -= RH_FEE*4
            if cr > 0 and ml <= cap*MAX_RISK:
                n = min(int(cap*MAX_RISK/max(ml,1)), 3)
                active = True
                td = dict(ed=date, ks=Ks, kb=Kb, cr=cr*n, n=n, d=0, w=w)

        elif active:
            td['d'] += 1
            Tr = max(30-td['d'],0)/252
            sv = (bs_put(S, td['ks'], Tr, RF_RATE, DIV_YIELD, iv) - bs_put(S, td['kb'], Tr, RF_RATE, DIV_YIELD, iv))*100*td['n']
            close = (td['cr'] > 0 and sv <= td['cr']*0.50) or td['d'] >= 30
            if close:
                if td['d'] >= 30:
                    if S >= td['ks']: pnl = td['cr']
                    elif S <= td['kb']: pnl = td['cr'] - td['w']*100*td['n']
                    else: pnl = td['cr'] - (td['ks']-S)*100*td['n']
                else:
                    pnl = td['cr'] - sv - RH_FEE*2*td['n']
                cap += pnl; cap = max(cap, 1); cd = 5; active = False
                trades.append(dict(entry=str(td['ed'].date()), exit=str(date.date()),
                    days=td['d'], pnl=round(pnl,2), type='PUT_SPREAD', premium=round(td['cr'],2), strike=td['ks']))
        di.append(date); eq.append(cap)
    return trades, pd.Series(eq, index=di, name='A_PutSpread')


# ══════════════════════════════════════════════════════════════════════════
# B: Oversold Bull Call Spread on QQQ — RSI<30, $5-wide debit spread
# Can't afford naked calls ($1000+), so buy ATM call + sell $5 OTM call.
# Max cost = call spread debit (~$250-350). Max gain = $500 - debit.
# ══════════════════════════════════════════════════════════════════════════
def strategy_b(data):
    print("\n=== B: QQQ Oversold Bull Call Spread (RSI<30) ===")
    df = add_indicators(data['QQQ'].copy()).loc[OOT_START:]
    cap = STARTING_CAPITAL
    trades, active = [], False
    eq, di = [cap], [df.index[0]]

    for i in range(1, len(df)):
        S, iv, rsi = df.iloc[i]['Close'], giv(df.iloc[i-1], 0.25), df.iloc[i-1].get('RSI', 50)
        date = df.index[i]
        if pd.isna(rsi): rsi = 50

        if not active and rsi < 30:
            T = 30/252; w = 2     # $2 wide to fit $129 max risk
            K_buy = round(S)       # ATM
            K_sell = K_buy + w     # $2 OTM
            buy_c = bs_call(S, K_buy, T, RF_RATE, DIV_YIELD, iv) * 100
            sell_c = bs_call(S, K_sell, T, RF_RATE, DIV_YIELD, iv) * 100
            debit = buy_c - sell_c + RH_FEE * 4  # net cost
            max_gain = w * 100 - debit

            if 10 < debit <= cap * MAX_RISK and max_gain > 0:
                active = True
                td = dict(ed=date, kb=K_buy, ks=K_sell, debit=debit, d=0, w=w)

        elif active:
            td['d'] += 1
            Tr = max(30-td['d'],0)/252
            cur_buy = bs_call(S, td['kb'], Tr, RF_RATE, DIV_YIELD, iv) * 100
            cur_sell = bs_call(S, td['ks'], Tr, RF_RATE, DIV_YIELD, iv) * 100
            spread_val = cur_buy - cur_sell

            gain_pct = (spread_val - td['debit']) / td['debit'] if td['debit'] > 0 else 0
            close = gain_pct >= 1.0 or gain_pct <= -0.50 or td['d'] >= 10

            if close:
                pnl = spread_val - td['debit'] - RH_FEE * 2
                cap += pnl; cap = max(cap, 1); active = False
                trades.append(dict(entry=str(td['ed'].date()), exit=str(date.date()),
                    days=td['d'], pnl=round(pnl,2), type='CALL_SPREAD',
                    premium=round(td['debit'],2), strike=td['kb']))
        di.append(date); eq.append(cap)
    return trades, pd.Series(eq, index=di, name='B_CallSpread')


# ══════════════════════════════════════════════════════════════════════════
# C: Monthly Bull Put Spread on QQQ — systematic, $1 wide
# ══════════════════════════════════════════════════════════════════════════
def strategy_c(data):
    print("\n=== C: QQQ Monthly Bull Put Spread (systematic) ===")
    df = add_indicators(data['QQQ'].copy()).loc[OOT_START:]
    cap = STARTING_CAPITAL
    trades, active = [], False
    eq, di = [cap], [df.index[0]]
    lm = None

    for i in range(1, len(df)):
        S, iv = df.iloc[i]['Close'], giv(df.iloc[i-1], 0.22)
        date = df.index[i]; cm = (date.year, date.month)

        if not active and cm != lm:
            lm = cm; T = 30/252; w = 1
            Ks = round(strike_from_delta(S, T, RF_RATE, DIV_YIELD, iv, -0.30, 'put'))
            Kb = Ks - w
            cr = (bs_put(S, Ks, T, RF_RATE, DIV_YIELD, iv) - bs_put(S, Kb, T, RF_RATE, DIV_YIELD, iv))*100
            ml = w*100-cr; cr -= RH_FEE*4
            if cr > 0 and ml <= cap*MAX_RISK:
                n = min(int(cap*MAX_RISK/max(ml,1)), 3)
                active = True
                td = dict(ed=date, ks=Ks, kb=Kb, cr=cr*n, n=n, d=0, w=w)

        elif active:
            td['d'] += 1
            Tr = max(30-td['d'],0)/252
            sv = (bs_put(S, td['ks'], Tr, RF_RATE, DIV_YIELD, iv) - bs_put(S, td['kb'], Tr, RF_RATE, DIV_YIELD, iv))*100*td['n']
            close = (td['cr'] > 0 and sv <= td['cr']*0.50) or td['d'] >= 30
            if close:
                if td['d'] >= 30:
                    if S >= td['ks']: pnl = td['cr']
                    elif S <= td['kb']: pnl = td['cr'] - td['w']*100*td['n']
                    else: pnl = td['cr'] - (td['ks']-S)*100*td['n']
                else:
                    pnl = td['cr'] - sv - RH_FEE*2*td['n']
                cap += pnl; cap = max(cap,1); active = False
                trades.append(dict(entry=str(td['ed'].date()), exit=str(date.date()),
                    days=td['d'], pnl=round(pnl,2), type='BPS',
                    premium=round(td['cr'],2), k_sell=td['ks'], k_buy=td['kb']))
            if cm != lm: lm = cm
        di.append(date); eq.append(cap)
    return trades, pd.Series(eq, index=di, name='C_BPS')


# ══════════════════════════════════════════════════════════════════════════
# D: Earnings IV Crush — Iron Condor on mega-caps
# Use $5 wings (wider) and only trade when expected IV crush > move.
# Pre-earnings IV = HV*1.3, post = HV*0.80.
# Only enter if stock hasn't had >4% move in prior 5 days (avoid double-counting).
# ══════════════════════════════════════════════════════════════════════════
def strategy_d(data):
    print("\n=== D: Earnings IV Crush (Iron Condor, $5 wings) ===")
    tickers = ['AAPL','MSFT','AMZN','GOOGL','NVDA','META']
    cap = STARTING_CAPITAL
    trades = []

    for tk in tickers:
        if tk not in data: continue
        df = add_indicators(data[tk].copy()).loc[OOT_START:]
        if len(df) < 10: continue
        df['AbsRet'] = df['Return'].abs()

        # Find earnings: |return| > 4% AND no >4% move in prior 5 days
        for idx_pos in range(6, len(df)):
            row = df.iloc[idx_pos]
            if pd.isna(row['Return']) or abs(row['Return']) <= 0.04:
                continue
            # Check no recent big move (avoid double-counting)
            recent_big = any(df.iloc[j]['AbsRet'] > 0.04 for j in range(max(idx_pos-5, 0), idx_pos) if not pd.isna(df.iloc[j].get('AbsRet', 0)))
            if recent_big:
                continue

            entry = df.iloc[idx_pos-1]
            exit_r = df.iloc[idx_pos]
            ed = df.index[idx_pos-1]
            xd = df.index[idx_pos]
            Se, Sx = entry['Close'], exit_r['Close']
            hv = entry.get('HV20', 0.30)
            if pd.isna(hv) or hv <= 0: hv = 0.30

            iv_pre = hv * 1.30
            iv_post = hv * 0.80
            Te, Tx = 7/252, 6/252
            K = round(Se)
            wing = 5

            # Iron condor: sell ATM straddle, buy wings
            cr = ((bs_call(Se,K,Te,RF_RATE,DIV_YIELD,iv_pre) + bs_put(Se,K,Te,RF_RATE,DIV_YIELD,iv_pre))
                 -(bs_call(Se,K+wing,Te,RF_RATE,DIV_YIELD,iv_pre) + bs_put(Se,K-wing,Te,RF_RATE,DIV_YIELD,iv_pre))) * 100
            ml = wing*100 - cr
            cr -= RH_FEE * 8

            if cr <= 0 or ml > cap * MAX_RISK: continue

            # Close next day
            cc = ((bs_call(Sx,K,Tx,RF_RATE,DIV_YIELD,iv_post) + bs_put(Sx,K,Tx,RF_RATE,DIV_YIELD,iv_post))
                 -(bs_call(Sx,K+wing,Tx,RF_RATE,DIV_YIELD,iv_post) + bs_put(Sx,K-wing,Tx,RF_RATE,DIV_YIELD,iv_post))) * 100

            pnl = cr - cc
            pnl = max(pnl, -ml)  # cap at max loss
            pnl = min(pnl, cr)   # cap at max gain

            cap += pnl; cap = max(cap, 1)
            move = abs(exit_r['Return'])*100 if not pd.isna(exit_r['Return']) else 0
            trades.append(dict(entry=str(ed.date()), exit=str(xd.date()), ticker=tk,
                pnl=round(pnl,2), type='IRON_CONDOR', premium=round(cr,2),
                close_cost=round(cc,2), move_pct=round(move,1)))

    if not trades:
        return trades, pd.Series([STARTING_CAPITAL], index=[pd.Timestamp(OOT_START)], name='D_Earnings')

    trades.sort(key=lambda x: x['exit'])
    running = STARTING_CAPITAL
    eq_d, eq_v = [pd.Timestamp(OOT_START)], [STARTING_CAPITAL]
    for t in trades:
        running += t['pnl']; running = max(running, 1)
        eq_d.append(pd.Timestamp(t['exit'])); eq_v.append(running)
    return trades, pd.Series(eq_v, index=eq_d, name='D_Earnings')


# ══════════════════════════════════════════════════════════════════════════
# E: VIX Spike Put Spread on SPY — $2 wide, weekly, VIX > 25
# Reduced cooldown to 1 day for more trades.
# ══════════════════════════════════════════════════════════════════════════
def strategy_e(data):
    print("\n=== E: VIX Spike Put Spread (SPY, VIX>25) ===")
    spy = add_indicators(data['SPY'].copy())
    vix = data['^VIX'].copy()
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    comb = spy[['Close','IV','HV20']].copy()
    comb['VIX'] = vix['Close']
    comb = comb.loc[OOT_START:].dropna()

    cap = STARTING_CAPITAL
    trades, active = [], False
    eq, di = [cap], [comb.index[0]]

    for i in range(1, len(comb)):
        S = comb.iloc[i]['Close']; date = comb.index[i]
        vv = comb.iloc[i-1]['VIX']; iv = giv(comb.iloc[i-1], 0.25)
        if pd.isna(vv): vv = 15

        if not active and vv > 25:
            T = 7/252; w = 2
            tiv = max(iv, vv/100)
            Ks = round(strike_from_delta(S, T, RF_RATE, DIV_YIELD, tiv, -0.35, 'put'))
            Kb = Ks - w
            cr = (bs_put(S, Ks, T, RF_RATE, DIV_YIELD, tiv) - bs_put(S, Kb, T, RF_RATE, DIV_YIELD, tiv))*100
            ml = w*100-cr; cr -= RH_FEE*4
            if cr > 0 and ml <= cap*MAX_RISK:
                n = min(int(cap*MAX_RISK/max(ml,1)), 2)
                active = True
                td = dict(ed=date, ks=Ks, kb=Kb, cr=cr*n, n=n, d=0, w=w, vix=vv, iv=tiv)

        elif active:
            td['d'] += 1
            Tr = max(7-td['d'],0)/252
            cur_iv = giv(comb.iloc[i-1], td['iv'])
            sv = (bs_put(S, td['ks'], Tr, RF_RATE, DIV_YIELD, cur_iv) - bs_put(S, td['kb'], Tr, RF_RATE, DIV_YIELD, cur_iv))*100*td['n']
            close = (td['cr'] > 0 and sv <= td['cr']*0.50) or sv >= td['w']*100*td['n']*0.90 or td['d'] >= 5
            if close:
                if td['d'] >= 5:
                    if S >= td['ks']: pnl = td['cr']
                    elif S <= td['kb']: pnl = td['cr'] - td['w']*100*td['n']
                    else: pnl = td['cr'] - (td['ks']-S)*100*td['n']
                else:
                    pnl = td['cr'] - sv - RH_FEE*2*td['n']
                cap += pnl; cap = max(cap,1); active = False
                trades.append(dict(entry=str(td['ed'].date()), exit=str(date.date()),
                    days=td['d'], pnl=round(pnl,2), type='VIX_SPREAD',
                    premium=round(td['cr'],2), strike=td['ks'], vix_at_entry=round(td['vix'],1)))
        di.append(date); eq.append(cap)
    return trades, pd.Series(eq, index=di, name='E_VIX')


# ══════════════════════════════════════════════════════════════════════════
# F: SOXL Wheel — CSP when affordable, put spreads when not
# ══════════════════════════════════════════════════════════════════════════
def strategy_f(data):
    print("\n=== F: SOXL Wheel ===")
    if 'SOXL' not in data:
        return [], pd.Series([STARTING_CAPITAL], index=[pd.Timestamp(OOT_START)], name='F_Wheel')
    df = add_indicators(data['SOXL'].copy()).loc[OOT_START:]
    cap = STARTING_CAPITAL
    trades = []
    eq, di = [cap], [df.index[0]]
    owns, shares, cb = False, 0, 0
    active, otype = False, None
    lm, td = None, {}

    for i in range(1, len(df)):
        S, iv = df.iloc[i]['Close'], giv(df.iloc[i-1], 0.60)
        date = df.index[i]; cm = (date.year, date.month)

        if not active and cm != lm:
            lm = cm; T = 30/252
            if not owns:
                K = round(S * 0.95); coll = K*100
                if coll <= cap and coll > 0:
                    pr = bs_put(S, K, T, RF_RATE, DIV_YIELD, iv)*100 - RH_FEE*2
                    if pr > 0:
                        active, otype = True, 'CSP'
                        td = dict(ed=date, k=K, pr=pr, d=0)
                else:
                    Ks = round(S*0.95); Kb = Ks-1
                    cr = (bs_put(S,Ks,T,RF_RATE,DIV_YIELD,iv)-bs_put(S,Kb,T,RF_RATE,DIV_YIELD,iv))*100 - RH_FEE*4
                    if cr > 0 and (100-cr) <= cap*MAX_RISK:
                        active, otype = True, 'SPREAD'
                        td = dict(ed=date, ks=Ks, kb=Kb, cr=cr, d=0)
            else:
                K = round(S*1.05)
                pr = bs_call(S, K, T, RF_RATE, DIV_YIELD, iv)*100 - RH_FEE*2
                if pr > 0:
                    active, otype = True, 'CC'
                    td = dict(ed=date, k=K, pr=pr, d=0)

        elif active:
            td['d'] += 1
            if td['d'] >= 21:
                pnl = 0; tt = otype
                if otype == 'CSP':
                    if S <= td['k']:
                        cap -= td['k']*100; cap += td['pr']
                        owns, shares, cb = True, 100, td['k'] - td['pr']/100
                        pnl = td['pr']; tt = 'CSP_ASSIGNED'
                    else:
                        pnl = td['pr']; cap += pnl; tt = 'CSP_EXPIRED'
                elif otype == 'CC':
                    if S >= td['k']:
                        sp = (td['k']-cb)*100
                        cap += td['k']*100 + td['pr']; owns, shares = False, 0
                        pnl = td['pr'] + sp; tt = 'CC_CALLED'
                    else:
                        pnl = td['pr']; cap += pnl; tt = 'CC_EXPIRED'
                elif otype == 'SPREAD':
                    if S >= td['ks']: pnl = td['cr']
                    elif S <= td['kb']: pnl = td['cr'] - 100
                    else: pnl = td['cr'] - (td['ks']-S)*100
                    cap += pnl; tt = 'SPREAD_'+('WIN' if pnl>0 else 'LOSS')
                cap = max(cap, 1); active = False
                trades.append(dict(entry=str(td['ed'].date()), exit=str(date.date()),
                    days=td['d'], pnl=round(pnl,2), type=tt,
                    premium=round(td.get('pr', td.get('cr', 0)),2), strike=td.get('k', td.get('ks',0))))

        pv = cap + (shares*S if owns else 0)
        di.append(date); eq.append(pv)
    return trades, pd.Series(eq, index=di, name='F_Wheel')


# ── Metrics & Validation ─────────────────────────────────────────────────
def metrics(trades, eqc, name, qr=None):
    if not trades:
        return dict(name=name, total_return_pct=0, cagr_pct=0, sharpe=0, sortino=0,
            max_drawdown_pct=0, win_rate_pct=0, profit_factor=0, trade_count=0,
            avg_premium=0, avg_risk=0, corr_qqq=0, bear_2022_ret_pct=0,
            regime_gap=0, final_capital=STARTING_CAPITAL, years=0, no_trades=True)

    eq = eqc[~eqc.index.duplicated(keep='last')].sort_index()
    dr = eq.pct_change().dropna().replace([np.inf,-np.inf], 0)
    tr = (eq.iloc[-1]/eq.iloc[0]-1)*100
    yr = max((eq.index[-1]-eq.index[0]).days/365.25, 0.1)
    cagr = ((eq.iloc[-1]/max(eq.iloc[0],1))**(1/yr)-1)*100
    sh = float((dr.mean()/dr.std())*np.sqrt(252)) if dr.std()>0 else 0
    ds = dr[dr<0]
    so = float((dr.mean()/ds.std())*np.sqrt(252)) if len(ds)>0 and ds.std()>0 else 0
    dd = ((eq-eq.cummax())/eq.cummax()).min()*100

    pnls = [t['pnl'] for t in trades]
    w = [p for p in pnls if p > 0]; l = [p for p in pnls if p <= 0]
    wr = len(w)/len(pnls)*100
    pf = sum(w)/abs(sum(l)) if l and sum(l)!=0 else 999.0
    ap = float(np.mean([abs(t.get('premium',0)) for t in trades]))

    # Avg risk per trade
    avg_risk = float(np.mean([abs(t['pnl']) for t in trades if t['pnl'] < 0])) if l else 0

    co, rg = 0.0, 0.0
    if qr is not None:
        al = pd.DataFrame({'s':dr,'q':qr}).dropna()
        if len(al)>10: co = float(al['s'].corr(al['q']))
        if len(al)>20:
            g2,r2 = al.loc[al['q']>0,'s'], al.loc[al['q']<=0,'s']
            sg = (g2.mean()/g2.std())*np.sqrt(252) if len(g2)>5 and g2.std()>0 else 0
            sr = (r2.mean()/r2.std())*np.sqrt(252) if len(r2)>5 and r2.std()>0 else 0
            rg = abs(sg-sr)/max(abs(sg),abs(sr),0.001)

    be = eq.loc['2022-01-01':'2022-12-31']
    br = float((be.iloc[-1]/be.iloc[0]-1)*100) if len(be)>1 else 0

    return dict(name=name, total_return_pct=round(float(tr),2), cagr_pct=round(float(cagr),2),
        sharpe=round(sh,3), sortino=round(so,3), max_drawdown_pct=round(float(dd),2),
        win_rate_pct=round(wr,1), profit_factor=round(float(pf),3), trade_count=len(trades),
        avg_premium=round(ap,2), avg_risk=round(avg_risk,2), corr_qqq=round(co,3),
        bear_2022_ret_pct=round(br,2), regime_gap=round(float(rg),3),
        final_capital=round(float(eq.iloc[-1]),2), years=round(float(yr),2))

def perm_test(trades, n=N_PERM):
    if len(trades) < 5: return 1.0
    pnls = np.array([t['pnl'] for t in trades])
    actual = np.mean(pnls)
    rng = np.random.default_rng(42)
    return sum(1 for _ in range(n) if np.mean(pnls*rng.choice([-1,1],size=len(pnls))) >= actual) / n

def validate(m, trades):
    g = {'sharpe_gt_0.5': m.get('sharpe',0) > 0.5,
         'perm_p_lt_0.05': perm_test(trades) < 0.05,
         'regime_gap_lt_0.5': m.get('regime_gap',1) < 0.5,
         'max_dd_gt_neg50': m.get('max_drawdown_pct',-100) > -50,
         'trades_gte_20': m.get('trade_count',0) >= 20}
    g['all_pass'] = all(g.values())
    return {k:bool(v) for k,v in g.items()}


def main():
    data = download_data()
    if 'QQQ' not in data:
        print("FATAL: Missing QQQ"); return

    qqq = add_indicators(data['QQQ'].copy()).loc[OOT_START:]
    qr = qqq['Return'].dropna()

    strats = [('A_PutSpread_RSI', strategy_a), ('B_CallSpread_RSI', strategy_b),
              ('C_MonthlyBPS', strategy_c), ('D_EarningsCrush', strategy_d),
              ('E_VIXspikePuts', strategy_e), ('F_SOXLwheel', strategy_f)]

    res = {}
    for name, fn in strats:
        try:
            trades, eq = fn(data)
            m = metrics(trades, eq, name, qr)
            gates = validate(m, trades)
            pp = perm_test(trades)
            m['permutation_p'] = round(pp, 4)
            m['validation_gates'] = gates
            m['trades_detail'] = trades[:10]
            res[name] = m

            print(f"\n{'='*62}")
            print(f"  {name}  |  {m['trade_count']} trades  |  Final ${m['final_capital']:.0f}")
            print(f"{'='*62}")
            print(f"  Return {m['total_return_pct']:.1f}% | CAGR {m['cagr_pct']:.1f}% | Sharpe {m['sharpe']:.3f} | Sortino {m['sortino']:.3f}")
            print(f"  MaxDD {m['max_drawdown_pct']:.1f}% | WR {m['win_rate_pct']:.1f}% | PF {m['profit_factor']:.2f} | AvgPrem ${m['avg_premium']:.0f}")
            print(f"  QQQ corr {m['corr_qqq']:.3f} | Bear2022 {m['bear_2022_ret_pct']:.1f}% | RegimeGap {m['regime_gap']:.3f} | Perm_p {pp:.4f}")
            print(f"  GATES: {'PASS' if gates['all_pass'] else 'FAIL'} -- {gates}")
        except Exception as e:
            import traceback; traceback.print_exc()
            res[name] = dict(name=name, error=str(e))

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"{'Strategy':<18} {'Ret%':>7} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>6} {'#Tr':>5} {'Perm_p':>7} {'Pass':>5}")
    print("-"*100)
    for n2, m2 in res.items():
        if 'error' in m2:
            print(f"{n2:<18} ERROR: {m2['error'][:60]}")
            continue
        p2 = 'YES' if m2.get('validation_gates',{}).get('all_pass') else 'NO'
        print(f"{n2:<18} {m2['total_return_pct']:>7.1f} {m2['cagr_pct']:>7.1f} {m2['sharpe']:>7.3f} "
              f"{m2['sortino']:>8.3f} {m2['max_drawdown_pct']:>7.1f} {m2['win_rate_pct']:>6.1f} "
              f"{m2['profit_factor']:>6.2f} {m2['trade_count']:>5} {m2['permutation_p']:>7.4f} {p2:>5}")

    qt = float((qqq['Close'].iloc[-1]/qqq['Close'].iloc[0]-1)*100)
    qy = max((qqq.index[-1]-qqq.index[0]).days/365.25, 0.1)
    qc = float(((qqq['Close'].iloc[-1]/qqq['Close'].iloc[0])**(1/qy)-1)*100)
    qs = float((qr.mean()/qr.std())*np.sqrt(252)) if qr.std()>0 else 0
    print(f"\n{'QQQ Buy&Hold':<18} {qt:>7.1f} {qc:>7.1f} {qs:>7.3f}")
    print(f"\nStarting capital: ${STARTING_CAPITAL:.0f}")

    viable = {k:v for k,v in res.items() if 'error' not in v and v.get('trade_count',0) > 0}
    if viable:
        best = max(viable.items(), key=lambda x: x[1]['sharpe'])
        print(f"Best risk-adjusted: {best[0]} (Sharpe {best[1]['sharpe']:.3f}, Return {best[1]['total_return_pct']:.1f}%)")
        passed = [k for k,v in res.items() if v.get('validation_gates',{}).get('all_pass')]
        if passed:
            print(f"Strategies passing ALL gates: {', '.join(passed)}")
        else:
            print("No strategy passed ALL validation gates.")

    out = Path('/home/jupiter/Lvl3Quant/data/options_income_results.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
