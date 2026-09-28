#!/usr/bin/env python3
"""
Post-Earnings Bounce Options v1 — Buy cheap calls AFTER earnings drops.

Validated equity signal: 63% WR, PF 1.70, Sharpe 1.50 (buy after 8%+ post-earnings drop, hold 10d).
Key edge: IV crushes after earnings → options are CHEAP.
This tests whether the equity edge transfers to options for the agentic account ($645).
"""

import os, json, time, warnings, math
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

warnings.filterwarnings('ignore')

# -- Config --
UNIVERSE = ['F','SOFI','RIVN','HOOD','MARA','SNAP','PLTR','AMD','UBER','LYFT',
            'SQ','COIN','OPEN','NIO','LCID','DKNG','RBLX','PINS','ROKU','UPST',
            'AAL','DAL','UAL','CCL','RCL','NCLH','PYPL','ABNB']
START_CAPITAL = 645.0
MAX_CONCURRENT = 3
MAX_COST_PCT = 0.50  # max 50% of capital per trade
MAX_COST_ABS = 300.0
COMMISSION_PER_LEG = 0.65
SLIPPAGE_PCT = 0.05

# Black-Scholes call price
def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(S - K, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)

def compute_rv(prices, window=20):
    """20-day realized vol, annualized."""
    lr = np.log(prices / prices.shift(1))
    return lr.rolling(window).std() * np.sqrt(252)

import yfinance as yf

print("=" * 70)
print("POST-EARNINGS BOUNCE OPTIONS v1")
print("Buy ATM calls after 8%+ post-earnings drops, hold 10 days")
print("=" * 70)

# Load data
print("\nLoading data...")
t0 = time.time()
all_data = {}
all_earnings = {}

for ticker in UNIVERSE:
    try:
        tk = yf.Ticker(ticker)
        h = tk.history(start='2017-01-01', end='2026-07-01', auto_adjust=True)
        if h.index.tz is not None:
            h.index = h.index.tz_convert(None)
        if len(h) < 300:
            continue
        all_data[ticker] = h
        
        # Get earnings dates
        try:
            ed = tk.get_earnings_dates(limit=100)
            if ed is not None and len(ed) > 0:
                if ed.index.tz is not None:
                    ed.index = ed.index.tz_convert(None)
                all_earnings[ticker] = sorted(ed.index.tolist())
        except:
            pass
    except Exception as e:
        pass

print(f"Loaded {len(all_data)} stocks, {len(all_earnings)} with earnings data ({time.time()-t0:.1f}s)")

# Get SPY for regime classification
spy = yf.Ticker('SPY').history(start='2017-01-01', end='2026-07-01', auto_adjust=True)
if spy.index.tz is not None:
    spy.index = spy.index.tz_convert(None)
spy_ret = spy['Close'].pct_change()

# Generate signals
print("\nScanning for post-earnings bounce signals...")
signals = []

for ticker in sorted(all_earnings.keys()):
    if ticker not in all_data:
        continue
    h = all_data[ticker]
    close = h['Close']
    rv = compute_rv(close)
    
    for ed in all_earnings[ticker]:
        # Find the trading day of/after earnings
        mask = close.index >= ed
        if mask.sum() < 3:
            continue
        ed_idx = close.index[mask][0]
        
        # Need at least 2 days before earnings for return calc
        ed_loc = close.index.get_loc(ed_idx)
        if ed_loc < 2 or ed_loc + 12 > len(close):
            continue
        
        # 2-day return around earnings
        pre_price = close.iloc[ed_loc - 2]
        post_price = close.iloc[ed_loc]
        ret_2d = (post_price / pre_price) - 1
        
        # Signal: dropped 8%+ in 2 days around earnings
        if ret_2d <= -0.08:
            # Entry: 1 business day after earnings
            entry_idx = ed_loc + 1
            if entry_idx >= len(close):
                continue
            entry_date = close.index[entry_idx]
            entry_price = close.iloc[entry_idx]
            
            # Exit: 10 trading days later
            exit_idx = min(entry_idx + 10, len(close) - 1)
            exit_date = close.index[exit_idx]
            exit_price = close.iloc[exit_idx]
            
            # Realized vol at entry
            rv_at_entry = rv.iloc[entry_idx] if entry_idx < len(rv) else 0.30
            if pd.isna(rv_at_entry) or rv_at_entry <= 0:
                rv_at_entry = 0.30
            
            # IV post-earnings (crushed to near realized)
            iv = rv_at_entry * 1.0  # post-crush, IV ≈ realized
            
            signals.append({
                'ticker': ticker,
                'earnings_date': ed,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'drop_pct': ret_2d * 100,
                'rv': rv_at_entry,
                'iv': iv,
            })

signals.sort(key=lambda x: x['entry_date'])
print(f"Found {len(signals)} post-earnings bounce signals")

# Simulate for multiple variants
VARIANTS = {
    'atm_call_10d':     {'delta': 0,    'hold': 10, 'dte': 21, 'spread': False},
    'itm_call_10d':     {'delta': -0.02, 'hold': 10, 'dte': 21, 'spread': False},  # 2% ITM
    'otm_call_10d':     {'delta': 0.03, 'hold': 10, 'dte': 21, 'spread': False},   # 3% OTM
    'spread_10d':       {'delta': 0,    'hold': 10, 'dte': 21, 'spread': True},     # bull spread
    'atm_call_5d':      {'delta': 0,    'hold': 5,  'dte': 14, 'spread': False},
    'itm_call_5d':      {'delta': -0.02, 'hold': 5, 'dte': 14, 'spread': False},
}

results = []

for vname, vconfig in VARIANTS.items():
    capital = START_CAPITAL
    open_positions = []
    trades = []
    equity_curve = {signals[0]['entry_date'].strftime('%Y-%m-%d'): capital} if signals else {}
    
    for sig in signals:
        # Close expired positions
        new_open = []
        for pos in open_positions:
            if sig['entry_date'] >= pos['exit_date']:
                # Reprice at exit
                S_exit = pos['exit_spot']
                K = pos['strike']
                T_exit = max(pos['dte_at_exit'] / 252, 0.001)
                sigma = pos['iv']
                exit_val = bs_call(S_exit, K, T_exit, 0.05, sigma)
                
                if vconfig['spread']:
                    K2 = pos.get('strike2', K * 1.05)
                    exit_val2 = bs_call(S_exit, K2, T_exit, 0.05, sigma)
                    exit_val = exit_val - exit_val2
                
                exit_val = max(exit_val, 0) * 100  # per contract
                exit_val *= (1 - SLIPPAGE_PCT)  # slippage
                
                cost_exit = COMMISSION_PER_LEG * (2 if vconfig['spread'] else 1)
                pnl = exit_val - pos['cost'] - cost_exit
                capital += exit_val - cost_exit
                
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'pnl': pnl,
                    'return_pct': pnl / pos['cost'] * 100 if pos['cost'] > 0 else 0,
                })
            else:
                new_open.append(pos)
        open_positions = new_open
        
        # Skip if too many open
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        
        # Price option at entry
        S = sig['entry_price']
        strike_offset = vconfig['delta']
        K = S * (1 + strike_offset)
        T = vconfig['dte'] / 252
        sigma = sig['iv']
        
        call_price = bs_call(S, K, T, 0.05, sigma)
        
        if vconfig['spread']:
            K2 = S * 1.05
            call_price2 = bs_call(S, K2, T, 0.05, sigma)
            net_price = call_price - call_price2
        else:
            net_price = call_price
        
        cost = net_price * 100 * (1 + SLIPPAGE_PCT)  # per contract, with slippage
        cost += COMMISSION_PER_LEG * (2 if vconfig['spread'] else 1)
        
        if cost > capital * MAX_COST_PCT or cost > MAX_COST_ABS or cost <= 0:
            continue
        
        capital -= cost
        
        # Compute exit spot
        hold_days = vconfig['hold']
        entry_loc = sig['entry_price']  # approximate
        
        # Find actual exit
        if sig['ticker'] in all_data:
            h = all_data[sig['ticker']]
            close = h['Close']
            try:
                entry_idx_loc = close.index.get_loc(sig['entry_date'])
                exit_idx_loc = min(entry_idx_loc + hold_days, len(close) - 1)
                exit_spot = close.iloc[exit_idx_loc]
                exit_date = close.index[exit_idx_loc]
                dte_at_exit = max(vconfig['dte'] - hold_days, 1)
            except:
                exit_spot = sig['exit_price']
                exit_date = sig['exit_date']
                dte_at_exit = max(vconfig['dte'] - 10, 1)
        else:
            exit_spot = sig['exit_price']
            exit_date = sig['exit_date']
            dte_at_exit = max(vconfig['dte'] - 10, 1)
        
        open_positions.append({
            'ticker': sig['ticker'],
            'entry_date': sig['entry_date'],
            'exit_date': exit_date,
            'exit_spot': exit_spot,
            'strike': K,
            'strike2': S * 1.05 if vconfig['spread'] else None,
            'iv': sigma,
            'cost': cost,
            'dte_at_exit': dte_at_exit,
        })
        
        equity_curve[sig['entry_date'].strftime('%Y-%m-%d')] = capital
    
    # Close remaining positions
    for pos in open_positions:
        S_exit = pos['exit_spot']
        K = pos['strike']
        T_exit = max(pos['dte_at_exit'] / 252, 0.001)
        sigma = pos['iv']
        exit_val = bs_call(S_exit, K, T_exit, 0.05, sigma)
        
        if vconfig['spread']:
            K2 = pos.get('strike2', K * 1.05)
            exit_val2 = bs_call(S_exit, K2, T_exit, 0.05, sigma)
            exit_val = exit_val - exit_val2
        
        exit_val = max(exit_val, 0) * 100
        exit_val *= (1 - SLIPPAGE_PCT)
        cost_exit = COMMISSION_PER_LEG * (2 if vconfig['spread'] else 1)
        pnl = exit_val - pos['cost'] - cost_exit
        capital += exit_val - cost_exit
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
            'pnl': pnl,
            'return_pct': pnl / pos['cost'] * 100 if pos['cost'] > 0 else 0,
        })
    
    # Compute metrics
    if not trades:
        results.append({'variant': vname, 'total_trades': 0, 'sharpe': 0})
        continue
    
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    total_pnl = sum(pnls)
    wr = len(wins) / len(pnls) if pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99
    
    # Simple Sharpe from trade returns
    if len(pnls) > 1:
        ret_arr = np.array(pnls) / START_CAPITAL
        sharpe = np.mean(ret_arr) / np.std(ret_arr) * np.sqrt(len(pnls)) if np.std(ret_arr) > 0 else 0
    else:
        sharpe = 0
    
    # Sortino
    neg_rets = [r for r in np.array(pnls)/START_CAPITAL if r < 0]
    downside_std = np.std(neg_rets) if neg_rets else 0.001
    sortino = np.mean(np.array(pnls)/START_CAPITAL) / downside_std * np.sqrt(len(pnls)) if downside_std > 0 else 0
    
    # CAGR
    years = max((pd.Timestamp(trades[-1]['entry_date']) - pd.Timestamp(trades[0]['entry_date'])).days / 365.25, 0.5)
    cagr = (capital / START_CAPITAL) ** (1/years) - 1 if capital > 0 else -1
    
    # MaxDD (simple from trade PnL)
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum + START_CAPITAL)
    dd = (cum + START_CAPITAL - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0
    
    # Regime classification
    regime_pnls = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        td = pd.Timestamp(t['entry_date'])
        try:
            idx = spy_ret.index.get_indexer([td], method='nearest')[0]
            sr = spy_ret.iloc[idx]
            if sr > 0.002: regime = 'green'
            elif sr < -0.002: regime = 'red'
            else: regime = 'flat'
        except:
            regime = 'flat'
        regime_pnls[regime].append(t['pnl'])
    
    regime_sharpes = {}
    for r, rp in regime_pnls.items():
        if len(rp) > 1:
            ra = np.array(rp) / START_CAPITAL
            regime_sharpes[r] = float(np.mean(ra) / np.std(ra) * np.sqrt(len(ra))) if np.std(ra) > 0 else 0
        else:
            regime_sharpes[r] = 0
    
    # Permutation test (100 shuffles for speed)
    perm_sharpes = []
    for _ in range(100):
        shuffled = np.random.permutation(pnls)
        ra = shuffled / START_CAPITAL
        if np.std(ra) > 0:
            perm_sharpes.append(float(np.mean(ra) / np.std(ra) * np.sqrt(len(ra))))
    p_value = np.mean([s >= sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    
    # Sub-period test
    half = len(trades) // 2
    first_pnl = sum(pnls[:half])
    second_pnl = sum(pnls[half:])
    
    # Outlier test
    sorted_pnls = sorted(pnls, reverse=True)
    n_remove = max(1, int(len(pnls) * 0.05))
    remaining_pnl = sum(sorted_pnls[n_remove:])
    
    r = {
        'variant': vname,
        'total_trades': len(trades),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr_pct': round(cagr * 100, 1),
        'max_dd_pct': round(max_dd * 100, 1),
        'calmar': round(cagr / max_dd, 2) if max_dd > 0 else 0,
        'final_capital': round(capital, 2),
        'regime_sharpes': regime_sharpes,
        'gates': {
            'permutation': {'p_value': round(float(p_value), 3), 'pass': p_value < 0.05},
            'regime': {
                'sharpes': regime_sharpes,
                'pass': True  # calculate below
            },
            'sub_period': {'first_half': round(first_pnl, 2), 'second_half': round(second_pnl, 2), 
                          'pass': first_pnl > 0 and second_pnl > 0},
            'outlier': {'remaining_pnl': round(remaining_pnl, 2), 'pass': remaining_pnl > 0},
        },
        'sample_trades': trades[:5],
    }
    
    # Regime divergence
    gs = regime_sharpes.get('green', 0)
    rs = regime_sharpes.get('red', 0)
    max_s = max(abs(gs), abs(rs))
    div = abs(gs - rs) / max_s if max_s > 0 else 0
    r['gates']['regime']['divergence'] = round(div, 3)
    r['gates']['regime']['pass'] = div <= 0.50
    
    results.append(r)
    
    gates_pass = sum(1 for g in r['gates'].values() if g.get('pass'))
    
    perm_flag = 'P' if r['gates']['permutation']['pass'] else 'F'
    regime_flag = 'P' if r['gates']['regime']['pass'] else 'F'
    sub_flag = 'P' if r['gates']['sub_period']['pass'] else 'F'
    out_flag = 'P' if r['gates']['outlier']['pass'] else 'F'
    
    print(f"\n{vname:20s}: Sharpe={sharpe:.3f} Sort={sortino:.3f} WR={wr*100:.1f}% PF={pf:.2f} CAGR={cagr*100:.1f}% DD={max_dd*100:.1f}% ${START_CAPITAL:.0f}→${capital:.0f} trades={len(trades)} [{perm_flag}{regime_flag}{sub_flag}{out_flag}] {gates_pass}/4")

# Save results
results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research', 'findings', 'post_earnings_bounce_options_v1_results.json')
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump({
        'metadata': {
            'script': 'post_earnings_bounce_options_v1.py',
            'signal': 'Buy calls 1d after 8%+ post-earnings drop, hold 5-10d',
            'universe': len(all_data),
            'total_signals': len(signals),
            'starting_capital': START_CAPITAL,
        },
        'variants': results,
        'best_variant': results[0]['variant'] if results else None,
    }, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"Results saved. Best variant: {results[0]['variant'] if results else 'none'}")
print(f"Total signals found: {len(signals)}")
print(f"{'='*70}")
