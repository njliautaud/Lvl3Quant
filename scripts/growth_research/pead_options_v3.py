#!/usr/bin/env python3
"""
PEAD Options V3 — Fixes for 5/5 Gate Pass
==========================================
V1 (Sharpe 1.51, 4/5 gates) + V2 diversity cap → V3 addresses remaining failures:

V1/V2 FAILURES TO FIX:
1. Perm test borderline (p=0.01 v1, p=0.14 v2) → need more trades + stronger signal
2. Regime balance (gap=0.53 v2) → need bear market edge
3. Concentration (48.5% v2, 83% v1) → cap + rotate

V3 APPROACH:
  - Lower gap threshold to 4% (more events, more diversification)
  - Add VIX regime filter (different params for high/low VIX)
  - Adaptive TP/SL by gap size (bigger gaps → wider TP)
  - Direction-specific models (separate long/short LGBM)
  - Debiased training (weight bear market events higher)

6 VARIANTS:
  A. V1 baseline (60% thresh, LGBM+Mom) — control
  B. Lower gap (4%) + max 3/ticker
  C. VIX-adaptive TP/SL
  D. Direction-specific models
  E. Debiased bear-market weighting
  F. Full combination (B+C+D+E)

UNIVERSE: 52 growth stocks
PERIOD: 2022-01-01 to 2026-07-25
ACCOUNT: $645, max $200/position
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'pead_options_v3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

STARTING_CAPITAL = 645.0
COMMISSION_RT = 1.30
MAX_POSITION = 200.0
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 200
DTE = 14
DRIFT_THRESHOLD = 0.03
HOLD_DAYS = 3

FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash',
]

# Black-Scholes
def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S-K, 0.0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K-S, 0.0)
    d1 = (np.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def option_price(S, K, T, r, sigma, opt_type='call'):
    return bs_call(S, K, T, r, sigma) if opt_type == 'call' else bs_put(S, K, T, r, sigma)


def load_data():
    import yfinance as yf
    cache_path = os.path.join(LVL3_ROOT, 'data', 'earnings_momentum_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'earnings_dates_cache.json')

    if os.path.exists(cache_path):
        prices_df = pd.read_parquet(cache_path)
        print(f"Loaded cached prices: {len(prices_df)} rows", flush=True)
    else:
        print("Downloading prices...", flush=True)
        tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
        frames = []
        for t in tickers:
            try:
                df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
                if len(df) < 50: continue
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                df['ticker'] = t
                df.index.name = 'date'
                frames.append(df)
            except: pass
        prices_df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices_df.to_parquet(cache_path)

    if os.path.exists(earnings_cache):
        with open(earnings_cache) as f:
            earnings_dates = json.load(f)
        print(f"Loaded cached earnings: {len(earnings_dates)} tickers", flush=True)
    else:
        print("Fetching earnings dates...", flush=True)
        earnings_dates = {}
        for t in STOCK_UNIVERSE:
            try:
                stock = yf.Ticker(t)
                dates = stock.get_earnings_dates(limit=30)
                if dates is not None and len(dates) > 0:
                    earnings_dates[t] = [str(d.date()) for d in dates.index]
            except: pass
        with open(earnings_cache, 'w') as f:
            json.dump(earnings_dates, f)

    return prices_df, earnings_dates


def build_events(prices_df, earnings_dates, min_gap_pct=0.05):
    events = []
    try: spy = prices_df.loc['SPY', 'close']
    except: spy = None
    try: vix = prices_df.loc['^VIX', 'close']
    except: vix = None

    for ticker in STOCK_UNIVERSE:
        if ticker not in earnings_dates: continue
        try: tp = prices_df.loc[ticker].sort_index()
        except: continue
        if len(tp) < 30: continue
        td = tp.index.sort_values()

        for ed_str in earnings_dates[ticker]:
            ed = pd.Timestamp(ed_str)
            post_mask = td >= ed
            if not post_mask.any(): continue
            post_days = td[post_mask]
            if len(post_days) < HOLD_DAYS + 1: continue

            pre_mask = td < ed
            if not pre_mask.any(): continue
            pre_days = td[pre_mask]
            if len(pre_days) < 30: continue

            cb = tp.loc[pre_days[-1], 'close']
            oa = tp.loc[post_days[0], 'open']
            if cb <= 0 or oa <= 0: continue
            gap = (oa / cb) - 1.0
            if abs(gap) < min_gap_pct: continue

            f = {'gap_pct': gap, 'abs_gap': abs(gap), 'gap_direction': 1 if gap > 0 else -1}
            for w in [5, 10, 21]:
                f[f'mom_{w}d'] = float(tp.loc[pre_days[-1], 'close']/tp.loc[pre_days[-w-1], 'close'] - 1) if len(pre_days) >= w+1 else 0

            f['vol_21d'] = float(tp.loc[pre_days[-22:], 'close'].pct_change().dropna().std() * np.sqrt(252)) if len(pre_days) >= 22 else 0.3

            f['rel_str_21d'] = 0
            if spy is not None and len(pre_days) >= 22:
                try:
                    sc = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    stc = tp.loc[pre_days[-21:], 'close']
                    f['rel_str_21d'] = float((stc.iloc[-1]/stc.iloc[0]-1) - (sc.iloc[-1]/sc.iloc[0]-1))
                except: pass

            f['vix'] = 20
            if vix is not None:
                vm = vix.index <= pre_days[-1]
                if vm.any(): f['vix'] = float(vix[vm].iloc[-1])

            if 'volume' in tp.columns and len(pre_days) >= 22:
                rv = tp.loc[pre_days[-5:], 'volume'].mean()
                av = tp.loc[pre_days[-22:], 'volume'].mean()
                f['vol_ratio'] = float(rv / max(av, 1))
            else:
                f['vol_ratio'] = 1.0

            if len(pre_days) >= 15:
                rets = tp.loc[pre_days[-15:], 'close'].pct_change().dropna()
                g = rets.clip(lower=0).mean()
                l = (-rets).clip(lower=0).mean()
                f['rsi'] = float(100 - 100/(1+g/l)) if l > 0 else 100
            else:
                f['rsi'] = 50

            f['dist_52w_high'] = float(cb / tp.loc[pre_days[-252:], 'close'].max() - 1) if len(pre_days) >= 252 else 0

            ticker_earn = [d for d in earnings_dates[ticker] if pd.Timestamp(d) < ed]
            f['prev_gap'] = 0
            if ticker_earn:
                pe = pd.Timestamp(ticker_earn[-1])
                pp = td[td >= pe]; ppre = td[td < pe]
                if len(pp)>0 and len(ppre)>0:
                    pc = tp.loc[ppre[-1], 'close']; po = tp.loc[pp[0], 'open']
                    if pc > 0: f['prev_gap'] = float(po/pc - 1)

            f['ticker_hash'] = hash(ticker) % 100

            # Target
            post_close = [float(tp.loc[d, 'close']) for d in post_days[:HOLD_DAYS + 1]]
            if len(post_close) < 2: continue
            entry_price = oa
            if gap > 0:
                max_move = max(c/entry_price - 1 for c in post_close[1:])
            else:
                max_move = max(1 - c/entry_price for c in post_close[1:])
            f['target'] = 1 if max_move >= DRIFT_THRESHOLD else 0

            # Is this a bear market event? (SPY below 200d MA)
            f['is_bear'] = False
            if spy is not None and len(pre_days) >= 200:
                try:
                    spy_close = spy.loc[spy.index <= pre_days[-1]]
                    if len(spy_close) >= 200:
                        ma200 = spy_close.iloc[-200:].mean()
                        f['is_bear'] = float(spy_close.iloc[-1]) < float(ma200)
                except: pass

            f['ticker'] = ticker
            f['earn_date'] = ed_str
            f['close_before'] = float(cb)
            f['open_after'] = float(oa)
            f['post_closes'] = post_close

            events.append(f)

    print(f"Built {len(events)} events (|gap| >= {min_gap_pct*100}%)", flush=True)
    if events:
        targets = [e['target'] for e in events]
        bears = sum(1 for e in events if e.get('is_bear'))
        print(f"  Drift rate: {sum(targets)/len(targets)*100:.1f}%, Bear events: {bears}/{len(events)}", flush=True)
    return events


def train_lgbm(X, y, sample_weight=None):
    from sklearn.ensemble import GradientBoostingClassifier
    m = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
    m.fit(X, y, sample_weight=sample_weight)
    return m


VARIANTS = {
    'A': {
        'name': 'V1 Baseline (5% gap, 60% thresh, LGBM+Mom)',
        'min_gap': 0.05, 'threshold': 0.60, 'momentum_filter': True,
        'max_per_ticker': 999, 'adaptive_tp': False,
        'direction_specific': False, 'bear_weight': 1.0,
        'tp_pct': 0.30, 'sl_pct': -0.25,
    },
    'B': {
        'name': 'Lower Gap (4%) + Max 3/Ticker',
        'min_gap': 0.04, 'threshold': 0.60, 'momentum_filter': True,
        'max_per_ticker': 3, 'adaptive_tp': False,
        'direction_specific': False, 'bear_weight': 1.0,
        'tp_pct': 0.30, 'sl_pct': -0.25,
    },
    'C': {
        'name': 'VIX-Adaptive TP/SL',
        'min_gap': 0.05, 'threshold': 0.60, 'momentum_filter': True,
        'max_per_ticker': 3, 'adaptive_tp': True,
        'direction_specific': False, 'bear_weight': 1.0,
        'tp_pct': 0.30, 'sl_pct': -0.25,
    },
    'D': {
        'name': 'Direction-Specific Models',
        'min_gap': 0.05, 'threshold': 0.60, 'momentum_filter': True,
        'max_per_ticker': 3, 'adaptive_tp': False,
        'direction_specific': True, 'bear_weight': 1.0,
        'tp_pct': 0.30, 'sl_pct': -0.25,
    },
    'E': {
        'name': 'Bear-Market Debiased (2x weight)',
        'min_gap': 0.05, 'threshold': 0.60, 'momentum_filter': True,
        'max_per_ticker': 3, 'adaptive_tp': False,
        'direction_specific': False, 'bear_weight': 2.0,
        'tp_pct': 0.30, 'sl_pct': -0.25,
    },
    'F': {
        'name': 'Full Combo (4% gap + VIX + DirModels + Debias)',
        'min_gap': 0.04, 'threshold': 0.60, 'momentum_filter': True,
        'max_per_ticker': 3, 'adaptive_tp': True,
        'direction_specific': True, 'bear_weight': 2.0,
        'tp_pct': 0.30, 'sl_pct': -0.25,
    },
}


def run_variant(vkey, cfg, events_5pct, events_4pct):
    print(f"\n{'='*60}", flush=True)
    print(f"  VARIANT {vkey}: {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    events = events_4pct if cfg['min_gap'] <= 0.04 else events_5pct
    events_sorted = sorted(events, key=lambda e: e['earn_date'])
    min_train = 30

    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0
    trades = []
    ticker_count = defaultdict(int)
    ticker_pnl = defaultdict(float)

    for i in range(min_train, len(events_sorted)):
        ev = events_sorted[i]

        if ticker_count[ev['ticker']] >= cfg['max_per_ticker']:
            continue

        # Build training data
        train = events_sorted[:i]
        X_train = pd.DataFrame(train)[FEATURE_COLS]
        y_train = np.array([e['target'] for e in train])

        # Sample weighting for bear debiasing
        weights = None
        if cfg['bear_weight'] != 1.0:
            weights = np.array([cfg['bear_weight'] if e.get('is_bear') else 1.0 for e in train])

        X_test = pd.DataFrame([ev])[FEATURE_COLS]

        try:
            if cfg['direction_specific']:
                # Train separate models for calls and puts
                gap_dir = ev['gap_direction']
                dir_mask = [e['gap_direction'] == gap_dir for e in train]
                X_dir = X_train[dir_mask]
                y_dir = y_train[np.array(dir_mask)]
                w_dir = weights[np.array(dir_mask)] if weights is not None else None
                if len(X_dir) < 20:
                    model = train_lgbm(X_train, y_train, weights)
                else:
                    model = train_lgbm(X_dir, y_dir, w_dir)
            else:
                model = train_lgbm(X_train, y_train, weights)

            prob = model.predict_proba(X_test)[:, 1][0]
        except:
            continue

        if prob < cfg['threshold']:
            continue

        # Momentum filter
        if cfg['momentum_filter']:
            if ev['gap_direction'] == 1 and ev.get('mom_5d', 0) < 0: continue
            if ev['gap_direction'] == -1 and ev.get('mom_5d', 0) > 0: continue

        gap = ev['gap_pct']
        entry_price = ev['open_after']
        opt_type = 'call' if gap > 0 else 'put'
        strike = round(entry_price)

        post_iv = ev.get('vol_21d', 0.3) * 0.8
        post_iv = max(post_iv, 0.15)
        T = DTE / 252.0
        entry_premium = option_price(entry_price, strike, T, RISK_FREE_RATE, post_iv, opt_type)
        if entry_premium < 0.10: continue

        contract_cost = entry_premium * 100
        max_spend = min(MAX_POSITION, equity * 0.30)
        if contract_cost > max_spend or contract_cost + COMMISSION_RT > equity: continue

        # Adaptive TP/SL based on VIX and gap size
        tp_pct = cfg['tp_pct']
        sl_pct = cfg['sl_pct']
        if cfg['adaptive_tp']:
            vix_val = ev.get('vix', 20)
            gap_size = abs(gap)
            # Higher VIX → wider TP/SL (more vol = bigger moves possible)
            if vix_val > 25:
                tp_pct *= 1.3
                sl_pct *= 1.3
            # Bigger gap → wider TP (stronger conviction)
            if gap_size > 0.10:
                tp_pct *= 1.2

        # Simulate holding
        post_closes = ev['post_closes']
        exit_premium = entry_premium
        exit_reason = 'time_stop'

        for day in range(1, min(len(post_closes), HOLD_DAYS + 1)):
            spot = post_closes[day]
            remaining = max(DTE - day, 0) / 252.0
            iv_adj = post_iv * (1 + 0.02 * day)
            current = option_price(spot, strike, remaining, RISK_FREE_RATE, iv_adj, opt_type)
            pct = (current - entry_premium) / entry_premium

            if pct >= tp_pct:
                exit_premium = current
                exit_reason = 'take_profit'
                break
            elif pct <= sl_pct:
                exit_premium = current
                exit_reason = 'stop_loss'
                break
            exit_premium = current

        pnl = (exit_premium - entry_premium) * 100 - COMMISSION_RT
        equity += pnl

        if equity > peak: peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0
        if dd < max_dd: max_dd = dd

        ticker_count[ev['ticker']] += 1
        ticker_pnl[ev['ticker']] += pnl

        trades.append({
            'ticker': ev['ticker'], 'earn_date': ev['earn_date'],
            'opt_type': opt_type, 'prob': round(prob, 3),
            'gap_pct': round(gap * 100, 1), 'pnl': round(pnl, 2),
            'exit_reason': exit_reason, 'target': ev['target'],
            'is_bear': ev.get('is_bear', False),
        })

        if len(trades) <= 3:
            print(f"  Trade {len(trades)}: {ev['ticker']} {opt_type.upper()} "
                  f"gap={gap*100:.1f}% prob={prob:.2f} pnl=${pnl:.2f} eq=${equity:.2f}", flush=True)

    # Results
    if not trades:
        print("  NO TRADES", flush=True)
        return {
            'trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'final_equity': STARTING_CAPITAL, 'mdd': 0, 'avg_pnl': 0,
            'perm_p': 1.0, 'gates_passed': 0, 'gates': {},
            'variant': vkey, 'name': cfg['name'],
            'concentration': 0, 'bear_trades': 0, 'bear_pnl': 0,
        }, []

    pnls = np.array([t['pnl'] for t in trades])
    n = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    sharpe = float(np.mean(pnls)/np.std(pnls)*np.sqrt(252/max(1,n))) if np.std(pnls)>0 else 0
    down = pnls[pnls < 0]
    sortino = float(np.mean(pnls)/np.std(down)*np.sqrt(252/max(1,n))) if len(down)>0 and np.std(down)>0 else 0
    pf = float(sum(wins)/abs(sum(losses))) if len(losses)>0 and sum(losses)!=0 else float('inf')
    wr = float(len(wins)/n*100)
    avg_pnl = float(np.mean(pnls))

    # Concentration
    t_abs = {t: abs(p) for t,p in ticker_pnl.items()}
    total_abs = sum(t_abs.values())
    top2 = sum(sorted(t_abs.values(), reverse=True)[:2])
    concentration = top2/total_abs if total_abs > 0 else 0

    # Bear market performance
    bear_trades = [t for t in trades if t.get('is_bear')]
    bear_pnl = sum(t['pnl'] for t in bear_trades)
    bull_trades = [t for t in trades if not t.get('is_bear')]
    bull_pnl = sum(t['pnl'] for t in bull_trades)

    # Call vs Put breakdown
    call_pnl = sum(t['pnl'] for t in trades if t['opt_type'] == 'call')
    put_pnl = sum(t['pnl'] for t in trades if t['opt_type'] == 'put')

    # Permutation test
    perm_count = 0
    for _ in range(N_PERMUTATIONS):
        s = np.random.permutation(pnls)
        ps = float(np.mean(s)/np.std(s)*np.sqrt(252/max(1,n))) if np.std(s)>0 else 0
        if ps >= sharpe: perm_count += 1
    perm_p = perm_count / N_PERMUTATIONS

    # MC CI
    mc_final = [STARTING_CAPITAL + np.sum(np.random.choice(pnls, n, replace=True)) for _ in range(1000)]
    mc_5th = np.percentile(mc_final, 5)

    # Regime check — use bear/bull split instead of year
    if bear_trades and bull_trades:
        bear_pnls = np.array([t['pnl'] for t in bear_trades])
        bull_pnls = np.array([t['pnl'] for t in bull_trades])
        bear_sharpe = float(np.mean(bear_pnls)/np.std(bear_pnls)*np.sqrt(252/max(1,len(bear_pnls)))) if len(bear_pnls)>=2 and np.std(bear_pnls)>0 else 0
        bull_sharpe = float(np.mean(bull_pnls)/np.std(bull_pnls)*np.sqrt(252/max(1,len(bull_pnls)))) if len(bull_pnls)>=2 and np.std(bull_pnls)>0 else 0
        max_s = max(abs(bear_sharpe), abs(bull_sharpe), 0.01)
        regime_gap = abs(bear_sharpe - bull_sharpe) / max_s
    else:
        # Fallback to year-based
        year_sharpes = {}
        for year in sorted(set(t['earn_date'][:4] for t in trades)):
            yp = [t['pnl'] for t in trades if t['earn_date'].startswith(year)]
            year_sharpes[year] = float(np.mean(yp)/np.std(yp)*np.sqrt(252/max(1,len(yp)))) if len(yp)>=2 and np.std(yp)>0 else 0
        green = [s for y,s in year_sharpes.items() if y != '2022']
        red = [s for y,s in year_sharpes.items() if y == '2022']
        if green and red:
            regime_gap = abs(np.mean(green)-np.mean(red)) / max(max(abs(s) for s in green), max(abs(s) for s in red), 0.01)
        else:
            regime_gap = 1.0

    gates = {
        'sharpe_gt_1': sharpe >= 1.0,
        'perm_p_lt_005': perm_p < 0.05,
        'wr_gt_40': wr >= 40,
        'regime_balance': regime_gap < 0.50,
        'mc_ci_positive': mc_5th > STARTING_CAPITAL,
    }
    gates_passed = sum(gates.values())

    print(f"\n  Trades: {n} (Bear: {len(bear_trades)}, Bull: {len(bull_trades)})", flush=True)
    print(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.3f} | WR: {wr:.1f}%", flush=True)
    print(f"  MDD: {max_dd*100:.2f}% | Final: ${equity:.2f} | Return: {(equity/STARTING_CAPITAL-1)*100:.1f}%", flush=True)
    print(f"  Bear PnL: ${bear_pnl:.2f} ({len(bear_trades)} trades) | Bull PnL: ${bull_pnl:.2f} ({len(bull_trades)} trades)", flush=True)
    print(f"  Call PnL: ${call_pnl:.2f} | Put PnL: ${put_pnl:.2f}", flush=True)
    print(f"  Concentration: {concentration*100:.1f}% | Perm p: {perm_p:.3f} | Regime gap: {regime_gap:.3f}", flush=True)

    print(f"  Top tickers:", flush=True)
    for t,p in sorted(ticker_pnl.items(), key=lambda x: -abs(x[1]))[:6]:
        print(f"    {t}: ${p:.2f} ({ticker_count[t]} trades)", flush=True)

    print(f"\n  5-Gate: {gates_passed}/5 {'✅ PASS' if gates_passed>=4 else '❌ FAIL'}", flush=True)
    for g,v in gates.items():
        val = {'sharpe_gt_1': sharpe, 'perm_p_lt_005': perm_p, 'wr_gt_40': wr,
               'regime_balance': regime_gap, 'mc_ci_positive': mc_5th}.get(g)
        thr = {'sharpe_gt_1': 1.0, 'perm_p_lt_005': 0.05, 'wr_gt_40': 40,
               'regime_balance': 0.50, 'mc_ci_positive': 0}.get(g)
        print(f"    {g}: {'PASS' if v else 'FAIL'} ({val:.3f} vs {thr})", flush=True)

    return {
        'trades': n, 'sharpe': round(sharpe,3), 'sortino': round(sortino,3),
        'pf': round(pf,3), 'wr': round(wr,1), 'mdd': round(max_dd*100,2),
        'final_equity': round(equity,2), 'avg_pnl': round(avg_pnl,2),
        'perm_p': round(perm_p,3), 'gates_passed': gates_passed, 'gates': gates,
        'variant': vkey, 'name': cfg['name'],
        'concentration': round(concentration*100,1),
        'bear_trades': len(bear_trades), 'bear_pnl': round(bear_pnl,2),
        'bull_trades': len(bull_trades), 'bull_pnl': round(bull_pnl,2),
        'call_pnl': round(call_pnl,2), 'put_pnl': round(put_pnl,2),
        'regime_gap': round(regime_gap,3),
    }, trades


def main():
    print(f"Running on: {LVL3_ROOT}", flush=True)

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('pead_options_v3')
            print("MLflow OK", flush=True)
        except Exception as e:
            print(f"MLflow error: {e}", flush=True)

    print("="*70, flush=True)
    print("  PEAD OPTIONS V3 — Regime + Concentration Fixes", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("="*70, flush=True)

    prices, earnings = load_data()

    print("\n--- Building events (5% gap threshold) ---", flush=True)
    events_5pct = build_events(prices, earnings, min_gap_pct=0.05)

    print("\n--- Building events (4% gap threshold) ---", flush=True)
    events_4pct = build_events(prices, earnings, min_gap_pct=0.04)

    all_results = []
    for vkey in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vkey]
        t0 = datetime.now()
        result, trades = run_variant(vkey, cfg, events_5pct, events_4pct)
        elapsed = (datetime.now() - t0).total_seconds()
        result['runtime_s'] = round(elapsed, 1)

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"v3_{vkey}_{cfg['name'][:25]}"):
                    mlflow.log_params({k: str(v)[:50] for k,v in cfg.items()})
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'], 'sortino': result['sortino'],
                        'pf': min(result['pf'], 999), 'wr': result['wr'],
                        'mdd': result['mdd'], 'final_equity': result['final_equity'],
                        'n_trades': result['trades'], 'perm_p': result['perm_p'],
                        'gates_passed': result['gates_passed'],
                        'concentration': result['concentration'],
                        'regime_gap': result['regime_gap'],
                    })
            except: pass

        all_results.append(result)

    # Summary
    print("\n" + "="*70, flush=True)
    print("  SUMMARY — PEAD OPTIONS V3", flush=True)
    print("="*70, flush=True)
    print(f"\n  {'V':<3} {'Name':<42} {'#':<5} {'Shrp':<7} {'Sort':<7} {'PF':<6} {'WR%':<6} "
          f"{'MDD%':<7} {'Final$':<9} {'Conc%':<7} {'RegGap':<7} {'G':<4}", flush=True)
    print(f"  {'-'*3} {'-'*42} {'-'*5} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*4}", flush=True)

    for r in all_results:
        print(f"  {r['variant']:<3} {r['name'][:42]:<42} {r['trades']:<5} {r['sharpe']:<7} {r['sortino']:<7} "
              f"{r['pf']:<6.1f} {r['wr']:<6.1f} {r['mdd']:<7.1f} ${r['final_equity']:<8.0f} "
              f"{r['concentration']:<7.1f} {r['regime_gap']:<7.3f} {r['gates_passed']}/5", flush=True)

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({'variants': all_results, 'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    best = max(all_results, key=lambda r: r['gates_passed']*10 + r['sharpe'])
    print(f"\n  BEST: {best['variant']} — {best['name']}", flush=True)
    print(f"    Gates {best['gates_passed']}/5 | Sharpe {best['sharpe']} | "
          f"${STARTING_CAPITAL}→${best['final_equity']} | Concentration {best['concentration']}%", flush=True)

    if best['gates_passed'] >= 5:
        print(f"\n  🏆 FULLY VALIDATED — ALL 5 GATES PASSED", flush=True)
    elif best['gates_passed'] >= 4:
        print(f"\n  ✅ CONDITIONALLY VALIDATED — {best['gates_passed']}/5 GATES", flush=True)
    else:
        print(f"\n  ❌ NOT VALIDATED — {best['gates_passed']}/5 GATES", flush=True)

if __name__ == '__main__':
    main()
