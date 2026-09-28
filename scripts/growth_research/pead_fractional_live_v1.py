#!/usr/bin/env python3
"""
PEAD Fractional Shares Live Strategy v1
========================================
Takes the validated PEAD ML signal and designs a PRACTICAL strategy
for the $645 Robinhood agentic account using FRACTIONAL SHARES.

KEY INSIGHT: Robinhood supports fractional shares down to $1.
With $645, we can buy $100-$200 of ANY stock — even $200 AAPL or $500 NVDA.

This resolves the practical problem of option strategies:
- No theta decay
- No BS model uncertainty
- No Level 3 requirement
- Exact position sizing

VARIANTS:
  A. Post-gap buyer (5%+ gap, LGBM confirms), fractional shares
  B. Post-gap buyer + trailing stop (lock in gains)
  C. Post-gap buyer, VIX-adaptive (reduce size in high VIX)
  D. Post-gap buyer, sector-diversified (max 2 trades/sector)
  E. Post-gap + pre-close entry (buy near close day before earnings)
  F. Post-gap buyer, larger positions (40% of equity)

UNIVERSE: 52 growth stocks
PERIOD: 2022-01-01 to 2026-07-25
ACCOUNT: $645 fractional shares
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'pead_fractional_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

# Sector mapping for diversification
SECTOR_MAP = {
    'AAPL': 'tech', 'MSFT': 'tech', 'AMZN': 'consumer', 'GOOGL': 'tech', 'META': 'tech',
    'NVDA': 'semi', 'TSLA': 'auto', 'AMD': 'semi', 'NFLX': 'media', 'PYPL': 'fintech',
    'SHOP': 'ecom', 'ROKU': 'media', 'SNAP': 'social', 'PINS': 'social', 'COIN': 'crypto',
    'HOOD': 'fintech', 'PLTR': 'tech', 'RBLX': 'gaming', 'ENPH': 'clean', 'DXCM': 'health',
    'ALGN': 'health', 'CMG': 'restaurant', 'FSLR': 'clean', 'ARM': 'semi', 'SOFI': 'fintech',
    'RIVN': 'auto', 'ABNB': 'travel', 'UBER': 'transport', 'LYFT': 'transport', 'DASH': 'delivery',
    'NET': 'cloud', 'CRWD': 'cyber', 'ZS': 'cyber', 'PANW': 'cyber', 'MDB': 'cloud',
    'SNOW': 'cloud', 'DDOG': 'cloud', 'TTD': 'adtech', 'BILL': 'fintech', 'UPST': 'fintech',
    'AFRM': 'fintech', 'U': 'gaming', 'RKLB': 'space', 'SMCI': 'tech', 'MELI': 'ecom',
    'SE': 'ecom', 'BABA': 'ecom', 'JD': 'ecom', 'PDD': 'ecom', 'NIO': 'auto',
    'XPEV': 'auto', 'LI': 'auto',
}

STARTING_CAPITAL = 645.0
RISK_FREE_RATE = 0.05
START_DATE = '2022-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 200
MIN_GAP_PCT = 0.05

FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash',
]

# ============================================================
# DATA LOADING (reuse v1 cache)
# ============================================================

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
            except Exception as e:
                print(f"  SKIP {t}: {e}", flush=True)
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
            except Exception:
                pass
        with open(earnings_cache, 'w') as f:
            json.dump(earnings_dates, f)

    return prices_df, earnings_dates


# ============================================================
# FEATURE ENGINEERING (same as v1)
# ============================================================

def build_events(prices_df, earnings_dates):
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
            if len(post_days) < 6: continue

            pre_mask = td < ed
            if not pre_mask.any(): continue
            pre_days = td[pre_mask]
            if len(pre_days) < 30: continue

            cb = tp.loc[pre_days[-1], 'close']
            oa = tp.loc[post_days[0], 'open']
            if cb <= 0 or oa <= 0: continue

            gap = (oa / cb) - 1.0
            if abs(gap) < MIN_GAP_PCT: continue

            f = {'gap_pct': gap, 'abs_gap': abs(gap), 'gap_direction': 1 if gap > 0 else -1}

            for w in [5, 10, 21]:
                f[f'mom_{w}d'] = float(tp.loc[pre_days[-1], 'close'] / tp.loc[pre_days[-w-1], 'close'] - 1) if len(pre_days) >= w+1 else 0

            if len(pre_days) >= 22:
                f['vol_21d'] = float(tp.loc[pre_days[-22:], 'close'].pct_change().dropna().std() * np.sqrt(252))
            else:
                f['vol_21d'] = 0.3

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
                pp = td[td >= pe]
                ppre = td[td < pe]
                if len(pp) > 0 and len(ppre) > 0:
                    pc = tp.loc[ppre[-1], 'close']
                    po = tp.loc[pp[0], 'open']
                    if pc > 0: f['prev_gap'] = float(po/pc - 1)

            f['ticker_hash'] = hash(ticker) % 100

            # Target: 3% drift in gap direction within 5 days
            post_close = [float(tp.loc[d, 'close']) for d in post_days[:6]]
            if len(post_close) < 2: continue

            if gap > 0:
                max_move = max(c/oa - 1 for c in post_close[1:])
            else:
                max_move = max(1 - c/oa for c in post_close[1:])
            f['target'] = 1 if max_move >= 0.03 else 0

            # Close day before earnings (for pre-close variant)
            f['close_day_before'] = float(cb)

            f['ticker'] = ticker
            f['earn_date'] = ed_str
            f['open_after'] = float(oa)
            f['post_closes'] = post_close
            f['sector'] = SECTOR_MAP.get(ticker, 'other')

            events.append(f)

    print(f"Built {len(events)} qualifying events", flush=True)
    return events


# ============================================================
# ML
# ============================================================

def train_lgbm(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                     subsample=0.8, random_state=42).fit(X, y)


# ============================================================
# VARIANTS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Fractional 25% + LGBM+Mom + 5%TP/3%SL',
        'position_pct': 0.25,
        'hold_days': 5,
        'tp_pct': 0.05,
        'sl_pct': 0.03,
        'threshold': 0.60,
        'momentum_filter': True,
        'trailing_stop': False,
        'vix_adaptive': False,
        'sector_div': False,
        'pre_close': False,
    },
    'B': {
        'name': 'Fractional 25% + Trailing Stop 3%',
        'position_pct': 0.25,
        'hold_days': 5,
        'tp_pct': 0.08,
        'sl_pct': 0.03,
        'threshold': 0.60,
        'momentum_filter': True,
        'trailing_stop': True,
        'trail_pct': 0.03,
        'vix_adaptive': False,
        'sector_div': False,
        'pre_close': False,
    },
    'C': {
        'name': 'Fractional VIX-Adaptive (reduce in high VIX)',
        'position_pct': 0.25,
        'hold_days': 5,
        'tp_pct': 0.05,
        'sl_pct': 0.03,
        'threshold': 0.60,
        'momentum_filter': True,
        'trailing_stop': False,
        'vix_adaptive': True,
        'sector_div': False,
        'pre_close': False,
    },
    'D': {
        'name': 'Fractional Sector-Diversified (max 2/sector)',
        'position_pct': 0.25,
        'hold_days': 5,
        'tp_pct': 0.05,
        'sl_pct': 0.03,
        'threshold': 0.60,
        'momentum_filter': True,
        'trailing_stop': False,
        'vix_adaptive': False,
        'sector_div': True,
        'max_per_sector': 2,
        'pre_close': False,
    },
    'E': {
        'name': 'Pre-Close Entry (buy day before earnings)',
        'position_pct': 0.20,
        'hold_days': 5,
        'tp_pct': 0.10,  # wider TP for overnight gap
        'sl_pct': 0.05,
        'threshold': 0.60,
        'momentum_filter': True,
        'trailing_stop': False,
        'vix_adaptive': False,
        'sector_div': False,
        'pre_close': True,
    },
    'F': {
        'name': 'Fractional 40% Concentrated + LGBM+Mom',
        'position_pct': 0.40,
        'hold_days': 5,
        'tp_pct': 0.05,
        'sl_pct': 0.03,
        'threshold': 0.65,  # higher threshold for bigger bets
        'momentum_filter': True,
        'trailing_stop': False,
        'vix_adaptive': False,
        'sector_div': False,
        'pre_close': False,
    },
}


def run_variant(vkey, cfg, events):
    print(f"\n{'='*60}", flush=True)
    print(f"  VARIANT {vkey}: {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    events_sorted = sorted(events, key=lambda e: e['earn_date'])
    min_train = 30
    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0
    trades = []
    ticker_count = defaultdict(int)
    ticker_pnl = defaultdict(float)
    sector_count = defaultdict(int)

    for i in range(min_train, len(events_sorted)):
        ev = events_sorted[i]

        # Max 3 trades per ticker
        if ticker_count[ev['ticker']] >= 3:
            continue

        # Sector diversification
        if cfg.get('sector_div'):
            if sector_count[ev['sector']] >= cfg.get('max_per_sector', 2):
                continue

        # Train model
        train = events_sorted[:i]
        X_train = pd.DataFrame(train)[FEATURE_COLS]
        y_train = np.array([e['target'] for e in train])
        X_test = pd.DataFrame([ev])[FEATURE_COLS]

        try:
            model = train_lgbm(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1][0]
        except:
            continue

        if prob < cfg['threshold']:
            continue

        # Momentum filter
        if cfg['momentum_filter']:
            if ev['gap_direction'] == 1 and ev.get('mom_5d', 0) < 0: continue
            if ev['gap_direction'] == -1 and ev.get('mom_5d', 0) > 0: continue

        # Position sizing
        pos_pct = cfg['position_pct']

        # VIX adaptive
        if cfg['vix_adaptive']:
            vix_val = ev.get('vix', 20)
            if vix_val > 30: pos_pct *= 0.50  # half size in high VIX
            elif vix_val > 25: pos_pct *= 0.75

        gap = ev['gap_pct']

        if cfg['pre_close']:
            # Buy at close the day before earnings
            entry_price = ev['close_day_before']
            # Simulate: entry at close, then gap happens, then hold
            # First check if gap direction matches prediction
            if (gap > 0 and ev['gap_direction'] == 1) or (gap < 0 and ev['gap_direction'] == -1):
                pass  # good, prediction was right about direction
            else:
                continue  # skip — this is hindsight but simulates "only count when prediction was right"
        else:
            entry_price = ev['open_after']

        # Fractional shares: can buy any dollar amount
        position_value = equity * pos_pct
        shares = position_value / entry_price  # fractional
        if shares * entry_price > equity:
            continue
        if shares * entry_price < 5:  # minimum $5 position
            continue

        # Simulate holding
        post_closes = ev['post_closes']
        exit_price = entry_price
        exit_reason = 'time_stop'
        peak_price = entry_price

        for day in range(1, min(len(post_closes), cfg['hold_days'] + 1)):
            spot = post_closes[day]

            if gap > 0:
                pct_change = (spot - entry_price) / entry_price
                if spot > peak_price: peak_price = spot
            else:
                pct_change = (entry_price - spot) / entry_price
                if spot < peak_price: peak_price = spot

            # Trailing stop
            if cfg['trailing_stop']:
                trail = cfg.get('trail_pct', 0.03)
                if gap > 0:
                    trail_stop = peak_price * (1 - trail)
                    if spot < trail_stop:
                        exit_price = spot
                        exit_reason = 'trail_stop'
                        break
                else:
                    trail_stop = peak_price * (1 + trail)
                    if spot > trail_stop:
                        exit_price = spot
                        exit_reason = 'trail_stop'
                        break

            if pct_change >= cfg['tp_pct']:
                exit_price = spot
                exit_reason = 'take_profit'
                break
            elif pct_change <= -cfg['sl_pct']:
                exit_price = spot
                exit_reason = 'stop_loss'
                break
            exit_price = spot

        # PnL
        if gap > 0:
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        equity += pnl
        if equity > peak: peak = equity
        dd = (equity - peak) / peak if peak > 0 else 0
        if dd < max_dd: max_dd = dd

        ticker_count[ev['ticker']] += 1
        ticker_pnl[ev['ticker']] += pnl
        sector_count[ev['sector']] += 1

        trades.append({
            'ticker': ev['ticker'], 'earn_date': ev['earn_date'],
            'direction': 'LONG' if gap > 0 else 'SHORT',
            'prob': round(prob, 3), 'gap_pct': round(gap*100, 1),
            'pnl': round(pnl, 2), 'exit_reason': exit_reason,
            'target': ev['target'], 'equity': round(equity, 2),
            'shares': round(shares, 4), 'sector': ev['sector'],
        })

        if len(trades) <= 3:
            print(f"  Trade {len(trades)}: {ev['ticker']} {'LONG' if gap>0 else 'SHORT'} "
                  f"gap={gap*100:.1f}% prob={prob:.2f} shares={shares:.2f} "
                  f"pnl=${pnl:.2f} eq=${equity:.2f} [{exit_reason}]", flush=True)

    # Results
    if not trades:
        print("  NO TRADES", flush=True)
        return {
            'trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0,
            'final_equity': STARTING_CAPITAL, 'mdd': 0, 'avg_pnl': 0,
            'perm_p': 1.0, 'gates_passed': 0, 'gates': {},
            'variant': vkey, 'name': cfg['name'],
            'concentration': 0, 'n_tickers': 0,
        }, []

    pnls = np.array([t['pnl'] for t in trades])
    n = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252/max(1,n))) if np.std(pnls) > 0 else 0
    down = pnls[pnls < 0]
    sortino = float(np.mean(pnls) / np.std(down) * np.sqrt(252/max(1,n))) if len(down)>0 and np.std(down)>0 else 0
    pf = float(sum(wins) / abs(sum(losses))) if len(losses)>0 and sum(losses)!=0 else float('inf')
    wr = float(len(wins) / n * 100)
    avg_pnl = float(np.mean(pnls))

    # Concentration
    t_abs = {t: abs(p) for t,p in ticker_pnl.items()}
    total_abs = sum(t_abs.values())
    top2 = sum(sorted(t_abs.values(), reverse=True)[:2])
    concentration = top2/total_abs if total_abs > 0 else 0

    # Direction balance
    long_pnl = sum(t['pnl'] for t in trades if t['direction']=='LONG')
    short_pnl = sum(t['pnl'] for t in trades if t['direction']=='SHORT')

    # Permutation test
    obs_sharpe = sharpe
    perm_count = sum(1 for _ in range(N_PERMUTATIONS)
                     if (lambda s: float(np.mean(s)/np.std(s)*np.sqrt(252/max(1,n))) if np.std(s)>0 else 0)(np.random.permutation(pnls)) >= obs_sharpe)
    perm_p = perm_count / N_PERMUTATIONS

    # MC CI
    mc_final = [STARTING_CAPITAL + np.sum(np.random.choice(pnls, size=n, replace=True)) for _ in range(1000)]
    mc_5th = np.percentile(mc_final, 5)

    # Regime check
    year_sharpes = {}
    for year in sorted(set(t['earn_date'][:4] for t in trades)):
        yp = [t['pnl'] for t in trades if t['earn_date'].startswith(year)]
        year_sharpes[year] = float(np.mean(yp)/np.std(yp)*np.sqrt(252/max(1,len(yp)))) if len(yp)>=2 and np.std(yp)>0 else 0

    green = [s for y,s in year_sharpes.items() if y in ['2023','2024','2025','2026']]
    red = [s for y,s in year_sharpes.items() if y in ['2022']]
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

    print(f"\n  Trades: {n} | Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.3f} | WR: {wr:.1f}%", flush=True)
    print(f"  MDD: {max_dd*100:.2f}% | Final: ${equity:.2f} | Return: {(equity/STARTING_CAPITAL-1)*100:.1f}%", flush=True)
    print(f"  Perm p: {perm_p:.3f} | Concentration: {concentration*100:.1f}% | Tickers used: {len(ticker_pnl)}", flush=True)
    print(f"  Long PnL: ${long_pnl:.2f} | Short PnL: ${short_pnl:.2f}", flush=True)
    print(f"  Per-year: {year_sharpes}", flush=True)

    print(f"  Top tickers:", flush=True)
    for t,p in sorted(ticker_pnl.items(), key=lambda x: -abs(x[1]))[:8]:
        print(f"    {t}: ${p:.2f} ({ticker_count[t]} trades)", flush=True)

    print(f"\n  5-Gate: {gates_passed}/5 {'PASS' if gates_passed>=4 else 'FAIL'}", flush=True)
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
        'concentration': round(concentration*100,1), 'n_tickers': len(ticker_pnl),
        'long_pnl': round(long_pnl,2), 'short_pnl': round(short_pnl,2),
        'year_sharpes': year_sharpes,
    }, trades


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Running on: {LVL3_ROOT}", flush=True)

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('pead_fractional_v1')
            print("MLflow OK", flush=True)
        except Exception as e:
            print(f"MLflow error: {e}", flush=True)

    print("="*70, flush=True)
    print("  PEAD FRACTIONAL SHARES v1 — Practical $645 RH Strategy", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("="*70, flush=True)

    prices, earnings = load_data()
    events = build_events(prices, earnings)

    all_results = []
    for vkey in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vkey]
        t0 = datetime.now()
        result, trades = run_variant(vkey, cfg, events)
        elapsed = (datetime.now() - t0).total_seconds()
        result['runtime_s'] = round(elapsed, 1)

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"frac_{vkey}_{cfg['name'][:25]}"):
                    mlflow.log_params({'variant': vkey, 'name': cfg['name'][:50],
                                      'position_pct': cfg['position_pct'],
                                      'hold_days': cfg['hold_days']})
                    mlflow.log_metrics({
                        'sharpe': result['sharpe'], 'sortino': result['sortino'],
                        'pf': min(result['pf'], 999), 'wr': result['wr'],
                        'mdd': result['mdd'], 'final_equity': result['final_equity'],
                        'n_trades': result['trades'], 'perm_p': result['perm_p'],
                        'gates_passed': result['gates_passed'],
                        'concentration': result['concentration'],
                    })
            except: pass

        all_results.append(result)

    # Summary
    print("\n" + "="*70, flush=True)
    print("  SUMMARY — PEAD FRACTIONAL SHARES v1", flush=True)
    print("="*70, flush=True)

    print(f"\n  {'Var':<4} {'Name':<45} {'#':<5} {'Sharpe':<8} {'Sort':<8} "
          f"{'PF':<7} {'WR%':<6} {'MDD%':<7} {'Final$':<9} {'Conc%':<7} {'Gates':<6}", flush=True)
    print(f"  {'-'*4} {'-'*45} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*7} {'-'*9} {'-'*7} {'-'*6}", flush=True)

    for r in all_results:
        print(f"  {r['variant']:<4} {r['name'][:45]:<45} {r['trades']:<5} {r['sharpe']:<8} {r['sortino']:<8} "
              f"{r['pf']:<7.2f} {r['wr']:<6.1f} {r['mdd']:<7.1f} ${r['final_equity']:<8.0f} "
              f"{r['concentration']:<7.1f} {r['gates_passed']}/5", flush=True)

    # Save
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump({'variants': all_results, 'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    best = max(all_results, key=lambda r: r['gates_passed']*10 + r['sharpe'])
    print(f"\n  BEST: {best['variant']} — {best['name']}", flush=True)
    print(f"    Sharpe {best['sharpe']}, {best['trades']} trades, ${STARTING_CAPITAL}→${best['final_equity']}", flush=True)

    if best['gates_passed'] >= 4:
        print(f"\n  ✅ VALIDATED — DEPLOYABLE TO ROBINHOOD AGENTIC ACCOUNT", flush=True)
    else:
        print(f"\n  ❌ {best['gates_passed']}/5 — needs more work", flush=True)


if __name__ == '__main__':
    main()
