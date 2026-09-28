#!/usr/bin/env python3
"""
Sector Combined — Adversarial Backtest Suite (Optimized)
========================================================

Pre-computes LGBM rankings ONCE for all rebalance dates, then reuses
them across all tests. This makes random timing feasible (shuffles
trade PnLs, not re-training LGBM).

Tests:
  1. Re-implementation: Independent walk-forward backtest
  2. Inverse signal: Trade opposite direction (flip ranking)
  3. Random timing: 500 PnL permutations, compute p-value
  4. Sub-period stability: Split trades into 4 time blocks
  5. Top-N removal: Remove best 3 sectors from universe
  6. Parameter sensitivity: Vary lookback, rebalance freq, profit target

Run: python3 -u paper_engines/sector_combined_adversarial.py
"""

import json
import sys
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    bs_call_price,
    bs_put_price,
    compute_atr,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not available — using momentum ranking")

# ===================== CONFIGS =====================

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']

VARIANT_CONFIGS = {
    'v7':  {'dte': 21, 'otm_pct': 2.0, 'rebal_days': 5,  'profit_target': None, 'top_k': 3, 'bottom_k': 3, 'high_vix_top_k': 2},
    'v8':  {'dte': 14, 'otm_pct': 2.0, 'rebal_days': 5,  'profit_target': None, 'top_k': 3, 'bottom_k': 3, 'high_vix_top_k': 2},
    'v9':  {'dte': 14, 'otm_pct': 2.0, 'rebal_days': 5,  'profit_target': None, 'top_k': 3, 'bottom_k': 3, 'high_vix_top_k': 2},
    'v91': {'dte': 28, 'otm_pct': 2.0, 'rebal_days': 10, 'profit_target': None, 'top_k': 3, 'bottom_k': 3, 'high_vix_top_k': 2},
    'v92': {'dte': 28, 'otm_pct': 3.0, 'rebal_days': 20, 'profit_target': None, 'top_k': 3, 'bottom_k': 3, 'high_vix_top_k': 2},
    'v93': {'dte': 28, 'otm_pct': 3.0, 'rebal_days': 20, 'profit_target': 0.50, 'top_k': 3, 'bottom_k': 3, 'high_vix_top_k': 2},
    'v10': {'dte': 28, 'otm_pct': 4.0, 'rebal_days': 20, 'profit_target': 0.30, 'top_k': 4, 'bottom_k': 4, 'high_vix_top_k': 4},
}

HAIRCUT = 0.15
SPREAD_COMM = 2.60
INITIAL_CAPITAL = 645.0
MAX_POS_SIZE = 200
MAX_POS_PCT = 0.40
VIX_THRESHOLD = 20.0
MIN_SPREAD_WIDTH = 3.0
SPREAD_PCT = 3.0

FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]


# ===================== DATA =====================

def download_data(start='2019-01-01'):
    import yfinance as yf
    all_tickers = SECTORS + ['SPY', '^VIX', 'TLT', 'GLD']
    print(f"Downloading data from {start}...", flush=True)
    raw = yf.download(all_tickers, start=start, progress=False)
    if raw is None or raw.empty:
        raise ValueError("yfinance returned empty data")

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill().rename(columns={'^VIX': 'VIX'})
    high = high.ffill().rename(columns={'^VIX': 'VIX'})
    low = low.ffill().rename(columns={'^VIX': 'VIX'})

    vix = close['VIX'].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    print(f"Data: {len(ix)} trading days ({ix[0].date()} to {ix[-1].date()})", flush=True)
    return close.loc[ix], sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]


# ===================== FEATURES =====================

def compute_features(px):
    from scipy import stats
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),
                   (63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std()+1e-10) * np.sqrt(252)) if len(r63)>10 else 0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:]>0).mean()) if len(monthly)>=12 else 0.5
    dr = r63[r63<0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr)>3 else 0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px)>=252 else 0
    f['calmar_1y'] = cagr/(abs(mdd)+1e-10)
    if len(px)>=63:
        y = np.log(px.iloc[-63:].values+1e-10)
        x = np.arange(len(y))
        slope,_,r_val,_,_ = stats.linregress(x,y)
        f['trend_r2_63d'] = r_val**2
        f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0; f['trend_slope_63d'] = 0
    return f


# ===================== PRE-COMPUTE RANKINGS =====================

def precompute_rankings(sc, rebal_interval=5, sectors=None):
    """Pre-compute LGBM rankings at every possible rebalance date.
    Returns dict: {date_idx: {ticker: score}}.
    """
    if sectors is None:
        sectors = list(sc.columns)
    sc_use = sc[sectors]

    warmup = 260
    all_indices = list(range(warmup, len(sc_use)))
    # Compute at every rebal_interval
    rebal_indices = all_indices[::rebal_interval]

    rankings = {}
    print(f"  Pre-computing rankings at {len(rebal_indices)} dates...", flush=True)

    for count, idx in enumerate(rebal_indices):
        if (count+1) % 50 == 0:
            print(f"    {count+1}/{len(rebal_indices)} dates done", flush=True)

        if HAS_LGBM:
            # Build training data from trailing 400 days
            start = max(0, idx - 400)
            train_dates = sc_use.index[start:idx]
            sample_dates = train_dates[::20]

            records = []
            for dt in sample_dates:
                dt_idx = sc_use.index.get_loc(dt)
                if dt_idx < 260:
                    continue
                for tk in sectors:
                    px = sc_use[tk].iloc[:dt_idx+1].dropna()
                    feats = compute_features(px)
                    if not feats:
                        continue
                    fi = min(dt_idx + 28, len(sc_use)-1)
                    feats['fwd_ret'] = float(sc_use[tk].iloc[fi]/sc_use[tk].iloc[dt_idx]-1)
                    records.append(feats)

            if len(records) >= 50:
                df = pd.DataFrame(records)
                for c in FEAT_COLS:
                    if c not in df.columns:
                        df[c] = 0.0
                df[FEAT_COLS] = df[FEAT_COLS].fillna(0)
                df['rank_label'] = df.groupby(df.index // len(sectors))['fwd_ret'].rank(pct=True)

                X = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
                y = df['rank_label'].values.astype(np.float32)
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                                      verbose=-1)
                m.fit(X, y)

                curr = {}
                for tk in sectors:
                    px = sc_use[tk].iloc[:idx+1].dropna()
                    feats = compute_features(px)
                    if feats:
                        curr[tk] = feats

                if curr:
                    pred_df = pd.DataFrame(curr).T
                    for c in FEAT_COLS:
                        if c not in pred_df.columns:
                            pred_df[c] = 0.0
                    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
                    scores = m.predict(X_pred)
                    rankings[idx] = dict(zip(pred_df.index, scores))
                    continue

        # Fallback: momentum
        rets = {}
        for tk in sectors:
            px = sc_use[tk].iloc[:idx+1].dropna()
            if len(px) > 21:
                rets[tk] = float(px.iloc[-1]/px.iloc[-21]-1)
        rankings[idx] = rets

    print(f"  Rankings computed for {len(rankings)} dates", flush=True)
    return rankings


# ===================== SPREAD PRICING =====================

def price_spread(S, mode, otm_pct, dte, sh_tk, sl_tk, sc_tk, vix_val):
    K_factor = 1 + otm_pct/100 if mode == 'bull' else 1 - otm_pct/100
    K1 = round(S * K_factor)
    width = max(MIN_SPREAD_WIDTH, K1 * SPREAD_PCT / 100)

    if mode == 'bull':
        K2 = round(K1 + width)
    else:
        K2 = round(K1 - width)
        if K2 >= K1:
            K2 = K1 - 1

    if sh_tk is not None and len(sh_tk) >= 14:
        atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
    else:
        atr = S * 0.015

    if mode == 'bull':
        entry_cost_ps, max_profit_ps = price_bull_call_spread(
            S=S, K1=K1, K2=K2, dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT)
    else:
        entry_cost_ps, max_profit_ps = price_bear_put_spread(
            S=S, K1=min(K1,K2), K2=max(K1,K2), dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT)

    cost = entry_cost_ps * 100 + SPREAD_COMM
    max_profit = max_profit_ps * 100 - SPREAD_COMM
    return cost, max_profit, K1, K2, entry_cost_ps


# ===================== CORE BACKTEST =====================

def run_backtest(sc, sh, sl, vix, spy, cfg, rankings_cache, sectors=None,
                 invert_signal=False):
    """Walk-forward backtest using pre-computed rankings."""
    if sectors is None:
        sectors = [s for s in SECTORS if s in sc.columns]
    sc_use = sc[sectors]
    sh_use = sh[[s for s in sectors if s in sh.columns]]
    sl_use = sl[[s for s in sectors if s in sl.columns]]

    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    open_positions = []

    warmup = 260
    all_dates = sc_use.index[warmup:]
    if len(all_dates) == 0:
        return _empty_result()

    rebal_interval = cfg['rebal_days']
    dte = cfg['dte']
    otm_pct = cfg['otm_pct']
    profit_target = cfg.get('profit_target')
    top_k = cfg['top_k']
    bottom_k = cfg['bottom_k']
    hvix_k = cfg.get('high_vix_top_k', 2)

    for day_offset, today in enumerate(all_dates):
        abs_idx = sc.index.get_loc(today)
        cur_vix = float(vix.loc[today]) if today in vix.index else 20.0

        # --- Exit logic ---
        to_close = []
        for i, pos in enumerate(open_positions):
            entry_dt = pd.Timestamp(pos['entry_date'])
            days_held = len(sc_use.index[(sc_use.index > entry_dt) & (sc_use.index <= today)])
            if pos['ticker'] not in sc_use.columns:
                continue
            cur_price = float(sc_use[pos['ticker']].loc[today])

            should_exit = False
            pnl = 0; reason = 'hold'

            # Profit target
            if profit_target is not None and days_held >= 1:
                remaining = max(dte - days_held, 0)
                T = remaining / 365.0
                sigma = max(cur_vix / 100.0, 0.10)
                K1, K2 = pos['K1'], pos['K2']
                if pos['mode'] == 'bull':
                    val_ps = bs_call_price(cur_price, K1, T, sigma=sigma) - bs_call_price(cur_price, K2, T, sigma=sigma)
                else:
                    val_ps = bs_put_price(cur_price, K2, T, sigma=sigma) - bs_put_price(cur_price, K1, T, sigma=sigma)
                val_ps = max(val_ps, 0)
                val_dollars = val_ps * 100
                cost_no_comm = pos['cost'] - SPREAD_COMM
                unrealized = val_dollars - cost_no_comm
                target = profit_target * pos['max_profit']
                if target > 0 and unrealized >= target:
                    pnl = (val_dollars - SPREAD_COMM) - pos['cost']
                    should_exit = True; reason = 'early_exit'

            # Expiry
            if not should_exit and days_held >= dte:
                S = cur_price; K1, K2 = pos['K1'], pos['K2']
                eps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM)/100.0)
                if pos['mode'] == 'bull':
                    intrinsic = max(S-K1,0) - max(S-K2,0)
                else:
                    intrinsic = max(K2-S,0) - max(K1-S,0)
                pnl = (intrinsic - eps) * 100 - SPREAD_COMM
                should_exit = True; reason = 'expiry'

            if should_exit:
                to_close.append((i, pnl, reason, days_held))

        for i, pnl, reason, dh in reversed(to_close):
            pos = open_positions.pop(i)
            equity += pnl
            trades.append({
                'ticker': pos['ticker'], 'mode': pos['mode'],
                'entry_date': pos['entry_date'], 'exit_date': str(today.date()),
                'days_held': dh, 'cost': pos['cost'], 'pnl': pnl,
                'exit_reason': reason,
            })

        # --- Rebalance? ---
        if day_offset % rebal_interval == 0:
            # Find nearest pre-computed ranking
            ranking = rankings_cache.get(abs_idx)
            if ranking is None:
                # Find closest
                closest = min(rankings_cache.keys(), key=lambda k: abs(k - abs_idx),
                            default=None)
                if closest is not None and abs(closest - abs_idx) <= rebal_interval:
                    ranking = rankings_cache[closest]

            if ranking:
                # Filter to current sectors
                ranking = {k: v for k, v in ranking.items() if k in sectors}
                if ranking:
                    ranked = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
                    if invert_signal:
                        ranked = list(reversed(ranked))

                    high_vix = cur_vix >= VIX_THRESHOLD
                    if high_vix:
                        bulls = [t for t,_ in ranked[:hvix_k]]
                        bears = []
                    else:
                        bulls = [t for t,_ in ranked[:top_k]]
                        bears = [t for t,_ in ranked[-bottom_k:]]

                    max_concurrent = max(len(bulls) + len(bears), 2)
                    eq_ratio = equity / INITIAL_CAPITAL
                    max_pos = min(MAX_POS_SIZE * eq_ratio, equity * MAX_POS_PCT,
                                 equity / max_concurrent)

                    if max_pos >= 20:
                        for tk in bulls:
                            if any(p['ticker']==tk and p['mode']=='bull' for p in open_positions):
                                continue
                            S = float(sc_use[tk].loc[today])
                            sh_tk = sh_use[tk] if tk in sh_use.columns else None
                            sl_tk = sl_use[tk] if tk in sl_use.columns else None
                            sc_tk = sc_use[tk] if tk in sc_use.columns else None
                            cost, mp, K1, K2, eps = price_spread(S, 'bull', otm_pct, dte,
                                                                  sh_tk, sl_tk, sc_tk, cur_vix)
                            if cost <= 0 or cost > max_pos: continue
                            sw = K2 - K1
                            if sw > 0 and eps/sw > 0.50: continue
                            open_positions.append({
                                'ticker': tk, 'mode': 'bull', 'entry_date': str(today.date()),
                                'entry_price': S, 'K1': K1, 'K2': K2,
                                'cost': round(cost,2), 'entry_cost_ps': round(eps,6),
                                'max_profit': round(mp,2),
                            })

                        for tk in bears:
                            if any(p['ticker']==tk and p['mode']=='bear' for p in open_positions):
                                continue
                            S = float(sc_use[tk].loc[today])
                            sh_tk = sh_use[tk] if tk in sh_use.columns else None
                            sl_tk = sl_use[tk] if tk in sl_use.columns else None
                            sc_tk = sc_use[tk] if tk in sc_use.columns else None
                            cost, mp, K1, K2, eps = price_spread(S, 'bear', otm_pct, dte,
                                                                  sh_tk, sl_tk, sc_tk, cur_vix)
                            if cost <= 0 or cost > max_pos: continue
                            sw = abs(K1 - K2)
                            if sw > 0 and eps/sw > 0.50: continue
                            open_positions.append({
                                'ticker': tk, 'mode': 'bear', 'entry_date': str(today.date()),
                                'entry_price': S, 'K1': K1, 'K2': K2,
                                'cost': round(cost,2), 'entry_cost_ps': round(eps,6),
                                'max_profit': round(mp,2),
                            })

        equity_curve.append({'date': today, 'equity': equity})

    # Force-close remaining
    for pos in open_positions:
        if pos['ticker'] in sc_use.columns:
            S = float(sc_use[pos['ticker']].iloc[-1])
            K1,K2 = pos['K1'],pos['K2']
            eps = pos.get('entry_cost_ps', (pos['cost']-SPREAD_COMM)/100.0)
            if pos['mode']=='bull':
                intr = max(S-K1,0)-max(S-K2,0)
            else:
                intr = max(K2-S,0)-max(K1-S,0)
            pnl = (intr - eps)*100 - SPREAD_COMM
            equity += pnl
            trades.append({
                'ticker': pos['ticker'], 'mode': pos['mode'],
                'entry_date': pos['entry_date'],
                'exit_date': str(sc_use.index[-1].date()),
                'days_held': cfg['dte'], 'cost': pos['cost'],
                'pnl': pnl, 'exit_reason': 'force_close',
            })

    return compute_metrics(equity_curve, trades, spy, INITIAL_CAPITAL)


def _empty_result():
    return {'sharpe':0,'sortino':0,'wr':0,'pf':0,'total_return':0,
            'n_trades':0,'mdd':0,'mean_pnl':0,'trades':[],'equity_curve':[],
            'regime_sharpe_green':0,'regime_sharpe_red':0,'final_equity': INITIAL_CAPITAL}


def compute_metrics(equity_curve, trades, spy, initial_capital):
    if not trades:
        return _empty_result()

    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wins = np.sum(pnls > 0)
    wr = wins / n if n > 0 else 0
    gp = np.sum(pnls[pnls > 0])
    gl = abs(np.sum(pnls[pnls <= 0]))
    pf = gp / gl if gl > 0 else float('inf')
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if n > 1 else 1e-10

    if equity_curve:
        days = (equity_curve[-1]['date'] - equity_curve[0]['date']).days
        years = max(days / 365.25, 0.5)
        tpy = n / years
    else:
        tpy = n

    sharpe = (mean_pnl / (std_pnl + 1e-10)) * np.sqrt(tpy)
    ds = pnls[pnls < 0]
    ds_std = np.std(ds) if len(ds) > 1 else 1e-10
    sortino = (mean_pnl / (ds_std + 1e-10)) * np.sqrt(tpy)

    if equity_curve:
        eq = np.array([e['equity'] for e in equity_curve])
        pk = np.maximum.accumulate(eq)
        dd = (eq - pk) / (pk + 1e-10)
        mdd = float(np.min(dd))
    else:
        mdd = 0

    final_eq = equity_curve[-1]['equity'] if equity_curve else initial_capital
    total_ret = (final_eq / initial_capital - 1) * 100

    # Regime stratification
    green_pnls, red_pnls = [], []
    for t in trades:
        try:
            ed = pd.Timestamp(t['exit_date'])
            if ed in spy.index:
                si = spy.index.get_loc(ed)
                if si > 0:
                    sr = float(spy.iloc[si]/spy.iloc[si-1]-1)
                    (green_pnls if sr >= 0 else red_pnls).append(t['pnl'])
        except: pass

    def _sh(ps):
        if len(ps) < 3: return 0
        ps = np.array(ps)
        return ps.mean() / (ps.std()+1e-10) * np.sqrt(max(len(ps),1))

    return {
        'sharpe': round(sharpe,3), 'sortino': round(sortino,3),
        'wr': round(wr*100,1), 'pf': round(pf,2),
        'total_return': round(total_ret,1), 'n_trades': n,
        'mdd': round(mdd*100,1), 'mean_pnl': round(mean_pnl,2),
        'final_equity': round(final_eq,2),
        'equity_curve': equity_curve, 'trades': trades,
        'regime_sharpe_green': round(_sh(green_pnls),3),
        'regime_sharpe_red': round(_sh(red_pnls),3),
    }


def _print_result(r):
    print(f"  Sharpe: {r['sharpe']:.3f} | Sortino: {r['sortino']:.3f} | "
          f"WR: {r['wr']:.1f}% | PF: {r['pf']:.2f}", flush=True)
    print(f"  Trades: {r['n_trades']} | Return: {r['total_return']:.1f}% | "
          f"MDD: {r['mdd']:.1f}% | Mean PnL: ${r.get('mean_pnl',0):.2f}", flush=True)
    print(f"  Regime: Green Sharpe={r['regime_sharpe_green']:.3f}, "
          f"Red Sharpe={r['regime_sharpe_red']:.3f}", flush=True)


# ===================== ADVERSARIAL TESTS =====================

def test_reimplementation(sc, sh, sl, vix, spy, cfg, rankings, label=''):
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 1: Re-implementation backtest {label}", flush=True)
    print(f"{'='*60}", flush=True)
    r = run_backtest(sc, sh, sl, vix, spy, cfg, rankings)
    _print_result(r)
    return r


def test_inverse(sc, sh, sl, vix, spy, cfg, rankings, label=''):
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 2: Inverse signal {label}", flush=True)
    print(f"{'='*60}", flush=True)
    r = run_backtest(sc, sh, sl, vix, spy, cfg, rankings, invert_signal=True)
    _print_result(r)
    if r['sharpe'] < 0:
        print("  PASS: Inverse is negative", flush=True)
    elif r['sharpe'] < 0.5:
        print("  MARGINAL: Inverse slightly positive", flush=True)
    else:
        print("  FAIL: Inverse is profitable — no directional edge", flush=True)
    return r


def test_random_timing(sc, sh, sl, vix, spy, cfg, rankings, n_perms=500, label=''):
    """Bootstrap: shuffle the PnL series and compare Sharpe."""
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 3: Random timing ({n_perms} permutations) {label}", flush=True)
    print(f"{'='*60}", flush=True)

    real = run_backtest(sc, sh, sl, vix, spy, cfg, rankings)
    real_sharpe = real['sharpe']
    print(f"  Real Sharpe: {real_sharpe:.3f}", flush=True)

    if not real['trades']:
        print("  SKIP: No trades to permute", flush=True)
        return {'real_sharpe': real_sharpe, 'p_value': 1.0, 'pass': False,
                'random_mean': 0, 'random_std': 0}

    pnls = np.array([t['pnl'] for t in real['trades']])
    n_trades = len(pnls)

    # Estimate trades_per_year
    ec = real['equity_curve']
    if ec:
        days = (ec[-1]['date'] - ec[0]['date']).days
        years = max(days/365.25, 0.5)
        tpy = n_trades / years
    else:
        tpy = n_trades

    rng = np.random.default_rng(42)
    random_sharpes = []
    for i in range(n_perms):
        shuffled = rng.permutation(pnls)
        m = shuffled.mean()
        s = shuffled.std()
        sh_val = m / (s+1e-10) * np.sqrt(tpy)
        random_sharpes.append(sh_val)

    random_sharpes = np.array(random_sharpes)
    p_value = np.mean(random_sharpes >= real_sharpe)

    print(f"  Random Sharpe: mean={np.mean(random_sharpes):.3f}, "
          f"std={np.std(random_sharpes):.3f}, max={np.max(random_sharpes):.3f}", flush=True)
    print(f"  P-value: {p_value:.4f}", flush=True)

    if p_value < 0.05:
        print("  PASS: p < 0.05 — real edge", flush=True)
    elif p_value < 0.10:
        print("  MARGINAL: p < 0.10", flush=True)
    else:
        print("  FAIL: p >= 0.10 — random does as well", flush=True)

    return {'real_sharpe': real_sharpe, 'random_mean': round(np.mean(random_sharpes),3),
            'random_std': round(np.std(random_sharpes),3),
            'p_value': round(p_value,4), 'pass': p_value < 0.05}


def test_subperiod(sc, sh, sl, vix, spy, cfg, rankings, label=''):
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 4: Sub-period stability {label}", flush=True)
    print(f"{'='*60}", flush=True)

    # Run full backtest, then split trades by time
    full = run_backtest(sc, sh, sl, vix, spy, cfg, rankings)
    if not full['trades']:
        print("  SKIP: No trades", flush=True)
        return []

    # Sort trades by exit date and split into 4 groups
    sorted_trades = sorted(full['trades'], key=lambda t: t['exit_date'])
    n = len(sorted_trades)
    q = n // 4

    results = []
    for i in range(4):
        start = i * q
        end = (i+1)*q if i < 3 else n
        period_trades = sorted_trades[start:end]
        if not period_trades:
            results.append({'period': i+1, 'sharpe': 0, 'skip': True})
            continue

        pnls = np.array([t['pnl'] for t in period_trades])
        m = pnls.mean(); s = pnls.std() if len(pnls)>1 else 1e-10
        # Estimate annualization
        d0 = pd.Timestamp(period_trades[0]['exit_date'])
        d1 = pd.Timestamp(period_trades[-1]['exit_date'])
        days = max((d1-d0).days, 30)
        tpy = len(pnls) / (days/365.25)
        sh_val = m/(s+1e-10)*np.sqrt(tpy)
        wr = np.sum(pnls>0)/len(pnls)*100
        gp = np.sum(pnls[pnls>0])
        gl = abs(np.sum(pnls[pnls<=0]))
        pf_val = gp/(gl+1e-10)

        label_p = f"{period_trades[0]['exit_date']} to {period_trades[-1]['exit_date']}"
        print(f"  Period {i+1} ({label_p}): Sharpe={sh_val:.3f}, WR={wr:.1f}%, "
              f"PF={pf_val:.2f}, Trades={len(pnls)}", flush=True)
        results.append({'period': i+1, 'sharpe': round(sh_val,3), 'wr': round(wr,1),
                       'pf': round(pf_val,2), 'n_trades': len(pnls), 'label': label_p})

    positive = sum(1 for r in results if not r.get('skip') and r['sharpe'] > 0)
    total = sum(1 for r in results if not r.get('skip'))
    print(f"\n  {positive}/{total} periods positive", flush=True)
    if positive == total and total >= 3:
        print("  PASS: All sub-periods positive", flush=True)
    elif positive >= total - 1 and total >= 3:
        print("  MARGINAL: 1 negative period", flush=True)
    else:
        print("  FAIL: Multiple negative periods", flush=True)
    return results


def test_topn_removal(sc, sh, sl, vix, spy, cfg, rankings, remove_n=3, label=''):
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 5: Remove top-{remove_n} sectors {label}", flush=True)
    print(f"{'='*60}", flush=True)

    # Find top-N by total return
    rets = {}
    for tk in SECTORS:
        if tk in sc.columns:
            px = sc[tk].dropna()
            if len(px) > 0:
                rets[tk] = float(px.iloc[-1]/px.iloc[0]-1)
    sorted_s = sorted(rets.items(), key=lambda x: x[1], reverse=True)
    top_n = [t for t,_ in sorted_s[:remove_n]]
    remaining = [t for t in SECTORS if t not in top_n and t in sc.columns]

    print(f"  Removed: {top_n}", flush=True)
    print(f"  Remaining: {remaining}", flush=True)

    # Need to re-compute rankings for reduced universe
    rankings_reduced = precompute_rankings(sc, rebal_interval=cfg['rebal_days'],
                                           sectors=remaining)
    r = run_backtest(sc, sh, sl, vix, spy, cfg, rankings_reduced, sectors=remaining)
    _print_result(r)

    if r['sharpe'] > 0.5:
        print("  PASS: Still profitable", flush=True)
    elif r['sharpe'] > 0:
        print("  MARGINAL: Positive but degraded", flush=True)
    else:
        print("  FAIL: Negative — edge was sector exposure", flush=True)
    return r


def test_param_sensitivity(sc, sh, sl, vix, spy, cfg, rankings, label=''):
    print(f"\n{'='*60}", flush=True)
    print(f"TEST 6: Parameter sensitivity {label}", flush=True)
    print(f"{'='*60}", flush=True)

    baseline = run_backtest(sc, sh, sl, vix, spy, cfg, rankings)
    print(f"  Baseline: Sharpe={baseline['sharpe']:.3f}", flush=True)

    variations = []

    # Vary rebalance frequency (use pre-computed rankings — just change check interval)
    for rebal in [5, 10, 15, 20, 25, 30]:
        cfg_mod = dict(cfg)
        cfg_mod['rebal_days'] = rebal
        r = run_backtest(sc, sh, sl, vix, spy, cfg_mod, rankings)
        print(f"  Rebal={rebal}d: Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1f}%, "
              f"Trades={r['n_trades']}", flush=True)
        variations.append({'param': 'rebal_days', 'value': rebal, 'sharpe': r['sharpe']})

    # Vary OTM
    for otm in [1.0, 2.0, 3.0, 4.0, 5.0]:
        cfg_mod = dict(cfg)
        cfg_mod['otm_pct'] = otm
        r = run_backtest(sc, sh, sl, vix, spy, cfg_mod, rankings)
        print(f"  OTM={otm}%: Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1f}%, "
              f"Trades={r['n_trades']}", flush=True)
        variations.append({'param': 'otm_pct', 'value': otm, 'sharpe': r['sharpe']})

    # Vary profit target
    if cfg.get('profit_target') is not None:
        for pt in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, None]:
            cfg_mod = dict(cfg)
            cfg_mod['profit_target'] = pt
            r = run_backtest(sc, sh, sl, vix, spy, cfg_mod, rankings)
            pt_l = f"{int(pt*100)}%" if pt else "None"
            print(f"  PT={pt_l}: Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1f}%, "
                  f"Trades={r['n_trades']}", flush=True)
            variations.append({'param': 'profit_target', 'value': pt_l, 'sharpe': r['sharpe']})

    # Vary DTE
    for dte_v in [7, 14, 21, 28, 35]:
        cfg_mod = dict(cfg)
        cfg_mod['dte'] = dte_v
        r = run_backtest(sc, sh, sl, vix, spy, cfg_mod, rankings)
        print(f"  DTE={dte_v}: Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1f}%, "
              f"Trades={r['n_trades']}", flush=True)
        variations.append({'param': 'dte', 'value': dte_v, 'sharpe': r['sharpe']})

    sharpes = [v['sharpe'] for v in variations]
    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes)
    cv = std_s / (abs(mean_s)+1e-10)
    pos_pct = sum(1 for s in sharpes if s > 0) / len(sharpes) * 100

    print(f"\n  Sharpe across {len(variations)} variations: mean={mean_s:.3f}, "
          f"std={std_s:.3f}, CV={cv:.2f}", flush=True)
    print(f"  Positive: {pos_pct:.0f}%", flush=True)

    if pos_pct >= 80 and cv < 0.5:
        print(f"  PASS: {pos_pct:.0f}% positive, low CV", flush=True)
    elif pos_pct >= 60:
        print(f"  MARGINAL: {pos_pct:.0f}% positive", flush=True)
    else:
        print(f"  FAIL: Only {pos_pct:.0f}% positive — overfitted", flush=True)

    return {'baseline_sharpe': baseline['sharpe'], 'variations': variations,
            'mean_sharpe': round(mean_s,3), 'std_sharpe': round(std_s,3),
            'cv': round(cv,2), 'pct_positive': round(pos_pct,1)}


# ===================== MAIN =====================

def main():
    t0 = time.time()
    print("="*70, flush=True)
    print("  SECTOR COMBINED — ADVERSARIAL BACKTEST SUITE", flush=True)
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print("="*70, flush=True)

    close_df, sc, sh, sl, spy, vix = download_data(start='2019-01-01')

    results = {}

    # ========== PHASE 1: QUICK SCREEN ==========
    quick_variants = ['v7', 'v8', 'v9', 'v91', 'v92', 'v10']

    print("\n" + "#"*70, flush=True)
    print("  PHASE 1: QUICK SCREEN", flush=True)
    print("#"*70, flush=True)

    for variant in quick_variants:
        cfg = VARIANT_CONFIGS[variant]
        print(f"\n{'*'*60}", flush=True)
        print(f"  VARIANT: {variant.upper()} — DTE={cfg['dte']}, OTM={cfg['otm_pct']}%, "
              f"Rebal={cfg['rebal_days']}d, PT={cfg.get('profit_target','None')}", flush=True)
        print(f"{'*'*60}", flush=True)

        # Pre-compute rankings for this rebalance interval
        t_rank = time.time()
        rankings = precompute_rankings(sc, rebal_interval=cfg['rebal_days'])
        print(f"  Rankings computed in {time.time()-t_rank:.0f}s", flush=True)

        reimpl = test_reimplementation(sc, sh, sl, vix, spy, cfg, rankings, f"[{variant}]")
        random = test_random_timing(sc, sh, sl, vix, spy, cfg, rankings, n_perms=500, label=f"[{variant}]")

        pass_reimpl = reimpl['sharpe'] > 0.5
        pass_random = random['p_value'] < 0.10

        if pass_reimpl and pass_random:
            verdict = "ALIVE"
        elif pass_reimpl or pass_random:
            verdict = "WEAK"
        else:
            verdict = "DEAD"

        results[variant] = {
            'reimpl_sharpe': reimpl['sharpe'], 'reimpl_wr': reimpl['wr'],
            'reimpl_pf': reimpl['pf'], 'reimpl_return': reimpl['total_return'],
            'reimpl_n_trades': reimpl['n_trades'], 'reimpl_mdd': reimpl['mdd'],
            'reimpl_sortino': reimpl['sortino'],
            'reimpl_regime_green': reimpl['regime_sharpe_green'],
            'reimpl_regime_red': reimpl['regime_sharpe_red'],
            'random_p': random['p_value'],
            'pass_reimpl': pass_reimpl, 'pass_random': pass_random,
            'verdict': verdict,
        }
        print(f"\n  >>> {variant.upper()}: {verdict}", flush=True)

    # ========== PHASE 2: FULL ADVERSARIAL V93 ==========
    print("\n" + "#"*70, flush=True)
    print("  PHASE 2: FULL ADVERSARIAL — V93", flush=True)
    print("#"*70, flush=True)

    cfg93 = VARIANT_CONFIGS['v93']
    t_rank = time.time()
    rankings93 = precompute_rankings(sc, rebal_interval=cfg93['rebal_days'])
    print(f"  Rankings computed in {time.time()-t_rank:.0f}s", flush=True)

    t1 = test_reimplementation(sc, sh, sl, vix, spy, cfg93, rankings93, "[v93]")
    t2 = test_inverse(sc, sh, sl, vix, spy, cfg93, rankings93, "[v93]")
    t3 = test_random_timing(sc, sh, sl, vix, spy, cfg93, rankings93, n_perms=500, label="[v93]")
    t4 = test_subperiod(sc, sh, sl, vix, spy, cfg93, rankings93, "[v93]")
    t5 = test_topn_removal(sc, sh, sl, vix, spy, cfg93, rankings93, label="[v93]")
    t6 = test_param_sensitivity(sc, sh, sl, vix, spy, cfg93, rankings93, "[v93]")

    v93_pass = {}
    v93_pass['t1_reimpl'] = t1['sharpe'] > 0.5
    v93_pass['t2_inverse'] = t2['sharpe'] < 0
    v93_pass['t3_random'] = t3['p_value'] < 0.05
    n_pos = sum(1 for r in t4 if not r.get('skip') and r.get('sharpe',0) > 0)
    n_tot = sum(1 for r in t4 if not r.get('skip'))
    v93_pass['t4_subperiod'] = n_pos >= max(n_tot - 1, 1)
    v93_pass['t5_topn'] = t5['sharpe'] > 0
    v93_pass['t6_params'] = t6['pct_positive'] >= 60

    n_passed = sum(v93_pass.values())

    results['v93'] = {
        'reimpl_sharpe': t1['sharpe'], 'reimpl_wr': t1['wr'], 'reimpl_pf': t1['pf'],
        'reimpl_return': t1['total_return'], 'reimpl_n_trades': t1['n_trades'],
        'reimpl_mdd': t1['mdd'], 'reimpl_sortino': t1['sortino'],
        'reimpl_regime_green': t1['regime_sharpe_green'],
        'reimpl_regime_red': t1['regime_sharpe_red'],
        'inverse_sharpe': t2['sharpe'],
        'random_p': t3['p_value'],
        'subperiod_results': [{k:v for k,v in r.items()} for r in t4],
        'topn_sharpe': t5['sharpe'],
        'param_pct_positive': t6['pct_positive'],
        'param_mean_sharpe': t6['mean_sharpe'],
        'pass_fail': v93_pass,
        'n_passed': n_passed,
    }

    # ========== SUMMARY ==========
    elapsed = time.time() - t0
    print("\n" + "="*70, flush=True)
    print("  FINAL SUMMARY", flush=True)
    print("="*70, flush=True)

    print(f"\n  QUICK SCREEN:", flush=True)
    print(f"  {'Var':<6} {'Sharpe':>8} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'Ret%':>7} {'MDD%':>7} {'Trades':>7} {'p-val':>7} {'Verdict':<8}", flush=True)
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}",
          flush=True)
    for v in quick_variants:
        r = results[v]
        print(f"  {v:<6} {r['reimpl_sharpe']:>8.3f} {r['reimpl_sortino']:>8.3f} "
              f"{r['reimpl_wr']:>6.1f} {r['reimpl_pf']:>6.2f} "
              f"{r['reimpl_return']:>7.1f} {r['reimpl_mdd']:>7.1f} "
              f"{r['reimpl_n_trades']:>7} {r['random_p']:>7.4f} {r['verdict']:<8}",
              flush=True)

    print(f"\n  V93 FULL ADVERSARIAL ({n_passed}/6 passed):", flush=True)
    for k, name in [('t1_reimpl','Re-implementation'), ('t2_inverse','Inverse signal'),
                    ('t3_random','Random timing'), ('t4_subperiod','Sub-period stability'),
                    ('t5_topn','Top-N removal'), ('t6_params','Parameter sensitivity')]:
        status = "PASS" if v93_pass[k] else "FAIL"
        print(f"    {name:<25} {status}", flush=True)

    # Regime check
    rg = t1['regime_sharpe_green']
    rr = t1['regime_sharpe_red']
    regime_gap = abs(rg - rr) / (max(abs(rg), abs(rr)) + 1e-10)
    print(f"\n  V93 Regime: Green={rg:.3f}, Red={rr:.3f}, "
          f"Gap={regime_gap:.2f} ({'PASS' if regime_gap < 0.50 else 'FAIL'} <0.50)", flush=True)

    if n_passed >= 5:
        v93_v = "STRONG EDGE — keep and trust"
    elif n_passed >= 4:
        v93_v = "MODERATE EDGE — keep, monitor"
    elif n_passed >= 3:
        v93_v = "WEAK EDGE — reduce size"
    else:
        v93_v = "NO EDGE — consider killing"
    print(f"\n  V93 VERDICT: {v93_v}", flush=True)

    print(f"\n  KILL LIST:", flush=True)
    for v in quick_variants:
        if results[v]['verdict'] == 'DEAD':
            print(f"    {v.upper()}: KILL — no edge", flush=True)
    for v in quick_variants:
        if results[v]['verdict'] == 'WEAK':
            print(f"    {v.upper()}: PROBATION", flush=True)
    for v in quick_variants:
        if results[v]['verdict'] == 'ALIVE':
            print(f"    {v.upper()}: KEEP", flush=True)

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

    # Save
    out = BASE / 'paper_engines' / 'adversarial_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved.", flush=True)


if __name__ == '__main__':
    main()
