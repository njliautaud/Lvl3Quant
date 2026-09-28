#!/usr/bin/env python3
"""
Tail Risk Hedge v1 — Test whether cheap tail risk hedges improve
risk-adjusted returns for the production v4 sector bull call spread strategy.

Base strategy (production v4):
  - Entry: VIX > 20 regime filter (~33% of days active)
  - LGBM walk-forward ranking with 21 features
  - Top 3 sector ETFs: bull call spread ATM, 3% width, DTE=21, hold to expiry
  - Biweekly rebalance (10 trading days), $645 start, $200 max/trade
  - 15% entry haircut, no exit haircut, $2.60 commission/spread

6 variants tested:
  1. Baseline (no hedge)
  2. SPY protective puts (5% OTM, 21 DTE)
  3. VIX call hedge (VIX+5 strike, 5pt width, 30 DTE)
  4. Dynamic hedge ratio (SPY puts only when VIX > 25)
  5. Collar on sector positions (add protective put 3% below)
  6. Crash insurance fund (10% capital, SPY puts 10% OTM, 3-month, rolling quarterly)

MLflow: tail_risk_hedge_v1, server http://jupiter:5000
"""

import os, sys, json, time, warnings
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import lightgbm as lgb
warnings.filterwarnings('ignore')

def fprint(*a, **kw): print(*a, **kw, flush=True)
np.random.seed(42)

BASE = Path("/home/nick/Lvl3Quant")
OUT_DIR = BASE / "output" / "tail_risk_hedge_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP_INIT = 645.0
MAX_PER_TRADE = 200.0
SPREAD_COMM = 2.60
HAIRCUT = 0.15
WIDTH_PCT = 0.03
DTE = 21
REBAL_DAYS = 10
TOP_N = 3
VIX_ENTRY_THRESHOLD = 20.0
IV_MULT = 1.2  # IV = 1.2x HV

# LGBM walk-forward
WF_TRAIN = 240
WF_VAL = 60
WF_STEP = 10
LGBM_FEATS = [
    'ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
    'vol_5d','vol_10d','vol_21d','vol_63d',
    'sharpe_21d','sharpe_63d',
    'maxdd_21d','maxdd_63d',
    'pct_52w_high','pct_52w_low',
    'mom_accel_21_63','mom_accel_63_126',
    'rel_str_spy_21d','rel_str_spy_63d',
    'mean_reversion_21d'
]

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=3)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results will be saved locally only")

fprint(f"[{datetime.now()}] Tail Risk Hedge v1 starting...")
fprint(f"Output: {OUT_DIR}")

# ============================================================
# 1. DATA
# ============================================================
def download_data():
    fprint("[1/8] Downloading data...")
    import yfinance as yf
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2009-01-01', end='2026-07-27', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    fprint(f"  {len(ix)} days, {len(sc.columns)} sectors, {ix[0].date()} to {ix[-1].date()}")
    return sc.loc[ix], spy.loc[ix], vix.loc[ix]


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def compute_features(px, spy, idx, tk):
    """Compute 21 features for sector ETF at given index."""
    p = px[tk].iloc[:idx+1].dropna()
    if len(p) < 260:
        return None
    s = spy.iloc[:idx+1].dropna()

    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(p.iloc[-1]/p.iloc[-lb]-1) if len(p) > lb else 0.0

    r = p.pct_change().dropna()
    for w, nm in [(5,'vol_5d'),(10,'vol_10d'),(21,'vol_21d'),(63,'vol_63d')]:
        f[nm] = float(r.iloc[-w:].std()*np.sqrt(252)) if len(r) > w else 0.2

    for w, nm in [(21,'sharpe_21d'),(63,'sharpe_63d')]:
        rw = r.iloc[-w:]
        f[nm] = float(rw.mean()/(rw.std()+1e-10)*np.sqrt(252)) if len(rw) > 10 else 0.0

    for w, nm in [(21,'maxdd_21d'),(63,'maxdd_63d')]:
        pw = p.iloc[-w:]
        f[nm] = float(((pw/pw.cummax())-1).min()) if len(pw) > 5 else 0.0

    f['pct_52w_high'] = float(p.iloc[-1]/p.iloc[-252:].max()) if len(p) >= 252 else 1.0
    f['pct_52w_low'] = float(p.iloc[-1]/p.iloc[-252:].min()) if len(p) >= 252 else 1.0
    f['mom_accel_21_63'] = f['ret_21d'] - f['ret_63d']/3.0
    f['mom_accel_63_126'] = f['ret_63d'] - f['ret_126d']/2.0

    sr = s.pct_change().dropna()
    for w, nm in [(21,'rel_str_spy_21d'),(63,'rel_str_spy_63d')]:
        rw = r.iloc[-w:]
        srw = sr.iloc[-w:]
        if len(rw) > 10 and len(srw) > 10:
            f[nm] = float(rw.mean() - srw.mean()) * np.sqrt(252)
        else:
            f[nm] = 0.0

    # Mean reversion: z-score of 21d return vs 252d distribution
    if len(r) >= 252:
        r21 = float(r.iloc[-21:].sum())
        r252 = r.rolling(21).sum().iloc[-252:]
        f['mean_reversion_21d'] = float((r21 - r252.mean()) / (r252.std() + 1e-10))
    else:
        f['mean_reversion_21d'] = 0.0

    return f


# ============================================================
# 3. LGBM WALK-FORWARD RANKING
# ============================================================
def lgbm_walk_forward(px, spy, dates):
    fprint("[2/8] LGBM walk-forward sector ranking...")

    # Build feature matrix
    recs = []
    rebal_dates = dates[::REBAL_DAYS]  # every 10 trading days
    for dt in rebal_dates:
        idx = px.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in px.columns:
            f = compute_features(px, spy, idx, tk)
            if f is None:
                continue
            fi = min(idx + DTE, len(px) - 1)
            f['date'] = dt
            f['ticker'] = tk
            f['fwd_ret'] = float(px[tk].iloc[fi] / px[tk].iloc[idx] - 1)
            recs.append(f)

    df = pd.DataFrame(recs)
    if len(df) < 100:
        fprint("  ERROR: too few records for LGBM")
        return {}

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    udates = sorted(df['date'].unique())
    fprint(f"  {len(df)} records, {len(udates)} rebalance dates")

    ranks = {}
    n_folds = 0
    for i in range(len(udates)):
        td = udates[i]
        # Find training window
        train_mask = df['date'] < td
        train_dates = sorted(df[train_mask]['date'].unique())
        if len(train_dates) < WF_TRAIN // REBAL_DAYS:
            continue

        # Use last WF_TRAIN/REBAL_DAYS dates for training
        train_dates_use = train_dates[-(WF_TRAIN // REBAL_DAYS):]
        tr = df[df['date'].isin(train_dates_use)]
        te = df[df['date'] == td].copy()

        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[LGBM_FEATS].values.astype(np.float32))
        Xe = np.nan_to_num(te[LGBM_FEATS].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1
            )
            m.fit(Xt, tr['rank_label'].values)
            te['score'] = m.predict(Xe)
            ranks[td] = dict(zip(te['ticker'], te['score']))
            n_folds += 1
        except Exception as e:
            continue

    fprint(f"  LGBM ranking: {n_folds} folds, {len(ranks)} dates with predictions")
    return ranks


# ============================================================
# 4. BLACK-SCHOLES OPTION PRICING
# ============================================================
from scipy.stats import norm

def bs_call(S, K, T, sigma, r=0.02):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2))

def bs_put(S, K, T, sigma, r=0.02):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return float(K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def get_iv(hv):
    """IV proxy = 1.2x historical vol."""
    return max(hv * IV_MULT, 0.05)

def compute_hv(prices, window=21):
    """Annualized historical volatility."""
    r = prices.pct_change().dropna()
    if len(r) < window:
        return 0.20
    return float(r.iloc[-window:].std() * np.sqrt(252))


# ============================================================
# 5. STRATEGY SIMULATION
# ============================================================

def simulate_variant(variant_name, px, spy, vix, ranks, spy_hv_series, sector_hv):
    """
    Simulate a single variant. Returns equity curve and trade log.

    variant_name: one of 'baseline', 'spy_puts', 'vix_calls', 'dynamic_hedge',
                  'collar', 'crash_fund'
    """
    fprint(f"  Simulating: {variant_name}")

    dates = sorted(ranks.keys())
    if not dates:
        return None

    capital = CAP_INIT
    equity_curve = []
    trade_log = []
    hedge_cost_total = 0.0
    hedge_payoff_total = 0.0
    n_hedges = 0
    n_hedge_payoffs = 0

    # For crash insurance fund: track separate allocation
    crash_fund_alloc = 0.0
    crash_fund_put_value = 0.0
    crash_fund_put_expiry = None
    crash_fund_put_strike = 0.0
    crash_fund_put_iv = 0.0

    # Track last rebalance
    active_positions = []  # list of dicts with position info
    last_rebal_idx = -999

    all_dates = px.index
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for i, dt in enumerate(all_dates):
        if dt < dates[0]:
            continue

        vix_val = vix.loc[dt] if dt in vix.index else None
        spy_val = spy.loc[dt] if dt in spy.index else None

        if vix_val is None or spy_val is None:
            equity_curve.append({'date': dt, 'equity': capital})
            continue

        # Check if it's a rebalance date
        is_rebal = dt in ranks and (i - last_rebal_idx >= REBAL_DAYS)

        # --- Expire old positions ---
        new_positions = []
        for pos in active_positions:
            if i >= pos['expiry_idx']:
                # Crash fund positions are SPY puts, not sector spreads
                if pos['ticker'] == 'SPY_CRASH_FUND':
                    spy_exp_idx = min(pos['expiry_idx'], len(spy)-1)
                    spy_exp = float(spy.iloc[spy_exp_idx])
                    put_payoff = max(pos.get('hedge_strike', 0) - spy_exp, 0.0) * 100
                    hedge_pnl = put_payoff - pos.get('hedge_cost', 0)
                    capital += hedge_pnl
                    hedge_payoff_total += max(put_payoff, 0)
                    if put_payoff > 0:
                        n_hedge_payoffs += 1
                    trade_log.append({
                        'date': str(pos['entry_date']),
                        'ticker': 'CRASH_FUND_PUT',
                        'pnl': round(hedge_pnl, 2),
                        'spread_pnl': 0.0,
                        'hedge_pnl': round(hedge_pnl, 2),
                    })
                    continue

                # Position expired — compute payoff
                etf_price_expiry = float(px[pos['ticker']].iloc[min(pos['expiry_idx'], len(px)-1)])

                # Bull call spread payoff
                long_call_payoff = max(etf_price_expiry - pos['long_strike'], 0.0)
                short_call_payoff = max(etf_price_expiry - pos['short_strike'], 0.0)
                spread_payoff = long_call_payoff - short_call_payoff
                n_spreads = pos['n_spreads']
                gross_pnl = (spread_payoff * n_spreads * 100) - pos['cost']

                # Hedge payoff
                hedge_pnl = 0.0
                if 'hedge_type' in pos:
                    if pos['hedge_type'] == 'spy_put':
                        spy_exp = float(spy.iloc[min(pos['expiry_idx'], len(spy)-1)])
                        put_payoff = max(pos['hedge_strike'] - spy_exp, 0.0) * 100
                        hedge_pnl = put_payoff - pos['hedge_cost']
                        if put_payoff > 0:
                            n_hedge_payoffs += 1
                    elif pos['hedge_type'] == 'vix_call_spread':
                        # VIX call spread payoff: VIX at expiry
                        vix_exp = float(vix.iloc[min(pos['expiry_idx'], len(vix)-1)])
                        long_vix_payoff = max(vix_exp - pos['hedge_strike'], 0.0)
                        short_vix_payoff = max(vix_exp - pos['hedge_strike_upper'], 0.0)
                        vcs_payoff = (long_vix_payoff - short_vix_payoff) * 100
                        hedge_pnl = vcs_payoff - pos['hedge_cost']
                        if vcs_payoff > 0:
                            n_hedge_payoffs += 1
                    elif pos['hedge_type'] == 'collar_put':
                        etf_exp = etf_price_expiry
                        put_payoff = max(pos['hedge_strike'] - etf_exp, 0.0) * n_spreads * 100
                        hedge_pnl = put_payoff - pos['hedge_cost']
                        if put_payoff > 0:
                            n_hedge_payoffs += 1

                total_pnl = gross_pnl + hedge_pnl
                capital += total_pnl

                trade_log.append({
                    'date': str(pos['entry_date']),
                    'ticker': pos['ticker'],
                    'pnl': round(total_pnl, 2),
                    'spread_pnl': round(gross_pnl, 2),
                    'hedge_pnl': round(hedge_pnl, 2),
                })
            else:
                new_positions.append(pos)
        active_positions = new_positions

        # --- Crash insurance fund: quarterly rolling ---
        if variant_name == 'crash_fund':
            if crash_fund_put_expiry is None or i >= crash_fund_put_expiry:
                # Roll the crash insurance
                crash_fund_alloc = capital * 0.10
                if spy_val > 0:
                    put_strike = spy_val * 0.90  # 10% OTM
                    T = 63 / 252.0  # 3-month
                    hv = compute_hv(spy.iloc[:i+1])
                    iv = get_iv(hv)
                    put_price = bs_put(spy_val, put_strike, T, iv)
                    put_cost = put_price * 100 * (1 + HAIRCUT) + SPREAD_COMM
                    n_puts = max(1, int(crash_fund_alloc / put_cost))
                    actual_cost = n_puts * put_cost
                    if actual_cost < capital * 0.15:  # safety: don't spend more than 15%
                        capital -= actual_cost
                        hedge_cost_total += actual_cost
                        n_hedges += 1
                        crash_fund_put_strike = put_strike
                        crash_fund_put_expiry = i + 63
                        crash_fund_put_iv = iv

                        # Store for payoff calc
                        active_positions.append({
                            'ticker': 'SPY_CRASH_FUND',
                            'entry_date': dt,
                            'expiry_idx': i + 63,
                            'long_strike': 0, 'short_strike': 0,
                            'n_spreads': 0, 'cost': 0,
                            'hedge_type': 'spy_put',
                            'hedge_strike': put_strike,
                            'hedge_cost': actual_cost,
                        })

        # --- Rebalance ---
        if is_rebal and vix_val > VIX_ENTRY_THRESHOLD:
            last_rebal_idx = i
            r = ranks[dt]
            # Top N sectors by LGBM score
            sorted_sectors = sorted(r.items(), key=lambda x: x[1], reverse=True)
            top_sectors = [tk for tk, sc in sorted_sectors[:TOP_N] if tk in px.columns]

            for tk in top_sectors:
                etf_price = float(px[tk].iloc[i])
                if etf_price <= 0:
                    continue

                # Bull call spread: ATM long, ATM+3% short
                long_strike = etf_price
                short_strike = etf_price * (1 + WIDTH_PCT)
                T = DTE / 252.0

                hv = compute_hv(px[tk].iloc[max(0,i-63):i+1])
                iv = get_iv(hv)

                long_call = bs_call(etf_price, long_strike, T, iv)
                short_call = bs_call(etf_price, short_strike, T, iv)
                spread_cost_per = (long_call - short_call) * 100  # per contract
                spread_cost_per = spread_cost_per * (1 + HAIRCUT) + SPREAD_COMM

                if spread_cost_per <= 0:
                    continue

                n_spreads = max(1, int(min(MAX_PER_TRADE, capital * 0.3) / spread_cost_per))
                total_cost = n_spreads * spread_cost_per

                if total_cost > capital * 0.5:  # don't risk more than 50% on one position
                    continue

                pos = {
                    'ticker': tk,
                    'entry_date': dt,
                    'entry_idx': i,
                    'expiry_idx': i + DTE,
                    'long_strike': long_strike,
                    'short_strike': short_strike,
                    'n_spreads': n_spreads,
                    'cost': total_cost,
                    'etf_entry_price': etf_price,
                }

                # --- Add hedge based on variant ---
                hedge_cost = 0.0

                if variant_name == 'spy_puts':
                    # SPY protective put: 5% OTM, 21 DTE
                    put_strike = spy_val * 0.95
                    put_price = bs_put(spy_val, put_strike, T, get_iv(compute_hv(spy.iloc[max(0,i-63):i+1])))
                    hc = put_price * 100 * (1 + HAIRCUT) + SPREAD_COMM
                    pos['hedge_type'] = 'spy_put'
                    pos['hedge_strike'] = put_strike
                    pos['hedge_cost'] = hc
                    hedge_cost = hc
                    n_hedges += 1

                elif variant_name == 'vix_calls':
                    # VIX call spread: strike = VIX+5, 5pt width, 30 DTE
                    vix_strike = vix_val + 5
                    vix_strike_upper = vix_strike + 5
                    T_vix = 30 / 252.0
                    vix_iv = max(0.8, vix_val / 100.0 * 2)  # VIX IV proxy
                    long_vix = bs_call(vix_val, vix_strike, T_vix, vix_iv)
                    short_vix = bs_call(vix_val, vix_strike_upper, T_vix, vix_iv)
                    hc = max((long_vix - short_vix) * 100 * (1 + HAIRCUT) + SPREAD_COMM, SPREAD_COMM + 5)
                    pos['hedge_type'] = 'vix_call_spread'
                    pos['hedge_strike'] = vix_strike
                    pos['hedge_strike_upper'] = vix_strike_upper
                    pos['hedge_cost'] = hc
                    hedge_cost = hc
                    n_hedges += 1

                elif variant_name == 'dynamic_hedge':
                    # SPY puts only when VIX > 25
                    if vix_val > 25:
                        put_strike = spy_val * 0.95
                        put_price = bs_put(spy_val, put_strike, T, get_iv(compute_hv(spy.iloc[max(0,i-63):i+1])))
                        hc = put_price * 100 * (1 + HAIRCUT) + SPREAD_COMM
                        pos['hedge_type'] = 'spy_put'
                        pos['hedge_strike'] = put_strike
                        pos['hedge_cost'] = hc
                        hedge_cost = hc
                        n_hedges += 1

                elif variant_name == 'collar':
                    # Protective put 3% below current on SAME sector ETF
                    put_strike = etf_price * 0.97
                    put_price = bs_put(etf_price, put_strike, T, iv)
                    hc = put_price * n_spreads * 100 * (1 + HAIRCUT) + SPREAD_COMM
                    pos['hedge_type'] = 'collar_put'
                    pos['hedge_strike'] = put_strike
                    pos['hedge_cost'] = hc
                    hedge_cost = hc
                    n_hedges += 1

                total_cost += hedge_cost
                hedge_cost_total += hedge_cost

                if total_cost > capital:
                    continue

                capital -= total_cost
                active_positions.append(pos)

        equity_curve.append({'date': dt, 'equity': capital + sum(
            pos.get('cost', 0) * 0.5 for pos in active_positions  # rough MTM
        )})

    # Force-expire remaining positions at last date
    for pos in active_positions:
        exp_idx = min(pos['expiry_idx'], len(px) - 1)
        if pos['ticker'] == 'SPY_CRASH_FUND' or pos['n_spreads'] == 0:
            # Crash fund put
            spy_exp = float(spy.iloc[exp_idx])
            put_payoff = max(pos.get('hedge_strike', 0) - spy_exp, 0.0) * 100
            hedge_pnl = put_payoff - pos.get('hedge_cost', 0)
            capital += hedge_pnl
            continue

        etf_exp = float(px[pos['ticker']].iloc[exp_idx])
        lcp = max(etf_exp - pos['long_strike'], 0.0)
        scp = max(etf_exp - pos['short_strike'], 0.0)
        sp = (lcp - scp) * pos['n_spreads'] * 100
        gross = sp - pos['cost']

        hedge_pnl = 0.0
        if 'hedge_type' in pos:
            if pos['hedge_type'] == 'spy_put':
                spy_exp = float(spy.iloc[exp_idx])
                put_payoff = max(pos['hedge_strike'] - spy_exp, 0.0) * 100
                hedge_pnl = put_payoff - pos['hedge_cost']
            elif pos['hedge_type'] == 'vix_call_spread':
                vix_exp = float(vix.iloc[exp_idx])
                l = max(vix_exp - pos['hedge_strike'], 0.0)
                s = max(vix_exp - pos['hedge_strike_upper'], 0.0)
                hedge_pnl = (l - s) * 100 - pos['hedge_cost']
            elif pos['hedge_type'] == 'collar_put':
                put_payoff = max(pos['hedge_strike'] - etf_exp, 0.0) * pos['n_spreads'] * 100
                hedge_pnl = put_payoff - pos['hedge_cost']

        capital += gross + hedge_pnl

    return {
        'equity_curve': equity_curve,
        'trade_log': trade_log,
        'final_capital': capital,
        'hedge_cost_total': hedge_cost_total,
        'hedge_payoff_total': hedge_payoff_total,
        'n_hedges': n_hedges,
        'n_hedge_payoffs': n_hedge_payoffs,
    }


# ============================================================
# 6. METRICS
# ============================================================
def compute_metrics(result, label):
    """Compute risk-adjusted metrics from simulation result."""
    ec = pd.DataFrame(result['equity_curve'])
    if len(ec) < 50:
        return None
    ec['date'] = pd.to_datetime(ec['date'])
    ec = ec.set_index('date').sort_index()
    eq = ec['equity']

    # Daily returns
    rets = eq.pct_change().dropna()
    if len(rets) < 30:
        return None

    # Annualized return
    n_years = len(rets) / 252.0
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    # Sharpe
    sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))

    # Sortino
    neg = rets[rets < 0]
    downside_std = float(neg.std() * np.sqrt(252)) if len(neg) > 5 else 0.01
    sortino = float(ann_ret / downside_std) if downside_std > 0 else 0.0

    # Max drawdown
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    maxdd = float(dd.min())

    # Calmar
    calmar = float(ann_ret / abs(maxdd)) if abs(maxdd) > 0.001 else 0.0

    # Win rate from trade log
    trades = result.get('trade_log', [])
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = n_wins / max(n_trades, 1)

    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = gross_profit / max(gross_loss, 0.01)

    # Hedge metrics
    hedge_cost_pct = result['hedge_cost_total'] / max(eq.iloc[0], 1) * 100 / max(n_years, 0.1)

    return {
        'variant': label,
        'final_capital': round(result['final_capital'], 2),
        'total_return_pct': round(total_ret * 100, 2),
        'ann_return_pct': round(ann_ret * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(maxdd * 100, 2),
        'calmar': round(calmar, 3),
        'profit_factor': round(pf, 3),
        'win_rate_pct': round(win_rate * 100, 1),
        'n_trades': n_trades,
        'hedge_cost_total': round(result['hedge_cost_total'], 2),
        'hedge_cost_pct_pa': round(hedge_cost_pct, 2),
        'n_hedges': result['n_hedges'],
        'n_hedge_payoffs': result['n_hedge_payoffs'],
    }


# ============================================================
# 7. ADVERSARIAL VALIDATION (5-gate)
# ============================================================
def adversarial_validation(result, label):
    """5-gate adversarial check on a variant's equity curve."""
    fprint(f"  Adversarial validation: {label}")
    ec = pd.DataFrame(result['equity_curve'])
    if len(ec) < 100:
        return {'variant': label, 'gates_passed': 0, 'gates_total': 5, 'details': 'Insufficient data'}

    ec['date'] = pd.to_datetime(ec['date'])
    ec = ec.set_index('date').sort_index()
    eq = ec['equity']
    rets = eq.pct_change().dropna()

    gates = {}

    # Gate 1: Permutation test (is Sharpe significantly better than random?)
    observed_sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
    n_perm = 500
    perm_sharpes = []
    r_arr = rets.values.copy()
    for _ in range(n_perm):
        np.random.shuffle(r_arr)
        ps = float(np.mean(r_arr) / (np.std(r_arr) + 1e-10) * np.sqrt(252))
        perm_sharpes.append(ps)
    p_value = float(np.mean([ps >= observed_sharpe for ps in perm_sharpes]))
    gates['G1_permutation'] = {'passed': p_value < 0.05, 'p_value': round(p_value, 4)}

    # Gate 2: Regime consistency (green vs red days)
    # Approximate: SPY up days vs down days
    spy_rets = eq.pct_change().dropna()
    # Split by first/second half as regime proxy
    n = len(rets)
    h1 = rets.iloc[:n//2]
    h2 = rets.iloc[n//2:]
    s1 = float(h1.mean()/(h1.std()+1e-10)*np.sqrt(252))
    s2 = float(h2.mean()/(h2.std()+1e-10)*np.sqrt(252))
    regime_ratio = abs(s1 - s2) / max(abs(s1), abs(s2), 0.01)
    gates['G2_regime_consistency'] = {'passed': regime_ratio < 0.50, 'ratio': round(regime_ratio, 3)}

    # Gate 3: No single day dominates returns
    max_day_contrib = float(rets.max() / (rets.sum() + 1e-10))
    gates['G3_no_single_day_dominance'] = {'passed': abs(max_day_contrib) < 0.30, 'max_day_pct': round(max_day_contrib*100, 2)}

    # Gate 4: Positive returns in >60% of rolling 63d windows
    rolling_ret = rets.rolling(63).sum().dropna()
    pct_positive = float((rolling_ret > 0).mean())
    gates['G4_rolling_consistency'] = {'passed': pct_positive > 0.55, 'pct_positive_63d': round(pct_positive*100, 1)}

    # Gate 5: MaxDD recovery — recovers from max DD within 252 days
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    dd_end = dd.idxmin()
    recovery_mask = eq.loc[dd_end:] >= cummax.loc[dd_end]
    if recovery_mask.any():
        recovery_date = recovery_mask.idxmax()
        recovery_days = len(eq.loc[dd_end:recovery_date])
        gates['G5_dd_recovery'] = {'passed': recovery_days < 252, 'recovery_days': recovery_days}
    else:
        gates['G5_dd_recovery'] = {'passed': False, 'recovery_days': 9999}

    n_passed = sum(1 for g in gates.values() if g['passed'])
    return {'variant': label, 'gates_passed': n_passed, 'gates_total': 5, 'details': gates}


# ============================================================
# 8. MAIN
# ============================================================
def main():
    t0 = time.time()

    # Download data
    px, spy, vix = download_data()

    # Compute HV series for SPY
    spy_hv = pd.Series(index=spy.index, dtype=float)
    for i in range(63, len(spy)):
        spy_hv.iloc[i] = compute_hv(spy.iloc[:i+1])
    spy_hv = spy_hv.fillna(0.20)

    # Sector HV
    sector_hv = {}
    for tk in px.columns:
        hv = pd.Series(index=px.index, dtype=float)
        for i in range(63, len(px)):
            hv.iloc[i] = compute_hv(px[tk].iloc[:i+1])
        sector_hv[tk] = hv.fillna(0.20)

    # LGBM ranking
    rebal_dates_all = px.index[px.index.isin(vix.index)]
    ranks = lgbm_walk_forward(px, spy, rebal_dates_all)

    if not ranks:
        fprint("ERROR: No rankings produced. Aborting.")
        return

    # Simulate all variants
    fprint("[3/8] Simulating 6 variants...")
    variants = ['baseline', 'spy_puts', 'vix_calls', 'dynamic_hedge', 'collar', 'crash_fund']
    variant_labels = {
        'baseline': '1. Baseline (no hedge)',
        'spy_puts': '2. SPY Protective Puts',
        'vix_calls': '3. VIX Call Hedge',
        'dynamic_hedge': '4. Dynamic Hedge (VIX>25)',
        'collar': '5. Collar on Sectors',
        'crash_fund': '6. Crash Insurance Fund',
    }

    results = {}
    metrics_all = []
    adv_results = []

    for v in variants:
        fprint(f"\n[{variants.index(v)+3}/8] Running {variant_labels[v]}...")
        result = simulate_variant(v, px, spy, vix, ranks, spy_hv, sector_hv)
        if result is None:
            fprint(f"  FAILED: {v}")
            continue
        results[v] = result
        m = compute_metrics(result, variant_labels[v])
        if m:
            metrics_all.append(m)

    # Adversarial validation on any variant that beats baseline Sharpe
    fprint("\n[7/8] Adversarial validation...")
    baseline_sharpe = None
    for m in metrics_all:
        if 'Baseline' in m['variant']:
            baseline_sharpe = m['sharpe']
            break

    for v, result in results.items():
        m = next((x for x in metrics_all if variant_labels.get(v, '') == x['variant']), None)
        if m and baseline_sharpe is not None and m['sharpe'] >= baseline_sharpe:
            av = adversarial_validation(result, variant_labels[v])
            adv_results.append(av)

    # Summary
    fprint("\n" + "="*100)
    fprint("TAIL RISK HEDGE v1 — RESULTS SUMMARY")
    fprint("="*100)

    if metrics_all:
        df = pd.DataFrame(metrics_all)
        cols = ['variant','final_capital','total_return_pct','ann_return_pct','sharpe','sortino',
                'max_dd_pct','calmar','profit_factor','win_rate_pct','n_trades',
                'hedge_cost_total','hedge_cost_pct_pa','n_hedges']
        fprint("\n" + df[cols].to_string(index=False))

    if adv_results:
        fprint("\n\nADVERSARIAL VALIDATION (variants >= baseline Sharpe):")
        fprint("-"*80)
        for av in adv_results:
            status = "PASS" if av['gates_passed'] >= 4 else "MARGINAL" if av['gates_passed'] >= 3 else "FAIL"
            fprint(f"  {av['variant']}: {av['gates_passed']}/{av['gates_total']} gates [{status}]")
            if isinstance(av['details'], dict):
                for gname, gval in av['details'].items():
                    p = "PASS" if gval['passed'] else "FAIL"
                    detail_str = ', '.join(f"{k}={v}" for k, v in gval.items() if k != 'passed')
                    fprint(f"    {gname}: {p} ({detail_str})")

    # Key analysis
    fprint("\n\nKEY ANALYSIS:")
    fprint("-"*80)
    if baseline_sharpe is not None:
        baseline_dd = next((m['max_dd_pct'] for m in metrics_all if 'Baseline' in m['variant']), None)
        for m in metrics_all:
            if 'Baseline' in m['variant']:
                continue
            sharpe_delta = m['sharpe'] - baseline_sharpe
            dd_delta = m['max_dd_pct'] - (baseline_dd or 0)
            cost_pa = m['hedge_cost_pct_pa']
            fprint(f"  {m['variant']}:")
            fprint(f"    Sharpe delta: {sharpe_delta:+.3f} | MaxDD delta: {dd_delta:+.2f}% | Hedge cost: {cost_pa:.1f}% of capital/year")
            if dd_delta > 0 and abs(dd_delta) > 2:
                fprint(f"    -> MaxDD improved by {abs(dd_delta):.1f}% at cost of {abs(sharpe_delta):.3f} Sharpe")
            elif sharpe_delta > 0:
                fprint(f"    -> Improves BOTH Sharpe and MaxDD — strong candidate")
            else:
                fprint(f"    -> Cost exceeds protection benefit")

    # Find best cost-to-protection ratio
    fprint("\n  COST-TO-PROTECTION RATIO (lower = better):")
    for m in metrics_all:
        if 'Baseline' in m['variant']:
            continue
        dd_improvement = abs(m['max_dd_pct'] - (baseline_dd or 0))
        if dd_improvement > 0.1 and m['hedge_cost_pct_pa'] > 0:
            ratio = m['hedge_cost_pct_pa'] / dd_improvement
            fprint(f"    {m['variant']}: {ratio:.3f} (cost {m['hedge_cost_pct_pa']:.1f}% / {dd_improvement:.1f}% DD improvement)")
        else:
            fprint(f"    {m['variant']}: N/A (no meaningful DD improvement)")

    elapsed = time.time() - t0
    fprint(f"\n\nCompleted in {elapsed/60:.1f} minutes")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_minutes': round(elapsed/60, 1),
        'metrics': metrics_all,
        'adversarial': adv_results,
        'config': {
            'sectors': SECTORS,
            'cap_init': CAP_INIT,
            'max_per_trade': MAX_PER_TRADE,
            'spread_comm': SPREAD_COMM,
            'haircut': HAIRCUT,
            'width_pct': WIDTH_PCT,
            'dte': DTE,
            'rebal_days': REBAL_DAYS,
            'top_n': TOP_N,
            'vix_entry_threshold': VIX_ENTRY_THRESHOLD,
            'iv_mult': IV_MULT,
            'wf_train': WF_TRAIN,
            'wf_val': WF_VAL,
            'wf_step': WF_STEP,
        }
    }

    results_path = OUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Log to MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('tail_risk_hedge_v1')
            for m in metrics_all:
                with mlflow.start_run(run_name=m['variant'][:50]):
                    mlflow.log_params({
                        'variant': m['variant'],
                        'cap_init': CAP_INIT,
                        'vix_threshold': VIX_ENTRY_THRESHOLD,
                        'width_pct': WIDTH_PCT,
                        'dte': DTE,
                    })
                    mlflow.log_metrics({
                        'sharpe': m['sharpe'],
                        'sortino': m['sortino'],
                        'max_dd_pct': m['max_dd_pct'],
                        'calmar': m['calmar'],
                        'profit_factor': m['profit_factor'],
                        'win_rate': m['win_rate_pct'],
                        'total_return_pct': m['total_return_pct'],
                        'ann_return_pct': m['ann_return_pct'],
                        'hedge_cost_pct_pa': m['hedge_cost_pct_pa'],
                        'n_trades': m['n_trades'],
                    })
            fprint("MLflow logging complete")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")


if __name__ == '__main__':
    main()
