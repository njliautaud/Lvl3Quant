#!/usr/bin/env python3
"""
Selective Conviction Options V1
====================================

HYPOTHESIS: Instead of monthly systematic options rotation (which FAILS at $645),
trade ONLY when the LGBM sector ranking gives an EXCEPTIONALLY strong signal.
Trade 2-4 times per year, not 12. Selectivity overcomes theta decay.

SIX VARIANTS:
  A: Top-1 sector, ATM call, DTE 30, ranking score > 80th percentile
  B: Top-1 sector, ATM call, DTE 45, ranking score > 80th pctl + VIX < 25
  C: Top-1 sector, ATM call, DTE 30, ranking score > 90th pctl (ultra-selective)
  D: Top-1 sector, 5% OTM call, DTE 30, > 80th pctl (cheaper premium, more leverage)
  E: Top-1 ATM call + Bottom-1 ATM put (pairs), > 80th pctl spread
  F: Equity rotation monthly (top-2 shares) + options overlay when > 80th pctl

OPTIMIZATION: Pre-compute ALL LGBM rankings once, then run all 6 variants from cache.

Pricing: BS with 15% uplift (ATM), 20% uplift (OTM). KB #282.
Capital: $645, commission: $0 (Robinhood).
Validation: 5 gates (Sharpe>1, perm p<0.05, WR>40%, regime balance, beats random).
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Detect environment ──
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "selective_conviction_options_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "selective_conviction_options_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}, experiment={EXPERIMENT_NAME}")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ── LightGBM ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    fprint("WARNING: LightGBM not available. Will use simple momentum ranking.")

# ==================== CONFIG ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
INITIAL_CAPITAL = 645.0
RISK_FREE_RATE = 0.05

FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]


# ==================== DATA DOWNLOAD ====================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + ['SPY', '^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers from 2018-01-01...")
    raw = yf.download(all_tickers, start='2018-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()
    rename_map = {'^VIX': 'VIX'}
    close = close.rename(columns=rename_map)
    vc = 'VIX' if 'VIX' in close.columns else ('^VIX' if '^VIX' in close.columns else None)
    if vc is None:
        raise ValueError("VIX data not available")
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    return sc.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING ====================

def compute_features(px):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0
    return f


# ==================== LGBM RANKING ====================

def run_lgbm_ranking_wf(sc, idx_end):
    if not HAS_LGBM:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    start_i = max(260, idx_end - 400)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::20]

    for i in rebal_idx[:-1]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats.update({
                'date_idx': i, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].iloc[:idx_end + 1].dropna()
        feats = compute_features(px)
        if feats:
            current_feats[tk] = feats

    if not current_feats:
        return {}

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)
    return dict(zip(pred_df.index, scores))


# ==================== BLACK-SCHOLES PRICING ====================

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price_option(S, K, T, sigma, option_type='call', moneyness='atm'):
    r = RISK_FREE_RATE
    if option_type == 'call':
        bs = bs_call_price(S, K, T, r, sigma)
    else:
        bs = bs_put_price(S, K, T, r, sigma)
    uplift = 1.20 if moneyness == 'otm' else 1.15
    return bs * uplift


# ==================== PRE-COMPUTE ALL RANKINGS ====================

def precompute_all_rankings(sc, dates, start_idx, oot_start_idx):
    """Pre-compute LGBM rankings for ALL evaluation dates (warmup + OOT).

    Returns dict: {day_idx: rankings_dict}
    """
    # Identify all first-Friday indices
    eval_indices = []

    # Warmup: every 20 days from start_idx to oot_start_idx
    warmup_indices = list(range(start_idx, oot_start_idx, 20))
    eval_indices.extend(warmup_indices)

    # OOT: first Friday of each month
    for day_idx in range(oot_start_idx, len(dates)):
        today = dates[day_idx]
        if today.weekday() == 4 and today.day <= 7:
            eval_indices.append(day_idx)

    # Deduplicate and sort
    eval_indices = sorted(set(eval_indices))

    fprint(f"  Pre-computing {len(eval_indices)} LGBM rankings "
           f"({len(warmup_indices)} warmup + {len(eval_indices) - len(warmup_indices)} OOT)...")

    rankings_cache = {}
    t0 = time.time()
    for i, idx in enumerate(eval_indices):
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (len(eval_indices) - i - 1) if i > 0 else 0
            fprint(f"    Ranking {i+1}/{len(eval_indices)} "
                   f"(date={dates[idx].date()}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s)")
        rankings = run_lgbm_ranking_wf(sc, idx)
        if rankings:
            rankings_cache[idx] = rankings

    elapsed = time.time() - t0
    fprint(f"  Rankings computed: {len(rankings_cache)} in {elapsed:.0f}s")
    return rankings_cache, warmup_indices


# ==================== CONVICTION SCORING ====================

def compute_conviction_metrics(rankings, score_history):
    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    top1_ticker, top1_score = ranked[0]
    bottom1_ticker, bottom1_score = ranked[-1]
    spread = top1_score - bottom1_score

    if len(score_history) >= 5:
        top_scores = [h['top1_score'] for h in score_history]
        score_pctl = float(np.mean([1 if top1_score > s else 0 for s in top_scores])) * 100
        spreads = [h['spread'] for h in score_history]
        spread_pctl = float(np.mean([1 if spread > s else 0 for s in spreads])) * 100
    else:
        score_pctl = 50.0
        spread_pctl = 50.0

    return {
        'top1_ticker': top1_ticker,
        'top1_score': top1_score,
        'bottom1_ticker': bottom1_ticker,
        'bottom1_score': bottom1_score,
        'spread': spread,
        'score_pctl': score_pctl,
        'spread_pctl': spread_pctl,
    }


# ==================== BACKTEST ENGINE ====================

def run_variant(sc, spy, vix, variant, rankings_cache, warmup_indices, oot_start_idx, verbose=False):
    """Run a single selective conviction options backtest using pre-computed rankings."""
    dates = sc.index
    n_days = len(dates)
    start_idx = 280

    cash = INITIAL_CAPITAL
    equity_curve = []
    open_options = []
    equity_holdings = {}  # For variant F
    closed_trades = []
    n_signals = 0
    n_trades_opened = 0

    # SPY benchmark
    spy_entry_price = float(spy.iloc[oot_start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    # Build conviction history from warmup rankings
    score_history = []
    for wi in warmup_indices:
        if wi in rankings_cache:
            rankings = rankings_cache[wi]
            ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            score_history.append({
                'top1_score': ranked[0][1],
                'spread': ranked[0][1] - ranked[-1][1],
                'date_idx': wi,
            })

    last_rebalance_idx = None

    # ── Main OOT loop ──
    for day_idx in range(oot_start_idx, n_days):
        today = dates[day_idx]

        # ── Check for expiring options ──
        expired_opts = []
        for i, opt in enumerate(open_options):
            if day_idx >= opt['expiry_idx']:
                expired_opts.append(i)

        for i in sorted(expired_opts, reverse=True):
            opt = open_options.pop(i)
            S_exit = float(sc[opt['ticker']].iloc[day_idx])

            if opt['option_type'] == 'call':
                intrinsic = max(S_exit - opt['strike'], 0)
            else:
                intrinsic = max(opt['strike'] - S_exit, 0)

            pnl_per_share = intrinsic - opt['premium_per_share']
            pnl = pnl_per_share * opt['n_shares']
            cash += max(0, intrinsic * opt['n_shares'])

            closed_trades.append({
                'ticker': opt['ticker'],
                'option_type': opt['option_type'],
                'strike': opt['strike'],
                'entry_date': str(opt['entry_date'].date()),
                'expiry_date': str(today.date()),
                'entry_price': float(sc[opt['ticker']].iloc[opt['entry_idx']]),
                'exit_price': S_exit,
                'premium_paid': opt['premium_total'],
                'intrinsic_at_expiry': round(intrinsic * opt['n_shares'], 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl / opt['premium_total'] * 100, 2) if opt['premium_total'] > 0 else 0,
                'holding_days': day_idx - opt['entry_idx'],
                'variant': variant,
                'conviction_pctl': opt.get('conviction_pctl', 0),
                'move_pct': round((S_exit / opt['entry_underlying'] - 1) * 100, 2),
            })

        # ── Mark-to-market ──
        equity_mtm = 0.0
        if variant == 'F' and equity_holdings:
            for tk, pos in equity_holdings.items():
                if tk in sc.columns:
                    equity_mtm += pos['shares'] * float(sc[tk].iloc[day_idx])

        open_opt_value = 0.0
        for opt in open_options:
            S_now = float(sc[opt['ticker']].iloc[day_idx])
            if opt['option_type'] == 'call':
                iv = max(S_now - opt['strike'], 0)
            else:
                iv = max(opt['strike'] - S_now, 0)
            open_opt_value += iv * opt['n_shares']

        current_equity = cash + open_opt_value + equity_mtm
        spy_equity = spy_shares * float(spy.iloc[day_idx])

        equity_curve.append({
            'date': today,
            'equity': current_equity,
            'spy_equity': spy_equity,
            'cash': cash,
            'n_open_options': len(open_options),
        })

        # ── Only evaluate on first Friday ──
        is_friday = today.weekday() == 4
        is_first_fri = is_friday and today.day <= 7
        if not is_first_fri:
            continue

        if last_rebalance_idx is not None and (day_idx - last_rebalance_idx) < 20:
            continue

        last_rebalance_idx = day_idx
        n_signals += 1

        # ── Get cached ranking ──
        if day_idx not in rankings_cache:
            continue
        rankings = rankings_cache[day_idx]

        conv = compute_conviction_metrics(rankings, score_history)

        score_history.append({
            'top1_score': conv['top1_score'],
            'spread': conv['spread'],
            'date_idx': day_idx,
        })

        current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20.0

        if verbose and n_signals <= 5:
            fprint(f"    Sig#{n_signals} ({today.date()}): "
                   f"Top={conv['top1_ticker']} pctl={conv['score_pctl']:.0f}% "
                   f"VIX={current_vix:.1f}")

        # ── Variant F: always do equity rotation ──
        if variant == 'F':
            for tk in list(equity_holdings.keys()):
                pos = equity_holdings.pop(tk)
                sale = pos['shares'] * float(sc[tk].iloc[day_idx])
                cash += sale

            ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
            top2 = [t for t, _ in ranked[:2]]
            equity_alloc = cash * 0.80
            per_etf = equity_alloc / len(top2)
            for tk in top2:
                price = float(sc[tk].iloc[day_idx])
                shares = per_etf / price
                equity_holdings[tk] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_idx': day_idx,
                }
                cash -= shares * price

        # ── Conviction gate ──
        trade_options = False
        if variant == 'A':
            trade_options = conv['score_pctl'] >= 80
        elif variant == 'B':
            trade_options = conv['score_pctl'] >= 80 and current_vix < 25
        elif variant == 'C':
            trade_options = conv['score_pctl'] >= 90
        elif variant == 'D':
            trade_options = conv['score_pctl'] >= 80
        elif variant == 'E':
            trade_options = conv['spread_pctl'] >= 80
        elif variant == 'F':
            trade_options = conv['score_pctl'] >= 80

        if not trade_options:
            continue

        if len(open_options) >= 2:
            continue

        available_for_options = cash
        if available_for_options < 10:
            continue

        # ── Option parameters ──
        top1 = conv['top1_ticker']
        bottom1 = conv['bottom1_ticker']
        S = float(sc[top1].iloc[day_idx])

        px_hist = sc[top1].iloc[:day_idx + 1].dropna()
        rets_hist = px_hist.pct_change().dropna()
        sigma = float(rets_hist.iloc[-63:].std() * np.sqrt(252)) if len(rets_hist) >= 63 else 0.20
        sigma = max(sigma, 0.10)

        if variant in ['A', 'B', 'C', 'F']:
            K = round(S, 0)
            dte = 45 if variant == 'B' else 30
            T = dte / 365.0
            premium_per_share = price_option(S, K, T, sigma, 'call', 'atm')
            n_shares = int(available_for_options / premium_per_share) if premium_per_share > 0 else 0
            if n_shares < 1:
                continue
            premium_total = n_shares * premium_per_share
            cash -= premium_total
            n_trades_opened += 1

            open_options.append({
                'ticker': top1, 'option_type': 'call', 'strike': K,
                'premium_per_share': premium_per_share, 'premium_total': premium_total,
                'n_shares': n_shares, 'entry_date': today, 'entry_idx': day_idx,
                'entry_underlying': S, 'expiry_idx': min(day_idx + dte, n_days - 1),
                'conviction_pctl': conv['score_pctl'], 'sigma': sigma,
            })

        elif variant == 'D':
            K = round(S * 1.05, 0)
            dte = 30
            T = dte / 365.0
            premium_per_share = price_option(S, K, T, sigma, 'call', 'otm')
            n_shares = int(available_for_options / premium_per_share) if premium_per_share > 0 else 0
            if n_shares < 1:
                continue
            premium_total = n_shares * premium_per_share
            cash -= premium_total
            n_trades_opened += 1

            open_options.append({
                'ticker': top1, 'option_type': 'call', 'strike': K,
                'premium_per_share': premium_per_share, 'premium_total': premium_total,
                'n_shares': n_shares, 'entry_date': today, 'entry_idx': day_idx,
                'entry_underlying': S, 'expiry_idx': min(day_idx + dte, n_days - 1),
                'conviction_pctl': conv['score_pctl'], 'sigma': sigma,
            })

        elif variant == 'E':
            S_top = float(sc[top1].iloc[day_idx])
            S_bot = float(sc[bottom1].iloc[day_idx])
            K_top = round(S_top, 0)
            K_bot = round(S_bot, 0)
            dte = 30
            T = dte / 365.0

            px_bot = sc[bottom1].iloc[:day_idx + 1].dropna()
            rets_bot = px_bot.pct_change().dropna()
            sigma_bot = float(rets_bot.iloc[-63:].std() * np.sqrt(252)) if len(rets_bot) >= 63 else 0.20
            sigma_bot = max(sigma_bot, 0.10)

            prem_call = price_option(S_top, K_top, T, sigma, 'call', 'atm')
            prem_put = price_option(S_bot, K_bot, T, sigma_bot, 'put', 'atm')

            if prem_call <= 0 or prem_put <= 0:
                continue

            half_cap = available_for_options / 2
            n_shares_call = int(half_cap / prem_call)
            n_shares_put = int(half_cap / prem_put)

            if n_shares_call < 1 or n_shares_put < 1:
                continue

            total_premium = n_shares_call * prem_call + n_shares_put * prem_put
            cash -= total_premium
            n_trades_opened += 2

            open_options.append({
                'ticker': top1, 'option_type': 'call', 'strike': K_top,
                'premium_per_share': prem_call, 'premium_total': n_shares_call * prem_call,
                'n_shares': n_shares_call, 'entry_date': today, 'entry_idx': day_idx,
                'entry_underlying': S_top, 'expiry_idx': min(day_idx + dte, n_days - 1),
                'conviction_pctl': conv['spread_pctl'], 'sigma': sigma,
            })
            open_options.append({
                'ticker': bottom1, 'option_type': 'put', 'strike': K_bot,
                'premium_per_share': prem_put, 'premium_total': n_shares_put * prem_put,
                'n_shares': n_shares_put, 'entry_date': today, 'entry_idx': day_idx,
                'entry_underlying': S_bot, 'expiry_idx': min(day_idx + dte, n_days - 1),
                'conviction_pctl': conv['spread_pctl'], 'sigma': sigma_bot,
            })

    # ── Close remaining at end ──
    for opt in open_options:
        S_exit = float(sc[opt['ticker']].iloc[-1])
        if opt['option_type'] == 'call':
            intrinsic = max(S_exit - opt['strike'], 0)
        else:
            intrinsic = max(opt['strike'] - S_exit, 0)

        pnl_per_share = intrinsic - opt['premium_per_share']
        pnl = pnl_per_share * opt['n_shares']
        cash += max(0, intrinsic * opt['n_shares'])

        closed_trades.append({
            'ticker': opt['ticker'], 'option_type': opt['option_type'],
            'strike': opt['strike'],
            'entry_date': str(opt['entry_date'].date()),
            'expiry_date': str(dates[-1].date()),
            'entry_price': float(sc[opt['ticker']].iloc[opt['entry_idx']]),
            'exit_price': S_exit,
            'premium_paid': opt['premium_total'],
            'intrinsic_at_expiry': round(intrinsic * opt['n_shares'], 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / opt['premium_total'] * 100, 2) if opt['premium_total'] > 0 else 0,
            'holding_days': len(dates) - 1 - opt['entry_idx'],
            'variant': variant,
            'conviction_pctl': opt.get('conviction_pctl', 0),
            'move_pct': round((S_exit / opt['entry_underlying'] - 1) * 100, 2),
        })

    if variant == 'F' and equity_holdings:
        for tk in list(equity_holdings.keys()):
            pos = equity_holdings.pop(tk)
            cash += pos['shares'] * float(sc[tk].iloc[-1])

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) > 0:
        eq_df = eq_df.set_index('date')
        eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

    metrics = compute_metrics(closed_trades, eq_df, variant, n_signals)

    return {
        'trades': closed_trades,
        'equity_curve': eq_df,
        'metrics': metrics,
        'variant': variant,
        'n_signals': n_signals,
        'n_trades': n_trades_opened,
    }


# ==================== METRICS ====================

def compute_metrics(trades, eq_df, variant_name, n_signals):
    if not trades:
        return {
            'variant': variant_name, 'n_trades': 0, 'n_signals': n_signals,
            'selectivity': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0, 'cagr': 0,
            'total_pnl': 0, 'mean_pnl': 0, 'wins': 0, 'losses': 0,
            'spy_total_return': 0, 'spy_sharpe': 0, 'alpha_vs_spy': 0,
            'avg_premium': 0, 'avg_pnl_pct': 0,
        }

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0
    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls)

    if len(eq_df) > 5:
        daily_rets = eq_df['equity'].pct_change().dropna()
        daily_rets = daily_rets.replace([np.inf, -np.inf], 0).fillna(0)
        sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-10) * np.sqrt(252))
        downside_rets = daily_rets[daily_rets < 0]
        if len(downside_rets) > 1 and downside_rets.std() > 0:
            sortino = float(daily_rets.mean() / downside_rets.std() * np.sqrt(252))
        else:
            sortino = sharpe
    else:
        sharpe = 0.0
        sortino = 0.0

    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / (gross_losses + 1e-10)

    if len(eq_df) > 0:
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak
        mdd = float(dd.min())
    else:
        mdd = 0

    total_return = total_pnl / INITIAL_CAPITAL

    if len(eq_df) > 1:
        n_days_bt = (eq_df.index[-1] - eq_df.index[0]).days
        years = n_days_bt / 365.25
        if years > 0 and (1 + total_return) > 0:
            cagr = (1 + total_return) ** (1.0 / years) - 1
        else:
            cagr = 0
    else:
        cagr = 0

    spy_total_return = 0
    spy_sharpe = 0
    alpha_vs_spy = 0
    if 'spy_equity' in eq_df.columns and len(eq_df) > 5:
        spy_total_return = float(eq_df['spy_equity'].iloc[-1] / eq_df['spy_equity'].iloc[0] - 1)
        spy_daily_rets = eq_df['spy_equity'].pct_change().dropna()
        spy_daily_rets = spy_daily_rets.replace([np.inf, -np.inf], 0).fillna(0)
        if spy_daily_rets.std() > 0:
            spy_sharpe = float(spy_daily_rets.mean() / spy_daily_rets.std() * np.sqrt(252))
        alpha_vs_spy = total_return - spy_total_return

    avg_premium = np.mean([t['premium_paid'] for t in trades]) if trades else 0
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) if trades else 0
    selectivity = n / n_signals * 100 if n_signals > 0 else 0

    return {
        'variant': variant_name, 'n_trades': n, 'n_signals': n_signals,
        'selectivity': round(selectivity, 1),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'pf': round(pf, 3), 'wr': round(wr * 100, 1),
        'mdd': round(mdd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2),
        'total_pnl': round(total_pnl, 2),
        'mean_pnl': round(mean_pnl, 2),
        'wins': wins, 'losses': n - wins,
        'spy_total_return': round(spy_total_return * 100, 2),
        'spy_sharpe': round(spy_sharpe, 3),
        'alpha_vs_spy': round(alpha_vs_spy * 100, 2),
        'avg_premium': round(avg_premium, 2),
        'avg_pnl_pct': round(avg_pnl_pct, 1),
    }


# ==================== REGIME BREAKDOWN ====================

def compute_regime_breakdown(trades, spy):
    if not trades or len(spy) < 2:
        return {}

    regime_trades = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        exit_date = pd.Timestamp(t['expiry_date'])
        mask = (spy.index >= entry_date) & (spy.index <= exit_date)
        period_spy = spy[mask]
        if len(period_spy) >= 2:
            period_ret = float(period_spy.iloc[-1] / period_spy.iloc[0] - 1)
        else:
            period_ret = 0.0

        if period_ret > 0.005:
            regime_trades['green'].append(t)
        elif period_ret < -0.005:
            regime_trades['red'].append(t)
        else:
            regime_trades['flat'].append(t)

    breakdown = {}
    for regime, rtrades in regime_trades.items():
        if not rtrades:
            breakdown[regime] = {'n': 0, 'sharpe': 0, 'wr': 0, 'mean_pnl': 0, 'total_pnl': 0}
            continue
        pnls = [t['pnl'] for t in rtrades]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if n > 1 else 1.0
        trades_per_year = max(n / 4.5, 1.0)
        sharpe = (mean_pnl / (std_pnl + 1e-10)) * np.sqrt(trades_per_year) if std_pnl > 0 else 0
        breakdown[regime] = {
            'n': n, 'sharpe': round(sharpe, 2),
            'wr': round(wins / n * 100, 1) if n > 0 else 0,
            'mean_pnl': round(mean_pnl, 2),
            'total_pnl': round(sum(pnls), 2),
        }
    return breakdown


# ==================== 5-GATE VALIDATION ====================

def five_gate_validation(result, spy):
    metrics = result['metrics']
    trades = result['trades']
    gates = {}

    # Gate 1: Sharpe > 1
    gates['sharpe_gt_1'] = {
        'pass': metrics['sharpe'] > 1.0,
        'value': metrics['sharpe'],
        'threshold': 1.0,
    }

    # Gate 2: Permutation test (sign shuffle, 100 shuffles)
    if len(trades) >= 3:
        all_pnls = np.array([t['pnl'] for t in trades])
        actual_mean = np.mean(all_pnls)
        n_perm = 100
        rng = np.random.RandomState(42)
        perm_means = np.zeros(n_perm)
        for i in range(n_perm):
            signs = rng.choice([-1, 1], size=len(all_pnls))
            perm_means[i] = np.mean(all_pnls * signs)
        p_value = float(np.mean(perm_means >= actual_mean)) if actual_mean > 0 else 1.0
    else:
        p_value = 1.0

    gates['perm_p_lt_005'] = {
        'pass': p_value < 0.05,
        'value': round(p_value, 4),
        'threshold': 0.05,
    }

    # Gate 3: WR > 40%
    gates['wr_gt_40'] = {
        'pass': metrics['wr'] > 40.0,
        'value': metrics['wr'],
        'threshold': 40.0,
    }

    # Gate 4: Regime balance
    breakdown = compute_regime_breakdown(trades, spy)
    sharpe_green = breakdown.get('green', {}).get('sharpe', 0)
    sharpe_red = breakdown.get('red', {}).get('sharpe', 0)
    max_s = max(abs(sharpe_green), abs(sharpe_red), 0.01)
    regime_skew = abs(sharpe_green - sharpe_red) / max_s
    gates['regime_balance'] = {
        'pass': regime_skew <= 0.50,
        'value': round(regime_skew, 3),
        'threshold': 0.50,
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
    }

    # Gate 5: Beats random baseline
    if len(trades) >= 3:
        actual_total = sum(t['pnl'] for t in trades)
        rng = np.random.RandomState(99)
        n_sims = 100
        all_pnls = np.array([t['pnl'] for t in trades])
        random_totals = np.zeros(n_sims)
        for i in range(n_sims):
            idx = rng.choice(len(all_pnls), size=len(all_pnls), replace=True)
            random_totals[i] = np.sum(all_pnls[idx])
        beats_random = actual_total > np.median(random_totals)
        pct_beaten = float(np.mean(random_totals < actual_total)) * 100
    else:
        beats_random = False
        pct_beaten = 0

    gates['beats_random'] = {
        'pass': beats_random,
        'value': round(pct_beaten, 1),
        'threshold': 50.0,
    }

    n_pass = sum(1 for g in gates.values() if g['pass'])
    return {
        'gates': gates,
        'n_pass': n_pass,
        'n_total': len(gates),
        'all_pass': n_pass == len(gates),
        'regime_breakdown': breakdown,
    }


# ==================== MAIN ====================

def main():
    t0 = time.time()

    fprint("=" * 80)
    fprint("  SELECTIVE CONVICTION OPTIONS V1")
    fprint("  Hypothesis: Trade options ONLY when LGBM ranking conviction is exceptional")
    fprint("  Target: 2-8 trades/year instead of 12. Selectivity overcomes theta.")
    fprint("=" * 80)

    fprint("\nDownloading data...")
    sc, spy, vix = download_data()
    fprint(f"Data: {sc.index[0].date()} to {sc.index[-1].date()}, "
           f"{len(sc)} days, {len(sc.columns)} sectors")

    oot_start = pd.Timestamp('2021-01-01')
    oot_start_idx = 280
    dates = sc.index
    for i in range(280, len(dates)):
        if dates[i] >= oot_start:
            oot_start_idx = i
            break

    oot_days = len(sc[sc.index >= oot_start])
    fprint(f"OOT period: 2021-01-01 to {sc.index[-1].date()} ({oot_days} trading days)")
    fprint(f"Train warmup: idx 0-{oot_start_idx} ({oot_start_idx} days for features + conviction history)")

    if oot_days < 40:
        fprint(f"WARNING: Only {oot_days} OOT days (need >= 40)")

    # ── Pre-compute ALL rankings (the expensive step, done ONCE) ──
    fprint("\n" + "=" * 80)
    fprint("  PHASE 1: PRE-COMPUTING ALL LGBM RANKINGS")
    fprint("=" * 80)
    rankings_cache, warmup_indices = precompute_all_rankings(sc, dates, 280, oot_start_idx)
    rank_time = time.time() - t0
    fprint(f"  Ranking phase complete in {rank_time:.0f}s")

    # ── Run all 6 variants (fast, using cached rankings) ──
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'ATM Call DTE30 >80pctl',
        'B': 'ATM Call DTE45 >80p+VIX<25',
        'C': 'ATM Call DTE30 >90pctl',
        'D': '5%OTM Call DTE30 >80pctl',
        'E': 'Call+Put Pairs >80p spread',
        'F': 'Equity+Options Overlay',
    }

    results = {}
    validations = {}

    fprint("\n" + "=" * 80)
    fprint("  PHASE 2: RUNNING 6 VARIANTS (using cached rankings)")
    fprint("=" * 80)

    for v in variants:
        fprint(f"\n{'='*70}")
        fprint(f"  VARIANT {v}: {variant_names[v]}")
        fprint(f"{'='*70}")
        t_start = time.time()
        results[v] = run_variant(sc, spy, vix, v, rankings_cache, warmup_indices, oot_start_idx, verbose=True)
        elapsed_v = time.time() - t_start
        m = results[v]['metrics']
        fprint(f"  Signals: {m['n_signals']} | Trades: {m['n_trades']} | "
               f"Selectivity: {m['selectivity']:.1f}%")
        fprint(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | "
               f"PF: {m['pf']:.3f} | WR: {m['wr']:.1f}%")
        fprint(f"  MDD: {m['mdd']:.2f}% | Return: {m['total_return']:.2f}% | "
               f"CAGR: {m['cagr']:.2f}% | PnL: ${m['total_pnl']:.2f}")
        fprint(f"  Avg prem: ${m['avg_premium']:.2f} | Avg PnL%: {m['avg_pnl_pct']:.1f}%")
        fprint(f"  Runtime: {elapsed_v:.1f}s")

        val = five_gate_validation(results[v], spy)
        validations[v] = val
        fprint(f"  5-Gate: {val['n_pass']}/{val['n_total']} PASS")
        for name, gate in val['gates'].items():
            status = "PASS" if gate['pass'] else "FAIL"
            fprint(f"    {name}: {status} (val={gate['value']}, thr={gate['threshold']})")

    # ==================== SUMMARY TABLE ====================
    fprint("\n" + "=" * 130)
    fprint("  COMPARISON SUMMARY")
    fprint("=" * 130)
    fprint(f"{'Variant':<30} {'Sigs':>4} {'Trd':>4} {'Sel%':>5} {'Sharpe':>7} {'Sortino':>8} "
           f"{'PF':>6} {'WR%':>6} {'MDD%':>7} {'Ret%':>8} {'CAGR%':>7} {'PnL$':>8} {'AvgPnl%':>8} {'Gates':>6}")
    fprint("-" * 130)

    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        fprint(f"{v}: {variant_names[v]:<27} {m['n_signals']:>3} {m['n_trades']:>4} "
               f"{m['selectivity']:>5.1f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
               f"{m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>8.2f} {m['cagr']:>7.2f} {m['total_pnl']:>8.2f} "
               f"{m['avg_pnl_pct']:>8.1f} {val['n_pass']:>2}/{val['n_total']}")

    spy_m = results['F']['metrics']
    fprint(f"{'SPY Buy-Hold':<33} {'':>4} {'':>4} {'':>5} {spy_m['spy_sharpe']:>7.3f} "
           f"{'':>8} {'':>6} {'':>6} {'':>7} {spy_m['spy_total_return']:>8.2f}")

    # ── Trade details ──
    fprint("\n" + "=" * 130)
    fprint("  TRADE DETAILS")
    fprint("=" * 130)

    for v in variants:
        trades = results[v]['trades']
        if not trades:
            fprint(f"\n  Variant {v}: NO TRADES")
            continue
        fprint(f"\n  Variant {v} ({variant_names[v]}) -- {len(trades)} trades:")
        fprint(f"    {'Date':<12} {'Type':<5} {'Ticker':<6} {'K':>7} {'Prem$':>8} "
               f"{'Entry$':>8} {'Exit$':>8} {'Move%':>7} {'PnL$':>8} {'PnL%':>8} {'Conv%':>6}")
        for t in trades:
            fprint(f"    {t['entry_date']:<12} {t['option_type']:<5} {t['ticker']:<6} "
                   f"{t['strike']:>7.0f} {t['premium_paid']:>8.2f} "
                   f"{t['entry_price']:>8.2f} {t['exit_price']:>8.2f} "
                   f"{t['move_pct']:>7.2f} {t['pnl']:>8.2f} {t['pnl_pct']:>8.1f} "
                   f"{t['conviction_pctl']:>6.0f}")

    # ── Regime breakdowns ──
    fprint("\n" + "=" * 130)
    fprint("  REGIME BREAKDOWNS")
    fprint("=" * 130)

    for v in variants:
        bd = validations[v]['regime_breakdown']
        fprint(f"\n  Variant {v} ({variant_names[v]}):")
        for regime in ['green', 'red', 'flat']:
            b = bd.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean=${b.get('mean_pnl',0):8.2f}  "
                   f"total=${b.get('total_pnl',0):8.2f}")

    # ── Alpha vs SPY ──
    fprint("\n" + "=" * 130)
    fprint("  ALPHA vs SPY")
    fprint("=" * 130)
    for v in variants:
        m = results[v]['metrics']
        fprint(f"  {v}: {variant_names[v]:<30} Ret={m['total_return']:>7.2f}%  "
               f"SPY={m['spy_total_return']:>7.2f}%  Alpha={m['alpha_vs_spy']:>7.2f}%")

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    fprint(f"\n  BEST VARIANT: {best_v} ({variant_names[best_v]}) -- Sharpe {best_m['sharpe']:.3f}")

    # ── Key conclusions ──
    fprint("\n" + "=" * 130)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 130)

    any_pass_all = any(validations[v]['all_pass'] for v in variants)

    for v in variants:
        m = results[v]['metrics']
        if m['n_trades'] > 0:
            years_oot = oot_days / 252.0
            trades_per_year = m['n_trades'] / years_oot if years_oot > 0 else 0
            fprint(f"  {v}: {m['n_trades']} trades / {years_oot:.1f}y = {trades_per_year:.1f}/yr "
                   f"(target 2-8/yr)")

    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"\n  VERDICT: Selective conviction options WORKS!")
        fprint(f"  Passing all 5 gates: {', '.join(f'{v}' for v in passing)}")
        fprint(f"  Selectivity hypothesis CONFIRMED.")
    elif best_m['sharpe'] > 0.5:
        fprint(f"\n  VERDICT: Some edge but does not pass all gates.")
        fprint(f"  Best Sharpe: {best_m['sharpe']:.3f}")
    elif best_m['sharpe'] > 0:
        fprint(f"\n  VERDICT: Weak positive edge, not tradeable.")
        fprint(f"  Theta still dominates. Stick with equity rotation.")
    else:
        fprint(f"\n  VERDICT: Selective conviction options FAILS.")
        fprint(f"  Even with selectivity, options negative EV at $645 scale.")
        fprint(f"  Equity rotation remains the correct approach.")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{sc.index[0].date()} to {sc.index[-1].date()}",
        'oot_period': f"2021-01-01 to {sc.index[-1].date()}",
        'oot_days': oot_days,
        'initial_capital': INITIAL_CAPITAL,
        'commission': 0.0,
        'hypothesis': 'Trade options ONLY when LGBM conviction > 80th/90th percentile',
        'pricing': 'BS + 15% uplift ATM, 20% uplift OTM (KB #282)',
        'metrics': {},
        'validations': {},
        'regime_breakdowns': {},
    }

    for v in variants:
        save_results['metrics'][v] = results[v]['metrics']
        val_clean = {
            'n_pass': validations[v]['n_pass'],
            'n_total': validations[v]['n_total'],
            'all_pass': validations[v]['all_pass'],
            'gates': {},
        }
        for gname, gval in validations[v]['gates'].items():
            val_clean['gates'][gname] = {k: v2 for k, v2 in gval.items()}
        save_results['validations'][v] = val_clean
        save_results['regime_breakdowns'][v] = validations[v]['regime_breakdown']

    results_path = OUTPUT_DIR / "backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    for v in variants:
        trades_path = OUTPUT_DIR / f"trades_variant_{v}.json"
        with open(trades_path, 'w') as f:
            json.dump(results[v]['trades'], f, indent=2, default=str)

    for v in variants:
        eq_path = OUTPUT_DIR / f"equity_curve_{v}.csv"
        results[v]['equity_curve'].to_csv(eq_path)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            run_name = f"selective_conviction_opts_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("n_oot_days", oot_days)
                mlflow.log_param("oot_period", f"2021-01-01 to {sc.index[-1].date()}")
                mlflow.log_param("pricing", "BS+15%ATM/+20%OTM")
                mlflow.log_param("best_variant", f"{best_v}_{variant_names[best_v]}")

                for v in variants:
                    m = results[v]['metrics']
                    val = validations[v]
                    prefix = f"v{v}_"
                    mlflow.log_metric(f"{prefix}sharpe", m['sharpe'])
                    mlflow.log_metric(f"{prefix}sortino", m['sortino'])
                    mlflow.log_metric(f"{prefix}pf", m['pf'])
                    mlflow.log_metric(f"{prefix}wr", m['wr'])
                    mlflow.log_metric(f"{prefix}mdd", m['mdd'])
                    mlflow.log_metric(f"{prefix}total_return", m['total_return'])
                    mlflow.log_metric(f"{prefix}cagr", m['cagr'])
                    mlflow.log_metric(f"{prefix}n_trades", m['n_trades'])
                    mlflow.log_metric(f"{prefix}selectivity", m['selectivity'])
                    mlflow.log_metric(f"{prefix}alpha_vs_spy", m['alpha_vs_spy'])
                    mlflow.log_metric(f"{prefix}gates_pass", val['n_pass'])

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
