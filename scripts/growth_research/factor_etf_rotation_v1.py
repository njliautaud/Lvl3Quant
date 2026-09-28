#!/usr/bin/env python3
"""
Factor ETF Rotation V1 (Rebuilt 2026-07-27)
=============================================

HYPOTHESIS: Rotating across FACTOR ETFs (momentum, value, quality, size, min-vol,
FCF, dividend quality) using LGBM ranking may outperform sector ETF rotation
(Sharpe 1.40 baseline from KB #285).

UNIVERSE (10 Factor ETFs + 2 Benchmarks):
  MTUM  — iShares MSCI USA Momentum
  VLUE  — iShares MSCI USA Value
  QUAL  — iShares MSCI USA Quality
  SIZE  — iShares MSCI USA Size
  USMV  — iShares MSCI USA Min Volatility
  VTV   — Vanguard Value
  VUG   — Vanguard Growth
  MOAT  — VanEck Morningstar Wide Moat
  COWZ  — Pacer US Cash Cows 100 (FCF factor)
  NOBL  — ProShares S&P 500 Dividend Aristocrats
  SPY, QQQ — benchmarks

SIX VARIANTS:
  A: Long Top-2 factors, monthly rebalance, $645 capital
  B: Long Top-1 factor only (concentrated)
  C: Long Top-2, biweekly rebalance (faster rotation)
  D: Long Top-2 with 15% trailing stop
  E: Long Top-2 + Short Bottom-1 (hedged)
  F: Long Top-2 + ATM call on #1 pick when score > 80th percentile

FEATURES: 21 (17 momentum + 4 enhanced cross-asset)
VALIDATION: 5 gates (Sharpe>1, perm p<0.05, WR>40%, regime balance <0.50, MC CI>0)
Capital: $645, zero-commission (Robinhood), data from 2020-01-01, OOT from 2021-01-01

Output: output/growth_research/factor_etf_rotation_v1/
MLflow experiment: factor_etf_rotation_v1
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

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Detect environment ──
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "factor_etf_rotation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "factor_etf_rotation_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ── LightGBM ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    fprint("WARNING: LightGBM not available, falling back to momentum ranking.")


# ==================== CONFIG ====================

FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'SIZE', 'USMV', 'VTV', 'VUG', 'MOAT', 'COWZ', 'NOBL']
BENCHMARKS = ['SPY', 'QQQ']
INITIAL_CAPITAL = 645.0

# 21 features: 17 momentum + 4 enhanced
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
    # 4 enhanced cross-asset features
    'corr_to_spy_63d', 'beta_to_spy_63d', 'rel_strength_vs_mean', 'rank_21d',
]

DATA_START = '2020-01-01'
OOT_START = '2021-01-01'


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download factor ETF + benchmark + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = FACTOR_ETFS + BENCHMARKS + ['^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers from {DATA_START}...")
    raw = yf.download(all_tickers, start=DATA_START, progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill().rename(columns={'^VIX': 'VIX'})

    vix = close['VIX'].dropna()
    spy = close['SPY'].dropna()
    qqq = close['QQQ'].dropna() if 'QQQ' in close.columns else spy.copy()

    available_etfs = [c for c in FACTOR_ETFS if c in close.columns]
    missing = [c for c in FACTOR_ETFS if c not in close.columns]
    if missing:
        fprint(f"WARNING: Missing ETFs: {missing}")

    fc = close[available_etfs].dropna(how='all')
    ix = fc.index.intersection(spy.index).intersection(vix.index)
    fprint(f"Data: {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}, "
           f"{len(ix)} days, {len(available_etfs)} factor ETFs")
    fprint(f"Factor ETFs: {available_etfs}")
    return fc.loc[ix], spy.loc[ix], vix.loc[ix], qqq.loc[ix] if 'QQQ' in close.columns else spy.loc[ix]


# ==================== FEATURE ENGINEERING (21 features) ====================

def compute_features(px, spy_px=None, all_rets_21d=None):
    """Compute 21 momentum + cross-asset features for a single ETF."""
    if len(px) < 260:
        return None
    f = {}
    # 6 return features
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    # Volatility features
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    # Risk-adjusted features
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

    # Trend features
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    # ── 4 Enhanced cross-asset features ──
    if spy_px is not None and len(spy_px) >= 63:
        spy_rets = spy_px.pct_change().dropna()
        # Align lengths
        min_len = min(len(rets), len(spy_rets))
        etf_r = rets.iloc[-min_len:]
        spy_r = spy_rets.iloc[-min_len:]
        # Correlation to SPY (63d)
        if min_len >= 63:
            f['corr_to_spy_63d'] = float(etf_r.iloc[-63:].corr(spy_r.iloc[-63:]))
            # Beta to SPY (63d)
            cov = np.cov(etf_r.iloc[-63:].values, spy_r.iloc[-63:].values)
            f['beta_to_spy_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f['corr_to_spy_63d'] = 0.0
            f['beta_to_spy_63d'] = 1.0
    else:
        f['corr_to_spy_63d'] = 0.0
        f['beta_to_spy_63d'] = 1.0

    # Relative strength vs universe mean
    if all_rets_21d is not None:
        mean_ret = np.mean(list(all_rets_21d.values()))
        f['rel_strength_vs_mean'] = f['ret_21d'] - mean_ret
    else:
        f['rel_strength_vs_mean'] = 0.0

    # Rank of 21d return within universe (will be overwritten with actual cross-sectional rank)
    f['rank_21d'] = 0.5  # placeholder, will be filled after all ETFs computed

    return f


# ==================== LGBM RANKING (walk-forward, sliding window) ====================

def run_lgbm_ranking_wf(fc, spy, idx_end, train_window=400, sample_every=20):
    """Walk-forward LGBM ranking using sliding window.

    Train on samples from [idx_end - train_window, idx_end) sampled every 20 days.
    Predict forward 28-day returns.
    """
    if not HAS_LGBM:
        # Fallback: simple 21-day momentum ranking
        rets_21d = fc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False)), None

    records = []
    start_i = max(260, idx_end - train_window)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::sample_every]

    spy_px = spy.iloc[:idx_end + 1]

    for i in rebal_idx[:-1]:
        # Compute 21d returns for all ETFs at this point for cross-sectional features
        rets_21d_all = {}
        for tk in fc.columns:
            px_tk = fc[tk].iloc[:i + 1].dropna()
            if len(px_tk) > 21:
                rets_21d_all[tk] = float(px_tk.iloc[-1] / px_tk.iloc[-21] - 1)

        for tk in fc.columns:
            px = fc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px, spy_px=spy_px.iloc[:i + 1], all_rets_21d=rets_21d_all)
            if not feats:
                continue
            # Cross-sectional rank
            if rets_21d_all:
                sorted_rets = sorted(rets_21d_all.values())
                n_etfs = len(sorted_rets)
                if tk in rets_21d_all and n_etfs > 1:
                    feats['rank_21d'] = float(sorted_rets.index(rets_21d_all[tk])) / (n_etfs - 1)

            fi = min(i + 28, len(fc) - 1)
            feats['date_idx'] = i
            feats['ticker'] = tk
            feats['fwd_ret'] = float(fc[tk].iloc[fi] / fc[tk].iloc[i] - 1)
            records.append(feats)

    if len(records) < 50:
        rets_21d = fc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False)), None

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    # Predict current ranks
    rets_21d_all = {}
    for tk in fc.columns:
        px_tk = fc[tk].iloc[:idx_end + 1].dropna()
        if len(px_tk) > 21:
            rets_21d_all[tk] = float(px_tk.iloc[-1] / px_tk.iloc[-21] - 1)

    current_feats = {}
    for tk in fc.columns:
        px = fc[tk].iloc[:idx_end + 1].dropna()
        feats = compute_features(px, spy_px=spy.iloc[:idx_end + 1], all_rets_21d=rets_21d_all)
        if feats:
            if rets_21d_all:
                sorted_rets = sorted(rets_21d_all.values())
                n_etfs = len(sorted_rets)
                if tk in rets_21d_all and n_etfs > 1:
                    feats['rank_21d'] = float(sorted_rets.index(rets_21d_all[tk])) / (n_etfs - 1)
            current_feats[tk] = feats

    if not current_feats:
        return {}, None

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)
    return dict(zip(pred_df.index, scores)), m


# ==================== SIMPLE BS CALL PRICER (for Variant F) ====================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price. Used for variant F's options overlay."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


# ==================== BACKTEST ENGINE ====================

def run_variant(fc, spy, vix, qqq, variant, verbose=False):
    """Run a single factor ETF rotation backtest variant.

    A: Long Top-2, monthly rebalance
    B: Long Top-1 (concentrated), monthly
    C: Long Top-2, biweekly rebalance
    D: Long Top-2 with 15% trailing stop
    E: Long Top-2 + Short Bottom-1 (hedged)
    F: Long Top-2 + ATM call on #1 when score > 80th pctl
    """
    dates = fc.index
    n_days = len(dates)
    start_idx = 280  # Need 260 for features + warm-up

    # Rebalance interval
    if variant == 'C':
        rebal_days = 14  # Biweekly
    else:
        rebal_days = 28  # Monthly

    equity = INITIAL_CAPITAL
    equity_curve = []
    holdings = {}  # ticker -> {shares, entry_price, entry_date, entry_idx, side, peak_price}
    options_positions = []  # For variant F: {ticker, entry_date, premium, strike, expiry_idx, vol}
    closed_trades = []
    last_rebalance_idx = None
    n_rebalances = 0
    options_pnl = 0.0  # Track options P&L separately

    # Benchmark tracking
    spy_entry_price = float(spy.iloc[start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # ── Trailing stop check (Variant D) ──
        if variant == 'D':
            stops_triggered = []
            for tk, pos in holdings.items():
                if pos['side'] != 'long':
                    continue
                current_price = float(fc[tk].iloc[day_idx])
                if current_price > pos.get('peak_price', pos['entry_price']):
                    pos['peak_price'] = current_price
                peak = pos.get('peak_price', pos['entry_price'])
                drawdown = (peak - current_price) / peak
                if drawdown >= 0.15:  # 15% trailing stop
                    stops_triggered.append(tk)

            for tk in stops_triggered:
                pos = holdings.pop(tk)
                exit_price = float(fc[tk].iloc[day_idx])
                pnl = pos['shares'] * (exit_price - pos['entry_price'])
                closed_trades.append({
                    'ticker': tk, 'side': 'long', 'exit_reason': 'trailing_stop',
                    'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                    'exit_date': str(today.date()),
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'shares': pos['shares'], 'pnl': round(pnl, 4),
                    'holding_days': day_idx - pos['entry_idx'],
                })

        # ── Expire options (Variant F) ──
        if variant == 'F' and options_positions:
            expired = []
            for i, opt in enumerate(options_positions):
                if day_idx >= opt['expiry_idx']:
                    # Calculate intrinsic value at expiry
                    if opt['ticker'] in fc.columns:
                        spot = float(fc[opt['ticker']].iloc[min(opt['expiry_idx'], n_days - 1)])
                        intrinsic = max(spot - opt['strike'], 0)
                        opt_pnl = intrinsic - opt['premium']
                    else:
                        opt_pnl = -opt['premium']
                    options_pnl += opt_pnl
                    closed_trades.append({
                        'ticker': opt['ticker'] + '_CALL', 'side': 'long',
                        'exit_reason': 'option_expiry',
                        'entry_date': str(opt['entry_date']),
                        'exit_date': str(today.date()),
                        'entry_price': opt['premium'], 'exit_price': max(0, intrinsic),
                        'shares': 1, 'pnl': round(opt_pnl, 4),
                        'holding_days': day_idx - opt['entry_idx'],
                    })
                    expired.append(i)
            for i in sorted(expired, reverse=True):
                options_positions.pop(i)

        # ── Mark-to-market ──
        mtm_pnl = 0.0
        for tk, pos in holdings.items():
            if tk not in fc.columns:
                continue
            current_price = float(fc[tk].iloc[day_idx])
            if pos['side'] == 'long':
                mtm_pnl += pos['shares'] * (current_price - pos['entry_price'])
            else:
                mtm_pnl += pos['shares'] * (pos['entry_price'] - current_price)

        realized_pnl = sum(t['pnl'] for t in closed_trades)
        current_equity = INITIAL_CAPITAL + mtm_pnl + realized_pnl

        spy_equity = spy_shares * float(spy.iloc[day_idx])
        equity_curve.append({
            'date': today,
            'equity': current_equity,
            'spy_equity': spy_equity,
            'n_holdings': len(holdings),
        })

        # ── Check for rebalance ──
        is_friday = today.weekday() == 4
        if not is_friday:
            continue

        do_rebalance = False
        if last_rebalance_idx is None:
            do_rebalance = True
        elif today.day <= 7:  # First Friday of month
            do_rebalance = True
        elif day_idx - last_rebalance_idx >= rebal_days:
            do_rebalance = True

        if not do_rebalance:
            continue

        last_rebalance_idx = day_idx
        n_rebalances += 1

        # ── Get LGBM ranking ──
        rankings, model = run_lgbm_ranking_wf(fc, spy, day_idx)
        if not rankings:
            continue

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        scores = [s for _, s in ranked]

        # ── Determine target portfolio ──
        target_longs = {}
        target_shorts = {}

        if variant in ('A', 'D'):
            # Long Top-2, equal weight
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for tk in top2:
                target_longs[tk] = w

        elif variant == 'B':
            # Long Top-1 (concentrated)
            top1 = ranked[0][0]
            target_longs[top1] = 1.0

        elif variant == 'C':
            # Long Top-2, biweekly (same allocation, different rebal freq)
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for tk in top2:
                target_longs[tk] = w

        elif variant == 'E':
            # Long Top-2 + Short Bottom-1 (hedged)
            top2 = [t for t, _ in ranked[:2]]
            bot1 = ranked[-1][0]
            w_long = 0.40  # 40% each long
            w_short = 0.20  # 20% short
            for tk in top2:
                target_longs[tk] = w_long
            target_shorts[bot1] = w_short

        elif variant == 'F':
            # Long Top-2 + ATM call on #1 when score > 80th percentile
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for tk in top2:
                target_longs[tk] = w

            # Options overlay: buy ATM call on #1 if score > 80th pctl
            top1_ticker = ranked[0][0]
            top1_score = ranked[0][1]
            score_80th = np.percentile(scores, 80) if len(scores) > 2 else 999
            if top1_score > score_80th and top1_ticker in fc.columns:
                spot = float(fc[top1_ticker].iloc[day_idx])
                # Use realized vol for BS pricing
                tk_rets = fc[top1_ticker].iloc[max(0, day_idx-63):day_idx+1].pct_change().dropna()
                vol = float(tk_rets.std() * np.sqrt(252)) if len(tk_rets) > 10 else 0.20
                # ATM call, 28-day expiry
                T = 28.0 / 365.0
                premium = bs_call_price(spot, spot, T, 0.05, vol)
                # Allocate up to 5% of equity to option premium
                max_option_spend = current_equity * 0.05
                if premium > 0 and premium < max_option_spend:
                    options_positions.append({
                        'ticker': top1_ticker,
                        'entry_date': str(today.date()),
                        'entry_idx': day_idx,
                        'premium': round(premium, 4),
                        'strike': spot,
                        'expiry_idx': day_idx + 20,  # ~28 calendar = ~20 trading days
                        'vol': vol,
                    })

        # ── Close positions not in target ──
        tickers_to_close = []
        for tk, pos in holdings.items():
            in_target = (tk in target_longs and pos['side'] == 'long') or \
                        (tk in target_shorts and pos['side'] == 'short')
            if not in_target:
                tickers_to_close.append(tk)

        for tk in tickers_to_close:
            pos = holdings.pop(tk)
            current_price = float(fc[tk].iloc[day_idx])
            if pos['side'] == 'long':
                pnl = pos['shares'] * (current_price - pos['entry_price'])
            else:
                pnl = pos['shares'] * (pos['entry_price'] - current_price)
            closed_trades.append({
                'ticker': tk, 'side': pos['side'], 'exit_reason': 'rebalance',
                'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                'exit_date': str(today.date()),
                'entry_price': pos['entry_price'], 'exit_price': current_price,
                'shares': pos['shares'], 'pnl': round(pnl, 4),
                'holding_days': day_idx - pos['entry_idx'],
            })

        # ── Open new positions ──
        realized_pnl = sum(t['pnl'] for t in closed_trades)
        available_capital = INITIAL_CAPITAL + realized_pnl
        deployed = sum(pos['shares'] * pos['entry_price'] for pos in holdings.values())
        free_capital = available_capital - deployed

        for tk, weight in target_longs.items():
            if tk in holdings:
                continue
            if tk not in fc.columns:
                continue
            current_price = float(fc[tk].iloc[day_idx])
            alloc = available_capital * weight
            shares = alloc / current_price
            if shares < 0.001:
                continue
            holdings[tk] = {
                'shares': shares, 'entry_price': current_price,
                'entry_date': today, 'entry_idx': day_idx,
                'side': 'long', 'peak_price': current_price,
            }

        for tk, weight in target_shorts.items():
            if tk in holdings:
                continue
            if tk not in fc.columns:
                continue
            current_price = float(fc[tk].iloc[day_idx])
            alloc = available_capital * weight
            shares = alloc / current_price
            if shares < 0.001:
                continue
            holdings[tk] = {
                'shares': shares, 'entry_price': current_price,
                'entry_date': today, 'entry_idx': day_idx,
                'side': 'short', 'peak_price': current_price,
            }

        if verbose and n_rebalances <= 3:
            longs_str = ', '.join(f"{tk}" for tk in target_longs)
            shorts_str = ', '.join(f"{tk}" for tk in target_shorts) if target_shorts else 'none'
            fprint(f"  Rebalance {n_rebalances} ({today.date()}): "
                   f"LONG [{longs_str}], SHORT [{shorts_str}], "
                   f"equity=${current_equity:.2f}")

    # ── Close all remaining positions ──
    for tk, pos in list(holdings.items()):
        if tk not in fc.columns:
            continue
        current_price = float(fc[tk].iloc[-1])
        if pos['side'] == 'long':
            pnl = pos['shares'] * (current_price - pos['entry_price'])
        else:
            pnl = pos['shares'] * (pos['entry_price'] - current_price)
        closed_trades.append({
            'ticker': tk, 'side': pos['side'], 'exit_reason': 'end_of_backtest',
            'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
            'exit_date': str(dates[-1].date()),
            'entry_price': pos['entry_price'], 'exit_price': current_price,
            'shares': pos['shares'], 'pnl': round(pnl, 4),
            'holding_days': len(dates) - 1 - pos['entry_idx'],
        })

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) > 0:
        eq_df = eq_df.set_index('date')
        eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

    metrics = compute_metrics(closed_trades, eq_df, variant)

    return {
        'trades': closed_trades,
        'equity_curve': eq_df,
        'metrics': metrics,
        'variant': variant,
        'n_rebalances': n_rebalances,
    }


# ==================== METRICS ====================

def compute_metrics(trades, eq_df, variant_name):
    if not trades:
        return {
            'variant': variant_name, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0, 'cagr': 0,
            'total_pnl': 0, 'mean_pnl': 0, 'wins': 0, 'losses': 0,
            'spy_total_return': 0, 'spy_sharpe': 0, 'alpha_vs_spy': 0,
        }

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0
    total_pnl = sum(pnls)

    # Sharpe from daily equity returns
    if len(eq_df) > 5:
        daily_rets = eq_df['equity'].pct_change().dropna()
        daily_rets = daily_rets.replace([np.inf, -np.inf], 0).fillna(0)
        sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-10) * np.sqrt(252))
        downside = daily_rets[daily_rets < 0]
        sortino = float(daily_rets.mean() / (downside.std() + 1e-10) * np.sqrt(252)) if len(downside) > 1 else sharpe
    else:
        sharpe = sortino = 0.0

    # Profit Factor
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / (gross_losses + 1e-10)

    # MDD
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

    # SPY benchmark
    spy_total_return = spy_sharpe = alpha_vs_spy = 0
    if 'spy_equity' in eq_df.columns and len(eq_df) > 5:
        spy_total_return = float(eq_df['spy_equity'].iloc[-1] / eq_df['spy_equity'].iloc[0] - 1)
        spy_daily = eq_df['spy_equity'].pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)
        spy_sharpe = float(spy_daily.mean() / (spy_daily.std() + 1e-10) * np.sqrt(252))
        alpha_vs_spy = total_return - spy_total_return

    return {
        'variant': variant_name,
        'n_trades': n, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'pf': round(pf, 3), 'wr': round(wr * 100, 1), 'mdd': round(mdd * 100, 2),
        'total_return': round(total_return * 100, 2), 'cagr': round(cagr * 100, 2),
        'total_pnl': round(total_pnl, 2), 'mean_pnl': round(np.mean(pnls), 4),
        'wins': wins, 'losses': n - wins,
        'spy_total_return': round(spy_total_return * 100, 2),
        'spy_sharpe': round(spy_sharpe, 3), 'alpha_vs_spy': round(alpha_vs_spy * 100, 2),
    }


# ==================== REGIME BREAKDOWN ====================

def compute_regime_breakdown(trades, spy):
    if not trades or len(spy) < 2:
        return {}

    regime_trades = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        exit_date = pd.Timestamp(t['exit_date'])
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
        trades_per_year = 13.0
        sharpe = (mean_pnl / (std_pnl + 1e-10)) * np.sqrt(trades_per_year) if std_pnl > 0 else 0
        breakdown[regime] = {
            'n': n, 'sharpe': round(sharpe, 2),
            'wr': round(wins / n * 100, 1) if n > 0 else 0,
            'mean_pnl': round(mean_pnl, 4), 'total_pnl': round(sum(pnls), 2),
        }
    return breakdown


# ==================== 5-GATE VALIDATION ====================

def five_gate_validation(result, spy):
    """
    Gate 1: Sharpe > 1.0
    Gate 2: Permutation p-value < 0.05 (bootstrap test)
    Gate 3: Win rate > 40%
    Gate 4: Regime balance (|Sharpe_green - Sharpe_red| / max < 0.50)
    Gate 5: Monte Carlo 95% CI lower bound > 0
    """
    metrics = result['metrics']
    trades = result['trades']
    gates = {}

    # Gate 1
    gates['sharpe_gt_1'] = {
        'pass': metrics['sharpe'] > 1.0,
        'value': metrics['sharpe'], 'threshold': 1.0,
    }

    # Gate 2: Bootstrap permutation test (100 shuffles for speed)
    n_perm = 100
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades])
        actual_mean = np.mean(pnls)
        rng = np.random.RandomState(42)
        perm_means = np.zeros(n_perm)
        for i in range(n_perm):
            idx = rng.choice(len(pnls), size=len(pnls), replace=True)
            perm_means[i] = np.mean(pnls[idx])
        if actual_mean > 0:
            p_value = float(np.mean(perm_means <= 0))
        else:
            p_value = 1.0
    else:
        p_value = 1.0
    gates['perm_p_lt_005'] = {
        'pass': p_value < 0.05,
        'value': round(p_value, 4), 'threshold': 0.05,
    }

    # Gate 3
    gates['wr_gt_40'] = {
        'pass': metrics['wr'] > 40.0,
        'value': metrics['wr'], 'threshold': 40.0,
    }

    # Gate 4: Regime balance
    breakdown = compute_regime_breakdown(trades, spy)
    sharpe_green = breakdown.get('green', {}).get('sharpe', 0)
    sharpe_red = breakdown.get('red', {}).get('sharpe', 0)
    max_s = max(abs(sharpe_green), abs(sharpe_red), 0.01)
    regime_skew = abs(sharpe_green - sharpe_red) / max_s
    gates['regime_balance'] = {
        'pass': regime_skew <= 0.50,
        'value': round(regime_skew, 3), 'threshold': 0.50,
        'sharpe_green': sharpe_green, 'sharpe_red': sharpe_red,
    }

    # Gate 5: MC 95% CI
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades])
        rng = np.random.RandomState(123)
        n_mc = 5000
        mc_totals = np.zeros(n_mc)
        for i in range(n_mc):
            idx = rng.choice(len(pnls), size=len(pnls), replace=True)
            mc_totals[i] = np.sum(pnls[idx])
        ci_95_lower = float(np.percentile(mc_totals, 2.5))
        ci_95_upper = float(np.percentile(mc_totals, 97.5))
    else:
        ci_95_lower = ci_95_upper = 0
    gates['mc_ci_positive'] = {
        'pass': ci_95_lower > 0,
        'value': round(ci_95_lower, 2), 'ci_upper': round(ci_95_upper, 2),
        'threshold': 0.0,
    }

    n_pass = sum(1 for g in gates.values() if g['pass'])
    return {
        'gates': gates, 'n_pass': n_pass, 'n_total': len(gates),
        'all_pass': n_pass == len(gates), 'regime_breakdown': breakdown,
    }


# ==================== MAIN ====================

def main():
    t0 = time.time()

    fprint("=" * 70)
    fprint("  FACTOR ETF ROTATION V1 — LGBM Ranking")
    fprint("  Hypothesis: Factor ETFs rotate better than sector ETFs")
    fprint("  Baseline: Sector rotation Sharpe 1.40 (KB #285)")
    fprint("=" * 70)

    fprint("\nDownloading data...")
    fc, spy, vix, qqq = download_data()

    oot_days = len(fc) - 280
    fprint(f"OOT days available: {oot_days}")
    if oot_days < 40:
        fprint(f"WARNING: Only {oot_days} OOT days (need >= 40)")

    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'Long Top-2 Monthly',
        'B': 'Long Top-1 Concentrated',
        'C': 'Long Top-2 Biweekly',
        'D': 'Long Top-2 + 15% Trail Stop',
        'E': 'Long Top-2 Short Bot-1',
        'F': 'Long Top-2 + ATM Call',
    }

    results = {}
    validations = {}

    for v in variants:
        fprint(f"\n{'='*60}")
        fprint(f"  VARIANT {v}: {variant_names[v]}")
        fprint(f"{'='*60}")
        t_start = time.time()
        results[v] = run_variant(fc, spy, vix, qqq, v, verbose=True)
        elapsed_v = time.time() - t_start
        m = results[v]['metrics']
        fprint(f"  Trades: {m['n_trades']} | Sharpe: {m['sharpe']:.3f} | "
               f"Sortino: {m['sortino']:.3f} | PF: {m['pf']:.3f} | "
               f"WR: {m['wr']:.1f}% | MDD: {m['mdd']:.2f}% | "
               f"Total Return: {m['total_return']:.2f}% | CAGR: {m['cagr']:.2f}%")
        fprint(f"  Alpha vs SPY: {m['alpha_vs_spy']:.2f}%")
        fprint(f"  Runtime: {elapsed_v:.1f}s")

        val = five_gate_validation(results[v], spy)
        validations[v] = val
        fprint(f"  5-Gate Validation: {val['n_pass']}/{val['n_total']} PASS")
        for name, gate in val['gates'].items():
            status = "PASS" if gate['pass'] else "FAIL"
            fprint(f"    {name}: {status} (value={gate['value']}, threshold={gate['threshold']})")

    # ==================== SUMMARY ====================
    fprint("\n" + "=" * 95)
    fprint("  COMPARISON SUMMARY — Factor ETF Rotation V1")
    fprint("=" * 95)
    fprint(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
           f"{'WR%':>6} {'MDD%':>7} {'TotRet%':>8} {'CAGR%':>7} {'Gates':>6}")
    fprint("-" * 95)

    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        fprint(f"{v}: {variant_names[v]:<27} {m['n_trades']:>5} {m['sharpe']:>7.3f} "
               f"{m['sortino']:>8.3f} {m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>8.2f} {m['cagr']:>7.2f} {val['n_pass']:>2}/{val['n_total']}")

    # SPY benchmark
    spy_m = results['A']['metrics']
    fprint(f"{'SPY Buy-Hold':<33} {'':>5} {spy_m['spy_sharpe']:>7.3f} "
           f"{'':>8} {'':>6} {'':>6} {'':>7} {spy_m['spy_total_return']:>8.2f} {'':>7} {'':>6}")

    # ── Regime breakdowns ──
    fprint("\n" + "=" * 95)
    fprint("  REGIME BREAKDOWNS")
    fprint("=" * 95)
    for v in variants:
        bd = validations[v]['regime_breakdown']
        fprint(f"\n  Variant {v} ({variant_names[v]}):")
        for regime in ['green', 'red', 'flat']:
            b = bd.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean_PnL=${b.get('mean_pnl',0):8.4f}  "
                   f"total_PnL=${b.get('total_pnl',0):8.2f}")

    # ── Alpha analysis ──
    fprint("\n" + "=" * 95)
    fprint("  ALPHA vs SPY BUY-AND-HOLD")
    fprint("=" * 95)
    for v in variants:
        m = results[v]['metrics']
        fprint(f"  {v}: {variant_names[v]:<30} Return={m['total_return']:>7.2f}%  "
               f"SPY={m['spy_total_return']:>7.2f}%  Alpha={m['alpha_vs_spy']:>7.2f}%")

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    fprint(f"\n  BEST VARIANT: {best_v} ({variant_names[best_v]}) — Sharpe {best_m['sharpe']:.3f}")

    # ── Conclusions ──
    fprint("\n" + "=" * 95)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 95)

    any_pass_all = any(validations[v]['all_pass'] for v in variants)
    sector_baseline_sharpe = 1.40

    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"  VERDICT: Factor rotation WORKS — passes all 5 gates!")
        fprint(f"  Passing variants: {', '.join(f'{v} ({variant_names[v]})' for v in passing)}")
        if best_m['sharpe'] > sector_baseline_sharpe:
            fprint(f"  BEATS sector rotation baseline (Sharpe {best_m['sharpe']:.3f} > {sector_baseline_sharpe})")
        else:
            fprint(f"  Below sector rotation baseline ({best_m['sharpe']:.3f} < {sector_baseline_sharpe})")
    elif best_m['sharpe'] > 1.0:
        fprint(f"  VERDICT: Factor rotation has edge but fails some gates.")
        fprint(f"  Best Sharpe: {best_m['sharpe']:.3f}")
    elif best_m['sharpe'] > 0:
        fprint(f"  VERDICT: Weak positive edge from factor rotation.")
        fprint(f"  Best Sharpe: {best_m['sharpe']:.3f}")
    else:
        fprint(f"  VERDICT: Factor rotation does NOT work with this config.")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{fc.index[0].date()} to {fc.index[-1].date()}",
        'n_days': len(fc), 'oot_days': oot_days,
        'initial_capital': INITIAL_CAPITAL, 'commission': 0.0,
        'factor_etfs': FACTOR_ETFS, 'n_features': len(FEAT_COLS),
        'sector_baseline_sharpe': sector_baseline_sharpe,
        'metrics': {}, 'validations': {}, 'regime_breakdowns': {},
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
        eq_path = OUTPUT_DIR / f"equity_curve_{v}.csv"
        results[v]['equity_curve'].to_csv(eq_path)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            run_name = f"factor_etf_rotation_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_factor_etfs", len(FACTOR_ETFS))
                mlflow.log_param("factor_etfs", ','.join(FACTOR_ETFS))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("data_range", f"{fc.index[0].date()} to {fc.index[-1].date()}")
                mlflow.log_param("best_variant", f"{best_v}_{variant_names[best_v]}")
                mlflow.log_param("sector_baseline_sharpe", sector_baseline_sharpe)

                for v in variants:
                    m = results[v]['metrics']
                    val = validations[v]
                    pfx = f"v{v}_"
                    mlflow.log_metric(f"{pfx}sharpe", m['sharpe'])
                    mlflow.log_metric(f"{pfx}sortino", m['sortino'])
                    mlflow.log_metric(f"{pfx}pf", m['pf'])
                    mlflow.log_metric(f"{pfx}wr", m['wr'])
                    mlflow.log_metric(f"{pfx}mdd", m['mdd'])
                    mlflow.log_metric(f"{pfx}total_return", m['total_return'])
                    mlflow.log_metric(f"{pfx}cagr", m['cagr'])
                    mlflow.log_metric(f"{pfx}n_trades", m['n_trades'])
                    mlflow.log_metric(f"{pfx}alpha_vs_spy", m['alpha_vs_spy'])
                    mlflow.log_metric(f"{pfx}gates_pass", val['n_pass'])

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
