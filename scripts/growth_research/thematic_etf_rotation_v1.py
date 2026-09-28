#!/usr/bin/env python3
"""
Thematic ETF Rotation V1 (2026-07-28)
======================================

HYPOTHESIS: Thematic/innovation ETFs have higher volatility and stronger
momentum persistence than broad market ETFs, making LGBM momentum-based
rotation potentially more profitable.

UNIVERSE (16 Thematic ETFs + 2 Benchmarks):
  ARKK  — ARK Innovation
  ARKG  — ARK Genomics
  ARKW  — ARK Next Gen Internet
  ICLN  — iShares Global Clean Energy
  TAN   — Invesco Solar
  QCLN  — First Trust NASDAQ Clean Edge Green
  HACK  — ETFMG Prime Cyber Security
  CIBR  — First Trust NASDAQ Cybersecurity
  SOXX  — iShares Semiconductor
  SMH   — VanEck Semiconductor
  XBI   — SPDR S&P Biotech
  IBB   — iShares Biotechnology
  KWEB  — KraneShares CSI China Internet
  SKYY  — First Trust Cloud Computing
  BOTZ  — Global X Robotics & AI
  ROBO  — ROBO Global Robotics & Automation
  SPY, QQQ — benchmarks

SIX VARIANTS:
  A: Long Top-2, monthly rebalance
  B: Long Top-1 concentrated
  C: Long Top-3, monthly (more diversified)
  D: Long Top-2 + 15% trailing stop
  E: Long Top-2 + VIX filter (cash when VIX>25)
  F: Long Top-2 + Short Bottom-1

FEATURES: 22 (17 momentum + 5 cross-asset)
VALIDATION: 5 gates (Sharpe>1, perm p<0.05, WR>40%, regime gap<0.50, beats random)
Capital: $645, zero-commission (Robinhood), data from 2017-01-01
Walk-forward: 252-day sliding train window, predict next 21 trading days

Output: output/growth_research/thematic_etf_rotation_v1/
MLflow experiment: thematic_etf_rotation_v1
"""

import json
import os
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
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "thematic_etf_rotation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "thematic_etf_rotation_v1"
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

THEMATIC_ETFS = [
    'ARKK', 'ARKG', 'ARKW',           # ARK Innovation
    'ICLN', 'TAN', 'QCLN',            # Clean Energy
    'HACK', 'CIBR',                    # Cybersecurity
    'SOXX', 'SMH',                     # Semiconductors
    'XBI', 'IBB',                      # Biotech
    'KWEB', 'SKYY',                    # China Internet / Cloud
    'BOTZ', 'ROBO',                    # Robotics & AI
]
BENCHMARKS = ['SPY', 'QQQ']
INITIAL_CAPITAL = 645.0
DATA_START = '2017-01-01'
TRAIN_WINDOW = 252   # 1 year sliding window
REBAL_DAYS = 21      # Monthly rebalance
FWD_RETURN_DAYS = 21  # Forward 21-day return target
N_PERM = 100          # Permutation shuffles

# 22 features
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
    # 5 cross-asset features
    'corr_to_spy_63d', 'beta_to_spy_63d', 'rel_strength_vs_mean',
    'rank_21d', 'vol_ratio_vs_spy',
]


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download thematic ETF + benchmark + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = THEMATIC_ETFS + BENCHMARKS + ['^VIX']
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

    available_etfs = [c for c in THEMATIC_ETFS if c in close.columns and close[c].dropna().shape[0] > 260]
    missing = [c for c in THEMATIC_ETFS if c not in available_etfs]
    if missing:
        fprint(f"Skipped (insufficient data): {missing}")

    fc = close[available_etfs].dropna(how='all')
    ix = fc.index.intersection(spy.index).intersection(vix.index)
    fprint(f"Data: {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}, "
           f"{len(ix)} days, {len(available_etfs)} thematic ETFs")
    fprint(f"ETFs: {available_etfs}")
    return fc.loc[ix], spy.loc[ix], vix.loc[ix], qqq.loc[ix] if 'QQQ' in close.columns else spy.loc[ix]


# ==================== FEATURE ENGINEERING (22 features) ====================

def compute_features(px, spy_px=None, all_rets_21d=None):
    """Compute 22 momentum + cross-asset features for a single ETF."""
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

    # ── 5 Cross-asset features ──
    if spy_px is not None and len(spy_px) >= 63:
        spy_rets = spy_px.pct_change().dropna()
        min_len = min(len(rets), len(spy_rets))
        etf_r = rets.iloc[-min_len:]
        spy_r = spy_rets.iloc[-min_len:]
        if min_len >= 63:
            f['corr_to_spy_63d'] = float(etf_r.iloc[-63:].corr(spy_r.iloc[-63:]))
            cov = np.cov(etf_r.iloc[-63:].values, spy_r.iloc[-63:].values)
            f['beta_to_spy_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            spy_vol = float(spy_r.iloc[-21:].std() * np.sqrt(252))
            f['vol_ratio_vs_spy'] = f['vol_21d'] / (spy_vol + 1e-10)
        else:
            f['corr_to_spy_63d'] = 0.0
            f['beta_to_spy_63d'] = 1.0
            f['vol_ratio_vs_spy'] = 1.0
    else:
        f['corr_to_spy_63d'] = 0.0
        f['beta_to_spy_63d'] = 1.0
        f['vol_ratio_vs_spy'] = 1.0

    if all_rets_21d is not None:
        mean_ret = np.mean(list(all_rets_21d.values()))
        f['rel_strength_vs_mean'] = f['ret_21d'] - mean_ret
    else:
        f['rel_strength_vs_mean'] = 0.0

    f['rank_21d'] = 0.5  # placeholder
    return f


# ==================== LGBM RANKING (walk-forward, sliding window) ====================

def run_lgbm_ranking_wf(fc, spy, idx_end, train_window=TRAIN_WINDOW, sample_every=REBAL_DAYS):
    """Walk-forward LGBM ranking with sliding window."""
    if not HAS_LGBM:
        rets_21d = fc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False)), None

    records = []
    start_i = max(260, idx_end - train_window)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::sample_every]
    spy_px = spy.iloc[:idx_end + 1]

    for i in rebal_idx[:-1]:
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
            if rets_21d_all:
                sorted_rets = sorted(rets_21d_all.values())
                n_etfs = len(sorted_rets)
                if tk in rets_21d_all and n_etfs > 1:
                    feats['rank_21d'] = float(sorted_rets.index(rets_21d_all[tk])) / (n_etfs - 1)

            fi = min(i + FWD_RETURN_DAYS, len(fc) - 1)
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


# ==================== BACKTEST ENGINE ====================

def run_variant(fc, spy, vix, qqq, variant, verbose=False):
    """Run a single thematic ETF rotation backtest variant."""
    dates = fc.index
    n_days = len(dates)
    start_idx = 280

    equity = INITIAL_CAPITAL
    equity_curve = []
    holdings = {}
    closed_trades = []
    last_rebalance_idx = None
    n_rebalances = 0
    in_cash = False  # For VIX filter (variant E)

    spy_entry_price = float(spy.iloc[start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    # Determine top-N for each variant
    top_n = {'A': 2, 'B': 1, 'C': 3, 'D': 2, 'E': 2, 'F': 2}[variant]

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # ── Trailing stop (Variant D) ──
        if variant == 'D':
            stops = []
            for tk, pos in holdings.items():
                if pos['side'] != 'long':
                    continue
                cp = float(fc[tk].iloc[day_idx])
                if cp > pos.get('peak_price', pos['entry_price']):
                    pos['peak_price'] = cp
                peak = pos.get('peak_price', pos['entry_price'])
                if (peak - cp) / peak >= 0.15:
                    stops.append(tk)
            for tk in stops:
                pos = holdings.pop(tk)
                ep = float(fc[tk].iloc[day_idx])
                pnl = pos['shares'] * (ep - pos['entry_price'])
                closed_trades.append({
                    'ticker': tk, 'side': 'long', 'exit_reason': 'trailing_stop',
                    'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                    'exit_date': str(today.date()),
                    'entry_price': pos['entry_price'], 'exit_price': ep,
                    'shares': pos['shares'], 'pnl': round(pnl, 4),
                    'holding_days': day_idx - pos['entry_idx'],
                })

        # ── Mark-to-market ──
        mtm_pnl = 0.0
        for tk, pos in holdings.items():
            if tk not in fc.columns:
                continue
            cp = float(fc[tk].iloc[day_idx])
            if pos['side'] == 'long':
                mtm_pnl += pos['shares'] * (cp - pos['entry_price'])
            else:
                mtm_pnl += pos['shares'] * (pos['entry_price'] - cp)

        realized_pnl = sum(t['pnl'] for t in closed_trades)
        current_equity = INITIAL_CAPITAL + mtm_pnl + realized_pnl
        spy_equity = spy_shares * float(spy.iloc[day_idx])
        equity_curve.append({
            'date': today, 'equity': current_equity,
            'spy_equity': spy_equity, 'n_holdings': len(holdings),
        })

        # ── Rebalance check (monthly = every 21 trading days) ──
        do_rebalance = False
        if last_rebalance_idx is None:
            do_rebalance = True
        elif day_idx - last_rebalance_idx >= REBAL_DAYS:
            do_rebalance = True
        if not do_rebalance:
            continue

        last_rebalance_idx = day_idx
        n_rebalances += 1

        # ── VIX filter (Variant E): go to cash when VIX > 25 ──
        if variant == 'E':
            current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20
            if current_vix > 25:
                if not in_cash:
                    # Close all positions
                    for tk in list(holdings.keys()):
                        pos = holdings.pop(tk)
                        cp = float(fc[tk].iloc[day_idx])
                        pnl = pos['shares'] * (cp - pos['entry_price']) if pos['side'] == 'long' \
                            else pos['shares'] * (pos['entry_price'] - cp)
                        closed_trades.append({
                            'ticker': tk, 'side': pos['side'], 'exit_reason': 'vix_filter',
                            'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                            'exit_date': str(today.date()),
                            'entry_price': pos['entry_price'], 'exit_price': cp,
                            'shares': pos['shares'], 'pnl': round(pnl, 4),
                            'holding_days': day_idx - pos['entry_idx'],
                        })
                    in_cash = True
                continue
            else:
                in_cash = False

        # ── Get LGBM ranking ──
        rankings, model = run_lgbm_ranking_wf(fc, spy, day_idx)
        if not rankings:
            continue

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        # ── Determine target portfolio ──
        target_longs = {}
        target_shorts = {}

        top = [t for t, _ in ranked[:top_n]]
        w = 1.0 / len(top)

        if variant == 'F':
            # Long Top-2 (40% each) + Short Bottom-1 (20%)
            bot1 = ranked[-1][0]
            for tk in top:
                target_longs[tk] = 0.40
            target_shorts[bot1] = 0.20
        else:
            for tk in top:
                target_longs[tk] = w

        # ── Close positions not in target ──
        for tk in list(holdings.keys()):
            pos = holdings[tk]
            in_target = (tk in target_longs and pos['side'] == 'long') or \
                        (tk in target_shorts and pos['side'] == 'short')
            if not in_target:
                holdings.pop(tk)
                cp = float(fc[tk].iloc[day_idx])
                if pos['side'] == 'long':
                    pnl = pos['shares'] * (cp - pos['entry_price'])
                else:
                    pnl = pos['shares'] * (pos['entry_price'] - cp)
                closed_trades.append({
                    'ticker': tk, 'side': pos['side'], 'exit_reason': 'rebalance',
                    'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                    'exit_date': str(today.date()),
                    'entry_price': pos['entry_price'], 'exit_price': cp,
                    'shares': pos['shares'], 'pnl': round(pnl, 4),
                    'holding_days': day_idx - pos['entry_idx'],
                })

        # ── Open new positions ──
        realized_pnl = sum(t['pnl'] for t in closed_trades)
        available_capital = INITIAL_CAPITAL + realized_pnl

        for tk, weight in {**target_longs, **target_shorts}.items():
            if tk in holdings:
                continue
            if tk not in fc.columns:
                continue
            cp = float(fc[tk].iloc[day_idx])
            shares = (available_capital * weight) / cp
            if shares < 0.001:
                continue
            side = 'long' if tk in target_longs else 'short'
            holdings[tk] = {
                'shares': shares, 'entry_price': cp,
                'entry_date': today, 'entry_idx': day_idx,
                'side': side, 'peak_price': cp,
            }

        if verbose and n_rebalances <= 3:
            longs_str = ', '.join(target_longs.keys())
            shorts_str = ', '.join(target_shorts.keys()) if target_shorts else 'none'
            fprint(f"  Rebalance {n_rebalances} ({today.date()}): "
                   f"LONG [{longs_str}], SHORT [{shorts_str}], eq=${current_equity:.2f}")

    # ── Close remaining positions ──
    for tk, pos in list(holdings.items()):
        if tk not in fc.columns:
            continue
        cp = float(fc[tk].iloc[-1])
        pnl = pos['shares'] * (cp - pos['entry_price']) if pos['side'] == 'long' \
            else pos['shares'] * (pos['entry_price'] - cp)
        closed_trades.append({
            'ticker': tk, 'side': pos['side'], 'exit_reason': 'end_of_backtest',
            'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
            'exit_date': str(dates[-1].date()),
            'entry_price': pos['entry_price'], 'exit_price': cp,
            'shares': pos['shares'], 'pnl': round(pnl, 4),
            'holding_days': len(dates) - 1 - pos['entry_idx'],
        })

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) > 0:
        eq_df = eq_df.set_index('date')
        eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

    metrics = compute_metrics(closed_trades, eq_df, variant)
    return {
        'trades': closed_trades, 'equity_curve': eq_df,
        'metrics': metrics, 'variant': variant, 'n_rebalances': n_rebalances,
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

    if len(eq_df) > 5:
        daily_rets = eq_df['equity'].pct_change().dropna()
        daily_rets = daily_rets.replace([np.inf, -np.inf], 0).fillna(0)
        sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-10) * np.sqrt(252))
        downside = daily_rets[daily_rets < 0]
        sortino = float(daily_rets.mean() / (downside.std() + 1e-10) * np.sqrt(252)) if len(downside) > 1 else sharpe
    else:
        sharpe = sortino = 0.0

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
    Gate 2: Permutation p-value < 0.05 (100 shuffles)
    Gate 3: Win rate > 40%
    Gate 4: Regime balance (|Sharpe_green - Sharpe_red| / max < 0.50)
    Gate 5: Beats random (MC 95% CI lower bound > 0)
    """
    metrics = result['metrics']
    trades = result['trades']
    gates = {}

    gates['sharpe_gt_1'] = {
        'pass': metrics['sharpe'] > 1.0,
        'value': metrics['sharpe'], 'threshold': 1.0,
    }

    # Gate 2: Permutation test
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades])
        actual_mean = np.mean(pnls)
        rng = np.random.RandomState(42)
        perm_means = np.zeros(N_PERM)
        for i in range(N_PERM):
            shuffled = pnls.copy()
            rng.shuffle(shuffled)
            # Random entry timing: assign random signs
            signs = rng.choice([-1, 1], size=len(shuffled))
            perm_means[i] = np.mean(shuffled * signs)
        p_value = float(np.mean(perm_means >= actual_mean)) if actual_mean > 0 else 1.0
    else:
        p_value = 1.0
    gates['perm_p_lt_005'] = {
        'pass': p_value < 0.05,
        'value': round(p_value, 4), 'threshold': 0.05,
    }

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

    # Gate 5: MC CI
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
    pid = os.getpid()

    fprint("=" * 70)
    fprint("  THEMATIC ETF ROTATION V1 — LGBM Momentum Ranking")
    fprint("  Hypothesis: Thematic ETFs have stronger momentum persistence")
    fprint(f"  PID: {pid}")
    fprint("=" * 70)

    fprint("\nDownloading data...")
    fc, spy, vix, qqq = download_data()

    oot_days = len(fc) - 280
    fprint(f"OOT days available: {oot_days}")

    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'Long Top-2 Monthly',
        'B': 'Long Top-1 Concentrated',
        'C': 'Long Top-3 Diversified',
        'D': 'Long Top-2 + 15% Trail Stop',
        'E': 'Long Top-2 + VIX Filter',
        'F': 'Long Top-2 Short Bot-1',
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
    fprint("  COMPARISON SUMMARY — Thematic ETF Rotation V1")
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

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    fprint(f"\n  BEST VARIANT: {best_v} ({variant_names[best_v]}) — Sharpe {best_m['sharpe']:.3f}")

    # ── Conclusions ──
    fprint("\n" + "=" * 95)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 95)

    any_pass_all = any(validations[v]['all_pass'] for v in variants)
    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"  VERDICT: Thematic ETF rotation WORKS — passes all 5 gates!")
        fprint(f"  Passing variants: {', '.join(f'{v} ({variant_names[v]})' for v in passing)}")
    elif best_m['sharpe'] > 1.0:
        fprint(f"  VERDICT: Thematic rotation has edge but fails some gates.")
    elif best_m['sharpe'] > 0:
        fprint(f"  VERDICT: Weak positive edge from thematic rotation.")
    else:
        fprint(f"  VERDICT: Thematic rotation does NOT work with this config.")
    fprint(f"  Best Sharpe: {best_m['sharpe']:.3f}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s | PID: {pid}")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{fc.index[0].date()} to {fc.index[-1].date()}",
        'n_days': len(fc), 'oot_days': oot_days,
        'initial_capital': INITIAL_CAPITAL, 'commission': 0.0,
        'thematic_etfs': list(fc.columns),
        'n_features': len(FEAT_COLS),
        'train_window': TRAIN_WINDOW,
        'fwd_return_days': FWD_RETURN_DAYS,
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
    with open(results_path, 'w') as f_out:
        json.dump(save_results, f_out, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    for v in variants:
        trades_path = OUTPUT_DIR / f"trades_variant_{v}.json"
        with open(trades_path, 'w') as f_out:
            json.dump(results[v]['trades'], f_out, indent=2, default=str)
        eq_path = OUTPUT_DIR / f"equity_curve_{v}.csv"
        results[v]['equity_curve'].to_csv(eq_path)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            run_name = f"thematic_etf_rotation_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_thematic_etfs", len(fc.columns))
                mlflow.log_param("thematic_etfs", ','.join(fc.columns))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("train_window", TRAIN_WINDOW)
                mlflow.log_param("fwd_return_days", FWD_RETURN_DAYS)
                mlflow.log_param("data_range", f"{fc.index[0].date()} to {fc.index[-1].date()}")
                mlflow.log_param("best_variant", f"{best_v}_{variant_names[best_v]}")

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

    fprint(f"\nDone. PID={pid}")


if __name__ == '__main__':
    main()
