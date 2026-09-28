#!/usr/bin/env python3
"""
International ETF Rotation V1 (2026-07-28)
============================================

HYPOTHESIS: LGBM momentum-based rotation across international/country ETFs
should work as well as US sector rotation (Sharpe 1.40) and US factor ETFs
(Sharpe 1.13). Country-level macro divergence may provide even stronger
rotation signals than sector rotation.

UNIVERSE (13 Country/Region ETFs + 1 Benchmark):
  EWJ  — Japan          EWG  — Germany        EWU  — UK
  EFA  — EAFE           EEM  — Emerging Mkts  FXI  — China
  EWZ  — Brazil         INDA — India          EWT  — Taiwan
  EWY  — Korea          EWA  — Australia       EWC  — Canada
  VGK  — Europe         SPY  — benchmark

SIX VARIANTS:
  A: Long Top-2, monthly rebalance (21 trading days)
  B: Long Top-1 concentrated
  C: Long Top-2 biweekly (every 10 trading days)
  D: Long Top-2 + 15% trailing stop
  E: Long Top-2 + VIX filter (cash when VIX>25)
  F: Long Top-2 + Short Bottom-1

FEATURES: 22 momentum + cross-asset
VALIDATION: 5 gates (Sharpe>1, perm p<0.05, WR>40%, regime balance <0.50, beats random)
Capital: $645, zero-commission (Robinhood), data from 2015-01-01

Output: output/growth_research/international_etf_rotation_v1/
MLflow experiment: international_etf_rotation_v1
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

OUTPUT_DIR = BASE / "output" / "growth_research" / "international_etf_rotation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "international_etf_rotation_v1"
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

INTL_ETFS = ['EWJ', 'EWG', 'EWU', 'EFA', 'EEM', 'FXI', 'EWZ', 'INDA',
             'EWT', 'EWY', 'EWA', 'EWC', 'VGK']
BENCHMARKS = ['SPY']
INITIAL_CAPITAL = 645.0

# 22 features
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_21d', 'sharpe_63d', 'max_dd_63d',
    'sma_ratio_10_50', 'sma_ratio_20_200', 'rsi_14', 'macd_signal',
    'bb_pct', 'skew_21d', 'corr_to_spy_63d', 'beta_to_spy_63d',
    'rel_strength_vs_mean', 'ret_21d_minus_spy', 'vol_ratio_21_63',
]

DATA_START = '2015-01-01'
TRAIN_WINDOW = 252  # 1 year sliding window
REBAL_MONTHLY = 21
REBAL_BIWEEKLY = 10


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download international ETF + benchmark + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = INTL_ETFS + BENCHMARKS + ['^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers from {DATA_START}...")
    raw = yf.download(all_tickers, start=DATA_START, progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill().rename(columns={'^VIX': 'VIX'})

    vix = close['VIX'].dropna()
    spy = close['SPY'].dropna()

    available_etfs = [c for c in INTL_ETFS if c in close.columns]
    missing = [c for c in INTL_ETFS if c not in close.columns]
    if missing:
        fprint(f"WARNING: Missing ETFs: {missing}")

    fc = close[available_etfs].dropna(how='all')
    ix = fc.index.intersection(spy.index).intersection(vix.index)
    fprint(f"Data: {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}, "
           f"{len(ix)} days, {len(available_etfs)} intl ETFs")
    fprint(f"ETFs: {available_etfs}")
    return fc.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (22 features) ====================

def compute_features(px, spy_px=None, all_rets_21d=None):
    """Compute 22 momentum + cross-asset features for a single ETF."""
    if len(px) < 260:
        return None
    f = {}
    rets = px.pct_change().dropna()

    # 6 return lookbacks
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    # Volatility
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    # Sharpe
    r21 = rets.iloc[-21:]
    f['sharpe_21d'] = float(r21.mean() / (r21.std() + 1e-10) * np.sqrt(252)) if len(r21) > 5 else 0.0
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    # Max drawdown 63d
    pk63 = px.iloc[-63:].cummax()
    f['max_dd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())

    # SMA ratios
    sma10 = px.iloc[-10:].mean()
    sma50 = px.iloc[-50:].mean() if len(px) >= 50 else sma10
    sma20 = px.iloc[-20:].mean()
    sma200 = px.iloc[-200:].mean() if len(px) >= 200 else sma20
    f['sma_ratio_10_50'] = float(sma10 / (sma50 + 1e-10))
    f['sma_ratio_20_200'] = float(sma20 / (sma200 + 1e-10))

    # RSI 14
    delta = px.diff().iloc[-15:]
    gain = delta.clip(lower=0).mean()
    loss = (-delta.clip(upper=0)).mean()
    rs = gain / (loss + 1e-10)
    f['rsi_14'] = float(100 - 100 / (1 + rs))

    # MACD signal
    ema12 = px.ewm(span=12).mean().iloc[-1]
    ema26 = px.ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    signal = px.ewm(span=9).mean().iloc[-1]  # simplified
    f['macd_signal'] = float((macd - signal) / (px.iloc[-1] + 1e-10))

    # Bollinger band %
    sma20_val = px.iloc[-20:].mean()
    std20 = px.iloc[-20:].std()
    upper = sma20_val + 2 * std20
    lower = sma20_val - 2 * std20
    f['bb_pct'] = float((px.iloc[-1] - lower) / (upper - lower + 1e-10))

    # Skewness 21d
    f['skew_21d'] = float(rets.iloc[-21:].skew()) if len(rets) > 21 else 0.0

    # Cross-asset features
    if spy_px is not None and len(spy_px) >= 63:
        spy_rets = spy_px.pct_change().dropna()
        min_len = min(len(rets), len(spy_rets))
        etf_r = rets.iloc[-min_len:]
        spy_r = spy_rets.iloc[-min_len:]
        if min_len >= 63:
            f['corr_to_spy_63d'] = float(etf_r.iloc[-63:].corr(spy_r.iloc[-63:]))
            cov = np.cov(etf_r.iloc[-63:].values, spy_r.iloc[-63:].values)
            f['beta_to_spy_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
        else:
            f['corr_to_spy_63d'] = 0.0
            f['beta_to_spy_63d'] = 1.0
        # Ret minus SPY
        spy_21d = float(spy_px.iloc[-1] / spy_px.iloc[-21] - 1) if len(spy_px) > 21 else 0.0
        f['ret_21d_minus_spy'] = f['ret_21d'] - spy_21d
    else:
        f['corr_to_spy_63d'] = 0.0
        f['beta_to_spy_63d'] = 1.0
        f['ret_21d_minus_spy'] = 0.0

    # Relative strength vs universe mean
    if all_rets_21d is not None and len(all_rets_21d) > 0:
        mean_ret = np.mean(list(all_rets_21d.values()))
        f['rel_strength_vs_mean'] = f['ret_21d'] - mean_ret
    else:
        f['rel_strength_vs_mean'] = 0.0

    # Vol ratio
    f['vol_ratio_21_63'] = f['vol_21d'] / (f['vol_63d'] + 1e-10)

    return f


# ==================== LGBM RANKING (walk-forward, sliding window) ====================

def run_lgbm_ranking_wf(fc, spy, idx_end, train_window=TRAIN_WINDOW, sample_every=21):
    """Walk-forward LGBM ranking using sliding window.
    Train on samples from [idx_end - train_window, idx_end) sampled every 21 days.
    Predict forward 21-day returns (target).
    """
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
            # Forward 21-day return as target
            fi = min(i + 21, len(fc) - 1)
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

def run_variant(fc, spy, vix, variant, verbose=False):
    """Run a single international ETF rotation backtest variant."""
    dates = fc.index
    n_days = len(dates)
    start_idx = 280  # Need 260 for features + warm-up

    # Rebalance interval
    rebal_days = REBAL_BIWEEKLY if variant == 'C' else REBAL_MONTHLY

    equity = INITIAL_CAPITAL
    equity_curve = []
    holdings = {}
    closed_trades = []
    last_rebalance_idx = None
    n_rebalances = 0

    spy_entry_price = float(spy.iloc[start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]
        current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20.0

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
                if drawdown >= 0.15:
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

        # ── VIX filter: go to cash (Variant E) ──
        if variant == 'E' and current_vix > 25.0 and holdings:
            for tk in list(holdings.keys()):
                pos = holdings.pop(tk)
                exit_price = float(fc[tk].iloc[day_idx])
                pnl = pos['shares'] * (exit_price - pos['entry_price'])
                closed_trades.append({
                    'ticker': tk, 'side': 'long', 'exit_reason': 'vix_filter',
                    'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                    'exit_date': str(today.date()),
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'shares': pos['shares'], 'pnl': round(pnl, 4),
                    'holding_days': day_idx - pos['entry_idx'],
                })

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
            'date': today, 'equity': current_equity,
            'spy_equity': spy_equity, 'n_holdings': len(holdings),
        })

        # ── Check for rebalance ──
        do_rebalance = False
        if last_rebalance_idx is None and day_idx >= start_idx + 5:
            do_rebalance = True
        elif last_rebalance_idx is not None and day_idx - last_rebalance_idx >= rebal_days:
            do_rebalance = True

        # VIX filter blocks new entries for variant E
        if variant == 'E' and current_vix > 25.0:
            do_rebalance = False

        if not do_rebalance:
            continue

        last_rebalance_idx = day_idx
        n_rebalances += 1

        # ── Get LGBM ranking ──
        rankings, model = run_lgbm_ranking_wf(fc, spy, day_idx)
        if not rankings:
            continue

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        # ── Determine target portfolio ──
        target_longs = {}
        target_shorts = {}

        if variant in ('A', 'D', 'E'):
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for tk in top2:
                target_longs[tk] = w

        elif variant == 'B':
            top1 = ranked[0][0]
            target_longs[top1] = 1.0

        elif variant == 'C':
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for tk in top2:
                target_longs[tk] = w

        elif variant == 'F':
            top2 = [t for t, _ in ranked[:2]]
            bot1 = ranked[-1][0]
            w_long = 0.40
            w_short = 0.20
            for tk in top2:
                target_longs[tk] = w_long
            target_shorts[bot1] = w_short

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
    Gate 5: Beats random baseline (shuffle rankings 100 times)
    """
    metrics = result['metrics']
    trades = result['trades']
    gates = {}

    # Gate 1
    gates['sharpe_gt_1'] = {
        'pass': metrics['sharpe'] > 1.0,
        'value': metrics['sharpe'], 'threshold': 1.0,
    }

    # Gate 2: Permutation test (100 shuffles)
    n_perm = 100
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades])
        actual_mean = np.mean(pnls)
        rng = np.random.RandomState(42)
        count_ge = 0
        for i in range(n_perm):
            shuffled = pnls.copy()
            rng.shuffle(shuffled)
            signs = rng.choice([-1, 1], size=len(shuffled))
            if np.mean(shuffled * signs) >= actual_mean:
                count_ge += 1
        p_value = float(count_ge / n_perm)
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

    # Gate 5: Beats random baseline (shuffle rankings 100 times, compare mean PnL)
    n_random = 100
    if len(trades) >= 10:
        actual_total = sum(t['pnl'] for t in trades)
        rng = np.random.RandomState(99)
        pnls_arr = np.array([t['pnl'] for t in trades])
        count_beat = 0
        for i in range(n_random):
            shuffled = pnls_arr.copy()
            rng.shuffle(shuffled)
            random_total = np.sum(shuffled[:len(shuffled)])
            if actual_total > random_total:
                count_beat += 1
        beat_pct = count_beat / n_random
    else:
        beat_pct = 0.0
    gates['beats_random'] = {
        'pass': beat_pct > 0.50,
        'value': round(beat_pct, 3), 'threshold': 0.50,
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
    fprint("  INTERNATIONAL ETF ROTATION V1 — LGBM Ranking")
    fprint("  Hypothesis: Country ETF rotation via LGBM momentum ranking")
    fprint("  Baselines: Sector rotation Sharpe 1.40, Factor rotation Sharpe 1.13")
    fprint("=" * 70)

    fprint("\nDownloading data...")
    fc, spy, vix = download_data()

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
        'E': 'Long Top-2 + VIX Filter (<25)',
        'F': 'Long Top-2 Short Bot-1',
    }

    results = {}
    validations = {}

    for v in variants:
        fprint(f"\n{'='*60}")
        fprint(f"  VARIANT {v}: {variant_names[v]}")
        fprint(f"{'='*60}")
        t_start = time.time()
        results[v] = run_variant(fc, spy, vix, v, verbose=True)
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
    fprint("  COMPARISON SUMMARY — International ETF Rotation V1")
    fprint("=" * 95)
    fprint(f"{'Variant':<32} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
           f"{'WR%':>6} {'MDD%':>7} {'TotRet%':>8} {'CAGR%':>7} {'Gates':>6}")
    fprint("-" * 95)

    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        fprint(f"{v}: {variant_names[v]:<29} {m['n_trades']:>5} {m['sharpe']:>7.3f} "
               f"{m['sortino']:>8.3f} {m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>8.2f} {m['cagr']:>7.2f} {val['n_pass']:>2}/{val['n_total']}")

    spy_m = results['A']['metrics']
    fprint(f"{'SPY Buy-Hold':<35} {'':>5} {spy_m['spy_sharpe']:>7.3f} "
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
    factor_baseline_sharpe = 1.13

    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"  VERDICT: International rotation WORKS — passes all 5 gates!")
        fprint(f"  Passing variants: {', '.join(f'{v} ({variant_names[v]})' for v in passing)}")
        if best_m['sharpe'] > sector_baseline_sharpe:
            fprint(f"  BEATS sector rotation baseline (Sharpe {best_m['sharpe']:.3f} > {sector_baseline_sharpe})")
        elif best_m['sharpe'] > factor_baseline_sharpe:
            fprint(f"  BEATS factor rotation baseline (Sharpe {best_m['sharpe']:.3f} > {factor_baseline_sharpe})")
        else:
            fprint(f"  Below factor rotation baseline ({best_m['sharpe']:.3f} < {factor_baseline_sharpe})")
    elif best_m['sharpe'] > 1.0:
        fprint(f"  VERDICT: International rotation has edge but fails some gates.")
        fprint(f"  Best Sharpe: {best_m['sharpe']:.3f}")
    elif best_m['sharpe'] > 0:
        fprint(f"  VERDICT: Weak positive edge from international rotation.")
        fprint(f"  Best Sharpe: {best_m['sharpe']:.3f}")
    else:
        fprint(f"  VERDICT: International rotation does NOT work with this config.")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{fc.index[0].date()} to {fc.index[-1].date()}",
        'n_days': len(fc), 'oot_days': oot_days,
        'initial_capital': INITIAL_CAPITAL, 'commission': 0.0,
        'intl_etfs': INTL_ETFS, 'n_features': len(FEAT_COLS),
        'sector_baseline_sharpe': sector_baseline_sharpe,
        'factor_baseline_sharpe': factor_baseline_sharpe,
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
            run_name = f"intl_etf_rotation_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_intl_etfs", len(INTL_ETFS))
                mlflow.log_param("intl_etfs", ','.join(INTL_ETFS))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("train_window", TRAIN_WINDOW)
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

    fprint("\nDone.")


if __name__ == '__main__':
    main()
