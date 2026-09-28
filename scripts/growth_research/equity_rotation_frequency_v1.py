#!/usr/bin/env python3
"""
Equity Rotation Frequency V1
=================================

Tests different rebalance frequencies for the sector equity rotation strategy.

CONTEXT:
  - Monthly Top-2 equity rotation gives Sharpe 1.40, +17.5% alpha (validated, KB #285)
  - We want to know if more frequent rebalancing captures more alpha or just adds noise
  - The leveraged rotation test showed leverage doesn't help Sharpe

SIX VARIANTS:
  A: Monthly Top-2 (control) — reproduce Sharpe ~1.4
  B: Biweekly Top-2 — rebalance every 10 trading days
  C: Weekly Top-2 — rebalance every 5 trading days
  D: Monthly Top-3 — more diversified, monthly
  E: Monthly Top-4 — even more diversified, monthly
  F: Adaptive — weekly if VIX>20, monthly if VIX<20

IMPLEMENTATION:
  - SAME 17-feature LGBM from V10 production (exact copy)
  - 60-day sliding train, sliding walk-forward (HC #0)
  - All available OOT days
  - Universe: 11 SPDR sector ETFs
  - $0 commission, $645 starting capital
  - 5-gate validation per variant
  - KEY METRIC: Sharpe per transaction cost (trades-per-year)

Output: output/growth_research/equity_rotation_frequency_v1/
MLflow experiment: equity_rotation_frequency_v1
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
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "equity_rotation_frequency_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "equity_rotation_frequency_v1"
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

# 17 LGBM momentum features (identical to V10)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

# Variant configurations
VARIANT_CONFIG = {
    'A': {'name': 'Monthly Top-2 (control)', 'n_picks': 2, 'rebal_days': 21, 'adaptive': False},
    'B': {'name': 'Biweekly Top-2',          'n_picks': 2, 'rebal_days': 10, 'adaptive': False},
    'C': {'name': 'Weekly Top-2',             'n_picks': 2, 'rebal_days': 5,  'adaptive': False},
    'D': {'name': 'Monthly Top-3',            'n_picks': 3, 'rebal_days': 21, 'adaptive': False},
    'E': {'name': 'Monthly Top-4',            'n_picks': 4, 'rebal_days': 21, 'adaptive': False},
    'F': {'name': 'Adaptive (VIX-gated)',     'n_picks': 2, 'rebal_days': None, 'adaptive': True},
}


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + SPY + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + ['SPY', '^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2023-01-01', progress=False)
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


# ==================== FEATURE ENGINEERING (17 momentum features, identical to V10) ====================

def compute_features(px):
    """Compute the 17 momentum features for a single sector ETF."""
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


# ==================== LGBM RANKING (walk-forward, sliding window) ====================

def run_lgbm_ranking_wf(sc, idx_end, train_window=60):
    """Walk-forward LGBM ranking using sliding window of data up to idx_end.

    Training data: samples from [idx_end - train_window*20, idx_end) resampled every 20 days.
    This matches the V10 production LGBM pattern.

    Args:
        sc: sector close prices DataFrame
        idx_end: index of the current day (exclusive upper bound for training features)
        train_window: number of rebalance periods back to use for training (~60 monthly samples)

    Returns:
        dict of {ticker: predicted_rank_score}
    """
    if not HAS_LGBM:
        # Fallback: simple 21-day momentum ranking
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    # Build training samples: go back ~400 trading days, sample every 20 days
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

    # Predict current ranks
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


# ==================== BACKTEST ENGINE ====================

def run_variant(sc, spy, vix, variant_key, verbose=False):
    """Run a single equity rotation backtest variant.

    Args:
        sc: sector close prices DataFrame
        spy: SPY close prices Series
        vix: VIX close prices Series
        variant_key: one of A-F
        verbose: print first few rebalances

    Returns dict with equity_curve, trades, metrics, n_rebalances.
    """
    cfg = VARIANT_CONFIG[variant_key]
    n_picks = cfg['n_picks']
    rebal_days = cfg['rebal_days']
    adaptive = cfg['adaptive']

    dates = sc.index
    n_days = len(dates)
    start_idx = 280  # Need 260 days for features + warm-up

    equity = INITIAL_CAPITAL
    equity_curve = []
    holdings = {}  # ticker -> {'shares': float, 'entry_price': float, 'entry_date': date, 'side': 'long', 'entry_idx': int}
    closed_trades = []
    last_rebalance_idx = None
    n_rebalances = 0

    # SPY benchmark
    spy_entry_price = float(spy.iloc[start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # Mark-to-market current holdings
        mtm_pnl = 0.0
        for tk, pos in holdings.items():
            if tk not in sc.columns:
                continue
            current_price = float(sc[tk].iloc[day_idx])
            mtm_pnl += pos['shares'] * (current_price - pos['entry_price'])

        current_equity = INITIAL_CAPITAL + mtm_pnl + sum(t['pnl'] for t in closed_trades)

        # Record daily equity
        spy_equity = spy_shares * float(spy.iloc[day_idx])
        equity_curve.append({
            'date': today,
            'equity': current_equity,
            'spy_equity': spy_equity,
            'n_holdings': len(holdings),
        })

        # ── Determine if we should rebalance today ──
        do_rebalance = False

        if adaptive:
            # Variant F: weekly if VIX>20, monthly if VIX<=20
            current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 15.0
            if current_vix > 20:
                # Weekly: every 5 trading days
                freq = 5
            else:
                # Monthly: every ~21 trading days
                freq = 21

            if last_rebalance_idx is None:
                do_rebalance = True
            elif (day_idx - last_rebalance_idx) >= freq:
                do_rebalance = True
        else:
            # Fixed frequency
            if last_rebalance_idx is None:
                do_rebalance = True
            elif (day_idx - last_rebalance_idx) >= rebal_days:
                do_rebalance = True

        if not do_rebalance:
            continue

        last_rebalance_idx = day_idx
        n_rebalances += 1

        # ── Get LGBM ranking ──
        rankings = run_lgbm_ranking_wf(sc, day_idx)
        if not rankings:
            continue

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        # ── Determine target portfolio: Long top-N, equal weight ──
        top_n = [t for t, _ in ranked[:n_picks]]
        w = 1.0 / len(top_n) if top_n else 0
        target_longs = {tk: w for tk in top_n}

        # ── Close positions not in target ──
        tickers_to_close = []
        for tk, pos in holdings.items():
            if tk not in target_longs:
                tickers_to_close.append(tk)

        for tk in tickers_to_close:
            pos = holdings.pop(tk)
            current_price = float(sc[tk].iloc[day_idx])
            pnl = pos['shares'] * (current_price - pos['entry_price'])
            closed_trades.append({
                'ticker': tk,
                'side': 'long',
                'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                'exit_date': str(today.date()),
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'shares': pos['shares'],
                'pnl': round(pnl, 4),
                'holding_days': day_idx - pos['entry_idx'],
            })

        # ── Open new positions ──
        realized_pnl = sum(t['pnl'] for t in closed_trades)
        available_capital = INITIAL_CAPITAL + realized_pnl

        for tk, weight in target_longs.items():
            if tk in holdings:
                continue  # Already holding
            if tk not in sc.columns:
                continue
            current_price = float(sc[tk].iloc[day_idx])
            alloc = available_capital * weight
            shares = alloc / current_price
            if shares < 0.001:
                continue
            holdings[tk] = {
                'shares': shares,
                'entry_price': current_price,
                'entry_date': today,
                'entry_idx': day_idx,
                'side': 'long',
            }

        if verbose and n_rebalances <= 3:
            longs_str = ', '.join(f"{tk}" for tk in target_longs)
            fprint(f"  Rebalance {n_rebalances} ({today.date()}): "
                   f"LONG [{longs_str}], equity=${current_equity:.2f}")

    # ── Close all remaining positions at end ──
    for tk, pos in holdings.items():
        if tk not in sc.columns:
            continue
        current_price = float(sc[tk].iloc[-1])
        pnl = pos['shares'] * (current_price - pos['entry_price'])
        closed_trades.append({
            'ticker': tk,
            'side': 'long',
            'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
            'exit_date': str(dates[-1].date()),
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'shares': pos['shares'],
            'pnl': round(pnl, 4),
            'holding_days': len(dates) - 1 - pos['entry_idx'],
        })

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) > 0:
        eq_df = eq_df.set_index('date')
        eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

    metrics = compute_metrics(closed_trades, eq_df, variant_key, n_rebalances)

    return {
        'trades': closed_trades,
        'equity_curve': eq_df,
        'metrics': metrics,
        'variant': variant_key,
        'n_rebalances': n_rebalances,
    }


# ==================== METRICS ====================

def compute_metrics(trades, eq_df, variant_name, n_rebalances):
    """Compute comprehensive performance metrics including trades-per-year."""
    if not trades:
        return {
            'variant': variant_name, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0, 'cagr': 0,
            'n_rebalances': n_rebalances, 'trades_per_year': 0,
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
        if daily_rets.std() > 0:
            sharpe = float(daily_rets.mean() / daily_rets.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino
        downside_rets = daily_rets[daily_rets < 0]
        if len(downside_rets) > 1 and downside_rets.std() > 0:
            sortino = float(daily_rets.mean() / downside_rets.std() * np.sqrt(252))
        else:
            sortino = sharpe
    else:
        sharpe = 0.0
        sortino = 0.0

    # Profit Factor
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / (gross_losses + 1e-10)

    # MDD from equity curve
    if len(eq_df) > 0:
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak
        mdd = float(dd.min())
    else:
        mdd = 0

    # Total return and CAGR
    total_return = total_pnl / INITIAL_CAPITAL

    if len(eq_df) > 1:
        n_days_bt = (eq_df.index[-1] - eq_df.index[0]).days
        years = n_days_bt / 365.25
        if years > 0 and (1 + total_return) > 0:
            cagr = (1 + total_return) ** (1.0 / years) - 1
        else:
            cagr = 0
    else:
        years = 1.0
        cagr = 0

    # Trades per year
    trades_per_year = n / years if years > 0 else 0

    # SPY benchmark metrics
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

    metrics = {
        'variant': variant_name,
        'n_trades': n,
        'n_rebalances': n_rebalances,
        'trades_per_year': round(trades_per_year, 1),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr * 100, 1),
        'mdd': round(mdd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2),
        'total_pnl': round(total_pnl, 2),
        'mean_pnl': round(np.mean(pnls), 4),
        'wins': wins,
        'losses': n - wins,
        'spy_total_return': round(spy_total_return * 100, 2),
        'spy_sharpe': round(spy_sharpe, 3),
        'alpha_vs_spy': round(alpha_vs_spy * 100, 2),
    }
    return metrics


# ==================== REGIME BREAKDOWN ====================

def compute_regime_breakdown(trades, spy):
    """Per-regime (green/red/flat) breakdown using SPY close-to-close over holding period."""
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
            'n': n,
            'sharpe': round(sharpe, 2),
            'wr': round(wins / n * 100, 1) if n > 0 else 0,
            'mean_pnl': round(mean_pnl, 4),
            'total_pnl': round(sum(pnls), 2),
        }
    return breakdown


# ==================== 5-GATE VALIDATION ====================

def five_gate_validation(result, spy):
    """5-gate validation per HC #428.

    Gate 1: Sharpe > 1.0
    Gate 2: Permutation p-value < 0.05 (bootstrap test)
    Gate 3: Win rate > 40%
    Gate 4: Regime balance (|Sharpe_green - Sharpe_red| / max < 0.50)
    Gate 5: Monte Carlo 95% CI lower bound > 0
    """
    metrics = result['metrics']
    trades = result['trades']

    gates = {}

    # Gate 1: Sharpe
    gates['sharpe_gt_1'] = {
        'pass': metrics['sharpe'] > 1.0,
        'value': metrics['sharpe'],
        'threshold': 1.0,
    }

    # Gate 2: Permutation test
    if len(trades) >= 10:
        all_pnls = np.array([t['pnl'] for t in trades])
        n_trades = len(all_pnls)
        actual_mean = np.mean(all_pnls)
        n_perm = 5000
        rng = np.random.RandomState(42)
        perm_means = np.zeros(n_perm)
        for i in range(n_perm):
            idx = rng.choice(n_trades, size=n_trades, replace=True)
            perm_means[i] = np.mean(all_pnls[idx])
        if actual_mean > 0:
            p_value = float(np.mean(perm_means <= 0))
        else:
            p_value = 1.0
    else:
        p_value = 1.0
    gates['perm_p_lt_005'] = {
        'pass': p_value < 0.05,
        'value': round(p_value, 4),
        'threshold': 0.05,
    }

    # Gate 3: Win rate > 40%
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

    # Gate 5: Monte Carlo 95% CI lower bound > 0
    if len(trades) >= 10:
        all_pnls = np.array([t['pnl'] for t in trades])
        n_mc = 5000
        rng = np.random.RandomState(123)
        mc_totals = np.zeros(n_mc)
        for i in range(n_mc):
            idx = rng.choice(len(all_pnls), size=len(all_pnls), replace=True)
            mc_totals[i] = np.sum(all_pnls[idx])
        ci_95_lower = float(np.percentile(mc_totals, 2.5))
        ci_95_upper = float(np.percentile(mc_totals, 97.5))
    else:
        ci_95_lower = 0
        ci_95_upper = 0
    gates['mc_ci_positive'] = {
        'pass': ci_95_lower > 0,
        'value': round(ci_95_lower, 2),
        'ci_upper': round(ci_95_upper, 2),
        'threshold': 0.0,
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

    fprint("=" * 70)
    fprint("  EQUITY ROTATION FREQUENCY V1")
    fprint("  Question: Does more frequent rebalancing capture more alpha?")
    fprint("=" * 70)

    # Download data
    fprint("\nDownloading data...")
    sc, spy, vix = download_data()
    fprint(f"Data: {sc.index[0].date()} to {sc.index[-1].date()}, "
           f"{len(sc)} days, {len(sc.columns)} sectors")

    # Check we have enough OOT days
    oot_days = len(sc) - 280
    fprint(f"OOT days available: {oot_days}")
    if oot_days < 40:
        fprint(f"WARNING: Only {oot_days} OOT days (need >= 40). Results may be unreliable.")

    # Run all 6 variants
    variants = list(VARIANT_CONFIG.keys())

    results = {}
    validations = {}

    for v in variants:
        cfg = VARIANT_CONFIG[v]
        fprint(f"\n{'='*60}")
        fprint(f"  VARIANT {v}: {cfg['name']}")
        fprint(f"{'='*60}")
        t_start = time.time()
        results[v] = run_variant(sc, spy, vix, v, verbose=True)
        elapsed_v = time.time() - t_start
        m = results[v]['metrics']
        fprint(f"  Trades: {m['n_trades']} | Rebalances: {m['n_rebalances']} | "
               f"Trades/Yr: {m['trades_per_year']:.1f}")
        fprint(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | "
               f"PF: {m['pf']:.3f} | WR: {m['wr']:.1f}%")
        fprint(f"  MDD: {m['mdd']:.2f}% | Total Return: {m['total_return']:.2f}% | "
               f"CAGR: {m['cagr']:.2f}%")
        fprint(f"  Alpha vs SPY: {m['alpha_vs_spy']:.2f}%")
        fprint(f"  Runtime: {elapsed_v:.1f}s")

        # 5-gate validation
        val = five_gate_validation(results[v], spy)
        validations[v] = val
        fprint(f"  5-Gate Validation: {val['n_pass']}/{val['n_total']} PASS")
        for name, gate in val['gates'].items():
            status = "PASS" if gate['pass'] else "FAIL"
            fprint(f"    {name}: {status} (value={gate['value']}, threshold={gate['threshold']})")

    # ==================== SUMMARY ====================
    fprint("\n" + "=" * 110)
    fprint("  COMPARISON SUMMARY — REBALANCE FREQUENCY")
    fprint("=" * 110)
    fprint(f"{'Variant':<30} {'Trades':>6} {'Tr/Yr':>6} {'Rebal':>6} {'Sharpe':>7} {'Sortino':>8} "
           f"{'PF':>6} {'WR%':>6} {'MDD%':>7} {'TotRet%':>8} {'CAGR%':>7} {'Alpha%':>7} {'Gates':>6}")
    fprint("-" * 110)

    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        cfg = VARIANT_CONFIG[v]
        fprint(f"{v}: {cfg['name']:<27} {m['n_trades']:>5} {m['trades_per_year']:>6.1f} "
               f"{m['n_rebalances']:>5} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
               f"{m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>8.2f} {m['cagr']:>7.2f} {m['alpha_vs_spy']:>7.2f} "
               f"{val['n_pass']:>2}/{val['n_total']}")

    # ── SPY Benchmark Row ──
    spy_m = results['A']['metrics']
    fprint(f"{'SPY Buy-Hold':<37} {'':>5} {'':>6} {'':>5} {spy_m['spy_sharpe']:>7.3f} "
           f"{'':>8} {'':>6} {'':>6} {'':>7} {spy_m['spy_total_return']:>8.2f} {'':>7} {'':>7} {'':>6}")

    # ── Frequency vs Sharpe Analysis ──
    fprint("\n" + "=" * 110)
    fprint("  FREQUENCY vs EFFICIENCY ANALYSIS")
    fprint("=" * 110)
    fprint(f"{'Variant':<30} {'Rebal Freq':>12} {'Trades/Yr':>10} {'Sharpe':>8} {'Sharpe/Trade':>13} {'Verdict':>10}")
    fprint("-" * 110)

    for v in variants:
        m = results[v]['metrics']
        cfg = VARIANT_CONFIG[v]
        if cfg['adaptive']:
            freq_str = "VIX-adaptive"
        elif cfg['rebal_days'] == 5:
            freq_str = "Weekly"
        elif cfg['rebal_days'] == 10:
            freq_str = "Biweekly"
        else:
            freq_str = "Monthly"

        sharpe_per_trade = m['sharpe'] / max(m['trades_per_year'], 1) if m['trades_per_year'] > 0 else 0

        # Verdict: is more frequent better?
        control_sharpe = results['A']['metrics']['sharpe']
        if m['sharpe'] > control_sharpe * 1.05:
            verdict = "BETTER"
        elif m['sharpe'] < control_sharpe * 0.95:
            verdict = "WORSE"
        else:
            verdict = "SIMILAR"

        fprint(f"{v}: {cfg['name']:<27} {freq_str:>12} {m['trades_per_year']:>10.1f} "
               f"{m['sharpe']:>8.3f} {sharpe_per_trade:>13.4f} {verdict:>10}")

    # ── Regime breakdowns ──
    fprint("\n" + "=" * 110)
    fprint("  REGIME BREAKDOWNS (per-trade SPY return during holding period)")
    fprint("=" * 110)

    for v in variants:
        cfg = VARIANT_CONFIG[v]
        bd = validations[v]['regime_breakdown']
        fprint(f"\n  Variant {v} ({cfg['name']}):")
        for regime in ['green', 'red', 'flat']:
            b = bd.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean_PnL=${b.get('mean_pnl',0):8.4f}  "
                   f"total_PnL=${b.get('total_pnl',0):8.2f}")

    # ── N-picks comparison (D vs E vs A) ──
    fprint("\n" + "=" * 110)
    fprint("  CONCENTRATION ANALYSIS (Top-2 vs Top-3 vs Top-4, all monthly)")
    fprint("=" * 110)
    for v in ['A', 'D', 'E']:
        m = results[v]['metrics']
        cfg = VARIANT_CONFIG[v]
        fprint(f"  {v}: {cfg['name']:<25} Sharpe={m['sharpe']:.3f}  MDD={m['mdd']:.2f}%  "
               f"Alpha={m['alpha_vs_spy']:.2f}%  Trades/Yr={m['trades_per_year']:.1f}")

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    best_cfg = VARIANT_CONFIG[best_v]
    fprint(f"\n  BEST VARIANT: {best_v} ({best_cfg['name']}) — Sharpe {best_m['sharpe']:.3f}")

    # ── Key conclusions ──
    fprint("\n" + "=" * 110)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 110)

    any_pass_all = any(validations[v]['all_pass'] for v in variants)
    control_sharpe = results['A']['metrics']['sharpe']
    weekly_sharpe = results['C']['metrics']['sharpe']
    biweekly_sharpe = results['B']['metrics']['sharpe']

    # Frequency verdict
    if weekly_sharpe > control_sharpe * 1.1:
        fprint(f"  FREQUENCY: More frequent rebalancing HELPS. Weekly Sharpe {weekly_sharpe:.3f} vs Monthly {control_sharpe:.3f}")
    elif weekly_sharpe < control_sharpe * 0.9:
        fprint(f"  FREQUENCY: More frequent rebalancing HURTS. Weekly Sharpe {weekly_sharpe:.3f} vs Monthly {control_sharpe:.3f}")
        fprint(f"  Higher frequency = more trades, more tax drag, same or worse risk-adjusted returns.")
    else:
        fprint(f"  FREQUENCY: Rebalance frequency has MINIMAL impact. Weekly {weekly_sharpe:.3f} vs Monthly {control_sharpe:.3f}")

    # Concentration verdict
    top2_sharpe = results['A']['metrics']['sharpe']
    top3_sharpe = results['D']['metrics']['sharpe']
    top4_sharpe = results['E']['metrics']['sharpe']
    if top2_sharpe > max(top3_sharpe, top4_sharpe):
        fprint(f"  CONCENTRATION: Top-2 (concentrated) is BEST. Sharpe: Top-2={top2_sharpe:.3f}, Top-3={top3_sharpe:.3f}, Top-4={top4_sharpe:.3f}")
    elif top3_sharpe > max(top2_sharpe, top4_sharpe):
        fprint(f"  CONCENTRATION: Top-3 is the sweet spot. Sharpe: Top-2={top2_sharpe:.3f}, Top-3={top3_sharpe:.3f}, Top-4={top4_sharpe:.3f}")
    else:
        fprint(f"  CONCENTRATION: Top-4 (diversified) is BEST. Sharpe: Top-2={top2_sharpe:.3f}, Top-3={top3_sharpe:.3f}, Top-4={top4_sharpe:.3f}")

    # Adaptive verdict
    adaptive_sharpe = results['F']['metrics']['sharpe']
    if adaptive_sharpe > control_sharpe * 1.05:
        fprint(f"  ADAPTIVE: VIX-gated rebalancing HELPS. Sharpe {adaptive_sharpe:.3f} vs Monthly {control_sharpe:.3f}")
    else:
        fprint(f"  ADAPTIVE: VIX-gated rebalancing does NOT help. Sharpe {adaptive_sharpe:.3f} vs Monthly {control_sharpe:.3f}")

    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        passing_strs = [f'{v} ({VARIANT_CONFIG[v]["name"]})' for v in passing]
        fprint(f"\n  PASSING ALL 5 GATES: {', '.join(passing_strs)}")
    else:
        fprint(f"\n  NO variant passes all 5 gates.")

    # Transaction cost note
    fprint(f"\n  NOTE: Even at $0 commission, more frequent rebalancing adds tax drag")
    fprint(f"  and execution complexity. Trades-per-year is a key efficiency metric.")
    control_tpy = results['A']['metrics']['trades_per_year']
    weekly_tpy = results['C']['metrics']['trades_per_year']
    fprint(f"  Monthly: {control_tpy:.0f} trades/yr vs Weekly: {weekly_tpy:.0f} trades/yr "
           f"({weekly_tpy/max(control_tpy,1):.1f}x more trading)")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{sc.index[0].date()} to {sc.index[-1].date()}",
        'n_days': len(sc),
        'oot_days': oot_days,
        'initial_capital': INITIAL_CAPITAL,
        'commission': 0.0,
        'variant_configs': {v: {k: str(val) for k, val in cfg.items()} for v, cfg in VARIANT_CONFIG.items()},
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

    results_path = OUTPUT_DIR / "frequency_results.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trades for each variant
    for v in variants:
        trades_path = OUTPUT_DIR / f"trades_variant_{v}.json"
        with open(trades_path, 'w') as f:
            json.dump(results[v]['trades'], f, indent=2, default=str)

    # Save equity curves
    for v in variants:
        eq_path = OUTPUT_DIR / f"equity_curve_{v}.csv"
        results[v]['equity_curve'].to_csv(eq_path)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            run_name = f"freq_rotation_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                # Config params
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("n_days", len(sc))
                mlflow.log_param("oot_days", oot_days)
                mlflow.log_param("data_range", f"{sc.index[0].date()} to {sc.index[-1].date()}")
                mlflow.log_param("best_variant", f"{best_v}_{best_cfg['name']}")

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
                    mlflow.log_metric(f"{prefix}trades_per_year", m['trades_per_year'])
                    mlflow.log_metric(f"{prefix}alpha_vs_spy", m['alpha_vs_spy'])
                    mlflow.log_metric(f"{prefix}gates_pass", val['n_pass'])

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
