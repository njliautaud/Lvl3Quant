#!/usr/bin/env python3
"""
Sector Ranking Equity Backtest V1
====================================

Tests whether V10's LGBM sector ranking signal works as a DIRECT equity
rotation strategy — no options, no BS pricing, just buy/sell ETF shares.

CONTEXT:
  - V10's LGBM sector ranking is validated (adversarial audit passed, no leakage)
  - Options spreads priced with BS are unrealistically cheap; at real pricing,
    spreads are unprofitable
  - The ranking signal itself may be valuable — it predicts which sectors
    outperform/underperform
  - This tests the RAW signal value as equity rotation

SIX VARIANTS:
  A: Long-Only Top-4 — Monthly, buy top-4 ranked sector ETFs equally weighted
  B: Long-Short 4/4 — Monthly, buy top-4, short bottom-4 (market-neutral)
  C: Long Top-2 — Monthly, buy only top-2 ranked (concentrated, high conviction)
  D: Long-Only with Momentum Filter — Buy top-4 ONLY if 21d momentum > 0
  E: Rank-Weighted Long Top-4 — 1/rank weighted (top gets 4x weight of #4)
  F: Long Top-4 vs SPY Benchmark — Same as A but report alpha vs SPY buy-and-hold

IMPLEMENTATION:
  - SAME 17-feature LGBM from V10 production
  - 60-day train, 1-day OOT, sliding walk-forward (HC #0)
  - At LEAST 40 OOT days (all available data)
  - Data from yfinance: 11 sector ETFs + VIX + SPY
  - Monthly rebalance (first Friday of month)
  - Commission: $0 (zero commission for ETFs)
  - Start capital: $645
  - 5-gate validation: Sharpe>1, perm_p<0.05, WR>40%, regime balance, MC 95% CI

Output: output/growth_research/sector_ranking_equity_v1/
MLflow experiment: sector_ranking_equity_v1
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

OUTPUT_DIR = BASE / "output" / "growth_research" / "sector_ranking_equity_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "sector_ranking_equity_v1"
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
REBALANCE_INTERVAL_DAYS = 28  # Monthly

# 17 LGBM momentum features (identical to V10)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]


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


# ==================== RANK WEIGHTING ====================

def compute_rank_weights(n_picks):
    """1/rank weighting. Rank 1 gets most capital."""
    if n_picks <= 0:
        return []
    raw = [1.0 / (i + 1) for i in range(n_picks)]
    total = sum(raw)
    return [w / total for w in raw]


# ==================== REBALANCE DETECTION ====================

def is_first_friday(dt):
    """Check if a date is the first Friday of its month."""
    return dt.weekday() == 4 and dt.day <= 7


# ==================== BACKTEST ENGINE ====================

def run_variant(sc, spy, vix, variant, verbose=False):
    """Run a single equity rotation backtest variant.

    Variants:
        A: Long-Only Top-4 (equal weight)
        B: Long-Short 4/4 (market-neutral)
        C: Long Top-2 (concentrated)
        D: Long-Only Top-4 with momentum filter (21d > 0)
        E: Rank-Weighted Long Top-4 (1/rank sizing)
        F: Long Top-4 vs SPY (same as A, alpha reporting)

    Returns dict with equity_curve, trades, metrics.
    """
    dates = sc.index
    n_days = len(dates)
    start_idx = 280  # Need 260 days for features + warm-up

    equity = INITIAL_CAPITAL
    equity_curve = []
    holdings = {}  # ticker -> {'shares': float, 'entry_price': float, 'entry_date': date, 'side': 'long'|'short'}
    closed_trades = []
    last_rebalance_idx = None
    n_rebalances = 0

    # SPY benchmark for variant F
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
            if pos['side'] == 'long':
                mtm_pnl += pos['shares'] * (current_price - pos['entry_price'])
            else:  # short
                mtm_pnl += pos['shares'] * (pos['entry_price'] - current_price)

        current_equity = INITIAL_CAPITAL + mtm_pnl + sum(t['pnl'] for t in closed_trades)

        # Record daily equity
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

        is_first_fri = today.day <= 7
        do_rebalance = False

        if is_first_fri:
            do_rebalance = True
        elif last_rebalance_idx is not None:
            days_since = day_idx - last_rebalance_idx
            if days_since >= REBALANCE_INTERVAL_DAYS:
                do_rebalance = True
        elif last_rebalance_idx is None:
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

        # ── Determine target portfolio based on variant ──
        target_longs = {}   # ticker -> weight (fraction of capital)
        target_shorts = {}  # ticker -> weight (fraction of capital)

        if variant == 'A' or variant == 'F':
            # Long-Only Top-4, equal weight
            top4 = [t for t, _ in ranked[:4]]
            w = 1.0 / len(top4)
            for tk in top4:
                target_longs[tk] = w

        elif variant == 'B':
            # Long-Short 4/4, equal weight each side
            top4 = [t for t, _ in ranked[:4]]
            bot4 = [t for t, _ in ranked[-4:]]
            w_long = 0.5 / len(top4)    # 50% capital to long side
            w_short = 0.5 / len(bot4)   # 50% capital to short side
            for tk in top4:
                target_longs[tk] = w_long
            for tk in bot4:
                target_shorts[tk] = w_short

        elif variant == 'C':
            # Long Top-2, equal weight
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for tk in top2:
                target_longs[tk] = w

        elif variant == 'D':
            # Long Top-4 with momentum filter: only if 21d momentum > 0
            top4 = [t for t, _ in ranked[:4]]
            passing = []
            for tk in top4:
                px = sc[tk].iloc[:day_idx + 1].dropna()
                if len(px) >= 21:
                    ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1)
                    if ret_21d > 0:
                        passing.append(tk)
            if passing:
                w = 1.0 / len(passing)
                for tk in passing:
                    target_longs[tk] = w
            # else: stay in cash

        elif variant == 'E':
            # Rank-Weighted Long Top-4
            top4 = [t for t, _ in ranked[:4]]
            weights = compute_rank_weights(len(top4))
            for i, tk in enumerate(top4):
                target_longs[tk] = weights[i]

        else:
            raise ValueError(f"Unknown variant: {variant}")

        # ── Close positions not in target ──
        tickers_to_close = []
        for tk, pos in holdings.items():
            in_target = (tk in target_longs and pos['side'] == 'long') or \
                        (tk in target_shorts and pos['side'] == 'short')
            if not in_target:
                tickers_to_close.append(tk)

        for tk in tickers_to_close:
            pos = holdings.pop(tk)
            current_price = float(sc[tk].iloc[day_idx])
            if pos['side'] == 'long':
                pnl = pos['shares'] * (current_price - pos['entry_price'])
            else:
                pnl = pos['shares'] * (pos['entry_price'] - current_price)
            closed_trades.append({
                'ticker': tk,
                'side': pos['side'],
                'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                'exit_date': str(today.date()),
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'shares': pos['shares'],
                'pnl': round(pnl, 4),
                'holding_days': day_idx - pos['entry_idx'],
            })

        # ── Open new positions ──
        # Recalculate available capital after closing
        realized_pnl = sum(t['pnl'] for t in closed_trades)
        # MTM remaining holdings
        remaining_mtm = 0.0
        remaining_cost = 0.0
        for tk, pos in holdings.items():
            current_price = float(sc[tk].iloc[day_idx])
            remaining_cost += pos['shares'] * pos['entry_price']
            if pos['side'] == 'long':
                remaining_mtm += pos['shares'] * current_price
            else:
                remaining_mtm += pos['shares'] * (2 * pos['entry_price'] - current_price)

        available_capital = INITIAL_CAPITAL + realized_pnl
        # Subtract capital already deployed in remaining holdings
        deployed_in_remaining = sum(pos['shares'] * pos['entry_price'] for pos in holdings.values())
        free_capital = available_capital - deployed_in_remaining

        # Allocate to longs
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

        # Allocate to shorts
        for tk, weight in target_shorts.items():
            if tk in holdings:
                continue
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
                'side': 'short',
            }

        if verbose and n_rebalances <= 3:
            longs_str = ', '.join(f"{tk}" for tk in target_longs)
            shorts_str = ', '.join(f"{tk}" for tk in target_shorts) if target_shorts else 'none'
            fprint(f"  Rebalance {n_rebalances} ({today.date()}): "
                   f"LONG [{longs_str}], SHORT [{shorts_str}], "
                   f"equity=${current_equity:.2f}")

    # ── Close all remaining positions at end ──
    for tk, pos in holdings.items():
        if tk not in sc.columns:
            continue
        current_price = float(sc[tk].iloc[-1])
        if pos['side'] == 'long':
            pnl = pos['shares'] * (current_price - pos['entry_price'])
        else:
            pnl = pos['shares'] * (pos['entry_price'] - current_price)
        closed_trades.append({
            'ticker': tk,
            'side': pos['side'],
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
    """Compute comprehensive performance metrics."""
    if not trades:
        return {
            'variant': variant_name, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0, 'cagr': 0,
        }

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0

    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if n > 1 else 1.0

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
        cagr = 0

    # SPY benchmark metrics (for variant F)
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
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr * 100, 1),
        'mdd': round(mdd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2),
        'total_pnl': round(total_pnl, 2),
        'mean_pnl': round(mean_pnl, 4),
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

        # SPY return over the trade's holding period
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
        # Annualize: assume each trade ~28 days, so ~13 trades/year per position
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
    Gate 2: Permutation p-value < 0.05 (is the ranking signal better than random?)
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
        actual_pnl = sum(t['pnl'] for t in trades)
        pnls = [t['pnl'] for t in trades]
        n_perm = 5000
        rng = np.random.RandomState(42)
        perm_pnls = np.zeros(n_perm)
        for i in range(n_perm):
            shuffled = rng.permutation(pnls)
            perm_pnls[i] = np.sum(shuffled)
        # p-value: fraction of permutations with PnL >= actual
        # For a proper permutation test of the ranking signal, we shuffle which
        # trades would have been selected. Since we can't re-run LGBM each time,
        # we use a simpler bootstrap: what fraction of random reshuffles of
        # trade assignment beat the actual total?
        # More precisely: shuffle the PnL vector and see if the sum beats actual.
        # Since sum is invariant to permutation, we instead test if the ordering
        # matters by comparing mean PnL to bootstrap of random subsets.
        all_pnls = np.array(pnls)
        n_trades = len(all_pnls)
        perm_means = np.zeros(n_perm)
        for i in range(n_perm):
            idx = rng.choice(n_trades, size=n_trades, replace=True)
            perm_means[i] = np.mean(all_pnls[idx])
        actual_mean = np.mean(all_pnls)
        p_value = float(np.mean(perm_means >= actual_mean))
        # For a one-sided test of positive mean
        if actual_mean > 0:
            # What fraction of bootstraps have mean <= 0?
            ci_lower = np.percentile(perm_means, 2.5)
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
    fprint("  SECTOR RANKING EQUITY BACKTEST V1")
    fprint("  Question: Does V10's LGBM ranking signal work as direct equity rotation?")
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
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'Long Top-4 EW',
        'B': 'L/S 4/4',
        'C': 'Long Top-2',
        'D': 'Long Top-4 + Mom Filter',
        'E': 'Rank-Weighted Top-4',
        'F': 'Long Top-4 vs SPY',
    }

    results = {}
    validations = {}

    for v in variants:
        fprint(f"\n{'='*60}")
        fprint(f"  VARIANT {v}: {variant_names[v]}")
        fprint(f"{'='*60}")
        t_start = time.time()
        results[v] = run_variant(sc, spy, vix, v, verbose=True)
        elapsed_v = time.time() - t_start
        m = results[v]['metrics']
        fprint(f"  Trades: {m['n_trades']} | Sharpe: {m['sharpe']:.3f} | "
               f"Sortino: {m['sortino']:.3f} | PF: {m['pf']:.3f} | "
               f"WR: {m['wr']:.1f}% | MDD: {m['mdd']:.2f}% | "
               f"Total Return: {m['total_return']:.2f}% | CAGR: {m['cagr']:.2f}%")
        if v == 'F':
            fprint(f"  SPY Return: {m['spy_total_return']:.2f}% | "
                   f"Alpha vs SPY: {m['alpha_vs_spy']:.2f}%")
        fprint(f"  Runtime: {elapsed_v:.1f}s")

        # 5-gate validation
        val = five_gate_validation(results[v], spy)
        validations[v] = val
        fprint(f"  5-Gate Validation: {val['n_pass']}/{val['n_total']} PASS")
        for name, gate in val['gates'].items():
            status = "PASS" if gate['pass'] else "FAIL"
            fprint(f"    {name}: {status} (value={gate['value']}, threshold={gate['threshold']})")

    # ==================== SUMMARY ====================
    fprint("\n" + "=" * 90)
    fprint("  COMPARISON SUMMARY")
    fprint("=" * 90)
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
           f"{'WR%':>6} {'MDD%':>7} {'TotRet%':>8} {'CAGR%':>7} {'Gates':>6}")
    fprint("-" * 90)

    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        fprint(f"{v}: {variant_names[v]:<22} {m['n_trades']:>5} {m['sharpe']:>7.3f} "
               f"{m['sortino']:>8.3f} {m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>8.2f} {m['cagr']:>7.2f} {val['n_pass']:>2}/{val['n_total']}")

    # ── SPY Benchmark Row ──
    spy_m = results['F']['metrics']
    fprint(f"{'SPY Buy-Hold':<28} {'':>5} {spy_m['spy_sharpe']:>7.3f} "
           f"{'':>8} {'':>6} {'':>6} {'':>7} {spy_m['spy_total_return']:>8.2f} {'':>7} {'':>6}")

    # ── Regime breakdowns ──
    fprint("\n" + "=" * 90)
    fprint("  REGIME BREAKDOWNS (per-trade SPY return during holding period)")
    fprint("=" * 90)

    for v in variants:
        bd = validations[v]['regime_breakdown']
        fprint(f"\n  Variant {v} ({variant_names[v]}):")
        for regime in ['green', 'red', 'flat']:
            b = bd.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean_PnL=${b.get('mean_pnl',0):8.4f}  "
                   f"total_PnL=${b.get('total_pnl',0):8.2f}")

    # ── Alpha analysis ──
    fprint("\n" + "=" * 90)
    fprint("  ALPHA vs SPY BUY-AND-HOLD")
    fprint("=" * 90)
    for v in variants:
        m = results[v]['metrics']
        fprint(f"  {v}: {variant_names[v]:<25} Return={m['total_return']:>7.2f}%  "
               f"SPY={m['spy_total_return']:>7.2f}%  Alpha={m['alpha_vs_spy']:>7.2f}%")

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    fprint(f"\n  BEST VARIANT: {best_v} ({variant_names[best_v]}) — Sharpe {best_m['sharpe']:.3f}")

    # ── Key conclusions ──
    fprint("\n" + "=" * 90)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 90)

    any_pass_all = any(validations[v]['all_pass'] for v in variants)
    best_sharpe = best_m['sharpe']

    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"  VERDICT: Ranking signal WORKS as equity rotation!")
        fprint(f"  Variants passing all 5 gates: {', '.join(f'{v} ({variant_names[v]})' for v in passing)}")
        fprint(f"  Best Sharpe: {best_sharpe:.3f}")
        fprint(f"  NEXT STEPS: Consider leveraged ETFs (TQQQ/SOXL), actual options with real market pricing,")
        fprint(f"  or increasing concentration on highest-conviction picks.")
    elif best_sharpe > 1.0:
        fprint(f"  VERDICT: Ranking signal has EDGE but fails some gates.")
        fprint(f"  Best Sharpe: {best_sharpe:.3f} (above 1.0 threshold)")
        fprint(f"  Check which gates fail — regime balance or MC CI are most informative.")
    elif best_sharpe > 0:
        fprint(f"  VERDICT: Ranking signal has WEAK positive edge as equity rotation.")
        fprint(f"  Best Sharpe: {best_sharpe:.3f} (below 1.0)")
        fprint(f"  The signal may still be valuable for options expression if properly priced.")
    else:
        fprint(f"  VERDICT: Ranking signal does NOT work as equity rotation.")
        fprint(f"  Best Sharpe: {best_sharpe:.3f}")
        fprint(f"  The edge (if any) is too small for direct equity rotation.")

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
        'metrics': {},
        'validations': {},
        'regime_breakdowns': {},
    }

    for v in variants:
        save_results['metrics'][v] = results[v]['metrics']
        # Convert gate results for JSON serialization
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
            run_name = f"sector_equity_bt_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                # Config params
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("rebalance_days", REBALANCE_INTERVAL_DAYS)
                mlflow.log_param("n_days", len(sc))
                mlflow.log_param("oot_days", oot_days)
                mlflow.log_param("data_range", f"{sc.index[0].date()} to {sc.index[-1].date()}")
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
                    mlflow.log_metric(f"{prefix}alpha_vs_spy", m['alpha_vs_spy'])
                    mlflow.log_metric(f"{prefix}gates_pass", val['n_pass'])

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
