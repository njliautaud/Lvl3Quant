#!/usr/bin/env python3
"""
Leveraged ETF Rotation V1
============================

Tests whether the sector ranking signal can be amplified using leveraged sector ETFs.

CONTEXT:
  - LGBM sector ranking model passed adversarial validation, produces real alpha
  - Pure equity rotation (top-2 monthly) gives Sharpe 1.40, +17.5% alpha vs SPY
  - Options-based approaches fail due to BS pricing underestimation
  - Leveraged ETFs offer 2-3x exposure WITHOUT options pricing complexity

WARNING: Leveraged ETFs have VOLATILITY DECAY (beta slippage). A 2x ETF does NOT
deliver 2x the return of the underlying over long periods. Daily rebalancing of
leverage causes a drag that increases with volatility. This script documents any
decay observed by comparing leveraged returns to theoretical leveraged returns.

8 VARIANTS:
  A: 2x Leveraged Top-2 — Buy top-2 ranked sectors using ProShares Ultra ETFs (2x)
  B: 3x Leveraged Top-2 — Buy top-2 using Direxion 3x ETFs
  C: 2x Leveraged Top-1 — Most concentrated: top-1 pick with 2x leverage
  D: Inverse Bottom-2 — Short bottom-2 sectors using ProShares Short ETFs (-1x)
  E: Long 2x Top-2 + Short Bottom-2 — Combine bullish 2x with bearish -1x
  F: 3x Top-1 with SMA filter — Buy 3x top-1 ONLY if above 50-day SMA, cash otherwise
  G: TQQQ/SOXL rotation — Rotate between tech-heavy leveraged ETFs based on sector rank
  H: Unleveraged Top-2 (control) — Same as validated equity rotation for comparison

IMPLEMENTATION:
  - SAME 17-feature LGBM from V10 production (rank PARENT 1x sector ETFs, then
    buy the leveraged version)
  - 60-day train, 1-day OOT, sliding walk-forward
  - All available OOT days
  - Monthly rebalance
  - Commission: $0
  - Start capital: $645

Output: output/growth_research/leveraged_etf_rotation_v1/
MLflow experiment: leveraged_etf_rotation_v1
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

OUTPUT_DIR = BASE / "output" / "growth_research" / "leveraged_etf_rotation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "leveraged_etf_rotation_v1"
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

# ==================== LEVERAGED ETF MAPPING ====================
LEVERAGED_MAP = {
    'XLK': {'2x': 'ROM', '3x': 'TECL', '-1x': 'REK'},
    'XLF': {'2x': 'UYG', '3x': 'FAS', '-1x': 'SKF'},
    'XLE': {'2x': 'DIG', '3x': 'ERX', '-1x': 'DDG'},
    'XLI': {'2x': 'UXI', '3x': None, '-1x': 'SIJ'},
    'XLV': {'2x': 'RXL', '3x': 'LABU', '-1x': 'RXD'},
    'XLU': {'2x': 'UPW', '3x': None, '-1x': 'SDP'},
    'XLY': {'2x': 'UCC', '3x': None, '-1x': 'SCC'},
    'XLP': {'2x': None, '3x': None, '-1x': None},
    'XLB': {'2x': 'UYM', '3x': None, '-1x': 'SMN'},
    'XLRE': {'2x': 'URE', '3x': 'DRN', '-1x': 'SRS'},
    'XLC': {'2x': None, '3x': None, '-1x': None},
}

# Tech-heavy leveraged ETFs for variant G
TECH_LEVERAGED = {
    'TQQQ': 'XLK',   # TQQQ tracks QQQ, closest sector = XLK
    'SOXL': 'XLK',   # SOXL tracks semiconductors, subset of XLK
    'SPXL': 'SPY',   # SPXL tracks SPY (broad market)
}

# Broad leveraged for reference
BROAD_LEVERAGED = {
    'SPY': {'2x': 'SSO', '3x': 'SPXL'},
    'QQQ': {'2x': 'QLD', '3x': 'TQQQ'},
}


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETFs + leveraged ETFs + SPY + VIX data via yfinance."""
    import yfinance as yf

    # Collect all leveraged tickers we might need
    lev_tickers = set()
    for sector, mapping in LEVERAGED_MAP.items():
        for mult, ticker in mapping.items():
            if ticker is not None:
                lev_tickers.add(ticker)
    for ticker in TECH_LEVERAGED.keys():
        lev_tickers.add(ticker)
    for _, mapping in BROAD_LEVERAGED.items():
        for _, ticker in mapping.items():
            lev_tickers.add(ticker)

    all_tickers = SECTORS + ['SPY', 'QQQ', '^VIX'] + sorted(lev_tickers)
    fprint(f"Downloading {len(all_tickers)} tickers ({len(lev_tickers)} leveraged)...")
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

    # Sector close prices (1x only, for ranking)
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)

    # All close prices (including leveraged)
    all_close = close.loc[ix]

    # Report which leveraged ETFs are available
    available_lev = [t for t in lev_tickers if t in all_close.columns and all_close[t].notna().sum() > 50]
    missing_lev = [t for t in lev_tickers if t not in available_lev]
    fprint(f"Leveraged ETFs available: {len(available_lev)}/{len(lev_tickers)}")
    if missing_lev:
        fprint(f"  Missing/insufficient data: {', '.join(sorted(missing_lev))}")

    return sc.loc[ix], spy.loc[ix], vix.loc[ix], all_close


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

    IMPORTANT: Rankings are computed on the PARENT 1x sector ETFs. The leveraged
    version is only used for trade execution (buying/selling). This ensures the
    signal quality is not contaminated by leveraged ETF volatility decay.

    Returns:
        dict of {ticker: predicted_rank_score}
    """
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


# ==================== LEVERAGED TICKER RESOLUTION ====================

def resolve_leveraged_ticker(parent_sector, leverage_type, all_close):
    """Resolve the leveraged ticker for a parent sector.

    Args:
        parent_sector: e.g., 'XLK'
        leverage_type: '2x', '3x', or '-1x'
        all_close: DataFrame with all close prices

    Returns:
        (ticker, is_leveraged) — the leveraged ticker if available, else the parent
    """
    if parent_sector not in LEVERAGED_MAP:
        return parent_sector, False

    lev_ticker = LEVERAGED_MAP[parent_sector].get(leverage_type)
    if lev_ticker is None:
        return parent_sector, False

    if lev_ticker in all_close.columns and all_close[lev_ticker].notna().sum() > 50:
        return lev_ticker, True

    return parent_sector, False


def get_sma(series, window=50):
    """Compute simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


# ==================== BACKTEST ENGINE ====================

def run_variant(sc, spy, vix, all_close, variant, verbose=False):
    """Run a single leveraged ETF rotation backtest variant.

    Key design: Rankings are ALWAYS computed on 1x parent sector ETFs. Only the
    trade execution uses leveraged tickers. This isolates the signal from leverage
    effects.

    Returns dict with equity_curve, trades, metrics, decay_analysis.
    """
    dates = sc.index
    n_days = len(dates)
    start_idx = 280  # Need 260 days for features + warm-up

    equity = INITIAL_CAPITAL
    equity_curve = []
    holdings = {}  # ticker -> {'shares', 'entry_price', 'entry_date', 'entry_idx', 'side', 'parent_sector', 'leverage_type'}
    closed_trades = []
    last_rebalance_idx = None
    n_rebalances = 0
    fallback_count = 0  # How many times we fell back to 1x

    # SPY benchmark
    spy_entry_price = float(spy.iloc[start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # Mark-to-market current holdings
        mtm_pnl = 0.0
        for tk, pos in holdings.items():
            if tk not in all_close.columns:
                continue
            current_price = float(all_close[tk].iloc[day_idx])
            if pd.isna(current_price):
                continue
            if pos['side'] == 'long':
                mtm_pnl += pos['shares'] * (current_price - pos['entry_price'])
            else:  # short (inverse ETFs are bought long, but represent short exposure)
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

        # ── Get LGBM ranking (ALWAYS on 1x parent sectors) ──
        rankings = run_lgbm_ranking_wf(sc, day_idx)
        if not rankings:
            continue

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        # ── Determine target portfolio based on variant ──
        target_positions = []  # list of (trade_ticker, weight, side, parent_sector, leverage_type)

        if variant == 'A':
            # 2x Leveraged Top-2
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for parent in top2:
                lev_tk, is_lev = resolve_leveraged_ticker(parent, '2x', all_close)
                if not is_lev:
                    fallback_count += 1
                target_positions.append((lev_tk, w, 'long', parent, '2x'))

        elif variant == 'B':
            # 3x Leveraged Top-2
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for parent in top2:
                lev_tk, is_lev = resolve_leveraged_ticker(parent, '3x', all_close)
                if not is_lev:
                    # Try 2x as second fallback
                    lev_tk, is_lev = resolve_leveraged_ticker(parent, '2x', all_close)
                    if not is_lev:
                        fallback_count += 1
                target_positions.append((lev_tk, w, 'long', parent, '3x'))

        elif variant == 'C':
            # 2x Leveraged Top-1 (most concentrated)
            top1_parent = ranked[0][0]
            lev_tk, is_lev = resolve_leveraged_ticker(top1_parent, '2x', all_close)
            if not is_lev:
                fallback_count += 1
            target_positions.append((lev_tk, 1.0, 'long', top1_parent, '2x'))

        elif variant == 'D':
            # Inverse Bottom-2 (buy -1x inverse ETFs of bottom-2 sectors)
            bot2 = [t for t, _ in ranked[-2:]]
            w = 1.0 / len(bot2)
            for parent in bot2:
                lev_tk, is_lev = resolve_leveraged_ticker(parent, '-1x', all_close)
                if not is_lev:
                    fallback_count += 1
                    continue  # Skip if no inverse ETF available
                # Inverse ETFs are BOUGHT (long position = short exposure)
                target_positions.append((lev_tk, w, 'long', parent, '-1x'))

        elif variant == 'E':
            # Long 2x Top-2 + Short Bottom-2 (via inverse ETFs)
            top2 = [t for t, _ in ranked[:2]]
            bot2 = [t for t, _ in ranked[-2:]]
            w_long = 0.5 / max(len(top2), 1)
            w_short = 0.5 / max(len(bot2), 1)
            for parent in top2:
                lev_tk, is_lev = resolve_leveraged_ticker(parent, '2x', all_close)
                if not is_lev:
                    fallback_count += 1
                target_positions.append((lev_tk, w_long, 'long', parent, '2x'))
            for parent in bot2:
                lev_tk, is_lev = resolve_leveraged_ticker(parent, '-1x', all_close)
                if not is_lev:
                    fallback_count += 1
                    continue
                target_positions.append((lev_tk, w_short, 'long', parent, '-1x'))

        elif variant == 'F':
            # 3x Top-1 with SMA filter — buy ONLY if above 50-day SMA
            top1_parent = ranked[0][0]
            # Check SMA on the PARENT sector
            parent_px = sc[top1_parent].iloc[:day_idx + 1].dropna()
            if len(parent_px) >= 50:
                sma_50 = parent_px.iloc[-50:].mean()
                current_px = parent_px.iloc[-1]
                if current_px > sma_50:
                    lev_tk, is_lev = resolve_leveraged_ticker(top1_parent, '3x', all_close)
                    if not is_lev:
                        lev_tk, is_lev = resolve_leveraged_ticker(top1_parent, '2x', all_close)
                        if not is_lev:
                            fallback_count += 1
                    target_positions.append((lev_tk, 1.0, 'long', top1_parent, '3x'))
                # else: stay in cash (no positions)

        elif variant == 'G':
            # TQQQ/SOXL/SPXL rotation — pick the one whose parent sector ranks highest
            # Map each tech leveraged ETF to a rank score
            tech_scores = {}
            for lev_tk, parent_approx in TECH_LEVERAGED.items():
                if lev_tk not in all_close.columns:
                    continue
                if all_close[lev_tk].iloc[:day_idx + 1].notna().sum() < 50:
                    continue
                # Get score for the closest parent sector
                if parent_approx in rankings:
                    tech_scores[lev_tk] = rankings[parent_approx]
                elif parent_approx == 'SPY':
                    # Use average of all sector scores for SPY proxy
                    tech_scores[lev_tk] = np.mean(list(rankings.values()))

            if tech_scores:
                best_lev = max(tech_scores, key=tech_scores.get)
                target_positions.append((best_lev, 1.0, 'long', 'TECH', '3x'))

        elif variant == 'H':
            # Unleveraged Top-2 (control) — identical to validated equity rotation
            top2 = [t for t, _ in ranked[:2]]
            w = 1.0 / len(top2)
            for parent in top2:
                target_positions.append((parent, w, 'long', parent, '1x'))

        else:
            raise ValueError(f"Unknown variant: {variant}")

        # ── Close positions not in target ──
        target_tickers = {pos[0] for pos in target_positions}
        tickers_to_close = [tk for tk in holdings if tk not in target_tickers]

        for tk in tickers_to_close:
            pos = holdings.pop(tk)
            if tk not in all_close.columns:
                continue
            current_price = float(all_close[tk].iloc[day_idx])
            if pd.isna(current_price):
                current_price = pos['entry_price']  # flat
            pnl = pos['shares'] * (current_price - pos['entry_price'])
            closed_trades.append({
                'ticker': tk,
                'parent_sector': pos.get('parent_sector', tk),
                'leverage_type': pos.get('leverage_type', '1x'),
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
        realized_pnl = sum(t['pnl'] for t in closed_trades)
        available_capital = INITIAL_CAPITAL + realized_pnl

        # Subtract capital deployed in remaining holdings
        deployed = sum(pos['shares'] * pos['entry_price'] for pos in holdings.values())
        free_capital = available_capital - deployed

        for trade_tk, weight, side, parent, lev_type in target_positions:
            if trade_tk in holdings:
                continue
            if trade_tk not in all_close.columns:
                continue
            current_price = float(all_close[trade_tk].iloc[day_idx])
            if pd.isna(current_price) or current_price <= 0:
                continue
            alloc = available_capital * weight
            shares = alloc / current_price
            if shares < 0.001:
                continue
            holdings[trade_tk] = {
                'shares': shares,
                'entry_price': current_price,
                'entry_date': today,
                'entry_idx': day_idx,
                'side': side,
                'parent_sector': parent,
                'leverage_type': lev_type,
            }

        if verbose and n_rebalances <= 3:
            pos_str = ', '.join(f"{tk}({holdings[tk].get('leverage_type','1x')})" for tk in holdings)
            fprint(f"  Rebalance {n_rebalances} ({today.date()}): [{pos_str}], "
                   f"equity=${current_equity:.2f}")

    # ── Close all remaining positions at end ──
    for tk, pos in list(holdings.items()):
        if tk not in all_close.columns:
            continue
        current_price = float(all_close[tk].iloc[-1])
        if pd.isna(current_price):
            current_price = pos['entry_price']
        pnl = pos['shares'] * (current_price - pos['entry_price'])
        closed_trades.append({
            'ticker': tk,
            'parent_sector': pos.get('parent_sector', tk),
            'leverage_type': pos.get('leverage_type', '1x'),
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

    # Volatility decay analysis
    decay_info = analyze_volatility_decay(closed_trades, sc, all_close, variant)

    return {
        'trades': closed_trades,
        'equity_curve': eq_df,
        'metrics': metrics,
        'variant': variant,
        'n_rebalances': n_rebalances,
        'fallback_count': fallback_count,
        'decay_analysis': decay_info,
    }


# ==================== VOLATILITY DECAY ANALYSIS ====================

def analyze_volatility_decay(trades, sc, all_close, variant):
    """Analyze volatility decay in leveraged ETF trades.

    Compares actual leveraged ETF return to the theoretical return
    (leverage_factor * parent_return) for each trade.
    """
    if variant == 'H':
        return {'has_decay_data': False, 'reason': 'unleveraged control'}

    leverage_factors = {'2x': 2.0, '3x': 3.0, '-1x': -1.0, '1x': 1.0}
    decay_records = []

    for t in trades:
        parent = t.get('parent_sector')
        lev_type = t.get('leverage_type', '1x')
        ticker = t['ticker']

        if parent is None or parent == ticker or parent == 'TECH':
            continue
        if lev_type == '1x':
            continue
        if parent not in sc.columns:
            continue

        entry_date = pd.Timestamp(t['entry_date'])
        exit_date = pd.Timestamp(t['exit_date'])

        # Parent return over the period
        parent_mask = (sc.index >= entry_date) & (sc.index <= exit_date)
        parent_px = sc[parent][parent_mask]
        if len(parent_px) < 2:
            continue

        parent_return = float(parent_px.iloc[-1] / parent_px.iloc[0] - 1)
        actual_return = float(t['exit_price'] / t['entry_price'] - 1)
        lev_factor = leverage_factors.get(lev_type, 1.0)
        theoretical_return = lev_factor * parent_return
        decay = actual_return - theoretical_return

        decay_records.append({
            'ticker': ticker,
            'parent': parent,
            'leverage_type': lev_type,
            'holding_days': t['holding_days'],
            'parent_return': round(parent_return * 100, 2),
            'theoretical_return': round(theoretical_return * 100, 2),
            'actual_return': round(actual_return * 100, 2),
            'decay_pct': round(decay * 100, 2),
        })

    if not decay_records:
        return {'has_decay_data': False, 'reason': 'no leveraged trades with parent data'}

    decays = [r['decay_pct'] for r in decay_records]
    return {
        'has_decay_data': True,
        'n_trades': len(decay_records),
        'mean_decay_pct': round(np.mean(decays), 3),
        'median_decay_pct': round(np.median(decays), 3),
        'worst_decay_pct': round(min(decays), 3),
        'best_decay_pct': round(max(decays), 3),
        'std_decay_pct': round(np.std(decays), 3),
        'details': decay_records,
    }


# ==================== METRICS ====================

def compute_metrics(trades, eq_df, variant_name):
    """Compute comprehensive performance metrics."""
    if not trades:
        return {
            'variant': variant_name, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0, 'cagr': 0,
            'max_daily_loss': 0,
        }

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0

    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls)

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

        # Max daily loss
        max_daily_loss = float(daily_rets.min()) * 100
    else:
        sharpe = 0.0
        sortino = 0.0
        max_daily_loss = 0.0

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
        'max_daily_loss': round(max_daily_loss, 2),
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
    Gate 2: Permutation p-value < 0.05
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
    fprint("  LEVERAGED ETF ROTATION V1")
    fprint("  Question: Can sector ranking signal be amplified with leveraged ETFs?")
    fprint("=" * 70)
    fprint("  WARNING: Leveraged ETFs have volatility decay (beta slippage).")
    fprint("  A 2x ETF does NOT deliver 2x returns over longer periods.")
    fprint("=" * 70)

    # Download data
    fprint("\nDownloading data...")
    sc, spy, vix, all_close = download_data()
    fprint(f"Data: {sc.index[0].date()} to {sc.index[-1].date()}, "
           f"{len(sc)} days, {len(sc.columns)} sectors")

    oot_days = len(sc) - 280
    fprint(f"OOT days available: {oot_days}")
    if oot_days < 40:
        fprint(f"WARNING: Only {oot_days} OOT days (need >= 40). Results may be unreliable.")

    # Run all 8 variants
    variants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    variant_names = {
        'A': '2x Lev Top-2',
        'B': '3x Lev Top-2',
        'C': '2x Lev Top-1',
        'D': 'Inverse Bottom-2',
        'E': '2x Top-2 + Inv Bot-2',
        'F': '3x Top-1 + SMA Filter',
        'G': 'TQQQ/SOXL Rotation',
        'H': 'Unleveraged Top-2 (ctrl)',
    }

    results = {}
    validations = {}

    for v in variants:
        fprint(f"\n{'='*60}")
        fprint(f"  VARIANT {v}: {variant_names[v]}")
        fprint(f"{'='*60}")
        t_start = time.time()
        results[v] = run_variant(sc, spy, vix, all_close, v, verbose=True)
        elapsed_v = time.time() - t_start
        m = results[v]['metrics']
        fprint(f"  Trades: {m['n_trades']} | Sharpe: {m['sharpe']:.3f} | "
               f"Sortino: {m['sortino']:.3f} | PF: {m['pf']:.3f} | "
               f"WR: {m['wr']:.1f}% | MDD: {m['mdd']:.2f}% | "
               f"Total Return: {m['total_return']:.2f}% | CAGR: {m['cagr']:.2f}%")
        fprint(f"  Max Daily Loss: {m['max_daily_loss']:.2f}% | "
               f"Alpha vs SPY: {m['alpha_vs_spy']:.2f}%")
        if results[v]['fallback_count'] > 0:
            fprint(f"  Leveraged ETF fallbacks to 1x: {results[v]['fallback_count']}")
        fprint(f"  Runtime: {elapsed_v:.1f}s")

        # Volatility decay analysis
        decay = results[v]['decay_analysis']
        if decay.get('has_decay_data'):
            fprint(f"  VOLATILITY DECAY: mean={decay['mean_decay_pct']:.3f}%, "
                   f"median={decay['median_decay_pct']:.3f}%, "
                   f"worst={decay['worst_decay_pct']:.3f}%, "
                   f"n={decay['n_trades']}")

        # 5-gate validation
        val = five_gate_validation(results[v], spy)
        validations[v] = val
        fprint(f"  5-Gate Validation: {val['n_pass']}/{val['n_total']} PASS")
        for name, gate in val['gates'].items():
            status = "PASS" if gate['pass'] else "FAIL"
            fprint(f"    {name}: {status} (value={gate['value']}, threshold={gate['threshold']})")

    # ==================== SUMMARY ====================
    fprint("\n" + "=" * 100)
    fprint("  COMPARISON SUMMARY")
    fprint("=" * 100)
    fprint(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
           f"{'WR%':>6} {'MDD%':>7} {'TotRet%':>8} {'CAGR%':>7} {'MaxDL%':>7} {'Gates':>6}")
    fprint("-" * 100)

    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        fprint(f"{v}: {variant_names[v]:<27} {m['n_trades']:>5} {m['sharpe']:>7.3f} "
               f"{m['sortino']:>8.3f} {m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>8.2f} {m['cagr']:>7.2f} {m['max_daily_loss']:>7.2f} "
               f"{val['n_pass']:>2}/{val['n_total']}")

    # ── SPY Benchmark Row ──
    spy_m = results['H']['metrics']
    fprint(f"{'SPY Buy-Hold':<33} {'':>5} {spy_m['spy_sharpe']:>7.3f} "
           f"{'':>8} {'':>6} {'':>6} {'':>7} {spy_m['spy_total_return']:>8.2f} {'':>7} {'':>7} {'':>6}")

    # ── Leverage amplification analysis ──
    fprint("\n" + "=" * 100)
    fprint("  LEVERAGE AMPLIFICATION ANALYSIS (vs Control H)")
    fprint("=" * 100)
    ctrl_sharpe = results['H']['metrics']['sharpe']
    ctrl_ret = results['H']['metrics']['total_return']
    ctrl_mdd = results['H']['metrics']['mdd']

    for v in variants:
        if v == 'H':
            continue
        m = results[v]['metrics']
        sharpe_ratio = m['sharpe'] / ctrl_sharpe if ctrl_sharpe != 0 else 0
        ret_ratio = m['total_return'] / ctrl_ret if ctrl_ret != 0 else 0
        mdd_ratio = m['mdd'] / ctrl_mdd if ctrl_mdd != 0 else 0
        fprint(f"  {v}: {variant_names[v]:<27} Sharpe ratio: {sharpe_ratio:.2f}x  "
               f"Return ratio: {ret_ratio:.2f}x  MDD ratio: {mdd_ratio:.2f}x")

    # ── Volatility Decay Summary ──
    fprint("\n" + "=" * 100)
    fprint("  VOLATILITY DECAY SUMMARY")
    fprint("=" * 100)
    for v in variants:
        decay = results[v]['decay_analysis']
        if decay.get('has_decay_data'):
            fprint(f"  {v}: {variant_names[v]:<27} Mean decay: {decay['mean_decay_pct']:>+.3f}%  "
                   f"Median: {decay['median_decay_pct']:>+.3f}%  "
                   f"Worst: {decay['worst_decay_pct']:>+.3f}%  "
                   f"n={decay['n_trades']}")
        else:
            reason = decay.get('reason', 'N/A')
            fprint(f"  {v}: {variant_names[v]:<27} No decay data ({reason})")

    # ── Regime breakdowns ──
    fprint("\n" + "=" * 100)
    fprint("  REGIME BREAKDOWNS (per-trade SPY return during holding period)")
    fprint("=" * 100)

    for v in variants:
        bd = validations[v]['regime_breakdown']
        fprint(f"\n  Variant {v} ({variant_names[v]}):")
        for regime in ['green', 'red', 'flat']:
            b = bd.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean_PnL=${b.get('mean_pnl',0):8.4f}  "
                   f"total_PnL=${b.get('total_pnl',0):8.2f}")

    # ── Alpha analysis ──
    fprint("\n" + "=" * 100)
    fprint("  ALPHA vs SPY BUY-AND-HOLD")
    fprint("=" * 100)
    for v in variants:
        m = results[v]['metrics']
        fprint(f"  {v}: {variant_names[v]:<27} Return={m['total_return']:>7.2f}%  "
               f"SPY={m['spy_total_return']:>7.2f}%  Alpha={m['alpha_vs_spy']:>7.2f}%")

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    fprint(f"\n  BEST VARIANT: {best_v} ({variant_names[best_v]}) — Sharpe {best_m['sharpe']:.3f}")

    # ── Key conclusions ──
    fprint("\n" + "=" * 100)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 100)

    any_pass_all = any(validations[v]['all_pass'] for v in variants)

    # Check if leverage improved risk-adjusted returns
    lev_variants = [v for v in variants if v != 'H']
    better_than_ctrl = [v for v in lev_variants
                        if results[v]['metrics']['sharpe'] > ctrl_sharpe]

    if any_pass_all:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"  VERDICT: Leveraged rotation WORKS — passes all 5 gates!")
        fprint(f"  Variants passing all gates: {', '.join(f'{v} ({variant_names[v]})' for v in passing)}")

    if better_than_ctrl:
        fprint(f"  Leverage IMPROVES risk-adjusted returns for: "
               f"{', '.join(f'{v} ({variant_names[v]})' for v in better_than_ctrl)}")
    else:
        fprint(f"  Leverage does NOT improve risk-adjusted returns (Sharpe) vs unleveraged control.")
        fprint(f"  Volatility decay and amplified drawdowns negate the return amplification.")

    # Decay verdict
    all_decay_data = [results[v]['decay_analysis'] for v in variants
                      if results[v]['decay_analysis'].get('has_decay_data')]
    if all_decay_data:
        avg_decay = np.mean([d['mean_decay_pct'] for d in all_decay_data])
        fprint(f"  Average volatility decay across leveraged variants: {avg_decay:+.3f}% per trade")
        if avg_decay < -1.0:
            fprint(f"  WARNING: Significant vol decay detected. Leveraged ETFs are a poor vehicle for monthly rotation.")
        elif avg_decay < 0:
            fprint(f"  Moderate vol decay. Leverage may still add value in trending markets.")

    fprint(f"\n  Control (unleveraged Top-2): Sharpe={ctrl_sharpe:.3f}, Return={ctrl_ret:.2f}%, MDD={ctrl_mdd:.2f}%")
    fprint(f"  Best leveraged variant: {best_v} ({variant_names[best_v]}): "
           f"Sharpe={best_m['sharpe']:.3f}, Return={best_m['total_return']:.2f}%, MDD={best_m['mdd']:.2f}%")

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
        'decay_analyses': {},
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
        # Save decay analysis (without per-trade details for JSON size)
        decay = results[v]['decay_analysis']
        decay_summary = {k: v2 for k, v2 in decay.items() if k != 'details'}
        save_results['decay_analyses'][v] = decay_summary

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

    # Save volatility decay details
    for v in variants:
        decay = results[v]['decay_analysis']
        if decay.get('has_decay_data') and 'details' in decay:
            decay_path = OUTPUT_DIR / f"decay_details_{v}.json"
            with open(decay_path, 'w') as f:
                json.dump(decay['details'], f, indent=2)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            run_name = f"lev_etf_rotation_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("commission", 0.0)
                mlflow.log_param("n_sectors", len(SECTORS))
                mlflow.log_param("n_features", len(FEAT_COLS))
                mlflow.log_param("rebalance_days", REBALANCE_INTERVAL_DAYS)
                mlflow.log_param("n_days", len(sc))
                mlflow.log_param("oot_days", oot_days)
                mlflow.log_param("data_range", f"{sc.index[0].date()} to {sc.index[-1].date()}")
                mlflow.log_param("best_variant", f"{best_v}_{variant_names[best_v]}")
                mlflow.log_param("n_variants", len(variants))

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
                    mlflow.log_metric(f"{prefix}max_daily_loss", m['max_daily_loss'])
                    mlflow.log_metric(f"{prefix}gates_pass", val['n_pass'])

                    # Decay metrics
                    decay = results[v]['decay_analysis']
                    if decay.get('has_decay_data'):
                        mlflow.log_metric(f"{prefix}vol_decay_mean", decay['mean_decay_pct'])
                        mlflow.log_metric(f"{prefix}vol_decay_median", decay['median_decay_pct'])

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
