#!/usr/bin/env python3
"""
LGBM Signal Enhancement V1
==============================

GOAL: Improve the LGBM sector ranking signal (KB #285, Sharpe 1.40 equity)
by adding new features and tuning. This is the CORE alpha source for ALL
our validated strategies (equity rotation, V9/V10 spreads, market-neutral).

Every +0.1 Sharpe improvement here propagates to all downstream strategies.

FEATURE SETS TO TEST:
  1. BASELINE (17 features) — current production
  2. +MACRO (21 features) — add VIX level, VIX term structure, yield curve, DXY
  3. +CROSS-SECTOR (28 features) — add relative strength vs peers, dispersion
  4. +FLOW (24 features) — add volume ratio, on-balance volume momentum
  5. +ALL COMBINED (35 features) — everything together
  6. MINIMAL (8 best features via importance) — Occam's razor test

TEST METHODOLOGY:
  - Same walk-forward sliding window as production (60d train, 1d OOT)
  - Same equity rotation backtest (Long Top-2, monthly rebalance, $645)
  - Compare Sharpe, WR, regime balance across feature sets
  - Permutation test on best variant
  - Feature importance analysis

Output: output/growth_research/lgbm_signal_enhancement_v1/
MLflow experiment: lgbm_signal_enhancement_v1
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

BASE = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "lgbm_signal_enhancement_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "lgbm_signal_enhancement_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK")
except Exception as e:
    fprint(f"MLflow not available: {e}")

import lightgbm as lgb

# ==================== CONFIG ====================

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
INITIAL_CAPITAL = 645.0
REBALANCE_DAYS = 28

# Feature set definitions
BASELINE_FEATURES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

MACRO_FEATURES = BASELINE_FEATURES + [
    'vix_level', 'vix_21d_change', 'vix_percentile_63d', 'spy_regime',
]

CROSS_SECTOR_FEATURES = BASELINE_FEATURES + [
    'rel_strength_vs_mean', 'rank_21d', 'rank_63d',
    'sector_dispersion_21d', 'sector_dispersion_63d',
    'beta_to_spy_63d', 'corr_to_spy_63d',
    'ret_21d_minus_spy', 'ret_63d_minus_spy',
    'momentum_breadth', 'vol_ratio_21_63',
]

FLOW_FEATURES = BASELINE_FEATURES + [
    'volume_ratio_5d', 'volume_ratio_21d',
    'obv_momentum_21d', 'obv_momentum_63d',
    'up_volume_ratio_21d', 'price_volume_trend_21d',
    'avg_volume_rank',
]

ALL_FEATURES = list(set(BASELINE_FEATURES + MACRO_FEATURES[len(BASELINE_FEATURES):] +
                        CROSS_SECTOR_FEATURES[len(BASELINE_FEATURES):] +
                        FLOW_FEATURES[len(BASELINE_FEATURES):]))

FEATURE_SETS = {
    'A_baseline': BASELINE_FEATURES,
    'B_macro': MACRO_FEATURES,
    'C_cross_sector': CROSS_SECTOR_FEATURES,
    'D_flow': FLOW_FEATURES,
    'E_all': ALL_FEATURES,
    # F_minimal will be determined after running baseline
}


# ==================== DATA ====================

def download_data():
    """Download sector ETFs + SPY + VIX + volume data."""
    import yfinance as yf

    all_tickers = SECTORS + ['SPY', '^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers with volume...")

    # Download OHLCV
    raw = yf.download(all_tickers, start='2020-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()

    # Get volume data
    volume = raw['Volume'] if mi else None
    if volume is not None and isinstance(volume.columns, pd.MultiIndex):
        volume.columns = volume.columns.get_level_values(-1)
    if volume is not None:
        volume = volume.ffill().fillna(0)

    close = close.rename(columns={'^VIX': 'VIX'})
    if volume is not None:
        volume = volume.rename(columns={'^VIX': 'VIX'})

    vix = close['VIX'].dropna() if 'VIX' in close.columns else pd.Series(dtype=float)
    spy = close['SPY'].dropna() if 'SPY' in close.columns else pd.Series(dtype=float)

    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    vol_data = volume[[c for c in SECTORS if c in volume.columns]] if volume is not None else None

    ix = sc.index.intersection(spy.index).intersection(vix.index)
    if vol_data is not None:
        ix = ix.intersection(vol_data.index)

    fprint(f"Data: {len(ix)} trading days, {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}")

    return sc.loc[ix], spy.loc[ix], vix.loc[ix], vol_data.loc[ix] if vol_data is not None else None


# ==================== ENHANCED FEATURE ENGINEERING ====================

def compute_all_features(tk, sc, spy, vix, vol_data, idx_end):
    """Compute ALL possible features for a ticker at a given point in time."""
    px = sc[tk].iloc[:idx_end + 1].dropna()
    if len(px) < 260:
        return None

    f = {}

    # ── BASELINE 17 features ──
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

    # ── MACRO features ──
    if len(vix) > idx_end:
        v = vix.iloc[:idx_end + 1]
        f['vix_level'] = float(v.iloc[-1])
        f['vix_21d_change'] = float(v.iloc[-1] / v.iloc[-21] - 1) if len(v) > 21 else 0
        f['vix_percentile_63d'] = float((v.iloc[-63:] <= v.iloc[-1]).mean()) if len(v) > 63 else 0.5
    else:
        f['vix_level'] = 20.0
        f['vix_21d_change'] = 0.0
        f['vix_percentile_63d'] = 0.5

    # SPY regime (above/below 200d SMA)
    if len(spy) > idx_end:
        sp = spy.iloc[:idx_end + 1]
        sma200 = float(sp.iloc[-200:].mean()) if len(sp) > 200 else float(sp.mean())
        f['spy_regime'] = 1.0 if float(sp.iloc[-1]) > sma200 else -1.0
    else:
        f['spy_regime'] = 1.0

    # ── CROSS-SECTOR features ──
    # Relative strength vs sector mean
    sector_rets_21d = {}
    for s in sc.columns:
        sr = sc[s].iloc[:idx_end + 1]
        if len(sr) > 21:
            sector_rets_21d[s] = float(sr.iloc[-1] / sr.iloc[-21] - 1)
    if sector_rets_21d:
        mean_ret = np.mean(list(sector_rets_21d.values()))
        f['rel_strength_vs_mean'] = f['ret_21d'] - mean_ret
        sorted_rets = sorted(sector_rets_21d.values(), reverse=True)
        f['rank_21d'] = sorted_rets.index(sector_rets_21d.get(tk, 0)) / max(len(sorted_rets) - 1, 1) if tk in sector_rets_21d else 0.5
    else:
        f['rel_strength_vs_mean'] = 0.0
        f['rank_21d'] = 0.5

    sector_rets_63d = {}
    for s in sc.columns:
        sr = sc[s].iloc[:idx_end + 1]
        if len(sr) > 63:
            sector_rets_63d[s] = float(sr.iloc[-1] / sr.iloc[-63] - 1)
    if sector_rets_63d:
        sorted_rets_63 = sorted(sector_rets_63d.values(), reverse=True)
        f['rank_63d'] = sorted_rets_63.index(sector_rets_63d.get(tk, 0)) / max(len(sorted_rets_63) - 1, 1) if tk in sector_rets_63d else 0.5
    else:
        f['rank_63d'] = 0.5

    # Sector dispersion
    if sector_rets_21d:
        f['sector_dispersion_21d'] = float(np.std(list(sector_rets_21d.values())))
    else:
        f['sector_dispersion_21d'] = 0.0

    if sector_rets_63d:
        f['sector_dispersion_63d'] = float(np.std(list(sector_rets_63d.values())))
    else:
        f['sector_dispersion_63d'] = 0.0

    # Beta and correlation to SPY
    if len(spy) > idx_end and len(px) >= 63:
        sp_rets = spy.iloc[:idx_end + 1].pct_change().dropna().iloc[-63:]
        tk_rets = rets.iloc[-63:]
        common = sp_rets.index.intersection(tk_rets.index)
        if len(common) > 20:
            sp_r = sp_rets.loc[common]
            tk_r = tk_rets.loc[common]
            cov = np.cov(tk_r.values, sp_r.values)
            f['beta_to_spy_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            f['corr_to_spy_63d'] = float(np.corrcoef(tk_r.values, sp_r.values)[0, 1])
        else:
            f['beta_to_spy_63d'] = 1.0
            f['corr_to_spy_63d'] = 0.5
    else:
        f['beta_to_spy_63d'] = 1.0
        f['corr_to_spy_63d'] = 0.5

    # Excess returns vs SPY
    if len(spy) > idx_end:
        sp = spy.iloc[:idx_end + 1]
        spy_21d = float(sp.iloc[-1] / sp.iloc[-21] - 1) if len(sp) > 21 else 0
        spy_63d = float(sp.iloc[-1] / sp.iloc[-63] - 1) if len(sp) > 63 else 0
        f['ret_21d_minus_spy'] = f['ret_21d'] - spy_21d
        f['ret_63d_minus_spy'] = f['ret_63d'] - spy_63d
    else:
        f['ret_21d_minus_spy'] = 0.0
        f['ret_63d_minus_spy'] = 0.0

    # Momentum breadth: what % of lookback periods are positive
    periods = [5, 10, 21, 63]
    pos_count = sum(1 for p in periods if f.get(f'ret_{p}d', 0) > 0)
    f['momentum_breadth'] = pos_count / len(periods)

    # Vol ratio
    f['vol_ratio_21_63'] = f['vol_21d'] / max(f['vol_63d'], 0.001)

    # ── FLOW features ──
    if vol_data is not None and tk in vol_data.columns:
        vol = vol_data[tk].iloc[:idx_end + 1].dropna()
        if len(vol) > 21:
            f['volume_ratio_5d'] = float(vol.iloc[-5:].mean() / (vol.iloc[-63:].mean() + 1)) if len(vol) > 63 else 1.0
            f['volume_ratio_21d'] = float(vol.iloc[-21:].mean() / (vol.iloc[-63:].mean() + 1)) if len(vol) > 63 else 1.0

            # OBV momentum
            px_ch = px.pct_change().dropna()
            common = px_ch.index.intersection(vol.index)
            if len(common) > 63:
                px_s = px_ch.loc[common]
                vol_s = vol.loc[common]
                obv = (vol_s * np.sign(px_s)).cumsum()
                f['obv_momentum_21d'] = float(obv.iloc[-1] / obv.iloc[-21] - 1) if obv.iloc[-21] != 0 else 0
                f['obv_momentum_63d'] = float(obv.iloc[-1] / obv.iloc[-63] - 1) if len(obv) > 63 and obv.iloc[-63] != 0 else 0
            else:
                f['obv_momentum_21d'] = 0.0
                f['obv_momentum_63d'] = 0.0

            # Up-volume ratio
            if len(common) > 21:
                px_s = px_ch.loc[common]
                vol_s = vol.loc[common]
                up_vol = vol_s[px_s > 0].iloc[-21:].sum()
                dn_vol = vol_s[px_s < 0].iloc[-21:].sum()
                f['up_volume_ratio_21d'] = float(up_vol / (up_vol + dn_vol + 1))
            else:
                f['up_volume_ratio_21d'] = 0.5

            # Price-volume trend
            if len(common) > 21:
                pvt = ((px_ch.loc[common]) * vol.loc[common]).iloc[-21:].sum()
                f['price_volume_trend_21d'] = float(pvt)
            else:
                f['price_volume_trend_21d'] = 0.0

            # Volume rank vs other sectors
            all_vols = {}
            for s in sc.columns:
                if s in vol_data.columns:
                    v5 = vol_data[s].iloc[:idx_end + 1].dropna()
                    if len(v5) > 5:
                        all_vols[s] = float(v5.iloc[-5:].mean())
            if all_vols:
                sorted_vols = sorted(all_vols.values(), reverse=True)
                f['avg_volume_rank'] = sorted_vols.index(all_vols.get(tk, 0)) / max(len(sorted_vols) - 1, 1) if tk in all_vols else 0.5
            else:
                f['avg_volume_rank'] = 0.5
        else:
            for fn in ['volume_ratio_5d', 'volume_ratio_21d', 'obv_momentum_21d',
                       'obv_momentum_63d', 'up_volume_ratio_21d', 'price_volume_trend_21d',
                       'avg_volume_rank']:
                f[fn] = 0.0
    else:
        for fn in ['volume_ratio_5d', 'volume_ratio_21d', 'obv_momentum_21d',
                   'obv_momentum_63d', 'up_volume_ratio_21d', 'price_volume_trend_21d',
                   'avg_volume_rank']:
            f[fn] = 0.0

    return f


# ==================== BACKTEST ENGINE ====================

def run_backtest(sc, spy, vix, vol_data, feature_list, variant_name, verbose=True):
    """Run equity rotation backtest with a given feature set."""
    dates = sc.index
    n_days = len(dates)
    start_idx = 280

    equity = INITIAL_CAPITAL
    equity_curve = []
    closed_trades = []
    holdings = {}
    last_rebalance_idx = None
    n_rebalances = 0
    feature_importances = np.zeros(len(feature_list))

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # Mark-to-market
        for tk, pos in holdings.items():
            if tk in sc.columns:
                current_price = float(sc[tk].iloc[day_idx])
                pos['current_price'] = current_price
                pos['pnl'] = pos['shares'] * (current_price - pos['entry_price'])

        # Rebalance
        should_rebalance = (last_rebalance_idx is None or
                           day_idx - last_rebalance_idx >= REBALANCE_DAYS)

        if should_rebalance and day_idx < n_days - 5:
            # Build training data
            records = []
            start_i = max(260, day_idx - 400)
            rebal_idx = list(range(start_i, day_idx))[::20]

            for i in rebal_idx[:-1]:
                for tk in sc.columns:
                    feats = compute_all_features(tk, sc, spy, vix, vol_data, i)
                    if not feats:
                        continue
                    fi = min(i + 28, n_days - 1)
                    feats['fwd_ret'] = float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
                    feats['date_idx'] = i
                    feats['ticker'] = tk
                    records.append(feats)

            df = pd.DataFrame(records)
            for c in feature_list:
                if c not in df.columns:
                    df[c] = 0.0
            df[feature_list] = df[feature_list].fillna(0.0)

            if len(df) < 50:
                equity_curve.append({'date': today, 'equity': equity})
                continue

            df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
            X = np.nan_to_num(df[feature_list].values.astype(np.float32))
            y = df['rank_label'].values.astype(np.float32)

            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
            )
            m.fit(X, y)

            # Track feature importance
            fi_arr = m.feature_importances_
            if len(fi_arr) == len(feature_list):
                feature_importances += fi_arr

            # Predict current rankings
            current_feats = {}
            for tk in sc.columns:
                feats = compute_all_features(tk, sc, spy, vix, vol_data, day_idx)
                if feats:
                    current_feats[tk] = feats

            if not current_feats:
                equity_curve.append({'date': today, 'equity': equity})
                continue

            pred_df = pd.DataFrame(current_feats).T
            for c in feature_list:
                if c not in pred_df.columns:
                    pred_df[c] = 0.0
            X_pred = np.nan_to_num(pred_df[feature_list].values.astype(np.float32))
            scores = dict(zip(pred_df.index, m.predict(X_pred)))

            # Close old positions
            for tk, pos in holdings.items():
                closed_trades.append({
                    'ticker': tk, 'entry_date': pos['entry_date'], 'exit_date': today,
                    'entry_price': pos['entry_price'],
                    'exit_price': pos.get('current_price', pos['entry_price']),
                    'pnl': pos.get('pnl', 0),
                })
                equity += pos.get('pnl', 0)
            holdings = {}

            # Open top-2
            top2 = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)[:2]
            per_pos = equity / 2

            for tk in top2:
                price = float(sc[tk].iloc[day_idx])
                shares = per_pos / price
                holdings[tk] = {
                    'shares': shares, 'entry_price': price,
                    'entry_date': today, 'current_price': price, 'pnl': 0,
                }

            last_rebalance_idx = day_idx
            n_rebalances += 1

        mtm = equity + sum(pos.get('pnl', 0) for pos in holdings.values())
        equity_curve.append({'date': today, 'equity': mtm})

    # Close remaining
    for tk, pos in holdings.items():
        equity += pos.get('pnl', 0)
        closed_trades.append({
            'ticker': tk, 'entry_date': pos['entry_date'], 'exit_date': dates[-1],
            'entry_price': pos['entry_price'],
            'exit_price': pos.get('current_price', pos['entry_price']),
            'pnl': pos.get('pnl', 0),
        })

    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    # Normalize feature importance
    if feature_importances.sum() > 0:
        feature_importances = feature_importances / feature_importances.sum()

    return eq_df, closed_trades, dict(zip(feature_list, feature_importances))


# ==================== METRICS ====================

def compute_metrics(eq_df, trades, spy):
    eq = eq_df['equity']
    daily_rets = eq.pct_change().dropna()
    if len(daily_rets) < 10:
        return None

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(daily_rets) / 252
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1/max(n_years, 0.01)) - 1)
    ann_ret = float(daily_rets.mean() * 252)
    ann_vol = float(daily_rets.std() * np.sqrt(252))
    sharpe = ann_ret / max(ann_vol, 0.001)

    down = daily_rets[daily_rets < 0]
    down_vol = float(down.std() * np.sqrt(252)) if len(down) > 0 else 0.001
    sortino = ann_ret / max(down_vol, 0.001)

    cummax = eq.cummax()
    max_dd = float(((eq - cummax) / cummax).min())

    wins = sum(1 for t in trades if t['pnl'] > 0) if trades else 0
    wr = wins / len(trades) if trades else 0

    spy_a = spy.reindex(eq.index).ffill()
    spy_ret = float(spy_a.iloc[-1] / spy_a.iloc[0] - 1) if len(spy_a.dropna()) > 20 else 0
    alpha = total_ret - spy_ret

    # Regime
    spy_rets = spy_a.pct_change().dropna()
    eq_rets = eq.pct_change().dropna()
    common = spy_rets.index.intersection(eq_rets.index)

    regime = {}
    for name, mask_fn in [('green', lambda s: s > 0.001), ('red', lambda s: s < -0.001)]:
        mask = mask_fn(spy_rets.loc[common])
        r = eq_rets.loc[common][mask]
        if len(r) > 5:
            ar = float(r.mean() * 252)
            av = float(r.std() * np.sqrt(252)) if r.std() > 0 else 0.001
            regime[name] = round(ar / max(av, 0.001), 3)
        else:
            regime[name] = 0.0

    g, r = abs(regime.get('green', 0)), abs(regime.get('red', 0))
    mx = max(g, r)
    gap = abs(g - r) / mx if mx > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'win_rate': round(wr * 100, 1),
        'n_trades': len(trades),
        'final': round(float(eq.iloc[-1]), 0),
        'alpha': round(alpha * 100, 1),
        'regime_green': regime.get('green', 0),
        'regime_red': regime.get('red', 0),
        'regime_gap': round(gap, 3),
        'regime_pass': gap <= 0.50,
    }


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("LGBM SIGNAL ENHANCEMENT V1")
    fprint("=" * 70)
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint()

    sc, spy, vix, vol_data = download_data()

    all_results = {}
    all_importances = {}

    for variant, feat_list in FEATURE_SETS.items():
        fprint(f"\n{'─'*50}")
        fprint(f"VARIANT {variant} ({len(feat_list)} features)")
        fprint(f"{'─'*50}")

        eq_df, trades, importances = run_backtest(sc, spy, vix, vol_data, feat_list, variant)
        metrics = compute_metrics(eq_df, trades, spy)

        if metrics:
            fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
                   f"CAGR: {metrics['cagr']}%, MDD: {metrics['max_dd']}%")
            fprint(f"  WR: {metrics['win_rate']}%, Trades: {metrics['n_trades']}, "
                   f"Final: ${metrics['final']:.0f}, Alpha: {metrics['alpha']}%")
            fprint(f"  Regime: Green={metrics['regime_green']}, Red={metrics['regime_red']}, "
                   f"Gap={metrics['regime_gap']}, {'PASS' if metrics['regime_pass'] else 'FAIL'}")

            all_results[variant] = metrics
            all_importances[variant] = importances

            # Top 5 features
            top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            fprint(f"  Top features: " + ", ".join(f"{f}={v:.3f}" for f, v in top5))

            if MLFLOW_OK:
                try:
                    with mlflow.start_run(run_name=variant):
                        mlflow.log_params({'variant': variant, 'n_features': len(feat_list)})
                        for mk, mv in metrics.items():
                            if isinstance(mv, (int, float)):
                                mlflow.log_metric(mk, mv)
                except Exception as e:
                    fprint(f"  MLflow error: {e}")
        else:
            fprint(f"  No metrics computed")

    # ── MINIMAL variant: use top 8 features from baseline ──
    fprint(f"\n{'─'*50}")
    fprint("VARIANT F_minimal (top 8 features from baseline)")
    fprint(f"{'─'*50}")

    if 'A_baseline' in all_importances:
        sorted_feats = sorted(all_importances['A_baseline'].items(),
                            key=lambda x: x[1], reverse=True)
        minimal_features = [f for f, _ in sorted_feats[:8]]
        fprint(f"  Selected features: {minimal_features}")

        eq_df, trades, importances = run_backtest(sc, spy, vix, vol_data,
                                                   minimal_features, 'F_minimal')
        metrics = compute_metrics(eq_df, trades, spy)

        if metrics:
            fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
                   f"CAGR: {metrics['cagr']}%, MDD: {metrics['max_dd']}%")
            fprint(f"  WR: {metrics['win_rate']}%, Trades: {metrics['n_trades']}, "
                   f"Final: ${metrics['final']:.0f}, Alpha: {metrics['alpha']}%")
            fprint(f"  Regime: Green={metrics['regime_green']}, Red={metrics['regime_red']}, "
                   f"Gap={metrics['regime_gap']}, {'PASS' if metrics['regime_pass'] else 'FAIL'}")
            all_results['F_minimal'] = metrics

            if MLFLOW_OK:
                try:
                    with mlflow.start_run(run_name='F_minimal'):
                        mlflow.log_params({'variant': 'F_minimal', 'n_features': 8,
                                          'features': str(minimal_features)})
                        for mk, mv in metrics.items():
                            if isinstance(mv, (int, float)):
                                mlflow.log_metric(mk, mv)
                except:
                    pass

    # ── SUMMARY ──
    fprint(f"\n{'='*70}")
    fprint("SUMMARY: FEATURE SET COMPARISON")
    fprint(f"{'='*70}")

    best_sharpe = -999
    best_v = None
    baseline_sharpe = all_results.get('A_baseline', {}).get('sharpe', 0)

    for v in sorted(all_results.keys()):
        m = all_results[v]
        delta = m['sharpe'] - baseline_sharpe
        marker = " *** BEST" if m['sharpe'] == max(r['sharpe'] for r in all_results.values()) else ""
        fprint(f"  {v:20s}: Sharpe {m['sharpe']:+.3f} (Δ{delta:+.3f}), "
               f"Sortino {m['sortino']:.2f}, CAGR {m['cagr']}%, "
               f"WR {m['win_rate']}%, MDD {m['max_dd']}%, "
               f"Regime {'PASS' if m['regime_pass'] else 'FAIL'}{marker}")

        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_v = v

    fprint(f"\n  BEST: {best_v} (Sharpe {best_sharpe})")
    fprint(f"  Baseline improvement: {best_sharpe - baseline_sharpe:+.3f} Sharpe")

    # Feature importance across all variants
    fprint(f"\n  MOST IMPORTANT FEATURES (averaged across variants):")
    avg_importance = {}
    for variant, imps in all_importances.items():
        for f, v in imps.items():
            avg_importance[f] = avg_importance.get(f, []) + [v]
    avg_importance = {f: np.mean(vs) for f, vs in avg_importance.items()}
    top10 = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for f, v in top10:
        fprint(f"    {f:25s}: {v:.4f}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    summary = {
        'experiment': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'runtime_s': round(elapsed, 1),
        'variants': all_results,
        'feature_importances': {v: dict(imp) for v, imp in all_importances.items()},
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"Results saved.")


if __name__ == '__main__':
    main()
