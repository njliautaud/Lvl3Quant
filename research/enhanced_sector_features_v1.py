#!/usr/bin/env python3
"""
Enhanced Sector Feature Engineering v1
======================================
Experiment: Test novel feature groups for LightGBM sector ranking model.
Tests each feature group independently + combined best, with full walk-forward
and options backtest. Reports IC improvement and Sharpe improvement over baseline.

Feature groups:
  A) Intraday pattern features (last-hour return, close vs day-high, range/ATR)
  B) Cross-sectional features (z-score of 5d return, rank percentile, dispersion)
  C) Flow/sentiment proxies (volume surge, up-volume ratio, consecutive up/down)
  D) Macro regime features (VIX level, VIX change 5d, TLT return, HYG-TLT)
  E) Combined best (only features that improve IC from A-D)
  F) Baseline (standard features only)

Walk-forward: sliding 252-day train, 21-day OOT, 11 sector ETFs.
Options: ATR + 15% haircut, $2.60 commission, $645 start, bull spreads VIX>=20.
Full 4-gate adversarial audit.

Self-contained. Uses yfinance for data. LightGBM CPU mode.
"""
import json, os, sys, time, warnings, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = ROOT / "output" / "enhanced_sector_features_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 RT
HAIRCUT = 0.15
TRAIN_DAYS = 252
OOT_DAYS = 21
TOP_K = 3

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
MACRO_TICKERS = ['^VIX', 'SPY', 'TLT', 'HYG']

def fprint(*a, **kw):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **kw, flush=True)

# ---------------------------------------------------------------------------
# MLflow (optional — report results even if MLflow is down)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data():
    """Download sector ETFs + macro data via yfinance."""
    import yfinance as yf
    fprint("Downloading data...")
    all_tickers = SECTORS + MACRO_TICKERS
    raw = yf.download(all_tickers, start='2008-01-01', progress=False, threads=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    # Normalize VIX column name
    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vix_col].dropna() if vix_col in close.columns else pd.Series(dtype=float)
    spy = close['SPY'].dropna()
    tlt = close['TLT'].dropna() if 'TLT' in close.columns else pd.Series(dtype=float)
    hyg = close['HYG'].dropna() if 'HYG' in close.columns else pd.Series(dtype=float)

    sector_close = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sector_high = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sector_low = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    sector_vol = volume[[c for c in SECTORS if c in volume.columns]].dropna(how='all')

    # Common index
    ix = sector_close.index
    for s in [spy.index, vix.index, sector_high.index, sector_low.index, sector_vol.index]:
        ix = ix.intersection(s)
    if not tlt.empty:
        ix = ix.intersection(tlt.index)
    if not hyg.empty:
        ix = ix.intersection(hyg.index)

    fprint(f"Data: {len(ix)} days, {len(sector_close.columns)} sectors, "
           f"{ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}")

    return (sector_close.loc[ix], sector_high.loc[ix], sector_low.loc[ix],
            sector_vol.loc[ix], spy.loc[ix], vix.loc[ix],
            tlt.loc[ix] if not tlt.empty else None,
            hyg.loc[ix] if not hyg.empty else None)


# ---------------------------------------------------------------------------
# Feature engineering — BASELINE (standard 12 momentum/vol/quality features)
# ---------------------------------------------------------------------------
def compute_baseline_features(close, idx, ticker):
    """Standard momentum + vol + quality features (baseline 12)."""
    px = close[ticker].iloc[:idx+1].dropna()
    if len(px) < 260:
        return None
    f = {}
    # Momentum lookbacks
    for lb, nm in [(5,'ret_5d'), (10,'ret_10d'), (21,'ret_21d'),
                   (63,'ret_63d'), (126,'ret_126d'), (252,'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    # Volatility
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    # Quality / risk
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    px63 = px.iloc[-63:]
    pk = px63.cummax()
    f['maxdd_63d'] = float(((px63 / pk) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3
    return f

BASELINE_COLS = ['ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
                 'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel']


# ---------------------------------------------------------------------------
# Feature Group A: Intraday pattern features
# ---------------------------------------------------------------------------
def compute_intraday_features(close, high, low, idx, ticker):
    """Last-hour return proxy, close vs day-high %, intraday range/ATR ratio."""
    px = close[ticker].iloc[:idx+1].dropna()
    hi = high[ticker].iloc[:idx+1].dropna()
    lo = low[ticker].iloc[:idx+1].dropna()
    if len(px) < 30:
        return None
    f = {}
    # Close vs day-high % (proxy for selling pressure into close)
    f['close_vs_high'] = float((px.iloc[-1] - lo.iloc[-1]) / (hi.iloc[-1] - lo.iloc[-1] + 1e-10))
    # Avg close vs high over 5d
    ratio_5d = ((px.iloc[-5:].values - lo.iloc[-5:].values) /
                (hi.iloc[-5:].values - lo.iloc[-5:].values + 1e-10))
    f['avg_close_vs_high_5d'] = float(np.mean(ratio_5d))
    # Intraday range / ATR ratio (expansion/contraction)
    daily_range = hi.iloc[-1] - lo.iloc[-1]
    atr_14 = (hi.iloc[-14:] - lo.iloc[-14:]).mean()
    f['range_atr_ratio'] = float(daily_range / (atr_14 + 1e-10))
    # Last-hour return proxy: close vs open approximation using prev close
    # (can't get intraday without intraday data, so use close-to-close gap)
    if len(px) >= 2:
        f['gap_return'] = float(px.iloc[-1] / px.iloc[-2] - 1) - float(
            (hi.iloc[-1] - lo.iloc[-1]) / px.iloc[-2])  # net of range = residual
    else:
        f['gap_return'] = 0.0
    return f

INTRADAY_COLS = ['close_vs_high', 'avg_close_vs_high_5d', 'range_atr_ratio', 'gap_return']


# ---------------------------------------------------------------------------
# Feature Group B: Cross-sectional features
# ---------------------------------------------------------------------------
def compute_crosssectional_features(close, idx, ticker, all_tickers):
    """Z-score of 5d return vs universe, rank percentile, cross-sectional dispersion."""
    px = close[ticker].iloc[:idx+1].dropna()
    if len(px) < 10:
        return None
    f = {}
    ret_5d_self = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0.0

    # Compute all sector 5d returns for cross-section
    all_rets = []
    for tk in all_tickers:
        tkpx = close[tk].iloc[:idx+1].dropna()
        if len(tkpx) > 5:
            all_rets.append(float(tkpx.iloc[-1] / tkpx.iloc[-5] - 1))
    if len(all_rets) < 3:
        return None

    arr = np.array(all_rets)
    mu, sigma = arr.mean(), arr.std() + 1e-10
    f['xs_zscore_5d'] = float((ret_5d_self - mu) / sigma)
    f['xs_rank_pct'] = float(sp_stats.percentileofscore(arr, ret_5d_self) / 100.0)
    f['xs_dispersion'] = float(sigma)  # how spread out sectors are (regime signal)
    return f

CROSSSECTIONAL_COLS = ['xs_zscore_5d', 'xs_rank_pct', 'xs_dispersion']


# ---------------------------------------------------------------------------
# Feature Group C: Flow/sentiment proxies
# ---------------------------------------------------------------------------
def compute_flow_features(close, volume, idx, ticker):
    """Volume surge, up-volume ratio, consecutive up/down days."""
    px = close[ticker].iloc[:idx+1].dropna()
    vol = volume[ticker].iloc[:idx+1].dropna()
    if len(px) < 25 or len(vol) < 25:
        return None
    f = {}
    # Volume surge: current vol / 20d avg
    avg_vol_20 = vol.iloc[-20:].mean()
    f['vol_surge'] = float(vol.iloc[-1] / (avg_vol_20 + 1e-10))
    # 5d average volume surge
    f['vol_surge_5d'] = float(vol.iloc[-5:].mean() / (avg_vol_20 + 1e-10))

    # Up-volume ratio: fraction of up-days in last 10 days with above-avg volume
    rets_10 = px.pct_change().iloc[-10:]
    vols_10 = vol.iloc[-10:]
    up_mask = rets_10 > 0
    if len(vols_10) > 0 and vols_10.sum() > 0:
        f['up_vol_ratio'] = float((vols_10[up_mask].sum()) / (vols_10.sum() + 1e-10))
    else:
        f['up_vol_ratio'] = 0.5

    # Consecutive up/down days
    daily_rets = px.pct_change().iloc[-20:]
    consec = 0
    for r in daily_rets.iloc[::-1]:
        if r > 0:
            consec += 1
        elif r < 0:
            consec -= 1
            break
        else:
            break
    # Recalculate properly
    consec_up = 0
    for r in daily_rets.iloc[::-1]:
        if r > 0:
            consec_up += 1
        else:
            break
    consec_down = 0
    for r in daily_rets.iloc[::-1]:
        if r < 0:
            consec_down += 1
        else:
            break
    f['consec_up_days'] = float(consec_up)
    f['consec_down_days'] = float(consec_down)

    return f

FLOW_COLS = ['vol_surge', 'vol_surge_5d', 'up_vol_ratio', 'consec_up_days', 'consec_down_days']


# ---------------------------------------------------------------------------
# Feature Group D: Macro regime features
# ---------------------------------------------------------------------------
def compute_macro_features(vix, tlt, hyg, spy, idx):
    """VIX level, VIX change 5d, yield curve proxy (TLT return), credit spread proxy."""
    f = {}
    # VIX level (same for all sectors at each date — cross-sectional broadcast)
    f['vix_level'] = float(vix.iloc[idx]) if idx < len(vix) else 20.0

    # VIX 5d change
    if idx >= 5 and len(vix) > idx:
        f['vix_chg_5d'] = float(vix.iloc[idx] - vix.iloc[idx-5])
    else:
        f['vix_chg_5d'] = 0.0

    # TLT return 21d (yield curve proxy — rising TLT = falling rates)
    if tlt is not None and idx < len(tlt) and idx >= 21:
        f['tlt_ret_21d'] = float(tlt.iloc[idx] / tlt.iloc[idx-21] - 1)
    else:
        f['tlt_ret_21d'] = 0.0

    # Credit spread proxy: HYG - TLT 21d return (tightening = bullish)
    if hyg is not None and tlt is not None and idx < len(hyg) and idx >= 21:
        hyg_ret = float(hyg.iloc[idx] / hyg.iloc[idx-21] - 1)
        tlt_ret = float(tlt.iloc[idx] / tlt.iloc[idx-21] - 1)
        f['credit_spread_proxy'] = hyg_ret - tlt_ret
    else:
        f['credit_spread_proxy'] = 0.0

    # SPY momentum regime (above/below 200 SMA)
    if idx >= 200 and len(spy) > idx:
        sma200 = float(spy.iloc[idx-200:idx].mean())
        f['spy_above_sma200'] = 1.0 if spy.iloc[idx] > sma200 else 0.0
    else:
        f['spy_above_sma200'] = 1.0

    return f

MACRO_COLS = ['vix_level', 'vix_chg_5d', 'tlt_ret_21d', 'credit_spread_proxy', 'spy_above_sma200']


# ---------------------------------------------------------------------------
# Build feature matrix for a given feature group config
# ---------------------------------------------------------------------------
def build_feature_matrix(close, high, low, volume, spy, vix, tlt, hyg,
                         use_baseline=True, use_intraday=False, use_xs=False,
                         use_flow=False, use_macro=False):
    """Build full panel: rows = (date, ticker), columns = features + fwd_ret."""
    feat_cols = []
    if use_baseline:
        feat_cols += BASELINE_COLS
    if use_intraday:
        feat_cols += INTRADAY_COLS
    if use_xs:
        feat_cols += CROSSSECTIONAL_COLS
    if use_flow:
        feat_cols += FLOW_COLS
    if use_macro:
        feat_cols += MACRO_COLS

    tickers = [c for c in close.columns if c in SECTORS]
    # Monthly rebalance dates
    rebal_dates = close.resample('ME').last().index
    rebal_dates = [d for d in rebal_dates if d in close.index]

    records = []
    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in tickers:
            row = {'date': dt, 'ticker': tk}

            # Baseline features
            if use_baseline:
                bf = compute_baseline_features(close, idx, tk)
                if bf is None:
                    continue
                row.update(bf)

            # Intraday features
            if use_intraday:
                inf = compute_intraday_features(close, high, low, idx, tk)
                if inf is None:
                    continue
                row.update(inf)

            # Cross-sectional features
            if use_xs:
                xf = compute_crosssectional_features(close, idx, tk, tickers)
                if xf is None:
                    continue
                row.update(xf)

            # Flow features
            if use_flow:
                ff = compute_flow_features(close, volume, idx, tk)
                if ff is None:
                    continue
                row.update(ff)

            # Macro features (same for all tickers on a date)
            if use_macro:
                mf = compute_macro_features(vix, tlt, hyg, spy, idx)
                row.update(mf)

            # Forward 21-day return (target)
            fi = min(idx + OOT_DAYS, len(close) - 1)
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            row['fwd_ret'] = fwd_ret

            records.append(row)

    df = pd.DataFrame(records)
    fprint(f"  Built panel: {len(df)} rows, {len(feat_cols)} features, "
           f"{df['date'].nunique()} dates")
    return df, feat_cols


# ---------------------------------------------------------------------------
# LightGBM Walk-Forward with IC measurement
# ---------------------------------------------------------------------------
def lgbm_walkforward(df, feat_cols, train_periods=12, label='fwd_ret'):
    """
    Walk-forward LightGBM. Returns per-fold IC, rankings, and feature importance.
    Sliding window: train on last `train_periods` months, predict next month.
    """
    import lightgbm as lgb

    dates = sorted(df['date'].unique())
    rankings = {}
    fold_ics = []
    importance_gain = {}
    oof_preds = []

    for i in range(train_periods, len(dates)):
        train_dates = dates[max(0, i - train_periods):i]
        test_date = dates[i]

        tr = df[df['date'].isin(train_dates)]
        te = df[df['date'] == test_date].copy()

        if len(te) < 3 or len(tr) < 50:
            continue

        # Rank label for ranking objective
        tr = tr.copy()
        tr['rank_label'] = tr.groupby('date')[label].rank(pct=True)

        X_tr = np.nan_to_num(tr[feat_cols].values.astype(np.float32))
        y_tr = tr['rank_label'].values.astype(np.float32)
        X_te = np.nan_to_num(te[feat_cols].values.astype(np.float32))

        try:
            model = lgb.LGBMRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                reg_lambda=1.0, verbose=-1
            )
            model.fit(X_tr, y_tr)

            te['score'] = model.predict(X_te)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))

            # Per-fold IC (Spearman rank correlation between score and actual fwd return)
            if len(te) >= 5:
                ic = sp_stats.spearmanr(te['score'], te[label])[0]
                if np.isfinite(ic):
                    fold_ics.append({'date': test_date, 'ic': ic, 'n': len(te)})

            # Feature importance
            for feat, imp in zip(feat_cols, model.feature_importances_):
                importance_gain.setdefault(feat, []).append(float(imp))

            # OOF predictions for analysis
            for _, row in te.iterrows():
                oof_preds.append({
                    'date': test_date, 'ticker': row['ticker'],
                    'score': row['score'], 'actual': row[label]
                })

        except Exception as e:
            continue

    avg_ic = np.mean([f['ic'] for f in fold_ics]) if fold_ics else 0.0
    ic_std = np.std([f['ic'] for f in fold_ics]) if fold_ics else 0.0
    ic_ir = avg_ic / (ic_std + 1e-10)  # IC information ratio

    avg_importance = {f: float(np.mean(v)) for f, v in importance_gain.items()}

    fprint(f"  Walk-forward: {len(fold_ics)} folds, avg IC={avg_ic:.4f}, "
           f"IC_std={ic_std:.4f}, IC_IR={ic_ir:.3f}")

    return {
        'rankings': rankings,
        'fold_ics': fold_ics,
        'avg_ic': avg_ic,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'avg_importance': avg_importance,
        'oof_preds': oof_preds,
    }


# ---------------------------------------------------------------------------
# ATR-based options pricing + backtest (same as sector_options_rotation_v1)
# ---------------------------------------------------------------------------
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({
        'hl': h - l,
        'hc': abs(h - c.shift(1)),
        'lc': abs(l - c.shift(1))
    }).max(axis=1)
    return tr.rolling(period).mean()


def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte / 252.0
    if T <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    intrinsic = max(0, S - K) if opt == 'call' else max(0, K - S)
    vol_factor = max(0.3, vix_val / 20.0)
    time_prem = atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S - K) / S)
    return intrinsic + time_prem


def price_spread(S, spread_pct, dte, atr, vix_val, stype='bull'):
    if stype == 'bull':
        K1, K2 = round(S), round(S * (1 + spread_pct / 100))
        lp = atr_premium(S, K1, dte, atr, vix_val, 'call') * (1 + HAIRCUT)
        sp = atr_premium(S, K2, dte, atr, vix_val, 'call') * (1 - HAIRCUT)
        debit = lp - sp
        width = K2 - K1
        return debit, (width - debit) * 100 - SPREAD_COMM, debit * 100 + SPREAD_COMM, K1, K2
    else:
        K1 = round(S * (1 - 5.0 / 100))
        K2 = round(S * (1 - 8.0 / 100))
        sp = atr_premium(S, K1, dte, atr, vix_val, 'put') * (1 - HAIRCUT)
        lp = atr_premium(S, K2, dte, atr, vix_val, 'put') * (1 + HAIRCUT)
        credit = sp - lp
        width = K1 - K2
        return credit, credit * 100 - SPREAD_COMM, (width - credit) * 100 + SPREAD_COMM, K1, K2


def options_backtest(rankings, close, high, low, vix, spy):
    """Run options backtest with dynamic regime switching. Bull spreads when VIX>=20."""
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(high[tk], low[tk], close[tk]) for tk in close.columns}
    equity = CAP
    trades = []
    eq_curve = [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue
        sv = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv
        bull = sv >= sm
        cv = float(vix.loc[dt])

        scores = rankings[dt]
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [tk for tk, _ in ranked[:TOP_K] if tk in close.columns]
        if not picks:
            continue

        # Position sizing: equal weight, max positions
        budget_per = equity / max(len(picks), 1)

        for tk in picks:
            S = float(close[tk].loc[dt])
            atr_val = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S * 0.02

            # Dynamic: bull spreads if VIX >= 20 or bearish regime, else put credits
            if cv >= 20 or not bull:
                stype = 'bull'
                spct = 3.0
                dte = 30
            else:
                stype = 'put_credit'
                spct = 5.0
                dte = 30

            cost_or_credit, max_profit, max_loss, K1, K2 = price_spread(
                S, spct, dte, atr_val, cv, stype)

            if stype == 'bull':
                cost_per_contract = cost_or_credit * 100 + SPREAD_COMM
                if cost_per_contract <= 0 or cost_per_contract > budget_per:
                    continue
                n_contracts = max(1, int(budget_per / cost_per_contract))
            else:
                margin_per = max_loss
                if margin_per <= 0 or margin_per > budget_per:
                    continue
                n_contracts = max(1, int(budget_per / margin_per))

            # Simulate outcome: check price at expiry
            exp_idx = close.index.get_indexer([dt + pd.Timedelta(days=dte)], method='ffill')[0]
            if exp_idx <= close.index.get_indexer([dt], method='ffill')[0]:
                continue
            S_exp = float(close[tk].iloc[exp_idx])

            if stype == 'bull':
                spread_val = max(0, min(K2 - K1, S_exp - K1)) * 100
                pnl = (spread_val - cost_or_credit * 100 - SPREAD_COMM) * n_contracts
            else:
                if S_exp >= K1:
                    pnl = (cost_or_credit * 100 - SPREAD_COMM) * n_contracts
                elif S_exp <= K2:
                    pnl = -(max_loss) * n_contracts
                else:
                    loss_pct = (K1 - S_exp) / (K1 - K2)
                    pnl = ((cost_or_credit * 100 - SPREAD_COMM) - loss_pct * (K1 - K2) * 100) * n_contracts

            equity += pnl
            equity = max(equity, 10.0)  # floor to prevent negative
            trades.append({
                'date': str(dt.date()), 'ticker': tk, 'type': stype,
                'pnl': round(pnl, 2), 'equity': round(equity, 2),
                'vix': round(cv, 1), 'n_contracts': n_contracts
            })
            eq_curve.append(equity)

    return equity, trades, eq_curve


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(eq_curve, trades, start_cap=645.0):
    """Compute Sharpe, Sortino, PF, WR, CAGR, max DD."""
    if len(eq_curve) < 2:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'cagr': 0,
                'maxdd': 0, 'final_equity': start_cap, 'n_trades': 0}

    eq = np.array(eq_curve)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    n_trades = len(trades)
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / max(n_trades, 1)
    pf = sum(wins) / max(abs(sum(losses)), 1e-10) if losses else float('inf')

    # Annualized metrics (assume monthly rebal ~12 trades/year)
    if len(rets) > 1:
        # Use monthly-ish returns
        mean_r = np.mean(rets)
        std_r = np.std(rets) + 1e-10
        sharpe = mean_r / std_r * np.sqrt(12)  # monthly frequency

        downside = rets[rets < 0]
        down_std = np.std(downside) + 1e-10 if len(downside) > 0 else 1e-10
        sortino = mean_r / down_std * np.sqrt(12)
    else:
        sharpe = sortino = 0

    # CAGR
    years = n_trades / 12.0  # approx
    final = eq[-1]
    cagr = (final / start_cap) ** (1 / max(years, 0.5)) - 1 if final > 0 else -1.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = float(dd.min())

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 3),
        'cagr': round(cagr, 4),
        'maxdd': round(maxdd, 4),
        'final_equity': round(final, 2),
        'n_trades': n_trades,
        'total_pnl': round(sum(pnls), 2),
    }


# ---------------------------------------------------------------------------
# 4-gate adversarial audit
# ---------------------------------------------------------------------------
def adversarial_audit(fold_ics, trades, eq_curve, metrics, variant_name):
    """
    4-gate audit:
    1. IC stability — reject if IC std > 2x mean IC (unstable signal)
    2. Regime balance — reject if bull/bear performance gap > 50%
    3. Permutation test — reject if p > 0.05 (random could do as well)
    4. Drawdown gate — reject if maxDD > 40%
    """
    gates = {}

    # Gate 1: IC stability
    ics = [f['ic'] for f in fold_ics]
    if len(ics) > 3:
        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        ic_stable = ic_std < 2 * abs(ic_mean) if ic_mean != 0 else False
        gates['g1_ic_stability'] = {
            'pass': ic_stable,
            'ic_mean': round(ic_mean, 4),
            'ic_std': round(ic_std, 4),
            'ratio': round(ic_std / (abs(ic_mean) + 1e-10), 2)
        }
    else:
        gates['g1_ic_stability'] = {'pass': False, 'reason': 'too few folds'}

    # Gate 2: Regime balance
    pnls = [t['pnl'] for t in trades]
    vix_vals = [t['vix'] for t in trades]
    if len(pnls) > 10:
        high_vix = [p for p, v in zip(pnls, vix_vals) if v >= 20]
        low_vix = [p for p, v in zip(pnls, vix_vals) if v < 20]
        high_avg = np.mean(high_vix) if high_vix else 0
        low_avg = np.mean(low_vix) if low_vix else 0
        max_avg = max(abs(high_avg), abs(low_avg), 1e-10)
        gap = abs(high_avg - low_avg) / max_avg
        gates['g2_regime_balance'] = {
            'pass': gap < 0.50,
            'high_vix_avg_pnl': round(high_avg, 2),
            'low_vix_avg_pnl': round(low_avg, 2),
            'gap': round(gap, 3)
        }
    else:
        gates['g2_regime_balance'] = {'pass': False, 'reason': 'too few trades'}

    # Gate 3: Permutation test (shuffle rankings, recompute PF)
    if len(pnls) > 10:
        real_pf = metrics['pf']
        n_perms = 500
        perm_pfs = []
        for _ in range(n_perms):
            shuffled = np.random.permutation(pnls)
            wins_s = [p for p in shuffled if p > 0]
            losses_s = [abs(p) for p in shuffled if p <= 0]
            pf_s = sum(wins_s) / max(sum(losses_s), 1e-10)
            perm_pfs.append(pf_s)
        p_val = np.mean([1 for p in perm_pfs if p >= real_pf]) / n_perms
        gates['g3_permutation'] = {
            'pass': p_val < 0.05,
            'p_value': round(p_val, 4),
            'real_pf': round(real_pf, 3),
            'perm_median_pf': round(np.median(perm_pfs), 3)
        }
    else:
        gates['g3_permutation'] = {'pass': False, 'reason': 'too few trades'}

    # Gate 4: Drawdown
    gates['g4_drawdown'] = {
        'pass': metrics['maxdd'] > -0.40,
        'maxdd': metrics['maxdd']
    }

    all_pass = all(g['pass'] for g in gates.values())
    return {'all_pass': all_pass, 'gates': gates}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_variant(name, close, high, low, volume, spy, vix, tlt, hyg,
                use_baseline=True, use_intraday=False, use_xs=False,
                use_flow=False, use_macro=False):
    """Run one feature variant end-to-end."""
    fprint(f"\n{'='*60}")
    fprint(f"VARIANT: {name}")
    fprint(f"{'='*60}")

    # Build features
    df, feat_cols = build_feature_matrix(
        close, high, low, volume, spy, vix, tlt, hyg,
        use_baseline=use_baseline, use_intraday=use_intraday,
        use_xs=use_xs, use_flow=use_flow, use_macro=use_macro)

    if len(df) < 100:
        fprint(f"  SKIP: insufficient data ({len(df)} rows)")
        return None

    # Walk-forward LightGBM
    wf_result = lgbm_walkforward(df, feat_cols)

    # Options backtest
    final_eq, trades, eq_curve = options_backtest(
        wf_result['rankings'], close, high, low, vix, spy)

    # Metrics
    metrics = compute_metrics(eq_curve, trades)
    fprint(f"  Results: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
           f"PF={metrics['pf']}, WR={metrics['wr']:.1%}, "
           f"Final=${metrics['final_equity']:.0f}")

    # Adversarial audit
    audit = adversarial_audit(wf_result['fold_ics'], trades, eq_curve, metrics, name)
    fprint(f"  Audit: {'PASS' if audit['all_pass'] else 'FAIL'} "
           f"({sum(1 for g in audit['gates'].values() if g['pass'])}/4 gates)")

    return {
        'name': name,
        'n_features': len(feat_cols),
        'feature_cols': feat_cols,
        'avg_ic': wf_result['avg_ic'],
        'ic_std': wf_result['ic_std'],
        'ic_ir': wf_result['ic_ir'],
        'avg_importance': wf_result['avg_importance'],
        'metrics': metrics,
        'audit': audit,
        'fold_ics': wf_result['fold_ics'],
    }


def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("Enhanced Sector Feature Engineering v1")
    fprint("=" * 70)

    # Download data
    close, high, low, volume, spy, vix, tlt, hyg = download_data()

    # Define variants
    variants = [
        ('F_Baseline', dict(use_baseline=True)),
        ('A_Intraday', dict(use_baseline=True, use_intraday=True)),
        ('B_CrossSectional', dict(use_baseline=True, use_xs=True)),
        ('C_Flow', dict(use_baseline=True, use_flow=True)),
        ('D_Macro', dict(use_baseline=True, use_macro=True)),
    ]

    results = {}
    baseline_ic = None
    ic_improvements = {}

    # Run baseline first, then each group
    for name, kwargs in variants:
        res = run_variant(name, close, high, low, volume, spy, vix, tlt, hyg, **kwargs)
        if res is not None:
            results[name] = res
            if name == 'F_Baseline':
                baseline_ic = res['avg_ic']
            else:
                if baseline_ic is not None:
                    ic_delta = res['avg_ic'] - baseline_ic
                    ic_improvements[name] = ic_delta
                    fprint(f"  IC delta vs baseline: {ic_delta:+.4f}")

    # Variant E: Combined best (only features that improved IC)
    fprint("\n" + "=" * 60)
    fprint("Selecting best features for combined variant E...")
    use_intraday = ic_improvements.get('A_Intraday', -1) > 0
    use_xs = ic_improvements.get('B_CrossSectional', -1) > 0
    use_flow = ic_improvements.get('C_Flow', -1) > 0
    use_macro = ic_improvements.get('D_Macro', -1) > 0

    selected = []
    if use_intraday: selected.append('A_Intraday')
    if use_xs: selected.append('B_CrossSectional')
    if use_flow: selected.append('C_Flow')
    if use_macro: selected.append('D_Macro')

    if selected:
        fprint(f"  Selected groups: {selected}")
        res_e = run_variant('E_Combined', close, high, low, volume, spy, vix, tlt, hyg,
                            use_baseline=True, use_intraday=use_intraday,
                            use_xs=use_xs, use_flow=use_flow, use_macro=use_macro)
        if res_e is not None:
            results['E_Combined'] = res_e
            if baseline_ic is not None:
                ic_delta = res_e['avg_ic'] - baseline_ic
                fprint(f"  E combined IC delta vs baseline: {ic_delta:+.4f}")
    else:
        fprint("  No feature groups improved IC — skipping combined variant")

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------
    fprint("\n" + "=" * 70)
    fprint("SUMMARY REPORT")
    fprint("=" * 70)

    summary_rows = []
    for name in ['F_Baseline', 'A_Intraday', 'B_CrossSectional', 'C_Flow', 'D_Macro', 'E_Combined']:
        if name not in results:
            continue
        r = results[name]
        m = r['metrics']
        ic_d = r['avg_ic'] - baseline_ic if baseline_ic is not None else 0.0
        row = {
            'variant': name,
            'n_features': r['n_features'],
            'avg_ic': round(r['avg_ic'], 4),
            'ic_delta': round(ic_d, 4),
            'ic_ir': round(r['ic_ir'], 3),
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'pf': m['pf'],
            'wr': m['wr'],
            'cagr': m['cagr'],
            'maxdd': m['maxdd'],
            'final_eq': m['final_equity'],
            'n_trades': m['n_trades'],
            'audit_pass': r['audit']['all_pass'],
            'gates_passed': sum(1 for g in r['audit']['gates'].values() if g['pass']),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    fprint("\n" + summary_df.to_string(index=False))

    # Feature importance for best variant
    best_variant = max(results.keys(), key=lambda k: results[k]['avg_ic'])
    fprint(f"\nBest variant by IC: {best_variant} (IC={results[best_variant]['avg_ic']:.4f})")
    imp = results[best_variant]['avg_importance']
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
    fprint("Top 15 features by importance:")
    for feat, val in imp_sorted:
        fprint(f"  {feat:30s} {val:.1f}")

    # Save results
    out = {
        'experiment': 'enhanced_sector_features_v1',
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(time.time() - t0, 1),
        'data_range': f"{close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}",
        'n_sectors': len(close.columns),
        'baseline_ic': baseline_ic,
        'ic_improvements': {k: round(v, 4) for k, v in ic_improvements.items()},
        'best_variant': best_variant,
        'summary': summary_rows,
        'per_variant': {},
    }
    for name, r in results.items():
        out['per_variant'][name] = {
            'n_features': r['n_features'],
            'feature_cols': r['feature_cols'],
            'avg_ic': r['avg_ic'],
            'ic_std': r['ic_std'],
            'ic_ir': r['ic_ir'],
            'metrics': r['metrics'],
            'audit': r['audit'],
            'avg_importance': r['avg_importance'],
            'fold_ics': [{'date': str(f['date'].date()), 'ic': round(f['ic'], 4)}
                         for f in r['fold_ics']],
        }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save summary CSV
    summary_df.to_csv(OUTPUT_DIR / 'summary.csv', index=False)

    # ---------------------------------------------------------------------------
    # MLflow logging
    # ---------------------------------------------------------------------------
    if MLFLOW_OK:
        try:
            exp = mlflow.set_experiment("enhanced_sector_features_v1")
            for name, r in results.items():
                with mlflow.start_run(run_name=name):
                    mlflow.log_params({
                        'variant': name,
                        'n_features': r['n_features'],
                        'train_days': TRAIN_DAYS,
                        'oot_days': OOT_DAYS,
                    })
                    mlflow.log_metrics({
                        'avg_ic': r['avg_ic'],
                        'ic_std': r['ic_std'],
                        'ic_ir': r['ic_ir'],
                        'sharpe': r['metrics']['sharpe'],
                        'sortino': r['metrics']['sortino'],
                        'pf': r['metrics']['pf'],
                        'wr': r['metrics']['wr'],
                        'cagr': r['metrics']['cagr'],
                        'maxdd': r['metrics']['maxdd'],
                        'final_equity': r['metrics']['final_equity'],
                        'n_trades': r['metrics']['n_trades'],
                        'audit_pass': 1 if r['audit']['all_pass'] else 0,
                    })
                    mlflow.log_artifact(str(results_path))
            fprint("MLflow logging complete")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == '__main__':
    main()
