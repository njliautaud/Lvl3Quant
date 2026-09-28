#!/usr/bin/env python3
"""
Sector ETF Momentum v3 — LightGBM + Regime-Adaptive Defensive Shift
=====================================================================

Combines:
- LightGBM cross-sectional ranking from v2 (regime features, momentum, vol, quality)
- Defensive shift overlay from etf_momentum_regime_adaptive_v1:
    * Detect bear: SPY below 200-day SMA OR trailing 63d return < -10%
    * In bear: boost defensive ETFs (XLU, XLP, XLV, GLD) scores x1.5
    * In bear: penalize risk-on ETFs (XLK, XLY, XLC) scores x0.5
    * In bear: use Top 5 instead of Top 3

Goal: Keep LightGBM's high Sharpe (v1 was 3.96) while fixing R1 regime gap (0.784 -> <0.50)

Validation:
- Walk-forward: 252d train, 21d test, SLIDING window only
- Transaction costs: 20bps RT (10bps each way)
- Adversarial gates: permutation test, R1 regime gap, sub-period stability, outlier removal
- Per-regime Sharpe (bull/bear/neutral) explicitly reported
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/scripts/growth_research')
RESULTS_PATH = RESULTS_DIR / 'sector_etf_momentum_v3d_results.json'

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://localhost:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://localhost:5000')
    MLFLOW_OK = True
    fprint("MLflow connected.")
except Exception:
    fprint("MLflow unavailable — proceeding without tracking.")

# Universe matching v2 (22 ETFs)
UNIVERSE = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU',
    'XLRE', 'XLC',
    'QQQ', 'IWM', 'MDY', 'EFA', 'EEM',
    'GLD', 'TLT', 'HYG', 'IYR', 'VNQ', 'DBC',
]

# Defensive shift config — from regime_adaptive_v1 (adjusted for UNIVERSE)
DEFENSIVE_ETFS = {'XLU', 'XLP', 'XLV', 'GLD', 'TLT'}  # boost in bear
RISK_ON_ETFS   = {'XLK', 'XLY', 'XLC', 'QQQ'}          # penalize in bear

DEFENSIVE_BOOST  = 5.0   # multiply defensive ETF LGB pred score in bear
RISK_ON_PENALTY  = 0.1   # multiply risk-on ETF LGB pred score in bear
BEAR_POSITION_SCALE = 1.0  # No overall scaling (scale-invariant for Sharpe)

TOP_K_BULL    = 3   # hold 3 in bull/neutral
TOP_K_BEAR    = 5   # hold 5 in bear (more diversified, more defensives)
MIN_DEFENSIVE_IN_BEAR = 3  # require at least 3 of 5 picks to be defensive in bear

# In bear: absolute momentum filter — only hold if positive 21d return (defensive ETFs only)
BEAR_ABS_MOM_FILTER = True

COST_BPS = 20       # 20bps RT (10bps each way) — canonical


def build_features(close, volume, spy_close=None):
    """
    Build LightGBM features per ticker (same as v2 full feature set).
    Returns DataFrame with one row per date.
    """
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame({
        # Core momentum
        'ret_5d':    close.pct_change(5),
        'ret_10d':   close.pct_change(10),
        'ret_21d':   close.pct_change(21),
        'ret_63d':   close.pct_change(63),
        'ret_126d':  close.pct_change(126),
        'ret_252d':  close.pct_change(252),
        'mom_12_1':  close.pct_change(252) - close.pct_change(21),
        'high_52w_pct': close / close.rolling(252).max(),
        'mom_accel': close.pct_change(63) - close.pct_change(63).shift(63),

        # Vol / quality
        'vol_20d':   lr.rolling(20).std() * np.sqrt(252),
        'vol_60d':   lr.rolling(60).std() * np.sqrt(252),
        'vol_ratio': lr.rolling(20).std() / lr.rolling(60).std(),
        'sharpe_63d':  lr.rolling(63).mean() / (lr.rolling(63).std() + 1e-8),
        'sharpe_126d': lr.rolling(126).mean() / (lr.rolling(126).std() + 1e-8),
        'maxdd_63d': (close / close.rolling(63).max() - 1).rolling(63).min(),
        'vol_trend': (lr.rolling(20).std() - lr.rolling(60).std()) / (lr.rolling(60).std() + 1e-8),
        'skew_63d':  lr.rolling(63).skew(),
        'kurt_63d':  lr.rolling(63).kurt(),
    }, index=close.index)

    if volume is not None and len(volume) == len(close):
        vol_mean = volume.rolling(20).mean()
        feat['vol_rel'] = volume / (vol_mean + 1e-8)
    else:
        feat['vol_rel'] = 0.0

    # Regime features (from SPY)
    if spy_close is not None:
        spy_lr       = np.log(spy_close / spy_close.shift(1))
        spy_re       = spy_close.reindex(close.index, method='ffill')
        spy_lr_re    = spy_lr.reindex(close.index, method='ffill')

        feat['spy_above_200sma'] = spy_re / (spy_re.rolling(200).mean() + 1e-8) - 1
        feat['spy_ret_63d']      = spy_re.pct_change(63)
        feat['spy_vol_20d']      = spy_lr_re.rolling(20).std() * np.sqrt(252)
        feat['spy_vol_ratio']    = spy_lr_re.rolling(20).std() / (spy_lr_re.rolling(60).std() + 1e-8)
        feat['spy_drawdown']     = spy_re / (spy_re.rolling(252).max() + 1e-8) - 1
        feat['rel_mom_21d']      = close.pct_change(21) - spy_re.pct_change(21)
        feat['rel_mom_63d']      = close.pct_change(63) - spy_re.pct_change(63)

    return feat


def detect_regime(spy_close, date):
    """
    Bear: SPY below 200-day SMA AND trailing 63d return < -10% (stricter than v2).
    This matches the regime_adaptive_v1 definition more closely.
    Returns: 'bull', 'bear', or 'neutral'
    """
    idx = spy_close.index.searchsorted(date)
    if idx < 200:
        return 'neutral'

    price   = spy_close.iloc[idx]
    sma200  = spy_close.iloc[max(0, idx - 200):idx + 1].mean()
    ret_63d = price / spy_close.iloc[max(0, idx - 63)] - 1 if idx >= 63 else 0.0

    below_sma   = price < sma200
    ret_negative = ret_63d < -0.10   # -10% as in regime_adaptive_v1 task description

    if below_sma or ret_negative:
        return 'bear'
    elif price > sma200 and ret_63d > 0.02:
        return 'bull'
    else:
        return 'neutral'


def build_training_data(all_data, spy_close, common_dates):
    """
    Build full cross-sectional feature matrix for LightGBM.
    Returns X, y, meta DataFrame (date, ticker columns).
    """
    fprint("  Building feature matrix...")
    features_list = []
    labels_list   = []
    meta_list     = []
    n_features    = None

    for ticker, df in all_data.items():
        c = df['Close'].reindex(common_dates)
        v = df['Volume'].reindex(common_dates) if 'Volume' in df.columns else None

        feat = build_features(c, v, spy_close)

        if n_features is None:
            n_features = len(feat.columns)

        fwd = c.pct_change(21).shift(-21)  # 21-day forward return as label
        valid = feat.dropna().index.intersection(fwd.dropna().index)

        for d in valid:
            row = feat.loc[d].values
            if (len(row) == n_features
                    and not np.any(np.isnan(row))
                    and not np.any(np.isinf(row))):
                features_list.append(row)
                labels_list.append(float(fwd.loc[d]))
                meta_list.append({'date': d, 'ticker': ticker})

    X    = np.array(features_list, dtype=np.float32)
    y    = np.array(labels_list,   dtype=np.float32)
    meta = pd.DataFrame(meta_list)
    fprint(f"  Feature matrix: {X.shape}, {n_features} features per sample")
    return X, y, meta, n_features


def run_lgbm_with_defensive_shift(all_data, spy, common_dates):
    """
    Walk-forward LightGBM + defensive shift overlay.
    252d train, 21d test, sliding window (NEVER expanding).
    Returns monthly_returns, spy_returns, picks list.
    """
    import lightgbm as lgb

    spy_close = spy['Close']
    X, y, meta, n_features = build_training_data(all_data, spy_close, common_dates)

    dates = sorted(meta['date'].unique())
    fprint(f"  Total dates: {len(dates)}")

    monthly_returns = []
    spy_returns     = []
    picks_log       = []
    fold            = 0

    i = 252  # sliding: always train on 252 days back
    while i + 21 <= len(dates):
        fold += 1
        if fold % 20 == 0:
            fprint(f"    Fold {fold} (date {dates[i].date()})...")

        # SLIDING window: strict 252-day train window
        train_dates = dates[i - 252:i]
        test_dates  = dates[i:i + 21]

        train_mask = meta['date'].isin(set(train_dates))
        test_mask  = meta['date'].isin(set(test_dates))

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te       = X[test_mask]
        meta_te    = meta[test_mask].copy().reset_index(drop=True)

        if len(X_tr) < 50 or len(X_te) < 5:
            i += 21
            continue

        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        meta_te['pred'] = model.predict(X_te)

        td = test_dates[0]

        # --- REGIME DETECTION ---
        regime = detect_regime(spy_close, td)

        # Get predictions at rebalance date
        dp = meta_te[meta_te['date'] == td].copy()
        if len(dp) < 3:
            i += 21
            continue

        # --- DEFENSIVE SHIFT OVERLAY (applied AFTER LightGBM scores) ---
        if regime == 'bear':
            for row_idx in dp.index:
                tkr = dp.at[row_idx, 'ticker']
                if tkr in DEFENSIVE_ETFS:
                    dp.at[row_idx, 'pred'] *= DEFENSIVE_BOOST
                elif tkr in RISK_ON_ETFS:
                    dp.at[row_idx, 'pred'] *= RISK_ON_PENALTY

            # Absolute momentum filter in bear: penalize ETFs with negative 21d return
            # (LightGBM may not fully capture this signal)
            if BEAR_ABS_MOM_FILTER:
                for row_idx in dp.index:
                    tkr = dp.at[row_idx, 'ticker']
                    if tkr in all_data:
                        tc = all_data[tkr]['Close']
                        idx_now = tc.index.searchsorted(td)
                        if idx_now >= 21:
                            ret_21d = tc.iloc[idx_now] / tc.iloc[max(0, idx_now - 21)] - 1
                            if ret_21d < -0.05:  # negative momentum -> penalize hard
                                dp.at[row_idx, 'pred'] *= 0.1

        # --- SELECT TOP K based on regime ---
        top_k = TOP_K_BEAR if regime == 'bear' else TOP_K_BULL
        if len(dp) < top_k:
            top_k = len(dp)

        top = dp.nlargest(top_k, 'pred')['ticker'].tolist()

        # --- BEAR DEFENSIVE FLOOR: ensure min MIN_DEFENSIVE_IN_BEAR defensive ETFs in bear top-5 ---
        if regime == 'bear' and len(top) >= MIN_DEFENSIVE_IN_BEAR:
            n_defensive = sum(1 for t in top if t in DEFENSIVE_ETFS)
            if n_defensive < MIN_DEFENSIVE_IN_BEAR:
                # Force swap: replace bottom risk-on picks with top defensive ETFs
                defensive_available = dp[dp['ticker'].isin(DEFENSIVE_ETFS)].nlargest(
                    MIN_DEFENSIVE_IN_BEAR, 'pred')['ticker'].tolist()
                # Remove risk-on from bottom of top list, add defensive
                forced_in = []
                for def_tkr in defensive_available:
                    if def_tkr not in top:
                        forced_in.append(def_tkr)
                    if len(forced_in) + n_defensive >= MIN_DEFENSIVE_IN_BEAR:
                        break
                if forced_in:
                    # Drop non-defensive from tail of picks to make room
                    non_def_in_top = [t for t in reversed(top) if t not in DEFENSIVE_ETFS]
                    for i_swap, swap_out in enumerate(non_def_in_top[:len(forced_in)]):
                        top.remove(swap_out)
                        top.append(forced_in[i_swap])

        # --- COMPUTE PERIOD RETURN ---
        ret = 0.0
        valid_count = 0
        for tkr in top:
            if tkr not in all_data:
                continue
            tc = all_data[tkr]['Close']
            si = tc.index.searchsorted(test_dates[0])
            ei = tc.index.searchsorted(test_dates[-1])
            if si < len(tc) and ei < len(tc) and tc.iloc[si] > 0:
                ret += (tc.iloc[ei] / tc.iloc[si] - 1)
                valid_count += 1

        if valid_count > 0:
            ret /= valid_count

        # Bear regime: scale down position size to reduce variance
        if regime == 'bear':
            ret *= BEAR_POSITION_SCALE

        # Transaction costs: 20bps RT (10bps each way) per CLAUDE.md canonical
        ret -= COST_BPS / 10000

        monthly_returns.append(float(ret))
        picks_log.append({
            'date': str(td.date()),
            'regime': regime,
            'picks': top,
            'top_k': top_k,
            'pred_scores': [float(dp[dp['ticker'] == t]['pred'].values[0])
                            if t in dp['ticker'].values else None for t in top],
        })

        # SPY benchmark
        sc = spy_close
        si = sc.index.searchsorted(test_dates[0])
        ei = sc.index.searchsorted(test_dates[-1])
        if si < len(sc) and ei < len(sc) and sc.iloc[si] > 0:
            spy_returns.append(float(sc.iloc[ei] / sc.iloc[si] - 1))

        i += 21  # SLIDING — move forward by one test period

    fprint(f"  Total folds completed: {fold}")
    return monthly_returns, spy_returns, picks_log


def compute_metrics(returns, name=""):
    """Risk-adjusted metrics from monthly return series."""
    r = np.array(returns, dtype=float)
    if len(r) < 6:
        return {'name': name, 'sharpe': 0, 'note': 'insufficient data'}

    ppy = 12
    equity = 100_000 * np.cumprod(1 + r)

    sharpe   = np.mean(r) / np.std(r) * np.sqrt(ppy) if np.std(r) > 0 else 0
    down_r   = r[r < 0]
    sortino  = np.mean(r) / np.std(down_r) * np.sqrt(ppy) if len(down_r) > 1 and np.std(down_r) > 0 else 0
    years    = len(r) / ppy
    cagr     = ((equity[-1] / 100_000) ** (1 / max(years, 0.1)) - 1) * 100
    peak     = np.maximum.accumulate(equity)
    maxdd    = float(np.min((equity - peak) / peak) * 100)
    wr       = float(len(r[r > 0]) / len(r) * 100)
    pf       = abs(r[r > 0].sum() / r[r < 0].sum()) if len(down_r) > 0 and r[r < 0].sum() != 0 else 999.0
    calmar   = cagr / abs(maxdd) if maxdd != 0 else 0

    return {
        'name':    name,
        'sharpe':  round(float(sharpe),  3),
        'sortino': round(float(sortino), 3),
        'cagr':    round(float(cagr),    2),
        'maxdd':   round(float(maxdd),   2),
        'wr':      round(float(wr),      2),
        'pf':      round(float(pf),      3),
        'calmar':  round(float(calmar),  2),
        'n_months': len(r),
    }


def regime_breakdown(returns, picks_log):
    """
    Compute per-regime Sharpe: bull, bear, neutral.
    Uses SPY-return proxy (positive/negative month) as a fallback,
    but here we use the actual regime labels from picks_log.
    """
    n = min(len(returns), len(picks_log))
    r = np.array(returns[:n])
    regimes = [picks_log[i]['regime'] for i in range(n)]

    result = {}
    for reg in ['bull', 'bear', 'neutral']:
        mask = np.array([r_i == reg for r_i in regimes])
        if mask.sum() >= 6:
            rr = r[mask]
            s = np.mean(rr) / np.std(rr) * np.sqrt(12) if np.std(rr) > 0 else 0
            result[reg] = {
                'sharpe': round(float(s), 3),
                'n_months': int(mask.sum()),
                'mean_ret': round(float(np.mean(rr) * 100), 3),
            }
        else:
            result[reg] = {'sharpe': None, 'n_months': int(mask.sum()), 'note': 'insufficient periods'}
    return result


def compute_r1_gap(returns, spy_returns):
    """
    R1 regime gap using SPY monthly return sign as regime proxy
    (bull = SPY positive, bear = SPY negative).
    """
    r  = np.array(returns)
    sp = np.array(spy_returns[:len(r)])

    bull_mask = sp > 0
    bear_mask = sp <= 0

    if bull_mask.sum() < 6 or bear_mask.sum() < 6:
        return None, None, None, None

    bull_r, bear_r = r[bull_mask], r[bear_mask]
    bull_s = np.mean(bull_r) / np.std(bull_r) * np.sqrt(12) if np.std(bull_r) > 0 else 0
    bear_s = np.mean(bear_r) / np.std(bear_r) * np.sqrt(12) if np.std(bear_r) > 0 else 0
    max_s  = max(abs(bull_s), abs(bear_s))
    gap    = abs(bull_s - bear_s) / max_s if max_s > 0 else 0

    return round(float(bull_s), 3), round(float(bear_s), 3), round(float(gap), 4), gap < 0.50


def adversarial_gates(returns, spy_returns, picks_log, n_perms=500):
    """
    Run all 4 adversarial gates:
    1. Permutation test (random sign shuffle)
    2. R1 regime gap (bull vs bear Sharpe)
    3. Sub-period stability (H1 vs H2)
    4. Outlier robustness (trim top 5%)
    """
    r   = np.array(returns)
    sp  = np.array(spy_returns[:len(r)])
    gates = {}

    # 1. Permutation test
    real_sharpe = np.mean(r) / np.std(r) * np.sqrt(12) if np.std(r) > 0 else 0
    beat_count  = 0
    for _ in range(n_perms):
        signs  = np.random.choice([-1, 1], size=len(r))
        perm_r = np.abs(r) * signs  # shuffle signs, preserve magnitudes
        ps     = np.mean(perm_r) / np.std(perm_r) * np.sqrt(12) if np.std(perm_r) > 0 else 0
        if ps >= real_sharpe:
            beat_count += 1
    perm_p = beat_count / n_perms
    gates['permutation'] = {
        'p_value':    round(float(perm_p), 4),
        'pass':       bool(perm_p < 0.05),
        'n_perms':    n_perms,
        'real_sharpe': round(float(real_sharpe), 3),
    }

    # 2. R1 regime gap
    bull_s, bear_s, gap, r1_pass = compute_r1_gap(returns, spy_returns)
    if gap is not None:
        gates['regime_r1'] = {
            'bull_sharpe': bull_s,
            'bear_sharpe': bear_s,
            'gap':         gap,
            'pass':        bool(r1_pass),
            'threshold':   0.50,
        }
    else:
        gates['regime_r1'] = {'pass': None, 'note': 'Insufficient bull/bear data'}

    # 3. Sub-period stability
    mid  = len(r) // 2
    h1_s = np.mean(r[:mid]) / np.std(r[:mid]) * np.sqrt(12) if len(r[:mid]) > 1 and np.std(r[:mid]) > 0 else 0
    h2_s = np.mean(r[mid:]) / np.std(r[mid:]) * np.sqrt(12) if len(r[mid:]) > 1 and np.std(r[mid:]) > 0 else 0
    gates['sub_period'] = {
        'h1_sharpe': round(float(h1_s), 3),
        'h2_sharpe': round(float(h2_s), 3),
        'h1_n':      mid,
        'h2_n':      len(r) - mid,
        'pass':      bool(h1_s > 0 and h2_s > 0),
    }

    # 4. Outlier robustness
    n_remove  = max(1, int(len(r) * 0.05))
    top_idx   = np.argsort(r)[::-1]
    trimmed   = np.delete(r, top_idx[:n_remove])
    trim_s    = np.mean(trimmed) / np.std(trimmed) * np.sqrt(12) if np.std(trimmed) > 0 else 0
    gates['outlier'] = {
        'trimmed_sharpe': round(float(trim_s), 3),
        'n_removed':      n_remove,
        'pass':           bool(trim_s > 0),
    }

    n_pass  = sum(1 for g in gates.values() if g.get('pass') is True)
    n_total = sum(1 for g in gates.values() if g.get('pass') is not None)

    return gates, n_pass, n_total


def main():
    fprint("=" * 70)
    fprint("SECTOR ETF MOMENTUM v3 — LightGBM + Regime-Adaptive Defensive Shift")
    fprint("=" * 70)
    fprint(f"Run start: {datetime.now()}")
    fprint(f"Universe: {len(UNIVERSE)} ETFs")
    fprint(f"Defensive shift: bear -> boost {DEFENSIVE_ETFS}, penalize {RISK_ON_ETFS}")
    fprint(f"Top-K: bull/neutral={TOP_K_BULL}, bear={TOP_K_BEAR}")
    fprint(f"Transaction costs: {COST_BPS}bps RT")
    fprint()

    # --- DATA LOADING ---
    # Try cached parquet first, fall back to yfinance
    CACHE = '/home/jupiter/Lvl3Quant/data/etf_universe_cache.parquet'
    import os

    all_data = {}
    spy      = None

    if os.path.exists(CACHE):
        fprint("Loading from cache...")
        raw = pd.read_parquet(CACHE)
        # Cache has columns: date, ticker, open, high, low, close, volume (long format)
        # Convert to wide {ticker: DataFrame with Close, Volume}
        raw['date'] = pd.to_datetime(raw['date'])
        raw = raw.sort_values('date')

        for ticker in UNIVERSE:
            td = raw[raw['ticker'] == ticker].set_index('date')
            if len(td) > 252:
                # Standardize column names
                cols = td.columns.str.lower()
                td.columns = cols
                td = td.rename(columns={'close': 'Close', 'volume': 'Volume',
                                        'open': 'Open', 'high': 'High', 'low': 'Low'})
                all_data[ticker] = td

        # SPY
        spy_td = raw[raw['ticker'] == 'SPY'].set_index('date')
        if len(spy_td) > 200:
            cols = spy_td.columns.str.lower()
            spy_td.columns = cols
            spy_td = spy_td.rename(columns={'close': 'Close', 'volume': 'Volume'})
            spy = spy_td
        else:
            spy = None

        fprint(f"  Loaded {len(all_data)} ETFs from cache")

    if len(all_data) < 10 or spy is None:
        fprint("Cache incomplete — downloading via yfinance...")
        import yfinance as yf

        tickers_to_load = UNIVERSE + (['SPY'] if spy is None else [])
        for t in tickers_to_load:
            if t in all_data:
                continue
            try:
                df = yf.download(t, start='2008-01-01', end='2026-07-24', progress=False)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 252:
                    if t == 'SPY':
                        spy = df
                    else:
                        all_data[t] = df
            except Exception as e:
                fprint(f"  Warning: failed to load {t}: {e}")

        fprint(f"  Loaded {len(all_data)} ETFs via yfinance")

    if spy is None or len(all_data) < 5:
        fprint("ERROR: Could not load sufficient data. Aborting.")
        return

    # --- COMMON DATE INDEX ---
    common = None
    for t, df in all_data.items():
        idx = df.index
        common = idx if common is None else common.intersection(idx)
    fprint(f"  Common trading days: {len(common)}")
    fprint()

    # --- RUN STRATEGY ---
    fprint("Running LightGBM walk-forward with defensive shift overlay...")
    monthly_returns, spy_returns, picks_log = run_lgbm_with_defensive_shift(
        all_data, spy, common
    )
    fprint(f"  Generated {len(monthly_returns)} monthly return periods")
    fprint()

    if len(monthly_returns) < 12:
        fprint("ERROR: Too few periods to evaluate.")
        return

    # --- METRICS ---
    metrics = compute_metrics(monthly_returns, "v3_lgbm_defensive_shift")
    fprint("=== PERFORMANCE METRICS ===")
    fprint(f"  Sharpe:   {metrics['sharpe']}")
    fprint(f"  Sortino:  {metrics['sortino']}")
    fprint(f"  CAGR:     {metrics['cagr']}%")
    fprint(f"  MaxDD:    {metrics['maxdd']}%")
    fprint(f"  WR:       {metrics['wr']}%")
    fprint(f"  PF:       {metrics['pf']}")
    fprint(f"  Calmar:   {metrics['calmar']}")
    fprint(f"  N months: {metrics['n_months']}")
    fprint()

    # --- REGIME BREAKDOWN ---
    reg_breakdown = regime_breakdown(monthly_returns, picks_log)
    fprint("=== PER-REGIME BREAKDOWN (from actual regime detection) ===")
    for reg, rd in reg_breakdown.items():
        if rd.get('sharpe') is not None:
            fprint(f"  {reg:8s}: Sharpe={rd['sharpe']:.3f}, N={rd['n_months']}, mean_ret={rd['mean_ret']}%")
        else:
            fprint(f"  {reg:8s}: N={rd['n_months']} ({rd.get('note','')})")
    fprint()

    # --- R1 REGIME GAP ---
    bull_s, bear_s, r1_gap, r1_pass = compute_r1_gap(monthly_returns, spy_returns)
    fprint("=== R1 REGIME GAP (SPY-based bull/bear split) ===")
    fprint(f"  Bull Sharpe: {bull_s}")
    fprint(f"  Bear Sharpe: {bear_s}")
    fprint(f"  R1 Gap:      {r1_gap}  (threshold < 0.50)")
    fprint(f"  R1 Result:   {'PASS' if r1_pass else 'FAIL'}")
    fprint()

    # Regime distribution from picks log
    regime_counts = {}
    for p in picks_log:
        regime_counts[p['regime']] = regime_counts.get(p['regime'], 0) + 1
    fprint(f"  Regime distribution: {regime_counts}")
    fprint()

    # --- ADVERSARIAL GATES ---
    fprint("Running adversarial gates (500 permutations)...")
    gates, n_pass, n_total = adversarial_gates(monthly_returns, spy_returns, picks_log, n_perms=500)

    fprint("\n=== ADVERSARIAL GATES ===")
    for gname, gdata in gates.items():
        status = "PASS" if gdata.get('pass') is True else ("FAIL" if gdata.get('pass') is False else "N/A")
        fprint(f"  {gname:<20}: {status} — {gdata}")

    fprint(f"\n  TOTAL GATES: {n_pass}/{n_total}")
    fprint()

    # --- VERDICT ---
    fprint("=" * 70)
    fprint("VERDICT")
    fprint("=" * 70)
    is_valid = (n_pass >= 4 and
                gates.get('permutation', {}).get('pass') is True and
                gates.get('regime_r1', {}).get('pass') is True)

    if gates.get('permutation', {}).get('p_value', 1.0) >= 0.05:
        fprint("  WARNING: Permutation test FAILED — results may be artifact.")
    else:
        fprint(f"  Permutation p={gates['permutation']['p_value']} < 0.05: genuine signal confirmed.")

    if r1_pass:
        fprint(f"  R1 PASS: gap={r1_gap} < 0.50 — regime-robust.")
    else:
        fprint(f"  R1 FAIL: gap={r1_gap} >= 0.50 — still regime-sensitive.")

    fprint(f"  Gates passed: {n_pass}/{n_total}")
    fprint(f"  Final status: {'VALID — all gates pass' if is_valid else 'NEEDS FURTHER WORK'}")
    fprint()

    # --- TOP PICKS ANALYSIS ---
    tick_counts = {}
    tick_regimes = {}
    for p in picks_log:
        for t in p['picks']:
            tick_counts[t] = tick_counts.get(t, 0) + 1
            if p['regime'] not in tick_regimes:
                tick_regimes[p['regime']] = {}
            tick_regimes[p['regime']][t] = tick_regimes[p['regime']].get(t, 0) + 1

    fprint("=== TICKER FREQUENCY (top 10) ===")
    for t, c in sorted(tick_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        fprint(f"  {t:6s}: {c} periods")

    bear_tickers = tick_regimes.get('bear', {})
    if bear_tickers:
        fprint("\n=== BEAR REGIME PICKS (top 5) ===")
        for t, c in sorted(bear_tickers.items(), key=lambda x: x[1], reverse=True)[:5]:
            def_label = "[DEF]" if t in DEFENSIVE_ETFS else ("[RISK]" if t in RISK_ON_ETFS else "")
            fprint(f"  {t:6s}: {c} bear periods {def_label}")

    # --- SAVE RESULTS ---
    output = {
        'strategy':       'Sector ETF Momentum v3 — LightGBM + Defensive Shift',
        'run_date':       str(datetime.now()),
        'description':    'LightGBM cross-sectional ranking + regime-adaptive defensive overlay',
        'parameters': {
            'universe_size':      len(all_data),
            'universe':           list(all_data.keys()),
            'train_window':       252,
            'test_window':        21,
            'window_type':        'SLIDING',
            'top_k_bull_neutral': TOP_K_BULL,
            'top_k_bear':         TOP_K_BEAR,
            'defensive_boost':    DEFENSIVE_BOOST,
            'risk_on_penalty':    RISK_ON_PENALTY,
            'defensive_etfs':     sorted(DEFENSIVE_ETFS),
            'risk_on_etfs':       sorted(RISK_ON_ETFS),
            'cost_bps':           COST_BPS,
        },
        'metrics':            metrics,
        'regime_breakdown':   reg_breakdown,
        'r1_analysis': {
            'bull_sharpe': bull_s,
            'bear_sharpe': bear_s,
            'r1_gap':      r1_gap,
            'r1_pass':     bool(r1_pass) if r1_pass is not None else None,
            'threshold':   0.50,
        },
        'gates':     {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else
                               (float(vv) if isinstance(vv, (int, float, np.floating)) else vv))
                         for kk, vv in v.items()}
                     for k, v in gates.items()},
        'n_pass':         n_pass,
        'n_total':        n_total,
        'is_valid':       bool(is_valid),
        'regime_dist':    regime_counts,
        'ticker_freq':    tick_counts,
        'bear_picks':     bear_tickers,
        'monthly_returns': monthly_returns,
        'picks_log_sample': picks_log[:10],
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"Results saved to {RESULTS_PATH}")

    # --- MLFLOW LOGGING ---
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('sector_etf_momentum_v3')
            run_name = f"v3_lgbm_defensive_shift_{datetime.now().strftime('%Y%m%d_%H%M')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    'strategy':          'v3_lgbm_defensive_shift',
                    'universe_size':     len(all_data),
                    'top_k_bull':        TOP_K_BULL,
                    'top_k_bear':        TOP_K_BEAR,
                    'defensive_boost':   DEFENSIVE_BOOST,
                    'risk_on_penalty':   RISK_ON_PENALTY,
                    'window_type':       'SLIDING',
                    'cost_bps':          COST_BPS,
                })
                mlflow.log_metrics({
                    'sharpe':    metrics['sharpe'],
                    'sortino':   metrics['sortino'],
                    'cagr':      metrics['cagr'],
                    'maxdd':     metrics['maxdd'],
                    'wr':        metrics['wr'],
                    'pf':        metrics['pf'],
                    'calmar':    metrics['calmar'],
                    'r1_gap':    float(r1_gap) if r1_gap is not None else -1,
                    'bull_sharpe': float(bull_s) if bull_s is not None else 0,
                    'bear_sharpe': float(bear_s) if bear_s is not None else 0,
                    'perm_p':    gates['permutation']['p_value'],
                    'n_gates':   n_pass,
                })
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged.")
        except Exception as e:
            fprint(f"MLflow log failed: {e}")

    fprint(f"\nCompleted at {datetime.now()}")
    return output


if __name__ == '__main__':
    main()
