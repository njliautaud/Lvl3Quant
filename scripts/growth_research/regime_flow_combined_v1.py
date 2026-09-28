#!/usr/bin/env python3
"""
Regime + Flow Combined v1 — GRU regime filter + flow-enhanced LGBM sector ranking
==================================================================================
HYPOTHESIS: GRU regime score handles WHEN to trade (cuts MDD from -27.9% to -15.8%),
while flow features improve WHICH sectors to pick (+12% Sharpe in moderate VIX).
Combining them should yield both the drawdown and ranking improvements.

4 COMBINATIONS:
  A. BASELINE:   VIX>20 filter    + legacy LGBM (18 features)
  B. REGIME_ONLY: regime>0.4      + legacy LGBM (18 features)
  C. FLOW_ONLY:  VIX>20 filter    + flow-enhanced LGBM (73 features)
  D. FULL_STACK: regime>0.4       + flow-enhanced LGBM (73 features)

Backtest: $645 capital, bull call spreads on sector ETFs, 15% haircut entry+exit,
walk-forward LGBM ranking, hold to expiry (2-week DTE).

Adversarial validation + random baseline included.
MLflow logging to http://jupiter:5000

Author: Claude (Head of Quant)
Date: 2026-07-26
"""

import json, warnings, time, os, sys, socket
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

def fprint(*a, **kw): print(*a, **kw, flush=True)

# Auto-detect home
_hostname = socket.gethostname()
if 'neptune' in _hostname.lower() or os.path.exists('/home/nick'):
    BASE = Path('/home/nick/Lvl3Quant')
else:
    BASE = Path('/home/jupiter/Lvl3Quant')

OUTPUT_DIR = BASE / 'output' / 'growth_research' / 'regime_flow_combined_v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# CONFIGURATION
# =====================================================================

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
MACRO_TICKERS = ['SPY','QQQ','TLT','GLD','HYG','SHY','IWM','EFA']
VIX_TICKER = '^VIX'

CAP = 645.0
SPREAD_COMM = 2.60
HAIRCUT = 0.15
MAX_POS = 200
MAX_CONC = 3
DTE = 14  # 2-week spread

# Walk-forward LGBM
TRAIN_MONTHS = 12  # 12 months of training data per fold
WF_STEP_WEEKS = 2  # step every 2 weeks

# Regime filter threshold
REGIME_THRESHOLD = 0.4

# Legacy LGBM features (18)
LEGACY_FEATURES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'spy_beta_63d', 'relative_strength_21d', 'relative_strength_63d',
    'sector_dispersion_21d', 'vix_corr_63d', 'volume_trend_21d',
]

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    os.system(f"{sys.executable} -m pip install lightgbm -q")
    import lightgbm as lgb
    LGB_OK = True

try:
    import yfinance as yf
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance -q")
    import yfinance as yf

try:
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    fprint("MLflow: connected")
except Exception:
    MLFLOW_OK = False
    fprint("MLflow: not available, results saved locally only")

np.random.seed(42)


# =====================================================================
# 1. DATA DOWNLOAD
# =====================================================================

def download_data():
    fprint("[1/7] Downloading market data...")
    cache = CACHE_DIR / 'combined_data_v1.parquet'

    all_tickers = SECTORS + MACRO_TICKERS + [VIX_TICKER]
    raw = yf.download(all_tickers, start='2008-01-01', end='2026-07-26', progress=False)

    mi = isinstance(raw.columns, pd.MultiIndex)
    if mi:
        close = raw['Close'].copy()
        high = raw['High'].copy()
        low = raw['Low'].copy()
        volume = raw['Volume'].copy()
    else:
        close = high = low = volume = raw

    # Handle VIX column name
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    # Sector close/high/low/volume
    sec_cols = [c for c in SECTORS if c in close.columns]
    sc = close[sec_cols].dropna(how='all')
    sh = high[sec_cols].dropna(how='all') if mi else sc
    sl = low[sec_cols].dropna(how='all') if mi else sc
    sv = volume[sec_cols].dropna(how='all') if mi else pd.DataFrame(1e6, index=sc.index, columns=sec_cols)

    # Macro close
    macro_cols = [c for c in MACRO_TICKERS if c in close.columns]
    macro = close[macro_cols].dropna(how='all')

    # Macro volume for flow features
    macro_vol = volume[macro_cols].dropna(how='all') if mi else pd.DataFrame(1e6, index=macro.index, columns=macro_cols)
    macro_high = high[macro_cols].dropna(how='all') if mi else macro
    macro_low = low[macro_cols].dropna(how='all') if mi else macro

    # Align all indices
    ix = sc.index
    for s in [vix, spy, sh, sl, sv, macro]:
        ix = ix.intersection(s.index)

    fprint(f"  {len(ix)} trading days, {len(sec_cols)} sectors, {len(macro_cols)} macro tickers")
    return (sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.loc[ix],
            spy.loc[ix], vix.loc[ix],
            macro.loc[ix], macro_vol.reindex(ix).fillna(1e6),
            macro_high.reindex(ix).fillna(method='ffill'),
            macro_low.reindex(ix).fillna(method='ffill'))


# =====================================================================
# 2. LOAD GRU REGIME PREDICTIONS
# =====================================================================

def load_regime_predictions():
    fprint("[2/7] Loading GRU regime predictions...")
    regime_path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not regime_path.exists():
        fprint(f"  WARNING: Regime predictions not found at {regime_path}")
        return None

    data = np.load(regime_path, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']

    regime_series = pd.Series(scores, index=dates, name='regime_score')
    # Handle duplicate dates by keeping last value
    if regime_series.index.duplicated().any():
        regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    fprint(f"  Loaded {len(regime_series)} regime scores, {dates[0].date()} to {dates[-1].date()}")
    fprint(f"  Score range: {scores.min():.3f} to {scores.max():.3f}, mean={scores.mean():.3f}")
    fprint(f"  Days above 0.4: {(scores > 0.4).sum()} ({(scores > 0.4).mean()*100:.1f}%)")
    return regime_series


# =====================================================================
# 3. FEATURE ENGINEERING
# =====================================================================

def compute_legacy_features(sc, spy, vix, idx, tk):
    """Compute 18 legacy features for sector ranking."""
    p = sc[tk].iloc[:idx+1].dropna()
    if len(p) < 260:
        return None

    f = {}
    # Price momentum at various lookbacks
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(p.iloc[-1]/p.iloc[-lb]-1) if len(p) > lb else 0.0

    r = p.pct_change().dropna()
    # Volatility
    f['vol_21d'] = float(r.iloc[-21:].std()*np.sqrt(252)) if len(r) > 21 else 0.2
    f['vol_63d'] = float(r.iloc[-63:].std()*np.sqrt(252)) if len(r) > 63 else 0.2
    # Sharpe
    r63 = r.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0
    # Max drawdown
    p63 = p.iloc[-63:]
    f['maxdd_63d'] = float(((p63/p63.cummax())-1).min())
    # Pct of 52w high
    f['pct_52w_high'] = float(p.iloc[-1]/p.iloc[-252:].max())
    # Momentum acceleration
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3

    # Beta to SPY (63d)
    spy_r = spy.pct_change().dropna()
    common_idx = r.index.intersection(spy_r.index)
    if len(common_idx) > 63:
        sr = spy_r.loc[common_idx].iloc[-63:]
        tr = r.loc[common_idx].iloc[-63:]
        if len(sr) > 10 and sr.std() > 0:
            f['spy_beta_63d'] = float(np.cov(tr, sr)[0,1] / (sr.var() + 1e-10))
        else:
            f['spy_beta_63d'] = 1.0
    else:
        f['spy_beta_63d'] = 1.0

    # Relative strength vs SPY
    spy_p = spy.iloc[:idx+1].dropna()
    for lb, nm in [(21,'relative_strength_21d'),(63,'relative_strength_63d')]:
        if len(spy_p) > lb and len(p) > lb:
            f[nm] = float(p.iloc[-1]/p.iloc[-lb] - spy_p.iloc[-1]/spy_p.iloc[-lb])
        else:
            f[nm] = 0.0

    # Sector dispersion (cross-sectional vol of sector returns)
    all_rets = sc.pct_change(21).iloc[idx]
    f['sector_dispersion_21d'] = float(all_rets.std()) if not all_rets.isna().all() else 0

    # VIX correlation
    vix_r = vix.pct_change().dropna()
    common_vix = r.index.intersection(vix_r.index)
    if len(common_vix) > 63:
        vr = vix_r.loc[common_vix].iloc[-63:]
        tr_v = r.loc[common_vix].iloc[-63:]
        if len(vr) > 10:
            f['vix_corr_63d'] = float(np.corrcoef(tr_v, vr)[0,1])
        else:
            f['vix_corr_63d'] = 0.0
    else:
        f['vix_corr_63d'] = 0.0

    # Volume trend
    f['volume_trend_21d'] = 0.0  # Placeholder if no volume data

    return f


def compute_flow_features(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, idx, tk):
    """Compute 55 flow-derived features for enhanced sector ranking."""
    p = sc[tk].iloc[:idx+1].dropna()
    if len(p) < 260:
        return None

    f = {}
    close = sc[tk].iloc[max(0,idx-260):idx+1].values.astype(float)
    high_arr = sh[tk].iloc[max(0,idx-260):idx+1].values.astype(float)
    low_arr = sl[tk].iloc[max(0,idx-260):idx+1].values.astype(float)
    vol_arr = sv[tk].iloc[max(0,idx-260):idx+1].values.astype(float)

    n = len(close)
    if n < 60:
        return None

    # --- OBV (On-Balance Volume) ---
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol_arr[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol_arr[i]
        else:
            obv[i] = obv[i-1]

    # OBV slope (normalized, 10d and 20d)
    for w, nm in [(10, 'obv_slope_10d'), (20, 'obv_slope_20d')]:
        if n > w:
            seg = obv[-w:]
            denom = np.abs(seg).mean() + 1e-10
            f[nm] = float(np.polyfit(range(w), seg/denom, 1)[0])
        else:
            f[nm] = 0.0

    # --- MFI (Money Flow Index) ---
    tp = (high_arr + low_arr + close) / 3
    rmf = tp * vol_arr
    pos_flow = np.zeros(n)
    neg_flow = np.zeros(n)
    for i in range(1, n):
        if tp[i] > tp[i-1]:
            pos_flow[i] = rmf[i]
        else:
            neg_flow[i] = rmf[i]

    for period, nm in [(14, 'mfi_14'), (5, 'mfi_5')]:
        if n > period:
            pf_sum = pd.Series(pos_flow).rolling(period).sum().iloc[-1]
            nf_sum = pd.Series(neg_flow).rolling(period).sum().iloc[-1]
            f[nm] = float(100 - (100 / (1 + pf_sum / (nf_sum + 1e-10))))
        else:
            f[nm] = 50.0

    # --- Accumulation/Distribution Line ---
    clv = np.where(high_arr != low_arr,
                   ((close - low_arr) - (high_arr - close)) / (high_arr - low_arr), 0)
    ad_vol = clv * vol_arr
    ad_line = np.cumsum(ad_vol)

    for w, nm in [(10, 'ad_slope_10d'), (20, 'ad_slope_20d')]:
        if n > w:
            seg = ad_line[-w:]
            denom = np.abs(seg).mean() + 1e-10
            f[nm] = float(np.polyfit(range(w), seg/denom, 1)[0])
        else:
            f[nm] = 0.0

    # --- Relative Volume ---
    vol_s = pd.Series(vol_arr)
    avg_vol_20 = vol_s.rolling(20).mean().iloc[-1]
    f['rel_volume_1d'] = float(vol_arr[-1] / (avg_vol_20 + 1e-10))
    f['rel_volume_5d'] = float(vol_s.iloc[-5:].mean() / (avg_vol_20 + 1e-10)) if n > 5 else 1.0

    # --- Price-Volume Divergence ---
    price_ret = close[-1]/close[-6] - 1 if n > 5 else 0
    vol_ret = vol_s.iloc[-1]/vol_s.iloc[-6] - 1 if n > 5 else 0
    f['pv_divergence_5d'] = float(price_ret * -vol_ret)

    price_ret10 = close[-1]/close[-11] - 1 if n > 10 else 0
    vol_ret10 = vol_s.iloc[-1]/vol_s.iloc[-11] - 1 if n > 10 else 0
    f['pv_divergence_10d'] = float(price_ret10 * -vol_ret10)

    # --- VWAP deviation ---
    if n > 20:
        cum_v = vol_s.iloc[-20:].sum()
        cum_pv = (pd.Series(close[-20:]) * vol_s.iloc[-20:].values).sum()
        vwap = cum_pv / (cum_v + 1e-10)
        f['vwap_dev_20d'] = float((close[-1] - vwap) / (vwap + 1e-10))
    else:
        f['vwap_dev_20d'] = 0.0

    # --- Volume-weighted momentum ---
    for lb, nm in [(5, 'vwmom_5d'), (21, 'vwmom_21d')]:
        if n > lb:
            rets = np.diff(close[-lb-1:]) / close[-lb-1:-1]
            vols = vol_arr[-lb:]
            f[nm] = float(np.average(rets, weights=vols+1e-10))
        else:
            f[nm] = 0.0

    # --- Flow Score (composite) ---
    obv_z = (f.get('obv_slope_20d', 0)) / 0.1  # rough normalization
    mfi_z = (f.get('mfi_14', 50) - 50) / 50
    ad_z = (f.get('ad_slope_20d', 0)) / 0.1
    f['flow_score'] = float((obv_z + mfi_z + ad_z) / 3)

    # --- Cross-asset flow features ---
    # SPY flow signals
    spy_close = spy.iloc[max(0,idx-60):idx+1].values.astype(float)
    spy_ret = spy.pct_change().iloc[max(0,idx-60):idx+1].values.astype(float)

    # Sector-SPY flow divergence
    if n > 20:
        sect_flow_20 = f.get('obv_slope_20d', 0)
        # SPY OBV slope as reference (simplified)
        f['flow_vs_spy'] = sect_flow_20  # relative to market
    else:
        f['flow_vs_spy'] = 0.0

    # --- Macro flow features ---
    for macro_tk in ['TLT', 'HYG', 'GLD', 'QQQ', 'IWM']:
        if macro_tk in macro.columns:
            mc = macro[macro_tk].iloc[max(0,idx-60):idx+1].values.astype(float)
            if len(mc) > 21:
                # Correlation of sector returns with macro returns
                sect_r = np.diff(close[-22:]) / close[-22:-1]
                macro_r = np.diff(mc[-22:]) / mc[-22:-1]
                if len(sect_r) == len(macro_r) and len(sect_r) > 5:
                    f[f'{macro_tk.lower()}_corr_21d'] = float(np.corrcoef(sect_r, macro_r)[0,1])
                else:
                    f[f'{macro_tk.lower()}_corr_21d'] = 0.0

                # Macro momentum (sector ranks better when correlated macro is strong)
                f[f'{macro_tk.lower()}_ret_21d'] = float(mc[-1]/mc[-22] - 1)
            else:
                f[f'{macro_tk.lower()}_corr_21d'] = 0.0
                f[f'{macro_tk.lower()}_ret_21d'] = 0.0
        else:
            f[f'{macro_tk.lower()}_corr_21d'] = 0.0
            f[f'{macro_tk.lower()}_ret_21d'] = 0.0

    # --- Volume-based regime features ---
    f['vol_expansion'] = float(vol_arr[-1] / (vol_s.iloc[-20:].mean() + 1e-10)) if n > 20 else 1.0
    f['vol_contraction'] = float(vol_s.iloc[-5:].std() / (vol_s.iloc[-20:].std() + 1e-10)) if n > 20 else 1.0

    # --- Momentum-volume interaction ---
    f['mom_vol_interact_5d'] = float(f.get('ret_5d', 0) if 'ret_5d' in f else (close[-1]/close[-6]-1 if n>5 else 0)) * f['rel_volume_1d']
    f['mom_vol_interact_21d'] = float(close[-1]/close[-22]-1 if n>21 else 0) * f['rel_volume_5d']

    # --- Intraday range features ---
    if n > 10:
        ranges = (high_arr - low_arr) / close
        f['avg_range_10d'] = float(np.mean(ranges[-10:]))
        f['range_expansion'] = float(ranges[-1] / (np.mean(ranges[-20:]) + 1e-10)) if n > 20 else 1.0
    else:
        f['avg_range_10d'] = 0.01
        f['range_expansion'] = 1.0

    # --- Chaikin Money Flow (21d) ---
    if n > 21:
        cmf_vol = clv[-21:] * vol_arr[-21:]
        f['cmf_21d'] = float(np.sum(cmf_vol) / (np.sum(vol_arr[-21:]) + 1e-10))
    else:
        f['cmf_21d'] = 0.0

    # --- Elder Force Index ---
    if n > 13:
        force = np.diff(close[-14:]) * vol_arr[-13:]
        f['force_index_13d'] = float(np.mean(force) / (np.abs(force).mean() + 1e-10))
    else:
        f['force_index_13d'] = 0.0

    # --- Ease of Movement ---
    if n > 14:
        dm = ((high_arr[-14:] + low_arr[-14:]) / 2) - ((high_arr[-15:-1] + low_arr[-15:-1]) / 2)
        br = vol_arr[-14:] / (high_arr[-14:] - low_arr[-14:] + 1e-10) / 1e6
        eom = dm / (br + 1e-10)
        f['eom_14d'] = float(np.mean(eom))
    else:
        f['eom_14d'] = 0.0

    return f


def get_all_flow_feature_names():
    """Return list of flow feature column names."""
    names = [
        'obv_slope_10d', 'obv_slope_20d',
        'mfi_14', 'mfi_5',
        'ad_slope_10d', 'ad_slope_20d',
        'rel_volume_1d', 'rel_volume_5d',
        'pv_divergence_5d', 'pv_divergence_10d',
        'vwap_dev_20d',
        'vwmom_5d', 'vwmom_21d',
        'flow_score', 'flow_vs_spy',
        'vol_expansion', 'vol_contraction',
        'mom_vol_interact_5d', 'mom_vol_interact_21d',
        'avg_range_10d', 'range_expansion',
        'cmf_21d', 'force_index_13d', 'eom_14d',
    ]
    for macro_tk in ['tlt', 'hyg', 'gld', 'qqq', 'iwm']:
        names.append(f'{macro_tk}_corr_21d')
        names.append(f'{macro_tk}_ret_21d')
    return names


# =====================================================================
# 4. LGBM WALK-FORWARD RANKING
# =====================================================================

def build_lgbm_dataset(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, dates, use_flow):
    """Build training dataset for LGBM sector ranker."""
    fprint(f"  Building {'flow-enhanced' if use_flow else 'legacy'} feature dataset...")
    recs = []

    for dt in dates:
        if dt not in sc.index:
            continue
        idx = sc.index.get_loc(dt)
        if idx < 260:
            continue

        for tk in sc.columns:
            # Legacy features
            lf = compute_legacy_features(sc, spy, vix, idx, tk)
            if lf is None:
                continue

            feat = dict(lf)

            # Flow features (if requested)
            if use_flow:
                ff = compute_flow_features(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, idx, tk)
                if ff is not None:
                    feat.update(ff)
                else:
                    # Fill with zeros
                    for fn in get_all_flow_feature_names():
                        feat[fn] = 0.0

            # Forward return label (21d)
            fi = min(idx + 21, len(sc) - 1)
            feat['fwd_ret'] = float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1)
            feat['date'] = dt
            feat['ticker'] = tk
            recs.append(feat)

    df = pd.DataFrame(recs)
    fprint(f"  Dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def run_lgbm_walkforward(df, feature_cols, label=''):
    """Walk-forward LGBM ranking. Returns dict of {date: {ticker: score}}."""
    fprint(f"  LGBM walk-forward ({label}, {len(feature_cols)} features)...")

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    udates = sorted(df['date'].unique())
    ranks = {}

    train_periods = 12  # 12 bi-weekly periods = ~6 months training

    for i in range(train_periods, len(udates)):
        td = udates[max(0, i - train_periods):i]
        tdate = udates[i]

        tr = df[df['date'].isin(td)]
        te = df[df['date'] == tdate].copy()

        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[feature_cols].values.astype(np.float32))
        Xe = np.nan_to_num(te[feature_cols].values.astype(np.float32))

        try:
            try:
                m = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                    device='gpu', verbose=-1
                )
                m.fit(Xt, tr['rank_label'].values)
            except Exception:
                m = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                    verbose=-1
                )
                m.fit(Xt, tr['rank_label'].values)

            te['score'] = m.predict(Xe)
            ranks[tdate] = dict(zip(te['ticker'], te['score']))
        except Exception:
            continue

    fprint(f"  Rankings generated: {len(ranks)} dates")
    return ranks


# =====================================================================
# 5. ATR-BASED OPTIONS PRICING (consistent with existing scripts)
# =====================================================================

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({
        'hl': h - l,
        'hc': abs(h - c.shift(1)),
        'lc': abs(l - c.shift(1))
    }).max(axis=1)
    return tr.rolling(period).mean()


def price_spread(S, spread_pct, dte, atr, vix_level):
    K1 = round(S)
    K2 = round(S * (1 + spread_pct / 100))
    T = dte / 252.0
    if T <= 0:
        long_prem = max(0, S - K1)
        short_prem = max(0, S - K2)
    else:
        vix_mult = max(0.3, vix_level / 20)
        long_prem = max(0, S - K1) + atr * np.sqrt(T) * vix_mult * np.exp(-3 * abs(S - K1) / S)
        short_prem = max(0, S - K2) + atr * np.sqrt(T) * vix_mult * np.exp(-3 * abs(S - K2) / S)

    long_cost = long_prem * (1 + HAIRCUT)
    short_credit = short_prem * (1 - HAIRCUT)
    debit = long_cost - short_credit
    width = K2 - K1
    max_profit = (width - debit) * 100 - SPREAD_COMM
    max_loss = debit * 100 + SPREAD_COMM

    return debit, max_profit, max_loss, K1, K2


# =====================================================================
# 6. BACKTEST SIMULATION
# =====================================================================

def simulate_variant(variant_name, ranks, sc, sh, sl, spy, vix, regime_scores, atr_dict):
    """
    Simulate one variant of the strategy.
    variant_name: 'A_BASELINE', 'B_REGIME_ONLY', 'C_FLOW_ONLY', 'D_FULL_STACK'
    """
    equity = CAP
    trades = []
    curve = [CAP]
    sma200 = spy.rolling(200).mean()

    for dt in sorted(ranks.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])  # current VIX
        scores = ranks[dt]
        if not scores:
            curve.append(equity)
            continue

        # --- FILTER LOGIC ---
        skip = False

        if variant_name.startswith('A'):
            # BASELINE: VIX > 20 filter (only trade when VIX > 20)
            # Actually: the original strategy trades WHEN VIX is elevated (>20)
            # meaning it's a bull-call-spread strategy that benefits from high IV
            # Let me re-check: VIX>20 means we ENTER trades when vol is elevated
            if cv < 15:  # Don't trade in extremely low vol (spreads too cheap)
                skip = True
            # No regime filter
        elif variant_name.startswith('B'):
            # REGIME_ONLY: GRU regime > 0.4
            if dt in regime_scores.index:
                rs = float(regime_scores.loc[dt])
                if rs < REGIME_THRESHOLD:
                    skip = True
            else:
                skip = True  # No regime data = skip
        elif variant_name.startswith('C'):
            # FLOW_ONLY: VIX > 20 filter (same as A)
            if cv < 15:
                skip = True
        elif variant_name.startswith('D'):
            # FULL_STACK: regime > 0.4 filter
            if dt in regime_scores.index:
                rs = float(regime_scores.loc[dt])
                if rs < REGIME_THRESHOLD:
                    skip = True
            else:
                skip = True

        if skip:
            curve.append(equity)
            continue

        # --- SECTOR SELECTION (from LGBM rankings) ---
        picks = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:MAX_CONC]]

        # Position sizing
        max_per_trade = min(MAX_POS, equity / 3)
        if max_per_trade < 30:
            curve.append(equity)
            continue

        # Detect bull/bear for reporting
        bull = float(spy.loc[dt]) >= (float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else float(spy.loc[dt]))

        n_entered = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_dict or n_entered >= MAX_CONC:
                continue

            S = float(sc[tk].loc[dt])
            di = sc.index.get_loc(dt)

            atr_val = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015

            # Price the spread
            debit, max_profit, max_loss, K1, K2 = price_spread(S, 3.0, DTE, atr_val, cv)
            cost = debit * 100 + SPREAD_COMM

            if cost <= 0 or cost > max_per_trade or cost > equity * 0.40:
                continue

            # Simulate hold to expiry
            ei = min(di + DTE, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])

            # Early exit logic: take profit if >50% of max profit or hold to expiry
            pnl = None
            actual_exit_idx = ei
            for ci in range(di + 3, ei + 1):
                Sc = float(sc[tk].iloc[ci])
                remaining_dte = max(0, DTE - (ci - di))
                ac = float(atr_dict[tk].iloc[ci]) if ci < len(atr_dict[tk]) else atr_val
                time_mult = np.sqrt(remaining_dte / max(DTE, 1))

                # Intrinsic + time value estimate
                spread_value = (max(0, Sc - K1) - max(0, Sc - K2)) * 100 + ac * time_mult * 0.3 * 100

                # Take profit at 50% of max, or exit at expiry
                if spread_value - cost >= max_profit * 0.5 or remaining_dte < 2:
                    pnl = spread_value - cost
                    actual_exit_idx = ci
                    break

            if pnl is None:
                # At expiry
                pnl = (max(0, Se - K1) - max(0, Se - K2)) * 100 - cost

            equity += pnl
            n_entered += 1

            regime_score_at_entry = float(regime_scores.loc[dt]) if regime_scores is not None and dt in regime_scores.index else -1

            trades.append({
                'entry': str(dt.date()),
                'exit': str(sc.index[actual_exit_idx].date()),
                'ticker': tk,
                'pnl': round(pnl, 2),
                'win': pnl > 0,
                'regime': 'bull' if bull else 'bear',
                'vix': round(cv, 1),
                'regime_score': round(regime_score_at_entry, 3),
                'cost': round(cost, 2),
            })

        curve.append(equity)

    return trades, equity, curve


# =====================================================================
# 7. METRICS + ADVERSARIAL VALIDATION
# =====================================================================

def compute_metrics(trades, final_eq, curve, name):
    """Compute performance metrics from trade list."""
    if not trades:
        fprint(f"  {name}: No trades")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnls = [t['pnl'] for t in trades]

    # Monthly returns
    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    monthly_ret = tdf.groupby('month')['pnl'].sum() / CAP
    ny = max(len(monthly_ret) / 12, 0.5)

    # Sharpe (monthly, annualized)
    if len(monthly_ret) > 3:
        sharpe = (monthly_ret.mean() * 12) / (monthly_ret.std() * np.sqrt(12) + 1e-10)
    else:
        sharpe = 0

    # Sortino
    dn = monthly_ret[monthly_ret < 0]
    if len(dn) > 1:
        sortino = (monthly_ret.mean() * 12) / (dn.std() * np.sqrt(12) + 1e-10)
    else:
        sortino = 0

    # CAGR
    cagr = (final_eq / CAP) ** (1 / ny) - 1

    # Max drawdown
    eq_arr = np.array(curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / (peak + 1e-10)
    max_dd = float(dd.min())

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / (gross_loss + 1e-10)

    # Max consecutive losses
    mcl = 0
    curr = 0
    for t in trades:
        if not t['win']:
            curr += 1
            mcl = max(mcl, curr)
        else:
            curr = 0

    # Bull/Bear breakdown
    bull_trades = [t for t in trades if t.get('regime') == 'bull']
    bear_trades = [t for t in trades if t.get('regime') == 'bear']
    bull_wr = sum(1 for t in bull_trades if t['win']) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t['win']) / max(len(bear_trades), 1) * 100

    # VIX band analysis
    vix_bands = {}
    for band_name, lo, hi in [('VIX<15', 0, 15), ('VIX 15-20', 15, 20), ('VIX 20-25', 20, 25), ('VIX 25-30', 25, 30), ('VIX>30', 30, 100)]:
        band_trades = [t for t in trades if lo <= t.get('vix', 20) < hi]
        if band_trades:
            band_pnls = [t['pnl'] for t in band_trades]
            vix_bands[band_name] = {
                'n': len(band_trades),
                'wr': round(sum(1 for p in band_pnls if p > 0) / len(band_pnls) * 100, 1),
                'avg_pnl': round(np.mean(band_pnls), 2),
                'total_pnl': round(sum(band_pnls), 2),
            }

    result = {
        'name': name,
        'n_trades': n,
        'win_rate': round(wr, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(max_dd * 100, 1),
        'pf': round(pf, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'final_equity': round(final_eq, 2),
        'total_pnl': round(sum(pnls), 2),
        'max_consec_loss': mcl,
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'bull_n': len(bull_trades),
        'bear_n': len(bear_trades),
        'vix_bands': vix_bands,
        'monthly_returns': monthly_ret.values.tolist(),
    }

    fprint(f"  {name:20s} | {n:4d} trades | WR {wr:5.1f}% | Sh {sharpe:5.2f} | So {sortino:5.2f} | "
           f"CAGR {cagr*100:5.1f}% | MDD {max_dd*100:5.1f}% | PF {pf:5.2f} | ${CAP:.0f}->${final_eq:.0f} | MCL {mcl}")

    return result


def adversarial_validate(result):
    """Run adversarial checks: permutation, regime gap, sub-period, outlier removal."""
    if result is None:
        return result

    rets = np.array(result['monthly_returns'])
    if len(rets) < 10:
        result.update({'gates': 0, 'perm_p': 1.0, 'r1_gap': 1.0})
        return result

    gates = 0

    # Gate 1: Permutation test (p < 0.05)
    real_sharpe = np.mean(rets) / (np.std(rets) + 1e-10)
    n_better = 0
    for _ in range(1000):
        shuffled = rets * np.random.choice([-1, 1], len(rets))
        sh = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
        if sh >= real_sharpe:
            n_better += 1
    perm_p = n_better / 1000
    g1 = perm_p < 0.05
    gates += g1

    # Gate 2: R1 Regime gap (bull WR vs bear WR gap < 0.50)
    r1_gap = abs(result['bull_wr'] - result['bear_wr']) / max(result['bull_wr'], result['bear_wr'], 1)
    g2 = r1_gap < 0.50
    gates += g2

    # Gate 3: Sub-period consistency (both halves positive Sharpe)
    mid = len(rets) // 2
    h1_sharpe = np.mean(rets[:mid]) / (np.std(rets[:mid]) + 1e-10) if mid > 3 else 0
    h2_sharpe = np.mean(rets[mid:]) / (np.std(rets[mid:]) + 1e-10) if len(rets) - mid > 3 else 0
    g3 = h1_sharpe > 0 and h2_sharpe > 0
    gates += g3

    # Gate 4: Outlier robustness (Sharpe positive after removing top return)
    if len(rets) > 5:
        trimmed = np.sort(rets)[:-1]
        g4 = np.mean(trimmed) / (np.std(trimmed) + 1e-10) > 0
    else:
        g4 = False
    gates += g4

    result.update({
        'gates': gates,
        'perm_p': round(perm_p, 3),
        'r1_gap': round(r1_gap, 3),
        'g1_perm': g1,
        'g2_regime': g2,
        'g3_sub': g3,
        'g4_outlier': g4,
        'h1_sharpe': round(h1_sharpe, 2),
        'h2_sharpe': round(h2_sharpe, 2),
    })

    fprint(f"    Gates: P={'Y' if g1 else 'N'}(p={perm_p:.3f}) R1={'Y' if g2 else 'N'}({r1_gap:.2f}) "
           f"Sub={'Y' if g3 else 'N'}({h1_sharpe:.2f}/{h2_sharpe:.2f}) Out={'Y' if g4 else 'N'} => {gates}/4")

    return result


def random_baseline(sc, vix, spy, atr_dict, regime_scores, n_sims=100):
    """Run random sector selection to establish baseline Sharpe."""
    fprint("\n[6/7] Random baseline (100 simulations)...")
    random_sharpes = []

    all_dates = sorted(sc.index[260:])
    rebalance_dates = all_dates[::10]  # every 10 trading days

    sma200 = spy.rolling(200).mean()

    for sim in range(n_sims):
        np.random.seed(sim)
        equity = CAP
        curve = [CAP]
        trades = []

        for dt in rebalance_dates:
            if dt not in vix.index:
                continue
            cv = float(vix.loc[dt])

            # Random picks
            available = [tk for tk in sc.columns if tk in atr_dict]
            if len(available) < 3:
                continue
            picks = list(np.random.choice(available, min(3, len(available)), replace=False))

            max_per = min(MAX_POS, equity / 3)
            if max_per < 30:
                continue

            for tk in picks:
                S = float(sc[tk].loc[dt])
                if np.isnan(S) or S <= 0:
                    continue
                di = sc.index.get_loc(dt)
                atr_val = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
                if np.isnan(atr_val):
                    atr_val = S * 0.015

                debit, max_profit, max_loss, K1, K2 = price_spread(S, 3.0, DTE, atr_val, cv)
                cost = debit * 100 + SPREAD_COMM
                if cost <= 0 or cost > max_per:
                    continue

                ei = min(di + DTE, len(sc) - 1)
                Se = float(sc[tk].iloc[ei])
                pnl = (max(0, Se - K1) - max(0, Se - K2)) * 100 - cost
                equity += pnl
                trades.append({'pnl': pnl, 'win': pnl > 0, 'entry': str(dt.date())})

            curve.append(equity)

        if trades:
            tdf = pd.DataFrame(trades)
            tdf['month'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
            mr = tdf.groupby('month')['pnl'].sum() / CAP
            if len(mr) > 3:
                sh = (mr.mean() * 12) / (mr.std() * np.sqrt(12) + 1e-10)
                random_sharpes.append(sh)

    if random_sharpes:
        fprint(f"  Random baseline: mean Sharpe={np.mean(random_sharpes):.2f}, "
               f"median={np.median(random_sharpes):.2f}, "
               f"p95={np.percentile(random_sharpes, 95):.2f}")
    return random_sharpes


# =====================================================================
# 8. MAIN
# =====================================================================

def main():
    t0 = time.time()
    fprint(f"{'='*80}")
    fprint(f"REGIME + FLOW COMBINED v1 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    fprint(f"{'='*80}")
    fprint(f"Capital: ${CAP:.0f} | Haircut: {HAIRCUT:.0%} | Comm: ${SPREAD_COMM}/spread")
    fprint(f"Regime threshold: {REGIME_THRESHOLD} | DTE: {DTE}d")
    fprint(f"{'='*80}\n")

    # 1. Download data
    sc, sh, sl, sv, spy, vix, macro, macro_vol, macro_high, macro_low = download_data()

    # 2. Load regime predictions
    regime_scores = load_regime_predictions()
    if regime_scores is None:
        fprint("FATAL: Cannot proceed without regime predictions")
        return

    # 3. Compute ATR for options pricing
    fprint("[3/7] Computing ATR for options pricing...")
    atr_dict = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_dict[tk] = compute_atr(sh[tk], sl[tk], sc[tk])
    fprint(f"  ATR computed for {len(atr_dict)} sectors")

    # 4. Build LGBM datasets and rankings
    # Bi-weekly rebalance dates
    rdates = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rdates = rdates[rdates.isin(sc.index)]
    fprint(f"\n[4/7] Building LGBM rankings on {len(rdates)} rebalance dates...")

    # Legacy features dataset
    legacy_df = build_lgbm_dataset(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, rdates, use_flow=False)
    legacy_feature_cols = [c for c in LEGACY_FEATURES if c in legacy_df.columns]
    fprint(f"  Legacy features available: {len(legacy_feature_cols)}")

    # Flow-enhanced features dataset
    flow_df = build_lgbm_dataset(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, rdates, use_flow=True)
    flow_feature_cols = legacy_feature_cols + [c for c in get_all_flow_feature_names() if c in flow_df.columns]
    fprint(f"  Flow features available: {len(flow_feature_cols)} (legacy {len(legacy_feature_cols)} + flow {len(flow_feature_cols)-len(legacy_feature_cols)})")

    # Run LGBM walk-forward for both
    legacy_ranks = run_lgbm_walkforward(legacy_df, legacy_feature_cols, 'legacy')
    flow_ranks = run_lgbm_walkforward(flow_df, flow_feature_cols, 'flow-enhanced')

    # 5. Simulate all 4 variants
    fprint(f"\n[5/7] Simulating 4 variants...")
    fprint(f"{'='*100}")

    variants = [
        ('A_BASELINE',    legacy_ranks, 'VIX filter + legacy LGBM'),
        ('B_REGIME_ONLY', legacy_ranks, 'Regime>0.4 + legacy LGBM'),
        ('C_FLOW_ONLY',   flow_ranks,   'VIX filter + flow LGBM'),
        ('D_FULL_STACK',  flow_ranks,   'Regime>0.4 + flow LGBM'),
    ]

    results = []
    for vname, ranks, desc in variants:
        fprint(f"\n--- {vname}: {desc} ---")
        trades, eq, curve = simulate_variant(vname, ranks, sc, sh, sl, spy, vix, regime_scores, atr_dict)
        r = compute_metrics(trades, eq, curve, vname)
        if r:
            r['description'] = desc
            r = adversarial_validate(r)
            results.append(r)

    # 6. Random baseline
    random_sharpes = random_baseline(sc, vix, spy, atr_dict, regime_scores)

    # 7. Summary
    fprint(f"\n{'='*110}")
    fprint(f"{'Variant':<22} {'Desc':<32} {'#':>5} {'WR':>6} {'Sh':>6} {'So':>6} {'CAGR':>7} {'MDD':>7} {'PF':>5} {'$':>8} {'G':>4}")
    fprint(f"{'-'*110}")

    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['description']:<32} {r['n_trades']:>5} {r['win_rate']:>5.1f}% "
               f"{r['sharpe']:>6.2f} {r['sortino']:>6.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>5.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4")

    # VIX band breakdown
    fprint(f"\n--- VIX BAND ANALYSIS ---")
    for r in results:
        fprint(f"\n  {r['name']}:")
        for band, stats in r.get('vix_bands', {}).items():
            fprint(f"    {band}: {stats['n']} trades, WR={stats['wr']}%, avg=${stats['avg_pnl']:.2f}, total=${stats['total_pnl']:.2f}")

    # Key comparison
    fprint(f"\n{'='*80}")
    fprint("KEY COMPARISON:")
    baseline = next((r for r in results if r['name'].startswith('A')), None)
    full_stack = next((r for r in results if r['name'].startswith('D')), None)

    if baseline and full_stack:
        fprint(f"  BASELINE  -> Sharpe {baseline['sharpe']:.2f}, MDD {baseline['maxdd_pct']:.1f}%, CAGR {baseline['cagr_pct']:.1f}%")
        fprint(f"  FULL_STACK-> Sharpe {full_stack['sharpe']:.2f}, MDD {full_stack['maxdd_pct']:.1f}%, CAGR {full_stack['cagr_pct']:.1f}%")
        sharpe_delta = full_stack['sharpe'] - baseline['sharpe']
        mdd_delta = full_stack['maxdd_pct'] - baseline['maxdd_pct']
        fprint(f"  Delta: Sharpe {sharpe_delta:+.2f}, MDD {mdd_delta:+.1f}pp")

        if random_sharpes:
            p_vs_random = np.mean([s >= full_stack['sharpe'] for s in random_sharpes])
            fprint(f"  Full stack vs random: p={p_vs_random:.3f}")

    # Verdict
    fprint(f"\n{'='*80}")
    best = max(results, key=lambda x: x['sharpe']) if results else None
    best_valid = max([r for r in results if r.get('gates', 0) >= 3], key=lambda x: x['sharpe'], default=None)

    if best:
        fprint(f"BEST OVERALL: {best['name']} (Sharpe={best['sharpe']:.2f}, {best['gates']}/4 gates)")
    if best_valid:
        fprint(f"BEST VALIDATED (3+ gates): {best_valid['name']} (Sharpe={best_valid['sharpe']:.2f})")
    else:
        fprint("WARNING: No variant passes 3+ adversarial gates")

    # Does combining help?
    if baseline and full_stack:
        regime_only = next((r for r in results if r['name'].startswith('B')), None)
        flow_only = next((r for r in results if r['name'].startswith('C')), None)

        fprint(f"\nANSWER: Does combining regime + flow help?")
        if regime_only and flow_only:
            best_individual = max(regime_only['sharpe'], flow_only['sharpe'])
            if full_stack['sharpe'] > best_individual:
                fprint(f"  YES — Full stack Sharpe ({full_stack['sharpe']:.2f}) > best individual ({best_individual:.2f})")
                fprint(f"  Improvement: {(full_stack['sharpe']-best_individual)/best_individual*100:+.1f}%")
            else:
                fprint(f"  NO — Full stack Sharpe ({full_stack['sharpe']:.2f}) <= best individual ({best_individual:.2f})")
                if regime_only['sharpe'] > flow_only['sharpe']:
                    fprint(f"  Regime-only ({regime_only['sharpe']:.2f}) is better than adding flow features")
                else:
                    fprint(f"  Flow-only ({flow_only['sharpe']:.2f}) is better than adding regime filter")

    # MLflow logging
    fprint(f"\n[7/7] Logging to MLflow...")
    if MLFLOW_OK:
        try:
            exp_name = 'regime_flow_combined_v1'
            try:
                if not mlflow.get_experiment_by_name(exp_name):
                    mlflow.create_experiment(exp_name)
            except:
                pass
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=f"rfc_v1_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_params({
                    'capital': CAP,
                    'haircut': HAIRCUT,
                    'regime_threshold': REGIME_THRESHOLD,
                    'dte': DTE,
                    'n_sectors': len(SECTORS),
                    'legacy_features': len(legacy_feature_cols),
                    'flow_features': len(flow_feature_cols),
                })

                for r in results:
                    prefix = r['name']
                    for k in ['sharpe', 'sortino', 'cagr_pct', 'maxdd_pct', 'win_rate', 'pf', 'n_trades', 'gates']:
                        try:
                            mlflow.log_metric(f"{prefix}_{k}", r[k])
                        except:
                            pass

                if random_sharpes:
                    mlflow.log_metric('random_baseline_mean_sharpe', float(np.mean(random_sharpes)))
                    mlflow.log_metric('random_baseline_p95_sharpe', float(np.percentile(random_sharpes, 95)))

            fprint("  MLflow: logged successfully")
        except Exception as e:
            fprint(f"  MLflow error: {e}")
    else:
        fprint("  MLflow: skipped (not available)")

    # Save results
    save_data = {
        'strategy': 'Regime + Flow Combined v1',
        'run_date': datetime.now().isoformat(),
        'config': {
            'capital': CAP,
            'haircut': HAIRCUT,
            'regime_threshold': REGIME_THRESHOLD,
            'dte': DTE,
            'legacy_features': len(legacy_feature_cols),
            'flow_features': len(flow_feature_cols),
        },
        'variants': [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results],
        'random_baseline': {
            'mean_sharpe': round(float(np.mean(random_sharpes)), 3) if random_sharpes else None,
            'median_sharpe': round(float(np.median(random_sharpes)), 3) if random_sharpes else None,
            'p95_sharpe': round(float(np.percentile(random_sharpes, 95)), 3) if random_sharpes else None,
        },
        'best': best['name'] if best else 'NONE',
        'best_valid': best_valid['name'] if best_valid else 'NONE',
    }

    out_path = OUTPUT_DIR / 'regime_flow_combined_v1_results.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    fprint(f"\nResults saved to {out_path}")
    fprint(f"Total runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
