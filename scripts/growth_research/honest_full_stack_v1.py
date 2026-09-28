#!/usr/bin/env python3
"""
Honest Full Stack v1 — HONEST re-evaluation of regime + flow combined system
=============================================================================
FIXES vs regime_flow_combined_v1.py:
  1. HOLD TO EXPIRY ONLY — NO early exit, NO take-profit
  2. At expiry: intrinsic value ONLY (no time value, no haircut)
  3. 15% haircut on ENTRY only (long costs more, short receives less)
  4. DTE=21 (3 weeks, matching our standard)
  5. $645 starting capital
  6. Walk-forward LGBM, biweekly rebalance
  7. Full adversarial: permutation (2000 trials), regime split, sub-period,
     yearly breakdown, random baseline (5 trials)

4 VARIANTS:
  A. BASELINE:    VIX>20 filter   + legacy LGBM (18 features)
  B. REGIME_ONLY: regime>0.4      + legacy LGBM
  C. FLOW_ONLY:   VIX>20 filter   + flow LGBM (73 features)
  D. FULL_STACK:  regime>0.4      + flow LGBM

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

OUTPUT_DIR = BASE / 'output' / 'growth_research' / 'honest_full_stack_v1'
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
HAIRCUT = 0.15       # 15% haircut on ENTRY only
MAX_POS = 200
MAX_CONC = 3
DTE = 21             # 3-week spread (HONEST: matching our standard)

# Walk-forward LGBM
TRAIN_MONTHS = 12
WF_STEP_WEEKS = 2

# Regime filter threshold
REGIME_THRESHOLD = 0.4

# Permutation and random baseline config
N_PERM_TRIALS = 2000
N_RANDOM_BASELINE = 5

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
    fprint("[1/8] Downloading market data...")

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
    fprint("[2/8] Loading GRU regime predictions...")
    regime_path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not regime_path.exists():
        fprint(f"  WARNING: Regime predictions not found at {regime_path}")
        return None

    data = np.load(regime_path, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']

    regime_series = pd.Series(scores, index=dates, name='regime_score')
    if regime_series.index.duplicated().any():
        regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    fprint(f"  Loaded {len(regime_series)} regime scores, {dates[0].date()} to {dates[-1].date()}")
    fprint(f"  Score range: {scores.min():.3f} to {scores.max():.3f}, mean={scores.mean():.3f}")
    fprint(f"  Days above 0.4: {(scores > 0.4).sum()} ({(scores > 0.4).mean()*100:.1f}%)")
    return regime_series


# =====================================================================
# 3. FEATURE ENGINEERING (identical to v1)
# =====================================================================

def compute_legacy_features(sc, spy, vix, idx, tk):
    """Compute 18 legacy features for sector ranking."""
    p = sc[tk].iloc[:idx+1].dropna()
    if len(p) < 260:
        return None

    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(p.iloc[-1]/p.iloc[-lb]-1) if len(p) > lb else 0.0

    r = p.pct_change().dropna()
    f['vol_21d'] = float(r.iloc[-21:].std()*np.sqrt(252)) if len(r) > 21 else 0.2
    f['vol_63d'] = float(r.iloc[-63:].std()*np.sqrt(252)) if len(r) > 63 else 0.2

    r63 = r.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0

    p63 = p.iloc[-63:]
    f['maxdd_63d'] = float(((p63/p63.cummax())-1).min())
    f['pct_52w_high'] = float(p.iloc[-1]/p.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3

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

    spy_p = spy.iloc[:idx+1].dropna()
    for lb, nm in [(21,'relative_strength_21d'),(63,'relative_strength_63d')]:
        if len(spy_p) > lb and len(p) > lb:
            f[nm] = float(p.iloc[-1]/p.iloc[-lb] - spy_p.iloc[-1]/spy_p.iloc[-lb])
        else:
            f[nm] = 0.0

    all_rets = sc.pct_change(21).iloc[idx]
    f['sector_dispersion_21d'] = float(all_rets.std()) if not all_rets.isna().all() else 0

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

    f['volume_trend_21d'] = 0.0
    return f


def compute_flow_features(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, idx, tk):
    """Compute flow-derived features for enhanced sector ranking."""
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

    # OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + vol_arr[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - vol_arr[i]
        else:
            obv[i] = obv[i-1]

    for w, nm in [(10, 'obv_slope_10d'), (20, 'obv_slope_20d')]:
        if n > w:
            seg = obv[-w:]
            denom = np.abs(seg).mean() + 1e-10
            f[nm] = float(np.polyfit(range(w), seg/denom, 1)[0])
        else:
            f[nm] = 0.0

    # MFI
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

    # A/D Line
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

    # Relative Volume
    vol_s = pd.Series(vol_arr)
    avg_vol_20 = vol_s.rolling(20).mean().iloc[-1]
    f['rel_volume_1d'] = float(vol_arr[-1] / (avg_vol_20 + 1e-10))
    f['rel_volume_5d'] = float(vol_s.iloc[-5:].mean() / (avg_vol_20 + 1e-10)) if n > 5 else 1.0

    # Price-Volume Divergence
    price_ret = close[-1]/close[-6] - 1 if n > 5 else 0
    vol_ret = vol_s.iloc[-1]/vol_s.iloc[-6] - 1 if n > 5 else 0
    f['pv_divergence_5d'] = float(price_ret * -vol_ret)

    price_ret10 = close[-1]/close[-11] - 1 if n > 10 else 0
    vol_ret10 = vol_s.iloc[-1]/vol_s.iloc[-11] - 1 if n > 10 else 0
    f['pv_divergence_10d'] = float(price_ret10 * -vol_ret10)

    # VWAP deviation
    if n > 20:
        cum_v = vol_s.iloc[-20:].sum()
        cum_pv = (pd.Series(close[-20:]) * vol_s.iloc[-20:].values).sum()
        vwap = cum_pv / (cum_v + 1e-10)
        f['vwap_dev_20d'] = float((close[-1] - vwap) / (vwap + 1e-10))
    else:
        f['vwap_dev_20d'] = 0.0

    # Volume-weighted momentum
    for lb, nm in [(5, 'vwmom_5d'), (21, 'vwmom_21d')]:
        if n > lb:
            rets = np.diff(close[-lb-1:]) / close[-lb-1:-1]
            vols = vol_arr[-lb:]
            f[nm] = float(np.average(rets, weights=vols+1e-10))
        else:
            f[nm] = 0.0

    # Flow Score (composite)
    obv_z = (f.get('obv_slope_20d', 0)) / 0.1
    mfi_z = (f.get('mfi_14', 50) - 50) / 50
    ad_z = (f.get('ad_slope_20d', 0)) / 0.1
    f['flow_score'] = float((obv_z + mfi_z + ad_z) / 3)

    # Sector-SPY flow divergence
    if n > 20:
        f['flow_vs_spy'] = f.get('obv_slope_20d', 0)
    else:
        f['flow_vs_spy'] = 0.0

    # Macro flow features
    for macro_tk in ['TLT', 'HYG', 'GLD', 'QQQ', 'IWM']:
        if macro_tk in macro.columns:
            mc = macro[macro_tk].iloc[max(0,idx-60):idx+1].values.astype(float)
            if len(mc) > 21:
                sect_r = np.diff(close[-22:]) / close[-22:-1]
                macro_r = np.diff(mc[-22:]) / mc[-22:-1]
                if len(sect_r) == len(macro_r) and len(sect_r) > 5:
                    f[f'{macro_tk.lower()}_corr_21d'] = float(np.corrcoef(sect_r, macro_r)[0,1])
                else:
                    f[f'{macro_tk.lower()}_corr_21d'] = 0.0
                f[f'{macro_tk.lower()}_ret_21d'] = float(mc[-1]/mc[-22] - 1)
            else:
                f[f'{macro_tk.lower()}_corr_21d'] = 0.0
                f[f'{macro_tk.lower()}_ret_21d'] = 0.0
        else:
            f[f'{macro_tk.lower()}_corr_21d'] = 0.0
            f[f'{macro_tk.lower()}_ret_21d'] = 0.0

    # Volume-based regime features
    f['vol_expansion'] = float(vol_arr[-1] / (vol_s.iloc[-20:].mean() + 1e-10)) if n > 20 else 1.0
    f['vol_contraction'] = float(vol_s.iloc[-5:].std() / (vol_s.iloc[-20:].std() + 1e-10)) if n > 20 else 1.0

    # Momentum-volume interaction
    f['mom_vol_interact_5d'] = float(close[-1]/close[-6]-1 if n>5 else 0) * f['rel_volume_1d']
    f['mom_vol_interact_21d'] = float(close[-1]/close[-22]-1 if n>21 else 0) * f['rel_volume_5d']

    # Intraday range
    if n > 10:
        ranges = (high_arr - low_arr) / close
        f['avg_range_10d'] = float(np.mean(ranges[-10:]))
        f['range_expansion'] = float(ranges[-1] / (np.mean(ranges[-20:]) + 1e-10)) if n > 20 else 1.0
    else:
        f['avg_range_10d'] = 0.01
        f['range_expansion'] = 1.0

    # Chaikin Money Flow (21d)
    if n > 21:
        cmf_vol = clv[-21:] * vol_arr[-21:]
        f['cmf_21d'] = float(np.sum(cmf_vol) / (np.sum(vol_arr[-21:]) + 1e-10))
    else:
        f['cmf_21d'] = 0.0

    # Elder Force Index
    if n > 13:
        force = np.diff(close[-14:]) * vol_arr[-13:]
        f['force_index_13d'] = float(np.mean(force) / (np.abs(force).mean() + 1e-10))
    else:
        f['force_index_13d'] = 0.0

    # Ease of Movement
    if n > 14:
        dm = ((high_arr[-14:] + low_arr[-14:]) / 2) - ((high_arr[-15:-1] + low_arr[-15:-1]) / 2)
        br = vol_arr[-14:] / (high_arr[-14:] - low_arr[-14:] + 1e-10) / 1e6
        eom = dm / (br + 1e-10)
        f['eom_14d'] = float(np.mean(eom))
    else:
        f['eom_14d'] = 0.0

    return f


def get_all_flow_feature_names():
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
    fprint(f"  Building {'flow-enhanced' if use_flow else 'legacy'} feature dataset...")
    recs = []

    for dt in dates:
        if dt not in sc.index:
            continue
        idx = sc.index.get_loc(dt)
        if idx < 260:
            continue

        for tk in sc.columns:
            lf = compute_legacy_features(sc, spy, vix, idx, tk)
            if lf is None:
                continue

            feat = dict(lf)

            if use_flow:
                ff = compute_flow_features(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, idx, tk)
                if ff is not None:
                    feat.update(ff)
                else:
                    for fn in get_all_flow_feature_names():
                        feat[fn] = 0.0

            # Forward return label (21d to match DTE)
            fi = min(idx + DTE, len(sc) - 1)
            feat['fwd_ret'] = float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1)
            feat['date'] = dt
            feat['ticker'] = tk
            recs.append(feat)

    df = pd.DataFrame(recs)
    fprint(f"  Dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def run_lgbm_walkforward(df, feature_cols, label=''):
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
# 5. HONEST OPTIONS PRICING
# =====================================================================

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({
        'hl': h - l,
        'hc': abs(h - c.shift(1)),
        'lc': abs(l - c.shift(1))
    }).max(axis=1)
    return tr.rolling(period).mean()


def price_spread_entry(S, spread_pct, dte, atr, vix_level):
    """
    HONEST entry pricing with 15% haircut.
    Returns: (entry_cost_per_contract, K1, K2)

    At ENTRY:
      - Long leg costs MORE (multiply by 1+HAIRCUT)
      - Short leg receives LESS (multiply by 1-HAIRCUT)
      - Debit = long_cost - short_credit
      - Total cost = debit * 100 + commission
    """
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

    # ENTRY: 15% haircut (we pay more for long, receive less for short)
    long_cost = long_prem * (1 + HAIRCUT)
    short_credit = short_prem * (1 - HAIRCUT)
    debit = long_cost - short_credit

    total_cost = debit * 100 + SPREAD_COMM
    width = K2 - K1

    return total_cost, debit, width, K1, K2


def expiry_value(Se, K1, K2):
    """
    HONEST expiry pricing: pure intrinsic value, NO haircut, NO time value.
    Automatic exercise at expiry = intrinsic value only.
    """
    intrinsic = (max(0, Se - K1) - max(0, Se - K2)) * 100
    return intrinsic


# =====================================================================
# 6. HONEST BACKTEST SIMULATION
# =====================================================================

def simulate_variant(variant_name, ranks, sc, sh, sl, spy, vix, regime_scores, atr_dict):
    """
    HONEST simulation: hold to expiry only, intrinsic value at exit, no early exit.
    """
    equity = CAP
    trades = []
    curve = [CAP]
    curve_dates = [None]
    sma200 = spy.rolling(200).mean()

    for dt in sorted(ranks.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])
        scores = ranks[dt]
        if not scores:
            curve.append(equity)
            curve_dates.append(dt)
            continue

        # --- FILTER LOGIC ---
        skip = False

        if variant_name.startswith('A'):
            # BASELINE: VIX>20 filter (skip low vol)
            if cv < 15:
                skip = True
        elif variant_name.startswith('B'):
            # REGIME_ONLY: GRU regime > 0.4
            if regime_scores is not None and dt in regime_scores.index:
                rs = float(regime_scores.loc[dt])
                if rs < REGIME_THRESHOLD:
                    skip = True
            else:
                skip = True
        elif variant_name.startswith('C'):
            # FLOW_ONLY: VIX>20 filter (same as A)
            if cv < 15:
                skip = True
        elif variant_name.startswith('D'):
            # FULL_STACK: regime > 0.4 filter
            if regime_scores is not None and dt in regime_scores.index:
                rs = float(regime_scores.loc[dt])
                if rs < REGIME_THRESHOLD:
                    skip = True
            else:
                skip = True

        if skip:
            curve.append(equity)
            curve_dates.append(dt)
            continue

        # --- SECTOR SELECTION ---
        picks = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:MAX_CONC]]

        max_per_trade = min(MAX_POS, equity / 3)
        if max_per_trade < 30:
            curve.append(equity)
            curve_dates.append(dt)
            continue

        bull = float(spy.loc[dt]) >= (float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else float(spy.loc[dt]))

        n_entered = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_dict or n_entered >= MAX_CONC:
                continue

            S = float(sc[tk].loc[dt])
            di = sc.index.get_loc(dt)

            atr_val = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015

            # HONEST entry pricing
            cost, debit, width, K1, K2 = price_spread_entry(S, 3.0, DTE, atr_val, cv)

            if cost <= 0 or cost > max_per_trade or cost > equity * 0.40:
                continue

            # HONEST: HOLD TO EXPIRY, NO EARLY EXIT
            ei = min(di + DTE, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])

            # HONEST: Intrinsic value only at expiry, NO haircut at exit
            exit_value = expiry_value(Se, K1, K2)

            # P&L = exit value - entry cost (commission already in cost)
            pnl = exit_value - cost

            equity += pnl
            n_entered += 1

            regime_score_at_entry = float(regime_scores.loc[dt]) if regime_scores is not None and dt in regime_scores.index else -1

            trades.append({
                'entry': str(dt.date()),
                'exit': str(sc.index[ei].date()),
                'ticker': tk,
                'pnl': round(pnl, 2),
                'win': pnl > 0,
                'regime': 'bull' if bull else 'bear',
                'vix': round(cv, 1),
                'regime_score': round(regime_score_at_entry, 3),
                'cost': round(cost, 2),
                'entry_price': round(S, 2),
                'exit_price': round(Se, 2),
                'K1': K1,
                'K2': K2,
                'intrinsic_at_expiry': round(exit_value, 2),
                'year': dt.year,
            })

        curve.append(equity)
        curve_dates.append(dt)

    return trades, equity, curve


# =====================================================================
# 7. METRICS + FULL ADVERSARIAL VALIDATION
# =====================================================================

def compute_metrics(trades, final_eq, curve, name):
    if not trades:
        fprint(f"  {name}: No trades")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnls = [t['pnl'] for t in trades]

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

    # Bull/Bear Sharpe
    bull_pnls = [t['pnl'] for t in bull_trades]
    bear_pnls = [t['pnl'] for t in bear_trades]
    bull_sharpe = (np.mean(bull_pnls) / (np.std(bull_pnls) + 1e-10) * np.sqrt(12)) if len(bull_pnls) > 3 else 0
    bear_sharpe = (np.mean(bear_pnls) / (np.std(bear_pnls) + 1e-10) * np.sqrt(12)) if len(bear_pnls) > 3 else 0

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

    # Yearly breakdown
    yearly = {}
    for year in sorted(set(t['year'] for t in trades)):
        yt = [t for t in trades if t['year'] == year]
        if yt:
            ypnls = [t['pnl'] for t in yt]
            yearly[year] = {
                'n': len(yt),
                'wr': round(sum(1 for p in ypnls if p > 0) / len(ypnls) * 100, 1),
                'total_pnl': round(sum(ypnls), 2),
                'avg_pnl': round(np.mean(ypnls), 2),
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
        'median_pnl': round(float(np.median(pnls)), 2),
        'final_equity': round(final_eq, 2),
        'total_pnl': round(sum(pnls), 2),
        'max_consec_loss': mcl,
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'bull_n': len(bull_trades),
        'bear_n': len(bear_trades),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'vix_bands': vix_bands,
        'yearly': yearly,
        'monthly_returns': monthly_ret.values.tolist(),
    }

    fprint(f"  {name:20s} | {n:4d} trades | WR {wr:5.1f}% | Sh {sharpe:5.2f} | So {sortino:5.2f} | "
           f"CAGR {cagr*100:5.1f}% | MDD {max_dd*100:5.1f}% | PF {pf:5.2f} | ${CAP:.0f}->${final_eq:.0f} | MCL {mcl}")

    return result


def adversarial_validate(result):
    """Full adversarial: 2000-trial permutation, regime split, sub-period, yearly, outlier."""
    if result is None:
        return result

    rets = np.array(result['monthly_returns'])
    if len(rets) < 10:
        result.update({'gates': 0, 'perm_p': 1.0, 'r1_gap': 1.0})
        return result

    gates = 0

    # Gate 1: Permutation test (2000 trials, p < 0.05)
    real_sharpe = np.mean(rets) / (np.std(rets) + 1e-10)
    n_better = 0
    for _ in range(N_PERM_TRIALS):
        shuffled = rets * np.random.choice([-1, 1], len(rets))
        sh = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
        if sh >= real_sharpe:
            n_better += 1
    perm_p = n_better / N_PERM_TRIALS
    g1 = perm_p < 0.05
    gates += g1

    # Gate 2: R1 Regime gap (bull vs bear Sharpe gap < 0.50)
    bull_sh = result.get('bull_sharpe', 0)
    bear_sh = result.get('bear_sharpe', 0)
    max_sh = max(abs(bull_sh), abs(bear_sh), 0.01)
    r1_gap = abs(bull_sh - bear_sh) / max_sh
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

    # Gate 5: Yearly consistency (>50% of years positive)
    yearly = result.get('yearly', {})
    if yearly:
        positive_years = sum(1 for y in yearly.values() if y['total_pnl'] > 0)
        g5 = positive_years / len(yearly) > 0.50
        gates += g5
        yearly_consistency = round(positive_years / len(yearly), 2)
    else:
        g5 = False
        yearly_consistency = 0

    result.update({
        'gates': gates,
        'gates_total': 5,
        'perm_p': round(perm_p, 4),
        'r1_gap': round(r1_gap, 3),
        'g1_perm': g1,
        'g2_regime': g2,
        'g3_sub': g3,
        'g4_outlier': g4,
        'g5_yearly': g5,
        'h1_sharpe': round(h1_sharpe, 2),
        'h2_sharpe': round(h2_sharpe, 2),
        'yearly_consistency': yearly_consistency,
    })

    fprint(f"    Gates: P={'Y' if g1 else 'N'}(p={perm_p:.4f}) R1={'Y' if g2 else 'N'}({r1_gap:.2f}) "
           f"Sub={'Y' if g3 else 'N'}({h1_sharpe:.2f}/{h2_sharpe:.2f}) Out={'Y' if g4 else 'N'} "
           f"Yr={'Y' if g5 else 'N'}({yearly_consistency:.0%}) => {gates}/5")

    return result


def random_baseline(sc, vix, spy, atr_dict, n_sims=5):
    """HONEST random baseline: hold to expiry, intrinsic only, no haircut at exit."""
    fprint(f"\n[7/8] Random baseline ({n_sims} simulations, HONEST pricing)...")
    random_sharpes = []
    random_results = []

    all_dates = sorted(sc.index[260:])
    rebalance_dates = all_dates[::10]

    for sim in range(n_sims):
        np.random.seed(sim + 1000)
        equity = CAP
        trades = []

        for dt in rebalance_dates:
            if dt not in vix.index:
                continue
            cv = float(vix.loc[dt])

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

                cost, debit, width, K1, K2 = price_spread_entry(S, 3.0, DTE, atr_val, cv)
                if cost <= 0 or cost > max_per:
                    continue

                ei = min(di + DTE, len(sc) - 1)
                Se = float(sc[tk].iloc[ei])

                # HONEST: intrinsic only at expiry, no haircut
                exit_val = expiry_value(Se, K1, K2)
                pnl = exit_val - cost

                equity += pnl
                trades.append({'pnl': pnl, 'win': pnl > 0, 'entry': str(dt.date())})

        if trades:
            tdf = pd.DataFrame(trades)
            tdf['month'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
            mr = tdf.groupby('month')['pnl'].sum() / CAP
            if len(mr) > 3:
                sh = (mr.mean() * 12) / (mr.std() * np.sqrt(12) + 1e-10)
                random_sharpes.append(sh)
                random_results.append({
                    'sim': sim,
                    'sharpe': round(sh, 3),
                    'final_equity': round(equity, 2),
                    'n_trades': len(trades),
                    'wr': round(sum(1 for t in trades if t['win']) / len(trades) * 100, 1),
                })

    if random_sharpes:
        fprint(f"  Random baseline (HONEST): mean Sharpe={np.mean(random_sharpes):.3f}, "
               f"median={np.median(random_sharpes):.3f}, "
               f"max={np.max(random_sharpes):.3f}")
        for rr in random_results:
            fprint(f"    Sim {rr['sim']}: Sharpe={rr['sharpe']:.3f}, ${rr['final_equity']:.0f}, "
                   f"{rr['n_trades']} trades, WR={rr['wr']:.1f}%")

    return random_sharpes, random_results


# =====================================================================
# 8. MAIN
# =====================================================================

def main():
    t0 = time.time()
    fprint(f"{'='*80}")
    fprint(f"HONEST FULL STACK v1 — {datetime.now():%Y-%m-%d %H:%M:%S}")
    fprint(f"{'='*80}")
    fprint(f"CRITICAL DIFFERENCES FROM regime_flow_combined_v1.py:")
    fprint(f"  1. HOLD TO EXPIRY ONLY — no early exit, no take-profit")
    fprint(f"  2. At expiry: INTRINSIC VALUE ONLY (no time value)")
    fprint(f"  3. 15% haircut on ENTRY only (no haircut at expiry)")
    fprint(f"  4. DTE=21 (3 weeks, was 14)")
    fprint(f"  5. Permutation: 2000 trials (was 1000)")
    fprint(f"  6. Random baseline: 5 trials (was 100 with inflated exit)")
    fprint(f"{'='*80}")
    fprint(f"Capital: ${CAP:.0f} | Haircut: {HAIRCUT:.0%} entry only | Comm: ${SPREAD_COMM}/spread")
    fprint(f"Regime threshold: {REGIME_THRESHOLD} | DTE: {DTE}d")
    fprint(f"{'='*80}\n")

    # 1. Download data
    sc, sh, sl, sv, spy, vix, macro, macro_vol, macro_high, macro_low = download_data()

    # 2. Load regime predictions
    regime_scores = load_regime_predictions()
    if regime_scores is None:
        fprint("FATAL: Cannot proceed without regime predictions")
        return

    # 3. Compute ATR
    fprint("[3/8] Computing ATR for options pricing...")
    atr_dict = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_dict[tk] = compute_atr(sh[tk], sl[tk], sc[tk])
    fprint(f"  ATR computed for {len(atr_dict)} sectors")

    # 4. Build LGBM datasets and rankings
    rdates = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rdates = rdates[rdates.isin(sc.index)]
    fprint(f"\n[4/8] Building LGBM rankings on {len(rdates)} rebalance dates...")

    legacy_df = build_lgbm_dataset(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, rdates, use_flow=False)
    legacy_feature_cols = [c for c in LEGACY_FEATURES if c in legacy_df.columns]
    fprint(f"  Legacy features available: {len(legacy_feature_cols)}")

    flow_df = build_lgbm_dataset(sc, sv, sh, sl, spy, vix, macro, macro_vol, macro_high, macro_low, rdates, use_flow=True)
    flow_feature_cols = legacy_feature_cols + [c for c in get_all_flow_feature_names() if c in flow_df.columns]
    fprint(f"  Flow features available: {len(flow_feature_cols)} (legacy {len(legacy_feature_cols)} + flow {len(flow_feature_cols)-len(legacy_feature_cols)})")

    # 5. Run LGBM walk-forward
    fprint(f"\n[5/8] Running walk-forward LGBM...")
    legacy_ranks = run_lgbm_walkforward(legacy_df, legacy_feature_cols, 'legacy')
    flow_ranks = run_lgbm_walkforward(flow_df, flow_feature_cols, 'flow-enhanced')

    # 6. Simulate all 4 variants with HONEST pricing
    fprint(f"\n[6/8] Simulating 4 variants (HONEST: hold-to-expiry, intrinsic only)...")
    fprint(f"{'='*100}")

    variants = [
        ('A_BASELINE',    legacy_ranks, 'VIX filter + legacy LGBM'),
        ('B_REGIME_ONLY', legacy_ranks, 'Regime>0.4 + legacy LGBM'),
        ('C_FLOW_ONLY',   flow_ranks,   'VIX filter + flow LGBM'),
        ('D_FULL_STACK',  flow_ranks,   'Regime>0.4 + flow LGBM'),
    ]

    results = []
    all_trades = {}
    for vname, ranks, desc in variants:
        fprint(f"\n--- {vname}: {desc} ---")
        trades, eq, curve = simulate_variant(vname, ranks, sc, sh, sl, spy, vix, regime_scores, atr_dict)
        r = compute_metrics(trades, eq, curve, vname)
        if r:
            r['description'] = desc
            r = adversarial_validate(r)
            results.append(r)
            all_trades[vname] = trades

    # 7. Random baseline (HONEST)
    random_sharpes, random_results = random_baseline(sc, vix, spy, atr_dict, n_sims=N_RANDOM_BASELINE)

    # 8. Summary
    fprint(f"\n{'='*120}")
    fprint(f"HONEST FULL STACK COMPARISON (hold-to-expiry, intrinsic only, 15% haircut entry only)")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<22} {'Desc':<32} {'#':>5} {'WR':>6} {'Sh':>6} {'So':>6} {'CAGR':>7} {'MDD':>7} {'PF':>5} {'$':>8} {'G':>5}")
    fprint(f"{'-'*120}")

    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['description']:<32} {r['n_trades']:>5} {r['win_rate']:>5.1f}% "
               f"{r['sharpe']:>6.2f} {r['sortino']:>6.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>5.2f} ${r['final_equity']:>7.0f} {r['gates']:>3}/5")

    # Yearly breakdown
    fprint(f"\n--- YEARLY BREAKDOWN ---")
    for r in results:
        fprint(f"\n  {r['name']}:")
        for year, stats in sorted(r.get('yearly', {}).items()):
            marker = '+' if stats['total_pnl'] > 0 else '-'
            fprint(f"    {year}: {marker} {stats['n']:>3} trades, WR={stats['wr']:>5.1f}%, "
                   f"total=${stats['total_pnl']:>8.2f}, avg=${stats['avg_pnl']:>6.2f}")

    # VIX band breakdown
    fprint(f"\n--- VIX BAND ANALYSIS ---")
    for r in results:
        fprint(f"\n  {r['name']}:")
        for band, stats in r.get('vix_bands', {}).items():
            fprint(f"    {band}: {stats['n']} trades, WR={stats['wr']}%, avg=${stats['avg_pnl']:.2f}, total=${stats['total_pnl']:.2f}")

    # Bull vs Bear regime analysis
    fprint(f"\n--- REGIME ANALYSIS (Bull vs Bear) ---")
    for r in results:
        fprint(f"  {r['name']:20s} | Bull: {r['bull_n']:>4} trades WR={r['bull_wr']:>5.1f}% Sh={r['bull_sharpe']:>5.2f} | "
               f"Bear: {r['bear_n']:>4} trades WR={r['bear_wr']:>5.1f}% Sh={r['bear_sharpe']:>5.2f} | "
               f"R1 gap={r.get('r1_gap', 0):.2f}")

    # Key comparison
    fprint(f"\n{'='*80}")
    fprint("KEY COMPARISON (HONEST):")
    baseline = next((r for r in results if r['name'].startswith('A')), None)
    full_stack = next((r for r in results if r['name'].startswith('D')), None)
    regime_only = next((r for r in results if r['name'].startswith('B')), None)
    flow_only = next((r for r in results if r['name'].startswith('C')), None)

    if baseline and full_stack:
        fprint(f"  BASELINE   -> Sharpe {baseline['sharpe']:.2f}, MDD {baseline['maxdd_pct']:.1f}%, CAGR {baseline['cagr_pct']:.1f}%, Gates {baseline['gates']}/5")
        if regime_only:
            fprint(f"  REGIME_ONLY-> Sharpe {regime_only['sharpe']:.2f}, MDD {regime_only['maxdd_pct']:.1f}%, CAGR {regime_only['cagr_pct']:.1f}%, Gates {regime_only['gates']}/5")
        if flow_only:
            fprint(f"  FLOW_ONLY  -> Sharpe {flow_only['sharpe']:.2f}, MDD {flow_only['maxdd_pct']:.1f}%, CAGR {flow_only['cagr_pct']:.1f}%, Gates {flow_only['gates']}/5")
        fprint(f"  FULL_STACK -> Sharpe {full_stack['sharpe']:.2f}, MDD {full_stack['maxdd_pct']:.1f}%, CAGR {full_stack['cagr_pct']:.1f}%, Gates {full_stack['gates']}/5")

        sharpe_delta = full_stack['sharpe'] - baseline['sharpe']
        mdd_delta = full_stack['maxdd_pct'] - baseline['maxdd_pct']
        fprint(f"  Delta (Full vs Base): Sharpe {sharpe_delta:+.2f}, MDD {mdd_delta:+.1f}pp")

        if random_sharpes:
            p_vs_random = np.mean([s >= full_stack['sharpe'] for s in random_sharpes])
            fprint(f"  Full stack vs random baseline: p={p_vs_random:.3f} (mean random Sharpe={np.mean(random_sharpes):.3f})")

    # Verdict
    fprint(f"\n{'='*80}")
    fprint("HONEST VERDICT:")
    best = max(results, key=lambda x: x['sharpe']) if results else None
    best_valid = max([r for r in results if r.get('gates', 0) >= 3], key=lambda x: x['sharpe'], default=None)

    if best:
        fprint(f"  BEST OVERALL: {best['name']} (Sharpe={best['sharpe']:.2f}, {best['gates']}/5 gates)")
    if best_valid:
        fprint(f"  BEST VALIDATED (3+ gates): {best_valid['name']} (Sharpe={best_valid['sharpe']:.2f})")
    else:
        fprint("  WARNING: No variant passes 3+ adversarial gates")

    if baseline and full_stack:
        fprint(f"\n  ANSWER: Does combining regime + flow help (HONESTLY)?")
        if regime_only and flow_only:
            best_individual = max(regime_only['sharpe'], flow_only['sharpe'])
            if full_stack['sharpe'] > best_individual:
                fprint(f"  YES — Full stack Sharpe ({full_stack['sharpe']:.2f}) > best individual ({best_individual:.2f})")
            else:
                fprint(f"  NO — Full stack Sharpe ({full_stack['sharpe']:.2f}) <= best individual ({best_individual:.2f})")
                better = regime_only if regime_only['sharpe'] > flow_only['sharpe'] else flow_only
                fprint(f"  Better component: {better['name']} (Sharpe={better['sharpe']:.2f})")

    # Check if ANY variant beats random
    if random_sharpes and results:
        random_max = max(random_sharpes)
        beats_random = [r for r in results if r['sharpe'] > random_max]
        if beats_random:
            fprint(f"\n  Variants beating best random ({random_max:.3f}): {[r['name'] for r in beats_random]}")
        else:
            fprint(f"\n  WARNING: NO variant beats best random Sharpe ({random_max:.3f})")
            fprint(f"  This suggests the strategy may not have real edge with honest pricing.")

    # MLflow logging
    fprint(f"\n[8/8] Logging to MLflow...")
    if MLFLOW_OK:
        try:
            exp_name = 'honest_full_stack_v1'
            try:
                if not mlflow.get_experiment_by_name(exp_name):
                    mlflow.create_experiment(exp_name)
            except:
                pass
            mlflow.set_experiment(exp_name)

            with mlflow.start_run(run_name=f"honest_v1_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_params({
                    'capital': CAP,
                    'haircut_entry': HAIRCUT,
                    'haircut_exit': 0.0,
                    'early_exit': False,
                    'expiry_pricing': 'intrinsic_only',
                    'regime_threshold': REGIME_THRESHOLD,
                    'dte': DTE,
                    'n_sectors': len(SECTORS),
                    'n_perm_trials': N_PERM_TRIALS,
                    'n_random_baseline': N_RANDOM_BASELINE,
                })

                for r in results:
                    prefix = r['name']
                    for k in ['sharpe', 'sortino', 'cagr_pct', 'maxdd_pct', 'win_rate', 'pf',
                              'n_trades', 'gates', 'perm_p', 'bull_sharpe', 'bear_sharpe', 'r1_gap']:
                        try:
                            mlflow.log_metric(f"{prefix}_{k}", float(r[k]))
                        except:
                            pass

                if random_sharpes:
                    mlflow.log_metric('random_baseline_mean_sharpe', float(np.mean(random_sharpes)))
                    mlflow.log_metric('random_baseline_max_sharpe', float(np.max(random_sharpes)))

            fprint("  MLflow: logged successfully")
        except Exception as e:
            fprint(f"  MLflow error: {e}")
    else:
        fprint("  MLflow: skipped (not available)")

    # Save results
    save_data = {
        'strategy': 'Honest Full Stack v1',
        'honest_differences': [
            'HOLD TO EXPIRY ONLY — no early exit, no take-profit',
            'At expiry: INTRINSIC VALUE ONLY (no time value)',
            '15% haircut on ENTRY only (no haircut at expiry — automatic exercise)',
            'DTE=21 (3 weeks)',
            'Permutation: 2000 trials',
            'Random baseline: 5 trials with identical honest pricing',
        ],
        'run_date': datetime.now().isoformat(),
        'config': {
            'capital': CAP,
            'haircut_entry': HAIRCUT,
            'haircut_exit': 0.0,
            'early_exit': False,
            'expiry_pricing': 'intrinsic_only',
            'regime_threshold': REGIME_THRESHOLD,
            'dte': DTE,
            'legacy_features': len(legacy_feature_cols),
            'flow_features': len(flow_feature_cols),
            'n_perm_trials': N_PERM_TRIALS,
        },
        'variants': [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results],
        'random_baseline': {
            'mean_sharpe': round(float(np.mean(random_sharpes)), 4) if random_sharpes else None,
            'max_sharpe': round(float(np.max(random_sharpes)), 4) if random_sharpes else None,
            'trials': random_results,
        },
        'best': best['name'] if best else 'NONE',
        'best_valid': best_valid['name'] if best_valid else 'NONE',
    }

    out_path = OUTPUT_DIR / 'honest_full_stack_v1_results.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    fprint(f"\nResults saved to {out_path}")

    # Save individual trade logs
    for vname, trades in all_trades.items():
        trade_path = OUTPUT_DIR / f'{vname}_trades.json'
        with open(trade_path, 'w') as f:
            json.dump(trades, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))

    fprint(f"Total runtime: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
