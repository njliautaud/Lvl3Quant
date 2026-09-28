#!/usr/bin/env python3
"""
V8 Liquidity-Adjusted v1 — Sector Universe Filtering by Options Liquidity
==========================================================================

Tests whether filtering sectors by options liquidity improves V8 real-world
performance. Liquidity analysis showed only XLF (87.3), XLU (78.2), XLE (75.5)
are truly "tradable" for options spreads. XLK and XLY rank bottom 2 despite
being large ETFs.

4 Variants (all V8 config: DTE=14, 2% OTM, weekly, pairs VIX<20, 17 features, LGBM 100 trees):
  A: Full 11 sectors (baseline, should match known Sharpe ~3.23)
  B: Top 6 liquid sectors (XLF, XLU, XLE, XLI, XLP, XLV — tradability > 50)
  C: Top 3 liquid only (XLF, XLU, XLE — tradability > 75)
  D: Full 11 but liquidity-weighted position sizing (more capital to liquid sectors)

Honest pricing rules (all variants):
  - HOLD TO EXPIRY ONLY (no early exit)
  - At expiry: INTRINSIC VALUE ONLY (no time value)
  - 15% haircut on ENTRY only (automatic exercise at expiry = no exit haircut)
  - $645 starting capital
  - Walk-forward LGBM, regime filter via GRU (>0.4)
  - Commission: $2.60/spread

Full 5-gate adversarial validation + random baseline comparison (5 trials).
Regime stratification, yearly breakdown, PnL by side for each variant.
MLflow experiment: 'v8_liquidity_adjusted_v1'
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


# ── Standardized tools ──
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "v8_liquidity_adjusted_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Full sector universe (used for features/data download regardless of variant)
ALL_SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
SPREAD_PCT = 3.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Liquidity tradability scores (from liquidity analysis)
LIQUIDITY_SCORES = {
    "XLF": 1.0,
    "XLU": 0.9,
    "XLE": 0.87,
    "XLI": 0.7,
    "XLP": 0.65,
    "XLV": 0.6,
    "XLB": 0.5,
    "XLC": 0.45,
    "XLRE": 0.4,
    "XLK": 0.3,
    "XLY": 0.25,
}

# Sector universes per variant
SECTORS_TOP6 = ["XLF", "XLU", "XLE", "XLI", "XLP", "XLV"]
SECTORS_TOP3 = ["XLF", "XLU", "XLE"]

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v8_liquidity_adjusted_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable -- results saved to disk only")


# ══════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════

VARIANTS = {
    "A_full_11_baseline": {
        "desc": "V8 Full 11 sectors (baseline)",
        "sectors": ALL_SECTORS,
        "rebal_freq": "W-FRI",
        "otm_pct": 0.02,
        "pairs": True,
        "dte": 14,
        "feature_set": "v6",
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
        "liquidity_weighted": False,
    },
    "B_top6_liquid": {
        "desc": "V8 Top 6 liquid (tradability > 50)",
        "sectors": SECTORS_TOP6,
        "rebal_freq": "W-FRI",
        "otm_pct": 0.02,
        "pairs": True,
        "dte": 14,
        "feature_set": "v6",
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
        "liquidity_weighted": False,
    },
    "C_top3_liquid": {
        "desc": "V8 Top 3 liquid (tradability > 75)",
        "sectors": SECTORS_TOP3,
        "rebal_freq": "W-FRI",
        "otm_pct": 0.02,
        "pairs": True,
        "dte": 14,
        "feature_set": "v6",
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
        "liquidity_weighted": False,
    },
    "D_liquidity_weighted": {
        "desc": "V8 Full 11 + liquidity-weighted sizing",
        "sectors": ALL_SECTORS,
        "rebal_freq": "W-FRI",
        "otm_pct": 0.02,
        "pairs": True,
        "dte": 14,
        "feature_set": "v6",
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
        "liquidity_weighted": True,
    },
}


# ══════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS (V6: 17 features)
# ══════════════════════════════════════════════════════════════

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d",
    "cross_sector_dispersion",
]

assert len(V6_FEATURES) == 17, f"Expected 17 V6 features, got {len(V6_FEATURES)}"

FEATURE_SETS = {"v6": V6_FEATURES}


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf

    all_tickers = ALL_SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    needed = ["SPY", "VIX"]
    for t in needed:
        if t not in close.columns:
            raise ValueError(f"Missing critical ticker: {t}")

    fprint(f"Data: {len(close)} days, columns: {list(close.columns)}")
    fprint(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    return close, high, low


# ══════════════════════════════════════════════════════════════
# REGIME LOADING
# ══════════════════════════════════════════════════════════════

def load_regime_predictions():
    """Load GRU regime predictions and build a date-indexed Series."""
    if not REGIME_FILE.exists():
        fprint(f"WARNING: Regime file not found at {REGIME_FILE}")
        fprint("  Will use VIX-based regime proxy instead")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    fprint(f"Regime predictions loaded: {len(regime_series)} days "
           f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    fprint(f"  Mean score: {regime_series.mean():.3f}, "
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}")
    return regime_series


def get_regime_score_at(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback."""
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def compute_legacy_features(px, spy_slice):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, "ret_5d"), (10, "ret_10d"), (21, "ret_21d"),
                   (63, "ret_63d"), (126, "ret_126d"), (252, "ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()
    f["vol_21d"] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f["vol_63d"] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f["sharpe_63d"] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f["maxdd_63d"] = float(((px.iloc[-63:] / pk63) - 1).min())
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3

    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f["sortino_63d"] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2
        f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0
        f["trend_slope_63d"] = 0.0

    return f


def compute_cross_asset_features(sector_ticker, dt_idx, close_df):
    """Compute all 3 cross-asset features."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "sector_relative_vol_21d": 1.0,
                "cross_sector_dispersion": 0.01}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f["sector_spy_beta_63d"] = float(beta)
        else:
            f["sector_spy_beta_63d"] = 1.0
    else:
        f["sector_spy_beta_63d"] = 1.0

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f["sector_relative_vol_21d"] = float(sec_vol / (spy_vol + 1e-10))
        else:
            f["sector_relative_vol_21d"] = 1.0
    else:
        f["sector_relative_vol_21d"] = 1.0

    # 3. Cross-sector dispersion (rolling 21d stdev of sector returns)
    sector_cols = [c for c in ALL_SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:dt_idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f["cross_sector_dispersion"] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f["cross_sector_dispersion"] = 0.01
    else:
        f["cross_sector_dispersion"] = 0.01

    return f


# ══════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series, dte, sector_universe):
    """
    Build feature + target records for sectors in the given universe on all rebal dates.
    Uses regime>0.4 filter. Forward return target uses the variant-specific DTE.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features, "
           f"DTE={dte}, sectors={len(sector_universe)}")

    records = []
    sector_cols_avail = [c for c in sector_universe if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Regime filter: only trade when GRU says bull (>0.4)
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols_avail:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features
            cross_asset = compute_cross_asset_features(tk, idx, close)

            # Forward return target (dte days forward)
            fi = min(idx + dte, len(close) - 1)
            if fi <= idx:
                continue
            fwd_ret = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)

            rec = {**legacy, **cross_asset, "date": dt, "ticker": tk, "fwd_ret": fwd_ret}
            records.append(rec)

    df = pd.DataFrame(records)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    fprint(f"    {len(df)} records, {len(df['date'].unique())} dates")
    return df


def walk_forward_lgbm_rank(df, feature_cols, variant_name):
    """Walk-forward LGBM ranking: 12-period sliding train, predict next period."""
    import lightgbm as lgb

    if len(df) < 100:
        fprint(f"    {variant_name}: Insufficient data ({len(df)} records)")
        return {}, None

    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())

    rankings = {}
    all_importances = np.zeros(len(feature_cols))
    n_models = 0

    for i in range(WF_TRAIN_PERIODS, len(dates)):
        train_dates = dates[max(0, i - WF_TRAIN_PERIODS):i]
        test_date = dates[i]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"] == test_date].copy()

        if len(test_df) < 3 or len(train_df) < 50:
            continue

        Xt = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        yt = train_df["rank_label"].values.astype(np.float32)
        Xe = np.nan_to_num(test_df[feature_cols].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                verbose=-1,
            )
            m.fit(Xt, yt)
            test_df["score"] = m.predict(Xe)

            rankings[test_date] = dict(zip(test_df["ticker"], test_df["score"]))

            all_importances += m.feature_importances_
            n_models += 1
        except Exception:
            continue

    if n_models > 0:
        all_importances /= n_models
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": all_importances,
        }).sort_values("importance", ascending=False)
    else:
        imp_df = None

    fprint(f"    {variant_name}: {len(rankings)} ranking dates, {n_models} models trained")
    return rankings, imp_df


# ══════════════════════════════════════════════════════════════
# ATR COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_atr_series(high, low, close, period=14):
    """Compute ATR series for all sectors."""
    atr_dict = {}
    for tk in ALL_SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna()
            l = low[tk].dropna()
            c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


# ══════════════════════════════════════════════════════════════
# STRIKE COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """Compute strike prices for a spread."""
    if direction == "bull":
        if otm_pct > 0:
            K1 = round(S * (1 + otm_pct), 2)
            K2 = round(K1 * (1 + spread_pct / 100), 2)
        else:
            K1 = round(S, 2)
            K2 = round(S * (1 + spread_pct / 100), 2)
    else:  # bear
        if otm_pct > 0:
            K2 = round(S * (1 - otm_pct), 2)
            K1 = round(K2 * (1 - spread_pct / 100), 2)
        else:
            K1 = round(S * (1 - spread_pct / 100), 2)
            K2 = round(S, 2)

    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, dte, close, atr_dict, cv, equity):
    """
    Execute a single spread trade. Returns PnL or None if trade cannot be entered.
    Uses honest pricing: BS with ATR-based IV, 15% entry haircut,
    hold to expiry, intrinsic value only.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + dte, len(close) - 1)
    if ei <= di:
        return None

    # ATR for pricing
    if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
        av = float(atr_dict[tk].loc[dt])
    else:
        av = S * 0.015

    # Compute strikes
    K1, K2 = compute_strikes(S, direction, otm_pct, SPREAD_PCT)

    try:
        if direction == "bull":
            entry_cost_ps, max_profit_ps = price_bull_call_spread(
                S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
            )
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K1, K2=K2, dte=dte, atr=av, vix=cv
            )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None

    # HOLD TO EXPIRY: compute intrinsic value at expiry
    Se = float(close[tk].iloc[ei])

    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)

    exit_value_ps = intrinsic

    # PnL: exit value - entry cost - commission (no exit haircut at expiry)
    pnl = (exit_value_ps - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return pnl


def simulate_trades(name, rankings, close, high, low, atr_dict, variant_cfg):
    """
    Simulate trades for a given variant configuration.

    HONEST RULES:
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)

    Pair trade rules:
      - VIX < 20: top-3 bull + bottom-3 bear (pairs mode)
      - VIX >= 20: top-3 bull only

    Liquidity-weighted sizing (variant D):
      - Each sector's max_pos is scaled by its liquidity score
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    otm_pct = variant_cfg["otm_pct"]
    use_pairs = variant_cfg["pairs"]
    dte = variant_cfg["dte"]
    max_pos_bull = variant_cfg["max_pos_bull"]
    max_pos_pair_leg = variant_cfg["max_pos_pair_leg"]
    liquidity_weighted = variant_cfg["liquidity_weighted"]

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # Determine trade mode based on VIX and pairs setting
        if use_pairs and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        # Pick sectors
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        # Position sizing
        if trade_mode == "pairs":
            base_max_pos = min(max_pos_pair_leg, equity / 6)
        else:
            base_max_pos = min(max_pos_bull, equity / 3)

        if base_max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            if liquidity_weighted:
                liq_score = LIQUIDITY_SCORES.get(tk, 0.3)
                max_pos = base_max_pos * liq_score
            else:
                max_pos = base_max_pos

            if max_pos < 20:
                continue

            pnl = _execute_single_trade(
                tk, dt, "bull", otm_pct, max_pos, dte, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bull",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                    "liq_score": LIQUIDITY_SCORES.get(tk, 0.3) if liquidity_weighted else None,
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            if liquidity_weighted:
                liq_score = LIQUIDITY_SCORES.get(tk, 0.3)
                max_pos = base_max_pos * liq_score
            else:
                max_pos = base_max_pos

            if max_pos < 20:
                continue

            pnl = _execute_single_trade(
                tk, dt, "bear", otm_pct, max_pos, dte, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + dte, len(close) - 1)
                sv = float(spy.loc[dt])
                se = float(spy.iloc[ei]) if ei < len(spy) else sv
                spy_regime = "bull" if se >= sv else "bear"
                trades.append({
                    "pnl": round(pnl, 2),
                    "entry_date": str(dt.date()),
                    "exit_date": str(close.index[ei].date()),
                    "ticker": tk,
                    "regime": spy_regime,
                    "direction": "bear",
                    "vix": round(cv, 1),
                    "win": pnl > 0,
                    "trade_mode": trade_mode,
                    "liq_score": LIQUIDITY_SCORES.get(tk, 0.3) if liquidity_weighted else None,
                })

    return trades, equity


# ══════════════════════════════════════════════════════════════
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_rebal_dates(close, freq_str):
    """Generate rebalance dates from close index based on frequency string."""
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict, variant_cfg, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict, variant_cfg
        )

        if trades and len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=CAP,
                spy_prices=close["SPY"],
                strategy_name=f"Random_{trial}",
                n_perms=500,
            )
            random_sharpes.append(result.sharpe)
            fprint(f"    Random trial {trial}: Sharpe {result.sharpe:.2f}, "
                   f"${CAP:.0f}->${final_eq:.0f}")
        else:
            random_sharpes.append(0.0)

    return random_sharpes


# ══════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════

def regime_stratification(trades):
    """Stratify results by SPY regime (bull vs bear market at exit)."""
    fprint("\n  REGIME STRATIFICATION:")
    fprint(f"  {'Regime':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>8}")
    fprint(f"  {'-'*55}")

    for regime in ["bull", "bear"]:
        rt = [t for t in trades if t["regime"] == regime]
        if not rt:
            fprint(f"  {regime:<10} {'(none)':>7}")
            continue
        pnls = [t["pnl"] for t in rt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))
        fprint(f"  {regime:<10} {len(rt):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} {sh:>8.2f}")

    # Regime imbalance check (HC #428 R1)
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]
    if bull_trades and bear_trades:
        bull_sh = float(np.mean([t["pnl"] for t in bull_trades]) /
                        (np.std([t["pnl"] for t in bull_trades]) + 1e-10) * np.sqrt(52))
        bear_sh = float(np.mean([t["pnl"] for t in bear_trades]) /
                        (np.std([t["pnl"] for t in bear_trades]) + 1e-10) * np.sqrt(52))
        imbalance = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
        fprint(f"\n  Regime imbalance: |Sharpe_bull - Sharpe_bear| / max = {imbalance:.2f}")
        if imbalance > 0.50:
            fprint(f"  WARNING: Regime imbalance > 0.50 threshold")
        else:
            fprint(f"  OK: Regime-agnostic (imbalance < 0.50)")


def pnl_by_side(trades):
    """Break down PnL by bull vs bear side."""
    fprint("\n  PnL BY SIDE:")
    fprint(f"  {'Side':<10} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'MaxWin':>9} {'MaxLoss':>9}")
    fprint(f"  {'-'*65}")

    for side in ["bull", "bear"]:
        st = [t for t in trades if t["direction"] == side]
        if not st:
            fprint(f"  {side:<10} {'(none)':>7}")
            continue
        pnls = [t["pnl"] for t in st]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        mx = max(pnls)
        mn = min(pnls)
        fprint(f"  {side:<10} {len(st):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} ${mx:>8.2f} ${mn:>8.2f}")

    # Trade mode breakdown
    fprint("\n  TRADE MODE BREAKDOWN:")
    pair_t = [t for t in trades if t.get("trade_mode") == "pairs"]
    bull_only_t = [t for t in trades if t.get("trade_mode") == "bull_only"]
    if pair_t:
        pair_pnl = sum(t["pnl"] for t in pair_t)
        pair_wr = sum(1 for t in pair_t if t["win"]) / len(pair_t) * 100
        fprint(f"    Pairs mode (VIX<20): {len(pair_t)} trades, WR {pair_wr:.1f}%, PnL ${pair_pnl:.0f}")
    if bull_only_t:
        bo_pnl = sum(t["pnl"] for t in bull_only_t)
        bo_wr = sum(1 for t in bull_only_t if t["win"]) / len(bull_only_t) * 100
        fprint(f"    Bull-only (VIX>=20): {len(bull_only_t)} trades, WR {bo_wr:.1f}%, PnL ${bo_pnl:.0f}")


def sector_breakdown(trades, variant_name):
    """Break down PnL by sector for this variant."""
    fprint(f"\n  SECTOR BREAKDOWN ({variant_name}):")
    fprint(f"  {'Sector':<8} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10} {'LiqScore':>9}")
    fprint(f"  {'-'*55}")

    by_sector = {}
    for t in trades:
        by_sector.setdefault(t["ticker"], []).append(t)

    for tk in sorted(by_sector.keys(), key=lambda x: LIQUIDITY_SCORES.get(x, 0), reverse=True):
        st = by_sector[tk]
        pnls = [t["pnl"] for t in st]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        avg = np.mean(pnls)
        tot = sum(pnls)
        liq = LIQUIDITY_SCORES.get(tk, 0.0)
        fprint(f"  {tk:<8} {len(st):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f} {liq:>9.2f}")


def yearly_breakdown(trades):
    """Break down results by calendar year."""
    fprint("\n  YEARLY BREAKDOWN:")
    fprint(f"  {'Year':<6} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Sharpe':>8} {'MaxDD':>8}")
    fprint(f"  {'-'*50}")

    by_year = {}
    for t in trades:
        yr = t["entry_date"][:4]
        by_year.setdefault(yr, []).append(t)

    profitable_years = 0
    total_years = 0
    for yr in sorted(by_year.keys()):
        yt = by_year[yr]
        pnls = [t["pnl"] for t in yt]
        wr = sum(1 for p in pnls if p > 0) / len(pnls)
        tot = sum(pnls)
        sh = float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(52))

        # Drawdown within year
        eq = np.cumsum(pnls) + CAP
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        mdd = float(np.min(dd)) if len(dd) > 0 else 0

        fprint(f"  {yr:<6} {len(yt):>7} {wr:>6.1%} ${tot:>9.0f} {sh:>8.2f} {mdd:>7.1%}")
        total_years += 1
        if tot > 0:
            profitable_years += 1

    fprint(f"\n  Profitable years: {profitable_years}/{total_years} ({profitable_years/max(total_years,1)*100:.0f}%)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V8 LIQUIDITY-ADJUSTED v1 — Sector Universe Filtering — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Capital: ${CAP:.0f} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime filter: GRU score > {REGIME_BULL_THRESHOLD}")
    fprint(f"All variants: V8 config (DTE=14, 2% OTM, weekly, pairs VIX<20, 17 features, LGBM 100 trees)")
    fprint(f"Testing {len(VARIANTS)} variants:")
    fprint()

    for vname, vcfg in VARIANTS.items():
        fprint(f"  {vname}: {vcfg['desc']} ({len(vcfg['sectors'])} sectors)")
    fprint()

    fprint("Liquidity scores:")
    for tk, score in sorted(LIQUIDITY_SCORES.items(), key=lambda x: x[1], reverse=True):
        fprint(f"  {tk}: {score:.2f}")
    fprint()

    # 1. Download data (always full universe for feature computation)
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # 4. Build LGBM rankings for each variant's sector universe
    # Each variant may have a different sector universe, so we need separate rankings
    ranking_cache = {}
    imp_cache = {}

    for vname, vcfg in VARIANTS.items():
        sector_key = tuple(sorted(vcfg["sectors"]))
        cache_key = (vcfg["rebal_freq"], vcfg["feature_set"], vcfg["dte"], sector_key)
        if cache_key in ranking_cache:
            fprint(f"\n  Reusing cached rankings for {vname}")
            continue

        feature_cols = FEATURE_SETS[vcfg["feature_set"]]

        fprint(f"\n{'=' * 80}")
        fprint(f"BUILDING LGBM RANKINGS: {vname}")
        fprint(f"  Sectors: {vcfg['sectors']}")
        fprint(f"  freq={vcfg['rebal_freq']}, features={vcfg['feature_set']}({len(feature_cols)}), DTE={vcfg['dte']}")
        fprint(f"{'=' * 80}")

        rebal_dates = generate_rebal_dates(close, vcfg["rebal_freq"])
        fprint(f"  Rebalance dates: {len(rebal_dates)} "
               f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

        records = build_feature_records(
            close, high, low, rebal_dates, feature_cols, regime_series, vcfg["dte"],
            sector_universe=vcfg["sectors"]
        )
        rankings, imp = walk_forward_lgbm_rank(records, feature_cols, vname)
        ranking_cache[cache_key] = rankings
        imp_cache[cache_key] = imp

    # 5. Simulate all variants
    fprint(f"\n{'=' * 100}")
    fprint("SIMULATING ALL VARIANTS")
    fprint(f"{'=' * 100}")

    all_results = {}
    all_trades = {}
    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 90}")

        sector_key = tuple(sorted(vcfg["sectors"]))
        cache_key = (vcfg["rebal_freq"], vcfg["feature_set"], vcfg["dte"], sector_key)
        rankings = ranking_cache.get(cache_key, {})
        if not rankings:
            fprint(f"  No rankings available, skipping")
            continue

        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict, vcfg
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            continue

        all_trades[vname] = trades
        fprint(f"\n  Total trades: {len(trades)}")
        fprint(f"  Final equity: ${final_eq:,.0f} (from ${CAP:.0f})")
        fprint(f"  Total return: {(final_eq/CAP - 1)*100:.1f}%")

        # 5-gate adversarial validation
        fprint(f"\n  5-GATE ADVERSARIAL VALIDATION:")
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Regime stratification
        regime_stratification(trades)

        # PnL by side (all variants use pairs)
        pnl_by_side(trades)

        # Sector breakdown (key for this experiment)
        sector_breakdown(trades, vname)

        # Yearly breakdown
        yearly_breakdown(trades)

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict, vcfg,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"\n  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        rd = result.to_dict()
        all_results[vname] = {
            "description": vcfg["desc"],
            **rd,
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "otm_pct": vcfg["otm_pct"],
            "pairs": vcfg["pairs"],
            "dte": vcfg["dte"],
            "rebal_freq": vcfg["rebal_freq"],
            "feature_set": vcfg["feature_set"],
            "n_features": len(FEATURE_SETS[vcfg["feature_set"]]),
            "n_sectors": len(vcfg["sectors"]),
            "sectors": vcfg["sectors"],
            "liquidity_weighted": vcfg["liquidity_weighted"],
        }

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 130}")
    fprint("SUMMARY COMPARISON -- ALL 4 VARIANTS")
    fprint(f"{'=' * 130}")
    fprint(f"{'Variant':<25} {'Sectors':>7} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7}")
    fprint("-" * 130)

    for vname in VARIANTS.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} -- NO DATA --")
            continue
        fprint(f"  {vname:<25} {r['n_sectors']:>7} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f}")

    # ── LIQUIDITY IMPACT ANALYSIS ──
    fprint(f"\n{'=' * 100}")
    fprint("LIQUIDITY IMPACT ANALYSIS")
    fprint(f"{'=' * 100}")

    baseline = all_results.get("A_full_11_baseline")
    if baseline:
        baseline_sharpe = baseline["sharpe"]
        fprint(f"  Baseline (A - Full 11): Sharpe {baseline_sharpe:.2f}")
        fprint()
        for vname in ["B_top6_liquid", "C_top3_liquid", "D_liquidity_weighted"]:
            r = all_results.get(vname)
            if not r:
                continue
            delta = r["sharpe"] - baseline_sharpe
            pct = (delta / max(abs(baseline_sharpe), 0.01)) * 100
            arrow = "+" if delta > 0 else ""
            gates_status = "ALL PASS" if r.get("gates_passed", 0) == r.get("gates_total", 5) else f"{r['gates_passed']}/{r['gates_total']} gates"
            fprint(f"  {vname:<25} {arrow}{delta:.2f} Sharpe ({arrow}{pct:.0f}%) | "
                   f"Sortino {r['sortino']:.2f} | WR {r['win_rate']*100:.1f}% | "
                   f"{gates_status} | {r['n_sectors']} sectors, {r['n_trades']} trades")

    # ── SECTOR-LEVEL PnL CORRELATION WITH LIQUIDITY ──
    fprint(f"\n{'=' * 100}")
    fprint("SECTOR PnL vs LIQUIDITY SCORE CORRELATION (Variant A baseline)")
    fprint(f"{'=' * 100}")

    if "A_full_11_baseline" in all_trades:
        a_trades = all_trades["A_full_11_baseline"]
        sector_pnl = {}
        for t in a_trades:
            sector_pnl.setdefault(t["ticker"], []).append(t["pnl"])

        if len(sector_pnl) >= 3:
            liq_scores = []
            avg_pnls = []
            win_rates = []
            fprint(f"  {'Sector':<8} {'LiqScore':>9} {'Trades':>7} {'WR':>7} {'AvgPnL':>9} {'TotalPnL':>10}")
            fprint(f"  {'-'*55}")
            for tk in sorted(sector_pnl.keys(), key=lambda x: LIQUIDITY_SCORES.get(x, 0), reverse=True):
                pnls = sector_pnl[tk]
                liq = LIQUIDITY_SCORES.get(tk, 0)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                avg = np.mean(pnls)
                tot = sum(pnls)
                fprint(f"  {tk:<8} {liq:>9.2f} {len(pnls):>7} {wr:>6.1%} ${avg:>8.2f} ${tot:>9.0f}")
                liq_scores.append(liq)
                avg_pnls.append(avg)
                win_rates.append(wr)

            if len(liq_scores) >= 3:
                corr_pnl, p_pnl = stats.pearsonr(liq_scores, avg_pnls)
                corr_wr, p_wr = stats.pearsonr(liq_scores, win_rates)
                fprint(f"\n  Correlation (Liquidity vs AvgPnL): r={corr_pnl:.3f}, p={p_pnl:.3f}")
                fprint(f"  Correlation (Liquidity vs WinRate): r={corr_wr:.3f}, p={p_wr:.3f}")
                if corr_pnl > 0.3 and p_pnl < 0.1:
                    fprint("  FINDING: Higher liquidity sectors tend to produce better PnL")
                elif corr_pnl < -0.3 and p_pnl < 0.1:
                    fprint("  FINDING: Higher liquidity sectors actually produce WORSE PnL (unexpected)")
                else:
                    fprint("  FINDING: No significant relationship between liquidity and PnL")

    # ── BEST VARIANT ──
    fprint(f"\n{'=' * 100}")
    fprint("BEST VARIANT RECOMMENDATION")
    fprint(f"{'=' * 100}")

    valid_results = {k: v for k, v in all_results.items() if v.get("sharpe", 0) > 0}
    if valid_results:
        gated = {k: v for k, v in valid_results.items()
                 if v.get("gates_passed", 0) == v.get("gates_total", 5)}

        if gated:
            best_name = max(gated.keys(), key=lambda k: gated[k]["sharpe"])
            best = gated[best_name]
            fprint(f"\n  BEST (all gates passed): {best_name}")
            fprint(f"    Sharpe: {best['sharpe']:.2f} | Sortino: {best['sortino']:.2f} | "
                   f"WR: {best['win_rate']*100:.1f}% | PF: {best['profit_factor']:.2f}")
            fprint(f"    MaxDD: {best['max_dd']*100:.1f}% | Final: ${best['final_equity']:,.0f} | "
                   f"Trades: {best['n_trades']} | Sectors: {best['n_sectors']}")
            fprint(f"    vs Random: {best['sharpe']:.2f} vs {best['random_mean_sharpe']:.2f} "
                   f"({best['sharpe']/max(best['random_mean_sharpe'],0.01):.1f}x alpha)")

            if baseline:
                delta = best["sharpe"] - baseline["sharpe"]
                if delta > 0:
                    fprint(f"\n    VERDICT: Liquidity filtering IMPROVES performance by +{delta:.2f} Sharpe")
                elif abs(delta) < 0.1:
                    fprint(f"\n    VERDICT: Liquidity filtering has NEGLIGIBLE impact ({delta:+.2f} Sharpe)")
                else:
                    fprint(f"\n    VERDICT: Liquidity filtering HURTS performance by {delta:.2f} Sharpe")
        else:
            fprint("  WARNING: No variant passes all gates. Needs investigation.")
            best_name = max(valid_results.keys(), key=lambda k: valid_results[k]["sharpe"])
            best = valid_results[best_name]
            fprint(f"  Best by Sharpe (not all gates): {best_name} (Sharpe {best['sharpe']:.2f}, "
                   f"{best['gates_passed']}/{best['gates_total']} gates)")
    else:
        fprint("  ERROR: No valid results produced.")

    # ── KEY FINDING ──
    fprint(f"\n{'=' * 100}")
    fprint("KEY FINDING")
    fprint(f"{'=' * 100}")

    if baseline and len(all_results) >= 3:
        sharpes = {k: v["sharpe"] for k, v in all_results.items()}
        best_k = max(sharpes, key=sharpes.get)
        worst_k = min(sharpes, key=sharpes.get)
        spread = sharpes[best_k] - sharpes[worst_k]

        fprint(f"  Sharpe range across variants: {sharpes[worst_k]:.2f} to {sharpes[best_k]:.2f} (spread: {spread:.2f})")
        fprint(f"  Best: {best_k} ({sharpes[best_k]:.2f})")
        fprint(f"  Worst: {worst_k} ({sharpes[worst_k]:.2f})")

        # Concentrated vs diversified
        b_sh = all_results.get("B_top6_liquid", {}).get("sharpe", 0)
        c_sh = all_results.get("C_top3_liquid", {}).get("sharpe", 0)
        a_sh = baseline["sharpe"]

        if c_sh > a_sh and b_sh > a_sh:
            fprint("\n  CONCLUSION: Concentrating in liquid sectors IMPROVES risk-adjusted returns.")
            fprint("  The illiquid sectors (XLK, XLY, XLRE, XLC, XLB) are dragging down performance.")
        elif c_sh < a_sh and b_sh < a_sh:
            fprint("\n  CONCLUSION: Removing sectors HURTS performance. Diversification matters more than liquidity.")
            fprint("  Even illiquid sectors contribute positive edge through the LGBM ranking.")
        else:
            fprint("\n  CONCLUSION: Mixed results. Moderate filtering (Top 6) may help but extreme concentration hurts.")
            fprint("  Liquidity-weighted sizing (Variant D) may be the best compromise.")

    # ── Feature importance ──
    fprint(f"\n{'=' * 80}")
    fprint("FEATURE IMPORTANCE (Top 10 per model)")
    fprint(f"{'=' * 80}")
    for cache_key, imp_df in imp_cache.items():
        if imp_df is not None:
            freq, fset, dte, sectors = cache_key
            fprint(f"\n  Model: {len(sectors)} sectors, freq={freq}, DTE={dte}")
            for _, row in imp_df.head(10).iterrows():
                bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
                fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # ── Save results ──
    results_data = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "capital": CAP,
            "spread_pct": SPREAD_PCT,
            "haircut": DEFAULT_HAIRCUT,
            "commission": COMMISSION_RT_SPREAD,
            "regime_threshold": REGIME_BULL_THRESHOLD,
            "hold_to_expiry": True,
            "entry_haircut_only": True,
            "lgbm_n_estimators": 100,
            "lgbm_max_depth": 4,
            "lgbm_lr": 0.05,
            "wf_train_periods": WF_TRAIN_PERIODS,
        },
        "liquidity_scores": LIQUIDITY_SCORES,
        "variants": {},
    }

    for vname, r in all_results.items():
        results_data["variants"][vname] = r

    results_path = OUTPUT_DIR / "v8_liquidity_adjusted_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save individual trade logs
    for vname, trades in all_trades.items():
        tpath = OUTPUT_DIR / f"{vname}_trades.json"
        with open(tpath, "w") as f:
            json.dump(trades, f, indent=2, default=str)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v8_liq_adj_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                for vname, r in all_results.items():
                    prefix = vname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))
                    mlflow.log_metric(f"{prefix}_n_sectors", r.get("n_sectors", 11))

                mlflow.log_params({
                    "capital": CAP,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "commission": COMMISSION_RT_SPREAD,
                    "regime_threshold": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "n_variants": len(VARIANTS),
                    "variants": ", ".join(VARIANTS.keys()),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "experiment_type": "liquidity_filtering",
                })

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
