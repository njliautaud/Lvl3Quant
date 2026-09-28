#!/usr/bin/env python3
"""
Production v6 Candidate v1 — Combined Cross-Validated Improvements Test
=========================================================================

Tests 8 variants combining ALL cross-validated improvements from extensive testing:

  1. Weekly rebalance — +19% Sharpe (2.21 vs 1.86 biweekly). MLflow exp 179.
  2. 2% OTM moneyness — From KB, 2% OTM best for bull spreads.
  3. Pair trades (VIX<20) — Bull+pairs combined: Sharpe 1.69, 5/5 gates, MDD -20%. MLflow exp 177.
  4. 3-day rebalancing — Ties weekly Sharpe (2.21) but 2x more returns. MLflow exp 179.

8 Variants:
  A: Production v4 baseline — Biweekly, ATM, bull-only, $200/trade. Must reproduce ~1.86 Sharpe.
  B: Weekly only — Weekly rebalance, ATM, bull-only, $200/trade.
  C: Weekly + 2% OTM — Weekly rebalance, 2% OTM calls, bull-only, $200/trade.
  D: Weekly + pairs — Weekly rebalance, ATM, bull VIX>20 + pairs VIX<20, $200/trade ($100/leg).
  E: Weekly + OTM + pairs — Weekly, 2% OTM, bull VIX>20 + pairs VIX<20, $200/trade ($100/leg).
  F: 3-day + OTM + pairs — 3-day rebalance, 2% OTM, bull VIX>20 + pairs VIX<20, $200/trade ($100/leg).
  G: 3-day + OTM — 3-day rebalance, 2% OTM, bull-only, $200/trade.
  H: Kitchen sink — 3-day, 2% OTM, bull VIX>20 + pairs VIX<20, $200/leg for pairs too.

OTM Strikes:
  - Bull call: buy call at S*1.02 (2% OTM), sell call at S*1.02*1.03 (3% width above)
  - Bear put (pairs): buy put at S*0.98 (2% OTM), sell put at S*0.98*0.97 (3% width below)

Pair Trade Rules:
  - VIX >= 20: top-3 bull call spreads only (same as production)
  - VIX < 20: top-3 bull + bottom-3 bear put spreads
  - Pairs: $100/trade per leg (half the bull-only size per leg)
  - Kitchen sink (H): $200/leg for pairs

Honest Pricing Rules:
  - HOLD TO EXPIRY ONLY (no early exit)
  - At expiry: INTRINSIC VALUE ONLY (no time value)
  - 15% haircut on ENTRY only (automatic exercise at expiry = no exit haircut)
  - DTE=21, $645 starting capital
  - Walk-forward LGBM, regime filter via GRU
  - Commission: $2.60/spread

Full 5-gate adversarial validation + random baseline comparison (5 trials) per variant.
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
OUTPUT_DIR = BASE / "output" / "growth_research" / "v6_candidate_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0  # 3% width for all spreads
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "production_v6_candidate_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")


# ══════════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════

VARIANTS = {
    "A_baseline": {
        "desc": "Biweekly, ATM, bull-only, $200/trade",
        "rebal_freq": "2W-FRI",
        "otm_pct": 0.0,       # ATM
        "pairs": False,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,  # unused
    },
    "B_weekly": {
        "desc": "Weekly, ATM, bull-only, $200/trade",
        "rebal_freq": "W-FRI",
        "otm_pct": 0.0,
        "pairs": False,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
    },
    "C_weekly_otm": {
        "desc": "Weekly, 2% OTM, bull-only, $200/trade",
        "rebal_freq": "W-FRI",
        "otm_pct": 0.02,
        "pairs": False,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
    },
    "D_weekly_pairs": {
        "desc": "Weekly, ATM, bull VIX>20 + pairs VIX<20, $200/trade ($100/leg)",
        "rebal_freq": "W-FRI",
        "otm_pct": 0.0,
        "pairs": True,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
    },
    "E_weekly_otm_pairs": {
        "desc": "Weekly, 2% OTM, bull VIX>20 + pairs VIX<20, $200/trade ($100/leg)",
        "rebal_freq": "W-FRI",
        "otm_pct": 0.02,
        "pairs": True,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
    },
    "F_3day_otm_pairs": {
        "desc": "3-day, 2% OTM, bull VIX>20 + pairs VIX<20, $200/trade ($100/leg)",
        "rebal_freq": "3B",     # 3 business days
        "otm_pct": 0.02,
        "pairs": True,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
    },
    "G_3day_otm": {
        "desc": "3-day, 2% OTM, bull-only, $200/trade",
        "rebal_freq": "3B",
        "otm_pct": 0.02,
        "pairs": False,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 100,
    },
    "H_kitchen_sink": {
        "desc": "3-day, 2% OTM, bull VIX>20 + pairs VIX<20, $200/leg for pairs",
        "rebal_freq": "3B",
        "otm_pct": 0.02,
        "pairs": True,
        "max_pos_bull": 200,
        "max_pos_pair_leg": 200,  # full size per leg
    },
}


# ══════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════

def download_data():
    """Download all required tickers via yfinance."""
    import yfinance as yf

    all_tickers = SECTORS + EXTRA_TICKERS
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
           f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}, "
           f"Days <0.2: {(regime_series < REGIME_BEAR_THRESHOLD).sum()}")
    return regime_series


def get_regime_score_at(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback."""
    if regime_series is None:
        return 0.5  # neutral default
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

# Legacy 18 features (quality-momentum) — same as production v4
LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

# 3 validated cross-asset features
VALIDATED_CROSS_ASSET = [
    "sector_spy_beta_63d",
    "sector_relative_vol_21d",
    "cross_sector_dispersion",
]

V4_FEATURES = LEGACY_FEATURES + VALIDATED_CROSS_ASSET  # 21 total


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
    """Compute the 3 validated cross-asset features only."""
    f = {}

    spy = close_df["SPY"].iloc[:dt_idx + 1].dropna()
    sector_px = close_df[sector_ticker].iloc[:dt_idx + 1].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in VALIDATED_CROSS_ASSET}

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
    sector_cols = [c for c in SECTORS if c in close_df.columns]
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

def build_feature_records(close, high, low, rebal_dates, feature_cols, regime_series):
    """
    Build feature + target records for all sectors on all rebal dates.
    Uses bull_only regime mode (regime>0.4) — same as production v4.
    Bear direction is handled at trade time via VIX-based pair logic, NOT regime.
    """
    fprint(f"  Building records: {len(rebal_dates)} dates, {len(feature_cols)} features")

    records = []
    sector_cols = [c for c in SECTORS if c in close.columns]
    spy = close["SPY"]

    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260:
            continue

        # Regime filter: only trade when GRU says bull (>0.4)
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        for tk in sector_cols:
            px = close[tk].iloc[:idx + 1].dropna()
            spy_s = spy.iloc[:idx + 1]

            legacy = compute_legacy_features(px, spy_s)
            if not legacy:
                continue

            # Cross-asset features if needed
            cross_asset = {}
            for col in feature_cols:
                if col in VALIDATED_CROSS_ASSET:
                    cross_asset = compute_cross_asset_features(tk, idx, close)
                    break

            # Forward return target (DTE days forward)
            fi = min(idx + DTE, len(close) - 1)
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
    """Walk-forward LGBM ranking: sliding train, predict next period."""
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
    for tk in SECTORS:
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
# STRIKE COMPUTATION (ATM vs OTM)
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """
    Compute strike prices for a spread.

    ATM (otm_pct=0):
      Bull call: K1=S, K2=S*(1+spread_pct/100)
      Bear put:  K1=S*(1-spread_pct/100), K2=S

    OTM (otm_pct>0, e.g. 0.02 for 2%):
      Bull call: K1=S*(1+otm_pct), K2=K1*(1+spread_pct/100)
      Bear put:  K2=S*(1-otm_pct), K1=K2*(1-spread_pct/100)

    Returns (K1, K2) where K1 < K2 always.
    """
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

def simulate_trades(name, rankings, close, high, low, atr_dict, variant_cfg):
    """
    Simulate trades for a given variant configuration.

    HONEST RULES:
      - Hold to expiry
      - At expiry: intrinsic value only
      - 15% haircut on entry only
      - No exit haircut (automatic exercise)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    otm_pct = variant_cfg["otm_pct"]
    use_pairs = variant_cfg["pairs"]
    max_pos_bull = variant_cfg["max_pos_bull"]
    max_pos_pair_leg = variant_cfg["max_pos_pair_leg"]

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
        # Pairs mode: VIX < 20 → bull + bear; VIX >= 20 → bull only
        # Non-pairs mode: always bull only
        if use_pairs and cv < 20.0:
            trade_mode = "pairs"
        else:
            trade_mode = "bull_only"

        # Pick sectors: top K for bull, bottom K for bear
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        # Position sizing
        if trade_mode == "pairs":
            max_pos = min(max_pos_pair_leg, equity / 6)  # 6 positions total (3 bull + 3 bear)
        else:
            max_pos = min(max_pos_bull, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            pnl = _execute_single_trade(
                tk, dt, "bull", otm_pct, max_pos, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
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
                })

        # Execute bear leg (pairs mode only)
        for tk in bear_picks:
            pnl = _execute_single_trade(
                tk, dt, "bear", otm_pct, max_pos, close, atr_dict, cv, equity
            )
            if pnl is not None:
                equity += pnl
                di = close.index.get_loc(dt)
                ei = min(di + DTE, len(close) - 1)
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
                })

    return trades, equity


def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity):
    """
    Execute a single spread trade. Returns PnL or None if trade could not be entered.

    Uses production v4 pricing: BS with ATR-based IV, 15% entry haircut,
    hold to expiry, intrinsic value only.
    """
    if tk not in close.columns or tk not in atr_dict:
        return None

    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
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
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
            )
        else:
            entry_cost_ps, max_profit_ps = price_bear_put_spread(
                S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=cv
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
# REBALANCE DATE GENERATION
# ══════════════════════════════════════════════════════════════

def generate_rebal_dates(close, freq_str):
    """
    Generate rebalance dates from close index based on frequency string.

    Supports: '2W-FRI', 'W-FRI', '3B' (every 3 business days)
    """
    if freq_str == "3B":
        # Every 3 business days
        bdays = close.index[close.index.dayofweek < 5]  # weekdays only
        rebal_dates = pd.DatetimeIndex([bdays[i] for i in range(0, len(bdays), 3)])
    else:
        rebal_dates = pd.DatetimeIndex(
            close.index.to_series().resample(freq_str).last().dropna().values
        )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint(f"PRODUCTION V6 CANDIDATE V1 — Combined Improvements Test — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 90)
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"Haircut: {DEFAULT_HAIRCUT:.0%} entry only | Comm: ${COMMISSION_RT_SPREAD:.2f}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint(f"Testing {len(VARIANTS)} variants")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # 4. Build LGBM rankings for each unique rebalance frequency
    # Multiple variants share the same rankings if they have the same rebal freq
    unique_freqs = set(v["rebal_freq"] for v in VARIANTS.values())
    fprint(f"\nUnique rebalance frequencies: {sorted(unique_freqs)}")

    freq_rankings = {}
    freq_imp = {}
    for freq in sorted(unique_freqs):
        fprint(f"\n{'=' * 80}")
        fprint(f"BUILDING LGBM RANKINGS: rebal_freq={freq}")
        fprint(f"{'=' * 80}")

        rebal_dates = generate_rebal_dates(close, freq)
        fprint(f"  Rebalance dates: {len(rebal_dates)} "
               f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

        records = build_feature_records(
            close, high, low, rebal_dates, V4_FEATURES, regime_series
        )
        rankings, imp = walk_forward_lgbm_rank(records, V4_FEATURES, f"LGBM_{freq}")
        freq_rankings[freq] = rankings
        freq_imp[freq] = imp

    # 5. Simulate all variants
    fprint(f"\n{'=' * 90}")
    fprint("SIMULATING ALL VARIANTS")
    fprint(f"{'=' * 90}")

    all_results = {}
    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'─' * 80}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'─' * 80}")

        rankings = freq_rankings.get(vcfg["rebal_freq"], {})
        if not rankings:
            fprint(f"  No rankings available for freq={vcfg['rebal_freq']}, skipping")
            continue

        trades, final_eq = simulate_trades(
            vname, rankings, close, high, low, atr_dict, vcfg
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=vname,
        )
        result.print_summary()

        # Direction breakdown for pair variants
        if vcfg["pairs"]:
            bull_trades = [t for t in trades if t["direction"] == "bull"]
            bear_trades = [t for t in trades if t["direction"] == "bear"]
            pair_trades = [t for t in trades if t.get("trade_mode") == "pairs"]
            bull_pnl = sum(t["pnl"] for t in bull_trades)
            bear_pnl = sum(t["pnl"] for t in bear_trades)
            bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
            bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
            fprint(f"  Direction breakdown:")
            fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
            fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")
            fprint(f"    Pair-mode trades: {len(pair_trades)} (VIX<20 dates)")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict, vcfg,
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[vname] = {
            "description": vcfg["desc"],
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "otm_pct": vcfg["otm_pct"],
            "pairs": vcfg["pairs"],
            "rebal_freq": vcfg["rebal_freq"],
        }

    # ── SUMMARY COMPARISON ──
    fprint(f"\n{'=' * 110}")
    fprint("SUMMARY COMPARISON — ALL 8 VARIANTS")
    fprint(f"{'=' * 110}")
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} "
           f"{'PF':>6} {'MaxDD':>7} {'Gates':>6} {'Final$':>8} {'RandSh':>7} {'Rebal':>8}")
    fprint("-" * 110)

    for vname in VARIANTS.keys():
        r = all_results.get(vname)
        if not r:
            fprint(f"  {vname:<25} — NO DATA —")
            continue
        fprint(f"  {vname:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>7,.0f} {r['random_mean_sharpe']:>7.2f} "
               f"{r['rebal_freq']:>8}")

    # ── BEST VARIANT ANALYSIS ──
    fprint(f"\n{'=' * 90}")
    fprint("BEST VARIANT ANALYSIS")
    fprint(f"{'=' * 90}")

    valid_results = {k: v for k, v in all_results.items() if v.get("sharpe", 0) > 0}
    if valid_results:
        # Best by Sharpe
        best_sharpe_name = max(valid_results.keys(), key=lambda k: valid_results[k]["sharpe"])
        best = valid_results[best_sharpe_name]
        fprint(f"\n  BEST BY SHARPE: {best_sharpe_name}")
        fprint(f"    Sharpe: {best['sharpe']:.2f} | Sortino: {best['sortino']:.2f} | "
               f"WR: {best['win_rate']*100:.1f}% | PF: {best['profit_factor']:.2f}")
        fprint(f"    MaxDD: {best['max_dd']*100:.1f}% | Gates: {best['gates_passed']}/{best['gates_total']} | "
               f"Final: ${best['final_equity']:,.0f}")
        fprint(f"    vs Random: {best['sharpe']:.2f} vs {best['random_mean_sharpe']:.2f} "
               f"({best['sharpe']/max(best['random_mean_sharpe'],0.01):.1f}x alpha)")

        # Best with all gates passed
        gated = {k: v for k, v in valid_results.items() if v.get("gates_passed", 0) == v.get("gates_total", 5)}
        if gated:
            best_gated_name = max(gated.keys(), key=lambda k: gated[k]["sharpe"])
            bg = gated[best_gated_name]
            fprint(f"\n  BEST WITH ALL GATES PASSED: {best_gated_name}")
            fprint(f"    Sharpe: {bg['sharpe']:.2f} | Sortino: {bg['sortino']:.2f} | "
                   f"WR: {bg['win_rate']*100:.1f}% | PF: {bg['profit_factor']:.2f}")
            fprint(f"    MaxDD: {bg['max_dd']*100:.1f}% | Final: ${bg['final_equity']:,.0f}")

        # Baseline comparison
        baseline = all_results.get("A_baseline")
        if baseline:
            baseline_sharpe = baseline["sharpe"]
            fprint(f"\n  BASELINE (A) SHARPE: {baseline_sharpe:.2f}")
            for vname, r in sorted(valid_results.items()):
                if vname == "A_baseline":
                    continue
                delta = r["sharpe"] - baseline_sharpe
                pct = (delta / max(abs(baseline_sharpe), 0.01)) * 100
                arrow = "+" if delta > 0 else ""
                fprint(f"    {vname:<25} {arrow}{delta:.2f} ({arrow}{pct:.0f}%) Sharpe")

    # ── RECOMMENDATION ──
    fprint(f"\n{'=' * 90}")
    fprint("RECOMMENDATION")
    fprint(f"{'=' * 90}")

    if valid_results:
        # Find best that passes all gates
        gated = {k: v for k, v in valid_results.items()
                 if v.get("gates_passed", 0) == v.get("gates_total", 5)}
        if gated:
            rec_name = max(gated.keys(), key=lambda k: gated[k]["sharpe"])
            rec = gated[rec_name]
            baseline_sharpe = all_results.get("A_baseline", {}).get("sharpe", 0)
            delta = rec["sharpe"] - baseline_sharpe
            if delta > 0:
                fprint(f"  UPGRADE to {rec_name}: +{delta:.2f} Sharpe over baseline")
                fprint(f"  Config: {rec.get('description', '')}")
            else:
                fprint(f"  KEEP BASELINE: Best gated variant ({rec_name}, Sharpe {rec['sharpe']:.2f}) "
                       f"does not beat baseline ({baseline_sharpe:.2f})")
        else:
            fprint("  WARNING: No variant passes all 5 gates. Stick with baseline.")
    else:
        fprint("  ERROR: No valid results produced.")

    # Feature importance
    fprint(f"\n{'=' * 80}")
    fprint("FEATURE IMPORTANCE (Top 10) — from 21-feature V4 model")
    fprint(f"{'=' * 80}")
    for freq, imp_df in freq_imp.items():
        if imp_df is not None:
            fprint(f"\n  Frequency: {freq}")
            for _, row in imp_df.head(10).iterrows():
                bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
                fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "v6_candidate_v1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save individual variant trade logs
    for vname in VARIANTS.keys():
        r = all_results.get(vname)
        if r:
            vpath = OUTPUT_DIR / f"{vname}_summary.json"
            with open(vpath, "w") as f:
                json.dump(r, f, indent=2, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"v6_cand_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log all variant metrics
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

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "haircut": DEFAULT_HAIRCUT,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_variants": len(VARIANTS),
                    "variants_tested": ", ".join(VARIANTS.keys()),
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
