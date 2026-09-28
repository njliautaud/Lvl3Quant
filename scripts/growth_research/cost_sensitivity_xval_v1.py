#!/usr/bin/env python3
"""
Cost Sensitivity Cross-Validation v1 — How Much Room on Costs?
================================================================

Tests V6 strategy (weekly rebalance, 2% OTM, bull VIX>20 + pairs VIX<20)
under 8 different cost assumptions to find the BREAK POINT where the strategy
stops being viable (Sharpe < 1.0).

V6 Config (fixed across all scenarios):
  - Weekly rebalance (W-FRI)
  - 2% OTM moneyness
  - Bull call spreads VIX >= 20, bull + pair trades (bear puts) VIX < 20
  - Top 3 sectors long, bottom 3 short (pairs)
  - Hold to expiry, DTE=21, 3% spread width
  - $645 starting capital, $200/trade ($100/leg pairs)
  - Walk-forward LGBM, 21 features, GRU regime filter

8 Cost Scenarios:
  A: Zero commission      — $0.00 commission, 0% haircut.  Upper bound.
  B: Production costs     — $2.60 commission, 15% haircut. Should match V6 (~2.79 Sharpe).
  C: Higher commission    — $5.00 commission, 15% haircut.
  D: Double commission    — $10.00 commission, 15% haircut.
  E: Higher haircut       — $2.60 commission, 25% haircut (worse fills).
  F: Extreme haircut      — $2.60 commission, 33% haircut (very conservative fills).
  G: Higher comm+haircut  — $5.00 commission, 25% haircut.
  H: Worst case           — $10.00 commission, 33% haircut.

Key output: at what cost level does Sharpe drop below 1.0?

Based on production_v4_honest_test.py and production_v6_candidate_v1.py.
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
    bs_call_price,
    bs_put_price,
)
from research.tools.adversarial_validator import validate_trades

# ── Config ──
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "cost_sensitivity_xval_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
DTE = 21
SPREAD_PCT = 3.0  # 3% width for all spreads
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
REGIME_BEAR_THRESHOLD = 0.2

# V6 config: weekly rebalance, 2% OTM, pairs enabled
V6_REBAL_FREQ = "W-FRI"
V6_OTM_PCT = 0.02
V6_PAIRS = True
V6_MAX_POS_BULL = 200
V6_MAX_POS_PAIR_LEG = 100

# Regime predictions path
REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"

# LGBM walk-forward
WF_TRAIN_PERIODS = 12

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "cost_sensitivity_xval_v1"

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
# COST SCENARIOS
# ══════════════════════════════════════════════════════════════

COST_SCENARIOS = {
    "A_zero_cost": {
        "desc": "Zero commission, no haircut (upper bound)",
        "commission_rt": 0.00,
        "haircut": 0.00,
    },
    "B_production": {
        "desc": "Production costs ($2.60 comm, 15% haircut)",
        "commission_rt": 2.60,
        "haircut": 0.15,
    },
    "C_higher_comm": {
        "desc": "Higher commission ($5.00 comm, 15% haircut)",
        "commission_rt": 5.00,
        "haircut": 0.15,
    },
    "D_double_comm": {
        "desc": "Double commission ($10.00 comm, 15% haircut)",
        "commission_rt": 10.00,
        "haircut": 0.15,
    },
    "E_higher_haircut": {
        "desc": "Higher haircut ($2.60 comm, 25% haircut)",
        "commission_rt": 2.60,
        "haircut": 0.25,
    },
    "F_extreme_haircut": {
        "desc": "Extreme haircut ($2.60 comm, 33% haircut)",
        "commission_rt": 2.60,
        "haircut": 0.33,
    },
    "G_higher_both": {
        "desc": "Higher comm + haircut ($5.00 comm, 25% haircut)",
        "commission_rt": 5.00,
        "haircut": 0.25,
    },
    "H_worst_case": {
        "desc": "Worst case ($10.00 comm, 33% haircut)",
        "commission_rt": 10.00,
        "haircut": 0.33,
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
# FEATURE ENGINEERING (21 features: 18 legacy + 3 cross-asset)
# ══════════════════════════════════════════════════════════════

LEGACY_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_21d", "vol_63d", "sharpe_63d", "maxdd_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
]

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
    Uses bull_only regime mode (regime>0.4) — same as V6.
    Bear direction is handled at trade time via VIX-based pair logic.
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
# STRIKE COMPUTATION (2% OTM per V6)
# ══════════════════════════════════════════════════════════════

def compute_strikes(S, direction, otm_pct, spread_pct):
    """
    Compute strike prices for a spread.

    OTM (otm_pct>0, e.g. 0.02 for 2%):
      Bull call: K1=S*(1+otm_pct), K2=K1*(1+spread_pct/100)
      Bear put:  K2=S*(1-otm_pct), K1=K2*(1-spread_pct/100)

    ATM (otm_pct=0):
      Bull call: K1=S, K2=S*(1+spread_pct/100)
      Bear put:  K1=S*(1-spread_pct/100), K2=S

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
# COST-PARAMETERIZED SPREAD PRICING
# ══════════════════════════════════════════════════════════════

def price_spread_with_costs(S, K1, K2, direction, dte, atr, vix, haircut):
    """
    Price a spread with a CUSTOM haircut (overriding the module default).
    Returns (entry_cost_ps, max_profit_ps) per share.
    """
    T = dte / 365.0
    sigma = estimate_iv(atr, S, vix)

    if direction == "bull":
        fair_value = bs_call_price(S, K1, T, sigma=sigma) - bs_call_price(S, K2, T, sigma=sigma)
    else:
        fair_value = bs_put_price(S, K2, T, sigma=sigma) - bs_put_price(S, K1, T, sigma=sigma)

    fair_value = max(fair_value, 0.001)

    # Entry: pay MORE than fair (haircut UP)
    entry_cost = fair_value * (1.0 + haircut)

    # Max profit at expiry = spread width - entry cost
    spread_width = K2 - K1
    max_profit = spread_width - entry_cost

    return float(entry_cost), float(max_profit)


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V6 config, parameterized costs)
# ══════════════════════════════════════════════════════════════

def simulate_trades_with_costs(name, rankings, close, high, low, atr_dict,
                                commission_rt, haircut):
    """
    Simulate V6 strategy with custom cost parameters.

    V6 config: weekly rebalance, 2% OTM, bull VIX>=20 + pairs VIX<20.
    Cost parameters (commission_rt, haircut) vary per scenario.

    HONEST RULES:
      - Hold to expiry
      - At expiry: intrinsic value only
      - Custom haircut on entry only
      - No exit haircut (automatic exercise)
    """
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = rankings[dt]
        if not scores:
            continue

        # V6 pair logic: VIX < 20 = bull + bear; VIX >= 20 = bull only
        if V6_PAIRS and cv < 20.0:
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
            max_pos = min(V6_MAX_POS_PAIR_LEG, equity / 6)  # 6 positions (3 bull + 3 bear)
        else:
            max_pos = min(V6_MAX_POS_BULL, equity / 3)

        if max_pos < 30:
            continue

        # Execute bull leg
        for tk in bull_picks:
            pnl = _execute_single_trade(
                tk, dt, "bull", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity,
                commission_rt=commission_rt, haircut=haircut,
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
                tk, dt, "bear", V6_OTM_PCT, max_pos, close, atr_dict, cv, equity,
                commission_rt=commission_rt, haircut=haircut,
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


def _execute_single_trade(tk, dt, direction, otm_pct, max_pos, close, atr_dict, cv, equity,
                           commission_rt, haircut):
    """
    Execute a single spread trade with custom cost parameters.
    Returns PnL or None if trade could not be entered.
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

    # Compute strikes (V6: 2% OTM)
    K1, K2 = compute_strikes(S, direction, otm_pct, SPREAD_PCT)

    try:
        entry_cost_ps, max_profit_ps = price_spread_with_costs(
            S=S, K1=K1, K2=K2, direction=direction,
            dte=DTE, atr=av, vix=cv, haircut=haircut,
        )
    except Exception:
        return None

    total_cost = entry_cost_ps * 100 + commission_rt

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
    pnl = (exit_value_ps - entry_cost_ps) * 100 - commission_rt
    return pnl


# ══════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════

def random_baseline_test(rankings, close, high, low, atr_dict,
                          commission_rt, haircut, n_trials=5):
    """Test if random sector selection also produces similar returns."""
    fprint(f"\n  Random baseline test ({n_trials} trials)...")
    random_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_scores = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = rand_scores

        trades, final_eq = simulate_trades_with_costs(
            f"Random_{trial}", rand_rankings, close, high, low, atr_dict,
            commission_rt=commission_rt, haircut=haircut,
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
    """Generate rebalance dates from close index based on frequency string."""
    rebal_dates = pd.DatetimeIndex(
        close.index.to_series().resample(freq_str).last().dropna().values
    )
    return rebal_dates


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"COST SENSITIVITY CROSS-VALIDATION V1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Strategy: V6 (weekly rebalance, 2% OTM, bull VIX>=20 + pairs VIX<20)")
    fprint(f"Capital: ${CAP:.0f} | DTE: {DTE} | Spread: {SPREAD_PCT:.0f}% | "
           f"OTM: {V6_OTM_PCT*100:.0f}% | Pairs: {V6_PAIRS}")
    fprint(f"HOLD TO EXPIRY | Intrinsic value only at expiry | No exit haircut")
    fprint(f"Regime bull threshold: >{REGIME_BULL_THRESHOLD}")
    fprint(f"Testing {len(COST_SCENARIOS)} cost scenarios")
    fprint()
    fprint("Cost scenarios:")
    for sname, scfg in COST_SCENARIOS.items():
        fprint(f"  {sname}: comm=${scfg['commission_rt']:.2f}, "
               f"haircut={scfg['haircut']*100:.0f}% — {scfg['desc']}")
    fprint()

    # 1. Download data
    close, high, low = download_data()

    # 2. Load regime predictions
    regime_series = load_regime_predictions()

    # 3. Pre-compute ATR
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]

    # 4. Build LGBM rankings ONCE (V6 config: weekly rebalance)
    # Rankings are cost-independent — only the trade simulation changes per scenario.
    fprint(f"\n{'=' * 80}")
    fprint(f"BUILDING LGBM RANKINGS: V6 config, rebal_freq={V6_REBAL_FREQ}")
    fprint(f"{'=' * 80}")

    rebal_dates = generate_rebal_dates(close, V6_REBAL_FREQ)
    fprint(f"  Rebalance dates: {len(rebal_dates)} "
           f"({rebal_dates[0].date()} to {rebal_dates[-1].date()})")

    records = build_feature_records(
        close, high, low, rebal_dates, V4_FEATURES, regime_series
    )
    rankings, imp_df = walk_forward_lgbm_rank(records, V4_FEATURES, "V6_LGBM")

    if not rankings:
        fprint("FATAL: No rankings produced. Cannot proceed.")
        return

    # 5. Simulate all cost scenarios using the SAME rankings
    fprint(f"\n{'=' * 100}")
    fprint("SIMULATING ALL COST SCENARIOS")
    fprint(f"{'=' * 100}")

    all_results = {}
    for sname, scfg in COST_SCENARIOS.items():
        fprint(f"\n{'─' * 90}")
        fprint(f"SCENARIO {sname}: {scfg['desc']}")
        fprint(f"  Commission: ${scfg['commission_rt']:.2f}/spread | "
               f"Entry haircut: {scfg['haircut']*100:.0f}%")
        fprint(f"{'─' * 90}")

        trades, final_eq = simulate_trades_with_costs(
            sname, rankings, close, high, low, atr_dict,
            commission_rt=scfg["commission_rt"],
            haircut=scfg["haircut"],
        )

        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping validation")
            all_results[sname] = {
                "description": scfg["desc"],
                "commission_rt": scfg["commission_rt"],
                "haircut": scfg["haircut"],
                "n_trades": len(trades) if trades else 0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_dd": 0.0,
                "gates_passed": 0,
                "gates_total": 5,
                "final_equity": final_eq,
            }
            continue

        # 5-gate adversarial validation
        result = validate_trades(
            trades, initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=sname,
        )
        result.print_summary()

        # Direction breakdown
        bull_trades = [t for t in trades if t["direction"] == "bull"]
        bear_trades = [t for t in trades if t["direction"] == "bear"]
        bull_pnl = sum(t["pnl"] for t in bull_trades)
        bear_pnl = sum(t["pnl"] for t in bear_trades)
        bull_wr = sum(1 for t in bull_trades if t["win"]) / max(len(bull_trades), 1) * 100
        bear_wr = sum(1 for t in bear_trades if t["win"]) / max(len(bear_trades), 1) * 100
        fprint(f"  Direction breakdown:")
        fprint(f"    Bull: {len(bull_trades)} trades, WR {bull_wr:.1f}%, PnL ${bull_pnl:.0f}")
        fprint(f"    Bear: {len(bear_trades)} trades, WR {bear_wr:.1f}%, PnL ${bear_pnl:.0f}")

        # Random baseline
        random_sharpes = random_baseline_test(
            rankings, close, high, low, atr_dict,
            commission_rt=scfg["commission_rt"],
            haircut=scfg["haircut"],
        )
        mean_random = np.mean(random_sharpes) if random_sharpes else 0
        fprint(f"  ML Sharpe: {result.sharpe:.2f} vs Random mean: {mean_random:.2f}")
        if result.sharpe > 0 and mean_random > 0:
            fprint(f"  ML alpha ratio: {result.sharpe / mean_random:.2f}x")

        all_results[sname] = {
            "description": scfg["desc"],
            "commission_rt": scfg["commission_rt"],
            "haircut": scfg["haircut"],
            **result.to_dict(),
            "random_sharpes": [round(s, 3) for s in random_sharpes],
            "random_mean_sharpe": round(mean_random, 3),
            "bull_trades": len(bull_trades),
            "bear_trades": len(bear_trades),
            "bull_pnl": round(bull_pnl, 2),
            "bear_pnl": round(bear_pnl, 2),
            "bull_wr": round(bull_wr, 1),
            "bear_wr": round(bear_wr, 1),
        }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 120}")
    fprint("COST SENSITIVITY SUMMARY — V6 Strategy Under Different Cost Assumptions")
    fprint(f"{'=' * 120}")
    fprint(f"{'Scenario':<22} {'Comm$':>6} {'Hair%':>6} {'Trades':>7} {'Sharpe':>7} "
           f"{'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>6} "
           f"{'Final$':>9} {'RandSh':>7}")
    fprint("-" * 120)

    for sname in COST_SCENARIOS.keys():
        r = all_results.get(sname, {})
        if not r:
            fprint(f"  {sname:<22} — NO DATA —")
            continue
        sharpe = r.get("sharpe", 0)
        sortino = r.get("sortino", 0)
        wr = r.get("win_rate", 0)
        pf = r.get("profit_factor", 0)
        mdd = r.get("max_dd", 0)
        gp = r.get("gates_passed", 0)
        gt = r.get("gates_total", 5)
        feq = r.get("final_equity", CAP)
        rsh = r.get("random_mean_sharpe", 0)
        # Mark scenarios where strategy breaks
        flag = " *** BREAK ***" if sharpe < 1.0 else ""
        fprint(f"  {sname:<22} ${r['commission_rt']:>5.2f} {r['haircut']*100:>5.0f}% "
               f"{r.get('n_trades', 0):>6} {sharpe:>7.2f} {sortino:>8.2f} "
               f"{wr*100:>5.1f}% {pf:>5.2f} {mdd*100:>6.1f}% {gp}/{gt} "
               f"${feq:>8,.0f} {rsh:>7.2f}{flag}")

    # ══════════════════════════════════════════════════════════════
    # BREAK POINT ANALYSIS
    # ══════════════════════════════════════════════════════════════
    fprint(f"\n{'=' * 100}")
    fprint("BREAK POINT ANALYSIS — Where Does Sharpe Drop Below 1.0?")
    fprint(f"{'=' * 100}")

    production_sharpe = all_results.get("B_production", {}).get("sharpe", 0)
    fprint(f"\n  Production (B) Sharpe: {production_sharpe:.2f}")

    # Commission sensitivity (holding haircut at 15%)
    fprint(f"\n  COMMISSION SENSITIVITY (haircut fixed at 15%):")
    comm_scenarios = ["A_zero_cost", "B_production", "C_higher_comm", "D_double_comm"]
    for sname in comm_scenarios:
        r = all_results.get(sname, {})
        sharpe = r.get("sharpe", 0)
        delta = sharpe - production_sharpe
        fprint(f"    {sname:<22} comm=${r.get('commission_rt', 0):>5.2f}: "
               f"Sharpe {sharpe:.2f} ({delta:+.2f} vs production)")

    # Haircut sensitivity (holding commission at $2.60)
    fprint(f"\n  HAIRCUT SENSITIVITY (commission fixed at $2.60):")
    hair_scenarios = ["A_zero_cost", "B_production", "E_higher_haircut", "F_extreme_haircut"]
    for sname in hair_scenarios:
        r = all_results.get(sname, {})
        sharpe = r.get("sharpe", 0)
        delta = sharpe - production_sharpe
        fprint(f"    {sname:<22} haircut={r.get('haircut', 0)*100:>3.0f}%: "
               f"Sharpe {sharpe:.2f} ({delta:+.2f} vs production)")

    # Combined sensitivity
    fprint(f"\n  COMBINED SENSITIVITY:")
    combined = ["B_production", "G_higher_both", "H_worst_case"]
    for sname in combined:
        r = all_results.get(sname, {})
        sharpe = r.get("sharpe", 0)
        delta = sharpe - production_sharpe
        fprint(f"    {sname:<22} comm=${r.get('commission_rt', 0):>5.2f} + "
               f"haircut={r.get('haircut', 0)*100:>3.0f}%: "
               f"Sharpe {sharpe:.2f} ({delta:+.2f} vs production)")

    # Find break point
    fprint(f"\n  VERDICT:")
    broken = [sname for sname in COST_SCENARIOS if all_results.get(sname, {}).get("sharpe", 0) < 1.0]
    viable = [sname for sname in COST_SCENARIOS if all_results.get(sname, {}).get("sharpe", 0) >= 1.0]
    all_gated = [sname for sname in COST_SCENARIOS
                 if all_results.get(sname, {}).get("gates_passed", 0) == all_results.get(sname, {}).get("gates_total", 5)]

    if broken:
        first_break = broken[0]
        r = all_results[first_break]
        fprint(f"    Strategy BREAKS (Sharpe < 1.0) at scenario: {first_break}")
        fprint(f"    Break point: comm=${r['commission_rt']:.2f}, haircut={r['haircut']*100:.0f}%")
        fprint(f"    Sharpe at break: {r.get('sharpe', 0):.2f}")
    else:
        fprint(f"    Strategy SURVIVES all cost scenarios (Sharpe >= 1.0 everywhere)")

    fprint(f"\n    Viable scenarios (Sharpe >= 1.0): {len(viable)}/{len(COST_SCENARIOS)}")
    for sname in viable:
        r = all_results[sname]
        fprint(f"      {sname}: Sharpe {r.get('sharpe', 0):.2f}")

    fprint(f"\n    Scenarios passing all 5 gates: {len(all_gated)}/{len(COST_SCENARIOS)}")
    for sname in all_gated:
        r = all_results[sname]
        fprint(f"      {sname}: Sharpe {r.get('sharpe', 0):.2f}, gates {r['gates_passed']}/{r['gates_total']}")

    # Cost headroom calculation
    if production_sharpe > 1.0:
        headroom = production_sharpe - 1.0
        fprint(f"\n    Cost headroom: {headroom:.2f} Sharpe units above break threshold")
        fprint(f"    Production Sharpe of {production_sharpe:.2f} can absorb "
               f"{headroom/production_sharpe*100:.0f}% cost increase before breaking")

    # Feature importance
    fprint(f"\n{'=' * 80}")
    fprint("FEATURE IMPORTANCE (Top 10) — V6 LGBM model")
    fprint(f"{'=' * 80}")
    if imp_df is not None:
        for _, row in imp_df.head(10).iterrows():
            bar = "*" * int(row["importance"] / imp_df["importance"].max() * 30)
            fprint(f"    {row['feature']:<30} {row['importance']:>6.1f} {bar}")

    # Save results
    results_path = OUTPUT_DIR / "cost_sensitivity_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save individual scenario summaries
    for sname in COST_SCENARIOS.keys():
        r = all_results.get(sname)
        if r:
            vpath = OUTPUT_DIR / f"{sname}_summary.json"
            with open(vpath, "w") as f:
                json.dump(r, f, indent=2, default=str)

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"cost_sens_v1_{t0.strftime('%Y%m%d_%H%M')}"):
                # Log all scenario metrics
                for sname, r in all_results.items():
                    prefix = sname.split("_")[0]
                    mlflow.log_metric(f"{prefix}_sharpe", r.get("sharpe", 0))
                    mlflow.log_metric(f"{prefix}_sortino", r.get("sortino", 0))
                    mlflow.log_metric(f"{prefix}_win_rate", r.get("win_rate", 0))
                    mlflow.log_metric(f"{prefix}_profit_factor", r.get("profit_factor", 0))
                    mlflow.log_metric(f"{prefix}_max_dd", r.get("max_dd", 0))
                    mlflow.log_metric(f"{prefix}_n_trades", r.get("n_trades", 0))
                    mlflow.log_metric(f"{prefix}_gates_passed", r.get("gates_passed", 0))
                    mlflow.log_metric(f"{prefix}_final_equity", r.get("final_equity", 0))
                    mlflow.log_metric(f"{prefix}_random_mean_sharpe", r.get("random_mean_sharpe", 0))
                    mlflow.log_metric(f"{prefix}_commission", r.get("commission_rt", 0))
                    mlflow.log_metric(f"{prefix}_haircut", r.get("haircut", 0))

                mlflow.log_params({
                    "capital": CAP,
                    "dte": DTE,
                    "spread_pct": SPREAD_PCT,
                    "otm_pct": V6_OTM_PCT,
                    "rebal_freq": V6_REBAL_FREQ,
                    "pairs": V6_PAIRS,
                    "regime_bull_thresh": REGIME_BULL_THRESHOLD,
                    "hold_to_expiry": True,
                    "entry_haircut_only": True,
                    "n_features": len(V4_FEATURES),
                    "wf_train_periods": WF_TRAIN_PERIODS,
                    "n_scenarios": len(COST_SCENARIOS),
                    "scenarios_tested": ", ".join(COST_SCENARIOS.keys()),
                    "break_threshold": "Sharpe < 1.0",
                })

                # Log break point finding
                if broken:
                    mlflow.log_param("first_break_scenario", broken[0])
                    mlflow.log_metric("n_viable_scenarios", len(viable))
                else:
                    mlflow.log_param("first_break_scenario", "NONE")
                    mlflow.log_metric("n_viable_scenarios", len(COST_SCENARIOS))

                mlflow.log_artifact(str(results_path))
            fprint(f"MLflow run logged to experiment '{EXPERIMENT_NAME}'")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint("DONE")


if __name__ == "__main__":
    main()
